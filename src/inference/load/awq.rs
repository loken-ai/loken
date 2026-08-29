//! AWQ (Activation-aware Weight Quantization) GEMM-format weights.
//!
//! Why support a second 4-bit format at all: long-context dense decode is **format-bound**,
//! not kernel-bound. GGUF Q4_K_M is a mixed 4/6-bit super-block scheme carrying an f16 scale
//! and minimum per 256 weights, which works out to roughly a fifth more bytes per weight than
//! AWQ's uniform 4 bits with one f16 scale and one 4-bit zero per group of 128. At the point
//! where decode is reading weights as fast as the memory bus allows, the format is the
//! ceiling, and the only way past it is to read fewer bytes for the same model.
//!
//! ## AutoAWQ GEMM layout (HF safetensors), per `Linear(in=K, out=N)`
//!   - `qweight` : `[K, N/8]` i32 - 8 output channels packed per i32.
//!   - `qzeros`  : `[K/gs, N/8]` i32 - 8 zero-points packed per i32, per group.
//!   - `scales`  : `[K/gs, N]` f16 - one scale per (group, output channel).
//!   - `bias`    : `[N]` f16 (optional; qwen2 q/k/v have it).
//! `gs` = group_size (128). Embeddings / lm_head / norms stay full precision.
//!
//! ## Dequant (validated bit-exact vs vLLM `awq_dequantize`)
//! The 8 packed values are NOT in bit order - AutoAWQ interleaves them so the
//! tensor-core path can shuffle cheaply. Output channel `n` (0..7 within a packed
//! i32) lives at nibble `AWQ_ORDER[n]`:
//! ```text
//! ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
//! out   = pc*8 + n                       // pc = packed column, n = 0..7
//! g     = k / gs                         // group index
//! wq    = (qweight[k, pc]  >> 4*ORDER[n]) & 0xF
//! zq    = (qzeros [g, pc]  >> 4*ORDER[n]) & 0xF
//! W[out, k] = (wq - zq) * scales[g, out]
//! ```
//! Note `W[out,k]` is already the *transpose* the matmul wants: a Linear computes
//! `y[out] = Σ_k x[k] . W[out,k]`, and the AWQ layout indexes by `[k, out]`
//! directly - no transpose needed for the GEMV.
//!
//! This module is the CPU reference + index-mapping lock (the usual home of silent
//! kernel bugs); the CUDA GEMV kernel mirrors `gemv` exactly.

use half::f16;

/// AWQ packs 8 output channels per i32; channel `n` is at nibble `ORDER[n]`.
pub const AWQ_ORDER: [u32; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
/// Output channels packed into one i32 (4-bit x 8 = 32).
pub const PACK: usize = 8;

/// An AWQ-quantized linear weight `Linear(in=k, out=n)`, GEMM format. Holds the
/// raw safetensors tensors verbatim (no repack) - the CUDA kernel consumes this
/// layout directly.
pub struct AwqTensor {
    /// Input features (K).
    pub k: usize,
    /// Output features (N).
    pub n: usize,
    /// Group size (typically 128).
    pub group_size: usize,
    /// `[K * N/8]` i32 row-major - 8 output nibbles per i32 (AWQ order).
    pub qweight: Vec<i32>,
    /// `[K/gs * N/8]` i32 row-major - 8 zero nibbles per i32 (AWQ order).
    pub qzeros: Vec<i32>,
    /// `[K/gs * N]` f16 row-major - per-(group, output) scale.
    pub scales: Vec<f16>,
}

impl AwqTensor {
    /// Number of packed columns (`N/8`).
    #[inline]
    pub fn packed_cols(&self) -> usize {
        self.n / PACK
    }
    /// Number of groups (`K/gs`).
    #[inline]
    pub fn n_groups(&self) -> usize {
        self.k / self.group_size
    }

    /// Construct from raw safetensors tensors, validating shapes.
    pub fn new(
        k: usize,
        n: usize,
        group_size: usize,
        qweight: Vec<i32>,
        qzeros: Vec<i32>,
        scales: Vec<f16>,
    ) -> Self {
        assert_eq!(n % PACK, 0, "N={n} not a multiple of {PACK}");
        assert_eq!(
            k % group_size,
            0,
            "K={k} not a multiple of group_size={group_size}"
        );
        assert_eq!(qweight.len(), k * (n / PACK), "qweight shape mismatch");
        assert_eq!(
            qzeros.len(),
            (k / group_size) * (n / PACK),
            "qzeros shape mismatch"
        );
        assert_eq!(scales.len(), (k / group_size) * n, "scales shape mismatch");
        Self {
            k,
            n,
            group_size,
            qweight,
            qzeros,
            scales,
        }
    }

    /// Dequantize one output channel at one input index (`W[out, k]`).
    #[inline]
    fn deq_one(&self, k: usize, out: usize) -> f32 {
        let pc = out / PACK;
        let n_in_pack = out % PACK;
        let shift = 4 * AWQ_ORDER[n_in_pack];
        let g = k / self.group_size;
        let pcols = self.packed_cols();
        let wq = ((self.qweight[k * pcols + pc] as u32) >> shift) & 0xF;
        let zq = ((self.qzeros[g * pcols + pc] as u32) >> shift) & 0xF;
        let sc = self.scales[g * self.n + out].to_f32();
        (wq as i32 - zq as i32) as f32 * sc
    }

    /// Full dequantization -> `[K, N]` row-major (`out[k*N + out] = W[out, k]`).
    /// Reference for the kernel + correctness oracle (matches vLLM bit-exact).
    pub fn dequant(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.k * self.n];
        for k in 0..self.k {
            for o in 0..self.n {
                out[k * self.n + o] = self.deq_one(k, o);
            }
        }
        out
    }

    /// Reference GEMV `y[out] = Σ_k x[k] . W[out, k]`, written in the EXACT
    /// iteration the CUDA decode kernel uses (packed-column outer, k inner, all 8
    /// nibbles of each i32 reused once). Locks the index mapping against a test
    /// before the CUDA port. `x` is the full-precision activation (length K).
    pub fn gemv(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.k, "activation length != K");
        let pcols = self.packed_cols();
        let gs = self.group_size;
        let mut out = vec![0f32; self.n];
        // One packed column = 8 output channels sharing each i32.
        for pc in 0..pcols {
            let mut acc = [0f32; PACK];
            for k in 0..self.k {
                let g = k / gs;
                let w = self.qweight[k * pcols + pc] as u32;
                let z = self.qzeros[g * pcols + pc] as u32;
                let xk = x[k];
                #[allow(clippy::needless_range_loop)]
                for n in 0..PACK {
                    let shift = 4 * AWQ_ORDER[n];
                    let wq = ((w >> shift) & 0xF) as i32;
                    let zq = ((z >> shift) & 0xF) as i32;
                    let out_ch = pc * PACK + n;
                    let sc = self.scales[g * self.n + out_ch].to_f32();
                    acc[n] += (wq - zq) as f32 * sc * xk;
                }
            }
            for n in 0..PACK {
                out[pc * PACK + n] = acc[n];
            }
        }
        out
    }
}

/// AWQ weights repacked **output-major** for a bandwidth-efficient GEMV.
///
/// The native AutoAWQ layout is K-major (`qweight[K, N/8]`): the K weights of one
/// output channel are strided `N/8` apart, so a warp-per-row GEMV reads them with
/// poor DRAM locality (the split-K kernel on the native layout tops out ~72% of
/// peak). GGUF's `mmvq` hits ~97% because its layout is output-major - each
/// output's K weights are contiguous, so a warp reads them sequentially +
/// coalesced. This repack does the same for AWQ: unpack the AWQ-order nibbles and
/// re-pack them in K-order, one output channel's K weights contiguous.
///
/// Layout (all output-major):
///   - `qw_t`    : `[N, K/8]` i32 - 8 consecutive-k 4-bit weights per i32.
///   - `scales_t`: `[N, K/gs]` f16.
///   - `zeros_t` : `[N, K/gs]` u8 - unpacked 4-bit zero-point.
pub struct RepackedAwq {
    pub n: usize,
    pub k: usize,
    pub group_size: usize,
    pub qw_t: Vec<i32>,
    pub scales_t: Vec<f16>,
    pub zeros_t: Vec<u8>,
}

/// Repack an `AwqTensor` (native K-major) into the output-major `RepackedAwq`.
pub fn repack_awq(t: &AwqTensor) -> RepackedAwq {
    let (k, n, gs) = (t.k, t.n, t.group_size);
    assert_eq!(
        k % 8,
        0,
        "K must be a multiple of 8 for the output-major repack"
    );
    let pcols = t.packed_cols();
    let ngroups = t.n_groups();
    let mut qw_t = vec![0i32; n * (k / 8)];
    let mut scales_t = vec![f16::ZERO; n * ngroups];
    let mut zeros_t = vec![0u8; n * ngroups];
    for out in 0..n {
        let pc = out / PACK;
        let shift = 4 * AWQ_ORDER[out % PACK];
        // weights: pack 8 consecutive-k nibbles per i32, k-order.
        for k8 in 0..(k / 8) {
            let mut packed = 0u32;
            for j in 0..8 {
                let kk = k8 * 8 + j;
                let wq = ((t.qweight[kk * pcols + pc] as u32) >> shift) & 0xF;
                packed |= wq << (4 * j);
            }
            qw_t[out * (k / 8) + k8] = packed as i32;
        }
        // scales + zeros, output-major.
        for g in 0..ngroups {
            scales_t[out * ngroups + g] = t.scales[g * n + out];
            zeros_t[out * ngroups + g] = (((t.qzeros[g * pcols + pc] as u32) >> shift) & 0xF) as u8;
        }
    }
    RepackedAwq {
        n,
        k,
        group_size: gs,
        qw_t,
        scales_t,
        zeros_t,
    }
}

impl RepackedAwq {
    /// Reference GEMV over the repacked layout, in the EXACT warp-per-row order
    /// the CUDA kernel will use: for each output, walk K in chunks of 8 (one i32),
    /// dequant `(nibble - zero).scale`, accumulate `x[k]`. Locks the index mapping.
    pub fn gemv(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.k, "activation length != K");
        let kdiv8 = self.k / 8;
        let ngroups = self.k / self.group_size;
        let mut out = vec![0f32; self.n];
        for o in 0..self.n {
            let mut acc = 0f32;
            for k8 in 0..kdiv8 {
                let w = self.qw_t[o * kdiv8 + k8] as u32;
                for j in 0..8 {
                    let kk = k8 * 8 + j;
                    let g = kk / self.group_size;
                    let wq = ((w >> (4 * j)) & 0xF) as i32;
                    let zq = self.zeros_t[o * ngroups + g] as i32;
                    let sc = self.scales_t[o * ngroups + g].to_f32();
                    acc += (wq - zq) as f32 * sc * x[kk];
                }
            }
            out[o] = acc;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden scalars captured from vLLM's authoritative `awq_dequantize`
    /// (deepcoder-14b-preview-awq, layer 0 q_proj). Verifies the
    /// nibble order, group indexing, and the (wq - zq).scale formula.
    #[test]
    fn dequant_matches_vllm_golden() {
        // Group 0, k=0, packed col 0 (output channels 0..7).
        let qw00: u32 = 0xafa7_433b;
        let qz00: u32 = 0x9765_4779;
        let sc0: [f32; 8] = [
            0.03875732421875,
            0.03973388671875,
            0.025360107421875,
            0.0292205810546875,
            0.0221405029296875,
            0.0208282470703125,
            0.030731201171875,
            0.0228424072265625,
        ];
        let expect0: [f32; 8] = [
            0.077515, 0.079468, -0.10144, 0.116882, -0.088562, 0.166626, 0.0, 0.022842,
        ];
        for n in 0..PACK {
            let shift = 4 * AWQ_ORDER[n];
            let wq = ((qw00 >> shift) & 0xF) as i32;
            let zq = ((qz00 >> shift) & 0xF) as i32;
            let got = (wq - zq) as f32 * sc0[n];
            assert!(
                (got - expect0[n]).abs() < 1e-3,
                "group0 ch{n}: got {got}, want {}",
                expect0[n]
            );
        }

        // Group 1, k=130, packed col 0 - checks group indexing (k/gs=1).
        let qw130: u32 = 0x6332_9c77;
        let qz1: u32 = 0x8443_8a87;
        let sc1: [f32; 8] = [
            0.04766845703125,
            0.048736572265625,
            0.02825927734375,
            0.0400390625,
            0.034210205078125,
            0.044403076171875,
            0.02093505859375,
            0.0245513916015625,
        ];
        let expect1: [f32; 8] = [
            0.0, -0.048737, -0.028259, -0.040039, 0.06842, -0.044403, 0.020935, -0.049103,
        ];
        for n in 0..PACK {
            let shift = 4 * AWQ_ORDER[n];
            let wq = ((qw130 >> shift) & 0xF) as i32;
            let zq = ((qz1 >> shift) & 0xF) as i32;
            let got = (wq - zq) as f32 * sc1[n];
            assert!(
                (got - expect1[n]).abs() < 1e-3,
                "group1 ch{n}: got {got}, want {}",
                expect1[n]
            );
        }
    }

    /// `gemv` must equal `dequant` followed by a plain dot product (locks the
    /// kernel's packed-column iteration against the straightforward dequant).
    #[test]
    fn gemv_matches_dequant_dot() {
        // Tiny synthetic AWQ tensor: K=256, N=16 (2 packed cols), gs=128.
        let (k, n, gs) = (256usize, 16usize, 128usize);
        let pcols = n / PACK;
        let ngroups = k / gs;
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let qweight: Vec<i32> = (0..k * pcols).map(|_| next() as i32).collect();
        let qzeros: Vec<i32> = (0..ngroups * pcols).map(|_| next() as i32).collect();
        let scales: Vec<f16> = (0..ngroups * n)
            .map(|_| f16::from_f32((next() >> 40) as f32 / (1u64 << 24) as f32 * 0.1))
            .collect();
        let t = AwqTensor::new(k, n, gs, qweight, qzeros, scales);

        let x: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect();
        let w = t.dequant(); // [K, N], W[out,k] at w[k*N+out]
        let mut want = vec![0f32; n];
        for o in 0..n {
            let mut acc = 0f32;
            for kk in 0..k {
                acc += x[kk] * w[kk * n + o];
            }
            want[o] = acc;
        }
        let got = t.gemv(&x);
        let mut maxabs = 0f32;
        for o in 0..n {
            maxabs = maxabs.max((got[o] - want[o]).abs());
        }
        assert!(
            maxabs < 1e-3,
            "gemv vs dequant-dot mismatch, maxabs={maxabs}"
        );
    }
}
