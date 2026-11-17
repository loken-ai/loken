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

/// One AWQ projection as its checkpoint stores it, read in place.
pub struct StoredAwq<'a> {
    /// `[k, n / 8]` little-endian i32, 8 output nibbles per word in AWQ order.
    pub qweight: &'a [u8],
    /// `[k / group, n / 8]` little-endian i32, 8 zero nibbles per word in AWQ order.
    pub qzeros: &'a [u8],
    /// `[k / group, n]` scales, each held f16.
    pub scales: Vec<f16>,
    /// Output features.
    pub n: usize,
}

/// The bias of a `Q5_0` code: code `c` holds the value `c - 16`.
const Q5_0_BIAS: u32 = 16;

/// The host block formats an AWQ projection is written in.
///
/// `Q5_0` holds every stored value exactly: the group's scale and the code less the zero, which
/// lies in -15..15, in 5.5 bits a value.
/// `Q4_1` holds the codes as stored, the group's scale and `-scale * zero` rounded to f16 as the
/// minimum: a value differs from the stored one by the rounding of the minimum alone.
pub fn host_block_format(exact: bool) -> crate::tensor::quantized::GgmlDType {
    use crate::tensor::quantized::GgmlDType;
    if exact {
        GgmlDType::Q5_0
    } else {
        GgmlDType::Q4_1
    }
}

/// Write the projections `parts`, all of `k` inputs, into `out` as rows of `format` blocks, one
/// of [`host_block_format`]'s: the rows of each part in order, one row per output.
pub fn write_block_rows(
    parts: &[StoredAwq],
    k: usize,
    group_size: usize,
    format: crate::tensor::quantized::GgmlDType,
    out: &mut [u8],
) -> Result<(), String> {
    use crate::tensor::quantized::GgmlDType;
    use rayon::prelude::*;
    let exact = match format {
        GgmlDType::Q5_0 => true,
        GgmlDType::Q4_1 => false,
        other => return Err(format!("no AWQ host blocks in {other:?}")),
    };
    let block = format.block_size();
    let block_bytes = format.type_size();
    if group_size == 0 || !group_size.is_multiple_of(block) || !k.is_multiple_of(group_size) {
        return Err(format!(
            "AWQ groups of {group_size} over {k} inputs do not hold whole blocks of {block}"
        ));
    }
    let row_bytes = k / block * block_bytes;
    let rows: usize = parts.iter().map(|p| p.n).sum();
    if out.len() != rows * row_bytes {
        return Err(format!(
            "{} bytes for {rows} rows of {row_bytes}",
            out.len()
        ));
    }
    let groups = k / group_size;
    let mut rest = out;
    for part in parts {
        let pcols = part.n / PACK;
        if !part.n.is_multiple_of(PACK)
            || part.qweight.len() != k * pcols * 4
            || part.qzeros.len() != groups * pcols * 4
            || part.scales.len() != groups * part.n
        {
            return Err(format!(
                "an AWQ projection of {} outputs over {k} inputs is not stored whole",
                part.n
            ));
        }
        let (mine, after) = rest.split_at_mut(part.n * row_bytes);
        rest = after;
        let word = |bytes: &[u8], at: usize| -> u32 {
            u32::from_le_bytes(bytes[at * 4..at * 4 + 4].try_into().expect("4 bytes"))
        };
        // One packed column holds 8 outputs: its rows are written by one task.
        mine.par_chunks_mut(PACK * row_bytes)
            .enumerate()
            .for_each(|(pc, rows8)| {
                let mut words = vec![0u32; block];
                for b in 0..k / block {
                    let g = b * block / group_size;
                    for (t, w) in words.iter_mut().enumerate() {
                        *w = word(part.qweight, (b * block + t) * pcols + pc);
                    }
                    let zeros = word(part.qzeros, g * pcols + pc);
                    for (j, row) in rows8.chunks_exact_mut(row_bytes).enumerate() {
                        let shift = 4 * AWQ_ORDER[j];
                        let scale = part.scales[g * part.n + pc * PACK + j];
                        let zero = (zeros >> shift) & 0xF;
                        let code = |t: usize| (words[t] >> shift) & 0xF;
                        let dst = &mut row[b * block_bytes..(b + 1) * block_bytes];
                        dst[0..2].copy_from_slice(&scale.to_le_bytes());
                        let half = block / 2;
                        if exact {
                            // The code less the zero lies in -15..15: its biased code fits 5 bits.
                            let biased = |t: usize| code(t) + Q5_0_BIAS - zero;
                            let mut high = 0u32;
                            for t in 0..half {
                                let (lo, hi) = (biased(t), biased(t + half));
                                dst[6 + t] = (lo & 0xF | (hi & 0xF) << 4) as u8;
                                high |= (lo >> 4) << t | (hi >> 4) << (t + half);
                            }
                            dst[2..6].copy_from_slice(&high.to_le_bytes());
                        } else {
                            let min = f16::from_f32(-(zero as f32) * scale.to_f32());
                            dst[2..4].copy_from_slice(&min.to_le_bytes());
                            for t in 0..half {
                                dst[4 + t] = (code(t) | code(t + half) << 4) as u8;
                            }
                        }
                    }
                }
            });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rows written from a stored projection give the stored values back: exactly in the
    /// exact format, up to the rounding of each block's minimum in the other.
    #[test]
    fn block_rows_give_the_stored_values() {
        use crate::tensor::quant_cpu::to_float_bytes;
        let (k, gs) = (256usize, 128usize);
        let mut state = 0x9e37_79b9_u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let build = |n: usize, next: &mut dyn FnMut() -> u32| {
            let qweight: Vec<i32> = (0..k * n / PACK).map(|_| next() as i32).collect();
            let qzeros: Vec<i32> = (0..k / gs * n / PACK).map(|_| next() as i32).collect();
            let scales: Vec<f16> = (0..k / gs * n)
                .map(|_| f16::from_f32(1e-3 + (next() % 1000) as f32 * 1e-5))
                .collect();
            AwqTensor::new(k, n, gs, qweight, qzeros, scales)
        };
        let parts = [build(16, &mut next), build(8, &mut next)];
        let bytes = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|w| w.to_le_bytes()).collect() };
        let stored: Vec<(Vec<u8>, Vec<u8>)> = parts
            .iter()
            .map(|t| (bytes(&t.qweight), bytes(&t.qzeros)))
            .collect();
        let views: Vec<StoredAwq> = parts
            .iter()
            .zip(&stored)
            .map(|(t, (w, z))| StoredAwq {
                qweight: w,
                qzeros: z,
                scales: t.scales.clone(),
                n: t.n,
            })
            .collect();
        let rows = 24;
        for exact in [true, false] {
            let format = host_block_format(exact);
            let row_bytes = k / format.block_size() * format.type_size();
            let mut out = vec![0u8; rows * row_bytes];
            write_block_rows(&views, k, gs, format, &mut out).unwrap();
            let mut got = vec![0f32; rows * k];
            to_float_bytes(format, &out, &mut got).unwrap();
            let mut row = 0;
            for t in &parts {
                let want = t.dequant();
                for o in 0..t.n {
                    for kk in 0..k {
                        let g = kk / gs;
                        let pc = o / PACK;
                        let shift = 4 * AWQ_ORDER[o % PACK];
                        let zero = ((t.qzeros[g * (t.n / PACK) + pc] as u32 >> shift) & 0xF) as f32;
                        let scale = t.scales[g * t.n + o].to_f32();
                        // Half a unit in the last place of an f16 minimum, nothing when exact.
                        let bound = match exact {
                            true => 0.0,
                            false => (zero * scale).abs() * 2f32.powi(-11),
                        };
                        let (w, g2) = (want[kk * t.n + o], got[row * k + kk]);
                        assert!(
                            (w - g2).abs() <= bound,
                            "{format:?} row {row} input {kk}: {g2} against {w}"
                        );
                    }
                    row += 1;
                }
            }
        }
    }

    /// The exact rows multiply as rows of `Q8_0` blocks holding the same values do: the same
    /// signed codes against the same quantised activation under the same scale, to the bit.
    #[test]
    fn exact_rows_multiply_as_eight_bit_rows_of_the_same_values() {
        use crate::tensor::quant_cpu::{matmul_bytes, to_float_bytes};
        use crate::tensor::quantized::GgmlDType;
        let (k, n, gs) = (512usize, 24usize, 128usize);
        let mut state = 0x2545_f491_u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let qweight: Vec<i32> = (0..k * n / PACK).map(|_| next() as i32).collect();
        let qzeros: Vec<i32> = (0..k / gs * n / PACK).map(|_| next() as i32).collect();
        let scales: Vec<f16> = (0..k / gs * n)
            .map(|_| f16::from_f32(1e-3 + (next() % 1000) as f32 * 1e-5))
            .collect();
        let bytes = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|w| w.to_le_bytes()).collect() };
        let (w, z) = (bytes(&qweight), bytes(&qzeros));
        let part = StoredAwq {
            qweight: &w,
            qzeros: &z,
            scales: scales.clone(),
            n,
        };
        let exact = host_block_format(true);
        let mut rows = vec![0u8; n * k / exact.block_size() * exact.type_size()];
        write_block_rows(&[part], k, gs, exact, &mut rows).unwrap();

        // The same values as Q8_0 blocks: the group's scale, the code less the zero.
        let t = AwqTensor::new(k, n, gs, qweight, qzeros, scales);
        let q8 = GgmlDType::Q8_0;
        let mut eight = vec![0u8; n * k / q8.block_size() * q8.type_size()];
        for (o, row) in eight
            .chunks_exact_mut(k / q8.block_size() * q8.type_size())
            .enumerate()
        {
            for (b, block) in row.chunks_exact_mut(q8.type_size()).enumerate() {
                let g = b * q8.block_size() / gs;
                let shift = 4 * AWQ_ORDER[o % PACK];
                let zero = (t.qzeros[g * (n / PACK) + o / PACK] as u32 >> shift) & 0xF;
                block[0..2].copy_from_slice(&t.scales[g * n + o].to_le_bytes());
                for j in 0..q8.block_size() {
                    let kk = b * q8.block_size() + j;
                    let code = (t.qweight[kk * (n / PACK) + o / PACK] as u32 >> shift) & 0xF;
                    block[2 + j] = (code as i8 - zero as i8) as u8;
                }
            }
        }
        let (mut a, mut b) = (vec![0f32; n * k], vec![0f32; n * k]);
        to_float_bytes(exact, &rows, &mut a).unwrap();
        to_float_bytes(q8, &eight, &mut b).unwrap();
        assert_eq!(a, b, "the 2 forms hold different values");
        for m in [1usize, 3] {
            let x: Vec<f32> = (0..m * k)
                .map(|i| ((next() % 2001) as f32 - 1000.0) * 1e-3 + (i % 5) as f32 * 0.1)
                .collect();
            let (mut got, mut want) = (vec![0f32; m * n], vec![0f32; m * n]);
            matmul_bytes(exact, (m, k, n), &x, &rows, &mut got).unwrap();
            matmul_bytes(q8, (m, k, n), &x, &eight, &mut want).unwrap();
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&want), "{m} rows");
        }
        // The rows of a prompt through the host matmul the layers call.
        let matmul = |format: GgmlDType, blocks: &[u8]| {
            let held = crate::tensor::quantized::QTensor::host_filled(format, vec![n, k], |out| {
                out.copy_from_slice(blocks);
                Ok(())
            })
            .unwrap();
            crate::tensor::quantized::QMatMul::from_qtensor(held).unwrap()
        };
        let (wide, narrow) = (matmul(q8, &eight), matmul(exact, &rows));
        let rows_in = 9;
        let x: Vec<f32> = (0..rows_in * k)
            .map(|i| ((next() % 2001) as f32 - 1000.0) * 1e-3 + (i % 3) as f32 * 0.2)
            .collect();
        let x = crate::tensor::Tensor::from_vec_f32(x, vec![rows_in, k]).unwrap();
        let bits = |t: crate::tensor::Tensor| {
            t.to_vec_f32()
                .iter()
                .map(|f| f.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            bits(narrow.forward(&x).unwrap()),
            bits(wide.forward(&x).unwrap()),
            "{rows_in} prompt rows"
        );
    }

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
