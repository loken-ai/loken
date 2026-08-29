//! The host paths: a quantised matmul that runs where the blocks already are.
//!
//! These take and return plain slices rather than tensors. The MoE expert loop calls them per
//! expert with borrowed rows, and going through a tensor for each would allocate once per
//! expert per token.

use super::*;

impl QKernelMatMul {
    pub fn forward_slice_cpu(&self, x: &[f32], out: &mut [f32]) -> Result<bool> {
        if !matches!(self.device, crate::tensor::Device::Cpu)
            || !crate::tensor::quant_cpu::supports(self.dtype)
            || self.k % self.dtype.block_size() != 0
        {
            return Ok(false);
        }
        if x.len() != self.k {
            return Err(Error(format!(
                "forward_slice_cpu: x len {} != k {}",
                x.len(),
                self.k
            )));
        }
        if out.len() != self.n {
            return Err(Error(format!(
                "forward_slice_cpu: out len {} != n {}",
                out.len(),
                self.n
            )));
        }
        #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
        if let Some(x8) = self.q4k_x8() {
            crate::tensor::quant_cpu::matmul_q4k_plain((1, self.k, self.n), x, x8, out)?;
            return Ok(true);
        }
        crate::tensor::quant_cpu::matmul_bytes(
            self.dtype,
            (1, self.k, self.n),
            x,
            self.host.data(),
            out,
        )?;
        Ok(true)
    }

    /// Run several weights against ONE activation, quantising it once.
    ///
    /// At decode the same normed vector feeds `q`, `k` and `v`, and the same FFN
    /// norm feeds `gate` and `up`; `forward_slice_cpu` quantises inside the GEMV,
    /// so each projection repeats that work. The quantisation is the same routine
    /// on the same input, so the result is bit-identical to calling them one by
    /// one - this only stops doing it N times.
    ///
    /// Returns `false` without writing anything when any weight is off the CPU
    /// fast path, so the caller falls back to the per-weight calls unchanged.
    pub fn forward_slice_cpu_shared(
        x: &[f32],
        mats: &[&QKernelMatMul],
        outs: &mut [&mut [f32]],
    ) -> Result<bool> {
        #[cfg(not(all(target_feature = "avx2", target_arch = "x86_64")))]
        {
            let _ = (x, mats, outs);
            return Ok(false);
        }
        #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
        {
            if mats.len() != outs.len() || mats.is_empty() {
                return Ok(false);
            }
            let k = mats[0].k;
            if k % crate::tensor::quant_cpu::QK_K != 0 {
                return Ok(false);
            }
            let mut packs = Vec::with_capacity(mats.len());
            for (i, m) in mats.iter().enumerate() {
                if m.k != k || x.len() != k || outs[i].len() != m.n {
                    return Ok(false);
                }
                match m.q4k_x8() {
                    Some(p) => packs.push(p),
                    None => return Ok(false),
                }
            }
            let nb = k / crate::tensor::quant_cpu::QK_K;
            let aq = crate::tensor::quant_cpu::quantize_activation_q8k(x, nb);
            for (i, p) in packs.into_iter().enumerate() {
                crate::tensor::quant_cpu::matmul_q4k_plain_pre((k, mats[i].n), &aq, p, outs[i])?;
            }
            Ok(true)
        }
    }

    /// Raw CPU quantized weight access `(dtype, k, n, bytes)` for callers that
    /// run a fused multi-weight region (e.g. the MoE flat `matmul_bytes_multi`).
    /// `None` when not on the supported CPU quantized fast path (same gate as
    /// `forward_slice_cpu`).
    pub fn cpu_raw(&self) -> Option<(GgmlDType, usize, usize, &[u8])> {
        if matches!(self.device, crate::tensor::Device::Cpu)
            && crate::tensor::quant_cpu::supports(self.dtype)
            && self.k % self.dtype.block_size() == 0
        {
            Some((self.dtype, self.k, self.n, self.host.data()))
        } else {
            None
        }
    }
}
