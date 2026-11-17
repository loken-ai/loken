//! The interleaved repack GEMM: load a weight tile once, spend it on four activation rows.
//!
//! The per-column path walks `(row, col)` doing one dot per pair, so it re-reads every weight
//! column once per prompt row. That is fine at decode, where there is one row, and is most of
//! the cost of a long prefill. Repacking the blocks once into an eight-column interleave lets
//! a loaded tile serve four rows before it is evicted.
//!
//! Seven formats support it, and they differ in exactly four things: the block type the weight
//! bytes cast to, the repack that builds the interleave, the GEMM that consumes it, and the
//! cell the repacked copy lives in. The list at the bottom of this file is that declaration,
//! and there is no second place where a format's participation is recorded.

use super::*;

macro_rules! repack_gemm_formats {
    ($( $variant:ident => $block:ident, $repack:ident, $matmul:ident, $cache:ident ; )*) => {
        impl QKernelMatMul {
            /// Run the tiled GEMM over `lhs`, or decline and let the caller fall onward.
            ///
            /// Declines - rather than erroring - when the shape does not tile, when the format
            /// has no repack GEMM, or when the weight bytes do not cast to the block type the
            /// format claims. Every one of those is a reason to take a slower path, not to
            /// fail the matmul.
            pub(super) fn try_repack_gemm(
                &self,
                lhs: &[half::f16],
                rows: usize,
                odims: &[usize],
            ) -> Option<crate::tensor::Result<crate::tensor::Tensor>> {
                use crate::tensor::quant_cpu as qc;

                // Four rows is where reuse begins to pay for the repack; eight columns is the
                // interleave width every one of these layouts is built around.
                let bs = self.dtype.block_size();
                if rows < 4 || self.n % 8 != 0 || bs == 0 || self.k % bs != 0 {
                    return None;
                }
                let nb = self.k / bs;

                let dst32 = match self.dtype {
                    $(
                        GgmlDType::$variant => {
                            let blocks = qc::cast_blocks::<qc::$block>(self.host.data()).ok()?;
                            if blocks.len() != self.n * nb {
                                return None;
                            }
                            let packed = self
                                .$cache
                                // CONVERSION: x8 interleaved layout the host dot kernels read; the stored blocks stay for the paths that read them.
                                .get_or_init(|| qc::$repack::repack(blocks, self.n, nb));
                            let lhs_f32: Vec<f32> = lhs.iter().map(|&v| v.to_f32()).collect();
                            let mut dst32 = vec![0f32; rows * self.n];
                            if let Err(e) = qc::$matmul(
                                (rows, self.k, self.n),
                                &lhs_f32,
                                packed,
                                &mut dst32,
                            ) {
                                return Some(Err(e));
                            }
                            dst32
                        }
                    )*
                    _ => return None,
                };

                // The output carries f16 because the caller reached here from an f16
                // activation, and a half-carrier model's residual stream must come back half.
                let dst: Vec<half::f16> = dst32.into_iter().map(half::f16::from_f32).collect();
                Some(crate::tensor::Tensor::from_storage(
                    crate::tensor::CpuStorage::F16(dst),
                    odims.to_vec(),
                ))
            }
        }
    };
}

repack_gemm_formats! {
    Q4K   => BlockQ4K,   repack_q4k,      matmul_q4k_repacked_tiled,  cpu_repack_q4k;
    Q5K   => BlockQ5K,   repack_q5k,      matmul_q5k_repacked_tiled,  cpu_repack_q5k;
    Q6K   => BlockQ6K,   repack_q6k,      matmul_q6k_repacked_tiled,  cpu_repack_q6k;
    Q4_0  => BlockQ4_0,  repack_q4_0,     matmul_q4_0_repacked_tiled, cpu_repack_q4_0;
    Q5_0  => BlockQ5_0,  repack_q5_0_wide, matmul_q8_0_x8_tiled,      cpu_repack_q8_0_x8;
    Q8_0  => BlockQ8_0,  repack_q8_0_x8,  matmul_q8_0_x8_tiled,       cpu_repack_q8_0_x8;
    MxFp4 => BlockMxFp4, repack_mxfp4_x8, matmul_mxfp4_x8_tiled,      cpu_repack_mxfp4_x8;
}

impl QKernelMatMul {
    /// Bytes of the repacked copy a host weight of `values` values in `dtype` builds beside its
    /// stored blocks, 0 for a format that builds none.
    ///
    /// The copy is built at the weight's first host GEMM that reads it, and only while the
    /// measured host budget holds it.
    pub fn host_copy_bytes(dtype: GgmlDType, values: usize) -> u64 {
        use crate::tensor::quant_cpu as qc;
        use std::mem::size_of;
        // Each interleaved element holds one block of 8 output rows.
        let x8 = |elem: usize, block: usize| (values / (8 * block) * elem) as u64;
        match dtype {
            GgmlDType::Q4K => x8(size_of::<qc::repack_q4k::BlockQ4Kx8>(), qc::QK_K),
            GgmlDType::Q5K => x8(size_of::<qc::repack_q5k::BlockQ5Kx8>(), qc::QK_K),
            GgmlDType::Q6K => x8(size_of::<qc::repack_q6k::BlockQ6Kx8>(), qc::QK_K),
            GgmlDType::MxFp4 => x8(
                size_of::<qc::repack_mxfp4_x8::BlockMxFp4x8>(),
                dtype.block_size(),
            ),
            // Q5_0 reads its prompts from Q8_0 blocks of the same values.
            GgmlDType::Q8_0 | GgmlDType::Q5_0 => x8(
                size_of::<qc::repack_q8_0_x8::BlockQ8_0x8>(),
                dtype.block_size(),
            ),
            GgmlDType::Q4_0 => (values / dtype.block_size() * size_of::<qc::BlockQ4_0>()) as u64,
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quant_cpu as qc;

    /// The size [`QKernelMatMul::host_copy_bytes`] states is the size each repack builds.
    #[test]
    fn the_copy_size_is_the_size_each_repack_builds() {
        let (n, k) = (16usize, 512usize);
        let values: Vec<f32> = (0..n * k)
            .map(|i| ((i * 37) % 101) as f32 / 101.0 - 0.5)
            .collect();
        let stored = |dtype: GgmlDType| qc::from_float_bytes(dtype, &values).unwrap();
        let built = |dtype: GgmlDType| -> u64 {
            let bytes = stored(dtype);
            let nb = k / dtype.block_size();
            let len = |v: usize, elem: usize| (v * elem) as u64;
            use std::mem::size_of;
            match dtype {
                GgmlDType::Q4K => {
                    let b = qc::cast_blocks::<qc::BlockQ4K>(&bytes).unwrap();
                    let r = qc::repack_q4k::repack(b, n, nb);
                    len(r.len(), size_of::<qc::repack_q4k::BlockQ4Kx8>())
                }
                GgmlDType::Q5K => {
                    let b = qc::cast_blocks::<qc::BlockQ5K>(&bytes).unwrap();
                    let r = qc::repack_q5k::repack(b, n, nb);
                    len(r.len(), size_of::<qc::repack_q5k::BlockQ5Kx8>())
                }
                GgmlDType::Q6K => {
                    let b = qc::cast_blocks::<qc::BlockQ6K>(&bytes).unwrap();
                    let r = qc::repack_q6k::repack(b, n, nb);
                    len(r.len(), size_of::<qc::repack_q6k::BlockQ6Kx8>())
                }
                GgmlDType::MxFp4 => {
                    let b = qc::cast_blocks::<qc::BlockMxFp4>(&bytes).unwrap();
                    let r = qc::repack_mxfp4_x8::repack(b, n, nb);
                    len(r.len(), size_of::<qc::repack_mxfp4_x8::BlockMxFp4x8>())
                }
                GgmlDType::Q8_0 => {
                    let b = qc::cast_blocks::<qc::BlockQ8_0>(&bytes).unwrap();
                    let r = qc::repack_q8_0_x8::repack(b, n, nb);
                    len(r.len(), size_of::<qc::repack_q8_0_x8::BlockQ8_0x8>())
                }
                GgmlDType::Q5_0 => {
                    let b = qc::cast_blocks::<qc::BlockQ5_0>(&bytes).unwrap();
                    let r = qc::repack_q5_0_wide::repack(b, n, nb);
                    len(r.len(), size_of::<qc::repack_q8_0_x8::BlockQ8_0x8>())
                }
                GgmlDType::Q4_0 => {
                    let b = qc::cast_blocks::<qc::BlockQ4_0>(&bytes).unwrap();
                    let r = qc::repack_q4_0::repack(b, n, nb);
                    len(r.len(), size_of::<qc::BlockQ4_0>())
                }
                _ => 0,
            }
        };
        for dtype in [
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
            GgmlDType::MxFp4,
            GgmlDType::Q8_0,
            GgmlDType::Q5_0,
            GgmlDType::Q4_0,
            GgmlDType::Q4_1,
        ] {
            assert_eq!(
                QKernelMatMul::host_copy_bytes(dtype, n * k),
                built(dtype),
                "{dtype:?}"
            );
        }
    }
}
