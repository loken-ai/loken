//! The GEMM precision a media render runs at, held for exactly the length of the render.
//!
//! The cuBLAS reduced-precision flags are process-wide atomics. A render that does not set
//! them inherits whatever the last model load or render left, so the same seed can produce
//! different output depending on what ran before it. Every media render pins the precision
//! it needs with [`GemmPrecision`] and the previous state comes back when it ends.

/// Pins the process-wide GEMM precision flags for a scope; the previous flags are restored
/// on drop. Without CUDA there is nothing to pin and the guard is inert.
#[must_use = "the precision is restored when the guard drops"]
pub struct GemmPrecision {
    #[cfg(feature = "cuda")]
    was: (bool, bool, bool),
}

impl GemmPrecision {
    /// Full precision: f32 GEMMs in IEEE f32, bf16 and f16 GEMMs accumulated in f32.
    pub fn full() -> Self {
        Self::pin(false)
    }

    /// `reduced` selects the tensor-core paths (TF32 for f32, fast bf16 accumulation);
    /// f16 accumulates in f32 either way, as `cuda_ext::set_gemm_reduced_precision` sets it.
    pub fn pin(reduced: bool) -> Self {
        #[cfg(feature = "cuda")]
        {
            use crate::tensor::cuda as c;
            let was = (
                c::gemm_reduced_precision_f32(),
                c::gemm_reduced_precision_bf16(),
                c::gemm_reduced_precision_f16(),
            );
            crate::tensor::cuda_ext::set_gemm_reduced_precision(reduced);
            Self { was }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = reduced;
            Self {}
        }
    }
}

impl Drop for GemmPrecision {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            use crate::tensor::cuda as c;
            let (f32_reduced, bf16_reduced, f16_reduced) = self.was;
            c::set_gemm_reduced_precision_f32(f32_reduced);
            c::set_gemm_reduced_precision_bf16(bf16_reduced);
            c::set_gemm_reduced_precision_f16(f16_reduced);
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::tensor::cuda as c;

    #[test]
    fn a_render_scope_sets_the_precision_and_gives_the_old_one_back() {
        let before = (
            c::gemm_reduced_precision_f32(),
            c::gemm_reduced_precision_bf16(),
            c::gemm_reduced_precision_f16(),
        );
        {
            let _reduced = GemmPrecision::pin(true);
            assert!(c::gemm_reduced_precision_f32());
            assert!(c::gemm_reduced_precision_bf16());
            {
                let _full = GemmPrecision::full();
                assert!(!c::gemm_reduced_precision_f32());
                assert!(!c::gemm_reduced_precision_bf16());
            }
            assert!(
                c::gemm_reduced_precision_f32(),
                "inner scope did not restore"
            );
        }
        let after = (
            c::gemm_reduced_precision_f32(),
            c::gemm_reduced_precision_bf16(),
            c::gemm_reduced_precision_f16(),
        );
        assert_eq!(before, after);
    }
}
