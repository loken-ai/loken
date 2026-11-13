//! The GEMM precision a media render runs at, held for exactly the length of the render.
//!
//! A render pins the precision it needs with [`GemmPrecision`] on the thread that renders.
//! The choice ends with the render.
//! The products of another thread, a language model's among them, keep their own level.

/// Pins the GEMM precision of this thread for a scope.
/// The previous choice of the thread comes back on drop.
/// Without CUDA there is nothing to pin and the guard is inert.
#[must_use = "the precision is restored when the guard drops"]
pub struct GemmPrecision {
    #[cfg(feature = "gpu")]
    _scope: crate::tensor::cuda::GemmScope,
}

impl GemmPrecision {
    /// Full precision: f32 GEMMs in IEEE f32, bf16 and f16 GEMMs accumulated in f32.
    pub fn full() -> Self {
        Self::pin(false)
    }

    /// `reduced` selects the tensor-core paths (TF32 for f32, fast bf16 accumulation).
    /// F16 accumulates in f32 either way.
    pub fn pin(reduced: bool) -> Self {
        #[cfg(feature = "gpu")]
        {
            Self {
                _scope: crate::tensor::cuda::GemmScope::enter(reduced),
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = reduced;
            Self {}
        }
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use crate::tensor::cuda as c;

    fn levels() -> (bool, bool, bool) {
        (
            c::gemm_reduced_precision_f32(),
            c::gemm_reduced_precision_bf16(),
            c::gemm_reduced_precision_f16(),
        )
    }

    #[test]
    fn a_render_scope_sets_the_precision_and_gives_the_old_one_back() {
        let before = levels();
        {
            let _reduced = GemmPrecision::pin(true);
            assert!(c::gemm_reduced_precision_f32());
            assert!(c::gemm_reduced_precision_bf16());
            assert!(!c::gemm_reduced_precision_f16());
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
        assert_eq!(before, levels());
    }

    /// A render's choice holds on its own thread only.
    #[test]
    fn a_render_scope_leaves_the_other_threads_at_their_level() {
        let process = std::thread::spawn(levels).join().unwrap();
        let _reduced = GemmPrecision::pin(!process.0);
        assert_eq!(c::gemm_reduced_precision_f32(), !process.0);
        assert_eq!(std::thread::spawn(levels).join().unwrap(), process);
    }
}
