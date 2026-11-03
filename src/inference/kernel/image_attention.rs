//! What the `image_attention` OpenCL kernels accept.

/// The widest head the kernels hold: they cache one query row as 32 `float4` registers.
pub const MAX_HEAD_DIM: usize = 128;

/// Refuses a head width the kernels would read wrongly.
///
/// The kernels load a head 4 floats at a time into a fixed cache.
/// A width above `MAX_HEAD_DIM` overruns the cache.
/// A width that is not a multiple of 4 drops its last floats.
pub fn check_head_dim(head_dim: usize) -> anyhow::Result<()> {
    if head_dim == 0 || head_dim > MAX_HEAD_DIM || !head_dim.is_multiple_of(4) {
        anyhow::bail!(
            "image attention needs a head dimension that is a multiple of 4 and at most {}, got {}",
            MAX_HEAD_DIM,
            head_dim
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_head_dimension_the_kernels_hold_is_accepted() {
        for d in [4, 64, 120, MAX_HEAD_DIM] {
            assert!(check_head_dim(d).is_ok(), "{d}");
        }
    }

    #[test]
    fn a_head_dimension_past_the_cache_is_refused() {
        assert!(check_head_dim(MAX_HEAD_DIM + 4).is_err());
        assert!(check_head_dim(256).is_err());
    }

    #[test]
    fn a_head_dimension_off_a_multiple_of_4_is_refused() {
        for d in [0, 1, 3, 62, 126] {
            assert!(check_head_dim(d).is_err(), "{d}");
        }
    }
}
