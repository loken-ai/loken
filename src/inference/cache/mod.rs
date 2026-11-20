//! Everything that remembers between tokens: the KV caches (paged, quantised,
//! host-side) and the checkpoint cache.
//!
//! Callers name a module through this directory - `crate::inference::cache::<module>` - so the path says which
//! part of the system a file belongs to, which is the whole reason the directory exists.

/// The window a quantised KV cache is built for, and the window the placement budget
/// plans against - one number, because those two must not drift apart.
///
/// A growing cache starts here and doubles on the first append past it, up to the context
/// the configuration allows. The budget plans for THIS rather than for the model's full GGUF
/// context, and that is deliberate: planning the worst case reserved many gigabytes the cache
/// never allocates, which measured as an unnecessary two-card split on a model that fits one,
/// costing about a fifth of its decode rate.
///
/// It was written out separately in the two caches, in the loader and in the planner. Four
/// copies of one policy is four chances for the allocation and the budget to disagree, and
/// that disagreement is either an out-of-memory or a placement nobody asked for.
pub const KV_WORKING_WINDOW_TOKENS: usize = 4096;

/// Positions of a quantised history that a prompt's attention reads back to F16 at once.
///
/// The F16 K and V of a tile take no more bytes than the F32 state the prompt's `rows`
/// carry between tiles: `carry_floats` values per row and query head.
/// The tile is a whole number of the kernel's `keys_per_tile`, at least one.
/// It follows the rows: a smaller chunk of the prompt reads a smaller tile.
pub fn history_tile(
    rows: usize,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    carry_floats: usize,
    keys_per_tile: usize,
) -> usize {
    let f32_bytes = 4;
    let f16_bytes = 2;
    let state = rows * n_head * carry_floats * f32_bytes;
    let per_position = (2 * n_kv * head_dim * f16_bytes).max(1);
    let keys = keys_per_tile.max(1);
    (state / per_position / keys).max(1) * keys
}

pub mod container;
pub mod cpu_f16_kv;
pub mod cpu_q8_kv;
pub mod hf;
pub mod kv_core;
pub mod kv_disk;
pub mod paged_attention;
pub mod paged_kv;
#[cfg(feature = "gpu")]
pub mod q4_kv;
#[cfg(feature = "gpu")]
pub mod q8_kv;
#[cfg(feature = "gpu")]
pub mod quant_kv;
pub mod qvb;
pub mod state_disk;
pub mod state_tune;
/// A 3-bit KV scheme that exists on the host only - no kernel, and no cache type the
/// loader can select. It is here for the measurement in `turboquant::measure`, which is
/// what decides whether the device path gets written at all.
pub mod turboquant;

#[cfg(test)]
mod history_tile_tests {
    use super::history_tile;

    /// A 256-wide head, 16 query heads over 2 KV heads, 264 carried values a row, 64-key tiles.
    fn tile(rows: usize) -> usize {
        history_tile(rows, 16, 2, 256, 264, 64)
    }

    /// The F16 K and V a prompt reads back at once do not grow with the history: the tile
    /// takes no more bytes than the state the rows carry.
    #[test]
    fn the_read_back_is_bounded_by_the_rows_not_the_history() {
        for rows in [36, 512, 2048] {
            let read_back = 2 * tile(rows) * 2 * 256 * 2;
            let carried = rows * 16 * 264 * 4;
            assert!(
                read_back <= carried.max(2 * 64 * 2 * 256 * 2),
                "{rows} rows"
            );
        }
    }

    /// Halving the chunk halves the tile, so a walk that halves the chunk frees room.
    #[test]
    fn a_smaller_chunk_reads_a_smaller_tile() {
        assert!(tile(1024) < tile(2048));
        assert!(tile(1024) * 2 <= tile(2048) + 64);
    }

    /// Every tile but the history's last is a whole number of the kernel's key tiles, at least
    /// one of them.
    #[test]
    fn the_tile_is_whole_kernel_tiles() {
        for rows in [1, 7, 36, 2048] {
            let t = tile(rows);
            assert!(t >= 64 && t.is_multiple_of(64), "{rows} rows: {t}");
        }
    }
}
