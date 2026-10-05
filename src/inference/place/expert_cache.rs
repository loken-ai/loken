//! How much of a card a cache of routed experts may take: what the load left of the budget the
//! plan gave the card.

use crate::tensor::Device;

/// Bytes the card behind `dev` has free now, `0` for a device that is not a card.
pub fn free_now(dev: &Device) -> u64 {
    #[cfg(feature = "cuda")]
    {
        // The reading itself: the cache is sized from it, here, with the other placement reads.
        if let Ok((free, _)) = crate::tensor::cuda_ext::mem_get_info(dev) {
            return free as u64;
        }
    }
    let _ = dev;
    0
}

/// Experts each of `layers` cached layers may hold: the budget the plan gave the card, less what
/// the load took of it since `free_before`, shared evenly, each expert `expert_bytes` wide.
pub fn slots_per_layer(
    budget: u64,
    free_before: u64,
    free_after: u64,
    layers: usize,
    expert_bytes: usize,
) -> usize {
    if layers == 0 || expert_bytes == 0 {
        return 0;
    }
    let used = free_before.saturating_sub(free_after);
    let left = budget.saturating_sub(used);
    (left / (layers as u64 * expert_bytes as u64)) as usize
}

#[cfg(test)]
mod tests {
    use super::slots_per_layer;

    #[test]
    fn the_cache_takes_what_the_load_left_of_the_budget() {
        // 10 GB budget, 4 GB taken by the load, 20 layers of 50 MB experts.
        assert_eq!(
            slots_per_layer(
                10_000_000_000,
                16_000_000_000,
                12_000_000_000,
                20,
                50_000_000
            ),
            6
        );
        // The load took more than the budget: nothing left, and no underflow.
        assert_eq!(slots_per_layer(1_000, 9_000, 1_000, 4, 10), 0);
        assert_eq!(slots_per_layer(1_000, 9_000, 1_000, 0, 10), 0);
    }
}
