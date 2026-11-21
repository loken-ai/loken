//! How fast the host's threads read their memory.

/// The host bytes a rate probe of `want` bytes takes: all of them while they fit in the burst
/// headroom and the host's room now, the smallest probe otherwise.
pub(crate) fn probe_bytes(want: usize) -> usize {
    use super::layer_executor::{host_burst_headroom_bytes, host_spill_budget_bytes};
    probe_size(
        want,
        host_spill_budget_bytes().min(host_burst_headroom_bytes()),
    )
}

/// [`probe_bytes`] for a host with `room` bytes to give a probe.
fn probe_size(want: usize, room: u64) -> usize {
    const SMALLEST: usize = 1 << 20;
    if want as u64 <= room {
        want.max(SMALLEST)
    } else {
        SMALLEST
    }
}

/// Bytes per microsecond the host's threads read a stack of `bytes` bytes at, as the host
/// computation of an expert does: the best of a few reads of a block that is not in any cache.
/// Zero when it could not be measured.
pub(crate) fn host_read_bandwidth(bytes: usize) -> f64 {
    use rayon::prelude::*;
    let len = probe_bytes(bytes.min(1 << 28));
    let block = vec![1u8; len];
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        let sum: u64 = block
            .par_chunks(1 << 16)
            .map(|c| c.iter().map(|&b| b as u64).sum::<u64>())
            .sum();
        let us = t.elapsed().as_secs_f64() * 1e6;
        std::hint::black_box(sum);
        best = best.min(us);
    }
    if best.is_finite() && best > 0.0 {
        len as f64 / best
    } else {
        0.0
    }
}

/// The host's memory bandwidth in GB/s, measured once by the largest probe the host's room
/// gives: what a decode step of a block on the host is paced by.
/// `None` when it could not be measured.
pub fn host_bandwidth() -> Option<f64> {
    static BW: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    // Bytes per microsecond are thousandths of GB/s.
    let gbs = *BW.get_or_init(|| host_read_bandwidth(usize::MAX) / 1e3);
    (gbs > 0.0).then_some(gbs)
}

#[cfg(test)]
mod tests {
    /// A probe takes its bytes while the room holds them, and the smallest probe otherwise.
    #[test]
    fn a_probe_fits_the_room_or_takes_the_smallest_size() {
        assert_eq!(super::probe_size(64 << 20, 1 << 30), 64 << 20);
        assert_eq!(super::probe_size(64 << 20, 32 << 20), 1 << 20);
        assert_eq!(super::probe_size(4096, 1 << 30), 1 << 20);
        assert_eq!(super::probe_size(64 << 20, 0), 1 << 20);
    }
}
