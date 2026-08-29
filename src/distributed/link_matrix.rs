//! What each link can carry, as a matrix rather than a scalar.
//!
//! The single-node planner asks every card what it can HOLD - free VRAM - and ranks cards by
//! compute (`device_probe`: cores x clock). It never asks what the path between two of them can
//! CARRY, so two cards with identical compute look interchangeable while one of them sits behind
//! four lanes and the other behind sixteen. A pipeline split that crosses the narrow link pays
//! for it on every token, and nothing in the plan can see the difference.
//!
//! This is the clustering plan's fabric profiler applied to the case that exists today: the links
//! inside one host. The shape is deliberately the same one a multi-host cluster needs - a cost per
//! ORDERED pair, not a single "interconnect speed" - because a real fabric is heterogeneous down
//! to its wiring, and one plan may span a fast local link and a slow uplink at once.
//!
//! Everything here is measured or read from the kernel. Nothing is a nameplate constant: a card
//! negotiates its width and generation at boot, and a x16 card in a x4 slot reports x4.

use std::collections::BTreeMap;

/// Per-lane throughput of one PCI Express generation, in GB/s, after line coding.
///
/// Gen1/2 use 8b/10b (20% lost to the encoding), Gen3 and later use 128b/130b. These are the
/// signalling rates the specification fixes, so they belong in code; what must never be assumed
/// is which of them a given slot NEGOTIATED, which is why `pcie_link` reads that from the kernel.
fn pcie_lane_gbps(generation: u32) -> f64 {
    match generation {
        1 => 0.250,
        2 => 0.500,
        3 => 0.985,
        4 => 1.969,
        5 => 3.938,
        6 => 7.563,
        _ => 0.0,
    }
}

/// One end of a link, as the kernel currently reports it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcieLink {
    /// Lanes actually negotiated (not the connector's width).
    pub width: u32,
    /// Generation actually negotiated.
    pub generation: u32,
}

impl PcieLink {
    /// Usable throughput of this link, one direction.
    pub fn gbps(&self) -> f64 {
        pcie_lane_gbps(self.generation) * f64::from(self.width)
    }
}

/// The cost of moving a tensor from one device to another.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinkCost {
    /// Throughput of the narrower end of the path, GB/s. A transfer between two cards with no
    /// peer-to-peer support bounces through host memory, so it crosses BOTH links and the slower
    /// one sets the pace - which is what makes this a property of the PAIR, not of a card.
    pub gbps: f64,
    /// Whether the two devices can address each other directly. Without it every cross-card move
    /// is two transfers and a host buffer.
    pub peer_to_peer: bool,
}

impl LinkCost {
    /// Milliseconds to move `bytes` across this link, ignoring per-transfer latency.
    pub fn transfer_ms(&self, bytes: u64) -> f64 {
        if self.gbps <= 0.0 {
            return f64::INFINITY;
        }
        (bytes as f64 / 1e9) / self.gbps * 1000.0
    }
}

/// What every ordered pair of local devices can carry.
#[derive(Debug, Clone, Default)]
pub struct LinkMatrix {
    /// Per-device link as negotiated, keyed by CUDA ordinal.
    links: BTreeMap<usize, PcieLink>,
    /// Pair costs, keyed by (from, to).
    costs: BTreeMap<(usize, usize), LinkCost>,
}

impl LinkMatrix {
    /// Fill the pair costs from the per-device links.
    ///
    /// Without peer-to-peer a cross-device move is device -> host -> device, so it is bounded by
    /// the SLOWER of the two links. That bound is symmetric, which the test below relies on: a
    /// matrix that disagrees with itself across the diagonal is measuring noise, not a fabric.
    fn derive_costs(&mut self) {
        let ids: Vec<usize> = self.links.keys().copied().collect();
        for &a in &ids {
            for &b in &ids {
                if a == b {
                    continue;
                }
                let gbps = self.links[&a].gbps().min(self.links[&b].gbps());
                self.costs.insert(
                    (a, b),
                    LinkCost {
                        gbps,
                        peer_to_peer: false,
                    },
                );
            }
        }
    }

    /// The cost of moving from `from` to `to`, if known.
    pub fn cost(&self, from: usize, to: usize) -> Option<LinkCost> {
        self.costs.get(&(from, to)).copied()
    }

    /// Devices ordered by how well they are connected, best first.
    ///
    /// The compute ranking in `device_probe` answers "which card computes fastest"; this answers
    /// "which card is cheapest to reach", and a placement that spans cards needs both. They can
    /// disagree - a faster card behind fewer lanes is the whole reason this module exists.
    pub fn by_link_quality(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.links.keys().copied().collect();
        v.sort_by(|a, b| {
            self.links[b]
                .gbps()
                .partial_cmp(&self.links[a].gbps())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(b))
        });
        v
    }

    /// True when the devices are not all connected alike - the case a scalar cannot express.
    pub fn is_heterogeneous(&self) -> bool {
        let mut seen: Option<PcieLink> = None;
        for l in self.links.values() {
            match seen {
                None => seen = Some(*l),
                Some(first) if first != *l => return true,
                _ => {}
            }
        }
        false
    }

    /// One line per device, for the load-time log.
    pub fn describe(&self) -> Vec<String> {
        self.by_link_quality()
            .into_iter()
            .map(|i| {
                let l = self.links[&i];
                format!(
                    "GPU {i}: PCIe gen{} x{} = {:.1} GB/s",
                    l.generation,
                    l.width,
                    l.gbps()
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_rates_follow_the_generation() {
        // A generation this code does not know must read as zero rather than as a guess, so a
        // future card reports "unknown" instead of a plausible wrong number.
        assert_eq!(pcie_lane_gbps(99), 0.0);
        assert!(pcie_lane_gbps(5) > pcie_lane_gbps(4));
        assert!(pcie_lane_gbps(4) > pcie_lane_gbps(3));
    }

    #[test]
    fn a_narrow_link_costs_more_than_a_wide_one() {
        let wide = PcieLink {
            width: 16,
            generation: 4,
        };
        let narrow = PcieLink {
            width: 4,
            generation: 4,
        };
        assert!(
            (wide.gbps() / narrow.gbps() - 4.0).abs() < 1e-9,
            "four times the lanes"
        );
        let c = LinkCost {
            gbps: narrow.gbps(),
            peer_to_peer: false,
        };
        // A hidden-size activation is small; a KV block is not. The matrix has to answer for both.
        assert!(c.transfer_ms(8 * 1024) < 1.0);
        assert!(c.transfer_ms(2 * 1024 * 1024 * 1024) > 100.0);
    }

    #[test]
    fn the_matrix_is_symmetric_and_reports_asymmetric_wiring() {
        let mut m = LinkMatrix::default();
        m.links.insert(
            0,
            PcieLink {
                width: 16,
                generation: 4,
            },
        );
        m.links.insert(
            1,
            PcieLink {
                width: 4,
                generation: 4,
            },
        );
        m.derive_costs();

        // Bounded by the slower end, so the cost cannot depend on the direction.
        assert_eq!(m.cost(0, 1), m.cost(1, 0));
        // And it is the NARROW link that sets it, not the average and not the wide one.
        assert!((m.cost(0, 1).unwrap().gbps - m.links[&1].gbps()).abs() < 1e-9);

        assert!(
            m.is_heterogeneous(),
            "x16 next to x4 is not a uniform fabric"
        );
        assert_eq!(m.by_link_quality(), vec![0, 1]);

        // A pair with no measurement must say so rather than return a default.
        assert!(m.cost(0, 7).is_none());
    }

    /// The regression this module was built for: on the host it was written on, both naive
    /// readings of sysfs make a x16 card and a x4 card look identical, in opposite directions.
    #[test]
    fn neither_naive_reading_of_the_link_can_see_the_narrow_card() {
        // What the driver actually reported: (max GT/s, max lanes, current GT/s, current lanes).
        let gpu0 = (32.0_f64, 16_u32, 2.5_f64, 16_u32);
        let gpu1 = (32.0_f64, 16_u32, 8.0_f64, 4_u32);
        let gen = |gts: f64| {
            if gts >= 31.0 {
                5
            } else if gts >= 7.0 {
                3
            } else {
                1
            }
        };

        // Everything from `current`: the widths differ but the speeds differ the other way.
        let cur = |g: (f64, u32, f64, u32)| pcie_lane_gbps(gen(g.2)) * f64::from(g.3);
        assert!(
            (cur(gpu0) - cur(gpu1)).abs() < 1.0,
            "current-only makes them comparable"
        );

        // Everything from `max`: both cards claim the same, because that is the DEVICE's rating.
        let max = |g: (f64, u32, f64, u32)| pcie_lane_gbps(gen(g.0)) * f64::from(g.1);
        assert_eq!(max(gpu0), max(gpu1), "max-only makes them identical");

        // The crossing used by `pcie_link`: generation from max, lanes from current.
        let real = |g: (f64, u32, f64, u32)| pcie_lane_gbps(gen(g.0)) * f64::from(g.3);
        assert!(
            (real(gpu0) / real(gpu1) - 4.0).abs() < 1e-9,
            "four times the lanes, visible at last"
        );
    }

    #[test]
    fn identical_wiring_is_not_heterogeneous() {
        let mut m = LinkMatrix::default();
        m.links.insert(
            0,
            PcieLink {
                width: 8,
                generation: 3,
            },
        );
        m.links.insert(
            1,
            PcieLink {
                width: 8,
                generation: 3,
            },
        );
        m.derive_costs();
        assert!(!m.is_heterogeneous());
    }
}
