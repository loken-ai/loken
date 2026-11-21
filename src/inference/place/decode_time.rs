//! The seconds a decoded token takes on a plan, estimated before the load.
//!
//! A token runs through the blocks in order, then the output head on the device of the last block.
//! A block on a card costs its bytes over the card's bandwidth.
//! A block on the host costs its bytes over the host's bandwidth.
//! A mixture's block costs its dense bytes and the routed set of a token, not every expert.
//! A routed set on the host costs what a load measured of 1 set on the host's cores.
//! A routed set kept apart from its layer on a card costs what a load measured of 1 set there.
//!
//! The estimate ranks plans of one model.
//! [`calibrated`] scales it to the rate the engine measured on the placement it loaded.

use super::ladder::Demand;
use super::layer_executor::{DeviceKind, HeteroPlan};
use super::tensor_parallel::RoutedExperts;

/// Seconds 1 routed set of 1 layer takes over 1 row, as a load measured it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SetSecs {
    /// On the host's cores.
    pub host_secs: f64,
    /// On a card that keeps the set: the row up, the kernels, the outputs down.
    pub card_secs: f64,
}

/// A mixture's routed experts as the estimate sees them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoutedSets {
    /// Bytes of 1 layer's routed experts, all of them, that the demand's block bytes count.
    pub layer_bytes: u64,
    /// Bytes of the experts 1 token routes to in 1 layer.
    pub set_bytes: u64,
    /// Layers on a card from this index on run their routed sets apart from the rest of the layer.
    pub apart_from: usize,
    /// Share of the sets run apart that a card runs, from 0 to 1.
    /// The host's cores run the rest.
    pub card_share: f64,
    /// The last measure of 1 set, `None` before a load measured one.
    pub measure: Option<SetSecs>,
}

impl RoutedSets {
    /// The routed sets of `experts` on `plan`, with the last `measure` of 1 set.
    ///
    /// With `in_blocks` the demand's block bytes count every expert, and the layers from the
    /// plan's first layer with host experts run their sets apart.
    /// Without it the experts sit apart from every layer.
    /// Every set run apart is counted on the host's cores: the slowest the cards' caches give.
    pub fn on(
        plan: &HeteroPlan,
        experts: RoutedExperts,
        in_blocks: bool,
        measure: Option<SetSecs>,
    ) -> Self {
        let set_bytes =
            (experts.bytes as u128 * experts.used as u128 / experts.experts.max(1) as u128) as u64;
        let (layer_bytes, apart_from) = match in_blocks {
            true => (
                experts.bytes,
                plan.host_experts_from.unwrap_or(plan.total_layers),
            ),
            false => (0, 0),
        };
        Self {
            layer_bytes,
            set_bytes,
            apart_from,
            card_share: 0.0,
            measure,
        }
    }

    /// Seconds 1 set run apart from its layer takes, `None` before a measure.
    fn apart_secs(&self) -> Option<f64> {
        let m = self.measure?;
        let share = self.card_share.clamp(0.0, 1.0);
        Some(share * m.card_secs + (1.0 - share) * m.host_secs)
    }
}

impl Demand {
    /// Seconds a decoded token takes on `plan`: each block on the device that holds it, the
    /// output head on the device of the last block, `card` giving a card's bandwidth and `host`
    /// the host's, in GB/s.
    ///
    /// `routed` describes a mixture's routed experts.
    /// `None` when a device's bandwidth is not known, when a routed set sits on the host or apart
    /// from its layer with no measure of it, or when every layer is split across cards.
    pub fn decode_secs(
        &self,
        plan: &HeteroPlan,
        card: &impl Fn(usize) -> Option<f64>,
        host: &impl Fn() -> Option<f64>,
        routed: Option<&RoutedSets>,
    ) -> Option<f64> {
        if plan.tp_cards.is_some() || plan.segments.is_empty() {
            return None;
        }
        let bytes_per_sec = |kind: DeviceKind| -> Option<f64> {
            let gbs = match kind {
                DeviceKind::Cuda(idx) => card(idx),
                DeviceKind::Cpu => host(),
                DeviceKind::OpenCL(_) | DeviceKind::Vulkan(_) => None,
            }?;
            (gbs.is_finite() && gbs > 0.0).then_some(gbs * 1e9)
        };
        let per_layer = self.block_bytes as f64 / self.layers.max(1) as f64;
        let mut secs = 0.0;
        for segment in &plan.segments {
            if segment.layer_end <= segment.layer_start {
                continue;
            }
            let speed = bytes_per_sec(segment.kind)?;
            for layer in segment.layer_start..segment.layer_end {
                secs += match routed {
                    None => per_layer / speed,
                    Some(r) => {
                        let dense = (per_layer - r.layer_bytes as f64).max(0.0) / speed;
                        let set = match segment.kind {
                            DeviceKind::Cpu => r.measure?.host_secs,
                            _ if layer >= r.apart_from => r.apart_secs()?,
                            _ => r.set_bytes as f64 / speed,
                        };
                        dense + set
                    }
                };
            }
        }
        if self.head_bytes > 0 {
            let last = plan.segments.last()?.kind;
            secs += self.head_bytes as f64 / bytes_per_sec(last)?;
        }
        Some(secs)
    }
}

/// `secs`, an estimate of a plan of a model, scaled by the rate the engine measured on the
/// placement it loaded: `loaded_secs` is the estimate of that placement and `loaded_tok_s` the
/// decode rate measured on it.
/// `None` when an input is not a positive finite number.
pub fn calibrated(secs: f64, loaded_secs: f64, loaded_tok_s: f64) -> Option<f64> {
    let fine = |v: f64| v.is_finite() && v > 0.0;
    (fine(secs) && fine(loaded_secs) && fine(loaded_tok_s))
        .then(|| secs / (loaded_secs * loaded_tok_s))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::place::layer_executor::HeteroSegment;

    const GB: u64 = 1_000_000_000;

    fn segment(kind: DeviceKind, from: usize, to: usize) -> HeteroSegment {
        HeteroSegment {
            kind,
            layer_start: from,
            layer_end: to,
            free_memory_bytes: 0,
        }
    }

    fn plan(segments: Vec<HeteroSegment>, host_experts_from: Option<usize>) -> HeteroPlan {
        let total_layers = segments.last().map_or(0, |s| s.layer_end);
        HeteroPlan {
            segments,
            tp_cards: None,
            total_layers,
            host_experts_from,
        }
    }

    fn card(idx: usize) -> Option<f64> {
        match idx {
            0 => Some(100.0),
            1 => Some(50.0),
            _ => None,
        }
    }

    fn host() -> Option<f64> {
        Some(20.0)
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * b.abs().max(1.0)
    }

    /// A plan on the cards costs each card's bytes over its bandwidth, as the token cost says.
    #[test]
    fn a_plan_on_the_cards_costs_its_bytes_over_the_bandwidths() {
        let d = Demand {
            layers: 10,
            block_bytes: 10 * GB,
            head_bytes: GB,
            ..Demand::default()
        };
        let p = plan(
            vec![
                segment(DeviceKind::Cuda(1), 0, 4),
                segment(DeviceKind::Cuda(0), 4, 10),
            ],
            None,
        );
        let secs = d.decode_secs(&p, &card, &host, None).unwrap();
        assert!(close(secs, 4.0 / 50.0 + 7.0 / 100.0));
        let cost = d.token_cost(&p, &card).unwrap();
        assert!(close(secs, cost / 1e9));
        // A card whose bandwidth is not known gives no estimate.
        let unknown = plan(vec![segment(DeviceKind::Cuda(2), 0, 10)], None);
        assert_eq!(d.decode_secs(&unknown, &card, &host, None), None);
    }

    /// Blocks on the host cost their bytes over the host's bandwidth, the head with them.
    #[test]
    fn host_layers_cost_their_bytes_over_the_host_bandwidth() {
        let d = Demand {
            layers: 10,
            block_bytes: 10 * GB,
            head_bytes: GB,
            ..Demand::default()
        };
        let p = plan(
            vec![
                segment(DeviceKind::Cuda(0), 0, 6),
                segment(DeviceKind::Cpu, 6, 10),
            ],
            None,
        );
        let secs = d.decode_secs(&p, &card, &host, None).unwrap();
        assert!(close(secs, 6.0 / 100.0 + 4.0 / 20.0 + 1.0 / 20.0));
        let full = plan(vec![segment(DeviceKind::Cuda(0), 0, 10)], None);
        assert!(d.decode_secs(&full, &card, &host, None).unwrap() < secs);
        // A host whose bandwidth is not known gives no estimate.
        assert_eq!(d.decode_secs(&p, &card, &|| None, None), None);
    }

    fn mixture() -> (Demand, RoutedExperts) {
        let d = Demand {
            layers: 4,
            block_bytes: 4 * GB,
            ..Demand::default()
        };
        // 800 MB of experts a layer, 8 of 64 routed: a set of 100 MB.
        let experts = RoutedExperts {
            bytes: 8 * GB / 10,
            experts: 64,
            used: 8,
        };
        (d, experts)
    }

    const MEASURE: SetSecs = SetSecs {
        host_secs: 0.01,
        card_secs: 0.002,
    };

    /// Routed sets on the host cost the measure of 1 set on the host's cores; the dense part of
    /// each layer stays on its card.
    #[test]
    fn host_experts_cost_the_measured_set() {
        let (d, experts) = mixture();
        let p = plan(vec![segment(DeviceKind::Cuda(0), 0, 4)], Some(2));
        let mut sets = RoutedSets::on(&p, experts, true, Some(MEASURE));
        assert_eq!(sets.set_bytes, GB / 10);
        assert_eq!(sets.apart_from, 2);
        let secs = d.decode_secs(&p, &card, &host, Some(&sets)).unwrap();
        // 2 layers read 200 MB dense and a 100 MB set on the card, 2 read their dense part there
        // and run their set on the host.
        assert!(close(secs, 2.0 * 0.3 / 100.0 + 2.0 * (0.2 / 100.0 + 0.01)));
        // A share run by a card costs the card's measure.
        sets.card_share = 0.5;
        let shared = d.decode_secs(&p, &card, &host, Some(&sets)).unwrap();
        assert!(close(
            shared,
            2.0 * 0.3 / 100.0 + 2.0 * (0.2 / 100.0 + 0.006)
        ));
        // A whole layer on the host reads its dense part there and runs its set on its cores.
        let spilled = plan(
            vec![
                segment(DeviceKind::Cuda(0), 0, 3),
                segment(DeviceKind::Cpu, 3, 4),
            ],
            None,
        );
        let sets = RoutedSets::on(&spilled, experts, true, Some(MEASURE));
        let secs = d.decode_secs(&spilled, &card, &host, Some(&sets)).unwrap();
        assert!(close(secs, 3.0 * 0.3 / 100.0 + 0.2 / 20.0 + 0.01));
    }

    /// Experts the demand's blocks do not count sit apart from every layer.
    #[test]
    fn experts_outside_the_blocks_run_apart_from_every_layer() {
        let (d, experts) = mixture();
        let p = plan(vec![segment(DeviceKind::Cuda(0), 0, 4)], None);
        let sets = RoutedSets::on(&p, experts, false, Some(MEASURE));
        let secs = d.decode_secs(&p, &card, &host, Some(&sets)).unwrap();
        assert!(close(secs, 4.0 * (1.0 / 100.0 + 0.01)));
    }

    /// Routed sets on the host with no measure of them give no estimate; the same model with
    /// every expert on the cards still has one.
    #[test]
    fn host_experts_without_a_measure_give_no_estimate() {
        let (d, experts) = mixture();
        let apart = plan(vec![segment(DeviceKind::Cuda(0), 0, 4)], Some(2));
        let sets = RoutedSets::on(&apart, experts, true, None);
        assert_eq!(d.decode_secs(&apart, &card, &host, Some(&sets)), None);
        let spilled = plan(
            vec![
                segment(DeviceKind::Cuda(0), 0, 3),
                segment(DeviceKind::Cpu, 3, 4),
            ],
            None,
        );
        let sets = RoutedSets::on(&spilled, experts, true, None);
        assert_eq!(d.decode_secs(&spilled, &card, &host, Some(&sets)), None);
        let full = plan(vec![segment(DeviceKind::Cuda(0), 0, 4)], None);
        let sets = RoutedSets::on(&full, experts, true, None);
        let secs = d.decode_secs(&full, &card, &host, Some(&sets)).unwrap();
        assert!(close(secs, 4.0 * 0.3 / 100.0));
    }

    /// The calibration scales every plan by the measured rate over the estimate of the placement
    /// it was measured on.
    #[test]
    fn calibration_scales_by_the_measured_rate_of_the_loaded_plan() {
        // The loaded plan was estimated at 20 ms a token and decodes at 40 tokens a second.
        assert!(close(calibrated(0.02, 0.02, 40.0).unwrap(), 0.025));
        assert!(close(calibrated(0.04, 0.02, 40.0).unwrap(), 0.05));
        assert_eq!(calibrated(0.04, 0.0, 40.0), None);
        assert_eq!(calibrated(0.04, 0.02, f64::NAN), None);
        assert_eq!(calibrated(-1.0, 0.02, 40.0), None);
    }
}
