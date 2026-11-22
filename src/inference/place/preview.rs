//! The report of a placement before the load: the layers each device takes, the bytes of
//! weights and cache it holds, the host's share, and the context the cache is built for.
//!
//! The plan comes from the code a load runs.
//! This module only counts what the plan puts where.

use super::ladder::Demand;
use super::layer_executor::{DeviceKind, HeteroPlan};
use serde::{Deserialize, Serialize};

/// Where a load would place a model, and what it would hold on each device.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PlacementPreview {
    pub model: String,
    pub architecture: String,
    /// Repeating blocks the plan divides.
    pub layers: usize,
    /// Bytes of weights the devices and the host place together.
    pub weight_bytes: u64,
    /// Tokens the cache is built for: the window a request may open.
    pub context_length: usize,
    /// The cache element format: `q8_0`, `q4_0`, `f16` or `f32`.
    pub kv_format: String,
    /// Each device the plan gives layers to, in the order the layers cross them.
    pub devices: Vec<DevicePreview>,
    pub host: HostShare,
    /// Why no plan may be built, when the allowed backends refuse every one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refused: Option<String>,
    /// Each device whose weights exceed what it offered: the plan promised more than the device
    /// has.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
    /// What a load would take back from the other models on the cards before it places this
    /// one, present when it would take anything.
    /// The devices then report the plan built on the room taken back.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reclaimed: Option<super::reclaim::Reclaimed>,
}

/// What one device takes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DevicePreview {
    /// `CUDA`, `Vulkan`, `OpenCL` or `CPU`.
    pub device: String,
    /// The device's ordinal among those of its backend, 0 for the host.
    pub index: usize,
    /// The layer ranges it runs, each `[first, end)` with the end excluded.
    pub layers: Vec<[usize; 2]>,
    /// Bytes of weights resident on it.
    pub weight_bytes: u64,
    /// Bytes of cache its layers hold at the planned context.
    pub kv_bytes: u64,
    /// Bytes it offered the plan, absent for the host.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offered_bytes: Option<u64>,
    /// Bytes of routed experts a wide residual keeps on it in the room the plan leaves, counted
    /// in `weight_bytes`.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub expert_bytes: u64,
}

fn is_zero(bytes: &u64) -> bool {
    *bytes == 0
}

/// What stays in host memory.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HostShare {
    /// Layers the host runs.
    pub layers: usize,
    /// Bytes of weights no device holds: host layers, routed experts kept in host memory and
    /// tables read row by row.
    pub weight_bytes: u64,
    /// `weight_bytes` over all the weights, from 0 to 1.
    pub share: f64,
}

/// What the report is counted from.
pub struct Facts<'a> {
    pub model: &'a str,
    pub architecture: &'a str,
    /// The plan the load builds.
    pub plan: &'a HeteroPlan,
    /// What the model asks of each device.
    pub demand: &'a Demand,
    /// Each layer's dense and routed expert bytes, when the source can say.
    pub per_layer: Option<&'a [(u64, u64)]>,
    /// A layer's routed expert bytes, the figure the plan prices its experts with.
    pub routed_layer_bytes: Option<u64>,
    /// Whether the demand's block bytes hold the routed experts.
    /// Without it the experts sit in host memory apart from every layer.
    pub experts_in_blocks: bool,
    /// Bytes of weights the devices and the host place together.
    pub weight_bytes: u64,
    pub context_length: usize,
    pub kv_format: &'a str,
}

/// The name and the ordinal a device is reported by.
pub fn device_label(kind: DeviceKind) -> (&'static str, usize) {
    match kind {
        DeviceKind::Cuda(i) => ("CUDA", i),
        DeviceKind::Vulkan(i) => ("Vulkan", i),
        DeviceKind::OpenCL(i) => ("OpenCL", i),
        DeviceKind::Cpu => ("CPU", 0),
    }
}

/// Bytes of weights of layers `[start, end)` that sit with their layer.
///
/// With the experts in the blocks, a layer from the plan's first layer with host experts keeps
/// its routed experts in host memory.
/// Without them, the block bytes are what the devices hold and the experts are all apart.
fn layer_weights(facts: &Facts<'_>, start: usize, end: usize) -> u64 {
    let demand = facts.demand;
    let uniform = |l: usize| {
        let layers = demand.layers.max(1) as u128;
        let upto = |n: usize| (demand.block_bytes as u128 * n as u128 / layers) as u64;
        upto(l + 1) - upto(l)
    };
    if !facts.experts_in_blocks {
        return (start..end).map(uniform).sum();
    }
    let apart_from = facts
        .plan
        .host_experts_from
        .unwrap_or(facts.plan.total_layers);
    let apart = |l: usize| l >= apart_from;
    (start..end)
        .map(|l| match facts.per_layer.and_then(|p| p.get(l)) {
            Some(&(dense, _)) if apart(l) => dense,
            Some(&(dense, experts)) => dense + experts,
            None if apart(l) => uniform(l).saturating_sub(facts.routed_layer_bytes.unwrap_or(0)),
            None => uniform(l),
        })
        .sum()
}

/// The report of `facts`, each device offering the bytes `offered` gives it.
pub fn build(facts: Facts<'_>, offered: impl Fn(DeviceKind) -> Option<u64>) -> PlacementPreview {
    let plan = facts.plan;
    let demand = facts.demand;
    let mut devices: Vec<(DeviceKind, DevicePreview)> = Vec::new();
    let mut add = |kind: DeviceKind, range: [usize; 2], weights: u64, kv: u64| {
        let at = match devices.iter().position(|(k, _)| *k == kind) {
            Some(at) => at,
            None => {
                let (device, index) = device_label(kind);
                devices.push((
                    kind,
                    DevicePreview {
                        device: device.to_string(),
                        index,
                        offered_bytes: offered(kind),
                        ..DevicePreview::default()
                    },
                ));
                devices.len() - 1
            }
        };
        let d = &mut devices[at].1;
        if !d.layers.contains(&range) {
            d.layers.push(range);
        }
        d.weight_bytes += weights;
        d.kv_bytes += kv;
    };
    for seg in &plan.segments {
        let range = [seg.layer_start, seg.layer_end];
        let mut weights = layer_weights(&facts, seg.layer_start, seg.layer_end);
        if seg.layer_end == demand.layers {
            weights += demand.head_bytes;
        }
        let kv = demand
            .kv_bytes_per_layer
            .saturating_mul(seg.num_layers() as u64);
        // A split layer's weights and cache are shared by the group in proportion to the
        // cards' weights.
        match (&plan.tp_cards, seg.kind) {
            (Some(group), DeviceKind::Cuda(_)) if !group.is_empty() => {
                let total: f64 = group.iter().map(|c| c.weight).sum();
                for card in group {
                    let part = if total > 0.0 {
                        card.weight / total
                    } else {
                        1.0 / group.len() as f64
                    };
                    add(
                        DeviceKind::Cuda(card.index),
                        range,
                        (weights as f64 * part) as u64,
                        (kv as f64 * part) as u64,
                    );
                }
            }
            _ => add(seg.kind, range, weights, kv),
        }
    }
    let on_devices: u64 = devices
        .iter()
        .filter(|(k, _)| !matches!(k, DeviceKind::Cpu))
        .map(|(_, d)| d.weight_bytes)
        .sum();
    let host_weights = facts.weight_bytes.saturating_sub(on_devices);
    let host = HostShare {
        layers: plan
            .segments
            .iter()
            .filter(|s| matches!(s.kind, DeviceKind::Cpu))
            .map(|s| s.num_layers())
            .sum(),
        weight_bytes: host_weights,
        share: if facts.weight_bytes > 0 {
            host_weights as f64 / facts.weight_bytes as f64
        } else {
            0.0
        },
    };
    let devices: Vec<DevicePreview> = devices.into_iter().map(|(_, d)| d).collect();
    let warnings = devices
        .iter()
        .filter_map(|d| {
            let offered = d.offered_bytes?;
            (d.weight_bytes > offered).then(|| {
                format!(
                    "{} {} holds {} of weights, more than the {} it offered",
                    d.device,
                    d.index,
                    gb(d.weight_bytes),
                    gb(offered)
                )
            })
        })
        .collect();
    PlacementPreview {
        model: facts.model.to_string(),
        architecture: facts.architecture.to_string(),
        layers: demand.layers,
        weight_bytes: facts.weight_bytes,
        context_length: facts.context_length,
        kv_format: facts.kv_format.to_string(),
        devices,
        host,
        refused: None,
        warnings,
        reclaimed: None,
    }
}

/// The report of a component placed whole on one device by `plan`, `demand` its one block.
/// `None` for `plan` when the allowed backends refuse every rung.
pub fn whole(
    model: &str,
    architecture: &str,
    plan: Option<&HeteroPlan>,
    demand: &Demand,
    context_length: usize,
    kv_format: &str,
    offered: impl Fn(DeviceKind) -> Option<u64>,
) -> PlacementPreview {
    let weight_bytes = demand.block_bytes.saturating_add(demand.head_bytes);
    let Some(plan) = plan else {
        return refused(model, architecture, demand.layers, weight_bytes);
    };
    build(
        Facts {
            model,
            architecture,
            plan,
            demand,
            per_layer: None,
            routed_layer_bytes: None,
            experts_in_blocks: true,
            weight_bytes,
            context_length,
            kv_format,
        },
        offered,
    )
}

/// The report of a model of `layers` blocks and `weight_bytes` of weights that the allowed
/// backends refuse to place.
pub fn refused(
    model: &str,
    architecture: &str,
    layers: usize,
    weight_bytes: u64,
) -> PlacementPreview {
    PlacementPreview {
        model: model.to_string(),
        architecture: architecture.to_string(),
        layers,
        weight_bytes,
        refused: Some(
            "the model does not fit the allowed backends, and the host is not one of them"
                .to_string(),
        ),
        ..PlacementPreview::default()
    }
}

/// Bytes as gigabytes with 2 decimals.
fn gb(bytes: u64) -> String {
    format!("{:.2} GB", bytes as f64 / 1e9)
}

impl PlacementPreview {
    /// Adds the routed experts each card keeps beside its layers: `(ordinal, offered, bytes)`.
    /// A card the plan gives no layer joins the devices with its experts alone.
    /// The host keeps what the cards then do not.
    pub fn add_card_experts(&mut self, experts: &[(usize, u64, u64)]) {
        for &(ordinal, offered, bytes) in experts.iter().filter(|e| e.2 > 0) {
            let at = match self
                .devices
                .iter()
                .position(|d| d.device == "CUDA" && d.index == ordinal)
            {
                Some(at) => at,
                None => {
                    self.devices.push(DevicePreview {
                        device: "CUDA".to_string(),
                        index: ordinal,
                        offered_bytes: Some(offered),
                        ..DevicePreview::default()
                    });
                    self.devices.len() - 1
                }
            };
            let d = &mut self.devices[at];
            d.weight_bytes = d.weight_bytes.saturating_add(bytes);
            d.expert_bytes = d.expert_bytes.saturating_add(bytes);
        }
        let on_devices: u64 = self
            .devices
            .iter()
            .filter(|d| d.device != "CPU")
            .map(|d| d.weight_bytes)
            .sum();
        self.host.weight_bytes = self.weight_bytes.saturating_sub(on_devices);
        self.host.share = if self.weight_bytes > 0 {
            self.host.weight_bytes as f64 / self.weight_bytes as f64
        } else {
            0.0
        };
    }

    /// The report as a short table for a terminal.
    pub fn table(&self) -> String {
        let mut out = format!(
            "{} ({}, {} layers, {} of weights)\n",
            self.model,
            self.architecture,
            self.layers,
            gb(self.weight_bytes)
        );
        if let Some(why) = &self.refused {
            out.push_str(&format!("no plan: {why}\n"));
            return out;
        }
        if !self.kv_format.is_empty() {
            out.push_str(&format!(
                "context {} tokens, cache {}\n",
                self.context_length, self.kv_format
            ));
        }
        let rows: Vec<[String; 5]> = self
            .devices
            .iter()
            .map(|d| {
                let name = match d.device.as_str() {
                    "CPU" => "host".to_string(),
                    _ => format!("{} {}", d.device, d.index),
                };
                let layers = d
                    .layers
                    .iter()
                    .map(|[a, b]| format!("{a}-{b}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                [
                    name,
                    layers,
                    gb(d.weight_bytes),
                    gb(d.kv_bytes),
                    d.offered_bytes.map_or_else(|| "-".to_string(), gb),
                ]
            })
            .collect();
        let header = [
            "device".to_string(),
            "layers".to_string(),
            "weights".to_string(),
            "cache".to_string(),
            "offered".to_string(),
        ];
        let widths: Vec<usize> = (0..header.len())
            .map(|c| {
                rows.iter()
                    .chain(std::iter::once(&header))
                    .map(|r| r[c].len())
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        for row in std::iter::once(&header).chain(rows.iter()) {
            let line = row
                .iter()
                .zip(&widths)
                .map(|(cell, w)| format!("{cell:<w$}"))
                .collect::<Vec<_>>()
                .join("  ");
            out.push_str(line.trim_end());
            out.push('\n');
        }
        out.push_str(&format!(
            "host share: {} layers, {} of weights ({:.0}%)\n",
            self.host.layers,
            gb(self.host.weight_bytes),
            self.host.share * 100.0
        ));
        let experts: u64 = self.devices.iter().map(|d| d.expert_bytes).sum();
        if experts > 0 {
            out.push_str(&format!(
                "routed experts in the room the cards keep: {}\n",
                gb(experts)
            ));
        }
        if let Some(r) = &self.reclaimed {
            let cards = r
                .cards
                .iter()
                .map(|c| format!("CUDA {} {}", c.index, gb(c.bytes)))
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!("taken back from the other models: {cards}\n"));
            if !r.models.is_empty() {
                out.push_str(&format!("unloaded first: {}\n", r.models.join(", ")));
            }
        }
        for warning in &self.warnings {
            out.push_str(&format!("warning: {warning}\n"));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::place::layer_executor::{HeteroSegment, TpCard};

    fn segment(kind: DeviceKind, layer_start: usize, layer_end: usize) -> HeteroSegment {
        HeteroSegment {
            kind,
            layer_start,
            layer_end,
            free_memory_bytes: 0,
        }
    }

    fn demand() -> Demand {
        Demand {
            layers: 4,
            block_bytes: 400,
            head_bytes: 50,
            kv_bytes_per_layer: 10,
            runtime_reserve: 0,
        }
    }

    fn facts<'a>(plan: &'a HeteroPlan, demand: &'a Demand) -> Facts<'a> {
        Facts {
            model: "m",
            architecture: "a",
            plan,
            demand,
            per_layer: None,
            routed_layer_bytes: None,
            experts_in_blocks: true,
            weight_bytes: 500,
            context_length: 4096,
            kv_format: "q8_0",
        }
    }

    #[test]
    fn a_card_and_the_host_share_the_layers_and_the_host_keeps_the_rest() {
        let plan = HeteroPlan {
            segments: vec![
                segment(DeviceKind::Cuda(0), 0, 3),
                segment(DeviceKind::Cpu, 3, 4),
            ],
            tp_cards: None,
            total_layers: 4,
            host_experts_from: None,
        };
        let d = demand();
        let p = build(facts(&plan, &d), |k| match k {
            DeviceKind::Cuda(0) => Some(1000),
            _ => None,
        });
        assert_eq!(p.devices.len(), 2);
        assert_eq!(p.devices[0].device, "CUDA");
        assert_eq!(p.devices[0].layers, vec![[0, 3]]);
        assert_eq!(p.devices[0].weight_bytes, 300);
        assert_eq!(p.devices[0].kv_bytes, 30);
        assert_eq!(p.devices[0].offered_bytes, Some(1000));
        // The head rides with the last layer, on the host.
        assert_eq!(p.devices[1].weight_bytes, 150);
        assert_eq!(p.host.layers, 1);
        assert_eq!(p.host.weight_bytes, 200);
        assert!((p.host.share - 0.4).abs() < 1e-9);
    }

    #[test]
    fn routed_experts_past_the_cut_stay_in_host_memory() {
        let plan = HeteroPlan {
            segments: vec![segment(DeviceKind::Cuda(1), 0, 4)],
            tp_cards: None,
            total_layers: 4,
            host_experts_from: Some(2),
        };
        let d = demand();
        let per_layer = [(40, 60), (40, 60), (40, 60), (40, 60)];
        let p = build(
            Facts {
                per_layer: Some(&per_layer),
                ..facts(&plan, &d)
            },
            |_| None,
        );
        assert_eq!(p.devices[0].weight_bytes, 100 + 100 + 40 + 40 + 50);
        assert_eq!(p.host.layers, 0);
        assert_eq!(p.host.weight_bytes, 500 - 330);
    }

    #[test]
    fn experts_kept_on_the_host_from_the_first_layer_count_on_the_host() {
        let plan = HeteroPlan {
            segments: vec![segment(DeviceKind::Cuda(0), 0, 4)],
            tp_cards: None,
            total_layers: 4,
            host_experts_from: Some(0),
        };
        let d = demand();
        let per_layer = [(40, 60), (40, 60), (40, 60), (40, 60)];
        let p = build(
            Facts {
                per_layer: Some(&per_layer),
                ..facts(&plan, &d)
            },
            |_| Some(1000),
        );
        // The card holds the dense part of every layer and the head, no routed expert.
        assert_eq!(p.devices[0].weight_bytes, 4 * 40 + 50);
        assert_eq!(p.host.weight_bytes, 500 - 210);
        assert!(p.host.weight_bytes >= 4 * 60, "the host holds every expert");
        assert!(p.warnings.is_empty());
    }

    #[test]
    fn without_per_layer_bytes_the_plans_expert_figure_leaves_the_card() {
        let plan = HeteroPlan {
            segments: vec![segment(DeviceKind::Cuda(0), 0, 4)],
            tp_cards: None,
            total_layers: 4,
            host_experts_from: Some(0),
        };
        let d = demand();
        let p = build(
            Facts {
                routed_layer_bytes: Some(60),
                ..facts(&plan, &d)
            },
            |_| None,
        );
        assert_eq!(p.devices[0].weight_bytes, 4 * (100 - 60) + 50);
    }

    #[test]
    fn experts_outside_the_blocks_never_count_on_a_card() {
        let plan = HeteroPlan {
            segments: vec![segment(DeviceKind::Cuda(0), 0, 4)],
            tp_cards: None,
            total_layers: 4,
            host_experts_from: None,
        };
        // The block bytes are the token mixers the plan puts on the card.
        let d = Demand {
            block_bytes: 80,
            head_bytes: 0,
            ..demand()
        };
        let per_layer = [(40, 60), (40, 60), (40, 60), (40, 60)];
        let p = build(
            Facts {
                per_layer: Some(&per_layer),
                experts_in_blocks: false,
                ..facts(&plan, &d)
            },
            |_| None,
        );
        assert_eq!(p.devices[0].weight_bytes, 80);
        assert_eq!(p.host.weight_bytes, 420);
    }

    #[test]
    fn weights_past_the_offer_are_reported() {
        let plan = HeteroPlan {
            segments: vec![segment(DeviceKind::Cuda(0), 0, 4)],
            tp_cards: None,
            total_layers: 4,
            host_experts_from: None,
        };
        let d = demand();
        let p = build(facts(&plan, &d), |_| Some(100));
        assert_eq!(p.warnings.len(), 1, "{:?}", p.warnings);
        assert!(
            p.warnings[0].starts_with("CUDA 0 holds"),
            "{:?}",
            p.warnings
        );
        assert!(p.table().contains("warning: CUDA 0 holds"));
    }

    #[test]
    fn a_split_layer_is_shared_by_its_cards_in_proportion() {
        let plan = HeteroPlan {
            segments: vec![segment(DeviceKind::Cuda(0), 0, 4)],
            tp_cards: Some(vec![
                TpCard {
                    index: 0,
                    weight: 3.0,
                },
                TpCard {
                    index: 1,
                    weight: 1.0,
                },
            ]),
            total_layers: 4,
            host_experts_from: None,
        };
        let d = demand();
        let p = build(facts(&plan, &d), |_| None);
        assert_eq!(p.devices.len(), 2);
        assert_eq!(p.devices[0].weight_bytes, 337);
        assert_eq!(p.devices[1].weight_bytes, 112);
        assert_eq!(p.devices[1].layers, vec![[0, 4]]);
    }

    #[test]
    fn the_table_names_each_device_and_the_host_share() {
        let plan = HeteroPlan {
            segments: vec![
                segment(DeviceKind::Cuda(0), 0, 3),
                segment(DeviceKind::Cpu, 3, 4),
            ],
            tp_cards: None,
            total_layers: 4,
            host_experts_from: None,
        };
        let d = demand();
        let table = build(facts(&plan, &d), |_| None).table();
        assert!(table.contains("context 4096 tokens, cache q8_0"), "{table}");
        assert!(table.contains("CUDA 0  0-3"), "{table}");
        assert!(table.contains("host    3-4"), "{table}");
        assert!(table.contains("host share: 1 layers"), "{table}");
    }

    #[test]
    fn the_report_reads_back_from_its_json() {
        let plan = HeteroPlan {
            segments: vec![segment(DeviceKind::Cpu, 0, 4)],
            tp_cards: None,
            total_layers: 4,
            host_experts_from: None,
        };
        let d = demand();
        let p = build(facts(&plan, &d), |_| None);
        let back: PlacementPreview =
            serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
        assert_eq!(back, p);
    }
}
