//! Text adapters a request selects on a loaded base model.
//!
//! The base weights are loaded once.
//! Each adapter a request names is read once, on its first request, and kept resident.
//! Its pairs sit on the device of the projection they correct, as the plan placed that layer.
//! A request selects its adapters before its forwards and clears them after.
//! Each projection then adds the fused pair of the selected adapters to the weight's answer.
//! A request that selects none runs on the base weights alone.

use super::projection::QMatMul;
use super::*;
use crate::inference::load::lora::{transformer_keys, LoraFile, ADAPTER_SITES};
use crate::tensor::lora::{fuse_loras, LoraDelta};
use crate::tensor::Error;

/// The slot of a layer's fused QKV projection, after the slots of [`ADAPTER_SITES`].
const FUSED_QKV: usize = ADAPTER_SITES.len();

/// The slots of a layer's gate and up projections in [`ADAPTER_SITES`].
const GATE: usize = 4;
const UP: usize = 5;

/// One projection an adapter corrects, and its pair at strength 1.
pub(crate) struct AdapterPair {
    /// Index of the layer in the model's stack.
    layer: usize,
    /// The projection: a slot of [`site_mut`].
    slot: usize,
    /// The tensor-parallel share of the layer: 0 for the layer itself.
    share: usize,
    /// `[1, in, r]`.
    down: Tensor,
    /// `[1, r, out]`, the file's `alpha / r` folded in.
    up: Tensor,
}

impl AdapterPair {
    /// Bytes the pair holds on its device.
    fn bytes(&self) -> u64 {
        [&self.down, &self.up]
            .iter()
            .map(|t| (t.elem_count() * t.dtype().size_in_bytes()) as u64)
            .sum()
    }
}

/// The adapters a model holds for its requests, and the ones the running request selected.
#[derive(Default)]
pub(crate) struct AdapterLibrary {
    /// Each resident adapter by name, with the pairs of every projection it corrects.
    resident: Vec<(String, Vec<AdapterPair>)>,
    /// The adapters the running request selected, by name.
    selected: Vec<String>,
}

impl AdapterLibrary {
    fn pairs(&self, name: &str) -> Option<&[AdapterPair]> {
        self.resident
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, p)| p.as_slice())
    }
}

/// The projection of `part` in `slot`: a site of [`ADAPTER_SITES`], or [`FUSED_QKV`].
/// `None` where the layer has no such projection.
pub(crate) fn site_mut(part: &mut GenericTransformerLayer, slot: usize) -> Option<&mut QMatMul> {
    match slot {
        0 => part.attn_q.as_mut(),
        1 => part.attn_k.as_mut(),
        2 => part.attn_v.as_mut(),
        3 => Some(&mut part.attn_output),
        GATE => part.ffn_gate.as_mut(),
        UP => Some(&mut part.ffn_up),
        6 => Some(&mut part.ffn_down),
        FUSED_QKV => part.attn_qkv.as_mut(),
        _ => None,
    }
}

/// One delta of an adapter file placed on one projection.
pub(crate) struct SiteDelta {
    pub layer: usize,
    /// The site of [`ADAPTER_SITES`] the file names.
    pub site: usize,
    /// The projection that takes the delta: `site`, or [`FUSED_QKV`] for q, k and v of a layer
    /// that fuses them.
    pub slot: usize,
    pub share: usize,
    /// On the device of the projection it corrects.
    pub delta: LoraDelta,
}

/// `up` `[r, w]` placed at column `offset` of `[r, total]` zeros.
fn place_columns(up: &Tensor, offset: usize, total: usize) -> Result<Tensor> {
    let (r, w) = up.dims2()?;
    let device = up.device();
    let mut parts = Vec::with_capacity(3);
    if offset > 0 {
        parts.push(Tensor::zeros_on((r, offset), up.dtype(), &device)?);
    }
    parts.push(up.clone());
    if total > offset + w {
        parts.push(Tensor::zeros_on(
            (r, total - offset - w),
            up.dtype(),
            &device,
        )?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, 1)?.contiguous()
}

impl GenericHeteroTransformer {
    /// Calls `f` on every projection an adapter may reach, in every layer and every share.
    pub(crate) fn each_adapted_projection(&mut self, mut f: impl FnMut(&mut QMatMul)) {
        for layer in self.layers.iter_mut() {
            let mut shards = std::mem::take(&mut layer.tp_shards);
            for part in std::iter::once(&mut *layer).chain(shards.iter_mut()) {
                for slot in 0..=FUSED_QKV {
                    if let Some(p) = site_mut(part, slot) {
                        f(p);
                    }
                }
            }
            layer.tp_shards = shards;
        }
    }

    /// The deltas of `file` at `strength` for the projections of this model.
    ///
    /// Each delta is moved to the device of the projection it corrects.
    /// A projection cut by rows across a layer's shares takes its slice of the delta's outputs.
    /// A projection cut by columns takes its slice of the delta's inputs.
    /// A fused QKV projection takes the q, k and v deltas, each `up` placed at the columns of its
    /// part of the fused outputs and zero elsewhere.
    /// A gate and up the load fused into `ffn_up` take the gate and up deltas the same way, the
    /// gate's columns first.
    pub(crate) fn adapter_deltas(
        &mut self,
        file: &LoraFile,
        strength: f32,
    ) -> Result<Vec<SiteDelta>> {
        let first = self.first_layer;
        let q_out = self.config.n_head * self.config.head_dim;
        let kv_out = self.config.n_kv_head * self.config.head_dim;
        let mut out = Vec::new();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let ranks = 1 + layer.tp_shards.len();
            for (site, (projection, by_rows)) in ADAPTER_SITES.iter().enumerate() {
                let mut delta = None;
                for key in transformer_keys(first + i, projection) {
                    if let Some(d) = file.delta_for_key(&key, strength)? {
                        delta = Some(d);
                        break;
                    }
                }
                let Some(delta) = delta else { continue };
                let mut shards = std::mem::take(&mut layer.tp_shards);
                let parts = std::iter::once(&mut *layer).chain(shards.iter_mut());
                for (share, part) in parts.enumerate() {
                    let (down, up) = match (ranks, by_rows) {
                        (1, _) => (delta.down.clone(), delta.up.clone()),
                        (_, true) => {
                            let per = delta.up.dim(1)? / ranks;
                            (
                                delta.down.clone(),
                                delta.up.narrow(1, share * per, per)?.contiguous()?,
                            )
                        }
                        (_, false) => {
                            let per = delta.down.dim(0)? / ranks;
                            (
                                delta.down.narrow(0, share * per, per)?.contiguous()?,
                                delta.up.clone(),
                            )
                        }
                    };
                    // A share of a fused QKV projection holds its q, then its k, then its v.
                    let fused = site < 3 && site_mut(part, site).is_none();
                    // A share whose gate and up the load fused holds its gate, then its up, in
                    // `ffn_up`, twice as wide as either.
                    let width = up.dim(1)?;
                    let fused_ffn = (site == GATE || site == UP)
                        && part.ffn_gate.is_none()
                        && part.ffn_up.out_dim() == 2 * width;
                    let (slot, up) = if fused {
                        let (q, kv) = (q_out / ranks, kv_out / ranks);
                        let offset = [0, q, q + kv][site];
                        (FUSED_QKV, place_columns(&up, offset, q + 2 * kv)?)
                    } else if fused_ffn {
                        let offset = (site - GATE) * width;
                        (UP, place_columns(&up, offset, 2 * width)?)
                    } else {
                        (site, up)
                    };
                    let Some(weight) = site_mut(part, slot) else {
                        continue;
                    };
                    let n = weight.out_dim();
                    if n != 0 && up.dim(1)? != n {
                        return Err(Error(format!(
                            "adapter: layer {} {projection} projects to {} on a weight producing {n}",
                            first + i,
                            up.dim(1)?
                        )));
                    }
                    let device = weight.device();
                    out.push(SiteDelta {
                        layer: i,
                        site,
                        slot,
                        share,
                        delta: LoraDelta {
                            down: down.to_device(&device)?,
                            up: up.to_device(&device)?,
                            scale: delta.scale,
                        },
                    });
                }
                layer.tp_shards = shards;
            }
        }
        Ok(out)
    }

    /// Reads the adapter file at `path` and keeps it resident under `name`.
    ///
    /// Returns the bytes its pairs hold on their devices.
    /// A file already resident under `name` is not read again.
    /// A file that corrects no projection of this model is refused.
    pub fn load_adapter(&mut self, name: &str, path: &std::path::Path) -> Result<u64> {
        if let Some(pairs) = self.adapter_library.pairs(name) {
            return Ok(pairs.iter().map(AdapterPair::bytes).sum());
        }
        let file = LoraFile::load(&path.to_string_lossy(), &Device::Cpu)?;
        // The deltas of one projection fuse into one pair.
        let mut by_slot: std::collections::BTreeMap<(usize, usize, usize), Vec<LoraDelta>> =
            std::collections::BTreeMap::new();
        for d in self.adapter_deltas(&file, 1.0)? {
            by_slot
                .entry((d.layer, d.slot, d.share))
                .or_default()
                .push(d.delta);
        }
        let mut pairs = Vec::with_capacity(by_slot.len());
        for ((layer, slot, share), deltas) in by_slot {
            let din = deltas[0].down.dim(0)?;
            let dout = deltas[0].up.dim(1)?;
            let Some((down, up)) = fuse_loras(&deltas, din, dout) else {
                continue;
            };
            pairs.push(AdapterPair {
                layer,
                slot,
                share,
                down,
                up,
            });
        }
        if pairs.is_empty() {
            return Err(Error(format!(
                "adapter {name} matched no projection on this checkpoint"
            )));
        }
        let bytes = pairs.iter().map(AdapterPair::bytes).sum();
        self.adapter_library
            .resident
            .push((name.to_string(), pairs));
        Ok(bytes)
    }

    /// Names of the adapters resident for requests, in the order they were read.
    pub fn resident_adapters(&self) -> Vec<String> {
        self.adapter_library
            .resident
            .iter()
            .map(|(n, _)| n.clone())
            .collect()
    }

    /// Bytes the resident adapters hold, by device.
    pub fn resident_adapter_bytes(&self) -> Vec<(crate::tensor::DeviceLocation, u64)> {
        let mut by_device: Vec<(crate::tensor::DeviceLocation, u64)> = Vec::new();
        for (_, pairs) in &self.adapter_library.resident {
            for p in pairs {
                let at = p.down.device().location();
                match by_device.iter_mut().find(|(d, _)| *d == at) {
                    Some((_, b)) => *b += p.bytes(),
                    None => by_device.push((at, p.bytes())),
                }
            }
        }
        by_device
    }

    /// Selects the adapters the next forwards add to the base, each at its strength.
    ///
    /// An empty list returns every projection to the base and the attached adapters.
    /// An adapter not yet resident is resolved by name in the adapter directory and read.
    /// Several adapters on one projection fuse into one pair.
    ///
    /// A change of the selection moves the graph generation.
    pub fn select_adapters(&mut self, wanted: &[(String, f32)]) -> Result<()> {
        if !wanted.is_empty() {
            if let Some(why) = self.adapter_refusal() {
                return Err(Error(format!("adapters refused: {why}")));
            }
        }
        if !wanted.is_empty() || !self.adapter_library.selected.is_empty() {
            self.note_adapters_changed();
        }
        self.each_adapted_projection(|p| p.select_lora(None));
        self.adapter_library.selected.clear();
        for (name, strength) in wanted {
            if !strength.is_finite() {
                return Err(Error(format!("adapter {name}: strength {strength}")));
            }
            if self.adapter_library.pairs(name).is_none() {
                let path = crate::inference::load::lora::resolve(name).map_err(Error)?;
                self.load_adapter(name, &path)?;
            }
        }
        let mut by_slot: std::collections::BTreeMap<
            (usize, usize, usize),
            Vec<(&AdapterPair, f32)>,
        > = std::collections::BTreeMap::new();
        for (name, strength) in wanted {
            for p in self.adapter_library.pairs(name).unwrap_or_default() {
                by_slot
                    .entry((p.layer, p.slot, p.share))
                    .or_default()
                    .push((p, *strength));
            }
        }
        let mut chosen = Vec::with_capacity(by_slot.len());
        for (key, pairs) in by_slot {
            let pair = match pairs.as_slice() {
                [(p, s)] if *s == 1.0 => Some((p.down.clone(), p.up.clone())),
                _ => {
                    let (din, dout) = (pairs[0].0.down.dim(1)?, pairs[0].0.up.dim(2)?);
                    let deltas = pairs
                        .iter()
                        .map(|(p, s)| {
                            Ok(LoraDelta {
                                down: p.down.squeeze(0)?,
                                up: p.up.squeeze(0)?,
                                scale: *s,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    fuse_loras(&deltas, din, dout)
                }
            };
            chosen.push((key, pair));
        }
        for ((layer, slot, share), pair) in chosen {
            let mut part = &mut self.layers[layer];
            if share > 0 {
                part = &mut part.tp_shards[share - 1];
            }
            if let Some(weight) = site_mut(part, slot) {
                weight.select_lora(pair);
            }
        }
        self.adapter_library.selected = wanted.iter().map(|(n, _)| n.clone()).collect();
        Ok(())
    }

    /// Names of the adapters the running request selected.
    pub fn selected_adapters(&self) -> &[String] {
        &self.adapter_library.selected
    }

    /// Why an adapter cannot reach every layer of this model, `None` when it can.
    ///
    /// A Vulkan or OpenCL stack computes its layers from the file's blocks, which carry no adapter.
    pub(crate) fn adapter_refusal(&self) -> Option<&'static str> {
        if !self.vulkan.is_empty() {
            return Some("layers run through Vulkan, which adds no adapter");
        }
        if !self.opencl.is_empty() {
            return Some("layers run through OpenCL, which adds no adapter");
        }
        None
    }

    /// Moves the graph generation.
    ///
    /// A graph recorded before the change of the adapters is not replayed after it.
    pub(crate) fn note_adapters_changed(&mut self) {
        self.graph_generation = self.graph_generation.wrapping_add(1);
    }
}
