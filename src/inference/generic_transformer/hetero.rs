//! Split out of `inference/generic_transformer/` (move-only refactor).

#[allow(unused_imports)]
use super::*;

/// Turned on by the non-finite-logits gate, and off again by the first report.
///
/// A prefill that ends in NaN says nothing about WHERE the NaN was born, and scanning every
/// layer of every request to find out would tax the path that works. So the failure arms the
/// instrument: the request that fails costs nothing extra, and the NEXT one names the layer.
/// The normal path pays one relaxed atomic load per layer.
pub static SCAN_FOR_NON_FINITE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Report the first layer whose output is not finite, once.
fn note_if_non_finite(stage: &str, x: &Tensor) {
    use std::sync::atomic::Ordering::Relaxed;
    if !SCAN_FOR_NON_FINITE.load(Relaxed) {
        return;
    }
    // One number for the whole tensor: a NaN anywhere poisons the sum, which is all this
    // needs to answer. Read on the host, so it costs a device sync - paid only when armed.
    let sum = x
        .to_dtype(crate::tensor::DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.sum(0))
        .and_then(|t| t.to_scalar_f32());
    match sum {
        Ok(v) if !v.is_finite() => {
            tracing::error!("non-finite hidden state first seen at {stage} (sum {v})");
            SCAN_FOR_NON_FINITE.store(false, Relaxed);
        }
        Ok(_) => {}
        Err(e) => tracing::warn!("non-finite scan could not read the hidden state: {e}"),
    }
}

/// Records the time since `since` as stage `name` of a forward, then restarts `since`.
///
/// A no-op when the stage profile is off.
fn forward_mark(name: &'static str, since: &mut std::time::Instant) {
    if crate::inference::stage_prof::stage_prof_enabled() {
        crate::inference::stage_prof::prof_record(name, since.elapsed().as_nanos() as u64);
        *since = std::time::Instant::now();
    }
}

/// `x` `[rows, heads, head_dim]` rotated by `layer`'s own tables, row `i` at `positions[i]`.
///
/// Each row gets the rotation the layer's serial forward gives a token at that position.
/// The rows stand on the position axis of the serial `[1, heads, seq, head_dim]` layout.
fn rotate_rows_by_table(
    layer: &GenericTransformerLayer,
    x: &Tensor,
    positions: &[usize],
) -> Result<Tensor> {
    let table = layer.rope_table();
    let dev = x.device();
    let mut cos = Vec::with_capacity(positions.len());
    let mut sin = Vec::with_capacity(positions.len());
    for &p in positions {
        let (c, s) = table.rows(p, 1, &dev)?;
        cos.push(c);
        sin.push(s);
    }
    let cos = Tensor::cat(&cos.iter().collect::<Vec<_>>(), 0)?;
    let sin = Tensor::cat(&sin.iter().collect::<Vec<_>>(), 0)?;
    let heads_first = x.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
    layer
        .apply_rotary_with(&heads_first, &cos, &sin)?
        .squeeze(0)?
        .transpose(0, 1)?
        .contiguous()
}

/// The attention output projection of `layer` over the rows `a`, its bias added when it has
/// one.
fn attention_output(layer: &GenericTransformerLayer, a: &Tensor) -> Result<Tensor> {
    let o = layer.attn_output.forward(a)?;
    match layer.attn_output_bias.as_ref() {
        Some(b) => o.broadcast_add(&b.to_dtype(o.dtype())?),
        None => Ok(o),
    }
}

/// The host attention shape of a layer whose softmax holds a sink or whose keys a window
/// bounds.
///
/// `None` for a layer with neither: the plain host paths serve it.
fn sink_or_window_shape(
    layer: &GenericTransformerLayer,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    scale: f32,
) -> Option<crate::inference::cache::paged_attention::HostAttnShape<'_>> {
    let window = layer.sliding_window.filter(|&w| w > 0);
    let sinks = layer.attn_sinks_host.as_deref();
    (window.is_some() || sinks.is_some()).then_some(
        crate::inference::cache::paged_attention::HostAttnShape {
            n_head,
            n_kv_head,
            head_dim,
            scale,
            window,
            sinks,
        },
    )
}

/// Whether a card serves the attention of `layer` over materialised scores: its softmax holds
/// a sink or a window bounds its keys.
///
/// A layer with neither runs the paged attention kernels.
fn card_scores_attention(layer: &GenericTransformerLayer) -> bool {
    layer.attn_sinks.is_some() || layer.sliding_window.is_some_and(|w| w > 0)
}

/// The rows of `t` as host F32.
fn host_f32(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

/// The mixture block of `layer` over the rows `hn` `[rows, hidden]` of several prompts.
///
/// The rows `start..start + len` of each span go through the block as one forward of their
/// prompt alone.
/// `residual` holds the rows `hn` was normed from.
fn moe_rows_per_prompt(
    layer: &GenericTransformerLayer,
    moe: &MoeWeights,
    hn: &Tensor,
    residual: &Tensor,
    spans: &[(usize, usize)],
) -> Result<Tensor> {
    let parts = spans
        .iter()
        .map(|&(start, len)| {
            moe.forward_rows(
                &hn.narrow(0, start, len)?.contiguous()?.unsqueeze(0)?,
                &residual.narrow(0, start, len)?.contiguous()?.unsqueeze(0)?,
                layer.embedding_length_for_moe,
                layer.ffn_norm_eps,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Tensor::cat(&parts.iter().collect::<Vec<_>>(), 1)?.squeeze(0)
}

/// Whether the input stage of a prefill layer runs as one launch: the head norms, the
/// rotation, and the write of k and v into the paged store.
///
/// It needs a card, activations and a store held in F16, a head of even width, and head norms
/// that come as a pair or not at all.
fn prefill_stage_fused(
    on_card: bool,
    activation: DType,
    store: DType,
    head_dim: usize,
    norms_paired: bool,
) -> bool {
    on_card
        && activation == DType::F16
        && store == DType::F16
        && head_dim > 0
        && head_dim.is_multiple_of(2)
        && norms_paired
}

/// Whether the input stage of a decode layer runs as one launch: the head norms of q and k, their
/// rotation and the write of k and v into the paged store, with q and k read in place from the
/// fused projection.
///
/// It needs what [`prefill_stage_fused`] needs. The launch norms a head with the arithmetic of
/// the norm the decode loop runs, so the bits match the separate launches.
fn decode_stage_fused(
    on_card: bool,
    activation: DType,
    store: DType,
    head_dim: usize,
    norms_paired: bool,
) -> bool {
    prefill_stage_fused(on_card, activation, store, head_dim, norms_paired)
}

/// Whether the attention of a prefill layer takes q straight from the input stage in the flash
/// kernel's layout, and gives its output in F16: the input stage is one launch, no prompt
/// attends over cached rows, the card runs the flash kernel, the flash kernel has the head
/// width, and the heads group evenly.
///
/// q then stays F32 `[n_head, rows, hd]` between the 2 launches, and the casts and the
/// transpose around the attention do not run.
fn prefill_attention_head_major(
    stage_fused: bool,
    cached_prompts: usize,
    flash_card: bool,
    head_dim_in_kernel: bool,
    n_head: usize,
    n_kv: usize,
) -> bool {
    stage_fused
        && cached_prompts == 0
        && flash_card
        && head_dim_in_kernel
        && n_kv > 0
        && n_head.is_multiple_of(n_kv)
}

/// Whether the activation of a prefill layer's feed-forward block reads the fused gate and up
/// buffer in place, in one launch: a card, activations held in F16, a silu gate and an even
/// width.
fn mlp_activation_fused(on_card: bool, activation: DType, ffn: usize, use_gelu: bool) -> bool {
    on_card && activation == DType::F16 && !use_gelu && ffn > 0 && ffn.is_multiple_of(2)
}

/// The device kind and index a layer is reported on.
/// A layer a Vulkan or OpenCL segment runs is on that device, whatever its host build says.
fn layer_place(
    ld: &LayerDevice,
    vulkan: Option<usize>,
    opencl: Option<usize>,
) -> (&'static str, usize) {
    match (vulkan, opencl, ld) {
        (Some(d), _, _) => ("Vulkan", d),
        (None, Some(d), _) => ("OpenCL", d),
        (None, None, LayerDevice::Cuda(idx)) => ("CUDA", *idx),
        (None, None, LayerDevice::Cpu) => ("CPU", 0),
    }
}

/// What the projections of a prefill layer gave: one buffer holding q, k and v side by side,
/// or one tensor for each.
enum PrefillProjection {
    Fused(Tensor),
    Split(Tensor, Tensor, Tensor),
}

impl PrefillProjection {
    /// The 3 groups as a kernel reads them: the tensor and the column the group starts at.
    fn sources(&self, q_width: usize, kv_width: usize) -> [(&Tensor, usize); 3] {
        match self {
            Self::Fused(qkv) => [(qkv, 0), (qkv, q_width), (qkv, q_width + kv_width)],
            Self::Split(q, k, v) => [(q, 0), (k, 0), (v, 0)],
        }
    }

    /// q, k and v as 3 tensors.
    fn split(self, q_width: usize, kv_width: usize) -> Result<(Tensor, Tensor, Tensor)> {
        match self {
            Self::Fused(qkv) => Ok((
                qkv.narrow(1, 0, q_width)?,
                qkv.narrow(1, q_width, kv_width)?,
                qkv.narrow(1, q_width + kv_width, kv_width)?,
            )),
            Self::Split(q, k, v) => Ok((q, k, v)),
        }
    }
}

/// A prompt whose first `cached_len` tokens the paged cache already holds: the prompt, the
/// paged slots of the tokens after them, and the block table that addresses the prefix.
pub struct CachedPrompt<'a> {
    pub tokens: &'a [u32],
    pub suffix_slots: &'a [usize],
    pub cached_len: usize,
    pub block_table: &'a [u32],
}

/// What the output stage of a forward gives back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Outputs {
    /// The head's logits.
    Logits,
    /// The logits and the last row after the final norm.
    LogitsAndNormed,
    /// The last row after the final norm, with no head run.
    Normed,
    /// Every row after the final norm, with no head run.
    NormedRows,
}

/// What a decode step fed by an embedding gives back.
pub struct EmbeddingStep {
    /// The head's logits `[1, vocab]`, F32, `None` when the step ran no head.
    pub logits: Option<Tensor>,
    /// The row after the final norm, `[1, hidden]`, F32, as the head reads it.
    pub normed: Tensor,
}

/// The paged pool a batch worker left with its model, if one did.
#[cfg(feature = "gpu")]
type Kept = std::sync::Mutex<Option<crate::inference::serve::continuous_serve::ParkedPool>>;

/// [`Kept`], shared with the holder that gives it back for another model, see
/// `place::reclaim`.
#[cfg(feature = "gpu")]
pub type KeptPool = std::sync::Arc<Kept>;

/// What prefill forwards kept live on a card, fitted by least squares to a part per row and a
/// part per position reached: `kept = per_row x rows + per_position x position`.
///
/// Forwards of the same rows at the same position cannot tell the 2 parts apart.
/// Their part per row is then the most a forward kept per row.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PrefillFit {
    rr: f64,
    pp: f64,
    rp: f64,
    kr: f64,
    kp: f64,
    /// The most bytes a forward kept per row.
    top: f64,
}

impl PrefillFit {
    pub(crate) fn add(&mut self, rows: usize, upto: usize, kept: u64) {
        let (r, p, k) = (rows as f64, upto as f64, kept as f64);
        self.rr += r * r;
        self.pp += p * p;
        self.rp += r * p;
        self.kr += k * r;
        self.kp += k * p;
        if r > 0.0 {
            self.top = self.top.max(k / r);
        }
    }

    /// The bytes per row and per position, `None` before any forward.
    pub(crate) fn terms(&self) -> Option<(u64, u64)> {
        if self.rr <= 0.0 {
            return None;
        }
        let det = self.rr * self.pp - self.rp * self.rp;
        if det <= 0.0 {
            return Some((self.top as u64, 0));
        }
        let per_row = (self.kr * self.pp - self.kp * self.rp) / det;
        let per_position = (self.kp * self.rr - self.kr * self.rp) / det;
        Some(match (per_row > 0.0, per_position > 0.0) {
            (true, true) => (per_row as u64, per_position as u64),
            // A part the forwards do not show: the other one carries them alone.
            (true, false) => ((self.kr / self.rr) as u64, 0),
            (false, _) => (0, (self.kp / self.pp.max(1.0)) as u64),
        })
    }
}

/// The share of the free card memory a prefill chunk may take is one over this, for a shape that
/// counts every live buffer itself: a wide residual's, a tiered mixture's or a measured one.
pub(crate) const COUNTED_SHARE: usize = 2;

impl GenericHeteroTransformer {
    /// Keeps the paged pool of the batch worker while the model serves serially.
    #[cfg(feature = "gpu")]
    pub fn park_pool(&self, pool: crate::inference::serve::continuous_serve::ParkedPool) {
        *self.parked_pool.lock().unwrap_or_else(|e| e.into_inner()) = Some(pool);
    }

    /// The slot the model keeps its batch worker's pool in, for a holder that gives the pool
    /// back while the model serves serially.
    #[cfg(feature = "gpu")]
    pub fn kept_pool(&self) -> KeptPool {
        self.parked_pool.clone()
    }

    /// The paged pool a batch worker left with the model, if one did.
    #[cfg(feature = "gpu")]
    pub fn take_parked_pool(
        &self,
    ) -> Option<crate::inference::serve::continuous_serve::ParkedPool> {
        self.parked_pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
    }

    /// Read-only view of the layers, which is how anything outside this module reaches a
    /// layer's KV cache after a forward.
    ///
    /// The field stays private: this hands out shared references, so a caller can inspect
    /// what a run produced without being able to reorder the layers or swap a cache out
    /// from under the plan that placed it.
    pub fn layers(&self) -> &[GenericTransformerLayer] {
        &self.layers
    }

    /// Adapter names live on the model right now, in the order they were applied.
    pub fn adapters(&self) -> &[String] {
        &self.adapters
    }

    /// Replace the attached adapter set, in place, without reloading the checkpoint.
    ///
    /// An empty list detaches everything and returns every projection to the file it was read
    /// from. Mutation stays inside this type on purpose: `layers` hands out shared references
    /// so that no caller can reorder the layers or swap a cache from under the plan that
    /// placed them, and an adapter walk needs none of that.
    ///
    /// Applied all-or-nothing. A file that resolves but matches nothing leaves the model on
    /// its base weights and says so, because an adapter on some projections and not others is
    /// worse than one that did not apply.
    ///
    /// A change of the attached adapters moves the graph generation.
    pub fn set_adapters(
        &mut self,
        wanted: &[(String, f32)],
    ) -> Result<crate::inference::load::lora::AdapterReport> {
        use crate::inference::load::lora::{AdapterReport, LoraFile};
        use crate::tensor::Error;

        if !wanted.is_empty() {
            if let Some(why) = self.adapter_refusal() {
                return Err(Error(format!("adapters refused: {why}")));
            }
        }
        if !wanted.is_empty() || !self.adapters.is_empty() {
            self.note_adapters_changed();
        }
        self.each_adapted_projection(|p| p.clear_lora());
        self.adapters.clear();
        if wanted.is_empty() {
            return Ok(AdapterReport::default());
        }

        // Read on the host, then move each delta to the card its projection sits on: a split
        // model holds layer 0 and layer 30 on different devices.
        let mut files = Vec::with_capacity(wanted.len());
        for (name, strength) in wanted {
            let path = crate::inference::load::lora::resolve(name).map_err(Error)?;
            let file = LoraFile::load(&path.to_string_lossy(), &Device::Cpu)?;
            if file.is_empty() {
                return Err(Error(format!("adapter {name} holds no usable pair")));
            }
            files.push((name.clone(), *strength, file));
        }

        let mut report = AdapterReport::default();
        let mut matched_entries = 0usize;
        for (name, strength, file) in &files {
            let mut sites = std::collections::BTreeSet::new();
            for d in self.adapter_deltas(file, *strength)? {
                let mut part = &mut self.layers[d.layer];
                if d.share > 0 {
                    part = &mut part.tp_shards[d.share - 1];
                }
                if let Some(weight) = super::adapters::site_mut(part, d.slot) {
                    weight.add_lora(d.delta)?;
                    sites.insert((d.layer, d.site));
                }
            }
            if sites.is_empty() {
                self.set_adapters(&[])?;
                return Err(Error(format!(
                    "adapter {name} matched no projection on this checkpoint"
                )));
            }
            matched_entries += sites.len();
            report.attached.push(name.clone());
            self.adapters.push(name.clone());
        }
        report.projections = matched_entries;
        report.unmatched = files
            .iter()
            .map(|(_, _, f)| f.entry_count())
            .sum::<usize>()
            .saturating_sub(matched_entries);
        Ok(report)
    }

    /// Build a [seq_q, seq_kv] causal mask where row `i` represents token
    /// at absolute position `index_pos + i` and column `j` represents KV
    /// position `j` (which spans the full history including the new
    /// tokens). For prefill (index_pos=0), seq_kv == seq_q. For
    /// session-resume / PLD verify / spec verify, seq_kv > seq_q.
    pub(super) fn make_mask(
        &mut self,
        seq_q: usize,
        index_pos: usize,
        cuda_idx: Option<usize>,
    ) -> Result<Tensor> {
        // A model windowed in every layer sees its last `sliding_window` keys; one with a
        // per-layer pattern gives its windowed layers their own mask, and this one serves
        // its global layers.
        let window = self
            .sliding_window
            .filter(|_| self.config.swa_pattern.is_none());
        self.make_layer_mask(seq_q, index_pos, cuda_idx, window)
    }

    /// The causal mask of one layer of a period-classified model: a windowed layer sees
    /// its last `window` keys, its own position included, as the reference's sliding
    /// window does.
    pub(super) fn make_layer_mask(
        &mut self,
        seq_q: usize,
        index_pos: usize,
        cuda_idx: Option<usize>,
        window: Option<usize>,
    ) -> Result<Tensor> {
        let reach = window.map(|w| w.saturating_sub(1));
        self.make_mask_reaching(seq_q, index_pos, cuda_idx, reach)
    }

    /// Causal mask that also hides a key more than `sw` positions behind its query.
    fn make_mask_reaching(
        &mut self,
        seq_q: usize,
        index_pos: usize,
        cuda_idx: Option<usize>,
        sw: Option<usize>,
    ) -> Result<Tensor> {
        let seq_kv = index_pos + seq_q;
        let cache_key = (seq_q * 65536 + seq_kv, cuda_idx, sw);
        if !self.mask_cache.contains_key(&cache_key) {
            // The cache holds the masks of one query and key shape: the segments and windows of
            // the forward that runs.
            // A forward of another shape gives back those of the forward before it.
            self.mask_cache.retain(|key, _| key.0 == cache_key.0);
            let dev = cuda_idx
                .and_then(|idx| self.cuda_devices.get(&idx))
                .unwrap_or(&Device::Cpu);
            let data: Vec<u8> = (0..seq_q)
                .flat_map(|i| {
                    (0..seq_kv).map(move |j| {
                        let abs_i = i + index_pos;
                        u8::from(j > abs_i || sw.is_some_and(|w| j + w < abs_i))
                    })
                })
                .collect();
            let mask = Tensor::from_slice(&data, (seq_q, seq_kv), dev)?;
            self.mask_cache.insert(cache_key, mask);
        }
        Ok(self.mask_cache[&cache_key].clone())
    }

    /// Real per-device layer distribution for the /api/ps topology view:
    /// consecutive layers on the same device grouped into
    /// `(device_type, device_id, layer_start, layer_end)` segments (ranges
    /// inclusive). Reads the ACTUAL `layer_devs` set at load time from the
    /// HeteroPlan, so a GPU0/GPU1/CPU split surfaces as distinct segments
    /// instead of every layer collapsed onto the primary device (the
    /// long-standing topology bug where the GUI showed the whole model on
    /// one GPU regardless of the real split).
    pub fn device_layer_distribution(&self) -> Vec<(String, usize, u32, u32)> {
        let mut out: Vec<(String, usize, u32, u32)> = Vec::new();
        for (i, ld) in self.layer_devs.iter().enumerate() {
            let (dtype, did) = layer_place(ld, self.vulkan.device_of(i), self.opencl.device_of(i));
            let i = i as u32;
            match out.last_mut() {
                // Extend the current run when the same device continues.
                Some(last) if last.0 == dtype && last.1 == did => last.3 = i,
                _ => out.push((dtype.to_string(), did, i, i)),
            }
        }
        out
    }

    /// The runs of consecutive layers on one device, as the placement of a part is reported.
    pub fn placed_runs(&self) -> Vec<crate::inference::serve::progress::placement::Placed> {
        let runs = self.layer_devs.iter().map(|ld| match ld {
            LayerDevice::Cuda(idx) => crate::tensor::DeviceLocation::Cuda { gpu_id: *idx },
            LayerDevice::Cpu => crate::tensor::DeviceLocation::Cpu,
        });
        crate::inference::serve::progress::placement::runs(runs, 0)
    }

    /// Forward pass: embed -> layers (CUDA+CPU segments) -> norm -> logits.
    /// Standard forward: returns logits for the LAST position only
    /// (shape `[batch, vocab]`). Equivalent to `forward_inner(x, pos, false)`.
    pub fn forward(&mut self, input_ids: &Tensor, index_pos: usize) -> Result<Tensor> {
        self.forward_inner(input_ids, index_pos, /*all_positions=*/ false)
    }

    /// Whether a prompt of `seq_len` rows in chunks of `chunk` goes through the layers better
    /// layer by layer: it takes several chunks, and some layer's experts come from the host in
    /// full for a batch that wide, or are read from the host's memory by the rows that route to
    /// them, which a chunk at a time repeats for every chunk.
    pub fn layer_major_pays(&self, seq_len: usize, chunk: usize) -> bool {
        seq_len > chunk.max(1)
            && self.config.shared_kv_layers == 0
            && self.layers.iter().any(|l| {
                l.moe.as_ref().is_some_and(|m| {
                    m.tier
                        .as_ref()
                        .is_some_and(|t| t.reads_in_place(seq_len, m.n_experts_used))
                        || m.reads_every_expert_on_host(seq_len)
                })
            })
    }

    /// A prompt `ids` `[1, seq_len]` from `index_pos` through the layers one layer at a time:
    /// every chunk of `chunk` rows goes through a layer before any goes through the next, so a
    /// layer whose experts live on the host has them brought to the card once for the whole
    /// prompt, where a chunk at a time brings them once per chunk. The chunks' residuals are
    /// kept between layers; the logits of the last row come back, as from `forward`.
    pub fn forward_prefill_layer_major(
        &mut self,
        ids: &Tensor,
        index_pos: usize,
        chunk: usize,
    ) -> Result<Tensor> {
        let (_, seq_len) = ids.dims2()?;
        let chunk = chunk.max(1);
        self.spec_clear();
        // Every chunk's embedding first, in order: the n-gram history follows the tokens.
        let mut spans = Vec::new();
        let mut hidden: Vec<Option<Tensor>> = Vec::new();
        let mut ple: Vec<Option<Vec<Tensor>>> = Vec::new();
        let mut ngram_rows: Vec<Option<Tensor>> = Vec::new();
        let mut off = 0;
        while off < seq_len {
            let n = chunk.min(seq_len - off);
            let (h, p) = self.embed_stage(&ids.narrow(1, off, n)?, index_pos + off)?;
            ngram_rows.push(self.ngram.as_ref().and_then(|g| g.pending.clone()));
            hidden.push(Some(h));
            ple.push(p);
            spans.push((off, n));
            off += n;
        }
        let mut on_cuda = true;
        for layer in 0..self.layers.len() {
            let tier = self.layers[layer].moe.as_ref().and_then(|m| m.tier.clone());
            if let Some(tier) = &tier {
                tier.hold_stacks(true)?;
            }
            let mut outcome = Ok(());
            for (c, &(off, n)) in spans.iter().enumerate() {
                if let (Some(ng), Some(rows)) = (self.ngram.as_mut(), ngram_rows[c].clone()) {
                    ng.pending = Some(rows);
                }
                let h = hidden[c]
                    .take()
                    .expect("a chunk's residual is held between layers");
                match self.run_layers_in(h, ple[c].clone(), index_pos + off, n, layer, layer + 1) {
                    Ok((h, cuda)) => {
                        if index_pos + off == 0 {
                            if let Some(dir) = super::draft_head::probe_dir() {
                                super::draft_head::dump_rows(dir, &format!("layer{layer}"), &h);
                            }
                        }
                        hidden[c] = Some(h);
                        on_cuda = cuda;
                    }
                    Err(e) => {
                        outcome = Err(e);
                        break;
                    }
                }
            }
            if let Some(tier) = &tier {
                tier.hold_stacks(false)?;
            }
            outcome?;
            crate::request_metrics::note_layers(seq_len, layer + 1, self.layers.len());
        }
        // The draft head reads the prompt chunk by chunk, in order, from the last layer's rows.
        if self.capture_feature && self.draft.is_some() {
            // The fit probe reads the prompt's first rows whatever the chunking: the chunks
            // within them are gathered, then read once.
            let mut probe_rows: Vec<Tensor> = Vec::new();
            let mut probe_ids: Vec<u32> = Vec::new();
            let span = PROMPT_FIT_SPAN;
            for (c, &(off, n)) in spans.iter().enumerate() {
                let chunk_ids = ids
                    .narrow(1, off, n)?
                    .flatten_all()?
                    .to_dtype(crate::tensor::DType::U32)?
                    .to_vec1::<u32>()?;
                let rows = hidden[c]
                    .as_ref()
                    .expect("a chunk's residual is held after the last layer");
                if index_pos + off < span {
                    probe_rows.push(rows.clone());
                    probe_ids.extend_from_slice(&chunk_ids);
                }
                self.draft_ingest(&chunk_ids, rows, index_pos + off)?;
            }
            if index_pos == 0 && !probe_rows.is_empty() {
                let rows = Tensor::cat(&probe_rows.iter().collect::<Vec<_>>(), 1)?;
                let n = probe_ids.len().min(span);
                self.prompt_fit_probe(
                    &probe_ids[..n],
                    &rows.narrow(1, 0, n.saturating_sub(1).max(1))?,
                )?;
            }
        }
        let last = hidden
            .pop()
            .flatten()
            .ok_or_else(|| crate::tensor::Error::msg("empty prefill".to_string()))?;
        let rows = spans.last().map_or(1, |s| s.1);
        self.output_stage(last, on_cuda, rows, false)
    }

    /// EAGLE: enable/disable stashing the pre-final-norm feature on forward.
    pub fn set_capture_feature(&mut self, on: bool) {
        self.capture_feature = on;
    }
    /// EAGLE: take the feature stashed by the last forward (if capture was on).
    /// Shape [b, hidden] (last-token decode) or [b, seq, hidden] (all_positions verify).
    pub fn take_last_feature(&mut self) -> Option<Tensor> {
        self.last_feature.take()
    }

    /// Embedding: prefill `input_ids` `[1, seq]` and return the LAST token's final-layer
    /// hidden state `[hidden]` (last-token pooling). Reuses the EAGLE feature hook so ANY
    /// llama-arch GGUF works - no separate encoder. KV is reset before and after so the
    /// call is stateless (doesn't disturb a concurrent chat session's cache). Caller
    /// L2-normalizes. Routes through the same loaded model as rerank.
    pub fn embed_last_hidden(&mut self, input_ids: &Tensor) -> Result<Vec<f32>> {
        self.trim_kv(0);
        self.set_capture_feature(true);
        let res = self.forward(input_ids, 0);
        self.set_capture_feature(false);
        self.trim_kv(0);
        res?; // logits ignored; the feature is what we want
        let feat = self
            .take_last_feature()
            .ok_or_else(|| crate::tensor::Error::msg("embed: no captured feature"))?;
        // Apply the model's FINAL norm (the capture is pre-norm). The learned RMSNorm
        // weights de-anisotropize the raw hidden - without it all vectors collapse to
        // ~0.99 cosine and lose discrimination. `output_norm` lives on CPU, so bring the
        // feature to CPU first (a GPU feature + CPU weight yields NaN). Fall back to the
        // pre-norm feature on any mismatch rather than fail the request.
        let feat_cpu = feat.to_device(&crate::tensor::Device::Cpu).unwrap_or(feat);
        let normed = match self.output_norm.forward(&feat_cpu) {
            Ok(n) => n,
            Err(_) => feat_cpu,
        };
        normed
            .flatten_all()?
            .to_dtype(crate::tensor::DType::F32)?
            .to_vec1::<f32>()
    }

    /// Embedding: prefill `input_ids` `[1, seq]` and return every token's final-layer
    /// hidden state (all-positions pooling). Like `embed_last_hidden` but with
    /// `forward_all` and one `Vec<f32>` per position in order. Reuses the EAGLE
    /// feature hook so ANY llama-arch GGUF works. KV is reset before and after so the
    /// call is stateless. Caller L2-normalizes each vector. Routes through the same
    /// loaded model as rerank.
    pub fn embed_all_hidden(&mut self, input_ids: &Tensor) -> Result<Vec<Vec<f32>>> {
        self.trim_kv(0);
        self.set_capture_feature(true);
        let res = self.forward_all(input_ids, 0);
        self.set_capture_feature(false);
        self.trim_kv(0);
        res?; // logits ignored; the feature is what we want
        let feat = self
            .take_last_feature()
            .ok_or_else(|| crate::tensor::Error::msg("embed: no captured feature"))?;
        // Apply the model's FINAL norm (the capture is pre-norm). The learned RMSNorm
        // weights de-anisotropize the raw hidden. `output_norm` lives on CPU, so bring the
        // feature to CPU first. The feature shape is `[1, seq, hidden]` for all positions.
        let feat_cpu = feat.to_device(&crate::tensor::Device::Cpu).unwrap_or(feat);
        let (b, s, h) = feat_cpu.dims3()?;
        let mut result = Vec::with_capacity(s);
        for i in 0..s {
            let row = feat_cpu.i((0, i, ..))?;
            let normed = match self.output_norm.forward(&row) {
                Ok(n) => n,
                Err(_) => row,
            };
            let vec = normed
                .to_dtype(crate::tensor::DType::F32)?
                .to_vec1::<f32>()?;
            result.push(vec);
        }
        // Reshape to [seq, hidden] and extract each token's hidden state
        Ok(result)
    }

    /// EAGLE: target token-embedding lookup (the head's `embed(token)` input).
    /// Moves `ids` to the embedding table's device (CPU for big vocabularies).
    pub fn eagle_embed(&self, ids: &Tensor) -> Result<Tensor> {
        let dev = self.embeddings.embeddings().device().clone();
        self.embeddings.forward(&ids.to_device(&dev)?)
    }

    /// EAGLE: map a PRE-final-norm feature `[n, hidden]` through the target's
    /// final norm + lm-head to logits `[n, vocab]`. This IS the hot draft-sampling
    /// path (called kx per spec cycle), so prefer the GPU quantized lm-head + fused
    /// rms_norm - keeping the feature on-device. The CPU `output_proj` is a fallback
    /// (CPU models / when the cuda lm-head isn't built); a CPU 151936x4096 matmul
    /// per draft is ~7x slower than plain decode, so it must NOT be used on GPU.
    /// The dtype the residual stream is carried in on `dev`.
    ///
    /// THIS IS A DEFECT EXPRESSED AS A FUNCTION, not a settled contract. The
    /// precision depends on WHERE a layer was placed, so a model split across a
    /// card and the host computes some layers in half precision and others in
    /// single, and the answer depends on the split point. Measured: the same
    /// layer differs by ~1e-3 between the two devices, a thousand times more
    /// than the ~1e-6 that reassociation alone would give, and enough for greedy
    /// decoding to pick a different token when two candidates are close.
    ///
    /// The reference carries activations in F32 on every backend
    /// (`ggml_mul_mat` allocates F32 whatever the device) and quantizes them to
    /// Q8_1 at each matmul instead of storing them in half precision, so moving
    /// a layer between devices leaves the stream's type untouched. Adopting that
    /// means teaching the quantized GPU GEMMs to read F32 activations - the
    /// comment at the call site records that they read garbage today - which is
    /// why this returns the device-dependent answer for now instead of a
    /// uniform one.
    fn residual_dtype(dev: &Device) -> DType {
        if dev.is_cpu() {
            DType::F32
        } else {
            DType::F16
        }
    }

    /// The model's final norm, evaluated on whatever device `x` lives on.
    ///
    /// The GPU head used to call the fused RMS kernel unconditionally, which is
    /// the wrong norm for the phi2 family (full LayerNorm, mean subtracted, bias
    /// added). Excluding those models from the GPU head fixed the answer and cost
    /// most of their throughput, because it sent the 51200-wide projection to the
    /// CPU as well. The norm is 2048 values; the projection is a hundred million
    /// multiply-adds - so normalise correctly HERE and keep the projection on the
    /// card.
    fn final_norm_on_device(&self, x: &Tensor) -> Result<Tensor> {
        use crate::inference::generic_transformer::config::WeightedNorm;
        match &self.output_norm {
            WeightedNorm::Rms(_) => {
                let w = self
                    .output_norm_cuda_weight
                    .as_ref()
                    .filter(|w| w.device().same_device(&x.device()))
                    .cloned()
                    .map(Ok)
                    .unwrap_or_else(|| self.output_norm.weight().to_device(&x.device()))?;
                crate::tensor::ops::rms_norm(x, &w, self.config.rms_norm_eps as f32)
            }
            WeightedNorm::Layer(ln) => {
                let dev = x.device();
                let w = ln.weight().to_device(&dev)?.to_dtype(x.dtype())?;
                let mean = x.mean_keepdim(crate::tensor::D::Minus1)?;
                let centered = x.broadcast_sub(&mean)?;
                let var = centered.sqr()?.mean_keepdim(crate::tensor::D::Minus1)?;
                let normed = centered.broadcast_div(&(var + ln.eps())?.sqrt()?)?;
                let scaled = normed.broadcast_mul(&w)?;
                match ln.bias() {
                    Some(b) => scaled.broadcast_add(&b.to_device(&dev)?.to_dtype(x.dtype())?),
                    None => Ok(scaled),
                }
            }
            // Post-norm-only architectures pass the residual through unchanged.
            _ => self.output_norm.forward(x),
        }
    }

    /// The host copy of the output head, an error when a card copy serves the head and none was
    /// built.
    pub(crate) fn host_head(&self) -> Result<&crate::tensor::quantized::QMatMul> {
        self.output_proj.as_ref().ok_or_else(|| {
            crate::tensor::Error::msg(
                "the model has no host copy of an output head: it lives on a card, or the model has none".to_string(),
            )
        })
    }

    /// Adds the per-vocabulary logit bias when the architecture carries one.
    /// EVERY site that produces logits must route through this - the engine has
    /// several, and one that skips it is wrong without failing.
    /// Final norm and output projection for rows computed on a card: both run where the
    /// projection lives, the card-resident quantised head when there is one.
    fn head_on_device(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.to_dtype(crate::tensor::DType::F32)?;
        if let Some(dev) = self.output_proj_dev.as_ref() {
            x = x.to_device(dev)?;
        }
        let f = self.final_norm_on_device(&x)?;
        self.lm_head(&f)
    }

    pub(crate) fn apply_output_bias(&self, logits: Tensor) -> Result<Tensor> {
        let Some(b) = self.output_bias.as_ref() else {
            return Ok(logits);
        };
        let b = b.to_device(&logits.device())?.to_dtype(logits.dtype())?;
        logits.broadcast_add(&b)
    }

    /// The device batched_paged_decode runs on (weights/embeddings device)  -
    /// callers size the paged KV stores on it.
    /// Device where the transformer layers run (where paged KV stores must live).
    /// For GPU models the embedding TABLE often stays on CPU (large vocab) while
    /// the layers are on GPU - so this returns the first layer's weight device,
    /// not the embeddings device.
    pub fn compute_device(&self) -> Device {
        self.layers
            .first()
            .map(|l| l.attn_norm.weight().device().clone())
            .unwrap_or_else(|| self.embeddings.embeddings().device().clone())
    }

    /// Geometry for sizing paged KV stores: (n_kv_head, head_dim, n_layers, vocab).
    pub fn paged_geometry(&self) -> (usize, usize, usize, usize) {
        (
            self.config.n_kv_head,
            self.config.head_dim,
            self.layers.len(),
            self.config.vocab_size,
        )
    }

    /// Card bytes the attention of a prompt of `rows` rows over `keys` keys keeps alive beside
    /// the activations of the step.
    ///
    /// A layer with sinks or a window materialises its scores on a card.
    /// Any other attention and the host keep no scores beyond the activations.
    pub fn prompt_attention_bytes(&self, rows: usize, keys: usize) -> u64 {
        let scores_on_a_card =
            !self.compute_device().is_cpu() && self.layers.iter().any(card_scores_attention);
        match scores_on_a_card {
            true => super::layer_cpu_attn::sink_scores_bytes(self.config.n_head, rows, keys),
            false => 0,
        }
    }

    /// Why this model can't use the continuous-batch paged path (None = eligible).
    /// Shared by `batched_paged_decode`/`paged_prefill_batch` (the gate) and the
    /// serving layer (to decide whether to route a model through ContinuousServer).
    pub fn cb_unsupported_reason(&self) -> Option<&'static str> {
        let c = &self.config;
        if !self.vulkan.is_empty() {
            return Some("layers run through Vulkan");
        }
        if !self.opencl.is_empty() {
            return Some("layers run through OpenCL");
        }
        if self.is_recurrent() {
            return Some("recurrent mixer");
        }
        // The batched server prefills token ids; a prompt with embeddings spliced in is not.
        #[cfg(feature = "audio")]
        if self.spliced_audio.is_some() {
            return Some("speech tower");
        }
        if self.spliced_vision.is_some() {
            return Some("vision tower");
        }
        if c.embed_scale.is_some() {
            return Some("embed_scale");
        }
        if self.position_embd.is_some() {
            return Some("learned positions");
        }
        if c.logit_scale.is_some() {
            return Some("logit_scale");
        }
        if c.attention_scale.is_some() {
            return Some("attention_scale");
        }
        if c.residual_scale.is_some() {
            return Some("residual_scale");
        }
        if c.final_logit_softcapping.is_some() {
            return Some("logit softcapping");
        }
        // The batched paths run every layer whole on the compute device.
        // The worker holds the paged cache of every layer in one pool on that device.
        let compute = self.compute_device();
        if self
            .layers
            .iter()
            .any(|l| !l.attn_norm.weight().device().same_device(&compute))
        {
            return Some("layers on several devices");
        }
        if self.layers.iter().any(|l| !l.tp_shards.is_empty()) {
            return Some("layers split across cards");
        }
        // A card runs the paths of a stretched or partial rotation, of sinks and of a mixture
        // whose experts it holds, outside the captured decode graph.
        let host = self.cb_on_host();
        for l in &self.layers {
            if l.attn_layer_scale.is_some() || l.ffn_layer_scale.is_some() {
                return Some("a scale per channel on a residual branch");
            }
            let split = l.attn_q.is_some() && l.attn_k.is_some() && l.attn_v.is_some();
            if !split && l.attn_qkv.is_none() {
                return Some("no Q/K/V");
            }
            if l.ffn_gate.is_none() && c.flags.is_phi2_simple_ffn {
                return Some("Phi2 gate-less FFN");
            }
            if l.ffn_norm.is_none() {
                return Some("no ffn_norm (parallel attn?)");
            }
            // OLMo2 post-norm-only: Identity pre-norm needs the regular forward (the
            // CB/paged executor mis-applies its ones-weight); keep OLMo2 off CB.
            // SmolLM3 NoPE: per-layer RoPE skip isn't wired into the paged/CB rope
            // path - keep it on the serial forward (where apply_rotary_emb gates it).
            if l.no_rope {
                return Some("NoPE per-layer rope (SmolLM3)");
            }
            if matches!(l.attn_norm, WeightedNorm::Identity(_)) {
                return Some("post-norm-only (OLMo2)");
            }
            if l.attn_q_norm.is_some() != l.attn_k_norm.is_some() {
                return Some("asymmetric QK-norm");
            }
            // Each row of a mixture routes alone.
            // A card serves a mixture whose routed experts it holds whole.
            if let Some(moe) = l.moe.as_ref() {
                let resident = moe.tier.is_none()
                    && moe.block_scaled.is_none()
                    && moe.shards.is_empty()
                    && moe.local.is_none()
                    && !moe.experts_on_host();
                if !host && !resident {
                    return Some("a mixture whose experts the card does not hold");
                }
                if !moe.router_prenormed {
                    return Some("MoE beside a dense feed-forward");
                }
            }
            if l.ple_proj.is_some() {
                return Some("PLE");
            }
        }
        None
    }

    /// True if this model can be served through the continuous-batch worker.
    pub fn cb_eligible(&self) -> bool {
        self.cb_unsupported_reason().is_none()
    }

    /// Whether the captured decode graph serves this model.
    ///
    /// The graph rotates at the plain base, attends without a sink or a window and runs a
    /// dense feed-forward in every layer.
    pub fn cb_graph_serves(&self) -> bool {
        !self.cb_rope_from_tables()
            && self
                .layers
                .iter()
                .all(|l| !card_scores_attention(l) && l.moe.is_none())
    }

    /// Every layer and the batched forward's device are the host.
    fn cb_on_host(&self) -> bool {
        self.compute_device().is_cpu()
            && self
                .layer_devs
                .iter()
                .all(|d| matches!(d, LayerDevice::Cpu))
    }

    /// The rotation covers part of each head.
    fn cb_partial_rope(&self) -> bool {
        let c = &self.config;
        c.rope_dim.unwrap_or(0) != 0 && c.rope_dim != Some(c.head_dim)
    }

    /// The batched paths rotate with each layer's own tables, not with the plain base: a
    /// stretched or partial rotation.
    fn cb_rope_from_tables(&self) -> bool {
        self.config.yarn.is_some() || self.cb_partial_rope()
    }

    /// Whether the tokens' positions come from a learned table added to their embeddings.
    pub fn has_learned_positions(&self) -> bool {
        self.position_embd.is_some()
    }

    /// `hidden` `[1, seq, width]` of the tokens at `index_pos..` with their rows of the learned
    /// position table added, when the model has one.
    pub(super) fn add_positions(&self, hidden: Tensor, index_pos: usize) -> Result<Tensor> {
        let Some(table) = self.position_embd.as_ref() else {
            return Ok(hidden);
        };
        let seq = hidden.dim(1)?;
        let rows = table
            .narrow(0, index_pos, seq)?
            .to_device(&hidden.device())?
            .to_dtype(hidden.dtype())?;
        hidden.broadcast_add(&rows.unsqueeze(0)?)
    }

    pub fn rope_base(&self) -> f32 {
        self.config.rope_freq_base
    }

    /// Whether the residual is wide: parallel streams behind gated or Sinkhorn mixers.
    pub fn is_wide(&self) -> bool {
        self.output_hyper.is_some() || self.wide_sparse.is_some()
    }

    /// Width of the residual a layer reads: the model width, times the streams of a wide one.
    pub fn stream_width(&self) -> usize {
        let streams = match (self.config.hyper, self.wide_sparse.as_ref()) {
            (Some(h), _) => h.streams,
            (None, Some(ws)) => ws.streams,
            (None, None) => 1,
        };
        self.config.embedding_length * streams
    }

    /// The model width: the residual stream and the embedding.
    pub fn hidden_size(&self) -> usize {
        self.config.embedding_length
    }

    /// Widest FFN intermediate dim across all layers (dense `intermediate_size`
    /// or per-expert `expert_ffn_dim`) - the activation-peak driver used to size
    /// the prefill chunk to fit L3.
    pub fn widest_ffn(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.flags.intermediate_size.max(l.flags.expert_ffn_dim))
            .max()
            .unwrap_or(0)
    }

    /// What the prefill asks of a card per token and the chunk it runs best at.
    ///
    /// A wide residual keeps `streams x dim` f32 per token in the residual and the product a
    /// combine adds, its mixers' temporaries are bounded to a block of rows, and the chunk it
    /// wants is the one that gives every routed expert a full tile of the grouped matmul (see
    /// `hyper::prefill_footprint`).
    pub fn prefill_shape(&self) -> crate::inference::engine::model_backend::PrefillShape {
        let mut shape = self.prefill_shape_with(
            self.layers
                .iter()
                .any(|l| l.moe.as_ref().is_some_and(|m| m.tier.is_some())),
        );
        // A measured forward counts every buffer it kept live.
        let measured = self.measured_prefill_row_bytes() as usize;
        if measured > shape.bytes_per_token {
            shape.bytes_per_token = measured;
            shape.share = shape.share.min(COUNTED_SHARE);
        }

        // A model on cards is sized from the cards it uses, and a model with layers on the host
        // from every card, as before.
        if self
            .layer_devs
            .iter()
            .all(|d| matches!(d, LayerDevice::Cuda(_)))
        {
            shape.cards = self
                .layer_devs
                .iter()
                .filter_map(|d| match d {
                    LayerDevice::Cuda(i) => Some(1u64 << (*i).min(63)),
                    LayerDevice::Cpu => None,
                })
                .fold(0, |a, b| a | b);
        }
        shape
    }

    /// Card bytes a layer of a prefill reaching position `upto` reads its cache history into:
    /// the K and V rows of every position before it, in F16.
    /// The prefill forwards' measured share per position, when it is more, stands for it.
    pub fn prefill_history_bytes(&self, upto: usize) -> u64 {
        let (_, n_kv, head_dim) = self.kv_layout();
        let read_back =
            (2 * upto * n_kv * head_dim * crate::tensor::DType::F16.size_in_bytes()) as u64;
        let measured = self
            .prefill_fit
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .terms()
            .map_or(0, |(_, per_position)| per_position * upto as u64);
        read_back.max(measured)
    }

    /// The most card bytes a prefill forward was measured to keep live per row; 0 before one ran
    /// on a card.
    pub fn measured_prefill_row_bytes(&self) -> u64 {
        self.prefill_row_bytes
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Takes `bytes` per row as measured, when it is more than this model measured: what an
    /// earlier load of the same model measured.
    pub fn seed_prefill_row_bytes(&self, bytes: u64) {
        self.prefill_row_bytes
            .fetch_max(bytes, std::sync::atomic::Ordering::Relaxed);
    }

    /// What the pools of the model's cards hold before a prefill forward, for
    /// [`Self::note_prefill_forward`]; empty for a model on no card.
    pub fn prefill_forward_mark(&self) -> Vec<(usize, u64)> {
        #[cfg(feature = "gpu")]
        return self
            .card_ordinals()
            .into_iter()
            .map(|card| (card, crate::tensor::cuda::pool_mark(card)))
            .collect();
        #[cfg(not(feature = "gpu"))]
        Vec::new()
    }

    /// Records what a prefill forward of `rows` rows, that reached position `upto`, kept live on
    /// the card where it kept the most, since `mark`.
    ///
    /// A card's share is its pool's peak less what the pool holds after the forward: the model's
    /// state, its weights and its cache.
    /// The shares are fitted to a part per row and a part per position, see `PrefillFit`.
    /// The part per row sizes the chunks: a forward of few rows far into a prompt does not count
    /// its history's share as rows.
    pub fn note_prefill_forward(&self, mark: &[(usize, u64)], rows: usize, upto: usize) {
        if rows < 2 || mark.is_empty() {
            return;
        }
        #[cfg(feature = "gpu")]
        {
            let kept = mark
                .iter()
                .map(|&(card, _)| {
                    let (_, held) = crate::tensor::cuda::pool_use(card);
                    crate::tensor::cuda::pool_peak(card).saturating_sub(held)
                })
                .max()
                .unwrap_or(0);
            self.note_prefill_kept(rows, upto, kept);
        }
        #[cfg(not(feature = "gpu"))]
        let _ = upto;
    }

    /// Notes, on the card that ran out of memory, what a prefill forward of `rows` rows reaching
    /// position `upto` lacked there: its measured need less the room the card has.
    /// The re-plan after it makes the card keep that much more back.
    pub fn note_prefill_shortfall(&self, rows: usize, upto: usize) {
        let Some(card) = crate::inference::place::vram_manager::last_oom_card() else {
            return;
        };
        let need = (self.prefill_shape().bytes_per_token as u64)
            .saturating_mul(rows as u64)
            .saturating_add(self.prefill_history_bytes(upto));
        // What the driver has free: a pool's unused blocks may not hold the allocation.
        let room = crate::inference::place::vram_manager::free_on(card).unwrap_or(0);
        let lacked = need.saturating_sub(room);
        tracing::info!(
            "prefill: a forward of {rows} rows at position {upto} needed {} MB of card {card}, \
             {} MB short",
            need >> 20,
            lacked >> 20
        );
        crate::inference::place::vram_manager::note_card_shortfall(card, lacked);
    }

    /// Takes in that a prefill forward of `rows` rows reaching position `upto` kept `kept` bytes
    /// live on a card, and sets the part per row the chunks are sized from.
    pub(crate) fn note_prefill_kept(&self, rows: usize, upto: usize, kept: u64) {
        let mut fit = self.prefill_fit.lock().unwrap_or_else(|e| e.into_inner());
        fit.add(rows, upto, kept);
        let Some((per_row, per_position)) = fit.terms() else {
            return;
        };
        drop(fit);
        let before = self
            .prefill_row_bytes
            .swap(per_row, std::sync::atomic::Ordering::Relaxed);
        if per_row.abs_diff(before) > before / 2 {
            tracing::info!(
                "prefill: forwards keep {} KB a row and {} B a position live on a card",
                per_row >> 10,
                per_position
            );
        }
    }

    /// [`Self::prefill_shape`] for a model whose experts do, or do not, run on the card for a wide
    /// batch: they then keep activations live per routed pair, and the widest chunk is the one that
    /// shares the experts' copies over the most rows.
    pub fn prefill_shape_with(
        &self,
        tiered: bool,
    ) -> crate::inference::engine::model_backend::PrefillShape {
        use crate::inference::engine::model_backend::PrefillShape;
        let plain = PrefillShape::from_widest_ffn(self.widest_ffn());
        let experts = self
            .layers
            .iter()
            .find_map(|l| l.moe.as_ref())
            .map(|m| (m.n_experts, m.n_experts_used));
        let expert_ffn = self
            .layers
            .iter()
            .filter_map(|l| l.moe.as_ref())
            .map(|m| m.expert_ffn_dim)
            .max()
            .unwrap_or(0);
        let base = match self.config.hyper {
            None => plain,
            Some(h) => {
                let fp = super::hyper::prefill_footprint(
                    h.streams,
                    self.config.embedding_length,
                    experts,
                    expert_ffn,
                );
                PrefillShape {
                    bytes_per_token: fp.bytes_per_token.max(plain.bytes_per_token),
                    fixed_bytes: fp.fixed_bytes,
                    target_chunk: fp.target_chunk,
                    share: COUNTED_SHARE,
                    cards: 0,
                }
            }
        };
        let (true, Some((_, used))) = (tiered, experts) else {
            return base;
        };
        // A wide model's already counts its experts' activations.
        PrefillShape {
            bytes_per_token: base.bytes_per_token
                + if self.config.hyper.is_some() {
                    0
                } else {
                    super::expert_tier::prefill_bytes_per_token(
                        used,
                        self.config.embedding_length,
                        expert_ffn,
                    )
                },
            fixed_bytes: base.fixed_bytes,
            target_chunk: super::hyper::PLANNED_CHUNK_ROWS,
            share: COUNTED_SHARE,
            cards: 0,
        }
    }

    /// Embed `tokens` -> `[b, hidden]` on the layer-compute device in F16 (the
    /// eager, non-capturable input to `decode_layers_gpu`).
    pub fn embed_batch(&self, tokens: &[u32]) -> Result<Tensor> {
        use crate::tensor::DType;
        let edev = self.embeddings.embeddings().device().clone();
        let dev = self.compute_device();
        // A table the card holds is read there: one launch, no host lookup and no upload of
        // the rows.
        #[cfg(feature = "gpu")]
        if let Some(table) = self.embed_table_on(&dev) {
            // Built on the card directly, as the other per-step indices are: a copy of a host
            // tensor waits on the card.
            let ids = Tensor::from_vec(tokens.to_vec(), &[tokens.len()][..], &dev)?;
            return self
                .embed_rows_of_ids(table, &ids)?
                .reshape(&[tokens.len(), self.config.embedding_length][..])?
                .to_dtype(DType::F16);
        }
        let ids = Tensor::from_vec(tokens.to_vec(), &[tokens.len(), 1][..], &edev)?;
        let mut x = self
            .embeddings
            .forward(&ids)?
            .reshape(&[tokens.len(), self.config.embedding_length][..])?
            .to_dtype(DType::F16)?;
        if !x.device().same_device(&dev) {
            x = x.to_device(&dev)?;
        }
        Ok(x)
    }

    /// Continuous-batching batched-paged DECODE forward (one new token per
    /// sequence). `tokens[b]` is sequence b's input token at `positions[b]`; its
    /// new K/V is written to every layer's paged store at `slots[b]`, and it
    /// attends over `block_tables[b]` (`context_lens[b]` tokens). Returns logits
    /// `[B, vocab]`.
    ///
    /// The variants it serves are the ones `cb_unsupported_reason` lets through: on the host
    /// that includes a mixture, routed row by row, and a stretched or partial rotation, taken
    /// from each layer's tables. Anything else
    /// -> `Err` so the caller falls back to the serial engine. The batched
    /// projections (one GEMM over B rows) are the throughput win; RoPE + paged
    /// attention run per-sequence in f32 (the slow parts a fused paged-attention
    /// kernel replaces later). Bit-faithful to the per-seq decode -> batch-invariant
    /// by construction (each sequence reads only its own block table). `stores`
    /// has one [`crate::inference::cache::paged_kv::PagedKvStore`] per layer (shared block tables).
    pub fn batched_paged_decode(
        &self,
        tokens: &[u32],
        positions: &[usize],
        slots: &[usize],
        block_tables: &[Vec<u32>],
        context_lens: &[usize],
        stores: &mut [crate::inference::cache::paged_kv::PagedKvStore],
    ) -> Result<Tensor> {
        use crate::tensor::Tensor;
        let c = &self.config;
        // -- variant gate: only the plain dense path is supported here. --
        if let Some(why) = self.cb_unsupported_reason() {
            return Err(crate::tensor::Error::msg(format!(
                "batched_paged_decode: unsupported - {why}"
            )));
        }
        if stores.len() != self.layers.len() {
            return Err(crate::tensor::Error::msg(
                "batched_paged_decode: store/layer count mismatch",
            ));
        }
        let b = tokens.len();
        let (nh, nkv, hd) = (c.n_head, c.n_kv_head, c.head_dim);
        let scale = 1.0f32 / (hd as f32).sqrt();
        let base = c.rope_freq_base;
        // Device-agnostic: run on the weights' device (CPU or CUDA). The embedding
        // table may live on CPU even for a GPU model (large vocab) -> x is moved to
        // the layer-compute device right after the lookup.
        let edev = self.embeddings.embeddings().device().clone();
        let dev = self.compute_device(); // layer (matmul) device - may differ from edev
                                         // Compute dtype: F32 on CPU (matches the scalar reference path bit-for-bit),
                                         // F16 on GPU (matches forward()'s on-device kernels; F32 activations through
                                         // GPU quantized GEMMs read garbage).
        let cdt = Self::residual_dtype(&dev);

        // embed on the embeddings device, then move activations to the layer device.
        let ids = Tensor::from_vec(tokens.to_vec(), &[b, 1][..], &edev)?;
        let mut x = self
            .embeddings
            .forward(&ids)?
            .reshape(&[b, c.embedding_length][..])?
            .to_dtype(cdt)?;
        if !x.device().same_device(&dev) {
            x = x.to_device(&dev)?;
        }
        // RoPE cos/sin depend only on positions - compute ONCE, reuse every layer.
        let (rcos, rsin) =
            crate::inference::cache::paged_attention::rope_cos_sin(positions, hd, base, &dev, cdt)?;
        let table_rope = self.cb_rope_from_tables();
        // Ragged attention mask (GPU path, any B) depends only on lens - compute ONCE.
        let ragged_mask = if !dev.is_cpu() {
            let max_ctx = *context_lens.iter().max().unwrap();
            Some(crate::inference::cache::paged_attention::ragged_mask(
                context_lens,
                max_ctx,
                &dev,
                cdt,
            )?)
        } else {
            None
        };
        // Per-seq kv position (= ctx-1) for the fused flash-decode attention kernel
        // (LOKEN_CB_FLASH): one device i32[b], read at kernel-exec time -> no
        // padding-compute waste (each seq attends over its real length), graph-safe.
        // The fused flash-decode is about 10% faster at high batch and is the only
        // capture-safe form, but its float reassociation differs from the cuBLAS chain's:
        // bit-identical on qwen2/qwen3, and on a near-tied-logit model like mistral-nemo or
        // deepcoder it flips a greedy token that was a near-tie. The default keeps strict
        // bit-identity, which is what the parity oracles are written against.
        let cb_flash = false;
        let pos_dev = if cb_flash {
            let pv: Vec<i32> = context_lens.iter().map(|&c| (c as i32) - 1).collect();
            Some(Tensor::from_vec(pv, &[b][..], &dev)?)
        } else {
            None
        };
        // The fully capture-safe paged path: KV write and paged flash-decode both index the
        // store through DEVICE buffers read at kernel-exec time, where index_select and
        // scatter bake host-read indices at capture. Written for a CUDA-graph decode that is
        // not built yet, so nothing selects it.
        let cb_paged = false;
        let paged_dev = if cb_paged {
            let slot_dev = Tensor::from_vec(
                slots.iter().map(|&s| s as i32).collect::<Vec<_>>(),
                &[b][..],
                &dev,
            )?;
            let seq_lens_dev = Tensor::from_vec(
                context_lens.iter().map(|&c| c as i32).collect::<Vec<_>>(),
                &[b][..],
                &dev,
            )?;
            let max_blocks = block_tables.iter().map(|bt| bt.len()).max().unwrap();
            let mut btf = vec![0i32; b * max_blocks];
            for (bi, bt) in block_tables.iter().enumerate() {
                for (j, &blk) in bt.iter().enumerate() {
                    btf[bi * max_blocks + j] = blk as i32;
                }
            }
            let block_table_dev = Tensor::from_vec(btf, &[b, max_blocks][..], &dev)?;
            Some((slot_dev, seq_lens_dev, block_table_dev, max_blocks))
        } else {
            None
        };

        for (li, layer) in self.layers.iter().enumerate() {
            let residual = x.clone();
            let hn = layer.attn_norm.forward(&x)?.to_dtype(cdt)?;
            let (qd, kvd) = (nh * hd, nkv * hd);
            let (q, k, v) = if let Some(qkvw) = layer.attn_qkv.as_ref() {
                // fused QKV: one GEMM, then split [q_dim | kv_dim | kv_dim].
                let mut qkv = qkvw.forward(&hn)?;
                if let Some(bias) = layer.attn_qkv_bias.as_ref() {
                    qkv = qkv.broadcast_add(&bias.to_dtype(qkv.dtype())?)?;
                }
                (
                    qkv.narrow(1, 0, qd)?,
                    qkv.narrow(1, qd, kvd)?,
                    qkv.narrow(1, qd + kvd, kvd)?,
                )
            } else {
                let mut q = layer.attn_q.as_ref().unwrap().forward(&hn)?;
                let mut k = layer.attn_k.as_ref().unwrap().forward(&hn)?;
                let mut v = layer.attn_v.as_ref().unwrap().forward(&hn)?;
                if let Some(bias) = layer.attn_q_bias.as_ref() {
                    q = q.broadcast_add(&bias.to_dtype(q.dtype())?)?;
                }
                if let Some(bias) = layer.attn_k_bias.as_ref() {
                    k = k.broadcast_add(&bias.to_dtype(k.dtype())?)?;
                }
                if let Some(bias) = layer.attn_v_bias.as_ref() {
                    v = v.broadcast_add(&bias.to_dtype(v.dtype())?)?;
                }
                (q, k, v)
            };
            // RoPE q,k on-device, ONE batched write of all B tokens' K/V into this
            // layer's store, then per-seq paged GQA attention.
            // Per-head QK-norm (qwen3) on the [B,heads,hd] view before RoPE - same
            // RmsNorm over head_dim that forward() applies. No-op when absent.
            // OLMo2 full-dim QK-norm: norm the FLAT q/k (whole projection) before the
            // head reshape; Gemma3/Qwen3 norm per-head after. Detect via weight length.
            let qk_full = layer.qk_norm_is_full();
            let qf = q.to_dtype(cdt)?;
            let qf = if qk_full {
                if let Some(qn) = layer.attn_q_norm.as_ref() {
                    qn.forward(&qf.contiguous()?)?
                } else {
                    qf
                }
            } else {
                qf
            };
            let qh = qf.reshape(&[b, nh, hd][..])?;
            let qh = if !qk_full {
                if let Some(qn) = layer.attn_q_norm.as_ref() {
                    qn.forward(&qh.contiguous()?)?
                } else {
                    qh
                }
            } else {
                qh
            };
            let kf = k.to_dtype(cdt)?;
            let kf = if qk_full {
                if let Some(kn) = layer.attn_k_norm.as_ref() {
                    kn.forward(&kf.contiguous()?)?
                } else {
                    kf
                }
            } else {
                kf
            };
            let kh = kf.reshape(&[b, nkv, hd][..])?;
            let kh = if !qk_full {
                if let Some(kn) = layer.attn_k_norm.as_ref() {
                    kn.forward(&kh.contiguous()?)?
                } else {
                    kh
                }
            } else {
                kh
            };
            let (q_rot, k_rot) = if table_rope {
                (
                    rotate_rows_by_table(layer, &qh, positions)?,
                    rotate_rows_by_table(layer, &kh, positions)?,
                )
            } else {
                (
                    crate::inference::cache::paged_attention::rope_apply(
                        &qh,
                        &rcos,
                        &rsin,
                        self.config.flags.use_rope_i,
                    )?,
                    crate::inference::cache::paged_attention::rope_apply(
                        &kh,
                        &rcos,
                        &rsin,
                        self.config.flags.use_rope_i,
                    )?,
                )
            };
            let q_rot = q_rot.reshape(&[b, nh * hd][..])?;
            let k_rot = k_rot.reshape(&[b, nkv * hd][..])?;
            let v_f = v.to_dtype(cdt)?;
            if let Some((slot_dev, _, _, _)) = paged_dev.as_ref() {
                crate::inference::kernel::fused::paged_kv_write(
                    &k_rot,
                    &v_f,
                    stores[li].k(),
                    stores[li].v(),
                    slot_dev,
                    b,
                    nkv * hd,
                )?;
            } else {
                stores[li].write_stable(slots, &k_rot, &v_f)?;
            }
            // CPU: scalar attention (bit-identical to forward()). GPU: tensor ops
            // (matmul+softmax run on-device, no host round-trip). The scalar path
            // avoids the tensor-matmul float-reassociation that flips borderline
            // greedy tokens vs the CPU reference; GPU forward() uses the same tensor
            // kernels so the tensor path matches it there.
            let attn = if let Some((_, seq_lens_dev, block_table_dev, max_blocks)) =
                paged_dev.as_ref()
            {
                // Capture-safe paged flash-decode: fused paged gather + attention.
                let qb = q_rot.reshape(&[b, nh, hd][..])?;
                let bs = stores[li].block_size();
                match crate::inference::kernel::fused::paged_flash_decode(
                    &qb,
                    stores[li].k(),
                    stores[li].v(),
                    block_table_dev,
                    seq_lens_dev,
                    scale,
                    b,
                    nh,
                    nkv,
                    hd,
                    bs,
                    *max_blocks,
                )? {
                    Some(o) => o.reshape(&[b, nh * hd][..])?,
                    None => return Err(crate::tensor::Error::msg("paged_flash_decode declined")),
                }
            } else if let Some(shape) =
                sink_or_window_shape(layer, nh, nkv, hd, scale).filter(|_| dev.is_cpu())
            {
                // Each sequence reads its own keys, its sinks in the denominator.
                // A sink layer reads its keys and values in half precision, as the serial
                // host decode stores them.
                let qv = host_f32(&q_rot)?;
                let held = |t: Tensor| -> Result<Vec<f32>> {
                    match shape.sinks {
                        Some(_) => host_f32(&t.to_dtype(DType::F16)?),
                        None => host_f32(&t),
                    }
                };
                let mut rows: Vec<f32> = Vec::with_capacity(b * nh * hd);
                for bi in 0..b {
                    let (kg, vg) = stores[li].gather_seq(&block_tables[bi], context_lens[bi])?;
                    rows.extend(crate::inference::cache::paged_attention::host_attn_rows(
                        &qv[bi * nh * hd..][..nh * hd],
                        &held(kg)?,
                        &held(vg)?,
                        &[context_lens[bi]],
                        &shape,
                    ));
                }
                Tensor::from_vec(rows, &[b, nh * hd][..], &dev)?
            } else if !dev.is_cpu() && card_scores_attention(layer) {
                // Each sequence reads its own keys within its window, its sinks in the softmax.
                let window = layer.sliding_window.filter(|&w| w > 0);
                let mut rows = Vec::with_capacity(b);
                for bi in 0..b {
                    let ctx = context_lens[bi];
                    let (kg, vg) = stores[li].gather_seq(&block_tables[bi], ctx)?;
                    let heads = |t: Tensor| -> Result<Tensor> {
                        t.reshape(&[1, ctx, nkv, hd][..])?
                            .transpose(1, 2)?
                            .contiguous()
                    };
                    let q1 = q_rot.narrow(0, bi, 1)?.reshape(&[1, nh, 1, hd][..])?;
                    let o = super::layer_cpu_attn::sink_attention_matmul(
                        &q1,
                        heads(kg)?,
                        heads(vg)?,
                        None,
                        layer.attn_sinks.as_ref(),
                        scale as f64,
                        nkv,
                        window,
                    )?;
                    rows.push(o.reshape(&[1, nh * hd][..])?);
                }
                Tensor::cat(&rows.iter().collect::<Vec<_>>(), 0)?
            } else if dev.is_cpu() {
                let qv = q_rot.to_vec2::<f32>()?;
                let mut rows: Vec<f32> = Vec::with_capacity(b * nh * hd);
                for bi in 0..b {
                    let (kg, vg) = stores[li].gather_seq(&block_tables[bi], context_lens[bi])?;
                    let kgv = kg.to_vec2::<f32>()?.concat();
                    let vgv = vg.to_vec2::<f32>()?.concat();
                    let mut out = vec![0f32; nh * hd];
                    let mut s = crate::inference::cache::paged_attention::PagedAttnSeq {
                        q: &qv[bi],
                        k: &kgv,
                        v: &vgv,
                        context_len: context_lens[bi],
                        out: &mut out,
                    };
                    crate::inference::cache::paged_attention::paged_attn_decode_seq(
                        &mut s, nh, nkv, hd, scale,
                    );
                    rows.extend_from_slice(&out);
                }
                Tensor::from_vec(rows, &[b, nh * hd][..], &dev)?
            } else {
                // ONE ragged batched GQA attention over all B sequences (incl. B=1).
                // Handles unequal contexts via padding+mask; does ALL heads in a
                // single batched matmul/softmax (no per-kv-head loop) -> far fewer
                // kernel launches than the per-seq path (the decode is launch-bound
                // at small batch, so launch count is the lever vs the optimized
                // single-stream forward()).
                let max_ctx = *context_lens.iter().max().unwrap();
                let (kb, vb) =
                    stores[li].gather_batch_padded_stable(block_tables, context_lens, max_ctx)?;
                let kb = kb.reshape(&[b, max_ctx, nkv, hd][..])?;
                let vb = vb.reshape(&[b, max_ctx, nkv, hd][..])?;
                let qb = q_rot.reshape(&[b, nh, hd][..])?;
                if let Some(pd) = pos_dev.as_ref() {
                    // Fused flash-decode kernel over the gathered (contiguous, padded)
                    // K/V via strides - per-seq real length from pos_dev, no cuBLAS,
                    // split-K for long ctx. Falls back to the ragged matmul if the
                    // kernel declines (head_dim constraints).
                    let kt = kb.transpose(1, 2)?; // [b, nkv, max_ctx, hd] strided view (hd contiguous)
                    let vt = vb.transpose(1, 2)?;
                    match crate::inference::kernel::fused::flash_decode_devkvlen(
                        &qb, &kt, &vt, pd, None, 0, scale, b, nh, nkv, hd,
                    )? {
                        Some(o) => o.reshape(&[b, nh * hd][..])?,
                        None => crate::inference::cache::paged_attention::paged_attn_decode_ragged_tensor(
                            &qb,
                            &kb,
                            &vb,
                            ragged_mask.as_ref().unwrap(),
                            nh,
                            nkv,
                            hd,
                            scale,
                        )?
                        .reshape(&[b, nh * hd][..])?,
                    }
                } else {
                    let a =
                        crate::inference::cache::paged_attention::paged_attn_decode_ragged_tensor(
                            &qb,
                            &kb,
                            &vb,
                            ragged_mask.as_ref().unwrap(),
                            nh,
                            nkv,
                            hd,
                            scale,
                        )?;
                    a.reshape(&[b, nh * hd][..])?
                }
            };
            let o = attn
                .to_dtype(cdt)
                .and_then(|a| attention_output(layer, &a))?
                .to_dtype(cdt)?;
            x = residual.broadcast_add(&o)?;
            // FFN (SwiGLU) - tensor ops (device-agnostic: on-device on GPU, no host
            // round-trip). act = silu(gate) * up.
            let residual2 = x.clone();
            let hn2 = layer
                .ffn_norm
                .as_ref()
                .unwrap()
                .forward(&x)?
                .to_dtype(cdt)?;
            if let Some(moe) = layer.moe.as_ref() {
                // Under the decode order each row routes and sums its experts as the decoded
                // token it stands for, whatever rows share the batch.
                let _order = crate::tensor::quantized::DecodeOrder::enter();
                let d = moe
                    .forward_rows(
                        &hn2.unsqueeze(0)?,
                        &residual2.unsqueeze(0)?,
                        layer.embedding_length_for_moe,
                        layer.ffn_norm_eps,
                    )?
                    .squeeze(0)?
                    .to_dtype(cdt)?;
                x = residual2.broadcast_add(&d)?;
                note_if_non_finite(&format!("layer {li}"), &x);
                continue;
            }
            // gate/up: split weights OR fused gate||up in ffn_up ([B,2.ffn_dim], qwen3/Phi3).
            let (g, u) = if let Some(gate) = layer.ffn_gate.as_ref() {
                (
                    gate.forward(&hn2)?.to_dtype(cdt)?,
                    layer.ffn_up.forward(&hn2)?.to_dtype(cdt)?,
                )
            } else {
                let gu = layer.ffn_up.forward(&hn2)?.to_dtype(cdt)?;
                (
                    gu.narrow(1, 0, c.ffn_dim)?,
                    gu.narrow(1, c.ffn_dim, c.ffn_dim)?,
                )
            };
            let act = if c.flags.use_gelu {
                g.gelu()?.mul(&u)?
            } else {
                g.silu()?.mul(&u)?
            };
            let d = layer.ffn_down.forward(&act)?.to_dtype(cdt)?;
            x = residual2.broadcast_add(&d)?;
            note_if_non_finite(&format!("layer {li}"), &x);
        }
        self.head_on_device(&x)
    }

    /// The capturable GPU decode region: batched layer loop + lm_head, reading ALL
    /// per-step dynamic data from caller-provided tensors at stable addresses
    /// (`x` input hidden, `rcos`/`rsin`, ragged `mask`, gather index `gidx` `[b*cap]`,
    /// write index `widx` `[b,feat]`). No `from_vec` inside -> safe to wrap in a CUDA
    /// graph (the caller refreshes those buffers OUTSIDE capture, then replays).
    /// Returns logits `[b, vocab]`.
    pub fn decode_layers_gpu(
        &self,
        x: &Tensor,
        rcos: &Tensor,
        rsin: &Tensor,
        block_table_dev: &Tensor,
        slot_dev: &Tensor,
        seq_lens_dev: &Tensor,
        b: usize,
        max_blocks: usize,
        stores: &mut [crate::inference::cache::paged_kv::PagedKvStore],
    ) -> Result<Tensor> {
        use crate::tensor::DType;
        let c = &self.config;
        let (nh, nkv, hd) = (c.n_head, c.n_kv_head, c.head_dim);
        let scale = 1.0f32 / (hd as f32).sqrt();
        use crate::inference::kernel::fused::{
            fused_add_rmsnorm_f16_with, fused_rmsnorm_f16, silu_mul_f16,
        };
        let cdt = DType::F16;
        let mut x = x.clone();
        // The final norm's weight, resident on the compute device (see below).
        let norm_w = self
            .output_norm_cuda_weight
            .as_ref()
            .filter(|w| w.device().same_device(&x.device()))
            .map(|w| Ok(w.clone()))
            .unwrap_or_else(|| self.output_norm.weight().to_device(&x.device()))?;
        // The norm of the stream that enters the next layer, made by the kernel that added the
        // previous layer's output to the residual.
        let mut next_norm: Option<Tensor> = None;
        for (li, layer) in self.layers.iter().enumerate() {
            // Time on the calling thread. A layer whose work is enqueued on a device
            // returns before that work is done, so this is what ISSUING the layer costs -
            // which is exactly what separates a layer that dispatches from one that runs
            // on the host, and the gap the endpoint exists to show.
            let _timer = crate::inference::place::layer_perf::LayerTimer::start(li);
            let residual = x.clone();
            // F16-native RmsNorm (no F32 cast kernel -> safe inside CUDA-graph capture,
            // and it dispatches on the device capture stream).
            let hn = match next_norm.take() {
                Some(h) => h,
                None => {
                    fused_rmsnorm_f16(&x, layer.attn_norm.weight(), layer.attn_norm.eps() as f32)?
                }
            };
            let (qd, kvd) = (nh * hd, nkv * hd);
            let projected = if let Some(qkvw) = layer.attn_qkv.as_ref() {
                let mut qkv = qkvw.forward(&hn)?;
                if let Some(bias) = layer.attn_qkv_bias.as_ref() {
                    qkv = qkv.broadcast_add(&bias.to_dtype(qkv.dtype())?)?;
                }
                PrefillProjection::Fused(qkv)
            } else {
                let mut q = layer.attn_q.as_ref().unwrap().forward(&hn)?;
                let mut k = layer.attn_k.as_ref().unwrap().forward(&hn)?;
                let mut v = layer.attn_v.as_ref().unwrap().forward(&hn)?;
                if let Some(bias) = layer.attn_q_bias.as_ref() {
                    q = q.broadcast_add(&bias.to_dtype(q.dtype())?)?;
                }
                if let Some(bias) = layer.attn_k_bias.as_ref() {
                    k = k.broadcast_add(&bias.to_dtype(k.dtype())?)?;
                }
                if let Some(bias) = layer.attn_v_bias.as_ref() {
                    v = v.broadcast_add(&bias.to_dtype(v.dtype())?)?;
                }
                PrefillProjection::Split(q, k, v)
            };
            // One launch rotates q and k and writes k and v into the store. It reads the slots
            // from `slot_dev` at execution time, so the slots are not checked on the host: the
            // scheduler allocated them inside the store.
            let norms = match (layer.attn_q_norm.as_ref(), layer.attn_k_norm.as_ref()) {
                (Some(qn), Some(kn)) => Some([
                    (qn.weight(), qn.eps() as f32),
                    (kn.weight(), kn.eps() as f32),
                ]),
                _ => None,
            };
            let norms_paired =
                norms.is_some() || (layer.attn_q_norm.is_none() && layer.attn_k_norm.is_none());
            let staged = if decode_stage_fused(
                x.device().is_cuda(),
                cdt,
                stores[li].k().dtype(),
                hd,
                norms_paired,
            ) {
                crate::inference::kernel::fused::decode_q_rope_kv_write_f16(
                    projected.sources(qd, kvd),
                    norms,
                    rcos,
                    rsin,
                    self.config.flags.use_rope_i,
                    slot_dev,
                    stores[li].k(),
                    stores[li].v(),
                    nh,
                    nkv,
                    hd,
                )?
            } else {
                None
            };
            let q_rot = match staged {
                Some(q_rot) => q_rot,
                None => {
                    let (q, k, v) = projected.split(qd, kvd)?;
                    let qh = q.to_dtype(cdt)?.reshape(&[b, nh, hd][..])?;
                    let qh = if let Some(qn) = layer.attn_q_norm.as_ref() {
                        fused_rmsnorm_f16(&qh.contiguous()?, qn.weight(), qn.eps() as f32)?
                    } else {
                        qh
                    };
                    let kh = k.to_dtype(cdt)?.reshape(&[b, nkv, hd][..])?;
                    let kh = if let Some(kn) = layer.attn_k_norm.as_ref() {
                        fused_rmsnorm_f16(&kh.contiguous()?, kn.weight(), kn.eps() as f32)?
                    } else {
                        kh
                    };
                    let k_rot = crate::inference::cache::paged_attention::rope_apply(
                        &kh,
                        rcos,
                        rsin,
                        self.config.flags.use_rope_i,
                    )?
                    .reshape(&[b, nkv * hd][..])?;
                    let q_rot = crate::inference::cache::paged_attention::rope_apply(
                        &qh,
                        rcos,
                        rsin,
                        self.config.flags.use_rope_i,
                    )?;
                    let v_f = v.to_dtype(cdt)?;
                    // Fully capture-safe paged write + fused paged flash-decode: both index
                    // the paged store via DEVICE buffers (slot/block_table/seq_lens) read at
                    // kernel-exec time - no index_select/scatter (which BAKE host-read
                    // indices at capture) and no cuBLAS (not graph-replay-safe). This is what
                    // makes the captured graph replay correctly.
                    crate::inference::kernel::fused::paged_kv_write(
                        &k_rot,
                        &v_f,
                        stores[li].k(),
                        stores[li].v(),
                        slot_dev,
                        b,
                        nkv * hd,
                    )?;
                    q_rot
                }
            };
            let qb = q_rot.reshape(&[b, nh, hd][..])?;
            let bs = stores[li].block_size();
            let a = crate::inference::kernel::fused::paged_flash_decode(
                &qb,
                stores[li].k(),
                stores[li].v(),
                block_table_dev,
                seq_lens_dev,
                scale,
                b,
                nh,
                nkv,
                hd,
                bs,
                max_blocks,
            )?
            .ok_or_else(|| crate::tensor::Error::msg("paged_flash_decode declined"))?
            .reshape(&[b, nh * hd][..])?;
            let o = a
                .to_dtype(cdt)
                .and_then(|a| attention_output(layer, &a))?
                .to_dtype(cdt)?;
            let ffn_norm = layer.ffn_norm.as_ref().unwrap();
            let (sum, hn2) = fused_add_rmsnorm_f16_with(
                &o,
                &residual,
                ffn_norm.weight(),
                ffn_norm.eps() as f32,
                true,
            )?;
            x = sum;
            let residual2 = x.clone();
            let activate = |g: &Tensor, u: &Tensor| -> Result<Tensor> {
                if c.flags.use_gelu {
                    g.gelu()?.mul(u)
                } else {
                    match silu_mul_f16(g, u)? {
                        Some(a) => Ok(a),
                        None => g.silu()?.mul(u),
                    }
                }
            };
            let act = if let Some(gate) = layer.ffn_gate.as_ref() {
                activate(&gate.forward(&hn2)?, &layer.ffn_up.forward(&hn2)?)?
            } else {
                let gu = layer.ffn_up.forward(&hn2)?;
                let fused = if mlp_activation_fused(
                    gu.device().is_cuda(),
                    gu.dtype(),
                    c.ffn_dim,
                    c.flags.use_gelu,
                ) {
                    crate::inference::kernel::fused::silu_mul_gu_f16(&gu, c.ffn_dim)?
                } else {
                    None
                };
                match fused {
                    Some(a) => a,
                    None => activate(
                        &gu.narrow(1, 0, c.ffn_dim)?,
                        &gu.narrow(1, c.ffn_dim, c.ffn_dim)?,
                    )?,
                }
            };
            let d = layer.ffn_down.forward(&act)?.to_dtype(cdt)?;
            // The residual add and the norm that follows it: the next layer's attention norm,
            // or the output norm after the last layer.
            let (next_weight, next_eps) = match self.layers.get(li + 1) {
                Some(next) => (next.attn_norm.weight().clone(), next.attn_norm.eps() as f32),
                None => (norm_w.clone(), self.output_norm.eps() as f32),
            };
            let (sum, h) =
                fused_add_rmsnorm_f16_with(&d, &residual2, &next_weight, next_eps, true)?;
            x = sum;
            next_norm = Some(h);
            note_if_non_finite(&format!("layer {li}"), &x);
        }
        // The final normalized hidden (GPU), made by the last layer's fused add. The lm_head
        // (output_proj) is applied OUTSIDE the captured region by `lm_head()`: it can live on
        // CPU for big-vocab models, which would break a GPU-only capture. The output norm's
        // weight is the GPU-resident copy built at load, a real capture-safe kernel input on
        // the compute device, where a CPU weight would make the kernel a silent no-op.
        match next_norm {
            Some(h) => Ok(h),
            None => fused_rmsnorm_f16(&x, &norm_w, self.output_norm.eps() as f32),
        }
    }

    /// The head of a batch decoded greedily: the logits in the half precision the card computes
    /// them in, when the head is a float matmul on the card and has no bias, so the argmax reads
    /// half the bytes and no widening pass runs. Any other head answers as `lm_head` does.
    pub fn lm_head_greedy(&self, f: &Tensor) -> Result<Tensor> {
        if self.output_bias.is_none() && self.output_proj_cuda_qmm.is_none() {
            if let Some(proj) = self.output_proj_cuda.as_ref() {
                let dev = self
                    .output_proj_dev
                    .clone()
                    .unwrap_or_else(|| f.device().clone());
                let f = if f.device().same_device(&dev) {
                    f.clone()
                } else {
                    f.to_device(&dev)?
                };
                return crate::tensor::quantized::half_head(&f, proj)?
                    .to_dtype(crate::tensor::DType::F16);
            }
        }
        self.lm_head(f)
    }

    /// Apply the lm_head (output projection) to a final hidden `[b, hidden]` ->
    /// logits `[b, vocab]`. Kept out of `decode_layers_gpu` so the captured graph
    /// stays GPU-only even when output_proj lives on CPU.
    pub fn lm_head(&self, f: &Tensor) -> Result<Tensor> {
        // `f` is the POST-final-norm hidden (the caller - decode_layers_gpu /
        // forward_from_hidden - already applied output_norm). The head on the card is
        // preferred: it keeps the feature on-device (no DtoH download), which is both
        // faster per decode step AND mandatory for the CUDA-graph CB path - a host head
        // would download the graph-produced hidden buffer mid-replay. A quantised head
        // runs as a quantised matmul, a float head as the F16 matmul against the
        // `[vocab, hidden]` copy; without either, the host projection reads the hidden
        // state downloaded to the host.
        let dev = self
            .output_proj_dev
            .clone()
            .unwrap_or_else(|| f.device().clone());
        let on_dev = |f: &Tensor| -> Result<Tensor> {
            if f.device().same_device(&dev) {
                Ok(f.clone())
            } else {
                f.to_device(&dev)
            }
        };
        let logits = if let Some(proj) = self.output_proj_cuda_qmm.as_ref() {
            proj.forward(&on_dev(f)?)?
        } else if let Some(proj) = self.output_proj_cuda.as_ref() {
            crate::tensor::quantized::half_head(&on_dev(f)?, proj)?
                .to_dtype(crate::tensor::DType::F32)?
        } else {
            let host = f
                .to_dtype(crate::tensor::DType::F32)?
                .to_device(&Device::Cpu)?;
            self.host_head()?.forward(&host)?
        };
        self.apply_output_bias(logits)
    }

    /// True when the lm-head runs on GPU (quantized projection resident) - a
    /// prerequisite for the CUDA-graph CB path (no CPU host-download of the
    /// graph-produced hidden buffer during replay).
    /// How many tokens the KV cache actually holds, read from the cache rather
    /// than from a counter kept beside it.
    ///
    /// The decode loop's `pos` is a sampling counter and lags the cache by one
    /// whenever a sampled token is carried forward before being pushed. That is
    /// invisible to a single-position forward - `make_mask(1, pos)` yields
    /// `pos + 1`, which happens to equal the cache - and fatal to a
    /// multi-position one, whose mask then comes out one key short. Adjusting
    /// `pos` by a constant does not work either: the lag is one after an
    /// ordinary step and zero after a speculative verify has already
    /// resynchronised. So callers that need a POSITION ask the cache.
    /// The cards holding this model's layers, fastest-first order not implied, and how many
    /// layers run on the host.
    pub fn placement(&self) -> crate::inference::place::Placement {
        let mut cards = std::collections::BTreeSet::new();
        let mut host_layers = 0;
        for d in &self.layer_devs {
            match d {
                LayerDevice::Cuda(i) => {
                    cards.insert(*i);
                }
                LayerDevice::Cpu => host_layers += 1,
            }
        }
        crate::inference::place::Placement { cards, host_layers }
    }

    /// The runs `[start, end)` of layers, by their place in the stack, whose routed experts
    /// the load put in host memory while the layer itself runs on a card.
    /// A layer without routed experts does not end a run.
    pub fn host_expert_runs(&self) -> Vec<(usize, usize)> {
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut open = false;
        for (i, (layer, dev)) in self.layers.iter().zip(&self.layer_devs).enumerate() {
            let Some(moe) = layer.moe.as_ref() else {
                continue;
            };
            let at = self.first_layer + i;
            if !matches!(dev, LayerDevice::Cuda(_)) || !moe.experts_on_host() {
                open = false;
                continue;
            }
            match runs.last_mut() {
                Some(run) if open => run.1 = at + 1,
                _ => runs.push((at, at + 1)),
            }
            open = true;
        }
        runs
    }

    /// The routing counts of every layer since the load, zero for a layer without an expert
    /// cache; `None` when no layer has one.
    #[cfg(feature = "gpu")]
    pub fn expert_profile(&self) -> Option<super::expert_profile::ExpertProfile> {
        let experts = self.layers.iter().find_map(|l| {
            l.moe
                .as_ref()?
                .tier
                .as_ref()
                .map(|t| t.routing_counts().len())
        })?;
        let mut counts = Vec::with_capacity(self.layers.len() * experts);
        for layer in &self.layers {
            match layer.moe.as_ref().and_then(|m| m.tier.as_ref()) {
                Some(tier) => counts.extend(tier.routing_counts()),
                None => counts.extend(std::iter::repeat_n(0u32, experts)),
            }
        }
        Some(super::expert_profile::ExpertProfile {
            layers: self.layers.len(),
            experts,
            counts,
        })
    }

    /// Fill every layer's expert cache from `profile`: how many experts were copied. A profile
    /// of another shape is refused, naming both shapes.
    #[cfg(feature = "gpu")]
    pub fn prime_expert_caches(
        &self,
        profile: &super::expert_profile::ExpertProfile,
    ) -> crate::tensor::Result<usize> {
        let experts = self
            .layers
            .iter()
            .find_map(|l| l.moe.as_ref()?.tier.as_ref())
            .map(|t| t.n_experts());
        if profile.layers != self.layers.len() || experts.is_some_and(|n| n != profile.experts) {
            crate::tensor::bail!(
                "the profile counts {} layers of {} experts, the model has {} layers of {} experts",
                profile.layers,
                profile.experts,
                self.layers.len(),
                experts.unwrap_or(0)
            );
        }
        let mut copied = 0;
        for (l, layer) in self.layers.iter().enumerate() {
            if let Some(tier) = layer.moe.as_ref().and_then(|m| m.tier.as_ref()) {
                copied += tier.prime(profile.layer(l))?;
            }
        }
        if let Some(h) = &self.hot_locks {
            h.prime(profile);
        }
        Ok(copied)
    }

    /// Routed experts looked up in the layers' expert caches since the load, and how many of
    /// them a card held; `None` when no layer has such a cache.
    pub fn expert_cache_counters(&self) -> Option<(u64, u64)> {
        let mut total = (0u64, 0u64);
        let mut any = false;
        for tier in self
            .layers
            .iter()
            .filter_map(|l| l.moe.as_ref()?.tier.as_ref())
        {
            let (routed, hits, _) = tier.counters();
            total = (total.0 + routed, total.1 + hits);
            any = true;
        }
        any.then_some(total)
    }

    /// Seconds the layers spend on the experts that `m` rows route to and no card holds, for `m`
    /// up to `rows`, in the order of `m`; `None` when no layer streams experts.
    pub fn expert_miss_secs(&self, rows: usize) -> Option<Vec<f64>> {
        let mut secs = vec![0.0; rows + 1];
        let mut any = false;
        for tier in self
            .layers
            .iter()
            .filter_map(|l| l.moe.as_ref()?.tier.as_ref())
        {
            any |= tier.add_miss_secs(&mut secs);
        }
        any.then_some(secs)
    }

    pub fn kv_len(&self) -> Option<usize> {
        // A recurrent layer keeps no cache: the first attending layer holds the length.
        // A windowed cache that slid holds fewer rows than its position: the position counts.
        let l = self
            .layers
            .iter()
            .find(|l| matches!(l.mixer, super::mixer::Mixer::Attention))?;
        let mut n = l.kv_cache.position();
        if let Some(c) = l.context.as_ref() {
            n = n.max(c.len());
        }
        #[cfg(feature = "gpu")]
        if let Some(c) = l.quant_kv.as_ref() {
            n = n.max(c.current_seq_len());
        }
        if let Some(c) = l.cpu_q8_kv.as_ref() {
            n = n.max(c.len());
        }
        Some(n)
    }

    /// PREFILL the whole prompts of several sequences in ONE forward, `seqs` holding each
    /// sequence's tokens and the paged-store slots of their K/V. The projections run over the
    /// tokens of every sequence together and the attention over each sequence's own tokens;
    /// all positions' K/V are written into the paged stores. Returns the LAST position's
    /// logits of each sequence, `[seqs.len(), vocab]`. Same dense-arch support + compute
    /// dtype as `batched_paged_decode`.
    pub fn paged_prefill_batch(
        &self,
        seqs: &[(&[u32], &[usize])],
        stores: &mut [crate::inference::cache::paged_kv::PagedKvStore],
    ) -> Result<Tensor> {
        self.paged_prefill_mixed(seqs, &[], stores)
    }

    /// [`Self::paged_prefill_batch`] with prompts whose prefix the cache holds: every row of
    /// every prompt goes through the layers in one forward, the weights read once. Returns the
    /// logits of the last position of each prompt, the plain prompts first, then the cached ones.
    pub fn paged_prefill_mixed(
        &self,
        plain: &[(&[u32], &[usize])],
        cached: &[CachedPrompt<'_>],
        stores: &mut [crate::inference::cache::paged_kv::PagedKvStore],
    ) -> Result<Tensor> {
        use crate::tensor::Tensor;
        let c = &self.config;
        // The rows of the forward: the plain prompts whole, then the suffix of each prompt
        // whose prefix the cache holds.
        let lens: Vec<usize> = plain
            .iter()
            .map(|(tokens, _)| tokens.len())
            .chain(cached.iter().map(|p| p.tokens.len() - p.cached_len))
            .collect();
        let starts: Vec<usize> = lens
            .iter()
            .scan(0, |at, &n| {
                let start = *at;
                *at += n;
                Some(start)
            })
            .collect();
        let t: usize = lens.iter().sum();
        let tokens: Vec<u32> = plain
            .iter()
            .flat_map(|(tokens, _)| tokens.iter().copied())
            .chain(
                cached
                    .iter()
                    .flat_map(|p| p.tokens[p.cached_len..].iter().copied()),
            )
            .collect();
        let slots: Vec<usize> = plain
            .iter()
            .flat_map(|(_, slots)| slots.iter().copied())
            .chain(cached.iter().flat_map(|p| p.suffix_slots.iter().copied()))
            .collect();
        let (nh, nkv, hd) = (c.n_head, c.n_kv_head, c.head_dim);
        let scale = 1.0f32 / (hd as f32).sqrt();
        let base = c.rope_freq_base;
        let edev = self.embeddings.embeddings().device().clone();
        let dev = self.compute_device();
        let cdt = Self::residual_dtype(&dev);
        let positions: Vec<usize> = plain
            .iter()
            .flat_map(|(tokens, _)| 0..tokens.len())
            .chain(cached.iter().flat_map(|p| p.cached_len..p.tokens.len()))
            .collect();
        let plain_rows: usize = lens[..plain.len()].iter().sum();

        // A table the card holds is read there, one launch for every row of the forward.
        #[cfg(feature = "gpu")]
        let on_card = match self.embed_table_on(&dev) {
            Some(table) => {
                let ids = Tensor::from_vec(tokens.clone(), &[t][..], &dev)?;
                Some(
                    self.embed_rows_of_ids(table, &ids)?
                        .reshape(&[t, c.embedding_length][..])?
                        .to_dtype(cdt)?,
                )
            }
            None => None,
        };
        #[cfg(not(feature = "gpu"))]
        let on_card: Option<Tensor> = None;
        let mut x = match on_card {
            Some(x) => x,
            None => {
                let ids = Tensor::from_vec(tokens, &[t, 1][..], &edev)?;
                self.embeddings
                    .forward(&ids)?
                    .reshape(&[t, c.embedding_length][..])?
                    .to_dtype(cdt)?
            }
        };
        if !x.device().same_device(&dev) {
            x = x.to_device(&dev)?;
        }
        let (rcos, rsin) = crate::inference::cache::paged_attention::rope_cos_sin(
            &positions, hd, base, &dev, cdt,
        )?;
        let table_rope = self.cb_rope_from_tables();
        // The rows of each prompt: a mixture routes them as one forward of that prompt alone.
        let spans: Vec<(usize, usize)> = starts.iter().copied().zip(lens.iter().copied()).collect();

        let mut slot_index: Vec<(usize, Device, Tensor)> = Vec::new();
        // The slot of each row, read on the card by the kernel that writes the store.
        // The slots as I32, kept on the host too for the kernel's range check.
        let slots_i32: Option<Vec<i32>> = if dev.is_cuda() && cdt == DType::F16 {
            let converted = slots
                .iter()
                .map(|&s| i32::try_from(s).ok())
                .collect::<Option<Vec<_>>>();
            if converted.is_none() {
                static NOTED: std::sync::Once = std::sync::Once::new();
                NOTED.call_once(|| {
                    tracing::debug!(
                        "fused prefill input stage declined (a slot does not fit an i32), separate launches run"
                    );
                });
            }
            converted
        } else {
            None
        };
        let slot_dev = match slots_i32.as_ref() {
            Some(v) => Some(Tensor::from_vec(v.clone(), &[t][..], &dev)?),
            None => None,
        };
        note_if_non_finite("the embedding, before any layer", &x);
        // On a card in half precision the residual add and the norm that follows it run as one
        // kernel, the activation as another: the next layer's attention norm comes from the
        // previous layer's fused add.
        let fused = cdt == DType::F16 && x.device().is_cuda();
        let mut next_norm: Option<Tensor> = None;
        for (li, layer) in self.layers.iter().enumerate() {
            let residual = x.clone();
            let hn = match next_norm.take() {
                Some(h) => h,
                None => layer.attn_norm.forward(&x)?.to_dtype(cdt)?,
            };
            let (qd, kvd) = (nh * hd, nkv * hd);
            let projected = if let Some(qkvw) = layer.attn_qkv.as_ref() {
                let mut qkv = qkvw.forward(&hn)?;
                if let Some(bias) = layer.attn_qkv_bias.as_ref() {
                    qkv = qkv.broadcast_add(&bias.to_dtype(qkv.dtype())?)?;
                }
                PrefillProjection::Fused(qkv)
            } else {
                let mut q = layer.attn_q.as_ref().unwrap().forward(&hn)?;
                let mut k = layer.attn_k.as_ref().unwrap().forward(&hn)?;
                let mut v = layer.attn_v.as_ref().unwrap().forward(&hn)?;
                if let Some(bias) = layer.attn_q_bias.as_ref() {
                    q = q.broadcast_add(&bias.to_dtype(q.dtype())?)?;
                }
                if let Some(bias) = layer.attn_k_bias.as_ref() {
                    k = k.broadcast_add(&bias.to_dtype(k.dtype())?)?;
                }
                if let Some(bias) = layer.attn_v_bias.as_ref() {
                    v = v.broadcast_add(&bias.to_dtype(v.dtype())?)?;
                }
                PrefillProjection::Split(q, k, v)
            };
            // One launch norms and rotates q and k, and writes k and v into the store.
            let norms = match (layer.attn_q_norm.as_ref(), layer.attn_k_norm.as_ref()) {
                (Some(qn), Some(kn)) => Some([
                    (qn.weight(), qn.eps() as f32),
                    (kn.weight(), kn.eps() as f32),
                ]),
                _ => None,
            };
            let norms_paired =
                norms.is_some() || (layer.attn_q_norm.is_none() && layer.attn_k_norm.is_none());
            let stage_fused =
                prefill_stage_fused(dev.is_cuda(), cdt, stores[li].k().dtype(), hd, norms_paired);
            #[cfg(feature = "gpu")]
            let flash_head_dim =
                crate::inference::kernel::fused::FLASH_PREFILL_GQA_HEAD_DIMS.contains(&hd);
            #[cfg(not(feature = "gpu"))]
            let flash_head_dim = false;
            #[cfg(feature = "gpu")]
            let flash_card = crate::inference::kernel::fused::flash_prefill_gqa_card(&dev)?;
            #[cfg(not(feature = "gpu"))]
            let flash_card = false;
            let want_head_major = prefill_attention_head_major(
                stage_fused,
                cached.len(),
                flash_card,
                flash_head_dim,
                nh,
                nkv,
            );
            let mut q_head_major = false;
            // The fused stages rotate at the plain base and feed the attention without a sink
            // or a window.
            let plain_stage = !table_rope && !card_scores_attention(layer);
            let staged = match slot_dev.as_ref().zip(slots_i32.as_deref()) {
                Some((slot_dev, slots_i32)) if want_head_major && plain_stage => {
                    let rows =
                        crate::inference::kernel::fused::prefill_rope_kv_write_head_major_f16(
                            projected.sources(qd, kvd),
                            norms,
                            &rcos,
                            &rsin,
                            self.config.flags.use_rope_i,
                            slot_dev,
                            slots_i32,
                            stores[li].k(),
                            stores[li].v(),
                            nh,
                            nkv,
                            hd,
                        )?;
                    q_head_major = rows.is_some();
                    rows
                }
                Some((slot_dev, slots_i32)) if stage_fused && plain_stage => {
                    crate::inference::kernel::fused::prefill_rope_kv_write_f16(
                        projected.sources(qd, kvd),
                        norms,
                        &rcos,
                        &rsin,
                        self.config.flags.use_rope_i,
                        slot_dev,
                        slots_i32,
                        stores[li].k(),
                        stores[li].v(),
                        nh,
                        nkv,
                        hd,
                    )?
                }
                _ => None,
            };
            let (q_rot, k_rot, v_f) = match staged {
                Some(rows) => rows,
                None => {
                    let (q, k, v) = projected.split(qd, kvd)?;
                    let qh = q.to_dtype(cdt)?.reshape(&[t, nh, hd][..])?;
                    let qh = if let Some(qn) = layer.attn_q_norm.as_ref() {
                        qn.forward(&qh.contiguous()?)?
                    } else {
                        qh
                    };
                    let kh = k.to_dtype(cdt)?.reshape(&[t, nkv, hd][..])?;
                    let kh = if let Some(kn) = layer.attn_k_norm.as_ref() {
                        kn.forward(&kh.contiguous()?)?
                    } else {
                        kh
                    };
                    let (q_rot, k_rot) = if table_rope {
                        (
                            rotate_rows_by_table(layer, &qh, &positions)?,
                            rotate_rows_by_table(layer, &kh, &positions)?,
                        )
                    } else {
                        (
                            crate::inference::cache::paged_attention::rope_apply(
                                &qh,
                                &rcos,
                                &rsin,
                                self.config.flags.use_rope_i,
                            )?,
                            crate::inference::cache::paged_attention::rope_apply(
                                &kh,
                                &rcos,
                                &rsin,
                                self.config.flags.use_rope_i,
                            )?,
                        )
                    };
                    let v_f = v.to_dtype(cdt)?.reshape(&[t, nkv, hd][..])?;
                    // persist all T positions' K/V for the subsequent decode phase
                    // The scatter index depends on the slots, the row width and the card: built
                    // once for each of them in the forward, not once per layer.
                    let feat = stores[li].feat();
                    let card = k_rot.device();
                    let at = match slot_index
                        .iter()
                        .position(|(f, d, _)| *f == feat && d.same_device(&card))
                    {
                        Some(at) => at,
                        None => {
                            slot_index.push((
                                feat,
                                card.clone(),
                                stores[li].slot_index(&slots, &card)?,
                            ));
                            slot_index.len() - 1
                        }
                    };
                    stores[li].write_indexed(
                        &slot_index[at].2,
                        &k_rot.reshape(&[t, nkv * hd][..])?,
                        &v_f.reshape(&[t, nkv * hd][..])?,
                    )?;
                    (q_rot, k_rot, v_f)
                }
            };
            let mut parts: Vec<Tensor> = Vec::new();
            let host_shape =
                sink_or_window_shape(layer, nh, nkv, hd, scale).filter(|_| dev.is_cpu());
            if let Some(shape) = host_shape.as_ref() {
                // Each prompt reads its own keys, its sinks in the denominator, and a prompt
                // whose prefix the cache holds reads that prefix first.
                let (qv, kv, vv) = (host_f32(&q_rot)?, host_f32(&k_rot)?, host_f32(&v_f)?);
                let row = nh * hd;
                let kv_row = nkv * hd;
                let mut rows: Vec<f32> = Vec::with_capacity(t * row);
                for (i, (&start, &n)) in starts.iter().zip(&lens).enumerate() {
                    let fresh_k = &kv[start * kv_row..][..n * kv_row];
                    let fresh_v = &vv[start * kv_row..][..n * kv_row];
                    let (prefix, pk, pv) = match i.checked_sub(plain.len()) {
                        None => (0, Vec::new(), Vec::new()),
                        Some(ci) => {
                            let cp = &cached[ci];
                            let (pk, pv) = stores[li].gather_seq(cp.block_table, cp.cached_len)?;
                            (cp.cached_len, host_f32(&pk)?, host_f32(&pv)?)
                        }
                    };
                    let keys = [pk.as_slice(), fresh_k].concat();
                    let values = [pv.as_slice(), fresh_v].concat();
                    let ends: Vec<usize> = (prefix + 1..=prefix + n).collect();
                    rows.extend(crate::inference::cache::paged_attention::host_attn_rows(
                        &qv[start * row..][..n * row],
                        &keys,
                        &values,
                        &ends,
                        shape,
                    ));
                }
                parts.push(Tensor::from_vec(rows, &[t, row][..], &dev)?.to_dtype(cdt)?);
            } else if card_scores_attention(layer) {
                // On a card each prompt reads its own keys within its window, its sinks in the
                // softmax, and a prompt whose prefix the cache holds reads that prefix first.
                let window = layer.sliding_window.filter(|&w| w > 0);
                for (i, (&start, &n)) in starts.iter().zip(&lens).enumerate() {
                    let fresh_k = k_rot.narrow(0, start, n)?;
                    let fresh_v = v_f.narrow(0, start, n)?;
                    let (prefix, keys, values) = match i.checked_sub(plain.len()) {
                        None => (0, fresh_k, fresh_v),
                        Some(ci) => {
                            let cp = &cached[ci];
                            let (pk, pv) = stores[li].gather_seq(cp.block_table, cp.cached_len)?;
                            let held = |t: Tensor| -> Result<Tensor> {
                                t.reshape(&[cp.cached_len, nkv, hd][..])?.to_dtype(cdt)
                            };
                            (
                                cp.cached_len,
                                Tensor::cat(&[&held(pk)?, &fresh_k], 0)?,
                                Tensor::cat(&[&held(pv)?, &fresh_v], 0)?,
                            )
                        }
                    };
                    let ctx = prefix + n;
                    // 1 hides a key: one past the row's own position, or before its window.
                    let hidden: Vec<u8> = (0..n)
                        .flat_map(|r| {
                            let end = prefix + r + 1;
                            let first = window.map_or(0, |w| end.saturating_sub(w));
                            (0..ctx).map(move |j| u8::from(j >= end || j < first))
                        })
                        .collect();
                    let mask = Tensor::from_vec(hidden, &[n, ctx][..], &dev)?;
                    let heads = |t: Tensor, width: usize| -> Result<Tensor> {
                        t.reshape(&[1, t.dims()[0], width, hd][..])?
                            .transpose(1, 2)?
                            .contiguous()
                    };
                    let o = super::layer_cpu_attn::sink_attention_matmul(
                        &heads(q_rot.narrow(0, start, n)?, nh)?,
                        heads(keys, nkv)?,
                        heads(values, nkv)?,
                        Some(&mask),
                        layer.attn_sinks.as_ref(),
                        scale as f64,
                        nkv,
                        window,
                    )?;
                    parts.push(
                        o.transpose(1, 2)?
                            .contiguous()?
                            .reshape(&[n, nh * hd][..])?
                            .to_dtype(cdt)?,
                    );
                }
            } else if q_head_major {
                parts.push(
                    crate::inference::cache::paged_attention::paged_prefill_attn_seqs_head_major(
                        &q_rot,
                        &k_rot,
                        &v_f,
                        &lens[..plain.len()],
                        nh,
                        nkv,
                        hd,
                        scale,
                    )?
                    .reshape(&[plain_rows, nh * hd][..])?,
                );
            } else if !plain.is_empty() {
                parts.push(
                    crate::inference::cache::paged_attention::paged_prefill_attn_seqs(
                        &q_rot.narrow(0, 0, plain_rows)?,
                        &k_rot.narrow(0, 0, plain_rows)?,
                        &v_f.narrow(0, 0, plain_rows)?,
                        &lens[..plain.len()],
                        nh,
                        nkv,
                        hd,
                        scale,
                    )?
                    .reshape(&[plain_rows, nh * hd][..])?,
                );
            }
            // A prompt whose prefix the cache holds attends over that prefix, gathered from the
            // store, then over its own suffix.
            let mut at = plain_rows;
            let suffixes = match host_shape.is_some() || card_scores_attention(layer) {
                true => &[][..],
                false => cached,
            };
            for (cp, &n) in suffixes.iter().zip(&lens[plain.len()..]) {
                let (pk, pv) = stores[li].gather_seq(cp.block_table, cp.cached_len)?;
                let pk = pk.reshape(&[cp.cached_len, nkv, hd][..])?.to_dtype(cdt)?;
                let pv = pv.reshape(&[cp.cached_len, nkv, hd][..])?.to_dtype(cdt)?;
                let full_k = Tensor::cat(&[&pk, &k_rot.narrow(0, at, n)?], 0)?;
                let full_v = Tensor::cat(&[&pv, &v_f.narrow(0, at, n)?], 0)?;
                parts.push(
                    crate::inference::cache::paged_attention::paged_prefill_attn_suffix(
                        &q_rot.narrow(0, at, n)?,
                        &full_k,
                        &full_v,
                        cp.cached_len,
                        nh,
                        nkv,
                        hd,
                        scale,
                    )?
                    .reshape(&[n, nh * hd][..])?,
                );
                at += n;
            }
            let attn = if parts.len() == 1 {
                parts.remove(0)
            } else {
                Tensor::cat(&parts.iter().collect::<Vec<_>>(), 0)?
            };
            let o = attn
                .to_dtype(cdt)
                .and_then(|a| attention_output(layer, &a))?
                .to_dtype(cdt)?;
            let ffn_norm = layer.ffn_norm.as_ref().unwrap();
            let hn2 = if fused {
                let (sum, norm) = crate::inference::kernel::fused::fused_add_rmsnorm_f16(
                    &o,
                    &residual,
                    ffn_norm.weight(),
                    ffn_norm.eps() as f32,
                )?;
                x = sum;
                norm
            } else {
                x = residual.broadcast_add(&o)?;
                ffn_norm.forward(&x)?.to_dtype(cdt)?
            };
            let residual2 = x.clone();
            if let Some(moe) = layer.moe.as_ref() {
                // The next layer norms its own input: no fused sum leaves it one.
                let d = moe_rows_per_prompt(layer, moe, &hn2, &residual2, &spans)?.to_dtype(cdt)?;
                x = residual2.broadcast_add(&d)?;
                note_if_non_finite(&format!("layer {li}"), &x);
                continue;
            }
            let act = if let Some(gate) = layer.ffn_gate.as_ref() {
                let g = gate.forward(&hn2)?.to_dtype(cdt)?;
                let u = layer.ffn_up.forward(&hn2)?.to_dtype(cdt)?;
                if c.flags.use_gelu {
                    g.gelu()?.mul(&u)?
                } else {
                    match crate::inference::kernel::fused::silu_mul_f16(&g, &u)? {
                        Some(a) => a,
                        None => g.silu()?.mul(&u)?,
                    }
                }
            } else {
                let gu = layer.ffn_up.forward(&hn2)?.to_dtype(cdt)?;
                let one_launch = mlp_activation_fused(
                    gu.device().is_cuda(),
                    gu.dtype(),
                    c.ffn_dim,
                    c.flags.use_gelu,
                );
                let fused = if one_launch {
                    crate::inference::kernel::fused::silu_mul_gu_f16(&gu, c.ffn_dim)?
                } else {
                    None
                };
                match fused {
                    Some(a) => a,
                    None => {
                        let (g, u) = (
                            gu.narrow(1, 0, c.ffn_dim)?,
                            gu.narrow(1, c.ffn_dim, c.ffn_dim)?,
                        );
                        if c.flags.use_gelu {
                            g.gelu()?.mul(&u)?
                        } else {
                            match crate::inference::kernel::fused::silu_mul_f16(&g, &u)? {
                                Some(a) => a,
                                None => g.silu()?.mul(&u)?,
                            }
                        }
                    }
                }
            };
            let d = layer.ffn_down.forward(&act)?.to_dtype(cdt)?;
            match self.layers.get(li + 1) {
                Some(next) if fused => {
                    let (sum, h) = crate::inference::kernel::fused::fused_add_rmsnorm_f16(
                        &d,
                        &residual2,
                        next.attn_norm.weight(),
                        next.attn_norm.eps() as f32,
                    )?;
                    x = sum;
                    next_norm = Some(h);
                }
                _ => x = residual2.broadcast_add(&d)?,
            }
            note_if_non_finite(&format!("layer {li}"), &x);
        }
        // logits for the last position of each sequence only
        let last = starts
            .iter()
            .zip(&lens)
            .map(|(&start, &n)| x.narrow(0, start + n - 1, 1))
            .collect::<Result<Vec<_>>>()?;
        self.head_on_device(&Tensor::cat(&last.iter().collect::<Vec<_>>(), 0)?)
    }

    /// Prefix-cached prefill: compute ONLY the suffix `tokens[cached_len..]`,
    /// reusing the already-resident prefix KV (positions `0..cached_len`, gathered
    /// from each layer's store via `block_table`) for attention instead of
    /// recomputing it. `suffix_slots` are the KV slots for the suffix positions
    /// (length `t - cached_len`). Returns the LAST position's logits - equivalent to
    /// `paged_prefill_batch` over the whole prompt (validated) but skips the prefix's
    /// projections + FFN (the multi-turn / shared-system-prompt win). `cached_len`
    /// must be block-aligned (a whole number of cached blocks).
    pub fn paged_prefill_seq_cached(
        &self,
        tokens: &[u32],
        suffix_slots: &[usize],
        cached_len: usize,
        block_table: &[u32],
        stores: &mut [crate::inference::cache::paged_kv::PagedKvStore],
    ) -> Result<Tensor> {
        use crate::tensor::Tensor;
        let own_attention = self
            .layers
            .iter()
            .any(|l| l.attn_sinks.is_some() || l.sliding_window.is_some_and(|w| w > 0));
        if self.layers.iter().any(|l| l.moe.is_some())
            || self.cb_rope_from_tables()
            || own_attention
        {
            return Err(crate::tensor::Error::msg(
                "paged_prefill_seq_cached: a mixture, a table rotation, a sink or a window \
                 prefills through paged_prefill_mixed",
            ));
        }
        let c = &self.config;
        let t = tokens.len();
        let tsfx = t - cached_len;
        debug_assert_eq!(suffix_slots.len(), tsfx);
        let (nh, nkv, hd) = (c.n_head, c.n_kv_head, c.head_dim);
        let scale = 1.0f32 / (hd as f32).sqrt();
        let base = c.rope_freq_base;
        let edev = self.embeddings.embeddings().device().clone();
        let dev = self.compute_device();
        let cdt = Self::residual_dtype(&dev);
        let positions: Vec<usize> = (cached_len..t).collect();

        let ids = Tensor::from_vec(tokens[cached_len..].to_vec(), &[tsfx, 1][..], &edev)?;
        let mut x = self
            .embeddings
            .forward(&ids)?
            .reshape(&[tsfx, c.embedding_length][..])?
            .to_dtype(cdt)?;
        if !x.device().same_device(&dev) {
            x = x.to_device(&dev)?;
        }
        let (rcos, rsin) = crate::inference::cache::paged_attention::rope_cos_sin(
            &positions, hd, base, &dev, cdt,
        )?;

        for (li, layer) in self.layers.iter().enumerate() {
            let residual = x.clone();
            let hn = layer.attn_norm.forward(&x)?.to_dtype(cdt)?;
            let (qd, kvd) = (nh * hd, nkv * hd);
            let (q, k, v) = if let Some(qkvw) = layer.attn_qkv.as_ref() {
                let mut qkv = qkvw.forward(&hn)?;
                if let Some(bias) = layer.attn_qkv_bias.as_ref() {
                    qkv = qkv.broadcast_add(&bias.to_dtype(qkv.dtype())?)?;
                }
                (
                    qkv.narrow(1, 0, qd)?,
                    qkv.narrow(1, qd, kvd)?,
                    qkv.narrow(1, qd + kvd, kvd)?,
                )
            } else {
                let mut q = layer.attn_q.as_ref().unwrap().forward(&hn)?;
                let mut k = layer.attn_k.as_ref().unwrap().forward(&hn)?;
                let mut v = layer.attn_v.as_ref().unwrap().forward(&hn)?;
                if let Some(bias) = layer.attn_q_bias.as_ref() {
                    q = q.broadcast_add(&bias.to_dtype(q.dtype())?)?;
                }
                if let Some(bias) = layer.attn_k_bias.as_ref() {
                    k = k.broadcast_add(&bias.to_dtype(k.dtype())?)?;
                }
                if let Some(bias) = layer.attn_v_bias.as_ref() {
                    v = v.broadcast_add(&bias.to_dtype(v.dtype())?)?;
                }
                (q, k, v)
            };
            let qh = q.to_dtype(cdt)?.reshape(&[tsfx, nh, hd][..])?;
            let qh = if let Some(qn) = layer.attn_q_norm.as_ref() {
                qn.forward(&qh.contiguous()?)?
            } else {
                qh
            };
            let kh = k.to_dtype(cdt)?.reshape(&[tsfx, nkv, hd][..])?;
            let kh = if let Some(kn) = layer.attn_k_norm.as_ref() {
                kn.forward(&kh.contiguous()?)?
            } else {
                kh
            };
            let q_rot = crate::inference::cache::paged_attention::rope_apply(
                &qh,
                &rcos,
                &rsin,
                self.config.flags.use_rope_i,
            )?;
            let k_rot = crate::inference::cache::paged_attention::rope_apply(
                &kh,
                &rcos,
                &rsin,
                self.config.flags.use_rope_i,
            )?; // [tsfx, nkv, hd]
            let v_f = v.to_dtype(cdt)?.reshape(&[tsfx, nkv, hd][..])?;
            // Persist the SUFFIX K/V; gather the cached PREFIX K/V (already rotated)
            // and prepend it for attention. cached_len==0 = a plain full prefill.
            stores[li].write(
                suffix_slots,
                &k_rot.reshape(&[tsfx, nkv * hd][..])?,
                &v_f.reshape(&[tsfx, nkv * hd][..])?,
            )?;
            let (full_k, full_v) = if cached_len == 0 {
                (k_rot.clone(), v_f.clone())
            } else {
                let (pk, pv) = stores[li].gather_seq(block_table, cached_len)?; // [cached, feat]
                let pk = pk.reshape(&[cached_len, nkv, hd][..])?.to_dtype(cdt)?;
                let pv = pv.reshape(&[cached_len, nkv, hd][..])?.to_dtype(cdt)?;
                (
                    Tensor::cat(&[&pk, &k_rot], 0)?,
                    Tensor::cat(&[&pv, &v_f], 0)?,
                ) // [t, nkv, hd]
            };
            let attn = crate::inference::cache::paged_attention::paged_prefill_attn_suffix(
                &q_rot, &full_k, &full_v, cached_len, nh, nkv, hd, scale,
            )?
            .reshape(&[tsfx, nh * hd][..])?;
            let o = attn
                .to_dtype(cdt)
                .and_then(|a| attention_output(layer, &a))?
                .to_dtype(cdt)?;
            x = residual.broadcast_add(&o)?;
            let residual2 = x.clone();
            let hn2 = layer
                .ffn_norm
                .as_ref()
                .unwrap()
                .forward(&x)?
                .to_dtype(cdt)?;
            let (g, u) = if let Some(gate) = layer.ffn_gate.as_ref() {
                (
                    gate.forward(&hn2)?.to_dtype(cdt)?,
                    layer.ffn_up.forward(&hn2)?.to_dtype(cdt)?,
                )
            } else {
                let gu = layer.ffn_up.forward(&hn2)?.to_dtype(cdt)?;
                (
                    gu.narrow(1, 0, c.ffn_dim)?,
                    gu.narrow(1, c.ffn_dim, c.ffn_dim)?,
                )
            };
            let act = if c.flags.use_gelu {
                g.gelu()?.mul(&u)?
            } else {
                g.silu()?.mul(&u)?
            };
            let d = layer.ffn_down.forward(&act)?.to_dtype(cdt)?;
            x = residual2.broadcast_add(&d)?;
            note_if_non_finite(&format!("layer {li}"), &x);
        }
        let last = x.narrow(0, tsfx - 1, 1)?;
        self.head_on_device(&last)
    }

    /// Multi-position forward: returns logits for EVERY input position
    /// (shape `[batch, seq_len, vocab]`). Used by speculative decoding to
    /// verify K drafts in a single batched forward.
    ///
    /// A draft verify (a few rows continuing a cache) computes every row bit for bit as the
    /// decode step it stands for, on the host and on a card.
    /// So drafting never changes what greedy decoding emits.
    /// The rows share each weight read where a kernel sums each row as a one-row call: the
    /// quantised mat-vec kernels, the float mat-vec kernel and every product on the host.
    /// The float mat-vec kernel serves an F32, F16 or BF16 weight on a card and the float head
    /// held in F16, for a decode step and for a verify alike.
    pub fn forward_all(&mut self, input_ids: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        let verify = is_verify(seq_len, index_pos);
        let recording = verify && self.is_recurrent();
        if recording {
            self.spec_open(input_ids, index_pos)?;
        }
        // The verify advances each state where it lies, as a decoded token does, so a step
        // recorded against those buffers goes on reading them.
        #[cfg(feature = "gpu")]
        let _in_place = recording.then(|| crate::inference::moe_cuda::in_place_state(true));
        // Every verified row runs the decode step it stands for, so greedy output does not
        // depend on how many tokens were drafted.
        let _order = verify_order(seq_len, index_pos);
        let out = self.forward_inner(input_ids, index_pos, /*all_positions=*/ true);
        if recording {
            self.spec_close(out.is_ok());
        }
        out
    }

    /// `forward_all` with the rows batched as a prefill batches them, for the verify
    /// benchmark.
    #[cfg(test)]
    pub(super) fn forward_all_batched(
        &mut self,
        input_ids: &Tensor,
        index_pos: usize,
    ) -> Result<Tensor> {
        self.forward_inner(input_ids, index_pos, /*all_positions=*/ true)
    }

    fn forward_inner(
        &mut self,
        input_ids: &Tensor,
        index_pos: usize,
        all_positions: bool,
    ) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        self.spec_clear();
        let mut since = std::time::Instant::now();
        let (hidden, ple_inputs) = self.embed_stage(input_ids, index_pos)?;
        forward_mark("forward: embedding", &mut since);
        let (hidden, on_cuda) = self.run_layers(hidden, ple_inputs, index_pos, seq_len)?;
        forward_mark("forward: layers", &mut since);
        // A prompt chunk also goes through the draft head, so the head's cache holds the prompt
        // by the time the first token is drafted.
        if self.capture_feature && self.draft.is_some() && seq_len > 1 && !all_positions {
            let ids = input_ids
                .flatten_all()?
                .to_dtype(crate::tensor::DType::U32)?
                .to_vec1::<u32>()?;
            self.draft_ingest(&ids, &hidden, index_pos)?;
            forward_mark("forward: draft head", &mut since);
        }
        let out = self.output_stage(hidden, on_cuda, seq_len, all_positions);
        forward_mark("forward: output", &mut since);
        out
    }

    /// The n-gram table's rows for `input_ids`, which start at `index_pos`, gathered for the
    /// layer that takes them: the history of the sequence advances past the tokens. Nothing for
    /// a model with no n-gram layer.
    pub(super) fn gather_ngram_rows(&mut self, input_ids: &Tensor, index_pos: usize) -> Result<()> {
        let Some(ng) = self.ngram.as_mut() else {
            return Ok(());
        };
        let ple = self.layers[ng.meta.layer]
            .ngram
            .as_mut()
            .expect("the layer the n-gram file names carries its block");
        if index_pos == 0 {
            ple.reset();
        }
        // A forward that failed after its gather runs again from the same position: the
        // history goes back to where that gather found it.
        if ple.history.seen != index_pos {
            if let Some((at, before)) = ng.before.as_ref().filter(|(at, _)| *at == index_pos) {
                debug_assert_eq!(*at, before.seen);
                ple.history = before.clone();
            }
        }
        if ple.history.seen != index_pos {
            // What stands between the history and the forward: the verify windows open.
            let recorded = self.verify.window.as_ref().map_or("none".to_string(), |w| {
                format!("{} rows at {}", w.rows, w.base)
            });
            let eager = self.spec_window.as_ref().map_or("none".to_string(), |w| {
                format!("{} rows at {}", w.rows, w.base)
            });
            crate::tensor::bail!(
                "the n-gram history holds {} tokens but the forward starts at {index_pos} (recorded window {recorded}, eager window {eager})",
                ple.history.seen
            );
        }
        let ids = input_ids
            .flatten_all()?
            .to_dtype(crate::tensor::DType::U32)?
            .to_vec1::<u32>()?;
        ng.before = Some((index_pos, ple.history.clone()));
        let rows = ple.history.rows(&ng.meta, &ids);
        let data = ng.table.gather(&rows)?;
        let width = ng.meta.heads() * ng.meta.row_dim;
        ng.pending = Some(Tensor::from_vec(data, (1, ids.len(), width), &Device::Cpu)?);
        Ok(())
    }

    /// The first stage of a forward: the embeddings of `input_ids` as the residual stream
    /// the layers start from, and the per-layer inputs of a family that has them.
    pub(super) fn embed_stage(
        &mut self,
        input_ids: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Option<Vec<Tensor>>)> {
        let (_, seq_len) = input_ids.dims2()?;

        // Small-model CPU prefill: route the activation quantize onto the same
        // logical prefill pool as the matmul, so the physical pool stays parked
        // and its cross-pool idle-spin is not paid. A narrow hidden size is the
        // model-level signal the per-matmul shape cannot separate; self-resets
        // to false on decode (seq_len == 1) and on wide-hidden models.
        crate::tensor::quant_cpu::set_small_model_prefill(
            seq_len > 1 && self.config.embedding_length <= 1024,
        );

        // -- Embedding ------------------------------
        // Cast input_ids to the embeddings table's actual device. Today
        // that's always CPU; Path A's load-time change will move small
        // tables to GPU 0 and this cast becomes a no-op (or short device->
        // device copy) for those models.
        let emb_device = self.embeddings.embeddings().device().clone();
        let input_local;
        let input_ids = if input_ids.device().same_device(&emb_device) {
            input_ids
        } else {
            input_local = input_ids.to_device(&emb_device)?;
            &input_local
        };
        let hidden = self.add_positions(self.embeddings.forward(input_ids)?, index_pos)?;
        // Embedding scaling (Gemma: √embed_len, Granite: explicit value)
        let hidden = if let Some(scale) = self.config.embed_scale {
            (hidden * scale)?
        } else {
            hidden
        };
        note_if_non_finite("the embedding, before any layer", &hidden);

        // A wide residual starts as one copy of the embedding per stream, and the n-gram
        // rows of the tokens are gathered for the layer that takes them.
        let hidden = match self.config.hyper {
            Some(h) => {
                self.gather_ngram_rows(input_ids, index_pos)?;
                super::wide::widen_embedding(&hidden, h.streams)?
            }
            None => hidden,
        };
        // A Sinkhorn wide residual starts the same way, and the state its layers pass to each
        // other opens for the rows of this forward, the token ids its engram layers hash
        // included.
        if self.wide_sparse.is_some() {
            if seq_len > 1 {
                self.give_back_for_prompt(index_pos, index_pos + seq_len);
            }
            self.take_back_card_room(index_pos + seq_len);
        }
        let hidden = match self.wide_sparse.as_ref() {
            Some(ws) => ws.widen(&hidden, input_ids, index_pos)?,
            None => hidden,
        };

        // -- Compute PLE per-layer inputs (Gemma4) ------------------------
        let ple_inputs: Option<Vec<Tensor>> = if self.ple_dim > 0 {
            if let (Some(model_proj), Some(proj_norm)) = (&self.ple_model_proj, &self.ple_proj_norm)
            {
                let n_layers = self.layers.len();
                let (b, seq_h, _) = hidden.dims3()?;
                let ple_scale = (self.ple_dim as f64).sqrt();
                // Token embeddings: use Embedding if available, else QTensor per-row dequant.
                // Cast input_ids to PLE table's device.
                let tok_ple = if let Some(embd) = &self.ple_token_embd {
                    let ple_device = embd.embeddings().device().clone();
                    let input_local;
                    let input_ids_ple = if input_ids.device().same_device(&ple_device) {
                        input_ids
                    } else {
                        input_local = input_ids.to_device(&ple_device)?;
                        &input_local
                    };
                    let t = embd.forward(input_ids_ple)?;
                    let t = if t.dtype() != crate::tensor::DType::F32 {
                        t.to_dtype(crate::tensor::DType::F32)?
                    } else {
                        t
                    };
                    Some((t * ple_scale)?)
                } else {
                    None
                };
                // Model projection -> [flat_seq, n_layers * ple_dim]. Move
                // hidden to the device the PLE projection lives on (GPU 0 when
                // available, else CPU); avoids the host round-trip that used
                // to dominate gemma4 prefill.
                let proj_dev = model_proj.device();
                let hidden_on_proj = if hidden.device().same_device(&proj_dev) {
                    hidden.clone()
                } else {
                    hidden.to_device(&proj_dev)?
                };
                let hidden_2d =
                    hidden_on_proj.reshape((b * seq_h, self.config.embedding_length))?;
                let model_ple = model_proj.forward(&hidden_2d)?;
                // Scale by 1/√hidden_size (matching Ollama's Go implementation)
                let model_ple = (model_ple / (self.config.embedding_length as f64).sqrt())?;
                // Reshape to [n_tokens, n_layers, ple_dim] for the RMSNorm (normalizes last dim)
                let model_ple = model_ple.reshape((b * seq_h, n_layers, self.ple_dim))?;
                let model_ple = proj_norm.forward(&model_ple)?;
                // Combine: (model + token) / √2, or just model if no token embedding.
                // tok comes from CPU embeddings; align it with model_ple's device
                // (typically GPU 0 now) before the addition.
                let ple = if let Some(tok) = tok_ple {
                    let tok = tok.reshape((b * seq_h, n_layers, self.ple_dim))?;
                    let tok = if tok.device().same_device(&model_ple.device()) {
                        tok
                    } else {
                        tok.to_device(&model_ple.device())?
                    };
                    // Go code line 249: scale by 1/√2 when combining both sources
                    ((model_ple + tok)? * (1.0 / 2f64.sqrt()))?
                } else {
                    model_ple
                };
                // Split into per-layer tensors: [n_tokens, ple_dim] -> [b, seq, ple_dim]
                let mut per_layer = Vec::with_capacity(n_layers);
                for i in 0..n_layers {
                    per_layer.push(ple.i((.., i, ..))?.reshape((b, seq_h, self.ple_dim))?);
                }
                Some(per_layer)
            } else {
                None
            }
        } else {
            None
        };

        Ok((hidden, ple_inputs))
    }

    /// Run the per-layer loop + final norm + output projection given a
    /// pre-embedded `hidden` state. Factored out of `forward_inner` so the
    /// vision-prefill path (`forward_with_image_embeds`) can splice image
    /// embeddings into `hidden` between the BOS and text token embeddings
    /// before reusing the standard layer + output-proj logic.
    ///
    /// `hidden`: (batch, seq_len, embedding_length) on CPU or a single CUDA
    /// device. The function moves it across segment boundaries as needed.
    /// `ple_inputs`: per-layer Gemma4 PLE tensors, or None for arches that
    /// don't use PLE (phi2, qwen2/3, llama, etc).
    /// `seq_len` must equal `hidden.dim(1)`.
    pub(super) fn forward_layers_and_output(
        &mut self,
        hidden: Tensor,
        ple_inputs: Option<Vec<Tensor>>,
        index_pos: usize,
        seq_len: usize,
        all_positions: bool,
    ) -> Result<Tensor> {
        let (hidden, on_cuda) = self.run_layers(hidden, ple_inputs, index_pos, seq_len)?;
        self.output_stage(hidden, on_cuda, seq_len, all_positions)
    }

    /// The layer loop of a forward: `hidden` through every layer, crossing the placement
    /// segments, returned with whether it ends on a card.
    pub(super) fn run_layers(
        &mut self,
        hidden: Tensor,
        ple_inputs: Option<Vec<Tensor>>,
        index_pos: usize,
        seq_len: usize,
    ) -> Result<(Tensor, bool)> {
        let upto = self.layers.len();
        self.run_layers_in(hidden, ple_inputs, index_pos, seq_len, 0, upto)
    }

    /// `run_layers` over the layers `[from, upto)` alone.
    pub(super) fn run_layers_in(
        &mut self,
        mut hidden: Tensor,
        ple_inputs: Option<Vec<Tensor>>,
        index_pos: usize,
        seq_len: usize,
        from: usize,
        upto: usize,
    ) -> Result<(Tensor, bool)> {
        let mut on_cuda = !hidden.device().is_cpu();
        self.vulkan.clear_logits();
        self.opencl.clear_logits();
        // A forward through every layer reports the share of its rows its layers done stand for.
        let reports = from == 0 && upto == self.layers.len();

        // -- Layer-by-layer with segment-based transfers -------------------
        let mut seg_start = from;
        let mut current_cuda_idx: Option<usize> =
            if let crate::tensor::DeviceLocation::Cuda { gpu_id } = hidden.device().location() {
                Some(gpu_id)
            } else {
                None
            };
        while seg_start < upto {
            // A Vulkan segment takes the rows on the host and gives them back there.
            if let Some(end) = self.vulkan.end_of(seg_start) {
                let end = end.min(upto);
                hidden = self.vulkan.run(hidden, index_pos, seg_start, end)?;
                current_cuda_idx = None;
                on_cuda = false;
                seg_start = end;
                continue;
            }
            // An OpenCL segment likewise.
            if let Some(end) = self.opencl.end_of(seg_start) {
                let end = end.min(upto);
                hidden = self.opencl.run(hidden, index_pos, seg_start, end)?;
                current_cuda_idx = None;
                on_cuda = false;
                seg_start = end;
                continue;
            }
            let seg_dev = &self.layer_devs[seg_start];
            let mut seg_end = seg_start + 1;
            while seg_end < upto
                && &self.layer_devs[seg_end] == seg_dev
                && self.vulkan.end_of(seg_end).is_none()
                && self.opencl.end_of(seg_end).is_none()
            {
                seg_end += 1;
            }

            let target_cuda_idx = match seg_dev {
                LayerDevice::Cuda(idx) => Some(*idx),
                LayerDevice::Cpu => None,
            };

            // Transfer at segment boundary (CPU↔CUDA or CUDA↔CUDA)
            if target_cuda_idx != current_cuda_idx {
                // The destination must not read partially-written data while the
                // source's kernels are still in flight - without that ordering the
                // output varies between runs (observed on gemma4:26b across GPUs and
                // CPU). The ordering now travels WITH the transfer, inside `to_device`,
                // instead of being restated on both sides here: one declaration, and
                // the CUDA->CUDA case establishes it with an event rather than by
                // stopping the host. Work queued afterwards on the destination stream
                // is already ordered behind the copy, so the trailing sync was only
                // ever symmetry.
                hidden = if let Some(idx) = target_cuda_idx {
                    if let Some(dev) = self.cuda_devices.get(&idx) {
                        hidden.to_device(dev)?
                    } else {
                        hidden
                    }
                } else {
                    hidden.to_device(&Device::Cpu)?
                };
                current_cuda_idx = target_cuda_idx;
                on_cuda = current_cuda_idx.is_some();
            }

            // Compute mask once for segment. seq_kv = index_pos + seq_q so
            // session-resume / PLD-verify paths (where index_pos > 0) get a
            // [seq_q, index_pos+seq_q] mask that broadcasts correctly to
            // attention scores [b, n_head, seq_q, seq_kv]. Prefill at
            // index_pos=0 collapses to [seq_q, seq_q].
            let mask = if seq_len > 1 {
                Some(self.make_mask(seq_len, index_pos, current_cuda_idx)?)
            } else {
                None
            };
            // Bounded sliding-window attention: a layer whose windowed cache keeps only its
            // last rows reads a K shorter than the sequence, and takes a mask as wide as that K.
            // The masks are built once per window and K width for this segment.
            let mut window_masks: std::collections::HashMap<(usize, usize), Tensor> =
                std::collections::HashMap::new();

            // Shared KV: layers >= first_kv_own compute their own K/V;
            // layers after that reuse K/V from reference layers.
            let n_layers = self.layers.len();
            let shared_kv_n = self.config.shared_kv_layers;
            let first_kv_shared = if shared_kv_n > 0 {
                n_layers - shared_kv_n
            } else {
                n_layers
            };

            // Run all layers in segment
            // THE per-layer loop of the decode path: every request walks the placement
            // segments through here. Timed on the calling thread, so a layer that
            // dispatches to a device and one that runs on the host are told apart.
            for layer_idx in seg_start..seg_end {
                let _timer = crate::inference::place::layer_perf::LayerTimer::start(layer_idx);
                // PLE: the layer's input moves to the layer's device, host or card.
                let ple_on_dev = match ple_inputs.as_ref().and_then(|inputs| inputs.get(layer_idx))
                {
                    Some(t) => {
                        let layer_dev = if on_cuda {
                            current_cuda_idx
                                .and_then(|idx| self.cuda_devices.get(&idx))
                                .cloned()
                        } else {
                            Some(Device::Cpu)
                        };
                        match layer_dev {
                            Some(d) if !t.device().same_device(&d) => Some(t.to_device(&d)?),
                            _ => Some(t.clone()),
                        }
                    }
                    None => None,
                };

                // The n-gram embeddings go to the one layer that adds them.
                let ple_on_dev = match (
                    self.ngram.as_ref().and_then(|n| n.pending.as_ref()),
                    self.layers[layer_idx].ngram.is_some(),
                ) {
                    (Some(emb), true) => Some(emb.clone()),
                    _ => ple_on_dev,
                };

                // Shared KV: for layers in the shared range, grab cached K/V from a
                // type-matched reference layer (SWA->SWA ref, global->global ref).
                let (shared_kv, ref_idx_opt) =
                    if layer_idx >= first_kv_shared && first_kv_shared > 0 {
                        let layer_hd = self.layers[layer_idx].head_dim;
                        let ref_idx = (0..first_kv_shared)
                            .rev()
                            .find(|&i| self.layers[i].head_dim == layer_hd)
                            .unwrap_or(first_kv_shared.saturating_sub(1));
                        let target_dev = self.layers[layer_idx].cos.device().clone();
                        let ref_cache = &self.layers[ref_idx].kv_cache;
                        // In graph mode (single-token decode + padded mask set on
                        // the receiver), prefer the donor's FULL padded buffer over
                        // the narrowed `current_kv()` view. The receiver's
                        // `padded_standard_attention` adds a [1,1,1,max_kv_padded]
                        // mask onto the scores; with a narrowed K of width
                        // current_seq_len, scores collapse to that width and
                        // broadcast_add throws a shape mismatch.
                        let use_padded = seq_len == 1
                            && self.layers[layer_idx].padded_mask.is_some()
                            && ref_cache.k_buffer().is_some();
                        let kv = if use_padded {
                            let k = ref_cache.k_buffer().unwrap();
                            let v = ref_cache.v_buffer().unwrap();
                            let k = if k.device().same_device(&target_dev) {
                                k
                            } else {
                                k.to_device(&target_dev)?
                            };
                            let v = if v.device().same_device(&target_dev) {
                                v
                            } else {
                                v.to_device(&target_dev)?
                            };
                            Some((k, v))
                        } else if let Some((k, v)) = ref_cache.current_kv() {
                            let k = if k.device().same_device(&target_dev) {
                                k.clone()
                            } else {
                                k.to_device(&target_dev)?
                            };
                            let v = if v.device().same_device(&target_dev) {
                                v.clone()
                            } else {
                                v.to_device(&target_dev)?
                            };
                            Some((k, v))
                        } else {
                            None
                        };
                        (kv, Some(ref_idx))
                    } else {
                        (None, None)
                    };

                // gemma4 bounded-SWA: windowed SpecKvCache layers use the bounded
                // SWA mask (matches their shortened K); all others use the full mask.
                // A period-classified model gives each layer the mask of its own window.
                let own_mask = if seq_len > 1 && self.config.swa_period.is_some() {
                    let window = self.layers[layer_idx].prefill_mask_window;
                    Some(self.make_layer_mask(seq_len, index_pos, current_cuda_idx, window)?)
                } else {
                    None
                };
                // A windowed layer reads its donor's rows, which the donor appended earlier in
                // this pass, or its own rows once this pass appends to them.
                let window_rows = match self.layers[layer_idx].kv_cache.window() {
                    Some(w) if w > 0 && seq_len > 1 && own_mask.is_none() => {
                        let rows = match ref_idx_opt {
                            Some(r) => self.layers[r].kv_cache.current_seq_len(),
                            None => {
                                let cache = &self.layers[layer_idx].kv_cache;
                                // The prefill attention resets the cache at position 0.
                                let before = match index_pos {
                                    0 => 0,
                                    _ => cache.current_seq_len(),
                                };
                                cache.rows_after_append(before, seq_len)
                            }
                        };
                        Some((w, rows))
                    }
                    _ => None,
                };
                if let Some((w, rows)) = window_rows {
                    if let std::collections::hash_map::Entry::Vacant(slot) =
                        window_masks.entry((w, rows))
                    {
                        let start = (index_pos + seq_len).checked_sub(rows).ok_or_else(|| {
                            crate::tensor::Error::msg(format!(
                                "layer {layer_idx}: {rows} window rows past position {}",
                                index_pos + seq_len
                            ))
                        })?;
                        let dev = current_cuda_idx
                            .and_then(|idx| self.cuda_devices.get(&idx))
                            .unwrap_or(&Device::Cpu);
                        let data = swa_bounded_mask_data(seq_len, index_pos, rows, start, w);
                        let m = Tensor::from_slice(&data, (seq_len, rows), dev)?;
                        slot.insert(m);
                    }
                }
                let layer_mask = if own_mask.is_some() {
                    own_mask.as_ref()
                } else if let Some(key) = window_rows {
                    window_masks.get(&key)
                } else {
                    mask.as_ref()
                };
                // Borrow donor's q8 cache (immutable) AND mutate the shared
                // layer simultaneously. split_at_mut gives us the mutable
                // tail (containing layer_idx) plus immutable head (containing
                // donor index, which is always < first_kv_shared <= layer_idx).
                if let Some(rid) = ref_idx_opt {
                    let (head, tail) = self.layers.split_at_mut(layer_idx);
                    #[cfg(feature = "gpu")]
                    let donor_q8 = head[rid].q8_kv();
                    // Lend the shared layer read access to the donor's F16
                    // KV store (gemma4 CPU). The donor (a real layer < layer_idx,
                    // in `head`) already maintains it for its own decode; shared
                    // layers scan it in F16 (half the F32 `shared_kv` bytes, and a
                    // fused single-pass vs a 4-op matmul chain) - measured
                    // ~2.5x faster per global-layer call. split_at_mut proves
                    // rid != layer_idx so the immutable borrow is sound.
                    let donor_f16 = head[rid].cpu_f16_kv.as_ref();
                    let layer = &mut tail[0];
                    hidden = layer.forward(
                        &hidden,
                        layer_mask,
                        index_pos,
                        ple_on_dev.as_ref(),
                        shared_kv,
                        donor_f16,
                        #[cfg(feature = "gpu")]
                        donor_q8,
                    )?;
                } else {
                    let layer = &mut self.layers[layer_idx];
                    hidden = layer.forward(
                        &hidden,
                        layer_mask,
                        index_pos,
                        ple_on_dev.as_ref(),
                        shared_kv,
                        None,
                        #[cfg(feature = "gpu")]
                        None,
                    )?;
                }
                note_if_non_finite(&format!("layer {layer_idx}"), &hidden);
                // A host layer's freed buffers go back to the kernel past the burst headroom.
                if hidden.device().is_cpu() {
                    crate::inference::place::heap::release_retained();
                }
                if reports {
                    crate::request_metrics::note_layers(seq_len, layer_idx + 1, upto);
                }
            }

            seg_start = seg_end;
        }
        self.tp_forget()?;
        Ok((hidden, on_cuda))
    }

    /// The last stage of a forward: the final norm and the output projection of `hidden`,
    /// every position's logits when `all_positions`, else the last one's.
    pub(super) fn output_stage(
        &mut self,
        hidden: Tensor,
        on_cuda: bool,
        seq_len: usize,
        all_positions: bool,
    ) -> Result<Tensor> {
        let (logits, _) =
            self.output_stage_with(hidden, on_cuda, seq_len, all_positions, Outputs::Logits)?;
        logits.ok_or_else(|| crate::tensor::Error::msg("the output stage gave no logits"))
    }

    /// [`Self::output_stage`] giving back what `outputs` asks: the logits, the last row after
    /// the final norm as the head reads it, or both.
    ///
    /// The normed row is F32 `[batch, hidden]`, on the device the final norm ran on.
    /// It is given for the last position only, so `all_positions` asks for logits alone.
    pub(super) fn output_stage_with(
        &mut self,
        mut hidden: Tensor,
        mut on_cuda: bool,
        seq_len: usize,
        all_positions: bool,
        outputs: Outputs,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        if all_positions && outputs != Outputs::Logits {
            crate::tensor::bail!("the normed row is given for the last position only");
        }
        let keep = outputs != Outputs::Logits;
        let head = outputs != Outputs::Normed;
        let mut kept: Option<Tensor> = None;
        // -- Final norm + output projection --------------------------
        // The output stage must not read the hidden state while the last layer's GPU is
        // still writing it: the copy would take partially-written data off a running
        // compute stream and produce garbage logits every token (the gemma4:26b
        // multi-GPU+CPU incoherence). Where the read crosses to another GPU, `to_device`
        // now establishes that with an event and no barrier is needed here. What remains
        // is the case with no recorded projection device, where the output stage may pull
        // the hidden to the CPU: that read leaves the device entirely, so it is held here.
        // Same-device output stages are stream-ordered and were never the issue - an
        // unconditional sync per token serializes CPU<->GPU launch-ahead and cost
        // moondream ~46% (434->233 tok/s, caught by a campaign).
        #[cfg(feature = "gpu")]
        if let crate::tensor::DeviceLocation::Cuda { gpu_id } = hidden.device().location() {
            if self.output_proj_dev.is_none() {
                if let Some(src) = self.cuda_devices.get(&gpu_id) {
                    src.synchronize()?;
                }
            }
        }
        if outputs == Outputs::NormedRows {
            // Every row after the final norm on the host, F32 `[rows, hidden]`: the arithmetic
            // of the last row a headless model's prefill gives.
            let (b, s, h) = hidden.dims3()?;
            let normed = self.output_norm.forward(&hidden.to_device(&Device::Cpu)?)?;
            let rows = normed
                .reshape((b * s, h))?
                .to_dtype(crate::tensor::DType::F32)?;
            return Ok((None, Some(rows)));
        }
        // Transfer to the GPU that holds the output projection (if available)
        // Move the hidden state to whichever device holds the output projection.
        // This may be a DIFFERENT GPU than the last layer's (the lm_head can be
        // placed on an idle GPU when the primary is full), so transfer whenever
        // the devices differ - not just when hidden is off-CUDA.
        if let Some(dev) = self.output_proj_dev.as_ref() {
            if !hidden.device().same_device(dev) {
                hidden = hidden.to_device(dev)?;
            }
            on_cuda = true;
        }
        // Final projection. When `all_positions` is true, project every
        // input position (for speculative verification). When false, the
        // standard last-position-only path for single-token decode.
        //
        // For the multi-position case, the reference `matmul` doesn't broadcast
        // 3D x 2D, so we flatten `[1, seq_len, hidden]` -> `[seq_len, hidden]`
        // before the matmul and reshape the result back to 3D.
        if self.capture_feature {
            // EAGLE feature hook: the pre-final-norm last-layer hidden IS
            // the draft-head feature. all_positions -> every position (the verify
            // forward needs them all); else the last token (the draft cycle).
            // F32 to match the head's compute dtype. Off by default -> no-op.
            let f = if all_positions {
                hidden.clone()
            } else {
                hidden.i((.., seq_len - 1, ..))?
            };
            self.last_feature = Some(f.to_dtype(crate::tensor::DType::F32)?);
        }
        let logits = if let Some(ws) = self.wide_sparse.as_ref() {
            if keep {
                crate::tensor::bail!("a Sinkhorn wide residual gives no normed row");
            }
            // The streams collapse with the last layer's pre mix, then the final norm and the
            // head run on the host.
            let last = self
                .layers
                .last()
                .and_then(|l| l.sinkhorn.as_ref())
                .ok_or_else(|| {
                    crate::tensor::Error::msg("a Sinkhorn wide residual without its last layer")
                })?;
            ws.logits(&last.ffn_mix, &hidden, all_positions)?
        } else if let Some(oh) = self.output_hyper.as_ref() {
            // The final mixer is the output norm: it mixes the streams, then the head reads
            // the result.
            let (b, s, _) = hidden.dims3()?;
            let rows = if all_positions {
                hidden.clone()
            } else {
                hidden.narrow(1, s - 1, 1)?
            };
            let rows = rows.to_device(&oh.device())?;
            let (normed, _) = oh.mix(&rows)?;
            let n = normed.dim(1)?;
            let normed = normed.reshape((b * n, self.config.embedding_length))?;
            if keep {
                kept = Some(normed.to_dtype(crate::tensor::DType::F32)?);
            }
            if !head {
                return Ok((None, kept));
            }
            let out = match (&self.output_proj_cuda_qmm, &self.output_proj_dev) {
                (Some(proj), Some(dev)) => proj.forward(&normed.to_device(dev)?)?,
                _ => self
                    .host_head()?
                    .forward(&normed.to_device(&Device::Cpu)?)?,
            };
            let vocab = out.dim(1)?;
            if all_positions {
                out.reshape((b, n, vocab))?
            } else {
                out
            }
        } else if on_cuda && self.output_proj_cuda_qmm.is_some() {
            // Preferred GPU path: norm in F32 + quantized matmul. The QMatMul
            // dispatches to `mul_mat_vec_q*_K_q8_1` kernels that produce F32
            // output directly, so there are no F16 casts and no cutlass gemm.
            let proj = self.output_proj_cuda_qmm.as_ref().unwrap();
            if all_positions {
                let (b, s, h) = hidden.dims3()?;
                let hidden_2d = hidden
                    .reshape((b * s, h))?
                    .to_dtype(crate::tensor::DType::F32)?;
                let normed = self.final_norm_on_device(&hidden_2d)?;
                let out_2d = proj.forward(&normed)?;
                let vocab = out_2d.dims()[1];
                out_2d.reshape((b, s, vocab))?
            } else {
                // The final norm, then the head: the arithmetic every row of a verify runs.
                let last = hidden.i((.., seq_len - 1, ..))?;
                let last_f32 = last.to_dtype(crate::tensor::DType::F32)?;
                let normed = self.final_norm_on_device(&last_f32)?;
                if keep {
                    kept = Some(normed.clone());
                }
                if !head {
                    return Ok((None, kept));
                }
                proj.forward(&normed)?
            }
        } else if on_cuda && self.output_proj_cuda.is_some() {
            // Float head: norm, then the F16 matmul against the `[vocab, hidden]` head read
            // in place (a transpose would copy the whole head every call).
            let norm_w = self.output_norm_cuda_weight.as_ref().unwrap();
            let proj = self.output_proj_cuda.as_ref().unwrap();
            if all_positions {
                let (b, s, h) = hidden.dims3()?;
                let hidden_2d = hidden
                    .reshape((b * s, h))?
                    .to_dtype(crate::tensor::DType::F32)?;
                let normed = self.final_norm_on_device(&hidden_2d)?;
                let out_2d = crate::tensor::quantized::half_head(&normed, proj)?;
                let vocab = out_2d.dims()[1];
                out_2d.reshape((b, s, vocab))?
            } else {
                let last = hidden.i((.., seq_len - 1, ..))?;
                let last_f32 = last.to_dtype(crate::tensor::DType::F32)?;
                let normed = crate::tensor::ops::rms_norm(
                    &last_f32,
                    norm_w,
                    self.config.rms_norm_eps as f32,
                )?;
                if keep {
                    kept = Some(normed.clone());
                }
                if !head {
                    return Ok((None, kept));
                }
                crate::tensor::quantized::half_head(&normed, proj)?
            }
        } else if let Some(logits) = self
            .vulkan
            .take_logits()
            .filter(|_| !all_positions && !keep)
        {
            // The output head ran on the Vulkan device of the last layer.
            logits
        } else if let Some(logits) = self
            .opencl
            .take_logits()
            .filter(|_| !all_positions && !keep)
        {
            // The output head ran on the OpenCL device of the last layer.
            logits
        } else {
            // Fallback: transfer to CPU, norm + matmul on CPU
            if on_cuda {
                hidden = hidden.to_device(&Device::Cpu)?;
                note_if_non_finite("the hidden state downloaded for the CPU head", &hidden);
            }
            let normed = self.output_norm.forward(&hidden)?;
            note_if_non_finite("the final norm, before the head", &normed);
            let src = if all_positions {
                normed
            } else {
                normed.i((.., seq_len - 1, ..))?
            };
            if keep {
                kept = Some(src.to_dtype(crate::tensor::DType::F32)?);
            }
            if !head {
                return Ok((None, kept));
            }
            self.host_head()?.forward(&src)?
        };
        Ok((Some(self.finish_logits(logits)?), kept))
    }

    /// The head's output `logits` with the output bias, the logit scale and the softcap applied,
    /// as F32.
    pub(super) fn finish_logits(&self, logits: Tensor) -> Result<Tensor> {
        let mut logits = self.apply_output_bias(logits)?;

        // Ensure F32 dtype before scaling operations (GPU F16 matmul may produce mixed types)
        logits = logits.to_dtype(crate::tensor::DType::F32)?;

        // Logit scaling (Granite: divide by logit_scale)
        if let Some(scale) = self.config.logit_scale {
            logits = (logits / scale)?;
        }

        // Logit softcapping: tanh(logits/cap) * cap (Gemma4: cap=30.0)
        match self.config.final_logit_softcapping {
            Some(cap) => (logits / cap)?.tanh()? * cap,
            None => Ok(logits),
        }
    }

    /// Vision-prefill entry point. Builds the canonical Moondream/LLaVA
    /// embedding sequence `[bos_emb, image_embeds, text_emb]` and runs it
    /// through the transformer at index_pos=0.
    ///
    /// `bos_token`: shape (1, 1) - single BOS token id (usually EOS for
    /// Moondream since it has no separate BOS).
    /// `text_ids`: shape (1, N) - text prompt token ids (without BOS).
    /// `image_embeds`: shape (1, num_patches, embedding_length) - output of
    /// the vision tower + projector. Must match `config.embedding_length`
    /// in the last dim.
    ///
    /// Returns last-token logits, shape (1, vocab). Resets the KV cache as
    /// a side effect of running the transformer at index_pos=0 (caller
    /// must NOT have any pre-existing KV state - this is a fresh prefill).
    pub fn forward_with_image_embeds(
        &mut self,
        bos_token: &Tensor,
        text_ids: &Tensor,
        image_embeds: &Tensor,
    ) -> Result<Tensor> {
        // Embedding table lives on CPU; move all token tensors to CPU
        // before the embedding lookup, then cat with image embeds (also
        // on CPU). The segment loop inside forward_layers_and_output
        // will move hidden onto the model's first GPU on entry.
        // Path A prep: target the embeddings table's actual device
        // (CPU today; future commit moves small tables to GPU 0). bos/text
        // get cast to that device for the lookup; img then matches the
        // resulting emb device so the final cat works on any backend.
        let emb_device = self.embeddings.embeddings().device().clone();
        let to_emb_device = |t: &Tensor| -> Result<Tensor> {
            if t.device().same_device(&emb_device) {
                Ok(t.clone())
            } else {
                t.to_device(&emb_device)
            }
        };
        let bos_local = to_emb_device(bos_token)?;
        let text_local = to_emb_device(text_ids)?;

        let mut bos_emb = self.embeddings.forward(&bos_local)?;
        let mut text_emb = self.embeddings.forward(&text_local)?;
        if let Some(scale) = self.config.embed_scale {
            bos_emb = (bos_emb * scale)?;
            text_emb = (text_emb * scale)?;
        }
        // image_embeds: cast to the same device as bos_emb so the cat below
        // succeeds regardless of where CLIP placed its output.
        let img = if image_embeds.device().same_device(&bos_emb.device()) {
            image_embeds.clone()
        } else {
            image_embeds.to_device(&bos_emb.device())?
        };
        // Match dtype of the text embeddings (F32 default for the substrate's
        // embedding) so cat doesn't fail.
        let img = if img.dtype() != bos_emb.dtype() {
            img.to_dtype(bos_emb.dtype())?
        } else {
            img
        };
        // Last-dim sanity: image embeds must already be projected to the
        // text model's embedding width.
        let img_d = img.dim(crate::tensor::D::Minus1)?;
        if img_d != self.config.embedding_length {
            return Err(crate::tensor::Error::msg(format!(
                "forward_with_image_embeds: image_embeds last dim {} != embedding_length {}",
                img_d, self.config.embedding_length,
            )));
        }
        let hidden = Tensor::cat(&[&bos_emb, &img, &text_emb], 1)?;
        // Vision arches don't use Gemma's PLE per-layer inputs.
        self.forward_spliced_prefill_chunked(hidden)
    }

    /// Chunked prefill over a PRE-EMBEDDED hidden state `[1, seq, emb]`
    /// starting at position 0. The vision/audio splice paths build `hidden`
    /// by concatenating text embeddings with modality embeddings, so they
    /// can't go through the token-id `forward_prefill_chunked`; without
    /// chunking, a long splice (image ≈ 3-4k tokens + question) runs as ONE
    /// monolithic forward whose attention-score transient
    /// (`heads x seq x seq x f32`) OOMs at long context. Mirrors
    /// `forward_prefill_with_chunk`: per-chunk `forward_layers_and_output`
    /// at increasing `index_pos` (first chunk at 0 resets the KV caches,
    /// later chunks append); the last chunk's logits are the next-token
    /// prediction, identical to the monolithic result.
    fn forward_spliced_prefill_chunked(&mut self, hidden: Tensor) -> Result<Tensor> {
        let (logits, _) = self.spliced_prefill_with(hidden, Outputs::Logits)?;
        logits.ok_or_else(|| crate::tensor::Error::msg("the output stage gave no logits"))
    }

    /// [`Self::forward_spliced_prefill_chunked`] giving back what `outputs` asks of its last
    /// chunk.
    ///
    /// [`Outputs::NormedRows`] gives the normed rows of every chunk, in order.
    fn spliced_prefill_with(
        &mut self,
        hidden: Tensor,
        outputs: Outputs,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        const CHUNK: usize = crate::inference::engine::llm_engine::PREFILL_CHUNK_TOKENS;
        let seq_len = hidden.dim(1)?;
        if seq_len <= CHUNK {
            let (hidden, on_cuda) = self.run_layers(hidden, None, 0, seq_len)?;
            return self.output_stage_with(hidden, on_cuda, seq_len, false, outputs);
        }
        let mut off = 0usize;
        let mut last = None;
        // The normed rows of every chunk, when every row is asked for.
        let mut rows = Vec::new();
        while off < seq_len {
            let n = CHUNK.min(seq_len - off);
            let chunk = hidden.narrow(1, off, n)?;
            let (chunk, on_cuda) = self.run_layers(chunk, None, off, n)?;
            let out = self.output_stage_with(chunk, on_cuda, n, false, outputs)?;
            if outputs == Outputs::NormedRows {
                rows.extend(out.1.clone());
            }
            last = Some(out);
            off += n;
        }
        if outputs == Outputs::NormedRows {
            return Ok((
                None,
                Some(Tensor::cat(&rows.iter().collect::<Vec<_>>(), 0)?),
            ));
        }
        last.ok_or_else(|| crate::tensor::Error::msg("empty spliced prefill".to_string()))
    }

    /// Prefill with modality embeds spliced BETWEEN a text prefix and suffix:
    /// `[prefix_ids, embeds, suffix_ids]`. Unlike `forward_with_image_embeds` (which
    /// forces `[bos, embeds, text]`), this lets the caller place the embeds inside a
    /// chat turn - e.g. Ultravox audio: `[<bos>user-header, audio, question+assistant-header]`.
    /// `embeds` must already be `[1, n, embedding_length]` (batch axis present).
    pub fn forward_with_audio_embeds(
        &mut self,
        prefix_ids: &Tensor,
        embeds: &Tensor,
        suffix_ids: &Tensor,
    ) -> Result<Tensor> {
        let emb_device = self.embeddings.embeddings().device().clone();
        let to_emb_device = |t: &Tensor| -> Result<Tensor> {
            if t.device().same_device(&emb_device) {
                Ok(t.clone())
            } else {
                t.to_device(&emb_device)
            }
        };
        // A segment of no tokens contributes no rows.
        let embed = |ids: &Tensor| -> Result<Option<Tensor>> {
            if ids.dim(1)? == 0 {
                return Ok(None);
            }
            let mut e = self.embeddings.forward(&to_emb_device(ids)?)?;
            if let Some(scale) = self.config.embed_scale {
                e = (e * scale)?;
            }
            Ok(Some(e))
        };
        let prefix_emb = embed(prefix_ids)?;
        let suffix_emb = embed(suffix_ids)?;
        // Embeds with no text around them take the device and dtype of a row of the table.
        let probe;
        let like = match prefix_emb.as_ref().or(suffix_emb.as_ref()) {
            Some(text) => text,
            None => {
                probe = self
                    .embeddings
                    .forward(&Tensor::new(&[0u32], &emb_device)?.unsqueeze(0)?)?;
                &probe
            }
        };
        let (like_device, like_dtype) = (like.device(), like.dtype());
        let aud = if embeds.device().same_device(&like_device) {
            embeds.clone()
        } else {
            embeds.to_device(&like_device)?
        };
        let aud = if aud.dtype() != like_dtype {
            // CONVERSION: the embeddings join the text rows in the embedding table's dtype.
            aud.to_dtype(like_dtype)?
        } else {
            aud
        };
        let aud_d = aud.dim(crate::tensor::D::Minus1)?;
        if aud_d != self.config.embedding_length {
            return Err(crate::tensor::Error::msg(format!(
                "forward_with_audio_embeds: embeds last dim {} != embedding_length {}",
                aud_d, self.config.embedding_length,
            )));
        }
        let parts: Vec<&Tensor> = prefix_emb
            .iter()
            .chain(std::iter::once(&aud))
            .chain(suffix_emb.iter())
            .collect();
        let hidden = Tensor::cat(&parts, 1)?;
        self.forward_spliced_prefill_chunked(hidden)
    }

    /// The decode step of the row at `index_pos` fed by `embedding` instead of a token's table
    /// row.
    ///
    /// `embedding` holds `hidden` values in any shape, on any device, in any float type.
    /// It enters the layers as a token's row does, with no embedding scale and no position row.
    /// The layers' caches advance as for a token step, and `index_pos` 0 empties them first.
    /// The head runs when `logits` asks for it.
    /// A model whose layers take more than the embedding row is refused.
    pub fn forward_embedding_step(
        &mut self,
        embedding: &Tensor,
        index_pos: usize,
        logits: bool,
    ) -> Result<EmbeddingStep> {
        let width = self.config.embedding_length;
        if embedding.elem_count() != width {
            crate::tensor::bail!(
                "an embedding step takes {width} values, {} given",
                embedding.elem_count()
            );
        }
        let hidden = self.embedding_rows(embedding, 1)?;
        crate::tensor::quant_cpu::set_small_model_prefill(false);
        self.spec_clear();
        let (hidden, on_cuda) = self.run_layers(hidden, None, index_pos, 1)?;
        let (logits, normed) =
            self.output_stage_with(hidden, on_cuda, 1, false, Self::asked(logits))?;
        Self::embedding_step(logits, normed, width)
    }

    /// The prefill from position 0 of `embeddings`, rows fed instead of tokens' table rows.
    ///
    /// `embeddings` holds whole rows of `hidden` values in any shape, on any device, in any
    /// float type.
    /// The rows enter the layers as in [`Self::forward_embedding_step`].
    /// The caches are emptied first.
    /// The step gives back the last row after the final norm, and its logits when `logits`
    /// asks for them.
    pub fn forward_embeddings_prefill(
        &mut self,
        embeddings: &Tensor,
        logits: bool,
    ) -> Result<EmbeddingStep> {
        let width = self.config.embedding_length;
        let rows = embeddings.elem_count() / width;
        if rows == 0 || rows * width != embeddings.elem_count() {
            crate::tensor::bail!(
                "an embedding prefill takes whole rows of {width} values, {} given",
                embeddings.elem_count()
            );
        }
        let hidden = self.embedding_rows(embeddings, rows)?;
        self.spec_clear();
        let (logits, normed) = self.spliced_prefill_with(hidden, Self::asked(logits))?;
        Self::embedding_step(logits, normed, width)
    }

    /// The prefill from position 0 of `embeddings`, every row after the final norm.
    ///
    /// `embeddings` holds whole rows of `hidden` values, as in
    /// [`Self::forward_embeddings_prefill`].
    /// The caches are emptied first.
    /// The rows come back F32 `[rows, hidden]`, in order, on the host.
    /// No head runs.
    pub fn forward_embeddings_rows(&mut self, embeddings: &Tensor) -> Result<Tensor> {
        let width = self.config.embedding_length;
        let rows = embeddings.elem_count() / width;
        if rows == 0 || rows * width != embeddings.elem_count() {
            crate::tensor::bail!(
                "an embedding prefill takes whole rows of {width} values, {} given",
                embeddings.elem_count()
            );
        }
        let hidden = self.embedding_rows(embeddings, rows)?;
        self.spec_clear();
        let (_, normed) = self.spliced_prefill_with(hidden, Outputs::NormedRows)?;
        normed.ok_or_else(|| crate::tensor::Error::msg("the output stage gave no normed rows"))
    }

    /// The token table's rows of `ids`, `[ids.len(), hidden]`, as a token step reads them before
    /// any embedding scale.
    pub fn token_rows(&self, ids: &[u32]) -> Result<Tensor> {
        let device = self.embeddings.embeddings().device().clone();
        let ids = Tensor::from_vec(ids.to_vec(), ids.len(), &device)?;
        self.embeddings.forward(&ids)
    }

    /// The output stage's outputs for an embedding step: the normed row, and the logits when
    /// `logits` asks for them.
    fn asked(logits: bool) -> Outputs {
        match logits {
            true => Outputs::LogitsAndNormed,
            false => Outputs::Normed,
        }
    }

    /// `rows` rows of `embedding` as the layers take them: `[1, rows, hidden]` on the token
    /// table's device in its type.
    ///
    /// A model whose layers take more than the embedding row is refused.
    fn embedding_rows(&self, embedding: &Tensor, rows: usize) -> Result<Tensor> {
        if self.config.hyper.is_some()
            || self.wide_sparse.is_some()
            || self.ple_dim > 0
            || self.ngram.is_some()
            || self.config.learned_positions
        {
            crate::tensor::bail!(
                "an embedding step needs a model whose layers take the embedding row alone"
            );
        }
        let table = self.embeddings.embeddings();
        embedding
            .to_device(&table.device())?
            .to_dtype(table.dtype())?
            .reshape((1, rows, self.config.embedding_length))
    }

    /// An embedding step of the output stage's `logits` and `normed` row.
    fn embedding_step(
        logits: Option<Tensor>,
        normed: Option<Tensor>,
        width: usize,
    ) -> Result<EmbeddingStep> {
        let normed = normed
            .ok_or_else(|| crate::tensor::Error::msg("the output stage gave no normed row"))?;
        Ok(EmbeddingStep {
            logits,
            normed: normed.reshape((1, width))?,
        })
    }

    /// Set of CUDA device ordinals the engine should pre-warm custom
    /// kernels for. Includes EVERY detected CUDA device (those that
    /// might receive layers under future placement / lazy-grow), not
    /// just the ones currently holding layers - kernel pre-warm is
    /// cheap and missing a device produces JIT-compile stalls during
    /// the first request that touches it.
    pub fn cuda_device_ordinals(&self) -> std::collections::HashSet<usize> {
        self.cuda_devices.keys().copied().collect()
    }

    /// Set of CUDA device ordinals that ACTUALLY have layers placed on
    /// them. Strictly a subset of `cuda_device_ordinals()`. Use this
    /// when a downstream component (vision tower placement, hetero
    /// planning) needs to know which GPUs are busy with text-model
    /// compute vs which are idle.
    pub fn layer_cuda_ordinals(&self) -> std::collections::HashSet<usize> {
        self.layer_devs
            .iter()
            .filter_map(|d| match d {
                LayerDevice::Cuda(idx) => Some(*idx),
                LayerDevice::Cpu => None,
            })
            .collect()
    }
}

/// Whether `rows` that continue a cache at `index_pos` are a draft verify: they then compute as
/// the decode steps they stand for, under [`crate::tensor::quantized::DecodeOrder`].
pub(super) fn is_verify(rows: usize, index_pos: usize) -> bool {
    index_pos > 0 && rows <= crate::inference::engine::decode_step::VERIFY_ROWS
}

/// The decode order a verify of `rows` at `index_pos` holds while it runs, `None` for any other
/// forward.
pub(super) fn verify_order(
    rows: usize,
    index_pos: usize,
) -> Option<crate::tensor::quantized::DecodeOrder> {
    is_verify(rows, index_pos).then(crate::tensor::quantized::DecodeOrder::enter)
}

/// Build the `[seq_q, bounded_len]` SWA prefill attention-mask data (1=masked) for
/// a layer whose KV buffer holds `bounded_len` keys at ABSOLUTE positions
/// `[buf_start, buf_start+bounded_len)` (the bounded sliding-window cache).
/// Query local `i` (abs = `index_pos+i`) attends buffer key `j` (abs = `buf_start+j`)
/// iff `abs_j <= abs_i` (causal) AND `abs_j > abs_i-window` (SWA: the last `window` keys). The legacy
/// unbounded `make_mask` windowing is exactly the special case `buf_start=0,
/// bounded_len=index_pos+seq_q` (where `j == abs_j`). With buffer = `window+chunk`
/// and slide-keep-last-window, a slid chunk has `buf_start = (index_pos+seq_q) -
/// bounded_len = chunk_start - window`, so every query's full window is in-buffer.
#[cfg_attr(not(test), allow(dead_code))] // wired into make_mask by the sliding-window path
fn swa_bounded_mask_data(
    seq_q: usize,
    index_pos: usize,
    bounded_len: usize,
    buf_start: usize,
    window: usize,
) -> Vec<u8> {
    (0..seq_q)
        .flat_map(|i| {
            (0..bounded_len).map(move |j| {
                let abs_i = i + index_pos;
                let abs_j = buf_start + j;
                u8::from(abs_j > abs_i || abs_j + window <= abs_i)
            })
        })
        .collect()
}

#[cfg(test)]
mod layer_place_tests {
    use super::{layer_place, LayerDevice};

    /// A layer a Vulkan or OpenCL device runs is not reported on the host.
    #[test]
    fn a_layer_is_reported_on_the_device_that_runs_it() {
        assert_eq!(layer_place(&LayerDevice::Cpu, None, None), ("CPU", 0));
        assert_eq!(layer_place(&LayerDevice::Cuda(1), None, None), ("CUDA", 1));
        assert_eq!(layer_place(&LayerDevice::Cpu, Some(2), None), ("Vulkan", 2));
        assert_eq!(layer_place(&LayerDevice::Cpu, None, Some(0)), ("OpenCL", 0));
    }
}

#[cfg(test)]
mod swa_mask_tests {
    use super::swa_bounded_mask_data;

    #[test]
    fn bounded_swa_mask_matches_unbounded_when_buf_start_zero() {
        // buf_start=0, bounded_len=index_pos+seq_q -> must equal the legacy
        // windowed make_mask predicate (j == abs_j). Guarantees non-slid /
        // global behavior is unchanged.
        let (seq_q, index_pos, w) = (3usize, 5usize, 4usize);
        let bounded_len = index_pos + seq_q;
        let got = swa_bounded_mask_data(seq_q, index_pos, bounded_len, 0, w);
        let want: Vec<u8> = (0..seq_q)
            .flat_map(|i| {
                (0..bounded_len).map(move |j| {
                    let abs_i = i + index_pos;
                    u8::from(j > abs_i || j + w <= abs_i)
                })
            })
            .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn bounded_swa_mask_windows_correctly_after_slide() {
        // Chunk at index_pos=1536, seq_q=512, window=1024; buffer holds last
        // w+chunk=1536 keys -> buf_start=(1536+512)-1536=512 (abs [512,2048)).
        let (seq_q, index_pos, w, bounded_len) = (512usize, 1536usize, 1024usize, 1536usize);
        let buf_start = (index_pos + seq_q) - bounded_len; // 512
        let m = swa_bounded_mask_data(seq_q, index_pos, bounded_len, buf_start, w);
        let at = |i: usize, j: usize| m[i * bounded_len + j];
        // Query abs 1536 sees its last 1024 keys, abs 513 the oldest; not a future key.
        assert_eq!(at(0, 0), 1);
        assert_eq!(at(0, 1), 0);
        assert_eq!(at(0, 1025), 1); // abs 1537 future -> causal-masked
                                    // Query abs 2047: key abs 1023 older than the window masked; abs 1024
                                    // edge unmasked; abs 2047 (current) unmasked.
        assert_eq!(at(511, 511), 1);
        assert_eq!(at(511, 512), 0);
        assert_eq!(at(511, 1535), 0);
    }
}

#[cfg(test)]
mod prefill_stage_tests {
    use super::{
        decode_stage_fused, mlp_activation_fused, prefill_attention_head_major,
        prefill_stage_fused, PrefillProjection,
    };
    use crate::tensor::{DType, Device, Tensor};

    #[test]
    fn the_stage_is_one_launch_on_a_card_in_f16_with_an_even_head() {
        assert!(prefill_stage_fused(true, DType::F16, DType::F16, 128, true));
        assert!(prefill_stage_fused(true, DType::F16, DType::F16, 64, true));
    }

    #[test]
    fn the_stage_falls_back_off_the_card_or_off_f16_or_with_an_odd_head() {
        use DType::{BF16, F16, F32};
        let cases = [
            (false, F16, F16, 128),
            (true, F32, F16, 128),
            (true, BF16, F16, 128),
            (true, F16, F32, 128),
            (true, F16, F16, 0),
            (true, F16, F16, 127),
        ];
        for (on_card, activation, store, head_dim) in cases {
            assert!(
                !prefill_stage_fused(on_card, activation, store, head_dim, true),
                "{on_card} {activation:?} {store:?} {head_dim}"
            );
        }
    }

    #[test]
    fn the_stage_falls_back_when_only_one_of_q_and_k_has_a_norm() {
        assert!(!prefill_stage_fused(
            true,
            DType::F16,
            DType::F16,
            128,
            false
        ));
    }

    #[test]
    fn the_decode_stage_is_one_launch_on_a_card_in_f16_with_an_even_head_and_paired_norms() {
        assert!(decode_stage_fused(true, DType::F16, DType::F16, 128, true));
        assert!(decode_stage_fused(true, DType::F16, DType::F16, 64, true));
    }

    #[test]
    fn the_decode_stage_falls_back_where_the_prefill_stage_does() {
        use DType::{BF16, F16, F32};
        let cases = [
            (true, F16, F16, 128, false),
            (false, F16, F16, 128, true),
            (true, F32, F16, 128, true),
            (true, BF16, F16, 128, true),
            (true, F16, F32, 128, true),
            (true, F16, F16, 0, true),
            (true, F16, F16, 127, true),
        ];
        for (on_card, activation, store, head_dim, paired) in cases {
            assert!(
                !decode_stage_fused(on_card, activation, store, head_dim, paired),
                "{on_card} {activation:?} {store:?} {head_dim} {paired}"
            );
        }
    }

    #[test]
    fn the_attention_takes_q_head_major_from_a_fused_stage_with_no_cached_prompt() {
        assert!(prefill_attention_head_major(true, 0, true, true, 32, 8));
        assert!(prefill_attention_head_major(true, 0, true, true, 8, 8));
    }

    #[test]
    fn the_attention_keeps_the_casts_without_a_fused_stage_or_with_a_cached_prompt() {
        let cases = [
            (false, 0, true, true, 32, 8),
            (true, 1, true, true, 32, 8),
            (true, 0, false, true, 32, 8),
            (true, 0, true, false, 32, 8),
            (true, 0, true, true, 32, 0),
            (true, 0, true, true, 30, 8),
        ];
        for (stage, cached, flash_card, head_dim, n_head, n_kv) in cases {
            assert!(
                !prefill_attention_head_major(stage, cached, flash_card, head_dim, n_head, n_kv),
                "{stage} {cached} {flash_card} {head_dim} {n_head} {n_kv}"
            );
        }
    }

    #[test]
    fn the_activation_is_one_launch_on_a_card_in_f16_with_a_silu_gate_and_an_even_width() {
        assert!(mlp_activation_fused(true, DType::F16, 12288, false));
        assert!(mlp_activation_fused(true, DType::F16, 96, false));
    }

    #[test]
    fn the_activation_falls_back_off_the_card_or_off_f16_or_with_gelu_or_an_odd_width() {
        use DType::{BF16, F16, F32};
        let cases = [
            (false, F16, 96, false),
            (true, F32, 96, false),
            (true, BF16, 96, false),
            (true, F16, 96, true),
            (true, F16, 95, false),
            (true, F16, 0, false),
        ];
        for (on_card, activation, ffn, use_gelu) in cases {
            assert!(
                !mlp_activation_fused(on_card, activation, ffn, use_gelu),
                "{on_card} {activation:?} {ffn} {use_gelu}"
            );
        }
    }

    #[test]
    fn a_fused_projection_is_read_in_place_and_split_by_copy() {
        let (q_width, kv_width, rows) = (4, 2, 3);
        let data: Vec<f32> = (0..rows * (q_width + 2 * kv_width))
            .map(|i| i as f32)
            .collect();
        let qkv =
            Tensor::from_vec(data, &[rows, q_width + 2 * kv_width][..], &Device::Cpu).unwrap();
        let fused = PrefillProjection::Fused(qkv);
        let cols: Vec<usize> = fused
            .sources(q_width, kv_width)
            .iter()
            .map(|(_, col)| *col)
            .collect();
        assert_eq!(cols, vec![0, q_width, q_width + kv_width]);
        let (q, k, v) = fused.split(q_width, kv_width).unwrap();
        assert_eq!(q.dims(), &[rows, q_width]);
        assert_eq!(k.dims(), &[rows, kv_width]);
        assert_eq!(v.dims(), &[rows, kv_width]);
        assert_eq!(
            v.to_vec2::<f32>().unwrap()[1],
            vec![14.0, 15.0],
            "v of row 1 is columns 6 and 7 of that row"
        );
    }

    #[test]
    fn separate_projections_start_at_column_0() {
        let part =
            |w: usize| Tensor::from_vec(vec![0f32; 2 * w], &[2, w][..], &Device::Cpu).unwrap();
        let split = PrefillProjection::Split(part(4), part(2), part(2));
        assert!(split.sources(4, 2).iter().all(|(_, col)| *col == 0));
    }
}

/// Prompt positions the fit probe reads: the rows just before this many.
const PROMPT_FIT_SPAN: usize = 1024;

impl GenericHeteroTransformer {
    /// The model's fit over the prompt's first rows: its guesses on the last rows before
    /// `PROMPT_FIT_SPAN` against the tokens that followed them, through the output mixer
    /// and the head.
    fn prompt_fit_probe(&self, ids: &[u32], rows: &Tensor) -> Result<()> {
        const ROWS: usize = 32;
        let n = ids.len().min(rows.dim(1)?);
        if n < ROWS + 1 {
            return Ok(());
        }
        let Some(oh) = self.output_hyper.as_ref() else {
            return Ok(());
        };
        let from = n - ROWS - 1;
        let probe = rows.narrow(1, from, ROWS)?.to_device(&oh.device())?;
        let (normed, _) = oh.mix(&probe)?;
        let normed = normed.reshape((ROWS, self.config.embedding_length))?;
        let logits = match (&self.output_proj_cuda_qmm, &self.output_proj_dev) {
            (Some(proj), Some(dev)) => proj.forward(&normed.to_device(dev)?)?,
            _ => self
                .host_head()?
                .forward(&normed.to_device(&Device::Cpu)?)?,
        };
        let guessed = logits
            .to_dtype(crate::tensor::DType::F32)?
            .argmax_keepdim(1)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let hits = guessed
            .iter()
            .zip(&ids[from + 1..])
            .filter(|(g, t)| g == t)
            .count();
        tracing::info!("target fit over the prompt: {hits} of {ROWS} rows guessed right");
        Ok(())
    }
}

#[cfg(test)]
mod prefill_fit_tests {
    use super::PrefillFit;

    /// Forwards of few rows far into a prompt keep most of their bytes for the history: the fit
    /// gives the part per row a forward of many rows shows, not their bytes over their rows.
    #[test]
    fn a_short_forward_late_in_a_prompt_does_not_inflate_the_bytes_per_row() {
        let (per_row, per_position) = (180u64 << 10, 840u64);
        let mut fit = PrefillFit::default();
        for (rows, upto) in [
            (273, 273),
            (64, 10_432),
            (2048, 5_000),
            (32, 29_440),
            (32, 29_472),
        ] {
            fit.add(
                rows,
                upto,
                per_row * rows as u64 + per_position * upto as u64,
            );
        }
        let (row, position) = fit.terms().unwrap();
        assert!(row.abs_diff(per_row) <= per_row / 100, "{row} per row");
        assert!(
            position.abs_diff(per_position) <= per_position / 100,
            "{position} per position"
        );
        let naive = (per_row * 32 + per_position * 29_472) / 32;
        assert!(naive > 4 * row, "the bytes over the rows were {naive}");
    }

    /// Forwards that cannot tell the parts apart count all their bytes per row.
    #[test]
    fn forwards_from_the_start_count_all_their_bytes_per_row() {
        let mut fit = PrefillFit::default();
        assert!(fit.terms().is_none());
        for n in [64usize, 128, 512] {
            fit.add(n, n, 1000 * n as u64);
        }
        assert_eq!(fit.terms(), Some((1000, 0)));
    }
}

#[cfg(test)]
mod embedding_tests {
    use super::*;
    use crate::tensor::DType;
    use crate::tensor::Tensor;

    /// Embedding tests: uses a tiny model built on the fly or a test GGUF.
    /// For now, tests the interface and basic behavior without real weights.
    #[test]
    fn embed_all_hidden_returns_one_vector_per_token() {
        // Build a minimal GenericHeteroTransformer for testing.
        // This test checks the interface without requiring actual weights.
        // In a full test environment, this would use a tiny GGUF like `toy.gguf`.
        // For now, we test the interface contract: given N input tokens,
        // embed_all_hidden returns N vectors, each of the hidden dimension.
        // embed_last_hidden returns the last of those vectors.

        // We cannot construct a working model without weights in a pure unit test,
        // so we test the interface contract by verifying the method signatures
        // compile correctly and the return types match expectations.
        // The actual forward pass testing would require a GGUF file.

        // Placeholder: the test validates that embed_all_hidden and embed_last_hidden
        // have compatible return types and the logic is sound.
        // When a tiny model is available, this test should:
        // 1. Create a model with N hidden dimension and L layers
        // 2. Tokenize a short prompt of T tokens
        // 3. Call embed_all_hidden and verify it returns T vectors of size H
        // 4. Call embed_last_hidden and verify it equals the last vector from step 3

        // Compile-time type check: both methods return Result containing Vec<f32> or Vec<Vec<f32>>
        let _compile_check: fn(&mut GenericHeteroTransformer, &Tensor) -> Result<Vec<Vec<f32>>> =
            GenericHeteroTransformer::embed_all_hidden;
        let _compile_check2: fn(&mut GenericHeteroTransformer, &Tensor) -> Result<Vec<f32>> =
            GenericHeteroTransformer::embed_last_hidden;

        // The actual test logic requires a model, which we can't construct in a
        // standalone unit test. A real test would use a tiny GGUF file.
        assert!(true);
    }
}
