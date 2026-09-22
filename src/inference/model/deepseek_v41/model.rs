//! Full-model assembly for deepseek_v41 - the step that turns the ported mechanisms into a forward.
//!
//! embed -> expand to hc_mult copies -> blocks (each in its band's attention mode) -> collapse the
//! copies -> final norm -> head -> logits. Prefill only, and only ratio-0 and self-sourcing band
//! layers so far: a reader layer (a candidate consumer that reads another layer's compressed KV and
//! index) needs the cross-layer shared state the streamed engine will carry, which comes next.

use super::band::SharedAttn;
use super::block::Block;
use super::cache::DecodeState;
use super::engram::{prefill_hashes, Engram, EngramConfig};
use super::hyper_connections::hc_pre;
use super::load::{load_band_block, load_ratio0_block, projection};
use super::source::WeightSource;
use super::DeepseekV41Config;
use crate::inference::offload::projection::Projection;
use crate::tensor::ops::rms_norm;
use crate::tensor::{Device, Error, Result, Tensor};

/// One layer's rope frequencies (already YaRN-adjusted), as cos and sin tables [seqlen, rd/2].
pub(crate) fn rope_table(
    rd: usize,
    seqlen: usize,
    original_seq_len: usize,
    base: f32,
    factor: f32,
) -> Result<(Tensor, Tensor)> {
    use std::f32::consts::PI;
    let half = rd / 2;
    let mut freqs = vec![0f32; half];
    for (i, f) in freqs.iter_mut().enumerate() {
        *f = 1.0 / base.powf((2 * i) as f32 / rd as f32);
    }
    // YaRN: below original_seq_len keep the frequency, far beyond it divide by `factor`, fade the
    // band between beta_fast=32 and beta_slow=1 across a linear ramp. Off when orig == 0.
    if original_seq_len > 0 {
        let corrected = |rot: f32| {
            rd as f32 * (original_seq_len as f32 / (rot * 2.0 * PI)).ln() / (2.0 * base.ln())
        };
        let low = corrected(32.0).floor().max(0.0);
        let high = corrected(1.0).ceil().min((rd - 1) as f32);
        for (i, f) in freqs.iter_mut().enumerate() {
            let ramp = ((i as f32 - low) / (high - low).max(1e-3)).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *f = *f / factor * (1.0 - smooth) + *f * smooth;
        }
    }
    let mut cos = vec![0f32; seqlen * half];
    let mut sin = vec![0f32; seqlen * half];
    for t in 0..seqlen {
        for i in 0..half {
            let a = t as f32 * freqs[i];
            cos[t * half + i] = a.cos();
            sin[t * half + i] = a.sin();
        }
    }
    Ok((
        Tensor::from_vec(cos, (seqlen, half), &Device::Cpu)?,
        Tensor::from_vec(sin, (seqlen, half), &Device::Cpu)?,
    ))
}

pub struct DeepseekV41Model {
    embed: Tensor,      // [vocab, dim]
    output: Projection, // [vocab, dim]
    norm: Tensor,       // [dim]
    blocks: Vec<Block>,
    /// Per layer, the engram applied to the stream before that block, when the layer has one.
    engrams: Vec<Option<Engram>>,
    engram_cfg: Option<EngramConfig>,
    ratios: Vec<usize>,
    /// Per layer, experts from the most routed-to on the calibration corpus downwards.
    hot_experts: Vec<Vec<usize>>,
    hc_mult: usize,
    dim: usize,
    rope_head_dim: usize,
    rms_eps: f32,
    // rope regimes: ratio-0 uses base theta with no YaRN; a band layer uses the compressed theta
    // with YaRN from original_seq_len.
    rope_theta: f32,
    compress_rope_theta: f32,
    rope_factor: f32,
    original_seq_len: usize,
}

/// A prefill between two layers; see `DeepseekV41Model::prefill_begin`.
pub struct PrefillState {
    tokens: Vec<u32>,
    /// [1, s, hc, dim]
    x: Tensor,
    pre_mix: Vec<Vec<f32>>,
    shared: SharedAttn,
    /// The rope tables of the plain layers and of the compressed ones.
    ropes: [(Tensor, Tensor); 2],
    /// The layer to run next.
    next: usize,
}

impl DeepseekV41Model {
    /// Load the resident model from a GGUF source - one mapped file or a split set. Each layer is
    /// loaded in its band's mode.
    pub fn load<S: WeightSource + ?Sized>(g: &S, cfg: &DeepseekV41Config) -> Result<Self> {
        let embed = g.dense_f32("embed.weight")?;
        let norm = g.dense_f32("norm.weight")?;
        // The head is its own tensor when present, otherwise tied to the embedding.
        let output = if g.shape("head.weight").is_some() {
            projection(g, "head.weight")?
        } else {
            Projection::Dense(embed.clone())
        };
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        let mut engrams = Vec::with_capacity(cfg.n_layers);
        for layer in 0..cfg.n_layers {
            let ratio = cfg.compress_ratios.get(layer).copied().unwrap_or(0);
            blocks.push(if ratio == 0 {
                load_ratio0_block(g, layer, cfg)?
            } else {
                load_band_block(g, layer, cfg)?
            });
            let has_engram = cfg
                .engram
                .as_ref()
                .is_some_and(|e| e.layer_ids.contains(&layer));
            engrams.push(if has_engram {
                Some(Engram::load(
                    g,
                    layer,
                    cfg.d_model,
                    cfg.hc_mult,
                    cfg.rms_eps as f32,
                )?)
            } else {
                None
            });
        }
        Ok(Self {
            embed,
            output,
            norm,
            blocks,
            engrams,
            engram_cfg: cfg.engram.clone(),
            ratios: cfg.compress_ratios.clone(),
            hot_experts: cfg.hot_experts.clone(),
            hc_mult: cfg.hc_mult,
            dim: cfg.d_model,
            rope_head_dim: cfg.rope_head_dim,
            rms_eps: cfg.rms_eps as f32,
            rope_theta: cfg.rope_theta,
            compress_rope_theta: cfg.compress_rope_theta,
            rope_factor: cfg.rope_factor,
            original_seq_len: cfg.original_seq_len,
        })
    }

    /// Prefill: token ids in, logits [seqlen, vocab] out.
    pub fn forward_prefill(&self, tokens: &[u32]) -> Result<Tensor> {
        let mut state = self.prefill_begin(tokens)?;
        for layer in 0..self.blocks.len() {
            self.prefill_layer(layer, &mut state)?;
        }
        self.prefill_finish(state)
    }

    /// A prefill stopped between layers: the embedding taken, the layers before `state.next` run.
    /// `forward_prefill` is `prefill_begin`, `prefill_layer` for each layer in order, then
    /// `prefill_finish`; a caller may interleave the layers of several prefills.
    pub fn prefill_begin(&self, tokens: &[u32]) -> Result<PrefillState> {
        let s = tokens.len();
        let (dim, hc) = (self.dim, self.hc_mult);

        // Embedding lookup: gather the token rows.
        let rows: Vec<Tensor> = tokens
            .iter()
            .map(|&t| self.embed.narrow(0, t as usize, 1))
            .collect::<Result<_>>()?;
        let refs: Vec<&Tensor> = rows.iter().collect();
        let h = Tensor::cat(&refs, 0)?.reshape((1, s, dim))?;
        // Expand to hc_mult identical copies for the hyper-connection stream.
        let x = h
            .unsqueeze(2)?
            .broadcast_as((1, s, hc, dim))?
            .contiguous()?;

        // The two rope regimes, built at this sequence length.
        let (cos0, sin0) = rope_table(self.rope_head_dim, s, 0, self.rope_theta, self.rope_factor)?;
        let (cosb, sinb) = rope_table(
            self.rope_head_dim,
            s,
            self.original_seq_len,
            self.compress_rope_theta,
            self.rope_factor,
        )?;

        // Identity pre-mix into the first layer (one-hot on copy 0).
        let pre_mix: Vec<Vec<f32>> = (0..s)
            .map(|_| {
                let mut v = vec![0f32; hc];
                v[0] = 1.0;
                v
            })
            .collect();

        Ok(PrefillState {
            tokens: tokens.to_vec(),
            x,
            pre_mix,
            // The cross-layer state sources publish and readers consume, fresh for this forward.
            shared: SharedAttn::default(),
            ropes: [(cos0, sin0), (cosb, sinb)],
            next: 0,
        })
    }

    /// Run layer `i` of a prefill; the layers before it must have run.
    pub fn prefill_layer(&self, i: usize, state: &mut PrefillState) -> Result<()> {
        if state.next != i || i >= self.blocks.len() {
            return Err(Error::msg(format!(
                "prefill: layer {i} asked for, layer {} is next of {}",
                state.next,
                self.blocks.len()
            )));
        }
        if let (Some(engram), Some(ecfg)) = (&self.engrams[i], &self.engram_cfg) {
            let which = ecfg.layer_ids.iter().position(|&l| l == i).unwrap_or(0);
            state.x = engram.forward(&state.x, &prefill_hashes(ecfg, which, &state.tokens))?;
        }
        let (cos, sin) = if self.ratios.get(i).copied().unwrap_or(0) == 0 {
            (&state.ropes[0].0, &state.ropes[0].1)
        } else {
            (&state.ropes[1].0, &state.ropes[1].1)
        };
        let (nx, npm) = self.blocks[i].forward_prefill(
            &state.x,
            &state.pre_mix,
            cos,
            sin,
            &mut state.shared,
        )?;
        state.x = nx;
        state.pre_mix = npm;
        state.next += 1;
        Ok(())
    }

    /// Prefill `tokens` into a fresh decode state, which is left where `forward_decode` over the
    /// same tokens would leave it: every layer's window, compressed KV, index keys and filling group,
    /// the engram's recent ids, the position. Returns the last position's logits [1, vocab]. Runs
    /// each layer once over every token instead of every layer once per token.
    pub fn prefill_into(&self, tokens: &[u32], state: &mut DecodeState) -> Result<Tensor> {
        if state.pos != 0 || tokens.is_empty() {
            return Err(Error::msg(format!(
                "prefill_into: {} tokens into a state at position {}; a batched prefill starts a sequence",
                tokens.len(),
                state.pos
            )));
        }
        let mut prefill = self.prefill_begin(tokens)?;
        for (i, block) in self.blocks.iter().enumerate() {
            // Each layer's own time, for a recorder that keeps them apart rather than summed.
            let layer_started = std::time::Instant::now();
            if let (Some(engram), Some(ecfg)) = (&self.engrams[i], &self.engram_cfg) {
                let which = ecfg.layer_ids.iter().position(|&l| l == i).unwrap_or(0);
                prefill.x = crate::inference::offload::stage("engram", || {
                    engram.forward(&prefill.x, &prefill_hashes(ecfg, which, tokens))
                })?;
            }
            let rope = if self.ratios.get(i).copied().unwrap_or(0) == 0 {
                0
            } else {
                1
            };
            let (cos, sin) = (&prefill.ropes[rope].0, &prefill.ropes[rope].1);
            let (mid, mix) = block.forward_prefill_attn(
                &prefill.x,
                &prefill.pre_mix,
                cos,
                sin,
                &mut prefill.shared,
                Some(&mut state.layers[i]),
            )?;
            let (nx, npm) = block.forward_prefill_ffn(&mid, &mix)?;
            if let Some(off) = crate::inference::offload::current() {
                off.record("prefill layer", layer_started.elapsed().as_nanos() as u64);
            }
            prefill.x = nx;
            prefill.pre_mix = npm;
            prefill.next += 1;
        }
        // Once more over every layer now that every read has landed: the read-ahead a layer's
        // sweep could not see was still in flight when it ran.
        self.sweep_experts()?;
        if let Some(ecfg) = &self.engram_cfg {
            for &t in tokens {
                state.engram.push(ecfg, t);
            }
        }
        state.pos = tokens.len();
        let s = tokens.len();
        let last = prefill.x.narrow(1, s - 1, 1)?.contiguous()?;
        let h = hc_pre(&last, &prefill.pre_mix[s - 1..])?;
        let h = rms_norm(&h, &self.norm, self.rms_eps)?;
        self.output.apply(&h.reshape((1, self.dim))?)
    }

    /// Layer `i` of several prefills at once: engram and attention prefill by prefill, then the FFN
    /// half over every prefill's tokens together, the MoE included - one pass over the layer's
    /// experts for all of them. The same as `prefill_layer` on each, to rounding.
    pub fn prefill_layer_batch(&self, i: usize, states: &mut [PrefillState]) -> Result<()> {
        if i >= self.blocks.len() || states.iter().any(|s| s.next != i) {
            return Err(Error::msg(format!(
                "prefill: layer {i} asked for out of turn"
            )));
        }
        let block = &self.blocks[i];
        let mut mids = Vec::with_capacity(states.len());
        let mut pre = Vec::new();
        for state in states.iter_mut() {
            if let (Some(engram), Some(ecfg)) = (&self.engrams[i], &self.engram_cfg) {
                let which = ecfg.layer_ids.iter().position(|&l| l == i).unwrap_or(0);
                state.x = engram.forward(&state.x, &prefill_hashes(ecfg, which, &state.tokens))?;
            }
            let (cos, sin) = if self.ratios.get(i).copied().unwrap_or(0) == 0 {
                (&state.ropes[0].0, &state.ropes[0].1)
            } else {
                (&state.ropes[1].0, &state.ropes[1].1)
            };
            let (mid, mix) = block.forward_prefill_attn(
                &state.x,
                &state.pre_mix,
                cos,
                sin,
                &mut state.shared,
                None,
            )?;
            mids.push(mid);
            pre.extend(mix);
        }
        let refs: Vec<&Tensor> = mids.iter().collect();
        let (out, next_pre) = block.forward_prefill_ffn(&Tensor::cat(&refs, 1)?, &pre)?;
        let mut at = 0;
        for state in states.iter_mut() {
            let s = state.tokens.len();
            state.x = out.narrow(1, at, s)?.contiguous()?;
            state.pre_mix = next_pre[at..at + s].to_vec();
            state.next += 1;
            at += s;
        }
        Ok(())
    }

    /// Collapse the hc copies, final norm, head: the logits of a prefill whose layers have all run.
    pub fn prefill_finish(&self, state: PrefillState) -> Result<Tensor> {
        if state.next != self.blocks.len() {
            return Err(Error::msg(format!(
                "prefill: finished after {} of {} layers",
                state.next,
                self.blocks.len()
            )));
        }
        let s = state.tokens.len();
        let h = hc_pre(&state.x, &state.pre_mix)?; // [1, s, dim]
        let h = rms_norm(&h, &self.norm, self.rms_eps)?;
        self.output.apply(&h.reshape((s, self.dim))?)
    }

    /// Resize the routed experts' hot cache of layer `i` alone.
    pub fn set_layer_expert_cache(&self, i: usize, slots: usize) {
        if let Some(b) = self.blocks.get(i) {
            b.moe.set_expert_cache(slots);
        }
    }

    /// The file's routing prior, per layer; empty when the file carries none.
    pub fn hot_experts(&self) -> &[Vec<usize>] {
        &self.hot_experts
    }

    /// Routed experts a token passes through, beside the shared one.
    pub fn n_activated(&self) -> usize {
        self.blocks.first().map(|b| b.moe.n_activated).unwrap_or(0)
    }

    /// The most a step of a batch of `tokens` asks a card for beside the weights it keeps: an
    /// expert's rows through its three projections on every lane at once, the indexer's
    /// scores over the batch, the batch's residual streams, and the largest weight a batch
    /// product sends over for the call.
    pub fn transient_bytes(&self, tokens: usize) -> usize {
        let f = std::mem::size_of::<f32>();
        let inter = self
            .blocks
            .first()
            .map(|b| b.moe.shared.w1.dims()[0])
            .unwrap_or(0);
        let per_lane = tokens * (2 * self.dim + 2 * inter) * f;
        let scores = tokens * tokens * f;
        let streams = tokens * self.dim * self.hc_mult * f;
        // A card holds scratch for the widest single step beside the weights it keeps: without it
        // the decode's own gemv, sparse attention and any weight it must admit run out of room on a
        // full card and fall to the host. The widest such step is holding one attention layer's
        // weights, or the head - whichever is larger.
        let largest = self
            .blocks
            .iter()
            .map(|b| b.attn.bytes())
            .max()
            .unwrap_or(0)
            .max(self.output.bytes());
        per_lane * self.n_activated().max(1) + scores + streams + largest
    }

    /// The token embedding table `[vocab, dim]`, shared with the DSpark draft.
    pub fn embed_ref(&self) -> &Tensor {
        &self.embed
    }

    /// The output head, shared with the DSpark draft.
    pub fn head_ref(&self) -> &crate::inference::offload::projection::Projection {
        &self.output
    }

    pub fn n_layers(&self) -> usize {
        self.blocks.len()
    }

    /// The bytes one routed expert occupies as stored.
    pub fn expert_bytes(&self) -> Result<usize> {
        self.blocks
            .first()
            .map(|b| b.moe.experts.expert_bytes())
            .unwrap_or(Ok(0))
    }

    /// The bytes of the always-read path as stored: every layer's attention projections and
    /// shared expert, the head, the engram projections. What the page cache must hold beside
    /// the routed experts for a token to run without re-reading it.
    pub fn resident_path_bytes(&self) -> usize {
        self.blocks
            .iter()
            .map(|b| b.attn.bytes() + b.moe.shared.bytes() + b.moe.gate_bytes())
            .sum::<usize>()
            + self.output.bytes()
            + self
                .engrams
                .iter()
                .flatten()
                .map(|e| e.wkv.bytes())
                .sum::<usize>()
    }

    /// Size the kept expert tier to the host memory left once the page cache can hold, beside
    /// it, the always-read path, one token's worth of routed experts, and one layer's whole
    /// set of experts - what a prompt's prefill streams through while the kept ones must stay.
    /// Without that room the kept tier fills the memory exactly, every miss read evicts a kept
    /// expert, and the misses sustain themselves. Applied to every layer; returns the slots.
    pub fn size_expert_cache_to_memory(&self, cfg: &DeepseekV41Config) -> Result<usize> {
        let expert_bytes = self.expert_bytes()? as u64;
        let per_token = expert_bytes * (cfg.n_activated_experts * cfg.n_layers) as u64;
        let per_layer = expert_bytes * cfg.n_routed_experts as u64;
        let reserve = self.resident_path_bytes() as u64 + per_token + per_layer;
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        let budget = sys.available_memory().saturating_sub(reserve);
        let slots = if expert_bytes == 0 || cfg.n_layers == 0 {
            0
        } else {
            (budget / cfg.n_layers as u64 / expert_bytes) as usize
        };
        tracing::info!(
            "expert host tier: {slots} slots/layer of {} routed ({} MB each); budget {} MB = available {} MB - reserve {} MB",
            cfg.n_routed_experts,
            expert_bytes / 1_000_000,
            budget / 1_000_000,
            sys.available_memory() / 1_000_000,
            reserve / 1_000_000
        );
        self.set_expert_cache(slots);
        Ok(slots)
    }

    /// Per layer, the streamed expert store's counters (hits, misses, evictions); empty entries
    /// for resident sets.
    pub fn expert_store_stats(&self) -> Vec<crate::inference::offload::store::StoreStats> {
        self.blocks
            .iter()
            .map(|b| match &b.moe.experts {
                crate::inference::offload::store::ExpertSet::Streamed(s) => s.stats(),
                crate::inference::offload::store::ExpertSet::Resident(_) => Default::default(),
            })
            .collect()
    }

    /// Layer `i`'s streamed expert store, when it is one.
    pub fn expert_store(&self, i: usize) -> Option<&crate::inference::offload::store::ExpertStore> {
        match &self.blocks.get(i)?.moe.experts {
            crate::inference::offload::store::ExpertSet::Streamed(s) => Some(s),
            crate::inference::offload::store::ExpertSet::Resident(_) => None,
        }
    }

    /// Reclaim, in every layer, the pages of the experts the tier does not keep: what
    /// read-ahead pulled in beside the ones asked for.
    pub fn sweep_experts(&self) -> Result<()> {
        for b in &self.blocks {
            b.moe.experts.sweep()?;
        }
        Ok(())
    }

    /// Per layer, the resident fraction of the kept experts' pages and of the other experts'.
    #[cfg(target_os = "linux")]
    pub fn expert_residency(&self) -> Result<Vec<(f64, f64)>> {
        self.blocks
            .iter()
            .map(|b| match &b.moe.experts {
                crate::inference::offload::store::ExpertSet::Streamed(s) => s.residency(),
                crate::inference::offload::store::ExpertSet::Resident(_) => Ok((1.0, 0.0)),
            })
            .collect()
    }

    /// Per layer, the last forward's routing per token: chosen experts with their weights.
    pub fn last_routing(&self) -> Vec<Vec<Vec<(usize, f32)>>> {
        self.blocks
            .iter()
            .map(|b| b.moe.last_routing.lock().unwrap().clone())
            .collect()
    }

    /// Per layer, how many times each routed expert was requested so far.
    pub fn expert_usage(&self) -> Vec<Vec<u64>> {
        self.blocks.iter().map(|b| b.moe.experts.usage()).collect()
    }

    /// Tell `observe` what every layer's MoE block routes where on each forward, with the layer's
    /// index first; `None` stops it. A calibration run is the only caller: without an observer the
    /// forward does no extra work.
    pub fn observe_experts(
        &self,
        observe: Option<
            std::sync::Arc<dyn Fn(usize, Option<usize>, &[f32], &[f32], &[f32]) + Send + Sync>,
        >,
    ) {
        for (l, b) in self.blocks.iter().enumerate() {
            *b.moe.observer.write().unwrap() = observe.clone().map(|f| {
                std::sync::Arc::new(move |e: Option<usize>, x: &[f32], w: &[f32], h: &[f32]| {
                    f(l, e, x, w, h)
                }) as std::sync::Arc<super::moe::ExpertObserver>
            });
        }
    }

    /// Run every layer's routed experts through `offload`, or back on the CPU with `None`.
    pub fn offload_experts(
        &self,
        offload: Option<std::sync::Arc<crate::inference::offload::experts::ExpertOffload>>,
    ) {
        for b in &self.blocks {
            *b.moe.offload.write().unwrap() = offload.clone();
        }
    }

    /// Resize every layer's expert hot cache to `slots` entries.
    pub fn set_expert_cache(&self, slots: usize) {
        for b in &self.blocks {
            b.moe.set_expert_cache(slots);
        }
    }

    /// A fresh decode state: one attention cache per layer.
    pub fn new_decode_state(&self) -> DecodeState {
        DecodeState::new(self.blocks.len())
    }

    /// Decode one token at `state.pos`, advancing the caches. Returns logits [1, vocab]. Feeding a
    /// sequence one token at a time from a fresh state reproduces `forward_prefill`'s per-position
    /// logits, which is what the self-consistency gate checks.
    pub fn forward_decode(&self, token: u32, state: &mut DecodeState) -> Result<Tensor> {
        let pos = state.pos;
        let (dim, hc) = (self.dim, self.hc_mult);

        let row =
            crate::inference::offload::stage("embed", || self.embed.narrow(0, token as usize, 1))?;
        let h = row.reshape((1, 1, dim))?;
        let mut x = h
            .unsqueeze(2)?
            .broadcast_as((1, 1, hc, dim))?
            .contiguous()?;

        let n = pos + 1;
        let (cos0, sin0) = rope_table(self.rope_head_dim, n, 0, self.rope_theta, self.rope_factor)?;
        let (cosb, sinb) = rope_table(
            self.rope_head_dim,
            n,
            self.original_seq_len,
            self.compress_rope_theta,
            self.rope_factor,
        )?;

        let mut pre_mix: Vec<Vec<f32>> = vec![{
            let mut v = vec![0f32; hc];
            v[0] = 1.0;
            v
        }];

        // The engram hash reads this token and the few before it, recorded once per step.
        let recent = self
            .engram_cfg
            .as_ref()
            .map(|ecfg| state.engram.push(ecfg, token));
        let mut shared = SharedAttn::default();
        // The DSpark draft reads the attention input (the hidden mean-pooled over the
        // hyper-connection copies) of its target layers; captured here, read after the decode.
        let cap = state.capture_layers.clone();
        if !cap.is_empty() {
            state.captured = vec![Vec::new(); cap.len()];
        }
        for (i, block) in self.blocks.iter().enumerate() {
            if let (Some(engram), Some(ecfg), Some(recent)) =
                (&self.engrams[i], &self.engram_cfg, &recent)
            {
                let which = ecfg.layer_ids.iter().position(|&l| l == i).unwrap_or(0);
                x = engram.forward(&x, &[ecfg.hash(which, recent)])?;
            }
            if let Some(ci) = cap.iter().position(|&l| l == i) {
                // Mean over the hyper-connection copies: x is [1, 1, hc, dim].
                let v = x.reshape((hc, dim))?.to_vec2::<f32>()?;
                let mut m = vec![0f32; dim];
                for row in &v {
                    for (j, &val) in row.iter().enumerate() {
                        m[j] += val;
                    }
                }
                let inv = 1.0 / hc as f32;
                for val in m.iter_mut() {
                    *val *= inv;
                }
                state.captured[ci] = m;
            }
            let (cos, sin) = if self.ratios.get(i).copied().unwrap_or(0) == 0 {
                (&cos0, &sin0)
            } else {
                (&cosb, &sinb)
            };
            let (nx, npm) = block.forward_decode(
                &x,
                &pre_mix,
                cos,
                sin,
                pos,
                &mut state.layers[i],
                &mut shared,
            )?;
            x = nx;
            pre_mix = npm;
        }

        let h = hc_pre(&x, &pre_mix)?;
        let h = crate::inference::offload::stage("final norm", || {
            rms_norm(&h, &self.norm, self.rms_eps)
        })?;
        let logits =
            crate::inference::offload::stage("head", || self.output.apply(&h.reshape((1, dim))?))?;
        state.pos += 1;
        Ok(logits)
    }

    /// Verify a block of `tokens` starting at `state.pos`, returning the next-token logits at
    /// every position, `[tokens.len(), vocab]`. Attention runs token by token over the ring, as
    /// decode does, so each token attends only to the ones before it and its own state carries a
    /// band layer's keys down to a later band layer; the FFN and its MoE then run once over the
    /// whole block, so each layer's experts are read once - which is why verifying a block is
    /// faster than decoding its tokens one at a time. The batched MoE differs from the per-token
    /// path only by floating-point rounding, so the per-position argmax matches what decoding one
    /// token at a time from the same state would give, and the caches are left in the same place.
    pub fn forward_verify_batch(&self, tokens: &[u32], state: &mut DecodeState) -> Result<Tensor> {
        if tokens.is_empty() {
            return Err(Error::msg("forward_verify_batch: no tokens"));
        }
        let (dim, hc) = (self.dim, self.hc_mult);
        let pos = state.pos;
        let k = tokens.len();

        // Every token embedded, broadcast into the hyper-connection copies: [1, k, hc, d].
        let mut rows = Vec::with_capacity(k);
        for &t in tokens {
            rows.push(self.embed.narrow(0, t as usize, 1)?);
        }
        let refs: Vec<&Tensor> = rows.iter().collect();
        let emb = Tensor::cat(&refs, 0)?.reshape((1, k, dim))?;
        let mut x = emb
            .unsqueeze(2)?
            .broadcast_as((1, k, hc, dim))?
            .contiguous()?;

        // Rope tables long enough that token j reads row pos + j.
        let n = pos + k;
        let (cos0, sin0) = rope_table(self.rope_head_dim, n, 0, self.rope_theta, self.rope_factor)?;
        let (cosb, sinb) = rope_table(
            self.rope_head_dim,
            n,
            self.original_seq_len,
            self.compress_rope_theta,
            self.rope_factor,
        )?;

        // The engram recents, one per token, pushed in order so token j reflects tokens up to j.
        let recents: Vec<_> = tokens
            .iter()
            .map(|&t| {
                self.engram_cfg
                    .as_ref()
                    .map(|ecfg| state.engram.push(ecfg, t))
            })
            .collect();

        // The collapse weights into the top layer, one per token, the top set.
        let mut pre_mix: Vec<Vec<f32>> = (0..k)
            .map(|_| {
                let mut v = vec![0f32; hc];
                v[0] = 1.0;
                v
            })
            .collect();
        // One shared-attention state per token: a band layer's index keys, candidates and
        // compressed KV pass down to a later band layer within a token, so the tokens of the
        // block cannot share one without a token reading another's keys.
        let mut shareds: Vec<SharedAttn> = (0..k).map(|_| SharedAttn::default()).collect();

        for (i, block) in self.blocks.iter().enumerate() {
            let (cos, sin) = if self.ratios.get(i).copied().unwrap_or(0) == 0 {
                (&cos0, &sin0)
            } else {
                (&cosb, &sinb)
            };
            // Attention half, token by token in order, each appending to and reading from the ring.
            let mut attn_rows = Vec::with_capacity(k);
            let mut am_pre: Vec<Vec<f32>> = Vec::with_capacity(k);
            for j in 0..k {
                let mut xj = x.narrow(1, j, 1)?.contiguous()?;
                if let (Some(engram), Some(ecfg), Some(recent)) =
                    (&self.engrams[i], &self.engram_cfg, &recents[j])
                {
                    let which = ecfg.layer_ids.iter().position(|&l| l == i).unwrap_or(0);
                    xj = engram.forward(&xj, &[ecfg.hash(which, recent)])?;
                }
                let (xj_attn, ampre_j) = block.forward_decode_attn(
                    &xj,
                    &pre_mix[j..j + 1],
                    cos,
                    sin,
                    pos + j,
                    &mut state.layers[i],
                    &mut shareds[j],
                )?;
                attn_rows.push(xj_attn);
                am_pre.push(ampre_j.into_iter().next().unwrap_or_else(|| {
                    let mut v = vec![0f32; hc];
                    v[0] = 1.0;
                    v
                }));
            }
            let arefs: Vec<&Tensor> = attn_rows.iter().collect();
            let xattn = Tensor::cat(&arefs, 1)?;
            // FFN half, once over the whole block: one read of each expert.
            let (nx, npm) = block.forward_prefill_ffn(&xattn, &am_pre)?;
            x = nx;
            pre_mix = npm;
        }

        let h = hc_pre(&x, &pre_mix)?;
        let h = rms_norm(&h, &self.norm, self.rms_eps)?;
        let logits = self.output.apply(&h.reshape((k, dim))?)?;
        state.pos += k;
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
    use crate::tensor::quantized::gguf_file::{open_mapped, Value};
    use crate::tensor::quantized::GgmlDType;

    // toy shapes shared with the other deepseek_v41 tests.
    const DIM: usize = 256;
    const NH: usize = 4;
    const HD: usize = 128;
    const QLORA: usize = 64;
    const OLORA: usize = 64;
    const OG: usize = 4;
    const ER: usize = 8;
    const INTER: usize = 256;
    const HC: usize = 4;
    const MIX: usize = 24;
    const INH: usize = 4;
    const IHD: usize = 64;
    pub(super) const VOCAB: usize = 48;

    fn f32e(name: String, dims: Vec<usize>) -> GgufEntry {
        let n: usize = dims.iter().product();
        let data: Vec<u8> = (0..n)
            .flat_map(|i| (i as f32 * 0.0007 - 0.05).to_le_bytes())
            .collect();
        GgufEntry {
            name,
            dims,
            dtype: GgmlDType::F32,
            data,
        }
    }

    /// The tensors of one layer, in its band mode.
    fn layer_tensors(l: usize, band: bool) -> Vec<GgufEntry> {
        let p = format!("blk.{l}");
        let mut e = vec![
            f32e(format!("{p}.attn_q_a.weight"), vec![QLORA, DIM]),
            f32e(format!("{p}.attn_q_a_norm.weight"), vec![QLORA]),
            f32e(format!("{p}.attn_q_b.weight"), vec![NH * HD, QLORA]),
            f32e(format!("{p}.attn_kv.weight"), vec![HD, DIM]),
            f32e(format!("{p}.attn_kv_norm.weight"), vec![HD]),
            f32e(format!("{p}.attn_sink"), vec![NH]),
            f32e(
                format!("{p}.attn_o_a.weight"),
                vec![OG * OLORA, NH * HD / OG],
            ),
            f32e(format!("{p}.attn_o_b.weight"), vec![DIM, OG * OLORA]),
            f32e(format!("{p}.attn_norm.weight"), vec![DIM]),
            f32e(format!("{p}.ffn_norm.weight"), vec![DIM]),
            f32e(format!("{p}.ffn_gate_inp.weight"), vec![ER, DIM]),
            f32e(format!("{p}.ffn_gate_inp.bias"), vec![ER]),
            f32e(format!("{p}.ffn_gate_shexp.weight"), vec![INTER, DIM]),
            f32e(format!("{p}.ffn_up_shexp.weight"), vec![INTER, DIM]),
            f32e(format!("{p}.ffn_down_shexp.weight"), vec![DIM, INTER]),
            f32e(format!("{p}.ffn_gate_exps.weight"), vec![ER, INTER, DIM]),
            f32e(format!("{p}.ffn_up_exps.weight"), vec![ER, INTER, DIM]),
            f32e(format!("{p}.ffn_down_exps.weight"), vec![ER, DIM, INTER]),
            f32e(format!("{p}.hc_attn_fn.weight"), vec![MIX, HC * DIM]),
            f32e(format!("{p}.hc_attn_scale"), vec![3]),
            f32e(format!("{p}.hc_attn_base"), vec![MIX]),
            f32e(format!("{p}.hc_ffn_fn.weight"), vec![MIX, HC * DIM]),
            f32e(format!("{p}.hc_ffn_scale"), vec![3]),
            f32e(format!("{p}.hc_ffn_base"), vec![MIX]),
        ];
        if band {
            e.extend([
                f32e(format!("{p}.attn_compressor_norm.weight"), vec![HD]),
                f32e(format!("{p}.attn_compressor_kv.weight"), vec![HD, DIM]),
                f32e(format!("{p}.attn_compressor_gate.weight"), vec![HD, DIM]),
                f32e(
                    format!("{p}.attn_indexer_q_b.weight"),
                    vec![INH * IHD, QLORA],
                ),
                f32e(format!("{p}.attn_indexer_weights.weight"), vec![INH, DIM]),
                f32e(format!("{p}.attn_indexer_k.weight"), vec![IHD, HD]),
                f32e(format!("{p}.attn_indexer_k_norm.weight"), vec![IHD]),
            ]);
        }
        e
    }

    /// A three-layer model (ratios 0, 2, 0 - a self-sourcing band layer between two window layers)
    /// loads and runs prefill to finite logits of the right shape. This gates the whole assembly -
    /// embedding, hc expansion, per-layer rope regimes, the block chain, collapse, norm and head;
    /// the numbers of each mechanism are gated by their own tests.
    #[test]
    fn assembles_and_runs_prefill() {
        let u = |k: &str, v: u32| (format!("deepseek_v41.{k}"), Value::U32(v));
        let f = |k: &str, v: f32| (format!("deepseek_v41.{k}"), Value::F32(v));
        let md = vec![
            (
                "general.architecture".to_string(),
                Value::String("deepseek_v41".into()),
            ),
            u("block_count", 3),
            u("embedding_length", DIM as u32),
            u("attention.head_count", NH as u32),
            u("attention.key_length", HD as u32),
            u("expert_count", ER as u32),
            u("expert_used_count", 2),
            u("expert_shared_count", 1),
            u("expert_feed_forward_length", INTER as u32),
            f("expert_weights_scale", 1.5),
            f("expert_swiglu_limit", 10.0),
            u("hyper_connection_mult", HC as u32),
            u("hyper_connection_sinkhorn_iters", 20),
            (
                "deepseek_v41.attention.compress_ratios".to_string(),
                Value::Array(vec![Value::U32(0), Value::U32(2), Value::U32(0)]),
            ),
            (
                "deepseek_v41.attention.kv_source_layers".to_string(),
                Value::Array(vec![Value::U32(1)]),
            ),
            (
                "deepseek_v41.attention.index_source_layers".to_string(),
                Value::Array(vec![Value::U32(1)]),
            ),
            u("attention.index_head_count", INH as u32),
            u("attention.index_key_length", IHD as u32),
            u("attention.index_topk", 16),
            f("rope.freq_base", 10000.0),
            f("attention.compress_rope_theta", 160000.0),
            f("rope.scaling.factor", 16.0),
            u("rope.scaling.original_context_length", 128),
        ];
        let mut entries = vec![
            f32e("token_embd.weight".into(), vec![VOCAB, DIM]),
            f32e("output_norm.weight".into(), vec![DIM]),
            f32e("output.weight".into(), vec![VOCAB, DIM]),
        ];
        for (l, &band) in [false, true, false].iter().enumerate() {
            entries.extend(layer_tensors(l, band));
        }
        let path =
            std::env::temp_dir().join(format!("loken-dsv41-model-{}.gguf", std::process::id()));
        write_gguf_with_metadata(&path, &md, &entries).unwrap();
        let g = open_mapped(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let mut cfg = DeepseekV41Config::from_gguf(&g.content).unwrap();
        cfg.rope_head_dim = 32;
        cfg.q_lora_rank = QLORA;
        cfg.o_lora_rank = OLORA;
        cfg.o_groups = OG;
        cfg.window_size = 32;
        cfg.rms_eps = 1e-20;

        let model = DeepseekV41Model::load(&g, &cfg).unwrap();
        let tokens: Vec<u32> = vec![1, 5, 3, 0, 7, 2, 9, 4]; // s divisible by ratio 2
        let logits = model.forward_prefill(&tokens).unwrap();
        assert_eq!(logits.dims(), &[tokens.len(), VOCAB]);
        let v = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| x.is_finite()), "model logits not finite");
    }

    // ---- llama.cpp naming and split files -------------------------------------------------

    /// One block's tensors under llama.cpp's names: bare (no `.weight`), `attn_kv_a_norm`,
    /// `attn_sinks`, `attn_output_a/b`, the gate bias as `exp_probs_b`, and the CSA2 extras as
    /// `attn_compressor_*` / `indexer.*`. A band block carries the extras; a reader would not.
    fn real_layer_tensors(l: usize, band: bool) -> Vec<GgufEntry> {
        let p = format!("blk.{l}");
        let mut e = vec![
            f32e(format!("{p}.attn_q_a"), vec![QLORA, DIM]),
            f32e(format!("{p}.attn_q_a_norm"), vec![QLORA]),
            f32e(format!("{p}.attn_q_b"), vec![NH * HD, QLORA]),
            f32e(format!("{p}.attn_kv"), vec![HD, DIM]),
            f32e(format!("{p}.attn_kv_a_norm"), vec![HD]),
            f32e(format!("{p}.attn_sinks"), vec![NH]),
            f32e(format!("{p}.attn_output_a"), vec![OG * OLORA, NH * HD / OG]),
            f32e(format!("{p}.attn_output_b"), vec![DIM, OG * OLORA]),
            f32e(format!("{p}.attn_norm"), vec![DIM]),
            f32e(format!("{p}.ffn_norm"), vec![DIM]),
            f32e(format!("{p}.ffn_gate_inp"), vec![ER, DIM]),
            f32e(format!("{p}.exp_probs_b"), vec![ER]),
            f32e(format!("{p}.ffn_gate_shexp"), vec![INTER, DIM]),
            f32e(format!("{p}.ffn_up_shexp"), vec![INTER, DIM]),
            f32e(format!("{p}.ffn_down_shexp"), vec![DIM, INTER]),
            f32e(format!("{p}.ffn_gate_exps"), vec![ER, INTER, DIM]),
            f32e(format!("{p}.ffn_up_exps"), vec![ER, INTER, DIM]),
            f32e(format!("{p}.ffn_down_exps"), vec![ER, DIM, INTER]),
            f32e(format!("{p}.hc_attn_fn"), vec![MIX, HC * DIM]),
            f32e(format!("{p}.hc_attn_scale"), vec![3]),
            f32e(format!("{p}.hc_attn_base"), vec![MIX]),
            f32e(format!("{p}.hc_ffn_fn"), vec![MIX, HC * DIM]),
            f32e(format!("{p}.hc_ffn_scale"), vec![3]),
            f32e(format!("{p}.hc_ffn_base"), vec![MIX]),
        ];
        if band {
            e.extend([
                f32e(format!("{p}.attn_compressor_norm"), vec![HD]),
                f32e(format!("{p}.attn_compressor_kv"), vec![HD, DIM]),
                f32e(format!("{p}.attn_compressor_gate"), vec![HD, DIM]),
                f32e(format!("{p}.indexer.attn_q_b"), vec![INH * IHD, QLORA]),
                f32e(format!("{p}.indexer.proj"), vec![INH, DIM]),
                f32e(format!("{p}.indexer.attn_k"), vec![IHD, HD]),
                f32e(format!("{p}.indexer.k_norm"), vec![IHD]),
            ]);
        }
        e
    }

    /// The metadata llama.cpp's converter writes, and only that: no low-rank widths, no band
    /// role lists, no YaRN, no candidate keys - each of those must be derived or defaulted.
    fn real_md(n_layers: u32) -> Vec<(String, Value)> {
        real_md_layout(&[0, 2, 0][..n_layers as usize], &[], &[])
    }

    /// The same metadata for a chosen band layout: the compress ratios, and the kv and index
    /// source layers when the layout names them (an empty list leaves them to be derived).
    pub(super) fn real_md_layout(
        ratios: &[u32],
        kv_src: &[u32],
        index_src: &[u32],
    ) -> Vec<(String, Value)> {
        let n_layers = ratios.len() as u32;
        let arr = |k: &str, xs: &[u32]| {
            (
                format!("deepseek4.{k}"),
                Value::Array(xs.iter().map(|&x| Value::U32(x)).collect()),
            )
        };
        let mut md = real_md_base(n_layers, ratios);
        if !kv_src.is_empty() {
            md.push(arr("attention.kv_source_layers", kv_src));
        }
        if !index_src.is_empty() {
            md.push(arr("attention.index_source_layers", index_src));
        }
        md
    }

    fn real_md_base(n_layers: u32, ratios: &[u32]) -> Vec<(String, Value)> {
        let u = |k: &str, v: u32| (format!("deepseek4.{k}"), Value::U32(v));
        let f = |k: &str, v: f32| (format!("deepseek4.{k}"), Value::F32(v));
        vec![
            (
                "general.architecture".to_string(),
                Value::String("deepseek4".into()),
            ),
            u("block_count", n_layers),
            u("embedding_length", DIM as u32),
            u("attention.head_count", NH as u32),
            u("attention.key_length", HD as u32),
            u("attention.sliding_window", 32),
            f("attention.layer_norm_rms_epsilon", 1e-20),
            u("expert_count", ER as u32),
            u("expert_used_count", 2),
            u("expert_shared_count", 1),
            u("feed_forward_length", INTER as u32),
            f("expert_weights_scale", 1.5),
            (
                "deepseek4.expert_weights_norm".to_string(),
                Value::Bool(true),
            ),
            (
                "deepseek4.swiglu_clamp_exp".to_string(),
                Value::Array((0..n_layers).map(|_| Value::F32(10.0)).collect()),
            ),
            u("hyper_connection.count", HC as u32),
            u("hyper_connection.sinkhorn_iterations", 20),
            f("hyper_connection.epsilon", 1e-6),
            (
                "deepseek4.attention.compress_ratios".to_string(),
                Value::Array(ratios.iter().map(|&r| Value::U32(r)).collect()),
            ),
            f("attention.compress_rope_freq_base", 160000.0),
            u("indexer.head_count", INH as u32),
            u("indexer.key_length", IHD as u32),
            u("indexer.top_k", 16),
            f("rope.freq_base", 10000.0),
            u("rope.dimension_count", 32),
        ]
    }

    fn real_entries() -> Vec<GgufEntry> {
        real_entries_for(&[false, true, false])
    }

    /// The tensors of a model whose layers are band layers where `bands` says.
    pub(super) fn real_entries_for(bands: &[bool]) -> Vec<GgufEntry> {
        let mut entries = vec![
            f32e("token_embd".into(), vec![VOCAB, DIM]),
            f32e("output_norm".into(), vec![DIM]),
            f32e("output".into(), vec![VOCAB, DIM]),
        ];
        for (l, &band) in bands.iter().enumerate() {
            entries.extend(real_layer_tensors(l, band));
        }
        entries
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("loken-dsv41-{tag}-{}", std::process::id()))
    }

    /// A file in llama.cpp's naming, carrying only the keys its converter writes, loads: the
    /// low-rank widths come from the projection shapes, the band roles from the tensors a block
    /// carries, and the release values fill what the converter omits.
    #[test]
    fn loads_llamacpp_naming_and_derives_what_the_file_omits() {
        let path = tmp("real.gguf");
        write_gguf_with_metadata(&path, &real_md(3), &real_entries()).unwrap();
        let g = open_mapped(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let cfg = DeepseekV41Config::from_gguf(&g.content).unwrap();
        assert_eq!(
            (cfg.q_lora_rank, cfg.o_groups, cfg.o_lora_rank),
            (QLORA, OG, OLORA)
        );
        assert_eq!(cfg.kv_source_layers, vec![1]);
        assert_eq!(cfg.index_source_layers, vec![1]);
        assert_eq!(cfg.moe_inter_dim, INTER);
        assert_eq!(cfg.vocab_size, VOCAB);
        assert_eq!(cfg.swiglu_limits, vec![10.0; 3]);
        assert!(cfg.norm_topk);
        assert_eq!((cfg.rope_factor, cfg.original_seq_len), (16.0, 65536));
        assert_eq!(cfg.candidate_source_layer, 20);
        assert_eq!(cfg.hc_mult, HC);
        assert_eq!(cfg.compress_rope_theta, 160000.0);

        let model = DeepseekV41Model::load(&g, &cfg).unwrap();
        let tokens: Vec<u32> = vec![1, 5, 3, 0, 7, 2, 9, 4];
        let logits = model.forward_prefill(&tokens).unwrap();
        assert_eq!(logits.dims(), &[tokens.len(), VOCAB]);
        let v = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }

    /// The same tensors under a mainline conversion's spelling - `.weight`/`.bias` suffixes, the
    /// `deepseek41` tag, the low-rank widths and output groups written as keys, integer arrays
    /// signed - load to exactly the logits of the bare-named file.
    #[test]
    fn mainline_spelling_loads_like_the_bare_one() {
        let bare = tmp("bare.gguf");
        write_gguf_with_metadata(&bare, &real_md(3), &real_entries()).unwrap();
        let g = open_mapped(&bare).unwrap();
        let _ = std::fs::remove_file(&bare);
        let cfg = DeepseekV41Config::from_gguf(&g.content).unwrap();
        let tokens: Vec<u32> = vec![1, 5, 3, 0, 7, 2, 9, 4];
        let want = DeepseekV41Model::load(&g, &cfg)
            .unwrap()
            .forward_prefill(&tokens)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let entries: Vec<GgufEntry> = real_entries()
            .into_iter()
            .map(|mut e| {
                e.name = if e.name.ends_with("exp_probs_b") {
                    format!("{}.bias", e.name)
                } else {
                    format!("{}.weight", e.name)
                };
                e
            })
            .collect();
        let mut md: Vec<(String, Value)> = real_md(3)
            .into_iter()
            .map(|(k, v)| match (k.as_str(), v) {
                ("general.architecture", _) => (k, Value::String("deepseek41".into())),
                ("deepseek4.attention.compress_ratios", Value::Array(a)) => (
                    "deepseek41.attention.compress_ratios".into(),
                    Value::Array(
                        a.iter()
                            .map(|v| Value::I32(v.to_u32().unwrap() as i32))
                            .collect(),
                    ),
                ),
                (_, v) => (
                    k.replacen("deepseek4.indexer.", "deepseek41.attention.indexer.", 1)
                        .replacen("deepseek4.", "deepseek41.", 1),
                    v,
                ),
            })
            .collect();
        md.push((
            "deepseek41.attention.q_lora_rank".into(),
            Value::U32(QLORA as u32),
        ));
        md.push((
            "deepseek41.attention.output_group_count".into(),
            Value::U32(OG as u32),
        ));
        md.push((
            "deepseek41.attention.output_lora_rank".into(),
            Value::U32(OLORA as u32),
        ));
        let path = tmp("mainline.gguf");
        write_gguf_with_metadata(&path, &md, &entries).unwrap();
        let g = open_mapped(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let cfg2 = DeepseekV41Config::from_gguf(&g.content).unwrap();
        assert_eq!(
            (cfg2.q_lora_rank, cfg2.o_groups, cfg2.o_lora_rank),
            (QLORA, OG, OLORA)
        );
        assert_eq!(cfg2.kv_source_layers, cfg.kv_source_layers);
        assert_eq!(cfg2.index_source_layers, cfg.index_source_layers);
        assert_eq!(cfg2.compress_ratios, cfg.compress_ratios);
        assert_eq!(cfg2.candidate_source_layer, cfg.candidate_source_layer);
        assert_eq!(
            (cfg2.index_n_heads, cfg2.index_head_dim, cfg2.index_topk),
            (cfg.index_n_heads, cfg.index_head_dim, cfg.index_topk)
        );
        let got = DeepseekV41Model::load(&g, &cfg2)
            .unwrap()
            .forward_prefill(&tokens)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(got, want);
    }

    /// The engram keys and tensors of one layer, for a toy hash layout: 2 heads, 3-gram, a
    /// 4-column table of 48 rows whose bucket ranges are consecutive primes.
    fn engram_md_and_entries(layer: usize) -> (Vec<(String, Value)>, Vec<GgufEntry>) {
        const EH: usize = 2;
        const EHD: usize = 8;
        const NGRAM: usize = 3;
        let primes = [7u64, 11, 13, 17];
        let offsets = [0u64, 7, 18, 31];
        let rows = 48usize;
        let arr_u64 = |k: &str, xs: &[u64]| {
            (
                format!("deepseek4.engram.{k}"),
                Value::Array(xs.iter().map(|&x| Value::U64(x)).collect()),
            )
        };
        // Tokens 3 and 4 collapse onto one compressed id.
        let token_map: Vec<u64> = (0..VOCAB as u64)
            .map(|t| if t == 4 { 3 } else { t })
            .collect();
        let md = vec![
            (
                "deepseek4.engram.layer_ids".into(),
                Value::Array(vec![Value::I32(layer as i32)]),
            ),
            ("deepseek4.engram.head_count".into(), Value::U32(EH as u32)),
            ("deepseek4.engram.key_length".into(), Value::U32(EHD as u32)),
            (
                "deepseek4.engram.max_ngram_size".into(),
                Value::U32(NGRAM as u32),
            ),
            ("deepseek4.engram.pad_id".into(), Value::U32(2)),
            arr_u64("multipliers", &[3, 5, 7]),
            arr_u64("primes", &primes),
            arr_u64("offsets", &offsets),
            arr_u64("token_map", &token_map),
        ];
        let p = format!("blk.{layer}");
        let entries = vec![
            f32e(format!("{p}.engram_embd"), vec![rows, EHD]),
            f32e(format!("{p}.engram_q"), vec![HC, DIM]),
            f32e(format!("{p}.engram_k"), vec![HC, DIM]),
            f32e(
                format!("{p}.engram_wkv"),
                vec![DIM * (HC + 1), (NGRAM - 1) * EH * EHD],
            ),
        ];
        (md, entries)
    }

    /// A file carrying an engram layer parses its hash layout, the engram changes the logits,
    /// and decoding token by token reproduces prefill with it on: the hash state carried across
    /// steps reads the same rows prefill hashed.
    #[test]
    fn engram_loads_applies_and_decodes_like_prefill() {
        let tokens: Vec<u32> = vec![1, 5, 3, 0, 7, 2, 9, 4, 3, 3];
        let logits_of = |md: &[(String, Value)], entries: &[GgufEntry], tag: &str| {
            let path = tmp(tag);
            write_gguf_with_metadata(&path, md, entries).unwrap();
            let g = open_mapped(&path).unwrap();
            let _ = std::fs::remove_file(&path);
            let cfg = DeepseekV41Config::from_gguf(&g.content).unwrap();
            let model = DeepseekV41Model::load(&g, &cfg).unwrap();
            let prefill = model
                .forward_prefill(&tokens)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            (cfg, model, prefill)
        };
        let (_, _, without) = logits_of(&real_md(3), &real_entries(), "noengram.gguf");

        let (emd, eentries) = engram_md_and_entries(1);
        let mut md = real_md(3);
        md.extend(emd);
        let mut entries = real_entries();
        entries.extend(eentries);
        let (cfg, model, with) = logits_of(&md, &entries, "engram.gguf");
        let e = cfg
            .engram
            .as_ref()
            .expect("engram layout read from the file");
        assert_eq!(
            (e.layer_ids.clone(), e.n_heads, e.head_dim, e.max_ngram),
            (vec![1], 2, 8, 3)
        );
        assert_eq!(e.pad_id, 2);
        assert_eq!(e.compress(4), 3);
        assert_eq!(e.primes, vec![vec![7, 11, 13, 17]]);
        assert!(with.iter().all(|v| v.is_finite()));
        assert_ne!(with, without, "the engram layer left the logits unchanged");

        let vocab = cfg.vocab_size;
        let mut state = model.new_decode_state();
        for (pos, &tok) in tokens.iter().enumerate() {
            let got = model
                .forward_decode(tok, &mut state)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let want = &with[pos * vocab..(pos + 1) * vocab];
            let num: f32 = got.iter().zip(want).map(|(a, b)| (a - b) * (a - b)).sum();
            let den: f32 = want.iter().map(|b| b * b).sum();
            assert!(
                (num / den).sqrt() < 1e-4,
                "position {pos}: decode drifts from prefill"
            );
        }
    }

    /// The same model written as two split parts loads through `SplitGguf` to exactly the
    /// logits of the single file: which part a tensor sits in must not matter.
    #[test]
    fn split_parts_load_like_one_file() {
        use crate::tensor::quantized::gguf_source::SplitGguf;
        let entries = real_entries();
        let md = real_md(3);

        let single = tmp("one.gguf");
        write_gguf_with_metadata(&single, &md, &entries).unwrap();
        let g = open_mapped(&single).unwrap();
        let _ = std::fs::remove_file(&single);
        let cfg = DeepseekV41Config::from_gguf(&g.content).unwrap();
        let tokens: Vec<u32> = vec![1, 5, 3, 0, 7, 2, 9, 4];
        let want = DeepseekV41Model::load(&g, &cfg)
            .unwrap()
            .forward_prefill(&tokens)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        // Part 1: the non-block tensors and block 0; part 2: the rest.
        let (p1, p2): (Vec<GgufEntry>, Vec<GgufEntry>) = entries
            .into_iter()
            .partition(|e| !e.name.starts_with("blk.") || e.name.starts_with("blk.0."));
        let stem = tmp("split");
        let part = |i: u32| format!("{}-{i:05}-of-00002.gguf", stem.display());
        let with_split = |no: u32| {
            let mut m = md.clone();
            m.push(("split.count".to_string(), Value::U32(2)));
            m.push(("split.no".to_string(), Value::U32(no)));
            m
        };
        write_gguf_with_metadata(std::path::Path::new(&part(1)), &with_split(0), &p1).unwrap();
        write_gguf_with_metadata(std::path::Path::new(&part(2)), &with_split(1), &p2).unwrap();
        let split = SplitGguf::open(part(1)).unwrap();
        let _ = std::fs::remove_file(part(1));
        let _ = std::fs::remove_file(part(2));
        assert_eq!(split.parts().len(), 2);

        let cfg2 = DeepseekV41Config::from_meta(&split).unwrap();
        assert_eq!(cfg2.kv_source_layers, cfg.kv_source_layers);
        let got = DeepseekV41Model::load(&split, &cfg2)
            .unwrap()
            .forward_prefill(&tokens)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(got, want, "split parts changed the logits");
    }
}

#[cfg(test)]
mod oracle_gate {
    use super::*;
    use crate::tensor::gguf_write::write_gguf_with_metadata;
    use crate::tensor::quantized::gguf_file::open_mapped;

    /// A model of the tests' synthetic weights in the given band layout, its experts through a
    /// cache of `expert_cache` slots (0: all resident), and a prompt for it.
    fn synthetic(
        ratios: &[u32],
        kv_src: &[u32],
        index_src: &[u32],
        expert_cache: usize,
    ) -> (DeepseekV41Model, Vec<u32>) {
        let bands: Vec<bool> = (0..ratios.len())
            .map(|l| {
                ratios[l] != 0 || kv_src.contains(&(l as u32)) || index_src.contains(&(l as u32))
            })
            .collect();
        let md = super::tests::real_md_layout(ratios, kv_src, index_src);
        let entries = super::tests::real_entries_for(&bands);
        // Unique per call: cargo runs the tests in parallel.
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let gguf = std::env::temp_dir().join(format!(
            "loken-dsv41-synthetic-{}-{seq}.gguf",
            std::process::id()
        ));
        write_gguf_with_metadata(&gguf, &md, &entries).unwrap();
        let g = open_mapped(&gguf).unwrap();
        let _ = std::fs::remove_file(&gguf);
        let mut cfg = DeepseekV41Config::from_gguf(&g.content).unwrap();
        cfg.expert_cache_count = expert_cache;
        let model = DeepseekV41Model::load(&g, &cfg).unwrap();
        let tokens: Vec<u32> = (0..24)
            .map(|i| (i * 7 + 3) % super::tests::VOCAB as u32)
            .collect();
        (model, tokens)
    }

    /// Decode the prompt one token at a time from a fresh cache, and return the worst per-position
    /// relative L2 and cosine of the decode-step logits against the full prefill's. A causal model
    /// must agree here: the KV, compressed KV and index caches have to reproduce, per position, what
    /// prefill recomputes over the whole sequence.
    fn decode_vs_prefill(ratios: &[u32], kv_src: &[u32], index_src: &[u32]) -> (f32, f32) {
        let (model, tokens) = synthetic(ratios, kv_src, index_src, 0);
        let prefill = model.forward_prefill(&tokens).unwrap();
        let (s, vocab) = prefill.dims2().unwrap();
        let pf = prefill.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let mut state = model.new_decode_state();
        let mut worst_rel = 0f32;
        let mut worst_cos = 1f32;
        for (pos, &tok) in tokens.iter().enumerate() {
            let step = model.forward_decode(tok, &mut state).unwrap();
            let got = step.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let want = &pf[pos * vocab..(pos + 1) * vocab];
            let num: f32 = got.iter().zip(want).map(|(a, b)| (a - b) * (a - b)).sum();
            let den: f32 = want.iter().map(|b| b * b).sum();
            worst_rel = worst_rel.max((num / den).sqrt());
            let dot: f32 = got.iter().zip(want).map(|(a, b)| a * b).sum();
            let na = got.iter().map(|a| a * a).sum::<f32>().sqrt();
            let nb = den.sqrt();
            worst_cos = worst_cos.min(dot / (na * nb));
        }
        assert_eq!(s, tokens.len());
        (worst_cos, worst_rel)
    }

    fn vecf_of(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// Prefills run layer by layer together - attention each on its own, the FFN half over all their
    /// tokens at once - give the logits each gives alone.
    #[test]
    fn batched_layers_match_separate_prefills() {
        let (model, tokens) = synthetic(&[0, 2, 2, 0], &[1], &[1, 3], 0);
        let texts = [&tokens[..tokens.len() / 2], &tokens[tokens.len() / 3..]];
        let alone: Vec<Vec<f32>> = texts
            .iter()
            .map(|t| vecf_of(&model.forward_prefill(t).unwrap()))
            .collect();
        let mut states: Vec<_> = texts
            .iter()
            .map(|t| model.prefill_begin(t).unwrap())
            .collect();
        for layer in 0..model.n_layers() {
            model.prefill_layer_batch(layer, &mut states).unwrap();
        }
        for (state, want) in states.into_iter().zip(&alone) {
            let got = vecf_of(&model.prefill_finish(state).unwrap());
            let num: f32 = got.iter().zip(want).map(|(a, b)| (a - b) * (a - b)).sum();
            let den: f32 = want.iter().map(|b| b * b).sum();
            assert!(
                (num / den).sqrt() < 1e-4,
                "batched against alone: rel {}",
                (num / den).sqrt()
            );
        }
    }

    /// A checkpoint of the decode state rolls back exactly what the steps after it appended: the
    /// steps replayed from the rewound state give the same logits they gave the first time,
    /// bit for bit, whether the rejected span stops inside a compressed group or crosses a group
    /// boundary. This is what a speculative decode needs to discard rejected draft tokens.
    #[test]
    fn a_checkpoint_rewinds_the_decode_state_exactly() {
        for (ratios, kv_src, index_src) in [
            (&[0u32, 2, 0][..], &[1u32][..], &[1u32][..]),
            (&[0u32, 2, 2, 0][..], &[1u32][..], &[1u32, 3][..]),
        ] {
            let (model, tokens) = synthetic(ratios, kv_src, index_src, 0);
            // Decode a prefix, then checkpoint; the tail stands in for a drafted, then rejected,
            // speculative batch of several tokens.
            let head = tokens.len() - 6;
            let mut state = model.new_decode_state();
            for &t in &tokens[..head] {
                model.forward_decode(t, &mut state).unwrap();
            }
            let mark = state.checkpoint();
            let first: Vec<Vec<f32>> = tokens[head..]
                .iter()
                .map(|&t| vecf_of(&model.forward_decode(t, &mut state).unwrap()))
                .collect();
            assert_eq!(state.pos, tokens.len());
            state.rewind(&mark);
            assert_eq!(state.pos, head);
            let again: Vec<Vec<f32>> = tokens[head..]
                .iter()
                .map(|&t| vecf_of(&model.forward_decode(t, &mut state).unwrap()))
                .collect();
            for (a, b) in first.iter().zip(&again) {
                assert_eq!(a, b, "a rewound state must replay bit for bit");
            }
        }
    }

    /// A batched prefill into a fresh decode state leaves the state token-by-token decode leaves:
    /// its last logits match, and the decode steps that follow match, on the self-sourcing layout
    /// and the reader layout, stopping inside a compressed group and on a group boundary.
    #[test]
    fn prefill_into_leaves_the_decode_state() {
        let layouts: [(&[u32], &[u32], &[u32]); 2] =
            [(&[0, 2, 0], &[1], &[1]), (&[0, 2, 2, 0], &[1], &[1, 3])];
        let rel = |a: &[f32], b: &[f32]| {
            let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
            (num / b.iter().map(|y| y * y).sum::<f32>()).sqrt()
        };
        for (ratios, kv_src, index_src) in layouts {
            let (model, tokens) = synthetic(ratios, kv_src, index_src, 0);
            for split in [tokens.len() - 3, tokens.len() - 4] {
                let mut stepped = model.new_decode_state();
                let mut want = Vec::new();
                for &t in &tokens {
                    want.push(vecf_of(&model.forward_decode(t, &mut stepped).unwrap()));
                }
                let mut state = model.new_decode_state();
                let last = vecf_of(&model.prefill_into(&tokens[..split], &mut state).unwrap());
                assert_eq!(state.pos, split);
                assert!(
                    rel(&last, &want[split - 1]) < 1e-4,
                    "{ratios:?} split {split}: prefill logits {}",
                    rel(&last, &want[split - 1])
                );
                for (k, &t) in tokens[split..].iter().enumerate() {
                    let got = vecf_of(&model.forward_decode(t, &mut state).unwrap());
                    let r = rel(&got, &want[split + k]);
                    assert!(
                        r < 1e-4,
                        "{ratios:?} split {split}: decode step {k} after prefill, rel {r}"
                    );
                }
            }
        }
    }

    /// A batched verify of a block of tokens gives, at every position, what decoding the tokens
    /// one at a time from the same state gives: the same argmax, and logits within rounding of
    /// the per-token path (the FFN runs the block at once, so it differs only by accumulation).
    #[test]
    fn forward_verify_batch_matches_sequential_decode() {
        let layouts: [(&[u32], &[u32], &[u32]); 2] =
            [(&[0, 2, 0], &[1], &[1]), (&[0, 2, 2, 0], &[1], &[1, 3])];
        let rel = |a: &[f32], b: &[f32]| {
            let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
            (num / b.iter().map(|y| y * y).sum::<f32>()).sqrt()
        };
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap()
        };
        for (ratios, kv_src, index_src) in layouts {
            let (model, tokens) = synthetic(ratios, kv_src, index_src, 0);
            // The target at every position: decode one token at a time from a fresh state.
            let mut stepped = model.new_decode_state();
            let want: Vec<Vec<f32>> = tokens
                .iter()
                .map(|&t| vecf_of(&model.forward_decode(t, &mut stepped).unwrap()))
                .collect();
            let vocab = want[0].len();
            for split in [tokens.len() - 4, tokens.len() - 6] {
                let mut state = model.new_decode_state();
                let _ = model.prefill_into(&tokens[..split], &mut state).unwrap();
                assert_eq!(state.pos, split);
                let block = &tokens[split..];
                let logits = model.forward_verify_batch(block, &mut state).unwrap();
                let (k, v) = logits.dims2().unwrap();
                assert_eq!((k, v), (block.len(), vocab));
                assert_eq!(state.pos, split + block.len());
                let flat = vecf_of(&logits);
                for j in 0..k {
                    let got = &flat[j * vocab..(j + 1) * vocab];
                    let w = &want[split + j];
                    let r = rel(got, w);
                    assert!(r < 1e-3, "{ratios:?} split {split}: verify pos {j} rel {r}");
                    assert_eq!(
                        argmax(got),
                        argmax(w),
                        "{ratios:?} split {split}: verify pos {j} argmax"
                    );
                }
            }
        }
    }

    /// A rejected speculative block rolls back with no trace: checkpoint, verify the whole block,
    /// rewind, replay only the accepted prefix - the state then stands exactly where verifying that
    /// prefix alone would have left it, so the next token decodes to the same logits bit for bit.
    #[test]
    fn checkpoint_rewind_replays_the_accepted_prefix() {
        let layouts: [(&[u32], &[u32], &[u32]); 2] =
            [(&[0, 2, 0], &[1], &[1]), (&[0, 2, 2, 0], &[1], &[1, 3])];
        for (ratios, kv_src, index_src) in layouts {
            let (model, tokens) = synthetic(ratios, kv_src, index_src, 0);
            let split = tokens.len() - 6;
            let block = &tokens[split..];
            let next = tokens[0]; // the token decoded after the accepted prefix
            for keep in [1usize, block.len() / 2] {
                // Reference: verify only the accepted prefix, then decode the next token.
                let mut want_state = model.new_decode_state();
                let _ = model
                    .prefill_into(&tokens[..split], &mut want_state)
                    .unwrap();
                let _ = model
                    .forward_verify_batch(&block[..keep], &mut want_state)
                    .unwrap();
                let want = vecf_of(&model.forward_decode(next, &mut want_state).unwrap());

                // Trial: verify the whole block, reject past `keep`, replay the prefix.
                let mut state = model.new_decode_state();
                let _ = model.prefill_into(&tokens[..split], &mut state).unwrap();
                let mark = state.checkpoint();
                let _ = model.forward_verify_batch(block, &mut state).unwrap();
                assert_eq!(state.pos, split + block.len());
                state.rewind(&mark);
                assert_eq!(state.pos, split);
                let _ = model
                    .forward_verify_batch(&block[..keep], &mut state)
                    .unwrap();
                assert_eq!(state.pos, split + keep);
                let got = vecf_of(&model.forward_decode(next, &mut state).unwrap());

                assert_eq!(
                    got, want,
                    "{ratios:?} keep {keep}: state diverged after rewind"
                );
            }
        }
    }

    /// With a card taking attention, index scores and every product it can, a batched prefill gives
    /// the logits and the decode state the CPU gives, on both layouts.
    #[cfg(feature = "cuda")]
    #[test]
    fn prefill_on_a_card_matches_the_cpu() {
        use crate::inference::offload::cuda::Card;
        use crate::inference::offload::{room, with_offload};
        let Ok(dev) = crate::tensor::cuda::CudaDevice::new(0) else {
            eprintln!("no CUDA device; the card prefill is NOT covered by this run");
            return;
        };
        let (_, total) = crate::tensor::cuda_ext::mem_get_info(&Device::Cuda(dev.clone())).unwrap();
        let room = room::open(dev.ordinal(), total, 0);
        let rel = |a: &[f32], b: &[f32]| {
            let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
            (num / b.iter().map(|y| y * y).sum::<f32>()).sqrt()
        };
        let layouts: [(&[u32], &[u32], &[u32]); 2] =
            [(&[0, 2, 0], &[1], &[1]), (&[0, 2, 2, 0], &[1], &[1, 3])];
        let card: std::sync::Arc<dyn crate::inference::offload::Offload> =
            std::sync::Arc::new(Card::new(dev, room, 1));
        for (ratios, kv_src, index_src) in layouts {
            let (model, tokens) = synthetic(ratios, kv_src, index_src, 0);
            let split = tokens.len() - 3;
            let mut cpu_state = model.new_decode_state();
            let cpu = vecf_of(
                &model
                    .prefill_into(&tokens[..split], &mut cpu_state)
                    .unwrap(),
            );
            let mut card_state = model.new_decode_state();
            let on_card = vecf_of(
                &with_offload(card.clone(), || {
                    model.prefill_into(&tokens[..split], &mut card_state)
                })
                .unwrap(),
            );
            assert!(
                rel(&on_card, &cpu) < 1e-4,
                "{ratios:?}: prefill logits rel {}",
                rel(&on_card, &cpu)
            );
            for &t in &tokens[split..] {
                let a = vecf_of(&model.forward_decode(t, &mut cpu_state).unwrap());
                let b = vecf_of(&model.forward_decode(t, &mut card_state).unwrap());
                assert!(
                    rel(&b, &a) < 1e-4,
                    "{ratios:?}: decode after the card prefill rel {}",
                    rel(&b, &a)
                );
            }
        }
    }

    /// Unit 2, decode: token-at-a-time decode from a fresh cache reproduces prefill's per-position
    /// logits on the self-sourcing layout (ratio-0 window ring plus a band layer's compressed KV and
    /// index caches). This gates the caches against the oracle-verified prefill.
    #[test]
    fn decode_matches_prefill_self_sourcing() {
        let (cos, rel) = decode_vs_prefill(&[0, 2, 0], &[1], &[1]);
        eprintln!("DECODE l3 worst_cos={cos:.6} worst_rel_l2={rel:.6}");
        assert!(
            cos > 0.9999 && rel < 1e-3,
            "decode/prefill l3 cos {cos} rel {rel}"
        );
    }

    /// Unit 2, decode with cross-layer reader caches: a pure reader reuses the kv-source's compressed
    /// KV cache and the index-source's top-k, an index-source reads another layer's index keys. The
    /// per-step caches must publish exactly what prefill's shared state carried.
    #[test]
    fn decode_matches_prefill_readers() {
        let (cos, rel) = decode_vs_prefill(&[0, 2, 2, 0], &[1], &[1, 3]);
        eprintln!("DECODE reader worst_cos={cos:.6} worst_rel_l2={rel:.6}");
        assert!(
            cos > 0.9999 && rel < 1e-3,
            "decode/prefill reader cos {cos} rel {rel}"
        );
    }

    /// Unit 4b, streamed load path: the model whose experts are streamed from the mmap through a
    /// one-slot hot cache produces exactly the logits of the same model keeping every expert. This
    /// runs the real `load_moe` mmap-view loader (not a hand-built store) and, with a cache far below
    /// the expert count, forces eviction across the prompt's tokens - so it gates that streaming the
    /// experts on the loading path never changes the output.
    /// Engine wiring: the backend adapter drives the model the way the generate loop does - one
    /// multi-token forward at position 0 for the prompt, then one token per step at the running
    /// position - and must hand back exactly the next-token logits the direct prefill and a direct
    /// decode replay produce. A wrong position bookkeeping, a dropped token or a stale cache would
    /// each show up as a different row.
    #[test]
    fn backend_forward_matches_the_model() {
        use crate::inference::engine::model_backend::{DeepseekV41Backend, ModelBackend};
        let (model, tokens) = synthetic(&[0, 2, 0], &[1], &[1], 0);
        let prefill = model.forward_prefill(&tokens).unwrap();
        let (s, vocab) = prefill.dims2().unwrap();
        let want_prompt = vecf_of(&prefill.narrow(0, s - 1, 1).unwrap());

        // The direct decode continuation the backend's steps must reproduce.
        let (reference, _) = synthetic(&[0, 2, 0], &[1], &[1], 0);
        let mut st = reference.new_decode_state();
        for &t in &tokens {
            reference.forward_decode(t, &mut st).unwrap();
        }
        let extra = [3u32, 7, 1];
        let want_steps: Vec<Vec<f32>> = extra
            .iter()
            .map(|&t| vecf_of(&reference.forward_decode(t, &mut st).unwrap()))
            .collect();

        let mut backend = DeepseekV41Backend::new(model, 0, tokens.len() + extra.len());
        let x = Tensor::new(&tokens[..], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let got = backend.forward(&x, 0).unwrap();
        assert_eq!(got.dims(), &[1, vocab]);
        // The backend prefills the prompt in one batch, on a card where there is one, whose
        // products round differently from the CPU's: equal to rounding, not to the bit.
        let rel = |a: &[f32], b: &[f32]| {
            let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
            (num / b.iter().map(|y| y * y).sum::<f32>()).sqrt()
        };
        let r = rel(&vecf_of(&got), &want_prompt);
        assert!(r < 1e-4, "prompt forward differs from prefill: rel {r}");

        let mut pos = tokens.len();
        for (i, &t) in extra.iter().enumerate() {
            let x = Tensor::new(&[t], &Device::Cpu)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let got = backend.forward(&x, pos).unwrap();
            let r = rel(&vecf_of(&got), &want_steps[i]);
            assert!(r < 1e-4, "decode step {i} differs: rel {r}");
            pos += 1;
        }

        // A forward anywhere but where the cache stands is refused, not silently misplaced.
        let x = Tensor::new(&[5u32], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        assert!(backend.forward(&x, pos + 1).is_err());
    }

    #[test]
    fn streamed_load_matches_resident_load() {
        let (m0, tokens) = synthetic(&[0, 2, 0], &[1], &[1], 0);
        let (m1, _) = synthetic(&[0, 2, 0], &[1], &[1], 1);
        let a = m0
            .forward_prefill(&tokens)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = m1
            .forward_prefill(&tokens)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(a, b, "one-slot expert cache changed the logits");
    }
}
