//! Full-model assembly for deepseek_v41 - the step that turns the ported mechanisms into a forward.
//!
//! embed -> expand to hc_mult copies -> blocks (each in its band's attention mode) -> collapse the
//! copies -> final norm -> head -> logits. Prefill only, and only ratio-0 and self-sourcing band
//! layers so far: a reader layer (a candidate consumer that reads another layer's compressed KV and
//! index) needs the cross-layer shared state the streamed engine will carry, which comes next.

use super::band::SharedAttn;
use super::block::Block;
use super::cache::DecodeState;
use super::engram::{prefill_hashes, prefill_hashes_from, Engram, EngramConfig, EngramState};
use super::hyper_connections::hc_pre;
use super::load::{load_band_block, load_ratio0_block, projection};
use super::source::WeightSource;
use super::DeepseekV41Config;
use crate::inference::engine::llm_engine::QualityLevel;
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

/// Windows the read-ahead depth runs on its winner between rounds of trying its neighbours: four
/// times the three windows of a round, so a round is a fifth of the time.
const READAHEAD_EXPLOIT_WINDOWS: usize = 12;

pub struct DeepseekV41Model {
    embed: Tensor,      // [vocab, dim]
    output: Projection, // [vocab, dim]
    norm: Tensor,       // [dim]
    /// One per layer. A node that runs only a layer range holds `None` for the layers outside it -
    /// their weights are never fetched - so `blocks.len()` is the model's layer count whatever a
    /// node loaded, and every index into it is the absolute layer.
    blocks: Vec<Option<Block>>,
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
    /// How many layers ahead a decode starts reading the routed experts of, from the stream it
    /// has by then; 0 reads none ahead. Moved by the decode's own timing; timing only, the gate
    /// still decides the routing.
    readahead: crate::inference::offload::tuner::Tuner,
    /// A depth held fixed in place of the controller's: for a test of the read-ahead itself.
    readahead_pin: Option<usize>,
    /// The share of a token's routed experts whose host compute may finish a layer late; 0 keeps
    /// every layer exact.
    expert_deferral: f32,
    /// The quality budget to calibrate at backend start, with the model's identity and the
    /// calibration text's tokens; taken once.
    quality: std::sync::Mutex<Option<(QualityLevel, String, Vec<u32>)>>,
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

/// The activation crossing a pipeline cut mid-stack, for `forward_decode_range`: the residual
/// stream, its collapse weights, the band state that flows down within a token, and the engram
/// recent this token pushed. A node running a later layer range resumes from this.
pub struct DecodeBoundary {
    pub x: crate::tensor::Tensor,
    pub pre_mix: Vec<Vec<f32>>,
    pub shared: SharedAttn,
    pub recent: Option<Vec<u32>>,
}

impl DecodeBoundary {
    /// True when the band state is empty - the case at a ratio-0 layer boundary, where a pipeline
    /// cut is cheapest and the crossing is only the residual.
    pub fn shared_is_empty(&self) -> bool {
        self.shared.compress_kv.is_none()
            && self.shared.index_k.is_none()
            && self.shared.topk_idxs.is_none()
            && self.shared.candidates.is_none()
    }

    /// Serialise a boundary whose band state is empty (a ratio-0 cut) to the bytes that cross a
    /// node in a wire frame: the residual stream, its collapse weights, and the engram recent.
    /// Declines a boundary carrying band state - an interior-band cut - which needs the compressed
    /// KV and index keys serialised too, added when a plan places a cut there.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let (b, s, hc, dim) = self.x.dims4()?;
        if b != 1 || s != 1 {
            return Err(Error::msg(
                "DecodeBoundary::to_bytes: expected a single decode row",
            ));
        }
        let mut out = Vec::new();
        let put_u32 = |o: &mut Vec<u8>, v: usize| o.extend_from_slice(&(v as u32).to_le_bytes());
        put_u32(&mut out, hc);
        put_u32(&mut out, dim);
        for f in self.x.flatten_all()?.to_vec1::<f32>()? {
            out.extend_from_slice(&f.to_le_bytes());
        }
        put_u32(&mut out, self.pre_mix.len());
        for row in &self.pre_mix {
            put_u32(&mut out, row.len());
            for &f in row {
                out.extend_from_slice(&f.to_le_bytes());
            }
        }
        match &self.recent {
            Some(r) => {
                put_u32(&mut out, 1);
                put_u32(&mut out, r.len());
                for &t in r {
                    out.extend_from_slice(&t.to_le_bytes());
                }
            }
            None => put_u32(&mut out, 0),
        }
        // The band state, each field present-flagged. Empty at a ratio-0 cut (four zero flags), and
        // the compressor and indexer state a mid-band cut carries down, so the boundary crosses the
        // wire whatever layer it is taken at, not only where the band state happens to be empty.
        for t in [&self.shared.compress_kv, &self.shared.index_k] {
            match t {
                None => put_u32(&mut out, 0),
                Some(t) => {
                    put_u32(&mut out, 1);
                    let dims = t.dims();
                    put_u32(&mut out, dims.len());
                    for &d in dims {
                        put_u32(&mut out, d);
                    }
                    for f in t.flatten_all()?.to_vec1::<f32>()? {
                        out.extend_from_slice(&f.to_le_bytes());
                    }
                }
            }
        }
        match &self.shared.topk_idxs {
            None => put_u32(&mut out, 0),
            Some(v) => {
                put_u32(&mut out, 1);
                put_u32(&mut out, v.len());
                for &x in v {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
        match &self.shared.candidates {
            None => put_u32(&mut out, 0),
            Some(v) => {
                put_u32(&mut out, 1);
                put_u32(&mut out, v.len());
                for &b in v {
                    out.push(b as u8);
                }
            }
        }
        Ok(out)
    }

    /// Rebuild a boundary from `to_bytes`, band state and all, so a cut taken at any layer resumes
    /// exactly where it left off.
    pub fn from_bytes(bytes: &[u8]) -> Result<DecodeBoundary> {
        let mut p = 0usize;
        let need = |p: usize, n: usize| -> Result<()> {
            if p + n > bytes.len() {
                Err(Error::msg("DecodeBoundary::from_bytes: truncated"))
            } else {
                Ok(())
            }
        };
        let get_u32 = |p: &mut usize| -> Result<usize> {
            need(*p, 4)?;
            let v = u32::from_le_bytes(bytes[*p..*p + 4].try_into().unwrap()) as usize;
            *p += 4;
            Ok(v)
        };
        let hc = get_u32(&mut p)?;
        let dim = get_u32(&mut p)?;
        let get_f32 = |p: &mut usize| -> Result<f32> {
            need(*p, 4)?;
            let v = f32::from_le_bytes(bytes[*p..*p + 4].try_into().unwrap());
            *p += 4;
            Ok(v)
        };
        let xv: Vec<f32> = (0..hc * dim)
            .map(|_| get_f32(&mut p))
            .collect::<Result<_>>()?;
        let x = crate::tensor::Tensor::from_vec(xv, (1, 1, hc, dim), &crate::tensor::Device::Cpu)?;
        let rows = get_u32(&mut p)?;
        let mut pre_mix = Vec::with_capacity(rows);
        for _ in 0..rows {
            let cols = get_u32(&mut p)?;
            pre_mix.push((0..cols).map(|_| get_f32(&mut p)).collect::<Result<_>>()?);
        }
        let recent = if get_u32(&mut p)? == 1 {
            let n = get_u32(&mut p)?;
            Some(
                (0..n)
                    .map(|_| get_u32(&mut p).map(|v| v as u32))
                    .collect::<Result<_>>()?,
            )
        } else {
            None
        };
        // The band state, in the order `to_bytes` wrote it: the two tensors, then the two index
        // vectors. Absent fields (a ratio-0 cut) read back as `None`. Inlined rather than a helper
        // so the byte cursor is not borrowed by two closures at once.
        let mut tensors: Vec<Option<crate::tensor::Tensor>> = Vec::with_capacity(2);
        for _ in 0..2 {
            if get_u32(&mut p)? == 0 {
                tensors.push(None);
                continue;
            }
            let nd = get_u32(&mut p)?;
            let dims: Vec<usize> = (0..nd).map(|_| get_u32(&mut p)).collect::<Result<_>>()?;
            let count: usize = dims.iter().product();
            let data: Vec<f32> = (0..count).map(|_| get_f32(&mut p)).collect::<Result<_>>()?;
            tensors.push(Some(crate::tensor::Tensor::from_vec(
                data,
                crate::tensor::Shape::from_dims(&dims),
                &crate::tensor::Device::Cpu,
            )?));
        }
        let index_k = tensors.pop().unwrap();
        let compress_kv = tensors.pop().unwrap();
        let topk_idxs = if get_u32(&mut p)? == 1 {
            let n = get_u32(&mut p)?;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                need(p, 4)?;
                v.push(i32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()));
                p += 4;
            }
            Some(v)
        } else {
            None
        };
        let candidates = if get_u32(&mut p)? == 1 {
            let n = get_u32(&mut p)?;
            need(p, n)?;
            let v = bytes[p..p + n].iter().map(|&b| b != 0).collect();
            Some(v)
        } else {
            None
        };
        Ok(DecodeBoundary {
            x,
            pre_mix,
            shared: SharedAttn {
                compress_kv,
                index_k,
                topk_idxs,
                candidates,
            },
            recent,
        })
    }
}

/// What a decode layer range starts from: the token at the head of the stack, or the boundary a
/// previous range produced.
pub enum DecodeRangeIn {
    Token(u32),
    Resume(DecodeBoundary),
}

/// What a decode layer range yields: the boundary for the next range, or the token's logits when
/// the range runs to the end of the stack.
pub enum DecodeRangeOut {
    Boundary(DecodeBoundary),
    Logits(crate::tensor::Tensor),
}

impl DeepseekV41Model {
    /// Load the resident model from a GGUF source - one mapped file or a split set. Each layer is
    /// loaded in its band's mode.
    pub fn load<S: WeightSource + ?Sized>(g: &S, cfg: &DeepseekV41Config) -> Result<Self> {
        Self::load_range(g, cfg, 0)
    }

    /// Load only the layers `[from, n)`, for a node that runs that range of a pipeline. The layers
    /// before `from` are left unheld and their weights are never fetched, which is the whole point
    /// of the split: a node streams and holds only its own stage. The head and tail tensors - the
    /// embedding, the final norm, the output projection - are loaded whatever the range, so the
    /// range runs to logits on its own. `from == 0` loads the whole model.
    pub fn load_range<S: WeightSource + ?Sized>(
        g: &S,
        cfg: &DeepseekV41Config,
        from: usize,
    ) -> Result<Self> {
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
            if layer < from {
                // Outside this node's range: never fetched, never run.
                blocks.push(None);
                engrams.push(None);
                continue;
            }
            let ratio = cfg.compress_ratios.get(layer).copied().unwrap_or(0);
            blocks.push(Some(if ratio == 0 {
                load_ratio0_block(g, layer, cfg)?
            } else {
                load_band_block(g, layer, cfg)?
            }));
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
            expert_deferral: 0.0,
            quality: std::sync::Mutex::new(None),
            // Starts one layer ahead (the nearest stream that stands in for the layer's own) and
            // may go as deep as the stack; a window is as many tokens as the stack has layers.
            readahead: crate::inference::offload::tuner::Tuner::new(
                "expert read-ahead depth",
                1,
                cfg.n_layers.saturating_sub(1),
                cfg.n_layers,
                READAHEAD_EXPLOIT_WINDOWS,
            ),
            readahead_pin: None,
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
        self.prefill_begin_at(tokens, 0)
    }

    /// `prefill_begin` for tokens that continue a sequence of `base` tokens: the rope tables run
    /// to the end of them, and the first of `tokens` reads row `base`.
    pub fn prefill_begin_at(&self, tokens: &[u32], base: usize) -> Result<PrefillState> {
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
        let (cos0, sin0) = rope_table(
            self.rope_head_dim,
            base + s,
            0,
            self.rope_theta,
            self.rope_factor,
        )?;
        let (cosb, sinb) = rope_table(
            self.rope_head_dim,
            base + s,
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
        let (nx, npm) = self.block(i)?.forward_prefill(
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
        self.prefill_into_grid(tokens, state, &[])
            .map(|(logits, _)| logits)
    }

    /// `prefill_into`, also keeping every layer's window at each token count of `grid` below the
    /// prompt's length: `windows[k][layer]` is the window section of `grid[k]` (see
    /// `snapshot::window_section`). The windows exist only while the pass runs, so this is the one
    /// place a mid-prompt state can be had.
    pub fn prefill_into_grid(
        &self,
        tokens: &[u32],
        state: &mut DecodeState,
        grid: &[usize],
    ) -> Result<(Tensor, Vec<Vec<Vec<u8>>>)> {
        if tokens.is_empty() {
            return Err(Error::msg("prefill_into: no tokens"));
        }
        // A state that holds tokens is continued: the pass appends to each layer's window,
        // compressed rows and filling group as the same tokens decoded would have.
        let base = state.pos;
        let grid: Vec<usize> = grid
            .iter()
            .copied()
            .filter(|&p| p > base && p < base + tokens.len())
            .collect();
        for layer in &mut state.layers {
            layer.base = base;
            layer.grid = grid.clone();
            layer.at_grid.clear();
        }
        let engram_start = state.engram.clone();
        // A prompt this wide routes to nearly every expert of every layer, so each layer is read
        // in while the one before it computes, instead of faulting in under its own MoE.
        let reads_whole_layers = self.routes_whole_layers(tokens.len());
        if reads_whole_layers {
            if let Some(b) = self.blocks.iter().flatten().next() {
                b.moe.prefetch_layer();
            }
        }
        let mut prefill = self.prefill_begin_at(tokens, base)?;
        // The residual stream goes to a card for the whole prompt when one has room: the
        // hyper-connection steps run there and only each sublayer's input and output cross.
        let mut held: Option<Box<dyn crate::inference::offload::HcStream>> =
            match crate::inference::offload::current() {
                Some(off) => {
                    let x = prefill.x.flatten_all()?.to_vec1::<f32>()?;
                    off.hc_stream(&x, self.hc_mult, self.dim)
                }
                None => None,
            };
        for (i, block) in self.blocks.iter().enumerate() {
            let Some(block) = block.as_ref() else {
                continue;
            };
            if reads_whole_layers {
                if let Some(next) = self.blocks.get(i + 1).and_then(Option::as_ref) {
                    next.moe.prefetch_layer();
                }
            }
            // Each layer's own time, for a recorder that keeps them apart rather than summed.
            let layer_started = std::time::Instant::now();
            if let (Some(engram), Some(ecfg)) = (&self.engrams[i], &self.engram_cfg) {
                let which = ecfg.layer_ids.iter().position(|&l| l == i).unwrap_or(0);
                // The engram rewrites the whole stream on the host.
                if let Some(ds) = held.as_mut() {
                    let s = tokens.len();
                    prefill.x = Tensor::from_vec(
                        ds.rows(0, s)?,
                        (1, s, self.hc_mult, self.dim),
                        &Device::Cpu,
                    )?;
                }
                prefill.x = crate::inference::offload::stage("engram", || {
                    engram.forward(
                        &prefill.x,
                        &prefill_hashes_from(ecfg, which, tokens, engram_start.clone()),
                    )
                })?;
                if let Some(ds) = held.as_mut() {
                    ds.replace(&prefill.x.flatten_all()?.to_vec1::<f32>()?)?;
                }
            }
            let rope = if self.ratios.get(i).copied().unwrap_or(0) == 0 {
                0
            } else {
                1
            };
            let (cos, sin) = (&prefill.ropes[rope].0, &prefill.ropes[rope].1);
            if let Some(ds) = held.as_mut() {
                match block.forward_prefill_dev(
                    ds.as_mut(),
                    &prefill.pre_mix,
                    cos,
                    sin,
                    &mut prefill.shared,
                    Some(&mut state.layers[i]),
                )? {
                    super::block::DevLayer::Device(pre) => prefill.pre_mix = pre,
                    super::block::DevLayer::Host(x, pre) => {
                        // The card let go of the stream: the host has it from here.
                        prefill.x = x;
                        prefill.pre_mix = pre;
                        held = None;
                    }
                }
            } else {
                let (mid, mix) = block.forward_prefill_attn(
                    &prefill.x,
                    &prefill.pre_mix,
                    cos,
                    sin,
                    &mut prefill.shared,
                    Some(&mut state.layers[i]),
                )?;
                let (nx, npm) = block.forward_prefill_ffn(&mid, &mix)?;
                prefill.x = nx;
                prefill.pre_mix = npm;
            }
            if let Some(off) = crate::inference::offload::current() {
                off.record("prefill layer", layer_started.elapsed().as_nanos() as u64);
            }
            prefill.next += 1;
        }
        // The head reads the last token's stream.
        if let Some(ds) = held.as_mut() {
            let s = tokens.len();
            let last = ds.rows(s - 1, 1)?;
            prefill.x = Tensor::from_vec(last, (1, 1, self.hc_mult, self.dim), &Device::Cpu)?;
        }
        // Once more over every layer now that every read has landed: the read-ahead a layer's
        // sweep could not see was still in flight when it ran.
        self.sweep_experts()?;
        if let Some(ecfg) = &self.engram_cfg {
            for &t in tokens {
                state.engram.push(ecfg, t);
            }
        }
        state.pos = base + tokens.len();
        let s = tokens.len();
        let last = prefill
            .x
            .narrow(1, prefill.x.dim(1)? - 1, 1)?
            .contiguous()?;
        let h = hc_pre(&last, &prefill.pre_mix[s - 1..])?;
        let h = rms_norm(&h, &self.norm, self.rms_eps)?;
        let logits = self.output.apply(&h.reshape((1, self.dim))?)?;
        let empty = super::snapshot::window_section(&Default::default());
        let windows = (0..grid.len())
            .map(|k| {
                state
                    .layers
                    .iter()
                    .map(|l| l.at_grid.get(k).cloned().unwrap_or_else(|| empty.clone()))
                    .collect()
            })
            .collect();
        for layer in &mut state.layers {
            layer.base = 0;
            layer.grid.clear();
            layer.at_grid.clear();
        }
        Ok((logits, windows))
    }

    /// Whether a batch of `tokens` routes to most of a layer's experts: with `k` of `E` chosen per
    /// token, the batch covers a share of the layer of about `1 - (1 - k/E)^tokens`, and reading
    /// the whole layer ahead pays once that share passes one half.
    fn routes_whole_layers(&self, tokens: usize) -> bool {
        let Some(b) = self.blocks.iter().flatten().next() else {
            return false;
        };
        let (k, e) = (b.moe.n_activated as f64, b.moe.n_routed as f64);
        e > 0.0 && 1.0 - (1.0 - k / e).powf(tokens as f64) > 0.5
    }

    /// The engram ids a decode holds after `tokens`: the last `max_ngram` of them, compressed.
    pub fn engram_after(&self, tokens: &[u32]) -> EngramState {
        let mut st = EngramState::default();
        if let Some(cfg) = &self.engram_cfg {
            for &t in &tokens[tokens.len().saturating_sub(cfg.max_ngram)..] {
                st.push(cfg, t);
            }
        }
        st
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
        let block = self.block(i)?;
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
        if let Some(b) = self.blocks.get(i).and_then(|b| b.as_ref()) {
            b.moe.set_expert_cache(slots);
        }
    }

    /// The file's routing prior, per layer; empty when the file carries none.
    pub fn hot_experts(&self) -> &[Vec<usize>] {
        &self.hot_experts
    }

    /// Routed experts a token passes through, beside the shared one.
    pub fn n_activated(&self) -> usize {
        self.blocks
            .iter()
            .flatten()
            .next()
            .map(|b| b.moe.n_activated)
            .unwrap_or(0)
    }

    /// The most a step of a batch of `tokens` asks a card for beside the weights it keeps: an
    /// expert's rows through its three projections on every lane at once, the indexer's
    /// scores over the batch, the batch's residual streams, and the largest weight a batch
    /// product sends over for the call.
    pub fn transient_bytes(&self, tokens: usize) -> usize {
        /// Experts a card runs at once during a prompt: the threads that feed it.
        const EXPERT_LANES_IN_FLIGHT: usize = 4; // lanes per card times the pair of cards a plan spans
        let f = std::mem::size_of::<f32>();
        let inter = self
            .blocks
            .iter()
            .flatten()
            .next()
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
            .flatten()
            .map(|b| b.attn.bytes())
            .max()
            .unwrap_or(0)
            .max(self.output.bytes());
        // A card that holds a prompt's stream and runs its attention and experts needs room for
        // them as well: the sublayers' inputs and outputs, the layer's token context, the tiles
        // of the attention weights expanded to f32, and the weights of the experts in flight.
        let prompt_steps = 4 * tokens * self.dim * f;
        let tiles = 3 * crate::inference::offload::WEIGHT_TILE_BYTES;
        let in_flight = EXPERT_LANES_IN_FLIGHT * 2 * inter * self.dim * f;
        per_lane + scores + streams + largest + prompt_steps + tiles + in_flight
    }

    /// The token embedding table `[vocab, dim]`, shared with the DSpark draft.
    pub fn embed_ref(&self) -> &Tensor {
        &self.embed
    }

    /// The output head, shared with the DSpark draft.
    pub fn head_ref(&self) -> &crate::inference::offload::projection::Projection {
        &self.output
    }

    /// Each layer's compression ratio, 0 for a layer that keeps only a window.
    pub fn compress_ratios(&self) -> &[usize] {
        &self.ratios
    }

    pub fn n_layers(&self) -> usize {
        self.blocks.len()
    }

    /// The block at absolute layer `i`, or an error when this node did not load that layer - it is
    /// outside the range this node runs.
    fn block(&self, i: usize) -> Result<&Block> {
        self.blocks
            .get(i)
            .and_then(|b| b.as_ref())
            .ok_or_else(|| Error::msg(format!("layer {i} is not held on this node")))
    }

    /// The shape a node needs to measure its own cost of computing one missed routed expert:
    /// `(dim, inter, gate_up_dtype, down_dtype)`, read from a real expert on the first layer this
    /// node holds. `None` when no layer is held or the experts are not in a GGUF quant format the
    /// miss bench models (a dense or block-scaled expert). The model declares the shape; the engine
    /// turns it into a cost.
    pub fn expert_cost_shape(
        &self,
    ) -> Option<(
        usize,
        usize,
        crate::tensor::quantized::GgmlDType,
        crate::tensor::quantized::GgmlDType,
    )> {
        let l = self.blocks.iter().position(|b| b.is_some())?;
        let e = self.expert_store(l)?.fetch(0).ok()?;
        let gate_dt = e.w1.quant_dtype()?;
        let down_dt = e.w2.quant_dtype()?;
        // w1 is [inter, dim].
        let dims = e.w1.dims();
        let inter = *dims.first()?;
        let dim = *dims.get(1)?;
        Some((dim, inter, gate_dt, down_dt))
    }

    /// The model-side inputs the cut planner needs to place this model across the cluster:
    /// `(attn_bytes_per_held_layer, expert_bytes, working_set, activated)`. `None` when a piece is
    /// not measurable - no held layer, no expert bytes, or no calibrated hot-expert count to size
    /// residency against. The working set is the measured hot-expert count per layer, never a guess.
    pub fn placement_model_inputs(&self) -> Option<(Vec<u64>, u64, u64, u64)> {
        let attn_per_layer: Vec<u64> = self
            .blocks
            .iter()
            .flatten()
            .map(|b| b.attn.bytes() as u64)
            .collect();
        if attn_per_layer.is_empty() {
            return None;
        }
        let expert_bytes = self.expert_bytes().ok()? as u64;
        if expert_bytes == 0 {
            return None;
        }
        let lens: Vec<usize> = self
            .hot_experts()
            .iter()
            .map(|h| h.len())
            .filter(|&l| l > 0)
            .collect();
        if lens.is_empty() {
            return None;
        }
        let working_set = (lens.iter().sum::<usize>() / lens.len()) as u64;
        Some((
            attn_per_layer,
            expert_bytes,
            working_set,
            self.n_activated() as u64,
        ))
    }

    /// The bytes one routed expert occupies as stored.
    pub fn expert_bytes(&self) -> Result<usize> {
        self.blocks
            .iter()
            .flatten()
            .next()
            .map(|b| b.moe.experts.expert_bytes())
            .unwrap_or(Ok(0))
    }

    /// The bytes of the always-read path as stored: every layer's attention projections and
    /// shared expert, the head, the engram projections. What the page cache must hold beside
    /// the routed experts for a token to run without re-reading it.
    pub fn resident_path_bytes(&self) -> usize {
        self.blocks
            .iter()
            .flatten()
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
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        // A hard free-memory headroom the kept tier never sizes into. The tier keeps its experts
        // hot, so without it the host runs with MemFree near zero - every byte is reclaimable page
        // cache, but a co-process or a prefill's allocation burst can outrun the kernel's reclaim
        // and OOM the host. One eighth of total RAM keeps enough genuinely free for a burst.
        let headroom = sys.total_memory() / 8;
        let reserve = self.resident_path_bytes() as u64 + per_token + per_layer + headroom;
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
            .flatten()
            .map(|b| match &b.moe.experts {
                crate::inference::offload::store::ExpertSet::Streamed(s) => s.stats(),
                crate::inference::offload::store::ExpertSet::Resident(_) => Default::default(),
            })
            .collect()
    }

    /// Layer `i`'s streamed expert store, when it is one.
    pub fn expert_store(&self, i: usize) -> Option<&crate::inference::offload::store::ExpertStore> {
        match &self.blocks.get(i)?.as_ref()?.moe.experts {
            crate::inference::offload::store::ExpertSet::Streamed(s) => Some(s),
            crate::inference::offload::store::ExpertSet::Resident(_) => None,
        }
    }

    /// Reclaim, in every layer, the pages of the experts the tier does not keep: what
    /// read-ahead pulled in beside the ones asked for.
    pub fn sweep_experts(&self) -> Result<()> {
        for b in self.blocks.iter().flatten() {
            b.moe.experts.sweep()?;
        }
        Ok(())
    }

    /// Per layer, the resident fraction of the kept experts' pages and of the other experts'.
    #[cfg(target_os = "linux")]
    pub fn expert_residency(&self) -> Result<Vec<(f64, f64)>> {
        self.blocks
            .iter()
            .flatten()
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
            .flatten()
            .map(|b| b.moe.last_routing.lock().unwrap().clone())
            .collect()
    }

    /// Per layer, how many times each routed expert was requested so far.
    pub fn expert_usage(&self) -> Vec<Vec<u64>> {
        self.blocks
            .iter()
            .flatten()
            .map(|b| b.moe.experts.usage())
            .collect()
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
            let Some(b) = b.as_ref() else { continue };
            *b.moe.observer.write().unwrap() = observe.clone().map(|f| {
                std::sync::Arc::new(move |e: Option<usize>, x: &[f32], w: &[f32], h: &[f32]| {
                    f(l, e, x, w, h)
                }) as std::sync::Arc<super::moe::ExpertObserver>
            });
        }
    }

    /// Let up to `share` of a decoding token's routed experts finish on the host a layer late,
    /// overlapping the next layer's attention: their output joins the stream after that
    /// attention instead of before it, so the logits drift from the exact ones. 0 is exact.
    pub fn set_expert_deferral(&mut self, share: f32) {
        self.expert_deferral = share.clamp(0.0, 1.0);
    }

    /// Give up experts for speed, on every layer: keep at most `keep_fraction` of a token's
    /// experts (rounded up, the heaviest), drop the lightest while they carry under `drop_mass`
    /// of the weight, and replace (or, with `miss_drops`, drop) a light expert that is in no
    /// tier when its share is under `miss_below`. Zeros are exact routing.
    pub fn set_expert_truncation(
        &self,
        keep_fraction: f32,
        drop_mass: f32,
        miss_below: f32,
        miss_drops: bool,
    ) {
        let k = self.n_activated();
        let keep = if keep_fraction > 0.0 && keep_fraction < 1.0 {
            // A fraction that lands a hair over a whole count (two thirds of six) keeps that count.
            (((keep_fraction * k as f32) - 1e-3).ceil() as usize).clamp(1, k)
        } else {
            0
        };
        for b in self.blocks.iter().flatten() {
            *b.moe.truncation.write().unwrap() = super::moe::Truncation {
                keep,
                drop_mass,
                miss_below,
                miss_drops,
            };
        }
    }

    /// Ask for the routing that gives up the most disk reads within what `level` accepts of the
    /// exact output, to be calibrated when the backend starts.
    pub fn set_quality_policy(&self, level: QualityLevel, identity: String, ids: Vec<u32>) {
        *self.quality.lock().unwrap() = Some((level, identity, ids));
    }

    /// The policy `set_quality_policy` left, once.
    pub fn take_quality_policy(&self) -> Option<(QualityLevel, String, Vec<u32>)> {
        self.quality.lock().unwrap().take()
    }

    /// Set every layer's lossy routing options as given.
    pub fn set_truncation(&self, t: super::moe::Truncation) {
        for b in self.blocks.iter().flatten() {
            *b.moe.truncation.write().unwrap() = t;
        }
    }

    /// The disk reads the lossy options saved since last asked, summed over layers.
    pub fn take_saved(&self) -> u64 {
        self.blocks
            .iter()
            .flatten()
            .map(|b| b.moe.saved.swap(0, std::sync::atomic::Ordering::Relaxed))
            .sum()
    }

    /// Per layer, the experts the plan keeps in memory: those on a card now and the most routed-to
    /// ones the kept tier will hold, from the routing prior.
    pub fn plan_residency(&self) -> Vec<std::collections::HashSet<usize>> {
        self.blocks
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let Some(b) = b else {
                    return Default::default();
                };
                let mut set = b.moe.cards_hold();
                let slots = b.moe.experts.capacity();
                set.extend(
                    self.hot_experts()
                        .get(i)
                        .into_iter()
                        .flatten()
                        .copied()
                        .filter(|e| !set.contains(e))
                        .take(slots)
                        .collect::<Vec<_>>(),
                );
                set
            })
            .collect()
    }

    /// Treat, per layer, the given experts as the resident ones in place of the live tiers.
    pub fn set_resident_override(&self, per_layer: Option<&[std::collections::HashSet<usize>]>) {
        for (i, b) in self.blocks.iter().enumerate() {
            if let Some(b) = b {
                *b.moe.resident_override.write().unwrap() =
                    per_layer.and_then(|p| p.get(i).cloned());
            }
        }
    }

    /// How many routed experts per token the share allows to finish late.
    fn deferred_experts(&self) -> usize {
        (self.expert_deferral * self.n_activated() as f32).floor() as usize
    }

    /// Run every layer's routed experts through `offload`, or back on the CPU with `None`.
    pub fn offload_experts(
        &self,
        offload: Option<std::sync::Arc<crate::inference::offload::experts::ExpertOffload>>,
    ) {
        for b in self.blocks.iter().flatten() {
            *b.moe.offload.write().unwrap() = offload.clone();
        }
    }

    /// Resize every layer's expert hot cache to `slots` entries.
    pub fn set_expert_cache(&self, slots: usize) {
        for b in self.blocks.iter().flatten() {
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
        let started = std::time::Instant::now();
        let depth = self.readahead_pin.unwrap_or_else(|| self.readahead.value());
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
        let defer = self.deferred_experts();
        let mut pending: Option<(usize, super::moe::DeferredExperts, Vec<f32>)> = None;
        for (i, block) in self.blocks.iter().enumerate() {
            let Some(block) = block.as_ref() else {
                continue;
            };
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
            if depth > 0 {
                if let Some(Some(ahead)) = self.blocks.get(i + depth) {
                    crate::inference::offload::stage("expert read-ahead", || {
                        ahead.advise_experts(&x)
                    })?;
                }
            }
            let (cos, sin) = if self.ratios.get(i).copied().unwrap_or(0) == 0 {
                (&cos0, &sin0)
            } else {
                (&cosb, &sinb)
            };
            if defer == 0 {
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
                continue;
            }
            // The experts the layer above left running overlap this layer's attention; their
            // share joins the stream before this layer's FFN reads it.
            let (xa, am_pre) = block.forward_decode_attn(
                &x,
                &pre_mix,
                cos,
                sin,
                pos,
                &mut state.layers[i],
                &mut shared,
            )?;
            let xa = match pending.take() {
                Some((layer, job, post)) => self.block(layer)?.add_deferred(&xa, job, &post)?,
                None => xa,
            };
            let (nx, npm, left) = block.forward_decode_ffn_deferring(&xa, &am_pre, defer)?;
            pending = left.map(|(job, post)| (i, job, post));
            x = nx;
            pre_mix = npm;
        }
        // Nothing stays running past the token: the head reads the stream whole.
        if let Some((layer, job, post)) = pending.take() {
            x = self.block(layer)?.add_deferred(&x, job, &post)?;
        }

        let h = hc_pre(&x, &pre_mix)?;
        let h = crate::inference::offload::stage("final norm", || {
            rms_norm(&h, &self.norm, self.rms_eps)
        })?;
        let logits =
            crate::inference::offload::stage("head", || self.output.apply(&h.reshape((1, dim))?))?;
        state.pos += 1;
        if self.readahead_pin.is_none() {
            self.readahead
                .note_token(started.elapsed().as_nanos() as u64);
        }
        Ok(logits)
    }

    /// Hold the read-ahead depth at `depth` (or hand it back to the controller with `None`).
    pub fn pin_readahead(&mut self, depth: Option<usize>) {
        self.readahead_pin = depth;
    }

    /// Run layers `[from, to)` of a decode. This is `forward_decode` split at a layer boundary, so
    /// a pipeline can place one contiguous run of layers on each node: `from == 0` embeds the token
    /// and starts the residual; `to == n_layers` finishes with the final norm and head and advances
    /// the position; an interior range returns the boundary the next range resumes from - the
    /// residual stream, its collapse weights, the band state that flows down within a token, and the
    /// engram recent this token pushed. Each range advances only its own layers' caches
    /// (`state.layers[from..to]`), which on a cluster is exactly the layers that node holds. The
    /// split is bit-for-bit `forward_decode`: the same per-layer block, only the loop is cut.
    pub fn forward_decode_range(
        &self,
        input: DecodeRangeIn,
        from: usize,
        to: usize,
        state: &mut DecodeState,
    ) -> Result<DecodeRangeOut> {
        let pos = state.pos;
        let (dim, hc) = (self.dim, self.hc_mult);
        let n = pos + 1;
        let (cos0, sin0) = rope_table(self.rope_head_dim, n, 0, self.rope_theta, self.rope_factor)?;
        let (cosb, sinb) = rope_table(
            self.rope_head_dim,
            n,
            self.original_seq_len,
            self.compress_rope_theta,
            self.rope_factor,
        )?;

        // Node 0 embeds the token and pushes the engram recent once; a later range resumes from the
        // boundary the previous one produced.
        let (mut x, mut pre_mix, mut shared, recent) = match input {
            DecodeRangeIn::Token(token) => {
                let row = crate::inference::offload::stage("embed", || {
                    self.embed.narrow(0, token as usize, 1)
                })?;
                let x = row
                    .reshape((1, 1, dim))?
                    .unsqueeze(2)?
                    .broadcast_as((1, 1, hc, dim))?
                    .contiguous()?;
                let pre_mix = vec![{
                    let mut v = vec![0f32; hc];
                    v[0] = 1.0;
                    v
                }];
                let recent = self
                    .engram_cfg
                    .as_ref()
                    .map(|ecfg| state.engram.push(ecfg, token));
                (x, pre_mix, SharedAttn::default(), recent)
            }
            DecodeRangeIn::Resume(b) => (b.x, b.pre_mix, b.shared, b.recent),
        };

        let cap = state.capture_layers.clone();
        if from == 0 && !cap.is_empty() {
            state.captured = vec![Vec::new(); cap.len()];
        }
        for i in from..to {
            if let (Some(engram), Some(ecfg), Some(recent)) =
                (&self.engrams[i], &self.engram_cfg, &recent)
            {
                let which = ecfg.layer_ids.iter().position(|&l| l == i).unwrap_or(0);
                x = engram.forward(&x, &[ecfg.hash(which, recent)])?;
            }
            if let Some(ci) = cap.iter().position(|&l| l == i) {
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
            let (nx, npm) = self.block(i)?.forward_decode(
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

        if to >= self.blocks.len() {
            let h = hc_pre(&x, &pre_mix)?;
            let h = crate::inference::offload::stage("final norm", || {
                rms_norm(&h, &self.norm, self.rms_eps)
            })?;
            let logits = crate::inference::offload::stage("head", || {
                self.output.apply(&h.reshape((1, dim))?)
            })?;
            state.pos += 1;
            Ok(DecodeRangeOut::Logits(logits))
        } else {
            Ok(DecodeRangeOut::Boundary(DecodeBoundary {
                x,
                pre_mix,
                shared,
                recent,
            }))
        }
    }

    /// Decode one token by running the stack in `segments` (half-open layer ranges that must
    /// partition the layers in order), each a `forward_decode_range`, the boundary handed from one
    /// to the next. This is the single-process rehearsal of the pipeline a cluster runs with each
    /// segment on its node, and equals `forward_decode` bit for bit whatever the cut. The engine's
    /// generalizable form drives `ModelBackend::decode_range` the same way across nodes.
    pub fn pipeline_decode(
        &self,
        token: u32,
        segments: &[(usize, usize)],
        state: &mut DecodeState,
    ) -> Result<Tensor> {
        let n = self.blocks.len();
        let mut next = 0;
        for &(a, b) in segments {
            if a != next || b <= a || b > n {
                return Err(Error::msg(format!(
                    "pipeline_decode: segment ({a},{b}) does not continue a partition of {n} layers"
                )));
            }
            next = b;
        }
        if next != n {
            return Err(Error::msg(
                "pipeline_decode: segments do not cover the stack",
            ));
        }
        let mut input = DecodeRangeIn::Token(token);
        for &(a, b) in segments {
            match self.forward_decode_range(input, a, b, state)? {
                DecodeRangeOut::Logits(l) => return Ok(l),
                DecodeRangeOut::Boundary(bd) => input = DecodeRangeIn::Resume(bd),
            }
        }
        Err(Error::msg(
            "pipeline_decode: last segment yielded no logits",
        ))
    }

    /// The head node of a pipeline: run layers `[0, cut)` for `token`, ship the ratio-0 boundary
    /// over the kept `link`, and read back the tail node's logits. The reply is f32 bytes, the
    /// vocab row. Reusable serving code, the same path the socket test rehearses.
    pub fn pipeline_head_step(
        &self,
        token: u32,
        cut: usize,
        state: &mut DecodeState,
        link: &mut crate::distributed::wire_link::WireLink,
    ) -> Result<Vec<f32>> {
        let bd = match self.forward_decode_range(DecodeRangeIn::Token(token), 0, cut, state)? {
            DecodeRangeOut::Boundary(bd) => bd,
            DecodeRangeOut::Logits(_) => {
                return Err(Error::msg("pipeline_head_step: cut runs the whole stack"))
            }
        };
        // The head runs a range on its own state, so its position advances per token here, the way
        // the whole-stack path advances it on the logits. Without this the head's KV and rope stay
        // at position zero and a band layer indexes past the single cached step on the next token.
        state.pos += 1;
        let blob = bd.to_bytes()?;
        let (_h, reply) = link
            .request(
                cut as u32,
                crate::distributed::wire::WireDType::Q8,
                vec![blob.len() as u32],
                &blob,
            )
            .map_err(|e| Error::msg(format!("pipeline_head_step: link: {e}")))?;
        if reply.len() % 4 != 0 {
            return Err(Error::msg(
                "pipeline_head_step: logits reply not f32-aligned",
            ));
        }
        Ok(reply
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    /// One head-side decode step over a multiplexed link. Same as `pipeline_head_step`, but the
    /// boundary carries its `seq_id` ahead of the bytes so a tail serving several sequences on one
    /// connection knows which decode state to resume, and the link is a shared `MuxLink` so many
    /// sequences run their steps concurrently on it - the reply is matched to its request by the
    /// link, not by call order. The model is read-only here and each sequence owns its `state`, so
    /// the concurrent calls share nothing that mutates.
    /// One head-side decode step over a plain synchronous link, carrying the same `seq_id`-tagged
    /// boundary a mux tail expects. For a single sequence this is the whole of the head's work and
    /// wants no reader thread: send the boundary, read the logits, on the one caller's thread.
    pub fn pipeline_head_step_wire(
        &self,
        token: u32,
        cut: usize,
        seq_id: u32,
        state: &mut DecodeState,
        link: &mut crate::distributed::wire_link::WireLink,
    ) -> Result<Vec<f32>> {
        let bd = match self.forward_decode_range(DecodeRangeIn::Token(token), 0, cut, state)? {
            DecodeRangeOut::Boundary(bd) => bd,
            DecodeRangeOut::Logits(_) => {
                return Err(Error::msg(
                    "pipeline_head_step_wire: cut runs the whole stack",
                ))
            }
        };
        // Advance this state's position per token; see `pipeline_head_step`.
        state.pos += 1;
        let blob = bd.to_bytes()?;
        let mut payload = Vec::with_capacity(4 + blob.len());
        payload.extend_from_slice(&seq_id.to_le_bytes());
        payload.extend_from_slice(&blob);
        let (_h, reply) = link
            .request(
                cut as u32,
                crate::distributed::wire::WireDType::Q8,
                vec![payload.len() as u32],
                &payload,
            )
            .map_err(|e| Error::msg(format!("pipeline_head_step_wire: link: {e}")))?;
        if reply.len() % 4 != 0 {
            return Err(Error::msg(
                "pipeline_head_step_wire: logits reply not f32-aligned",
            ));
        }
        Ok(reply
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    pub fn pipeline_head_step_mux(
        &self,
        token: u32,
        cut: usize,
        seq_id: u32,
        state: &mut DecodeState,
        link: &crate::distributed::wire_mux::MuxLink,
    ) -> Result<Vec<f32>> {
        let bd = match self.forward_decode_range(DecodeRangeIn::Token(token), 0, cut, state)? {
            DecodeRangeOut::Boundary(bd) => bd,
            DecodeRangeOut::Logits(_) => {
                return Err(Error::msg(
                    "pipeline_head_step_mux: cut runs the whole stack",
                ))
            }
        };
        // Advance this state's position per token; see `pipeline_head_step`.
        state.pos += 1;
        let blob = bd.to_bytes()?;
        let mut payload = Vec::with_capacity(4 + blob.len());
        payload.extend_from_slice(&seq_id.to_le_bytes());
        payload.extend_from_slice(&blob);
        let (_h, reply) = link
            .request(
                cut as u32,
                crate::distributed::wire::WireDType::Q8,
                vec![payload.len() as u32],
                &payload,
            )
            .map_err(|e| Error::msg(format!("pipeline_head_step_mux: link: {e}")))?;
        if reply.len() % 4 != 0 {
            return Err(Error::msg(
                "pipeline_head_step_mux: logits reply not f32-aligned",
            ));
        }
        Ok(reply
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    /// The tail node of a pipeline: handle one boundary frame - rebuild it, run layers `[from, n)`
    /// on `state`, and return the logits as an f32-byte reply. Given to `wire_link::serve_once` as
    /// its handler. Errors surface as an empty reply, which the head reads as a short row.
    pub fn pipeline_tail_reply(
        &self,
        payload: &[u8],
        from: usize,
        state: &mut DecodeState,
    ) -> Vec<u8> {
        let logits = DecodeBoundary::from_bytes(payload)
            .and_then(|bd| {
                self.forward_decode_range(DecodeRangeIn::Resume(bd), from, self.blocks.len(), state)
            })
            .and_then(|out| match out {
                DecodeRangeOut::Logits(l) => l.flatten_all()?.to_vec1::<f32>(),
                DecodeRangeOut::Boundary(_) => Err(Error::msg(
                    "pipeline_tail_reply: tail did not reach the head",
                )),
            });
        match logits {
            Ok(v) => {
                let mut bytes = Vec::with_capacity(v.len() * 4);
                for f in &v {
                    bytes.extend_from_slice(&f.to_le_bytes());
                }
                bytes
            }
            Err(_) => Vec::new(),
        }
    }

    /// Run this node as a pipeline tail: accept one head connection on `listener` and serve every
    /// boundary that arrives over it, running layers `[from, n)` on one decode state and replying
    /// with the logits, until the head hangs up. This is the whole of a tail node's serving loop -
    /// additive, touching no other serving path. `from == 0` would run the whole stack, which a
    /// head node does instead.
    pub fn serve_pipeline_tail(
        &self,
        from: usize,
        listener: std::net::TcpListener,
    ) -> std::io::Result<()> {
        use crate::distributed::wire::{FrameHeader, WireDType};
        let mut state = self.new_decode_state();
        let n = self.blocks.len();
        crate::distributed::wire_link::serve_once(listener, |_h, payload| {
            let bytes = self.pipeline_tail_reply(payload, from, &mut state);
            let n_f32 = (bytes.len() / 4) as u32;
            (
                FrameHeader {
                    request_id: 0,
                    layer_id: n as u32,
                    dtype: WireDType::F32,
                    shape: vec![n_f32],
                },
                bytes,
            )
        })
    }

    /// Run this node as a pipeline tail serving several sequences multiplexed on one connection.
    /// Each boundary frame carries its `seq_id` in the first four bytes, ahead of the boundary the
    /// head sent; the tail keeps one decode state per sequence and runs the frames as they arrive,
    /// one at a time, so the forward path is never entered concurrently on this node. Replies echo
    /// the frame's request id (`serve_connection` does that), so a head multiplexing many sequences
    /// matches each logits row to the boundary it sent. Additive, touching no other serving path.
    pub fn serve_pipeline_tail_mux(
        &self,
        from: usize,
        listener: std::net::TcpListener,
    ) -> std::io::Result<()> {
        self.serve_pipeline_segment(from, self.blocks.len(), None, listener)
    }

    /// Answer one boundary frame for an interior or final segment `[from, to)`. Runs the range on
    /// this node's state; if `to` reaches the end it returns the logits, otherwise it relays its own
    /// boundary to the next node over `down` and returns the logits that come back. An interior
    /// segment with no downstream, or any failure, answers an empty row - which the caller reads as
    /// a short reply, not a resumed state.
    fn pipeline_segment_reply(
        &self,
        payload: &[u8],
        from: usize,
        to: usize,
        seq_id: u32,
        state: &mut DecodeState,
        down: Option<&mut crate::distributed::wire_link::WireLink>,
    ) -> Vec<u8> {
        let out = DecodeBoundary::from_bytes(payload)
            .and_then(|bd| self.forward_decode_range(DecodeRangeIn::Resume(bd), from, to, state));
        match out {
            Ok(DecodeRangeOut::Logits(l)) => {
                match l.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                    Ok(v) => {
                        let mut bytes = Vec::with_capacity(v.len() * 4);
                        for f in &v {
                            bytes.extend_from_slice(&f.to_le_bytes());
                        }
                        bytes
                    }
                    Err(_) => Vec::new(),
                }
            }
            Ok(DecodeRangeOut::Boundary(bd2)) => {
                // An interior range yields a boundary, not logits, so its position does not advance
                // inside the forward the way the final segment's does; advance it here, as the head
                // does, or the band layers index past this segment's single cached step next token.
                state.pos += 1;
                let Some(down) = down else {
                    return Vec::new();
                };
                let blob = match bd2.to_bytes() {
                    Ok(b) => b,
                    Err(_) => return Vec::new(),
                };
                let mut fwd = Vec::with_capacity(4 + blob.len());
                fwd.extend_from_slice(&seq_id.to_le_bytes());
                fwd.extend_from_slice(&blob);
                match down.request(
                    to as u32,
                    crate::distributed::wire::WireDType::Q8,
                    vec![fwd.len() as u32],
                    &fwd,
                ) {
                    Ok((_h, reply)) => reply,
                    Err(_) => Vec::new(),
                }
            }
            Err(_) => Vec::new(),
        }
    }

    /// Run this node as one segment of a pipeline of any length: serve layers `[from, to)` for every
    /// boundary a head or an upstream segment sends, keeping one decode state per sequence. `next` is
    /// the node running `[to, ...)`; a final segment (`to == n`) passes `None` and replies with the
    /// logits, an interior segment relays to `next` and returns what it answers. This is the whole of
    /// a segment node's serving loop, and a two-node tail is just the `to == n`, `next == None` case.
    pub fn serve_pipeline_segment(
        &self,
        from: usize,
        to: usize,
        next: Option<std::net::SocketAddr>,
        listener: std::net::TcpListener,
    ) -> std::io::Result<()> {
        use crate::distributed::wire::{FrameHeader, WireDType};
        use std::collections::HashMap;
        let mut down = match next {
            Some(a) => Some(crate::distributed::wire_link::WireLink::connect(a)?),
            None => None,
        };
        let mut states: HashMap<u32, DecodeState> = HashMap::new();
        let n = self.blocks.len();
        crate::distributed::wire_link::serve_once(listener, |_h, payload| {
            let bytes = if payload.len() < 4 {
                Vec::new()
            } else {
                let seq_id = u32::from_le_bytes(payload[..4].try_into().unwrap());
                let state = states
                    .entry(seq_id)
                    .or_insert_with(|| self.new_decode_state());
                self.pipeline_segment_reply(&payload[4..], from, to, seq_id, state, down.as_mut())
            };
            let n_f32 = (bytes.len() / 4) as u32;
            (
                FrameHeader {
                    request_id: 0,
                    layer_id: n as u32,
                    dtype: WireDType::F32,
                    shape: vec![n_f32],
                },
                bytes,
            )
        })
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
        self.forward_block(tokens, state, false, &[])
            .map(|(logits, _)| logits)
    }

    /// Continue a sequence whose state stands at `state.pos` over `tokens`, returning the
    /// next-token logits after the last one, `[1, vocab]`. This is the prefill of a prompt's
    /// remainder after a restored prefix: one layer-major pass over the block, attending over the
    /// restored window and compressed rows and the new ones, each layer's experts read once for
    /// all of it, and only the last position through the head.
    pub fn prefill_continue(&self, tokens: &[u32], state: &mut DecodeState) -> Result<Tensor> {
        self.prefill_continue_grid(tokens, state, &[])
            .map(|(logits, _)| logits)
    }

    /// `prefill_continue`, also keeping every layer's window at each token count of `grid` that
    /// falls strictly inside the block: `windows[k][layer]` for `grid[k]`, as `prefill_into_grid`.
    /// The block runs as one layer-major prefill onto the state, as a cold prompt does.
    pub fn prefill_continue_grid(
        &self,
        tokens: &[u32],
        state: &mut DecodeState,
        grid: &[usize],
    ) -> Result<(Tensor, Vec<Vec<Vec<u8>>>)> {
        self.prefill_into_grid(tokens, state, grid)
    }

    /// The block as `forward_verify_batch` runs it, token by token through attention, for the
    /// tests that hold the batched continuation against it.
    #[cfg(test)]
    pub fn prefill_continue_stepped(
        &self,
        tokens: &[u32],
        state: &mut DecodeState,
    ) -> Result<Tensor> {
        self.forward_block(tokens, state, true, &[]).map(|(l, _)| l)
    }

    /// The block pass behind `forward_verify_batch` and `prefill_continue`: the logits of every
    /// position, or of the last one alone when `last_only`.
    fn forward_block(
        &self,
        tokens: &[u32],
        state: &mut DecodeState,
        last_only: bool,
        grid: &[usize],
    ) -> Result<(Tensor, Vec<Vec<Vec<u8>>>)> {
        if tokens.is_empty() {
            return Err(Error::msg("forward_block: no tokens"));
        }
        let grid: Vec<usize> = grid
            .iter()
            .copied()
            .filter(|&p| p > state.pos && p < state.pos + tokens.len())
            .collect();
        let empty = super::snapshot::window_section(&Default::default());
        let mut windows: Vec<Vec<Vec<u8>>> = vec![vec![empty; self.blocks.len()]; grid.len()];
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
            let Some(block) = block.as_ref() else {
                continue;
            };
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
                if let Some(k) = grid.iter().position(|&p| p == pos + j + 1) {
                    windows[k][i] = super::snapshot::window_section(&state.layers[i].win);
                }
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

        let (x, pre_mix, rows) = if last_only {
            (
                x.narrow(1, k - 1, 1)?.contiguous()?,
                pre_mix[k - 1..].to_vec(),
                1,
            )
        } else {
            (x, pre_mix, k)
        };
        let h = hc_pre(&x, &pre_mix)?;
        let h = rms_norm(&h, &self.norm, self.rms_eps)?;
        let logits = self.output.apply(&h.reshape((rows, dim))?)?;
        state.pos += k;
        Ok((logits, windows))
    }

    /// Decode one token for each of several independent sequences at once, returning the
    /// next-token logits per sequence `[reqs.len(), vocab]`. Attention runs per sequence over its
    /// own ring at its own position - a sequence never reads another's keys - and the FFN with its
    /// MoE runs once over the whole batch, so each active expert is read once for all the
    /// sequences that route to it. This is where a batch earns its throughput: the memory-bound
    /// expert reads amortise across the sequences while single-token decode leaves the card idle.
    /// Each sequence's `pos` advances by one. Capture (the draft's hidden) is not taken here.
    pub fn forward_decode_batch(&self, reqs: &mut [(u32, &mut DecodeState)]) -> Result<Tensor> {
        if reqs.is_empty() {
            return Err(Error::msg("forward_decode_batch: no requests"));
        }
        let (dim, hc) = (self.dim, self.hc_mult);
        let n = reqs.len();

        // Each sequence's token embedded, broadcast into the hyper-connection copies: [1, n, hc, d].
        let mut rows = Vec::with_capacity(n);
        for (t, _) in reqs.iter() {
            rows.push(self.embed.narrow(0, *t as usize, 1)?);
        }
        let refs: Vec<&Tensor> = rows.iter().collect();
        let emb = Tensor::cat(&refs, 0)?.reshape((1, n, dim))?;
        let mut x = emb
            .unsqueeze(2)?
            .broadcast_as((1, n, hc, dim))?
            .contiguous()?;

        // Rope tables long enough for the furthest-along sequence; each reads its own row `pos`.
        let len = reqs.iter().map(|(_, s)| s.pos).max().unwrap_or(0) + 1;
        let (cos0, sin0) = rope_table(
            self.rope_head_dim,
            len,
            0,
            self.rope_theta,
            self.rope_factor,
        )?;
        let (cosb, sinb) = rope_table(
            self.rope_head_dim,
            len,
            self.original_seq_len,
            self.compress_rope_theta,
            self.rope_factor,
        )?;

        // One engram recent per sequence, from that sequence's own history.
        let recents: Vec<_> = reqs
            .iter_mut()
            .map(|(t, s)| self.engram_cfg.as_ref().map(|ecfg| s.engram.push(ecfg, *t)))
            .collect();

        // The collapse weights into the top layer and the band state, one per sequence.
        let mut pre_mix: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                let mut v = vec![0f32; hc];
                v[0] = 1.0;
                v
            })
            .collect();
        let mut shareds: Vec<SharedAttn> = (0..n).map(|_| SharedAttn::default()).collect();

        for (i, block) in self.blocks.iter().enumerate() {
            let Some(block) = block.as_ref() else {
                continue;
            };
            let (cos, sin) = if self.ratios.get(i).copied().unwrap_or(0) == 0 {
                (&cos0, &sin0)
            } else {
                (&cosb, &sinb)
            };
            // Attention half, one sequence at a time, each over its own ring at its own position.
            let mut attn_rows = Vec::with_capacity(n);
            let mut am_pre: Vec<Vec<f32>> = Vec::with_capacity(n);
            for j in 0..n {
                let pos_j = reqs[j].1.pos;
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
                    pos_j,
                    &mut reqs[j].1.layers[i],
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
            // FFN half, once over the whole batch: one read of each expert for all the sequences.
            let (nx, npm) = block.forward_prefill_ffn(&xattn, &am_pre)?;
            x = nx;
            pre_mix = npm;
        }

        let h = hc_pre(&x, &pre_mix)?;
        let h = rms_norm(&h, &self.norm, self.rms_eps)?;
        let logits = self.output.apply(&h.reshape((n, dim))?)?;
        for (_, s) in reqs.iter_mut() {
            s.pos += 1;
        }
        Ok(logits)
    }

    /// Greedily generate up to `max_new[i]` tokens for each prompt at once. Every prompt is
    /// prefilled, then the still-running sequences decode one token together each step through
    /// `forward_decode_batch`, so each active expert is read once for the batch; the batch shrinks
    /// as sequences hit their length or `eos`, so a long generation never holds a finished one's
    /// place. Returns the generated tokens per prompt, the prompt excluded. Greedy, so a sequence
    /// takes the argmax path it would take decoded alone.
    pub fn generate_batch(
        &self,
        prompts: &[&[u32]],
        max_new: &[usize],
        eos: Option<u32>,
    ) -> Result<Vec<Vec<u32>>> {
        assert_eq!(prompts.len(), max_new.len());
        let n = prompts.len();
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        };
        // Prefill each prompt; the first token to decode is the argmax of its last position.
        let mut states: Vec<DecodeState> = Vec::with_capacity(n);
        let mut next: Vec<u32> = Vec::with_capacity(n);
        let mut out: Vec<Vec<u32>> = (0..n).map(|_| Vec::new()).collect();
        let mut done: Vec<bool> = vec![false; n];
        for (i, p) in prompts.iter().enumerate() {
            let mut st = self.new_decode_state();
            let logits = self.prefill_into(p, &mut st)?;
            let (rows, vocab) = logits.dims2()?;
            let flat = logits.flatten_all()?.to_vec1::<f32>()?;
            next.push(argmax(&flat[(rows - 1) * vocab..rows * vocab]));
            states.push(st);
            if max_new[i] == 0 {
                done[i] = true;
            }
        }
        // Each step decodes one token for every running sequence at once; the token just decoded is
        // committed and the returned logits give the next.
        loop {
            let mut idxs: Vec<usize> = Vec::new();
            let mut batch: Vec<(u32, &mut DecodeState)> = Vec::new();
            for (i, s) in states.iter_mut().enumerate() {
                if !done[i] {
                    idxs.push(i);
                    batch.push((next[i], s));
                }
            }
            if batch.is_empty() {
                break;
            }
            let logits = self.forward_decode_batch(&mut batch)?;
            let (_rows, vocab) = logits.dims2()?;
            let flat = logits.flatten_all()?.to_vec1::<f32>()?;
            drop(batch);
            for (row, &i) in idxs.iter().enumerate() {
                let tok = next[i];
                out[i].push(tok);
                if out[i].len() >= max_new[i] || Some(tok) == eos {
                    done[i] = true;
                    continue;
                }
                next[i] = argmax(&flat[row * vocab..(row + 1) * vocab]);
            }
        }
        Ok(out)
    }
}

/// The model as the engine's streamed placement sees it: resident footprint, routing and experts,
/// with no V4.1 concept crossing the seam. The figures delegate to the model's own accounting.
impl crate::inference::offload::streamed::StreamedModel for DeepseekV41Model {
    fn resident_bytes(&self) -> usize {
        self.resident_path_bytes()
    }
    fn peak_transient_bytes(&self, tokens: usize) -> usize {
        self.transient_bytes(tokens)
    }
    fn active_experts(&self) -> usize {
        self.n_activated()
    }
    fn layer_count(&self) -> usize {
        self.n_layers()
    }
    fn routing_prior(&self) -> &[Vec<usize>] {
        self.hot_experts()
    }
    fn expert(
        &self,
        layer: usize,
        id: usize,
    ) -> Option<std::sync::Arc<crate::inference::offload::experts::Expert>> {
        self.expert_store(layer)?.fetch(id).ok()
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

    /// The model declares its expert cost shape from a real held expert: dims from the loaded
    /// weight, the stored quant formats. This is what a node measures its miss cost against.
    #[test]
    fn declares_its_expert_cost_shape() {
        let path = tmp("shape.gguf");
        write_gguf_with_metadata(&path, &real_md(3), &real_entries()).unwrap();
        let g = open_mapped(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let cfg = DeepseekV41Config::from_gguf(&g.content).unwrap();
        let model = DeepseekV41Model::load(&g, &cfg).unwrap();
        let (dim, inter, _gate_dt, _down_dt) =
            model.expert_cost_shape().expect("shape from a held expert");
        assert_eq!((dim, inter), (DIM, INTER));
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

    /// A snapshot written after a prefix and read back is that prefix's state to the bit: the
    /// logits of the remainder prefilled onto it equal, bit for bit, those of the same remainder
    /// prefilled onto the live state, and the decode that follows too. Against a cold prefill of the
    /// whole prompt - another summation order, the one `prefill_into_leaves_the_decode_state`
    /// bounds - the logits agree to rounding and name the same token. Splits fall inside a
    /// compression group, shorter than the window, on the window, past it and on a group boundary.
    #[test]
    fn a_restored_snapshot_continues_exactly_like_the_live_state() {
        use super::super::snapshot::{decode, encode, identity, same_state};
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
        let id = identity(&[b"synthetic"]);
        let layouts: [(&[u32], &[u32], &[u32]); 2] =
            [(&[0, 2, 0], &[1], &[1]), (&[0, 2, 2, 0], &[1], &[1, 3])];
        for (ratios, kv_src, index_src) in layouts {
            let (model, _) = synthetic(ratios, kv_src, index_src, 0);
            let tokens: Vec<u32> = (0..80)
                .map(|i| (i * 7 + 3) % super::tests::VOCAB as u32)
                .collect();
            let mut cold_state = model.new_decode_state();
            let cold = vecf_of(&model.prefill_into(&tokens, &mut cold_state).unwrap());
            for split in [1usize, 2, 3, 5, 17, 32, 33, 41, 64] {
                let mut live = model.new_decode_state();
                let at_split = vecf_of(&model.prefill_into(&tokens[..split], &mut live).unwrap());
                let bytes = encode(&live, Some(&at_split), &id);
                // The rows are on their grids, so the file is a fraction of the f32 state.
                let floats = live
                    .layers
                    .iter()
                    .map(|l| {
                        l.win.iter().map(Vec::len).sum::<usize>()
                            + l.comp_kv.len()
                            + l.index_k.len()
                    })
                    .sum::<usize>()
                    * 4;
                assert!(
                    bytes.len() < floats / 2 + at_split.len() * 4 + 4096 + live.layers.len() * 8192,
                    "{ratios:?} split {split}: {} bytes for {floats} of rows",
                    bytes.len()
                );
                let back = decode(&bytes, &id, model.n_layers()).unwrap();
                assert!(same_state(&live, &back.state), "{ratios:?} split {split}");
                assert_eq!(back.logits.as_deref(), Some(&at_split[..]));
                let mut restored = back.state;
                let rest = &tokens[split..];
                let want = vecf_of(&model.prefill_continue(rest, &mut live).unwrap());
                let got = vecf_of(&model.prefill_continue(rest, &mut restored).unwrap());
                assert_eq!(got, want, "{ratios:?} split {split}: restored against live");
                assert!(same_state(&live, &restored), "{ratios:?} split {split}");
                assert_eq!(got.len(), cold.len());
                let r = rel(&got, &cold);
                assert!(r < 1e-4, "{ratios:?} split {split}: against cold rel {r}");
                assert_eq!(argmax(&got), argmax(&cold), "{ratios:?} split {split}");
                // The decode after the remainder agrees too: bit for bit with the live twin, and to
                // rounding with the state a cold prefill of everything left.
                let mut cold_twin = cold_state.clone();
                for t in [3u32, 9, 11] {
                    let a = vecf_of(&model.forward_decode(t, &mut live).unwrap());
                    let b = vecf_of(&model.forward_decode(t, &mut restored).unwrap());
                    assert_eq!(a, b, "{ratios:?} split {split}: decode after restore");
                    let c = vecf_of(&model.forward_decode(t, &mut cold_twin).unwrap());
                    assert!(
                        rel(&b, &c) < 1e-3,
                        "{ratios:?} split {split}: against cold rel {}",
                        rel(&b, &c)
                    );
                }
            }
        }
    }

    /// A state cut into a manifest and block files comes back to the bit when the blocks are
    /// read in order, at every split: the blocks cover whole groups, the manifest the tail and the
    /// window. The blocks of a state that later grew are byte-identical to the earlier ones, which
    /// is what lets a store keep each once.
    #[test]
    fn a_state_split_into_blocks_restores_exactly_and_shares_its_blocks() {
        use super::super::snapshot::{
            block_count, decode_split, encode_block, encode_manifest, identity, same_state,
            BlockLayout,
        };
        let id = identity(&[b"synthetic"]);
        let layouts: [(&[u32], &[u32], &[u32]); 2] =
            [(&[0, 2, 0], &[1], &[1]), (&[0, 2, 2, 0], &[1], &[1, 3])];
        for (ratios, kv_src, index_src) in layouts {
            let (model, _) = synthetic(ratios, kv_src, index_src, 0);
            let tokens: Vec<u32> = (0..80)
                .map(|i| (i * 7 + 3) % super::tests::VOCAB as u32)
                .collect();
            let layout = BlockLayout {
                ratios: model.compress_ratios(),
                block_tokens: 8,
            };
            for split in [1usize, 5, 17, 32, 41, 64] {
                let mut live = model.new_decode_state();
                model.prefill_into(&tokens[..split], &mut live).unwrap();
                let nb = block_count(&live, &layout);
                assert_eq!(nb, split / 8, "{ratios:?} split {split}");
                let (manifest, n) = encode_manifest(&live, None, &id, Some(&layout));
                assert_eq!(n, nb);
                let blocks: Vec<Vec<u8>> =
                    (0..nb).map(|b| encode_block(&live, b, &layout)).collect();
                let back = decode_split(&manifest, &blocks, &id, model.n_layers()).unwrap();
                assert!(same_state(&live, &back.state), "{ratios:?} split {split}");
                // Too few or too many blocks are refused.
                if nb > 0 {
                    assert!(decode_split(&manifest, &blocks[1..], &id, model.n_layers()).is_err());
                }
                let mut restored = back.state;
                let want = vecf_of(&model.prefill_continue(&tokens[split..], &mut live).unwrap());
                let got = vecf_of(
                    &model
                        .prefill_continue(&tokens[split..], &mut restored)
                        .unwrap(),
                );
                assert_eq!(got, want, "{ratios:?} split {split}");
            }
            // One live sequence snapshotted at two lengths shares its blocks byte for byte.
            let mut live = model.new_decode_state();
            model.prefill_into(&tokens[..24], &mut live).unwrap();
            let early: Vec<Vec<u8>> = (0..3).map(|b| encode_block(&live, b, &layout)).collect();
            model.prefill_continue(&tokens[24..48], &mut live).unwrap();
            for (b, blk) in early.iter().enumerate() {
                assert_eq!(
                    &encode_block(&live, b, &layout),
                    blk,
                    "{ratios:?} block {b}"
                );
            }
            assert_eq!(block_count(&live, &layout), 6);
        }
    }

    /// The states a prefill passes on the way, kept at block boundaries, are states of their own
    /// prefixes: restored from a manifest of the window at that point and the blocks of the final
    /// state, the remainder prefilled onto them gives the logits a cold prefill of the whole prompt
    /// gives, to rounding and with the same token, and the logits of a prefix prefilled alone. The
    /// pass need not be cut anywhere for this: the windows are taken as it goes. The same holds for
    /// a prompt continued from a restored state, whose boundaries lie inside the continued block.
    #[test]
    fn grid_states_taken_during_a_prefill_restore_like_prefixes_of_their_own() {
        use super::super::snapshot::{
            decode_split, encode_block, encode_grid_manifest, identity, BlockLayout, GridPoint,
        };
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
        let id = identity(&[b"synthetic"]);
        let layouts: [(&[u32], &[u32], &[u32]); 2] =
            [(&[0, 2, 0], &[1], &[1]), (&[0, 2, 2, 0], &[1], &[1, 3])];
        for (ratios, kv_src, index_src) in layouts {
            let (model, _) = synthetic(ratios, kv_src, index_src, 0);
            let tokens: Vec<u32> = (0..80)
                .map(|i| (i * 7 + 3) % super::tests::VOCAB as u32)
                .collect();
            let layout = BlockLayout {
                ratios: model.compress_ratios(),
                block_tokens: 8,
            };
            let grid: Vec<usize> = (1..10).map(|k| k * 8).collect();
            // One pass from nothing, and one continued from a restored prefix of 20 tokens.
            let mut cold_state = model.new_decode_state();
            let (cold, windows) = model
                .prefill_into_grid(&tokens, &mut cold_state, &grid)
                .unwrap();
            let cold = vecf_of(&cold);
            assert_eq!(windows.len(), grid.len());
            let mut head = model.new_decode_state();
            model.prefill_into(&tokens[..20], &mut head).unwrap();
            let (_, cont_windows) = model
                .prefill_continue_grid(&tokens[20..], &mut head, &grid)
                .unwrap();
            let sources = [
                ("prefill", &cold_state, grid.clone(), windows),
                (
                    "continued",
                    &head,
                    grid.iter().copied().filter(|&p| p > 20).collect(),
                    cont_windows,
                ),
            ];
            for (what, last, points, windows) in sources {
                assert_eq!(windows.len(), points.len(), "{ratios:?} {what}");
                for (p, win) in points.iter().copied().zip(&windows) {
                    let point = GridPoint {
                        pos: p,
                        windows: win,
                        engram: model.engram_after(&tokens[..p]),
                    };
                    let (manifest, nb) =
                        encode_grid_manifest(last, &point, &id, &layout).expect("boundary state");
                    assert_eq!(nb, p / 8);
                    let blocks: Vec<Vec<u8>> =
                        (0..nb).map(|b| encode_block(last, b, &layout)).collect();
                    let mut restored = decode_split(&manifest, &blocks, &id, model.n_layers())
                        .unwrap()
                        .state;
                    assert_eq!(restored.pos, p);
                    let got =
                        vecf_of(&model.prefill_continue(&tokens[p..], &mut restored).unwrap());
                    assert!(
                        rel(&got, &cold) < 1e-4,
                        "{ratios:?} {what} at {p}: rel {}",
                        rel(&got, &cold)
                    );
                    assert_eq!(argmax(&got), argmax(&cold), "{ratios:?} {what} at {p}");
                    // A prefix prefilled alone, the state's own twin, continued the same way.
                    let mut twin = model.new_decode_state();
                    model.prefill_into(&tokens[..p], &mut twin).unwrap();
                    let want = vecf_of(&model.prefill_continue(&tokens[p..], &mut twin).unwrap());
                    assert!(
                        rel(&got, &want) < 1e-4,
                        "{ratios:?} {what} at {p}: twin rel {}",
                        rel(&got, &want)
                    );
                    assert_eq!(argmax(&got), argmax(&want));
                    // And what follows decodes the same from the restored state as from its twin.
                    let a = vecf_of(&model.forward_decode(5, &mut restored).unwrap());
                    let b = vecf_of(&model.forward_decode(5, &mut twin).unwrap());
                    assert!(
                        rel(&a, &b) < 1e-3,
                        "{ratios:?} {what} at {p}: decode rel {}",
                        rel(&a, &b)
                    );
                }
            }
            // A point that is not on a block boundary is not a state this can write.
            let point = GridPoint {
                pos: 12,
                windows: &vec![Vec::new(); model.n_layers()],
                engram: model.engram_after(&tokens[..12]),
            };
            assert!(encode_grid_manifest(&cold_state, &point, &id, &layout).is_none());
        }
    }

    /// Two prompts that share a prefix ending inside a block: the second restores at the last grid
    /// point at or below the shared length and answers as a cold prefill would.
    #[test]
    fn a_prompt_sharing_a_prefix_mid_block_restores_at_the_grid_point_below() {
        use crate::inference::engine::model_backend::DeepseekV41Backend;
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
        let first: Vec<u32> = (0..40)
            .map(|i| (i * 7 + 3) % super::tests::VOCAB as u32)
            .collect();
        // The same for 27 tokens, then its own tail: shared prefix ends mid-block (blocks of 8).
        let mut second = first[..27].to_vec();
        second.extend((0..18).map(|i| (i * 5 + 11) % super::tests::VOCAB as u32));
        let dir = scratch_dir("midblock");
        let (mut a, store) = cached_backend(&dir);
        let _ = prompt_logits(&mut a, &first);
        wait_for_len(&store, 40);
        assert_eq!(store.lengths(), vec![8, 16, 24, 32, 40]);

        let (mut b, store_b) = cached_backend(&dir);
        let warm = prompt_logits(&mut b, &second);
        assert_eq!(b.last_restored(), 24);
        let (model, _) = synthetic(&[0, 2, 0], &[1], &[1], 0);
        let mut plain = DeepseekV41Backend::new(model, 0, 128);
        let cold = prompt_logits(&mut plain, &second);
        assert!(rel(&warm, &cold) < 1e-4, "rel {}", rel(&warm, &cold));
        assert_eq!(argmax(&warm), argmax(&cold));
        // It wrote its own grid on the way: 32 and 40 are the new tail's, 24 was its prefix.
        wait_for_len(&store_b, 45);
        assert!(store_b.lengths().contains(&40));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A state built by decoding, as a turn's end leaves it, round trips as well, and a restored
    /// state rewinds and checkpoints like any other: the replay after a rewind is bit for bit the
    /// first run, and the rewound state is the snapshot's.
    #[test]
    fn a_restored_state_checkpoints_and_rewinds_exactly() {
        use super::super::snapshot::{decode, encode, identity, same_state};
        let id = identity(&[b"synthetic"]);
        for (ratios, kv_src, index_src) in [
            (&[0u32, 2, 0][..], &[1u32][..], &[1u32][..]),
            (&[0u32, 2, 2, 0][..], &[1u32][..], &[1u32, 3][..]),
        ] {
            let (model, tokens) = synthetic(ratios, kv_src, index_src, 0);
            let head = tokens.len() - 7;
            let mut state = model.new_decode_state();
            for &t in &tokens[..head] {
                model.forward_decode(t, &mut state).unwrap();
            }
            let snap = encode(&state, None, &id);
            let mut restored = decode(&snap, &id, model.n_layers()).unwrap().state;
            assert!(same_state(&state, &restored));
            let mark = restored.checkpoint();
            let first: Vec<Vec<f32>> = tokens[head..]
                .iter()
                .map(|&t| vecf_of(&model.forward_decode(t, &mut restored).unwrap()))
                .collect();
            restored.rewind(&mark);
            assert!(same_state(&state, &restored));
            assert_eq!(encode(&restored, None, &id), snap);
            let again = vecf_of(
                &model
                    .prefill_continue(&tokens[head..], &mut restored)
                    .unwrap(),
            );
            // A verified block and a decode agree to rounding on the last position.
            let last = first.last().unwrap();
            let num: f32 = again.iter().zip(last).map(|(a, b)| (a - b) * (a - b)).sum();
            let den: f32 = last.iter().map(|b| b * b).sum();
            assert!((num / den).sqrt() < 1e-3);
            restored.rewind(&mark);
            let replay: Vec<Vec<f32>> = tokens[head..]
                .iter()
                .map(|&t| vecf_of(&model.forward_decode(t, &mut restored).unwrap()))
                .collect();
            assert_eq!(
                first, replay,
                "a rewound restored state replays bit for bit"
            );
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

    /// Reading the next layers' experts ahead only changes when bytes arrive: a streamed decode
    /// gives the same logits, bit for bit, whether or not it does, and the guesses are scored.
    #[test]
    fn expert_readahead_leaves_the_decode_unchanged() {
        let (mut model, tokens) = synthetic(&[0, 2, 0], &[1], &[1], 3);
        let decode = |model: &DeepseekV41Model| {
            let mut state = model.new_decode_state();
            tokens
                .iter()
                .map(|&t| vecf_of(&model.forward_decode(t, &mut state).unwrap()))
                .collect::<Vec<_>>()
        };
        model.pin_readahead(Some(0));
        let plain = decode(&model);
        let (routed_before, _, _, _) = crate::inference::offload::store::prediction_report();
        model.pin_readahead(Some(1));
        let ahead = decode(&model);
        let (routed_after, _, _, _) = crate::inference::offload::store::prediction_report();
        assert_eq!(plain, ahead, "read-ahead changed a decode's logits");
        assert!(
            routed_after > routed_before,
            "no routing was compared with a guess"
        );
    }

    /// A decode that lets some host experts finish a layer late stays close to the exact one:
    /// every position's logits keep their direction and most keep their argmax. A replay from a
    /// checkpoint reproduces the deferred decode bit for bit, and a share of zero is the exact
    /// decode.
    #[test]
    fn expert_deferral_stays_close_and_replays_exactly() {
        use crate::inference::offload::experts::ExpertOffload;
        let (mut model, tokens) = synthetic(&[0, 2, 0], &[1], &[1], 3);
        // Every expert is the host's: the cards decline all of them.
        model.offload_experts(Some(std::sync::Arc::new(ExpertOffload {
            host_expert_ns: Default::default(),
            lanes: 1,
            device: None,
            timings: std::sync::Arc::new(Default::default()),
            warm: None,
            run: Box::new(|_, _, _, _, _| None),
            run_batch: Some(Box::new(|active, _, _, _| {
                active.iter().map(|_| None).collect()
            })),
            on_card: Some(Box::new(|_| false)),
            run_multi: None,
        })));
        let decode = |model: &DeepseekV41Model, state: &mut DecodeState, toks: &[u32]| {
            toks.iter()
                .map(|&t| vecf_of(&model.forward_decode(t, state).unwrap()))
                .collect::<Vec<_>>()
        };
        let cosine = |a: &[f32], b: &[f32]| {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            dot / (na * nb)
        };
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap()
        };
        let mut exact_state = model.new_decode_state();
        let exact = decode(&model, &mut exact_state, &tokens);

        model.set_expert_deferral(0.0);
        let mut zero_state = model.new_decode_state();
        assert_eq!(decode(&model, &mut zero_state, &tokens), exact);

        let share = 0.5;
        model.set_expert_deferral(share);
        assert!(model.deferred_experts() > 0, "the share defers nothing");
        let mut state = model.new_decode_state();
        let split = tokens.len() / 2;
        let first = decode(&model, &mut state, &tokens[..split]);
        let mark = state.checkpoint();
        let rest = decode(&model, &mut state, &tokens[split..]);
        state.rewind(&mark);
        assert_eq!(
            decode(&model, &mut state, &tokens[split..]),
            rest,
            "a replay after a rewind differs"
        );

        let drifted: Vec<Vec<f32>> = first.into_iter().chain(rest).collect();
        let mut agree = 0;
        for (got, want) in drifted.iter().zip(&exact) {
            let c = cosine(got, want);
            // The bound a share of the experts moved one layer late may cost a position.
            assert!(c > 1.0 - 0.1 * share, "logits turned: cosine {c}");
            agree += (argmax(got) == argmax(want)) as usize;
        }
        assert!(
            agree * 4 >= drifted.len() * 3,
            "argmax kept at {agree} of {} positions",
            drifted.len()
        );
    }

    /// The quality budget end to end on a small model: a zero budget runs nothing and leaves the
    /// logits bit for bit, a budget measures every candidate (divergence no smaller than zero,
    /// reads saved counted) and a pick within it is applied.
    #[test]
    fn a_quality_budget_calibrates_and_zero_changes_nothing() {
        use super::super::calibrate;
        let (model, tokens) = synthetic(&[0, 2, 0], &[1], &[1], 3);
        let run = |m: &DeepseekV41Model, ids: &[u32]| -> Result<Vec<Vec<f32>>> {
            m.forward_prefill(ids)?.to_vec2::<f32>()
        };
        let plain = run(&model, &tokens).unwrap();
        let resident = model.plan_residency();
        let nothing = |_: &DeepseekV41Model, _: &[u32]| -> Result<Vec<Vec<f32>>> {
            panic!("a zero budget must not run the model")
        };
        assert!(
            calibrate::apply_budget(&model, 0.0, "k", &tokens, &resident, &nothing)
                .unwrap()
                .is_none()
        );
        assert_eq!(
            run(&model, &tokens).unwrap(),
            plain,
            "a zero budget changed the logits"
        );

        let table = calibrate::measure(&model, &tokens, &resident, &run).unwrap();
        assert_eq!(
            table.len(),
            calibrate::candidates(model.n_activated()).len()
        );
        assert!(table
            .iter()
            .all(|m| m.kl >= -1e-9 && m.saved_per_token >= 0.0));
        assert_eq!(
            run(&model, &tokens).unwrap(),
            plain,
            "measuring left a setting on"
        );
        let chosen = calibrate::pick(&table, 1e9);
        if let Some(m) = chosen {
            model.set_truncation(m.truncation());
            assert!(model.take_saved() == 0);
        }
    }

    /// Independent sequences decoded together give each the logits it gives alone: attention runs
    /// per sequence over its own ring at its own position, so no sequence reads another's keys, and
    /// the shared FFN over the batch differs only by accumulation. This is the invariant a
    /// throughput batcher rests on.
    #[test]
    fn forward_decode_batch_matches_separate_decodes() {
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
            // Three independent sequences: distinct token streams and distinct lengths, so the
            // batch step runs three different positions at once.
            let seqs: Vec<Vec<u32>> = (0..3)
                .map(|j| {
                    let len = tokens.len() - 2 * j - 1;
                    (0..len)
                        .map(|i| tokens[(i + 5 * j) % tokens.len()])
                        .collect()
                })
                .collect();
            // The target for each sequence: its last logits from a lone token-by-token decode.
            let want: Vec<Vec<f32>> = seqs
                .iter()
                .map(|s| {
                    let mut st = model.new_decode_state();
                    let mut last = Vec::new();
                    for &t in s {
                        last = vecf_of(&model.forward_decode(t, &mut st).unwrap());
                    }
                    last
                })
                .collect();
            let vocab = want[0].len();
            // The batched step: every sequence decoded up to its last token alone, then one
            // batched step over all their last tokens together.
            let mut states: Vec<DecodeState> = seqs
                .iter()
                .map(|s| {
                    let mut st = model.new_decode_state();
                    for &t in &s[..s.len() - 1] {
                        model.forward_decode(t, &mut st).unwrap();
                    }
                    st
                })
                .collect();
            let mut reqs: Vec<(u32, &mut DecodeState)> = seqs
                .iter()
                .zip(states.iter_mut())
                .map(|(s, st)| (*s.last().unwrap(), st))
                .collect();
            let logits = model.forward_decode_batch(&mut reqs).unwrap();
            let (nrows, v) = logits.dims2().unwrap();
            assert_eq!((nrows, v), (3, vocab));
            let flat = vecf_of(&logits);
            drop(reqs); // releases the mutable borrows into `states` so its positions can be read
            for (j, s) in seqs.iter().enumerate() {
                assert_eq!(states[j].pos, s.len(), "{ratios:?} seq {j} position");
                let got = &flat[j * vocab..(j + 1) * vocab];
                let r = rel(got, &want[j]);
                assert!(r < 1e-3, "{ratios:?} seq {j} rel {r}");
                assert_eq!(argmax(got), argmax(&want[j]), "{ratios:?} seq {j} argmax");
            }
        }
    }

    /// A batched greedy generation gives each prompt exactly the tokens it gets generated alone,
    /// and the batch shrinks as sequences hit their length: the shorter ones finish and drop out
    /// while the longer ones keep decoding, each still on its own path.
    #[test]
    fn generate_batch_matches_separate_generations() {
        let layouts: [(&[u32], &[u32], &[u32]); 2] =
            [(&[0, 2, 0], &[1], &[1]), (&[0, 2, 2, 0], &[1], &[1, 3])];
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap()
        };
        for (ratios, kv_src, index_src) in layouts {
            let (model, tokens) = synthetic(ratios, kv_src, index_src, 0);
            // Distinct streams, lengths and generation budgets, so the sequences finish on
            // different steps and the batch narrows under them.
            let prompts: Vec<Vec<u32>> = (0..3)
                .map(|j| {
                    let len = 8 + 2 * j;
                    (0..len)
                        .map(|i| tokens[(i + 5 * j) % tokens.len()])
                        .collect()
                })
                .collect();
            let max_new = vec![6usize, 3, 5];
            // The target: each prompt generated alone, greedily, one token at a time.
            let want: Vec<Vec<u32>> = prompts
                .iter()
                .zip(max_new.iter().copied())
                .map(|(p, m)| {
                    let mut st = model.new_decode_state();
                    let logits = model.prefill_into(p, &mut st).unwrap();
                    let (rows, vocab) = logits.dims2().unwrap();
                    let flat = vecf_of(&logits);
                    let mut next = argmax(&flat[(rows - 1) * vocab..rows * vocab]);
                    let mut out = Vec::new();
                    while out.len() < m {
                        out.push(next);
                        let lg = model.forward_decode(next, &mut st).unwrap();
                        next = argmax(&vecf_of(&lg));
                    }
                    out
                })
                .collect();
            let pr: Vec<&[u32]> = prompts.iter().map(|p| p.as_slice()).collect();
            let got = model.generate_batch(&pr, &max_new, None).unwrap();
            assert_eq!(
                got, want,
                "{ratios:?}: batched generation must match separate greedy decodes"
            );
        }
    }

    /// The batch worker streams each concurrent request exactly the tokens `generate_batch`
    /// produces: requests submitted together are prefilled and decoded through the same batched
    /// step, and each row's compute is independent of the batch, so the worker's scheduling does
    /// not change any sequence's output.
    #[test]
    fn batch_server_streams_the_batched_tokens() {
        use crate::inference::model::deepseek_v41::batch_server::V41BatchServer;
        use crate::inference::serve::continuous_serve::CbToken;
        use std::sync::Arc;
        let (model, tokens) = synthetic(&[0, 2, 0], &[1], &[1], 0);
        let prompts: Vec<Vec<u32>> = (0..3)
            .map(|j| {
                let len = 8 + 2 * j;
                (0..len)
                    .map(|i| tokens[(i + 5 * j) % tokens.len()])
                    .collect()
            })
            .collect();
        let max_new = vec![6usize, 3, 5];
        let pr: Vec<&[u32]> = prompts.iter().map(|p| p.as_slice()).collect();
        let want = model.generate_batch(&pr, &max_new, None).unwrap();

        let server = V41BatchServer::spawn(Arc::new(model), None, None, 4, None);
        let rxs: Vec<_> = prompts
            .iter()
            .zip(&max_new)
            .map(|(p, &m)| server.submit(p.clone(), m))
            .collect();
        let got: Vec<Vec<u32>> = rxs
            .into_iter()
            .map(|rx| {
                let mut seq = Vec::new();
                while let Ok(t) = rx.recv() {
                    match t {
                        CbToken::Tok(x) => seq.push(x),
                        CbToken::Done(_) => break,
                    }
                }
                seq
            })
            .collect();
        assert_eq!(got, want, "the worker must stream the batched tokens");
    }

    /// The batch worker prefills through the cache: a second prompt that extends a first one is
    /// restored from the state the first left (its prompt-end and grid states), continues as an
    /// ordinary sequence, and streams the same tokens a worker without a cache streams. The state a
    /// finished sequence leaves is written for the next one.
    #[test]
    fn the_batch_worker_restores_a_stored_prefix_and_streams_the_same_tokens() {
        use crate::inference::cache::state_disk::StateStore;
        use crate::inference::model::deepseek_v41::batch_server::V41BatchServer;
        use crate::inference::model::deepseek_v41::state_cache::StateCache;
        use crate::inference::serve::continuous_serve::CbToken;
        use std::sync::Arc;
        let run = |server: &V41BatchServer, prompt: &[u32]| -> Vec<u32> {
            let rx = server.submit(prompt.to_vec(), 4);
            let mut seq = Vec::new();
            while let Ok(t) = rx.recv() {
                match t {
                    CbToken::Tok(x) => seq.push(x),
                    CbToken::Done(_) => break,
                }
            }
            seq
        };
        let tokens: Vec<u32> = (0..40)
            .map(|i| (i * 7 + 3) % super::tests::VOCAB as u32)
            .collect();
        let dir = scratch_dir("batch");
        let store = Arc::new(StateStore::open(dir.clone(), 0, 8).unwrap());
        let id = super::super::snapshot::identity(&[b"synthetic"]);
        let (model, _) = synthetic(&[0, 2, 0], &[1], &[1], 0);
        let cache = StateCache::new(&model, store.clone(), id, 1).unwrap();
        let cached = V41BatchServer::spawn(Arc::new(model), None, None, 4, Some(cache));
        let first = run(&cached, &tokens[..24]);
        // The prompt's states and the finished sequence's: 8, 16, 24 and 24 + 4 tokens fed.
        wait_for_len(&store, 28);
        assert_eq!(store.lengths(), vec![8, 16, 24, 28]);
        let longer = run(&cached, &tokens);
        wait_for_len(&store, 40);

        let (model, _) = synthetic(&[0, 2, 0], &[1], &[1], 0);
        let plain = V41BatchServer::spawn(Arc::new(model), None, None, 4, None);
        assert_eq!(first, run(&plain, &tokens[..24]));
        assert_eq!(
            longer,
            run(&plain, &tokens),
            "the restored sequence streams the same tokens"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Decoding split at a layer boundary equals decoding the whole stack, bit for bit: the same
    /// per-layer block runs in the same order, only the loop is cut - the property a pipeline that
    /// places one run of layers per node rests on. The boundary carries the residual, its collapse
    /// weights, the band state and the engram recent across the cut.
    #[test]
    fn forward_decode_range_splits_the_stack() {
        let layouts: [(&[u32], &[u32], &[u32]); 2] =
            [(&[0, 2, 0], &[1], &[1]), (&[0, 2, 2, 0], &[1], &[1, 3])];
        for (ratios, kv_src, index_src) in layouts {
            let (model, tokens) = synthetic(ratios, kv_src, index_src, 0);
            let n_layers = ratios.len();
            for cut in [1usize, n_layers / 2, n_layers - 1] {
                let mut whole = model.new_decode_state();
                let mut split = model.new_decode_state();
                let run_split =
                    |model: &DeepseekV41Model, t: u32, st: &mut DecodeState| -> Vec<f32> {
                        let b = model
                            .forward_decode_range(DecodeRangeIn::Token(t), 0, cut, st)
                            .unwrap();
                        let bd = match b {
                            DecodeRangeOut::Boundary(bd) => bd,
                            _ => panic!("interior range must yield a boundary"),
                        };
                        match model
                            .forward_decode_range(DecodeRangeIn::Resume(bd), cut, n_layers, st)
                            .unwrap()
                        {
                            DecodeRangeOut::Logits(l) => vecf_of(&l),
                            _ => panic!("final range must yield logits"),
                        }
                    };
                // Decode a shared prefix identically into both states.
                for &t in &tokens[..tokens.len() - 1] {
                    model.forward_decode(t, &mut whole).unwrap();
                    run_split(&model, t, &mut split);
                }
                let last = *tokens.last().unwrap();
                let wl = vecf_of(&model.forward_decode(last, &mut whole).unwrap());
                let sl = run_split(&model, last, &mut split);
                assert_eq!(
                    wl, sl,
                    "{ratios:?} cut {cut}: split decode must equal whole decode bit for bit"
                );
            }
        }
    }

    /// A multi-segment pipeline decode equals the whole decode bit for bit, whatever the partition:
    /// the rehearsal a cluster runs with one segment per node.
    #[test]
    fn pipeline_decode_over_segments_equals_whole() {
        let layouts: [(&[u32], &[u32], &[u32]); 2] =
            [(&[0, 2, 0], &[1], &[1]), (&[0, 2, 2, 0], &[1], &[1, 3])];
        for (ratios, kv_src, index_src) in layouts {
            let (model, tokens) = synthetic(ratios, kv_src, index_src, 0);
            let n = ratios.len();
            let segs: Vec<(usize, usize)> = if n >= 3 {
                vec![(0, 1), (1, n - 1), (n - 1, n)]
            } else {
                vec![(0, 1), (1, n)]
            };
            let mut whole = model.new_decode_state();
            let mut pipe = model.new_decode_state();
            for &t in &tokens[..tokens.len() - 1] {
                model.forward_decode(t, &mut whole).unwrap();
                model.pipeline_decode(t, &segs, &mut pipe).unwrap();
            }
            let last = *tokens.last().unwrap();
            let wl = vecf_of(&model.forward_decode(last, &mut whole).unwrap());
            let pl = vecf_of(&model.pipeline_decode(last, &segs, &mut pipe).unwrap());
            assert_eq!(
                wl, pl,
                "{ratios:?} segments {segs:?}: pipeline decode must equal whole decode"
            );
        }
    }

    /// A decode boundary at a ratio-0 cut survives serialisation to bytes and back - the crossing a
    /// node makes - and resuming from the rebuilt boundary equals the whole decode bit for bit.
    #[test]
    fn decode_boundary_survives_a_byte_crossing() {
        use crate::inference::model::deepseek_v41::model::DecodeBoundary;
        // Layer 0 is ratio-0, so the cut-at-1 boundary carries no band state and serialises.
        let (model, tokens) = synthetic(&[0, 2, 0], &[1], &[1], 0);
        let (n, cut) = (3usize, 1usize);
        let cross = |model: &DeepseekV41Model, t: u32, st: &mut DecodeState| -> Vec<f32> {
            let b = model
                .forward_decode_range(DecodeRangeIn::Token(t), 0, cut, st)
                .unwrap();
            let bd = match b {
                DecodeRangeOut::Boundary(bd) => bd,
                _ => panic!("interior range must yield a boundary"),
            };
            let bytes = bd.to_bytes().expect("ratio-0 boundary serialises");
            let bd2 = DecodeBoundary::from_bytes(&bytes).expect("boundary rebuilds");
            match model
                .forward_decode_range(DecodeRangeIn::Resume(bd2), cut, n, st)
                .unwrap()
            {
                DecodeRangeOut::Logits(l) => vecf_of(&l),
                _ => panic!("final range must yield logits"),
            }
        };
        let mut whole = model.new_decode_state();
        let mut split = model.new_decode_state();
        for &t in &tokens[..tokens.len() - 1] {
            model.forward_decode(t, &mut whole).unwrap();
            cross(&model, t, &mut split);
        }
        let last = *tokens.last().unwrap();
        let wl = vecf_of(&model.forward_decode(last, &mut whole).unwrap());
        let sl = cross(&model, last, &mut split);
        assert_eq!(
            wl, sl,
            "a boundary crossed through bytes must resume bit for bit"
        );
    }

    #[test]
    fn a_mid_band_boundary_survives_a_byte_crossing() {
        use crate::inference::model::deepseek_v41::model::DecodeBoundary;
        // Cut at layer 2, inside the band: the boundary carries the compressor and indexer state,
        // not just the residual, so a cut anywhere - not only where the band state is empty -
        // crosses the wire and resumes exactly.
        let (model, tokens) = synthetic(&[0, 2, 2, 2, 0], &[1, 2, 3], &[1, 2, 3], 0);
        let (n, cut) = (5usize, 2usize);
        let cross = |model: &DeepseekV41Model, t: u32, st: &mut DecodeState| -> Vec<f32> {
            let bd = match model
                .forward_decode_range(DecodeRangeIn::Token(t), 0, cut, st)
                .unwrap()
            {
                DecodeRangeOut::Boundary(bd) => bd,
                _ => panic!("interior range must yield a boundary"),
            };
            assert!(
                !bd.shared_is_empty(),
                "a cut inside the band must carry non-empty band state"
            );
            let bytes = bd.to_bytes().expect("a mid-band boundary serialises");
            let bd2 = DecodeBoundary::from_bytes(&bytes).expect("boundary rebuilds");
            match model
                .forward_decode_range(DecodeRangeIn::Resume(bd2), cut, n, st)
                .unwrap()
            {
                DecodeRangeOut::Logits(l) => vecf_of(&l),
                _ => panic!("final range must yield logits"),
            }
        };
        let mut whole = model.new_decode_state();
        let mut split = model.new_decode_state();
        for &t in &tokens[..tokens.len() - 1] {
            model.forward_decode(t, &mut whole).unwrap();
            cross(&model, t, &mut split);
        }
        let last = *tokens.last().unwrap();
        let wl = vecf_of(&model.forward_decode(last, &mut whole).unwrap());
        let sl = cross(&model, last, &mut split);
        assert_eq!(
            wl, sl,
            "a mid-band boundary crossed through bytes must resume bit for bit"
        );
    }

    /// A decode boundary crosses a real wire frame end to end: serialised to bytes, framed with a
    /// header, read back off the stream, rebuilt, and resumed - the exact path between two pipeline
    /// nodes. The frame carries the opaque boundary blob as bytes (Q8, shape [len]).
    #[test]
    fn a_boundary_crosses_a_wire_frame() {
        use crate::distributed::wire::{read_frame, write_frame, FrameHeader, WireDType};
        use crate::inference::model::deepseek_v41::model::DecodeBoundary;
        let (model, tokens) = synthetic(&[0, 2, 0], &[1], &[1], 0);
        let (n, cut) = (3usize, 1usize);
        let cross = |model: &DeepseekV41Model, t: u32, st: &mut DecodeState| -> Vec<f32> {
            let bd = match model
                .forward_decode_range(DecodeRangeIn::Token(t), 0, cut, st)
                .unwrap()
            {
                DecodeRangeOut::Boundary(bd) => bd,
                _ => panic!("interior range must yield a boundary"),
            };
            let blob = bd.to_bytes().expect("ratio-0 boundary serialises");
            let header = FrameHeader {
                request_id: 7,
                layer_id: cut as u32,
                dtype: WireDType::Q8,
                shape: vec![blob.len() as u32],
            };
            let mut framed = Vec::new();
            write_frame(&mut framed, &header, &blob).expect("frame writes");
            let mut cursor = std::io::Cursor::new(&framed);
            let (got_header, payload) = read_frame(&mut cursor).expect("frame reads");
            assert_eq!(got_header, header, "the header must survive the frame");
            let bd2 = DecodeBoundary::from_bytes(&payload).expect("boundary rebuilds off the wire");
            match model
                .forward_decode_range(DecodeRangeIn::Resume(bd2), cut, n, st)
                .unwrap()
            {
                DecodeRangeOut::Logits(l) => vecf_of(&l),
                _ => panic!("final range must yield logits"),
            }
        };
        let mut whole = model.new_decode_state();
        let mut split = model.new_decode_state();
        for &t in &tokens[..tokens.len() - 1] {
            model.forward_decode(t, &mut whole).unwrap();
            cross(&model, t, &mut split);
        }
        let last = *tokens.last().unwrap();
        let wl = vecf_of(&model.forward_decode(last, &mut whole).unwrap());
        let sl = cross(&model, last, &mut split);
        assert_eq!(
            wl, sl,
            "a boundary crossed through a wire frame must resume bit for bit"
        );
    }

    /// A two-node pipeline over a real socket equals the whole decode: node A runs the head layer
    /// range, sends the boundary frame over a kept link, node B runs the tail range on its own
    /// state and sends the logits back. Request and reply are lockstep, so the shared toy weights
    /// are never touched from two threads at once; each node advances only its own layers' caches.
    #[test]
    fn a_two_node_pipeline_over_a_socket_equals_the_whole() {
        use crate::distributed::wire_link::WireLink;
        use std::net::{Ipv4Addr, TcpListener};
        use std::sync::Arc;

        let (model, tokens) = synthetic(&[0, 2, 0], &[1], &[1], 0);
        let cut = 1usize;
        let model = Arc::new(model);

        // Node B: run the tail range [cut, n) for each boundary that arrives, reply with the logits.
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let model_b = model.clone();
        let node_b = std::thread::spawn(move || {
            model_b.serve_pipeline_tail(cut, listener).unwrap();
        });

        // Node A: run the head range [0, cut), ship the boundary, read back the logits.
        let mut link = WireLink::connect(addr).unwrap();
        let mut state_a = model.new_decode_state();
        let step = |link: &mut WireLink, state_a: &mut DecodeState, t: u32| -> Vec<f32> {
            model.pipeline_head_step(t, cut, state_a, link).unwrap()
        };

        let mut whole = model.new_decode_state();
        for &t in &tokens[..tokens.len() - 1] {
            model.forward_decode(t, &mut whole).unwrap();
            step(&mut link, &mut state_a, t);
        }
        let last = *tokens.last().unwrap();
        let wl = vecf_of(&model.forward_decode(last, &mut whole).unwrap());
        let sl = step(&mut link, &mut state_a, last);
        assert_eq!(
            wl, sl,
            "a two-node pipeline over a socket must equal the whole decode"
        );
        drop(link);
        node_b.join().unwrap();
    }

    #[test]
    fn a_wire_head_step_over_a_mux_tail_equals_the_whole() {
        // The synchronous head step (no reader thread) tagged for a mux tail: one sequence should
        // decode exactly as the whole model, over the same tail serve loop the concurrent path uses.
        use crate::distributed::wire_link::WireLink;
        use std::net::{Ipv4Addr, TcpListener};
        use std::sync::Arc;

        let (model, tokens) = synthetic(&[0, 2, 0], &[1], &[1], 0);
        let cut = 1usize;
        let model = Arc::new(model);

        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let model_b = model.clone();
        let node_b = std::thread::spawn(move || {
            model_b.serve_pipeline_tail_mux(cut, listener).unwrap();
        });

        let mut link = WireLink::connect(addr).unwrap();
        let mut state_a = model.new_decode_state();
        let mut whole = model.new_decode_state();
        for &t in &tokens[..tokens.len() - 1] {
            model.forward_decode(t, &mut whole).unwrap();
            model
                .pipeline_head_step_wire(t, cut, 0, &mut state_a, &mut link)
                .unwrap();
        }
        let last = *tokens.last().unwrap();
        let wl = vecf_of(&model.forward_decode(last, &mut whole).unwrap());
        let sl = model
            .pipeline_head_step_wire(last, cut, 0, &mut state_a, &mut link)
            .unwrap();
        assert_eq!(
            wl, sl,
            "the wire head step over a mux tail must equal the whole decode"
        );
        drop(link);
        node_b.join().unwrap();
    }

    #[test]
    fn a_band_cut_over_several_tokens_on_separate_states_equals_the_whole() {
        // The head and tail keep their own states and the cut is inside the band, so a band layer
        // runs on the head every token. If the head does not advance its position per token, the
        // second token indexes the rope table past its one cached step - the bug the single-state
        // and ratio-0 pipeline tests never reach.
        use crate::distributed::wire_link::WireLink;
        use std::net::{Ipv4Addr, TcpListener};
        use std::sync::Arc;

        let (model, _t) = synthetic(&[0, 2, 2, 2, 0], &[1, 2, 3], &[1, 2, 3], 0);
        let cut = 2usize;
        let model = Arc::new(model);

        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let model_b = model.clone();
        let node_b = std::thread::spawn(move || {
            model_b.serve_pipeline_tail_mux(cut, listener).unwrap();
        });

        let mut link = WireLink::connect(addr).unwrap();
        let mut head = model.new_decode_state();
        let mut whole = model.new_decode_state();
        let seq = [1u32, 5, 3, 7, 2];
        for &t in &seq {
            let wl = vecf_of(&model.forward_decode(t, &mut whole).unwrap());
            let sl = model
                .pipeline_head_step_wire(t, cut, 0, &mut head, &mut link)
                .unwrap();
            assert_eq!(wl, sl, "a band-cut token must match the whole decode");
        }
        drop(link);
        node_b.join().unwrap();
    }

    #[test]
    fn concurrent_sequences_over_one_mux_link_each_equal_their_whole_decode() {
        use crate::distributed::wire_mux::MuxLink;
        use std::net::{Ipv4Addr, TcpListener, TcpStream};
        use std::sync::Arc;

        let (model, _tokens) = synthetic(&[0, 2, 0], &[1], &[1], 0);
        let cut = 1usize;
        let model = Arc::new(model);

        // Distinct sequences, decoded greedily, each starting from its own token.
        let starts: [u32; 3] = [1, 7, 3];
        let steps = 4usize;
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap()
        };

        // Oracle: each sequence decoded alone through the whole stack.
        let want: Vec<Vec<u32>> = starts
            .iter()
            .map(|&s| {
                let mut st = model.new_decode_state();
                let mut cur = s;
                (0..steps)
                    .map(|_| {
                        cur = argmax(&vecf_of(&model.forward_decode(cur, &mut st).unwrap()));
                        cur
                    })
                    .collect()
            })
            .collect();

        // Tail node: serve every sequence multiplexed on one connection.
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let model_b = model.clone();
        let node_b = std::thread::spawn(move || {
            model_b.serve_pipeline_tail_mux(cut, listener).unwrap();
        });

        // Head node: one shared link, one thread per sequence, all in flight at once.
        let link = MuxLink::new(TcpStream::connect(addr).unwrap(), starts.len()).unwrap();
        let got: Vec<Vec<u32>> = std::thread::scope(|scope| {
            let handles: Vec<_> = starts
                .iter()
                .enumerate()
                .map(|(i, &s)| {
                    let model = &model;
                    let link = &link;
                    scope.spawn(move || -> Vec<u32> {
                        let mut st = model.new_decode_state();
                        let mut cur = s;
                        (0..steps)
                            .map(|_| {
                                let l = model
                                    .pipeline_head_step_mux(cur, cut, i as u32, &mut st, link)
                                    .unwrap();
                                cur = argmax(&l);
                                cur
                            })
                            .collect()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        assert_eq!(
            got, want,
            "each sequence multiplexed on one link must equal its own whole decode"
        );

        // Drop the link so its socket closes and the tail's serve loop returns.
        drop(link);
        node_b.join().unwrap();
    }

    #[test]
    fn a_model_streamed_over_the_wire_decodes_like_the_local_one() {
        use crate::inference::offload::weight_cache::WeightCache;
        use crate::inference::offload::weight_stream::{serve_weights, GgufBytes, RemoteGguf};
        use crate::tensor::quantized::gguf_source::SplitGguf;
        use std::net::{Ipv4Addr, TcpListener};

        // A synthetic model written to a file that survives, so one node can serve it while another
        // streams it.
        let ratios = [0u32, 2, 0];
        let kv = [1u32];
        let index = [1u32];
        let bands: Vec<bool> = (0..ratios.len())
            .map(|l| ratios[l] != 0 || kv.contains(&(l as u32)) || index.contains(&(l as u32)))
            .collect();
        let md = super::tests::real_md_layout(&ratios, &kv, &index);
        let entries = super::tests::real_entries_for(&bands);
        let dir = std::env::temp_dir().join(format!(
            "loken-wire-model-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        write_gguf_with_metadata(&path, &md, &entries).unwrap();

        // The local model, loaded straight from the file.
        let g = open_mapped(&path).unwrap();
        let mut cfg = DeepseekV41Config::from_gguf(&g.content).unwrap();
        cfg.expert_cache_count = 0;
        let local = DeepseekV41Model::load(&g, &cfg).unwrap();

        // The serving node holds the same file; the running node streams it.
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let served = SplitGguf::open(&path).unwrap();
        let server = std::thread::spawn(move || {
            let _ = serve_weights(&GgufBytes(&served), listener);
        });

        let cache = WeightCache::ephemeral(&dir, 64 << 20).unwrap();
        let remote = RemoteGguf::connect(addr, cache).unwrap();
        let mut rcfg = DeepseekV41Config::from_meta(&remote).unwrap();
        rcfg.expert_cache_count = 0;
        let streamed = DeepseekV41Model::load(&remote, &rcfg).unwrap();

        // Decode a few tokens through each and compare the logits: same weights, same math.
        let tokens = [1u32, 5, 3, 0];
        let mut ls = local.new_decode_state();
        let mut rs = streamed.new_decode_state();
        for &t in &tokens[..tokens.len() - 1] {
            local.forward_decode(t, &mut ls).unwrap();
            streamed.forward_decode(t, &mut rs).unwrap();
        }
        let last = *tokens.last().unwrap();
        let ll = vecf_of(&local.forward_decode(last, &mut ls).unwrap());
        let rl = vecf_of(&streamed.forward_decode(last, &mut rs).unwrap());

        // Every tensor is streamed in its native bytes and mapped in place, the same path a local
        // file takes, so the model decodes bit for bit like the local one, not merely close.
        assert_eq!(
            ll, rl,
            "a model streamed over the wire must decode identically to the local one"
        );

        drop(streamed);
        drop(remote);
        let _ = server.join();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_partial_model_runs_its_range_like_the_whole() {
        // A node that loaded only [cut, n) runs that range on a boundary and gets the same logits
        // as the whole model does over the same range - the layers it did not load never mattered.
        let (whole, _tokens) = synthetic(&[0, 2, 0], &[1], &[1], 0);
        let cut = 1usize;
        let n = whole.n_layers();

        // Reload the same weights holding only [cut, n); the layers before cut are absent.
        let part = {
            // Rebuild the source the synthetic model came from, keeping only the tail range.
            let ratios = [0u32, 2, 0];
            let kv = [1u32];
            let index = [1u32];
            let bands: Vec<bool> = (0..ratios.len())
                .map(|l| ratios[l] != 0 || kv.contains(&(l as u32)) || index.contains(&(l as u32)))
                .collect();
            let md = super::tests::real_md_layout(&ratios, &kv, &index);
            let entries = super::tests::real_entries_for(&bands);
            let dir = std::env::temp_dir().join(format!(
                "loken-partial-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("model.gguf");
            write_gguf_with_metadata(&path, &md, &entries).unwrap();
            let g = open_mapped(&path).unwrap();
            let mut cfg = DeepseekV41Config::from_gguf(&g.content).unwrap();
            cfg.expert_cache_count = 0;
            let part = DeepseekV41Model::load_range(&g, &cfg, cut).unwrap();
            let _ = std::fs::remove_dir_all(&dir);
            part
        };
        assert!(
            part.blocks[0].is_none(),
            "a partial model does not hold the layers before its range"
        );

        // The boundary the head hands over, produced by the whole model running [0, cut). The run
        // is deterministic, so producing it once per consumer gives two identical boundaries -
        // simpler than cloning one.
        let head_boundary = || {
            let mut hs = whole.new_decode_state();
            match whole
                .forward_decode_range(DecodeRangeIn::Token(1), 0, cut, &mut hs)
                .unwrap()
            {
                DecodeRangeOut::Boundary(bd) => bd,
                DecodeRangeOut::Logits(_) => panic!("cut runs the whole stack"),
            }
        };

        // The whole model and the partial one each run [cut, n) from that boundary.
        let mut ws = whole.new_decode_state();
        let mut ps = part.new_decode_state();
        let wl = match whole
            .forward_decode_range(DecodeRangeIn::Resume(head_boundary()), cut, n, &mut ws)
            .unwrap()
        {
            DecodeRangeOut::Logits(l) => vecf_of(&l),
            DecodeRangeOut::Boundary(_) => panic!("tail did not reach the head"),
        };
        let pl = match part
            .forward_decode_range(DecodeRangeIn::Resume(head_boundary()), cut, n, &mut ps)
            .unwrap()
        {
            DecodeRangeOut::Logits(l) => vecf_of(&l),
            DecodeRangeOut::Boundary(_) => panic!("tail did not reach the head"),
        };
        assert_eq!(
            wl, pl,
            "a partial model must run its range exactly as the whole model does"
        );
    }

    /// A cut chosen by the planner runs as a pipeline and equals the whole decode: two nodes each
    /// too small for the stack force a split, the plan places one contiguous run of layers on each,
    /// and decoding through those segments gives what one node would. This is the planner driving
    /// the pipeline, end to end but in one process.
    #[test]
    fn a_planned_cut_runs_as_a_pipeline() {
        use crate::distributed::cut_plan::{plan, CutCosts, CutPlan, NodeCapacity};
        let (model, tokens) = synthetic(&[0, 2, 0], &[1], &[1], 0);
        let n = 3usize;
        // Equal layer weights; two nodes each holding at most two of the three layers, so the only
        // arrangement that fits uses both - a cut.
        let layer_bytes = vec![100u64; n];
        let nodes = vec![
            NodeCapacity {
                id: "a".into(),
                bytes: 200,
            },
            NodeCapacity {
                id: "b".into(),
                bytes: 200,
            },
        ];
        let costs = CutCosts::from_link_gbps(&[("a", "b", 10.0)], 8192);
        let cut: CutPlan = plan(&layer_bytes, &nodes, &costs).expect("a plan fits");
        assert!(cut.covers(n), "the plan must cover every layer once");
        assert!(
            cut.segments.len() >= 2,
            "capacity forces a split: {:?}",
            cut.segments
        );
        let segs: Vec<(usize, usize)> = cut.segments.iter().map(|s| (s.start, s.end)).collect();

        let mut whole = model.new_decode_state();
        let mut pipe = model.new_decode_state();
        for &t in &tokens[..tokens.len() - 1] {
            model.forward_decode(t, &mut whole).unwrap();
            model.pipeline_decode(t, &segs, &mut pipe).unwrap();
        }
        let last = *tokens.last().unwrap();
        let wl = vecf_of(&model.forward_decode(last, &mut whole).unwrap());
        let pl = vecf_of(&model.pipeline_decode(last, &segs, &mut pipe).unwrap());
        assert_eq!(
            wl, pl,
            "the planned cut must decode as the whole stack: {segs:?}"
        );
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
    #[cfg(feature = "gpu")]
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

    /// With the residual stream held on a card for the whole prompt - its mixes, collapse and
    /// update done there - a batched prefill, and a continuation of a state, give the logits and
    /// the decode state the CPU gives, on both layouts and at splits inside and past the window.
    #[cfg(feature = "gpu")]
    #[test]
    fn a_prefill_with_the_stream_on_a_card_matches_the_cpu() {
        use crate::inference::offload::cuda::{Card, Cards};
        use crate::inference::offload::{room, with_offload};
        let Ok(dev) = crate::tensor::cuda::CudaDevice::new(0) else {
            eprintln!("no CUDA device; the card stream is NOT covered by this run");
            return;
        };
        let (_, total) = crate::tensor::cuda_ext::mem_get_info(&Device::Cuda(dev.clone())).unwrap();
        let room = room::open(dev.ordinal(), total, 0);
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
        let cards: std::sync::Arc<dyn crate::inference::offload::Offload> =
            std::sync::Arc::new(Cards(vec![std::sync::Arc::new(Card::new(dev, room, 1))]));
        let layouts: [(&[u32], &[u32], &[u32]); 2] =
            [(&[0, 2, 0], &[1], &[1]), (&[0, 2, 2, 0], &[1], &[1, 3])];
        for (ratios, kv_src, index_src) in layouts {
            let (model, _) = synthetic(ratios, kv_src, index_src, 0);
            let tokens: Vec<u32> = (0..80)
                .map(|i| (i * 7 + 3) % super::tests::VOCAB as u32)
                .collect();
            for split in [5usize, 33, 64] {
                // The CPU: a prefix, then the rest onto it.
                let mut cpu = model.new_decode_state();
                model.prefill_into(&tokens[..split], &mut cpu).unwrap();
                let want = vecf_of(&model.prefill_continue(&tokens[split..], &mut cpu).unwrap());
                // The card holding the stream: the same two passes.
                let mut dev_state = model.new_decode_state();
                let got = with_offload(cards.clone(), || {
                    model
                        .prefill_into(&tokens[..split], &mut dev_state)
                        .unwrap();
                    vecf_of(
                        &model
                            .prefill_continue(&tokens[split..], &mut dev_state)
                            .unwrap(),
                    )
                });
                assert!(
                    rel(&got, &want) < 1e-4,
                    "{ratios:?} split {split}: rel {}",
                    rel(&got, &want)
                );
                assert_eq!(argmax(&got), argmax(&want), "{ratios:?} split {split}");
                for t in [3u32, 9] {
                    let a = vecf_of(&model.forward_decode(t, &mut cpu).unwrap());
                    let b = vecf_of(&model.forward_decode(t, &mut dev_state).unwrap());
                    assert!(
                        rel(&b, &a) < 1e-3,
                        "{ratios:?} split {split}: decode rel {}",
                        rel(&b, &a)
                    );
                }
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

    /// A backend over the synthetic model with a decode-state cache in `dir`, blocks of 8 tokens.
    fn cached_backend(
        dir: &std::path::Path,
    ) -> (
        crate::inference::engine::model_backend::DeepseekV41Backend,
        std::sync::Arc<crate::inference::cache::state_disk::StateStore>,
    ) {
        use crate::inference::cache::state_disk::StateStore;
        use crate::inference::engine::model_backend::DeepseekV41Backend;
        let (model, _) = synthetic(&[0, 2, 0], &[1], &[1], 0);
        let store = std::sync::Arc::new(StateStore::open(dir.to_path_buf(), 0, 8).unwrap());
        let id = super::super::snapshot::identity(&[b"synthetic"]);
        (
            DeepseekV41Backend::new(model, 0, 128).with_state_cache(store.clone(), id, 1),
            store,
        )
    }

    fn prompt_logits(
        backend: &mut crate::inference::engine::model_backend::DeepseekV41Backend,
        tokens: &[u32],
    ) -> Vec<f32> {
        use crate::inference::engine::model_backend::ModelBackend;
        let x = Tensor::new(tokens, &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        vecf_of(&backend.forward(&x, 0).unwrap())
    }

    fn wait_for_len(store: &crate::inference::cache::state_disk::StateStore, len: usize) {
        let start = std::time::Instant::now();
        while !store.lengths().contains(&len) {
            assert!(
                start.elapsed() < std::time::Duration::from_secs(20),
                "the state was not written"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        let d =
            std::env::temp_dir().join(format!("loken-dsv41-cache-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// A prompt that begins with one the cache holds is answered from the restored state and its
    /// remainder: the logits match a cold prefill to rounding with the same token, a prompt seen
    /// whole is answered with the logits it was stored with, and the turn a backend finishes is
    /// stored as the state after the tokens it fed.
    #[test]
    fn a_backend_restores_a_stored_prefix_and_matches_a_cold_prompt() {
        use crate::inference::engine::model_backend::{DeepseekV41Backend, ModelBackend};
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
        let tokens: Vec<u32> = (0..48)
            .map(|i| (i * 7 + 3) % super::tests::VOCAB as u32)
            .collect();
        let dir = scratch_dir("restore");
        let (mut first, store) = cached_backend(&dir);
        let cold_24 = prompt_logits(&mut first, &tokens[..24]);
        wait_for_len(&store, 24);
        // The grid of the prefill, blocks of 8: the states at 8 and 16 on the way to 24.
        assert_eq!(store.lengths(), vec![8, 16, 24]);

        // A longer prompt on a fresh backend over the same directory restores the 24 and prefills 24.
        let (mut second, store2) = cached_backend(&dir);
        let warm_48 = prompt_logits(&mut second, &tokens);
        let (model, _) = synthetic(&[0, 2, 0], &[1], &[1], 0);
        let mut plain = DeepseekV41Backend::new(model, 0, 128);
        let cold_48 = prompt_logits(&mut plain, &tokens);
        assert!(
            rel(&warm_48, &cold_48) < 1e-4,
            "rel {}",
            rel(&warm_48, &cold_48)
        );
        assert_eq!(argmax(&warm_48), argmax(&cold_48));
        wait_for_len(&store2, 48);
        assert_eq!(second.last_restored(), 24);

        // The prompt seen whole comes back with the logits it was stored with, to the bit.
        let (mut third, _) = cached_backend(&dir);
        assert_eq!(prompt_logits(&mut third, &tokens[..24]), cold_24);

        // A decode after a restored prompt agrees with the one after a cold prompt.
        let x = Tensor::new(&[5u32], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let warm_step = vecf_of(&second.forward(&x, tokens.len()).unwrap());
        let cold_step = vecf_of(&plain.forward(&x, tokens.len()).unwrap());
        assert!(rel(&warm_step, &cold_step) < 1e-3);

        // The turn's end: the state after the tokens fed is stored, whatever came after them.
        let mut turn = tokens.clone();
        turn.push(5);
        second.snapshot_kv(turn.clone(), turn.len()).unwrap();
        wait_for_len(&store2, 49);
        // A position the state does not stand at is not stored.
        let held = store2.lengths();
        second.snapshot_kv(turn.clone(), turn.len() - 1).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert_eq!(store2.lengths(), held);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A damaged snapshot - a manifest or a block, truncated, flipped, emptied, or written by
    /// another format version - is passed over with the prompt prefilled cold, never a panic and
    /// never a wrong answer, and the file is dropped. The files hold no run of the prompt's ids.
    #[test]
    fn a_damaged_snapshot_is_ignored_and_the_files_hold_no_tokens() {
        use crate::inference::engine::model_backend::DeepseekV41Backend;
        let tokens: Vec<u32> = (0..40)
            .map(|i| (i * 7 + 3) % super::tests::VOCAB as u32)
            .collect();
        let dir = scratch_dir("damaged");
        let (mut first, store) = cached_backend(&dir);
        let _ = prompt_logits(&mut first, &tokens[..24]);
        wait_for_len(&store, 24);

        let files = |d: &std::path::Path, ext: &str| -> Vec<std::path::PathBuf> {
            std::fs::read_dir(d)
                .unwrap()
                .map(|e| e.unwrap().path())
                .filter(|p| p.extension().is_some_and(|x| x == ext))
                .collect()
        };
        let mut manifests = files(&dir, "snap");
        manifests.sort();
        let blocks = files(&dir.join("blocks"), "blk");
        // The states at 8, 16 and 24 tokens, and the three blocks they share.
        assert_eq!(manifests.len(), 3);
        assert_eq!(blocks.len(), 3);
        let all_manifests: Vec<Vec<u8>> = manifests
            .iter()
            .map(|m| std::fs::read(m).unwrap())
            .collect();
        // The 24-token state is the longest and the first a 40-token prompt tries.
        let longest = manifests.len() - 1;
        let good_manifest = all_manifests[longest].clone();
        let good_blocks: Vec<Vec<u8>> = blocks.iter().map(|b| std::fs::read(b).unwrap()).collect();
        let good_block = good_blocks[0].clone();
        // No run of six consecutive prompt ids, as written, in any file of the directory.
        for run in tokens.windows(6) {
            let needle: Vec<u8> = run.iter().flat_map(|t| t.to_le_bytes()).collect();
            for bytes in [&good_manifest, &good_block] {
                assert!(!bytes.windows(needle.len()).any(|w| w == &needle[..]));
            }
        }

        let (model, _) = synthetic(&[0, 2, 0], &[1], &[1], 0);
        let mut plain = DeepseekV41Backend::new(model, 0, 128);
        let cold = prompt_logits(&mut plain, &tokens);

        let version = super::super::snapshot::FORMAT_VERSION + 1;
        // The manifest's snapshot payload starts after the store's header: magic, count, hashes.
        let payload = 12 + 16 * 3;
        let mut other_version = good_manifest.clone();
        other_version[payload + 8..payload + 12].copy_from_slice(&version.to_le_bytes());
        let mut flipped = good_manifest.clone();
        let mid = payload + (flipped.len() - payload) / 2;
        flipped[mid] ^= 0x01;
        let mut other_block = good_block.clone();
        other_block[8..12].copy_from_slice(&version.to_le_bytes());
        let mut flipped_block = good_block.clone();
        let mid = flipped_block.len() / 2;
        flipped_block[mid] ^= 0x01;
        let manifest_cases = [
            (
                "manifest truncated",
                good_manifest[..good_manifest.len() / 2].to_vec(),
            ),
            ("manifest empty", Vec::new()),
            ("manifest flipped", flipped),
            ("manifest other version", other_version),
        ];
        let block_cases = [
            (
                "block truncated",
                good_block[..good_block.len() / 2].to_vec(),
            ),
            ("block empty", Vec::new()),
            ("block flipped", flipped_block),
            ("block other version", other_block),
        ];
        let close = |got: &[f32], name: &str| {
            let num: f32 = got.iter().zip(&cold).map(|(a, b)| (a - b) * (a - b)).sum();
            let den: f32 = cold.iter().map(|b| b * b).sum();
            assert!((num / den).sqrt() < 1e-4, "{name}: answered wrongly");
        };
        for (name, bad) in manifest_cases.into_iter().chain(block_cases) {
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(dir.join("blocks")).unwrap();
            for (m, bytes) in manifests.iter().zip(&all_manifests) {
                std::fs::write(m, bytes).unwrap();
            }
            for (b, bytes) in blocks.iter().zip(&good_blocks) {
                std::fs::write(b, bytes).unwrap();
            }
            let target = if name.starts_with("manifest") {
                &manifests[longest]
            } else {
                &blocks[0]
            };
            std::fs::write(target, &bad).unwrap();
            // A fresh store reads the directory as a restarted server would; blocks other than
            // the damaged one are the real ones, so only the case under test can fail.
            let (mut backend, store) = cached_backend(&dir);
            let got = prompt_logits(&mut backend, &tokens);
            close(&got, name);
            // The prompt just prefilled is stored; the damaged file is not served again.
            wait_for_len(&store, 40);
            assert!(
                !std::fs::read(target).is_ok_and(|b| b == bad),
                "{name}: kept"
            );
        }
        let _ = std::fs::remove_dir_all(dir);
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

    /// A backend serving as a pipeline head over a loopback tail draws the same tokens, step for
    /// step, as the whole stack on one node. This gates the serve-side split end to end: the head
    /// runs `[0, cut)`, ships each boundary over a real socket to a tail running `[cut, n)`, and
    /// the tokens must be identical to the single-node path - the boundary crosses the wire whole.
    #[test]
    fn a_pipeline_head_backend_draws_the_same_tokens_as_one_node() {
        use crate::inference::engine::model_backend::{DeepseekV41Backend, ModelBackend};
        use std::net::TcpListener;
        let (whole_model, tokens) = synthetic(&[0, 2, 0], &[1], &[1], 0);
        let n = whole_model.n_layers();
        let cut = n / 2;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        // The tail loads the same weights and serves layers [cut, n) until the head hangs up. Its
        // model is built inside the thread, so it never crosses a thread boundary.
        let tail = std::thread::spawn(move || {
            let (tail_model, _) = synthetic(&[0, 2, 0], &[1], &[1], 0);
            let _ = tail_model.serve_pipeline_tail_mux(cut, listener);
        });
        let (head_model, _) = synthetic(&[0, 2, 0], &[1], &[1], 0);
        let mut head = DeepseekV41Backend::new(head_model, 0, tokens.len() + 8);
        head.connect_pipeline_tail(addr, cut).unwrap();
        let mut whole = DeepseekV41Backend::new(whole_model, 0, tokens.len() + 8);
        let argmax_last = |t: &Tensor| -> u32 {
            let vocab = *t.dims().last().unwrap();
            let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            v[v.len() - vocab..]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap()
        };
        // Feed the same tokens one at a time to both, at increasing positions: the head decodes
        // through the wire, the reference on one node, and every drawn token must agree.
        for (pos, &tok) in tokens.iter().take(8).enumerate() {
            let x = Tensor::new(&[tok], &Device::Cpu)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let ht = argmax_last(&head.forward(&x, pos).unwrap());
            let wt = argmax_last(&whole.forward(&x, pos).unwrap());
            assert_eq!(ht, wt, "pipeline and whole stack differ at position {pos}");
        }
        drop(head);
        let _ = tail.join();
    }

    /// A three-node pipeline draws the same tokens as one node: the head runs `[0, c1)`, an interior
    /// segment `[c1, c2)` relays its boundary to the final segment `[c2, n)`, and the logits return
    /// up the chain. This gates the N-node relay - a segment that is neither head nor tail.
    #[test]
    fn a_three_node_pipeline_draws_the_same_tokens_as_one_node() {
        use crate::inference::engine::model_backend::{DeepseekV41Backend, ModelBackend};
        use std::net::TcpListener;
        let (whole_model, tokens) = synthetic(&[0, 2, 0], &[1], &[1], 0);
        let n = whole_model.n_layers();
        let (c1, c2) = (1usize, 2usize); // segments [0,1) [1,2) [2,3)
        assert!(c1 < c2 && c2 < n);
        // The final segment [c2, n), then the interior [c1, c2) pointing at it. Both listeners are
        // bound here so the interior can connect to the final before either accepts.
        let l_final = TcpListener::bind("127.0.0.1:0").unwrap();
        let a_final = l_final.local_addr().unwrap();
        let l_mid = TcpListener::bind("127.0.0.1:0").unwrap();
        let a_mid = l_mid.local_addr().unwrap();
        let t_final = std::thread::spawn(move || {
            let (m, _) = synthetic(&[0, 2, 0], &[1], &[1], 0);
            let _ = m.serve_pipeline_segment(c2, n, None, l_final);
        });
        let t_mid = std::thread::spawn(move || {
            let (m, _) = synthetic(&[0, 2, 0], &[1], &[1], 0);
            let _ = m.serve_pipeline_segment(c1, c2, Some(a_final), l_mid);
        });
        let (head_model, _) = synthetic(&[0, 2, 0], &[1], &[1], 0);
        let mut head = DeepseekV41Backend::new(head_model, 0, tokens.len() + 8);
        head.connect_pipeline_tail(a_mid, c1).unwrap();
        let mut whole = DeepseekV41Backend::new(whole_model, 0, tokens.len() + 8);
        let argmax_last = |t: &Tensor| -> u32 {
            let vocab = *t.dims().last().unwrap();
            let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            v[v.len() - vocab..]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap()
        };
        for (pos, &tok) in tokens.iter().take(6).enumerate() {
            let x = Tensor::new(&[tok], &Device::Cpu)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let ht = argmax_last(&head.forward(&x, pos).unwrap());
            let wt = argmax_last(&whole.forward(&x, pos).unwrap());
            assert_eq!(ht, wt, "three-node pipeline differs at position {pos}");
        }
        drop(head);
        let _ = t_mid.join();
        let _ = t_final.join();
    }
}
