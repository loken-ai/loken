//! What mixes a layer's tokens: softmax attention over the KV cache, or one of the
//! recurrent operators the hybrid families interleave with it.
//!
//! A recurrent mixer carries a state beside the KV cache. The state advances with every
//! token and cannot be rewound to an earlier position; it resets when a forward starts at
//! position 0, and it is copied whole by a prefix snapshot. The mixer returns its output
//! before the layer's output projection, which the layer holds as `attn_output`.
//!
//! Which mixer a layer runs is read from the tensors its block carries.

#[allow(unused_imports)]
use super::*;
use crate::tensor::D;

/// The operator a block's tensors name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixerKind {
    Attention,
    /// LFM2 gated short convolution: `shortconv.{in_proj,conv,out_proj}`.
    ShortConv,
    /// Mamba2 selective state space: `ssm_{in,conv1d,a,d,dt,norm,out}`.
    Ssm,
    /// Gated delta-rule linear attention: `attn_qkv`, `attn_gate`, `ssm_{alpha,beta}` or
    /// `ssm_ba`, `ssm_{conv1d,a,dt,norm,out}`.
    DeltaNet,
    /// No mixing sublayer: the block is a feed-forward alone.
    Absent,
}

/// The mixer of block `il`, from the tensors the file carries for it.
pub fn layer_mixer_kind(content: &gguf_file::Content, il: usize) -> MixerKind {
    let has = |suffix: &str| {
        content
            .tensor_infos
            .contains_key(&format!("blk.{il}.{suffix}"))
    };
    if has("shortconv.conv.weight") {
        MixerKind::ShortConv
    } else if has("ssm_in.weight") {
        MixerKind::Ssm
    } else if has("ssm_beta.weight") || has("ssm_ba.weight") {
        MixerKind::DeltaNet
    } else if has("attn_q.weight") || has("attn_qkv.weight") {
        MixerKind::Attention
    } else {
        MixerKind::Absent
    }
}

/// Whether block `il` carries a feed-forward sublayer, dense or routed.
pub fn layer_has_ffn(content: &gguf_file::Content, il: usize) -> bool {
    [
        "ffn_up.weight",
        "ffn_gate_inp.weight",
        "ffn_up_exps.weight",
        "ffn_gate_up_exps.weight",
    ]
    .iter()
    .any(|t| content.tensor_infos.contains_key(&format!("blk.{il}.{t}")))
}

/// The first block whose mixer is attention, where the attention variant flags are read.
/// Block 0 when no block attends.
pub fn attention_probe_layer(content: &gguf_file::Content, n_layers: usize) -> usize {
    (0..n_layers)
        .find(|&il| layer_mixer_kind(content, il) == MixerKind::Attention)
        .unwrap_or(0)
}

/// Blocks whose mixer is attention, the ones that keep a KV cache.
pub fn attention_layer_count(content: &gguf_file::Content, n_layers: usize) -> usize {
    (0..n_layers)
        .filter(|&il| layer_mixer_kind(content, il) == MixerKind::Attention)
        .count()
}

/// A layer's token mixer.
pub enum Mixer {
    /// Softmax attention through the layer's attention fields and KV cache.
    Attention,
    ShortConv(ShortConv),
    Ssm(Mamba2),
    DeltaNet(GatedDeltaNet),
    /// The block has no mixing sublayer.
    Absent,
}

/// A recurrent mixer's state, copied whole.
#[derive(Clone)]
pub enum MixerState {
    ShortConv(Option<ConvState>),
    Ssm(Option<SsmState>),
    DeltaNet(Option<SsmState>),
    /// A wide-residual layer's: its mixer's state beside the caches of its indexer and of its
    /// n-gram features.
    Wide(Box<WideState>),
}

/// What a wide-residual layer carries beyond its mixer's own state.
#[derive(Clone)]
pub struct WideState {
    pub mixer: Option<MixerState>,
    pub index: Option<super::qsa::IndexerCache>,
    pub ngram: Option<super::ngram::PleState>,
}

impl MixerState {
    /// Device memory the copy holds.
    pub fn device_bytes(&self) -> u64 {
        match self {
            MixerState::ShortConv(Some(ConvState::Card(t))) => {
                (t.elem_count() * t.dtype().size_in_bytes()) as u64
            }
            MixerState::ShortConv(_) => 0,
            MixerState::Ssm(Some(SsmState::Card { conv, h }))
            | MixerState::DeltaNet(Some(SsmState::Card { conv, h })) => [conv, h]
                .iter()
                .map(|t| (t.elem_count() * t.dtype().size_in_bytes()) as u64)
                .sum(),
            MixerState::Ssm(_) | MixerState::DeltaNet(_) => 0,
            MixerState::Wide(w) => {
                w.mixer.as_ref().map_or(0, MixerState::device_bytes)
                    + w.ngram.as_ref().map_or(0, |n| n.device_bytes())
            }
        }
    }
}

/// `src` written into `dst`'s card buffers where they lie; an error when the two differ in
/// kind or shape, or live on the host.
/// The card address of `t`'s first element, `None` off a card.
#[cfg(feature = "cuda")]
fn card_address(t: &Tensor) -> Option<u64> {
    use crate::tensor::gpu_runtime::driver::DevicePtr;
    let (storage, layout) = t.storage_and_layout();
    match &*storage {
        crate::tensor::StorageView::Cuda(c) => {
            let bytes = t.dtype().size_in_bytes();
            c.as_cuda_slice::<f32>()
                .ok()
                .map(|s| s.device_ptr(s.stream()).0 + (layout.start_offset() * bytes) as u64)
        }
        _ => None,
    }
}

#[cfg(not(feature = "cuda"))]
fn card_address(_t: &Tensor) -> Option<u64> {
    None
}

/// Whether `a` and `b` are the same buffer: the same tensor, or the same card address.
fn same_buffer(a: &Tensor, b: &Tensor) -> bool {
    a.id() == b.id() || matches!((card_address(a), card_address(b)), (Some(x), Some(y)) if x == y)
}

/// `fresh` becomes the convolution's state, written into the held buffer when it matches: a
/// recorded step reads the state where it lies, so the buffer never changes under it.
fn hold_conv(slot: &mut Option<ConvState>, fresh: Tensor, moves: &mut u64) -> Result<()> {
    if let Some(ConvState::Card(held)) = slot.as_ref() {
        if same_buffer(held, &fresh) {
            return Ok(());
        }
        if held.dims() == fresh.dims()
            && held.dtype() == fresh.dtype()
            && held.device().same_device(&fresh.device())
        {
            return held.slice_set(&fresh, 0, 0);
        }
    }
    *slot = Some(ConvState::Card(fresh));
    *moves += 1;
    Ok(())
}

/// `conv` and `h` become the recurrence's state, written into the held buffers when they
/// match: a recorded step reads the state where it lies, so the buffers never change under it.
fn hold_ssm(slot: &mut Option<SsmState>, conv: Tensor, h: Tensor, moves: &mut u64) -> Result<()> {
    if let Some(SsmState::Card { conv: c0, h: h0 }) = slot.as_ref() {
        let same = |a: &Tensor, b: &Tensor| {
            a.dims() == b.dims() && a.dtype() == b.dtype() && a.device().same_device(&b.device())
        };
        if same(c0, &conv) && same(h0, &h) {
            if !same_buffer(c0, &conv) {
                c0.slice_set(&conv, 0, 0)?;
            }
            if !same_buffer(h0, &h) {
                h0.slice_set(&h, 0, 0)?;
            }
            return Ok(());
        }
    }
    *slot = Some(SsmState::Card { conv, h });
    *moves += 1;
    Ok(())
}

/// Writes zeros over `t` where it lies.
fn zero_in_place(t: &Tensor) -> Result<()> {
    t.slice_set(&t.zeros_like()?, 0, 0)
}

/// Starts the convolution's window again from zeros where it lies, so a recorded step keeps
/// reading the same buffer. A card window of another batch than `batch`, or one that cannot be
/// written, is dropped instead, and counted in `moves`.
fn restart_conv(slot: &mut Option<ConvState>, batch: Option<usize>, moves: &mut u64) {
    let kept = match slot.as_mut() {
        None => return,
        Some(ConvState::Host(v)) => {
            v.fill(0.0);
            true
        }
        Some(ConvState::Card(t)) => {
            batch.is_none_or(|b| t.dims().first() == Some(&b)) && zero_in_place(t).is_ok()
        }
    };
    if !kept {
        *slot = None;
        *moves += 1;
    }
}

/// Starts a recurrence's state again from zeros where it lies (see [`restart_conv`]).
fn restart_ssm(slot: &mut Option<SsmState>, batch: Option<usize>, moves: &mut u64) {
    let kept = match slot.as_mut() {
        None => return,
        Some(SsmState::Host { conv, h }) => {
            conv.fill(0.0);
            h.fill(0.0);
            true
        }
        Some(SsmState::Card { conv, h }) => {
            batch.is_none_or(|b| conv.dims().first() == Some(&b) && h.dims().first() == Some(&b))
                && zero_in_place(conv).is_ok()
                && zero_in_place(h).is_ok()
        }
    };
    if !kept {
        *slot = None;
        *moves += 1;
    }
}

fn copy_state_in_place(dst: &MixerState, src: &MixerState) -> Result<()> {
    let copy = |d: &Tensor, s: &Tensor| -> Result<()> {
        if d.dims() != s.dims() || d.dtype() != s.dtype() {
            return Err(crate::tensor::Error::msg("recurrent state buffers differ"));
        }
        d.slice_set(s, 0, 0)
    };
    match (dst, src) {
        (
            MixerState::ShortConv(Some(ConvState::Card(d))),
            MixerState::ShortConv(Some(ConvState::Card(s))),
        ) => copy(d, s),
        (
            MixerState::Ssm(Some(SsmState::Card { conv: dc, h: dh })),
            MixerState::Ssm(Some(SsmState::Card { conv: sc, h: sh })),
        )
        | (
            MixerState::DeltaNet(Some(SsmState::Card { conv: dc, h: dh })),
            MixerState::DeltaNet(Some(SsmState::Card { conv: sc, h: sh })),
        ) => {
            copy(dc, sc)?;
            copy(dh, sh)
        }
        _ => Err(crate::tensor::Error::msg("recurrent state not on a card")),
    }
}

impl Mixer {
    /// Makes, once, the buffers a verify window of up to `rows` rows keeps its rows' inputs
    /// in, for a mixer that returns to its kept rows by running them again from those inputs:
    /// `false` for a mixer that does not.
    pub fn window_rows_ready(&mut self, rows: usize, device: &Device) -> Result<bool> {
        match self {
            Mixer::DeltaNet(m) => m.window_rows_ready(rows, device),
            _ => Ok(false),
        }
    }

    /// Bytes of the buffers of [`Self::window_rows_ready`].
    pub fn window_rows_bytes(&self) -> u64 {
        match self {
            Mixer::DeltaNet(m) => m.window.as_ref().map_or(0, WindowRows::bytes),
            _ => 0,
        }
    }

    /// The row of the window the next forward's rows stand at, `None` outside a window.
    pub fn set_window_at(&mut self, at: Option<usize>) {
        if let Mixer::DeltaNet(m) = self {
            m.window_at = at;
        }
    }

    /// Advances the state over the window's first `kept` rows from the inputs it kept:
    /// `false` when the mixer keeps none.
    pub fn replay_window(&mut self, kept: usize) -> Result<bool> {
        match self {
            Mixer::DeltaNet(m) => m.replay_window(kept),
            _ => Ok(false),
        }
    }

    /// Forgets the buffers of [`Self::window_rows_ready`].
    pub fn drop_window_rows(&mut self) {
        if let Mixer::DeltaNet(m) = self {
            m.window = None;
            m.window_at = None;
        }
    }

    /// True for a mixer whose decode step advances its state where it lies, so a step run
    /// again reads and writes the same buffers, and that is worth recording: the delta rule
    /// through its fused recurrence, and the short convolution, on a card.
    pub fn decodes_in_place(&self, device: &Device) -> bool {
        match self {
            Mixer::DeltaNet(m) => m.steps_in_place(device),
            Mixer::Attention | Mixer::Absent => true,
            Mixer::ShortConv(_) => device.is_cuda(),
            // Its step advances in place too, but the recorded step measured slower than the
            // eager one, so it is not recorded.
            Mixer::Ssm(_) => false,
        }
    }

    /// One decode row through the mixer, its state advanced in place.
    #[cfg(feature = "cuda")]
    pub fn step_in_place(&mut self, x: &Tensor) -> Result<Tensor> {
        let _in_place = crate::inference::moe_cuda::in_place_state(true);
        self.forward(x, 1)
    }

    /// The state as it stands, copied, for `restore_in_place` to put back.
    pub fn checkpoint(&self) -> Result<Option<MixerState>> {
        self.snapshot()
    }

    /// Copies the state as it stands into the buffers of `keep`, a copy taken earlier, without
    /// moving either: a recorded step can take it, and [`Self::restore_in_place`] put it back.
    pub fn save_in_place(&self, keep: &MixerState) -> Result<()> {
        let current = match self {
            Mixer::Attention | Mixer::Absent => return Ok(()),
            Mixer::ShortConv(m) => m.state.clone().map(|s| MixerState::ShortConv(Some(s))),
            Mixer::Ssm(m) => m.state.clone().map(|s| MixerState::Ssm(Some(s))),
            Mixer::DeltaNet(m) => m.state.clone().map(|s| MixerState::DeltaNet(Some(s))),
        };
        match current {
            Some(cur) => copy_state_in_place(keep, &cur),
            None => Err(crate::tensor::Error::msg("no recurrent state to keep")),
        }
    }

    /// Writes `state` back into the buffers the state lives in, without moving them.
    pub fn restore_in_place(&mut self, state: &MixerState) -> Result<()> {
        let current = match self {
            Mixer::Attention | Mixer::Absent => return Ok(()),
            Mixer::ShortConv(m) => m.state.clone().map(|s| MixerState::ShortConv(Some(s))),
            Mixer::Ssm(m) => m.state.clone().map(|s| MixerState::Ssm(Some(s))),
            Mixer::DeltaNet(m) => m.state.clone().map(|s| MixerState::DeltaNet(Some(s))),
        };
        match current {
            Some(cur) => copy_state_in_place(&cur, state),
            None => self.restore(state),
        }
    }

    /// True for a mixer that carries a state beside the KV cache.
    pub fn is_recurrent(&self) -> bool {
        !matches!(self, Mixer::Attention | Mixer::Absent)
    }

    /// The mixed tokens of `x` `[b, seq, hidden]`, before the output projection. Starts a
    /// fresh state at position 0 and advances it over every row otherwise.
    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        match self {
            Mixer::Attention | Mixer::Absent => Err(crate::tensor::Error::msg(
                "no recurrent mixer on this layer",
            )),
            Mixer::ShortConv(m) => m.forward(x, index_pos),
            Mixer::Ssm(m) => m.forward(x, index_pos),
            Mixer::DeltaNet(m) => m.forward(x, index_pos),
        }
    }

    /// Starts the state again from zeros, written where it lies: a recorded step keeps
    /// reading the same buffers.
    pub fn reset(&mut self) {
        match self {
            Mixer::Attention | Mixer::Absent => {}
            Mixer::ShortConv(m) => restart_conv(&mut m.state, None, &mut m.moves),
            Mixer::Ssm(m) => restart_ssm(&mut m.state, None, &mut m.moves),
            Mixer::DeltaNet(m) => restart_ssm(&mut m.state, None, &mut m.moves),
        }
    }

    /// Times the state's card buffers were made anew or dropped.
    pub fn buffer_moves(&self) -> u64 {
        match self {
            Mixer::Attention | Mixer::Absent => 0,
            Mixer::ShortConv(m) => m.moves,
            Mixer::Ssm(m) => m.moves,
            Mixer::DeltaNet(m) => m.moves,
        }
    }

    /// A deep copy of the state, `None` for attention.
    pub fn snapshot(&self) -> Result<Option<MixerState>> {
        Ok(match self {
            Mixer::Attention | Mixer::Absent => None,
            Mixer::ShortConv(m) => Some(MixerState::ShortConv(
                m.state.as_ref().map(ConvState::deep_copy).transpose()?,
            )),
            Mixer::Ssm(m) => Some(MixerState::Ssm(
                m.state.as_ref().map(SsmState::deep_copy).transpose()?,
            )),
            Mixer::DeltaNet(m) => Some(MixerState::DeltaNet(
                m.state.as_ref().map(SsmState::deep_copy).transpose()?,
            )),
        })
    }

    /// Device bytes `snapshot` would copy: the recurrent state held on a card.
    pub fn state_device_bytes(&self) -> u64 {
        let bytes = |t: &Tensor| (t.elem_count() * t.dtype().size_in_bytes()) as u64;
        match self {
            Mixer::ShortConv(m) => match &m.state {
                Some(ConvState::Card(t)) => bytes(t),
                _ => 0,
            },
            Mixer::Ssm(Mamba2 { state, .. }) | Mixer::DeltaNet(GatedDeltaNet { state, .. }) => {
                match state {
                    Some(SsmState::Card { conv, h }) => bytes(conv) + bytes(h),
                    _ => 0,
                }
            }
            Mixer::Attention | Mixer::Absent => 0,
        }
    }

    /// Makes `state` the mixer's state; the copy stays reusable.
    pub fn restore(&mut self, state: &MixerState) -> Result<()> {
        match (self, state) {
            (Mixer::ShortConv(m), MixerState::ShortConv(s)) => {
                m.state = s.as_ref().map(ConvState::deep_copy).transpose()?;
                m.moves += 1;
                Ok(())
            }
            (Mixer::Ssm(m), MixerState::Ssm(s)) => {
                m.state = s.as_ref().map(SsmState::deep_copy).transpose()?;
                m.moves += 1;
                Ok(())
            }
            (Mixer::DeltaNet(m), MixerState::DeltaNet(s)) => {
                m.state = s.as_ref().map(SsmState::deep_copy).transpose()?;
                m.moves += 1;
                Ok(())
            }
            _ => Err(crate::tensor::Error::msg(
                "mixer state restored onto a mixer of another kind",
            )),
        }
    }
}

/// The rolling window of a convolution: on a card a tensor `[b, channels, window]` F32,
/// on the host the same values for one sequence, channel-major.
#[derive(Clone)]
pub enum ConvState {
    Card(Tensor),
    Host(Vec<f32>),
}

impl ConvState {
    fn deep_copy(&self) -> Result<Self> {
        Ok(match self {
            ConvState::Card(t) => ConvState::Card(t.affine(1.0, 0.0)?),
            ConvState::Host(v) => ConvState::Host(v.clone()),
        })
    }
}

/// LFM2 gated short convolution: `in_proj` splits into `B, C, x`, the product `B.x` runs
/// through a causal depthwise convolution of `l_cache` taps, and `C` gates its output.
/// The state holds the last `l_cache - 1` products of every channel.
pub struct ShortConv {
    pub in_proj: QMatMul,
    /// Convolution taps `[d_model, l_cache]` F32 on the layer's device.
    pub conv_w: Tensor,
    /// The same taps on the host, channel-major.
    pub conv_w_host: Vec<f32>,
    pub d_model: usize,
    pub l_cache: usize,
    pub state: Option<ConvState>,
    /// Times the state's card buffer was made anew: a step recorded against it is stale once
    /// this moves.
    pub(super) moves: u64,
}

impl ShortConv {
    /// The mixer over `conv_w` `[d_model, l_cache]`, its taps placed on `device`; the width
    /// and the tap count are the tensor's.
    pub fn new(in_proj: QMatMul, conv_w: Tensor, device: &Device) -> Result<Self> {
        let conv_w = conv_w.to_dtype(DType::F32)?.contiguous()?;
        let (d_model, l_cache) = conv_w.dims2()?;
        if l_cache < 2 {
            return Err(crate::tensor::Error::msg(format!(
                "short convolution of {l_cache} taps keeps no state"
            )));
        }
        let conv_w_host = conv_w.flatten_all()?.to_vec1::<f32>()?;
        Ok(Self {
            in_proj,
            conv_w: conv_w.to_device(device)?,
            conv_w_host,
            d_model,
            l_cache,
            state: None,
            moves: 0,
        })
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        if index_pos == 0 {
            restart_conv(&mut self.state, Some(b), &mut self.moves);
        }
        let proj = self.in_proj.forward(x)?;
        if b == 1 && !x.device().is_cuda() {
            let y = self.host_steps(&proj.to_dtype(DType::F32)?, seq)?;
            return Tensor::from_vec(y, (1, seq, self.d_model), &x.device())?.to_dtype(x.dtype());
        }
        let state = match self.state.clone() {
            Some(ConvState::Card(t)) => t,
            _ => Tensor::zeros_on((b, self.d_model, self.l_cache - 1), DType::F32, &x.device())?,
        };
        let mut state = state;
        let mut rows = Vec::with_capacity(seq);
        for t in 0..seq {
            let bcx = proj.i((.., t, ..))?.contiguous()?;
            let (y, next) = match crate::inference::moe_cuda::lfm2_shortconv_f16io(
                &bcx,
                &state,
                &self.conv_w,
                self.d_model,
                self.l_cache,
            )? {
                Some(out) => out,
                None => crate::inference::kernel::fused::fused_lfm2_shortconv(
                    &bcx.to_dtype(DType::F32)?,
                    &state,
                    &self.conv_w,
                    self.d_model,
                    self.l_cache,
                )?,
            };
            state = next;
            rows.push(y.to_dtype(x.dtype())?.unsqueeze(1)?);
        }
        hold_conv(&mut self.state, state, &mut self.moves)?;
        if rows.len() == 1 {
            return Ok(rows.pop().unwrap());
        }
        Tensor::cat(&rows.iter().collect::<Vec<_>>(), 1)
    }

    /// The convolution over `proj` `[1, seq, 3 * d_model]` F32 on the host, advancing the
    /// host state: `bx = B.x`, `acc = sum(window . taps)`, `y = C.acc`, then the window
    /// shifts in `bx`. Returns `y` `[seq * d_model]`.
    fn host_steps(&mut self, proj: &Tensor, seq: usize) -> Result<Vec<f32>> {
        let (d, l) = (self.d_model, self.l_cache);
        if !matches!(self.state, Some(ConvState::Host(_))) {
            self.state = Some(ConvState::Host(vec![0f32; d * (l - 1)]));
        }
        let Some(ConvState::Host(window)) = self.state.as_mut() else {
            unreachable!("host state set above")
        };
        let proj = proj.contiguous()?;
        let rows = proj.f32_data()?;
        Ok(short_conv_host(rows, window, &self.conv_w_host, d, l, seq))
    }
}

/// [`ShortConv::host_steps`] over plain slices: `rows` `[seq, 3d]` as `B | C | x`,
/// `window` `[d, l - 1]`, `taps` `[d, l]`.
pub fn short_conv_host(
    rows: &[f32],
    window: &mut [f32],
    taps: &[f32],
    d: usize,
    l: usize,
    seq: usize,
) -> Vec<f32> {
    let mut y = vec![0f32; seq * d];
    for t in 0..seq {
        let row = &rows[t * 3 * d..(t + 1) * 3 * d];
        let (bv, cv, xv) = (&row[..d], &row[d..2 * d], &row[2 * d..]);
        let yt = &mut y[t * d..(t + 1) * d];
        for c in 0..d {
            let bx = bv[c] * xv[c];
            let s = &mut window[c * (l - 1)..(c + 1) * (l - 1)];
            let w = &taps[c * l..(c + 1) * l];
            let mut acc = w[l - 1] * bx;
            for j in 0..l - 1 {
                acc += w[j] * s[j];
            }
            s.rotate_left(1);
            s[l - 2] = bx;
            yt[c] = cv[c] * acc;
        }
    }
    y
}

/// A state-space state: a convolution window per channel and the recurrent hidden. On a
/// card as tensors `[b, ..]` F32, on the host as flat values for one sequence. Mamba2
/// keeps the last `d_conv` inputs and a hidden `[nheads, headdim, d_state]`; the delta rule
/// the last `conv_kernel - 1` inputs and a matrix `[n_v, v_dim, k_dim]` per value head.
#[derive(Clone)]
pub enum SsmState {
    Card { conv: Tensor, h: Tensor },
    Host { conv: Vec<f32>, h: Vec<f32> },
}

impl SsmState {
    fn deep_copy(&self) -> Result<Self> {
        Ok(match self {
            SsmState::Card { conv, h } => SsmState::Card {
                conv: conv.affine(1.0, 0.0)?,
                h: h.affine(1.0, 0.0)?,
            },
            SsmState::Host { conv, h } => SsmState::Host {
                conv: conv.clone(),
                h: h.clone(),
            },
        })
    }
}

/// Mamba2 dimensions, from the file's `ssm.*` metadata and the `ssm_a` length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mamba2Dims {
    pub d_inner: usize,
    pub d_state: usize,
    pub ngroups: usize,
    pub d_conv: usize,
    pub nheads: usize,
}

impl Mamba2Dims {
    /// Channels the convolution runs over: `x`, then `B` and `C` for every group.
    pub fn d_xbc(&self) -> usize {
        self.d_inner + 2 * self.ngroups * self.d_state
    }
    pub fn headdim(&self) -> usize {
        self.d_inner / self.nheads.max(1)
    }
}

/// The weights the host recurrence reads, flat F32.
struct Mamba2Host {
    conv_w: Vec<f32>,
    conv_b: Vec<f32>,
    a: Vec<f32>,
    d: Vec<f32>,
    dt_bias: Vec<f32>,
    norm: Vec<f32>,
}

/// Mamba2 selective state space. `in_proj` gives `z | xBC | dt`; `xBC` runs through a
/// causal depthwise convolution with bias and SiLU; per head the state decays by
/// `exp(softplus(dt + dt_bias) . a)` and takes `dt . x (x) B`; the output reads the state
/// through `C`, adds `D . x`, is gated by `silu(z)` and RMS-normalised per group. `a` is
/// stored already negated (`-exp(A_log)`).
pub struct Mamba2 {
    pub in_proj: QMatMul,
    conv_w: Tensor,
    conv_b: Tensor,
    a: Tensor,
    d: Tensor,
    dt_bias: Tensor,
    norm_w: Tensor,
    host: Mamba2Host,
    pub dims: Mamba2Dims,
    eps: f64,
    pub state: Option<SsmState>,
    /// Times the state's card buffers were made anew: a step recorded against them is stale
    /// once this moves.
    pub(super) moves: u64,
}

/// Mamba2 weights as the file stores them, before placement.
pub struct Mamba2Weights {
    pub in_proj: QMatMul,
    /// `[d_xbc, d_conv]`.
    pub conv_w: Tensor,
    /// `[d_xbc]`, zeros when the file has none.
    pub conv_b: Tensor,
    pub a: Tensor,
    pub d: Tensor,
    pub dt_bias: Tensor,
    /// `[ngroups, d_inner / ngroups]`.
    pub norm_w: Tensor,
}

impl Mamba2 {
    /// The mixer over host `w`, its small tensors placed on `device`.
    pub fn new(w: Mamba2Weights, dims: Mamba2Dims, eps: f64, device: &Device) -> Result<Self> {
        let f32_flat = |t: &Tensor| -> Result<Tensor> { t.to_dtype(DType::F32)?.flatten_all() };
        let conv_w = w.conv_w.to_dtype(DType::F32)?.contiguous()?;
        let (conv_b, a, d, dt_bias) = (
            f32_flat(&w.conv_b)?,
            f32_flat(&w.a)?,
            f32_flat(&w.d)?,
            f32_flat(&w.dt_bias)?,
        );
        let norm_w = w.norm_w.to_dtype(DType::F32)?.contiguous()?;
        if conv_w.dims() != [dims.d_xbc(), dims.d_conv] || a.dims1()? != dims.nheads {
            return Err(crate::tensor::Error::msg(format!(
                "ssm weights {:?} / {} heads do not match {dims:?}",
                conv_w.dims(),
                a.dims1()?
            )));
        }
        let host = Mamba2Host {
            conv_w: conv_w.flatten_all()?.to_vec1()?,
            conv_b: conv_b.to_vec1()?,
            a: a.to_vec1()?,
            d: d.to_vec1()?,
            dt_bias: dt_bias.to_vec1()?,
            norm: norm_w.flatten_all()?.to_vec1()?,
        };
        Ok(Self {
            in_proj: w.in_proj,
            conv_w: conv_w.to_device(device)?,
            conv_b: conv_b.to_device(device)?,
            a: a.to_device(device)?,
            d: d.to_device(device)?,
            dt_bias: dt_bias.to_device(device)?,
            norm_w: norm_w.to_device(device)?,
            host,
            dims,
            eps,
            state: None,
            moves: 0,
        })
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        if index_pos == 0 {
            restart_ssm(&mut self.state, Some(b), &mut self.moves);
        }
        let dims = self.dims;
        let proj = self.in_proj.forward(x)?.to_dtype(DType::F32)?;
        if b == 1 && !x.device().is_cuda() {
            if !matches!(self.state, Some(SsmState::Host { .. })) {
                self.state = Some(SsmState::Host {
                    conv: vec![0f32; dims.d_xbc() * dims.d_conv],
                    h: vec![0f32; dims.nheads * dims.headdim() * dims.d_state],
                });
            }
            let Some(SsmState::Host { conv, h }) = self.state.as_mut() else {
                unreachable!("host state set above")
            };
            let proj = proj.contiguous()?;
            let y = mamba2_host(
                proj.f32_data()?,
                conv,
                h,
                &self.host,
                dims,
                self.eps as f32,
                seq,
            );
            return Tensor::from_vec(y, (1, seq, dims.d_inner), &x.device())?.to_dtype(x.dtype());
        }
        let (mut conv, mut h) = match self.state.clone() {
            Some(SsmState::Card { conv, h }) => (conv, h),
            _ => (
                Tensor::zeros_on((b, dims.d_xbc(), dims.d_conv), DType::F32, &x.device())?,
                Tensor::zeros_on(
                    (b, dims.nheads, dims.headdim(), dims.d_state),
                    DType::F32,
                    &x.device(),
                )?,
            ),
        };
        use crate::inference::kernel::fused;
        let per_group = dims.ngroups * dims.d_state;
        let mut rows = Vec::with_capacity(seq);
        for t in 0..seq {
            let p = proj.i((.., t, ..))?;
            let z = p.narrow(D::Minus1, 0, dims.d_inner)?;
            let xbc = p.narrow(D::Minus1, dims.d_inner, dims.d_xbc())?;
            let dt = p.narrow(D::Minus1, dims.d_inner + dims.d_xbc(), dims.nheads)?;
            let (xbc, next_conv) = fused::fused_causal_conv1d_silu(
                &xbc,
                &conv,
                &self.conv_w,
                &self.conv_b,
                dims.d_xbc(),
                dims.d_conv,
            )?;
            conv = next_conv;
            let x_c = xbc.narrow(D::Minus1, 0, dims.d_inner)?;
            let b_ = xbc.narrow(D::Minus1, dims.d_inner, per_group)?;
            let c_ = xbc.narrow(D::Minus1, dims.d_inner + per_group, per_group)?;
            let dt = match fused::softplus_bias(&dt, &self.dt_bias)? {
                Some(y) => y,
                None => ((dt.broadcast_add(&self.dt_bias)?.exp()? + 1.0)?).log()?,
            };
            let (y, next_h) = fused::fused_mamba2_ssm_step(
                &x_c,
                &b_,
                &c_,
                &dt,
                &self.a,
                &h,
                dims.nheads,
                dims.headdim(),
                dims.d_state,
                dims.ngroups,
            )?;
            h = next_h;
            let y = fused::fused_mamba2_gate_gnorm(
                &y.reshape((b, dims.d_inner))?,
                &x_c,
                &self.d,
                &z,
                &self.norm_w,
                b,
                dims.d_inner,
                dims.headdim(),
                dims.ngroups,
                self.eps,
            )?;
            rows.push(y.to_dtype(x.dtype())?.unsqueeze(1)?);
        }
        hold_ssm(&mut self.state, conv, h, &mut self.moves)?;
        if rows.len() == 1 {
            return Ok(rows.pop().unwrap());
        }
        Tensor::cat(&rows.iter().collect::<Vec<_>>(), 1)
    }
}

/// [`Mamba2::forward`] on the host over `proj` `[seq, d_inner + d_xbc + nheads]`, advancing
/// `conv` `[d_xbc, d_conv]` and `h` `[nheads, headdim, d_state]`. Returns `y`
/// `[seq * d_inner]`, before the output projection.
fn mamba2_host(
    proj: &[f32],
    conv: &mut [f32],
    h: &mut [f32],
    w: &Mamba2Host,
    dims: Mamba2Dims,
    eps: f32,
    seq: usize,
) -> Vec<f32> {
    use rayon::prelude::*;
    let (nh, hd, ds) = (dims.nheads, dims.headdim(), dims.d_state);
    let (di, dx, ng, l) = (dims.d_inner, dims.d_xbc(), dims.ngroups, dims.d_conv);
    let heads_per_group = nh / ng.max(1);
    let group = di / ng.max(1);
    let width = di + dx + nh;
    let silu = |v: f32| v / (1.0 + (-v).exp());
    let mut y_all = vec![0f32; seq * di];
    let mut conv_out = vec![0f32; dx];
    for t in 0..seq {
        let row = &proj[t * width..(t + 1) * width];
        let (z, xbc_in, dt_in) = (&row[..di], &row[di..di + dx], &row[di + dx..]);
        // The window holds the last `l` inputs, the current one included.
        for c in 0..dx {
            let s = &mut conv[c * l..(c + 1) * l];
            s.rotate_left(1);
            s[l - 1] = xbc_in[c];
            let taps = &w.conv_w[c * l..(c + 1) * l];
            let mut acc = w.conv_b[c];
            for j in 0..l {
                acc += taps[j] * s[j];
            }
            conv_out[c] = silu(acc);
        }
        let x_c = &conv_out[..di];
        let b_ = &conv_out[di..di + ng * ds];
        let c_ = &conv_out[di + ng * ds..di + 2 * ng * ds];
        let yt = &mut y_all[t * di..(t + 1) * di];
        h.par_chunks_mut(hd * ds)
            .zip(yt.par_chunks_mut(hd))
            .enumerate()
            .for_each(|(head, (hs, yh))| {
                let g = head / heads_per_group.max(1);
                let dt = (1.0 + (dt_in[head] + w.dt_bias[head]).exp()).ln();
                let decay = (dt * w.a[head]).exp();
                let bv = &b_[g * ds..(g + 1) * ds];
                let cv = &c_[g * ds..(g + 1) * ds];
                for p in 0..hd {
                    let xv = x_c[head * hd + p];
                    let dx_ = dt * xv;
                    let hrow = &mut hs[p * ds..(p + 1) * ds];
                    let mut acc = 0f32;
                    for s in 0..ds {
                        hrow[s] = hrow[s] * decay + dx_ * bv[s];
                        acc += hrow[s] * cv[s];
                    }
                    yh[p] = acc + w.d[head] * xv;
                }
            });
        // Gate by silu(z), then RMS-normalise each group and apply the weight.
        for i in 0..di {
            yt[i] *= silu(z[i]);
        }
        for g in 0..ng {
            let yg = &mut yt[g * group..(g + 1) * group];
            let var = yg.iter().map(|v| v * v).sum::<f32>() / group as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for (i, v) in yg.iter_mut().enumerate() {
                *v = *v * inv * w.norm[g * group + i];
            }
        }
    }
    y_all
}

/// The delta rule's input projections: `qkv | z | alpha | beta`, in one matrix when the
/// file's four share a format, apart otherwise, or with `beta, alpha` packed per key head
/// in one `ssm_ba`.
pub enum DeltaProj {
    Fused(QMatMul),
    Split {
        qkv: QMatMul,
        z: QMatMul,
        alpha: QMatMul,
        beta: QMatMul,
    },
    Ba {
        qkv: QMatMul,
        z: QMatMul,
        ba: QMatMul,
    },
}

/// Delta-rule dimensions, from the file's `ssm.*` metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaDims {
    pub k_heads: usize,
    pub v_heads: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    pub conv_kernel: usize,
    /// Value head `h` reads key head `h % k_heads`, else `h / (v_heads / k_heads)`.
    pub v_heads_tiled: bool,
    /// The per-head output norm is gated by `sigmoid(z)`, else by `silu(z)`.
    pub gate_sigmoid: bool,
}

impl DeltaDims {
    /// Channels of the convolution: `q`, `k` over the key heads, `v` over the value heads.
    pub fn conv_dim(&self) -> usize {
        2 * self.k_heads * self.k_dim + self.v_heads * self.v_dim
    }
    /// The key head value head `h` reads.
    pub fn key_head(&self, h: usize) -> usize {
        if self.v_heads_tiled {
            h % self.k_heads
        } else {
            h / (self.v_heads / self.k_heads).max(1)
        }
    }
}

/// The delta rule's host weights, flat F32.
struct DeltaHost {
    conv_w: Vec<f32>,
    a: Vec<f32>,
    dt_bias: Vec<f32>,
    norm: Vec<f32>,
}

/// Gated delta-rule linear attention. `qkv` runs through a causal depthwise convolution and
/// SiLU; `q` and `k` are L2-normalised per key head, `q` scaled by `1/sqrt(k_dim)`; per value
/// head the state decays by `exp(a . softplus(alpha + dt_bias))` (`a` stored negated), takes
/// the correction `sigmoid(beta) . (v - S k) (x) k`, and is read by `q`; the output is
/// RMS-normalised per head and gated by `silu(z)`.
pub struct GatedDeltaNet {
    pub proj: DeltaProj,
    conv_w: Tensor,
    a: Tensor,
    dt_bias: Tensor,
    norm_w: Tensor,
    host: DeltaHost,
    pub dims: DeltaDims,
    eps: f64,
    pub state: Option<SsmState>,
    /// Times the state's card buffers were made anew: a step recorded against them is stale
    /// once this moves.
    pub(super) moves: u64,
    /// The inputs a verify window's rows give the recurrence, kept for a return to its rows.
    window: Option<WindowRows>,
    /// The window's row the forward's rows stand at, set for a window's forward only.
    window_at: Option<usize>,
}

/// The inputs of the recurrence a verify window's rows give it, F32 in buffers made once:
/// the projected `qkv` `[rows, conv_dim]` and the gates' `alpha` and `beta` `[rows, v_heads]`,
/// with the dtypes the projection gave them, which a return gives them back exactly.
pub struct WindowRows {
    qkv: Tensor,
    alpha: Tensor,
    beta: Tensor,
    dtypes: [DType; 3],
}

impl WindowRows {
    fn bytes(&self) -> u64 {
        [&self.qkv, &self.alpha, &self.beta]
            .iter()
            .map(|t| (t.elem_count() * t.dtype().size_in_bytes()) as u64)
            .sum()
    }

    /// Writes the forward's `seq` rows at row `at`.
    fn store(&mut self, at: usize, seq: usize, rows: [&Tensor; 3]) -> Result<()> {
        for (i, (buf, src)) in [&self.qkv, &self.alpha, &self.beta]
            .into_iter()
            .zip(rows)
            .enumerate()
        {
            let width = buf.dim(1)?;
            if at + seq > buf.dim(0)? {
                crate::tensor::bail!("a window's rows reach past the rows kept for it");
            }
            buf.slice_set(&src.reshape((seq, width))?.to_dtype(DType::F32)?, 0, at)?;
            self.dtypes[i] = src.dtype();
        }
        Ok(())
    }

    /// The first `kept` rows, in the dtypes they came in.
    fn first(&self, kept: usize) -> Result<[Tensor; 3]> {
        let cut = |t: &Tensor, dt: DType| t.narrow(0, 0, kept)?.to_dtype(dt)?.contiguous();
        Ok([
            cut(&self.qkv, self.dtypes[0])?,
            cut(&self.alpha, self.dtypes[1])?,
            cut(&self.beta, self.dtypes[2])?,
        ])
    }
}

impl GatedDeltaNet {
    fn window_rows_ready(&mut self, rows: usize, device: &Device) -> Result<bool> {
        if !self.steps_in_place(device) {
            return Ok(false);
        }
        if self.window.is_none() {
            let d = self.dims;
            let zeros = |w: usize| Tensor::zeros_on((rows, w), DType::F32, device);
            self.window = Some(WindowRows {
                qkv: zeros(d.conv_dim())?,
                alpha: zeros(d.v_heads)?,
                beta: zeros(d.v_heads)?,
                dtypes: [DType::F32; 3],
            });
        }
        Ok(true)
    }

    /// Advances the held state over the window's first `kept` rows from the inputs the window
    /// kept, through the launches the window ran them through: the state and the convolution's
    /// window are those the rows leave decoded one at a time.
    #[cfg(feature = "cuda")]
    fn replay_window(&mut self, kept: usize) -> Result<bool> {
        let Some(window) = self.window.as_ref() else {
            return Ok(false);
        };
        if kept == 0 {
            return Ok(true);
        }
        let [qkv, alpha, beta] = window.first(kept)?;
        self.advance_card(&qkv, &alpha, &beta, kept, true)?;
        Ok(true)
    }

    #[cfg(not(feature = "cuda"))]
    fn replay_window(&mut self, _kept: usize) -> Result<bool> {
        Ok(false)
    }

    /// The mixer over host tensors `conv_w` `[conv_dim, conv_kernel]`, `a`, `dt_bias`
    /// `[v_heads]` and `norm_w` `[v_dim]`, placed on `device`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        proj: DeltaProj,
        conv_w: Tensor,
        a: Tensor,
        dt_bias: Tensor,
        norm_w: Tensor,
        dims: DeltaDims,
        eps: f64,
        device: &Device,
    ) -> Result<Self> {
        let flat = |t: Tensor| -> Result<Tensor> { t.to_dtype(DType::F32)?.flatten_all() };
        let conv_w = conv_w.to_dtype(DType::F32)?.contiguous()?;
        let (a, dt_bias, norm_w) = (flat(a)?, flat(dt_bias)?, flat(norm_w)?);
        if conv_w.dims() != [dims.conv_dim(), dims.conv_kernel]
            || a.dims1()? != dims.v_heads
            || norm_w.dims1()? != dims.v_dim
            || dims.conv_kernel < 2
            || !dims.v_heads.is_multiple_of(dims.k_heads.max(1))
        {
            return Err(crate::tensor::Error::msg(format!(
                "delta-rule weights {:?} do not match {dims:?}",
                conv_w.dims()
            )));
        }
        let host = DeltaHost {
            conv_w: conv_w.flatten_all()?.to_vec1()?,
            a: a.to_vec1()?,
            dt_bias: dt_bias.to_vec1()?,
            norm: norm_w.to_vec1()?,
        };
        Ok(Self {
            proj,
            conv_w: conv_w.to_device(device)?,
            a: a.to_device(device)?,
            dt_bias: dt_bias.to_device(device)?,
            norm_w: norm_w.to_device(device)?,
            host,
            dims,
            eps,
            state: None,
            moves: 0,
            window: None,
            window_at: None,
        })
    }

    /// Whether a decode row runs through the fused card recurrence, the path that advances
    /// its state in place.
    pub fn steps_in_place(&self, device: &Device) -> bool {
        let d = self.dims;
        device.is_cuda() && d.k_dim == d.v_dim && matches!(d.v_dim, 64 | 128 | 256)
    }

    /// `(qkv, z, alpha, beta)` of `x`, in the projection's dtype.
    pub(super) fn project(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let d = self.dims;
        let (zn, nv) = (d.v_heads * d.v_dim, d.v_heads);
        match &self.proj {
            DeltaProj::Fused(w) => {
                let full = w.forward(x)?;
                let cut = |at: usize, n: usize| full.narrow(D::Minus1, at, n)?.contiguous();
                Ok((
                    cut(0, d.conv_dim())?,
                    cut(d.conv_dim(), zn)?,
                    cut(d.conv_dim() + zn, nv)?,
                    cut(d.conv_dim() + zn + nv, nv)?,
                ))
            }
            DeltaProj::Split {
                qkv,
                z,
                alpha,
                beta,
            } => {
                let mut out = QMatMul::forward_shared(&[qkv, z, alpha, beta], x)?.into_iter();
                match (out.next(), out.next(), out.next(), out.next()) {
                    (Some(qkv), Some(z), Some(alpha), Some(beta)) => Ok((qkv, z, alpha, beta)),
                    _ => crate::tensor::bail!("a delta-net projection returned fewer outputs"),
                }
            }
            DeltaProj::Ba { qkv, z, ba } => {
                // One group per key head, its betas ahead of its alphas.
                let per = nv / d.k_heads.max(1);
                let mut shared = QMatMul::forward_shared(&[ba, qkv, z], x)?.into_iter();
                let (Some(ba), Some(qkv_out), Some(z_out)) =
                    (shared.next(), shared.next(), shared.next())
                else {
                    crate::tensor::bail!("a delta-net projection returned fewer outputs")
                };
                let mut lead = ba.dims().to_vec();
                lead.pop();
                let mut grouped = lead.clone();
                grouped.extend([d.k_heads, 2 * per]);
                let ba = ba.reshape(grouped)?;
                let mut flat = lead;
                flat.push(nv);
                let beta = ba
                    .narrow(D::Minus1, 0, per)?
                    .contiguous()?
                    .reshape(flat.clone())?;
                let alpha = ba
                    .narrow(D::Minus1, per, per)?
                    .contiguous()?
                    .reshape(flat)?;
                Ok((qkv_out, z_out, alpha, beta))
            }
        }
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        if index_pos == 0 {
            restart_ssm(&mut self.state, Some(b), &mut self.moves);
        }
        let d = self.dims;
        #[cfg(feature = "cuda")]
        if b == 1 && x.device().is_cuda() && d.k_dim == d.v_dim && matches!(d.v_dim, 64 | 128 | 256)
        {
            return self.forward_card(x, seq);
        }
        if b == 1 && !x.device().is_cuda() {
            let (qkv, z, alpha, beta) = self.project(x)?;
            let rows = |t: Tensor| -> Result<Tensor> { t.to_dtype(DType::F32)?.contiguous() };
            let (qkv, z, alpha, beta) = (rows(qkv)?, rows(z)?, rows(alpha)?, rows(beta)?);
            if !matches!(self.state, Some(SsmState::Host { .. })) {
                self.state = Some(SsmState::Host {
                    conv: vec![0f32; d.conv_dim() * (d.conv_kernel - 1)],
                    h: vec![0f32; d.v_heads * d.v_dim * d.k_dim],
                });
            }
            let Some(SsmState::Host { conv, h }) = self.state.as_mut() else {
                unreachable!("host state set above")
            };
            let y = delta_host(
                DeltaRows {
                    qkv: qkv.f32_data()?,
                    z: z.f32_data()?,
                    alpha: alpha.f32_data()?,
                    beta: beta.f32_data()?,
                },
                conv,
                h,
                &self.host,
                d,
                self.eps as f32,
                seq,
            );
            return Tensor::from_vec(y, (1, seq, d.v_heads * d.v_dim), &x.device())?
                .to_dtype(x.dtype());
        }
        self.forward_rows(x, b, seq)
    }

    /// One token at a time through tensor ops: the path for shapes the fused recurrence
    /// does not cover.
    pub(super) fn forward_rows(&mut self, x: &Tensor, b: usize, seq: usize) -> Result<Tensor> {
        let d = self.dims;
        let (kg, vg, kd, vd) = (d.k_heads, d.v_heads, d.k_dim, d.v_dim);
        let rep = vg / kg;
        let qk = kd * kg;
        let (mut conv, mut s) = match self.state.clone() {
            Some(SsmState::Card { conv, h }) => (conv, h),
            _ => (
                Tensor::zeros_on(
                    (b, d.conv_dim(), d.conv_kernel - 1),
                    DType::F32,
                    &x.device(),
                )?,
                Tensor::zeros_on((b, vg, vd, kd), DType::F32, &x.device())?,
            ),
        };
        let l2 = |t: &Tensor| -> Result<Tensor> {
            let n = (t.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
            t.broadcast_div(&n)
        };
        let mut outs = Vec::with_capacity(seq);
        for t in 0..seq {
            let (qkv, z, alpha, beta) = self.project(&x.i((.., t, ..))?)?;
            let qkv = qkv.to_dtype(DType::F32)?;
            let win = Tensor::cat(&[&conv, &qkv.unsqueeze(qkv.rank())?], D::Minus1)?;
            let mut acc = self
                .conv_w
                .i((.., 0))?
                .broadcast_mul(&win.i((.., .., 0))?)?;
            for k in 1..d.conv_kernel {
                acc = (acc
                    + self
                        .conv_w
                        .i((.., k))?
                        .broadcast_mul(&win.i((.., .., k))?)?)?;
            }
            conv = win.narrow(D::Minus1, 1, d.conv_kernel - 1)?.contiguous()?;
            let c = crate::tensor::ops::silu(&acc)?;
            let q = c.narrow(D::Minus1, 0, qk)?.reshape((b, kg, kd))?;
            let k = c.narrow(D::Minus1, qk, qk)?.reshape((b, kg, kd))?;
            let v = c.narrow(D::Minus1, 2 * qk, vd * vg)?.reshape((b, vg, vd))?;
            let q = (l2(&q)? * (1.0 / (kd as f64).sqrt()))?;
            let k = l2(&k)?;
            let spread = |t: Tensor| -> Result<Tensor> {
                if d.v_heads_tiled {
                    t.unsqueeze(1)?
                        .broadcast_as((b, rep, kg, kd))?
                        .reshape((b, vg, kd))
                } else {
                    t.unsqueeze(2)?
                        .broadcast_as((b, kg, rep, kd))?
                        .reshape((b, vg, kd))
                }
            };
            let (q, k) = (spread(q)?, spread(k)?);
            let alpha = alpha.to_dtype(DType::F32)?;
            let sp = match crate::inference::kernel::fused::softplus_bias(&alpha, &self.dt_bias)? {
                Some(y) => y,
                None => ((alpha.broadcast_add(&self.dt_bias)?.exp()? + 1.0)?).log()?,
            };
            let decay = self.a.broadcast_mul(&sp)?.exp()?;
            let beta = crate::tensor::ops::sigmoid(&beta.to_dtype(DType::F32)?)?;
            let st = s.broadcast_mul(&decay.reshape((b, vg, 1, 1))?)?;
            let kv = st
                .broadcast_mul(&k.reshape((b, vg, 1, kd))?)?
                .sum(D::Minus1)?;
            let delta = (&v - kv)?.broadcast_mul(&beta.reshape((b, vg, 1))?)?;
            let st = (st
                + delta
                    .reshape((b, vg, vd, 1))?
                    .broadcast_mul(&k.reshape((b, vg, 1, kd))?)?)?;
            let o = st
                .broadcast_mul(&q.reshape((b, vg, 1, kd))?)?
                .sum(D::Minus1)?;
            s = st;
            let on = RmsNorm::new(self.norm_w.clone(), self.eps as f32).forward(&o)?;
            let z = z.to_dtype(DType::F32)?.reshape((b, vg, vd))?;
            let gate = if d.gate_sigmoid {
                crate::tensor::ops::sigmoid(&z)?
            } else {
                crate::tensor::ops::silu(&z)?
            };
            let y = (on * gate)?.reshape((b, vg * vd))?;
            outs.push(y.to_dtype(x.dtype())?.unsqueeze(1)?);
        }
        hold_ssm(&mut self.state, conv, s, &mut self.moves)?;
        Tensor::cat(&outs.iter().collect::<Vec<_>>(), 1)
    }

    /// The card path for one sequence: the projections and the convolution over every row
    /// at once, then one fused recurrence launch and the fused gated norm.
    #[cfg(feature = "cuda")]
    pub(super) fn forward_card(&mut self, x: &Tensor, seq: usize) -> Result<Tensor> {
        use super::wide::staged;
        use crate::inference::moe_cuda as k;
        let d = self.dims;
        let (vg, vd) = (d.v_heads, d.v_dim);
        let dev = x.device();
        let (qkv, z, alpha, beta) = staged("gdn_project", &dev, false, || self.project(x))?;
        let sequential = self.window_at.is_some();
        if let (Some(at), Some(window)) = (self.window_at, self.window.as_mut()) {
            window.store(at, seq, [&qkv, &alpha, &beta])?;
        }
        let o = self.advance_card(&qkv, &alpha, &beta, seq, sequential)?;
        let o = o.reshape((seq * vg, vd))?;
        let z = z.reshape((seq * vg, vd))?;
        let y = match k::zgate_rmsnorm_f16io(&o, &z, &self.norm_w, self.eps as f32, d.gate_sigmoid)?
        {
            Some(y) => y,
            None => k::zgate_rmsnorm(
                &o,
                &z.to_dtype(DType::F32)?,
                &self.norm_w,
                self.eps as f32,
                d.gate_sigmoid,
            )?,
        };
        y.reshape((1, seq, vg * vd))?.to_dtype(x.dtype())
    }

    /// The recurrence over `seq` projected rows, `qkv` `[seq, conv_dim]`, `alpha` and `beta`
    /// `[seq, v_heads]`: the convolution and gates, then the scan, the state held advanced.
    /// `sequential` scans the rows one after the other, as a window's rows are, else a long
    /// sequence takes the chunked form. Returns the scan's output `[seq, v_heads, v_dim]`.
    #[cfg(feature = "cuda")]
    fn advance_card(
        &mut self,
        qkv: &Tensor,
        alpha: &Tensor,
        beta: &Tensor,
        seq: usize,
        sequential: bool,
    ) -> Result<Tensor> {
        use super::wide::staged;
        use crate::inference::moe_cuda as k;
        let d = self.dims;
        let (kg, vg, kd, vd) = (d.k_heads, d.v_heads, d.k_dim, d.v_dim);
        let (rep, qk) = (vg / kg, kd * kg);
        let dev = qkv.device();
        let (conv, s) = match self.state.clone() {
            Some(SsmState::Card { conv, h }) => (conv, h),
            _ => (
                Tensor::zeros_on((1, d.conv_dim(), d.conv_kernel - 1), DType::F32, &dev)?,
                Tensor::zeros_on((1, vg, vd, kd), DType::F32, &dev)?,
            ),
        };
        let qkv = qkv.reshape((seq, d.conv_dim()))?.contiguous()?;
        let conv = conv
            .reshape((d.conv_dim(), d.conv_kernel - 1))?
            .contiguous()?;
        let prep = std::time::Instant::now();
        let prepping = crate::inference::offload::stage_prof_enabled();
        if prepping {
            let _ = dev.synchronize();
        }
        // A decoded token, and the rows of a draft verify, which are prepared one after the
        // other as the decode steps they stand for.
        let fused = if seq == 1
            || (crate::tensor::quantized::DecodeOrder::active()
                && crate::inference::moe_cuda::in_place_state_on())
        {
            k::gdn_prep(
                &qkv,
                &conv,
                &self.conv_w,
                alpha,
                beta,
                &self.a,
                &self.dt_bias,
                k::GdnShape {
                    k_heads: kg,
                    k_dim: kd,
                    v_heads: vg,
                    v_dim: vd,
                    taps: d.conv_kernel,
                    tiled: d.v_heads_tiled,
                    eps: 1e-6,
                },
            )?
        } else {
            None
        };
        let (q, key, v, g, beta, conv) = match fused {
            Some(p) => (
                p.q,
                p.k,
                p.v,
                p.g,
                p.beta,
                p.window.reshape((1, d.conv_dim(), d.conv_kernel - 1))?,
            ),
            None => {
                let (c, conv) =
                    match k::fused_conv_silu_f16in(&qkv, &conv, &self.conv_w, d.conv_kernel)? {
                        Some(out) => out,
                        None => k::fused_conv_silu(
                            &qkv.to_dtype(DType::F32)?,
                            &conv,
                            &self.conv_w,
                            d.conv_kernel,
                        )?,
                    };
                let conv = conv.reshape((1, d.conv_dim(), d.conv_kernel - 1))?;
                let q = c
                    .narrow(D::Minus1, 0, qk)?
                    .reshape((seq, kg, kd))?
                    .contiguous()?;
                let key = c
                    .narrow(D::Minus1, qk, qk)?
                    .reshape((seq, kg, kd))?
                    .contiguous()?;
                let v = c
                    .narrow(D::Minus1, 2 * qk, vd * vg)?
                    .reshape((seq, vg, vd))?
                    .contiguous()?;
                let q = k::l2norm_gqa(&q, kg, kd, rep, d.v_heads_tiled, 1e-6)?;
                let key = k::l2norm_gqa(&key, kg, kd, rep, d.v_heads_tiled, 1e-6)?;
                let (alpha, beta) = (alpha.reshape((seq, vg))?, beta.reshape((seq, vg))?);
                let (g, beta) = match k::deltanet_gate_f16in(&alpha, &beta, &self.a, &self.dt_bias)?
                {
                    Some(out) => out,
                    None => k::deltanet_gate(
                        &alpha.to_dtype(DType::F32)?,
                        &beta.to_dtype(DType::F32)?,
                        &self.a,
                        &self.dt_bias,
                    )?,
                };
                (q, key, v, g, beta, conv)
            }
        };
        let s = s.reshape((vg, vd, kd))?.contiguous()?;
        let scale = (1.0 / (kd as f64).sqrt()) as f32;
        if prepping {
            let _ = dev.synchronize();
            crate::inference::offload::prof_record(
                "gdn_conv_gate",
                prep.elapsed().as_nanos() as u64,
            );
        }
        let (o, s) = staged("gdn_scan", &dev, false, || match sequential {
            true => k::delta_net_launch(&q, &key, &v, &g, &beta, &s, scale, false),
            false => k::gated_delta_net(&q, &key, &v, &g, &beta, &s, scale),
        })?;
        hold_ssm(
            &mut self.state,
            conv,
            s.reshape((1, vg, vd, kd))?,
            &mut self.moves,
        )?;
        Ok(o)
    }
}

/// The projected rows of a sequence, host F32: `qkv` `[seq, conv_dim]`, `z`
/// `[seq, v_heads * v_dim]`, `alpha` and `beta` `[seq, v_heads]`.
struct DeltaRows<'a> {
    qkv: &'a [f32],
    z: &'a [f32],
    alpha: &'a [f32],
    beta: &'a [f32],
}

/// [`GatedDeltaNet::forward`] on the host, advancing `conv` `[conv_dim, conv_kernel - 1]`
/// and `h` `[v_heads, v_dim, k_dim]`. Returns `y` `[seq * v_heads * v_dim]`.
fn delta_host(
    rows: DeltaRows<'_>,
    conv: &mut [f32],
    h: &mut [f32],
    w: &DeltaHost,
    d: DeltaDims,
    eps: f32,
    seq: usize,
) -> Vec<f32> {
    use rayon::prelude::*;
    let (kg, vg, kd, vd) = (d.k_heads, d.v_heads, d.k_dim, d.v_dim);
    let (ck, cd) = (d.conv_kernel, d.conv_dim());
    let qk = kd * kg;
    let scale = 1.0f32 / (kd as f32).sqrt();
    let silu = |v: f32| v / (1.0 + (-v).exp());
    let mut y = vec![0f32; seq * vg * vd];
    let mut conv_out = vec![0f32; cd];
    let mut qn = vec![0f32; qk];
    let mut kn = vec![0f32; qk];
    for t in 0..seq {
        let xt = &rows.qkv[t * cd..(t + 1) * cd];
        for c in 0..cd {
            let taps = &w.conv_w[c * ck..(c + 1) * ck];
            let s = &conv[c * (ck - 1)..(c + 1) * (ck - 1)];
            let mut acc = taps[ck - 1] * xt[c];
            for j in 0..ck - 1 {
                acc += taps[j] * s[j];
            }
            conv_out[c] = silu(acc);
        }
        for c in 0..cd {
            let s = &mut conv[c * (ck - 1)..(c + 1) * (ck - 1)];
            s.rotate_left(1);
            s[ck - 2] = xt[c];
        }
        for g in 0..kg {
            let q = &conv_out[g * kd..(g + 1) * kd];
            let k = &conv_out[qk + g * kd..qk + (g + 1) * kd];
            let qs = (q.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
            let ks = (k.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
            for i in 0..kd {
                qn[g * kd + i] = q[i] / qs * scale;
                kn[g * kd + i] = k[i] / ks;
            }
        }
        let v_all = &conv_out[2 * qk..2 * qk + vg * vd];
        let zr = &rows.z[t * vg * vd..(t + 1) * vg * vd];
        let (ar, br) = (
            &rows.alpha[t * vg..(t + 1) * vg],
            &rows.beta[t * vg..(t + 1) * vg],
        );
        let yt = &mut y[t * vg * vd..(t + 1) * vg * vd];
        let (qn, kn) = (&qn, &kn);
        h.par_chunks_mut(vd * kd)
            .zip(yt.par_chunks_mut(vd))
            .enumerate()
            .for_each(|(head, (s, yh))| {
                let j = d.key_head(head);
                let (q, k) = (&qn[j * kd..(j + 1) * kd], &kn[j * kd..(j + 1) * kd]);
                let v = &v_all[head * vd..(head + 1) * vd];
                let sp = (1.0 + (ar[head] + w.dt_bias[head]).exp()).ln();
                let decay = (w.a[head] * sp).exp();
                let beta = 1.0 / (1.0 + (-br[head]).exp());
                let mut sumsq = 0f32;
                let mut o = vec![0f32; vd];
                for r in 0..vd {
                    let row = &mut s[r * kd..(r + 1) * kd];
                    let mut kv = 0f32;
                    for c in 0..kd {
                        row[c] *= decay;
                        kv += row[c] * k[c];
                    }
                    let delta = (v[r] - kv) * beta;
                    let mut or = 0f32;
                    for c in 0..kd {
                        row[c] += delta * k[c];
                        or += row[c] * q[c];
                    }
                    o[r] = or;
                    sumsq += or * or;
                }
                let inv = 1.0 / (sumsq / vd as f32 + eps).sqrt();
                for r in 0..vd {
                    let z = zr[head * vd + r];
                    let gate = if d.gate_sigmoid {
                        1.0 / (1.0 + (-z).exp())
                    } else {
                        silu(z)
                    };
                    yh[r] = o[r] * inv * w.norm[r] * gate;
                }
            });
    }
    y
}

impl GenericTransformerLayer {
    /// Whether this layer mixes by attention, the layers that keep a KV cache.
    pub(super) fn attends(&self) -> bool {
        matches!(self.mixer, Mixer::Attention)
    }

    /// The hybrid layer as the graph's decode step: attention through its graph path, a
    /// recurrent mixer stepping its state in place.
    #[cfg(feature = "cuda")]
    pub(super) fn forward_hybrid_graph(&mut self, x: &Tensor) -> Result<Tensor> {
        let x = match self.mixer {
            Mixer::Absent => x.clone(),
            Mixer::Attention => {
                let x_norm = self.attn_norm.forward(x)?;
                let mut out = self.forward_attn_graph(&x_norm)?;
                if let Some(b) = &self.attn_output_bias {
                    out = out.broadcast_add(b)?;
                }
                (x + out.to_dtype(x.dtype())?)?
            }
            _ => {
                let x_norm = self.attn_norm.forward(x)?;
                let mixed = self.mixer.step_in_place(&x_norm)?;
                let mixed = self.attn_output.forward(&mixed)?;
                (x + mixed.to_dtype(x.dtype())?)?
            }
        };
        match self.has_ffn {
            true => self.ffn_sublayer(&x),
            false => Ok(x),
        }
    }

    /// True for a layer the hybrid forward runs: a recurrent mixer, no mixer, or no
    /// feed-forward.
    pub(super) fn is_hybrid(&self) -> bool {
        !matches!(self.mixer, Mixer::Attention) || !self.has_ffn
    }

    /// A hybrid family's layer: `x + out(mix(norm(x)))` where it mixes, then
    /// `x + ffn(ffn_norm(x))` where it has a feed-forward.
    pub(super) fn forward_hybrid(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let x = match self.mixer {
            Mixer::Absent => x.clone(),
            Mixer::Attention => {
                let x_norm = self.attn_norm.forward(x)?;
                let mut out = self.forward_attn(
                    &x_norm,
                    mask,
                    index_pos,
                    None,
                    None,
                    #[cfg(feature = "cuda")]
                    None,
                )?;
                if let Some(b) = &self.attn_output_bias {
                    out = out.broadcast_add(b)?;
                }
                (x + out.to_dtype(x.dtype())?)?
            }
            _ => {
                let x_norm = self.attn_norm.forward(x)?;
                // A decoded token advances the state where it lies, the way the recorded
                // step does, so the two can follow each other on the same buffers.
                #[cfg(feature = "cuda")]
                let _in_place = crate::inference::moe_cuda::in_place_state(
                    index_pos > 0 && x.dim(1).is_ok_and(|n| n == 1),
                );
                let mixed = self.mixer.forward(&x_norm, index_pos)?;
                let mixed = self.attn_output.forward(&mixed)?;
                (x + mixed.to_dtype(x.dtype())?)?
            }
        };
        match self.has_ffn {
            true => self.ffn_sublayer(&x),
            false => Ok(x),
        }
    }

    /// `x + ffn(ffn_norm(x))`, the routed experts standing for the feed-forward on a layer
    /// that has them.
    pub(super) fn ffn_sublayer(&mut self, x: &Tensor) -> Result<Tensor> {
        let norm = self
            .ffn_norm
            .as_ref()
            .ok_or_else(|| crate::tensor::Error::msg("ffn_norm missing on a serial layer"))?;
        let x_norm = norm.forward(x)?;
        let out = match self.moe.as_ref() {
            Some(moe) => {
                moe.forward_rows(&x_norm, x, self.embedding_length_for_moe, self.ffn_norm_eps)?
            }
            None => self.forward_ffn(&x_norm)?,
        };
        x + out.to_dtype(x.dtype())?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_restart_zeroes_the_card_state_where_it_lies() {
        let dev = Device::Cpu;
        let held = Tensor::from_vec(values(12, 1, 1.0), (1, 4, 3), &dev).unwrap();
        let mut slot = Some(ConvState::Card(held.clone()));
        let mut moves = 0;
        restart_conv(&mut slot, Some(1), &mut moves);
        let Some(ConvState::Card(t)) = slot.as_ref() else {
            panic!("the state was dropped")
        };
        assert!(t.shares_storage(&held));
        assert!(t
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|&v| v == 0.0));
        assert_eq!(moves, 0);

        let conv = Tensor::from_vec(values(6, 2, 1.0), (1, 2, 3), &dev).unwrap();
        let h = Tensor::from_vec(values(8, 3, 1.0), (1, 2, 2, 2), &dev).unwrap();
        let mut slot = Some(SsmState::Card {
            conv: conv.clone(),
            h: h.clone(),
        });
        restart_ssm(&mut slot, None, &mut moves);
        let Some(SsmState::Card { conv: c, h: hh }) = slot.as_ref() else {
            panic!("the state was dropped")
        };
        assert!(c.shares_storage(&conv) && hh.shares_storage(&h));
        assert!(hh
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|&v| v == 0.0));
        assert_eq!(moves, 0);
    }

    #[test]
    fn a_restart_for_another_batch_drops_the_state_and_counts_the_move() {
        let dev = Device::Cpu;
        let held = Tensor::zeros_on((1, 4, 3), DType::F32, &dev).unwrap();
        let mut slot = Some(ConvState::Card(held));
        let mut moves = 0;
        restart_conv(&mut slot, Some(2), &mut moves);
        assert!(slot.is_none());
        assert_eq!(moves, 1);
        let fresh = Tensor::zeros_on((2, 4, 3), DType::F32, &dev).unwrap();
        hold_conv(&mut slot, fresh.clone(), &mut moves).unwrap();
        assert_eq!(moves, 2);
        hold_conv(&mut slot, fresh.affine(1.0, 1.0).unwrap(), &mut moves).unwrap();
        assert_eq!(moves, 2);
    }

    fn values(n: usize, seed: u64, amp: f32) -> Vec<f32> {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 2.0 * amp
            })
            .collect()
    }

    /// The reference convolution: the whole history of `B.x` products, zero-padded on the
    /// left, dotted with the taps at every step.
    fn short_conv_reference(
        rows: &[f32],
        taps: &[f32],
        d: usize,
        l: usize,
        seq: usize,
    ) -> Vec<f32> {
        let mut y = vec![0f32; seq * d];
        for t in 0..seq {
            for c in 0..d {
                let bx = |u: usize| rows[u * 3 * d + c] * rows[u * 3 * d + 2 * d + c];
                let mut acc = 0f32;
                for j in 0..l {
                    // Tap j reads the product l - 1 - j steps back.
                    let back = l - 1 - j;
                    if t >= back {
                        acc += taps[c * l + j] * bx(t - back);
                    }
                }
                y[t * d + c] = rows[t * 3 * d + d + c] * acc;
            }
        }
        y
    }

    #[test]
    fn the_host_short_conv_matches_the_convolution_over_the_history() {
        let (d, l, seq) = (6usize, 3usize, 7usize);
        let rows = values(seq * 3 * d, 1, 1.0);
        let taps = values(d * l, 2, 0.5);
        let reference = short_conv_reference(&rows, &taps, d, l, seq);
        let mut window = vec![0f32; d * (l - 1)];
        let whole = short_conv_host(&rows, &mut window, &taps, d, l, seq);
        let diff = whole
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(diff < 1e-6, "max diff {diff}");
        // Split in two calls, the state carries the window across.
        let mut window = vec![0f32; d * (l - 1)];
        let mut split = short_conv_host(&rows[..3 * 3 * d], &mut window, &taps, d, l, 3);
        split.extend(short_conv_host(
            &rows[3 * 3 * d..],
            &mut window,
            &taps,
            d,
            l,
            seq - 3,
        ));
        assert_eq!(split, whole);
    }

    #[test]
    fn the_short_conv_mixer_resets_at_position_zero_and_restores_a_snapshot() {
        let (hidden, l, seq) = (8usize, 3usize, 5usize);
        let dev = Device::Cpu;
        let w = Tensor::from_vec(
            values(3 * hidden * hidden, 3, 0.3),
            (3 * hidden, hidden),
            &dev,
        )
        .unwrap();
        let proj = QMatMul::from_qtensor(
            crate::tensor::quantized::QTensor::quantize_onto(
                &w,
                crate::tensor::quantized::GgmlDType::F32,
                &dev,
            )
            .unwrap(),
        )
        .unwrap();
        let taps = Tensor::from_vec(values(hidden * l, 4, 0.5), (hidden, l), &dev).unwrap();
        let mut mixer = Mixer::ShortConv(ShortConv::new(proj, taps, &dev).unwrap());
        let x = Tensor::from_vec(values(seq * hidden, 5, 1.0), (1, seq, hidden), &dev).unwrap();
        let whole = mixer.forward(&x, 0).unwrap();
        // Prefix, snapshot, continue; then restore and continue again.
        let head = x.narrow(1, 0, 2).unwrap();
        let tail = x.narrow(1, 2, seq - 2).unwrap();
        mixer.forward(&head, 0).unwrap();
        let snap = mixer.snapshot().unwrap().unwrap();
        let tail_a = mixer.forward(&tail, 2).unwrap();
        mixer.forward(&x, 0).unwrap();
        mixer.restore(&snap).unwrap();
        let tail_b = mixer.forward(&tail, 2).unwrap();
        let v = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v(&tail_a), v(&tail_b));
        assert_eq!(v(&whole.narrow(1, 2, seq - 2).unwrap()), v(&tail_a));
    }

    /// The Mamba2 block written out over the whole history: the convolution reads the
    /// inputs directly, the state is carried per head and channel.
    #[allow(clippy::too_many_arguments)]
    fn mamba2_reference(
        proj: &[f32],
        conv_w: &[f32],
        conv_b: &[f32],
        a: &[f32],
        dvec: &[f32],
        dt_bias: &[f32],
        norm: &[f32],
        dims: Mamba2Dims,
        eps: f32,
        seq: usize,
    ) -> Vec<f32> {
        let (di, dx, nh, ds, ng, l) = (
            dims.d_inner,
            dims.d_xbc(),
            dims.nheads,
            dims.d_state,
            dims.ngroups,
            dims.d_conv,
        );
        let hd = dims.headdim();
        let width = di + dx + nh;
        let silu = |v: f32| v / (1.0 + (-v).exp());
        let mut h = vec![0f32; nh * hd * ds];
        let mut out = vec![0f32; seq * di];
        for t in 0..seq {
            let row = |u: usize| &proj[u * width..(u + 1) * width];
            let xbc: Vec<f32> = (0..dx)
                .map(|c| {
                    let mut acc = conv_b[c];
                    for j in 0..l {
                        // Tap j reads the input l - 1 - j steps back.
                        let back = l - 1 - j;
                        if t >= back {
                            acc += conv_w[c * l + j] * row(t - back)[di + c];
                        }
                    }
                    silu(acc)
                })
                .collect();
            let r = row(t);
            let mut y = vec![0f32; di];
            for head in 0..nh {
                let g = head / (nh / ng);
                let dt = (1.0 + (r[di + dx + head] + dt_bias[head]).exp()).ln();
                for p in 0..hd {
                    let x = xbc[head * hd + p];
                    let mut acc = 0f32;
                    for st in 0..ds {
                        let hi = (head * hd + p) * ds + st;
                        let b = xbc[di + g * ds + st];
                        let c = xbc[di + ng * ds + g * ds + st];
                        h[hi] = h[hi] * (dt * a[head]).exp() + dt * x * b;
                        acc += h[hi] * c;
                    }
                    y[head * hd + p] = (acc + dvec[head] * x) * silu(r[head * hd + p]);
                }
            }
            let group = di / ng;
            for g in 0..ng {
                let ys = &y[g * group..(g + 1) * group];
                let rms = (ys.iter().map(|v| v * v).sum::<f32>() / group as f32 + eps).sqrt();
                for i in 0..group {
                    out[t * di + g * group + i] = ys[i] / rms * norm[g * group + i];
                }
            }
        }
        out
    }

    #[test]
    fn the_host_mamba2_matches_the_block_over_the_history() {
        let dims = Mamba2Dims {
            d_inner: 8,
            d_state: 4,
            ngroups: 2,
            d_conv: 4,
            nheads: 4,
        };
        let (hidden, seq, eps) = (12usize, 6usize, 1e-5f32);
        let width = dims.d_inner + dims.d_xbc() + dims.nheads;
        let dev = Device::Cpu;
        let t = |v: Vec<f32>, shape: &[usize]| Tensor::from_vec(v, shape, &dev).unwrap();
        let w_in = values(width * hidden, 11, 0.3);
        let conv_w = values(dims.d_xbc() * dims.d_conv, 12, 0.5);
        let conv_b = values(dims.d_xbc(), 13, 0.2);
        let a: Vec<f32> = values(dims.nheads, 14, 0.5)
            .iter()
            .map(|v| -(0.5 + v.abs()))
            .collect();
        let dvec = values(dims.nheads, 15, 1.0);
        let dt_bias = values(dims.nheads, 16, 0.5);
        let norm = values(dims.d_inner, 17, 1.0);
        let in_proj = QMatMul::from_qtensor(
            crate::tensor::quantized::QTensor::quantize_onto(
                &t(w_in.clone(), &[width, hidden]),
                crate::tensor::quantized::GgmlDType::F32,
                &dev,
            )
            .unwrap(),
        )
        .unwrap();
        let weights = Mamba2Weights {
            in_proj,
            conv_w: t(conv_w.clone(), &[dims.d_xbc(), dims.d_conv]),
            conv_b: t(conv_b.clone(), &[dims.d_xbc()]),
            a: t(a.clone(), &[dims.nheads, 1]),
            d: t(dvec.clone(), &[dims.nheads, 1]),
            dt_bias: t(dt_bias.clone(), &[dims.nheads]),
            norm_w: t(norm.clone(), &[dims.ngroups, dims.d_inner / dims.ngroups]),
        };
        let mut mixer = Mixer::Ssm(Mamba2::new(weights, dims, eps as f64, &dev).unwrap());
        let xs = values(seq * hidden, 18, 1.0);
        let x = t(xs.clone(), &[1, seq, hidden]);
        let proj: Vec<f32> = (0..seq)
            .flat_map(|u| {
                let xr = &xs[u * hidden..(u + 1) * hidden];
                (0..width)
                    .map(|o| {
                        (0..hidden)
                            .map(|k| w_in[o * hidden + k] * xr[k])
                            .sum::<f32>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let want = mamba2_reference(
            &proj, &conv_w, &conv_b, &a, &dvec, &dt_bias, &norm, dims, eps, seq,
        );
        let v = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let whole = v(&mixer.forward(&x, 0).unwrap());
        let diff = whole
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(diff < 1e-4, "max diff {diff}");
        // Two passes through a snapshot give the single pass.
        mixer.forward(&x.narrow(1, 0, 4).unwrap(), 0).unwrap();
        let snap = mixer.snapshot().unwrap().unwrap();
        mixer.forward(&x, 0).unwrap();
        mixer.restore(&snap).unwrap();
        let tail = v(&mixer.forward(&x.narrow(1, 4, seq - 4).unwrap(), 4).unwrap());
        let resumed = tail
            .iter()
            .zip(&whole[4 * dims.d_inner..])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(resumed < 1e-5, "resumed max diff {resumed}");
    }

    #[test]
    fn the_host_delta_rule_matches_the_row_by_row_tensor_path() {
        let dev = Device::Cpu;
        for tiled in [false, true] {
            let dims = DeltaDims {
                k_heads: 2,
                v_heads: 4,
                k_dim: 4,
                v_dim: 3,
                conv_kernel: 4,
                v_heads_tiled: tiled,
                gate_sigmoid: false,
            };
            let (hidden, seq, eps) = (10usize, 5usize, 1e-6f64);
            let q = |seed, rows| {
                let w = Tensor::from_vec(values(rows * hidden, seed, 0.4), (rows, hidden), &dev)
                    .unwrap();
                QMatMul::from_qtensor(
                    crate::tensor::quantized::QTensor::quantize_onto(
                        &w,
                        crate::tensor::quantized::GgmlDType::F32,
                        &dev,
                    )
                    .unwrap(),
                )
                .unwrap()
            };
            let zn = dims.v_heads * dims.v_dim;
            let make = |packed: bool| {
                let proj = match packed {
                    false => DeltaProj::Split {
                        qkv: q(41, dims.conv_dim()),
                        z: q(42, zn),
                        alpha: q(43, dims.v_heads),
                        beta: q(44, dims.v_heads),
                    },
                    true => DeltaProj::Ba {
                        qkv: q(41, dims.conv_dim()),
                        z: q(42, zn),
                        ba: q(45, 2 * dims.v_heads),
                    },
                };
                let t = |v: Vec<f32>, shape: &[usize]| Tensor::from_vec(v, shape, &dev).unwrap();
                let a: Vec<f32> = values(dims.v_heads, 46, 0.5)
                    .iter()
                    .map(|v| -(0.3 + v.abs()))
                    .collect();
                GatedDeltaNet::new(
                    proj,
                    t(
                        values(dims.conv_dim() * dims.conv_kernel, 47, 0.5),
                        &[dims.conv_dim(), dims.conv_kernel],
                    ),
                    t(a, &[dims.v_heads]),
                    t(values(dims.v_heads, 48, 0.5), &[dims.v_heads]),
                    t(values(dims.v_dim, 49, 1.0), &[dims.v_dim]),
                    dims,
                    eps,
                    &dev,
                )
                .unwrap()
            };
            let x =
                Tensor::from_vec(values(seq * hidden, 50, 1.0), (1, seq, hidden), &dev).unwrap();
            let v = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for packed in [false, true] {
                let mut host = make(packed);
                let mut rows = make(packed);
                let a = v(&host.forward(&x, 0).unwrap());
                let b = v(&rows.forward_rows(&x, 1, seq).unwrap());
                let diff = a
                    .iter()
                    .zip(&b)
                    .map(|(p, q)| (p - q).abs())
                    .fold(0f32, f32::max);
                assert!(
                    diff < 1e-4,
                    "tiled={tiled} packed={packed}: max diff {diff}"
                );
                // Resumed from a snapshot, the host path continues the same sequence.
                let mut m = Mixer::DeltaNet(make(packed));
                m.forward(&x.narrow(1, 0, 2).unwrap(), 0).unwrap();
                let snap = m.snapshot().unwrap().unwrap();
                m.forward(&x, 0).unwrap();
                m.restore(&snap).unwrap();
                let tail = v(&m.forward(&x.narrow(1, 2, seq - 2).unwrap(), 2).unwrap());
                let resumed = tail
                    .iter()
                    .zip(&a[2 * zn..])
                    .map(|(p, q)| (p - q).abs())
                    .fold(0f32, f32::max);
                assert!(resumed < 1e-5, "resumed max diff {resumed}");
            }
        }
    }

    /// A verify window over its rows, returned to every count of kept rows, eagerly and
    /// replayed from a graph, leaves the recurrent state and the convolution window the kept
    /// rows leave decoded one by one, bit for bit.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn a_window_returned_to_its_kept_rows_leaves_the_state_of_decoding_them() {
        use crate::inference::engine::decode_step::VERIFY_ROWS;
        let Ok(dev) = Device::new_cuda(0) else {
            return;
        };
        let stream = crate::tensor::cuda_ext::stream_of(&dev).unwrap();
        let dims = DeltaDims {
            k_heads: 2,
            v_heads: 4,
            k_dim: 64,
            v_dim: 64,
            conv_kernel: 4,
            v_heads_tiled: true,
            gate_sigmoid: false,
        };
        let (hidden, prefix) = (32usize, 2usize);
        let host = Device::Cpu;
        let q = |seed, rows| {
            let w =
                Tensor::from_vec(values(rows * hidden, seed, 0.4), (rows, hidden), &host).unwrap();
            QMatMul::from_qtensor(
                crate::tensor::quantized::QTensor::quantize_onto(
                    &w,
                    crate::tensor::quantized::GgmlDType::Q8_0,
                    &dev,
                )
                .unwrap(),
            )
            .unwrap()
        };
        let make = || {
            let t = |v: Vec<f32>, shape: &[usize]| Tensor::from_vec(v, shape, &host).unwrap();
            let a: Vec<f32> = values(dims.v_heads, 46, 0.5)
                .iter()
                .map(|v| -(0.3 + v.abs()))
                .collect();
            Mixer::DeltaNet(
                GatedDeltaNet::new(
                    DeltaProj::Split {
                        qkv: q(41, dims.conv_dim()),
                        z: q(42, dims.v_heads * dims.v_dim),
                        alpha: q(43, dims.v_heads),
                        beta: q(44, dims.v_heads),
                    },
                    t(
                        values(dims.conv_dim() * dims.conv_kernel, 47, 0.5),
                        &[dims.conv_dim(), dims.conv_kernel],
                    ),
                    t(a, &[dims.v_heads]),
                    t(values(dims.v_heads, 48, 0.5), &[dims.v_heads]),
                    t(values(dims.v_dim, 49, 1.0), &[dims.v_dim]),
                    dims,
                    1e-6,
                    &dev,
                )
                .unwrap(),
            )
        };
        let n = prefix + VERIFY_ROWS;
        let x = Tensor::from_vec(values(n * hidden, 50, 1.0), (1, n, hidden), &host)
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let _order = crate::tensor::quantized::DecodeOrder::enter();
        let _in_place = crate::inference::moe_cuda::in_place_state(true);
        let decode = |m: &mut Mixer, from: usize, to: usize| {
            for i in from..to {
                m.forward(&x.narrow(1, i, 1).unwrap(), i).unwrap();
            }
        };
        let state = |m: &Mixer| {
            let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            stream.synchronize().unwrap();
            match m.checkpoint().unwrap() {
                Some(MixerState::DeltaNet(Some(SsmState::Card { conv, h }))) => {
                    (flat(&h), flat(&conv))
                }
                _ => panic!("no card state"),
            }
        };
        let mut kept_bytes = 0;
        for rows in 2..=VERIFY_ROWS {
            for kept in 0..=rows {
                for recorded in [false, true] {
                    let mut m = make();
                    decode(&mut m, 0, prefix);
                    assert!(m.window_rows_ready(VERIFY_ROWS, &dev).unwrap());
                    kept_bytes = m.window_rows_bytes();
                    let keep = m.checkpoint().unwrap().unwrap();
                    let window = x.narrow(1, prefix, rows).unwrap();
                    m.set_window_at(Some(0));
                    if recorded {
                        stream.synchronize().unwrap();
                        crate::tensor::cuda_ext::begin_capture(&stream).unwrap();
                        m.save_in_place(&keep).unwrap();
                        m.forward(&window, prefix).unwrap();
                        let graph = crate::tensor::cuda_ext::end_capture(&stream)
                            .unwrap()
                            .expect("a window records kernels");
                        graph.launch().unwrap();
                    } else {
                        m.save_in_place(&keep).unwrap();
                        m.forward(&window, prefix).unwrap();
                    }
                    m.set_window_at(None);
                    if kept < rows {
                        m.restore_in_place(&keep).unwrap();
                        assert!(m.replay_window(kept).unwrap());
                    }
                    let mut alone = make();
                    decode(&mut alone, 0, prefix + kept);
                    let (got, want) = (state(&m), state(&alone));
                    let at = format!("{rows} rows, {kept} kept, recorded {recorded}");
                    assert!(got.0 == want.0, "{at}: the recurrent state differs");
                    assert!(got.1 == want.1, "{at}: the convolution window differs");
                }
            }
        }
        let d = dims;
        let one = (d.v_heads * d.v_dim * d.k_dim + d.conv_dim() * (d.conv_kernel - 1)) * 4;
        eprintln!("a state {one} bytes, the window's rows {kept_bytes} bytes");
    }
}
