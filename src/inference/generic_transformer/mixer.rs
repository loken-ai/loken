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

/// The operator a block's tensors name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixerKind {
    Attention,
    /// LFM2 gated short convolution: `shortconv.{in_proj,conv,out_proj}`.
    ShortConv,
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
    } else {
        MixerKind::Attention
    }
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
}

/// A recurrent mixer's state, copied whole.
#[derive(Clone)]
pub enum MixerState {
    ShortConv(Option<ConvState>),
}

impl MixerState {
    /// Device memory the copy holds.
    pub fn device_bytes(&self) -> u64 {
        match self {
            MixerState::ShortConv(Some(ConvState::Card(t))) => {
                (t.elem_count() * t.dtype().size_in_bytes()) as u64
            }
            MixerState::ShortConv(_) => 0,
        }
    }
}

impl Mixer {
    /// True for a mixer that carries a state beside the KV cache.
    pub fn is_recurrent(&self) -> bool {
        !matches!(self, Mixer::Attention)
    }

    /// The mixed tokens of `x` `[b, seq, hidden]`, before the output projection. Starts a
    /// fresh state at position 0 and advances it over every row otherwise.
    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        match self {
            Mixer::Attention => Err(crate::tensor::Error::msg(
                "attention runs through the layer's attention path",
            )),
            Mixer::ShortConv(m) => m.forward(x, index_pos),
        }
    }

    /// Drops the state; the next forward starts from zeros.
    pub fn reset(&mut self) {
        match self {
            Mixer::Attention => {}
            Mixer::ShortConv(m) => m.state = None,
        }
    }

    /// A deep copy of the state, `None` for attention.
    pub fn snapshot(&self) -> Result<Option<MixerState>> {
        Ok(match self {
            Mixer::Attention => None,
            Mixer::ShortConv(m) => Some(MixerState::ShortConv(
                m.state.as_ref().map(ConvState::deep_copy).transpose()?,
            )),
        })
    }

    /// Makes `state` the mixer's state; the copy stays reusable.
    pub fn restore(&mut self, state: &MixerState) -> Result<()> {
        match (self, state) {
            (Mixer::ShortConv(m), MixerState::ShortConv(s)) => {
                m.state = s.as_ref().map(ConvState::deep_copy).transpose()?;
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
        })
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        if index_pos == 0 {
            self.state = None;
        }
        let (b, seq, _) = x.dims3()?;
        let proj = self.in_proj.forward(x)?;
        if b == 1 && !x.device().is_cuda() {
            let y = self.host_steps(&proj.to_dtype(DType::F32)?, seq)?;
            return Tensor::from_vec(y, (1, seq, self.d_model), &x.device())?.to_dtype(x.dtype());
        }
        let state = match self.state.take() {
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
        self.state = Some(ConvState::Card(state));
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

impl GenericTransformerLayer {
    /// A layer whose tokens mix through a recurrent operator: `x + out(mixer(norm(x)))`,
    /// then the feed-forward sublayer.
    pub(super) fn forward_recurrent(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let x_norm = self.attn_norm.forward(x)?;
        let mixed = self.mixer.forward(&x_norm, index_pos)?;
        let mixed = self.attn_output.forward(&mixed)?;
        let x = (x + mixed.to_dtype(x.dtype())?)?;
        self.ffn_sublayer(&x)
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
}
