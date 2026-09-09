//! Mel-Band RoFormer (vocal source separation) - full-Rust port of the
//! Kimberley Jensen vocal checkpoint (MIT weights), architecture per lucidrains'
//! MIT `BS-RoFormer` repo as vendored (frozen) in ZFTurbo's
//! `Music-Source-Separation-Training` (`models/bs_roformer/mel_band_roformer.py`).
//!
//! Pipeline per 8 s chunk (stereo 44.1 kHz, 352800 samples):
//!   STFT (2048/441 Hann periodic, center REFLECT pad) -> stereo folded into the
//!   freq axis -> 60 overlapping mel-band gathers -> per-band RMSNorm+Linear -> 384-d
//!   band tokens -> 6 x (time transformer over frames, freq transformer over bands)
//!   with INTERLEAVED (GPT-J) rotary embeddings shared across depth per axis and
//!   per-head sigmoid gating -> per-band mask MLP (Tanh, GLU) -> overlapping bands
//!   scatter-ADD then averaged per bin -> complex mask x STFT -> DC bin zeroed
//!   (`zero_dc=True` in the vendored copy) -> iSTFT.
//!
//! KJ config: dim 384, depth 6, heads 8, dim_head 64, time/freq_transformer_depth 1,
//! num_bands 60, stereo, num_stems 1, mask_estimator_depth 2. 228.2M params fp32.
//!
//! The 60 mel-band bin ranges come from `librosa.filters.mel(44100, 2048, 60) > 0`
//! binarization with the first/last-bin coverage fixes. They are STATIC for the
//! checkpoint and the #1 parity risk (librosa slaney-norm edge behavior), so they
//! are embedded as constants - dumped once from the Python oracle and verified
//! against the phase-1 manifest (bands are contiguous, <=2 bands overlap per bin).

use crate::inference::codec::stft::{istft, stft_reflect, Frame};
use crate::inference::model::acestep::ops::sdpa;
use crate::tensor::layer::{Linear, Mlp};
use crate::tensor::VarBuilder;
use crate::tensor::{DType, Device, Error, Result, Tensor, D};

pub const SAMPLE_RATE: u32 = 44_100;
pub const CHUNK_SIZE: usize = 352_800; // 8 s
pub const N_FFT: usize = 2048;
pub const HOP: usize = 441;
const N_BINS: usize = N_FFT / 2 + 1; // 1025
const DIM: usize = 384;
const DEPTH: usize = 6;
const HEADS: usize = 8;
const DIM_HEAD: usize = 64;
const FF_MULT: usize = 4;
const MASK_HIDDEN: usize = DIM * 4; // 1536
const CHANNELS: usize = 2; // stereo
/// The model's RMSNorm is `F.normalize(x, dim=-1) * sqrt(dim) * gamma`, i.e. the
/// l2 NORM is clamped at 1e-12 (not an eps under the mean-square). The distinction
/// matters on near-silent band rows (high-freq bands of quiet audio): torch still
/// emits unit-scaled direction vectors there, while a mean-square eps collapses
/// them to ~0 - a real parity gap in `band_split` (cosine 0.997 vs 1.0).
const NORM_CLAMP: f32 = 1e-12;

/// `F.normalize(x, dim=-1) * sqrt(d) * gamma` - the exact RMSNorm of the model.
fn melband_rms_norm(x: &Tensor, gamma: &Tensor) -> Result<Tensor> {
    let d = *x
        .dims()
        .last()
        .ok_or_else(|| Error::msg("rms_norm on rank-0"))?;
    let norm = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let clamp = Tensor::full(NORM_CLAMP, 1usize, &x.device())?;
    let denom = norm.broadcast_maximum(&clamp)?;
    x.broadcast_div(&denom)?
        .affine((d as f32).sqrt(), 0.0)?
        .broadcast_mul(gamma)
}

/// The 60 mel bands as `(first_bin, n_bins)` over the 1025 mono freq bins.
/// Every band is a contiguous range; adjacent bands overlap by <= a few bins
/// (each bin is covered by 1 or 2 bands). Verified against the phase-1 oracle
/// dump (`manifest.json:band_bin_lists_mono`).
const BAND_RANGES: [(usize, usize); 60] = [
    (0, 7),
    (4, 6),
    (7, 6),
    (10, 6),
    (13, 6),
    (16, 6),
    (19, 6),
    (22, 6),
    (25, 6),
    (28, 6),
    (31, 6),
    (34, 6),
    (37, 6),
    (40, 6),
    (43, 6),
    (46, 7),
    (49, 7),
    (53, 7),
    (56, 9),
    (60, 9),
    (65, 9),
    (69, 10),
    (74, 10),
    (79, 11),
    (84, 13),
    (90, 13),
    (97, 13),
    (103, 15),
    (110, 16),
    (118, 17),
    (126, 19),
    (135, 20),
    (145, 20),
    (155, 22),
    (165, 24),
    (177, 26),
    (189, 28),
    (203, 29),
    (217, 31),
    (232, 33),
    (248, 36),
    (265, 39),
    (284, 41),
    (304, 44),
    (325, 47),
    (348, 50),
    (372, 54),
    (398, 57),
    (426, 61),
    (455, 66),
    (487, 71),
    (521, 76),
    (558, 80),
    (597, 86),
    (638, 93),
    (683, 99),
    (731, 105),
    (782, 113),
    (836, 122),
    (895, 130),
];
const NUM_BANDS: usize = BAND_RANGES.len();

/// Per-band model input width: bins x 2 (stereo) x 2 (re/im).
fn band_dim(cnt: usize) -> usize {
    cnt * CHANNELS * 2
}

/// Sum of all per-band input widths (= 7916 for this checkpoint).
fn total_band_dim() -> usize {
    BAND_RANGES.iter().map(|&(_, c)| band_dim(c)).sum()
}

/// How many bands cover each mono freq bin (1 or 2) - the mask-average denominator.
fn bands_per_bin() -> [f32; N_BINS] {
    let mut cover = [0f32; N_BINS];
    for &(start, cnt) in BAND_RANGES.iter() {
        for f in start..start + cnt {
            cover[f] += 1.0;
        }
    }
    cover
}

/// Optional per-component intermediate captures for the parity harness. Every
/// field mirrors one phase-1 oracle dump, in the SAME memory layout.
#[derive(Default)]
pub struct MelbandTaps {
    /// `[2, 1025, T, 2]` - per-channel complex STFT (dump `stft_complex`).
    pub stft: Option<Vec<f32>>,
    /// `[T, 60, 384]` - BandSplit output (dump `band_split_out`, batch dim dropped).
    pub band_split: Option<Vec<f32>>,
    /// `[60, T, 384]` - layer-0 time-transformer output, packed (b.f, t, d).
    pub layer0_time: Option<Vec<f32>>,
    /// `[T, 60, 384]` - layer-0 freq-transformer output, packed (b.t, f, d).
    pub layer0_freq: Option<Vec<f32>>,
    /// `[T, 7916]` - mask estimator output pre-scatter (dump `mask_raw`).
    pub mask_raw: Option<Vec<f32>>,
    /// `[2050, T, 2]` - scatter-added / averaged complex mask (dump `mask_avg_complex`).
    pub mask_avg: Option<Vec<f32>>,
}

/// Attention with per-head sigmoid gating (`out * to_gates(x).sigmoid()`),
/// bias-less fused QKV, pre-RMSNorm, interleaved rotary embedding.
struct Attention {
    norm_g: Tensor, // [384]
    qkv: Linear,    // [3*512, 384] no bias
    gates: Linear,  // [8, 384] + bias
    out: Linear,    // [384, 512] no bias
}

impl Attention {
    fn load(vb: &VarBuilder) -> Result<Self> {
        let dim_inner = HEADS * DIM_HEAD;
        Ok(Self {
            norm_g: vb.get(DIM, "norm.gamma")?,
            qkv: Linear::new(vb.get((3 * dim_inner, DIM), "to_qkv.weight")?, None)?,
            gates: Linear::new(
                vb.get((HEADS, DIM), "to_gates.weight")?,
                Some(vb.get(HEADS, "to_gates.bias")?),
            )?,
            out: Linear::new(vb.get((DIM, dim_inner), "to_out.0.weight")?, None)?,
        })
    }

    /// `x`: `[b, n, 384]`; `cos`/`sin`: `[n, 32]`.
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, n, _d) = x.dims3()?;
        let xn = melband_rms_norm(x, &self.norm_g)?;
        // (qkv h d) packing order -> [3, b, h, n, dh].
        let qkv = self
            .qkv
            .forward(&xn)?
            .reshape((b, n, 3, HEADS, DIM_HEAD))?
            .permute([2, 0, 3, 1, 4])?
            .contiguous()?;
        let q = qkv.get_on_dim(0, 0)?.rope_i(cos, sin)?;
        let k = qkv.get_on_dim(0, 1)?.rope_i(cos, sin)?;
        let v = qkv.get_on_dim(0, 2)?;
        let scale = (DIM_HEAD as f32).powf(-0.5);
        // The [b, h, n, n] score matrix for the time axis (b=60 bands, n=~801
        // frames) is ~1.2 GB f32 - chunk the independent batch rows to cap the
        // transient (bit-exact: rows never interact).
        let budget = 256usize << 20;
        let per_row = HEADS * n * n * 4;
        let step = (budget / per_row.max(1)).clamp(1, b);
        let out = if step >= b {
            sdpa(&q, &k, &v, None, false, scale, 1.0)?
        } else {
            let mut outs = Vec::with_capacity(b.div_ceil(step));
            let mut off = 0;
            while off < b {
                let l = step.min(b - off);
                outs.push(sdpa(
                    &q.narrow(0, off, l)?,
                    &k.narrow(0, off, l)?,
                    &v.narrow(0, off, l)?,
                    None,
                    false,
                    scale,
                    1.0,
                )?);
                off += l;
            }
            let refs: Vec<&Tensor> = outs.iter().collect();
            Tensor::cat(&refs, 0)?
        };
        // per-head sigmoid gates: [b, n, h] -> [b, h, n, 1]
        let g = self
            .gates
            .forward(&xn)?
            .sigmoid()?
            .transpose(1, 2)?
            .contiguous()?
            .unsqueeze(3)?;
        let out = out.broadcast_mul(&g)?;
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, n, HEADS * DIM_HEAD))?;
        self.out.forward(&out)
    }
}

/// RMSNorm -> Linear(x4) -> GELU (exact/erf) -> Linear.
///
/// Only the norm is this codec's own: everything after it is the shared widen-activate-narrow
/// triple, held as an [`Mlp`]. `Activation::Gelu` is the error-function form, which is what this
/// checkpoint was trained with - the tanh approximation is a different function here, not a
/// spelling of the same one.
struct FeedForward {
    norm_g: Tensor,
    tail: Mlp,
}

impl FeedForward {
    fn load(vb: &VarBuilder) -> Result<Self> {
        let inner = DIM * FF_MULT;
        Ok(Self {
            norm_g: vb.get(DIM, "net.0.gamma")?,
            // net indices: 0 RMSNorm, 1 Linear, 2 GELU, 3 Dropout, 4 Linear, 5 Dropout
            tail: Mlp::new(
                Linear::new(
                    vb.get((inner, DIM), "net.1.weight")?,
                    Some(vb.get(inner, "net.1.bias")?),
                )?,
                crate::tensor::ops::Activation::Gelu,
                Linear::new(
                    vb.get((DIM, inner), "net.4.weight")?,
                    Some(vb.get(DIM, "net.4.bias")?),
                )?,
            ),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        melband_rms_norm(x, &self.norm_g)?.apply(&self.tail)
    }
}

/// One axial transformer (depth 1 for this checkpoint): attn + FF residuals,
/// then an output RMSNorm.
struct TransformerBlock {
    attn: Attention,
    ff: FeedForward,
    out_norm_g: Tensor,
}

impl TransformerBlock {
    fn load(vb: &VarBuilder) -> Result<Self> {
        // depth is 1 -> the single inner layer lives at `layers.0.{0,1}`.
        let inner = vb.pp("layers.0");
        Ok(Self {
            attn: Attention::load(&inner.pp("0"))?,
            ff: FeedForward::load(&inner.pp("1"))?,
            out_norm_g: vb.get(DIM, "norm.gamma")?,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let x = self.attn.forward(x, cos, sin)?.add(x)?;
        let x = self.ff.forward(&x)?.add(&x)?;
        melband_rms_norm(&x, &self.out_norm_g)
    }
}

/// Per-band input featurizer: RMSNorm(dim_in) + Linear(dim_in -> 384).
struct BandFeature {
    norm_g: Tensor,
    lin: Linear,
}

/// Per-band mask MLP (depth 2): Linear(384->1536) -> Tanh -> Linear(1536->1536)
/// -> Tanh -> Linear(1536->2.dim_in) -> GLU.
struct MaskMlp {
    l0: Linear,
    l1: Linear,
    l2: Linear,
}

impl MaskMlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.l0.forward(x)?.tanh()?;
        let h = self.l1.forward(&h)?.tanh()?;
        let y = self.l2.forward(&h)?;
        // GLU over the last dim: first half gated by sigmoid(second half).
        let parts = y.chunk(2, D::Minus1)?;
        parts[0].mul(&parts[1].sigmoid()?)
    }
}

pub struct MelBandRoformer {
    bands: Vec<BandFeature>,
    layers: Vec<(TransformerBlock, TransformerBlock)>, // (time, freq)
    masks: Vec<MaskMlp>,
    time_freqs: Vec<f32>, // [32] rotary inv-freqs, shared across depth
    freq_freqs: Vec<f32>,
    device: Device,
}

impl MelBandRoformer {
    /// Load the fp32 safetensors state_dict (plain module-tree keys).
    pub fn load(path: &str, device: &Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_files(&[path], DType::F32, device)? };

        let mut bands = Vec::with_capacity(NUM_BANDS);
        for (b, &(_, cnt)) in BAND_RANGES.iter().enumerate() {
            let dim_in = band_dim(cnt);
            let p = vb.pp(format!("band_split.to_features.{b}"));
            bands.push(BandFeature {
                norm_g: p.get(dim_in, "0.gamma")?,
                lin: Linear::new(
                    p.get((DIM, dim_in), "1.weight")?,
                    Some(p.get(DIM, "1.bias")?),
                )?,
            });
        }

        let mut layers = Vec::with_capacity(DEPTH);
        for l in 0..DEPTH {
            let time = TransformerBlock::load(&vb.pp(format!("layers.{l}.0")))?;
            let freq = TransformerBlock::load(&vb.pp(format!("layers.{l}.1")))?;
            layers.push((time, freq));
        }

        let mut masks = Vec::with_capacity(NUM_BANDS);
        for (b, &(_, cnt)) in BAND_RANGES.iter().enumerate() {
            let dim_in = band_dim(cnt);
            let p = vb.pp(format!("mask_estimators.0.to_freqs.{b}.0"));
            masks.push(MaskMlp {
                l0: Linear::new(
                    p.get((MASK_HIDDEN, DIM), "0.weight")?,
                    Some(p.get(MASK_HIDDEN, "0.bias")?),
                )?,
                l1: Linear::new(
                    p.get((MASK_HIDDEN, MASK_HIDDEN), "2.weight")?,
                    Some(p.get(MASK_HIDDEN, "2.bias")?),
                )?,
                l2: Linear::new(
                    p.get((2 * dim_in, MASK_HIDDEN), "4.weight")?,
                    Some(p.get(2 * dim_in, "4.bias")?),
                )?,
            });
        }

        // Rotary inv-freqs from the checkpoint (one shared instance per axis,
        // aliased across the 6 layers).
        let time_freqs = vb
            .get(DIM_HEAD / 2, "layers.0.0.layers.0.0.rotary_embed.freqs")?
            .to_device(&Device::Cpu)?
            .to_vec1_f32()?;
        let freq_freqs = vb
            .get(DIM_HEAD / 2, "layers.0.1.layers.0.0.rotary_embed.freqs")?
            .to_device(&Device::Cpu)?
            .to_vec1_f32()?;

        Ok(Self {
            bands,
            layers,
            masks,
            time_freqs,
            freq_freqs,
            device: device.clone(),
        })
    }

    /// Interleaved-rope cos/sin tables `[n, 32]` from the checkpoint inv-freqs.
    fn rope_tables(&self, freqs: &[f32], n: usize) -> Result<(Tensor, Tensor)> {
        let half = freqs.len();
        let (mut c, mut s) = (vec![0f32; n * half], vec![0f32; n * half]);
        for p in 0..n {
            for (j, &f) in freqs.iter().enumerate() {
                let a = p as f32 * f;
                c[p * half + j] = a.cos();
                s[p * half + j] = a.sin();
            }
        }
        Ok((
            Tensor::from_vec_f32(c, (n, half))?.to_device(&self.device)?,
            Tensor::from_vec_f32(s, (n, half))?.to_device(&self.device)?,
        ))
    }

    /// Separate one stereo chunk. `input` is planar `[2 * len]` (ch0 then ch1)
    /// at 44.1 kHz; returns the planar vocal stem of the same shape
    /// (`istft(length=None)` -> `(n_frames-1).hop` = len for hop-multiple lengths).
    pub fn separate_chunk(
        &self,
        input: &[f32],
        mut taps: Option<&mut MelbandTaps>,
    ) -> Result<Vec<f32>> {
        let len = input.len() / CHANNELS;
        if input.len() != len * CHANNELS {
            return Err(Error::msg("separate_chunk: planar stereo input expected"));
        }
        // ---- STFT per channel (reflect center pad, periodic Hann) ----
        let specs: [Vec<Frame>; 2] = [
            stft_reflect(&input[..len], N_FFT, HOP),
            stft_reflect(&input[len..], N_FFT, HOP),
        ];
        let t = specs[0].len();
        if let Some(tp) = taps.as_deref_mut() {
            let mut v = vec![0f32; 2 * N_BINS * t * 2];
            for (s, spec) in specs.iter().enumerate() {
                for (ti, frame) in spec.iter().enumerate() {
                    for (f, &(re, im)) in frame.iter().enumerate() {
                        let base = ((s * N_BINS + f) * t + ti) * 2;
                        v[base] = re;
                        v[base + 1] = im;
                    }
                }
            }
            tp.stft = Some(v);
        }

        // ---- gather band inputs: [t, 7916] (freq-major, then channel, then re/im) ----
        let width = total_band_dim();
        let mut x = vec![0f32; t * width];
        for ti in 0..t {
            let row = &mut x[ti * width..(ti + 1) * width];
            let mut col = 0;
            for &(start, cnt) in BAND_RANGES.iter() {
                for f in start..start + cnt {
                    for spec in specs.iter() {
                        let (re, im) = spec[ti][f];
                        row[col] = re;
                        row[col + 1] = im;
                        col += 2;
                    }
                }
            }
        }
        let x = Tensor::from_vec_f32(x, (t, width))?.to_device(&self.device)?;

        // ---- band split: per-band RMSNorm + Linear -> [t, 60, 384] ----
        let mut feats = Vec::with_capacity(NUM_BANDS);
        let mut off = 0;
        for (band, &(_, cnt)) in BAND_RANGES.iter().enumerate() {
            let dim_in = band_dim(cnt);
            let xb = x.narrow(1, off, dim_in)?.contiguous()?;
            let h = melband_rms_norm(&xb, &self.bands[band].norm_g)?;
            feats.push(self.bands[band].lin.forward(&h)?);
            off += dim_in;
        }
        let mut x = Tensor::stack(&feats, 1)?; // [t, 60, 384]
        if let Some(tp) = taps.as_deref_mut() {
            tp.band_split = Some(x.flatten_all()?.to_vec1_f32()?);
        }

        // ---- 6 x axial (time, freq) transformers ----
        let (cos_t, sin_t) = self.rope_tables(&self.time_freqs, t)?;
        let (cos_f, sin_f) = self.rope_tables(&self.freq_freqs, NUM_BANDS)?;
        for (li, (time_block, freq_block)) in self.layers.iter().enumerate() {
            // time attention: batch = bands, seq = frames
            let xt = x.transpose(0, 1)?.contiguous()?; // [60, t, 384]
            let xt = time_block.forward(&xt, &cos_t, &sin_t)?;
            if li == 0 {
                if let Some(tp) = taps.as_deref_mut() {
                    tp.layer0_time = Some(xt.flatten_all()?.to_vec1_f32()?);
                }
            }
            // freq attention: batch = frames, seq = bands
            let xf = xt.transpose(0, 1)?.contiguous()?; // [t, 60, 384]
            x = freq_block.forward(&xf, &cos_f, &sin_f)?;
            if li == 0 {
                if let Some(tp) = taps.as_deref_mut() {
                    tp.layer0_freq = Some(x.flatten_all()?.to_vec1_f32()?);
                }
            }
        }

        // ---- mask estimator (single stem): per-band MLP + GLU -> [t, 7916] ----
        let mut outs = Vec::with_capacity(NUM_BANDS);
        for (band, mlp) in self.masks.iter().enumerate() {
            let xb = x.narrow(1, band, 1)?.squeeze(1)?.contiguous()?; // [t, 384]
            outs.push(mlp.forward(&xb)?);
        }
        let refs: Vec<&Tensor> = outs.iter().collect();
        let mask = Tensor::cat(&refs, D::Minus1)?; // [t, 7916]
        let mask_v = mask.to_device(&Device::Cpu)?.flatten_all()?.to_vec1_f32()?;
        if let Some(tp) = taps.as_deref_mut() {
            tp.mask_raw = Some(mask_v.clone());
        }

        // ---- scatter-add the overlapping band masks, average per stereo-folded bin ----
        // folded rows: 2050 = 1025 mono bins x 2 channels, row = f.2 + s.
        let cover = bands_per_bin();
        let folded = N_BINS * CHANNELS;
        let mut acc = vec![0f32; folded * t * 2];
        let mut col = 0; // complex column pairs within the [t, 7916] mask rows
        for &(start, cnt) in BAND_RANGES.iter() {
            for f in start..start + cnt {
                for s in 0..CHANNELS {
                    let row = f * CHANNELS + s;
                    for ti in 0..t {
                        let m = ti * total_band_dim() + col * 2;
                        acc[(row * t + ti) * 2] += mask_v[m];
                        acc[(row * t + ti) * 2 + 1] += mask_v[m + 1];
                    }
                    col += 1;
                }
            }
        }
        for f in 0..N_BINS {
            let d = cover[f].max(1e-8);
            for s in 0..CHANNELS {
                let row = f * CHANNELS + s;
                for v in acc[row * t * 2..(row + 1) * t * 2].iter_mut() {
                    *v /= d;
                }
            }
        }
        if let Some(tp) = taps {
            tp.mask_avg = Some(acc.clone());
        }

        // ---- complex mask x STFT, zero DC, iSTFT per channel ----
        let out_len = (t - 1) * HOP;
        let mut out = vec![0f32; CHANNELS * out_len];
        for (s, spec) in specs.iter().enumerate() {
            let mut masked: Vec<Frame> = Vec::with_capacity(t);
            for ti in 0..t {
                let mut frame = Vec::with_capacity(N_BINS);
                for f in 0..N_BINS {
                    if f == 0 {
                        frame.push((0.0, 0.0)); // zero_dc
                        continue;
                    }
                    let row = f * CHANNELS + s;
                    let (mr, mi) = (acc[(row * t + ti) * 2], acc[(row * t + ti) * 2 + 1]);
                    let (sr, si) = spec[ti][f];
                    frame.push((sr * mr - si * mi, sr * mi + si * mr));
                }
                masked.push(frame);
            }
            let wav = istft(&masked, N_FFT, HOP, out_len);
            out[s * out_len..(s + 1) * out_len].copy_from_slice(&wav);
        }
        Ok(out)
    }

    /// Full-length separation with the reference chunked overlap-add loop
    /// (ZFTurbo `demix`, generic mode): chunk 352800, 50% overlap, linear fade
    /// window (fade = chunk/10), reflect border padding, accumulate window.out
    /// and the window itself, divide. `mix` is planar `[2 * len]`; returns the
    /// planar vocal stem `[2 * len]`.
    pub fn separate(&self, mix: &[f32], progress: bool) -> Result<Vec<f32>> {
        let len = mix.len() / CHANNELS;
        let chunk = CHUNK_SIZE;
        let step = chunk / 2; // num_overlap = 2
        let border = chunk - step;
        let fade = chunk / 10;

        // reflect-pad both channels by `border` when the mix is long enough
        let (padded, pad): (Vec<Vec<f32>>, usize) = if len > 2 * border {
            let mut chans = Vec::with_capacity(CHANNELS);
            for s in 0..CHANNELS {
                let ch = &mix[s * len..(s + 1) * len];
                let mut p = Vec::with_capacity(len + 2 * border);
                for i in 0..border {
                    p.push(ch[border - i]);
                }
                p.extend_from_slice(ch);
                for j in 0..border {
                    p.push(ch[len - 2 - j]);
                }
                chans.push(p);
            }
            (chans, border)
        } else {
            (
                (0..CHANNELS)
                    .map(|s| mix[s * len..(s + 1) * len].to_vec())
                    .collect(),
                0,
            )
        };
        let plen = padded[0].len();

        // linear fade-in/out window
        let mut window = vec![1f32; chunk];
        for i in 0..fade {
            let r = i as f32 / (fade - 1) as f32;
            window[i] = r; // fade-in 0->1
            window[chunk - fade + i] = 1.0 - r; // fade-out 1->0
        }

        let mut result = vec![0f32; CHANNELS * plen];
        let mut counter = vec![0f32; CHANNELS * plen];
        let mut i = 0usize;
        let mut n_chunks = 0usize;
        let total_chunks = plen.div_ceil(step);
        while i < plen {
            let seg_len = chunk.min(plen - i);
            // chunk extraction + padding to full size (reflect when > chunk/2 remains)
            let mut part = vec![0f32; CHANNELS * chunk];
            for s in 0..CHANNELS {
                part[s * chunk..s * chunk + seg_len].copy_from_slice(&padded[s][i..i + seg_len]);
                if seg_len < chunk && seg_len > chunk / 2 {
                    for j in 0..chunk - seg_len {
                        // torch reflect continuation of the seg_len-long segment
                        part[s * chunk + seg_len + j] = padded[s][i + seg_len - 2 - j];
                    }
                }
            }
            let vocals = self.separate_chunk(&part, None)?;
            let out_len = vocals.len() / CHANNELS;

            // per-chunk window: first chunk no fade-in, last chunk no fade-out
            let mut w = window.clone();
            if i == 0 {
                for v in w[..fade].iter_mut() {
                    *v = 1.0;
                }
            }
            if i + step >= plen {
                for v in w[chunk - fade..].iter_mut() {
                    *v = 1.0;
                }
            }
            for s in 0..CHANNELS {
                for j in 0..seg_len.min(out_len) {
                    result[s * plen + i + j] += vocals[s * out_len + j] * w[j];
                    counter[s * plen + i + j] += w[j];
                }
            }
            n_chunks += 1;
            if progress {
                eprintln!("[melband] chunk {n_chunks}/{total_chunks} done");
            }
            i += step;
        }

        let mut out = vec![0f32; CHANNELS * len];
        for s in 0..CHANNELS {
            for j in 0..len {
                let c = counter[s * plen + pad + j];
                let v = result[s * plen + pad + j];
                out[s * len + j] = if c > 0.0 { v / c } else { 0.0 };
            }
        }
        Ok(out)
    }
}

impl MelBandRoformer {
    /// Where this model's layers sit, by device.
    pub fn placement(&self) -> Vec<crate::inference::serve::progress::placement::Placed> {
        crate::inference::serve::progress::placement::whole(&self.device, self.layers.len())
    }
}
