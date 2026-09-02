//! Descript Audio Codec (DAC) on the NATIVE tensor substrate (independence
//! The audio codec parler-tts uses to turn discrete audio
//! codes back into a waveform: the codebooks and the decoder, which is
//! everything `decode_codes` reaches.
//!
//! `weight_norm` convs are recomputed HOST-side at load (inference form):
//! `w = weight_v * weight_g / ||weight_v||` - identical numerics to the
//! original helpers, so weights load bit-for-bit from the same safetensors.
//! The snake activation runs as one fused kernel (its per-channel broadcasts
//! aren't tail-aligned), and the decoder's stride-upsampling convs hit the
//! native transposed-conv CUDA gather kernel.

use crate::tensor::layer::{same_length_1d, Conv1d, Conv1dConfig, Embedding};
use crate::tensor::VarBuilder;
use crate::tensor::{DType, Device, Result, Tensor};

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    pub num_codebooks: usize,
    pub model_bitrate: u32,
    pub codebook_size: usize,
    pub latent_dim: usize,
    pub frame_rate: u32,
    pub sampling_rate: u32,
}

// --- weight-norm builders (inference form, host-side fold) ------------------

/// `w[o,i,k] = v[o,i,k] * g[o] / sqrt(sum_{i,k} v[o,i,k]^2)`, folded on host
/// then moved to the compute device.
fn weight_norm_fold(
    vb: &VarBuilder,
    g_shape: (usize, usize, usize),
    v_shape: (usize, usize, usize),
) -> Result<Tensor> {
    let host = vb.to(DType::F32, &Device::Cpu);
    let g = host.get(g_shape, "weight_g")?.to_vec_f32();
    let v = host.get(v_shape, "weight_v")?;
    let (d0, d1, d2) = v_shape;
    let vd = v.to_vec_f32();
    let mut w = vec![0f32; vd.len()];
    let slab = d1 * d2;
    for o in 0..d0 {
        let src = &vd[o * slab..][..slab];
        let norm = src.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let scale = g[o] / norm;
        for (dst, &sv) in w[o * slab..][..slab].iter_mut().zip(src) {
            *dst = sv * scale;
        }
    }
    Tensor::from_vec_f32(w, vec![d0, d1, d2])?.to_device(vb.device())
}

fn conv1d_weight_norm(
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    config: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight = weight_norm_fold(&vb, (out_c, 1, 1), (out_c, in_c, kernel_size))?;
    let bias = vb.get(out_c, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), config))
}

/// Transposed-conv weights are `[c_in, c_out, k]` with `g` per IN channel.
#[derive(Clone, Debug)]
struct ConvTr1d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
}

impl ConvTr1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv_transpose1d(&self.weight, self.padding, 0, self.stride, 1, 1)?;
        match &self.bias {
            Some(b) => y.add_channel_bias(b),
            None => Ok(y),
        }
    }
}

fn conv_transpose1d_weight_norm(
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    bias: bool,
    stride: usize,
    padding: usize,
    vb: VarBuilder,
) -> Result<ConvTr1d> {
    let weight = weight_norm_fold(&vb, (in_c, 1, 1), (in_c, out_c, kernel_size))?;
    let bias = if bias {
        Some(vb.get(out_c, "bias")?)
    } else {
        None
    };
    Ok(ConvTr1d {
        weight,
        bias,
        stride,
        padding,
    })
}

// --- Snake1d activation: x + (1/alpha) * sin(alpha*x)^2 ----------------------

#[derive(Clone, Debug)]
pub struct Snake1d {
    alpha: Tensor,
    inv_alpha: Tensor,
}

impl Snake1d {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let host = vb.to(DType::F32, &Device::Cpu);
        let a = host.get((1, channels, 1), "alpha")?.to_vec_f32();
        let inv: Vec<f32> = a.iter().map(|&x| 1.0 / (x + 1e-9)).collect();
        let alpha = Tensor::from_vec_f32(a, vec![channels])?.to_device(vb.device())?;
        let inv_alpha = Tensor::from_vec_f32(inv, vec![channels])?.to_device(vb.device())?;
        Ok(Self { alpha, inv_alpha })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.snake1d(&self.alpha, &self.inv_alpha)
    }
}

// --- ResidualUnit -----------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ResidualUnit {
    snake1: Snake1d,
    conv1: Conv1d,
    snake2: Snake1d,
    conv2: Conv1d,
}

impl ResidualUnit {
    pub fn new(dim: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("block");
        let snake1 = Snake1d::new(dim, vb.pp(0))?;
        let conv1 = conv1d_weight_norm(dim, dim, 7, same_length_1d(7, dilation), vb.pp(1))?;
        let snake2 = Snake1d::new(dim, vb.pp(2))?;
        let conv2 = conv1d_weight_norm(dim, dim, 1, Default::default(), vb.pp(3))?;
        Ok(Self {
            snake1,
            conv1,
            snake2,
            conv2,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = self.conv2.forward(
            &self
                .snake2
                .forward(&self.conv1.forward(&self.snake1.forward(xs)?)?)?,
        )?;
        let l_dim = xs.rank() - 1;
        let l_x = *xs.dims().last().unwrap_or(&0);
        let l_y = *ys.dims().last().unwrap_or(&0);
        let pad = (l_x - l_y) / 2;
        if pad > 0 {
            ys.add(&xs.narrow(l_dim, pad, l_y)?)
        } else {
            ys.add(xs)
        }
    }
}

// --- The residual stack -----------------------------------------------------

/// The dilations a block's residual units take, in order.
///
/// Widening by three each time is what gives the stack its reach: three units span thirteen
/// times what one does, for three times the cost.
const DILATIONS: [usize; 3] = [1, 3, 9];

/// The three units, numbered from `first` in the checkpoint's own numbering.
fn residual_stack(
    dim: usize,
    first: usize,
    vb: &VarBuilder,
) -> Result<[ResidualUnit; DILATIONS.len()]> {
    let mut units = Vec::with_capacity(DILATIONS.len());
    for (i, &dilation) in DILATIONS.iter().enumerate() {
        units.push(ResidualUnit::new(dim, dilation, vb.pp(first + i))?);
    }
    units
        .try_into()
        .map_err(|_| crate::tensor::Error::msg("dac: the residual stack is not three units"))
}

fn apply_residual(units: &[ResidualUnit; DILATIONS.len()], xs: &Tensor) -> Result<Tensor> {
    let mut h = xs.clone();
    for unit in units {
        h = unit.forward(&h)?;
    }
    Ok(h)
}

// --- DecoderBlock / Decoder -------------------------------------------------

#[derive(Clone, Debug)]
pub struct DecoderBlock {
    snake1: Snake1d,
    conv_tr1: ConvTr1d,
    residual: [ResidualUnit; DILATIONS.len()],
}

impl DecoderBlock {
    pub fn new(in_dim: usize, out_dim: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("block");
        let snake1 = Snake1d::new(in_dim, vb.pp(0))?;
        let conv_tr1 = conv_transpose1d_weight_norm(
            in_dim,
            out_dim,
            2 * stride,
            true,
            stride,
            stride.div_ceil(2),
            vb.pp(1),
        )?;
        let residual = residual_stack(out_dim, 2, &vb)?;
        Ok(Self {
            snake1,
            conv_tr1,
            residual,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv_tr1.forward(&self.snake1.forward(xs)?)?;
        apply_residual(&self.residual, &xs)
    }
}

#[derive(Clone, Debug)]
pub struct Decoder {
    conv1: Conv1d,
    blocks: Vec<DecoderBlock>,
    snake1: Snake1d,
    conv2: Conv1d,
}

impl Decoder {
    /// `channels` is the width the head hands to the first block; every block after that trades
    /// half of it for the length its stride buys, and the tail runs at whatever is left.
    pub fn new(
        in_c: usize,
        channels: usize,
        rates: &[usize],
        d_out: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Halving `n` times is a shift by `n`, so a block's two widths are a function of where
        // it sits rather than of a counter walked alongside the loop.
        let width = |depth: usize| channels >> depth;
        let n = rates.len();
        let tail = width(n);
        let model = vb.pp("model");
        let conv1 = conv1d_weight_norm(in_c, width(0), 7, same_length_1d(7, 1), model.pp(0))?;
        let blocks = rates
            .iter()
            .enumerate()
            .map(|(d, &stride)| DecoderBlock::new(width(d), width(d + 1), stride, model.pp(d + 1)))
            .collect::<Result<Vec<_>>>()?;
        let snake1 = Snake1d::new(tail, model.pp(n + 1))?;
        let conv2 = conv1d_weight_norm(tail, d_out, 7, same_length_1d(7, 1), model.pp(n + 2))?;
        Ok(Self {
            conv1,
            blocks,
            snake1,
            conv2,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = self.conv1.forward(xs)?;
        for block in self.blocks.iter() {
            xs = block.forward(&xs)?;
        }
        self.conv2.forward(&self.snake1.forward(&xs)?)
    }
}

// --- (Residual) Vector Quantizer --------------------------------------------

#[derive(Clone, Debug)]
pub struct VectorQuantizer {
    out_proj: Conv1d,
    codebook: Embedding,
}

impl VectorQuantizer {
    pub fn new(in_dim: usize, cb_size: usize, cb_dim: usize, vb: VarBuilder) -> Result<Self> {
        // in_proj only matters for encoding; decode goes codebook -> out_proj.
        let out_proj =
            conv1d_weight_norm(cb_dim, in_dim, 1, Default::default(), vb.pp("out_proj"))?;
        let codebook = crate::tensor::layer::embedding(cb_size, cb_dim, &vb.pp("codebook"))?;
        Ok(Self { out_proj, codebook })
    }

    /// `[1, seq]` u32 ids -> `[1, cb_dim, seq]`.
    pub fn decode_code(&self, embed_id: &Tensor) -> Result<Tensor> {
        self.codebook.forward(embed_id)?.transpose(1, 2)
    }
}

#[derive(Clone, Debug)]
pub struct ResidualVectorQuantizer {
    quantizers: Vec<VectorQuantizer>,
}

impl ResidualVectorQuantizer {
    /// The codebooks the configuration names, in the order they are applied: each one codes the
    /// part of the signal the ones before it left behind, and `cb_dim` is how wide an entry is
    /// before `out_proj` takes it back up to the latent.
    pub fn new(cfg: &Config, cb_dim: usize, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("quantizers");
        let mut quantizers = Vec::with_capacity(cfg.num_codebooks);
        for i in 0..cfg.num_codebooks {
            let q = VectorQuantizer::new(cfg.latent_dim, cfg.codebook_size, cb_dim, vb.pp(i))?;
            quantizers.push(q);
        }
        Ok(Self { quantizers })
    }

    /// Host codes `[nq, seq]` (row-major) -> summed latent `[1, latent_dim, seq]`
    /// on `device`.
    pub fn from_codes(&self, codes: &[u32], seq: usize, device: &Device) -> Result<Tensor> {
        let mut sum: Option<Tensor> = None;
        for (idx, quantizer) in self.quantizers.iter().enumerate() {
            let ids = Tensor::from_vec_u32(codes[idx * seq..][..seq].to_vec(), vec![1, seq])?
                .to_device(device)?;
            let z_p_i = quantizer.decode_code(&ids)?;
            let z_q_i = quantizer.out_proj.forward(&z_p_i)?;
            sum = Some(match sum {
                None => z_q_i,
                Some(s) => s.add(&z_q_i)?,
            });
        }
        sum.ok_or_else(|| crate::tensor::Error::msg("empty codebooks"))
    }
}

// --- Model ------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Model {
    pub quantizer: ResidualVectorQuantizer,
    pub decoder: Decoder,
    device: Device,
}

/// How wide one codebook entry is. The configuration says how many codebooks and how many
/// entries each holds, but not this: it is fixed by the checkpoint's own shape.
const CODEBOOK_DIM: usize = 8;

/// What the decoder's head hands to its first block.
const DECODER_WIDTH: usize = 1536;

/// What each block lengthens the signal by. Together they take 1536 channels down to 96 and the
/// frame rate up 512-fold, which is the codec's compression written as four steps.
const DECODER_RATES: [usize; 4] = [8, 8, 4, 2];

/// One waveform out.
const AUDIO_CHANNELS: usize = 1;

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let quantizer = ResidualVectorQuantizer::new(cfg, CODEBOOK_DIM, vb.pp("quantizer"))?;
        let decoder = Decoder::new(
            cfg.latent_dim,
            DECODER_WIDTH,
            &DECODER_RATES,
            AUDIO_CHANNELS,
            vb.pp("decoder"),
        )?;
        Ok(Self {
            quantizer,
            decoder,
            device,
        })
    }

    /// Audio codes `[1, n_codebooks, seq]` (any integer dtype, facade) ->
    /// waveform `[1, 1, samples]` on the caller's device.
    pub fn decode_codes(
        &self,
        audio_codes: &crate::tensor::Tensor,
    ) -> crate::tensor::Result<crate::tensor::Tensor> {
        let dims = audio_codes.dims().to_vec();
        if dims.len() != 3 || dims[0] != 1 {
            return Err(crate::tensor::Error::msg(format!(
                "decode_codes expects [1, nq, seq], got {dims:?}"
            )));
        }
        let seq = dims[2];
        let codes: Vec<u32> = audio_codes
            .flatten_all()?
            .to_dtype(crate::tensor::DType::U32)?
            .to_vec1::<u32>()?;
        let latent = self
            .quantizer
            .from_codes(&codes, seq, &self.device)
            .and_then(|z| self.decoder.forward(&z))?;
        latent.to_device(&audio_codes.device())
    }
}
