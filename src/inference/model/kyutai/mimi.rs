//! Kyutai `tts-1.6b-en_fr` - Mimi RVQ codec DECODER (component 3 of the TTS port).
//!
//! codes `[K, T]` (1 semantic + the acoustic RVQ codebooks, text codebook dropped first)
//!   -> RVQ dequant -> latent [C, T]
//!   -> upsample (depthwise convtr) -> decoder-transformer (RoPE + LayerScale)
//!   -> SEANet decoder (strided transposed convs) -> 24 kHz mono waveform.
//!
//! Every width, codebook count, layer count and stride is read from the tensors the
//! checkpoint stores. Codebook = `embedding_sum / max(cluster_usage, eps)` (Mimi EMA).
//!
//! The decoder-transformer's layers are a transformer of the generic route, placed by
//! `HeteroPlan` and the ladder with their own demand.
//! The rest sits on the decoder's device.

use crate::inference::generic_transformer::class_load::{self, Rows};
use crate::inference::generic_transformer::GenericHeteroTransformer;
#[cfg(test)]
use crate::inference::model::acestep::fsq::rope_tables;
use crate::inference::model::codec_blocks::{causal_conv1d, causal_convtr1d, Rvq};
#[cfg(test)]
use crate::inference::model::sdpa::sdpa;
use crate::tensor::VarBuilder;
use crate::tensor::{DType, Device, Error, Result, Tensor};

/// The class of the decoder-transformer on the generic route.
pub const TRANSFORMER_CLASS: &str = "ProjectedTransformer";

/// Where the decoder-transformer's tensors are in the codec's file.
const TRANSFORMER_PREFIX: &str = "decoder_transformer.transformer";

/// Mimi's transformer settings, which its reference code fixes and its checkpoint does not
/// store: the head count its attention splits the width into, and the positions the
/// sliding causal window reaches back.
pub(crate) const ATTENTION_HEADS: usize = 8;
pub(crate) const ATTENTION_WINDOW: usize = 250;
const ROPE_THETA: f32 = 10_000.0;
const LN_EPS: f32 = 1e-5;
const VQ_EPS: f32 = 1e-5;
/// Mimi's output sample rate, fixed by the codec.
pub const SAMPLE_RATE: u32 = 24_000;

/// ELU(1.0) = relu(x) + (exp(min(x,0)) - 1).
fn elu(x: &Tensor) -> Result<Tensor> {
    let pos = x.relu()?;
    let min_x0 = x.affine(-1.0, 0.0)?.relu()?.affine(-1.0, 0.0)?;
    pos.add(&min_x0.exp()?.affine(1.0, -1.0)?)
}

struct Conv {
    w: Tensor,
    b: Option<Tensor>,
}
impl Conv {
    /// A conv whose shape is the stored `{name}.weight`, with its bias when `bias`.
    fn load(vb: &VarBuilder, name: &str, bias: bool) -> Result<Self> {
        let shape = vb.shape_of(&format!("{name}.weight"))?;
        let w = vb.get(shape.clone(), &format!("{name}.weight"))?;
        let b = if bias {
            Some(vb.get(shape[0], &format!("{name}.bias"))?)
        } else {
            None
        };
        Ok(Self { w, b })
    }
    /// A transposed conv `[in, out, k]` with its bias over `out`.
    fn load_tr(vb: &VarBuilder, name: &str) -> Result<Self> {
        let shape = vb.shape_of(&format!("{name}.weight"))?;
        Ok(Self {
            b: Some(vb.get(shape[1], &format!("{name}.bias"))?),
            w: vb.get(shape, &format!("{name}.weight"))?,
        })
    }
    /// The stride of a SEANet resampling conv, whose kernel is twice its stride.
    fn resampling_stride(&self) -> Result<usize> {
        Ok(self.w.shape().dims3()?.2 / 2)
    }
}
/// SEANet residual block: ELU -> conv(k3) -> ELU -> conv(k1) -> + input.
struct ResBlock {
    conv1: Conv,
    conv2: Conv,
}
impl ResBlock {
    fn load(vb: &VarBuilder, idx: usize) -> Result<Self> {
        let p = format!("{idx}.block");
        Ok(Self {
            conv1: Conv::load(vb, &format!("{p}.1.conv.conv"), true)?,
            conv2: Conv::load(vb, &format!("{p}.3.conv.conv"), true)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = elu(x)?;
        let h = causal_conv1d(&h, &self.conv1.w, self.conv1.b.as_ref(), 1, 1, 1)?;
        let h = elu(&h)?;
        let h = causal_conv1d(&h, &self.conv2.w, self.conv2.b.as_ref(), 1, 1, 1)?;
        x.add(&h)
    }
}

// -- decoder transformer layer (pre-norm, fused-QKV MHA + RoPE-i + LayerScale) -
#[cfg(test)]
struct XfLayer {
    norm1_w: Tensor,
    norm1_b: Tensor,
    norm2_w: Tensor,
    norm2_b: Tensor,
    in_proj: Tensor,
    out_proj: Tensor,
    linear1: Tensor,
    linear2: Tensor,
    ls1: Tensor,
    ls2: Tensor,
    dim: usize,
}
#[cfg(test)]
impl XfLayer {
    fn load(vb: &VarBuilder, idx: usize) -> Result<Self> {
        let p = vb.pp(format!("layers.{idx}"));
        let dim = p.shape_of("norm1.weight")?[0];
        let ffn = p.shape_of("linear1.weight")?[0];
        Ok(Self {
            norm1_w: p.get(dim, "norm1.weight")?,
            norm1_b: p.get(dim, "norm1.bias")?,
            norm2_w: p.get(dim, "norm2.weight")?,
            norm2_b: p.get(dim, "norm2.bias")?,
            in_proj: p.get((3 * dim, dim), "self_attn.in_proj_weight")?,
            out_proj: p.get((dim, dim), "self_attn.out_proj.weight")?,
            linear1: p.get((ffn, dim), "linear1.weight")?,
            linear2: p.get((dim, ffn), "linear2.weight")?,
            ls1: p.get(dim, "layer_scale_1.scale")?,
            ls2: p.get(dim, "layer_scale_2.scale")?,
            dim,
        })
    }
    /// `x: [S, dim]` - sliding-causal RoPE attention.
    fn forward(&self, x: &Tensor, dev: &Device) -> Result<Tensor> {
        let dim = self.dim;
        let s = x.shape().dims2()?.0;
        let hd = dim / ATTENTION_HEADS;
        let h = x.layer_norm(&self.norm1_w, Some(&self.norm1_b), LN_EPS)?;
        let qkv = h.matmul_t(&self.in_proj)?;
        let q = qkv.narrow(1, 0, dim)?;
        let k = qkv.narrow(1, dim, dim)?;
        let v = qkv.narrow(1, 2 * dim, dim)?;
        let shp = |t: &Tensor| -> Result<Tensor> {
            t.reshape((s, ATTENTION_HEADS, hd))?
                .transpose(0, 1)?
                .unsqueeze(0)?
                .contiguous()
        };
        let (cosv, sinv) = rope_tables(s, hd, ROPE_THETA);
        let cos = Tensor::from_vec_f32(cosv, (s, hd / 2))?.to_device(dev)?;
        let sin = Tensor::from_vec_f32(sinv, (s, hd / 2))?.to_device(dev)?;
        let q = shp(&q)?.rope_i(&cos, &sin)?;
        let k = shp(&k)?.rope_i(&cos, &sin)?;
        let v = shp(&v)?;
        let scale = 1.0 / (hd as f32).sqrt();
        // sliding causal window (a sequence shorter than the window is full causal)
        let mut data = vec![0f32; s * s];
        for i in 0..s {
            for j in 0..s {
                if j > i || i - j >= ATTENTION_WINDOW {
                    data[i * s + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec_f32(data, (s, s))?.to_device(dev)?;
        let att = sdpa(&q, &k, &v, Some(&mask), false, scale, 1.0)?;
        let att = att
            .transpose(1, 2)?
            .contiguous()?
            .reshape((s, dim))?
            .matmul_t(&self.out_proj)?;
        let x = x.add(&att.broadcast_mul(&self.ls1.reshape((1, dim))?)?)?;
        let h = x.layer_norm(&self.norm2_w, Some(&self.norm2_b), LN_EPS)?;
        let h = h
            .matmul_t(&self.linear1)?
            .gelu_erf()?
            .matmul_t(&self.linear2)?;
        x.add(&h.broadcast_mul(&self.ls2.reshape((1, dim))?)?)
    }
}

/// Full Mimi decoder for tts-1.6b: codes -> 24 kHz waveform.
///
/// The decoder-transformer's layers sit where their placement put them.
/// The rest sits on the decoder's device.
pub struct KyutaiMimiDecoder {
    rvq_first: Rvq,   // the semantic codebooks
    rvq_rest: Rvq,    // the acoustic codebooks
    upsample: Tensor, // [dim,1,k] depthwise convtr
    layers: std::sync::Mutex<GenericHeteroTransformer>,
    conv_in: Conv,
    up: Vec<(Conv, ResBlock)>, // (convtr, resblock) per SEANet stage
    conv_out: Conv,
    device: Device,
}
impl KyutaiMimiDecoder {
    /// The decoder of the codec's file `path` on `device`.
    ///
    /// The transformer's layers are placed on the cards the ladder sees when `device` is a
    /// card, on the host otherwise.
    pub fn from_safetensors(path: &str, device: Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_files(&[path], DType::F32, &device)? };
        let q = vb.pp("quantizer");
        let rvq_first = Rvq::load(&q.pp("rvq_first"), VQ_EPS)?;
        let rvq_rest = Rvq::load(&q.pp("rvq_rest"), VQ_EPS)?;
        let up_name = "upsample.convtr.convtr.convtr.weight";
        let upsample = vb.get(vb.shape_of(up_name)?, up_name)?;
        let cards = match device.is_cuda() {
            true => crate::inference::place::ladder::cards(0),
            false => Vec::new(),
        };
        let layers = transformer_layers(path, &vb, &cards)?;
        // SEANet decoder: `model.0` is the input conv, then each stage is an ELU, a
        // transposed conv and a residual block (three entries), and the output conv
        // follows the last stage's ELU.
        let dec = vb.pp("decoder.model");
        let conv_in = Conv::load(&dec, "0.conv.conv", true)?;
        let mut up = Vec::new();
        let mut idx = 2;
        while dec.contains(&format!("{idx}.convtr.convtr.weight")) {
            up.push((
                Conv::load_tr(&dec, &format!("{idx}.convtr.convtr"))?,
                ResBlock::load(&dec, idx + 1)?,
            ));
            idx += 3;
        }
        let conv_out = Conv::load(&dec, &format!("{idx}.conv.conv"), true)?;
        Ok(Self {
            rvq_first,
            rvq_rest,
            upsample,
            layers: std::sync::Mutex::new(layers),
            conv_in,
            up,
            conv_out,
            device,
        })
    }

    /// Audio samples one code frame decodes to: the product of every upsampling stride.
    pub fn frame_samples(&self) -> Result<usize> {
        let mut n = self.upsample.shape().dims3()?.2 / 2;
        for (convtr, _) in &self.up {
            n *= convtr.resampling_stride()?;
        }
        Ok(n)
    }

    /// RVQ dequant only (for parity vs mimi_latent): `codes [K,T]` -> `[dim,T]`.
    pub fn decode_latent(&self, codes: &Tensor) -> Result<Tensor> {
        let t = codes.shape().dims2()?.1;
        let n_first = self.rvq_first.len();
        let sem = self.rvq_first.decode(&codes.narrow(0, 0, n_first)?)?;
        let rest = self
            .rvq_rest
            .decode(&codes.narrow(0, n_first, self.rvq_rest.len())?)?;
        debug_assert_eq!(t, sem.shape().dims2()?.1);
        sem.add(&rest)
    }

    /// Full decode: `codes [K,T]` (u32) -> 24 kHz mono waveform `Vec<f32>`.
    pub fn decode(&self, codes: &Tensor) -> Result<Vec<f32>> {
        let rows = self.upsampled_rows(codes)?;
        let rows = self.transform(&rows)?;
        self.seanet(&rows)
    }

    /// `codes [K, T]` dequantised and upsampled to the transformer's rate, `[S, dim]`.
    fn upsampled_rows(&self, codes: &Tensor) -> Result<Tensor> {
        let latent = self.decode_latent(codes)?; // [dim,T]
        let (dim, t) = latent.shape().dims2()?;
        let emb = latent.reshape((1, dim, t))?;
        // upsample (depthwise convtr)
        let up_stride = self.upsample.shape().dims3()?.2 / 2;
        let x = causal_convtr1d(&emb, &self.upsample, None, up_stride, dim)?;
        let s = x.shape().dims3()?.2;
        x.reshape((dim, s))?.transpose(0, 1)?.contiguous()
    }

    /// `rows [S, dim]` at positions `0..S` through the transformer's layers, `[S, dim]` F32 on
    /// the decoder's device.
    fn transform(&self, rows: &Tensor) -> Result<Tensor> {
        let out = {
            let mut layers = self
                .layers
                .lock()
                .map_err(|_| Error::msg("Mimi's transformer is poisoned"))?;
            layers.forward_embeddings_rows(rows)?
        };
        let out = match out.dtype() {
            DType::F32 => out,
            _ => out.to_dtype(DType::F32)?,
        };
        out.to_device(&self.device)
    }

    /// `rows [S, dim]` through the SEANet decoder: the input conv, then per stage an ELU, a
    /// transposed conv and a residual block, then an ELU and the output conv.
    fn seanet(&self, rows: &Tensor) -> Result<Vec<f32>> {
        let (s, dim) = rows.shape().dims2()?;
        let mut y = rows.transpose(0, 1)?.reshape((1, dim, s))?.contiguous()?; // [1,dim,S]
        y = causal_conv1d(&y, &self.conv_in.w, self.conv_in.b.as_ref(), 1, 1, 1)?;
        for (convtr, res) in &self.up {
            y = elu(&y)?;
            let stride = convtr.resampling_stride()?;
            y = causal_convtr1d(&y, &convtr.w, convtr.b.as_ref(), stride, 1)?;
            y = res.forward(&y)?;
        }
        y = elu(&y)?;
        y = causal_conv1d(&y, &self.conv_out.w, self.conv_out.b.as_ref(), 1, 1, 1)?;
        y.flatten_all()?.to_vec1_f32()
    }
}

impl KyutaiMimiDecoder {
    /// Where the decoder sits: 1 block on its device, the transformer's layers aside.
    pub fn placement(&self) -> Vec<crate::inference::serve::progress::placement::Placed> {
        crate::inference::serve::progress::placement::whole(&self.device, 1)
    }

    /// Where the transformer's layers sit, by device.
    pub fn transformer_placement(
        &self,
    ) -> Vec<crate::inference::serve::progress::placement::Placed> {
        match self.layers.lock() {
            Ok(layers) => layers.placed_runs(),
            Err(_) => Vec::new(),
        }
    }
}

/// The configuration of the decoder-transformer under `vb`: the shapes its tensors have and
/// the settings Mimi's reference fixes.
fn transformer_config(vb: &VarBuilder) -> Result<serde_json::Value> {
    let xt = vb.pp(TRANSFORMER_PREFIX);
    let first = xt.pp("layers.0");
    Ok(serde_json::json!({
        "architectures": [TRANSFORMER_CLASS],
        "hidden_size": first.shape_of("norm1.weight")?[0],
        "num_attention_heads": ATTENTION_HEADS,
        "num_key_value_heads": ATTENTION_HEADS,
        "intermediate_size": first.shape_of("linear1.weight")?[0],
        "num_hidden_layers": xt.count_children("layers"),
        "sliding_window": ATTENTION_WINDOW,
        "rope_theta": ROPE_THETA,
        "layer_norm_eps": LN_EPS,
    }))
}

/// The decoder-transformer's layers in the codec's file `path`, placed on `cards`.
///
/// The cache holds the rows a layer windowed in every layer keeps.
/// A decode runs every row from position 0, whatever its length.
fn transformer_layers(
    path: &str,
    vb: &VarBuilder,
    cards: &[(usize, u64)],
) -> Result<GenericHeteroTransformer> {
    let root = transformer_config(vb)?;
    let shards = [path.to_string()];
    let held = class_load::class_demand(&root, &shards, TRANSFORMER_CLASS, 0)?
        .held_rows
        .ok_or_else(|| Error::msg("Mimi's transformer is windowed in every layer"))?;
    let loaded = class_load::load_class_from(
        &root,
        &shards,
        TRANSFORMER_CLASS,
        Rows::Exactly(held as usize),
        0,
        cards,
    )?;
    Ok(loaded.model)
}

#[cfg(test)]
mod tests {
    //! The decoder-transformer on the generic layers against its own layer code, on a toy and
    //! on the published checkpoint, on the host and on a card.

    use super::*;
    use crate::inference::generic_transformer::class_load::tests::{scratch, values};

    /// The rows of the toy: beyond the window and a prefill chunk.
    fn toy_rows() -> usize {
        crate::inference::engine::llm_engine::PREFILL_CHUNK_TOKENS + 2 * ATTENTION_WINDOW + 3
    }

    /// A codec's file holding a decoder-transformer of `layers` layers `hidden` wide, its
    /// feed-forward `ffn` wide.
    fn toy_file(tag: &str, hidden: usize, ffn: usize, layers: usize) -> std::path::PathBuf {
        let dir = scratch(tag);
        let mut seed = 23u32;
        let mut stored = Vec::new();
        for il in 0..layers {
            for (part, dims, centre, scale) in [
                ("norm1.weight", vec![hidden], 1.0, 0.2),
                ("norm1.bias", vec![hidden], 0.0, 0.2),
                ("norm2.weight", vec![hidden], 1.0, 0.2),
                ("norm2.bias", vec![hidden], 0.0, 0.2),
                (
                    "self_attn.in_proj_weight",
                    vec![3 * hidden, hidden],
                    0.0,
                    0.2,
                ),
                ("self_attn.out_proj.weight", vec![hidden, hidden], 0.0, 0.2),
                ("linear1.weight", vec![ffn, hidden], 0.0, 0.2),
                ("linear2.weight", vec![hidden, ffn], 0.0, 0.2),
                ("layer_scale_1.scale", vec![hidden], 0.5, 0.4),
                ("layer_scale_2.scale", vec![hidden], 0.5, 0.4),
            ] {
                let v = values(&mut seed, dims.iter().product(), centre, scale);
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                stored.push((
                    format!("{TRANSFORMER_PREFIX}.layers.{il}.{part}"),
                    dims,
                    bytes,
                ));
            }
        }
        let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = stored
            .iter()
            .map(|(name, dims, bytes)| {
                let view = safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    dims.clone(),
                    bytes,
                )
                .unwrap();
                (name.clone(), view)
            })
            .collect();
        let file = dir.join("model.safetensors");
        safetensors::serialize_to_file(views, None, &file).unwrap();
        file
    }

    /// The decoder-transformer's own layers in the file `path`, on the host.
    fn own_layers(path: &str) -> Vec<XfLayer> {
        let vb = unsafe { VarBuilder::from_files(&[path], DType::F32, &Device::Cpu).unwrap() };
        let xt = vb.pp(TRANSFORMER_PREFIX);
        (0..xt.count_children("layers"))
            .map(|i| XfLayer::load(&xt, i).unwrap())
            .collect()
    }

    /// `rows [S, dim]` through the own layers on the host.
    fn own_run(layers: &[XfLayer], rows: &Tensor) -> Tensor {
        let mut h = rows.to_device(&Device::Cpu).unwrap();
        for l in layers {
            h = l.forward(&h, &Device::Cpu).unwrap();
        }
        h
    }

    /// The generic layers in the file `path`, placed on `cards`.
    fn generic_layers(path: &str, cards: &[(usize, u64)]) -> GenericHeteroTransformer {
        let vb = unsafe { VarBuilder::from_files(&[path], DType::F32, &Device::Cpu).unwrap() };
        transformer_layers(path, &vb, cards).unwrap()
    }

    /// `t` as F32 values on the host.
    fn host(t: &Tensor) -> Vec<f32> {
        t.to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// The largest difference of `got` from `want`, and the largest magnitude of `want`.
    fn error(got: &[f32], want: &[f32]) -> (f32, f32) {
        assert_eq!(got.len(), want.len());
        assert!(got.iter().all(|v| v.is_finite()));
        let err = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        (err, want.iter().map(|v| v.abs()).fold(0f32, f32::max))
    }

    /// The tolerance on a value of magnitude `scale` out of `layers` layers whose widest product
    /// sums `width` terms, in a float type of machine epsilon `eps`, over `rows` positions.
    ///
    /// A product of `n` terms rounds to about `eps * sqrt(n)` of its scale.
    /// A layer chains 4 products: the Q, K and V projection, the attention output, the up and
    /// the down projections.
    /// A rotary angle of position `p` rounds to about `eps * p` of a radian in F32, which each
    /// layer adds once.
    fn tolerance(layers: usize, width: usize, rows: usize, eps: f32, scale: f32) -> f32 {
        let products = eps * (width as f32).sqrt() * (4 * layers + 1) as f32;
        let angles = f32::EPSILON * rows as f32 * layers as f32;
        (products + angles) * scale
    }

    /// The width of the widest product of `own`'s layers over rows `dim` wide.
    fn widest(own: &[XfLayer], dim: usize) -> usize {
        own[0].linear1.dims()[0].max(3 * dim)
    }

    /// The generic layers against the own layers over `rows`, within the tolerance of `eps`.
    fn layers_agree(path: &str, rows: &Tensor, cards: &[(usize, u64)], eps: f32) {
        let own = own_layers(path);
        let want = host(&own_run(&own, rows));
        let mut model = generic_layers(path, cards);
        let got = host(&model.forward_embeddings_rows(rows).unwrap());
        let (s, dim) = rows.dims2().unwrap();
        let (err, scale) = error(&got, &want);
        let tol = tolerance(own.len(), widest(&own, dim), s, eps, scale);
        eprintln!("{s} rows: {err} away at a scale of {scale}, tolerance {tol}");
        assert!(err <= tol, "{err} away, tolerance {tol}");
    }

    /// The toy's rows, `hidden` wide.
    fn toy_input(hidden: usize) -> Tensor {
        let mut seed = 5u32;
        let n = toy_rows();
        Tensor::from_vec(
            values(&mut seed, n * hidden, 0.0, 1.0),
            (n, hidden),
            &Device::Cpu,
        )
        .unwrap()
    }

    /// The toy on the generic layers on the host gives the rows of its own layers.
    #[test]
    fn the_toy_transformer_on_generic_layers_matches_its_own_layers() {
        let (hidden, ffn) = (8 * ATTENTION_HEADS, 16 * ATTENTION_HEADS);
        let file = toy_file("mimi-toy", hidden, ffn, 2);
        let path = file.to_string_lossy().to_string();
        layers_agree(&path, &toy_input(hidden), &[], f32::EPSILON);
        let _ = std::fs::remove_dir_all(file.parent().unwrap());
    }

    /// The same toy with the layers on the cards, within the rounding of F32.
    #[cfg(feature = "gpu")]
    #[test]
    fn the_card_toy_transformer_on_generic_layers_matches_its_own_layers() {
        let cards = crate::inference::place::ladder::cards(0);
        if cards.is_empty() {
            eprintln!("no card");
            return;
        }
        let (hidden, ffn) = (8 * ATTENTION_HEADS, 16 * ATTENTION_HEADS);
        let file = toy_file("mimi-toy-card", hidden, ffn, 2);
        let path = file.to_string_lossy().to_string();
        assert!(!generic_layers(&path, &cards)
            .layer_cuda_ordinals()
            .is_empty());
        layers_agree(&path, &toy_input(hidden), &cards, f32::EPSILON);
        let _ = std::fs::remove_dir_all(file.parent().unwrap());
    }

    /// The published codec's file, `None` when the test store lacks it.
    fn published() -> Option<String> {
        let file = crate::inference::cache::hf::snapshot("models--kyutai--tts-1.6b-en_fr")
            .map(|d| d.join("tokenizer-e351c8d8-checkpoint125.safetensors"))
            .filter(|f| f.exists());
        if file.is_none() {
            eprintln!("the published checkpoint is not in the test store");
        }
        file.map(|f| f.to_string_lossy().to_string())
    }

    /// A fixed code list of `frames` frames for the decoder `dec`.
    fn codes(dec: &KyutaiMimiDecoder, frames: usize) -> Tensor {
        let books = dec.rvq_first.len() + dec.rvq_rest.len();
        let mut seed = 7u32;
        let ids: Vec<u32> = (0..books * frames)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 8) % 2048
            })
            .collect();
        Tensor::from_vec_u32(ids, vec![books, frames]).unwrap()
    }

    /// The published transformer on the generic layers of a decoder on `device` gives the rows
    /// of its own layers.
    /// The waveform of a fixed code list stays within the same share of its peak.
    fn published_agrees(device: Device, eps: f32) {
        let Some(path) = published() else { return };
        let dec = KyutaiMimiDecoder::from_safetensors(&path, device).unwrap();
        // Frames whose rows pass the window and a prefill chunk.
        let frames = toy_rows().div_ceil(2);
        let codes = codes(&dec, frames).to_device(&dec.device).unwrap();
        let rows = dec.upsampled_rows(&codes).unwrap();
        let own = own_layers(&path);
        let own_rows = own_run(&own, &rows).to_device(&dec.device).unwrap();
        let (s, dim) = rows.dims2().unwrap();
        let new_rows = dec.transform(&rows).unwrap();
        let (err, scale) = error(&host(&new_rows), &host(&own_rows));
        let tol = tolerance(own.len(), widest(&own, dim), s, eps, scale);
        eprintln!("{s} rows: {err} away at a scale of {scale}, tolerance {tol}");
        assert!(err <= tol, "{err} away, tolerance {tol}");
        let own_wave = dec.seanet(&own_rows).unwrap();
        let new_wave = dec.seanet(&new_rows).unwrap();
        let (werr, peak) = error(&new_wave, &own_wave);
        let same = own_wave
            .iter()
            .zip(&new_wave)
            .filter(|(a, b)| a == b)
            .count();
        eprintln!(
            "waveform of {} samples: {werr} away at a peak of {peak}, {same} samples equal",
            own_wave.len()
        );
        assert!(werr <= tol / scale * peak, "waveform {werr} away");
    }

    /// The published transformer's own demand places every layer on a card that holds it,
    /// layers on the host beside a card smaller than its blocks, and every layer on the host
    /// with no card.
    #[test]
    fn the_published_transformer_places_by_its_own_demand() {
        use crate::inference::place::layer_executor::DeviceKind;
        let Some(path) = published() else { return };
        let vb = unsafe { VarBuilder::from_files(&[&path], DType::F32, &Device::Cpu).unwrap() };
        let root = transformer_config(&vb).unwrap();
        let demand = class_load::class_demand(&root, &[path], TRANSFORMER_CLASS, 0).unwrap();
        let held = demand.held_rows.unwrap();
        let at = demand.at(held);
        eprintln!("{at:?}");
        let mut host_counts = Vec::new();
        for (name, cards) in [
            ("large card", vec![(0usize, 4 * at.whole())]),
            ("small card", vec![(0usize, at.block_bytes / 2)]),
            ("host", Vec::new()),
        ] {
            let plan = at.plan(&cards);
            let runs: Vec<String> = plan
                .segments
                .iter()
                .map(|s| format!("{:?} {}", s.kind, s.num_layers()))
                .collect();
            let host: usize = plan
                .segments
                .iter()
                .filter(|s| !matches!(s.kind, DeviceKind::Cuda(_)))
                .map(|s| s.num_layers())
                .sum();
            eprintln!("{name}: {}", runs.join(", "));
            host_counts.push(host);
        }
        assert_eq!(host_counts[0], 0);
        assert!(host_counts[1] > 0);
        assert_eq!(host_counts[2], demand.layers);
    }

    /// On the host.
    #[test]
    fn the_published_transformer_on_generic_layers_matches_its_own_layers() {
        published_agrees(Device::Cpu, f32::EPSILON);
    }

    /// With the layers on the cards.
    #[cfg(feature = "gpu")]
    #[test]
    fn the_card_published_transformer_on_generic_layers_matches_its_own_layers() {
        if crate::inference::place::ladder::cards(0).is_empty() {
            eprintln!("no card");
            return;
        }
        published_agrees(Device::new_cuda(0).unwrap(), f32::EPSILON);
    }
}
