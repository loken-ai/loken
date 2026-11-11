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
use crate::inference::model::codec_blocks::{causal_conv1d, causal_convtr1d, Rvq};
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
    //! The decoder of the published checkpoint, on the host and on a card.
    //!
    //! `generic_transformer::mimi_class_tests` holds the transformer's layers to a plain host
    //! run in F64.

    use super::*;

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

    /// The published decoder on `device` gives the samples of every frame of a fixed code list,
    /// finite, and the same samples on a second decode.
    fn published_decodes(device: Device) {
        let Some(path) = published() else { return };
        let dec = KyutaiMimiDecoder::from_safetensors(&path, device).unwrap();
        // Frames whose rows pass the window and a prefill chunk.
        let rows =
            crate::inference::engine::llm_engine::PREFILL_CHUNK_TOKENS + 2 * ATTENTION_WINDOW + 3;
        let frames = rows.div_ceil(2);
        let codes = codes(&dec, frames).to_device(&dec.device).unwrap();
        let wave = dec.decode(&codes).unwrap();
        assert_eq!(wave.len(), frames * dec.frame_samples().unwrap());
        assert!(wave.iter().all(|v| v.is_finite()));
        assert_eq!(dec.decode(&codes).unwrap(), wave);
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
    fn the_published_decoder_gives_the_samples_of_its_frames() {
        published_decodes(Device::Cpu);
    }

    /// With the decoder and the transformer's layers on the cards.
    #[cfg(feature = "gpu")]
    #[test]
    fn the_card_published_decoder_gives_the_samples_of_its_frames() {
        if crate::inference::place::ladder::cards(0).is_empty() {
            eprintln!("no card");
            return;
        }
        published_decodes(Device::new_cuda(0).unwrap());
    }
}
