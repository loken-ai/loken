//! Moondream vision tower compatible with Ollama's CLIP projector blob.
//!
//! Ollama packages moondream as two GGUF files: a phi2 language model and a
//! separate CLIP-style vision encoder + LLaVA-style projector. This module
//! loads the latter and runs it to produce per-patch embeddings that get
//! prepended (between BOS and text tokens) into the phi2 prefill via
//! `GenericHeteroTransformer::forward_with_image_embeds`.
//!
//! Weight layout (verified against the Ollama moondream:latest blob):
//!   - `v.patch_embd.{weight,bias}` - Conv2d(3, 1152, k=14, s=14)
//!   - `v.position_embd.weight` - (1, 729, 1152) F16
//!   - `v.blk.{0..27}.attn_{q,k,v,out}.{weight,bias}` - separate Q/K/V
//!   - `v.blk.{0..27}.ffn_{up,down}.{weight,bias}` - GELU between
//!   - `v.blk.{0..27}.ln{1,2}.{weight,bias}` - LayerNorm(1152, eps=1e-6)
//!   - `v.post_ln.{weight,bias}` - LayerNorm after the last block
//!   - `mm.0.{weight,bias}` - projector fc1 (1152 -> 8192) + GELU
//!   - `mm.2.{weight,bias}` - projector fc2 (8192 -> 2048)
//!
//! The generic quantized-Moondream path is
//! NOT compatible - it expects a fused `attn.qkv` weight, `linear(588,1152)`
//! flat patch embed, and a 2-layer projector with `cfg.hidden_dim=2048`
//! instead of Ollama's 8192. Forcing a name-remap would also have to
//! repack tensors, so a dedicated module is simpler.
use std::path::Path;

use crate::inference::model::vit::{Qkv, VitBlock};
use crate::tensor::layer::qlinear::{QLinear, QMlp, Weight};
use crate::tensor::layer::{Conv2d, Conv2dConfig};
use crate::tensor::layer::{LayerNorm, Linear};
use crate::tensor::ops::Activation;
use crate::tensor::quantized::gguf_file;
use crate::tensor::{DType, Device, Tensor};
use anyhow::{anyhow, Result as AnyResult};

/// Architectural constants for Ollama's Moondream CLIP. Hard-coded because
/// the blob's GGUF metadata records the same values and there's only one
/// supported shape today; we'll generalise if a second variant ships.
const IMAGE_SIZE: usize = 378;
const PATCH_SIZE: usize = 14;
const EMBED_DIM: usize = 1152;
const FFN_DIM: usize = 4304;
const NUM_HEADS: usize = 16;
const NUM_BLOCKS: usize = 28;
const LN_EPS: f32 = 1e-6;
const PROJ_OUT: usize = 2048;
const NUM_PATCHES: usize = (IMAGE_SIZE / PATCH_SIZE) * (IMAGE_SIZE / PATCH_SIZE); // 27 * 27 = 729

/// Full Moondream vision encoder: ViT (patch embed -> 28 blocks -> post-norm)
/// followed by the LLaVA-style 2-layer MLP projector that maps from the
/// vision width (1152) to the phi2 hidden width (2048).
pub struct MoondreamVisionEncoder {
    patch_embed: Conv2d,
    pos_embed: Tensor, // (1, 729, 1152)
    blocks: Vec<VitBlock>,
    post_ln: LayerNorm,
    proj_fc1: Linear,
    proj_fc2: Linear,
    device: Device,
}

impl MoondreamVisionEncoder {
    /// Load + parse the Ollama CLIP projector GGUF blob on `device`.
    pub fn from_clip_gguf(blob_path: &Path, device: &Device) -> AnyResult<Self> {
        let content = gguf_file::open_mapped(blob_path)
            .map_err(|e| anyhow!("parse CLIP GGUF {}: {}", blob_path.display(), e))?;

        // Sanity: confirm this is a clip projector with vision encoder and
        // matches the shapes we hard-coded. If the metadata disagrees we
        // bail loudly rather than producing garbage embeddings.
        check_metadata(&content)?;

        // Helper: load a single tensor from the GGUF, cast to F16 on the
        // target device. The CUDA-fast `dequantize_f16` only handles
        // truly quantized inputs (Q4_*, Q8_*, etc) and bails for F16 /
        // F32 storage, so we go through plain `dequantize` (which
        // produces F32) and cast. To keep the cudaMallocAsync pool from
        // accumulating 457 transient F32 buffers (~10 GB) we sync the
        // device after each load - cudaFreeAsync is stream-ordered and
        // only "actually frees" once the stream catches up.
        // Vision weights cast to F16 after the reference QTensor::dequantize
        // (which always returns F32). F16 was experimentally OK: produces
        // semantically-correct captions with degraded grammar; an F32
        // attempt OOMed even on a dedicated GPU because per-block
        // attention scores [1, 16, 729, 729] F32 ≈ 34 MB and the
        // cudaMallocAsync pool fragments badly across 28 ViT blocks.
        // Per-load sync caps the transient F32-then-F16 buffers.
        let cuda_dev = if device.is_cuda() {
            Some(device.clone())
        } else {
            None
        };
        let load = |name: &str| -> AnyResult<Tensor> {
            let qt = content
                .tensor(name, device)
                .map_err(|e| anyhow!("get {}: {}", name, e))?;
            let t = qt
                .dequantize(device)
                .map_err(|e| anyhow!("dequant {}: {}", name, e))?;
            // Keep F32 for the ViT. llama.cpp's clip computes the
            // vision tower in F32; running 28 blocks in F16 accumulates enough
            // error that moondream - which is extremely prompt/embedding
            // sensitive - mis-captions (gets the scene, misses the subject).
            // The earlier F32-OOM is now avoided by the per-block detach+
            // synchronize below (reclaims the [1,16,729,729] scores each block),
            // and the tower is small (~1.8 GB F32) on the mostly-idle GPU1.
            let t = if t.dtype() == DType::F32 {
                t
            } else {
                t.to_dtype(DType::F32)
                    .map_err(|e| anyhow!("cast {} to F32: {}", name, e))?
            };
            if let Some(ref d) = cuda_dev {
                d.synchronize()
                    .map_err(|e| anyhow!("sync after {}: {}", name, e))?;
            }
            Ok(t)
        };

        // -- Patch embedding (Conv2d 3->1152, k=14, s=14) ---------------
        // GGUF stores conv weight as [out, in, kH, kW] which matches
        // the reference Conv2d expectation directly.
        let patch_w = load("v.patch_embd.weight")?;
        let patch_b = load("v.patch_embd.bias")?;
        let patch_embed = Conv2d::new(
            patch_w,
            Some(patch_b),
            Conv2dConfig {
                stride: PATCH_SIZE,
                ..Default::default()
            },
        );

        // -- Position embedding (1, 729, 1152) -------------------------
        let pos_embed = load("v.position_embd.weight")?;

        // -- 28 ViT blocks ---------------------------------------------
        let mut blocks = Vec::with_capacity(NUM_BLOCKS);
        for i in 0..NUM_BLOCKS {
            let p = format!("v.blk.{i}");
            let ln1 = LayerNorm::new(
                load(&format!("{p}.ln1.weight"))?,
                Some(load(&format!("{p}.ln1.bias"))?),
                LN_EPS,
            );
            let ln2 = LayerNorm::new(
                load(&format!("{p}.ln2.weight"))?,
                Some(load(&format!("{p}.ln2.bias"))?),
                LN_EPS,
            );
            // The blob stores the three projections apart, and names the FFN's expansion
            // `ffn_down` and its contraction `ffn_up` - the reverse of the convention the
            // language models use. What the layer does is what decides which is which:
            // `ffn_down` widens 1152 to 4304, so it is the first of the two.
            let dense = |w: Tensor, b: Tensor| -> AnyResult<QLinear> {
                let (out_dim, in_dim) = (w.dim(0)?, w.dim(1)?);
                Ok(QLinear::new(
                    Weight::Dense(Linear::new(w, None)?, DType::F32),
                    Some(b),
                    in_dim,
                    out_dim,
                ))
            };
            let at = |n: &str| -> AnyResult<QLinear> {
                dense(
                    load(&format!("{p}.{n}.weight"))?,
                    load(&format!("{p}.{n}.bias"))?,
                )
            };
            blocks.push(VitBlock::new(
                ln1,
                Qkv::Split {
                    q: at("attn_q")?,
                    k: at("attn_k")?,
                    v: at("attn_v")?,
                },
                at("attn_out")?,
                ln2,
                QMlp::new(
                    at("ffn_down")?,
                    Activation::GeluPytorchTanh,
                    at("ffn_up")?,
                ),
                EMBED_DIM,
                NUM_HEADS,
            )?);
        }

        // -- Post-ViT norm ---------------------------------------------
        let post_ln = LayerNorm::new(
            load("v.post_ln.weight")?,
            Some(load("v.post_ln.bias")?),
            LN_EPS,
        );

        // -- Projector (1152 -> 8192 -> 2048) ---------------------------
        let proj_fc1 = Linear::new(load("mm.0.weight")?, Some(load("mm.0.bias")?))?;
        let proj_fc2 = Linear::new(load("mm.2.weight")?, Some(load("mm.2.bias")?))?;

        Ok(Self {
            patch_embed,
            pos_embed,
            blocks,
            post_ln,
            proj_fc1,
            proj_fc2,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Run vision encoder + projector on a preprocessed image tensor.
    /// Input: `(1, 3, 378, 378)` F32, normalised to mean=0.5 std=0.5
    /// (matches `crate::inference::media::image_processor::preprocess_moondream`).
    /// Output: `(1, 729, 2048)` F32 - embeddings ready to splice between
    /// BOS and text tokens in the phi2 prefill.
    pub fn forward(&self, image: &Tensor) -> crate::tensor::Result<Tensor> {
        // Ensure input is on the encoder's device and dtype matches the
        // patch-embed conv weight. CLIP weights are F16 in the GGUF; we
        // dequantized to whatever the device default is - usually F32 on
        // CPU and F32 on CUDA (the reference QTensor::dequantize keeps F16 as
        // F16). Match patch_embed's weight dtype to avoid GEMM errors.
        let target_dtype = self.patch_embed.weight().dtype();
        let image = if image.device().same_device(&self.device) {
            image.clone()
        } else {
            image.to_device(&self.device)?
        };
        let image = if image.dtype() != target_dtype {
            image.to_dtype(target_dtype)?
        } else {
            image
        };
        // (B, 3, 378, 378) -> (B, 1152, 27, 27) via stride-14 conv
        let h = self.patch_embed.forward(&image)?;
        let (b, c, hh, ww) = h.dims4()?;
        debug_assert_eq!(c, EMBED_DIM);
        debug_assert_eq!(hh * ww, NUM_PATCHES);
        // (B, 1152, 27, 27) -> (B, 27*27, 1152)
        let h = h.flatten_from(2)?.transpose(1, 2)?.contiguous()?;
        // Add position embedding, broadcasting batch dim.
        let pos = if self.pos_embed.dtype() != h.dtype() {
            self.pos_embed.to_dtype(h.dtype())?
        } else {
            self.pos_embed.clone()
        };
        let h = h.broadcast_add(&pos)?;
        let mut h = h;
        for block in &self.blocks {
            h = block.forward(&h)?;
            // Break the tensor-op Op backprop chain. Without this, every
            // intermediate (Q, K, V projections, attention scores,
            // softmax probs, FFN expansion) stays alive via Op refs all
            // the way to the final return value - 28 blocks x ~85 MB
            // working set blows the entire GPU pool. detach() returns
            // a fresh leaf tensor with the same storage, dropping the
            // op chain so the previous block's allocs become reclaimable.
            h = h.detach();
            // Synchronize so the cudaMallocAsync pool actually reclaims
            // the freed slots (stream-ordered cudaFreeAsync only
            // completes after the stream flushes). Without this the pool
            // grows to multiple GB across 28 blocks even though only
            // ~85 MB working set is live at any one moment.
            if self.device.is_cuda() {
                self.device.synchronize()?;
            }
        }
        let h = self.post_ln.forward(&h)?;
        // Projector - same gelu_pytorch_tanh activation as the ViT.
        let h = self.proj_fc1.forward(&h)?;
        let h = h.gelu()?;
        let h = self.proj_fc2.forward(&h)?;
        let h = h.detach();
        // phi2 expects F32 in its embedding cat (text_emb is F32). Cast
        // the projector output if it isn't already.
        let h = if h.dtype() != DType::F32 {
            h.to_dtype(DType::F32)?
        } else {
            h
        };
        debug_assert_eq!(h.dim(1)?, NUM_PATCHES);
        debug_assert_eq!(h.dim(2)?, PROJ_OUT);
        let _ = b;
        Ok(h)
    }
}

fn check_metadata(content: &gguf_file::Content) -> AnyResult<()> {
    let get_u32 = |k: &str| -> Option<u32> {
        content.metadata.get(k).and_then(|v| match v {
            gguf_file::Value::U32(x) => Some(*x),
            gguf_file::Value::I32(x) => Some(*x as u32),
            _ => None,
        })
    };
    let get_str = |k: &str| -> Option<String> {
        content.metadata.get(k).and_then(|v| match v {
            gguf_file::Value::String(s) => Some(s.clone()),
            _ => None,
        })
    };
    let arch = get_str("general.architecture").unwrap_or_default();
    if arch != "clip" {
        return Err(anyhow!(
            "CLIP blob has unexpected architecture '{arch}', want 'clip'"
        ));
    }
    if let Some(img) = get_u32("clip.vision.image_size") {
        if img as usize != IMAGE_SIZE {
            return Err(anyhow!("clip.vision.image_size {} != {}", img, IMAGE_SIZE));
        }
    }
    if let Some(p) = get_u32("clip.vision.patch_size") {
        if p as usize != PATCH_SIZE {
            return Err(anyhow!("clip.vision.patch_size {} != {}", p, PATCH_SIZE));
        }
    }
    if let Some(e) = get_u32("clip.vision.embedding_length") {
        if e as usize != EMBED_DIM {
            return Err(anyhow!(
                "clip.vision.embedding_length {} != {}",
                e,
                EMBED_DIM
            ));
        }
    }
    if let Some(f) = get_u32("clip.vision.feed_forward_length") {
        if f as usize != FFN_DIM {
            return Err(anyhow!(
                "clip.vision.feed_forward_length {} != {}",
                f,
                FFN_DIM
            ));
        }
    }
    if let Some(h) = get_u32("clip.vision.attention.head_count") {
        if h as usize != NUM_HEADS {
            return Err(anyhow!(
                "clip.vision.attention.head_count {} != {}",
                h,
                NUM_HEADS
            ));
        }
    }
    if let Some(n) = get_u32("clip.vision.block_count") {
        if n as usize != NUM_BLOCKS {
            return Err(anyhow!("clip.vision.block_count {} != {}", n, NUM_BLOCKS));
        }
    }
    if let Some(p) = get_u32("clip.vision.projection_dim") {
        if p as usize != PROJ_OUT {
            return Err(anyhow!("clip.vision.projection_dim {} != {}", p, PROJ_OUT));
        }
    }
    Ok(())
}
