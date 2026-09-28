//! Moondream vision tower compatible with Ollama's CLIP projector blob.
//!
//! Ollama packages moondream as two GGUF files: a phi2 language model and a
//! separate CLIP-style vision encoder + LLaVA-style projector. This module
//! loads the latter and runs it to produce per-patch embeddings that get
//! prepended (between BOS and text tokens) into the phi2 prefill via
//! `GenericHeteroTransformer::forward_with_image_embeds`.
//!
//! Weight layout (verified against the Ollama moondream:latest blob), with every width,
//! count and epsilon read from the blob's `clip.vision.*` metadata:
//!   - `v.patch_embd.{weight,bias}` - Conv2d(3, width, k=patch, s=patch)
//!   - `v.position_embd.weight` - (1, grid*grid, width) F16
//!   - `v.blk.{i}.attn_{q,k,v,out}.{weight,bias}` - separate Q/K/V
//!   - `v.blk.{i}.ffn_{up,down}.{weight,bias}` - GELU between
//!   - `v.blk.{i}.ln{1,2}.{weight,bias}` - LayerNorm(width)
//!   - `v.post_ln.{weight,bias}` - LayerNorm after the last block
//!   - `mm.0.{weight,bias}` - projector fc1 + GELU
//!   - `mm.2.{weight,bias}` - projector fc2 to the language model's width
//!
//! The generic quantized-Moondream path is
//! NOT compatible - it expects a fused `attn.qkv` weight, `linear(588,1152)`
//! flat patch embed, and a 2-layer projector with `cfg.hidden_dim=2048`
//! instead of Ollama's 8192. Forcing a name-remap would also have to
//! repack tensors, so a dedicated module is simpler.
use std::path::Path;

use crate::inference::model::mmproj::MmprojVision;
use crate::inference::model::vit::{Qkv, VitBlock};
use crate::tensor::layer::qlinear::{QLinear, QMlp, Weight};
use crate::tensor::layer::{Conv2d, Conv2dConfig};
use crate::tensor::layer::{LayerNorm, Linear};
use crate::tensor::ops::Activation;
use crate::tensor::quantized::gguf_file;
use crate::tensor::{DType, Device, Tensor};
use anyhow::{anyhow, Result as AnyResult};

/// Full Moondream vision encoder: ViT (patch embed -> blocks -> post-norm)
/// followed by the LLaVA-style 2-layer MLP projector that maps from the
/// vision width to the phi2 hidden width.
pub struct MoondreamVisionEncoder {
    geo: MmprojVision,
    patch_embed: Conv2d,
    pos_embed: Tensor, // (1, grid*grid, width)
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

        // Confirm this is a clip projector, and read the tower's shape from it.
        check_architecture(&content)?;
        let geo =
            MmprojVision::read(&content).map_err(|e| anyhow!("{}: {e}", blob_path.display()))?;

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

        // -- Patch embedding (Conv2d 3->width, k=patch, s=patch) ---------
        // GGUF stores conv weight as [out, in, kH, kW] which matches
        // the reference Conv2d expectation directly.
        let patch_w = load("v.patch_embd.weight")?;
        let patch_b = load("v.patch_embd.bias")?;
        let patch_embed = Conv2d::new(
            patch_w,
            Some(patch_b),
            Conv2dConfig {
                stride: geo.patch,
                ..Default::default()
            },
        );

        // -- Position embedding (1, grid*grid, width) -------------------
        let pos_embed = load("v.position_embd.weight")?;

        // -- ViT blocks -------------------------------------------------
        let mut blocks = Vec::with_capacity(geo.blocks);
        for i in 0..geo.blocks {
            let p = format!("v.blk.{i}");
            let ln1 = LayerNorm::new(
                load(&format!("{p}.ln1.weight"))?,
                Some(load(&format!("{p}.ln1.bias"))?),
                geo.eps,
            );
            let ln2 = LayerNorm::new(
                load(&format!("{p}.ln2.weight"))?,
                Some(load(&format!("{p}.ln2.bias"))?),
                geo.eps,
            );
            // The blob stores the three projections apart, and names the FFN's expansion
            // `ffn_down` and its contraction `ffn_up` - the reverse of the convention the
            // language models use. What the layer does is what decides which is which:
            // `ffn_down` widens the model width to the feed-forward width, so it is the first
            // of the two.
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
                QMlp::new(at("ffn_down")?, Activation::GeluPytorchTanh, at("ffn_up")?),
                geo.width,
                geo.heads,
            )?);
        }

        // -- Post-ViT norm ---------------------------------------------
        let post_ln = LayerNorm::new(
            load("v.post_ln.weight")?,
            Some(load("v.post_ln.bias")?),
            geo.eps,
        );

        // -- Projector (width -> hidden -> projection) -------------------
        let proj_fc1 = Linear::new(load("mm.0.weight")?, Some(load("mm.0.bias")?))?;
        let proj_fc2 = Linear::new(load("mm.2.weight")?, Some(load("mm.2.bias")?))?;

        Ok(Self {
            geo,
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
    /// Input: `(1, 3, image_size, image_size)` F32, normalised with the blob's statistics
    /// (matches `crate::inference::media::image_processor::preprocess_moondream`).
    /// Output: `(1, grid*grid, projection)` F32 - embeddings ready to splice between
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
        // (B, 3, S, S) -> (B, width, grid, grid) via the strided patch conv
        let h = self.patch_embed.forward(&image)?;
        let (b, c, hh, ww) = h.dims4()?;
        debug_assert_eq!(c, self.geo.width);
        debug_assert_eq!(hh * ww, self.geo.grid() * self.geo.grid());
        // (B, width, grid, grid) -> (B, grid*grid, width)
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
        debug_assert_eq!(h.dim(1)?, self.geo.grid() * self.geo.grid());
        debug_assert_eq!(h.dim(2)?, self.geo.projection);
        let _ = b;
        Ok(h)
    }
}

fn check_architecture(content: &gguf_file::Content) -> AnyResult<()> {
    let arch = content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .cloned()
        .unwrap_or_default();
    if arch != "clip" {
        return Err(anyhow!(
            "CLIP blob has unexpected architecture '{arch}', want 'clip'"
        ));
    }
    Ok(())
}
