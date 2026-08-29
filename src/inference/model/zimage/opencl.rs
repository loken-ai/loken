//! OpenCL Z-Image transformer block - runs on Intel Arc via custom OpenCL kernels.
//!
//! Loads F32 weight data from safetensors directly into OpenCL buffers (bypassing the substrate),
//! then runs the full ZImageTransformerBlock forward pass using OpenCL kernels:
//!   f32_matmul, rms_norm, image_attention, apply_rotary_emb, silu_mul, scale_after_norm,
//!   gated_residual, broadcast_mul, add.
//!
//! Weight layout: All weights stored as row-major F32 in OpenCL buffers.
//! BF16 safetensors are converted to F32 on load (OpenCL hardware may not support BF16 natively).

#![allow(clippy::too_many_arguments)]

use anyhow::Result;
use opencl3::command_queue::CL_BLOCKING;
use opencl3::kernel::ExecuteKernel;
use opencl3::memory::{Buffer, ClMem, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use std::ptr::null_mut;
use tracing::info;

use crate::inference::kernel::opencl::OpenCLPipelines;

/// Find a tensor by name across multiple safetensors shards.
/// Returns the TensorView from whichever shard contains it.
fn find_tensor<'a>(
    shards: &[safetensors::SafeTensors<'a>],
    name: &str,
) -> Result<safetensors::tensor::TensorView<'a>> {
    for st in shards {
        if let Ok(tv) = st.tensor(name) {
            return Ok(tv);
        }
    }
    Err(anyhow::anyhow!("tensor `{}` not found in any shard", name))
}

/// Check if a tensor exists in any shard
fn has_tensor(shards: &[safetensors::SafeTensors<'_>], name: &str) -> bool {
    shards.iter().any(|st| st.tensor(name).is_ok())
}

/// F32 weight buffer on OpenCL device
struct F32Weight {
    buf: Buffer<u8>, // Raw F32 bytes on GPU
    cols: usize,     // in_dim (only `cols` is needed downstream;
                     //  `rows` is derived from out-buffer shape)
}

/// F32 bias buffer on OpenCL device
struct F32Bias {
    buf: Buffer<u8>, // Raw F32 bytes on GPU [out_dim]
}

/// RmsNorm weights on OpenCL device
struct OclRmsNorm {
    scale: Buffer<u8>, // F32 [dim]
    eps: f32,
}

/// The RMS scales Z-Image applies to q and k before RoPE.
///
/// Present in the checkpoint means required: skipping it does not fail, it computes a
/// different attention. These were loaded and NOT applied here for a while, so a layer on
/// an Arc card answered differently from the same layer on a CUDA card, on every step.
struct OclQkNorm {
    norm_q_scale: Buffer<u8>, // F32 [head_dim]
    norm_k_scale: Buffer<u8>, // F32 [head_dim]
    head_dim: usize,
    eps: f32,
}

/// OpenCL Z-Image attention weights
struct OclZImageAttention {
    to_q: F32Weight,   // [n_heads * head_dim, dim]
    to_k: F32Weight,   // [n_kv_heads * head_dim, dim]
    to_v: F32Weight,   // [n_kv_heads * head_dim, dim]
    to_out: F32Weight, // [dim, n_heads * head_dim]
    qk_norm: Option<OclQkNorm>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

/// OpenCL Z-Image FeedForward (SwiGLU) weights
struct OclFeedForward {
    w1: F32Weight, // gate: [hidden_dim, dim]
    w2: F32Weight, // down: [dim, hidden_dim]
    w3: F32Weight, // up:   [hidden_dim, dim]
}

/// OpenCL Z-Image transformer block
pub struct OpenCLZImageBlock {
    attention: OclZImageAttention,
    ffn: OclFeedForward,
    attention_norm1: OclRmsNorm,
    attention_norm2: OclRmsNorm,
    ffn_norm1: OclRmsNorm,
    ffn_norm2: OclRmsNorm,
    adaln: Option<OclAdaLN>,
    dim: usize,
    hidden_dim: usize,
}

/// AdaLN modulation: linear(adaln_dim, 4*dim) with bias
struct OclAdaLN {
    weight: F32Weight, // [4*dim, adaln_dim]
    bias: F32Bias,     // [4*dim]
}

/// Scratch buffers for one OpenCL Z-Image block forward pass
pub struct OclZImageScratch {
    // Main sequence buffers: [max_seq_len, dim]
    pub buf_a: Buffer<u8>, // general purpose
    pub buf_b: Buffer<u8>, // general purpose
    pub buf_c: Buffer<u8>, // general purpose
    // QKV buffers
    pub q_buf: Buffer<u8>,    // [max_seq_len, n_heads * head_dim]
    pub k_buf: Buffer<u8>,    // [max_seq_len, n_kv_heads * head_dim]
    pub v_buf: Buffer<u8>,    // [max_seq_len, n_kv_heads * head_dim]
    pub attn_out: Buffer<u8>, // [max_seq_len, n_heads * head_dim]
    // FFN buffers
    pub ffn_act: Buffer<u8>, // [max_seq_len, hidden_dim]
    // AdaLN buffers (4 separate chunks instead of one big buffer)
    pub adaln_full: Buffer<u8>,      // [1, 4*dim] raw matmul output
    pub adaln_scale_msa: Buffer<u8>, // [1, dim]
    pub adaln_gate_msa: Buffer<u8>,  // [1, dim]
    pub adaln_scale_mlp: Buffer<u8>, // [1, dim]
    pub adaln_gate_mlp: Buffer<u8>,  // [1, dim]
    pub max_seq_len: usize,
    pub dim: usize,
}

impl OclZImageScratch {
    pub fn new(
        max_seq_len: usize,
        dim: usize,
        hidden_dim: usize,
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
        pipelines: &OpenCLPipelines,
    ) -> Result<Self> {
        let alloc = |size: usize| -> Result<Buffer<u8>> {
            unsafe {
                let buf =
                    Buffer::<u8>::create(&pipelines.context, CL_MEM_READ_WRITE, size, null_mut())?;
                Ok(buf)
            }
        };

        let seq_dim = max_seq_len * dim * 4; // F32 = 4 bytes
        let q_size = max_seq_len * n_heads * head_dim * 4;
        let kv_size = max_seq_len * n_kv_heads * head_dim * 4;
        let ffn_size = max_seq_len * hidden_dim * 4;
        let _adaln_size = 4 * dim * 4; // 4 modulation vectors of dim each

        Ok(Self {
            buf_a: alloc(seq_dim)?,
            buf_b: alloc(seq_dim)?,
            buf_c: alloc(seq_dim)?,
            q_buf: alloc(q_size)?,
            k_buf: alloc(kv_size)?,
            v_buf: alloc(kv_size)?,
            attn_out: alloc(q_size)?,
            ffn_act: alloc(ffn_size)?,
            adaln_full: alloc(4 * dim * 4)?,
            adaln_scale_msa: alloc(dim * 4)?,
            adaln_gate_msa: alloc(dim * 4)?,
            adaln_scale_mlp: alloc(dim * 4)?,
            adaln_gate_mlp: alloc(dim * 4)?,
            max_seq_len,
            dim,
        })
    }
}

// ==================== Weight Loading ====================

/// Convert BF16 bytes to F32 bytes
fn bf16_to_f32(data: &[u8]) -> Vec<u8> {
    let n = data.len() / 2;
    let mut out = vec![0u8; n * 4];
    for i in 0..n {
        // BF16 is the upper 16 bits of F32 - just pad with zeros
        let bf16_bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        let f32_bits = (bf16_bits as u32) << 16;
        out[i * 4..i * 4 + 4].copy_from_slice(&f32_bits.to_le_bytes());
    }
    out
}

/// Upload raw F32 data to OpenCL buffer
fn upload_f32(data: &[u8], pipelines: &OpenCLPipelines) -> Result<Buffer<u8>> {
    unsafe {
        let mut buf =
            Buffer::<u8>::create(&pipelines.context, CL_MEM_READ_ONLY, data.len(), null_mut())?;
        pipelines
            .queue
            .enqueue_write_buffer(&mut buf, CL_BLOCKING, 0, data, &[])?;
        Ok(buf)
    }
}

/// Load a weight tensor from safetensors shards, converting BF16->F32 if needed
fn load_weight(
    shards: &[safetensors::SafeTensors<'_>],
    name: &str,
    rows: usize,
    cols: usize,
    pipelines: &OpenCLPipelines,
) -> Result<F32Weight> {
    let tv = find_tensor(shards, name)?;

    let f32_data = match tv.dtype() {
        safetensors::Dtype::F32 => tv.data().to_vec(),
        safetensors::Dtype::BF16 => bf16_to_f32(tv.data()),
        safetensors::Dtype::F16 => {
            // F16->F32 conversion
            let n = tv.data().len() / 2;
            let mut out = vec![0u8; n * 4];
            for i in 0..n {
                let bits = u16::from_le_bytes([tv.data()[i * 2], tv.data()[i * 2 + 1]]);
                let f = half::f16::from_bits(bits).to_f32();
                out[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
            }
            out
        }
        dt => {
            return Err(anyhow::anyhow!(
                "Unsupported dtype {:?} for tensor '{}'",
                dt,
                name
            ))
        }
    };

    let expected = rows * cols * 4;
    if f32_data.len() != expected {
        return Err(anyhow::anyhow!(
            "Tensor '{}' size mismatch: got {} bytes, expected {} ({}x{}x4)",
            name,
            f32_data.len(),
            expected,
            rows,
            cols
        ));
    }

    let buf = upload_f32(&f32_data, pipelines)?;
    Ok(F32Weight { buf, cols })
}

/// Load a bias tensor
fn load_bias(
    shards: &[safetensors::SafeTensors<'_>],
    name: &str,
    _dim: usize, // size hint only; bias length is derived from tensor data
    pipelines: &OpenCLPipelines,
) -> Result<F32Bias> {
    let tv = find_tensor(shards, name)?;

    let f32_data = match tv.dtype() {
        safetensors::Dtype::F32 => tv.data().to_vec(),
        safetensors::Dtype::BF16 => bf16_to_f32(tv.data()),
        safetensors::Dtype::F16 => {
            let n = tv.data().len() / 2;
            let mut out = vec![0u8; n * 4];
            for i in 0..n {
                let bits = u16::from_le_bytes([tv.data()[i * 2], tv.data()[i * 2 + 1]]);
                let f = half::f16::from_bits(bits).to_f32();
                out[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
            }
            out
        }
        dt => {
            return Err(anyhow::anyhow!(
                "Unsupported dtype {:?} for bias '{}'",
                dt,
                name
            ))
        }
    };

    let buf = upload_f32(&f32_data, pipelines)?;
    Ok(F32Bias { buf })
}

/// Load RmsNorm scale weights
fn load_rms_norm(
    shards: &[safetensors::SafeTensors<'_>],
    prefix: &str,
    _dim: usize, // size hint only; the scale tensor's data length is authoritative
    eps: f64,
    pipelines: &OpenCLPipelines,
) -> Result<OclRmsNorm> {
    let name = format!("{}.weight", prefix);
    let tv = find_tensor(shards, &name)?;

    let f32_data = match tv.dtype() {
        safetensors::Dtype::F32 => tv.data().to_vec(),
        safetensors::Dtype::BF16 => bf16_to_f32(tv.data()),
        safetensors::Dtype::F16 => {
            let n = tv.data().len() / 2;
            let mut out = vec![0u8; n * 4];
            for i in 0..n {
                let bits = u16::from_le_bytes([tv.data()[i * 2], tv.data()[i * 2 + 1]]);
                let f = half::f16::from_bits(bits).to_f32();
                out[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
            }
            out
        }
        dt => return Err(anyhow::anyhow!("Unsupported dtype {:?} for '{}'", dt, name)),
    };

    let scale = upload_f32(&f32_data, pipelines)?;
    Ok(OclRmsNorm {
        scale,
        eps: eps as f32,
    })
}

impl OpenCLZImageBlock {
    /// Load a Z-Image transformer block from safetensors data into OpenCL buffers.
    ///
    /// `prefix` is e.g. "layers.0" for main layer 0, "noise_refiner.0", etc.
    ///
    /// `adaln_dim` is the width the modulation projection READS - already narrowed against
    /// `dim`, which is what `Config::adaln_dim` answers. It arrives narrowed rather than as
    /// the embedding constant to be narrowed here, because narrowing it here is how a block
    /// comes to carry its own copy of a width the config already states.
    pub fn from_safetensors(
        shards: &[safetensors::SafeTensors<'_>],
        prefix: &str,
        dim: usize,
        hidden_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        norm_eps: f64,
        modulation: bool,
        adaln_dim: usize,
        pipelines: &OpenCLPipelines,
    ) -> Result<Self> {
        info!(
            "  OpenCL Z-Image block: loading '{}' (dim={}, hidden={})",
            prefix, dim, hidden_dim
        );

        // Attention weights (no bias - linear_no_bias)
        let to_q = load_weight(
            shards,
            &format!("{}.attention.to_q.weight", prefix),
            n_heads * head_dim,
            dim,
            pipelines,
        )?;
        let to_k = load_weight(
            shards,
            &format!("{}.attention.to_k.weight", prefix),
            n_kv_heads * head_dim,
            dim,
            pipelines,
        )?;
        let to_v = load_weight(
            shards,
            &format!("{}.attention.to_v.weight", prefix),
            n_kv_heads * head_dim,
            dim,
            pipelines,
        )?;
        let to_out = load_weight(
            shards,
            &format!("{}.attention.to_out.0.weight", prefix),
            dim,
            n_heads * head_dim,
            pipelines,
        )?;

        // QK norm (optional - check if tensors exist)
        let qk_norm = {
            let q_name = format!("{}.attention.norm_q.weight", prefix);
            if has_tensor(shards, &q_name) {
                let norm_q = load_rms_norm(
                    shards,
                    &format!("{}.attention.norm_q", prefix),
                    head_dim,
                    1e-5,
                    pipelines,
                )?;
                let norm_k = load_rms_norm(
                    shards,
                    &format!("{}.attention.norm_k", prefix),
                    head_dim,
                    1e-5,
                    pipelines,
                )?;
                // The kernel reads four floats at a time, so a head that is not a
                // multiple of four would leave its tail unnormalised - silently, which
                // is the failure this whole path just came out of.
                if !head_dim.is_multiple_of(4) {
                    return Err(anyhow::anyhow!(
                        "OpenCL Z-Image: head_dim {head_dim} is not a multiple of 4, so the \
                         q/k normalisation cannot be applied here; place this model on CUDA \
                         or CPU rather than getting a different attention on this card"
                    ));
                }
                Some(OclQkNorm {
                    norm_q_scale: norm_q.scale,
                    norm_k_scale: norm_k.scale,
                    head_dim,
                    eps: 1e-5,
                })
            } else {
                None
            }
        };

        let attention = OclZImageAttention {
            to_q,
            to_k,
            to_v,
            to_out,
            qk_norm,
            n_heads,
            n_kv_heads,
            head_dim,
        };

        // FFN weights (no bias)
        let ffn = OclFeedForward {
            w1: load_weight(
                shards,
                &format!("{}.feed_forward.w1.weight", prefix),
                hidden_dim,
                dim,
                pipelines,
            )?,
            w2: load_weight(
                shards,
                &format!("{}.feed_forward.w2.weight", prefix),
                dim,
                hidden_dim,
                pipelines,
            )?,
            w3: load_weight(
                shards,
                &format!("{}.feed_forward.w3.weight", prefix),
                hidden_dim,
                dim,
                pipelines,
            )?,
        };

        // Norms
        let attention_norm1 = load_rms_norm(
            shards,
            &format!("{}.attention_norm1", prefix),
            dim,
            norm_eps,
            pipelines,
        )?;
        let attention_norm2 = load_rms_norm(
            shards,
            &format!("{}.attention_norm2", prefix),
            dim,
            norm_eps,
            pipelines,
        )?;
        let ffn_norm1 = load_rms_norm(
            shards,
            &format!("{}.ffn_norm1", prefix),
            dim,
            norm_eps,
            pipelines,
        )?;
        let ffn_norm2 = load_rms_norm(
            shards,
            &format!("{}.ffn_norm2", prefix),
            dim,
            norm_eps,
            pipelines,
        )?;

        // AdaLN modulation (with bias)
        let adaln = if modulation {
            let weight = load_weight(
                shards,
                &format!("{}.adaLN_modulation.0.weight", prefix),
                4 * dim,
                adaln_dim,
                pipelines,
            )?;
            let bias = load_bias(
                shards,
                &format!("{}.adaLN_modulation.0.bias", prefix),
                4 * dim,
                pipelines,
            )?;
            Some(OclAdaLN { weight, bias })
        } else {
            None
        };

        Ok(Self {
            attention,
            ffn,
            attention_norm1,
            attention_norm2,
            ffn_norm1,
            ffn_norm2,
            adaln,
            dim,
            hidden_dim,
        })
    }

    /// Forward pass using OpenCL kernels.
    ///
    /// Input: x_buf contains [seq_len, dim] F32 data on OpenCL device
    /// Output: x_buf is updated in-place with the block output
    ///
    /// adaln_buf: optional [1, adaln_dim] F32 on OpenCL device
    /// cos_buf, sin_buf: [seq_len, head_dim/2] precomputed RoPE embeddings on OpenCL
    /// mask_buf: optional [1, seq_len] attention mask on OpenCL
    pub fn forward(
        &self,
        x_buf: &Buffer<u8>, // [seq_len, dim] input/output
        seq_len: usize,
        cos_buf: &Buffer<u8>,           // RoPE cos [seq_len, head_dim/2]
        sin_buf: &Buffer<u8>,           // RoPE sin [seq_len, head_dim/2]
        adaln_buf: Option<&Buffer<u8>>, // [1, adaln_dim] conditioning
        mask_buf: Option<&Buffer<u8>>,  // [1, seq_len] attention mask
        scratch: &OclZImageScratch,
        pipelines: &OpenCLPipelines,
    ) -> Result<()> {
        if let Some(ref adaln) = self.adaln {
            self.forward_with_modulation(
                x_buf,
                seq_len,
                cos_buf,
                sin_buf,
                adaln_buf.unwrap(),
                adaln,
                mask_buf,
                scratch,
                pipelines,
            )
        } else {
            self.forward_without_modulation(
                x_buf, seq_len, cos_buf, sin_buf, mask_buf, scratch, pipelines,
            )
        }
    }

    fn forward_with_modulation(
        &self,
        x_buf: &Buffer<u8>,
        seq_len: usize,
        cos_buf: &Buffer<u8>,
        sin_buf: &Buffer<u8>,
        adaln_input: &Buffer<u8>,
        adaln: &OclAdaLN,
        mask_buf: Option<&Buffer<u8>>,
        scratch: &OclZImageScratch,
        p: &OpenCLPipelines,
    ) -> Result<()> {
        let dim = self.dim as u32;
        let wg = 64u32; // F32_WG from kernel

        // 1. AdaLN: adaln_buf = matmul(adaln_input, weight) + bias -> [1, 4*dim]
        let adaln_out_dim = (4 * self.dim) as u32;
        let adaln_in_dim = adaln.weight.cols as u32;
        unsafe {
            ExecuteKernel::new(&p.f32_matmul_bias)
                .set_arg(&adaln.weight.buf)
                .set_arg(adaln_input)
                .set_arg(&adaln.bias.buf)
                .set_arg(&scratch.adaln_full)
                .set_arg(&1u32) // seq_len=1
                .set_arg(&adaln_in_dim)
                .set_arg(&adaln_out_dim)
                .set_global_work_size(adaln_out_dim as usize * wg as usize)
                .set_local_work_size(wg as usize)
                .enqueue_nd_range(&p.queue)?;
        }

        // adaln_full now has [1, 4*dim] - split into 4 separate buffers via copy
        let dim_bytes = self.dim * 4; // F32 bytes
        unsafe {
            copy_buffer_region(
                &p.queue,
                &scratch.adaln_full,
                &scratch.adaln_scale_msa,
                0,
                0,
                dim_bytes,
            )?;
            copy_buffer_region(
                &p.queue,
                &scratch.adaln_full,
                &scratch.adaln_gate_msa,
                dim_bytes,
                0,
                dim_bytes,
            )?;
            copy_buffer_region(
                &p.queue,
                &scratch.adaln_full,
                &scratch.adaln_scale_mlp,
                2 * dim_bytes,
                0,
                dim_bytes,
            )?;
            copy_buffer_region(
                &p.queue,
                &scratch.adaln_full,
                &scratch.adaln_gate_mlp,
                3 * dim_bytes,
                0,
                dim_bytes,
            )?;
        }

        // 2. Attention norm1: buf_a = rms_norm(x_buf) -> [seq, dim]
        let total_elements = (seq_len * self.dim) as u32;
        unsafe {
            ExecuteKernel::new(&p.rms_norm)
                .set_arg(x_buf)
                .set_arg(&scratch.buf_a)
                .set_arg(&self.attention_norm1.scale)
                .set_arg(&dim)
                .set_arg(&self.attention_norm1.eps)
                .set_global_work_size(seq_len * 256)
                .set_local_work_size(256)
                .enqueue_nd_range(&p.queue)?;
        }

        // 3. Scale: buf_b = (1 + scale_msa) * buf_a -> [seq, dim]
        unsafe {
            ExecuteKernel::new(&p.scale_after_norm)
                .set_arg(&scratch.buf_a)
                .set_arg(&scratch.adaln_scale_msa)
                .set_arg(&scratch.buf_b)
                .set_arg(&dim)
                .set_arg(&total_elements)
                .set_global_work_size(total_elements.div_ceil(4) as usize)
                .enqueue_nd_range(&p.queue)?;
        }

        // 4. QKV projection: Q, K, V = matmul(buf_b, weights)
        self.project_qkv(&scratch.buf_b, seq_len, scratch, p)?;

        // 5. Normalise q and k, before RoPE - the order the reference uses.
        self.apply_qk_norm(seq_len, scratch, p)?;

        // 6. Apply RoPE to Q and K
        self.apply_rope(seq_len, cos_buf, sin_buf, scratch, p)?;

        // 7. Attention: attn_out = attention(Q, K, V, mask)
        self.compute_attention(seq_len, mask_buf, scratch, p)?;

        // 8. Project attention output back to dim, then norm2
        self.project_attn_output(seq_len, scratch, p)?;
        unsafe {
            ExecuteKernel::new(&p.rms_norm)
                .set_arg(&scratch.buf_a)
                .set_arg(&scratch.buf_b)
                .set_arg(&self.attention_norm2.scale)
                .set_arg(&dim)
                .set_arg(&self.attention_norm2.eps)
                .set_global_work_size(seq_len * 256)
                .set_local_work_size(256)
                .enqueue_nd_range(&p.queue)?;
        }

        // 9. Gated residual: buf_c = x + tanh(gate_msa) * attn_norm2_out
        unsafe {
            ExecuteKernel::new(&p.gated_residual)
                .set_arg(x_buf)
                .set_arg(&scratch.buf_b)
                .set_arg(&scratch.adaln_gate_msa)
                .set_arg(&scratch.buf_c)
                .set_arg(&dim)
                .set_arg(&total_elements)
                .set_global_work_size(total_elements.div_ceil(4) as usize)
                .enqueue_nd_range(&p.queue)?;
        }

        // 10. FFN norm1: buf_a = rms_norm(buf_c)
        unsafe {
            ExecuteKernel::new(&p.rms_norm)
                .set_arg(&scratch.buf_c)
                .set_arg(&scratch.buf_a)
                .set_arg(&self.ffn_norm1.scale)
                .set_arg(&dim)
                .set_arg(&self.ffn_norm1.eps)
                .set_global_work_size(seq_len * 256)
                .set_local_work_size(256)
                .enqueue_nd_range(&p.queue)?;
        }

        // 11. Scale: buf_b = (1 + scale_mlp) * buf_a
        unsafe {
            ExecuteKernel::new(&p.scale_after_norm)
                .set_arg(&scratch.buf_a)
                .set_arg(&scratch.adaln_scale_mlp)
                .set_arg(&scratch.buf_b)
                .set_arg(&dim)
                .set_arg(&total_elements)
                .set_global_work_size(total_elements.div_ceil(4) as usize)
                .enqueue_nd_range(&p.queue)?;
        }

        // 12. FFN: fused gate_up_silu then down projection
        self.compute_ffn(&scratch.buf_b, seq_len, scratch, p)?;
        // ffn output is in scratch.buf_a

        // 12. FFN norm2
        unsafe {
            ExecuteKernel::new(&p.rms_norm)
                .set_arg(&scratch.buf_a)
                .set_arg(&scratch.buf_b)
                .set_arg(&self.ffn_norm2.scale)
                .set_arg(&dim)
                .set_arg(&self.ffn_norm2.eps)
                .set_global_work_size(seq_len * 256)
                .set_local_work_size(256)
                .enqueue_nd_range(&p.queue)?;
        }

        // 13. Gated residual: output = buf_c + tanh(gate_mlp) * buf_b -> x_buf
        unsafe {
            ExecuteKernel::new(&p.gated_residual)
                .set_arg(&scratch.buf_c)
                .set_arg(&scratch.buf_b)
                .set_arg(&scratch.adaln_gate_mlp)
                .set_arg(x_buf)
                .set_arg(&dim)
                .set_arg(&total_elements)
                .set_global_work_size(total_elements.div_ceil(4) as usize)
                .enqueue_nd_range(&p.queue)?;
        }

        Ok(())
    }

    fn forward_without_modulation(
        &self,
        x_buf: &Buffer<u8>,
        seq_len: usize,
        cos_buf: &Buffer<u8>,
        sin_buf: &Buffer<u8>,
        mask_buf: Option<&Buffer<u8>>,
        scratch: &OclZImageScratch,
        p: &OpenCLPipelines,
    ) -> Result<()> {
        let dim = self.dim as u32;
        let total_elements = (seq_len * self.dim) as u32;

        // 1. Attention norm1
        unsafe {
            ExecuteKernel::new(&p.rms_norm)
                .set_arg(x_buf)
                .set_arg(&scratch.buf_a)
                .set_arg(&self.attention_norm1.scale)
                .set_arg(&dim)
                .set_arg(&self.attention_norm1.eps)
                .set_global_work_size(seq_len * 256)
                .set_local_work_size(256)
                .enqueue_nd_range(&p.queue)?;
        }

        // 2. QKV + RoPE + Attention
        self.project_qkv(&scratch.buf_a, seq_len, scratch, p)?;
        self.apply_rope(seq_len, cos_buf, sin_buf, scratch, p)?;
        self.compute_attention(seq_len, mask_buf, scratch, p)?;
        self.project_attn_output(seq_len, scratch, p)?;

        // 3. Attention norm2
        unsafe {
            ExecuteKernel::new(&p.rms_norm)
                .set_arg(&scratch.buf_a)
                .set_arg(&scratch.buf_b)
                .set_arg(&self.attention_norm2.scale)
                .set_arg(&dim)
                .set_arg(&self.attention_norm2.eps)
                .set_global_work_size(seq_len * 256)
                .set_local_work_size(256)
                .enqueue_nd_range(&p.queue)?;
        }

        // 4. Residual: buf_c = x + attn_out
        unsafe {
            ExecuteKernel::new(&p.add)
                .set_arg(x_buf)
                .set_arg(&scratch.buf_b)
                .set_arg(&scratch.buf_c)
                .set_arg(&total_elements)
                .set_global_work_size(total_elements.div_ceil(4) as usize)
                .enqueue_nd_range(&p.queue)?;
        }

        // 5. FFN norm1
        unsafe {
            ExecuteKernel::new(&p.rms_norm)
                .set_arg(&scratch.buf_c)
                .set_arg(&scratch.buf_a)
                .set_arg(&self.ffn_norm1.scale)
                .set_arg(&dim)
                .set_arg(&self.ffn_norm1.eps)
                .set_global_work_size(seq_len * 256)
                .set_local_work_size(256)
                .enqueue_nd_range(&p.queue)?;
        }

        // 6. FFN
        self.compute_ffn(&scratch.buf_a, seq_len, scratch, p)?;
        // FFN output in scratch.buf_a

        // 7. FFN norm2
        unsafe {
            ExecuteKernel::new(&p.rms_norm)
                .set_arg(&scratch.buf_a)
                .set_arg(&scratch.buf_b)
                .set_arg(&self.ffn_norm2.scale)
                .set_arg(&dim)
                .set_arg(&self.ffn_norm2.eps)
                .set_global_work_size(seq_len * 256)
                .set_local_work_size(256)
                .enqueue_nd_range(&p.queue)?;
        }

        // 8. Residual: x = buf_c + buf_b
        unsafe {
            ExecuteKernel::new(&p.add)
                .set_arg(&scratch.buf_c)
                .set_arg(&scratch.buf_b)
                .set_arg(x_buf)
                .set_arg(&total_elements)
                .set_global_work_size(total_elements.div_ceil(4) as usize)
                .enqueue_nd_range(&p.queue)?;
        }

        Ok(())
    }

    // ==================== Helper methods ====================

    fn project_qkv(
        &self,
        input: &Buffer<u8>,
        seq_len: usize,
        scratch: &OclZImageScratch,
        p: &OpenCLPipelines,
    ) -> Result<()> {
        let attn = &self.attention;
        let m = seq_len as u32;
        let k = self.dim as u32;

        // Q projection: [seq, dim] x [n_heads*head_dim, dim]^T -> [seq, n_heads*head_dim]
        let n_q = (attn.n_heads * attn.head_dim) as u32;
        Self::dispatch_tiled_matmul(
            &p.f32_tiled_matmul,
            input,
            &attn.to_q.buf,
            &scratch.q_buf,
            m,
            n_q,
            k,
            p,
        )?;

        // K projection
        let n_k = (attn.n_kv_heads * attn.head_dim) as u32;
        Self::dispatch_tiled_matmul(
            &p.f32_tiled_matmul,
            input,
            &attn.to_k.buf,
            &scratch.k_buf,
            m,
            n_k,
            k,
            p,
        )?;

        // V projection
        let n_v = (attn.n_kv_heads * attn.head_dim) as u32;
        Self::dispatch_tiled_matmul(
            &p.f32_tiled_matmul,
            input,
            &attn.to_v.buf,
            &scratch.v_buf,
            m,
            n_v,
            k,
            p,
        )?;

        Ok(())
    }

    /// `q = rms(q).norm_q` and `k = rms(k).norm_k`, per head, in place.
    ///
    /// The same `rms_norm` kernel the block norms use. Its row is its workgroup and its row
    /// length is the argument, so a `[seq, heads.head_dim]` buffer is `seq.heads` rows of
    /// `head_dim` without moving a byte - the reshape the tensor path writes as
    /// `(b, seq, heads, head_dim)` is already the memory layout here.
    ///
    /// In place is safe: after the reduction barrier each thread reads and writes the same
    /// four elements.
    fn apply_qk_norm(
        &self,
        seq_len: usize,
        scratch: &OclZImageScratch,
        p: &OpenCLPipelines,
    ) -> Result<()> {
        let Some(norm) = self.attention.qk_norm.as_ref() else {
            return Ok(());
        };
        let head_dim = norm.head_dim as u32;
        for (buf, scale, heads) in [
            (&scratch.q_buf, &norm.norm_q_scale, self.attention.n_heads),
            (
                &scratch.k_buf,
                &norm.norm_k_scale,
                self.attention.n_kv_heads,
            ),
        ] {
            unsafe {
                ExecuteKernel::new(&p.rms_norm)
                    .set_arg(buf)
                    .set_arg(buf)
                    .set_arg(scale)
                    .set_arg(&head_dim)
                    .set_arg(&norm.eps)
                    .set_global_work_size(seq_len * heads * 256)
                    .set_local_work_size(256)
                    .enqueue_nd_range(&p.queue)?;
            }
        }
        Ok(())
    }

    /// Dispatch a tiled GEMM: `C[M,N] = A[M,K] x B^T[N,K]`
    fn dispatch_tiled_matmul(
        kernel: &opencl3::kernel::Kernel,
        a: &Buffer<u8>, // [M, K]
        b: &Buffer<u8>, // [N, K]
        c: &Buffer<u8>, // [M, N]
        m: u32,
        n: u32,
        k: u32,
        p: &OpenCLPipelines,
    ) -> Result<()> {
        let ts = 16usize;
        let gm = (m as usize).div_ceil(ts) * ts;
        let gn = (n as usize).div_ceil(ts) * ts;
        unsafe {
            ExecuteKernel::new(kernel)
                .set_arg(a)
                .set_arg(b)
                .set_arg(c)
                .set_arg(&m)
                .set_arg(&n)
                .set_arg(&k)
                .set_global_work_sizes(&[gm, gn])
                .set_local_work_sizes(&[ts, ts])
                .enqueue_nd_range(&p.queue)?;
        }
        Ok(())
    }

    fn apply_rope(
        &self,
        seq_len: usize,
        cos_buf: &Buffer<u8>,
        sin_buf: &Buffer<u8>,
        scratch: &OclZImageScratch,
        p: &OpenCLPipelines,
    ) -> Result<()> {
        let attn = &self.attention;
        // Apply RoPE to Q (in-place via temp buffer)
        // Q is [seq_len, n_heads * head_dim] but RoPE needs [n_heads_total, seq_len, head_dim]
        // Since it's already flat and the kernel works on pairs, we just dispatch correctly
        let q_total_pairs = (attn.n_heads * seq_len * attn.head_dim / 2) as u32;
        let n_head_total = attn.n_heads as u32;
        let seq = seq_len as u32;
        let hd = attn.head_dim as u32;

        // Q: apply_rotary_emb in-place (write to attn_out, then swap)
        unsafe {
            ExecuteKernel::new(&p.apply_rotary_emb)
                .set_arg(&scratch.q_buf)
                .set_arg(cos_buf)
                .set_arg(sin_buf)
                .set_arg(&scratch.attn_out) // temp
                .set_arg(&n_head_total)
                .set_arg(&seq)
                .set_arg(&hd)
                .set_global_work_size(q_total_pairs as usize)
                .enqueue_nd_range(&p.queue)?;
        }
        // Copy back: attn_out -> q_buf
        unsafe {
            let size = attn.n_heads * seq_len * attn.head_dim * 4;
            copy_buffer_region(&p.queue, &scratch.attn_out, &scratch.q_buf, 0, 0, size)?;
        }

        // K: apply_rotary_emb
        let k_total_pairs = (attn.n_kv_heads * seq_len * attn.head_dim / 2) as u32;
        let kv_head_total = attn.n_kv_heads as u32;
        unsafe {
            ExecuteKernel::new(&p.apply_rotary_emb)
                .set_arg(&scratch.k_buf)
                .set_arg(cos_buf)
                .set_arg(sin_buf)
                .set_arg(&scratch.attn_out)
                .set_arg(&kv_head_total)
                .set_arg(&seq)
                .set_arg(&hd)
                .set_global_work_size(k_total_pairs as usize)
                .enqueue_nd_range(&p.queue)?;
        }
        unsafe {
            let size = attn.n_kv_heads * seq_len * attn.head_dim * 4;
            copy_buffer_region(&p.queue, &scratch.attn_out, &scratch.k_buf, 0, 0, size)?;
        }

        Ok(())
    }

    fn compute_attention(
        &self,
        seq_len: usize,
        mask_buf: Option<&Buffer<u8>>,
        scratch: &OclZImageScratch,
        p: &OpenCLPipelines,
    ) -> Result<()> {
        let attn = &self.attention;
        let batch = 1u32;

        // Q is [seq_len, n_heads * head_dim] laid out as [batch * n_heads, seq_len, head_dim]
        // after reshape. Since batch=1 and data is contiguous, the layout matches.
        // The attention kernel expects [batch*n_head, seq_len, head_dim] which is exactly
        // [n_heads, seq_len, head_dim] - but our data is [seq_len, n_heads * head_dim].
        // We need to think of this as already transposed: [n_heads, seq_len, head_dim] via reshape.
        //
        // Actually: QKV are [seq_len, n_heads * head_dim] = [seq_len, n_heads, head_dim] flat.
        // The attention kernel needs [batch * n_heads, seq_len, head_dim].
        // Our data has head as the inner dimension, but kernel expects head as outer.
        // This is a TRANSPOSE issue. For now, dispatch with batch * n_head * seq_len work items
        // where the kernel handles the indexing internally.
        //
        // Note: The image_attention kernel handles this with the layout:
        //   [batch * n_head, seq_len, head_dim]
        // We need our Q to be in this layout. Currently Q is [seq_len, n_heads * head_dim].
        // Reshape to [seq_len, n_heads, head_dim], transpose to [n_heads, seq_len, head_dim].
        // For batch=1 this is the same as [batch*n_heads, seq_len, head_dim].
        //
        // HOWEVER: doing a transpose on OpenCL requires an extra kernel or we handle it
        // in the attention kernel itself. For simplicity, let's use a "strided" approach:
        // The kernel can compute Q[b,h,q,d] = q_buf[q * n_heads * head_dim + h * head_dim + d]
        // instead of q_buf[(b*n_head+h) * seq_len * head_dim + q * head_dim + d]
        //
        // This means our image_attention kernel needs a "transposed Q/K/V" variant.
        // For now, let's use the existing kernel but note this layout mismatch needs fixing.
        // TODO: Add a transpose kernel or modify image_attention to handle [seq, heads, dim] layout

        // For correctness, we dispatch with the layout as-is. The kernel expects
        // contiguous [batch*n_head, seq_len, head_dim], so we need a quick transpose.
        // Since this would be complex to do efficiently on OpenCL, let's add a note
        // and use a simple transpose kernel.

        // Simple approach: the data is batch=1, so we can use n_head as the "batch" dim
        // Q_transposed[h, s, d] = Q_orig[s, h, d]  - just swap the first two dims
        // We can do this as part of the attention computation.

        // For MVP: use a simple single-work-item-per-output transpose
        // TODO: optimize with local memory tiling

        let n_h = attn.n_heads as u32;
        let n_kvh = attn.n_kv_heads as u32;
        let seq = seq_len as u32;
        let hd = attn.head_dim as u32;
        let mask_present = if mask_buf.is_some() { 1u32 } else { 0u32 };

        // Use a dummy mask buffer if no mask
        let dummy_mask;
        let mask_ref = if let Some(m) = mask_buf {
            m
        } else {
            // Create a small dummy buffer (won't be read since mask_present=0)
            dummy_mask =
                unsafe { Buffer::<u8>::create(&p.context, CL_MEM_READ_ONLY, 4, null_mut())? };
            &dummy_mask
        };

        unsafe {
            ExecuteKernel::new(&p.image_attention)
                .set_arg(&scratch.q_buf)
                .set_arg(&scratch.k_buf)
                .set_arg(&scratch.v_buf)
                .set_arg(mask_ref)
                .set_arg(&scratch.attn_out)
                .set_arg(&batch)
                .set_arg(&n_h)
                .set_arg(&n_kvh)
                .set_arg(&seq)
                .set_arg(&hd)
                .set_arg(&mask_present)
                .set_global_work_size((batch * n_h * seq) as usize)
                .enqueue_nd_range(&p.queue)?;
        }

        Ok(())
    }

    fn project_attn_output(
        &self,
        seq_len: usize,
        scratch: &OclZImageScratch,
        p: &OpenCLPipelines,
    ) -> Result<()> {
        let m = seq_len as u32;
        let attn = &self.attention;
        let n = self.dim as u32;
        let k = (attn.n_heads * attn.head_dim) as u32;

        // attn_out [seq, n_heads * head_dim] -> buf_a [seq, dim]
        Self::dispatch_tiled_matmul(
            &p.f32_tiled_matmul,
            &scratch.attn_out,
            &attn.to_out.buf,
            &scratch.buf_a,
            m,
            n,
            k,
            p,
        )?;

        Ok(())
    }

    fn compute_ffn(
        &self,
        input: &Buffer<u8>,
        seq_len: usize,
        scratch: &OclZImageScratch,
        p: &OpenCLPipelines,
    ) -> Result<()> {
        let m = seq_len as u32;
        let k_in = self.dim as u32;
        let n_hidden = self.hidden_dim as u32;

        // Fused tiled gate_up_silu: ffn_act = silu(input x w1^T) * (input x w3^T)
        let ts = 16usize;
        let gm = (m as usize).div_ceil(ts) * ts;
        let gn = (n_hidden as usize).div_ceil(ts) * ts;
        unsafe {
            ExecuteKernel::new(&p.f32_tiled_gate_up_silu)
                .set_arg(&self.ffn.w1.buf)
                .set_arg(&self.ffn.w3.buf)
                .set_arg(input)
                .set_arg(&scratch.ffn_act)
                .set_arg(&m)
                .set_arg(&n_hidden)
                .set_arg(&k_in)
                .set_global_work_sizes(&[gm, gn])
                .set_local_work_sizes(&[ts, ts])
                .enqueue_nd_range(&p.queue)?;
        }

        // Down projection: buf_a = ffn_act x w2^T -> [seq, dim]
        Self::dispatch_tiled_matmul(
            &p.f32_tiled_matmul,
            &scratch.ffn_act,
            &self.ffn.w2.buf,
            &scratch.buf_a,
            m,
            k_in,
            n_hidden,
            p,
        )?;

        Ok(())
    }
}

// SAFETY: OpenCL buffers are GPU memory handles accessed only through the OpenCL command queue.
// The queue serializes all operations, making cross-thread access safe.
unsafe impl Send for OpenCLZImageBlock {}
unsafe impl Sync for OpenCLZImageBlock {}
unsafe impl Send for OclZImageScratch {}
unsafe impl Sync for OclZImageScratch {}

/// Read an OpenCL status code as a result, naming the call that returned it.
///
/// The raw entry points below report failure in their return value rather than by any other
/// means, and a code on its own says nothing about where it came from; the caller's name is
/// what makes the message usable, so it is asked for rather than reconstructed.
fn cl_status(call: &str, status: i32) -> Result<()> {
    match status {
        0 => Ok(()),
        code => Err(anyhow::anyhow!("{call} failed: {code}")),
    }
}

/// Copy a region from one OpenCL buffer to another (immutable references).
/// Uses raw OpenCL API to avoid Rust mutability requirements.
unsafe fn copy_buffer_region(
    queue: &opencl3::command_queue::CommandQueue,
    src: &Buffer<u8>,
    dst: &Buffer<u8>,
    src_offset: usize,
    dst_offset: usize,
    size: usize,
) -> Result<()> {
    use opencl3::types::*;
    extern "C" {
        fn clEnqueueCopyBuffer(
            command_queue: cl_command_queue,
            src_buffer: cl_mem,
            dst_buffer: cl_mem,
            src_offset: usize,
            dst_offset: usize,
            cb: usize,
            num_events_in_wait_list: cl_uint,
            event_wait_list: *const cl_event,
            event: *mut cl_event,
        ) -> cl_int;
    }
    let ret = clEnqueueCopyBuffer(
        queue.get(),
        src.get(),
        dst.get(),
        src_offset,
        dst_offset,
        size,
        0,
        std::ptr::null(),
        std::ptr::null_mut(),
    );
    cl_status("clEnqueueCopyBuffer", ret)
}

/// Write data to an OpenCL buffer (immutable reference).
unsafe fn write_buffer_raw(
    queue: &opencl3::command_queue::CommandQueue,
    dst: &Buffer<u8>,
    data: &[u8],
) -> Result<()> {
    use opencl3::types::*;
    extern "C" {
        fn clEnqueueWriteBuffer(
            command_queue: cl_command_queue,
            buffer: cl_mem,
            blocking_write: cl_bool,
            offset: usize,
            cb: usize,
            ptr: *const std::ffi::c_void,
            num_events_in_wait_list: cl_uint,
            event_wait_list: *const cl_event,
            event: *mut cl_event,
        ) -> cl_int;
    }
    let ret = clEnqueueWriteBuffer(
        queue.get(),
        dst.get(),
        1, // CL_BLOCKING
        0,
        data.len(),
        data.as_ptr() as *const std::ffi::c_void,
        0,
        std::ptr::null(),
        std::ptr::null_mut(),
    );
    cl_status("clEnqueueWriteBuffer", ret)
}

/// Public wrapper for write_buffer_raw for cross-module use.
///
/// # Safety
///
/// `dst` must be a valid OpenCL buffer with at least `data.len()` bytes
/// of allocated capacity, and the OpenCL command queue + buffer must
/// belong to the same context. The host pointer `data.as_ptr()` is
/// dereferenced for the length of `data` by `clEnqueueWriteBuffer`,
/// so `data` must remain valid for the duration of the (blocking)
/// call. Caller is responsible for matching buffer dtype to the byte
/// stream layout (this wrapper takes a raw `&[u8]`).
pub unsafe fn write_buffer_raw_pub(
    queue: &opencl3::command_queue::CommandQueue,
    dst: &Buffer<u8>,
    data: &[u8],
) -> Result<()> {
    write_buffer_raw(queue, dst, data)
}
