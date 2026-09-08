//! OpenCL Flux transformer blocks - runs on Intel Arc via custom OpenCL kernels.
//!
//! Loads Q4 weight data from GGUF directly into OpenCL buffers (bypassing the substrate),
//! then runs the Flux DoubleBlock and SingleBlock forward passes using OpenCL kernels:
//!   q4_matmul/q4k_matmul (dequant-on-the-fly), layer_norm, gelu, image_attention,
//!   apply_rotary_emb, scale_after_norm, gated_residual, broadcast_mul.
//!
//! Flux architecture:
//!   - 19 DoubleBlocks (joint img+txt attention with AdaLN modulation)
//!   - 38 SingleBlocks (merged sequence with gated residual)
//!
//! Weight layout: Q4_0 or Q4_K raw bytes in OpenCL buffers.
//! Bias and norm weights stored as F32.

#![allow(clippy::too_many_arguments)]

use anyhow::Result;
use opencl3::command_queue::CL_BLOCKING;
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use std::ptr::null_mut;
use tracing::info;

use crate::inference::kernel::opencl::{OpenCLPipelines, OpenCLQuantFormat};

// ==================== Weight Structures ====================

/// Weight buffer on OpenCL device - either quantized (Q4/Q6K) or dense F32
enum OclWeight {
    /// Quantized weight (Q4_0, Q4_K, Q6_K) - uses dequant matmul kernels
    Quantized {
        buf: Buffer<u8>,
        rows: usize, // out_dim
        cols: usize, // in_dim
        fmt: OpenCLQuantFormat,
    },
    /// Dense F32 weight (from Q8_0 dequant or native F32) - uses f32_matmul kernels
    Dense {
        buf: Buffer<u8>, // Raw F32 bytes on GPU
        rows: usize,     // out_dim
        cols: usize,     // in_dim
    },
}

impl OclWeight {
    fn rows(&self) -> usize {
        match self {
            OclWeight::Quantized { rows, .. } | OclWeight::Dense { rows, .. } => *rows,
        }
    }
}

/// F32 bias buffer on OpenCL device
struct F32Buf {
    buf: Buffer<u8>, // Raw F32 bytes on GPU
}

/// A linear layer: quantized or F32 weight + optional bias
struct OclQLinear {
    weight: OclWeight,
    bias: Option<F32Buf>,
}

/// LayerNorm weights (scale only, no bias - Flux uses no-bias LayerNorm)
struct OclLayerNorm {
    scale: F32Buf, // F32 [dim]
    dim: usize,
}

/// RmsNorm weights (for QK norm)
struct OclRmsNorm {
    scale: F32Buf,
}

/// QK normalization (query_norm + key_norm)
struct OclQkNorm {
    query_norm: OclRmsNorm,
    key_norm: OclRmsNorm,
}

/// Self-attention weights for one stream (img or txt side of DoubleBlock)
struct OclSelfAttention {
    qkv: OclQLinear,  // [dim -> 3*dim] with bias
    proj: OclQLinear, // [dim -> dim] with bias
    qk_norm: OclQkNorm,
    num_heads: usize,
}

/// MLP: lin1 (GELU activation) -> lin2
struct OclMlp {
    lin1: OclQLinear, // [dim -> mlp_sz] with bias
    lin2: OclQLinear, // [mlp_sz -> dim] with bias
}

/// Modulation2 (for DoubleBlock): silu -> linear -> chunk(6)
/// Produces 2x (shift, scale, gate) = 6 * dim
struct OclModulation2 {
    lin: OclQLinear, // [dim -> 6*dim] with bias
}

/// Modulation1 (for SingleBlock): silu -> linear -> chunk(3)
/// Produces (shift, scale, gate) = 3 * dim
struct OclModulation1 {
    lin: OclQLinear, // [dim -> 3*dim] with bias
}

// ==================== Block Structures ====================

/// OpenCL Flux DoubleBlock
pub struct OpenCLFluxDoubleBlock {
    img_mod: OclModulation2,
    img_norm1: OclLayerNorm,
    img_attn: OclSelfAttention,
    img_norm2: OclLayerNorm,
    img_mlp: OclMlp,
    txt_mod: OclModulation2,
    txt_norm1: OclLayerNorm,
    txt_attn: OclSelfAttention,
    txt_norm2: OclLayerNorm,
    txt_mlp: OclMlp,
    dim: usize,
    mlp_sz: usize,
}

/// OpenCL Flux SingleBlock
pub struct OpenCLFluxSingleBlock {
    linear1: OclQLinear, // [dim -> 3*dim + mlp_sz] with bias
    linear2: OclQLinear, // [dim + mlp_sz -> dim] with bias
    pre_norm: OclLayerNorm,
    modulation: OclModulation1,
    qk_norm: OclQkNorm,
    dim: usize,
    mlp_sz: usize,
    num_heads: usize,
}

/// Scratch buffers for OpenCL Flux forward pass
pub struct OclFluxScratch {
    // Sequence buffers for img and txt streams
    pub img_buf: Buffer<u8>,    // [max_seq_len, dim] F32
    pub txt_buf: Buffer<u8>,    // [max_txt_len, dim] F32
    pub merged_buf: Buffer<u8>, // [max_seq_len + max_txt_len, dim] F32
    // Modulation outputs (6 * dim for double, 3 * dim for single)
    pub mod_buf: Buffer<u8>, // [1, 6*dim] F32
    // QKV buffers (for combined img+txt attention)
    pub q_buf: Buffer<u8>,
    pub k_buf: Buffer<u8>,
    pub v_buf: Buffer<u8>,
    pub attn_out: Buffer<u8>,
    // MLP intermediates
    pub mlp_buf: Buffer<u8>, // [max_seq_len, mlp_sz] F32
    // Conditioning (persist across blocks within a segment)
    pub vec_buf: Buffer<u8>, // [1, dim] F32 - conditioning vector
    pub pe_buf: Buffer<u8>,  // [total_seq, head_dim/2, 2, 2] F32 - RoPE PE
    // Cached silu(vec_buf) - persists across ALL blocks within ONE
    // denoise step. HeteroFlux::forward calls invalidate_silu_vec()
    // right after uploading vec_buf to mark the cache dirty; the next
    // block-forward that needs silu(vec) populates it, and every
    // subsequent block in the same step skips the recomputation.
    pub silu_vec_buf: Buffer<u8>, // [1, dim] F32
    silu_vec_dirty: bool,
    // General purpose
    pub tmp_a: Buffer<u8>,
    pub tmp_b: Buffer<u8>,
    pub max_img_seq: usize,
    pub max_txt_seq: usize,
    pub dim: usize,
}

impl OclFluxScratch {
    /// Mark silu(vec_buf) cache dirty. Call this right after writing a
    /// new vec_buf at the start of each denoise step.
    pub fn invalidate_silu_vec(&mut self) {
        self.silu_vec_dirty = true;
    }

    /// Ensure silu_vec_buf holds an up-to-date silu(vec_buf).
    /// Computes only when dirty; otherwise no-op.
    pub fn ensure_silu_vec(&mut self, p: &OpenCLPipelines) -> Result<()> {
        if !self.silu_vec_dirty {
            return Ok(());
        }
        let mut vec_cpu = vec![0.0f32; self.dim];
        unsafe {
            read_f32(&p.queue, &self.vec_buf, &mut vec_cpu)?;
        }
        for v in vec_cpu.iter_mut() {
            *v = *v / (1.0 + (-*v).exp());
        }
        unsafe {
            write_f32(&p.queue, &mut self.silu_vec_buf, &vec_cpu)?;
        }
        self.silu_vec_dirty = false;
        Ok(())
    }

    pub fn new(
        max_img_seq: usize,
        max_txt_seq: usize,
        dim: usize,
        mlp_sz: usize,
        num_heads: usize,
        head_dim: usize,
        pipelines: &OpenCLPipelines,
    ) -> Result<Self> {
        let alloc = |size: usize| -> Result<Buffer<u8>> {
            unsafe {
                let buf =
                    Buffer::<u8>::create(&pipelines.context, CL_MEM_READ_WRITE, size, null_mut())?;
                Ok(buf)
            }
        };

        let total_seq = max_img_seq + max_txt_seq;
        let f32_bytes = 4;

        Ok(Self {
            img_buf: alloc(max_img_seq * dim * f32_bytes)?,
            txt_buf: alloc(max_txt_seq * dim * f32_bytes)?,
            merged_buf: alloc(total_seq * dim * f32_bytes)?,
            mod_buf: alloc(6 * dim * f32_bytes)?,
            q_buf: alloc(total_seq * num_heads * head_dim * f32_bytes)?,
            k_buf: alloc(total_seq * num_heads * head_dim * f32_bytes)?,
            v_buf: alloc(total_seq * num_heads * head_dim * f32_bytes)?,
            attn_out: alloc(total_seq * num_heads * head_dim * f32_bytes)?,
            mlp_buf: alloc(total_seq * mlp_sz * f32_bytes)?,
            vec_buf: alloc(dim * f32_bytes)?,
            pe_buf: alloc(total_seq * head_dim * 2 * f32_bytes)?, // pe is [seq, hd/2, 2, 2] = [seq, hd*2]
            silu_vec_buf: alloc(dim * f32_bytes)?,
            silu_vec_dirty: true,
            tmp_a: alloc(total_seq * dim * f32_bytes)?,
            tmp_b: alloc(total_seq * dim * f32_bytes)?,
            max_img_seq,
            max_txt_seq,
            dim,
        })
    }
}

// ==================== Weight Loading ====================

/// Helper: upload quantized raw bytes to an OpenCL buffer
fn upload_quantized(
    data: &[u8],
    rows: usize,
    cols: usize,
    fmt: OpenCLQuantFormat,
    pipelines: &OpenCLPipelines,
) -> Result<OclWeight> {
    unsafe {
        let mut buf =
            Buffer::<u8>::create(&pipelines.context, CL_MEM_READ_ONLY, data.len(), null_mut())?;
        pipelines
            .queue
            .enqueue_write_buffer(&mut buf, CL_BLOCKING, 0, data, &[])?;
        Ok(OclWeight::Quantized {
            buf,
            rows,
            cols,
            fmt,
        })
    }
}

/// Helper: upload dense F32 weight matrix to an OpenCL buffer
fn upload_f32_weight(
    data: &[f32],
    rows: usize,
    cols: usize,
    pipelines: &OpenCLPipelines,
) -> Result<OclWeight> {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    unsafe {
        let mut buf = Buffer::<u8>::create(
            &pipelines.context,
            CL_MEM_READ_ONLY,
            bytes.len(),
            null_mut(),
        )?;
        pipelines
            .queue
            .enqueue_write_buffer(&mut buf, CL_BLOCKING, 0, bytes, &[])?;
        Ok(OclWeight::Dense { buf, rows, cols })
    }
}

/// Helper: upload F32 data to an OpenCL buffer
fn upload_f32(data: &[f32], pipelines: &OpenCLPipelines) -> Result<F32Buf> {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    unsafe {
        let mut buf = Buffer::<u8>::create(
            &pipelines.context,
            CL_MEM_READ_ONLY,
            bytes.len(),
            null_mut(),
        )?;
        pipelines
            .queue
            .enqueue_write_buffer(&mut buf, CL_BLOCKING, 0, bytes, &[])?;
        Ok(F32Buf { buf })
    }
}

/// Load a QTensor as an OclWeight - Q4_0/Q4_K/Q6_K stay quantized, Q8_0/others get dequantized to F32.
fn load_weight_from_qtensor(
    qt: &crate::tensor::quantized::QTensor,
    rows: usize,
    cols: usize,
    pipelines: &OpenCLPipelines,
) -> Result<OclWeight> {
    use crate::tensor::quantized::GgmlDType;
    match qt.dtype() {
        GgmlDType::Q4_0 => {
            upload_quantized(&qt.data()?, rows, cols, OpenCLQuantFormat::Q4_0, pipelines)
        }
        GgmlDType::Q4K => {
            upload_quantized(&qt.data()?, rows, cols, OpenCLQuantFormat::Q4_K, pipelines)
        }
        GgmlDType::Q6K => {
            upload_quantized(&qt.data()?, rows, cols, OpenCLQuantFormat::Q6_K, pipelines)
        }
        other => {
            // Q8_0, F16, F32, etc. - dequantize to F32 on CPU and upload as dense
            tracing::debug!(
                "Dequantizing {:?} weight [{rows}x{cols}] to F32 for OpenCL",
                other
            );
            let f32_data = qt
                .dequantize(&crate::tensor::Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            upload_f32_weight(&f32_data, rows, cols, pipelines)
        }
    }
}

/// Load a linear layer (weight + optional bias) from GGUF content.
/// `prefix` is e.g. "double_blocks.0.img_attn.qkv"
fn load_qlinear(
    content: &crate::tensor::quantized::gguf_file::Content,
    reader: &mut std::io::Cursor<&[u8]>,
    prefix: &str,
    has_bias: bool,
    out_dim: usize,
    in_dim: usize,
    pipelines: &OpenCLPipelines,
) -> Result<OclQLinear> {
    let weight_name = format!("{}.weight", prefix);
    let qt = content.tensor(reader, &weight_name, &crate::tensor::Device::Cpu)?;
    let weight = load_weight_from_qtensor(&qt, out_dim, in_dim, pipelines)?;

    let bias = if has_bias {
        let bias_name = format!("{}.bias", prefix);
        let bias_tensor = content.tensor(reader, &bias_name, &crate::tensor::Device::Cpu)?;
        let bias_f32 = bias_tensor
            .dequantize(&crate::tensor::Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Some(upload_f32(&bias_f32, pipelines)?)
    } else {
        None
    };

    Ok(OclQLinear { weight, bias })
}

/// Load LayerNorm scale (Flux uses no-bias LayerNorm with ones init)
fn load_layer_norm(dim: usize, pipelines: &OpenCLPipelines) -> Result<OclLayerNorm> {
    // Flux LayerNorm is initialized with ones(dim) at construction time - not from GGUF.
    // The GGUF file doesn't store LayerNorm weights for Flux (they're always identity).
    let ones: Vec<f32> = vec![1.0f32; dim];
    let scale = upload_f32(&ones, pipelines)?;
    Ok(OclLayerNorm { scale, dim })
}

/// The query and key norms an attention publishes, query first.
///
/// Every attention in FLUX has this pair and spells it the same way - `norm.query_norm.scale`
/// and `norm.key_norm.scale` under the attention's own path - so the names and their order
/// are stated once. Written out at each of the three attentions instead, they are three
/// chances to read one of them out of turn, and a tensor read out of turn moves every tensor
/// after it.
fn load_qk_norm(
    content: &crate::tensor::quantized::gguf_file::Content,
    reader: &mut std::io::Cursor<&[u8]>,
    attn_prefix: &str,
    head_dim: usize,
    pipelines: &OpenCLPipelines,
) -> Result<OclQkNorm> {
    let query = format!("{attn_prefix}.norm.query_norm.scale");
    let key = format!("{attn_prefix}.norm.key_norm.scale");
    Ok(OclQkNorm {
        query_norm: load_rms_norm(content, reader, &query, head_dim, pipelines)?,
        key_norm: load_rms_norm(content, reader, &key, head_dim, pipelines)?,
    })
}

/// Load RmsNorm scale from GGUF
fn load_rms_norm(
    content: &crate::tensor::quantized::gguf_file::Content,
    reader: &mut std::io::Cursor<&[u8]>,
    name: &str,
    _dim: usize,
    pipelines: &OpenCLPipelines,
) -> Result<OclRmsNorm> {
    let qt = content.tensor(reader, name, &crate::tensor::Device::Cpu)?;
    let f32_data = qt
        .dequantize(&crate::tensor::Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let scale = upload_f32(&f32_data, pipelines)?;
    Ok(OclRmsNorm { scale })
}

// ==================== Block Loading ====================

impl OpenCLFluxDoubleBlock {
    /// Load a DoubleBlock from GGUF content to OpenCL buffers.
    /// `block_idx` is 0..18 (Flux Schnell has 19 double blocks, depth=19)
    pub fn from_gguf(
        content: &crate::tensor::quantized::gguf_file::Content,
        reader: &mut std::io::Cursor<&[u8]>,
        block_idx: usize,
        cfg: &crate::inference::model::flux::common::Config,
        pipelines: &OpenCLPipelines,
    ) -> Result<Self> {
        let prefix = format!("double_blocks.{}", block_idx);
        let dim = cfg.hidden_size;
        let num_heads = cfg.num_heads;
        let mlp_sz = (dim as f64 * cfg.mlp_ratio) as usize;
        let head_dim = dim / num_heads;

        info!(
            "Loading OpenCL DoubleBlock {} (dim={}, mlp_sz={}, heads={})",
            block_idx, dim, mlp_sz, num_heads
        );

        // Image modulation: silu -> linear(dim -> 6*dim)
        let img_mod = OclModulation2 {
            lin: load_qlinear(
                content,
                reader,
                &format!("{}.img_mod.lin", prefix),
                true,
                6 * dim,
                dim,
                pipelines,
            )?,
        };

        // Image attention
        let img_attn = OclSelfAttention {
            qkv: load_qlinear(
                content,
                reader,
                &format!("{}.img_attn.qkv", prefix),
                true,
                3 * dim,
                dim,
                pipelines,
            )?,
            proj: load_qlinear(
                content,
                reader,
                &format!("{}.img_attn.proj", prefix),
                true,
                dim,
                dim,
                pipelines,
            )?,
            qk_norm: load_qk_norm(
                content,
                reader,
                &format!("{prefix}.img_attn"),
                head_dim,
                pipelines,
            )?,
            num_heads,
        };

        // Image MLP: lin1(dim -> mlp_sz) -> GELU -> lin2(mlp_sz -> dim)
        let img_mlp = OclMlp {
            lin1: load_qlinear(
                content,
                reader,
                &format!("{}.img_mlp.0", prefix),
                true,
                mlp_sz,
                dim,
                pipelines,
            )?,
            lin2: load_qlinear(
                content,
                reader,
                &format!("{}.img_mlp.2", prefix),
                true,
                dim,
                mlp_sz,
                pipelines,
            )?,
        };

        // Image norms (identity LayerNorm - not stored in GGUF)
        let img_norm1 = load_layer_norm(dim, pipelines)?;
        let img_norm2 = load_layer_norm(dim, pipelines)?;

        // Text modulation
        let txt_mod = OclModulation2 {
            lin: load_qlinear(
                content,
                reader,
                &format!("{}.txt_mod.lin", prefix),
                true,
                6 * dim,
                dim,
                pipelines,
            )?,
        };

        // Text attention
        let txt_attn = OclSelfAttention {
            qkv: load_qlinear(
                content,
                reader,
                &format!("{}.txt_attn.qkv", prefix),
                true,
                3 * dim,
                dim,
                pipelines,
            )?,
            proj: load_qlinear(
                content,
                reader,
                &format!("{}.txt_attn.proj", prefix),
                true,
                dim,
                dim,
                pipelines,
            )?,
            qk_norm: load_qk_norm(
                content,
                reader,
                &format!("{prefix}.txt_attn"),
                head_dim,
                pipelines,
            )?,
            num_heads,
        };

        // Text MLP
        let txt_mlp = OclMlp {
            lin1: load_qlinear(
                content,
                reader,
                &format!("{}.txt_mlp.0", prefix),
                true,
                mlp_sz,
                dim,
                pipelines,
            )?,
            lin2: load_qlinear(
                content,
                reader,
                &format!("{}.txt_mlp.2", prefix),
                true,
                dim,
                mlp_sz,
                pipelines,
            )?,
        };

        // Text norms (identity LayerNorm)
        let txt_norm1 = load_layer_norm(dim, pipelines)?;
        let txt_norm2 = load_layer_norm(dim, pipelines)?;

        Ok(Self {
            img_mod,
            img_norm1,
            img_attn,
            img_norm2,
            img_mlp,
            txt_mod,
            txt_norm1,
            txt_attn,
            txt_norm2,
            txt_mlp,
            dim,
            mlp_sz,
        })
    }
}

impl OpenCLFluxSingleBlock {
    /// Load a SingleBlock from GGUF content to OpenCL buffers.
    /// `block_idx` is 0..37 (Flux Schnell has 38 single blocks)
    pub fn from_gguf(
        content: &crate::tensor::quantized::gguf_file::Content,
        reader: &mut std::io::Cursor<&[u8]>,
        block_idx: usize,
        cfg: &crate::inference::model::flux::common::Config,
        pipelines: &OpenCLPipelines,
    ) -> Result<Self> {
        let prefix = format!("single_blocks.{}", block_idx);
        let dim = cfg.hidden_size;
        let num_heads = cfg.num_heads;
        let mlp_sz = (dim as f64 * cfg.mlp_ratio) as usize;
        let head_dim = dim / num_heads;

        info!(
            "Loading OpenCL SingleBlock {} (dim={}, mlp_sz={})",
            block_idx, dim, mlp_sz
        );

        let linear1 = load_qlinear(
            content,
            reader,
            &format!("{}.linear1", prefix),
            true,
            3 * dim + mlp_sz,
            dim,
            pipelines,
        )?;
        let linear2 = load_qlinear(
            content,
            reader,
            &format!("{}.linear2", prefix),
            true,
            dim,
            dim + mlp_sz,
            pipelines,
        )?;
        let pre_norm = load_layer_norm(dim, pipelines)?;
        let modulation = OclModulation1 {
            lin: load_qlinear(
                content,
                reader,
                &format!("{}.modulation.lin", prefix),
                true,
                3 * dim,
                dim,
                pipelines,
            )?,
        };
        let qk_norm = load_qk_norm(content, reader, &prefix, head_dim, pipelines)?;

        Ok(Self {
            linear1,
            linear2,
            pre_norm,
            modulation,
            qk_norm,
            dim,
            mlp_sz,
            num_heads,
        })
    }
}

// ==================== Kernel Dispatch Helpers ====================

use opencl3::kernel::ExecuteKernel;

/// Dispatch matmul: output[seq_len, out_dim] = input[seq_len, in_dim] x weights^T
/// Selects Q4/Q6K dequant kernel or F32 dense kernel based on weight type.
fn ocl_mm(
    weights: &OclWeight,
    input: &Buffer<u8>,
    output: &Buffer<u8>,
    seq_len: usize,
    p: &OpenCLPipelines,
) -> Result<()> {
    match weights {
        OclWeight::Quantized {
            buf,
            rows,
            cols,
            fmt,
        } => {
            let kernel = match fmt {
                OpenCLQuantFormat::Q4_0 => &p.q4_matmul,
                OpenCLQuantFormat::Q4_K => &p.q4k_matmul,
                OpenCLQuantFormat::Q6_K => &p.q6k_matmul,
            };
            let num_outputs = seq_len * rows;
            unsafe {
                ExecuteKernel::new(kernel)
                    .set_arg(buf)
                    .set_arg(input)
                    .set_arg(output)
                    .set_arg(&(seq_len as u32))
                    .set_arg(&(*cols as u32))
                    .set_arg(&(*rows as u32))
                    .set_global_work_size(num_outputs * 64)
                    .set_local_work_size(64)
                    .enqueue_nd_range(&p.queue)?;
            }
        }
        OclWeight::Dense { buf, rows, cols } => {
            let num_outputs = seq_len * rows;
            unsafe {
                ExecuteKernel::new(&p.f32_matmul)
                    .set_arg(buf)
                    .set_arg(input)
                    .set_arg(output)
                    .set_arg(&(seq_len as u32))
                    .set_arg(&(*cols as u32))
                    .set_arg(&(*rows as u32))
                    .set_global_work_size(num_outputs * 64)
                    .set_local_work_size(64)
                    .enqueue_nd_range(&p.queue)?;
            }
        }
    }
    Ok(())
}

/// Matmul + bias add: output = matmul(input, weights^T) + bias
fn ocl_mm_bias(
    qlinear: &OclQLinear,
    input: &Buffer<u8>,
    output: &Buffer<u8>,
    seq_len: usize,
    p: &OpenCLPipelines,
) -> Result<()> {
    ocl_mm(&qlinear.weight, input, output, seq_len, p)?;
    if let Some(ref bias) = qlinear.bias {
        let total = (seq_len * qlinear.weight.rows()) as u32;
        let dim = qlinear.weight.rows() as u32;
        unsafe {
            ExecuteKernel::new(&p.broadcast_add)
                .set_arg(output)
                .set_arg(&bias.buf)
                .set_arg(output) // in-place
                .set_arg(&dim)
                .set_arg(&total)
                .set_global_work_size(total.div_ceil(4) as usize)
                .enqueue_nd_range(&p.queue)?;
        }
    }
    Ok(())
}

/// Copy a region from one buffer to another
unsafe fn copy_region(
    queue: &opencl3::command_queue::CommandQueue,
    src: &Buffer<u8>,
    dst: &mut Buffer<u8>,
    src_offset: usize,
    dst_offset: usize,
    size: usize,
) -> Result<()> {
    queue
        .enqueue_copy_buffer::<u8>(src, dst, src_offset, dst_offset, size, &[])
        .map_err(|e| anyhow::anyhow!("buffer copy: {}", e))?;
    Ok(())
}

/// Read F32 data from an OpenCL u8 buffer into a CPU `Vec<f32>`
unsafe fn read_f32(
    queue: &opencl3::command_queue::CommandQueue,
    buf: &Buffer<u8>,
    data: &mut [f32],
) -> Result<()> {
    let byte_slice = std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4);
    queue
        .enqueue_read_buffer(buf, CL_BLOCKING, 0, byte_slice, &[])
        .map_err(|e| anyhow::anyhow!("read_f32: {}", e))?;
    Ok(())
}

/// Write F32 data from CPU to an OpenCL u8 buffer
unsafe fn write_f32(
    queue: &opencl3::command_queue::CommandQueue,
    buf: &mut Buffer<u8>,
    data: &[f32],
) -> Result<()> {
    let byte_slice = std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4);
    queue
        .enqueue_write_buffer(buf, CL_BLOCKING, 0, byte_slice, &[])
        .map_err(|e| anyhow::anyhow!("write_f32: {}", e))?;
    Ok(())
}

// Note: a previous attempt reused silu_mul (out = silu(gate) * up) for the
// conditioning vector's plain-silu step. Flux's conditioning silu only runs
// on a [1, 3072] vector (tiny), so CPU is fine - no OpenCL kernel needed.

/// RmsNorm on Q or K head slices: applied per head of head_dim elements
fn dispatch_rms_norm_per_head(
    input: &Buffer<u8>,
    output: &Buffer<u8>,
    scale: &F32Buf,
    num_slices: usize, // total number of head slices (n_heads * seq_len)
    head_dim: usize,
    p: &OpenCLPipelines,
) -> Result<()> {
    // RmsNorm kernel: one workgroup (256 threads) per row of head_dim elements
    unsafe {
        ExecuteKernel::new(&p.rms_norm)
            .set_arg(input)
            .set_arg(output)
            .set_arg(&scale.buf)
            .set_arg(&(head_dim as u32))
            .set_arg(&1e-6f32)
            .set_global_work_size(num_slices * 256)
            .set_local_work_size(256)
            .enqueue_nd_range(&p.queue)?;
    }
    Ok(())
}

/// LayerNorm: output = (x - mean) / sqrt(var + eps) * scale
fn dispatch_layer_norm(
    input: &Buffer<u8>,
    output: &Buffer<u8>,
    scale: &OclLayerNorm,
    seq_len: usize,
    p: &OpenCLPipelines,
) -> Result<()> {
    unsafe {
        ExecuteKernel::new(&p.layer_norm_no_bias)
            .set_arg(input)
            .set_arg(output)
            .set_arg(&scale.scale.buf)
            .set_arg(&(scale.dim as u32))
            .set_arg(&1e-6f32)
            .set_global_work_size(seq_len * 256)
            .set_local_work_size(256)
            .enqueue_nd_range(&p.queue)?;
    }
    Ok(())
}

// ==================== Dispatch helpers for attention pipeline ====================

/// QKV split + transpose: [seq, 3*dim] -> Q,K,V each [n_heads, seq, head_dim]
fn dispatch_qkv_split(
    input: &Buffer<u8>,
    q: &Buffer<u8>,
    k: &Buffer<u8>,
    v: &Buffer<u8>,
    n_heads: usize,
    seq_len: usize,
    head_dim: usize,
    p: &OpenCLPipelines,
) -> Result<()> {
    let total = n_heads * seq_len * head_dim;
    unsafe {
        ExecuteKernel::new(&p.qkv_split)
            .set_arg(input)
            .set_arg(q)
            .set_arg(k)
            .set_arg(v)
            .set_arg(&(n_heads as u32))
            .set_arg(&(seq_len as u32))
            .set_arg(&(head_dim as u32))
            .set_global_work_size(total)
            .enqueue_nd_range(&p.queue)?;
    }
    Ok(())
}

/// Flux RoPE: apply 2x2 rotation from pe to Q or K
fn dispatch_flux_rope(
    input: &Buffer<u8>,
    pe: &Buffer<u8>,
    output: &Buffer<u8>,
    n_heads: usize,
    seq_len: usize,
    head_dim: usize,
    p: &OpenCLPipelines,
) -> Result<()> {
    let total_pairs = n_heads * seq_len * (head_dim / 2);
    unsafe {
        ExecuteKernel::new(&p.flux_rope)
            .set_arg(input)
            .set_arg(pe)
            .set_arg(output)
            .set_arg(&(n_heads as u32))
            .set_arg(&(seq_len as u32))
            .set_arg(&(head_dim as u32))
            .set_global_work_size(total_pairs)
            .enqueue_nd_range(&p.queue)?;
    }
    Ok(())
}

/// Concatenate two sequences: a[n_heads, seq_a, hd] + b[n_heads, seq_b, hd] -> out[n_heads, seq_a+seq_b, hd]
fn dispatch_concat_seq(
    a: &Buffer<u8>,
    b: &Buffer<u8>,
    output: &Buffer<u8>,
    n_heads: usize,
    seq_a: usize,
    seq_b: usize,
    head_dim: usize,
    p: &OpenCLPipelines,
) -> Result<()> {
    let total = n_heads * (seq_a + seq_b) * head_dim;
    unsafe {
        ExecuteKernel::new(&p.concat_seq)
            .set_arg(a)
            .set_arg(b)
            .set_arg(output)
            .set_arg(&(n_heads as u32))
            .set_arg(&(seq_a as u32))
            .set_arg(&(seq_b as u32))
            .set_arg(&(head_dim as u32))
            .set_global_work_size(total)
            .enqueue_nd_range(&p.queue)?;
    }
    Ok(())
}

/// Split + transpose: input[n_heads, total_seq, hd] -> a[seq_a, dim], b[seq_b, dim]
fn dispatch_split_heads(
    input: &Buffer<u8>,
    output_a: &Buffer<u8>,
    output_b: &Buffer<u8>,
    n_heads: usize,
    seq_a: usize,
    seq_b: usize,
    head_dim: usize,
    p: &OpenCLPipelines,
) -> Result<()> {
    let dim = n_heads * head_dim;
    let total = (seq_a + seq_b) * dim;
    unsafe {
        ExecuteKernel::new(&p.split_heads)
            .set_arg(input)
            .set_arg(output_a)
            .set_arg(output_b)
            .set_arg(&(n_heads as u32))
            .set_arg(&(seq_a as u32))
            .set_arg(&(seq_b as u32))
            .set_arg(&(head_dim as u32))
            .set_global_work_size(total)
            .enqueue_nd_range(&p.queue)?;
    }
    Ok(())
}

/// Dispatch image_attention (bidirectional, no causal mask)
fn dispatch_image_attention(
    q: &Buffer<u8>,
    k: &Buffer<u8>,
    v: &Buffer<u8>,
    output: &Buffer<u8>,
    batch: usize,
    n_heads: usize,
    seq_len: usize,
    head_dim: usize,
    p: &OpenCLPipelines,
) -> Result<()> {
    let total_work = batch * n_heads * seq_len;
    unsafe {
        ExecuteKernel::new(&p.image_attention)
            .set_arg(q)
            .set_arg(k)
            .set_arg(v)
            .set_arg(q) // mask placeholder (mask_present=0 so not read)
            .set_arg(output)
            .set_arg(&(batch as u32))
            .set_arg(&(n_heads as u32))
            .set_arg(&(n_heads as u32)) // n_kv_head = n_head (no GQA in Flux)
            .set_arg(&(seq_len as u32))
            .set_arg(&(head_dim as u32))
            .set_arg(&0u32) // mask_present = 0
            .set_global_work_size(total_work)
            .enqueue_nd_range(&p.queue)?;
    }
    Ok(())
}

/// Apply modulation scale_shift: output = (1 + scale) * normed + shift
/// mod_buf layout: [shift(dim), scale(dim), gate(dim), ...]
/// shift_offset and scale_offset are in F32 elements.
fn dispatch_scale_shift(
    normed: &Buffer<u8>,
    mod_buf: &Buffer<u8>,
    output: &Buffer<u8>,
    shift_offset_bytes: usize,
    scale_offset_bytes: usize,
    dim: usize,
    seq_len: usize,
    tmp_scale: &mut Buffer<u8>,
    p: &OpenCLPipelines,
) -> Result<()> {
    let total = (seq_len * dim) as u32;
    let d4 = dim * 4;
    // Copy scale to tmp
    unsafe {
        copy_region(&p.queue, mod_buf, tmp_scale, scale_offset_bytes, 0, d4)?;
    }
    // Apply (1+scale)*normed
    unsafe {
        ExecuteKernel::new(&p.scale_after_norm)
            .set_arg(normed)
            .set_arg(&*tmp_scale)
            .set_arg(output)
            .set_arg(&(dim as u32))
            .set_arg(&total)
            .set_global_work_size(total.div_ceil(4) as usize)
            .enqueue_nd_range(&p.queue)?;
    }
    // Copy shift to tmp and add
    unsafe {
        copy_region(&p.queue, mod_buf, tmp_scale, shift_offset_bytes, 0, d4)?;
    }
    unsafe {
        ExecuteKernel::new(&p.broadcast_add)
            .set_arg(output)
            .set_arg(&*tmp_scale)
            .set_arg(output)
            .set_arg(&(dim as u32))
            .set_arg(&total)
            .set_global_work_size(total.div_ceil(4) as usize)
            .enqueue_nd_range(&p.queue)?;
    }
    Ok(())
}

/// Dispatch gated residual: output = residual + tanh(gate) * y
fn dispatch_gated_residual(
    residual: &Buffer<u8>,
    y: &Buffer<u8>,
    gate_buf: &Buffer<u8>,
    gate_offset_bytes: usize,
    output: &Buffer<u8>,
    dim: usize,
    seq_len: usize,
    tmp_gate: &mut Buffer<u8>,
    p: &OpenCLPipelines,
) -> Result<()> {
    let total = (seq_len * dim) as u32;
    let d4 = dim * 4;
    // Copy gate to tmp
    unsafe {
        copy_region(&p.queue, gate_buf, tmp_gate, gate_offset_bytes, 0, d4)?;
    }
    unsafe {
        ExecuteKernel::new(&p.gated_residual)
            .set_arg(residual)
            .set_arg(y)
            .set_arg(&*tmp_gate)
            .set_arg(output)
            .set_arg(&(dim as u32))
            .set_arg(&total)
            .set_global_work_size(total.div_ceil(4) as usize)
            .enqueue_nd_range(&p.queue)?;
    }
    Ok(())
}

/// Dispatch GELU activation: `output[i] = x * 0.5 * (1 + erf(x / sqrt(2)))`
fn dispatch_gelu(
    input: &Buffer<u8>,
    output: &Buffer<u8>,
    total_elements: usize,
    p: &OpenCLPipelines,
) -> Result<()> {
    unsafe {
        ExecuteKernel::new(&p.gelu)
            .set_arg(input)
            .set_arg(output)
            .set_arg(&(total_elements as u32))
            .set_global_work_size(total_elements.div_ceil(4))
            .enqueue_nd_range(&p.queue)?;
    }
    Ok(())
}

// ==================== Forward Pass ====================

impl OpenCLFluxDoubleBlock {
    /// Forward pass for DoubleBlock on OpenCL.
    ///
    /// All inputs/outputs are in scratch buffers:
    ///   scratch.img_buf: [img_seq, dim] - updated in-place
    ///   scratch.txt_buf: [txt_seq, dim] - updated in-place
    ///   scratch.vec_buf: [1, dim] - conditioning vector (read-only)
    ///   scratch.pe_buf: [total_seq, head_dim/2, 2, 2] - RoPE PE (read-only)
    pub fn forward(
        &self,
        img_seq: usize,
        txt_seq: usize,
        scratch: &mut OclFluxScratch,
        p: &OpenCLPipelines,
    ) -> Result<()> {
        let dim = self.dim;
        let n_heads = self.img_attn.num_heads;
        let head_dim = dim / n_heads;
        let total_seq = txt_seq + img_seq;
        let d4 = dim * 4;

        // -- 1. silu(vec) - cached in scratch.silu_vec_buf for the whole step --
        scratch.ensure_silu_vec(p)?;

        // -- 2. Image modulation: silu_vec -> [1, 6*dim] --
        ocl_mm_bias(
            &self.img_mod.lin,
            &scratch.silu_vec_buf,
            &scratch.mod_buf,
            1,
            p,
        )?;

        // -- 3. Text modulation: silu_vec -> [1, 6*dim] -> tmp_b --
        ocl_mm_bias(
            &self.txt_mod.lin,
            &scratch.silu_vec_buf,
            &scratch.tmp_b,
            1,
            p,
        )?;

        // -- 4. Image path: LayerNorm -> scale_shift -> QKV --
        // LayerNorm reads img_buf, writes to merged_buf (can't read+write same buffer)
        dispatch_layer_norm(
            &scratch.img_buf,
            &scratch.merged_buf,
            &self.img_norm1,
            img_seq,
            p,
        )?;
        dispatch_scale_shift(
            &scratch.merged_buf,
            &scratch.mod_buf,
            &scratch.merged_buf,
            0,  // shift1 at offset 0
            d4, // scale1 at offset dim
            dim,
            img_seq,
            &mut scratch.q_buf,
            p,
        )?;
        // img QKV: [img_seq, 3*dim] -> tmp_a (silu is done)
        ocl_mm_bias(
            &self.img_attn.qkv,
            &scratch.merged_buf,
            &scratch.tmp_a,
            img_seq,
            p,
        )?;
        // Split -> img_q, img_k, img_v
        dispatch_qkv_split(
            &scratch.tmp_a,
            &scratch.q_buf,
            &scratch.k_buf,
            &scratch.v_buf,
            n_heads,
            img_seq,
            head_dim,
            p,
        )?;
        // QK norm (RmsNorm per head)
        let img_head_slices = n_heads * img_seq;
        dispatch_rms_norm_per_head(
            &scratch.q_buf,
            &scratch.q_buf,
            &self.img_attn.qk_norm.query_norm.scale,
            img_head_slices,
            head_dim,
            p,
        )?;
        dispatch_rms_norm_per_head(
            &scratch.k_buf,
            &scratch.k_buf,
            &self.img_attn.qk_norm.key_norm.scale,
            img_head_slices,
            head_dim,
            p,
        )?;

        // -- 5. Text path: LayerNorm -> scale_shift -> QKV --
        dispatch_layer_norm(
            &scratch.txt_buf,
            &scratch.merged_buf,
            &self.txt_norm1,
            txt_seq,
            p,
        )?;
        dispatch_scale_shift(
            &scratch.merged_buf,
            &scratch.tmp_b,
            &scratch.merged_buf,
            0,
            d4,
            dim,
            txt_seq,
            &mut scratch.tmp_a,
            p,
        )?;
        // txt QKV: [txt_seq, 3*dim] -> tmp_a (reused)
        ocl_mm_bias(
            &self.txt_attn.qkv,
            &scratch.merged_buf,
            &scratch.tmp_a,
            txt_seq,
            p,
        )?;
        // Split -> txt Q,K,V into attn_out, mlp_buf, merged_buf (temp usage)
        dispatch_qkv_split(
            &scratch.tmp_a,
            &scratch.attn_out,
            &scratch.mlp_buf,
            &scratch.merged_buf,
            n_heads,
            txt_seq,
            head_dim,
            p,
        )?;
        // QK norm
        let txt_head_slices = n_heads * txt_seq;
        dispatch_rms_norm_per_head(
            &scratch.attn_out,
            &scratch.attn_out,
            &self.txt_attn.qk_norm.query_norm.scale,
            txt_head_slices,
            head_dim,
            p,
        )?;
        dispatch_rms_norm_per_head(
            &scratch.mlp_buf,
            &scratch.mlp_buf,
            &self.txt_attn.qk_norm.key_norm.scale,
            txt_head_slices,
            head_dim,
            p,
        )?;

        // -- 6. Concatenate [txt, img] Q,K,V for joint attention --
        // Q: concat(txt_q=attn_out, img_q=q_buf) -> tmp_a [n_heads, total_seq, head_dim]
        dispatch_concat_seq(
            &scratch.attn_out,
            &scratch.q_buf,
            &scratch.tmp_a,
            n_heads,
            txt_seq,
            img_seq,
            head_dim,
            p,
        )?;
        // K: concat(txt_k=mlp_buf, img_k=k_buf) -> attn_out (reuse)
        dispatch_concat_seq(
            &scratch.mlp_buf,
            &scratch.k_buf,
            &scratch.attn_out,
            n_heads,
            txt_seq,
            img_seq,
            head_dim,
            p,
        )?;
        // V: concat(txt_v=merged_buf, img_v=v_buf) -> mlp_buf (reuse)
        dispatch_concat_seq(
            &scratch.merged_buf,
            &scratch.v_buf,
            &scratch.mlp_buf,
            n_heads,
            txt_seq,
            img_seq,
            head_dim,
            p,
        )?;

        // -- 7. Apply Flux RoPE to Q and K --
        dispatch_flux_rope(
            &scratch.tmp_a,
            &scratch.pe_buf,
            &scratch.q_buf,
            n_heads,
            total_seq,
            head_dim,
            p,
        )?;
        dispatch_flux_rope(
            &scratch.attn_out,
            &scratch.pe_buf,
            &scratch.k_buf,
            n_heads,
            total_seq,
            head_dim,
            p,
        )?;

        // -- 8. Bidirectional attention: Q,K,V -> attn_out [n_heads, total_seq, head_dim] --
        dispatch_image_attention(
            &scratch.q_buf,
            &scratch.k_buf,
            &scratch.mlp_buf,
            &scratch.attn_out,
            1,
            n_heads,
            total_seq,
            head_dim,
            p,
        )?;

        // -- 9. Save original hidden states before split_heads overwrites them --
        // After attention, v_buf and tmp_a are free (were used for V and Q concat)
        unsafe {
            copy_region(
                &p.queue,
                &scratch.img_buf,
                &mut scratch.v_buf,
                0,
                0,
                img_seq * dim * 4,
            )?;
            copy_region(
                &p.queue,
                &scratch.txt_buf,
                &mut scratch.tmp_a,
                0,
                0,
                txt_seq * dim * 4,
            )?;
        }

        // -- 10. Split attention output: [n_heads, total_seq, hd] -> txt_attn, img_attn --
        dispatch_split_heads(
            &scratch.attn_out,
            &scratch.txt_buf,
            &scratch.img_buf,
            n_heads,
            txt_seq,
            img_seq,
            head_dim,
            p,
        )?;
        // Now: img_buf = img attn output, txt_buf = txt attn output
        // Saved: v_buf = original img, tmp_a = original txt

        // -- 11. Attention projection + gated residual --
        // img: proj(img_attn) -> merged_buf, then img = original + gate1 * proj
        ocl_mm_bias(
            &self.img_attn.proj,
            &scratch.img_buf,
            &scratch.merged_buf,
            img_seq,
            p,
        )?;
        dispatch_gated_residual(
            &scratch.v_buf,
            &scratch.merged_buf,
            &scratch.mod_buf,
            2 * d4,
            &scratch.img_buf,
            dim,
            img_seq,
            &mut scratch.q_buf,
            p,
        )?;

        // txt: proj(txt_attn) -> merged_buf, then txt = original + gate1 * proj
        ocl_mm_bias(
            &self.txt_attn.proj,
            &scratch.txt_buf,
            &scratch.merged_buf,
            txt_seq,
            p,
        )?;
        dispatch_gated_residual(
            &scratch.tmp_a,
            &scratch.merged_buf,
            &scratch.tmp_b,
            2 * d4,
            &scratch.txt_buf,
            dim,
            txt_seq,
            &mut scratch.q_buf,
            p,
        )?;

        // -- 12. MLP branch: LayerNorm -> scale_shift(mod2) -> MLP(GELU) -> gated residual --
        // Image MLP
        dispatch_layer_norm(
            &scratch.img_buf,
            &scratch.merged_buf,
            &self.img_norm2,
            img_seq,
            p,
        )?;
        dispatch_scale_shift(
            &scratch.merged_buf,
            &scratch.mod_buf,
            &scratch.merged_buf,
            3 * d4, // shift2 at offset 3*dim
            4 * d4, // scale2 at offset 4*dim
            dim,
            img_seq,
            &mut scratch.q_buf,
            p,
        )?;
        ocl_mm_bias(
            &self.img_mlp.lin1,
            &scratch.merged_buf,
            &scratch.mlp_buf,
            img_seq,
            p,
        )?;
        dispatch_gelu(&scratch.mlp_buf, &scratch.mlp_buf, img_seq * self.mlp_sz, p)?;
        ocl_mm_bias(
            &self.img_mlp.lin2,
            &scratch.mlp_buf,
            &scratch.tmp_a,
            img_seq,
            p,
        )?;
        // img = img + gate2 * mlp_output
        dispatch_gated_residual(
            &scratch.img_buf,
            &scratch.tmp_a,
            &scratch.mod_buf,
            5 * d4,
            &scratch.img_buf,
            dim,
            img_seq,
            &mut scratch.q_buf,
            p,
        )?;

        // Text MLP
        dispatch_layer_norm(
            &scratch.txt_buf,
            &scratch.merged_buf,
            &self.txt_norm2,
            txt_seq,
            p,
        )?;
        dispatch_scale_shift(
            &scratch.merged_buf,
            &scratch.tmp_b,
            &scratch.merged_buf,
            3 * d4,
            4 * d4,
            dim,
            txt_seq,
            &mut scratch.q_buf,
            p,
        )?;
        ocl_mm_bias(
            &self.txt_mlp.lin1,
            &scratch.merged_buf,
            &scratch.mlp_buf,
            txt_seq,
            p,
        )?;
        dispatch_gelu(&scratch.mlp_buf, &scratch.mlp_buf, txt_seq * self.mlp_sz, p)?;
        ocl_mm_bias(
            &self.txt_mlp.lin2,
            &scratch.mlp_buf,
            &scratch.tmp_a,
            txt_seq,
            p,
        )?;
        // txt = txt + gate2 * mlp_output
        dispatch_gated_residual(
            &scratch.txt_buf,
            &scratch.tmp_a,
            &scratch.tmp_b,
            5 * d4,
            &scratch.txt_buf,
            dim,
            txt_seq,
            &mut scratch.q_buf,
            p,
        )?;

        p.queue
            .finish()
            .map_err(|e| anyhow::anyhow!("queue finish: {}", e))?;
        Ok(())
    }
}

impl OpenCLFluxSingleBlock {
    /// Forward pass for SingleBlock on OpenCL.
    ///
    /// All inputs/outputs in scratch buffers:
    ///   scratch.merged_buf: [seq_len, dim] - updated in-place
    ///   scratch.vec_buf: [1, dim] - conditioning vector (read-only)
    ///   scratch.pe_buf: [seq_len, head_dim/2, 2, 2] - RoPE PE (read-only)
    pub fn forward(
        &self,
        seq_len: usize,
        scratch: &mut OclFluxScratch,
        p: &OpenCLPipelines,
    ) -> Result<()> {
        let dim = self.dim;
        let n_heads = self.num_heads;
        let head_dim = dim / n_heads;
        let d4 = dim * 4;

        // -- 1. Modulation: silu(vec) -> linear -> [1, 3*dim] -> (shift, scale, gate) --
        // silu(vec) is shared across all blocks in this step (cached).
        scratch.ensure_silu_vec(p)?;
        ocl_mm_bias(
            &self.modulation.lin,
            &scratch.silu_vec_buf,
            &scratch.mod_buf,
            1,
            p,
        )?;
        // mod_buf: [shift(dim), scale(dim), gate(dim)]

        // -- 2. LayerNorm -> scale_shift -> linear1 --
        dispatch_layer_norm(
            &scratch.merged_buf,
            &scratch.tmp_b,
            &self.pre_norm,
            seq_len,
            p,
        )?;
        dispatch_scale_shift(
            &scratch.tmp_b,
            &scratch.mod_buf,
            &scratch.tmp_b,
            0,
            d4,
            dim,
            seq_len,
            &mut scratch.tmp_a,
            p,
        )?;

        // linear1: [seq, dim] -> [seq, 3*dim + mlp_sz]
        ocl_mm_bias(
            &self.linear1,
            &scratch.tmp_b,
            &scratch.merged_buf,
            seq_len,
            p,
        )?;

        // -- 3. QKV split from first 3*dim columns (strided rows) --
        // merged_buf layout per row: [qkv(3*dim), mlp(mlp_sz)]
        // Use qkv_split_strided to handle row_stride directly (single kernel dispatch)
        {
            let row_stride = 3 * dim + self.mlp_sz;
            let total = n_heads * seq_len * head_dim;
            unsafe {
                ExecuteKernel::new(&p.qkv_split_strided)
                    .set_arg(&scratch.merged_buf)
                    .set_arg(&scratch.q_buf)
                    .set_arg(&scratch.k_buf)
                    .set_arg(&scratch.v_buf)
                    .set_arg(&(n_heads as u32))
                    .set_arg(&(seq_len as u32))
                    .set_arg(&(head_dim as u32))
                    .set_arg(&(row_stride as u32))
                    .set_global_work_size(total)
                    .enqueue_nd_range(&p.queue)?;
            }
        }

        // QK norm
        let head_slices = n_heads * seq_len;
        dispatch_rms_norm_per_head(
            &scratch.q_buf,
            &scratch.q_buf,
            &self.qk_norm.query_norm.scale,
            head_slices,
            head_dim,
            p,
        )?;
        dispatch_rms_norm_per_head(
            &scratch.k_buf,
            &scratch.k_buf,
            &self.qk_norm.key_norm.scale,
            head_slices,
            head_dim,
            p,
        )?;

        // -- 4. Flux RoPE --
        dispatch_flux_rope(
            &scratch.q_buf,
            &scratch.pe_buf,
            &scratch.q_buf,
            n_heads,
            seq_len,
            head_dim,
            p,
        )?;
        dispatch_flux_rope(
            &scratch.k_buf,
            &scratch.pe_buf,
            &scratch.k_buf,
            n_heads,
            seq_len,
            head_dim,
            p,
        )?;

        // -- 5. Attention --
        dispatch_image_attention(
            &scratch.q_buf,
            &scratch.k_buf,
            &scratch.v_buf,
            &scratch.attn_out,
            1,
            n_heads,
            seq_len,
            head_dim,
            p,
        )?;

        // -- 6. Transpose attention output: [n_heads, seq, head_dim] -> [seq, dim] --
        // split_heads with seq_a=seq_len, seq_b=0 acts as pure head-transpose
        // Actually we can do this more simply - use split_heads with output_a only
        // But split_heads needs seq_a + seq_b. Let's just copy to tmp_a as [seq, dim].
        // Use a trivial split: seq_a=seq_len, seq_b=0
        dispatch_split_heads(
            &scratch.attn_out,
            &scratch.tmp_a,
            &scratch.tmp_b,
            n_heads,
            seq_len,
            0,
            head_dim,
            p,
        )?;
        // tmp_a now has attention output as [seq_len, dim]

        // -- 7. GELU on mlp portion --
        // Extract mlp columns from merged_buf using strided_slice, then apply GELU
        {
            let row_stride = 3 * dim + self.mlp_sz;
            let mlp_total = seq_len * self.mlp_sz;
            unsafe {
                ExecuteKernel::new(&p.strided_slice)
                    .set_arg(&scratch.merged_buf)
                    .set_arg(&scratch.mlp_buf)
                    .set_arg(&(row_stride as u32))
                    .set_arg(&((3 * dim) as u32)) // col_offset
                    .set_arg(&(self.mlp_sz as u32)) // width
                    .set_arg(&(mlp_total as u32))
                    .set_global_work_size(mlp_total)
                    .enqueue_nd_range(&p.queue)?;
            }
        }
        dispatch_gelu(&scratch.mlp_buf, &scratch.mlp_buf, seq_len * self.mlp_sz, p)?;

        // -- 8. Concat [attn, gelu_mlp] -> linear2: [seq, dim+mlp_sz] -> [seq, dim] --
        {
            let concat_total = seq_len * (dim + self.mlp_sz);
            unsafe {
                ExecuteKernel::new(&p.concat_cols)
                    .set_arg(&scratch.tmp_a) // attn output [seq, dim]
                    .set_arg(&scratch.mlp_buf) // gelu mlp [seq, mlp_sz]
                    .set_arg(&scratch.tmp_b) // output [seq, dim+mlp_sz]
                    .set_arg(&(dim as u32)) // width_a
                    .set_arg(&(self.mlp_sz as u32)) // width_b
                    .set_arg(&(concat_total as u32))
                    .set_global_work_size(concat_total)
                    .enqueue_nd_range(&p.queue)?;
            }
        }
        ocl_mm_bias(&self.linear2, &scratch.tmp_b, &scratch.tmp_a, seq_len, p)?;

        // -- 9. Gated residual: seq = seq + gate * output --
        // gate at offset 2*dim in mod_buf
        dispatch_gated_residual(
            &scratch.merged_buf,
            &scratch.tmp_a,
            &scratch.mod_buf,
            2 * d4,
            &scratch.merged_buf,
            dim,
            seq_len,
            &mut scratch.q_buf,
            p,
        )?;

        p.queue
            .finish()
            .map_err(|e| anyhow::anyhow!("queue finish: {}", e))?;

        Ok(())
    }
}
