//! OpenCL layer implementation for Intel Arc / AMD GPU inference.
//!
//! Provides Q4 weight storage on GPU with on-the-fly dequantization via OpenCL kernels.
//! Native OpenCL C kernels, which is what reaches XMX on an Intel Arc card.

#![allow(clippy::too_many_arguments)]

use anyhow::Result;
#[cfg(feature = "opencl")]
use opencl3::{
    command_queue::CommandQueue, context::Context, device::Device, kernel::Kernel, memory::Buffer,
    memory::CL_MEM_READ_WRITE,
};
use std::sync::Arc;

/// Pre-allocated scratch buffers for forward pass intermediates.
/// Avoids 12 GPU allocations per layer per forward call.
#[derive(Debug)]
pub struct OpenCLScratch {
    normed: Buffer<f32>,
    q_buf: Buffer<f32>,
    k_buf: Buffer<f32>,
    v_buf: Buffer<f32>,
    attn_out: Buffer<f32>,
    proj: Buffer<f32>,
    residual1: Buffer<f32>,
    ffn_in: Buffer<f32>,
    gate: Buffer<f32>,
    up_buf: Buffer<f32>,
    act: Buffer<f32>,
    ffn_out: Buffer<f32>,
    /// Sequence length these buffers were allocated for
    alloc_seq_len: usize,
}

impl OpenCLScratch {
    fn allocate(
        seq_len: usize,
        hidden_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        ffn_dim: usize,
        ctx: &Context,
    ) -> Result<Self> {
        use std::ptr::null_mut;
        let n = seq_len * hidden_dim;
        let q_dim = n_head * head_dim;
        let kv_dim = n_kv_head * head_dim;

        let alloc = |size: usize| -> Result<Buffer<f32>> {
            unsafe {
                Buffer::<f32>::create(ctx, CL_MEM_READ_WRITE, size, null_mut())
                    .map_err(|e| anyhow::anyhow!("scratch alloc failed: {}", e))
            }
        };

        Ok(Self {
            normed: alloc(n)?,
            q_buf: alloc(seq_len * q_dim)?,
            k_buf: alloc(seq_len * kv_dim)?,
            v_buf: alloc(seq_len * kv_dim)?,
            attn_out: alloc(seq_len * q_dim)?,
            proj: alloc(n)?,
            residual1: alloc(n)?,
            ffn_in: alloc(n)?,
            gate: alloc(seq_len * ffn_dim)?,
            up_buf: alloc(seq_len * ffn_dim)?,
            act: alloc(seq_len * ffn_dim)?,
            ffn_out: alloc(n)?,
            alloc_seq_len: seq_len,
        })
    }
}

/// Quantization format for OpenCL layers
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(non_camel_case_types)]
pub enum OpenCLQuantFormat {
    Q4_0, // 18 bytes per 32 elements
    Q4_K, // 144 bytes per 256 elements
    Q6_K, // 210 bytes per 256 elements
}

/// OpenCL layer weights - Q4 bytes on GPU, dequantized on-the-fly
#[derive(Debug)]
pub struct OpenCLLayerWeights {
    /// Q4 weight buffers - raw bytes from GGUF, uploaded to GPU
    pub wq_q4: Buffer<u8>,
    pub wk_q4: Buffer<u8>,
    pub wv_q4: Buffer<u8>,
    pub wo_q4: Buffer<u8>,
    pub ffn_gate_q4: Buffer<u8>,
    pub ffn_up_q4: Buffer<u8>,
    pub ffn_down_q4: Buffer<u8>,
    /// Norm scales (f32)
    pub attn_norm: Buffer<f32>,
    pub ffn_norm: Buffer<f32>,
    /// RoPE tables (shared via Arc)
    pub cos: Arc<Buffer<f32>>,
    pub sin: Arc<Buffer<f32>>,
    /// KV cache (f32 - [max_seq, n_kv_head, head_dim])
    pub kv_k: Buffer<f32>,
    pub kv_v: Buffer<f32>,
    pub cached_len: usize,
    /// Per-weight quantization formats (determines which matmul kernel to use)
    pub fmt_wq: OpenCLQuantFormat,
    pub fmt_wk: OpenCLQuantFormat,
    pub fmt_wv: OpenCLQuantFormat,
    pub fmt_wo: OpenCLQuantFormat,
    pub fmt_gate: OpenCLQuantFormat,
    pub fmt_up: OpenCLQuantFormat,
    pub fmt_down: OpenCLQuantFormat,
    /// Dims
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub max_seq_len: usize,
    pub sliding_window: u32,
    /// Reusable scratch buffers (allocated lazily, resized when seq_len changes)
    scratch: Option<OpenCLScratch>,
}

impl OpenCLLayerWeights {
    /// Ensure scratch buffers are allocated for the given seq_len.
    /// Reuses existing buffers if already the right size.
    fn ensure_scratch(&mut self, seq_len: usize, ctx: &Context) -> Result<()> {
        let needs_alloc = match &self.scratch {
            Some(s) => s.alloc_seq_len != seq_len,
            None => true,
        };
        if needs_alloc {
            self.scratch = Some(OpenCLScratch::allocate(
                seq_len,
                self.hidden_dim,
                self.n_head,
                self.n_kv_head,
                self.head_dim,
                self.ffn_dim,
                ctx,
            )?);
        }
        Ok(())
    }

    /// Forward pass: 16-step transformer layer, allocates output buffer.
    pub fn forward(
        &mut self,
        x_buf: &Buffer<f32>,
        seq_len: usize,
        index_pos: usize,
        p: &OpenCLPipelines,
    ) -> Result<Buffer<f32>> {
        use opencl3::memory::CL_MEM_READ_WRITE;
        use std::ptr::null_mut;

        let n = seq_len * self.hidden_dim;
        let output = unsafe {
            Buffer::<f32>::create(&p.context, CL_MEM_READ_WRITE, n, null_mut())
                .map_err(|e| anyhow::anyhow!("output alloc failed: {}", e))?
        };
        self.forward_into(x_buf, &output, seq_len, index_pos, p)?;
        Ok(output)
    }

    /// Forward pass that writes output to a caller-provided buffer (zero allocation).
    /// Used by ping-pong buffer scheme during generation (seq_len==1).
    pub fn forward_into(
        &mut self,
        x_buf: &Buffer<f32>,
        out_buf: &Buffer<f32>,
        seq_len: usize,
        index_pos: usize,
        p: &OpenCLPipelines,
    ) -> Result<()> {
        use opencl3::kernel::ExecuteKernel;

        let start = std::time::Instant::now();
        let ctx = &*p.context;
        let q = &*p.queue;
        let n = seq_len * self.hidden_dim;
        let q_dim = self.n_head * self.head_dim;
        let kv_dim = self.n_kv_head * self.head_dim;

        self.ensure_scratch(seq_len, ctx)?;
        let scratch = self.scratch.as_ref().unwrap();

        let mut kernel_count = 0;

        macro_rules! q4mm {
            ($input:expr, $weights:expr, $out_buf:expr, $in_dim:expr, $out_dim:expr, $fmt:expr) => {{
                let num_outputs = seq_len * $out_dim;
                kernel_count += 1;
                let kernel = match $fmt {
                    OpenCLQuantFormat::Q4_0 => &p.q4_matmul,
                    OpenCLQuantFormat::Q4_K => &p.q4k_matmul,
                    OpenCLQuantFormat::Q6_K => &p.q6k_matmul,
                };
                unsafe {
                    ExecuteKernel::new(kernel)
                        .set_arg($weights)
                        .set_arg($input)
                        .set_arg($out_buf)
                        .set_arg(&(seq_len as u32))
                        .set_arg(&($in_dim as u32))
                        .set_arg(&($out_dim as u32))
                        .set_global_work_size(num_outputs * 64)
                        .set_local_work_size(64)
                        .enqueue_nd_range(q)?;
                }
            }};
        }

        // 1. Attention RMSNorm
        kernel_count += 1;
        unsafe {
            ExecuteKernel::new(&p.rms_norm)
                .set_arg(x_buf)
                .set_arg(&scratch.normed)
                .set_arg(&self.attn_norm)
                .set_arg(&(self.hidden_dim as u32))
                .set_arg(&1e-6f32)
                .set_global_work_size(seq_len * 256)
                .set_local_work_size(256)
                .enqueue_nd_range(q)?;
        }

        // 2-4. Q, K, V projections
        q4mm!(
            &scratch.normed,
            &self.wq_q4,
            &scratch.q_buf,
            self.hidden_dim,
            q_dim,
            self.fmt_wq
        );
        q4mm!(
            &scratch.normed,
            &self.wk_q4,
            &scratch.k_buf,
            self.hidden_dim,
            kv_dim,
            self.fmt_wk
        );
        q4mm!(
            &scratch.normed,
            &self.wv_q4,
            &scratch.v_buf,
            self.hidden_dim,
            kv_dim,
            self.fmt_wv
        );

        // 5-6. Fused RoPE on Q and K (single dispatch for both)
        kernel_count += 1;
        unsafe {
            let total_heads = self.n_head + self.n_kv_head;
            ExecuteKernel::new(&p.dual_rope)
                .set_arg(&scratch.q_buf)
                .set_arg(&scratch.k_buf)
                .set_arg(&*self.cos)
                .set_arg(&*self.sin)
                .set_arg(&(seq_len as u32))
                .set_arg(&(self.n_head as u32))
                .set_arg(&(self.n_kv_head as u32))
                .set_arg(&(self.head_dim as u32))
                .set_arg(&(index_pos as u32))
                .set_global_work_size(seq_len * total_heads * self.head_dim / 2)
                .enqueue_nd_range(q)?;
        }

        // 7. Fused append K and V to cache (single dispatch for both)
        kernel_count += 1;
        unsafe {
            ExecuteKernel::new(&p.dual_append_kv)
                .set_arg(&scratch.k_buf)
                .set_arg(&scratch.v_buf)
                .set_arg(&self.kv_k)
                .set_arg(&self.kv_v)
                .set_arg(&(self.n_kv_head as u32))
                .set_arg(&(self.head_dim as u32))
                .set_arg(&(self.cached_len as u32))
                .set_arg(&(seq_len as u32))
                .set_global_work_size(seq_len * self.n_kv_head * self.head_dim)
                .enqueue_nd_range(q)?;
        }
        let new_cache_len = self.cached_len + seq_len;

        // 8. Attention (online softmax)
        kernel_count += 1;
        unsafe {
            ExecuteKernel::new(&p.attention)
                .set_arg(&scratch.q_buf)
                .set_arg(&self.kv_k)
                .set_arg(&self.kv_v)
                .set_arg(&scratch.attn_out)
                .set_arg(&(self.n_head as u32))
                .set_arg(&(self.n_kv_head as u32))
                .set_arg(&(seq_len as u32))
                .set_arg(&(new_cache_len as u32))
                .set_arg(&(self.head_dim as u32))
                .set_arg(&self.sliding_window)
                .set_global_work_size(seq_len * self.n_head)
                .enqueue_nd_range(q)?;
        }

        // 9-10. Output projection + residual add (fused for Q4_K)
        if self.fmt_wo == OpenCLQuantFormat::Q4_K {
            let num_outputs = seq_len * self.hidden_dim;
            kernel_count += 1;
            unsafe {
                ExecuteKernel::new(&p.q4k_matmul_add)
                    .set_arg(&self.wo_q4)
                    .set_arg(&scratch.attn_out)
                    .set_arg(x_buf)
                    .set_arg(&scratch.residual1)
                    .set_arg(&(seq_len as u32))
                    .set_arg(&(q_dim as u32))
                    .set_arg(&(self.hidden_dim as u32))
                    .set_global_work_size(num_outputs * 64)
                    .set_local_work_size(64)
                    .enqueue_nd_range(q)?;
            }
        } else {
            q4mm!(
                &scratch.attn_out,
                &self.wo_q4,
                &scratch.proj,
                q_dim,
                self.hidden_dim,
                self.fmt_wo
            );
            kernel_count += 1;
            unsafe {
                ExecuteKernel::new(&p.add)
                    .set_arg(x_buf)
                    .set_arg(&scratch.proj)
                    .set_arg(&scratch.residual1)
                    .set_arg(&(n as u32))
                    .set_global_work_size(n.div_ceil(4))
                    .enqueue_nd_range(q)?;
            }
        }

        // 11. FFN RMSNorm
        kernel_count += 1;
        unsafe {
            ExecuteKernel::new(&p.rms_norm)
                .set_arg(&scratch.residual1)
                .set_arg(&scratch.ffn_in)
                .set_arg(&self.ffn_norm)
                .set_arg(&(self.hidden_dim as u32))
                .set_arg(&1e-6f32)
                .set_global_work_size(seq_len * 256)
                .set_local_work_size(256)
                .enqueue_nd_range(q)?;
        }

        // 12-14. FFN gate + up + SiLU_mul (fused for Q4_K)
        if self.fmt_gate == OpenCLQuantFormat::Q4_K && self.fmt_up == OpenCLQuantFormat::Q4_K {
            let num_outputs = seq_len * self.ffn_dim;
            kernel_count += 1;
            unsafe {
                ExecuteKernel::new(&p.q4k_gate_up_silu)
                    .set_arg(&self.ffn_gate_q4)
                    .set_arg(&self.ffn_up_q4)
                    .set_arg(&scratch.ffn_in)
                    .set_arg(&scratch.act)
                    .set_arg(&(seq_len as u32))
                    .set_arg(&(self.hidden_dim as u32))
                    .set_arg(&(self.ffn_dim as u32))
                    .set_global_work_size(num_outputs * 64)
                    .set_local_work_size(64)
                    .enqueue_nd_range(q)?;
            }
        } else {
            q4mm!(
                &scratch.ffn_in,
                &self.ffn_gate_q4,
                &scratch.gate,
                self.hidden_dim,
                self.ffn_dim,
                self.fmt_gate
            );
            q4mm!(
                &scratch.ffn_in,
                &self.ffn_up_q4,
                &scratch.up_buf,
                self.hidden_dim,
                self.ffn_dim,
                self.fmt_up
            );
            let silu_n = seq_len * self.ffn_dim;
            kernel_count += 1;
            unsafe {
                ExecuteKernel::new(&p.silu_mul)
                    .set_arg(&scratch.gate)
                    .set_arg(&scratch.up_buf)
                    .set_arg(&scratch.act)
                    .set_arg(&(silu_n as u32))
                    .set_global_work_size(silu_n.div_ceil(4))
                    .enqueue_nd_range(q)?;
            }
        }

        // 15-16. FFN down + final residual add -> out_buf
        self.forward_down_residual(out_buf, seq_len, &mut kernel_count, p)?;

        let elapsed = start.elapsed();
        tracing::trace!(
            "[Arc Layer] {} kernels enqueued, {:.2}ms (no host sync), seq_len={}",
            kernel_count,
            elapsed.as_secs_f64() * 1000.0,
            seq_len
        );

        self.cached_len = new_cache_len;
        Ok(())
    }

    /// Shared final step: down projection + residual add into provided output buffer.
    fn forward_down_residual(
        &self,
        output: &Buffer<f32>,
        seq_len: usize,
        kernel_count: &mut usize,
        p: &OpenCLPipelines,
    ) -> Result<()> {
        use opencl3::kernel::ExecuteKernel;
        let q = &*p.queue;
        let scratch = self.scratch.as_ref().unwrap();
        let n = seq_len * self.hidden_dim;

        if self.fmt_down == OpenCLQuantFormat::Q4_K {
            // Fused: output = q4k_matmul(act, down) + residual1
            let num_outputs = seq_len * self.hidden_dim;
            *kernel_count += 1;
            unsafe {
                ExecuteKernel::new(&p.q4k_matmul_add)
                    .set_arg(&self.ffn_down_q4)
                    .set_arg(&scratch.act)
                    .set_arg(&scratch.residual1) // residual
                    .set_arg(output)
                    .set_arg(&(seq_len as u32))
                    .set_arg(&(self.ffn_dim as u32))
                    .set_arg(&(self.hidden_dim as u32))
                    .set_global_work_size(num_outputs * 64)
                    .set_local_work_size(64)
                    .enqueue_nd_range(q)?;
            }
        } else {
            // Separate: matmul then add
            let num_outputs = seq_len * self.hidden_dim;
            *kernel_count += 1;
            let kernel = match self.fmt_down {
                OpenCLQuantFormat::Q4_0 => &p.q4_matmul,
                OpenCLQuantFormat::Q4_K => &p.q4k_matmul,
                OpenCLQuantFormat::Q6_K => &p.q6k_matmul,
            };
            unsafe {
                ExecuteKernel::new(kernel)
                    .set_arg(&self.ffn_down_q4)
                    .set_arg(&scratch.act)
                    .set_arg(&scratch.ffn_out)
                    .set_arg(&(seq_len as u32))
                    .set_arg(&(self.ffn_dim as u32))
                    .set_arg(&(self.hidden_dim as u32))
                    .set_global_work_size(num_outputs * 64)
                    .set_local_work_size(64)
                    .enqueue_nd_range(q)?;
            }
            *kernel_count += 1;
            unsafe {
                ExecuteKernel::new(&p.add)
                    .set_arg(&scratch.residual1)
                    .set_arg(&scratch.ffn_out)
                    .set_arg(output)
                    .set_arg(&(n as u32))
                    .set_global_work_size(n.div_ceil(4))
                    .enqueue_nd_range(q)?;
            }
        }

        Ok(())
    }
}

/// Pre-compiled OpenCL kernels (created once, shared across all OpenCLLayers)
///
/// SAFETY: OpenCL handles are raw pointers but are only accessed through the
/// serialized command queue. Safe to share across threads when access is synchronized.
#[derive(Debug)]
pub struct OpenCLPipelines {
    pub context: Arc<Context>,
    pub queue: Arc<CommandQueue>,
    /// Compiled kernels (one per operation)
    pub q4_matmul: Kernel, // Q4_0 XMX path on Intel Arc
    pub q4k_matmul: Kernel,       // Q4_K GEMV kernel
    pub q4k_gate_up_silu: Kernel, // Fused gate+up+silu (reads input once)
    pub q4k_matmul_add: Kernel,   // Fused matmul + residual add
    pub q6k_matmul: Kernel,       // Q6_K GEMV kernel
    pub rms_norm: Kernel,
    pub rope: Kernel,
    pub dual_rope: Kernel,
    pub attention: Kernel,
    pub silu_mul: Kernel,
    pub add: Kernel,
    pub append_kv: Kernel,
    pub dual_append_kv: Kernel,
    // Image model kernels (F32 dense matmul, bidirectional attention, AdaLN)
    pub f32_matmul: Kernel,
    pub f32_matmul_bias: Kernel,
    pub f32_gate_up_silu: Kernel,
    pub f32_matmul_add: Kernel,
    // Tiled GEMM variants (5-10x faster for seq_len > 1)
    pub f32_tiled_matmul: Kernel,
    pub f32_tiled_matmul_bias: Kernel,
    pub f32_tiled_matmul_add: Kernel,
    pub f32_tiled_gate_up_silu: Kernel,
    pub image_attention: Kernel,
    pub apply_rotary_emb: Kernel,
    pub scale_after_norm: Kernel,
    pub gated_residual: Kernel,
    pub broadcast_mul: Kernel,
    pub broadcast_add: Kernel,
    // Flux-specific kernels (LayerNorm, GELU)
    pub layer_norm: Kernel,
    pub layer_norm_no_bias: Kernel,
    pub gelu: Kernel,
    pub gelu_slice: Kernel,
    // Flux attention pipeline kernels
    pub qkv_split: Kernel,
    pub flux_rope: Kernel,
    pub concat_seq: Kernel,
    pub split_heads: Kernel,
    pub qkv_split_strided: Kernel,
    pub strided_slice: Kernel,
    pub concat_cols: Kernel,
}

impl OpenCLPipelines {
    /// Create OpenCL pipelines for a device. Compiles all 8 kernels.
    #[allow(deprecated)]
    pub fn new(device_id: opencl3::types::cl_device_id) -> Result<Self> {
        use opencl3::device::Device;
        use opencl3::program::Program;

        let device = Device::new(device_id);
        let context = Context::from_device(&device)?;
        let queue = CommandQueue::create_default(&context, 0)?;

        // Concatenate all kernel sources
        let src = concat!(
            include_str!("../../kernels/q4_matmul.cl"),
            "\n",
            include_str!("../../kernels/q4k_matmul.cl"),
            "\n",
            include_str!("../../kernels/q6k_matmul.cl"),
            "\n",
            include_str!("../../kernels/rms_norm.cl"),
            "\n",
            include_str!("../../kernels/rope.cl"),
            "\n",
            include_str!("../../kernels/attention.cl"),
            "\n",
            include_str!("../../kernels/silu_mul.cl"),
            "\n",
            include_str!("../../kernels/add.cl"),
            "\n",
            include_str!("../../kernels/append_kv.cl"),
            "\n",
            // Image model kernels
            include_str!("../../kernels/f32_matmul.cl"),
            "\n",
            include_str!("../../kernels/image_attention.cl"),
            "\n",
            include_str!("../../kernels/adaln.cl"),
            "\n",
            // Flux-specific kernels
            include_str!("../../kernels/layer_norm.cl"),
            "\n",
            include_str!("../../kernels/gelu.cl"),
        );

        // Build program from source
        let program = Program::create_and_build_from_source(&context, src, "")
            .map_err(|e| anyhow::anyhow!("OpenCL program build error: {}", e))?;

        // Extract all kernels
        let q4_matmul = Kernel::create(&program, "q4_matmul")?;
        let q4k_matmul = Kernel::create(&program, "q4k_matmul")?;
        let q4k_gate_up_silu = Kernel::create(&program, "q4k_gate_up_silu")?;
        let q4k_matmul_add = Kernel::create(&program, "q4k_matmul_add")?;
        let q6k_matmul = Kernel::create(&program, "q6k_matmul")?;
        let rms_norm = Kernel::create(&program, "rms_norm")?;
        let rope = Kernel::create(&program, "rope")?;
        let dual_rope = Kernel::create(&program, "dual_rope")?;
        let attention = Kernel::create(&program, "attention")?;
        let silu_mul = Kernel::create(&program, "silu_mul")?;
        let add = Kernel::create(&program, "add")?;
        let append_kv = Kernel::create(&program, "append_kv")?;
        let dual_append_kv = Kernel::create(&program, "dual_append_kv")?;
        // Image model kernels
        let f32_matmul = Kernel::create(&program, "f32_matmul")?;
        let f32_matmul_bias = Kernel::create(&program, "f32_matmul_bias")?;
        let f32_gate_up_silu = Kernel::create(&program, "f32_gate_up_silu")?;
        let f32_matmul_add = Kernel::create(&program, "f32_matmul_add")?;
        let f32_tiled_matmul = Kernel::create(&program, "f32_tiled_matmul")?;
        let f32_tiled_matmul_bias = Kernel::create(&program, "f32_tiled_matmul_bias")?;
        let f32_tiled_matmul_add = Kernel::create(&program, "f32_tiled_matmul_add")?;
        let f32_tiled_gate_up_silu = Kernel::create(&program, "f32_tiled_gate_up_silu")?;
        let image_attention = Kernel::create(&program, "image_attention")?;
        let apply_rotary_emb = Kernel::create(&program, "apply_rotary_emb")?;
        let scale_after_norm = Kernel::create(&program, "scale_after_norm")?;
        let gated_residual = Kernel::create(&program, "gated_residual")?;
        let broadcast_mul = Kernel::create(&program, "broadcast_mul")?;
        let broadcast_add = Kernel::create(&program, "broadcast_add")?;
        // Flux-specific kernels
        let layer_norm = Kernel::create(&program, "layer_norm")?;
        let layer_norm_no_bias = Kernel::create(&program, "layer_norm_no_bias")?;
        let gelu = Kernel::create(&program, "gelu")?;
        let gelu_slice = Kernel::create(&program, "gelu_slice")?;
        let qkv_split = Kernel::create(&program, "qkv_split")?;
        let flux_rope = Kernel::create(&program, "flux_rope")?;
        let concat_seq = Kernel::create(&program, "concat_seq")?;
        let split_heads = Kernel::create(&program, "split_heads")?;
        let qkv_split_strided = Kernel::create(&program, "qkv_split_strided")?;
        let strided_slice = Kernel::create(&program, "strided_slice")?;
        let concat_cols = Kernel::create(&program, "concat_cols")?;

        Ok(Self {
            context: Arc::new(context),
            queue: Arc::new(queue),
            q4_matmul,
            q4k_matmul,
            q4k_gate_up_silu,
            q4k_matmul_add,
            q6k_matmul,
            rms_norm,
            rope,
            dual_rope,
            attention,
            silu_mul,
            add,
            append_kv,
            dual_append_kv,
            f32_matmul,
            f32_matmul_bias,
            f32_gate_up_silu,
            f32_matmul_add,
            f32_tiled_matmul,
            f32_tiled_matmul_bias,
            f32_tiled_matmul_add,
            f32_tiled_gate_up_silu,
            image_attention,
            apply_rotary_emb,
            scale_after_norm,
            gated_residual,
            broadcast_mul,
            broadcast_add,
            layer_norm,
            layer_norm_no_bias,
            gelu,
            gelu_slice,
            qkv_split,
            flux_rope,
            concat_seq,
            split_heads,
            qkv_split_strided,
            strided_slice,
            concat_cols,
        })
    }
}

/// Get total global memory for an OpenCL device (bytes).
unsafe impl Send for OpenCLPipelines {}
unsafe impl Sync for OpenCLPipelines {}

#[cfg(feature = "opencl")]
pub fn get_opencl_device_memory(device_id: opencl3::types::cl_device_id) -> u64 {
    let device = Device::new(device_id);
    device.global_mem_size().unwrap_or(0)
}

/// Enumerate OpenCL GPU devices (non-NVIDIA, non-CPU).
/// Returns device IDs in enumeration order; index i corresponds to the i-th OpenCL GPU found.
///
/// The vendor rule lives in `opencl_probe`; this returns the raw ids that only a caller
/// building kernels can use, and skips the whole stack when it has been found unresponsive.
#[cfg(feature = "opencl")]
pub fn enumerate_opencl_devices() -> Vec<opencl3::types::cl_device_id> {
    use opencl3::device::CL_DEVICE_TYPE_GPU;
    use opencl3::platform::get_platforms;

    if !crate::inference::kernel::opencl_probe::responsive() {
        return Vec::new();
    }
    let mut devices = Vec::new();
    if let Ok(platforms) = get_platforms() {
        for platform in platforms {
            if let Ok(device_ids) = platform.get_devices(CL_DEVICE_TYPE_GPU) {
                for device_id in device_ids {
                    let device = Device::new(device_id);
                    // NVIDIA belongs to CUDA: counting a card on both paths plans onto it twice.
                    if let Ok(vendor) = device.vendor() {
                        if vendor.to_lowercase().contains("nvidia") {
                            continue;
                        }
                    }
                    devices.push(device_id);
                }
            }
        }
    }
    devices
}

/// The `rms_norm` kernel on a real device, against a host reference.
///
/// Z-Image's q/k normalisation reuses this kernel over a row length of `head_dim` rather
/// than `dim`, treating a `[seq, heads.head_dim]` buffer as `seq.heads` short rows. That
/// reinterpretation is the whole of the fix, and it is a property of the kernel's indexing  -
/// so it is checked here, on the card, rather than inferred from reading it.
///
/// Needs an OpenCL device. On the Arc laptop: `RUSTICL_ENABLE=iris`, or rusticl reports no
/// device at all and this skips while looking like it passed.
#[cfg(test)]
mod rms_norm_on_device {
    use super::*;
    use opencl3::kernel::ExecuteKernel;
    use opencl3::types::CL_BLOCKING;

    fn host_rms_norm(x: &[f32], scale: &[f32], row_len: usize, eps: f32) -> Vec<f32> {
        x.chunks_exact(row_len)
            .flat_map(|row| {
                let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / row_len as f32;
                let inv = 1.0 / (mean_sq + eps).sqrt();
                row.iter()
                    .zip(scale)
                    .map(move |(v, s)| v * inv * s)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Z-Image's shape: several heads of 64 in one row, which is what the reuse relies on.
    const SEQ: usize = 5;
    const HEADS: usize = 3;
    const HEAD_DIM: usize = 64;
    const EPS: f32 = 1e-5;

    fn signal() -> Vec<f32> {
        (0..SEQ * HEADS * HEAD_DIM)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.11)
            .collect()
    }

    fn scales() -> Vec<f32> {
        (0..HEAD_DIM).map(|i| 0.5 + (i % 5) as f32 * 0.25).collect()
    }

    /// What the device test would be worth if the kernel ignored its row-length argument
    /// and normalised the whole `heads.head_dim` row instead.
    ///
    /// Runs everywhere, needs no device, and is the reason the device test means something:
    /// if these two agreed, a kernel that read the wrong length would pass.
    #[test]
    fn normalising_per_head_is_not_the_same_as_per_row() {
        let x = signal();
        let per_head = host_rms_norm(&x, &scales(), HEAD_DIM, EPS);
        let wide: Vec<f32> = {
            let wide_scale: Vec<f32> = scales()
                .iter()
                .cycle()
                .take(HEADS * HEAD_DIM)
                .copied()
                .collect();
            host_rms_norm(&x, &wide_scale, HEADS * HEAD_DIM, EPS)
        };
        let worst = per_head
            .iter()
            .zip(&wide)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst > 1e-2,
            "the two readings differ by at most {worst}, so the device test could not tell              a per-head normalisation from a per-row one"
        );
    }

    #[test]
    #[ignore = "needs an OpenCL device: run on the Arc host with RUSTICL_ENABLE=iris"]
    fn a_short_row_length_normalises_each_head_on_its_own() {
        let devices = enumerate_opencl_devices();
        assert!(
            !devices.is_empty(),
            "no OpenCL device - this test cannot say anything; on the Arc laptop set \
             RUSTICL_ENABLE=iris"
        );

        let x = signal();
        let scale = scales();
        let want = host_rms_norm(&x, &scale, HEAD_DIM, EPS);

        for &dev in &devices {
            use std::ptr::null_mut;
            let p = OpenCLPipelines::new(dev).expect("pipelines");
            let mut upload = |v: &[f32]| -> Buffer<f32> {
                let mut b = unsafe {
                    Buffer::<f32>::create(&p.context, CL_MEM_READ_WRITE, v.len(), null_mut())
                }
                .expect("alloc");
                unsafe { p.queue.enqueue_write_buffer(&mut b, CL_BLOCKING, 0, v, &[]) }
                    .expect("write");
                b
            };
            let x_buf = upload(&x);
            let scale_buf = upload(&scale);
            let out_buf = upload(&vec![0f32; x.len()]);

            let row_len = HEAD_DIM as u32;
            unsafe {
                ExecuteKernel::new(&p.rms_norm)
                    .set_arg(&x_buf)
                    .set_arg(&out_buf)
                    .set_arg(&scale_buf)
                    .set_arg(&row_len)
                    .set_arg(&EPS)
                    .set_global_work_size(SEQ * HEADS * 256)
                    .set_local_work_size(256)
                    .enqueue_nd_range(&p.queue)
                    .expect("enqueue");
            }
            p.queue.finish().expect("finish");
            let mut got = vec![0f32; x.len()];
            unsafe {
                p.queue
                    .enqueue_read_buffer(&out_buf, CL_BLOCKING, 0, &mut got, &[])
            }
            .expect("read");
            p.queue.finish().expect("finish read");

            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                    "element {i} (head {}, lane {}): device {g}, host {w}",
                    i / HEAD_DIM % HEADS,
                    i % HEAD_DIM
                );
            }
        }
    }
}
