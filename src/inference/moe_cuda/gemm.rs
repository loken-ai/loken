//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.
//!
//! Every entry point below describes one expert launch the same way: read the extents out of
//! the activation and the expert stack, take the address of each buffer the kernel walks,
//! launch, and wrap what was written back into a tensor. The three steps live in [`Extents`],
//! [`dev_addr`] and [`tensor_from_f32_slice`], so an entry point is left saying only which
//! kernel it drives and what that kernel needs beyond the common set.

use super::*;
use crate::tensor::cuda_ext::tensor_from_f32_slice;
use crate::tensor::kernel_ffi::CudaDType;
use crate::tensor::quantized::GgmlDType;
use core::ffi::c_void;
use half::{bf16, f16};

/// The weight formats the expert kernels serve, and the number each is known by.
///
/// `cuda/moe/moe_gguf.cu` dispatches on that number and its table is these same eight rows  -
/// the entry points used to carry a copy each, differing by a row or two, which is how a
/// format came to be accepted here and unhandled there.
const EXPERT_QUANT_CODES: [(GgmlDType, i32); 8] = [
    (GgmlDType::Q8_0, 0),
    (GgmlDType::Q4K, 1),
    (GgmlDType::Q2K, 2),
    (GgmlDType::Q3K, 3),
    (GgmlDType::Q5K, 4),
    (GgmlDType::Q6K, 5),
    (GgmlDType::Q5_0, 6),
    (GgmlDType::MxFp4, 7),
];

/// Whether the expert kernels serve this weight format at all.
///
/// The loaders ask this to decide whether to fuse a layer's projections into one tensor or
/// keep them apart, and they used to answer it with a list of their own. Four such lists
/// disagreed: two of them left MXFP4 out, so a gpt-oss checkpoint was refused a path its
/// kernels serve. The table above is the only thing that knows, so it is what answers.
pub(crate) fn expert_kernels_serve(dtype: GgmlDType) -> bool {
    gguf_quant_code(dtype, "").is_ok()
}

/// The number the kernels know `dtype` by, or the refusal naming the entry point that asked.
fn gguf_quant_code(dtype: GgmlDType, entry: &str) -> Result<i32> {
    match EXPERT_QUANT_CODES
        .iter()
        .find(|(served, _)| *served == dtype)
    {
        Some((_, code)) => Ok(*code),
        None => crate::tensor::bail!("{entry}: no expert kernel for {dtype:?} weights"),
    }
}

/// What one expert launch is described by, in the width the kernels read it at: `m` rows of
/// activation against a stack of `experts` matrices of `n` rows by `k` columns, each token
/// visiting `topk` of them.
#[derive(Clone, Copy)]
struct Extents {
    experts: i32,
    topk: i32,
    m: i32,
    n: i32,
    k: i32,
}

impl Extents {
    /// Read the extents from the activation and the expert stack it runs against.
    ///
    /// `stack` is the weight shape - `[experts, n, k]`, or `[n, k]` for a lone expert.
    /// `expanded` says whether the activation already holds one row per (token, expert)
    /// pair: when it does not, the launch writes `topk` rows for every row it reads.
    /// The invariant every expert kernel rests on is checked here - the activation and the
    /// stack agree on `k`.
    fn read(
        entry: &str,
        input: &Tensor,
        stack: &[usize],
        topk: usize,
        expanded: bool,
    ) -> Result<Self> {
        let (rows, k) = input.dims2()?;
        let (experts, n, stack_k) = match stack {
            [n, k] => (1usize, *n, *k),
            [e, n, k] => (*e, *n, *k),
            s => crate::tensor::bail!("{entry}: an expert stack is 2D or 3D, got {s:?}"),
        };
        if k != stack_k {
            crate::tensor::bail!("{entry}: activation K={k} against expert K={stack_k}");
        }
        Ok(Self {
            experts: experts as i32,
            topk: topk as i32,
            m: (if expanded { rows } else { rows * topk }) as i32,
            n: n as i32,
            k: k as i32,
        })
    }

    /// Refuse a `k` the kernels cannot walk: they step eight quantised values at a time.
    fn require_k_by_eight(&self, entry: &str) -> Result<()> {
        if self.k % 8 != 0 {
            crate::tensor::bail!("{entry}: K={} is not a multiple of 8", self.k);
        }
        Ok(())
    }

    /// Halve the output width: a gate‖up stack holds both projections in one matrix and the
    /// launch writes only what they combine into.
    fn combine_halves(&mut self, entry: &str) -> Result<()> {
        if self.n % 2 != 0 {
            crate::tensor::bail!(
                "{entry}: a gate‖up stack has an even row count, got {}",
                self.n
            );
        }
        self.n /= 2;
        Ok(())
    }

    /// The `[rows, columns]` the launch writes.
    fn out_shape(&self) -> (usize, usize) {
        (self.m as usize, self.n as usize)
    }

    /// How many f32 the launch writes.
    fn out_elems(&self) -> usize {
        let (rows, cols) = self.out_shape();
        rows * cols
    }
}

/// Where a tensor's elements sit on the device.
///
/// Every buffer handed to an expert kernel is contiguous from its first element - that is what
/// `contiguous()` returns, and what the tensors coming out of a kernel are - so the address of
/// the storage is the address of the data.
fn dev_addr<T: CudaDType>(t: &Tensor, what: &str) -> Result<u64> {
    let (storage, _) = t.storage_and_layout();
    match &*storage {
        StorageView::Cuda(c) => {
            let slice = c.as_cuda_slice::<T>()?;
            let (addr, _read) = slice.device_ptr(slice.stream());
            Ok(addr)
        }
        _ => crate::tensor::bail!("{what} must be on a CUDA device"),
    }
}

/// The addresses of the routing plan the kernels walk: `slots[i]` is the (token, choice) pair
/// the i-th sorted row is built from, and `experts[i]` the expert that row is multiplied by.
/// Sorting by expert is what makes each expert's rows one contiguous run. Both lists are u32
/// here and signed on the kernel's side - the same thirty-two bits either way.
fn plan_addrs(slots: &Tensor, experts: &Tensor) -> Result<(*const i32, *const i32)> {
    Ok((
        dev_addr::<u32>(slots, "sorted_token_ids")? as *const i32,
        dev_addr::<u32>(experts, "experts_ids")? as *const i32,
    ))
}

/// Address of an f32 argument a launch may or may not have - the router's weights, a
/// per-expert bias - with null standing for absent, which is how the kernels read it.
fn opt_f32_addr(t: Option<&Tensor>, what: &str) -> Result<*const f32> {
    match t {
        Some(t) => Ok(dev_addr::<f32>(t, what)? as *const f32),
        None => Ok(std::ptr::null()),
    }
}

/// A per-expert bias table packed for the launch, and the tensor that owns it. The address is
/// only valid while that tensor lives, so the caller holds it until the kernel has been given
/// its arguments.
fn bias_arg(bias: Option<&Tensor>, what: &str) -> Result<(Option<Tensor>, *const f32)> {
    let held = bias.map(|t| t.contiguous()).transpose()?;
    let addr = opt_f32_addr(held.as_ref(), what)?;
    Ok((held, addr))
}

/// MoE expert GEMM over GGUF-quantized weights (decode mmvq / prefill WMMA).
/// Batched GGUF-quantised GEMM, one matmul per expert over a shared activation.
pub fn moe_gemm_gguf(
    input: &Tensor,
    weights: &crate::tensor::quantized::QTensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
    dtype: DType,
) -> Result<Tensor> {
    const ENTRY: &str = "moe_gemm_gguf";
    if input.dtype() != DType::F32 {
        crate::tensor::bail!("{ENTRY} only accepts f32 inputs");
    }
    // With the router's weights the launch reduces rows that are already one per (token,
    // expert) pair; without them it is the launch that expands a token into those rows.
    let ext = Extents::read(
        ENTRY,
        input,
        weights.shape().dims(),
        topk,
        topk_weights.is_some(),
    )?;
    ext.require_k_by_eight(ENTRY)?;
    let dev = input.device().as_cuda_device()?;
    let gguf_dtype = gguf_quant_code(weights.dtype(), ENTRY)?;
    // MXFP4 experts have no WMMA prefill kernel; route prefill through the
    // (row-per-warp) mmvq path too. Correct numerics, slightly slower prefill  -
    // acceptable since decode is the bottleneck and prefill is infrequent.
    // The WMMA prefill kernel needs bf16 fragments, which are Ampere and above; below that
    // the family ships no code at all, so a Turing card takes the mmvq path for prefill too.
    let is_prefill =
        is_prefill && weights.dtype() != GgmlDType::MxFp4 && dev.has_ampere_tensor_cores();
    let weight_ptr = weights.device_ptr()?;
    let scale = opt_f32_addr(topk_weights.as_ref(), "topk_weights")?;
    let (slots, experts) = plan_addrs(sorted_token_ids, experts_ids)?;
    let output = unsafe { dev.alloc::<f32>(ext.out_elems()) }?;
    let out = output.device_ptr(output.stream()).0 as *mut c_void;
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        if is_prefill {
            // The WMMA kernel reads the activation at the model's own width; `inp` owns that
            // copy until the launch has been handed its address.
            let inp = input.to_dtype(dtype)?;
            let (input_ptr, input_dtype) = match dtype {
                DType::F16 => (dev_addr::<f16>(&inp, "input")? as *const c_void, 0),
                DType::BF16 => (dev_addr::<bf16>(&inp, "input")? as *const c_void, 1),
                d => crate::tensor::bail!("{ENTRY}: prefill reads f16 or bf16, not {d:?}"),
            };
            loken_moe_gemm_gguf_prefill(
                input_ptr,
                weight_ptr,
                slots,
                experts,
                scale,
                out,
                ext.experts,
                ext.topk,
                ext.m,
                ext.n,
                ext.k,
                input_dtype,
                gguf_dtype,
                stream,
            );
        } else {
            loken_moe_gemm_gguf(
                dev_addr::<f32>(input, "input")? as *const f32,
                weight_ptr as *const c_void,
                slots,
                experts,
                scale,
                out,
                ext.experts,
                ext.topk,
                ext.m,
                ext.n,
                ext.k,
                gguf_dtype,
                stream,
            );
        }
    }
    tensor_from_f32_slice(output, ext.out_shape(), &input.device())
}

/// Fused MoE gate GEMM + up GEMM + SiLU.mul in ONE launch (separate gate/up
/// weights). Decode (mmvq) path: quantizes the input to q8_1 once internally.
/// `input` [M, K] f32; `gate_weights`/`up_weights` [E, N, K]; returns [M.topk, N].
pub fn moe_gemm_gguf_gate_up_silu_mul(
    input: &Tensor,
    gate_weights: &crate::tensor::quantized::QTensor,
    up_weights: &crate::tensor::quantized::QTensor,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
) -> Result<Tensor> {
    const ENTRY: &str = "gate_up_silu_mul";
    if input.dtype() != DType::F32 {
        crate::tensor::bail!("{ENTRY} only accepts f32 inputs");
    }
    if up_weights.shape().dims() != gate_weights.shape().dims() {
        crate::tensor::bail!("{ENTRY}: the gate and up stacks differ in shape");
    }
    let ext = Extents::read(ENTRY, input, gate_weights.shape().dims(), topk, false)?;
    ext.require_k_by_eight(ENTRY)?;
    let gguf_dtype = gguf_quant_code(gate_weights.dtype(), ENTRY)?;
    let dev = input.device().as_cuda_device()?;
    let (slots, experts) = plan_addrs(sorted_token_ids, experts_ids)?;
    let output = unsafe { dev.alloc::<f32>(ext.out_elems()) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_moe_gemm_gguf_gate_up_silu_mul(
            dev_addr::<f32>(input, "input")? as *const f32,
            gate_weights.device_ptr()? as *const c_void,
            up_weights.device_ptr()? as *const c_void,
            slots,
            experts,
            output.device_ptr(output.stream()).0 as *mut f32,
            ext.experts,
            ext.topk,
            ext.m,
            ext.n,
            ext.k,
            gguf_dtype,
            stream,
        );
    }
    tensor_from_f32_slice(output, ext.out_shape(), &input.device())
}

/// Fused MoE gate GEMM + up GEMM + per-expert bias + OAI clamped-SwiGLU in ONE
/// launch (gpt-oss). Avoids materializing the [M.topk, N] gate/up intermediates.
/// `gate_bias`/`up_bias` are [E, N] f32 (or None). Supports MXFP4 experts (case 7).
pub fn moe_gemm_gguf_gate_up_swiglu_oai(
    input: &Tensor,
    gate_weights: &crate::tensor::quantized::QTensor,
    up_weights: &crate::tensor::quantized::QTensor,
    gate_bias: Option<&Tensor>,
    up_bias: Option<&Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    alpha: f64,
    limit: f64,
) -> Result<Tensor> {
    const ENTRY: &str = "gate_up_swiglu_oai";
    if input.dtype() != DType::F32 {
        crate::tensor::bail!("{ENTRY} only accepts f32 inputs");
    }
    if up_weights.shape().dims() != gate_weights.shape().dims() {
        crate::tensor::bail!("{ENTRY}: the gate and up stacks differ in shape");
    }
    let ext = Extents::read(ENTRY, input, gate_weights.shape().dims(), topk, false)?;
    ext.require_k_by_eight(ENTRY)?;
    let gguf_dtype = gguf_quant_code(gate_weights.dtype(), ENTRY)?;
    let dev = input.device().as_cuda_device()?;
    let (gate_held, gate_bias_ptr) = bias_arg(gate_bias, "gate_bias")?;
    let (up_held, up_bias_ptr) = bias_arg(up_bias, "up_bias")?;
    let (slots, experts) = plan_addrs(sorted_token_ids, experts_ids)?;
    let output = unsafe { dev.alloc::<f32>(ext.out_elems()) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_moe_gemm_gguf_gate_up_swiglu_oai(
            dev_addr::<f32>(input, "input")? as *const f32,
            gate_weights.device_ptr()? as *const c_void,
            up_weights.device_ptr()? as *const c_void,
            slots,
            experts,
            gate_bias_ptr,
            up_bias_ptr,
            output.device_ptr(output.stream()).0 as *mut f32,
            ext.experts,
            ext.topk,
            ext.m,
            ext.n,
            ext.k,
            gguf_dtype,
            alpha as f32,
            limit as f32,
            stream,
        );
    }
    drop(gate_held);
    drop(up_held);
    tensor_from_f32_slice(output, ext.out_shape(), &input.device())
}

/// Fused MoE gate||up GEMM + GeLU.mul -> concat.
pub fn moe_gemm_gguf_gate_up_gelu_mul_concat(
    input: &Tensor,
    gate_up_weights: &crate::tensor::quantized::QTensor,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
) -> Result<Tensor> {
    const ENTRY: &str = "gate_up_gelu_mul_concat";
    if input.dtype() != DType::F32 {
        crate::tensor::bail!("{ENTRY}: input must be F32");
    }
    let mut ext = Extents::read(ENTRY, input, gate_up_weights.shape().dims(), topk, false)?;
    ext.combine_halves(ENTRY)?;
    ext.require_k_by_eight(ENTRY)?;
    let gguf_dtype = gguf_quant_code(gate_up_weights.dtype(), ENTRY)?;
    let dev = input.device().as_cuda_device()?;
    let (slots, experts) = plan_addrs(sorted_token_ids, experts_ids)?;
    let output = unsafe { dev.alloc::<f32>(ext.out_elems()) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_moe_gemm_gguf_gate_up_gelu_mul_concat(
            dev_addr::<f32>(input, "input")? as *const f32,
            gate_up_weights.device_ptr()? as *const c_void,
            slots,
            experts,
            output.device_ptr(output.stream()).0 as *mut c_void,
            ext.experts,
            ext.topk,
            ext.m,
            ext.n,
            ext.k,
            gguf_dtype,
            stream,
        );
    }
    tensor_from_f32_slice(output, ext.out_shape(), &input.device())
}

/// Fused MoE down-projection + top-k reduction (optionally folding a residual
/// add).
pub fn moe_gemm_gguf_down_reduce(
    input: &Tensor,
    weights: &crate::tensor::quantized::QTensor,
    sorted_token_ids: &Tensor,
    expert_ids: &Tensor,
    topk_weights: &Tensor,
    topk: usize,
    n_real_tokens: usize,
    residual: Option<&Tensor>,
    down_bias: Option<&Tensor>,
) -> Result<Tensor> {
    const ENTRY: &str = "down_reduce";
    if input.dtype() != DType::F32 {
        crate::tensor::bail!("{ENTRY}: input must be F32");
    }
    // The rows arriving here are the sorted (token, expert) pairs the gate/up launch wrote;
    // the reduction folds each token's `topk` of them back into one.
    let ext = Extents::read(ENTRY, input, weights.shape().dims(), topk, true)?;
    if ext.m as usize != n_real_tokens * topk {
        crate::tensor::bail!(
            "{ENTRY}: input M={} != n_real_tokens={n_real_tokens} x topk={topk}",
            ext.m
        );
    }
    let gguf_dtype = gguf_quant_code(weights.dtype(), ENTRY)?;
    let dev = input.device().as_cuda_device()?;
    let (slots, experts) = plan_addrs(sorted_token_ids, expert_ids)?;
    let (_, hidden) = ext.out_shape();
    let out_count = n_real_tokens * hidden;
    let stream = dev.cuda_stream().cu_stream() as i64;
    // The kernel accumulates into its output with atomicAdd, so a residual is folded by
    // starting the buffer at the residual instead of at zero: a device copy when it already
    // is f32, a widening one when it is not.
    let out_alloc = match residual {
        Some(res) => {
            let res = res.contiguous()?;
            if res.shape().elem_count() != out_count {
                crate::tensor::bail!("{ENTRY}: residual elem mismatch");
            }
            let alloc = unsafe { dev.alloc::<f32>(out_count) }?;
            let dst = alloc.device_ptr(alloc.stream()).0;
            match res.dtype() {
                DType::F32 => {
                    dev.cuda_stream().context().bind_to_thread()?;
                    let src = dev_addr::<f32>(&res, "residual")?;
                    unsafe {
                        let st = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                            dst,
                            src,
                            out_count * std::mem::size_of::<f32>(),
                            dev.cuda_stream().cu_stream(),
                        );
                        if st != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                            crate::tensor::bail!(
                                "cuMemcpyDtoDAsync (residual init) failed: {st:?}"
                            );
                        }
                    }
                }
                DType::F16 => unsafe {
                    let src = dev_addr::<f16>(&res, "residual")? as *const c_void;
                    loken_cast_init_f32_from_dtype(
                        dst as *mut f32,
                        src,
                        out_count as i32,
                        0,
                        stream,
                    );
                },
                DType::BF16 => unsafe {
                    let src = dev_addr::<bf16>(&res, "residual")? as *const c_void;
                    loken_cast_init_f32_from_dtype(
                        dst as *mut f32,
                        src,
                        out_count as i32,
                        1,
                        stream,
                    );
                },
                d => crate::tensor::bail!("{ENTRY}: residual dtype {d:?} unsupported"),
            }
            alloc
        }
        None => dev.alloc_zeros::<f32>(out_count)?,
    };
    // Per-expert down bias [num_experts, size_n] F32: folded into the kernel's
    // atomicAdd (Σ_slot w.db) - replaces an index_select+broadcast_mul+sum+add
    // chain whose temporaries broke CUDA-graph capture. Persistent F32 weight
    // (ld_f32_opt at load) -> graph-safe pointer.
    if let Some(db) = down_bias {
        if db.dtype() != DType::F32 {
            crate::tensor::bail!("{ENTRY}: down_bias must be F32, got {:?}", db.dtype());
        }
        if db.dims2()?.1 != hidden {
            crate::tensor::bail!(
                "{ENTRY}: down_bias dim1 {} != size_n {hidden}",
                db.dims2()?.1
            );
        }
    }
    let (bias_held, down_bias_ptr) = bias_arg(down_bias, "down_bias")?;
    unsafe {
        loken_moe_gemm_gguf_down_reduce(
            dev_addr::<f32>(input, "input")? as *const f32,
            weights.device_ptr()? as *const c_void,
            slots,
            experts,
            dev_addr::<f32>(topk_weights, "topk_weights")? as *const f32,
            out_alloc.device_ptr(out_alloc.stream()).0 as *mut f32,
            ext.experts,
            ext.topk,
            ext.m,
            ext.n,
            ext.k,
            gguf_dtype,
            down_bias_ptr,
            stream,
        );
    }
    drop(bias_held);
    tensor_from_f32_slice(out_alloc, (n_real_tokens, hidden), &input.device())
}
