//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Every kernel this file launches is built on the `m16n8k32` integer MMA, which arrived with
/// Ampere. A Turing card has no code for any of them, so the answer here is "not this path"  -
/// exactly as for an unsupported dtype, and never a launch that would fail.
fn card_has_the_instruction(input: &Tensor) -> bool {
    input
        .device()
        .as_cuda_device()
        .is_ok_and(|d| d.has_ampere_tensor_cores())
}

/// The card a tensor lives on, and the raw handle of its compute stream - what every
/// launcher below opens with, since the `extern "C"` entry points take the stream as a
/// plain integer rather than as a typed handle.
fn card_and_stream(input: &Tensor) -> Result<(crate::tensor::kernel_ffi::CudaDevice, i64)> {
    let dev = input.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    Ok((dev, stream))
}

/// The device address of a tensor's buffer, for the FFI launchers below.
///
/// Reaching it takes three steps - project the storage, prove it is on a card, borrow the
/// slice at the element type the kernel expects - and every launcher here needs it for
/// several of its operands. The address outlives the borrow: the buffer belongs to `t`,
/// which the caller holds across the launch.
fn device_address<T: crate::tensor::kernel_ffi::CudaDType>(
    t: &Tensor,
    what: &str,
) -> Result<cudarc::driver::sys::CUdeviceptr> {
    let (storage, _) = t.storage_and_layout();
    match &*storage {
        StorageView::Cuda(c) => {
            let slice = c.as_cuda_slice::<T>()?;
            Ok(slice.device_ptr(slice.stream()).0)
        }
        _ => crate::tensor::bail!("{what} must be on a CUDA device"),
    }
}

/// Stage `rows` rows of `size_k` F32 values as the q8_1 activation every kernel here reads.
///
/// How many bytes that takes is the format's business, so it is asked of the format rather
/// than written out again at each call site. `size_k` is a multiple of 256 at every caller,
/// so the row divides into whole blocks and needs no padding.
fn q8_1_activation(
    input: &Tensor,
    size_k: usize,
    rows: usize,
    dev: &crate::tensor::kernel_ffi::CudaDevice,
) -> Result<cudarc::driver::CudaSlice<u8>> {
    use crate::tensor::quantized::GgmlDType;
    let q8_1 = GgmlDType::Q8_1;
    let bytes = rows * (size_k / q8_1.block_size()) * q8_1.type_size();
    let mut staged = unsafe { dev.alloc::<u8>(bytes)? };
    let (storage, _) = input.storage_and_layout();
    let slice = match &*storage {
        StorageView::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => crate::tensor::bail!("input must be on a CUDA device"),
    };
    crate::inference::quantized_cuda::quantize_q8_1_mmvq_multirow_f32(
        &slice.slice(..),
        &mut staged,
        size_k,
        rows,
        dev,
    )
    .map_err(|e| crate::tensor::Error::msg(e.to_string()))?;
    Ok(staged)
}

/// plain single-weight Q4_K IMMA (INT8 tensor-core) GEMM `out = x . Wᵀ`
/// for dense projections (attention QKV/O, FFN down) that otherwise run dp4a MMVQ
/// at prefill. `input` F32 [size_m, K]; `weight` Q4_K [N, K] -> out F32 [size_m, N].
/// Returns None off the eligible path (caller falls back to QMatMul).
pub fn dense_q4k_imma_m8_matmul(
    input: &Tensor,
    weight: &crate::tensor::quantized::QTensor,
) -> Result<Option<Tensor>> {
    if !card_has_the_instruction(input) {
        return Ok(None);
    }
    use crate::tensor::quantized::GgmlDType;
    use core::ffi::c_void;
    let is_q4k = weight.dtype() == GgmlDType::Q4K;
    let is_q5k = weight.dtype() == GgmlDType::Q5K;
    if (!is_q4k && !is_q5k) || input.dtype() != DType::F32 {
        return Ok(None);
    }
    let (size_m, size_k) = input.dims2()?;
    if size_k % 256 != 0 {
        return Ok(None);
    }
    let (size_n, size_k_w) = match weight.shape().dims() {
        [n, k] => (*n, *k),
        _ => return Ok(None),
    };
    if size_k != size_k_w {
        return Ok(None);
    }
    let (dev, stream) = card_and_stream(input)?;
    let w_ptr = weight.device_ptr()?;
    let q81_alloc = q8_1_activation(input, size_k, size_m, &dev)?;
    let output = unsafe { dev.alloc::<f32>(size_m * size_n)? };
    let q81_ptr = q81_alloc.device_ptr(q81_alloc.stream()).0 as *const c_void;
    let out_ptr = output.device_ptr(output.stream()).0 as *mut f32;
    let (m_i, n_i, k_i) = (size_m as i32, size_n as i32, size_k as i32);
    let plain = match is_q5k {
        true => loken_dense_q5k_imma_m8_plain,
        false => loken_dense_q4k_imma_m8_plain,
    };
    unsafe {
        plain(
            w_ptr as *const c_void,
            q81_ptr,
            out_ptr,
            m_i,
            n_i,
            k_i,
            stream,
        );
    }
    let storage = CudaStorage::wrap_cuda_slice(output, dev.clone());
    Ok(Some(tensor_from_cuda_storage(storage, (size_m, size_n))?))
}

/// Fused MoE gate+up Q4_K tensor-core (IMMA, M=8) GEMM with gelu(gate)*up.
/// Q4_K IMMA tensor-core path at M=8: gate and up projections, GeLU.mul, concatenated in
/// one launch so the intermediate never reaches memory.
pub fn moe_q4k_imma_m8_gate_up_gelu_mul_concat(
    input: &Tensor,
    gate_up_weights: &crate::tensor::quantized::QTensor,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
) -> Result<Tensor> {
    use crate::tensor::quantized::GgmlDType;
    use core::ffi::c_void;
    if gate_up_weights.dtype() != GgmlDType::Q4K {
        crate::tensor::bail!("moe_q4k_imma_m8_gate_up: requires Q4_K");
    }
    if input.dtype() != DType::F32 {
        crate::tensor::bail!("moe_q4k_imma_m8_gate_up: input must be F32");
    }
    let (size_m_in, size_k) = input.dims2()?;
    if size_k % 256 != 0 {
        crate::tensor::bail!("K={} not a multiple of 256", size_k);
    }
    let size_m = size_m_in * topk;
    let (num_experts, two_n, size_k_g) = match gate_up_weights.shape().dims() {
        [n, k] => (1usize, *n, *k),
        [e, n, k] => (*e, *n, *k),
        s => crate::tensor::bail!("gate_up_weights must be 2D or 3D, got {:?}", s),
    };
    if two_n % 2 != 0 || size_k != size_k_g {
        crate::tensor::bail!("bad shape: 2N={} K_in={} K_w={}", two_n, size_k, size_k_g);
    }
    let size_n = two_n / 2;
    let (dev, stream) = card_and_stream(input)?;
    let so_ptr = device_address::<u32>(sorted_token_ids, "sorted_token_ids")? as *const i32;
    let ex_ptr = device_address::<u32>(experts_ids, "experts_ids")? as *const i32;
    let weight_ptr = gate_up_weights.device_ptr()?;
    let q81_alloc = q8_1_activation(input, size_k, size_m_in, &dev)?;
    let output = dev.alloc_zeros::<f32>(size_m * size_n)?;
    let (e_i, t_i) = (num_experts as i32, topk as i32);
    let (m_i, n_i, k_i) = (size_m as i32, size_n as i32, size_k as i32);
    unsafe {
        loken_moe_q4k_imma_m8_gate_up(
            weight_ptr as *const c_void,
            q81_alloc.device_ptr(q81_alloc.stream()).0 as *const c_void,
            so_ptr,
            ex_ptr,
            output.device_ptr(output.stream()).0 as *mut f32,
            e_i,
            t_i,
            m_i,
            n_i,
            k_i,
            stream,
        );
    }
    let storage = CudaStorage::wrap_cuda_slice(output, dev.clone());
    Ok(tensor_from_cuda_storage(storage, (size_m, size_n))?)
}

/// Fused MoE down Q4_K IMMA M=8 GEMM + top-k reduction (optional residual add).
/// Q4_K IMMA down projection at M=8, reducing the per-expert results into one row.
#[allow(clippy::too_many_arguments)]
pub fn moe_q4k_imma_m8_down_reduce(
    input: &Tensor,
    weights: &crate::tensor::quantized::QTensor,
    sorted_token_ids: &Tensor,
    expert_ids: &Tensor,
    topk_weights: &Tensor,
    topk: usize,
    n_real_tokens: usize,
    residual: Option<&Tensor>,
) -> Result<Tensor> {
    use crate::tensor::quantized::GgmlDType;
    use core::ffi::c_void;
    use half::{bf16, f16};
    if weights.dtype() != GgmlDType::Q4K {
        crate::tensor::bail!("moe_q4k_imma_m8_down_reduce: requires Q4_K");
    }
    if input.dtype() != DType::F32 {
        crate::tensor::bail!("moe_q4k_imma_m8_down_reduce: input must be F32");
    }
    let (size_m, size_k) = input.dims2()?;
    if size_k % 256 != 0 {
        crate::tensor::bail!("K={} not a multiple of 256", size_k);
    }
    if size_m != n_real_tokens * topk {
        crate::tensor::bail!(
            "input M={} != n_real_tokens*topk={}",
            size_m,
            n_real_tokens * topk
        );
    }
    let (num_experts, hidden, size_k1) = weights.shape().dims3()?;
    if size_k != size_k1 {
        crate::tensor::bail!("input K={} != weight K={}", size_k, size_k1);
    }
    let (dev, stream) = card_and_stream(input)?;
    let so_ptr = device_address::<u32>(sorted_token_ids, "sorted_token_ids")? as *const i32;
    let ex_ptr = device_address::<u32>(expert_ids, "expert_ids")? as *const i32;
    let tw_ptr = device_address::<f32>(topk_weights, "topk_weights")? as *const f32;
    let weight_ptr = weights.device_ptr()?;
    let q81_alloc = q8_1_activation(input, size_k, size_m, &dev)?;
    let out_count = n_real_tokens * hidden;
    let out_alloc = if let Some(res) = residual {
        let res_c = res.contiguous()?;
        if res_c.shape().elem_count() != out_count {
            crate::tensor::bail!("residual elem mismatch");
        }
        let mut out = dev.alloc_zeros::<f32>(out_count)?;
        match res.dtype() {
            DType::F32 => {
                let (res_storage, _) = res_c.storage_and_layout();
                let rs = match &*res_storage {
                    StorageView::Cuda(c) => c.as_cuda_slice::<f32>()?,
                    _ => crate::tensor::bail!("residual must be cuda"),
                };
                dev.cuda_stream()
                    .memcpy_dtod(rs, &mut out)
                    .map_err(|e| crate::tensor::Error::msg(format!("residual memcpy: {e}")))?;
            }
            // Both half carriers widen to f32 through the same kernel, which is told which
            // one it was handed by a code rather than by a second launcher.
            dt @ (DType::F16 | DType::BF16) => {
                let (src, code) = match dt {
                    DType::F16 => (device_address::<f16>(&res_c, "residual")?, 0),
                    _ => (device_address::<bf16>(&res_c, "residual")?, 1),
                };
                let out_ptr = out.device_ptr(out.stream()).0 as *mut f32;
                unsafe {
                    loken_cast_init_f32_from_dtype(
                        out_ptr,
                        src as *const c_void,
                        out_count as i32,
                        code,
                        stream,
                    );
                }
            }
            d => crate::tensor::bail!("residual dtype {:?} unsupported", d),
        }
        out
    } else {
        dev.alloc_zeros::<f32>(out_count)?
    };
    let (e_i, t_i) = (num_experts as i32, topk as i32);
    let (m_i, h_i, k_i) = (size_m as i32, hidden as i32, size_k as i32);
    unsafe {
        loken_moe_q4k_imma_m8_down(
            weight_ptr as *const c_void,
            q81_alloc.device_ptr(q81_alloc.stream()).0 as *const c_void,
            so_ptr,
            ex_ptr,
            tw_ptr,
            out_alloc.device_ptr(out_alloc.stream()).0 as *mut f32,
            e_i,
            t_i,
            m_i,
            h_i,
            k_i,
            stream,
        );
    }
    let storage = CudaStorage::wrap_cuda_slice(out_alloc, dev.clone());
    Ok(tensor_from_cuda_storage(storage, (n_real_tokens, hidden))?)
}

/// Dense (non-MoE) Q4_K IMMA M=8 gate/up GEMM with silu(gate)*up.
/// Dense (non-MoE) Q4_K IMMA at M=8 with SiLU.mul fused into the epilogue.
pub fn dense_q4k_imma_m8_silu_mul(
    input: &Tensor,
    gate_weights: &crate::tensor::quantized::QTensor,
    up_weights: &crate::tensor::quantized::QTensor,
) -> Result<Tensor> {
    dense_q4k_imma_m8_act_mul(input, gate_weights, up_weights, false)
}

/// Dense Q4_K IMMA M=8 gate/up GEMM with gelu_pytorch_tanh(gate)*up (gemma FFN).
/// Dense (non-MoE) Q4_K IMMA at M=8 with GeLU.mul fused into the epilogue.
pub fn dense_q4k_imma_m8_gelu_mul(
    input: &Tensor,
    gate_weights: &crate::tensor::quantized::QTensor,
    up_weights: &crate::tensor::quantized::QTensor,
) -> Result<Tensor> {
    dense_q4k_imma_m8_act_mul(input, gate_weights, up_weights, true)
}

pub(super) fn dense_q4k_imma_m8_act_mul(
    input: &Tensor,
    gate_weights: &crate::tensor::quantized::QTensor,
    up_weights: &crate::tensor::quantized::QTensor,
    gelu: bool,
) -> Result<Tensor> {
    use crate::tensor::quantized::GgmlDType;
    use core::ffi::c_void;
    let is_q4k = gate_weights.dtype() == GgmlDType::Q4K && up_weights.dtype() == GgmlDType::Q4K;
    let is_q5k = gate_weights.dtype() == GgmlDType::Q5K && up_weights.dtype() == GgmlDType::Q5K;
    if !is_q4k && !is_q5k {
        crate::tensor::bail!("dense_q4k_imma_m8: requires Q4_K or Q5_K weights");
    }
    if input.dtype() != DType::F32 {
        crate::tensor::bail!("dense_q4k_imma_m8: input must be F32");
    }
    let (size_m, size_k) = input.dims2()?;
    if size_k % 256 != 0 {
        crate::tensor::bail!("K={} not a multiple of 256", size_k);
    }
    let (size_n, size_k_g) = match gate_weights.shape().dims() {
        [n, k] => (*n, *k),
        s => crate::tensor::bail!("dense_q4k_imma_m8: gate must be 2D, got {:?}", s),
    };
    let (size_n_u, size_k_u) = match up_weights.shape().dims() {
        [n, k] => (*n, *k),
        s => crate::tensor::bail!("up must be 2D, got {:?}", s),
    };
    if size_n != size_n_u || size_k != size_k_g || size_k != size_k_u {
        crate::tensor::bail!("gate/up/input shape mismatch");
    }
    let (dev, stream) = card_and_stream(input)?;
    let gate_ptr = gate_weights.device_ptr()?;
    let up_ptr = up_weights.device_ptr()?;
    let q81_alloc = q8_1_activation(input, size_k, size_m, &dev)?;
    let output = unsafe { dev.alloc::<f32>(size_m * size_n)? };
    let q81_ptr = q81_alloc.device_ptr(q81_alloc.stream()).0 as *const c_void;
    let out_ptr = output.device_ptr(output.stream()).0 as *mut f32;
    let (m_i, n_i, k_i) = (size_m as i32, size_n as i32, size_k as i32);
    // The two formats now answer to the same call, so the choice is the weight's and the
    // activation travels as an argument rather than as a second symbol.
    let actmul = match is_q5k {
        true => loken_dense_q5k_imma_m8_actmul,
        false => loken_dense_q4k_imma_m8_actmul,
    };
    unsafe {
        actmul(
            gate_ptr as *const c_void,
            up_ptr as *const c_void,
            q81_ptr,
            out_ptr,
            m_i,
            n_i,
            k_i,
            if gelu { 1 } else { 0 },
            stream,
        );
    }
    let storage = CudaStorage::wrap_cuda_slice(output, dev.clone());
    Ok(tensor_from_cuda_storage(storage, (size_m, size_n))?)
}
