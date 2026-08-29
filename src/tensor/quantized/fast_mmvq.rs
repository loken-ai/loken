use super::super::{DType, Error, Result, Shape};
use super::QCudaStorage;
use crate::tensor;
use crate::tensor::kernel_ffi::{CudaStorage, Layout};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use std::sync::{Arc, Mutex, OnceLock};

use super::MATRIX_ROW_PADDING;
const Q8_1_BLOCK_SIZE: usize = 32;
const Q8_1_TYPE_SIZE: usize = 36; // 2 halves (d, s) + 32 int8 qs
use super::CUDA_QUANTIZE_BLOCK_SIZE;

use super::pad_to as pad;

/// Grow-only Q8_1 scratch for the quantized activation, one per STREAM.
///
/// Per device was wrong: a device carries as many streams as it has users - a language model
/// and an image engine on one card, two requests in flight - and each quantises its own
/// activation into this buffer before reading it back in the next kernel. The host mutex
/// orders the entry, and the entry is all it orders: a launch returns before it runs.
///
/// The tiled matmul's workspace has the same hazard and is fixed by an event, because it is
/// megabytes and cannot be duplicated per stream. This one is kilobytes - `k` padded, over
/// blocks of 32, at 36 bytes each - so giving each stream its own removes the sharing rather
/// than sequencing it. That also keeps this path capturable: a decode runs inside a CUDA
/// graph, and waiting there on an event recorded outside the capture invalidates it.
static WORKSPACE: OnceLock<Mutex<std::collections::HashMap<(usize, usize), Arc<CudaSlice<u8>>>>> =
    OnceLock::new();

fn workspace_ensure(
    dev: &tensor::cuda::CudaDevice,
    min_bytes: usize,
) -> Result<Arc<CudaSlice<u8>>> {
    use cudarc::driver::sys::CUstream;
    let map = WORKSPACE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    let mut g = map.lock().unwrap_or_else(|e| e.into_inner());
    let key = (dev.ordinal(), dev.stream().cu_stream() as CUstream as usize);
    if g.get(&key).map(|b| b.len()) < Some(min_bytes) {
        let fresh = dev
            .stream()
            .alloc_zeros::<u8>(min_bytes)
            .map_err(|er| Error::msg(format!("fast_mmvq workspace: {er}")))?;
        g.insert(key, Arc::new(fresh));
    }
    Ok(g[&key].clone())
}

/// Pre-size the per-device scratch (model load).
pub fn ensure_workspace_capacity(
    dev: &crate::tensor::kernel_ffi::CudaDevice,
    min_bytes: usize,
) -> Result<()> {
    let _ = workspace_ensure(dev.native(), min_bytes)?;
    Ok(())
}

/// Max scratch bytes one decode MMVQ with input dim `max_k` needs.
pub fn max_scratch_bytes_for_k(max_k: usize) -> usize {
    pad(max_k, MATRIX_ROW_PADDING) / Q8_1_BLOCK_SIZE * Q8_1_TYPE_SIZE
}

/// Drop the per-device scratch (model unload). try_lock + skip: the
/// emergency-reclaim path may run while a forward holds this map
/// (same self-deadlock class as `release_mmq_workspaces`); an in-use
/// workspace must not be dropped anyway.
pub fn release_workspaces() {
    if let Some(map) = WORKSPACE.get() {
        if let Ok(mut g) = map.try_lock() {
            g.clear();
        }
    }
}

enum Act {
    Silu,
    Gelu,
}

fn try_fused(
    up_storage: &QCudaStorage,
    gate_storage: &QCudaStorage,
    self_shape: &Shape,
    rhs: &CudaStorage,
    rhs_l: &Layout,
    act: Act,
) -> Result<Option<(CudaStorage, Shape)>> {
    use super::GgmlDType;
    if up_storage.dtype() != GgmlDType::Q4K || gate_storage.dtype() != GgmlDType::Q4K {
        return Ok(None);
    }
    let rdt = rhs.dtype()?;
    if rdt != DType::F32 && rdt != DType::BF16 {
        return Ok(None);
    }
    if up_storage.device().ordinal() != gate_storage.device().ordinal() {
        return Ok(None);
    }
    let (nrows, ncols) = self_shape.dims2().map_err(|e| Error::msg(e.0))?;
    let (b_size, k) = match rhs_l.shape().dims() {
        [b, m, k] => (b * m, *k),
        [b, k] => (*b, *k),
        _ => return Ok(None),
    };
    if ncols != k || b_size != 1 {
        return Ok(None);
    }
    let (o1, o2) = rhs_l.contiguous_offsets();

    let dev = up_storage.device().native().clone();
    let stream = dev.stream().clone();

    // Kernel pair for (input dtype, activation): BF16 GELU has no
    // in-tree kernel yet - graceful None (caller runs the unfused
    // chain).
    let (quant_kname, fused_kname, bf16_out) = match (rdt, &act) {
        (DType::F32, Act::Silu) => (
            "mmvq_gguf_quantize_q8_1_f32",
            "mmvq_gguf_q4_k_f32_fused_silu_cuda1",
            false,
        ),
        (DType::F32, Act::Gelu) => (
            "mmvq_gguf_quantize_q8_1_f32",
            "mmvq_gguf_q4_k_f32_fused_gelu_cuda1",
            false,
        ),
        (DType::BF16, Act::Silu) => (
            "mmvq_gguf_quantize_q8_1_bf16",
            "mmvq_gguf_q4_k_bf16_fused_silu_cuda1",
            true,
        ),
        (DType::BF16, Act::Gelu) => return Ok(None),
        _ => return Ok(None),
    };
    let Ok(quant_fn) = dev.mmvq_fn(quant_kname) else {
        return Ok(None);
    };
    let Ok(fused_fn) = dev.mmvq_fn(fused_kname) else {
        return Ok(None);
    };

    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let scratch_bytes = (k_padded / Q8_1_BLOCK_SIZE) * Q8_1_TYPE_SIZE;
    let scratch = workspace_ensure(&dev, scratch_bytes)?;

    // 1) quantize activation -> Q8_1 scratch
    let q_cfg = LaunchConfig {
        grid_dim: (k_padded.div_ceil(CUDA_QUANTIZE_BLOCK_SIZE) as u32, 1, 1),
        block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let kx = k as i32;
    let kxp = k_padded as i32;
    match rdt {
        DType::F32 => {
            let x = rhs.as_cuda_slice::<f32>()?.slice(o1..o2);
            let mut b = stream.launch_builder(&quant_fn);
            b.arg(&x).arg(scratch.as_ref()).arg(&kx).arg(&kxp);
            unsafe { b.launch(q_cfg) }
                .map_err(|e| Error::msg(format!("fast_mmvq quantize: {e}")))?;
        }
        DType::BF16 => {
            let x = rhs.as_cuda_slice::<half::bf16>()?.slice(o1..o2);
            let mut b = stream.launch_builder(&quant_fn);
            b.arg(&x).arg(scratch.as_ref()).arg(&kx).arg(&kxp);
            unsafe { b.launch(q_cfg) }
                .map_err(|e| Error::msg(format!("fast_mmvq quantize: {e}")))?;
        }
        _ => unreachable!(),
    }

    // 2) fused (gate + up + activation) MMVQ
    let f_cfg = LaunchConfig {
        grid_dim: (nrows as u32, 1, 1),
        block_dim: (32, 4, 1),
        shared_mem_bytes: 0,
    };
    let ncols_i = k as i32;
    let nrows_i = nrows as i32;
    let stride_col_y = (k_padded / Q8_1_BLOCK_SIZE) as i32;
    let stride_col_dst = nrows as i32;
    let up_blob = up_storage.weight_cuda_slice();
    let gate_blob = gate_storage.weight_cuda_slice();

    let mut out_dims = rhs_l.shape().dims().to_vec();
    out_dims.pop();
    out_dims.push(nrows);
    let dev_compat = up_storage.device().clone();

    let out_storage = if bf16_out {
        let mut out =
            crate::tensor::cuda::with_oom_retry(&dev_compat.0, "fast_mmvq out", || unsafe {
                stream.alloc::<half::bf16>(nrows)
            })
            .map_err(|e: crate::tensor::Error| Error::msg(e.to_string()))?;
        let mut b = stream.launch_builder(&fused_fn);
        b.arg(up_blob)
            .arg(gate_blob)
            .arg(scratch.as_ref())
            .arg(&mut out)
            .arg(&ncols_i)
            .arg(&nrows_i)
            .arg(&stride_col_y)
            .arg(&stride_col_dst);
        unsafe { b.launch(f_cfg) }.map_err(|e| Error::msg(format!("fast_mmvq fused: {e}")))?;
        CudaStorage::wrap_cuda_slice(out, dev_compat)
    } else {
        let mut out =
            crate::tensor::cuda::with_oom_retry(&dev_compat.0, "fast_mmvq out", || unsafe {
                stream.alloc::<f32>(nrows)
            })
            .map_err(|e: crate::tensor::Error| Error::msg(e.to_string()))?;
        let mut b = stream.launch_builder(&fused_fn);
        b.arg(up_blob)
            .arg(gate_blob)
            .arg(scratch.as_ref())
            .arg(&mut out)
            .arg(&ncols_i)
            .arg(&nrows_i)
            .arg(&stride_col_y)
            .arg(&stride_col_dst);
        unsafe { b.launch(f_cfg) }.map_err(|e| Error::msg(format!("fast_mmvq fused: {e}")))?;
        CudaStorage::wrap_cuda_slice(out, dev_compat)
    };
    Ok(Some((out_storage, Shape::from(out_dims))))
}

/// Fused (up + gate + SiLU) Q4_K decode MMVQ.
pub fn try_fused_silu<L: std::borrow::Borrow<Layout>>(
    up_storage: &QCudaStorage,
    gate_storage: &QCudaStorage,
    self_shape: &Shape,
    rhs: &CudaStorage,
    rhs_l: L,
) -> Result<Option<(CudaStorage, Shape)>> {
    try_fused(
        up_storage,
        gate_storage,
        self_shape,
        rhs,
        rhs_l.borrow(),
        Act::Silu,
    )
}

/// Fused (up + gate + tanh-GELU) Q4_K decode MMVQ (gemma4 dense).
pub fn try_fused_gelu<L: std::borrow::Borrow<Layout>>(
    up_storage: &QCudaStorage,
    gate_storage: &QCudaStorage,
    self_shape: &Shape,
    rhs: &CudaStorage,
    rhs_l: L,
) -> Result<Option<(CudaStorage, Shape)>> {
    try_fused(
        up_storage,
        gate_storage,
        self_shape,
        rhs,
        rhs_l.borrow(),
        Act::Gelu,
    )
}

/// The fused gate-and-up kernel against the two projections done separately.
///
/// One kernel with the activation as a template argument: the seven lines that compute it are
/// all that separates the two, against eighty they share doing the reduction. The mat-vec
/// oracle covers the plain path only, so this is what covers the fused one.
#[cfg(test)]
mod fused_gate_up {
    use super::*;
    use crate::tensor::quantized::{GgmlDType, QHostTensor};
    use crate::tensor::{cuda::CudaDevice, quant_cpu, Device, Shape, StorageView, Tensor};

    fn silu(g: f64) -> f64 {
        g / (1.0 + (-g).exp())
    }

    fn gelu_tanh(g: f64) -> f64 {
        const ALPHA: f64 = 0.797_884_560_802_865_4;
        0.5 * g * (1.0 + (ALPHA * (g + 0.044715 * g * g * g)).tanh())
    }

    #[test]
    fn the_fused_kernel_applies_the_activation_it_was_asked_for() {
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the fused gate/up kernel is NOT covered by this run");
            return;
        };
        let device = Device::Cuda(dev.clone());

        // Q4_K, one row: the shape the fused path accepts.
        let (n, k) = (16usize, 256usize);
        let weight = |i: usize, seed: u32, scale: f32| -> f32 {
            let h = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
            ((h >> 8) as f32 / (1 << 24) as f32 - 0.5) * scale
        };
        // The gate lands near +-1, which is where silu and the tanh GELU are furthest apart.
        let upf: Vec<f32> = (0..n * k).map(|i| weight(i, 1, 0.5)).collect();
        let gatef: Vec<f32> = (0..n * k).map(|i| weight(i, 7, 0.14)).collect();

        let quantised = |w: &[f32]| {
            let bytes = quant_cpu::from_float_bytes(GgmlDType::Q4K, w).unwrap();
            let qt = QHostTensor::from_bytes(&bytes, GgmlDType::Q4K, vec![n, k]).unwrap();
            let dq = qt.dequantize_f32().unwrap();
            let storage = QCudaStorage::upload(
                &crate::tensor::kernel_ffi::CudaDevice(dev.clone()),
                &bytes,
                GgmlDType::Q4K,
            )
            .unwrap();
            (storage, dq)
        };
        let (up_s, up_dq) = quantised(&upf);
        let (gate_s, gate_dq) = quantised(&gatef);

        let x: Vec<f32> = (0..k).map(|i| ((i % 53) as f32) * 0.02 - 0.5).collect();
        let xt = Tensor::from_vec_f32(x.clone(), vec![1, k])
            .unwrap()
            .to_device(&device)
            .unwrap();
        let (guard, layout) = xt.storage_and_layout();
        let StorageView::Cuda(rhs) = &*guard else {
            panic!("activation is not on the card")
        };
        let shape = Shape::from(vec![n, k]);

        let run = |which: &str| -> Vec<f32> {
            let out = if which == "silu" {
                try_fused_silu(&up_s, &gate_s, &shape, rhs, &layout).unwrap()
            } else {
                try_fused_gelu(&up_s, &gate_s, &shape, rhs, &layout).unwrap()
            };
            let (storage, out_shape) =
                out.unwrap_or_else(|| panic!("{which}: the fused path declined its own shape"));
            assert_eq!(out_shape.dims(), &[1, n], "{which}: wrong output shape");
            crate::tensor::cuda_ext::tensor_from_cuda_storage(storage, vec![1, n])
                .unwrap()
                .to_vec_f32()
        };
        let with_silu = run("silu");
        let with_gelu = run("gelu");

        // The two runs quantise the SAME weights and the SAME activation, so every source of
        // error is common to both and cancels in the difference. What is left is exactly
        // `up . (silu(gate) - gelu(gate))`, which the host can compute from the dequantised
        // weights to a few parts in ten thousand - a far tighter statement than either run
        // against a reference, where the q8_1 envelope swamps the activation's shape.
        let mut worst = 0f64;
        let mut separation = 0f64;
        for o in 0..n {
            let (mut up, mut gate) = (0f64, 0f64);
            for j in 0..k {
                up += x[j] as f64 * up_dq[o * k + j] as f64;
                gate += x[j] as f64 * gate_dq[o * k + j] as f64;
            }
            let want = up * (silu(gate) - gelu_tanh(gate));
            let got = with_silu[o] as f64 - with_gelu[o] as f64;
            let scale = (up * silu(gate)).abs().max(1e-3);
            worst = worst.max((got - want).abs() / scale);
            separation = separation.max(want.abs() / scale);
        }
        assert!(
            separation > 1e-2,
            "the two activations differ by at most {separation:.1e} of the output on this \
             data, so swapping them would not show - pick a gate range where they diverge"
        );
        assert!(
            worst < 1e-2,
            "the difference between the two runs is {worst:.2e} of the output away from \
             `up . (silu(gate) - gelu(gate))`, so at least one of them is not applying the \
             activation it was asked for"
        );
        eprintln!(
            "fused gate/up: the two activations separate by {separation:.1e} of the output \
             and each applies its own to within {worst:.1e}"
        );
    }
}
