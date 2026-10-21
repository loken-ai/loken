//! Activations a decode kernel left already quantised to q8_1.
//!
//! A norm or an activation product that feeds a mat-vec can write the q8_1 row the mat-vec
//! needs in the same kernel. The row waits here, keyed by the address of the f32 buffer it
//! was made from, until the mat-vec's quantise step asks for that buffer and takes it instead
//! of launching a kernel.
//!
//! Invariants: an entry is the quantisation of the f32 buffer at its address as the kernel
//! left it, and it leaves the table when that address is handed out or released again, so a
//! buffer reused for other values never meets a stale row. Only the HIP allocator reports
//! its handouts and releases, so only cards it serves take part.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::tensor::gpu_runtime::driver::CudaSlice;

/// The q8_1 row of the f32 buffer at an address, and the row length it was made for.
type Table = HashMap<usize, (usize, CudaSlice<u8>)>;

fn table() -> MutexGuard<'static, Table> {
    static TABLE: OnceLock<Mutex<Table>> = OnceLock::new();
    TABLE
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Whether the card of `ordinal` reports its buffer handouts, so a row kept for one stays true.
pub fn supported(ordinal: usize) -> bool {
    !crate::tensor::gpu_runtime::is_nvidia_ordinal(ordinal)
}

/// Keeps `q81`, the quantisation of the `k` f32 values at `ptr`.
pub fn stash(ptr: usize, k: usize, q81: CudaSlice<u8>) {
    let replaced = table().insert(ptr, (k, q81));
    drop(replaced);
}

/// Takes the q8_1 row of the `k` f32 values at `ptr`, if a kernel left one.
pub fn take(ptr: usize, k: usize) -> Option<CudaSlice<u8>> {
    let mut t = table();
    match t.get(&ptr) {
        Some((len, _)) if *len == k => t.remove(&ptr).map(|(_, q)| q),
        _ => None,
    }
}

/// Drops the row kept for the buffer at `ptr`: the buffer is changing hands.
pub(crate) fn forget(ptr: usize) {
    let dropped = {
        let mut t = table();
        if t.is_empty() {
            return;
        }
        t.remove(&ptr)
    };
    drop(dropped);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::cuda::{quantize_q8_1, rms_norm_f32, CudaDevice};

    /// The row a producer kernel leaves is the row the quantise kernel makes of the buffer the
    /// producer wrote, byte for byte; and a buffer handed out again carries no row.
    #[test]
    fn a_producers_row_equals_the_quantise_kernels() {
        let Ok(dev) = CudaDevice::new(0) else {
            return;
        };
        if !supported(dev.ordinal()) {
            return;
        }
        let (cols, eps) = (2560usize, 1e-6f32);
        let stream = dev.stream();
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i * 37 % 211) as f32 - 100.0) * 0.013)
            .collect();
        let w: Vec<f32> = (0..cols).map(|i| 0.5 + (i % 17) as f32 * 0.03).collect();
        let xg = stream.clone_htod(&x).unwrap();
        let wg = stream.clone_htod(&w).unwrap();
        let out = rms_norm_f32(&dev, &xg, &wg, 1, cols, eps).unwrap();
        // The producer's row, then the quantise kernel's, over the same buffer.
        let (ptr, _g) = {
            use crate::tensor::gpu_runtime::driver::DevicePtr;
            out.device_ptr(&stream)
        };
        let fused = take(ptr as usize, cols).expect("the norm leaves its row");
        let plain = quantize_q8_1(&dev, &out, cols).unwrap();
        let (a, b) = (
            stream.clone_dtoh(&fused).unwrap(),
            stream.clone_dtoh(&plain).unwrap(),
        );
        dev.synchronize().unwrap();
        assert_eq!(a, b);
        // Asking again finds nothing: the row was taken.
        assert!(take(ptr as usize, cols).is_none());
        // The f32 row is the unfused kernel's.
        use crate::tensor::gpu_runtime::driver::PushKernelArg;
        let func = dev.fused_fn("fused_rmsnorm_wide_f32").unwrap();
        let reference = unsafe { stream.alloc::<f32>(cols) }.unwrap();
        let cfg = crate::tensor::cuda::row_norm_cfg(1, cols, true);
        let cols_i32 = cols as i32;
        let mut l = stream.launch_builder(&func);
        l.arg(&xg).arg(&wg).arg(&reference).arg(&eps).arg(&cols_i32);
        unsafe { l.launch(cfg) }.unwrap();
        assert_eq!(
            stream.clone_dtoh(&out).unwrap(),
            stream.clone_dtoh(&reference).unwrap()
        );
    }

    /// The tensor-level producers leave rows the quantise kernel would have made of their
    /// outputs.
    #[test]
    fn the_fused_products_leave_the_quantise_kernels_rows() {
        use crate::inference::kernel::fused::{
            fused_add_rmsnorm_dual, fused_split_gelu_mul, fused_split_silu_mul,
        };
        use crate::tensor::{Device, Tensor};
        let Ok(dev) = Device::new_cuda(0) else {
            return;
        };
        let Ok(raw) = dev.as_cuda_device() else {
            return;
        };
        if !supported(raw.native().ordinal()) {
            return;
        }
        let ndev = raw.native().clone();
        let cols = 2560usize;
        let v = |m: usize, a: f32| -> Vec<f32> {
            (0..cols)
                .map(|i| ((i * m % 197) as f32 - 90.0) * a)
                .collect()
        };
        let x = Tensor::from_vec(v(31, 0.011), (1, cols), &dev).unwrap();
        let r = Tensor::from_vec(v(17, 0.007), (1, cols), &dev).unwrap();
        let w = Tensor::from_vec(v(13, 0.004), cols, &dev).unwrap();
        let gu = Tensor::from_vec(
            (0..2 * cols)
                .map(|i| ((i * 29 % 211) as f32 - 100.0) * 0.02)
                .collect(),
            (1, 2 * cols),
            &dev,
        )
        .unwrap();
        let (_, norm) = fused_add_rmsnorm_dual(&x, &r, &w, 1e-6).unwrap();
        let silu = fused_split_silu_mul(&gu).unwrap();
        let gelu = fused_split_gelu_mul(&gu).unwrap();
        for (name, t) in [("add_rmsnorm", norm), ("silu", silu), ("gelu", gelu)] {
            let (slice, _) = t.cuda_f32_slice().unwrap();
            let stream = ndev.stream();
            let ptr = {
                use crate::tensor::gpu_runtime::driver::DevicePtr;
                slice.device_ptr(&stream).0 as usize
            };
            let fused = take(ptr, cols).unwrap_or_else(|| panic!("{name} leaves its row"));
            let plain = quantize_q8_1(&ndev, slice, cols).unwrap();
            assert_eq!(
                stream.clone_dtoh(&fused).unwrap(),
                stream.clone_dtoh(&plain).unwrap(),
                "{name}"
            );
        }
    }

    /// The post-norm, add and feed-forward norm in one launch give the bits of the three, and
    /// leave the quantised row of the last.
    #[test]
    fn the_post_norm_add_norm_matches_its_three_launches() {
        use crate::inference::kernel::fused::fused_post_norm_add_norm;
        use crate::tensor::layer::RmsNorm;
        use crate::tensor::{Device, Tensor};
        let Ok(dev) = Device::new_cuda(0) else {
            return;
        };
        let Ok(raw) = dev.as_cuda_device() else {
            return;
        };
        if !supported(raw.native().ordinal()) {
            return;
        }
        let ndev = raw.native().clone();
        let cols = 2560usize;
        let v = |m: usize, a: f32| -> Vec<f32> {
            (0..cols)
                .map(|i| ((i * m % 197) as f32 - 90.0) * a)
                .collect()
        };
        let a = Tensor::from_vec(v(31, 0.011), (1, cols), &dev).unwrap();
        let r = Tensor::from_vec(v(17, 0.007), (1, cols), &dev).unwrap();
        let wp = Tensor::from_vec(v(13, 0.004), cols, &dev).unwrap();
        let wf = Tensor::from_vec(v(11, 0.005), cols, &dev).unwrap();
        let (x, norm) = fused_post_norm_add_norm(&a, &wp, &r, &wf, 1e-6, 1e-6)
            .unwrap()
            .expect("the row is one the kernel serves");
        let post = RmsNorm::from_tensor(wp, 1e-6);
        let ffn = RmsNorm::from_tensor(wf, 1e-6);
        let x_ref = (post.forward(&a).unwrap() + &r).unwrap();
        let norm_ref = ffn.forward(&x_ref).unwrap();
        let stream = ndev.stream();
        let host = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(host(&x), host(&x_ref));
        assert_eq!(host(&norm), host(&norm_ref));
        let (slice, _) = norm.cuda_f32_slice().unwrap();
        let ptr = {
            use crate::tensor::gpu_runtime::driver::DevicePtr;
            slice.device_ptr(&stream).0 as usize
        };
        let fused = take(ptr, cols).expect("the kernel leaves its row");
        let plain = quantize_q8_1(&ndev, slice, cols).unwrap();
        assert_eq!(
            stream.clone_dtoh(&fused).unwrap(),
            stream.clone_dtoh(&plain).unwrap()
        );
    }

    /// The wide norm-then-add leaves the bits of the narrow kernel's.
    #[test]
    fn the_wide_norm_then_add_matches_the_narrow_kernel() {
        use crate::inference::kernel::fused::fused_rmsnorm_then_add;
        use crate::tensor::gpu_runtime::driver::PushKernelArg;
        use crate::tensor::{Device, Tensor};
        let Ok(dev) = Device::new_cuda(0) else {
            return;
        };
        let Ok(raw) = dev.as_cuda_device() else {
            return;
        };
        let ndev = raw.native().clone();
        let cols = 2560usize;
        let v = |m: usize, a: f32| -> Vec<f32> {
            (0..cols)
                .map(|i| ((i * m % 197) as f32 - 90.0) * a)
                .collect()
        };
        let x = Tensor::from_vec(v(31, 0.011), (1, cols), &dev).unwrap();
        let r = Tensor::from_vec(v(17, 0.007), (1, cols), &dev).unwrap();
        let w = Tensor::from_vec(v(13, 0.004), cols, &dev).unwrap();
        let wide = fused_rmsnorm_then_add(&x, &w, &r, 1e-6).unwrap();
        let stream = ndev.stream();
        let func = ndev.fused_fn("fused_rmsnorm_then_add_f32").unwrap();
        let reference = unsafe { stream.alloc::<f32>(cols) }.unwrap();
        let (xs, _) = x.cuda_f32_slice().unwrap();
        let (ws, _) = w.cuda_f32_slice().unwrap();
        let (rs, _) = r.cuda_f32_slice().unwrap();
        let cfg = crate::tensor::cuda_ext::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 1024,
        };
        let (eps, cols_i32) = (1e-6f32, cols as i32);
        let mut l = stream.launch_builder(&func);
        l.arg(xs)
            .arg(ws)
            .arg(rs)
            .arg(&reference)
            .arg(&eps)
            .arg(&cols_i32);
        unsafe { l.launch(cfg) }.unwrap();
        let got = wide.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got, stream.clone_dtoh(&reference).unwrap());
    }

    /// Two weights of one format over one activation in one launch give each the bits of its
    /// own launch.
    #[test]
    fn a_pair_of_weights_matches_two_launches() {
        use crate::tensor::quantized::{GgmlDType, QTensor};
        use crate::tensor::{Device, Tensor};
        let Ok(dev) = Device::new_cuda(0) else {
            return;
        };
        let k = 2560usize;
        let weight = |n: usize,
                      seed: usize|
         -> crate::inference::generic_transformer::projection::QMatMul {
            let t = Tensor::from_vec(
                (0..n * k)
                    .map(|i| (((i * 7919 + seed) % 2003) as f32 - 1000.0) * 0.0007)
                    .collect(),
                (n, k),
                &dev,
            )
            .unwrap();
            let qt = QTensor::quantize_onto(&t, GgmlDType::Q4K, &dev).unwrap();
            crate::inference::generic_transformer::projection::QMatMul::from_qtensor(qt).unwrap()
        };
        let (wq, wk) = (weight(2048, 3), weight(1024, 11));
        let x = Tensor::from_vec(
            (0..k)
                .map(|i| ((i * 31 % 211) as f32 - 100.0) * 0.013)
                .collect(),
            (1, 1, k),
            &dev,
        )
        .unwrap();
        let a = wq
            .quantize_for_matvec(&x)
            .unwrap()
            .expect("the mat-vec takes the row");
        let (q, kk) = wq
            .forward_quantized_pair(&wk, &a)
            .unwrap()
            .expect("the pair is two weights of one format");
        let host = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(host(&q), host(&wq.forward_quantized(&a).unwrap().unwrap()));
        assert_eq!(host(&kk), host(&wk.forward_quantized(&a).unwrap().unwrap()));
    }
}
