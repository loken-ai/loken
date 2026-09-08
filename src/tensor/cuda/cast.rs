//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

pub fn cast_f32_to_bf16(
    dev: &CudaDevice,
    x: &CudaSlice<f32>,
    n: usize,
) -> Result<CudaSlice<half::bf16>> {
    let func = dev.elementwise_fn("native_cast_f32_bf16")?;
    let stream = dev.stream();
    let out = with_oom_retry(dev, "cast", || unsafe { stream.alloc::<half::bf16>(n) })?;
    let cfg = elementwise_launch(n);
    let n_i = n as i32;
    let mut b = stream.launch_builder(&func);
    b.arg(x);
    b.arg(&out);
    b.arg(&n_i);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("cast launch: {e}")))?;
    Ok(out)
}

pub fn cast_bf16_to_f32(
    dev: &CudaDevice,
    x: &CudaSlice<half::bf16>,
    n: usize,
) -> Result<CudaSlice<f32>> {
    let func = dev.elementwise_fn("native_cast_bf16_f32")?;
    let stream = dev.stream();
    let out = with_oom_retry(dev, "cast", || unsafe { stream.alloc::<f32>(n) })?;
    let cfg = elementwise_launch(n);
    let n_i = n as i32;
    let mut b = stream.launch_builder(&func);
    b.arg(x);
    b.arg(&out);
    b.arg(&n_i);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("cast launch: {e}")))?;
    Ok(out)
}

/// Batched row-major matmul via cuBLAS for the half-width carriers:
/// `[b, m, k] x [b, k, n] -> [b, m, n]`, both operands untransposed.
///
/// Row-major through column-major cuBLAS by computing Cᵀ = Bᵀ.Aᵀ (operand swap), so the
/// result lands row-major with no transposes. Everything but the element type and the
/// accumulation choice is the same call, so the call is written once and each carrier
/// supplies only what distinguishes it. `$rt` maps the carrier's reduced-precision flag to
/// (compute type, alpha, beta); `_hold` keeps the alpha/beta values alive for the launch,
/// since which type they are depends on that flag.
macro_rules! matmul_nn_impl {
    ($name:ident, $t:ty, $cuda_t:ident, $reduced:ident, $ctx:literal, $err:literal, $rt:expr) => {
        pub fn $name(
            dev: &CudaDevice,
            a: &CudaSlice<$t>,
            b: &CudaSlice<$t>,
            batch: usize,
            m: usize,
            k: usize,
            n: usize,
        ) -> Result<CudaSlice<$t>> {
            use cudarc::cublas::sys;
            use cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let stream = dev.stream();
            let mut out =
                with_oom_retry(dev, $ctx, || unsafe { stream.alloc::<$t>(batch * m * n) })?;
            let (compute_type, alpha_ptr, beta_ptr, _hold): (
                _,
                *const std::ffi::c_void,
                *const std::ffi::c_void,
                Box<dyn std::any::Any>,
            ) = $rt($reduced());
            let blas = dev.blas()?;
            {
                let (a_ptr, _ga) = a.device_ptr(stream);
                let (b_ptr, _gb) = b.device_ptr(stream);
                let (c_ptr, _gc) = out.device_ptr_mut(stream);
                unsafe {
                    cudarc::cublas::result::gemm_strided_batched_ex(
                        *blas.handle(),
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        n as i32,
                        m as i32,
                        k as i32,
                        alpha_ptr,
                        b_ptr as *const _,
                        sys::cudaDataType_t::$cuda_t,
                        n as i32,
                        (k * n) as i64,
                        a_ptr as *const _,
                        sys::cudaDataType_t::$cuda_t,
                        k as i32,
                        (m * k) as i64,
                        beta_ptr,
                        c_ptr as *mut _,
                        sys::cudaDataType_t::$cuda_t,
                        n as i32,
                        (m * n) as i64,
                        batch as i32,
                        compute_type,
                        sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                    )
                }
                .map_err(|e| cublas_err($err, e))?;
            }
            Ok(out)
        }
    };
}

// bf16 accumulates in f32 with f32 alpha/beta either way; the reduced flag only chooses
// FAST_16BF over plain 32F.
matmul_nn_impl!(
    matmul_bf16,
    half::bf16,
    CUDA_R_16BF,
    gemm_reduced_precision_bf16,
    "matmul bf16",
    "bf16 gemm",
    |red: bool| {
        use cudarc::cublas::sys;
        let ab = Box::new((1.0f32, 0.0f32));
        let ct = if red {
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF
        } else {
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
        };
        let (ap, bp) = (
            &ab.0 as *const f32 as *const std::ffi::c_void,
            &ab.1 as *const f32 as *const std::ffi::c_void,
        );
        (ct, ap, bp, ab as Box<dyn std::any::Any>)
    }
);
// f16 defaults to f32 accumulation (COMPUTE_32F, f32 alpha/beta); the reduced flag switches
// to full-fp16 (COMPUTE_16F, f16 alpha/beta). A typed Hgemm always accumulates in f16 - a
// parity drift against the f32 path whenever the flag is off (e.g. the parity harness).
matmul_nn_impl!(
    matmul_f16,
    half::f16,
    CUDA_R_16F,
    gemm_reduced_precision_f16,
    "matmul f16",
    "hgemm",
    |red: bool| {
        use cudarc::cublas::sys;
        if red {
            let ab = Box::new((half::f16::ONE, half::f16::ZERO));
            let (ap, bp) = (
                &ab.0 as *const half::f16 as *const std::ffi::c_void,
                &ab.1 as *const half::f16 as *const std::ffi::c_void,
            );
            (
                sys::cublasComputeType_t::CUBLAS_COMPUTE_16F,
                ap,
                bp,
                ab as Box<dyn std::any::Any>,
            )
        } else {
            let ab = Box::new((1.0f32, 0.0f32));
            let (ap, bp) = (
                &ab.0 as *const f32 as *const std::ffi::c_void,
                &ab.1 as *const f32 as *const std::ffi::c_void,
            );
            (
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                ap,
                bp,
                ab as Box<dyn std::any::Any>,
            )
        }
    }
);

// im2col (1-D) for one (batch, group) slab; returns [c_in_g*k, l_out].
/// Column window [lo_off, lo_off+lo_len) of the im2col matrix, so the
/// transient stays bounded regardless of sequence length (a 47 s audio
/// decode would otherwise need a multi-GB col buffer in one piece).
pub fn im2col1d_f32(
    dev: &CudaDevice,
    x: &cudarc::driver::CudaView<f32>,
    c_in_g: usize,
    l: usize,
    k: usize,
    l_out: usize,
    lo_off: usize,
    lo_len: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> Result<CudaSlice<f32>> {
    let func = dev.elementwise_fn("native_im2col1d_f32")?;
    let st = dev.stream();
    let total = c_in_g * k * lo_len;
    let col = with_oom_retry(dev, "im2col", || st.alloc_zeros::<f32>(total))?;
    let cfg = elementwise_launch(total);
    let (a, b_, c, d) = (c_in_g as i32, l as i32, k as i32, l_out as i32);
    let (o0, o1) = (lo_off as i32, lo_len as i32);
    let (e, f, g) = (padding as i32, stride as i32, dilation as i32);
    let mut bld = st.launch_builder(&func);
    bld.arg(x);
    bld.arg(&col);
    bld.arg(&a);
    bld.arg(&b_);
    bld.arg(&c);
    bld.arg(&d);
    bld.arg(&o0);
    bld.arg(&o1);
    bld.arg(&e);
    bld.arg(&f);
    bld.arg(&g);
    unsafe { bld.launch(cfg) }.map_err(|e| Error(format!("im2col1d launch: {e}")))?;
    Ok(col)
}

/// 1-D convolution contraction (groups 1, N=1) via im2col + cuBLAS hgemm with
/// F32 accumulation: the im2col columns are cast to F16 and multiplied by a
/// pre-cast F16 weight on tensor cores (COMPUTE_32F -> the products accumulate
/// in F32), so the heavy conv runs at roughly half the memory traffic of the
/// F32 path while keeping accumulation precision. `x` is the F32 input slab
/// `[c_in, l]`; `w_f16` the F16 kernel flattened `[c_out, c_in*k]`; returns the
/// F32 output `[c_out, l_out]`. F32 accumulation holds as long as the global
/// reduced-precision-f16 switch is off (its default).
pub fn conv1d_im2col_f16(
    dev: &CudaDevice,
    x: &cudarc::driver::CudaView<f32>,
    w_f16: &CudaSlice<half::f16>,
    c_in: usize,
    l: usize,
    c_out: usize,
    k: usize,
    l_out: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> Result<CudaSlice<f32>> {
    let col_f32 = im2col1d_f32(
        dev, x, c_in, l, k, l_out, 0, l_out, padding, stride, dilation,
    )?;
    let col_f16 = cast_f32_to_f16(dev, &col_f32, c_in * k * l_out)?;
    // [c_out, c_in*k] . [c_in*k, l_out] -> [c_out, l_out], tensor cores / F32 accum.
    let y_f16 = matmul_f16(dev, w_f16, &col_f16, 1, c_out, c_in * k, l_out)?;
    cast_f16_to_f32(dev, &y_f16, c_out * l_out)
}

/// Transposed 1-D convolution (groups 1, dilation 1, N=1) via an F16 tensor-core
/// GEMM for the heavy `c_in` contraction + a `col2im` overlap-add. `x` is the
/// F32 input `[c_in, l_in]`; `wr_f16` the rearranged F16 weight `[k*c_out, c_in]`
/// (`wr[kk*c_out+oc, ci] = W[ci,oc,kk]`); `bias` F32 `[c_out]`. Returns the F32
/// output `[c_out, l_out]`. The GEMM accumulates in F32 (COMPUTE_32F) unless the
/// global reduced-precision-f16 switch is on, so audio precision holds while the
/// `c_in` reduction runs on tensor cores instead of the serial gather kernel.
pub fn convt1d_gemm_f16(
    dev: &CudaDevice,
    x: &CudaSlice<f32>,
    wr_f16: &CudaSlice<half::f16>,
    bias: &CudaSlice<f32>,
    c_in: usize,
    c_out: usize,
    l_in: usize,
    l_out: usize,
    k: usize,
    stride: usize,
    padding: usize,
) -> Result<CudaSlice<f32>> {
    let x_f16 = cast_f32_to_f16(dev, x, c_in * l_in)?;
    // [k*c_out, c_in] . [c_in, l_in] -> colT [k*c_out, l_in], tensor cores / F32 accum.
    let col_t = matmul_f16(dev, wr_f16, &x_f16, 1, k * c_out, c_in, l_in)?;
    let func = dev.elementwise_fn("native_col2im1d_f16")?;
    let stream = dev.stream();
    let n = c_out * l_out;
    let out = with_oom_retry(dev, "col2im", || stream.alloc_zeros::<f32>(n))?;
    let cfg = elementwise_launch(n);
    let args: Vec<i32> = vec![
        c_out as i32,
        l_out as i32,
        k as i32,
        l_in as i32,
        stride as i32,
        padding as i32,
    ];
    let mut b = stream.launch_builder(&func);
    b.arg(&col_t);
    b.arg(bias);
    b.arg(&out);
    for v in &args {
        b.arg(v);
    }
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("col2im launch: {e}")))?;
    Ok(out)
}

/// im2col (2-D) for one (batch, group) slab; returns [c_in_g*kh*kw, h_out*w_out].
pub fn im2col2d_f32(
    dev: &CudaDevice,
    x: &cudarc::driver::CudaView<f32>,
    c_in_g: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    w_out: usize,
    col_off: usize,
    col_len: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> Result<CudaSlice<f32>> {
    let func = dev.elementwise_fn("native_im2col2d_f32")?;
    let st = dev.stream();
    let total = c_in_g * kh * kw * col_len;
    // the gather kernel writes every element - no zero-fill needed
    let col = with_oom_retry(dev, "im2col", || unsafe { st.alloc::<f32>(total) })?;
    let cfg = elementwise_launch(total);
    let args: Vec<i32> = vec![
        c_in_g as i32,
        h as i32,
        w as i32,
        kh as i32,
        kw as i32,
        w_out as i32,
        col_off as i32,
        col_len as i32,
        padding as i32,
        stride as i32,
        dilation as i32,
    ];
    let mut bld = st.launch_builder(&func);
    bld.arg(x);
    bld.arg(&col);
    for v in &args {
        bld.arg(v);
    }
    unsafe { bld.launch(cfg) }.map_err(|e| Error(format!("im2col2d launch: {e}")))?;
    Ok(col)
}

// --- MoE expert GEMV: the build.rs-compiled FFI launcher (libmoe)
// --- driven from native storage. Third kernel-family boundary after NVRTC + cuBLAS.

/// Fused MoE gate+up+SiLU.mul (the production decode MoE kernel) on native
/// buffers. `x` [m, k] f32; gate/up blobs = [e, n, k] GGML blocks;
/// sorted/expert ids per the routing contract. Returns [m*topk, n].
pub fn moe_gate_up_silu_mul(
    dev: &CudaDevice,
    x: &CudaSlice<f32>,
    gate_blob: &CudaSlice<u8>,
    up_blob: &CudaSlice<u8>,
    sorted_token_ids: &CudaSlice<u32>,
    expert_ids: &CudaSlice<u32>,
    num_experts: usize,
    topk: usize,
    m: usize,
    n: usize,
    k: usize,
    quant_code: i32,
) -> Result<CudaSlice<f32>> {
    use cudarc::driver::DevicePtr;
    let stream = dev.stream();
    let size_m = m * topk;
    let out = with_oom_retry(dev, "moe out", || unsafe {
        stream.alloc::<f32>(size_m * n)
    })?;
    let raw_stream = stream.cu_stream() as i64;
    unsafe {
        loken_moe_gemm_gguf_gate_up_silu_mul(
            x.device_ptr(stream).0 as *const f32,
            gate_blob.device_ptr(stream).0 as *const core::ffi::c_void,
            up_blob.device_ptr(stream).0 as *const core::ffi::c_void,
            sorted_token_ids.device_ptr(stream).0 as *const i32,
            expert_ids.device_ptr(stream).0 as *const i32,
            out.device_ptr(stream).0 as *mut f32,
            num_experts as i32,
            topk as i32,
            size_m as i32,
            n as i32,
            k as i32,
            quant_code,
            raw_stream,
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    //! The ignored cases here need real weights, a device, or a reference dump on
    //! this machine; nothing about them is automatic. Run one by name with
    //!   cargo test --release -p loken --lib NAME -- --ignored --nocapture
    use super::*;
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    fn gpu() -> Option<Arc<CudaDevice>> {
        CudaDevice::new(0).ok()
    }

    #[test]
    fn roundtrip_upload_download() {
        let Some(dev) = gpu() else { return }; // no GPU -> skip
        let data: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.5 - 1000.0).collect();
        let cpu = CpuStorage::F32(data.clone());
        let cuda = CudaStorage::upload(&dev, &cpu).unwrap();
        assert_eq!(cuda.dtype(), DType::F32);
        assert_eq!(cuda.len(), data.len());
        match cuda.download(&dev).unwrap() {
            CpuStorage::F32(v) => assert_eq!(v, data),
            other => panic!("wrong dtype {}", other.dtype()),
        }
    }

    /// THE BOUNDARY PROOF: compile a kernel with NVRTC and launch it on
    /// native-owned device memory - no other substrate anywhere in the path.
    /// This is exactly how the existing fused kernels will bind to native
    /// tensors (same stream type, same slice type, same launch builder).
    #[test]
    fn nvrtc_kernel_on_native_storage() {
        let Some(dev) = gpu() else { return };
        const SRC: &str = r#"
extern "C" __global__ void native_silu_f32(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; y[i] = v / (1.0f + expf(-v)); }
}
"#;
        let ptx = cudarc::nvrtc::compile_ptx(SRC).unwrap();
        let module = dev.context().load_module(ptx).unwrap();
        let func = module.load_function("native_silu_f32").unwrap();

        let n = 1543usize;
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 7.0).collect();
        let x = CudaStorage::upload(&dev, &CpuStorage::F32(data.clone())).unwrap();
        let stream = dev.stream();
        let y = stream.alloc_zeros::<f32>(n).unwrap();
        let n_i = n as i32;
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&func);
        b.arg(x.as_f32_slice().unwrap());
        b.arg(&y);
        b.arg(&n_i);
        unsafe { b.launch(cfg) }.unwrap();
        let got = stream.clone_dtoh(&y).unwrap();
        dev.synchronize().unwrap();

        // CPU reference via the native op.
        let want = crate::tensor::Tensor::from_vec_f32(data, vec![n])
            .unwrap()
            .silu()
            .unwrap()
            .to_vec_f32();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1e-6, "idx {i}: {g} vs {w}");
        }
    }

    /// The same native-Tensor op call runs on the GPU when
    /// the storage is CUDA (production kernels) and must match the CPU path.
    #[test]
    fn op_dispatch_gpu_matches_cpu() {
        use crate::tensor::{Device, Tensor};
        let Some(dev) = gpu() else { return };
        let device = Device::Cuda(dev.clone());
        let (rows, cols) = (5usize, 96usize);
        let xs: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 31) as f32) * 0.13 - 2.0)
            .collect();
        let ws: Vec<f32> = (0..cols).map(|i| 1.0 + ((i % 7) as f32) * 0.1).collect();
        let x = Tensor::from_vec_f32(xs, vec![rows, cols]).unwrap();
        let w = Tensor::from_vec_f32(ws, vec![cols]).unwrap();

        let cpu_rms = x.rms_norm(&w, 1e-6).unwrap().to_vec_f32();
        let gpu_rms = x
            .to_device(&device)
            .unwrap()
            .rms_norm(&w.to_device(&device).unwrap(), 1e-6)
            .unwrap()
            .to_vec_f32();
        let cpu_sm = x.softmax_last_dim().unwrap().to_vec_f32();
        let gpu_sm = x
            .to_device(&device)
            .unwrap()
            .softmax_last_dim()
            .unwrap()
            .to_vec_f32();
        for (i, (c, g)) in cpu_rms.iter().zip(&gpu_rms).enumerate() {
            assert!((c - g).abs() < 1e-4, "rms idx {i}: {c} vs {g}");
        }
        for (i, (c, g)) in cpu_sm.iter().zip(&gpu_sm).enumerate() {
            assert!((c - g).abs() < 1e-5, "softmax idx {i}: {c} vs {g}");
        }

        // matmul (cuBLAS, batched) + elementwise add/mul/silu/gelu
        let (bb, m, k, n) = (3usize, 8usize, 16usize, 5usize);
        let av: Vec<f32> = (0..bb * m * k)
            .map(|i| ((i % 23) as f32) * 0.11 - 1.2)
            .collect();
        let bv: Vec<f32> = (0..bb * k * n)
            .map(|i| ((i % 19) as f32) * 0.09 - 0.8)
            .collect();
        let a = Tensor::from_vec_f32(av, vec![bb, m, k]).unwrap();
        let b = Tensor::from_vec_f32(bv, vec![bb, k, n]).unwrap();
        let cpu_mm = a.matmul(&b).unwrap().to_vec_f32();
        let gpu_mm = a
            .to_device(&device)
            .unwrap()
            .matmul(&b.to_device(&device).unwrap())
            .unwrap()
            .to_vec_f32();
        for (i, (c, g)) in cpu_mm.iter().zip(&gpu_mm).enumerate() {
            assert!((c - g).abs() < 1e-3, "matmul idx {i}: {c} vs {g}");
        }

        let xg = x.to_device(&device).unwrap();
        for (cpu_v, gpu_v, name) in [
            (
                x.add(&x).unwrap().to_vec_f32(),
                xg.add(&xg).unwrap().to_vec_f32(),
                "add",
            ),
            (
                x.mul(&x).unwrap().to_vec_f32(),
                xg.mul(&xg).unwrap().to_vec_f32(),
                "mul",
            ),
            (
                x.silu().unwrap().to_vec_f32(),
                xg.silu().unwrap().to_vec_f32(),
                "silu",
            ),
            (
                x.gelu().unwrap().to_vec_f32(),
                xg.gelu().unwrap().to_vec_f32(),
                "gelu",
            ),
        ] {
            for (i, (c, g)) in cpu_v.iter().zip(&gpu_v).enumerate() {
                assert!((c - g).abs() < 1e-5, "{name} idx {i}: {c} vs {g}");
            }
        }
    }

    /// Half-precision compute (unblocks the big f16/bf16 encoders): device
    /// casts roundtrip exactly vs the host path, and hgemm matches the f32
    /// matmul within f16 tolerance.
    #[test]
    fn f16_matmul_and_casts() {
        use crate::tensor::{DType, Device, Tensor};
        let Some(dev) = gpu() else { return };
        let device = Device::Cuda(dev.clone());
        let (b, m, k, n) = (2usize, 6usize, 32usize, 5usize);
        let av: Vec<f32> = (0..b * m * k)
            .map(|i| ((i % 17) as f32) * 0.05 - 0.4)
            .collect();
        let bv: Vec<f32> = (0..b * k * n)
            .map(|i| ((i % 13) as f32) * 0.07 - 0.45)
            .collect();
        let a = Tensor::from_vec_f32(av.clone(), vec![b, m, k]).unwrap();
        let bb = Tensor::from_vec_f32(bv.clone(), vec![b, k, n]).unwrap();

        // device cast roundtrip == host cast roundtrip (identical rounding)
        let host_rt = a
            .to_dtype(DType::F16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec_f32();
        let dev_rt = a
            .to_device(&device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec_f32();
        assert_eq!(host_rt, dev_rt);

        // f16 hgemm vs f32 cpu matmul
        let want = a.matmul(&bb).unwrap().to_vec_f32();
        let got = a
            .to_device(&device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
            .matmul(&bb.to_device(&device).unwrap().to_dtype(DType::F16).unwrap())
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec_f32();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 2e-2 * w.abs().max(0.5),
                "idx {i}: {g} vs {w}"
            );
        }
    }

    /// The mixed-precision recipe the big-encoder migrations use: layer math
    /// at f16 on GPU (hgemm matmuls, f32-accumulated norms/softmax via device
    /// casts) must track the f32 CPU reference within f16 tolerance.
    #[test]
    fn transformer_ops_f16_gpu_track_f32() {
        use crate::tensor::{DType, Device, Tensor};
        let Some(dev) = gpu() else { return };
        let device = Device::Cuda(dev.clone());
        let (b, h, seq, hd) = (1usize, 2usize, 6usize, 16usize);
        let dm = h * hd;
        let data = |n: usize, seed: u32| -> Vec<f32> {
            let mut st = seed.wrapping_mul(2654435761).wrapping_add(12345);
            (0..n)
                .map(|_| {
                    st = st.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((st >> 8) as f32 / (1 << 24) as f32) * 2.0 - 1.0
                })
                .collect()
        };
        let x0 = data(b * seq * dm, 41);
        let wq = data(dm * dm, 42);
        let n1: Vec<f32> = data(dm, 43).iter().map(|v| 1.0 + 0.1 * v).collect();

        let run = |dev: Option<&Device>| -> Vec<f32> {
            let t = |v: &[f32], s: Vec<usize>| {
                let t = Tensor::from_vec_f32(v.to_vec(), s).unwrap();
                match dev {
                    Some(d) => t.to_device(d).unwrap().to_dtype(DType::F16).unwrap(),
                    None => t,
                }
            };
            let x = t(&x0, vec![b, seq, dm]);
            let xn = x.rms_norm(&t(&n1, vec![dm]), 1e-6).unwrap();
            let q = xn.matmul(&t(&wq, vec![1, dm, dm])).unwrap();
            let scores = q
                .reshape(vec![b, seq, h, hd])
                .unwrap()
                .transpose(1, 2)
                .unwrap()
                .scale(1.0 / (hd as f32).sqrt())
                .unwrap();
            let probs = scores.softmax_last_dim().unwrap();
            probs.silu().unwrap().add(&probs).unwrap().to_vec_f32()
        };
        let want = run(None);
        let got = run(Some(&device));
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 2e-2 * w.abs().max(0.2),
                "idx {i}: {g} vs {w}"
            );
        }
    }

    /// Same as the f16 test but at bf16 (T5's dtype - wider exponent range).
    /// The BF16 softmax must equal cast -> f32 softmax -> cast.
    ///
    /// That equality is the WHOLE claim of the fused kernel: it exists to skip two
    /// passes over the largest buffer in attention, and it is only allowed to do so if
    /// the result is the one the f32 path would have produced. The kernel shipped with
    /// no test at all, on my assertion that it was bit-identical - and Flux's attention
    /// routes through it, so an error here is a whole model family rendering wrong.
    ///
    /// Covers the shapes attention actually produces (a wide last dim, many rows) and
    /// the values it actually holds (scores spanning tens, not a tidy 0..1).
    #[test]
    fn bf16_softmax_matches_the_f32_path() {
        use crate::tensor::{DType, Device, Tensor};
        let Some(dev) = gpu() else { return };
        let device = Device::Cuda(dev.clone());
        for (rows, last) in [(8usize, 64usize), (3, 257), (16, 1024), (1, 7)] {
            let v: Vec<f32> = (0..rows * last)
                .map(|i| ((i as f32 * 0.37).sin() * 18.0) - 4.0)
                .collect();
            let t = Tensor::from_vec_f32(v, vec![rows, last])
                .unwrap()
                .to_device(&device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            // The fused BF16 path.
            let got = t
                .softmax_last_dim()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec_f32();
            // The path it replaces, written out explicitly.
            let want = t
                .to_dtype(DType::F32)
                .unwrap()
                .softmax_last_dim()
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec_f32();
            assert_eq!(got.len(), want.len(), "{rows}x{last}: length");
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g, w,
                    "{rows}x{last} idx {i}: fused {g} vs f32-path {w} - the fused kernel is \
                     not the f32 result rounded to bf16"
                );
            }
            // And every row must still be a distribution.
            for r in 0..rows {
                let sum: f32 = got[r * last..(r + 1) * last].iter().sum();
                assert!(
                    (sum - 1.0).abs() < 0.05,
                    "{rows}x{last} row {r} sums to {sum}, not 1 - softmax is broken"
                );
            }
        }
    }

    #[test]
    fn bf16_matmul_and_roundtrip() {
        use crate::tensor::{DType, Device, Tensor};
        let Some(dev) = gpu() else { return };
        let device = Device::Cuda(dev.clone());
        let (b, m, k, n) = (2usize, 4usize, 24usize, 5usize);
        let av: Vec<f32> = (0..b * m * k)
            .map(|i| ((i % 19) as f32) * 0.06 - 0.5)
            .collect();
        let bv: Vec<f32> = (0..b * k * n)
            .map(|i| ((i % 11) as f32) * 0.08 - 0.4)
            .collect();
        let a = Tensor::from_vec_f32(av, vec![b, m, k]).unwrap();
        let bb = Tensor::from_vec_f32(bv, vec![b, k, n]).unwrap();
        let host_rt = a
            .to_dtype(DType::BF16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec_f32();
        let dev_rt = a
            .to_device(&device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec_f32();
        assert_eq!(host_rt, dev_rt);
        let want = a.matmul(&bb).unwrap().to_vec_f32();
        let got = a
            .to_device(&device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .matmul(
                &bb.to_device(&device)
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap(),
            )
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec_f32();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 5e-2 * w.abs().max(0.5),
                "idx {i}: {g} vs {w}"
            );
        }
    }

    #[test]
    #[ignore = "hogs a GPU's free VRAM; run solo"]
    fn conv2d_completes_under_vram_pressure() {
        use crate::tensor::{DType, Device, Tensor};
        let Some(dev) = gpu() else { return };
        let device = Device::Cuda(dev.clone());
        // Reference on CPU first.
        let (b, ci, hw, co, k) = (1usize, 32usize, 96usize, 32usize, 3usize);
        let n_in = b * ci * hw * hw;
        let x_v: Vec<f32> = (0..n_in)
            .map(|i| ((i % 251) as f32 / 125.5) - 1.0)
            .collect();
        let w_v: Vec<f32> = (0..co * ci * k * k)
            .map(|i| ((i % 97) as f32 / 48.5) - 1.0)
            .collect();
        let x = Tensor::from_vec_f32(x_v, vec![b, ci, hw, hw]).unwrap();
        let w = Tensor::from_vec_f32(w_v, vec![co, ci, k, k]).unwrap();
        let want = x.conv2d(&w, 1, 1, 1, 1).unwrap().to_vec_f32();
        // Hog the device down to ~64 MB free, then run the conv there: the
        // OOM nets (reclaim/retry then CPU bounce) must complete it.
        let (free, _) = crate::tensor::cuda_ext::mem_get_info(&device).expect("mem info");
        let hog_elems = free.saturating_sub(64 << 20) / 4;
        let _hog = Tensor::zeros_on((hog_elems.max(1),), DType::F32, &device).expect("hog");
        let got = x
            .to_device(&device)
            .unwrap()
            .conv2d(&w.to_device(&device).unwrap(), 1, 1, 1, 1)
            .expect("conv2d must not OOM under pressure")
            .to_vec_f32();
        for (i, (g, wv)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - wv).abs() <= 1e-3 * wv.abs().max(0.5),
                "idx {i}: {g} vs {wv}"
            );
        }
    }

    /// GPU conv (im2col + cuBLAS) must match the oracle-verified CPU conv.
    #[test]
    fn conv_gpu_matches_cpu() {
        use crate::tensor::{Device, Tensor};
        let Some(dev) = gpu() else { return };
        let device = Device::Cuda(dev.clone());
        let data = |n: usize, seed: u32| -> Vec<f32> {
            let mut st = seed.wrapping_mul(2654435761).wrapping_add(12345);
            (0..n)
                .map(|_| {
                    st = st.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((st >> 8) as f32 / (1 << 24) as f32) * 2.0 - 1.0
                })
                .collect()
        };
        // conv1d
        let (b, ci, l, co, k) = (2usize, 5usize, 33usize, 7usize, 5usize);
        let x = Tensor::from_vec_f32(data(b * ci * l, 70), vec![b, ci, l]).unwrap();
        let w = Tensor::from_vec_f32(data(co * ci * k, 71), vec![co, ci, k]).unwrap();
        let want = x.conv1d(&w, 2, 2, 1, 1).unwrap().to_vec_f32();
        let got = x
            .to_device(&device)
            .unwrap()
            .conv1d(&w.to_device(&device).unwrap(), 2, 2, 1, 1)
            .unwrap()
            .to_vec_f32();
        for (i, (g, wv)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - wv).abs() <= 1e-3 * wv.abs().max(0.5),
                "1d idx {i}: {g} vs {wv}"
            );
        }
        // conv2d
        let (h, w2) = (14usize, 12usize);
        let x = Tensor::from_vec_f32(data(b * ci * h * w2, 72), vec![b, ci, h, w2]).unwrap();
        let wk = Tensor::from_vec_f32(data(co * ci * 9, 73), vec![co, ci, 3, 3]).unwrap();
        let want = x.conv2d(&wk, 1, 2, 1, 1).unwrap().to_vec_f32();
        let got = x
            .to_device(&device)
            .unwrap()
            .conv2d(&wk.to_device(&device).unwrap(), 1, 2, 1, 1)
            .unwrap()
            .to_vec_f32();
        for (i, (g, wv)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - wv).abs() <= 1e-3 * wv.abs().max(0.5),
                "2d idx {i}: {g} vs {wv}"
            );
        }
    }

    /// The flux DiT op set added for the native image-gen path: layer_norm
    /// (one block/row), sigmoid, the general N-d broadcast kernel (middle-axis,
    /// lhs-broadcast "gate", and the rank-5 rope pattern), and the device
    /// f16<->bf16 cast route. GPU must match the oracle-verified CPU paths.
    #[test]
    fn flux_ops_gpu_match_cpu() {
        use crate::tensor::{DType, Device, Tensor};
        let Some(dev) = gpu() else { return };
        let device = Device::Cuda(dev.clone());
        let data = |n: usize, seed: u32| -> Vec<f32> {
            let mut st = seed.wrapping_mul(2654435761).wrapping_add(12345);
            (0..n)
                .map(|_| {
                    st = st.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((st >> 8) as f32 / (1 << 24) as f32) * 2.0 - 1.0
                })
                .collect()
        };
        let close = |a: &[f32], b: &[f32], tol: f32, what: &str| {
            assert_eq!(a.len(), b.len(), "{what}: len");
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                assert!(
                    (x - y).abs() <= tol * (1.0 + y.abs()),
                    "{what}[{i}]: gpu {x} vs cpu {y}"
                );
            }
        };

        // layer_norm, with and without bias (flux norms are weight-less ->
        // unit weight, no bias; also test a real affine)
        let (rows, cols) = (7usize, 192usize);
        let x = Tensor::from_vec_f32(data(rows * cols, 90), vec![rows, cols]).unwrap();
        let w = Tensor::from_vec_f32(
            data(cols, 91)
                .iter()
                .map(|v| 1.0 + 0.1 * v)
                .collect::<Vec<_>>(),
            vec![cols],
        )
        .unwrap();
        let b = Tensor::from_vec_f32(data(cols, 92), vec![cols]).unwrap();
        let xg = x.to_device(&device).unwrap();
        let wg = w.to_device(&device).unwrap();
        let bg = b.to_device(&device).unwrap();
        let want = x.layer_norm(&w, Some(&b), 1e-6).unwrap().to_vec_f32();
        let got = xg.layer_norm(&wg, Some(&bg), 1e-6).unwrap();
        assert!(got.device().is_cuda(), "layer_norm must stay on GPU");
        close(&got.to_vec_f32(), &want, 1e-4, "layer_norm");
        let want_nb = x.layer_norm(&w, None, 1e-6).unwrap().to_vec_f32();
        let got_nb = xg.layer_norm(&wg, None, 1e-6).unwrap().to_vec_f32();
        close(&got_nb, &want_nb, 1e-4, "layer_norm_nobias");
        // CPU-resident params coerced onto the device (straggler path)
        let got_cw = xg.layer_norm(&w, Some(&b), 1e-6).unwrap().to_vec_f32();
        close(&got_cw, &want, 1e-4, "layer_norm_cpu_params");

        // sigmoid
        let want = x.sigmoid().unwrap().to_vec_f32();
        let got = xg.sigmoid().unwrap();
        assert!(got.device().is_cuda(), "sigmoid must stay on GPU");
        close(&got.to_vec_f32(), &want, 1e-6, "sigmoid");

        // general broadcast: middle axis [2,1,8] over [2,3,8]
        let a = Tensor::from_vec_f32(data(2 * 3 * 8, 93), vec![2, 3, 8]).unwrap();
        let m = Tensor::from_vec_f32(data(2 * 8, 94), vec![2, 1, 8]).unwrap();
        let (ag, mg) = (a.to_device(&device).unwrap(), m.to_device(&device).unwrap());
        let want = a.broadcast_add(&m).unwrap().to_vec_f32();
        let got = ag.broadcast_add(&mg).unwrap();
        assert!(got.device().is_cuda(), "broadcast_add nd must stay on GPU");
        close(&got.to_vec_f32(), &want, 0.0, "badd_mid");

        // lhs-broadcast (the flux modulation gate): [1,1,8] * [1,5,8]
        let gate = Tensor::from_vec_f32(data(8, 95), vec![1, 1, 8]).unwrap();
        let xs = Tensor::from_vec_f32(data(5 * 8, 96), vec![1, 5, 8]).unwrap();
        let want = gate.broadcast_mul(&xs).unwrap();
        let got = gate
            .to_device(&device)
            .unwrap()
            .broadcast_mul(&xs.to_device(&device).unwrap())
            .unwrap();
        assert!(got.device().is_cuda(), "gate broadcast must stay on GPU");
        assert_eq!(got.dims(), want.dims());
        close(&got.to_vec_f32(), &want.to_vec_f32(), 0.0, "bmul_gate");

        // the rank-5 rope pattern: [1,1,4,6,2] * [1,3,4,6,1]
        let fr = Tensor::from_vec_f32(data(4 * 6 * 2, 97), vec![1, 1, 4, 6, 2]).unwrap();
        let xq = Tensor::from_vec_f32(data(3 * 4 * 6, 98), vec![1, 3, 4, 6, 1]).unwrap();
        let want = fr.broadcast_mul(&xq).unwrap();
        let got = fr
            .to_device(&device)
            .unwrap()
            .broadcast_mul(&xq.to_device(&device).unwrap())
            .unwrap();
        assert!(got.device().is_cuda(), "rope broadcast must stay on GPU");
        assert_eq!(got.dims(), want.dims());
        close(&got.to_vec_f32(), &want.to_vec_f32(), 0.0, "bmul_rope");

        // f16 <-> bf16 device cast routes through f32 on device
        let h = x.to_dtype(DType::F16).unwrap();
        let want = h
            .to_dtype(DType::BF16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec_f32();
        let got = h
            .to_device(&device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec_f32();
        close(&got, &want, 0.0, "f16_bf16_cast");
    }

    /// Device cat (the KV-cache concat pattern) matches host cat exactly.
    #[test]
    fn cat_gpu_matches_cpu() {
        use crate::tensor::{Device, Tensor};
        let Some(dev) = gpu() else { return };
        let device = Device::Cuda(dev.clone());
        let a: Vec<f32> = (0..2 * 3 * 4 * 5).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..2 * 3 * 2 * 5).map(|i| (i as f32) * -0.5).collect();
        let ta = Tensor::from_vec_f32(a, vec![2, 3, 4, 5]).unwrap();
        let tb = Tensor::from_vec_f32(b, vec![2, 3, 2, 5]).unwrap();
        let want = Tensor::cat(&[&ta, &tb], 2).unwrap().to_vec_f32();
        let ga = ta.to_device(&device).unwrap();
        let gb = tb.to_device(&device).unwrap();
        let got = Tensor::cat(&[&ga, &gb], 2).unwrap();
        assert!(got.device().is_cuda(), "device cat must stay on GPU");
        assert_eq!(got.to_vec_f32(), want);
    }

    /// The VAE op set (group_norm / upsample / pad / affine / channel-bias /
    /// exp): GPU kernels must match the CPU reference.
    #[test]
    fn vae_ops_gpu_match_cpu() {
        use crate::tensor::{Device, Tensor};
        let Some(dev) = gpu() else { return };
        let device = Device::Cuda(dev.clone());
        let close = |a: &[f32], b: &[f32], tol: f32, what: &str| {
            assert_eq!(a.len(), b.len(), "{what}: len");
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                assert!(
                    (x - y).abs() <= tol * (1.0 + y.abs()),
                    "{what}[{i}]: gpu {x} vs cpu {y}"
                );
            }
        };

        // group_norm on [2, 8, 6] with 4 groups
        let x: Vec<f32> = (0..2 * 8 * 6)
            .map(|i| ((i * 37 % 23) as f32) * 0.3 - 2.0)
            .collect();
        let w: Vec<f32> = (0..8).map(|i| 0.5 + i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..8).map(|i| -0.2 + i as f32 * 0.05).collect();
        let tx = Tensor::from_vec_f32(x, vec![2, 8, 6]).unwrap();
        let tw = Tensor::from_vec_f32(w, vec![8]).unwrap();
        let tb = Tensor::from_vec_f32(b, vec![8]).unwrap();
        let want = tx.group_norm(4, &tw, &tb, 1e-6).unwrap().to_vec_f32();
        let got = tx
            .to_device(&device)
            .unwrap()
            .group_norm(
                4,
                &tw.to_device(&device).unwrap(),
                &tb.to_device(&device).unwrap(),
                1e-6,
            )
            .unwrap();
        assert!(got.device().is_cuda());
        close(&got.to_vec_f32(), &want, 1e-5, "group_norm");

        // upsample_nearest2d 2x on [1, 3, 4, 5]
        let x: Vec<f32> = (0..3 * 4 * 5).map(|i| i as f32).collect();
        let tx = Tensor::from_vec_f32(x, vec![1, 3, 4, 5]).unwrap();
        let want = tx.upsample_nearest2d(8, 10).unwrap().to_vec_f32();
        let got = tx
            .to_device(&device)
            .unwrap()
            .upsample_nearest2d(8, 10)
            .unwrap();
        assert_eq!(got.dims(), &[1, 3, 8, 10]);
        assert_eq!(got.to_vec_f32(), want, "upsample");

        // pad_with_zeros right/bottom by 1 on [1, 2, 3, 4] (the downsample pad)
        let x: Vec<f32> = (0..2 * 3 * 4).map(|i| i as f32 + 1.0).collect();
        let tx = Tensor::from_vec_f32(x, vec![1, 2, 3, 4]).unwrap();
        let want = tx
            .pad_with_zeros(3, 0, 1)
            .unwrap()
            .pad_with_zeros(2, 0, 1)
            .unwrap()
            .to_vec_f32();
        let gx = tx.to_device(&device).unwrap();
        let got = gx
            .pad_with_zeros(3, 0, 1)
            .unwrap()
            .pad_with_zeros(2, 0, 1)
            .unwrap();
        assert_eq!(got.dims(), &[1, 2, 4, 5]);
        assert_eq!(got.to_vec_f32(), want, "pad");

        // affine + exp + add_channel_bias on [1, 4, 6]
        let x: Vec<f32> = (0..4 * 6).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let bias: Vec<f32> = vec![0.5, -0.5, 1.0, 0.0];
        let tx = Tensor::from_vec_f32(x, vec![1, 4, 6]).unwrap();
        let tbias = Tensor::from_vec_f32(bias, vec![4]).unwrap();
        let want = tx
            .affine(0.7, -0.3)
            .unwrap()
            .exp()
            .unwrap()
            .add_channel_bias(&tbias)
            .unwrap()
            .to_vec_f32();
        let got = tx
            .to_device(&device)
            .unwrap()
            .affine(0.7, -0.3)
            .unwrap()
            .exp()
            .unwrap()
            .add_channel_bias(&tbias.to_device(&device).unwrap())
            .unwrap();
        close(&got.to_vec_f32(), &want, 1e-6, "affine/exp/bias");
    }

    /// PHASE-3 CAPSTONE: the full pre-norm transformer layer computed on GPU
    /// native tensors (cuBLAS matmul, production rmsnorm/softmax kernels,
    /// native rope/permute/broadcast/elementwise kernels) must match the
    /// CPU-path result.
    #[test]
    fn transformer_layer_gpu_matches_cpu() {
        use crate::tensor::{Device, Tensor};
        let Some(dev) = gpu() else { return };
        let device = Device::Cuda(dev.clone());
        let (b, h, seq, hd) = (1usize, 2usize, 6usize, 16usize);
        let dm = h * hd;
        let ff = 3 * dm;
        let data = |n: usize, seed: u32| -> Vec<f32> {
            let mut st = seed.wrapping_mul(2654435761).wrapping_add(12345);
            (0..n)
                .map(|_| {
                    st = st.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((st >> 8) as f32 / (1 << 24) as f32) * 2.0 - 1.0
                })
                .collect()
        };
        let x0 = data(b * seq * dm, 21);
        let weights: Vec<Vec<f32>> = vec![
            data(dm * dm, 22), // wq
            data(dm * dm, 23), // wk
            data(dm * dm, 24), // wv
            data(dm * dm, 25), // wo
            data(dm * ff, 26), // wg
            data(dm * ff, 27), // wu
            data(ff * dm, 28), // wd
        ];
        let n1: Vec<f32> = data(dm, 29).iter().map(|v| 1.0 + 0.1 * v).collect();
        let n2: Vec<f32> = data(dm, 30).iter().map(|v| 1.0 + 0.1 * v).collect();
        let theta = 10000f32;
        let mut cs = vec![0f32; seq * hd / 2];
        let mut sn = vec![0f32; seq * hd / 2];
        for t in 0..seq {
            for i in 0..hd / 2 {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / hd as f32);
                cs[t * hd / 2 + i] = (t as f32 * freq).cos();
                sn[t * hd / 2 + i] = (t as f32 * freq).sin();
            }
        }
        let mut mask = vec![0f32; seq * seq];
        for q in 0..seq {
            for k in (q + 1)..seq {
                mask[q * seq + k] = f32::NEG_INFINITY;
            }
        }
        let scale = 1.0 / (hd as f32).sqrt();

        // The same layer math, parameterized by target device.
        let layer = |target: &Device| -> Vec<f32> {
            let t = |v: &[f32], s: Vec<usize>| {
                Tensor::from_vec_f32(v.to_vec(), s)
                    .unwrap()
                    .to_device(target)
                    .unwrap()
            };
            let x = t(&x0, vec![b, seq, dm]);
            let xn = x.rms_norm(&t(&n1, vec![dm]), 1e-6).unwrap();
            let proj = |w: &[f32], o: usize| xn.matmul(&t(w, vec![1, dm, o])).unwrap();
            let split = |p: Tensor| {
                p.reshape(vec![b, seq, h, hd])
                    .unwrap()
                    .transpose(1, 2)
                    .unwrap()
            };
            let q = split(proj(&weights[0], dm));
            let k = split(proj(&weights[1], dm));
            let v = split(proj(&weights[2], dm));
            let (tc, ts) = (t(&cs, vec![seq, hd / 2]), t(&sn, vec![seq, hd / 2]));
            let qr = q.rope(&tc, &ts).unwrap();
            let kr = k.rope(&tc, &ts).unwrap();
            let scores = qr
                .matmul(&kr.transpose(2, 3).unwrap())
                .unwrap()
                .scale(scale)
                .unwrap()
                .broadcast_add(&t(&mask, vec![1, 1, seq, seq]))
                .unwrap();
            let attn = scores.softmax_last_dim().unwrap().matmul(&v).unwrap();
            let merged = attn
                .transpose(1, 2)
                .unwrap()
                .reshape(vec![b, seq, dm])
                .unwrap();
            let o = merged.matmul(&t(&weights[3], vec![1, dm, dm])).unwrap();
            let x1 = x.add(&o).unwrap();
            let x1n = x1.rms_norm(&t(&n2, vec![dm]), 1e-6).unwrap();
            let g = x1n
                .matmul(&t(&weights[4], vec![1, dm, ff]))
                .unwrap()
                .silu()
                .unwrap();
            let u = x1n.matmul(&t(&weights[5], vec![1, dm, ff])).unwrap();
            let m = g
                .mul(&u)
                .unwrap()
                .matmul(&t(&weights[6], vec![1, ff, dm]))
                .unwrap();
            x1.add(&m).unwrap().to_vec_f32()
        };

        let cpu_out = layer(&Device::Cpu);
        let gpu_out = layer(&device);
        for (i, (c, g)) in cpu_out.iter().zip(&gpu_out).enumerate() {
            assert!((c - g).abs() < 1e-3, "idx {i}: cpu {c} vs gpu {g}");
        }
    }

    /// PHASE-4 MILESTONE: the production quantized GEMV (mmvq, the kernel
    /// behind every GGUF weight matmul in live decode) driven entirely from
    /// raw native buffers - native GGUF reader loads a REAL Q4_K weight,
    /// native q8_1 quantize + mmvq launch - must match the production
    /// QMatMul wrapper on the same weight and input.
    #[test]
    fn production_mmvq_gemv_matches_oracle() {
        use crate::tensor::quantized::{self, gguf_file};
        let Some(dev) = gpu() else { return };
        // find a real 2-D Q4_K weight in the model store
        let dir = crate::config::Config::load_test()
            .get_ollama_models_dir()
            .join("blobs");
        let Ok(rd) = std::fs::read_dir(&dir) else {
            return;
        };
        let mut blobs: Vec<_> = rd
            .flatten()
            .filter_map(|e| {
                let p = e.path();
                let mut f = std::fs::File::open(&p).ok()?;
                let mut magic = [0u8; 4];
                std::io::Read::read_exact(&mut f, &mut magic).ok()?;
                (magic == crate::tensor::quantized::gguf_file::layout::MAGIC_BYTES)
                    .then_some((e.metadata().ok()?.len(), p))
            })
            .collect();
        blobs.sort();
        let mut found = None;
        'outer: for (_, path) in &blobs {
            let mut f = std::fs::File::open(path).unwrap();
            let Ok(content) = gguf_file::Content::read(&mut f) else {
                continue;
            };
            let mut names: Vec<&String> = content.tensor_infos.keys().collect();
            names.sort();
            for name in names {
                let info = &content.tensor_infos[name];
                if info.ggml_dtype == quantized::GgmlDType::Q4K
                    && info.shape.dims().len() == 2
                    && info.elem_count() < 30_000_000
                {
                    found = Some((path.clone(), name.clone()));
                    break 'outer;
                }
            }
        }
        let Some((path, name)) = found else { return };

        // native: GGUF read -> upload blocks -> q8_1 quantize -> mmvq GEMV
        let mut f = std::fs::File::open(&path).unwrap();
        let content = gguf_file::Content::read(&mut f).unwrap();
        let qt = content.host_tensor(&mut f, &name).unwrap();
        let (n, k) = (qt.dims[0], qt.dims[1]);
        let x: Vec<f32> = (0..k).map(|i| ((i % 89) as f32) * 0.021 - 0.9).collect();

        let stream = dev.stream();
        let blob = stream.clone_htod(qt.data()).unwrap();
        let xg = stream.clone_htod(&x).unwrap();
        let q81 = quantize_q8_1(&dev, &xg, k).unwrap();
        let y = mmvq_gemv_f32(&dev, "q4_k", &blob, &q81, k, n).unwrap();
        let got = stream.clone_dtoh(&y).unwrap();
        dev.synchronize().unwrap();

        // oracle: the production QMatMul wrapper (the live path) on the
        // same weight + input
        let cdev = crate::tensor::Device::new_cuda(0).unwrap();
        let mut f2 = std::fs::File::open(&path).unwrap();
        let ocontent = crate::tensor::quantized::gguf_file::Content::read(&mut f2).unwrap();
        let oq = ocontent.tensor(&mut f2, &name, &cdev).unwrap();
        let qm = crate::tensor::quantized::QMatMul::from_arc(std::sync::Arc::new(oq)).unwrap();
        let ox = crate::tensor::Tensor::from_vec(x, (1, k), &cdev).unwrap();
        let want = qm
            .forward(&ox)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                "{name} [{n}x{k}] idx {i}: native {g} vs reference {w}"
            );
        }
        eprintln!("mmvq parity on {name} [{n}x{k}] OK");
    }

    /// PHASE-4: the production MoE expert kernel (libmoe FFI, the decode MoE
    /// path of every MoE model) driven from raw native buffers, vs the
    /// production rust wrapper on the same REAL expert weights - same
    /// kernel + same inputs, so near bit-exact.
    #[test]
    fn production_moe_kernel_on_native_storage() {
        use crate::tensor::quantized::{self, gguf_file};
        let Some(dev) = gpu() else { return };
        // find a real 3-D Q4_K gate_exps + matching up_exps pair
        let dir = crate::config::Config::load_test()
            .get_ollama_models_dir()
            .join("blobs");
        let Ok(rd) = std::fs::read_dir(&dir) else {
            return;
        };
        let mut blobs: Vec<_> = rd
            .flatten()
            .filter_map(|e| {
                let p = e.path();
                let mut f = std::fs::File::open(&p).ok()?;
                let mut magic = [0u8; 4];
                std::io::Read::read_exact(&mut f, &mut magic).ok()?;
                (magic == crate::tensor::quantized::gguf_file::layout::MAGIC_BYTES)
                    .then_some((e.metadata().ok()?.len(), p))
            })
            .collect();
        blobs.sort();
        let mut found = None;
        'outer: for (_, path) in &blobs {
            let mut f = std::fs::File::open(path).unwrap();
            let Ok(content) = gguf_file::Content::read(&mut f) else {
                continue;
            };
            let mut names: Vec<&String> = content.tensor_infos.keys().collect();
            names.sort();
            for name in names {
                let info = &content.tensor_infos[name];
                if name.ends_with("ffn_gate_exps.weight")
                    && info.ggml_dtype == quantized::GgmlDType::Q4K
                    && info.shape.dims().len() == 3
                    && info.elem_count() < 300_000_000
                {
                    let up = name.replace("ffn_gate_exps", "ffn_up_exps");
                    if content.tensor_infos.contains_key(&up) {
                        found = Some((path.clone(), name.clone(), up));
                        break 'outer;
                    }
                }
            }
        }
        let Some((path, gate_name, up_name)) = found else {
            return;
        };

        let mut f = std::fs::File::open(&path).unwrap();
        let content = gguf_file::Content::read(&mut f).unwrap();
        let gate = content.host_tensor(&mut f, &gate_name).unwrap();
        let up = content.host_tensor(&mut f, &up_name).unwrap();
        let (e, n, k) = (gate.dims[0], gate.dims[1], gate.dims[2]);
        let (m, topk) = (1usize, 2usize);
        let x: Vec<f32> = (0..m * k).map(|i| ((i % 61) as f32) * 0.03 - 0.9).collect();
        let sorted: Vec<u32> = vec![0, 1];
        let experts: Vec<u32> = vec![1.min(e as u32 - 1), (e as u32 - 1).min(3)];

        // native: raw blobs + ids on native storage, FFI launch
        let stream = dev.stream();
        let g_blob = stream.clone_htod(gate.data()).unwrap();
        let u_blob = stream.clone_htod(up.data()).unwrap();
        let xg = stream.clone_htod(&x).unwrap();
        let sg = stream.clone_htod(&sorted).unwrap();
        let eg = stream.clone_htod(&experts).unwrap();
        let y = moe_gate_up_silu_mul(
            &dev, &xg, &g_blob, &u_blob, &sg, &eg, e, topk, m, n, k, /*Q4K=*/ 1,
        )
        .unwrap();
        let got = stream.clone_dtoh(&y).unwrap();
        dev.synchronize().unwrap();

        // oracle: the production rust wrapper, same weights/input
        let cdev = crate::tensor::Device::new_cuda(0).unwrap();
        let mut f2 = std::fs::File::open(&path).unwrap();
        let ocontent = crate::tensor::quantized::gguf_file::Content::read(&mut f2).unwrap();
        let og = ocontent.tensor(&mut f2, &gate_name, &cdev).unwrap();
        let ou = ocontent.tensor(&mut f2, &up_name, &cdev).unwrap();
        let ox = crate::tensor::Tensor::from_vec(x, (m, k), &cdev).unwrap();
        let os = crate::tensor::Tensor::from_vec(sorted, 2, &cdev).unwrap();
        let oe = crate::tensor::Tensor::from_vec(experts, 2, &cdev).unwrap();
        let want = crate::inference::moe_cuda::moe_gemm_gguf_gate_up_silu_mul(
            &ox, &og, &ou, &os, &oe, topk,
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                "{gate_name} [{e}x{n}x{k}] idx {i}: native {g} vs prod {w}"
            );
        }
        eprintln!("MoE kernel parity on {gate_name} [{e}x{n}x{k}] OK");
    }

    /// MILESTONE: the REAL production kernel (fused_rmsnorm_f32, the one the
    /// live decode path launches every layer) compiled from its in-crate source
    /// and run on NATIVE tensors end-to-end: to_device upload -> launch ->
    /// from_cuda_storage wrap -> download. Must match the native CPU rms_norm.
    #[test]
    fn production_rmsnorm_kernel_on_native_tensors() {
        use crate::tensor::{Device, Tensor};
        let Some(dev) = gpu() else { return };
        let ptx =
            cudarc::nvrtc::compile_ptx(crate::inference::kernel::fused::fused_cuda_src()).unwrap();
        let module = dev.context().load_module(ptx).unwrap();
        let func = module.load_function("fused_rmsnorm_f32").unwrap();

        let (rows, cols) = (7usize, 384usize);
        let xs: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 97) as f32) * 0.07 - 3.0)
            .collect();
        let ws: Vec<f32> = (0..cols).map(|i| 1.0 + ((i % 13) as f32) * 0.05).collect();
        let eps = 1e-6f32;

        let device = Device::Cuda(dev.clone());
        let x = Tensor::from_vec_f32(xs.clone(), vec![rows, cols])
            .unwrap()
            .to_device(&device)
            .unwrap();
        let w = Tensor::from_vec_f32(ws.clone(), vec![cols])
            .unwrap()
            .to_device(&device)
            .unwrap();

        let stream = dev.stream();
        let out = stream.alloc_zeros::<f32>(rows * cols).unwrap();
        let block = 256u32.min(cols as u32).max(1).next_power_of_two();
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: block * 4,
        };
        let cols_i32 = cols as i32;
        let (x_slice, _) = x.cuda_f32_slice().unwrap();
        let (w_slice, _) = w.cuda_f32_slice().unwrap();
        let mut b = stream.launch_builder(&func);
        b.arg(x_slice);
        b.arg(w_slice);
        b.arg(&out);
        b.arg(&eps);
        b.arg(&cols_i32);
        unsafe { b.launch(cfg) }.unwrap();

        let result =
            Tensor::from_cuda_storage(CudaStorage::F32(out), dev.clone(), vec![rows, cols])
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap();

        let want = Tensor::from_vec_f32(xs, vec![rows, cols])
            .unwrap()
            .rms_norm(&Tensor::from_vec_f32(ws, vec![cols]).unwrap(), eps)
            .unwrap();
        let (got, want) = (result.to_vec_f32(), want.to_vec_f32());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1e-4, "idx {i}: {g} vs {w}");
        }
    }

    /// every new op's GPU dispatch (device kernel or
    /// documented host fallback) must match the oracle-verified CPU path.
    #[test]
    fn w1_ops_gpu_match_cpu() {
        use crate::tensor::{DType, Device, Tensor};
        let Some(dev) = gpu() else { return };
        let device = Device::Cuda(dev.clone());
        let data = |n: usize, seed: u32| -> Vec<f32> {
            let mut st = seed.wrapping_mul(2654435761).wrapping_add(12345);
            (0..n)
                .map(|_| {
                    st = st.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((st >> 8) as f32 / (1 << 24) as f32) * 2.0 - 1.0
                })
                .collect()
        };
        let close = |a: &[f32], b: &[f32], tol: f32, what: &str| {
            assert_eq!(a.len(), b.len(), "{what}: len");
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                assert!(
                    (x - y).abs() <= tol * (1.0 + y.abs()),
                    "{what}[{i}]: gpu {x} vs cpu {y}"
                );
            }
        };

        // unaries (+ powf scalar) on device kernels
        let xp: Vec<f32> = data(97, 1).iter().map(|v| v.abs() + 0.4).collect();
        let x = Tensor::from_vec_f32(xp, vec![97]).unwrap();
        let xg = x.to_device(&device).unwrap();
        for (c, g, name) in [
            (x.tanh(), xg.tanh(), "tanh"),
            (x.abs(), xg.abs(), "abs"),
            (x.recip(), xg.recip(), "recip"),
            (x.sqrt(), xg.sqrt(), "sqrt"),
            (x.powf(1.3), xg.powf(1.3), "powf"),
        ] {
            let g = g.unwrap();
            assert!(g.device().is_cuda(), "{name} must stay on GPU");
            close(&g.to_vec_f32(), &c.unwrap().to_vec_f32(), 1e-5, name);
        }

        // broadcast_div: tail-aligned + middle-axis (nd kernel op 2)
        let a = Tensor::from_vec_f32(data(2 * 3 * 8, 2), vec![2, 3, 8]).unwrap();
        let w = Tensor::from_vec_f32(
            data(8, 3).iter().map(|v| v + 2.0).collect::<Vec<_>>(),
            vec![8],
        )
        .unwrap();
        let m = Tensor::from_vec_f32(
            data(2 * 8, 4).iter().map(|v| v + 2.0).collect::<Vec<_>>(),
            vec![2, 1, 8],
        )
        .unwrap();
        let (ag, wg, mg) = (
            a.to_device(&device).unwrap(),
            w.to_device(&device).unwrap(),
            m.to_device(&device).unwrap(),
        );
        let got = ag.broadcast_div(&wg).unwrap();
        assert!(got.device().is_cuda(), "bdiv tail must stay on GPU");
        close(
            &got.to_vec_f32(),
            &a.broadcast_div(&w).unwrap().to_vec_f32(),
            1e-6,
            "bdiv_tail",
        );
        let got = ag.broadcast_div(&mg).unwrap();
        assert!(got.device().is_cuda(), "bdiv nd must stay on GPU");
        close(
            &got.to_vec_f32(),
            &a.broadcast_div(&m).unwrap().to_vec_f32(),
            1e-6,
            "bdiv_nd",
        );

        // broadcast_as f32 (stride-gather kernel) + f16 cast route
        let want = m.broadcast_as(vec![2, 3, 8]).unwrap().to_vec_f32();
        let got = mg.broadcast_as(vec![2, 3, 8]).unwrap();
        assert!(got.device().is_cuda(), "broadcast_as must stay on GPU");
        close(&got.to_vec_f32(), &want, 0.0, "broadcast_as");
        let got16 = mg
            .to_dtype(DType::F16)
            .unwrap()
            .broadcast_as(vec![2, 3, 8])
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        close(&got16.to_vec_f32(), &want, 1e-2, "broadcast_as_f16");

        // gather: f32 + f16 values, u32 + i64 indices (device kernel)
        let table = Tensor::from_vec_f32(data(3 * 5, 5), vec![3, 5]).unwrap();
        let ids_u = Tensor::from_vec_u32(vec![4, 0, 2, 2, 1, 3], vec![3, 2]).unwrap();
        let ids_i = Tensor::from_vec_i64(vec![1, 3, 0, 4, 2, 2], vec![3, 2]).unwrap();
        let (tg, iug, iig) = (
            table.to_device(&device).unwrap(),
            ids_u.to_device(&device).unwrap(),
            ids_i.to_device(&device).unwrap(),
        );
        for (ids_c, ids_g, name) in [(&ids_u, &iug, "gather_u32"), (&ids_i, &iig, "gather_i64")] {
            let want = table.gather(ids_c, 1).unwrap().to_vec_f32();
            let got = tg.gather(ids_g, 1).unwrap();
            assert!(got.device().is_cuda(), "{name} must stay on GPU");
            close(&got.to_vec_f32(), &want, 0.0, name);
        }
        let t16 = tg.to_dtype(DType::F16).unwrap();
        let got = t16.gather(&iug, 1).unwrap();
        assert_eq!(got.dtype(), DType::F16);
        close(
            &got.to_dtype(DType::F32).unwrap().to_vec_f32(),
            &table.gather(&ids_u, 1).unwrap().to_vec_f32(),
            1e-2,
            "gather_f16",
        );

        // where_cond: u8 + u32 condition kernels on f32 values
        let cu: Vec<u32> = (0..24).map(|i| (i % 3 == 0) as u32).collect();
        let cb: Vec<u8> = cu.iter().map(|&x| x as u8).collect();
        let tt = Tensor::from_vec_f32(data(24, 6), vec![4, 6]).unwrap();
        let ff = Tensor::from_vec_f32(data(24, 7), vec![4, 6]).unwrap();
        let cu_t = Tensor::from_vec_u32(cu, vec![4, 6]).unwrap();
        let cb_t = Tensor::from_storage(crate::tensor::CpuStorage::U8(cb), vec![4, 6]).unwrap();
        let want = cu_t.where_cond(&tt, &ff).unwrap().to_vec_f32();
        let (tg2, fg2) = (
            tt.to_device(&device).unwrap(),
            ff.to_device(&device).unwrap(),
        );
        for (c, name) in [(&cu_t, "where_u32"), (&cb_t, "where_u8")] {
            let got = c
                .to_device(&device)
                .unwrap()
                .where_cond(&tg2, &fg2)
                .unwrap();
            assert!(got.device().is_cuda(), "{name} must stay on GPU");
            close(&got.to_vec_f32(), &want, 0.0, name);
        }

        // slice_set in place on device, f32 + f16 (the KV-append shape) + i64
        for dt in [DType::F32, DType::F16] {
            let dst_c = Tensor::from_vec_f32(data(2 * 5 * 3, 8), vec![2, 5, 3])
                .unwrap()
                .to_dtype(dt)
                .unwrap();
            let src_c = Tensor::from_vec_f32(data(2 * 2 * 3, 9), vec![2, 2, 3])
                .unwrap()
                .to_dtype(dt)
                .unwrap();
            let dst_g = dst_c.to_device(&device).unwrap();
            let src_g = src_c.to_device(&device).unwrap();
            dst_c.slice_set(&src_c, 1, 2).unwrap();
            dst_g.slice_set(&src_g, 1, 2).unwrap();
            assert!(dst_g.device().is_cuda());
            close(
                &dst_g.to_dtype(DType::F32).unwrap().to_vec_f32(),
                &dst_c.to_dtype(DType::F32).unwrap().to_vec_f32(),
                0.0,
                "slice_set",
            );
        }
        let pos_c = Tensor::from_vec_i64(vec![0; 6], vec![6]).unwrap();
        let pos_g = pos_c.to_device(&device).unwrap();
        let fill = Tensor::from_vec_i64(vec![41, 42], vec![2]).unwrap();
        pos_g
            .slice_set(&fill.to_device(&device).unwrap(), 0, 3)
            .unwrap();
        assert_eq!(pos_g.to_vec_i64().unwrap(), vec![0, 0, 0, 41, 42, 0]);

        // scatter_set in place on device: i64 idx (the graph KV write) on
        // f32 and f16 values
        for dt in [DType::F32, DType::F16] {
            let dst_c = Tensor::from_vec_f32(data(2 * 6 * 3, 10), vec![2, 6, 3])
                .unwrap()
                .to_dtype(dt)
                .unwrap();
            let src_c = Tensor::from_vec_f32(data(2 * 3, 11), vec![2, 1, 3])
                .unwrap()
                .to_dtype(dt)
                .unwrap();
            let idx = Tensor::from_vec_i64(vec![4, 4, 4, 2, 2, 2], vec![2, 1, 3]).unwrap();
            let dst_g = dst_c.to_device(&device).unwrap();
            dst_c.scatter_set(&idx, &src_c, 1).unwrap();
            dst_g
                .scatter_set(
                    &idx.to_device(&device).unwrap(),
                    &src_c.to_device(&device).unwrap(),
                    1,
                )
                .unwrap();
            close(
                &dst_g.to_dtype(DType::F32).unwrap().to_vec_f32(),
                &dst_c.to_dtype(DType::F32).unwrap().to_vec_f32(),
                0.0,
                "scatter_set",
            );
        }

        // sort / argmax on GPU tensors (documented host fallback) round-trip
        let sx = Tensor::from_vec_f32(data(2 * 7, 12), vec![2, 7]).unwrap();
        let sg = sx.to_device(&device).unwrap();
        let (cv, ci) = sx.sort_last_dim(false).unwrap();
        let (gv, gi) = sg.sort_last_dim(false).unwrap();
        assert!(gv.device().is_cuda() && gi.device().is_cuda());
        close(&gv.to_vec_f32(), &cv.to_vec_f32(), 0.0, "sort_vals");
        assert_eq!(gi.to_vec_u32().unwrap(), ci.to_vec_u32().unwrap());
        assert_eq!(
            sg.argmax(1).unwrap().to_vec_u32().unwrap(),
            sx.argmax(1).unwrap().to_vec_u32().unwrap()
        );

        // i64 device storage upload/download round-trip
        let iv: Vec<i64> = vec![-9, 0, 1 << 40, 7];
        let it = Tensor::from_vec_i64(iv.clone(), vec![4])
            .unwrap()
            .to_device(&device)
            .unwrap();
        assert_eq!(it.to_vec_i64().unwrap(), iv);
    }
}
