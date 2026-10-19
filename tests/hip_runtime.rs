//! Checks of the HIP runtime surface on the first AMD card listed: the first device of a ROCm
//! build, the first card after the NVIDIA ones in a build with both runtimes. Each test passes
//! without running when no AMD card is listed. Only the runtime module is used, so the binary
//! links without the static kernel libraries.

#![cfg(feature = "rocm")]

use std::sync::Arc;

use loken::tensor::gpu_runtime::cublas::{self, CudaBlas};
use loken::tensor::gpu_runtime::driver::{
    result, sys, CudaContext, CudaFunction, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig,
    PushKernelArg,
};
use loken::tensor::gpu_runtime::nvrtc;

/// The first AMD card and its default stream, or `None` when none is listed.
fn device() -> Option<(Arc<CudaContext>, Arc<CudaStream>)> {
    let n = CudaContext::device_count().ok()?.max(0) as usize;
    let ordinal = (0..n).find(|&i| !loken::tensor::gpu_runtime::is_nvidia_ordinal(i))?;
    let ctx = CudaContext::new(ordinal).expect("open the AMD card");
    let stream = ctx.default_stream();
    Some((ctx, stream))
}

const ADD_ONE: &str = r#"
extern "C" __global__ void add_one(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0f;
}
"#;

fn add_one_fn(ctx: &Arc<CudaContext>) -> CudaFunction {
    let ptx = nvrtc::compile_ptx(ADD_ONE).expect("compile");
    let module = ctx.load_module(ptx).expect("load module");
    module.load_function("add_one").expect("function")
}

fn add_one(
    stream: &Arc<CudaStream>,
    f: &CudaFunction,
    x: &mut loken::tensor::gpu_runtime::driver::CudaSlice<f32>,
) {
    let n = x.len() as i32;
    let mut launch = stream.launch_builder(f);
    launch.arg(x).arg(&n);
    unsafe { launch.launch(LaunchConfig::for_num_elems(n as u32)) }.unwrap();
}

#[test]
fn device_count_name_and_arch() {
    let Some((ctx, _)) = device() else { return };
    assert!(!ctx.name().unwrap().is_empty());
    let arch = result::device::gcn_arch_name(ctx.cu_device()).unwrap();
    assert!(!arch.is_empty());
    let sm = sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
    assert!(ctx.attribute(sm).unwrap() > 0);
    let cc = sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
    assert_eq!(
        ctx.attribute(cc).unwrap_err().0,
        sys::CUresult::CUDA_ERROR_NOT_SUPPORTED
    );
}

#[test]
fn alloc_zeros_reads_zero() {
    let Some((_, stream)) = device() else { return };
    let z = stream.alloc_zeros::<u32>(4096).unwrap();
    assert!(stream.clone_dtoh(&z).unwrap().iter().all(|&v| v == 0));
}

#[test]
fn htod_dtoh_round_trip() {
    let Some((_, stream)) = device() else { return };
    let host: Vec<f32> = (0..10_000).map(|i| i as f32 * 0.5).collect();
    let dev = stream.clone_htod(&host).unwrap();
    assert_eq!(stream.clone_dtoh(&dev).unwrap(), host);
    let mut back = vec![0f32; host.len()];
    stream.memcpy_dtoh(&dev, &mut back).unwrap();
    assert_eq!(back, host);
}

#[test]
fn dtod_copy_of_a_range() {
    let Some((_, stream)) = device() else { return };
    let host: Vec<i32> = (0..256).collect();
    let src = stream.clone_htod(&host).unwrap();
    let mut dst = stream.alloc_zeros::<i32>(128).unwrap();
    stream.memcpy_dtod(&src.slice(64..192), &mut dst).unwrap();
    assert_eq!(stream.clone_dtoh(&dst).unwrap(), host[64..192].to_vec());
    let copy = src.try_clone().unwrap();
    assert_eq!(stream.clone_dtoh(&copy).unwrap(), host);
}

#[test]
fn memset_bytes_and_zeros() {
    let Some((_, stream)) = device() else { return };
    let mut buf = stream.clone_htod(&[7u8; 1000]).unwrap();
    {
        let (p, _g) = buf.device_ptr_mut(&stream);
        let r = unsafe { sys::cuMemsetD8Async(p + 100, 0xAB, 50, stream.cu_stream()) };
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS);
    }
    let got = stream.clone_dtoh(&buf).unwrap();
    assert!(got[..100].iter().all(|&b| b == 7));
    assert!(got[100..150].iter().all(|&b| b == 0xAB));
    assert!(got[150..].iter().all(|&b| b == 7));
    stream.memset_zeros(&mut buf).unwrap();
    assert!(stream.clone_dtoh(&buf).unwrap().iter().all(|&b| b == 0));
}

#[test]
fn compiled_kernel_adds_one() {
    let Some((ctx, stream)) = device() else {
        return;
    };
    let f = add_one_fn(&ctx);
    let host: Vec<f32> = (0..3000).map(|i| i as f32).collect();
    let mut x = stream.clone_htod(&host).unwrap();
    add_one(&stream, &f, &mut x);
    let got = stream.clone_dtoh(&x).unwrap();
    assert!(got.iter().zip(&host).all(|(g, h)| *g == h + 1.0));
}

#[test]
fn code_object_round_trips_through_text() {
    let Some((ctx, stream)) = device() else {
        return;
    };
    let text = nvrtc::compile_ptx(ADD_ONE).unwrap().to_src();
    let module = ctx.load_module(nvrtc::Ptx::from_src(text)).unwrap();
    let f = module.load_function("add_one").unwrap();
    let mut x = stream.alloc_zeros::<f32>(10).unwrap();
    add_one(&stream, &f, &mut x);
    assert!(stream.clone_dtoh(&x).unwrap().iter().all(|&v| v == 1.0));
    let foreign = ctx.load_module(nvrtc::Ptx::from_src(".version 8.0\n.target sm_80\n"));
    assert_eq!(
        foreign.unwrap_err().0,
        sys::CUresult::CUDA_ERROR_NOT_SUPPORTED
    );
}

#[test]
fn events_record_elapsed_time() {
    let Some((ctx, stream)) = device() else {
        return;
    };
    let f = add_one_fn(&ctx);
    let mut x = stream.alloc_zeros::<f32>(1 << 20).unwrap();
    let timing = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
    let start = stream.record_event(timing).unwrap();
    for _ in 0..8 {
        add_one(&stream, &f, &mut x);
    }
    let end = stream.record_event(timing).unwrap();
    assert!(start.elapsed_ms(&end).unwrap() >= 0.0);
    assert!(end.is_complete());
}

#[test]
fn streams_join_and_synchronize() {
    let Some((ctx, stream)) = device() else {
        return;
    };
    let f = add_one_fn(&ctx);
    let side = stream.fork().unwrap();
    let mut x = side.alloc_zeros::<f32>(4096).unwrap();
    add_one(&side, &f, &mut x);
    stream.join(&side).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(
        unsafe { sys::cuStreamQuery(side.cu_stream()) },
        sys::CUresult::CUDA_SUCCESS
    );
    assert!(side.clone_dtoh(&x).unwrap().iter().all(|&v| v == 1.0));
}

#[test]
fn captured_graph_replays() {
    let Some((ctx, _)) = device() else { return };
    let f = add_one_fn(&ctx);
    let stream = ctx.new_stream().unwrap();
    let mut x = stream.alloc_zeros::<f32>(2048).unwrap();
    stream.synchronize().unwrap();
    stream
        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .unwrap();
    assert_eq!(
        stream.capture_status().unwrap(),
        sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE
    );
    add_one(&stream, &f, &mut x);
    let graph = stream
        .end_capture(sys::CUgraphInstantiate_flags(0))
        .unwrap()
        .expect("a captured graph");
    let mut count = 0usize;
    unsafe { sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count) }
        .result()
        .unwrap();
    assert_eq!(count, 1);
    let mut node = std::ptr::null_mut();
    unsafe { sys::cuGraphGetNodes(graph.cu_graph(), &mut node, &mut count) }
        .result()
        .unwrap();
    let mut ty = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
    unsafe { sys::cuGraphNodeGetType(node, &mut ty) }
        .result()
        .unwrap();
    assert_eq!(ty, sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL);
    graph.upload().unwrap();
    for _ in 0..3 {
        graph.launch().unwrap();
    }
    assert!(stream.clone_dtoh(&x).unwrap().iter().all(|&v| v == 3.0));
}

#[test]
fn registered_host_memory_copies_async() {
    let Some((_, stream)) = device() else { return };
    let host: Vec<u64> = (0..65_536).collect();
    let bytes = host.len() * std::mem::size_of::<u64>();
    let p = host.as_ptr() as *mut std::ffi::c_void;
    unsafe { sys::cuMemHostRegister_v2(p, bytes, sys::CU_MEMHOSTREGISTER_PORTABLE) }
        .result()
        .unwrap();
    let dev = unsafe { stream.alloc::<u64>(host.len()) }.unwrap();
    {
        let (d, _g) = dev.device_ptr(&stream);
        unsafe { sys::cuMemcpyHtoDAsync_v2(d, p, bytes, stream.cu_stream()) }
            .result()
            .unwrap();
    }
    stream.synchronize().unwrap();
    unsafe { sys::cuMemHostUnregister(p) }.result().unwrap();
    assert_eq!(stream.clone_dtoh(&dev).unwrap(), host);
}

#[test]
fn pinned_host_memory_round_trip() {
    let Some((_, stream)) = device() else { return };
    let n = 1024usize;
    let bytes = n * std::mem::size_of::<f32>();
    let host = unsafe { result::malloc_host(bytes, sys::CU_MEMHOSTALLOC_PORTABLE) }.unwrap();
    let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let dev = stream.clone_htod(&src).unwrap();
    // SAFETY: the allocation holds `n` floats and nothing else uses it.
    let out = unsafe { std::slice::from_raw_parts_mut(host as *mut f32, n) };
    {
        let (d, _g) = dev.device_ptr(&stream);
        unsafe { result::memcpy_dtoh_async(out, d, stream.cu_stream()) }.unwrap();
    }
    stream.synchronize().unwrap();
    assert_eq!(out, &src[..]);
    unsafe { result::free_host(host) }.unwrap();
}

#[test]
fn sgemm_matches_host_product() {
    use cublas::sys::{cublasComputeType_t, cublasGemmAlgo_t, cublasOperation_t, cudaDataType_t};
    let Some((_, stream)) = device() else { return };
    let (m, n, k) = (33usize, 17usize, 29usize);
    let a: Vec<f32> = (0..m * k).map(|i| ((i * 7) % 13) as f32 - 6.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i * 5) % 11) as f32 - 5.0).collect();
    // Row-major C = A B, run as the column-major product C^T = B^T A^T. Small integers keep
    // every partial sum exact, so the device result equals the host one.
    let mut want = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            want[i * n + j] = (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum();
        }
    }
    let da = stream.clone_htod(&a).unwrap();
    let db = stream.clone_htod(&b).unwrap();
    let mut dc = stream.alloc_zeros::<f32>(m * n).unwrap();
    let mut dc2 = stream.alloc_zeros::<f32>(m * n).unwrap();
    let blas = CudaBlas::new(stream.clone()).unwrap();
    let (alpha, beta) = (1f32, 0f32);
    let op = cublasOperation_t::CUBLAS_OP_N;
    {
        let (pa, _ga) = da.device_ptr(&stream);
        let (pb, _gb) = db.device_ptr(&stream);
        let (pc, _gc) = dc.device_ptr_mut(&stream);
        let s = unsafe {
            cublas::sys::cublasSgemm_v2(
                *blas.handle(),
                op,
                op,
                n as i32,
                m as i32,
                k as i32,
                &alpha,
                pb as *const f32,
                n as i32,
                pa as *const f32,
                k as i32,
                &beta,
                pc as *mut f32,
                n as i32,
            )
        };
        assert_eq!(s, cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        let (pc2, _gc2) = dc2.device_ptr_mut(&stream);
        unsafe {
            cublas::result::gemm_strided_batched_ex(
                *blas.handle(),
                op,
                op,
                n as i32,
                m as i32,
                k as i32,
                &alpha as *const f32 as *const _,
                pb as *const _,
                cudaDataType_t::CUDA_R_32F,
                n as i32,
                (k * n) as i64,
                pa as *const _,
                cudaDataType_t::CUDA_R_32F,
                k as i32,
                (m * k) as i64,
                &beta as *const f32 as *const _,
                pc2 as *mut _,
                cudaDataType_t::CUDA_R_32F,
                n as i32,
                (m * n) as i64,
                1,
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        }
        .unwrap();
    }
    assert_eq!(stream.clone_dtoh(&dc).unwrap(), want);
    assert_eq!(stream.clone_dtoh(&dc2).unwrap(), want);
}

#[test]
fn cuda_headers_compile_with_their_hip_meaning() {
    if device().is_none() {
        return;
    }
    let src = r#"
#include <cuda_fp16.h>
extern "C" __global__ void halve(__half* x) {
    x[threadIdx.x] = __float2half(__half2float(x[threadIdx.x]) * 0.5f);
}
"#;
    if let Err(e) = nvrtc::compile_ptx(src) {
        panic!("{e}");
    }
}
