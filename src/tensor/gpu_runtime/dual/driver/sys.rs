//! The raw driver items under their CUDA names. The types are the HIP implementation's; each
//! call goes to the runtime that owns what it names, converting to `cudarc`'s types for CUDA.

#![allow(non_camel_case_types, non_snake_case, clippy::missing_safety_doc)]

use std::ffi::{c_char, c_int, c_uint, c_void};

use super::super::{bound, stream_runtime, topology, Handles, Runtime};
use crate::tensor::gpu_runtime::hip::driver::sys as hs;
use cudarc::driver::sys as cu;

pub use hs::{
    cudaError_enum, CUDA_KERNEL_NODE_PARAMS_v2_st, CUDA_MEMCPY2D_st, CUarray, CUarray_st,
    CUcontext, CUctx_st, CUdevice, CUdevice_attribute, CUdevice_attribute_enum, CUdeviceptr,
    CUevent, CUevent_flags, CUevent_flags_enum, CUevent_record_flags, CUevent_record_flags_enum,
    CUevent_st, CUfunc_st, CUfunction, CUgraph, CUgraphExec, CUgraphExecUpdateResult,
    CUgraphExecUpdateResultInfo, CUgraphExecUpdateResultInfo_st, CUgraphExecUpdateResult_enum,
    CUgraphExec_st, CUgraphInstantiate_flags, CUgraphInstantiate_flags_enum, CUgraphMem_attribute,
    CUgraphMem_attribute_enum, CUgraphNode, CUgraphNodeType, CUgraphNodeType_enum, CUgraphNode_st,
    CUgraph_st, CUkern_st, CUkernel, CUmemAllocationGranularity_flags,
    CUmemAllocationGranularity_flags_enum, CUmemAllocationProp, CUmemAllocationProp_st,
    CUmemAllocationProp_st__bindgen_ty_1, CUmemAllocationType, CUmemAllocationType_enum,
    CUmemLocation, CUmemLocationType, CUmemLocationType_enum, CUmemLocation_st,
    CUmemLocation_st__bindgen_ty_1, CUmemPoolHandle_st, CUmemPoolProps, CUmemPoolProps_st,
    CUmemPool_attribute, CUmemPool_attribute_enum, CUmemoryPool, CUmemorytype, CUmemorytype_enum,
    CUmod_st, CUmodule, CUresult, CUstream, CUstreamCaptureMode, CUstreamCaptureMode_enum,
    CUstreamCaptureStatus, CUstreamCaptureStatus_enum, CUstream_st, CUDA_KERNEL_NODE_PARAMS,
    CUDA_MEMCPY2D, CU_MEMHOSTALLOC_DEVICEMAP, CU_MEMHOSTALLOC_PORTABLE,
    CU_MEMHOSTALLOC_WRITECOMBINED, CU_MEMHOSTREGISTER_DEVICEMAP, CU_MEMHOSTREGISTER_IOMEMORY,
    CU_MEMHOSTREGISTER_PORTABLE, CU_MEMHOSTREGISTER_READ_ONLY,
};

/// Whether a runtime whose driver library opens is in the process.
pub unsafe fn is_culib_present() -> bool {
    super::super::cuda_usable() || super::super::hip_usable()
}

/// A CUDA status as the CUDA-numbered status the GPU paths match on.
#[inline]
pub(crate) fn res(r: cu::CUresult) -> CUresult {
    CUresult::from_hip(r as u32 as c_int)
}

/// The CUDA status of a status: every variant carries CUDA's own number.
pub(crate) fn to_cuda_result(e: CUresult) -> Option<cu::CUresult> {
    // SAFETY: `cudarc`'s status enum is CUDA's, which holds every number the HIP
    // implementation's variants carry.
    Some(unsafe { same::<CUresult, cu::CUresult>(e) })
}

/// A value of one of the HIP implementation's CUDA-numbered enums as `cudarc`'s enum of the same
/// name.
///
/// # Safety
/// `B` must hold the number `a` carries.
#[inline]
pub(crate) unsafe fn same<A: Copy, B: Copy>(a: A) -> B {
    const { assert!(std::mem::size_of::<A>() == std::mem::size_of::<B>()) };
    unsafe { std::mem::transmute_copy(&a) }
}

/// A handle of the HIP implementation as `cudarc`'s handle of the same kind.
#[inline]
pub(crate) fn cuh<A, B>(p: *mut A) -> *mut B {
    p as *mut B
}

/// The runtime and number of the card `dev` names.
fn device(dev: CUdevice) -> Option<(Runtime, c_int)> {
    let (rt, local) = topology().locate(usize::try_from(dev).ok()?)?;
    Some((rt, local as c_int))
}

/// Routes a call on a card: CUDA's form with the card's CUDA number, HIP's with its HIP number.
macro_rules! on_device {
    ($dev:expr, |$d:ident| cuda => $c:expr, hip => $h:expr) => {
        match device($dev) {
            None => CUresult::CUDA_ERROR_INVALID_DEVICE,
            Some((Runtime::Cuda, $d)) => res($c),
            Some((Runtime::Hip, $d)) => $h,
        }
    };
}

/// Routes a call on a stream to the runtime that created it.
macro_rules! on_stream {
    ($s:expr, cuda => $c:expr, hip => $h:expr) => {
        match stream_runtime($s) {
            Runtime::Cuda => res($c),
            Runtime::Hip => $h,
        }
    };
}

/// Routes a call to the runtime bound on the calling thread.
macro_rules! on_bound {
    (cuda => $c:expr, hip => $h:expr) => {
        match bound() {
            Runtime::Cuda => res($c),
            Runtime::Hip => $h,
        }
    };
}

pub unsafe fn cuInit(flags: c_uint) -> CUresult {
    let _ = topology();
    let mut r = CUresult::CUDA_ERROR_NOT_INITIALIZED;
    if super::super::cuda_usable() {
        r = res(unsafe { cu::cuInit(flags) });
    }
    if r != CUresult::CUDA_SUCCESS && super::super::hip_usable() {
        r = unsafe { hs::cuInit(flags) };
    }
    r
}

pub unsafe fn cuDeviceGetCount(count: *mut c_int) -> CUresult {
    unsafe { *count = topology().count() as c_int };
    CUresult::CUDA_SUCCESS
}

pub unsafe fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult {
    if ordinal < 0 || ordinal as usize >= topology().count() {
        return CUresult::CUDA_ERROR_INVALID_DEVICE;
    }
    unsafe { *device = ordinal };
    CUresult::CUDA_SUCCESS
}

pub unsafe fn cuDeviceGetAttribute(
    value: *mut c_int,
    attrib: CUdevice_attribute,
    dev: CUdevice,
) -> CUresult {
    on_device!(dev, |d| cuda => unsafe { cu::cuDeviceGetAttribute(value, same(attrib), d) },
        hip => unsafe { hs::cuDeviceGetAttribute(value, attrib, d) })
}

pub unsafe fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult {
    on_device!(dev, |d| cuda => unsafe { cu::cuDeviceGetName(name, len, d) },
        hip => unsafe { hs::cuDeviceGetName(name, len, d) })
}

pub unsafe fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult {
    on_device!(dev, |d| cuda => unsafe { cu::cuDeviceTotalMem_v2(bytes, d) },
        hip => unsafe { hs::cuDeviceTotalMem_v2(bytes, d) })
}

pub unsafe fn cuDeviceGetPCIBusId(id: *mut c_char, len: c_int, dev: CUdevice) -> CUresult {
    on_device!(dev, |d| cuda => unsafe { cu::cuDeviceGetPCIBusId(id, len, d) },
        hip => unsafe { hs::cuDeviceGetPCIBusId(id, len, d) })
}

/// Cards of different runtimes cannot address each other: their tensors move through host
/// memory.
pub unsafe fn cuDeviceCanAccessPeer(can: *mut c_int, dev: CUdevice, peer: CUdevice) -> CUresult {
    match (device(dev), device(peer)) {
        (Some((Runtime::Cuda, d)), Some((Runtime::Cuda, p))) => {
            res(unsafe { cu::cuDeviceCanAccessPeer(can, d, p) })
        }
        (Some((Runtime::Hip, d)), Some((Runtime::Hip, p))) => unsafe {
            hs::cuDeviceCanAccessPeer(can, d, p)
        },
        (Some(_), Some(_)) => {
            unsafe { *can = 0 };
            CUresult::CUDA_SUCCESS
        }
        _ => CUresult::CUDA_ERROR_INVALID_DEVICE,
    }
}

/// The primary contexts and memory pools the HIP runtime handed out.
static HIP_CONTEXTS: Handles = Handles::new();
static HIP_POOLS: Handles = Handles::new();

pub unsafe fn cuDevicePrimaryCtxRetain(ctx: *mut CUcontext, dev: CUdevice) -> CUresult {
    on_device!(dev, |d| cuda => unsafe { cu::cuDevicePrimaryCtxRetain(ctx as *mut cu::CUcontext, d) },
    hip => {
        let r = unsafe { hs::cuDevicePrimaryCtxRetain(ctx, d) };
        if r == CUresult::CUDA_SUCCESS {
            HIP_CONTEXTS.insert(unsafe { *ctx });
        }
        r
    })
}

/// Makes `ctx` current on the calling thread, and its runtime the bound one.
pub unsafe fn cuCtxPushCurrent_v2(ctx: CUcontext) -> CUresult {
    if HIP_CONTEXTS.contains(ctx) {
        super::super::bind(Runtime::Hip);
        unsafe { hs::cuCtxPushCurrent_v2(ctx) }
    } else {
        super::super::bind(Runtime::Cuda);
        res(unsafe { cu::cuCtxPushCurrent_v2(cuh(ctx)) })
    }
}

pub unsafe fn cuCtxPopCurrent_v2(ctx: *mut CUcontext) -> CUresult {
    on_bound!(cuda => unsafe { cu::cuCtxPopCurrent_v2(ctx as *mut cu::CUcontext) },
        hip => unsafe { hs::cuCtxPopCurrent_v2(ctx) })
}

/// The runtime of a memory pool.
fn pool_runtime(pool: CUmemoryPool) -> Runtime {
    if HIP_POOLS.contains(pool) {
        Runtime::Hip
    } else if topology().cuda == 0 {
        bound()
    } else {
        Runtime::Cuda
    }
}

/// Routes a call on a memory pool to the runtime that made it.
macro_rules! on_pool {
    ($p:expr, cuda => $c:expr, hip => $h:expr) => {
        match pool_runtime($p) {
            Runtime::Cuda => res($c),
            Runtime::Hip => $h,
        }
    };
}

pub unsafe fn cuDevicePrimaryCtxRelease_v2(dev: CUdevice) -> CUresult {
    on_device!(dev, |d| cuda => unsafe { cu::cuDevicePrimaryCtxRelease_v2(d) },
        hip => unsafe { hs::cuDevicePrimaryCtxRelease_v2(d) })
}

pub unsafe fn cuDevicePrimaryCtxGetState(
    dev: CUdevice,
    flags: *mut c_uint,
    active: *mut c_int,
) -> CUresult {
    on_device!(dev, |d| cuda => unsafe { cu::cuDevicePrimaryCtxGetState(d, flags, active) },
        hip => unsafe { hs::cuDevicePrimaryCtxGetState(d, flags, active) })
}

pub unsafe fn cuCtxSynchronize() -> CUresult {
    on_bound!(cuda => unsafe { cu::cuCtxSynchronize() }, hip => unsafe { hs::cuCtxSynchronize() })
}

/// The page-locked host allocations and registrations the HIP runtime made, freed by it.
static HIP_HOST: Handles = Handles::new();

pub unsafe fn cuMemHostAlloc(p: *mut *mut c_void, bytes: usize, flags: c_uint) -> CUresult {
    match bound() {
        Runtime::Cuda => res(unsafe { cu::cuMemHostAlloc(p, bytes, flags) }),
        Runtime::Hip => {
            let r = unsafe { hs::cuMemHostAlloc(p, bytes, flags) };
            if r == CUresult::CUDA_SUCCESS {
                HIP_HOST.insert(unsafe { *p });
            }
            r
        }
    }
}

pub unsafe fn cuMemFreeHost(p: *mut c_void) -> CUresult {
    if HIP_HOST.remove(p) {
        unsafe { hs::cuMemFreeHost(p) }
    } else {
        res(unsafe { cu::cuMemFreeHost(p) })
    }
}

pub unsafe fn cuMemHostRegister_v2(p: *mut c_void, bytes: usize, flags: c_uint) -> CUresult {
    match bound() {
        Runtime::Cuda => res(unsafe { cu::cuMemHostRegister_v2(p, bytes, flags) }),
        Runtime::Hip => {
            let r = unsafe { hs::cuMemHostRegister_v2(p, bytes, flags) };
            if r == CUresult::CUDA_SUCCESS {
                HIP_HOST.insert(p);
            }
            r
        }
    }
}

pub unsafe fn cuMemHostUnregister(p: *mut c_void) -> CUresult {
    if HIP_HOST.remove(p) {
        unsafe { hs::cuMemHostUnregister(p) }
    } else {
        res(unsafe { cu::cuMemHostUnregister(p) })
    }
}

pub unsafe fn cuMemHostGetDevicePointer_v2(
    d: *mut CUdeviceptr,
    host: *mut c_void,
    flags: c_uint,
) -> CUresult {
    if HIP_HOST.contains(host) {
        unsafe { hs::cuMemHostGetDevicePointer_v2(d, host, flags) }
    } else {
        res(unsafe { cu::cuMemHostGetDevicePointer_v2(d, host, flags) })
    }
}

pub unsafe fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, bytes: usize) -> CUresult {
    on_bound!(cuda => unsafe { cu::cuMemcpyHtoD_v2(dst, src, bytes) },
        hip => unsafe { hs::cuMemcpyHtoD_v2(dst, src, bytes) })
}

pub unsafe fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> CUresult {
    on_bound!(cuda => unsafe { cu::cuMemcpyDtoH_v2(dst, src, bytes) },
        hip => unsafe { hs::cuMemcpyDtoH_v2(dst, src, bytes) })
}

pub unsafe fn cuMemcpyHtoDAsync_v2(
    dst: CUdeviceptr,
    src: *const c_void,
    bytes: usize,
    stream: CUstream,
) -> CUresult {
    on_stream!(stream, cuda => unsafe { cu::cuMemcpyHtoDAsync_v2(dst, src, bytes, cuh(stream)) },
        hip => unsafe { hs::cuMemcpyHtoDAsync_v2(dst, src, bytes, stream) })
}

pub unsafe fn cuMemcpyDtoHAsync_v2(
    dst: *mut c_void,
    src: CUdeviceptr,
    bytes: usize,
    stream: CUstream,
) -> CUresult {
    on_stream!(stream, cuda => unsafe { cu::cuMemcpyDtoHAsync_v2(dst, src, bytes, cuh(stream)) },
        hip => unsafe { hs::cuMemcpyDtoHAsync_v2(dst, src, bytes, stream) })
}

pub unsafe fn cuMemcpyDtoDAsync_v2(
    dst: CUdeviceptr,
    src: CUdeviceptr,
    bytes: usize,
    stream: CUstream,
) -> CUresult {
    on_stream!(stream, cuda => unsafe { cu::cuMemcpyDtoDAsync_v2(dst, src, bytes, cuh(stream)) },
        hip => unsafe { hs::cuMemcpyDtoDAsync_v2(dst, src, bytes, stream) })
}

pub unsafe fn cuMemcpy2DAsync_v2(desc: *const CUDA_MEMCPY2D, stream: CUstream) -> CUresult {
    match stream_runtime(stream) {
        Runtime::Hip => unsafe { hs::cuMemcpy2DAsync_v2(desc, stream) },
        Runtime::Cuda => {
            let d = unsafe { &*desc };
            // SAFETY: both memory types carry CUDA's own numbers.
            let c = cu::CUDA_MEMCPY2D {
                srcXInBytes: d.srcXInBytes,
                srcY: d.srcY,
                srcMemoryType: unsafe { same(d.srcMemoryType) },
                srcHost: d.srcHost,
                srcDevice: d.srcDevice,
                srcArray: cuh(d.srcArray),
                srcPitch: d.srcPitch,
                dstXInBytes: d.dstXInBytes,
                dstY: d.dstY,
                dstMemoryType: unsafe { same(d.dstMemoryType) },
                dstHost: d.dstHost,
                dstDevice: d.dstDevice,
                dstArray: cuh(d.dstArray),
                dstPitch: d.dstPitch,
                WidthInBytes: d.WidthInBytes,
                Height: d.Height,
            };
            res(unsafe { cu::cuMemcpy2DAsync_v2(&c, cuh(stream)) })
        }
    }
}

pub unsafe fn cuMemsetD8_v2(dst: CUdeviceptr, value: u8, n: usize) -> CUresult {
    on_bound!(cuda => unsafe { cu::cuMemsetD8_v2(dst, value, n) },
        hip => unsafe { hs::cuMemsetD8_v2(dst, value, n) })
}

pub unsafe fn cuMemsetD8Async(dst: CUdeviceptr, value: u8, n: usize, stream: CUstream) -> CUresult {
    on_stream!(stream, cuda => unsafe { cu::cuMemsetD8Async(dst, value, n, cuh(stream)) },
        hip => unsafe { hs::cuMemsetD8Async(dst, value, n, stream) })
}

pub unsafe fn cuStreamQuery(stream: CUstream) -> CUresult {
    on_stream!(stream, cuda => unsafe { cu::cuStreamQuery(cuh(stream)) },
        hip => unsafe { hs::cuStreamQuery(stream) })
}

pub unsafe fn cuStreamIsCapturing(
    stream: CUstream,
    status: *mut CUstreamCaptureStatus,
) -> CUresult {
    match stream_runtime(stream) {
        Runtime::Hip => unsafe { hs::cuStreamIsCapturing(stream, status) },
        Runtime::Cuda => {
            let mut s = cu::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
            let r = res(unsafe { cu::cuStreamIsCapturing(cuh(stream), &mut s) });
            if r == CUresult::CUDA_SUCCESS {
                unsafe {
                    *status = match s {
                        cu::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE => {
                            CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
                        }
                        cu::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE => {
                            CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE
                        }
                        _ => CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED,
                    }
                };
            }
            r
        }
    }
}

pub unsafe fn cuStreamBeginCapture_v2(stream: CUstream, mode: CUstreamCaptureMode) -> CUresult {
    on_stream!(stream, cuda => unsafe { cu::cuStreamBeginCapture_v2(cuh(stream), same(mode)) },
        hip => unsafe { hs::cuStreamBeginCapture_v2(stream, mode) })
}

pub unsafe fn cuStreamEndCapture(stream: CUstream, graph: *mut CUgraph) -> CUresult {
    on_stream!(stream, cuda => unsafe { cu::cuStreamEndCapture(cuh(stream), graph as *mut cu::CUgraph) },
        hip => unsafe { hs::cuStreamEndCapture(stream, graph) })
}

pub unsafe fn cuEventRecordWithFlags(event: CUevent, stream: CUstream, flags: c_uint) -> CUresult {
    on_stream!(stream, cuda => unsafe { cu::cuEventRecordWithFlags(cuh(event), cuh(stream), flags) },
        hip => unsafe { hs::cuEventRecordWithFlags(event, stream, flags) })
}

pub unsafe fn cuGraphGetNodes(graph: CUgraph, nodes: *mut CUgraphNode, n: *mut usize) -> CUresult {
    on_bound!(cuda => unsafe { cu::cuGraphGetNodes(cuh(graph), nodes as *mut cu::CUgraphNode, n) },
        hip => unsafe { hs::cuGraphGetNodes(graph, nodes, n) })
}

pub unsafe fn cuGraphNodeGetType(node: CUgraphNode, ty: *mut CUgraphNodeType) -> CUresult {
    match bound() {
        Runtime::Hip => unsafe { hs::cuGraphNodeGetType(node, ty) },
        Runtime::Cuda => {
            let mut t = cu::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
            let r = res(unsafe { cu::cuGraphNodeGetType(cuh(node), &mut t) });
            if r == CUresult::CUDA_SUCCESS {
                unsafe { *ty = CUgraphNodeType::from_hip(t as u32 as c_int) };
            }
            r
        }
    }
}

pub unsafe fn cuGraphNodeSetEnabled(exec: CUgraphExec, node: CUgraphNode, on: c_uint) -> CUresult {
    on_bound!(cuda => unsafe { cu::cuGraphNodeSetEnabled(cuh(exec), cuh(node), on) },
        hip => unsafe { hs::cuGraphNodeSetEnabled(exec, node, on) })
}

pub unsafe fn cuGraphKernelNodeGetParams_v2(
    node: CUgraphNode,
    params: *mut CUDA_KERNEL_NODE_PARAMS,
) -> CUresult {
    match bound() {
        Runtime::Hip => unsafe { hs::cuGraphKernelNodeGetParams_v2(node, params) },
        Runtime::Cuda => {
            // SAFETY: an all-zero parameter block is a valid one for the driver to fill.
            let mut p: cu::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
            let r = res(unsafe { cu::cuGraphKernelNodeGetParams_v2(cuh(node), &mut p) });
            if r == CUresult::CUDA_SUCCESS {
                unsafe {
                    *params = CUDA_KERNEL_NODE_PARAMS {
                        func: cuh(p.func),
                        gridDimX: p.gridDimX,
                        gridDimY: p.gridDimY,
                        gridDimZ: p.gridDimZ,
                        blockDimX: p.blockDimX,
                        blockDimY: p.blockDimY,
                        blockDimZ: p.blockDimZ,
                        sharedMemBytes: p.sharedMemBytes,
                        kernelParams: p.kernelParams,
                        extra: p.extra,
                        kern: cuh(p.kern),
                        ctx: cuh(p.ctx),
                    }
                };
            }
            r
        }
    }
}

pub unsafe fn cuFuncGetName(name: *mut *const c_char, func: CUfunction) -> CUresult {
    on_bound!(cuda => unsafe { cu::cuFuncGetName(name, cuh(func)) },
        hip => unsafe { hs::cuFuncGetName(name, func) })
}

pub unsafe fn cuGraphExecUpdate_v2(
    exec: CUgraphExec,
    graph: CUgraph,
    info: *mut CUgraphExecUpdateResultInfo,
) -> CUresult {
    match bound() {
        Runtime::Hip => unsafe { hs::cuGraphExecUpdate_v2(exec, graph, info) },
        Runtime::Cuda => {
            // SAFETY: an all-zero report is a valid one for the driver to fill.
            let mut i: cu::CUgraphExecUpdateResultInfo = unsafe { std::mem::zeroed() };
            let r = res(unsafe { cu::cuGraphExecUpdate_v2(cuh(exec), cuh(graph), &mut i) });
            if !info.is_null() {
                let result =
                    if i.result == cu::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_SUCCESS {
                        CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_SUCCESS
                    } else {
                        CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_ERROR
                    };
                unsafe {
                    *info = CUgraphExecUpdateResultInfo {
                        result,
                        errorNode: cuh(i.errorNode),
                        errorFromNode: cuh(i.errorFromNode),
                    }
                };
            }
            r
        }
    }
}

pub unsafe fn cuGraphInstantiateWithFlags(
    exec: *mut CUgraphExec,
    graph: CUgraph,
    flags: u64,
) -> CUresult {
    on_bound!(cuda => unsafe { cu::cuGraphInstantiateWithFlags(exec as *mut cu::CUgraphExec, cuh(graph), flags) },
        hip => unsafe { hs::cuGraphInstantiateWithFlags(exec, graph, flags) })
}

pub unsafe fn cuGraphExecDestroy(exec: CUgraphExec) -> CUresult {
    on_bound!(cuda => unsafe { cu::cuGraphExecDestroy(cuh(exec)) },
        hip => unsafe { hs::cuGraphExecDestroy(exec) })
}

pub unsafe fn cuGraphDestroy(graph: CUgraph) -> CUresult {
    on_bound!(cuda => unsafe { cu::cuGraphDestroy(cuh(graph)) },
        hip => unsafe { hs::cuGraphDestroy(graph) })
}

pub unsafe fn cuGraphLaunch(exec: CUgraphExec, stream: CUstream) -> CUresult {
    on_stream!(stream, cuda => unsafe { cu::cuGraphLaunch(cuh(exec), cuh(stream)) },
        hip => unsafe { hs::cuGraphLaunch(exec, stream) })
}

pub unsafe fn cuGraphUpload(exec: CUgraphExec, stream: CUstream) -> CUresult {
    on_stream!(stream, cuda => unsafe { cu::cuGraphUpload(cuh(exec), cuh(stream)) },
        hip => unsafe { hs::cuGraphUpload(exec, stream) })
}

pub unsafe fn cuDeviceGetDefaultMemPool(pool: *mut CUmemoryPool, dev: CUdevice) -> CUresult {
    on_device!(dev, |d| cuda => unsafe { cu::cuDeviceGetDefaultMemPool(pool as *mut cu::CUmemoryPool, d) },
    hip => {
        let r = unsafe { hs::cuDeviceGetDefaultMemPool(pool, d) };
        if r == CUresult::CUDA_SUCCESS {
            HIP_POOLS.insert(unsafe { *pool });
        }
        r
    })
}

pub unsafe fn cuMemPoolGetAttribute(
    pool: CUmemoryPool,
    attr: CUmemPool_attribute,
    value: *mut c_void,
) -> CUresult {
    on_pool!(pool, cuda => unsafe { cu::cuMemPoolGetAttribute(cuh(pool), same(attr), value) },
        hip => unsafe { hs::cuMemPoolGetAttribute(pool, attr, value) })
}

pub unsafe fn cuMemPoolSetAttribute(
    pool: CUmemoryPool,
    attr: CUmemPool_attribute,
    value: *mut c_void,
) -> CUresult {
    on_pool!(pool, cuda => unsafe { cu::cuMemPoolSetAttribute(cuh(pool), same(attr), value) },
        hip => unsafe { hs::cuMemPoolSetAttribute(pool, attr, value) })
}

pub unsafe fn cuMemPoolTrimTo(pool: CUmemoryPool, keep: usize) -> CUresult {
    on_pool!(pool, cuda => unsafe { cu::cuMemPoolTrimTo(cuh(pool), keep) },
        hip => unsafe { hs::cuMemPoolTrimTo(pool, keep) })
}

/// A memory location with its card numbered in its own runtime, and that runtime; `None` for a
/// card no runtime lists.
fn locate(l: CUmemLocation) -> Option<(Option<Runtime>, CUmemLocation)> {
    if l.type_ != CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE {
        return Some((None, l));
    }
    // SAFETY: the union's only member is the identifier.
    let (rt, local) = device(unsafe { l.__bindgen_anon_1.id })?;
    Some((
        Some(rt),
        CUmemLocation {
            type_: l.type_,
            __bindgen_anon_1: CUmemLocation_st__bindgen_ty_1 { id: local },
        },
    ))
}

fn cuda_location(l: CUmemLocation) -> cu::CUmemLocation {
    // SAFETY: an all-zero location is a valid one; both fields are set below.
    let mut c: cu::CUmemLocation = unsafe { std::mem::zeroed() };
    c.type_ = unsafe { same(l.type_) };
    // SAFETY: the union's only member is the identifier.
    c.__bindgen_anon_1.id = unsafe { l.__bindgen_anon_1.id };
    c
}

pub unsafe fn cuMemPoolCreate(pool: *mut CUmemoryPool, props: *const CUmemPoolProps) -> CUresult {
    let mut p = unsafe { *props };
    let Some((rt, location)) = locate(p.location) else {
        return CUresult::CUDA_ERROR_INVALID_DEVICE;
    };
    p.location = location;
    match rt.unwrap_or_else(bound) {
        Runtime::Hip => {
            let r = unsafe { hs::cuMemPoolCreate(pool, &p) };
            if r == CUresult::CUDA_SUCCESS {
                HIP_POOLS.insert(unsafe { *pool });
            }
            r
        }
        Runtime::Cuda => {
            // SAFETY: all-zero properties are valid ones; the fields in use are set below.
            let mut c: cu::CUmemPoolProps = unsafe { std::mem::zeroed() };
            c.allocType = unsafe { same(p.allocType) };
            c.handleTypes = unsafe { same(p.handleTypes) };
            c.location = cuda_location(p.location);
            c.win32SecurityAttributes = p.win32SecurityAttributes;
            c.maxSize = p.maxSize;
            c.usage = p.usage;
            res(unsafe { cu::cuMemPoolCreate(pool as *mut cu::CUmemoryPool, &c) })
        }
    }
}

pub unsafe fn cuMemAllocFromPoolAsync(
    d: *mut CUdeviceptr,
    bytes: usize,
    pool: CUmemoryPool,
    stream: CUstream,
) -> CUresult {
    on_stream!(stream, cuda => unsafe { cu::cuMemAllocFromPoolAsync(d, bytes, cuh(pool), cuh(stream)) },
        hip => unsafe { hs::cuMemAllocFromPoolAsync(d, bytes, pool, stream) })
}

pub unsafe fn cuMemGetAllocationGranularity(
    granularity: *mut usize,
    prop: *const CUmemAllocationProp,
    option: CUmemAllocationGranularity_flags,
) -> CUresult {
    let mut p = unsafe { *prop };
    let Some((rt, location)) = locate(p.location) else {
        return CUresult::CUDA_ERROR_INVALID_DEVICE;
    };
    p.location = location;
    match rt.unwrap_or_else(bound) {
        Runtime::Hip => unsafe { hs::cuMemGetAllocationGranularity(granularity, &p, option) },
        Runtime::Cuda => {
            // SAFETY: all-zero properties are valid ones; the fields in use are set below.
            let mut c: cu::CUmemAllocationProp = unsafe { std::mem::zeroed() };
            c.type_ = unsafe { same(p.type_) };
            c.requestedHandleTypes = unsafe { same(p.requestedHandleTypes) };
            c.location = cuda_location(p.location);
            c.win32HandleMetaData = p.win32HandleMetaData;
            c.allocFlags.compressionType = p.allocFlags.compressionType;
            c.allocFlags.gpuDirectRDMACapable = p.allocFlags.gpuDirectRDMACapable;
            c.allocFlags.usage = p.allocFlags.usage;
            res(unsafe { cu::cuMemGetAllocationGranularity(granularity, &c, same(option)) })
        }
    }
}

pub unsafe fn cuDeviceGetGraphMemAttribute(
    dev: CUdevice,
    attr: CUgraphMem_attribute,
    value: *mut c_void,
) -> CUresult {
    on_device!(dev, |d| cuda => unsafe { cu::cuDeviceGetGraphMemAttribute(d, same(attr), value) },
        hip => unsafe { hs::cuDeviceGetGraphMemAttribute(d, attr, value) })
}

pub unsafe fn cuDeviceGraphMemTrim(dev: CUdevice) -> CUresult {
    on_device!(dev, |d| cuda => unsafe { cu::cuDeviceGraphMemTrim(d) },
        hip => unsafe { hs::cuDeviceGraphMemTrim(d) })
}
