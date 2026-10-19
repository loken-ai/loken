//! Driver calls that return a `Result`, under the names the GPU paths use, each one routed like
//! the raw call it wraps.

use std::ffi::{c_int, c_uint, c_void};
use std::mem::MaybeUninit;

use super::super::{bound, topology, Runtime};
use super::sys::{self, CUresult};
use crate::tensor::gpu_runtime::hip::driver::result as hr;
use cudarc::driver::result as cr;

pub use hr::DriverError;

/// A `cudarc` error as the error the GPU paths match on.
#[inline]
pub(crate) fn err(e: cudarc::driver::DriverError) -> DriverError {
    DriverError(sys::res(e.0))
}

/// Initializes every runtime that lists cards.
pub fn init() -> Result<(), DriverError> {
    unsafe { sys::cuInit(0) }.result()
}

pub mod device {
    use super::*;

    /// The device of an ordinal.
    pub fn get(ordinal: c_int) -> Result<sys::CUdevice, DriverError> {
        let mut dev = MaybeUninit::uninit();
        unsafe {
            sys::cuDeviceGet(dev.as_mut_ptr(), ordinal).result()?;
            Ok(dev.assume_init())
        }
    }

    /// The number of devices of both runtimes.
    pub fn get_count() -> Result<c_int, DriverError> {
        Ok(topology().count() as c_int)
    }

    /// The device's memory in bytes.
    ///
    /// # Safety
    /// `dev` must come from [get].
    pub unsafe fn total_mem(dev: sys::CUdevice) -> Result<usize, DriverError> {
        let mut bytes = MaybeUninit::uninit();
        unsafe {
            sys::cuDeviceTotalMem_v2(bytes.as_mut_ptr(), dev).result()?;
            Ok(bytes.assume_init())
        }
    }

    /// One attribute of the device.
    ///
    /// # Safety
    /// `dev` must come from [get].
    pub unsafe fn get_attribute(
        dev: sys::CUdevice,
        attrib: sys::CUdevice_attribute,
    ) -> Result<i32, DriverError> {
        let mut value = MaybeUninit::uninit();
        unsafe {
            sys::cuDeviceGetAttribute(value.as_mut_ptr(), attrib, dev).result()?;
            Ok(value.assume_init())
        }
    }

    /// The device's name.
    pub fn get_name(dev: sys::CUdevice) -> Result<String, DriverError> {
        match topology().locate(dev as usize) {
            Some((Runtime::Cuda, d)) => cr::device::get_name(d as c_int).map_err(err),
            Some((Runtime::Hip, d)) => hr::device::get_name(d as c_int),
            None => Err(DriverError(CUresult::CUDA_ERROR_INVALID_DEVICE)),
        }
    }

    /// An AMD card's architecture name as the runtime reports it; not supported for an NVIDIA
    /// card.
    pub fn gcn_arch_name(dev: sys::CUdevice) -> Result<String, DriverError> {
        match topology().locate(dev as usize) {
            Some((Runtime::Hip, d)) => hr::device::gcn_arch_name(d as c_int),
            Some((Runtime::Cuda, _)) => Err(DriverError(CUresult::CUDA_ERROR_NOT_SUPPORTED)),
            None => Err(DriverError(CUresult::CUDA_ERROR_INVALID_DEVICE)),
        }
    }
}

/// Allocates device memory outside any stream, on the bound runtime's current device.
///
/// # Safety
/// The memory is uninitialized.
pub unsafe fn malloc_sync(num_bytes: usize) -> Result<sys::CUdeviceptr, DriverError> {
    match bound() {
        Runtime::Cuda => unsafe { cr::malloc_sync(num_bytes) }.map_err(err),
        Runtime::Hip => unsafe { hr::malloc_sync(num_bytes) },
    }
}

/// Allocates device memory in stream order.
///
/// # Safety
/// The memory is uninitialized; `stream` must be valid.
pub unsafe fn malloc_async(
    stream: sys::CUstream,
    num_bytes: usize,
) -> Result<sys::CUdeviceptr, DriverError> {
    match super::super::stream_runtime(stream) {
        Runtime::Cuda => unsafe { cr::malloc_async(sys::cuh(stream), num_bytes) }.map_err(err),
        Runtime::Hip => unsafe { hr::malloc_async(stream, num_bytes) },
    }
}

/// Frees device memory outside any stream, through the bound runtime.
///
/// # Safety
/// `dptr` must come from an allocation and not be freed twice.
pub unsafe fn free_sync(dptr: sys::CUdeviceptr) -> Result<(), DriverError> {
    match bound() {
        Runtime::Cuda => unsafe { cr::free_sync(dptr) }.map_err(err),
        Runtime::Hip => unsafe { hr::free_sync(dptr) },
    }
}

/// Frees device memory in stream order.
///
/// # Safety
/// `dptr` must come from an allocation and not be freed twice.
pub unsafe fn free_async(dptr: sys::CUdeviceptr, stream: sys::CUstream) -> Result<(), DriverError> {
    match super::super::stream_runtime(stream) {
        Runtime::Cuda => unsafe { cr::free_async(dptr, sys::cuh(stream)) }.map_err(err),
        Runtime::Hip => unsafe { hr::free_async(dptr, stream) },
    }
}

/// Allocates page-locked host memory through the bound runtime.
///
/// # Safety
/// The memory is uninitialized.
pub unsafe fn malloc_host(num_bytes: usize, flags: c_uint) -> Result<*mut c_void, DriverError> {
    let mut p: *mut c_void = std::ptr::null_mut();
    unsafe { sys::cuMemHostAlloc(&mut p, num_bytes, flags) }.result()?;
    Ok(p)
}

/// Frees page-locked host memory through the runtime that allocated it.
///
/// # Safety
/// `host_ptr` must come from [malloc_host] and not be freed twice.
pub unsafe fn free_host(host_ptr: *mut c_void) -> Result<(), DriverError> {
    unsafe { sys::cuMemFreeHost(host_ptr) }.result()
}

/// Copies device memory to the host in stream order.
///
/// # Safety
/// `src` must hold at least `dst`'s bytes and `dst` must outlive the copy.
pub unsafe fn memcpy_dtoh_async<T>(
    dst: &mut [T],
    src: sys::CUdeviceptr,
    stream: sys::CUstream,
) -> Result<(), DriverError> {
    unsafe {
        sys::cuMemcpyDtoHAsync_v2(
            dst.as_mut_ptr() as *mut c_void,
            src,
            std::mem::size_of_val(dst),
            stream,
        )
    }
    .result()
}

/// The free and total memory of the bound runtime's current device.
pub fn mem_get_info() -> Result<(usize, usize), DriverError> {
    match bound() {
        // PLACEMENT-EXEMPT: the runtime's reading itself; its callers read through `place/`.
        Runtime::Cuda => cr::mem_get_info().map_err(err),
        // PLACEMENT-EXEMPT: the runtime's reading itself; its callers read through `place/`.
        Runtime::Hip => hr::mem_get_info(),
    }
}

pub mod stream {
    use super::*;

    /// Starts capturing the stream's work into a graph.
    ///
    /// # Safety
    /// `stream` must be valid.
    pub unsafe fn begin_capture(
        stream: sys::CUstream,
        mode: sys::CUstreamCaptureMode,
    ) -> Result<(), DriverError> {
        unsafe { sys::cuStreamBeginCapture_v2(stream, mode) }.result()
    }

    /// Ends the capture and returns the captured graph.
    ///
    /// # Safety
    /// `stream` must be valid.
    pub unsafe fn end_capture(stream: sys::CUstream) -> Result<sys::CUgraph, DriverError> {
        let mut graph = std::ptr::null_mut();
        unsafe { sys::cuStreamEndCapture(stream, &mut graph) }.result()?;
        Ok(graph)
    }

    /// The stream's capture state.
    ///
    /// # Safety
    /// `stream` must be valid.
    pub unsafe fn is_capturing(
        stream: sys::CUstream,
    ) -> Result<sys::CUstreamCaptureStatus, DriverError> {
        let mut status = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        unsafe { sys::cuStreamIsCapturing(stream, &mut status) }.result()?;
        Ok(status)
    }
}

pub mod graph {
    use super::*;

    /// Instantiates a graph into an executable.
    ///
    /// # Safety
    /// `graph` must be valid.
    pub unsafe fn instantiate(
        graph: sys::CUgraph,
        flags: sys::CUgraphInstantiate_flags,
    ) -> Result<sys::CUgraphExec, DriverError> {
        let mut exec = std::ptr::null_mut();
        unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, graph, u64::from(flags)) }.result()?;
        Ok(exec)
    }

    /// Destroys an executable graph.
    ///
    /// # Safety
    /// `graph_exec` must be valid.
    pub unsafe fn exec_destroy(graph_exec: sys::CUgraphExec) -> Result<(), DriverError> {
        unsafe { sys::cuGraphExecDestroy(graph_exec) }.result()
    }

    /// Destroys a graph.
    ///
    /// # Safety
    /// `graph` must be valid.
    pub unsafe fn destroy(graph: sys::CUgraph) -> Result<(), DriverError> {
        unsafe { sys::cuGraphDestroy(graph) }.result()
    }

    /// Launches an executable graph on a stream.
    ///
    /// # Safety
    /// `graph_exec` and `stream` must be valid.
    pub unsafe fn launch(
        graph_exec: sys::CUgraphExec,
        stream: sys::CUstream,
    ) -> Result<(), DriverError> {
        unsafe { sys::cuGraphLaunch(graph_exec, stream) }.result()
    }

    /// Uploads an executable graph's resources ahead of its first launch.
    ///
    /// # Safety
    /// `graph_exec` and `stream` must be valid.
    pub unsafe fn upload(
        graph_exec: sys::CUgraphExec,
        stream: sys::CUstream,
    ) -> Result<(), DriverError> {
        unsafe { sys::cuGraphUpload(graph_exec, stream) }.result()
    }
}
