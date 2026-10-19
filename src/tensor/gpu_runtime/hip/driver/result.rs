//! Driver calls that return a `Result`, under the names the GPU paths use.

use std::ffi::{c_int, c_uint, c_void, CStr};
use std::mem::MaybeUninit;

use super::sys::{self, hip, CUresult};

/// A failed driver call: its status.
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct DriverError(pub sys::CUresult);

impl sys::CUresult {
    /// `Ok` for success, the status as an error otherwise.
    #[inline]
    pub fn result(self) -> Result<(), DriverError> {
        match self {
            sys::CUresult::CUDA_SUCCESS => Ok(()),
            _ => Err(DriverError(self)),
        }
    }
}

impl DriverError {
    /// The runtime's description of the status.
    pub fn error_string(&self) -> Result<&CStr, DriverError> {
        let api = sys::Hip::get().ok_or(DriverError(CUresult::CUDA_ERROR_NOT_INITIALIZED))?;
        let f = api
            .hipGetErrorString
            .ok_or(DriverError(CUresult::CUDA_ERROR_NOT_SUPPORTED))?;
        let s = unsafe { f(self.0 as u32 as c_int) };
        if s.is_null() {
            return Err(DriverError(CUresult::CUDA_ERROR_UNKNOWN));
        }
        // SAFETY: the runtime returns a static, terminated string.
        Ok(unsafe { CStr::from_ptr(s) })
    }

    /// The runtime's name for the status.
    pub fn error_name(&self) -> Result<&CStr, DriverError> {
        let api = sys::Hip::get().ok_or(DriverError(CUresult::CUDA_ERROR_NOT_INITIALIZED))?;
        let f = api
            .hipGetErrorName
            .ok_or(DriverError(CUresult::CUDA_ERROR_NOT_SUPPORTED))?;
        let s = unsafe { f(self.0 as u32 as c_int) };
        if s.is_null() {
            return Err(DriverError(CUresult::CUDA_ERROR_UNKNOWN));
        }
        // SAFETY: the runtime returns a static, terminated string.
        Ok(unsafe { CStr::from_ptr(s) })
    }
}

impl std::fmt::Debug for DriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut t = f.debug_tuple("DriverError");
        t.field(&self.0);
        #[cfg(feature = "cuda")]
        if let Some(s) = crate::tensor::gpu_runtime::dual::cuda_error_string(self.0) {
            t.field(&s);
            return t.finish();
        }
        match self.error_string() {
            Ok(s) => t.field(&s),
            Err(_) => t.field(&"<no description>"),
        };
        t.finish()
    }
}

impl std::fmt::Display for DriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for DriverError {}

/// Initializes the runtime.
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

    /// The number of devices.
    pub fn get_count() -> Result<c_int, DriverError> {
        let mut count = MaybeUninit::uninit();
        unsafe {
            sys::cuDeviceGetCount(count.as_mut_ptr()).result()?;
            Ok(count.assume_init())
        }
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
        let mut buf = [0 as std::ffi::c_char; 256];
        unsafe { sys::cuDeviceGetName(buf.as_mut_ptr(), buf.len() as c_int, dev) }.result()?;
        // SAFETY: the runtime terminates the name within the buffer it is given.
        let name = unsafe { CStr::from_ptr(buf.as_ptr()) };
        Ok(name.to_string_lossy().into_owned())
    }

    /// The device's architecture name as the runtime reports it, the target its code objects
    /// are compiled for.
    pub fn gcn_arch_name(dev: sys::CUdevice) -> Result<String, DriverError> {
        let mut prop = MaybeUninit::<sys::HipDeviceProp>::zeroed();
        unsafe { hip!(hipGetDevicePropertiesR0600(prop.as_mut_ptr(), dev)) }.result()?;
        // SAFETY: the runtime filled the structure; the name is terminated within its array.
        let prop = unsafe { prop.assume_init() };
        let name = unsafe { CStr::from_ptr(prop.gcn_arch_name.as_ptr()) };
        Ok(name.to_string_lossy().into_owned())
    }
}

/// Allocates device memory outside any stream.
///
/// # Safety
/// The memory is uninitialized.
pub unsafe fn malloc_sync(num_bytes: usize) -> Result<sys::CUdeviceptr, DriverError> {
    let mut p: *mut c_void = std::ptr::null_mut();
    unsafe { hip!(hipMalloc(&mut p, num_bytes)) }.result()?;
    Ok(p as usize as sys::CUdeviceptr)
}

/// Allocates device memory in stream order.
///
/// # Safety
/// The memory is uninitialized; `stream` must be valid.
pub unsafe fn malloc_async(
    stream: sys::CUstream,
    num_bytes: usize,
) -> Result<sys::CUdeviceptr, DriverError> {
    let mut p: *mut c_void = std::ptr::null_mut();
    unsafe { hip!(hipMallocAsync(&mut p, num_bytes, stream)) }.result()?;
    Ok(p as usize as sys::CUdeviceptr)
}

/// Frees device memory outside any stream.
///
/// # Safety
/// `dptr` must come from an allocation and not be freed twice.
pub unsafe fn free_sync(dptr: sys::CUdeviceptr) -> Result<(), DriverError> {
    unsafe { hip!(hipFree(dptr as usize as *mut c_void)) }.result()
}

/// Frees device memory in stream order.
///
/// # Safety
/// `dptr` must come from an allocation and not be freed twice.
pub unsafe fn free_async(dptr: sys::CUdeviceptr, stream: sys::CUstream) -> Result<(), DriverError> {
    unsafe { hip!(hipFreeAsync(dptr as usize as *mut c_void, stream)) }.result()
}

/// Allocates page-locked host memory.
///
/// # Safety
/// The memory is uninitialized.
pub unsafe fn malloc_host(num_bytes: usize, flags: c_uint) -> Result<*mut c_void, DriverError> {
    let mut p: *mut c_void = std::ptr::null_mut();
    unsafe { sys::cuMemHostAlloc(&mut p, num_bytes, flags) }.result()?;
    Ok(p)
}

/// Frees page-locked host memory.
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

/// The free and total memory of the current device.
pub fn mem_get_info() -> Result<(usize, usize), DriverError> {
    let (mut free, mut total) = (0usize, 0usize);
    unsafe { hip!(hipMemGetInfo(&mut free, &mut total)) }.result()?;
    Ok((free, total))
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
