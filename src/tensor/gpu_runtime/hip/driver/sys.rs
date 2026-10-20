//! The raw driver items the GPU paths name, under their CUDA names, each one calling its HIP
//! counterpart. Types are HIP's handles; enums keep the CUDA variant names and are translated
//! to HIP's values at the call. An item HIP lacks returns `CUDA_ERROR_NOT_SUPPORTED`.

#![allow(non_camel_case_types, non_snake_case, clippy::missing_safety_doc)]

use std::ffi::{c_char, c_int, c_uint, c_void};

crate::tensor::gpu_runtime::hip::dynamic_api! {
    /// The entry points of the HIP runtime library.
    pub(crate) struct Hip from "amdhip64" {
        fn hipInit(flags: c_uint) -> c_int;
        fn hipGetErrorString(e: c_int) -> *const c_char;
        fn hipGetErrorName(e: c_int) -> *const c_char;
        fn hipGetDeviceCount(n: *mut c_int) -> c_int;
        fn hipDeviceGet(d: *mut c_int, ordinal: c_int) -> c_int;
        fn hipDeviceGetAttribute(v: *mut c_int, a: c_int, d: c_int) -> c_int;
        fn hipDeviceGetName(name: *mut c_char, len: c_int, d: c_int) -> c_int;
        fn hipDeviceTotalMem(bytes: *mut usize, d: c_int) -> c_int;
        fn hipGetDevicePropertiesR0600(p: *mut HipDeviceProp, d: c_int) -> c_int;
        fn hipDeviceGetPCIBusId(id: *mut c_char, len: c_int, d: c_int) -> c_int;
        fn hipDeviceCanAccessPeer(can: *mut c_int, d: c_int, peer: c_int) -> c_int;
        fn hipDevicePrimaryCtxRetain(ctx: *mut CUcontext, d: c_int) -> c_int;
        fn hipDevicePrimaryCtxRelease(d: c_int) -> c_int;
        fn hipDevicePrimaryCtxReset(d: c_int) -> c_int;
        fn hipDevicePrimaryCtxGetState(d: c_int, flags: *mut c_uint, active: *mut c_int) -> c_int;
        fn hipCtxPushCurrent(ctx: CUcontext) -> c_int;
        fn hipCtxPopCurrent(ctx: *mut CUcontext) -> c_int;
        fn hipCtxSynchronize() -> c_int;
        fn hipSetDevice(d: c_int) -> c_int;
        fn hipSetDeviceFlags(flags: c_uint) -> c_int;
        fn hipGetDevice(d: *mut c_int) -> c_int;
        fn hipDeviceSynchronize() -> c_int;
        fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;
        fn hipMalloc(p: *mut *mut c_void, n: usize) -> c_int;
        fn hipFree(p: *mut c_void) -> c_int;
        fn hipMallocAsync(p: *mut *mut c_void, n: usize, s: CUstream) -> c_int;
        fn hipFreeAsync(p: *mut c_void, s: CUstream) -> c_int;
        fn hipHostMalloc(p: *mut *mut c_void, n: usize, flags: c_uint) -> c_int;
        fn hipHostFree(p: *mut c_void) -> c_int;
        fn hipHostRegister(p: *mut c_void, n: usize, flags: c_uint) -> c_int;
        fn hipHostUnregister(p: *mut c_void) -> c_int;
        fn hipHostGetDevicePointer(d: *mut *mut c_void, h: *mut c_void, flags: c_uint) -> c_int;
        fn hipMemcpyHtoD(dst: *mut c_void, src: *const c_void, n: usize) -> c_int;
        fn hipMemcpyDtoH(dst: *mut c_void, src: *mut c_void, n: usize) -> c_int;
        fn hipMemcpyHtoDAsync(dst: *mut c_void, src: *const c_void, n: usize, s: CUstream) -> c_int;
        fn hipMemcpyDtoHAsync(dst: *mut c_void, src: *mut c_void, n: usize, s: CUstream) -> c_int;
        fn hipMemcpyDtoDAsync(dst: *mut c_void, src: *mut c_void, n: usize, s: CUstream) -> c_int;
        fn hipMemcpyParam2DAsync(p: *const HipMemcpy2D, s: CUstream) -> c_int;
        fn hipMemsetD8(dst: *mut c_void, v: u8, n: usize) -> c_int;
        fn hipMemsetD8Async(dst: *mut c_void, v: u8, n: usize, s: CUstream) -> c_int;
        fn hipStreamCreateWithFlags(s: *mut CUstream, flags: c_uint) -> c_int;
        fn hipStreamDestroy(s: CUstream) -> c_int;
        fn hipStreamSynchronize(s: CUstream) -> c_int;
        fn hipStreamQuery(s: CUstream) -> c_int;
        fn hipStreamWaitEvent(s: CUstream, e: CUevent, flags: c_uint) -> c_int;
        fn hipStreamBeginCapture(s: CUstream, mode: c_int) -> c_int;
        fn hipStreamEndCapture(s: CUstream, g: *mut CUgraph) -> c_int;
        fn hipStreamIsCapturing(s: CUstream, status: *mut c_int) -> c_int;
        fn hipStreamGetDevice(s: CUstream, d: *mut c_int) -> c_int;
        fn hipEventCreateWithFlags(e: *mut CUevent, flags: c_uint) -> c_int;
        fn hipEventRecord(e: CUevent, s: CUstream) -> c_int;
        fn hipEventRecordWithFlags(e: CUevent, s: CUstream, flags: c_uint) -> c_int;
        fn hipEventSynchronize(e: CUevent) -> c_int;
        fn hipEventQuery(e: CUevent) -> c_int;
        fn hipEventElapsedTime(ms: *mut f32, start: CUevent, end: CUevent) -> c_int;
        fn hipEventDestroy(e: CUevent) -> c_int;
        fn hipModuleLoadData(m: *mut CUmodule, image: *const c_void) -> c_int;
        fn hipModuleUnload(m: CUmodule) -> c_int;
        fn hipModuleGetFunction(f: *mut CUfunction, m: CUmodule, name: *const c_char) -> c_int;
        fn hipModuleLaunchKernel(
            f: CUfunction,
            gx: c_uint,
            gy: c_uint,
            gz: c_uint,
            bx: c_uint,
            by: c_uint,
            bz: c_uint,
            shared: c_uint,
            s: CUstream,
            params: *mut *mut c_void,
            extra: *mut *mut c_void,
        ) -> c_int;
        fn hipGraphInstantiateWithFlags(e: *mut CUgraphExec, g: CUgraph, flags: u64) -> c_int;
        fn hipGraphExecDestroy(e: CUgraphExec) -> c_int;
        fn hipGraphDestroy(g: CUgraph) -> c_int;
        fn hipGraphLaunch(e: CUgraphExec, s: CUstream) -> c_int;
        fn hipGraphUpload(e: CUgraphExec, s: CUstream) -> c_int;
        fn hipGraphGetNodes(g: CUgraph, nodes: *mut CUgraphNode, n: *mut usize) -> c_int;
        fn hipGraphNodeGetType(node: CUgraphNode, t: *mut c_int) -> c_int;
        fn hipGraphNodeSetEnabled(e: CUgraphExec, node: CUgraphNode, on: c_uint) -> c_int;
        fn hipGraphKernelNodeGetParams(node: CUgraphNode, p: *mut HipKernelNodeParams) -> c_int;
        fn hipGraphExecUpdate(
            e: CUgraphExec,
            g: CUgraph,
            error_node: *mut CUgraphNode,
            result: *mut c_int,
        ) -> c_int;
        fn hipDeviceGetDefaultMemPool(p: *mut CUmemoryPool, d: c_int) -> c_int;
        fn hipMemPoolGetAttribute(p: CUmemoryPool, a: c_int, v: *mut c_void) -> c_int;
        fn hipMemPoolSetAttribute(p: CUmemoryPool, a: c_int, v: *mut c_void) -> c_int;
        fn hipMemPoolTrimTo(p: CUmemoryPool, keep: usize) -> c_int;
        fn hipMemPoolDestroy(p: CUmemoryPool) -> c_int;
        fn hipMemPoolCreate(p: *mut CUmemoryPool, props: *const HipMemPoolProps) -> c_int;
        fn hipMallocFromPoolAsync(
            d: *mut *mut c_void,
            n: usize,
            p: CUmemoryPool,
            s: CUstream,
        ) -> c_int;
        fn hipMemGetAllocationGranularity(
            g: *mut usize,
            prop: *const HipMemAllocationProp,
            option: c_int,
        ) -> c_int;
        fn hipDeviceGetGraphMemAttribute(d: c_int, a: c_int, v: *mut c_void) -> c_int;
        fn hipDeviceGraphMemTrim(d: c_int) -> c_int;
    }
}

/// Calls a HIP runtime entry point and returns its status as a `CUresult`: not initialized when
/// the library does not open, not supported when it lacks the entry point.
macro_rules! hip {
    ($f:ident ( $( $a:expr ),* $(,)? )) => {
        match $crate::tensor::gpu_runtime::hip::driver::sys::Hip::get() {
            None => $crate::tensor::gpu_runtime::hip::driver::sys::CUresult::CUDA_ERROR_NOT_INITIALIZED,
            Some(api) => match api.$f {
                None => $crate::tensor::gpu_runtime::hip::driver::sys::CUresult::CUDA_ERROR_NOT_SUPPORTED,
                Some(f) => $crate::tensor::gpu_runtime::hip::driver::sys::CUresult::from_hip(f($( $a ),*)),
            },
        }
    };
}
pub(crate) use hip;

/// Whether the HIP runtime library opens.
pub unsafe fn is_culib_present() -> bool {
    Hip::present()
}

macro_rules! opaque {
    ($($handle:ident = $st:ident;)*) => {$(
        #[repr(C)]
        #[derive(Debug, Copy, Clone)]
        pub struct $st {
            _unused: [u8; 0],
        }
        pub type $handle = *mut $st;
    )*};
}

opaque! {
    CUcontext = CUctx_st;
    CUstream = CUstream_st;
    CUevent = CUevent_st;
    CUmodule = CUmod_st;
    CUfunction = CUfunc_st;
    CUgraph = CUgraph_st;
    CUgraphExec = CUgraphExec_st;
    CUgraphNode = CUgraphNode_st;
    CUmemoryPool = CUmemPoolHandle_st;
    CUarray = CUarray_st;
    CUkernel = CUkern_st;
}

/// A device address.
pub type CUdeviceptr = u64;
/// A device ordinal.
pub type CUdevice = c_int;

/// A driver status. Each variant's discriminant is the HIP code that maps to it, which is also
/// the CUDA code of the same name.
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_MAP_FAILED = 205,
    CUDA_ERROR_ALREADY_MAPPED = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_ILLEGAL_STATE = 401,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_ILLEGAL_ADDRESS = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
    CUDA_ERROR_CAPTURED_EVENT = 907,
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
    CUDA_ERROR_UNKNOWN = 999,
}
pub use self::cudaError_enum as CUresult;

impl cudaError_enum {
    /// The status a HIP code stands for; a code with no CUDA name is `CUDA_ERROR_UNKNOWN`.
    pub(crate) fn from_hip(code: c_int) -> Self {
        use cudaError_enum::*;
        const ALL: [cudaError_enum; 34] = [
            CUDA_SUCCESS,
            CUDA_ERROR_INVALID_VALUE,
            CUDA_ERROR_OUT_OF_MEMORY,
            CUDA_ERROR_NOT_INITIALIZED,
            CUDA_ERROR_DEINITIALIZED,
            CUDA_ERROR_NO_DEVICE,
            CUDA_ERROR_INVALID_DEVICE,
            CUDA_ERROR_INVALID_IMAGE,
            CUDA_ERROR_INVALID_CONTEXT,
            CUDA_ERROR_MAP_FAILED,
            CUDA_ERROR_ALREADY_MAPPED,
            CUDA_ERROR_NO_BINARY_FOR_GPU,
            CUDA_ERROR_INVALID_SOURCE,
            CUDA_ERROR_FILE_NOT_FOUND,
            CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
            CUDA_ERROR_OPERATING_SYSTEM,
            CUDA_ERROR_INVALID_HANDLE,
            CUDA_ERROR_ILLEGAL_STATE,
            CUDA_ERROR_NOT_FOUND,
            CUDA_ERROR_NOT_READY,
            CUDA_ERROR_ILLEGAL_ADDRESS,
            CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
            CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
            CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
            CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,
            CUDA_ERROR_LAUNCH_FAILED,
            CUDA_ERROR_NOT_SUPPORTED,
            CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
            CUDA_ERROR_STREAM_CAPTURE_INVALIDATED,
            CUDA_ERROR_STREAM_CAPTURE_ISOLATION,
            CUDA_ERROR_STREAM_CAPTURE_IMPLICIT,
            CUDA_ERROR_CAPTURED_EVENT,
            CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD,
            CUDA_ERROR_UNKNOWN,
        ];
        ALL.into_iter()
            .find(|e| *e as u32 as c_int == code)
            .unwrap_or(CUDA_ERROR_UNKNOWN)
    }
}

/// A device attribute, under its CUDA name and number.
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
}
pub use self::CUdevice_attribute_enum as CUdevice_attribute;

impl CUdevice_attribute_enum {
    /// HIP's number for the attribute. `None` for the compute capability: HIP reports its own
    /// architecture numbers there, which do not order like CUDA's, so it is not offered under the
    /// CUDA name.
    fn to_hip(self) -> Option<c_int> {
        use CUdevice_attribute_enum::*;
        Some(match self {
            CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK => 56,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK => 74,
            CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY => 83,
            CU_DEVICE_ATTRIBUTE_WARP_SIZE => 87,
            CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK => 71,
            CU_DEVICE_ATTRIBUTE_CLOCK_RATE => 5,
            CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT => 81,
            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT => 63,
            CU_DEVICE_ATTRIBUTE_INTEGRATED => 16,
            CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY => 3,
            CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS => 8,
            CU_DEVICE_ATTRIBUTE_PCI_BUS_ID => 67,
            CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID => 68,
            CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE => 60,
            CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH => 59,
            CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE => 19,
            CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR => 57,
            CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT => 2,
            CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING => 85,
            CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID => 69,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
            | CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR => return None,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR => 10002,
            CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY => 24,
            CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS => 9,
            CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH => 10,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN => 75,
            CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED => 90,
            CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR => 25,
            CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED => 88,
        })
    }
}

/// Event creation flags; the numbers are HIP's and CUDA's alike.
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUevent_flags_enum {
    CU_EVENT_DEFAULT = 0,
    CU_EVENT_BLOCKING_SYNC = 1,
    CU_EVENT_DISABLE_TIMING = 2,
    CU_EVENT_INTERPROCESS = 4,
}
pub use self::CUevent_flags_enum as CUevent_flags;

/// Event record flags; the numbers are HIP's and CUDA's alike.
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUevent_record_flags_enum {
    CU_EVENT_RECORD_DEFAULT = 0,
    CU_EVENT_RECORD_EXTERNAL = 1,
}
pub use self::CUevent_record_flags_enum as CUevent_record_flags;

/// Stream capture modes; the numbers are HIP's and CUDA's alike.
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstreamCaptureMode_enum {
    CU_STREAM_CAPTURE_MODE_GLOBAL = 0,
    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1,
    CU_STREAM_CAPTURE_MODE_RELAXED = 2,
}
pub use self::CUstreamCaptureMode_enum as CUstreamCaptureMode;

/// Stream capture states; the numbers are HIP's and CUDA's alike.
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstreamCaptureStatus_enum {
    CU_STREAM_CAPTURE_STATUS_NONE = 0,
    CU_STREAM_CAPTURE_STATUS_ACTIVE = 1,
    CU_STREAM_CAPTURE_STATUS_INVALIDATED = 2,
}
pub use self::CUstreamCaptureStatus_enum as CUstreamCaptureStatus;

impl CUstreamCaptureStatus_enum {
    pub(crate) fn from_hip(v: c_int) -> Self {
        match v {
            0 => Self::CU_STREAM_CAPTURE_STATUS_NONE,
            1 => Self::CU_STREAM_CAPTURE_STATUS_ACTIVE,
            _ => Self::CU_STREAM_CAPTURE_STATUS_INVALIDATED,
        }
    }
}

/// Graph node kinds; the numbers are HIP's and CUDA's alike.
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphNodeType_enum {
    CU_GRAPH_NODE_TYPE_KERNEL = 0,
    CU_GRAPH_NODE_TYPE_MEMCPY = 1,
    CU_GRAPH_NODE_TYPE_MEMSET = 2,
    CU_GRAPH_NODE_TYPE_HOST = 3,
    CU_GRAPH_NODE_TYPE_GRAPH = 4,
    CU_GRAPH_NODE_TYPE_EMPTY = 5,
    CU_GRAPH_NODE_TYPE_WAIT_EVENT = 6,
    CU_GRAPH_NODE_TYPE_EVENT_RECORD = 7,
    CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL = 8,
    CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT = 9,
    CU_GRAPH_NODE_TYPE_MEM_ALLOC = 10,
    CU_GRAPH_NODE_TYPE_MEM_FREE = 11,
    CU_GRAPH_NODE_TYPE_BATCH_MEM_OP = 12,
    CU_GRAPH_NODE_TYPE_CONDITIONAL = 13,
}
pub use self::CUgraphNodeType_enum as CUgraphNodeType;

impl CUgraphNodeType_enum {
    pub(crate) fn from_hip(v: c_int) -> Self {
        use CUgraphNodeType_enum::*;
        [
            CU_GRAPH_NODE_TYPE_KERNEL,
            CU_GRAPH_NODE_TYPE_MEMCPY,
            CU_GRAPH_NODE_TYPE_MEMSET,
            CU_GRAPH_NODE_TYPE_HOST,
            CU_GRAPH_NODE_TYPE_GRAPH,
            CU_GRAPH_NODE_TYPE_EMPTY,
            CU_GRAPH_NODE_TYPE_WAIT_EVENT,
            CU_GRAPH_NODE_TYPE_EVENT_RECORD,
            CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL,
            CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT,
            CU_GRAPH_NODE_TYPE_MEM_ALLOC,
            CU_GRAPH_NODE_TYPE_MEM_FREE,
            CU_GRAPH_NODE_TYPE_BATCH_MEM_OP,
            CU_GRAPH_NODE_TYPE_CONDITIONAL,
        ]
        .into_iter()
        .find(|t| *t as u32 as c_int == v)
        .unwrap_or(CU_GRAPH_NODE_TYPE_EMPTY)
    }
}

/// Graph instantiation flags as a bit set; the bits are HIP's and CUDA's alike.
#[repr(transparent)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUgraphInstantiate_flags_enum(pub c_uint);
pub use self::CUgraphInstantiate_flags_enum as CUgraphInstantiate_flags;

impl CUgraphInstantiate_flags_enum {
    pub const CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH: Self = Self(1);
    pub const CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD: Self = Self(2);
    pub const CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH: Self = Self(4);
    pub const CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY: Self = Self(8);
}

impl std::ops::BitOr for CUgraphInstantiate_flags_enum {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl From<CUgraphInstantiate_flags_enum> for u64 {
    fn from(f: CUgraphInstantiate_flags_enum) -> u64 {
        u64::from(f.0)
    }
}

/// The outcome of a graph update.
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphExecUpdateResult_enum {
    CU_GRAPH_EXEC_UPDATE_SUCCESS = 0,
    CU_GRAPH_EXEC_UPDATE_ERROR = 1,
    CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED = 2,
    CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED = 3,
    CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED = 4,
    CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED = 5,
    CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED = 6,
    CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE = 7,
    CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED = 8,
}
pub use self::CUgraphExecUpdateResult_enum as CUgraphExecUpdateResult;

/// What a graph update reports.
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUgraphExecUpdateResultInfo_st {
    pub result: CUgraphExecUpdateResult,
    pub errorNode: CUgraphNode,
    pub errorFromNode: CUgraphNode,
}
pub type CUgraphExecUpdateResultInfo = CUgraphExecUpdateResultInfo_st;

/// Memory pool attributes; the numbers are HIP's and CUDA's alike.
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemPool_attribute_enum {
    CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES = 1,
    CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC = 2,
    CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES = 3,
    CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = 4,
    CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT = 5,
    CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH = 6,
    CU_MEMPOOL_ATTR_USED_MEM_CURRENT = 7,
    CU_MEMPOOL_ATTR_USED_MEM_HIGH = 8,
}
pub use self::CUmemPool_attribute_enum as CUmemPool_attribute;

/// Graph memory attributes; the numbers are HIP's and CUDA's alike.
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphMem_attribute_enum {
    CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT = 0,
    CU_GRAPH_MEM_ATTR_USED_MEM_HIGH = 1,
    CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT = 2,
    CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH = 3,
}
pub use self::CUgraphMem_attribute_enum as CUgraphMem_attribute;

/// Where a copy operand lives, under CUDA's numbers.
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemorytype_enum {
    CU_MEMORYTYPE_HOST = 1,
    CU_MEMORYTYPE_DEVICE = 2,
    CU_MEMORYTYPE_ARRAY = 3,
    CU_MEMORYTYPE_UNIFIED = 4,
}
pub use self::CUmemorytype_enum as CUmemorytype;

impl CUmemorytype_enum {
    fn to_hip(self) -> c_int {
        match self {
            Self::CU_MEMORYTYPE_HOST => 1,
            Self::CU_MEMORYTYPE_DEVICE => 2,
            Self::CU_MEMORYTYPE_ARRAY => 10,
            Self::CU_MEMORYTYPE_UNIFIED => 11,
        }
    }
}

/// The kind of a memory location; the numbers are HIP's and CUDA's alike.
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemLocationType_enum {
    CU_MEM_LOCATION_TYPE_INVALID = 0,
    CU_MEM_LOCATION_TYPE_DEVICE = 1,
    CU_MEM_LOCATION_TYPE_HOST = 2,
    CU_MEM_LOCATION_TYPE_HOST_NUMA = 3,
    CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT = 4,
}
pub use self::CUmemLocationType_enum as CUmemLocationType;

/// The kind of an allocation; the numbers are HIP's and CUDA's alike.
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemAllocationType_enum {
    CU_MEM_ALLOCATION_TYPE_INVALID = 0,
    CU_MEM_ALLOCATION_TYPE_PINNED = 1,
    CU_MEM_ALLOCATION_TYPE_MANAGED = 2,
}
pub use self::CUmemAllocationType_enum as CUmemAllocationType;

/// Which granularity a query asks for; the numbers are HIP's and CUDA's alike.
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemAllocationGranularity_flags_enum {
    CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0,
    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1,
}
pub use self::CUmemAllocationGranularity_flags_enum as CUmemAllocationGranularity_flags;

/// The identifier of a memory location.
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUmemLocation_st__bindgen_ty_1 {
    pub id: c_int,
}

/// A memory location: a kind and an identifier.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUmemLocation_st {
    pub type_: CUmemLocationType,
    pub __bindgen_anon_1: CUmemLocation_st__bindgen_ty_1,
}
pub type CUmemLocation = CUmemLocation_st;

impl CUmemLocation_st {
    fn to_hip(self) -> HipMemLocation {
        HipMemLocation {
            type_: self.type_ as c_int,
            // SAFETY: the union's only member is the identifier.
            id: unsafe { self.__bindgen_anon_1.id },
        }
    }
}

/// The properties of a memory pool.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUmemPoolProps_st {
    pub allocType: CUmemAllocationType,
    pub handleTypes: c_uint,
    pub location: CUmemLocation,
    pub win32SecurityAttributes: *mut c_void,
    pub maxSize: usize,
    pub usage: u16,
    pub reserved: [u8; 54],
}
pub type CUmemPoolProps = CUmemPoolProps_st;

/// The properties of a physical allocation.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUmemAllocationProp_st {
    pub type_: CUmemAllocationType,
    pub requestedHandleTypes: c_uint,
    pub location: CUmemLocation,
    pub win32HandleMetaData: *mut c_void,
    pub allocFlags: CUmemAllocationProp_st__bindgen_ty_1,
}
pub type CUmemAllocationProp = CUmemAllocationProp_st;

/// The flags of a physical allocation.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUmemAllocationProp_st__bindgen_ty_1 {
    pub compressionType: u8,
    pub gpuDirectRDMACapable: u8,
    pub usage: u16,
    pub reserved: [u8; 4],
}

/// A two-dimensional copy.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUDA_MEMCPY2D_st {
    pub srcXInBytes: usize,
    pub srcY: usize,
    pub srcMemoryType: CUmemorytype,
    pub srcHost: *const c_void,
    pub srcDevice: CUdeviceptr,
    pub srcArray: CUarray,
    pub srcPitch: usize,
    pub dstXInBytes: usize,
    pub dstY: usize,
    pub dstMemoryType: CUmemorytype,
    pub dstHost: *mut c_void,
    pub dstDevice: CUdeviceptr,
    pub dstArray: CUarray,
    pub dstPitch: usize,
    pub WidthInBytes: usize,
    pub Height: usize,
}
pub type CUDA_MEMCPY2D = CUDA_MEMCPY2D_st;

/// The parameters of a kernel node.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUDA_KERNEL_NODE_PARAMS_v2_st {
    pub func: CUfunction,
    pub gridDimX: c_uint,
    pub gridDimY: c_uint,
    pub gridDimZ: c_uint,
    pub blockDimX: c_uint,
    pub blockDimY: c_uint,
    pub blockDimZ: c_uint,
    pub sharedMemBytes: c_uint,
    pub kernelParams: *mut *mut c_void,
    pub extra: *mut *mut c_void,
    pub kern: CUkernel,
    pub ctx: CUcontext,
}
pub type CUDA_KERNEL_NODE_PARAMS = CUDA_KERNEL_NODE_PARAMS_v2_st;

/// Page-locked host allocation flags; the bits are HIP's and CUDA's alike.
pub const CU_MEMHOSTALLOC_PORTABLE: u32 = 1;
pub const CU_MEMHOSTALLOC_DEVICEMAP: u32 = 2;
pub const CU_MEMHOSTALLOC_WRITECOMBINED: u32 = 4;
/// Host registration flags; the bits are HIP's and CUDA's alike.
pub const CU_MEMHOSTREGISTER_PORTABLE: u32 = 1;
pub const CU_MEMHOSTREGISTER_DEVICEMAP: u32 = 2;
pub const CU_MEMHOSTREGISTER_IOMEMORY: u32 = 4;
pub const CU_MEMHOSTREGISTER_READ_ONLY: u32 = 8;

// HIP's own layouts, filled from the CUDA-named structures above.

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct HipMemLocation {
    type_: c_int,
    id: c_int,
}

#[repr(C)]
pub(crate) struct HipMemPoolProps {
    alloc_type: c_int,
    handle_types: c_int,
    location: HipMemLocation,
    win32_security_attributes: *mut c_void,
    max_size: usize,
    reserved: [u8; 56],
}

#[repr(C)]
pub(crate) struct HipMemAllocationProp {
    type_: c_int,
    requested_handle_type: c_int,
    location: HipMemLocation,
    win32_handle_meta_data: *mut c_void,
    compression_type: u8,
    gpu_direct_rdma_capable: u8,
    usage: u16,
}

#[repr(C)]
pub(crate) struct HipMemcpy2D {
    src_x_in_bytes: usize,
    src_y: usize,
    src_memory_type: c_int,
    src_host: *const c_void,
    src_device: *mut c_void,
    src_array: *mut c_void,
    src_pitch: usize,
    dst_x_in_bytes: usize,
    dst_y: usize,
    dst_memory_type: c_int,
    dst_host: *mut c_void,
    dst_device: *mut c_void,
    dst_array: *mut c_void,
    dst_pitch: usize,
    width_in_bytes: usize,
    height: usize,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct Dim3 {
    x: c_uint,
    y: c_uint,
    z: c_uint,
}

#[repr(C)]
pub(crate) struct HipKernelNodeParams {
    block_dim: Dim3,
    extra: *mut *mut c_void,
    func: *mut c_void,
    grid_dim: Dim3,
    kernel_params: *mut *mut c_void,
    shared_mem_bytes: c_uint,
}

/// HIP's device properties, as `hipGetDevicePropertiesR0600` writes them.
#[repr(C)]
pub(crate) struct HipDeviceProp {
    pub name: [c_char; 256],
    uuid: [c_char; 16],
    luid: [c_char; 8],
    luid_device_node_mask: c_uint,
    total_global_mem: usize,
    shared_mem_per_block: usize,
    regs_per_block: c_int,
    warp_size: c_int,
    mem_pitch: usize,
    max_threads_per_block: c_int,
    max_threads_dim: [c_int; 3],
    max_grid_size: [c_int; 3],
    clock_rate: c_int,
    total_const_mem: usize,
    major: c_int,
    minor: c_int,
    texture_alignment: usize,
    texture_pitch_alignment: usize,
    device_overlap: c_int,
    multi_processor_count: c_int,
    kernel_exec_timeout_enabled: c_int,
    integrated: c_int,
    can_map_host_memory: c_int,
    compute_mode: c_int,
    max_texture_1d: c_int,
    max_texture_1d_mipmap: c_int,
    max_texture_1d_linear: c_int,
    max_texture_2d: [c_int; 2],
    max_texture_2d_mipmap: [c_int; 2],
    max_texture_2d_linear: [c_int; 3],
    max_texture_2d_gather: [c_int; 2],
    max_texture_3d: [c_int; 3],
    max_texture_3d_alt: [c_int; 3],
    max_texture_cubemap: c_int,
    max_texture_1d_layered: [c_int; 2],
    max_texture_2d_layered: [c_int; 3],
    max_texture_cubemap_layered: [c_int; 2],
    max_surface_1d: c_int,
    max_surface_2d: [c_int; 2],
    max_surface_3d: [c_int; 3],
    max_surface_1d_layered: [c_int; 2],
    max_surface_2d_layered: [c_int; 3],
    max_surface_cubemap: c_int,
    max_surface_cubemap_layered: [c_int; 2],
    surface_alignment: usize,
    concurrent_kernels: c_int,
    ecc_enabled: c_int,
    pci_bus_id: c_int,
    pci_device_id: c_int,
    pci_domain_id: c_int,
    tcc_driver: c_int,
    async_engine_count: c_int,
    unified_addressing: c_int,
    memory_clock_rate: c_int,
    memory_bus_width: c_int,
    l2_cache_size: c_int,
    persisting_l2_cache_max_size: c_int,
    max_threads_per_multi_processor: c_int,
    stream_priorities_supported: c_int,
    global_l1_cache_supported: c_int,
    local_l1_cache_supported: c_int,
    shared_mem_per_multiprocessor: usize,
    regs_per_multiprocessor: c_int,
    managed_memory: c_int,
    is_multi_gpu_board: c_int,
    multi_gpu_board_group_id: c_int,
    host_native_atomic_supported: c_int,
    single_to_double_precision_perf_ratio: c_int,
    pageable_memory_access: c_int,
    concurrent_managed_access: c_int,
    compute_preemption_supported: c_int,
    can_use_host_pointer_for_registered_mem: c_int,
    cooperative_launch: c_int,
    cooperative_multi_device_launch: c_int,
    shared_mem_per_block_optin: usize,
    pageable_memory_access_uses_host_page_tables: c_int,
    direct_managed_mem_access_from_host: c_int,
    max_blocks_per_multi_processor: c_int,
    access_policy_max_window_size: c_int,
    reserved_shared_mem_per_block: usize,
    host_register_supported: c_int,
    sparse_hip_array_supported: c_int,
    host_register_read_only_supported: c_int,
    timeline_semaphore_interop_supported: c_int,
    memory_pools_supported: c_int,
    gpu_direct_rdma_supported: c_int,
    gpu_direct_rdma_flush_writes_options: c_uint,
    gpu_direct_rdma_writes_ordering: c_int,
    memory_pool_supported_handle_types: c_uint,
    deferred_mapping_hip_array_supported: c_int,
    ipc_event_supported: c_int,
    cluster_launch: c_int,
    unified_function_pointers: c_int,
    reserved: [c_int; 63],
    hip_reserved: [c_int; 32],
    pub gcn_arch_name: [c_char; 256],
    max_shared_memory_per_multi_processor: usize,
    clock_instruction_rate: c_int,
    arch: c_uint,
    hdp_mem_flush_cntl: *mut c_uint,
    hdp_reg_flush_cntl: *mut c_uint,
    cooperative_multi_device_unmatched_func: c_int,
    cooperative_multi_device_unmatched_grid_dim: c_int,
    cooperative_multi_device_unmatched_block_dim: c_int,
    cooperative_multi_device_unmatched_shared_mem: c_int,
    is_large_bar: c_int,
    asic_revision: c_int,
}

fn ptr(d: CUdeviceptr) -> *mut c_void {
    d as usize as *mut c_void
}

pub unsafe fn cuInit(flags: c_uint) -> CUresult {
    unsafe { hip!(hipInit(flags)) }
}

pub unsafe fn cuDeviceGetCount(count: *mut c_int) -> CUresult {
    unsafe { hip!(hipGetDeviceCount(count)) }
}

pub unsafe fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult {
    unsafe { hip!(hipDeviceGet(device, ordinal)) }
}

pub unsafe fn cuDeviceGetAttribute(
    value: *mut c_int,
    attrib: CUdevice_attribute,
    dev: CUdevice,
) -> CUresult {
    match attrib.to_hip() {
        Some(a) => unsafe { hip!(hipDeviceGetAttribute(value, a, dev)) },
        None => CUresult::CUDA_ERROR_NOT_SUPPORTED,
    }
}

pub unsafe fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult {
    unsafe { hip!(hipDeviceGetName(name, len, dev)) }
}

pub unsafe fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult {
    unsafe { hip!(hipDeviceTotalMem(bytes, dev)) }
}

pub unsafe fn cuDeviceGetPCIBusId(id: *mut c_char, len: c_int, dev: CUdevice) -> CUresult {
    unsafe { hip!(hipDeviceGetPCIBusId(id, len, dev)) }
}

pub unsafe fn cuDeviceCanAccessPeer(can: *mut c_int, dev: CUdevice, peer: CUdevice) -> CUresult {
    unsafe { hip!(hipDeviceCanAccessPeer(can, dev, peer)) }
}

pub unsafe fn cuDevicePrimaryCtxRetain(ctx: *mut CUcontext, dev: CUdevice) -> CUresult {
    unsafe { hip!(hipDevicePrimaryCtxRetain(ctx, dev)) }
}

pub unsafe fn cuDevicePrimaryCtxRelease_v2(dev: CUdevice) -> CUresult {
    unsafe { hip!(hipDevicePrimaryCtxRelease(dev)) }
}

pub unsafe fn cuDevicePrimaryCtxReset_v2(dev: CUdevice) -> CUresult {
    unsafe { hip!(hipDevicePrimaryCtxReset(dev)) }
}

pub unsafe fn cuDevicePrimaryCtxGetState(
    dev: CUdevice,
    flags: *mut c_uint,
    active: *mut c_int,
) -> CUresult {
    unsafe { hip!(hipDevicePrimaryCtxGetState(dev, flags, active)) }
}

pub unsafe fn cuCtxPushCurrent_v2(ctx: CUcontext) -> CUresult {
    unsafe { hip!(hipCtxPushCurrent(ctx)) }
}

pub unsafe fn cuCtxPopCurrent_v2(ctx: *mut CUcontext) -> CUresult {
    unsafe { hip!(hipCtxPopCurrent(ctx)) }
}

pub unsafe fn cuCtxSynchronize() -> CUresult {
    unsafe { hip!(hipCtxSynchronize()) }
}

pub unsafe fn cuMemHostAlloc(p: *mut *mut c_void, bytes: usize, flags: c_uint) -> CUresult {
    unsafe { hip!(hipHostMalloc(p, bytes, flags)) }
}

pub unsafe fn cuMemFreeHost(p: *mut c_void) -> CUresult {
    unsafe { hip!(hipHostFree(p)) }
}

pub unsafe fn cuMemHostRegister_v2(p: *mut c_void, bytes: usize, flags: c_uint) -> CUresult {
    unsafe { hip!(hipHostRegister(p, bytes, flags)) }
}

pub unsafe fn cuMemHostUnregister(p: *mut c_void) -> CUresult {
    unsafe { hip!(hipHostUnregister(p)) }
}

pub unsafe fn cuMemHostGetDevicePointer_v2(
    d: *mut CUdeviceptr,
    host: *mut c_void,
    flags: c_uint,
) -> CUresult {
    let mut out: *mut c_void = std::ptr::null_mut();
    let r = unsafe { hip!(hipHostGetDevicePointer(&mut out, host, flags)) };
    if r == CUresult::CUDA_SUCCESS {
        unsafe { *d = out as usize as CUdeviceptr };
    }
    r
}

/// Runs `copy(offset, len)` over `bytes` from `src`: at once, or in pieces the runtime stages
/// without trying to page-lock them when `src` lies in memory that cannot be page-locked.
fn in_pieces(
    src: *const c_void,
    bytes: usize,
    mut copy: impl FnMut(usize, usize) -> CUresult,
) -> CUresult {
    let piece = super::unpinnable::PIECE;
    if bytes <= piece || !super::unpinnable::contains(src as usize, bytes) {
        return copy(0, bytes);
    }
    let mut at = 0;
    while at < bytes {
        let len = piece.min(bytes - at);
        let r = copy(at, len);
        if r != CUresult::CUDA_SUCCESS {
            return r;
        }
        at += len;
    }
    CUresult::CUDA_SUCCESS
}

pub unsafe fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, bytes: usize) -> CUresult {
    in_pieces(src, bytes, |at, len| unsafe {
        hip!(hipMemcpyHtoD(
            ptr(dst + at as CUdeviceptr),
            (src as *const u8).add(at) as *const c_void,
            len
        ))
    })
}

pub unsafe fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> CUresult {
    unsafe { hip!(hipMemcpyDtoH(dst, ptr(src), bytes)) }
}

pub unsafe fn cuMemcpyHtoDAsync_v2(
    dst: CUdeviceptr,
    src: *const c_void,
    bytes: usize,
    stream: CUstream,
) -> CUresult {
    in_pieces(src, bytes, |at, len| unsafe {
        hip!(hipMemcpyHtoDAsync(
            ptr(dst + at as CUdeviceptr),
            (src as *const u8).add(at) as *const c_void,
            len,
            stream
        ))
    })
}

pub unsafe fn cuMemcpyDtoHAsync_v2(
    dst: *mut c_void,
    src: CUdeviceptr,
    bytes: usize,
    stream: CUstream,
) -> CUresult {
    unsafe { hip!(hipMemcpyDtoHAsync(dst, ptr(src), bytes, stream)) }
}

pub unsafe fn cuMemcpyDtoDAsync_v2(
    dst: CUdeviceptr,
    src: CUdeviceptr,
    bytes: usize,
    stream: CUstream,
) -> CUresult {
    unsafe { hip!(hipMemcpyDtoDAsync(ptr(dst), ptr(src), bytes, stream)) }
}

pub unsafe fn cuMemcpy2DAsync_v2(desc: *const CUDA_MEMCPY2D, stream: CUstream) -> CUresult {
    let d = unsafe { &*desc };
    let hip_desc = HipMemcpy2D {
        src_x_in_bytes: d.srcXInBytes,
        src_y: d.srcY,
        src_memory_type: d.srcMemoryType.to_hip(),
        src_host: d.srcHost,
        src_device: ptr(d.srcDevice),
        src_array: d.srcArray as *mut c_void,
        src_pitch: d.srcPitch,
        dst_x_in_bytes: d.dstXInBytes,
        dst_y: d.dstY,
        dst_memory_type: d.dstMemoryType.to_hip(),
        dst_host: d.dstHost,
        dst_device: ptr(d.dstDevice),
        dst_array: d.dstArray as *mut c_void,
        dst_pitch: d.dstPitch,
        width_in_bytes: d.WidthInBytes,
        height: d.Height,
    };
    unsafe { hip!(hipMemcpyParam2DAsync(&hip_desc, stream)) }
}

pub unsafe fn cuMemsetD8_v2(dst: CUdeviceptr, value: u8, n: usize) -> CUresult {
    unsafe { hip!(hipMemsetD8(ptr(dst), value, n)) }
}

pub unsafe fn cuMemsetD8Async(dst: CUdeviceptr, value: u8, n: usize, stream: CUstream) -> CUresult {
    unsafe { hip!(hipMemsetD8Async(ptr(dst), value, n, stream)) }
}

pub unsafe fn cuStreamQuery(stream: CUstream) -> CUresult {
    unsafe { hip!(hipStreamQuery(stream)) }
}

pub unsafe fn cuStreamIsCapturing(
    stream: CUstream,
    status: *mut CUstreamCaptureStatus,
) -> CUresult {
    let mut s: c_int = 0;
    let r = unsafe { hip!(hipStreamIsCapturing(stream, &mut s)) };
    if r == CUresult::CUDA_SUCCESS {
        unsafe { *status = CUstreamCaptureStatus::from_hip(s) };
    }
    r
}

pub unsafe fn cuStreamBeginCapture_v2(stream: CUstream, mode: CUstreamCaptureMode) -> CUresult {
    let r = unsafe { hip!(hipStreamBeginCapture(stream, mode as c_int)) };
    if r == CUresult::CUDA_SUCCESS {
        super::capture_hold::begin(stream);
    }
    r
}

pub unsafe fn cuStreamEndCapture(stream: CUstream, graph: *mut CUgraph) -> CUresult {
    let mut g: CUgraph = std::ptr::null_mut();
    let r = unsafe { hip!(hipStreamEndCapture(stream, &mut g)) };
    let g = if r == CUresult::CUDA_SUCCESS {
        g
    } else {
        std::ptr::null_mut()
    };
    super::capture_hold::end(stream, g);
    unsafe { *graph = g };
    r
}

pub unsafe fn cuEventRecordWithFlags(event: CUevent, stream: CUstream, flags: c_uint) -> CUresult {
    unsafe { hip!(hipEventRecordWithFlags(event, stream, flags)) }
}

pub unsafe fn cuGraphGetNodes(graph: CUgraph, nodes: *mut CUgraphNode, n: *mut usize) -> CUresult {
    unsafe { hip!(hipGraphGetNodes(graph, nodes, n)) }
}

pub unsafe fn cuGraphNodeGetType(node: CUgraphNode, ty: *mut CUgraphNodeType) -> CUresult {
    let mut t: c_int = 0;
    let r = unsafe { hip!(hipGraphNodeGetType(node, &mut t)) };
    if r == CUresult::CUDA_SUCCESS {
        unsafe { *ty = CUgraphNodeType::from_hip(t) };
    }
    r
}

pub unsafe fn cuGraphNodeSetEnabled(exec: CUgraphExec, node: CUgraphNode, on: c_uint) -> CUresult {
    unsafe { hip!(hipGraphNodeSetEnabled(exec, node, on)) }
}

pub unsafe fn cuGraphKernelNodeGetParams_v2(
    node: CUgraphNode,
    params: *mut CUDA_KERNEL_NODE_PARAMS,
) -> CUresult {
    let mut p = HipKernelNodeParams {
        block_dim: Dim3 { x: 0, y: 0, z: 0 },
        extra: std::ptr::null_mut(),
        func: std::ptr::null_mut(),
        grid_dim: Dim3 { x: 0, y: 0, z: 0 },
        kernel_params: std::ptr::null_mut(),
        shared_mem_bytes: 0,
    };
    let r = unsafe { hip!(hipGraphKernelNodeGetParams(node, &mut p)) };
    if r == CUresult::CUDA_SUCCESS {
        unsafe {
            *params = CUDA_KERNEL_NODE_PARAMS {
                func: p.func as CUfunction,
                gridDimX: p.grid_dim.x,
                gridDimY: p.grid_dim.y,
                gridDimZ: p.grid_dim.z,
                blockDimX: p.block_dim.x,
                blockDimY: p.block_dim.y,
                blockDimZ: p.block_dim.z,
                sharedMemBytes: p.shared_mem_bytes,
                kernelParams: p.kernel_params,
                extra: p.extra,
                kern: std::ptr::null_mut(),
                ctx: std::ptr::null_mut(),
            }
        };
    }
    r
}

/// HIP names no function of a kernel node: always not supported.
pub unsafe fn cuFuncGetName(_name: *mut *const c_char, _func: CUfunction) -> CUresult {
    CUresult::CUDA_ERROR_NOT_SUPPORTED
}

pub unsafe fn cuGraphExecUpdate_v2(
    exec: CUgraphExec,
    graph: CUgraph,
    info: *mut CUgraphExecUpdateResultInfo,
) -> CUresult {
    let mut node: CUgraphNode = std::ptr::null_mut();
    let mut outcome: c_int = CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_ERROR as c_int;
    let r = unsafe { hip!(hipGraphExecUpdate(exec, graph, &mut node, &mut outcome)) };
    if !info.is_null() {
        let result = if outcome == 0 {
            CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_SUCCESS
        } else {
            CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_ERROR
        };
        unsafe {
            *info = CUgraphExecUpdateResultInfo {
                result,
                errorNode: node,
                errorFromNode: std::ptr::null_mut(),
            }
        };
    }
    if r == CUresult::CUDA_SUCCESS && outcome == 0 {
        super::capture_hold::updated(exec, graph);
    }
    r
}

pub unsafe fn cuGraphInstantiateWithFlags(
    exec: *mut CUgraphExec,
    graph: CUgraph,
    flags: u64,
) -> CUresult {
    let r = unsafe { hip!(hipGraphInstantiateWithFlags(exec, graph, flags)) };
    if r == CUresult::CUDA_SUCCESS {
        super::capture_hold::instantiated(graph, unsafe { *exec });
    }
    r
}

pub unsafe fn cuGraphExecDestroy(exec: CUgraphExec) -> CUresult {
    let r = unsafe { hip!(hipGraphExecDestroy(exec)) };
    super::capture_hold::exec_destroyed(exec);
    r
}

pub unsafe fn cuGraphDestroy(graph: CUgraph) -> CUresult {
    let r = unsafe { hip!(hipGraphDestroy(graph)) };
    super::capture_hold::graph_destroyed(graph);
    r
}

pub unsafe fn cuGraphLaunch(exec: CUgraphExec, stream: CUstream) -> CUresult {
    unsafe { hip!(hipGraphLaunch(exec, stream)) }
}

pub unsafe fn cuGraphUpload(exec: CUgraphExec, stream: CUstream) -> CUresult {
    unsafe { hip!(hipGraphUpload(exec, stream)) }
}

pub unsafe fn cuDeviceGetDefaultMemPool(pool: *mut CUmemoryPool, dev: CUdevice) -> CUresult {
    unsafe { hip!(hipDeviceGetDefaultMemPool(pool, dev)) }
}

pub unsafe fn cuMemPoolGetAttribute(
    pool: CUmemoryPool,
    attr: CUmemPool_attribute,
    value: *mut c_void,
) -> CUresult {
    unsafe { hip!(hipMemPoolGetAttribute(pool, attr as c_int, value)) }
}

pub unsafe fn cuMemPoolSetAttribute(
    pool: CUmemoryPool,
    attr: CUmemPool_attribute,
    value: *mut c_void,
) -> CUresult {
    unsafe { hip!(hipMemPoolSetAttribute(pool, attr as c_int, value)) }
}

pub unsafe fn cuMemPoolTrimTo(pool: CUmemoryPool, keep: usize) -> CUresult {
    unsafe { hip!(hipMemPoolTrimTo(pool, keep)) }
}

pub unsafe fn cuMemPoolDestroy(pool: CUmemoryPool) -> CUresult {
    unsafe { hip!(hipMemPoolDestroy(pool)) }
}

pub unsafe fn cuMemPoolCreate(pool: *mut CUmemoryPool, props: *const CUmemPoolProps) -> CUresult {
    let p = unsafe { &*props };
    let hip_props = HipMemPoolProps {
        alloc_type: p.allocType as c_int,
        handle_types: p.handleTypes as c_int,
        location: p.location.to_hip(),
        win32_security_attributes: p.win32SecurityAttributes,
        max_size: p.maxSize,
        reserved: [0; 56],
    };
    unsafe { hip!(hipMemPoolCreate(pool, &hip_props)) }
}

pub unsafe fn cuMemAllocFromPoolAsync(
    d: *mut CUdeviceptr,
    bytes: usize,
    pool: CUmemoryPool,
    stream: CUstream,
) -> CUresult {
    let mut out: *mut c_void = std::ptr::null_mut();
    let r = unsafe { hip!(hipMallocFromPoolAsync(&mut out, bytes, pool, stream)) };
    if r == CUresult::CUDA_SUCCESS {
        unsafe { *d = out as usize as CUdeviceptr };
    }
    r
}

pub unsafe fn cuMemGetAllocationGranularity(
    granularity: *mut usize,
    prop: *const CUmemAllocationProp,
    option: CUmemAllocationGranularity_flags,
) -> CUresult {
    let p = unsafe { &*prop };
    let hip_prop = HipMemAllocationProp {
        type_: p.type_ as c_int,
        requested_handle_type: p.requestedHandleTypes as c_int,
        location: p.location.to_hip(),
        win32_handle_meta_data: p.win32HandleMetaData,
        compression_type: p.allocFlags.compressionType,
        gpu_direct_rdma_capable: p.allocFlags.gpuDirectRDMACapable,
        usage: p.allocFlags.usage,
    };
    unsafe {
        hip!(hipMemGetAllocationGranularity(
            granularity,
            &hip_prop,
            option as c_int
        ))
    }
}

pub unsafe fn cuDeviceGetGraphMemAttribute(
    dev: CUdevice,
    attr: CUgraphMem_attribute,
    value: *mut c_void,
) -> CUresult {
    unsafe { hip!(hipDeviceGetGraphMemAttribute(dev, attr as c_int, value)) }
}

pub unsafe fn cuDeviceGraphMemTrim(dev: CUdevice) -> CUresult {
    unsafe { hip!(hipDeviceGraphMemTrim(dev)) }
}
