//! Collective communication across devices under the NCCL names, done by RCCL, which keeps
//! NCCL's entry points and numbering.

pub mod sys {
    #![allow(non_camel_case_types, clippy::missing_safety_doc)]

    use std::ffi::{c_int, c_void};

    crate::tensor::gpu_runtime::hip::dynamic_api! {
        /// The entry points of the RCCL library.
        pub(crate) struct Rccl from "rccl" {
            fn ncclCommInitAll(comms: *mut ncclComm_t, n: c_int, devices: *const c_int) -> ncclResult_t;
            fn ncclCommAbort(comm: ncclComm_t) -> ncclResult_t;
            fn ncclGroupStart() -> ncclResult_t;
            fn ncclGroupEnd() -> ncclResult_t;
            fn ncclAllReduce(
                send: *const c_void,
                recv: *mut c_void,
                count: usize,
                ty: ncclDataType_t,
                op: ncclRedOp_t,
                comm: ncclComm_t,
                stream: *mut c_void,
            ) -> ncclResult_t;
            fn ncclReduce(
                send: *const c_void,
                recv: *mut c_void,
                count: usize,
                ty: ncclDataType_t,
                op: ncclRedOp_t,
                root: c_int,
                comm: ncclComm_t,
                stream: *mut c_void,
            ) -> ncclResult_t;
            fn ncclSend(
                send: *const c_void,
                count: usize,
                ty: ncclDataType_t,
                peer: c_int,
                comm: ncclComm_t,
                stream: *mut c_void,
            ) -> ncclResult_t;
            fn ncclRecv(
                recv: *mut c_void,
                count: usize,
                ty: ncclDataType_t,
                peer: c_int,
                comm: ncclComm_t,
                stream: *mut c_void,
            ) -> ncclResult_t;
        }
    }

    /// Whether the RCCL library opens.
    pub unsafe fn is_culib_present() -> bool {
        Rccl::present()
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct ncclComm {
        _unused: [u8; 0],
    }
    /// A communicator handle.
    pub type ncclComm_t = *mut ncclComm;

    /// An element type.
    #[repr(u32)]
    #[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
    pub enum ncclDataType_t {
        ncclInt8 = 0,
        ncclUint8 = 1,
        ncclInt32 = 2,
        ncclUint32 = 3,
        ncclInt64 = 4,
        ncclUint64 = 5,
        ncclFloat16 = 6,
        ncclFloat32 = 7,
        ncclFloat64 = 8,
        ncclBfloat16 = 9,
    }

    /// A reduction.
    #[repr(u32)]
    #[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
    pub enum ncclRedOp_t {
        ncclSum = 0,
        ncclProd = 1,
        ncclMax = 2,
        ncclMin = 3,
        ncclAvg = 4,
    }

    /// A library status.
    #[repr(u32)]
    #[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
    pub enum ncclResult_t {
        ncclSuccess = 0,
        ncclUnhandledCudaError = 1,
        ncclSystemError = 2,
        ncclInternalError = 3,
        ncclInvalidArgument = 4,
        ncclInvalidUsage = 5,
        ncclRemoteError = 6,
        ncclInProgress = 7,
        ncclNumResults = 8,
    }
}

pub mod result {
    use super::sys;

    /// A failed library call: its status.
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct NcclError(pub sys::ncclResult_t);

    impl std::fmt::Debug for NcclError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "NcclError({:?})", self.0)
        }
    }

    /// A call's outcome when it did not fail.
    #[derive(Clone, PartialEq, Eq, Debug)]
    pub enum NcclStatus {
        Success,
        InProgress,
        NumResults,
    }

    impl sys::ncclResult_t {
        /// `Ok` for success or a call still running, the status as an error otherwise.
        pub fn result(self) -> Result<NcclStatus, NcclError> {
            match self {
                sys::ncclResult_t::ncclSuccess => Ok(NcclStatus::Success),
                sys::ncclResult_t::ncclInProgress => Ok(NcclStatus::InProgress),
                sys::ncclResult_t::ncclNumResults => Ok(NcclStatus::NumResults),
                e => Err(NcclError(e)),
            }
        }
    }

    /// Calls an RCCL entry point; a missing library or entry point is a system error.
    macro_rules! rccl {
        ($f:ident ( $( $a:expr ),* $(,)? )) => {
            match sys::Rccl::get().and_then(|api| api.$f) {
                None => sys::ncclResult_t::ncclSystemError,
                Some(f) => f($( $a ),*),
            }
        };
    }
    pub(super) use rccl;

    /// Opens a group of calls issued together.
    pub fn group_start() -> Result<NcclStatus, NcclError> {
        unsafe { rccl!(ncclGroupStart()) }.result()
    }

    /// Closes a group of calls and issues them.
    pub fn group_end() -> Result<NcclStatus, NcclError> {
        unsafe { rccl!(ncclGroupEnd()) }.result()
    }
}

pub mod safe {
    use std::ffi::{c_int, c_void};
    use std::sync::Arc;

    use super::result::{rccl, NcclError, NcclStatus};
    use super::sys;
    use crate::tensor::gpu_runtime::hip::driver::{
        CudaContext, CudaStream, DevicePtr, DevicePtrMut,
    };

    pub use super::result::{group_end, group_start};

    /// One device's member of a communicator, issuing on its stream.
    #[derive(Debug)]
    pub struct Comm {
        comm: sys::ncclComm_t,
        stream: Arc<CudaStream>,
        rank: usize,
        world_size: usize,
    }

    unsafe impl Send for Comm {}
    unsafe impl Sync for Comm {}

    impl Drop for Comm {
        fn drop(&mut self) {
            let _ = unsafe { rccl!(ncclCommAbort(self.comm)) };
        }
    }

    /// A reduction.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ReduceOp {
        Sum,
        Prod,
        Max,
        Min,
        Avg,
    }

    fn op(o: &ReduceOp) -> sys::ncclRedOp_t {
        match o {
            ReduceOp::Sum => sys::ncclRedOp_t::ncclSum,
            ReduceOp::Prod => sys::ncclRedOp_t::ncclProd,
            ReduceOp::Max => sys::ncclRedOp_t::ncclMax,
            ReduceOp::Min => sys::ncclRedOp_t::ncclMin,
            ReduceOp::Avg => sys::ncclRedOp_t::ncclAvg,
        }
    }

    /// An element type the library reduces.
    pub trait NcclType {
        fn as_nccl_type() -> sys::ncclDataType_t;
    }

    macro_rules! nccl_type {
        ($($t:ty => $v:ident),* $(,)?) => {$(
            impl NcclType for $t {
                fn as_nccl_type() -> sys::ncclDataType_t {
                    sys::ncclDataType_t::$v
                }
            }
        )*};
    }
    nccl_type!(
        f32 => ncclFloat32,
        f64 => ncclFloat64,
        i8 => ncclInt8,
        i32 => ncclInt32,
        i64 => ncclInt64,
        u8 => ncclUint8,
        u32 => ncclUint32,
        u64 => ncclUint64,
        half::f16 => ncclFloat16,
        half::bf16 => ncclBfloat16,
    );

    impl Comm {
        /// One member per stream, all in this process, ranked in the order given.
        pub fn from_devices(streams: Vec<Arc<CudaStream>>) -> Result<Vec<Self>, NcclError> {
            let n = streams.len();
            let mut comms = vec![std::ptr::null_mut(); n];
            let ordinals: Vec<c_int> = streams
                .iter()
                .map(|s| s.context().ordinal() as c_int)
                .collect();
            unsafe {
                rccl!(ncclCommInitAll(
                    comms.as_mut_ptr(),
                    n as c_int,
                    ordinals.as_ptr()
                ))
            }
            .result()?;
            Ok(comms
                .into_iter()
                .zip(streams)
                .enumerate()
                .map(|(rank, (comm, stream))| Self {
                    comm,
                    stream,
                    rank,
                    world_size: n,
                })
                .collect())
        }

        /// The stream the member issues on.
        pub fn stream(&self) -> Arc<CudaStream> {
            self.stream.clone()
        }

        /// The member's device.
        pub fn context(&self) -> &Arc<CudaContext> {
            self.stream.context()
        }

        /// The member's device ordinal.
        pub fn ordinal(&self) -> usize {
            self.stream.context().ordinal()
        }

        /// The member's rank.
        pub fn rank(&self) -> usize {
            self.rank
        }

        /// The number of members.
        pub fn world_size(&self) -> usize {
            self.world_size
        }

        fn s(&self) -> *mut c_void {
            self.stream.cu_stream() as *mut c_void
        }

        /// Sends a device range to `peer`.
        pub fn send<S: DevicePtr<T>, T: NcclType>(
            &self,
            data: &S,
            peer: i32,
        ) -> Result<NcclStatus, NcclError> {
            let (src, _record) = data.device_ptr(&self.stream);
            unsafe {
                rccl!(ncclSend(
                    src as usize as *const c_void,
                    data.len(),
                    T::as_nccl_type(),
                    peer,
                    self.comm,
                    self.s()
                ))
            }
            .result()
        }

        /// Receives a device range from `peer`.
        pub fn recv<R: DevicePtrMut<T>, T: NcclType>(
            &self,
            buff: &mut R,
            peer: i32,
        ) -> Result<NcclStatus, NcclError> {
            let count = buff.len();
            let (dst, _record) = buff.device_ptr_mut(&self.stream);
            unsafe {
                rccl!(ncclRecv(
                    dst as usize as *mut c_void,
                    count,
                    T::as_nccl_type(),
                    peer,
                    self.comm,
                    self.s()
                ))
            }
            .result()
        }

        /// Reduces `sendbuff` across members into every member's `recvbuff`.
        pub fn all_reduce<S: DevicePtr<T>, R: DevicePtrMut<T>, T: NcclType>(
            &self,
            sendbuff: &S,
            recvbuff: &mut R,
            reduce_op: &ReduceOp,
        ) -> Result<NcclStatus, NcclError> {
            let count = sendbuff.len();
            let (src, _record_src) = sendbuff.device_ptr(&self.stream);
            let (dst, _record_dst) = recvbuff.device_ptr_mut(&self.stream);
            unsafe {
                rccl!(ncclAllReduce(
                    src as usize as *const c_void,
                    dst as usize as *mut c_void,
                    count,
                    T::as_nccl_type(),
                    op(reduce_op),
                    self.comm,
                    self.s()
                ))
            }
            .result()
        }

        /// Reduces `buff` across members in place.
        pub fn all_reduce_in_place<R: DevicePtrMut<T>, T: NcclType>(
            &self,
            buff: &mut R,
            reduce_op: &ReduceOp,
        ) -> Result<NcclStatus, NcclError> {
            let count = buff.len();
            let (dst, _record) = buff.device_ptr_mut(&self.stream);
            unsafe {
                rccl!(ncclAllReduce(
                    dst as usize as *const c_void,
                    dst as usize as *mut c_void,
                    count,
                    T::as_nccl_type(),
                    op(reduce_op),
                    self.comm,
                    self.s()
                ))
            }
            .result()
        }

        /// Reduces `sendbuff` across members into the `recvbuff` of member `root`.
        pub fn reduce<S: DevicePtr<T>, R: DevicePtrMut<T>, T: NcclType>(
            &self,
            sendbuff: &S,
            recvbuff: Option<&mut R>,
            reduce_op: &ReduceOp,
            root: i32,
        ) -> Result<NcclStatus, NcclError> {
            let count = sendbuff.len();
            let (src, _record_src) = sendbuff.device_ptr(&self.stream);
            let (dst, _record_dst) = recvbuff.map(|b| b.device_ptr_mut(&self.stream)).unzip();
            unsafe {
                rccl!(ncclReduce(
                    src as usize as *const c_void,
                    dst.map_or(std::ptr::null_mut(), |p| p as usize as *mut c_void),
                    count,
                    T::as_nccl_type(),
                    op(reduce_op),
                    root,
                    self.comm,
                    self.s()
                ))
            }
            .result()
        }
    }
}
