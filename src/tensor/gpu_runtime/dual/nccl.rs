//! Collective communication over both runtimes: NCCL among NVIDIA cards, RCCL among AMD cards.
//! A communicator spans the cards of one runtime; cards of different runtimes exchange through
//! host memory instead.

pub mod sys {
    #![allow(non_camel_case_types, clippy::missing_safety_doc)]

    pub use crate::tensor::gpu_runtime::hip::nccl::sys::{
        ncclComm, ncclComm_t, ncclDataType_t, ncclRedOp_t, ncclResult_t,
    };

    /// Whether a runtime's collective library opens.
    pub unsafe fn is_culib_present() -> bool {
        unsafe {
            (crate::tensor::gpu_runtime::dual::cuda_usable()
                && cudarc::nccl::sys::is_culib_present())
                || crate::tensor::gpu_runtime::hip::nccl::sys::is_culib_present()
        }
    }
}

pub mod result {
    use super::sys;

    pub use crate::tensor::gpu_runtime::hip::nccl::result::{NcclError, NcclStatus};

    /// An NCCL outcome under its own number.
    pub(crate) fn outcome(
        r: Result<cudarc::nccl::result::NcclStatus, cudarc::nccl::result::NcclError>,
    ) -> Result<NcclStatus, NcclError> {
        use sys::ncclResult_t::*;
        match r {
            Ok(cudarc::nccl::result::NcclStatus::Success) => Ok(NcclStatus::Success),
            Ok(cudarc::nccl::result::NcclStatus::InProgress) => Ok(NcclStatus::InProgress),
            Ok(cudarc::nccl::result::NcclStatus::NumResults) => Ok(NcclStatus::NumResults),
            Err(e) => Err(NcclError(match e.0 as u32 {
                1 => ncclUnhandledCudaError,
                2 => ncclSystemError,
                3 => ncclInternalError,
                4 => ncclInvalidArgument,
                5 => ncclInvalidUsage,
                6 => ncclRemoteError,
                _ => ncclInternalError,
            })),
        }
    }

    /// Whether NCCL opens, read once: the library is not opened and closed at every group.
    pub(crate) fn nccl_present() -> bool {
        static PRESENT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *PRESENT.get_or_init(|| {
            crate::tensor::gpu_runtime::dual::cuda_usable()
                && unsafe { cudarc::nccl::sys::is_culib_present() }
        })
    }

    /// Opens a group of calls issued together, in every collective library that opens.
    pub fn group_start() -> Result<NcclStatus, NcclError> {
        let mut out = Ok(NcclStatus::Success);
        if nccl_present() {
            out = outcome(cudarc::nccl::result::group_start());
        }
        if unsafe { crate::tensor::gpu_runtime::hip::nccl::sys::is_culib_present() } {
            let rccl = crate::tensor::gpu_runtime::hip::nccl::result::group_start();
            // A group RCCL refuses closes the one NCCL opened, so neither stays open.
            if rccl.is_err() && out.is_ok() && nccl_present() {
                let _ = outcome(cudarc::nccl::result::group_end());
            }
            out = out.and(rccl);
        }
        out
    }

    /// Closes a group of calls and issues them, in every collective library that opens.
    pub fn group_end() -> Result<NcclStatus, NcclError> {
        let mut out = Ok(NcclStatus::Success);
        if nccl_present() {
            out = outcome(cudarc::nccl::result::group_end());
        }
        if unsafe { crate::tensor::gpu_runtime::hip::nccl::sys::is_culib_present() } {
            out = out.and(crate::tensor::gpu_runtime::hip::nccl::result::group_end());
        }
        out
    }
}

pub mod safe {
    use std::ffi::{c_int, c_void};
    use std::sync::Arc;

    use super::result::{outcome, NcclError, NcclStatus};
    use super::sys;
    use crate::tensor::gpu_runtime::dual::driver::sys::same;
    use crate::tensor::gpu_runtime::dual::driver::{
        CudaContext, CudaStream, DevicePtr, DevicePtrMut,
    };
    use crate::tensor::gpu_runtime::hip::nccl::sys::Rccl;
    use cudarc::nccl::result as cr;
    use cudarc::nccl::sys as cs;

    pub use super::result::{group_end, group_start};
    pub use crate::tensor::gpu_runtime::hip::nccl::safe::{NcclType, ReduceOp};

    /// The communicator handle of the library of the members' runtime.
    #[derive(Debug, Clone, Copy)]
    enum Handle {
        Cuda(cs::ncclComm_t),
        Hip(sys::ncclComm_t),
    }

    /// One card's member of a communicator, issuing on its stream.
    #[derive(Debug)]
    pub struct Comm {
        comm: Handle,
        stream: Arc<CudaStream>,
        rank: usize,
        world_size: usize,
    }

    unsafe impl Send for Comm {}
    unsafe impl Sync for Comm {}

    impl Drop for Comm {
        fn drop(&mut self) {
            match self.comm {
                Handle::Cuda(c) => {
                    let _ = unsafe { cr::comm_abort(c) };
                }
                Handle::Hip(c) => {
                    if let Some(f) = Rccl::get().and_then(|a| a.ncclCommAbort) {
                        let _ = unsafe { f(c) };
                    }
                }
            }
        }
    }

    /// Calls an RCCL entry point; a missing library or entry point is a system error.
    macro_rules! rccl {
        ($f:ident ( $( $a:expr ),* $(,)? )) => {
            match Rccl::get().and_then(|api| api.$f) {
                None => Err(NcclError(sys::ncclResult_t::ncclSystemError)),
                Some(f) => f($( $a ),*).result(),
            }
        };
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

    impl Comm {
        /// One member per stream, all in this process and on cards of one runtime, ranked in
        /// the order given.
        pub fn from_devices(streams: Vec<Arc<CudaStream>>) -> Result<Vec<Self>, NcclError> {
            let n = streams.len();
            let nvidia = streams.iter().all(|s| s.context().is_nvidia());
            let amd = streams.iter().all(|s| !s.context().is_nvidia());
            let handles: Vec<Handle> = if nvidia {
                let mut comms = vec![std::ptr::null_mut(); n];
                let ordinals: Vec<c_int> = streams
                    .iter()
                    .map(|s| s.context().ordinal() as c_int)
                    .collect();
                outcome(unsafe {
                    cr::comm_init_all(comms.as_mut_ptr(), n as c_int, ordinals.as_ptr())
                })?;
                // A communicator is used once its initialisation has finished: a collective
                // issued while one is still in progress finds it not ready.
                for &c in &comms {
                    let mut state = cs::ncclResult_t::ncclInProgress;
                    while state == cs::ncclResult_t::ncclInProgress {
                        unsafe { cs::ncclCommGetAsyncError(c, &mut state) };
                        if state == cs::ncclResult_t::ncclInProgress {
                            std::thread::yield_now();
                        }
                    }
                    if state != cs::ncclResult_t::ncclSuccess {
                        tracing::info!("an NCCL communicator did not initialise: {state:?}");
                        return Err(NcclError(sys::ncclResult_t::ncclInternalError));
                    }
                }
                comms.into_iter().map(Handle::Cuda).collect()
            } else if amd {
                let mut comms = vec![std::ptr::null_mut(); n];
                let t = crate::tensor::gpu_runtime::dual::topology();
                let ordinals: Vec<c_int> = streams
                    .iter()
                    .map(|s| (s.context().ordinal() - t.cuda) as c_int)
                    .collect();
                unsafe {
                    rccl!(ncclCommInitAll(
                        comms.as_mut_ptr(),
                        n as c_int,
                        ordinals.as_ptr()
                    ))
                }?;
                comms.into_iter().map(Handle::Hip).collect()
            } else {
                return Err(NcclError(sys::ncclResult_t::ncclInvalidUsage));
            };
            Ok(handles
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

        /// The member's card.
        pub fn context(&self) -> &Arc<CudaContext> {
            self.stream.context()
        }

        /// The member's card ordinal.
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
            let (ty, src) = (T::as_nccl_type(), src as usize as *const c_void);
            match self.comm {
                Handle::Cuda(c) => outcome(unsafe {
                    cr::send(
                        src,
                        data.len(),
                        same(ty),
                        peer,
                        c,
                        self.s() as cs::cudaStream_t,
                    )
                }),
                Handle::Hip(c) => unsafe {
                    rccl!(ncclSend(src, data.len(), ty, peer, c, self.s()))
                },
            }
        }

        /// Receives a device range from `peer`.
        pub fn recv<R: DevicePtrMut<T>, T: NcclType>(
            &self,
            buff: &mut R,
            peer: i32,
        ) -> Result<NcclStatus, NcclError> {
            let count = buff.len();
            let (dst, _record) = buff.device_ptr_mut(&self.stream);
            let (ty, dst) = (T::as_nccl_type(), dst as usize as *mut c_void);
            match self.comm {
                Handle::Cuda(c) => outcome(unsafe {
                    cr::recv(dst, count, same(ty), peer, c, self.s() as cs::cudaStream_t)
                }),
                Handle::Hip(c) => unsafe { rccl!(ncclRecv(dst, count, ty, peer, c, self.s())) },
            }
        }

        fn reduce_raw(
            &self,
            src: *const c_void,
            dst: *mut c_void,
            count: usize,
            ty: sys::ncclDataType_t,
            reduce_op: &ReduceOp,
            root: Option<i32>,
        ) -> Result<NcclStatus, NcclError> {
            let o = op(reduce_op);
            match (self.comm, root) {
                (Handle::Cuda(c), None) => outcome(unsafe {
                    cr::all_reduce(
                        src,
                        dst,
                        count,
                        same(ty),
                        same(o),
                        c,
                        self.s() as cs::cudaStream_t,
                    )
                }),
                (Handle::Cuda(c), Some(r)) => outcome(unsafe {
                    cr::reduce(
                        src,
                        dst,
                        count,
                        same(ty),
                        same(o),
                        r,
                        c,
                        self.s() as cs::cudaStream_t,
                    )
                }),
                (Handle::Hip(c), None) => unsafe {
                    rccl!(ncclAllReduce(src, dst, count, ty, o, c, self.s()))
                },
                (Handle::Hip(c), Some(r)) => unsafe {
                    rccl!(ncclReduce(src, dst, count, ty, o, r, c, self.s()))
                },
            }
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
            self.reduce_raw(
                src as usize as *const c_void,
                dst as usize as *mut c_void,
                count,
                T::as_nccl_type(),
                reduce_op,
                None,
            )
        }

        /// Reduces `buff` across members in place.
        pub fn all_reduce_in_place<R: DevicePtrMut<T>, T: NcclType>(
            &self,
            buff: &mut R,
            reduce_op: &ReduceOp,
        ) -> Result<NcclStatus, NcclError> {
            let count = buff.len();
            let (dst, _record) = buff.device_ptr_mut(&self.stream);
            self.reduce_raw(
                dst as usize as *const c_void,
                dst as usize as *mut c_void,
                count,
                T::as_nccl_type(),
                reduce_op,
                None,
            )
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
            self.reduce_raw(
                src as usize as *const c_void,
                dst.map_or(std::ptr::null_mut(), |p| p as usize as *mut c_void),
                count,
                T::as_nccl_type(),
                reduce_op,
                Some(root),
            )
        }
    }
}
