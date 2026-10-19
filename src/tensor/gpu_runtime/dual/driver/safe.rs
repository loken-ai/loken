//! The owning wrappers of the driver over both runtimes: a card's context, its streams and
//! events, device buffers and their views, loaded modules and kernel launches.
//!
//! A context, a stream, an event and a module wrap the object of the runtime that opened the
//! card; every operation on one binds that runtime on the calling thread. Buffers hold their
//! address and the runtime's owner that frees it; views, copies, launches and graphs are made
//! here from addresses, on the stream's runtime.

use std::ffi::{c_int, c_void};
use std::marker::PhantomData;
use std::ops::{Bound, RangeBounds};
use std::sync::Arc;

use super::super::{bind, topology, Runtime, HIP_STREAMS};
use super::result::{self, err, DriverError};
use super::sys::{self, cuh, res, same, CUresult};
use crate::tensor::gpu_runtime::hip::driver as hd;
use cudarc::driver as cd;

pub use hd::LaunchConfig;

/// The context of the runtime that opened a card.
#[derive(Debug)]
enum Ctx {
    Cuda(Arc<cd::CudaContext>),
    Hip(Arc<hd::CudaContext>),
}

/// A card and the runtime state the process holds on it.
#[derive(Debug)]
pub struct CudaContext {
    inner: Ctx,
    ordinal: usize,
}

impl PartialEq for CudaContext {
    fn eq(&self, other: &Self) -> bool {
        self.ordinal == other.ordinal
    }
}
impl Eq for CudaContext {}

/// An error of a `cudarc` call that returns its result.
fn cuda<T>(r: Result<T, cd::DriverError>) -> Result<T, DriverError> {
    r.map_err(err)
}

/// A `cudarc` error of the error the GPU paths carry.
fn to_cuda(e: DriverError) -> cd::DriverError {
    cd::DriverError(
        sys::to_cuda_result(e.0).unwrap_or(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN),
    )
}

impl CudaContext {
    /// Opens the card of `ordinal` and binds it to the calling thread.
    pub fn new(ordinal: usize) -> Result<Arc<Self>, DriverError> {
        let inner = match topology().locate(ordinal) {
            Some((Runtime::Cuda, local)) => Ctx::Cuda(cuda(cd::CudaContext::new(local))?),
            Some((Runtime::Hip, local)) => Ctx::Hip(hd::CudaContext::new(local)?),
            None => return Err(DriverError(CUresult::CUDA_ERROR_INVALID_DEVICE)),
        };
        let ctx = Arc::new(Self { inner, ordinal });
        ctx.mark();
        Ok(ctx)
    }

    /// The number of cards of both runtimes.
    pub fn device_count() -> Result<i32, DriverError> {
        Ok(topology().count() as i32)
    }

    /// The runtime that opened the card.
    pub(crate) fn runtime(&self) -> Runtime {
        match self.inner {
            Ctx::Cuda(_) => Runtime::Cuda,
            Ctx::Hip(_) => Runtime::Hip,
        }
    }

    /// Whether NVIDIA's runtime drives the card.
    pub fn is_nvidia(&self) -> bool {
        self.runtime() == Runtime::Cuda
    }

    /// Records the card's runtime as the one bound on the calling thread.
    #[inline]
    fn mark(&self) {
        bind(self.runtime());
    }

    /// The card's ordinal, over both runtimes.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// The card's name.
    pub fn name(&self) -> Result<String, DriverError> {
        match &self.inner {
            Ctx::Cuda(c) => cuda(c.name()),
            Ctx::Hip(c) => c.name(),
        }
    }

    /// The card's memory in bytes.
    pub fn total_mem(&self) -> Result<usize, DriverError> {
        match &self.inner {
            Ctx::Cuda(c) => cuda(c.total_mem()),
            Ctx::Hip(c) => c.total_mem(),
        }
    }

    /// The free and total memory of the card, in bytes.
    pub fn mem_get_info(&self) -> Result<(usize, usize), DriverError> {
        self.mark();
        match &self.inner {
            // PLACEMENT-EXEMPT: the runtime's reading itself; its callers read through `place/`.
            Ctx::Cuda(c) => cuda(c.mem_get_info()),
            // PLACEMENT-EXEMPT: the runtime's reading itself; its callers read through `place/`.
            Ctx::Hip(c) => c.mem_get_info(),
        }
    }

    /// The card handle: its ordinal over both runtimes, which every raw call taking a card
    /// routes by.
    pub fn cu_device(&self) -> sys::CUdevice {
        self.ordinal as sys::CUdevice
    }

    /// The card's primary context, null when the runtime has none.
    pub fn cu_ctx(&self) -> sys::CUcontext {
        match &self.inner {
            Ctx::Cuda(c) => cuh(c.cu_ctx()),
            Ctx::Hip(c) => c.cu_ctx(),
        }
    }

    /// Whether buffers are allocated in stream order.
    pub fn has_async_alloc(&self) -> bool {
        match &self.inner {
            Ctx::Cuda(c) => c.has_async_alloc(),
            Ctx::Hip(c) => c.has_async_alloc(),
        }
    }

    /// Makes this card current on the calling thread, and its runtime the bound one.
    pub fn bind_to_thread(&self) -> Result<(), DriverError> {
        self.mark();
        match &self.inner {
            Ctx::Cuda(c) => cuda(c.bind_to_thread()),
            Ctx::Hip(c) => c.bind_to_thread(),
        }
    }

    /// One attribute of the card.
    pub fn attribute(&self, attrib: sys::CUdevice_attribute) -> Result<i32, DriverError> {
        match &self.inner {
            // SAFETY: CUDA's attribute enum holds every number the CUDA-named attributes carry.
            Ctx::Cuda(c) => cuda(c.attribute(unsafe { same(attrib) })),
            Ctx::Hip(c) => c.attribute(attrib),
        }
    }

    /// Waits for all work on the card.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.mark();
        match &self.inner {
            Ctx::Cuda(c) => cuda(c.synchronize()),
            Ctx::Hip(c) => c.synchronize(),
        }
    }

    /// Makes host waits on this card block the thread instead of spinning.
    pub fn set_blocking_synchronize(&self) -> Result<(), DriverError> {
        self.mark();
        match &self.inner {
            Ctx::Cuda(c) => cuda(c.set_blocking_synchronize()),
            Ctx::Hip(c) => c.set_blocking_synchronize(),
        }
    }

    /// Whether more than one stream is in use.
    pub fn is_in_multi_stream_mode(&self) -> bool {
        match &self.inner {
            Ctx::Cuda(c) => c.is_in_multi_stream_mode(),
            Ctx::Hip(c) => c.is_in_multi_stream_mode(),
        }
    }

    /// Whether buffers record events for cross-stream ordering. Buffers made here record none.
    pub fn is_event_tracking(&self) -> bool {
        match &self.inner {
            Ctx::Cuda(c) => c.is_event_tracking(),
            Ctx::Hip(c) => c.is_event_tracking(),
        }
    }

    /// Whether the context orders streams on the caller's behalf.
    pub fn is_managing_stream_synchronization(&self) -> bool {
        match &self.inner {
            Ctx::Cuda(c) => c.is_managing_stream_synchronization(),
            Ctx::Hip(c) => c.is_managing_stream_synchronization(),
        }
    }

    /// Turns on the runtime's own per-buffer event tracking.
    ///
    /// # Safety
    /// As the runtime's.
    pub unsafe fn enable_event_tracking(&self) {
        match &self.inner {
            Ctx::Cuda(c) => unsafe { c.enable_event_tracking() },
            Ctx::Hip(c) => unsafe { c.enable_event_tracking() },
        }
    }

    /// Turns off the runtime's own per-buffer event tracking.
    ///
    /// # Safety
    /// As the runtime's.
    pub unsafe fn disable_event_tracking(&self) {
        match &self.inner {
            Ctx::Cuda(c) => unsafe { c.disable_event_tracking() },
            Ctx::Hip(c) => unsafe { c.disable_event_tracking() },
        }
    }

    /// The first error a deferred call recorded, cleared by reading it.
    pub fn check_err(&self) -> Result<(), DriverError> {
        match &self.inner {
            Ctx::Cuda(c) => cuda(c.check_err()),
            Ctx::Hip(c) => c.check_err(),
        }
    }

    /// Keeps the error of a call that cannot return it, for the next [Self::check_err].
    pub fn record_err<T>(&self, result: Result<T, DriverError>) {
        match &self.inner {
            Ctx::Cuda(c) => c.record_err(result.map_err(to_cuda)),
            Ctx::Hip(c) => c.record_err(result),
        }
    }
}

/// The event of the runtime that owns it.
#[derive(Debug)]
enum Ev {
    Cuda(cd::CudaEvent),
    Hip(hd::CudaEvent),
}

/// A point in a stream's work.
#[derive(Debug)]
pub struct CudaEvent {
    inner: Ev,
    ctx: Arc<CudaContext>,
}

impl CudaContext {
    /// A new event with no work recorded; without `flags` it records no timing.
    pub fn new_event(
        self: &Arc<Self>,
        flags: Option<sys::CUevent_flags>,
    ) -> Result<CudaEvent, DriverError> {
        self.mark();
        let inner = match &self.inner {
            // SAFETY: CUDA's event flags hold every number the CUDA-named flags carry.
            Ctx::Cuda(c) => Ev::Cuda(cuda(c.new_event(flags.map(|f| unsafe { same(f) })))?),
            Ctx::Hip(c) => Ev::Hip(c.new_event(flags)?),
        };
        Ok(CudaEvent {
            inner,
            ctx: self.clone(),
        })
    }
}

impl CudaEvent {
    /// The event handle.
    pub fn cu_event(&self) -> sys::CUevent {
        match &self.inner {
            Ev::Cuda(e) => cuh(e.cu_event()),
            Ev::Hip(e) => e.cu_event(),
        }
    }

    /// The card the event belongs to.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Records the work queued on `stream` so far.
    pub fn record(&self, stream: &CudaStream) -> Result<(), DriverError> {
        self.ctx.mark();
        match (&self.inner, &stream.inner) {
            (Ev::Cuda(e), St::Cuda(s)) => cuda(e.record(s)),
            (Ev::Hip(e), St::Hip(s)) => e.record(s),
            _ => Err(DriverError(CUresult::CUDA_ERROR_INVALID_CONTEXT)),
        }
    }

    /// Waits on the host for the recorded work.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.ctx.mark();
        match &self.inner {
            Ev::Cuda(e) => cuda(e.synchronize()),
            Ev::Hip(e) => e.synchronize(),
        }
    }

    /// Milliseconds between this event and `end`; both must record timing.
    pub fn elapsed_ms(&self, end: &Self) -> Result<f32, DriverError> {
        self.ctx.mark();
        match (&self.inner, &end.inner) {
            (Ev::Cuda(a), Ev::Cuda(b)) => cuda(a.elapsed_ms(b)),
            (Ev::Hip(a), Ev::Hip(b)) => a.elapsed_ms(b),
            _ => Err(DriverError(CUresult::CUDA_ERROR_INVALID_CONTEXT)),
        }
    }

    /// Whether the recorded work is done; false on error.
    pub fn is_complete(&self) -> bool {
        self.try_is_complete().unwrap_or(false)
    }

    /// Whether the recorded work is done.
    pub fn try_is_complete(&self) -> Result<bool, DriverError> {
        match &self.inner {
            Ev::Cuda(e) => cuda(e.try_is_complete()),
            Ev::Hip(e) => e.try_is_complete(),
        }
    }
}

/// The stream of the runtime that owns it.
#[derive(Debug)]
enum St {
    Cuda(Arc<cd::CudaStream>),
    Hip(Arc<hd::CudaStream>),
}

/// A queue of work on a card.
#[derive(Debug)]
pub struct CudaStream {
    inner: St,
    ctx: Arc<CudaContext>,
}

impl PartialEq for CudaStream {
    fn eq(&self, other: &Self) -> bool {
        self.cu_stream() == other.cu_stream() && self.ctx == other.ctx
    }
}
impl Eq for CudaStream {}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if let St::Hip(s) = &self.inner {
            if !s.cu_stream().is_null() {
                HIP_STREAMS.remove(s.cu_stream());
            }
        }
    }
}

impl CudaContext {
    fn wrap_stream(self: &Arc<Self>, inner: St) -> Arc<CudaStream> {
        if let St::Hip(s) = &inner {
            if !s.cu_stream().is_null() {
                HIP_STREAMS.insert(s.cu_stream());
            }
        }
        Arc::new(CudaStream {
            inner,
            ctx: self.clone(),
        })
    }

    /// The card's default stream.
    pub fn default_stream(self: &Arc<Self>) -> Arc<CudaStream> {
        let inner = match &self.inner {
            Ctx::Cuda(c) => St::Cuda(c.default_stream()),
            Ctx::Hip(c) => St::Hip(c.default_stream()),
        };
        self.wrap_stream(inner)
    }

    /// A new stream that does not wait on the default stream.
    pub fn new_stream(self: &Arc<Self>) -> Result<Arc<CudaStream>, DriverError> {
        self.mark();
        let inner = match &self.inner {
            Ctx::Cuda(c) => St::Cuda(cuda(c.new_stream())?),
            Ctx::Hip(c) => St::Hip(c.new_stream()?),
        };
        Ok(self.wrap_stream(inner))
    }
}

impl CudaStream {
    /// A new stream on the same card that starts after the work queued here so far.
    pub fn fork(&self) -> Result<Arc<Self>, DriverError> {
        let stream = self.ctx.new_stream()?;
        stream.join(self)?;
        Ok(stream)
    }

    /// The stream handle.
    pub fn cu_stream(&self) -> sys::CUstream {
        match &self.inner {
            St::Cuda(s) => cuh(s.cu_stream()),
            St::Hip(s) => s.cu_stream(),
        }
    }

    /// The card the stream belongs to.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// The CUDA stream, for an NVIDIA card's stream.
    pub(crate) fn cuda_stream(&self) -> Option<&Arc<cd::CudaStream>> {
        match &self.inner {
            St::Cuda(s) => Some(s),
            St::Hip(_) => None,
        }
    }

    /// The HIP stream, for an AMD card's stream.
    pub(crate) fn hip_stream(&self) -> Option<&Arc<hd::CudaStream>> {
        match &self.inner {
            St::Hip(s) => Some(s),
            St::Cuda(_) => None,
        }
    }

    /// Binds the stream's card on the calling thread.
    #[inline]
    fn bind(&self) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()
    }

    /// Waits on the host for the stream's work.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.ctx.mark();
        match &self.inner {
            St::Cuda(s) => cuda(s.synchronize()),
            St::Hip(s) => s.synchronize(),
        }
    }

    /// A new event recording the stream's work so far.
    pub fn record_event(
        &self,
        flags: Option<sys::CUevent_flags>,
    ) -> Result<CudaEvent, DriverError> {
        let event = self.ctx.new_event(flags)?;
        event.record(self)?;
        Ok(event)
    }

    /// Makes later work on this stream wait for the work `event` recorded.
    pub fn wait(&self, event: &CudaEvent) -> Result<(), DriverError> {
        self.ctx.mark();
        match (&self.inner, &event.inner) {
            (St::Cuda(s), Ev::Cuda(e)) => cuda(s.wait(e)),
            (St::Hip(s), Ev::Hip(e)) => s.wait(e),
            // An event of the other runtime is waited for on the host.
            _ => event.synchronize(),
        }
    }

    /// Makes later work on this stream wait for the work queued on `other` so far.
    pub fn join(&self, other: &CudaStream) -> Result<(), DriverError> {
        self.wait(&other.record_event(None)?)
    }

    /// Starts capturing the stream's work into a graph.
    pub fn begin_capture(&self, mode: sys::CUstreamCaptureMode) -> Result<(), DriverError> {
        self.bind()?;
        unsafe { result::stream::begin_capture(self.cu_stream(), mode) }
    }

    /// Ends the capture; `None` when nothing was captured.
    pub fn end_capture(
        self: &Arc<Self>,
        flags: sys::CUgraphInstantiate_flags,
    ) -> Result<Option<CudaGraph>, DriverError> {
        self.bind()?;
        let cu_graph = unsafe { result::stream::end_capture(self.cu_stream()) }?;
        if cu_graph.is_null() {
            return Ok(None);
        }
        let cu_graph_exec = match unsafe { result::graph::instantiate(cu_graph, flags) } {
            Ok(exec) => exec,
            Err(e) => {
                self.ctx
                    .record_err(unsafe { result::graph::destroy(cu_graph) });
                return Err(e);
            }
        };
        Ok(Some(CudaGraph {
            cu_graph,
            cu_graph_exec,
            stream: self.clone(),
        }))
    }

    /// The stream's capture state.
    pub fn capture_status(&self) -> Result<sys::CUstreamCaptureStatus, DriverError> {
        self.bind()?;
        unsafe { result::stream::is_capturing(self.cu_stream()) }
    }
}

/// A captured graph and its executable.
#[derive(Debug)]
pub struct CudaGraph {
    cu_graph: sys::CUgraph,
    cu_graph_exec: sys::CUgraphExec,
    stream: Arc<CudaStream>,
}

unsafe impl Send for CudaGraph {}
unsafe impl Sync for CudaGraph {}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        let ctx = &self.stream.ctx;
        ctx.record_err(ctx.bind_to_thread());
        ctx.record_err(unsafe { result::graph::exec_destroy(self.cu_graph_exec) });
        ctx.record_err(unsafe { result::graph::destroy(self.cu_graph) });
    }
}

impl CudaGraph {
    /// Runs the graph on the stream it was captured from.
    pub fn launch(&self) -> Result<(), DriverError> {
        self.stream.bind()?;
        unsafe { result::graph::launch(self.cu_graph_exec, self.stream.cu_stream()) }
    }

    /// Uploads the graph's resources ahead of its first launch.
    pub fn upload(&self) -> Result<(), DriverError> {
        self.stream.bind()?;
        unsafe { result::graph::upload(self.cu_graph_exec, self.stream.cu_stream()) }
    }

    /// The graph handle.
    pub fn cu_graph(&self) -> sys::CUgraph {
        self.cu_graph
    }

    /// The executable handle.
    pub fn cu_graph_exec(&self) -> sys::CUgraphExec {
        self.cu_graph_exec
    }
}

/// What frees a buffer: the runtime's own byte buffer at the same address.
#[derive(Debug)]
enum Owner {
    None,
    Cuda(cd::CudaSlice<u8>),
    Hip(hd::CudaSlice<u8>),
}

impl Owner {
    /// Gives up the memory without freeing it.
    fn leak(self) {
        match self {
            Owner::None => {}
            Owner::Cuda(s) => {
                s.leak();
            }
            Owner::Hip(s) => {
                s.leak();
            }
        }
    }
}

/// An array of `T` in device memory, freed on drop.
#[derive(Debug)]
pub struct CudaSlice<T> {
    pub(crate) cu_device_ptr: sys::CUdeviceptr,
    pub(crate) len: usize,
    pub(crate) stream: Arc<CudaStream>,
    owner: Owner,
    marker: PhantomData<*const T>,
}

unsafe impl<T> Send for CudaSlice<T> {}
unsafe impl<T> Sync for CudaSlice<T> {}

impl<T> Drop for CudaSlice<T> {
    fn drop(&mut self) {
        if !matches!(self.owner, Owner::None) {
            self.stream.ctx.mark();
        }
    }
}

impl<T> CudaSlice<T> {
    /// The number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// The size in bytes.
    pub fn num_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Whether the array has no element.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The card's ordinal.
    pub fn ordinal(&self) -> usize {
        self.stream.ctx.ordinal
    }

    /// The card the array lives on.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.stream.ctx
    }

    /// The stream the array is ordered on.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Gives up ownership: the memory is no longer freed, and its address is returned.
    pub fn leak(mut self) -> sys::CUdeviceptr {
        let ptr = self.cu_device_ptr;
        std::mem::replace(&mut self.owner, Owner::None).leak();
        ptr
    }
}

impl<T: DeviceRepr> CudaSlice<T> {
    /// A device copy of the array.
    pub fn try_clone(&self) -> Result<Self, DriverError> {
        self.stream.clone_dtod(self)
    }
}

impl<T: DeviceRepr> Clone for CudaSlice<T> {
    fn clone(&self) -> Self {
        self.try_clone().expect("device copy of a buffer")
    }
}

impl<T: Clone + Default + DeviceRepr> TryFrom<CudaSlice<T>> for Vec<T> {
    type Error = DriverError;
    fn try_from(value: CudaSlice<T>) -> Result<Self, Self::Error> {
        value.stream.clone_dtoh(&value)
    }
}

/// A borrowed, read-only range of a device array.
#[derive(Debug)]
pub struct CudaView<'a, T> {
    pub(crate) ptr: sys::CUdeviceptr,
    pub(crate) len: usize,
    pub(crate) stream: &'a Arc<CudaStream>,
    marker: PhantomData<&'a [T]>,
}

/// A borrowed, writable range of a device array.
#[derive(Debug)]
pub struct CudaViewMut<'a, T> {
    pub(crate) ptr: sys::CUdeviceptr,
    pub(crate) len: usize,
    pub(crate) stream: &'a Arc<CudaStream>,
    marker: PhantomData<&'a mut [T]>,
}

fn bounds(range: impl RangeBounds<usize>, len: usize) -> Option<(usize, usize)> {
    let start = match range.start_bound() {
        Bound::Included(&s) => s,
        Bound::Excluded(&s) => s.checked_add(1)?,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&e) => e.checked_add(1)?,
        Bound::Excluded(&e) => e,
        Bound::Unbounded => len,
    };
    (start <= end && end <= len).then_some((start, end))
}

fn offset<T>(ptr: sys::CUdeviceptr, elems: usize) -> sys::CUdeviceptr {
    ptr + (elems * std::mem::size_of::<T>()) as u64
}

impl<T> CudaSlice<T> {
    /// The whole array as a view.
    pub fn as_view(&self) -> CudaView<'_, T> {
        CudaView {
            ptr: self.cu_device_ptr,
            len: self.len,
            stream: &self.stream,
            marker: PhantomData,
        }
    }

    /// The whole array as a writable view.
    pub fn as_view_mut(&mut self) -> CudaViewMut<'_, T> {
        CudaViewMut {
            ptr: self.cu_device_ptr,
            len: self.len,
            stream: &self.stream,
            marker: PhantomData,
        }
    }

    /// A range of the array; panics when out of bounds.
    pub fn slice(&self, range: impl RangeBounds<usize>) -> CudaView<'_, T> {
        self.try_slice(range).expect("slice out of bounds")
    }

    /// A range of the array, `None` when out of bounds.
    pub fn try_slice(&self, range: impl RangeBounds<usize>) -> Option<CudaView<'_, T>> {
        let (s, e) = bounds(range, self.len)?;
        Some(CudaView {
            ptr: offset::<T>(self.cu_device_ptr, s),
            len: e - s,
            stream: &self.stream,
            marker: PhantomData,
        })
    }

    /// A writable range of the array; panics when out of bounds.
    pub fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> CudaViewMut<'_, T> {
        self.try_slice_mut(range).expect("slice out of bounds")
    }

    /// A writable range of the array, `None` when out of bounds.
    pub fn try_slice_mut(&mut self, range: impl RangeBounds<usize>) -> Option<CudaViewMut<'_, T>> {
        let (s, e) = bounds(range, self.len)?;
        Some(CudaViewMut {
            ptr: offset::<T>(self.cu_device_ptr, s),
            len: e - s,
            stream: &self.stream,
            marker: PhantomData,
        })
    }

    /// The array's first bytes read as `len` elements of `S`, `None` when they do not fit.
    ///
    /// # Safety
    /// The bytes must be valid values of `S`.
    pub unsafe fn transmute<S>(&self, len: usize) -> Option<CudaView<'_, S>> {
        (len * std::mem::size_of::<S>() <= self.num_bytes()).then_some(CudaView {
            ptr: self.cu_device_ptr,
            len,
            stream: &self.stream,
            marker: PhantomData,
        })
    }

    /// The array's first bytes written as `len` elements of `S`, `None` when they do not fit.
    ///
    /// # Safety
    /// The bytes must be valid values of `S`.
    pub unsafe fn transmute_mut<S>(&mut self, len: usize) -> Option<CudaViewMut<'_, S>> {
        (len * std::mem::size_of::<S>() <= self.num_bytes()).then_some(CudaViewMut {
            ptr: self.cu_device_ptr,
            len,
            stream: &self.stream,
            marker: PhantomData,
        })
    }

    /// The array split in two views at `mid`; panics past the end.
    pub fn split_at(&self, mid: usize) -> (CudaView<'_, T>, CudaView<'_, T>) {
        self.as_view().split_at(mid)
    }

    /// The array split in two writable views at `mid`; panics past the end.
    pub fn split_at_mut(&mut self, mid: usize) -> (CudaViewMut<'_, T>, CudaViewMut<'_, T>) {
        assert!(mid <= self.len, "split past the end");
        let stream = &self.stream;
        (
            CudaViewMut {
                ptr: self.cu_device_ptr,
                len: mid,
                stream,
                marker: PhantomData,
            },
            CudaViewMut {
                ptr: offset::<T>(self.cu_device_ptr, mid),
                len: self.len - mid,
                stream,
                marker: PhantomData,
            },
        )
    }
}

impl<'a, T> CudaView<'a, T> {
    /// The number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the view has no element.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The stream the underlying array is ordered on.
    pub fn stream(&self) -> &Arc<CudaStream> {
        self.stream
    }

    /// A range of the view; panics when out of bounds.
    pub fn slice(&self, range: impl RangeBounds<usize>) -> Self {
        self.try_slice(range).expect("slice out of bounds")
    }

    /// A range of the view, `None` when out of bounds.
    pub fn try_slice(&self, range: impl RangeBounds<usize>) -> Option<Self> {
        let (s, e) = bounds(range, self.len)?;
        Some(Self {
            ptr: offset::<T>(self.ptr, s),
            len: e - s,
            stream: self.stream,
            marker: PhantomData,
        })
    }

    /// The view's bytes read as `len` elements of `S`, `None` when they do not fit.
    ///
    /// # Safety
    /// The bytes must be valid values of `S`.
    pub unsafe fn transmute<S>(&self, len: usize) -> Option<CudaView<'a, S>> {
        (len * std::mem::size_of::<S>() <= self.len * std::mem::size_of::<T>()).then_some(
            CudaView {
                ptr: self.ptr,
                len,
                stream: self.stream,
                marker: PhantomData,
            },
        )
    }

    /// The view split in two at `mid`; panics past the end.
    pub fn split_at(&self, mid: usize) -> (Self, Self) {
        assert!(mid <= self.len, "split past the end");
        (self.slice(..mid), self.slice(mid..))
    }

    /// Consecutive views of `chunk_size` elements; a remainder is left out.
    pub fn chunks_exact(&self, chunk_size: usize) -> impl Iterator<Item = CudaView<'a, T>> + '_ {
        (0..self.len / chunk_size).map(move |i| CudaView {
            ptr: offset::<T>(self.ptr, i * chunk_size),
            len: chunk_size,
            stream: self.stream,
            marker: PhantomData,
        })
    }

    /// The view's address, for a read ordered on `stream`.
    pub fn view_ptr(self, _stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Record(None))
    }
}

impl<'a, T> CudaViewMut<'a, T> {
    /// The number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the view has no element.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The stream the underlying array is ordered on.
    pub fn stream(&self) -> &Arc<CudaStream> {
        self.stream
    }

    /// The view read-only.
    pub fn as_view<'b>(&'b self) -> CudaView<'b, T> {
        CudaView {
            ptr: self.ptr,
            len: self.len,
            stream: self.stream,
            marker: PhantomData,
        }
    }

    /// The view read-only, for the whole borrow.
    pub fn into_view(self) -> CudaView<'a, T> {
        CudaView {
            ptr: self.ptr,
            len: self.len,
            stream: self.stream,
            marker: PhantomData,
        }
    }

    /// A read-only range of the view; panics when out of bounds.
    pub fn slice<'b>(&'b self, range: impl RangeBounds<usize>) -> CudaView<'b, T> {
        self.as_view().slice(range)
    }

    /// A read-only range of the view, `None` when out of bounds.
    pub fn try_slice<'b>(&'b self, range: impl RangeBounds<usize>) -> Option<CudaView<'b, T>> {
        let (s, e) = bounds(range, self.len)?;
        Some(CudaView {
            ptr: offset::<T>(self.ptr, s),
            len: e - s,
            stream: self.stream,
            marker: PhantomData,
        })
    }

    /// A writable range of the view; panics when out of bounds.
    pub fn slice_mut<'b>(&'b mut self, range: impl RangeBounds<usize>) -> CudaViewMut<'b, T> {
        self.try_slice_mut(range).expect("slice out of bounds")
    }

    /// A writable range of the view, `None` when out of bounds.
    pub fn try_slice_mut<'b>(
        &'b mut self,
        range: impl RangeBounds<usize>,
    ) -> Option<CudaViewMut<'b, T>> {
        let (s, e) = bounds(range, self.len)?;
        Some(CudaViewMut {
            ptr: offset::<T>(self.ptr, s),
            len: e - s,
            stream: self.stream,
            marker: PhantomData,
        })
    }

    /// The view's bytes read as `len` elements of `S`, `None` when they do not fit.
    ///
    /// # Safety
    /// The bytes must be valid values of `S`.
    pub unsafe fn transmute<'b, S>(&'b self, len: usize) -> Option<CudaView<'b, S>> {
        (len * std::mem::size_of::<S>() <= self.len * std::mem::size_of::<T>()).then_some(
            CudaView {
                ptr: self.ptr,
                len,
                stream: self.stream,
                marker: PhantomData,
            },
        )
    }

    /// The view's bytes written as `len` elements of `S`, `None` when they do not fit.
    ///
    /// # Safety
    /// The bytes must be valid values of `S`.
    pub unsafe fn transmute_mut<'b, S>(&'b mut self, len: usize) -> Option<CudaViewMut<'b, S>> {
        (len * std::mem::size_of::<S>() <= self.len * std::mem::size_of::<T>()).then_some(
            CudaViewMut {
                ptr: self.ptr,
                len,
                stream: self.stream,
                marker: PhantomData,
            },
        )
    }

    /// The view split in two writable views at `mid`; panics past the end.
    pub fn split_at_mut<'b>(&'b mut self, mid: usize) -> (CudaViewMut<'b, T>, CudaViewMut<'b, T>) {
        assert!(mid <= self.len, "split past the end");
        (
            CudaViewMut {
                ptr: self.ptr,
                len: mid,
                stream: self.stream,
                marker: PhantomData,
            },
            CudaViewMut {
                ptr: offset::<T>(self.ptr, mid),
                len: self.len - mid,
                stream: self.stream,
                marker: PhantomData,
            },
        )
    }

    /// Consecutive writable views of `chunk_size` elements; a remainder is left out.
    pub fn chunks_exact_mut(self, chunk_size: usize) -> impl Iterator<Item = CudaViewMut<'a, T>> {
        let (ptr, stream) = (self.ptr, self.stream);
        (0..self.len / chunk_size).map(move |i| CudaViewMut {
            ptr: offset::<T>(ptr, i * chunk_size),
            len: chunk_size,
            stream,
            marker: PhantomData,
        })
    }

    /// The view's address, for a write ordered on `stream`.
    pub fn view_ptr_mut(self, _stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Record(None))
    }
}

pub use hd::{DeviceRepr, ValidAsZeroBits};

/// What has a length and an ordering stream on the device.
pub trait DeviceSlice<T> {
    fn len(&self) -> usize;
    fn num_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn stream(&self) -> &Arc<CudaStream>;
}

impl<T> DeviceSlice<T> for CudaSlice<T> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl<T> DeviceSlice<T> for CudaView<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        self.stream
    }
}

impl<T> DeviceSlice<T> for CudaViewMut<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        self.stream
    }
}

/// What to do once an operation on a borrowed address is queued.
#[derive(Debug)]
#[must_use]
pub enum SyncOnDrop<'a> {
    /// Record the stream's work in the event; nothing when `None`.
    Record(Option<(&'a CudaEvent, &'a CudaStream)>),
    /// Wait on the host for the stream; nothing when `None`.
    Sync(Option<&'a CudaStream>),
}

impl<'a> SyncOnDrop<'a> {
    /// Records `stream`'s work in `event` on drop, when there is an event.
    pub fn record_event(event: &'a Option<CudaEvent>, stream: &'a CudaStream) -> Self {
        SyncOnDrop::Record(event.as_ref().map(|e| (e, stream)))
    }

    /// Waits for `stream` on drop.
    pub fn sync_stream(stream: &'a CudaStream) -> Self {
        SyncOnDrop::Sync(Some(stream))
    }
}

impl Drop for SyncOnDrop<'_> {
    fn drop(&mut self) {
        match self {
            SyncOnDrop::Record(target) => {
                if let Some((event, stream)) = target.take() {
                    stream.ctx.record_err(event.record(stream));
                }
            }
            SyncOnDrop::Sync(target) => {
                if let Some(stream) = target.take() {
                    stream.ctx.record_err(stream.synchronize());
                }
            }
        }
    }
}

/// A device range that can be read.
pub trait DevicePtr<T>: DeviceSlice<T> {
    /// The address, for a read ordered on `stream`.
    fn device_ptr<'a>(&'a self, stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>);
}

/// A device range that can be written.
pub trait DevicePtrMut<T>: DeviceSlice<T> {
    /// The address, for a write ordered on `stream`.
    fn device_ptr_mut<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (sys::CUdeviceptr, SyncOnDrop<'a>);
}

impl<T> DevicePtr<T> for CudaSlice<T> {
    fn device_ptr<'a>(&'a self, _stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.cu_device_ptr, SyncOnDrop::Record(None))
    }
}

impl<T> DevicePtr<T> for CudaView<'_, T> {
    fn device_ptr<'a>(&'a self, _stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Record(None))
    }
}

impl<T> DevicePtr<T> for CudaViewMut<'_, T> {
    fn device_ptr<'a>(&'a self, _stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Record(None))
    }
}

impl<T> DevicePtrMut<T> for CudaSlice<T> {
    fn device_ptr_mut<'a>(
        &'a mut self,
        _stream: &'a CudaStream,
    ) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.cu_device_ptr, SyncOnDrop::Record(None))
    }
}

impl<T> DevicePtrMut<T> for CudaViewMut<'_, T> {
    fn device_ptr_mut<'a>(
        &'a mut self,
        _stream: &'a CudaStream,
    ) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Record(None))
    }
}

/// Host memory a copy can read or write.
pub trait HostSlice<T> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The memory to read, and what to do once the copy is queued.
    ///
    /// # Safety
    /// The slice must be read only by the copy queued on `stream`.
    unsafe fn stream_synced_slice<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> (&'a [T], SyncOnDrop<'a>);
    /// The memory to write, and what to do once the copy is queued.
    ///
    /// # Safety
    /// The slice must be written only by the copy queued on `stream`.
    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (&'a mut [T], SyncOnDrop<'a>);
}

impl<T> HostSlice<T> for [T] {
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
    unsafe fn stream_synced_slice<'a>(
        &'a self,
        _stream: &'a CudaStream,
    ) -> (&'a [T], SyncOnDrop<'a>) {
        (self, SyncOnDrop::Sync(None))
    }
    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (&'a mut [T], SyncOnDrop<'a>) {
        (self, SyncOnDrop::sync_stream(stream))
    }
}

impl<T, const N: usize> HostSlice<T> for [T; N] {
    fn len(&self) -> usize {
        N
    }
    unsafe fn stream_synced_slice<'a>(
        &'a self,
        _stream: &'a CudaStream,
    ) -> (&'a [T], SyncOnDrop<'a>) {
        (self, SyncOnDrop::Sync(None))
    }
    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (&'a mut [T], SyncOnDrop<'a>) {
        (self, SyncOnDrop::sync_stream(stream))
    }
}

impl<T> HostSlice<T> for Vec<T> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
    unsafe fn stream_synced_slice<'a>(
        &'a self,
        _stream: &'a CudaStream,
    ) -> (&'a [T], SyncOnDrop<'a>) {
        (self, SyncOnDrop::Sync(None))
    }
    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (&'a mut [T], SyncOnDrop<'a>) {
        (self, SyncOnDrop::sync_stream(stream))
    }
}

fn check_len(a: usize, b: usize) -> Result<(), DriverError> {
    if a == b {
        Ok(())
    } else {
        Err(DriverError(CUresult::CUDA_ERROR_INVALID_VALUE))
    }
}

impl CudaStream {
    /// An empty array ordered on this stream.
    pub fn null<T>(self: &Arc<Self>) -> Result<CudaSlice<T>, DriverError> {
        Ok(CudaSlice {
            cu_device_ptr: 0,
            len: 0,
            stream: self.clone(),
            owner: Owner::None,
            marker: PhantomData,
        })
    }

    /// An uninitialized array of `len` elements, allocated by the stream's runtime.
    ///
    /// # Safety
    /// The contents must be written before they are read.
    pub unsafe fn alloc<T: DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError> {
        let bytes = len * std::mem::size_of::<T>();
        if bytes == 0 {
            return self.null().map(|mut s| {
                s.len = len;
                s
            });
        }
        self.ctx.mark();
        let (cu_device_ptr, owner) = match &self.inner {
            St::Cuda(s) => {
                let b = cuda(unsafe { s.alloc::<u8>(bytes) })?;
                let (p, _) = cd::DevicePtr::device_ptr(&b, s);
                (p, Owner::Cuda(b))
            }
            St::Hip(s) => {
                let b = unsafe { s.alloc::<u8>(bytes) }?;
                (b.cu_device_ptr, Owner::Hip(b))
            }
        };
        Ok(CudaSlice {
            cu_device_ptr,
            len,
            stream: self.clone(),
            owner,
            marker: PhantomData,
        })
    }

    /// A zeroed array of `len` elements.
    pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError> {
        let mut dst = unsafe { self.alloc(len) }?;
        self.memset_zeros(&mut dst)?;
        Ok(dst)
    }

    /// Zeroes a device range.
    pub fn memset_zeros<T: DeviceRepr + ValidAsZeroBits, Dst: DevicePtrMut<T>>(
        self: &Arc<Self>,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        let bytes = dst.num_bytes();
        if bytes == 0 {
            return Ok(());
        }
        self.bind()?;
        let (ptr, _record) = dst.device_ptr_mut(self);
        self.raw_memset(ptr, 0, bytes).result()
    }

    fn raw_memset(&self, ptr: sys::CUdeviceptr, value: u8, bytes: usize) -> CUresult {
        match &self.inner {
            St::Cuda(s) => res(unsafe {
                cudarc::driver::sys::cuMemsetD8Async(ptr, value, bytes, s.cu_stream())
            }),
            St::Hip(s) => unsafe {
                crate::tensor::gpu_runtime::hip::driver::sys::cuMemsetD8Async(
                    ptr,
                    value,
                    bytes,
                    s.cu_stream(),
                )
            },
        }
    }

    fn raw_htod(&self, dst: sys::CUdeviceptr, src: *const c_void, bytes: usize) -> CUresult {
        match &self.inner {
            St::Cuda(s) => res(unsafe {
                cudarc::driver::sys::cuMemcpyHtoDAsync_v2(dst, src, bytes, s.cu_stream())
            }),
            St::Hip(s) => unsafe {
                crate::tensor::gpu_runtime::hip::driver::sys::cuMemcpyHtoDAsync_v2(
                    dst,
                    src,
                    bytes,
                    s.cu_stream(),
                )
            },
        }
    }

    fn raw_dtoh(&self, dst: *mut c_void, src: sys::CUdeviceptr, bytes: usize) -> CUresult {
        match &self.inner {
            St::Cuda(s) => res(unsafe {
                cudarc::driver::sys::cuMemcpyDtoHAsync_v2(dst, src, bytes, s.cu_stream())
            }),
            St::Hip(s) => unsafe {
                crate::tensor::gpu_runtime::hip::driver::sys::cuMemcpyDtoHAsync_v2(
                    dst,
                    src,
                    bytes,
                    s.cu_stream(),
                )
            },
        }
    }

    fn raw_dtod(&self, dst: sys::CUdeviceptr, src: sys::CUdeviceptr, bytes: usize) -> CUresult {
        match &self.inner {
            St::Cuda(s) => res(unsafe {
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(dst, src, bytes, s.cu_stream())
            }),
            St::Hip(s) => unsafe {
                crate::tensor::gpu_runtime::hip::driver::sys::cuMemcpyDtoDAsync_v2(
                    dst,
                    src,
                    bytes,
                    s.cu_stream(),
                )
            },
        }
    }

    /// A device array holding a copy of host memory.
    pub fn memcpy_stod<T: DeviceRepr, Src: HostSlice<T> + ?Sized>(
        self: &Arc<Self>,
        src: &Src,
    ) -> Result<CudaSlice<T>, DriverError> {
        let mut dst = unsafe { self.alloc(src.len()) }?;
        self.memcpy_htod(src, &mut dst)?;
        Ok(dst)
    }

    /// A device array holding a copy of host memory.
    pub fn clone_htod<T: DeviceRepr, Src: HostSlice<T> + ?Sized>(
        self: &Arc<Self>,
        src: &Src,
    ) -> Result<CudaSlice<T>, DriverError> {
        self.memcpy_stod(src)
    }

    /// Copies host memory into a device range of the same length.
    pub fn memcpy_htod<T: DeviceRepr, Src: HostSlice<T> + ?Sized, Dst: DevicePtrMut<T>>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        check_len(src.len(), dst.len())?;
        if src.is_empty() {
            return Ok(());
        }
        self.bind()?;
        let (src, _record_src) = unsafe { src.stream_synced_slice(self) };
        let (dst, _record_dst) = dst.device_ptr_mut(self);
        self.raw_htod(
            dst,
            src.as_ptr() as *const c_void,
            std::mem::size_of_val(src),
        )
        .result()
    }

    /// A host vector holding a copy of a device range; waits for the copy.
    pub fn memcpy_dtov<T: DeviceRepr, Src: DevicePtr<T>>(
        self: &Arc<Self>,
        src: &Src,
    ) -> Result<Vec<T>, DriverError> {
        let mut dst = Vec::with_capacity(src.len());
        // The pages are written once on the host before the copy: the device does not fault in
        // pages the host has never touched. SAFETY: the copy below writes every element before
        // the vector is returned.
        #[allow(clippy::uninit_vec)]
        unsafe {
            std::ptr::write_bytes(dst.as_mut_ptr(), 0, src.len());
            dst.set_len(src.len())
        };
        self.memcpy_dtoh(src, &mut dst)?;
        Ok(dst)
    }

    /// A host vector holding a copy of a device range; waits for the copy.
    pub fn clone_dtoh<T: DeviceRepr, Src: DevicePtr<T>>(
        self: &Arc<Self>,
        src: &Src,
    ) -> Result<Vec<T>, DriverError> {
        self.memcpy_dtov(src)
    }

    /// Copies a device range into host memory of the same length.
    pub fn memcpy_dtoh<T: DeviceRepr, Src: DevicePtr<T>, Dst: HostSlice<T> + ?Sized>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        check_len(src.len(), dst.len())?;
        if src.is_empty() {
            return Ok(());
        }
        self.bind()?;
        let (src, _record_src) = src.device_ptr(self);
        let (dst, _record_dst) = unsafe { dst.stream_synced_mut_slice(self) };
        self.raw_dtoh(
            dst.as_mut_ptr() as *mut c_void,
            src,
            std::mem::size_of_val(dst),
        )
        .result()
    }

    /// Copies a device range into another of the same length. Across cards of different
    /// runtimes the copy goes through host memory and is waited for.
    pub fn memcpy_dtod<T, Src: DevicePtr<T>, Dst: DevicePtrMut<T>>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        check_len(src.len(), dst.len())?;
        let bytes = src.num_bytes();
        if bytes == 0 {
            return Ok(());
        }
        let src_stream = src.stream().clone();
        if src_stream.ctx.runtime() != self.ctx.runtime() {
            let mut host = vec![0u8; bytes];
            src_stream.bind()?;
            let (s, _record_src) = src.device_ptr(&src_stream);
            src_stream
                .raw_dtoh(host.as_mut_ptr() as *mut c_void, s, bytes)
                .result()?;
            src_stream.synchronize()?;
            self.bind()?;
            let (d, _record_dst) = dst.device_ptr_mut(self);
            self.raw_htod(d, host.as_ptr() as *const c_void, bytes)
                .result()?;
            return self.synchronize();
        }
        self.bind()?;
        let (src, _record_src) = src.device_ptr(self);
        let (dst, _record_dst) = dst.device_ptr_mut(self);
        self.raw_dtod(dst, src, bytes).result()
    }

    /// A new device array holding a copy of a device range.
    pub fn clone_dtod<T: DeviceRepr, Src: DevicePtr<T>>(
        self: &Arc<Self>,
        src: &Src,
    ) -> Result<CudaSlice<T>, DriverError> {
        let mut dst = unsafe { self.alloc(src.len()) }?;
        self.memcpy_dtod(src, &mut dst)?;
        Ok(dst)
    }

    /// Takes ownership of device memory allocated elsewhere, freed on drop by the stream's
    /// runtime.
    ///
    /// # Safety
    /// `cu_device_ptr` must hold `len` elements of `T` and be freed by nothing else.
    pub unsafe fn upgrade_device_ptr<T>(
        self: &Arc<Self>,
        cu_device_ptr: sys::CUdeviceptr,
        len: usize,
    ) -> CudaSlice<T> {
        let bytes = len * std::mem::size_of::<T>();
        let owner = if cu_device_ptr == 0 {
            Owner::None
        } else {
            match &self.inner {
                St::Cuda(s) => {
                    Owner::Cuda(unsafe { s.upgrade_device_ptr::<u8>(cu_device_ptr, bytes) })
                }
                St::Hip(s) => {
                    Owner::Hip(unsafe { s.upgrade_device_ptr::<u8>(cu_device_ptr, bytes) })
                }
            }
        };
        CudaSlice {
            cu_device_ptr,
            len,
            stream: self.clone(),
            owner,
            marker: PhantomData,
        }
    }
}

/// The module of the runtime that loaded it.
#[derive(Debug)]
enum Md {
    Cuda(Arc<cd::CudaModule>),
    Hip(Arc<hd::CudaModule>),
}

/// A loaded module.
#[derive(Debug)]
pub struct CudaModule {
    inner: Md,
    ctx: Arc<CudaContext>,
}

impl CudaContext {
    /// Loads compiled code onto the card: PTX or a cubin for an NVIDIA card, a code object for
    /// an AMD card.
    pub fn load_module(
        self: &Arc<Self>,
        ptx: crate::tensor::gpu_runtime::dual::nvrtc::Ptx,
    ) -> Result<Arc<CudaModule>, DriverError> {
        self.mark();
        let inner = match &self.inner {
            Ctx::Cuda(c) => Md::Cuda(cuda(c.load_module(ptx.into_cuda()))?),
            Ctx::Hip(c) => Md::Hip(c.load_module(ptx.into_hip())?),
        };
        Ok(Arc::new(CudaModule {
            inner,
            ctx: self.clone(),
        }))
    }
}

/// The kernel of the runtime that loaded it.
#[derive(Debug, Clone)]
enum Func {
    Cuda(cd::CudaFunction),
    Hip(hd::CudaFunction),
}

/// A kernel of a loaded module.
#[derive(Debug, Clone)]
pub struct CudaFunction {
    inner: Func,
}

unsafe impl Send for CudaFunction {}
unsafe impl Sync for CudaFunction {}

impl CudaModule {
    /// The kernel named `fn_name`.
    pub fn load_function(self: &Arc<Self>, fn_name: &str) -> Result<CudaFunction, DriverError> {
        self.ctx.mark();
        let inner = match &self.inner {
            Md::Cuda(m) => Func::Cuda(cuda(m.load_function(fn_name))?),
            Md::Hip(m) => Func::Hip(m.load_function(fn_name)?),
        };
        Ok(CudaFunction { inner })
    }
}

impl CudaFunction {
    /// The kernel handle.
    pub fn cu_function(&self) -> sys::CUfunction {
        match &self.inner {
            Func::Cuda(f) => cuh(f.cu_function()),
            Func::Hip(f) => f.cu_function(),
        }
    }
}

/// A kernel launch being assembled: the kernel, its stream and its arguments.
#[derive(Debug)]
pub struct LaunchArgs<'a> {
    stream: &'a CudaStream,
    func: &'a CudaFunction,
    args: Vec<*mut c_void>,
    flags: Option<sys::CUevent_flags>,
}

impl CudaStream {
    /// Starts a launch of `func` on this stream.
    pub fn launch_builder<'a>(&'a self, func: &'a CudaFunction) -> LaunchArgs<'a> {
        LaunchArgs {
            stream: self,
            func,
            args: Vec::new(),
            flags: None,
        }
    }
}

/// An argument a kernel launch can take.
///
/// # Safety
/// The pushed address must stay valid until the launch.
pub unsafe trait PushKernelArg<T> {
    fn arg(&mut self, arg: T) -> &mut Self;
}

unsafe impl<'a, 'b: 'a, T: DeviceRepr> PushKernelArg<&'b T> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b T) -> &mut Self {
        self.args.push(arg as *const T as *mut c_void);
        self
    }
}

unsafe impl<'a, 'b: 'a, T> PushKernelArg<&'b CudaSlice<T>> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b CudaSlice<T>) -> &mut Self {
        self.args
            .push(&arg.cu_device_ptr as *const sys::CUdeviceptr as *mut c_void);
        self
    }
}

unsafe impl<'a, 'b: 'a, T> PushKernelArg<&'b mut CudaSlice<T>> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b mut CudaSlice<T>) -> &mut Self {
        self.args
            .push(&arg.cu_device_ptr as *const sys::CUdeviceptr as *mut c_void);
        self
    }
}

unsafe impl<'a, 'b: 'a, 'c: 'b, T> PushKernelArg<&'b CudaView<'c, T>> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b CudaView<'c, T>) -> &mut Self {
        self.args
            .push(&arg.ptr as *const sys::CUdeviceptr as *mut c_void);
        self
    }
}

unsafe impl<'a, 'b: 'a, 'c: 'b, T> PushKernelArg<&'b mut CudaViewMut<'c, T>> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b mut CudaViewMut<'c, T>) -> &mut Self {
        self.args
            .push(&arg.ptr as *const sys::CUdeviceptr as *mut c_void);
        self
    }
}

impl LaunchArgs<'_> {
    /// Records events around the launch, returned by [Self::launch].
    pub fn record_kernel_launch(&mut self, flags: sys::CUevent_flags) -> &mut Self {
        self.flags = Some(flags);
        self
    }

    /// Queues the kernel; with [Self::record_kernel_launch], returns the events recorded before
    /// and after it.
    ///
    /// # Safety
    /// The arguments must match the kernel's parameters.
    #[inline(always)]
    pub unsafe fn launch(
        &mut self,
        cfg: LaunchConfig,
    ) -> Result<Option<(CudaEvent, CudaEvent)>, DriverError> {
        self.stream.bind()?;
        let start = self
            .flags
            .map(|f| self.stream.record_event(Some(f)))
            .transpose()?;
        match (&self.stream.inner, &self.func.inner) {
            (St::Cuda(s), Func::Cuda(f)) => cuda(unsafe {
                cudarc::driver::result::launch_kernel(
                    f.cu_function(),
                    cfg.grid_dim,
                    cfg.block_dim,
                    cfg.shared_mem_bytes,
                    s.cu_stream(),
                    &mut self.args,
                )
            })?,
            (St::Hip(s), Func::Hip(f)) => {
                let mut launch = s.launch_builder(f);
                launch.args = std::mem::take(&mut self.args);
                let out = unsafe { launch.launch(cfg) };
                self.args = std::mem::take(&mut launch.args);
                out?;
            }
            _ => return Err(DriverError(CUresult::CUDA_ERROR_INVALID_CONTEXT)),
        }
        let end = self
            .flags
            .map(|f| self.stream.record_event(Some(f)))
            .transpose()?;
        Ok(start.zip(end))
    }
}

/// The ordinal over both runtimes of a runtime's device `local`.
#[allow(dead_code)]
pub(crate) fn global_ordinal(runtime: Runtime, local: c_int) -> usize {
    topology().global(runtime, local as usize)
}
