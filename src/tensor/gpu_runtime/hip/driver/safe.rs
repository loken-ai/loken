//! The owning wrappers of the driver: a device's context, its streams and events, device
//! buffers and their views, loaded modules and kernel launches.
//!
//! A buffer is not tracked across streams: work on one buffer from two streams is ordered by
//! the caller with events, as it is with the CUDA runtime once event tracking is disabled.

use std::ffi::{c_int, c_void, CString};
use std::marker::PhantomData;
use std::ops::{Bound, RangeBounds};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use super::result::{self, DriverError};
use super::sys::{self, hip, CUresult};

/// A device and the runtime state the process holds on it.
#[derive(Debug)]
pub struct CudaContext {
    pub(crate) cu_device: sys::CUdevice,
    pub(crate) cu_ctx: sys::CUcontext,
    pub(crate) ordinal: usize,
    pub(crate) error_state: AtomicU32,
}

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl Drop for CudaContext {
    fn drop(&mut self) {
        if !self.cu_ctx.is_null() {
            let _ = unsafe { sys::cuDevicePrimaryCtxRelease_v2(self.cu_device) };
        }
    }
}

impl PartialEq for CudaContext {
    fn eq(&self, other: &Self) -> bool {
        self.ordinal == other.ordinal
    }
}
impl Eq for CudaContext {}

impl CudaContext {
    /// Opens the device of `ordinal` and binds it to the calling thread.
    pub fn new(ordinal: usize) -> Result<Arc<Self>, DriverError> {
        result::init()?;
        let cu_device = result::device::get(ordinal as c_int)?;
        let mut cu_ctx: sys::CUcontext = std::ptr::null_mut();
        // A runtime without primary contexts still runs: the device is then selected per thread.
        if unsafe { sys::cuDevicePrimaryCtxRetain(&mut cu_ctx, cu_device) }
            != CUresult::CUDA_SUCCESS
        {
            cu_ctx = std::ptr::null_mut();
        }
        let ctx = Arc::new(Self {
            cu_device,
            cu_ctx,
            ordinal,
            error_state: AtomicU32::new(0),
        });
        ctx.bind_to_thread()?;
        Ok(ctx)
    }

    /// The number of devices.
    pub fn device_count() -> Result<i32, DriverError> {
        result::init()?;
        result::device::get_count()
    }

    /// The device's ordinal.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// The device's name.
    pub fn name(&self) -> Result<String, DriverError> {
        result::device::get_name(self.cu_device)
    }

    /// The device's memory in bytes.
    pub fn total_mem(&self) -> Result<usize, DriverError> {
        unsafe { result::device::total_mem(self.cu_device) }
    }

    /// The free and total memory of the device, in bytes.
    pub fn mem_get_info(&self) -> Result<(usize, usize), DriverError> {
        self.bind_to_thread()?;
        // PLACEMENT-EXEMPT: the runtime's reading itself; its callers read through `place/`.
        result::mem_get_info()
    }

    /// The device handle.
    pub fn cu_device(&self) -> sys::CUdevice {
        self.cu_device
    }

    /// The device's primary context, null when the runtime has none.
    pub fn cu_ctx(&self) -> sys::CUcontext {
        self.cu_ctx
    }

    /// Whether buffers are allocated in stream order: never, see [CudaStream::alloc].
    pub fn has_async_alloc(&self) -> bool {
        false
    }

    /// Makes this device current on the calling thread.
    pub fn bind_to_thread(&self) -> Result<(), DriverError> {
        self.check_err()?;
        unsafe { hip!(hipSetDevice(self.ordinal as c_int)) }.result()
    }

    /// One attribute of the device.
    pub fn attribute(&self, attrib: sys::CUdevice_attribute) -> Result<i32, DriverError> {
        unsafe { result::device::get_attribute(self.cu_device, attrib) }
    }

    /// Waits for all work on the device.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.bind_to_thread()?;
        unsafe { hip!(hipDeviceSynchronize()) }.result()
    }

    /// Makes host waits on this device block the thread instead of spinning.
    pub fn set_blocking_synchronize(&self) -> Result<(), DriverError> {
        self.bind_to_thread()?;
        let schedule_blocking_sync = 4;
        unsafe { hip!(hipSetDeviceFlags(schedule_blocking_sync)) }.result()
    }

    /// Whether more than one stream is in use. Buffers are not tracked across streams, so this
    /// is always false.
    pub fn is_in_multi_stream_mode(&self) -> bool {
        false
    }

    /// Whether buffers record events for cross-stream ordering: never.
    pub fn is_event_tracking(&self) -> bool {
        false
    }

    /// Whether the context orders streams on the caller's behalf: never.
    pub fn is_managing_stream_synchronization(&self) -> bool {
        false
    }

    /// Buffers are not tracked across streams; this does nothing.
    ///
    /// # Safety
    /// None: it changes nothing.
    pub unsafe fn enable_event_tracking(&self) {}

    /// Buffers are not tracked across streams; this does nothing.
    ///
    /// # Safety
    /// None: it changes nothing.
    pub unsafe fn disable_event_tracking(&self) {}

    /// The first error a deferred call recorded, cleared by reading it.
    pub fn check_err(&self) -> Result<(), DriverError> {
        let code = self.error_state.swap(0, Ordering::Relaxed);
        if code == 0 {
            Ok(())
        } else {
            Err(DriverError(CUresult::from_hip(code as c_int)))
        }
    }

    /// Keeps the error of a call that cannot return it, for the next [Self::check_err].
    pub fn record_err<T>(&self, result: Result<T, DriverError>) {
        if let Err(err) = result {
            self.error_state.store(err.0 as u32, Ordering::Relaxed);
        }
    }
}

/// A point in a stream's work.
#[derive(Debug)]
pub struct CudaEvent {
    pub(crate) cu_event: sys::CUevent,
    pub(crate) ctx: Arc<CudaContext>,
}

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        let _ = unsafe { hip!(hipEventDestroy(self.cu_event)) };
    }
}

impl CudaContext {
    /// A new event with no work recorded; without `flags` it records no timing.
    pub fn new_event(
        self: &Arc<Self>,
        flags: Option<sys::CUevent_flags>,
    ) -> Result<CudaEvent, DriverError> {
        let flags = flags.unwrap_or(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING);
        self.bind_to_thread()?;
        let mut cu_event = std::ptr::null_mut();
        unsafe { hip!(hipEventCreateWithFlags(&mut cu_event, flags as u32)) }.result()?;
        Ok(CudaEvent {
            cu_event,
            ctx: self.clone(),
        })
    }
}

impl CudaEvent {
    /// The event handle.
    pub fn cu_event(&self) -> sys::CUevent {
        self.cu_event
    }

    /// The device the event belongs to.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Records the work queued on `stream` so far.
    pub fn record(&self, stream: &CudaStream) -> Result<(), DriverError> {
        if self.ctx != stream.ctx {
            return Err(DriverError(CUresult::CUDA_ERROR_INVALID_CONTEXT));
        }
        self.ctx.bind_to_thread()?;
        unsafe { hip!(hipEventRecord(self.cu_event, stream.cu_stream)) }.result()
    }

    /// Waits on the host for the recorded work.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { hip!(hipEventSynchronize(self.cu_event)) }.result()
    }

    /// Milliseconds between this event and `end`; both must record timing.
    pub fn elapsed_ms(&self, end: &Self) -> Result<f32, DriverError> {
        if self.ctx != end.ctx {
            return Err(DriverError(CUresult::CUDA_ERROR_INVALID_CONTEXT));
        }
        self.synchronize()?;
        end.synchronize()?;
        let mut ms = 0f32;
        unsafe { hip!(hipEventElapsedTime(&mut ms, self.cu_event, end.cu_event)) }.result()?;
        Ok(ms)
    }

    /// Whether the recorded work is done; false on error.
    pub fn is_complete(&self) -> bool {
        self.try_is_complete().unwrap_or(false)
    }

    /// Whether the recorded work is done.
    pub fn try_is_complete(&self) -> Result<bool, DriverError> {
        match unsafe { hip!(hipEventQuery(self.cu_event)) } {
            CUresult::CUDA_SUCCESS => Ok(true),
            CUresult::CUDA_ERROR_NOT_READY => Ok(false),
            e => Err(DriverError(e)),
        }
    }
}

/// A queue of work on a device.
#[derive(Debug)]
pub struct CudaStream {
    pub(crate) cu_stream: sys::CUstream,
    pub(crate) ctx: Arc<CudaContext>,
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl PartialEq for CudaStream {
    fn eq(&self, other: &Self) -> bool {
        self.cu_stream == other.cu_stream && self.ctx == other.ctx
    }
}
impl Eq for CudaStream {}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.cu_stream.is_null() {
            self.ctx.record_err(self.ctx.bind_to_thread());
            let _ = unsafe { hip!(hipStreamDestroy(self.cu_stream)) };
        }
    }
}

impl CudaContext {
    /// The device's default stream.
    pub fn default_stream(self: &Arc<Self>) -> Arc<CudaStream> {
        Arc::new(CudaStream {
            cu_stream: std::ptr::null_mut(),
            ctx: self.clone(),
        })
    }

    /// A new stream that does not wait on the default stream.
    pub fn new_stream(self: &Arc<Self>) -> Result<Arc<CudaStream>, DriverError> {
        self.bind_to_thread()?;
        let mut cu_stream = std::ptr::null_mut();
        let non_blocking = 1;
        unsafe { hip!(hipStreamCreateWithFlags(&mut cu_stream, non_blocking)) }.result()?;
        Ok(Arc::new(CudaStream {
            cu_stream,
            ctx: self.clone(),
        }))
    }
}

impl CudaStream {
    /// A new stream on the same device that starts after the work queued here so far.
    pub fn fork(&self) -> Result<Arc<Self>, DriverError> {
        let stream = self.ctx.new_stream()?;
        stream.join(self)?;
        Ok(stream)
    }

    /// The stream handle.
    pub fn cu_stream(&self) -> sys::CUstream {
        self.cu_stream
    }

    /// The device the stream belongs to.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Waits on the host for the stream's work.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { hip!(hipStreamSynchronize(self.cu_stream)) }.result()
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
        self.ctx.bind_to_thread()?;
        unsafe { hip!(hipStreamWaitEvent(self.cu_stream, event.cu_event, 0)) }.result()
    }

    /// Makes later work on this stream wait for the work queued on `other` so far.
    pub fn join(&self, other: &CudaStream) -> Result<(), DriverError> {
        self.wait(&other.record_event(None)?)
    }

    /// Starts capturing the stream's work into a graph.
    pub fn begin_capture(&self, mode: sys::CUstreamCaptureMode) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { result::stream::begin_capture(self.cu_stream, mode) }
    }

    /// Ends the capture; `None` when nothing was captured.
    pub fn end_capture(
        self: &Arc<Self>,
        flags: sys::CUgraphInstantiate_flags,
    ) -> Result<Option<CudaGraph>, DriverError> {
        self.ctx.bind_to_thread()?;
        let cu_graph = unsafe { result::stream::end_capture(self.cu_stream) }?;
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
        self.ctx.bind_to_thread()?;
        unsafe { result::stream::is_capturing(self.cu_stream) }
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
        ctx.record_err(unsafe { result::graph::exec_destroy(self.cu_graph_exec) });
        ctx.record_err(unsafe { result::graph::destroy(self.cu_graph) });
    }
}

impl CudaGraph {
    /// Runs the graph on the stream it was captured from.
    pub fn launch(&self) -> Result<(), DriverError> {
        self.stream.ctx.bind_to_thread()?;
        unsafe { result::graph::launch(self.cu_graph_exec, self.stream.cu_stream) }
    }

    /// Uploads the graph's resources ahead of its first launch.
    pub fn upload(&self) -> Result<(), DriverError> {
        self.stream.ctx.bind_to_thread()?;
        unsafe { result::graph::upload(self.cu_graph_exec, self.stream.cu_stream) }
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

/// An array of `T` in device memory, freed on drop.
#[derive(Debug)]
pub struct CudaSlice<T> {
    pub(crate) cu_device_ptr: sys::CUdeviceptr,
    pub(crate) len: usize,
    pub(crate) stream: Arc<CudaStream>,
    marker: PhantomData<*const T>,
}

unsafe impl<T> Send for CudaSlice<T> {}
unsafe impl<T> Sync for CudaSlice<T> {}

impl<T> Drop for CudaSlice<T> {
    fn drop(&mut self) {
        if self.cu_device_ptr == 0 {
            return;
        }
        let ctx = &self.stream.ctx;
        // A buffer released while its card captures stays allocated for the captured graph.
        if super::capture_hold::hold(ctx.cu_device, self.cu_device_ptr) {
            return;
        }
        ctx.record_err(ctx.bind_to_thread());
        ctx.record_err(unsafe { result::free_sync(self.cu_device_ptr) });
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

    /// The device's ordinal.
    pub fn ordinal(&self) -> usize {
        self.stream.ctx.ordinal
    }

    /// The device the array lives on.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.stream.ctx
    }

    /// The stream the array is ordered on.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Gives up ownership: the memory is no longer freed, and its address is returned.
    pub fn leak(self) -> sys::CUdeviceptr {
        let ptr = self.cu_device_ptr;
        std::mem::forget(self);
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

/// A type whose all-zero bytes are a valid value.
///
/// # Safety
/// Zeroed memory must be a valid `Self`.
pub unsafe trait ValidAsZeroBits {}

/// A type that can be copied to the device byte for byte and passed to a kernel by value.
///
/// # Safety
/// `Self` must have the layout the device code expects.
pub unsafe trait DeviceRepr {}

macro_rules! plain {
    ($($t:ty),*) => {$(
        unsafe impl ValidAsZeroBits for $t {}
        unsafe impl DeviceRepr for $t {}
    )*};
}
plain!(bool, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64);
plain!(half::f16, half::bf16);
unsafe impl<T: ValidAsZeroBits, const M: usize> ValidAsZeroBits for [T; M] {}
unsafe impl<T: DeviceRepr, const N: usize> DeviceRepr for [T; N] {}

macro_rules! zero_tuples {
    ($($t:ident),+) => {
        unsafe impl<$($t: ValidAsZeroBits),+> ValidAsZeroBits for ($($t,)+) {}
    };
}
zero_tuples!(A);
zero_tuples!(A, B);
zero_tuples!(A, B, C);
zero_tuples!(A, B, C, D);
zero_tuples!(A, B, C, D, E);
zero_tuples!(A, B, C, D, E, F);

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
            marker: PhantomData,
        })
    }

    /// An uninitialized array of `len` elements.
    ///
    /// # Safety
    /// The contents must be written before they are read.
    pub unsafe fn alloc<T: DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError> {
        self.ctx.bind_to_thread()?;
        let bytes = len * std::mem::size_of::<T>();
        // Allocations are made outside stream order: the runtime's stream-ordered allocator
        // faults now and then on a fresh allocation's first write.
        let cu_device_ptr = if bytes == 0 {
            0
        } else {
            unsafe { result::malloc_sync(bytes) }?
        };
        Ok(CudaSlice {
            cu_device_ptr,
            len,
            stream: self.clone(),
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
        self.ctx.bind_to_thread()?;
        let (ptr, _record) = dst.device_ptr_mut(self);
        unsafe { sys::cuMemsetD8Async(ptr, 0, bytes, self.cu_stream) }.result()
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
        self.ctx.bind_to_thread()?;
        let (src, _record_src) = unsafe { src.stream_synced_slice(self) };
        let (dst, _record_dst) = dst.device_ptr_mut(self);
        unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                dst,
                src.as_ptr() as *const c_void,
                std::mem::size_of_val(src),
                self.cu_stream,
            )
        }
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
        self.ctx.bind_to_thread()?;
        let (src, _record_src) = src.device_ptr(self);
        let (dst, _record_dst) = unsafe { dst.stream_synced_mut_slice(self) };
        unsafe {
            sys::cuMemcpyDtoHAsync_v2(
                dst.as_mut_ptr() as *mut c_void,
                src,
                std::mem::size_of_val(dst),
                self.cu_stream,
            )
        }
        .result()
    }

    /// Copies a device range into another of the same length.
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
        self.ctx.bind_to_thread()?;
        let (src, _record_src) = src.device_ptr(self);
        let (dst, _record_dst) = dst.device_ptr_mut(self);
        unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, bytes, self.cu_stream) }.result()
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

    /// Takes ownership of device memory allocated elsewhere, freed on drop. The runtime frees
    /// stream-ordered and pool allocations this way too.
    ///
    /// # Safety
    /// `cu_device_ptr` must hold `len` elements of `T` and be freed by nothing else.
    pub unsafe fn upgrade_device_ptr<T>(
        self: &Arc<Self>,
        cu_device_ptr: sys::CUdeviceptr,
        len: usize,
    ) -> CudaSlice<T> {
        CudaSlice {
            cu_device_ptr,
            len,
            stream: self.clone(),
            marker: PhantomData,
        }
    }
}

/// A loaded code object.
#[derive(Debug)]
pub struct CudaModule {
    pub(crate) cu_module: sys::CUmodule,
    pub(crate) ctx: Arc<CudaContext>,
}

unsafe impl Send for CudaModule {}
unsafe impl Sync for CudaModule {}

impl Drop for CudaModule {
    fn drop(&mut self) {
        self.ctx.record_err(self.ctx.bind_to_thread());
        let _ = unsafe { hip!(hipModuleUnload(self.cu_module)) };
    }
}

impl CudaContext {
    /// Loads a compiled module onto the device. Only code objects load: text that is not one
    /// (assembly for another vendor's cards) is not supported.
    pub fn load_module(
        self: &Arc<Self>,
        ptx: crate::tensor::gpu_runtime::hip::nvrtc::Ptx,
    ) -> Result<Arc<CudaModule>, DriverError> {
        self.bind_to_thread()?;
        let image = ptx
            .code_object()
            .ok_or(DriverError(CUresult::CUDA_ERROR_NOT_SUPPORTED))?;
        let mut cu_module = std::ptr::null_mut();
        unsafe {
            hip!(hipModuleLoadData(
                &mut cu_module,
                image.as_ptr() as *const c_void
            ))
        }
        .result()?;
        Ok(Arc::new(CudaModule {
            cu_module,
            ctx: self.clone(),
        }))
    }
}

/// A kernel of a loaded module.
#[derive(Debug, Clone)]
pub struct CudaFunction {
    pub(crate) cu_function: sys::CUfunction,
    #[allow(unused)]
    pub(crate) module: Arc<CudaModule>,
}

unsafe impl Send for CudaFunction {}
unsafe impl Sync for CudaFunction {}

impl CudaModule {
    /// The kernel named `fn_name`.
    pub fn load_function(self: &Arc<Self>, fn_name: &str) -> Result<CudaFunction, DriverError> {
        let name =
            CString::new(fn_name).map_err(|_| DriverError(CUresult::CUDA_ERROR_INVALID_VALUE))?;
        let mut cu_function = std::ptr::null_mut();
        unsafe {
            hip!(hipModuleGetFunction(
                &mut cu_function,
                self.cu_module,
                name.as_ptr()
            ))
        }
        .result()?;
        Ok(CudaFunction {
            cu_function,
            module: self.clone(),
        })
    }
}

impl CudaFunction {
    /// The kernel handle.
    pub fn cu_function(&self) -> sys::CUfunction {
        self.cu_function
    }
}

/// The shape of a kernel launch.
#[derive(Clone, Copy, Debug)]
pub struct LaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    /// One thread per element in blocks of the largest size every device accepts.
    pub fn for_num_elems(n: u32) -> Self {
        const NUM_THREADS: u32 = 1024;
        Self {
            grid_dim: (n.div_ceil(NUM_THREADS), 1, 1),
            block_dim: (NUM_THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

/// A kernel launch being assembled: the kernel, its stream and its arguments.
#[derive(Debug)]
pub struct LaunchArgs<'a> {
    pub(crate) stream: &'a CudaStream,
    pub(crate) func: &'a CudaFunction,
    pub(crate) args: Vec<*mut c_void>,
    pub(crate) flags: Option<sys::CUevent_flags>,
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
        let ctx = &self.stream.ctx;
        ctx.bind_to_thread()?;
        let start = self
            .flags
            .map(|f| self.stream.record_event(Some(f)))
            .transpose()?;
        unsafe {
            hip!(hipModuleLaunchKernel(
                self.func.cu_function,
                cfg.grid_dim.0,
                cfg.grid_dim.1,
                cfg.grid_dim.2,
                cfg.block_dim.0,
                cfg.block_dim.1,
                cfg.block_dim.2,
                cfg.shared_mem_bytes,
                self.stream.cu_stream,
                self.args.as_mut_ptr(),
                std::ptr::null_mut(),
            ))
        }
        .result()?;
        let end = self
            .flags
            .map(|f| self.stream.record_event(Some(f)))
            .transpose()?;
        Ok(start.zip(end))
    }
}
