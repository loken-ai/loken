//! A captured CUDA graph with the node introspection and the in-place update that loken's
//! decode paths use, built on cudarc's raw driver surface.
//!
//! Invariants: a `CapturedGraph` owns one `CUgraph` and the `CUgraphExec` instantiated from it,
//! and destroys both on drop, the executable first. An update keeps the executable and replaces
//! the graph that backs it; a failed update re-instantiates from the new graph with the flags of
//! the first instantiation.
//! The driver does not synchronise graph objects, so one is never used from two threads at once.
//!
//! A card's device has one stream that every thread of the process queues on. A capture never
//! runs on it: the driver records whatever any thread queues on a stream while it captures. A
//! capture begun on that stream is made on a recording stream of the card instead, which only
//! the capturing thread is given while it records (see `CudaDevice::stream`); the graph replays
//! on the card's stream. One thread records on a card at a time, and a recording stream is
//! recorded on again only once no buffer allocated on it is left, so no other thread's free can
//! reach a capture.

use super::CudaDevice;
use crate::tensor::gpu_runtime::cublas::CudaBlas;
use crate::tensor::gpu_runtime::driver::{result, sys, CudaStream, DriverError};
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// A stream graphs are recorded on, and the cuBLAS handle bound to it.
pub(super) struct Recorder {
    pub(super) stream: Arc<CudaStream>,
    pub(super) blas: OnceLock<CudaBlas>,
    /// The handle's workspace when the card has none pinned, allocated on the card's stream.
    pub(super) workspace: OnceLock<crate::tensor::gpu_runtime::driver::CudaSlice<u8>>,
}

impl Recorder {
    /// Whether the recorder can take a capture: nothing but itself holds its stream, so no
    /// buffer allocated on it is left, and no capture its thread left open stands on it.
    fn is_idle(&self) -> bool {
        Arc::strong_count(&self.stream) == 1 + usize::from(self.blas.get().is_some())
            && self.stream.context().bind_to_thread().is_ok()
            && unsafe { result::stream::is_capturing(self.stream.cu_stream()) }
                .is_ok_and(|s| s == sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE)
    }
}

/// A card's graph recording: the thread that records and the recorder it records on.
#[derive(Default)]
pub(super) struct Recording {
    /// The token of the recording thread, zero when none records. Read without the lock on
    /// every stream lookup.
    owner: AtomicU64,
    /// The same token, under the lock the next recorder sleeps on.
    held: Mutex<u64>,
    ended: Condvar,
    /// Every recorder made for the card, each boxed so it keeps its address as the list grows.
    /// They live as long as the card, so a reference to one stays valid for as long as one to
    /// the card does.
    #[allow(clippy::vec_box)]
    pub(super) made: Mutex<Vec<Box<Recorder>>>,
    /// The recorder of the capture in progress, null when none.
    current: AtomicPtr<Recorder>,
}

/// Recordings in progress over all cards; zero lets a graph launch skip the lookup.
static RECORDING: AtomicUsize = AtomicUsize::new(0);

/// A number naming the calling thread, never zero. A thread whose locals are already torn down
/// records nothing, and is named by a number no thread is given.
fn thread_token() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    thread_local! {
        static TOKEN: u64 = NEXT.fetch_add(1, Ordering::Relaxed);
    }
    TOKEN.try_with(|t| *t).unwrap_or(u64::MAX)
}

/// The cards whose recording the calling thread holds, given back if it ends without ending
/// its capture.
struct Held(Vec<Arc<CudaDevice>>, u64);

impl Drop for Held {
    fn drop(&mut self) {
        for dev in self.0.drain(..) {
            dev.recording.release(self.1);
        }
    }
}

thread_local! {
    static HELD: std::cell::RefCell<Held> = const { std::cell::RefCell::new(Held(Vec::new(), 0)) };
}

impl Recording {
    /// The recorder of the calling thread's capture on this card, if it records on it.
    #[inline]
    pub(super) fn mine(&self) -> Option<&Recorder> {
        let owner = self.owner.load(Ordering::Acquire);
        if owner == 0 || owner != thread_token() {
            return None;
        }
        let current = self.current.load(Ordering::Acquire);
        // Safety: a recorder is boxed in `made` and never dropped before the card.
        (!current.is_null()).then(|| unsafe { &*current })
    }

    /// Whether `stream` is a recording stream of this card.
    pub(super) fn made(&self, stream: &Arc<CudaStream>) -> bool {
        self.made
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .any(|r| Arc::ptr_eq(&r.stream, stream))
    }

    /// Ends the recording of the thread `token`, if it holds it, and wakes the next recorder.
    fn release(&self, token: u64) {
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        if *held == token {
            *held = 0;
            self.current.store(std::ptr::null_mut(), Ordering::Release);
            self.owner.store(0, Ordering::Release);
            RECORDING.fetch_sub(1, Ordering::AcqRel);
            drop(held);
            self.ended.notify_all();
        }
    }
}

impl CudaDevice {
    /// Gives the calling thread this card's recording once no other thread holds it, on a
    /// recorder no buffer is left on, and answers the stream to capture on.
    fn take_recording(self: &Arc<Self>) -> Result<Arc<CudaStream>, DriverError> {
        let me = thread_token();
        let mut held = self
            .recording
            .held
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        while *held != 0 && *held != me {
            held = self
                .recording
                .ended
                .wait(held)
                .unwrap_or_else(|e| e.into_inner());
        }
        if *held == me {
            if let Some(r) = self.recording.mine() {
                return Ok(r.stream.clone());
            }
        }
        let recorder = {
            let mut made = self
                .recording
                .made
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            match made.iter().position(|r| r.is_idle()) {
                Some(i) => &*made[i] as *const Recorder,
                None => {
                    made.push(Box::new(Recorder {
                        stream: self.ctx.new_stream()?,
                        blas: OnceLock::new(),
                        workspace: OnceLock::new(),
                    }));
                    &**made.last().expect("pushed above") as *const Recorder
                }
            }
        };
        // Safety: boxed in `made`, which keeps it as long as the card.
        self.ready_recorder(unsafe { &*recorder }).map_err(|e| {
            tracing::warn!("graph recording: the recorder's cuBLAS handle: {e}");
            DriverError(sys::CUresult::CUDA_ERROR_NOT_INITIALIZED)
        })?;
        if *held != me {
            *held = me;
            RECORDING.fetch_add(1, Ordering::AcqRel);
            HELD.with(|h| {
                let mut h = h.borrow_mut();
                h.1 = me;
                h.0.push(self.clone());
            });
        }
        self.recording
            .current
            .store(recorder as *mut Recorder, Ordering::Release);
        self.recording.owner.store(me, Ordering::Release);
        // Safety: boxed in `made`, which keeps it as long as the card.
        Ok(unsafe { &*recorder }.stream.clone())
    }

    /// Ends the calling thread's recording on this card; nothing when it holds none.
    fn give_back_recording(&self) {
        let _ = HELD.try_with(|h| {
            h.borrow_mut()
                .0
                .retain(|d| !std::ptr::eq(Arc::as_ptr(d), self as *const Self));
        });
        self.recording.release(thread_token());
    }

    /// Orders the free of a buffer allocated on `stream` behind the work queued on the card's
    /// stream so far, when `stream` is one of the card's recording streams: the buffers a
    /// capture allocated are used by its replays and by the work around them, all on the card's
    /// stream. A buffer the caller frees inside its own capture is a node of the graph.
    pub(crate) fn before_free(&self, stream: &Arc<CudaStream>) {
        if Arc::ptr_eq(stream, &self.stream) || !self.recording.made(stream) {
            return;
        }
        if self
            .recording
            .mine()
            .is_some_and(|r| Arc::ptr_eq(&r.stream, stream))
        {
            return;
        }
        let _ = stream.join(&self.stream);
    }
}

/// The card whose shared stream `stream` is.
fn card_of(stream: &Arc<CudaStream>) -> Option<Arc<CudaDevice>> {
    super::shared_card_of(stream)
}

/// The stream work meant for `stream` goes to from the calling thread: the card's recording
/// stream while the caller records on the card `stream` is shared on, `stream` otherwise.
pub fn routed_stream(stream: &Arc<CudaStream>) -> Arc<CudaStream> {
    if RECORDING.load(Ordering::Acquire) == 0 {
        return stream.clone();
    }
    match card_of(stream) {
        Some(dev) if dev.recording.mine().is_some() => dev.stream().clone(),
        _ => stream.clone(),
    }
}

/// How `end_capture_or_reinstantiate` brought the executable up to date.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateOutcome {
    /// The new capture patched the executable in place.
    Updated,
    /// The structure differed: a new executable was instantiated.
    Reinstantiated,
}

pub struct CapturedGraph {
    cu_graph: sys::CUgraph,
    cu_graph_exec: sys::CUgraphExec,
    flags: sys::CUgraphInstantiate_flags,
    stream: Arc<CudaStream>,
}

/// Begin capturing `stream` into a graph.
pub fn begin_capture(
    stream: &Arc<CudaStream>,
    mode: sys::CUstreamCaptureMode,
) -> Result<(), DriverError> {
    stream.context().bind_to_thread()?;
    let Some(dev) = card_of(stream) else {
        return unsafe { result::stream::begin_capture(stream.cu_stream(), mode) };
    };
    let target = dev.take_recording()?;
    let begun = unsafe { result::stream::begin_capture(target.cu_stream(), mode) };
    if begun.is_err() {
        dev.give_back_recording();
    }
    begun
}

/// End the capture begun on `stream` and instantiate the graph with `flags`. `Ok(None)` when
/// the driver returns no graph.
pub fn end_capture(
    stream: &Arc<CudaStream>,
    flags: sys::CUgraphInstantiate_flags,
) -> Result<Option<CapturedGraph>, DriverError> {
    let ctx = stream.context();
    ctx.bind_to_thread()?;
    let cu_graph = end_on(stream)?;
    if cu_graph.is_null() {
        return Ok(None);
    }
    let cu_graph_exec = match unsafe { result::graph::instantiate(cu_graph, flags) } {
        Ok(exec) => exec,
        Err(error) => {
            ctx.record_err(unsafe { result::graph::destroy(cu_graph) });
            return Err(error);
        }
    };
    Ok(Some(CapturedGraph {
        cu_graph,
        cu_graph_exec,
        flags,
        stream: stream.clone(),
    }))
}

/// Ends the capture begun on `stream`, on the stream it was made on, and gives back the card's
/// recording when the caller held it.
fn end_on(stream: &Arc<CudaStream>) -> Result<sys::CUgraph, DriverError> {
    let Some(dev) = card_of(stream).filter(|d| d.recording.mine().is_some()) else {
        return unsafe { result::stream::end_capture(stream.cu_stream()) };
    };
    let target = dev.stream().clone();
    let ended = unsafe { result::stream::end_capture(target.cu_stream()) };
    dev.give_back_recording();
    ended
}

impl Drop for CapturedGraph {
    fn drop(&mut self) {
        let ctx = self.stream.context();
        let exec = std::mem::replace(&mut self.cu_graph_exec, std::ptr::null_mut());
        if !exec.is_null() {
            ctx.record_err(unsafe { result::graph::exec_destroy(exec) });
        }
        let graph = std::mem::replace(&mut self.cu_graph, std::ptr::null_mut());
        if !graph.is_null() {
            ctx.record_err(unsafe { result::graph::destroy(graph) });
        }
    }
}

impl CapturedGraph {
    /// Replay the graph on the stream it was captured on.
    pub fn launch(&self) -> Result<(), DriverError> {
        self.stream.context().bind_to_thread()?;
        let stream = routed_stream(&self.stream);
        unsafe { result::graph::launch(self.cu_graph_exec, stream.cu_stream()) }
    }

    /// Upload the graph's resources so the first launch carries no setup cost.
    pub fn upload(&self) -> Result<(), DriverError> {
        self.stream.context().bind_to_thread()?;
        let stream = routed_stream(&self.stream);
        unsafe { result::graph::upload(self.cu_graph_exec, stream.cu_stream()) }
    }

    /// The number of nodes in the captured graph.
    pub fn num_nodes(&self) -> Result<usize, DriverError> {
        self.stream.context().bind_to_thread()?;
        let mut count: usize = 0;
        unsafe { sys::cuGraphGetNodes(self.cu_graph, std::ptr::null_mut(), &mut count) }
            .result()?;
        Ok(count)
    }

    /// Up to `cap` node handles of the graph; `cap = num_nodes()?` returns all of them.
    pub fn nodes(&self, cap: usize) -> Result<Vec<sys::CUgraphNode>, DriverError> {
        self.stream.context().bind_to_thread()?;
        let mut nodes: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); cap];
        let mut count = cap;
        unsafe { sys::cuGraphGetNodes(self.cu_graph, nodes.as_mut_ptr(), &mut count) }.result()?;
        nodes.truncate(count);
        Ok(nodes)
    }

    /// The type of each of `nodes`, in the same order.
    pub fn node_types(
        &self,
        nodes: &[sys::CUgraphNode],
    ) -> Result<Vec<sys::CUgraphNodeType>, DriverError> {
        self.stream.context().bind_to_thread()?;
        let mut types = Vec::with_capacity(nodes.len());
        for &node in nodes {
            let mut ty = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL;
            unsafe { sys::cuGraphNodeGetType(node, &mut ty) }.result()?;
            types.push(ty);
        }
        Ok(types)
    }

    /// Switch `node` on or off in the executable: a node that is off is skipped at launch.
    pub fn set_node_enabled(
        &self,
        nodes: &[sys::CUgraphNode],
        at: usize,
        on: bool,
    ) -> Result<(), DriverError> {
        self.stream.context().bind_to_thread()?;
        let node = *nodes
            .get(at)
            .ok_or(DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE))?;
        unsafe { sys::cuGraphNodeSetEnabled(self.cu_graph_exec, node, u32::from(on)) }.result()
    }

    /// The name of the kernel each of `nodes` launches, `None` for a node that is not a kernel.
    pub fn kernel_names(
        &self,
        nodes: &[sys::CUgraphNode],
    ) -> Result<Vec<Option<String>>, DriverError> {
        self.stream.context().bind_to_thread()?;
        let mut names = Vec::with_capacity(nodes.len());
        for &node in nodes {
            let mut ty = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL;
            unsafe { sys::cuGraphNodeGetType(node, &mut ty) }.result()?;
            if ty != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
                names.push(None);
                continue;
            }
            let mut params = std::mem::MaybeUninit::<sys::CUDA_KERNEL_NODE_PARAMS>::uninit();
            unsafe { sys::cuGraphKernelNodeGetParams_v2(node, params.as_mut_ptr()) }.result()?;
            let params = unsafe { params.assume_init() };
            let mut name: *const std::ffi::c_char = std::ptr::null();
            let named = unsafe { sys::cuFuncGetName(&mut name, params.func) }.result();
            names.push(match named {
                Ok(()) if !name.is_null() => Some(
                    unsafe { std::ffi::CStr::from_ptr(name) }
                        .to_string_lossy()
                        .into_owned(),
                ),
                _ => Some(String::from("?")),
            });
        }
        Ok(names)
    }

    /// End the capture begun on this graph's stream and bring the executable up to date with
    /// it: patched in place when the structure matches, instantiated afresh when it does not.
    pub fn end_capture_or_reinstantiate(&mut self) -> Result<UpdateOutcome, DriverError> {
        let ctx = self.stream.context();
        ctx.bind_to_thread()?;
        let new_graph = end_on(&self.stream)?;
        if new_graph.is_null() {
            return Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE));
        }
        let mut info = sys::CUgraphExecUpdateResultInfo_st {
            result: sys::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_SUCCESS,
            errorNode: std::ptr::null_mut(),
            errorFromNode: std::ptr::null_mut(),
        };
        let status = unsafe { sys::cuGraphExecUpdate_v2(self.cu_graph_exec, new_graph, &mut info) };
        if status == sys::CUresult::CUDA_SUCCESS
            && info.result == sys::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_SUCCESS
        {
            unsafe { result::graph::destroy(self.cu_graph) }?;
            self.cu_graph = new_graph;
            return Ok(UpdateOutcome::Updated);
        }
        let new_exec = match unsafe { result::graph::instantiate(new_graph, self.flags) } {
            Ok(exec) => exec,
            Err(error) => {
                ctx.record_err(unsafe { result::graph::destroy(new_graph) });
                return Err(error);
            }
        };
        ctx.record_err(unsafe { result::graph::exec_destroy(self.cu_graph_exec) });
        self.cu_graph_exec = new_exec;
        ctx.record_err(unsafe { result::graph::destroy(self.cu_graph) });
        self.cu_graph = new_graph;
        Ok(UpdateOutcome::Reinstantiated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::gpu_runtime::driver::CudaContext;

    const CAPTURE: sys::CUstreamCaptureMode =
        sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED;

    fn default_flags() -> sys::CUgraphInstantiate_flags {
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
    }

    /// Capture, replay, introspect and recapture a one-node graph on a card.
    #[test]
    #[ignore = "needs a CUDA device; run in a lock window"]
    fn a_captured_graph_replays_reports_its_nodes_and_updates_in_place() {
        let ctx = CudaContext::new(0).unwrap();
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream().unwrap();
        let mut data = stream.clone_htod(&[1u32; 4]).unwrap();
        stream.synchronize().unwrap();

        begin_capture(&stream, CAPTURE).unwrap();
        stream.memset_zeros(&mut data).unwrap();
        let mut graph = end_capture(&stream, default_flags()).unwrap().unwrap();

        graph.launch().unwrap();
        assert_eq!(stream.clone_dtoh(&data).unwrap(), [0; 4]);

        let count = graph.num_nodes().unwrap();
        assert!(count >= 1);
        let nodes = graph.nodes(count).unwrap();
        assert_eq!(nodes.len(), count);
        assert_eq!(graph.node_types(&nodes).unwrap().len(), count);

        stream.memcpy_htod(&[2u32; 4], &mut data).unwrap();
        begin_capture(&stream, CAPTURE).unwrap();
        stream.memset_zeros(&mut data).unwrap();
        assert_eq!(
            graph.end_capture_or_reinstantiate().unwrap(),
            UpdateOutcome::Updated
        );
        graph.upload().unwrap();
        graph.launch().unwrap();
        assert_eq!(stream.clone_dtoh(&data).unwrap(), [0; 4]);
    }
}
