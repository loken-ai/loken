//! The device memory a captured graph addresses, kept for as long as the graph lives.
//!
//! HIP frees a buffer the moment it is released, and its graphs own no memory of their own: a
//! buffer released while its stream was capturing would leave the graph's nodes addressing freed
//! memory at every replay. So while a card has a stream capturing, the buffers of that card that
//! are released are held instead of freed. The graphs that capture ends with, and the
//! executables instantiated from them or updated with them, share the hold; the memory is freed
//! when the last of them is destroyed, or at once when the capture ends with no graph.
//!
//! Invariants: a card's hold collects releases from its first capture's begin to its last
//! capture's end, so captures that overlap on one card share one hold. A graph or an executable
//! is in the tables only while it holds memory.

use std::collections::HashMap;
use std::ffi::{c_int, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use super::sys::{self, hip};

/// Device memory freed when the last graph or executable sharing it is gone.
#[derive(Default)]
struct Held(Mutex<Vec<usize>>);

impl Drop for Held {
    fn drop(&mut self) {
        let ptrs = std::mem::take(self.0.get_mut().unwrap_or_else(|e| e.into_inner()));
        for p in ptrs {
            let _ = unsafe { hip!(hipFree(p as *mut c_void)) };
        }
    }
}

/// The captures in flight on one card and the hold their releases go to.
struct Capture {
    depth: usize,
    held: Arc<Held>,
}

#[derive(Default)]
struct Tables {
    capturing: HashMap<c_int, Capture>,
    graphs: HashMap<usize, Arc<Held>>,
    execs: HashMap<usize, Arc<Held>>,
}

/// The number of captures in flight over every card: a release with none frees at once.
static CAPTURES: AtomicUsize = AtomicUsize::new(0);

fn tables() -> MutexGuard<'static, Tables> {
    static TABLES: std::sync::OnceLock<Mutex<Tables>> = std::sync::OnceLock::new();
    TABLES
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// The card `stream` runs on; the calling thread's current card for the null stream.
fn device_of(stream: sys::CUstream) -> Option<c_int> {
    let mut d: c_int = 0;
    let r = unsafe { hip!(hipStreamGetDevice(stream, &mut d)) };
    (r == sys::CUresult::CUDA_SUCCESS).then_some(d)
}

/// A capture began on `stream`.
pub(crate) fn begin(stream: sys::CUstream) {
    let Some(device) = device_of(stream) else {
        return;
    };
    let mut t = tables();
    t.capturing
        .entry(device)
        .or_insert_with(|| Capture {
            depth: 0,
            held: Arc::default(),
        })
        .depth += 1;
    CAPTURES.fetch_add(1, Ordering::SeqCst);
}

/// The capture on `stream` ended with `graph`, null when it produced none.
pub(crate) fn end(stream: sys::CUstream, graph: sys::CUgraph) {
    let Some(device) = device_of(stream) else {
        return;
    };
    let mut t = tables();
    let Some(capture) = t.capturing.get_mut(&device) else {
        return;
    };
    let held = capture.held.clone();
    capture.depth -= 1;
    if capture.depth == 0 {
        t.capturing.remove(&device);
    }
    CAPTURES.fetch_sub(1, Ordering::SeqCst);
    if !graph.is_null() {
        t.graphs.insert(graph as usize, held);
    }
}

/// Holds the release of `ptr`, a buffer of card `device`, when that card is capturing; the
/// caller frees it otherwise.
pub(crate) fn hold(device: c_int, ptr: sys::CUdeviceptr) -> bool {
    if CAPTURES.load(Ordering::SeqCst) == 0 {
        return false;
    }
    let t = tables();
    match t.capturing.get(&device) {
        Some(capture) => {
            capture
                .held
                .0
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(ptr as usize);
            true
        }
        None => false,
    }
}

/// `exec` was instantiated from `graph`: it shares the graph's hold.
pub(crate) fn instantiated(graph: sys::CUgraph, exec: sys::CUgraphExec) {
    let mut t = tables();
    if let Some(held) = t.graphs.get(&(graph as usize)).cloned() {
        t.execs.insert(exec as usize, held);
    }
}

/// `exec` was updated to run `graph`: it holds the graph's memory in place of its own.
pub(crate) fn updated(exec: sys::CUgraphExec, graph: sys::CUgraph) {
    let released = {
        let mut t = tables();
        match t.graphs.get(&(graph as usize)).cloned() {
            Some(held) => t.execs.insert(exec as usize, held),
            None => t.execs.remove(&(exec as usize)),
        }
    };
    drop(released);
}

/// `graph` was destroyed.
pub(crate) fn graph_destroyed(graph: sys::CUgraph) {
    let released = tables().graphs.remove(&(graph as usize));
    drop(released);
}

/// `exec` was destroyed.
pub(crate) fn exec_destroyed(exec: sys::CUgraphExec) {
    let released = tables().execs.remove(&(exec as usize));
    drop(released);
}

#[cfg(test)]
mod tests {
    use super::super::{sys, CudaContext};
    use super::*;

    /// A buffer the capture allocates and releases stays allocated while the graph lives: every
    /// replay copies through it, and the graph's destruction releases it.
    #[test]
    fn a_graph_replays_through_a_buffer_released_while_it_was_captured() {
        let Ok(ctx) = CudaContext::new(0) else {
            return;
        };
        let stream = ctx.new_stream().unwrap();
        let src = stream.clone_htod(&[7u32; 64]).unwrap();
        let mut dst = stream.alloc_zeros::<u32>(64).unwrap();
        stream.synchronize().unwrap();

        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .unwrap();
        let mut transient = unsafe { stream.alloc::<u32>(64) }.unwrap();
        stream.memcpy_dtod(&src, &mut transient).unwrap();
        stream.memcpy_dtod(&transient, &mut dst).unwrap();
        drop(transient);
        let flags = sys::CUgraphInstantiate_flags_enum(0);
        let graph = stream.end_capture(flags).unwrap().unwrap();
        assert!(tables().graphs.contains_key(&(graph.cu_graph() as usize)));

        for _ in 0..3 {
            stream.memset_zeros(&mut dst).unwrap();
            graph.launch().unwrap();
            assert_eq!(stream.clone_dtoh(&dst).unwrap(), [7u32; 64]);
        }
        let (g, e) = (graph.cu_graph() as usize, graph.cu_graph_exec() as usize);
        drop(graph);
        let t = tables();
        assert!(!t.graphs.contains_key(&g) && !t.execs.contains_key(&e));
    }
}
