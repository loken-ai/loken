//! A captured CUDA graph with the node introspection and the in-place update that loken's
//! decode paths use, built on cudarc's raw driver surface.
//!
//! Invariants: a `CapturedGraph` owns one `CUgraph` and the `CUgraphExec` instantiated from it,
//! and destroys both on drop, the executable first. An update keeps the executable and replaces
//! the graph that backs it; a failed update re-instantiates from the new graph with the flags of
//! the first instantiation.
//! The driver does not synchronise graph objects, so one is never used from two threads at once.

use cudarc::driver::{result, sys, CudaStream, DriverError};
use std::sync::Arc;

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
    unsafe { result::stream::begin_capture(stream.cu_stream(), mode) }
}

/// End the capture begun on `stream` and instantiate the graph with `flags`. `Ok(None)` when
/// the driver returns no graph.
pub fn end_capture(
    stream: &Arc<CudaStream>,
    flags: sys::CUgraphInstantiate_flags,
) -> Result<Option<CapturedGraph>, DriverError> {
    let ctx = stream.context();
    ctx.bind_to_thread()?;
    let cu_graph = unsafe { result::stream::end_capture(stream.cu_stream()) }?;
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
        unsafe { result::graph::launch(self.cu_graph_exec, self.stream.cu_stream()) }
    }

    /// Upload the graph's resources so the first launch carries no setup cost.
    pub fn upload(&self) -> Result<(), DriverError> {
        self.stream.context().bind_to_thread()?;
        unsafe { result::graph::upload(self.cu_graph_exec, self.stream.cu_stream()) }
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
        let new_graph = unsafe { result::stream::end_capture(self.stream.cu_stream()) }?;
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
    use cudarc::driver::CudaContext;

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
