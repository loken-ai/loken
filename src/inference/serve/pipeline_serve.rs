//! Running a serve node as a pipeline tail, triggered by the cluster rather than an operator.
//!
//! When the auto-tune decision splits a model's layers across nodes, the head asks each tail node
//! to run this: load only its own layer range from the local model file and serve the boundary
//! frames the head sends, replying with the logits. It is the same tail mechanism the model exposes
//! (`serve_pipeline_tail_mux`), driven from the serving path instead of a command line. Additive -
//! it touches no other serving path, and a node that is never asked to be a tail never runs it.

use std::net::TcpListener;

/// Whether the model at `gguf` serves a pipeline tail of its own from the file, as opposed to
/// being a generic transformer whose share the engine loads. A file that cannot be read claims
/// nothing.
pub fn claims_tail(gguf: &str) -> bool {
    use crate::tensor::quantized::gguf_source::SplitGguf;
    SplitGguf::open(gguf)
        .map(|src| crate::inference::model::deepseek_v41::claims_gguf(&src.merged_content()))
        .unwrap_or(false)
}

/// Load the model at `gguf` from layer `from` and serve segment `[from, to)` on `listener` until the
/// head hangs up. `next` is the node running the layers after `to`; a final segment passes `None`
/// and replies with logits, an interior segment relays to `next`. Blocking: run it on its own
/// thread. The model is loaded here, inside the call, so it never crosses a thread boundary.
pub fn run_tail_local(
    gguf: &str,
    from: usize,
    to: usize,
    next: Option<std::net::SocketAddr>,
    listener: TcpListener,
) -> Result<(), String> {
    use crate::inference::model::deepseek_v41::model::DeepseekV41Model;
    use crate::inference::model::deepseek_v41::DeepseekV41Config;
    use crate::inference::offload::streamed::{Demand, Streamed};
    use crate::inference::offload::with_offload;
    use crate::tensor::quantized::gguf_source::SplitGguf;

    let src = SplitGguf::open(gguf).map_err(|e| format!("pipeline tail: open {gguf}: {e}"))?;
    let cfg = DeepseekV41Config::from_meta(&src)
        .map_err(|e| format!("pipeline tail: config {gguf}: {e}"))?;
    let model = DeepseekV41Model::load_range(&src, &cfg, from)
        .map_err(|e| format!("pipeline tail: load range from {from}: {e}"))?;
    let n = model.n_layers();
    if from == 0 || from >= n || to <= from || to > n {
        return Err(format!(
            "pipeline tail: segment [{from}, {to}) invalid for {n} layers"
        ));
    }
    // Only the experts this node's layers route are ever admitted, so the segment keeps its own
    // range resident and none of the head's - which is the point of splitting the stack.
    let streamed = Streamed::open(&Demand {
        always_read: model.resident_path_bytes(),
        transient: model.transient_bytes(64),
        concurrency: model.n_activated(),
        layers: model.n_layers(),
        prior: model.hot_experts(),
        fetch: &|l, id| model.expert_store(l)?.fetch(id).ok(),
    })
    .ok_or_else(|| "pipeline tail: no card for the placement".to_string())?;
    model.offload_experts(Some(streamed.lanes.clone()));
    let card = streamed.offload.clone();
    with_offload(card, || {
        model.serve_pipeline_segment(from, to, next, listener)
    })
    .map_err(|e| format!("pipeline tail: serve: {e}"))
}
