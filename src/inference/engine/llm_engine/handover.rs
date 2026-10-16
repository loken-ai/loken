//! Part of `impl LlmEngine`: a prompt read here for another instance, and a prompt whose reading
//! another instance did for this one.

use super::*;

impl LlmEngine {
    /// Reads `prompt` and answers the KV cache it leaves, the rows of the prompt's tokens, as the
    /// bytes of a handover (see `distributed::kv_handover`).
    pub async fn read_prompt_for_peer(&self, prompt: &str) -> AnyResult<Vec<u8>> {
        let params = GenerationParams {
            max_tokens: Some(1),
            temperature: Some(0.0),
            ..Default::default()
        };
        self.generate(prompt, params)
            .await
            .map_err(|e| anyhow!("{e}"))?;
        let model_state = self.model_state.clone();
        let prompt = prompt.to_string();
        tokio::task::spawn_blocking(move || -> AnyResult<Vec<u8>> {
            let guard = model_state.blocking_lock();
            let state = guard.as_ref().ok_or_else(|| anyhow!("no model loaded"))?;
            let tokens = state
                .tokenizer
                .encode(prompt.as_str(), true)
                .map_err(|e| anyhow!("tokenize: {e}"))?
                .get_ids()
                .to_vec();
            handover_of(state.model.as_ref(), &tokens)
        })
        .await?
    }
}

/// The handover of the snapshot holding `tokens`: the rows of the part of them it holds.
fn handover_of(
    model: &dyn crate::inference::engine::model_backend::ModelBackend,
    tokens: &[u32],
) -> AnyResult<Vec<u8>> {
    use crate::distributed::kv_handover::{encode, Handover, Header};
    if model.handover_carries_state() {
        // The snapshot's whole cache must be the prompt's own: a recurrent state, and a window
        // that ends past the prompt, cannot be cut back.
        let (held, state) = model
            .export_state_handover(tokens)?
            .ok_or_else(|| anyhow!("no snapshot holds the prompt"))?;
        return Ok(encode(&Handover {
            header: Header {
                tokens: tokens[..held].to_vec(),
                kv_len: held,
                dtype: 0,
                layout: kv_layout_id(model),
                layers: 0,
                state_len: state.len(),
            },
            rows: Vec::new(),
            state,
        }));
    }
    let (index, common) = model
        .best_kv_snapshot(tokens)
        .ok_or_else(|| anyhow!("no snapshot holds the prompt"))?;
    let (_, kv_len) = model
        .kv_snapshot_ref(index)
        .ok_or_else(|| anyhow!("kv snapshot {index} gone"))?;
    let held = common.min(kv_len);
    if held == 0 {
        return Err(anyhow!("the snapshot holds none of the prompt"));
    }
    if model
        .export_window_rows(index, held)?
        .iter()
        .any(Option::is_some)
    {
        return Err(anyhow!("a windowed layer's rows are not handed over"));
    }
    let rows = model.export_kv_rows(index, 0, held)?;
    Ok(encode(&Handover {
        header: Header {
            tokens: tokens[..held].to_vec(),
            kv_len: held,
            dtype: model.kv_snapshot_dtype(index).unwrap_or(0),
            layout: kv_layout_id(model),
            layers: rows.len(),
            state_len: 0,
        },
        rows,
        state: Vec::new(),
    }))
}

/// Has the instance at `peer` read `prompt` when its cache holds less of it than a prefill chunk
/// short of the whole, and imports the rows it hands back as a snapshot that the prefix reuse
/// then restores. Anything that goes wrong leaves the prompt to be read here.
pub(super) fn pull_prompt_from_peer(
    model: &mut dyn crate::inference::engine::model_backend::ModelBackend,
    model_name: &str,
    prompt: &str,
    prompt_tokens: &[u32],
    peer: &str,
) {
    use crate::distributed::kv_handover::decode;
    let carries_state = model.handover_carries_state();
    if !carries_state && (model.kv_layout().is_none() || !model.capabilities().trim_kv) {
        return;
    }
    let chunk = crate::inference::engine::model_backend::adaptive_prefill_chunk(
        crate::inference::engine::model_backend::PrefillShape::from_widest_ffn(model.widest_ffn()),
    );
    let held = if carries_state {
        model.state_snapshot_len(prompt_tokens).unwrap_or(0)
    } else {
        model.best_kv_snapshot(prompt_tokens).map_or(0, |(_, c)| c)
    };
    if prompt_tokens.len() < held + chunk {
        return;
    }
    let started = std::time::Instant::now();
    // A long prompt takes the peer as long as it takes: the request has no deadline of its own.
    let reply = reqwest::blocking::Client::builder()
        .timeout(None)
        .build()
        .and_then(|client| {
            client
                .post(format!(
                    "{}/api/cluster/prefill",
                    peer.trim_end_matches('/')
                ))
                .json(&serde_json::json!({ "model": model_name, "prompt": prompt }))
                .send()
        })
        .and_then(|r| r.error_for_status())
        .and_then(|r| r.bytes());
    let bytes = match reply {
        Ok(b) => b,
        Err(e) => {
            warn!("prefill peer {peer}: {e}; the prompt is read here");
            return;
        }
    };
    let handover = match decode(&bytes) {
        Ok(h) => h,
        Err(e) => {
            warn!("prefill peer {peer}: {e}; the prompt is read here");
            return;
        }
    };
    let header = &handover.header;
    if header.layout != kv_layout_id(model) {
        warn!("prefill peer {peer}: its cache has another shape; the prompt is read here");
        return;
    }
    if !prompt_tokens.starts_with(&header.tokens) {
        warn!(
            "prefill peer {peer}: it read other tokens than this prompt's; the prompt is read here"
        );
        return;
    }
    if carries_state != (header.state_len > 0) {
        warn!("prefill peer {peer}: it handed over another kind of state; the prompt is read here");
        return;
    }
    let (n, dtype, kv_len) = (header.tokens.len(), header.dtype, header.kv_len);
    if kv_len != n {
        warn!("prefill peer {peer}: its cache covers other positions than its tokens; the prompt is read here");
        return;
    }
    let imported = if carries_state {
        model.import_state_handover(handover.header.tokens, &handover.state)
    } else {
        model.import_kv_snapshot(handover.header.tokens, kv_len, handover.rows, dtype)
    };
    match imported {
        Ok(()) => info!(
            "prompt read by {peer}: {n} tokens, {:.1} MB handed over in {:.0} ms",
            bytes.len() as f64 / 1e6,
            started.elapsed().as_secs_f64() * 1e3
        ),
        Err(e) => warn!("prefill peer {peer}: import: {e}; the prompt is read here"),
    }
}
