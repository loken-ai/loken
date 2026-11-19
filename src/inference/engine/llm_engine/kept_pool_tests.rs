//! A model that leaves the batch worker for the serial path keeps the worker's paged pool.
//! A serial prefill that needs the room of that pool takes it.

use super::load::try_spawn_continuous;
use crate::inference::engine::model_backend::{GenericBackend, ModelBackend};
use crate::tensor::{Device, Tensor};

/// A model whose serial prefill on a small card needs the room the kept pool holds.
const MODEL: &str = "qwen3:1.7b";

/// The blob an ollama tag names, `None` when the store does not hold its weights.
fn ollama_blob(tag: &str) -> Option<std::path::PathBuf> {
    let dir = crate::config::Config::load_test().get_ollama_models_dir();
    let (name, ver) = tag.split_once(':').unwrap_or((tag, "latest"));
    let text = std::fs::read_to_string(
        dir.join("manifests/registry.ollama.ai/library")
            .join(name)
            .join(ver),
    )
    .ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let digest = v["layers"]
        .as_array()?
        .iter()
        .find(|l| l["mediaType"] == "application/vnd.ollama.image.model")?["digest"]
        .as_str()?
        .replace(':', "-");
    let blob = dir.join("blobs").join(digest);
    blob.exists().then_some(blob)
}

/// The value of the architecture's `key` in the metadata of the GGUF at `path`.
fn arch_value(path: &std::path::Path, key: &str) -> usize {
    let g = crate::tensor::quantized::gguf_file::open_mapped(path).unwrap();
    let arch = g.content.metadata["general.architecture"]
        .to_string()
        .unwrap()
        .clone();
    g.content.metadata[&format!("{arch}.{key}")]
        .to_u32()
        .unwrap() as usize
}

/// The worker sizes its pool from the card's free memory, then the model leaves it for the
/// serial path with the pool kept.
/// A prompt whose cache needs more than the card has left is read serially, the kept pool given
/// back.
///
/// Ignored by default: it fills card 0 and needs the model in the store.
/// Run it alone with `a_serial_prefill_takes_the_room -- --ignored`.
#[test]
#[ignore]
fn a_serial_prefill_takes_the_room_of_the_pool_the_worker_kept() {
    if Device::new_cuda(0).is_err() {
        eprintln!("no card 0: skipped");
        return;
    }
    let Some(blob) = ollama_blob(MODEL) else {
        eprintln!("{MODEL} is not in the store: skipped");
        return;
    };
    let layers = arch_value(&blob, "block_count");
    let context = arch_value(&blob, "context_length");
    let model = crate::inference::generic_transformer::tiny_model::load_on_card(&blob, layers);
    let Ok(server) = try_spawn_continuous(MODEL, model, Vec::new(), context) else {
        eprintln!("the card holds no batch worker for {MODEL}: skipped");
        return;
    };
    let (model, pool) = server.park().unwrap();
    model.park_pool(pool.expect("the worker kept its pool"));
    let mut serial = GenericBackend(model);

    super::trim_cuda_pools();
    let free = serial.0.free_device_memory().unwrap();
    let (kv_layers, n_kv, head_dim) = serial.0.kv_layout();
    // K and V of every layer, in the F32 the unquantised cache of this model holds on the card.
    let row = (kv_layers * 2 * n_kv * head_dim * crate::tensor::DType::F32.size_in_bytes()) as u64;
    let rows = (free / row) as usize + 1;
    if rows >= context {
        eprintln!("the card leaves room for {rows} rows beside the pool: skipped");
        return;
    }
    let vocab = serial.0.paged_geometry().3 as u32;
    let prompt: Vec<u32> = (0..rows as u32).map(|i| (i * 7 + 13) % vocab).collect();
    let ids = Tensor::from_vec(prompt, (1, rows), &Device::Cpu).unwrap();
    let read = serial.forward_prefill_chunked(&ids, 0);
    assert!(
        read.is_ok(),
        "{rows} rows failed beside the kept pool: {:?}",
        read.err()
    );
    assert!(
        serial.0.take_parked_pool().is_none(),
        "the kept pool stayed on the card"
    );
}

/// The worker sizes its pool from the card's free memory, then the model leaves it for the
/// serial path with the pool kept.
/// 2 prompts handed over in a row, each restored as the request that asked for it does, both
/// stay in the snapshot table for that request.
///
/// Ignored by default: it fills card 0 and needs the model in the store.
/// Run it alone with `handed_over_prompts_land -- --ignored`.
#[test]
#[ignore]
fn handed_over_prompts_land_beside_the_pool_the_worker_kept() {
    if Device::new_cuda(0).is_err() {
        eprintln!("no card 0: skipped");
        return;
    }
    let Some(blob) = ollama_blob(MODEL) else {
        eprintln!("{MODEL} is not in the store: skipped");
        return;
    };
    let layers = arch_value(&blob, "block_count");
    let context = arch_value(&blob, "context_length");
    let model = crate::inference::generic_transformer::tiny_model::load_on_card(&blob, layers);
    let Ok(server) = try_spawn_continuous(MODEL, model, Vec::new(), context) else {
        eprintln!("the card holds no batch worker for {MODEL}: skipped");
        return;
    };
    let (model, pool) = server.park().unwrap();
    model.park_pool(pool.expect("the worker kept its pool"));
    let mut serial = GenericBackend(model);

    super::trim_cuda_pools();
    let free = serial.0.free_device_memory().unwrap();
    let (kv_layers, n_kv, head_dim) = serial.0.kv_layout();
    // K and V of every layer, in the F32 the rows of a quantised cache come in.
    let row = (kv_layers * 2 * n_kv * head_dim * crate::tensor::DType::F32.size_in_bytes()) as u64;
    // A handover the card holds beside the pool only once the pool is given back.
    let rows = ((free / row) as usize + 1).min(context / 2);
    let vocab = serial.0.paged_geometry().3 as u32;
    for round in 0..2u32 {
        let tokens: Vec<u32> = (0..rows as u32)
            .map(|i| (i * 7 + 13 + round) % vocab)
            .collect();
        let values: Vec<f32> = (0..rows * n_kv * head_dim)
            .map(|i| ((i % 251) as f32 - 125.0) / 128.0)
            .collect();
        let layer_rows = (0..kv_layers)
            .map(|_| Some((values.clone(), values.clone())))
            .collect();
        // Rows of a quantised cache come dequantised to F32, the import's code 1.
        serial
            .import_kv_snapshot(tokens.clone(), rows, layer_rows, 1)
            .unwrap_or_else(|e| panic!("handover {round} failed to import: {e}"));
        let (index, common) = serial
            .best_kv_snapshot(&tokens)
            .unwrap_or_else(|| panic!("handover {round} of {rows} rows was dropped"));
        assert_eq!(common, rows, "handover {round} holds part of its prompt");
        serial
            .restore_kv_snapshot(index)
            .unwrap_or_else(|e| panic!("handover {round} failed to restore: {e}"));
        // The request reads a token past the handed-over prompt from the restored cache.
        let last = Tensor::from_vec(vec![tokens[0]], (1, 1), &Device::Cpu).unwrap();
        serial
            .forward_prefill_chunked(&last, rows)
            .unwrap_or_else(|e| panic!("the request of handover {round} failed: {e}"));
    }
}
