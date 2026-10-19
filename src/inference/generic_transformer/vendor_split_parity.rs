//! A model split over an NVIDIA card and an AMD card answers as the host does: its first
//! layers on the GeForce through CUDA, the rest on the Radeon through HIP, the activations
//! moving between them through host memory.
//!
//! Ignored by default: it needs the model in the store and a card of each vendor. Run with
//!   cargo test --features cuda,rocm --profile gate split_over_two_vendors -- --ignored

use super::*;
use crate::inference::place::layer_executor::{DeviceKind, HeteroSegment};

/// The model the test reads from the store.
const TAG: &str = "qwen3:0.6b";

/// The tokens the greedy answer runs for.
const STEPS: usize = 16;

fn model_blob(tag: &str) -> std::path::PathBuf {
    let store = crate::config::Config::load_test().get_ollama_models_dir();
    let (name, tag) = tag.split_once(':').unwrap_or((tag, "latest"));
    let m = crate::inference::load::ollama_manager::OllamaManager::new(store.clone());
    let manifest = m
        .read_manifest_json(name, tag)
        .expect("manifest in the store");
    let digest = manifest["layers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|l| l["mediaType"] == "application/vnd.ollama.image.model")
        .and_then(|l| l["digest"].as_str())
        .expect("a model layer")
        .replace(':', "-");
    store.join("blobs").join(digest)
}

fn argmax(t: &Tensor) -> u32 {
    let v = t
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    v.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv {
                (i, x)
            } else {
                (bi, bv)
            }
        })
        .0 as u32
}

/// The greedy continuation of `prompt` by the model placed as `plan` says.
fn greedy(
    mmap: &std::sync::Arc<memmap2::Mmap>,
    cards: &HashMap<usize, Device>,
    plan: &HeteroPlan,
    prompt: &[u32],
) -> Vec<u32> {
    let read = || {
        gguf_file::Content::read_mapped(&mut std::io::Cursor::new(&mmap[..]), mmap.clone()).unwrap()
    };
    let mut m = GenericHeteroTransformer::from_gguf(read(), &mmap[..], cards, plan).unwrap();
    let ids = |t: &[u32]| Tensor::new(t, &Device::Cpu).unwrap().unsqueeze(0).unwrap();
    let mut tokens = vec![argmax(&m.forward(&ids(prompt), 0).unwrap())];
    for i in 0..STEPS - 1 {
        let next = m.forward(&ids(&tokens[i..=i]), prompt.len() + i).unwrap();
        tokens.push(argmax(&next));
    }
    tokens
}

#[test]
#[ignore = "needs the model in the store, an NVIDIA card and an AMD card"]
fn split_over_two_vendors() {
    let n = crate::inference::place::device_probe::card_count();
    let nvidia = (0..n).find(|&i| crate::tensor::gpu_runtime::is_nvidia_ordinal(i));
    let amd = (0..n).find(|&i| !crate::tensor::gpu_runtime::is_nvidia_ordinal(i));
    let (Some(nvidia), Some(amd)) = (nvidia, amd) else {
        eprintln!("needs a card of each vendor");
        return;
    };
    let path = model_blob(TAG);
    let file = std::fs::File::open(&path).unwrap();
    let mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&file).unwrap() });
    let n_layers = {
        let c = gguf_file::Content::read_mapped(&mut std::io::Cursor::new(&mmap[..]), mmap.clone())
            .unwrap();
        let arch = c.metadata["general.architecture"]
            .to_string()
            .unwrap()
            .clone();
        GenericTransformerConfig::from_gguf(&c, &arch)
            .unwrap()
            .n_layers
    };
    let cards: HashMap<usize, Device> = [nvidia, amd]
        .into_iter()
        .map(|i| (i, Device::new_cuda(i).expect("open the card")))
        .collect();
    let half = n_layers / 2;
    let segment = |kind, layer_start, layer_end| HeteroSegment {
        kind,
        layer_start,
        layer_end,
        free_memory_bytes: u64::MAX,
    };
    let split = HeteroPlan {
        segments: vec![
            segment(DeviceKind::Cuda(nvidia), 0, half),
            segment(DeviceKind::Cuda(amd), half, n_layers),
        ],
        tp_cards: None,
        total_layers: n_layers,
        host_experts_from: None,
    };
    let prompt: Vec<u32> = (0..24u32).map(|i| 1000 + (i * 37) % 20000).collect();
    let host = greedy(
        &mmap,
        &cards,
        &HeteroPlan::forced_gpu(n_layers, 0, 0),
        &prompt,
    );
    let both = greedy(&mmap, &cards, &split, &prompt);
    println!("{TAG}: host {host:?}");
    println!("{TAG}: cards {nvidia} and {amd}: {both:?}");
    assert_eq!(
        both, host,
        "the split model's greedy answer differs from the host's"
    );
}
