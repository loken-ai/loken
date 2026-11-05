//! What the plan reserves for a Sinkhorn wide-residual model with latent attention, against what
//! the V4.1 path holds, on the synthetic model of the frozen V4.1 oracle.
//!
//! The V4.1 path keeps its caches as host vectors of f32 and sizes none of them ahead.
//! The proof is what both paths hold after the same tokens.

use super::block_scaled_oracle_tests::{checkpoint, generic_model, v41_model};
use super::wide_sparse_config::WideSparseMeta;
use super::wide_sparse_oracle_tests::{generic, toy_file, LAYOUTS};
use super::*;
use crate::inference::model::deepseek_v41::cache::DecodeState;
use crate::inference::model::deepseek_v41::model::oracle_gate::synthetic;
use crate::inference::model::deepseek_v41::model::tests::VOCAB;
use crate::inference::model::deepseek_v41::DeepseekV41Config;
use crate::inference::place::llm::wide_sparse_demand;

/// Prompt lengths: inside a compressed group, on a group boundary, past the window.
const PROMPTS: [usize; 6] = [1, 2, 3, 5, 33, 40];

/// Tokens decoded 1 at a time after each prompt.
const DECODED: usize = 3;

fn tokens(n: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 7 + 3) % VOCAB) as u32).collect()
}

/// Bytes each V4.1 layer holds: its window rows, compressed rows, index keys and filling group.
fn v41_bytes(state: &DecodeState) -> Vec<u64> {
    state
        .layers
        .iter()
        .map(|c| {
            let values = c.win.iter().map(Vec::len).sum::<usize>()
                + c.comp_kv.len()
                + c.index_k.len()
                + c.group_x.iter().map(Vec::len).sum::<usize>();
            (values * std::mem::size_of::<f32>()) as u64
        })
        .collect()
}

/// Bytes each generic layer's token mixer holds.
fn generic_bytes(model: &GenericHeteroTransformer) -> Vec<u64> {
    model
        .layers
        .iter()
        .map(|l| {
            let values = match &l.mixer {
                mixer::Mixer::LatentMqa(m) => m.state.as_ref().map_or(0, |s| s.window.len()),
                mixer::Mixer::CompressedBand(b) => b.state.as_ref().map_or(0, |s| {
                    s.window.window.len()
                        + s.compressed.len()
                        + s.index_keys.len()
                        + s.filling.len()
                }),
                _ => 0,
            };
            (values * std::mem::size_of::<f32>()) as u64
        })
        .collect()
}

/// After every prompt, and after each token decoded after it, each layer of both paths holds
/// the bytes `cache_bytes` gives, and the plan reserves the largest layer's for every layer.
#[test]
fn every_layer_holds_the_cache_bytes_the_v41_path_holds() {
    let mut differ = Vec::new();
    for (ratios, kv_src, index_src) in LAYOUTS {
        let meta = WideSparseMeta::from_file(&toy_file(ratios, kv_src, index_src).content).unwrap();
        let (v41, _) = synthetic(ratios, kv_src, index_src, 0);
        for n in PROMPTS {
            let toks = tokens(n + DECODED);
            let mut state = v41.new_decode_state();
            v41.prefill_into(&toks[..n], &mut state).unwrap();
            let mut model = generic(ratios, kv_src, index_src);
            let prompt = Tensor::from_vec(toks[..n].to_vec(), (1, n), &Device::Cpu).unwrap();
            model.forward(&prompt, 0).unwrap();
            let mut check = |held: usize, want: Vec<u64>, got: Vec<u64>| {
                let planned: Vec<u64> = (0..meta.n_layers)
                    .map(|il| meta.cache_bytes(il, held))
                    .collect();
                let at = format!("{ratios:?} at {held} tokens");
                if want != got || want != planned {
                    differ.push(format!(
                        "{at}: V4.1 {want:?}, generic {got:?}, planned {planned:?}"
                    ));
                }
                if meta.kv_bytes_per_layer(held) != want.iter().copied().max().unwrap_or(0) {
                    differ.push(format!("{at}: the plan's layer is not the largest"));
                }
            };
            check(n, v41_bytes(&state), generic_bytes(&model));
            for (i, &t) in toks[n..].iter().enumerate() {
                v41.forward_decode(t, &mut state).unwrap();
                let step = Tensor::from_vec(vec![t], (1, 1), &Device::Cpu).unwrap();
                model.forward(&step, n + i).unwrap();
                check(n + i + 1, v41_bytes(&state), generic_bytes(&model));
            }
        }
    }
    assert!(
        differ.is_empty(),
        "the cache bytes differ:\n{}",
        differ.join("\n")
    );
}

/// A stored state takes the bytes per token the V4.1 configuration gives, for every layout.
#[test]
fn a_stored_state_takes_the_v41_bytes_per_token() {
    for (ratios, kv_src, index_src) in LAYOUTS {
        let file = toy_file(ratios, kv_src, index_src);
        let meta = WideSparseMeta::from_file(&file.content).unwrap();
        let cfg = DeepseekV41Config::from_gguf(&file.content).unwrap();
        assert_eq!(
            meta.state_bytes_per_token(),
            cfg.state_bytes_per_token(),
            "{ratios:?}"
        );
    }
}

/// The weights every token reads weigh the same in the generic model as in the V4.1 model of
/// the same file, for every layout.
#[test]
fn the_generic_model_reads_the_bytes_of_the_v41_resident_path() {
    for (ratios, kv_src, index_src) in LAYOUTS {
        let (v41, _) = synthetic(ratios, kv_src, index_src, 0);
        let model = generic(ratios, kv_src, index_src);
        assert_eq!(
            model.resident_path_bytes(),
            Some(v41.resident_path_bytes() as u64),
            "{ratios:?}"
        );
    }
}

/// The weights every token reads weigh the same in the generic model as in the V4.1 model of
/// the block-scaled checkpoint: fp8, fp4 and bf16 matrices as stored, the dense ones as F32.
#[test]
fn the_checkpoint_model_reads_the_bytes_of_the_v41_resident_path() {
    let dir = checkpoint("resident");
    let v41 = v41_model(&dir);
    let model = generic_model(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(
        model.resident_path_bytes(),
        Some(v41.resident_path_bytes() as u64)
    );
}

/// The demand of a wide-residual header reserves its largest layer's cache for every block, and
/// a plan without cards places every block on the host exactly once.
#[test]
fn the_demand_reserves_the_largest_layer_cache_for_every_block() {
    for (ratios, kv_src, index_src) in LAYOUTS {
        let meta = WideSparseMeta::from_file(&toy_file(ratios, kv_src, index_src).content).unwrap();
        for n in PROMPTS {
            let demand = wide_sparse_demand(&meta, 1 << 20, 1 << 10, n, 0);
            let largest = (0..meta.n_layers)
                .map(|il| meta.cache_bytes(il, n))
                .max()
                .unwrap();
            assert_eq!(demand.kv_bytes_per_layer, largest, "{ratios:?} at {n}");
            assert_eq!(demand.layers, meta.n_layers);
            let plan = demand.plan(&[]);
            assert_eq!(plan.range(), (0, meta.n_layers), "{ratios:?} at {n}");
            assert!(plan
                .segments
                .iter()
                .all(|s| matches!(s.kind, DeviceKind::Cpu)));
        }
    }
}
