//! What the plan reserves for a Sinkhorn wide-residual model with latent attention, against what
//! the V4.1 path holds, on the synthetic model of the frozen V4.1 oracle.
//!
//! The V4.1 path keeps its caches as host vectors of f32 and sizes none of them ahead.
//! The proof is what both paths hold after the same tokens.

use super::block_scaled_oracle_tests::{checkpoint, generic_model, v41_model};
use super::wide_sparse_config::WideSparseMeta;
use super::wide_sparse_oracle_tests::{generic, toy_file, LAYOUTS};
use super::wide_sparse_plan::card_mixer_bytes;
use super::*;
use crate::inference::model::deepseek_v41::cache::DecodeState;
use crate::inference::model::deepseek_v41::model::oracle_gate::synthetic;
use crate::inference::model::deepseek_v41::model::tests::VOCAB;
use crate::inference::model::deepseek_v41::DeepseekV41Config;
use crate::inference::place::ladder::Demand;
use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan};
use crate::inference::place::llm::wide_sparse_demand;
use crate::inference::place::runtime_demand::{
    llm_work_reserve, tensor_load_staging, MEASURED_LOAD_FLOOR,
};

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

/// The bytes the real file's mixers hold on a card, over its 40 layers.
const REAL_MIXER_BYTES: u64 = 2_892_722_176;

/// The largest projection of the real file's mixers.
const REAL_LARGEST_UPLOAD: u64 = 34_406_400;

/// The layers of the real file.
const REAL_LAYERS: usize = 40;

/// The cache the plan reserves for a layer of the real file, over its planned window.
const REAL_KV_PER_LAYER: u64 = 8 << 20;

/// The width of the real file's residual stream.
const REAL_WIDTH: usize = 5120;

/// The largest tensor the load staged before the engram tables were left out: the table of an
/// engram layer, 256 bytes a row.
const ENGRAM_TABLE_BYTES: u64 = 384_006_168 * 256;

/// The demand of a file shaped like the real one: 40 layers whose mixers weigh 2.89 GB, no head
/// on a card (the head, the feed-forward and the experts stay on the host).
fn real_demand() -> Demand {
    Demand {
        layers: REAL_LAYERS,
        block_bytes: REAL_MIXER_BYTES,
        head_bytes: 0,
        kv_bytes_per_layer: REAL_KV_PER_LAYER,
        runtime_reserve: llm_work_reserve(REAL_WIDTH, 512),
    }
}

/// What a card of `free` bytes offers once the load's staging is kept free on it.
fn offered(free: u64) -> Vec<(usize, u64)> {
    let staging = tensor_load_staging(REAL_LARGEST_UPLOAD);
    vec![(0, free.saturating_sub(staging))]
}

/// Layers of `plan` on a card, and whether they are the first ones.
fn on_cards(plan: &HeteroPlan) -> (usize, bool) {
    let mut count = 0;
    let mut prefix = true;
    for s in &plan.segments {
        if matches!(s.kind, DeviceKind::Cuda(_)) {
            prefix &= s.layer_start == count;
            count += s.num_layers();
        }
    }
    (count, prefix)
}

/// The staging of the real file is its largest mixer projection, floored at what a first launch
/// needs, so it leaves most of a small card to the layers.
#[test]
fn the_staging_is_a_mixer_projection_not_an_engram_table() {
    assert!(REAL_LARGEST_UPLOAD < MEASURED_LOAD_FLOOR);
    assert_eq!(
        tensor_load_staging(REAL_LARGEST_UPLOAD),
        MEASURED_LOAD_FLOOR
    );
    assert_eq!(tensor_load_staging(ENGRAM_TABLE_BYTES), ENGRAM_TABLE_BYTES);
}

/// Layers of the real file a card of `capacity` bytes holds, a layer costing its mixer and its
/// cache: the plan fills the budget beside the reserve, then lets the reserve give way before it
/// sends a layer to the host.
fn layers_held(capacity: u64) -> usize {
    let per_layer = REAL_MIXER_BYTES / REAL_LAYERS as u64 + REAL_KV_PER_LAYER;
    ((capacity / per_layer) as usize).min(REAL_LAYERS)
}

/// A card of 3.8 GB holds all 40 layers beside a reduced reserve, and a card of 3 GB takes the
/// first 30 and leaves the other 10 to the host.
#[test]
fn a_small_card_takes_the_layers_that_fit_and_the_host_the_rest() {
    let demand = real_demand();
    for (free, want) in [(3_800_000_000u64, 40usize), (3_000_000_000, 30)] {
        let cards = offered(free);
        assert_eq!(cards[0].1, free - MEASURED_LOAD_FLOOR);
        assert_eq!(layers_held(cards[0].1), want, "{free} bytes free");
        // The whole model plus the reserve does not fit, so the plan is the spread one.
        assert!(demand.whole() > cards[0].1);
        let plan = demand.plan(&cards);
        assert_eq!(on_cards(&plan), (want, true), "{free} bytes free");
        let host: usize = plan
            .segments
            .iter()
            .filter(|s| matches!(s.kind, DeviceKind::Cpu))
            .map(|s| s.num_layers())
            .sum();
        assert_eq!(host, REAL_LAYERS - want);
        assert_eq!(plan.range(), (0, REAL_LAYERS));
    }
}

/// A card of 16 GB holds all 40 layers with the whole reserve beside them.
#[test]
fn a_large_card_takes_every_layer() {
    let cards = offered(16_000_000_000);
    let demand = real_demand();
    assert_eq!(
        demand.whole(),
        REAL_MIXER_BYTES + REAL_KV_PER_LAYER * REAL_LAYERS as u64 + demand.runtime_reserve
    );
    assert!(cards[0].1 >= demand.whole());
    let plan = demand.plan(&cards);
    assert_eq!(on_cards(&plan), (REAL_LAYERS, true));
    assert!(plan
        .segments
        .iter()
        .all(|s| matches!(s.kind, DeviceKind::Cuda(0))));
}

/// A card of 1 GB offers room for the first 5 layers and the rest stays on the host.
/// With the engram table as the staging, no card was offered at all.
#[test]
fn a_card_of_1_gb_takes_a_prefix() {
    let free = 1_000_000_000u64;
    let cards = offered(free);
    let demand = real_demand();
    assert_eq!(cards[0].1, free - MEASURED_LOAD_FLOOR);
    let held = layers_held(cards[0].1);
    assert_eq!(held, 5);
    let plan = demand.plan(&cards);
    assert_eq!(on_cards(&plan), (held, true));
    assert_eq!(plan.range(), (0, REAL_LAYERS));
    let before = free.saturating_sub(tensor_load_staging(ENGRAM_TABLE_BYTES));
    assert_eq!(before, 0);
}

/// The mixers of a synthetic file weigh what its mixer tensors weigh, and nothing else of a layer:
/// not the feed-forward, the experts, the residual mixing or the output projection.
#[test]
fn the_card_counts_the_mixer_tensors_and_nothing_else() {
    let window = [
        "attn_q_a",
        "attn_q_a_norm",
        "attn_q_b",
        "attn_kv",
        "attn_kv_a_norm",
        "attn_sinks",
        "attn_output_a",
    ];
    let band = [
        "attn_compressor_norm",
        "attn_compressor_kv",
        "attn_compressor_gate",
        "indexer.attn_q_b",
        "indexer.proj",
        "indexer.attn_k",
        "indexer.k_norm",
    ];
    for (ratios, kv_src, index_src) in LAYOUTS {
        let file = toy_file(ratios, kv_src, index_src);
        let meta = WideSparseMeta::from_file(&file.content).unwrap();
        let counted = card_mixer_bytes(&file.content).expect("a wide residual");
        assert_eq!(counted.per_layer.len(), meta.n_layers);
        // The toy file holds F32 values: 4 bytes each, as a card holds a dense weight.
        let values = |il: usize, stems: &[&str]| -> u64 {
            stems
                .iter()
                .map(|s| {
                    file.content
                        .tensor_infos
                        .get(&format!("blk.{il}.{s}"))
                        .map_or(0, |t| t.elem_count() as u64 * 4)
                })
                .sum()
        };
        for il in 0..meta.n_layers {
            assert!(values(il, &window) > 0);
            assert_eq!(
                counted.per_layer[il],
                values(il, &window) + values(il, &band),
                "layer {il} of {ratios:?}"
            );
        }
        let largest = counted.per_layer.iter().copied().max().unwrap();
        assert!(counted.largest_upload > 0 && counted.largest_upload <= largest);
        assert_eq!(counted.total(), counted.per_layer.iter().sum::<u64>());
    }
}

/// A file that is not a wide residual has no mixers to count.
#[test]
fn a_file_that_is_not_a_wide_residual_has_no_card_mixers() {
    let mut file = toy_file(LAYOUTS[0].0, LAYOUTS[0].1, LAYOUTS[0].2);
    file.content.metadata.clear();
    assert!(card_mixer_bytes(&file.content).is_none());
}

/// The split by rank refuses a wide residual from its file alone.
/// Its architecture has no entry in the facts table, and the refusal builds no generic config.
///
/// The 2 cards offered hold the real file's mixers only together: the plan has 2 card segments,
/// and the group of both cards holds a split's shares, so a load asks the file whether it splits.
#[test]
fn the_split_by_rank_refuses_a_wide_residual_before_its_config() {
    let demand = real_demand();
    let held = demand.whole() - demand.runtime_reserve;
    let gb = 1_000_000_000u64;
    let card = held / 2 + demand.runtime_reserve + gb;
    let cards = [(0, card), (1, card)];
    let plan = demand.plan(&cards);
    let on = |k: DeviceKind| plan.segments.iter().any(|s| s.kind == k);
    assert!(on(DeviceKind::Cuda(0)) && on(DeviceKind::Cuda(1)));
    assert!(!on(DeviceKind::Cpu));
    assert_eq!(
        crate::inference::place::llm::tensor_parallel(&demand, &cards, 2, true, |_| 1.0),
        Some(vec![0, 1])
    );
    let file = toy_file(LAYOUTS[0].0, LAYOUTS[0].1, LAYOUTS[0].2);
    for arch in ["deepseek4", "an architecture the facts table does not name"] {
        assert_eq!(
            super::tp_file_refusal(&file.content, arch, 2).as_deref(),
            Some("a wide residual")
        );
    }
}
