//! What the cluster's planner reads of a wide-residual model on the generic route against the
//! V4.1 backend, and the cuts the planner makes from both.
//!
//! The small files are the frozen V4.1 oracle's, with a routing prior written into their
//! metadata.
//! The released file is read through its header only.

use super::block_draft_tests::{generic_of, v41_of};
use super::compressed_band::BandDims;
use super::wide_sparse_config::WideSparseMeta;
use super::wide_sparse_engram_tests::engram_file;
use super::wide_sparse_oracle_tests::toy_file;
use super::wide_sparse_routing_tests::{turning, ROUTED};
use super::wide_sparse_sibling as sibling;
use crate::distributed::auto_tune::auto_distribution;
use crate::distributed::cut_plan::{NodeProfile, TimePlan};
use crate::inference::engine::model_backend::{DeepseekV41Backend, GenericBackend, ModelBackend};
use crate::inference::generic_transformer::wide_sparse_toy::{real_entries_for, real_md_layout};
use crate::inference::model::deepseek_v41::model::DeepseekV41Model;
use crate::inference::model::deepseek_v41::DeepseekV41Config;
use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
use crate::tensor::quantized::gguf_file::{open_mapped, Content, MappedGguf, TensorInfo, Value};
use crate::tensor::quantized::gguf_source::SplitGguf;
use crate::tensor::quantized::GgmlDType;
use std::path::Path;

/// The shape a node times a missed expert against.
type CostShape = (usize, usize, GgmlDType, GgmlDType);

/// The model's inputs to the cut planner.
type PlanInputs = (Vec<u64>, u64, u64, u64);

/// The oracle's band layouts, the last one of several band layers.
const LAYOUTS: [(&[u32], &[u32], &[u32]); 3] = [
    (&[0, 2, 0], &[1], &[1]),
    (&[0, 2, 2, 0], &[1], &[1, 3]),
    (&[0, 2, 2, 2, 0], &[1, 2, 3], &[1, 2, 3]),
];

/// Per layer, a prior of a length that turns with the layer.
fn prior(layers: usize) -> Vec<Vec<u32>> {
    turning(layers, ROUTED as u32)
        .into_iter()
        .enumerate()
        .map(|(l, list)| list.into_iter().take(2 + l % 4).collect())
        .collect()
}

/// `file` with `lists` written into its metadata as the prior it carries.
fn with_prior(mut file: MappedGguf, lists: &[Vec<u32>]) -> MappedGguf {
    let arch = file
        .content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok().cloned())
        .expect("an architecture");
    for (l, list) in lists.iter().enumerate() {
        file.content.metadata.insert(
            format!("{arch}.hot_experts.{l}"),
            Value::Array(list.iter().map(|&e| Value::U32(e)).collect()),
        );
    }
    file
}

/// How a small file stores what the oracle's file stores.
#[derive(Clone, Copy, Debug)]
enum Form {
    /// The oracle's file: each layer's routed experts in 3 stacks.
    Stacks,
    /// The oracle's file with engram layers.
    Engram,
    /// The oracle's file with each routed expert in 3 tensors of its own.
    Units,
}

/// The forms every test runs over.
const FORMS: [Form; 3] = [Form::Stacks, Form::Engram, Form::Units];

/// `entries` with every stack of routed experts cut into 1 tensor per expert, named by its id.
fn as_units(entries: Vec<GgufEntry>) -> Vec<GgufEntry> {
    let stacks = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];
    entries
        .into_iter()
        .flat_map(|e| {
            if !stacks.iter().any(|s| e.name.ends_with(s)) {
                return vec![e];
            }
            let experts = e.dims[0];
            let size = e.data.len() / experts;
            (0..experts)
                .map(|x| GgufEntry {
                    name: format!("{}.{x}", e.name),
                    dims: e.dims[1..].to_vec(),
                    dtype: e.dtype,
                    data: e.data[x * size..(x + 1) * size].to_vec(),
                })
                .collect()
        })
        .collect()
}

/// The oracle's file in a band layout with each routed expert in tensors of its own, mapped.
fn unit_file(ratios: &[u32], kv_src: &[u32], index_src: &[u32]) -> MappedGguf {
    let bands: Vec<bool> = (0..ratios.len())
        .map(|l| ratios[l] != 0 || kv_src.contains(&(l as u32)) || index_src.contains(&(l as u32)))
        .collect();
    let md = real_md_layout(ratios, kv_src, index_src);
    let entries = as_units(real_entries_for(&bands));
    // Unique per call: cargo runs the tests in parallel.
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "loken-wide-cluster-units-{}-{seq}.gguf",
        std::process::id()
    ));
    write_gguf_with_metadata(&path, &md, &entries).unwrap();
    let g = open_mapped(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    g
}

/// The small file of `layout` in `form`, carrying `lists` as its prior.
fn toy(
    (ratios, kv_src, index_src): (&[u32], &[u32], &[u32]),
    form: Form,
    lists: &[Vec<u32>],
) -> MappedGguf {
    let file = match form {
        Form::Stacks => toy_file(ratios, kv_src, index_src),
        Form::Engram => engram_file(ratios, kv_src, index_src),
        Form::Units => unit_file(ratios, kv_src, index_src),
    };
    with_prior(file, lists)
}

/// The V4.1 model of `model`, with the prior beside it when the file carries none, as the V4.1
/// engine arm loads it.
pub(crate) fn v41_with_prior(model: &Path) -> DeepseekV41Model {
    let source = SplitGguf::open(model).unwrap();
    let mut cfg = DeepseekV41Config::from_meta(&source).unwrap();
    if cfg.hot_experts.iter().all(Vec::is_empty) {
        if let Some(prior) = sibling::prior_beside(model, cfg.n_layers, cfg.n_routed_experts) {
            cfg.hot_experts = prior;
        }
    }
    DeepseekV41Model::load(&source, &cfg).unwrap()
}

/// What the 3 hooks of a backend answer.
type Hooks = (
    Option<CostShape>,
    Option<PlanInputs>,
    Option<Vec<(usize, u64)>>,
);

/// The hooks of `backend`.
fn hooks(backend: &dyn ModelBackend) -> Hooks {
    (
        backend.expert_cost_shape(),
        backend.placement_model_inputs(),
        backend.card_residency(),
    )
}

/// What the planner reads of `hooks`: the cost shape and the placement inputs.
///
/// The card residency is left out.
/// On a build with cards, the V4.1 backend reports 0 bytes on card 0 for a small file on the
/// host.
fn planner(hooks: &Hooks) -> (Option<CostShape>, Option<PlanInputs>) {
    (hooks.0, hooks.1.clone())
}

/// The hooks of the generic backend and of the V4.1 backend over the same small file.
fn both(layout: (&[u32], &[u32], &[u32]), form: Form, lists: &[Vec<u32>]) -> [Hooks; 2] {
    let v41 = DeepseekV41Backend::new(v41_of(&toy(layout, form, lists)), 0, 128);
    let generic = GenericBackend(generic_of(toy(layout, form, lists)));
    [hooks(&generic), hooks(&v41)]
}

/// The node sets a test places a model on, sized from the model's own bytes.
///
/// `full` holds every layer's mixer and every expert of the prior's working set.
/// Each set is a list of node ids with their profiles, and the round trips between them.
fn node_sets(inputs: &PlanInputs) -> Vec<(Vec<(String, NodeProfile)>, Vec<(String, String, f64)>)> {
    let (attn, expert_bytes, working_set, _) = inputs;
    let full = attn.iter().sum::<u64>() + expert_bytes * working_set * attn.len() as u64;
    let node = |vram: u64, layer_ms: f64, miss_ms: f64| NodeProfile {
        vram,
        layer_ms,
        miss_ms,
    };
    let memories: [&[(u64, f64, f64)]; 4] = [
        &[(full / 2, 1.0, 0.5), (full / 2, 1.0, 0.5)],
        &[(full * 3 / 4, 1.0, 0.5), (full / 4, 1.0, 0.5)],
        &[
            (full / 2, 1.0, 0.5),
            (full / 3, 1.0, 0.5),
            (full / 6, 1.0, 0.5),
        ],
        &[(full, 1.0, 0.5), (full / 3, 2.0, 4.0), (full / 5, 1.0, 0.5)],
    ];
    memories
        .iter()
        .map(|set| {
            let nodes: Vec<(String, NodeProfile)> = set
                .iter()
                .enumerate()
                .map(|(i, &(vram, layer_ms, miss_ms))| {
                    (format!("n{i}"), node(vram, layer_ms, miss_ms))
                })
                .collect();
            let mut rtt = Vec::new();
            for a in 0..nodes.len() {
                for b in a + 1..nodes.len() {
                    rtt.push((nodes[a].0.clone(), nodes[b].0.clone(), 0.2));
                }
            }
            (nodes, rtt)
        })
        .collect()
}

/// The plan the handler forms from `inputs` over `nodes`.
fn plan(
    inputs: &PlanInputs,
    nodes: &[(String, NodeProfile)],
    rtt: &[(String, String, f64)],
) -> Option<TimePlan> {
    let (attn, expert_bytes, working_set, activated) = inputs;
    let rtt: Vec<(&str, &str, f64)> = rtt
        .iter()
        .map(|(a, b, r)| (a.as_str(), b.as_str(), *r))
        .collect();
    auto_distribution(
        attn,
        *expert_bytes,
        *working_set,
        *activated,
        nodes,
        &rtt,
        |_| true,
    )
}

/// Every layer of `n` covered once, in order.
fn covers(plan: &TimePlan, n: usize) -> bool {
    let mut next = 0;
    for s in &plan.segments {
        if s.start != next || s.end <= s.start {
            return false;
        }
        next = s.end;
    }
    next == n
}

/// The 3 hooks give the numbers of the V4.1 backend on every layout and form, and a model on
/// the host reports no card residency.
#[test]
fn the_generic_hooks_give_the_v41_numbers() {
    for layout in LAYOUTS {
        for form in FORMS {
            let label = format!("layout {:?}, {form:?}", layout.0);
            let lists = prior(layout.0.len());
            let [generic, v41] = both(layout, form, &lists);
            assert!(generic.0.is_some(), "{label}: a cost shape");
            assert!(generic.1.is_some(), "{label}: placement inputs");
            assert_eq!(generic.2, None, "{label}: nothing on a card");
            assert_eq!(planner(&generic), planner(&v41), "{label}");
            let (attn, _, working_set, activated) = generic.1.unwrap();
            assert_eq!(attn.len(), layout.0.len(), "{label}: 1 entry per layer");
            let lens: Vec<usize> = lists.iter().map(Vec::len).collect();
            assert_eq!(
                working_set as usize,
                lens.iter().sum::<usize>() / lens.len(),
                "{label}"
            );
            assert_eq!(
                activated, 2,
                "{label}: the oracle routes a token to 2 experts"
            );
        }
    }
}

/// A file without a prior gives the planner no inputs on either route, and still a cost shape.
#[test]
fn a_file_without_a_prior_gives_no_placement_inputs_on_either_route() {
    let [generic, v41] = both(LAYOUTS[0], Form::Stacks, &[]);
    assert!(generic.0.is_some());
    assert_eq!(generic.1, None);
    assert_eq!(planner(&generic), planner(&v41));
}

/// A mixture model that is not a wide residual declares nothing to the cluster.
#[test]
fn a_model_of_another_kind_declares_nothing() {
    use super::tiny_model;
    let path = tiny_model::write_moe("cluster-hooks", 2);
    let model = GenericBackend(tiny_model::load(&path, 2));
    assert_eq!(hooks(&model), (None, None, None));
    tiny_model::remove(&path);
}

/// The planner cuts a generic node's model where it cuts the V4.1 node's, on every node set,
/// and at least 1 set splits the stack.
#[test]
fn the_planner_cuts_a_generic_node_as_it_cuts_a_v41_node() {
    for layout in LAYOUTS {
        for form in FORMS {
            let label = format!("layout {:?}, {form:?}", layout.0);
            let lists = prior(layout.0.len());
            let [generic, v41] = both(layout, form, &lists);
            let (generic, v41) = (generic.1.unwrap(), v41.1.unwrap());
            let mut split = false;
            for (nodes, rtt) in node_sets(&generic) {
                let got = plan(&generic, &nodes, &rtt).expect("a plan");
                let want = plan(&v41, &nodes, &rtt).expect("a plan");
                assert_eq!(got, want, "{label}: nodes {nodes:?}");
                assert!(covers(&got, layout.0.len()), "{label}: {got:?}");
                split |= got.segments.len() > 1;
            }
            assert!(split, "{label}: some node set splits the stack");
        }
    }
}

/// A model whose token mixers run on card 0 reports the bytes they hold there, and gives the
/// planner the numbers it gives on the host.
#[cfg(feature = "gpu")]
#[test]
fn a_model_with_its_mixers_on_a_card_reports_the_bytes_they_hold_there() {
    use super::mixer::Mixer;
    use crate::inference::place::layer_executor::HeteroSegment;
    if crate::tensor::Device::new_cuda(0).is_err() {
        eprintln!("no card 0: skipped");
        return;
    }
    let layout = LAYOUTS[1];
    let lists = prior(layout.0.len());
    let layers = layout.0.len();
    let MappedGguf { content, mmap, .. } = toy(layout, Form::Stacks, &lists);
    let plan = super::HeteroPlan {
        segments: vec![HeteroSegment {
            kind: super::DeviceKind::Cuda(0),
            layer_start: 0,
            layer_end: layers,
            free_memory_bytes: 1 << 30,
        }],
        tp_cards: None,
        total_layers: layers,
        host_experts_from: None,
    };
    let model = super::GenericHeteroTransformer::from_gguf(
        content,
        &mmap[..],
        &std::collections::HashMap::new(),
        &plan,
    )
    .unwrap();
    let held: u64 = model
        .layers
        .iter()
        .map(|l| match &l.mixer {
            Mixer::LatentMqa(m) => m.card_bytes() as u64,
            Mixer::CompressedBand(b) => b.card_bytes() as u64,
            _ => 0,
        })
        .sum();
    assert!(held > 0, "the mixers are on the card");
    let card = GenericBackend(model);
    assert_eq!(card.card_residency(), Some(vec![(0, held)]));
    let [host, v41] = both(layout, Form::Stacks, &lists);
    assert_eq!(card.expert_cost_shape(), host.0);
    assert_eq!(card.placement_model_inputs(), host.1);
    assert_eq!(planner(&host), planner(&v41));
}

/// The inputs the header of `content` gives, by the rules the generic route holds its weights
/// by: a projection a host kernel reads as stored, any other weight as F32 values.
fn header_inputs(content: &Content) -> (CostShape, PlanInputs) {
    let meta = WideSparseMeta::from_file(content).unwrap();
    let info = |il: usize, stems: &[&str]| -> Option<&TensorInfo> {
        stems
            .iter()
            .flat_map(|s| [format!("blk.{il}.{s}"), format!("blk.{il}.{s}.weight")])
            .find_map(|n| content.tensor_infos.get(&n))
    };
    let dense = |t: &TensorInfo| t.elem_count() * std::mem::size_of::<f32>();
    let stored =
        |t: &TensorInfo| match crate::tensor::quant_view::native_view_supported(t.ggml_dtype) {
            true => t.size_in_bytes(),
            false => dense(t),
        };
    let need = |il: usize, stems: &[&str]| {
        info(il, stems).unwrap_or_else(|| panic!("blk.{il}.{}", stems[0]))
    };
    let mixers = (0..meta.n_layers)
        .map(|il| {
            let window = [
                &["attn_q_a"][..],
                &["attn_q_b"],
                &["attn_kv"],
                &["attn_output_a", "attn_o_a"],
                &["attn_output_b", "attn_o_b"],
            ];
            let mut bytes: usize = window.iter().map(|s| stored(need(il, s))).sum();
            if let Some(band) = BandDims::from_meta(&meta, il) {
                if band.kv_source {
                    bytes += dense(need(il, &["attn_compressor_kv"]));
                    bytes += info(il, &["attn_compressor_gate"]).map_or(0, dense);
                }
                if band.index_source {
                    bytes += stored(need(il, &["indexer.attn_q_b", "attn_indexer_q_b"]));
                    bytes += dense(need(il, &["indexer.proj", "attn_indexer_weights"]));
                    bytes += info(il, &["indexer.attn_k", "attn_indexer_k"]).map_or(0, dense);
                }
            }
            bytes as u64
        })
        .collect();
    // Expert 0 of layer 0: its rows of each stack, or its own tensors.
    let routed = meta.mixture.routed;
    let (gate, down, expert_bytes) = match info(0, &["ffn_gate_exps"]) {
        Some(gate) => {
            let (up, down) = (need(0, &["ffn_up_exps"]), need(0, &["ffn_down_exps"]));
            let stacks = gate.size_in_bytes() + up.size_in_bytes() + down.size_in_bytes();
            (gate, down, stacks / routed)
        }
        None => {
            let gate = need(0, &["ffn_gate_exps.0"]);
            let (up, down) = (need(0, &["ffn_up_exps.0"]), need(0, &["ffn_down_exps.0"]));
            (gate, down, stored(gate) + stored(up) + stored(down))
        }
    };
    // A stack is `[experts, inter, dim]`, an expert's own gate `[inter, dim]`.
    let dims = gate.shape.dims();
    let shape = (
        dims[dims.len() - 1],
        dims[dims.len() - 2],
        gate.ggml_dtype,
        down.ggml_dtype,
    );
    let lens: Vec<usize> = meta
        .mixture
        .hot
        .iter()
        .map(Vec::len)
        .filter(|&l| l > 0)
        .collect();
    let working_set = (lens.iter().sum::<usize>() / lens.len().max(1)) as u64;
    (
        shape,
        (
            mixers,
            expert_bytes as u64,
            working_set,
            meta.mixture.used as u64,
        ),
    )
}

/// The header alone gives the numbers both backends give on the small files, so the released
/// file's header stands for both routes.
#[test]
fn the_header_gives_the_numbers_of_both_routes() {
    for layout in LAYOUTS {
        for form in FORMS {
            let label = format!("layout {:?}, {form:?}", layout.0);
            let lists = prior(layout.0.len());
            let (shape, inputs) = header_inputs(&toy(layout, form, &lists).content);
            let [generic, v41] = both(layout, form, &lists);
            assert_eq!(Some(shape), generic.0, "{label}");
            assert_eq!(Some(&inputs), generic.1.as_ref(), "{label}");
            assert_eq!(planner(&generic), planner(&v41), "{label}");
        }
    }
}

/// The released file, by its header: 40 layers, 2 of them engram layers, its routed experts
/// as stored.
///
/// Its stack is split across 2 nodes of different memory, and across 3, where no node holds
/// the working set of every layer.
/// The cost of a missed expert is the one this machine's cores measure for the file's shape.
///
/// Ignored by default: it reads the header of the released file in the store.
///   cargo test --release --no-default-features --features cpu --lib -- --ignored released_file_header --nocapture
#[test]
#[ignore]
fn the_released_file_header_splits_across_nodes_of_different_memory() {
    let path = super::wide_sparse_real_tests::blob(super::wide_sparse_real_tests::TAG);
    let mut reader = std::io::BufReader::new(std::fs::File::open(&path).unwrap());
    let content = Content::read(&mut reader).unwrap();
    let meta = WideSparseMeta::from_file(&content).unwrap();
    assert_eq!(meta.n_layers, 40);
    assert_eq!(
        meta.ngram.as_ref().map(|n| n.layer_ids.len()),
        Some(2),
        "2 engram layers"
    );
    let (shape, inputs) = header_inputs(&content);
    let (attn, expert_bytes, working_set, activated) = &inputs;
    assert_eq!(attn.len(), 40);
    assert!(attn.iter().all(|&b| b > 0));
    assert!(*expert_bytes > 0 && *working_set > 0 && *activated > 0);
    let (dim, inter, gate_dt, down_dt) = shape;
    let profile = crate::distributed::auto_tune::local_profile(dim, inter, gate_dt, down_dt)
        .expect("this machine's cores time the file's expert");
    println!("shape {shape:?}, expert {expert_bytes} B, working set {working_set}, activated {activated}");
    println!(
        "mixers {} B, miss {:.3} ms, layer {:.3} ms",
        attn.iter().sum::<u64>(),
        profile.miss_ms,
        profile.layer_ms
    );
    let full = attn.iter().sum::<u64>() + expert_bytes * working_set * attn.len() as u64;
    let node = |vram: u64| NodeProfile { vram, ..profile };
    let sets: [&[u64]; 2] = [&[full / 2, full / 3], &[full / 2, full / 3, full / 4]];
    for set in sets {
        let nodes: Vec<(String, NodeProfile)> = set
            .iter()
            .enumerate()
            .map(|(i, &vram)| (format!("n{i}"), node(vram)))
            .collect();
        let mut rtt = Vec::new();
        for a in 0..nodes.len() {
            for b in a + 1..nodes.len() {
                rtt.push((nodes[a].0.clone(), nodes[b].0.clone(), 0.2));
            }
        }
        let got = plan(&inputs, &nodes, &rtt).expect("a plan");
        println!("{} nodes {set:?}: {got:?}", nodes.len());
        assert!(covers(&got, 40), "{got:?}");
        assert!(
            got.segments.len() > 1,
            "no node holds the working set: the stack is split"
        );
        let layers: Vec<usize> = nodes
            .iter()
            .map(|(id, _)| {
                got.segments
                    .iter()
                    .filter(|s| &s.node == id)
                    .map(|s| s.len())
                    .sum()
            })
            .collect();
        assert!(
            layers.windows(2).all(|w| w[0] >= w[1]),
            "a node of more memory runs as many layers or more: {layers:?}"
        );
    }
}
