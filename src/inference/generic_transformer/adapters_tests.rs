//! Adapters a request selects on one loaded base.
//!
//! The base is the tiny Llama checkpoint of `tests/vectors/hf/llama`, loaded at the `reference`
//! level so its weights stay F32 as stored.
//! The adapters are written by each test in the PEFT layout.
//! An adapter selected on the base must give the logits of the checkpoint merged with it.

use super::*;
use crate::inference::engine::llm_engine::GenericWeights;
use crate::inference::engine::model_backend::{GenericBackend, ModelBackend};
use crate::inference::load::lora::{text_adapter_bytes_in, AdapterWidths, ADAPTER_SITES};
use crate::inference::place::layer_executor::HeteroPlan;
use safetensors::tensor::TensorView;

/// The prompt every test reads, and the tokens decoded after it.
const PROMPT: [u32; 6] = [1, 17, 42, 99, 5, 73];
const NEXT: [u32; 2] = [8, 64];

fn vector_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/hf/llama")
}

/// A directory of its own for each call.
fn scratch(tag: &str) -> std::path::PathBuf {
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "loken-adapters-{tag}-{}-{call}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A float tensor by name: its shape and values.
type Floats = std::collections::BTreeMap<String, (Vec<usize>, Vec<f32>)>;

fn read_floats(path: &std::path::Path) -> Floats {
    let bytes = std::fs::read(path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    st.tensors()
        .into_iter()
        .map(|(name, view)| {
            assert_eq!(view.dtype(), safetensors::Dtype::F32, "{name}");
            let values = view
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            (name, (view.shape().to_vec(), values))
        })
        .collect()
}

fn write_floats(path: &std::path::Path, tensors: &Floats) {
    let bytes: Vec<(String, Vec<usize>, Vec<u8>)> = tensors
        .iter()
        .map(|(n, (shape, v))| {
            let b = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            (n.clone(), shape.clone(), b)
        })
        .collect();
    let views: Vec<(String, TensorView)> = bytes
        .iter()
        .map(|(n, shape, b)| {
            let view = TensorView::new(safetensors::Dtype::F32, shape.clone(), b).unwrap();
            (n.clone(), view)
        })
        .collect();
    safetensors::serialize_to_file(views, None, path).unwrap();
}

/// The base checkpoint's tensors.
fn base() -> Floats {
    read_floats(&vector_dir().join("model.safetensors"))
}

fn layer_count() -> usize {
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(vector_dir().join("config.json")).unwrap())
            .unwrap();
    config["num_hidden_layers"].as_u64().unwrap() as usize
}

fn weight_name(layer: usize, projection: &str) -> String {
    format!("model.layers.{layer}.{projection}.weight")
}

/// An adapter of rank `rank` on the projections `sites` of [`ADAPTER_SITES`], its values drawn
/// from `seed`, with `alpha` stored when given.
struct Adapter {
    rank: usize,
    sites: Vec<usize>,
    seed: u32,
    alpha: Option<f32>,
}

impl Adapter {
    /// Its tensors in the PEFT layout: `down` `[r, in]` and `up` `[out, r]` of each projection.
    fn tensors(&self, base: &Floats) -> Floats {
        let mut seed = self.seed;
        let mut draw = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((seed >> 16) as f32 / 65536.0 - 0.5) * 0.5
                })
                .collect()
        };
        let mut out = Floats::new();
        for layer in 0..layer_count() {
            for &site in &self.sites {
                let projection = ADAPTER_SITES[site].0;
                let (shape, _) = &base[&weight_name(layer, projection)];
                let (rows, cols) = (shape[0], shape[1]);
                let key = format!("base_model.model.model.layers.{layer}.{projection}");
                let down = draw(self.rank * cols);
                let up = draw(rows * self.rank);
                out.insert(
                    format!("{key}.lora_A.weight"),
                    (vec![self.rank, cols], down),
                );
                out.insert(format!("{key}.lora_B.weight"), (vec![rows, self.rank], up));
                if let Some(a) = self.alpha {
                    out.insert(format!("{key}.alpha"), (vec![1], vec![a]));
                }
            }
        }
        out
    }

    /// `alpha / r`, or 1 without `alpha`.
    fn scale(&self) -> f32 {
        self.alpha.map_or(1.0, |a| a / self.rank as f32)
    }

    /// `base` with this adapter merged at `strength`: `W + scale * strength * up . down`.
    fn merge_into(&self, base: &mut Floats, strength: f32) {
        let tensors = self.tensors(base);
        let s = self.scale() * strength;
        for layer in 0..layer_count() {
            for &site in &self.sites {
                let projection = ADAPTER_SITES[site].0;
                let key = format!("base_model.model.model.layers.{layer}.{projection}");
                let (_, down) = &tensors[&format!("{key}.lora_A.weight")];
                let (_, up) = &tensors[&format!("{key}.lora_B.weight")];
                let (shape, w) = base.get_mut(&weight_name(layer, projection)).unwrap();
                let (rows, cols, r) = (shape[0], shape[1], self.rank);
                for i in 0..rows {
                    for j in 0..cols {
                        let d: f32 = (0..r).map(|k| up[i * r + k] * down[k * cols + j]).sum();
                        w[i * cols + j] += s * d;
                    }
                }
            }
        }
    }

    fn write(&self, dir: &std::path::Path, name: &str) {
        write_floats(
            &dir.join(format!("{name}.safetensors")),
            &self.tensors(&base()),
        );
    }
}

/// Attention alone, as PEFT trains by default, without `alpha`.
fn attention_adapter() -> Adapter {
    Adapter {
        rank: 2,
        sites: vec![0, 2],
        seed: 11,
        alpha: None,
    }
}

/// Every projection, at `alpha = 2 r`.
fn full_adapter() -> Adapter {
    Adapter {
        rank: 3,
        sites: (0..ADAPTER_SITES.len()).collect(),
        seed: 29,
        alpha: Some(6.0),
    }
}

/// The checkpoint `tensors` loaded on the host as the server loads a dense checkpoint, at the
/// level that keeps its weights as stored.
fn load(tensors: &Floats) -> GenericHeteroTransformer {
    use crate::inference::engine::llm_engine::FloatWeights;
    use crate::inference::generic_transformer::set_float_weights;
    use crate::inference::load::awq_loader as awql;
    let dir = scratch("model");
    std::fs::copy(vector_dir().join("config.json"), dir.join("config.json")).unwrap();
    std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    write_floats(&dir.join("model.safetensors"), tensors);
    let dir_str = dir.to_string_lossy().to_string();
    let _level = super::LEVEL_LOCK.write().unwrap_or_else(|e| e.into_inner());
    set_float_weights(FloatWeights::AsStored);
    let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let (_, shards) = awql::checkpoint_shards(&dir).unwrap();
        let layers = awql::parse_config_json(&dir_str).unwrap().n_layers;
        let plan = HeteroPlan::calculate_with_kv(layers, 1 << 20, &[], &[], 1.0, 0);
        GenericWeights::Awq {
            dir: &dir_str,
            shards: &shards,
        }
        .build(
            &HashMap::new(),
            &plan,
            crate::inference::engine::llm_engine::KvQuant::Off,
            None,
            false,
            false,
        )
    }));
    set_float_weights(FloatWeights::ByPlan);
    let _ = std::fs::remove_dir_all(&dir);
    built
        .unwrap_or_else(|e| std::panic::resume_unwind(e))
        .unwrap_or_else(|e| panic!("load failed: {e}"))
}

fn floats_of(t: &Tensor) -> Vec<f32> {
    t.flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// The logits of the prompt, then of each next token fed in turn, from an empty cache.
fn run(model: &mut GenericHeteroTransformer) -> Vec<Vec<f32>> {
    model.trim_kv(0);
    let mut out = Vec::new();
    let x = Tensor::new(&PROMPT[..], &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    out.push(floats_of(&model.forward(&x, 0).unwrap()));
    for (i, t) in NEXT.iter().enumerate() {
        let x = Tensor::new(&[*t], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        out.push(floats_of(&model.forward(&x, PROMPT.len() + i).unwrap()));
    }
    out
}

/// The largest distance between two runs, and the largest logit of `b`.
fn distance(a: &[Vec<f32>], b: &[Vec<f32>]) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let mut far = 0f32;
    let mut top = 0f32;
    for (x, y) in a.iter().zip(b) {
        assert_eq!(x.len(), y.len());
        for (p, q) in x.iter().zip(y) {
            far = far.max((p - q).abs());
            top = top.max(q.abs());
        }
    }
    (far, top)
}

/// The runs agree within the rounding of F32 sums taken in another order.
fn assert_close(what: &str, got: &[Vec<f32>], want: &[Vec<f32>]) {
    let (far, top) = distance(got, want);
    assert!(
        far <= 1e-4 * top.max(1.0),
        "{what}: logits {far} apart, the largest is {top}"
    );
}

/// The runs differ by more than any rounding.
fn assert_apart(what: &str, a: &[Vec<f32>], b: &[Vec<f32>]) {
    let (far, top) = distance(a, b);
    assert!(
        far > 1e-2 * top.max(1.0),
        "{what}: logits only {far} apart, the largest is {top}"
    );
}

/// Writes the two adapters in a directory of their own and keeps them resident on `model`.
fn load_both(model: &mut GenericHeteroTransformer) -> std::path::PathBuf {
    let dir = scratch("dir");
    attention_adapter().write(&dir, "attention");
    full_adapter().write(&dir, "full");
    for name in ["attention", "full"] {
        model
            .load_adapter(name, &dir.join(format!("{name}.safetensors")))
            .unwrap();
    }
    dir
}

/// A request that selects no adapter answers as the base did before any adapter was read,
/// bit for bit, before and after another request selected one.
#[test]
fn a_request_without_adapters_runs_on_the_base_alone() {
    let mut model = load(&base());
    let before = run(&mut model);
    let dir = load_both(&mut model);
    model.select_adapters(&[]).unwrap();
    assert_eq!(run(&mut model), before, "resident adapters left unselected");
    model.select_adapters(&[("full".to_string(), 1.0)]).unwrap();
    assert_apart("the full adapter", &run(&mut model), &before);
    model.select_adapters(&[]).unwrap();
    assert!(model.selected_adapters().is_empty());
    assert_eq!(run(&mut model), before, "after a request with an adapter");
    let _ = std::fs::remove_dir_all(dir);
}

/// Each adapter, at each strength, gives the logits of the checkpoint merged with it.
#[test]
fn an_adapter_gives_the_logits_of_the_merged_checkpoint() {
    let mut model = load(&base());
    let dir = load_both(&mut model);
    for (name, adapter, strength) in [
        ("attention", attention_adapter(), 1.0f32),
        ("full", full_adapter(), 1.0),
        ("full", full_adapter(), 0.5),
    ] {
        model
            .select_adapters(&[(name.to_string(), strength)])
            .unwrap();
        let got = run(&mut model);
        let mut merged = base();
        adapter.merge_into(&mut merged, strength);
        let want = run(&mut load(&merged));
        assert_close(&format!("{name} at {strength}"), &got, &want);
    }
    let _ = std::fs::remove_dir_all(dir);
}

/// One loaded base serves requests with different adapters, each with its own answer, through
/// the backend a request reaches.
/// 2 adapters selected together give the checkpoint merged with both.
#[test]
fn requests_with_different_adapters_share_one_base() {
    let mut model = load(&base());
    let base_run = run(&mut model);
    let dir = load_both(&mut model);
    let mut backend = GenericBackend(model);
    let mut answer = |wanted: &[(&str, f32)]| {
        let wanted: Vec<(String, f32)> = wanted.iter().map(|(n, s)| (n.to_string(), *s)).collect();
        backend.select_adapters(&wanted).unwrap();
        let out = run(&mut backend.0);
        backend.select_adapters(&[]).unwrap();
        out
    };
    let first = answer(&[("attention", 1.0)]);
    let second = answer(&[("full", 1.0)]);
    let both = answer(&[("attention", 1.0), ("full", 1.0)]);
    let base_again = answer(&[]);
    let first_again = answer(&[("attention", 1.0)]);
    assert_apart("the 2 adapters", &first, &second);
    assert_apart("the attention adapter and the base", &first, &base_run);
    assert_eq!(base_again, base_run, "the base between 2 adapters");
    assert_eq!(first_again, first, "the same adapter a second time");

    let mut merged = base();
    attention_adapter().merge_into(&mut merged, 1.0);
    assert_close("attention", &first, &run(&mut load(&merged)));
    full_adapter().merge_into(&mut merged, 1.0);
    assert_close("both", &both, &run(&mut load(&merged)));
    let mut merged = base();
    full_adapter().merge_into(&mut merged, 1.0);
    assert_close("full", &second, &run(&mut load(&merged)));

    assert_eq!(backend.0.resident_adapters(), ["attention", "full"]);
    let _ = std::fs::remove_dir_all(dir);
}

/// The placement counts the bytes the resident adapters hold, from the files' headers alone.
/// An adapter of another width or of an image model counts nothing.
#[test]
fn the_placement_counts_the_adapter_bytes() {
    let mut model = load(&base());
    let dir = load_both(&mut model);
    let held: Vec<(crate::tensor::DeviceLocation, u64)> = model.resident_adapter_bytes();
    assert_eq!(held.len(), 1, "every layer is on the host: {held:?}");
    let held = held[0].1;

    let tensors = base();
    let rows = |site: usize| tensors[&weight_name(0, ADAPTER_SITES[site].0)].0[0];
    // This checkpoint loads with q, k and v fused into 1 projection.
    let qkv = rows(0) + rows(1) + rows(2);
    // `fused_ffn` counts the up of a gate or up pair at the width of a fused gate and up.
    let pair_values = |a: &Adapter, fused_ffn: bool| -> usize {
        let mut n = 0;
        for layer in 0..layer_count() {
            for &site in &a.sites {
                let (shape, _) = &tensors[&weight_name(layer, ADAPTER_SITES[site].0)];
                let up_width = match site {
                    0..=2 => qkv,
                    4 | 5 if fused_ffn => 2 * shape[0],
                    _ => shape[0],
                };
                n += a.rank * (up_width + shape[1]);
            }
        }
        n
    };
    let bytes = |fused_ffn: bool| -> u64 {
        let values =
            pair_values(&attention_adapter(), fused_ffn) + pair_values(&full_adapter(), fused_ffn);
        (values * DType::F32.size_in_bytes()) as u64
    };
    // This checkpoint keeps its gate and up apart.
    assert_eq!(held, bytes(false), "the bytes the resident adapters hold");
    let want = bytes(true);

    let hidden = tensors["model.embed_tokens.weight"].0[1];
    let widths = AdapterWidths { hidden, qkv };
    let layers = layer_count();
    let narrow: Floats = attention_adapter()
        .tensors(&tensors)
        .into_iter()
        .map(|(n, (shape, v))| {
            let shape: Vec<usize> = shape
                .iter()
                .map(|&d| if d == hidden { d / 2 } else { d })
                .collect();
            let count = shape.iter().product();
            (n, (shape, v[..count].to_vec()))
        })
        .collect();
    write_floats(&dir.join("narrow.safetensors"), &narrow);
    let image: Floats = [
        (
            "lora_unet_mid_block_attn1_to_q.lora_down.weight",
            vec![2, hidden],
        ),
        (
            "lora_unet_mid_block_attn1_to_q.lora_up.weight",
            vec![hidden, 2],
        ),
    ]
    .into_iter()
    .map(|(n, s)| {
        let count = s.iter().product();
        (n.to_string(), (s, vec![0.5; count]))
    })
    .collect();
    write_floats(&dir.join("image.safetensors"), &image);

    assert_eq!(text_adapter_bytes_in(&dir, 0..layers, widths), want);
    assert_eq!(
        text_adapter_bytes_in(&dir, 0..1, widths) * 2,
        want,
        "one layer of 2"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// A graph is kept under the generation the backend reports when it was recorded.
/// A change of the selected adapters moves that generation, so no graph recorded on one
/// selection is replayed on another.
/// A request without adapters after another without adapters keeps the graphs.
#[test]
fn a_change_of_adapters_moves_the_graph_generation() {
    let model = load(&base());
    let mut backend = GenericBackend(model);
    let dir = load_both(&mut backend.0);
    let base_at = backend.graph_generation();
    assert!(base_at.is_some(), "the model keys its graphs");
    backend.select_adapters(&[]).unwrap();
    assert_eq!(backend.graph_generation(), base_at, "base after base");

    backend
        .select_adapters(&[("attention".to_string(), 1.0)])
        .unwrap();
    let attention_at = backend.graph_generation();
    assert_ne!(attention_at, base_at, "an adapter selected");
    backend
        .select_adapters(&[("full".to_string(), 1.0)])
        .unwrap();
    let full_at = backend.graph_generation();
    assert_ne!(full_at, attention_at, "another adapter selected");
    backend.select_adapters(&[]).unwrap();
    let cleared_at = backend.graph_generation();
    assert_ne!(cleared_at, full_at, "the selection cleared");
    assert_ne!(cleared_at, base_at, "the base again, after an adapter");
    let _ = std::fs::remove_dir_all(dir);
}

/// An adapter on the gate or the up of a layer whose load fused them into one projection
/// corrects the columns of that part of the fused outputs and leaves the other part as it was.
#[test]
fn an_adapter_reaches_a_gate_and_up_the_load_fused() {
    let layers = 2;
    let path = super::tiny_model::write_quant("adapters-fused-ffn", layers);
    let mut model = super::tiny_model::load(&path, layers);
    assert!(
        model.layers[0].ffn_gate.is_none(),
        "the load fuses the gate and the up of this checkpoint"
    );
    let ffn = model.layers[0].ffn_up.out_dim() / 2;
    let hidden = model.config.embedding_length;
    let rank = 2;
    let dir = scratch("fused-ffn");
    let mut seed = 5u32;
    let mut draw = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 16) as f32 / 65536.0 - 0.5) * 0.5
            })
            .collect()
    };
    let mut pairs = Vec::new();
    for (name, site) in [("gate", 4usize), ("up", 5)] {
        let key = format!("base_model.model.model.layers.0.{}", ADAPTER_SITES[site].0);
        let (down, up) = (draw(rank * hidden), draw(ffn * rank));
        let tensors: Floats = [
            (
                format!("{key}.lora_A.weight"),
                (vec![rank, hidden], down.clone()),
            ),
            (
                format!("{key}.lora_B.weight"),
                (vec![ffn, rank], up.clone()),
            ),
        ]
        .into_iter()
        .collect();
        write_floats(&dir.join(format!("{name}.safetensors")), &tensors);
        model
            .load_adapter(name, &dir.join(format!("{name}.safetensors")))
            .unwrap();
        pairs.push((name, site, down, up));
    }

    let x: Vec<f32> = (0..hidden).map(|i| (i % 7) as f32 / 7.0 - 0.4).collect();
    let xt = Tensor::from_vec(x.clone(), (1, 1, hidden), &Device::Cpu).unwrap();
    let base = floats_of(&model.layers[0].ffn_up.forward(&xt).unwrap());
    for (name, site, down, up) in pairs {
        model.select_adapters(&[(name.to_string(), 1.0)]).unwrap();
        let got = floats_of(&model.layers[0].ffn_up.forward(&xt).unwrap());
        model.select_adapters(&[]).unwrap();
        // `x . A^T . B^T`, the file's `[r, in]` and `[out, r]`, without `alpha`.
        let xa: Vec<f32> = (0..rank)
            .map(|k| (0..hidden).map(|j| down[k * hidden + j] * x[j]).sum())
            .collect();
        let delta: Vec<f32> = (0..ffn)
            .map(|i| (0..rank).map(|k| up[i * rank + k] * xa[k]).sum())
            .collect();
        let at = (site - 4) * ffn;
        for c in 0..2 * ffn {
            let want = if (at..at + ffn).contains(&c) {
                base[c] + delta[c - at]
            } else {
                base[c]
            };
            let far = (got[c] - want).abs();
            assert!(
                far <= 1e-4 * want.abs().max(1.0),
                "{name}: column {c} gives {} for {want}",
                got[c]
            );
            if !(at..at + ffn).contains(&c) {
                assert_eq!(
                    got[c], base[c],
                    "{name}: column {c} of the other part moved"
                );
            }
        }
    }
    super::tiny_model::remove(&path);
    let _ = std::fs::remove_dir_all(dir);
}
