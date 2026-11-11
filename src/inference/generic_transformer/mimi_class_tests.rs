//! A transformer with no token table and no final norm, its blocks normed with a bias,
//! projecting Q, K and V as one matrix, rotating adjacent lanes, windowed, scaled on both
//! residual branches and feeding forward through the error-function GELU of 2 linears.
//! It is loaded from a codec's file and held to a plain host run in F64.
//!
//! The toys are the transformers of Mimi's decoder and encoder, under both names a codec's
//! file gives their Q, K and V matrix.
//!
//! The same blocks unwindowed and unscaled, followed by a norm with a bias, are the
//! transformer of pocket-tts's flow model.
//! Its toy is held to the plain run as a prefill and as a prefill followed by single steps.

use super::class_load::{load_class_held, Holding, Rows};
use super::codec_class_tests::{error, host};
use super::*;
use crate::inference::generic_transformer::class_load::tests::{scratch, values};
use serde_json::{json, Value as Json};

/// How a codec's file names a transformer's tensors.
#[derive(Debug, Clone, Copy)]
struct Spelling {
    /// The class of the transformer.
    class: &'static str,
    /// The prefix of its blocks.
    blocks: &'static str,
    /// The name of a block's Q, K and V matrix.
    qkv: &'static str,
    /// The path of the blocks under the model's prefix the caller gives, when it is not the
    /// layout's own.
    under: Option<&'static str>,
    /// Each block scales both residual branches.
    scaled: bool,
    /// The prefix of the norm with a bias that follows the last block, when there is one.
    norm: Option<&'static str>,
}

/// The decoder's transformer, its Q, K and V matrix a parameter.
const DECODER: Spelling = Spelling {
    class: "ProjectedTransformer",
    blocks: "decoder_transformer.transformer.layers",
    qkv: "self_attn.in_proj_weight",
    under: None,
    scaled: true,
    norm: None,
};

/// The encoder's transformer under a model's prefix, its Q, K and V matrix a linear's weight.
const ENCODER: Spelling = Spelling {
    class: "ProjectedTransformer",
    blocks: "mimi.encoder_transformer.transformer.layers",
    qkv: "self_attn.in_proj.weight",
    under: Some("encoder_transformer.transformer.layers"),
    scaled: true,
    norm: None,
};

/// The transformer of pocket-tts's flow model and the norm that follows it.
const FLOW: Spelling = Spelling {
    class: "FlowLMModel",
    blocks: "flow_lm.transformer.layers",
    qkv: "self_attn.in_proj.weight",
    under: None,
    scaled: false,
    norm: Some("flow_lm.out_norm"),
};

/// The sizes of the toy's transformer.
#[derive(Debug, Clone, Copy)]
struct Geometry {
    hidden: usize,
    heads: usize,
    ffn: usize,
    layers: usize,
    /// The rows a row attends to, itself included: every row before it when unwindowed.
    window: usize,
    theta: f64,
    eps: f64,
}

impl Geometry {
    fn of(root: &Json) -> Self {
        let u = |k: &str| root[k].as_u64().unwrap() as usize;
        let f = |k: &str| root[k].as_f64().unwrap();
        Self {
            hidden: u("hidden_size"),
            heads: u("num_attention_heads"),
            ffn: u("intermediate_size"),
            layers: u("num_hidden_layers"),
            window: root["sliding_window"]
                .as_u64()
                .map_or(usize::MAX, |w| w as usize),
            theta: f("rope_theta"),
            eps: f("layer_norm_eps"),
        }
    }

    fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }
}

/// A configuration of `layers` layers of a window of `window`, of the class of `sp`.
/// A window of 0 leaves the layers unwindowed.
fn toy_root(sp: Spelling, layers: usize, window: usize) -> Json {
    let mut root = json!({
        "architectures": [sp.class],
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "intermediate_size": 64,
        "num_hidden_layers": layers,
        "rope_theta": 10000.0,
        "layer_norm_eps": 1e-5
    });
    if window > 0 {
        root["sliding_window"] = json!(window);
    }
    root
}

/// The tensors of geometry `g` spelled as `sp`, by name: shape, centre and spread of the
/// values.
///
/// A tensor outside the transformer stands beside it, as in a codec's file.
fn tensors(g: &Geometry, sp: Spelling) -> Vec<(String, Vec<usize>, f32, f32)> {
    let (h, f) = (g.hidden, g.ffn);
    let mut out = vec![(
        "upsample.convtr.weight".to_string(),
        vec![h, 1, 4],
        0.0,
        0.2,
    )];
    for il in 0..g.layers {
        for (part, dims, centre, scale) in [
            ("norm1.weight", vec![h], 1.0, 0.2),
            ("norm1.bias", vec![h], 0.0, 0.2),
            ("norm2.weight", vec![h], 1.0, 0.2),
            ("norm2.bias", vec![h], 0.0, 0.2),
            (sp.qkv, vec![3 * h, h], 0.0, 0.2),
            ("self_attn.out_proj.weight", vec![h, h], 0.0, 0.2),
            ("linear1.weight", vec![f, h], 0.0, 0.2),
            ("linear2.weight", vec![h, f], 0.0, 0.2),
            ("layer_scale_1.scale", vec![h], 0.5, 0.4),
            ("layer_scale_2.scale", vec![h], 0.5, 0.4),
        ] {
            if part.starts_with("layer_scale") && !sp.scaled {
                continue;
            }
            out.push((format!("{}.{il}.{part}", sp.blocks), dims, centre, scale));
        }
    }
    if let Some(norm) = sp.norm {
        out.push((format!("{norm}.weight"), vec![h], 1.0, 0.2));
        out.push((format!("{norm}.bias"), vec![h], 0.0, 0.2));
    }
    out
}

/// The toy written as a codec's file in F32, and its weights as the reference reads them.
struct Toy {
    dir: std::path::PathBuf,
    root: Json,
    sp: Spelling,
    weights: HashMap<String, Vec<f32>>,
}

impl Toy {
    fn write(tag: &str, sp: Spelling, layers: usize, window: usize) -> Self {
        let dir = scratch(tag);
        let root = toy_root(sp, layers, window);
        let mut seed = 17u32;
        let mut weights = HashMap::new();
        let mut stored = Vec::new();
        for (name, dims, centre, scale) in tensors(&Geometry::of(&root), sp) {
            let v = values(&mut seed, dims.iter().product(), centre, scale);
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            stored.push((name.clone(), dims, bytes));
            weights.insert(name, v);
        }
        let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = stored
            .iter()
            .map(|(name, dims, bytes)| {
                let view = safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    dims.clone(),
                    bytes,
                )
                .unwrap();
                (name.clone(), view)
            })
            .collect();
        safetensors::serialize_to_file(views, None, &dir.join("model.safetensors")).unwrap();
        Self {
            dir,
            root,
            sp,
            weights,
        }
    }

    fn file(&self) -> String {
        self.dir
            .join("model.safetensors")
            .to_string_lossy()
            .to_string()
    }

    /// The layers placed on `cards`, a cache of `rows` rows.
    fn load(&self, rows: usize, cards: &[(usize, u64)]) -> GenericHeteroTransformer {
        load_class_held(
            &self.root,
            &[self.file()],
            self.sp.class,
            Rows::Exactly(rows),
            0,
            cards,
            Holding {
                blocks: self.sp.under,
                wide_floats: false,
            },
        )
        .unwrap()
        .model
    }
}

impl Drop for Toy {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// `x` layer-normed, scaled by `g` and shifted by `b`.
fn layer_norm(x: &[f64], g: &[f32], b: &[f32], eps: f64) -> Vec<f64> {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let r = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(g.iter().zip(b))
        .map(|(v, (g, b))| (v - mean) * r * *g as f64 + *b as f64)
        .collect()
}

/// `w [out, in]` times `x [in]`.
fn mm(w: &[f32], x: &[f64]) -> Vec<f64> {
    w.chunks(x.len())
        .map(|row| row.iter().zip(x).map(|(a, b)| *a as f64 * b).sum())
        .collect()
}

/// Each head of `x` rotated at position `pos`: lane `2j` with lane `2j + 1`.
fn rope(x: &mut [f64], d: usize, theta: f64, pos: usize) {
    for head in x.chunks_mut(d) {
        for j in 0..d / 2 {
            let a = pos as f64 * theta.powf(-((2 * j) as f64) / d as f64);
            let (c, s) = (a.cos(), a.sin());
            let (x0, x1) = (head[2 * j], head[2 * j + 1]);
            head[2 * j] = x0 * c - x1 * s;
            head[2 * j + 1] = x1 * c + x0 * s;
        }
    }
}

/// The error-function GELU.
fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + crate::inference::cache::turboquant::codebook::erf(x / 2f64.sqrt()))
}

/// The layers spelled as `sp` over `x`, rows of `hidden` values at positions `0..`, as F64.
///
/// A row attends to itself and the `window - 1` rows before it.
/// Each branch of a scaled spelling is scaled by channel before the residual add.
/// The spelling's norm follows the last layer when it has one.
fn reference_rows(
    w: &HashMap<String, Vec<f32>>,
    g: &Geometry,
    sp: Spelling,
    x: &[f64],
) -> Vec<f64> {
    let (h, d) = (g.hidden, g.head_dim());
    let s = x.len() / h;
    let mut rows: Vec<Vec<f64>> = x.chunks(h).map(|r| r.to_vec()).collect();
    let unscaled = vec![1f32; h];
    for il in 0..g.layers {
        let l = |n: &str| &w[&format!("{}.{il}.{n}", sp.blocks)];
        let scale = |n: &str| match sp.scaled {
            true => l(n),
            false => &unscaled,
        };
        let mut q = Vec::with_capacity(s);
        let mut k = Vec::with_capacity(s);
        let mut v = Vec::with_capacity(s);
        for (pos, row) in rows.iter().enumerate() {
            let n = layer_norm(row, l("norm1.weight"), l("norm1.bias"), g.eps);
            let qkv = mm(l(sp.qkv), &n);
            let mut qi = qkv[..h].to_vec();
            let mut ki = qkv[h..2 * h].to_vec();
            rope(&mut qi, d, g.theta, pos);
            rope(&mut ki, d, g.theta, pos);
            q.push(qi);
            k.push(ki);
            v.push(qkv[2 * h..].to_vec());
        }
        for i in 0..s {
            let first = (i + 1).saturating_sub(g.window);
            let mut out = vec![0f64; h];
            for hd in 0..g.heads {
                let lanes = hd * d..(hd + 1) * d;
                let scores: Vec<f64> = (first..=i)
                    .map(|j| {
                        q[i][lanes.clone()]
                            .iter()
                            .zip(&k[j][lanes.clone()])
                            .map(|(a, b)| a * b)
                            .sum::<f64>()
                            / (d as f64).sqrt()
                    })
                    .collect();
                let top = scores.iter().cloned().fold(f64::MIN, f64::max);
                let e: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
                let sum: f64 = e.iter().sum();
                for (jj, j) in (first..=i).enumerate() {
                    for (o, vv) in out[lanes.clone()].iter_mut().zip(&v[j][lanes.clone()]) {
                        *o += e[jj] / sum * vv;
                    }
                }
            }
            let att = mm(l("self_attn.out_proj.weight"), &out);
            for ((r, a), g) in rows[i]
                .iter_mut()
                .zip(&att)
                .zip(scale("layer_scale_1.scale"))
            {
                *r += a * *g as f64;
            }
        }
        for row in rows.iter_mut() {
            let n = layer_norm(row, l("norm2.weight"), l("norm2.bias"), g.eps);
            let up: Vec<f64> = mm(l("linear1.weight"), &n).into_iter().map(gelu).collect();
            let down = mm(l("linear2.weight"), &up);
            for ((r, a), g) in row.iter_mut().zip(&down).zip(scale("layer_scale_2.scale")) {
                *r += a * *g as f64;
            }
        }
    }
    if let Some(norm) = sp.norm {
        let (gain, bias) = (&w[&format!("{norm}.weight")], &w[&format!("{norm}.bias")]);
        for row in rows.iter_mut() {
            *row = layer_norm(row, gain, bias, g.eps);
        }
    }
    rows.concat()
}

/// The tolerance on a value of magnitude `scale` out of `g`'s layers, computed in a float type
/// of machine epsilon `eps`.
///
/// A product of `n` terms rounds to about `eps * sqrt(n)` of its scale.
/// A layer chains 4 products: the Q, K and V projection, the attention output, the up and the
/// down projections.
/// The roundings of the chain add up.
fn tolerance(g: &Geometry, eps: f64, scale: f64) -> f64 {
    let widest = g.ffn.max(3 * g.hidden);
    eps * (widest as f64).sqrt() * (4 * g.layers + 1) as f64 * scale
}

/// Every row of a prefill of `n` rows, the layers' cache holding `cache` rows, within the
/// tolerance of the plain host run.
fn every_row_matches(toy: &Toy, n: usize, cache: usize, cards: &[(usize, u64)], eps: f64) {
    let g = Geometry::of(&toy.root);
    let mut seed = 5u32;
    let x = values(&mut seed, n * g.hidden, 0.0, 1.0);
    let want = reference_rows(
        &toy.weights,
        &g,
        toy.sp,
        &x.iter().map(|v| *v as f64).collect::<Vec<_>>(),
    );
    let mut model = toy.load(cache, cards);
    let xs = Tensor::from_vec(x, (n, g.hidden), &Device::Cpu).unwrap();
    let rows = model.forward_embeddings_rows(&xs).unwrap();
    assert_eq!(rows.dims(), [n, g.hidden]);
    let (err, scale) = error(&host(&rows), &want);
    let tol = tolerance(&g, eps, scale);
    eprintln!("every row {err} away at a scale of {scale}, tolerance {tol}");
    assert!(err <= tol, "every row {err} away, tolerance {tol}");
}

/// The rows of a prefill longer than the window and than a prefill chunk.
fn long(g: &Geometry) -> usize {
    crate::inference::engine::llm_engine::PREFILL_CHUNK_TOKENS + 2 * g.window + 3
}

/// The layers place on the host: no token table, no final norm, a layer norm with a bias,
/// one Q, K and V projection, adjacent rotary lanes, the error-function GELU of an ungated
/// feed-forward, every layer windowed and both branches scaled.
#[test]
fn a_tableless_layer_normed_class_matches_a_plain_host_run() {
    let toy = Toy::write("mimi-host", DECODER, 2, 5);
    let g = Geometry::of(&toy.root);
    let model = toy.load(64, &[]);
    assert!(model.config.flags.use_rope_i && model.config.flags.erf_gelu);
    assert!(model.config.flags.is_phi2_simple_ffn && model.config.flags.layer_norm_with_bias);
    assert!(matches!(model.output_norm, WeightedNorm::Identity(_)));
    for l in model.layers() {
        assert_eq!(l.sliding_window, Some(5));
        assert!(l.attn_qkv.is_some() && l.ffn_gate.is_none());
        assert!(l.attn_layer_scale.is_some() && l.ffn_layer_scale.is_some());
    }
    drop(model);
    every_row_matches(&toy, long(&g), long(&g), &[], f32::EPSILON as f64);
}

/// A cache held to fewer rows than the prefill gives the same windowed run.
#[test]
fn a_prefill_longer_than_the_cache_is_the_windowed_run() {
    let toy = Toy::write("mimi-short-cache", DECODER, 2, 5);
    let g = Geometry::of(&toy.root);
    every_row_matches(&toy, long(&g), g.window, &[], f32::EPSILON as f64);
}

/// The same run with the layers on the cards.
/// The layers keep their F32 weights there, and their rows stay within the rounding of F32.
#[cfg(feature = "gpu")]
#[test]
fn the_card_tableless_layer_normed_class_matches_a_plain_host_run() {
    let cards = crate::inference::place::ladder::cards(0);
    if cards.is_empty() {
        eprintln!("no card");
        return;
    }
    let toy = Toy::write("mimi-card", DECODER, 2, 5);
    let g = Geometry::of(&toy.root);
    let model = toy.load(long(&g), &cards);
    assert!(
        !model.layer_cuda_ordinals().is_empty(),
        "the toy sits on a card"
    );
    drop(model);
    every_row_matches(&toy, long(&g), long(&g), &cards, f32::EPSILON as f64);
}

/// The encoder's transformer under a model's prefix, its Q, K and V matrix a linear's weight,
/// gives the same run on the host.
#[test]
fn an_encoder_spelled_class_matches_a_plain_host_run() {
    let toy = Toy::write("mimi-encoder-host", ENCODER, 2, 5);
    let g = Geometry::of(&toy.root);
    let model = toy.load(64, &[]);
    for l in model.layers() {
        assert_eq!(l.sliding_window, Some(5));
        assert!(l.attn_qkv.is_some() && l.attn_layer_scale.is_some());
    }
    drop(model);
    every_row_matches(&toy, long(&g), g.window, &[], f32::EPSILON as f64);
}

/// The same run with the layers on the cards.
#[cfg(feature = "gpu")]
#[test]
fn the_card_encoder_spelled_class_matches_a_plain_host_run() {
    let cards = crate::inference::place::ladder::cards(0);
    if cards.is_empty() {
        eprintln!("no card");
        return;
    }
    let toy = Toy::write("mimi-encoder-card", ENCODER, 2, 5);
    let g = Geometry::of(&toy.root);
    every_row_matches(&toy, long(&g), long(&g), &cards, f32::EPSILON as f64);
}

/// The rows of an unwindowed prefill longer than a prefill chunk.
fn long_unwindowed() -> usize {
    crate::inference::engine::llm_engine::PREFILL_CHUNK_TOKENS + 3
}

/// A prefill of `p` rows, then a step for each of the `n - p` rows after it, the layers' cache
/// holding `n` rows.
/// Each step's normed row is within the tolerance of the plain host run's row at its position.
fn every_step_matches(toy: &Toy, p: usize, n: usize, cards: &[(usize, u64)], eps: f64) {
    let g = Geometry::of(&toy.root);
    let mut seed = 9u32;
    let x = values(&mut seed, n * g.hidden, 0.0, 1.0);
    let want = reference_rows(
        &toy.weights,
        &g,
        toy.sp,
        &x.iter().map(|v| *v as f64).collect::<Vec<_>>(),
    );
    let mut model = toy.load(n, cards);
    let xs = Tensor::from_vec(x, (n, g.hidden), &Device::Cpu).unwrap();
    let first = model
        .forward_embeddings_prefill(&xs.narrow(0, 0, p).unwrap(), false)
        .unwrap();
    let mut got = host(&first.normed);
    for pos in p..n {
        let row = xs.narrow(0, pos, 1).unwrap();
        let step = model.forward_embedding_step(&row, pos, false).unwrap();
        assert_eq!(step.normed.dims(), [1, g.hidden]);
        got.extend(host(&step.normed));
    }
    let (err, scale) = error(&got, &want[(p - 1) * g.hidden..]);
    let tol = tolerance(&g, eps, scale);
    eprintln!("every step {err} away at a scale of {scale}, tolerance {tol}");
    assert!(err <= tol, "every step {err} away, tolerance {tol}");
}

/// The flow model's transformer places on the host: unwindowed and unscaled blocks, and its
/// norm with a bias after the last one.
/// A prefill gives every row, and a prefill followed by single steps gives every last row.
#[test]
fn an_unwindowed_class_with_a_final_layer_norm_matches_a_plain_host_run() {
    let toy = Toy::write("flow-host", FLOW, 2, 0);
    let n = long_unwindowed();
    let model = toy.load(n, &[]);
    assert!(model.config.flags.use_rope_i && model.config.flags.erf_gelu);
    assert!(model.config.flags.layer_norm_with_bias);
    assert!(matches!(model.output_norm, WeightedNorm::Layer(_)));
    for l in model.layers() {
        assert_eq!(l.sliding_window, None);
        assert!(l.attn_layer_scale.is_none() && l.ffn_layer_scale.is_none());
    }
    drop(model);
    every_row_matches(&toy, n, n, &[], f32::EPSILON as f64);
    every_step_matches(&toy, 7, n, &[], f32::EPSILON as f64);
}

/// The same runs with the layers on the cards, within the rounding of F32.
#[cfg(feature = "gpu")]
#[test]
fn the_card_unwindowed_class_with_a_final_layer_norm_matches_a_plain_host_run() {
    let cards = crate::inference::place::ladder::cards(0);
    if cards.is_empty() {
        eprintln!("no card");
        return;
    }
    let toy = Toy::write("flow-card", FLOW, 2, 0);
    let n = long_unwindowed();
    let model = toy.load(n, &cards);
    assert!(
        !model.layer_cuda_ordinals().is_empty(),
        "the toy sits on a card"
    );
    drop(model);
    every_row_matches(&toy, n, n, &cards, f32::EPSILON as f64);
    every_step_matches(&toy, 7, n, &cards, f32::EPSILON as f64);
}
