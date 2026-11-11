//! A transformer whose layers scale both residual branches by channel and window their
//! attention, loaded from a codec's file and held to a plain host run in F64.
//!
//! The toy is the transformer of a codec's decoder: no token table, no head, every row read.

use super::class_load::{load_class_from, Rows};
use super::*;
use crate::inference::generic_transformer::class_load::tests::{scratch, values};
use serde_json::{json, Value as Json};

/// The class of the toy.
pub(crate) const CODEC_CLASS: &str = "Qwen3TTSTokenizerV2DecoderTransformerModel";

/// The prefix of the transformer's tensors in the codec's file.
pub(crate) const PREFIX: &str = "decoder.pre_transformer.";

/// The sizes of a codec's transformer, read from its `decoder_config`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Geometry {
    pub latent: usize,
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub layers: usize,
    pub window: usize,
    pub theta: f64,
    pub eps: f64,
}

impl Geometry {
    pub(crate) fn of(root: &Json) -> Self {
        let d = &root["decoder_config"];
        let u = |k: &str| d[k].as_u64().unwrap() as usize;
        let f = |k: &str| d[k].as_f64().unwrap();
        Self {
            latent: u("latent_dim"),
            hidden: u("hidden_size"),
            heads: u("num_attention_heads"),
            head_dim: u("head_dim"),
            ffn: u("intermediate_size"),
            layers: u("num_hidden_layers"),
            window: u("sliding_window"),
            theta: f("rope_theta"),
            eps: f("rms_norm_eps"),
        }
    }
}

/// A codec configuration whose transformer has `layers` layers of a window of `window`.
pub(crate) fn toy_root(layers: usize, window: usize) -> Json {
    json!({
        "architectures": ["Qwen3TTSTokenizerV2Model"],
        "decoder_config": {
            "latent_dim": 24,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 16,
            "intermediate_size": 48,
            "num_hidden_layers": layers,
            "sliding_window": window,
            "rope_theta": 10000,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 4096,
            "hidden_act": "silu",
            "attention_bias": false
        }
    })
}

/// The transformer's tensors of geometry `g`, by name: shape, centre and spread of the values.
fn tensors(g: &Geometry) -> Vec<(String, Vec<usize>, f32, f32)> {
    let (h, q, f, l) = (g.hidden, g.heads * g.head_dim, g.ffn, g.latent);
    let mut out = vec![
        (format!("{PREFIX}input_proj.weight"), vec![h, l], 0.0, 0.2),
        (format!("{PREFIX}input_proj.bias"), vec![h], 0.0, 0.1),
        (format!("{PREFIX}output_proj.weight"), vec![l, h], 0.0, 0.2),
        (format!("{PREFIX}output_proj.bias"), vec![l], 0.0, 0.1),
        (format!("{PREFIX}norm.weight"), vec![h], 1.0, 0.2),
    ];
    for il in 0..g.layers {
        for (part, dims, centre, scale) in [
            ("input_layernorm.weight", vec![h], 1.0, 0.2),
            ("post_attention_layernorm.weight", vec![h], 1.0, 0.2),
            ("self_attn.q_proj.weight", vec![q, h], 0.0, 0.2),
            ("self_attn.k_proj.weight", vec![q, h], 0.0, 0.2),
            ("self_attn.v_proj.weight", vec![q, h], 0.0, 0.2),
            ("self_attn.o_proj.weight", vec![h, q], 0.0, 0.2),
            ("mlp.gate_proj.weight", vec![f, h], 0.0, 0.2),
            ("mlp.up_proj.weight", vec![f, h], 0.0, 0.2),
            ("mlp.down_proj.weight", vec![h, f], 0.0, 0.2),
            ("self_attn_layer_scale.scale", vec![h], 0.5, 0.4),
            ("mlp_layer_scale.scale", vec![h], 0.5, 0.4),
        ] {
            out.push((format!("{PREFIX}layers.{il}.{part}"), dims, centre, scale));
        }
    }
    out
}

/// The toy written as a codec's file in F32, and its weights as the reference reads them.
pub(crate) struct Toy {
    pub dir: std::path::PathBuf,
    pub root: Json,
    pub weights: HashMap<String, Vec<f32>>,
}

impl Toy {
    pub(crate) fn write(tag: &str, layers: usize, window: usize) -> Self {
        let dir = scratch(tag);
        let root = toy_root(layers, window);
        let mut seed = 11u32;
        let mut weights = HashMap::new();
        let mut stored = Vec::new();
        for (name, dims, centre, scale) in tensors(&Geometry::of(&root)) {
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
        std::fs::write(dir.join("config.json"), root.to_string()).unwrap();
        Self { dir, root, weights }
    }

    /// The codec's file.
    pub(crate) fn file(&self) -> String {
        self.dir
            .join("model.safetensors")
            .to_string_lossy()
            .to_string()
    }

    /// The layers placed on `cards`, a cache of `rows` rows.
    fn load(&self, rows: usize, cards: &[(usize, u64)]) -> GenericHeteroTransformer {
        load_class_from(
            &self.root,
            &[self.file()],
            CODEC_CLASS,
            Rows::Exactly(rows),
            0,
            cards,
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

/// `x` RMS-normed and scaled by `g`.
fn rms(x: &[f64], g: &[f32], eps: f64) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let r = 1.0 / (ms + eps).sqrt();
    x.iter().zip(g).map(|(v, g)| v * r * *g as f64).collect()
}

/// `w [out, in]` times `x [in]`.
fn mm(w: &[f32], x: &[f64]) -> Vec<f64> {
    w.chunks(x.len())
        .map(|row| row.iter().zip(x).map(|(a, b)| *a as f64 * b).sum())
        .collect()
}

/// Each head of `x` rotated at position `pos`: lane `j` with lane `j + d/2`.
fn rope(x: &mut [f64], g: &Geometry, pos: usize) {
    let (d, half) = (g.head_dim, g.head_dim / 2);
    for head in x.chunks_mut(d) {
        for j in 0..half {
            let a = pos as f64 * g.theta.powf(-((2 * j) as f64) / d as f64);
            let (c, s) = (a.cos(), a.sin());
            let (x0, x1) = (head[j], head[j + half]);
            head[j] = x0 * c - x1 * s;
            head[j + half] = x1 * c + x0 * s;
        }
    }
}

/// The layers and the final norm over `x`, rows of `hidden` values at positions `0..`, as F64.
///
/// `w` names the transformer's tensors under [`PREFIX`].
/// A row attends to itself and the `window - 1` rows before it.
/// Each branch is scaled by channel before the residual add.
pub(crate) fn reference_rows(w: &HashMap<String, Vec<f32>>, g: &Geometry, x: &[f64]) -> Vec<f64> {
    let (h, d) = (g.hidden, g.head_dim);
    let s = x.len() / h;
    let mut rows: Vec<Vec<f64>> = x.chunks(h).map(|r| r.to_vec()).collect();
    let t = |n: &str| &w[&format!("{PREFIX}{n}")];
    for il in 0..g.layers {
        let l = |n: &str| &w[&format!("{PREFIX}layers.{il}.{n}")];
        let mut q = Vec::with_capacity(s);
        let mut k = Vec::with_capacity(s);
        let mut v = Vec::with_capacity(s);
        for (pos, row) in rows.iter().enumerate() {
            let n = rms(row, l("input_layernorm.weight"), g.eps);
            let mut qi = mm(l("self_attn.q_proj.weight"), &n);
            let mut ki = mm(l("self_attn.k_proj.weight"), &n);
            rope(&mut qi, g, pos);
            rope(&mut ki, g, pos);
            q.push(qi);
            k.push(ki);
            v.push(mm(l("self_attn.v_proj.weight"), &n));
        }
        for i in 0..s {
            let first = (i + 1).saturating_sub(g.window);
            let mut out = vec![0f64; g.heads * d];
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
            let att = mm(l("self_attn.o_proj.weight"), &out);
            let scale = l("self_attn_layer_scale.scale");
            for ((r, a), g) in rows[i].iter_mut().zip(&att).zip(scale) {
                *r += a * *g as f64;
            }
        }
        for row in rows.iter_mut() {
            let n = rms(row, l("post_attention_layernorm.weight"), g.eps);
            let gate = mm(l("mlp.gate_proj.weight"), &n);
            let up = mm(l("mlp.up_proj.weight"), &n);
            let act: Vec<f64> = gate
                .iter()
                .zip(&up)
                .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            let down = mm(l("mlp.down_proj.weight"), &act);
            let scale = l("mlp_layer_scale.scale");
            for ((r, a), g) in row.iter_mut().zip(&down).zip(scale) {
                *r += a * *g as f64;
            }
        }
    }
    rows.iter()
        .flat_map(|r| rms(r, t("norm.weight"), g.eps))
        .collect()
}

/// `t` as F32 values on the host.
pub(crate) fn host(t: &Tensor) -> Vec<f32> {
    t.to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// The largest difference of `got` from `want`, and the largest magnitude of `want`.
pub(crate) fn error(got: &[f32], want: &[f64]) -> (f64, f64) {
    assert_eq!(got.len(), want.len());
    assert!(got.iter().all(|v| v.is_finite()));
    let err = got
        .iter()
        .zip(want)
        .map(|(a, b)| (*a as f64 - b).abs())
        .fold(0f64, f64::max);
    (err, want.iter().map(|v| v.abs()).fold(0f64, f64::max))
}

/// The tolerance on a value of magnitude `scale` out of `g`'s layers and final norm, computed
/// in a float type of machine epsilon `eps`.
///
/// A product of `n` terms rounds to about `eps * sqrt(n)` of its scale.
/// A layer chains 4 products: the attention projections, the attention output, the gate and up
/// projections, the down projection.
/// The roundings of the chain add up.
pub(crate) fn tolerance(g: &Geometry, eps: f64, scale: f64) -> f64 {
    let widest = g.ffn.max(g.heads * g.head_dim).max(g.hidden);
    eps * (widest as f64).sqrt() * (4 * g.layers + 1) as f64 * scale
}

/// Rows of the toy's width, `rows` of them.
fn rows_of(g: &Geometry, rows: usize) -> Vec<f32> {
    let mut seed = 5u32;
    values(&mut seed, rows * g.hidden, 0.0, 1.0)
}

/// Every normed row of a prefill longer than the window and than a prefill chunk, within the
/// tolerance of the plain host run.
/// The last row is the normed row of the plain prefill, bit for bit.
fn every_row_matches(toy: &Toy, cards: &[(usize, u64)], eps: f64) {
    let g = Geometry::of(&toy.root);
    let n = crate::inference::engine::llm_engine::PREFILL_CHUNK_TOKENS + 2 * g.window + 3;
    let x = rows_of(&g, n);
    let want = reference_rows(
        &toy.weights,
        &g,
        &x.iter().map(|v| *v as f64).collect::<Vec<_>>(),
    );
    let mut model = toy.load(n, cards);
    let xs = Tensor::from_vec(x.clone(), (n, g.hidden), &Device::Cpu).unwrap();
    let rows = model.forward_embeddings_rows(&xs).unwrap();
    assert_eq!(rows.dims(), [n, g.hidden]);
    let got = host(&rows);
    let (err, scale) = error(&got, &want);
    let tol = tolerance(&g, eps, scale);
    eprintln!("every row {err} away at a scale of {scale}, tolerance {tol}");
    assert!(err <= tol, "every row {err} away, tolerance {tol}");
    let last = model.forward_embeddings_prefill(&xs, false).unwrap();
    assert_eq!(host(&last.normed), got[(n - 1) * g.hidden..].to_vec());
}

/// The layers place on the host with every layer windowed, its cache held to the window and a
/// prefill chunk, and both branches scaled.
#[test]
fn a_windowed_scaled_class_matches_a_plain_host_run() {
    let toy = Toy::write("codec-host", 2, 5);
    let model = toy.load(64, &[]);
    for l in model.layers() {
        assert_eq!(l.sliding_window, Some(5));
        assert!(l.attn_layer_scale.is_some() && l.ffn_layer_scale.is_some());
    }
    every_row_matches(&toy, &[], f32::EPSILON as f64);
}

/// A window wider than the rows gives the causal run of every row.
#[test]
fn a_window_wider_than_the_rows_is_the_causal_run() {
    let toy = Toy::write("codec-wide", 1, 4096);
    every_row_matches(&toy, &[], f32::EPSILON as f64);
}

/// A class windowed in every layer prices its cache at the window and a prefill chunk, whatever
/// the context.
#[test]
fn a_windowed_class_prices_its_cache_at_the_window() {
    let g = |held| super::class_load::ClassDemand {
        layers: 2,
        block_bytes: 1 << 20,
        head_bytes: 0,
        row_bytes: 1024,
        held_rows: held,
        runtime_reserve: 0,
    };
    assert_eq!(g(Some(600)).at(8000).kv_bytes_per_layer, 600 * 1024);
    assert_eq!(g(Some(600)).at(100).kv_bytes_per_layer, 100 * 1024);
    assert_eq!(g(None).at(8000).kv_bytes_per_layer, 8000 * 1024);
}

/// The same run with the layers on the cards, their residual carried in half precision there.
#[cfg(feature = "gpu")]
#[test]
fn the_card_windowed_scaled_class_matches_a_plain_host_run() {
    let cards = crate::inference::place::ladder::cards(0);
    if cards.is_empty() {
        eprintln!("no card");
        return;
    }
    let toy = Toy::write("codec-card", 2, 5);
    let model = toy.load(64, &cards);
    assert!(
        !model.layer_cuda_ordinals().is_empty(),
        "the toy sits on a card"
    );
    drop(model);
    every_row_matches(&toy, &cards, half::f16::EPSILON.to_f32() as f64);
}
