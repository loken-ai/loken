//! The decode step fed by an embedding, held to a plain host run of the same layers, to the token
//! step, and to a fresh model after position 0.

use super::*;
use crate::inference::place::layer_executor::HeteroPlan;
use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
use crate::tensor::quantized::gguf_file::Value;
use crate::tensor::quantized::GgmlDType;

/// Blocks, width, heads, key-value heads, head width, feed-forward width and vocabulary of the
/// toy: grouped-query attention with query and key norms, heads wider than the residual.
const LAYERS: usize = 2;
const EMBD: usize = 128;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const FFN: usize = 256;
const VOCAB: usize = 48;
const EPS: f32 = 1e-6;
const ROPE_BASE: f32 = 1e6;

/// Rows each sequence of the tests decodes.
const STEPS: usize = 6;

/// The toy written as a GGUF file, and its weights as the reference reads them.
struct Toy {
    path: std::path::PathBuf,
    weights: HashMap<String, Vec<f32>>,
}

/// Values of a deterministic pattern in `[-scale, scale)`, around `centre`.
fn values(seed: &mut u32, n: usize, centre: f32, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            centre + ((*seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

fn write_toy(tag: &str) -> Toy {
    let mut seed = 31u32;
    let mut weights = HashMap::new();
    let mut entries = Vec::new();
    let mut put = |name: String, dims: Vec<usize>, centre: f32, scale: f32| {
        let data = values(&mut seed, dims.iter().product(), centre, scale);
        entries.push(GgufEntry {
            name: name.clone(),
            dims,
            dtype: GgmlDType::F32,
            data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        });
        weights.insert(name, data);
    };
    let (q, kv) = (HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
    put("token_embd.weight".into(), vec![VOCAB, EMBD], 0.0, 1.0);
    put("output.weight".into(), vec![VOCAB, EMBD], 0.0, 0.1);
    put("output_norm.weight".into(), vec![EMBD], 1.0, 0.2);
    for il in 0..LAYERS {
        for (part, dims, centre, scale) in [
            ("attn_norm", vec![EMBD], 1.0, 0.2),
            ("ffn_norm", vec![EMBD], 1.0, 0.2),
            ("attn_q_norm", vec![HEAD_DIM], 1.0, 0.2),
            ("attn_k_norm", vec![HEAD_DIM], 1.0, 0.2),
            ("attn_q", vec![q, EMBD], 0.0, 0.1),
            ("attn_k", vec![kv, EMBD], 0.0, 0.1),
            ("attn_v", vec![kv, EMBD], 0.0, 0.1),
            ("attn_output", vec![EMBD, q], 0.0, 0.1),
            ("ffn_gate", vec![FFN, EMBD], 0.0, 0.1),
            ("ffn_up", vec![FFN, EMBD], 0.0, 0.1),
            ("ffn_down", vec![EMBD, FFN], 0.0, 0.1),
        ] {
            put(format!("blk.{il}.{part}.weight"), dims, centre, scale);
        }
    }
    let u = |k: &str, v: usize| (format!("qwen3.{k}"), Value::U32(v as u32));
    let metadata = vec![
        ("general.architecture".into(), Value::String("qwen3".into())),
        u("block_count", LAYERS),
        u("embedding_length", EMBD),
        u("feed_forward_length", FFN),
        u("attention.head_count", HEADS),
        u("attention.head_count_kv", KV_HEADS),
        u("attention.key_length", HEAD_DIM),
        u("attention.value_length", HEAD_DIM),
        u("context_length", 64),
        (
            "qwen3.attention.layer_norm_rms_epsilon".into(),
            Value::F32(EPS),
        ),
        ("qwen3.rope.freq_base".into(), Value::F32(ROPE_BASE)),
    ];
    let dir = std::env::temp_dir().join(format!("loken-step-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.gguf");
    write_gguf_with_metadata(&path, &metadata, &entries).unwrap();
    Toy { path, weights }
}

impl Toy {
    /// The toy loaded through the route with its first `on_card` layers on `cards`' card 0.
    fn load(&self, cards: &HashMap<usize, Device>, on_card: usize) -> GenericHeteroTransformer {
        let g = gguf_file::open_mapped(&self.path).unwrap();
        let bytes = g.mmap.clone();
        GenericHeteroTransformer::from_gguf(
            g.content,
            &bytes[..],
            cards,
            &HeteroPlan::forced_gpu(LAYERS, on_card, 0),
        )
        .unwrap()
    }

    /// Whether the loaded toy pairs adjacent rotary lanes.
    fn adjacent_rope(&self) -> bool {
        let g = gguf_file::open_mapped(&self.path).unwrap();
        GenericTransformerConfig::from_gguf(&g.content, "qwen3")
            .unwrap()
            .flags
            .use_rope_i
    }

    fn remove(&self) {
        if let Some(dir) = self.path.parent() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

/// The plain host run of the toy: f64 sums over the stored F32 weights, one row at a time.
struct Reference<'a> {
    w: &'a HashMap<String, Vec<f32>>,
    adjacent: bool,
    /// Each layer's rotated keys and values of the rows so far.
    cache: Vec<(Vec<Vec<f32>>, Vec<Vec<f32>>)>,
}

impl<'a> Reference<'a> {
    fn new(toy: &'a Toy) -> Self {
        Self {
            w: &toy.weights,
            adjacent: toy.adjacent_rope(),
            cache: vec![(Vec::new(), Vec::new()); LAYERS],
        }
    }

    fn rms(x: &[f32], g: &[f32]) -> Vec<f32> {
        let ms = x.iter().map(|a| *a as f64 * *a as f64).sum::<f64>() / x.len() as f64;
        let r = 1.0 / (ms + EPS as f64).sqrt();
        x.iter()
            .zip(g)
            .map(|(a, b)| (*a as f64 * r * *b as f64) as f32)
            .collect()
    }

    /// `w` `[rows, x.len()]` times `x`.
    fn mm(w: &[f32], x: &[f32]) -> Vec<f32> {
        w.chunks(x.len())
            .map(|row| {
                row.iter()
                    .zip(x)
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum::<f64>() as f32
            })
            .collect()
    }

    /// Each head of `x` normed by `g` and rotated to `pos`.
    fn heads(&self, x: &[f32], g: &[f32], pos: usize) -> Vec<f32> {
        let half = HEAD_DIM / 2;
        x.chunks(HEAD_DIM)
            .flat_map(|head| {
                let mut h = Self::rms(head, g);
                for i in 0..half {
                    let freq = (ROPE_BASE as f64).powf(-((2 * i) as f64) / HEAD_DIM as f64);
                    let (sn, cs) = (pos as f64 * freq).sin_cos();
                    let (a, b) = if self.adjacent {
                        (2 * i, 2 * i + 1)
                    } else {
                        (i, i + half)
                    };
                    let (x0, x1) = (h[a] as f64, h[b] as f64);
                    h[a] = (x0 * cs - x1 * sn) as f32;
                    h[b] = (x0 * sn + x1 * cs) as f32;
                }
                h
            })
            .collect()
    }

    /// The row `x` at `pos` through every layer: its logits and its row after the final norm.
    fn step(&mut self, x: &[f32], pos: usize) -> (Vec<f32>, Vec<f32>) {
        if pos == 0 {
            self.cache = vec![(Vec::new(), Vec::new()); LAYERS];
        }
        let weights: &'a HashMap<String, Vec<f32>> = self.w;
        let w = |n: String| -> &'a [f32] { &weights[&n] };
        let mut x = x.to_vec();
        for il in 0..LAYERS {
            let p = |part: &str| format!("blk.{il}.{part}.weight");
            let xn = Self::rms(&x, w(p("attn_norm")));
            let q = self.heads(&Self::mm(w(p("attn_q")), &xn), w(p("attn_q_norm")), pos);
            let k = self.heads(&Self::mm(w(p("attn_k")), &xn), w(p("attn_k_norm")), pos);
            let v = Self::mm(w(p("attn_v")), &xn);
            self.cache[il].0.push(k);
            self.cache[il].1.push(v);
            let (keys, vals) = &self.cache[il];
            let scale = 1.0 / (HEAD_DIM as f64).sqrt();
            let mut attn = vec![0f32; HEADS * HEAD_DIM];
            for h in 0..HEADS {
                let g = h / (HEADS / KV_HEADS);
                let qh = &q[h * HEAD_DIM..(h + 1) * HEAD_DIM];
                let s: Vec<f64> = keys
                    .iter()
                    .map(|k| {
                        qh.iter()
                            .zip(&k[g * HEAD_DIM..(g + 1) * HEAD_DIM])
                            .map(|(a, b)| *a as f64 * *b as f64)
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = s.iter().map(|v| (v - m).exp()).collect();
                let sum: f64 = e.iter().sum();
                for d in 0..HEAD_DIM {
                    let o: f64 = e
                        .iter()
                        .zip(vals)
                        .map(|(e, v)| e * v[g * HEAD_DIM + d] as f64)
                        .sum();
                    attn[h * HEAD_DIM + d] = (o / sum) as f32;
                }
            }
            let o = Self::mm(w(p("attn_output")), &attn);
            let x1: Vec<f32> = x.iter().zip(&o).map(|(a, b)| a + b).collect();
            let xn2 = Self::rms(&x1, w(p("ffn_norm")));
            let gate = Self::mm(w(p("ffn_gate")), &xn2);
            let up = Self::mm(w(p("ffn_up")), &xn2);
            let act: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(g, u)| (*g as f64 / (1.0 + (-*g as f64).exp()) * *u as f64) as f32)
                .collect();
            let down = Self::mm(w(p("ffn_down")), &act);
            x = x1.iter().zip(&down).map(|(a, b)| a + b).collect();
        }
        let normed = Self::rms(&x, w("output_norm.weight".into()));
        (Self::mm(w("output.weight".into()), &normed), normed)
    }
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// The embeddings of a sequence: rows unlike any of the token table.
fn embeddings(seed: u32) -> Vec<Vec<f32>> {
    let mut seed = seed;
    (0..STEPS)
        .map(|_| values(&mut seed, EMBD, 0.0, 1.0))
        .collect()
}

/// The tolerance on a value of magnitude `scale` computed in a float type of machine epsilon
/// `eps`.
///
/// A product of `n` terms rounds to about `eps * sqrt(n)` of its scale.
/// A layer chains 4 products: the attention projections, the attention output, the gate and up
/// projections, the down projection.
/// The head adds 1.
/// The roundings of the chain add up: `eps * sqrt(FFN) * (4 * LAYERS + 1) * scale`, `FFN` being
/// the longest product.
fn tolerance(eps: f32, scale: f32) -> f32 {
    eps * (FFN as f32).sqrt() * (4 * LAYERS + 1) as f32 * scale
}

/// The largest difference of `got` from `want`, and the tolerance on `want`'s largest magnitude.
fn error(got: &[f32], want: &[f32], eps: f32) -> (f32, f32) {
    assert_eq!(got.len(), want.len());
    let err = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
    (err, tolerance(eps, scale))
}

/// Every step of a sequence of embeddings within the tolerance of the plain host run, its
/// logits and its normed row, and the normed row the same whether the head runs or not.
fn steps_match_the_reference(toy: &Toy, cards: &HashMap<usize, Device>, on_card: usize, eps: f32) {
    let mut model = toy.load(cards, on_card);
    assert_eq!(on_card_of(&model), on_card);
    let mut headless = toy.load(cards, on_card);
    let mut reference = Reference::new(toy);
    for (pos, row) in embeddings(5).iter().enumerate() {
        let x = Tensor::from_slice(row, (1, 1, EMBD), &Device::Cpu).unwrap();
        let step = model.forward_embedding_step(&x, pos, true).unwrap();
        let (want_logits, want_normed) = reference.step(row, pos);
        assert_eq!(step.normed.dims(), [1, EMBD]);
        let logits = step.logits.as_ref().unwrap();
        assert_eq!(logits.dims(), [1, VOCAB]);
        for (what, got, want) in [
            ("logits", host(logits), &want_logits),
            ("normed row", host(&step.normed), &want_normed),
        ] {
            let (err, tol) = error(&got, want, eps);
            assert!(err <= tol, "step {pos}: {what} {err} away, tolerance {tol}");
        }
        let alone = headless.forward_embedding_step(&x, pos, false).unwrap();
        assert!(alone.logits.is_none());
        assert_eq!(host(&alone.normed), host(&step.normed), "step {pos}");
    }
}

/// The token step and the embedding step fed the token's table row give the same bits.
fn the_table_row_is_the_token_step(toy: &Toy, cards: &HashMap<usize, Device>, on_card: usize) {
    let mut by_token = toy.load(cards, on_card);
    let mut by_row = toy.load(cards, on_card);
    let table = &toy.weights["token_embd.weight"];
    for (pos, token) in [3u32, 17, 0, 41, 17, 9].into_iter().enumerate() {
        let ids = Tensor::new(&[token], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let want = host(&by_token.forward(&ids, pos).unwrap());
        let row = &table[token as usize * EMBD..(token as usize + 1) * EMBD];
        let x = Tensor::from_slice(row, EMBD, &Device::Cpu).unwrap();
        let got = by_row.forward_embedding_step(&x, pos, true).unwrap();
        assert_eq!(host(got.logits.as_ref().unwrap()), want, "step {pos}");
    }
}

/// A step at position 0 after a sequence gives the bits of a fresh model.
fn position_zero_starts_afresh(toy: &Toy, cards: &HashMap<usize, Device>, on_card: usize) {
    let mut used = toy.load(cards, on_card);
    let mut fresh = toy.load(cards, on_card);
    for (pos, row) in embeddings(9).iter().enumerate() {
        let x = Tensor::from_slice(row, EMBD, &Device::Cpu).unwrap();
        used.forward_embedding_step(&x, pos, true).unwrap();
    }
    for (pos, row) in embeddings(13).iter().enumerate() {
        let x = Tensor::from_slice(row, EMBD, &Device::Cpu).unwrap();
        let a = used.forward_embedding_step(&x, pos, true).unwrap();
        let b = fresh.forward_embedding_step(&x, pos, true).unwrap();
        assert_eq!(
            host(a.logits.as_ref().unwrap()),
            host(b.logits.as_ref().unwrap()),
            "step {pos}"
        );
        assert_eq!(host(&a.normed), host(&b.normed), "step {pos}");
    }
}

/// The layers of `model` placed on a card.
fn on_card_of(model: &GenericHeteroTransformer) -> usize {
    model
        .layer_devs
        .iter()
        .filter(|d| matches!(d, LayerDevice::Cuda(_)))
        .count()
}

#[test]
fn an_embedding_step_matches_a_plain_host_run() {
    let toy = write_toy("host");
    steps_match_the_reference(&toy, &HashMap::new(), 0, f32::EPSILON);
    toy.remove();
}

#[test]
fn an_embedding_step_fed_a_table_row_is_the_token_step() {
    let toy = write_toy("token");
    the_table_row_is_the_token_step(&toy, &HashMap::new(), 0);
    toy.remove();
}

#[test]
fn an_embedding_step_at_position_zero_starts_afresh() {
    let toy = write_toy("fresh");
    position_zero_starts_afresh(&toy, &HashMap::new(), 0);
    toy.remove();
}

#[test]
fn an_embedding_step_refuses_a_row_of_another_width() {
    let toy = write_toy("width");
    let mut model = toy.load(&HashMap::new(), 0);
    let x = Tensor::zeros(EMBD + 1, DType::F32).unwrap();
    assert!(model.forward_embedding_step(&x, 0, true).is_err());
    toy.remove();
}

/// The same checks with every layer on card 0, the residual carried in half precision there.
#[cfg(feature = "gpu")]
#[test]
fn the_card_embedding_step_matches_a_plain_host_run() {
    let Ok(card) = Device::new_cuda(0) else {
        eprintln!("no card");
        return;
    };
    let cards: HashMap<usize, Device> = [(0usize, card)].into_iter().collect();
    let toy = write_toy("card");
    steps_match_the_reference(&toy, &cards, LAYERS, half::f16::EPSILON.to_f32());
    the_table_row_is_the_token_step(&toy, &cards, LAYERS);
    position_zero_starts_afresh(&toy, &cards, LAYERS);
    toy.remove();
}

/// A prefill of a sequence of embeddings gives the last row of the plain host run, and the logits
/// a splice of the same rows with no text gives, bit for bit.
fn prefill_matches_the_reference(
    toy: &Toy,
    cards: &HashMap<usize, Device>,
    on_card: usize,
    eps: f32,
) {
    let rows = embeddings(17);
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let x = Tensor::from_vec(flat, (STEPS, EMBD), &Device::Cpu).unwrap();
    let mut model = toy.load(cards, on_card);
    let got = model.forward_embeddings_prefill(&x, true).unwrap();
    let mut reference = Reference::new(toy);
    let mut want = (Vec::new(), Vec::new());
    for (pos, row) in rows.iter().enumerate() {
        want = reference.step(row, pos);
    }
    for (what, got, want) in [
        ("logits", host(got.logits.as_ref().unwrap()), &want.0),
        ("normed row", host(&got.normed), &want.1),
    ] {
        let (err, tol) = error(&got, want, eps);
        assert!(err <= tol, "{what} {err} away, tolerance {tol}");
    }
    let mut spliced = toy.load(cards, on_card);
    let none = Tensor::zeros((1, 0), DType::U32).unwrap();
    let splice = spliced
        .forward_with_audio_embeds(&none, &x.unsqueeze(0).unwrap(), &none)
        .unwrap();
    assert_eq!(host(&splice), host(got.logits.as_ref().unwrap()));
    let mut headless = toy.load(cards, on_card);
    let alone = headless.forward_embeddings_prefill(&x, false).unwrap();
    assert!(alone.logits.is_none());
    assert_eq!(host(&alone.normed), host(&got.normed));
}

#[test]
fn an_embedding_prefill_matches_a_plain_host_run() {
    let toy = write_toy("prefill");
    prefill_matches_the_reference(&toy, &HashMap::new(), 0, f32::EPSILON);
    toy.remove();
}

/// The prefill with every layer on card 0.
#[cfg(feature = "gpu")]
#[test]
fn the_card_embedding_prefill_matches_a_plain_host_run() {
    let Ok(card) = Device::new_cuda(0) else {
        eprintln!("no card");
        return;
    };
    let cards: HashMap<usize, Device> = [(0usize, card)].into_iter().collect();
    let toy = write_toy("card-prefill");
    prefill_matches_the_reference(&toy, &cards, LAYERS, half::f16::EPSILON.to_f32());
    toy.remove();
}
