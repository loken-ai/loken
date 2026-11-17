//! The channel scale a quantised key store holds its keys under.
//!
//! A layer that norms its queries and keys folds a scale per key lane into the two norm weights.
//! The K norm weight is divided by the scale, the Q norm weight is multiplied by it.
//! Every score is the same: a lane of the key is divided by what the same lane of the query is
//! multiplied by.
//! The scale is equal on the 2 lanes of a rotation pair, so the rotation turns the scaled pair as
//! it turns the plain one.
//!
//! The scale is the magnitude of the K norm weight, so the scaled lanes of a key carry
//! comparable magnitudes.
//! A block of a quantised store then shares its step among lanes of one size, and a lane with a
//! large weight no longer sets the step of the lanes beside it.
//!
//! Every store of a scaled layer holds scaled keys: the quantised store, the full-width cache it
//! moves to, the host stores and the rows a state cache keeps.
//! The queries are scaled wherever they are computed, as the weight carries the scale.

use super::mixer::Mixer;
use super::GenericTransformerLayer;
use crate::tensor::layer::RmsNorm;
use crate::tensor::{DType, Device, Result, Tensor};

/// The lane a rotation turns together with `lane`, `None` for a lane it leaves alone.
///
/// The first `rotated` lanes of a head rotate.
/// With `adjacent`, lane `2i` pairs with lane `2i + 1`; otherwise lane `i` pairs with lane
/// `i + rotated / 2`.
pub(super) fn rotation_partner(lane: usize, rotated: usize, adjacent: bool) -> Option<usize> {
    let half = rotated / 2;
    if lane >= 2 * half {
        return None;
    }
    Some(match (adjacent, lane < half) {
        (true, _) => lane ^ 1,
        (false, true) => lane + half,
        (false, false) => lane - half,
    })
}

/// The scale of each lane of one head, from the K norm weight of that head.
///
/// A lane's magnitude is the absolute value of its weight.
/// The 2 lanes of a rotation pair take the root mean square of their 2 magnitudes.
/// The scales are divided by their root mean square over the head, so the scaled key keeps the
/// length the plain key has on average.
/// A lane whose pair carries no weight keeps the scale 1: its key lane is zero.
pub(super) fn head_scale(weight: &[f32], rotated: usize, adjacent: bool) -> Vec<f64> {
    let magnitude: Vec<f64> = weight.iter().map(|w| f64::from(*w).abs()).collect();
    let paired: Vec<f64> = (0..magnitude.len())
        .map(|i| match rotation_partner(i, rotated, adjacent) {
            Some(j) => ((magnitude[i].powi(2) + magnitude[j].powi(2)) / 2.0).sqrt(),
            None => magnitude[i],
        })
        .collect();
    let rms = (paired.iter().map(|s| s * s).sum::<f64>() / paired.len().max(1) as f64).sqrt();
    paired
        .iter()
        .map(|&s| {
            if s > 0.0 && rms.is_finite() {
                s / rms
            } else {
                1.0
            }
        })
        .collect()
}

/// The Q and K norm weights with the scale folded in, `None` when the weights do not fit the
/// heads.
///
/// `query` holds one head of `head_dim` lanes shared by every query head, or one per query head.
/// `key` holds one head shared by every key head, or one per key head.
/// A query head reads the key head of its group, and takes that head's scale.
/// A key weight per head beside a shared query weight has no single scale per query lane.
pub(super) fn fold(
    query: &[f32],
    key: &[f32],
    shape: (usize, usize, usize),
    rotated: usize,
    adjacent: bool,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let (n_head, n_kv_head, head_dim) = shape;
    if head_dim == 0 || n_kv_head == 0 || !n_head.is_multiple_of(n_kv_head) {
        return None;
    }
    let key_heads = key.len() / head_dim;
    let query_heads = query.len() / head_dim;
    let shapes_fit = key.len().is_multiple_of(head_dim)
        && query.len().is_multiple_of(head_dim)
        && (key_heads == 1 || key_heads == n_kv_head)
        && (query_heads == 1 || query_heads == n_head)
        && !(key_heads > 1 && query_heads == 1);
    if !shapes_fit {
        return None;
    }
    let scales: Vec<Vec<f64>> = key
        .chunks(head_dim)
        .map(|w| head_scale(w, rotated, adjacent))
        .collect();
    let key_out = key
        .chunks(head_dim)
        .zip(&scales)
        .flat_map(|(w, s)| w.iter().zip(s).map(|(&w, &s)| (f64::from(w) / s) as f32))
        .collect();
    let group = n_head / n_kv_head;
    let query_out = query
        .chunks(head_dim)
        .enumerate()
        .flat_map(|(h, w)| {
            let s = &scales[if key_heads == 1 { 0 } else { h / group }];
            w.iter().zip(s).map(|(&w, &s)| (f64::from(w) * s) as f32)
        })
        .collect();
    Some((query_out, key_out))
}

/// A norm with `values` in place of its weight, at the weight's shape, type and device.
fn renormed(norm: &RmsNorm, values: Vec<f32>) -> Result<RmsNorm> {
    let w = norm.weight();
    let weight = Tensor::from_vec(values, w.shape(), &Device::Cpu)?
        .to_dtype(w.dtype())?
        .to_device(&w.device())?;
    Ok(RmsNorm::from_tensor(weight, norm.eps()))
}

/// The values of a norm's weight, as F32 on the host.
fn host_values(norm: &RmsNorm) -> Result<Vec<f32>> {
    norm.weight()
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
}

impl GenericTransformerLayer {
    /// Whether the store this layer holds its keys in is quantised.
    /// On a card it is the quantised cache the layer holds.
    /// On the host it is the Q8 store the configured cache format asks for.
    fn holds_quantised_keys(&self) -> bool {
        if self.cos.device().is_cuda() {
            #[cfg(feature = "gpu")]
            return self.quant_kv.is_some();
            #[cfg(not(feature = "gpu"))]
            return false;
        }
        self.host_q8_kv
    }

    /// Folds the key scale into the Q and K norm weights of an attention layer that holds its
    /// keys in a quantised store, and records it in `key_scaled`.
    /// A layer whose keys another layer reads, or that reads another layer's keys, keeps its
    /// weights: `shares_keys` says so.
    /// A layer without both norms keeps its weights.
    pub(super) fn fold_key_scale(&mut self, shares_keys: bool) -> Result<()> {
        if shares_keys
            || !matches!(self.mixer, Mixer::Attention)
            || self.populate_dual_kv
            || !self.holds_quantised_keys()
        {
            return Ok(());
        }
        let (Some(query), Some(key)) = (self.attn_q_norm.as_ref(), self.attn_k_norm.as_ref())
        else {
            return Ok(());
        };
        let rotated = match (self.no_rope, self.rope_dim) {
            (true, _) => 0,
            (false, 0) => self.head_dim,
            (false, d) => d,
        };
        let Some((q, k)) = fold(
            &host_values(query)?,
            &host_values(key)?,
            (self.n_head, self.n_kv_head, self.head_dim),
            rotated,
            self.flags.use_rope_i,
        ) else {
            return Ok(());
        };
        let (query, key) = (renormed(query, q)?, renormed(key, k)?);
        self.attn_q_norm = Some(query);
        self.attn_k_norm = Some(key);
        self.key_scaled = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic spread of values in [-1, 1).
    fn values(n: usize, seed: u64) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let h = (i as u64 + 1)
                    .wrapping_mul(2654435761)
                    .wrapping_add(seed.wrapping_mul(40503));
                ((h >> 8) % 2001) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    /// `x` rotated by `angle` times the pair's rank, pairs as `rotation_partner` names them.
    fn rotate(x: &[f64], angle: f64, rotated: usize, adjacent: bool) -> Vec<f64> {
        let mut out = x.to_vec();
        for i in 0..x.len() {
            let Some(j) = rotation_partner(i, rotated, adjacent) else {
                continue;
            };
            // The first lane of a pair takes cos and -sin, the second sin and cos.
            let first = if adjacent { i % 2 == 0 } else { i < j };
            let rank = if adjacent { i / 2 } else { i.min(j) };
            let (c, s) = (
                (angle * (rank + 1) as f64).cos(),
                (angle * (rank + 1) as f64).sin(),
            );
            out[i] = if first {
                x[i] * c - x[j] * s
            } else {
                x[j] * s + x[i] * c
            };
        }
        out
    }

    /// The Q8_0 rounding of `x` in blocks of 32: each block shares the step of its largest
    /// magnitude over 127, held as F16.
    fn q8_0(x: &[f64]) -> Vec<f64> {
        x.chunks(32)
            .flat_map(|b| {
                let max = b.iter().fold(0f64, |m, v| m.max(v.abs()));
                let d = f64::from(half::f16::from_f64(max / 127.0).to_f32());
                b.iter()
                    .map(move |v| if d > 0.0 { (v / d).round() * d } else { 0.0 })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn a_rotation_pairs_lanes_by_its_layout() {
        assert_eq!(rotation_partner(0, 8, false), Some(4));
        assert_eq!(rotation_partner(5, 8, false), Some(1));
        assert_eq!(rotation_partner(0, 8, true), Some(1));
        assert_eq!(rotation_partner(5, 8, true), Some(4));
        assert_eq!(rotation_partner(8, 8, false), None);
        assert_eq!(rotation_partner(3, 0, false), None);
    }

    #[test]
    fn a_pair_shares_one_scale_and_the_scales_keep_unit_mean_square() {
        for (rotated, adjacent) in [(64, false), (64, true), (32, false), (0, false)] {
            let w = values(64, 7);
            let s = head_scale(&w, rotated, adjacent);
            for i in 0..64 {
                if let Some(j) = rotation_partner(i, rotated, adjacent) {
                    assert_eq!(s[i], s[j], "lanes {i} and {j}, rotated {rotated}");
                }
            }
            let ms = s.iter().map(|v| v * v).sum::<f64>() / 64.0;
            assert!((ms - 1.0).abs() < 1e-12, "mean square {ms}");
        }
        assert_eq!(head_scale(&[0.0; 8], 8, false), vec![1.0; 8]);
    }

    /// Queries and keys normed, rotated and scored with the folded weights score as with the
    /// plain ones, for each rotation layout and for a weight per head.
    #[test]
    fn the_folded_weights_score_as_the_plain_ones() {
        let (n_head, n_kv, hd) = (4usize, 2usize, 64usize);
        for (rotated, adjacent, per_head) in [
            (64, false, false),
            (64, true, false),
            (32, false, false),
            (0, false, false),
            (64, false, true),
        ] {
            let heads = |n: usize| if per_head { n } else { 1 };
            let q_w = values(heads(n_head) * hd, 1);
            let k_w: Vec<f32> = values(heads(n_kv) * hd, 2)
                .iter()
                .enumerate()
                .map(|(i, v)| if i % hd == 3 { 40.0 } else { *v })
                .collect();
            let (q_f, k_f) = fold(&q_w, &k_w, (n_head, n_kv, hd), rotated, adjacent).unwrap();
            let (xq, xk) = (values(n_head * hd, 3), values(n_kv * hd, 4));
            for h in 0..n_head {
                let kv = h / (n_head / n_kv);
                let lanes = |w: &[f32], x: &[f32], head: usize, scaled_head: usize| -> Vec<f64> {
                    let w = &w[if w.len() == hd { 0 } else { scaled_head * hd }..][..hd];
                    (0..hd)
                        .map(|i| f64::from(w[i]) * f64::from(x[head * hd + i]))
                        .collect()
                };
                let rotated_pair = |qw: &[f32], kw: &[f32]| {
                    let q = rotate(&lanes(qw, &xq, h, h), 0.3, rotated, adjacent);
                    let k = rotate(&lanes(kw, &xk, kv, kv), 0.7, rotated, adjacent);
                    (q, k)
                };
                let (q, k) = rotated_pair(&q_w, &k_w);
                let (qf, kf) = rotated_pair(&q_f, &k_f);
                let (plain, folded) = (dot(&q, &k), dot(&qf, &kf));
                // Each folded weight is one F32 rounding off its exact value.
                let bound = 4.0 * f64::from(f32::EPSILON) * dot(&q, &q).sqrt() * dot(&k, &k).sqrt();
                assert!(
                    (plain - folded).abs() <= bound,
                    "head {h}, rotated {rotated}, adjacent {adjacent}: {plain} against {folded}"
                );
            }
        }
    }

    /// A key rounded to Q8_0 scores as close under the fold as plain when its lanes carry one
    /// size, and closer than plain when one rotation pair's weight sets the step of its blocks.
    /// The error is the summed distance of the rounded scores to the exact ones.
    #[test]
    fn a_scaled_key_rounds_closer_when_one_pair_is_an_outlier() {
        let hd = 64usize;
        let error = |k_w: &[f32]| -> (f64, f64) {
            let q_w = vec![1.0f32; hd];
            let (q_f, k_f) = fold(&q_w, k_w, (1, 1, hd), hd, false).unwrap();
            let (mut plain, mut scaled) = (0f64, 0f64);
            for seed in 0..64u64 {
                let (xq, xk) = (values(hd, 100 + seed), values(hd, 200 + seed));
                let lanes = |w: &[f32], x: &[f32]| -> Vec<f64> {
                    (0..hd).map(|i| f64::from(w[i]) * f64::from(x[i])).collect()
                };
                let (q, k) = (lanes(&q_w, &xq), lanes(k_w, &xk));
                let (qs, ks) = (lanes(&q_f, &xq), lanes(&k_f, &xk));
                let exact = dot(&q, &k);
                plain += (dot(&q, &q8_0(&k)) - exact).abs();
                scaled += (dot(&qs, &q8_0(&ks)) - exact).abs();
            }
            (plain, scaled)
        };
        let uniform = vec![0.5f32; hd];
        let (plain, scaled) = error(&uniform);
        assert!(
            (plain - scaled).abs() <= 1e-6 * plain.max(1e-30),
            "uniform lanes: {plain} plain, {scaled} scaled"
        );
        let mut outlier = uniform.clone();
        outlier[3] = 40.0;
        outlier[3 + hd / 2] = 40.0;
        let (plain, scaled) = error(&outlier);
        eprintln!("an outlier pair: {plain} plain, {scaled} scaled");
        assert!(
            scaled < plain,
            "an outlier pair: {plain} plain, {scaled} scaled"
        );
    }
}

#[cfg(all(test, feature = "gpu"))]
mod card_tests {
    use super::super::{tiny_model, GenericHeteroTransformer};
    use crate::inference::engine::llm_engine::KvQuant;
    use crate::tensor::{DType, Device, Tensor};

    const LAYERS: usize = 2;
    /// Tokens before the rows the paths compare.
    const PROMPT: usize = 24;
    /// Rows each path computes after the prompt.
    const ROWS: usize = 4;

    fn ids(toks: &[u32]) -> Tensor {
        Tensor::new(toks, &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
    }

    fn rows(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    fn relative_l2(exact: &[f32], got: &[f32]) -> f64 {
        let (mut dd, mut ww) = (0f64, 0f64);
        for (&w, &g) in exact.iter().zip(got) {
            let (w, g) = (f64::from(w), f64::from(g));
            dd += (w - g) * (w - g);
            ww += w * w;
        }
        (dd / ww).sqrt()
    }

    /// The logits of the `ROWS` tokens after the prompt, by 3 paths: rows of 1 prefill of every
    /// token, decode steps after a prefill of the prompt, and 1 verify of the rows after it.
    fn paths(m: &mut GenericHeteroTransformer) -> [Vec<f32>; 3] {
        let vocab = tiny_model::QUANT_VOCAB as u32;
        let toks: Vec<u32> = (0..PROMPT + ROWS)
            .map(|i| ((i * 7 + i / 3) as u32) % vocab)
            .collect();
        let (prompt, next) = toks.split_at(PROMPT);
        let all = rows(&m.forward_all(&ids(&toks), 0).unwrap());
        let width = all.len() / toks.len();
        let prefill = all[(PROMPT - 1) * width..(PROMPT + ROWS - 1) * width].to_vec();
        // The prompt's last row leads the decode steps and the verify rows.
        let mut decode = rows(&m.forward(&ids(prompt), 0).unwrap());
        for (i, &t) in next[..ROWS - 1].iter().enumerate() {
            decode.extend(rows(&m.forward(&ids(&[t]), PROMPT + i).unwrap()));
        }
        let mut verify = rows(&m.forward(&ids(prompt), 0).unwrap());
        verify.extend(rows(
            &m.forward_all(&ids(&next[..ROWS - 1]), PROMPT).unwrap(),
        ));
        [prefill, decode, verify]
    }

    /// A model whose K norm weight has one large rotation pair, on a card: the layers of a load
    /// with a quantised cache store scaled keys, and its prefill rows, decode steps and verify
    /// rows are no farther from the full-width cache than those of the same cache with plain
    /// keys.
    /// A load with a full-width cache keeps the file's norm weights.
    #[test]
    fn scaled_keys_read_closer_to_the_full_width_cache_on_every_path() {
        if Device::new_cuda(0).is_err() {
            eprintln!("skipping: no usable card");
            return;
        }
        let head_dim = tiny_model::QUANT_EMBD / tiny_model::QUANT_HEADS;
        let mut key_norm = vec![1.0f32; head_dim];
        key_norm[3] = 24.0;
        key_norm[3 + head_dim / 2] = 24.0;
        let path = tiny_model::write_quant_with(
            "key-scale-card",
            LAYERS,
            tiny_model::QuantExtras {
                key_norm: Some(key_norm),
                ..Default::default()
            },
        );
        let load = |kv| tiny_model::load_on_cards_with_kv(&path, LAYERS, &[(0, 0)], kv);
        let mut full = load(KvQuant::Off);
        let mut scaled = load(KvQuant::Q8);
        let mut plain = load(KvQuant::Q8);
        assert!(full.layers.iter().all(|l| !l.key_scaled));
        assert!(scaled.layers.iter().all(|l| l.key_scaled));
        for (p, f) in plain.layers.iter_mut().zip(&full.layers) {
            p.attn_q_norm = f.attn_q_norm.clone();
            p.attn_k_norm = f.attn_k_norm.clone();
            p.key_scaled = false;
        }
        let want = paths(&mut full);
        let (got, base) = (paths(&mut scaled), paths(&mut plain));
        for (i, name) in ["prefill rows", "decode steps", "verify rows"]
            .iter()
            .enumerate()
        {
            let (d_scaled, d_plain) = (
                relative_l2(&want[i], &got[i]),
                relative_l2(&want[i], &base[i]),
            );
            eprintln!("{name}: {d_scaled} scaled, {d_plain} plain, off the full-width cache");
            assert!(
                d_scaled < d_plain,
                "{name}: scaled keys {d_scaled} off the full-width cache, plain keys {d_plain}"
            );
        }
        tiny_model::remove(&path);
    }
}
