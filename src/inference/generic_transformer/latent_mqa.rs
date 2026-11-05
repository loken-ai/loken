//! Latent attention over 1 shared key head, on the host.
//!
//! A layer projects its input to a low-rank query factor, normalises it and lifts it to the
//! heads. It projects the same input to 1 key row per token, normalised. That row serves every
//! head as its key and as its value.
//!
//! The last `rope_dim` lanes of each query head and of the key row are rotated by position.
//! The key row then goes through an fp8 e4m3 round trip, before it is attended or kept.
//!
//! A query attends over the last `window` key rows, its own included. A learned logit per head
//! takes softmax mass beside them and adds no value.
//!
//! The output of each head is rotated back by its own position, so every kept row stays in the
//! same rotated form. A block-diagonal low-rank projection then maps each group of heads to its own
//! slice. The mixer returns those slices side by side, before the layer's output projection.
//!
//! The state is the last `window` key rows.

use super::wide_sparse_config::WideSparseMeta;
use crate::tensor::blockscaled::projection::Projection;
use crate::tensor::ops::rms_norm;
use crate::tensor::{DType, Device, Result, Tensor};
use rayon::prelude::*;

/// The largest magnitude of e4m3fn. The round trip clamps there.
pub(super) const FP8_E4M3_MAX: f32 = 448.0;

/// Values of a key row that share 1 power-of-two scale in the round trip.
pub(crate) const FP8_BLOCK: usize = 32;

/// The smallest scale of a block, so a block of zeros divides by a positive number.
pub(super) const SCALE_FLOOR: f32 = 1e-30;

/// The widths of a latent attention over 1 shared key head.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatentMqaDims {
    pub heads: usize,
    pub head_dim: usize,
    /// The rotated lanes at the end of a head.
    pub rope_dim: usize,
    /// Groups of heads the output projection maps apart.
    pub o_groups: usize,
    /// Output width of each group.
    pub o_lora_rank: usize,
    /// Key rows a query attends over, its own included.
    pub window: usize,
    /// Epsilon of the norms on the query factor and on the key row.
    pub eps: f32,
}

impl LatentMqaDims {
    /// The widths a wide-residual model's header gives every layer.
    pub fn from_meta(meta: &WideSparseMeta) -> Self {
        Self {
            heads: meta.heads,
            head_dim: meta.head_dim,
            rope_dim: meta.rope_dim,
            o_groups: meta.o_groups,
            o_lora_rank: meta.o_lora_rank,
            window: meta.window,
            eps: meta.rms_eps as f32,
        }
    }

    /// Width of the mixer's output row: every group's slice side by side.
    pub fn out_width(&self) -> usize {
        self.o_groups * self.o_lora_rank
    }

    /// Lanes of the heads' output that 1 group projects.
    fn group_width(&self) -> usize {
        self.heads * self.head_dim / self.o_groups
    }
}

/// The weights of the mixer.
pub struct LatentMqaWeights {
    /// `[q_lora_rank, hidden]`.
    pub wq_a: Projection,
    /// Scale of the norm on the query factor, `[q_lora_rank]` F32.
    pub q_norm: Tensor,
    /// `[heads * head_dim, q_lora_rank]`.
    pub wq_b: Projection,
    /// `[head_dim, hidden]`.
    pub wkv: Projection,
    /// Scale of the norm on the key row, `[head_dim]` F32.
    pub kv_norm: Tensor,
    /// The logit each head gives the sink, `[heads]`.
    pub sink: Vec<f32>,
    /// `[o_groups * o_lora_rank, heads * head_dim / o_groups]`: group `g` is rows
    /// `g * o_lora_rank..(g + 1) * o_lora_rank`.
    pub wo_a: Projection,
}

/// What a layer keeps of the tokens it has seen.
#[derive(Clone, Debug, Default)]
pub struct LatentMqaState {
    /// Tokens seen.
    pub len: usize,
    /// The last `min(len, window)` key rows, oldest first, `head_dim` values each, rotated and
    /// rounded to the fp8 grid.
    pub window: Vec<f32>,
}

/// A latent attention layer over 1 shared key head.
pub struct LatentMqa {
    pub(super) dims: LatentMqaDims,
    pub(super) w: LatentMqaWeights,
    /// `wo_a` cut into its groups.
    wo_groups: Vec<Projection>,
    /// Rotation frequency of each lane pair of the rotated tail.
    pub(super) inv_freq: Vec<f32>,
    pub state: Option<LatentMqaState>,
    /// Times the state was dropped or replaced.
    pub(super) moves: u64,
}

impl LatentMqa {
    /// Bytes of the projections every token reads, as held: the query factors, the key row
    /// and the grouped output.
    pub fn resident_bytes(&self) -> usize {
        self.w.wq_a.bytes() + self.w.wq_b.bytes() + self.w.wkv.bytes() + self.w.wo_a.bytes()
    }

    /// The layer over `w`. `inv_freq` holds `rope_dim / 2` frequencies.
    pub fn new(dims: LatentMqaDims, w: LatentMqaWeights, inv_freq: Vec<f32>) -> Result<Self> {
        let d = dims;
        let refuse = |what: String| {
            crate::tensor::Error::msg(format!("latent attention over 1 key head: {what}"))
        };
        if d.o_groups == 0 || !(d.heads * d.head_dim).is_multiple_of(d.o_groups) {
            let (h, hd, g) = (d.heads, d.head_dim, d.o_groups);
            return Err(refuse(format!("{h} heads of {hd} in {g} groups")));
        }
        if !d.head_dim.is_multiple_of(FP8_BLOCK) {
            let hd = d.head_dim;
            return Err(refuse(format!(
                "a key row of {hd} in blocks of {FP8_BLOCK}"
            )));
        }
        if d.rope_dim > d.head_dim || inv_freq.len() * 2 != d.rope_dim {
            return Err(refuse(format!(
                "{} frequencies for {} rotated lanes of {}",
                inv_freq.len(),
                d.rope_dim,
                d.head_dim
            )));
        }
        if d.window == 0 {
            return Err(refuse("a window of no row".to_string()));
        }
        if w.sink.len() != d.heads {
            let (n, h) = (w.sink.len(), d.heads);
            return Err(refuse(format!("{n} sink logits for {h} heads")));
        }
        let shapes = [
            ("wq_b", w.wq_b.dims()[0], d.heads * d.head_dim),
            ("wkv", w.wkv.dims()[0], d.head_dim),
            ("wo_a rows", w.wo_a.dims()[0], d.out_width()),
            ("wo_a columns", w.wo_a.dims()[1], d.group_width()),
        ];
        for (what, got, want) in shapes {
            if got != want {
                return Err(refuse(format!("{what} {got}, the header says {want}")));
            }
        }
        let wo_groups = (0..d.o_groups)
            .map(|g| w.wo_a.rows(g * d.o_lora_rank, d.o_lora_rank))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            dims,
            w,
            wo_groups,
            inv_freq,
            state: None,
            moves: 0,
        })
    }

    /// Multiplies a score before the softmax.
    pub(super) fn scale(&self) -> f32 {
        (self.dims.head_dim as f32).powf(-0.5)
    }

    /// The query factor of the rows `x2` `[seq, hidden]` F32, normalised: `[seq, q_lora_rank]`.
    pub(super) fn query_factor(&self, x2: &Tensor) -> Result<Tensor> {
        rms_norm(&self.w.wq_a.apply_here(x2)?, &self.w.q_norm, self.dims.eps)
    }

    /// The heads' queries from the factor `qr` `[seq, q_lora_rank]`, each head's tail rotated by
    /// the phases of its row: `[seq, heads * head_dim]`.
    pub(super) fn queries(&self, qr: &Tensor, cos: &[f32], sin: &[f32]) -> Result<Vec<f32>> {
        let d = self.dims;
        let q = self.w.wq_b.apply_here(qr)?;
        let mut q = q.flatten_all()?.to_vec1::<f32>()?;
        rotate_tails(&mut q, d.heads, d.head_dim, d.rope_dim, cos, sin);
        Ok(q)
    }

    /// The key rows of `x2` `[seq, hidden]` F32: projection, norm, rotation of the tail by the
    /// phases of the row, fp8 round trip. `[seq, head_dim]`.
    pub(super) fn key_rows(&self, x2: &Tensor, cos: &[f32], sin: &[f32]) -> Result<Vec<f32>> {
        let d = self.dims;
        let kv = rms_norm(&self.w.wkv.apply_here(x2)?, &self.w.kv_norm, d.eps)?;
        let mut kv = kv.flatten_all()?.to_vec1::<f32>()?;
        rotate_tails(&mut kv, 1, d.head_dim, d.rope_dim, cos, sin);
        fp8_round_trip(&mut kv, FP8_BLOCK);
        Ok(kv)
    }

    /// The heads' outputs `o` `[seq, heads * head_dim]`, each turned back by its query's phase,
    /// through the grouped low-rank projection. Group `g` maps its own lanes of the heads.
    /// Returns `[seq, o_groups * o_lora_rank]` on the host.
    pub(super) fn project_out(
        &self,
        mut o: Vec<f32>,
        cos: &[f32],
        sin: &[f32],
        seq: usize,
    ) -> Result<Tensor> {
        let d = self.dims;
        let back: Vec<f32> = sin.iter().map(|&s| 0.0 - s).collect();
        rotate_tails(&mut o, d.heads, d.head_dim, d.rope_dim, cos, &back);
        let p = d.group_width();
        let og = Tensor::from_vec(o, (seq, d.o_groups, p), &Device::Cpu)?;
        let mut parts = Vec::with_capacity(d.o_groups);
        for (g, wa) in self.wo_groups.iter().enumerate() {
            let slice = og.narrow(1, g, 1)?.reshape((seq, p))?;
            parts.push(wa.apply_here(&slice)?);
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        Tensor::cat(&refs, 1)
    }

    /// Tokens seen.
    pub fn len(&self) -> usize {
        self.state.as_ref().map_or(0, |s| s.len)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The groups' slices for the rows of `x` `[1, seq, hidden]`, which stand at positions
    /// `index_pos..index_pos + seq`. A forward at position 0 starts the state again. Any other
    /// must continue it. The result has the shape `[1, seq, o_groups * o_lora_rank]` and `x`'s
    /// device and type.
    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (b, seq, hidden) = x.dims3()?;
        if b != 1 {
            return Err(crate::tensor::Error::msg(
                "latent attention keeps 1 sequence's window",
            ));
        }
        if index_pos == 0 && self.state.is_some() {
            self.state = None;
            self.moves += 1;
        }
        if self.len() != index_pos {
            return Err(crate::tensor::Error::msg(format!(
                "latent attention holds {} tokens and cannot continue from position {index_pos}",
                self.len()
            )));
        }
        let (device, dtype) = (x.device(), x.dtype());
        let x2 = host_rows(x, seq, hidden)?;
        let d = self.dims;
        let (nh, hd) = (d.heads, d.head_dim);
        let (cos, sin) = phases(&self.inv_freq, index_pos..index_pos + seq);
        let q = self.queries(&self.query_factor(&x2)?, &cos, &sin)?;
        let kv = self.key_rows(&x2, &cos, &sin)?;

        // The kept rows come before the new ones. Row `old + i` is query `i`'s own.
        let mut state = self.state.take().unwrap_or_default();
        let old = state.window.len() / hd;
        let mut keys = std::mem::take(&mut state.window);
        keys.extend_from_slice(&kv);
        let o = attend_window(
            &q,
            &keys,
            &self.w.sink,
            (nh, hd),
            d.window,
            self.scale(),
            old,
        );
        let kept = (old + seq).min(d.window) * hd;
        state.window = keys.split_off(keys.len() - kept);
        state.len += seq;
        self.state = Some(state);

        self.project_out(o, &cos, &sin, seq)?
            .reshape((1, seq, d.out_width()))?
            .to_device(&device)?
            .to_dtype(dtype)
    }
}

/// The rows of `x` `[1, seq, hidden]` as `[seq, hidden]` F32 on the host.
pub(super) fn host_rows(x: &Tensor, seq: usize, hidden: usize) -> Result<Tensor> {
    x.to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .contiguous()?
        .reshape((seq, hidden))
}

/// The rotation frequency of each lane pair of `rope_dim` rotated lanes at base `base`.
///
/// With `original_context` above 0 the frequencies are stretched:
/// - a pair that turns more than `FAST_TURNS` times over that context keeps its frequency,
/// - a pair that turns fewer than `SLOW_TURNS` times has it divided by `factor`,
/// - the pairs between fade from one to the other along a linear ramp.
pub fn frequencies(rope_dim: usize, original_context: usize, base: f32, factor: f32) -> Vec<f32> {
    use std::f32::consts::PI;
    let mut freqs: Vec<f32> = (0..rope_dim / 2)
        .map(|i| 1.0 / base.powf((2 * i) as f32 / rope_dim as f32))
        .collect();
    if original_context > 0 {
        // The lane pair that turns `rot` times over the original context.
        let corrected = |rot: f32| {
            rope_dim as f32 * (original_context as f32 / (rot * 2.0 * PI)).ln() / (2.0 * base.ln())
        };
        let low = corrected(FAST_TURNS).floor().max(0.0);
        let high = corrected(SLOW_TURNS).ceil().min((rope_dim - 1) as f32);
        for (i, f) in freqs.iter_mut().enumerate() {
            let ramp = ((i as f32 - low) / (high - low).max(RAMP_FLOOR)).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *f = *f / factor * (1.0 - smooth) + *f * smooth;
        }
    }
    freqs
}

/// Turns over the original context above which a lane pair keeps its frequency.
const FAST_TURNS: f32 = 32.0;

/// Turns over the original context below which a lane pair's frequency is divided.
const SLOW_TURNS: f32 = 1.0;

/// The smallest width of the ramp between the 2 bounds.
const RAMP_FLOOR: f32 = 1e-3;

/// The phases of `positions`: `cos` and `sin` `[positions, inv_freq.len()]`.
pub(super) fn phases(
    inv_freq: &[f32],
    positions: impl IntoIterator<Item = usize>,
) -> (Vec<f32>, Vec<f32>) {
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for t in positions {
        for f in inv_freq {
            let a = t as f32 * f;
            cos.push(a.cos());
            sin.push(a.sin());
        }
    }
    (cos, sin)
}

/// Rotates the last `rd` lanes of every head of `rows` `[n, heads * hd]`. Lane pair
/// `(2j, 2j + 1)` of row `t` turns by the phase `cos[t * rd / 2 + j]`, `sin[t * rd / 2 + j]`.
/// The leading lanes of a head stay as they are.
pub(super) fn rotate_tails(
    rows: &mut [f32],
    heads: usize,
    hd: usize,
    rd: usize,
    cos: &[f32],
    sin: &[f32],
) {
    let half = rd / 2;
    rows.par_chunks_mut(heads * hd)
        .enumerate()
        .for_each(|(t, token)| {
            let (c, s) = (
                &cos[t * half..(t + 1) * half],
                &sin[t * half..(t + 1) * half],
            );
            for head in token.chunks_mut(hd) {
                let tail = &mut head[hd - rd..];
                for (j, (&cj, &sj)) in c.iter().zip(s).enumerate() {
                    let (x0, x1) = (tail[2 * j], tail[2 * j + 1]);
                    tail[2 * j] = x0 * cj - x1 * sj;
                    tail[2 * j + 1] = x0 * sj + x1 * cj;
                }
            }
        });
}

/// Every query head of `q` `[seq, heads * hd]` against the key rows `keys` `[n, hd]`, which are
/// the values too. Query `i` is key row `q_off + i` and sees rows
/// `max(0, q_off + i + 1 - window)..=q_off + i`. The sink logit of its head takes softmax mass
/// and adds no value. Returns `[seq, heads * hd]`.
fn attend_window(
    q: &[f32],
    keys: &[f32],
    sink: &[f32],
    (heads, hd): (usize, usize),
    window: usize,
    scale: f32,
    q_off: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; q.len()];
    out.par_chunks_mut(hd).enumerate().for_each(|(pair, o)| {
        let (i, h) = (pair / heads, pair % heads);
        let iq = q_off + i;
        let lo = iq.saturating_sub(window - 1);
        let qv = &q[pair * hd..(pair + 1) * hd];
        let seen = &keys[lo * hd..(iq + 1) * hd];
        // The logits over the seen rows and their running maximum, the sink included.
        let mut m = sink[h];
        let logits: Vec<f32> = seen
            .chunks_exact(hd)
            .map(|key| {
                let mut dot = 0f32;
                for (a, k) in qv.iter().zip(key) {
                    dot += a * k;
                }
                let l = dot * scale;
                if l > m {
                    m = l;
                }
                l
            })
            .collect();
        // The softmax denominator with the sink, and the rows summed by their weights.
        let mut denom = (sink[h] - m).exp();
        for (l, key) in logits.iter().zip(seen.chunks_exact(hd)) {
            let w = (l - m).exp();
            denom += w;
            for (acc, &v) in o.iter_mut().zip(key) {
                *acc += w * v;
            }
        }
        for v in o.iter_mut() {
            *v /= denom;
        }
    });
    out
}

/// Rounds every value of `v` to what fp8 e4m3 holds under a power-of-two scale per `block`
/// values, and reads it back. The scale puts the block's largest magnitude at or below the
/// format's largest. `v.len()` is a multiple of `block`.
pub(crate) fn fp8_round_trip(v: &mut [f32], block: usize) {
    for seg in v.chunks_exact_mut(block) {
        let amax = seg.iter().fold(0f32, |m, &e| m.max(e.abs()));
        let scale = pow2_ceil((amax / FP8_E4M3_MAX).max(SCALE_FLOOR));
        for e in seg.iter_mut() {
            let clamped = (*e / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX);
            *e = e4m3_round(clamped) * scale;
        }
    }
}

/// The smallest power of two at or above `x`, for `x > 0`.
pub(crate) fn pow2_ceil(x: f32) -> f32 {
    x.log2().ceil().exp2()
}

/// `a` rounded to the nearest e4m3fn value, ties to even. `a` is finite and within the format's
/// largest magnitude. The format has 3 mantissa bits, its smallest normal is 2^-6 and its
/// subnormal step 2^-9.
fn e4m3_round(a: f32) -> f32 {
    const MIN_NORMAL: f32 = 0.015625;
    const SUB_STEP: f32 = 0.001953125;
    if a == 0.0 {
        return 0.0;
    }
    let mag = a.abs();
    // The step between the values of `mag`'s binade is 2^(e - 3).
    let step = if mag < MIN_NORMAL {
        SUB_STEP
    } else {
        2f32.powi(mag.log2().floor() as i32 - 3)
    };
    let r = round_half_to_even(mag / step);
    (a.signum() * r * step).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
}

/// `y` rounded to the nearest integer, a tie to the even one.
fn round_half_to_even(y: f32) -> f32 {
    let f = y.floor();
    let d = y - f;
    if d < 0.5 {
        f
    } else if d > 0.5 {
        f + 1.0
    } else if (f as i64) % 2 == 0 {
        f
    } else {
        f + 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::model::deepseek_v41::attention::{act_quant_fp8_e4m3, Ratio0Attention};
    use crate::inference::model::deepseek_v41::cache::AttnCache;
    use crate::inference::model::deepseek_v41::model::rope_table;

    const HIDDEN: usize = 48;
    const Q_RANK: usize = 24;
    const THETA: f32 = 10_000.0;

    fn dims() -> LatentMqaDims {
        LatentMqaDims {
            heads: 4,
            head_dim: 64,
            rope_dim: 16,
            o_groups: 2,
            o_lora_rank: 12,
            window: 5,
            eps: 1e-6,
        }
    }

    fn values(n: usize, seed: u64, amp: f32) -> Vec<f32> {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 2.0 * amp
            })
            .collect()
    }

    fn dense(out: usize, inp: usize, seed: u64) -> Projection {
        let w = Tensor::from_vec(values(out * inp, seed, 0.4), (out, inp), &Device::Cpu).unwrap();
        Projection::Dense(w)
    }

    fn norm(n: usize, seed: u64) -> Tensor {
        let scale: Vec<f32> = values(n, seed, 0.5).iter().map(|v| 1.0 + v).collect();
        Tensor::from_vec(scale, n, &Device::Cpu).unwrap()
    }

    fn input(seq: usize, seed: u64) -> Tensor {
        Tensor::from_vec(
            values(seq * HIDDEN, seed, 1.0),
            (1, seq, HIDDEN),
            &Device::Cpu,
        )
        .unwrap()
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn window_of(cache: &AttnCache) -> Vec<f32> {
        cache.win.iter().flatten().copied().collect()
    }

    fn weights(old: &Ratio0Attention, sink: Vec<f32>) -> LatentMqaWeights {
        LatentMqaWeights {
            wq_a: old.wq_a.clone(),
            q_norm: old.q_norm.clone(),
            wq_b: old.wq_b.clone(),
            wkv: old.wkv.clone(),
            kv_norm: old.kv_norm.clone(),
            sink,
            wo_a: old.wo_a.clone(),
        }
    }

    /// The frequencies of the window layers, unstretched.
    fn plain() -> Vec<f32> {
        frequencies(dims().rope_dim, 0, THETA, 1.0)
    }

    /// The V4.1 attention and the mixer over the same weights.
    fn pair(seed: u64) -> (Ratio0Attention, LatentMqa) {
        let d = dims();
        let sink = values(d.heads, seed + 5, 1.0);
        let old = Ratio0Attention {
            wq_a: dense(Q_RANK, HIDDEN, seed),
            q_norm: norm(Q_RANK, seed + 1),
            wq_b: dense(d.heads * d.head_dim, Q_RANK, seed + 2),
            wkv: dense(d.head_dim, HIDDEN, seed + 3),
            kv_norm: norm(d.head_dim, seed + 4),
            attn_sink: Tensor::from_vec(sink.clone(), d.heads, &Device::Cpu).unwrap(),
            wo_a: dense(d.out_width(), d.group_width(), seed + 6),
            wo_b: dense(HIDDEN, d.out_width(), seed + 7),
            n_heads: d.heads,
            head_dim: d.head_dim,
            rope_head_dim: d.rope_dim,
            o_groups: d.o_groups,
            o_lora_rank: d.o_lora_rank,
            window_size: d.window,
            eps: d.eps,
        };
        let new = LatentMqa::new(d, weights(&old, sink), plain()).unwrap();
        (old, new)
    }

    /// The mixer's rows through `wo_b` as the V4.1 attention applies it, `[seq * hidden]`.
    fn full(old: &Ratio0Attention, mixed: &Tensor) -> Vec<f32> {
        let (_, seq, width) = mixed.dims3().unwrap();
        flat(
            &old.wo_b
                .apply_here(&mixed.reshape((seq, width)).unwrap())
                .unwrap(),
        )
    }

    #[test]
    fn a_prompt_gives_the_bits_of_the_v41_window_attention() {
        let (old, mut new) = pair(11);
        let d = dims();
        // More tokens than the window holds.
        let seq = 2 * d.window - 1;
        let x = input(seq, 3);
        let (cos, sin) = rope_table(d.rope_dim, seq, 0, THETA, 1.0).unwrap();
        let mut cache = AttnCache::default();
        let want = old
            .forward_prefill_cached(&x, &cos, &sin, Some(&mut cache))
            .unwrap();
        let got = new.forward(&x, 0).unwrap();
        assert_eq!(got.dims(), [1, seq, d.out_width()]);
        assert_eq!(bits(&full(&old, &got)), bits(&flat(&want)));
        let state = new.state.as_ref().unwrap();
        assert_eq!(state.len, seq);
        assert_eq!(state.window.len(), d.window * d.head_dim);
        assert_eq!(bits(&state.window), bits(&window_of(&cache)));
    }

    #[test]
    fn decoding_gives_the_bits_of_the_v41_decode() {
        let (old, mut new) = pair(23);
        let d = dims();
        let prompt = input(3, 5);
        let (cos, sin) = rope_table(d.rope_dim, 3, 0, THETA, 1.0).unwrap();
        let mut cache = AttnCache::default();
        old.forward_prefill_cached(&prompt, &cos, &sin, Some(&mut cache))
            .unwrap();
        new.forward(&prompt, 0).unwrap();
        // Past the window, so the oldest rows leave it.
        for pos in 3..3 + 2 * d.window {
            let x = input(1, 100 + pos as u64);
            let (cos, sin) = rope_table(d.rope_dim, pos + 1, 0, THETA, 1.0).unwrap();
            let want = old.forward_decode(&x, &cos, &sin, pos, &mut cache).unwrap();
            let got = new.forward(&x, pos).unwrap();
            assert_eq!(
                bits(&full(&old, &got)),
                bits(&flat(&want)),
                "token at {pos}"
            );
            let window = &new.state.as_ref().unwrap().window;
            assert_eq!(bits(window), bits(&window_of(&cache)), "window at {pos}");
        }
    }

    #[test]
    fn a_prompt_that_continues_the_state_gives_the_bits_of_the_v41_continuation() {
        let (old, mut new) = pair(37);
        let d = dims();
        let (first, second) = (d.window - 1, d.window + 1);
        let (cos, sin) = rope_table(d.rope_dim, first + second, 0, THETA, 1.0).unwrap();
        let mut cache = AttnCache::default();
        let x = input(first, 7);
        old.forward_prefill_cached(&x, &cos, &sin, Some(&mut cache))
            .unwrap();
        new.forward(&x, 0).unwrap();
        cache.base = first;
        let x = input(second, 8);
        let want = old
            .forward_prefill_cached(&x, &cos, &sin, Some(&mut cache))
            .unwrap();
        let got = new.forward(&x, first).unwrap();
        assert_eq!(bits(&full(&old, &got)), bits(&flat(&want)));
        let state = new.state.as_ref().unwrap();
        assert_eq!(state.len, first + second);
        assert_eq!(bits(&state.window), bits(&window_of(&cache)));
    }

    #[test]
    fn the_fp8_round_trip_gives_the_bits_of_the_v41_one() {
        // Blocks of ordinary, of large and of tiny magnitudes, then a block of zeros.
        let mut v = values(2 * FP8_BLOCK, 41, 3.0);
        v.extend(values(FP8_BLOCK, 42, 3.0e4));
        v.extend(values(FP8_BLOCK, 43, 1.0e-6));
        v.extend(vec![0f32; FP8_BLOCK]);
        let n = v.len();
        let t = Tensor::from_vec(v.clone(), (n / FP8_BLOCK, FP8_BLOCK), &Device::Cpu).unwrap();
        let want = flat(&act_quant_fp8_e4m3(&t, FP8_BLOCK).unwrap());
        fp8_round_trip(&mut v, FP8_BLOCK);
        assert_eq!(bits(&v), bits(&want));
        assert!(v[n - FP8_BLOCK..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn values_round_to_the_e4m3_grid_with_ties_to_even() {
        assert_eq!(e4m3_round(FP8_E4M3_MAX), FP8_E4M3_MAX);
        // Steps of 2 between 16 and 32: 17 and 19 are ties.
        assert_eq!(e4m3_round(17.0), 16.0);
        assert_eq!(e4m3_round(19.0), 20.0);
        assert_eq!(e4m3_round(-17.0), -16.0);
        // 1.5 subnormal steps of 2^-9: a tie, to 2 steps.
        assert_eq!(e4m3_round(0.0029296875), 0.00390625);
        assert_eq!(pow2_ceil(3.0), 4.0);
    }

    #[test]
    fn the_window_restarts_at_position_zero_and_refuses_a_gap() {
        let (_, mut new) = pair(51);
        new.forward(&input(3, 1), 0).unwrap();
        assert_eq!(new.len(), 3);
        assert!(new.forward(&input(1, 2), 5).is_err());
        assert_eq!(new.len(), 3);
        new.forward(&input(2, 3), 0).unwrap();
        assert_eq!(new.len(), 2);
        assert_eq!(new.moves, 1);
        assert_eq!(
            new.state.as_ref().unwrap().window.len(),
            2 * dims().head_dim
        );
    }

    #[test]
    fn a_restored_snapshot_decodes_the_same_bits() {
        use super::super::mixer::Mixer;
        let (_, new) = pair(61);
        let mut mixer = Mixer::LatentMqa(new);
        assert!(mixer.is_recurrent());
        mixer.forward(&input(4, 1), 0).unwrap();
        let mark = mixer.snapshot().unwrap().unwrap();
        let x = input(1, 2);
        let first = flat(&mixer.forward(&x, 4).unwrap());
        mixer.restore(&mark).unwrap();
        let again = flat(&mixer.forward(&x, 4).unwrap());
        assert_eq!(bits(&first), bits(&again));
    }

    #[test]
    fn a_wrong_width_is_refused() {
        let (old, _) = pair(71);
        let d = dims();
        let sink = || vec![0f32; d.heads];
        assert!(LatentMqa::new(d, weights(&old, sink()), plain()).is_ok());
        let short = vec![0f32; d.heads - 1];
        assert!(LatentMqa::new(d, weights(&old, short), plain()).is_err());
        let no_window = LatentMqaDims { window: 0, ..d };
        assert!(LatentMqa::new(no_window, weights(&old, sink()), plain()).is_err());
        let fewer = plain()[1..].to_vec();
        assert!(LatentMqa::new(d, weights(&old, sink()), fewer).is_err());
        let groups = LatentMqaDims { o_groups: 3, ..d };
        assert!(LatentMqa::new(groups, weights(&old, sink()), plain()).is_err());
    }

    #[test]
    fn the_phases_give_the_bits_of_the_v41_rope_tables() {
        let rd = dims().rope_dim;
        let n = 300;
        // Unstretched, then stretched past an original context shorter than the positions.
        for (original, factor) in [(0, 1.0), (64, 4.0)] {
            let (cos, sin) = rope_table(rd, n, original, THETA, factor).unwrap();
            let (c, s) = phases(&frequencies(rd, original, THETA, factor), 0..n);
            assert_eq!(
                bits(&c),
                bits(&flat(&cos)),
                "cos, original context {original}"
            );
            assert_eq!(
                bits(&s),
                bits(&flat(&sin)),
                "sin, original context {original}"
            );
        }
    }
}
