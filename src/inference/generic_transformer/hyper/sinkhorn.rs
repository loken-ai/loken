//! The Sinkhorn wide residual: the second kind of wide residual, beside the low-rank gate.
//!
//! The residual is `streams` parallel copies of the hidden state.
//! Before a sublayer, a pre mix collapses the copies into one input.
//! After it, a post mix writes the output back to every copy.
//! The write-back also mixes the copies through a `streams x streams` comb matrix.
//!
//! The pre, post and comb coefficients come from one full-rank projection of the whole residual.
//! Sinkhorn iterations make the comb doubly stochastic.
//! Every coefficient is computed in f32 on the host, one token at a time.

use super::super::config::{meta_f32, meta_u32};
use crate::tensor::quantized::gguf_file;
use crate::tensor::{DType, Device, Error, Result, Tensor};

/// The Sinkhorn wide residual's parameters, from the file's `hyper_connection.*` metadata.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SinkhornMeta {
    /// Parallel copies of the hidden state.
    pub streams: usize,
    /// Sinkhorn passes over the comb matrix.
    pub iterations: usize,
    /// The floor added in the pre mix and in every Sinkhorn normalisation.
    pub eps: f32,
}

impl SinkhornMeta {
    /// The metadata of `arch`, `None` when the file declares a low-rank gate, fewer than 2
    /// streams, no Sinkhorn pass or no epsilon.
    ///
    /// The stream count and the passes are read under each spelling a converter has written.
    pub fn from_gguf(content: &gguf_file::Content, arch: &str) -> Option<Self> {
        if meta_u32(content, arch, "hyper_connection.low_rank").is_some() {
            return None;
        }
        let first = |keys: &[&str]| keys.iter().find_map(|k| meta_u32(content, arch, k));
        let streams = first(&["hyper_connection.count", "hyper_connection_mult"])? as usize;
        let iterations = first(&[
            "hyper_connection.sinkhorn_iterations",
            "hyper_connection_sinkhorn_iters",
        ])? as usize;
        let eps = meta_f32(content, arch, "hyper_connection.epsilon")?;
        (streams > 1 && iterations > 0).then_some(Self {
            streams,
            iterations,
            eps,
        })
    }

    /// Rows of the coefficient projection: `streams` pre, `streams` post and `streams^2` comb.
    pub fn mix_rows(&self) -> usize {
        (2 + self.streams) * self.streams
    }
}

/// The pre, post and comb coefficients of one sublayer, per token.
pub struct SinkhornMixes {
    /// `[n][streams]`
    pub pre: Vec<Vec<f32>>,
    /// `[n][streams]`
    pub post: Vec<Vec<f32>>,
    /// `[n][streams][streams]`, row `i` column `j`.
    pub comb: Vec<Vec<Vec<f32>>>,
}

/// One mixer of the Sinkhorn wide residual: the coefficient projection, its 3 scales and its
/// biases.
pub struct SinkhornMixer {
    /// `[mix_rows, streams * dim]` F32 on the host.
    func: Tensor,
    /// The scales of the pre, post and comb sets, in that order.
    scale: Vec<f32>,
    /// `[mix_rows]`
    base: Vec<f32>,
    meta: SinkhornMeta,
    dim: usize,
    /// The epsilon of the RMS statistic taken over the whole residual.
    norm_eps: f32,
}

impl SinkhornMixer {
    /// `func` is `[mix_rows, streams * dim]`, `scale` holds 3 values and `base` `mix_rows`.
    pub fn new(
        func: &Tensor,
        scale: &Tensor,
        base: &Tensor,
        meta: SinkhornMeta,
        dim: usize,
        norm_eps: f32,
    ) -> Result<Self> {
        let host = |t: &Tensor| -> Result<Tensor> {
            t.to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .contiguous()
        };
        let func = host(func)?;
        let scale = host(scale)?.flatten_all()?.to_vec1::<f32>()?;
        let base = host(base)?.flatten_all()?.to_vec1::<f32>()?;
        let rows = meta.mix_rows();
        let shaped = func.dims() == [rows, meta.streams * dim];
        if !shaped || scale.len() != 3 || base.len() != rows {
            return Err(Error::msg(format!(
                "Sinkhorn mixer: projection {:?}, {} scales and {} biases do not match {} streams of {dim}",
                func.dims(),
                scale.len(),
                base.len(),
                meta.streams
            )));
        }
        Ok(Self {
            func,
            scale,
            base,
            meta,
            dim,
            norm_eps,
        })
    }

    /// Streams the mixer mixes.
    pub fn streams(&self) -> usize {
        self.meta.streams
    }

    /// Width of one stream.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The coefficients for every row of `r` `[b, seq, streams * dim]`.
    ///
    /// Each token takes one RMS statistic over its whole residual.
    /// The projection of the residual is scaled by that statistic.
    /// A batch of tokens takes the projection in 8 lanes, a single token in one running sum.
    pub fn mixes(&self, r: &Tensor) -> Result<SinkhornMixes> {
        use rayon::prelude::*;
        let hc = self.meta.streams;
        let flat = hc * self.dim;
        let (b, s, wide) = r.dims3()?;
        if wide != flat {
            return Err(Error::msg(format!(
                "Sinkhorn mixer: a residual of {wide} for {hc} streams of {}",
                self.dim
            )));
        }
        let n = b * s;
        let xf = r.reshape((n, flat))?;
        let rows = dense_f32(&xf)?;
        let w = dense_f32(&self.func)?;
        let mix_hc = self.meta.mix_rows();
        let mixes: Vec<f32> = if n > 1 {
            let mut out = vec![0f32; n * mix_hc];
            out.par_chunks_mut(mix_hc).enumerate().for_each(|(t, o)| {
                let row = &rows[t * flat..(t + 1) * flat];
                for (m, v) in o.iter_mut().enumerate() {
                    *v = dot8(row, &w[m * flat..(m + 1) * flat]);
                }
            });
            out
        } else {
            (0..mix_hc * n)
                .map(|m| {
                    w[m * flat..(m + 1) * flat]
                        .iter()
                        .zip(rows.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                })
                .collect()
        };

        let (scale, base) = (&self.scale, &self.base);
        let (iters, hc_eps, norm_eps) = (self.meta.iterations, self.meta.eps, self.norm_eps);
        // Tokens are independent: each is computed in the order a single token would be.
        let per_token: Vec<(Vec<f32>, Vec<f32>, Vec<Vec<f32>>)> = (0..n)
            .into_par_iter()
            .map(|t| {
                let row = &rows[t * flat..(t + 1) * flat];
                let ss: f32 = if n > 1 {
                    dot8(row, row)
                } else {
                    row.iter().map(|v| v * v).sum()
                };
                let rsqrt = 1.0 / (ss / flat as f32 + norm_eps).sqrt();
                let m: Vec<f32> = mixes[t * mix_hc..(t + 1) * mix_hc]
                    .iter()
                    .map(|v| v * rsqrt)
                    .collect();
                let pre_t: Vec<f32> = (0..hc)
                    .map(|c| sigmoid(m[c] * scale[0] + base[c]) + hc_eps)
                    .collect();
                let post_t: Vec<f32> = (0..hc)
                    .map(|c| 2.0 * sigmoid(m[hc + c] * scale[1] + base[hc + c]))
                    .collect();
                let mut cm = vec![vec![0f32; hc]; hc];
                for i in 0..hc {
                    for j in 0..hc {
                        cm[i][j] = m[2 * hc + i * hc + j] * scale[2] + base[2 * hc + i * hc + j];
                    }
                }
                sinkhorn(&mut cm, iters, hc_eps);
                (pre_t, post_t, cm)
            })
            .collect();
        let mut pre = Vec::with_capacity(n);
        let mut post = Vec::with_capacity(n);
        let mut comb = Vec::with_capacity(n);
        for (p, q, c) in per_token {
            pre.push(p);
            post.push(q);
            comb.push(c);
        }
        Ok(SinkhornMixes { pre, post, comb })
    }

    /// The sublayer input `[b, seq, dim]`: the streams of `r` `[b, seq, streams * dim]` summed,
    /// stream `c` of token `t` weighted by `pre[t][c]`.
    pub fn pre(&self, r: &Tensor, pre: &[Vec<f32>]) -> Result<Tensor> {
        use rayon::prelude::*;
        let (b, s, _) = r.dims3()?;
        let (hc, d) = (self.meta.streams, self.dim);
        let xv = dense_f32(r)?;
        let mut out = vec![0f32; b * s * d];
        out.par_chunks_mut(d).enumerate().for_each(|(t, o)| {
            for c in 0..hc {
                let w = pre[t][c];
                let base = (t * hc + c) * d;
                for k in 0..d {
                    o[k] += w * xv[base + k];
                }
            }
        });
        Tensor::from_vec(out, (b, s, d), &Device::Cpu)
    }

    /// The new residual `[b, seq, streams * dim]` from the sublayer output `out` `[b, seq, dim]`
    /// and the residual `residual` the sublayer read.
    ///
    /// Stream `j` is `post[j] * out + sum_i comb[i][j] * residual_i`, per token.
    pub fn post(
        &self,
        out: &Tensor,
        residual: &Tensor,
        post: &[Vec<f32>],
        comb: &[Vec<Vec<f32>>],
    ) -> Result<Tensor> {
        use rayon::prelude::*;
        let (b, s, d) = out.dims3()?;
        let hc = self.meta.streams;
        let xv = dense_f32(out)?;
        let rv = dense_f32(residual)?;
        let mut res = vec![0f32; b * s * hc * d];
        res.par_chunks_mut(hc * d)
            .enumerate()
            .for_each(|(t, token)| {
                for j in 0..hc {
                    let o = j * d;
                    let pj = post[t][j];
                    for k in 0..d {
                        token[o + k] = pj * xv[t * d + k];
                    }
                    for (i, crow) in comb[t].iter().enumerate() {
                        let c = crow[j];
                        let r = (t * hc + i) * d;
                        for k in 0..d {
                            token[o + k] += c * rv[r + k];
                        }
                    }
                }
            });
        Tensor::from_vec(res, (b, s, hc * d), &Device::Cpu)
    }

    /// `r` `[b, seq, streams * dim]` with `delta` `[dim]` added to every stream, stream `j`
    /// scaled by `post[j]`.
    ///
    /// The result is what [`Self::post`] gives when `delta` is part of the sublayer output.
    pub fn add_to_streams(&self, r: &Tensor, post: &[f32], delta: &[f32]) -> Result<Tensor> {
        let (b, s, wide) = r.dims3()?;
        let (hc, d) = (self.meta.streams, self.dim);
        let mut out = r.flatten_all()?.to_vec1::<f32>()?;
        for (j, copy) in out.chunks_mut(d).enumerate() {
            let pj = post[j % hc];
            for (o, v) in copy.iter_mut().zip(delta) {
                *o += pj * v;
            }
        }
        Tensor::from_vec(out, (b, s, wide), &Device::Cpu)
    }
}

/// The elements of a host f32 tensor in row-major order: borrowed when the tensor owns exactly
/// them, copied otherwise.
fn dense_f32(t: &Tensor) -> Result<std::borrow::Cow<'_, [f32]>> {
    if t.is_contiguous() {
        if let Ok(v) = t.cpu_f32_data() {
            if v.len() == t.elem_count() {
                return Ok(std::borrow::Cow::Borrowed(v));
            }
        }
    }
    Ok(std::borrow::Cow::Owned(t.flatten_all()?.to_vec1::<f32>()?))
}

/// The dot product of `a` and `b` in 8 independent lanes, summed lane by lane, then the tail.
fn dot8(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0f32; 8];
    let chunks = a.len() / 8;
    for c in 0..chunks {
        for l in 0..8 {
            lanes[l] += a[c * 8 + l] * b[c * 8 + l];
        }
    }
    let mut sum = lanes.iter().sum::<f32>();
    for i in chunks * 8..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Make `cm` doubly stochastic: a row softmax plus the floor, 1 column normalisation, then
/// `iters - 1` passes of a row and a column normalisation.
fn sinkhorn(cm: &mut [Vec<f32>], iters: usize, eps: f32) {
    let hc = cm.len();
    for row in cm.iter_mut() {
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut den = 0f32;
        for v in row.iter_mut() {
            *v = (*v - mx).exp();
            den += *v;
        }
        for v in row.iter_mut() {
            *v = *v / den + eps;
        }
    }
    normalise_cols(cm, hc, eps);
    for _ in 0..iters.saturating_sub(1) {
        normalise_rows(cm, eps);
        normalise_cols(cm, hc, eps);
    }
}

/// Each row divided by its sum plus `eps`.
fn normalise_rows(cm: &mut [Vec<f32>], eps: f32) {
    for row in cm.iter_mut() {
        let sum: f32 = row.iter().sum::<f32>() + eps;
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

/// Each column divided by its sum plus `eps`.
fn normalise_cols(cm: &mut [Vec<f32>], hc: usize, eps: f32) {
    for j in 0..hc {
        let sum: f32 = (0..hc).map(|i| cm[i][j]).sum::<f32>() + eps;
        for row in cm.iter_mut() {
            row[j] /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::model::deepseek_v41::hyper_connections as v41;
    use crate::tensor::quantized::gguf_file::{Content, Value, VersionedMagic};
    use std::collections::HashMap;

    const HC: usize = 4;
    const DIM: usize = 20;
    const ITERS: usize = 20;
    const HC_EPS: f32 = 1e-6;
    const NORM_EPS: f32 = 1e-6;

    /// `n` values in `[-1, 1)` drawn from `seed` by a 64-bit linear congruential generator.
    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn meta() -> SinkhornMeta {
        SinkhornMeta {
            streams: HC,
            iterations: ITERS,
            eps: HC_EPS,
        }
    }

    /// The projection, scales and biases of one sublayer, drawn from `seed`.
    fn weights(seed: u64) -> (Tensor, Tensor, Tensor) {
        let rows = meta().mix_rows();
        let func =
            Tensor::from_vec(noise(rows * HC * DIM, seed), (rows, HC * DIM), &Device::Cpu).unwrap();
        let scale = Tensor::from_vec(vec![0.7f32, 1.3, 2.1], 3, &Device::Cpu).unwrap();
        let base = Tensor::from_vec(noise(rows, seed + 1), rows, &Device::Cpu).unwrap();
        (func, scale, base)
    }

    fn mixer(seed: u64) -> SinkhornMixer {
        let (func, scale, base) = weights(seed);
        SinkhornMixer::new(&func, &scale, &base, meta(), DIM, NORM_EPS).unwrap()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// The coefficients, the collapsed input and the new residual have the bits of the V4.1
    /// hyper-connections on the same residual and weights, for a batch and for a single token.
    #[test]
    fn the_mixes_pre_and_post_give_the_bits_of_the_v41_hyper_connections() {
        for (seq, seed) in [(7usize, 11u64), (1, 23)] {
            let (func, scale, base) = weights(seed);
            let m = SinkhornMixer::new(&func, &scale, &base, meta(), DIM, NORM_EPS).unwrap();
            let values = noise(seq * HC * DIM, seed + 2);
            let r = Tensor::from_vec(values.clone(), (1, seq, HC * DIM), &Device::Cpu).unwrap();
            let x = Tensor::from_vec(values, (1, seq, HC, DIM), &Device::Cpu).unwrap();

            let got = m.mixes(&r).unwrap();
            let want = v41::hc_mixes(
                &x,
                &func,
                &flat(&scale),
                &flat(&base),
                HC,
                ITERS,
                NORM_EPS,
                HC_EPS,
            )
            .unwrap();
            for t in 0..seq {
                assert_eq!(bits(&got.pre[t]), bits(&want.pre[t]), "pre, token {t}");
                assert_eq!(bits(&got.post[t]), bits(&want.post[t]), "post, token {t}");
                for i in 0..HC {
                    assert_eq!(
                        bits(&got.comb[t][i]),
                        bits(&want.comb[t][i]),
                        "comb row {i}, token {t}"
                    );
                }
            }

            let collapsed = m.pre(&r, &got.pre).unwrap();
            assert_eq!(collapsed.dims(), [1, seq, DIM]);
            let want_collapsed = v41::hc_pre(&x, &want.pre).unwrap();
            assert_eq!(bits(&flat(&collapsed)), bits(&flat(&want_collapsed)));

            let out =
                Tensor::from_vec(noise(seq * DIM, seed + 3), (1, seq, DIM), &Device::Cpu).unwrap();
            let next = m.post(&out, &r, &got.post, &got.comb).unwrap();
            assert_eq!(next.dims(), [1, seq, HC * DIM]);
            let want_next = v41::hc_post(&out, &x, &want.post, &want.comb, HC).unwrap();
            assert_eq!(bits(&flat(&next)), bits(&flat(&want_next)));
        }
    }

    /// After the Sinkhorn passes, which end on a column pass, every column of the comb sums to
    /// 1, the pre weights lie in `(eps, 1 + eps)` and the post weights in `(0, 2)`.
    #[test]
    fn the_comb_columns_sum_to_one_and_the_mixes_are_bounded() {
        let m = mixer(5);
        let seq = 3;
        let r =
            Tensor::from_vec(noise(seq * HC * DIM, 6), (1, seq, HC * DIM), &Device::Cpu).unwrap();
        let mixes = m.mixes(&r).unwrap();
        for t in 0..seq {
            for i in 0..HC {
                let col: f32 = (0..HC).map(|k| mixes.comb[t][k][i]).sum();
                assert!(
                    (col - 1.0).abs() < 1e-3,
                    "token {t} column {i} sums to {col}"
                );
            }
            for c in 0..HC {
                let (p, q) = (mixes.pre[t][c], mixes.post[t][c]);
                assert!(p > HC_EPS && p < 1.0 + HC_EPS, "pre {p}");
                assert!(q > 0.0 && q < 2.0, "post {q}");
            }
        }
    }

    /// Adding a share to the residual after the fact is what `post` gives when the share is in
    /// the sublayer output: `post(x + d) - post(x) = post_j * d` per stream.
    #[test]
    fn adding_to_the_streams_is_what_post_gives_a_larger_output() {
        let m = mixer(9);
        let x = Tensor::from_vec(
            (0..DIM).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            (1, 1, DIM),
            &Device::Cpu,
        )
        .unwrap();
        let delta: Vec<f32> = (0..DIM).map(|i| 1.0 - i as f32 * 0.07).collect();
        let residual = Tensor::from_vec(
            (0..HC * DIM).map(|i| (i as f32).sin()).collect::<Vec<_>>(),
            (1, 1, HC * DIM),
            &Device::Cpu,
        )
        .unwrap();
        let post = vec![vec![0.5f32, 1.5, -0.25, 0.75]];
        let comb = vec![vec![vec![0.2f32; HC]; HC]];
        let plain = m.post(&x, &residual, &post, &comb).unwrap();
        let both = Tensor::from_vec(
            flat(&x)
                .iter()
                .zip(&delta)
                .map(|(a, b)| a + b)
                .collect::<Vec<_>>(),
            (1, 1, DIM),
            &Device::Cpu,
        )
        .unwrap();
        let want = flat(&m.post(&both, &residual, &post, &comb).unwrap());
        let got = flat(&m.add_to_streams(&plain, &post[0], &delta).unwrap());
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "{g} vs {w}");
        }
        // The same share added by the V4.1 hyper-connections gives the same bits.
        let plain4 = plain.reshape((1, 1, HC, DIM)).unwrap();
        let v41_added = flat(&v41::hc_add(&plain4, &post[0], &delta).unwrap());
        assert_eq!(bits(&got), bits(&v41_added));
    }

    fn content(meta: &[(&str, Value)]) -> Content {
        let mut metadata = HashMap::new();
        for (k, v) in meta {
            metadata.insert(format!("a.{k}"), v.clone());
        }
        Content {
            magic: VersionedMagic::GgufV3,
            metadata,
            tensor_infos: HashMap::new(),
            tensor_data_offset: 0,
            mmap_owner: None,
        }
    }

    /// Both spellings of the stream count and of the passes are read, and a file with a
    /// low-rank gate is not of this kind.
    #[test]
    fn the_metadata_names_the_sinkhorn_kind() {
        let dotted = content(&[
            ("hyper_connection.count", Value::U32(4)),
            ("hyper_connection.sinkhorn_iterations", Value::U32(20)),
            ("hyper_connection.epsilon", Value::F32(1e-6)),
        ]);
        assert_eq!(SinkhornMeta::from_gguf(&dotted, "a"), Some(meta()));
        let underscored = content(&[
            ("hyper_connection_mult", Value::U32(4)),
            ("hyper_connection_sinkhorn_iters", Value::U32(20)),
            ("hyper_connection.epsilon", Value::F32(1e-6)),
        ]);
        assert_eq!(SinkhornMeta::from_gguf(&underscored, "a"), Some(meta()));
        let low_rank = content(&[
            ("hyper_connection.count", Value::U32(4)),
            ("hyper_connection.low_rank", Value::U32(8)),
            ("hyper_connection.sinkhorn_iterations", Value::U32(20)),
            ("hyper_connection.epsilon", Value::F32(1e-6)),
        ]);
        assert_eq!(SinkhornMeta::from_gguf(&low_rank, "a"), None);
        let one = content(&[
            ("hyper_connection.count", Value::U32(1)),
            ("hyper_connection.sinkhorn_iterations", Value::U32(20)),
            ("hyper_connection.epsilon", Value::F32(1e-6)),
        ]);
        assert_eq!(SinkhornMeta::from_gguf(&one, "a"), None);
    }

    /// A projection of the wrong height is refused.
    #[test]
    fn a_projection_of_the_wrong_shape_is_refused() {
        let (func, scale, base) = weights(3);
        let short = func.narrow(0, 0, meta().mix_rows() - 1).unwrap();
        assert!(SinkhornMixer::new(&short, &scale, &base, meta(), DIM, NORM_EPS).is_err());
    }
}
