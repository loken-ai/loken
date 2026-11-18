//! The rows of a draft verify through a layer that reads a donor's cache on the host, at the heads
//! of a shared-cache model: one masked attention over all the rows, against one decode-step
//! attention per row over the donor's rows up to its own position, read in place, without a
//! tensor between the steps: on a global layer the rows of a key-value head share one read of the
//! donor's rows, on a windowed layer 2 regions make every score, then every output.
//! A plain decode step is timed beside them.
//!
//! The heads come from the GGUF header `--file`: the global layers' head width and the windowed layers'
//! head width and window, each a layer kind of its own.
//! Without a file, `--heads`, `--kv-heads` and `--head-dim` give one global layer kind.
//! `--context` is the donor's rows before the verify and `--block` the verify widths.
//! A windowed donor holds its window and the verify's rows, the fewest the masked attention
//! reads.
//! Every call reads the donor's rows as a new view of a larger buffer, as the cache hands them to
//! each forward.

use super::prefill_layer::ratio_note;
use super::stats::run_for;
use super::{Context, Measured};
use crate::inference::generic_transformer::GenericTransformerLayer;
use crate::tensor::{Device, Tensor};
use std::time::Duration;

/// One layer kind of the model: its heads and the window it reads, `None` for a global layer.
struct Kind {
    name: &'static str,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    window: Option<usize>,
}

/// The layer kinds the header `--file` names, else the one global kind of the options.
fn kinds(cx: &Context) -> Result<Vec<Kind>, String> {
    let Some(path) = cx.options.file.as_ref() else {
        let n_head = cx.options.heads.ok_or(NEEDS)?;
        return Ok(vec![Kind {
            name: "global",
            n_head,
            n_kv: cx.options.kv_heads.unwrap_or(n_head),
            head_dim: cx.options.head_dim.ok_or(NEEDS)?,
            window: None,
        }]);
    };
    let content = crate::tensor::quantized::gguf_file::open_header(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let arch = content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .cloned()
        .ok_or_else(|| "the file names no general.architecture".to_string())?;
    let meta_u = |key: &str| {
        content
            .metadata
            .get(&format!("{arch}.{key}"))
            .and_then(|v| v.to_u64().ok())
            .map(|v| v as usize)
    };
    let n_head = meta_u("attention.head_count").ok_or(NEEDS)?;
    let n_kv = meta_u("attention.head_count_kv").unwrap_or(n_head);
    let head_dim = meta_u("attention.key_length").ok_or(NEEDS)?;
    let mut kinds = vec![Kind {
        name: "global",
        n_head,
        n_kv,
        head_dim,
        window: None,
    }];
    if let Some(window) = meta_u("attention.sliding_window").filter(|&w| w > 0) {
        kinds.push(Kind {
            name: "windowed",
            n_head,
            n_kv,
            head_dim: meta_u("attention.key_length_swa").unwrap_or(head_dim),
            window: Some(window),
        });
    }
    Ok(kinds)
}

const NEEDS: &str = "shared-verify needs --file, or --heads and --head-dim";

/// The buffer of the donor's keys or values: `held` rows and room for the widest verify more.
fn donor(kind: &Kind, held: usize, seed: usize) -> Result<Tensor, String> {
    let room = held + crate::inference::engine::decode_step::VERIFY_ROWS;
    values(&[1, kind.n_kv, room, kind.head_dim], seed)
}

/// The first `held` rows of the donor's buffers `k` and `v`, a new view each call, as the cache
/// hands its rows to each forward.
fn held_rows(k: &Tensor, v: &Tensor, held: usize) -> crate::tensor::Result<(Tensor, Tensor)> {
    Ok((k.narrow(2, 0, held)?, v.narrow(2, 0, held)?))
}

/// Deterministic values in `[-0.5, 0.5)`, a tensor of `dims` on the host.
fn values(dims: &[usize], seed: usize) -> Result<Tensor, String> {
    let n: usize = dims.iter().product();
    let v: Vec<f32> = (0..n)
        .map(|i| (((i + seed).wrapping_mul(2_654_435_761) >> 7) % 61) as f32 / 61.0 - 0.5)
        .collect();
    Tensor::from_vec(v, dims, &Device::Cpu).map_err(|e| e.to_string())
}

/// The decode step's attention of one query row `q` `[1, n_head, 1, d]` over `k`, `v`
/// `[1, n_kv, len, d]`: the last `window` keys of a windowed layer, the grouped heads' scores,
/// their softmax and the weighted values, as the step runs them on the host.
fn step(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    kind: &Kind,
    scale: f32,
) -> crate::tensor::Result<Tensor> {
    let len = k.dims()[2];
    let (k, v) = match kind.window {
        Some(w) if len > w => (k.narrow(2, len - w, w)?, v.narrow(2, len - w, w)?),
        _ => (k.clone(), v.clone()),
    };
    let n_rep = kind.n_head / kind.n_kv;
    let att = q
        .reshape((1, kind.n_kv, n_rep, kind.head_dim))?
        .matmul_t(&k)?
        .affine(scale, 0.0)?;
    let att = crate::tensor::ops::softmax_last_dim(&att)?;
    att.matmul(&v)?.reshape((1, kind.n_head, 1, kind.head_dim))
}

/// The rows of `q` `[1, n_head, rows, d]`, each as its decode step over the donor's rows up to
/// its own, the last `rows` of the donor's rows being the verify's: the host's path.
/// The rows of a key-value head share one pass of dots over the keys and one product of the
/// weights by the values, both over the donor's rows in place, each row's weights the softmax of
/// its own scores and zero past them.
fn rows_as_steps(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    kind: &Kind,
    scale: f32,
) -> crate::tensor::Result<Tensor> {
    let rows = q.dims()[2];
    let base = k.dims()[2] - rows;
    let (n_rep, d) = (kind.n_head / kind.n_kv, kind.head_dim);
    let first = |r: usize| match kind.window {
        Some(w) if base + r + 1 > w => base + r + 1 - w,
        _ => 0,
    };
    let lo = first(0);
    let width = base + rows - lo;
    let m = rows * n_rep;
    let queries = q.flatten_all()?.to_vec1::<f32>()?;
    let starts: Vec<usize> = (0..rows).map(|r| first(r) - lo).collect();
    if kind.window.is_some() {
        return windowed_rows(&queries, k, v, kind, &starts, lo, base - lo, scale);
    }
    let heads = (0..kind.n_kv)
        .map(|h| {
            Ok((
                k.narrow(1, h, 1)?.narrow(2, lo, width)?,
                v.narrow(1, h, 1)?.narrow(2, lo, width)?,
            ))
        })
        .collect::<crate::tensor::Result<Vec<(Tensor, Tensor)>>>()?;
    if heads
        .iter()
        .any(|(kh, vh)| kh.host_f32_in_place().is_none() || vh.host_f32_in_place().is_none())
    {
        crate::tensor::bail!("the donor's rows are not F32 on the host");
    }
    use rayon::prelude::*;
    let outputs: Vec<Vec<f32>> = heads
        .par_iter()
        .enumerate()
        .map(|(h, (kh, vh))| {
            let (keys, values) = (
                kh.host_f32_in_place().unwrap_or_default(),
                vh.host_f32_in_place().unwrap_or_default(),
            );
            let mut qh = Vec::with_capacity(m * d);
            for r in 0..rows {
                for e in 0..n_rep {
                    let at = ((h * n_rep + e) * rows + r) * d;
                    qh.extend_from_slice(&queries[at..at + d]);
                }
            }
            let dots = crate::tensor::dots_by_key_f32(&qh, keys, m, d, width);
            let mut weights = vec![0f32; m * width];
            for (r, &from) in starts.iter().enumerate() {
                let to = base + r + 1 - lo;
                for e in 0..n_rep {
                    let i = r * n_rep + e;
                    let own = &mut weights[i * width + from..i * width + to];
                    for (j, w) in own.iter_mut().enumerate() {
                        *w = dots[(from + j) * m + i] * scale + 0.0;
                    }
                    crate::tensor::ops::softmax_row_in_place(own);
                }
            }
            crate::tensor::gemm_f32_rows(&weights, values, m, width, d)
        })
        .collect();
    let mut out = vec![0f32; kind.n_head * rows * d];
    for (h, y) in outputs.iter().enumerate() {
        for r in 0..rows {
            for e in 0..n_rep {
                let at = ((h * n_rep + e) * rows + r) * d;
                out[at..at + d].copy_from_slice(&y[(r * n_rep + e) * d..][..d]);
            }
        }
    }
    Tensor::from_vec(out, (1, kind.n_head, rows, d), &Device::Cpu)
}

/// The verify rows of a windowed layer in 2 regions of the prefill pool, read in place from the
/// first key any row reads, `lo`: every score, an item a key of a key-value head dotted with
/// every query row of the head; then every output, an item a row of the query heads of a key-value
/// head, each the softmax of its own scores and its weighted sum of the values.
#[allow(clippy::too_many_arguments)]
fn windowed_rows(
    queries: &[f32],
    k: &Tensor,
    v: &Tensor,
    kind: &Kind,
    starts: &[usize],
    lo: usize,
    held: usize,
    scale: f32,
) -> crate::tensor::Result<Tensor> {
    let rows = starts.len();
    let (n_rep, d) = (kind.n_head / kind.n_kv, kind.head_dim);
    let (m, width) = (rows * n_rep, held + rows);
    let heads = (0..kind.n_kv)
        .map(|h| {
            Ok((
                k.narrow(1, h, 1)?.narrow(2, lo, width)?,
                v.narrow(1, h, 1)?.narrow(2, lo, width)?,
            ))
        })
        .collect::<crate::tensor::Result<Vec<(Tensor, Tensor)>>>()?;
    let slices = heads
        .iter()
        .map(
            |(kh, vh)| match (kh.host_f32_in_place(), vh.host_f32_in_place()) {
                (Some(keys), Some(values)) => Ok((keys, values)),
                _ => crate::tensor::bail!("the donor's rows are not F32 on the host"),
            },
        )
        .collect::<crate::tensor::Result<Vec<_>>>()?;
    let lanes = (0..kind.n_kv)
        .map(|h| {
            let mut rows_of_head = Vec::with_capacity(m * d);
            for r in 0..rows {
                for e in 0..n_rep {
                    let at = ((h * n_rep + e) * rows + r) * d;
                    rows_of_head.extend_from_slice(&queries[at..at + d]);
                }
            }
            crate::tensor::dot_lanes_f32(&rows_of_head, m, d)
        })
        .collect::<Vec<_>>();
    let mut dots = vec![0f32; kind.n_kv * width * m];
    crate::tensor::quant_cpu::pool_par_chunks_mut_prefill(&mut dots, m, &|item, out| {
        let (h, j) = (item / width, item % width);
        crate::tensor::dots_of_key_f32(&lanes[h], m, d, &slices[h].0[j * d..][..d], out);
    });
    let mut items = vec![0f32; kind.n_kv * rows * n_rep * d];
    crate::tensor::quant_cpu::pool_par_chunks_mut_prefill(&mut items, n_rep * d, &|item, acc| {
        let (h, r) = (item / rows, item % rows);
        let (from, to) = (starts[r], held + r + 1);
        let span = to - from;
        let mut weights = vec![0f32; n_rep * span];
        for (e, own) in weights.chunks_mut(span).enumerate() {
            let i = r * n_rep + e;
            for (j, w) in own.iter_mut().enumerate() {
                *w = dots[(h * width + from + j) * m + i] * scale + 0.0;
            }
            crate::tensor::ops::softmax_row_in_place(own);
        }
        let values = slices[h].1;
        for j in 0..span {
            let row = &values[(from + j) * d..][..d];
            for (e, out) in acc.chunks_mut(d).enumerate() {
                let w = weights[e * span + j];
                for (o, &x) in out.iter_mut().zip(row) {
                    *o += w * x;
                }
            }
        }
    });
    let mut out = vec![0f32; kind.n_head * rows * d];
    for h in 0..kind.n_kv {
        for r in 0..rows {
            for e in 0..n_rep {
                let at = ((h * n_rep + e) * rows + r) * d;
                out[at..at + d].copy_from_slice(&items[((h * rows + r) * n_rep + e) * d..][..d]);
            }
        }
    }
    Tensor::from_vec(out, (1, kind.n_head, rows, d), &Device::Cpu)
}

/// The largest absolute difference between 2 outputs of one shape.
fn largest_gap(a: &Tensor, b: &Tensor) -> Result<f32, String> {
    let flat = |t: &Tensor| {
        t.flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| e.to_string())
    };
    let (a, b) = (flat(a)?, flat(b)?);
    if a.len() != b.len() {
        return Err(format!("outputs of {} and {} values", a.len(), b.len()));
    }
    Ok(a.iter()
        .zip(&b)
        .fold(0f32, |m, (x, y)| m.max((x - y).abs())))
}

/// The mask of `rows` verify rows over `held` donor rows that end with them: row `r` keeps the
/// keys up to its own and, for a windowed layer, the last `window` of them. 1 hides a key.
fn mask(rows: usize, held: usize, window: Option<usize>) -> Result<Tensor, String> {
    let base = held - rows;
    let data: Vec<u8> = (0..rows)
        .flat_map(|r| {
            let last = base + r;
            let first = window.map_or(0, |w| (last + 1).saturating_sub(w));
            (0..held).map(move |t| u8::from(t < first || t > last))
        })
        .collect();
    Tensor::from_vec(data, (rows, held), &Device::Cpu).map_err(|e| e.to_string())
}

pub fn run(cx: &Context) -> Result<Vec<Measured>, String> {
    let target = Duration::from_secs_f64(cx.options.seconds);
    let context = cx.options.context;
    let mut out = Vec::new();
    for kind in kinds(cx)? {
        if kind.n_kv == 0 || !kind.n_head.is_multiple_of(kind.n_kv) {
            return Err(format!(
                "{} heads over {} key-value heads do not group",
                kind.n_head, kind.n_kv
            ));
        }
        let scale = 1.0 / (kind.head_dim as f32).sqrt();
        let n_rep = kind.n_head / kind.n_kv;
        let what = format!(
            "{} layer, {} heads over {} key-value heads of {}{}",
            kind.name,
            kind.n_head,
            kind.n_kv,
            kind.head_dim,
            kind.window
                .map_or(String::new(), |w| format!(", window {w}"))
        );
        // The decode step: 1 row over the donor's rows.
        let held = kind.window.map_or(context + 1, |w| w + 1);
        let k = donor(&kind, held, 1)?;
        let v = donor(&kind, held, 2)?;
        let q = values(&[1, kind.n_head, 1, kind.head_dim], 3)?;
        let stepped = || {
            let (k, v) = held_rows(&k, &v, held)?;
            step(&q, &k, &v, &kind, scale)
        };
        stepped().map_err(|e| e.to_string())?;
        out.push(Measured {
            variant: format!("{}: decode step", kind.name),
            bytes: None,
            samples_us: run_for(target, || {
                let t = std::time::Instant::now();
                let _ = stepped();
                t.elapsed().as_secs_f64() * 1e6
            }),
            note: Some(format!("{what}, {held} donor rows")),
        });
        for &rows in cx.options.block.iter().filter(|&&r| r > 1) {
            let held = kind.window.map_or(context + rows, |w| w + rows);
            let k = donor(&kind, held, 1)?;
            let v = donor(&kind, held, 2)?;
            let q = values(&[1, kind.n_head, rows, kind.head_dim], 3)?;
            let m = mask(rows, held, kind.window)?;
            let masked = || {
                let (k, v) = held_rows(&k, &v, held)?;
                GenericTransformerLayer::flash_attn_prefill_cpu(&q, &k, &v, &m, scale, n_rep, None)
            };
            let stepped = || {
                let (k, v) = held_rows(&k, &v, held)?;
                rows_as_steps(&q, &k, &v, &kind, scale)
            };
            let gap = largest_gap(
                &masked().map_err(|e| e.to_string())?,
                &stepped().map_err(|e| e.to_string())?,
            )?;
            let old = Measured {
                variant: format!("{}: {rows} rows, one masked attention", kind.name),
                bytes: None,
                samples_us: run_for(target, || {
                    let t = std::time::Instant::now();
                    let _ = masked();
                    t.elapsed().as_secs_f64() * 1e6
                }),
                note: Some(format!("{what}, {held} donor rows")),
            };
            let mut new = Measured {
                variant: format!("{}: {rows} rows, one decode step each", kind.name),
                bytes: None,
                samples_us: run_for(target, || {
                    let t = std::time::Instant::now();
                    let _ = stepped();
                    t.elapsed().as_secs_f64() * 1e6
                }),
                note: None,
            };
            new.note = Some(format!(
                "{}; the outputs differ by {gap:.2e} at most",
                ratio_note(&old, &new).unwrap_or_default()
            ));
            out.push(old);
            out.push(new);
        }
    }
    Ok(out)
}
