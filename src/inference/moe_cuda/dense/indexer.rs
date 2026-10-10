//! The block-sparse indexer and its attention on a card: the launches of `cuda/moe/indexer.cu`.
//!
//! The indexer's cache, its selection of blocks and the attention over the kept tokens read the
//! position of the step from a device scalar, so one recorded step serves every position.

use super::*;

/// Device address of a tensor's first element.
fn address<T: crate::tensor::kernel_ffi::CudaDType>(t: &Tensor, what: &str) -> Result<u64> {
    let (st, layout) = t.storage_and_layout();
    match &*st {
        StorageView::Cuda(c) => {
            let slice = c.as_cuda_slice::<T>()?;
            let (base, _) = slice.device_ptr(slice.stream());
            Ok(base + (layout.start_offset() * std::mem::size_of::<T>()) as u64)
        }
        _ => crate::tensor::bail!("{what} must be on a CUDA device"),
    }
}

fn stream_of_card(t: &Tensor) -> Result<i64> {
    Ok(t.device().as_cuda_device()?.cuda_stream().cu_stream() as i64)
}

/// The dimensions of one indexer: its head width and block, the lanes its rotation covers
/// (`half` pairs) and the budget of blocks a query keeps.
#[derive(Clone, Copy, Debug)]
pub struct IndexerShape {
    pub heads: usize,
    pub hd: usize,
    pub block: usize,
    pub budget: usize,
    pub half: usize,
    pub eps: f32,
}

/// Where a launch reads the position of the step: from the card, which a recorded step needs, or
/// the host's own count.
#[derive(Clone, Copy)]
pub enum StepPos<'a> {
    Card(&'a Tensor),
    Host(usize),
}

impl StepPos<'_> {
    /// The pointer a kernel reads, null for the host's count, and the count it falls back on.
    fn parts(&self) -> Result<(*const i32, i32)> {
        match self {
            StepPos::Card(t) => Ok((address::<i32>(t, "pos")? as *const i32, 0)),
            StepPos::Host(p) => Ok((std::ptr::null(), *p as i32)),
        }
    }
}

/// The device address of an F32 tensor's first element.
pub fn address_f32(t: &Tensor) -> Result<u64> {
    address::<f32>(t, "tensor")
}

/// The raw key row `key` of the token at `pos` joins the cache: it is stored in `tail`, the
/// rows of the block being filled, and the block it completes, if any, is pooled into `pooled`.
/// `rot` holds the cosine then the sine of each rotation pair at the first position of that
/// block.
pub fn indexer_append(
    key: &Tensor,
    tail: &Tensor,
    pooled: &Tensor,
    gamma: &Tensor,
    rot: &Tensor,
    pos: StepPos,
    shape: &IndexerShape,
) -> Result<()> {
    let stream = stream_of_card(key)?;
    let (pos_dev, pos_host) = pos.parts()?;
    unsafe {
        loken_indexer_append(
            address::<f32>(key, "key")? as *const f32,
            address::<f32>(tail, "tail")? as *mut f32,
            address::<f32>(pooled, "pooled")? as *mut f32,
            address::<f32>(gamma, "gamma")? as *const f32,
            address::<f32>(rot, "rot")? as *const f32,
            pos_dev,
            pos_host,
            shape.hd as i32,
            shape.block as i32,
            shape.half as i32,
            shape.eps,
            stream,
        );
    }
    Ok(())
}

/// The tokens the query `q` (the raw heads, `heads * hd` F32) keeps at `pos`,
/// ascending, into `sel` with their number in `count`. `rot` holds the rotation tables of the
/// query's position; `scores` is scratch for one score per block.
#[allow(clippy::too_many_arguments)]
pub fn indexer_select(
    q: &Tensor,
    gamma: &Tensor,
    rot: &Tensor,
    pooled: &Tensor,
    scores: &Tensor,
    sel: &Tensor,
    count: &Tensor,
    pos: StepPos,
    shape: &IndexerShape,
) -> Result<()> {
    let stream = stream_of_card(q)?;
    let (pos_dev, pos_host) = pos.parts()?;
    let scale = 1.0 / (shape.hd as f32).sqrt();
    unsafe {
        loken_indexer_select(
            address::<f32>(q, "q")? as *const f32,
            address::<f32>(gamma, "gamma")? as *const f32,
            address::<f32>(rot, "rot")? as *const f32,
            address::<f32>(pooled, "pooled")? as *const f32,
            address::<f32>(scores, "scores")? as *mut f32,
            address::<u32>(sel, "sel")? as *mut u32,
            address::<u32>(count, "count")? as *mut u32,
            pos_dev,
            pos_host,
            shape.heads as i32,
            shape.hd as i32,
            shape.block as i32,
            shape.budget as i32,
            shape.half as i32,
            shape.eps,
            scale,
            scores.elem_count() as i32,
            1,
            0,
            0,
            stream,
        );
    }
    Ok(())
}

/// [`indexer_select`] for the rows of a prompt chunk, the first at position `first`: `q` is
/// `[rows, heads * hd]` raw, `rot` `[rows, 2 * half]` the tables of each row's position, `scores`
/// room for `rows` times one score per block, `sel` `[rows, room]` and `count` `[rows]`.
#[allow(clippy::too_many_arguments)]
pub fn indexer_select_rows(
    q: &Tensor,
    gamma: &Tensor,
    rot: &Tensor,
    pooled: &Tensor,
    scores: &Tensor,
    sel: &Tensor,
    count: &Tensor,
    first: usize,
    shape: &IndexerShape,
) -> Result<()> {
    let stream = stream_of_card(q)?;
    let rows = q.dim(0)?;
    let scale = 1.0 / (shape.hd as f32).sqrt();
    unsafe {
        loken_indexer_select(
            address::<f32>(q, "q")? as *const f32,
            address::<f32>(gamma, "gamma")? as *const f32,
            address::<f32>(rot, "rot")? as *const f32,
            address::<f32>(pooled, "pooled")? as *const f32,
            address::<f32>(scores, "scores")? as *mut f32,
            address::<u32>(sel, "sel")? as *mut u32,
            address::<u32>(count, "count")? as *mut u32,
            std::ptr::null(),
            first as i32,
            shape.heads as i32,
            shape.hd as i32,
            shape.block as i32,
            shape.budget as i32,
            shape.half as i32,
            shape.eps,
            scale,
            (scores.elem_count() / rows.max(1)) as i32,
            rows as i32,
            (2 * shape.half) as i32,
            sel.dim(1)? as i32,
            stream,
        );
    }
    Ok(())
}

/// Rows `[n, hd]` F32 into the pooled keys' groups, from block `first`.
pub fn indexer_rows_in(
    rows: &Tensor,
    pooled: &Tensor,
    first: usize,
    n: usize,
    hd: usize,
) -> Result<()> {
    let stream = stream_of_card(rows)?;
    unsafe {
        loken_indexer_rows_in(
            address::<f32>(rows, "rows")? as *const f32,
            address::<f32>(pooled, "pooled")? as *mut f32,
            first as i32,
            n as i32,
            hd as i32,
            stream,
        );
    }
    Ok(())
}

/// The pooled keys of blocks `first..first + n` as rows `[n, hd]` F32.
pub fn indexer_rows_out(pooled: &Tensor, first: usize, n: usize, hd: usize) -> Result<Tensor> {
    let dev = pooled.device().as_cuda_device()?;
    let stream = stream_of_card(pooled)?;
    let out = unsafe { dev.alloc::<f32>(n * hd) }?;
    unsafe {
        loken_indexer_rows_out(
            address::<f32>(pooled, "pooled")? as *const f32,
            out.device_ptr(out.stream()).0 as *mut f32,
            first as i32,
            n as i32,
            hd as i32,
            stream,
        );
    }
    tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(out, dev.clone()),
        crate::tensor::Shape::from((n, hd)),
    )
}

/// The key and value rows of a decoded token (`[n_kv, hd]` F32 each) into the F16 cache buffers
/// `[n_kv, cap, hd]` at `pos`.
pub fn kv_append_pos(
    k_new: &Tensor,
    v_new: &Tensor,
    kbuf: &Tensor,
    vbuf: &Tensor,
    pos: StepPos,
    n_kv: usize,
    cap: usize,
    hd: usize,
) -> Result<()> {
    let stream = stream_of_card(k_new)?;
    let (pos_dev, pos_host) = pos.parts()?;
    unsafe {
        loken_kv_append_pos(
            address::<f32>(k_new, "k_new")? as *const f32,
            address::<f32>(v_new, "v_new")? as *const f32,
            address::<half::f16>(kbuf, "kbuf")? as *mut c_void,
            address::<half::f16>(vbuf, "vbuf")? as *mut c_void,
            pos_dev,
            pos_host,
            n_kv as i32,
            cap as i32,
            hd as i32,
            stream,
        );
    }
    Ok(())
}

/// Attention of one decoded token over the kept tokens `sel[..count]` of the F16 cache buffers
/// `[n_kv, cap, hd]`: `q` is `[n_head, hd]` F32, already rotated; the answer is
/// `[1, 1, n_head * hd]` F32. `ncap` is the room of `sel`.
#[allow(clippy::too_many_arguments)]
pub fn selected_attention(
    q: &Tensor,
    kbuf: &Tensor,
    vbuf: &Tensor,
    sel: &Tensor,
    count: &Tensor,
    n_head: usize,
    n_kv: usize,
    cap: usize,
    hd: usize,
    scale: f32,
) -> Result<Tensor> {
    let dev = q.device().as_cuda_device()?;
    let stream = stream_of_card(q)?;
    let ncap = sel.elem_count();
    let splits = attention_splits(ncap);
    let out = unsafe { dev.alloc::<f32>(n_head * hd) }?;
    let part_o = unsafe { dev.alloc::<f32>(n_head * splits * hd) }?;
    let part_ml = unsafe { dev.alloc::<f32>(n_head * splits * 2) }?;
    unsafe {
        loken_selected_attn(
            address::<f32>(q, "q")? as *const f32,
            address::<half::f16>(kbuf, "kbuf")? as *const c_void,
            address::<half::f16>(vbuf, "vbuf")? as *const c_void,
            address::<u32>(sel, "sel")? as *const u32,
            address::<u32>(count, "count")? as *const u32,
            out.device_ptr(out.stream()).0 as *mut f32,
            part_o.device_ptr(part_o.stream()).0 as *mut f32,
            part_ml.device_ptr(part_ml.stream()).0 as *mut f32,
            n_head as i32,
            n_kv as i32,
            cap as i32,
            hd as i32,
            ncap as i32,
            splits as i32,
            scale,
            stream,
        );
    }
    tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(out, dev.clone()),
        crate::tensor::Shape::from((1, 1, n_head * hd)),
    )
}

/// Whether [`selected_attention_rows`] serves heads of width `hd`, `n_head` query heads over
/// `n_kv` key-value heads.
pub fn selected_attention_rows_serves(hd: usize, n_head: usize, n_kv: usize) -> bool {
    hd == 256 && n_kv > 0 && n_head % n_kv == 0 && n_head / n_kv <= 16
}

/// Attention of the rows of a prompt chunk, each over its own kept tokens, in one launch: `q` is
/// `[rows, n_head * hd]` F32, already rotated; the F16 cache buffers are `[n_kv, cap, hd]`; row `r`
/// keeps `sel[r]` `[rows, ncap]` up to `count[r]`; the answer is `[rows, n_head * hd]` F32.
#[allow(clippy::too_many_arguments)]
pub fn selected_attention_rows(
    q: &Tensor,
    kbuf: &Tensor,
    vbuf: &Tensor,
    sel: &Tensor,
    count: &Tensor,
    n_head: usize,
    n_kv: usize,
    cap: usize,
    hd: usize,
    scale: f32,
) -> Result<Tensor> {
    if !selected_attention_rows_serves(hd, n_head, n_kv) {
        crate::tensor::bail!("selected attention over rows: {n_head} heads of {hd} over {n_kv}");
    }
    let dev = q.device().as_cuda_device()?;
    let stream = stream_of_card(q)?;
    let rows = q.dim(0)?;
    let ncap = sel.dim(1)?;
    let out = unsafe { dev.alloc::<f32>(rows * n_head * hd) }?;
    unsafe {
        loken_selected_attn_rows(
            address::<f32>(q, "q")? as *const f32,
            address::<half::f16>(kbuf, "kbuf")? as *const c_void,
            address::<half::f16>(vbuf, "vbuf")? as *const c_void,
            address::<u32>(sel, "sel")? as *const u32,
            address::<u32>(count, "count")? as *const u32,
            out.device_ptr(out.stream()).0 as *mut f32,
            rows as i32,
            n_head as i32,
            n_kv as i32,
            cap as i32,
            ncap as i32,
            scale,
            stream,
        );
    }
    tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(out, dev.clone()),
        crate::tensor::Shape::from((rows, n_head * hd)),
    )
}

/// Tokens one block of the attention scores and weighs.
const ATTENTION_RUN_TOKENS: usize = 128;

/// The runs the kept tokens are cut into, for a room of `ncap` of them: a block takes a run of
/// [`ATTENTION_RUN_TOKENS`], and no more than thirty-two runs are joined.
fn attention_splits(ncap: usize) -> usize {
    ncap.div_ceil(ATTENTION_RUN_TOKENS).clamp(1, 32)
}

/// The shared memory a block of the attention kernel needs, in bytes.
pub fn selected_attention_smem(ncap: usize, hd: usize) -> usize {
    let run = ncap.div_ceil(attention_splits(ncap));
    (run + hd + 8 * hd) * std::mem::size_of::<f32>()
}

/// The shared memory the selection kernel needs, in bytes.
pub fn indexer_select_smem(heads: usize, hd: usize) -> usize {
    heads * hd * std::mem::size_of::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Device;

    fn pseudo(i: usize, seed: usize) -> f32 {
        (((i * 2654435761usize + seed * 40503) >> 5) % 2001) as f32 / 1000.0 - 1.0
    }

    /// The attention of one token over a kept subset of the cache, in double precision.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        q: &[f32],
        k: &[half::f16],
        v: &[half::f16],
        sel: &[u32],
        n_head: usize,
        n_kv: usize,
        cap: usize,
        hd: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut out = vec![0f32; n_head * hd];
        for h in 0..n_head {
            let kvh = h / (n_head / n_kv);
            let row = |buf: &[half::f16], j: u32| {
                let at = (kvh * cap + j as usize) * hd;
                buf[at..at + hd]
                    .iter()
                    .map(|x| x.to_f32() as f64)
                    .collect::<Vec<f64>>()
            };
            let scores: Vec<f64> = sel
                .iter()
                .map(|&j| {
                    let kr = row(k, j);
                    (0..hd).map(|d| q[h * hd + d] as f64 * kr[d]).sum::<f64>() * scale as f64
                })
                .collect();
            let max = scores.iter().cloned().fold(f64::MIN, f64::max);
            let exp: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f64 = exp.iter().sum();
            for (e, &j) in exp.iter().zip(sel) {
                let vr = row(v, j);
                for d in 0..hd {
                    out[h * hd + d] += (e / total * vr[d]) as f32;
                }
            }
        }
        out
    }

    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn the_selected_attention_matches_a_host_reference() {
        let dev = Device::new_cuda(0).expect("a card for the card test");
        for (n_head, n_kv, hd, n, cap) in [
            (24, 2, 256, 2051, 4096),
            (8, 2, 128, 100, 256),
            (4, 4, 64, 1, 16),
        ] {
            let q: Vec<f32> = (0..n_head * hd).map(|i| pseudo(i, 1)).collect();
            let kv = |seed: usize| -> Vec<half::f16> {
                (0..n_kv * cap * hd)
                    .map(|i| half::f16::from_f32(pseudo(i, seed)))
                    .collect()
            };
            let (k, v) = (kv(2), kv(3));
            // every third token of the cache, ascending
            let sel: Vec<u32> = (0..n as u32).map(|j| (j * 3) % cap as u32).collect();
            let mut sorted = sel.clone();
            sorted.sort_unstable();
            sorted.dedup();
            let n = sorted.len();
            let mut room = sorted.clone();
            room.resize(n + 5, 0);
            let on = |t: Tensor| t.to_device(&dev).unwrap();
            let tq = on(Tensor::from_vec(q.clone(), (n_head, hd), &Device::Cpu).unwrap());
            let tk = on(Tensor::from_vec(k.clone(), (n_kv, cap, hd), &Device::Cpu).unwrap());
            let tv = on(Tensor::from_vec(v.clone(), (n_kv, cap, hd), &Device::Cpu).unwrap());
            let tsel = on(Tensor::from_vec(room, (n + 5,), &Device::Cpu).unwrap());
            let tcount = on(Tensor::from_vec(vec![n as u32], (1,), &Device::Cpu).unwrap());
            let scale = 1.0 / (hd as f32).sqrt();
            let got =
                selected_attention(&tq, &tk, &tv, &tsel, &tcount, n_head, n_kv, cap, hd, scale)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
            let want = reference(&q, &k, &v, &sorted, n_head, n_kv, cap, hd, scale);
            let diff = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(diff < 2e-4, "{n_head} heads, {n} tokens: differs by {diff}");
        }
    }

    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn the_selected_attention_of_rows_matches_a_host_reference_row_by_row() {
        let dev = Device::new_cuda(0).expect("a card for the card test");
        let (n_head, n_kv, hd, cap, rows) = (24usize, 2usize, 256usize, 700usize, 37usize);
        let q: Vec<f32> = (0..rows * n_head * hd).map(|i| pseudo(i, 1)).collect();
        let kv = |seed: usize| -> Vec<half::f16> {
            (0..n_kv * cap * hd)
                .map(|i| half::f16::from_f32(pseudo(i, seed)))
                .collect()
        };
        let (k, v) = (kv(2), kv(3));
        // Row r keeps a different, unordered run of the cache: from one token to several hundred.
        let kept: Vec<Vec<u32>> = (0..rows)
            .map(|r| {
                let n = 1 + (r * r * 7 + 13 * r) % 300;
                (0..n as u32)
                    .map(|j| (j * 5 + r as u32 * 3) % cap as u32)
                    .collect()
            })
            .collect();
        let ncap = kept.iter().map(Vec::len).max().unwrap();
        let mut sel = vec![0u32; rows * ncap];
        for (r, row) in kept.iter().enumerate() {
            sel[r * ncap..r * ncap + row.len()].copy_from_slice(row);
        }
        let count: Vec<u32> = kept.iter().map(|k| k.len() as u32).collect();
        let on = |t: Tensor| t.to_device(&dev).unwrap();
        let tq = on(Tensor::from_vec(q.clone(), (rows, n_head * hd), &Device::Cpu).unwrap());
        let tk = on(Tensor::from_vec(k.clone(), (n_kv, cap, hd), &Device::Cpu).unwrap());
        let tv = on(Tensor::from_vec(v.clone(), (n_kv, cap, hd), &Device::Cpu).unwrap());
        let tsel = on(Tensor::from_vec(sel, (rows, ncap), &Device::Cpu).unwrap());
        let tcount = on(Tensor::from_vec(count, (rows,), &Device::Cpu).unwrap());
        let scale = 1.0 / (hd as f32).sqrt();
        let got =
            selected_attention_rows(&tq, &tk, &tv, &tsel, &tcount, n_head, n_kv, cap, hd, scale)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
        for (r, row) in kept.iter().enumerate() {
            let want = reference(
                &q[r * n_head * hd..(r + 1) * n_head * hd],
                &k,
                &v,
                row,
                n_head,
                n_kv,
                cap,
                hd,
                scale,
            );
            let diff = got[r * n_head * hd..(r + 1) * n_head * hd]
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                diff < 2e-4,
                "row {r} of {} kept tokens differs by {diff}",
                row.len()
            );
        }
    }

    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn a_token_is_written_to_the_position_the_card_holds() {
        let dev = Device::new_cuda(0).expect("a card for the card test");
        let (n_kv, cap, hd) = (2, 8, 16);
        let rows = |seed: usize| -> Tensor {
            let v: Vec<f32> = (0..n_kv * hd).map(|i| pseudo(i, seed)).collect();
            Tensor::from_vec(v, (n_kv, hd), &dev).unwrap()
        };
        let buf = || Tensor::zeros_on((n_kv, cap, hd), crate::tensor::DType::F16, &dev).unwrap();
        let (kb, vb) = (buf(), buf());
        for pos in [0usize, 3, 7] {
            let at = Tensor::new(&[pos as i32], &dev).unwrap();
            let at = match pos {
                7 => StepPos::Host(pos),
                _ => StepPos::Card(&at),
            };
            kv_append_pos(&rows(1), &rows(2), &kb, &vb, at, n_kv, cap, hd).unwrap();
        }
        let host = |t: &Tensor| {
            t.to_dtype(crate::tensor::DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let (k, v) = (host(&kb), host(&vb));
        let (k_src, v_src) = (host(&rows(1)), host(&rows(2)));
        for h in 0..n_kv {
            for pos in 0..cap {
                let at = (h * cap + pos) * hd;
                let written = [0usize, 3, 7].contains(&pos);
                for d in 0..hd {
                    let (want_k, want_v) = match written {
                        true => (k_src[h * hd + d], v_src[h * hd + d]),
                        false => (0.0, 0.0),
                    };
                    assert!((k[at + d] - want_k).abs() < 1e-3, "k at {h} {pos} {d}");
                    assert!((v[at + d] - want_v).abs() < 1e-3, "v at {h} {pos} {d}");
                }
            }
        }
    }
}
