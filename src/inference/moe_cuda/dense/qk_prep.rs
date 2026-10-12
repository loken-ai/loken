//! The input stage of an attention layer's decoded token as one launch: the norm of each query
//! and key head, the rotation of its leading lanes and the cast of the three rows to F32.

use super::hyper::addr;
use super::*;

/// Shape of the attention the launch serves.
#[derive(Clone, Copy)]
pub struct QkShape {
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    /// Leading lanes of a head that are rotated; `0` for none.
    pub rope_dim: usize,
    /// Pairs lane `2 i` with `2 i + 1` instead of lane `i` with `i + rope_dim / 2`.
    pub interleaved: bool,
    pub eps: f32,
}

/// `q` `[n_head * head_dim]`, `k` and `v` `[n_kv * head_dim]`, all F16 or all F32, normed by the
/// weights `qw` and `kw` (`[head_dim]` F32; a head with none is not normed), rotated by the
/// token's `cos` and `sin` (`[rope_dim / 2]` F32 each) and returned as F32 `[1, heads, 1,
/// head_dim]`. `Ok(None)` when the inputs are not of the types or shapes the launch takes.
#[allow(clippy::too_many_arguments)]
pub fn qk_prep(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    qw: Option<&Tensor>,
    kw: Option<&Tensor>,
    cos: &Tensor,
    sin: &Tensor,
    shape: QkShape,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    let hd = shape.head_dim;
    let Some((q, k, v)) = qk_prep_rows(q, k, v, qw, kw, cos, sin, shape, 1)? else {
        return Ok(None);
    };
    Ok(Some((
        q.reshape((1, shape.n_head, 1, hd))?,
        k.reshape((1, shape.n_kv, 1, hd))?,
        v.reshape((1, shape.n_kv, 1, hd))?,
    )))
}

/// [`qk_prep`] over `rows` rows in one launch: `q`, `k`, `v` hold `rows` rows each, `cos` and
/// `sin` one row of tables a row; `[rows, heads * head_dim]` F32 back for each, every row as
/// the one-row launch gives it.
#[allow(clippy::too_many_arguments)]
pub fn qk_prep_rows(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    qw: Option<&Tensor>,
    kw: Option<&Tensor>,
    cos: &Tensor,
    sin: &Tensor,
    shape: QkShape,
    rows: usize,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    use core::ffi::c_void;
    let QkShape {
        n_head,
        n_kv,
        head_dim: hd,
        rope_dim,
        interleaved,
        eps,
    } = shape;
    let f16 = match q.dtype() {
        DType::F16 => true,
        DType::F32 => false,
        _ => return Ok(None),
    };
    let floats = |t: &Tensor, n: usize| {
        t.dtype() == DType::F32 && t.is_contiguous() && t.elem_count() == n && t.device().is_cuda()
    };
    let half = rope_dim / 2;
    if k.dtype() != q.dtype()
        || v.dtype() != q.dtype()
        || !q.device().is_cuda()
        || !(q.is_contiguous() && k.is_contiguous() && v.is_contiguous())
        || rows == 0
        || q.elem_count() != rows * n_head * hd
        || k.elem_count() != rows * n_kv * hd
        || v.elem_count() != rows * n_kv * hd
        || hd == 0
        || n_head == 0
        || n_kv == 0
        || rope_dim > hd
        || rope_dim % 2 != 0
        || qw.is_some_and(|w| !floats(w, hd))
        || kw.is_some_and(|w| !floats(w, hd))
        || (half > 0 && !(floats(cos, rows * half) && floats(sin, rows * half)))
    {
        return Ok(None);
    }
    let dev = q.device().as_cuda_device()?;
    let qo = unsafe { dev.alloc::<f32>(rows * n_head * hd) }?;
    let ko = unsafe { dev.alloc::<f32>(rows * n_kv * hd) }?;
    let vo = unsafe { dev.alloc::<f32>(rows * n_kv * hd) }?;
    let (qp, kp, vp) = if f16 {
        (
            addr::<half::f16>(q, "q")?,
            addr::<half::f16>(k, "k")?,
            addr::<half::f16>(v, "v")?,
        )
    } else {
        (
            addr::<f32>(q, "q")?,
            addr::<f32>(k, "k")?,
            addr::<f32>(v, "v")?,
        )
    };
    let weight = |w: Option<&Tensor>| -> Result<*const f32> {
        match w {
            Some(w) => Ok(addr::<f32>(w, "norm")? as *const f32),
            None => Ok(std::ptr::null()),
        }
    };
    let table = |t: &Tensor| -> Result<*const f32> {
        if half == 0 {
            Ok(std::ptr::null())
        } else {
            Ok(addr::<f32>(t, "rope")? as *const f32)
        }
    };
    let stream = dev.cuda_stream().cu_stream() as i64;
    if rows > 1 {
        unsafe {
            loken_qk_prep_rows(
                qp as *const c_void,
                kp as *const c_void,
                vp as *const c_void,
                weight(qw)?,
                weight(kw)?,
                table(cos)?,
                table(sin)?,
                qo.device_ptr(qo.stream()).0 as *mut f32,
                ko.device_ptr(ko.stream()).0 as *mut f32,
                vo.device_ptr(vo.stream()).0 as *mut f32,
                rows as i32,
                n_head as i32,
                n_kv as i32,
                hd as i32,
                rope_dim as i32,
                i32::from(interleaved),
                eps,
                i32::from(f16),
                stream,
            );
        }
    } else {
        unsafe {
            loken_qk_prep(
                qp as *const c_void,
                kp as *const c_void,
                vp as *const c_void,
                weight(qw)?,
                weight(kw)?,
                table(cos)?,
                table(sin)?,
                qo.device_ptr(qo.stream()).0 as *mut f32,
                ko.device_ptr(ko.stream()).0 as *mut f32,
                vo.device_ptr(vo.stream()).0 as *mut f32,
                n_head as i32,
                n_kv as i32,
                hd as i32,
                rope_dim as i32,
                i32::from(interleaved),
                eps,
                i32::from(f16),
                stream,
            );
        }
    }
    let wrap = |s, heads: usize| {
        tensor_from_cuda_storage(
            CudaStorage::wrap_cuda_slice(s, dev.clone()),
            (rows, heads * hd),
        )
    };
    Ok(Some((wrap(qo, n_head)?, wrap(ko, n_kv)?, wrap(vo, n_kv)?)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Device;

    fn ramp(n: usize, salt: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 2654435761usize + salt * 97) >> 5) % 2001) as f32 / 1000.0 - 1.0)
            .collect()
    }

    /// The launch gives what the separate stages give, for both layouts of the rotation, with
    /// and without a norm, from F16 and F32 rows.
    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn the_input_stage_matches_its_separate_launches() {
        let dev = Device::new_cuda(0).expect("a card for the card test");
        let (n_head, n_kv, hd, rope_dim) = (6usize, 2usize, 64usize, 32usize);
        let on = |v: Vec<f32>, dims: &[usize]| {
            Tensor::from_vec(v, dims.to_vec(), &Device::Cpu)
                .unwrap()
                .to_device(&dev)
                .unwrap()
        };
        let cos = on(ramp(rope_dim / 2, 5), &[1, rope_dim / 2]);
        let sin = on(ramp(rope_dim / 2, 6), &[1, rope_dim / 2]);
        let qw = on(ramp(hd, 7), &[hd]);
        let kw = on(ramp(hd, 8), &[hd]);
        for interleaved in [false, true] {
            for f16 in [false, true] {
                for normed in [false, true] {
                    let mut q = on(ramp(n_head * hd, 1), &[1, n_head * hd]);
                    let mut k = on(ramp(n_kv * hd, 2), &[1, n_kv * hd]);
                    let mut v = on(ramp(n_kv * hd, 3), &[1, n_kv * hd]);
                    if f16 {
                        q = q.to_dtype(DType::F16).unwrap();
                        k = k.to_dtype(DType::F16).unwrap();
                        v = v.to_dtype(DType::F16).unwrap();
                    }
                    let shape = QkShape {
                        n_head,
                        n_kv,
                        head_dim: hd,
                        rope_dim,
                        interleaved,
                        eps: 1e-6,
                    };
                    let (gq, gk, gv) = qk_prep(
                        &q,
                        &k,
                        &v,
                        normed.then_some(&qw),
                        normed.then_some(&kw),
                        &cos,
                        &sin,
                        shape,
                    )
                    .unwrap()
                    .expect("the launch takes this shape");
                    // The reference: norm per head, rotate the leading lanes, in F32.
                    let reference = |t: &Tensor, heads: usize, w: &Tensor| {
                        let x = t
                            .to_dtype(DType::F32)
                            .unwrap()
                            .reshape((1, heads, 1, hd))
                            .unwrap();
                        let x = if normed {
                            let ms = x
                                .sqr()
                                .unwrap()
                                .mean_keepdim(crate::tensor::D::Minus1)
                                .unwrap();
                            x.broadcast_mul(&(ms + 1e-6).unwrap().sqrt().unwrap().recip().unwrap())
                                .unwrap()
                                .broadcast_mul(w)
                                .unwrap()
                        } else {
                            x
                        };
                        let rot = x
                            .narrow(crate::tensor::D::Minus1, 0, rope_dim)
                            .unwrap()
                            .contiguous()
                            .unwrap();
                        let pass = x
                            .narrow(crate::tensor::D::Minus1, rope_dim, hd - rope_dim)
                            .unwrap();
                        let rope = if interleaved {
                            crate::tensor::ops::rope_i
                        } else {
                            crate::tensor::ops::rope
                        };
                        let rot = rope(&rot, &cos, &sin).unwrap();
                        Tensor::cat(&[&rot, &pass], crate::tensor::D::Minus1).unwrap()
                    };
                    let flat = |t: &Tensor| -> Vec<f32> {
                        t.to_device(&Device::Cpu)
                            .unwrap()
                            .flatten_all()
                            .unwrap()
                            .to_vec1()
                            .unwrap()
                    };
                    let worst = |a: &[f32], b: &[f32]| {
                        a.iter()
                            .zip(b)
                            .map(|(x, y)| (x - y).abs())
                            .fold(0.0, f32::max)
                    };
                    let (wq, wk) = (reference(&q, n_head, &qw), reference(&k, n_kv, &kw));
                    assert!(
                        worst(&flat(&gq), &flat(&wq)) < 1e-4,
                        "q {interleaved} {f16} {normed}"
                    );
                    assert!(
                        worst(&flat(&gk), &flat(&wk)) < 1e-4,
                        "k {interleaved} {f16} {normed}"
                    );
                    let wv = flat(&v.to_dtype(DType::F32).unwrap());
                    assert_eq!(flat(&gv), wv, "v {interleaved} {f16} {normed}");
                }
            }
        }
    }

    /// Rows at once give each row what the one-row launch gives it, bit for bit, from F16 and
    /// F32 rows, each row with its own tables.
    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn rows_at_once_give_each_row_its_own_launch() {
        let dev = Device::new_cuda(0).expect("a card for the card test");
        let (n_head, n_kv, hd, rope_dim, rows) = (6usize, 2usize, 64usize, 32usize, 3usize);
        let on = |v: Vec<f32>, dims: &[usize]| Tensor::from_vec(v, dims.to_vec(), &dev).unwrap();
        let cos = on(ramp(rows * rope_dim / 2, 5), &[rows, rope_dim / 2]);
        let sin = on(ramp(rows * rope_dim / 2, 6), &[rows, rope_dim / 2]);
        let qw = on(ramp(hd, 7), &[hd]);
        let kw = on(ramp(hd, 8), &[hd]);
        let host = |t: &Tensor| -> Vec<f32> { t.flatten_all().unwrap().to_vec1().unwrap() };
        for f16 in [false, true] {
            let cast = |t: Tensor| {
                if f16 {
                    t.to_dtype(DType::F16).unwrap()
                } else {
                    t
                }
            };
            let q = cast(on(ramp(rows * n_head * hd, 1), &[rows, n_head * hd]));
            let k = cast(on(ramp(rows * n_kv * hd, 2), &[rows, n_kv * hd]));
            let v = cast(on(ramp(rows * n_kv * hd, 3), &[rows, n_kv * hd]));
            let shape = QkShape {
                n_head,
                n_kv,
                head_dim: hd,
                rope_dim,
                interleaved: false,
                eps: 1e-6,
            };
            let (gq, gk, gv) =
                qk_prep_rows(&q, &k, &v, Some(&qw), Some(&kw), &cos, &sin, shape, rows)
                    .unwrap()
                    .expect("the launch takes these rows");
            for r in 0..rows {
                let row = |t: &Tensor| t.narrow(0, r, 1).unwrap();
                let (oq, ok, ov) = qk_prep(
                    &row(&q),
                    &row(&k),
                    &row(&v),
                    Some(&qw),
                    Some(&kw),
                    &row(&cos),
                    &row(&sin),
                    shape,
                )
                .unwrap()
                .expect("the launch takes this row");
                assert_eq!(host(&row(&gq)), host(&oq), "row {r}'s queries, f16 {f16}");
                assert_eq!(host(&row(&gk)), host(&ok), "row {r}'s keys, f16 {f16}");
                assert_eq!(host(&row(&gv)), host(&ov), "row {r}'s values, f16 {f16}");
            }
        }
    }
}
