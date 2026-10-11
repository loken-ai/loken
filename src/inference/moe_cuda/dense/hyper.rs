//! The wide residual's elementwise stages as launches: the grouped norm, the low-rank
//! nonlinearity, the gated mean over the streams, and the write-back with the next norm.
//!
//! The residual is F32 `[rows, streams * dim]`; the sublayer reads and writes F32 or F16.

use super::*;

/// Device address of an F32 or F16 tensor's first element.
pub(super) fn addr<T: crate::tensor::kernel_ffi::CudaDType>(t: &Tensor, what: &str) -> Result<u64> {
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

fn f32_t(t: &Tensor, what: &str) -> Result<Tensor> {
    if t.dtype() != DType::F32 {
        crate::tensor::bail!("{what}: expected F32, got {:?}", t.dtype());
    }
    t.contiguous()
}

/// `r` `[rows, streams * dim]` normalised stream by stream and scaled by `gamma`
/// `[streams * dim]`.
pub fn hc_norm(r: &Tensor, gamma: &Tensor, streams: usize, dim: usize, eps: f32) -> Result<Tensor> {
    let dev = r.device().as_cuda_device()?;
    let (r, gamma) = (f32_t(r, "hc_norm input")?, f32_t(gamma, "hc_norm gamma")?);
    let rows = r.elem_count() / (streams * dim);
    let out = unsafe { dev.alloc::<f32>(r.elem_count()) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_hc_norm(
            addr::<f32>(&r, "r")? as *const f32,
            addr::<f32>(&gamma, "gamma")? as *const f32,
            out.device_ptr(out.stream()).0 as *mut f32,
            (rows * streams) as i32,
            streams as i32,
            dim as i32,
            eps,
            stream,
        );
    }
    tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(out, dev.clone()),
        r.shape().clone(),
    )
}

/// `silu(down / streams)` elementwise over an F32 tensor.
pub fn hc_low(down: &Tensor, streams: usize) -> Result<Tensor> {
    let dev = down.device().as_cuda_device()?;
    let down = f32_t(down, "hc_low input")?;
    let n = down.elem_count();
    let out = unsafe { dev.alloc::<f32>(n) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_hc_low(
            addr::<f32>(&down, "down")? as *const f32,
            out.device_ptr(out.stream()).0 as *mut f32,
            n as i32,
            streams as i32,
            stream,
        );
    }
    tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(out, dev.clone()),
        down.shape().clone(),
    )
}

/// The sublayer input: the mean over the streams of `xn * sigmoid(up)`, `[rows, dim]`, F16 when
/// `f16_out` else F32.
pub fn hc_mean_gate(
    xn: &Tensor,
    up: &Tensor,
    streams: usize,
    dim: usize,
    f16_out: bool,
) -> Result<Tensor> {
    let dev = xn.device().as_cuda_device()?;
    let (xn, up) = (f32_t(xn, "hc_mean_gate xn")?, f32_t(up, "hc_mean_gate up")?);
    let rows = xn.elem_count() / (streams * dim);
    let stream = dev.cuda_stream().cu_stream() as i64;
    let shape = crate::tensor::Shape::from((1, rows, dim));
    if f16_out {
        let out = unsafe { dev.alloc::<half::f16>(rows * dim) }?;
        unsafe {
            loken_hc_mean_gate(
                addr::<f32>(&xn, "xn")? as *const f32,
                addr::<f32>(&up, "up")? as *const f32,
                out.device_ptr(out.stream()).0 as *mut c_void,
                rows as i32,
                streams as i32,
                dim as i32,
                1,
                stream,
            );
        }
        tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(out, dev.clone()), shape)
    } else {
        let out = unsafe { dev.alloc::<f32>(rows * dim) }?;
        unsafe {
            loken_hc_mean_gate(
                addr::<f32>(&xn, "xn")? as *const f32,
                addr::<f32>(&up, "up")? as *const f32,
                out.device_ptr(out.stream()).0 as *mut c_void,
                rows as i32,
                streams as i32,
                dim as i32,
                0,
                stream,
            );
        }
        tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(out, dev.clone()), shape)
    }
}

/// Most rows a step may carry for [`hc_mix_fused`]: a decode step or a draft verify, whose rows
/// then see the arithmetic of the decode steps they stand for. A longer chunk reads each weight
/// once for all its rows in a matrix product instead.
pub const FUSED_MIX_MAX_ROWS: usize = crate::inference::engine::decode_step::VERIFY_ROWS;

/// A mixer weight `[n, k]` as a launch reads it: dense F32 or F16, or int8 with a scale for each
/// row.
#[derive(Clone, Copy)]
pub enum MixWeight<'a> {
    Dense(&'a Tensor),
    Int8 { q: &'a Tensor, scale: &'a Tensor },
}

/// The values of an int8 row that share a scale.
pub const INT8_GROUP: usize = 64;

/// How a launch reads a weight: 0 for F32, 1 for F16, 2 for int8 with row scales.
const WEIGHT_F32: i32 = 0;
const WEIGHT_F16: i32 = 1;
const WEIGHT_INT8: i32 = 2;

/// A weight `[n, k]` that is contiguous on a card: its rows and how a launch reads it, or `None`.
fn dense_rows(w: MixWeight<'_>, n: Option<usize>, k: usize) -> Option<(usize, i32)> {
    match w {
        MixWeight::Dense(w) => {
            let [rows, cols] = w.dims() else { return None };
            ((w.dtype() == DType::F32 || w.dtype() == DType::F16)
                && w.device().is_cuda()
                && w.is_contiguous()
                && *cols == k
                && n.is_none_or(|n| n == *rows))
            .then_some((
                *rows,
                if w.dtype() == DType::F16 {
                    WEIGHT_F16
                } else {
                    WEIGHT_F32
                },
            ))
        }
        MixWeight::Int8 { q, scale } => {
            let [rows, cols] = q.dims() else { return None };
            (q.dtype() == DType::U8
                && scale.dtype() == DType::F32
                && q.device().is_cuda()
                && q.is_contiguous()
                && scale.is_contiguous()
                && cols % INT8_GROUP == 0
                && scale.elem_count() == *rows * (*cols / INT8_GROUP)
                && *cols == k
                && n.is_none_or(|n| n == *rows))
            .then_some((*rows, WEIGHT_INT8))
        }
    }
}

/// The mixer of a decode step in two launches: `down` and `inject` together, then `up` with the
/// gated mean over the streams. Returns the sublayer input `[1, rows, dim]` (F16 when `f16_out`)
/// and the raw write-back weights `[1, rows, streams]`; `None` when the weights or the step are
/// not of the shapes the launches take, and the caller runs the stage-by-stage chain.
pub fn hc_mix_fused(
    xn: &Tensor,
    down: &Tensor,
    up: &Tensor,
    inject: Option<&Tensor>,
    streams: usize,
    dim: usize,
    f16_out: bool,
) -> Result<Option<(Tensor, Option<Tensor>)>> {
    hc_mix_fused_over(
        xn,
        MixWeight::Dense(down),
        MixWeight::Dense(up),
        inject.map(MixWeight::Dense),
        streams,
        dim,
        f16_out,
    )
}

/// [`hc_mix_fused`] over weights held dense or as int8 rows.
pub fn hc_mix_fused_over(
    xn: &Tensor,
    down: MixWeight<'_>,
    up: MixWeight<'_>,
    inject: Option<MixWeight<'_>>,
    streams: usize,
    dim: usize,
    f16_out: bool,
) -> Result<Option<(Tensor, Option<Tensor>)>> {
    let k = streams * dim;
    let rows = xn.elem_count() / k.max(1);
    if rows == 0 || rows > FUSED_MIX_MAX_ROWS || xn.dtype() != DType::F32 || !xn.is_contiguous() {
        return Ok(None);
    }
    let Some((rank, half_w)) = dense_rows(down, None, k) else {
        return Ok(None);
    };
    match dense_rows(up, Some(k), rank) {
        Some((_, h)) if h == half_w => {}
        _ => return Ok(None),
    }
    let heads = match inject {
        Some(w) => match dense_rows(w, None, k) {
            Some((n, h)) if h == half_w => n,
            _ => return Ok(None),
        },
        None => 0,
    };
    let dev = xn.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let low = unsafe { dev.alloc::<f32>(rows * rank) }?;
    let inj = unsafe { dev.alloc::<f32>((rows * heads).max(1)) }?;
    let xn_p = addr::<f32>(xn, "xn")? as *const f32;
    let weight = |w: MixWeight<'_>, name: &str| -> Result<(*const c_void, *const f32)> {
        Ok(match w {
            MixWeight::Dense(w) if half_w == WEIGHT_F16 => (
                addr::<half::f16>(w, name)? as *const c_void,
                std::ptr::null(),
            ),
            MixWeight::Dense(w) => (addr::<f32>(w, name)? as *const c_void, std::ptr::null()),
            MixWeight::Int8 { q, scale } => (
                addr::<u8>(q, name)? as *const c_void,
                addr::<f32>(scale, name)? as *const f32,
            ),
        })
    };
    let (down_p, down_scale) = weight(down, "down")?;
    let (up_p, up_scale) = weight(up, "up")?;
    let (inj_w, inj_scale) = match inject {
        Some(w) => weight(w, "inject")?,
        None => (std::ptr::null(), std::ptr::null()),
    };
    let low_p = low.device_ptr(low.stream()).0;
    unsafe {
        loken_hc_down_inject(
            xn_p,
            down_p,
            inj_w,
            low_p as *mut f32,
            inj.device_ptr(inj.stream()).0 as *mut f32,
            rows as i32,
            k as i32,
            rank as i32,
            heads as i32,
            streams as i32,
            half_w,
            down_scale,
            inj_scale,
            stream,
        );
    }
    let shape = crate::tensor::Shape::from((1, rows, dim));
    let mixed = if f16_out {
        let out = unsafe { dev.alloc::<half::f16>(rows * dim) }?;
        unsafe {
            loken_hc_up_mean(
                xn_p,
                up_p,
                low_p as *const f32,
                out.device_ptr(out.stream()).0 as *mut c_void,
                rows as i32,
                streams as i32,
                dim as i32,
                rank as i32,
                1,
                half_w,
                up_scale,
                stream,
            );
        }
        tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(out, dev.clone()), shape)?
    } else {
        let out = unsafe { dev.alloc::<f32>(rows * dim) }?;
        unsafe {
            loken_hc_up_mean(
                xn_p,
                up_p,
                low_p as *const f32,
                out.device_ptr(out.stream()).0 as *mut c_void,
                rows as i32,
                streams as i32,
                dim as i32,
                rank as i32,
                0,
                half_w,
                up_scale,
                stream,
            );
        }
        tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(out, dev.clone()), shape)?
    };
    let injected = if heads == 0 {
        None
    } else {
        Some(tensor_from_cuda_storage(
            CudaStorage::wrap_cuda_slice(inj, dev.clone()),
            crate::tensor::Shape::from((1, rows, heads)),
        )?)
    };
    Ok(Some((mixed, injected)))
}

/// `r + out (x) 2 sigmoid(inject / streams)` over `r` `[rows, streams * dim]`, with `inject`
/// `[rows, streams]` F32 and `out` `[rows, dim]` F32 or F16. When `next_gamma` is given the next
/// mixer's grouped norm of the result comes back too.
pub fn hc_combine(
    r: &Tensor,
    out: &Tensor,
    inject: &Tensor,
    next_gamma: Option<(&Tensor, f32)>,
    streams: usize,
    dim: usize,
) -> Result<(Tensor, Option<Tensor>)> {
    let dev = r.device().as_cuda_device()?;
    let r = f32_t(r, "hc_combine residual")?;
    let inject = f32_t(inject, "hc_combine inject")?;
    let out = out.contiguous()?;
    let out_f16 = match out.dtype() {
        DType::F16 => true,
        DType::F32 => false,
        other => crate::tensor::bail!("hc_combine: the sublayer output is {other:?}"),
    };
    let rows = r.elem_count() / (streams * dim);
    let out_addr = if out_f16 {
        addr::<half::f16>(&out, "out")?
    } else {
        addr::<f32>(&out, "out")?
    };
    let new_r = unsafe { dev.alloc::<f32>(r.elem_count()) }?;
    let normed = match next_gamma {
        Some(_) => Some(unsafe { dev.alloc::<f32>(r.elem_count()) }?),
        None => None,
    };
    let gamma_c = next_gamma
        .map(|(g, _)| f32_t(g, "hc_combine gamma"))
        .transpose()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_hc_combine(
            addr::<f32>(&r, "r")? as *const f32,
            out_addr as *const c_void,
            addr::<f32>(&inject, "inject")? as *const f32,
            new_r.device_ptr(new_r.stream()).0 as *mut f32,
            normed.as_ref().map_or(std::ptr::null_mut(), |n| {
                n.device_ptr(n.stream()).0 as *mut f32
            }),
            match &gamma_c {
                Some(g) => addr::<f32>(g, "gamma")? as *const f32,
                None => std::ptr::null(),
            },
            (rows * streams) as i32,
            streams as i32,
            dim as i32,
            next_gamma.map_or(0.0, |(_, e)| e),
            out_f16 as i32,
            stream,
        );
    }
    let shape = r.shape().clone();
    let new_r = tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(new_r, dev.clone()),
        shape.clone(),
    )?;
    let normed = normed
        .map(|n| tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(n, dev.clone()), shape))
        .transpose()?;
    Ok((new_r, normed))
}

/// An int8 weight `[n, k]` with a scale for each group of [`INT8_GROUP`] values of a row, widened
/// to F32 or to F16.
pub fn hc_dequant(q: &Tensor, scale: &Tensor, f32_out: bool) -> Result<Tensor> {
    let [n, k] = q.dims() else {
        crate::tensor::bail!("hc_dequant: the weight is not a matrix");
    };
    let (n, k) = (*n, *k);
    let dev = q.device().as_cuda_device()?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let shape = crate::tensor::Shape::from((n, k));
    macro_rules! widen {
        ($ty:ty, $flag:expr) => {{
            let out = unsafe { dev.alloc::<$ty>(n * k) }?;
            unsafe {
                loken_hc_dequant(
                    addr::<u8>(q, "q")? as *const c_void,
                    addr::<f32>(scale, "scale")? as *const f32,
                    out.device_ptr(out.stream()).0 as *mut c_void,
                    n as i32,
                    k as i32,
                    $flag,
                    stream,
                );
            }
            tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(out, dev.clone()), shape)
        }};
    }
    if f32_out {
        widen!(f32, 1)
    } else {
        widen!(half::f16, 0)
    }
}

#[cfg(test)]
mod timing {
    use super::*;
    use crate::tensor::Device;

    /// What the mixer's launches cost at the served shapes, int8 rows, one row of input: the
    /// down and inject launch, the up and mean launch, and the combine with the next norm.
    #[test]
    #[ignore = "measurement: run explicitly with --release --ignored --nocapture"]
    fn what_a_mixer_costs_a_sublayer() {
        let Ok(card) = crate::tensor::cuda::CudaDevice::new(0).map(Device::Cuda) else {
            return;
        };
        let (streams, dim, rank) = (4usize, 2560usize, 320usize);
        let k = streams * dim;
        let bytes = |n: usize, seed: u32| -> Vec<u8> {
            (0..n)
                .map(|i| ((i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed) >> 24) as u8)
                .collect()
        };
        let scales =
            |n: usize| -> Vec<f32> { (0..n).map(|i| 0.01 + (i % 7) as f32 * 1e-3).collect() };
        let q = |rows: usize, cols: usize, seed: u32| -> (Tensor, Tensor) {
            (
                Tensor::from_vec(bytes(rows * cols, seed), (rows, cols), &card).unwrap(),
                Tensor::from_vec(
                    scales(rows * cols / INT8_GROUP),
                    (rows, cols / INT8_GROUP),
                    &card,
                )
                .unwrap(),
            )
        };
        let (down_q, down_s) = q(rank, k, 1);
        let (up_q, up_s) = q(k, rank, 2);
        let (inj_q, inj_s) = q(streams, k, 3);
        let r = Tensor::from_vec(
            (0..k).map(|i| ((i % 97) as f32) * 0.01 - 0.4).collect(),
            (1, 1, k),
            &card,
        )
        .unwrap();
        let gamma = Tensor::from_vec(vec![1.0f32; k], (k,), &card).unwrap();
        let out = Tensor::from_vec(
            (0..dim).map(|i| ((i % 31) as f32) * 0.02).collect(),
            (1, 1, dim),
            &card,
        )
        .unwrap();
        let cd = card.as_cuda_device().unwrap();
        let time = |what: &str, f: &dyn Fn()| {
            for _ in 0..10 {
                f();
            }
            cd.0.stream().synchronize().unwrap();
            let n = 200;
            let t = std::time::Instant::now();
            for _ in 0..n {
                f();
            }
            cd.0.stream().synchronize().unwrap();
            eprintln!(
                "{what}: {:.1} us a launch",
                t.elapsed().as_secs_f64() * 1e6 / n as f64
            );
        };
        let xn = hc_norm(&r, &gamma, streams, dim, 1e-6).unwrap();
        time("norm", &|| {
            hc_norm(&r, &gamma, streams, dim, 1e-6).unwrap();
        });
        time("down and inject, up and mean", &|| {
            hc_mix_fused_over(
                &xn,
                MixWeight::Int8 {
                    q: &down_q,
                    scale: &down_s,
                },
                MixWeight::Int8 {
                    q: &up_q,
                    scale: &up_s,
                },
                Some(MixWeight::Int8 {
                    q: &inj_q,
                    scale: &inj_s,
                }),
                streams,
                dim,
                true,
            )
            .unwrap()
            .expect("the launches take these");
        });
        let (_, inj) = hc_mix_fused_over(
            &xn,
            MixWeight::Int8 {
                q: &down_q,
                scale: &down_s,
            },
            MixWeight::Int8 {
                q: &up_q,
                scale: &up_s,
            },
            Some(MixWeight::Int8 {
                q: &inj_q,
                scale: &inj_s,
            }),
            streams,
            dim,
            true,
        )
        .unwrap()
        .unwrap();
        let inj = inj.unwrap();
        time("combine with the next norm", &|| {
            hc_combine(&r, &out, &inj, Some((&gamma, 1e-6)), streams, dim).unwrap();
        });
    }
}
