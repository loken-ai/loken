//! The gated delta-net's input stage for a decoded token as one launch.

use super::hyper::addr;
use super::*;

/// What the input stage hands the recurrence: the normalised `q` and `k` and the value row, one
/// per value head (`[1, vg, d]`), the decay and write gates `[1, vg]`, and the convolution window
/// `[conv_dim, K - 1]` advanced by the token.
pub struct GdnPrep {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub g: Tensor,
    pub beta: Tensor,
    pub window: Tensor,
}

/// Longest convolution the launch holds in registers.
const MAX_TAPS: usize = 8;

/// Shape of the delta-net the launch serves.
#[derive(Clone, Copy)]
pub struct GdnShape {
    pub k_heads: usize,
    pub k_dim: usize,
    pub v_heads: usize,
    pub v_dim: usize,
    pub taps: usize,
    pub tiled: bool,
    pub eps: f32,
}

/// Convolution with SiLU, the window advance, the L2 norms and head copies, and the gates, for
/// one token: `qkv` `[1, conv_dim]`, `alpha` and `beta_in` `[1, v_heads]`, all F16 or all F32.
/// `Ok(None)` when the inputs are not of the types or shapes the launch takes.
#[allow(clippy::too_many_arguments)]
pub fn gdn_prep(
    qkv: &Tensor,
    window: &Tensor,
    conv_w: &Tensor,
    alpha: &Tensor,
    beta_in: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    shape: GdnShape,
) -> Result<Option<GdnPrep>> {
    use core::ffi::c_void;
    let GdnShape {
        k_heads: kg,
        k_dim: kd,
        v_heads: vg,
        v_dim: vd,
        taps,
        tiled,
        eps,
    } = shape;
    let conv_dim = 2 * kg * kd + vg * vd;
    let f16 = match qkv.dtype() {
        DType::F16 => true,
        DType::F32 => false,
        _ => return Ok(None),
    };
    let same = alpha.dtype() == qkv.dtype() && beta_in.dtype() == qkv.dtype();
    let floats = [window, conv_w, a_log, dt_bias]
        .iter()
        .all(|t| t.dtype() == DType::F32 && t.is_contiguous());
    if !same
        || !floats
        || !qkv.device().is_cuda()
        || taps < 2
        || taps > MAX_TAPS
        || kg == 0
        || vg % kg != 0
        || qkv.elem_count() != conv_dim
        || alpha.elem_count() != vg
        || beta_in.elem_count() != vg
        || window.elem_count() != conv_dim * (taps - 1)
        || conv_w.elem_count() != conv_dim * taps
        || !qkv.is_contiguous()
        || !alpha.is_contiguous()
        || !beta_in.is_contiguous()
    {
        return Ok(None);
    }
    let dev = qkv.device().as_cuda_device()?;
    let in_place = super::in_place_state_on();
    let q = unsafe { dev.alloc::<f32>(vg * kd) }?;
    let k = unsafe { dev.alloc::<f32>(vg * kd) }?;
    let v = unsafe { dev.alloc::<f32>(vg * vd) }?;
    let g = unsafe { dev.alloc::<f32>(vg) }?;
    let beta = unsafe { dev.alloc::<f32>(vg) }?;
    let fresh = if in_place {
        None
    } else {
        Some(unsafe { dev.alloc::<f32>(conv_dim * (taps - 1)) }?)
    };
    let window_p = addr::<f32>(window, "window")? as *mut f32;
    let (qkv_p, alpha_p, beta_p) = if f16 {
        (
            addr::<half::f16>(qkv, "qkv")?,
            addr::<half::f16>(alpha, "alpha")?,
            addr::<half::f16>(beta_in, "beta")?,
        )
    } else {
        (
            addr::<f32>(qkv, "qkv")?,
            addr::<f32>(alpha, "alpha")?,
            addr::<f32>(beta_in, "beta")?,
        )
    };
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_gdn_prep(
            qkv_p as *const c_void,
            window_p,
            addr::<f32>(conv_w, "conv_w")? as *const f32,
            alpha_p as *const c_void,
            beta_p as *const c_void,
            addr::<f32>(a_log, "a_log")? as *const f32,
            addr::<f32>(dt_bias, "dt_bias")? as *const f32,
            q.device_ptr(q.stream()).0 as *mut f32,
            k.device_ptr(k.stream()).0 as *mut f32,
            v.device_ptr(v.stream()).0 as *mut f32,
            g.device_ptr(g.stream()).0 as *mut f32,
            beta.device_ptr(beta.stream()).0 as *mut f32,
            match &fresh {
                Some(n) => n.device_ptr(n.stream()).0 as *mut f32,
                None => window_p,
            },
            kg as i32,
            kd as i32,
            vg as i32,
            vd as i32,
            taps as i32,
            i32::from(tiled),
            eps,
            i32::from(f16),
            stream,
        );
    }
    let wrap = |s, dims: (usize, usize)| {
        tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(s, dev.clone()), dims)
    };
    let window = match fresh {
        Some(n) => wrap(n, (conv_dim, taps - 1))?,
        None => window.clone(),
    };
    Ok(Some(GdnPrep {
        q: tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(q, dev.clone()), (1, vg, kd))?,
        k: tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(k, dev.clone()), (1, vg, kd))?,
        v: tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(v, dev.clone()), (1, vg, vd))?,
        g: wrap(g, (1, vg))?,
        beta: wrap(beta, (1, vg))?,
        window,
    }))
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

    fn flat(t: &Tensor) -> Vec<f32> {
        t.to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    fn worst(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// The launch gives what the separate stages give: the convolution with SiLU, the window,
    /// the normalised heads copied to the value heads in either order, and the gates.
    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn the_input_stage_matches_its_separate_launches() {
        let dev = Device::new_cuda(0).expect("a card for the card test");
        let (kg, kd, vg, vd, taps) = (2usize, 64usize, 6usize, 48usize, 4usize);
        let conv_dim = 2 * kg * kd + vg * vd;
        for tiled in [false, true] {
            for f16 in [false, true] {
                let on = |v: Vec<f32>, dims: (usize, usize)| {
                    Tensor::from_vec(v, dims, &Device::Cpu)
                        .unwrap()
                        .to_device(&dev)
                        .unwrap()
                };
                let mut qkv = on(ramp(conv_dim, 1), (1, conv_dim));
                let mut alpha = on(ramp(vg, 2), (1, vg));
                let mut beta_in = on(ramp(vg, 3), (1, vg));
                if f16 {
                    qkv = qkv.to_dtype(DType::F16).unwrap();
                    alpha = alpha.to_dtype(DType::F16).unwrap();
                    beta_in = beta_in.to_dtype(DType::F16).unwrap();
                }
                let window = on(ramp(conv_dim * (taps - 1), 4), (conv_dim, taps - 1));
                let conv_w = on(ramp(conv_dim * taps, 5), (conv_dim, taps));
                let a_log = on(ramp(vg, 6), (1, vg)).flatten_all().unwrap();
                let dt_bias = on(ramp(vg, 7), (1, vg)).flatten_all().unwrap();
                let shape = GdnShape {
                    k_heads: kg,
                    k_dim: kd,
                    v_heads: vg,
                    v_dim: vd,
                    taps,
                    tiled,
                    eps: 1e-6,
                };
                let got = gdn_prep(
                    &qkv, &window, &conv_w, &alpha, &beta_in, &a_log, &dt_bias, shape,
                )
                .unwrap()
                .expect("the launch takes this shape");

                let q32 = qkv.to_dtype(DType::F32).unwrap();
                let (c, new_window) = fused_conv_silu(&q32, &window, &conv_w, taps).unwrap();
                let qk = kg * kd;
                let part = |at: usize, n: usize, h: usize, d: usize| {
                    c.narrow(crate::tensor::D::Minus1, at, n)
                        .unwrap()
                        .reshape((1, h, d))
                        .unwrap()
                        .contiguous()
                        .unwrap()
                };
                let rep = vg / kg;
                let q = l2norm_gqa(&part(0, qk, kg, kd), kg, kd, rep, tiled, 1e-6).unwrap();
                let k = l2norm_gqa(&part(qk, qk, kg, kd), kg, kd, rep, tiled, 1e-6).unwrap();
                let v = part(2 * qk, vg * vd, vg, vd);
                let (g, beta) = deltanet_gate(
                    &alpha.to_dtype(DType::F32).unwrap(),
                    &beta_in.to_dtype(DType::F32).unwrap(),
                    &a_log,
                    &dt_bias,
                )
                .unwrap();
                let label = format!("tiled {tiled}, f16 {f16}");
                assert!(worst(&flat(&got.q), &flat(&q)) < 1e-5, "{label}: q");
                assert!(worst(&flat(&got.k), &flat(&k)) < 1e-5, "{label}: k");
                assert!(worst(&flat(&got.v), &flat(&v)) < 1e-6, "{label}: v");
                assert!(worst(&flat(&got.g), &flat(&g)) < 1e-6, "{label}: g");
                assert!(
                    worst(&flat(&got.beta), &flat(&beta)) < 1e-6,
                    "{label}: beta"
                );
                assert_eq!(flat(&got.window), flat(&new_window), "{label}: window");
            }
        }
    }
}
