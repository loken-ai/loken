//! The elementwise stages around a mixture's shared expert, for one decoded token, as launches.

use super::hyper::addr;
use super::*;
use core::ffi::c_void;

fn width(t: &Tensor) -> Option<bool> {
    match t.dtype() {
        DType::F16 => Some(true),
        DType::F32 => Some(false),
        _ => None,
    }
}

/// `silu(gate) * up` for two `[1, n]` projections, each F32 or F16, as F32 `[1, n]`.
/// `Ok(None)` when they are not of the types or shapes the launch takes.
pub fn shexp_swiglu(gate: &Tensor, up: &Tensor) -> Result<Option<Tensor>> {
    let (Some(gate_f16), Some(up_f16)) = (width(gate), width(up)) else {
        return Ok(None);
    };
    if !gate.device().is_cuda()
        || gate.dims() != up.dims()
        || gate.dims().len() != 2
        || gate.dims()[0] != 1
    {
        return Ok(None);
    }
    let (gate, up) = (gate.contiguous()?, up.contiguous()?);
    let n = gate.elem_count();
    let dev = gate.device().as_cuda_device()?;
    let out = unsafe { dev.alloc::<f32>(n) }?;
    let ptr = |t: &Tensor, f16: bool, what: &str| -> Result<*const c_void> {
        Ok(if f16 {
            addr::<half::f16>(t, what)? as *const c_void
        } else {
            addr::<f32>(t, what)? as *const c_void
        })
    };
    unsafe {
        loken_shexp_swiglu(
            ptr(&gate, gate_f16, "gate")?,
            ptr(&up, up_f16, "up")?,
            out.device_ptr(out.stream()).0 as *mut f32,
            n as i32,
            i32::from(gate_f16),
            i32::from(up_f16),
            dev.cuda_stream().cu_stream() as i64,
        );
    }
    let shape = gate.shape().clone();
    Ok(Some(tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(out, dev.clone()),
        shape,
    )?))
}

/// `routed + sigmoid(w . x) * down` for `routed` and `x` F32 `[1, n]`, `w` F32 with `n`
/// elements and `down` `[1, n]` F32 or F16: the shared expert's scaled output added to the
/// routed ones. `Ok(None)` when the inputs are not of the types or shapes the launch takes.
pub fn shexp_tail(
    routed: &Tensor,
    down: &Tensor,
    x: &Tensor,
    w: &Tensor,
) -> Result<Option<Tensor>> {
    let Some(down_f16) = width(down) else {
        return Ok(None);
    };
    let n = routed.elem_count();
    let f32_row = |t: &Tensor| t.dtype() == DType::F32 && t.is_contiguous() && t.device().is_cuda();
    if !(f32_row(routed) && f32_row(x) && f32_row(w))
        || routed.dims().len() != 2
        || routed.dims()[0] != 1
        || down.elem_count() != n
        || x.elem_count() != n
        || w.elem_count() != n
    {
        return Ok(None);
    }
    let down = down.contiguous()?;
    let dev = routed.device().as_cuda_device()?;
    let out = unsafe { dev.alloc::<f32>(n) }?;
    let down_p = if down_f16 {
        addr::<half::f16>(&down, "down")? as *const c_void
    } else {
        addr::<f32>(&down, "down")? as *const c_void
    };
    unsafe {
        loken_shexp_tail(
            addr::<f32>(routed, "routed")? as *const f32,
            down_p,
            addr::<f32>(x, "x")? as *const f32,
            addr::<f32>(w, "w")? as *const f32,
            out.device_ptr(out.stream()).0 as *mut f32,
            n as i32,
            i32::from(down_f16),
            dev.cuda_stream().cu_stream() as i64,
        );
    }
    let shape = routed.shape().clone();
    Ok(Some(tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(out, dev.clone()),
        shape,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Device;

    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn the_shared_expert_launches_match_the_tensor_ops() {
        let Ok(card) = crate::tensor::cuda::CudaDevice::new(0).map(Device::Cuda) else {
            return;
        };
        let n = 2560;
        let wave = |k: f32| -> Vec<f32> { (0..n).map(|i| ((i as f32) * k).sin()).collect() };
        let t = |v: Vec<f32>| Tensor::from_vec(v, (1, n), &card).unwrap();
        let flat = |t: &Tensor| {
            t.to_device(&Device::Cpu)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let (gate, up) = (t(wave(0.013)), t(wave(0.029)));
        for half in [false, true] {
            let cast = |x: &Tensor| {
                if half {
                    x.to_dtype(DType::F16).unwrap()
                } else {
                    x.clone()
                }
            };
            let (g, u) = (cast(&gate), cast(&up));
            let h = shexp_swiglu(&g, &u)
                .unwrap()
                .expect("the launch takes these");
            let want = (crate::tensor::ops::silu(&g.to_dtype(DType::F32).unwrap()).unwrap()
                * u.to_dtype(DType::F32).unwrap())
            .unwrap();
            let gap = flat(&h)
                .iter()
                .zip(flat(&want))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(gap < 1e-5, "swiglu gap {gap} (half {half})");

            let (routed, x) = (t(wave(0.05)), t(wave(0.011)));
            let w = t(wave(0.007));
            let down = cast(&t(wave(0.021)));
            let got = shexp_tail(&routed, &down, &x, &w)
                .unwrap()
                .expect("the launch takes these");
            let dot = (&x * &w).unwrap().sum_keepdim(1).unwrap();
            let scale = crate::tensor::ops::sigmoid(&dot).unwrap();
            let want = (&routed
                + down
                    .to_dtype(DType::F32)
                    .unwrap()
                    .broadcast_mul(&scale)
                    .unwrap())
            .unwrap();
            let gap = flat(&got)
                .iter()
                .zip(flat(&want))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(gap < 1e-4, "tail gap {gap} (half {half})");
        }
    }
}
