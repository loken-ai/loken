//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Scatter a per-token F16 K/V residual into the ring slot.
///
/// # Safety
/// `src` and `dst` must be valid CUDA device pointers on the device `stream`
/// belongs to, `src` readable for `n_kv * head_dim` F16 elements and `dst`
/// writable for the whole ring, `slot` in `[0, 32)`. The kernel is enqueued on
/// `stream`, so both must stay alive until that stream has been synchronised.
pub unsafe fn kv_residual_scatter_f16_raw(
    src: *const core::ffi::c_void,
    dst: *mut core::ffi::c_void,
    n_kv: i32,
    head_dim: i32,
    slot: i32,
    stream: i64,
) {
    loken_kv_residual_scatter_f16(src, dst, n_kv, head_dim, slot, stream)
}

/// Device-slot variant of [`kv_residual_scatter_f16_raw`] - reads the slot from
/// `slot_dev[0] % 32` on the GPU (needed under CUDA-graph capture).
///
/// # Safety
/// As [`kv_residual_scatter_f16_raw`], plus `slot_dev` must be a valid device
/// pointer readable for one 32-bit slot index. The slot is read on the GPU, so
/// it is not validated host-side - a wild value indexes the ring out of bounds.
pub unsafe fn kv_residual_scatter_f16_dev_slot_raw(
    src: *const core::ffi::c_void,
    dst: *mut core::ffi::c_void,
    slot_dev: *const core::ffi::c_void,
    n_kv: i32,
    head_dim: i32,
    stream: i64,
) {
    loken_kv_residual_scatter_f16_dev_slot(src, dst, slot_dev, n_kv, head_dim, stream)
}

/// Byte-copy scatter for the Q4_0 V append path, with a device-side position.
///
/// # Safety
/// `src`, `dst` and `pos_dev` must be valid device pointers on `stream`'s device:
/// `src` readable for `token_bytes`, `dst` writable at the position `pos_dev`
/// names, `pos_dev` readable for one index. `token_bytes` must be a multiple of 4
/// (checked only in debug builds). The position is read on the GPU and therefore
/// unvalidated here.
pub unsafe fn q4_v_scatter_bytes_dev_pos_raw(
    src: *const core::ffi::c_void,
    dst: *mut core::ffi::c_void,
    pos_dev: *const core::ffi::c_void,
    token_bytes: i32,
    stream: i64,
) {
    debug_assert!(token_bytes % 4 == 0, "token_bytes must be multiple of 4");
    loken_q4_v_scatter_bytes_dev_pos(src, dst, pos_dev, token_bytes, stream)
}

/// Conditional Q4_0 quantize+flush of the K residual under graph capture.
///
/// # Safety
/// All four pointers must be valid device pointers on `stream`'s device:
/// `residual` readable for `n_kv * head_dim` elements, `k_blocks` writable for
/// `max_seq_blocks` Q4_0 blocks, `pos_dev` readable for one index. The flush is
/// conditional on the device-side position, which is not validated host-side.
pub unsafe fn flush_k_residual_q4_dev_pos_raw(
    residual: *const core::ffi::c_void,
    k_blocks: *mut core::ffi::c_void,
    pos_dev: *const core::ffi::c_void,
    n_kv: i32,
    head_dim: i32,
    max_seq_blocks: i32,
    stream: i64,
) {
    loken_flush_k_residual_q4_dev_pos(
        residual,
        k_blocks,
        pos_dev,
        n_kv,
        head_dim,
        max_seq_blocks,
        stream,
    )
}

/// Gated DeltaNet linear-attention recurrence. Returns `(output, new_state)`; a sequence of
/// at least one chunk runs in the chunked form.
pub fn gated_delta_net(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    scale: f32,
) -> Result<(Tensor, Tensor)> {
    let chunked = q.dims3()?.0 >= unsafe { loken_gated_delta_net_chunk_rows() } as usize;
    delta_net_launch(q, k, v, g, beta, state, scale, chunked)
}

/// [`gated_delta_net`] with the form named: `chunked` runs the rows in chunks of matrix
/// products, otherwise one row after the other, each row's arithmetic that of a single row.
#[allow(clippy::too_many_arguments)]
pub(crate) fn delta_net_launch(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    scale: f32,
    chunked: bool,
) -> Result<(Tensor, Tensor)> {
    let dev = q.device().as_cuda_device()?;
    let (n_tokens, h, s_v) = q.dims3()?;
    let cuptr = |t: &Tensor| -> Result<*const f32> {
        let (st, layout) = t.storage_and_layout();
        match &*st {
            StorageView::Cuda(c) => {
                let c = c.as_cuda_slice::<f32>()?;
                let base = c.device_ptr(c.stream()).0 as *const f32;
                // `device_ptr` is the storage base; a contiguous tensor can still
                // carry a non-zero layout start_offset (e.g. `v` is a contiguous
                // narrow of the conv output at seq==1, offset 2*kd*kg). Without
                // this the kernel reads the wrong slice and the recurrence drifts.
                Ok(unsafe { base.add(layout.start_offset()) })
            }
            _ => crate::tensor::bail!("gated_delta_net: inputs must be CUDA f32"),
        }
    };
    let (qp, kp, vp) = (cuptr(q)?, cuptr(k)?, cuptr(v)?);
    let (gp, bp, sp) = (cuptr(g)?, cuptr(beta)?, cuptr(state)?);
    let in_place = super::in_place_state_on();
    if in_place && !state.is_contiguous() {
        crate::tensor::bail!("gated_delta_net: an in-place state must be contiguous");
    }
    let o_slice = unsafe { dev.alloc::<f32>(n_tokens * h * s_v) }?;
    // In place, the state is written where it was read: each warp owns a column of it, reads
    // the column into registers first and writes it back last.
    let ns_slice = if in_place {
        None
    } else {
        Some(unsafe { dev.alloc::<f32>(h * s_v * s_v) }?)
    };
    let stream = dev.cuda_stream().cu_stream() as i64;
    let (sq1, sq2) = (s_v as i64, (h * s_v) as i64);
    let (sv1, sv2) = (s_v as i64, (h * s_v) as i64);
    let (sg1, sg2) = (1i64, h as i64);
    let state_out = match &ns_slice {
        Some(ns) => ns.device_ptr(ns.stream()).0 as *mut f32,
        None => sp as *mut f32,
    };
    if chunked {
        let floats =
            unsafe { loken_gated_delta_net_chunked_scratch(h as i32, n_tokens as i32, s_v as i32) };
        let scratch = unsafe { dev.alloc::<f32>(floats as usize) }?;
        unsafe {
            loken_gated_delta_net_chunked(
                qp,
                kp,
                vp,
                gp,
                bp,
                sp,
                o_slice.device_ptr(o_slice.stream()).0 as *mut f32,
                state_out,
                h as i32,
                n_tokens as i32,
                s_v as i32,
                sq1,
                sq2,
                sv1,
                sv2,
                sg1,
                sg2,
                scale,
                scratch.device_ptr(scratch.stream()).0 as *mut f32,
                stream,
            );
        }
    } else {
        unsafe {
            loken_gated_delta_net(
                qp,
                kp,
                vp,
                gp,
                bp,
                sp,
                o_slice.device_ptr(o_slice.stream()).0 as *mut f32,
                state_out,
                h as i32,
                n_tokens as i32,
                s_v as i32,
                sq1,
                sq2,
                sv1,
                sv2,
                sg1,
                sg2,
                scale,
                stream,
            );
        }
    }
    let o = tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(o_slice, dev.clone()),
        (n_tokens, h, s_v),
    )?;
    let ns = match ns_slice {
        Some(ns_slice) => tensor_from_cuda_storage(
            CudaStorage::wrap_cuda_slice(ns_slice, dev.clone()),
            (h, s_v, s_v),
        )?,
        None => state.reshape((h, s_v, s_v))?,
    };
    Ok((o, ns))
}

/// Fused causal depthwise conv1d + SiLU for the DeltaNet input projection.
/// `qkv` [seq, C], `conv_state` [C, K-1], `w` [C, K] (all f32, CUDA). Returns
/// `(out [seq, C], new_conv_state [C, K-1])`. Replaces ~16 tensor ops/layer.
pub fn fused_conv_silu(
    qkv: &Tensor,
    conv_state: &Tensor,
    w: &Tensor,
    conv_kernel: usize,
) -> Result<(Tensor, Tensor)> {
    let dev = qkv.device().as_cuda_device()?;
    let (seq, c) = qkv.dims2()?;
    let k = conv_kernel;
    let cuptr = |t: &Tensor| -> Result<*const f32> {
        let (st, layout) = t.storage_and_layout();
        match &*st {
            StorageView::Cuda(cs) => {
                let cs = cs.as_cuda_slice::<f32>()?;
                Ok(unsafe {
                    (cs.device_ptr(cs.stream()).0 as *const f32).add(layout.start_offset())
                })
            }
            _ => crate::tensor::bail!("fused_conv_silu: inputs must be CUDA f32"),
        }
    };
    let (qp, csp, wp) = (cuptr(qkv)?, cuptr(conv_state)?, cuptr(w)?);
    let in_place = super::in_place_state_on();
    if in_place && !conv_state.is_contiguous() {
        crate::tensor::bail!("fused_conv_silu: an in-place state must be contiguous");
    }
    let out_slice = unsafe { dev.alloc::<f32>(seq * c) }?;
    // In place, each channel reads its whole row of state before it writes it back.
    let ncs_slice = if in_place {
        None
    } else {
        Some(unsafe { dev.alloc::<f32>(c * (k - 1)) }?)
    };
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_fused_conv_silu(
            qp,
            csp,
            wp,
            out_slice.device_ptr(out_slice.stream()).0 as *mut f32,
            match &ncs_slice {
                Some(n) => n.device_ptr(n.stream()).0 as *mut f32,
                None => csp as *mut f32,
            },
            c as i32,
            seq as i32,
            k as i32,
            stream,
        );
    }
    let out = tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(out_slice, dev.clone()),
        (seq, c),
    )?;
    let ncs = match ncs_slice {
        Some(ncs_slice) => tensor_from_cuda_storage(
            CudaStorage::wrap_cuda_slice(ncs_slice, dev.clone()),
            (c, k - 1),
        )?,
        None => conv_state.clone(),
    };
    Ok((out, ncs))
}

/// Fused z-gated RMSNorm for the DeltaNet output. `o`,`z` [N, D], `norm_w` [D]
/// (f32, CUDA). Returns `y` [N, D] = rmsnorm(o).norm_w.silu(z). Replaces ~9
/// tensor ops/layer.
pub fn zgate_rmsnorm(
    o: &Tensor,
    z: &Tensor,
    norm_w: &Tensor,
    eps: f32,
    sigmoid_gate: bool,
) -> Result<Tensor> {
    let dev = o.device().as_cuda_device()?;
    let (n, d) = o.dims2()?;
    let cuptr = |t: &Tensor| -> Result<*const f32> {
        let (st, layout) = t.storage_and_layout();
        match &*st {
            StorageView::Cuda(cs) => {
                let cs = cs.as_cuda_slice::<f32>()?;
                Ok(unsafe {
                    (cs.device_ptr(cs.stream()).0 as *const f32).add(layout.start_offset())
                })
            }
            _ => crate::tensor::bail!("zgate_rmsnorm: inputs must be CUDA f32"),
        }
    };
    let (op, zp, wp) = (cuptr(o)?, cuptr(z)?, cuptr(norm_w)?);
    let y_slice = unsafe { dev.alloc::<f32>(n * d) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_zgate_rmsnorm(
            op,
            zp,
            wp,
            y_slice.device_ptr(y_slice.stream()).0 as *mut f32,
            n as i32,
            d as i32,
            eps,
            sigmoid_gate as i32,
            stream,
        );
    }
    tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(y_slice, dev.clone()), (n, d))
}

/// Fused DeltaNet gating. `alpha`,`beta_in` [N, H]; `a_log`,`dt_bias` [H]
/// (f32, CUDA). Returns `(g_pre [N,H], beta [N,H])` where
/// `g_pre = a_log.softplus(alpha+dt_bias)` (PRE-exp) and `beta = sigmoid(beta_in)`.
pub fn deltanet_gate(
    alpha: &Tensor,
    beta_in: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let dev = alpha.device().as_cuda_device()?;
    let (n, h) = alpha.dims2()?;
    let cuptr = |t: &Tensor| -> Result<*const f32> {
        let (st, layout) = t.storage_and_layout();
        match &*st {
            StorageView::Cuda(cs) => {
                let cs = cs.as_cuda_slice::<f32>()?;
                Ok(unsafe {
                    (cs.device_ptr(cs.stream()).0 as *const f32).add(layout.start_offset())
                })
            }
            _ => crate::tensor::bail!("deltanet_gate: inputs must be CUDA f32"),
        }
    };
    let (ap, bp, alp, dtp) = (
        cuptr(alpha)?,
        cuptr(beta_in)?,
        cuptr(a_log)?,
        cuptr(dt_bias)?,
    );
    let g_slice = unsafe { dev.alloc::<f32>(n * h) }?;
    let b_slice = unsafe { dev.alloc::<f32>(n * h) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_deltanet_gate(
            ap,
            bp,
            alp,
            dtp,
            g_slice.device_ptr(g_slice.stream()).0 as *mut f32,
            b_slice.device_ptr(b_slice.stream()).0 as *mut f32,
            n as i32,
            h as i32,
            stream,
        );
    }
    let g = tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(g_slice, dev.clone()), (n, h))?;
    let beta =
        tensor_from_cuda_storage(CudaStorage::wrap_cuda_slice(b_slice, dev.clone()), (n, h))?;
    Ok((g, beta))
}

#[cfg(test)]
mod timing {
    use super::*;
    use crate::tensor::{Device, Tensor};

    /// The scan over the first rows of a sequence leaves the state the same tokens leave one
    /// at a time, bit for bit, for every count of rows: a window returned to its kept rows
    /// runs them again in one launch.
    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn the_scan_over_any_first_rows_is_the_rows_one_at_a_time() {
        let Ok(dev) = Device::new_cuda(0) else {
            return;
        };
        let (h, s_v, n) = (4usize, 64usize, 5usize);
        let q = Tensor::randn(0f32, 1.0, (n, h, s_v), &dev).unwrap();
        let k = Tensor::randn(0f32, 1.0, (n, h, s_v), &dev).unwrap();
        let v = Tensor::randn(0f32, 1.0, (n, h, s_v), &dev).unwrap();
        let g = Tensor::randn(-1f32, 0.1, (n, h), &dev).unwrap();
        let beta = Tensor::randn(0.5f32, 0.1, (n, h), &dev).unwrap();
        let _in_place = super::super::in_place_state(true);
        let steps = Tensor::zeros_on((h, s_v, s_v), crate::tensor::DType::F32, &dev).unwrap();
        let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for t in 0..n {
            let row = |x: &Tensor| x.narrow(0, t, 1).unwrap();
            gated_delta_net(
                &row(&q),
                &row(&k),
                &row(&v),
                &row(&g),
                &row(&beta),
                &steps,
                0.1,
            )
            .unwrap();
            let first = |x: &Tensor| x.narrow(0, 0, t + 1).unwrap();
            let state = Tensor::zeros_on((h, s_v, s_v), crate::tensor::DType::F32, &dev).unwrap();
            delta_net_launch(
                &first(&q),
                &first(&k),
                &first(&v),
                &first(&g),
                &first(&beta),
                &state,
                0.1,
                false,
            )
            .unwrap();
            assert_eq!(flat(&state), flat(&steps), "after {} rows", t + 1);
        }
    }

    /// Inputs of the recurrence as the layer prepares them: unit-norm q and k, decay logs from
    /// near zero (long memory) to strongly negative, gates in (0, 1), and a carried state.
    fn delta_inputs(
        dev: &Device,
        n: usize,
        h: usize,
        s_v: usize,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
        let unit = |t: Tensor| {
            let norm = t.sqr().unwrap().sum_keepdim(2).unwrap().sqrt().unwrap();
            t.broadcast_div(&norm).unwrap()
        };
        let q = unit(Tensor::randn(0f32, 1.0, (n, h, s_v), dev).unwrap());
        let k = unit(Tensor::randn(0f32, 1.0, (n, h, s_v), dev).unwrap());
        let v = Tensor::randn(0f32, 1.0, (n, h, s_v), dev).unwrap();
        let g = Tensor::randn(0f32, 1.0, (n, h), dev)
            .unwrap()
            .abs()
            .unwrap()
            .neg()
            .unwrap();
        let beta =
            crate::tensor::ops::sigmoid(&Tensor::randn(0f32, 1.0, (n, h), dev).unwrap()).unwrap();
        let state = Tensor::randn(0f32, 0.1, (h, s_v, s_v), dev).unwrap();
        (q, k, v, g, beta, state)
    }

    fn worst_relative(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let peak = b.iter().fold(0f32, |m, x| m.max(x.abs()));
        let worst = a
            .iter()
            .zip(&b)
            .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        worst / peak
    }

    /// The chunked form gives the outputs and the final state of the row-by-row scan, across
    /// head dims, partial last chunks and many chunks.
    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn the_chunked_delta_rule_matches_the_row_by_row_scan() {
        let Ok(dev) = Device::new_cuda(0) else {
            return;
        };
        let _copy = super::super::in_place_state(false);
        for (h, s_v) in [(4usize, 64usize), (32, 128), (4, 256)] {
            for n in [64usize, 65, 127, 300, 1000] {
                let (q, k, v, g, beta, state) = delta_inputs(&dev, n, h, s_v);
                let scale = 1.0 / (s_v as f32).sqrt();
                let (o_row, s_row) =
                    delta_net_launch(&q, &k, &v, &g, &beta, &state, scale, false).unwrap();
                let (o_chunk, s_chunk) =
                    delta_net_launch(&q, &k, &v, &g, &beta, &state, scale, true).unwrap();
                let (eo, es) = (
                    worst_relative(&o_chunk, &o_row),
                    worst_relative(&s_chunk, &s_row),
                );
                eprintln!("h={h} s_v={s_v} n={n}: output {eo:.2e}, state {es:.2e} of the peak");
                assert!(eo < 1e-4 && es < 1e-4, "h={h} s_v={s_v} n={n}: {eo} {es}");
            }
        }
    }

    /// The convolution over many rows in one launch is, bit for bit, the rows launched one at
    /// a time, each advancing the window the row before it left.
    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn the_rows_of_one_convolution_are_the_rows_convolved_in_turn() {
        let Ok(dev) = Device::new_cuda(0) else {
            return;
        };
        let _copy = super::super::in_place_state(false);
        let (c, taps, seq) = (96usize, 4usize, 70usize);
        let x = Tensor::randn(0f32, 1.0, (seq, c), &dev).unwrap();
        let w = Tensor::randn(0f32, 0.5, (c, taps), &dev).unwrap();
        let window = Tensor::randn(0f32, 1.0, (c, taps - 1), &dev).unwrap();
        let (all, all_window) = fused_conv_silu(&x, &window, &w, taps).unwrap();
        let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut step_window = window;
        for t in 0..seq {
            let (row, next) =
                fused_conv_silu(&x.narrow(0, t, 1).unwrap(), &step_window, &w, taps).unwrap();
            assert_eq!(flat(&row), flat(&all.narrow(0, t, 1).unwrap()), "row {t}");
            step_window = next;
        }
        assert_eq!(
            flat(&step_window),
            flat(&all_window),
            "the window after every row"
        );
    }

    fn median_ms(mut runs: Vec<f64>) -> f64 {
        runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        runs[runs.len() / 2]
    }

    /// One layer-chunk of prefill rows through the recurrence and the convolution at the served
    /// shapes, the state in F32: the row-by-row scan against the chunked form, the median of
    /// seven runs.
    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn a_prefill_chunk_through_the_delta_rule_timed() {
        let Ok(dev) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let cd = dev.as_cuda_device().unwrap().clone();
        let sync = || cd.0.stream().synchronize().unwrap();
        let time = |f: &dyn Fn()| -> f64 {
            f();
            sync();
            let runs = (0..7)
                .map(|_| {
                    let started = std::time::Instant::now();
                    f();
                    sync();
                    started.elapsed().as_secs_f64() * 1e3
                })
                .collect();
            median_ms(runs)
        };
        let _copy = super::super::in_place_state(false);
        let (k_heads, h, s_v, taps) = (16usize, 32usize, 128usize, 4usize);
        let channels = 2 * k_heads * s_v + h * s_v;
        for n in [1000usize, 4000] {
            let (q, k, v, g, beta, state) = delta_inputs(&dev, n, h, s_v);
            let scale = 1.0 / (s_v as f32).sqrt();
            let rows = time(&|| {
                delta_net_launch(&q, &k, &v, &g, &beta, &state, scale, false).unwrap();
            });
            let chunked = time(&|| {
                delta_net_launch(&q, &k, &v, &g, &beta, &state, scale, true).unwrap();
            });
            let x = Tensor::randn(0f32, 1.0, (n, channels), &dev)
                .unwrap()
                .to_dtype(crate::tensor::DType::F16)
                .unwrap();
            let w = Tensor::randn(0f32, 0.5, (channels, taps), &dev).unwrap();
            let window =
                Tensor::zeros_on((channels, taps - 1), crate::tensor::DType::F32, &dev).unwrap();
            let conv = time(&|| {
                crate::inference::moe_cuda::fused_conv_silu_f16in(&x, &window, &w, taps)
                    .unwrap()
                    .unwrap();
            });
            eprintln!(
                "rows={n} heads={h} s_v={s_v}: scan {rows:.3} ms, chunked {chunked:.3} ms, \
                 conv {conv:.3} ms"
            );
        }
    }

    /// What one token through the recurrence costs at the served shapes, the state advanced in
    /// place: the launch alone, timed over many, against what its bytes take at the card's
    /// bandwidth.
    #[test]
    #[ignore]
    fn a_token_through_the_delta_rule_timed() {
        let Ok(dev) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        for (h, s_v) in [(32usize, 128usize), (48, 128), (16, 128)] {
            let q = Tensor::randn(0f32, 1.0, (1, h, s_v), &dev).unwrap();
            let k = Tensor::randn(0f32, 1.0, (1, h, s_v), &dev).unwrap();
            let v = Tensor::randn(0f32, 1.0, (1, h, s_v), &dev).unwrap();
            let g = Tensor::randn(-1f32, 0.1, (1, h), &dev).unwrap();
            let beta = Tensor::randn(0.5f32, 0.1, (1, h), &dev).unwrap();
            let state = Tensor::zeros_on((h, s_v, s_v), crate::tensor::DType::F32, &dev).unwrap();
            let _in_place = super::super::in_place_state(true);
            for _ in 0..5 {
                gated_delta_net(&q, &k, &v, &g, &beta, &state, 0.1).unwrap();
            }
            let cd = dev.as_cuda_device().unwrap();
            cd.0.stream().synchronize().unwrap();
            let n = 200;
            let started = std::time::Instant::now();
            for _ in 0..n {
                gated_delta_net(&q, &k, &v, &g, &beta, &state, 0.1).unwrap();
            }
            cd.0.stream().synchronize().unwrap();
            let us = started.elapsed().as_secs_f64() * 1e6 / n as f64;
            let bytes = 2.0 * (h * s_v * s_v * 4) as f64;
            eprintln!(
                "gdn step h={h} s_v={s_v}: {us:.1} us a launch, {:.0} GB/s over the state read and written",
                bytes / us * 1e-3
            );
        }
    }
}
