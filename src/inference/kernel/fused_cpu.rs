//! CPU fallback shim for `fused_kernels` (the CUDA fused-op launchers).
//!
//! Selected via `#[cfg(not(feature = "cuda"))]` in `mod.rs`, this provides the
//! same public API as the real CUDA `fused_kernels` so the MoE/hybrid arches
//! (gpt-oss, nemotron, lfm2, qwen3.5) build and run on the CPU-only target.
//!
//! Strategy:
//!  - `Result<Option<Tensor>>` ops return `Ok(None)` -> the caller falls back to
//!    its own portable path (the fused kernel is only an optimisation).
//!  - Non-optional ops (fused norms/activations, attention helpers) get a plain
//!    a portable implementation matching the fused kernel's math.
//!  - Device-scalar in-place helpers used only by the CUDA-graph decode path are
//!    no-ops on CPU (the CPU decode path uses host positions, not device pos).
//!
//! Only the CPU-typed subset of the API lives here; functions that take/return
//! CUDA-only types (CudaSlice/CudaStream/CudaDevice) are never referenced by
//! CPU-compiled code (their callers are themselves `#[cfg(feature="cuda")]`).

use crate::tensor::{DType, Result, Tensor, D};

// -- Option-returning ops: return None so the caller uses its portable fallback --
pub fn gptoss_flash_decode(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _mask: Option<&Tensor>,
    _sinks: Option<&Tensor>,
    _scale: f32,
    _batch: usize,
    _n_head: usize,
    _n_kv: usize,
    _kv_len: usize,
    _head_dim: usize,
) -> Result<Option<Tensor>> {
    Ok(None)
}

pub fn gptoss_flash_decode_win(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _mask: Option<&Tensor>,
    _sinks: Option<&Tensor>,
    _scale: f32,
    _batch: usize,
    _n_head: usize,
    _n_kv: usize,
    _kv_start: usize,
    _kv_len: usize,
    _head_dim: usize,
) -> Result<Option<Tensor>> {
    Ok(None)
}

pub fn flash_decode(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _mask: Option<&Tensor>,
    _sinks: Option<&Tensor>,
    _scale: f32,
    _batch: usize,
    _n_head: usize,
    _n_kv: usize,
    _kv_len: usize,
    _head_dim: usize,
) -> Result<Option<Tensor>> {
    Ok(None)
}

pub fn flash_dit_bf16(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _scale: f32,
    _frame_tokens: usize,
    _grid_w: usize,
) -> Result<Option<Tensor>> {
    Ok(None)
}

pub fn flash_decode_devkvlen(
    _q: &Tensor,
    _kbuf: &Tensor,
    _vbuf: &Tensor,
    _pos_dev: &Tensor,
    _sinks: Option<&Tensor>,
    _window: usize,
    _scale: f32,
    _batch: usize,
    _n_head: usize,
    _n_kv: usize,
    _head_dim: usize,
) -> Result<Option<Tensor>> {
    Ok(None)
}

pub fn embed_gather_f16(_table: &Tensor, _tok: &Tensor, _d_model: usize) -> Result<Option<Tensor>> {
    Ok(None)
}
pub fn silu_mul_f16(_g: &Tensor, _u: &Tensor) -> Result<Option<Tensor>> {
    Ok(None)
}
pub fn neox_rope_f16(
    _x: &Tensor,
    _cos: &Tensor,
    _sin: &Tensor,
    _rope_dim: usize,
) -> Result<Option<Tensor>> {
    Ok(None)
}
pub fn neox_rope_devpos_f16(
    _x: &Tensor,
    _c: &Tensor,
    _s: &Tensor,
    _p: &Tensor,
    _rope_dim: usize,
) -> Result<Option<Tensor>> {
    Ok(None)
}
pub fn paged_rope_f16(
    _x: &Tensor,
    _cos: &Tensor,
    _sin: &Tensor,
    _interleaved: bool,
) -> Result<Option<Tensor>> {
    Ok(None)
}
pub fn add_to_f16(_a: &Tensor, _b: &Tensor) -> Result<Option<Tensor>> {
    Ok(None)
}
pub fn softplus_bias(_dt: &Tensor, _bias: &Tensor) -> Result<Option<Tensor>> {
    Ok(None)
}
pub fn relu2_f16(_x: &Tensor) -> Result<Option<Tensor>> {
    Ok(None)
}
pub fn fused_shexp_out(
    _routed: &Tensor,
    _down: &Tensor,
    _gate_logit: &Tensor,
) -> Result<Option<Tensor>> {
    Ok(None)
}
pub fn fused_softmax_sinks(
    _scores: &Tensor,
    _mask: Option<&Tensor>,
    _sinks: &Tensor,
    _scale: f32,
) -> Result<Option<Tensor>> {
    Ok(None)
}

pub fn paged_flash_decode(
    _q: &Tensor,
    _kp: &Tensor,
    _vp: &Tensor,
    _block_table: &Tensor,
    _seq_lens: &Tensor,
    _scale: f32,
    _batch: usize,
    _n_head: usize,
    _n_kv: usize,
    _head_dim: usize,
    _block_size: usize,
    _max_blocks: usize,
) -> Result<Option<Tensor>> {
    Ok(None)
}

// The CUDA launcher is itself a no-op for non-CUDA/non-F16 tensors; mirror that.
pub fn paged_kv_write(
    _k_new: &Tensor,
    _v_new: &Tensor,
    _kp: &Tensor,
    _vp: &Tensor,
    _slot_dev: &Tensor,
    _batch: usize,
    _feat: usize,
) -> Result<()> {
    Ok(())
}

// -- Device-scalar in-place helpers (CUDA-graph decode only): no-op on CPU --
pub fn set_i32_inplace(_t: &Tensor, _val: i32) -> Result<()> {
    Ok(())
}
pub fn copy_u32_dev(_src: &Tensor, _dst: &Tensor) -> Result<()> {
    Ok(())
}
pub fn copy_f32_dev(_src: &Tensor, _dst: &Tensor) -> Result<()> {
    Ok(())
}

// -- rmsnorm helper (matches the kernel: x * rsqrt(mean(x²)+eps) * weight) --
fn rmsnorm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xf.broadcast_div(&(var + eps as f64)?.sqrt()?)?;
    let w = weight.to_dtype(DType::F32)?;
    normed.broadcast_mul(&w)?.to_dtype(dt)
}

pub fn fused_rmsnorm_f16(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    rmsnorm(x, weight, eps)
}

// Matches the CUDA launcher's fallback branch: rmsnorm in F32, rounded through
// the input dtype, emitted as F32 (bit-identical to the fused kernel's store).
pub fn fused_rmsnorm_f16_out_f32(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    crate::tensor::ops::rms_norm(&x.to_dtype(DType::F32)?, weight, eps)?
        .to_dtype(x.dtype())?
        .to_dtype(DType::F32)
}

pub fn fused_add_rmsnorm_f16(
    x: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    let s = (x + residual)?;
    let n = rmsnorm(&s, weight, eps)?;
    Ok((n, s))
}

pub fn fused_add_rmsnorm_dual(
    x: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    fused_add_rmsnorm_f16(x, residual, weight, eps)
}

pub fn fused_rmsnorm_then_add(
    x: &Tensor,
    weight: &Tensor,
    residual: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    rmsnorm(x, weight, eps)? + residual
}

pub fn fused_rmsnorm_add_scale(
    x: &Tensor,
    weight: &Tensor,
    residual: &Tensor,
    scale: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let n = rmsnorm(x, weight, eps)?;
    n.broadcast_mul(scale)?.add(residual)
}

pub fn fused_dual_rmsnorm_add(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    w1: &Tensor,
    w2: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    // (rmsnorm(a,w1) + rmsnorm(b,w2)) + c  - matches the fused gemma-style dual norm.
    let na = rmsnorm(a, w1, eps)?;
    let nb = rmsnorm(b, w2, eps)?;
    (na + nb)?.add(c)
}

pub fn fused_gemma4_post_add_norm(
    attn_out: &Tensor,
    residual: &Tensor,
    w_post: &Tensor,
    w_ffn: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    // post-attn rmsnorm(attn_out)+residual -> new residual; rmsnorm(new_res, w_ffn) -> ffn input.
    let s = (rmsnorm(attn_out, w_post, eps)? + residual)?;
    let n = rmsnorm(&s, w_ffn, eps)?;
    Ok((n, s))
}

// -- activations / GLU combines --
pub fn fused_silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    up * crate::tensor::ops::silu(gate)?
}
pub fn fused_gelu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    up * gate.gelu()?
}
pub fn fused_split_silu_mul(gu: &Tensor) -> Result<Tensor> {
    let n = gu.dim(D::Minus1)? / 2;
    let g = gu.narrow(D::Minus1, 0, n)?;
    let u = gu.narrow(D::Minus1, n, n)?;
    fused_silu_mul(&g, &u)
}
pub fn fused_split_gelu_mul(gu: &Tensor) -> Result<Tensor> {
    let n = gu.dim(D::Minus1)? / 2;
    let g = gu.narrow(D::Minus1, 0, n)?;
    let u = gu.narrow(D::Minus1, n, n)?;
    fused_gelu_mul(&g, &u)
}
pub fn fused_swiglu_oai(gate: &Tensor, up: &Tensor, alpha: f64, limit: f64) -> Result<Tensor> {
    fused_swiglu_oai_bias(gate, up, None, None, alpha, limit)
}
pub fn fused_swiglu_oai_bias(
    gate: &Tensor,
    up: &Tensor,
    gbias: Option<&Tensor>,
    ubias: Option<&Tensor>,
    alpha: f64,
    limit: f64,
) -> Result<Tensor> {
    let g = match gbias {
        Some(b) => gate.broadcast_add(b)?,
        None => gate.clone(),
    };
    let u = match ubias {
        Some(b) => up.broadcast_add(b)?,
        None => up.clone(),
    };
    let lim = Tensor::full(limit as f32, g.shape(), &g.device())?.to_dtype(g.dtype())?;
    let x = g.minimum(&lim)?;
    let gg = u.minimum(&lim)?.maximum(&lim.affine(-1.0, 0.0)?)?;
    let act = x.mul(&crate::tensor::ops::sigmoid(&x.affine(alpha as f32, 0.0)?)?)?;
    act.mul(&gg.affine(1.0, 1.0)?)
}
pub fn fused_bias_gelu_new(x: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.broadcast_add(bias)?.gelu()
}
pub fn fused_add_three(a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
    (a + b)? + c
}
pub fn fused_phi2_residual_merge(
    residual: &Tensor,
    attn_out: &Tensor,
    attn_bias: &Tensor,
    ffn_out: &Tensor,
    ffn_bias: &Tensor,
) -> Result<Tensor> {
    // residual + (attn_out+attn_bias) + (ffn_out+ffn_bias)  (phi2 parallel block)
    let a = attn_out.broadcast_add(attn_bias)?;
    let f = ffn_out.broadcast_add(ffn_bias)?;
    (residual + a)? + f
}

// -- KV-cache write at host position (CPU uses pos_dev[0] read back to host) --
pub fn kv_write_at_pos(
    k_new: &Tensor,
    v_new: &Tensor,
    kbuf: &Tensor,
    vbuf: &Tensor,
    pos_dev: &Tensor,
    _kv_max: usize,
) -> Result<()> {
    // pos_dev is a [1] i32 tensor; read it to host and slice_set the new K/V.
    let pos = pos_dev.to_dtype(DType::U32)?.to_vec1::<u32>()?[0] as usize;
    // kbuf/vbuf: [b, n_kv, kv_max, hd]; write k_new [b,n_kv,seq,hd] at kv position `pos`.
    let kv_dim = kbuf.rank() - 2; // the kv_max axis
    kbuf.slice_set(k_new, kv_dim, pos)?;
    vbuf.slice_set(v_new, kv_dim, pos)?;
    Ok(())
}

// -- attention sink mask (additive [seq, kv]); None -> caller builds its own --
pub fn fused_gptoss_mask(
    _seq: usize,
    _kv: usize,
    _input_pos: usize,
    _window: usize,
    _device: &crate::tensor::Device,
) -> Result<Option<Tensor>> {
    Ok(None)
}

// -- lfm2 gated short causal conv (per-token decode) --
// bcx: [b, 3*d_model] = (B, C, x) for the current token; state_in: [b, d_model,
// l_cache-1] (the previous bx window); conv_w: [d_model, l_cache]. Matches the
// fused kernel: bx = B.x; window = [state, bx]; acc = Σ_k window[:,:,k].conv_w[:,k];
// y = C.acc; new_state = shift(state, bx). Returns (y [b, d_model], new_state).
pub fn fused_lfm2_shortconv(
    bcx: &Tensor,
    state_in: &Tensor,
    conv_w: &Tensor,
    d_model: usize,
    l_cache: usize,
) -> Result<(Tensor, Tensor)> {
    let b = bcx.dim(0)?;
    let bx = bcx.narrow(D::Minus1, 0, d_model)?; // B  [b,d]
    let cx = bcx.narrow(D::Minus1, d_model, d_model)?; // C  [b,d]
    let xx = bcx.narrow(D::Minus1, 2 * d_model, d_model)?; // x  [b,d]
    let bx_x = (&bx * &xx)?.reshape((b, d_model, 1))?; // B.x  [b,d,1]
    let window = Tensor::cat(&[&state_in.to_dtype(bx_x.dtype())?, &bx_x], 2)?; // [b,d,l_cache]
    let w = conv_w
        .reshape((1, d_model, l_cache))?
        .to_dtype(window.dtype())?;
    let acc = window.broadcast_mul(&w)?.sum(2)?; // [b,d]
    let y = (&cx * &acc)?; // C.acc  [b,d]
    let new_state = window.narrow(2, 1, l_cache - 1)?.contiguous()?; // drop oldest
    Ok((y, new_state))
}

// -- nemotron Mamba2 causal depthwise conv1d + bias + silu (per-token decode) --
// x: [b, d] (current token); state_in: [b, d, l-1]; conv_w: [d,1,l]; conv_b: [d].
// window = [state, x]; acc = Σ_k window.conv_w + bias; y = silu(acc); state shifts.
pub fn fused_causal_conv1d_silu(
    x: &Tensor,
    state_in: &Tensor,
    conv_w: &Tensor,
    conv_b: &Tensor,
    d: usize,
    l: usize,
) -> Result<(Tensor, Tensor)> {
    // conv_state holds the last `l` (= d_conv) inputs (full window). Roll it:
    // drop the oldest, append the new token x, keeping width l. new_state = window.
    let b = x.dim(0)?;
    let x3 = x.reshape((b, d, 1))?;
    let state = state_in.to_dtype(x3.dtype())?;
    let window = Tensor::cat(&[&state.narrow(2, 1, l - 1)?, &x3], 2)?; // [b,d,l]
    let w = conv_w.reshape((1, d, l))?.to_dtype(window.dtype())?;
    let acc = window
        .broadcast_mul(&w)?
        .sum(2)? // [b,d]
        .broadcast_add(&conv_b.reshape((1, d))?)?;
    let y = crate::tensor::ops::silu(&acc)?;
    Ok((y, window.contiguous()?))
}

// -- Mamba2 gate + group-norm epilogue: y = gnorm(y_ssm + D⊙x_c) . silu(z) . w --
// y_ssm: [b, d_inner] (caller reshapes the SSM output); x_c: [b, d_inner];
// dvec(D): [nh]; z: [b, d_inner]; norm_w: [d_inner]. nh = d_inner / hd.
pub fn fused_mamba2_gate_gnorm(
    y_ssm: &Tensor,
    x_c: &Tensor,
    dvec: &Tensor,
    z: &Tensor,
    norm_w: &Tensor,
    bsz: usize,
    d_inner: usize,
    hd: usize,
    ngroups: usize,
    eps: f64,
) -> Result<Tensor> {
    // EXACT kernel order: yv = (y_ssm + D⊙x).silu(z)  [GATE FIRST], then
    // group-rmsnorm over yv, then .norm_w. (Gating before the norm matters - the
    // RMS is computed over the gated values.)
    let nh = d_inner / hd;
    let ys = y_ssm.reshape((bsz, nh, hd))?;
    let xh = x_c.reshape((bsz, nh, hd))?;
    let dh = dvec.reshape((1, nh, 1))?;
    let pre = (ys + xh.broadcast_mul(&dh)?)?.reshape((bsz, d_inner))?; // y_ssm + D⊙x
    let yv = (pre * crate::tensor::ops::silu(z)?)?; // gate by silu(z)
                                                    // group rmsnorm over (d_inner/ngroups) of the GATED yv, then per-channel weight.
    let g = d_inner / ngroups;
    let yg = yv.reshape((bsz, ngroups, g))?;
    let var = yg.sqr()?.mean_keepdim(D::Minus1)?;
    let yn = yg.broadcast_div(&(var + eps)?.sqrt()?)?; // [b, ngroups, g]
    yn.broadcast_mul(&norm_w.reshape((1, ngroups, g))?)?
        .reshape((bsz, d_inner))
}

// -- Mamba2 single-token SSM step. Returns y [b, nh, hd] (NOT flattened) --
// x_c [b, nh*hd], B/C [b, ngroups*ds], dt [b, nh], A [nh], h_in [b, nh, hd, ds].
pub fn fused_mamba2_ssm_step(
    x_c: &Tensor,
    b_: &Tensor,
    c_: &Tensor,
    dt: &Tensor,
    a: &Tensor,
    h_in: &Tensor,
    nh: usize,
    hd: usize,
    ds: usize,
    ngroups: usize,
) -> Result<(Tensor, Tensor)> {
    let bsz = x_c.dim(0)?;
    let x = x_c.reshape((bsz, nh, hd, 1))?;
    let dt_ = dt.reshape((bsz, nh, 1, 1))?;
    let a_ = a.reshape((1, nh, 1, 1))?;
    let da = dt_.broadcast_mul(&a_)?.exp()?; // [bsz,nh,1,1]
    let gh = nh / ngroups.max(1);
    let bg = b_
        .reshape((bsz, ngroups, 1, ds))?
        .broadcast_as((bsz, ngroups, gh, ds))?
        .reshape((bsz, nh, 1, ds))?;
    let cg = c_
        .reshape((bsz, ngroups, 1, ds))?
        .broadcast_as((bsz, ngroups, gh, ds))?
        .reshape((bsz, nh, 1, ds))?;
    let dbx = dt_.broadcast_mul(&x)?.broadcast_mul(&bg)?; // [bsz,nh,hd,ds]
    let h_new = (h_in.broadcast_mul(&da)? + dbx)?; // [bsz,nh,hd,ds]
    let y = h_new.broadcast_mul(&cg)?.sum(D::Minus1)?; // [bsz,nh,hd]
    Ok((y, h_new))
}

// -- greedy argmax with repetition penalty (CPU sampler) --
pub fn fused_penalty_argmax(
    logits: &Tensor,
    penalty_token_ids: &[u32],
    repeat_penalty: f32,
) -> Result<u32> {
    let mut v: Vec<f32> = logits.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
    if repeat_penalty != 1.0 {
        for &t in penalty_token_ids {
            let i = t as usize;
            if i < v.len() {
                v[i] = if v[i] > 0.0 {
                    v[i] / repeat_penalty
                } else {
                    v[i] * repeat_penalty
                };
            }
        }
    }
    // AVX2-vectorized argmax, bit-identical first-max-wins tie-break vs the
    // scalar loop (greedy exactness) - once per token over the full vocab.
    Ok(crate::inference::kernel::cpu_decode_exec::argmax_f32(&v))
}
