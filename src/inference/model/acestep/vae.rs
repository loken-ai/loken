//! ACE-Step 1.5 - AutoencoderOobleck VAE decoder support ( M1, build-order step 2).
//!
//! The Oobleck decoder upsamples a 5 Hz latent to 48 kHz stereo audio through
//! five `ConvTranspose1d` upsampling stages (ratios 2/4/4/6/10 -> 1920x). Naive
//! `conv_transpose_1d` is O(T_in.K.C) with no GEMM reuse and dominates the VAE's
//! FLOP budget (~40%, measured against the oracle `tmp/acestep.cpp`). The oracle
//! decomposes each transposed conv as `mul_mat(W_perm, xᵀ) -> col2im_1d`, routing
//! the heavy contraction through the optimized matmul and leaving only a pure-
//! bandwidth scatter-add (`col2im_1d`) - the one net-new custom op the VAE needs.
//!
//! This module ports that `col2im_1d` scatter (CPU first, correctness-validated
//! against a naive direct `conv_transpose1d` reference) and the GEMM-based
//! `conv_transpose1d` wrapper that composes it. GPU/tiling come later (M1 step 3).
//!
//! Conventions (match the substrate's row-major layout, N=1):
//!   x      : `[C_in, T_in]`            (input signal, channel-major)
//!   weight : `[C_in, C_out, K]`        (PyTorch ConvTranspose1d weight layout)
//!   y      : `[C_out, T_out]`          where `T_out = (T_in-1).stride + K - 2.padding`
//! The GEMM produces columns `col[t_in, k.C_out + oc] = Σ_ci x[ci,t_in].W[ci,oc,k]`;
//! `col2im_1d` scatters input position `t_in`, tap `k` to output position
//! `t_in.stride + k - padding`, accumulating overlaps and cropping the padding.

/// `col2im_1d` scatter-add. Reverses the im2col GEMM step of a transposed 1-D
/// convolution: `col` holds, for every input position and kernel tap, the
/// per-output-channel contribution; this places each at its output position and
/// sums overlaps. `output_padding` extends `T_out` (matches PyTorch).
///
/// `col`: flat `[T_in . (K.C_out)]`, row-major - outer `T_in`, inner `K.C_out`
///        ordered `k.C_out + oc`.
/// returns `(y, t_out)` with `y` flat `[C_out . T_out]`, row-major (C_out outer).
pub fn col2im_1d_f32(
    col: &[f32],
    t_in: usize,
    k: usize,
    c_out: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
) -> (Vec<f32>, usize) {
    // PyTorch: L_out = (L_in-1).stride - 2.padding + (K-1) + output_padding + 1
    //        = (L_in-1).stride + K - 2.padding + output_padding   (dilation 1).
    let t_out =
        ((t_in.saturating_sub(1)) * stride + k + output_padding).saturating_sub(2 * padding);

    let mut y = vec![0f32; c_out * t_out];
    let kco = k * c_out;
    for ti in 0..t_in {
        let base = ti * stride;
        let col_row = ti * kco;
        for kk in 0..k {
            let p = base + kk;
            if p < padding {
                continue;
            }
            let to = p - padding;
            if to >= t_out {
                continue;
            }
            let col_tap = col_row + kk * c_out;
            for o in 0..c_out {
                y[o * t_out + to] += col[col_tap + o];
            }
        }
    }
    (y, t_out)
}

/// GEMM-based transposed 1-D convolution (N=1), the oracle's `vae_conv_t1d`
/// decomposition in pure CPU f32: build the im2col columns via a contraction over
/// `C_in`, then scatter with [`col2im_1d_f32`]. `x`: `[C_in.T_in]` row-major
/// (C_in outer), `weight`: `[C_in.C_out.K]` row-major (C_in,C_out,K), `bias`:
/// `Some([C_out])`. Returns `(y, t_out)`, `y` = `[C_out.T_out]` (C_out outer).
pub fn conv_transpose1d_gemm_f32(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    c_in: usize,
    c_out: usize,
    t_in: usize,
    k: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
) -> (Vec<f32>, usize) {
    debug_assert_eq!(x.len(), c_in * t_in);
    debug_assert_eq!(weight.len(), c_in * c_out * k);
    // col[t_in, k.C_out + oc] = Σ_ci x[ci,t_in] . W[ci,oc,k].
    let kco = k * c_out;
    let mut col = vec![0f32; t_in * kco];
    for ci in 0..c_in {
        let xrow = ci * t_in;
        let wrow = ci * c_out * k;
        for ti in 0..t_in {
            let xv = x[xrow + ti];
            if xv == 0.0 {
                continue;
            }
            let col_row = ti * kco;
            for o in 0..c_out {
                let wco = wrow + o * k;
                for kk in 0..k {
                    col[col_row + kk * c_out + o] += xv * weight[wco + kk];
                }
            }
        }
    }
    let (mut y, t_out) = col2im_1d_f32(&col, t_in, k, c_out, stride, padding, output_padding);
    if let Some(b) = bias {
        debug_assert_eq!(b.len(), c_out);
        for o in 0..c_out {
            let bo = b[o];
            for t in 0..t_out {
                y[o * t_out + t] += bo;
            }
        }
    }
    (y, t_out)
}

/// Snake activation (Oobleck VAE), CPU f32, in place. Matches the substrate's CUDA
/// `native_snake1d_f32`: `y = x + inv_alpha[c].sin(alpha[c].x)²`, per channel. The
/// params are pre-computed at load (`alpha = exp(α)`, `inv_alpha = 1/exp(β)`  -
/// oracle `vae_load_snake`/`vae_load_snake_inv`). `x`: `[C.T]` row-major (C outer);
/// `alpha`/`inv_alpha`: `[C]`.
pub fn snake1d_f32(x: &mut [f32], alpha: &[f32], inv_alpha: &[f32], c: usize, t: usize) {
    debug_assert_eq!(x.len(), c * t);
    debug_assert_eq!(alpha.len(), c);
    debug_assert_eq!(inv_alpha.len(), c);
    for ch in 0..c {
        let (a, ia) = (alpha[ch], inv_alpha[ch]);
        for v in x[ch * t..(ch + 1) * t].iter_mut() {
            let s = (a * *v).sin();
            *v += ia * s * s;
        }
    }
}

/// Standard 1-D convolution, CPU f32 (the VAE's `k=7` dilated and `k=1` convs).
/// Direct form (correctness-first; the perf path uses im2col+matmul later).
/// `x`: `[C_in.T_in]` (C_in outer), `weight`: `[C_out.C_in.K]` (PyTorch
/// `[OC,IC,K]` layout), `bias`: `Some([C_out])`. Returns `(y[C_out.T_out], t_out)`,
/// `t_out = (T_in + 2.padding - dilation.(K-1) - 1)/stride + 1`.
pub fn conv1d_f32(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    c_in: usize,
    c_out: usize,
    t_in: usize,
    k: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> (Vec<f32>, usize) {
    debug_assert_eq!(x.len(), c_in * t_in);
    debug_assert_eq!(weight.len(), c_out * c_in * k);
    let span = dilation * (k - 1) + 1;
    let t_out = (t_in + 2 * padding).saturating_sub(span) / stride + 1;
    let mut y = vec![0f32; c_out * t_out];
    for oc in 0..c_out {
        let wbase = oc * c_in * k;
        for to in 0..t_out {
            // input window start (signed: padding can push it negative)
            let start = (to * stride) as isize - padding as isize;
            let mut acc = 0f32;
            for ci in 0..c_in {
                let xrow = ci * t_in;
                let wrow = wbase + ci * k;
                for kk in 0..k {
                    let ti = start + (kk * dilation) as isize;
                    if ti < 0 || ti as usize >= t_in {
                        continue;
                    }
                    acc += x[xrow + ti as usize] * weight[wrow + kk];
                }
            }
            y[oc * t_out + to] = acc + bias.map_or(0.0, |b| b[oc]);
        }
    }
    (y, t_out)
}

/// Weight-norm fusion (PyTorch `weight_norm`, default `dim=0`): the stored
/// parameterization keeps a direction `v` and a per-`dim0` magnitude `g`; the
/// effective weight is `w[i,...] = g[i] . v[i,...] / ‖v[i,...]‖₂`. The Oobleck VAE
/// stores every conv/convT as `weight_g`+`weight_v` (oracle `vae_fuse_wn`); fuse
/// once at load. Operates on a `[D0 . rest]` row-major array where `D0` is the
/// weight-norm dim (Conv1d `C_out`, ConvTranspose1d `C_in`) - the caller arranges
/// the GGUF tensor into that layout (ggml `[K,Cin,Cout]`/`[K,Cout,Cin]` ->
/// `[D0, rest]`) before calling. `eps` guards a zero-norm row.
pub fn fuse_weight_norm(v: &[f32], g: &[f32], d0: usize, eps: f32) -> Vec<f32> {
    debug_assert_eq!(g.len(), d0);
    debug_assert_eq!(v.len() % d0, 0);
    let rest = v.len() / d0;
    let mut w = vec![0f32; v.len()];
    for i in 0..d0 {
        let row = &v[i * rest..(i + 1) * rest];
        let mut nrm = 0f32;
        for &x in row {
            nrm += x * x;
        }
        let scale = g[i] / (nrm.sqrt() + eps);
        let wrow = &mut w[i * rest..(i + 1) * rest];
        for (o, &x) in wrow.iter_mut().zip(row) {
            *o = x * scale;
        }
    }
    w
}

// -- Oobleck decoder assembly -------------------------------------------------
// Composes the validated kernels into the decoder forward. Weights are held
// post-fusion in this module's kernel layouts: `Conv` weight = `[C_out.C_in.K]`
// (conv1d_f32, PyTorch [OC,IC,K]); `ConvT` weight = `[C_in.C_out.K]`
// (conv_transpose1d_gemm_f32); snake params pre-exp'd (`alpha=exp(α)`,
// `inv_beta=1/exp(β)`). The GGUF loader arranges + fuses into these (next step).

#[cfg(feature = "cuda")]
use crate::tensor::{DType, Storage};
use crate::tensor::{Device as ND, Tensor as NT};

/// Standard free-VRAM-aware placement (same mechanism as every other model). The VAE is
/// a conv pipeline, not a uniform layer stack, so the HeteroPlan degenerates to N=1: the
/// whole decoder packs onto the fastest GPU when it fits its budget (weights + a reserve
/// for the chunked-conv intermediates), else falls back to CPU. `model_size` is what the
/// caller's component actually puts on the card - the file's size on disk is neither: it
/// covers components this loader does not read, and under-states the ones it does by the
/// factor their weights expand by on the way to the device.
pub fn vae_best_device(model_size: u64) -> ND {
    vae_best_device_with_reserve(
        model_size,
        crate::inference::place::audio_demand::oobleck_reserve(
            crate::inference::place::audio_demand::ACE_VAE_REFERENCE_WINDOW,
            &crate::inference::place::audio_demand::ACE_VAE_REFERENCE_LEVELS,
            crate::inference::place::audio_demand::OOBLECK_KERNEL,
        ),
    )
}

/// [`vae_best_device`] for a caller that knows what its decode actually costs.
///
/// The reserve is what stops the planner from putting one more component on the card, so
/// it has to be the decode's, not a family's. One figure covered a 48 kHz decoder that
/// upsamples by 1920, a 24 kHz one that upsamples by 480, and a text encoder that
/// upsamples nothing at all; it can only have been right for one of them.
///
/// [`vae_best_device`] still answers with the 48 kHz decoder's figure, so a caller that has
/// not yet derived its own is placed exactly as it was. That is deliberate: lowering a
/// reserve does not free memory, it lets the planner put one more component on the card,
/// and a caller whose real peak nobody has computed is not the place to find that out.
pub fn vae_best_device_with_reserve(model_size: u64, reserve: u64) -> ND {
    use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan};
    let cudas = crate::inference::place::vram_manager::probe_under_pressure(reserve);
    let budget: Vec<(usize, u64)> = cudas.iter().map(|(i, f, _)| (*i, *f)).collect();
    // The budgets below ALREADY exclude the reserve: `probe_under_pressure` probes through
    // `probe_cuda_devices`, which returns `stable_free - reserve`. Passing it again here
    // subtracted it TWICE - invisible at half a gigabyte, and fatal at twelve, where it
    // took both cards to zero usable and sent a whole video DiT to the host.
    let plan = HeteroPlan::calculate_with_kv_reserve(1, model_size, &budget, &[], 1.0, 0, 0);
    let dev = match plan.segments.first().map(|s| s.kind) {
        Some(DeviceKind::Cuda(idx)) => cudas
            .iter()
            .find(|(i, _, _)| *i == idx)
            .map(|(_, _, d)| d.clone())
            .unwrap_or(ND::Cpu),
        _ => ND::Cpu,
    };
    eprintln!(
        "[ace-vae] placement: {} (free-VRAM gate)",
        plan.segments
            .first()
            .map(|s| s.kind.to_string())
            .unwrap_or_else(|| "CPU".into())
    );
    dev
}

/// What the decoder puts on the card, from the checkpoint's own tensor directory.
///
/// Every weight is dequantised and fused into f32 on the way to the device, so the
/// quantised bytes on disk are not what goes resident - and the file also holds the
/// encoder, which this loader does not read.
fn decoder_resident_bytes(c: &crate::tensor::quantized::gguf_file::Content) -> u64 {
    /// Bytes per element once the loader has expanded and fused the weight.
    const RESIDENT_BYTES: u64 = 4;
    let matched: u64 = c
        .tensor_infos
        .iter()
        .filter(|(n, _)| n.starts_with("decoder."))
        .map(|(_, i)| i.elem_count() as u64 * RESIDENT_BYTES)
        .sum();
    if matched > 0 {
        return matched;
    }
    // A repacked checkpoint that names its tensors differently must not read as "needs
    // nothing": charge the whole directory rather than plan against zero.
    c.tensor_infos
        .values()
        .map(|i| i.elem_count() as u64 * RESIDENT_BYTES)
        .sum()
}

/// The decoder's `(channels out, cumulative upsample)` per block, from its transposed
/// convolutions. This is what makes the decode reserve follow the DECODER: the channel
/// widths and the stride product are the whole of what one window costs.
fn decoder_levels(
    c: &crate::tensor::quantized::gguf_file::Content,
    block_prefix: &str,
    strides: &[usize],
) -> Vec<(usize, usize)> {
    let mut levels = Vec::with_capacity(strides.len());
    let mut up = 1usize;
    for (i, &stride) in strides.iter().enumerate() {
        up *= stride.max(1);
        // ConvT weight is `[c_in, c_out, k]`; the level's cost follows what it EMITS.
        let name = format!("{block_prefix}.{i}.conv_t1.weight");
        let ch = c
            .tensor_infos
            .get(&name)
            .and_then(|t| t.shape.dims().get(1).copied())
            .unwrap_or(0);
        levels.push((ch, up));
    }
    levels
}

fn vae_nt(v: Vec<f32>, shape: (usize, usize, usize), dev: &ND) -> crate::tensor::Result<NT> {
    NT::from_vec_f32(v, shape)?.to_device(dev)
}

/// A fused conv (or convT) weight + bias, as on-device native tensors. `wt` is the
/// native conv kernel: conv `[c_out,c_in,k]`, convT `[c_in,c_out,k]`. `bt` `[1,c_out,1]`.
pub struct ConvW {
    pub wt: NT,
    pub bt: NT,
    pub c_in: usize,
    pub c_out: usize,
    pub k: usize,
}
/// Pre-exp'd Snake params, per channel, as `[1,c,1]` device tensors.
pub struct SnakeW {
    pub alpha: NT,
    pub inv_beta: NT,
    pub c: usize,
}

/// Snake on a `[1,C,T]` device tensor: `y = x + inv_beta.sin(alpha.x)²` (per-channel
/// broadcast). GPU Tensor ops.
fn snake_t(x: &NT, s: &SnakeW) -> crate::tensor::Result<NT> {
    let t = x.broadcast_mul(&s.alpha)?.sin()?;
    let t = t.powf(2.0)?.broadcast_mul(&s.inv_beta)?;
    x.broadcast_add(&t)
}
/// ResUnit: snake1 -> conv1(k7,dil,pad=3.dil) -> snake2 -> conv2(k1) -> +skip.
pub struct ResUnitW {
    pub snake1: SnakeW,
    pub conv1: ConvW,
    pub snake2: SnakeW,
    pub conv2: ConvW,
    pub dilation: usize,
}
/// Decoder block: snake1 -> convT(C_in->C_out, stride upsample) -> 3x ResUnit.
pub struct BlockW {
    pub snake1: SnakeW,
    pub conv_t: ConvW,
    pub stride: usize,
    pub padding: usize,
    pub res: Vec<ResUnitW>,
}
/// Full Oobleck decoder: conv1 -> 5xblock -> snake -> conv2. Weights on `device`.
pub struct OobleckDecoder {
    pub conv1: ConvW,
    pub blocks: Vec<BlockW>,
    pub snake_final: SnakeW,
    pub conv2: ConvW,
    pub device: ND,
}

impl ConvW {
    /// conv1d: `x [1,c_in,T]` -> `[1,c_out,T']` + bias. When the fused weight is
    /// F16 on CUDA, route the contraction through the tensor-core im2col GEMM
    /// (F16 in / F32 accumulate); otherwise the generic F32/CPU `conv1d`.
    fn conv(&self, x: &NT, padding: usize, dilation: usize) -> crate::tensor::Result<NT> {
        #[cfg(feature = "cuda")]
        if self.wt.dtype() == DType::F16 {
            if let (
                Storage::Cuda { data: xd, dev },
                Storage::Cuda {
                    data: wd,
                    dev: wdev,
                },
            ) = (x.storage_arc().as_ref(), self.wt.storage_arc().as_ref())
            {
                if dev.ordinal() == wdev.ordinal() && x.dtype() == DType::F32 {
                    let d = x.dims();
                    // VAE is N=1; conv calls are always stride 1.
                    let (c_in, l) = (d[1], d[2]);
                    let l_out = (l + 2 * padding).saturating_sub(dilation * (self.k - 1) + 1) + 1;
                    let xs = xd.as_f32_slice()?;
                    let xv = xs.slice(0..c_in * l);
                    let wf16 = wd.as_f16_slice()?;
                    let y = crate::tensor::cuda::conv1d_im2col_f16(
                        dev, &xv, wf16, c_in, l, self.c_out, self.k, l_out, padding, 1, dilation,
                    )?;
                    let yt = NT::from_cuda_storage(
                        crate::tensor::cuda::CudaStorage::F32(y),
                        dev.clone(),
                        vec![1, self.c_out, l_out],
                    )?;
                    return yt.broadcast_add(&self.bt);
                }
            }
        }
        x.conv1d(&self.wt, padding, 1, dilation, 1)?
            .broadcast_add(&self.bt)
    }
    /// Strided downsample conv (encoder): `x [1,c_in,T]` -> `[1,c_out,T/stride]` + bias,
    /// K=2.stride, dilation 1. Generic F32 `conv1d` (the encoder keeps weights F32).
    fn conv_down(&self, x: &NT, padding: usize, stride: usize) -> crate::tensor::Result<NT> {
        x.conv1d(&self.wt, padding, stride, 1, 1)?
            .broadcast_add(&self.bt)
    }
    /// convT: `x [1,c_in,T]` -> `[1,c_out,T.stride]` + bias (Oobleck: outpad 0, dil 1).
    /// When the weight is the F16 rearranged form (CUDA), route the heavy `c_in`
    /// contraction through the tensor-core GEMM + col2im (F16 in / F32 accumulate)
    /// instead of the serial F32 gather kernel; otherwise the generic F32/CPU op.
    fn conv_t(&self, x: &NT, padding: usize, stride: usize) -> crate::tensor::Result<NT> {
        #[cfg(feature = "cuda")]
        if self.wt.dtype() == DType::F16 {
            if let (
                Storage::Cuda { data: xd, dev },
                Storage::Cuda {
                    data: wd,
                    dev: wdev,
                },
                Storage::Cuda { data: bd, .. },
            ) = (
                x.storage_arc().as_ref(),
                self.wt.storage_arc().as_ref(),
                self.bt.storage_arc().as_ref(),
            ) {
                if dev.ordinal() == wdev.ordinal() && x.dtype() == DType::F32 {
                    let d = x.dims();
                    let (c_in, l_in) = (d[1], d[2]);
                    // Oobleck convT: output_padding 0, dilation 1.
                    let l_out = (l_in - 1) * stride + self.k - 2 * padding;
                    let xs = xd.as_f32_slice()?;
                    let wr = wd.as_f16_slice()?;
                    let bias = bd.as_f32_slice()?;
                    let y = crate::tensor::cuda::convt1d_gemm_f16(
                        dev, xs, wr, bias, c_in, self.c_out, l_in, l_out, self.k, stride, padding,
                    )?;
                    return NT::from_cuda_storage(
                        crate::tensor::cuda::CudaStorage::F32(y),
                        dev.clone(),
                        vec![1, self.c_out, l_out],
                    );
                }
            }
        }
        x.conv_transpose1d(&self.wt, padding, 0, stride, 1, 1)?
            .broadcast_add(&self.bt)
    }
}

impl ResUnitW {
    /// `x [1,ch,T]` -> length-preserving (k7 same pad=3.dil, k1 pad0) -> +skip.
    fn forward(&self, x: &NT) -> crate::tensor::Result<NT> {
        let y = snake_t(x, &self.snake1)?;
        let y = self.conv1.conv(&y, 3 * self.dilation, self.dilation)?;
        let y = snake_t(&y, &self.snake2)?;
        let y = self.conv2.conv(&y, 0, 1)?;
        x.broadcast_add(&y)
    }
}

impl BlockW {
    /// `x [1,c_in,T]` -> `[1,c_out,T.stride]` (K=2.stride, pad=stride/2, no outpad).
    fn forward(&self, x: &NT) -> crate::tensor::Result<NT> {
        let y = snake_t(x, &self.snake1)?;
        let mut z = self.conv_t.conv_t(&y, self.padding, self.stride)?;
        for ru in &self.res {
            z = ru.forward(&z)?;
        }
        Ok(z)
    }
}

// WAV encode/decode moved verbatim to the shared audio-I/O module; re-exported
// under the historical names so the many callers (pipeline, music modes,
// ezaudio, render bins) keep working. `encode_wav_s16le` writes planar `[c.t]`
// f32 -> 16-bit PCM WAV; `decode_wav_s16le` is its inverse.
pub use crate::inference::media::audio_io::{
    read_wav_planar as decode_wav_s16le, write_wav_planar as encode_wav_s16le,
};

/// Load one GGUF tensor -> (f32 values, dims). BF16/F16/F32 all dequantize then
/// cast to F32 (the facade `dequantize` preserves the dense-float dtype, so the
/// explicit `to_dtype(F32)` is required for BF16/F16). Dims are row-major.
fn vae_load_t(
    content: &crate::tensor::quantized::gguf_file::Content,
    file: &mut std::fs::File,
    name: &str,
) -> crate::tensor::Result<(Vec<f32>, Vec<usize>)> {
    use crate::tensor::{DType as CDType, Device as CDevice};
    let info = content
        .tensor_infos
        .get(name)
        .ok_or_else(|| crate::tensor::Error(format!("ace-vae: missing tensor `{name}`")))?;
    let dims = info.shape.dims().to_vec();
    let qt = content.tensor(file, name, &CDevice::Cpu)?;
    let v = qt
        .dequantize(&CDevice::Cpu)?
        .to_dtype(CDType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok((v, dims))
}

/// Load a conv (`is_convt=false`, weight `[Cout,Cin,K]`) or convT (`is_convt=true`,
/// weight `[Cin,Cout,K]`), fusing weight-norm (g per dim0; row-major flatten already
/// matches the kernels' layouts). Bias is optional (decoder.conv2 has none).
fn vae_load_conv(
    content: &crate::tensor::quantized::gguf_file::Content,
    file: &mut std::fs::File,
    prefix: &str,
    is_convt: bool,
    eps: f32,
    dev: &ND,
) -> crate::tensor::Result<ConvW> {
    let (v, vd) = vae_load_t(content, file, &format!("{prefix}.weight_v"))?;
    let (g, _) = vae_load_t(content, file, &format!("{prefix}.weight_g"))?;
    let d0 = g.len();
    let w = fuse_weight_norm(&v, &g, d0, eps);
    let (c_out, c_in, k) = if is_convt {
        (vd[1], vd[0], vd[2])
    } else {
        (vd[0], vd[1], vd[2])
    };
    let b = match vae_load_t(content, file, &format!("{prefix}.bias")) {
        Ok((b, _)) => b,
        Err(_) => vec![0.0; c_out],
    };
    // native conv kernel: conv [c_out,c_in,k]; convT [c_in,c_out,k] (= raw weight_v dims).
    // conv1d weights on CUDA are stored F16 to feed the tensor-core im2col GEMM
    // (F32-accumulate) - halves the weight VRAM + bandwidth; convT weights stay
    // F32 (the convT runs the direct gather kernel, not a cuBLAS GEMM).
    let wt = if is_convt {
        #[cfg(feature = "cuda")]
        if matches!(dev, ND::Cuda(_)) {
            // Rearrange [c_in,c_out,k] -> [k*c_out, c_in] so the convT contraction
            // is a plain GEMM (W_r . x), stored F16 for tensor cores. wr[kk*c_out+oc, ci] = w[ci,oc,kk].
            let mut wr = vec![0f32; k * c_out * c_in];
            for ci in 0..c_in {
                for oc in 0..c_out {
                    for kk in 0..k {
                        wr[(kk * c_out + oc) * c_in + ci] = w[(ci * c_out + oc) * k + kk];
                    }
                }
            }
            vae_nt(wr, (k * c_out, c_in, 1), dev)?.to_dtype(DType::F16)?
        } else {
            vae_nt(w, (c_in, c_out, k), dev)?
        }
        #[cfg(not(feature = "cuda"))]
        vae_nt(w, (c_in, c_out, k), dev)?
    } else {
        let t = vae_nt(w, (c_out, c_in, k), dev)?;
        #[cfg(feature = "cuda")]
        let t = if matches!(dev, ND::Cuda(_)) {
            t.to_dtype(DType::F16)?
        } else {
            t
        };
        t
    };
    let bt = vae_nt(b, (1, c_out, 1), dev)?;
    Ok(ConvW {
        wt,
        bt,
        c_in,
        c_out,
        k,
    })
}

/// Load a Snake param pair -> pre-exp'd (`alpha=exp(α)`, `inv_beta=1/exp(β)`),
/// matching the oracle's vae_load_snake/_inv.
fn vae_load_snake(
    content: &crate::tensor::quantized::gguf_file::Content,
    file: &mut std::fs::File,
    prefix: &str,
    dev: &ND,
) -> crate::tensor::Result<SnakeW> {
    let (a, _) = vae_load_t(content, file, &format!("{prefix}.alpha"))?;
    let (b, _) = vae_load_t(content, file, &format!("{prefix}.beta"))?;
    let c = a.len();
    let alpha: Vec<f32> = a.iter().map(|x| x.exp()).collect();
    let inv_beta: Vec<f32> = b.iter().map(|x| 1.0 / x.exp()).collect();
    Ok(SnakeW {
        alpha: vae_nt(alpha, (1, c, 1), dev)?,
        inv_beta: vae_nt(inv_beta, (1, c, 1), dev)?,
        c,
    })
}

impl OobleckDecoder {
    /// Load the AutoencoderOobleck DECODER from an ACE-Step VAE GGUF
    /// (`vae-BF16.gguf`). Arranges + weight-norm-fuses the 365-tensor inventory
    /// into the decoder's kernel layouts. ⚠️ convT stride/padding are derived from
    /// the known Oobleck ratios [10,6,4,4,2] (pad=stride/2); the EXACT padding +
    /// fusion eps must still be confirmed per-tensor vs `acestep.cpp --dump`
    /// (cmake-blocked) before trusting the output numerically.
    pub fn from_gguf(path: &str, eps: f32) -> crate::tensor::Result<Self> {
        use crate::tensor::quantized::gguf_file;
        let mut file = std::fs::File::open(path)
            .map_err(|e| crate::tensor::Error(format!("ace-vae: open `{path}`: {e}")))?;
        let content = gguf_file::read_mapped_file(&file)?;
        let strides = [10usize, 6, 4, 4, 2];
        // What this decoder occupies, and what one decode window of it costs, both read
        // from the checkpoint's own tensor directory. The typed 500 MB that used to stand
        // in when the file could not be stat'd was a figure nobody measured deciding a
        // real placement, and the 2 GB reserve behind it was the 48 kHz decoder's - handed
        // unchanged to a 24 kHz one that upsamples by a quarter as much.
        let model_size = decoder_resident_bytes(&content);
        let dev = vae_best_device_with_reserve(
            model_size,
            crate::inference::place::audio_demand::oobleck_reserve(
                crate::inference::place::audio_demand::ACE_VAE_REFERENCE_WINDOW,
                &decoder_levels(&content, "decoder.block", &strides),
                crate::inference::place::audio_demand::OOBLECK_KERNEL,
            ),
        );

        let conv1 = vae_load_conv(&content, &mut file, "decoder.conv1", false, eps, &dev)?;
        let dils = [1usize, 3, 9];
        let mut blocks = Vec::with_capacity(5);
        for (i, &stride) in strides.iter().enumerate() {
            let bp = format!("decoder.block.{i}");
            let snake1 = vae_load_snake(&content, &mut file, &format!("{bp}.snake1"), &dev)?;
            let conv_t = vae_load_conv(
                &content,
                &mut file,
                &format!("{bp}.conv_t1"),
                true,
                eps,
                &dev,
            )?;
            let mut res = Vec::with_capacity(3);
            for (r, &dilation) in dils.iter().enumerate() {
                let rp = format!("{bp}.res_unit{}", r + 1);
                res.push(ResUnitW {
                    snake1: vae_load_snake(&content, &mut file, &format!("{rp}.snake1"), &dev)?,
                    conv1: vae_load_conv(
                        &content,
                        &mut file,
                        &format!("{rp}.conv1"),
                        false,
                        eps,
                        &dev,
                    )?,
                    snake2: vae_load_snake(&content, &mut file, &format!("{rp}.snake2"), &dev)?,
                    conv2: vae_load_conv(
                        &content,
                        &mut file,
                        &format!("{rp}.conv2"),
                        false,
                        eps,
                        &dev,
                    )?,
                    dilation,
                });
            }
            blocks.push(BlockW {
                snake1,
                conv_t,
                stride,
                padding: stride / 2,
                res,
            });
        }
        let snake_final = vae_load_snake(&content, &mut file, "decoder.snake1", &dev)?;
        let conv2 = vae_load_conv(&content, &mut file, "decoder.conv2", false, eps, &dev)?;
        crate::inference::serve::progress::placement::note(
            "vae",
            &crate::inference::serve::progress::placement::runs(
                blocks.iter().map(|_| dev.location()),
                model_size / blocks.len().max(1) as u64,
            ),
        );
        Ok(OobleckDecoder {
            conv1,
            blocks,
            snake_final,
            conv2,
            device: dev,
        })
    }

    /// Decode a latent `[c_latent.t_latent]` (channel-major) -> audio `[c_audio.t_audio]`.
    /// `t_audio = t_latent . Πstride` (1920x for the production config). NOTE: untiled  -
    /// production must tile (chunk 1024 / overlap 64) for ~500K-sample widths.
    pub fn decode(
        &self,
        latent: &[f32],
        c_latent: usize,
        t_latent: usize,
    ) -> crate::tensor::Result<(Vec<f32>, usize, usize)> {
        // latent [c_latent.t_latent] channel-major -> [1,c_latent,T] on device.
        let x0 =
            NT::from_vec_f32(latent.to_vec(), (1, c_latent, t_latent))?.to_device(&self.device)?;
        let mut x = self.conv1.conv(&x0, (self.conv1.k - 1) / 2, 1)?;
        for blk in &self.blocks {
            x = blk.forward(&x)?;
        }
        let x = snake_t(&x, &self.snake_final)?;
        let audio = self.conv2.conv(&x, (self.conv2.k - 1) / 2, 1)?; // [1,2,T] channel-major
        let c_out = self.conv2.c_out;
        let v = audio.flatten_all()?.to_vec_f32();
        let t_audio = v.len() / c_out;
        Ok((v, c_out, t_audio))
    }

    /// Memory-bounded decode with OOM protection: tries `chunk`, and on a CUDA
    /// out-of-memory it HALVES the chunk and retries (the conv im2col scales with the
    /// chunk), so the decode fits whatever VRAM is free at runtime - even with the DiT
    /// still resident - instead of crashing. Floors at 16 frames before giving up.
    pub fn decode_chunked(
        &self,
        latent: &[f32],
        c: usize,
        t: usize,
        chunk: usize,
        overlap: usize,
    ) -> crate::tensor::Result<(Vec<f32>, usize, usize)> {
        self.decode_chunked_reporting(latent, c, t, chunk, overlap, None)
    }

    /// [`Self::decode_chunked`], reporting each chunk as it is finished.
    pub fn decode_chunked_reporting(
        &self,
        latent: &[f32],
        c: usize,
        t: usize,
        chunk: usize,
        overlap: usize,
        on_chunk: Option<&dyn Fn(usize, usize) -> crate::tensor::Result<()>>,
    ) -> crate::tensor::Result<(Vec<f32>, usize, usize)> {
        let mut ch = chunk.max(16);
        loop {
            let ov = overlap.min(ch.saturating_sub(1)).max(1);
            match self.decode_chunked_at(latent, c, t, ch, ov, on_chunk) {
                Ok(r) => return Ok(r),
                Err(e) => {
                    let oom = {
                        let s = format!("{e:?}").to_lowercase();
                        s.contains("out of memory") || s.contains("out_of_memory")
                    };
                    if oom && ch > 16 {
                        ch /= 2;
                        eprintln!("[ace-vae] decode OOM - retrying at chunk={ch} (im2col shrinks with the chunk)");
                    } else {
                        return Err(e);
                    }
                }
            }
        }
    }

    /// Single-pass chunked decode (no retry). The Oobleck pipeline upsamples by exactly
    /// `UP`=1920x and (away from the zero-padded signal edges) is translation-equivariant,
    /// so decoding the latent in time-chunks with an `overlap` >= the receptive field and
    /// cropping that overlap reproduces the full decode bit-for-bit while bounding the
    /// per-conv im2col buffers. `chunk`/`overlap` are in latent frames.
    fn decode_chunked_at(
        &self,
        latent: &[f32],
        c: usize,
        t: usize,
        chunk: usize,
        overlap: usize,
        on_chunk: Option<&dyn Fn(usize, usize) -> crate::tensor::Result<()>>,
    ) -> crate::tensor::Result<(Vec<f32>, usize, usize)> {
        // Total upsample ratio = product of the decoder block strides (ACE-Step 48kHz
        // [10,6,4,4,2]=1920; EzAudio 24kHz [10,6,4,2]=480). Derive it from the loaded
        // blocks so the chunk offsets are correct for any Oobleck config.
        let up: usize = self
            .blocks
            .iter()
            .map(|b| b.stride)
            .product::<usize>()
            .max(1);
        if t <= chunk + 2 * overlap {
            return self.decode(latent, c, t);
        }
        let c_out = self.conv2.c_out;
        let total = t * up;
        let mut audio = vec![0f32; c_out * total];
        let mut s = 0usize;
        while s < t {
            // Say how far the decode has got. It already walks the latent in chunks, so the
            // count was there - it just never left the function, and a decode that takes
            // minutes then shows a bare phase name, which reads as a hang.
            if let Some(f) = on_chunk {
                f(s, t)?;
            }
            let e = (s + chunk).min(t);
            let a = s.saturating_sub(overlap);
            let b = (e + overlap).min(t);
            let w = b - a;
            // sub-latent [c, w] channel-major
            let mut sub = vec![0f32; c * w];
            for ch in 0..c {
                for f in 0..w {
                    sub[ch * w + f] = latent[ch * t + (a + f)];
                }
            }
            let (chunk_audio, cc, ct) = self.decode(&sub, c, w)?;
            debug_assert_eq!(cc, c_out);
            debug_assert_eq!(ct, w * up);
            // keep the [s,e) window: it starts at sample (s-a)*up inside this chunk.
            let left = (s - a) * up;
            let keep = (e - s) * up;
            for ch in 0..c_out {
                let dst = ch * total + s * up;
                let src = ch * ct + left;
                audio[dst..dst + keep].copy_from_slice(&chunk_audio[src..src + keep]);
            }
            s = e;
        }
        Ok((audio, c_out, total))
    }
}

/// One Oobleck ENCODER block: 3x ResUnit (at in_ch) -> snake -> strided downsample conv
/// (in_ch -> out_ch, K=2.stride, pad=stride/2). The mirror of [`BlockW`] (which upsamples).
pub struct EncBlockW {
    pub res: Vec<ResUnitW>,
    pub snake1: SnakeW,
    pub conv_down: ConvW,
    pub stride: usize,
    pub padding: usize,
}

impl EncBlockW {
    /// `x [1,c_in,T]` -> `[1,c_out,T/stride]`.
    fn forward(&self, x: &NT) -> crate::tensor::Result<NT> {
        let mut z = x.clone();
        for ru in &self.res {
            z = ru.forward(&z)?;
        }
        let z = snake_t(&z, &self.snake1)?;
        self.conv_down.conv_down(&z, self.padding, self.stride)
    }
}

/// Full Oobleck ENCODER: conv1 -> 5xblock (downsampling) -> snake -> conv2 -> take the 64
/// mean channels of the 128-ch output. The inverse of [`OobleckDecoder`]; downsamples by
/// Πstride (1920x for the 48 kHz config). Runs on CPU (the well-tested F32 conv1d path)  -
/// encode is one-shot, not in a hot loop.
pub struct OobleckEncoder {
    pub conv1: ConvW,
    pub blocks: Vec<EncBlockW>,
    pub snake_final: SnakeW,
    pub conv2: ConvW,
    pub device: ND,
}

impl OobleckEncoder {
    /// Load the AutoencoderOobleck ENCODER from the ACE-Step VAE GGUF (`vae-BF16.gguf`,
    /// the same file the decoder loads). Weights are kept F32 (no F16/im2col) so the
    /// generic strided `conv1d` is used; on CPU for numerical robustness.
    pub fn from_gguf(path: &str, eps: f32) -> crate::tensor::Result<Self> {
        use crate::tensor::quantized::gguf_file;
        let mut file = std::fs::File::open(path)
            .map_err(|e| crate::tensor::Error(format!("ace-vae-enc: open `{path}`: {e}")))?;
        let content = gguf_file::read_mapped_file(&file)?;
        let dev = ND::Cpu;
        let conv1 = enc_load_conv(&content, &mut file, "encoder.conv1", eps, &dev)?;
        let strides = [2usize, 4, 4, 6, 10]; // mirror of the decoder's [10,6,4,4,2]
        let dils = [1usize, 3, 9];
        let mut blocks = Vec::with_capacity(5);
        for (i, &stride) in strides.iter().enumerate() {
            let bp = format!("encoder.block.{i}");
            let mut res = Vec::with_capacity(3);
            for (r, &dilation) in dils.iter().enumerate() {
                let rp = format!("{bp}.res_unit{}", r + 1);
                res.push(ResUnitW {
                    snake1: vae_load_snake(&content, &mut file, &format!("{rp}.snake1"), &dev)?,
                    conv1: enc_load_conv(&content, &mut file, &format!("{rp}.conv1"), eps, &dev)?,
                    snake2: vae_load_snake(&content, &mut file, &format!("{rp}.snake2"), &dev)?,
                    conv2: enc_load_conv(&content, &mut file, &format!("{rp}.conv2"), eps, &dev)?,
                    dilation,
                });
            }
            let snake1 = vae_load_snake(&content, &mut file, &format!("{bp}.snake1"), &dev)?;
            let conv_down = enc_load_conv(&content, &mut file, &format!("{bp}.conv1"), eps, &dev)?;
            blocks.push(EncBlockW {
                res,
                snake1,
                conv_down,
                stride,
                padding: stride / 2,
            });
        }
        let snake_final = vae_load_snake(&content, &mut file, "encoder.snake1", &dev)?;
        let conv2 = enc_load_conv(&content, &mut file, "encoder.conv2", eps, &dev)?;
        Ok(OobleckEncoder {
            conv1,
            blocks,
            snake_final,
            conv2,
            device: dev,
        })
    }

    /// Encode audio `[c_audio.t_audio]` (channel-major, 2ch) -> latent `[64.t_latent]`
    /// (channel-major), `t_latent = t_audio / Πstride`. Returns the 64 mean channels of the
    /// 128-ch output (the deterministic posterior mean; no reparameterization). The latent is
    /// in the SAME raw space the decoder consumes (our pipeline applies no scale/shift), so
    /// `decode(encode(a)) ≈ a`.
    pub fn encode(
        &self,
        audio: &[f32],
        c_audio: usize,
        t_audio: usize,
    ) -> crate::tensor::Result<(Vec<f32>, usize, usize)> {
        let x0 =
            NT::from_vec_f32(audio.to_vec(), (1, c_audio, t_audio))?.to_device(&self.device)?;
        let mut x = self.conv1.conv(&x0, (self.conv1.k - 1) / 2, 1)?;
        for blk in &self.blocks {
            x = blk.forward(&x)?;
        }
        let x = snake_t(&x, &self.snake_final)?;
        let raw = self.conv2.conv(&x, (self.conv2.k - 1) / 2, 1)?; // [1,128,T_latent]
        let c128 = self.conv2.c_out;
        let v = raw.flatten_all()?.to_vec_f32();
        let t_latent = v.len() / c128;
        // channel-major [c128.t_latent] -> take the first 64 channels (the mean).
        let mut latent = vec![0f32; 64 * t_latent];
        latent.copy_from_slice(&v[..64 * t_latent]);
        Ok((latent, 64, t_latent))
    }

    /// Memory-bounded encode: process the audio in time-windows (with `overlap` latent frames
    /// of context cropped off each side) so the per-conv buffers stay bounded for long refs.
    /// `chunk`/`overlap` are in LATENT frames; the mirror of [`OobleckDecoder::decode_chunked`].
    pub fn encode_chunked(
        &self,
        audio: &[f32],
        c_audio: usize,
        t_audio: usize,
        chunk: usize,
        overlap: usize,
    ) -> crate::tensor::Result<(Vec<f32>, usize, usize)> {
        let down: usize = self
            .blocks
            .iter()
            .map(|b| b.stride)
            .product::<usize>()
            .max(1);
        let t_lat = t_audio / down;
        let chunk = chunk.max(16);
        let overlap = overlap.min(chunk.saturating_sub(1)).max(1);
        if t_lat <= chunk + 2 * overlap {
            return self.encode(audio, c_audio, t_audio);
        }
        let mut latent = vec![0f32; 64 * t_lat];
        let mut s = 0usize;
        while s < t_lat {
            let e = (s + chunk).min(t_lat);
            let a = s.saturating_sub(overlap);
            let b = (e + overlap).min(t_lat);
            let (w_lat, a_aud, w_aud) = (b - a, a * down, (b - a) * down);
            let mut sub = vec![0f32; c_audio * w_aud];
            for ch in 0..c_audio {
                for f in 0..w_aud {
                    sub[ch * w_aud + f] = audio[ch * t_audio + (a_aud + f)];
                }
            }
            let (clat, _, ct) = self.encode(&sub, c_audio, w_aud)?;
            debug_assert_eq!(ct, w_lat);
            let (left, keep) = (s - a, e - s);
            for ch in 0..64 {
                let dst = ch * t_lat + s;
                let src = ch * ct + left;
                latent[dst..dst + keep].copy_from_slice(&clat[src..src + keep]);
            }
            s = e;
        }
        Ok((latent, 64, t_lat))
    }
}

/// Load an ENCODER conv (weight `[Cout,Cin,K]`), weight-norm-fused, kept F32 (the generic
/// `conv1d` handles both stride-1 and strided downsample). Bias defaults to zero if absent.
fn enc_load_conv(
    content: &crate::tensor::quantized::gguf_file::Content,
    file: &mut std::fs::File,
    prefix: &str,
    eps: f32,
    dev: &ND,
) -> crate::tensor::Result<ConvW> {
    let (v, vd) = vae_load_t(content, file, &format!("{prefix}.weight_v"))?;
    let (g, _) = vae_load_t(content, file, &format!("{prefix}.weight_g"))?;
    let w = fuse_weight_norm(&v, &g, g.len(), eps);
    let (c_out, c_in, k) = (vd[0], vd[1], vd[2]);
    let b = vae_load_t(content, file, &format!("{prefix}.bias"))
        .map(|(b, _)| b)
        .unwrap_or(vec![0.0; c_out]);
    let wt = vae_nt(w, (c_out, c_in, k), dev)?;
    let bt = vae_nt(b, (1, c_out, 1), dev)?;
    Ok(ConvW {
        wt,
        bt,
        c_in,
        c_out,
        k,
    })
}

#[cfg(test)]
mod tests {
    //! The ignored cases here need real weights, a device, or a reference dump on
    //! this machine; nothing about them is automatic. Run one by name with
    //!   cargo test --release -p loken --lib NAME -- --ignored --nocapture
    use super::*;

    /// Naive direct transposed conv (the reference / oracle): for each input
    /// position and tap, FMA into the output. Independent of the GEMM+col2im path.
    fn conv_transpose1d_naive(
        x: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        c_in: usize,
        c_out: usize,
        t_in: usize,
        k: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
    ) -> (Vec<f32>, usize) {
        let t_out =
            ((t_in.saturating_sub(1)) * stride + k + output_padding).saturating_sub(2 * padding);
        let mut y = vec![0f32; c_out * t_out];
        for ci in 0..c_in {
            for ti in 0..t_in {
                let xv = x[ci * t_in + ti];
                for o in 0..c_out {
                    let w = &weight[(ci * c_out + o) * k..(ci * c_out + o) * k + k];
                    for kk in 0..k {
                        let p = ti * stride + kk;
                        if p < padding {
                            continue;
                        }
                        let to = p - padding;
                        if to >= t_out {
                            continue;
                        }
                        y[o * t_out + to] += xv * w[kk];
                    }
                }
            }
        }
        if let Some(b) = bias {
            for o in 0..c_out {
                for t in 0..t_out {
                    y[o * t_out + t] += b[o];
                }
            }
        }
        (y, t_out)
    }

    fn fill(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 * 0.137 + seed).sin()) * 0.5)
            .collect()
    }

    /// The GEMM+col2im decomposition must match the naive direct conv-transpose
    /// across the Oobleck upsample ratios (stride ∈ {2,4,6,10}), kernels, padding,
    /// output_padding, and GQA-irrelevant channel shapes.
    #[test]
    fn col2im_matches_naive_conv_transpose() {
        // (c_in, c_out, t_in, k, stride, padding, output_padding)
        let cases = [
            (1, 1, 5, 3, 1, 0, 0),
            (2, 3, 7, 4, 2, 1, 0),   // stride-2 upsample with padding
            (4, 2, 6, 8, 4, 3, 0),   // Oobleck-style k=2.stride, pad=stride-1
            (3, 5, 9, 12, 6, 5, 0),  // stride-6
            (2, 2, 8, 20, 10, 9, 0), // stride-10 (the widest Oobleck ratio)
            (3, 3, 5, 4, 2, 1, 1),   // output_padding=1
            (8, 4, 11, 7, 1, 3, 0),  // dilation-1 same-length conv shape
        ];
        for (ci, co, ti, k, s, p, op) in cases {
            let x = fill(ci * ti, 1.0);
            let w = fill(ci * co * k, 2.0);
            let b = fill(co, 3.0);
            let (got, t_out_g) =
                conv_transpose1d_gemm_f32(&x, &w, Some(&b), ci, co, ti, k, s, p, op);
            let (want, t_out_w) = conv_transpose1d_naive(&x, &w, Some(&b), ci, co, ti, k, s, p, op);
            assert_eq!(
                t_out_g,
                t_out_w,
                "t_out mismatch for {:?}",
                (ci, co, ti, k, s, p, op)
            );
            assert_eq!(got.len(), want.len());
            for (i, (a, e)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (a - e).abs() < 1e-4,
                    "mismatch @ {i} for case {:?}: {a} vs {e}",
                    (ci, co, ti, k, s, p, op)
                );
            }
        }
    }

    /// col2im_1d alone (no bias, identity-ish GEMM) preserves total mass under
    /// stride-1, k=1, no padding - a degenerate transposed conv = a channel copy.
    #[test]
    fn col2im_stride1_k1_is_copy() {
        let (c_in, c_out, t_in) = (1usize, 1usize, 6usize);
        let x = fill(c_in * t_in, 7.0);
        let w = vec![1.0f32; c_in * c_out * 1]; // k=1 identity weight
        let (y, t_out) = conv_transpose1d_gemm_f32(&x, &w, None, c_in, c_out, t_in, 1, 1, 0, 0);
        assert_eq!(t_out, t_in);
        for i in 0..t_in {
            assert!(
                (y[i] - x[i]).abs() < 1e-6,
                "copy mismatch @ {i}: {} vs {}",
                y[i],
                x[i]
            );
        }
    }

    /// Snake matches the closed form `x + inv_alpha.sin(alpha.x)²` per channel.
    #[test]
    fn snake_matches_closed_form() {
        let (c, t) = (3usize, 5usize);
        let mut x = fill(c * t, 4.0);
        let x0 = x.clone();
        let alpha = vec![0.7f32, 1.3, 2.0];
        let inv_alpha = vec![1.0f32, 0.5, 1.8];
        snake1d_f32(&mut x, &alpha, &inv_alpha, c, t);
        for ch in 0..c {
            for i in 0..t {
                let v = x0[ch * t + i];
                let s = (alpha[ch] * v).sin();
                let want = v + inv_alpha[ch] * s * s;
                assert!((x[ch * t + i] - want).abs() < 1e-6, "snake @ ({ch},{i})");
            }
        }
    }

    /// Naive reference for conv1d (independent index math) to validate conv1d_f32.
    fn conv1d_naive(
        x: &[f32],
        w: &[f32],
        b: Option<&[f32]>,
        c_in: usize,
        c_out: usize,
        t_in: usize,
        k: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> (Vec<f32>, usize) {
        let span = dilation * (k - 1) + 1;
        let t_out = (t_in + 2 * padding).saturating_sub(span) / stride + 1;
        let mut y = vec![0f32; c_out * t_out];
        for oc in 0..c_out {
            for to in 0..t_out {
                let mut acc = b.map_or(0.0, |bb| bb[oc]);
                for ci in 0..c_in {
                    for kk in 0..k {
                        let ti = (to * stride + kk * dilation) as isize - padding as isize;
                        if ti >= 0 && (ti as usize) < t_in {
                            acc += x[ci * t_in + ti as usize] * w[(oc * c_in + ci) * k + kk];
                        }
                    }
                }
                y[oc * t_out + to] = acc;
            }
        }
        (y, t_out)
    }

    /// PROBE (ignored; needs the real GGUF on disk). Resolves the substrate's
    /// GGUF dim ordering (ggml-ne vs row-major) for conv/convT weights and
    /// confirms the decoder tensor inventory + that BF16 dequantizes finite  -
    /// the facts the from_gguf arranger needs. Run:
    ///   cargo test --release --lib inference::model::acestep::vae::tests::probe_vae_gguf -- --ignored --nocapture
    #[test]
    #[ignore]
    fn probe_vae_gguf() {
        use crate::tensor::quantized::gguf_file;
        use crate::tensor::Device;
        let path = crate::inference::model::acestep::fsq::acestep_gguf("vae-BF16.gguf");
        let mut file = std::fs::File::open(path).expect("open vae gguf");
        let content = gguf_file::read_mapped_file(&file).expect("read gguf");
        println!("tensor count: {}", content.tensor_infos.len());
        for name in [
            "decoder.conv1.weight_v",                 // ggml ne [7,64,2048]
            "decoder.conv1.weight_g",                 // ggml ne [1,1,2048]
            "decoder.block.0.conv_t1.weight_v",       // ggml ne [20,1024,2048]
            "decoder.block.0.res_unit1.snake1.alpha", // ggml ne [1,1024,1]
            "decoder.conv2.weight_v",                 // ggml ne [7,128,2]
        ] {
            let info = content
                .tensor_infos
                .get(name)
                .unwrap_or_else(|| panic!("missing {name}"));
            let dims = info.shape.dims().to_vec();
            match content.tensor(&mut file, name, &Device::Cpu) {
                Ok(qt) => {
                    let dq = qt
                        .dequantize(&Device::Cpu)
                        .and_then(|t| t.to_dtype(crate::tensor::DType::F32));
                    match dq
                        .and_then(|t| t.flatten_all())
                        .and_then(|t| t.to_vec1::<f32>())
                    {
                        Ok(v) => {
                            let finite = v.iter().all(|x| x.is_finite());
                            println!("{name:42} infodims={dims:?} qtshape={:?} elems={} finite={finite} first={:.4}",
                                qt.shape().dims(), v.len(), v[0]);
                        }
                        Err(e) => println!("{name:42} infodims={dims:?} DEQUANT ERR: {e}"),
                    }
                }
                Err(e) => println!("{name:42} infodims={dims:?} TENSOR-READ ERR: {e}"),
            }
        }
        // Confirm the full decoder inventory is present.
        let mut missing = vec![];
        for b in 0..5 {
            for t in [
                "conv_t1.weight_v",
                "conv_t1.weight_g",
                "conv_t1.bias",
                "snake1.alpha",
                "snake1.beta",
            ] {
                let n = format!("decoder.block.{b}.{t}");
                if !content.tensor_infos.contains_key(&n) {
                    missing.push(n);
                }
            }
            for r in 1..=3 {
                for t in [
                    "conv1.weight_v",
                    "conv1.weight_g",
                    "conv1.bias",
                    "conv2.weight_v",
                    "snake1.alpha",
                    "snake1.beta",
                    "snake2.alpha",
                    "snake2.beta",
                ] {
                    let n = format!("decoder.block.{b}.res_unit{r}.{t}");
                    if !content.tensor_infos.contains_key(&n) {
                        missing.push(n);
                    }
                }
            }
        }
        println!("decoder inventory missing: {:?}", missing);
        assert!(missing.is_empty(), "missing decoder tensors: {missing:?}");
    }

    /// INTEGRATION (ignored; needs the real GGUF). Loads the actual Oobleck
    /// decoder from vae-BF16.gguf and decodes a latent -> asserts finite stereo
    /// audio of the exact 1920x length. Validates the loader + decoder run on real
    /// weights end-to-end. ⚠️ NOT a numerical-vs-oracle check (convT padding / eps
    /// pending acestep.cpp --dump). Run:
    ///   cargo test --release --lib inference::model::acestep::vae::tests::load_and_decode_real_vae -- --ignored --nocapture
    #[test]
    #[ignore]
    fn load_and_decode_real_vae() {
        let dec = OobleckDecoder::from_gguf(
            crate::inference::model::acestep::fsq::acestep_gguf("vae-BF16.gguf")
                .to_str()
                .unwrap(),
            1e-12,
        )
        .expect("load vae gguf");
        // Sanity on the loaded geometry.
        assert_eq!(dec.conv1.c_in, 64, "latent channels");
        assert_eq!(dec.conv1.c_out, 2048);
        assert_eq!(dec.blocks.len(), 5);
        assert_eq!(dec.conv2.c_out, 2, "stereo");
        let t_latent = 6usize;
        let latent = fill(64 * t_latent, 0.2);
        let (audio, c_audio, t_audio) = dec.decode(&latent, 64, t_latent).unwrap();
        assert_eq!(c_audio, 2);
        assert_eq!(t_audio, t_latent * 1920, "1920x upsample");
        assert_eq!(audio.len(), 2 * t_audio);
        let finite = audio.iter().all(|x| x.is_finite());
        let peak = audio.iter().fold(0f32, |m, &x| m.max(x.abs()));
        println!(
            "decoded {t_latent} latent -> {t_audio} samples/ch, finite={finite} peak={peak:.4}"
        );
        assert!(finite, "non-finite audio");
    }

    /// Encoder geometry + VAE round-trip: a structured latent -> decode -> encode -> decode.
    /// Asserts the encoder mirrors the decoder (2->128 in, 128 out, 5 blocks, 1920x down)
    /// and that `encode∘decode ≈ identity` on the latent manifold (audio cosine high).
    #[test]
    #[ignore]
    fn encode_decode_roundtrip() {
        let p = crate::inference::model::acestep::fsq::acestep_gguf("vae-BF16.gguf");
        let p = p.to_str().unwrap();
        let dec = OobleckDecoder::from_gguf(p, 1e-12).expect("load vae decoder");
        let enc = OobleckEncoder::from_gguf(p, 1e-12).expect("load vae encoder");
        assert_eq!(enc.conv1.c_in, 2, "stereo in");
        assert_eq!(enc.conv1.c_out, 128);
        assert_eq!(enc.blocks.len(), 5);
        assert_eq!(enc.conv2.c_out, 128, "128 = 64 mean + 64 scale");
        let t_lat = 16usize;
        let latent: Vec<f32> = (0..64 * t_lat)
            .map(|i| ((i as f32) * 0.137).sin() * 0.3)
            .collect();
        let (audio, c, t_aud) = dec.decode(&latent, 64, t_lat).unwrap();
        assert_eq!(c, 2);
        let (lat2, cl, tl2) = enc.encode(&audio, 2, t_aud).unwrap();
        assert_eq!(cl, 64, "64 mean channels");
        assert_eq!(tl2, t_lat, "1920x downsample (t_aud/1920)");
        assert!(lat2.iter().all(|x| x.is_finite()), "non-finite latent");
        let (audio2, _, t_aud2) = dec.decode(&lat2, 64, tl2).unwrap();
        assert_eq!(t_aud2, t_aud);
        let cos = |a: &[f32], b: &[f32]| {
            let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
            for i in 0..a.len() {
                d += a[i] as f64 * b[i] as f64;
                na += (a[i] as f64).powi(2);
                nb += (b[i] as f64).powi(2);
            }
            d / (na.sqrt() * nb.sqrt() + 1e-12)
        };
        let ca = cos(&audio, &audio2);
        let cl_ = cos(&latent, &lat2);
        println!("encoder geometry OK; latent2 [{cl},{tl2}]; latent cosine={cl_:.4}; audio round-trip cosine={ca:.4}");
        assert!(
            ca > 0.7,
            "audio round-trip cosine too low: {ca:.4} (encoder likely wrong)"
        );
    }

    /// `decode_chunked` must reproduce the full `decode` away from the chunk seams.
    /// Compares full vs chunked on a length that fits, across overlaps, to confirm the
    /// overlap covers the receptive field (interior samples must match to ~f32 eps).
    #[test]
    #[ignore]
    fn chunked_decode_matches_full() {
        let dec = OobleckDecoder::from_gguf(
            crate::inference::model::acestep::fsq::acestep_gguf("vae-BF16.gguf")
                .to_str()
                .unwrap(),
            1e-12,
        )
        .expect("load vae gguf");
        let t = 192usize;
        let latent = fill(64 * t, 0.3);
        let (full, c, ts) = dec.decode(&latent, 64, t).unwrap();
        for overlap in [16usize, 32, 48, 64] {
            let (ch, cc, tc) = dec.decode_chunked(&latent, 64, t, 48, overlap).unwrap();
            assert_eq!((cc, tc), (c, ts), "geometry");
            let max_d = full
                .iter()
                .zip(&ch)
                .fold(0f32, |m, (&a, &b)| m.max((a - b).abs()));
            println!("overlap={overlap}: max|full-chunked|={max_d:.3e}");
            if overlap >= 48 {
                assert!(
                    max_d < 1e-3,
                    "overlap {overlap} insufficient (seam diff {max_d:.3e})"
                );
            }
        }
    }

    /// Load an acestep.cpp `--dump` .bin: `int32 ndims, int32 shape[ndims],
    /// f32 data`. Returns (data, shape).
    #[cfg(test)]
    fn load_dump(path: &str) -> (Vec<f32>, Vec<usize>) {
        let b = std::fs::read(path).unwrap_or_else(|_| panic!("read {path}"));
        let nd = i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
        let mut shape = Vec::with_capacity(nd);
        for i in 0..nd {
            let o = 4 + i * 4;
            shape.push(i32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize);
        }
        let off = 4 + nd * 4;
        let data: Vec<f32> = b[off..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        (data, shape)
    }

    /// NUMERICAL VALIDATION vs the oracle (ignored; needs acestep.cpp --dump in
    /// /tmp/acedump). Loads the oracle's DiT-output latent, runs MY Oobleck decode,
    /// and compares to the oracle's vae_audio - the ground-truth test that the
    /// convT padding, weight-norm fusion, snake, and conv kernels are all correct.
    /// Run: cargo test --release --lib inference::model::acestep::vae::tests::validate_vae_vs_oracle -- --ignored --nocapture
    #[test]
    #[ignore]
    fn validate_vae_vs_oracle() {
        // DiT output = the latent fed to the VAE, dumped [T, 64] (T-major).
        let (lat_tmajor, lshape) = load_dump("/tmp/acedump/dit_output.bin");
        assert_eq!(lshape.len(), 2);
        let (t_lat, c_lat) = (lshape[0], lshape[1]); // [320, 64]
        assert_eq!(c_lat, 64);
        // transpose [T,64] -> channel-major [64,T] for our decode.
        let mut latent = vec![0f32; c_lat * t_lat];
        for ti in 0..t_lat {
            for c in 0..c_lat {
                latent[c * t_lat + ti] = lat_tmajor[ti * c_lat + c];
            }
        }
        let dec = OobleckDecoder::from_gguf(
            crate::inference::model::acestep::fsq::acestep_gguf("vae-BF16.gguf")
                .to_str()
                .unwrap(),
            1e-12,
        )
        .expect("load vae");
        let (audio, c_audio, t_audio) = dec.decode(&latent, c_lat, t_lat).unwrap();
        let (oracle, oshape) = load_dump("/tmp/acedump/vae_audio.bin"); // [2, T_audio]
        println!("mine: c={c_audio} t={t_audio}; oracle shape={oshape:?}");
        assert_eq!(c_audio, oshape[0]);
        assert_eq!(t_audio, oshape[1], "audio length");
        // cosine similarity + max abs diff over both channels.
        let (mut dot, mut na, mut nb, mut maxd) = (0f64, 0f64, 0f64, 0f32);
        for (a, o) in audio.iter().zip(&oracle) {
            dot += (*a as f64) * (*o as f64);
            na += (*a as f64).powi(2);
            nb += (*o as f64).powi(2);
            maxd = maxd.max((a - o).abs());
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        println!(
            "VAE vs oracle: cosine={cos:.6} max_abs_diff={maxd:.5} (oracle peak {:.4})",
            oracle.iter().fold(0f32, |m, &x| m.max(x.abs()))
        );
        assert!(cos > 0.99, "VAE decode diverges from oracle: cosine {cos}");
    }

    /// PRODUCE AUDIO (ignored; needs the GGUF + /tmp/acedump): decode the oracle's
    /// DiT-output latent through MY validated Oobleck VAE and write a 48kHz stereo
    /// WAV - the first real audio produced end-to-end by the native pipeline.
    /// Run: cargo test --release --lib inference::model::acestep::vae::tests::produce_wav -- --ignored --nocapture
    #[test]
    #[ignore]
    fn produce_wav() {
        let (lat_tmajor, ls) = load_dump("/tmp/acedump/dit_output.bin"); // [T,64]
        let (t_lat, c) = (ls[0], ls[1]);
        let mut latent = vec![0f32; c * t_lat];
        for ti in 0..t_lat {
            for ch in 0..c {
                latent[ch * t_lat + ti] = lat_tmajor[ti * c + ch];
            }
        }
        let dec = OobleckDecoder::from_gguf(
            crate::inference::model::acestep::fsq::acestep_gguf("vae-BF16.gguf")
                .to_str()
                .unwrap(),
            1e-12,
        )
        .unwrap();
        let (audio, c_audio, t_audio) = dec.decode(&latent, c, t_lat).unwrap();
        let wav = encode_wav_s16le(&audio, c_audio, t_audio, 48000);
        // Written into the repository's ignored results/ directory and nowhere else: a
        // test may leave an artefact behind, but not outside the tree it was run from.
        // Anchored to the crate directory, since a test's working directory is not it.
        let dir = format!("{}/results/acestep", env!("CARGO_MANIFEST_DIR"));
        std::fs::create_dir_all(&dir).ok();
        let path = format!("{dir}/sample.wav");
        std::fs::write(&path, &wav).unwrap();
        let peak = audio.iter().fold(0f32, |m, &x| m.max(x.abs()));
        println!(
            "WROTE {path}: {} bytes, {:.1}s @48kHz stereo, peak {:.3}",
            wav.len(),
            t_audio as f32 / 48000.0,
            peak
        );
        assert_eq!(&wav[0..4], b"RIFF");
        assert!(peak > 0.001 && peak <= 1.0, "sane audio level");
    }

    /// WAV encode is a correct 44-byte-header PCM16 stream and round-trips the
    /// samples within the s16 quantization step (interleaving + scale preserved).
    #[test]
    fn wav_s16_roundtrip() {
        let (c, t, sr) = (2usize, 5usize, 48000u32);
        // distinct per-channel ramps so interleaving order is checkable.
        let mut planar = vec![0f32; c * t];
        for ti in 0..t {
            planar[ti] = (ti as f32 / t as f32) * 0.8 - 0.4; // ch0
            planar[t + ti] = 0.5 - (ti as f32 / t as f32) * 0.6; // ch1
        }
        let wav = encode_wav_s16le(&planar, c, t, sr);
        // header checks
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), c as u16, "channels");
        assert_eq!(
            u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
            sr,
            "sample rate"
        );
        assert_eq!(u16::from_le_bytes([wav[34], wav[35]]), 16, "bits");
        assert_eq!(&wav[36..40], b"data");
        let data_len = u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]) as usize;
        assert_eq!(data_len, c * t * 2);
        assert_eq!(wav.len(), 44 + data_len);
        // round-trip the samples (interleaved s16 -> f32) within 1 LSB.
        for ti in 0..t {
            for ci in 0..c {
                let off = 44 + (ti * c + ci) * 2;
                let s = i16::from_le_bytes([wav[off], wav[off + 1]]) as f32 / 32767.0;
                let want = planar[ci * t + ti];
                assert!(
                    (s - want).abs() < 1.0 / 32767.0 + 1e-6,
                    "wav rt @ ({ci},{ti}): {s} vs {want}"
                );
            }
        }
    }

    fn snake_w(c: usize, alpha: f32) -> SnakeW {
        SnakeW {
            alpha: vae_nt(vec![alpha; c], (1, c, 1), &ND::Cpu).unwrap(),
            inv_beta: vae_nt(vec![1.0; c], (1, c, 1), &ND::Cpu).unwrap(),
            c,
        }
    }
    fn conv_w(c_in: usize, c_out: usize, k: usize, seed: f32) -> ConvW {
        ConvW {
            wt: vae_nt(fill(c_out * c_in * k, seed), (c_out, c_in, k), &ND::Cpu).unwrap(),
            bt: vae_nt(fill(c_out, seed + 1.0), (1, c_out, 1), &ND::Cpu).unwrap(),
            c_in,
            c_out,
            k,
        }
    }
    /// Transposed-conv test weight: the native `conv_transpose1d` kernel (and the
    /// production `vae_load_conv(is_convt=true)` path) take `[c_in, c_out, k]` - the
    /// raw weight_v layout - unlike regular convs' `[c_out, c_in, k]`.
    fn conv_tw(c_in: usize, c_out: usize, k: usize, seed: f32) -> ConvW {
        ConvW {
            wt: vae_nt(fill(c_in * c_out * k, seed), (c_in, c_out, k), &ND::Cpu).unwrap(),
            bt: vae_nt(fill(c_out, seed + 1.0), (1, c_out, 1), &ND::Cpu).unwrap(),
            c_in,
            c_out,
            k,
        }
    }
    fn conv_zero(c_in: usize, c_out: usize, k: usize) -> ConvW {
        ConvW {
            wt: vae_nt(vec![0.0; c_out * c_in * k], (c_out, c_in, k), &ND::Cpu).unwrap(),
            bt: vae_nt(vec![0.0; c_out], (1, c_out, 1), &ND::Cpu).unwrap(),
            c_in,
            c_out,
            k,
        }
    }

    /// ResUnit with zero convs + α=0 snakes is the identity (output == skip): the
    /// snake reduces to x (sin(0)=0), the convs to 0, so x + 0 = x. Pins the skip
    /// connection + length-preservation wiring.
    #[test]
    fn res_unit_zero_is_identity() {
        let (ch, t, dil) = (3usize, 9usize, 3usize);
        let ru = ResUnitW {
            snake1: snake_w(ch, 0.0),
            conv1: conv_zero(ch, ch, 7),
            snake2: snake_w(ch, 0.0),
            conv2: conv_zero(ch, ch, 1),
            dilation: dil,
        };
        let xv = fill(ch * t, 5.0);
        let x = NT::from_vec_f32(xv.clone(), (1, ch, t)).unwrap();
        let y = ru.forward(&x).unwrap().flatten_all().unwrap().to_vec_f32();
        assert_eq!(y.len(), xv.len());
        for (a, e) in y.iter().zip(&xv) {
            assert!((a - e).abs() < 1e-6, "resunit identity broke: {a} vs {e}");
        }
    }

    /// Decoder assembly produces the right channels and the exact 1920x-style
    /// upsample length (t_audio = t_latent . Πstride), validating the block channel
    /// flow + per-block T.stride growth end-to-end on a tiny config.
    #[test]
    fn decoder_upsample_shape() {
        // tiny: latent 4ch -> conv1 4->8, two blocks (stride 2 then 3), -> conv2 ->2.
        let resunits = |ch: usize| {
            [1usize, 3, 9]
                .iter()
                .map(|&d| ResUnitW {
                    snake1: snake_w(ch, 0.3),
                    conv1: conv_w(ch, ch, 7, 0.2),
                    snake2: snake_w(ch, 0.3),
                    conv2: conv_w(ch, ch, 1, 0.1),
                    dilation: d,
                })
                .collect::<Vec<_>>()
        };
        let dec = OobleckDecoder {
            conv1: conv_w(4, 8, 7, 0.5),
            blocks: vec![
                BlockW {
                    snake1: snake_w(8, 0.3),
                    conv_t: conv_tw(8, 6, 4, 0.4),
                    stride: 2,
                    padding: 1,
                    res: resunits(6),
                },
                BlockW {
                    snake1: snake_w(6, 0.3),
                    conv_t: conv_tw(6, 5, 6, 0.4),
                    stride: 3,
                    padding: 0,
                    res: resunits(5),
                },
            ],
            snake_final: snake_w(5, 0.3),
            conv2: conv_w(5, 2, 7, 0.5),
            device: ND::Cpu,
        };
        let t_latent = 4usize;
        let (audio, c_audio, t_audio) = dec.decode(&fill(4 * t_latent, 1.0), 4, t_latent).unwrap();
        assert_eq!(c_audio, 2, "stereo output channels");
        // block0 stride2 pad1 K4: T.2 ; block1 stride3 pad0 K6: T_out=(T-1).3+6 (pad0).
        // verify via the kernel's own formula rather than hard-coding.
        let t0 = ((t_latent - 1) * 2 + 4) - 2 * 1; // = t_latent*2
        let t1 = (t0 - 1) * 3 + 6 - 0; // pad0
        assert_eq!(t_audio, t1, "upsample length mismatch: {t_audio} vs {t1}");
        assert_eq!(audio.len(), c_audio * t_audio);
        assert!(
            audio.iter().all(|x| x.is_finite()),
            "non-finite audio sample"
        );
    }

    /// Weight-norm fusion: (a) elementwise = g.v/‖v‖ per dim0 row; (b) when
    /// g equals each row's L2 norm, the fused weight is exactly v (the param is
    /// initialized that way so the fused weight starts equal to the direction).
    #[test]
    fn weight_norm_fuse_correct() {
        let (d0, rest) = (4usize, 5usize);
        let v = fill(d0 * rest, 2.0);
        let g = vec![0.5f32, 1.0, 2.0, 3.0];
        let w = fuse_weight_norm(&v, &g, d0, 0.0);
        for i in 0..d0 {
            let row = &v[i * rest..(i + 1) * rest];
            let nrm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            for (j, &vv) in row.iter().enumerate() {
                let want = g[i] * vv / nrm;
                assert!((w[i * rest + j] - want).abs() < 1e-5, "fuse @ ({i},{j})");
            }
        }
        // g = ‖v_row‖ ⟹ fused == v.
        let gn: Vec<f32> = (0..d0)
            .map(|i| {
                v[i * rest..(i + 1) * rest]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt()
            })
            .collect();
        let w2 = fuse_weight_norm(&v, &gn, d0, 0.0);
        for (a, e) in w2.iter().zip(&v) {
            assert!((a - e).abs() < 1e-5, "g=‖v‖ should give v: {a} vs {e}");
        }
    }

    /// conv1d_f32 must match the naive reference across the VAE's conv shapes:
    /// k=7 dilated (ResUnit), k=1 pointwise, k=7 stride-1 same-length (decoder
    /// conv1/conv2), with the matching `pad = (k-1).dil/2` "same" padding.
    #[test]
    fn conv1d_matches_naive() {
        // (c_in, c_out, t_in, k, stride, padding, dilation)
        let cases = [
            (2, 3, 9, 7, 1, 3, 1),   // k7 same-length (decoder conv1/conv2)
            (4, 4, 11, 7, 1, 9, 3),  // k7 dilation-3 (ResUnit dilated conv, pad=3.dil)
            (5, 2, 8, 1, 1, 0, 1),   // k1 pointwise
            (3, 3, 7, 3, 1, 1, 1),   // k3 same
            (2, 2, 10, 7, 1, 12, 5), // k7 dilation-5 (ResUnit, pad=3.5=15? here 12 -> crops)
        ];
        for (ci, co, ti, k, s, p, d) in cases {
            let x = fill(ci * ti, 1.5);
            let w = fill(co * ci * k, 2.5);
            let b = fill(co, 0.3);
            let (got, tg) = conv1d_f32(&x, &w, Some(&b), ci, co, ti, k, s, p, d);
            let (want, tw) = conv1d_naive(&x, &w, Some(&b), ci, co, ti, k, s, p, d);
            assert_eq!(tg, tw, "t_out mismatch {:?}", (ci, co, ti, k, s, p, d));
            for (i, (a, e)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (a - e).abs() < 1e-4,
                    "conv1d @ {i} case {:?}: {a} vs {e}",
                    (ci, co, ti, k, s, p, d)
                );
            }
        }
    }
}
