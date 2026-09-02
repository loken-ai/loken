//! ACE-Step 1.5 - turbo DiT support ( M1, build-order step 4).
//!
//! The DiT is a Qwen3-shaped transformer (24 blocks, hidden 2048, 16 heads / 8 kv
//! GQA, head_dim 128, ffn 6144 SwiGLU, rope 1e6, alternating SWA(128)/full,
//! qk-norm) that predicts a flow-matching velocity field, plus AdaLN-single
//! timestep modulation and cross-attention to the text/cond encoding. Most of the
//! per-block math reuses existing kernels (the generic Qwen3 attention + flux DiT
//! pattern); the net-new scaffolding ported here is the turbo flow-match sampler
//! (8-step Euler) and the AdaLN modulation, both small pure functions validated in
//! isolation. The attention/MLP assembly + GGUF loader follow (per-tensor parity
//! vs `acestep.cpp --dump`, oracle-gated).
//!
//! Config (from acestep-v15-turbo-Q8_0.gguf): is_turbo ⟹ shift=3.0, 8 steps,
//! in_channels 192 (latent|src|mask), patch_size 2, text_hidden_dim 1024 (cross-
//! attn cond), fsq_input_levels [8,8,8,5,5,5].

/// Build the flow-matching turbo timestep schedule (oracle `ops_build_schedule`):
/// `t_i = shift.t/(1+(shift-1).t)` with `t = 1 - i/num_steps`, for `i in
/// 0..num_steps`. Turbo uses `num_steps=8, shift=3.0`. The implicit final `t=0`
/// endpoint is handled by the integrator (it is NOT in the returned schedule).
pub fn turbo_schedule(num_steps: usize, shift: f32) -> Vec<f32> {
    (0..num_steps)
        .map(|i| {
            let t = 1.0 - (i as f32) / (num_steps as f32);
            shift * t / (1.0 + (shift - 1.0) * t)
        })
        .collect()
}

/// Patchify reshape (oracle `proj_in`: `reshape(latent, in_ch.P, S)`): group `P`
/// consecutive latent frames into one token. `latent`: `[in_ch.T]` (channel-major),
/// returns `(patched[(in_ch.P).S], S=T/P)` with
/// `patched[(p.in_ch + c).S + s] = latent[c.T + (P.s + p)]`. The DiT then applies
/// `proj_in` linear `[in_ch.P -> hidden]` (= conv1d_f32 k=1) to `patched`.
pub fn patchify(latent: &[f32], in_ch: usize, t: usize, patch: usize) -> (Vec<f32>, usize) {
    debug_assert_eq!(latent.len(), in_ch * t);
    let s = t / patch;
    let mut out = vec![0f32; in_ch * patch * s];
    for p in 0..patch {
        for c in 0..in_ch {
            for si in 0..s {
                out[(p * in_ch + c) * s + si] = latent[c * t + (patch * si + p)];
            }
        }
    }
    (out, s)
}

/// Unpatchify reshape (oracle `proj_out`, after its linear `[hidden -> out_ch.P]`):
/// the exact inverse of [`patchify`]. `y`: `[(out_ch.P).S]`, returns
/// `(out[out_ch.T], T=S.P)` with `out[c.T + (P.s + p)] = y[(p.out_ch + c).S + s]`.
pub fn unpatchify(y: &[f32], out_ch: usize, s: usize, patch: usize) -> (Vec<f32>, usize) {
    debug_assert_eq!(y.len(), out_ch * patch * s);
    let t = s * patch;
    let mut out = vec![0f32; out_ch * t];
    for p in 0..patch {
        for c in 0..out_ch {
            for si in 0..s {
                out[c * t + (patch * si + p)] = y[(p * out_ch + c) * s + si];
            }
        }
    }
    (out, t)
}

/// Sinusoidal timestep embedding (ggml `ggml_timestep_embedding` convention, the
/// oracle's `dit_ggml_build_temb`): `half=dim/2`, `freq_j=exp(-ln(max_period).j/half)`,
/// `embed[j]=cos(t.freq_j)`, `embed[j+half]=sin(t.freq_j)` (an odd `dim` zero-pads
/// the last lane). The DiT scales the timestep by 1000 first (diffusion convention)
/// and uses `dim=256, max_period=10000`, then linear_1->silu->linear_2 (GGUF weights,
/// applied by the loader) -> time_proj -> 6-way AdaLN.
pub fn timestep_embedding(t: f32, dim: usize, max_period: f32) -> Vec<f32> {
    let half = dim / 2;
    let mut e = vec![0f32; dim];
    let lnmp = max_period.ln();
    for j in 0..half {
        let freq = (-lnmp * (j as f32) / (half as f32)).exp();
        let arg = t * freq;
        e[j] = arg.cos();
        e[j + half] = arg.sin();
    }
    e
}

/// Scale a raw flow-match timestep by the diffusion 1000x convention before the
/// sinusoidal embedding (oracle `ggml_scale(t, 1000)`).
pub fn dit_timestep_embedding(t: f32) -> Vec<f32> {
    timestep_embedding(t * 1000.0, 256, 10000.0)
}

/// AdaLN-single modulation: `out = x_norm . (1 + scale) + shift`, per channel
/// (the scale/shift broadcast over the token/time axis). `x`: `[C.T]` (C outer),
/// `scale`/`shift`: `[C]`. Matches the DiT's scale_shift_table + time_proj split.
pub fn adaln_modulate(x: &[f32], scale: &[f32], shift: &[f32], c: usize, t: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), c * t);
    debug_assert_eq!(scale.len(), c);
    debug_assert_eq!(shift.len(), c);
    let mut out = vec![0f32; c * t];
    for ch in 0..c {
        let (sc, sh) = (1.0 + scale[ch], shift[ch]);
        for ti in 0..t {
            out[ch * t + ti] = x[ch * t + ti] * sc + sh;
        }
    }
    out
}

/// The six AdaLN-single modulation vectors for one DiT block (each `[hidden]`),
/// in the oracle's order. Apply as: SA = `gate_sa ⊙ attn(modulate(norm(x),
/// scale_sa, shift_sa))`; MLP = `gate_mlp ⊙ swiglu(modulate(norm(x), scale_mlp,
/// shift_mlp))`.
pub struct AdalnSix {
    pub shift_sa: Vec<f32>,
    pub scale_sa: Vec<f32>,
    pub gate_sa: Vec<f32>,
    pub shift_mlp: Vec<f32>,
    pub scale_mlp: Vec<f32>,
    pub gate_mlp: Vec<f32>,
}

/// Build the six AdaLN vectors (oracle `dit-graph.h`): `adaln[6H] =
/// scale_shift_table[6H] + tproj[6H]` (elementwise), then split into six `[H]`
/// chunks ordered `[shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp]`.
/// `tproj` = the per-step `time_proj` output (shared across layers); `table` = the
/// per-layer `scale_shift_table` (GGUF [6,H], row-major flatten = the 6 chunks).
pub fn adaln_split(tproj: &[f32], table: &[f32], hidden: usize) -> AdalnSix {
    debug_assert_eq!(tproj.len(), 6 * hidden);
    debug_assert_eq!(table.len(), 6 * hidden);
    let chunk = |i: usize| -> Vec<f32> {
        (0..hidden)
            .map(|h| table[i * hidden + h] + tproj[i * hidden + h])
            .collect()
    };
    AdalnSix {
        shift_sa: chunk(0),
        scale_sa: chunk(1),
        gate_sa: chunk(2),
        shift_mlp: chunk(3),
        scale_mlp: chunk(4),
        gate_mlp: chunk(5),
    }
}

/// Apply an AdaLN gate to a block's output before the residual add: `out = gate .
/// y`, per channel. Used as `x = x + gate ⊙ sublayer(modulate(norm(x)))`.
pub fn adaln_gate(y: &[f32], gate: &[f32], c: usize, t: usize) -> Vec<f32> {
    debug_assert_eq!(y.len(), c * t);
    debug_assert_eq!(gate.len(), c);
    let mut out = vec![0f32; c * t];
    for ch in 0..c {
        let g = gate[ch];
        for ti in 0..t {
            out[ch * t + ti] = y[ch * t + ti] * g;
        }
    }
    out
}

// -- DiT weight structs + GGUF loader -----------------------------------------
// Weights held as dequantized f32 Tensors (Q8_0/BF16 -> F32 at load), in the
// substrate's row-major layout: Linear weight `[out, in]` (apply as `x.matmul(w.t())`).
use crate::inference::model::acestep::fsq::in_heads;
use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan};
use crate::tensor::layer::qlinear::{QLinear, Weight};
use crate::tensor::lora::LoraDelta;
use crate::tensor::{Device, Tensor};

/// A weight + optional bias, kept dequantized to F32, for the small conv-shaped tensors:
/// proj_in/proj_out, cond_emb, time_embed.
///
/// NOT a projection, despite the name it is applied under. `proj_in.weight` is `[hidden,
/// in_ch, patch]` - a three-axis kernel that the patchify path reshapes itself - so this
/// cannot be the shared `Linear`, whose contract is a two-dimensional matrix and whose
/// forward would have to guess which axes to contract.
pub struct DitLinear {
    pub w: Tensor,
    pub b: Option<Tensor>,
}

/// A quantized Linear `[out,in]`: the GGUF blocks stay quantized on the weight's
/// device and dequantize on-the-fly in the matmul kernel (the generic hetero
/// path, like the LM/Flux DiT). Used for the per-block attention/FFN linears  -
/// the model's parameter mass - so a 4B-class DiT keeps its compact (≈Q8) device
/// footprint instead of a 4x F32 blow-up. Bias is `None` for these (bias=false),
/// ACE-Step publishes its adapters as `a [rank, in]` and `b [out, rank]`, applied with
/// `matmul_t`. The shared projection holds them the other way round - `down [in, rank]` and
/// `up [rank, out]`, multiplied directly - so they are transposed once, when they are
/// attached, rather than on every forward.
fn acestep_delta(a: &Tensor, b: &Tensor, scale: f32) -> crate::tensor::Result<LoraDelta> {
    Ok(LoraDelta {
        down: a.transpose(0, 1)?.contiguous()?,
        up: b.transpose(0, 1)?.contiguous()?,
        scale,
    })
}

/// One DiT decoder block (Qwen3-shaped self-attn + cross-attn + SwiGLU, AdaLN-single).
pub struct DitLayer {
    pub self_attn_norm: Tensor, // rms weight [H]
    pub cross_attn_norm: Tensor,
    pub mlp_norm: Tensor,
    pub sa_q: QLinear,
    pub sa_k: QLinear,
    pub sa_v: QLinear,
    pub sa_o: QLinear,
    pub sa_q_norm: Tensor,
    pub sa_k_norm: Tensor, // qk-norm [head_dim]
    pub ca_q: QLinear,
    pub ca_k: QLinear,
    pub ca_v: QLinear,
    pub ca_o: QLinear,
    pub ca_q_norm: Tensor,
    pub ca_k_norm: Tensor,
    pub mlp_gate: QLinear,
    pub mlp_up: QLinear,
    pub mlp_down: QLinear,
    pub scale_shift_table: Tensor, // [6, H]
    pub layer_type_full: bool,     // true = full attention, false = sliding-window(128)
    /// HeteroPlan segment device for this block's weights (GPU0/GPU1/CPU). The
    /// inter-block activation is a host Vec<f32>, so a spilled block needs no extra
    /// transfer; the block's own ops build their tensors on this device.
    pub device: Device, // native
}

/// Time-embedding tower: linear_1 -> silu -> linear_2 -> time_proj (-> 6H).
pub struct DitTimeEmbed {
    pub linear_1: DitLinear,
    pub linear_2: DitLinear,
    pub time_proj: DitLinear,
}

/// The turbo DiT (velocity-field predictor).
pub struct DitModel {
    pub proj_in: DitLinear,  // [hidden, in_ch.patch] applied after patchify
    pub cond_emb: DitLinear, // condition_embedder [hidden, hidden]
    pub time_embed: DitTimeEmbed,
    pub time_embed_r: DitTimeEmbed,
    pub layers: Vec<DitLayer>,
    pub norm_out: Tensor,
    pub out_scale_shift: Tensor, // [2, H] final AdaLN
    pub proj_out: DitLinear,     // [out_ch.patch, hidden] before unpatchify
    pub hidden: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub in_ch: usize,
    pub out_ch: usize,
    pub patch: usize,
    pub rope_theta: f32,
    pub sliding_window: usize,
    pub device: Device, // native
    /// Optional TIME-VARYING style conditioning for the cross-attention: when set, `enc` is the
    /// concatenation of several style encodings and `enc_bounds` marks each style's token range
    /// `[enc_bounds[r], enc_bounds[r+1])`. Each query frame then attends mostly to its time
    /// region's style tokens (triangular weights, smooth blend at the boundaries), so the SAME
    /// lyric/melody codes get rendered with a style that drifts over the timeline. `None` =
    /// ordinary single-style cross-attention (unchanged numerics).
    pub xattn_morph: Option<XattnMorph>,
}

/// Per-style enc token boundaries for the time-varying cross-attention (see `xattn_morph`).
/// `enc_bounds` is cumulative: `[0, S_0, S_0+S_1, ..., S_total]` (length = n_styles + 1).
#[derive(Clone)]
pub struct XattnMorph {
    pub enc_bounds: Vec<usize>,
}

/// Load one GGUF tensor -> a NATIVE F32 device tensor. The GGUF reader is the compat-stack
/// `gguf_file::Content` (one-time, file IO); the dequantized values are rebuilt as a native
/// tensor (like the VAE / encoders), so the whole DiT forward runs on the native stack.
fn dit_load_t(
    device: &Device,
    content: &crate::tensor::quantized::gguf_file::Content,
    file: &mut std::fs::File,
    name: &str,
) -> crate::tensor::Result<Tensor> {
    use crate::tensor::{DType as CDType, Device as CDevice};
    let dq = content
        .tensor(file, name, &CDevice::Cpu)?
        .dequantize(&CDevice::Cpu)?
        .to_dtype(CDType::F32)?;
    let dims = dq.dims().to_vec();
    let v = dq.flatten_all()?.to_vec1::<f32>()?;
    Ok(Tensor::from_vec_f32(v, dims)?.to_device(device)?)
}

fn dit_lin(
    device: &Device,
    content: &crate::tensor::quantized::gguf_file::Content,
    file: &mut std::fs::File,
    prefix: &str,
    bias: bool,
) -> crate::tensor::Result<DitLinear> {
    let w = dit_load_t(device, content, file, &format!("{prefix}.weight"))?;
    let b = if bias {
        Some(dit_load_t(
            device,
            content,
            file,
            &format!("{prefix}.bias"),
        )?)
    } else {
        None
    };
    Ok(DitLinear { w, b })
}

impl DitModel {
    /// Load the turbo DiT from acestep-v15-turbo-Q8_0.gguf (678 tensors). Geometry
    /// (hidden 2048, 24 blocks, 16h/8kv, hd128, in_ch192, out_ch64, patch2,
    /// rope 1e6, SWA128 alternating) from the gguf metadata / known config.
    /// ⚠️ STRUCTURAL load only - numerical correctness pending acestep.cpp --dump.
    pub fn from_gguf(path: &str) -> crate::tensor::Result<Self> {
        Self::from_gguf_for(
            path,
            crate::inference::place::audio_demand::ACE_REFERENCE_FRAMES,
        )
    }

    /// [`Self::from_gguf`] for a render of a KNOWN length.
    ///
    /// The denoiser's scratch follows the clip: the sequence it attends over is the latent
    /// timeline, so a four-minute track holds ten times the residual stream of a
    /// twenty-four-second one. A fixed reserve is right at exactly one length - generous on
    /// a sketch, and short on the track that then has to denoise on a card the reserve said
    /// would hold it - so the length decides the placement.
    ///
    /// `frames` is the LATENT frame count the render will produce, which the caller knows
    /// before it loads the model. [`Self::from_gguf`] stands in the reference clip for the
    /// callers that genuinely have no request in hand - the probes and the round-trips -
    /// so their placement is exactly what it has always been.
    pub fn from_gguf_for(path: &str, frames: usize) -> crate::tensor::Result<Self> {
        use crate::tensor::quantized::gguf_file;
        let mut f = std::fs::File::open(path)?;
        let c = gguf_file::read_mapped_file(&f)?;
        // Geometry from the GGUF metadata (turbo values as the fallback when a key is
        // genuinely absent), so the 2B turbo/sft (hidden 2048 / 24 blocks / 16h) AND the
        // 4B XL (hidden 2560 / 32 blocks / 32h, where head_count.head_dim != hidden) all
        // load with their real dims. The latent geometry (in_ch 192, out_ch 64, patch 2)
        // and the GQA kv-head/head_dim are shared across the family.
        let mu = |k: &str, d: usize| {
            c.metadata
                .get(k)
                .and_then(|v| v.to_u32().ok())
                .map(|x| x as usize)
                .unwrap_or(d)
        };
        let mf = |k: &str, d: f32| {
            c.metadata
                .get(k)
                .and_then(|v| {
                    v.to_f32()
                        .ok()
                        .or_else(|| v.to_u32().ok().map(|x| x as f32))
                })
                .unwrap_or(d)
        };
        let n_layers = mu("acestep-dit.block_count", 24);
        let hidden = mu("acestep-dit.embedding_length", 2048);
        let n_head = mu("acestep-dit.attention.head_count", 16);
        let n_kv = mu("acestep-dit.attention.head_count_kv", 8);
        let head_dim = mu("acestep-dit.attention.key_length", 128);
        let in_ch = mu("acestep.in_channels", 192);
        let out_ch = mu("acestep.audio_acoustic_hidden_dim", 64);
        let patch = mu("acestep.patch_size", 2);
        let sliding_window = mu("acestep.sliding_window", 128);
        let rope_theta = mf("acestep-dit.rope.freq_base", 1e6);
        // Standard placement (same mechanism as every other model): a HeteroPlan over
        // the 24 blocks from real free VRAM (pack-first on the fastest GPU, else split,
        // remainder -> CPU). The inter-block activation is a host Vec<f32>, so a
        // CPU-spilled / second-GPU block needs no extra transfer code. No KV cache (the
        // DiT runs full-sequence, not autoregressive) -> kv reserve 0.
        // Placement budgets against the REAL device footprint. The per-block linears (the
        // parameter mass) now stay QUANTIZED on device (QMatMul) and dequantize on the fly in
        // the matmul kernel, so the footprint is the quantized byte size - not a 4x F32 blow-up.
        // Budgeting on the quantized size lets the planner pack the compact model (a 4B-class
        // DiT fits a single card) instead of over-reserving for a phantom F32 weight set.
        // The checkpoint's tensor table below is the authority on the footprint; the file
        // is only a floor under it. When it cannot be stat'd the floor is simply absent -
        // it used to be a typed 4 GB, which is a figure nobody measured standing in for
        // one that was already known exactly two lines down.
        let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let model_size = c
            .tensor_infos
            .iter()
            .filter(|(n, _)| n.starts_with("decoder."))
            .map(|(_, info)| info.size_in_bytes() as u64)
            .sum::<u64>()
            .max(file_size);
        // Headroom for the full-sequence attention/FFN scratch AND a moderate concurrent
        // VRAM consumer, now sized from THIS render rather than fixed: the figure the
        // family was validated at, grown by what the clip and this checkpoint's width add
        // over the one it was validated with. Ample-VRAM placement at the reference clip
        // is unchanged (packs on GPU0); a longer track asks for what it actually needs.
        let ffn = mu("acestep-dit.feed_forward_length", 6144);
        let reserve: u64 =
            crate::inference::place::audio_demand::dit_reserve(frames, hidden, n_head, ffn, patch);
        let cudas = crate::inference::place::vram_manager::probe_under_pressure(reserve);
        let cuda_budget: Vec<(usize, u64)> = cudas.iter().map(|(i, fr, _)| (*i, *fr)).collect();
        let plan = HeteroPlan::calculate_with_kv_reserve(
            // The budgets below ALREADY exclude the reserve: `probe_under_pressure` probes through
            // `probe_cuda_devices`, which returns `stable_free - reserve`. Passing it again here
            // subtracted it TWICE - invisible at half a gigabyte, and fatal at twelve, where it
            // took both cards to zero usable and sent a whole video DiT to the host.
            n_layers,
            model_size,
            &cuda_budget,
            &[],
            1.0,
            0,
            0,
        );
        let dev_of_kind = |k: DeviceKind| -> Device {
            match k {
                DeviceKind::Cuda(idx) => cudas
                    .iter()
                    .find(|(i, _, _)| *i == idx)
                    .map(|(_, _, dv)| dv.clone())
                    .unwrap_or(Device::Cpu),
                _ => Device::Cpu,
            }
        };
        let layer_device = |l: usize| -> Device {
            plan.segments
                .iter()
                .find(|s| l >= s.layer_start && l < s.layer_end)
                .map(|s| dev_of_kind(s.kind))
                .unwrap_or(Device::Cpu)
        };
        let device = cudas
            .first()
            .map(|(_, _, dv)| dv.clone())
            .unwrap_or(Device::Cpu);
        // BLOCKING sync (yield the host core) instead of the default SPIN: the on-device
        // Euler trajectory hands the GPU a long fixed block sequence per step and then
        // blocks on a single readback - a spinning sync would peg a core doing nothing
        // while the GPU runs. Blocking sync lets that core idle. acestep-only, no numerics.
        #[cfg(feature = "cuda")]
        if let Device::Cuda(c) = &device {
            let _ = c.context().set_blocking_synchronize();
        }
        eprintln!("[ace-dit] geometry: hidden={hidden} layers={n_layers} heads={n_head}/{n_kv}kv hd={head_dim} ffn={ffn} in_ch={in_ch} out_ch={out_ch} patch={patch} rope={rope_theta} swa={sliding_window}");
        eprintln!(
            "[ace-dit] placement: {}",
            plan.segments
                .iter()
                .map(|s| format!("{}:{}-{}", s.kind, s.layer_start, s.layer_end))
                .collect::<Vec<_>>()
                .join(" ")
        );
        // QVarBuilder over the same GGUF for the per-block linears (kept quantized on device).
        // The F32 conv/small weights still load via the `Content`+`File` reader (`dit_load_t`).
        let vb = crate::inference::cache::qvb::from_gguf_cached(path, &device)?;
        let ffn = mu("acestep-dit.feed_forward_length", 6144);
        let (qd, kvd) = (n_head * head_dim, n_kv * head_dim); // q/o dim, k/v dim (GQA)
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            // The builder above only MAPS the file; the seconds are spent here, uploading a
            // block's weights to its device. Nothing in this loader goes through the shared
            // readers that count themselves, so the render's published reporter received
            // nothing at all and a caller watched a stage line sit still through the whole
            // load. Counted per block, like every other transformer in the fleet.
            crate::inference::serve::progress::scoped::note(
                crate::inference::serve::progress::phase::LOAD_MODEL,
                l,
                n_layers,
            );
            let p = format!("decoder.layers.{l}");
            let ld = layer_device(l);
            // Quantized per-block linear `[out,in]` on the block's device (bias=false).
            let ql = |indim: usize, outdim: usize, nm: &str| -> crate::tensor::Result<QLinear> {
                Ok(QLinear::new(
                    Weight::Quant(vb.qmatmul_on(
                        indim,
                        outdim,
                        &format!("{p}.{nm}.weight"),
                        &ld,
                    )?),
                    None,
                    indim,
                    outdim,
                ))
            };
            // layer_types alternate sliding/full starting with sliding (even=SWA).
            layers.push(DitLayer {
                self_attn_norm: dit_load_t(&ld, &c, &mut f, &format!("{p}.self_attn_norm.weight"))?,
                cross_attn_norm: dit_load_t(
                    &ld,
                    &c,
                    &mut f,
                    &format!("{p}.cross_attn_norm.weight"),
                )?,
                mlp_norm: dit_load_t(&ld, &c, &mut f, &format!("{p}.mlp_norm.weight"))?,
                sa_q: ql(hidden, qd, "self_attn.q_proj")?,
                sa_k: ql(hidden, kvd, "self_attn.k_proj")?,
                sa_v: ql(hidden, kvd, "self_attn.v_proj")?,
                sa_o: ql(qd, hidden, "self_attn.o_proj")?,
                sa_q_norm: dit_load_t(&ld, &c, &mut f, &format!("{p}.self_attn.q_norm.weight"))?,
                sa_k_norm: dit_load_t(&ld, &c, &mut f, &format!("{p}.self_attn.k_norm.weight"))?,
                ca_q: ql(hidden, qd, "cross_attn.q_proj")?,
                ca_k: ql(hidden, kvd, "cross_attn.k_proj")?,
                ca_v: ql(hidden, kvd, "cross_attn.v_proj")?,
                ca_o: ql(qd, hidden, "cross_attn.o_proj")?,
                ca_q_norm: dit_load_t(&ld, &c, &mut f, &format!("{p}.cross_attn.q_norm.weight"))?,
                ca_k_norm: dit_load_t(&ld, &c, &mut f, &format!("{p}.cross_attn.k_norm.weight"))?,
                mlp_gate: ql(hidden, ffn, "mlp.gate_proj")?,
                mlp_up: ql(hidden, ffn, "mlp.up_proj")?,
                mlp_down: ql(ffn, hidden, "mlp.down_proj")?,
                scale_shift_table: dit_load_t(&ld, &c, &mut f, &format!("{p}.scale_shift_table"))?,
                layer_type_full: l % 2 == 1,
                device: ld,
            });
        }
        Ok(DitModel {
            proj_in: dit_lin(&device, &c, &mut f, "decoder.proj_in.1", true)?,
            cond_emb: dit_lin(&device, &c, &mut f, "decoder.condition_embedder", true)?,
            time_embed: DitTimeEmbed {
                linear_1: dit_lin(&device, &c, &mut f, "decoder.time_embed.linear_1", true)?,
                linear_2: dit_lin(&device, &c, &mut f, "decoder.time_embed.linear_2", true)?,
                time_proj: dit_lin(&device, &c, &mut f, "decoder.time_embed.time_proj", true)?,
            },
            time_embed_r: DitTimeEmbed {
                linear_1: dit_lin(&device, &c, &mut f, "decoder.time_embed_r.linear_1", true)?,
                linear_2: dit_lin(&device, &c, &mut f, "decoder.time_embed_r.linear_2", true)?,
                time_proj: dit_lin(&device, &c, &mut f, "decoder.time_embed_r.time_proj", true)?,
            },
            layers,
            norm_out: dit_load_t(&device, &c, &mut f, "decoder.norm_out.weight")?,
            out_scale_shift: dit_load_t(&device, &c, &mut f, "decoder.scale_shift_table")?,
            proj_out: dit_lin(&device, &c, &mut f, "decoder.proj_out.1", true)?,
            hidden,
            n_head,
            n_kv,
            head_dim,
            in_ch,
            out_ch,
            patch,
            rope_theta,
            sliding_window,
            device,
            xattn_morph: None,
        })
    }

    /// Attach a LoRA adapter (safetensors) onto the attention projections. Tensors are named
    /// `...layers.{i}.{self_attn|cross_attn}.{q,k,v,o}_proj.lora_{A,B}.weight` (BF16/F32). The
    /// delta `scale.B.A` is applied at forward (low-rank, base weight untouched). `scale` is the
    /// user adapter weight (x alpha/rank, assumed 1 when the adapter ships no config). The
    /// adapter's in-dim must match this DiT's hidden (2B = 2048; an XL-only adapter won't fit).
    /// Returns the number of attached deltas.
    pub fn apply_lora(&mut self, path: &str, scale: f32) -> crate::tensor::Result<usize> {
        let err = |m: String| crate::tensor::Error(m);
        let bytes = std::fs::read(path).map_err(|e| err(format!("lora read `{path}`: {e}")))?;
        if bytes.len() < 8 {
            return Err(err("lora: truncated".into()));
        }
        let hlen = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let hdr: serde_json::Value = serde_json::from_slice(&bytes[8..8 + hlen])
            .map_err(|e| err(format!("lora header json: {e}")))?;
        let data0 = 8 + hlen;
        let obj = hdr
            .as_object()
            .ok_or_else(|| err("lora header not an object".into()))?;
        // Read one tensor -> (shape, f32 values). LoRA factors ship BF16/F16/F32.
        let read_t = |name: &str| -> Option<(Vec<usize>, Vec<f32>)> {
            let e = obj.get(name)?.as_object()?;
            let dtype = e.get("dtype")?.as_str()?;
            let shape: Vec<usize> = e
                .get("shape")?
                .as_array()?
                .iter()
                .filter_map(|v| v.as_u64().map(|x| x as usize))
                .collect();
            let off = e.get("data_offsets")?.as_array()?;
            let (s, en) = (off[0].as_u64()? as usize, off[1].as_u64()? as usize);
            let raw = &bytes[data0 + s..data0 + en];
            let vals: Vec<f32> = match dtype {
                "BF16" => raw
                    .chunks_exact(2)
                    .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
                    .collect(),
                "F32" => raw
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect(),
                _ => return None, // unsupported adapter dtype
            };
            Some((shape, vals))
        };
        let mut attached = 0usize;
        let n = self.layers.len();
        for i in 0..n {
            let dev = self.layers[i].device.clone();
            for (attn, is_self) in [("self_attn", true), ("cross_attn", false)] {
                for proj in ['q', 'k', 'v', 'o'] {
                    // Names carry repeated "base_model.model." wrapping prefixes -> match by suffix.
                    let sa = format!("layers.{i}.{attn}.{proj}_proj.lora_A.weight");
                    let sb = format!("layers.{i}.{attn}.{proj}_proj.lora_B.weight");
                    let (ka, kb) = match (
                        obj.keys().find(|k| k.ends_with(&sa)),
                        obj.keys().find(|k| k.ends_with(&sb)),
                    ) {
                        (Some(a), Some(b)) => (a.clone(), b.clone()),
                        _ => continue,
                    };
                    let (sha, va) = read_t(&ka).ok_or_else(|| err(format!("lora read {ka}")))?;
                    let (shb, vb) = read_t(&kb).ok_or_else(|| err(format!("lora read {kb}")))?;
                    if sha.len() != 2 || shb.len() != 2 || sha[1] != self.hidden {
                        return Err(err(format!(
                            "lora {ka}: in-dim {:?} != DiT hidden {} (wrong model size?)",
                            sha, self.hidden
                        )));
                    }
                    let a = Tensor::from_vec_f32(va, (sha[0], sha[1]))?.to_device(&dev)?;
                    let b = Tensor::from_vec_f32(vb, (shb[0], shb[1]))?.to_device(&dev)?;
                    let ld = acestep_delta(&a, &b, scale)?;
                    let lin = match (is_self, proj) {
                        (true, 'q') => &mut self.layers[i].sa_q,
                        (true, 'k') => &mut self.layers[i].sa_k,
                        (true, 'v') => &mut self.layers[i].sa_v,
                        (true, _) => &mut self.layers[i].sa_o,
                        (false, 'q') => &mut self.layers[i].ca_q,
                        (false, 'k') => &mut self.layers[i].ca_k,
                        (false, 'v') => &mut self.layers[i].ca_v,
                        (false, _) => &mut self.layers[i].ca_o,
                    };
                    lin.add_lora(ld)?;
                    attached += 1;
                }
            }
        }
        Ok(attached)
    }
}

impl DitModel {
    /// proj_in: `input` `[in_ch.T]` (channel-major, = concat(context,xt)) -> patchify
    /// (group P frames) -> the permuted proj_in linear `[in_ch.P -> hidden]` + bias.
    /// Returns `(hidden[hidden.S], S=T/patch)` in hidden-major layout (matches the
    /// oracle's `hidden_after_proj_in` [H,S]). The permute aligns proj_in's in-dim
    /// with patchify's `p.in_ch+c` ordering (oracle dit_load_proj_in_w).
    pub fn proj_in_forward(
        &self,
        input: &[f32],
        t: usize,
    ) -> crate::tensor::Result<(Vec<f32>, usize)> {
        let (in_ch, p, h) = (self.in_ch, self.patch, self.hidden);
        let (patched, s) = patchify(input, in_ch, t, p); // [(in_ch.P), S], (in_ch.P) outer
                                                         // proj_in.w raw [h, in_ch, p] -> transpose last two -> [h, p, in_ch] ->
                                                         // reshape [h, in_ch.p]; in-index = p.in_ch + c (matches patchify).
        let w = self
            .proj_in
            .w
            .transpose(1, 2)?
            .contiguous()?
            .reshape((h, in_ch * p))?;
        // patched [in_ch.P, S] -> tokens-major [S, in_ch.P]; linear: [S,K].[K,H] -> [S,H].
        let pt = Tensor::from_vec_f32(patched, (in_ch * p, s))?
            .to_device(&self.device)?
            .transpose(0, 1)?
            .contiguous()?;
        let hid = pt.matmul_t(&w)?; // [S, H]
        let hid = hid.broadcast_add(self.proj_in.b.as_ref().unwrap())?;
        // Return [S,H] row-major (flat[s.H+h]) - matches the oracle's ggml-contiguous
        // [H,S] dump (ne0=H fastest = column-major = the same flat order).
        Ok((hid.flatten_all()?.to_vec1_f32()?, s))
    }
}

/// RoPE cos/sin tables (NEOX): `cos[p,j]=cos(p.θ^(-2j/D))`, `sin` likewise,
/// `j in 0..D/2`, `p in 0..S`. Returns ([S.D/2], [S.D/2]).
fn rope_tables(s: usize, d: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = d / 2;
    let (mut cos, mut sin) = (vec![0f32; s * half], vec![0f32; s * half]);
    for p in 0..s {
        for j in 0..half {
            let freq = theta.powf(-2.0 * (j as f32) / (d as f32));
            let a = (p as f32) * freq;
            cos[p * half + j] = a.cos();
            sin[p * half + j] = a.sin();
        }
    }
    (cos, sin)
}

impl DitModel {
    /// Self-attn Q/K up to RoPE: `x` `[S.H]` (the AdaLN-modulated norm, row-major
    /// [S,H]) -> projects, reshapes to heads, applies qk-norm (RMS over D x {q,k}_norm)
    /// then NEOX RoPE (θ from cfg). Returns q,k as `[heads.S.D]` (head-major,
    /// matching the oracle's [D,Nh,S] dumps when sliced). For validating the
    /// rope/qk-norm conventions before the full attention.
    pub fn self_attn_qk_roped(
        &self,
        lin: &QLinear,
        qknorm: &Tensor,
        x: &Tensor,
        n_heads: usize,
        s: usize,
    ) -> crate::tensor::Result<Tensor> {
        let d = self.head_dim;
        // x [S,H] @ W[heads.D, H]^T -> [S, heads.D]
        let proj = lin.forward(x)?;
        // [S, heads, D] -> [1, heads, S, D], then qk-norm: RMS over D x per-D weight.
        let q = in_heads(proj, n_heads, s, d)?;
        let q = q.rms_norm(qknorm, self.rms_eps())?;
        // NEOX rope.
        let (cosv, sinv) = rope_tables(s, d, self.rope_theta);
        // cos/sin follow the input tensor's device so a CPU/2nd-GPU-placed block's RoPE
        // runs where its weights are (q is derived from x on the block's device).
        let cos = Tensor::from_vec_f32(cosv, (s, d / 2))?.to_device(&x.device())?;
        let sin = Tensor::from_vec_f32(sinv, (s, d / 2))?.to_device(&x.device())?;
        q.rope(&cos, &sin) // [1, heads, S, D]
    }

    fn rms_eps(&self) -> f32 {
        1e-6
    }

    /// One time-embedding tower (oracle dit_ggml_build_temb): sinusoid(t.1000) ->
    /// lin1+silu -> lin2 = temb [H]; tproj [6H] = time_proj(silu(temb)). Returns (temb, tproj).
    fn build_temb(te: &DitTimeEmbed, t: f32) -> crate::tensor::Result<(Vec<f32>, Vec<f32>)> {
        let lin = |w: &Tensor, b: &Option<Tensor>, x: &[f32]| -> crate::tensor::Result<Vec<f32>> {
            let dims = w.dims();
            let (out, inn) = (dims[0], dims[1]);
            let wv: Vec<f32> = w.flatten_all()?.to_vec1_f32()?;
            let bv: Vec<f32> = match b {
                Some(bb) => bb.flatten_all()?.to_vec1_f32()?,
                None => vec![0.0; out],
            };
            Ok((0..out)
                .map(|o| {
                    let mut s = bv[o];
                    for i in 0..inn {
                        s += x[i] * wv[o * inn + i];
                    }
                    s
                })
                .collect())
        };
        let silu = |v: &mut [f32]| {
            for x in v.iter_mut() {
                *x = *x / (1.0 + (-*x).exp());
            }
        };
        let sin = dit_timestep_embedding(t); // [256]
        let mut h = lin(&te.linear_1.w, &te.linear_1.b, &sin)?;
        silu(&mut h);
        let temb = lin(&te.linear_2.w, &te.linear_2.b, &h)?;
        let mut h2 = temb.clone();
        silu(&mut h2);
        let tproj = lin(&te.time_proj.w, &te.time_proj.b, &h2)?;
        Ok((temb, tproj))
    }

    /// Compute (temb [H], tproj [6H]) for timestep `t` (turbo: `t_r=t` so the `_r`
    /// tower gets input 0). temb = temb_t + temb_r; tproj = tproj_t + tproj_r.
    pub fn time_embed_forward(&self, t: f32) -> crate::tensor::Result<(Vec<f32>, Vec<f32>)> {
        let (temb_t, tproj_t) = Self::build_temb(&self.time_embed, t)?;
        let (temb_r, tproj_r) = Self::build_temb(&self.time_embed_r, 0.0)?;
        let temb = temb_t.iter().zip(&temb_r).map(|(a, b)| a + b).collect();
        let tproj = tproj_t.iter().zip(&tproj_r).map(|(a, b)| a + b).collect();
        Ok((temb, tproj))
    }

    /// Full self-attention for one layer: `x` `[S,H]` device tensor (the AdaLN-
    /// modulated norm) -> `[S,H]` device tensor (post o_proj). Bidirectional GQA +
    /// qk-norm + NEOX RoPE + sliding-window mask (`win`; pass usize::MAX for full
    /// attention). Reuses the cond_emb-validated `x@w.t()` matmul. The activation
    /// stays on `l.device` for the whole block - no host round-trip.
    pub fn self_attn_forward(
        &self,
        l: &DitLayer,
        x: &Tensor,
        s: usize,
        win: usize,
    ) -> crate::tensor::Result<Tensor> {
        use crate::inference::model::acestep::ops::{repeat_kv, sdpa};
        let (nh, nkv, d) = (self.n_head, self.n_kv, self.head_dim);
        let q = self.self_attn_qk_roped(&l.sa_q, &l.sa_q_norm, x, nh, s)?; // [1,nh,s,d]
        let k = self.self_attn_qk_roped(&l.sa_k, &l.sa_k_norm, x, nkv, s)?; // [1,nkv,s,d]
        let v = in_heads(l.sa_v.forward(x)?, nkv, s, d)?; // [1,nkv,s,d]
        let nrep = nh / nkv;
        let (k, v) = (repeat_kv(k, nrep)?, repeat_kv(v, nrep)?); // [1,nh,s,d]
        let scale = 1.0f32 / (d as f32).sqrt();
        // Bidirectional sliding-window mask (even layers); None = full attention.
        let mask = if win < usize::MAX {
            let mut m = vec![0f32; s * s];
            for i in 0..s {
                for j in 0..s {
                    if (i as i64 - j as i64).unsigned_abs() as usize > win {
                        m[i * s + j] = f32::NEG_INFINITY;
                    }
                }
            }
            Some(Tensor::from_vec_f32(m, (s, s))?.to_device(&l.device)?)
        } else {
            None
        };
        let attn = sdpa(&q, &k, &v, mask.as_ref(), false, scale, 1.0)?; // [1,nh,s,d]
        let ao = attn.transpose(1, 2)?.contiguous()?.reshape((s, nh * d))?; // [s, nh.d]
        l.sa_o.forward(&ao) // [s, H]
    }

    /// proj -> heads -> qk-norm (NO RoPE). For cross-attention Q/K. Returns `[1,heads,seq,d]`.
    fn proj_heads_qknorm(
        &self,
        lin: &QLinear,
        qknorm: &Tensor,
        x: &Tensor,
        n_heads: usize,
        seq: usize,
    ) -> crate::tensor::Result<Tensor> {
        let d = self.head_dim;
        let q = in_heads(lin.forward(x)?, n_heads, seq, d)?;
        q.rms_norm(qknorm, self.rms_eps())
    }

    /// Full cross-attention: `xq` `[S,H]` device tensor (Q source) attends to `enc`
    /// `[enc_S.H]` host (K/V source - a per-step constant, uploaded to the block
    /// device here). GQA + qk-norm, NO RoPE, NO mask. Returns `[S,H]` device tensor
    /// post o_proj. The query activation stays on device - no host round-trip.
    pub fn cross_attn_forward(
        &self,
        l: &DitLayer,
        xq: &Tensor,
        enc: &[f32],
        s: usize,
        enc_s: usize,
    ) -> crate::tensor::Result<Tensor> {
        use crate::inference::model::acestep::ops::{repeat_kv, sdpa};
        let (h, nh, nkv, d) = (self.hidden, self.n_head, self.n_kv, self.head_dim);
        let xe = Tensor::from_vec_f32(enc.to_vec(), (enc_s, h))?.to_device(&l.device)?;
        let q = self.proj_heads_qknorm(&l.ca_q, &l.ca_q_norm, xq, nh, s)?; // [1,nh,s,d]
        let k = self.proj_heads_qknorm(&l.ca_k, &l.ca_k_norm, &xe, nkv, enc_s)?; // [1,nkv,enc_s,d]
        let v = in_heads(l.ca_v.forward(&xe)?, nkv, enc_s, d)?; // [1,nkv,enc_s,d]
        let nrep = nh / nkv;
        let (k, v) = (repeat_kv(k, nrep)?, repeat_kv(v, nrep)?);
        let scale = 1.0f32 / (d as f32).sqrt();
        // Time-varying style morph: an additive cross-attn bias `[s, enc_s]` so query frame `si`
        // (continuous style position p = si/s.N) attends mostly to its region's style enc tokens  -
        // bias = LOG.(w-1), w = triangular weight of that token's style at p (1 at the region
        // centre, ramping to 0.5 at a boundary -> adjacent styles blend smoothly). None = unchanged.
        let xmask = match &self.xattn_morph {
            Some(m) if m.enc_bounds.len() >= 3 && *m.enc_bounds.last().unwrap() == enc_s => {
                const LOG: f32 = 24.0;
                let n = m.enc_bounds.len() - 1;
                let mut tok_style = vec![0usize; enc_s];
                for r in 0..n {
                    for j in m.enc_bounds[r]..m.enc_bounds[r + 1] {
                        tok_style[j] = r;
                    }
                }
                let mut data = vec![0f32; s * enc_s];
                for si in 0..s {
                    let p = (si as f32 + 0.5) / s as f32 * n as f32;
                    for j in 0..enc_s {
                        let w = (1.0 - (p - (tok_style[j] as f32 + 0.5)).abs()).max(0.0);
                        data[si * enc_s + j] = LOG * (w - 1.0);
                    }
                }
                Some(Tensor::from_vec_f32(data, (s, enc_s))?.to_device(&l.device)?)
            }
            _ => None,
        };
        let attn = sdpa(&q, &k, &v, xmask.as_ref(), false, scale, 1.0)?; // [1,nh,s,d]
        let ao = attn.transpose(1, 2)?.contiguous()?.reshape((s, nh * d))?;
        l.ca_o.forward(&ao) // [s, H]
    }

    /// SwiGLU FFN: `x` `[S,H]` device tensor -> `silu(gate).up` -> down. Returns
    /// `[S,H]` device tensor. Activation stays on `l.device` - no host round-trip.
    pub fn swiglu_ffn(&self, l: &DitLayer, x: &Tensor, _s: usize) -> crate::tensor::Result<Tensor> {
        let gate = l.mlp_gate.weight().forward(x)?; // [S, inter]
        let up = l.mlp_up.weight().forward(x)?;
        // silu(gate).up = (gate . sigmoid(gate)) . up, on-device.
        let act = gate.silu()?.mul(&up)?;
        l.mlp_down.weight().forward(&act) // [S, H]
    }

    /// Full DiT layer: AdaLN -> SA(gated) -> CA(plain add) -> FFN(gated). `hidden` is a
    /// device `[S,H]` tensor that stays on the block's device across all three sub-
    /// layers (no host round-trip); `enc` row-major [enc_S,H] host (per-step const);
    /// `tproj` `[6H]` host. Returns the updated `[S,H]` device tensor.
    pub fn layer_forward(
        &self,
        l: &DitLayer,
        hidden: &Tensor,
        tproj: &[f32],
        enc: &[f32],
        s: usize,
        enc_s: usize,
        win: usize,
    ) -> crate::tensor::Result<Tensor> {
        let h = self.hidden;
        let eps = self.rms_eps();
        // AdaLN-single: table[6H] + tproj[6H], split into six [H] modulation vectors.
        // Build each as a [1,H] device tensor so the affine/gate broadcast over the
        // token axis [S,H] runs on the block's device.
        let sst: Vec<f32> = l.scale_shift_table.flatten_all()?.to_vec1_f32()?;
        let a = adaln_split(tproj, &sst, h);
        let dev = |v: &[f32]| -> crate::tensor::Result<Tensor> {
            Tensor::from_vec_f32(v.to_vec(), (1, h))?.to_device(&l.device)
        };
        // (1 + scale) as device tensors for the modulate step.
        let one_plus = |v: &[f32]| -> crate::tensor::Result<Tensor> {
            dev(&v.iter().map(|x| 1.0 + x).collect::<Vec<f32>>())
        };
        let (scale_sa, shift_sa, gate_sa) =
            (one_plus(&a.scale_sa)?, dev(&a.shift_sa)?, dev(&a.gate_sa)?);
        let (scale_mlp, shift_mlp, gate_mlp) = (
            one_plus(&a.scale_mlp)?,
            dev(&a.shift_mlp)?,
            dev(&a.gate_mlp)?,
        );
        // `modulate(rms_norm(x, w)) = norm.(1+scale) + shift`, per-channel broadcast.
        let modulate =
            |x: &Tensor, w: &Tensor, sc: &Tensor, sh: &Tensor| -> crate::tensor::Result<Tensor> {
                x.rms_norm(w, eps)?.broadcast_mul(sc)?.broadcast_add(sh)
            };
        let mut hid = hidden.to_device(&l.device)?;
        // self-attn (AdaLN-modulated norm -> attn -> gate -> residual add).
        let norm_sa = modulate(&hid, &l.self_attn_norm, &scale_sa, &shift_sa)?;
        let sa = self.self_attn_forward(l, &norm_sa, s, win)?;
        hid = hid.add(&sa.broadcast_mul(&gate_sa)?)?;
        // cross-attn (rms-norm only, plain residual add - no gate, no modulation).
        let norm_ca = hid.rms_norm(&l.cross_attn_norm, eps)?;
        let ca = self.cross_attn_forward(l, &norm_ca, enc, s, enc_s)?;
        hid = hid.add(&ca)?;
        // FFN (AdaLN-modulated norm -> swiglu -> gate -> residual add).
        let norm_ffn = modulate(&hid, &l.mlp_norm, &scale_mlp, &shift_mlp)?;
        let ff = self.swiglu_ffn(l, &norm_ffn, s)?;
        hid = hid.add(&ff.broadcast_mul(&gate_mlp)?)?;
        Ok(hid)
    }

    /// Full DiT velocity prediction for one step. `input` = channel-major `[in_ch.T]`
    /// = concat(context, xt). `tproj` `[6H]`, `temb` `[H]`, `enc` row-major `[enc_S.H]`.
    /// Returns velocity `[out_ch.T]` (channel-major).
    pub fn velocity_forward(
        &self,
        input: &[f32],
        tproj: &[f32],
        temb: &[f32],
        enc: &[f32],
        t: usize,
        enc_s: usize,
    ) -> crate::tensor::Result<Vec<f32>> {
        let (h, oc, p) = (self.hidden, self.out_ch, self.patch);
        // proj_in stays host (input is a host vec); the resulting hidden goes on-device
        // and STAYS on-device across all 24 blocks (only re-homed if a block is placed
        // on a different segment device) - no per-block host↔GPU round-trip.
        let (hid0, s) = self.proj_in_forward(input, t)?; // [S.H] host
        let mut hid = Tensor::from_vec_f32(hid0, (s, h))?.to_device(&self.device)?;
        for l in &self.layers {
            // even layers = sliding-window(128), odd layers = full attention (no mask).
            let win = if l.layer_type_full {
                usize::MAX
            } else {
                self.sliding_window
            };
            hid = self.layer_forward(l, &hid, tproj, enc, s, enc_s, win)?;
        }
        hid = hid.to_device(&self.device)?;
        // Final AdaLN-single: rms_norm(hid).(1 + (out_scale + temb)) + (out_shift + temb),
        // per-channel broadcast over the token axis - on-device.
        let oss: Vec<f32> = self.out_scale_shift.flatten_all()?.to_vec1_f32()?; // [2,H]->2H
        let oshift: Vec<f32> = (0..h).map(|i| oss[i] + temb[i]).collect();
        let oscale: Vec<f32> = (0..h).map(|i| 1.0 + oss[h + i] + temb[i]).collect();
        let oshift_t = Tensor::from_vec_f32(oshift, (1, h))?.to_device(&self.device)?;
        let oscale_t = Tensor::from_vec_f32(oscale, (1, h))?.to_device(&self.device)?;
        let nout = hid
            .rms_norm(&self.norm_out, self.rms_eps())?
            .broadcast_mul(&oscale_t)?
            .broadcast_add(&oshift_t)?; // [S,H]
        let w2d = self
            .proj_out
            .w
            .transpose(0, 2)?
            .contiguous()?
            .reshape((oc * p, h))?;
        let proj: Vec<f32> = w2d.matmul_t(&nout)?.flatten_all()?.to_vec1_f32()?; // [oc.P, S] flat[j.S+s]
        let (mut vel, t_out) = unpatchify(&proj, oc, s, p); // [oc.T]
        let b = self
            .proj_out
            .b
            .as_ref()
            .unwrap()
            .flatten_all()?
            .to_vec1_f32()?;
        for c in 0..oc {
            for ti in 0..t_out {
                vel[c * t_out + ti] += b[c];
            }
        }
        Ok(vel)
    }
}

/// Per-step-FIXED device tensors for the on-device Euler trajectory: everything that
/// is constant across all 8 Euler steps (and across the 24 blocks) - the cross-attn
/// `enc` source, the RoPE cos/sin tables, the sliding-window mask, the optional style-
/// morph cross-attn bias, and the two permuted proj weights - built ONCE so the per-
/// step loop refreshes only the latent and the timestep AdaLN vectors.
struct DitFixed {
    enc: Tensor,                // [enc_s, H] cross-attn K/V source (uploaded once)
    cos: Tensor,                // [S, D/2] NEOX RoPE cos
    sin: Tensor,                // [S, D/2] NEOX RoPE sin
    swa_mask: Tensor,           // [S, S] sliding-window additive mask (even/SWA layers)
    morph_mask: Option<Tensor>, // [S, enc_s] style-morph cross-attn bias, if enabled
    proj_in_w: Tensor,          // proj_in permuted to [H, in_ch.P]
    proj_out_w: Tensor,         // proj_out permuted to [oc.P, H]
}

impl DitModel {
    /// True when the whole DiT is resident on the one primary CUDA device (the only
    /// case the on-device trajectory handles - a HeteroPlan that spilled blocks onto a
    /// second GPU / the CPU keeps the eager `dit_latent` path, which re-homes per block).
    pub fn all_on_primary(&self) -> bool {
        self.device.is_cuda()
            && self
                .layers
                .iter()
                .all(|l| l.device.same_device(&self.device))
    }

    /// Self-attn Q/K with PRE-BUILT RoPE tables (no per-call host rope rebuild).
    /// Identical math to [`self_attn_qk_roped`] - only the cos/sin source differs.
    fn sa_qk_fast(
        &self,
        lin: &QLinear,
        qknorm: &Tensor,
        x: &Tensor,
        n_heads: usize,
        s: usize,
        cos: &Tensor,
        sin: &Tensor,
    ) -> crate::tensor::Result<Tensor> {
        let d = self.head_dim;
        let q = in_heads(lin.forward(x)?, n_heads, s, d)?;
        let q = q.rms_norm(qknorm, self.rms_eps())?;
        q.rope(cos, sin)
    }

    /// Self-attention with pre-built RoPE/mask (no per-call host rebuilds or uploads).
    fn self_attn_fast(
        &self,
        l: &DitLayer,
        x: &Tensor,
        s: usize,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
    ) -> crate::tensor::Result<Tensor> {
        use crate::inference::model::acestep::ops::{repeat_kv, sdpa};
        let (nh, nkv, d) = (self.n_head, self.n_kv, self.head_dim);
        let q = self.sa_qk_fast(&l.sa_q, &l.sa_q_norm, x, nh, s, cos, sin)?;
        let k = self.sa_qk_fast(&l.sa_k, &l.sa_k_norm, x, nkv, s, cos, sin)?;
        let v = in_heads(l.sa_v.forward(x)?, nkv, s, d)?;
        let nrep = nh / nkv;
        let (k, v) = (repeat_kv(k, nrep)?, repeat_kv(v, nrep)?);
        let scale = 1.0f32 / (d as f32).sqrt();
        let attn = sdpa(&q, &k, &v, mask, false, scale, 1.0)?;
        let ao = attn.transpose(1, 2)?.contiguous()?.reshape((s, nh * d))?;
        l.sa_o.forward(&ao)
    }

    /// Cross-attention against a PRE-UPLOADED `enc` device tensor and a pre-built morph
    /// bias (no per-call enc upload / morph rebuild). Identical math to [`cross_attn_forward`].
    fn cross_attn_fast(
        &self,
        l: &DitLayer,
        xq: &Tensor,
        enc: &Tensor,
        s: usize,
        enc_s: usize,
        morph: Option<&Tensor>,
    ) -> crate::tensor::Result<Tensor> {
        use crate::inference::model::acestep::ops::{repeat_kv, sdpa};
        let (nh, nkv, d) = (self.n_head, self.n_kv, self.head_dim);
        let q = self.proj_heads_qknorm(&l.ca_q, &l.ca_q_norm, xq, nh, s)?;
        let k = self.proj_heads_qknorm(&l.ca_k, &l.ca_k_norm, enc, nkv, enc_s)?;
        let v = in_heads(l.ca_v.forward(enc)?, nkv, enc_s, d)?;
        let nrep = nh / nkv;
        let (k, v) = (repeat_kv(k, nrep)?, repeat_kv(v, nrep)?);
        let scale = 1.0f32 / (d as f32).sqrt();
        let attn = sdpa(&q, &k, &v, morph, false, scale, 1.0)?;
        let ao = attn.transpose(1, 2)?.contiguous()?.reshape((s, nh * d))?;
        l.ca_o.forward(&ao)
    }

    /// One DiT block, fully on-device. The AdaLN-single vectors are derived on the GPU
    /// (`scale_shift_table + tproj`, sliced into the six [1,H] rows) instead of the eager
    /// path's per-block `scale_shift_table` device->host readback + host `adaln_split`  -
    /// removing a forced sync per block. `tproj` is a per-step [6,H] device tensor.
    /// Bit-identical to [`layer_forward`] (same kernels/values).
    fn layer_forward_fast(
        &self,
        l: &DitLayer,
        hidden: &Tensor,
        tproj: &Tensor,
        fx: &DitFixed,
        s: usize,
        enc_s: usize,
        swa: bool,
    ) -> crate::tensor::Result<Tensor> {
        let eps = self.rms_eps();
        // AdaLN-single on-device: adaln[6,H] = scale_shift_table + tproj, sliced into the
        // six modulation rows (order: shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp,
        // gate_mlp); `(1+scale)` folded via affine. Each row is a [1,H] broadcast operand.
        let adaln = l.scale_shift_table.add(tproj)?; // [6,H]
        let row = |i: usize| adaln.narrow(0, i, 1); // [1,H]
        let (shift_sa, scale_sa, gate_sa) = (row(0)?, row(1)?.affine(1.0, 1.0)?, row(2)?);
        let (shift_mlp, scale_mlp, gate_mlp) = (row(3)?, row(4)?.affine(1.0, 1.0)?, row(5)?);
        let modulate =
            |x: &Tensor, w: &Tensor, sc: &Tensor, sh: &Tensor| -> crate::tensor::Result<Tensor> {
                x.rms_norm(w, eps)?.broadcast_mul(sc)?.broadcast_add(sh)
            };
        let mut hid = hidden.clone();
        let norm_sa = modulate(&hid, &l.self_attn_norm, &scale_sa, &shift_sa)?;
        let sa = self.self_attn_fast(
            l,
            &norm_sa,
            s,
            &fx.cos,
            &fx.sin,
            if swa { Some(&fx.swa_mask) } else { None },
        )?;
        hid = hid.add(&sa.broadcast_mul(&gate_sa)?)?;
        let norm_ca = hid.rms_norm(&l.cross_attn_norm, eps)?;
        let ca = self.cross_attn_fast(l, &norm_ca, &fx.enc, s, enc_s, fx.morph_mask.as_ref())?;
        hid = hid.add(&ca)?;
        let norm_ffn = modulate(&hid, &l.mlp_norm, &scale_mlp, &shift_mlp)?;
        let ff = self.swiglu_ffn(l, &norm_ffn, s)?;
        hid = hid.add(&ff.broadcast_mul(&gate_mlp)?)?;
        Ok(hid)
    }

    /// On-device velocity prediction for one Euler step. `hid0` is the proj_in output
    /// `[S,H]` device tensor (kept on-device across all 24 blocks). `tproj`/`temb` are
    /// the per-step AdaLN device tensors ([6,H] / [1,H]). Returns the proj_out result
    /// `[oc.P, S]` as a device tensor (the host then unpatchifies + adds bias). The final
    /// AdaLN reads `out_scale_shift` on-device (no host readback). Bit-identical to the
    /// post-proj_in tail of [`velocity_forward`].
    fn velocity_tail_fast(
        &self,
        hid0: Tensor,
        tproj: &Tensor,
        temb: &Tensor,
        fx: &DitFixed,
        s: usize,
        enc_s: usize,
    ) -> crate::tensor::Result<Tensor> {
        let mut hid = hid0;
        for l in self.layers.iter() {
            let swa = !l.layer_type_full; // even layers = sliding-window, odd = full
            hid = self.layer_forward_fast(l, &hid, tproj, fx, s, enc_s, swa)?;
        }
        // Final AdaLN-single on-device: rms_norm.(1 + out_scale + temb) + (out_shift + temb).
        let oshift = self.out_scale_shift.narrow(0, 0, 1)?.add(temb)?; // [1,H]
        let oscale = self
            .out_scale_shift
            .narrow(0, 1, 1)?
            .add(temb)?
            .affine(1.0, 1.0)?; // [1,H]
        let nout = hid
            .rms_norm(&self.norm_out, self.rms_eps())?
            .broadcast_mul(&oscale)?
            .broadcast_add(&oshift)?;
        fx.proj_out_w.matmul_t(&nout) // [oc.P, S]
    }

    /// 8-step turbo Euler trajectory, fully on-device (CUDA fast path for the launch-
    /// bound DiT). Builds all per-step-FIXED tensors ONCE (cross-attn enc, RoPE tables,
    /// SWA mask, style-morph bias, permuted proj weights) and keeps the activation on the
    /// GPU across all 24 blocks; per step it refreshes only the latent `xt` and the
    /// timestep AdaLN vectors (`tproj`/`temb`). This removes the eager path's per-block
    /// `scale_shift_table` device->host syncs and its redundant per-block host rebuilds of
    /// the mask / RoPE tables / enc upload that starved the GPU. Numerically bit-identical
    /// to the eager `dit_latent` (same kernels, same values). Returns the latent `[T,64]`
    /// token-major. Only valid when [`all_on_primary`]; the caller falls back otherwise.
    pub fn dit_trajectory_ondevice(
        &self,
        context: &[f32],
        enc: &[f32],
        enc_s: usize,
        uncond: Option<(&[f32], usize)>,
        noise: &[f32],
        t: usize,
        sp: &crate::inference::model::acestep::pipeline::DitSample,
        progress: Option<&crate::inference::serve::progress::ProgressTryFn<'_>>,
    ) -> crate::tensor::Result<Vec<f32>> {
        let (h, oc, p, d) = (self.hidden, self.out_ch, self.patch, self.head_dim);
        let s = t / p;
        let dev = &self.device;
        // -- per-step-FIXED tensors (built once) --
        let enc_t = Tensor::from_vec_f32(enc.to_vec(), (enc_s, h))?.to_device(dev)?;
        let (cosv, sinv) = rope_tables(s, d, self.rope_theta);
        let cos = Tensor::from_vec_f32(cosv, (s, d / 2))?.to_device(dev)?;
        let sin = Tensor::from_vec_f32(sinv, (s, d / 2))?.to_device(dev)?;
        // sliding-window additive mask (matches the eager even-layer build exactly).
        let win = self.sliding_window;
        let mut mv = vec![0f32; s * s];
        for i in 0..s {
            for j in 0..s {
                if (i as i64 - j as i64).unsigned_abs() as usize > win {
                    mv[i * s + j] = f32::NEG_INFINITY;
                }
            }
        }
        let swa_mask = Tensor::from_vec_f32(mv, (s, s))?.to_device(dev)?;
        // style-morph cross-attn bias (same construction as `cross_attn_forward`).
        let morph_mask = match &self.xattn_morph {
            Some(m) if m.enc_bounds.len() >= 3 && *m.enc_bounds.last().unwrap() == enc_s => {
                const LOG: f32 = 24.0;
                let n = m.enc_bounds.len() - 1;
                let mut tok_style = vec![0usize; enc_s];
                for r in 0..n {
                    for j in m.enc_bounds[r]..m.enc_bounds[r + 1] {
                        tok_style[j] = r;
                    }
                }
                let mut data = vec![0f32; s * enc_s];
                for si in 0..s {
                    let pp = (si as f32 + 0.5) / s as f32 * n as f32;
                    for j in 0..enc_s {
                        let w = (1.0 - (pp - (tok_style[j] as f32 + 0.5)).abs()).max(0.0);
                        data[si * enc_s + j] = LOG * (w - 1.0);
                    }
                }
                Some(Tensor::from_vec_f32(data, (s, enc_s))?.to_device(dev)?)
            }
            _ => None,
        };
        // permuted proj weights (the eager path rebuilds these every step).
        let proj_in_w = self
            .proj_in
            .w
            .transpose(1, 2)?
            .contiguous()?
            .reshape((h, self.in_ch * p))?;
        let proj_out_w = self
            .proj_out
            .w
            .transpose(0, 2)?
            .contiguous()?
            .reshape((oc * p, h))?;
        let fx = DitFixed {
            enc: enc_t,
            cos,
            sin,
            swa_mask,
            morph_mask,
            proj_in_w,
            proj_out_w,
        };
        // Classifier-free guidance: a SECOND per-step-fixed set for the unconditional
        // enc (the null caption/lyric encoding), sharing the geometry-only tensors
        // (cos/sin/SWA/proj weights) and carrying no style-morph bias. Built only when
        // `cfg > 1.0` and an uncond enc is supplied; absent ⟹ the validated single-pass
        // path runs byte-for-byte unchanged.
        let fx_un: Option<(DitFixed, usize)> = match uncond {
            Some((uenc, uenc_s)) if sp.cfg > 1.0 => {
                let uenc_t = Tensor::from_vec_f32(uenc.to_vec(), (uenc_s, h))?.to_device(dev)?;
                Some((
                    DitFixed {
                        enc: uenc_t,
                        cos: fx.cos.clone(),
                        sin: fx.sin.clone(),
                        swa_mask: fx.swa_mask.clone(),
                        morph_mask: None,
                        proj_in_w: fx.proj_in_w.clone(),
                        proj_out_w: fx.proj_out_w.clone(),
                    },
                    uenc_s,
                ))
            }
            _ => None,
        };
        let proj_in_b = self.proj_in.b.as_ref().unwrap();
        let proj_out_b = self
            .proj_out
            .b
            .as_ref()
            .unwrap()
            .flatten_all()?
            .to_vec1_f32()?;

        // -- Euler trajectory (only xt / tproj / temb change per step) --
        let sched = sp.schedule();
        let rescale = crate::inference::model::acestep::pipeline::omega_rescale(sp.omega_scale);
        let mut apg_avg: Option<Vec<f64>> = None;
        let mut x = noise.to_vec(); // [T,64] token-major
        let unpatch_bias = |proj: &Tensor| -> crate::tensor::Result<(Vec<f32>, usize)> {
            let proj_host = proj.flatten_all()?.to_vec1_f32()?; // [oc.P, S] flat[j.S+s]
            let (mut vel, t_out) = unpatchify(&proj_host, oc, s, p); // [oc.T] channel-major
            for c in 0..oc {
                for ti in 0..t_out {
                    vel[c * t_out + ti] += proj_out_b[c];
                }
            }
            Ok((vel, t_out))
        };
        for step in 0..sched.len() {
            crate::inference::serve::progress::try_note(
                progress,
                crate::inference::serve::progress::phase::DENOISE,
                step + 1,
                sched.len(),
            )?;
            let t_curr = sched[step];
            let t_next = if step + 1 < sched.len() {
                sched[step + 1]
            } else {
                0.0
            };
            let dt = t_next - t_curr;
            let (temb, tproj) = self.time_embed_forward(t_curr)?;
            let tproj_t = Tensor::from_vec_f32(tproj, (6, h))?.to_device(dev)?;
            let temb_t = Tensor::from_vec_f32(temb, (1, h))?.to_device(dev)?;
            // proj_in: build channel-major input [in_ch.T] = concat(context, xt) on host
            // (cheap, 1x/step) -> patchify -> permuted proj_in linear, on-device.
            let input = crate::inference::model::acestep::pipeline::build_dit_input(context, &x, t);
            let patched = Tensor::from_vec_f32(input, (self.in_ch, t))?
                .to_device(dev)?
                .reshape((self.in_ch, s, p))?
                .transpose(0, 2)?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((self.in_ch * p, s))?
                .transpose(0, 1)?
                .contiguous()?; // [s, in_ch.P]
            let hid0 = patched.matmul_t(&fx.proj_in_w)?.broadcast_add(proj_in_b)?; // [S,H]
                                                                                   // N blocks + final AdaLN + proj_out, all on-device (the input/proj_in is the
                                                                                   // same for cond+uncond, so it is computed once and the hidden is cloned).
                                                                                   // Guidance fires only inside the centred window (and only when CFG is on); the
                                                                                   // uncond pass is skipped on the other steps, saving that compute.
            let (vel, t_out) = match (&fx_un, sp.scale_at(step)) {
                (Some((fxu, uenc_s)), Some(scale)) => {
                    let proj_c =
                        self.velocity_tail_fast(hid0.clone(), &tproj_t, &temb_t, &fx, s, enc_s)?;
                    let proj_u =
                        self.velocity_tail_fast(hid0, &tproj_t, &temb_t, fxu, s, *uenc_s)?;
                    let (vc, t_out) = unpatch_bias(&proj_c)?;
                    let (vu, _) = unpatch_bias(&proj_u)?;
                    let v = crate::inference::model::acestep::pipeline::apply_guidance(
                        &vc,
                        &vu,
                        oc,
                        t_out,
                        scale,
                        sp.cfg_type,
                        step,
                        sp.zero_steps,
                        &mut apg_avg,
                    );
                    (v, t_out)
                }
                _ => {
                    let proj = self.velocity_tail_fast(hid0, &tproj_t, &temb_t, &fx, s, enc_s)?;
                    unpatch_bias(&proj)?
                }
            };
            // Euler update on host (token-major): x += dt . velⁿᵐ, with optional omega mean-shift.
            if rescale == 1.0 {
                for c in 0..oc {
                    for ti in 0..t_out {
                        x[ti * oc + c] += dt * vel[c * t_out + ti];
                    }
                }
            } else {
                let sum: f64 = vel.iter().map(|&v| v as f64).sum();
                let m = dt * (sum / vel.len() as f64) as f32;
                for c in 0..oc {
                    for ti in 0..t_out {
                        let dx = dt * vel[c * t_out + ti];
                        x[ti * oc + c] += (dx - m) * rescale + m;
                    }
                }
            }
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests;
