//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Load one GGUF tensor -> a NATIVE F32 device tensor (mirrors the ACE-Step `dit_load_t`).
/// Used for the 14B path's F32 bias/norm/modulation/patch tensors: the compat-stack gguf
/// reader dequantizes to F32, then the values are rebuilt as a native tensor on `device`.
pub(super) fn wan_load_t(
    device: &Device,
    content: &crate::tensor::quantized::gguf_file::Content,
    file: &mut std::fs::File,
    name: &str,
) -> Result<Tensor> {
    use crate::tensor::{DType as CDType, Device as CDevice};
    let dq = content
        .tensor(file, name, &CDevice::Cpu)?
        .dequantize(&CDevice::Cpu)?
        .to_dtype(CDType::F32)?;
    let dims = dq.dims().to_vec();
    let v = dq.flatten_all()?.to_vec1::<f32>()?;
    Tensor::from_vec_f32(v, dims)?.to_device(device)
}

/// Build a quantized `WanLinear` for the 14B path: Q8 weight `[out,in]` on `dev` (via the
/// QVarBuilder, dequant-on-the-fly) plus its separate F32 `.bias` tensor.
pub(super) fn wan_qlin(
    vb: &QVarBuilder,
    content: &crate::tensor::quantized::gguf_file::Content,
    file: &mut std::fs::File,
    in_dim: usize,
    out_dim: usize,
    prefix: &str,
    dev: &Device,
    // The adapter, if this render is correcting the checkpoint.
    lora: Option<&SafeTensorsLoader>,
) -> Result<WanLinear> {
    let mut w = vb.qmatmul_on(in_dim, out_dim, &format!("{prefix}.weight"), dev)?;
    let mut b = wan_load_t(dev, content, file, &format!("{prefix}.bias"))?;
    if let Some(l) = lora {
        // ATTACHED, not merged. A quantised weight cannot absorb a correction without being
        // dequantised, added to and requantised, and that round trip loses more than the
        // correction is worth. Kept as a rank-r term it is exact, and two extra matmuls
        // against dimensions in the thousands is a low-percent tax.
        //
        // The checkpoint stores `down` as `[r, in]` and `up` as `[out, r]`; this layer wants
        // both in its own transposed convention, so they are transposed here. Reading them
        // straight through gives a shape that happens to multiply and a correction that is
        // the wrong way round.
        let dn = format!("diffusion_model.{prefix}.lora_down.weight");
        let up = format!("diffusion_model.{prefix}.lora_up.weight");
        if l.contains(&dn) && l.contains(&up) {
            let down = l
                .load_to(&dn, DType::F32, dev)?
                .transpose(0, 1)?
                .contiguous()?;
            let upt = l
                .load_to(&up, DType::F32, dev)?
                .transpose(0, 1)?
                .contiguous()?;
            w.add_lora(crate::tensor::lora::LoraDelta {
                down,
                up: upt,
                scale: 1.0,
            })?;
        }
        let db = format!("diffusion_model.{prefix}.diff_b");
        if l.contains(&db) {
            b = b.add(&l.load_to(&db, DType::F32, dev)?)?;
        }
    }
    Ok(WanLinear::Quant(QLinear::new(
        Weight::Quant(w),
        Some(b),
        in_dim,
        out_dim,
    )))
}

/// One WanAttentionBlock: AdaLN-Zero self-attn (gated) + cross-attn (plain residual,
/// affine norm3) + FFN (gated). norm1/norm2 are non-affine LayerNorm (no params).
pub(super) struct WanBlock {
    pub(super) modulation: Vec<f32>, // [6.dim] learned AdaLN bias, added to the time projection
    pub(super) norm3_w: Tensor,      // affine cross-attn LayerNorm weight [dim]
    pub(super) norm3_b: Tensor,      // affine cross-attn LayerNorm bias [dim]
    pub(super) sa_q: WanLinear,
    pub(super) sa_k: WanLinear,
    pub(super) sa_v: WanLinear,
    pub(super) sa_o: WanLinear,
    pub(super) sa_norm_q: Tensor,
    pub(super) sa_norm_k: Tensor, // qk-RMSNorm [dim]
    pub(super) ca_q: WanLinear,
    pub(super) ca_k: WanLinear,
    pub(super) ca_v: WanLinear,
    pub(super) ca_o: WanLinear,
    pub(super) ca_norm_q: Tensor,
    pub(super) ca_norm_k: Tensor,
    /// The IMAGE branch of cross-attention, present only on an image-to-video checkpoint.
    ///
    /// Wan's I2V conditions each block on two contexts, not one: the text, through the
    /// projections above, and a CLIP embedding of the frame the clip continues, through
    /// these. The two attentions are summed - the block reads what was asked for and what
    /// it is continuing at the same time. A text-to-video checkpoint has no such weights
    /// and this stays `None`, which is what makes the same block serve both.
    pub(super) ca_k_img: Option<WanLinear>,
    pub(super) ca_v_img: Option<WanLinear>,
    pub(super) ca_norm_k_img: Option<Tensor>,
    pub(super) ffn0: WanLinear,
    pub(super) ffn2: WanLinear,
    /// HeteroPlan segment device for this block - both variants plan per block, so this is
    /// the split card. All of this block's weights + its `ones`/norms live here; the forward
    /// stages the activation/context/rope tables onto this device (a no-op on one card).
    pub(super) device: Device,
    pub(super) ones: Tensor, // [dim] all-ones on `device`, for the non-affine LayerNorms
}

/// Projects a CLIP embedding of the frame a clip continues into the model's own width.
///
/// The checkpoint spells it `img_emb.proj.{0,1,3,4}`: a layer norm, a linear, a GELU (index
/// 2, which has no weights), a linear, a layer norm. Present only on image-to-video.
pub(super) struct ImgEmb {
    pub(super) ln1_w: Tensor,
    pub(super) ln1_b: Tensor,
    pub(super) fc1: WanLinear,
    pub(super) fc2: WanLinear,
    pub(super) ln2_w: Tensor,
    pub(super) ln2_b: Tensor,
}

impl ImgEmb {
    /// `[n, CLIP_WIDTH]` -> `[n, dim]`.
    pub(super) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.layer_norm(&self.ln1_w, Some(&self.ln1_b), EPS)?;
        let h = self.fc2.forward(&self.fc1.forward(&h)?.gelu()?)?;
        h.layer_norm(&self.ln2_w, Some(&self.ln2_b), EPS)
    }
}

/// Did every block end up on the processor?
///
/// Not a diagnostic: it is the difference between a render and an afternoon. A block that
/// runs in milliseconds on a card attends over every token of the clip on the host, on every
/// step, and at a large frame that is hours - during which the machine is unusable for
/// anything else. The caller uses this to say so BEFORE starting rather than to leave a
/// progress bar that never arrives.
impl WanDit {
    pub fn is_host_only(&self) -> bool {
        !self.blocks.is_empty() && self.blocks.iter().all(|b| !b.device.is_cuda())
    }
}

/// One place that turns "the client left" into the error every caller already handles.
///
/// Returning an error rather than a flag is deliberate: a render that is abandoned mid-block
/// has nothing worth finishing, and every caller on this path already propagates with `?`.
pub(super) fn cancelled(
    cancel: Option<&crate::inference::serve::cancel::CancelToken>,
) -> Result<()> {
    if cancel.is_some_and(|c| c.is_cancelled()) {
        return Err(crate::tensor::Error(
            "wan render cancelled (client disconnected)".into(),
        ));
    }
    Ok(())
}

/// Width of the CLIP ViT-H embedding the image branch consumes.
pub(super) const CLIP_WIDTH: usize = 1280;

/// What an image-to-video forward needs beyond the prompt: the frame it continues, twice.
///
/// Once as PIXELS-turned-semantics - a CLIP embedding, `[n, 1280]` flat - which reaches
/// every block's cross-attention and says WHAT is being continued. Once as LATENT, in the
/// twenty conditioning channels that ride alongside the noise into the patch embedding and
/// say WHERE everything is. A model given one and not the other has half a reference.
///
/// `cond20` is channel-major `[20, frames, h, w]`: four mask channels marking which latent
/// frames are given, then sixteen channels of the given frame's latent. The order is the
/// checkpoint's - mask first - and swapping it produces a video that is confidently wrong.
pub struct RefFrame<'a> {
    pub clip: &'a [f32],
    pub cond20: &'a [f32],
}

/// Conditioning channels an image-to-video input carries alongside the noise.
pub(super) const I2V_COND_CH: usize = I2V_IN_CH - OUT_CH; // 20 = 4 mask + 16 latent

// The Wan DiT (velocity-field predictor).

// Activation bytes of one denoise forward at `tokens` sequence length: ~12 concurrently-live
// `[tokens, dim]` F32 buffers (hidden, residual, q/k/v, attention out, ffn gate/up and their
// contiguous copies) plus the tiled-attention scores+softmax `[tile, tokens, heads]` (tile =
// the shared sdpa QUERY_TILE). Structural counts from the forward, scaled by the REQUEST.

/// Total GPU demand of the HOT component (DiT) for a clip of `tokens` patch tokens: checkpoint
/// bytes + the request-derived activation bytes. What the pressure protocol needs BEFORE load.
pub fn hot_demand_bytes(variant: WanVariant, tokens: usize) -> u64 {
    let (path, dim, heads, layers) = match variant {
        WanVariant::B14 => (wan_14b_dit_file(), 5120, 40, 40),
        _ => (wan_dit_file(), 1536, 12, 30),
    };
    // The checkpoint on disk, or what its own shape implies when it is not there -
    // rather than a byte count written by hand for one of the two variants.
    let file = std::fs::metadata(&path)
        .map(|m| m.len())
        .unwrap_or_else(|_| weight_bytes_from_shape(dim, layers));
    file + dit_activation_bytes(tokens, dim, heads)
}

/// Resident bytes a block stack of this shape implies, at half precision.
///
/// Per block: the four attention projections, then a feed-forward's matrices at
/// `MLP_RATIO` times the width. Used only when the checkpoint cannot be stat'd, so
/// the placement still plans against something derived from the model rather than
/// against a number that happened to match one file.
pub(super) fn weight_bytes_from_shape(dim: usize, layers: usize) -> u64 {
    const HALF_PRECISION: u64 = 2;
    let d = dim as u64;
    let per_layer = 4 * d * d + (2.0 * MLP_RATIO) as u64 * d * d;
    layers as u64 * per_layer * HALF_PRECISION
}

pub(crate) fn dit_activation_bytes(tokens: usize, dim: usize, heads: usize) -> u64 {
    // The shared derivation, which this function's own factor of 12 calibrated: it is
    // what `4 + 2 * mlp_ratio` returns at this family's ratio of 4, pinned by a test
    // there. What it ADDS is the measured allocator correction - the pool's
    // high-water mark follows what a denoise churns, not its instantaneous live set -
    // which this had always been missing, so video placements reserved about a third
    // of what a render takes.
    //
    // The TENSOR-CORE variant, because this family's self-attention goes through
    // `sdpa_tc`: its scores are BF16 and its tile is the larger one that buys. Charging
    // F32 here reserved twice the score buffer the forward allocates, and a reserve is
    // taken off the card before any block is placed - so a 14B on two EMPTY 16.6 GB cards
    // was handed 4.3 GB and put eight of its forty blocks on the host.
    crate::inference::place::runtime_demand::dit_activation_bytes_tc(
        tokens,
        dim,
        heads,
        MLP_RATIO,
        RESERVE_BYTES,
        materialised_score_seq(tokens),
    )
}

/// The sequence the score term is charged over: the whole clip.
///
/// It is tempting to charge only the attention that is still materialised. Self-attention
/// here goes through the flash kernel, which streams its scores through shared memory and
/// allocates nothing but its output, so on paper the clip-sized slab does not exist and
/// only the short cross attention onto the text should be charged. That was tried and it
/// made things worse, in a way that is easy to repeat: a reserve is not a description of
/// what a block allocates, it is what stops the planner putting ANOTHER block on the card.
/// Charging the text length instead of the clip freed enough budget for all forty blocks
/// to land on one device, and the forward then had nowhere to run - a frame size that had
/// been rendering fine came back as an out-of-memory. Charge the clip.
pub(crate) fn materialised_score_seq(tokens: usize) -> usize {
    tokens
}

/// Bytes per activation element the RESERVE is charged at.
///
/// Deliberately above the width the blocks carry their residual at. The stream is half
/// this, and halving the reserve to match looked like the obvious follow-on - it is not,
/// and the measurement is worth recording. What the blocks carry is only part of what a
/// render holds: the preparation, the patch embedding, the context and the head are all
/// still full width, so the peak did NOT halve when the stream did. Charging the stream
/// width freed enough budget for the planner to pack every block onto one card, and a
/// frame size that had been rendering fine came back as an out-of-memory - the reserve is
/// what stops another block landing on the card, so understating it is not a smaller
/// reserve, it is a bigger model on the same device.
///
/// It goes back to the stream width when something MEASURES the peak instead of deriving
/// it, and not before.
pub(crate) const RESERVE_BYTES: u64 = 4;

/// This family's feed-forward width as a multiple of the model width.
pub(super) const MLP_RATIO: f64 = 4.0;

pub struct WanDit {
    pub(super) patch_w: Tensor, // [dim,16,2,2] conv kernel (temporal kernel folded out)
    pub(super) patch_b: Tensor, // [dim]
    pub(super) text0: WanLinear,
    pub(super) text2: WanLinear, // text_embedding MLP (umT5 [.,4096] -> [.,dim])
    pub(super) time0: WanLinear,
    pub(super) time2: WanLinear, // time_embedding MLP (sinusoid[256] -> [dim])
    pub(super) time_proj: WanLinear, // time_projection [dim -> 6.dim]
    pub(super) blocks: Vec<WanBlock>,
    pub(super) head: WanLinear,    // [out_ch.prod(patch)=64, dim]
    pub(super) head_mod: Vec<f32>, // [2.dim] learned final-AdaLN bias, added to the time embedding
    pub(super) ones: Tensor,       // [dim] all-ones on `device`, for the final (head) LayerNorm
    pub(super) device: Device,     // primary card: patch/text/time/head + activation origin
    pub(super) dim: usize,
    pub(super) n_heads: usize,
    /// Present only on an image-to-video checkpoint: the projection that turns a CLIP
    /// embedding of the frame being continued into this model's width.
    pub(super) img_emb: Option<ImgEmb>,
}

/// Where each window begins, covering `frames` with windows of `window` and step `stride`.
///
/// The last window is pulled BACK to end on the final frame rather than allowed to run
/// short: a truncated window is a clip length the model has not seen, and the tail of a
/// render is exactly where that would show. Pulling it back costs a little more overlap on
/// the last join and nothing else.
pub(super) fn window_starts(frames: usize, window: usize, stride: usize) -> Vec<usize> {
    if frames <= window || window == 0 || stride == 0 {
        return vec![0];
    }
    let mut out = Vec::new();
    let mut at = 0usize;
    loop {
        let start = at.min(frames - window);
        out.push(start);
        if start + window >= frames {
            return out;
        }
        at = start + stride;
    }
}

impl WanDit {
    /// Load the configured variant (default 1.3B safetensors; 14B = Q8 GGUF, multi-GPU).
    /// `checkpoint` overrides the variant's own file - a fine-tune resolved by
    /// [`wan_dit_file_for`]. Its geometry is checked against the variant before anything
    /// is read, because a checkpoint of the wrong shape renders a video rather than
    /// failing.
    pub fn load_variant(
        variant: WanVariant,
        clip_tokens: usize,
        checkpoint: Option<&std::path::Path>,
        // Placing forty blocks of a 14B checkpoint takes tens of seconds, and it announced
        // only that it had begun - the longest silence in a render, and indistinguishable
        // from a hang. The blocks are placed in a plain loop, so the count was always there.
        progress: Option<&crate::inference::serve::progress::ProgressFn<'_>>,
    ) -> Result<Self> {
        match variant {
            WanVariant::B1_3 => Self::load(clip_tokens, checkpoint, progress),
            WanVariant::B14 => Self::load_gguf(clip_tokens, checkpoint, progress),
        }
    }

    /// Load the Wan 1.3B DiT from the resolved safetensors (weights -> F16), splitting the
    /// 30 blocks across the available devices with the SAME unified HeteroPlan the 14B path
    /// uses. This is the hot per-step component: it must be able to spread, with CPU as a
    /// spill of last resort rather than an all-or-nothing destination. Placing it whole
    /// meant that when no single card held it - a resident image model is enough - the
    /// entire denoise ran on CPU and the render timed out producing nothing.
    pub fn load(
        clip_tokens: usize,
        checkpoint: Option<&std::path::Path>,
        progress: Option<&crate::inference::serve::progress::ProgressFn<'_>>,
    ) -> Result<Self> {
        const DIM: usize = 1536;
        const N_LAYERS: usize = 30;
        const N_HEADS: usize = 12;
        const IN_CH: usize = 16;
        // A picked checkpoint is either a WHOLE DiT or a CORRECTION of one. The distilled
        // Wan variants ship as rank-32 LoRAs of a hundred megabytes rather than as a
        // three-gigabyte replacement, and the file says which it is: a LoRA carries
        // `lora_down` tensors and no `patch_embedding.weight`. Merging it at load rather
        // than evaluating it per call costs nothing at render time - the base weights are
        // dense, so `w + up*down` is exact and folds away.
        let picked = checkpoint.map(|p| p.to_path_buf());
        let (path, lora_path) = match picked {
            Some(p) if is_wan_lora(&p) => (wan_dit_file(), Some(p)),
            Some(p) => (p, None),
            None => (wan_dit_file(), None),
        };
        let ld = unsafe { SafeTensorsLoader::multi(&[&path]) }?;
        let lora_ld = match &lora_path {
            Some(p) => {
                eprintln!("[wan-dit] merging LoRA {}", p.display());
                Some(unsafe { SafeTensorsLoader::multi(&[p]) }?)
            }
            None => None,
        };
        // Apply this checkpoint's correction to one base tensor, if it has one.
        //
        // The naming is uniform, which is what makes this one function rather than a table:
        // a tensor `X.weight` is corrected by `X.diff` and, when X is a projection, by the
        // low-rank pair `X.lora_down` / `X.lora_up`; a tensor `X.bias` is corrected by
        // `X.diff_b`. There is no alpha in these files, so the scale is folded into the
        // factors already and the correction is exactly `up @ down`.
        let with_delta = |name: &str, t: Tensor, dv: &Device| -> Result<Tensor> {
            let Some(l) = lora_ld.as_ref() else {
                return Ok(t);
            };
            let (stem, delta) = match name.rsplit_once('.') {
                Some((s, "weight")) => (s, "diff"),
                Some((s, "bias")) => (s, "diff_b"),
                _ => return Ok(t),
            };
            let mut t = t;
            let dkey = format!("diffusion_model.{stem}.{delta}");
            if l.contains(&dkey) {
                t = t.add(&l.load_to(&dkey, DType::F32, dv)?)?;
            }
            if delta == "diff" {
                let dn = format!("diffusion_model.{stem}.lora_down.weight");
                let up = format!("diffusion_model.{stem}.lora_up.weight");
                if l.contains(&dn) && l.contains(&up) {
                    let d = l.load_to(&dn, DType::F32, dv)?; // [r, in]
                    let u = l.load_to(&up, DType::F32, dv)?; // [out, r]
                    t = t.add(&u.matmul(&d)?)?;
                }
            }
            Ok(t)
        };
        // Read a base tensor, correct it, and store it in the dtype it will be used in.
        // Without a LoRA this is the direct load it always was.
        let corrected = |name: &str, dt: DType, dv: &Device| -> Result<Tensor> {
            if lora_ld.is_none() {
                return ld.load_to(name, dt, dv);
            }
            let base = ld.load_to(name, DType::F32, dv)?;
            with_delta(name, base, dv)?.to_dtype(dt)
        };
        // REFUSE A CHECKPOINT OF THE WRONG SHAPE, before reading a single weight. An I2V
        // file or a 14B one shares this block stack and would load a long way in; what
        // comes out of a mismatched input is a video, just not the one that was asked for.
        if let Some(dims) = ld.shape_of("patch_embedding.weight") {
            if let Some(why) = check_wan_geometry(&dims, DIM, IN_CH) {
                return Err(crate::tensor::Error(format!("{}: {why}", path.display())));
            }
        }
        // ~2.8 GB F16 weights + the CLIP-DERIVED activation demand: the denoise keeps
        // ~12 live `[tokens, dim]` F32 buffers (hidden/residual/q/k/v/attn-out/ffn) plus the
        // tiled-attention scores `[tile, tokens, heads]` x2 (scores + softmax). Gating on
        // weights alone approved a card the denoise then OOMed on.
        let act = dit_activation_bytes(clip_tokens, DIM, N_HEADS);
        // The checkpoint's real size, not a figure copied from one build of it.
        let weights = std::fs::metadata(&path)
            .map(|m| m.len())
            .unwrap_or_else(|_| weight_bytes_from_shape(DIM, N_LAYERS));
        // Plan over the BLOCKS, not over the model as one unit. The activation demand is
        // the reserve, exactly as the 14B path computes it.
        let cudas = crate::inference::place::vram_manager::probe_under_pressure(act);
        let cuda_budget: Vec<(usize, u64)> = cudas.iter().map(|(i, fr, _)| (*i, *fr)).collect();
        // The budgets ALREADY exclude the activation reserve - `probe_under_pressure` probes
        // through `probe_cuda_devices`, which returns `stable_free - reserve`. Passing it
        // again subtracts it twice.
        let plan =
            HeteroPlan::calculate_with_kv_reserve(N_LAYERS, weights, &cuda_budget, &[], 1.0, 0, 0);
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
        // The stem and head are not blocks; they live on the fastest card, which is where
        // segment 0 starts. When one segment covers everything this is the old behaviour
        // and the placement is a numeric no-op.
        let device = cudas
            .first()
            .map(|(_, _, dv)| dv.clone())
            .unwrap_or(Device::Cpu);
        eprintln!(
            "[wan-dit] placement: {}",
            plan.segments
                .iter()
                .map(|s| format!("{}:{}-{}", s.kind, s.layer_start, s.layer_end))
                .collect::<Vec<_>>()
                .join(" ")
        );
        let f16_on =
            |name: &str, dv: &Device| -> Result<Tensor> { corrected(name, DType::F16, dv) };
        let f16 = |name: &str| -> Result<Tensor> { f16_on(name, &device) };
        // Load each projection in the dtype it will be MULTIPLIED in, once, instead of
        // widening it on every call. BF16 on a card because that is the tensor-core path and
        // the checkpoint is BF16 on disk anyway - there is no precision to give up. The host
        // has no BF16 matmul, so a CPU-resident block keeps the F16 store and the F32 widen
        // it always had. The bias is pre-widened for the same reason: it was being converted
        // per call to be added once.
        let lin_on = |prefix: &str, dv: &Device| -> Result<WanLinear> {
            let wdt = if dv.is_cuda() {
                DType::BF16
            } else {
                DType::F16
            };
            Ok(WanLinear::Dense(Linear::from_published(
                corrected(&format!("{prefix}.weight"), wdt, dv)?,
                Some(corrected(&format!("{prefix}.bias"), DType::F32, dv)?),
            )?))
        };
        let lin = |prefix: &str| -> Result<WanLinear> { lin_on(prefix, &device) };
        let patch_w = f16("patch_embedding.weight")?.reshape((DIM, 16, PATCH_H, PATCH_W))?;
        let ones = Tensor::from_vec_f32(vec![1.0f32; DIM], (DIM,))?.to_device(&device)?;
        let mut blocks = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            crate::inference::serve::progress::note(
                progress,
                crate::inference::serve::progress::phase::LOAD_MODEL,
                i,
                N_LAYERS,
            );
            let p = format!("blocks.{i}");
            let dv = layer_device(i);
            blocks.push(WanBlock {
                modulation: ld.load(&format!("{p}.modulation"))?.to_vec_f32(),
                norm3_w: f16_on(&format!("{p}.norm3.weight"), &dv)?,
                norm3_b: f16_on(&format!("{p}.norm3.bias"), &dv)?,
                sa_q: lin_on(&format!("{p}.self_attn.q"), &dv)?,
                sa_k: lin_on(&format!("{p}.self_attn.k"), &dv)?,
                sa_v: lin_on(&format!("{p}.self_attn.v"), &dv)?,
                sa_o: lin_on(&format!("{p}.self_attn.o"), &dv)?,
                sa_norm_q: f16_on(&format!("{p}.self_attn.norm_q.weight"), &dv)?,
                sa_norm_k: f16_on(&format!("{p}.self_attn.norm_k.weight"), &dv)?,
                ca_q: lin_on(&format!("{p}.cross_attn.q"), &dv)?,
                ca_k: lin_on(&format!("{p}.cross_attn.k"), &dv)?,
                ca_v: lin_on(&format!("{p}.cross_attn.v"), &dv)?,
                ca_o: lin_on(&format!("{p}.cross_attn.o"), &dv)?,
                ca_norm_q: f16_on(&format!("{p}.cross_attn.norm_q.weight"), &dv)?,
                ca_norm_k: f16_on(&format!("{p}.cross_attn.norm_k.weight"), &dv)?,
                // The 1.3B is text-to-video: it has no frame to continue and no weights
                // that would read one.
                ca_k_img: None,
                ca_v_img: None,
                ca_norm_k_img: None,
                ffn0: lin_on(&format!("{p}.ffn.0"), &dv)?,
                ffn2: lin_on(&format!("{p}.ffn.2"), &dv)?,
                // Each block normalises on its own card: `ones` must live there too, or the
                // first block off the stem's device faults on a cross-device operand.
                ones: Tensor::from_vec_f32(vec![1.0f32; DIM], (DIM,))?.to_device(&dv)?,
                device: dv,
            });
        }
        Ok(WanDit {
            patch_w,
            patch_b: f16("patch_embedding.bias")?,
            text0: lin("text_embedding.0")?,
            text2: lin("text_embedding.2")?,
            time0: lin("time_embedding.0")?,
            time2: lin("time_embedding.2")?,
            time_proj: lin("time_projection.1")?,
            blocks,
            head: lin("head.head")?,
            head_mod: ld.load("head.modulation")?.to_vec_f32(),
            ones,
            device,
            dim: DIM,
            n_heads: N_HEADS,
            img_emb: None,
        })
    }

    /// Load the Wan 14B DiT from the Q8_0 GGUF (city96 community quant). Architecture is
    /// IDENTICAL to the 1.3B - only the config (dim 5120, 40 blocks, 40 heads, ffn 13824)
    /// and the weight format/placement change: every weight MATRIX stays Q8_0 on device
    /// (`QMatMul`, dequant-on-the-fly), every bias/norm/modulation is the separate F32 GGUF
    /// tensor, and the 40 blocks are split across GPUs by the unified HeteroPlan (the model
    /// is ~16 GB Q8 -> genuinely needs a multi-GPU layer split, like the ACE-Step XL DiT).
    /// Wan self-attention is full MHA (no GQA), so q/k/v/o are all `[dim,dim]`.
    pub fn load_gguf(
        clip_tokens: usize,
        checkpoint: Option<&std::path::Path>,
        progress: Option<&crate::inference::serve::progress::ProgressFn<'_>>,
    ) -> Result<Self> {
        use crate::tensor::quantized::gguf_file;
        const DIM: usize = 5120;
        const N_LAYERS: usize = 40;
        const N_HEADS: usize = 40;
        const FFN: usize = 13824;
        const IN_CH: usize = 16;
        // A picked checkpoint is a whole DiT or a CORRECTION of one, exactly as on the 1.3B
        // path. The step-distillation adapters are published as rank-64 LoRAs of a few
        // hundred megabytes, and they are what turns forty guided steps into four unguided
        // ones - twenty times fewer forwards on the same weights.
        let picked = checkpoint.map(|p| p.to_path_buf());
        let (path, lora_path) = match picked {
            Some(p) if is_wan_lora(&p) => (base_for_lora(&p)?, Some(p)),
            Some(p) => (p, None),
            None => (wan_14b_dit_file(), None),
        };
        let path = path
            .to_str()
            .ok_or_else(|| crate::tensor::Error("wan-14b: non-UTF8 GGUF path".into()))?
            .to_string();
        let lora = match &lora_path {
            Some(p) => {
                eprintln!("[wan-14b] merging LoRA {}", p.display());
                Some(unsafe { SafeTensorsLoader::multi(&[p]) }?)
            }
            None => None,
        };

        // -- unified multi-GPU placement over the 40 blocks (pack-first per card, spill to
        //    CPU). Budget on the real Q8 file size (no F32 blow-up). KV reserve 0 (the DiT
        //    runs full-sequence, not autoregressive). Same mechanism as the ACE-Step DiT. --
        let model_size = std::fs::metadata(&path)
            .map(|m| m.len())
            .unwrap_or_else(|_| weight_bytes_from_shape(DIM, N_LAYERS));
        // Clip-derived activation demand at THIS request's dims - the first plan starts honest
        // (the oom_retry/vram_degrade escalation below remains the safety net).
        let act_reserve = dit_activation_bytes(clip_tokens, DIM, N_HEADS);
        // Base headroom only - NO hardcoded resolution-specific reserve. The 14B's activation
        // peak grows with resolution; rather than guess a magic number, the render wraps this in
        // `oom_retry` (see bin/wan_render.rs): a denoise/VAE OOM calls `vram_degrade()`, which
        // makes `probe_under_pressure` escalate the per-card reserve on the next attempt, re-planning
        // with a more balanced split (and CPU only as a last resort). Self-adapting to any
        // resolution / VRAM - same mechanism as the ACE-Step DiT.
        // The activation demand IS the reserve. A flat base sat on top of it, which at
        // this point is the same claim written twice: the allocator correction inside
        // `dit_activation_bytes` is what that base was standing in for.
        let reserve: u64 = act_reserve;
        let cudas = crate::inference::place::vram_manager::probe_under_pressure(reserve);
        let cuda_budget: Vec<(usize, u64)> = cudas.iter().map(|(i, fr, _)| (*i, *fr)).collect();
        let plan = HeteroPlan::calculate_with_kv_reserve(
            // The budgets below ALREADY exclude the reserve: `probe_under_pressure` probes through
            // `probe_cuda_devices`, which returns `stable_free - reserve`. Passing it again here
            // subtracted it TWICE - invisible at half a gigabyte, and fatal at twelve, where it
            // took both cards to zero usable and sent a whole video DiT to the host.
            N_LAYERS,
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
        eprintln!("[wan-14b] geometry: dim={DIM} blocks={N_LAYERS} heads={N_HEADS} hd={HEAD_DIM} ffn={FFN}");
        eprintln!(
            "[wan-14b] placement: {}",
            plan.segments
                .iter()
                .map(|s| format!("{}:{}-{}", s.kind, s.layer_start, s.layer_end))
                .collect::<Vec<_>>()
                .join(" ")
        );

        // QVarBuilder holds the Q8 weight matrices (host QTensors, uploaded per-block by
        // `qmatmul_on` to that block's device). A separate gguf Content reader supplies the
        // F32 bias/norm/modulation/patch tensors (dequantized -> native F32 on the target dev).
        let vb = crate::inference::cache::qvb::from_gguf_cached(&path, &device)?;
        let mut f = std::fs::File::open(&path)
            .map_err(|e| crate::tensor::Error(format!("wan-14b open {path}: {e}")))?;
        let c = gguf_file::read_mapped_file(&f)?;
        // The SAME refusal the safetensors path makes, for the format every community
        // fine-tune of this variant ships in. Without it a checkpoint of another shape -
        // an I2V, or the 1.3B - shares enough of the block stack to load a long way in
        // and then produce a video that is simply not the one asked for.
        // Which KIND of checkpoint this is, read off the weights rather than the name: an
        // image-to-video model carries an `img_emb` projection and takes 36 input channels
        // where text-to-video takes 16 (16 noise + 4 mask + 16 for the frame it continues).
        let is_i2v = c.tensor_infos.contains_key("img_emb.proj.1.weight");
        let in_ch = if is_i2v { I2V_IN_CH } else { IN_CH };
        if let Some(info) = c.tensor_infos.get("patch_embedding.weight") {
            if let Some(why) = check_wan_geometry(info.shape.dims(), DIM, in_ch) {
                return Err(crate::tensor::Error(format!("{path}: {why}")));
            }
        }
        if is_i2v {
            eprintln!(
                "[wan-14b] image-to-video checkpoint: {I2V_IN_CH} input channels, \
                       CLIP image conditioning"
            );
        }

        let qd = N_HEADS * HEAD_DIM; // q/k/v/o dim (full MHA, no GQA) == DIM
        let mut blocks = Vec::with_capacity(N_LAYERS);
        for l in 0..N_LAYERS {
            crate::inference::serve::progress::note(
                progress,
                crate::inference::serve::progress::phase::LOAD_MODEL,
                l,
                N_LAYERS,
            );
            let p = format!("blocks.{l}");
            let ld = layer_device(l);
            // Quantized weight `[out,in]` (Q8 on device) + its F32 bias.
            let ql = |in_dim: usize,
                      out_dim: usize,
                      nm: &str,
                      c: &gguf_file::Content,
                      f: &mut std::fs::File|
             -> Result<WanLinear> {
                wan_qlin(
                    &vb,
                    c,
                    f,
                    in_dim,
                    out_dim,
                    &format!("{p}.{nm}"),
                    &ld,
                    lora.as_ref(),
                )
            };
            blocks.push(WanBlock {
                modulation: wan_load_t(&ld, &c, &mut f, &format!("{p}.modulation"))?.to_vec_f32(),
                norm3_w: wan_load_t(&ld, &c, &mut f, &format!("{p}.norm3.weight"))?,
                norm3_b: wan_load_t(&ld, &c, &mut f, &format!("{p}.norm3.bias"))?,
                sa_q: ql(DIM, qd, "self_attn.q", &c, &mut f)?,
                sa_k: ql(DIM, qd, "self_attn.k", &c, &mut f)?,
                sa_v: ql(DIM, qd, "self_attn.v", &c, &mut f)?,
                sa_o: ql(qd, DIM, "self_attn.o", &c, &mut f)?,
                sa_norm_q: wan_load_t(&ld, &c, &mut f, &format!("{p}.self_attn.norm_q.weight"))?,
                sa_norm_k: wan_load_t(&ld, &c, &mut f, &format!("{p}.self_attn.norm_k.weight"))?,
                ca_q: ql(DIM, qd, "cross_attn.q", &c, &mut f)?,
                ca_k: ql(DIM, qd, "cross_attn.k", &c, &mut f)?,
                ca_v: ql(DIM, qd, "cross_attn.v", &c, &mut f)?,
                ca_o: ql(qd, DIM, "cross_attn.o", &c, &mut f)?,
                ca_norm_q: wan_load_t(&ld, &c, &mut f, &format!("{p}.cross_attn.norm_q.weight"))?,
                ca_norm_k: wan_load_t(&ld, &c, &mut f, &format!("{p}.cross_attn.norm_k.weight"))?,
                // Image-to-video only. The checkpoint carries these per block, or it does
                // not carry them at all - there is no half-conditioned variant.
                ca_k_img: if is_i2v {
                    Some(ql(DIM, qd, "cross_attn.k_img", &c, &mut f)?)
                } else {
                    None
                },
                ca_v_img: if is_i2v {
                    Some(ql(DIM, qd, "cross_attn.v_img", &c, &mut f)?)
                } else {
                    None
                },
                ca_norm_k_img: if is_i2v {
                    Some(wan_load_t(
                        &ld,
                        &c,
                        &mut f,
                        &format!("{p}.cross_attn.norm_k_img.weight"),
                    )?)
                } else {
                    None
                },
                ffn0: ql(DIM, FFN, "ffn.0", &c, &mut f)?,
                ffn2: ql(FFN, DIM, "ffn.2", &c, &mut f)?,
                ones: Tensor::from_vec_f32(vec![1.0f32; DIM], (DIM,))?.to_device(&ld)?,
                device: ld,
            });
        }
        // Built BEFORE the struct literal takes ownership of `device`.
        let img_emb = if is_i2v {
            Some(ImgEmb {
                ln1_w: wan_load_t(&device, &c, &mut f, "img_emb.proj.0.weight")?,
                ln1_b: wan_load_t(&device, &c, &mut f, "img_emb.proj.0.bias")?,
                fc1: wan_qlin(
                    &vb,
                    &c,
                    &mut f,
                    CLIP_WIDTH,
                    CLIP_WIDTH,
                    "img_emb.proj.1",
                    &device,
                    lora.as_ref(),
                )?,
                fc2: wan_qlin(
                    &vb,
                    &c,
                    &mut f,
                    CLIP_WIDTH,
                    DIM,
                    "img_emb.proj.3",
                    &device,
                    lora.as_ref(),
                )?,
                ln2_w: wan_load_t(&device, &c, &mut f, "img_emb.proj.4.weight")?,
                ln2_b: wan_load_t(&device, &c, &mut f, "img_emb.proj.4.bias")?,
            })
        } else {
            None
        };
        Ok(WanDit {
            patch_w: wan_load_t(&device, &c, &mut f, "patch_embedding.weight")?
                .reshape((DIM, in_ch, PATCH_H, PATCH_W))?,
            patch_b: wan_load_t(&device, &c, &mut f, "patch_embedding.bias")?,
            text0: wan_qlin(
                &vb,
                &c,
                &mut f,
                4096,
                DIM,
                "text_embedding.0",
                &device,
                lora.as_ref(),
            )?,
            text2: wan_qlin(
                &vb,
                &c,
                &mut f,
                DIM,
                DIM,
                "text_embedding.2",
                &device,
                lora.as_ref(),
            )?,
            time0: wan_qlin(
                &vb,
                &c,
                &mut f,
                FREQ_DIM,
                DIM,
                "time_embedding.0",
                &device,
                lora.as_ref(),
            )?,
            time2: wan_qlin(
                &vb,
                &c,
                &mut f,
                DIM,
                DIM,
                "time_embedding.2",
                &device,
                lora.as_ref(),
            )?,
            time_proj: wan_qlin(
                &vb,
                &c,
                &mut f,
                DIM,
                6 * DIM,
                "time_projection.1",
                &device,
                lora.as_ref(),
            )?,
            head: wan_qlin(
                &vb,
                &c,
                &mut f,
                DIM,
                OUT_CH * PATCH_H * PATCH_W,
                "head.head",
                &device,
                lora.as_ref(),
            )?,
            head_mod: wan_load_t(&device, &c, &mut f, "head.modulation")?.to_vec_f32(),
            ones: Tensor::from_vec_f32(vec![1.0f32; DIM], (DIM,))?.to_device(&device)?,
            blocks,
            device,
            dim: DIM,
            n_heads: N_HEADS,
            // The projection that turns a CLIP embedding of the continued frame into this
            // model's width. `img_emb.proj` is norm, linear, GELU (no weights), linear, norm.
            img_emb,
        })
    }

    /// Non-affine LayerNorm over the last dim (the final/head norm): ones weight, no bias.
    pub(super) fn ln(&self, x: &Tensor) -> Result<Tensor> {
        x.layer_norm(&self.ones, None, EPS)
    }

    /// Per-head-RMSNorm-then-reshape Q/K source -> `[1, n_heads, seq, HEAD_DIM]` (F32).
    /// The RMSNorm runs over the full projection (weight `[dim]`) BEFORE the head split,
    /// matching the checkpoint's `[dim]` norm weight.
    pub(super) fn to_heads_qknorm(
        &self,
        proj: &Tensor,
        norm_w: &Tensor,
        seq: usize,
    ) -> Result<Tensor> {
        let w = norm_w.to_dtype(DType::F32)?;
        // The normalisation widens its input to full precision internally and narrows the
        // result back, so it holds two full-width copies of the projection at once - on a
        // stream carried at half width that is four times the projection itself, and on a
        // video-sized sequence it was the largest transient in the block. Every token
        // normalises independently of the others, so slicing the token axis is exact, and
        // it bounds the transient the same way the feed-forward already bounds its
        // intermediate. Below the bound this is one call, unchanged.
        const NORM_WIDENING: f64 = 4.0;
        let chunk =
            crate::inference::model::acestep::ops::ffn_chunk_bytes_bounded(seq, NORM_WIDENING);
        let q = if chunk >= seq {
            proj.rms_norm(&w, EPS)?
        } else {
            let mut parts: Vec<Tensor> = Vec::with_capacity(seq.div_ceil(chunk));
            let mut at = 0usize;
            while at < seq {
                let n = chunk.min(seq - at);
                parts.push(proj.narrow(0, at, n)?.contiguous()?.rms_norm(&w, EPS)?);
                at += n;
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            Tensor::cat(&refs, 0)?
        };
        q.reshape((seq, self.n_heads, HEAD_DIM))?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .contiguous()
    }

    /// Reshape a V projection `[seq,dim]` -> `[1, n_heads, seq, HEAD_DIM]` (F32).
    pub(super) fn to_heads(&self, proj: &Tensor, seq: usize) -> Result<Tensor> {
        proj.reshape((seq, self.n_heads, HEAD_DIM))?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .contiguous()
    }

    /// Patch embed: conv3d with temporal kernel 1 ⟹ a per-frame 2x2/stride-2 conv2d.
    /// `latent` channel-major `[16.F.H.W]`. Returns the token sequence `[S, dim]` (F32,
    /// token order f-outer/h/w-inner) and the patched grid `(F, hp=H/2, wp=W/2)`.
    pub(super) fn patch_embed(
        &self,
        latent: &[f32],
        frames: usize,
        h: usize,
        w: usize,
        reference: Option<&RefFrame<'_>>,
    ) -> Result<(Tensor, usize, usize, usize)> {
        let (hp, wp) = (h / PATCH_H, w / PATCH_W);
        let kernel = self.patch_w.to_dtype(DType::F32)?; // [dim,in_ch,2,2]
        let frame_elems = h * w;
        let stride = frames * frame_elems;
        // An image-to-video checkpoint reads the noise AND the twenty conditioning channels
        // as one input: [16 noise | 4 mask | 16 reference latent], in that order.
        let cond = reference
            .filter(|_| self.img_emb.is_some())
            .map(|r| r.cond20);
        let in_ch = if cond.is_some() { I2V_IN_CH } else { OUT_CH };
        let mut rows: Vec<Tensor> = Vec::with_capacity(frames);
        for f in 0..frames {
            // gather frame f into [1,in_ch,H,W] (channel-major within the frame).
            let mut inp = vec![0f32; in_ch * frame_elems];
            for c in 0..OUT_CH {
                let src = &latent[c * stride + f * frame_elems..][..frame_elems];
                inp[c * frame_elems..][..frame_elems].copy_from_slice(src);
            }
            if let Some(c20) = cond {
                for c in 0..I2V_COND_CH {
                    let src = &c20[c * stride + f * frame_elems..][..frame_elems];
                    inp[(OUT_CH + c) * frame_elems..][..frame_elems].copy_from_slice(src);
                }
            }
            let x = Tensor::from_vec_f32(inp, (1, in_ch, h, w))?.to_device(&self.device)?;
            let o = x.conv2d(&kernel, 0, PATCH_H, 1, 1)?; // [1,dim,hp,wp]
                                                          // [dim, hp.wp] -> [hp.wp, dim]: one row per spatial token of this frame.
            let o = o
                .reshape((self.dim, hp * wp))?
                .transpose(0, 1)?
                .contiguous()?;
            rows.push(o);
        }
        let refs: Vec<&Tensor> = rows.iter().collect();
        let tokens = Tensor::cat(&refs, 0)?; // [F.hp.wp, dim]
        let tokens = tokens.broadcast_add(&self.patch_b.to_dtype(DType::F32)?)?;
        Ok((tokens, frames, hp, wp))
    }

    /// Sinusoidal timestep embedding (reference `sinusoidal_embedding_1d`): half=128,
    /// `freq_j = theta^(-j/half)`, embed = `[cos(t.freq), sin(t.freq)]` (cos first). No
    /// diffusion 1000x scale (the scheduler already passes t in the 0..1000 range).
    pub(super) fn time_sinusoid(t: f32) -> Vec<f32> {
        let half = FREQ_DIM / 2;
        let mut e = vec![0f32; FREQ_DIM];
        for j in 0..half {
            let freq = (ROPE_THETA as f32).powf(-(j as f32) / half as f32);
            let a = t * freq;
            e[j] = a.cos();
            e[j + half] = a.sin();
        }
        e
    }

    /// Build the 3D-RoPE per-token cos/sin tables `[S, HEAD_DIM/2 = 64]` (F32). The 64
    /// complex pairs are split 22/21/21 over the (temporal,height,width) axes; pair `j`
    /// of a token at grid `(f,h,w)` rotates by `coord . theta^(-(2k)/band_dim)` where
    /// `band_dim` is the per-axis rope dim (44/42/42) and `k` is the in-band index.
    /// Token order is f-outer/h/w-inner (matches `patch_embed`).
    pub(super) fn rope_tables(
        &self,
        frames: usize,
        hp: usize,
        wp: usize,
    ) -> Result<(Tensor, Tensor)> {
        let half = HEAD_DIM / 2; // 64
        let (nt, nh, nw) = (half - 2 * (half / 3), half / 3, half / 3); // 22, 21, 21
                                                                        // per-axis rope dims (the reference rope_params dim): nt*2=44, nh*2=42, nw*2=42.
        let band = |k: usize, band_dim: usize| ROPE_THETA.powf(-(2.0 * k as f64) / band_dim as f64);
        let s = frames * hp * wp;
        // Map a clip LONGER than the model's own into the temporal range it was trained on.
        //
        // The positional encoding is what tells a block where in time a token sits, and past
        // the trained length it is being asked about angles it has never seen. Left alone,
        // the model does not drift - it STOPS: a whole-clip thirty-second render came out
        // frozen, its frame 240 and its frame 470 the same picture. Compressing the temporal
        // positions so the whole clip spans the angular range of a trained-length one keeps
        // the encoding in distribution, which is what RIFLEx (arXiv 2502.15894) observes.
        //
        // Space is untouched: the frame size is what it always was, and only time is being
        // asked to stretch.
        // MEASURED WRONG, kept as a warning: scaling every temporal component by
        // native/actual squashes the encoding so hard that neighbouring frames become
        // indistinguishable, and a thirty-second clip came out as a repeating tile of the
        // same muzzle. RIFLEx does not do that - it lowers the frequency of the ONE
        // component whose period matches the trained length, which is the component that
        // wraps and therefore repeats, and leaves the rest alone.
        let t_scale = 1.0f64;
        let mut cos = vec![0f32; s * half];
        let mut sin = vec![0f32; s * half];
        for f in 0..frames {
            for hh in 0..hp {
                for ww in 0..wp {
                    let si = (f * hp + hh) * wp + ww;
                    let row = si * half;
                    for k in 0..nt {
                        let a = f as f64 * t_scale * band(k, nt * 2);
                        cos[row + k] = a.cos() as f32;
                        sin[row + k] = a.sin() as f32;
                    }
                    for k in 0..nh {
                        let a = hh as f64 * band(k, nh * 2);
                        cos[row + nt + k] = a.cos() as f32;
                        sin[row + nt + k] = a.sin() as f32;
                    }
                    for k in 0..nw {
                        let a = ww as f64 * band(k, nw * 2);
                        cos[row + nt + nh + k] = a.cos() as f32;
                        sin[row + nt + nh + k] = a.sin() as f32;
                    }
                }
            }
        }
        let cos = Tensor::from_vec_f32(cos, (s, half))?.to_device(&self.device)?;
        let sin = Tensor::from_vec_f32(sin, (s, half))?.to_device(&self.device)?;
        Ok((cos, sin))
    }

    /// Text embedding MLP: umT5 context `[S_txt, 4096]` -> `[TEXT_LEN, dim]` (F32). The
    /// context is zero-padded to TEXT_LEN BEFORE the MLP (reference convention; the
    /// embedded padding is non-zero and is attended to with no key mask).
    pub(super) fn text_embed(&self, ctx: &[f32], s_txt: usize) -> Result<Tensor> {
        let mut buf = vec![0f32; TEXT_LEN * 4096];
        let n = s_txt.min(TEXT_LEN);
        buf[..n * 4096].copy_from_slice(&ctx[..n * 4096]);
        let x = Tensor::from_vec_f32(buf, (TEXT_LEN, 4096))?.to_device(&self.device)?;
        let x = self.text0.forward(&x)?.gelu()?; // GELU(approximate='tanh')
        self.text2.forward(&x)
    }

    /// Self-attention: dense full-3D MHA with 3D-RoPE on q,k.
    /// `frame_tokens` is this clip's patches per frame - what makes the token index carry a
    /// time and a place, and therefore what lets the radial mask exist. Zero for a caller
    /// with no frame structure, which falls back to attending densely.
    pub(super) fn self_attn(
        &self,
        blk: &WanBlock,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        s: usize,
        frame_tokens: usize,
        grid_w: usize,
    ) -> Result<Tensor> {
        use crate::inference::model::acestep::ops::sdpa_tc as sdpa;
        let q = self
            .to_heads_qknorm(&blk.sa_q.forward(x)?, &blk.sa_norm_q, s)?
            .rope_i(cos, sin)?;
        let k = self
            .to_heads_qknorm(&blk.sa_k.forward(x)?, &blk.sa_norm_k, s)?
            .rope_i(cos, sin)?;
        let v = self.to_heads(&blk.sa_v.forward(x)?, s)?;
        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
        // Attention in a video volume decays with distance in space AND time, so the far
        // corners of the score matrix contribute almost nothing and cost most of the render.
        // The radial kernel skips those tiles outright - something the tiled cuBLAS path
        // cannot do, since a GEMM computes its whole rectangle. Measured at this clip's
        // shape: 1.43x on the 1.3B and 1.53x on the 14B against that path.
        let attn = match crate::inference::kernel::fused::flash_dit_bf16(
            &q,
            &k,
            &v,
            scale,
            frame_tokens,
            grid_w,
        ) {
            Ok(Some(o)) => o,
            // Anything the kernel does not cover - a head dimension it has no instantiation
            // for, a host tensor - takes the exact path, which is always there.
            _ => sdpa(&q, &k, &v, None, false, scale, 1.0)?,
        }; // [1,nh,S,HD]
        let ao = attn
            .transpose(1, 2)?
            .contiguous()?
            .reshape((s, self.n_heads * HEAD_DIM))?;
        blk.sa_o.forward(&ao)
    }

    /// Cross-attention: q from x, k/v from the text context. qk-RMSNorm, NO RoPE/mask.
    /// Cross-attention onto the text, plus - on an image-to-video checkpoint - onto the
    /// CLIP embedding of the frame this clip continues. `img` is `[n_img, dim]` already
    /// projected by `img_emb`, or `None` for text-to-video.
    pub(super) fn cross_attn(
        &self,
        blk: &WanBlock,
        xq: &Tensor,
        ctx: &Tensor,
        s: usize,
        s_txt: usize,
        img: Option<&Tensor>,
    ) -> Result<Tensor> {
        use crate::inference::model::acestep::ops::sdpa_tc as sdpa;
        let q = self.to_heads_qknorm(&blk.ca_q.forward(xq)?, &blk.ca_norm_q, s)?;
        let k = self.to_heads_qknorm(&blk.ca_k.forward(ctx)?, &blk.ca_norm_k, s_txt)?;
        let v = self.to_heads(&blk.ca_v.forward(ctx)?, s_txt)?;
        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
        let mut attn = sdpa(&q, &k, &v, None, false, scale, 1.0)?;
        // The image branch is a SECOND attention with the same queries, SUMMED - not a
        // longer context. Concatenating the two would put them in one softmax and let the
        // text and the reference frame compete for the same probability mass, which is not
        // what the weights were trained to expect.
        if let (Some(img), Some(ki), Some(vi), Some(nki)) =
            (img, &blk.ca_k_img, &blk.ca_v_img, &blk.ca_norm_k_img)
        {
            let n_img = img.dims()[0];
            let ki = self.to_heads_qknorm(&ki.forward(img)?, nki, n_img)?;
            let vi = self.to_heads(&vi.forward(img)?, n_img)?;
            attn = attn.add(&sdpa(&q, &ki, &vi, None, false, scale, 1.0)?)?;
        }
        let ao = attn
            .transpose(1, 2)?
            .contiguous()?
            .reshape((s, self.n_heads * HEAD_DIM))?;
        blk.ca_o.forward(&ao)
    }

    /// One AdaLN-Zero block. `e6` `[6.dim]` = the per-step time projection (added to the
    /// block's learned modulation); `x` `[S,dim]`; `ctx` `[TEXT_LEN,dim]`. The activation +
    /// context/rope tables are staged onto the block's device (a no-op on a single card; the
    /// host->device hop of `x` is what makes the 14B multi-GPU layer split work).
    pub(super) fn block_forward(
        &self,
        blk: &WanBlock,
        x_in: &Tensor,
        e6: &[f32],
        ctx_in: &Tensor,
        cos_in: &Tensor,
        sin_in: &Tensor,
        s: usize,
        s_txt: usize,
        frame_tokens: usize,
        grid_w: usize,
        img: Option<&Tensor>,
        cancel: Option<&crate::inference::serve::cancel::CancelToken>,
    ) -> Result<Tensor> {
        let dev = &blk.device;
        let x = x_in.to_device(dev)?;
        // The block runs at the width of the stream it is handed. Everything else it
        // combines with the activation - the modulation vectors, the text context, the
        // reference embedding - is brought to that width rather than forcing the
        // activation open to F32, which is what every mixed-dtype op would otherwise do.
        let dt = x.dtype();
        let cos = cos_in.to_device(dev)?;
        let sin = sin_in.to_device(dev)?;
        let ctx = ctx_in.to_device(dev)?.to_dtype(dt)?;
        let dim = self.dim;
        // AdaLN-Zero: six [dim] vectors = modulation + time-projection, split in order
        // [shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn].
        let vecd = |i: usize, add_one: bool| -> Result<Tensor> {
            let off = i * dim;
            let v: Vec<f32> = (0..dim)
                .map(|d| blk.modulation[off + d] + e6[off + d] + if add_one { 1.0 } else { 0.0 })
                .collect();
            Tensor::from_vec_f32(v, (1, dim))?
                .to_device(dev)?
                .to_dtype(dt)
        };
        let (shift_sa, scale_sa, gate_sa) = (vecd(0, false)?, vecd(1, true)?, vecd(2, false)?);
        let (shift_ffn, scale_ffn, gate_ffn) = (vecd(3, false)?, vecd(4, true)?, vecd(5, false)?);

        cancelled(cancel)?;
        // self-attn (non-affine norm -> modulate -> attn -> gate -> residual).
        let norm1 = x
            .layer_norm(&blk.ones, None, EPS)?
            .broadcast_mul(&scale_sa)?
            .broadcast_add(&shift_sa)?;
        let sa = self.self_attn(blk, &norm1, &cos, &sin, s, frame_tokens, grid_w)?;
        let x = x.add(&sa.broadcast_mul(&gate_sa)?)?;
        // cross-attn (affine norm3 -> attn -> plain residual, no gate).
        let norm3 = x.layer_norm(
            &blk.norm3_w.to_dtype(DType::F32)?,
            Some(&blk.norm3_b.to_dtype(DType::F32)?),
            EPS,
        )?;
        // The image context lives wherever it was built; a block on another card needs its
        // own copy, exactly as the text context does.
        let img_dev = match img {
            Some(t) => Some(t.to_device(dev)?.to_dtype(dt)?),
            None => None,
        };
        let x = x.add(&self.cross_attn(blk, &norm3, &ctx, s, s_txt, img_dev.as_ref())?)?;
        // A block is not an instant on every device. On the host, over a large frame, ONE of
        // them runs for minutes - so a check that only happens between blocks lets a render
        // whose client left keep the machine for as long as it takes to finish the one it is
        // in. Checked between the block's stages, and again per feed-forward slice below.
        cancelled(cancel)?;
        // FFN (non-affine norm -> modulate -> gelu-MLP -> gate -> residual).
        let norm2 = x
            .layer_norm(&blk.ones, None, EPS)?
            .broadcast_mul(&scale_ffn)?
            .broadcast_add(&shift_ffn)?;
        let ff = Self::ffn_chunked(blk, &norm2, cancel)?;
        x.add(&ff.broadcast_mul(&gate_ffn)?)
    }

    /// The feed-forward, over a BOUNDED slice of tokens at a time.
    ///
    /// Every token passes through this independently - two linears and a pointwise
    /// activation - so slicing the token axis is exact. Doing it whole materialises
    /// `[tokens, 4 * dim]` twice, and on a clip that is the largest thing in the forward:
    /// thirty seconds at 512^2 is ~124k tokens, so each intermediate is three gigabytes
    /// and the pair of them decides whether the model fits a card at all.
    ///
    /// Below the chunk this is one call, so an image-sized sequence behaves exactly as it
    /// did - same kernels, same launch count.
    pub(super) fn ffn_chunked(
        blk: &WanBlock,
        norm2: &Tensor,
        cancel: Option<&crate::inference::serve::cancel::CancelToken>,
    ) -> Result<Tensor> {
        let dims = norm2.dims().to_vec();
        let tokens = dims[dims.len() - 2];
        // Bounded in BYTES: the intermediate is `chunk x 4 x dim`, so a token count that is
        // comfortable on one model is gigabytes on a wider one at a larger frame.
        // Bound the intermediate against ONE token-major tensor rather than against a token
        // count. The shared helper stops bounding below its span, which is a threshold in
        // TOKENS - so at a width like this one a sequence well under that span still built
        // a multi-gigabyte intermediate, twice, and it was the single largest thing on the
        // card. Admitting `tokens / ratio` per call keeps the intermediate to the size of
        // the activation itself, whatever the width.
        let by_ratio = ((tokens as f64) / MLP_RATIO.max(1.0)).ceil() as usize;
        let chunk =
            crate::inference::model::acestep::ops::ffn_chunk_bytes_bounded(tokens, MLP_RATIO)
                .min(by_ratio.max(1));
        if chunk >= tokens {
            return blk.ffn2.forward(&blk.ffn0.forward(norm2)?.gelu()?);
        }
        let axis = dims.len() - 2;
        let mut parts: Vec<Tensor> = Vec::with_capacity(tokens.div_ceil(chunk));
        let mut at = 0usize;
        while at < tokens {
            cancelled(cancel)?;
            let n = chunk.min(tokens - at);
            let slice = norm2.narrow(axis, at, n)?.contiguous()?;
            parts.push(blk.ffn2.forward(&blk.ffn0.forward(&slice)?.gelu()?)?);
            at += n;
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        Tensor::cat(&refs, axis)
    }

    /// Full velocity prediction for ONE timestep. `latent` channel-major `[16.F.H.W]`,
    /// `t` the (unscaled) timestep, `ctx` row-major umT5 context `[s_txt.4096]`.
    /// Returns the flow velocity `[16.F.H.W]` (channel-major, same layout as `latent`).
    pub fn forward(
        &self,
        latent: &[f32],
        frames: usize,
        h: usize,
        w: usize,
        t: f32,
        ctx: &[f32],
        s_txt: usize,
    ) -> Result<Vec<f32>> {
        self.forward_cancellable(latent, frames, h, w, t, ctx, s_txt, None, None)
    }

    /// Does this checkpoint expect a frame to continue?
    ///
    /// An image-to-video model conditions on one, and asking it to render without is asking
    /// for a forward it has never seen. The caller has to know which kind it loaded, and the
    /// weights are the only honest source for that.
    pub fn wants_reference_frame(&self) -> bool {
        self.img_emb.is_some()
    }

    /// The clip length this model was TRAINED on, in latent frames.
    ///
    /// Wan 2.1 was trained at 81 output frames, and the temporal VAE folds four output
    /// frames into one latent frame past the first: `(81 - 1) / 4 + 1`. Attention is over
    /// the whole space-time volume, so this is not a soft preference - it is the length at
    /// which the positional encoding and the learnt temporal structure are in distribution.
    pub const NATIVE_LATENT_FRAMES: usize = 21;

    /// The narrow model this was measured on, and the share it needed.
    const NARROW_DIM: usize = 1536;
    const NARROW_OVERLAP: usize = 13;

    /// How much consecutive windows share, in latent frames, for a model `dim` wide.
    ///
    /// The overlap is the ONLY channel between windows: they all read the same input latent
    /// within a step and write their own velocity, so information crosses a boundary once
    /// per step, through these frames. Too little and a long clip drifts - reported as "it
    /// changes scenery and position very often", and measured as two frame-to-frame jumps
    /// above three times the average on a thirty-second 1.3B render.
    ///
    /// How much is needed depends on the MODEL, not on the clip. Measured at four steps,
    /// same prompt, same seed:
    ///   1.3B  5 -> 13 shared frames: mean frame-difference 6.12 -> 5.84, jumps 2 -> 0
    ///   14B   already continuous at 5, zero jumps either way, and 32% faster for it
    ///         (538 s against 787 s for fifteen seconds of video)
    /// A wider model holds a scene across a boundary that a narrow one loses, so the share
    /// is scaled by width against the narrow model that needed it, with a floor at the value
    /// the wide one was measured good at.
    pub(super) fn window_overlap(dim: usize) -> usize {
        (Self::NARROW_OVERLAP * Self::NARROW_DIM / dim.max(1))
            .clamp(5, Self::NATIVE_LATENT_FRAMES - 1)
    }

    /// Where each window of a `frames`-long clip begins.
    ///
    /// Public because the SAMPLER walks these, not the forward: a window has to be denoised
    /// to completion and handed to the next one as context, which is a loop around the
    /// sampler rather than inside a forward.
    pub fn window_starts_for(frames: usize, dim: usize) -> Vec<usize> {
        let window = Self::NATIVE_LATENT_FRAMES;
        window_starts(frames, window, window - Self::window_overlap(dim))
    }

    /// How many windows a clip of `frames` latent frames is denoised in - the unit of work
    /// a caller should count progress in.
    pub fn window_count(frames: usize, dim: usize) -> usize {
        Self::window_starts_for(frames, dim).len()
    }

    /// This model's width, for callers sizing windows or reserves against it.
    pub fn width(&self) -> usize {
        self.dim
    }

    /// The card the stem lives on - where a caller should build anything the forward reads
    /// whole, like the reference frame's latent or its CLIP embedding.
    pub fn stem_device(&self) -> &Device {
        &self.device
    }

    /// Denoise a clip LONGER than the model's window by running it over overlapping windows.
    ///
    /// Attention here is over the whole space-time volume, so a single pass costs the square
    /// of the clip length: measured, 5.8 times the tokens cost more than 26 times the time.
    /// That is what puts a thirty-second clip out of reach and a five-minute one out of the
    /// question - not the model, the shape of the sum.
    ///
    /// Running the model over overlapping windows of its OWN trained length makes the cost
    /// linear in duration instead. Each window is a clip the model has seen the like of, at
    /// its own positions, so nothing is extrapolated; the windows are blended where they
    /// overlap, weighted so each frame is taken mostly from the window that has the most
    /// context around it, which is what keeps the joins from showing.
    ///
    /// Below the window this is exactly one call, so an 81-frame clip runs the path it
    /// always did - same kernels, same launches, same numbers.
    /// `on_window` is called after each window finishes. A long clip is many windows inside
    /// ONE sampler step, and a step of a 14B model can run for minutes: without this, the
    /// only thing a client hears between steps is silence, which reads exactly like a hang.
    pub fn forward_windowed(
        &self,
        latent: &[f32],
        frames: usize,
        h: usize,
        w: usize,
        t: f32,
        ctx: &[f32],
        s_txt: usize,
        cancel: Option<&crate::inference::serve::cancel::CancelToken>,
        on_window: Option<&dyn Fn()>,
        reference: Option<&RefFrame<'_>>,
    ) -> Result<Vec<f32>> {
        let window = Self::NATIVE_LATENT_FRAMES;
        // A single pass over a long clip was TRIED and does not work, which is worth stating
        // because the cost argument for it is sound and will suggest itself again: with the
        // radial mask the whole clip costs about 3.8 times one window against the 8 windows
        // it is cut into, so it is cheaper AND has no seams to blend - and blending seams is
        // exactly what produces the morphing this was meant to cure.
        //
        // It fails on the model, not on the arithmetic. Wan is trained at 21 latent frames;
        // asked for 121 in one pass it does not drift, it degenerates - the render came back
        // as the same muzzle tiled down the frame. Compressing the temporal positions to map
        // the clip back into the trained range (the RIFLEx idea) made it worse, because
        // scaling every component leaves neighbouring frames indistinguishable.
        //
        // Curing the morphing therefore needs a model that was TRAINED to continue a clip -
        // an image-to-video checkpoint conditioned on the previous chunk's last frame -
        // rather than a cleverer way to cut a text-to-video one up.
        if frames <= window {
            let out =
                self.forward_cancellable(latent, frames, h, w, t, ctx, s_txt, cancel, reference)?;
            if let Some(f) = on_window {
                f();
            }
            return Ok(out);
        }
        // `h`/`w` are already the LATENT spatial dims, so one latent frame of one channel
        // is exactly h*w values and a window is a contiguous run within each channel.
        let plane = h * w;
        let overlap = Self::window_overlap(self.dim);
        let stride = window - overlap;
        let mut acc = vec![0f32; latent.len()];
        let mut wsum = vec![0f32; frames];

        for start in window_starts(frames, window, stride) {
            let mut sub = Vec::with_capacity(OUT_CH * window * plane);
            for c in 0..OUT_CH {
                let base = (c * frames + start) * plane;
                sub.extend_from_slice(&latent[base..base + window * plane]);
            }
            let v =
                self.forward_cancellable(&sub, window, h, w, t, ctx, s_txt, cancel, reference)?;
            if let Some(f) = on_window {
                f();
            }
            // Ramp the weight in and out across the overlap, so a frame covered twice is a
            // blend rather than a replacement and the transition is gradual in time.
            for i in 0..window {
                let edge = i.min(window - 1 - i) + 1;
                let gw = (edge as f32 / (overlap + 1) as f32).min(1.0);
                wsum[start + i] += gw;
                for c in 0..OUT_CH {
                    let dst = (c * frames + start + i) * plane;
                    let src = (c * window + i) * plane;
                    acc[dst..dst + plane]
                        .iter_mut()
                        .zip(&v[src..src + plane])
                        .for_each(|(a, s)| *a += s * gw);
                }
            }
        }
        for f in 0..frames {
            let n = if wsum[f] > 0.0 { wsum[f] } else { 1.0 };
            for c in 0..OUT_CH {
                let base = (c * frames + f) * plane;
                for p in 0..plane {
                    acc[base + p] /= n;
                }
            }
        }
        Ok(acc)
    }

    /// [`Self::forward`] that gives up between BLOCKS when the client is gone.
    ///
    /// The sampler already checks for cancellation between forwards, which is fine when a
    /// forward is short. It is not: one pass over every block at a long clip's token count
    /// runs for minutes, and longer still with blocks spilled to the host - so an abandoned
    /// render kept a card and half the cores for as long as the whole render would have
    /// taken. The block loop is where that time is actually spent, so that is where the
    /// question has to be asked. The check itself is an atomic load per block, against a
    /// block that is billions of operations.
    pub fn forward_cancellable(
        &self,
        latent: &[f32],
        frames: usize,
        h: usize,
        w: usize,
        t: f32,
        ctx: &[f32],
        s_txt: usize,
        cancel: Option<&crate::inference::serve::cancel::CancelToken>,
        // The frame this clip continues, for an image-to-video checkpoint. `None` on
        // text-to-video, and on a checkpoint that wants one it is the difference between
        // continuing a shot and starting a new one.
        reference: Option<&RefFrame<'_>>,
    ) -> Result<Vec<f32>> {
        let dim = self.dim;
        let (tokens, fr, hp, wp) = self.patch_embed(latent, frames, h, w, reference)?;
        let s = fr * hp * wp;
        let mut x = tokens; // [S,dim] F32

        // time embedding e [dim] and the 6-way projection e6 [6.dim] (per step).
        let sin = Self::time_sinusoid(t);
        let st = Tensor::from_vec_f32(sin, (1, FREQ_DIM))?.to_device(&self.device)?;
        let e = self.time2.forward(&self.time0.forward(&st)?.silu()?)?; // [1,dim]
        let e6 = self.time_proj.forward(&e.silu()?)?.to_vec_f32(); // [6.dim]
        let e_vec = e.to_vec_f32(); // [dim]

        let ctx_t = self.text_embed(ctx, s_txt)?; // [TEXT_LEN, dim]
                                                  // The image context is the same for every block, so it is projected once. It is
                                                  // built on the stem's device; each block stages its own copy, as it does the text.
        let img_ctx = match (&self.img_emb, reference) {
            // An empty embedding is not a zero-length context, it is NO context: building a
            // [0, 1280] tensor and attending to it would be a shape error, and passing zeros
            // would be a confident lie about what the frame contains.
            (_, Some(r)) if r.clip.is_empty() => None,
            (Some(emb), Some(r)) => {
                let n = r.clip.len() / CLIP_WIDTH;
                let t = Tensor::from_vec_f32(r.clip.to_vec(), (n, CLIP_WIDTH))?
                    .to_device(&self.device)?;
                Some(emb.forward(&t)?)
            }
            _ => None,
        };
        let (cos, sin_t) = self.rope_tables(fr, hp, wp)?;

        // The residual stream stays at full width here, and that is a MEASUREMENT, not an
        // oversight. The reference pipeline runs this denoiser at half width throughout -
        // norms, modulation and residual - so carrying it open to F32 looks like a pure
        // port artifact costing two copies per projection, and narrowing it did halve the
        // declared demand and moved a large frame from an all-host placement to a resident
        // one. End to end it was WORSE: the same clip that rendered at a little over a
        // second per pass came back at twenty-seven, with the allocator reclaiming its pool
        // on almost every block. Whatever the half-width stream saves inside a block, it
        // costs more in conversions where it meets the parts of the render that are still
        // full width - the preparation, the patch embedding, the context and the head.
        // Narrowing it is only worth retrying together with those, not on its own.
        for blk in &self.blocks {
            if cancel.is_some_and(|c| c.is_cancelled()) {
                return Err(crate::tensor::Error(
                    "wan render cancelled (client disconnected)".into(),
                ));
            }
            x = self.block_forward(
                blk,
                &x,
                &e6,
                &ctx_t,
                &cos,
                &sin_t,
                s,
                TEXT_LEN,
                hp * wp,
                wp,
                img_ctx.as_ref(),
                cancel,
            )?;
        }
        // Bring the activation back to the primary card for the head (no-op on one card).
        // The head reads it out to host floats, so it comes back to full width here.
        let x = x.to_device(&self.device)?.to_dtype(DType::F32)?;

        // Head: final AdaLN (head.modulation + time embedding e) -> linear -> unpatchify.
        let shift_h: Vec<f32> = (0..dim).map(|d| self.head_mod[d] + e_vec[d]).collect();
        let scale_h: Vec<f32> = (0..dim)
            .map(|d| 1.0 + self.head_mod[dim + d] + e_vec[d])
            .collect();
        let shift_t = Tensor::from_vec_f32(shift_h, (1, dim))?.to_device(&self.device)?;
        let scale_t = Tensor::from_vec_f32(scale_h, (1, dim))?.to_device(&self.device)?;
        let xn = self
            .ln(&x)?
            .broadcast_mul(&scale_t)?
            .broadcast_add(&shift_t)?;
        let head_out = self.head.forward(&xn)?.to_vec_f32(); // [S.64] row-major

        // unpatchify (reference 'fhwpqrc->cfphqwr', patch_t=1 so p=0): the 64-vector of
        // token (f,h,w) maps element ((q.2+r).16 + c) -> velocity[c, f, h.2+q, w.2+r].
        let (out_h, out_w) = (hp * PATCH_H, wp * PATCH_W);
        let plane = fr * out_h * out_w;
        let mut vel = vec![0f32; OUT_CH * plane];
        for f in 0..fr {
            for hh in 0..hp {
                for ww in 0..wp {
                    let si = (f * hp + hh) * wp + ww;
                    let base = si * (OUT_CH * PATCH_H * PATCH_W);
                    for q in 0..PATCH_H {
                        for r in 0..PATCH_W {
                            for c in 0..OUT_CH {
                                let v = head_out[base + (q * PATCH_W + r) * OUT_CH + c];
                                let oh = hh * PATCH_H + q;
                                let ow = ww * PATCH_W + r;
                                vel[c * plane + f * (out_h * out_w) + oh * out_w + ow] = v;
                            }
                        }
                    }
                }
            }
        }
        Ok(vel)
    }
}

#[cfg(test)]
mod dump_tests {
    //! The ignored cases here need real weights, a device, or a reference dump on
    //! this machine; nothing about them is automatic. Run one by name with
    //!   cargo test --release -p loken --lib NAME -- --ignored --nocapture
    use super::*;

    /// Dump the real safetensors key scheme and shapes, so the load mapping is
    /// locked against the actual checkpoint (run with `--ignored --nocapture`).
    #[test]
    #[ignore]
    pub(super) fn dump_wan_dit_keys() {
        let path = wan_dit_file();
        eprintln!("wan dit file: {path:?} exists={}", path.is_file());
        let ld = unsafe { SafeTensorsLoader::multi(&[&path]) }.unwrap();
        let mut names = ld.names();
        names.sort();
        eprintln!("total tensors: {}", names.len());
        for n in &names {
            let is_block = n.starts_with("blocks.");
            let block0 = n.starts_with("blocks.0.");
            if !is_block || block0 {
                let t = ld.load(n).unwrap();
                eprintln!("{n}  {:?}  {:?}", t.dims(), t.dtype());
            }
        }
        let max_blk = names
            .iter()
            .filter_map(|n| n.strip_prefix("blocks."))
            .filter_map(|s| s.split('.').next())
            .filter_map(|s| s.parse::<usize>().ok())
            .max()
            .unwrap_or(0);
        eprintln!("max block index = {max_blk}");
    }

    /// Load the DiT and run ONE velocity forward on a SMALL latent, so the dense
    /// 3D attention stays tiny (64 tokens). Asserts the output shape + all-finite.
    #[test]
    #[ignore]
    pub(super) fn wan_dit_forward_sanity() {
        let (frames, h, w) = (1usize, 16usize, 16usize); // -> 1.8.8 = 64 tokens
        let model = WanDit::load(1280, None, None).expect("load wan dit");
        // deterministic small latent [16,F,H,W] and a short umT5 ctx [5,4096].
        let n_lat = 16 * frames * h * w;
        let latent: Vec<f32> = (0..n_lat).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let s_txt = 5usize;
        let ctx: Vec<f32> = (0..s_txt * 4096)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.02)
            .collect();

        let vel = model
            .forward(&latent, frames, h, w, 1000.0, &ctx, s_txt)
            .expect("forward");
        assert_eq!(vel.len(), 16 * frames * h * w, "velocity element count");
        assert!(
            vel.iter().all(|x| x.is_finite()),
            "velocity must be all-finite"
        );
        let (mn, mx) = vel
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &x| {
                (a.min(x), b.max(x))
            });
        let mean = vel.iter().sum::<f32>() / vel.len() as f32;
        eprintln!(
            "velocity [16,{frames},{h},{w}] = {} elems, min={mn:.4} max={mx:.4} mean={mean:.5}",
            vel.len()
        );
    }

    // Ref-diff harness: forward the SAME fixed inputs as the official torch WanModel
    // and dump the output for offline comparison.
    //   WAN_REF_DIFF_DIR=... cargo test -p loken --profile fast wan_dit_ref_diff -- --ignored --nocapture
    #[test]
    #[ignore]
    pub(super) fn wan_dit_ref_diff() {
        let dir = std::env::var("WAN_REF_DIFF_DIR").expect("set WAN_REF_DIFF_DIR");
        let read_f32 = |name: &str| -> Vec<f32> {
            let bytes = std::fs::read(format!("{dir}/{name}")).expect(name);
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };
        let lat = read_f32("in_lat_16x5x32x32.f32");
        let ctx = read_f32("in_ctx_11x4096.f32");
        let model = WanDit::load(1280, None, None).expect("load wan dit");
        let out = model
            .forward(&lat, 5, 32, 32, 500.0, &ctx, 11)
            .expect("forward");
        let n = out.len() as f32;
        let mean: f32 = out.iter().sum::<f32>() / n;
        let std = (out.iter().map(|z| (z - mean) * (z - mean)).sum::<f32>() / n).sqrt();
        let bytes: Vec<u8> = out.iter().flat_map(|z| z.to_le_bytes()).collect();
        std::fs::write(format!("{dir}/rust_out_t500.f32"), bytes).unwrap();
        println!("rust out: mean={mean:.5} std={std:.5} -> rust_out_t500.f32");
    }
}

#[cfg(test)]
mod checkpoint_selection_tests {
    use super::*;

    /// The base 1.3B fits the 1.3B, in either storage order.
    #[test]
    pub(super) fn the_matching_checkpoint_is_accepted() {
        assert!(check_wan_geometry(&[1536, 16, 1, 2, 2], 1536, 16).is_none());
        // The GGUF reader hands the dims back reversed; both must read the same.
        assert!(check_wan_geometry(&[2, 2, 1, 16, 1536], 1536, 16).is_none());
        assert!(check_wan_geometry(&[5120, 16, 1, 2, 2], 5120, 16).is_none());
    }

    /// An image-to-video checkpoint is refused with the reason, not loaded.
    ///
    /// This is the case that matters: I2V shares the block stack, so it loads a long way
    /// in before anything looks wrong, and what comes out of a wrong input assembly is a
    /// video - just not the one that was asked for.
    #[test]
    pub(super) fn an_image_to_video_checkpoint_says_what_it_is() {
        let why = check_wan_geometry(&[5120, 36, 1, 2, 2], 5120, 16)
            .expect("36 input channels must be refused");
        assert!(why.contains("36"), "{why}");
        assert!(why.contains("image-to-video"), "{why}");
        // And reversed, as the GGUF gives it.
        assert!(check_wan_geometry(&[2, 2, 1, 36, 5120], 5120, 16).is_some());
    }

    /// A 14B checkpoint selected as the 1.3B is refused, and the message says how to fix it.
    #[test]
    pub(super) fn a_checkpoint_of_the_wrong_width_is_refused() {
        let why = check_wan_geometry(&[5120, 16, 1, 2, 2], 1536, 16)
            .expect("a 5120-wide model is not the 1.3B");
        assert!(why.contains("5120") && why.contains("1536"), "{why}");
        assert!(
            why.contains("14"),
            "the message must say how to select the 14B: {why}"
        );
    }

    /// Something that is not a patch embedding at all.
    #[test]
    pub(super) fn an_unrecognisable_shape_is_refused_rather_than_guessed() {
        assert!(check_wan_geometry(&[], 1536, 16).is_some());
        assert!(check_wan_geometry(&[7], 1536, 16).is_some());
    }

    /// A 14B fine-tune's own file name must still route it to the 14B loader.
    ///
    /// `from_token` picks the variant by looking for "14" in the requested name, and the
    /// requested name is the file's variant tag - so a checkpoint whose tag loses that
    /// substring would be loaded as the 1.3B and refused by the geometry guard, with the
    /// user left holding a file that "does not work" for no visible reason.
    #[test]
    pub(super) fn a_14b_finetune_still_selects_the_14b_loader() {
        let tag = wan_checkpoint_tag("Wan.FusionX.14b.Q8_0");
        assert_eq!(
            tag, "wan-fusionx-14b",
            "quantisation tokens drop, the size does not"
        );
        assert!(matches!(WanVariant::from_token(&tag), WanVariant::B14));
        // And a 1.3B fine-tune must NOT be dragged to the 14B by a stray digit.
        let small = wan_checkpoint_tag("Wan.Photoreal.v2.0");
        assert!(
            matches!(WanVariant::from_token(&small), WanVariant::B1_3),
            "{small}"
        );
    }

    /// The stem of a dropped file decides which name selects it, through the SAME
    /// normalisation the Ray families use - so the two mechanisms cannot drift.
    #[test]
    pub(super) fn a_dropped_file_answers_to_its_own_name() {
        assert_eq!(
            wan_checkpoint_tag("Wan.Photoreal.v2.0"),
            wan_checkpoint_tag("wan-photoreal")
        );
        // A quantisation of the same tune shares its tag, so both are candidates and the
        // larger file wins - which is the full-precision one.
        assert_eq!(
            wan_checkpoint_tag("Wan.Photoreal.v2.0.q8_0"),
            wan_checkpoint_tag("Wan.Photoreal.v2.0")
        );
        // Different tunes do NOT collide.
        assert_ne!(
            wan_checkpoint_tag("Wan.Photoreal"),
            wan_checkpoint_tag("Wan.Anime")
        );
    }
}

#[cfg(test)]
mod window_schedule_tests {
    use super::{window_starts, WanDit};

    /// A clip the model can hold runs as ONE call, exactly as it did before windowing.
    #[test]
    pub(super) fn a_clip_within_the_window_is_a_single_pass() {
        for frames in [1usize, 5, 20, 21] {
            assert_eq!(window_starts(frames, 21, 16), vec![0], "frames={frames}");
        }
    }

    /// Every frame must be covered, or it would be divided by a zero weight and come out
    /// black - and the last window must END on the last frame, never run past or short.
    #[test]
    pub(super) fn every_frame_is_covered_and_the_last_window_lands_on_the_end() {
        let window = WanDit::NATIVE_LATENT_FRAMES;
        let stride = window - WanDit::window_overlap(1536);
        for frames in [22usize, 31, 37, 121, 300, 1201] {
            let starts = window_starts(frames, window, stride);
            let mut covered = vec![false; frames];
            for s in &starts {
                assert!(s + window <= frames, "window {s} runs past {frames}");
                for f in *s..s + window {
                    covered[f] = true;
                }
            }
            assert!(covered.iter().all(|c| *c), "frames={frames} left a gap");
            let last = *starts.last().unwrap();
            assert_eq!(
                last + window,
                frames,
                "frames={frames} does not end on the last"
            );
        }
    }

    /// The point of windowing: cost grows with DURATION, not with duration squared.
    #[test]
    pub(super) fn the_window_count_is_linear_in_the_clip_length() {
        let window = WanDit::NATIVE_LATENT_FRAMES;
        let stride = window - WanDit::window_overlap(1536);
        let short = window_starts(121, window, stride).len();
        let long = window_starts(1201, window, stride).len();
        // Ten times the clip, about ten times the windows - not a hundred times the work.
        let ratio = long as f64 / short as f64;
        assert!(
            (8.0..12.0).contains(&ratio),
            "windows scaled {ratio:.1}x for 10x the clip"
        );
    }

    /// Overlapping windows must not walk backwards or stall.
    #[test]
    pub(super) fn the_windows_advance() {
        let starts = window_starts(300, 21, 16);
        assert!(starts.windows(2).all(|p| p[1] > p[0]), "{starts:?}");
    }
}

#[cfg(test)]
mod checkpoint_tag_tests {
    use super::wan_checkpoint_tag;

    /// The defect this replaced: a split parameter count lost its leading digit, so a
    /// 1.3B checkpoint was offered in the picker as a 3B one.
    #[test]
    pub(super) fn a_split_parameter_count_is_put_back_together() {
        assert_eq!(
            wan_checkpoint_tag("Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32"),
            "wan-causvid-1.3b"
        );
    }

    /// Packaging is not identity: the quantisation, the export rank and the task suffix
    /// describe how a checkpoint was shipped, not which checkpoint it is.
    #[test]
    pub(super) fn packaging_drops_and_the_family_stays() {
        assert_eq!(
            wan_checkpoint_tag("Wan.FusionX.14b.Q8_0"),
            "wan-fusionx-14b"
        );
        assert_eq!(
            wan_checkpoint_tag("Wan.CausVid.14b.Q5_0"),
            "wan-causvid-14b"
        );
        // Two packagings of the same weights answer to the same name, so the resolver can
        // pick between them rather than treating them as different models.
        assert_eq!(
            wan_checkpoint_tag("Wan.FusionX.14b.Q8_0"),
            wan_checkpoint_tag("Wan.FusionX.14b.fp16")
        );
    }

    /// The 1.3B and the 14B of the SAME distillation are different models and must not
    /// collide - one of them would silently serve the other's requests.
    #[test]
    pub(super) fn sizes_of_one_family_stay_apart() {
        assert_ne!(
            wan_checkpoint_tag("Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32"),
            wan_checkpoint_tag("Wan.CausVid.14b.Q5_0")
        );
    }

    /// Every tag names a video model, including one whose stem is nothing but packaging.
    #[test]
    pub(super) fn a_tag_always_says_it_is_video() {
        for stem in [
            "Wan.FusionX.14b.Q8_0",
            "wan21_t2v_lora",
            "Wan",
            "diffusion_model",
        ] {
            assert!(wan_checkpoint_tag(stem).starts_with("wan"), "{stem}");
        }
    }
}

#[cfg(test)]
mod distilled_defaults_tests {
    use super::wan_distilled_defaults;
    use std::path::Path;

    /// A step-distilled checkpoint is not a faster way to the same picture: it was trained
    /// to be integrated in a handful of steps with NO guidance. Running one at the base
    /// model's forty steps and guidance 5 does not render it better, it renders it wrong -
    /// and costs ten times as much doing so. The file has to be able to say so.
    #[test]
    pub(super) fn a_distillation_declares_its_own_schedule() {
        let causvid = Path::new("/m/Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors");
        assert_eq!(wan_distilled_defaults(causvid), Some((4, 1.0)));
        let fusionx = Path::new("/m/Wan.FusionX.14b.Q8_0.gguf");
        assert_eq!(wan_distilled_defaults(fusionx), Some((8, 1.0)));
    }

    /// Guidance 1.0 is the point: it means the unconditional branch is never evaluated,
    /// which is half the forwards. A distillation that declared a schedule but kept
    /// guidance would give back the larger half of the saving.
    #[test]
    pub(super) fn every_distillation_turns_guidance_off() {
        for f in ["A_CausVid_x.safetensors", "b-fusionx-14b.gguf"] {
            let (steps, cfg) = wan_distilled_defaults(Path::new(f)).expect(f);
            assert!(steps <= 8, "{f}: {steps} steps is not a distillation");
            assert_eq!(cfg, 1.0, "{f} still pays for an unconditional branch");
        }
    }

    /// A checkpoint nobody has taught this about must keep the BASE schedule. Guessing a
    /// four-step schedule for an ordinary fine-tune would quietly ruin it.
    #[test]
    pub(super) fn an_unrecognised_checkpoint_keeps_the_base_schedule() {
        for f in [
            "/m/Wan.Photoreal.v2.0.safetensors",
            "/m/wan2.1-i2v-14b-480p-Q8_0.gguf",
            "/m/diffusion_pytorch_model.safetensors",
        ] {
            assert_eq!(wan_distilled_defaults(Path::new(f)), None, "{f}");
        }
    }

    /// Publishers do not agree on capitalisation, and a checkpoint that fails to be
    /// recognised is one that renders at ten times the cost with no error anywhere.
    #[test]
    pub(super) fn recognition_does_not_depend_on_case() {
        for f in [
            "CAUSVID.safetensors",
            "causvid.safetensors",
            "CausVid.safetensors",
        ] {
            assert_eq!(wan_distilled_defaults(Path::new(f)), Some((4, 1.0)), "{f}");
        }
    }
}

#[cfg(test)]
mod window_overlap_tests {
    use super::WanDit;

    /// Both measured points, reproduced from ONE anchor and the model's width.
    #[test]
    pub(super) fn the_share_follows_the_model_width() {
        assert_eq!(
            WanDit::window_overlap(1536),
            13,
            "the 1.3B needed 13 to stop drifting"
        );
        assert_eq!(
            WanDit::window_overlap(5120),
            5,
            "the 14B was already continuous at 5"
        );
    }

    /// A wider model never asks for MORE sharing than a narrower one - the whole point is
    /// that capacity replaces overlap, and an inverted curve would cost the biggest models
    /// the most redundant work for the least reason.
    #[test]
    pub(super) fn wider_never_needs_more() {
        let mut prev = usize::MAX;
        for dim in [768usize, 1024, 1536, 2048, 3072, 5120, 8192] {
            let ov = WanDit::window_overlap(dim);
            assert!(ov <= prev, "dim {dim}: {ov} > {prev}");
            prev = ov;
        }
    }

    /// The share must leave the window something to generate, and must never be zero -
    /// windows that share nothing have no channel between them at all.
    #[test]
    pub(super) fn the_share_stays_inside_the_window() {
        for dim in [1usize, 256, 1536, 5120, 100_000] {
            let ov = WanDit::window_overlap(dim);
            assert!(
                ov >= 5 && ov < WanDit::NATIVE_LATENT_FRAMES,
                "dim {dim} -> {ov}"
            );
        }
    }
}

#[cfg(test)]
mod i2v_tag_tests {
    use super::wan_checkpoint_tag;

    /// A checkpoint that CONTINUES a frame and one that does not are different models and
    /// must not share a name - one of them would silently serve the other's requests.
    #[test]
    pub(super) fn image_to_video_keeps_its_own_name() {
        assert_eq!(wan_checkpoint_tag("Wan.I2V.14b.Q8_0"), "wan-i2v-14b");
        assert_eq!(
            wan_checkpoint_tag("wan2.1-i2v-14b-480p-Q8_0"),
            "wan-i2v-14b-480p"
        );
        assert_ne!(
            wan_checkpoint_tag("Wan.I2V.14b.Q8_0"),
            wan_checkpoint_tag("Wan.FusionX.14b.Q8_0")
        );
    }

    /// `t2v` is the default kind and carries no information, so it still goes - otherwise
    /// every text-to-video checkpoint would answer to a longer name for nothing.
    #[test]
    pub(super) fn text_to_video_is_the_unmarked_case() {
        assert_eq!(
            wan_checkpoint_tag("Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32"),
            "wan-causvid-1.3b"
        );
    }
}

#[cfg(test)]
mod i2v_distill_tests {
    use super::{wan_checkpoint_tag, wan_distilled_defaults};
    use std::path::Path;

    /// The image-to-video distillation adapter, which is what makes that model usable at
    /// all: forty guided steps against four unguided ones is twenty times the forwards.
    #[test]
    pub(super) fn the_image_to_video_distillation_declares_four_steps_and_no_guidance() {
        let p = Path::new("/m/Wan.I2Vdistill.lora.safetensors");
        assert_eq!(wan_distilled_defaults(p), Some((4, 1.0)));
        let q = Path::new("/m/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors");
        assert_eq!(wan_distilled_defaults(q), Some((4, 1.0)));
    }

    /// It must not collide with the checkpoint it corrects, or one would serve the other's
    /// requests and the render would silently be the undistilled forty-step one.
    #[test]
    pub(super) fn the_adapter_and_its_base_have_different_names() {
        assert_ne!(
            wan_checkpoint_tag("Wan.I2Vdistill.lora"),
            wan_checkpoint_tag("Wan.I2V.14b.Q8_0")
        );
    }

    /// A plain checkpoint keeps the base schedule: guessing four steps for a model that
    /// was not distilled produces mush, which is exactly what forty-step weights do at
    /// eight.
    #[test]
    pub(super) fn an_undistilled_checkpoint_is_left_alone() {
        assert_eq!(
            wan_distilled_defaults(Path::new("/m/Wan.I2V.14b.Q8_0.gguf")),
            None
        );
    }
}

impl WanDit {
    /// Where this model's layers sit, by device.
    pub fn placement(&self) -> Vec<crate::inference::serve::progress::placement::Placed> {
        crate::inference::serve::progress::placement::runs(
            self.blocks.iter().map(|b| b.device.location()),
            0,
        )
    }
}
