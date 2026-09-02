//! Z-Image transformer (ZImageTransformer2DModel) on the NATIVE tensor
//! substrate - substrate independence, port of `model/zimage/reference.rs`
//! following the `inference/model/flux/dit.rs` recipe (commit 5dce16d).
//!
//! Mirrors the facade file struct-for-struct (TimestepEmbedder, the feed-forward,
//! QkNorm, RopeEmbedder, ZImageAttention, ZImageTransformerBlock, FinalLayer,
//! ZImageTransformer2DModel) with the math helpers (`apply_rotary_emb`,
//! `attention`, `timestep_embedding`, `patchify`/`unpatchify`,
//! `create_coordinate_grid`) ported onto `crate::tensor`.
//!
//! Deviations from the facade implementation (correctness-neutral):
//!   - of the facade perf caches only a shape-keyed per-image RoPE/mask
//!     bundle is kept (`rope_cache` below); TIMESTEP_FREQS_CACHE and the
//!     hoisted run-once context refiner are still recomputed per forward
//!     (the refiner runs on ~30 caption tokens - negligible next to the
//!     30 main layers on the image sequence);
//!   - attention runs the basic matmul+softmax path: no flash-attn /
//!     Metal-SDPA dispatch, but QUERY-CHUNKED like the facade (and unlike it
//!     the chunking also covers the masked path - production always passes
//!     an all-ones caption mask) so the seq² score matrix is bounded at
//!     1024² latents - bit-exact either way (softmax normalizes per row);
//!   - TWO weight loaders: the original GGUF `QVarBuilder` (qmatmul +
//!     dequantized f32 for norms/biases/pad-tokens, like the flux port) and
//!     [`ZImageTransformer2DModel::from_varbuilder`] over the DENSE native
//!     safetensors `VarBuilder` - the production Tongyi-MAI/Z-Image-Turbo
//!     checkpoint is F32 safetensors with no GGUF anywhere. Dense 2-D matmul
//!     weights load at the builder dtype (BF16 on CUDA, matching the facade
//!     production path) wrapped in [`layer::Linear`]; activations stay F32
//!     (the substrate's elementwise/broadcast kernels are F32-native; halves
//!     run via cast wrappers) and are cast to the weight dtype only at the
//!     matmul boundary, so the GEMM precision matches the facade while the
//!     elementwise tail is computed at F32. Norms/biases/pad tokens load at
//!     exact F32 (the file dtype), mirroring the GGUF path's `get_f32`;
//!   - missing substrate ops are COMPOSED from existing ones: `tanh` =
//!     2.sigmoid(2x)-1, `permute` = transpose chain, `stack` = unsqueeze+cat,
//!     `sub` = add(scale(-1)), `ones` = from_vec; coordinate grids and the
//!     per-axis RoPE cos/sin caches live CPU-side in f32/u32 (index_select
//!     host-bounces on CUDA anyway), but the PER-IMAGE tables/masks are
//!     moved to the compute device ONCE and cached across denoise steps
//!     (shape-keyed `rope_cache` - the flux `pe_cache` recipe; without it
//!     every broadcast against a CPU cos/sin host-bounced the whole stream);
//!   - attention masks are kept as f32 0/1 tensors (the facade round-trips
//!     through U8); the unified RoPE tables are built as
//!     `cat(index_select(...))` instead of `index_select(cat(ids))` - the
//!     two are identical elementwise (avoids a u32 `cat`);
//!   - engine wiring: the `generate_zimage` denoise loop drives
//!     [`ZImageTransformer2DModel::forward`] directly on the native
//!     substrate; the engine bridges ONCE before the loop (text-encoder
//!     products + initial latent) and ONCE after (the final latent, to the
//!     facade VAE). The earlier per-step `ZImageBridge` adapter (the
//!     `FluxBridge` recipe, commit fbc703a) is retired.

// `Config` is plain data (no tensor fields) - shared with the facade model.
// ==================== Config ====================

/// Z-Image Transformer configuration
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    #[serde(default = "default_patch_size")]
    pub all_patch_size: Vec<usize>,
    #[serde(default = "default_f_patch_size")]
    pub all_f_patch_size: Vec<usize>,
    /// The latent's width, which is the VAE's.
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    /// The transformer's width, and how many heads share it.
    #[serde(default = "default_dim")]
    pub dim: usize,
    /// The main stack's depth, and the depth of the two refiners that precede it - one over the
    /// noise tokens, one over the caption's.
    #[serde(default = "default_n_layers")]
    pub n_layers: usize,
    #[serde(default = "default_n_refiner_layers")]
    pub n_refiner_layers: usize,
    /// Query heads, and key heads when fewer.
    #[serde(default = "default_n_heads")]
    pub n_heads: usize,
    #[serde(default = "default_n_kv_heads")]
    pub n_kv_heads: usize,
    /// What the normalisations add before dividing.
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    /// Whether queries and keys are normalised before they meet.
    #[serde(default = "default_qk_norm")]
    pub qk_norm: bool,
    /// The width the caption arrives at, before the projection into `dim`.
    #[serde(default = "default_cap_feat_dim")]
    pub cap_feat_dim: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_t_scale")]
    pub t_scale: f64,
    #[serde(default = "default_axes_dims")]
    pub axes_dims: Vec<usize>,
    #[serde(default = "default_axes_lens")]
    pub axes_lens: Vec<usize>,
    /// Whether to use accelerated attention (CUDA flash-attn / Metal SDPA)
    /// Default is true, automatically selects optimal implementation per platform
    #[serde(default = "default_use_accelerated_attn")]
    pub use_accelerated_attn: bool,
}

// What the published Z-Image transformer states when its config file does not.
crate::serde_defaults! {
    default_use_accelerated_attn: bool = true;
    default_patch_size: Vec<usize> = vec![2];
    default_f_patch_size: Vec<usize> = vec![1];
    default_in_channels: usize = 16;
    default_dim: usize = 3840;
    default_n_layers: usize = 30;
    default_n_refiner_layers: usize = 2;
    default_n_heads: usize = 30;
    default_n_kv_heads: usize = 30;
    default_norm_eps: f64 = 1e-5;
    default_qk_norm: bool = true;
    default_cap_feat_dim: usize = 2560;
    default_rope_theta: f64 = 256.0;
    default_t_scale: f64 = 1000.0;
    default_axes_dims: Vec<usize> = vec![32, 48, 48];
    default_axes_lens: Vec<usize> = vec![1536, 512, 512];
}

/// An unstated field falls back to the preset, because the preset is where the field defaults
/// are read from in the first place.
impl Default for Config {
    fn default() -> Self {
        Self::z_image_turbo()
    }
}

impl Config {
    /// The published Turbo checkpoint, said once.
    ///
    /// Assembled from the same functions the `serde` attributes name rather than from a second
    /// list of the fifteen numbers: two lists are two things to keep right, and were they ever
    /// to disagree, a config file that omitted a field would quietly build a different model
    /// than the preset of the same name.
    pub fn z_image_turbo() -> Self {
        Self {
            all_patch_size: default_patch_size(),
            all_f_patch_size: default_f_patch_size(),
            in_channels: default_in_channels(),
            dim: default_dim(),
            n_layers: default_n_layers(),
            n_refiner_layers: default_n_refiner_layers(),
            n_heads: default_n_heads(),
            n_kv_heads: default_n_kv_heads(),
            norm_eps: default_norm_eps(),
            qk_norm: default_qk_norm(),
            cap_feat_dim: default_cap_feat_dim(),
            rope_theta: default_rope_theta(),
            t_scale: default_t_scale(),
            axes_dims: default_axes_dims(),
            axes_lens: default_axes_lens(),
            use_accelerated_attn: default_use_accelerated_attn(),
        }
    }

    /// The width one head works in: the model's width, shared out over the heads. A rotation,
    /// a qk-norm scale and the softmax temperature are all sized against this and not `dim`.
    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }

    /// The width the gated feed-forward opens out to: eight thirds of the model's, the third
    /// taken before the eight so the result is a multiple of eight (3840 -> 10240).
    pub fn hidden_dim(&self) -> usize {
        self.dim / 3 * 8
    }

    /// The width the timestep is carried at, and so the width every modulation projection
    /// reads: the embedding's own, unless the model itself is narrower than that.
    pub fn adaln_dim(&self) -> usize {
        adaln_dim(self.dim)
    }

    /// How many numbers one patch holds: the latent's width times the volume it is cut
    /// into. The patch embedder reads that many in and the way back out writes that many
    /// back, so it is ONE number - it has been spelled as `patch_dim` in one loader and as
    /// `out_channels` in another, and two names is how a number comes to be two numbers.
    pub fn patch_dim(&self) -> usize {
        self.in_channels * self.all_f_patch_size[0] * self.all_patch_size[0].pow(2)
    }

    /// Choose the attention path explicitly. The two answer identically and differ in how much
    /// of the score matrix they hold at once, so this is a debugging switch.
    pub fn set_use_accelerated_attn(&mut self, enabled: bool) {
        self.use_accelerated_attn = enabled;
    }
}

use crate::inference::place::dry_plan::{DeviceLoad, PlanLoad};
use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan};
use crate::tensor::layer;
use crate::tensor::layer::qlinear::{QLinear, QSwiGlu, Weight};
use crate::tensor::layer::{LayerNorm, RmsNorm};
use crate::tensor::quantized::QVarBuilder;
use crate::tensor::VarBuilder;
use crate::tensor::{DType, Device, Error, Result, Tensor, D};
use std::sync::Mutex;

// ==================== Constants (mirror the facade) ====================

/// AdaLN embedding dimension (256)
pub const ADALN_EMBED_DIM: usize = 256;

/// The width the timestep is carried at, for a model of the given width.
///
/// Written as a function of the width alone because two of its readers hold a width and no
/// [`Config`]: [`FinalLayer`] is constructed from its own dimensions, and the OpenCL block is
/// built from loose numbers. [`Config::adaln_dim`] is this, asked of a config.
pub fn adaln_dim(dim: usize) -> usize {
    dim.min(ADALN_EMBED_DIM)
}
/// Frequency embedding size for timestep encoding
pub const FREQUENCY_EMBEDDING_SIZE: usize = 256;
/// Max period for sinusoidal encoding
pub const MAX_PERIOD: f64 = 10000.0;

/// What building and running this checkpoint WOULD allocate, having allocated
/// nothing.
#[derive(Debug, Clone, Copy)]
pub struct DryForward {
    /// What the model holds once built: the weights, at the dtype they load at and
    /// at the shapes the FILES declare - not at shapes derived from a config beside
    /// them, which is how this family came to be weighed three gigabytes heavier
    /// than it is.
    pub weights: u64,
    /// What the ALLOCATOR holds for those weights, which is what a card reports and
    /// what a placement has to be charged. Equal to `weights` when the caller did not
    /// say what the pool reserves in - see [`dry_forward`]'s `alloc` argument.
    pub weights_held: u64,
    /// The high-water mark of one forward, beyond the weights. What a card must have
    /// free for a render of this geometry to run, as far as the tensor layer sees.
    pub forward: u64,
    /// Allocations the forward performed.
    pub allocations: u64,
    /// The forward read values a dry run cannot provide, so its shapes are only
    /// right if none of them depended on a number that was never computed.
    pub blind: bool,
}

/// Run the real load and the real forward on a device that counts instead of
/// allocating, and report what they would have taken.
///
/// The whole value is in what this function does NOT do: it does not describe the
/// forward, it runs it. `from_varbuilder` is the production constructor, `forward`
/// is the production forward, and the only difference from a render is the device
/// they are handed. A demand computed this way cannot drift from the computation it
/// is a demand for, because there is no second description to drift.
///
/// Costs a safetensors header parse and a walk of the model's shape algebra - no
/// file data is read, no card is touched, and nothing here can disturb a render in
/// flight.
/// What the ALLOCATOR holds for a load, replayed from the events the ledger kept.
///
/// The counted figure is what the program asked for; a card reports what its pool
/// reserved to answer, and between the two sit the chunks the pool takes and the holes
/// the load's own staging leaves in them. Measured on this family it is a few percent -
/// small next to the weights, and NOT small next to the forward those same few percent
/// have to be charged beside.
///
/// `None` for the allocator's shape means the caller could not say what the pool
/// reserves in, and then the counted figure stands: an unmeasured guess at the chunk
/// would under-state what the card gives up, and under-stating is what admits a
/// placement that does not fit.
fn held_after_load(
    led: &std::sync::Arc<crate::tensor::dry::DryDevice>,
    alloc: Option<(u64, u64)>,
    counted: u64,
) -> u64 {
    let Some((page, align)) = alloc else {
        return counted;
    };
    crate::tensor::heap::replay(&led.trace(), page, align)
        .held_at_end
        .max(counted)
}

pub fn dry_forward(
    cfg: &Config,
    files: &[&str],
    prefix: Option<&str>,
    dtype: DType,
    latent_h: usize,
    latent_w: usize,
    text_len: usize,
    alloc: Option<(u64, u64)>,
) -> Result<DryForward> {
    let device = crate::tensor::Device::dry();
    let led = device
        .dry_ledger()
        .ok_or_else(|| Error::msg("dry_forward: no ledger"))?
        .clone();
    // SAFETY: the same mmap contract every load of these files takes; only the
    // headers are read.
    // The load's events, when the caller has said what the pool reserves in: the
    // weights are charged at what the ALLOCATOR holds for them, not at what they count.
    if alloc.is_some() {
        led.record_trace();
    }
    let vb = unsafe { VarBuilder::from_files(files, dtype, &device) }?;
    let vb = match prefix {
        Some(p) => vb.pp(p),
        None => vb,
    };
    let model = ZImageTransformer2DModel::from_varbuilder(cfg, &vb)?;
    // Everything still held once the constructor's staging copies are gone.
    let weights = led.live_bytes();
    let weights_held = held_after_load(&led, alloc, weights);
    let x = Tensor::dry(
        &device,
        DType::F32,
        (1, cfg.in_channels, 1, latent_h, latent_w),
    )?;
    let t = Tensor::dry(&device, DType::F32, 1)?;
    let cap_feats = Tensor::dry(&device, DType::F32, (1, text_len, cfg.cap_feat_dim))?;
    let cap_mask = Tensor::dry(&device, DType::F32, (1, text_len))?;
    // Everything the load churned through is behind us; what follows is one
    // forward, measured against a card that already holds the weights.
    let resident = led.open_window();
    let out = model.forward(&x, &t, &cap_feats, &cap_mask)?;
    let peak = led.peak_bytes();
    drop(out);
    Ok(DryForward {
        weights,
        weights_held,
        forward: peak.saturating_sub(resident),
        allocations: led.allocations(),
        blind: led.read_absent_data(),
    })
}

/// The transformer BUILT WHERE A PLAN SAYS, on the port a dry run can walk.
///
/// The same two calls [`dry_forward_planned`] makes, in the same order: one VarBuilder
/// per device of the plan, then `build_across` with a slot per device and the slot each
/// block belongs to. That is the whole point of it existing beside the walk rather than
/// somewhere else - the model that LOADS and the model that was MEASURED are built by
/// the same code from the same plan, so a reserve cannot describe one arrangement while
/// another one runs.
///
/// `devices` is one entry per slot in the plan's own order, and `layer_slot` says which
/// slot each main block belongs to. A plan of one slot is the single-device build,
/// unchanged: `build_across` takes the same path it takes for `from_varbuilder`.
///
/// # Safety
/// The same mmap contract every load of these files takes.
pub unsafe fn from_files_placed(
    cfg: &Config,
    files: &[&str],
    prefix: Option<&str>,
    dtype: DType,
    devices: &[Device],
    layer_slot: &[usize],
) -> Result<ZImageTransformer2DModel> {
    if devices.is_empty() {
        return Err(Error::msg("from_files_placed: a plan with no devices"));
    }
    let mut slots: Vec<(Vb, Device)> = Vec::with_capacity(devices.len());
    for device in devices {
        // A block on the host weighs twice what the same block weighs on a card, which
        // is what the walk charges it - so the load has to agree, dtype for dtype.
        let dt = if device.is_cuda() { dtype } else { DType::F32 };
        let vb = unsafe { VarBuilder::from_files(files, dt, device) }?;
        let vb = match prefix {
            Some(p) => vb.pp(p),
            None => vb,
        };
        slots.push((Vb::Dense(vb), device.clone()));
    }
    ZImageTransformer2DModel::build_across(cfg, &slots[0].0, Some((&slots, layer_slot)))
}

/// What one forward of a CANDIDATE PLACEMENT would hold, card by card.
#[derive(Debug, Clone)]
pub struct DryPlacedForward {
    /// Per device of the plan, in the plan's order.
    pub load: PlanLoad,
    /// Allocations the forward performed, across every device.
    pub allocations: u64,
    /// The forward read values a dry run cannot provide - same caveat as
    /// [`DryForward::blind`].
    pub blind: bool,
}

/// Run the real load and the real forward of a CANDIDATE PLACEMENT and report what
/// each of its cards would hold.
///
/// This is [`dry_forward`] with the plan as a parameter, and the parameter is the
/// whole point: the peak is a property of the placement, so a number that does not
/// follow the placement cannot be used to choose one. Blocks are built on the dry
/// device of the segment they were assigned to, the stream moves between those devices
/// exactly where the plan says a boundary is, and each device's ledger ends up holding
/// its own weights and its own transients - which is what a card has to have free, as
/// against a total that no card ever sees.
///
/// The host segments of a plan load at `host_dtype` and the card segments at
/// `gpu_dtype`, because that is what the loaders do: a block on the host weighs twice
/// what the same block weighs on a card, and a placement weighed at one dtype
/// throughout would be wrong about the very case it is trying to avoid.
///
/// Costs one safetensors header parse per distinct device plus a walk of the shape
/// algebra: no file data, no card, nothing that can disturb a render in flight.
pub fn dry_forward_planned(
    cfg: &Config,
    files: &[&str],
    prefix: Option<&str>,
    gpu_dtype: DType,
    host_dtype: DType,
    latent_h: usize,
    latent_w: usize,
    text_len: usize,
    plan: &HeteroPlan,
    alloc: Option<(u64, u64)>,
) -> Result<DryPlacedForward> {
    if plan.segments.is_empty() {
        return Err(Error::msg("dry_forward_planned: a plan with no segments"));
    }
    // One device per distinct segment, in the plan's order. The FIRST is the stem's:
    // the embedders, the refiners and the way back out are built where the plan starts,
    // which is where the loader builds them.
    let mut kinds: Vec<DeviceKind> = Vec::new();
    for seg in &plan.segments {
        if !kinds.contains(&seg.kind) {
            kinds.push(seg.kind);
        }
    }
    let mut slots: Vec<(Vb, Device)> = Vec::with_capacity(kinds.len());
    let mut ledgers = Vec::with_capacity(kinds.len());
    for kind in &kinds {
        let device = Device::dry();
        let led = device
            .dry_ledger()
            .ok_or_else(|| Error::msg("dry_forward_planned: no ledger"))?
            .clone();
        if alloc.is_some() {
            led.record_trace();
        }
        let dtype = if matches!(kind, DeviceKind::Cuda(_)) {
            gpu_dtype
        } else {
            host_dtype
        };
        // SAFETY: the same mmap contract every load of these files takes; only the
        // headers are read.
        let vb = unsafe { VarBuilder::from_files(files, dtype, &device) }?;
        let vb = match prefix {
            Some(p) => vb.pp(p),
            None => vb,
        };
        slots.push((Vb::Dense(vb), device));
        ledgers.push(led);
    }
    // Which slot each main block belongs to. A block no segment claims stays with the
    // stem rather than being dropped: a plan that does not cover its own stack is a
    // bug in the plan, and measuring it as if the block did not exist would hide it.
    let layer_slot: Vec<usize> = (0..cfg.n_layers)
        .map(|l| {
            plan.segments
                .iter()
                .find(|s| l >= s.layer_start && l < s.layer_end)
                .and_then(|s| kinds.iter().position(|k| *k == s.kind))
                .unwrap_or(0)
        })
        .collect();

    let placement = Some((&slots[..], &layer_slot[..]));
    let model = ZImageTransformer2DModel::build_across(cfg, &slots[0].0, placement)?;
    let weights: Vec<u64> = ledgers
        .iter()
        .map(|l| held_after_load(l, alloc, l.live_bytes()))
        .collect();

    let stem = &slots[0].1;
    let x = Tensor::dry(
        stem,
        DType::F32,
        (1, cfg.in_channels, 1, latent_h, latent_w),
    )?;
    let t = Tensor::dry(stem, DType::F32, 1)?;
    let cap_feats = Tensor::dry(stem, DType::F32, (1, text_len, cfg.cap_feat_dim))?;
    let cap_mask = Tensor::dry(stem, DType::F32, (1, text_len))?;
    // Every ledger restarts its high-water mark from what it holds now, so what each
    // reports next is one forward measured against a card that already has its share
    // of the weights resident.
    let resident: Vec<u64> = ledgers.iter().map(|l| l.open_window()).collect();
    let out = model.forward(&x, &t, &cap_feats, &cap_mask)?;
    let devices: Vec<DeviceLoad> = kinds
        .iter()
        .enumerate()
        .map(|(n, kind)| DeviceLoad {
            kind: *kind,
            weights: weights[n],
            forward: ledgers[n].peak_bytes().saturating_sub(resident[n]),
        })
        .collect();
    drop(out);
    Ok(DryPlacedForward {
        load: PlanLoad { devices },
        allocations: ledgers.iter().map(|l| l.allocations()).sum(),
        blind: ledgers.iter().any(|l| l.read_absent_data()),
    })
}

// ---------------------------------------------------------------------------
// Weight builder: GGUF-quantized (QVarBuilder) or dense safetensors
// (native VarBuilder). One constructor tree serves both - the
// loaders only differ in how `linear`/`get_f32` resolve a tensor.
// ---------------------------------------------------------------------------

/// Prefix-walking builder over either weight source.
pub enum Vb {
    /// GGUF: 2-D weights as [`QMatMul`], everything else dequantized F32.
    Q(QVarBuilder),
    /// Dense safetensors: 2-D matmul weights at the builder dtype (BF16 on
    /// CUDA - the facade production dtype), norms/biases/pad tokens at
    /// exact F32 (the checkpoint dtype).
    Dense(VarBuilder),
}

impl Vb {
    pub fn pp<S: ToString>(&self, s: S) -> Self {
        match self {
            Self::Q(vb) => Self::Q(vb.pp(s)),
            Self::Dense(vb) => Self::Dense(vb.pp(s)),
        }
    }

    /// `{prefix}.{name}` as an exact-F32 tensor on the compute device
    /// (norm scales, biases, pad tokens).
    fn get_f32<S: Into<crate::tensor::Shape>>(&self, shape: S, name: &str) -> Result<Tensor> {
        match self {
            Self::Q(vb) => vb.get_f32(shape, name),
            Self::Dense(vb) => vb.to(DType::F32, vb.device()).get(shape, name),
        }
    }
}

// ---------------------------------------------------------------------------
// Weight loading. The projection itself is `tensor::layer::qlinear::QLinear`.
// ---------------------------------------------------------------------------

fn weight(in_dim: usize, out_dim: usize, vb: &Vb) -> Result<Weight> {
    match vb {
        Vb::Q(vb) => Ok(Weight::Quant(vb.qmatmul(in_dim, out_dim, "weight")?)),
        Vb::Dense(vb) => {
            // Stage host-side: F32 load -> transpose -> cast -> ONE upload of
            // the final blob. Building `Linear::new` on a device-resident
            // BF16 weight materializes an F32 cast + transposed copy per
            // tensor on the GPU; for the 12 GB transformer those freed
            // transients stay retained in the CUDA mempool and starved the
            // device (1024² gen OOM with <1 GB reported free).
            let w = vb
                .to(DType::F32, &crate::tensor::Device::Cpu)
                .get((out_dim, in_dim), "weight")?;
            let wt = w.transpose(0, 1)?;
            let wt = if vb.dtype() == DType::F32 {
                wt
            } else {
                wt.to_dtype(vb.dtype())?
            };
            let wt = wt.to_device(vb.device())?;
            Ok(Weight::Dense(
                layer::Linear::from_transposed(wt, None)?,
                vb.dtype(),
            ))
        }
    }
}

/// The fused research dialect packs Q|K|V into one `qkv` weight `[q+2kv, in]`
/// instead of exporting `to_q`/`to_k`/`to_v`. Row-slice it into the three
/// projections: a pure copy at the file's precision, never a requantization
/// (requantizing a fused segment is how a V projection loses precision).
/// Staged host-side like [`weight`] - slice, transpose and cast on the CPU, then
/// one upload per projection.
fn fused_qkv(
    in_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    vb: &Vb,
) -> Result<(QLinear, QLinear, QLinear)> {
    match &vb.pp("qkv") {
        // GGUF exports of this dialect have not been observed; slicing a
        // quantized blob would need block-aligned cuts, so let the caller's
        // error surface rather than guess.
        Vb::Q(_) => Err(Error::msg("fused qkv: only the dense dialect is supported")),
        Vb::Dense(d) => {
            let w = d
                .to(DType::F32, &crate::tensor::Device::Cpu)
                .get((q_dim + 2 * kv_dim, in_dim), "weight")?;
            let mut parts = Vec::with_capacity(3);
            for (offset, rows) in [(0, q_dim), (q_dim, kv_dim), (q_dim + kv_dim, kv_dim)] {
                let wt = w.narrow(0, offset, rows)?.contiguous()?.transpose(0, 1)?;
                let wt = if d.dtype() == DType::F32 {
                    wt
                } else {
                    wt.to_dtype(d.dtype())?
                };
                let wt = wt.to_device(d.device())?;
                parts.push(QLinear::new(
                    Weight::Dense(layer::Linear::from_transposed(wt, None)?, d.dtype()),
                    None,
                    in_dim,
                    rows,
                ));
            }
            let v = parts.pop().unwrap();
            let k = parts.pop().unwrap();
            let q = parts.pop().unwrap();
            Ok((q, k, v))
        }
    }
}

/// `{prefix}.weight` (`[out, in]`) + `{prefix}.bias` (F32).
fn linear(in_dim: usize, out_dim: usize, vb: Vb) -> Result<QLinear> {
    let weight = weight(in_dim, out_dim, &vb)?;
    let bias = vb.get_f32(out_dim, "bias")?;
    Ok(QLinear::new(weight, Some(bias), in_dim, out_dim))
}

fn linear_no_bias(in_dim: usize, out_dim: usize, vb: Vb) -> Result<QLinear> {
    let weight = weight(in_dim, out_dim, &vb)?;
    Ok(QLinear::new(weight, None, in_dim, out_dim))
}

fn rms_norm(dim: usize, eps: f64, vb: Vb) -> Result<RmsNorm> {
    Ok(RmsNorm::new(vb.get_f32(dim, "weight")?, eps as f32))
}

// ---------------------------------------------------------------------------
// Composed ops
// ---------------------------------------------------------------------------

/// tanh(x) = 2.sigmoid(2x) - 1.
///
/// The identity, not the tensor method: the method calls a tanh kernel, and the two agree
/// mathematically and not in the last bit. This one gates every modulated block, so which of
/// the two runs is a property of the renders this model was checked against.
pub(crate) fn tanh(x: &Tensor) -> Result<Tensor> {
    x.scale(2.0)?.sigmoid()?.affine(2.0, -1.0)
}

/// a - b.
fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.add(&b.scale(-1.0)?)
}

/// General axis permutation, taking the order as a slice.
///
/// The tensor method's tuple form cannot spell a rank that is not written down, and both of
/// the six-dimensional permutations below are built at runtime.
pub(crate) fn permute(x: &Tensor, perm: &[usize]) -> Result<Tensor> {
    x.permute(perm)
}

// ==================== TimestepEmbedder ====================

/// Sinusoidal timestep encoding, `t: [b] -> [b, dim]`.
///
/// The same function FLUX carries in `flux::common`, to one difference that is not free: the
/// frequency row is built here on the HOST in double precision and there on the DEVICE in
/// single. Both spell `exp(-ln(10000) * i / half)`, and the two disagree in the last bits, so
/// each family keeps the one its renders were checked against. Folding them would need an
/// image compared before and after, not a passing test.
///
/// Unlike FLUX's, this one takes no factor on `t` and widens before it multiplies, which is
/// the ordering FLUX had to be corrected to.
pub(crate) fn timestep_embedding(t: &Tensor, dim: usize) -> Result<Tensor> {
    if dim % 2 == 1 {
        return Err(Error(format!("timestep_embedding: {dim} is odd")));
    }
    let dev = t.device();
    let half = dim / 2;
    // freqs[i] = exp(-ln(MAX_PERIOD) * i / half), a slim [1, half] broadcast
    // row computed host-side (the facade builds it from arange * exp).
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-(MAX_PERIOD.ln()) * i as f64 / half as f64).exp() as f32)
        .collect();
    let freqs = Tensor::from_vec_f32(freqs, (1, half))?.to_device(&dev)?;
    // Every timestep as a column against that one row of frequencies: (B, 1) against
    // (1, half) gives the angle each of them stands at, (B, half).
    let t = t.to_dtype(DType::F32)?.unsqueeze(1)?;
    let args = t.broadcast_mul(&freqs)?;
    Tensor::cat(&[&args.cos()?, &args.sin()?], D::Minus1)
}

/// Timestep embedding: sinusoidal encoding + 2-layer SiLU MLP.
pub struct TimestepEmbedder {
    linear1: QLinear,
    linear2: QLinear,
    frequency_embedding_size: usize,
}

impl TimestepEmbedder {
    pub fn new(out_size: usize, mid_size: usize, vb: Vb) -> Result<Self> {
        let linear1 = linear(FREQUENCY_EMBEDDING_SIZE, mid_size, vb.pp("mlp").pp("0"))?;
        let linear2 = linear(mid_size, out_size, vb.pp("mlp").pp("2"))?;
        Ok(Self {
            linear1,
            linear2,
            frequency_embedding_size: FREQUENCY_EMBEDDING_SIZE,
        })
    }

    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let t_freq = timestep_embedding(t, self.frequency_embedding_size)?;
        self.linear2
            .forward(&self.linear1.forward(&t_freq)?.silu()?)
    }
}

// ==================== Feed-forward (SwiGLU) ====================

/// `w1` / `w3` / `w2` - the names this checkpoint gives the gate, the parallel
/// projection and the narrowing of its gated feed-forward.
pub fn feed_forward(dim: usize, hidden_dim: usize, vb: Vb) -> Result<QSwiGlu> {
    Ok(QSwiGlu::new(
        linear_no_bias(dim, hidden_dim, vb.pp("w1"))?,
        linear_no_bias(dim, hidden_dim, vb.pp("w3"))?,
        linear_no_bias(hidden_dim, dim, vb.pp("w2"))?,
    ))
}

// ==================== QkNorm ====================

/// QK normalization using RMSNorm (per-head-dim).
pub use crate::tensor::layer::QkNorm;

/// Read a Z-Image block's qk-norm scales. `norm_q`/`norm_k` in the diffusers export,
/// `q_norm`/`k_norm` in the research layout - same tensor, same shape.
fn qk_norm(head_dim: usize, eps: f64, vb: Vb) -> Result<QkNorm> {
    let q = rms_norm(head_dim, eps, vb.pp("norm_q"))
        .or_else(|_| rms_norm(head_dim, eps, vb.pp("q_norm")))?;
    let k = rms_norm(head_dim, eps, vb.pp("norm_k"))
        .or_else(|_| rms_norm(head_dim, eps, vb.pp("k_norm")))?;
    Ok(QkNorm::new(q, k))
}

// ==================== RopeEmbedder (3D) ====================

/// 3D rotary position embedding. The per-axis cos/sin tables are part of
/// construction (not a perf cache) and live CPU-side in f32 - index_select
/// host-bounces on CUDA anyway in the native substrate.
pub struct RopeEmbedder {
    axes_dims: Vec<usize>,
    // `theta` and `axes_lens` are arguments to `new`, not state: they shape the tables
    // below once and are never consulted again.
    cos_cached: Vec<Tensor>,
    sin_cached: Vec<Tensor>,
}

impl RopeEmbedder {
    pub fn new(theta: f64, axes_dims: Vec<usize>, axes_lens: Vec<usize>) -> Result<Self> {
        if axes_dims.len() != axes_lens.len() {
            return Err(Error(format!(
                "RopeEmbedder: axes_dims {axes_dims:?} vs axes_lens {axes_lens:?}"
            )));
        }
        // One table per axis, each the outer product of every position that axis can hold with
        // the frequencies it turns at, taken through cos and sin once so that a lookup later is
        // a row selection and nothing more. The frequencies come from the shared rope helper,
        // in its single-precision form: an axis here is 32 or 48 wide against a theta of 256,
        // where the two forms disagree in the last bits, and the renders this port is checked
        // against were made with this one.
        let per_axis: Vec<(Tensor, Tensor)> = axes_dims
            .iter()
            .zip(&axes_lens)
            .map(|(&d, &len)| -> Result<(Tensor, Tensor)> {
                let inv_freq = crate::inference::model::rope::inverse_frequencies(d, theta as f32);
                let inv_freq = Tensor::from_vec_f32(inv_freq, (1, d / 2))?;
                let positions: Vec<f32> = (0..len).map(|p| p as f32).collect();
                let positions = Tensor::from_vec_f32(positions, (len, 1))?;
                let angles = positions.broadcast_mul(&inv_freq)?; // (len, d/2)
                Ok((angles.cos()?, angles.sin()?))
            })
            .collect::<Result<_>>()?;
        // Splitting the pairs keeps the axes in order, so table `i` is still axis `i`.
        let (cos_tables, sin_tables) = per_axis.into_iter().unzip();
        Ok(Self {
            axes_dims,
            cos_cached: cos_tables,
            sin_cached: sin_tables,
        })
    }

    /// ids: `(seq_len, 3)` u32 `[frame_id, height_id, width_id]` ->
    /// cos/sin `(seq_len, head_dim/2)`.
    pub fn forward(&self, ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let (seq_len, n_axes) = ids.shape().dims2()?;
        if n_axes != self.axes_dims.len() {
            return Err(Error(format!(
                "RopeEmbedder: ids axes {n_axes} != {}",
                self.axes_dims.len()
            )));
        }
        // u32 column extraction host-side (the substrate's narrow/cat are
        // f32-only; to_vec_f32 converts the u32 storage).
        let flat = ids.to_vec_f32();
        let mut cos_parts = Vec::with_capacity(n_axes);
        let mut sin_parts = Vec::with_capacity(n_axes);
        for i in 0..n_axes {
            let axis_ids: Vec<u32> = (0..seq_len).map(|s| flat[s * n_axes + i] as u32).collect();
            let axis_ids = Tensor::from_vec_u32(axis_ids, seq_len)?;
            cos_parts.push(self.cos_cached[i].index_select(&axis_ids, 0)?);
            sin_parts.push(self.sin_cached[i].index_select(&axis_ids, 0)?);
        }
        let cos_refs: Vec<&Tensor> = cos_parts.iter().collect();
        let sin_refs: Vec<&Tensor> = sin_parts.iter().collect();
        let cos = Tensor::cat(&cos_refs, D::Minus1)?;
        let sin = Tensor::cat(&sin_refs, D::Minus1)?;
        Ok((cos, sin))
    }
}

/// Apply RoPE, written on real pairs rather than on complex numbers: consecutive dimensions
/// are the real and imaginary parts, and the rotation is their complex product with cos+i.sin.
///
/// x: `(B, seq_len, n_heads, head_dim)`; cos, sin: `(seq_len, head_dim/2)`.
pub fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, seq_len, n_heads, head_dim) = x.shape().dims4()?;
    let half_dim = head_dim / 2;

    // Interleaved real/imag view: (B, seq_len, n_heads, half_dim, 2)
    let x = x.reshape(vec![b, seq_len, n_heads, half_dim, 2])?;
    let x_real = x.get_on_dim(D::Minus1, 0)?; // (B, seq_len, n_heads, half_dim)
    let x_imag = x.get_on_dim(D::Minus1, 1)?;

    // (seq_len, half_dim) -> (1, seq_len, 1, half_dim) for broadcasting
    let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

    // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    let y_real = sub(&x_real.broadcast_mul(&cos)?, &x_imag.broadcast_mul(&sin)?)?;
    let y_imag = x_real
        .broadcast_mul(&sin)?
        .add(&x_imag.broadcast_mul(&cos)?)?;

    // Re-interleave the pairs: the real and imaginary halves alternate again along head_dim.
    Tensor::stack(&[&y_real, &y_imag], 4)?.reshape(vec![b, seq_len, n_heads, head_dim])
}

// ==================== ZImageAttention ====================

/// Query chunk size for the memory-bounded attention, from the ONE declaration of it:
/// [`crate::inference::model::acestep::ops::dit_query_tile`], which derives it from
/// `head_dim` and says what the multiple is and why.
///
/// Two implementations once named their own numbers here - 512 against 1024 - and a split
/// placement ran the one nobody had measured. What is real and belongs here is that the
/// MASKED path chunks as well: production Z-Image always passes an all-ones caption mask,
/// so the bound has to hold whether or not a mask is present.
fn attn_query_chunk(head_dim: usize, seq: usize, heads: usize) -> usize {
    crate::inference::model::acestep::ops::dit_query_tile(head_dim, seq, heads)
}

/// Scaled-dot-product attention with an optional `(B, seq)` 0/1 padding mask.
/// q/k/v: `(B, n_heads, seq, head_dim)`. Queries are chunked so the score matrix never
/// materialises whole - see [`attn_query_chunk`].
pub(crate) fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f32,
) -> Result<Tensor> {
    let chunk = attn_query_chunk(q.dim(D::Minus1)?, q.dim(2)?, q.dim(1)?);
    attention_chunked(q, k, v, mask, scale, chunk)
}

/// [`attention`] with an explicit chunk size, for the tail-tile unit test.
///
/// The caption mask arrives as `(B, seq)` with one for a real token and zero for padding; the
/// shared attention takes an additive mask, so the conversion happens here - once, since it
/// broadcasts identically over every query tile.
///
/// The tiling itself is not written here. It is the one in `acestep::ops`, told which tile to
/// use rather than choosing its own, and told not to run its GEMMs in bf16: this DiT is not
/// one of the towers that measured faster that way.
fn attention_chunked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f32,
    chunk: usize,
) -> Result<Tensor> {
    let additive = match mask {
        Some(m) => Some(
            m.to_dtype(DType::F32)?
                .unsqueeze(1)?
                .unsqueeze(2)?
                .affine(1e9, -1e9)?,
        ),
        None => None,
    };
    crate::inference::model::acestep::ops::sdpa_tiled(
        q,
        k,
        v,
        additive.as_ref(),
        false,
        scale,
        1.0,
        chunk,
    )
}

/// Z-Image attention with QK normalization and 3D RoPE.
pub struct ZImageAttention {
    to_q: QLinear,
    to_k: QLinear,
    to_v: QLinear,
    to_out: QLinear,
    qk_norm: Option<QkNorm>,
    n_heads: usize,
    head_dim: usize,
}

impl ZImageAttention {
    pub fn new(cfg: &Config, vb: Vb) -> Result<Self> {
        // What the projections read and what they write: the model's width in, one head's
        // width times the head count out - and fewer key and value heads than query heads
        // whenever the checkpoint shares them.
        let (dim, n_heads, head_dim) = (cfg.dim, cfg.n_heads, cfg.head_dim());
        let (q_dim, kv_dim) = (n_heads * head_dim, cfg.n_kv_heads * head_dim);

        // Two checkpoint dialects, the same two the facade accepts: the official
        // diffusers export (separate to_q/to_k/to_v + to_out.0) and the
        // fused research layout the Ray fine-tunes ship (`qkv` + `out`).
        // Reading only the first dialect here is what sent those checkpoints down
        // the facade fallback and its CPU-segment plan - minutes per step.
        let (to_q, to_k, to_v) = match (
            linear_no_bias(dim, q_dim, vb.pp("to_q")),
            linear_no_bias(dim, kv_dim, vb.pp("to_k")),
            linear_no_bias(dim, kv_dim, vb.pp("to_v")),
        ) {
            (Ok(q), Ok(k), Ok(v)) => (q, k, v),
            _ => fused_qkv(dim, q_dim, kv_dim, &vb)?,
        };
        let to_out = linear_no_bias(q_dim, dim, vb.pp("to_out").pp("0"))
            .or_else(|_| linear_no_bias(q_dim, dim, vb.pp("out")))?;

        let qk_norm = if cfg.qk_norm {
            Some(qk_norm(head_dim, 1e-5, vb)?)
        } else {
            None
        };

        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            qk_norm,
            n_heads,
            head_dim,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = hidden_states.shape().dims3()?;

        // Three roles read the same states over the full width, and each answer is cut into
        // the heads that will read it independently: (B, seq, n_heads * head_dim) seen as
        // (B, seq, n_heads, head_dim).
        let per_head = (b, seq_len, self.n_heads, self.head_dim);
        let project =
            |proj: &QLinear| -> Result<Tensor> { proj.forward(hidden_states)?.reshape(per_head) };
        let q = project(&self.to_q)?;
        let k = project(&self.to_k)?;
        let v = project(&self.to_v)?;

        // Only the two sides of the dot product carry a position: they are brought to a common
        // scale together, then turned by the angle their patch coordinate stands at. Values are
        // whatever the projection made of them.
        let (q, k) = match &self.qk_norm {
            Some(norm) => norm.forward(&q, &k)?,
            None => (q, k),
        };
        let rotate = |t: &Tensor| apply_rotary_emb(t, cos, sin);
        let (q, k) = (rotate(&q)?, rotate(&k)?);

        // Heads move ahead of the sequence, so one head's scores are one matrix and the batch
        // of them is what the kernel walks: (B, n_heads, seq, head_dim). Native transposes
        // materialize.
        let swap_head_seq = |t: &Tensor| t.transpose(1, 2);
        let (q, k, v) = (swap_head_seq(&q)?, swap_head_seq(&k)?, swap_head_seq(&v)?);

        // A dot product over a head grows with that head's width, so the softmax reads it
        // divided by the width's square root - taken in double precision and narrowed once,
        // so the temperature is the same number wherever the attention runs.
        let scale = (1.0 / (self.head_dim as f64).sqrt()) as f32;
        let context = attention(&q, &k, &v, attention_mask, scale)?;

        // The heads fold back into one width: (B, n_heads, seq, head_dim) -> (B, seq, dim).
        let context =
            swap_head_seq(&context)?.reshape((b, seq_len, self.n_heads * self.head_dim))?;
        self.to_out.forward(&context)
    }
}

// ==================== ZImageTransformerBlock ====================

/// Z-Image transformer block with optional AdaLN modulation.
pub struct ZImageTransformerBlock {
    attention: ZImageAttention,
    feed_forward: QSwiGlu,
    attention_norm1: RmsNorm,
    attention_norm2: RmsNorm,
    ffn_norm1: RmsNorm,
    ffn_norm2: RmsNorm,
    adaln_modulation: Option<QLinear>,
}

/// The `n_refiner_layers` blocks that sit under one prefix.
///
/// Both refiners are the same stack read twice: the noise side is modulated by the timestep
/// and the context side is not, and in each of the two models that has a pair that was the
/// only thing telling the two loops apart.
pub(crate) fn refiner_stack(
    cfg: &Config,
    modulation: bool,
    vb: &Vb,
) -> Result<Vec<ZImageTransformerBlock>> {
    (0..cfg.n_refiner_layers)
        .map(|i| ZImageTransformerBlock::new(cfg, modulation, vb.pp(i)))
        .collect()
}

impl ZImageTransformerBlock {
    pub fn new(cfg: &Config, modulation: bool, vb: Vb) -> Result<Self> {
        let dim = cfg.dim;
        // Both halves of the block sit between a pair of scales - one over what the half reads,
        // one over what it hands back to the residual - and all four are read the same way,
        // differing only in the name this checkpoint files them under.
        let norm = |name: &str| rms_norm(dim, cfg.norm_eps, vb.pp(name));

        Ok(Self {
            attention: ZImageAttention::new(cfg, vb.pp("attention"))?,
            feed_forward: feed_forward(dim, cfg.hidden_dim(), vb.pp("feed_forward"))?,
            attention_norm1: norm("attention_norm1")?,
            attention_norm2: norm("attention_norm2")?,
            ffn_norm1: norm("ffn_norm1")?,
            ffn_norm2: norm("ffn_norm2")?,
            // Where the block is modulated, one projection answers with all four vectors the
            // timestep dictates - a scale and a gate for each half - so it writes four widths
            // at once, out of an embedding narrower than the model unless the model is
            // narrower still.
            adaln_modulation: if modulation {
                let adaln = vb.pp("adaLN_modulation").pp("0");
                Some(linear(cfg.adaln_dim(), 4 * dim, adaln)?)
            } else {
                None
            },
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        adaln_input: Option<&Tensor>,
    ) -> Result<Tensor> {
        // A modulated block scales what it norms and gates what it adds back; an unmodulated
        // one does neither. That is the whole difference, so it travels as four optional
        // tensors rather than as a second copy of the block that can drift from the first.
        let (scale_msa, gate_msa, scale_mlp, gate_mlp) = match &self.adaln_modulation {
            Some(adaln) => {
                let adaln_input = adaln_input
                    .ok_or_else(|| Error("adaln_input required when modulation=true".into()))?;
                // (B, 256) -> (B, 4*dim) -> (B, 1, 4*dim) -> chunk into 4
                let modulation = adaln.forward(adaln_input)?.unsqueeze(1)?;
                let chunks = modulation.chunk(4, D::Minus1)?;
                (
                    Some(chunks[0].affine(1.0, 1.0)?), // scale + 1
                    Some(tanh(&chunks[1])?),
                    Some(chunks[2].affine(1.0, 1.0)?),
                    Some(tanh(&chunks[3])?),
                )
            }
            None => (None, None, None, None),
        };
        let by = |t: Tensor, factor: &Option<Tensor>| match factor {
            Some(f) => t.broadcast_mul(f),
            None => Ok(t),
        };

        // Attention block
        let normed = by(self.attention_norm1.forward(x)?, &scale_msa)?;
        let attn_out = self.attention.forward(&normed, attn_mask, cos, sin)?;
        let attn_out = self.attention_norm2.forward(&attn_out)?;
        let x = x.add(&by(attn_out, &gate_msa)?)?;

        // FFN block
        let normed = by(self.ffn_norm1.forward(&x)?, &scale_mlp)?;
        let ffn_out = self.feed_forward.forward(&normed)?;
        let ffn_out = self.ffn_norm2.forward(&ffn_out)?;
        x.add(&by(ffn_out, &gate_mlp)?)
    }
}

// ==================== FinalLayer ====================

/// Final layer for output projection.
pub struct FinalLayer {
    norm_final: LayerNorm,
    linear: QLinear,
    adaln_silu: QLinear,
}

impl FinalLayer {
    pub fn new(hidden_size: usize, out_channels: usize, vb: Vb) -> Result<Self> {
        Ok(Self {
            // The unit weight is kept CPU-resident: it host-bounces with the input on
            // CUDA, so the two always meet on the same device.
            norm_final: layer::layer_norm_no_affine(hidden_size, 1e-6, &Device::Cpu)?,
            linear: linear(hidden_size, out_channels, vb.pp("linear"))?,
            adaln_silu: linear(
                adaln_dim(hidden_size),
                hidden_size,
                vb.pp("adaLN_modulation").pp("1"),
            )?,
        })
    }

    pub fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let scale = self.adaln_silu.forward(&c.silu()?)?;
        let scale = scale.affine(1.0, 1.0)?.unsqueeze(1)?;
        let x = self.norm_final.forward(x)?.broadcast_mul(&scale)?;
        self.linear.forward(&x)
    }
}

// ==================== Patchify / Unpatchify ====================

/// Convert image to patch sequence.
/// input: `(B, C, F, H, W)`; output: `(B, num_patches, patch_dim)` + the
/// original `(F, H, W)` size. A single frame with a frame-patch of one takes a six-dimensional
/// permute; anything else takes the general path.
/// How many patches a latent of this size is cut into, along each axis.
///
/// Every forward of this model needs the same three numbers - to cut, to place the rotary
/// coordinates, and to weave back - and the cut is the only one of the three that could tell
/// anyone it had been divided differently.
pub(crate) fn patch_grid(
    (f, h, w): (usize, usize, usize),
    patch_size: usize,
    f_patch_size: usize,
) -> (usize, usize, usize) {
    (f / f_patch_size, h / patch_size, w / patch_size)
}

pub fn patchify(
    x: &Tensor,
    patch_size: usize,
    f_patch_size: usize,
) -> Result<(Tensor, (usize, usize, usize))> {
    let xd = x.dims();
    if xd.len() != 5 {
        return Err(Error(format!(
            "patchify: expected rank-5 input, got {xd:?}"
        )));
    }
    let (b, c, f, h, w) = (xd[0], xd[1], xd[2], xd[3], xd[4]);
    let (pf, ph, pw) = (f_patch_size, patch_size, patch_size);
    let (f_tokens, h_tokens, w_tokens) = patch_grid((f, h, w), patch_size, f_patch_size);

    // Either route reads the same latent and hands back the same sequence, so what the size
    // was is answered once, at the end.
    let x = if f == 1 && pf == 1 {
        // A still image is the shared cut, in the order this family's patch projection wants.
        crate::inference::model::patches::cut(
            &x.squeeze(2)?,
            patch_size,
            crate::inference::model::patches::Order::ByPixel,
        )?
    } else {
        // Anything deeper than one frame is gathered by hand: the axes are interleaved so
        // that a patch's own numbers end up adjacent, then folded into one row each.
        let num_patches = f_tokens * h_tokens * w_tokens;
        let patch_dim = pf * ph * pw * c;
        let x = permute(x, &[0, 2, 3, 4, 1])?; // (B, F, H, W, C)
        let x = x.reshape(vec![b, f_tokens, pf, h_tokens, ph, w_tokens * pw * c])?;
        let x = permute(&x, &[0, 1, 3, 5, 2, 4])?;
        x.reshape(vec![b, num_patches, patch_dim])?
    };
    Ok((x, (f, h, w)))
}

/// Convert patch sequence back to image, against the `(F, H, W)` it was cut from.
/// input: `(B, seq_len, patch_dim)`; output: `(B, C, F, H, W)`.
pub fn unpatchify(
    x: &Tensor,
    (f, h, w): (usize, usize, usize),
    patch_size: usize,
    f_patch_size: usize,
    out_channels: usize,
) -> Result<Tensor> {
    let (pf, ph, pw) = (f_patch_size, patch_size, patch_size);
    let (f_tokens, h_tokens, w_tokens) = patch_grid((f, h, w), patch_size, f_patch_size);
    let ori_len = f_tokens * h_tokens * w_tokens;

    let (b, _, _) = x.shape().dims3()?;
    let x = x.narrow(1, 0, ori_len)?; // Remove padding

    if f == 1 && pf == 1 {
        // The shared weave, then the frame axis this family carries even for a still image.
        crate::inference::model::patches::weave(
            &x,
            h_tokens,
            w_tokens,
            patch_size,
            out_channels,
            crate::inference::model::patches::Order::ByPixel,
        )?
        .unsqueeze(2)
    } else {
        // General case
        let x = x.reshape(vec![
            b,
            f_tokens,
            h_tokens,
            w_tokens,
            pf * ph * pw * out_channels,
        ])?;
        let x = x.reshape(vec![
            b,
            f_tokens,
            h_tokens,
            w_tokens * pf,
            ph,
            pw * out_channels,
        ])?;
        let x = permute(&x, &[0, 5, 1, 3, 2, 4])?;
        x.reshape(vec![b, out_channels, f, h, w])
    }
}

/// Create the 3D coordinate grid of RoPE position IDs: `(F*H*W, 3)` u32,
/// CPU-resident (cache-free v1 - the facade memoizes per (device, size,
/// start); RopeEmbedder::forward reads the ids host-side anyway).
pub fn create_coordinate_grid(
    (f, h, w): (usize, usize, usize),
    (f0, h0, w0): (usize, usize, usize),
) -> Result<Tensor> {
    // The window's origin belongs to the axis, not to each visit, so it is carried in the
    // bounds. Row-major over frame, then row, then column, one row of three per position.
    let mut coords = Vec::with_capacity(f * h * w * 3);
    for fi in f0..f0 + f {
        for hi in h0..h0 + h {
            for wi in w0..w0 + w {
                coords.extend([fi as u32, hi as u32, wi as u32]);
            }
        }
    }
    Tensor::from_vec_u32(coords, (f * h * w, 3))
}

// ==================== Stem ====================

/// Everything this checkpoint carries OUTSIDE the main stack: the timestep and caption
/// embedders, the two refiners, the way back out, and the rotary tables.
///
/// Two loaders build it - the single-device one below and the multi-device one in the
/// sibling `hetero` module - and they agree on every name and every width while agreeing on
/// nothing else. They put the pieces on different devices, they read the two projections
/// into different types, and they DO NOT READ THEM IN THE SAME ORDER: the multi-device
/// loader reads the way back out after the thirty main blocks, this one reads it before the
/// refiners.
///
/// That order is not decoration. A card's ledger is measured from the order its tensors
/// arrive, and the placement chosen for this model is chosen against that ledger - so a
/// stem that assembled itself here would have to pick one of the two orders and would move
/// the other loader's peak. What is shared is therefore each PIECE, handed back on its own
/// for the caller to place where it already places it.
pub mod stem {
    use super::{
        refiner_stack, Config, FinalLayer, Result, RopeEmbedder, TimestepEmbedder, VarBuilder, Vb,
        ZImageTransformerBlock,
    };

    /// A weight source that can walk into a prefix.
    ///
    /// The two loaders hand back different types for the same tensor - a host-staged
    /// projection on one side, a dense one on the other - so what they have in common is
    /// the checkpoint's NAMES. This is what lets a name be written once even so.
    pub trait Prefixed: Sized {
        /// The builder walked into `{prefix}.{name}`.
        fn sub<S: ToString>(&self, name: S) -> Self;
    }

    impl Prefixed for Vb {
        fn sub<S: ToString>(&self, name: S) -> Self {
            self.pp(name)
        }
    }

    impl Prefixed for VarBuilder {
        fn sub<S: ToString>(&self, name: S) -> Self {
            self.pp(name)
        }
    }

    /// How the keyed export names its entry for a patch of two pixels by one frame.
    const KEYED_PATCH: &str = "2-1";

    /// Read a group under whichever of the two names this checkpoint files it by.
    ///
    /// The official export keys the patch projection and the way back out by the patch size
    /// they are for - `all_{name}.2-1` - where a flattened export files the selected entry
    /// flat, as `{name}`. Both spellings are the checkpoint's specification, and the RULE
    /// relating them is written here so that a call site need only say which group it wants.
    /// A loader that knows one spelling and not the other does not fail; it falls through to
    /// a slower path and stays there, which is why this is not left to each caller.
    pub fn by_dialect<B: Prefixed, T>(
        vb: &B,
        name: &str,
        load: impl Fn(B) -> Result<T>,
    ) -> Result<T> {
        load(vb.sub(format!("all_{name}")).sub(KEYED_PATCH)).or_else(|_| load(vb.sub(name)))
    }

    /// The timestep embedder, answering at the width every modulation projection reads.
    ///
    /// The width it passes through on the way there is the checkpoint's own and is derived
    /// from nothing.
    pub fn timestep_embedder(cfg: &Config, vb: &Vb) -> Result<TimestepEmbedder> {
        TimestepEmbedder::new(cfg.adaln_dim(), 1024, vb.pp("t_embedder"))
    }

    /// Both refiners, in the order a loader reads them: the noise side, which the timestep
    /// modulates, then the caption side, which it does not.
    pub fn refiners(
        cfg: &Config,
        vb: &Vb,
    ) -> Result<(Vec<ZImageTransformerBlock>, Vec<ZImageTransformerBlock>)> {
        let noise = refiner_stack(cfg, true, &vb.pp("noise_refiner"))?;
        let context = refiner_stack(cfg, false, &vb.pp("context_refiner"))?;
        Ok((noise, context))
    }

    /// The way back out: from the model's width to one patch's worth of numbers.
    pub fn final_layer(cfg: &Config, vb: &Vb) -> Result<FinalLayer> {
        by_dialect(vb, "final_layer", |vb| {
            FinalLayer::new(cfg.dim, cfg.patch_dim(), vb)
        })
    }

    /// The rotary tables this checkpoint's axes ask for. Nothing is read from the weight
    /// files here - the tables are computed, and computed host-side.
    pub fn rope_embedder(cfg: &Config) -> Result<RopeEmbedder> {
        RopeEmbedder::new(cfg.rope_theta, cfg.axes_dims.clone(), cfg.axes_lens.clone())
    }
}

// ==================== ZImageTransformer2DModel ====================

/// Shape-keyed per-image RoPE tables + image attention mask, resident on
/// the compute device. Everything here is a pure function of
/// `(b, f_tokens, h_tokens, w_tokens, text_len)` - constant across the
/// denoise loop - so one entry serves a whole generation (the flux
/// `pe_cache` recipe). Without it the CPU-side cos/sin tables host-bounce
/// every broadcast in `apply_rotary_emb` on CUDA.
struct RopeMaskCache {
    key: (usize, usize, usize, usize, usize),
    x_cos: Tensor,
    x_sin: Tensor,
    cap_cos: Tensor,
    cap_sin: Tensor,
    unified_cos: Tensor,
    unified_sin: Tensor,
    x_attn_mask: Tensor,
}

/// Z-Image Transformer 2D model. Of the facade's PerImageCache only the
/// shape-dependent RoPE/mask bundle is cached (`rope_cache`); the caption
/// embedding + context refiner are recomputed per forward (negligible  -
/// they run on the ~30-token caption, not the image sequence).
pub struct ZImageTransformer2DModel {
    t_embedder: TimestepEmbedder,
    cap_embedder_norm: RmsNorm,
    cap_embedder_linear: QLinear,
    x_embedder: QLinear,
    final_layer: FinalLayer,
    // Every checkpoint of this family carries `x_pad_token` and `cap_pad_token`, and no
    // forward here reaches either: the sequences are padded with a mask rather than with a
    // learnt token. They are not loaded, which is two reads and two device allocations that
    // this path used to pay for and the multi-device one already did not.
    noise_refiner: Vec<ZImageTransformerBlock>,
    context_refiner: Vec<ZImageTransformerBlock>,
    /// One entry per main block, and `None` where the block runs on a backend that is
    /// not this substrate.
    ///
    /// The port this replaces kept a FULL native block for every such position, "to keep
    /// the vec the right size", built from a host builder or - when the plan named no host
    /// slot - from the PRIMARY CARD's. So a stack with an accelerated segment quietly
    /// loaded a second copy of those blocks onto the first GPU and never called them: a
    /// capability whose cost nothing charged, because no measurement had walked that path.
    /// An absent block is absent here.
    layers: Vec<Option<ZImageTransformerBlock>>,
    /// Where each main block was built, when the stack spans devices.
    ///
    /// `None` is the production single-device model and the forward below is then the
    /// forward it always was. `Some` is a stack built ACROSS devices - what a
    /// candidate placement is measured on - and the forward moves the stream to each
    /// block's device at the segment boundary, which is where the cost of a split
    /// lives and therefore where a measurement of a split has to see it.
    layer_devices: Option<Vec<Device>>,
    rope_embedder: RopeEmbedder,
    rope_cache: Mutex<Option<RopeMaskCache>>,
    cfg: Config,
}

impl ZImageTransformer2DModel {
    /// GGUF-quantized weights (the flux-port loader).
    pub fn new(cfg: &Config, vb: QVarBuilder) -> Result<Self> {
        Self::build(cfg, Vb::Q(vb))
    }

    /// Dense safetensors weights - the production Tongyi-MAI/Z-Image-Turbo
    /// checkpoint (F32 shards). 2-D matmul weights load at `vb`'s dtype
    /// (BF16 on CUDA, the facade production dtype); norms/biases/pad tokens
    /// at exact F32. See the module doc for the mixed-precision contract.
    pub fn from_varbuilder(cfg: &Config, vb: &VarBuilder) -> Result<Self> {
        Self::build(cfg, Vb::Dense((*vb).clone()))
    }

    fn build(cfg: &Config, vb: Vb) -> Result<Self> {
        Self::build_across(cfg, &vb, None)
    }

    /// The same constructor, with the main blocks optionally placed across devices.
    ///
    /// `placement` is `(one builder + device per slot, the slot each main block goes
    /// to)`. `None` builds everything from `vb` - the production path, unchanged.
    /// Everything outside the main stack (embedders, refiners, final layer) is built
    /// from `vb` either way, which is what the multi-device loader does too.
    fn build_across(
        cfg: &Config,
        vb: &Vb,
        placement: Option<(&[(Vb, Device)], &[usize])>,
    ) -> Result<Self> {
        // THE ORDER THESE FIVE ARE READ IN IS THIS LOADER'S, and it is what the ledger this
        // model's placement is chosen against was measured from. The pieces come from
        // [`stem`], which the multi-device loader shares; where each one lands in the
        // sequence does not.
        let t_embedder = stem::timestep_embedder(cfg, vb)?;

        // The caption arrives at the text encoder's width, is normalised there, and is only
        // then projected into the model's - two parts of one embedder, under one prefix.
        let cap = vb.pp("cap_embedder");
        let cap_embedder_norm = rms_norm(cfg.cap_feat_dim, cfg.norm_eps, cap.pp("0"))?;
        let cap_embedder_linear = linear(cfg.cap_feat_dim, cfg.dim, cap.pp("1"))?;

        let x_embedder =
            stem::by_dialect(vb, "x_embedder", |vb| linear(cfg.patch_dim(), cfg.dim, vb))?;
        let final_layer = stem::final_layer(cfg, vb)?;

        let (noise_refiner, context_refiner) = stem::refiners(cfg, vb)?;

        // Main layers (with modulation), each on the device its slot names.
        let mut layers = Vec::with_capacity(cfg.n_layers);
        let mut layer_devices: Option<Vec<Device>> =
            placement.map(|_| Vec::with_capacity(cfg.n_layers));
        for i in 0..cfg.n_layers {
            let block_vb = match placement {
                Some((slots, of)) => {
                    let slot = of.get(i).copied().unwrap_or(0);
                    let (svb, dev) = slots.get(slot).ok_or_else(|| {
                        Error(format!("build_across: block {i} names slot {slot}"))
                    })?;
                    if let Some(devs) = layer_devices.as_mut() {
                        devs.push(dev.clone());
                    }
                    svb
                }
                None => vb,
            };
            layers.push(Some(ZImageTransformerBlock::new(
                cfg,
                true,
                block_vb.pp("layers").pp(i),
            )?));
        }

        let rope_embedder = stem::rope_embedder(cfg)?;

        Ok(Self {
            t_embedder,
            cap_embedder_norm,
            cap_embedder_linear,
            x_embedder,
            final_layer,
            noise_refiner,
            context_refiner,
            layers,
            layer_devices,
            rope_embedder,
            rope_cache: Mutex::new(None),
            cfg: cfg.clone(),
        })
    }

    /// Forward pass (same contract as the facade `forward`).
    ///
    /// * `x` - latent `(B, C, F, H, W)`
    /// * `t` - timesteps in `[0, 1]`, `(B,)`
    /// * `cap_feats` - caption features `(B, text_len, cap_feat_dim)`
    /// * `cap_mask` - caption mask `(B, text_len)`, 1=valid, 0=padding
    pub fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        cap_feats: &Tensor,
        cap_mask: &Tensor,
    ) -> Result<Tensor> {
        let xd = x.dims();
        if xd.len() != 5 {
            return Err(Error(format!(
                "forward: expected rank-5 latent, got {xd:?}"
            )));
        }
        let (b, f, h, w) = (xd[0], xd[2], xd[3], xd[4]);
        // The geometry this forward cuts in and must come back as, read once here rather
        // than reached for again at the far end of the stack.
        let (patch_size, f_patch_size, channels) = (
            self.cfg.all_patch_size[0],
            self.cfg.all_f_patch_size[0],
            self.cfg.in_channels,
        );

        // 1. Timestep embedding
        let t_scaled = t.scale(self.cfg.t_scale as f32)?;
        let adaln_input = self.t_embedder.forward(&t_scaled)?; // (B, 256)

        // 2. Patchify and embed image
        let (x_patches, orig_size) = patchify(x, patch_size, f_patch_size)?;
        let mut x = self.x_embedder.forward(&x_patches)?; // (B, img_seq, dim)
        let img_seq_len = x.dim(1)?;
        // The same cut the patchify above made, so the two cannot drift apart.
        let (f_tokens, h_tokens, w_tokens) = patch_grid((f, h, w), patch_size, f_patch_size);
        let text_len = cap_feats.dim(1)?;

        // 3 + 5 + 6 + 10 (shape-dependent parts). Position IDs, RoPE tables
        // and the all-ones image mask depend only on the grid/caption SHAPES
        // (constant across the denoise loop) - build once on the compute
        // device, then serve from the cache. See `RopeMaskCache`.
        let key = (b, f_tokens, h_tokens, w_tokens, text_len);
        let cached = {
            let guard = self.rope_cache.lock().unwrap_or_else(|e| e.into_inner());
            match &*guard {
                Some(c) if c.key == key => Some((
                    c.x_cos.clone(),
                    c.x_sin.clone(),
                    c.cap_cos.clone(),
                    c.cap_sin.clone(),
                    c.unified_cos.clone(),
                    c.unified_sin.clone(),
                    c.x_attn_mask.clone(),
                )),
                _ => None,
            }
        };
        let (x_cos, x_sin, cap_cos, cap_sin, unified_cos, unified_sin, x_attn_mask) = match cached {
            Some(t) => t,
            None => {
                let dev = x.device();
                // Image position IDs + RoPE
                let x_pos_ids =
                    create_coordinate_grid((f_tokens, h_tokens, w_tokens), (text_len + 1, 0, 0))?;
                let (x_cos, x_sin) = self.rope_embedder.forward(&x_pos_ids)?;
                let (x_cos, x_sin) = (x_cos.to_device(&dev)?, x_sin.to_device(&dev)?);
                // Caption position IDs + RoPE
                let cap_pos_ids = create_coordinate_grid((text_len, 1, 1), (1, 0, 0))?;
                let (cap_cos, cap_sin) = self.rope_embedder.forward(&cap_pos_ids)?;
                let (cap_cos, cap_sin) = (cap_cos.to_device(&dev)?, cap_sin.to_device(&dev)?);
                // Unified RoPE: cat(index_select(ids_part)) ==
                // index_select(cat(ids_parts)) elementwise, so concatenate
                // the per-part cos/sin tables directly (avoids a u32 cat).
                let unified_cos = Tensor::cat(&[&x_cos, &cap_cos], 0)?;
                let unified_sin = Tensor::cat(&[&x_sin, &cap_sin], 0)?;
                // All-ones image attention mask (f32 0/1 in the native port)
                let x_attn_mask =
                    Tensor::from_vec_f32(vec![1f32; b * img_seq_len], (b, img_seq_len))?
                        .to_device(&dev)?;
                let mut guard = self.rope_cache.lock().unwrap_or_else(|e| e.into_inner());
                *guard = Some(RopeMaskCache {
                    key,
                    x_cos: x_cos.clone(),
                    x_sin: x_sin.clone(),
                    cap_cos: cap_cos.clone(),
                    cap_sin: cap_sin.clone(),
                    unified_cos: unified_cos.clone(),
                    unified_sin: unified_sin.clone(),
                    x_attn_mask: x_attn_mask.clone(),
                });
                (
                    x_cos,
                    x_sin,
                    cap_cos,
                    cap_sin,
                    unified_cos,
                    unified_sin,
                    x_attn_mask,
                )
            }
        };

        // 4. Caption embedding (content-dependent - recomputed per step)
        let cap_embedded = self
            .cap_embedder_linear
            .forward(&self.cap_embedder_norm.forward(cap_feats)?)?;
        let cap_attn_mask = cap_mask.to_dtype(DType::F32)?;

        // 7. Noise refiner (image, with modulation)
        for layer in &self.noise_refiner {
            x = layer.forward(&x, Some(&x_attn_mask), &x_cos, &x_sin, Some(&adaln_input))?;
        }

        // 8. Context refiner (caption, no modulation; recomputed per step  -
        // the facade hoists this into its per-image cache)
        let mut cap = cap_embedded;
        for layer in &self.context_refiner {
            cap = layer.forward(&cap, Some(&cap_attn_mask), &cap_cos, &cap_sin, None)?;
        }

        // 9. Concatenate image and text: [image_tokens, text_tokens]
        let mut unified = Tensor::cat(&[&x, &cap], 1)?;

        // 10. Unified mask (the cos/sin tables come from the cache above)
        let unified_attn_mask = Tensor::cat(&[&x_attn_mask, &cap_attn_mask], 1)?;

        // 11. Main transformer layers.
        //
        // On a single-device model (`layer_devices` is None) this is the loop it has
        // always been: one enum test per block, no move, no copy. On a stack built
        // across devices the stream and everything the block reads alongside it - the
        // rotary tables, the mask, the modulation - are moved once per SEGMENT, not
        // once per block, which is the transfer a split actually pays.
        let stem_device = unified.device();
        let mut seg_cos = unified_cos;
        let mut seg_sin = unified_sin;
        let mut seg_mask = unified_attn_mask;
        let mut seg_adaln = adaln_input.clone();
        // THE CROSSING IS SHARED MACHINERY, not this model's. Comparing, moving and
        // remembering is the same at every segment boundary in this repository; what a
        // model knows that the crossing does not is WHAT TRAVELS - here the stream and
        // everything a block reads beside it. So the list is local and the mechanism is
        // not, which is also what leaves one place to teach about a backend that is not
        // a device.
        let mut seg_device = stem_device.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            // A position with no block on this substrate belongs to another backend. There
            // is none wired yet, so reaching one is a bug in the loader rather than a case
            // to skip quietly - a silently dropped block returns a plausible image.
            let Some(layer) = layer.as_ref() else {
                return Err(Error(format!(
                    "block {i} has no native block and no backend claimed it"
                )));
            };
            if let Some(devs) = &self.layer_devices {
                crate::inference::place::plan::cross_to(
                    &mut seg_device,
                    &devs[i],
                    &mut [
                        &mut unified,
                        &mut seg_cos,
                        &mut seg_sin,
                        &mut seg_mask,
                        &mut seg_adaln,
                    ],
                )?;
            }
            unified = layer.forward(
                &unified,
                Some(&seg_mask),
                &seg_cos,
                &seg_sin,
                Some(&seg_adaln),
            )?;
            // SERIALIZE EACH BLOCK, but only on a stack built across devices.
            //
            // The blocks form a chain - block N+1 consumes N's output - so there is no
            // parallelism to lose. What there is to lose is memory: a free is
            // stream-ordered, and until the stream reaches it the pool can neither
            // return the block nor hand it to the next allocation, so it reserves
            // instead. Measured on a replayed trace, leaving the frees in flight cost
            // 2.4 GB; measured on this forward, a split held 1.14 GB per card more than
            // the orchestrator this replaced, which serializes exactly here.
            //
            // The single-device stack does NOT take it: there it would be a barrier per
            // block bought for nothing, and the three geometries that run whole are
            // bit-identical and unchanged without it.
            if let Some(devs) = &self.layer_devices {
                let _ = devs[i].synchronize();
            }
        }

        // 12. Final layer (only on image portion) - back where the stem is, since that
        // is where its weights and the modulation it reads were built.
        if self.layer_devices.is_some() {
            unified = unified.to_device(&stem_device)?;
        }
        // The image tokens are the head of the unified stream; the caption travelled with them
        // to be read, not to be drawn.
        let img_tokens = unified.narrow(1, 0, img_seq_len)?;
        let x_out = self.final_layer.forward(&img_tokens, &adaln_input)?;

        // 13. Unpatchify
        unpatchify(&x_out, orig_size, patch_size, f_patch_size, channels)
    }

    /// Get model configuration
    pub fn config(&self) -> &Config {
        &self.cfg
    }
}

// ---------------------------------------------------------------------------
// Tests: the helper math against the substrate implementations as an oracle.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
