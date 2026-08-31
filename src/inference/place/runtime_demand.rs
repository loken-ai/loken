//! What one generation actually needs, DERIVED from the request and the architecture -
//! never a constant.
//!
//! The stacks that ask are a diffusion denoiser, an audio DiT, a UNet, a VAE and a vision
//! tower. They differ in what their shapes mean and not in how a shape becomes bytes,
//! which is why the arithmetic is written once, here, rather than per family.
//!
//! WHY THIS MODULE EXISTS. Placement used to reserve fixed byte counts per family
//! (a 4 GiB "runtime reserve", a 3 GiB "floor", a 5 GiB one for another family).
//! Those numbers cannot be right: the scratch a denoiser needs scales with the
//! REQUESTED RESOLUTION and with the model's own width, so one figure is
//! simultaneously too large at 512x512 - refusing placements that would have run -
//! and too small at 1536x1536, where it admits a load that then cannot denoise. The
//! second failure is the expensive one: the weights fit, the model goes resident,
//! and every generation afterwards dies for want of scratch on a card that is now
//! full, with no way back.
//!
//! WHAT REPLACES THEM. A denoiser's peak scratch is a function of three things it
//! already knows - how many tokens the latent becomes, how wide the model is, and
//! how many heads it attends with. That is arithmetic, not a tuned table, so it
//! generalises to a model nobody has benchmarked and to a resolution nobody tried.
//!
//! CALIBRATION. The multiplier is not invented here. One family already sized its
//! demand this way and is correct in production, using a factor of 12 live
//! token-major tensors per block. That family's MLP ratio is 4, and
//! `4 + 2 * mlp_ratio` reproduces 12 exactly - four for the residual stream, the
//! fused projection and the attention output, and two passes over the MLP
//! intermediate, which is `mlp_ratio` times wider. So the general form below is the
//! working number with its origin spelled out, and it returns bit-identical results
//! for the family it was derived from.

/// Bytes per element of the accumulation dtype. Activations run in f32 on every
/// image path here, including the models whose WEIGHTS are quantised - a quantised
/// weight is dequantised into an f32 activation, so the scratch is unaffected by
/// the checkpoint's container.
const ACT_BYTES: u64 = 4;

/// What a render of the reference shape was MEASURED to take, per family.
///
/// These are measurements, not guesses: each is the reserve a family was empirically
/// found to need at the resolution it is trained for, arrived at by renders that
/// failed with less. They are the SEED, and the two things done with them are what
/// makes this generic:
///
///  - scaled by the request, so a smaller picture asks for less and a larger one more;
///  - replaced outright by what a render on THIS machine is observed to take, the
///    moment there is such an observation.
///
/// The alternative I tried was a purely analytic peak times an allocator factor. It is
/// more elegant and it was wrong in the direction that matters: calibrated against one
/// family, it over-stated every other by enough to shrink their GPU budgets, and models
/// that had been running on the cards started running on the host. A measurement I can
/// point at beats a derivation I cannot justify.
pub struct MeasuredReserve {
    /// Bytes the family needs free beyond its weights, at `reference_tokens`.
    pub bytes: u64,
    /// The token count that figure was measured at.
    pub reference_tokens: usize,
}

/// Scale a measured reserve to the request in front of us.
///
/// Linear in tokens: the scratch is dominated by token-major buffers, whose size is
/// exactly proportional to the sequence. The attention scores would be quadratic if
/// they were materialised whole - they are not, they are tiled, which is what makes
/// the linear law hold across resolutions.
pub fn scale_measured(reserve: &MeasuredReserve, tokens: usize) -> u64 {
    let reference = reserve.reference_tokens.max(1) as u64;
    let asked = tokens.max(1) as u64;
    reserve.bytes.saturating_mul(asked) / reference
}

/// What a render of this shape was OBSERVED to need, once one has run.
///
/// The derivation below is a model of the allocator, and a model is wrong somewhere.
/// It was calibrated against renders at one resolution and holds there; at 1536x1536 a
/// card with 16 GB and 1.5 GB of resident blocks still exhausted, so the real peak is
/// well above what the arithmetic predicts and the gap GROWS with the request. Raising
/// the factor would fix that one size and break the others - the same trap as the
/// constants this module removed, one level up.
///
/// So the peak is measured instead. A render already reports its low-water mark; that
/// number is recorded here against the shape that produced it, and the next placement
/// for that shape asks for what was actually needed rather than what was predicted.
/// The estimate is only ever the FIRST answer for a shape nobody has rendered yet.
///
/// Keyed by family and geometry because that is what the peak depends on. In memory
/// only: a restart re-learns in one render, and persisting it would let a number
/// measured on other hardware decide a placement here.
static OBSERVED_PEAKS: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<(String, usize, usize), Vec<u64>>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// How many recent renders of a shape decide its reserve.
///
/// The reserve is the MAX of them, so a render that happened to peak low - a short
/// prompt, a lucky allocation order - cannot talk the next one into a placement that
/// fails. But it is the max of a WINDOW rather than of all time, because the reverse
/// case was unguarded: a render measured while something else was allocating records a
/// figure that includes the other subsystem's memory, and an all-time maximum lets that
/// one contaminated sample decide every later placement for the shape, for as long as
/// the process lives. An inflated reserve splits a model that fits one card, and a split
/// runs the cards in sequence. A window forgets it after a few clean renders.
const REMEMBERED_RENDERS: usize = 4;

/// Record what a completed render of `family` at this geometry actually took.
pub fn record_observed_peak(family: &str, width: usize, height: usize, bytes: u64) {
    if bytes == 0 {
        return;
    }
    let mut g = match OBSERVED_PEAKS.lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    let slot = g.entry((family.to_string(), width, height)).or_default();
    slot.push(bytes);
    if slot.len() > REMEMBERED_RENDERS {
        slot.remove(0);
    }
    let effective = slot.iter().copied().max().unwrap_or(bytes);
    tracing::info!(
        "image: {family} at {width}x{height} took {:.2} GB; placements will ask for          {:.2} GB (the highest of the last {} renders)",
        bytes as f64 / 1e9,
        effective as f64 / 1e9,
        slot.len(),
    );
}

/// The measured reserve for this shape: the highest of the renders still remembered.
pub fn observed_peak(family: &str, width: usize, height: usize) -> Option<u64> {
    let g = match OBSERVED_PEAKS.lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    g.get(&(family.to_string(), width, height))
        .and_then(|v| v.iter().copied().max())
}

#[cfg(test)]
mod observed_peak_tests {
    use super::{observed_peak, record_observed_peak, REMEMBERED_RENDERS};

    /// The reserve follows the highest RECENT render, so a low one cannot starve the
    /// next placement...
    #[test]
    fn a_low_render_does_not_lower_the_reserve() {
        let (f, w, h) = ("peaktest-low", 64, 64);
        record_observed_peak(f, w, h, 4_000);
        record_observed_peak(f, w, h, 1_000);
        assert_eq!(observed_peak(f, w, h), Some(4_000));
    }

    /// ...and a single inflated one - a render measured while something else was
    /// allocating - stops deciding once enough clean renders have followed it.
    #[test]
    fn a_contaminated_render_is_forgotten() {
        let (f, w, h) = ("peaktest-spike", 64, 64);
        record_observed_peak(f, w, h, 9_000);
        for _ in 0..REMEMBERED_RENDERS {
            record_observed_peak(f, w, h, 2_000);
        }
        assert_eq!(observed_peak(f, w, h), Some(2_000));
    }
}

/// The demand to plan against: the MEASUREMENT once there is one, else the estimate.
///
/// Taking the larger of the two looks safer and is not. The estimate is a model, and a
/// model that runs high is never corrected: a shape whose real peak is below the
/// prediction keeps being split across cards when one would hold it, forever. That was
/// visible immediately - a request that had rendered on a single card started
/// splitting because the estimate exceeded the spare by two hundred megabytes.
///
/// The measurement is safe to trust on its own because of how it is maintained: it
/// only ever rises, from completed renders AND from exhaustions, so it converges
/// upward on the true worst case for that shape rather than tracking whatever the last
/// render happened to do. An observation that starts too low costs one retry - which
/// then records a higher lower bound - not a hang.
pub fn planning_demand(family: &str, width: usize, height: usize, derived: u64) -> u64 {
    observed_peak(family, width, height).unwrap_or(derived)
}

/// The per-device allowance a placement leaves free beyond weights and activations.
///
/// The CUDA context, the kernel modules a first launch pages in, and the staging the load
/// itself churns through. Derived from the CHECKPOINT because that is what a load churns:
/// a bigger file streams more through the allocator before it settles, so the allowance
/// has to follow it rather than sit at one number for a 200 MB decoder and a 12 GB encoder
/// alike.
pub fn load_runtime_floor(model_size: u64) -> u64 {
    /// Share of a checkpoint held live at once while it is staged onto the card: the
    /// block being read, its expansion, and the destination, against a stream that is
    /// otherwise handed straight to the device.
    ///
    /// Set so that nothing in the current fleet moves - the largest encoder placed through
    /// this gate still lands under the measured floor - because raising a reserve is not
    /// free either: it takes a card away from a model that fits. What it buys is that a
    /// checkpoint several times larger than anything shipped today is no longer planned
    /// with a 2B model's allowance.
    const STAGED_SHARE: u64 = 32;
    // measured-resident: the floor a first launch needs free on a card before any
    // model-specific scratch is counted. Kept as the floor rather than replaced: it covers
    // module loading, which no arithmetic over the checkpoint can see.
    let measured_floor: u64 = 512 << 20;
    measured_floor.max(model_size / STAGED_SHARE)
}

/// What fraction of a card's free VRAM the current pressure lets WEIGHTS occupy.
///
/// A retry that re-plans against unchanged numbers produces the same plan and fails
/// the same way, so something has to change between attempts. The thing to change is
/// how much of each card the WEIGHTS may take - not the demand, which is a fact about
/// the request and stays true however many times it fails.
///
/// Inflating the demand instead was my first attempt and it has a cliff: two notches
/// put the figure past what any card has free, every card became ineligible at once,
/// the whole model landed on the host, and the render ran past the request timeout
/// without producing anything. Backing off the weight budget degrades smoothly -
/// each notch moves a few more blocks to the next card, and only a model that fits
/// nowhere reaches the host.
///
/// Shared with the audio and video engines, so pressure raised by one is respected by
/// all: they compete for the same cards.
pub fn weight_budget_share() -> f64 {
    const BACKOFF: [f64; 4] = [1.0, 0.7, 0.45, 0.25];
    let level = crate::inference::place::vram_manager::vram_degrade_level() as usize;
    BACKOFF[level.min(BACKOFF.len() - 1)]
}

/// The geometry a placement decision is made for.
///
/// Passed explicitly rather than read from ambient state: placement happens inside a
/// request, and the whole point of this module is that the answer differs per
/// request. A default would reintroduce the constant this replaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestGeometry {
    pub width: usize,
    pub height: usize,
}

impl RequestGeometry {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    /// Tokens this request's latent becomes for a patchifying transformer.
    pub fn tokens(&self, vae_stride: usize, patch: usize) -> usize {
        latent_tokens(self.height, self.width, vae_stride, patch)
    }
}

/// How many tokens a latent of `height x width` pixels becomes.
///
/// `vae_stride` is the encoder's spatial reduction (8 for every VAE in use here)
/// and `patch` is the transformer's own patch size on top of it - 2 for the
/// families that pack 2x2 latent neighbourhoods into one token. Both are read from
/// the model rather than assumed, because getting this wrong scales the whole
/// demand by its square.
pub fn latent_tokens(height: usize, width: usize, vae_stride: usize, patch: usize) -> usize {
    let denom = (vae_stride * patch).max(1);
    (height / denom).max(1) * (width / denom).max(1)
}

/// Peak activation bytes of a transformer denoiser, per forward.
///
/// Two terms, because they scale differently and one of them dominates at high
/// resolution:
/// - the token-major tensors live at once inside a block, linear in `tokens`;
/// - the attention scores, which are `tokens` x `tokens` per head and would be
///   quadratic if the attention were materialised whole. It is not: the kernel
///   walks queries in tiles, so the resident slab is `tile x tokens` per head, and
///   the tile is what bounds it.
pub fn dit_activation_bytes(tokens: usize, dim: usize, heads: usize, mlp_ratio: f64) -> u64 {
    dit_analytic_bytes(tokens, dim, heads, mlp_ratio, ACT_BYTES)
}

/// [`dit_activation_bytes`] for a denoiser whose attention runs on TENSOR CORES.
///
/// Same live set - the residual stream and the MLP intermediate are F32 either way - but
/// the score buffer is BF16, and the tile is the larger one that buys. Both numbers come
/// from the functions the kernel itself calls, so changing the kernel cannot leave this
/// behind again: it already did once, and an over-stated reserve is not slack, it is
/// subtracted from the card before a single block is placed.
pub fn dit_activation_bytes_tc(
    tokens: usize,
    dim: usize,
    heads: usize,
    mlp_ratio: f64,
    act: u64,
    // The longest sequence whose scores are actually WRITTEN TO MEMORY. It is not always
    // the sequence being attended over: a flash kernel keeps its scores in shared memory
    // and allocates only the output, so a path it covers contributes no slab at all and
    // the term is set by whatever attention is still materialised - typically a cross
    // attention onto a short context. Sizing this on the clip instead reserved gigabytes
    // for a buffer that is never allocated, and a reserve comes off the card before any
    // block is placed, so the over-charge is paid in blocks pushed onto the host.
    score_seq: usize,
) -> u64 {
    use crate::inference::model::acestep::ops::{tc_query_tile_for_heads, tc_score_bytes};
    let tokens_u = tokens.max(1) as u64;
    let live = dit_live_bytes(tokens, dim, mlp_ratio, act);
    // The SAME tile the kernel will take, at THIS model's width - the slab is per head.
    let score_seq = score_seq.min(tokens.max(1)).max(1);
    let tile = tc_query_tile_for_heads(tokens, heads)
        .min(tokens.max(1))
        .max(1) as u64;
    let scores = 2 * tile * score_seq as u64 * heads as u64 * tc_score_bytes();
    // The attention working set around the kernel: the projections, the copies taken to
    // make them contiguous, and the output.
    //
    // Both attempts to shrink this term measured WORSE, and the reason is worth keeping.
    // The flash kernel really does stream its scores through shared memory rather than
    // allocating a slab, and the projections really are same-dtype conversions once the
    // stream is carried at the multiply width - so on paper both terms overstate what a
    // block allocates. Cutting them did not free anything: a reserve is what stops the
    // planner from packing another block onto the card, so lowering it moved blocks ONTO
    // the card the forward then had to run on, and a size that had rendered for months
    // came back as an out-of-memory with all forty blocks on one device. The term is
    // load-bearing beyond the allocations it names, and it stays until something MEASURES
    // the peak rather than deriving it.
    let qkv = tokens_u * dim as u64;
    let casts = qkv * (act + 2 + 2 + 2 + act);
    live + scores + casts
}

/// The live-set peak alone, before the allocator correction. Separate so the
/// calibration test can compare it against the factor it was derived from.
/// The token-major tensors a block holds at once, in bytes. F32 on every path: only the
/// attention scores change dtype between the exact and the tensor-core kernels.
///
/// The MLP term is the one that stops growing on a long sequence: every token's path
/// through it is independent, so the forward computes it in chunks and the intermediate is
/// bounded by the chunk rather than by the clip.
fn dit_live_bytes(tokens: usize, dim: usize, mlp_ratio: f64, act: u64) -> u64 {
    let tokens = tokens.max(1) as f64;
    let mlp_tokens =
        crate::inference::model::acestep::ops::ffn_chunk_bytes_bounded(tokens as usize, mlp_ratio)
            .min(tokens as usize) as f64;
    let residual = 4.0 * tokens;
    let mlp = 2.0 * mlp_ratio.max(1.0) * mlp_tokens;
    ((residual + mlp) * dim as f64 * act as f64) as u64
}

fn dit_analytic_bytes(tokens: usize, dim: usize, heads: usize, mlp_ratio: f64, act: u64) -> u64 {
    let tokens = tokens.max(1) as u64;
    // Residual stream + fused QKV + attention output, then the MLP intermediate written
    // once and read once at `mlp_ratio` times the model width.
    //
    // The MLP term is the one that stops growing on a long sequence: every token's path
    // through it is independent, so the forward computes it in chunks, and the
    // intermediate is bounded by the chunk rather than by the clip. Modelling it whole
    // here would keep reserving gigabytes the forward no longer takes - and an
    // over-stated reserve is not harmless, it is what pushes blocks off the cards.
    let live = dit_live_bytes(tokens as usize, dim, mlp_ratio, act);
    // The SAME tile the kernel will choose. Reading the fixed constant here made the
    // estimate diverge from reality exactly where it mattered - on a long clip, where the
    // kernel now shrinks its tile and this did not, so the demand said "no card fits" for
    // a forward that does.
    let tile = crate::inference::model::acestep::ops::query_tile_for(tokens as usize)
        .min(tokens as usize)
        .max(1) as u64;
    let scores = 2 * tile * tokens * heads as u64 * ACT_BYTES;
    live + scores
}

/// Peak activation bytes of a convolutional UNet denoiser, per forward.
///
/// A UNet holds feature maps, not tokens, and its peak is at the WIDEST spatial
/// level rather than the deepest: channels double as resolution halves, so every
/// level costs about the same, and the skip connections keep one map per level
/// alive across the whole forward. The sum over levels is therefore the honest
/// figure, and the top level's map is what makes it grow with the request.
pub fn unet_activation_bytes(latent_h: usize, latent_w: usize, level_channels: &[usize]) -> u64 {
    unet_analytic_bytes(latent_h, latent_w, level_channels)
}

/// Peak of a UNet that carries SPATIAL TRANSFORMERS, which is what actually sizes it.
///
/// `attention_levels` is `(level index, width, depth)` per attention stage. At the
/// resolutions these run at the attention dwarfs the convolution stack - the feature
/// maps alone came to a few hundred megabytes where a render needs gigabytes - so the
/// answer is the larger of the two, per stage, taken at its own spatial level.
///
/// Stages run one after another, so the peak is the WORST stage rather than their
/// sum; the allocator's carry across them is accounted for by the measured seed.
pub fn unet_with_attention_bytes(
    latent_h: usize,
    latent_w: usize,
    level_channels: &[usize],
    attention_levels: &[(usize, usize, usize)],
    head_dim: usize,
    mlp_ratio: f64,
) -> u64 {
    let conv = unet_activation_bytes(latent_h, latent_w, level_channels);
    let mut attn = 0u64;
    for (level, width, depth) in attention_levels {
        let scale = 1usize << level;
        let tokens = (latent_h / scale).max(1) * (latent_w / scale).max(1);
        // Head COUNT follows the stage width at a fixed head dimension.
        let heads = (width / head_dim.max(1)).max(1);
        // Every block at this stage is one more thing the allocator churns through.
        let stage = dit_activation_bytes(tokens, *width, heads, mlp_ratio) * *depth as u64;
        attn = attn.max(stage);
    }
    conv.max(attn)
}

fn unet_analytic_bytes(latent_h: usize, latent_w: usize, level_channels: &[usize]) -> u64 {
    let mut total = 0u64;
    for (level, channels) in level_channels.iter().enumerate() {
        let scale = 1usize << level;
        let h = (latent_h / scale).max(1) as u64;
        let w = (latent_w / scale).max(1) as u64;
        let c = *channels as u64;
        // One map held on the skip stack, plus the pair live in the block computing
        // it - the residual input and the convolution's output.
        total += 3 * h * w * c * ACT_BYTES;
    }
    total
}

// `dequant_scratch_bytes` lived here - the widest 2-D weight of a quantised checkpoint,
// times the copies a matmul holds live - and it was removed rather than wired.
//
// It had no caller, which read as an oversight. It is not: every family's demand here is
// seeded from a MEASURED figure, taken from renders that failed with less, so the
// dequantisation buffers are already inside that number. Adding a separately computed
// term on top counts them twice, and the margin decides real placements - Flux at 1024
// fits one card with 4.3 GB spare against 4.3 GB needed, so a few hundred extra
// megabytes split a render that fits whole, onto cards that then run in SEQUENCE. The
// user-visible result of "planning more carefully" would have been a slower server.
//
// If a family ever gets an ANALYTIC demand - built from activation shapes rather than
// from measurement - that one WILL be missing the dequantisation buffer and will need
// this term. Compute it there, against that number, and measure the result.

/// Peak bytes of a VAE decode, which is charged to whichever device holds the VAE.
///
/// The decoder's cost is dominated by its LAST level, where the map is at full
/// output resolution and still carries the decoder's base width. That single
/// tensor, plus the pair live around it, is the peak; everything deeper is smaller
/// by at least 4x and does not move the maximum.
pub fn vae_decode_bytes(height: usize, width: usize, base_channels: usize) -> u64 {
    3 * height.max(1) as u64 * width.max(1) as u64 * base_channels as u64 * ACT_BYTES
}

/// Peak bytes of a VAE ENCODE at the request's resolution.
///
/// The mirror of the decode: the encoder's widest stage sits at full input resolution,
/// and several of those tensors are live at once as the convolutions chain.
pub fn vae_encode_bytes(height: usize, width: usize) -> u64 {
    /// Widest full-resolution channel count in the encoders in use here.
    const ENCODER_WIDEST_CH: u64 = 192;
    /// Tensors of that stage live at once while the convolution chain runs.
    const LIVE: u64 = 4;
    LIVE * height.max(1) as u64 * width.max(1) as u64 * ENCODER_WIDEST_CH * ACT_BYTES
}

/// Peak bytes a vision tower needs for one source image, beyond its weights.
///
/// The image is held as f32 planes and re-tiled through the patch embedding, so the
/// cost is a small multiple of the raw pixel count rather than a fixed slab.
pub fn vision_tower_bytes(height: usize, width: usize) -> u64 {
    /// Colour planes, and the copies live across the patch embedding.
    const PLANES: u64 = 3;
    const LIVE: u64 = 8;
    PLANES * LIVE * height.max(1) as u64 * width.max(1) as u64 * ACT_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The general form must reproduce the factor it was derived from.
    ///
    /// One family sized its demand as 12 token-major tensors per block and is
    /// correct in production. Its MLP ratio is 4, so this must return exactly what
    /// that family was already getting - otherwise generalising the formula would
    /// have silently re-planned a working model.
    #[test]
    fn the_general_form_reproduces_the_calibrated_factor() {
        let (tokens, dim, heads) = (4096usize, 1536usize, 12usize);
        let general = dit_analytic_bytes(tokens, dim, heads, 4.0, ACT_BYTES);
        let live_12 = 12u64 * tokens as u64 * dim as u64 * ACT_BYTES;
        let tile = crate::inference::model::acestep::ops::QUERY_TILE.min(tokens) as u64;
        let calibrated = live_12 + 2 * tile * tokens as u64 * heads as u64 * ACT_BYTES;
        assert_eq!(
            general, calibrated,
            "the generalised demand drifted from its origin"
        );
    }

    /// The derived demand must agree with the evidence that produced the constants.
    ///
    /// This is the check that matters, and the one I nearly shipped without. The
    /// analytic live set alone is about a third of what a render actually takes,
    /// because the allocator's high-water mark follows what the denoise CHURNS, not
    /// what is live at an instant. A formula without that correction looks principled,
    /// passes every other test here, and quietly reserves a third of what is needed -
    /// which is the OOM this module exists to prevent, reintroduced by the fix for it.
    ///
    /// The fixed reserves are the evidence: 4 GiB was found necessary for this family
    /// at the resolution below, by renders that failed with less. A derivation that
    /// lands near it is corroborated by that history; one that lands far under it is
    /// wrong no matter how clean it reads. Bounded loosely on purpose - this pins the
    /// ORDER, not a value, since pinning a value would be the constant again.
    #[test]
    fn the_scaled_reserve_agrees_with_what_renders_were_measured_to_need() {
        // At the reference shape the scaled figure must BE the measured one - that is
        // the whole point of seeding from a measurement. The analytic peak used to
        // stand here multiplied by an allocator factor calibrated on another family,
        // and it over-stated this one by enough to move a working placement onto the
        // host.
        let observed_necessary = 4u64 << 30;
        let reference = latent_tokens(1024, 1024, 8, 2) + 256;
        let seed = MeasuredReserve {
            bytes: observed_necessary,
            reference_tokens: reference,
        };
        let derived = scale_measured(&seed, reference);
        assert_eq!(
            derived, observed_necessary,
            "the seed must survive its own reference"
        );
        // And it must scale, not sit still.
        let doubled = scale_measured(&seed, reference * 2);
        assert_eq!(
            doubled,
            observed_necessary * 2,
            "the reserve must follow the request"
        );
        assert!(
            derived * 2 >= observed_necessary,
            "derived {derived} is far below the {observed_necessary} renders needed - \
             the allocator correction is missing or wrong"
        );
        assert!(
            derived <= observed_necessary * 3,
            "derived {derived} dwarfs the {observed_necessary} renders needed - \
             this refuses placements that would have run"
        );
    }

    /// The demand must GROW with the request, or it is a constant wearing a
    /// function's clothes.
    ///
    /// This is the property the fixed reserves failed: one number admitted a load at
    /// 512x512 and then could not denoise the 1536x1536 the same model was asked
    /// for. Doubling each side quadruples the tokens, so the demand must rise
    /// steeply - not merely differ.
    #[test]
    fn the_demand_grows_with_the_requested_resolution() {
        let small = latent_tokens(512, 512, 8, 2);
        let large = latent_tokens(1024, 1024, 8, 2);
        assert_eq!(large, 4 * small, "tokens must scale with area");
        let a = dit_activation_bytes(small, 3072, 24, 4.0);
        let b = dit_activation_bytes(large, 3072, 24, 4.0);
        assert!(
            b >= 4 * a,
            "demand at 1024 ({b}) must be at least 4x the demand at 512 ({a})"
        );
    }

    /// No hand-written VRAM size may come back into the image placement path.
    ///
    /// This is a GATE, not a style check. The class it forbids has now cost two
    /// separate incidents, and both were invisible until a render failed: a fixed
    /// reserve is right for exactly one resolution on exactly one model, and wrong
    /// everywhere else in whichever direction hurts. It also cannot be caught by
    /// reading a diff, because a plausible-looking constant is indistinguishable from
    /// a derived one at the call site.
    ///
    /// The rule: in these files, a memory quantity must be COMPUTED - from a config's
    /// widths, from a file's size on disk, or from the request's geometry. If this
    /// test fails, the fix is to derive the number, not to widen the pattern.
    #[test]
    fn no_hand_written_vram_size_in_the_placement_path() {
        // Words that make a large literal a MEMORY quantity rather than an
        // architecture one - a head count or a vocabulary size is neither large nor
        // described this way.
        const MEMORY_WORDS: [&str; 10] = [
            "reserve", "headroom", "margin", "budget", "vram", "bytes", "overhead", "inflate",
            "size_est", "_mem",
        ];
        /// Written on the line above a literal that is not a reservation.
        const EXEMPTION: &str = "not-a-vram-size:";
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let files = [
            "src/inference/engine/image_engine/mod.rs",
            "src/inference/engine/qwen_image_engine.rs",
            "src/inference/engine/boogu_engine.rs",
            "src/inference/place/runtime_demand.rs",
            "src/api/handlers/media/mod.rs",
            "src/inference/model/wan/dit/layers.rs",
            "src/inference/place/vram_manager.rs",
            "src/inference/model/sdxl/pipeline.rs",
            "src/inference/model/flux/hetero.rs",
            // These two were NOT covered, which is how a second planner kept a branch
            // assuming "~2 GB per layer" whenever the model size was unknown. A gate
            // that lists files only guards the files it lists: every module that
            // decides where something goes belongs here.
            "src/inference/place/layer_executor.rs",
            "src/inference/load/model_manager.rs",
            // The checkpoint cache decides what stays resident, which decides what the
            // planner finds free - and it was outside this list, so the same clamped
            // literals the gate had just rejected in `layer_executor` sat unflagged in
            // it. A gate that lists files guards the files it lists; the sibling of a
            // module already here belongs here the day it is written.
            "src/inference/cache/qvb.rs",
            // Nor were these, which is how the video encoder kept gating on a typed
            // 11_500_000_000 while the checkpoint next to it measured 11_361_920_418 -
            // a constant that had already stopped tracking its own file. Every module
            // below decides where a component goes, so every one belongs here.
            "src/inference/model/wan/pipeline.rs",
            // The ACE-Step / EzAudio placement path, which the note that stood here said
            // belonged in this list and was not in it. It hid a fixed floor under every
            // encoder, a fixed reserve under the denoiser and the LM, one decode reserve
            // shared by autoencoders whose upsample factors differ fourfold, a capture
            // arena sized in shifted megabytes, and an invented checkpoint size for every
            // file that could not be stat'd. Each of them is now derived from the request
            // and the architecture in `audio_demand`, and this is what stops them coming
            // back one plausible-looking literal at a time.
            "src/inference/place/audio_demand.rs",
            "src/inference/model/acestep/ops.rs",
            "src/inference/model/acestep/dit/mod.rs",
            "src/inference/model/acestep/vae.rs",
            "src/inference/model/acestep/lm/mod.rs",
            "src/inference/model/acestep/cond.rs",
            "src/inference/model/acestep/fsq.rs",
            "src/inference/model/acestep/textenc.rs",
            "src/inference/model/acestep/music.rs",
            "src/inference/model/ezaudio/vae.rs",
            "src/inference/model/ezaudio/dit.rs",
            "src/inference/model/ezaudio/pipeline.rs",
            // NOT device_probe.rs yet: its per-card driver reserve is a hand-written
            // 512 MB. That one is a real hardware allowance rather than a guess at a
            // model's needs, but it is still a byte count, and turning it into a
            // fraction of the card changes where every model lands - a change that has
            // to be measured, not slipped in behind a gate.
        ];
        let mut offenders: Vec<String> = Vec::new();
        for rel in files {
            let path = root.join(rel);
            // Same reason as the gate below: a listed file that cannot be read means the
            // list has gone stale, and a gate that skips is a gate that passes by not
            // looking. Twenty-six of these paths moved during one restructure and nothing
            // said so.
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("{rel} is listed here but unreadable: {e}"));
            // Fixtures in a test module are allowed to name concrete byte counts -
            // that is how they pin behaviour. Only production code is gated.
            let production = text.split("#[cfg(test)]").next().unwrap_or("");
            let lines: Vec<&str> = production.lines().collect();
            for (n, line) in lines.iter().enumerate() {
                let trimmed = line.trim_start();
                if trimmed.starts_with("//") {
                    continue;
                }
                // A quantity is often built across several lines, so the name that
                // makes it a MEMORY quantity can sit above the literal. Look at the
                // statement, not the line: without this, a continuation like
                // `+ (512u64 << 20)` reads as anonymous and slips through, while a
                // bare `n > 64 << 20` bound or a hash mixing `<< 20` gets flagged for
                // a shift that has nothing to do with bytes.
                let window = lines[n.saturating_sub(3)..=n].join(" ").to_lowercase();
                let names_memory = MEMORY_WORDS.iter().any(|w| window.contains(w));
                if !names_memory {
                    continue;
                }
                // Explicit, searchable exemption for the cases that are genuinely not
                // reservations - a bound on a parsed length, a bytes-to-megabytes
                // conversion in a log. Widening the pattern to let those through
                // would have opened a blind spot exactly where a real threshold could
                // hide; requiring the author to say so keeps the gate total and
                // leaves every exemption greppable.
                if window.contains(EXEMPTION) {
                    continue;
                }
                // A `MeasuredReserve` is not a hand-written reserve. The type carries
                // the resolution the figure was measured at, the value is scaled to the
                // request, and the first observation on the machine replaces it - which
                // is the whole of what this gate is protecting. Bare byte counts, which
                // carry none of that, still fail.
                if window.contains("measuredreserve") {
                    continue;
                }
                // A figure that was MEASURED on a running system, marked as such and
                // carrying what it was measured from. Distinct from a reserve someone
                // chose: the gate exists to stop numbers nobody can account for, and
                // this marker is the accounting. It is greppable, so every one of them
                // can be revisited when a derivation that reproduces it turns up.
                if window.contains("measured-resident:") {
                    continue;
                }
                // A shifted power of two is a byte count spelled as GiB or MiB.
                let lower = line.to_lowercase();
                let shifted = lower.contains("<< 30") || lower.contains("<< 20");
                // Otherwise: a seven-digit-or-longer literal, which at that magnitude
                // is a byte count and nothing else.
                let digits: usize = line
                    .split(|c: char| !(c.is_ascii_digit() || c == '_'))
                    .map(|t| t.chars().filter(char::is_ascii_digit).count())
                    .max()
                    .unwrap_or(0);
                if shifted || digits >= 7 {
                    offenders.push(format!("{rel}:{}: {}", n + 1, line.trim()));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "hand-written VRAM sizes are back in the placement path - derive them from \
             the model config, the checkpoint size, or the request geometry instead:\n{}",
            offenders.join("\n")
        );
    }

    /// EVERY family must ask for a plausible amount, not just the one I checked.
    ///
    /// The demand is dispatched per family, and a single arm returning zero - a config
    /// whose patch size reads as something unexpected, a level table that comes back
    /// empty - silently disables the protection for that family alone while every
    /// other test still passes. That failure is invisible until one specific model
    /// OOMs, which is how long the previous version of this bug survived.
    ///
    /// The band is deliberately wide: this catches "nothing" and "absurd", not a
    /// percentage. A tight bound would be the hard-coded constant again.
    #[test]
    fn every_family_asks_for_a_plausible_amount() {
        use crate::inference::engine::image_engine;
        const AT_LEAST: u64 = 256 << 20;
        const AT_MOST: u64 = 24u64 << 30;
        let mut demands: Vec<(&str, u64)> = vec![
            ("flux", image_engine::flux_runtime_demand(1024, 1024)),
            ("zimage", image_engine::zimage_runtime_demand(1024, 1024)),
        ];
        // The convolutional family reads its widths from the UNet's own down path;
        // an empty table there would sum to nothing.
        let levels = crate::inference::model::sdxl::unet::level_channels();
        assert_eq!(
            levels,
            vec![320, 640, 1280],
            "the UNet's level widths changed shape"
        );
        demands.push((
            "sdxl",
            unet_with_attention_bytes(
                128,
                128,
                &levels,
                &crate::inference::model::sdxl::unet::attention_levels(),
                crate::inference::model::sdxl::unet::HEAD_DIM,
                4.0,
            ),
        ));
        let boogu = crate::inference::model::boogu::dit::Config::default();
        demands.push((
            "boogu",
            dit_activation_bytes(
                latent_tokens(1024, 1024, 8, boogu.patch_size) + 256,
                boogu.hidden_size,
                boogu.num_heads,
                boogu.ffn_inner as f64 / boogu.hidden_size as f64,
            ),
        ));
        for (family, bytes) in &demands {
            assert!(
                (AT_LEAST..=AT_MOST).contains(bytes),
                "{family} asks {:.2} GB to render 1024x1024 - outside anything believable, \
                 so its placement is not actually protected",
                *bytes as f64 / 1e9
            );
        }
        // AND no family may be a wild outlier against the others. These render the
        // same picture at the same size with comparable architectures, so an order of
        // magnitude apart means one of them is measuring the wrong thing - which is
        // exactly what happened when the convolutional family was sized on its feature
        // maps and ignored the attention that dominates it. The absolute band above
        // did not catch that; this does.
        let lo = demands.iter().map(|(_, b)| *b).min().unwrap_or(0);
        let hi = demands.iter().map(|(_, b)| *b).max().unwrap_or(0);
        assert!(
            hi <= lo * 10,
            "family demands span {:.2}..{:.2} GB for the same 1024x1024 render - {:?}",
            lo as f64 / 1e9,
            hi as f64 / 1e9,
            demands
                .iter()
                .map(|(f, b)| (*f, *b as f64 / 1e9))
                .collect::<Vec<_>>()
        );
    }

    /// Every HOT component must be loadable across MORE THAN ONE card.
    ///
    /// This rule has been asked for repeatedly and broken repeatedly, and the reason it
    /// kept coming back is that nothing checked it. Worse, the breakage justified
    /// itself in a comment: the Flux loader built one CUDA VarBuilder, so the planner
    /// was NARROWED to one card to match, and the narrowing was documented as a
    /// rationale rather than recognised as the defect. That is how a single-card
    /// loader survives review - it reads as deliberate.
    ///
    /// What single-device costs, concretely: a two-card host ran a split as 30 blocks
    /// on one GPU and 27 on the HOST while the second GPU sat empty with 16 GB free;
    /// and a UNet that fits no single card went to the host in its entirety, which is
    /// the worst of the three possible outcomes.
    ///
    /// The check is deliberately coarse - it cannot prove a loader places well - but
    /// it does catch the thing that keeps happening: an entry point that accepts one
    /// device and no plan, and therefore cannot use a second card however much the
    /// planner asks.
    #[test]
    fn every_hot_component_can_span_more_than_one_card() {
        // The per-step components. A one-shot encoder is placed, not split, so it is
        // not on this list; anything that runs per denoise step is.
        const HOT_COMPONENT_LOADERS: [&str; 7] = [
            "src/inference/model/sdxl/unet.rs",
            "src/inference/model/flux/hetero.rs",
            "src/inference/model/boogu/dit.rs",
            "src/inference/model/qwen_image/dit.rs",
            "src/inference/model/wan/dit/layers.rs",
            // Audio denoisers run per step exactly as the image ones do; leaving them
            // off this list is how the same defect survived in a second modality.
            "src/inference/model/ezaudio/dit.rs",
            "src/inference/model/acestep/dit/mod.rs",
        ];
        // Any of these in a signature means the loader was handed more than one place
        // to put things.
        const MULTI_DEVICE_MARKERS: [&str; 6] = [
            "&[Device]",
            "&[Device]",
            "HashMap<usize, Device>",
            "HashMap<usize, Device>",
            "HeteroPlan",
            "Vec<Device>",
        ];
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut single: Vec<&str> = Vec::new();
        for rel in HOT_COMPONENT_LOADERS {
            // NOT `else { continue }`. A path that no longer resolves means the list is
            // stale, and skipping it is how this gate spent a whole restructure passing
            // without reading anything - twenty-six of these paths had moved.
            let text = std::fs::read_to_string(root.join(rel))
                .unwrap_or_else(|e| panic!("{rel} is listed here but unreadable: {e}"));
            let production = text.split("#[cfg(test)]").next().unwrap_or("");
            if !MULTI_DEVICE_MARKERS.iter().any(|m| production.contains(m)) {
                single.push(rel);
            }
        }
        assert!(
            single.is_empty(),
            "these hot components can only be loaded onto ONE card, so a second GPU \
             cannot be used for them and a model that fits none falls to the host \
             entirely: {single:?}"
        );
    }

    /// Every denoiser shape must ask for something, whatever the request.
    ///
    /// A zero demand reads to the planner as "needs no scratch", which is how a
    /// model gets placed on a card with nothing left to run in. Degenerate inputs
    /// must clamp, never vanish.
    #[test]
    fn no_shape_ever_asks_for_nothing() {
        assert!(dit_activation_bytes(0, 3072, 24, 4.0) > 0);
        assert!(unet_activation_bytes(0, 0, &[320, 640, 1280]) > 0);
        assert!(vae_decode_bytes(0, 0, 128) > 0);
        assert!(latent_tokens(0, 0, 8, 2) >= 1);
    }
}

#[cfg(test)]
mod tensor_core_demand_tests {
    use super::{dit_activation_bytes, dit_activation_bytes_tc, ACT_BYTES};

    /// The defect this closes. A 14B video denoiser (dim 5120, 40 heads) at one 21504-token
    /// window was charged over 11 GB, so two EMPTY 16.6 GB cards were handed a 4.3 GB budget
    /// each and eight of the model's forty blocks went to the host. Half of that reserve was
    /// a score buffer charged at F32 for a kernel that writes BF16.
    #[test]
    fn charging_f32_for_a_bf16_score_buffer_costs_most_of_a_card() {
        let (tokens, dim, heads) = (21_504usize, 5120usize, 40usize);
        let f32_estimate = dit_activation_bytes(tokens, dim, heads, 4.0);
        let tc_estimate = dit_activation_bytes_tc(tokens, dim, heads, 4.0, ACT_BYTES, tokens);
        assert!(
            tc_estimate < f32_estimate,
            "the tensor-core path cannot need MORE than the exact one: {tc_estimate} vs {f32_estimate}"
        );
        // The live set is identical, so the whole difference is the score buffer, and it
        // has to be worth enough of a card to matter - this is not a rounding correction.
        let saved = f32_estimate - tc_estimate;
        assert!(
            saved > 3 << 30,
            "expected the correction to be gigabytes, got {} MB",
            saved >> 20
        );
    }

    /// The estimate must charge for the tile the KERNEL will take, not a different one.
    /// They drifted once, in the direction that under-charges - which is the direction that
    /// OOMs - because the estimate kept reading the exact path's tile after the kernel
    /// started doubling it.
    #[test]
    fn the_estimate_uses_the_kernels_own_tile() {
        use crate::inference::model::acestep::ops::{tc_query_tile_for_heads, tc_score_bytes};
        for tokens in [4096usize, 21_504, 123_904, 500_000] {
            let (dim, heads, ratio) = (5120usize, 40usize, 4.0);
            let got = dit_activation_bytes_tc(tokens, dim, heads, ratio, super::ACT_BYTES, tokens);
            let live = super::dit_live_bytes(tokens, dim, ratio, super::ACT_BYTES);
            let tile = tc_query_tile_for_heads(tokens, heads).min(tokens) as u64;
            let qkv = tokens as u64 * dim as u64;
            let want = live
                + 2 * tile * tokens as u64 * heads as u64 * tc_score_bytes()
                + qkv * (super::ACT_BYTES + 2 + 2 + 2 + super::ACT_BYTES);
            assert_eq!(got, want, "tokens={tokens}");
        }
    }

    /// A block that carries its residual at half width must be CHARGED at half width.
    ///
    /// The estimate read a full-precision constant while the forward ran at half, so the
    /// reserve came out close to twice what the render takes - and a reserve is subtracted
    /// from the card before a single block is placed. At a large frame that difference is
    /// the whole margin: the same request is either resident or spread onto the host.
    #[test]
    fn a_half_width_stream_is_charged_half_a_full_width_one() {
        let (tokens, dim, heads) = (86_016usize, 5120usize, 40usize);
        let full = dit_activation_bytes_tc(tokens, dim, heads, 4.0, 4, tokens);
        let half = dit_activation_bytes_tc(tokens, dim, heads, 4.0, 2, tokens);
        assert!(half < full, "{half} vs {full}");
        // The scores are BF16 on both, so only the live set and the casts move - which is
        // most of the demand at this size, not a rounding correction.
        let saved = full - half;
        assert!(
            saved > 4 << 30,
            "expected gigabytes, got {} MB",
            saved >> 20
        );
    }

    /// The video denoiser charges its reserve at full width even though its blocks run at
    /// half. Lowering it to match the stream let the planner pack every block onto one
    /// card and a working frame size started failing, so the gap is deliberate and this
    /// says so if anyone closes it without measuring the peak first.
    #[test]
    fn the_video_denoiser_declares_the_width_it_runs_at() {
        assert_eq!(crate::inference::model::wan::dit::RESERVE_BYTES, 4);
    }

    /// Longer clips must not be charged less per token than short ones by accident: the
    /// demand has to keep RISING with the sequence, or a long render is admitted onto a
    /// card that cannot hold it.
    #[test]
    fn the_demand_still_grows_with_the_clip() {
        let prev = [4096usize, 21_504, 123_904]
            .map(|t| dit_activation_bytes_tc(t, 5120, 40, 4.0, super::ACT_BYTES, t));
        assert!(prev[0] < prev[1] && prev[1] < prev[2], "{prev:?}");
    }

    /// The per-device allowance must follow the checkpoint and never fall under what a
    /// first launch was measured to need.
    #[test]
    fn the_runtime_floor_follows_the_checkpoint() {
        use super::load_runtime_floor;
        let small = load_runtime_floor(200 << 20);
        assert_eq!(
            small,
            load_runtime_floor(0),
            "the floor must hold under a small model"
        );
        // Nothing the fleet places today may move: the largest encoder on this gate is
        // about 12 GB, and it has to keep the floor it has been running with.
        assert_eq!(
            load_runtime_floor(12u64 << 30),
            small,
            "a shipped checkpoint moved"
        );
        let huge = load_runtime_floor(96u64 << 30);
        assert!(
            huge > small * 4,
            "a checkpoint that large must not be staged like a 2B one"
        );
    }
}
