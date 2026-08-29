//! What one audio render actually needs, DERIVED from the request and the
//! architecture - never a constant.
//!
//! WHY THIS MODULE EXISTS. The ACE-Step / EzAudio placement path reserved fixed byte
//! counts: 512 MB under every encoder, 1536 MB under the denoiser and the LM, 2 GB under
//! the Oobleck decoder, and an invented checkpoint size whenever a file could not be
//! stat'd. A fixed figure can only be right at one clip length on one checkpoint. It is
//! simultaneously too large on a ten-second sketch - refusing cards that would have held
//! the render - and too small on a four-minute track, where it admits a load that then
//! cannot denoise. The second failure is the expensive one: the weights go resident, and
//! every render afterwards dies for want of scratch on a card that is now full.
//!
//! WHAT REPLACES THEM. The same three-part answer the image path already uses
//! ([`crate::inference::place::runtime_demand`], which this module reuses rather than
//! reimplements):
//!
//!  - an ANALYTIC term, computed from what the stage actually holds live - tokens, model
//!    width, head count, MLP ratio for a transformer; channels times upsampled length for
//!    a convolutional decoder;
//!  - a MEASURED seed, which is the constant being replaced, kept with the geometry it was
//!    validated at so it can be scaled instead of guessed at again;
//!  - GROWTH, not replacement: the reserve is the seed plus whatever the analytic term says
//!    THIS request adds over the reference one.
//!
//! WHY GROWTH RATHER THAN SUBSTITUTION. A reserve is not a description of what a block
//! allocates - it is what stops the planner from putting one more block on the card.
//! Lowering it does not free memory, it packs a bigger model onto the same hardware. The
//! analytic peak of these stages lands well BELOW the constants they replace (a 24-second
//! ACE-Step denoise holds about 130 MB live against a 1536 MB reserve), so swapping the
//! constant for the formula would have quadrupled what each card is asked to hold. The
//! gap is real - it is allocator churn, cuBLAS workspaces, dequantisation staging and the
//! concurrent-consumer headroom the comment on the constant asked for - and none of that
//! is in the analytic peak. So the constants stay as the floor they have earned, and the
//! formula supplies only the part that moves with the request.

use crate::inference::place::runtime_demand::{
    dit_activation_bytes, scale_measured, MeasuredReserve,
};

/// Bytes per element of the accumulation dtype. The audio stages run their activations in
/// f32 end to end, including the stages whose WEIGHTS are Q8: a quantised weight is
/// dequantised into an f32 activation, so the scratch does not follow the container.
const ACT_BYTES: u64 = 4;

// -- the request, in the units each stage counts in ---------------------------

/// Latent frames a clip of `seconds` becomes.
///
/// `vae_stride` is the autoencoder's total temporal reduction - the product of its block
/// strides, 1920 for the 48 kHz ACE-Step decoder and 480 for the 24 kHz EzAudio one. Read
/// from the model rather than assumed: the two differ by 4x, and charging one family the
/// other's figure is the whole defect this module exists for.
pub fn latent_frames(seconds: f64, sample_rate: usize, vae_stride: usize) -> usize {
    let samples = (seconds.max(0.0) * sample_rate as f64) as usize;
    (samples / vae_stride.max(1)).max(1)
}

/// Tokens the denoiser attends over for `frames` latent frames.
///
/// The DiT patchifies along time, so a patch size of 2 halves the sequence before a single
/// block runs. Getting this wrong scales the attention term by its square.
pub fn dit_tokens(frames: usize, patch: usize) -> usize {
    (frames / patch.max(1)).max(1)
}

// -- analytic terms -----------------------------------------------------------

/// Activation peak of one transformer stage over `tokens`, in bytes.
///
/// Delegates to the image path's denoiser term: an audio DiT block and an image DiT block
/// hold the same things live - the residual stream, the fused projection, the attention
/// output, the chunked MLP intermediate and the tiled score slab - so there is one
/// derivation, not two that can drift apart.
pub fn stage_activation_bytes(tokens: usize, dim: usize, heads: usize, ffn: usize) -> u64 {
    let mlp_ratio = ffn.max(1) as f64 / dim.max(1) as f64;
    dit_activation_bytes(tokens, dim, heads, mlp_ratio)
}

/// Peak bytes one Oobleck decode WINDOW takes, from the decoder's own level widths.
///
/// `levels` is `(output channels, cumulative upsample)` per block, in order. The peak is
/// the WORST level rather than their sum: the blocks run one after another, each freeing
/// the previous map. Channels halve as the length multiplies by the block's stride, so the
/// widest level is not the first or the last in general - it is whichever one the
/// arithmetic picks, which is exactly why this is computed rather than tabulated.
///
/// The dominant tensor at a level is not the feature map but the convolution's im2col
/// expansion, which is `kernel` times taller. That is what makes an audio decode cost
/// gigabytes where its feature maps cost hundreds of megabytes.
pub fn oobleck_window_bytes(window_frames: usize, levels: &[(usize, usize)], kernel: usize) -> u64 {
    /// The map itself, the block's residual input, and the convolution's output.
    const LIVE: u64 = 3;
    let window = window_frames.max(1) as u64;
    let cols = kernel.max(1) as u64 + LIVE;
    levels
        .iter()
        .map(|(ch, up)| cols * (*ch).max(1) as u64 * window * (*up).max(1) as u64 * ACT_BYTES)
        .max()
        .unwrap_or(0)
}

/// Transients one captured decode STEP of an autoregressive stage holds at once.
///
/// The graph arena never frees mid-capture, so it has to hold every layer's transients
/// simultaneously - which is what makes this a function of the layer COUNT and not just of
/// the width. `batch` is the rows the step carries (2 under classifier-free guidance).
pub fn decode_step_transient_bytes(layers: usize, dim: usize, ffn: usize, batch: usize) -> u64 {
    /// Token-major tensors a decoder layer holds live: the residual stream, the fused
    /// projection, the attention output, and the gate/up pair over the MLP intermediate.
    const LIVE_PER_LAYER: u64 = 4;
    let per_layer = LIVE_PER_LAYER * dim.max(1) as u64 + 2 * ffn.max(1) as u64;
    layers.max(1) as u64 * batch.max(1) as u64 * per_layer * ACT_BYTES
}

// -- turning a measurement into a reserve -------------------------------------

/// A measured reserve, GROWN by what this request adds over the one it was measured at.
///
/// Never below the seed, by construction. That asymmetry is the point and it was learned
/// the hard way twice in this repository: an over-stated reserve costs blocks pushed onto
/// the host, which is slow; an under-stated one costs a render that cannot allocate on a
/// card it was already placed on, which is a failure. A derivation that comes out under
/// the figure a working system was validated at is evidence that the derivation is missing
/// a term, not evidence that the figure was too big.
pub fn grown(seed: &MeasuredReserve, here: u64, at_reference: u64) -> u64 {
    seed.bytes.saturating_add(here.saturating_sub(at_reference))
}

// -- the reference geometries the seeds were validated at ---------------------
//
// These are ARCHITECTURE and REQUEST numbers - a clip length, a model width, a head count.
// They are here so that every seed below says what it was true of, which is the one thing
// the constants it replaces could not say.

/// The default ACE-Step render: 120 codes, five latent frames each.
pub const ACE_REFERENCE_FRAMES: usize = 600;
/// The turbo denoiser: hidden 2048, 16 heads, feed-forward 6144, patch 2.
pub const ACE_DIT_REFERENCE: (usize, usize, usize, usize) = (2048, 16, 6144, 2);
/// The ACE-Step LM: 36 layers, hidden 2560, feed-forward 9728, two CFG rows per step.
pub const ACE_LM_REFERENCE: (usize, usize, usize, usize) = (36, 2560, 9728, 2);
/// The 48 kHz Oobleck decoder's levels, `(channels, cumulative upsample)`, for the
/// strides `[10, 6, 4, 4, 2]` its loader reads, and its 7-tap convolutions.
pub const ACE_VAE_REFERENCE_LEVELS: [(usize, usize); 5] =
    [(768, 10), (384, 60), (192, 240), (96, 960), (48, 1920)];
/// Latent frames the render hands the decoder at a time. The decode is tiled, which is
/// what stops its peak from following the clip: a four-minute track is decoded in the same
/// window as a ten-second one, so the reserve must follow the WINDOW and not the request.
pub const DECODE_CHUNK_FRAMES: usize = 192;
/// Frames of context carried into each decode window on both sides, so the seams reproduce
/// the untiled decode. They are resident alongside the chunk, so they are charged with it.
pub const DECODE_OVERLAP_FRAMES: usize = 64;
/// The decode window: the chunk the render walks the latent in, plus both overlaps.
pub const ACE_VAE_REFERENCE_WINDOW: usize = DECODE_CHUNK_FRAMES + 2 * DECODE_OVERLAP_FRAMES;
/// Tap count of the Oobleck convolutions, which sets the im2col height.
pub const OOBLECK_KERNEL: usize = 7;
/// The decode-step sequence the captured graph is built for.
pub const ACE_LM_REFERENCE_BUCKET: usize = 256;

// -- the seeds ----------------------------------------------------------------

/// What the ACE-Step denoiser was found to need beyond its weights, at the default clip.
///
/// Reproduces the reserve the loader carried before this module: it is the same figure,
/// now with the geometry it is true of attached, so a longer clip is charged for the
/// scratch it really adds instead of running on a card sized for a shorter one.
pub const ACE_DIT_SEED: MeasuredReserve = MeasuredReserve {
    bytes: 1536 << 20,
    reference_tokens: ACE_REFERENCE_FRAMES / 2,
};

/// What the ACE-Step LM was found to need beyond its weights and its KV cache.
///
/// The KV is charged separately by the planner, so this covers the capture arena, the
/// cuBLAS workspace and the transient scratch of a decode step. It is keyed on the step's
/// transient footprint rather than on a clip length: an autoregressive decode does one
/// token at a time however long the track is, so the request that moves this number is a
/// WIDER model, not a longer song.
pub const ACE_LM_SEED: MeasuredReserve = MeasuredReserve {
    bytes: 1536 << 20,
    reference_tokens: 1,
};

/// What the Oobleck decode was found to need beyond its weights, at the reference window.
pub const ACE_VAE_SEED: MeasuredReserve = MeasuredReserve {
    bytes: 2 << 30,
    reference_tokens: ACE_VAE_REFERENCE_WINDOW,
};

/// The capture arena the ACE-Step LM's decode graph was validated with.
///
/// An arena, unlike the seeds above, is an ALLOCATION rather than a reserve - it is memory
/// actually taken, and the capture fails if it is short. It still cannot be a fixed number:
/// it has to hold every layer's transients at once, so a wider or deeper checkpoint needs
/// proportionally more, and the family already ships two.
pub const ACE_LM_ARENA_SEED: MeasuredReserve = MeasuredReserve {
    bytes: 2048 << 20,
    reference_tokens: 1,
};

/// The cuBLAS workspace pinned once for the captured decode, so an in-capture allocation
/// cannot invalidate the graph.
///
/// It backs the attention-score GEMMs of one captured step, so it follows the sequence the
/// graph is captured at: a wider bucket means wider score matrices and more split-k
/// staging behind them.
pub const ACE_LM_CUBLAS_SEED: MeasuredReserve = MeasuredReserve {
    bytes: 32 << 20,
    reference_tokens: ACE_LM_REFERENCE_BUCKET,
};

/// What one degradation notch adds to every card's reserve after an out-of-memory.
///
/// A retry that re-plans against unchanged numbers produces the same plan and fails the
/// same way, so a notch has to be COARSE enough to actually move a chunk of layers to the
/// next device rather than re-packing the card that just failed. It is deliberately not
/// derived from the model: what has to move is a segment of the placement, and the
/// escalation is the only lever the retry has.
pub const DEGRADE_NOTCH_SEED: MeasuredReserve = MeasuredReserve {
    bytes: 4 << 30,
    reference_tokens: 1,
};

// -- what each placement site asks for ----------------------------------------

/// Reserve for the ACE-Step denoiser rendering `frames` latent frames.
pub fn dit_reserve(frames: usize, dim: usize, heads: usize, ffn: usize, patch: usize) -> u64 {
    let (ref_dim, ref_heads, ref_ffn, ref_patch) = ACE_DIT_REFERENCE;
    let here = stage_activation_bytes(dit_tokens(frames, patch), dim, heads, ffn);
    let at_reference = stage_activation_bytes(
        dit_tokens(ACE_REFERENCE_FRAMES, ref_patch),
        ref_dim,
        ref_heads,
        ref_ffn,
    );
    grown(&ACE_DIT_SEED, here, at_reference)
}

/// Reserve for the ACE-Step LM, from the shape of the decode step it will capture.
pub fn lm_reserve(layers: usize, dim: usize, ffn: usize, batch: usize) -> u64 {
    let (ref_layers, ref_dim, ref_ffn, ref_batch) = ACE_LM_REFERENCE;
    let here = decode_step_transient_bytes(layers, dim, ffn, batch);
    let at_reference = decode_step_transient_bytes(ref_layers, ref_dim, ref_ffn, ref_batch);
    grown(&ACE_LM_SEED, here, at_reference)
}

/// Reserve for an Oobleck decode of this window on a decoder with these levels.
pub fn oobleck_reserve(window_frames: usize, levels: &[(usize, usize)], kernel: usize) -> u64 {
    let here = oobleck_window_bytes(window_frames, levels, kernel);
    let at_reference = oobleck_window_bytes(
        ACE_VAE_REFERENCE_WINDOW,
        &ACE_VAE_REFERENCE_LEVELS,
        OOBLECK_KERNEL,
    );
    grown(&ACE_VAE_SEED, here, at_reference)
}

/// Capture-arena bytes for a decode step of this shape.
pub fn lm_graph_arena_bytes(layers: usize, dim: usize, ffn: usize, batch: usize) -> u64 {
    let (ref_layers, ref_dim, ref_ffn, ref_batch) = ACE_LM_REFERENCE;
    let here = decode_step_transient_bytes(layers, dim, ffn, batch);
    let at_reference = decode_step_transient_bytes(ref_layers, ref_dim, ref_ffn, ref_batch);
    // Proportional, not additive: the arena IS the transients, so a step that holds twice
    // as much needs twice the arena. The seeds above are additive instead because most of
    // what they cover - allocator churn, a concurrent consumer - does not scale with the
    // model at all.
    let scaled = ACE_LM_ARENA_SEED.bytes as f64 * here as f64 / at_reference.max(1) as f64;
    (scaled as u64).max(ACE_LM_ARENA_SEED.bytes / ARENA_FLOOR_SHARE)
}

/// cuBLAS workspace for a decode graph captured at this bucket.
pub fn lm_cublas_workspace_bytes(bucket: usize) -> u64 {
    scale_measured(&ACE_LM_CUBLAS_SEED, bucket)
}

/// Extra reserve `level` degradation notches add to every placement probe.
pub fn degrade_reserve_bytes(level: u64) -> u64 {
    level.saturating_mul(DEGRADE_NOTCH_SEED.bytes)
}

/// Smallest useful share of the arena. The capture retries at half the arena when the
/// allocation loses a race with a concurrent consumer; below this share the arena no
/// longer holds a whole step and the eager path is the honest answer. A fraction rather
/// than a byte count, so it follows whatever the arena above works out to.
pub const ARENA_FLOOR_SHARE: u64 = 16;

/// Share of the arena still worth attempting when half of free VRAM is under it. Above
/// [`ARENA_FLOOR_SHARE`] so a first attempt that has to shrink still has somewhere to go.
pub const ARENA_HEADROOM_SHARE: u64 = 8;

#[cfg(test)]
mod tests {
    use super::*;

    /// Every reserve must reproduce, at its own reference geometry, the constant it
    /// replaced. This is the check that makes the change safe to ship: a formula that
    /// lands under the figure a working system was validated at does not free memory, it
    /// packs one more block onto the card the render then has to run on.
    #[test]
    fn each_reserve_reproduces_the_constant_it_replaced() {
        let (dim, heads, ffn, patch) = ACE_DIT_REFERENCE;
        assert_eq!(
            dit_reserve(ACE_REFERENCE_FRAMES, dim, heads, ffn, patch),
            ACE_DIT_SEED.bytes,
            "the denoiser reserve drifted from the figure it was validated at"
        );
        let (layers, ldim, lffn, batch) = ACE_LM_REFERENCE;
        assert_eq!(lm_reserve(layers, ldim, lffn, batch), ACE_LM_SEED.bytes);
        assert_eq!(
            oobleck_reserve(
                ACE_VAE_REFERENCE_WINDOW,
                &ACE_VAE_REFERENCE_LEVELS,
                OOBLECK_KERNEL
            ),
            ACE_VAE_SEED.bytes
        );
        assert_eq!(
            lm_graph_arena_bytes(layers, ldim, lffn, batch),
            ACE_LM_ARENA_SEED.bytes
        );
        assert_eq!(
            lm_cublas_workspace_bytes(ACE_LM_REFERENCE_BUCKET),
            ACE_LM_CUBLAS_SEED.bytes
        );
        assert_eq!(degrade_reserve_bytes(1), DEGRADE_NOTCH_SEED.bytes);
        assert_eq!(
            degrade_reserve_bytes(0),
            0,
            "an undegraded placement adds nothing"
        );
    }

    /// The captured graph's workspace must follow the bucket it is captured at: a wider
    /// bucket means wider score matrices, and a workspace short of them makes cuBLAS
    /// allocate mid-capture, which invalidates the graph.
    #[test]
    fn the_capture_workspace_follows_the_bucket() {
        let base = lm_cublas_workspace_bytes(ACE_LM_REFERENCE_BUCKET);
        assert_eq!(
            lm_cublas_workspace_bytes(ACE_LM_REFERENCE_BUCKET * 4),
            base * 4
        );
        assert!(
            lm_cublas_workspace_bytes(0) > 0,
            "a degenerate bucket must still allocate"
        );
    }

    /// A longer clip must cost more. This is the property the constants failed: one
    /// number admitted a load for a twenty-second sketch and then could not denoise the
    /// four-minute track the same model was asked for.
    #[test]
    fn a_longer_clip_costs_more() {
        let (dim, heads, ffn, patch) = ACE_DIT_REFERENCE;
        let short = dit_reserve(ACE_REFERENCE_FRAMES, dim, heads, ffn, patch);
        let long = dit_reserve(ACE_REFERENCE_FRAMES * 10, dim, heads, ffn, patch);
        assert!(long > short, "{long} vs {short}");
        // And the growth has to be worth reserving for, not a rounding correction: a ten
        // times longer track is the case that was OOMing.
        assert!(
            long - short > 256 << 20,
            "growth of only {} MB",
            (long - short) >> 20
        );
    }

    /// A longer clip must never cost LESS, at any length, including the degenerate ones.
    #[test]
    fn the_denoiser_reserve_never_falls() {
        let (dim, heads, ffn, patch) = ACE_DIT_REFERENCE;
        let mut prev = 0u64;
        for frames in [0usize, 1, 10, 100, 600, 6_000, 60_000] {
            let r = dit_reserve(frames, dim, heads, ffn, patch);
            assert!(r >= prev, "frames {frames}: {r} < {prev}");
            assert!(
                r >= ACE_DIT_SEED.bytes,
                "frames {frames}: below the validated floor"
            );
            prev = r;
        }
    }

    /// A wider or deeper checkpoint must be charged more, at the same clip length: the
    /// family already ships a 2B turbo and a 4B XL, and one reserve cannot be right for
    /// both.
    #[test]
    fn a_wider_checkpoint_costs_more() {
        let (dim, heads, ffn, patch) = ACE_DIT_REFERENCE;
        let turbo = dit_reserve(ACE_REFERENCE_FRAMES * 4, dim, heads, ffn, patch);
        let xl = dit_reserve(ACE_REFERENCE_FRAMES * 4, 2560, 32, 8960, patch);
        assert!(
            xl > turbo,
            "the 4B XL must not be charged the 2B turbo's reserve: {xl} vs {turbo}"
        );
        let (layers, ldim, lffn, batch) = ACE_LM_REFERENCE;
        assert!(lm_reserve(layers * 2, ldim, lffn, batch) > lm_reserve(layers, ldim, lffn, batch));
        assert!(
            lm_graph_arena_bytes(layers * 2, ldim, lffn, batch)
                > lm_graph_arena_bytes(layers, ldim, lffn, batch)
        );
    }

    /// The decode reserve must follow the DECODER, not the family that was measured.
    ///
    /// The 24 kHz decoder upsamples by 480 where the 48 kHz one upsamples by 1920, so
    /// charging both the same figure is the defect in miniature. It must not be charged
    /// MORE than the wider decoder, and a window twice as long must cost more.
    #[test]
    fn the_decode_reserve_follows_the_decoder_and_the_window() {
        let narrow: [(usize, usize); 4] = [(512, 10), (256, 60), (128, 240), (64, 480)];
        let ace = oobleck_reserve(
            ACE_VAE_REFERENCE_WINDOW,
            &ACE_VAE_REFERENCE_LEVELS,
            OOBLECK_KERNEL,
        );
        let ez = oobleck_reserve(ACE_VAE_REFERENCE_WINDOW, &narrow, OOBLECK_KERNEL);
        assert!(
            ez <= ace,
            "the 24 kHz decoder must not be charged more than the 48 kHz one"
        );
        let wide_window = oobleck_reserve(
            ACE_VAE_REFERENCE_WINDOW * 4,
            &ACE_VAE_REFERENCE_LEVELS,
            OOBLECK_KERNEL,
        );
        assert!(
            wide_window > ace,
            "a longer decode window must cost more: {wide_window} vs {ace}"
        );
    }

    /// No stage may ask for nothing. A zero reserve reads to the planner as "needs no
    /// scratch", which is how a model gets placed on a card with nothing left to run in.
    #[test]
    fn no_shape_ever_asks_for_nothing() {
        assert!(dit_reserve(0, 0, 0, 0, 0) > 0);
        assert!(lm_reserve(0, 0, 0, 0) > 0);
        assert!(oobleck_reserve(0, &[], 0) > 0);
        assert!(lm_graph_arena_bytes(0, 0, 0, 0) > 0);
        assert!(latent_frames(0.0, 0, 0) >= 1);
        assert!(dit_tokens(0, 0) >= 1);
    }

    /// The request geometry must be read in the units each family actually uses.
    #[test]
    fn a_clip_becomes_the_frames_its_own_autoencoder_makes_of_it() {
        // 48 kHz, stride 1920 -> 25 latent frames a second; 24 kHz, stride 480 -> 50.
        assert_eq!(latent_frames(24.0, 48_000, 1920), 600);
        assert_eq!(latent_frames(10.0, 24_000, 480), 500);
        assert_eq!(dit_tokens(600, 2), 300);
    }
}
