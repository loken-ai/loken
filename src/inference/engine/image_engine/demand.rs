//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// What a render cannot go UNDER, as against what it would prefer.
///
/// The comfortable figure carries room for the allocator's churn and for a concurrent
/// consumer; the floor is the live set the denoise actually holds. Between the two a
/// render is tighter than one would choose and still completes - which is why the
/// distinction exists, and why collapsing it made a model that fits one card split
/// across two.
///
/// Three quarters: a ratio, so it holds at any resolution and for any family.
pub fn runtime_floor(comfortable: u64) -> u64 {
    comfortable * 3 / 4
}

/// What a Z-Image generation of `width x height` needs FREE, beyond the resident
/// weights.
///
/// THE figure for this family, and the only one. The admission path used to derive its
/// own from the activation shapes - about 1.9 GB where this returns 3.2 GB at 1024^2 -
/// so the gate that decided whether a render could start and the planner that decided
/// where its blocks went were answering the same question with two numbers, and neither
/// could learn from the other. Worse, only this one is wired to the measurement loop:
/// the moment a render is observed to need more, this rises and the other did not, so a
/// shape that had just exhausted a card was still admitted against the figure that
/// admitted the failure.
///
/// The denoise and the decode are SEQUENTIAL - the denoiser's scratch is freed before
/// the decoder allocates - so the family needs the LARGER of the two, never their sum.
///
/// THE FALLBACK, not the answer. This scales one recorded figure LINEARLY in tokens, and
/// the forward it is standing in for does not grow that way: between 512 and 1024 square
/// the token count goes up by 3.40 and the walked forward by 3.38, so no seed value
/// reproduces both points - the shape of the formula is wrong, not its constant.
/// [`zimage_runtime_demand_from_files`] measures the forward instead and only comes back
/// here when there is nothing to measure.
pub fn zimage_runtime_demand(width: usize, height: usize) -> u64 {
    use crate::inference::place::runtime_demand::{latent_tokens, MeasuredReserve};
    let cfg = crate::inference::model::zimage::dit::Config::z_image_turbo();
    let patch = cfg.all_patch_size.first().copied().unwrap_or(1);
    // 3 GiB, sampled through a 1024^2 render of this family: the transformer's card
    // peaks well under it, and the figure placed the model whole. Scaled to the
    // request, superseded by measurement.
    let reference =
        (1024 / (FLUX_VAE_STRIDE * patch)) * (1024 / (FLUX_VAE_STRIDE * patch)) + FLUX_TEXT_TOKENS;
    let measured = MeasuredReserve {
        bytes: 3 << 30,
        reference_tokens: reference,
    };
    let tokens = latent_tokens(height, width, FLUX_VAE_STRIDE, patch) + FLUX_TEXT_TOKENS;
    let decode = crate::inference::place::runtime_demand::vae_decode_bytes(
        height,
        width,
        VAE_WIDEST_CH as usize,
    );
    let seed =
        crate::inference::place::runtime_demand::scale_measured(&measured, tokens).max(decode);
    crate::inference::place::runtime_demand::planning_demand("zimage", width, height, seed)
}

/// What the DRY RUN's figure is multiplied by before it becomes a reserve, in percent.
///
/// DERIVED FROM THE ERROR THAT WAS MEASURED, which is the first time this figure has
/// rested on anything but a round number. Four geometries, each walked and then measured
/// on the card with the denoise loop bracketed directly:
///
/// ```text
///     walked   measured   ratio
///      0.40      0.44     1.10
///      0.79      0.87     1.10
///      1.35      1.61     1.19
///      2.93      2.65     0.90   (per card, split across two)
/// ```
///
/// Two of those four - the smallest and the one between - were rendered AFTER the ratio
/// was established on the other two, so they are out of sample and they agree. The worst
/// under-estimate is 19 percent; 135 leaves 13 percent over it.
///
/// WHY THERE IS NO ADDITIVE TERM ANY MORE. There was one: renders of the same shape used
/// to land 0.31 GB apart, which no multiple of a walked figure can cover at a small
/// geometry. That variance was a property of an over-wide attention tile - more slices in
/// flight, more churn in the pool - and it went with it: four renders of the same shape
/// now read the same figure to the hundredth of a gigabyte. So the error is purely
/// multiplicative and the margin can be a multiple again.
///
/// WHY IT IS SAFE TO GO BELOW THE OLD 200. That figure was not a margin: it was covering
/// a 2.2x disagreement between the implementation the walk traversed and the one that
/// ran. The two are the same code now, and what is left over the walk is its own error,
/// which is the table above. The margin covers what was measured and nothing more.
///
/// AND IT ONLY WORKS BESIDE AN HONEST WEIGHTS FIGURE. The card holds a few percent more
/// for the weights than they count, and 200 was quietly covering that too - so quietly
/// that at 512 square it had already stopped: the total charged there sat 0.11 GB UNDER
/// what the card takes, on the smallest geometry, where nobody would look. The weights
/// are now charged at what the allocator holds for them, and the two changes belong
/// together: either alone moves the total the wrong way.
///
/// WHAT IT DOES NOT COVER. One checkpoint, one driver, one pair of cards. A family whose
/// walk diverges from its forward the way this one's did before it was unified would
/// need its own table before reading this number.
pub const ZIMAGE_DRY_MARGIN_PERCENT: u64 = 135;

/// The reserve a Z-Image render of `width x height` gets from a MEASURED forward.
///
/// Split out from the measurement so the arithmetic can be checked without a checkpoint
/// on disk: the recorded dry figures go in, the reserves come out, and a test can hold
/// them against the render peaks the same session recorded.
///
/// The decode term and the observed-peak store stay exactly where they were. The dry run
/// walks the TRANSFORMER, so it says nothing about the VAE that runs after it, and a
/// shape that has already exhausted a card is still worth more than any derivation.
pub fn zruntime_demand_from_dry(dry_forward: u64, width: usize, height: usize) -> u64 {
    let decode = crate::inference::place::runtime_demand::vae_decode_bytes(
        height,
        width,
        VAE_WIDEST_CH as usize,
    );
    let seed = dry_forward.saturating_mul(ZIMAGE_DRY_MARGIN_PERCENT) / 100;
    crate::inference::place::runtime_demand::planning_demand(
        "zimage",
        width,
        height,
        seed.max(decode),
    )
}

/// What each card of a MEASURED placement must have free, from what a forward of that
/// placement was counted to hold on it.
///
/// The counts go in per device and come out as reserves per device, every one of them
/// through [`zruntime_demand_from_dry`] - the same rule a card carrying the whole model
/// goes through, with only the forward it is fed changing. That is what makes this safe
/// to decide with: a placement of ONE segment counts one card carrying everything, so
/// the figure it comes out with is the figure this loader has always planned against,
/// value for value. What the placement changes is the answer for the OTHER cards, which
/// no single reserve could ever have expressed - a card given nine blocks of thirty was
/// being charged the peak of a card given all thirty.
pub(crate) fn zimage_placed_reserve(
    counted: &crate::inference::place::dry_plan::PlanLoad,
    width: usize,
    height: usize,
) -> crate::inference::place::dry_plan::PlanLoad {
    crate::inference::place::dry_plan::PlanLoad {
        devices: counted
            .devices
            .iter()
            .map(|d| crate::inference::place::dry_plan::DeviceLoad {
                kind: d.kind,
                weights: d.weights,
                forward: zruntime_demand_from_dry(d.forward, width, height),
            })
            .collect(),
    }
}

/// What a measured placement already claims on one card: its blocks and its reserve.
///
/// Zero for a card the placement does not use - which is the whole point of asking, when
/// what is being decided is where something ELSE may sit.
pub(super) fn zimage_claim_on(
    load: &crate::inference::place::dry_plan::PlanLoad,
    idx: usize,
) -> u64 {
    load.on(crate::inference::place::layer_executor::DeviceKind::Cuda(
        idx,
    ))
    .map(|d| d.peak())
    .unwrap_or(0)
}

/// Which card the caption encoder may sit on, given what the transformer's placement
/// already claims. `None` means the host.
///
/// A FUNCTION, because it decides gigabytes and it used to decide them from a question
/// that has no answer when the model is split. The old rule was "the roomiest card the
/// transformer does not take WHOLE" - and the moment no card takes it whole, `tf_gpu` is
/// `None`, nothing is excluded, and the encoder helps itself to the roomiest card. That
/// is exactly the card a split needs: six and a half gigabytes off one of two 16.4 GB
/// cards left the second segment eighty megabytes short of the blocks it was given, on a
/// machine with room for the model twice over.
///
/// With a measured placement in hand the question has an answer per card - free VRAM less
/// what the plan holds there, blocks and forward - and the encoder is offered what is
/// left. When nothing is left it encodes on the host: about six seconds once per request,
/// against a render that otherwise does not happen. The hot component gets first claim,
/// which is the fleet rule; this is only the first time it could be applied honestly.
///
/// Without a measurement the old rule stands, unchanged, so a checkpoint the walk cannot
/// read places the encoder exactly where it placed it before.
pub(crate) fn zimage_encoder_card(
    cuda_free: &[(usize, u64)],
    placed: Option<&crate::inference::place::dry_plan::PlanLoad>,
    transformer_whole_on: Option<usize>,
    want_bytes: u64,
) -> Option<usize> {
    // One card is the transformer's, and the encoder does not compete with it there:
    // that is the no-OOM guarantee tight boxes have always had.
    if cuda_free.len() < 2 {
        return None;
    }
    // PLACEMENT-EXEMPT: this is the ONE-SHOT encoder, offered what the hot component's
    // plan leaves behind - the transformer has already taken its cards above, and what is
    // ranked here is the remainder, not the fleet. Ranking by throughput instead would put
    // the encoder on the transformer's own card, which is the 6.5 GB that left a segment
    // eighty megabytes short of its blocks.
    match placed {
        // A CARD THE PLAN RUNS IS NOT A CARD WITH ROOM. Subtracting what the plan holds
        // there and offering the difference reads as arithmetic, and it is - but the
        // difference is a remainder, and a remainder is where the error of every term
        // above it lands. At 1536 square it came to 0.21 GB against a 6.5 GB tenant that
        // the sum said would fit: twelve blocks at 5.74, the encoder at 6.50 and a 3.95
        // reserve make 16.19 of 16.4, and the render died in the first attention GEMM.
        // The same geometry varies by 0.48 GB between identical runs, so that arrangement
        // was never inside its own noise.
        //
        // So the encoder is offered cards the plan does not touch AT ALL. What runs once
        // per request does not share with what runs every step of every image - which is
        // what the paragraph above already says about ranking, applied to the question of
        // room as well. When the plan takes every card, that is not a failure: the encoder
        // goes to the host for about six seconds, and the render happens.
        Some(load) => cuda_free
            .iter()
            .filter(|(idx, _)| zimage_claim_on(load, *idx) == 0)
            .filter(|(_, free)| *free >= want_bytes)
            .max_by_key(|(_, free)| *free)
            .map(|(idx, _)| *idx),
        None => cuda_free
            .iter()
            .filter(|(idx, free)| Some(*idx) != transformer_whole_on && *free >= want_bytes)
            .max_by_key(|(_, free)| *free)
            .map(|(idx, _)| *idx),
    }
}

/// One forward of this checkpoint at this geometry, counted rather than estimated.
///
/// MEMOISED on the checkpoint and the geometry, because the answer depends on nothing
/// else: the same files at the same shape count the same tensors. The walk costs a
/// safetensors header parse and a pass over the shape algebra - about a tenth of a
/// second, no card, no file data - which is cheap once and not cheap on the admission
/// path of every request. `None` is cached too: a checkpoint this cannot walk will not
/// become walkable by being asked again inside the same process.
pub(super) fn zimage_dry_forward_bytes(
    files: &[&str],
    prefix: Option<&str>,
    width: usize,
    height: usize,
) -> Option<u64> {
    static MEASURED: std::sync::LazyLock<
        std::sync::Mutex<std::collections::HashMap<(String, usize, usize), Option<u64>>>,
    > = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

    let key = (
        format!("{}\u{1}{}", prefix.unwrap_or(""), files.join("\u{1}")),
        width,
        height,
    );
    if let Some(hit) = MEASURED.lock().ok().and_then(|m| m.get(&key).copied()) {
        return hit;
    }
    let mut cfg = crate::inference::model::zimage::dit::Config::z_image_turbo();
    // What the loader builds. The attention path decides what the score matrix costs,
    // so a figure measured against the other one would not be this render's.
    cfg.set_use_accelerated_attn(false);
    let measured = match crate::inference::model::zimage::dit::dry_forward(
        &cfg,
        files,
        prefix,
        crate::tensor::DType::BF16,
        height / FLUX_VAE_STRIDE,
        width / FLUX_VAE_STRIDE,
        FLUX_TEXT_TOKENS,
        zimage_alloc_shape(),
    ) {
        Ok(d) => {
            info!(
                "Z-Image dry run at {}x{}: the weights come to {:.2} GB and the pool holds \
                 {:.2} GB for them, and one forward holds \
                 {:.2} GB on top of them, over {} allocations; the reserve is that forward at \
                 {} percent, {:.2} GB, against {:.2} GB from the scaled seed{}",
                width,
                height,
                d.weights as f64 / 1e9,
                d.weights_held as f64 / 1e9,
                d.forward as f64 / 1e9,
                d.allocations,
                ZIMAGE_DRY_MARGIN_PERCENT,
                zruntime_demand_from_dry(d.forward, width, height) as f64 / 1e9,
                zimage_runtime_demand(width, height) as f64 / 1e9,
                if d.blind {
                    " (the forward read values a dry run cannot provide - trust the shapes only \
                     if none of them depended on one)"
                } else {
                    ""
                },
            );
            Some(d.forward)
        }
        // Not a failure of the load: a checkpoint this cannot walk keeps the seeded
        // figure, which is what every placement in this family used before there was
        // anything to measure.
        Err(e) => {
            info!("Z-Image dry run unavailable ({e}); the scaled seed stands");
            None
        }
    };
    if let Ok(mut m) = MEASURED.lock() {
        m.insert(key, measured);
    }
    measured
}

/// Build the transformer where a PLAN says, through the port the measurement walked.
///
/// The plan names a device per range of blocks; the constructor wants one device per
/// SLOT plus the slot each block belongs to, which is the same information keyed the
/// other way round. This is that transposition and nothing else - the distinct devices
/// in the plan's own order, so that slot zero is the card the plan starts on and the
/// stem lands where the reserve charged it.
///
/// A block no segment claims stays with the stem rather than being dropped: a plan that
/// does not cover its own stack is a bug in the plan, and building it as if the block
/// did not exist would hide it. The same rule the walk applies.
pub(super) fn zimage_native_from_plan(
    plan: &crate::inference::place::layer_executor::HeteroPlan,
    files: &[&str],
    prefix: Option<&str>,
    cfg: &crate::inference::model::zimage::dit::Config,
    primary: &Device,
    total_layers: usize,
) -> AnyResult<crate::inference::model::zimage::dit::ZImageTransformer2DModel> {
    use crate::inference::place::layer_executor::DeviceKind;
    let mut kinds: Vec<DeviceKind> = Vec::new();
    for seg in &plan.segments {
        if !kinds.contains(&seg.kind) {
            kinds.push(seg.kind);
        }
    }
    let mut devices: Vec<crate::tensor::Device> = Vec::with_capacity(kinds.len());
    for kind in &kinds {
        let dev = match kind {
            DeviceKind::Cuda(idx) => {
                if gpu_index_of(primary) == Some(*idx) {
                    primary.clone()
                } else {
                    crate::tensor::cuda_ext::new_device_with_stream(*idx)
                        .map(|d| d.native_device())
                        .map_err(|e| anyhow!("{e}"))?
                }
            }
            DeviceKind::Cpu => crate::tensor::Device::Cpu,
            other => return Err(anyhow!("z-image: no native slot for {other}")),
        };
        devices.push(dev);
    }
    let layer_slot: Vec<usize> = (0..total_layers)
        .map(|l| {
            plan.segments
                .iter()
                .find(|s| l >= s.layer_start && l < s.layer_end)
                .and_then(|s| kinds.iter().position(|k| *k == s.kind))
                .unwrap_or(0)
        })
        .collect();
    // SAFETY: the same mmap contract every load of these files takes.
    unsafe {
        crate::inference::model::zimage::dit::from_files_placed(
            cfg,
            files,
            prefix,
            crate::tensor::DType::BF16,
            &devices,
            &layer_slot,
        )
    }
    .map_err(|e| anyhow!("{e}"))
}

/// The card that carries the WHOLE transformer, or `None` when no single card does -
/// which is the loader's cue to build the split the measurement asked for.
///
/// THE LAST LINK, and the one that was missing. Deciding that a request needs two cards
/// and then loading it onto one is not a smaller version of the same behaviour, it is
/// the opposite of it: the decision said the model does not fit, and the loader fitted
/// it anyway, with what was left over standing in for the margin. A plan of N segments
/// selects the implementation for N segments, and nothing downstream may re-open the
/// question by weighing the model its own way.
///
/// So it goes through the same two answers everything else here goes through, in the
/// only order that is safe: no card at all unless one card holds it, and then the card
/// the plan names.
pub(super) fn zimage_whole_card(
    measured: Option<&crate::inference::place::dry_plan::Solution>,
    cuda_free: &[(usize, u64)],
    size_est: u64,
    headroom: u64,
    on_cuda: bool,
) -> Option<usize> {
    if !on_cuda || !zimage_fits_one_card(measured, cuda_free, size_est, headroom) {
        return None;
    }
    zimage_transformer_card(measured, cuda_free, size_est, headroom)
}

/// Whether ONE card can carry the whole transformer - the question the loader branches
/// on, with ONE answer.
///
/// THE READER THIS EXISTS FOR. The solver answers this already: it measures candidate
/// placements, charges each card its own blocks at what the ALLOCATOR holds for them
/// beside its own forward, and keeps the first that fits. A second reader asking the
/// same question of a file-size estimate and a single headroom figure is a second
/// declaration, and it will disagree - it did: a measurement that had split a request
/// across two cards was overruled by an estimate that read the weights at what they
/// count, 0.47 GB lighter, which is exactly the difference between fitting a card and
/// not. The render went to one card with 0.36 GB to spare where the measurement had
/// asked for two.
///
/// So when there is a measurement, IT decides, and the count of its segments is the
/// answer: one segment is one card. Without one, the estimate decides exactly as it did
/// before - the arrangement every placement in this family had before anything was
/// measured.
pub(super) fn zimage_fits_one_card(
    measured: Option<&crate::inference::place::dry_plan::Solution>,
    cuda_free: &[(usize, u64)],
    size_est: u64,
    headroom: u64,
) -> bool {
    match measured {
        Some(s) => s.plan.segments.len() <= 1,
        None => cuda_free.iter().any(|(_, m)| *m >= size_est + headroom),
    }
}

/// Which card the transformer takes, by the same rule and from the same answer.
///
/// The plan names it when there is one; otherwise the fastest card the estimate fits.
pub(super) fn zimage_transformer_card(
    measured: Option<&crate::inference::place::dry_plan::Solution>,
    cuda_free: &[(usize, u64)],
    size_est: u64,
    headroom: u64,
) -> Option<usize> {
    match measured {
        Some(s) => match s.plan.segments.first().map(|seg| seg.kind) {
            Some(crate::inference::place::layer_executor::DeviceKind::Cuda(idx)) => Some(idx),
            _ => None,
        },
        None => cuda_free
            .iter()
            .find(|(_, m)| *m >= size_est + headroom)
            .map(|(idx, _)| *idx),
    }
}

/// What the pool on the fastest CUDA card reserves in, and what it aligns to - the two
/// numbers a load has to be priced against.
///
/// `None` when there is no card, or when the pool has never been caught empty enough to
/// answer. Then every figure derived from it stays at what it counted, which is the
/// arrangement this file has always had.
pub(super) fn zimage_alloc_shape() -> Option<(u64, u64)> {
    #[cfg(feature = "cuda")]
    {
        let dev = crate::tensor::cuda::CudaDevice::get(0).ok()?;
        let (_, align) = dev.alloc_shape();
        Some((dev.pool_chunk()?, align))
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

/// [`zimage_runtime_demand`] with the checkpoint in hand, which is what lets it MEASURE.
///
/// Falls back to the scaled seed - value for value, the figure this fleet has been
/// planning against - whenever the walk has nothing to say: files that cannot be opened,
/// a layout the dry device cannot follow, an operation it does not cover. Losing the
/// measurement puts the placement back where it was, never below it.
pub fn zimage_runtime_demand_from_files(
    files: &[&str],
    prefix: Option<&str>,
    width: usize,
    height: usize,
) -> u64 {
    match zimage_dry_forward_bytes(files, prefix, width, height) {
        Some(forward) => zruntime_demand_from_dry(forward, width, height),
        None => zimage_runtime_demand(width, height),
    }
}

/// The transformer shards a Z-Image load would open, and the prefix they sit under.
///
/// A drop-in fine-tune arrives as ONE local all-in-one safetensors in the bundled
/// layout, where the S3-DiT lives under `model.diffusion_model.`; otherwise the official
/// repo's sharded transformer out of the HuggingFace cache. The same resolution the
/// loader performs, so the reserve is measured against the checkpoint that will be
/// loaded rather than whichever one happens to be on disk.
pub fn zimage_checkpoint_files(
    hf_models_dir: &str,
    local_ckpt: Option<&std::path::Path>,
) -> (Vec<std::path::PathBuf>, Option<&'static str>) {
    match local_ckpt {
        Some(p) => (vec![p.to_path_buf()], Some("model.diffusion_model")),
        None => (ImageEngine::zimage_transformer_shards(hf_models_dir), None),
    }
}

/// [`zimage_runtime_demand`] for a family whose checkpoint has not been resolved yet:
/// resolve it the way the loader would, then measure.
///
/// Same shape as every other family's reserve entry point - the models root and the
/// optional explicit checkpoint - so the admission gate and the loader ask the same
/// question of the same files.
pub fn zimage_runtime_demand_for(
    hf_models_dir: &str,
    local_ckpt: Option<&std::path::Path>,
    width: usize,
    height: usize,
) -> u64 {
    let (files, prefix) = zimage_checkpoint_files(hf_models_dir, local_ckpt);
    let refs: Vec<&str> = files.iter().filter_map(|p| p.to_str()).collect();
    if refs.is_empty() || refs.len() != files.len() {
        return zimage_runtime_demand(width, height);
    }
    zimage_runtime_demand_from_files(&refs, prefix, width, height)
}

/// Resident bytes of this family's caption encoder at BF16, from its config alone.
///
/// The checkpoint on disk is sharded and may be stored at a wider dtype, so its file
/// size does not answer the question the placement is asking - which is how much a
/// card must have free to hold the encoder we are about to build.
pub(crate) fn zimage_text_encoder_bytes() -> u64 {
    // measured-resident: 6.5 GB, sampled while this encoder was loaded. Counting it
    // from the config gives 7.3-8.0 GB depending on what one assumes about the
    // vocabulary table, and I have no derivation that reproduces the measurement - so
    // the measurement stands and the arithmetic does not pretend otherwise.
    //
    // The gap is not academic: this encoder shares a card with the transformer, and
    // deducting 1.2 GB too much moved twelve of thirty layers onto the host. An
    // over-estimate here is not caution, it is a spill.
    6_500_000_000
}

/// The dense embedders and output projection this family keeps resident on the card
/// holding the stem: the caption projection, the latent patch embedding and the way
/// back out. Counted from the config's widths, like every other figure here.
/// What the PRIMARY card carries beyond its share of the main blocks.
///
/// The projections - caption features in, patches in and out - AND the refiner layers,
/// which are built on that card whatever the plan says about the main stack.
///
/// The refiners were missing, and the arithmetic shows exactly what that costs: a card
/// probed with 16.0 GB free was given a 7.1 GB block budget, filled to 12 layers, and
/// came out with 1.4 GB free instead of the reserved headroom - because four refiner
/// layers at half a gigabyte each were placed on it that nobody had counted. The
/// denoise then ran out on its first broadcast. A component the loader always places
/// has to appear in the budget that decides what else fits beside it.
pub(super) fn zimage_embedder_bytes() -> u64 {
    let cfg = crate::inference::model::zimage::dit::Config::z_image_turbo();
    let d = cfg.dim as u64;
    let patch = cfg.all_patch_size.first().copied().unwrap_or(1) as u64;
    let patch_elems = cfg.in_channels as u64 * patch * patch;
    ((cfg.cap_feat_dim as u64) * d + patch_elems * d + d * patch_elems) * DENSE_BYTES
}

/// Everything the PRIMARY card carries beyond its share of the main blocks: the
/// projections above, plus the refiner layers the loader always builds there.
///
/// `checkpoint_bytes` is the transformer file, and a layer is taken as its size over the
/// main-layer count - the SAME convention the layer planner uses to decide how many fit,
/// so the two cannot disagree about what a layer costs.
pub(super) fn zimage_primary_overhead(checkpoint_bytes: u64) -> u64 {
    let cfg = crate::inference::model::zimage::dit::Config::z_image_turbo();
    let per_layer = checkpoint_bytes / (cfg.n_layers.max(1) as u64);
    // One noise refiner and one context refiner per configured refiner layer.
    zimage_embedder_bytes() + 2 * (cfg.n_refiner_layers as u64) * per_layer
}

/// Where this family's main blocks go when no single card holds them.
///
/// A FUNCTION, not a paragraph inside the loader, because the arithmetic here is the
/// whole difference between a render and a machine with two idle GPUs and a pinned
/// core - and until it was one it could not be checked without a GPU and a twelve
/// gigabyte load.
///
/// `cuda_free` is fastest-first and already net of anything else the loader has
/// committed to those cards. `runtime_demand` is what ONE forward needs free on
/// whichever card is running a block, and it is charged EXACTLY ONCE, by the planner,
/// on every card.
///
/// Charging it twice is what this replaces: the loader subtracted it from each card's
/// budget AND handed the planner the same figure again as a per-layer cost, so a card
/// holding a third of the model paid it one and a third times over. At the seeded
/// figure that is a couple of layers; the moment a render was observed to need more,
/// the double charge scaled with it and sixteen blocks of thirty went to the HOST -
/// on two cards that between them had room for all of them. An over-correction that
/// lands the model on the CPU is worse than the exhaustion it answers: the request
/// never fails, it just never finishes.
pub(crate) fn zimage_block_plan(
    cuda_free: &[(usize, u64)],
    ocl_free: &[(usize, u64)],
    total_main_layers: usize,
    transformer_bytes: u64,
    runtime_demand: u64,
) -> HeteroPlan {
    // The projections and the refiner layers are built on the first card the plan
    // fills, whatever the plan says about the main stack - so they come off THAT card,
    // by name, and they come out of the stack the planner divides. Leaving them in both
    // reserved them twice and cost six layers of thirty to the host.
    let primary_overhead = zimage_primary_overhead(transformer_bytes);
    let primary_idx = cuda_free.first().map(|(i, _)| *i).unwrap_or(0);
    let devices: Vec<(usize, u64)> = cuda_free
        .iter()
        .map(|(idx, free)| {
            let budget = if *idx == primary_idx {
                free.saturating_sub(primary_overhead)
            } else {
                *free
            };
            (*idx, budget)
        })
        .collect();
    let main_stack = transformer_bytes.saturating_sub(primary_overhead);
    let plan = HeteroPlan::calculate_with_kv_reserve(
        total_main_layers,
        main_stack,
        &devices,
        ocl_free,
        1.0,
        // Nothing per-layer: the scratch a forward needs is a property of the REQUEST,
        // not of how many blocks happen to sit on the card computing it.
        0,
        runtime_demand,
    );
    // SAY IT when the answer is a render nobody will wait for. A block on the host is
    // minutes per step, so a plan that spills while the cards had room between them is
    // not a degraded placement, it is a failure with no error message - the request runs
    // until the client gives up. The reserve is what took the room, so the line names it.
    let host: usize = plan
        .segments
        .iter()
        .filter(|s| {
            matches!(
                s.kind,
                crate::inference::place::layer_executor::DeviceKind::Cpu
            )
        })
        .map(|s| s.layer_end - s.layer_start)
        .sum();
    if host > 0 {
        let free: u64 = cuda_free.iter().map(|(_, f)| *f).sum();
        warn!(
            "Z-Image: {host} of {total_main_layers} blocks are going to the HOST while the \
             cards hold {:.1} GB free between them for a {:.1} GB model - each card is being \
             asked to keep {:.1} GB free for one forward, and that is the figure to check \
             before anything else",
            free as f64 / 1e9,
            transformer_bytes as f64 / 1e9,
            runtime_demand as f64 / 1e9,
        );
    }
    plan
}

pub fn flux_runtime_demand(width: usize, height: usize) -> u64 {
    flux_runtime_demand_for(width, height, None)
}

/// Same, plus the on-the-fly dequantisation scratch the given checkpoint implies.
///
/// Every matmul against a quantised weight materialises it, so a quantised
/// checkpoint needs room its dense equivalent does not - and that room scales with
/// the widest weight in the file, not with anything the request knows.
pub fn flux_runtime_demand_for(
    width: usize,
    height: usize,
    // Unused ON PURPOSE, and it must stay that way. The seed below is a MEASURED total -
    // what a render was seen to need - so it already contains the buffers a quantised
    // checkpoint materialises to run its matmuls. Adding a separately computed
    // dequantisation term on top double-counts it, and the margin here is thin enough
    // that a few hundred megabytes decide the placement: at 1024 this model fits one
    // card with 4.3 GB spare against 4.3 GB needed, so the extra term would split a
    // render that fits whole - and a split runs the cards in sequence, so it is slower,
    // not faster. The parameter is kept because the caller has the path and a future
    // ANALYTIC demand (one built from activation shapes rather than measurement) would
    // need it.
    _checkpoint: Option<&std::path::Path>,
) -> u64 {
    use crate::inference::place::runtime_demand::{latent_tokens, MeasuredReserve};
    // 4 GiB was found necessary for this family at 1024x1024, by renders that failed
    // with less. Scaled to the request, and superseded by what a render on THIS
    // machine is seen to take.
    const MEASURED: MeasuredReserve = MeasuredReserve {
        bytes: 4 << 30,
        reference_tokens: (1024 / (FLUX_VAE_STRIDE * FLUX_PATCH))
            * (1024 / (FLUX_VAE_STRIDE * FLUX_PATCH))
            + FLUX_TEXT_TOKENS,
    };
    let tokens = latent_tokens(height, width, FLUX_VAE_STRIDE, FLUX_PATCH) + FLUX_TEXT_TOKENS;
    let seed = crate::inference::place::runtime_demand::scale_measured(&MEASURED, tokens);
    crate::inference::place::runtime_demand::planning_demand("flux", width, height, seed)
}
/// Bytes per weight of the components that stay DENSE (the embedders and the final
/// layer), as opposed to the quantised block stack.
pub(super) const DENSE_BYTES: u64 = 4;

/// Widest full-resolution channel count in the Flux/Z-Image VAE decoders. The decode
/// peak is that feature map plus the 3x3 im2col of the same tensor, which is why a
/// 1024^2 decode wants ~6 GB while the decoder weights are a few hundred MB.
pub(super) const VAE_WIDEST_CH: u64 = 128;
/// Latent pixels of convolution context carried into each tile of a tiled VAE decode.
pub(super) const VAE_TILE_OVERLAP: usize = 8;

/// Peak VRAM of decoding the whole image in one pass.
///
/// This used to charge `W*H*(128*9 + 128*2)`, where the `*9` is the 3x3 im2col of the
/// widest stage at full resolution. That buffer is never materialised: `conv2d` windows
/// the column matrix to a bounded transient precisely so a megapixel feature map cannot
/// ask for multi-gigabyte columns. Charging it anyway made a 1024^2 decode look like
/// 5.9 GB when it needs under 2, so the most common render size tiled its decode on
/// every request - slower, for a limit that was not real.
///
/// So: the live pair of full-resolution maps, the convolution's own output, and the
/// column transient at the bound `conv2d` itself applies.
/// Start a VRAM watch on `gpu`, or nothing when the platform has none.
pub(super) fn vram_watch_on(
    gpu: usize,
) -> Option<crate::inference::place::vram_manager::VramWatch> {
    crate::inference::place::vram_manager::VramWatch::start(gpu)
}

/// Say what a WHOLE decode on the device actually took, next to what was predicted.
///
/// Reported, not fed back: the estimate that admitted this decode is one of two in this
/// file that disagree by more than three times, and correcting either by eye is what
/// broke a working placement. This line is the evidence a real correction needs.
///
/// Only a whole decode measures a whole decode - a tiled run gives up a fraction of the
/// memory, and a failed one never reached its peak.
pub(super) fn report_vae_decode_cost(
    what: &str,
    width: usize,
    height: usize,
    predicted: u64,
    free_before: u64,
    watch: crate::inference::place::vram_manager::VramWatch,
    succeeded: bool,
) {
    let took = free_before.saturating_sub(watch.finish());
    if !succeeded || took == 0 {
        return;
    }
    info!(
        "{what} VAE decode {width}x{height}: took {:.2} GB, the estimate that admitted \
         it said {:.2} GB ({:.1}x)",
        took as f64 / 1e9,
        predicted as f64 / 1e9,
        predicted as f64 / (took.max(1) as f64),
    );
}

pub(super) fn vae_whole_decode_peak(width: usize, height: usize) -> u64 {
    /// Maps of the widest stage live at once across the convolution chain.
    const LIVE_MAPS: u64 = 3;
    let pixels = (width * height) as u64;
    let maps = pixels * LIVE_MAPS * VAE_WIDEST_CH * 4;
    // measured-resident: the bound conv2d applies to its own column buffer.
    let columns = crate::tensor::Tensor::CONV2D_COL_BUDGET_ELEMS as u64 * 4;
    maps + columns
}

/// VRAM a T5 encode needs beyond the encoder weights: the prompt's activations and
/// its attention scores. Small next to the weights, but a card that fits ONLY the
/// weights makes every matmul bounce to the host - slower than a plain CPU encode.
///
/// Derived from the encoder's own shape and the padded prompt length, on the same
/// footing as every other demand here: an encoder twice as wide needs twice the
/// scratch, and a fixed figure could not know that.
pub(super) fn t5_encode_scratch_bytes(cfg: &crate::inference::model::t5::encoder::Config) -> u64 {
    crate::inference::place::runtime_demand::dit_activation_bytes(
        FLUX_TEXT_TOKENS,
        cfg.d_model,
        cfg.num_heads,
        cfg.d_ff as f64 / cfg.d_model.max(1) as f64,
    )
}

/// Resident bytes of a T5 ENCODER stack at `dtype_bytes` per weight, from the config
/// alone - the checkpoint on disk also carries the decoder (and may be f32), so its
/// file size over-states the encoder several times over.
pub(super) fn t5_encoder_bytes(
    cfg: &crate::inference::model::t5::encoder::Config,
    dtype_bytes: u64,
) -> u64 {
    let d_model = cfg.d_model as u64;
    let inner = (cfg.num_heads * cfg.d_kv) as u64;
    let d_ff = cfg.d_ff as u64;
    // q, k, v, o.
    let attn = 4 * d_model * inner;
    // Gated variants carry a second input projection (wi_0 + wi_1 + wo).
    let ff = if cfg.feed_forward_proj.gated {
        3 * d_model * d_ff
    } else {
        2 * d_model * d_ff
    };
    // Two RMS norms per block, one final norm, and the layer-0 relative-position bias.
    let per_layer = attn + ff + 2 * d_model;
    let params = (cfg.vocab_size as u64) * d_model
        + (cfg.num_layers as u64) * per_layer
        + d_model
        + (cfg.relative_attention_num_buckets * cfg.num_heads) as u64;
    params * dtype_bytes
}
