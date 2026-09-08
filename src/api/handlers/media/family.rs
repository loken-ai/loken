//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

pub fn image_family(model_name: &str) -> &'static str {
    if is_sdxl_model(model_name) {
        "sdxl"
    } else if is_boogu_model(model_name) {
        "boogu"
    } else if is_qwen_image_model(model_name) {
        "qwen-image"
    } else if is_z_image_model(model_name) {
        "zimage"
    } else if crate::inference::engine::flux2_engine::is_flux2_model(model_name) {
        // BEFORE the flux fallback: "flux2-klein-4b" contains "flux", so the default arm would
        // claim it and render a FLUX.2 request with FLUX.1 weights under the FLUX.2 name.
        "flux2"
    } else {
        "flux"
    }
}

/// Every family name [`image_family`] and [`image_family_from_header`] can return.
///
/// The loader table is keyed on this list and the test
/// `every_advertised_family_routes_to_its_own_loader` walks it, so a family added to
/// the classifiers without a loader fails a test instead of rendering through whatever
/// the fallback arm happened to be.
#[cfg(test)]
pub(crate) const IMAGE_FAMILIES: &[&str] =
    &["sdxl", "boogu", "qwen-image", "zimage", "flux2", "flux"];

/// Which loader a family's weights have to go through.
///
/// This used to be an inline `match` on the family STRING, written once per call site,
/// each copy ending in a `_ =>` arm that ran the FLUX loader. That default is what
/// turns a missing arm into a silent wrong-model render rather than an error: the SSE
/// path of `/api/generate` never grew an "sdxl" arm, so an SDXL request fetched
/// FLUX.1-schnell and generated with it under the SDXL name, at SDXL's step count and
/// guidance. Naming the loader in a type makes every dispatch EXHAUSTIVE - a new
/// variant does not compile until each match handles it - and leaves exactly one place
/// where an unknown family becomes an error that names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageLoader {
    ZImage,
    QwenImage,
    Boogu,
    Sdxl,
    Flux2,
    Flux,
}

/// The loader for `family`, or an error NAMING the family when none is wired.
pub fn image_family_loader(family: &str) -> Result<ImageLoader, String> {
    match family {
        "zimage" => Ok(ImageLoader::ZImage),
        "qwen-image" => Ok(ImageLoader::QwenImage),
        "boogu" => Ok(ImageLoader::Boogu),
        "sdxl" => Ok(ImageLoader::Sdxl),
        "flux2" => Ok(ImageLoader::Flux2),
        "flux" => Ok(ImageLoader::Flux),
        other => Err(format!(
            "image family '{other}' has no loader wired; refusing to load it with another \
             family's weights (wire it in image_family_loader)"
        )),
    }
}

/// Put `model_name`'s weights up through `loader`'s own loading path.
///
/// THE dispatch - one copy, shared by every caller that has to get an image model
/// resident. The HTTP path reached it through a `match` written inline in the handler,
/// the render CLI through an if/else chain of its own, and both ended in a branch that
/// ran the FLUX loader for anything they did not name: `image_render` had no arm for
/// boogu and none for flux2, so asking it for either one loaded FLUX.1-schnell and
/// rendered with it, printing the requested name and the requested family beside the
/// wrong picture. A copy is what lets one caller fall behind a family the other already
/// serves; keyed on [`ImageLoader`], there is nothing left to fall behind and a new
/// variant does not compile until it loads something.
///
/// `explicit_ckpt` names a file instead of letting the family's own resolver pick one, so
/// a specific checkpoint can be measured or compared without renaming anything on disk.
/// The two families whose loaders read a weights DIRECTORY rather than a single file
/// refuse it by name instead of ignoring it.
pub async fn load_image_family(
    engine: &crate::inference::engine::image_engine::ImageEngine,
    hf_models_dir: &str,
    loader: ImageLoader,
    model_name: &str,
    geom: crate::inference::place::runtime_demand::RequestGeometry,
    explicit_ckpt: Option<PathBuf>,
    report: Option<tokio::sync::mpsc::Sender<ImageEngineLoadingProgress>>,
    cancel: Option<crate::inference::serve::cancel::CancelToken>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    match loader {
        ImageLoader::ZImage => {
            let local =
                explicit_ckpt.or_else(|| find_ray_checkpoint(hf_models_dir, "rayzist", model_name));
            engine
                .load_z_image_turbo_cancellable(
                    ModelRequest::new(hf_models_dir, model_name),
                    local,
                    geom,
                    report,
                    cancel,
                )
                .await
        }
        ImageLoader::Flux2 => {
            if let Some(p) = explicit_ckpt {
                return Err(format!(
                    "the flux2 loader resolves its weights from a directory, so it cannot be \
                     pointed at the single file {}",
                    p.display()
                )
                .into());
            }
            engine
                .load_flux2(
                    ModelRequest::new(hf_models_dir, model_name),
                    geom,
                    report,
                    cancel,
                )
                .await
        }
        ImageLoader::QwenImage => {
            let local = explicit_ckpt.or_else(|| requested_checkpoint(hf_models_dir, model_name));
            engine
                .load_qwen_image(
                    ModelRequest::new(hf_models_dir, model_name),
                    local,
                    geom,
                    report,
                    cancel,
                )
                .await
        }
        ImageLoader::Boogu => {
            if let Some(p) = explicit_ckpt {
                return Err(format!(
                    "the boogu loader resolves its weights from a directory, so it cannot be \
                     pointed at the single file {}",
                    p.display()
                )
                .into());
            }
            engine
                .load_boogu(
                    ModelRequest::new(hf_models_dir, model_name),
                    geom,
                    report,
                    cancel,
                )
                .await
        }
        ImageLoader::Sdxl => {
            match (
                explicit_ckpt.or_else(|| find_sdxl_checkpoint(hf_models_dir, model_name)),
                find_clip_tokenizer(),
            ) {
                (Some(ckpt), Some(tok)) => {
                    engine
                        .load_sdxl_cancellable(
                            ckpt,
                            model_name.to_string(),
                            tok,
                            geom,
                            report,
                            cancel,
                        )
                        .await
                }
                (None, _) => {
                    Err(format!("no SDXL checkpoint for {model_name} under {hf_models_dir}").into())
                }
                (_, None) => Err("the CLIP tokenizer is not in the HF cache"
                    .to_string()
                    .into()),
            }
        }
        ImageLoader::Flux => {
            let local_gguf =
                explicit_ckpt.or_else(|| find_local_flux_gguf(hf_models_dir, model_name));
            engine
                .load_flux_schnell(
                    ModelRequest::new(hf_models_dir, model_name),
                    local_gguf,
                    geom,
                    report,
                    cancel,
                )
                .await
        }
    }
}

/// Detect the image-gen family from a checkpoint's SAFETENSORS header (the robust, name-independent
/// signal the metadata question asks for): read the tensor-name signature of the DiT and map it to
/// a family. Returns `None` for a non-image / unreadable / GGUF file. This is the foundation for a
/// future metadata-driven router; today it verifies/overrides the name-based [`image_family`] where
/// a path is available.
pub(crate) fn image_family_from_header(path: &std::path::Path) -> Option<&'static str> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut len8 = [0u8; 8];
    f.read_exact(&mut len8).ok()?;
    let hlen = u64::from_le_bytes(len8) as usize;
    if hlen == 0 || hlen > 32 * 1024 * 1024 {
        return None; // not a safetensors header (e.g. GGUF magic) or implausibly large
    }
    let mut hbuf = vec![0u8; hlen];
    f.read_exact(&mut hbuf).ok()?;
    let hdr = std::str::from_utf8(&hbuf).ok()?;
    // Match on tensor-name markers unique to each architecture. Ordered most-specific
    // first: boogu's DiT is a Flux-shaped 3-stream MMDiT, so its marker has to be tested
    // before the plain Flux one or every boogu checkpoint reads as Flux.
    if hdr.contains("double_stream_layers") && hdr.contains("instruct") {
        Some("boogu")
    } else if hdr.contains("double_blocks") && hdr.contains("single_blocks") {
        Some("flux")
    } else if hdr.contains("single_transformer_blocks") && hdr.contains("x_embedder") {
        // FLUX.2: diffusers-named like Qwen-Image, but it has SINGLE blocks and an
        // `x_embedder` where Qwen-Image has `img_in`. Tested first for that reason.
        Some("flux2")
    } else if hdr.contains("transformer_blocks") && hdr.contains("img_in") {
        Some("qwen-image")
    } else if hdr.contains("model.diffusion_model.input_blocks")
        || hdr.contains("conditioner.embedders")
    {
        // SDXL single-file: the LDM UNet numbering, and/or the pair of text encoders
        // bundled with it. Either marker alone is enough - a UNet-only checkpoint has
        // the first, a full pack has both.
        Some("sdxl")
    } else if hdr.contains("time_text_embed") && hdr.contains("norm_out.linear") {
        Some("zimage")
    } else if hdr.contains("cap_embedder") && hdr.contains("context_refiner") {
        // The SINGLE-FILE pack of the same family, which names its tensors after the
        // Lumina lineage it descends from rather than in the diffusers style the arm
        // above matches. Recognising only the diffusers spelling is what made a
        // 12.6 GB checkpoint sitting on disk disappear from the catalogue: it loads
        // and renders perfectly when named directly, it simply could not be found.
        //
        // Tested after the SDXL arm on purpose - both live under
        // `model.diffusion_model.`, and only these two markers separate them.
        Some("zimage")
    } else {
        None
    }
}

/// Per-model-family default knobs for image generation. The four
/// public image endpoints (/api/chat image-gen path, /v1/images/
/// generations, /v1/images/edits, /v1/images/variations) all need
/// the same defaults when the caller omits them; previously each
/// site duplicated the if/else table and a tweak to one (e.g.
/// raising Z-Image's default steps) would silently miss the other
/// three.
///
/// Z-Image-Turbo's 1024- default reflects its native training
/// resolution; Flux's 512- is faster and the bench-validated
/// default. Step counts likewise - 9 is the empirical sweet spot
/// for Z-Image's flow-matching scheduler, 4 is Flux Schnell's
/// trained step count.
#[derive(Debug, Clone, Copy)]
pub struct ImageModelDefaults {
    pub steps: usize,
    pub guidance: f64,
    pub size: usize,
    /// What the output's WIDTH and HEIGHT must be a multiple of.
    ///
    /// Not a preference - an architectural requirement, and the families here do not
    /// share it. A latent is the image over eight, and whatever halves that latent
    /// internally has to divide it evenly or the up path cannot rejoin the down path's
    /// skip. SDXL's U-Net halves three more times, so it needs 8 x 8 = 64 where a plain
    /// DiT needs only 16.
    ///
    /// Getting it wrong does not soften the picture, it ENDS the request: a source 784
    /// pixels tall gives a latent of 98, which halves to 49, which halves to 24 by
    /// truncation - and the concat that follows sees 48 against 49 and refuses.
    pub align: usize,
}

/// Can the resident model serve a request for `requested`?
///
/// `resident` is the name the resident was LOADED under and `resident_ckpt` the file
/// it came from; `requested_local` is the file this request resolves to, when it
/// resolves to one.
///
/// The property this must have is REFLEXIVITY: a model loaded for tag T always
/// serves a later request for T. It lost that when the loaders recorded a fixed
/// family name instead of the requested one - a Ray fine-tune resident called itself
/// by its base model's name, the Ray-flag test then read it as a different model, and
/// every request unloaded and reloaded the model it already had (measured: 33 s of
/// reload plus a 90 s admission wait per render, against 16 s of actual sampling).
/// `every_advertised_tag_serves_itself` is the gate.
pub(crate) fn resident_serves_request(
    resident: Option<&str>,
    resident_ckpt: Option<&str>,
    requested: &str,
    requested_local: Option<&str>,
) -> bool {
    let Some(name) = resident else { return false };
    if image_family(name) != image_family(requested) {
        return false;
    }
    // SHARING THE RESIDENT MUST BE PROVEN, NOT ASSUMED.
    //
    // Both sides resolving to a file is the strong case: the FILE is the identity, so
    // two fine-tunes of one family can never alias. Otherwise the only other thing that
    // proves it is the NAME.
    //
    // What used to stand here instead was a Ray-flag comparison - "both are Ray
    // fine-tunes, close enough" - and it is not close enough: `rayctifier` and
    // `rayburn` are two different SDXL checkpoints, both Ray, so a request for the
    // second was served by the first's weights. Observed as byte-identical PNGs from
    // two models, the second answering in 17 s against 43 because it loaded nothing.
    // Nothing reported it; the user simply got the wrong model.
    match (resident_ckpt, requested_local) {
        (Some(ckpt), Some(f)) => f == ckpt,
        _ => name == requested,
    }
}

/// Where THIS model's weights live, whatever family it belongs to.
///
/// The resident check used to resolve the requested checkpoint with the FLUX finder
/// alone, so every other family answered "no file" and fell through to a comparison
/// that could not tell two of its fine-tunes apart.
/// Keyed on [`ImageLoader`], for the reason the loader and budget tables are: the match
/// on the family STRING ended in a `_ =>` arm calling the FLUX finder, and that finder
/// does not look at the model name once its Ray branch misses - it scans the models
/// directory for any `*flux*schnell*.gguf`. So a boogu or a flux2 request was answered
/// with the FLUX checkpoint's path, and the family-filtered fallback below - the branch
/// that would have found the family's OWN file - never ran, because `by_family` was
/// already `Some`. An unwired family gets `None` here: there is no error channel on an
/// `Option`, and the load that follows refuses it by name in `image_family_loader`.
pub fn requested_checkpoint(hf_models_dir: &str, model_name: &str) -> Option<PathBuf> {
    let want = image_family(model_name);
    let by_family = match image_family_loader(want).ok()? {
        ImageLoader::Sdxl => find_sdxl_checkpoint(hf_models_dir, model_name),
        ImageLoader::ZImage => find_ray_checkpoint(hf_models_dir, "rayzist", model_name),
        ImageLoader::QwenImage => find_ray_checkpoint(hf_models_dir, "rayqwest", model_name),
        // These two resolve their weights from a DIRECTORY of the models dir, not from a
        // single file selected by name: there is no per-family finder to ask, and asking
        // another family's is what put a FLUX path under a boogu name.
        ImageLoader::Boogu | ImageLoader::Flux2 => None,
        ImageLoader::Flux => find_local_flux_gguf(hf_models_dir, model_name),
    };
    // The per-family finders only look in that family's own weight directory. Anything
    // dropped in elsewhere - which is every checkpoint the catalogue advertises by its
    // directory name - resolves here instead. The header family must agree with the one
    // the request routes to, or a name that merely looks like another family's would
    // hand the loader the wrong architecture.
    by_family.or_else(|| {
        find_local_image_checkpoint(hf_models_dir, model_name)
            .filter(|(_, fam)| *fam == want)
            .map(|(p, _)| p)
    })
}

/// The sampling recipe a family was validated at, or an error NAMING the family when it
/// has none of its own.
///
/// Keyed on [`ImageLoader`] for the third time in this file, and for the same reason: the
/// match on the family STRING ended in a `_ =>` arm returning FLUX Schnell's recipe, so a
/// family added to `image_family` but never given a recipe rendered at 4 steps, guidance
/// 4.0 and 512 pixels - another model's numbers, applied silently, and read by the render
/// routes, by the placement reserve (which sizes itself from the default when a route
/// names no geometry) and by the knobs `/api/show` publishes to clients. A 20-step CFG
/// family run at Schnell's 4 steps does not fail; it returns a worse picture, under its
/// own name. Exhaustive now: a variant added without a recipe does not compile.
pub fn image_model_defaults(model_name: &str) -> Result<ImageModelDefaults, String> {
    image_model_defaults_for_family(image_family(model_name), model_name)
}

/// The recipe for an explicitly named `family`, with `model_name` deciding only between
/// the drop-in variants a family serves under one loader.
///
/// Split out from [`image_model_defaults`] so the FAMILY can be named directly: the
/// classifier one level up ends in its own fallback to "flux", so passing it a name is
/// not a way to ask what an unwired family gets.
pub(super) fn image_model_defaults_for_family(
    family: &str,
    model_name: &str,
) -> Result<ImageModelDefaults, String> {
    Ok(match image_family_loader(family)? {
        ImageLoader::ZImage => ImageModelDefaults {
            steps: 9,
            guidance: 5.0,
            size: 1024,
            align: 16,
        },
        ImageLoader::QwenImage => {
            // A step-distilled drop-in of the family: its own pipeline runs 4 steps and
            // passes no guidance, because the teacher's classifier-free guidance is folded
            // into the weights. Guidance 1.0 is what "no CFG branch" means to the sampler -
            // the conditional prediction IS the step - so it also halves the DiT forwards.
            if crate::inference::engine::qwen_image_engine::is_step_distilled(model_name) {
                ImageModelDefaults {
                    steps: 4,
                    guidance: 1.0,
                    size: 1024,
                    align: 16,
                }
            } else {
                // Qwen-Image: flow-match DiT with CFG; 20 steps is a reasonable speed/quality
                // default, guidance 4.0 matches the validated pipeline, 1024 is its native res.
                ImageModelDefaults {
                    steps: 20,
                    guidance: 4.0,
                    size: 1024,
                    align: 16,
                }
            }
        }
        // FLUX.2 Klein is step-distilled (`is_distilled: true`) and its DiT carries no guidance
        // embedding at all, so 4 steps and guidance 1.0 - which the sampler reads as "run one
        // branch". 1024 is its native resolution; align 16 because a DiT token here covers 16
        // pixels a side (VAE 8 x the pipeline's own 2x2 latent patchify), not the usual 8.
        ImageLoader::Flux2 => ImageModelDefaults {
            steps: 4,
            guidance: 1.0,
            size: 1024,
            align: 16,
        },
        // Boogu-Image-Turbo (reference: euler + sgm_uniform): 4 steps, CFG 1.0 (distilled, no
        // guidance), 1024 native.
        ImageLoader::Boogu => ImageModelDefaults {
            steps: 4,
            guidance: 1.0,
            size: 1024,
            align: 16,
        },
        // SDXL: a full CFG model (not distilled), trained at 1024. Its common
        // recipe is ~25 Euler steps at guidance 7.
        // align 64: its U-Net halves the latent three times over.
        ImageLoader::Sdxl => ImageModelDefaults {
            steps: 25,
            guidance: 7.0,
            size: 1024,
            align: 64,
        },
        ImageLoader::Flux => {
            // Ray fp8 fine-tunes are dev-family checkpoints (guidance-embedded, not
            // step-distilled): the dev recipe is ~20 steps at guidance ~3.5, native 1024.
            // Only the FLUX-family Ray drop-ins reach here - rayzist and rayqwest classify
            // as their own families above, which is what the string table did too.
            if is_ray_fp8_model(model_name) {
                ImageModelDefaults {
                    steps: 20,
                    guidance: 3.5,
                    size: 1024,
                    align: 16,
                }
            } else {
                // Flux Schnell: 4 trained steps, 512^2 fast default.
                ImageModelDefaults {
                    steps: 4,
                    guidance: 4.0,
                    size: 512,
                    align: 16,
                }
            }
        }
    })
}

/// True when the model name matches one of the image-gen
/// families the server can actually serve (Flux + Z-Image). Same
/// substring set as the dispatch checks in /api/chat + /api/generate
/// so the routes and the modality-aware /api/show capabilities
/// helper can't drift on which names count as image-gen.
pub(crate) fn is_image_gen_model(model_name: &str) -> bool {
    let lower = model_name.to_lowercase();
    lower.contains("flux")
        || lower.contains("z-image")
        || lower.contains("z_image")
        || lower.contains("qwen-image")
        || lower.contains("qwen_image")
        || lower.contains("boogu")
        // FLUX.2 Klein, including the bare `klein` tag a weights directory can produce.
        || crate::inference::engine::flux2_engine::is_flux2_model(&lower)
        // Ray fp8-scaled drop-in variants (rayflux already matches via "flux").
        || lower.contains("rayqwest")
        || lower.contains("rayzist")
        // The SDXL family: one single-file checkpoint per model.
        || is_sdxl_model(model_name)
}

/// Resident weight bytes of a family's hot component (what a card must hold).
///
/// Keyed on [`ImageLoader`] rather than on the family STRING, for the reason the loader
/// table is: the string match ended in a `_ =>` arm returning the FLUX checkpoint's
/// size, so a family nobody budgeted was reserved for silently as if it were Flux. A
/// reserve is not a description of what a component allocates - it is what stops the
/// planner from putting anything else on that card - so a family carrying another
/// family's figure is an OOM or a placement on the host with nothing to point at. The
/// enum makes this table exhaustive: a family added without a budget of its own does
/// not compile, and an unwired family is refused by name in `image_family_loader`
/// before any reserve is computed.
pub(super) fn family_hot_bytes(hf_models_dir: &str, loader: ImageLoader, model_name: &str) -> u64 {
    match loader {
        ImageLoader::Boogu => {
            crate::inference::engine::boogu_engine::hot_component_bytes(hf_models_dir)
        }
        ImageLoader::QwenImage => crate::inference::engine::qwen_image_engine::hot_component_bytes(
            hf_models_dir,
            requested_checkpoint(hf_models_dir, model_name).as_deref(),
        ),
        ImageLoader::ZImage => {
            crate::inference::engine::ImageEngine::zimage_hot_component_bytes(hf_models_dir)
        }
        // The UNet is the hot component; the checkpoint also holds the towers and
        // the VAE, which are placed separately, so scale the file down to it.
        ImageLoader::Sdxl => find_sdxl_checkpoint(hf_models_dir, model_name)
            .and_then(|p| std::fs::metadata(p).ok().map(|m| m.len() * 3 / 4))
            .unwrap_or(0),
        // Its own figure, from the parameter count: the checkpoint is bf16 and the loader
        // re-quantizes to Q8_0, so the file is twice what ends up resident.
        ImageLoader::Flux2 => {
            crate::inference::engine::flux2_engine::hot_component_bytes(hf_models_dir)
        }
        ImageLoader::Flux => find_local_flux_gguf(hf_models_dir, model_name)
            .and_then(|p| std::fs::metadata(p).ok().map(|m| m.len()))
            .unwrap_or(0),
    }
}

/// VRAM one generation of a family needs FREE beyond its weights, FOR THIS REQUEST.
///
/// There used to be two figures here - a "comfortable" one and a lower "floor" a
/// family could not go below - because neither was derived and the pair bought some
/// slack against being wrong. Both were fixed byte counts, so both were wrong in the
/// same direction at once: too large at 512x512, refusing placements that would have
/// run, and too small above the resolution they were written for, admitting a load
/// that then could not denoise. One derived number replaces them, and there is no
/// tier to relax because there is nothing left to be conservative about.
///
/// Every arm below reads the model's OWN architecture - width, heads, patch size,
/// feed-forward ratio - and the geometry of the request being served. Nothing here
/// is a memory size chosen by hand.
///
/// Keyed on [`ImageLoader`] for the same reason as [`family_hot_bytes`]: the `_ =>` arm
/// this replaces handed every unbudgeted family the FLUX reserve, which is a placement
/// decision made on another model's architecture. Exhaustive now - a new family cannot
/// reach the planner without stating what one of its generations needs.
pub(super) fn family_runtime_bytes(
    hf_models_dir: &str,
    loader: ImageLoader,
    model_name: &str,
    geom: crate::inference::place::runtime_demand::RequestGeometry,
) -> u64 {
    use crate::inference::place::runtime_demand as demand;
    // Architecture facts shared by every VAE in use here, not tuning knobs: the
    // encoder reduces space by 8, and the decoder's last level carries 128 channels
    // at full output resolution.
    const VAE_STRIDE: usize = 8;
    const VAE_DECODER_WIDTH: usize = 128;
    // The denoise and the decode are SEQUENTIAL phases - the denoiser's scratch is
    // freed before the decoder allocates - so a family needs the LARGER of the two,
    // never their sum. Summing them once cost enough headroom to refuse a
    // single-card placement that fits and split the model across GPUs instead.
    let vae = demand::vae_decode_bytes(geom.height, geom.width, VAE_DECODER_WIDTH);
    match loader {
        // Derived from this family's OWN architecture and the requested geometry - notably a
        // DiT token here is 16 pixels a side, not the 8 every other family uses.
        ImageLoader::Flux2 => {
            crate::inference::engine::flux2_engine::runtime_headroom_bytes(geom.width, geom.height)
        }
        ImageLoader::QwenImage => {
            crate::inference::engine::qwen_image_engine::runtime_headroom_bytes(
                hf_models_dir,
                requested_checkpoint(hf_models_dir, model_name).as_deref(),
                geom.width,
                geom.height,
            )
        }
        // The SAME helper the Z-Image loader uses to decide single-card vs split, for
        // the reason the Flux arm gives below. A second figure stood here, derived
        // independently from the activation shapes: it came out at 1.9 GB where the
        // placement asked for 3.2 GB at 1024^2, and - the part that mattered - it was
        // not wired to the measurement loop, so a shape that had just exhausted a card
        // went on being admitted against the estimate that admitted the failure while
        // the planner had already learned better. Its decode term is inside that helper
        // now, so this arm cannot drift from it again.
        //
        // The checkpoint goes in because the helper WALKS it: one forward of these files
        // at this geometry, counted rather than scaled from a seed. Resolved the way the
        // loader resolves it, so the gate and the load measure the same file.
        ImageLoader::ZImage => crate::inference::engine::image_engine::zimage_runtime_demand_for(
            hf_models_dir,
            find_ray_checkpoint(hf_models_dir, "rayzist", model_name).as_deref(),
            geom.width,
            geom.height,
        ),
        // 5 GiB was found necessary for this family at 1024x1024; scaled to the
        // request, and superseded by what a render is measured to take.
        ImageLoader::Boogu => {
            let cfg = crate::inference::model::boogu::dit::Config::default();
            let reference = (1024 / (VAE_STRIDE * cfg.patch_size))
                * (1024 / (VAE_STRIDE * cfg.patch_size))
                + crate::inference::engine::image_engine::FLUX_TEXT_TOKENS;
            let measured = demand::MeasuredReserve {
                bytes: 5 << 30,
                reference_tokens: reference,
            };
            let tokens = geom.tokens(VAE_STRIDE, cfg.patch_size)
                + crate::inference::engine::image_engine::FLUX_TEXT_TOKENS;
            demand::planning_demand(
                "boogu",
                geom.width,
                geom.height,
                demand::scale_measured(&measured, tokens).max(vae),
            )
        }
        // Same contract: the figure this family was found to need at 1024x1024,
        // scaled by the latent's area, and replaced by measurement once there is one.
        ImageLoader::Sdxl => {
            let reference = (1024 / VAE_STRIDE) * (1024 / VAE_STRIDE);
            let measured = demand::MeasuredReserve {
                bytes: 5 << 30,
                reference_tokens: reference,
            };
            let tokens = (geom.height / VAE_STRIDE) * (geom.width / VAE_STRIDE);
            demand::planning_demand(
                "sdxl",
                geom.width,
                geom.height,
                demand::scale_measured(&measured, tokens).max(vae),
            )
        }
        // Same helper the Flux loader itself uses to decide single-device vs split,
        // so "fits" here means "will actually place AND generate there". Two
        // independently written figures for one request is how admission and
        // placement came to disagree in the first place.
        ImageLoader::Flux => crate::inference::engine::image_engine::flux_runtime_demand_for(
            geom.width,
            geom.height,
            find_local_flux_gguf(hf_models_dir, model_name).as_deref(),
        )
        .max(vae),
    }
}

// `DIT_MLP_RATIO` stood here - a feed-forward width "every DiT here is built at" - and
// its last reader was the Z-Image arm above, which now asks the family's own loader.
// It was wrong for that family: Z-Image builds its feed-forward at eight thirds of the
// model width, not four times it, and a ratio applied to a model that does not have it
// is an architecture fact invented at the call site. A config that carries the number is
// the only thing entitled to answer.
