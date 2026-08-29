//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

impl WanVariant {
    /// Parse a model-selection token (env `WAN_MODEL` or a CLI flag). Anything containing
    /// "14" selects the 14B GGUF; everything else stays on the 1.3B default.
    pub fn from_token(s: &str) -> WanVariant {
        if s.contains("14") {
            WanVariant::B14
        } else {
            WanVariant::B1_3
        }
    }
}

/// Resolve the Wan 1.3B DiT safetensors via the Stage-1 HF-cache resolver (never a
/// hardcoded path / env var).
pub fn wan_dit_file() -> std::path::PathBuf {
    crate::inference::model::wan::vae::wan_file("diffusion_pytorch_model.safetensors")
}

/// Where a community fine-tune of the video model is looked for, beside the Ray families.
pub const WAN_CHECKPOINT_DIR: &str = "wan";

/// The 1.3B DiT a request asks for: a fine-tune dropped in `<models>/wan/`, or the base
/// checkpoint when the name matches none.
///
/// Same shape as the Ray families - drop a `.safetensors` in a directory and it becomes
/// selectable - because that is the mechanism this repo already has and a second one would
/// only be a second thing to learn. The file's stem is normalised to a variant tag by the
/// same function the Ray resolver uses, so `Wan.Photoreal.v2.safetensors` answers to
/// "wan-photoreal", and the largest match wins when several files share a tag (the
/// full-precision one over a quant of it).
///
/// Geometry is NOT checked here - see [`check_wan_geometry`], which the loader calls. A
/// checkpoint of the wrong shape must be refused with a sentence, not loaded into a video
/// that comes out plausible and wrong.
pub fn wan_dit_file_for(hf_models_dir: &str, model_name: &str) -> std::path::PathBuf {
    let dir = std::path::Path::new(hf_models_dir).join(WAN_CHECKPOINT_DIR);
    let want = wan_checkpoint_tag(model_name);
    let found = wan_checkpoint_in(&dir, &want);
    found.unwrap_or_else(wan_dit_file)
}

/// The checkpoint in `dir` whose stem answers to `want`, largest first.
///
/// BOTH formats: the 1.3B ships as safetensors and every community 14B is a GGUF, so a
/// scanner that took only one of them would have offered exactly the checkpoints nobody
/// has. The extension decides which loader runs, not which files are visible.
pub fn wan_checkpoint_in(dir: &std::path::Path, want: &str) -> Option<std::path::PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter(|e| {
            let p = e.path();
            let ext_ok = p.extension().is_some_and(|x| {
                x.eq_ignore_ascii_case("safetensors") || x.eq_ignore_ascii_case("gguf")
            });
            ext_ok
                && p.file_stem()
                    .and_then(|s| s.to_str())
                    .map(crate::inference::model::wan::dit::wan_checkpoint_tag)
                    .as_deref()
                    == Some(want)
        })
        .max_by_key(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
        .map(|e| e.path())
}

/// The catalogue name for a Wan checkpoint file, from its file stem.
///
/// The generic tag builder splits a stem on every separator and drops purely numeric
/// tokens, which is right for `v2` or `Q8` and wrong for a parameter count: `..._1_3B_...`
/// loses its leading digit and the checkpoint is advertised as a 3B model when it is a
/// 1.3B one. A user picking from a list has no way to see that is a lie.
///
/// So Wan checkpoints are named here instead: keep the family and the size, drop the
/// packaging - the quantisation, the export rank, the task suffix, the word "lora" - and
/// put a split parameter count back together. `Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32`
/// becomes `wan-causvid-1.3b`, which is short enough to pick from a menu and true.
pub fn wan_checkpoint_tag(stem: &str) -> String {
    let lower = stem.to_ascii_lowercase();
    let raw: Vec<&str> = lower
        .split(|c: char| matches!(c, '.' | '_' | '-' | ':'))
        .filter(|t| !t.is_empty())
        .collect();
    // Rejoin a parameter count that a separator split in two: a bare number immediately
    // followed by a number-with-a-unit is one figure, not two tokens.
    let mut toks: Vec<String> = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < raw.len() {
        let t = raw[i];
        let next = raw.get(i + 1).copied().unwrap_or("");
        let is_num = !t.is_empty() && t.chars().all(|c| c.is_ascii_digit());
        let next_sized = next.len() >= 2
            && next.chars().next().is_some_and(|c| c.is_ascii_digit())
            && next.ends_with('b');
        if is_num && next_sized {
            toks.push(format!("{t}.{next}"));
            i += 2;
            continue;
        }
        toks.push(t.to_string());
        i += 1;
    }
    let noise = |t: &str| -> bool {
        // `t2v` goes because it is the default kind and says nothing; `i2v` STAYS, because
        // it is what distinguishes a checkpoint that continues a frame from one that does
        // not - dropping it made the two 14B models collide on one name, and whichever
        // resolved first would have served the other's requests.
        matches!(t, "lora" | "t2v" | "safetensors" | "gguf" | "fp8" | "fp16" | "bf16"
                  | "full" | "diffusion" | "model" | "wan21" | "wan2" | "wan")
            || t == "latest"
            || t.starts_with("rank")
            || t.starts_with("bidirect")
            // A version marker: v2, v2.0. Two versions of one tune are one model to a
            // picker, and the larger file wins between them.
            || (t.starts_with('v') && t[1..].chars().next().is_some_and(|c| c.is_ascii_digit()))
            // A quantisation label: q8, q5, e4m3 and friends.
            || ((t.starts_with('q') || t.starts_with('e'))
                && t.len() >= 2
                && t[1..].chars().next().is_some_and(|c| c.is_ascii_digit()))
            || t.chars().all(|c| c.is_ascii_digit())
    };
    let kept: Vec<String> = toks.into_iter().filter(|t| !noise(t)).collect();
    // Always prefixed, so the listing can tell at a glance that this is a video model and
    // the family check downstream keeps working.
    if kept.is_empty() {
        return "wan".to_string();
    }
    format!("wan-{}", kept.join("-"))
}

/// The full checkpoint a LoRA corrects.
///
/// A LoRA carries no weights of its own worth rendering: it is a correction, and applying it
/// to the wrong base gives a model that loads and renders the wrong thing. The base is read
/// off the ADAPTER - an image-to-video adapter has the image cross-attention keys that only
/// an image-to-video checkpoint has - and looked up in the same directory the adapter came
/// from, so a user who dropped one in has the other beside it.
pub(super) fn base_for_lora(lora: &std::path::Path) -> Result<std::path::PathBuf> {
    let Ok(ld) = (unsafe { SafeTensorsLoader::multi(&[lora]) }) else {
        return Ok(wan_14b_dit_file());
    };
    let is_i2v = ld.names().iter().any(|n| n.contains("k_img"));
    if !is_i2v {
        return Ok(wan_14b_dit_file());
    }
    let dir = lora.parent().unwrap_or(std::path::Path::new("."));
    // Among the checkpoints this adapter can correct, take the SMALLEST.
    //
    // An adapter exists to make a model cheap to run, and the weights are what the forward
    // has to share a card with: an 18 GB checkpoint leaves 7.5 GB for a forward that wants
    // more, spills the remainder onto the host, and a block on the host costs an order of
    // magnitude - far more than the difference between two quantisations. The choice is
    // logged rather than silent, because it IS a quality decision and whoever is looking at
    // the output should be able to see which weights made it.
    let mut candidates: Vec<(u64, std::path::PathBuf)> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p != lora
                && p.extension()
                    .is_some_and(|x| x.eq_ignore_ascii_case("gguf"))
        })
        .filter(|p| checkpoint_wants_reference(p))
        .filter_map(|p| std::fs::metadata(&p).ok().map(|m| (m.len(), p)))
        .collect();
    candidates.sort_by_key(|(len, _)| *len);
    match candidates.first() {
        Some((len, p)) => {
            eprintln!(
                "[wan-14b] adapter base: {} ({:.1} GB, smallest of {} that it fits)",
                p.display(),
                *len as f64 / 1e9,
                candidates.len()
            );
            Ok(p.clone())
        }
        None => Err(crate::tensor::Error(format!(
            "{}: this is an image-to-video adapter and needs an image-to-video checkpoint \
             to correct. None is in {}.",
            lora.display(),
            dir.display()
        ))),
    }
}

/// Does this checkpoint render a clip that CONTINUES a frame?
///
/// Answered from the file and BEFORE any weight is loaded, because the answer decides
/// whether a start frame has to be produced first - and producing one means running an
/// image model, which needs the card the denoiser is about to take. Asking the loaded DiT
/// is too late by exactly the amount of memory that matters.
///
/// Both shapes are covered: a whole checkpoint carries `img_emb`, and an adapter for one
/// carries the image cross-attention keys.
pub fn checkpoint_wants_reference(path: &std::path::Path) -> bool {
    let ext = path.extension().and_then(|x| x.to_str()).unwrap_or("");
    if ext.eq_ignore_ascii_case("safetensors") {
        return match unsafe { SafeTensorsLoader::multi(&[path]) } {
            Ok(ld) => ld
                .names()
                .iter()
                .any(|n| n.contains("k_img") || n.contains("img_emb")),
            Err(_) => false,
        };
    }
    match crate::tensor::quantized::gguf_file::open_header(path) {
        Ok(c) => c.tensor_infos.keys().any(|n| n.starts_with("img_emb.")),
        Err(_) => false,
    }
}

/// Is this checkpoint a LoRA correction rather than a whole DiT?
///
/// The distilled Wan variants - the ones that render in four steps instead of forty - are
/// published as rank-32 corrections of about a hundred megabytes, not as replacement
/// checkpoints. A user picking one from the same list as a full checkpoint should get what
/// they asked for either way, so the FILE decides: a LoRA carries `lora_down` tensors and
/// has no `patch_embedding.weight` of its own.
///
/// Header-only: safetensors puts its tensor index in a JSON prefix, so this reads a few
/// kilobytes whatever the file weighs.
pub fn is_wan_lora(path: &std::path::Path) -> bool {
    if !path
        .extension()
        .is_some_and(|x| x.eq_ignore_ascii_case("safetensors"))
    {
        return false;
    }
    let Ok(ld) = (unsafe { SafeTensorsLoader::multi(&[path]) }) else {
        return false;
    };
    !ld.contains("patch_embedding.weight") && ld.names().iter().any(|n| n.contains("lora_down"))
}

/// How many steps and how much guidance this checkpoint was distilled for.
///
/// A step-distilled checkpoint is not merely faster at the same settings - it was trained
/// to be integrated in a handful of steps with NO classifier-free guidance, and running it
/// at the base model's forty steps and guidance 5 does not give a better picture, it gives
/// a broken one. The checkpoint therefore has to be able to say so, or every user of it
/// has to know a number that is not written anywhere they can see.
///
/// Returns `(steps, cfg)` for a recognised distillation, `None` for a checkpoint that wants
/// the base schedule. Matched on the family name in the file, which is what the publishers
/// name these by; an unrecognised file simply keeps the base defaults.
pub fn wan_distilled_defaults(path: &std::path::Path) -> Option<(usize, f32)> {
    let name = path.file_stem()?.to_str()?.to_lowercase();
    // Both are step-distillations of Wan 2.1 trained without guidance. The step counts are
    // the ones their authors publish; guidance 1.0 means the unconditional branch is not
    // evaluated at all, which is half the remaining work.
    if name.contains("causvid") {
        return Some((4, 1.0));
    }
    if name.contains("fusionx") {
        return Some((8, 1.0));
    }
    // The lightx2v step+cfg distillation of the image-to-video checkpoint. Four steps and
    // no guidance against forty guided ones is twenty times fewer forwards on the same
    // weights, which is the difference between an unusable render and one.
    if name.contains("lightx2v") || name.contains("stepdistill") || name.contains("distill") {
        return Some((4, 1.0));
    }
    None
}

/// Does this checkpoint have the shape the variant expects?
///
/// A fine-tune of another size, or an I2V checkpoint, loads tensor by tensor and only
/// fails somewhere deep - or does not fail at all and renders something that looks like a
/// video. The patch embedding carries both facts that matter in one tensor: its input
/// channel count and the model width. I2V has 36 input channels where T2V has 16, and the
/// 14B is 5120 wide where the 1.3B is 1536, so one header read separates every case the
/// user can currently get wrong.
///
/// `shape` is `patch_embedding.weight`'s dims as the file stores them. Returns the reason
/// it does not fit, or `None` when it does.
pub fn check_wan_geometry(shape: &[usize], want_dim: usize, want_in_ch: usize) -> Option<String> {
    // safetensors stores it [dim, in_ch, 1, 2, 2]; the GGUF reader hands back the reverse.
    // The model width is the LARGEST extent by a wide margin - the other four are a channel
    // count and the patch, all small - so whichever END carries it names the order, and the
    // channel count is its neighbour. Deciding by "the first is at least the second" does
    // not work: in GGUF order that compares two patch dimensions, which are equal.
    // Refuse a shape that is not one BEFORE reading anything out of it: returning "fits"
    // for a degenerate shape is the one answer that must never come out of a guard.
    if shape.len() < 2 {
        return Some(format!(
            "patch_embedding.weight has shape {shape:?}, which is not a Wan patch embedding"
        ));
    }
    let (first, last) = (shape[0], shape[shape.len() - 1]);
    let (dim, in_ch) = if first >= last {
        (first, shape[1])
    } else {
        (last, shape[shape.len() - 2])
    };
    if in_ch != want_in_ch {
        return Some(format!(
            "this checkpoint takes {in_ch} input channels, not {want_in_ch}. Sixteen is \
             text-to-video and thirty-six is image-to-video, which needs a starting picture \
             and a CLIP tower this variant does not load"
        ));
    }
    if dim != want_dim {
        return Some(format!(
            "this checkpoint is {dim} wide and the selected variant is {want_dim}. Name it \
             so it selects the matching variant - anything containing '14' picks the 14B"
        ));
    }
    None
}

/// Resolve the Wan 14B DiT Q8_0 GGUF (city96 community quant) via the HF-cache resolver
/// (`models--city96--Wan2.1-T2V-14B-gguf/snapshots/*/wan2.1-t2v-14b-Q8_0.gguf`).
pub fn wan_14b_dit_file() -> std::path::PathBuf {
    crate::inference::model::wan::vae::wan_file_in(
        "models--city96--Wan2.1-T2V-14B-gguf",
        "wan2.1-t2v-14b-Q8_0.gguf",
    )
}

// -- geometry shared by BOTH variants (patch (1,2,2), head_dim 128, freq_dim 256, in/out
//    channels 16, text_dim 4096, eps 1e-6); only dim/n_heads/n_layers differ per variant. --
pub(super) const HEAD_DIM: usize = 128;
pub(super) const EPS: f32 = 1e-6;
pub(super) const OUT_CH: usize = 16;
/// Input channels an image-to-video checkpoint takes: the 16 noise channels, a 4-channel
/// mask saying which frames are given, and the 16 channels of the frame being continued.
pub(super) const I2V_IN_CH: usize = 36;
pub(super) const FREQ_DIM: usize = 256;
pub(super) const TEXT_LEN: usize = 512; // umT5 context is zero-padded to this before text_embedding
pub(super) const PATCH_H: usize = 2;
pub(super) const PATCH_W: usize = 2;
pub(super) const ROPE_THETA: f64 = 10000.0;

/// The shared projection, which holds the published orientation and multiplies with the N-T
/// flag: this checkpoint's weights are half-precision and already on the device, so turning
/// them would materialise an F32 cast and a transposed copy per tensor.

/// A quantized Linear `[out,in]` (Q8_0 GGUF) + OPTIONAL F32 bias. The weight stays quantized
/// on its device and dequantizes on-the-fly inside `QMatMul::forward` (the same on-device
/// pattern as the ACE-Step DiT); the bias (a separate `.bias` GGUF tensor, F32) is added
/// after the matmul. Used for the 14B path's parameter mass so a 14B-class DiT keeps its
/// compact (~Q8) device footprint instead of a 4x F32 blow-up.
// MEASURED, and it is why this projection keeps the block-quant kernel: routing it through
// `forward_dequant_gpu` - dequantize the weight on the device, then one BF16 tensor-core GEMM,
// which is what the image DiTs do for wide activations - is a 2.9x REGRESSION here, 28.3 s/step
// becoming 81.1 s. At this model's weight sizes the dequant costs more per call than the
// block-quant kernel saves, even at twenty thousand rows. Do not switch it without an A/B on a
// 14B checkpoint. `Weight::Quant` is exactly that kernel.

/// One block linear: dense (1.3B safetensors) or quantized (14B GGUF). The forward math is
/// IDENTICAL across variants - only the underlying matmul op differs.
pub(super) enum WanLinear {
    Dense(Linear),
    Quant(QLinear),
}

impl WanLinear {
    pub(super) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            WanLinear::Dense(l) => l.forward(x),
            WanLinear::Quant(q) => q.forward(x),
        }
    }
}
