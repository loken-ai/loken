use super::*;
use crate::inference::media_facts::fixture::{flux, gguf, json, safetensors, Scratch};

/// A models store holding checkpoints of every image family, each declared only by what
/// the readers look at: tensor names, a pipeline index, a GGUF header. Drop-ins are sparse
/// files past the size floor, so the store costs no disk.
pub(crate) struct ImageStore {
    _scratch: Scratch,
    pub dir: String,
}

fn drop_in(root: &std::path::Path, rel: &str, names: &[&str]) -> std::path::PathBuf {
    let p = root.join(rel);
    safetensors(&p, names, None);
    std::fs::OpenOptions::new()
        .append(true)
        .open(&p)
        .unwrap()
        .set_len(SMALLEST_IMAGE_MODEL)
        .unwrap();
    p
}

fn cached(root: &std::path::Path, repo: &str) -> std::path::PathBuf {
    let dir = root
        .join("hub")
        .join(format!("models--{}", repo.replace('/', "--")));
    std::fs::create_dir_all(dir.join("refs")).unwrap();
    std::fs::write(dir.join("refs/main"), "r").unwrap();
    let snap = dir.join("snapshots/r");
    std::fs::create_dir_all(&snap).unwrap();
    snap
}

const ZIMAGE: &[&str] = &[
    "cap_embedder.0.weight",
    "context_refiner.0.attention.qkv.weight",
];
const QWEN: &[&str] = &["transformer_blocks.0.attn.to_q.weight", "img_in.weight"];
const SDXL: &[&str] = &["model.diffusion_model.input_blocks.1.0.in_layers.0.weight"];
const FLUX2: &[&str] = &[
    "single_transformer_blocks.0.attn.to_q.weight",
    "x_embedder.weight",
];
const BOOGU: &[&str] = &[
    "double_stream_layers.0.attn.qkv.weight",
    "instruct_proj.weight",
];

pub(crate) fn image_store(tag: &str) -> ImageStore {
    let scratch = Scratch::new(&format!("image-store-{tag}"));
    let root = scratch.0.clone();
    drop_in(&root, "fluxdev/Tune.v1.0.safetensors", &flux(true));
    drop_in(&root, "fluxpair/Pair.safetensors", &flux(true));
    drop_in(&root, "fluxpair/Pair.Other.safetensors", &flux(true));
    drop_in(&root, "editor/Editor.safetensors", &flux(true));
    json(
        &root.join("editor/model_index.json"),
        serde_json::json!({"_class_name": "FluxKontextPipeline"}),
    );
    drop_in(&root, "zimage-tune/Tune.safetensors", ZIMAGE);
    drop_in(&root, "qwen-tune/Tune.safetensors", QWEN);
    drop_in(&root, "qwen-distilled/Distilled.safetensors", QWEN);
    json(
        &root.join("qwen-distilled/model_index.json"),
        serde_json::json!({"_class_name": "QwenImagePipeline", "is_distilled": true}),
    );
    drop_in(&root, "sdxl-a/A.safetensors", SDXL);
    drop_in(&root, "sdxl-b/B.safetensors", SDXL);
    gguf(
        &cached(&root, "lmz/candle-flux").join("flux1-schnell.gguf"),
        &[],
        &flux(false),
    );
    safetensors(
        &cached(&root, "Tongyi-MAI/Z-Image-Turbo")
            .join("transformer/diffusion_pytorch_model-00001-of-00001.safetensors"),
        ZIMAGE,
        None,
    );
    let klein = cached(&root, "black-forest-labs/FLUX.2-klein-4B");
    json(
        &klein.join("model_index.json"),
        serde_json::json!({"_class_name": "Flux2KleinPipeline", "is_distilled": true}),
    );
    safetensors(
        &klein.join("transformer/diffusion_pytorch_model.safetensors"),
        FLUX2,
        None,
    );
    safetensors(
        &cached(&root, "realrebelai/Boogu-Image-Turbo").join("dit.safetensors"),
        BOOGU,
        None,
    );
    let text = cached(&root, "Owner/Text-Model");
    json(
        &text.join("config.json"),
        serde_json::json!({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
    );
    ImageStore {
        dir: root.to_string_lossy().to_string(),
        _scratch: scratch,
    }
}

/// Every id of the store, with the family its files declare.
const STORE_IDS: &[(&str, ImageFamily)] = &[
    ("fluxdev", ImageFamily::Flux),
    ("pair", ImageFamily::Flux),
    ("pair-other", ImageFamily::Flux),
    ("editor", ImageFamily::Flux),
    ("flux-schnell", ImageFamily::Flux),
    ("candle-flux", ImageFamily::Flux),
    ("lmz/candle-flux", ImageFamily::Flux),
    ("zimage-tune", ImageFamily::ZImage),
    ("z-image-turbo", ImageFamily::ZImage),
    ("Tongyi-MAI/Z-Image-Turbo", ImageFamily::ZImage),
    ("qwen-tune", ImageFamily::QwenImage),
    ("qwen-distilled", ImageFamily::QwenImage),
    ("sdxl-a", ImageFamily::Sdxl),
    ("sdxl-b", ImageFamily::Sdxl),
    ("flux.2-klein-4b", ImageFamily::Flux2),
    ("boogu-image-turbo", ImageFamily::Boogu),
];

/// Every id the store lists resolves to a checkpoint whose files declare the family, and
/// the drop-ins are listed under the names a request resolves them by.
#[test]
fn every_listed_id_resolves_to_its_own_checkpoint() {
    let store = image_store("ids");
    for (id, family) in STORE_IDS {
        assert_eq!(
            image_family(&store.dir, id),
            Some(*family),
            "'{id}' does not resolve to a {family:?} checkpoint"
        );
        assert!(is_image_gen_model(&store.dir, id), "{id}");
    }
    let mut listed: Vec<String> = drop_in_entries(&store.dir)
        .into_iter()
        .map(|e| e.id)
        .collect();
    listed.sort();
    assert_eq!(
        listed,
        [
            "editor",
            "fluxdev",
            "pair",
            "pair-other",
            "qwen-distilled",
            "qwen-tune",
            "sdxl-a",
            "sdxl-b",
            "zimage-tune"
        ]
    );
}

/// A checkpoint that declares a distillation does not inherit the family's guided recipe:
/// that would be 10x the work for a checkpoint trained to run in 4 steps without guidance.
#[test]
fn a_declared_distillation_changes_the_recipe() {
    let store = image_store("distilled");
    let distilled = image_model_defaults(&store.dir, "qwen-distilled").unwrap();
    assert_eq!(distilled.steps, 4);
    assert!((distilled.guidance - 1.0).abs() < f64::EPSILON);
    let base = image_model_defaults(&store.dir, "qwen-tune").unwrap();
    assert_eq!(base.steps, 20);
    assert!((base.guidance - 4.0).abs() < f64::EPSILON);
    assert_eq!((distilled.size, distilled.align), (base.size, base.align));
}

#[test]
fn is_hd_quality_accepts_dalle3_and_gpt_image_1_forms() {
    // dall-e-3 form
    assert!(is_hd_quality(Some("hd")));
    assert!(is_hd_quality(Some("HD")));
    assert!(is_hd_quality(Some(" Hd ")));
    // gpt-image-1 form
    assert!(is_hd_quality(Some("high")));
    assert!(is_hd_quality(Some("HIGH")));
    assert!(is_hd_quality(Some("auto")));
    // standard / lower-quality forms
    assert!(!is_hd_quality(Some("standard")));
    assert!(!is_hd_quality(Some("low")));
    assert!(!is_hd_quality(Some("medium")));
    // unknown
    assert!(!is_hd_quality(Some("ultra")));
    assert!(!is_hd_quality(None));
}

#[test]
fn validate_image_quality_accepts_documented_enum() {
    // dall-e-3
    assert!(validate_image_quality(Some("standard")).is_ok());
    assert!(validate_image_quality(Some("hd")).is_ok());
    // gpt-image-1
    assert!(validate_image_quality(Some("low")).is_ok());
    assert!(validate_image_quality(Some("medium")).is_ok());
    assert!(validate_image_quality(Some("high")).is_ok());
    assert!(validate_image_quality(Some("auto")).is_ok());
    // case + whitespace tolerance
    assert!(validate_image_quality(Some("HD")).is_ok());
    assert!(validate_image_quality(Some(" High ")).is_ok());
    // empty / unset
    assert!(validate_image_quality(Some("")).is_ok());
    assert!(validate_image_quality(None).is_ok());
    // unknown rejected
    assert!(validate_image_quality(Some("ultra")).is_err());
    assert!(validate_image_quality(Some("4k")).is_err());
}

#[test]
fn is_hd_quality_and_validate_image_quality_agree_on_full_enum() {
    // Drift guard: every value validate_image_quality accepts MUST
    // have a deterministic is_hd_quality classification. A future
    // addition (e.g. "ultra-hd") that updates one helper but not
    // the other would silently break HD-step doubling for the new
    // value. Pin the (accepted_value, is_hd) table here.
    let cases: &[(&str, bool)] = &[
        // dall-e-3 enum
        ("standard", false),
        ("hd", true),
        // gpt-image-1 enum
        ("low", false),
        ("medium", false),
        ("high", true),
        ("auto", true),
    ];
    for (value, expected_hd) in cases {
        assert!(
            validate_image_quality(Some(value)).is_ok(),
            "{value:?} should be accepted by validate_image_quality",
        );
        assert_eq!(
            is_hd_quality(Some(value)),
            *expected_hd,
            "{value:?} HD classification drift between validators",
        );
    }
}

#[test]
fn validate_image_output_format_accepts_known() {
    assert!(validate_image_output_format("").is_ok());
    assert!(validate_image_output_format("png").is_ok());
    assert!(validate_image_output_format("PNG").is_ok());
    assert!(validate_image_output_format("jpeg").is_ok());
    assert!(validate_image_output_format("jpg").is_ok());
    assert!(validate_image_output_format("webp").is_ok());
    assert!(validate_image_output_format(" png ").is_ok());
    // Unknown.
    assert!(validate_image_output_format("tiff").is_err());
    assert!(validate_image_output_format("gif").is_err());
}

// -- validate_image_response_format ------------
// The OpenAI docs define exactly two response_format values
// for /v1/images/*: `url` (default) and `b64_json`. The server
// always returns b64_json (privacy), but the input is still
// validated at the boundary so typos surface as 400s.

#[test]
fn validate_image_response_format_accepts_documented_values() {
    assert!(validate_image_response_format("url").is_ok());
    assert!(validate_image_response_format("b64_json").is_ok());
    // Case + surrounding whitespace tolerated (matches the
    // boundary cleanup the multipart path already does).
    assert!(validate_image_response_format("URL").is_ok());
    assert!(validate_image_response_format("B64_JSON").is_ok());
    assert!(validate_image_response_format(" url ").is_ok());
}

#[test]
fn validate_image_response_format_rejects_typos_and_legacy_aliases() {
    // Pin: arbitrary strings get rejected with a structured
    // error that names the bad input + the accepted set. A
    // silent default would mask client bugs and surface as
    // empty/incorrect responses downstream.
    for bad in ["xml", "json", "binary", "png", "data", ""] {
        let err = validate_image_response_format(bad).expect_err("must reject undocumented values");
        assert!(
            err.contains("not supported"),
            "error must mention rejection; got {err:?}"
        );
        assert!(
            err.contains("url") && err.contains("b64_json"),
            "error must name the accepted set so the client can fix the call; got {err:?}"
        );
    }
}

// -- validate_image_num_steps ------------
// Bound on caller-supplied num_steps. Without it a misconfigured
// (or hostile) client can pass num_steps=100000 and pin the GPU
// for minutes. Test the boundaries explicitly so a future refactor
// doesn't loosen the cap by accident.

// -- validate_image_input_size ------------
// A backstop against a runaway body, not a judgement about what a caller
// may send. Both endpoints share the helper so the limit cannot drift out
// of sync between them.

#[test]
fn validate_image_input_size_accepts_at_or_under_cap() {
    assert!(
        validate_image_input_size(0).is_ok(),
        "empty handled separately upstream"
    );
    assert!(validate_image_input_size(1024).is_ok(), "small image");
    assert!(
        validate_image_input_size(IMAGE_INPUT_MAX_BYTES).is_ok(),
        "exactly at the cap must accept"
    );
}

// -- validate_audio_input_size ------------
// Sibling to validate_image_input_size, covering
// /v1/audio/transcriptions + /translations and the /api/chat ASR path.

// -- format_seed_echo (wire-format contract) ------------
// The GUI parses `[seed: <u64>]` via parse_seed_prefix in
// chat_tab.rs to render a copyable + lockable seed badge. If
// the server-side format ever drifts (e.g. someone changes the
// brackets or drops the space) the GUI silently stops detecting
// image-gen seeds - the badge disappears, the lock UX breaks.
// Pin the exact byte sequence so a refactor-by-search-and-replace
// surfaces here.

#[test]
fn format_seed_echo_matches_gui_parser_contract() {
    assert_eq!(format_seed_echo(0), "[seed: 0]");
    assert_eq!(format_seed_echo(12345), "[seed: 12345]");
    assert_eq!(format_seed_echo(u64::MAX), "[seed: 18446744073709551615]");

    // The GUI looks for the literal `[seed:` prefix (no leading
    // whitespace, square bracket, lowercase keyword, colon).
    // Any drift on that prefix breaks parse_seed_prefix.
    let echoed = format_seed_echo(42);
    assert!(
        echoed.starts_with("[seed:"),
        "GUI parse_seed_prefix matches on this literal; got {echoed:?}"
    );
    assert!(
        echoed.ends_with("]"),
        "GUI parse_seed_prefix expects a closing ']'; got {echoed:?}"
    );
    // Single space after the colon is part of the contract - the
    // GUI's parser tolerates whitespace but the canonical form
    // is exactly one space.
    assert!(
        echoed.contains("[seed: "),
        "exactly one space after the colon - got {echoed:?}"
    );
}

#[test]
fn format_image_step_progress_matches_gui_parser_contract() {
    // GUI's parse_image_step_progress (chat_tab.rs) matches the
    // exact `Step <int>/<int>` literal - pin so a refactor that
    // tweaks the spacing or capitalisation surfaces here.
    assert_eq!(format_image_step_progress(0, 8), "Step 0/8");
    assert_eq!(format_image_step_progress(1, 30), "Step 1/30");
    assert_eq!(format_image_step_progress(50, 50), "Step 50/50");

    let line = format_image_step_progress(3, 20);
    assert!(
        line.starts_with("Step "),
        "GUI parser matches case-sensitive 'Step ' literal; got {line:?}"
    );
    assert!(line.contains("/"), "GUI parser splits on '/'; got {line:?}");
    // No newline / trailing whitespace - the chat-stream chunker
    // concatenates these into the response body and an extra
    // newline would corrupt the parser's substring match.
    assert!(
        !line.contains('\n') && !line.ends_with(' '),
        "no trailing whitespace; got {line:?}"
    );
}

#[test]
fn validate_image_input_size_rejects_over_cap_with_useful_error() {
    let err = validate_image_input_size(IMAGE_INPUT_MAX_BYTES + 1).expect_err("cap+1 must reject");
    assert!(
        err.contains("too large"),
        "error must mention rejection; got {err:?}"
    );
    // Error must surface both the violator (so the client can log
    // / report it) and the cap in human-readable form (so the
    // user knows what to compress to).
    assert!(
        err.contains(&(IMAGE_INPUT_MAX_BYTES + 1).to_string()),
        "error must surface the offending byte length; got {err:?}"
    );
    // Derived from the constant, so raising the cap does not need this line edited
    // and cannot leave the message quoting a figure that is no longer enforced.
    assert!(
        err.contains(&format!("{} MB", IMAGE_INPUT_MAX_BYTES / (1024 * 1024))),
        "error must surface the cap in MB; got {err:?}"
    );

    // Pathological case - well past any legitimate source image.
    assert!(validate_image_input_size(IMAGE_INPUT_MAX_BYTES * 8).is_err());
}

#[test]
fn validate_image_num_steps_accepts_reasonable_range() {
    assert!(
        validate_image_num_steps(1).is_ok(),
        "1 = minimum schedulable"
    );
    assert!(validate_image_num_steps(4).is_ok(), "Flux Schnell native");
    assert!(validate_image_num_steps(8).is_ok(), "Z-Image Turbo native");
    assert!(validate_image_num_steps(50).is_ok(), "GUI slider max");
    assert!(
        validate_image_num_steps(IMAGE_NUM_STEPS_HARD_CAP).is_ok(),
        "exactly at the cap must still accept"
    );
}

// -- validate_image_dimensions ------------
// Mirrors num_steps bounds: an unguarded 100000-100000 request
// allocates ~40 GB of latents and either OOMs or burns the GPU.
// /v1/images/* go through parse_image_size; /api/chat now calls
// validate_image_dimensions directly so both paths share the
// same cap.

#[test]
fn validate_image_dimensions_accepts_supported_sizes() {
    // The two server defaults + a few common image-gen sizes.
    assert!(
        validate_image_dimensions(512, 512).is_ok(),
        "Flux/Z-Image default"
    );
    assert!(validate_image_dimensions(1024, 1024).is_ok(), "Z-Image HD");
    assert!(
        validate_image_dimensions(768, 1024).is_ok(),
        "portrait orientation"
    );
    assert!(
        validate_image_dimensions(1024, 768).is_ok(),
        "landscape orientation"
    );
    assert!(
        validate_image_dimensions(IMAGE_MAX_DIM, IMAGE_MAX_DIM).is_ok(),
        "exactly at the cap must accept (Flux 2048² fits 12 GB)"
    );
}

#[test]
fn validate_image_dimensions_rejects_zero_misaligned_and_oversize() {
    // Zero dim - would emit a 0-area latent and crash.
    assert!(validate_image_dimensions(0, 512).is_err());
    assert!(validate_image_dimensions(512, 0).is_err());

    // Not a multiple of 16 - VAE stride mismatch.
    let err = validate_image_dimensions(513, 512).expect_err("must reject misaligned");
    assert!(
        err.contains("16"),
        "error must hint the alignment rule; got {err:?}"
    );

    // Above cap - DoS vector.
    let err = validate_image_dimensions(IMAGE_MAX_DIM + 16, 512).expect_err("oversize must reject");
    assert!(
        err.contains(&IMAGE_MAX_DIM.to_string()),
        "error must surface the cap; got {err:?}"
    );

    // Pathological (real DoS attempt).
    assert!(validate_image_dimensions(100_000, 100_000).is_err());
}

#[test]
fn parse_image_size_delegates_to_validate_image_dimensions() {
    // The /v1/images/* parser shares the same cap as the chat
    // path's direct validator - pin that the boundary errors
    // produced are the same (so clients see consistent messages
    // regardless of entry point).
    let parsed_err = parse_image_size("0x512").unwrap_err().to_string();
    let direct_err = validate_image_dimensions(0, 512).unwrap_err();
    assert!(
        parsed_err.contains(&direct_err),
        "parse_image_size should surface the same boundary error as validate_image_dimensions"
    );

    // Multiple of 16 but above cap - pin the size-cap branch
    // surfaces the shared IMAGE_MAX_DIM constant rather than a
    // re-typed magic number.
    let over = IMAGE_MAX_DIM + 16; // still a multiple of 16, so only the cap can reject it
    let oversize_err = parse_image_size(&format!("{over}x{over}"))
        .unwrap_err()
        .to_string();
    assert!(
        oversize_err.contains(&IMAGE_MAX_DIM.to_string()),
        "parse_image_size must use the shared IMAGE_MAX_DIM constant; got {oversize_err:?}"
    );
}

#[test]
fn validate_image_num_steps_rejects_zero_and_overflow() {
    // 0 timesteps = no denoising; engine would either crash or
    // return raw noise. Either way nonsense - reject early.
    let err = validate_image_num_steps(0).expect_err("0 must reject");
    assert!(
        err.contains(">= 1"),
        "error must hint the floor; got {err:?}"
    );

    // Above the cap: DoS vector if accepted. Error must surface
    // both the violator and the cap so the client can fix it.
    let err =
        validate_image_num_steps(IMAGE_NUM_STEPS_HARD_CAP + 1).expect_err("cap+1 must reject");
    assert!(
        err.contains(&IMAGE_NUM_STEPS_HARD_CAP.to_string()),
        "error must surface the cap; got {err:?}"
    );
    assert!(
        err.contains(&(IMAGE_NUM_STEPS_HARD_CAP + 1).to_string()),
        "error must surface the offending value; got {err:?}"
    );

    // Far above the cap: a real-world DoS attempt. Same rejection.
    assert!(validate_image_num_steps(100_000).is_err());
    assert!(validate_image_num_steps(u32::MAX).is_err());
}

#[test]
fn parse_image_size_accepts_dimensions() {
    assert_eq!(parse_image_size("512x512").unwrap(), (512, 512));
    assert_eq!(parse_image_size("1024X1024").unwrap(), (1024, 1024));
    assert_eq!(parse_image_size("512\u{00d7}768").unwrap(), (512, 768));
    // Non-square (portrait and landscape) - neither orientation
    // gets special-cased; both go through the same validator.
    assert_eq!(parse_image_size("1024x768").unwrap(), (1024, 768));
    assert_eq!(parse_image_size("768x1024").unwrap(), (768, 1024));
    // Exactly at the cap (multiple-of-16 AND below MAX_DIM+1) accepts.
    let max_size = format!("{IMAGE_MAX_DIM}x{IMAGE_MAX_DIM}");
    assert_eq!(
        parse_image_size(&max_size).unwrap(),
        (IMAGE_MAX_DIM, IMAGE_MAX_DIM)
    );
    // Minimum schedulable multiple-of-16 dimension accepts (Z-Image
    // patch_size=2, Flux patch_size=16 both happily encode 16- as
    // a one-token latent).
    assert_eq!(parse_image_size("16x16").unwrap(), (16, 16));
    // Not multiples of 16.
    assert!(parse_image_size("513x512").is_err());
    // Zero.
    assert!(parse_image_size("0x512").is_err());
    // Out of range - derived, so raising the bound does not silently turn this
    // assertion into a check that a legal size is rejected.
    let over = IMAGE_MAX_DIM + 16;
    assert!(parse_image_size(&format!("{over}x{over}")).is_err());
    // Trailing whitespace inside the dimensions tolerated by the
    // .trim() in parse_image_size.
    assert_eq!(parse_image_size(" 512 x 512 ").unwrap(), (512, 512));
}

//
// Builds one entry of the /v1/images/* `data` array. After the
// privacy fix (no on-disk image cache), every branch returns
// b64_json regardless of the caller's response_format request.
// Pin both the format passthrough/transcode logic AND the
// url---b64 contract so a future re-introduction of disk
// caching has to update this test deliberately.

#[test]
fn image_response_entry_b64_png_takes_fast_path_no_decode() {
    // b64_json + png + output_format=png (or empty) is the most
    // common request - skip the decode/re-encode round-trip.
    // Pin that the returned b64_json equals the input verbatim.
    let b64 = fake_png_b64();
    let entry = image_response_entry(b64.clone(), "png", 95, None);
    assert_eq!(entry["b64_json"].as_str(), Some(b64.as_str()));
    // No `url`, no `revised_prompt` on b64_json path.
    assert!(entry.get("url").is_none());
    assert!(entry.get("revised_prompt").is_none());
    // Empty output_format also routes to fast path.
    let entry2 = image_response_entry(b64.clone(), "", 95, None);
    assert_eq!(entry2["b64_json"].as_str(), Some(b64.as_str()));
}

#[test]
fn image_response_entry_b64_jpeg_re_encodes_via_transcode() {
    // b64_json + jpeg - transcode then re-base64. The output
    // bytes are a JPEG; pin that the result is base64-encoded
    // JPEG, not the original PNG.
    use base64::Engine as _;
    let b64 = fake_png_b64();
    let entry = image_response_entry(b64.clone(), "jpeg", 80, None);
    let out_b64 = entry["b64_json"].as_str().expect("b64_json present");
    assert_ne!(out_b64, b64.as_str(), "must be transcoded, not passthrough");
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(out_b64)
        .expect("valid base64");
    assert!(
        decoded.starts_with(&[0xFF, 0xD8]),
        "must be JPEG (SOI marker)"
    );
}

#[test]
fn image_response_entry_b64_webp_re_encodes_via_transcode() {
    // Same as jpeg but for webp - pin the magic.
    use base64::Engine as _;
    let b64 = fake_png_b64();
    let entry = image_response_entry(b64, "webp", 95, None);
    let out_b64 = entry["b64_json"].as_str().expect("b64_json present");
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(out_b64)
        .expect("valid base64");
    assert_eq!(&decoded[..4], b"RIFF");
    assert_eq!(&decoded[8..12], b"WEBP");
}

#[test]
fn image_response_entry_transcode_failure_falls_back_to_original_b64() {
    // Unsupported format triggers transcode_png_b64 to error.
    // image_response_entry catches it and falls back to
    // returning the original b64 - preferable to losing the
    // generation when only the OUTPUT format is bad.
    let b64 = fake_png_b64();
    let entry = image_response_entry(b64.clone(), "gif", 80, None);
    assert_eq!(
        entry["b64_json"].as_str(),
        Some(b64.as_str()),
        "transcode-fail must fall back to original b64"
    );
}

#[test]
fn image_response_entry_invalid_b64_input_falls_back_gracefully() {
    // If the engine somehow handed us garbage b64, the
    // transcode_png_b64 base64-decode fails, and we fall back to
    // returning that garbage verbatim. Bad UX but safer than
    // 500ing the whole request - the engine produced something
    // and we hand it back.
    let entry = image_response_entry("!!! not b64 !!!".to_string(), "jpeg", 80, None);
    assert_eq!(entry["b64_json"].as_str(), Some("!!! not b64 !!!"));
}

#[test]
fn image_response_headers_carry_server_timing_with_dur() {
    // Server-Timing: generate;dur=<ms> surfaces per-request
    // latency in browser DevTools without parsing the body.
    let h = image_response_headers(123.45);
    let st = h
        .get("server-timing")
        .and_then(|v| v.to_str().ok())
        .expect("server-timing header present");
    assert!(
        st.starts_with("generate;dur="),
        "server-timing must use the 'generate;dur=' format; got {st:?}"
    );
    assert!(
        st.contains("123.4") || st.contains("123.5"),
        "duration must be threaded through to one decimal place; got {st:?}"
    );
}

#[test]
fn image_response_entry_never_emits_url_or_revised_prompt() {
    // PRIVACY CONTRACT (structural): the helper does not accept a
    // `response_format` parameter at all - the legacy `url` mode
    // is gone. This test pins the OUTPUT side of the contract:
    // no matter what other inputs vary, the response only ever
    // carries `b64_json` (+ optional `seed`), never `url` or
    // `revised_prompt`. Reintroducing those fields would imply
    // a disk-persistence regression and must trip this test.
    let b64 = fake_png_b64();
    for entry in [
        image_response_entry(b64.clone(), "png", 95, None),
        image_response_entry(b64.clone(), "jpeg", 80, Some(7)),
        image_response_entry(b64.clone(), "webp", 95, None),
    ] {
        assert!(
            entry.get("url").is_none(),
            "url field would imply disk persistence; got entry={entry}"
        );
        assert!(
            entry.get("revised_prompt").is_none(),
            "revised_prompt was part of the url path; must be absent; got entry={entry}"
        );
        assert!(
            entry["b64_json"].is_string(),
            "every entry must surface bytes as b64_json; got entry={entry}"
        );
    }
}

#[test]
fn image_response_entry_echoes_seed_when_provided() {
    // Seed echo is the reproducibility contract - if the server
    // picked a random seed (caller didn't supply one), the
    // response must surface it so the client can re-roll.
    let b64 = fake_png_b64();
    let entry = image_response_entry(b64.clone(), "png", 95, Some(12345));
    assert_eq!(
        entry["seed"].as_u64(),
        Some(12345),
        "seed must be echoed verbatim when provided"
    );
}

#[test]
fn image_response_entry_omits_seed_key_when_none() {
    // If the caller explicitly opted out (passes None), the
    // response shouldn't manufacture a seed key - that would
    // confuse clients that don't expect it.
    let b64 = fake_png_b64();
    let entry = image_response_entry(b64, "png", 95, None);
    assert!(
        entry.get("seed").is_none(),
        "no seed param -> no seed key in response"
    );
}

#[test]
fn image_response_entry_echoes_seed_through_transcode_path() {
    // Both fast-path (PNG passthrough) and transcoded JPEG/WebP
    // branches must echo the seed. Pin the JPEG case so a future
    // refactor that re-orders the branches can't accidentally
    // drop seed on the non-PNG path.
    let b64 = fake_png_b64();
    let entry = image_response_entry(b64, "jpeg", 80, Some(99));
    assert_eq!(entry["seed"].as_u64(), Some(99));
    assert!(entry["b64_json"].is_string());
}

// -- image_model_defaults --
//
// Defaults table for image-gen knobs (steps / guidance / size), keyed on the family the
// checkpoint's files declare and completed by what else they declare.

/// REFLEXIVITY: whatever id a model is loaded under, a later request for that same id
/// must reuse it, or every request unloads and reloads the model it already has.
#[test]
fn every_listed_id_serves_itself() {
    let store = image_store("reflexive");
    for (id, _) in STORE_IDS {
        assert!(
            resident_serves_request(&store.dir, Some(id), None, id, None),
            "a resident loaded as '{id}' does not serve a request for '{id}'"
        );
    }
}

/// EVERY family reaches a loader of its OWN: the loader is the family the checkpoint's
/// files declare, and every family is exercised by a checkpoint of the store.
#[test]
fn every_family_routes_to_its_own_loader() {
    let store = image_store("loaders");
    let mut covered = Vec::new();
    for (id, family) in STORE_IDS {
        assert_eq!(
            image_loader(&store.dir, id),
            Ok(*family),
            "'{id}' does not route to its own loader"
        );
        covered.push(*family);
    }
    for family in ImageFamily::ALL {
        assert!(
            covered.contains(&family),
            "no checkpoint of the store exercises the {family:?} loader"
        );
    }
}

/// An SDXL checkpoint is never handed to the FLUX loader, whatever it is called.
#[test]
fn sdxl_never_routes_to_the_flux_loader() {
    let store = image_store("sdxl");
    for id in ["sdxl-a", "sdxl-b"] {
        assert_eq!(image_loader(&store.dir, id), Ok(ImageFamily::Sdxl), "{id}");
    }
}

/// A representative id per family, for the budget and recipe tests below.
const FAMILY_TAGS: &[(ImageFamily, &str)] = &[
    (ImageFamily::Flux, "flux-schnell"),
    (ImageFamily::Flux2, "flux.2-klein-4b"),
    (ImageFamily::ZImage, "z-image-turbo"),
    (ImageFamily::QwenImage, "qwen-tune"),
    (ImageFamily::Boogu, "boogu-image-turbo"),
    (ImageFamily::Sdxl, "sdxl-a"),
];

/// An empty directory, so nothing resolves to a checkpoint on disk and every reserve
/// is the architecture-derived figure - pure arithmetic, identical on any machine.
struct TmpDir(std::path::PathBuf);
impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn empty_models_dir(tag: &str) -> TmpDir {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let root = std::env::temp_dir()
        .join("loken-family-budget-tests")
        .join(format!("{tag}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    TmpDir(root)
}

/// The reserve each family's OWN architecture gives, written out here as an oracle.
///
/// A deliberate second copy of `family_runtime_bytes`. It is exhaustive, so a family
/// added to the enum without a budget fails to compile HERE as well as in the table.
fn oracle_runtime_bytes(
    dir: &str,
    loader: ImageFamily,
    model_name: &str,
    geom: crate::inference::place::runtime_demand::RequestGeometry,
) -> u64 {
    use crate::inference::place::runtime_demand as demand;
    const VAE_STRIDE: usize = 8;
    const VAE_DECODER_WIDTH: usize = 128;
    let vae = demand::vae_decode_bytes(geom.height, geom.width, VAE_DECODER_WIDTH);
    match loader {
        ImageFamily::Flux2 => crate::inference::engine::flux2_engine::runtime_headroom_bytes(
            dir,
            geom.width,
            geom.height,
        ),
        ImageFamily::QwenImage => {
            crate::inference::engine::qwen_image_engine::runtime_headroom_bytes(
                dir,
                requested_checkpoint(dir, model_name).as_deref(),
                geom.width,
                geom.height,
                geom.frames,
            )
        }
        ImageFamily::ZImage => crate::inference::engine::image_engine::zimage_runtime_demand_for(
            dir,
            requested_checkpoint(dir, model_name).as_deref(),
            geom.width,
            geom.height,
        ),
        ImageFamily::Boogu => {
            crate::inference::engine::boogu_engine::runtime_headroom_bytes(geom.width, geom.height)
        }
        ImageFamily::Sdxl => {
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
        ImageFamily::Flux => crate::inference::engine::image_engine::flux_runtime_demand_for(
            geom.width,
            geom.height,
            requested_checkpoint(dir, model_name).as_deref(),
        )
        .max(vae),
    }
}

/// Value by value, every family still gets the reserve its own architecture gives - at
/// several geometries, because the figure scales with the request and a family wired to
/// the wrong formula can agree at one size by accident.
#[test]
fn every_family_reserves_what_its_own_architecture_asks_for() {
    use crate::inference::place::runtime_demand::RequestGeometry;
    let tmp = empty_models_dir("reserve");
    let dir = tmp.0.to_string_lossy().to_string();
    let mut covered = Vec::new();
    for (loader, tag) in FAMILY_TAGS {
        for (w, h) in [(512, 512), (1024, 1024), (1024, 1536)] {
            let geom = RequestGeometry::new(w, h);
            let got = family_runtime_bytes(&dir, *loader, tag, geom);
            assert_eq!(
                got,
                oracle_runtime_bytes(&dir, *loader, tag, geom),
                "family {loader:?} at {w}x{h} no longer reserves what its own \
                 architecture asks for"
            );
            // A reserve of zero admits any placement, which is the same failure as
            // inheriting another family's: nothing holds the card for the render.
            assert!(got > 0, "family {loader:?} at {w}x{h} reserves nothing");
        }
        covered.push(*loader);
    }
    for family in ImageFamily::ALL {
        assert!(
            covered.contains(&family),
            "family {family:?} has no budget of its own exercised here"
        );
    }
}

/// ONE reserve per family, not one per call site: the admission gate asks the loader's
/// own helper, so the two cannot disagree.
#[test]
fn the_zimage_reserve_is_the_figure_its_loader_plans_against() {
    use crate::inference::place::runtime_demand::RequestGeometry;
    let tmp = empty_models_dir("zimage-reserve");
    let dir = tmp.0.to_string_lossy().to_string();
    for (w, h) in [
        (512, 512),
        (768, 768),
        (1024, 1024),
        (1024, 1536),
        (1536, 1536),
    ] {
        let geom = RequestGeometry::new(w, h);
        assert_eq!(
            family_runtime_bytes(&dir, ImageFamily::ZImage, "z-image-turbo", geom),
            crate::inference::engine::image_engine::zimage_runtime_demand_for(&dir, None, w, h),
            "the Z-Image admission gate and its placement disagree at {w}x{h}"
        );
    }
}

/// With ONLY a FLUX checkpoint on disk, a family that has no weights there reports none:
/// a family carrying another family's weights reserves a card for a model it is not.
#[test]
fn no_family_reports_another_familys_checkpoint_as_its_weights() {
    let tmp = empty_models_dir("weights");
    let dir = tmp.0.to_string_lossy().to_string();
    let flux_file = drop_in(&tmp.0, "fluxdev/Tune.safetensors", &flux(true));
    let flux_len = std::fs::metadata(&flux_file).unwrap().len();
    assert_eq!(
        family_hot_bytes(&dir, ImageFamily::Flux, "fluxdev"),
        flux_len,
        "the FLUX budget must read the FLUX checkpoint"
    );
    for (loader, tag) in FAMILY_TAGS {
        if *loader == ImageFamily::Flux {
            continue;
        }
        assert_eq!(
            family_hot_bytes(&dir, *loader, tag),
            0,
            "family {loader:?} claims the FLUX checkpoint as its own hot component"
        );
    }
}

/// A name whose files declare no image model FAILS, naming itself, rather than borrowing
/// some family's loader and rendering another model under its name.
#[test]
fn a_name_whose_files_declare_no_image_model_is_refused_by_name() {
    let store = image_store("unknown");
    for name in ["stable-cascade", "Owner/Text-Model", "text-model", ""] {
        let err = image_loader(&store.dir, name)
            .expect_err("a checkpoint that is no image model must not resolve to a loader");
        assert!(
            err.contains(name),
            "the error must name the model '{name}': {err}"
        );
        assert!(image_model_defaults(&store.dir, name).is_err(), "{name}");
        assert!(!is_image_gen_model(&store.dir, name), "{name}");
    }
}

/// The recipe each family was validated at, written out here as an oracle: a deliberate
/// second copy of `image_defaults_for`. Exhaustive on [`ImageFamily`], so a family added
/// tomorrow has to state its recipe HERE as well as in the table.
fn oracle_defaults(
    family: ImageFamily,
    facts: &crate::inference::media_facts::MediaFacts,
) -> (usize, f64, usize, usize) {
    match family {
        ImageFamily::ZImage => (9, 5.0, 1024, 16),
        ImageFamily::QwenImage if facts.distilled => (4, 1.0, 1024, 16),
        ImageFamily::QwenImage => (20, 4.0, 1024, 16),
        ImageFamily::Flux2 => (4, 1.0, 1024, 16),
        ImageFamily::Boogu => (4, 1.0, 1024, 16),
        ImageFamily::Sdxl => (25, 7.0, 1024, 64),
        ImageFamily::Flux if facts.instruction_edit => (28, 2.5, 1024, 16),
        ImageFamily::Flux if facts.guidance_embedded => (20, 3.5, 1024, 16),
        ImageFamily::Flux => (4, 4.0, 512, 16),
    }
}

/// Value by value, every checkpoint of the store renders at its OWN recipe: the guidance
/// embedding, the distillation and the instruction editor each move a FLUX.1 or a
/// Qwen-Image checkpoint off its family's base recipe.
#[test]
fn every_family_renders_at_its_own_recipe() {
    let store = image_store("recipes");
    for (id, family) in STORE_IDS {
        let facts = media_facts(&store.dir, id).unwrap();
        let got = image_model_defaults(&store.dir, id)
            .unwrap_or_else(|e| panic!("'{id}' has no recipe of its own: {e}"));
        assert_eq!(
            (got.steps, got.guidance, got.size, got.align),
            oracle_defaults(*family, &facts),
            "'{id}' ({family:?}) renders at another recipe"
        );
    }
    let recipe = |id: &str| {
        let d = image_model_defaults(&store.dir, id).unwrap();
        (d.steps, d.guidance, d.size)
    };
    assert_eq!(recipe("flux-schnell"), (4, 4.0, 512));
    assert_eq!(recipe("fluxdev"), (20, 3.5, 1024));
    assert_eq!(recipe("editor"), (28, 2.5, 1024));
    assert!(is_flux_instruction_editor(&store.dir, "editor"));
    assert!(!is_flux_instruction_editor(&store.dir, "fluxdev"));
}

/// The recipes do not collapse onto one another: a family that reads back as 4 steps at
/// guidance 4.0 and 512 pixels without being a FLUX.1 without guidance embedding has
/// inherited its recipe.
#[test]
fn no_family_inherits_the_flux_schnell_recipe() {
    let store = image_store("schnell");
    let schnell = image_model_defaults(&store.dir, "flux-schnell").unwrap();
    assert_eq!(
        (schnell.steps, schnell.guidance, schnell.size),
        (4, 4.0, 512)
    );
    for (family, tag) in FAMILY_TAGS {
        if *family == ImageFamily::Flux {
            continue;
        }
        let d = image_model_defaults(&store.dir, tag).unwrap();
        assert_ne!(
            (d.steps, d.guidance, d.size),
            (schnell.steps, schnell.guidance, schnell.size),
            "family {family:?} renders at FLUX Schnell's recipe"
        );
    }
}

/// THE CHECKPOINT RESOLVER: a family resolves only a checkpoint its own files declare.
#[test]
fn no_family_resolves_another_familys_checkpoint() {
    let store = image_store("ckpt");
    let dir = &store.dir;
    let schnell = requested_checkpoint(dir, "flux-schnell").expect("the FLUX GGUF");
    assert_eq!(
        schnell.file_name().and_then(|n| n.to_str()),
        Some("flux1-schnell.gguf")
    );
    assert!(requested_checkpoint(dir, "zimage-tune")
        .is_some_and(|p| p.ends_with("zimage-tune/Tune.safetensors")));
    // The family's own repository is loaded from its directory, by its own loader.
    assert_eq!(requested_checkpoint(dir, "z-image-turbo"), None);
    // These two families read their weights from a directory, never a single file.
    assert_eq!(requested_checkpoint(dir, "flux.2-klein-4b"), None);
    assert_eq!(requested_checkpoint(dir, "boogu-image-turbo"), None);
    assert_eq!(requested_checkpoint(dir, "Owner/Text-Model"), None);
}

/// A fine-tune is never served from a request for its base model, or the reverse, and
/// two checkpoints of one family are two models.
#[test]
fn a_finetune_never_aliases_its_base_model() {
    let store = image_store("alias");
    let d = store.dir.as_str();
    assert!(!resident_serves_request(
        d,
        Some("zimage-tune"),
        None,
        "z-image-turbo",
        None
    ));
    assert!(!resident_serves_request(
        d,
        Some("z-image-turbo"),
        None,
        "zimage-tune",
        None
    ));
    assert!(!resident_serves_request(
        d,
        Some("qwen-distilled"),
        None,
        "qwen-tune",
        None
    ));
    // Different families never match either.
    assert!(!resident_serves_request(
        d,
        Some("boogu-image-turbo"),
        None,
        "flux-schnell",
        None
    ));
    // Two files of the same family are distinct when both resolve locally.
    assert!(!resident_serves_request(
        d,
        Some("pair"),
        Some("A.safetensors"),
        "pair-other",
        Some("B.safetensors")
    ));
    // Two SDXL checkpoints with no resolvable file must not alias either.
    assert!(!resident_serves_request(
        d,
        Some("sdxl-a"),
        None,
        "sdxl-b",
        None
    ));
    assert!(!resident_serves_request(
        d,
        Some("sdxl-b"),
        None,
        "sdxl-a",
        None
    ));
    // The same id still serves itself.
    assert!(resident_serves_request(
        d,
        Some("sdxl-b"),
        None,
        "sdxl-b",
        None
    ));
    assert!(resident_serves_request(
        d,
        Some("pair"),
        Some("A.safetensors"),
        "pair",
        Some("A.safetensors")
    ));
}

/// SOURCE GATE: a loader that writes a literal name into its resident state
/// breaks reflexivity for every variant tag routed through it. The name must come from
/// the request.
#[test]
fn no_loader_hardcodes_its_resident_name() {
    let src = include_str!("../../../inference/engine/image_engine/mod.rs");
    let mut offenders = Vec::new();
    for (i, line) in src.lines().enumerate() {
        let t = line.trim();
        // `name: "..."` inside a resident-state construction.
        if t.starts_with("name:") && t.contains('"') && !t.contains("PLACEMENT-EXEMPT") {
            offenders.push(format!("image_engine.rs:{}: {t}", i + 1));
        }
    }
    assert!(
        offenders.is_empty(),
        "a loader hardcodes its resident name instead of recording the requested \
         one, so a variant tag cannot recognise itself:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn image_model_defaults_z_image_uses_1024_at_9_steps() {
    // Z-Image-Turbo's flow-matching scheduler converges in 9 steps at its native 1024
    // training resolution.
    let store = image_store("zimage-defaults");
    let d = image_model_defaults(&store.dir, "z-image-turbo").unwrap();
    assert_eq!(d.size, 1024);
    assert_eq!(d.steps, 9);
    assert_eq!(d.guidance, 5.0);
    let f = image_model_defaults(&store.dir, "flux-schnell").unwrap();
    assert_ne!(d.size, f.size);
    assert_ne!(d.steps, f.steps);
    assert_ne!(d.guidance, f.guidance);
}

// -- transcode_png_b64 --
//
// /v1/images/* endpoints accept output_format=png|jpeg|webp. The
// png path is a passthrough (avoid re-encoding a PNG we just
// generated); jpeg/webp re-encode through the image crate. A
// refactor that lost the passthrough would re-encode every PNG
// generation on the response path (wasted CPU + slight quality
// loss). A refactor that mis-mapped the format - ext string
// would break clients caching the response by file extension.

fn fake_png_b64() -> String {
    // Minimal 2-2 RGB PNG, encoded inline to avoid a fixture file.
    use base64::Engine as _;
    let img =
        image::ImageBuffer::<image::Rgb<u8>, _>::from_fn(2, 2, |_, _| image::Rgb([200u8, 100, 50]));
    let mut png = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("encode");
    base64::engine::general_purpose::STANDARD.encode(&png)
}

#[test]
fn transcode_png_b64_png_passthrough_returns_input_bytes_verbatim() {
    // Same-format request: return decoded bytes unchanged, no
    // re-encode pass. Pin this so a "simplify" refactor doesn't
    // accidentally re-encode the PNG (CPU waste + quality loss).
    use base64::Engine as _;
    let b64 = fake_png_b64();
    let original = base64::engine::general_purpose::STANDARD
        .decode(&b64)
        .unwrap();
    let (bytes, ext) = transcode_png_b64(&b64, "png", 95).unwrap();
    assert_eq!(ext, "png");
    assert_eq!(
        bytes, original,
        "PNG passthrough must return input verbatim"
    );
    // Empty format string also routes to passthrough.
    let (bytes2, ext2) = transcode_png_b64(&b64, "", 95).unwrap();
    assert_eq!(ext2, "png");
    assert_eq!(bytes2, original);
}

#[test]
fn transcode_png_b64_to_jpeg_encodes_and_uses_jpg_ext() {
    // JPEG branch: re-encode RGB at the given quality. Output
    // bytes must start with the JPEG SOI marker (0xFF 0xD8).
    // ext="jpg" (not "jpeg") - pin so a future swap doesn't
    // break clients that build URLs by extension.
    let b64 = fake_png_b64();
    let (bytes, ext) = transcode_png_b64(&b64, "jpeg", 80).unwrap();
    assert_eq!(ext, "jpg", "ext for jpeg request must be 'jpg'");
    assert!(bytes.starts_with(&[0xFF, 0xD8]), "missing JPEG SOI marker");
    // Both "jpeg" and "jpg" route to the same encoder.
    let (b2, ext2) = transcode_png_b64(&b64, "jpg", 80).unwrap();
    assert_eq!(ext2, "jpg");
    assert!(b2.starts_with(&[0xFF, 0xD8]));
}

#[test]
fn transcode_png_b64_to_webp_uses_webp_ext_and_magic() {
    // WebP: starts with "RIFF....WEBP" magic. ext="webp".
    let b64 = fake_png_b64();
    let (bytes, ext) = transcode_png_b64(&b64, "webp", 95).unwrap();
    assert_eq!(ext, "webp");
    assert!(bytes.len() >= 12, "too short to be WebP");
    assert_eq!(&bytes[..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WEBP");
}

#[test]
fn transcode_png_b64_quality_clamps_to_1_to_100() {
    // Quality 0 / 200 must not panic the JPEG encoder. The clamp
    // is documented; pin so a refactor that dropped it can't
    // crash the request path.
    let b64 = fake_png_b64();
    assert!(transcode_png_b64(&b64, "jpeg", 0).is_ok());
    assert!(transcode_png_b64(&b64, "jpeg", 200).is_ok());
    // Negative not representable (u8) - but cap at 100 in test.
    assert!(transcode_png_b64(&b64, "jpeg", 100).is_ok());
}

#[test]
fn transcode_png_b64_rejects_unsupported_format() {
    // gif/bmp/avif aren't in the supported set - must error
    // (not silently fall through to png passthrough or jpeg).
    let b64 = fake_png_b64();
    for fmt in ["gif", "bmp", "avif", "tiff", "heic"] {
        let err = transcode_png_b64(&b64, fmt, 80).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains(fmt),
            "{fmt}: error must name the format; got: {msg}"
        );
        assert!(
            msg.contains("not supported") || msg.contains("png"),
            "{fmt}: error must guide caller to supported set; got: {msg}"
        );
    }
}

#[test]
fn transcode_png_b64_rejects_invalid_base64() {
    let err = transcode_png_b64("!!! not b64 !!!", "jpeg", 80).unwrap_err();
    assert!(err.to_string().contains("base64"), "got: {err}");
}

#[test]
fn transcode_png_b64_rejects_non_png_bytes() {
    // Valid base64 but not a PNG - decode error from the image
    // crate's PNG-only loader.
    use base64::Engine as _;
    let not_png = base64::engine::general_purpose::STANDARD.encode(b"hello world");
    let err = transcode_png_b64(&not_png, "jpeg", 80).unwrap_err();
    assert!(
        err.to_string().contains("decode") || err.to_string().contains("png"),
        "got: {err}"
    );
}

#[test]
fn pick_largest_gguf_under_walks_recursively() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    // Tiny ad-hoc tmp dir cleaned up on Drop (same pattern as the
    // huggingface_manager tests - avoids a tempfile dep).
    struct Tmp(std::path::PathBuf);
    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let root = std::env::temp_dir()
        .join("loken-flux-pick-tests")
        .join(format!("{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let guard = Tmp(root.clone());

    // Three GGUFs at increasing depth, sizes 100 / 200 / 50 bytes.
    std::fs::write(guard.0.join("a.gguf"), vec![0u8; 100]).unwrap();
    std::fs::create_dir_all(guard.0.join("BF16")).unwrap();
    std::fs::write(guard.0.join("BF16/b.gguf"), vec![0u8; 200]).unwrap();
    std::fs::write(guard.0.join("c.txt"), b"not a gguf").unwrap();
    std::fs::create_dir_all(guard.0.join("deep/nest")).unwrap();
    std::fs::write(guard.0.join("deep/nest/d.gguf"), vec![0u8; 50]).unwrap();

    let pick = pick_largest_gguf_under(&guard.0).expect("must find a gguf");
    assert!(pick.ends_with("BF16/b.gguf"), "got: {pick:?}");

    // Empty dir - None.
    let empty = guard.0.join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    assert!(pick_largest_gguf_under(&empty).is_none());

    // Non-existent - None (no panic).
    assert!(pick_largest_gguf_under(&guard.0.join("nope")).is_none());
}

/// A parameter the resident family refuses answers 400; any other engine failure is not
/// a refusal and keeps the server status.
#[test]
fn a_refused_parameter_answers_bad_request() {
    use axum::response::IntoResponse;
    let refusal: Box<dyn std::error::Error + Send + Sync> = Box::new(
        crate::inference::engine::image_engine::UnsupportedParameter {
            family: "zimage",
            parameter: "mask",
        },
    );
    let answer = refused_parameter(refusal.as_ref()).expect("a refusal");
    assert_eq!(
        answer.into_response().status(),
        axum::http::StatusCode::BAD_REQUEST
    );
    let failure: Box<dyn std::error::Error + Send + Sync> = "render failed".into();
    assert!(refused_parameter(failure.as_ref()).is_none());
}
