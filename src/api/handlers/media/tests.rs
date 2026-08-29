use super::*;

/// The tag a checkpoint is ADVERTISED under and the one the resolver LOOKS UP are
/// the same rule, so a model can always be fetched by the name the catalogue shows.
/// They were two copies of the rule before, which is only harmless while they agree.
#[test]
fn catalogue_tag_is_the_name_a_request_resolves_by() {
    // A checkpoint whose stem names an image family keeps that name...
    assert_eq!(
        local_checkpoint_tag("rayqwest", "RayQwest.v1.0"),
        "rayqwest"
    );
    assert_eq!(
        local_checkpoint_tag("qwen-image-flash", "Qwen-Image-Flash"),
        "qwen-image-flash"
    );
    // ...and anything else is named after the directory it was dropped into, which
    // is what a shard-numbered filename would otherwise destroy.
    assert_eq!(
        local_checkpoint_tag("qwen-image-flash", "model-00001-of-00001"),
        "qwen-image-flash"
    );
    // Every advertised tag must route to a family that can load it, or the
    // catalogue offers a model the engine refuses.
    for tag in ["rayqwest", "qwen-image-flash"] {
        assert!(is_image_gen_model(tag), "{tag} is not routed as image-gen");
        assert_eq!(image_family(tag), "qwen-image");
    }
}

/// The distilled drop-in must not inherit the family's 20-step CFG recipe: that
/// would be 10x the work for a checkpoint trained to run in 4 steps without
/// guidance, and the guided sampler would push it off its own manifold.
#[test]
fn step_distilled_defaults_differ_from_the_family_recipe() {
    let flash = image_model_defaults("qwen-image-flash").unwrap();
    assert_eq!(flash.steps, 4);
    assert!((flash.guidance - 1.0).abs() < f64::EPSILON);
    let base = image_model_defaults("qwen-image").unwrap();
    assert_eq!(base.steps, 20);
    assert!((base.guidance - 4.0).abs() < f64::EPSILON);
    // Both are 1024-native and share the family's latent alignment.
    assert_eq!((flash.size, flash.align), (base.size, base.align));
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
// Defaults table for image-gen knobs (steps / guidance / size)
// routed by model family. Drift across the 4 endpoints that
// used to duplicate this if/else table was the real risk: a
// bump to Z-Image's default step count would silently miss
// /v1/images/edits if the developer only updated 3/4 sites.

/// REFLEXIVITY: whatever tag a model is loaded under, a later request for that
/// same tag must reuse it. This failed for every Ray variant and cost a full
/// unload + reload per render; the loaders now record the requested name, and
/// `no_loader_hardcodes_its_resident_name` keeps them honest.
#[test]
fn every_advertised_tag_serves_itself() {
    for tag in [
        "rayzist",
        "rayqwest",
        "rayflux",
        "rayflux-krea",
        "rayflux-horndog",
        "raymnants",
        "rayctifier",
        "rayburn",
        "z-image-turbo",
        "qwen-image",
        "qwen-image-edit",
        "boogu",
        "flux-schnell",
        "FLUX.1-schnell",
    ] {
        assert!(
            resident_serves_request(Some(tag), None, tag, None),
            "a resident loaded as '{tag}' does not serve a request for '{tag}'"
        );
    }
}

/// EVERY family the classifier can name must reach a loader of its OWN.
///
/// The dispatch was an inline `match` on the family string, written once per call
/// site and ending in a `_ =>` arm that ran the FLUX loader. The SSE branch of
/// /api/generate never grew an "sdxl" arm, so an SDXL request took that default:
/// FLUX.1-schnell was fetched and rendered with, under the SDXL name and at SDXL's
/// 25 steps / guidance 7. Nothing failed - the user simply got another model's
/// picture. This walks the advertised tags and the family list together, so a family
/// added to the classifier without a loader fails HERE.
#[test]
fn every_advertised_family_routes_to_its_own_loader() {
    let mut covered: Vec<&str> = Vec::new();
    for (tag, family, loader) in [
        ("flux-schnell", "flux", ImageLoader::Flux),
        ("FLUX.1-schnell", "flux", ImageLoader::Flux),
        ("flux-kontext", "flux", ImageLoader::Flux),
        ("rayflux", "flux", ImageLoader::Flux),
        // These three matter most: every one of them CONTAINS "flux", so the FLUX.1 arm
        // would claim them if the classifier tested it first.
        ("flux2-klein-4b", "flux2", ImageLoader::Flux2),
        ("FLUX.2-klein-9B", "flux2", ImageLoader::Flux2),
        ("klein", "flux2", ImageLoader::Flux2),
        ("z-image-turbo", "zimage", ImageLoader::ZImage),
        ("rayzist", "zimage", ImageLoader::ZImage),
        ("qwen-image", "qwen-image", ImageLoader::QwenImage),
        ("qwen-image-edit", "qwen-image", ImageLoader::QwenImage),
        ("rayqwest", "qwen-image", ImageLoader::QwenImage),
        ("boogu", "boogu", ImageLoader::Boogu),
        ("sdxl", "sdxl", ImageLoader::Sdxl),
        ("raymnants", "sdxl", ImageLoader::Sdxl),
        ("rayctifier", "sdxl", ImageLoader::Sdxl),
        ("rayburn", "sdxl", ImageLoader::Sdxl),
    ] {
        assert_eq!(
            image_family(tag),
            family,
            "'{tag}' is classified as another family"
        );
        assert!(
            IMAGE_FAMILIES.contains(&family),
            "family '{family}' is served but missing from IMAGE_FAMILIES"
        );
        assert_eq!(
            image_family_loader(family),
            Ok(loader),
            "'{tag}' (family {family}) does not route to its own loader"
        );
        if !covered.contains(&family) {
            covered.push(family);
        }
    }
    for family in IMAGE_FAMILIES {
        assert!(
            covered.contains(family),
            "family '{family}' is advertised but no tag above exercises its routing"
        );
        assert!(
            image_family_loader(family).is_ok(),
            "family '{family}' is advertised with no loader wired"
        );
    }
}

/// The defect itself: an SDXL checkpoint must never be handed to the Flux loader.
#[test]
fn sdxl_never_routes_to_the_flux_loader() {
    for tag in [
        "sdxl",
        "sdxl-base-1.0",
        "raymnants",
        "rayctifier",
        "rayburn",
    ] {
        let loader = image_family_loader(image_family(tag))
            .unwrap_or_else(|e| panic!("'{tag}' has no loader: {e}"));
        assert_eq!(loader, ImageLoader::Sdxl, "'{tag}' routes to {loader:?}");
    }
}

/// A representative model tag per advertised family, for the budget tests below.
const FAMILY_TAGS: &[(&str, &str)] = &[
    ("flux", "flux-schnell"),
    ("flux2", "flux2-klein-4b"),
    ("zimage", "z-image-turbo"),
    ("qwen-image", "qwen-image"),
    ("boogu", "boogu"),
    ("sdxl", "sdxl"),
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
/// It is a deliberate second copy of `family_runtime_bytes`, transcribed from the arms
/// as they stood when that table was keyed on the family STRING: equality proves the
/// re-keying moved no family onto another family's figure. It is also exhaustive, so a
/// family added to the loader enum without a budget fails to compile HERE as well as in
/// the table - which is the point, since a reserve is what stops the planner from
/// placing anything else on the card, and inheriting one silently is an OOM or a
/// placement on the host that nothing explains.
fn oracle_runtime_bytes(
    dir: &str,
    loader: ImageLoader,
    model_name: &str,
    geom: crate::inference::place::runtime_demand::RequestGeometry,
) -> u64 {
    use crate::inference::place::runtime_demand as demand;
    const VAE_STRIDE: usize = 8;
    const VAE_DECODER_WIDTH: usize = 128;
    let vae = demand::vae_decode_bytes(geom.height, geom.width, VAE_DECODER_WIDTH);
    match loader {
        ImageLoader::Flux2 => {
            crate::inference::engine::flux2_engine::runtime_headroom_bytes(geom.width, geom.height)
        }
        ImageLoader::QwenImage => {
            crate::inference::engine::qwen_image_engine::runtime_headroom_bytes(
                dir,
                requested_checkpoint(dir, model_name).as_deref(),
                geom.width,
                geom.height,
            )
        }
        ImageLoader::ZImage => crate::inference::engine::image_engine::zimage_runtime_demand_for(
            dir,
            find_ray_checkpoint(dir, "rayzist", model_name).as_deref(),
            geom.width,
            geom.height,
        ),
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
        ImageLoader::Flux => crate::inference::engine::image_engine::flux_runtime_demand_for(
            geom.width,
            geom.height,
            find_local_flux_gguf(dir, model_name).as_deref(),
        )
        .max(vae),
    }
}

/// Value by value, every advertised family still gets the reserve its own
/// architecture gives - at several geometries, because the figure scales with the
/// request and a family wired to the wrong formula can agree at one size by accident.
#[test]
fn every_family_reserves_what_its_own_architecture_asks_for() {
    use crate::inference::place::runtime_demand::RequestGeometry;
    let tmp = empty_models_dir("reserve");
    let dir = tmp.0.to_string_lossy().to_string();
    let mut covered: Vec<&str> = Vec::new();
    for (family, tag) in FAMILY_TAGS {
        let loader = image_family_loader(family)
            .unwrap_or_else(|e| panic!("family '{family}' has no loader: {e}"));
        assert_eq!(
            image_family(tag),
            *family,
            "'{tag}' is classified as another family"
        );
        for (w, h) in [(512, 512), (1024, 1024), (1024, 1536)] {
            let geom = RequestGeometry::new(w, h);
            let got = family_runtime_bytes(&dir, loader, tag, geom);
            assert_eq!(
                got,
                oracle_runtime_bytes(&dir, loader, tag, geom),
                "family '{family}' at {w}x{h} no longer reserves what its own \
                 architecture asks for"
            );
            // A reserve of zero admits any placement, which is the same failure as
            // inheriting another family's: nothing holds the card for the render.
            assert!(got > 0, "family '{family}' at {w}x{h} reserves nothing");
        }
        covered.push(*family);
    }
    for family in IMAGE_FAMILIES {
        assert!(
            covered.contains(family),
            "family '{family}' is advertised but no budget of its own is exercised here"
        );
    }
}

/// ONE reserve per family, not one per call site.
///
/// Z-Image carried two independent figures for the same request: this table derived
/// its own from the activation shapes, the loader scaled a measured seed, and at
/// 1024^2 they read 1.9 GB and 3.2 GB. Two numbers is not a redundancy, it is a
/// disagreement - and only one of them was wired to the measurement loop, so after a
/// render exhausted a card the planner asked for what it had learned while the gate
/// that admits the render went on quoting the figure that admitted the failure.
///
/// Equality is asserted against the LOADER's helper by name: the point is not that
/// the two arithmetics agree today, it is that there is only one of them.
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
            family_runtime_bytes(&dir, ImageLoader::ZImage, "z-image-turbo", geom),
            crate::inference::engine::image_engine::zimage_runtime_demand_for(&dir, None, w, h),
            "the Z-Image admission gate and its placement disagree at {w}x{h}"
        );
    }
}

/// The weight side of the same defect: with ONLY a Flux checkpoint on disk, a family
/// that has no weights there must report none. The `_ =>` arm this replaces returned
/// the Flux file's size for every family it did not name, so an unbudgeted family
/// reserved a card for weights that belong to another model.
#[test]
fn no_family_reports_another_familys_checkpoint_as_its_weights() {
    let tmp = empty_models_dir("weights");
    let dir = tmp.0.to_string_lossy().to_string();
    // A Ray Flux fine-tune: a plain safetensors under `rayflux/`, the one layout the
    // Flux finder resolves without an HF cache. The bytes are not a checkpoint, and
    // nothing here reads them - only the file's LENGTH is the figure under test.
    const FLUX_WEIGHTS: usize = 4096;
    std::fs::create_dir_all(tmp.0.join("rayflux")).unwrap();
    std::fs::write(
        tmp.0.join("rayflux/rayflux.safetensors"),
        vec![0u8; FLUX_WEIGHTS],
    )
    .unwrap();

    assert_eq!(
        family_hot_bytes(&dir, ImageLoader::Flux, "rayflux"),
        FLUX_WEIGHTS as u64,
        "the Flux budget must read the Flux checkpoint"
    );
    for (family, tag) in FAMILY_TAGS {
        let loader = image_family_loader(family)
            .unwrap_or_else(|e| panic!("family '{family}' has no loader: {e}"));
        if loader == ImageLoader::Flux {
            continue;
        }
        assert_eq!(
            family_hot_bytes(&dir, loader, tag),
            0,
            "family '{family}' claims the Flux checkpoint as its own hot component"
        );
    }
}

/// A family nobody wired must FAIL, naming itself. Falling back to some other
/// family's loader is the bug, not the safety net: it renders the wrong model
/// instead of reporting that the family is unhandled.
#[test]
fn an_unwired_family_errors_by_name_instead_of_falling_back() {
    // `flux2` used to stand in here as an example of an unwired family. It is wired now,
    // so the placeholders have to be families that genuinely are not - and the point of the
    // test is unchanged: an unknown family must ERROR by name rather than borrow a loader.
    for unwired in ["stable-cascade", "pixart", "", "SDXL"] {
        let err = image_family_loader(unwired)
            .expect_err("an unwired family must not resolve to a loader");
        assert!(
            err.contains(unwired),
            "the error must name the unhandled family '{unwired}': {err}"
        );
    }
}

/// The recipe each family was validated at, written out here as an oracle.
///
/// A deliberate second copy of `image_model_defaults`, transcribed arm for arm from
/// the table as the family STRING left it: equality proves the re-keying moved no
/// family onto another family's numbers. Exhaustive on [`ImageLoader`], so a family
/// added tomorrow has to state its recipe HERE as well as in the table - which is the
/// point, because a family with no recipe of its own does not fail, it renders at
/// someone else's step count and someone else's resolution under its own name.
fn oracle_defaults(loader: ImageLoader, model_name: &str) -> ImageModelDefaults {
    match loader {
        ImageLoader::ZImage => ImageModelDefaults {
            steps: 9,
            guidance: 5.0,
            size: 1024,
            align: 16,
        },
        ImageLoader::QwenImage => {
            if crate::inference::engine::qwen_image_engine::is_step_distilled(model_name) {
                ImageModelDefaults {
                    steps: 4,
                    guidance: 1.0,
                    size: 1024,
                    align: 16,
                }
            } else {
                ImageModelDefaults {
                    steps: 20,
                    guidance: 4.0,
                    size: 1024,
                    align: 16,
                }
            }
        }
        ImageLoader::Flux2 => ImageModelDefaults {
            steps: 4,
            guidance: 1.0,
            size: 1024,
            align: 16,
        },
        ImageLoader::Boogu => ImageModelDefaults {
            steps: 4,
            guidance: 1.0,
            size: 1024,
            align: 16,
        },
        ImageLoader::Sdxl => ImageModelDefaults {
            steps: 25,
            guidance: 7.0,
            size: 1024,
            align: 64,
        },
        ImageLoader::Flux => {
            if is_ray_fp8_model(model_name) {
                ImageModelDefaults {
                    steps: 20,
                    guidance: 3.5,
                    size: 1024,
                    align: 16,
                }
            } else {
                ImageModelDefaults {
                    steps: 4,
                    guidance: 4.0,
                    size: 512,
                    align: 16,
                }
            }
        }
    }
}

/// Value by value, every advertised family still renders at its OWN recipe.
///
/// The drop-in variants are exercised beside their base because the two arms that
/// used a guard rather than a family name - the step-distilled Qwen-Image, the Ray
/// FLUX fine-tunes - are the ones a re-keying can quietly reorder.
#[test]
fn every_family_renders_at_its_own_recipe() {
    let mut covered: Vec<&str> = Vec::new();
    for (family, tag) in FAMILY_TAGS.iter().copied().chain([
        ("qwen-image", "qwen-image-flash"),
        ("qwen-image", "rayqwest"),
        ("zimage", "rayzist"),
        ("flux", "rayflux"),
        ("flux", "rayflux-krea"),
        ("sdxl", "raymnants"),
    ]) {
        assert_eq!(
            image_family(tag),
            family,
            "'{tag}' is classified as another family"
        );
        let loader = image_family_loader(family)
            .unwrap_or_else(|e| panic!("family '{family}' has no loader: {e}"));
        let got = image_model_defaults(tag)
            .unwrap_or_else(|e| panic!("'{tag}' has no recipe of its own: {e}"));
        let want = oracle_defaults(loader, tag);
        assert_eq!(
            got.steps, want.steps,
            "'{tag}' ({family}) renders at another step count"
        );
        assert_eq!(
            got.guidance, want.guidance,
            "'{tag}' ({family}) renders at another guidance"
        );
        assert_eq!(
            got.size, want.size,
            "'{tag}' ({family}) renders at another size"
        );
        assert_eq!(
            got.align, want.align,
            "'{tag}' ({family}) uses another alignment"
        );
        if !covered.contains(&family) {
            covered.push(family);
        }
    }
    for family in IMAGE_FAMILIES {
        assert!(
            covered.contains(family),
            "family '{family}' is advertised but no recipe of its own is exercised here"
        );
    }
}

/// The recipes must not collapse onto one another: the FLUX Schnell arm is what the
/// removed `_ =>` handed out, so a family that reads back as 4 steps at guidance 4.0
/// and 512 pixels without being FLUX Schnell has inherited it.
#[test]
fn no_family_inherits_the_flux_schnell_recipe() {
    let schnell = image_model_defaults("flux-schnell").unwrap();
    assert_eq!(
        (schnell.steps, schnell.guidance, schnell.size),
        (4, 4.0, 512)
    );
    for (family, tag) in FAMILY_TAGS {
        if *family == "flux" {
            continue;
        }
        let d = image_model_defaults(tag).unwrap();
        assert_ne!(
            (d.steps, d.guidance, d.size),
            (schnell.steps, schnell.guidance, schnell.size),
            "family '{family}' renders at FLUX Schnell's recipe"
        );
    }
}

/// A family nobody wired gets NO recipe and NO checkpoint - it is named in an error
/// rather than handed FLUX Schnell's numbers, which is what the `_ =>` arm did.
#[test]
fn an_unwired_family_gets_no_recipe_of_its_own() {
    for unwired in ["stable-cascade", "pixart", ""] {
        let err = image_model_defaults_for_family(unwired, unwired)
            .expect_err("an unwired family must not inherit a recipe");
        assert!(
            err.contains(unwired),
            "the error must name the family with no recipe '{unwired}': {err}"
        );
    }
}

/// THE CHECKPOINT RESOLVER, with only a FLUX GGUF on disk.
///
/// `requested_checkpoint` matched on the family string and ended in a `_ =>` arm
/// calling the FLUX finder, and that finder stops looking at the model name once its
/// Ray branch misses: it takes any `*flux*schnell*.gguf` under the models directory.
/// So a boogu or flux2 request was answered with the FLUX checkpoint's path, and the
/// family-filtered fallback that would have found the family's own file never ran.
#[test]
fn no_family_resolves_another_familys_checkpoint() {
    let tmp = empty_models_dir("ckpt");
    let dir = tmp.0.to_string_lossy().to_string();
    // Only the FLUX family has anything on disk. The bytes are not a checkpoint;
    // nothing here reads them, only the file's NAME is what the finder selects on.
    let flux = tmp.0.join("flux1-schnell-Q4_K.gguf");
    std::fs::write(&flux, b"not a checkpoint").unwrap();

    assert_eq!(
        requested_checkpoint(&dir, "flux-schnell").as_deref(),
        Some(flux.as_path()),
        "the FLUX resolver must still find the FLUX checkpoint"
    );
    for (family, tag) in FAMILY_TAGS {
        if *family == "flux" {
            continue;
        }
        assert_eq!(
            requested_checkpoint(&dir, tag),
            None,
            "family '{family}' resolves to the FLUX checkpoint as its own"
        );
    }
}

/// The guard this reflexivity replaced must survive: a Ray fine-tune may never be
/// served from a base-model request, or the reverse.
#[test]
fn a_ray_finetune_never_aliases_its_base_model() {
    assert!(!resident_serves_request(
        Some("rayzist"),
        None,
        "z-image-turbo",
        None
    ));
    assert!(!resident_serves_request(
        Some("z-image-turbo"),
        None,
        "rayzist",
        None
    ));
    assert!(!resident_serves_request(
        Some("rayqwest"),
        None,
        "qwen-image",
        None
    ));
    // Different families never match either.
    assert!(!resident_serves_request(
        Some("boogu"),
        None,
        "flux-schnell",
        None
    ));
    // Two files of the same family are distinct when both resolve locally.
    assert!(!resident_serves_request(
        Some("rayflux"),
        Some("A.safetensors"),
        "rayflux-krea",
        Some("B.safetensors")
    ));
    // Two SDXL fine-tunes with no resolvable file must not alias either: they did,
    // and a request for one was answered with the other's weights.
    assert!(!resident_serves_request(
        Some("rayctifier"),
        None,
        "rayburn",
        None
    ));
    assert!(!resident_serves_request(
        Some("rayburn"),
        None,
        "rayctifier",
        None
    ));
    // The same tag still serves itself - a resident that reloaded on every request
    // would be the opposite failure.
    assert!(resident_serves_request(
        Some("rayburn"),
        None,
        "rayburn",
        None
    ));
    assert!(resident_serves_request(
        Some("rayflux"),
        Some("A.safetensors"),
        "rayflux",
        Some("A.safetensors")
    ));
}

/// SOURCE GATE: a loader that writes a literal name into its resident state
/// breaks reflexivity for every variant tag routed through it - the defect above,
/// which was present at four of the five load sites. The name must come from the
/// request.
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
    // Z-Image-Turbo's flow-matching scheduler converges in 9
    // steps at its native 1024- training resolution.
    let d = image_model_defaults("z-image").unwrap();
    assert_eq!(d.size, 1024);
    assert_eq!(d.steps, 9);
    assert_eq!(d.guidance, 5.0);
}

#[test]
fn image_model_defaults_flux_uses_512_at_4_steps() {
    // Flux Schnell is trained on 4 steps; 512- is the bench-
    // validated default that fits in <16 GB VRAM with margin.
    let d = image_model_defaults("flux").unwrap();
    assert_eq!(d.size, 512);
    assert_eq!(d.steps, 4);
    assert_eq!(d.guidance, 4.0);
}

#[test]
fn image_model_defaults_zimage_vs_flux_differ_on_every_field() {
    // Pin that the Z-Image path doesn't accidentally collapse
    // to the Flux defaults - a regression there would silently
    // halve Z-Image's default step count + guidance and produce
    // washed-out output.
    let z = image_model_defaults("z-image").unwrap();
    let f = image_model_defaults("flux").unwrap();
    assert_ne!(z.size, f.size);
    assert_ne!(z.steps, f.steps);
    assert_ne!(z.guidance, f.guidance);
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

#[test]
fn is_z_image_model_matches_naming_variants() {
    assert!(is_z_image_model("Z-Image-Turbo"));
    assert!(is_z_image_model("Tongyi-MAI/Z-Image-Turbo"));
    assert!(is_z_image_model("z-image"));
    assert!(is_z_image_model("z_image_local"));
    // Case-insensitive
    assert!(is_z_image_model("Z_IMAGE"));
    // Negatives
    assert!(!is_z_image_model("flux-dev"));
    assert!(!is_z_image_model("stable-diffusion"));
    assert!(!is_z_image_model("qwen3-coder"));
    assert!(!is_z_image_model(""));
}

// -- is_image_gen_model ------------
// Centralises the image-gen dispatch check that previously lived
// inline as `model_lower.contains("flux") || ...` at 3 sites
// (/api/chat + 2- /api/generate) plus the capabilities helper.
// A future image family (e.g. Stable Diffusion 3.5 weights we
// could load via the upstream transformers crate) adds one substring here
// and reaches all 4 routes at once.

#[test]
fn is_image_gen_model_matches_flux_and_zimage_variants() {
    // Flux family (Schnell, Dev, fp16, etc.)
    assert!(is_image_gen_model("flux"));
    assert!(is_image_gen_model("flux-schnell"));
    assert!(is_image_gen_model("flux-dev:fp16"));
    assert!(is_image_gen_model("black-forest-labs/FLUX.1-schnell"));
    // Z-Image family with all naming variants
    assert!(is_image_gen_model("z-image"));
    assert!(is_image_gen_model("z_image"));
    assert!(is_image_gen_model("Z-Image-Turbo"));
    assert!(is_image_gen_model("Tongyi-MAI/Z-Image-Turbo"));
    // Case-insensitive
    assert!(is_image_gen_model("Z_IMAGE_LOCAL"));
}

#[test]
fn is_image_gen_model_is_superset_of_is_z_image_model() {
    // Drift guard: is_z_image_model classifies a subset of names
    // that is_image_gen_model also accepts (both check the
    // "z-image" / "z_image" substrings). A future refactor that
    // tightens one without the other (e.g. is_image_gen_model
    // checking "flux|stable-diffusion" but dropping z-image, or
    // is_z_image_model gaining a new pattern not mirrored)
    // would break the dispatch invariant the two classifiers
    // encode together.
    for name in [
        "z-image",
        "z_image",
        "Z-Image-Turbo",
        "Tongyi-MAI/Z-Image-Turbo",
        "z_image_local",
    ] {
        if is_z_image_model(name) {
            assert!(
                is_image_gen_model(name),
                "{name} classifies as Z-Image but not as image-gen - \
                 dispatch invariant violated"
            );
        }
    }
}

#[test]
fn is_image_gen_model_negatives() {
    // Text / vision / audio / unknown - none should match.
    assert!(!is_image_gen_model("qwen3-coder"));
    assert!(!is_image_gen_model("deepseek-r1:32b"));
    assert!(!is_image_gen_model("moondream:1.8b"));
    assert!(!is_image_gen_model("openai/whisper-small"));
    assert!(!is_image_gen_model("parler-tts/parler-tts-mini-v1"));
    assert!(!is_image_gen_model(""));
    // Substring false-positives don't sneak in - "fluxbox" isn't an
    // image-gen model but matches "flux"; intentional accepted as
    // collateral damage (rare to name a text LLM this way).
    assert!(
        is_image_gen_model("fluxbox-tester"),
        "documenting the substring false-positive corner case"
    );
}
