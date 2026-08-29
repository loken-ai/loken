use super::*;

#[test]
fn clamp_prompt_to_window_keeps_bos_and_recent_tail() {
    // Under budget -> unchanged.
    let short: Vec<u32> = (0..100).collect();
    assert_eq!(clamp_prompt_to_window(short.clone(), 4096, 128), short);

    // Over budget -> BOS (tokens[0]) + the most-recent tail, sized to the
    // window minus a generation reserve. This is the `> n_ctx` crash
    // guard: without it a prompt longer than the KV window drives chunked
    // prefill past the KV buffer and the attention-mask broadcast panics
    // (mask [seq_q, index_pos+seq_q] vs the bounded scores).
    let long: Vec<u32> = (0..10_000).collect();
    let out = clamp_prompt_to_window(long, 4096, 256);
    assert!(out.len() <= 4096, "must fit the window, got {}", out.len());
    assert!(out.len() < 4096, "must reserve room for generation");
    assert_eq!(out[0], 0, "BOS / template-start token preserved");
    assert_eq!(
        *out.last().unwrap(),
        9_999,
        "keeps the most recent token (final instruction)"
    );
    // The kept tail is contiguous and recent (not the head).
    assert_eq!(out[1], 9_999 - (out.len() - 2) as u32);
}

#[test]
fn clamp_prompt_to_window_reserve_never_exceeds_half_the_window() {
    // A huge want_gen can't starve the prompt below half the window.
    let long: Vec<u32> = (0..10_000).collect();
    let out = clamp_prompt_to_window(long, 4096, 100_000);
    assert!(
        out.len() >= 4096 / 2,
        "prompt keeps at least half the window, got {}",
        out.len()
    );
}

#[test]
fn vision_extra_kv_reserves_embeds_plus_splice_controls() {
    let img = Tensor::zeros_on(
        (1usize, 10usize, 4usize),
        crate::tensor::DType::F32,
        &Device::Cpu,
    )
    .unwrap();
    assert_eq!(
        vision_extra_kv(Some(&img)),
        10 + VISION_SPLICE_CONTROL_TOKENS
    );
    assert_eq!(vision_extra_kv(None), 0);
}

#[test]
fn vision_clamp_bounds_total_spliced_prefill_to_window() {
    // Regression for the >kv_ctx spliced-prefill overflow (pixtral repro:
    // image 2925 embeds + 1322-token question at window 4096 -> total 4251
    // > 4096 -> per-layer Q8 KV append failed -> silent degenerate decode;
    // historically a mask-broadcast panic). The text clamp must reserve
    // the image's KV positions so prompt + splice always fits the window.
    let img = Tensor::zeros_on(
        (1usize, 2925usize, 8usize),
        crate::tensor::DType::F32,
        &Device::Cpu,
    )
    .unwrap();
    let extra = vision_extra_kv(Some(&img));
    let window = 4096usize;
    assert!(
        extra < window,
        "this image must be accepted at window {window}"
    );
    let toks: Vec<u32> = (0..1322).collect();
    let clamped = clamp_prompt_to_window(toks, window - extra, 96);
    assert!(
        clamped.len() + extra <= window,
        "clamped text {} + vision extra {} must fit window {window}",
        clamped.len(),
        extra
    );
    // An image alone larger than the window must be rejected by the
    // caller (vision_extra >= kv_window -> clean error, never a panic).
    let huge = Tensor::zeros_on(
        (1usize, 4093usize, 8usize),
        crate::tensor::DType::F32,
        &Device::Cpu,
    )
    .unwrap();
    assert!(vision_extra_kv(Some(&huge)) >= window);
}

#[test]
fn qwen35_vision_extra_kv_counts_merged_grid_minus_sentinel() {
    // grid 64x48 patches -> merged 32x24 = 768 image tokens; the prompt
    // already holds ONE sentinel, so the extra KV is n_merged - 1.
    let px = Tensor::zeros_on((1usize, 4usize), crate::tensor::DType::F32, &Device::Cpu).unwrap();
    assert_eq!(
        qwen35_vision_extra_kv(Some(&(px.clone(), (64, 48)))),
        32 * 24 - 1
    );
    assert_eq!(qwen35_vision_extra_kv(None), 0);
    // Degenerate 2x2 grid -> 1 merged token -> sentinel covers it -> 0 extra.
    assert_eq!(qwen35_vision_extra_kv(Some(&(px, (2, 2)))), 0);
}

#[test]
fn qwen35_clamp_preserves_image_sentinel() {
    // Same >window class as the pixtral repro, qwen35 form: the image is a
    // single front-positioned sentinel token that in-place expands to
    // n_merged copies at prefill. The clamp must (a) bound text so
    // text + (n_merged - 1) fits the window and (b) NEVER drop the
    // sentinel (generic BOS+tail clamp would -> "sentinel missing" error).
    const SENT: u32 = 248056;
    let window = 4096usize;
    let n_merged = 32 * 24; // grid (64, 48)
    let extra = n_merged - 1;
    // prompt: 8-token preamble, sentinel, 6000-token question.
    let mut toks: Vec<u32> = (1..=8).collect();
    toks.push(SENT);
    toks.extend(10_000..16_000);
    let clamped = clamp_qwen35_prompt_to_window(toks.clone(), window - extra, 96, SENT);
    assert!(
        clamped.contains(&SENT),
        "clamp must keep the image sentinel"
    );
    assert_eq!(
        clamped.iter().filter(|&&t| t == SENT).count(),
        1,
        "exactly one sentinel"
    );
    assert!(
        clamped.len() + extra <= window,
        "clamped text {} + image extra {extra} must fit window {window}",
        clamped.len()
    );
    // Head preserved through the sentinel, tail keeps the most recent text.
    assert_eq!(&clamped[..9], &toks[..9]);
    assert_eq!(*clamped.last().unwrap(), 15_999);
    // Under-budget prompts pass through untouched.
    let short: Vec<u32> = vec![1, 2, SENT, 4, 5];
    assert_eq!(
        clamp_qwen35_prompt_to_window(short.clone(), window - extra, 96, SENT),
        short
    );
    // No sentinel -> generic clamp behavior (BOS + tail).
    let no_sent: Vec<u32> = (0..6000).collect();
    let generic = clamp_qwen35_prompt_to_window(no_sent.clone(), window, 96, SENT);
    assert_eq!(generic, clamp_prompt_to_window(no_sent, window, 96));
}

#[test]
fn effective_kv_window_is_min_of_model_config_request() {
    // Request cap wins when smallest (Ollama num_ctx).
    assert_eq!(effective_kv_window(1_000_000, 32_768, Some(8_192)), 8_192);
    // Config cap wins with no request override and < model.
    assert_eq!(effective_kv_window(1_000_000, 4_096, None), 4_096);
    // A request can't exceed the model/config base (guards KV over-alloc).
    assert_eq!(effective_kv_window(4_096, 32_768, Some(999_999)), 4_096);
    // config_ctx == 0 means "unset" -> fall back to the model context.
    assert_eq!(effective_kv_window(8_192, 0, None), 8_192);
}

#[test]
fn build_sampling_from_temperature_zero_is_argmax_regardless_of_top_p_k() {
    // temperature=0 -> greedy decode (highest logit always picked).
    // top_p / top_k MUST be ignored in this branch; a regression
    // that picked TopP { p: 0.5 } here would silently re-introduce
    // randomness into "deterministic" greedy decode.
    assert_eq!(build_sampling_from(0.0, 0.9, 40), Sampling::ArgMax);
    assert_eq!(build_sampling_from(0.0, 1.0, 0), Sampling::ArgMax);
    assert_eq!(build_sampling_from(0.0, 0.5, 1), Sampling::ArgMax);
}

#[test]
fn build_sampling_from_routes_to_top_k_then_top_p_when_both_set() {
    // top_k > 0 AND top_p < 1.0 -> combine both. Pin so a refactor
    // that reorders the elseif chain (e.g. top_p first) doesn't
    // accidentally drop the top_k cap. Use exactly-representable
    // f32 values (powers of two) to avoid f32->f64 cast drift.
    let s = build_sampling_from(0.5, 0.25, 40);
    assert_eq!(
        s,
        Sampling::TopKThenTopP {
            k: 40,
            p: 0.25_f64,
            temperature: 0.5_f64,
        }
    );
}

#[test]
fn build_sampling_from_routes_to_top_k_when_top_p_is_one() {
    // top_p = 1.0 means "no top-p cutoff" -> top_k alone.
    let s = build_sampling_from(0.5, 1.0, 20);
    assert_eq!(
        s,
        Sampling::TopK {
            k: 20,
            temperature: 0.5_f64
        }
    );
}

#[test]
fn build_sampling_from_routes_to_top_p_when_no_top_k() {
    // top_k = 0 -> fall through to top_p only. Powers-of-2 values
    // round-trip cleanly through f32->f64 (0.95 doesn't, 0.5 does).
    let s = build_sampling_from(0.5, 0.25, 0);
    assert_eq!(
        s,
        Sampling::TopP {
            p: 0.25_f64,
            temperature: 0.5_f64
        }
    );
}
