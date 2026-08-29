use super::*;

// -- load cancellation ------------------------------------------
// A client that drops an image stream must stop the LOAD, not only the render.
// Measured: an abandoned `/v1/images/generations` stream ran the Z-Image load
// through to "Z-Image-Turbo fully loaded" and the next request was refused with
// "every GPU is busy with an in-flight generation". Flux, Qwen-Image and Boogu
// took a token; these two took none.
//
// These tests reach no device and open no file BY CONSTRUCTION: if the guard were
// missing, the failure would be a probe or a missing checkpoint instead of a
// cancellation, which is exactly what they assert.

fn already_abandoned() -> crate::inference::serve::cancel::CancelToken {
    let t = crate::inference::serve::cancel::CancelToken::new();
    t.cancel();
    t
}

#[tokio::test]
async fn an_abandoned_z_image_load_never_starts() {
    let engine = ImageEngine::new();
    let err = engine
        .load_z_image_turbo_cancellable(
            ModelRequest::new(
                crate::config::Config::default_hf_models_dir().to_string_lossy(),
                "z-image-turbo",
            ),
            None,
            crate::inference::place::runtime_demand::RequestGeometry::new(1024, 1024),
            None,
            Some(already_abandoned()),
        )
        .await
        .expect_err("a load with no client left must not proceed");
    assert!(
        err.to_string().contains("cancelled"),
        "the refusal must say why, and must not be a panic or a probe failure: {err}"
    );
    assert!(
        !engine.is_loaded().await,
        "an abandoned load must leave nothing resident"
    );
}

#[tokio::test]
async fn an_abandoned_sdxl_load_never_opens_its_checkpoint() {
    let engine = ImageEngine::new();
    // Paths that do not exist: reaching the pipeline at all would fail on them, so a
    // cancellation message proves the load stopped BEFORE the checkpoint was read -
    // which is the whole point, the checkpoint being the multi-gigabyte part.
    let err = engine
        .load_sdxl_cancellable(
            std::path::PathBuf::from("/nonexistent/sdxl.safetensors"),
            "sdxl".to_string(),
            std::path::PathBuf::from("/nonexistent/tokenizer.json"),
            crate::inference::place::runtime_demand::RequestGeometry::new(1024, 1024),
            None,
            Some(already_abandoned()),
        )
        .await
        .expect_err("a load with no client left must not proceed");
    assert!(
        err.to_string().contains("cancelled"),
        "stopped for the wrong reason - the checkpoint was reached: {err}"
    );
}

/// The nominal path is what it always was. A live token and no token at all both
/// let the load through; only a cancelled one turns it back.
#[test]
fn a_load_nobody_cancelled_is_let_through() {
    assert!(
        ImageEngine::refuse_if_abandoned(None).is_ok(),
        "the render binaries pass none"
    );
    let live = crate::inference::serve::cancel::CancelToken::new();
    assert!(ImageEngine::refuse_if_abandoned(Some(&live)).is_ok());
    assert!(ImageEngine::refuse_if_abandoned(Some(&already_abandoned())).is_err());
}

// -- pick_aux_device --------------------------------------------
// The replacement for vae_fits_on_gpu's single-device check.
// Pins the architectural fix for the OOM the user hit (GPU0 full
// from transformer, GPU1 idle, VAE always landed on GPU0).

/// The placement that stranded the VAE on the CPU for a process's lifetime.
/// Both tiers matter: without the floor a card 130 MB short of the whole-image
/// peak is rejected outright, and without the peak a roomy card would be sized
/// as if it had to tile.
#[test]
fn the_vae_falls_back_to_a_gpu_before_it_falls_back_to_the_cpu() {
    let weights = 600_000_000u64;

    // The measured pressure state: neither card holds the 4.29 GB peak, but one
    // has 4.16 GB - far more than the tiled floor needs.
    let tight = &[(0, 1_550_000_000u64), (1, 4_160_000_000u64)];
    assert!(
        pick_vae_gpu(tight, weights, 1024, 1024).is_some(),
        "a tiled decode fits here; the CPU is not the answer"
    );

    // Roomy: the fastest card that holds the whole-image peak, undivided.
    let roomy = &[(0, 12_000_000_000u64), (1, 12_000_000_000u64)];
    assert_eq!(pick_vae_gpu(roomy, weights, 1024, 1024), Some(0));

    // Genuinely full: no card can host even the smallest tile plus the weights,
    // and THEN the CPU is correct. Reporting a GPU here would be worse than the
    // fallback it replaced.
    let full = &[(0, 100_000_000u64), (1, 80_000_000u64)];
    assert_eq!(pick_vae_gpu(full, weights, 1024, 1024), None);

    // No GPUs at all.
    assert_eq!(pick_vae_gpu(&[], weights, 1024, 1024), None);
}

/// The tile choice decides, per request, whether the decode stays on the device.
/// It runs only when a model is loaded and a card is short, so without this it is
/// exercised by nothing.
#[test]
fn the_tile_shrinks_until_the_decode_fits_and_gives_up_honestly() {
    let peak = |w: usize, h: usize| (w * h) as u64 * 4 * (VAE_WIDEST_CH * 9 + VAE_WIDEST_CH * 2);
    let p1k = peak(1024, 1024);

    // Room to spare: the caller never asks, but a generous budget must still not
    // return a tile SMALLER than necessary - biggest-first is the whole point.
    assert_eq!(choose_vae_tile(6_000_000_000, p1k, 1024, 1024), Some(96));
    // 64 is the coarsest tile that actually splits a 128-px latent, and it is the
    // floor: finer tiles drift visibly (0.0185 at 3x3, 0.0338 at 4x4, measured on a
    // real latent) so they are refused even when they would fit the budget.
    assert_eq!(choose_vae_tile(3_500_000_000, p1k, 1024, 1024), Some(64));
    // 1.55 GB free is the state that used to pick a 32-px tile - a 4x4 split, the
    // regime that blocks. It now declines: one 2x2 tile needs ~2.3 GB and does not
    // fit, so the honest answer is the CPU rather than a faster wrong picture.
    assert_eq!(choose_vae_tile(1_550_000_000, p1k, 1024, 1024), None);
    assert_eq!(choose_vae_tile(200_000_000, p1k, 1024, 1024), None);
    // And it never returns a split finer than 2x2 at any budget.
    for gb in 1..=8u64 {
        if let Some(t) = choose_vae_tile(gb * 1_000_000_000, p1k, 1024, 1024) {
            assert!(t >= 64, "{gb} GB chose tile {t}, finer than the 2x2 floor");
        }
    }
    // A small image is already smaller than a tile: splitting buys nothing, and a
    // tile wider than the latent would make the "keep the interior" arithmetic moot.
    assert_eq!(
        choose_vae_tile(6_000_000_000, peak(256, 256), 256, 256),
        None
    );
    // Monotonicity: more room never yields a smaller tile.
    let mut prev = 0usize;
    for gb in 1..=8u64 {
        let t = choose_vae_tile(gb * 1_000_000_000, p1k, 1024, 1024).unwrap_or(0);
        assert!(t >= prev, "{gb} GB gave tile {t} after {prev}");
        prev = t;
    }
}

/// The state from the pressure run: a chat model on GPU0, the transformer spread,
/// and 4.16 GB free on GPU1 against a 4.29 GB whole-image decode peak. The old
/// single-tier placement missed by ~130 MB and sent the VAE to the CPU, where the
/// decode cost 116 s of a 176 s request. It never needed that much to LIVE there -
/// only to decode the whole image in one piece, which tiling makes optional.
#[test]
fn a_vae_that_misses_the_whole_image_peak_still_lands_on_a_gpu() {
    const NATIVE_SIDE: u64 = 1024;
    let peak = NATIVE_SIDE * NATIVE_SIDE * VAE_WIDEST_CH * 8 * 4;
    const SMALLEST_TILE_SIDE: u64 = (32 + 2 * 8) * 8;
    let weights = 300_000_000u64 * 2;
    let floor = weights + SMALLEST_TILE_SIDE * SMALLEST_TILE_SIDE * VAE_WIDEST_CH * 8 * 4;
    assert!(floor < peak, "the tiled floor must be the cheaper demand");

    let free = &[(0, 1_550_000_000u64), (1, 4_160_000_000u64)];
    assert_eq!(
        pick_aux_device(free, peak, None),
        None,
        "the peak fits neither card"
    );
    let tiered = pick_aux_device(free, peak, None).or_else(|| pick_aux_device(free, floor, None));
    // WHICH card is the fleet rule's call (fastest that fits); what this pins is that
    // some card is chosen at all, because the alternative is not a slower GPU - it is
    // the CPU, and that is the difference between seconds and minutes.
    assert!(
        tiered.is_some(),
        "the tiled floor must keep the VAE on a GPU, got {tiered:?}"
    );
}

#[test]
fn pick_aux_device_user_2x17gb_routes_vae_to_idle_gpu() {
    // Reproduces the user's exact post-transformer-load state from
    // an OOM log: GPU0 nearly full (1.1 GB free),
    // GPU1 nearly idle (15.2 GB free). VAE needs 1.5 GB.
    // Pre-fix: vae_fits_on_gpu(1.1 GB) = false -> CPU fallback
    //          (slow). VAE actually targeted GPU0 -> OOM.
    // Post-fix: pick_aux_device picks GPU1 (first with room), VAE lives
    //          there, no OOM, no PCIe transfer (last transformer
    //          segment is also on GPU1 per the new placement).
    let free = &[(0, 1_100_000_000u64), (1, 15_200_000_000u64)];
    let picked = pick_aux_device(free, 1_500_000_000, None);
    assert_eq!(
        picked,
        Some(1),
        "expected GPU1 (first that fits), got {picked:?}"
    );
}

#[test]
fn pick_aux_device_prefers_target_device_when_it_fits() {
    // Preferred device has enough room -> return it even if
    // another device has MORE room. Saves the PCIe transfer.
    let free = &[(0, 3_000_000_000u64), (1, 10_000_000_000u64)];
    // VAE prefers GPU0 (last transformer segment), needs 1.5 GB.
    // Both fit but prefer=0 wins.
    let picked = pick_aux_device(free, 1_500_000_000, Some(0));
    assert_eq!(
        picked,
        Some(0),
        "preferred device with enough room should win, got {picked:?}"
    );
}

#[test]
fn pick_aux_device_skips_preferred_when_it_does_not_fit() {
    // Preferred is too tight -> fall through to the first device with room.
    let free = &[(0, 500_000_000u64), (1, 10_000_000_000u64)];
    let picked = pick_aux_device(free, 1_500_000_000, Some(0));
    assert_eq!(
        picked,
        Some(1),
        "preferred too small, should fall back to first-fit, got {picked:?}"
    );
}

#[test]
fn pick_aux_device_returns_none_when_nothing_fits() {
    // Nothing has 1.5 GB -> caller falls back to CPU.
    let free = &[(0, 500_000_000u64), (1, 800_000_000u64)];
    assert_eq!(pick_aux_device(free, 1_500_000_000, None), None);
    assert_eq!(pick_aux_device(free, 1_500_000_000, Some(0)), None);
}

#[test]
fn pick_aux_device_handles_empty_device_list() {
    // No CUDA devices at all -> CPU.
    assert_eq!(pick_aux_device(&[], 1_500_000_000, None), None);
    assert_eq!(pick_aux_device(&[], 1_500_000_000, Some(0)), None);
}

#[test]
fn pick_aux_device_text_encoder_picks_fastest_that_fits() {
    // Realistic post-transformer state: GPU0 holds 17 layers
    // (~7 GB weights), GPU1 holds 13 (~5 GB). Residual free
    // ~9.8/11.4 GB, text encoder needs ~5 GB. Both fit -> the
    // FIRST device in the caller's fastest-first order wins (the
    // one-shot encode also runs fastest there). Memory-pressure
    // balancing is not a placement signal; a device without room
    // simply fails the fit filter.
    let free = &[(0, 9_800_000_000u64), (1, 11_400_000_000u64)];
    let picked = pick_aux_device(free, 5_000_000_000, None);
    assert_eq!(
        picked,
        Some(0),
        "text encoder should land on the fastest device that fits, got {picked:?}"
    );
}
