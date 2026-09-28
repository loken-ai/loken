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

// -- one-shot components placed whole --------------------------------
// A VAE or an encoder takes the first card its whole order lists, a preferred card
// first, and the host when the order is empty.

/// The card a whole component of `need` bytes, running best with `want`, lands on.
fn aux_card(free: &[(usize, u64)], need: u64, want: u64, prefer: Option<usize>) -> Option<usize> {
    let cards = crate::inference::place::ladder::prefer(free, prefer);
    crate::inference::place::ladder::whole_order(&cards, need, want)
        .first()
        .map(|(idx, _)| *idx)
}

/// The card a VAE of `weights` resident bytes lands on for a `width x height` decode.
fn vae_card(free: &[(usize, u64)], weights: u64, width: usize, height: usize) -> Option<usize> {
    let (need, want) = vae_placement_for(weights, width, height);
    aux_card(free, need, want, None)
}

/// The placement that stranded the VAE on the CPU for a process's lifetime.
/// Both tiers matter: without the floor a card short of the whole-image peak is
/// rejected outright, and without the peak a roomy card would be sized as if it had
/// to tile.
#[test]
fn the_vae_falls_back_to_a_gpu_before_it_falls_back_to_the_cpu() {
    let weights = 600_000_000u64;

    // Neither card holds the whole-image peak beside the weights, and one has far
    // more than the tiled floor needs.
    let (need, want) = vae_placement_for(weights, 2048, 2048);
    let tight = &[(0, need - 1), (1, (need + want) / 2)];
    assert_eq!(
        vae_card(tight, weights, 2048, 2048),
        Some(1),
        "a tiled decode fits here; the CPU is not the answer"
    );

    // Roomy: the fastest card that holds the whole-image peak, undivided.
    let roomy = &[(0, 12_000_000_000u64), (1, 12_000_000_000u64)];
    assert_eq!(vae_card(roomy, weights, 1024, 1024), Some(0));

    // Genuinely full: no card can host even the smallest tile plus the weights,
    // and THEN the CPU is correct.
    let full = &[(0, 100_000_000u64), (1, 80_000_000u64)];
    assert_eq!(vae_card(full, weights, 1024, 1024), None);

    // No GPUs at all.
    assert_eq!(vae_card(&[], weights, 1024, 1024), None);
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

/// The whole-image peak is a preference and the tiled floor a requirement: a card that
/// misses the first and holds the second keeps the VAE off the CPU.
#[test]
fn a_vae_that_misses_the_whole_image_peak_still_lands_on_a_gpu() {
    let weights = 600_000_000u64;
    let (need, want) = vae_placement_for(weights, 2048, 2048);
    assert!(need < want, "the tiled floor must be the cheaper demand");
    let free = &[(0, need / 2), (1, want - 1)];
    assert_eq!(
        aux_card(free, want, want, None),
        None,
        "the peak fits neither card"
    );
    assert_eq!(
        aux_card(free, need, want, None),
        Some(1),
        "the tiled floor must keep the VAE on a GPU"
    );
}

#[test]
fn a_one_shot_component_goes_to_the_idle_card() {
    // GPU0 nearly full after the transformer, GPU1 nearly idle.
    let free = &[(0, 1_100_000_000u64), (1, 15_200_000_000u64)];
    assert_eq!(aux_card(free, 1_500_000_000, 1_500_000_000, None), Some(1));
}

#[test]
fn a_preferred_card_with_room_wins() {
    // Preferred device has enough room: it wins even beside a roomier card, which
    // saves the transfer of the latent.
    let free = &[(0, 3_000_000_000u64), (1, 10_000_000_000u64)];
    assert_eq!(
        aux_card(free, 1_500_000_000, 1_500_000_000, Some(0)),
        Some(0)
    );
    assert_eq!(
        aux_card(free, 1_500_000_000, 1_500_000_000, Some(1)),
        Some(1)
    );
}

#[test]
fn a_preferred_card_without_room_is_passed_over() {
    let free = &[(0, 500_000_000u64), (1, 10_000_000_000u64)];
    assert_eq!(
        aux_card(free, 1_500_000_000, 1_500_000_000, Some(0)),
        Some(1)
    );
}

#[test]
fn nothing_fits_means_the_host() {
    let free = &[(0, 500_000_000u64), (1, 800_000_000u64)];
    assert_eq!(aux_card(free, 1_500_000_000, 1_500_000_000, None), None);
    assert_eq!(aux_card(free, 1_500_000_000, 1_500_000_000, Some(0)), None);
    assert_eq!(aux_card(&[], 1_500_000_000, 1_500_000_000, None), None);
    assert_eq!(aux_card(&[], 1_500_000_000, 1_500_000_000, Some(0)), None);
}

#[test]
fn a_one_shot_component_takes_the_fastest_card_that_fits() {
    // Both cards fit: the first in the fastest-first order wins, whatever the other
    // card has free.
    let free = &[(0, 9_800_000_000u64), (1, 11_400_000_000u64)];
    assert_eq!(aux_card(free, 5_000_000_000, 5_000_000_000, None), Some(0));
}

#[test]
fn a_one_shot_component_stays_off_the_hot_cards_while_another_has_room() {
    use crate::inference::place::ladder::aside;
    let free = &[(0, 12_000_000_000u64), (1, 6_000_000_000u64)];
    // The hot component runs on card 0: the encoder goes to card 1.
    assert_eq!(
        aux_card(
            &aside(free, &[0], 4_000_000_000),
            5_000_000_000,
            5_000_000_000,
            None
        ),
        Some(1)
    );
    // Card 1 too small: card 0 is shared, but only with the hot component's need kept.
    let small = &[(0, 12_000_000_000u64), (1, 2_000_000_000u64)];
    assert_eq!(
        aux_card(
            &aside(small, &[0], 4_000_000_000),
            5_000_000_000,
            5_000_000_000,
            None
        ),
        Some(0)
    );
    assert_eq!(
        aux_card(
            &aside(small, &[0], 8_000_000_000),
            5_000_000_000,
            5_000_000_000,
            None
        ),
        None
    );
    // Kept whole: the hot cards are not offered at all.
    assert_eq!(
        aux_card(
            &aside(small, &[0], u64::MAX),
            5_000_000_000,
            5_000_000_000,
            None
        ),
        None
    );
}

#[test]
fn a_kontext_edit_refuses_regions_and_masks_it_cannot_honour() {
    let z = || Tensor::zeros((1,), DType::F32).unwrap();
    let region = crate::inference::model::flux::sampling::Region {
        txt: z(),
        txt_ids: z(),
        vec: z(),
        weight: z(),
    };
    let regions = [region];
    assert!(refuse_what_kontext_drops(true, &None, &regions).is_err());
    assert!(refuse_what_kontext_drops(false, &None, &regions).is_ok());
    assert!(refuse_what_kontext_drops(true, &None, &[]).is_ok());
}

#[test]
fn a_dev_checkpoint_pushes_its_schedule_and_schnell_does_not() {
    let schnell = flux_schedule(None, 28, 1024, 1024);
    assert_eq!(
        schnell,
        crate::inference::model::flux::sampling::get_schedule(28, None)
    );
    let dev = flux_schedule(
        Some((FLUX_DEV_BASE_SHIFT, FLUX_DEV_MAX_SHIFT)),
        28,
        1024,
        1024,
    );
    assert_eq!(dev.len(), schnell.len());
    assert_eq!((dev[0], *dev.last().unwrap()), (1.0, 0.0));
    // The push moves every interior stop toward the noisy end.
    for (d, s) in dev.iter().zip(&schnell).skip(1).take(27) {
        assert!(d > s, "dev stop {d} is not above schnell stop {s}");
    }
}

#[test]
fn a_named_seed_is_kept_and_a_missing_one_is_drawn_once() {
    let mut named = ImageGenParams {
        seed: Some(7),
        ..Default::default()
    };
    assert_eq!(named.resolve_seed(), 7);
    assert_eq!(named.seed, Some(7));

    let mut unnamed = ImageGenParams::default();
    let drawn = unnamed.resolve_seed();
    assert_eq!(
        unnamed.seed,
        Some(drawn),
        "the drawn seed must be stored back"
    );
    assert_eq!(
        unnamed.resolve_seed(),
        drawn,
        "a second read must not redraw"
    );
}
