use super::*;

/// WHAT ONE ATTENTION OF THIS DENOISER ACTUALLY HOLDS, counted by running it.
///
/// This is the case the analytic reserve cannot get right on principle. The
/// scores of a 4096-token attention over 24 heads are 1.6 GB if they are
/// materialised whole - and they never are, because [`attention_chunked`] walks
/// the queries 512 at a time and drops each chunk's scores before the next. The
/// formula next to the placement has to GUESS that, through a "query tile" term
/// that is a second copy of a constant living in this file; when the kernel
/// changed its tile, the two drifted and the estimate under-charged in the
/// direction that OOMs.
///
/// The dry run does not guess: it runs the same function and reports what was
/// held. The bound below is deliberately loose - it pins the ORDER, which is
/// what the guess got wrong - and the point is that this number comes from the
/// production code path, not from a description of it.
#[test]
fn counting_one_attention_shows_the_chunking_the_formula_has_to_guess() {
    use crate::tensor::{DType, Device, Tensor};
    let dev = Device::dry();
    let led = dev.dry_ledger().unwrap().clone();
    let (b, heads, seq, head_dim) = (1usize, 24usize, 4096usize, 128usize);
    let q = Tensor::dry(&dev, DType::F32, (b, heads, seq, head_dim)).unwrap();
    let k = Tensor::dry(&dev, DType::F32, (b, heads, seq, head_dim)).unwrap();
    let v = Tensor::dry(&dev, DType::F32, (b, heads, seq, head_dim)).unwrap();
    let out = attention(&q, &k, &v, None, 0.088).unwrap();
    assert_eq!(
        out.dims(),
        &[b, heads, seq, head_dim],
        "the counted attention changed shape"
    );
    let whole_scores = (b * heads * seq * seq * 4) as u64;
    let peak = led.peak_bytes();
    assert!(
        peak < whole_scores,
        "counted {peak} bytes for an attention whose UNCHUNKED scores alone are \
         {whole_scores} - the chunking was not followed"
    );
    // ...and it is not free either: one chunk of scores is real memory, and a
    // reserve that omits it admits a render that cannot run.
    let one_chunk = (b * heads * attn_query_chunk(head_dim, seq, heads) * seq * 4) as u64;
    assert!(
        peak > one_chunk,
        "counted {peak} bytes, less than the {one_chunk} one chunk of scores takes"
    );
}

/// The count must FOLLOW the request, which is the property every fixed reserve
/// in this fleet has failed.
///
/// Note what the shape of the growth says. Four times the tokens costs about
/// four times the memory, not sixteen: the scores never exist whole, so nothing
/// here is quadratic. That is a fact about this attention's chunking, and the
/// analytic reserve can only reproduce it by being told - by a "tile" constant
/// copied out of the file it is describing, which is exactly the copy that
/// drifted. The dry run reads it off the code.
#[test]
fn the_counted_attention_grows_with_the_request() {
    use crate::tensor::{DType, Device, Tensor};
    let peak_at = |seq: usize| -> u64 {
        let dev = Device::dry();
        let led = dev.dry_ledger().unwrap().clone();
        let (b, heads, head_dim) = (1usize, 24usize, 128usize);
        let q = Tensor::dry(&dev, DType::F32, (b, heads, seq, head_dim)).unwrap();
        let k = Tensor::dry(&dev, DType::F32, (b, heads, seq, head_dim)).unwrap();
        let v = Tensor::dry(&dev, DType::F32, (b, heads, seq, head_dim)).unwrap();
        let _ = attention(&q, &k, &v, None, 0.088).unwrap();
        led.peak_bytes()
    };
    let small = peak_at(1024);
    let large = peak_at(4096);
    assert!(
        large > 3 * small,
        "{large} barely moved from {small} for 4x the tokens"
    );
    assert!(
        large < 6 * small,
        "{large} grew faster than the sequence did from {small}"
    );
}

/// WHAT THE WALK COUNTS CARD BY CARD, on the real checkpoint, at the geometry that
/// died and at the one that works.
///
/// The figures printed here are the ones the placement tests in `image_engine` are
/// written against, and they are printed TO THE BYTE because the decisions they feed
/// turn on tens of megabytes: at 1536 square a card was forty megabytes short of the
/// forward it was about to run, and no figure rounded to a tenth of a gigabyte can
/// show that.
///
/// What it asserts is the property a single reserve cannot have: the card carrying
/// the stem holds MORE of a forward than a card carrying only blocks, and neither
/// figure moves much with how many blocks it was given - a forward is a property of
/// the request, and the stream visits every card.
///
/// Ignored because it needs the local snapshot; it needs no GPU and reads no tensor
/// data, so it is safe to run beside a live server:
///
/// ```text
/// cargo test --release -p loken --lib -- --ignored --nocapture \
///     zimage_native::tests::the_walk_counts_the_real_checkpoint_card_by_card
/// ```
#[test]
#[ignore = "needs the local Z-Image-Turbo snapshot"]
fn the_walk_counts_the_real_checkpoint_card_by_card() {
    let files = real_checkpoint_shards();
    let refs: Vec<&str> = files.iter().map(String::as_str).collect();
    let cfg = Config::z_image_turbo();
    let n = cfg.n_layers;
    for (w, h) in [(1024usize, 1024usize), (1536, 1536)] {
        let run = |spans: &[(usize, usize, usize)]| {
            super::dry_forward_planned(
                &cfg,
                &refs,
                None,
                DType::BF16,
                DType::F32,
                h / 8,
                w / 8,
                256,
                &plan_of(spans, n),
                // What the pool on the cards this was measured on reserves in, so the
                // printed weights are what a card HOLDS rather than what they count.
                Some((32 << 20, 512)),
            )
            .expect("the placed dry run")
            .load
        };
        let whole = run(&[(0, 0, n)]);
        println!("{w}x{h} whole: {}", whole.describe());
        let mut stem_forward = 0u64;
        let mut rest_forward = 0u64;
        for first in [n / 2, (2 * n) / 3, (7 * n) / 10, (4 * n) / 5] {
            let load = run(&[(0, 0, first), (1, first, n)]);
            let a = load.on(DeviceKind::Cuda(0)).expect("no first card");
            let b = load.on(DeviceKind::Cuda(1)).expect("no second card");
            println!(
                "{w}x{h} {first}/{}: stem card weights {} forward {}, other card weights {} \
                 forward {}",
                n - first,
                a.weights,
                a.forward,
                b.weights,
                b.forward,
            );
            assert!(
                a.forward > b.forward,
                "the card carrying the stem holds {} of a forward and the other {} - a \
                 single reserve would be right about at most one of them",
                a.forward,
                b.forward,
            );
            // The same forward, whatever the split: what changes with the blocks is
            // the WEIGHTS. A test that models the walk this way is modelling it
            // correctly only while this holds.
            if stem_forward > 0 {
                assert_eq!(
                    stem_forward, a.forward,
                    "the stem card's forward moved with the split"
                );
                assert_eq!(
                    rest_forward, b.forward,
                    "the other card's forward moved with the split"
                );
            }
            stem_forward = a.forward;
            rest_forward = b.forward;
        }
        // The stem and one block, which is how the placement tests reconstruct any
        // split from two numbers.
        let half = run(&[(0, 0, n / 2), (1, n / 2, n)]);
        let per_block = (whole.devices[0].weights - half.devices[0].weights) / (n - n / 2) as u64;
        println!(
            "{w}x{h}: per block {} bytes, stem {} bytes, forward {} on the stem card and {} \
             on the others",
            per_block,
            whole.devices[0].weights - per_block * n as u64,
            stem_forward,
            rest_forward,
        );
    }
}

/// The dry run against the REAL checkpoint, at the two sizes that broke.
///
/// Ignored because it needs the local Z-Image-Turbo snapshot; it needs no GPU
/// and reads no tensor data - only the safetensors headers - so it is safe to
/// run beside a live server:
///
/// ```text
/// cargo test --release -p loken --lib -- --ignored --nocapture \
///     zimage_native::tests::the_dry_run_weighs_the_real_checkpoint
/// ```
///
/// What it pins is the fault that started this: the loader answered "what does
/// this checkpoint weigh" from the config and said 15.10 GB where the shards go
/// resident at 12.31 GB, and the three gigabytes of difference split a model
/// that fits one card. Here the answer comes from the files' own headers at the
/// dtype the model loads, so there is nothing left to disagree with.
#[test]
#[ignore = "needs the local Z-Image-Turbo snapshot"]
fn the_dry_run_weighs_the_real_checkpoint() {
    let dir = crate::config::Config::load_test().get_hf_models_dir();
    let root = if dir.file_name().is_some_and(|n| n == "hub") {
        dir
    } else {
        dir.join("hub")
    };
    let snaps = root.join("models--Tongyi-MAI--Z-Image-Turbo/snapshots");
    let mut files: Vec<String> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&snaps) {
        for snap in rd.flatten() {
            let paths: Vec<std::path::PathBuf> = (1..=3)
                .map(|i| {
                    snap.path().join(format!(
                        "transformer/diffusion_pytorch_model-{i:05}-of-00003.safetensors"
                    ))
                })
                .collect();
            if paths.iter().all(|p| p.is_file()) {
                files = paths
                    .iter()
                    .map(|p| p.to_string_lossy().into_owned())
                    .collect();
                break;
            }
        }
    }
    assert!(
        !files.is_empty(),
        "no Z-Image-Turbo snapshot under {}",
        snaps.display()
    );
    let refs: Vec<&str> = files.iter().map(String::as_str).collect();
    let cfg = Config::z_image_turbo();
    let mut last = 0u64;
    for (w, h) in [(512usize, 512usize), (1024, 1024)] {
        let d = super::dry_forward(&cfg, &refs, None, DType::BF16, h / 8, w / 8, 256, None)
            .expect("dry run");
        // The forward is printed to the BYTE as well as to the tenth of a gigabyte:
        // it is the figure the Z-Image reserve is a multiple of, and a reserve
        // compared against the one it replaced cannot be checked from two decimals.
        println!(
            "z-image {w}x{h}: weights {:.2} GB, forward {:.2} GB ({} bytes) over {} \
             allocations{}",
            d.weights as f64 / 1e9,
            d.forward as f64 / 1e9,
            d.forward,
            d.allocations,
            if d.blind { " (read absent data)" } else { "" },
        );
        // The transformer is ~12.3 GB at BF16. The band is wide on purpose: this
        // catches "weighed from a config again", not a percentage.
        assert!(
            (11_000_000_000..14_000_000_000).contains(&d.weights),
            "the checkpoint weighs {:.2} GB, which is not what these shards hold",
            d.weights as f64 / 1e9
        );
        assert!(
            d.forward > 0,
            "a forward that allocates nothing was not run"
        );
        assert!(d.forward > last, "the demand did not grow with the request");
        last = d.forward;
    }
}

/// THE FIXED POINT, END TO END, ON THE REAL CHECKPOINT AND WITHOUT A CARD.
///
/// Every candidate placement is built and run against the production shards, on
/// devices that count instead of allocating, and the one that fits is chosen. What
/// it must not do at either size is put a block on the host while two cards of
/// 16.5 GB sit there - the spill that has no error message and that a request
/// waits out.
///
/// Ignored because it needs the local snapshot; it needs no GPU and reads no
/// tensor data, so it is safe to run beside a live server:
///
/// ```text
/// cargo test --release -p loken --lib -- --ignored --nocapture \
///     zimage_native::tests::the_fixed_point_places_the_real_checkpoint
/// ```
#[test]
#[ignore = "needs the local Z-Image-Turbo snapshot"]
fn the_fixed_point_places_the_real_checkpoint() {
    let files = real_checkpoint_shards();
    let refs: Vec<&str> = files.iter().map(String::as_str).collect();
    let cfg = Config::z_image_turbo();
    let cards = [(0usize, 16_500_000_000u64), (1usize, 16_500_000_000u64)];
    for (w, h) in [(512usize, 512usize), (1024, 1024)] {
        let mut measure = |plan: &HeteroPlan| {
            super::dry_forward_planned(
                &cfg,
                &refs,
                None,
                DType::BF16,
                DType::F32,
                h / 8,
                w / 8,
                256,
                plan,
                None,
            )
            .ok()
            .map(|d| d.load)
        };
        let s = crate::inference::place::dry_plan::solve(cfg.n_layers, &cards, 0, &mut measure)
            .expect("no placement for a checkpoint two 16.5 GB cards hold twice over");
        println!(
            "z-image {w}x{h}: {} holds {} ({} dry runs)",
            crate::inference::place::dry_plan::describe_plan(&s.plan),
            s.load.describe(),
            s.measurements,
        );
        assert!(
            !s.plan
                .segments
                .iter()
                .any(|g| matches!(g.kind, DeviceKind::Cpu)),
            "a block went to the host while both cards had room"
        );
        assert_eq!(
            s.plan.segments.len(),
            1,
            "a stack that fits one card was split across two"
        );
        assert_eq!(
            s.plan.segments[0].kind,
            DeviceKind::Cuda(0),
            "not the fastest card"
        );
    }
}

/// ...and when no single card holds it, the SPLIT is the measured one.
///
/// Same checkpoint, same geometry, cards too small to take it whole: the solver
/// re-derives the split from what each round measured until every card fits, and
/// still nothing reaches the host. This is the case the formula gets wrong in the
/// expensive direction - an over-stated reserve here does not fail, it spills.
///
/// ```text
/// cargo test --release -p loken --lib -- --ignored --nocapture \
///     zimage_native::tests::the_fixed_point_splits_the_real_checkpoint_when_no_card_holds_it
/// ```
#[test]
#[ignore = "needs the local Z-Image-Turbo snapshot"]
fn the_fixed_point_splits_the_real_checkpoint_when_no_card_holds_it() {
    let files = real_checkpoint_shards();
    let refs: Vec<&str> = files.iter().map(String::as_str).collect();
    let cfg = Config::z_image_turbo();
    let cards = [(0usize, 9_000_000_000u64), (1usize, 9_000_000_000u64)];
    let mut measure = |plan: &HeteroPlan| {
        super::dry_forward_planned(
            &cfg,
            &refs,
            None,
            DType::BF16,
            DType::F32,
            128,
            128,
            256,
            plan,
            None,
        )
        .ok()
        .map(|d| d.load)
    };
    let s = crate::inference::place::dry_plan::solve(cfg.n_layers, &cards, 0, &mut measure)
        .expect("two 9 GB cards hold a 12.31 GB stack between them and found nothing");
    println!(
        "z-image 1024x1024 on two 9 GB cards: {} holds {} ({} dry runs)",
        crate::inference::place::dry_plan::describe_plan(&s.plan),
        s.load.describe(),
        s.measurements,
    );
    assert_eq!(
        s.plan.segments.len(),
        2,
        "a split was needed and was not made"
    );
    assert!(!s
        .plan
        .segments
        .iter()
        .any(|g| matches!(g.kind, DeviceKind::Cpu)));
    assert_eq!(
        s.plan
            .segments
            .iter()
            .map(|g| g.layer_end - g.layer_start)
            .sum::<usize>(),
        cfg.n_layers,
        "the split lost blocks"
    );
}

/// THE DECISION THE LOADER MAKES AT 1536 SQUARE, on the real checkpoint, with the
/// production reserve - and without a card.
///
/// Everything here is the loader's own: the placed walk, the per-card reserve
/// (`zimage_placed_reserve`), the solver, and the rule that decides where the caption
/// encoder goes. What it pins is the render that died on a machine with two 16.4 GB
/// cards: a split that fits BOTH cards exists, and the encoder must not be the one
/// holding the six and a half gigabytes it needs.
///
/// It also ties the figures the placement tests in `image_engine` are written against
/// to the checkpoint they were read from - if the walk moves, this fails here rather
/// than agreeing with a stale constant.
///
/// ```text
/// cargo test --release -p loken --lib -- --ignored --nocapture \
///     zimage_native::tests::the_1536_decision_on_the_real_checkpoint
/// ```
#[test]
#[ignore = "needs the local Z-Image-Turbo snapshot"]
fn the_1536_decision_on_the_real_checkpoint() {
    use crate::inference::engine::image_engine::{zimage_encoder_card, zimage_placed_reserve};
    let files = real_checkpoint_shards();
    let refs: Vec<&str> = files.iter().map(String::as_str).collect();
    let cfg = Config::z_image_turbo();
    // What NVML read on the machine that reported it, and what its caption encoder
    // was measured resident at.
    const CARD: u64 = 16_400_000_000;
    const ENCODER: u64 = 6_500_000_000;
    let cards = [(0usize, CARD), (1usize, CARD)];
    let mut measure = |plan: &HeteroPlan| {
        super::dry_forward_planned(
            &cfg,
            &refs,
            None,
            DType::BF16,
            DType::F32,
            192,
            192,
            256,
            plan,
            None,
        )
        .ok()
        .map(|d| zimage_placed_reserve(&d.load, 1536, 1536))
    };
    let s = crate::inference::place::dry_plan::solve(cfg.n_layers, &cards, 0, &mut measure)
        .expect("32.8 GB of cards found no placement for a 19.3 GB request");
    println!(
        "z-image 1536x1536: {} holds {} ({} dry runs)",
        crate::inference::place::dry_plan::describe_plan(&s.plan),
        s.load.describe(),
        s.measurements,
    );
    assert!(!s
        .plan
        .segments
        .iter()
        .any(|g| matches!(g.kind, DeviceKind::Cpu)));
    assert_eq!(
        s.plan
            .segments
            .iter()
            .map(|g| g.layer_end - g.layer_start)
            .sum::<usize>(),
        cfg.n_layers,
        "the split lost blocks",
    );
    for d in &s.load.devices {
        let DeviceKind::Cuda(i) = d.kind else {
            panic!("a segment on {}", d.kind)
        };
        assert!(
            d.peak() <= CARD,
            "CUDA({i}) holds {} of blocks and needs {} free for its own forward, on a \
             card with {CARD}",
            d.weights,
            d.forward,
        );
    }
    // ...and the encoder gets what is left, which at this geometry is nothing: it
    // encodes on the host so that the blocks have the cards.
    assert_eq!(
        zimage_encoder_card(&cards, Some(&s.load), None, ENCODER),
        None,
        "the encoder was given a card the split needs",
    );
}

/// The production shards, wherever the local hub keeps them.
fn real_checkpoint_shards() -> Vec<String> {
    let dir = crate::config::Config::load_test().get_hf_models_dir();
    let root = if dir.file_name().is_some_and(|n| n == "hub") {
        dir
    } else {
        dir.join("hub")
    };
    let snaps = root.join("models--Tongyi-MAI--Z-Image-Turbo/snapshots");
    let mut files: Vec<String> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&snaps) {
        for snap in rd.flatten() {
            let paths: Vec<std::path::PathBuf> = (1..=3)
                .map(|i| {
                    snap.path().join(format!(
                        "transformer/diffusion_pytorch_model-{i:05}-of-00003.safetensors"
                    ))
                })
                .collect();
            if paths.iter().all(|p| p.is_file()) {
                files = paths
                    .iter()
                    .map(|p| p.to_string_lossy().into_owned())
                    .collect();
                break;
            }
        }
    }
    assert!(
        !files.is_empty(),
        "no Z-Image-Turbo snapshot under {}",
        snaps.display()
    );
    files
}

/// A Z-Image checkpoint small enough to live in a test, written at the names and
/// the shapes THIS FILE's constructors ask for.
///
/// The dry run only reads the header, but the file is written whole so the loader
/// exercised is the production one - a fixture that only a dry run could open
/// would be measuring itself.
fn tiny_checkpoint(cfg: &Config, path: &std::path::Path) {
    use safetensors::tensor::{Dtype, TensorView};
    let d = cfg.dim;
    let head = cfg.head_dim();
    let q_dim = cfg.n_heads * head;
    let kv_dim = cfg.n_kv_heads * head;
    let hidden = cfg.hidden_dim();
    // The two widths the checkpoint's shapes are written from, asked of the config rather
    // than recomputed here: a fixture that derives a shape its own way can only agree with
    // the loader by luck.
    let adaln = cfg.adaln_dim();
    let patch_dim = cfg.patch_dim();
    let mut shapes: Vec<(String, Vec<usize>)> = vec![
        (
            "t_embedder.mlp.0.weight".into(),
            vec![1024, FREQUENCY_EMBEDDING_SIZE],
        ),
        ("t_embedder.mlp.0.bias".into(), vec![1024]),
        ("t_embedder.mlp.2.weight".into(), vec![adaln, 1024]),
        ("t_embedder.mlp.2.bias".into(), vec![adaln]),
        ("cap_embedder.0.weight".into(), vec![cfg.cap_feat_dim]),
        ("cap_embedder.1.weight".into(), vec![d, cfg.cap_feat_dim]),
        ("cap_embedder.1.bias".into(), vec![d]),
        ("x_embedder.weight".into(), vec![d, patch_dim]),
        ("x_embedder.bias".into(), vec![d]),
        ("final_layer.linear.weight".into(), vec![patch_dim, d]),
        ("final_layer.linear.bias".into(), vec![patch_dim]),
        (
            "final_layer.adaLN_modulation.1.weight".into(),
            vec![d, adaln],
        ),
        ("final_layer.adaLN_modulation.1.bias".into(), vec![d]),
        ("x_pad_token".into(), vec![1, d]),
        ("cap_pad_token".into(), vec![1, d]),
    ];
    let block = |prefix: String, modulation: bool, shapes: &mut Vec<(String, Vec<usize>)>| {
        shapes.push((format!("{prefix}.attention.to_q.weight"), vec![q_dim, d]));
        shapes.push((format!("{prefix}.attention.to_k.weight"), vec![kv_dim, d]));
        shapes.push((format!("{prefix}.attention.to_v.weight"), vec![kv_dim, d]));
        shapes.push((
            format!("{prefix}.attention.to_out.0.weight"),
            vec![d, q_dim],
        ));
        if cfg.qk_norm {
            shapes.push((format!("{prefix}.attention.norm_q.weight"), vec![head]));
            shapes.push((format!("{prefix}.attention.norm_k.weight"), vec![head]));
        }
        shapes.push((format!("{prefix}.feed_forward.w1.weight"), vec![hidden, d]));
        shapes.push((format!("{prefix}.feed_forward.w2.weight"), vec![d, hidden]));
        shapes.push((format!("{prefix}.feed_forward.w3.weight"), vec![hidden, d]));
        for n in [
            "attention_norm1",
            "attention_norm2",
            "ffn_norm1",
            "ffn_norm2",
        ] {
            shapes.push((format!("{prefix}.{n}.weight"), vec![d]));
        }
        if modulation {
            shapes.push((
                format!("{prefix}.adaLN_modulation.0.weight"),
                vec![4 * d, adaln],
            ));
            shapes.push((format!("{prefix}.adaLN_modulation.0.bias"), vec![4 * d]));
        }
    };
    for i in 0..cfg.n_refiner_layers {
        block(format!("noise_refiner.{i}"), true, &mut shapes);
        block(format!("context_refiner.{i}"), false, &mut shapes);
    }
    for i in 0..cfg.n_layers {
        block(format!("layers.{i}"), true, &mut shapes);
    }
    let bufs: Vec<Vec<u8>> = shapes
        .iter()
        .map(|(_, s)| vec![0u8; s.iter().product::<usize>() * 4])
        .collect();
    let tensors: Vec<(String, TensorView<'_>)> = shapes
        .iter()
        .zip(bufs.iter())
        .map(|((name, dims), buf)| {
            (
                name.clone(),
                TensorView::new(Dtype::F32, dims.clone(), buf).unwrap(),
            )
        })
        .collect();
    safetensors::serialize_to_file(tensors, None, path).unwrap();
}

/// The same model at a size a test can run: the shape algebra is what is being
/// measured, and it does not need the production widths to be exercised.
fn tiny_config() -> Config {
    let mut cfg = Config::z_image_turbo();
    cfg.dim = 64;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 4;
    cfg.n_layers = 8;
    cfg.n_refiner_layers = 1;
    cfg.cap_feat_dim = 32;
    cfg.in_channels = 4;
    cfg.axes_dims = vec![4, 6, 6];
    cfg.axes_lens = vec![64, 32, 32];
    cfg.set_use_accelerated_attn(false);
    cfg
}

fn plan_of(spans: &[(usize, usize, usize)], total: usize) -> HeteroPlan {
    use crate::inference::place::layer_executor::HeteroSegment;
    HeteroPlan {
        segments: spans
            .iter()
            .map(|(idx, start, end)| HeteroSegment {
                kind: DeviceKind::Cuda(*idx),
                layer_start: *start,
                layer_end: *end,
                free_memory_bytes: 0,
            })
            .collect(),
        total_layers: total,
    }
}

/// THE COUNT IS PER CARD, AND IT FOLLOWS THE PLAN.
///
/// This is what makes a measured placement worth having. A total says nothing
/// about whether a placement runs: what decides is what the BUSIEST card holds,
/// and that is a different number for every arrangement of the same blocks. So the
/// same checkpoint, at the same geometry, is measured twice - whole on one card,
/// then halved across two - and the first card's figure has to come down. If it
/// did not, the plan was not a parameter of the measurement and choosing between
/// placements with it would be choosing at random.
#[test]
fn the_counted_peak_is_per_card_and_follows_the_plan() {
    let cfg = tiny_config();
    let dir = std::env::temp_dir().join(format!("zimage_dry_placement_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tiny.safetensors");
    tiny_checkpoint(&cfg, &path);
    let files = [path.to_str().unwrap()];
    let run = |plan: &HeteroPlan| {
        super::dry_forward_planned(
            &cfg,
            &files,
            None,
            DType::BF16,
            DType::F32,
            8,
            8,
            8,
            plan,
            None,
        )
        .expect("the placed dry run")
    };

    let whole = run(&plan_of(&[(0, 0, cfg.n_layers)], cfg.n_layers));
    let split = run(&plan_of(&[(0, 0, 4), (1, 4, cfg.n_layers)], cfg.n_layers));
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(
        whole.load.devices.len(),
        1,
        "one card in the plan, two in the answer"
    );
    assert_eq!(
        split.load.devices.len(),
        2,
        "two cards in the plan, {:?}",
        split.load
    );
    assert_eq!(whole.load.devices[0].kind, DeviceKind::Cuda(0));
    assert_eq!(split.load.devices[1].kind, DeviceKind::Cuda(1));

    assert!(
        whole.load.devices[0].forward > 0,
        "a forward that allocated nothing was not run"
    );
    assert!(
        split.load.devices[1].forward > 0,
        "the second card was given blocks and never ran one"
    );
    assert!(
        split.load.devices[0].weights < whole.load.devices[0].weights,
        "moving half the blocks off card 0 left its weights at {} against {}",
        split.load.devices[0].weights,
        whole.load.devices[0].weights
    );
    assert!(
        split.load.devices[0].peak() < whole.load.devices[0].peak(),
        "the first card's peak did not follow the plan: {} split against {} whole",
        split.load.devices[0].peak(),
        whole.load.devices[0].peak()
    );
    // The blocks did not evaporate on the way: what one card held is what two cards
    // hold between them, give or take the stem that stays where the plan starts.
    let together: u64 = split.load.devices.iter().map(|d| d.weights).sum();
    let alone = whole.load.devices[0].weights;
    assert!(
        together >= alone && together - alone < alone / 20,
        "the split holds {together} against {alone} whole - blocks were lost or duplicated"
    );
}

/// A block placed on the HOST weighs what the host loads it at.
///
/// The multi-device loader keeps card segments at the compute dtype and host
/// segments at F32, so the same block costs twice as much once it spills. A
/// measurement blind to that would rate the placement it exists to avoid as the
/// cheap one.
#[test]
fn a_block_on_the_host_is_weighed_at_the_host_dtype() {
    use crate::inference::place::layer_executor::HeteroSegment;
    let cfg = tiny_config();
    let dir = std::env::temp_dir().join(format!("zimage_dry_host_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tiny.safetensors");
    tiny_checkpoint(&cfg, &path);
    let files = [path.to_str().unwrap()];
    let spilled = HeteroPlan {
        segments: vec![
            HeteroSegment {
                kind: DeviceKind::Cuda(0),
                layer_start: 0,
                layer_end: 4,
                free_memory_bytes: 0,
            },
            HeteroSegment {
                kind: DeviceKind::Cpu,
                layer_start: 4,
                layer_end: cfg.n_layers,
                free_memory_bytes: 0,
            },
        ],
        total_layers: cfg.n_layers,
    };
    let weigh = |host_dtype| {
        super::dry_forward_planned(
            &cfg,
            &files,
            None,
            DType::BF16,
            host_dtype,
            8,
            8,
            8,
            &spilled,
            None,
        )
        .expect("the placed dry run")
        .load
    };
    let wide = weigh(DType::F32);
    let narrow = weigh(DType::BF16);
    let _ = std::fs::remove_dir_all(&dir);

    let host_wide = wide.on(DeviceKind::Cpu).expect("no host segment").weights;
    let host_narrow = narrow.on(DeviceKind::Cpu).expect("no host segment").weights;
    // The matmul weights double and the norms and biases do not - they load at
    // exact F32 either way - so the ratio lands under two and well over one.
    assert!(
        host_wide > host_narrow * 3 / 2 && host_wide <= 2 * host_narrow,
        "the same host blocks weigh {host_wide} at F32 against {host_narrow} at BF16 - \
         the host dtype is not the one the segment was loaded at"
    );
    // ...and only the host segment moved: the cards were asked for the same dtype
    // both times.
    assert_eq!(
        wide.on(DeviceKind::Cuda(0))
            .expect("no card segment")
            .weights,
        narrow
            .on(DeviceKind::Cuda(0))
            .expect("no card segment")
            .weights,
        "the host dtype changed what the CARD holds"
    );
}

fn data(n: usize, seed: u32) -> Vec<f32> {
    let mut st = seed.wrapping_mul(2654435761).wrapping_add(12345);
    (0..n)
        .map(|_| {
            st = st.wrapping_mul(1664525).wrapping_add(1013904223);
            ((st >> 8) as f32 / (1 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(
            (a - b).abs() <= tol,
            "{what}: idx {i}: native {a} vs facade {b}"
        );
    }
}

fn facade(data: Vec<f32>, dims: &[usize]) -> crate::tensor::Tensor {
    crate::tensor::Tensor::from_vec(data, dims, &crate::tensor::Device::Cpu).unwrap()
}

fn facade_vec(t: &crate::tensor::Tensor) -> Vec<f32> {
    t.contiguous()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

#[test]
fn tanh_matches_facade() {
    let vals = data(64, 11);
    let nt = Tensor::from_vec_f32(vals.clone(), 64).unwrap();
    let got = tanh(&nt).unwrap();
    let want = facade(vals, &[64]).tanh().unwrap();
    assert_close(&got.to_vec_f32(), &facade_vec(&want), 1e-5, "tanh");
}

#[test]
fn permute_matches_facade() {
    // The two 6D permutations patchify/unpatchify use.
    let dims = [2usize, 3, 2, 2, 3, 2];
    let n: usize = dims.iter().product();
    let vals = data(n, 21);
    let nt = Tensor::from_vec_f32(vals.clone(), dims.to_vec()).unwrap();
    let ft = facade(vals, &dims);

    for perm in [[0usize, 2, 4, 3, 5, 1], [0, 5, 1, 3, 2, 4]] {
        let got = permute(&nt, &perm).unwrap();
        let want = ft
            .permute((perm[0], perm[1], perm[2], perm[3], perm[4], perm[5]))
            .unwrap();
        assert_eq!(got.dims(), want.dims(), "permute {perm:?}: dims");
        assert_close(
            &got.to_vec_f32(),
            &facade_vec(&want),
            0.0,
            &format!("permute {perm:?}"),
        );
    }
}

#[test]
fn timestep_embedding_matches_facade() {
    // Oracle: the exact op sequence of the facade
    // TimestepEmbedder::timestep_embedding (incl. its cached freqs row).
    let tvals = vec![0.0f32, 0.25, 1.0, 567.0];
    let dim = 32usize;
    let half = dim / 2;

    let nt = Tensor::from_vec_f32(tvals.clone(), tvals.len()).unwrap();
    let got = timestep_embedding(&nt, dim).unwrap();
    assert_eq!(got.dims(), &[4, dim]);

    let dev = crate::tensor::Device::Cpu;
    let arange = crate::tensor::Tensor::arange_u32(0u32, half as u32)
        .unwrap()
        .to_device(&dev)
        .unwrap()
        .to_dtype(crate::tensor::DType::F32)
        .unwrap();
    let freqs = (arange * (-MAX_PERIOD.ln() / half as f64))
        .unwrap()
        .exp()
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let ft = facade(tvals, &[4]);
    let args = ft.unsqueeze(1).unwrap().broadcast_mul(&freqs).unwrap();
    let (ac, asn) = (args.cos().unwrap(), args.sin().unwrap());
    let want = crate::tensor::Tensor::cat(&[&ac, &asn], crate::tensor::D::Minus1).unwrap();
    assert_close(
        &got.to_vec_f32(),
        &facade_vec(&want),
        1e-4,
        "timestep_embedding",
    );
}

/// The grid is its own definition: row `f*H*W + h*W + w` holds `(t0+f, y0+h, x0+w)`.
///
/// The rotary tables are pinned instead of derived, because what they encode is a CHOICE  - 
/// which half of the head dimension each axis owns, and whether a pair is adjacent or split
/// across halves - and a test that re-derived that choice would agree with any convention the
/// implementation happened to adopt. Regenerate with the ignored `print_fixtures`.
#[test]
fn the_coordinate_grid_is_its_index_and_rope_holds_its_table() {
    let size = (2usize, 2, 2);
    let start = (3usize, 0, 0);
    let ids = create_coordinate_grid(size, start).unwrap();
    assert_eq!(ids.dims(), &[size.0 * size.1 * size.2, 3]);

    let got = ids.to_vec_f32();
    let mut want = Vec::new();
    for f in 0..size.0 {
        for h in 0..size.1 {
            for w in 0..size.2 {
                want.extend([
                    (start.0 + f) as f32,
                    (start.1 + h) as f32,
                    (start.2 + w) as f32,
                ]);
            }
        }
    }
    assert_close(&got, &want, 0.0, "coordinate grid");

    let rope = RopeEmbedder::new(256.0, vec![4, 6, 6], vec![16, 8, 8]).unwrap();
    let ids = create_coordinate_grid((2, 2, 2), (1, 0, 0)).unwrap();
    let (cos, sin) = rope.forward(&ids).unwrap();
    assert_eq!(cos.dims(), &[8, 8]);
    assert_close(&cos.to_vec_f32()[..16], &ROPE_COS_HEAD, 1e-6, "rope cos");
    assert_close(&sin.to_vec_f32()[..16], &ROPE_SIN_HEAD, 1e-6, "rope sin");
}

/// The first sixteen values of the rotary tables for axes (4, 6, 6) over a 2x2x2 grid
/// starting at (1, 0, 0), theta 256.
const ROPE_COS_HEAD: [f32; 16] = [
    0.5403023, 0.99804753, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5403023, 0.99804753, 1.0, 1.0, 1.0,
    0.5403023, 0.98762405, 0.99969244,
];
const ROPE_SIN_HEAD: [f32; 16] = [
    0.84147096,
    0.062459316,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.84147096,
    0.062459316,
    0.0,
    0.0,
    0.0,
    0.84147096,
    0.15683989,
    0.024800597,
];

/// The rotation applied to a known activation, pinned.
///
/// Which two components of the head dimension form a rotating pair is the same convention the
/// tables above encode; deriving it here would only restate the implementation.
#[test]
fn apply_rotary_emb_rotates_the_pairs_it_is_given() {
    let (b, seq, heads, head_dim) = (1usize, 8usize, 2usize, 16usize);
    let xv = data(b * seq * heads * head_dim, 31);
    let ids = create_coordinate_grid((2, 2, 2), (1, 0, 0)).unwrap();
    let rope = RopeEmbedder::new(256.0, vec![4, 6, 6], vec![16, 8, 8]).unwrap();
    let (cos, sin) = rope.forward(&ids).unwrap();
    let x = Tensor::from_vec_f32(xv, vec![b, seq, heads, head_dim]).unwrap();

    let got = apply_rotary_emb(&x, &cos, &sin).unwrap();
    assert_eq!(got.dims(), &[b, seq, heads, head_dim]);
    assert_close(
        &got.to_vec_f32()[..16],
        &ROTATED_HEAD,
        1e-6,
        "apply_rotary_emb",
    );
}

const ROTATED_HEAD: [f32; 16] = [
    0.8538139,
    -0.079625994,
    -0.9303434,
    -0.1435341,
    0.1710546,
    -0.8584944,
    -0.7361772,
    -0.8010596,
    0.79731035,
    -0.4276229,
    -0.55776286,
    0.33929837,
    -0.8103868,
    -0.56299675,
    0.3816414,
    0.24989057,
];

#[test]
fn attention_matches_facade() {
    // q/k/v: [b=1, h=2, l=4, d=8]; oracle = the facade attention_basic
    // op sequence (matmul, additive padding mask, softmax, matmul).
    let (b, h, l, d) = (1usize, 2usize, 4usize, 8usize);
    let q = data(b * h * l * d, 1);
    let k = data(b * h * l * d, 2);
    let v = data(b * h * l * d, 3);
    let scale = 1.0 / (d as f64).sqrt();

    let nq = Tensor::from_vec_f32(q.clone(), vec![b, h, l, d]).unwrap();
    let nk = Tensor::from_vec_f32(k.clone(), vec![b, h, l, d]).unwrap();
    let nv = Tensor::from_vec_f32(v.clone(), vec![b, h, l, d]).unwrap();
    let fq = facade(q, &[b, h, l, d]);
    let fk = facade(k, &[b, h, l, d]);
    let fv = facade(v, &[b, h, l, d]);

    let facade_attn = |mask: Option<&crate::tensor::Tensor>| {
        let mut aw = (fq.matmul(&fk.transpose(2, 3).unwrap()).unwrap() * scale).unwrap();
        if let Some(m) = mask {
            let m = m.unsqueeze(1).unwrap().unsqueeze(2).unwrap();
            let m = ((m - 1.0).unwrap() * 1e9).unwrap();
            aw = aw.broadcast_add(&m).unwrap();
        }
        let probs = crate::tensor::ops::softmax_last_dim(&aw).unwrap();
        probs.matmul(&fv).unwrap()
    };

    // No mask
    let got = attention(&nq, &nk, &nv, None, scale as f32).unwrap();
    let want = facade_attn(None);
    assert_close(
        &got.to_vec_f32(),
        &facade_vec(&want),
        1e-4,
        "attention no-mask",
    );

    // Padding mask: last query/key position padded out
    let mvals = vec![1f32, 1.0, 1.0, 0.0];
    let nm = Tensor::from_vec_f32(mvals.clone(), (b, l)).unwrap();
    let fm = facade(mvals, &[b, l]);
    let got = attention(&nq, &nk, &nv, Some(&nm), scale as f32).unwrap();
    let want = facade_attn(Some(&fm));
    assert_close(
        &got.to_vec_f32(),
        &facade_vec(&want),
        1e-4,
        "attention masked",
    );
}

/// Query-chunked attention must be bit-exact vs the single-shot path
/// (softmax normalizes per query row). Exercises uneven tail tiles
/// (seq=10, chunk=3) with and without a padding mask - the flux
/// `sdpa_query_tiling_is_bit_exact` recipe, plus the masked path the
/// facade never chunks.
#[test]
fn attention_query_chunking_is_bit_exact() {
    let (b, h, l, d) = (1usize, 2usize, 10usize, 8usize);
    let q = Tensor::from_vec_f32(data(b * h * l * d, 71), vec![b, h, l, d]).unwrap();
    let k = Tensor::from_vec_f32(data(b * h * l * d, 72), vec![b, h, l, d]).unwrap();
    let v = Tensor::from_vec_f32(data(b * h * l * d, 73), vec![b, h, l, d]).unwrap();
    let scale = 1.0 / (d as f64).sqrt();
    let mut mvals = vec![1f32; l];
    mvals[l - 2] = 0.0;
    mvals[l - 1] = 0.0;
    let m = Tensor::from_vec_f32(mvals, (b, l)).unwrap();

    for mask in [None, Some(&m)] {
        let single = attention_chunked(&q, &k, &v, mask, scale as f32, l).unwrap();
        let tiled = attention_chunked(&q, &k, &v, mask, scale as f32, 3).unwrap();
        assert_eq!(single.dims(), tiled.dims());
        assert_close(
            &tiled.to_vec_f32(),
            &single.to_vec_f32(),
            0.0,
            &format!("attention chunking (mask={})", mask.is_some()),
        );
    }
}

/// The dense (safetensors) QLinear arm must match the facade Linear
/// math: y = x @ W^T + b. CPU/F32 - the dtype casts are no-ops there;
/// on CUDA the same arm runs the weight-dtype cublas matmul.
#[test]
fn dense_qlinear_matches_facade_linear() {
    let (rows, in_dim, out_dim) = (5usize, 12usize, 7usize);
    let wv = data(out_dim * in_dim, 81);
    let bv = data(out_dim, 82);
    let xv = data(rows * in_dim, 83);

    let w = Tensor::from_vec_f32(wv.clone(), vec![out_dim, in_dim]).unwrap();
    let b = Tensor::from_vec_f32(bv.clone(), out_dim).unwrap();
    let lin = QLinear::new(
        Weight::Dense(
            crate::tensor::layer::Linear::new(w, None).unwrap(),
            DType::F32,
        ),
        Some(b),
        in_dim,
        out_dim,
    );
    let x = Tensor::from_vec_f32(xv.clone(), vec![1, rows, in_dim]).unwrap();
    let got = lin.forward(&x).unwrap();
    assert_eq!(got.dims(), &[1, rows, out_dim]);

    let fw = facade(wv, &[out_dim, in_dim]);
    let fb = facade(bv, &[out_dim]);
    let fx = facade(xv, &[rows, in_dim]);
    let want = fx
        .matmul(&fw.t().unwrap())
        .unwrap()
        .broadcast_add(&fb)
        .unwrap();
    assert_close(&got.to_vec_f32(), &facade_vec(&want), 1e-5, "dense qlinear");
}

/// Patching a single frame, and putting it back.
///
/// Two things are checked and they answer different questions. The round trip says the two
/// halves agree with EACH OTHER - a permutation applied and undone leaves no trace, so this
/// alone would pass on a wrong layout consistently applied. The pinned head says which layout:
/// how a patch orders its channels against its rows and columns is a choice, and the fixture
/// is what holds today's choice still. Regenerate with the ignored `print_fixtures`.
#[test]
fn a_patched_frame_returns_to_itself_in_the_layout_it_states() {
    let (b, c, f, h, w) = (1usize, 3usize, 1usize, 4usize, 6usize);
    let vals = data(b * c * f * h * w, 41);
    let x = Tensor::from_vec_f32(vals.clone(), vec![b, c, f, h, w]).unwrap();

    let (patched, size) = patchify(&x, 2, 1).unwrap();
    assert_eq!(size, (f, h, w));
    // (h/2)*(w/2) patches of c*1*2*2 values each.
    assert_eq!(patched.dims(), &[b, (h / 2) * (w / 2), c * 2 * 2]);
    assert_close(
        &patched.to_vec_f32()[..16],
        &PATCHED_FRAME_HEAD,
        0.0,
        "patchify",
    );

    let back = unpatchify(&patched, size, 2, 1, c).unwrap();
    assert_eq!(back.dims(), &[b, c, f, h, w]);
    assert_close(&back.to_vec_f32(), &vals, 0.0, "patchify round trip");
}

const PATCHED_FRAME_HEAD: [f32; 16] = [
    0.8309306,
    0.38822615,
    0.7150568,
    -0.63070035,
    0.63248396,
    0.4476602,
    -0.8419695,
    -0.26037347,
    0.0073139668,
    -0.63904583,
    0.39034688,
    0.9440931,
    0.08708763,
    -0.15637887,
    0.1817522,
    0.18745041,
];

/// The same, over two frames, which takes the general path rather than the single-frame one.
#[test]
fn a_patched_pair_of_frames_returns_to_itself_too() {
    let (b, c, f, h, w) = (1usize, 3usize, 2usize, 4usize, 6usize);
    let vals = data(b * c * f * h * w, 43);
    let x = Tensor::from_vec_f32(vals.clone(), vec![b, c, f, h, w]).unwrap();

    let (patched, size) = patchify(&x, 2, 2).unwrap();
    assert_eq!(size, (f, h, w));
    assert_eq!(
        patched.dims(),
        &[b, (f / 2) * (h / 2) * (w / 2), c * 2 * 2 * 2]
    );
    assert_close(
        &patched.to_vec_f32()[..16],
        &PATCHED_PAIR_HEAD,
        0.0,
        "patchify general",
    );

    // No round trip is asserted here, and that is a statement about the code rather than
    // about the test: `unpatchify` does not invert `patchify` when the frame patch is 2  - 
    // measured at 142 of 144 values, and identically in the port this replaced, so it is
    // inherited rather than introduced. Nothing shipped reaches it: `all_f_patch_size` is
    // `[1]` for every Z-Image configuration, which is the single-frame path above.
    let back = unpatchify(&patched, size, 2, 2, c).unwrap();
    assert_eq!(back.dims(), &[b, c, f, h, w]);
}

const PATCHED_PAIR_HEAD: [f32; 16] = [
    0.9182538,
    0.3368721,
    -0.4013517,
    0.62335825,
    0.9492481,
    -0.08430362,
    -0.63616705,
    -0.28256226,
    0.35544515,
    -0.59176743,
    0.38060868,
    -0.27570152,
    0.995456,
    0.59335697,
    0.71959996,
    0.018454432,
];

/// Normalisation with no learned scale or shift, against its own definition.
#[test]
fn layer_norm_with_no_affine_is_its_formula() {
    use crate::tensor::Module as _;
    let (b, l, d) = (2usize, 3usize, 16usize);
    let vals = data(b * l * d, 61);
    let t = Tensor::from_vec_f32(vals.clone(), vec![b, l, d]).unwrap();
    let got = crate::tensor::layer::layer_norm_no_affine(d, 1e-6, &crate::tensor::Device::Cpu)
        .unwrap()
        .forward(&t)
        .unwrap();

    let eps = 1e-6f32;
    let mut want = Vec::with_capacity(vals.len());
    for row in vals.chunks(d) {
        let mean = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + eps).sqrt();
        want.extend(row.iter().map(|v| (v - mean) * inv));
    }
    assert_close(&got.to_vec_f32(), &want, 1e-4, "layer_norm_no_affine");
}

#[test]
#[ignore = "prints the fixtures the parity tests pin; not an assertion"]
fn print_fixtures() {
    let theta = 256.0f64;
    let (ad, al) = (vec![4usize, 6, 6], vec![16usize, 8, 8]);
    let ids = create_coordinate_grid((2, 2, 2), (1, 0, 0)).unwrap();
    let rope = RopeEmbedder::new(theta, ad.clone(), al.clone()).unwrap();
    let (cos, sin) = rope.forward(&ids).unwrap();
    println!("ROPE_DIMS {:?}", cos.dims());
    println!("ROPE_COS {:?}", &cos.to_vec_f32()[..16]);
    println!("ROPE_SIN {:?}", &sin.to_vec_f32()[..16]);

    let xv = data(1 * 8 * 2 * 16, 31);
    let nx = Tensor::from_vec_f32(xv, vec![1, 8, 2, 16]).unwrap();
    let rot = apply_rotary_emb(&nx, &cos, &sin).unwrap();
    println!("ROT {:?}", &rot.to_vec_f32()[..16]);

    let vals = data(1 * 3 * 1 * 4 * 6, 41);
    let px = Tensor::from_vec_f32(vals, vec![1, 3, 1, 4, 6]).unwrap();
    let (p, size) = patchify(&px, 2, 1).unwrap();
    println!("PATCH_DIMS {:?} size {:?}", p.dims(), size);
    println!("PATCH {:?}", &p.to_vec_f32()[..16]);

    let vals2 = data(1 * 3 * 2 * 4 * 6, 43);
    let px2 = Tensor::from_vec_f32(vals2, vec![1, 3, 2, 4, 6]).unwrap();
    let (p2, size2) = patchify(&px2, 2, 2).unwrap();
    println!("PATCH2_DIMS {:?} size {:?}", p2.dims(), size2);
    println!("PATCH2 {:?}", &p2.to_vec_f32()[..16]);
}
