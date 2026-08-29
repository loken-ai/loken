use super::{
    describe_plan, solve, zimage_block_plan, zimage_encoder_card, zimage_placed_reserve,
    zimage_runtime_demand, zimage_runtime_demand_from_files, zruntime_demand_from_dry, DeviceKind,
    DeviceLoad, HeteroPlan, PlanLoad, FLUX_TEXT_TOKENS, FLUX_VAE_STRIDE, ZIMAGE_DRY_MARGIN_PERCENT,
};

/// WHAT THE DRY WALK COUNTS, to the byte, at the two geometries it has been run at
/// against the official checkpoint (`the_dry_run_weighs_the_real_checkpoint` prints
/// them). Written out because the checkpoint is not present in every environment this
/// test runs in, and because the reserve is these numbers times a factor - so a
/// change to either shows up here rather than on a card.
/// WHAT THE DRY WALK COUNTS at the four geometries it has been run at against the
/// official checkpoint, and WHAT A RENDER OF THAT SHAPE WAS MEASURED TO HOLD - the
/// denoise loop bracketed on the card itself, not a request peak with a modelled
/// figure subtracted from it.
///
/// Recorded to a hundredth of a gigabyte, which is what they were read to. They are
/// a TABLE, and the reserve is asserted against it below: the pair is what says
/// whether a margin covers a render, and neither number alone can.
const DRY_512: u64 = 400_000_000;
const DRY_768: u64 = 790_000_000;
const DRY_1024: u64 = 1_350_000_000;
const DRY_1536: u64 = 2_930_000_000;
const RENDER_512: u64 = 440_000_000;
const RENDER_768: u64 = 870_000_000;
const RENDER_1024: u64 = 1_610_000_000;
/// Per CARD, the geometry being split across two - which is what its own walk
/// charges each of them, so the pair compares like for like.
const RENDER_1536: u64 = 2_650_000_000;

/// What the official Z-Image-Turbo transformer HOLDS: 24.6 GB of F32 shards, built
/// at BF16. Written out because the checkpoint is not present in every environment
/// this test runs in, and the figure is what the placement turns on.
const RESIDENT: u64 = 12_780_000_000;
/// Main blocks, from the config this family ships.
const MAIN_LAYERS: usize = 30;
/// Two cards with nothing else on them, the topology the failure was reported on.
const CARD: u64 = 16_500_000_000;
/// What NVML read on the machine that reported the 1536-square spill: two cards, 16.4
/// GB free apiece, 32.8 GB between them for a 19.33 GB request.
const REPORTED_CARD: u64 = 16_400_000_000;

fn blocks_on_the_host(plan: &crate::inference::place::layer_executor::HeteroPlan) -> usize {
    plan.segments
        .iter()
        .filter(|s| matches!(s.kind, DeviceKind::Cpu))
        .map(|s| s.layer_end - s.layer_start)
        .sum()
}

/// THE DEFECT. A demand of seven gigabytes, learned from an exhaustion, was charged
/// to every card AND again as a per-layer cost - so a card holding a third of the
/// model paid it one and a third times. Sixteen blocks of thirty went to the host on
/// a machine whose two cards had room for all thirty, and a render that used to take
/// fifteen seconds stopped arriving at all.
///
/// A model that fits the cards belongs ON the cards, whatever the reserve; a
/// placement that puts half of it on the processor is not a graceful fallback, it is
/// a failure disguised as slowness.
#[test]
fn a_model_the_cards_can_hold_never_goes_to_the_host() {
    for demand in [
        // The seeded figure at 1024^2.
        zimage_runtime_demand(1024, 1024),
        // And the largest figure an exhaustion has been seen to teach it. Under the
        // double charge this alone put a block on the host; at twice it, ten.
        7_050_000_000,
    ] {
        let plan = zimage_block_plan(&[(0, CARD), (1, CARD)], &[], MAIN_LAYERS, RESIDENT, demand);
        assert_eq!(
            blocks_on_the_host(&plan),
            0,
            "{:.2} GB demand put {} of {MAIN_LAYERS} blocks on the host across two \
             {:.1} GB cards: {:?}",
            demand as f64 / 1e9,
            blocks_on_the_host(&plan),
            CARD as f64 / 1e9,
            plan.segments
                .iter()
                .map(|s| (s.kind, s.layer_start, s.layer_end))
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            plan.segments
                .iter()
                .map(|s| s.layer_end - s.layer_start)
                .sum::<usize>(),
            MAIN_LAYERS,
            "the plan lost blocks"
        );
    }
}

/// The reserve is what one FORWARD needs free on the card running a block. It does
/// not grow with how many blocks happen to sit there, so a card must be charged for
/// it exactly once - and the planner is the one place that does the charging.
#[test]
fn the_reserve_is_charged_once_per_card_not_once_per_block() {
    let demand = 7_050_000_000u64;
    let plan = zimage_block_plan(&[(0, CARD), (1, CARD)], &[], MAIN_LAYERS, RESIDENT, demand);
    for seg in &plan.segments {
        let DeviceKind::Cuda(_) = seg.kind else {
            continue;
        };
        let blocks = (seg.layer_end - seg.layer_start) as u64;
        // What the card was told it may spend on weights, and what those weights
        // weigh. The reserve is already out of the first, so the second must fit
        // inside it with nothing further deducted per block.
        let per_layer =
            RESIDENT.saturating_sub(super::zimage_primary_overhead(RESIDENT)) / MAIN_LAYERS as u64;
        assert!(
            blocks * per_layer <= seg.free_memory_bytes,
            "{:?} was given {blocks} blocks against a {:.2} GB budget",
            seg.kind,
            seg.free_memory_bytes as f64 / 1e9,
        );
    }
    // And the cards must have taken the whole model between them.
    assert_eq!(blocks_on_the_host(&plan), 0);
}

/// THE REPORTED RENDER: 1536 square on two cards, and a block on the PROCESSOR.
///
/// The request weighs 19.33 GB - 12.31 GB of weights beside a 7.02 GB forward - on a
/// machine holding 32.8 GB of cards. It was planned "multi-device (CUDA+CPU)" and
/// then died of exhaustion, and the arithmetic says why: the caption encoder takes
/// 6.5 GB off the second card, the stem 1.68 GB off the first, and then the reserve
/// is charged AGAIN to each card - 14 GB of headroom held back for a forward that
/// only ever runs on one card at a time. What is left is fifty megabytes short of
/// thirty blocks, so one goes to the host and the render stops arriving.
///
/// The cards decide, not the budgets: a card dipping into its own headroom can OOM,
/// which is visible and re-plannable, whereas a block on the host is a request that
/// never finishes at all.
#[test]
fn the_1536_render_keeps_every_block_on_the_cards() {
    // As the planner is handed them: the encoder is placed first and comes off the
    // second card, which is the loader's own deduction.
    let cards = [
        (0, REPORTED_CARD),
        (1, REPORTED_CARD - super::zimage_text_encoder_bytes()),
    ];
    for demand in [
        // The seeded figure at 1536 square, and what the walk asked on the machine
        // that reported this - the two agree to ten megabytes.
        zimage_runtime_demand(1536, 1536),
        7_020_000_000,
    ] {
        let plan = zimage_block_plan(&cards, &[], MAIN_LAYERS, RESIDENT, demand);
        assert_eq!(
            blocks_on_the_host(&plan),
            0,
            "a {:.2} GB forward beside {:.2} GB of weights put {} of {MAIN_LAYERS} blocks \
             on the host, with {:.1} GB of cards in the machine: {:?}",
            demand as f64 / 1e9,
            RESIDENT as f64 / 1e9,
            blocks_on_the_host(&plan),
            cards.iter().map(|(_, m)| *m).sum::<u64>() as f64 / 1e9,
            plan.segments
                .iter()
                .map(|s| (s.kind, s.layer_start, s.layer_end))
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            plan.segments
                .iter()
                .map(|s| s.layer_end - s.layer_start)
                .sum::<usize>(),
            MAIN_LAYERS,
            "the plan lost blocks"
        );
        // No card may be handed more than it physically holds.
        let per_block =
            RESIDENT.saturating_sub(super::zimage_primary_overhead(RESIDENT)) / MAIN_LAYERS as u64;
        for seg in &plan.segments {
            let DeviceKind::Cuda(i) = seg.kind else {
                continue;
            };
            let free = cards.iter().find(|(c, _)| *c == i).unwrap().1;
            let held = (seg.layer_end - seg.layer_start) as u64 * per_block
                + if i == cards[0].0 {
                    super::zimage_primary_overhead(RESIDENT)
                } else {
                    0
                };
            assert!(
                held <= free,
                "CUDA({i}) was given {:.2} GB of a {:.2} GB card",
                held as f64 / 1e9,
                free as f64 / 1e9,
            );
        }
    }
}

/// THE TWO SIZES THAT WORK, BLOCK FOR BLOCK.
///
/// 512 and 1024 square render on ONE card today, in about 3 s and 55 s, and the
/// spill fix above is one edit away from moving them. Both halves of what makes them
/// single-card are pinned: the loader's gate (a card holds the weights AND the
/// forward, so the block planner is never reached) and, should it be reached
/// anyway, a plan of exactly one segment covering every block.
#[test]
fn the_working_sizes_still_place_on_one_card() {
    for (w, h) in [(512, 512), (1024, 1024)] {
        let demand = zimage_runtime_demand(w, h);
        assert!(
            RESIDENT + demand <= REPORTED_CARD,
            "at {w}x{h} one card no longer holds the model and its forward: {:.2} GB \
             against {:.2} GB",
            (RESIDENT + demand) as f64 / 1e9,
            REPORTED_CARD as f64 / 1e9,
        );
        let plan = zimage_block_plan(
            &[
                (0, REPORTED_CARD),
                (1, REPORTED_CARD - super::zimage_text_encoder_bytes()),
            ],
            &[],
            MAIN_LAYERS,
            RESIDENT,
            demand,
        );
        assert_eq!(
            plan.segments.len(),
            1,
            "at {w}x{h} the plan is no longer a single segment: {:?}",
            plan.segments
                .iter()
                .map(|s| (s.kind, s.layer_start, s.layer_end))
                .collect::<Vec<_>>(),
        );
        assert_eq!(plan.segments[0].kind, DeviceKind::Cuda(0));
        assert_eq!(plan.segments[0].layer_start, 0);
        assert_eq!(plan.segments[0].layer_end, MAIN_LAYERS);
    }
}

/// WHAT THE WALK COUNTS PER CARD, to the byte, against the official checkpoint
/// (`zimage_native::tests::the_walk_counts_the_real_checkpoint_card_by_card` prints
/// them). Written out because the checkpoint is not present in every environment this
/// test runs in, and to the byte because what they decide turns on tens of megabytes.
///
/// TWO forward figures per geometry, and they are the whole subject: the card the
/// plan STARTS on carries the stem's transients as well as its blocks', so it holds
/// more of a forward than a card carrying blocks alone. NEITHER figure moves with the
/// number of blocks the card was given - a forward is a property of the request and
/// the stream visits every card - which is what lets any split be rebuilt here from
/// two numbers.
const WALK_1024_STEM_CARD: u64 = 1_346_546_692;
const WALK_1024_OTHER_CARD: u64 = 1_276_406_784;
const WALK_1536_STEM_CARD: u64 = 2_926_025_732;
const WALK_1536_OTHER_CARD: u64 = 2_773_289_984;
/// What one main block weighs on a card, and what the stem weighs beside them: the
/// embedders, the refiners and the way back out, which the loader builds on the card
/// the plan starts on whatever it says about the main stack.
/// What a card HOLDS for them, which is what the placement is charged: the pool
/// reserves in chunks and the load's own staging leaves holes in them, so the counted
/// figures (361_882_624 and 1_455_429_888) are not what the card gives up.
const WALK_BLOCK: u64 = 369_098_752;
const WALK_STEM: u64 = 1_711_276_032;
/// What the caption encoder takes off a card it is placed on.
const ENCODER: u64 = 6_500_000_000;

/// The walk's answer for a candidate placement, rebuilt from the figures above and
/// put through the PRODUCTION reserve rule - so what this hands the solver is what a
/// machine hands it, arithmetic and margin included.
fn walked(
    width: usize,
    height: usize,
    stem_forward: u64,
    other_forward: u64,
) -> impl FnMut(&HeteroPlan) -> Option<PlanLoad> {
    move |plan: &HeteroPlan| {
        let devices = plan
            .segments
            .iter()
            .enumerate()
            .map(|(n, s)| DeviceLoad {
                kind: s.kind,
                weights: WALK_BLOCK * (s.layer_end - s.layer_start) as u64
                    + if n == 0 { WALK_STEM } else { 0 },
                forward: if n == 0 { stem_forward } else { other_forward },
            })
            .collect();
        Some(zimage_placed_reserve(&PlanLoad { devices }, width, height))
    }
}

/// The walk at 1536 square, the geometry that died.
fn walked_1536() -> impl FnMut(&HeteroPlan) -> Option<PlanLoad> {
    walked(1536, 1536, WALK_1536_STEM_CARD, WALK_1536_OTHER_CARD)
}

/// THE INVARIANT THAT WOULD HAVE CAUGHT IT: every card the retained placement names
/// holds ITS blocks and ITS OWN forward at the margin, inside what it has free.
///
/// No single reserve can satisfy that, and the failure says why. The plan taken at
/// 1536 square named both cards and put nothing on the host, which is what the last
/// fix was for - and the render still died, because the card holding twenty-two of
/// the thirty blocks had been filled until only a WHOLE model's forward would have
/// fitted beside it. The reserve was measured on a card carrying everything and then
/// charged to a card carrying part.
#[test]
fn the_1536_placement_fits_every_card_it_names() {
    let cards = [(0usize, REPORTED_CARD), (1usize, REPORTED_CARD)];
    let mut m = walked_1536();
    let s = solve(MAIN_LAYERS, &cards, 0, &mut m)
        .expect("32.8 GB of cards found no placement for a 19.3 GB request");
    assert!(
        !s.plan
            .segments
            .iter()
            .any(|g| matches!(g.kind, DeviceKind::Cpu)),
        "a block went to the host: {}",
        describe_plan(&s.plan),
    );
    assert_eq!(
        s.plan
            .segments
            .iter()
            .map(|g| g.layer_end - g.layer_start)
            .sum::<usize>(),
        MAIN_LAYERS,
        "the placement lost blocks: {}",
        describe_plan(&s.plan),
    );
    for d in &s.load.devices {
        let DeviceKind::Cuda(i) = d.kind else {
            panic!("a segment on {}", d.kind)
        };
        let free = cards.iter().find(|(c, _)| *c == i).unwrap().1;
        assert!(
            d.peak() <= free,
            "CUDA({i}) was given {:.2} GB of blocks and needs {:.2} GB free for the \
             forward IT runs, on a card holding {:.2} GB: {}",
            d.weights as f64 / 1e9,
            d.forward as f64 / 1e9,
            free as f64 / 1e9,
            describe_plan(&s.plan),
        );
    }
}

/// THE MECHANISM, not the answer: two cards carrying different work are charged
/// different reserves.
///
/// If both came back with the same figure the reserve would not be following the
/// plan, whatever else the placement got right - and every property above would hold
/// by accident.
#[test]
fn cards_carrying_different_work_get_different_reserves() {
    // Cards too small to take the stack whole and large enough to take half of it
    // beside the forward that half runs - which is a different pair of numbers at
    // each geometry, because the forward is what the geometry changes.
    for (w, h, stem, other, card) in [
        (
            1024,
            1024,
            WALK_1024_STEM_CARD,
            WALK_1024_OTHER_CARD,
            12_000_000_000u64,
        ),
        (
            1536,
            1536,
            WALK_1536_STEM_CARD,
            WALK_1536_OTHER_CARD,
            REPORTED_CARD,
        ),
    ] {
        let cards = [(0usize, card), (1usize, card)];
        let mut m = walked(w, h, stem, other);
        let s = solve(MAIN_LAYERS, &cards, 0, &mut m).expect(
            "two cards that hold half the stack apiece, beside its forward, found \
             no split",
        );
        assert_eq!(
            s.plan.segments.len(),
            2,
            "the split that was needed was not made"
        );
        assert!(
            s.load.devices[0].forward > s.load.devices[1].forward,
            "at {w}x{h} the two cards were charged {} and {} - one reserve for two \
             different forwards is the defect, not the fix",
            s.load.devices[0].forward,
            s.load.devices[1].forward,
        );
    }
}

/// THE SIZES THAT WORK DO NOT MOVE: same plan, same reserve, to the byte.
///
/// 512 and 1024 square render on one card, in about 3 s and 55 s. What the
/// measurement retains at those geometries is a placement of ONE segment covering
/// every block on the fastest card - and because a one-segment placement counts one
/// card carrying everything, the reserve it leaves is the figure this loader was
/// already planning against, value for value. That is the whole safety of the switch.
#[test]
fn the_working_sizes_keep_their_plan_and_their_reserve() {
    for (w, h, dry) in [(512, 512, DRY_512), (1024, 1024, DRY_1024)] {
        let cards = [(0usize, REPORTED_CARD), (1usize, REPORTED_CARD)];
        // At one segment the second figure is never read; a card that holds no block
        // runs no forward.
        let mut m = walked(w, h, dry, dry);
        let s = solve(MAIN_LAYERS, &cards, 0, &mut m)
            .expect("a size that renders on one card found no placement");
        assert_eq!(
            s.plan.segments.len(),
            1,
            "at {w}x{h} the placement is no longer one segment: {}",
            describe_plan(&s.plan),
        );
        assert_eq!(
            s.plan.segments[0].kind,
            DeviceKind::Cuda(0),
            "not the fastest card"
        );
        assert_eq!(s.plan.segments[0].layer_start, 0);
        assert_eq!(s.plan.segments[0].layer_end, MAIN_LAYERS);
        assert_eq!(
            s.load.devices[0].forward,
            zruntime_demand_from_dry(dry, w, h),
            "at {w}x{h} the card is being asked to hold back something other than the \
             figure this family has been planning against",
        );
    }
}

/// The walked figure for a whole card IS this family's recorded one, so the constants
/// above and the ones the reserve is built from cannot drift apart. Held to a
/// hundredth of a gigabyte, which is the precision the measured column was read to.
#[test]
fn the_whole_card_walk_is_the_recorded_one() {
    for (walked, recorded) in [
        (WALK_1024_STEM_CARD, DRY_1024),
        (WALK_1536_STEM_CARD, DRY_1536),
    ] {
        assert!(
            walked.abs_diff(recorded) * 100 < recorded,
            "the walked figure {walked} and the recorded one {recorded} are more than a \
             percent apart - the table and the walk have drifted"
        );
    }
}

/// A CARD GIVEN AWAY IS A CARD THE SPLIT CANNOT HAVE - and the measurement says how
/// much that costs instead of refusing outright.
///
/// Same request, same two cards, except that 6.5 GB of the second has gone to the
/// caption encoder before the transformer was placed. Under the round margin there
/// was then NO arrangement of thirty blocks at all. Under the measured one there is:
/// the blocks shift onto the card that still has room, and every card in it holds its
/// own blocks beside its own forward. What the measurement buys is not a bigger card,
/// it is the difference between "this does not fit" and "this fits like so".
#[test]
fn a_card_given_to_the_encoder_still_has_to_hold_what_it_is_given() {
    let cards = [(0usize, REPORTED_CARD), (1usize, REPORTED_CARD - ENCODER)];
    let mut m = walked_1536();
    let s = solve(MAIN_LAYERS, &cards, 0, &mut m)
        .expect("no arrangement, where the measured reserve leaves room for one");
    assert_eq!(blocks_on_the_host(&s.plan), 0, "a block went to the host");
    for d in &s.load.devices {
        let DeviceKind::Cuda(idx) = d.kind else {
            panic!("a non-CUDA segment")
        };
        let free = cards
            .iter()
            .find(|(i, _)| *i == idx)
            .expect("a card not offered")
            .1;
        assert!(
            d.peak() <= free,
            "{} holds {:.2} GB of the {:.2} GB it has free",
            describe_plan(&s.plan),
            d.peak() as f64 / 1e9,
            free as f64 / 1e9,
        );
    }
}

/// THE ENCODER TAKES WHAT THE PLACEMENT LEAVES, and at the working sizes that is the
/// same card it has always taken.
///
/// At 512 and 1024 square the transformer sits whole on the first card and the second
/// is untouched, so the encoder goes there and the render is unchanged.
///
/// AT 1536 IT GOES TO THE HOST, and this test used to assert the opposite.
///
/// The arithmetic said it fitted beside the split - subtract what the plan holds on the
/// second card and 0.21 GB was left over the encoder's 6.5 - and the render died in the
/// first attention GEMM. A remainder is where the error of every term above it lands,
/// and this remainder was smaller than the 0.48 GB the same geometry varies by between
/// identical runs. So a card the plan RUNS is not a card with room, whatever the
/// subtraction says: what happens once per request does not share with what happens on
/// every step of every image.
///
/// Six seconds of host encode, against a render that returned 500.
#[test]
fn the_caption_encoder_takes_only_what_the_placement_leaves() {
    let cards = [(0usize, REPORTED_CARD), (1usize, REPORTED_CARD)];
    for (w, h, dry) in [(512, 512, DRY_512), (1024, 1024, DRY_1024)] {
        let mut m = walked(w, h, dry, dry);
        let s = solve(MAIN_LAYERS, &cards, 0, &mut m).expect("no placement");
        assert_eq!(
            zimage_encoder_card(&cards, Some(&s.load), Some(0), ENCODER),
            Some(1),
            "at {w}x{h} the encoder no longer goes to the card the transformer leaves \
             alone",
        );
    }
    let mut m = walked_1536();
    let s = solve(MAIN_LAYERS, &cards, 0, &mut m).expect("no placement at 1536 square");
    // The split names both cards, so neither is free of it.
    assert_eq!(s.plan.segments.len(), 2, "1536 square stopped splitting");
    let card = zimage_encoder_card(&cards, Some(&s.load), None, ENCODER);
    assert_eq!(
        card,
        None,
        "the encoder was given a card that runs blocks every step: {} holds {}",
        describe_plan(&s.plan),
        s.load.describe(),
    );
    // And the reason the subtraction is not enough, in numbers: it says yes.
    let held = s
        .load
        .on(DeviceKind::Cuda(1))
        .expect("no second card")
        .peak();
    assert!(
        held + ENCODER <= REPORTED_CARD,
        "this test no longer pins what it was written for - the remainder used to be \
         positive ({held} + {ENCODER} against {REPORTED_CARD}), which is why an \
         arrangement that could not run was accepted",
    );
}

/// WITHOUT A MEASUREMENT NOTHING MOVES.
///
/// A checkpoint the walk cannot open, a layout it cannot follow, an operation it does
/// not cover: the solver has nothing to say, and both decisions it feeds fall back to
/// the rules that were there before - the encoder to "the roomiest card the
/// transformer does not take whole", the blocks to the planner and its single
/// reserve. Losing the measurement is allowed to lose the improvement, never to
/// change the answer.
#[test]
fn without_a_measurement_both_decisions_are_the_ones_they_were() {
    let cards = [(0usize, REPORTED_CARD), (1usize, REPORTED_CARD)];
    let mut nothing = |_: &HeteroPlan| None;
    assert!(solve(MAIN_LAYERS, &cards, 0, &mut nothing).is_none());
    assert_eq!(zimage_encoder_card(&cards, None, Some(0), ENCODER), Some(1));
    assert_eq!(zimage_encoder_card(&cards, None, Some(1), ENCODER), Some(0));
    // One card is the transformer's and the encoder does not compete with it there.
    assert_eq!(
        zimage_encoder_card(&cards[..1], None, Some(0), ENCODER),
        None
    );
    // And the plan the loader then takes is the planner's, block for block.
    let plan = zimage_block_plan(
        &[(0, REPORTED_CARD), (1, REPORTED_CARD - ENCODER)],
        &[],
        MAIN_LAYERS,
        RESIDENT,
        zimage_runtime_demand(1024, 1024),
    );
    assert_eq!(plan.segments.len(), 1, "{}", describe_plan(&plan));
    assert_eq!(plan.segments[0].layer_end, MAIN_LAYERS);
}

/// A card that genuinely cannot hold the model still spills - the guarantee is "not
/// while there is room", not "never". Without this the fix above could have been a
/// planner that ignores the reserve entirely.
#[test]
fn a_machine_without_the_room_still_spills() {
    let plan = zimage_block_plan(
        &[(0, 4_000_000_000)],
        &[],
        MAIN_LAYERS,
        RESIDENT,
        zimage_runtime_demand(1024, 1024),
    );
    assert!(
        blocks_on_the_host(&plan) > 0,
        "a 4 GB card cannot hold a 12 GB model and a 3 GB forward"
    );
}

/// 512x512 WORKS and must not move by a byte.
///
/// This is the half of the report that was fine - 2.94 s, unchanged since June - and
/// every fix for the broken size is one edit away from breaking it. The figure below
/// is the seeded 3 GiB scaled by this request's share of the reference token count;
/// it is pinned, not recomputed, so a change to the scaling shows up HERE rather than
/// as a render that stops arriving.
#[test]
fn the_working_size_does_not_move() {
    assert_eq!(zimage_runtime_demand(512, 512), 947_419_256);
}

/// THE MARGIN COVERS BOTH RENDERS THAT WERE MEASURED, and the numbers say by how
/// much.
///
/// THE GUARD ON THE READERS. Every decision that turns on "does this fit one card"
/// must read the MEASUREMENT when there is one, and none of them may fall back to a
/// figure that weighs the model differently.
///
/// This is the third time in one pass that a reader was found after the fact, and it
/// is the one that a test can hold: the solver had split a 1536-square request across
/// two cards, and a second reader - comparing a file-size estimate against free VRAM -
/// sent it to one card anyway, because the estimate weighs the weights 0.47 GB lighter
/// than the allocator holds them. The render then ran with 0.36 GB to spare where the
/// measurement had asked for two cards.
///
/// The property is not "the readers agree": they cannot, they read different things.
/// It is that when a measurement exists it is the only thing read.
#[test]
fn a_measured_split_is_never_overruled_by_an_estimate() {
    // A card that the COUNTED weights fit and the HELD weights do not - the gap the
    // estimate cannot see.
    let card = RESIDENT + (DRY_1536 * ZIMAGE_DRY_MARGIN_PERCENT / 100) - 200_000_000;
    let cards = [(0usize, card), (1usize, card)];
    let mut m = walked_1536();
    let split = solve(MAIN_LAYERS, &cards, 0, &mut m).expect("no placement at 1536 square");
    assert!(
        split.plan.segments.len() > 1,
        "this fixture is meant to be a request no single card holds, and it holds: {}",
        describe_plan(&split.plan),
    );
    // The estimate on its own would say one card is enough...
    let counted = 12_309_845_444u64;
    let headroom = zruntime_demand_from_dry(DRY_1536, 1536, 1536);
    assert!(
        cards.iter().any(|(_, m)| *m >= counted + headroom),
        "the fixture no longer exercises the disagreement it was built for",
    );
    // ...and the decision must not listen to it.
    assert!(
        !super::zimage_fits_one_card(Some(&split), &cards, counted, headroom),
        "a measured split was overruled by an estimate that weighs the model lighter",
    );
    assert_eq!(
        super::zimage_transformer_card(Some(&split), &cards, counted, headroom),
        Some(0),
        "the transformer was sent to a card the measured plan does not start on",
    );
}

/// THE LAST LINK. A plan of two segments must not produce a single-card load, however
/// light a downstream reader decides the model is.
///
/// Deciding that a request needs two cards and then loading it onto one is not a
/// milder version of the same behaviour: the decision said it does not fit and the
/// loader fitted it anyway, with whatever was left over standing in for the margin.
/// This is the fixture from the reader guard above - a card the counted weights fit
/// and the held weights do not - asked of the function the loader actually branches
/// on.
#[test]
fn a_measured_split_never_produces_a_single_card_load() {
    let card = RESIDENT + (DRY_1536 * ZIMAGE_DRY_MARGIN_PERCENT / 100) - 200_000_000;
    let cards = [(0usize, card), (1usize, card)];
    let mut m = walked_1536();
    let split = solve(MAIN_LAYERS, &cards, 0, &mut m).expect("no placement at 1536 square");
    assert!(
        split.plan.segments.len() > 1,
        "the fixture stopped being a split"
    );
    let counted = 12_309_845_444u64;
    let headroom = zruntime_demand_from_dry(DRY_1536, 1536, 1536);
    assert_eq!(
        super::zimage_whole_card(Some(&split), &cards, counted, headroom, true),
        None,
        "the loader was handed a single card for a placement the measurement split",
    );
}

/// ...and a plan of ONE segment loads on the card that segment names, which is what
/// the working sizes do and must keep doing.
#[test]
fn a_measured_whole_loads_on_the_card_the_plan_names() {
    let cards = [(0usize, 4_000_000_000u64), (1usize, 40_000_000_000)];
    let mut m = walked(1024, 1024, DRY_1024, DRY_1024);
    let whole = solve(MAIN_LAYERS, &cards, 0, &mut m).expect("no placement at 1024 square");
    assert_eq!(
        whole.plan.segments.len(),
        1,
        "the fixture stopped being a whole placement"
    );
    assert_eq!(
        super::zimage_whole_card(Some(&whole), &cards, RESIDENT, 1, true),
        Some(1),
        "the transformer did not follow the card its own plan names",
    );
    // ...and never a card at all when there is no CUDA to put it on.
    assert_eq!(
        super::zimage_whole_card(Some(&whole), &cards, RESIDENT, 1, false),
        None
    );
}

/// ...and with nothing measured, both answers are the ones this family always gave.
#[test]
fn without_a_measurement_the_estimate_still_decides() {
    let cards = [(0usize, 8_000_000_000u64), (1usize, 20_000_000_000)];
    let (est, headroom) = (12_000_000_000u64, 1_000_000_000u64);
    assert!(super::zimage_fits_one_card(None, &cards, est, headroom));
    assert_eq!(
        super::zimage_transformer_card(None, &cards, est, headroom),
        Some(1)
    );
    assert!(!super::zimage_fits_one_card(
        None,
        &cards,
        est,
        9_000_000_000
    ));
}

/// THE GUARD ON THE MARGIN. At every geometry that has been measured, what the
/// placement is charged must stay above what the card was seen to give up.
///
/// The two sides are charged separately and only their SUM protects a placement: the
/// weights are charged at what the allocator holds for them, and the forward at the
/// walked figure times the margin. This asserts the forward half, which is the half
/// a margin can move; the weights half is exact by construction now that it is
/// replayed rather than counted.
///
/// If this fails, the margin has been lowered below a measurement. The answer is
/// another measurement, not a smaller number.
#[test]
fn the_margin_covers_every_measured_render() {
    let table = [
        (DRY_512, RENDER_512, 512, 512),
        (DRY_768, RENDER_768, 768, 768),
        (DRY_1024, RENDER_1024, 1024, 1024),
        (DRY_1536, RENDER_1536, 1536, 1536),
    ];
    for (dry, render, w, h) in table {
        let asked = zruntime_demand_from_dry(dry, w, h);
        assert!(
            asked >= render,
            "at {w}x{h} the reserve is {:.2} GB and the denoise was MEASURED to hold \
             {:.2} GB - the margin has been lowered below what was measured",
            asked as f64 / 1e9,
            render as f64 / 1e9,
        );
    }
}

/// ...and the margin is not larger than the measurements need either. Six times the
/// worst measured error is not caution, it is a second model that cannot be resident
/// and a placement that splits when one card would have carried it.
#[test]
fn the_margin_is_not_wider_than_the_measurements_ask() {
    let worst = [
        (DRY_512, RENDER_512),
        (DRY_768, RENDER_768),
        (DRY_1024, RENDER_1024),
        (DRY_1536, RENDER_1536),
    ]
    .iter()
    .map(|(dry, render)| render * 100 / dry)
    .max()
    .expect("a table with no rows");
    assert!(
        ZIMAGE_DRY_MARGIN_PERCENT >= worst,
        "the margin is {ZIMAGE_DRY_MARGIN_PERCENT} percent and the worst measured \
         render needed {worst}"
    );
    assert!(
        ZIMAGE_DRY_MARGIN_PERCENT <= worst + 25,
        "the margin is {ZIMAGE_DRY_MARGIN_PERCENT} percent against a worst measured \
         {worst} - that is head-room nothing measured asks for"
    );
}

/// THE DEMAND IS NO LONGER LINEAR IN TOKENS - which is what says the walk is the
/// thing deciding.
///
/// The seeded formula scales one figure by the token count, so its answer at 1024
/// square is its answer at 512 times the token ratio, WHATEVER THE SEED. The walked
/// forward does not grow that way: 3.38 where the tokens grow by 3.40 (the image
/// tokens go up fourfold, the 256 text tokens do not go up at all). The gap is
/// small - and it is the whole point, because it is the one thing no choice of seed
/// can express, which is why the formula had to be replaced rather than retuned.
#[test]
fn the_demand_no_longer_scales_with_the_token_count() {
    use crate::inference::place::runtime_demand::latent_tokens;
    let patch = crate::inference::model::zimage::dit::Config::z_image_turbo()
        .all_patch_size
        .first()
        .copied()
        .unwrap_or(1);
    let tokens = |side: usize| {
        (latent_tokens(side, side, FLUX_VAE_STRIDE, patch) + FLUX_TEXT_TOKENS) as u128
    };
    let small = zruntime_demand_from_dry(DRY_512, 512, 512);
    let large = zruntime_demand_from_dry(DRY_1024, 1024, 1024);
    // What a token-linear rule would ask at 1024, having asked `small` at 512.
    let linear = (small as u128 * tokens(1024) / tokens(512)) as u64;
    assert!(
        large < linear,
        "the reserve at 1024 square is {large}, and a rule linear in tokens would ask \
         {linear} - the two agree, so the token scaling is still what decides"
    );
    // And the ratio it DOES follow is the walked forward's, to the percent.
    assert_eq!(large * 100 / small, DRY_1024 * 100 / DRY_512);
    assert_ne!(
        large * 100 / small,
        (tokens(1024) * 100 / tokens(512)) as u64,
        "the reserve still grows exactly as the tokens do"
    );
}

/// A CHECKPOINT THE WALK CANNOT READ KEEPS THE SEEDED FIGURE, value for value.
///
/// Files that are not there, a layout the dry device cannot follow, an operation it
/// does not cover: none of them may make a render ask for LESS than the figure this
/// family planned against before there was anything to measure. Losing the
/// measurement is allowed to lose the improvement, never the guarantee.
#[test]
fn a_checkpoint_the_walk_cannot_read_falls_back_to_the_seeded_figure() {
    for (w, h) in [(512, 512), (1024, 1024), (1024, 1536)] {
        assert_eq!(
            zimage_runtime_demand_from_files(
                &["/nonexistent/z-image/transformer.safetensors"],
                None,
                w,
                h
            ),
            zimage_runtime_demand(w, h),
            "a checkpoint that cannot be walked changed the reserve at {w}x{h}"
        );
    }
}

/// The demand must COVER what a render of that shape was measured to need.
///
/// A shape that exhausts a card leaves behind a lower bound - "even this much was not
/// enough" - and the whole point of recording it is that the next placement asks for
/// at least that. This family recorded them and then planned against the estimate
/// anyway on one of its two paths, so the gate that admits a render kept quoting the
/// figure that admitted the failure.
///
/// That there is now only ONE path for it to raise is the media-side test
/// `the_zimage_reserve_is_the_figure_its_loader_plans_against`.
#[test]
fn a_measured_peak_raises_the_demand() {
    // A geometry of its own: the store is process-wide and keyed by shape, so
    // recording against a size other tests read would decide their answers too.
    let (w, h) = (1024, 1024 + 64);
    let seeded = zimage_runtime_demand(w, h);
    let measured = seeded + (seeded / 2);
    crate::inference::place::runtime_demand::record_observed_peak("zimage", w, h, measured);
    assert!(
        zimage_runtime_demand(w, h) >= measured,
        "the demand still asks {:.2} GB for a shape measured to need {:.2} GB",
        zimage_runtime_demand(w, h) as f64 / 1e9,
        measured as f64 / 1e9,
    );
}
