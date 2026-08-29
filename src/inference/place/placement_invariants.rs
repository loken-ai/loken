//! THE FLEET RULE, as gates that fail rather than as a paragraph nobody reads.
//!
//! The rule is one sentence: a component spreads across the CARDS, and the host takes
//! only what no card can hold. It has been restated, agreed and then broken again -
//! most recently by a 1536-square render planned onto the processor while two cards sat
//! with gigabytes free - because every restatement was a convention, and a convention is
//! invisible in a diff. What has actually held in this repository is the opposite shape:
//! a match that does not compile when a family is added without a loader, a pre-commit
//! hook, a test that reads the sources and refuses a form. So the rule lives here, in
//! tests, and the sites it covers are found by reading the tree rather than by listing
//! files that then fall behind.
//!
//! Four invariants, one section each:
//!
//!  1. THE CARDS DECIDE, NOT THE BUDGETS. A block reaches the host only when no device
//!     can physically hold one more, and no device is ever handed more than it holds.
//!     Swept over topologies rather than asserted on the machine this was written on.
//!  2. NO PLACEMENT DECIDES ON A NAME OR AN INDEX. Every function in the tree that
//!     decides where something goes is found by what it calls and what it returns, and
//!     none of them may name a card by number, rank cards by free VRAM, or branch on
//!     which checkpoint asked.
//!  3. EVERY HOT COMPONENT CAN SPAN N CARDS. Each image/video family declares the loader
//!     that places its block stack, and that loader has to take a plan or a list of
//!     devices - a single `&Device` cannot spread. The declaration is an exhaustive
//!     match, so a family added without one does not compile.
//!  4. A RESERVE IS A PLACEMENT DECISION. Changing what a request reserves may not move
//!     a block to the host, and the geometries that render today keep the plan they have.
//!
//! Nothing here needs a GPU, a checkpoint or a server: it is division and text.

use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan};

// ============================================================================
// Shared helpers
// ============================================================================

/// Every `.rs` file under the crate's `src`, except `src/bin`: those binaries are probes
/// and benchmarks that place by intent on a quiet machine, and are not the served fleet.
fn fleet_sources() -> Vec<std::path::PathBuf> {
    fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                if p.file_name().is_some_and(|n| n == "bin") {
                    continue;
                }
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "rs")
                && !is_test_file(&p)
                // This file is the gate. It names cards and checkpoints on purpose - that
                // is what the fixtures below are made of - and it ships nothing.
                && !p.file_name().is_some_and(|n| n == "placement_invariants.rs")
            {
                out.push(p);
            }
        }
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut out = Vec::new();
    walk(&root, &mut out);
    out.sort();
    out
}

/// The part of a file that ships. Fixtures in a test module name concrete devices and
/// concrete byte counts on purpose - that is how they pin behaviour - so only production
/// code is read.
fn production(text: &str) -> &str {
    text.split("#[cfg(test)]").next().unwrap_or("")
}

/// Whether a whole FILE is test code.
///
/// The `#[cfg(test)]` split above assumes the tests sit inside the file they exercise. Once a
/// test module is extracted to its own file the attribute stays behind on the parent's `mod`
/// declaration, so the file reads as production from the first line - which is how a fixture's
/// deliberate `Content::read` started being reported as a shipping defect.
fn is_test_file(path: &std::path::Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n == "tests.rs" || n.ends_with("_tests.rs") || n.ends_with("_test.rs"))
}

/// A line with its trailing comment removed. A sentence ABOUT a card index is not a
/// decision taken on one, and these files carry a great deal of prose.
fn code_only(line: &str) -> &str {
    match line.find("//") {
        Some(p) => &line[..p],
        None => line,
    }
}

/// The name this line declares, if it declares a function.
///
/// Written out rather than pattern-matched because there is no regex crate here and this
/// has to see through the visibility and qualifier soup a real tree contains.
fn declared_fn(line: &str) -> Option<&str> {
    let mut t = line.trim_start();
    loop {
        if let Some(rest) = t.strip_prefix("pub(") {
            let close = rest.find(')')?;
            t = rest[close + 1..].trim_start();
            continue;
        }
        let mut advanced = false;
        for kw in ["pub ", "default ", "const ", "async ", "unsafe ", "extern "] {
            if let Some(rest) = t.strip_prefix(kw) {
                // `extern "C"` carries an ABI string before the qualifier ends.
                t = if kw == "extern " {
                    match rest.trim_start().strip_prefix('"') {
                        Some(abi) => match abi.find('"') {
                            Some(end) => abi[end + 1..].trim_start(),
                            None => rest.trim_start(),
                        },
                        None => rest.trim_start(),
                    }
                } else {
                    rest.trim_start()
                };
                advanced = true;
                break;
            }
        }
        if !advanced {
            break;
        }
    }
    let rest = t.strip_prefix("fn ")?;
    let name: &str = rest
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .next()
        .unwrap_or("");
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

/// Every function in `lines`, as `(name, start, end)` over the line index. A nested `fn`
/// simply ends the region above it, which narrows the window and never widens it.
fn fn_regions(lines: &[&str]) -> Vec<(String, usize, usize)> {
    let starts: Vec<(usize, String)> = lines
        .iter()
        .enumerate()
        .filter_map(|(i, l)| declared_fn(l).map(|n| (i, n.to_string())))
        .collect();
    starts
        .iter()
        .enumerate()
        .map(|(n, (i, name))| {
            let end = starts.get(n + 1).map_or(lines.len(), |(j, _)| *j);
            (name.clone(), *i, end)
        })
        .collect()
}

/// Written on or just above a line that is deliberately outside one of these rules. Every
/// exemption is greppable and has to say why in the same breath.
const EXEMPT: &str = "PLACEMENT-EXEMPT:";

// ============================================================================
// 1. The cards decide, not the budgets
// ============================================================================

/// How many blocks each device kind ended up with.
fn blocks_by_kind(plan: &HeteroPlan) -> std::collections::BTreeMap<String, usize> {
    let mut out: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for s in &plan.segments {
        *out.entry(format!("{}", s.kind)).or_default() += s.num_layers();
    }
    out
}

fn blocks_on(plan: &HeteroPlan, kind: DeviceKind) -> usize {
    plan.segments
        .iter()
        .filter(|s| s.kind == kind)
        .map(|s| s.num_layers())
        .sum()
}

fn blocks_on_the_host(plan: &HeteroPlan) -> usize {
    blocks_on(plan, DeviceKind::Cpu)
}

/// One case of the sweep: what the planner is handed, and what the cards physically are.
#[derive(Clone, Debug)]
struct Topology {
    cards: Vec<(usize, u64)>,
    total_layers: usize,
    model_bytes: u64,
    kv_per_layer: u64,
    reserve: u64,
}

impl Topology {
    fn plan(&self) -> HeteroPlan {
        HeteroPlan::calculate_with_kv_reserve(
            self.total_layers,
            self.model_bytes,
            &self.cards,
            &[],
            1.0,
            self.kv_per_layer,
            self.reserve,
        )
    }

    /// What one block costs a card, by the planner's own convention: its share of the
    /// weights plus whatever grows per layer.
    fn bytes_per_layer(&self) -> u64 {
        let weights = if self.total_layers > 0 {
            self.model_bytes / self.total_layers as u64
        } else {
            0
        };
        weights.saturating_add(self.kv_per_layer)
    }
}

/// The topologies the invariants are checked over: one to three cards of every size from
/// "holds nothing" to "holds it twice", against models and reserves that bracket the
/// fleet. Deliberately NOT the machine this was written on - a rule that only holds for
/// two 16 GB cards is not the rule.
fn sweep() -> Vec<Topology> {
    const GB: u64 = 1_000_000_000;
    let sizes = [GB, 2 * GB, 5 * GB, 8 * GB, 12 * GB, 16 * GB, 24 * GB];
    let mut fleets: Vec<Vec<u64>> = Vec::new();
    for a in sizes {
        fleets.push(vec![a]);
        for b in sizes {
            fleets.push(vec![a, b]);
            for c in sizes {
                fleets.push(vec![a, b, c]);
            }
        }
    }
    let mut out = Vec::new();
    for fleet in &fleets {
        let cards: Vec<(usize, u64)> = fleet.iter().copied().enumerate().collect();
        for total_layers in [1usize, 8, 30, 40] {
            for model_bytes in [4 * GB, 12 * GB, 30 * GB] {
                for kv_per_layer in [0u64, 100_000_000] {
                    for reserve in [0u64, GB, 3 * GB, 7 * GB] {
                        out.push(Topology {
                            cards: cards.clone(),
                            total_layers,
                            model_bytes,
                            kv_per_layer,
                            reserve,
                        });
                    }
                }
            }
        }
    }
    out
}

/// A BLOCK REACHES THE HOST ONLY WHEN NO CARD CAN TAKE ONE MORE.
///
/// The defect this exists for, in the shape it had: the per-card budget is free VRAM
/// minus the runtime reserve, and that reserve is charged to EVERY card because any of
/// them may be the one running the forward. Two 16.4 GB cards keeping 7 GB free apiece
/// therefore offer 18 GB of budget for 32.8 GB of silicon, a 12.3 GB block stack misses
/// by fifty megabytes, and one block goes to the processor - where a diffusion step takes
/// minutes, no error reaches the client, and the render simply never arrives.
///
/// The fix was to finish the fill against what the cards HOLD rather than what they were
/// budgeted. This gate is what keeps it: for every topology in the sweep, a host segment
/// implies that every device is full to within one block. It cannot be satisfied by
/// tuning a number, because the same arithmetic is checked on cards from 1 GB to 24 GB,
/// on one card and on three, and at four different reserves.
#[test]
fn no_block_reaches_the_host_while_a_card_can_hold_one() {
    for t in sweep() {
        let bpl = t.bytes_per_layer();
        if bpl == 0 {
            continue;
        }
        let plan = t.plan();
        let host = blocks_on_the_host(&plan);
        if host == 0 {
            continue;
        }
        for (idx, capacity) in &t.cards {
            let held = blocks_on(&plan, DeviceKind::Cuda(*idx)) as u64 * bpl;
            assert!(
                held + bpl > *capacity,
                "{host} of {} blocks went to the HOST while CUDA({idx}) held {:.2} GB of a \
                 {:.2} GB card and one block costs {:.2} GB - the budgets decided, not the \
                 cards. Topology: {:?}, plan: {:?}",
                t.total_layers,
                held as f64 / 1e9,
                *capacity as f64 / 1e9,
                bpl as f64 / 1e9,
                t,
                blocks_by_kind(&plan),
            );
        }
    }
}

/// AND NO CARD IS HANDED MORE THAN IT HOLDS.
///
/// The dual, and not a theoretical one: the pass that evens the segments out divides the
/// blocks in proportion to the budgets, and a ratio applied to a whole number of blocks
/// does not know what a card holds. Two cards of 5 GB and 2 GB handed thirty blocks of a
/// twelve gigabyte model came out 5/13 - 5.2 GB on the smaller card - where the fill
/// before it had them right at 5/8. That is an out-of-memory during the upload, and it
/// is the failure mode a planner that "spreads more" always drifts towards, so it is
/// checked over the same sweep and at the same time as the rule above.
#[test]
fn no_card_is_handed_more_than_it_physically_holds() {
    for t in sweep() {
        let bpl = t.bytes_per_layer();
        if bpl == 0 {
            continue;
        }
        let plan = t.plan();
        for (idx, capacity) in &t.cards {
            let held = blocks_on(&plan, DeviceKind::Cuda(*idx)) as u64 * bpl;
            assert!(
                held <= *capacity,
                "CUDA({idx}) was given {:.2} GB of blocks on a {:.2} GB card. Topology: {:?}, \
                 plan: {:?}",
                held as f64 / 1e9,
                *capacity as f64 / 1e9,
                t,
                blocks_by_kind(&plan),
            );
        }
    }
}

/// Every block is placed exactly once, wherever it lands.
///
/// The cheapest of these to satisfy and the one whose absence hides the others: a planner
/// that silently drops a block passes both rules above.
#[test]
fn the_plan_places_every_block_exactly_once() {
    for t in sweep() {
        let plan = t.plan();
        let placed: usize = plan.segments.iter().map(|s| s.num_layers()).sum();
        assert_eq!(
            placed, t.total_layers,
            "the plan lost or duplicated blocks: {t:?}"
        );
        let mut next = 0usize;
        for s in &plan.segments {
            assert_eq!(
                s.layer_start, next,
                "the segments are not contiguous: {t:?}"
            );
            next = s.layer_end;
        }
        assert_eq!(
            next, t.total_layers,
            "the segments do not cover the stack: {t:?}"
        );
    }
}

/// The rule is about DEVICES, not about CUDA. A slower accelerator is still a card.
///
/// The whole placement path is written over a CUDA list with a second list beside it, and
/// the second list is the one nobody runs. A fill that walks only the CUDA devices leaves
/// an OpenCL box spilling to the host with an idle accelerator next to it - the same bug,
/// on the hardware least likely to be tested. That is what this catches.
///
/// It does NOT exercise the relaxation, and cannot: this class of device carries no flat
/// runtime reserve, so its budget IS its capacity and there is nothing to relax. What is
/// under test here is that it is filled at all, and never past what it holds.
#[test]
fn a_non_cuda_accelerator_is_filled_before_the_host_too() {
    const GB: u64 = 1_000_000_000;
    // Weights expand when they are dequantized for this class of device, so its capacity
    // is the free VRAM over that expansion - which is what the planner budgets against.
    const EXPANSION: f64 = 3.5;
    for free in [4 * GB, 8 * GB, 16 * GB] {
        for total_layers in [8usize, 30] {
            for model_bytes in [4 * GB, 12 * GB] {
                let plan =
                    HeteroPlan::calculate(total_layers, model_bytes, &[], &[(0, free)], EXPANSION);
                let capacity = (free as f64 / EXPANSION) as u64;
                let bpl = model_bytes / total_layers as u64;
                let held = blocks_on(&plan, DeviceKind::OpenCL(0)) as u64 * bpl;
                if blocks_on_the_host(&plan) > 0 {
                    assert!(
                        held + bpl > capacity,
                        "blocks went to the host while the OpenCL device held {:.2} GB of a \
                         {:.2} GB effective budget",
                        held as f64 / 1e9,
                        capacity as f64 / 1e9,
                    );
                }
                assert!(held <= capacity, "the OpenCL device was over-filled");
            }
        }
    }
}

/// WHAT THE RULE CANNOT REACH: A CALLER THAT HIDES THE CARD FROM THE PLANNER.
///
/// The planner is given two figures per card - the budget the fill works against, and the
/// capacity it may dip into rather than send a block to the host - and it can only honour
/// the second if it is told it. A site that subtracts its reserve inside the PROBE hands
/// the same number as both, so the relaxation has nothing to relax and its blocks spill to
/// the host with the reserve still sitting unused on the cards.
///
/// Several do: the shared TTS placement in `hetero_place::plan_layers`, the Wan video DiT,
/// the ACE-Step DiT and LM. Each of them probes `stable_free - reserve` and then passes a
/// zero reserve - which is CORRECT for them, because passing it again would subtract it
/// twice, and that is a separate gate. It is the wiring that decides, not the planner.
///
/// This test is the difference, written out, so that it is a fact rather than a thing
/// somebody once knew. The same machine and the same model, planned both ways: hide the
/// capacity and blocks reach the host; show it and they do not. Anyone changing a caller
/// from one wiring to the other is changing where that model runs, and this says by how
/// much.
#[test]
fn hiding_the_capacity_in_the_probe_is_what_sends_a_block_to_the_host() {
    const GB: u64 = 1_000_000_000;
    // Two 8 GB cards, a 12 GB stack of thirty blocks, and a 5 GB forward. The cards hold
    // the stack twice over between them; their BUDGETS - 3 GB apiece - hold half of it.
    let free = [8 * GB, 8 * GB];
    let (blocks, stack, reserve) = (30usize, 12 * GB, 5 * GB);

    // The wiring that hides it: the probe returns free - reserve, and the planner is told
    // nothing more, so that figure is the card as far as it knows.
    let net: Vec<(usize, u64)> = free
        .iter()
        .enumerate()
        .map(|(i, f)| (i, f - reserve))
        .collect();
    let hidden = HeteroPlan::calculate_with_kv_reserve(blocks, stack, &net, &[], 1.0, 0, 0);

    // The wiring that shows it: the probe returns the card, the reserve goes to the
    // planner. The FILL is identical - the planner subtracts the same reserve - and only
    // the last blocks differ, because now there is somewhere for them to go.
    let gross: Vec<(usize, u64)> = free.iter().copied().enumerate().collect();
    let shown = HeteroPlan::calculate_with_kv_reserve(blocks, stack, &gross, &[], 1.0, 0, reserve);

    assert!(
        blocks_on_the_host(&hidden) > 0,
        "the point of this test is that hiding the capacity costs blocks to the host, and \
         it no longer does - the arithmetic moved, so the comparison below means nothing: \
         {:?}",
        blocks_by_kind(&hidden),
    );
    assert_eq!(
        blocks_on_the_host(&shown),
        0,
        "showing the planner the cards must keep every block on them: {:?}",
        blocks_by_kind(&shown),
    );
}

/// AND THE SAME RULE FOR THE SECOND PLANNER IN THE TREE.
///
/// The convolutional family does not have a stack of identical blocks, so it is not
/// placed by [`HeteroPlan`] but by its own stage packer - which means the invariant above
/// says nothing about it. That is exactly how a rule becomes true of the code someone was
/// looking at and false everywhere else, so the stage packer is swept here too: a stage
/// goes to the host only when NO card can hold it, and no card is handed more than its
/// budget.
///
/// The stage widths are deliberately lopsided, because that is this network's shape - the
/// deepest levels are twenty times the shallowest - and it is the large stage arriving
/// early that used to strand every small stage after it on the processor.
#[test]
fn the_stage_planner_reaches_the_host_only_for_stages_no_card_holds() {
    use crate::inference::model::sdxl::unet::plan_stage_slots;

    const MB: u64 = 1_000_000;
    let shapes: [Vec<u64>; 4] = [
        vec![100 * MB; 8],
        // A wall in the middle, the shape that stranded the tail.
        vec![100 * MB, 100 * MB, 4_000 * MB, 100 * MB, 100 * MB, 100 * MB],
        // Growing then shrinking, which is what a UNet actually looks like.
        vec![
            50 * MB,
            200 * MB,
            800 * MB,
            1_600 * MB,
            800 * MB,
            200 * MB,
            50 * MB,
        ],
        vec![3_000 * MB, 3_000 * MB, 3_000 * MB],
    ];
    // One card, two of the same size, a small card in front of a large one, and three.
    let fleets: [Vec<u64>; 5] = [
        vec![1_000 * MB],
        vec![2_000 * MB, 2_000 * MB],
        vec![500 * MB, 6_000 * MB],
        vec![1_000 * MB, 1_000 * MB, 1_000 * MB],
        vec![20_000 * MB],
    ];
    for weights in &shapes {
        for budgets in &fleets {
            let slots = plan_stage_slots(weights, budgets);
            assert_eq!(slots.len(), weights.len(), "the packer lost a stage");
            let mut used = vec![0u64; budgets.len()];
            for (w, slot) in weights.iter().zip(slots.iter()) {
                match slot {
                    Some(d) => used[*d] += *w,
                    // A stage on the host: no card may have had room for it AT THE MOMENT
                    // it was placed. The stages are replayed in order and `used` is
                    // accumulated as they go, so what is compared here is the state the
                    // packer itself saw, not the end of the run.
                    None => {
                        for (d, budget) in budgets.iter().enumerate() {
                            assert!(
                                budget.saturating_sub(used[d]) < *w,
                                "a {:.2} GB stage went to the host while card {d} still had \
                                 {:.2} GB of its {:.2} GB budget free - the host is for what \
                                 no card holds. Stages {weights:?}, fleet {budgets:?}, \
                                 placement {slots:?}",
                                *w as f64 / 1e9,
                                budget.saturating_sub(used[d]) as f64 / 1e9,
                                *budget as f64 / 1e9,
                            );
                        }
                    }
                }
            }
            for (d, budget) in budgets.iter().enumerate() {
                assert!(
                    used[d] <= *budget,
                    "card {d} was packed to {:.2} GB past a {:.2} GB budget",
                    used[d] as f64 / 1e9,
                    *budget as f64 / 1e9,
                );
            }
        }
    }
}

// ============================================================================
// 2. No placement decides on a name or an index
// ============================================================================

/// What a function has to CALL for its body to be a placement.
const PLACEMENT_CALLS: [&str; 14] = [
    "HeteroPlan::calculate",
    "HeteroPlan::split_across_cuda",
    "HeteroPlan::forced_gpu",
    "vram_manager::probe(",
    "pick_device_for",
    "pick_device_tiered",
    "pick_device_elsewhere",
    "run_staged(",
    "probe_cuda_devices",
    "probe_cuda_gpus",
    "plan_layers(",
    "place_whole(",
    "probe_under_pressure(",
    "place_aside(",
];

/// What it has to RETURN, or be NAMED, for that placement to be its job rather than one
/// step of a render. A function that returns a picture is allowed to know which
/// checkpoint it is drawing; a function that returns a device is not.
fn is_placement_helper(name: &str, signature: &str, body: &str) -> bool {
    if !PLACEMENT_CALLS.iter().any(|c| body.contains(c)) {
        return false;
    }
    let returns_a_placement = signature
        .split("->")
        .nth(1)
        .is_some_and(|r| r.contains("HeteroPlan") || r.contains("Device"));
    let named_like_one = name.ends_with("_device")
        || name.ends_with("_devices")
        || name.ends_with("_card")
        || name.ends_with("_plan")
        || name.ends_with("_placement")
        || name.starts_with("plan_")
        || name.starts_with("place_")
        || name.starts_with("pick_");
    returns_a_placement || named_like_one
}

/// An identifier that holds a card's POSITION rather than what the card can do.
fn is_an_index_name(name: &str) -> bool {
    let n = name.to_lowercase();
    n == "i"
        || n.ends_with("idx")
        || n.ends_with("index")
        || n.ends_with("ord")
        || n.contains("gpu")
        || n.contains("card")
        || n.ends_with("dev")
}

/// The identifier immediately to the left of `at`, if there is one.
fn ident_before(line: &str, at: usize) -> &str {
    let head = line[..at].trim_end().trim_end_matches(['*', '&', '(']);
    let start = head
        .char_indices()
        .rev()
        .take_while(|(_, c)| c.is_ascii_alphanumeric() || *c == '_')
        .last()
        .map_or(head.len(), |(i, _)| i);
    &head[start..]
}

/// A SMALL integer literal at the start of `rest` - a card number, not a byte count.
fn small_literal_at(rest: &str) -> bool {
    let t = rest.trim_start();
    let digits = t.chars().take_while(char::is_ascii_digit).count();
    // One or two digits, and nothing that turns them into part of something else: a
    // float, a hex escape, a underscore-grouped byte count, an identifier.
    digits > 0
        && digits <= 2
        && !t[digits..].starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_' || c == '.')
}

/// A card named by its number, in a function whose job is to choose a card.
///
/// Three spellings, because the defect has worn all three: a device built from a literal
/// index, an ordinal read back out of one, and - the one a gate written against the first
/// two walks straight past - an index COMPARED to a number, which is how a placement says
/// "the first card" without ever writing `Cuda(0)`.
fn names_a_card_by_number(line: &str) -> bool {
    let digit_after = |pat: &str| -> bool {
        line.match_indices(pat)
            .any(|(i, _)| small_literal_at(&line[i + pat.len()..]))
    };
    if digit_after("Cuda(")
        || digit_after("new_cuda(")
        || digit_after("CudaDevice::new(")
        || digit_after("cuda:")
        || line.contains(".ordinal()")
    {
        return true;
    }
    // `idx == 0`, `*i != 1`, `gpu < 2`: a decision on WHICH card, by number.
    for op in ["==", "!=", "<=", ">=", "<", ">"] {
        for (i, _) in line.match_indices(op) {
            if is_an_index_name(ident_before(line, i)) && small_literal_at(&line[i + op.len()..]) {
                return true;
            }
        }
    }
    // `cudas[0]`, `ranked[1]`: the probe's own list, subscripted by position.
    for (i, _) in line.match_indices('[') {
        let name = ident_before(line, i).to_lowercase();
        let is_a_card_list = ["cuda", "gpu", "card", "dev", "probe", "ranked"]
            .iter()
            .any(|w| name.contains(w));
        if is_a_card_list && small_literal_at(&line[i + 1..]) && line[i + 1..].contains(']') {
            return true;
        }
    }
    false
}

/// A decision taken on WHICH CHECKPOINT asked. The forms a placement can use to find out:
/// a variant enum, a checkpoint path, a model name.
fn names_a_checkpoint(line: &str) -> bool {
    let lower = line.to_lowercase();
    ["variant", "checkpoint", "ckpt", "model_name"]
        .iter()
        .any(|w| lower.contains(w))
}

/// EVERY PLACEMENT IN THE TREE, NOT THE ONE FILE THIS WAS FIRST WRITTEN FOR.
///
/// The gate this generalises read a single region of `inference/model/wan/pipeline.rs` and refused
/// four spellings there. It was right about the shape and blind everywhere else, which is
/// the failure mode of every gate that lists what it covers: the text encoder it guarded
/// stopped naming a card, and nothing stopped the next placement from doing it.
///
/// So the placements are FOUND. A function qualifies when its body reaches the planner or
/// the probe and it either returns a device/plan or is named like a chooser - twenty of
/// them across image, video, audio, TTS, faces and the LLM path at the time of writing,
/// and any new one is picked up without editing this list. Inside those functions, a card
/// index, a `.ordinal()`, or a branch on which checkpoint asked is refused: the answer has
/// to come from the ranked probe and the request's size, because those are the only two
/// things that are true on a machine nobody has seen.
///
/// An exemption is a `PLACEMENT-EXEMPT:` comment naming the reason, which keeps every one
/// of them greppable.
#[test]
fn no_placement_helper_names_a_card_or_a_checkpoint() {
    let mut offenders: Vec<String> = Vec::new();
    let mut helpers: Vec<String> = Vec::new();
    for path in fleet_sources() {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let prod = production(&text);
        let raw: Vec<&str> = prod.lines().collect();
        let code: Vec<&str> = raw.iter().map(|l| code_only(l)).collect();
        for (name, start, end) in fn_regions(&raw) {
            // The signature runs to the opening brace; the body is the rest.
            let mut brace = start;
            while brace < end && !code[brace].contains('{') {
                brace += 1;
            }
            let signature = code[start..=brace.min(end - 1)].join(" ");
            let body = code[start..end].join("\n");
            if !is_placement_helper(&name, &signature, &body) {
                continue;
            }
            let rel = path.display().to_string();
            helpers.push(format!("{rel}::{name}"));
            if raw[start..end].iter().any(|l| l.contains(EXEMPT)) {
                continue;
            }
            for (k, line) in code[start..end].iter().enumerate() {
                let mut why = Vec::new();
                if names_a_card_by_number(line) {
                    why.push("a card by number");
                }
                if names_a_checkpoint(line) {
                    why.push("which checkpoint asked");
                }
                if !why.is_empty() {
                    offenders.push(format!(
                        "{rel}:{}: {name} decides on {}: {}",
                        start + k + 1,
                        why.join(" and "),
                        raw[start + k].trim()
                    ));
                }
            }
        }
    }
    // The gate has to be finding placements at all: a refactor that renames the planner
    // would empty the search and turn this into a test that cannot fail.
    assert!(
        helpers.len() >= 15,
        "only {} placement helpers were found in the tree - the search has stopped \
         recognising them, so this gate is no longer checking anything: {:?}",
        helpers.len(),
        helpers,
    );
    assert!(
        offenders.is_empty(),
        "a placement is being decided by a card index or by which checkpoint asked, instead \
         of by the ranked probe and the size of the request:\n{}\n\nRank with \
         vram_manager::pick_device_for / the HeteroPlan, or write `{EXEMPT} <why>` in the \
         function.",
        offenders.join("\n"),
    );
}

/// NOR BY WHICH CARD HAS THE MOST ROOM.
///
/// The banned heuristic, and the one that reads most like common sense. The fleet packs
/// the FASTEST card first, and a placement that sorts by free VRAM instead hands the hot
/// component to whichever card happens to be empty - which on this class of machine is
/// the slower one, because the faster one is where everything else already went.
///
/// Repo-wide rather than inside the placement helpers, because the ranking is exactly the
/// kind of thing that gets written next to a probe in a loader rather than in a chooser.
/// The exceptions are real and are each named in the code: permanent ballast that must
/// NOT sit on the hot card, a one-shot encoder offered what the plan leaves, a monitoring
/// API, and a runtime correction when a card turns out to have less than the plan assumed.
#[test]
fn no_card_is_chosen_for_having_the_most_room() {
    const RANKINGS: [&str; 7] = [
        "max_by_key(",
        "min_by_key(",
        "sort_by_key(",
        "sort_by(",
        "max_by(",
        "min_by(",
        "sort_unstable_by(",
    ];
    const MEMORY: [&str; 4] = ["free", "avail", "memory", "vram"];
    let mut offenders = Vec::new();
    for path in fleet_sources() {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let raw: Vec<&str> = production(&text).lines().collect();
        let regions = fn_regions(&raw);
        for (i, line) in raw.iter().enumerate() {
            let code = code_only(line);
            if !RANKINGS.iter().any(|r| code.contains(r)) {
                continue;
            }
            let lower = code.to_lowercase();
            if !MEMORY.iter().any(|m| lower.contains(m)) {
                continue;
            }
            // The reason belongs to the FUNCTION, not to the line: a ranking is written
            // as a chain and the explanation sits at its head, which a fixed window of
            // lines misses as soon as the chain grows a `filter`. Outside any function -
            // a const, a static - fall back to the lines just above.
            let exempted = match regions.iter().find(|(_, s, e)| i >= *s && i < *e) {
                Some((_, s, e)) => raw[*s..*e].iter().any(|l| l.contains(EXEMPT)),
                None => raw[i.saturating_sub(8)..=i]
                    .iter()
                    .any(|l| l.contains(EXEMPT)),
            };
            if exempted {
                continue;
            }
            offenders.push(format!("{}:{}: {}", path.display(), i + 1, line.trim()));
        }
    }
    assert!(
        offenders.is_empty(),
        "a card is being chosen for having the most room. The fleet ranks by measured \
         throughput and takes the fastest card that FITS - use \
         vram_manager::pick_device_for, or write `{EXEMPT} <why this is not a hot \
         component's home>` above it:\n{}",
        offenders.join("\n"),
    );
}

// ============================================================================
// 3. Every hot component can span N cards
// ============================================================================

/// How a family's hot block stack is handed to its cards.
///
/// `file` and `symbol` name the loader; `spread` is the parameter that carries more than
/// one device. A loader that takes a single `&Device` for a component repeated per block
/// cannot spread, which is the structural form of "multi-GPU was never wired here".
struct HotStack {
    file: &'static str,
    symbol: &'static str,
}

/// EVERY IMAGE/VIDEO FAMILY DECLARES HOW ITS HOT STACK SPANS CARDS.
///
/// The match below is exhaustive over [`ImageLoader`], which is the enum every new family
/// already has to appear in before it can load at all. So a family added without an
/// answer here does not COMPILE - the same mechanism that stopped a family being wired to
/// another family's weights, pointed at the other half of the same question.
///
/// The declaration is then checked against the source, because a table can lie: the named
/// loader must exist and must take a plan or a slice of devices. A `&Device` parameter is
/// what a single-card loader looks like, and it is what this refuses.
#[test]
fn every_image_family_declares_how_its_hot_stack_spans_cards() {
    use crate::api::handlers::media::{image_family_loader, ImageLoader, IMAGE_FAMILIES};

    let mut checked = 0usize;
    for family in IMAGE_FAMILIES {
        let loader = image_family_loader(family)
            .unwrap_or_else(|e| panic!("advertised family '{family}' has no loader: {e}"));
        // EXHAUSTIVE ON PURPOSE. Adding a variant breaks this build until the new family
        // says which loader spreads its blocks over the cards.
        let hot = match loader {
            ImageLoader::ZImage => HotStack {
                file: "src/inference/model/zimage/hetero.rs",
                symbol: "fn from_safetensors",
            },
            ImageLoader::QwenImage => HotStack {
                file: "src/inference/model/qwen_image/dit.rs",
                symbol: "fn load_hetero",
            },
            ImageLoader::Boogu => HotStack {
                file: "src/inference/model/boogu/dit.rs",
                symbol: "fn load_cancellable",
            },
            ImageLoader::Sdxl => HotStack {
                file: "src/inference/model/sdxl/unet.rs",
                symbol: "fn load_planned",
            },
            ImageLoader::Flux2 => HotStack {
                file: "src/inference/model/flux2/dit.rs",
                symbol: "fn load_hetero",
            },
            ImageLoader::Flux => HotStack {
                file: "src/inference/model/flux/hetero.rs",
                symbol: "fn from_gguf",
            },
        };
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(hot.file);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{family}: {} is not readable: {e}", hot.file));
        // With the parenthesis, or the search finds a LONGER name that starts the same
        // way: `fn load` matched `fn load_state_dict` four hundred lines earlier and this
        // gate reported the wrong signature - correctly failing, for the wrong reason.
        let wanted = format!("{}(", hot.symbol);
        // Exactly one, or the name is ambiguous and this gate would read a DIFFERENT
        // function. `fn load` matched the per-block loader four hundred lines above the
        // model's, and reported that signature: a failure for the wrong reason is only
        // one edit away from a pass for the wrong reason.
        assert_eq!(
            src.matches(&wanted).count(),
            1,
            "{family}: `{}` names {} functions in {} - point this declaration at a name \
             that identifies the loader unambiguously",
            hot.symbol,
            src.matches(&wanted).count(),
            hot.file,
        );
        let at = src.find(&wanted).unwrap_or_else(|| {
            panic!(
                "{family}: `{}` is not in {} any more - the loader that spreads this \
                 family's blocks was renamed or removed, and this declaration has to \
                 follow it",
                hot.symbol, hot.file
            )
        });
        // The parameter list: from the symbol to the end of the signature.
        let tail = &src[at..];
        let end = tail
            .find(") ->")
            .or_else(|| tail.find(")\n"))
            .unwrap_or(tail.len().min(2000));
        let params = &tail[..end];
        let spreads = params.contains("&HeteroPlan")
            || params.contains("HeteroPlan,")
            || params.contains("&[Device]")
            || params.contains("&[Device]")
            || params.contains("&[crate::tensor::Device]");
        assert!(
            spreads,
            "{family}: {}::{} takes no plan and no list of devices, so its blocks cannot be \
             spread over the cards - a hot component held by one `&Device` is the \
             structural form of 'multi-GPU was never wired here'. Signature read:\n{}",
            hot.file,
            hot.symbol,
            params.lines().take(12).collect::<Vec<_>>().join("\n"),
        );
        checked += 1;
    }
    assert_eq!(
        checked,
        IMAGE_FAMILIES.len(),
        "not every advertised family was checked - the walk over IMAGE_FAMILIES stopped early"
    );
}

// ============================================================================
// 4. A reserve is a placement decision
// ============================================================================

/// CHANGING THE RESERVE MAY NOT MOVE A BLOCK TO THE HOST.
///
/// The rule a reserve edit has to be judged against, stated so that it can be checked
/// rather than remembered. A reserve is an estimate of what a forward needs free; it
/// moves whenever someone measures a render, and each of those edits is a PLACEMENT
/// change even though nothing in the placement code was touched. The last one took a
/// working 1536-square render and put a block on the processor.
///
/// So: for the same cards and the same model, the number of blocks the host receives must
/// not depend on the reserve at all. Where the blocks sit between the cards may - that is
/// what a reserve is for - but the host is decided by what the cards hold, and that does
/// not move when an estimate does.
#[test]
fn the_reserve_cannot_send_a_block_to_the_host() {
    const GB: u64 = 1_000_000_000;
    for t in sweep() {
        if t.reserve != 0 {
            continue;
        }
        let baseline = blocks_on_the_host(&t.plan());
        for reserve in [GB, 3 * GB, 7 * GB, 14 * GB] {
            let raised = Topology {
                reserve,
                ..t.clone()
            };
            let host = blocks_on_the_host(&raised.plan());
            assert_eq!(
                host,
                baseline,
                "raising the reserve to {:.1} GB moved {} block(s) to the host that a zero \
                 reserve kept on the cards. Topology: {:?}",
                reserve as f64 / 1e9,
                host as i64 - baseline as i64,
                raised,
            );
        }
    }
}

/// THE GEOMETRIES THAT RENDER TODAY KEEP THE PLAN THEY HAVE.
///
/// Z-Image renders at 512, 1024 and 1536 square, and each of those is a placement someone
/// measured on a machine. The acceptance criterion for any later change to what this
/// family reserves - and reserves are edited often - is that for every supported geometry
/// the number of cards used does not rise, no host segment appears where there was none,
/// and the first card does not lose blocks to a later one.
///
/// Written as a table rather than as a rule so that it can FAIL: the numbers below were
/// produced by the family's own planner, and a reserve edit that shifts any of them
/// stops the build and has to be looked at. They are outputs, not chosen constants - the
/// only way to update them is to run the planner again and read what it now says.
///
/// The fleets are synthetic and deliberately span shapes this machine does not have: one
/// card, two equal cards, a fast small card beside a slow large one, three cards, and a
/// box too small for the model.
#[test]
fn the_supported_geometries_keep_their_plan() {
    use crate::inference::engine::image_engine as zi;

    /// What the loader carries at the shape it was reported on. An input to the plan,
    /// not a memory reservation: the checkpoint on disk is this size.
    const RESIDENT: u64 = 12_309_845_444;
    const MAIN_BLOCKS: usize = 30;

    /// The loader's decision, in the order the loader makes it: does one card hold the
    /// weights AND this request's forward; if not, which card takes the caption encoder;
    /// and then where the blocks go.
    fn shape(fleet: &[u64], width: usize, height: usize) -> (usize, usize, usize) {
        let demand = zi::zimage_runtime_demand(width, height);
        let mut cards: Vec<(usize, u64)> = fleet.iter().copied().enumerate().collect();
        let whole_on = cards
            .iter()
            .find(|(_, f)| *f >= RESIDENT + demand)
            .map(|(i, _)| *i);
        if let Some(idx) = whole_on {
            // One card holds the weights AND this request's forward, so the block planner
            // is never reached at all - the loader's own single-card door. The third figure
            // is what the FASTEST card carries, and it is zero when the model went whole to
            // a slower, roomier one: that is a real answer, not a rounding.
            return (1, 0, if idx == 0 { MAIN_BLOCKS } else { 0 });
        }
        let encoder = zi::zimage_text_encoder_bytes();
        if let Some(idx) = zi::zimage_encoder_card(&cards, None, whole_on, encoder) {
            for e in cards.iter_mut() {
                if e.0 == idx {
                    e.1 = e.1.saturating_sub(encoder);
                }
            }
        }
        let plan = zi::zimage_block_plan(&cards, &[], MAIN_BLOCKS, RESIDENT, demand);
        let cards_used = plan
            .segments
            .iter()
            .filter(|s| matches!(s.kind, DeviceKind::Cuda(_)))
            .map(|s| format!("{}", s.kind))
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        (
            cards_used,
            blocks_on_the_host(&plan),
            blocks_on(&plan, DeviceKind::Cuda(0)),
        )
    }

    const GB: u64 = 1_000_000_000;
    // (name, cards) x (geometry) -> (cards used, blocks on the host, blocks on card 0)
    let table: [(&str, Vec<u64>, [(usize, usize, (usize, usize, usize)); 3]); 5] = [
        (
            "one 16.4 GB card - the single-card box. At 1536 the model no longer fits \
             beside its forward, so the blocks dip into the reserve rather than reach the \
             host: an exhaustion there is visible and re-plannable, a host block is not.",
            vec![16_400_000_000],
            [
                (512, 512, (1, 0, 30)),
                (1024, 1024, (1, 0, 30)),
                (1536, 1536, (1, 0, 30)),
            ],
        ),
        (
            "two 16.4 GB cards - the machine the 1536 spill was reported on. The two \
             working sizes stay WHOLE on the fastest card; 1536 splits, and nothing goes \
             to the processor.",
            vec![16_400_000_000, 16_400_000_000],
            [
                (512, 512, (1, 0, 30)),
                (1024, 1024, (1, 0, 30)),
                (1536, 1536, (2, 0, 22)),
            ],
        ),
        (
            "a 12 GB card in front of a 24 GB one - the fastest card does not hold this \
             model at any of these sizes, so the whole of it goes to the roomier one and \
             the fastest carries none of it. Fastest-that-FITS, not fastest.",
            vec![12 * GB, 24 * GB],
            [
                (512, 512, (1, 0, 0)),
                (1024, 1024, (1, 0, 0)),
                (1536, 1536, (1, 0, 0)),
            ],
        ),
        (
            "three 12 GB cards - no card holds it whole, the caption encoder takes the \
             third, and the blocks divide over the first two.",
            vec![12 * GB, 12 * GB, 12 * GB],
            [
                (512, 512, (2, 0, 13)),
                (1024, 1024, (2, 0, 13)),
                (1536, 1536, (2, 0, 16)),
            ],
        ),
        (
            "one 8 GB card - genuinely too small, so the host does take blocks. The count \
             is the same at every geometry because what the card holds is decided by the \
             WEIGHTS; the reserve moves the split between cards, never the host.",
            vec![8 * GB],
            [
                (512, 512, (1, 13, 17)),
                (1024, 1024, (1, 13, 17)),
                (1536, 1536, (1, 13, 17)),
            ],
        ),
    ];
    let mut diffs = Vec::new();
    for (name, fleet, rows) in &table {
        for (w, h, expected) in rows {
            let got = shape(fleet, *w, *h);
            if got != *expected {
                diffs.push(format!(
                    "{name} at {w}x{h}: (cards, host blocks, blocks on card 0) was {expected:?}, \
                     is now {got:?}"
                ));
            }
        }
    }
    assert!(
        diffs.is_empty(),
        "a supported geometry changed where it renders. This is a PLACEMENT change even if \
         only a reserve was edited: check that no card count rose, that no host segment \
         appeared, and that the first card did not lose blocks to a later one - then update \
         the table with what the planner now says.\n{}",
        diffs.join("\n"),
    );
}

// ============================================================================
// 5. BYTES THAT CAME FROM A MAPPING ARE PARSED WITH WHAT BACKS THEM
// ============================================================================

/// `Content::read` leaves `mmap_owner` at None, and the loader then COPIES every tensor
/// into owned host memory instead of viewing the mapping. Both loads succeed, produce the
/// same tokens, and differ only in committed RAM - so the defect has no symptom until a
/// machine runs short, and it lands on the largest models by construction: the re-parses
/// live in the out-of-memory recovery ladder, reached only by weights too big to place on
/// the first attempt.
///
/// Ten sites had drifted this way. Enumerating them fixes today; refusing the FORM is what
/// stops the eleventh, since a re-parse added later reads exactly like the ten that were
/// already there.
#[test]
fn a_gguf_parsed_over_a_mapping_keeps_its_owner() {
    let mut offenders = Vec::new();
    for path in fleet_sources() {
        // The module that DEFINES the sanctioned entry points necessarily calls the plain
        // constructor once, inside them. That module used to be `quantized.rs`; splitting it
        // moved the reader to `quantized/gguf_file.rs`, and the exemption follows the
        // definition rather than the old path - otherwise this gate reports its own
        // foundation as the defect.
        if path.ends_with("tensor/quantized/gguf_file.rs")
            || path.ends_with("tensor/quantized/mod.rs")
        {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        for (n, line) in production(&text).lines().enumerate() {
            let l = code_only(line).trim();
            // ANY plain parse in shipping code is the defect: a weight load through it
            // copies every tensor into committed heap, and the old same-line-mmap
            // heuristic caught zero of the ten sites that did exactly that.
            if l.contains("Content::read(") {
                offenders.push(format!(
                    "{}:{} - {}",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    n + 1,
                    l
                ));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "plain GGUF parse in shipping code - weight loads go through `open_mapped` / \
         `read_mapped_file` (zero-copy, file-backed), metadata probes through \
         `open_header` (declared intent):\n  {}",
        offenders.join("\n  ")
    );
}
