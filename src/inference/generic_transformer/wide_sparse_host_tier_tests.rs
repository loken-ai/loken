//! The host tier of the routed experts of a wide-residual model: what it keeps, what it lets go,
//! what leaves the page cache, how a guess is scored, and that it changes no bit of a forward.
//!
//! The model is the synthetic one of the frozen V4.1 oracle.

use super::expert_readahead::ReadAhead;
use super::moe_weights::{sqrt_softplus_choice, sqrt_softplus_truncated_choice, Truncation};
use super::wide_sparse_batch_tests::{batched, prompts, serial};
use super::wide_sparse_host_tier::{
    host_slot, page_cost, score_guess, slots_for, Guess, GuessCount, HostTier, KeepSet, Ranking,
};
use super::wide_sparse_oracle_tests::{generic, LAYOUTS};
use crate::inference::expert_counters::Found;
use crate::tensor::{Device, Tensor};
use std::sync::Arc;

/// A tier of `capacity` experts a layer over a ceiling of 8, its reads ahead `readahead`.
fn tier(capacity: usize, readahead: Option<Arc<ReadAhead>>) -> Arc<HostTier> {
    Arc::new(HostTier::new(capacity, 8, 0, 1, readahead))
}

#[test]
fn a_seed_keeps_the_hottest_and_counts_the_hottest_as_the_most_recent() {
    let mut set = KeepSet::new(8);
    set.seed(&[5, 2, 7, 1], 3);
    assert_eq!(set.len(), 3);
    assert!(set.is_kept(5) && set.is_kept(2) && set.is_kept(7));
    assert!(!set.is_kept(1));
    // The coldest of the seed gives way first.
    assert_eq!(set.shrink_to(1, &[]), vec![7, 2]);
    assert!(set.is_kept(5));
}

#[test]
fn a_decoded_row_admits_its_experts_and_the_least_recent_gives_way() {
    let mut set = KeepSet::new(8);
    assert_eq!(set.decoded(&[0, 1], 3), Vec::<usize>::new());
    assert_eq!(set.decoded(&[1], 3), Vec::<usize>::new());
    assert_eq!(set.decoded(&[2], 3), Vec::<usize>::new());
    // Expert 0 is the least recently decoded, then expert 1.
    assert_eq!(set.decoded(&[3, 4], 3), vec![0, 1]);
    assert_eq!(set.len(), 3);
    assert!(set.is_kept(2) && set.is_kept(3) && set.is_kept(4));
    // The experts of the row stay, even past the bound.
    assert_eq!(set.decoded(&[5, 6, 7, 0], 2), vec![2, 3, 4]);
    assert_eq!(set.len(), 4);
}

#[test]
fn a_prompt_stamps_each_expert_at_its_last_row() {
    let mut set = KeepSet::new(8);
    set.decoded(&[5], 8);
    // Rows tick after the decoded row: every expert of the prompt is newer than expert 5.
    assert_eq!(set.prompted(&[(1, 3), (2, 0), (4, 1)], 4, 3), vec![5]);
    // Expert 2's last row is the oldest of the prompt.
    assert_eq!(set.shrink_to(2, &[]), vec![2]);
    assert!(set.is_kept(1) && set.is_kept(4));
}

#[test]
fn ranked_by_decoded_count_the_least_decoded_gives_way_first() {
    let mut set = KeepSet::new(8).ranked(Ranking::DecodedCount);
    set.decoded(&[1], 8);
    set.decoded(&[1], 8);
    set.decoded(&[2], 8);
    set.decoded(&[3], 8);
    // Expert 1 is the least recent but the most decoded: expert 2 gives way, then expert 3.
    assert_eq!(set.shrink_to(1, &[]), vec![2, 3]);
    assert!(set.is_kept(1));
    // A count outlives its expert: expert 2 comes back with 2 decoded rows.
    set.decoded(&[2], 8);
    set.decoded(&[4], 8);
    assert_eq!(set.shrink_to(2, &[]), vec![4]);
    // The same rows ranked by recency let the least recent go.
    let mut set = KeepSet::new(8);
    for e in [1, 1, 2, 3] {
        set.decoded(&[e], 8);
    }
    assert_eq!(set.shrink_to(1, &[]), vec![1, 2]);
}

#[test]
fn a_set_tells_which_rows_let_an_expert_go() {
    let mut set = KeepSet::new(8);
    set.decoded(&[1], 1);
    assert_eq!(set.found(1), Found::Kept);
    assert_eq!(set.found(5), Found::NeverKept);
    set.decoded(&[2], 1);
    assert_eq!(set.found(1), Found::LeftAtDecode);
    set.prompted(&[(3, 0)], 4, 1);
    assert_eq!(set.found(2), Found::LeftAtPrompt);
    assert_eq!(set.by_recency(), vec![3]);
}

#[test]
fn the_tier_takes_the_ranking_whose_shadow_missed_fewer() {
    let t = tier(2, None);
    assert_eq!(t.ranking(), Ranking::DecodedCount);
    t.note_missed(5, 5);
    t.choose_ranking();
    assert_eq!(
        t.ranking(),
        Ranking::DecodedCount,
        "a tie keeps the ranking"
    );
    t.note_missed(5, 2);
    t.choose_ranking();
    assert_eq!(t.ranking(), Ranking::Recency);
    // The counts start again at each look.
    t.choose_ranking();
    assert_eq!(t.ranking(), Ranking::Recency);
}

/// Each layer ranks its kept experts as the tier does and its shadow the other way, follows the
/// tier when it changes, and no ranking moves a bit of a forward.
#[test]
fn a_layer_follows_the_tiers_ranking_without_moving_a_bit() {
    let (ratios, kv_src, index_src) = LAYOUTS[0];
    let mut plain = generic(ratios, kv_src, index_src);
    let mut tiered = generic(ratios, kv_src, index_src);
    let t = tier(1, None);
    let keeps = tiered.attach_host_tier(&t);
    for k in &keeps {
        assert_eq!(k.rankings(), (Ranking::DecodedCount, Ranking::Recency));
    }
    let ps = prompts();
    let want = serial(&mut plain, &ps);
    assert_eq!(serial(&mut tiered, &ps), want, "the first ranking");
    // The prompts may have moved the ranking already: the layers follow the tier's.
    let before = t.ranking();
    for k in &keeps {
        assert_eq!(k.rankings(), (before, before.other()));
    }
    // A look starts the counts again, then the shadow misses fewer: the tier takes the other
    // ranking.
    t.choose_ranking();
    let before = t.ranking();
    t.note_missed(3, 1);
    t.choose_ranking();
    assert_eq!(t.ranking(), before.other());
    assert_eq!(serial(&mut tiered, &ps), want, "the other ranking");
    for k in &keeps {
        assert_eq!(k.rankings(), (t.ranking(), t.ranking().other()));
    }
}

#[test]
fn a_page_cost_covers_every_page_an_expert_can_touch() {
    let page = 4096;
    // 1 page more than the bytes need in each span, whatever the alignment.
    assert_eq!(page_cost(&[(0, 4096)], page), 2 * 4096);
    assert_eq!(page_cost(&[(1, 10), (7, 8193)], page), 2 * 4096 + 4 * 4096);
    assert_eq!(page_cost(&[], page), 0);
}

#[test]
fn the_slots_are_the_room_past_the_reserve_over_a_step() {
    assert_eq!(slots_for(100, 30, 7), 10);
    assert_eq!(slots_for(30, 100, 7), 0);
    assert_eq!(slots_for(100, 0, 0), 100);
}

#[test]
fn a_slot_is_the_largest_expert_no_card_keeps() {
    // Experts 1 and 3 are on a card, read from the larger file.
    let costs = [10, 14, 10, 14];
    assert_eq!(host_slot(&costs, &|e| e % 2 == 1), 10);
    assert_eq!(host_slot(&costs, &|_| false), 14);
    assert_eq!(host_slot(&costs, &|_| true), 0);
}

#[test]
fn a_guess_counts_its_hits_and_names_its_wasted_reads() {
    let guess = Guess {
        named: vec![1, 2, 3, 4],
        read: vec![2, 4],
    };
    let held = |e: usize| e == 1 || e == 9;
    let (count, wasted) = score_guess(&guess, &[1, 2, 9, 6], held);
    assert_eq!(
        count,
        GuessCount {
            routed: 4,
            named: 2,
            unkept: 2,
            unkept_named: 1,
        }
    );
    assert_eq!(wasted, vec![4]);
}

/// A forward of the toy model with a tier of 1 expert a layer gives the bits of the same forward
/// without one, serial and batched, and each layer ends with what its last decoded row routed to.
#[test]
fn the_tier_moves_no_bit_of_a_forward() {
    for (ratios, kv_src, index_src) in LAYOUTS {
        let mut plain = generic(ratios, kv_src, index_src);
        let mut tiered = generic(ratios, kv_src, index_src);
        let t = tier(1, None);
        let keeps = tiered.attach_host_tier(&t);
        assert_eq!(keeps.len(), ratios.len());
        assert!(
            keeps.iter().all(|k| k.kept_count() == 1),
            "the seed fills the bound"
        );
        let ps = prompts();
        let want = serial(&mut plain, &ps);
        assert_eq!(serial(&mut tiered, &ps), want, "serial");
        assert_eq!(batched(&mut tiered, &ps), want, "batched");
        // A decoded row keeps what it routed to: 2 experts, past the bound of 1.
        for k in &keeps {
            assert_eq!(k.kept_count(), 2);
        }
    }
}

/// A forward whose host experts' bytes are asked for on reader threads gives the bits of the same
/// forward without a tier, serial and batched.
#[test]
fn the_reads_asked_on_reader_threads_move_no_bit_of_a_forward() {
    for (ratios, kv_src, index_src) in LAYOUTS {
        let mut plain = generic(ratios, kv_src, index_src);
        let mut tiered = generic(ratios, kv_src, index_src);
        let keeps = tiered.attach_host_tier(&tier(1, Some(Arc::new(ReadAhead::reading(2)))));
        assert!(keeps.iter().all(|k| k.read_now(&[0, 1]).is_some()));
        let ps = prompts();
        let want = serial(&mut plain, &ps);
        assert_eq!(serial(&mut tiered, &ps), want, "serial");
        assert_eq!(batched(&mut tiered, &ps), want, "batched");
    }
}

/// A prompt keeps the experts its last rows routed to: the others it computed leave the page
/// cache, the tier's older experts with them, and a decoded row then admits what it routed to.
#[test]
fn a_prompt_keeps_its_newest_experts_and_drops_the_others() {
    let (ratios, kv_src, index_src) = LAYOUTS[0];
    let mut model = generic(ratios, kv_src, index_src);
    let keeps = model.attach_host_tier(&tier(2, None));
    let k = &keeps[0];
    let seeded: Vec<usize> = (0..8).filter(|&e| k.keeps(e)).collect();
    assert_eq!(seeded.len(), 2);
    // A prompt of 64 rows computed every expert: experts 6 and 3 last at rows 63 and 62.
    let all: Vec<usize> = (0..8).collect();
    let last = [10, 20, 30, 62, 40, 50, 63, 0];
    let mut gone = k.computed(&all, &last, 64);
    gone.sort_unstable();
    assert_eq!(gone, vec![0, 1, 2, 4, 5, 7]);
    assert_eq!(
        (0..8).filter(|&e| k.keeps(e)).collect::<Vec<_>>(),
        vec![3, 6]
    );
    // A decoded row of 2 other experts admits them and lets the prompt's go.
    let mut gone = k.computed(&[0, 1], &[0, 0], 1);
    gone.sort_unstable();
    assert_eq!(gone, vec![3, 6]);
    assert!(k.keeps(0) && k.keeps(1));
}

/// A decoded row reads ahead the next layer's choice over its input that the next layer does
/// not keep, and the next layer scores the guess against its routing.
#[test]
fn a_read_ahead_names_the_next_routers_choice_and_is_scored() {
    let (ratios, kv_src, index_src) = LAYOUTS[0];
    let mut model = generic(ratios, kv_src, index_src);
    let keeps = model.attach_host_tier(&tier(1, Some(Arc::new(ReadAhead::reading(1)))));
    let ffn = &model.layers[1].sinkhorn.as_ref().unwrap().ffn;
    let dim = ffn.router.dims()[1];
    let row: Vec<f32> = (0..dim)
        .map(|i| ((i * 37) % 11) as f32 * 0.1 - 0.5)
        .collect();
    let x = Tensor::from_vec(row.clone(), (1, dim), &Device::Cpu).unwrap();
    let logits = ffn.router.apply_here(&x).unwrap().to_vec2::<f32>().unwrap();
    let (named, _) = sqrt_softplus_choice(
        &logits[0],
        Some(ffn.bias.as_slice()),
        ffn.used,
        ffn.temperature,
        ffn.normalize,
    );
    let next = &keeps[1];
    let unkept: Vec<usize> = named.iter().copied().filter(|&e| !next.keeps(e)).collect();
    keeps[0].read_ahead(&x);
    // The next layer routes to the first named expert and to 1 the guess did not name.
    let other = (0..8).find(|e| !named.contains(e)).unwrap();
    let routed = [named[0], other];
    let (count, gone) = next.opens(&routed, 1).expect("the guess is there");
    assert_eq!(count.routed, 2);
    assert_eq!(count.named, 1);
    let held = |e: usize| next.keeps(e);
    assert_eq!(
        count.unkept,
        routed.iter().filter(|&&e| !held(e)).count() as u64
    );
    assert_eq!(count.unkept_named, u64::from(!held(named[0])));
    // The reads of the guessed experts not routed to leave the page cache.
    let wasted: Vec<usize> = unkept
        .iter()
        .copied()
        .filter(|e| !routed.contains(e))
        .collect();
    assert_eq!(gone, wasted);
    // A guess is scored once.
    assert!(next.opens(&routed, 1).is_none());
}

/// The read-ahead guesses with the next router's own choice.
/// With no truncation it names the exact choice and reads its experts not in memory.
/// Under a truncation it names the truncated choice and reads no expert that choice replaced.
#[test]
fn a_read_ahead_follows_the_truncation_of_the_next_router() {
    let (ratios, kv_src, index_src) = LAYOUTS[0];
    let mut model = generic(ratios, kv_src, index_src);
    let keeps = model.attach_host_tier(&tier(1, Some(Arc::new(ReadAhead::reading(1)))));
    let ffn = &model.layers[1].sinkhorn.as_ref().unwrap().ffn;
    let dim = ffn.router.dims()[1];
    let row: Vec<f32> = (0..dim)
        .map(|i| ((i * 37) % 11) as f32 * 0.1 - 0.5)
        .collect();
    let x = Tensor::from_vec(row, (1, dim), &Device::Cpu).unwrap();
    let logits = ffn.router.apply_here(&x).unwrap().to_vec2::<f32>().unwrap();
    let (bias, used, temperature, normalize) =
        (ffn.bias.clone(), ffn.used, ffn.temperature, ffn.normalize);
    let (exact, _) = sqrt_softplus_choice(&logits[0], Some(&bias), used, temperature, normalize);
    let next = &keeps[1];
    // The next layer keeps 1 expert, outside the exact choice.
    let outside = (0..8).find(|e| !exact.contains(e)).unwrap();
    next.computed(&[outside], &[0], 1);
    let held = |e: usize| next.keeps(e);
    assert!(held(outside) && exact.iter().all(|&e| !held(e)));
    keeps[0].read_ahead(&x);
    let guess = next.guess().expect("a guess waits");
    assert_eq!(guess.named, exact);
    assert_eq!(guess.read, exact);
    // Every light expert not in memory gives its slot to one in memory.
    let t = Truncation {
        miss_below: 1.0,
        ..Default::default()
    };
    model.set_truncation(t);
    let (lossy, _, saved) = sqrt_softplus_truncated_choice(
        &logits[0],
        Some(&bias),
        used,
        temperature,
        normalize,
        t,
        &held,
    );
    let replaced: Vec<usize> = exact
        .iter()
        .copied()
        .filter(|e| !lossy.contains(e))
        .collect();
    assert!(saved > 0 && !replaced.is_empty());
    assert!(lossy.contains(&outside));
    keeps[0].read_ahead(&x);
    let guess = next.guess().expect("a guess waits");
    assert_eq!(guess.named, lossy);
    assert!(replaced.iter().all(|e| !guess.read.contains(e)));
    assert!(!guess.read.contains(&outside));
}

/// The tier admits after a prompt only the experts whose bytes the host read: an expert a card
/// read past the page cache in transit stays out, one read through the page cache comes in.
#[test]
fn an_expert_read_past_the_page_cache_in_transit_is_not_admitted() {
    use super::wide_sparse_layer::host_held;
    let (on_host, last, moved) = ([1usize, 3, 5, 6], [10usize, 62, 20, 63], [3usize, 6]);
    assert_eq!(
        host_held(&on_host, &last, &moved),
        (vec![1, 5], vec![10, 20])
    );
    assert_eq!(
        host_held(&on_host, &last, &[]),
        (on_host.to_vec(), last.to_vec())
    );
    for (past, want) in [(&moved[..], vec![1, 5]), (&[][..], vec![3, 6])] {
        let (ratios, kv_src, index_src) = LAYOUTS[0];
        let mut model = generic(ratios, kv_src, index_src);
        let keeps = model.attach_host_tier(&tier(2, None));
        let k = &keeps[0];
        let (held, held_last) = host_held(&on_host, &last, past);
        k.computed(&held, &held_last, 64);
        assert_eq!((0..8).filter(|&e| k.keeps(e)).collect::<Vec<_>>(), want);
    }
}

/// What the tier says it would keep after a forward is what it keeps once the forward's experts
/// are admitted, and asking changes nothing.
///
/// Reading through the page cache the experts in transit it would keep, and past it the others,
/// leaves the tier as reading every expert through the page cache does.
#[test]
fn the_experts_a_prompt_leaves_in_the_tier_are_those_it_would_keep() {
    use super::wide_sparse_layer::host_held;
    let (on_host, last, moved) = ([1usize, 3, 5, 6], [10usize, 62, 20, 63], [3usize, 5, 6]);
    let (ratios, kv_src, index_src) = LAYOUTS[0];
    for rows in [64usize, 1] {
        let mut model = generic(ratios, kv_src, index_src);
        let keeps = model.attach_host_tier(&tier(2, None));
        let all = &keeps[0];
        let before: Vec<usize> = (0..8).filter(|&e| all.keeps(e)).collect();
        let after = all.would_keep(&on_host, &last, rows);
        assert_eq!((0..8).filter(|&e| all.keeps(e)).collect::<Vec<_>>(), before);
        all.computed(&on_host, &last, rows);
        let want: Vec<bool> = on_host.iter().map(|&e| all.keeps(e)).collect();
        assert_eq!(after, want, "{rows} rows");
        let mut model = generic(ratios, kv_src, index_src);
        let keeps = model.attach_host_tier(&tier(2, None));
        let some = &keeps[0];
        let after = some.would_keep(&on_host, &last, rows);
        let past: Vec<usize> = moved
            .iter()
            .copied()
            .filter(|e| on_host.iter().zip(&after).any(|(h, &k)| !k && h == e))
            .collect();
        let (held, held_last) = host_held(&on_host, &last, &past);
        some.computed(&held, &held_last, rows);
        assert_eq!(
            (0..8).filter(|&e| some.keeps(e)).collect::<Vec<_>>(),
            (0..8).filter(|&e| all.keeps(e)).collect::<Vec<_>>(),
            "{rows} rows"
        );
    }
}
