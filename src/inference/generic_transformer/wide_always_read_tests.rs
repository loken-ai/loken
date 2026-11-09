//! The split of a card's room between the weights every token reads and the routed experts.

use super::wide_always_read::{count_shares, expert_share, split_room, Offer};

fn offer(bytes: u64, saving: f64) -> Offer {
    Offer { bytes, saving }
}

#[test]
fn a_weight_read_by_every_token_goes_before_experts_that_save_less_per_byte() {
    let weights = [offer(30, 1.0), offer(20, 1.0)];
    let experts = [offer(10, 0.5); 10];
    let (taken, experts_bytes) = split_room(70, &weights, &experts);
    assert_eq!(taken, vec![true, true]);
    assert_eq!(experts_bytes, 20, "the experts get what the weights leave");
}

#[test]
fn experts_that_save_more_per_byte_keep_the_room() {
    let weights = [offer(30, 1.0)];
    let experts = [offer(10, 2.0); 4];
    let (taken, experts_bytes) = split_room(40, &weights, &experts);
    assert_eq!(taken, vec![false]);
    assert_eq!(experts_bytes, 40);
}

#[test]
fn a_weight_that_does_not_fit_leaves_room_to_smaller_ones() {
    let weights = [offer(50, 3.0), offer(10, 1.0)];
    let (taken, _) = split_room(40, &weights, &[]);
    assert_eq!(taken, vec![false, true]);
}

#[test]
fn a_weight_the_card_does_not_speed_up_is_never_taken() {
    let weights = [offer(10, 0.0), offer(10, -1.0)];
    let (taken, experts_bytes) = split_room(100, &weights, &[offer(10, 0.0)]);
    assert_eq!(taken, vec![false, false]);
    assert_eq!(experts_bytes, 0);
}

#[test]
fn a_tie_goes_to_the_weight() {
    let (taken, experts_bytes) = split_room(10, &[offer(10, 1.0)], &[offer(10, 1.0)]);
    assert_eq!(taken, vec![true]);
    assert_eq!(experts_bytes, 0);
}

#[test]
fn a_rank_bounds_the_share_without_counts() {
    assert_eq!(expert_share(None, 0, 6), 1.0);
    assert_eq!(expert_share(None, 5, 6), 1.0);
    assert_eq!(expert_share(None, 11, 6), 0.5);
    assert_eq!(expert_share(Some(0.25), 0, 6), 0.25);
}

#[test]
fn counts_give_each_expert_its_share_of_the_tokens() {
    // 4 tokens, 2 experts each.
    let shares = count_shares(&[4, 2, 1, 1], 2).unwrap();
    assert_eq!(shares, vec![1.0, 0.5, 0.25, 0.25]);
    assert_eq!(count_shares(&[0, 0], 2), None);
}
