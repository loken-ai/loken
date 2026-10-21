//! A conversation's next turn against a small synthetic hybrid: the state kept at the point the
//! turns share, the tokens a turn reads, the lease and the order states go in.

use super::state_cache_tests::{argmax, floats, prompt, Fixture, VOCAB};
use super::*;
use crate::inference::engine::model_backend::{GenericBackend, ModelBackend, FORCED_CHUNK_ROWS};

/// `n` tokens that begin with `first`, so two conversations share nothing.
fn conversation(first: u32, n: usize) -> Vec<u32> {
    let mut t = prompt(n);
    t[0] = first;
    t
}

/// The next turn of `previous`: its tokens up to `shared`, then `added` of the turn's own.
fn next_turn(previous: &[u32], shared: usize, added: usize, salt: usize) -> Vec<u32> {
    let mut t = previous[..shared].to_vec();
    t.extend((0..added).map(|i| ((i * 11 + salt) % VOCAB) as u32));
    t
}

/// The logits of `tokens` read cold, in one forward per span between the `cuts`.
fn cold(f: &Fixture, tokens: &[u32], cuts: &[usize]) -> Vec<f32> {
    let mut m = f.model();
    let mut at = 0;
    let mut logits = None;
    for end in cuts.iter().copied().chain([tokens.len()]) {
        let x = Tensor::new(&tokens[at..end], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        logits = Some(floats(&m.forward(&x, at).unwrap()));
        at = end;
    }
    logits.unwrap()
}

fn assert_close(a: &[f32], b: &[f32]) {
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    eprintln!("worst difference {worst}");
    assert_eq!(
        argmax(a),
        argmax(b),
        "the same token comes next (worst {worst})"
    );
    assert!(worst < 1e-3, "logits differ by {worst}");
}

/// Reads `turn` the way a request does and returns its logits and the tokens it read.
fn read(b: &mut GenericBackend, turn: &[u32], boundary: usize) -> (Vec<f32>, usize) {
    b.set_prefix_boundary(Some(boundary));
    let logits = b
        .try_restore_prefix(turn)
        .unwrap()
        .expect("the turn is read whole or from a kept state");
    (floats(&logits), turn.len() - b.prefix_covered())
}

/// The second turn reads only what it adds: the answer and the new message, not the history.
#[test]
fn the_next_turn_reads_only_the_tokens_it_adds() {
    let f = Fixture::new("conv-two-turns");
    let mut b = GenericBackend(f.model());

    let first = prompt(60);
    let (logits, read_first) = read(&mut b, &first, 50);
    assert_eq!(read_first, 60, "a first turn is read whole");
    assert_close(&logits, &cold(&f, &first, &[50]));

    // The next prompt shares the first up to the boundary, then goes on with the answer.
    let second = next_turn(&first, 50, 40, 3);
    let (logits, read_second) = read(&mut b, &second, 80);
    assert_eq!(
        read_second, 40,
        "the second turn reads only its 40 new tokens"
    );
    assert_close(&logits, &cold(&f, &second, &[50, 80]));

    let third = next_turn(&second, 80, 25, 5);
    let (logits, read_third) = read(&mut b, &third, 95);
    assert_eq!(read_third, 25, "and the third its own");
    assert_close(&logits, &cold(&f, &third, &[50, 80, 95]));
}

/// Two conversations that take turns each keep their own state: the second does not evict the
/// first.
#[test]
fn interleaved_conversations_each_read_only_their_new_tokens() {
    let f = Fixture::new("conv-interleaved");
    let mut b = GenericBackend(f.model());

    let a1 = conversation(1, 70);
    let b1 = conversation(2, 55);
    assert_eq!(read(&mut b, &a1, 60).1, 70);
    assert_eq!(read(&mut b, &b1, 45).1, 55);

    let a2 = next_turn(&a1, 60, 30, 7);
    let (logits, n) = read(&mut b, &a2, 80);
    assert_eq!(n, 30, "the first conversation's turn, after the other's");
    assert_close(&logits, &cold(&f, &a2, &[60, 80]));

    let b2 = next_turn(&b1, 45, 20, 9);
    let (logits, n) = read(&mut b, &b2, 55);
    assert_eq!(n, 20, "the second conversation's turn, after the first's");
    assert_close(&logits, &cold(&f, &b2, &[45, 55]));

    let a3 = next_turn(&a2, 80, 15, 2);
    assert_eq!(read(&mut b, &a3, 90).1, 15);
}

/// A prompt the same as the last, read again, reads only what follows the shared point.
#[test]
fn the_same_prompt_again_reads_its_tail() {
    let f = Fixture::new("conv-repeat");
    let mut b = GenericBackend(f.model());
    let turn = prompt(64);
    assert_eq!(read(&mut b, &turn, 56).1, 64);
    let (logits, n) = read(&mut b, &turn, 56);
    assert_eq!(n, 8);
    assert_close(&logits, &cold(&f, &turn, &[56]));
}

/// A state is dropped once its lease has run out, and a turn after that reads its prompt whole.
#[test]
fn a_state_past_its_lease_is_dropped() {
    let f = Fixture::new("conv-lease");
    let mut b = GenericBackend(f.model());
    let first = prompt(60);
    // The lease is a multiple of how long a read takes on this machine at this moment, so a
    // loaded machine cannot let it run out between two reads of the test.
    let started = std::time::Instant::now();
    read(&mut b, &first, 50);
    let lease = (started.elapsed() * 20).max(std::time::Duration::from_millis(1500));
    b.set_kv_lease(Some(lease));

    let soon = next_turn(&first, 50, 10, 1);
    assert_eq!(read(&mut b, &soon, 55).1, 10, "within the lease");

    std::thread::sleep(lease + std::time::Duration::from_millis(700));
    let late = next_turn(&soon, 55, 10, 4);
    assert_eq!(
        read(&mut b, &late, 60).1,
        65,
        "past the lease the prompt is read whole"
    );
}

/// With no lease a state stays until memory is wanted.
#[test]
fn without_a_lease_a_state_stays() {
    let f = Fixture::new("conv-no-lease");
    let mut b = GenericBackend(f.model());
    b.set_kv_lease(None);
    let first = prompt(60);
    read(&mut b, &first, 50);
    std::thread::sleep(std::time::Duration::from_millis(120));
    let again = next_turn(&first, 50, 10, 1);
    assert_eq!(read(&mut b, &again, 55).1, 10);
}

/// When memory is wanted, a conversation that came back stays and one that never did goes first.
#[test]
fn a_conversation_that_came_back_outlasts_one_that_never_did() {
    let f = Fixture::new("conv-rank");
    let mut m = f.model();
    let a = conversation(1, 40);
    let b = conversation(2, 40);
    let (la, lb) = (a[..30].to_vec(), b[..30].to_vec());
    for (turn, shared) in [(&a, &la), (&b, &lb)] {
        let x = Tensor::new(&turn[..], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        m.forward(&x.narrow(1, 0, shared.len()).unwrap(), 0)
            .unwrap();
        assert!(m.snapshot_boundary(shared).unwrap());
        m.trim_kv(0);
    }
    // The first conversation's next turn restores its state; the second's was never asked for.
    let (covered, _) = m
        .restore_prefix(&next_turn(&a, 30, 5, 3), false)
        .unwrap()
        .unwrap();
    assert_eq!(covered, 30);

    let first_to_go = m
        .kv_snapshots
        .iter()
        .min_by_key(|s| s.rank())
        .map(|s| s.tokens.clone())
        .unwrap();
    assert_eq!(
        first_to_go, lb,
        "the conversation that never came back goes first"
    );
}

/// A chunk that runs out of memory is read again smaller, from the state the span began at, and
/// reads the same prompt as one that never failed.
#[test]
fn a_chunk_that_runs_out_of_memory_is_retried_smaller() {
    let f = Fixture::new("conv-oom");
    let mut b = GenericBackend(f.model());
    let first = prompt(60);
    read(&mut b, &first, 50);
    let second = next_turn(&first, 50, 150, 3);

    FORCED_CHUNK_ROWS.with(|c| c.set(40));
    let (logits, n) = read(&mut b, &second, 120);
    FORCED_CHUNK_ROWS.with(|c| c.set(usize::MAX));
    crate::inference::engine::llm_engine::reset_adaptive_prefill_chunk();

    assert_eq!(n, 150, "the span read was the same");
    // Reading in chunks of 35, the first half of the span that fits, from each span's start.
    assert_close(&logits, &cold(&f, &second, &[50, 85, 120, 155, 190]));
}
