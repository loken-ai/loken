//! Resuming a generation on another node and getting the same tokens.
//!
//! When a node dies mid-generation the router can re-route with `prompt ++ tokens already
//! emitted` and the same seed. The client sees a pause; the cost is a re-prefill rather than a
//! lost request. That only works if sampling at position N depends on N and the seed, and on
//! nothing else.
//!
//! A sequential generator does not have that property. `StdRng::seed_from_u64(seed)` carries
//! state: after N tokens it has consumed some number of draws, and a replay reproduces the
//! sequence only if the generator is advanced by exactly that many. The count is not fixed -
//! top-k, top-p and rejection sampling each draw a different number of times depending on the
//! distribution, so it varies with the text being generated. Recovery would work on greedy
//! decoding, appear to work in testing, and diverge in production at temperature.
//!
//! Seeding per POSITION removes the state. The draw for position N is a pure function of
//! (seed, N), so a resumed generation lands on the same stream whatever happened before, and
//! two runs agree even when the number of draws per token differs.
//!
//! The same property pays elsewhere: under continuous batching a request's position within a
//! batch varies run to run, which perturbs a shared sequential generator. Position seeding is
//! indifferent to it.

use rand::SeedableRng;

/// The generator for one position of one request.
///
/// Deterministic in (seed, position) alone: no history, no batch composition, no arrival order.
pub fn rng_for_position(request_seed: u64, position: u64) -> rand::rngs::StdRng {
    // Mixing rather than adding: `seed + position` collides across requests whose seeds differ
    // by a small amount, so two concurrent requests would share draws for shifted positions.
    // SplitMix64's finaliser decorrelates adjacent inputs, which is exactly the requirement.
    let mut z = request_seed
        .wrapping_add(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(position.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    rand::rngs::StdRng::seed_from_u64(z)
}

/// Where a resumed generation picks up: the tokens already emitted are replayed as prompt, and
/// sampling continues at the position they ended on.
#[derive(Debug, Clone, PartialEq)]
pub struct ResumePoint {
    pub request_seed: u64,
    /// Prompt followed by every token already sent to the client.
    pub tokens: Vec<u32>,
    /// Position of the next token to sample - the length of what has been emitted.
    pub position: u64,
    /// Digest of the weights the interrupted node was running. A replica holding different
    /// bytes under the same name would continue the generation differently, so replay must
    /// refuse it rather than produce a subtly different answer.
    pub weights_digest: String,
}

/// Whether a digest identifies no particular bytes.
///
/// Empty, or all zeros under any prefix: both are what a catalogue writes when it has nothing
/// to say, and neither is evidence that two nodes hold the same weights.
fn names_nothing(digest: &str) -> bool {
    let hex = digest.strip_prefix("sha256:").unwrap_or(digest).trim();
    hex.is_empty() || hex.chars().all(|c| c == '0')
}

impl ResumePoint {
    /// Whether `node_digest` may continue this generation.
    ///
    /// An unknown digest is refused rather than matched. Models cached from Hugging Face carry
    /// a placeholder of all zeros - there is no single file hash for a directory of shards - so
    /// two unrelated checkpoints compare equal under it, and a replay would continue on other
    /// bytes under the same name. That is the one outcome this check exists to prevent, so the
    /// placeholder fails it on both sides.
    pub fn may_resume_on(&self, node_digest: &str) -> bool {
        !names_nothing(&self.weights_digest)
            && !names_nothing(node_digest)
            && self.weights_digest == node_digest
    }
}

#[cfg(test)]
mod tests {
    // `random_range` is a RngExt method; the trait is needed HERE and nowhere else,
    // which is why the crate-level import read as unused to a `--lib` build.
    use super::*;
    use rand::RngExt;

    /// Stand-in for a sampler: draws a variable number of times, as top-p and rejection do.
    /// A sequential generator would desynchronise here; a position-seeded one cannot.
    fn sample_at(seed: u64, position: u64, difficulty: u32) -> u32 {
        let mut rng = rng_for_position(seed, position);
        let mut last = 0u32;
        for _ in 0..=(difficulty % 5) {
            last = rng.random_range(0..50_000);
        }
        last
    }

    fn generate(seed: u64, from: u64, to: u64) -> Vec<u32> {
        (from..to).map(|p| sample_at(seed, p, p as u32)).collect()
    }

    /// The plan's invariant: a generation interrupted at token N and resumed elsewhere produces
    /// the same stream as the uninterrupted one.
    #[test]
    fn a_generation_resumed_elsewhere_produces_the_same_tokens() {
        let seed = 0x5EED_1234_ABCD_0001;
        let whole = generate(seed, 0, 200);

        for cut in [1u64, 7, 63, 128, 199] {
            let mut resumed = generate(seed, 0, cut);
            resumed.extend(generate(seed, cut, 200));
            assert_eq!(resumed, whole, "resuming at {cut} changed the stream");
        }
    }

    /// And the reason a sequential generator cannot do it: the number of draws per token
    /// varies with the text, so a replay would have to advance by a count nobody knows.
    #[test]
    fn the_draw_count_per_token_varies_which_is_what_breaks_a_sequential_rng() {
        let seed = 7;
        let counts: Vec<u32> = (0..10u32).map(|p| p % 5 + 1).collect();
        assert!(
            counts.iter().any(|c| *c != counts[0]),
            "the fixture must exercise a varying draw count or it proves nothing"
        );
        // Position seeding is indifferent to it: same position, same token, regardless of what
        // was drawn before.
        assert_eq!(sample_at(seed, 42, 3), sample_at(seed, 42, 3));
    }

    /// Two requests must not share a stream. Seeds that differ by one are the case a naive
    /// `seed + position` gets wrong: request A at position 5 would draw what request B draws
    /// at position 4.
    #[test]
    fn adjacent_seeds_do_not_share_a_stream() {
        let a: Vec<u32> = (0..64).map(|p| sample_at(1000, p, 0)).collect();
        let b: Vec<u32> = (0..64).map(|p| sample_at(1001, p, 0)).collect();
        let shifted: Vec<u32> = (1..65).map(|p| sample_at(1001, p, 0)).collect();
        assert_ne!(a, b);
        assert_ne!(
            a, shifted,
            "a shifted seed reproduced the neighbour's stream"
        );
        let overlap = a.iter().zip(&b).filter(|(x, y)| x == y).count();
        assert!(
            overlap < 4,
            "{overlap} of 64 draws coincided between adjacent seeds"
        );
    }

    /// Position seeding is what makes a request indifferent to its place in a batch: the same
    /// positions yield the same tokens whatever order they are computed in.
    #[test]
    fn a_request_is_indifferent_to_its_place_in_a_batch() {
        let seed = 99;
        let in_order: Vec<u32> = (0..32).map(|p| sample_at(seed, p, 1)).collect();
        let mut shuffled: Vec<(u64, u32)> =
            (0..32).rev().map(|p| (p, sample_at(seed, p, 1))).collect();
        shuffled.sort_by_key(|(p, _)| *p);
        let out: Vec<u32> = shuffled.into_iter().map(|(_, t)| t).collect();
        assert_eq!(out, in_order);
    }

    /// A digest that names nothing is not evidence of anything. Models cached from Hugging
    /// Face all carry the same placeholder, so matching on it would let a generation continue
    /// on unrelated weights - the failure this check exists to prevent, reached through the
    /// check itself.
    #[test]
    fn replay_refuses_a_digest_that_names_nothing() {
        let unknown = "sha256:0000000000000000000000000000000000000000";
        let r = ResumePoint {
            request_seed: 1,
            tokens: vec![1, 2, 3],
            position: 3,
            weights_digest: unknown.into(),
        };
        assert!(!r.may_resume_on(unknown), "two unknowns are not a match");
        assert!(!r.may_resume_on("sha256:aaaa"));
        // And a known point refuses an unknown replica.
        let known = ResumePoint {
            weights_digest: "sha256:aaaa".into(),
            ..r
        };
        assert!(!known.may_resume_on(unknown));
        assert!(known.may_resume_on("sha256:aaaa"));
        // Empty says nothing either.
        assert!(names_nothing(""));
        assert!(names_nothing("0000"));
        assert!(!names_nothing("sha256:00a0"));
    }

    /// Replay must refuse a replica whose weights differ, however it is named. Continuing a
    /// generation on other bytes produces a plausible answer that is not the one in progress.
    #[test]
    fn replay_refuses_a_replica_holding_different_weights() {
        let r = ResumePoint {
            request_seed: 1,
            tokens: vec![1, 2, 3],
            position: 3,
            weights_digest: "sha256:aaaa".into(),
        };
        assert!(r.may_resume_on("sha256:aaaa"));
        assert!(
            !r.may_resume_on("sha256:bbbb"),
            "same name is not the same bytes"
        );
    }
}
