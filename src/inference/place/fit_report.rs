//! Will this model, at this size, run on these cards - and what does the answer cost?
//!
//! The interface offers models it cannot always serve well. A checkpoint that fits at
//! 384 square spills half its blocks to the host at 512, and a block on the host is not a
//! percentage slower, it is an order of magnitude: it runs a full attention over every token
//! of the clip, on every step. The user finds that out by waiting.
//!
//! Everything needed to say so in advance already exists - the weights are a file size, the
//! activation demand is derived from the request, the free memory is a probe, and the
//! manager already distinguishes "free now" from "free if the idle residents are asked for
//! it back". This puts them together and answers in one word.
//!
//! The answer depends on the REQUEST, not only on the model. The same checkpoint is green at
//! one size and red at another, so an indicator that is computed once and cached against a
//! model name will lie - and an indicator that lies is worse than none, because it is the
//! thing someone trusts instead of measuring.

/// What will happen if this render is asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Weights and the forward fit the cards as they are now.
    Fits,
    /// They fit once the idle residents hand their memory back - one reload, then full
    /// speed. Not a warning, an explanation of a pause.
    AfterReclaim,
    /// Some layers will run on the host. An order of magnitude, not a percentage.
    SpillsToHost,
    /// More than the machine has, host included.
    TooBig,
}

/// The verdict and the numbers behind it, so a caller can explain itself.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FitReport {
    pub verdict: Verdict,
    /// Checkpoint bytes as they will sit on the cards.
    pub weights: u64,
    /// Peak activation bytes of ONE forward at this geometry.
    pub activation: u64,
    /// Free VRAM across every card, right now.
    pub free_now: u64,
    /// Free VRAM plus what the idle residents would return.
    pub free_after_reclaim: u64,
    /// Layers that would not fit a card, out of the total.
    pub host_layers: usize,
    pub total_layers: usize,
    /// One line a user can read.
    pub detail: String,
}

impl FitReport {
    /// Decide from the pieces. Kept separate from any probing so it can be tested at
    /// numbers a machine does not have to have.
    ///
    /// The rule is per-CARD, not per-total: a model does not fit "20 GB across two cards"
    /// unless the layers can be split so that each card holds its own share AND the
    /// activation peak of the forward that runs on it. Summing the free memory and
    /// comparing is the mistake that puts eight blocks on the host and calls it a fit.
    pub fn decide(
        weights: u64,
        activation: u64,
        total_layers: usize,
        cards_now: &[u64],
        cards_after_reclaim: &[u64],
    ) -> Self {
        let layers = total_layers.max(1);
        let per_layer = weights / layers as u64;
        let placeable = |cards: &[u64]| -> usize {
            cards
                .iter()
                .map(|free| {
                    // Each card must keep the whole forward's peak: whichever card is
                    // executing holds it, so the headroom cannot be shared between them.
                    let budget = free.saturating_sub(activation);
                    if per_layer == 0 {
                        layers
                    } else {
                        (budget / per_layer) as usize
                    }
                })
                .sum::<usize>()
                .min(layers)
        };
        let now = placeable(cards_now);
        let after = placeable(cards_after_reclaim);
        let free_now: u64 = cards_now.iter().sum();
        let free_after: u64 = cards_after_reclaim.iter().sum();
        let gb = |b: u64| b as f64 / 1e9;
        let (verdict, host_layers, detail) = if now >= layers {
            (
                Verdict::Fits,
                0,
                format!(
                    "{:.1} GB of weights and a {:.1} GB forward fit the cards as they are",
                    gb(weights),
                    gb(activation)
                ),
            )
        } else if after >= layers {
            (
                Verdict::AfterReclaim,
                0,
                format!(
                    "fits once the resident models hand their memory back: needs \
                 {:.1} GB of weights and a {:.1} GB forward against {:.1} GB free now",
                    gb(weights),
                    gb(activation),
                    gb(free_now)
                ),
            )
        } else if after > 0 {
            (
                Verdict::SpillsToHost,
                layers - after,
                format!(
                    "{} of {} layers would run on the processor - each one attends over every \
                 token of the request, on every step",
                    layers - after,
                    layers
                ),
            )
        } else {
            (
                Verdict::TooBig,
                layers,
                format!(
                    "{:.1} GB of weights plus a {:.1} GB forward against {:.1} GB of card, \
                 reclaim included",
                    gb(weights),
                    gb(activation),
                    gb(free_after)
                ),
            )
        };
        Self {
            verdict,
            weights,
            activation,
            free_now,
            free_after_reclaim: free_after,
            host_layers,
            total_layers: layers,
            detail,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two cards of 17 GB, a 10 GB model and a 4 GB forward: it fits, and the answer must
    /// not depend on the free memory being summed.
    #[test]
    fn a_model_that_fits_says_so() {
        let r = FitReport::decide(
            10_000_000_000,
            4_000_000_000,
            40,
            &[17e9 as u64; 2],
            &[17e9 as u64; 2],
        );
        assert_eq!(r.verdict, Verdict::Fits);
        assert_eq!(r.host_layers, 0);
    }

    /// The case that started this: 18 GB of weights and an 8 GB forward on two 16.5 GB
    /// cards. Summing the free memory says 33 GB against 18 and calls it a fit; charging
    /// each card its own copy of the forward says four layers go to the host, which is what
    /// happened.
    #[test]
    fn the_forward_is_charged_to_every_card_not_shared() {
        let r = FitReport::decide(
            18_100_000_000,
            7_900_000_000,
            40,
            &[16_500_000_000, 16_500_000_000],
            &[16_500_000_000, 16_500_000_000],
        );
        assert_eq!(r.verdict, Verdict::SpillsToHost);
        assert!(r.host_layers > 0 && r.host_layers < 40, "{r:?}");
        assert!(r.detail.contains("processor"), "{}", r.detail);
    }

    /// A resident model in the way is a PAUSE, not a refusal - and saying so is the whole
    /// point of separating the two numbers.
    #[test]
    fn something_in_the_way_is_a_reload_not_a_refusal() {
        let r = FitReport::decide(
            10_000_000_000,
            4_000_000_000,
            40,
            &[2_000_000_000, 2_000_000_000],
            &[17_000_000_000, 17_000_000_000],
        );
        assert_eq!(r.verdict, Verdict::AfterReclaim);
        assert_eq!(r.host_layers, 0);
    }

    /// And when the machine simply is not big enough, say that rather than promising a
    /// slow render nobody will wait for.
    #[test]
    fn too_big_is_its_own_answer() {
        let r = FitReport::decide(
            80_000_000_000,
            20_000_000_000,
            60,
            &[8e9 as u64],
            &[8e9 as u64],
        );
        assert_eq!(r.verdict, Verdict::TooBig);
    }

    /// The verdict follows the REQUEST. The same checkpoint is fine small and spills large,
    /// which is why this can never be cached against a model name alone.
    #[test]
    fn the_same_model_answers_differently_at_two_sizes() {
        let cards = [16_500_000_000u64, 16_500_000_000];
        let small = FitReport::decide(18_100_000_000, 3_000_000_000, 40, &cards, &cards);
        let large = FitReport::decide(18_100_000_000, 7_900_000_000, 40, &cards, &cards);
        assert_eq!(small.verdict, Verdict::Fits);
        assert_eq!(large.verdict, Verdict::SpillsToHost);
    }
}
