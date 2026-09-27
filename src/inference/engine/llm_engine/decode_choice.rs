//! How a model decodes fastest when no drafter runs, learned from its own requests.
//!
//! A captured CUDA graph saves the launch of every kernel of a token and pays for the
//! capture, the probation replays and the fixed shapes it records. Prompt lookup verifies
//! n-gram drafts in one multi-position forward and pays for the drafts it loses. Which of
//! the modes wins depends on the model, the card and the text, so it is measured: each mode
//! the model can run is tried once, and from then on each request takes the mode with the
//! lower time per decoded token, every time kept current by the requests that use it.

/// Decoded tokens below which a request's rate says more about its capture and warmup than
/// about the mode it ran in.
const MIN_TOKENS: usize = 16;
/// Weight of the newest request in the running time per token.
const WEIGHT: f64 = 0.3;

/// One way to decode a request without a drafter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DecodeMode {
    /// One token per step, replayed from a captured CUDA graph.
    Captured,
    /// One token per step, launched eagerly.
    Eager,
    /// Eager steps that verify prompt-lookup drafts.
    Lookup,
}

impl DecodeMode {
    /// Order in which untried modes are tried.
    const ALL: [DecodeMode; 3] = [DecodeMode::Captured, DecodeMode::Lookup, DecodeMode::Eager];

    fn slot(self) -> usize {
        match self {
            DecodeMode::Captured => 0,
            DecodeMode::Eager => 1,
            DecodeMode::Lookup => 2,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub(crate) struct DecodeChoice {
    /// Running seconds per decoded token, per mode.
    per_token: [Option<f64>; 3],
}

impl DecodeChoice {
    /// The mode the next request should decode in, among those the model can run.
    pub(crate) fn choose(&self, can_capture: bool, can_lookup: bool) -> DecodeMode {
        let runnable = DecodeMode::ALL.into_iter().filter(|m| match m {
            DecodeMode::Captured => can_capture,
            DecodeMode::Lookup => can_lookup,
            DecodeMode::Eager => true,
        });
        let mut best: Option<(DecodeMode, f64)> = None;
        for mode in runnable {
            let Some(t) = self.per_token[mode.slot()] else {
                return mode;
            };
            if best.is_none_or(|(_, b)| t < b) {
                best = Some((mode, t));
            }
        }
        best.map_or(DecodeMode::Eager, |(m, _)| m)
    }

    /// A request that decoded `tokens` in `secs` in `mode`.
    pub(crate) fn record(&mut self, mode: DecodeMode, tokens: usize, secs: f64) {
        if tokens < MIN_TOKENS || secs <= 0.0 {
            return;
        }
        let per_token = secs / tokens as f64;
        let slot = &mut self.per_token[mode.slot()];
        *slot = Some(match *slot {
            Some(old) => old + (per_token - old) * WEIGHT,
            None => per_token,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use DecodeMode::*;

    #[test]
    fn it_tries_each_runnable_mode_then_keeps_the_fastest() {
        let mut c = DecodeChoice::default();
        assert_eq!(c.choose(true, true), Captured);
        c.record(Captured, 100, 1.0);
        assert_eq!(c.choose(true, true), Lookup);
        c.record(Lookup, 100, 0.8);
        assert_eq!(c.choose(true, true), Eager);
        c.record(Eager, 100, 2.0);
        assert_eq!(c.choose(true, true), Lookup);
        assert_eq!(c.choose(true, false), Captured);
    }

    #[test]
    fn a_model_that_cannot_capture_compares_lookup_and_eager() {
        let mut c = DecodeChoice::default();
        assert_eq!(c.choose(false, true), Lookup);
        c.record(Lookup, 100, 2.0);
        assert_eq!(c.choose(false, true), Eager);
        c.record(Eager, 100, 1.0);
        assert_eq!(c.choose(false, true), Eager);
        assert_eq!(c.choose(false, false), Eager);
    }

    #[test]
    fn short_requests_do_not_count() {
        let mut c = DecodeChoice::default();
        c.record(Captured, MIN_TOKENS - 1, 1.0);
        assert_eq!(c.choose(true, false), Captured);
        c.record(Captured, 100, 1.0);
        assert_eq!(c.choose(true, false), Eager);
    }
}
