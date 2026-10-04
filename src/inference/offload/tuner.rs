//! A feedback controller for one integer setting of the decode loop, moved by what the loop
//! measures of itself. It times decode tokens in windows, tries each neighbour of the current
//! value for one window, keeps a neighbour only when its mean token time beats the incumbent's by
//! more than the noise of the measurement, then runs on the winner before it looks again. Every
//! decision is logged. The setting stays inside the bounds it was given, which the hardware and
//! the plan set, never a value tuned on one machine.

use std::sync::Mutex;

struct State {
    value: usize,
    max: usize,
    /// Tokens in a window: as many as the model has layers, so that every layer's routing is
    /// seen as often as the model is deep.
    window: usize,
    /// Windows spent on the winner between two rounds of trials.
    exploit_windows: usize,
    phase: Phase,
    n: usize,
    sum: f64,
    sum_sq: f64,
    /// The last window of the incumbent: a trial is judged against the windows on both sides of
    /// it, so a speed that drifts as the run warms up is not taken for a gain.
    last_incumbent: (f64, f64),
    pending: Option<(usize, f64, f64)>,
    best: Option<(usize, f64)>,
    trials_left: Vec<usize>,
    name: &'static str,
}

enum Phase {
    /// Timing the current value.
    Incumbent,
    /// Timing a neighbour.
    Trial(usize),
    /// Running on the current value for this many more windows.
    Exploit(usize),
}

pub struct Tuner {
    state: Mutex<State>,
}

impl Tuner {
    /// A controller for `name`, starting at `start` and bounded by `max`, timing windows of
    /// `window` tokens and exploiting `exploit_windows` windows between rounds.
    pub fn new(
        name: &'static str,
        start: usize,
        max: usize,
        window: usize,
        exploit_windows: usize,
    ) -> Self {
        Self {
            state: Mutex::new(State {
                value: start.min(max),
                max,
                window: window.max(2),
                exploit_windows: exploit_windows.max(1),
                phase: Phase::Incumbent,
                n: 0,
                sum: 0.0,
                sum_sq: 0.0,
                last_incumbent: (0.0, 0.0),
                pending: None,
                best: None,
                trials_left: Vec::new(),
                name,
            }),
        }
    }

    /// The value to use for the next token.
    pub fn value(&self) -> usize {
        let s = self.state.lock().unwrap();
        match s.phase {
            Phase::Trial(v) => v,
            _ => s.value,
        }
    }

    /// One decode token took `ns` under the value last handed out.
    pub fn note_token(&self, ns: u64) {
        let mut s = self.state.lock().unwrap();
        s.n += 1;
        s.sum += ns as f64;
        s.sum_sq += (ns as f64) * (ns as f64);
        if s.n < s.window {
            return;
        }
        let n = s.n as f64;
        let mean = s.sum / n;
        let var = (s.sum_sq / n - mean * mean).max(0.0);
        let err = (var / n).sqrt();
        s.n = 0;
        s.sum = 0.0;
        s.sum_sq = 0.0;
        match std::mem::replace(&mut s.phase, Phase::Incumbent) {
            Phase::Incumbent => {
                if let Some((v, tm, te)) = s.pending.take() {
                    // The trial sat between two windows of the incumbent: the reference is their
                    // average.
                    let (pm, pe) = s.last_incumbent;
                    let reference = (pm + mean) / 2.0;
                    let ref_err = (pe * pe + err * err).sqrt() / 2.0;
                    let margin = 2.0 * (te * te + ref_err * ref_err).sqrt();
                    let wins = tm + margin < reference;
                    tracing::info!(
                        "{}: tried {v}: {:.1} ms per token against {:.1} at {}: {}",
                        s.name,
                        tm / 1e6,
                        reference / 1e6,
                        s.value,
                        if wins { "kept" } else { "not kept" }
                    );
                    if wins && s.best.is_none_or(|b| tm < b.1) {
                        s.best = Some((v, tm));
                    }
                } else {
                    let mut next = Vec::new();
                    if s.value < s.max {
                        next.push(s.value + 1);
                    }
                    if s.value > 0 {
                        next.push(s.value - 1);
                    }
                    s.trials_left = next;
                    s.best = None;
                }
                s.last_incumbent = (mean, err);
                s.phase = match s.trials_left.pop() {
                    Some(v) => Phase::Trial(v),
                    None => {
                        if let Some((v, _)) = s.best.take() {
                            tracing::info!("{}: now {v}", s.name);
                            s.value = v;
                        }
                        Phase::Exploit(s.exploit_windows)
                    }
                };
            }
            Phase::Trial(v) => {
                s.pending = Some((v, mean, err));
                s.phase = Phase::Incumbent;
            }
            Phase::Exploit(left) => {
                s.phase = if left > 1 {
                    Phase::Exploit(left - 1)
                } else {
                    Phase::Incumbent
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Token time as a bowl around a best value, with a little noise: the controller walks to the
    /// bottom from either side and stays inside its bounds.
    #[test]
    fn the_value_walks_to_the_fastest_and_stays_in_bounds() {
        for (start, best) in [(0usize, 3usize), (6, 3), (3, 3), (0, 0)] {
            let t = Tuner::new("test", start, 6, 8, 2);
            let mut seed = 12345u64;
            let mut noise = move || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 2.0e6
            };
            for step in 0..4000 {
                let v = t.value();
                assert!(v <= 6);
                let warm = 1.0 + 0.5 * (-(step as f64) / 600.0).exp();
                let base = (270.0e6 + 20.0e6 * (v as f64 - best as f64).abs()) * warm;
                t.note_token((base + noise()) as u64);
            }
            let settled = t.state.lock().unwrap().value;
            assert_eq!(settled, best, "from {start}");
        }
    }
}
