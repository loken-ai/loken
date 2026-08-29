//! Auto-tuning speculative decoding configuration.
//!
//! Decides at runtime whether speculative decoding is profitable based on
//! measured hardware timings. Works across heterogeneous device configurations:
//! CUDA+OpenCL, CUDA+CPU, multi-GPU, etc.
//!
//! The core insight: spec decode is profitable when the slow device is
//! significantly slower than the fast device AND the draft model produces
//! tokens that the full model accepts often enough.

/// Measured per-device timing profile, populated during calibration.
#[derive(Debug, Clone)]
pub struct SpeculativeProfile {
    /// Time for a single normal forward pass (all layers, all devices)
    pub baseline_ms: f64,
    /// Time for a draft forward (fast device layers only + norm + project)
    pub draft_ms: f64,
    /// Time for a single verify forward (slow device layers only)
    pub verify_ms: f64,
    /// Device transfer overhead per token (CUDA↔OpenCL/CPU)
    pub transfer_ms: f64,
    /// Fraction of model layers on the fast (draft) device
    pub draft_layer_fraction: f64,
    /// Number of layers on fast device
    pub draft_layers: usize,
    /// Number of layers on slow device
    pub verify_layers: usize,
    /// Kind of slow device ("opencl", "cpu", "cuda")
    pub slow_device: String,
    /// Measured acceptance rate (updated during generation)
    pub acceptance_rate: f64,
    /// Number of calibration samples collected
    pub samples: usize,
}

impl SpeculativeProfile {
    /// Estimate acceptance rate based on draft layer fraction.
    /// Empirical approximation from self-speculative decoding research:
    /// - 80% of layers -> ~80% acceptance
    /// - 50% of layers -> ~55% acceptance
    /// - 40% of layers -> ~35% acceptance
    /// - 25% of layers -> ~15% acceptance
    ///
    /// These assume contiguous early layers (first N layers as draft).
    /// Layer-skipping drafts would have higher acceptance at same fraction.
    pub fn estimated_acceptance(draft_fraction: f64) -> f64 {
        // Piecewise linear approximation
        if draft_fraction >= 0.8 {
            0.7 + (draft_fraction - 0.8) * 1.5 // 0.8->0.7, 1.0->1.0
        } else if draft_fraction >= 0.5 {
            0.35 + (draft_fraction - 0.5) * 1.167 // 0.5->0.35, 0.8->0.7
        } else if draft_fraction >= 0.25 {
            0.10 + (draft_fraction - 0.25) * 1.0 // 0.25->0.10, 0.5->0.35
        } else {
            draft_fraction * 0.4 // Very low acceptance below 25%
        }
    }

    /// Compute expected speedup for a given K (draft tokens per cycle).
    /// Returns speedup ratio (>1.0 means spec is faster, <1.0 means slower).
    pub fn predicted_speedup(&self, k: usize) -> f64 {
        let acceptance = if self.samples >= 4 {
            self.acceptance_rate // Use measured rate if we have enough samples
        } else {
            Self::estimated_acceptance(self.draft_layer_fraction)
        };
        speedup_ratio(
            self.baseline_ms,
            self.draft_ms,
            self.verify_ms,
            self.transfer_ms,
            k,
            acceptance,
        )
    }

    /// Find the optimal K value (1-8) and its predicted speedup.
    pub fn optimal_k(&self) -> (usize, f64) {
        let mut best_k = 1;
        let mut best_speedup = 0.0f64;
        for k in 1..=8 {
            let s = self.predicted_speedup(k);
            if s > best_speedup {
                best_speedup = s;
                best_k = k;
            }
        }
        (best_k, best_speedup)
    }
}

/// Core speedup calculation.
///
/// In one spec cycle with K draft tokens:
/// - Cost: K * draft_ms + K * (verify_ms + transfer_ms) + transfer_ms (initial)
/// - Tokens produced: 1 + acceptance * K (always get 1 from verify, plus accepted drafts)
/// - Baseline cost for same tokens: (1 + acceptance * K) * baseline_ms
///
/// speedup = baseline_tokens_time / spec_cycle_time
pub fn speedup_ratio(
    baseline_ms: f64,
    draft_ms: f64,
    verify_ms: f64,
    transfer_ms: f64,
    k: usize,
    acceptance: f64,
) -> f64 {
    let k = k as f64;
    // Spec cycle cost: draft K tokens + verify K tokens + transfers
    let spec_cycle_ms = k * draft_ms + k * (verify_ms + transfer_ms) + transfer_ms;
    // Expected tokens from one cycle: at minimum 1 (from verify), plus accepted drafts
    let expected_tokens = 1.0 + acceptance * k;
    // Baseline cost for the same number of tokens
    let baseline_cost = expected_tokens * baseline_ms;

    if spec_cycle_ms <= 0.0 {
        return 0.0;
    }
    baseline_cost / spec_cycle_ms
}

/// Decision engine: should we use speculative decoding?
#[derive(Debug, Clone)]
pub struct SpeculativeDecision {
    /// Whether to enable speculative decoding
    pub enabled: bool,
    /// Chosen K value
    pub k: usize,
    /// Predicted speedup ratio
    pub predicted_speedup: f64,
    /// Human-readable reason for the decision
    pub reason: String,
}

/// Minimum speedup ratio to enable spec decode (accounts for overhead not in model)
const MIN_SPEEDUP_THRESHOLD: f64 = 1.15;

/// Minimum draft layer fraction to even consider spec decode
const MIN_DRAFT_FRACTION: f64 = 0.15;

/// Decide whether to enable speculative decoding based on a profile.
pub fn decide(profile: &SpeculativeProfile) -> SpeculativeDecision {
    // Gate 1: Need enough draft layers for meaningful predictions
    if profile.draft_layer_fraction < MIN_DRAFT_FRACTION {
        return SpeculativeDecision {
            enabled: false,
            k: 0,
            predicted_speedup: 0.0,
            reason: format!(
                "Draft too weak: only {:.0}% of layers on fast device (need >{:.0}%)",
                profile.draft_layer_fraction * 100.0,
                MIN_DRAFT_FRACTION * 100.0
            ),
        };
    }

    // Gate 2: Need a meaningful speed differential between devices
    // If baseline is fast (all on one fast device), spec decode adds overhead
    if profile.verify_ms < profile.draft_ms * 1.5 {
        return SpeculativeDecision {
            enabled: false,
            k: 0,
            predicted_speedup: 0.0,
            reason: format!(
                "Devices too balanced: verify={:.1}ms vs draft={:.1}ms (need >1.5x ratio)",
                profile.verify_ms, profile.draft_ms
            ),
        };
    }

    // Find optimal K
    let (best_k, best_speedup) = profile.optimal_k();

    // Gate 3: Predicted speedup must exceed threshold
    if best_speedup < MIN_SPEEDUP_THRESHOLD {
        return SpeculativeDecision {
            enabled: false,
            k: 0,
            predicted_speedup: best_speedup,
            reason: format!("Predicted speedup {:.2}x below threshold {:.2}x (draft={:.0}ms verify={:.0}ms accept={:.0}%)",
                best_speedup, MIN_SPEEDUP_THRESHOLD,
                profile.draft_ms, profile.verify_ms,
                if profile.samples >= 4 { profile.acceptance_rate } else { SpeculativeProfile::estimated_acceptance(profile.draft_layer_fraction) } * 100.0),
        };
    }

    SpeculativeDecision {
        enabled: true,
        k: best_k,
        predicted_speedup: best_speedup,
        reason: format!("Spec decode profitable: K={} speedup={:.2}x (draft={:.0}ms verify={:.0}ms accept≈{:.0}%)",
            best_k, best_speedup,
            profile.draft_ms, profile.verify_ms,
            if profile.samples >= 4 { profile.acceptance_rate } else { SpeculativeProfile::estimated_acceptance(profile.draft_layer_fraction) } * 100.0),
    }
}

/// Auto-calibration state machine. Collects timing measurements during the first
/// N tokens of generation, then makes a decision.
#[derive(Debug)]
pub struct SpeculativeCalibrator {
    /// Calibration phase: Normal generation to measure baseline
    baseline_timings: Vec<f64>,
    /// Calibration phase: Draft timings
    draft_timings: Vec<f64>,
    /// Calibration phase: Verify timings
    verify_timings: Vec<f64>,
    /// Transfer overhead measurements
    transfer_timings: Vec<f64>,
    /// Acceptance tracking: (drafted, accepted)
    acceptance_counts: (usize, usize),
    /// Number of baseline tokens to collect before switching to spec calibration
    baseline_target: usize,
    /// Number of spec tokens to collect for calibration
    spec_target: usize,
    /// Layer info from the model
    draft_layers: usize,
    total_layers: usize,
    slow_device: String,
    /// Final decision (None = still calibrating)
    decision: Option<SpeculativeDecision>,
}

impl SpeculativeCalibrator {
    /// Create a new calibrator.
    /// `draft_layers`: number of layers on the fast device
    /// `total_layers`: total number of layers in the model
    /// `slow_device`: type of slow device ("opencl", "cpu", etc.)
    pub fn new(draft_layers: usize, total_layers: usize, slow_device: &str) -> Self {
        Self {
            baseline_timings: Vec::with_capacity(8),
            draft_timings: Vec::with_capacity(8),
            verify_timings: Vec::with_capacity(8),
            transfer_timings: Vec::with_capacity(8),
            acceptance_counts: (0, 0),
            baseline_target: 4,
            spec_target: 4,
            draft_layers,
            total_layers,
            slow_device: slow_device.to_string(),
            decision: None,
        }
    }

    /// Current phase: are we collecting baseline or spec measurements?
    pub fn phase(&self) -> CalibrationType {
        if self.decision.is_some() {
            CalibrationType::Done
        } else if self.baseline_timings.len() < self.baseline_target {
            CalibrationType::Baseline
        } else if self.draft_timings.len() < self.spec_target {
            CalibrationType::Speculative
        } else {
            CalibrationType::Done
        }
    }

    /// Record a baseline (normal forward) timing.
    pub fn record_baseline(&mut self, ms: f64) {
        self.baseline_timings.push(ms);
    }

    /// Record a draft forward timing.
    pub fn record_draft(&mut self, ms: f64) {
        self.draft_timings.push(ms);
    }

    /// Record a verify forward timing.
    pub fn record_verify(&mut self, ms: f64) {
        self.verify_timings.push(ms);
    }

    /// Record a transfer timing.
    pub fn record_transfer(&mut self, ms: f64) {
        self.transfer_timings.push(ms);
    }

    /// Record acceptance result for one spec cycle.
    pub fn record_acceptance(&mut self, drafted: usize, accepted: usize) {
        self.acceptance_counts.0 += drafted;
        self.acceptance_counts.1 += accepted;
    }

    /// Build a profile from collected measurements and make a decision.
    /// Returns the decision. Call this after phase() returns Done.
    pub fn finalize(&mut self) -> &SpeculativeDecision {
        // Memoise on first call. The is_some() + as_ref().unwrap() pair
        // is necessary instead of `if let Some(d) = self.decision.as_ref()`:
        // the latter holds the immutable borrow through to the assign
        // `self.decision = Some(...)` below, which the borrow checker
        // (correctly) refuses. clippy::unnecessary_unwrap doesn't reason
        // about borrowck conflicts.
        #[allow(clippy::unnecessary_unwrap)]
        if self.decision.is_some() {
            return self.decision.as_ref().unwrap();
        }

        let avg = |v: &[f64]| -> f64 {
            if v.is_empty() {
                return 0.0;
            }
            // Use median to be robust against outliers (first token is often slower)
            let mut sorted = v.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted[sorted.len() / 2]
        };

        let draft_fraction = self.draft_layers as f64 / self.total_layers as f64;
        let (drafted, accepted) = self.acceptance_counts;
        let measured_acceptance = if drafted > 0 {
            accepted as f64 / drafted as f64
        } else {
            0.0
        };

        let profile = SpeculativeProfile {
            baseline_ms: avg(&self.baseline_timings),
            draft_ms: avg(&self.draft_timings),
            verify_ms: avg(&self.verify_timings),
            transfer_ms: avg(&self.transfer_timings),
            draft_layer_fraction: draft_fraction,
            draft_layers: self.draft_layers,
            verify_layers: self.total_layers - self.draft_layers,
            slow_device: self.slow_device.clone(),
            acceptance_rate: measured_acceptance,
            samples: drafted,
        };

        let decision = decide(&profile);
        let acceptance_pct = if drafted > 0 {
            measured_acceptance * 100.0
        } else {
            SpeculativeProfile::estimated_acceptance(draft_fraction) * 100.0
        };
        tracing::info!(
            "[spec] baseline={:.0}ms draft={:.0}ms verify={:.0}ms accept={:.0}% -> {}",
            profile.baseline_ms,
            profile.draft_ms,
            profile.verify_ms,
            acceptance_pct,
            if decision.enabled {
                format!("ENABLED K={}", decision.k)
            } else {
                "DISABLED".into()
            }
        );

        self.decision = Some(decision);
        self.decision.as_ref().unwrap()
    }

    /// Get the current decision, if finalized.
    pub fn decision(&self) -> Option<&SpeculativeDecision> {
        self.decision.as_ref()
    }

    /// Access baseline timings for monitor initialization.
    pub fn baseline_timings_ref(&self) -> &[f64] {
        &self.baseline_timings
    }
}

/// Calibration phase.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CalibrationType {
    /// Collecting normal forward timings
    Baseline,
    /// Collecting spec decode timings (draft + verify)
    Speculative,
    /// Calibration complete, decision made
    Done,
}

/// Runtime monitor: tracks rolling performance and short-circuits spec decode
/// if it becomes slower than baseline during generation.
#[derive(Debug)]
pub struct SpeculativeMonitor {
    /// Baseline ms/tok from calibration
    baseline_ms: f64,
    /// Rolling window of spec cycle ms-per-token (last N cycles)
    recent_spec_ms_per_tok: Vec<f64>,
    /// Window size for rolling average
    window_size: usize,
    /// How often to check (every N cycles)
    check_interval: usize,
    /// Total cycles since last check
    cycles_since_check: usize,
    /// Whether spec decode has been short-circuited
    disabled: bool,
}

impl SpeculativeMonitor {
    /// Create a monitor from calibration results.
    pub fn new(baseline_ms: f64, _k: usize) -> Self {
        Self {
            baseline_ms,
            recent_spec_ms_per_tok: Vec::with_capacity(16),
            window_size: 8,
            check_interval: 4,
            cycles_since_check: 0,
            disabled: false,
        }
    }

    /// Record one spec cycle: total time and tokens produced.
    /// Returns true if spec decode should continue, false to short-circuit.
    pub fn record_cycle(&mut self, cycle_ms: f64, tokens_produced: usize) -> bool {
        if self.disabled {
            return false;
        }
        let ms_per_tok = if tokens_produced > 0 {
            cycle_ms / tokens_produced as f64
        } else {
            cycle_ms
        };
        self.recent_spec_ms_per_tok.push(ms_per_tok);
        if self.recent_spec_ms_per_tok.len() > self.window_size {
            self.recent_spec_ms_per_tok.remove(0);
        }
        self.cycles_since_check += 1;

        if self.cycles_since_check >= self.check_interval && self.recent_spec_ms_per_tok.len() >= 4
        {
            self.cycles_since_check = 0;
            let avg_spec = self.recent_spec_ms_per_tok.iter().sum::<f64>()
                / self.recent_spec_ms_per_tok.len() as f64;
            // Short-circuit if spec decode is slower than baseline (with 5% margin)
            if avg_spec > self.baseline_ms * 0.95 {
                tracing::info!("⚡ Short-circuit: spec decode {:.1}ms/tok > baseline {:.1}ms/tok - switching to normal", avg_spec, self.baseline_ms);
                self.disabled = true;
                return false;
            }
        }
        true
    }

    /// Whether spec decode has been disabled by the monitor.
    pub fn is_disabled(&self) -> bool {
        self.disabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speedup_ratio_basic() {
        // Baseline 400ms, draft 40ms, verify 350ms, transfer 5ms, K=2, 50% accept
        let s = speedup_ratio(400.0, 40.0, 350.0, 5.0, 2, 0.5);
        // spec_cycle = 2*40 + 2*(350+5) + 5 = 80 + 710 + 5 = 795ms
        // expected_tokens = 1 + 0.5*2 = 2.0
        // baseline_cost = 2.0 * 400 = 800ms
        // speedup = 800/795 ≈ 1.006
        assert!((s - 1.006).abs() < 0.01, "speedup={}", s);
    }

    #[test]
    fn test_speedup_cuda_cpu_pattern() {
        // CUDA+CPU: baseline 2400ms (CPU very slow), draft 40ms (CUDA), verify 2300ms, K=2, 35% accept
        let s = speedup_ratio(2400.0, 40.0, 2300.0, 10.0, 2, 0.35);
        // spec_cycle = 2*40 + 2*(2300+10) + 10 = 80 + 4620 + 10 = 4710ms
        // expected_tokens = 1 + 0.35*2 = 1.7
        // baseline_cost = 1.7 * 2400 = 4080ms
        // speedup = 4080/4710 ≈ 0.87 - not profitable!
        assert!(s < 1.0, "should not be profitable: speedup={}", s);
    }

    #[test]
    fn test_speedup_high_acceptance() {
        // With high acceptance (70%), same CUDA+CPU pattern
        let s = speedup_ratio(2400.0, 40.0, 2300.0, 10.0, 2, 0.70);
        // expected_tokens = 1 + 0.7*2 = 2.4
        // baseline_cost = 2.4 * 2400 = 5760ms
        // spec_cycle = 80 + 4620 + 10 = 4710ms
        // speedup = 5760/4710 ≈ 1.22
        assert!(
            s > 1.15,
            "should be profitable with high acceptance: speedup={}",
            s
        );
    }

    #[test]
    fn test_estimated_acceptance() {
        assert!((SpeculativeProfile::estimated_acceptance(0.80) - 0.70).abs() < 0.01);
        assert!((SpeculativeProfile::estimated_acceptance(0.50) - 0.35).abs() < 0.01);
        assert!((SpeculativeProfile::estimated_acceptance(0.25) - 0.10).abs() < 0.01);
        assert!(SpeculativeProfile::estimated_acceptance(1.0) > 0.95);
        assert!(SpeculativeProfile::estimated_acceptance(0.0) < 0.01);
    }

    #[test]
    fn test_decide_weak_draft() {
        let profile = SpeculativeProfile {
            baseline_ms: 400.0,
            draft_ms: 10.0,
            verify_ms: 380.0,
            transfer_ms: 5.0,
            draft_layer_fraction: 0.10,
            draft_layers: 4,
            verify_layers: 36,
            slow_device: "cpu".into(),
            acceptance_rate: 0.0,
            samples: 0,
        };
        let d = decide(&profile);
        assert!(!d.enabled, "should reject weak draft: {}", d.reason);
    }

    #[test]
    fn test_decide_balanced_devices() {
        // All layers on CUDA - no slow device
        let profile = SpeculativeProfile {
            baseline_ms: 40.0,
            draft_ms: 20.0,
            verify_ms: 25.0,
            transfer_ms: 1.0,
            draft_layer_fraction: 0.50,
            draft_layers: 20,
            verify_layers: 20,
            slow_device: "cuda".into(),
            acceptance_rate: 0.0,
            samples: 0,
        };
        let d = decide(&profile);
        assert!(!d.enabled, "should reject balanced: {}", d.reason);
    }

    #[test]
    fn test_calibrator_flow() {
        let mut cal = SpeculativeCalibrator::new(16, 40, "opencl");
        assert_eq!(cal.phase(), CalibrationType::Baseline);

        // Record 4 baseline timings
        for _ in 0..4 {
            cal.record_baseline(370.0);
        }
        assert_eq!(cal.phase(), CalibrationType::Speculative);

        // Record 4 spec timings
        for _ in 0..4 {
            cal.record_draft(35.0);
            cal.record_verify(330.0);
            cal.record_transfer(5.0);
            cal.record_acceptance(2, 0); // 0% acceptance
        }
        assert_eq!(cal.phase(), CalibrationType::Done);

        let decision = cal.finalize();
        // With 0% acceptance and similar verify/baseline, should be disabled
        assert!(!decision.enabled, "should be disabled: {}", decision.reason);
    }
}
