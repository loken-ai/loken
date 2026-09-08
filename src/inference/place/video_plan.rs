//! What to render, and how long it will take, from what someone actually asked for.
//!
//! The interface asks for a resolution, a frame count, a step count, a guidance scale, a
//! shift, a sampler and a checkpoint. Almost nobody wants to answer those, and answering
//! them wrongly is how a render turns into an afternoon. What a person means is: this
//! description, this long, roughly this good.
//!
//! So the settings are DERIVED, and the cost is stated before anything starts. The estimate
//! is not a guess dressed as a number: it is one measured render scaled by the two things
//! that actually move it - how many tokens a denoising pass sees, and how many passes there
//! are. Both are known before the first one runs.

/// How good, against how long to wait. The only dial worth showing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum Quality {
    /// Fewest passes at a modest frame, for looking at an idea.
    Draft,
    /// What a distilled checkpoint is meant for.
    #[default]
    Standard,
    /// More passes and a larger frame, when the result is the point.
    Fine,
}

/// The settings a render will use, and what it is expected to cost.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Plan {
    pub width: usize,
    pub height: usize,
    /// Output frames. A temporal VAE divides by four, so the count is 4k+1.
    pub frames: usize,
    pub steps: usize,
    pub seconds: f32,
    /// Denoising passes over the whole clip - what the estimate is really counting.
    pub passes: usize,
    pub estimated_seconds: f32,
    /// Said plainly, because a number without its basis invites more trust than it has
    /// earned.
    pub basis: String,
}

/// Frames per second of finished video. The checkpoints here are trained at this rate, so
/// it is a property of the model rather than a preference.
const FPS: usize = 16;

/// Seconds of denoising per million token-passes on the SMALL checkpoint, measured: a
/// 30-second clip at 512 square is 21504 tokens a pass over 24 passes, and its denoise took
/// 167 s.
///
/// Anchored on the DENOISE alone, deliberately. The first version of this scaled from whole
/// render times and was wrong by half, because loading and decoding are a large part of a
/// short render and neither grows with the model or the clip - folding them into a
/// per-token rate makes every extrapolation from it drift.
const SECONDS_PER_MTOKEN_PASS: f32 = 167.0 / (21_504.0 * 24.0 / 1e6);

/// What the wide checkpoint costs against the small one, per token-pass: 25 s a pass
/// against 7. Taken from the same two renders, again on the denoise only - measured on
/// TOTALS the same pair reads as 2.3, which is the dilution the comment above describes.
const WIDE_MODEL_FACTOR: f32 = (100.0 / 4.0) / (167.0 / 24.0);

/// Round a frame count to what a temporal VAE can produce: 4k+1.
fn latent_aligned(frames: usize) -> usize {
    let k = (frames.max(1).saturating_sub(1)).div_ceil(4);
    k * 4 + 1
}

/// What a render of exactly THESE settings will cost: the number of denoising passes, and
/// the seconds they are expected to take.
///
/// Separate from [`plan`] because the two questions are different. One is "choose for me";
/// this is "I have chosen - what does it cost?", which is what an interface showing its own
/// controls needs. Both go through the same arithmetic, so the estimate a caller is shown
/// cannot drift from the estimate the planner uses.
pub fn estimate(
    width: usize,
    height: usize,
    frames: usize,
    steps: usize,
    wide: bool,
) -> (usize, f32) {
    let per_frame_tokens = (width.max(16) / 16) * (height.max(16) / 16);
    let latent_frames = (frames.max(1) - 1) / 4 + 1;
    let window = latent_frames.min(21).max(1);
    let tokens = window * per_frame_tokens;
    let windows = latent_frames.div_ceil(window);
    let passes = steps.max(1) * windows;
    let mut est = (tokens as f32 * passes as f32 / 1e6) * SECONDS_PER_MTOKEN_PASS;
    if wide {
        est *= WIDE_MODEL_FACTOR;
    }
    (passes, est)
}

/// Turn "this long, roughly this good" into settings and a cost.
///
/// `wide_model` is whether the checkpoint chosen is the large one - the caller knows which
/// it resolved, and the difference is more than two to one, so hiding it would make the
/// estimate wrong by more than any of the choices here.
pub fn plan(seconds: f32, quality: Quality, wide_model: bool) -> Plan {
    let seconds = seconds.max(1.0);
    let (side, steps) = match quality {
        Quality::Draft => (384usize, 4usize),
        Quality::Standard => (512, 4),
        Quality::Fine => (768, 8),
    };
    let frames = latent_aligned((seconds * FPS as f32).round() as usize);
    // Tokens a pass: the denoiser walks windows of its own trained length, so a longer clip
    // costs more passes rather than a bigger one.
    let (passes, est) = estimate(side, side, frames, steps, wide_model);
    let latent_frames = (frames - 1) / 4 + 1;
    let tokens = latent_frames.min(21).max(1) * (side / 16) * (side / 16);
    Plan {
        width: side,
        height: side,
        frames,
        steps,
        seconds: frames as f32 / FPS as f32,
        passes,
        estimated_seconds: est,
        basis: format!(
            "{passes} denoising passes over {tokens} tokens, scaled from a measured render \
             of the same family; loading and decoding are not counted"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frame count must be one a temporal VAE can actually produce.
    #[test]
    fn a_frame_count_is_always_four_k_plus_one() {
        for s in [1.0f32, 2.0, 5.0, 30.0, 60.0] {
            let p = plan(s, Quality::Standard, false);
            assert_eq!((p.frames - 1) % 4, 0, "{} frames at {s}s", p.frames);
        }
    }

    /// Asking for longer must cost more. A plan that flattens out is a plan that lies about
    /// the one thing someone is choosing.
    #[test]
    fn a_longer_clip_costs_more() {
        let short = plan(5.0, Quality::Standard, false);
        let long = plan(30.0, Quality::Standard, false);
        assert!(long.frames > short.frames);
        assert!(long.estimated_seconds > short.estimated_seconds);
    }

    /// And so must asking for better.
    #[test]
    fn better_costs_more_than_faster() {
        let d = plan(10.0, Quality::Draft, false);
        let s = plan(10.0, Quality::Standard, false);
        let f = plan(10.0, Quality::Fine, false);
        assert!(d.estimated_seconds < s.estimated_seconds);
        assert!(s.estimated_seconds < f.estimated_seconds);
        assert_eq!(
            (d.frames, s.frames),
            (f.frames, f.frames),
            "duration is not the dial"
        );
    }

    /// Both anchors are real renders, so the plans that match them must land near what they
    /// actually took - not at some number the arithmetic happened to produce. This is what
    /// caught the first version scaling from whole render times: it read half again over on
    /// the clip it was supposed to reproduce.
    #[test]
    fn the_anchor_renders_reproduce_their_own_measurements() {
        // 30 s at 512 square on the small checkpoint: denoise measured at 167 s.
        let small = plan(30.0, Quality::Standard, false);
        assert_eq!((small.width, small.steps), (512, 4));
        let err = (small.estimated_seconds - 167.0).abs() / 167.0;
        assert!(
            err < 0.05,
            "estimated {:.0}s against a measured 167s",
            small.estimated_seconds
        );

        // 81 frames at 512 square on the wide one: denoise measured at 100 s.
        let wide = plan(81.0 / FPS as f32, Quality::Standard, true);
        assert_eq!((wide.frames, wide.passes), (81, 4));
        let err = (wide.estimated_seconds - 100.0).abs() / 100.0;
        assert!(
            err < 0.05,
            "estimated {:.0}s against a measured 100s",
            wide.estimated_seconds
        );
    }

    /// The wide checkpoint is more than twice the cost at the same geometry; an estimate
    /// that ignored which one runs would be wrong by more than every other choice combined.
    #[test]
    fn the_wide_model_is_charged_for() {
        let a = plan(10.0, Quality::Standard, false);
        let b = plan(10.0, Quality::Standard, true);
        assert!(b.estimated_seconds > a.estimated_seconds * 3.0);
    }
}
