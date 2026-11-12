//! The redraw of an utterance that speaks past its text.
//!
//! The pace of an utterance is its frames per token of its text.
//! An utterance overruns when its pace exceeds [`TOKEN_PACE`].
//! It must also exceed [`MEDIAN_FACTOR`] times the median pace of the utterances the loaded
//! checkpoint completed in the same language for a text of a near length.
//! A near length is at least half and at most twice the token count of the text.
//! Without such an utterance, nothing overruns.
//!
//! An overrun is drawn again once, with the next seed.
//! A second draw that overruns too keeps the shorter of the 2.
//! The kept draw enters the median when it drew the end of speech.
//! The paces live in memory, for the life of the loaded checkpoint.

use super::{Result, Stop, Utterance};

/// Frames per text token an overrun exceeds.
///
/// Read on the measure of 2026-11-12 and chosen by the owner.
pub const TOKEN_PACE: f64 = 5.0;

/// Times the median pace of its language and length an overrun exceeds.
///
/// Read on the measure of 2026-11-12 and chosen by the owner.
pub const MEDIAN_FACTOR: f64 = 1.75;

/// The language an utterance is spoken in: the codec id of the prompt's language, `None` when
/// the prompt leaves it to the model.
pub type Language = Option<u32>;

/// Frames per token of a text of `tokens` tokens spoken in `frames` frames.
///
/// A text of no token counts as 1.
pub fn pace(frames: usize, tokens: usize) -> f64 {
    frames as f64 / tokens.max(1) as f64
}

/// The paces of the utterances a checkpoint completed, by language.
#[derive(Debug, Default, Clone)]
pub struct Paces {
    /// The token count of each text and its pace, by language, in increasing token count.
    ///
    /// A text of no token counts as 1.
    by_language: std::collections::HashMap<Language, Vec<(usize, f64)>>,
}

/// An utterance judged an overrun: its pace and the median pace it is judged against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Overrun {
    pub pace: f64,
    pub median: f64,
}

impl Paces {
    /// The median pace of the utterances of `language` whose text holds from half to twice
    /// `tokens` tokens, `None` when there is none.
    pub fn median(&self, language: Language, tokens: usize) -> Option<f64> {
        let spoken = self.by_language.get(&language)?;
        let tokens = tokens.max(1);
        let from = spoken.partition_point(|(t, _)| 2 * t < tokens);
        let to = spoken.partition_point(|(t, _)| *t <= tokens.saturating_mul(2));
        let mut near: Vec<f64> = spoken[from..to].iter().map(|(_, p)| *p).collect();
        near.sort_by(f64::total_cmp);
        let n = near.len();
        match n {
            0 => None,
            _ if n % 2 == 1 => Some(near[n / 2]),
            _ => Some((near[n / 2 - 1] + near[n / 2]) / 2.0),
        }
    }

    /// Adds the pace of `frames` frames for a text of `tokens` tokens to `language`.
    pub fn record(&mut self, language: Language, frames: usize, tokens: usize) {
        let spoken = self.by_language.entry(language).or_default();
        let tokens = tokens.max(1);
        let at = spoken.partition_point(|(t, _)| *t <= tokens);
        spoken.insert(at, (tokens, pace(frames, tokens)));
    }

    /// The overrun of `frames` frames for a text of `tokens` tokens in `language`, `None`
    /// when they do not overrun.
    pub fn overrun(&self, language: Language, frames: usize, tokens: usize) -> Option<Overrun> {
        let median = self.median(language, tokens)?;
        let pace = pace(frames, tokens);
        (pace > TOKEN_PACE && pace > MEDIAN_FACTOR * median).then_some(Overrun { pace, median })
    }

    /// `first`, the utterance of a text of `tokens` tokens in `language` drawn with `seed`, or
    /// the draw `draw` gives with the next seed when `first` overruns.
    ///
    /// A second draw that overruns too keeps the shorter of the 2, `first` on a tie.
    /// The kept draw enters the median when it drew the end of speech.
    pub fn redraw(
        &mut self,
        language: Language,
        tokens: usize,
        seed: u64,
        first: Utterance,
        draw: impl FnOnce(u64) -> Result<Utterance>,
    ) -> Result<Utterance> {
        let kept = match self.overrun(language, first.frames.len(), tokens) {
            None => first,
            Some(flagged) => {
                let second = draw(seed.wrapping_add(1))?;
                let (first_frames, second_frames) = (first.frames.len(), second.frames.len());
                let again = self.overrun(language, second_frames, tokens);
                let kept = match again {
                    Some(_) if second_frames >= first_frames => first,
                    _ => second,
                };
                tracing::info!(
                    "qwen3-tts: an utterance of {tokens} text tokens drew {first_frames} frames, \
                     {:.2} per token against a median of {:.2} for texts of its length in its \
                     language: drawn again with the next seed, {second_frames} frames at {:.2} \
                     per token, {} frames kept",
                    flagged.pace,
                    flagged.median,
                    pace(second_frames, tokens),
                    kept.frames.len(),
                );
                kept
            }
        };
        if kept.stop == Stop::End {
            self.record(language, kept.frames.len(), tokens);
        }
        Ok(kept)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An utterance of `frames` frames that stopped at `stop`.
    fn said(frames: usize, stop: Stop) -> Utterance {
        Utterance {
            frames: vec![vec![0]; frames],
            stop,
        }
    }

    /// The language of these tests.
    const SPOKEN: Language = Some(3);

    /// A draw that must not happen.
    fn no_draw(_: u64) -> Result<Utterance> {
        panic!("an utterance that does not overrun is not drawn again")
    }

    /// A language without a completed utterance overruns nothing.
    /// An utterance overruns past both bounds, and not at either.
    #[test]
    fn an_utterance_overruns_past_both_bounds_once_its_language_has_a_median() {
        let mut paces = Paces::default();
        assert_eq!(paces.overrun(SPOKEN, 1000, 10), None);
        paces.record(SPOKEN, 40, 10);
        assert_eq!(paces.overrun(SPOKEN, 70, 10), None, "at the median bound");
        assert_eq!(
            paces.overrun(SPOKEN, 71, 10),
            Some(Overrun {
                pace: 7.1,
                median: 4.0
            })
        );
        assert_eq!(paces.overrun(None, 71, 10), None, "another language");
        let mut slow = Paces::default();
        slow.record(SPOKEN, 20, 10);
        assert_eq!(slow.overrun(SPOKEN, 50, 10), None, "at the token bound");
        assert!(slow.overrun(SPOKEN, 51, 10).is_some());
    }

    /// The median holds every recorded pace of a near length, the middle 2 averaged.
    #[test]
    fn the_median_follows_every_recorded_pace_of_a_near_length() {
        let mut paces = Paces::default();
        assert_eq!(paces.median(SPOKEN, 10), None);
        paces.record(SPOKEN, 30, 10);
        assert_eq!(paces.median(SPOKEN, 10), Some(3.0));
        paces.record(SPOKEN, 10, 10);
        assert_eq!(paces.median(SPOKEN, 10), Some(2.0));
        paces.record(SPOKEN, 50, 10);
        assert_eq!(paces.median(SPOKEN, 10), Some(3.0));
        paces.record(SPOKEN, 7, 0);
        assert_eq!(paces.median(SPOKEN, 1), Some(7.0), "no token counts as 1");
        assert_eq!(paces.median(SPOKEN, 0), Some(7.0));
        assert_eq!(
            paces.median(SPOKEN, 10),
            Some(3.0),
            "a text of 1 token is not near 10"
        );
        assert_eq!(paces.median(None, 10), None);
    }

    /// A near length holds from half to twice the token count, both bounds included.
    #[test]
    fn a_near_length_spans_half_to_twice_the_token_count() {
        let mut paces = Paces::default();
        paces.record(SPOKEN, 10, 5);
        paces.record(SPOKEN, 60, 20);
        assert_eq!(
            paces.median(SPOKEN, 10),
            Some(2.5),
            "5 and 20 are both near 10"
        );
        assert_eq!(paces.median(SPOKEN, 9), Some(2.0), "20 is past twice 9");
        assert_eq!(paces.median(SPOKEN, 11), Some(3.0), "5 is under half of 11");
        assert_eq!(paces.median(SPOKEN, 41), None);
        assert_eq!(paces.median(SPOKEN, 2), None);
    }

    /// A short text is judged against short texts and a long one against long ones.
    /// The same pace overruns among long texts and not among short ones.
    #[test]
    fn a_text_is_judged_against_texts_of_its_own_length() {
        let mut paces = Paces::default();
        for _ in 0..8 {
            paces.record(SPOKEN, 24, 6);
            paces.record(SPOKEN, 120, 40);
        }
        assert_eq!(paces.median(SPOKEN, 6), Some(4.0));
        assert_eq!(paces.median(SPOKEN, 40), Some(3.0));
        assert_eq!(
            paces.overrun(SPOKEN, 40, 6),
            None,
            "short among short texts"
        );
        assert_eq!(
            paces.overrun(SPOKEN, 240, 40),
            Some(Overrun {
                pace: 6.0,
                median: 3.0
            }),
            "long among long texts"
        );
        let kept = paces
            .redraw(SPOKEN, 6, 7, said(40, Stop::End), no_draw)
            .unwrap();
        assert_eq!(kept.frames.len(), 40);
        let mut seeds = Vec::new();
        let kept = paces
            .redraw(SPOKEN, 40, 7, said(240, Stop::End), |seed| {
                seeds.push(seed);
                Ok(said(130, Stop::End))
            })
            .unwrap();
        assert_eq!((seeds, kept.frames.len()), (vec![8], 130));
    }

    /// A text with no completed utterance of a near length is not drawn again.
    #[test]
    fn a_text_without_neighbours_is_not_drawn_again() {
        let mut paces = Paces::default();
        for _ in 0..8 {
            paces.record(SPOKEN, 120, 40);
        }
        let kept = paces
            .redraw(SPOKEN, 6, 7, said(600, Stop::End), no_draw)
            .unwrap();
        assert_eq!(kept.frames.len(), 600);
        assert_eq!(paces.median(SPOKEN, 6), Some(100.0), "the kept draw enters");
    }

    /// An utterance that does not overrun is kept and enters the median.
    #[test]
    fn an_utterance_within_its_bounds_is_kept_and_enters_the_median() {
        let mut paces = Paces::default();
        let first = paces
            .redraw(SPOKEN, 10, 7, said(100, Stop::End), no_draw)
            .unwrap();
        assert_eq!(first.frames.len(), 100, "no median yet");
        assert_eq!(paces.median(SPOKEN, 10), Some(10.0));
        let second = paces
            .redraw(SPOKEN, 10, 7, said(60, Stop::End), no_draw)
            .unwrap();
        assert_eq!(second.frames.len(), 60);
        assert_eq!(paces.median(SPOKEN, 10), Some(8.0));
    }

    /// An overrun is drawn again with the next seed.
    /// A second draw within its bounds is kept, and only it enters the median.
    #[test]
    fn an_overrun_is_drawn_again_and_a_second_draw_within_bounds_is_kept() {
        let mut paces = Paces::default();
        paces.record(SPOKEN, 40, 10);
        let mut seeds = Vec::new();
        let kept = paces
            .redraw(SPOKEN, 10, 7, said(100, Stop::End), |seed| {
                seeds.push(seed);
                Ok(said(45, Stop::End))
            })
            .unwrap();
        assert_eq!(seeds, vec![8]);
        assert_eq!(kept.frames.len(), 45);
        assert_eq!(
            paces.median(SPOKEN, 10),
            Some(4.25),
            "the flagged draw stays out"
        );
        let wrapped = paces
            .redraw(SPOKEN, 10, u64::MAX, said(100, Stop::End), |seed| {
                assert_eq!(seed, 0, "the next seed wraps");
                Ok(said(40, Stop::End))
            })
            .unwrap();
        assert_eq!(wrapped.frames.len(), 40);
    }

    /// 2 overruns keep the shorter draw, the first on a tie.
    /// A kept draw that did not draw the end of speech stays out of the median.
    #[test]
    fn two_overruns_keep_the_shorter_draw() {
        let mut paces = Paces::default();
        paces.record(SPOKEN, 40, 10);
        let shorter_second = paces
            .redraw(SPOKEN, 10, 1, said(100, Stop::End), |_| {
                Ok(said(90, Stop::End))
            })
            .unwrap();
        assert_eq!(shorter_second.frames.len(), 90);
        assert_eq!(paces.median(SPOKEN, 10), Some(6.5));
        let mut paces = Paces::default();
        paces.record(SPOKEN, 40, 10);
        let shorter_first = paces
            .redraw(SPOKEN, 10, 1, said(90, Stop::Budget), |_| {
                Ok(said(100, Stop::End))
            })
            .unwrap();
        assert_eq!(
            (shorter_first.frames.len(), shorter_first.stop),
            (90, Stop::Budget)
        );
        assert_eq!(paces.median(SPOKEN, 10), Some(4.0), "a cut draw stays out");
        let tie = paces
            .redraw(SPOKEN, 10, 1, said(90, Stop::End), |_| {
                Ok(said(90, Stop::Asked))
            })
            .unwrap();
        assert_eq!(tie.stop, Stop::End);
    }
}
