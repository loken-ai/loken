//! Where a model's reasoning ends and its answer begins: the `<think>...</think>` block
//! reasoning models emit, split out of the text so each surface can carry it in the
//! field its clients read, streamed or whole.

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";
/// gpt-oss reasons on Harmony's `analysis` channel and answers on `final`; the engine
/// keeps its channel tokens in the text so they can be read here. The answer's own
/// opening and the trailing end-of-turn tokens carry nothing and are dropped.
const OPEN_HARMONY: &str = "<|channel|>analysis<|message|>";
const CLOSE_HARMONY: &str = "<|end|>";
/// gemma4 reasons on a channel too, but names its tags asymmetrically: `<|channel>` opens
/// and `<channel|>` closes, with the reasoning channel called `thought`. A build trained on
/// it opens the channel itself, so without this the channel's NAME reaches the reader as the
/// first word of the answer while the tags around it are dropped as special tokens.
/// The newline is part of the opener, not of the reasoning: it is where the channel's name
/// ends and its message begins, as `<|message|>` is for Harmony. Without it the thinking
/// segment starts with a stray newline, which the whole-text path trims away and the
/// streaming path faithfully emits.
const OPEN_GEMMA: &str = "<|channel>thought\n";
const CLOSE_GEMMA: &str = "<channel|>";
/// Every marker this module parses out of an answer, as a vocabulary spells them. A token
/// listed here is registered NOT special when the model loads, so it survives a decode that
/// still skips everything else: the splitter needs its delimiters, the reader must not see
/// the rest of the control vocabulary.
///
/// Asking instead whether a vocabulary "uses channels" was the wrong question, and it missed
/// every model that reasons with `<think>`. Those tags are user-defined tokens, so they were
/// dropped before the split could see them, and the whole qwen3, qwen3.5, deepseek-r1,
/// deepcoder, smollm3 and lfm2 line delivered its chain of thought as the answer with the
/// thinking field left empty.
pub(crate) const SPLIT_MARKERS: [&str; 9] = [
    "<think>",
    "</think>",
    "<|channel|>",
    "<|message|>",
    "<|end|>",
    "<|return|>",
    "<|start|>",
    "<|channel>",
    "<channel|>",
];
const HARMONY_NOISE: [&str; 5] = [
    "<|start|>assistant<|channel|>final<|message|>",
    "<|channel|>final<|message|>",
    "<|start|>assistant",
    "<|return|>",
    "<|end|>",
];

#[derive(Debug, PartialEq, Eq)]
pub enum Segment {
    Thinking(String),
    Content(String),
}

/// Splits a stream of text chunks. A tag cut across two chunks is held back until it
/// can be read whole; everything else leaves as soon as it arrives.
#[derive(Default)]
pub struct ThinkSplit {
    inside: bool,
    /// The tag that closes the block now open, empty when none is. Three families share this
    /// splitter and a flag cannot name three.
    close: &'static str,
    pending: String,
    content_started: bool,
}

impl ThinkSplit {
    /// A splitter for an answer whose prompt already opened the thinking block. Some
    /// templates write the opener into the generation prompt when reasoning is on - qwen3.5
    /// does - so the model's text starts inside a block it never opens itself, and a splitter
    /// that waits for the opener reads the whole reasoning as the answer.
    pub fn opened(by_prompt: bool) -> Self {
        Self {
            inside: by_prompt,
            close: if by_prompt { CLOSE } else { "" },
            ..Self::default()
        }
    }

    pub fn push(&mut self, chunk: &str) -> Vec<Segment> {
        self.pending.push_str(chunk);
        let mut out = Vec::new();
        loop {
            // The tags that may come next: a close of the open family while inside,
            // either open otherwise. The earliest one in the buffer wins.
            let closing = [self.close];
            let candidates: &[&str] = if self.inside {
                &closing
            } else {
                &[OPEN, OPEN_HARMONY, OPEN_GEMMA]
            };
            let hit = candidates
                .iter()
                .filter_map(|t| self.pending.find(t).map(|i| (i, *t)))
                .min_by_key(|(i, _)| *i);
            match hit {
                Some((i, tag)) => {
                    let before = self.pending[..i].to_string();
                    self.pending = self.pending[i + tag.len()..].to_string();
                    self.emit(&mut out, before);
                    if !self.inside {
                        self.close = if tag == OPEN_HARMONY {
                            CLOSE_HARMONY
                        } else if tag == OPEN_GEMMA {
                            CLOSE_GEMMA
                        } else {
                            CLOSE
                        };
                    }
                    self.inside = !self.inside;
                }
                None => {
                    let keep = candidates
                        .iter()
                        .chain(HARMONY_NOISE.iter())
                        .map(|t| partial_tag_suffix(&self.pending, t))
                        .max()
                        .unwrap_or(0);
                    let cut = self.pending.len() - keep;
                    let release = self.pending[..cut].to_string();
                    self.pending = self.pending[cut..].to_string();
                    self.emit(&mut out, release);
                    break;
                }
            }
        }
        out
    }

    /// What a stream that ended mid-tag still holds.
    pub fn finish(&mut self) -> Vec<Segment> {
        let rest = std::mem::take(&mut self.pending);
        let mut out = Vec::new();
        self.emit(&mut out, rest);
        out
    }

    fn emit(&mut self, out: &mut Vec<Segment>, text: String) {
        if text.is_empty() {
            return;
        }
        if self.inside {
            out.push(Segment::Thinking(text));
            return;
        }
        let text = strip_harmony_noise(&text);
        if text.is_empty() {
            return;
        }
        // The blank line a model leaves between its reasoning and its answer belongs
        // to neither.
        let text = if self.content_started {
            text
        } else {
            let t = text.trim_start();
            if t.is_empty() {
                return;
            }
            self.content_started = true;
            t.to_string()
        };
        out.push(Segment::Content(text));
    }
}

/// How many trailing bytes of `s` could be the start of `tag`.
/// Removes Harmony's turn tokens from an answer: the `final` channel opening, a
/// `<|start|>assistant` before it, and the end-of-turn tokens.
fn strip_harmony_noise(text: &str) -> String {
    let mut t = text.to_string();
    for n in HARMONY_NOISE {
        t = t.replace(n, "");
    }
    t
}

fn partial_tag_suffix(s: &str, tag: &str) -> usize {
    let max = (tag.len() - 1).min(s.len());
    (1..=max)
        .rev()
        .find(|&k| s.is_char_boundary(s.len() - k) && tag.starts_with(&s[s.len() - k..]))
        .unwrap_or(0)
}

/// Whether a rendered prompt ends inside a thinking block: an opener with no closer after
/// it. A template that pre-closes the block, as qwen3 does when reasoning is off, does not
/// count, and neither does one that leaves the opener to the model.
pub fn prompt_opens_thinking(prompt: &str) -> bool {
    match (prompt.rfind(OPEN), prompt.rfind(CLOSE)) {
        (Some(open), Some(close)) => open > close,
        (Some(_), None) => true,
        _ => false,
    }
}

/// The whole-text form: the reasoning, if any, and the answer. `by_prompt` is true when the
/// rendered prompt already opened the thinking block, so the answer starts inside it.
pub fn split_thinking_opened(by_prompt: bool, text: &str) -> (Option<String>, String) {
    let mut split = ThinkSplit::opened(by_prompt);
    let mut segments = split.push(text);
    segments.extend(split.finish());
    let mut thinking = String::new();
    let mut content = String::new();
    for s in segments {
        match s {
            Segment::Thinking(t) => thinking.push_str(&t),
            Segment::Content(c) => content.push_str(&c),
        }
    }
    let thinking = thinking.trim().to_string();
    ((!thinking.is_empty()).then_some(thinking), content)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harmony_channels_split_like_think_tags() {
        let (thinking, content) = split_thinking_opened(false,
            "<|channel|>analysis<|message|>We must classify.<|end|><|start|>assistant<|channel|>final<|message|>{\"violations\": []}<|return|>",
        );
        assert_eq!(thinking.as_deref(), Some("We must classify."));
        assert_eq!(content, "{\"violations\": []}");
    }

    #[test]
    fn harmony_answer_without_analysis_is_plain_content() {
        let (thinking, content) =
            split_thinking_opened(false, "<|channel|>final<|message|>Hello.<|return|>");
        assert_eq!(thinking, None);
        assert_eq!(content, "Hello.");
    }

    #[test]
    fn harmony_markers_survive_a_chunk_boundary() {
        let mut s = ThinkSplit::opened(false);
        let mut segs = s.push("<|channel|>anal");
        segs.extend(s.push("ysis<|message|>think<|en"));
        segs.extend(s.push("d|><|start|>assistant<|channel|>final<|message|>answer"));
        segs.extend(s.finish());
        let thinking: String = segs
            .iter()
            .filter_map(|x| match x {
                Segment::Thinking(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        let content: String = segs
            .iter()
            .filter_map(|x| match x {
                Segment::Content(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(thinking, "think");
        assert_eq!(content, "answer");
    }

    #[test]
    fn whole_text_splits_reasoning_from_answer() {
        assert_eq!(
            split_thinking_opened(false, "<think>\nplan\n</think>\n\nanswer"),
            (Some("plan".to_string()), "answer".to_string())
        );
        assert_eq!(
            split_thinking_opened(false, "just text"),
            (None, "just text".to_string())
        );
        assert_eq!(
            split_thinking_opened(false, "<think>\n\n</think>\n\nhi"),
            (None, "hi".to_string())
        );
    }

    #[test]
    fn a_tag_cut_across_chunks_is_read_whole() {
        let mut s = ThinkSplit::opened(false);
        let mut segs = Vec::new();
        for c in ["<thi", "nk>ab", "c</th", "ink>\nhello", " world"] {
            segs.extend(s.push(c));
        }
        segs.extend(s.finish());
        assert_eq!(
            segs,
            vec![
                Segment::Thinking("ab".into()),
                Segment::Thinking("c".into()),
                Segment::Content("hello".into()),
                Segment::Content(" world".into()),
            ]
        );
    }

    #[test]
    fn a_lone_angle_bracket_is_content_not_a_tag() {
        let mut s = ThinkSplit::opened(false);
        let mut segs = s.push("a < b");
        segs.extend(s.finish());
        assert_eq!(segs, vec![Segment::Content("a < b".into())]);
    }

    /// gemma4:31b and :26b open a thought channel of their own and close it empty. The tags
    /// are special tokens and were dropped on decode, so the channel's name was left behind
    /// and every answer began with the word `thought`.
    #[test]
    fn a_gemma_thought_channel_never_reaches_the_answer() {
        let (thinking, content) = split_thinking_opened(
            false,
            "<|channel>thought\n<channel|>A CSV file is plain text.",
        );
        assert_eq!(thinking, None);
        assert_eq!(content, "A CSV file is plain text.");

        let (thinking, content) =
            split_thinking_opened(false, "<|channel>thought\nweigh it up<channel|>the answer");
        assert_eq!(thinking.as_deref(), Some("weigh it up"));
        assert_eq!(content, "the answer");
    }

    #[test]
    fn gemma_channel_tags_survive_a_chunk_boundary() {
        let mut s = ThinkSplit::opened(false);
        let mut segs = s.push("<|chan");
        segs.extend(s.push("nel>thought\nplan<chan"));
        segs.extend(s.push("nel|>answer"));
        segs.extend(s.finish());
        let thinking: String = segs
            .iter()
            .filter_map(|x| match x {
                Segment::Thinking(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        let content: String = segs
            .iter()
            .filter_map(|x| match x {
                Segment::Content(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(thinking, "plan");
        assert_eq!(content, "answer");
    }

    /// qwen3.5 writes the opener into the generation prompt when reasoning is on, so the
    /// model's text starts inside the block. A splitter that waited for the opener read the
    /// whole reasoning as the answer and left the thinking field empty.
    #[test]
    fn a_block_the_prompt_opened_is_still_reasoning() {
        assert!(prompt_opens_thinking("<|im_start|>assistant\n<think>\n"));
        assert!(!prompt_opens_thinking(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ));
        assert!(!prompt_opens_thinking("<|im_start|>assistant\n"));
        assert_eq!(
            split_thinking_opened(true, "weigh it up\n</think>\n\nthe answer"),
            (Some("weigh it up".to_string()), "the answer".to_string())
        );
        // Without an opener in the prompt the same text is an answer, tag and all.
        let (thinking, _) = split_thinking_opened(false, "weigh it up\n</think>\n\nthe answer");
        assert_eq!(thinking, None);
        // Streamed: the first chunk is already reasoning.
        let mut s = ThinkSplit::opened(true);
        let mut segs = s.push("weigh it");
        segs.extend(s.push(" up</think>answer"));
        segs.extend(s.finish());
        assert_eq!(segs[0], Segment::Thinking("weigh it".into()));
        assert!(
            segs.contains(&Segment::Content("answer".into())),
            "{segs:?}"
        );
    }
}
