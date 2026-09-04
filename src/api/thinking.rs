//! Where a model's reasoning ends and its answer begins: the `<think>...</think>` block
//! reasoning models emit, split out of the text so each surface can carry it in the
//! field its clients read, streamed or whole.

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

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
    pending: String,
    content_started: bool,
}

impl ThinkSplit {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, chunk: &str) -> Vec<Segment> {
        self.pending.push_str(chunk);
        let mut out = Vec::new();
        loop {
            let tag = if self.inside { CLOSE } else { OPEN };
            match self.pending.find(tag) {
                Some(i) => {
                    let before = self.pending[..i].to_string();
                    self.pending = self.pending[i + tag.len()..].to_string();
                    self.emit(&mut out, before);
                    self.inside = !self.inside;
                }
                None => {
                    let keep = partial_tag_suffix(&self.pending, tag);
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
fn partial_tag_suffix(s: &str, tag: &str) -> usize {
    let max = (tag.len() - 1).min(s.len());
    (1..=max)
        .rev()
        .find(|&k| s.is_char_boundary(s.len() - k) && tag.starts_with(&s[s.len() - k..]))
        .unwrap_or(0)
}

/// The whole-text form: the reasoning, if any, and the answer.
pub fn split_thinking(text: &str) -> (Option<String>, String) {
    let mut split = ThinkSplit::new();
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
    fn whole_text_splits_reasoning_from_answer() {
        assert_eq!(
            split_thinking("<think>\nplan\n</think>\n\nanswer"),
            (Some("plan".to_string()), "answer".to_string())
        );
        assert_eq!(split_thinking("just text"), (None, "just text".to_string()));
        assert_eq!(split_thinking("<think>\n\n</think>\n\nhi"), (None, "hi".to_string()));
    }

    #[test]
    fn a_tag_cut_across_chunks_is_read_whole() {
        let mut s = ThinkSplit::new();
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
        let mut s = ThinkSplit::new();
        let mut segs = s.push("a < b");
        segs.extend(s.finish());
        assert_eq!(segs, vec![Segment::Content("a < b".into())]);
    }
}
