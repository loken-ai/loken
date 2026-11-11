//! The parts a text splits into when one call of the talker cannot hold its speech.

/// The characters that end a sentence.
const SENTENCE_ENDS: [char; 6] = ['.', '!', '?', '\u{3002}', '\u{ff01}', '\u{ff1f}'];

/// The parts of `text`: its paragraphs when it holds more than one, else its sentences.
///
/// A paragraph ends at a blank line.
/// A sentence ends at a closing mark followed by a space or by the end of the text, the mark
/// kept with it.
/// Each part is trimmed, and an empty part is dropped.
pub fn parts(text: &str) -> Vec<String> {
    let paragraphs = paragraphs(text);
    if paragraphs.len() > 1 {
        return paragraphs;
    }
    sentences(text)
}

/// The paragraphs of `text`.
fn paragraphs(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            push(&mut out, &current.join("\n"));
            current.clear();
        } else {
            current.push(line);
        }
    }
    push(&mut out, &current.join("\n"));
    out
}

/// The sentences of `text`.
fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((at, c)) = chars.next() {
        if !SENTENCE_ENDS.contains(&c) {
            continue;
        }
        let end = at + c.len_utf8();
        let closes = match chars.peek() {
            None => true,
            Some((_, next)) => next.is_whitespace() || c.len_utf8() > 1,
        };
        if closes {
            push(&mut out, &text[start..end]);
            start = end;
        }
    }
    push(&mut out, &text[start..]);
    out
}

/// `part` trimmed, added to `out` unless it is empty.
fn push(out: &mut Vec<String>, part: &str) {
    let part = part.trim();
    if !part.is_empty() {
        out.push(part.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_text_of_paragraphs_splits_at_its_blank_lines() {
        let text = "First line.\nSame paragraph.\n\n  \nSecond one. Two sentences.\n";
        assert_eq!(
            parts(text),
            vec![
                "First line.\nSame paragraph.".to_string(),
                "Second one. Two sentences.".to_string()
            ]
        );
    }

    #[test]
    fn a_paragraph_splits_at_its_sentences() {
        let text = "It is 3.5 metres. Is it? Yes! 你好。再见。 Last";
        assert_eq!(
            parts(text),
            vec![
                "It is 3.5 metres.",
                "Is it?",
                "Yes!",
                "你好。",
                "再见。",
                "Last"
            ]
        );
    }

    #[test]
    fn a_single_sentence_is_its_only_part() {
        assert_eq!(parts("  One sentence.  "), vec!["One sentence."]);
        assert!(parts(" \n\n ").is_empty());
    }
}
