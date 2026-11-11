//! The rows of the talker's prompt.
//!
//! A row is the sum of a text part and a codec part, either of which may be absent.
//! A text part is a text id read through the text table and its projection.
//! A codec part is a codec id read through the talker's table.

use super::config::SpeechIds;
use crate::tensor::{Error, Result};

/// The turn that carries the design text, around it.
///
/// The reference wrapper writes these turns, and no file of the checkpoint states them.
pub const DESIGN_TURN: (&str, &str) = ("<|im_start|>user\n", "<|im_end|>\n");
/// The turn that carries the text to speak, around it.
pub const SPEECH_TURN: (&str, &str) = (
    "<|im_start|>assistant\n",
    "<|im_end|>\n<|im_start|>assistant\n",
);

/// One row of the prompt: a text id and a codec id, each read when present.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Row {
    pub text: Option<u32>,
    pub codec: Option<u32>,
}

/// The ids of the two turns of a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Turns {
    /// The whole design turn, empty when the request has no design text.
    pub design: Vec<u32>,
    /// The opening of the speech turn.
    pub role: Vec<u32>,
    /// The text to speak.
    pub text: Vec<u32>,
}

/// The ids of `turn` around `body`, the whole turn and the counts of its opening and closing.
fn turn_ids(
    tokenizer: &tokenizers::Tokenizer,
    turn: (&str, &str),
    body: &str,
) -> Result<(Vec<u32>, usize, usize)> {
    let ids = |s: &str| -> Result<Vec<u32>> {
        Ok(tokenizer
            .encode(s, false)
            .map_err(|e| Error::msg(format!("tokenizer: {e}")))?
            .get_ids()
            .to_vec())
    };
    let whole = ids(&format!("{}{body}{}", turn.0, turn.1))?;
    let (open, close) = (ids(turn.0)?, ids(turn.1)?);
    if !whole.starts_with(&open)
        || !whole.ends_with(&close)
        || whole.len() < open.len() + close.len()
    {
        return Err(Error::msg("the turn's text merges with its markers"));
    }
    Ok((whole, open.len(), close.len()))
}

impl Turns {
    /// The ids of `text` and of the optional `design` text in their turns.
    pub fn of(tokenizer: &tokenizers::Tokenizer, text: &str, design: Option<&str>) -> Result<Self> {
        let design = match design.filter(|d| !d.is_empty()) {
            Some(d) => turn_ids(tokenizer, DESIGN_TURN, d)?.0,
            None => Vec::new(),
        };
        let (speech, open, close) = turn_ids(tokenizer, SPEECH_TURN, text)?;
        Ok(Self {
            design,
            role: speech[..open].to_vec(),
            text: speech[open..speech.len() - close].to_vec(),
        })
    }
}

/// The rows of the prompt of `turns` in `language`, `None` leaving the language to the model.
///
/// The design turn, then the opening of the speech turn, each id a text row.
/// Then the control codes, each beside the pad text row, the last beside the start of text.
/// Then each text id beside the codec pad, and the end of text beside it.
/// Then the pad text row beside the start of codes.
pub fn rows(ids: &SpeechIds, turns: &Turns, language: Option<u32>) -> Vec<Row> {
    let text = |id: u32| Row {
        text: Some(id),
        codec: None,
    };
    let both = |t: u32, c: u32| Row {
        text: Some(t),
        codec: Some(c),
    };
    let mut out: Vec<Row> = turns
        .design
        .iter()
        .chain(&turns.role)
        .map(|&id| text(id))
        .collect();
    let control: Vec<u32> = match language {
        Some(lang) => vec![
            ids.codec_think,
            ids.codec_think_bos,
            lang,
            ids.codec_think_eos,
            ids.codec_pad,
        ],
        None => vec![
            ids.codec_nothink,
            ids.codec_think_bos,
            ids.codec_think_eos,
            ids.codec_pad,
        ],
    };
    let last = control.len() - 1;
    for (i, &code) in control.iter().enumerate() {
        out.push(both(if i < last { ids.tts_pad } else { ids.tts_bos }, code));
    }
    for &id in turns.text.iter().chain(std::iter::once(&ids.tts_eos)) {
        out.push(both(id, ids.codec_pad));
    }
    out.push(both(ids.tts_pad, ids.codec_bos));
    out
}
