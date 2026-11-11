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

/// The turn that carries the reference text of a clone prompt, around it.
///
/// The reference wrapper writes this turn, and no file of the checkpoint states it.
pub const REFERENCE_TURN: (&str, &str) = ("<|im_start|>assistant\n", "<|im_end|>\n");

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

/// The ids of the body of `text` in the reference turn.
pub fn reference_ids(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>> {
    let (whole, open, close) = turn_ids(tokenizer, REFERENCE_TURN, text)?;
    Ok(whole[open..whole.len() - close].to_vec())
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

/// One row of a clone prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloneRow {
    /// A row of ids.
    Ids(Row),
    /// The pad text row and the speaker embedding.
    Speaker,
    /// The pad text row and the code rows of the reference frame of this index, summed.
    Frame(usize),
}

/// The rows of a clone prompt of `turns` in `language`.
///
/// The opening of the speech turn, each id a text row.
/// Then the control codes and the speaker embedding, each beside the pad text row.
/// Then the codec pad beside the start of text.
///
/// Without a reference, each text id then the end of text beside the codec pad, and the pad
/// text row beside the start of codes.
///
/// With a reference of `reference.0` text ids and `reference.1` frames, the reference text ids,
/// the text ids and the end of text, each beside the codec pad.
/// Then the start of codes and each reference frame, each beside the pad text row.
/// The frames drawn follow the last reference frame.
pub fn clone_rows(
    ids: &SpeechIds,
    turns: &Turns,
    language: Option<u32>,
    reference: Option<(&[u32], usize)>,
) -> Vec<CloneRow> {
    let both = |t: u32, c: u32| {
        CloneRow::Ids(Row {
            text: Some(t),
            codec: Some(c),
        })
    };
    let mut out: Vec<CloneRow> = turns
        .role
        .iter()
        .map(|&id| {
            CloneRow::Ids(Row {
                text: Some(id),
                codec: None,
            })
        })
        .collect();
    let control: Vec<u32> = match language {
        Some(lang) => vec![
            ids.codec_think,
            ids.codec_think_bos,
            lang,
            ids.codec_think_eos,
        ],
        None => vec![ids.codec_nothink, ids.codec_think_bos, ids.codec_think_eos],
    };
    out.extend(control.iter().map(|&c| both(ids.tts_pad, c)));
    out.push(CloneRow::Speaker);
    out.push(both(ids.tts_bos, ids.codec_pad));
    let (reference_text, frames): (&[u32], usize) = reference.unwrap_or((&[], 0));
    for &id in reference_text
        .iter()
        .chain(&turns.text)
        .chain(std::iter::once(&ids.tts_eos))
    {
        out.push(both(id, ids.codec_pad));
    }
    out.push(both(ids.tts_pad, ids.codec_bos));
    out.extend((0..frames).map(CloneRow::Frame));
    out
}

/// The rows of the prompt of `turns` in `language`, spoken by the named speaker whose codec id
/// is `speaker`.
///
/// The design turn, which carries the style text, each id a text row.
/// Then the rows of a clone prompt without a reference.
/// Its speaker row is the pad text row beside the speaker's codec id.
pub fn named_rows(ids: &SpeechIds, turns: &Turns, language: Option<u32>, speaker: u32) -> Vec<Row> {
    let style = turns.design.iter().map(|&id| Row {
        text: Some(id),
        codec: None,
    });
    let rows = clone_rows(ids, turns, language, None)
        .into_iter()
        .filter_map(|r| match r {
            CloneRow::Ids(row) => Some(row),
            CloneRow::Speaker => Some(Row {
                text: Some(ids.tts_pad),
                codec: Some(speaker),
            }),
            CloneRow::Frame(_) => None,
        });
    style.chain(rows).collect()
}
