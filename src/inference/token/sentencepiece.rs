//! Minimal SentencePiece **unigram** tokenizer (full-Rust, no new crate) for pocket-tts.
//!
//! Parses the binary `tokenizer.model` (a protobuf `ModelProto`) by hand - only the
//! repeated `pieces` field is needed: each piece carries a string, a log-prob score
//! and a type. Encoding is the standard unigram Viterbi best-segmentation over the
//! pieces, with whitespace mapped to the `▁` meta-symbol and byte-fallback (`<0xHH>`)
//! for out-of-vocabulary characters.

use std::collections::HashMap;

const SPACE: char = '\u{2581}'; // ▁

// SentencePiece piece types.
const TYPE_NORMAL: i32 = 1;
const TYPE_USER_DEFINED: i32 = 4;
const TYPE_BYTE: i32 = 6;

pub struct SentencePiece {
    /// piece string -> (id, score) for segmentation-eligible pieces.
    map: HashMap<String, (u32, f32)>,
    /// id of `<0x00>`; byte `b` -> `byte_base + b` (256 contiguous byte pieces).
    byte_base: Option<u32>,
    max_len: usize, // longest piece length in chars
    min_score: f32,
}

/// Read a base-128 varint at `*i`, advancing it.
fn varint(b: &[u8], i: &mut usize) -> u64 {
    let (mut shift, mut out) = (0u32, 0u64);
    loop {
        let x = b[*i];
        *i += 1;
        out |= ((x & 0x7f) as u64) << shift;
        if x & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    out
}

/// The pieces of a `tokenizer.model` in id order, and what its normaliser says of a leading
/// space.
pub struct ModelPieces {
    /// Piece text, score and type, the id being the position.
    pub pieces: Vec<(String, f32, i32)>,
    /// The normaliser's `add_dummy_prefix`, `None` when the file does not set it.
    pub add_dummy_prefix: Option<bool>,
}

/// The field `field` of a protobuf message, each occurrence's payload: the bytes of a
/// length-delimited field, the little-endian bytes of a fixed one, the varint as eight bytes.
fn fields(b: &[u8], field: u64) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < b.len() {
        let tag = varint(b, &mut i);
        let payload = match tag & 7 {
            0 => varint(b, &mut i).to_le_bytes().to_vec(),
            1 if i + 8 <= b.len() => {
                i += 8;
                b[i - 8..i].to_vec()
            }
            2 => {
                let len = varint(b, &mut i) as usize;
                if i + len > b.len() {
                    break;
                }
                i += len;
                b[i - len..i].to_vec()
            }
            5 if i + 4 <= b.len() => {
                i += 4;
                b[i - 4..i].to_vec()
            }
            _ => break,
        };
        if tag >> 3 == field {
            out.push(payload);
        }
    }
    out
}

/// Reads a serialised `ModelProto`: field 1 the repeated pieces (text 1, score 2, type 3),
/// field 3 the normaliser spec (`add_dummy_prefix` 3).
pub fn model_pieces(b: &[u8]) -> ModelPieces {
    let pieces = fields(b, 1)
        .iter()
        .map(|sub| {
            let text = fields(sub, 1)
                .pop()
                .map(|t| String::from_utf8_lossy(&t).into_owned())
                .unwrap_or_default();
            let score = fields(sub, 2)
                .pop()
                .and_then(|f| Some(f32::from_le_bytes(f.get(..4)?.try_into().ok()?)))
                .unwrap_or(0.0);
            let ptype = fields(sub, 3)
                .pop()
                .and_then(|v| Some(u64::from_le_bytes(v.try_into().ok()?) as i32))
                .unwrap_or(TYPE_NORMAL);
            (text, score, ptype)
        })
        .collect();
    let add_dummy_prefix = fields(b, 3)
        .pop()
        .and_then(|spec| fields(&spec, 3).pop())
        .and_then(|v| Some(u64::from_le_bytes(v.try_into().ok()?) != 0));
    ModelPieces {
        pieces,
        add_dummy_prefix,
    }
}

impl SentencePiece {
    pub fn from_file(path: &str) -> std::io::Result<Self> {
        let b = std::fs::read(path)?;
        let mut map: HashMap<String, (u32, f32)> = HashMap::new();
        let mut byte_base: Option<u32> = None;
        let mut max_len = 1usize;
        let mut min_score = 0f32;
        for (id, (piece, score, ptype)) in model_pieces(&b).pieces.into_iter().enumerate() {
            let id = id as u32;
            if ptype == TYPE_BYTE && byte_base.is_none() && piece == "<0x00>" {
                byte_base = Some(id);
            }
            if ptype == TYPE_NORMAL || ptype == TYPE_USER_DEFINED {
                max_len = max_len.max(piece.chars().count());
                min_score = min_score.min(score);
                map.insert(piece, (id, score));
            }
        }
        Ok(Self {
            map,
            byte_base,
            max_len,
            min_score,
        })
    }

    /// Encode text -> token ids (unigram Viterbi + byte-fallback).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Preprocess: add a leading meta-space and map spaces to `▁`.
        let pre: String = format!(" {text}").replace(' ', &SPACE.to_string());
        let chars: Vec<char> = pre.chars().collect();
        let n = chars.len();
        if n == 0 {
            return Vec::new();
        }
        // Viterbi over char positions. best[i] = (score, prev, piece_id_or_NONE_for_bytefallback).
        const NEG: f32 = f32::NEG_INFINITY;
        let mut best = vec![(NEG, 0usize, None::<u32>); n + 1];
        best[0] = (0.0, 0, None);
        let unk_penalty = self.min_score - 10.0;
        for i in 0..n {
            if best[i].0 == NEG {
                continue;
            }
            let mut matched = false;
            let maxl = self.max_len.min(n - i);
            let mut piece = String::new();
            for l in 1..=maxl {
                piece.push(chars[i + l - 1]);
                if let Some(&(id, sc)) = self.map.get(&piece) {
                    matched = true;
                    let cand = best[i].0 + sc;
                    if cand > best[i + l].0 {
                        best[i + l] = (cand, i, Some(id));
                    }
                }
            }
            // Always allow a single-char step (byte-fallback) so a path exists.
            let cand = best[i].0 + unk_penalty;
            if (cand > best[i + 1].0 || (!matched && best[i + 1].0 == NEG)) && cand > best[i + 1].0
            {
                best[i + 1] = (cand, i, None);
            }
        }
        // Backtrack.
        let mut out_rev: Vec<u32> = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let (_, prev, pid) = best[pos];
            match pid {
                Some(id) => out_rev.push(id),
                None => {
                    // byte-fallback: emit the char's UTF-8 bytes as <0xHH> ids (reversed).
                    let c = chars[prev];
                    let mut buf = [0u8; 4];
                    let bytes = c.encode_utf8(&mut buf).as_bytes();
                    if let Some(base) = self.byte_base {
                        for &byte in bytes.iter().rev() {
                            out_rev.push(base + byte as u32);
                        }
                    }
                }
            }
            pos = prev;
        }
        out_rev.reverse();
        out_rev
    }
}

#[cfg(all(test, feature = "audio"))]
mod tests {
    //! The ignored cases here need real weights, a device, or a reference dump on
    //! this machine; nothing about them is automatic. Run one by name with
    //!   cargo test --release -p loken --lib NAME -- --ignored --nocapture
    use super::*;

    fn model() -> std::path::PathBuf {
        crate::inference::model::pocket_tts::resolve_paths().1
    }

    #[test]
    #[ignore = "needs pocket-tts tokenizer.model (config huggingface_models_dir)"]
    fn encode_basic() {
        let sp = SentencePiece::from_file(model().to_str().unwrap()).unwrap();
        let ids = sp.encode("hello world");
        println!("ids = {ids:?}");
        assert!(!ids.is_empty());
        // All ids within vocab (4000).
        assert!(ids.iter().all(|&id| id < 4000));
        // Different text -> different tokenization.
        assert_ne!(sp.encode("hello world"), sp.encode("good morning everyone"));
    }
}
