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

impl SentencePiece {
    pub fn from_file(path: &str) -> std::io::Result<Self> {
        let b = std::fs::read(path)?;
        let mut map: HashMap<String, (u32, f32)> = HashMap::new();
        let mut byte_base: Option<u32> = None;
        let mut max_len = 1usize;
        let mut min_score = 0f32;
        let mut id = 0u32;
        let mut i = 0usize;
        // Top-level ModelProto: field 1 (`pieces`) is a repeated length-delimited message.
        while i < b.len() {
            let tag = varint(&b, &mut i);
            let (field, wire) = ((tag >> 3) as u32, (tag & 7) as u32);
            match wire {
                2 => {
                    let len = varint(&b, &mut i) as usize;
                    let sub = &b[i..i + len];
                    i += len;
                    if field == 1 {
                        // SentencePiece sub-message: f1 piece (string), f2 score (float), f3 type.
                        let (mut j, mut piece, mut score, mut ptype) =
                            (0usize, String::new(), 0f32, TYPE_NORMAL);
                        while j < sub.len() {
                            let t2 = varint(sub, &mut j);
                            match (t2 >> 3, t2 & 7) {
                                (1, 2) => {
                                    let l = varint(sub, &mut j) as usize;
                                    piece = String::from_utf8_lossy(&sub[j..j + l]).into_owned();
                                    j += l;
                                }
                                (2, 5) => {
                                    score = f32::from_le_bytes(sub[j..j + 4].try_into().unwrap());
                                    j += 4;
                                }
                                (3, 0) => ptype = varint(sub, &mut j) as i32,
                                (_, 0) => {
                                    varint(sub, &mut j);
                                }
                                (_, 2) => {
                                    let l = varint(sub, &mut j) as usize;
                                    j += l;
                                }
                                (_, 5) => j += 4,
                                (_, 1) => j += 8,
                                _ => break,
                            }
                        }
                        if ptype == TYPE_BYTE && byte_base.is_none() && piece == "<0x00>" {
                            byte_base = Some(id);
                        }
                        if ptype == TYPE_NORMAL || ptype == TYPE_USER_DEFINED {
                            max_len = max_len.max(piece.chars().count());
                            min_score = min_score.min(score);
                            map.insert(piece, (id, score));
                        }
                        id += 1;
                    }
                }
                0 => {
                    varint(&b, &mut i);
                }
                5 => i += 4,
                1 => i += 8,
                _ => break,
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
