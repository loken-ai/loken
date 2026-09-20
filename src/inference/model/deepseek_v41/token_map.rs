//! The compressed token map the engram hash runs over, built from the tokenizer as the reference
//! builds it: every token id maps onto a smaller id space where tokens that normalise alike -
//! NFKC, NFD, accents stripped, lowercased, whitespace runs collapsed - collapse together, so
//! " The", "the" and "THE" hash the same way. A token that is a partial UTF-8 byte sequence has
//! nothing to normalise and is keyed by its raw piece. Ids are assigned in token order, first
//! occurrence first, which is what the released hash parameters were derived from.

use crate::tensor::{Error, Result};
use std::collections::HashMap;
use tokenizers::normalizers::{Lowercase, Replace, Strip, StripAccents, NFD, NFKC};
use tokenizers::{NormalizedString, Normalizer, Tokenizer};

/// The compressed id of every token, and the size of the compressed vocabulary.
pub fn build(tokenizer: &Tokenizer) -> Result<(Vec<u32>, usize)> {
    // A private-use character, so a token that is exactly one space survives the strip instead
    // of collapsing to the empty string and merging with unrelated tokens.
    let sentinel = "\u{e000}";
    let bad = |e: tokenizers::Error| Error::msg(format!("token map normaliser: {e}"));
    let collapse = Replace::new(
        tokenizers::normalizers::replace::ReplacePattern::Regex("[ \\t\\r\\n]+".into()),
        " ",
    )
    .map_err(bad)?;
    let lone_space = Replace::new(
        tokenizers::normalizers::replace::ReplacePattern::Regex("^ $".into()),
        sentinel,
    )
    .map_err(bad)?;
    let restore = Replace::new(
        tokenizers::normalizers::replace::ReplacePattern::String(sentinel.into()),
        " ",
    )
    .map_err(bad)?;
    let strip = Strip::new(true, true);
    let normalize = |text: &str| -> Result<String> {
        let mut s = NormalizedString::from(text);
        NFKC.normalize(&mut s).map_err(bad)?;
        NFD.normalize(&mut s).map_err(bad)?;
        StripAccents.normalize(&mut s).map_err(bad)?;
        Lowercase.normalize(&mut s).map_err(bad)?;
        collapse.normalize(&mut s).map_err(bad)?;
        lone_space.normalize(&mut s).map_err(bad)?;
        strip.normalize(&mut s).map_err(bad)?;
        restore.normalize(&mut s).map_err(bad)?;
        Ok(s.get().to_string())
    };

    let vocab = tokenizer.get_vocab_size(true);
    let mut key_to_new: HashMap<String, u32> = HashMap::new();
    let mut lookup = Vec::with_capacity(vocab);
    for id in 0..vocab as u32 {
        let text = tokenizer
            .decode(&[id], false)
            .map_err(|e| Error::msg(format!("token map decode {id}: {e}")))?;
        let key = if text.contains('\u{fffd}') {
            tokenizer
                .id_to_token(id)
                .ok_or_else(|| Error::msg(format!("token map: no piece for id {id}")))?
        } else {
            let n = normalize(&text)?;
            if n.is_empty() {
                text
            } else {
                n
            }
        };
        let next = key_to_new.len() as u32;
        let new_id = *key_to_new.entry(key).or_insert(next);
        lookup.push(new_id);
    }
    Ok((lookup, key_to_new.len()))
}

/// `build` over the tokenizer at `path` (a `tokenizer.json`).
pub fn build_from_file(path: &std::path::Path) -> Result<(Vec<u32>, usize)> {
    let tok =
        Tokenizer::from_file(path).map_err(|e| Error::msg(format!("{}: {e}", path.display())))?;
    build(&tok)
}
