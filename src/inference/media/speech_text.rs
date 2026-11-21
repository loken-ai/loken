//! Text preparation for speech synthesis.
//! This module provides lexicon parsing and text normalization for speech models.

use std::collections::HashMap;

/// A lexicon entry indicating how to handle a word.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Entry {
    /// Say this word as text (replacement text).
    Say(String),
    /// Pronounce this word using IPA transcription.
    Ipa(String),
}

/// A pronunciation lexicon mapping words to their speech representation.
#[derive(Debug, Default)]
pub struct Lexicon {
    entries: HashMap<String, Entry>,
}

impl Lexicon {
    /// Returns a lexicon built from the embedded speech_lexicon.tsv file.
    /// The builtin lexicon is parsed at compile time from the included TSV data.
    pub fn builtin() -> Lexicon {
        Lexicon::parse_ours(include_str!("speech_lexicon.tsv"))
            .unwrap_or_else(|e| panic!("Failed to parse builtin lexicon: {}", e))
    }

    /// Parses the custom format: TERM<TAB>say|ipa<TAB>TEXT.
    /// Returns a lexicon with all valid entries, erroring on malformed lines.
    /// Each line must have exactly three tab-separated fields.
    /// Line numbers are 1-indexed in error messages.
    pub fn parse_ours(text: &str) -> Result<Lexicon, String> {
        let mut entries = HashMap::new();
        for (line_num, line) in text.lines().enumerate() {
            let line_num = line_num + 1;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = trimmed.splitn(3, '\t').collect();
            if parts.len() != 3 {
                return Err(format!(
                    "Line {}: expected 3 tab-separated fields, got {}",
                    line_num,
                    parts.len()
                ));
            }
            let term = parts[0].to_string();
            let kind = parts[1];
            let value = parts[2].to_string();
            let entry = if kind == "say" {
                Entry::Say(value)
            } else if kind == "ipa" {
                Entry::Ipa(value)
            } else {
                return Err(format!(
                    "Line {}: unknown kind '{}', expected 'say' or 'ipa'",
                    line_num, kind
                ));
            };
            entries.insert(term, entry);
        }
        Ok(Lexicon { entries })
    }

    /// Parses CMUdict format: WORD  PH1 PH2 ... (two spaces after word).
    /// Comments start with ;;;, alternate pronunciations WORD(2) are skipped.
    /// Each ARPAbet phoneme is converted to IPA with stress markers.
    /// Words are stored in uppercase; lookups are case-insensitive.
    pub fn parse_cmudict(text: &str) -> Lexicon {
        let mut entries: HashMap<String, Entry> = HashMap::new();
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with(";;;") {
                continue;
            }
            if trimmed.starts_with('(') || trimmed.contains("(2)") {
                continue;
            }
            let mut parts = trimmed.splitn(2, "  ");
            let word = parts.next().unwrap_or("");
            let pron = parts.next().unwrap_or("");
            if word.is_empty() || pron.is_empty() {
                continue;
            }
            let ipa = Self::arpabet_to_ipa(pron);
            entries.insert(word.to_uppercase(), Entry::Ipa(ipa));
        }
        Lexicon { entries }
    }

    /// Converts ARPAbet phonemes to IPA with stress markers.
    /// Stress 1 adds ˈ before the vowel, stress 2 adds ˌ before the vowel.
    /// ER becomes ɚ when unstressed (stress digit 0).
    fn arpabet_to_ipa(phonemes: &str) -> String {
        let mut result = String::new();
        for token in phonemes.split_whitespace() {
            let (ipa, stress_digit) = Self::phoneme_to_ipa(token);
            if stress_digit == 1 {
                result.push('ˈ');
            } else if stress_digit == 2 {
                result.push('ˌ');
            }
            result.push_str(ipa);
        }
        result
    }

    /// Converts a single ARPAbet phoneme to IPA and returns the stress digit.
    fn phoneme_to_ipa(phoneme: &str) -> (&'static str, u8) {
        let phoneme_upper = phoneme.to_uppercase();
        let phoneme = phoneme_upper.as_str();
        let (base, stress_digit) = phoneme.split_at(
            phoneme
                .chars()
                .position(|c| c.is_ascii_digit())
                .unwrap_or(phoneme.len()),
        );
        let stress_digit = if stress_digit.is_empty() {
            0
        } else {
            stress_digit
                .chars()
                .next()
                .unwrap()
                .to_digit(10)
                .unwrap_or(0) as u8
        };
        let ipa = match base {
            "AA" => "ɑ",
            "AE" => "æ",
            "AH" if stress_digit == 0 => "ə",
            "AH" => "ʌ",
            "AO" => "ɔ",
            "AW" => "aʊ",
            "AY" => "aɪ",
            "B" => "b",
            "CH" => "tʃ",
            "D" => "d",
            "DH" => "ð",
            "EH" => "ɛ",
            "ER" if stress_digit == 0 => "ɚ",
            "ER" => "ɝ",
            "EY" => "eɪ",
            "F" => "f",
            "G" => "ɡ",
            "HH" => "h",
            "IH" => "ɪ",
            "IY" => "i",
            "JH" => "dʒ",
            "K" => "k",
            "L" => "l",
            "M" => "m",
            "N" => "n",
            "NG" => "ŋ",
            "OW" => "oʊ",
            "OY" => "ɔɪ",
            "P" => "p",
            "R" => "ɹ",
            "S" => "s",
            "SH" => "ʃ",
            "T" => "t",
            "TH" => "θ",
            "UH" => "ʊ",
            "UW" => "u",
            "V" => "v",
            "W" => "w",
            "Y" => "j",
            "Z" => "z",
            "ZH" => "ʒ",
            _ => "",
        };
        (ipa, stress_digit)
    }

    /// Merges another lexicon into this one.
    /// Entries from `other` are added only if not already in `self`.
    /// Conflicts favor `self`'s entries.
    pub fn merge(&mut self, other: Lexicon) {
        for (key, value) in other.entries {
            self.entries.entry(key).or_insert(value);
        }
    }
}

/// Prepares text for speech synthesis using a lexicon.
/// `reads_ipa` determines whether IPA pronunciations are kept as inline markers.
/// Words are normalized according to lexicon entries or spelling rules.
pub fn prepare(text: &str, lexicon: &Lexicon, reads_ipa: bool) -> String {
    let mut result = String::new();
    let mut i = 0;
    let chars: Vec<char> = text.chars().collect();

    while i < chars.len() {
        if chars[i] == '[' {
            if let Some((brace_pos, paren_end)) = find_inline_ipa(&chars[i..]) {
                // brace_pos is the position of ']' within chars[i..]
                // paren_end is the position of ')' within chars[i..]
                let marker_end = i + paren_end + 1;
                if reads_ipa {
                    result.push_str(&chars[i..marker_end].iter().collect::<String>());
                } else {
                    let word: String = chars[i + 1..i + brace_pos].iter().collect();
                    result.push_str(&word);
                }
                i = marker_end;
                continue;
            }
            result.push(chars[i]);
            i += 1;
        } else if chars[i].is_alphanumeric() {
            let word_start = i;
            while i < chars.len() && chars[i].is_alphanumeric() {
                i += 1;
            }
            let word: String = chars[word_start..i].iter().collect();
            let lookup = &word;
            if let Some(entry) = lexicon.entries.get(lookup) {
                match entry {
                    Entry::Say(t) => {
                        result.push_str(t);
                    }
                    Entry::Ipa(p) => {
                        if reads_ipa {
                            result.push_str(&format!("[{}](/{}/)", lookup, p));
                        } else {
                            result.push_str(lookup);
                        }
                    }
                }
            } else if let Some(entry) = lexicon.entries.get(&lookup.to_uppercase()) {
                if let Entry::Ipa(p) = entry {
                    if reads_ipa {
                        result.push_str(&format!("[{}](/{}/)", lookup, p));
                    } else {
                        result.push_str(lookup);
                    }
                } else {
                    result.push_str(lookup);
                }
            } else if should_spell(lookup) {
                let spelled: Vec<String> = lookup
                    .chars()
                    .map(|c| c.to_string().to_uppercase())
                    .collect();
                result.push_str(&spelled.join(" "));
            } else {
                result.push_str(lookup);
            }
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}

fn find_inline_ipa(chars: &[char]) -> Option<(usize, usize)> {
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == ']' {
            // Check if followed by ( ... /)
            if i + 4 < chars.len() && chars[i + 1] == '(' && chars[i + 2] == '/' {
                // Search for the closing /)
                let search_start = i + 3;
                for j in search_start..chars.len() {
                    if chars[j] == '/' && j + 1 < chars.len() && chars[j + 1] == ')' {
                        // Return the position of ']' and the position of ')'
                        return Some((i, j + 1));
                    }
                }
            }
        }
        i += 1;
    }
    None
}

fn should_spell(word: &str) -> bool {
    let upper_count = word.chars().filter(|c| c.is_ascii_uppercase()).count();
    let lower_count = word.chars().filter(|c| c.is_ascii_lowercase()).count();
    let len = word.len();
    if lower_count == 0 && upper_count >= 2 {
        return true;
    }
    if upper_count >= 2 && len <= 4 {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Punctuation around a word is kept once, on both sides.
    #[test]
    fn punctuation_around_a_word_is_kept_once() {
        let lexicon = Lexicon::builtin();
        assert_eq!(prepare("(GPU), [KV]!", &lexicon, false), "(G P U), [K V]!");
    }

    #[test]
    fn test_builtin_lexicon_parses() {
        let lexicon = Lexicon::builtin();
        assert!(lexicon.entries.contains_key("VRAM"));
        assert!(lexicon.entries.contains_key("loken"));
    }

    #[test]
    fn test_prepare_kv_vram() {
        let builtin = Lexicon::builtin();
        let result = prepare(
            "Fit the KV cache in VRAM, not on the host.",
            &builtin,
            false,
        );
        assert_eq!(result, "Fit the K V cache in V RAM, not on the host.");
    }

    #[test]
    fn test_gpu_moe_spelling() {
        let builtin = Lexicon::builtin();
        let result = prepare("GPU and MoE are important.", &builtin, false);
        assert!(result.contains("G P U"));
        assert!(result.contains("M O E"));
    }

    #[test]
    fn test_inline_ipa_reads_ipa() {
        let builtin = Lexicon::builtin();
        let result = prepare("[loken](/lˈoʊkən/)", &builtin, true);
        assert_eq!(result, "[loken](/lˈoʊkən/)");
    }

    #[test]
    fn test_inline_ipa_no_reads_ipa() {
        let builtin = Lexicon::builtin();
        let result = prepare("[loken](/lˈoʊkən/)", &builtin, false);
        assert_eq!(result, "loken");
    }

    #[test]
    fn test_loken_ipa_reads_ipa() {
        let builtin = Lexicon::builtin();
        let result = prepare("loken", &builtin, true);
        assert_eq!(result, "[loken](/lˈoʊkən/)");
    }

    #[test]
    fn test_cmudict_parses() {
        let text = "HELLO  HH AH0 L OW1\nHELLO(2)  HH EH0 L OW1";
        let lexicon = Lexicon::parse_cmudict(text);
        assert_eq!(lexicon.entries.len(), 1);
        let hello_ipa = lexicon.entries.get("HELLO").unwrap();
        if let Entry::Ipa(ipa) = hello_ipa {
            assert_eq!(ipa, "həlˈoʊ");
        } else {
            panic!("Expected Ipa entry for HELLO");
        }
    }

    #[test]
    fn test_builtin_wins_over_cmudict() {
        let mut builtin = Lexicon::builtin();
        let cmudict = Lexicon::parse_cmudict("VRAM  V R AE1 M");
        builtin.merge(cmudict);
        let vram_entry = builtin.entries.get("VRAM").unwrap();
        if let Entry::Say(s) = vram_entry {
            assert_eq!(s, "V RAM");
        } else {
            panic!("Expected Say entry for VRAM");
        }
    }

    #[test]
    fn test_malformed_line_error() {
        let input = "VRAM\tsay\tV RAM\nMALFORMED LINE";
        let result = Lexicon::parse_ours(input);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Line 2"));
    }
}
