//! Text preparation for speech synthesis.
//! This module provides lexicon parsing and text normalization for speech models.

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

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

/// Process-wide lexicon, set once at startup via `install`.
static LEXICON: OnceLock<Lexicon> = OnceLock::new();

/// Second OnceLock that returns the built-in lexicon if `LEXICON` is not set.
static LEXICON_BUILTIN: OnceLock<Lexicon> = OnceLock::new();

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
    ///
    /// This is also used by `parse_pls` to convert x-arpabet, arpabet, and cmu-arpabet
    /// phonemes to IPA when reading W3C Pronunciation Lexicon Specification files.
    pub(crate) fn arpabet_to_ipa(phonemes: &str) -> String {
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

/// Parses a W3C Pronunciation Lexicon Specification file.
///
/// The root `lexicon` carries an `alphabet` attribute; each `lexeme` holds one or more
/// `grapheme` elements and, in document order, `alias` and `phoneme` elements, a `phoneme`
/// possibly with its own `alphabet`. The first `alias` or `phoneme` of a lexeme gives every
/// grapheme of it its entry: an `alias` is `Entry::Say(text)`; a `phoneme` in the alphabet
/// `ipa` is `Entry::Ipa(text)`; a `phoneme` in `x-arpabet`, `arpabet` or `cmu-arpabet` is
/// converted with `Lexicon::arpabet_to_ipa`; a lexeme whose first pronunciation is in another
/// alphabet (such as `x-sampa`) is skipped. Tag names are matched by their local name,
/// whatever the namespace. Graphemes are stored in uppercase; lookups are case-insensitive.
pub fn parse_pls(text: &str) -> Result<Lexicon, String> {
    let doc = roxmltree::Document::parse(text).map_err(|e| format!("XML parse error: {e}"))?;

    let mut entries = HashMap::new();

    // Iterate over lexeme elements (local name "lexeme", any namespace)
    for lexeme in doc.descendants().filter(|node| node.has_tag_name("lexeme")) {
        // Collect all graphemes from this lexeme (grapheme elements under lexeme)
        let mut graphemes: Vec<String> = Vec::new();
        for grapheme_node in lexeme
            .descendants()
            .filter(|node| node.has_tag_name("grapheme"))
        {
            if let Some(text) = grapheme_node
                .text()
                .map(str::trim)
                .filter(|t| !t.is_empty())
            {
                graphemes.push(text.to_uppercase());
            }
        }

        // Skip if no graphemes
        if graphemes.is_empty() {
            continue;
        }

        // Iterate over alias and phoneme elements in document order
        let mut entry_added = false;
        for pron_node in lexeme
            .descendants()
            .filter(|node| node.has_tag_name("alias") || node.has_tag_name("phoneme"))
        {
            if entry_added {
                // Only the first pronunciation is used
                break;
            }

            // Check if this is an alias (no alphabet attribute) or phoneme
            let alphabet = pron_node.attribute("alphabet");
            let Some(text) = pron_node.text().map(str::trim).filter(|t| !t.is_empty()) else {
                continue;
            };

            let entry = if pron_node.has_tag_name("alias") {
                // Alias: use as text
                Entry::Say(text.to_string())
            } else {
                // Phoneme - check its alphabet
                match alphabet {
                    Some("ipa") => Entry::Ipa(text.to_string()),
                    Some("x-arpabet") | Some("arpabet") | Some("cmu-arpabet") => {
                        Entry::Ipa(Lexicon::arpabet_to_ipa(text))
                    }
                    Some(_) => {
                        // Other alphabet - skip this lexeme entirely
                        entry_added = true;
                        break;
                    }
                    None => {
                        // No alphabet attribute, check root alphabet
                        if let Some(root_alphabet) = doc.root_element().attribute("alphabet") {
                            match root_alphabet {
                                "ipa" => Entry::Ipa(text.to_string()),
                                "x-arpabet" | "arpabet" | "cmu-arpabet" => {
                                    Entry::Ipa(Lexicon::arpabet_to_ipa(text))
                                }
                                _ => {
                                    // Other alphabet - skip this lexeme entirely
                                    entry_added = true;
                                    break;
                                }
                            }
                        } else {
                            // No root alphabet, assume ipa
                            Entry::Ipa(text.to_string())
                        }
                    }
                }
            };

            // Add entry for all graphemes
            for grapheme in &graphemes {
                entries.insert(grapheme.clone(), entry.clone());
            }
            entry_added = true;
        }
    }

    Ok(Lexicon { entries })
}

/// Reads a lexicon file, detecting its format from the file content.
///
/// A file whose first non-whitespace character is `<` is read as a W3C PLS lexicon.
/// Otherwise, a file whose first non-comment, non-empty line contains a tab is read
/// as our format; any other file is read as CMUdict. An IO error or parse error names the path.
pub fn read_file(path: &std::path::Path) -> Result<Lexicon, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;

    let trimmed = content.trim();

    // Check if first non-whitespace character is '<' -> PLS
    if trimmed.starts_with('<') {
        parse_pls(&content)
    } else {
        // Determine format from first meaningful line
        let first_line = trimmed
            .lines()
            .find(|l| {
                !l.trim().is_empty() && !l.trim().starts_with('#') && !l.trim().starts_with(";;;")
            })
            .unwrap_or("");

        if first_line.contains('\t') {
            Lexicon::parse_ours(&content)
        } else {
            Ok(Lexicon::parse_cmudict(&content))
        }
    }
}

/// Merges lexicons from paths, starting from the built-in lexicon.
///
/// Later files win for entries they share. Returns the merged lexicon and a list of
/// warnings for files that could not be read. All files are processed regardless of
/// errors.
pub fn merged(paths: &[std::path::PathBuf]) -> (Lexicon, Vec<String>) {
    let mut lexicon = Lexicon::builtin();
    let mut warnings = Vec::new();

    for path in paths {
        match read_file(path) {
            Ok(file_lexicon) => {
                lexicon.merge(file_lexicon);
            }
            Err(e) => {
                warnings.push(e);
            }
        }
    }

    (lexicon, warnings)
}

/// Installs lexicons from paths, merging them into a process-wide lexicon.
///
/// Returns a list of warnings for files that could not be read. Files are processed
/// in order, so later files win for shared entries.
pub fn install(paths: &[std::path::PathBuf]) -> Vec<String> {
    let (lexicon, warnings) = merged(paths);
    let _ = LEXICON.set(lexicon);
    warnings
}

/// Returns the installed lexicon, or the built-in one if nothing has been installed.
///
/// This allows tests and tools that never call `install` to still get a lexicon.
pub fn lexicon() -> &'static Lexicon {
    LEXICON.get_or_init(Lexicon::builtin)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to write a temp file with given content, returns its path.
    /// The returned path is owned by a temp_dir kept in this test function's scope.
    fn write_temp_file() -> (std::path::PathBuf, tempfile::TempDir) {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test_lexicon.txt");
        std::fs::write(&path, "").unwrap();
        (path, temp_dir)
    }

    #[test]
    fn read_file_ours_format() {
        let (path, _guard) = write_temp_file();
        std::fs::write(&path, "TEST\tipa\t/tɛst/").unwrap();
        let lexicon = read_file(&path).unwrap();
        assert!(lexicon.entries.contains_key("TEST"));
        match lexicon.entries.get("TEST").unwrap() {
            Entry::Ipa(ipa) => assert_eq!(ipa, "/tɛst/"),
            _ => panic!("Expected Ipa entry"),
        }
    }

    #[test]
    fn read_file_cmudict_format() {
        let (path, _guard) = write_temp_file();
        std::fs::write(&path, "NASA  N AE1 S AH0").unwrap();
        let lexicon = read_file(&path).unwrap();
        assert!(lexicon.entries.contains_key("NASA"));
        match lexicon.entries.get("NASA").unwrap() {
            // NASA from CMUdict: N AE1 S AH0 -> nˈæsə (stress on 2nd syllable)
            Entry::Ipa(ipa) => assert_eq!(ipa, "nˈæsə"),
            _ => panic!("Expected Ipa entry"),
        }
    }

    #[test]
    fn read_file_pls_format() {
        let pls = r#"<?xml version="1.0"?>
<lexicon xmlns="http://www.w3.org/2005/01/pronunciation-lexicon" alphabet="ipa">
  <lexeme><grapheme>ABC</grapheme><grapheme>ABCDEF</grapheme><alias>see bee see</alias></lexeme>
  <lexeme><grapheme>IPATEST</grapheme><phoneme alphabet="ipa">aɪˈpeɪtɛst</phoneme></lexeme>
  <lexeme><grapheme>SKIPME</grapheme><phoneme alphabet="x-sampa">skIpmi</phoneme></lexeme>
</lexicon>"#;
        let (path, _guard) = write_temp_file();
        std::fs::write(&path, pls).unwrap();
        let lexicon = read_file(&path).unwrap();
        // ABC and ABCDEF should have alias entry
        assert!(lexicon.entries.contains_key("ABC"));
        match lexicon.entries.get("ABC").unwrap() {
            Entry::Say(s) => assert_eq!(s, "see bee see"),
            _ => panic!("Expected Say entry for ABC"),
        }
        // IPATEST should have IPA entry
        assert!(lexicon.entries.contains_key("IPATEST"));
        match lexicon.entries.get("IPATEST").unwrap() {
            Entry::Ipa(ipa) => assert_eq!(ipa, "aɪˈpeɪtɛst"),
            _ => panic!("Expected Ipa entry for IPATEST"),
        }
        // SKIPME should be skipped (x-sampa)
        assert!(!lexicon.entries.contains_key("SKIPME"));
    }

    #[test]
    fn read_file_missing() {
        let path = std::path::PathBuf::from("/nonexistent/path/file.txt");
        let result = read_file(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("/nonexistent/path/file.txt"));
    }

    /// A phoneme without its own alphabet takes the alphabet of the lexicon.
    #[test]
    fn parse_pls_reads_the_alphabet_of_the_lexicon() {
        let text = r#"<lexicon xmlns="http://www.w3.org/2005/01/pronunciation-lexicon" alphabet="x-arpabet">
  <lexeme><grapheme> NASA </grapheme><phoneme> N AE1 S AH0 </phoneme></lexeme>
</lexicon>"#;
        let lexicon = parse_pls(text).unwrap();
        let ipa = Lexicon::arpabet_to_ipa("N AE1 S AH0");
        assert_eq!(lexicon.entries.get("NASA"), Some(&Entry::Ipa(ipa)));
    }

    #[test]
    fn parse_pls_malformed_xml() {
        let result = parse_pls("<lexicon><lexeme><grapheme>TEST</lexeme>");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("XML parse error"));
    }

    #[test]
    fn install_with_warnings() {
        use std::fs;
        let temp_dir = tempfile::tempdir().unwrap();
        let good_path = temp_dir.path().join("good.txt");
        let bad_path = temp_dir.path().join("bad.txt");

        // Good file with a word not in the builtin lexicon
        std::fs::write(&good_path, "ABC\tsay\tV R A M").unwrap();

        let paths = vec![good_path, bad_path.as_path().to_path_buf()];
        let (lexicon, warnings) = merged(&paths);

        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("bad.txt"));

        // The good file's entry wins over builtin for a shared key
        assert!(lexicon.entries.contains_key("ABC"));
        match lexicon.entries.get("ABC").unwrap() {
            Entry::Say(s) => assert_eq!(s, "V R A M"),
            _ => panic!("Expected Say entry for ABC"),
        }
    }

    #[test]
    fn lexicon_without_install() {
        // If nothing was installed, lexicon() returns built-in
        let lex = lexicon();
        assert!(lex.entries.contains_key("VRAM")); // VRAM is in built-in
    }

    #[test]
    fn test_prepare() {
        let builtin = Lexicon::builtin();
        let result = prepare("GPU test", &builtin, false);
        assert!(result.contains("G P U"));
    }
}
