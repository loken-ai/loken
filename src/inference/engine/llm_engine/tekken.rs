//! Mistral's `tekken.json`: a tiktoken-style ranked byte-level BPE, read into the same
//! `tokenizers::Tokenizer` every other tokenizer here becomes.
//!
//! Layout of the file: `config` (the split `pattern`, `default_vocab_size`,
//! `default_num_special_tokens`), `vocab` (ranked entries whose `token_bytes` are base64),
//! `special_tokens` (ranked, with their `token_str`). The special tokens take the first
//! ids; a regular token's id is its rank offset by the number of special tokens, and the
//! regular vocabulary is cut where the ids reach `default_vocab_size`.

use super::*;
use base64::Engine as _;

/// The file name a checkpoint ships this tokenizer under.
pub const TEKKEN_FILE: &str = "tekken.json";

#[derive(serde::Deserialize)]
struct TekkenConfig {
    pattern: String,
    default_vocab_size: Option<usize>,
    default_num_special_tokens: Option<usize>,
}

#[derive(serde::Deserialize)]
struct TekkenToken {
    rank: usize,
    token_bytes: Option<String>,
    token_str: Option<String>,
}

#[derive(serde::Deserialize)]
struct TekkenFile {
    config: TekkenConfig,
    vocab: Vec<TekkenToken>,
    #[serde(default)]
    special_tokens: Vec<TekkenToken>,
}

/// The file's tokenizer: its special tokens in id order, and its regular tokens as
/// byte-level strings indexed by rank.
pub struct Tekken {
    pattern: String,
    specials: Vec<String>,
    regular: Vec<String>,
}

/// GPT-2's byte-to-unicode table: printable Latin-1 bytes stand for themselves, every other
/// byte takes the next code point from 256 up, in byte order.
fn byte_level_alphabet() -> [char; 256] {
    let printable = |b: u32| {
        (u32::from(b'!')..=u32::from(b'~')).contains(&b)
            || (0xA1..=0xAC).contains(&b)
            || (0xAE..=0xFF).contains(&b)
    };
    let mut table = ['\0'; 256];
    let mut next = 256u32;
    for (b, slot) in table.iter_mut().enumerate() {
        let b = b as u32;
        let cp = if printable(b) {
            b
        } else {
            next += 1;
            next - 1
        };
        *slot = char::from_u32(cp).unwrap_or('\u{FFFD}');
    }
    table
}

impl Tekken {
    pub fn open(path: &std::path::Path) -> AnyResult<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow!("tekken tokenizer {}: {e}", path.display()))?;
        Self::parse(&text).map_err(|e| anyhow!("tekken tokenizer {}: {e}", path.display()))
    }

    pub fn parse(text: &str) -> AnyResult<Self> {
        let file: TekkenFile = serde_json::from_str(text)?;
        let n_special = file
            .config
            .default_num_special_tokens
            .unwrap_or(file.special_tokens.len());
        if file.special_tokens.len() > n_special {
            return Err(anyhow!(
                "{} special tokens for {n_special} special ids",
                file.special_tokens.len()
            ));
        }
        let mut specials: Vec<Option<String>> = vec![None; n_special];
        for t in &file.special_tokens {
            let s = t
                .token_str
                .clone()
                .ok_or_else(|| anyhow!("special token of rank {} has no token_str", t.rank))?;
            *specials
                .get_mut(t.rank)
                .ok_or_else(|| anyhow!("special token rank {} out of range", t.rank))? = Some(s);
        }
        // Ids the file leaves unnamed are reserved slots, named by their id.
        let specials: Vec<String> = specials
            .into_iter()
            .enumerate()
            .map(|(i, s)| s.unwrap_or_else(|| format!("<SPECIAL_{i}>")))
            .collect();

        let n_regular = file
            .config
            .default_vocab_size
            .map_or(file.vocab.len(), |v| v.saturating_sub(n_special))
            .min(file.vocab.len());
        let alphabet = byte_level_alphabet();
        let mut regular: Vec<Option<String>> = vec![None; n_regular];
        for t in file.vocab.iter().filter(|t| t.rank < n_regular) {
            let b64 = t
                .token_bytes
                .as_deref()
                .ok_or_else(|| anyhow!("token of rank {} has no token_bytes", t.rank))?;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(b64)
                .map_err(|e| anyhow!("token of rank {}: {e}", t.rank))?;
            regular[t.rank] = Some(bytes.iter().map(|&b| alphabet[b as usize]).collect());
        }
        let regular = regular
            .into_iter()
            .enumerate()
            .map(|(r, s)| s.ok_or_else(|| anyhow!("no token of rank {r}")))
            .collect::<AnyResult<Vec<String>>>()?;
        Ok(Self {
            pattern: file.config.pattern,
            specials,
            regular,
        })
    }

    /// The id of the special token written `text`.
    pub fn special_id(&self, text: &str) -> Option<u32> {
        self.specials
            .iter()
            .position(|s| s == text)
            .map(|i| i as u32)
    }

    /// The merge table the ranks imply. Every split of a regular token into two regular
    /// tokens is a rule, the rules ordered by the rank of the token they form and, among
    /// the splits of one token, by the ranks of its halves. Merging the lowest-ranked rule
    /// first is the ranked BPE the file describes.
    fn merges(&self, rank_of: &ahash::AHashMap<&str, usize>) -> Vec<(String, String)> {
        let mut merges = Vec::new();
        for piece in &self.regular {
            let start = merges.len();
            for (at, _) in piece.char_indices().skip(1) {
                let (l, r) = piece.split_at(at);
                if let (Some(&rl), Some(&rr)) = (rank_of.get(l), rank_of.get(r)) {
                    merges.push((rl, rr, l, r));
                }
            }
            merges[start..].sort_by_key(|(rl, rr, _, _)| (*rl, *rr));
        }
        merges
            .into_iter()
            .map(|(_, _, l, r)| (l.to_string(), r.to_string()))
            .collect()
    }

    /// The tokenizer: byte-level BPE over the file's split pattern, the special tokens
    /// matched whole, a BOS leading every encoded sequence when the file names one.
    pub fn build(&self) -> AnyResult<Tokenizer> {
        use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
        use tokenizers::models::bpe::BPE;

        let n_special = self.specials.len();
        let rank_of: ahash::AHashMap<&str, usize> = self
            .regular
            .iter()
            .enumerate()
            .map(|(r, s)| (s.as_str(), r))
            .collect();
        let merges = self.merges(&rank_of);
        let mut vocab: ahash::AHashMap<String, u32> = self
            .specials
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();
        for (r, s) in self.regular.iter().enumerate() {
            if vocab.insert(s.clone(), (n_special + r) as u32).is_some() {
                return Err(anyhow!(
                    "regular token {s:?} shares its text with a special token"
                ));
            }
        }
        // A pre-token the vocabulary holds whole is one token, as ranked BPE finds it.
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .ignore_merges(true)
            .build()
            .map_err(|e| anyhow!("BPE build failed: {e}"))?;
        let mut tokenizer = Tokenizer::new(bpe);
        tokenizer.with_pre_tokenizer(Some(super::gguf::byte_level_split_sequence(&[self
            .pattern
            .as_str()])?));
        tokenizer.with_decoder(Some(ByteLevelDecoder::new(
            /*add_prefix_space=*/ false, /*trim_offsets=*/ false,
            /*use_regex=*/ false,
        )));
        let added: Vec<tokenizers::AddedToken> = self
            .specials
            .iter()
            .map(|s| tokenizers::AddedToken::from(s.clone(), !super::gguf::parsed_by_the_api(s)))
            .collect();
        tokenizer.add_special_tokens(&added);

        if let Some(bos) = self.bos() {
            use tokenizers::processors::template::TemplateProcessing;
            let text = &self.specials[bos as usize];
            let post = TemplateProcessing::builder()
                .try_single(format!("{text}:0 $A:0"))
                .map_err(|e| anyhow!("{e}"))?
                .try_pair(format!("{text}:0 $A:0 $B:1"))
                .map_err(|e| anyhow!("{e}"))?
                .special_tokens(vec![(text.clone(), bos)])
                .build()
                .map_err(|e| anyhow!("{e}"))?;
            tokenizer.with_post_processor(Some(post));
        }
        Ok(tokenizer)
    }

    /// The beginning-of-sequence id: the special token the format names `<s>`.
    pub fn bos(&self) -> Option<u32> {
        self.special_id("<s>")
    }

    /// The end-of-sequence id: the special token the format names `</s>`.
    pub fn eos(&self) -> Option<u32> {
        self.special_id("</s>")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The alphabet is a bijection onto 256 distinct characters, the printable bytes kept.
    #[test]
    fn byte_level_alphabet_is_gpt2s() {
        let t = byte_level_alphabet();
        let distinct: std::collections::HashSet<char> = t.iter().copied().collect();
        assert_eq!(distinct.len(), 256);
        assert_eq!(t[b'a' as usize], 'a');
        assert_eq!(t[b' ' as usize], 'Ġ');
        assert_eq!(t[b'\n' as usize], 'Ċ');
        assert_eq!(t[0], 'Ā');
    }

    /// A small file: special ids first, regular ids offset past them, merges applied by
    /// rank, the vocabulary cut at `default_vocab_size`, BOS leading an encoded sequence.
    #[test]
    fn a_small_file_encodes_by_rank() {
        let b = |s: &[u8]| base64::engine::general_purpose::STANDARD.encode(s);
        let mut vocab = Vec::new();
        for (r, t) in [&b"a"[..], b"b", b" ", b"ab", b" ab", b"zz"]
            .iter()
            .enumerate()
        {
            vocab.push(serde_json::json!({"rank": r, "token_bytes": b(t)}));
        }
        let file = serde_json::json!({
            "config": {"pattern": r" ?\p{L}+|\s+", "default_vocab_size": 8,
                       "default_num_special_tokens": 3},
            "vocab": vocab,
            "special_tokens": [
                {"rank": 0, "token_str": "<unk>"},
                {"rank": 1, "token_str": "<s>"},
                {"rank": 2, "token_str": "</s>"}
            ]
        });
        let tk = Tekken::parse(&file.to_string()).expect("parse");
        assert_eq!(tk.regular.len(), 5, "vocab cut at default_vocab_size");
        assert_eq!((tk.bos(), tk.eos()), (Some(1), Some(2)));
        let tok = tk.build().expect("build");
        let enc = tok.encode("ab abba", true).expect("encode");
        // <s>, "ab", " ab", "b", "a"
        assert_eq!(enc.get_ids(), &[1, 6, 7, 4, 3]);
        let dec = tok.decode(enc.get_ids(), true).expect("decode");
        assert_eq!(dec, "ab abba");
    }

    /// The weights blob an Ollama manifest names, header only.
    fn ollama_gguf_header(manifest: &str) -> Option<gguf_file::Content> {
        let store = crate::config::Config::default_ollama_models_dir();
        let man = store
            .join("manifests/registry.ollama.ai/library")
            .join(manifest);
        let mf: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(man).ok()?).ok()?;
        let digest = mf
            .get("layers")?
            .as_array()?
            .iter()
            .find(|l| {
                l.get("mediaType")
                    .and_then(|m| m.as_str())
                    .is_some_and(|m| m.ends_with(".model"))
            })?
            .get("digest")?
            .as_str()?
            .replace(':', "-");
        gguf_file::open_header(store.join("blobs").join(digest)).ok()
    }

    /// The first snapshot of an HF hub repository that ships a `tekken.json`.
    fn hub_tekken(repo_dir: &str) -> Option<std::path::PathBuf> {
        let root = crate::config::Config::default_hf_models_dir();
        let mut snaps: Vec<std::path::PathBuf> =
            std::fs::read_dir(root.join(repo_dir).join("snapshots"))
                .ok()?
                .flatten()
                .map(|e| e.path().join(TEKKEN_FILE))
                .filter(|p| p.exists())
                .collect();
        snaps.sort();
        snaps.into_iter().next()
    }

    /// The tokenizer read from a checkpoint's `tekken.json` matches, id for id, the one built
    /// from the GGUF of the same model family: same vocabulary, same encodings over code,
    /// accented text, emoji, whitespace runs, digits and control tokens, and each encoding
    /// decodes back to its text, after the round trip through the embedded JSON. Skipped
    /// when either file is not on this machine.
    #[test]
    fn tekken_json_matches_the_gguf_tokenizer() {
        let (Some(gguf), Some(tekken)) = (
            ollama_gguf_header("devstral/24b"),
            hub_tekken("models--cyankiwi--Devstral-Small-2507-AWQ-4bit"),
        ) else {
            eprintln!("skipped: the GGUF or the tekken.json is not on this machine");
            return;
        };
        let reference = super::super::build_tokenizer_from_gguf(&gguf).expect("GGUF tokenizer");
        let tk = Tekken::open(&tekken).expect("tekken.json");
        // Encoded through the JSON a checkpoint's header carries, as the loader reads it.
        let json = tk
            .build()
            .expect("tekken tokenizer")
            .to_string(false)
            .expect("serialise");
        let tok = Tokenizer::from_bytes(json.as_bytes()).expect("reload");

        // The GGUF is an earlier revision of the family: it may leave a special slot the
        // tekken.json names as its reserved placeholder. Every other id holds the same text.
        assert_eq!(tok.get_vocab_size(true), reference.get_vocab_size(true));
        for id in 0..tok.get_vocab_size(true) as u32 {
            let (ours, theirs) = (tok.id_to_token(id), reference.id_to_token(id));
            let reserved = format!("<SPECIAL_{id}>");
            let named_later =
                (id as usize) < tk.specials.len() && theirs.as_deref() == Some(reserved.as_str());
            assert!(
                ours == theirs || named_later,
                "id {id}: {ours:?} vs {theirs:?}"
            );
        }
        assert_eq!(
            tk.bos(),
            super::super::get_gguf_u32(&gguf, "tokenizer.ggml.bos_token_id")
        );
        assert_eq!(
            tk.eos(),
            super::super::get_gguf_u32(&gguf, "tokenizer.ggml.eos_token_id")
        );

        let texts = [
            "fn main() {\n    let v: Vec<u32> = (0..10).map(|x| x * 2).collect();\n    println!(\"{v:?}\");\n}\n",
            "def f(a, b=None):\n\treturn {'k': a // 3, \"s\": f\"{b!r}\"}  # comment\n",
            "L'été dernier, à Noël, j'ai mangé une crème brûlée près du château; où ça ? Ça va !",
            "ÉCOLE Ëlève ŒUVRE naïve cœur façade déjà-vu",
            "Emoji: 😀🎉👩‍💻 and flags 🇫🇷🇯🇵, CJK 日本語のテキスト, Ελληνικά, русский текст.",
            "spaces    between\t\ttabs\n\n\n   leading and trailing   \r\n",
            "Numbers 1234567890, 3.14159, -42, 1e-9, 0x1F, 2026-09-29 12:34:56",
            "URL https://example.com/a/b?c=d&e=f and path /usr/local/bin//x",
            "CamelCaseIdentifier snake_case_name SCREAMING_CASE mixedUPPERlower",
            "[INST]Write a haiku.[/INST]Sure.</s>[TOOL_CALLS]",
            "",
            " ",
        ];
        for text in texts {
            let want = reference.encode(text, true).expect("GGUF encode");
            let got = tok.encode(text, true).expect("tekken encode");
            assert_eq!(got.get_ids(), want.get_ids(), "ids differ for {text:?}");
            let ids = got.get_ids();
            assert_eq!(ids.first().copied(), tk.bos(), "no BOS for {text:?}");
            let back = tok.decode(&ids[1..], false).expect("decode");
            assert_eq!(back, text, "round trip differs for {text:?}");
        }
    }
}
