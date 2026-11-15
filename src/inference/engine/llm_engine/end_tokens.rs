//! The tokens that end a generation besides the end-of-sequence id, as a checkpoint's files
//! declare them.
//!
//! Each source is a statement of the files, never the spelling of a piece:
//!
//! - the metadata keys that name the token ending a turn or a message, and the FIM tokens
//!   that end a fill-in-the-middle span;
//! - the padding token, when the vocabulary marks it as a control token;
//! - the token the chat template writes right after the content of the assistant turn that
//!   ends a conversation;
//! - the stop sequences of the model's parameters that are the whole text of one control
//!   token.

use super::*;

/// The metadata keys whose token ends a generation: the end of a turn, the end of a message,
/// and the FIM tokens that close a span.
const END_KEYS: [&str; 5] = [
    "tokenizer.ggml.eot_token_id",
    "tokenizer.ggml.eom_token_id",
    "tokenizer.ggml.fim_pad_token_id",
    "tokenizer.ggml.fim_rep_token_id",
    "tokenizer.ggml.fim_sep_token_id",
];

/// The metadata key of the padding token.
const PADDING_KEY: &str = "tokenizer.ggml.padding_token_id";

/// One token that ends a generation, with what declares it.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct DeclaredEnd {
    pub id: u32,
    pub source: &'static str,
}

/// The ids besides `eos` that end a generation, each once: `listed`, then the ends the files
/// declare.
///
/// `listed` are the end-of-sequence ids the files list beside `eos`.
/// Each declared end is logged with its source.
pub(super) fn end_ids(
    eos: u32,
    listed: Vec<u32>,
    content: Option<&gguf_file::Content>,
    tokenizer: &Tokenizer,
    template: Option<&str>,
    stops: &[String],
) -> Vec<u32> {
    let mut ids: Vec<u32> = Vec::new();
    for id in listed {
        if id != eos && !ids.contains(&id) {
            ids.push(id);
        }
    }
    for end in declared_ends(content, tokenizer, template, stops) {
        if end.id != eos && !ids.contains(&end.id) {
            info!(
                "📋 end token {} {:?}, from {}",
                end.id,
                tokenizer.id_to_token(end.id).unwrap_or_default(),
                end.source
            );
            ids.push(end.id);
        }
    }
    ids
}

/// The tokens that end a generation besides the end-of-sequence id, each once.
///
/// `content` is the header that holds the metadata keys, `None` for a checkpoint without one.
/// `template` is the chat template the checkpoint declares.
/// `stops` are the stop sequences of the model's parameters.
fn declared_ends(
    content: Option<&gguf_file::Content>,
    tokenizer: &Tokenizer,
    template: Option<&str>,
    stops: &[String],
) -> Vec<DeclaredEnd> {
    let control = tokenizer.get_added_tokens_decoder();
    let is_control = |id: u32| control.get(&id).is_some_and(|t| t.special);
    let mut ends: Vec<DeclaredEnd> = Vec::new();
    let mut add = |id: u32, source: &'static str| {
        if !ends.iter().any(|e| e.id == id) {
            ends.push(DeclaredEnd { id, source });
        }
    };
    let key_id = |key: &str| content.and_then(|c| super::gguf::get_gguf_u32(c, key));
    for key in END_KEYS {
        if let Some(id) = key_id(key) {
            add(id, key);
        }
    }
    // Padding only follows a finished sequence.
    if let Some(id) = key_id(PADDING_KEY).filter(|&id| is_control(id)) {
        add(id, PADDING_KEY);
    }
    if let Some(id) = template
        .and_then(|t| turn_closer(t, tokenizer))
        .filter(|&id| is_control(id))
    {
        add(id, "the chat template");
    }
    for stop in stops {
        if let Some(id) = tokenizer.token_to_id(stop).filter(|&id| is_control(id)) {
            add(id, "the model's stop sequences");
        }
    }
    ends
}

/// The first token the template writes after the content of a closing assistant turn,
/// whitespace aside.
fn turn_closer(template: &str, tokenizer: &Tokenizer) -> Option<u32> {
    let after = crate::api::handlers::assistant_turn_close(template)?;
    let encoding = tokenizer.encode(after.as_str(), false).ok()?;
    encoding.get_ids().iter().copied().find(|&id| {
        tokenizer
            .decode(&[id], false)
            .is_ok_and(|text| !text.trim().is_empty())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gguf_file::Value;

    /// The ordinary pieces every test vocabulary starts with.
    const TEXT: [&str; 7] = ["a", "b", "c", "ab", "Ġ", "Ċ", "x"];

    /// The id of the `n`-th piece a test vocabulary adds after `TEXT`.
    fn piece(n: usize) -> u32 {
        (TEXT.len() + n) as u32
    }

    /// A byte-level vocabulary: `TEXT`, then `pieces` with their GGUF token types, and the
    /// metadata `keys`.
    fn vocabulary(pieces: &[(&str, i32)], keys: &[(&str, u32)]) -> (gguf_file::Content, Tokenizer) {
        let all = TEXT.iter().map(|t| (*t, 1)).chain(pieces.iter().copied());
        let (tokens, types): (Vec<Value>, Vec<Value>) = all
            .map(|(t, kind)| (Value::String(t.to_string()), Value::I32(kind)))
            .unzip();
        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            "tokenizer.ggml.model".to_string(),
            Value::String("gpt2".into()),
        );
        metadata.insert("tokenizer.ggml.tokens".to_string(), Value::Array(tokens));
        metadata.insert("tokenizer.ggml.token_type".to_string(), Value::Array(types));
        let merges = vec![Value::String("a b".into())];
        metadata.insert("tokenizer.ggml.merges".to_string(), Value::Array(merges));
        for (key, id) in keys {
            metadata.insert(key.to_string(), Value::U32(*id));
        }
        let content = gguf_file::Content {
            magic: gguf_file::VersionedMagic::GgufV3,
            metadata,
            tensor_infos: std::collections::HashMap::new(),
            tensor_data_offset: 0,
            mmap_owner: None,
        };
        let tokenizer = super::super::gguf::build_tokenizer_from_gguf(&content).unwrap();
        (content, tokenizer)
    }

    fn ids(ends: &[DeclaredEnd]) -> Vec<u32> {
        ends.iter().map(|e| e.id).collect()
    }

    /// An end-of-text control token the metadata names as padding ends a generation besides
    /// the eos.
    #[test]
    fn an_end_of_text_control_token_ends_a_generation_besides_the_eos() {
        let pieces = [("<|endoftext|>", 3), ("<|im_end|>", 3)];
        let keys = [
            ("tokenizer.ggml.eos_token_id", piece(1)),
            (PADDING_KEY, piece(0)),
        ];
        let (content, tokenizer) = vocabulary(&pieces, &keys);
        let ends = declared_ends(Some(&content), &tokenizer, None, &[]);
        assert_eq!(ids(&ends), vec![piece(0)]);
    }

    /// The end-of-turn, end-of-message and FIM keys name tokens that end a generation.
    #[test]
    fn the_end_keys_name_tokens_that_end_a_generation() {
        let pieces = [("<eot>", 3), ("<eom>", 3), ("<fim_pad>", 3)];
        let keys = [
            ("tokenizer.ggml.eot_token_id", piece(0)),
            ("tokenizer.ggml.eom_token_id", piece(1)),
            ("tokenizer.ggml.fim_pad_token_id", piece(2)),
        ];
        let (content, tokenizer) = vocabulary(&pieces, &keys);
        let ends = declared_ends(Some(&content), &tokenizer, None, &[]);
        assert_eq!(ids(&ends), vec![piece(0), piece(1), piece(2)]);
    }

    /// The token the template writes after the content of the last assistant turn ends it.
    #[test]
    fn the_token_closing_the_templates_turn_ends_a_generation() {
        let pieces = [("<start_of_turn>", 3), ("<end_of_turn>", 3)];
        let (content, tokenizer) = vocabulary(&pieces, &[]);
        let template = "{% for m in messages %}<start_of_turn>{{ m.role }}\n\
                        {{ m.content }}<end_of_turn>\n{% endfor %}\
                        {% if add_generation_prompt %}<start_of_turn>model\n{% endif %}";
        let ends = declared_ends(Some(&content), &tokenizer, Some(template), &[]);
        assert_eq!(ids(&ends), vec![piece(1)]);
        assert_eq!(ends[0].source, "the chat template");
    }

    /// A template that fails after the content of the turn still names the token closing it.
    #[test]
    fn a_template_failing_after_the_turn_still_names_its_closer() {
        let (content, tokenizer) = vocabulary(&[("<end_of_turn>", 3)], &[]);
        let template = "{% for m in messages %}{{ m.content }}\\n<end_of_turn>\\n{% endfor %}\
                        {{ raise_exception('a variable the probe does not set') }}";
        let ends = declared_ends(Some(&content), &tokenizer, Some(template), &[]);
        assert_eq!(ids(&ends), vec![piece(0)]);
    }

    /// A stop sequence of the model's parameters that is one control token ends a generation.
    #[test]
    fn a_stop_sequence_that_is_a_control_token_ends_a_generation() {
        let (content, tokenizer) = vocabulary(&[("<end_of_turn>", 3)], &[]);
        let stops = ["<end_of_turn>".to_string(), "abc".to_string()];
        let ends = declared_ends(Some(&content), &tokenizer, None, &stops);
        assert_eq!(ids(&ends), vec![piece(0)]);
    }

    /// A checkpoint directory ends on the end-of-sequence ids its generation config lists and
    /// on the token its template closes a turn with.
    #[test]
    fn a_checkpoint_directory_ends_on_the_ends_it_declares() {
        let pieces = [("<eos>", 3), ("<eot>", 3), ("<end_of_turn>", 3)];
        let (_, tokenizer) = vocabulary(&pieces, &[]);
        let dir = std::env::temp_dir().join(format!("loken-end-ids-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let generation = serde_json::json!({ "eos_token_id": [piece(0), piece(1)] });
        let template = "{% for m in messages %}{{ m.content }}<end_of_turn>{% endfor %}";
        let config = serde_json::json!({ "chat_template": template });
        std::fs::write(dir.join("generation_config.json"), generation.to_string()).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), config.to_string()).unwrap();
        let ends = super::super::load::checkpoint_end_ids(&dir, piece(0), &tokenizer, &[]);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(ends, vec![piece(1), piece(2)]);
    }

    /// A stop sequence of a checkpoint without a header ends a generation.
    #[test]
    fn a_checkpoint_without_a_header_ends_on_its_stop_sequence() {
        let (_, tokenizer) = vocabulary(&[("<end_of_turn>", 3)], &[]);
        let stops = ["<end_of_turn>".to_string()];
        assert_eq!(
            end_ids(0, Vec::new(), None, &tokenizer, None, &stops),
            vec![piece(0)]
        );
    }

    /// A piece that only looks like a tag, ordinary or user-defined, ends nothing, whatever
    /// names it.
    #[test]
    fn a_token_that_only_looks_like_a_tag_ends_nothing() {
        let pieces = [("<|endoftext|>", 1), ("<end>", 4)];
        let (content, tokenizer) = vocabulary(&pieces, &[(PADDING_KEY, piece(0))]);
        let template = "{% for m in messages %}{{ m.content }}<end>{% endfor %}";
        let stops = ["<|endoftext|>".to_string(), "<end>".to_string()];
        assert!(declared_ends(Some(&content), &tokenizer, Some(template), &stops).is_empty());
    }
}
