//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Helper: get string value from GGUF metadata
pub(super) fn get_gguf_string(ct: &gguf_file::Content, key: &str) -> Option<String> {
    match ct.metadata.get(key)? {
        gguf_file::Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

/// Helper: get u32 value from GGUF metadata
pub(super) fn get_gguf_u32(ct: &gguf_file::Content, key: &str) -> Option<u32> {
    match ct.metadata.get(key)? {
        gguf_file::Value::U32(v) => Some(*v),
        gguf_file::Value::U64(v) => Some(*v as u32),
        gguf_file::Value::I32(v) => Some(*v as u32),
        // Some keys are stored per-layer as arrays (e.g., gemma4:26b's
        // attention.head_count_kv = [8,8,8,8,8,2,...]). Take the MAX so
        // KV-cache reservation accommodates the worst-case layer.
        gguf_file::Value::Array(arr) => arr
            .iter()
            .filter_map(|v| match v {
                gguf_file::Value::U32(x) => Some(*x),
                gguf_file::Value::U64(x) => Some(*x as u32),
                gguf_file::Value::I32(x) => Some(*x as u32),
                _ => None,
            })
            .max(),
        _ => None,
    }
}

/// Helper: count tokens from GGUF metadata
pub(super) fn count_gguf_tokens(ct: &gguf_file::Content) -> Option<u32> {
    match ct.metadata.get("tokenizer.ggml.tokens")? {
        gguf_file::Value::Array(arr) => Some(arr.len() as u32),
        _ => None,
    }
}

/// Build a tokenizers::Tokenizer from GGUF embedded vocab metadata.
/// Ollama models never ship tokenizer.json - the full vocab and merges
/// are always embedded in the GGUF file under tokenizer.ggml.* keys.
pub fn build_tokenizer_from_gguf(content: &gguf_file::Content) -> AnyResult<Tokenizer> {
    use ahash::AHashMap;
    use tokenizers::models::bpe::BPE;

    let tokenizer_model =
        get_gguf_string(content, "tokenizer.ggml.model").unwrap_or_else(|| "llama".to_string());
    debug!(
        "🔤 GGUF tokenizer model type: '{}' (detected from tokenizer.ggml.model)",
        tokenizer_model
    );

    // Debug: log some tokenizer.ggml.* keys
    if let Some(gguf_file::Value::String(s)) = content.metadata.get("tokenizer.ggml.pre") {
        debug!("   tokenizer.ggml.pre = '{}'", s);
    }

    // Build a quick lookup of GGUF token types so we can keep special tokens
    // (types 2-5) out of the BPE vocab. Mirrors HF's tokenizer.json layout
    // where specials live only in `added_tokens`. Without this filter the
    // PAD-padded GGUF vocab (qwen3-coder reserves 267 [PADxxx] slots) breaks
    // llguidance's byte-trie compilation: derivre asserts when a BPE-vocab
    // entry's bytes don't fit the byte-level encoding rules.
    let token_types: Vec<i32> = match content.metadata.get("tokenizer.ggml.token_type") {
        Some(gguf_file::Value::Array(arr)) => arr
            .iter()
            .map(|v| {
                if let gguf_file::Value::I32(t) = v {
                    *t
                } else {
                    1
                }
            })
            .collect(),
        _ => Vec::new(),
    };
    // GGUF token types per llama.cpp: 1=NORMAL, 2=UNKNOWN, 3=CONTROL,
    // 4=USER_DEFINED, 5=UNUSED, 6=BYTE. We previously filtered CONTROL
    // and UNUSED from the BPE vocab, but lfm2 family models tag named
    // entities (~509 of them: "Mathias", etc.) as type 3 CONTROL while
    // BPE merges still reference them - filtering breaks BPE.build()
    // with "out of vocabulary" errors. Conservative fix: only drop
    // type 5 (UNUSED placeholders like [PAD151669]) which never appear
    // in merges. CONTROL/USER_DEFINED/UNKNOWN stay in vocab AND get
    // additionally registered as added_tokens below for skip-special
    // tokens support.
    let is_special = |idx: usize| -> bool { token_types.get(idx).is_some_and(|t| *t == 5) };

    // Extract vocab: token_string -> token_id, skipping special-typed slots.
    let mut vocab_skipped = 0usize;
    let vocab: AHashMap<String, u32> = match content.metadata.get("tokenizer.ggml.tokens") {
        Some(gguf_file::Value::Array(arr)) => arr
            .iter()
            .enumerate()
            .filter_map(|(i, v)| {
                if is_special(i) {
                    vocab_skipped += 1;
                    return None;
                }
                if let gguf_file::Value::String(s) = v {
                    Some((s.clone(), i as u32))
                } else {
                    None
                }
            })
            .collect(),
        _ => return Err(anyhow!("GGUF metadata has no tokenizer.ggml.tokens")),
    };
    debug!("📚 Extracted {} BPE tokens from GGUF metadata ({} special-typed slots routed to added_tokens)",
        vocab.len(), vocab_skipped);

    // Extract BPE merge rules (space-separated pairs like "Ġ t")
    let merges: Vec<(String, String)> = match content.metadata.get("tokenizer.ggml.merges") {
        Some(gguf_file::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| {
                if let gguf_file::Value::String(s) = v {
                    let mut parts = s.splitn(2, ' ');
                    match (parts.next(), parts.next()) {
                        (Some(a), Some(b)) => Some((a.to_string(), b.to_string())),
                        _ => None,
                    }
                } else {
                    None
                }
            })
            .collect(),
        _ => vec![],
    };
    debug!("🔗 Extracted {} BPE merge rules", merges.len());

    // Pre-tokenizer family from `tokenizer.ggml.pre` (authority: llama.cpp
    // `llama-vocab.cpp`, the LLAMA_VOCAB_PRE_TYPE_* table). Byte-level BPE
    // vocabularies share the `gpt2` model tag but split text with DIFFERENT
    // regexes before merging, and the llama3 family additionally takes the
    // greedy longest-vocab match (`ignore_merges`) instead of rank-ordered
    // merges. Using the wrong split silently mis-segments every prompt - e.g.
    // GPT-2's ` ?\p{L}+` splits "large-scale" into ["large","-","scale"], while
    // llama3's `[^\r\n\p{L}\p{N}]?\p{L}+` binds the hyphen: ["large","-scale"].
    // `regex_kind` selects the split regex; `ignore_merges` is a SEPARATE axis
    // (only the llama3 pre_type sets it - dbrx/smaug/glm4 reuse the llama3 regex
    // but keep rank-ordered merges). Unmapped `pre` keeps the existing ByteLevel
    // GPT-2 path (no behavior change -> no regression on models that match today).
    #[derive(Clone, Copy, PartialEq)]
    enum BpePreFamily {
        Llama3,
        Qwen2,
        Qwen35,
        Tekken,
        Gpt2Default,
    }
    let pre_id = get_gguf_string(content, "tokenizer.ggml.pre").unwrap_or_default();
    let pre_family = match pre_id.as_str() {
        // LLAMA_VOCAB_PRE_TYPE_LLAMA3 + DBRX/SMAUG/CHATGLM4 (share the regex)
        "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "falcon-h1" | "pixtral" | "midm-2.0"
        | "lfm2" | "jina-v5-nano" | "dbrx" | "smaug-bpe" | "glm4" | "chatglm-bpe" => {
            BpePreFamily::Llama3
        }
        // LLAMA_VOCAB_PRE_TYPE_QWEN2 (single-digit \p{N})
        "qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" | "stablelm2" | "hunyuan"
        | "solar-open" => BpePreFamily::Qwen2,
        // LLAMA_VOCAB_PRE_TYPE_QWEN35 (adds \p{M} to letter/symbol classes)
        "qwen35" => BpePreFamily::Qwen35,
        // LLAMA_VOCAB_PRE_TYPE_TEKKEN (Mistral tekken: case-split letter runs,
        // no contraction group, `/` folded into the punctuation run)
        "tekken" => BpePreFamily::Tekken,
        _ => BpePreFamily::Gpt2Default,
    };
    // Greedy longest-vocab match: the llama3 pre_type list + tekken.
    let ignore_merges = matches!(
        pre_id.as_str(),
        "llama3"
            | "llama-v3"
            | "llama-bpe"
            | "falcon3"
            | "falcon-h1"
            | "pixtral"
            | "midm-2.0"
            | "lfm2"
            | "jina-v5-nano"
            | "tekken"
    );

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .ignore_merges(ignore_merges)
        .build()
        .map_err(|e| anyhow!("BPE build failed: {e}"))?;

    let mut tokenizer = Tokenizer::new(bpe);

    // Apply pre-tokenizer and decoder pipeline to properly handle special tokens
    // This is critical for correct text output (Ġ->space, Ċ->newline)
    match tokenizer_model.as_str() {
        "gpt2" => {
            // Byte-level BPE. The byte->unicode mapping + byte-level decoder are
            // shared; the SEGMENTATION regex is per-family (see BpePreFamily).
            // Set add_prefix_space=false / use_regex=false on the decoder to
            // match Qwen/DeepSeek tokenizer.json conventions. The default
            // (add_prefix_space=true) silently rewrites token byte sequences
            // and breaks llguidance's byte-trie compilation (derivre aborts).
            use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
            use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPre;
            use tokenizers::pre_tokenizers::sequence::Sequence as PreSequence;
            use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
            use tokenizers::pre_tokenizers::PreTokenizerWrapper;
            use tokenizers::SplitDelimiterBehavior;

            // Per-`pre` split regexes, copied verbatim from llama.cpp
            // `llama-vocab.cpp` (the "adapted" explicit-case forms).
            const LLAMA3_RE: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
            const QWEN2_RE: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
            const QWEN35_RE: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
            // TEKKEN: the original tokenizer.json regex (unicode letter classes),
            // which HF's regex engine supports directly - cleaner than llama.cpp's
            // ASCII-lookahead reformulation of the same intent.
            const TEKKEN_RE: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

            let dec = ByteLevelDecoder::new(
                /*add_prefix_space=*/ false, /*trim_offsets=*/ false,
                /*use_regex=*/ false,
            );

            // Build a Split(regex, Isolated) -> ByteLevel(no built-in regex)
            // pipeline: the regex owns segmentation, ByteLevel only does the
            // byte->unicode (Ġ) mapping. Mirrors the model's HF tokenizer.json.
            let split_pipeline = |re: &str| -> AnyResult<PreTokenizerWrapper> {
                let split = Split::new(
                    SplitPattern::Regex(re.to_string()),
                    SplitDelimiterBehavior::Isolated,
                    /*invert=*/ false,
                )
                .map_err(|e| anyhow!("build split pre-tokenizer: {e}"))?;
                let bl = ByteLevelPre::new(
                    /*add_prefix_space=*/ false, /*trim_offsets=*/ true,
                    /*use_regex=*/ false,
                );
                Ok(PreTokenizerWrapper::Sequence(PreSequence::new(vec![
                    PreTokenizerWrapper::Split(split),
                    PreTokenizerWrapper::ByteLevel(bl),
                ])))
            };

            match pre_family {
                BpePreFamily::Llama3 => {
                    tokenizer.with_pre_tokenizer(Some(split_pipeline(LLAMA3_RE)?));
                    debug!(
                        "🔡 Configured LLAMA3 pre-tokenizer (pre='{pre_id}', ignore_merges=true)"
                    );
                }
                BpePreFamily::Qwen2 => {
                    tokenizer.with_pre_tokenizer(Some(split_pipeline(QWEN2_RE)?));
                    debug!("🔡 Configured QWEN2 pre-tokenizer (pre='{pre_id}')");
                }
                BpePreFamily::Qwen35 => {
                    tokenizer.with_pre_tokenizer(Some(split_pipeline(QWEN35_RE)?));
                    debug!("🔡 Configured QWEN35 pre-tokenizer (pre='{pre_id}')");
                }
                BpePreFamily::Tekken => {
                    tokenizer.with_pre_tokenizer(Some(split_pipeline(TEKKEN_RE)?));
                    debug!(
                        "🔡 Configured TEKKEN pre-tokenizer (pre='{pre_id}', ignore_merges=true)"
                    );
                }
                BpePreFamily::Gpt2Default => {
                    // Unchanged: HF ByteLevel with its built-in GPT-2 regex.
                    let pre = ByteLevelPre::new(
                        /*add_prefix_space=*/ false, /*trim_offsets=*/ true,
                        /*use_regex=*/ true,
                    );
                    tokenizer.with_pre_tokenizer(Some(pre));
                    debug!("🔡 Configured ByteLevel pre-tokenizer (gpt2/default, pre='{pre_id}')");
                }
            }
            tokenizer.with_decoder(Some(dec));
        }
        // `_` already absorbs "llama"; listing it explicitly was clippy::redundant_wildcard.
        _ => {
            // Llama/Mistral style: use metaspace pre-tokenizer which handles Ġ as space
            use tokenizers::decoders::metaspace::Metaspace as MetaspaceDec;
            use tokenizers::pre_tokenizers::metaspace::Metaspace as MetaspacePre;

            use tokenizers::decoders::byte_fallback::ByteFallback;
            use tokenizers::decoders::fuse::Fuse;
            use tokenizers::decoders::sequence::Sequence as DecSequence;
            use tokenizers::DecoderWrapper;

            // Ġ (U+0120) represents a leading space in this tokenizer
            let pre = MetaspacePre::default();
            // SPM vocabularies carry `<0xNN>` byte-fallback tokens for anything
            // outside the learned pieces (newlines, emoji, rare UTF-8). The HF
            // reference decoder for this family is a SEQUENCE ending in
            // ByteFallback + Fuse; with Metaspace alone those tokens leaked
            // into the output as literal "<0x0A>" strings.
            let dec = DecSequence::new(vec![
                DecoderWrapper::Metaspace(MetaspaceDec::default()),
                DecoderWrapper::ByteFallback(ByteFallback::new()),
                DecoderWrapper::Fuse(Fuse::new()),
            ]);
            tokenizer.with_pre_tokenizer(Some(pre));
            tokenizer.with_decoder(Some(dec));
            debug!("🔡 Configured Metaspace + ByteFallback + Fuse decoder (llama mode)");
        }
    }

    // Mark special tokens (for skip_special_tokens support).
    // Types: 1 = normal, 2 = unknown, 3 = control, 4 = user-defined, 5 = unused
    // (placeholder slots like [PAD151669]), 6 = byte token. We register only
    // 2-4 as added_tokens - matches HF's tokenizer.json layout. Skipping
    // type-5 PADs is what allows llguidance/derivre to compile a byte trie
    // over this vocab without aborting on placeholder strings.
    if let Some(gguf_file::Value::Array(type_arr)) =
        content.metadata.get("tokenizer.ggml.token_type")
    {
        let mut special_tokens = Vec::new();
        for (idx, type_val) in type_arr.iter().enumerate() {
            if let gguf_file::Value::I32(type_id) = type_val {
                if matches!(type_id, 2..=4) {
                    if let Some(token_str) = match content.metadata.get("tokenizer.ggml.tokens") {
                        Some(gguf_file::Value::Array(arr)) => arr.get(idx).and_then(|v| {
                            if let gguf_file::Value::String(s) = v {
                                Some(s.clone())
                            } else {
                                None
                            }
                        }),
                        _ => None,
                    } {
                        special_tokens.push(tokenizers::AddedToken::from(token_str, true));
                    }
                }
            }
        }
        if !special_tokens.is_empty() {
            tokenizer.add_special_tokens(&special_tokens);
            debug!(
                "🔐 Marked {} special tokens for skip_special_tokens support",
                special_tokens.len()
            );
        }
    }

    // BOS post-processor. The GGUF `add_bos_token` flag (mirrors HF's
    // `add_bos_token`) tells us the model expects a leading BOS on every
    // sequence. Our GGUF-built BPE tokenizer has no post-processor, so
    // `encode(prompt, true)` would NOT prepend it - and Gemma is acutely
    // BOS-sensitive (no `<bos>` -> degenerate/garbage output). llama.cpp /
    // Ollama add it automatically; we must too. Scoped to gemma-family
    // arches for now to avoid shifting the carefully-tuned non-gemma
    // benchmarks; honoring `add_bos_token` globally is the correct general
    // behavior and can be widened once each arch is re-validated.
    let declared_bos = matches!(
        content.metadata.get("tokenizer.ggml.add_bos_token"),
        Some(gguf_file::Value::Bool(true))
    );
    let arch = get_gguf_string(content, "general.architecture").unwrap_or_default();
    // The gemma4 GGUFs declare `add_bos_token = false` and then need one anyway: their chat
    // template carries `<bos>` as literal text, so the templated path is fed correctly and
    // only a bare continuation arrives without it. The model is acutely BOS-sensitive, and
    // what came back was a fragment of the prompt echoed and a stop token - nine tokens where
    // the reference produced a hundred and twenty-eight.
    //
    // llama.cpp overrides the same flag for the same family and logs that it did
    // (llama-vocab.cpp, "workaround for Gemma 4"). Following the file here would mean
    // following it into a known-wrong value, so the override is stated rather than silent.
    let forced_bos = !declared_bos && arch.starts_with("gemma4");
    if forced_bos {
        info!(
            "🔖 {arch} declares add_bos_token=false and needs one regardless - adding it, as \
             the reference implementation does for this family"
        );
    }
    let add_bos = declared_bos || forced_bos;
    // The flag is the model's own statement that it expects a leading BOS, and
    // the comment this replaces already said honouring it globally is correct -
    // it was nonetheless restricted to one family. A completion prompt reaching
    // a BOS-sensitive model bare degenerates: gemma4 answered "olde olde olde"
    // to a story opening. Gating on whether the chat template carries a BOS was
    // tried and MEASURED WRONG - it disabled the processor for exactly the model
    // that needed it, since that template does carry one, and the raw path
    // degenerated again. Doubling is not the risk it looked like: gemma4 ran
    // with both for a long time and answers correctly through the templated path.
    if add_bos {
        if let Some(gguf_file::Value::U32(bos_id)) =
            content.metadata.get("tokenizer.ggml.bos_token_id")
        {
            let bos_str = match content.metadata.get("tokenizer.ggml.tokens") {
                Some(gguf_file::Value::Array(arr)) => arr.get(*bos_id as usize).and_then(|v| {
                    if let gguf_file::Value::String(s) = v {
                        Some(s.clone())
                    } else {
                        None
                    }
                }),
                _ => None,
            };
            if let Some(bos_str) = bos_str {
                use tokenizers::processors::template::TemplateProcessing;
                let built = (|| -> AnyResult<TemplateProcessing> {
                    Ok(TemplateProcessing::builder()
                        .try_single(format!("{bos_str}:0 $A:0"))
                        .map_err(|e| anyhow!("{e}"))?
                        .try_pair(format!("{bos_str}:0 $A:0 $B:1"))
                        .map_err(|e| anyhow!("{e}"))?
                        .special_tokens(vec![(bos_str.clone(), *bos_id)])
                        .build()
                        .map_err(|e| anyhow!("{e}"))?)
                })();
                match built {
                    Ok(post) => {
                        tokenizer.with_post_processor(Some(post));
                        debug!("🔖 Added BOS post-processor ('{bos_str}' id={bos_id}) for {arch}");
                    }
                    Err(e) => {
                        warn!(
                            "Failed to build BOS post-processor for {arch}: {e}; \
                               proceeding without (gemma may be incoherent)"
                        );
                    }
                }
            }
        }
    }

    Ok(tokenizer)
}

/// Build sampling strategy from explicit parameters
/// Did a prefill fail because we resumed from a position the resident KV does not have?
///
/// The two lengths are in the message and nowhere else - `cannot broadcast [1591, 4543]
/// to [1, 16, 1591, 2635]` is an attention mask built for 4543 columns meeting a KV that
/// holds 2635. It means the prompt cache described more rows than the cache holds, which
/// can happen whenever the resident KV shrinks between one request writing its entry and
/// the next reading it: a trim from a request that never finished, a model reloaded
/// underneath it. The prefix is worth reusing, but never at the price of the request.
pub(crate) fn is_resume_length_mismatch(e: &crate::tensor::Error) -> bool {
    let m = e.to_string();
    m.contains("cannot broadcast") || m.contains("shape mismatch")
}

pub(crate) fn build_sampling_from(temperature: f32, top_p: f32, top_k: usize) -> Sampling {
    if temperature == 0.0 {
        Sampling::ArgMax
    } else if top_k > 0 && top_p < 1.0 {
        Sampling::TopKThenTopP {
            k: top_k,
            p: top_p as f64,
            temperature: temperature as f64,
        }
    } else if top_k > 0 {
        Sampling::TopK {
            k: top_k,
            temperature: temperature as f64,
        }
    } else {
        Sampling::TopP {
            p: top_p as f64,
            temperature: temperature as f64,
        }
    }
}

#[cfg(test)]
mod cuda_oom_detector_tests {
    use super::*;

    #[test]
    fn is_cuda_oom_matches_both_lower_and_upper_case_oom_strings() {
        // The two patterns documented in the body - both must match
        // any case since CUDA error messages mix cases ("CUDA_ERROR_..."
        // vs "out of memory"). String search is lowercase-after-
        // to_ascii_lowercase, so a refactor that dropped the .to_lowercase
        // would silently skip uppercased messages.
        let s = "DriverError(CUDA_ERROR_OUT_OF_MEMORY, ...)";
        assert!(is_cuda_oom(&s));
        let s = "alloc failed: out of memory";
        assert!(is_cuda_oom(&s));
        // Mixed case in middle of message - still matches.
        let s = "boom: Cuda_Error_OUT_OF_MEMORY at line 17";
        assert!(is_cuda_oom(&s));
    }

    #[test]
    fn is_cuda_oom_rejects_unrelated_errors() {
        // Other CUDA errors are NOT OOM - must not trigger the
        // OOM-fallback path (which would silently shuffle layers to
        // CPU when the real problem is e.g. invalid kernel arg).
        assert!(!is_cuda_oom(&"DriverError(CUDA_ERROR_ILLEGAL_ADDRESS)"));
        assert!(!is_cuda_oom(
            &"shape mismatch: expected (4, 64), got (4, 32)"
        ));
        assert!(
            !is_cuda_oom(&"oom is a substring of zoom but not of cool"),
            "false positives on substrings like 'zoom' would trigger spurious fallbacks"
        );
    }
}

#[cfg(test)]
mod apply_repeat_penalty_tests {
    use super::*;
    use crate::tensor::Device;

    #[test]
    fn apply_repeat_penalty_is_noop_when_penalty_is_one() {
        // Penalty <= 1.0 -> no change. Pin so a refactor doesn't drop
        // the early-return (any modification at penalty=1 is a bug
        // since /1 = identity, *1 = identity).
        let logits = Tensor::from_vec(vec![1.0_f32, -1.0, 2.0], 3, &Device::Cpu).unwrap();
        let out = apply_repeat_penalty(&logits, &[0, 1, 2], 1.0, 3).unwrap();
        let got: Vec<f32> = out.to_vec1().unwrap();
        assert_eq!(got, vec![1.0, -1.0, 2.0]);
    }

    #[test]
    fn apply_repeat_penalty_is_noop_when_last_n_or_recent_empty() {
        let logits = Tensor::from_vec(vec![1.0_f32, 2.0], 2, &Device::Cpu).unwrap();
        // last_n=0 -> no penalty applied.
        let out = apply_repeat_penalty(&logits, &[0, 1], 1.5, 0).unwrap();
        assert_eq!(out.to_vec1::<f32>().unwrap(), vec![1.0, 2.0]);
        // empty recent_tokens -> no penalty.
        let out = apply_repeat_penalty(&logits, &[], 1.5, 8).unwrap();
        assert_eq!(out.to_vec1::<f32>().unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn apply_repeat_penalty_divides_positive_logits_multiplies_negative() {
        // Matches llama.cpp / Ollama: positive logits get DIVIDED
        // by penalty (reduces probability mass on that token), negative
        // logits get MULTIPLIED (pushes them MORE negative, same goal).
        let logits = Tensor::from_vec(vec![2.0_f32, -2.0, 4.0], 3, &Device::Cpu).unwrap();
        let out = apply_repeat_penalty(&logits, &[0, 1, 2], 2.0, 8).unwrap();
        let got: Vec<f32> = out.to_vec1().unwrap();
        assert!((got[0] - 1.0).abs() < 1e-6, "2.0/2 = 1.0, got {}", got[0]);
        assert!(
            (got[1] - -4.0).abs() < 1e-6,
            "-2.0*2 = -4.0, got {}",
            got[1]
        );
        assert!((got[2] - 2.0).abs() < 1e-6, "4.0/2 = 2.0, got {}", got[2]);
    }

    #[test]
    fn apply_repeat_penalty_windows_to_last_n_only() {
        // last_n=2 means only the most-recent 2 tokens get penalised.
        // recent=[0, 1, 2, 3] last_n=2 -> only tokens 2 and 3 affected;
        // tokens 0 and 1 stay at original logit.
        let logits = Tensor::from_vec(vec![1.0_f32; 4], 4, &Device::Cpu).unwrap();
        let out = apply_repeat_penalty(&logits, &[0, 1, 2, 3], 2.0, 2).unwrap();
        let got: Vec<f32> = out.to_vec1().unwrap();
        assert_eq!(got[0], 1.0, "token 0 outside last-2 window - unchanged");
        assert_eq!(got[1], 1.0, "token 1 outside last-2 window - unchanged");
        assert_eq!(got[2], 0.5, "token 2 in window - divided by 2");
        assert_eq!(got[3], 0.5, "token 3 in window - divided by 2");
    }

    #[test]
    fn apply_repeat_penalty_silently_skips_out_of_range_tokens() {
        // Defensive: if recent_tokens contains an id past the vocab,
        // skip it rather than panic. Tokenizer mismatch in a fork
        // could otherwise crash decode.
        let logits = Tensor::from_vec(vec![1.0_f32, 1.0], 2, &Device::Cpu).unwrap();
        let out = apply_repeat_penalty(&logits, &[0, 9999, 1], 2.0, 8).unwrap();
        let got: Vec<f32> = out.to_vec1().unwrap();
        assert_eq!(got[0], 0.5, "in-range token penalised");
        assert_eq!(got[1], 0.5, "in-range token penalised");
        // No panic from the 9999 sentinel.
    }
}
