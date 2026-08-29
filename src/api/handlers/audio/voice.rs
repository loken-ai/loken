//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

/// Default TTS voice for an ASR-detected language (ISO code). Multilingual
/// output comes from per-language piper voices (fast VITS); English routes to
/// pocket-tts. This is the routing
/// Extend the map as piper voices are downloaded under <hf_dir>/piper/<voice>/.
/// A backend that can SPEAK this language, or `None` when none here can.
///
/// Distinct from [`default_voice_for_language`], which always answers - it has to, as it
/// serves a conversation that must say something. This one is allowed to say no, because
/// a caller who asked for a language deserves a refusal rather than English audio.
pub(super) fn speakable_backend(lang: &str) -> Option<String> {
    match lang {
        // Kyutai tts-1.6b-en_fr is trained on both and infers which from the text.
        "en" | "fr" => Some("kyutai".to_string()),
        _ => None,
    }
}

pub(super) fn default_voice_for_language(lang: &str) -> String {
    match lang {
        // Kyutai tts-1.6b-en_fr is multilingual (en/fr) - the best default for both.
        // Other languages fall back to Piper / pocket-tts.
        "fr" | "en" => "kyutai".to_string(),
        _ => "pocket-tts-default".to_string(), // fallback
    }
}

/// Same naming heuristic the GUI uses (chat_tab::ModelModality::from_
/// model_name) for whisper / ASR pipelines. Triggers handle_chat_asr
/// on /api/chat when the user attaches audio bytes via images[]
/// (the Ollama chat shape doesn't define a separate audios[] for input,
/// so the GUI re-uses images[] as a bytes carrier for ASR - see
/// the GUI's chat tab (separate repository)).
pub(crate) fn is_asr_model(model_name: &str) -> bool {
    let lower = model_name.to_lowercase();
    lower.contains("whisper")
}

/// Same naming heuristics the GUI uses (chat_tab::ModelModality::
/// from_model_name) so the server agrees about which models are TTS
/// pipelines. Triggers handle_chat_tts on /api/chat and /api/generate.
pub(crate) fn is_tts_model(model_name: &str) -> bool {
    let lower = model_name.to_lowercase();
    lower.contains("parler")
        || lower.contains("piper")
        || lower.contains("-tts")
        || lower.starts_with("tts")
        || lower.contains("/tts-")
        || lower.contains("bark")
        || lower.contains("kokoro")
        || lower.contains("f5-tts")
        || lower.contains("fish-speech")
        || lower.contains("metavoice")
        || lower.contains("kyutai")
        || lower.contains("pocket")
}

/// Detect a Piper (native VITS) TTS request and resolve its voice name. A model
/// id routes to Piper when it contains `piper`; the voice is whatever follows
/// `piper/` or `piper-` (e.g. `piper/fr_FR-tom-medium` -> `fr_FR-tom-medium`),
/// defaulting to the bundled French voice. Returns `None` for non-Piper ids.
pub(super) fn piper_voice_name(model_name: &str) -> Option<String> {
    let lower = model_name.to_lowercase();
    if !lower.contains("piper") {
        return None;
    }
    // Strip a leading `piper/` or `piper-` (case-insensitive) to get the voice.
    let voice = model_name
        .rsplit('/')
        .next()
        .unwrap_or(model_name)
        .trim_start_matches(|c| c == '/');
    let voice = if let Some(stripped) = voice
        .strip_prefix("piper-")
        .or_else(|| voice.strip_prefix("Piper-"))
        .or_else(|| voice.strip_prefix("PIPER-"))
    {
        stripped
    } else if voice.eq_ignore_ascii_case("piper") {
        ""
    } else {
        voice
    };
    let voice = if voice.is_empty() {
        "fr_FR-tom-medium"
    } else {
        voice
    };
    Some(voice.to_string())
}

/// Resolve a Piper voice's `.onnx` path under the configured HF models dir:
/// `<hf_dir>/piper/<voice>/<voice>.onnx` (the `.onnx.json` sibling is read by
/// the loader). Honours the models-via-config rule - no hardcoded path.
pub(super) fn piper_onnx_path(hf_models_dir: &str, voice: &str) -> std::path::PathBuf {
    let base = std::path::PathBuf::from(hf_models_dir);
    // The HF dir may already point at the `hub/` subfolder; Piper voices live
    // beside it under `piper/`, so step back out of `hub` when present.
    let root = if base.file_name().is_some_and(|n| n == "hub") {
        base.parent().map(|p| p.to_path_buf()).unwrap_or(base)
    } else {
        base
    };
    root.join("piper").join(voice).join(format!("{voice}.onnx"))
}

/// Detect a pocket-tts (Kyutai) request and resolve its voice name.
/// Routes when the id contains `pocket`; the voice is whatever follows
/// `pocket-tts-` / `pocket-tts/` (e.g. `pocket-tts-alba` -> `alba`), default `default`.
/// Route a requested TTS model name to the Kyutai `tts-1.6b-en_fr` backend and extract
/// the voice selector. `kyutai` / `kyutai-tts` / `kyutai-en_fr` -> default voice (None);
/// `kyutai-<voice>` or `kyutai/<voice>` -> that voice substring (e.g. `kyutai-alba` or
/// `kyutai/alba-mackenna/a-moment-by`). Returns `None` (not this backend) otherwise.
pub(super) fn kyutai_voice_name(model_name: &str) -> Option<Option<String>> {
    let lower = model_name.to_lowercase();
    if !lower.contains("kyutai") {
        return None;
    }
    // strip a leading org path, then the `kyutai` marker + separators
    let tail = model_name
        .rsplit_once("kyutai")
        .map(|(_, t)| t)
        .unwrap_or("");
    let voice = tail.trim_start_matches(['-', '/', '_']);
    let voice = voice
        .strip_prefix("tts")
        .or_else(|| voice.strip_prefix("en_fr"))
        .unwrap_or(voice)
        .trim_start_matches(['-', '/', '_']);
    Some(if voice.is_empty() {
        None
    } else {
        Some(voice.to_string())
    })
}

pub(super) fn pocket_tts_voice_name(model_name: &str) -> Option<String> {
    let lower = model_name.to_lowercase();
    if !lower.contains("pocket") {
        return None;
    }
    let tail = model_name.rsplit('/').next().unwrap_or(model_name);
    let voice = tail
        .strip_prefix("pocket-tts-")
        .or_else(|| tail.strip_prefix("pocket-tts_"))
        .unwrap_or("");
    Some(if voice.is_empty() {
        "default".to_string()
    } else {
        voice.to_string()
    })
}

/// Resolve a pocket-tts reference voice WAV: `<hf_dir>/pocket-tts-voices/<voice>.wav`
/// (steps out of `hub/` like the Piper path). Models-via-config - no hardcoded path.
pub(super) fn pocket_tts_voice_path(hf_models_dir: &str, voice: &str) -> std::path::PathBuf {
    let base = std::path::PathBuf::from(hf_models_dir);
    let root = if base.file_name().is_some_and(|n| n == "hub") {
        base.parent().map(|p| p.to_path_buf()).unwrap_or(base)
    } else {
        base
    };
    root.join("pocket-tts-voices").join(format!("{voice}.wav"))
}
