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

/// The synthesiser a bare reference-voice name is spoken with: its reference voices are
/// WAVs under `<models>/pocket-tts-voices/`.
pub(super) const CLONING_VOICES: &str = "pocket-tts";

pub(super) fn default_voice_for_language(lang: &str) -> String {
    match lang {
        // Kyutai tts-1.6b-en_fr is multilingual (en/fr) - the best default for both.
        // Other languages fall back to Piper / pocket-tts.
        "fr" | "en" => "kyutai".to_string(),
        _ => format!("{CLONING_VOICES}-default"),
    }
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
