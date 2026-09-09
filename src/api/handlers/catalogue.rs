//! What a request asks the cluster for, in the catalogue's own names.
//!
//! Each media route names its model in its own way - an alias, a family word, a language -
//! while a node's catalogue lists checkpoints and repositories. Every match between the two
//! is written here, once, and tested against the shapes the catalogue actually lists.

use crate::distributed::membership::NodeState;
use crate::distributed::routing::can_serve;

fn catalogue_has(node: &NodeState, matches: impl Fn(&str) -> bool) -> bool {
    node.serves
        .as_ref()
        .is_some_and(|catalogue| catalogue.iter().any(|entry| matches(entry)))
}

/// The catalogue entry a sound or music request asks for. ACE-Step is one name in the
/// request and one entry per DiT checkpoint in the catalogue, so the variant joins the
/// name; an empty answer means the request named nothing and takes any sound model.
pub(crate) fn sound_entry(body: &serde_json::Value) -> String {
    let field = |key: &str| {
        body.get(key)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase()
    };
    let requested = field("model");
    let is_music = requested == "ace-step"
        || requested == "acestep"
        || requested == "music"
        || requested.starts_with("ace-step:");
    if !is_music {
        return requested;
    }
    let variant = match field("dit_model").as_str() {
        "sft" => "sft",
        "base" => "base",
        "xl-turbo" => "xl-turbo",
        "xl-sft" | "xl" => "xl-sft",
        "xl-base" => "xl-base",
        _ => "turbo",
    };
    format!("ace-step-{variant}")
}

/// Whether a node holds the sound model a request asks for: the named entry, or any
/// sound model when the request named none.
pub(crate) fn serves_sound(node: &NodeState, entry: &str) -> bool {
    if entry.is_empty() {
        can_serve(node, "stable-audio") || can_serve(node, "ezaudio")
    } else {
        can_serve(node, entry)
    }
}

/// Whether a catalogue entry is the speech model a request names. A request names a
/// backend or a voice; the catalogue lists the repository that backend loads.
pub(crate) fn speech_entry_matches(entry: &str, requested: &str) -> bool {
    let entry = entry.to_ascii_lowercase();
    let requested = requested.trim().to_ascii_lowercase();
    if requested.is_empty() {
        return false;
    }
    if requested.starts_with("piper/") {
        return entry == requested;
    }
    if requested.starts_with("pocket") {
        return entry.contains("pocket-tts");
    }
    if requested.starts_with("kyutai") {
        return entry.starts_with("kyutai/tts");
    }
    if requested.contains("parler") {
        return entry.contains("parler-tts");
    }
    if requested.contains("xtts") {
        return entry.contains("xtts");
    }
    if requested.contains("openvoice") {
        return entry.contains("openvoice");
    }
    entry == requested
}

pub(crate) fn serves_speech(node: &NodeState, requested: &str) -> bool {
    catalogue_has(node, |entry| speech_entry_matches(entry, requested))
}

/// Whether a catalogue entry is the transcription model a request names. `whisper-1` is
/// the OpenAI alias of the small model; a bare size names the repository ending in it.
pub(crate) fn transcription_entry_matches(entry: &str, requested: &str) -> bool {
    let requested = match requested.trim().to_ascii_lowercase().as_str() {
        "" | "whisper-1" => "whisper-small".to_string(),
        other => other.to_string(),
    };
    let entry = entry.to_ascii_lowercase();
    entry == requested
        || entry.ends_with(&format!("/{requested}"))
        || entry.ends_with(&format!("/faster-{requested}"))
}

pub(crate) fn serves_transcription(node: &NodeState, requested: &str) -> bool {
    catalogue_has(node, |entry| transcription_entry_matches(entry, requested))
}

/// The `model` text field of an upload, read from the body as it arrived. An upload is
/// parsed once here to decide where it goes, and again by the route that runs it.
pub(crate) async fn upload_model(
    headers: &axum::http::HeaderMap,
    body: &axum::body::Bytes,
) -> Option<String> {
    let mut multipart = upload_multipart(headers, body.clone()).await.ok()?;
    while let Ok(Some(field)) = multipart.next_field().await {
        if field.name() == Some("model") {
            return field.text().await.ok().map(|m| m.trim().to_string());
        }
    }
    None
}

/// The multipart reader of an upload held as bytes.
pub(crate) async fn upload_multipart(
    headers: &axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Result<axum::extract::Multipart, String> {
    use axum::extract::FromRequest;
    let content_type = headers
        .get(axum::http::header::CONTENT_TYPE)
        .cloned()
        .ok_or("an upload names its content type")?;
    let request = axum::http::Request::builder()
        .header(axum::http::header::CONTENT_TYPE, content_type)
        .body(axum::body::Body::from(body))
        .map_err(|e| e.to_string())?;
    axum::extract::Multipart::from_request(request, &())
        .await
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_music_request_names_the_catalogue_entry_of_its_checkpoint() {
        assert_eq!(
            sound_entry(&json!({"model": "ace-step", "dit_model": "xl-sft"})),
            "ace-step-xl-sft"
        );
        assert_eq!(
            sound_entry(&json!({"model": "acestep", "dit_model": "xl"})),
            "ace-step-xl-sft"
        );
        assert_eq!(sound_entry(&json!({"model": "music"})), "ace-step-turbo");
        assert_eq!(
            sound_entry(&json!({"model": "stable-audio"})),
            "stable-audio"
        );
        assert_eq!(sound_entry(&json!({})), "");
    }

    #[test]
    fn a_speech_request_finds_the_repository_its_backend_loads() {
        assert!(speech_entry_matches("kyutai/tts-1.6b-en_fr", "kyutai"));
        assert!(!speech_entry_matches("kyutai/pocket-tts", "kyutai"));
        assert!(speech_entry_matches("kyutai/pocket-tts", "pocket-tts"));
        assert!(speech_entry_matches(
            "parler-tts/parler-tts-mini-v1",
            "parler-tts-mini-v1"
        ));
        assert!(speech_entry_matches("coqui/XTTS-v2", "xtts"));
        assert!(speech_entry_matches("myshell-ai/OpenVoiceV2", "openvoice"));
        assert!(speech_entry_matches(
            "piper/de_DE-thorsten-medium",
            "piper/de_DE-thorsten-medium"
        ));
        assert!(!speech_entry_matches("qwen3:8b", "kyutai"));
    }

    #[test]
    fn a_transcription_request_finds_the_repository_of_its_size() {
        assert!(transcription_entry_matches(
            "openai/whisper-small",
            "whisper-1"
        ));
        assert!(transcription_entry_matches(
            "Systran/faster-whisper-small",
            ""
        ));
        assert!(transcription_entry_matches(
            "openai/whisper-large-v3",
            "whisper-large-v3"
        ));
        assert!(!transcription_entry_matches(
            "openai/whisper-medium",
            "whisper-small"
        ));
    }

    #[test]
    fn a_node_serves_what_its_catalogue_matches() {
        let mut node = NodeState::default();
        assert!(!serves_speech(&node, "kyutai"));
        node.serves = Some(vec![
            "kyutai/tts-1.6b-en_fr".into(),
            "openai/whisper-small".into(),
            "ezaudio".into(),
        ]);
        assert!(serves_speech(&node, "kyutai"));
        assert!(serves_transcription(&node, "whisper-1"));
        assert!(serves_sound(&node, ""));
        assert!(!serves_sound(&node, "ace-step-turbo"));
    }
}
