//! Custom voices: `/v1/audio/voices` creates, retrieves and deletes a voice, and
//! `/v1/audio/voice_consents` keeps the consent recordings a voice from a clip rests on.
//!
//! A voice is a speaker embedding made by the speaker encoder of a speech talker that
//! clones, and `/v1/audio/speech` speaks it by its id.

use super::voice_store::{is_id, NewVoice, StoredConsent, StoredVoice, VoiceStore, CONSENT_PREFIX};
use super::*;
use crate::inference::engine::tts_engine::{speech_talker_voice, TalkerVoice};

/// The longest voice or consent name, in characters.
const NAME_MAX_CHARS: usize = 256;
/// The longest voice description and the longest script, in characters.
const PROMPT_MAX_CHARS: usize = 2000;
/// The largest audio sample or consent recording, in bytes.
const UPLOAD_MAX_BYTES: usize = 10 * 1024 * 1024;
/// The longest audio sample, in seconds.
const SAMPLE_MAX_SECONDS: f64 = 60.0;
/// The most consents a list returns, and the number it returns by default.
const LIST_MAX: usize = 100;
const LIST_DEFAULT: usize = 20;

fn err_resp(code: axum::http::StatusCode, msg: impl Into<String>) -> axum::response::Response {
    use axum::response::IntoResponse;
    (code, Json(openai_error_body(code, msg.into()))).into_response()
}

fn bad_request(msg: impl Into<String>) -> axum::response::Response {
    err_resp(axum::http::StatusCode::BAD_REQUEST, msg)
}

fn not_found(msg: impl Into<String>) -> axum::response::Response {
    err_resp(axum::http::StatusCode::NOT_FOUND, msg)
}

fn server_error(msg: impl Into<String>) -> axum::response::Response {
    err_resp(axum::http::StatusCode::INTERNAL_SERVER_ERROR, msg)
}

/// The Voice object of `voice`.
pub(super) fn voice_object(voice: &StoredVoice) -> serde_json::Value {
    let mut o = serde_json::json!({
        "id": voice.id,
        "created_at": voice.created_at,
        "name": voice.name,
        "object": "audio.voice",
        "type": voice.kind,
    });
    if let Some(prompt) = &voice.prompt {
        o["prompt"] = serde_json::Value::String(prompt.clone());
    }
    o
}

/// The VoiceConsent object of `consent`.
fn consent_object(consent: &StoredConsent) -> serde_json::Value {
    serde_json::json!({
        "id": consent.id,
        "created_at": consent.created_at,
        "language": consent.language,
        "name": consent.name,
        "object": "audio.voice_consent",
    })
}

/// A speech talker checkpoint: its catalogue id, its directory and what it offers a stored
/// voice.
#[derive(Debug, Clone)]
pub(super) struct Talker {
    pub(super) id: String,
    pub(super) voice: TalkerVoice,
}

/// The speech talker `requested` names, `None` when it names no speech talker.
pub(super) fn talker_named(hf: &str, requested: &str) -> Option<Result<Talker, String>> {
    let speech = speech_request(hf, requested)
        .filter(|s| s.model == crate::inference::media_facts::SpeechModel::Qwen3Tts)?;
    Some(speech_talker_voice(&speech.path).map(|voice| Talker {
        id: speech.id,
        voice,
    }))
}

/// Every speech talker checkpoint of the Hugging Face store, by catalogue id.
pub(super) fn store_talkers(hf: &str) -> Vec<Talker> {
    let base = std::path::PathBuf::from(hf);
    let mut ids: Vec<String> = [base.clone(), base.join("hub")]
        .iter()
        .filter_map(|d| std::fs::read_dir(d).ok())
        .flat_map(|d| d.flatten())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            name.strip_prefix("models--").map(|s| s.replace("--", "/"))
        })
        .collect();
    ids.sort();
    ids.dedup();
    ids.iter()
        .filter_map(|id| talker_named(hf, id).and_then(Result::ok))
        .collect()
}

/// The talker of the request's `model` when it names one that `fits`, else the first of the
/// store that fits.
///
/// `auto` and an absent model resolve from the store.
/// A named model that does not fit is refused with `what` it must be.
fn pick_talker(
    hf: &str,
    model: Option<&str>,
    fits: impl Fn(&TalkerVoice) -> bool,
    what: &str,
) -> Result<Talker, String> {
    if let Some(m) = model.filter(|m| !m.trim().is_empty() && *m != "auto") {
        if let Some(t) = talker_named(hf, m) {
            let t = t?;
            if fits(&t.voice) {
                return Ok(t);
            }
        }
    }
    store_talkers(hf)
        .into_iter()
        .find(|t| fits(&t.voice))
        .ok_or_else(|| format!("no checkpoint of the store is {what}"))
}

/// Whether the stored voice `voice` fits the talker `talker`, with the refusal when not.
pub(super) fn voice_fits(voice: &StoredVoice, talker: &Talker) -> Result<(), String> {
    if talker.voice.names {
        return Err(format!(
            "'{}' speaks the voices its checkpoint names: pass one of those names as `voice`, \
             or speak the stored voice '{}' by a checkpoint that clones, such as '{}', which \
             made it",
            talker.id, voice.id, voice.checkpoint
        ));
    }
    if !talker.voice.clones {
        return Err(format!(
            "'{}' speaks a voice described in words: a stored voice is spoken by a checkpoint \
             that clones, such as '{}', which made it",
            talker.id, voice.checkpoint
        ));
    }
    if talker.voice.width != voice.width {
        return Err(format!(
            "the voice '{}' is a speaker embedding of {} values made by '{}': '{}' takes {}",
            voice.id, voice.width, voice.checkpoint, talker.id, talker.voice.width
        ));
    }
    Ok(())
}

/// A request's text fields and its one audio file.
#[derive(Debug, Default)]
pub(super) struct Form {
    pub(super) fields: std::collections::HashMap<String, String>,
    /// The bytes, the file name and the field name of the audio file.
    pub(super) file: Option<(Vec<u8>, Option<String>, String)>,
}

impl Form {
    fn text(&self, name: &str) -> Option<&str> {
        self.fields.get(name).map(String::as_str)
    }
}

/// The form of `request`: multipart form data, or a JSON object of strings.
///
/// The multipart field `file_field` is read as the audio file.
pub(super) async fn read_form(
    request: axum::extract::Request,
    file_field: &str,
) -> Result<Form, String> {
    use axum::extract::FromRequest;
    let multipart = request
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("multipart/form-data"));
    let mut form = Form::default();
    if multipart {
        let mut parts = axum::extract::Multipart::from_request(request, &())
            .await
            .map_err(|e| format!("multipart parse: {e}"))?;
        while let Some(field) = parts
            .next_field()
            .await
            .map_err(|e| format!("multipart parse: {e}"))?
        {
            let name = field.name().unwrap_or("").to_string();
            if name == file_field {
                let file_name = field.file_name().map(str::to_string);
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|e| format!("multipart read of `{name}`: {e}"))?;
                form.file = Some((bytes.to_vec(), file_name, name));
            } else {
                let text = field
                    .text()
                    .await
                    .map_err(|e| format!("multipart read of `{name}`: {e}"))?;
                form.fields.insert(name, text);
            }
        }
        return Ok(form);
    }
    let bytes = axum::body::to_bytes(request.into_body(), usize::MAX)
        .await
        .map_err(|e| format!("body read: {e}"))?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("invalid request body: {e}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "invalid request body: a JSON object is expected".to_string())?;
    for (k, v) in object {
        match v {
            serde_json::Value::String(s) => {
                form.fields.insert(k.clone(), s.clone());
            }
            serde_json::Value::Null => {}
            _ => return Err(format!("`{k}` must be a string")),
        }
    }
    Ok(form)
}

/// The field `name` of `form`, required and between 1 and `max` characters.
fn bounded(form: &Form, name: &str, max: usize) -> Result<String, String> {
    let value = form
        .text(name)
        .ok_or_else(|| format!("`{name}` is required"))?;
    check_length(name, value, max)?;
    Ok(value.to_string())
}

/// The text a voice designed from a description speaks once: the field `script_hint` of
/// `form`, required and between 1 and [`PROMPT_MAX_CHARS`] characters.
fn script_hint(form: &Form) -> Result<String, String> {
    let Some(script) = form.text("script_hint") else {
        return Err(
            "`script_hint` is required with `type` `prompt`: the voice is made by speaking \
             this text once in the described voice, so give a sentence in the language the \
             voice will speak"
                .to_string(),
        );
    };
    check_length("script_hint", script, PROMPT_MAX_CHARS)?;
    Ok(script.to_string())
}

fn check_length(name: &str, value: &str, max: usize) -> Result<(), String> {
    let n = value.chars().count();
    if n == 0 || n > max {
        return Err(format!(
            "`{name}` must hold 1 to {max} characters, it holds {n}"
        ));
    }
    if value.trim().is_empty() {
        return Err(format!("`{name}` must not contain only whitespace"));
    }
    Ok(())
}

/// The mono samples and the rate of the clip `bytes`, with its length bounded.
fn decode_clip(bytes: &[u8], field: &str) -> Result<(Vec<f32>, u32), String> {
    if bytes.len() > UPLOAD_MAX_BYTES {
        return Err(format!(
            "`{field}` holds {} bytes: at most {UPLOAD_MAX_BYTES} are taken",
            bytes.len()
        ));
    }
    let (planar, channels, rate) =
        decode_audio_planar(bytes).map_err(|e| format!("`{field}` is not audio: {e}"))?;
    let frames = planar.len() / channels;
    let mono: Vec<f32> = (0..frames)
        .map(|i| (0..channels).map(|c| planar[c * frames + i]).sum::<f32>() / channels as f32)
        .collect();
    let seconds = frames as f64 / rate as f64;
    if seconds > SAMPLE_MAX_SECONDS {
        return Err(format!(
            "`{field}` lasts {seconds:.1} s: at most {SAMPLE_MAX_SECONDS} s are taken"
        ));
    }
    Ok((mono, rate))
}

/// Load the speech checkpoint `id` into the speech engine, with room made on the cards.
async fn load_talker(state: &APIServer, id: &str) -> Result<(), axum::response::Response> {
    use crate::api::handlers::catalogue;
    let demand = catalogue::listed_size(state, |e| catalogue::speech_entry_matches(e, id))
        .await
        .map(catalogue::whole_load_demand)
        .unwrap_or(0);
    crate::inference::place::vram_manager::ensure_gpu_headroom(
        "tts",
        demand,
        catalogue::WHOLE_LOAD_RESERVE,
    )
    .await;
    ensure_tts_model_loaded(state, Some(id))
        .await
        .map_err(|e| err_resp(http_status_for_load_error(&e), e))
}

/// The speaker embedding of the clip `samples` at `rate`, by the speaker encoder of `base`.
async fn embed(
    state: &APIServer,
    base: &Talker,
    samples: Vec<f32>,
    rate: u32,
) -> Result<Vec<f32>, axum::response::Response> {
    load_talker(state, &base.id).await?;
    state
        .tts_engine
        .speaker_embedding(samples, rate)
        .await
        .map_err(|e| server_error(format!("speaker encoder: {e:#}")))
}

/// `POST /v1/audio/voices`: a voice from a description (`type: prompt`) or from an audio
/// sample under a consent (`type: audio_sample`, the default).
pub(crate) async fn audio_voices_create(
    axum::extract::State(state): axum::extract::State<APIServer>,
    request: axum::extract::Request,
) -> axum::response::Response {
    let _inflight = crate::distributed::rate_meter::InFlight::enter();
    use axum::response::IntoResponse;
    let form = match read_form(request, "audio_sample").await {
        Ok(f) => f,
        Err(e) => return bad_request(e),
    };
    let name = match bounded(&form, "name", NAME_MAX_CHARS) {
        Ok(n) => n,
        Err(e) => return bad_request(e),
    };
    let hf = state.huggingface_models_dir.clone();
    let model = form.text("model").map(str::to_string);
    let store = VoiceStore::of(&hf);
    let made = match form.text("type").unwrap_or("audio_sample") {
        "prompt" => create_from_prompt(&state, &form, name, model.as_deref(), &hf).await,
        "audio_sample" => create_from_sample(&state, form, name, model.as_deref(), &hf).await,
        other => {
            return bad_request(format!(
                "`type` is '{other}': it is `prompt` or `audio_sample`"
            ))
        }
    };
    let (voice, speaker) = match made {
        Ok(m) => m,
        Err(r) => return r,
    };
    match store.create_voice(voice, &speaker) {
        Ok(stored) => {
            info!(
                "voices: {} made by {} ({} values)",
                stored.id, stored.checkpoint, stored.width
            );
            Json(voice_object(&stored)).into_response()
        }
        Err(e) => server_error(format!("voice store: {e:#}")),
    }
}

/// A voice designed from the description of `form`, then its speaker embedding.
async fn create_from_prompt(
    state: &APIServer,
    form: &Form,
    name: String,
    model: Option<&str>,
    hf: &str,
) -> Result<(NewVoice, Vec<f32>), axum::response::Response> {
    let prompt = bounded(form, "prompt", PROMPT_MAX_CHARS).map_err(bad_request)?;
    let script = script_hint(form).map_err(bad_request)?;
    let base = pick_talker(
        hf,
        model,
        |v| v.clones && v.encodes,
        "a speech talker that clones",
    )
    .map_err(bad_request)?;
    let design = pick_talker(
        hf,
        model,
        |v| v.describes,
        "a speech talker that speaks a described voice",
    )
    .map_err(bad_request)?;
    let _job = state.media_note(&design.id, "speech");
    load_talker(state, &design.id).await?;
    let params = TtsSynthParams {
        voice_description: prompt.clone(),
        seed: seed_of_text(&prompt),
        language: None,
        ..TtsSynthParams::default()
    };
    let spoken = state
        .tts_engine
        .synthesize(script, params)
        .await
        .map_err(|e| server_error(format!("voice design: {e:#}")))?;
    let speaker = embed(state, &base, spoken.pcm, spoken.sample_rate).await?;
    Ok((
        NewVoice {
            name,
            kind: "prompt",
            prompt: Some(prompt),
            consent: None,
            checkpoint: base.id,
        },
        speaker,
    ))
}

/// The speaker embedding of the audio sample of `form`, under its consent.
async fn create_from_sample(
    state: &APIServer,
    form: Form,
    name: String,
    model: Option<&str>,
    hf: &str,
) -> Result<(NewVoice, Vec<f32>), axum::response::Response> {
    let consent = form
        .text("consent")
        .ok_or_else(|| bad_request("`consent` is required: upload a consent recording to /v1/audio/voice_consents and pass its id"))?
        .to_string();
    if VoiceStore::of(hf).consent(&consent).is_none() {
        return Err(not_found(format!("consent '{consent}' not found")));
    }
    let Some((bytes, _, field)) = form.file else {
        return Err(bad_request("`audio_sample` is required: an audio file"));
    };
    let (samples, rate) = decode_clip(&bytes, &field).map_err(bad_request)?;
    let base = pick_talker(
        hf,
        model,
        |v| v.clones && v.encodes,
        "a speech talker that clones",
    )
    .map_err(bad_request)?;
    let _job = state.media_note(&base.id, "speech");
    let speaker = embed(state, &base, samples, rate).await?;
    Ok((
        NewVoice {
            name,
            kind: "audio_sample",
            prompt: None,
            consent: Some(consent),
            checkpoint: base.id,
        },
        speaker,
    ))
}

/// `GET /v1/audio/voices/{voice_id}`.
pub(crate) async fn audio_voice_get(
    axum::extract::State(state): axum::extract::State<APIServer>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    match VoiceStore::of(&state.huggingface_models_dir).voice(&id) {
        Some(v) => Json(voice_object(&v)).into_response(),
        None => not_found(format!("voice '{id}' not found")),
    }
}

/// `DELETE /v1/audio/voices/{voice_id}`: the voice goes, its consent stays.
pub(crate) async fn audio_voice_delete(
    axum::extract::State(state): axum::extract::State<APIServer>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    match VoiceStore::of(&state.huggingface_models_dir).delete_voice(&id) {
        Ok(true) => Json(serde_json::json!({
            "id": id,
            "deleted": true,
            "object": "audio.voice.deleted",
        }))
        .into_response(),
        Ok(false) => not_found(format!("voice '{id}' not found")),
        Err(e) => server_error(format!("voice store: {e}")),
    }
}

/// The extension a consent recording is kept under: `wav` for a WAV file, else the
/// extension of its file name when it is short and alphanumeric, else `audio`.
fn recording_extension(bytes: &[u8], file_name: Option<&str>) -> String {
    if matches!(detect_audio_format(bytes), AudioFormat::Wav) {
        return "wav".to_string();
    }
    file_name
        .and_then(|n| n.rsplit_once('.'))
        .map(|(_, ext)| ext.to_ascii_lowercase())
        .filter(|ext| {
            (1..=4).contains(&ext.len()) && ext.bytes().all(|b| b.is_ascii_alphanumeric())
        })
        .unwrap_or_else(|| "audio".to_string())
}

/// `POST /v1/audio/voice_consents`: multipart `name`, `language` and `recording`.
pub(crate) async fn audio_consent_create(
    axum::extract::State(state): axum::extract::State<APIServer>,
    request: axum::extract::Request,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let form = match read_form(request, "recording").await {
        Ok(f) => f,
        Err(e) => return bad_request(e),
    };
    let checked = bounded(&form, "name", NAME_MAX_CHARS).and_then(|name| {
        let language = form
            .text("language")
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .ok_or_else(|| "`language` is required: a BCP 47 tag".to_string())?;
        Ok((name, language.to_string()))
    });
    let (name, language) = match checked {
        Ok(c) => c,
        Err(e) => return bad_request(e),
    };
    let Some((bytes, file_name, field)) = form.file else {
        return bad_request("`recording` is required: an audio file");
    };
    if let Err(e) = decode_clip(&bytes, &field) {
        return bad_request(e);
    }
    let extension = recording_extension(&bytes, file_name.as_deref());
    match VoiceStore::of(&state.huggingface_models_dir)
        .create_consent(name, language, &bytes, &extension)
    {
        Ok(c) => Json(consent_object(&c)).into_response(),
        Err(e) => server_error(format!("voice store: {e:#}")),
    }
}

/// The query of a list: a cursor and a page size.
#[derive(Debug, serde::Deserialize)]
pub(crate) struct ListQuery {
    after: Option<String>,
    limit: Option<usize>,
}

/// The page of `ids` after the id `after`, of at most `limit` ids, and whether more follow.
pub(super) fn page(
    ids: &[String],
    after: Option<&str>,
    limit: Option<usize>,
) -> Result<(std::ops::Range<usize>, bool), String> {
    let limit = limit.unwrap_or(LIST_DEFAULT);
    if !(1..=LIST_MAX).contains(&limit) {
        return Err(format!("`limit` must be between 1 and {LIST_MAX}"));
    }
    let start = match after {
        Some(a) => ids
            .iter()
            .position(|i| i == a)
            .map(|p| p + 1)
            .ok_or_else(|| format!("`after` names no consent: '{a}'"))?,
        None => 0,
    };
    let end = (start + limit).min(ids.len());
    Ok((start..end, end < ids.len()))
}

/// `GET /v1/audio/voice_consents`, oldest first.
pub(crate) async fn audio_consent_list(
    axum::extract::State(state): axum::extract::State<APIServer>,
    axum::extract::Query(q): axum::extract::Query<ListQuery>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let consents = VoiceStore::of(&state.huggingface_models_dir).consents();
    let ids: Vec<String> = consents.iter().map(|c| c.id.clone()).collect();
    let (range, has_more) = match page(&ids, q.after.as_deref(), q.limit) {
        Ok(p) => p,
        Err(e) => return bad_request(e),
    };
    let data: Vec<serde_json::Value> = consents[range.clone()].iter().map(consent_object).collect();
    Json(serde_json::json!({
        "data": data,
        "has_more": has_more,
        "object": "list",
        "first_id": ids[range.clone()].first(),
        "last_id": ids[range].last(),
    }))
    .into_response()
}

/// `GET /v1/audio/voice_consents/{consent_id}`.
pub(crate) async fn audio_consent_get(
    axum::extract::State(state): axum::extract::State<APIServer>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    match VoiceStore::of(&state.huggingface_models_dir).consent(&id) {
        Some(c) => Json(consent_object(&c)).into_response(),
        None => not_found(format!("consent '{id}' not found")),
    }
}

/// `POST /v1/audio/voice_consents/{consent_id}`: a JSON `name` renames the consent.
pub(crate) async fn audio_consent_update(
    axum::extract::State(state): axum::extract::State<APIServer>,
    axum::extract::Path(id): axum::extract::Path<String>,
    body: Json<serde_json::Value>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let Some(name) = body.0.get("name").and_then(|v| v.as_str()) else {
        return bad_request("`name` is required: a string");
    };
    if let Err(e) = check_length("name", name, NAME_MAX_CHARS) {
        return bad_request(e);
    }
    match VoiceStore::of(&state.huggingface_models_dir).rename_consent(&id, name.to_string()) {
        Ok(Some(c)) => Json(consent_object(&c)).into_response(),
        Ok(None) => not_found(format!("consent '{id}' not found")),
        Err(e) => server_error(format!("voice store: {e:#}")),
    }
}

/// `DELETE /v1/audio/voice_consents/{consent_id}`: the consent and its recording go.
pub(crate) async fn audio_consent_delete(
    axum::extract::State(state): axum::extract::State<APIServer>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if !is_id(&id, CONSENT_PREFIX) {
        return not_found(format!("consent '{id}' not found"));
    }
    match VoiceStore::of(&state.huggingface_models_dir).delete_consent(&id) {
        Ok(true) => Json(serde_json::json!({
            "id": id,
            "deleted": true,
            "object": "audio.voice_consent",
        }))
        .into_response(),
        Ok(false) => not_found(format!("consent '{id}' not found")),
        Err(e) => server_error(format!("voice store: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stored(width: usize) -> StoredVoice {
        StoredVoice {
            id: super::super::voice_store::new_id(super::super::voice_store::VOICE_PREFIX),
            name: "v".to_string(),
            created_at: 1,
            kind: "prompt".to_string(),
            prompt: Some("calm".to_string()),
            consent: None,
            checkpoint: "org/base".to_string(),
            width,
        }
    }

    fn talker(clones: bool, width: usize) -> Talker {
        Talker {
            id: "org/talker".to_string(),
            voice: TalkerVoice {
                clones,
                encodes: clones,
                width,
                describes: !clones,
                names: false,
            },
        }
    }

    #[test]
    fn a_stored_voice_needs_a_talker_that_clones_at_its_width() {
        let v = stored(8);
        assert_eq!(voice_fits(&v, &talker(true, 8)), Ok(()));
        let narrow = voice_fits(&v, &talker(true, 4)).unwrap_err();
        assert!(
            narrow.contains("8 values") && narrow.contains("takes 4"),
            "{narrow}"
        );
        let design = voice_fits(&v, &talker(false, 8)).unwrap_err();
        assert!(design.contains("described in words"), "{design}");
        let mut named = talker(false, 8);
        named.voice.describes = false;
        named.voice.names = true;
        let named = voice_fits(&v, &named).unwrap_err();
        assert!(
            named.contains("its checkpoint names") && named.contains("org/base"),
            "{named}"
        );
    }

    #[test]
    fn the_voice_object_carries_the_reference_fields_only() {
        let v = stored(8);
        let o = voice_object(&v);
        assert_eq!(o["object"], "audio.voice");
        assert_eq!(o["type"], "prompt");
        assert_eq!(o["prompt"], "calm");
        assert!(o.get("checkpoint").is_none() && o.get("width").is_none());
        let mut s = v;
        s.prompt = None;
        assert!(voice_object(&s).get("prompt").is_none());
    }

    #[test]
    fn a_name_and_a_prompt_are_bounded_and_not_blank() {
        let mut form = Form::default();
        assert!(bounded(&form, "name", 4).unwrap_err().contains("required"));
        form.fields.insert("name".to_string(), "   ".to_string());
        assert!(bounded(&form, "name", 4)
            .unwrap_err()
            .contains("whitespace"));
        form.fields.insert("name".to_string(), "abcde".to_string());
        assert!(bounded(&form, "name", 4).unwrap_err().contains("1 to 4"));
        form.fields.insert("name".to_string(), "abcd".to_string());
        assert_eq!(bounded(&form, "name", 4).unwrap(), "abcd");
    }

    #[test]
    fn a_page_follows_its_cursor() {
        let ids: Vec<String> = (0..5).map(|i| format!("cons_{i}")).collect();
        assert_eq!(page(&ids, None, Some(2)).unwrap(), (0..2, true));
        assert_eq!(page(&ids, Some("cons_1"), Some(2)).unwrap(), (2..4, true));
        assert_eq!(page(&ids, Some("cons_3"), None).unwrap(), (4..5, false));
        assert!(page(&ids, Some("cons_9"), None).is_err());
        assert!(page(&ids, None, Some(0)).is_err());
        assert!(page(&ids, None, Some(LIST_MAX + 1)).is_err());
    }

    #[test]
    fn a_clip_too_long_or_not_audio_is_refused() {
        let rate = 8000;
        let long = vec![0f32; rate as usize * (SAMPLE_MAX_SECONDS as usize + 1)];
        let wav = crate::inference::media::audio_io::write_wav(&long, rate);
        assert!(decode_clip(&wav, "audio_sample")
            .unwrap_err()
            .contains("lasts"));
        assert!(decode_clip(b"not audio at all", "audio_sample")
            .unwrap_err()
            .contains("not audio"));
        let short = crate::inference::media::audio_io::write_wav(&[0.25f32; 800], rate);
        let (mono, got) = decode_clip(&short, "audio_sample").unwrap();
        assert_eq!((mono.len(), got), (800, rate));
        assert_eq!(recording_extension(&short, Some("x.mp3")), "wav");
        assert_eq!(recording_extension(b"ID3", Some("c.MP3")), "mp3");
        assert_eq!(recording_extension(b"ID3", Some("c/../x")), "audio");
    }

    /// A store of its own whose `hub` links the 2 published checkpoints of the test store,
    /// `None` when the test store lacks one.
    fn linked_store() -> Option<std::path::PathBuf> {
        let source = crate::config::Config::load_test()
            .get_hf_models_dir()
            .join("hub");
        let dir = crate::inference::generic_transformer::class_load::tests::scratch("voices-api");
        std::fs::create_dir_all(dir.join("hub")).unwrap();
        for repo in [
            "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base",
            "models--Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        ] {
            if !source.join(repo).is_dir() {
                return None;
            }
            std::os::unix::fs::symlink(source.join(repo), dir.join("hub").join(repo)).unwrap();
        }
        Some(dir)
    }

    fn json_request(body: serde_json::Value) -> axum::extract::Request {
        axum::http::Request::builder()
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap()
    }

    /// A multipart request of the text fields `fields` and the WAV file `file` under its
    /// field name.
    fn multipart_request(fields: &[(&str, &str)], file: (&str, &[u8])) -> axum::extract::Request {
        let mut body = Vec::new();
        for (name, value) in fields {
            body.extend_from_slice(
                format!(
                    "--b\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
                )
                .as_bytes(),
            );
        }
        body.extend_from_slice(
            format!(
                "--b\r\nContent-Disposition: form-data; name=\"{}\"; filename=\"clip.wav\"\r\n\
                 Content-Type: audio/x-wav\r\n\r\n",
                file.0
            )
            .as_bytes(),
        );
        body.extend_from_slice(file.1);
        body.extend_from_slice(b"\r\n--b--\r\n");
        axum::http::Request::builder()
            .header("content-type", "multipart/form-data; boundary=b")
            .body(axum::body::Body::from(body))
            .unwrap()
    }

    async fn json_of(response: axum::response::Response) -> (u16, serde_json::Value) {
        let status = response.status().as_u16();
        let body = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap();
        (status, serde_json::from_slice(&body).unwrap_or_default())
    }

    async fn speak(state: &APIServer, input: &str, voice: serde_json::Value) -> (u16, Vec<u8>) {
        let response = audio_speech(
            axum::extract::State(state.clone()),
            axum::http::HeaderMap::new(),
            Json(serde_json::json!({ "input": input, "voice": voice, "seed": 7 })),
        )
        .await;
        let status = response.status().as_u16();
        let body = axum::body::to_bytes(response.into_body(), 1 << 28)
            .await
            .unwrap();
        (status, body.to_vec())
    }

    /// A voice from a description speaks 2 texts by its id.
    /// A voice from a clip is made under a consent recording.
    /// A deleted voice is refused.
    #[tokio::test]
    #[ignore = "needs both published checkpoints in the test store"]
    async fn a_voice_is_created_spoken_and_deleted_through_the_routes() {
        let Some(hf) = linked_store() else {
            eprintln!("a published checkpoint is not in the test store");
            return;
        };
        let state = APIServer::new(
            "/nonexistent-ollama-models".to_string(),
            hf.to_string_lossy().to_string(),
        );
        let made = audio_voices_create(
            axum::extract::State(state.clone()),
            json_request(serde_json::json!({
                "type": "prompt",
                "name": "calm",
                "prompt": "A calm young woman with a warm, clear voice.",
                "script_hint": "Hello. This is how I sound when I read a few sentences aloud.",
            })),
        )
        .await;
        let (status, voice) = json_of(made).await;
        assert_eq!(status, 200, "{voice}");
        let id = voice["id"].as_str().unwrap().to_string();
        assert!(is_id(&id, super::super::voice_store::VOICE_PREFIX), "{id}");
        assert_eq!(voice["type"], "prompt");
        let (status, first) = speak(
            &state,
            "The weather stays mild until the end of the week.",
            serde_json::json!(id),
        )
        .await;
        assert_eq!(status, 200, "{}", String::from_utf8_lossy(&first));
        let (status, second) = speak(
            &state,
            "Please close the door quietly when you leave the room.",
            serde_json::json!({ "id": id }),
        )
        .await;
        assert_eq!(status, 200, "{}", String::from_utf8_lossy(&second));
        for wav in [&first, &second] {
            let (pcm, _, rate) = decode_audio_planar(wav).unwrap();
            let seconds = pcm.len() as f64 / rate as f64;
            assert!((1.0..20.0).contains(&seconds), "{seconds} s");
        }

        let consent = audio_consent_create(
            axum::extract::State(state.clone()),
            multipart_request(
                &[("name", "Jane"), ("language", "en-US")],
                ("recording", &first),
            ),
        )
        .await;
        let (status, consent) = json_of(consent).await;
        assert_eq!(status, 200, "{consent}");
        let consent_id = consent["id"].as_str().unwrap().to_string();
        let cloned = audio_voices_create(
            axum::extract::State(state.clone()),
            multipart_request(
                &[("name", "from a clip"), ("consent", &consent_id)],
                ("audio_sample", &second),
            ),
        )
        .await;
        let (status, cloned) = json_of(cloned).await;
        assert_eq!(status, 200, "{cloned}");
        assert_eq!(cloned["type"], "audio_sample");
        let cloned_id = cloned["id"].as_str().unwrap().to_string();
        let (status, _) = speak(
            &state,
            "Numbers like twelve are easy to say.",
            serde_json::json!(cloned_id),
        )
        .await;
        assert_eq!(status, 200);

        let unknown = audio_voices_create(
            axum::extract::State(state.clone()),
            multipart_request(
                &[
                    ("name", "x"),
                    (
                        "consent",
                        &super::super::voice_store::new_id(CONSENT_PREFIX),
                    ),
                ],
                ("audio_sample", &second),
            ),
        )
        .await;
        assert_eq!(unknown.status().as_u16(), 404);

        let deleted = audio_voice_delete(
            axum::extract::State(state.clone()),
            axum::extract::Path(id.clone()),
        )
        .await;
        let (status, deleted) = json_of(deleted).await;
        assert_eq!((status, deleted["deleted"].as_bool()), (200, Some(true)));
        let (status, refusal) = speak(&state, "Gone.", serde_json::json!(id)).await;
        assert_eq!(status, 404, "{}", String::from_utf8_lossy(&refusal));
        let _ = std::fs::remove_dir_all(&hf);
    }

    /// A voice from a description without `script_hint` is refused with what the field is for.
    /// A blank `script_hint` is refused.
    /// Neither refusal reads a checkpoint.
    #[tokio::test]
    async fn a_voice_from_a_description_needs_the_text_it_speaks() {
        let state = APIServer::new(String::new(), String::new());
        let mut body = serde_json::json!({
            "type": "prompt",
            "name": "calm",
            "prompt": "A calm voice.",
        });
        let missing = audio_voices_create(
            axum::extract::State(state.clone()),
            json_request(body.clone()),
        )
        .await;
        let (status, refusal) = json_of(missing).await;
        assert_eq!(status, 400, "{refusal}");
        let message = refusal["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains("`script_hint` is required") && message.contains("language"),
            "{refusal}"
        );
        body["script_hint"] = serde_json::json!("   ");
        let blank = audio_voices_create(axum::extract::State(state), json_request(body)).await;
        let (status, refusal) = json_of(blank).await;
        assert_eq!(status, 400, "{refusal}");
        assert!(refusal.to_string().contains("whitespace"), "{refusal}");
        let mut form = Form::default();
        form.fields
            .insert("script_hint".to_string(), "Bonjour.".to_string());
        assert_eq!(script_hint(&form).unwrap(), "Bonjour.");
    }

    async fn form_of(request: axum::extract::Request) -> Result<Form, String> {
        read_form(request, "audio_sample").await
    }

    #[tokio::test]
    async fn a_form_is_read_from_json_or_multipart() {
        let json = axum::http::Request::builder()
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"type":"prompt","name":"n","prompt":"p","model":null}"#,
            ))
            .unwrap();
        let form = form_of(json).await.unwrap();
        assert_eq!(form.text("type"), Some("prompt"));
        assert_eq!(form.text("model"), None);
        assert!(form.file.is_none());
        let bad = axum::http::Request::builder()
            .header("content-type", "application/json")
            .body(axum::body::Body::from(r#"{"name":3}"#))
            .unwrap();
        assert!(form_of(bad).await.unwrap_err().contains("`name`"));
        let body = "--b\r\nContent-Disposition: form-data; name=\"name\"\r\n\r\nJane\r\n\
                    --b\r\nContent-Disposition: form-data; name=\"consent\"\r\n\r\ncons_1\r\n\
                    --b\r\nContent-Disposition: form-data; name=\"audio_sample\"; filename=\"a.wav\"\r\n\
                    Content-Type: audio/x-wav\r\n\r\nRIFF\r\n--b--\r\n";
        let multipart = axum::http::Request::builder()
            .header("content-type", "multipart/form-data; boundary=b")
            .body(axum::body::Body::from(body))
            .unwrap();
        let form = form_of(multipart).await.unwrap();
        assert_eq!(form.text("name"), Some("Jane"));
        assert_eq!(form.text("consent"), Some("cons_1"));
        let (bytes, file_name, field) = form.file.unwrap();
        assert_eq!(
            (bytes.as_slice(), file_name.as_deref(), field.as_str()),
            (&b"RIFF"[..], Some("a.wav"), "audio_sample")
        );
    }
}
