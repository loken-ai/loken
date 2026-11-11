//! The custom voices a client creates and the consent recordings they rest on, kept on disk.
//!
//! Every voice and every consent is a directory of its own under one `voices` directory.
//! A voice directory holds its metadata and its speaker embedding.
//! A consent directory holds its metadata and its recording.
//! A directory is written under a staging name and renamed into place, so a reader sees a
//! whole entry or none.

use std::path::{Path, PathBuf};

/// The prefix of a voice id.
pub(crate) const VOICE_PREFIX: &str = "voice_";
/// The prefix of a consent id.
pub(crate) const CONSENT_PREFIX: &str = "cons_";
/// The hexadecimal digits after the prefix of an id: 128 random bits.
const ID_DIGITS: usize = 32;
/// The metadata file of a voice directory.
const VOICE_META: &str = "voice.json";
/// The speaker embedding file of a voice directory.
const VOICE_SPEAKER: &str = "speaker.safetensors";
/// The name of the speaker embedding tensor.
const SPEAKER_TENSOR: &str = "speaker";
/// The metadata file of a consent directory.
const CONSENT_META: &str = "consent.json";
/// The prefix of a directory being written.
const STAGING: &str = ".staging-";

/// A stored voice.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct StoredVoice {
    pub id: String,
    pub name: String,
    /// Unix seconds.
    pub created_at: u64,
    /// How the voice was made: `prompt` or `audio_sample`.
    #[serde(rename = "type")]
    pub kind: String,
    /// The description a `prompt` voice was made from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// The consent an `audio_sample` voice was made under.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub consent: Option<String>,
    /// The catalogue id of the checkpoint whose speaker encoder made the embedding.
    pub checkpoint: String,
    /// The number of values of the speaker embedding.
    pub width: usize,
}

/// A stored consent recording.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct StoredConsent {
    pub id: String,
    pub name: String,
    pub language: String,
    /// Unix seconds.
    pub created_at: u64,
    /// The file name of the recording inside the consent directory.
    pub recording: String,
}

/// What a voice is made from, before it has an id.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NewVoice {
    pub name: String,
    pub kind: &'static str,
    pub prompt: Option<String>,
    pub consent: Option<String>,
    pub checkpoint: String,
}

/// The `voices` directory and what it holds.
#[derive(Debug, Clone)]
pub(crate) struct VoiceStore {
    dir: PathBuf,
}

/// A fresh id: `prefix` and 32 lowercase hexadecimal digits.
pub(crate) fn new_id(prefix: &str) -> String {
    format!("{prefix}{:032x}", rand::random::<u128>())
}

/// Whether `id` is `prefix` followed by 32 lowercase hexadecimal digits.
///
/// Only such an id names a directory, so no request reaches a path outside the store.
pub(crate) fn is_id(id: &str, prefix: &str) -> bool {
    id.strip_prefix(prefix).is_some_and(|hex| {
        hex.len() == ID_DIGITS
            && hex
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Option<T> {
    serde_json::from_slice(&std::fs::read(path).ok()?).ok()
}

impl VoiceStore {
    /// The store under the root of the Hugging Face store `hf_models_dir` names: its parent
    /// when it names the `hub` directory, else itself.
    pub(crate) fn of(hf_models_dir: &str) -> Self {
        let base = PathBuf::from(hf_models_dir);
        let root = if base.file_name().is_some_and(|n| n == "hub") {
            base.parent().map(Path::to_path_buf).unwrap_or(base)
        } else {
            base
        };
        Self {
            dir: root.join("voices"),
        }
    }

    /// The store at `dir`.
    #[cfg(test)]
    pub(crate) fn at(dir: PathBuf) -> Self {
        Self { dir }
    }

    /// The directory of the store.
    pub(crate) fn dir(&self) -> &Path {
        &self.dir
    }

    /// Write the files `files` into the directory `id`, whole or not at all.
    fn write_entry(&self, id: &str, files: &[(&str, &[u8])]) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let staging = self.dir.join(format!("{STAGING}{id}"));
        let _ = std::fs::remove_dir_all(&staging);
        std::fs::create_dir(&staging)?;
        let written = files
            .iter()
            .try_for_each(|(name, bytes)| std::fs::write(staging.join(name), bytes))
            .and_then(|()| std::fs::rename(&staging, self.dir.join(id)));
        if written.is_err() {
            let _ = std::fs::remove_dir_all(&staging);
        }
        written
    }

    /// Replace the file `name` of the directory `id` by `bytes`, whole or not at all.
    fn rewrite(&self, id: &str, name: &str, bytes: &[u8]) -> std::io::Result<()> {
        let dir = self.dir.join(id);
        let staging = dir.join(format!("{STAGING}{name}"));
        std::fs::write(&staging, bytes)?;
        std::fs::rename(&staging, dir.join(name))
    }

    /// Every entry whose id has `prefix` and whose metadata file `meta` reads, oldest
    /// first.
    fn entries<T: serde::de::DeserializeOwned>(
        &self,
        prefix: &str,
        meta: &str,
        created: impl Fn(&T) -> (u64, String),
    ) -> Vec<T> {
        let Ok(dir) = std::fs::read_dir(&self.dir) else {
            return Vec::new();
        };
        let mut out: Vec<T> = dir
            .flatten()
            .filter(|e| is_id(&e.file_name().to_string_lossy(), prefix))
            .filter_map(|e| read_json(&e.path().join(meta)))
            .collect();
        out.sort_by_key(|e| created(e));
        out
    }

    /// Delete the directory `id`, `false` when there is none.
    fn delete(&self, id: &str, prefix: &str) -> std::io::Result<bool> {
        if !is_id(id, prefix) || !self.dir.join(id).is_dir() {
            return Ok(false);
        }
        std::fs::remove_dir_all(self.dir.join(id))?;
        Ok(true)
    }

    /// Store the voice `voice` with the speaker embedding `speaker`, under a fresh id.
    pub(crate) fn create_voice(
        &self,
        voice: NewVoice,
        speaker: &[f32],
    ) -> anyhow::Result<StoredVoice> {
        let stored = StoredVoice {
            id: new_id(VOICE_PREFIX),
            name: voice.name,
            created_at: now(),
            kind: voice.kind.to_string(),
            prompt: voice.prompt,
            consent: voice.consent,
            checkpoint: voice.checkpoint,
            width: speaker.len(),
        };
        let bytes: Vec<u8> = speaker.iter().flat_map(|x| x.to_le_bytes()).collect();
        let view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            vec![speaker.len()],
            &bytes,
        )?;
        let tensors = safetensors::serialize([(SPEAKER_TENSOR, view)], None)?;
        let meta = serde_json::to_vec_pretty(&stored)?;
        self.write_entry(
            &stored.id,
            &[(VOICE_META, &meta), (VOICE_SPEAKER, &tensors)],
        )?;
        Ok(stored)
    }

    /// The voice `id`, `None` when the store holds none.
    pub(crate) fn voice(&self, id: &str) -> Option<StoredVoice> {
        is_id(id, VOICE_PREFIX)
            .then(|| read_json(&self.dir.join(id).join(VOICE_META)))
            .flatten()
    }

    /// The speaker embedding of the voice `id`.
    pub(crate) fn speaker(&self, id: &str) -> anyhow::Result<Vec<f32>> {
        anyhow::ensure!(is_id(id, VOICE_PREFIX), "'{id}' is not a voice id");
        let bytes = std::fs::read(self.dir.join(id).join(VOICE_SPEAKER))?;
        let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
        let view = tensors.tensor(SPEAKER_TENSOR)?;
        anyhow::ensure!(
            view.dtype() == safetensors::Dtype::F32,
            "the speaker embedding of '{id}' is not F32"
        );
        Ok(view
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    /// Every stored voice, oldest first.
    pub(crate) fn voices(&self) -> Vec<StoredVoice> {
        self.entries(VOICE_PREFIX, VOICE_META, |v: &StoredVoice| {
            (v.created_at, v.id.clone())
        })
    }

    /// Delete the voice `id`, `false` when the store holds none.
    ///
    /// Its consent stays.
    pub(crate) fn delete_voice(&self, id: &str) -> std::io::Result<bool> {
        self.delete(id, VOICE_PREFIX)
    }

    /// Store the consent recording `recording`, a file of extension `extension`, under a
    /// fresh id.
    pub(crate) fn create_consent(
        &self,
        name: String,
        language: String,
        recording: &[u8],
        extension: &str,
    ) -> anyhow::Result<StoredConsent> {
        let stored = StoredConsent {
            id: new_id(CONSENT_PREFIX),
            name,
            language,
            created_at: now(),
            recording: format!("recording.{extension}"),
        };
        let meta = serde_json::to_vec_pretty(&stored)?;
        self.write_entry(
            &stored.id,
            &[
                (CONSENT_META, &meta),
                (stored.recording.as_str(), recording),
            ],
        )?;
        Ok(stored)
    }

    /// The consent `id`, `None` when the store holds none.
    pub(crate) fn consent(&self, id: &str) -> Option<StoredConsent> {
        is_id(id, CONSENT_PREFIX)
            .then(|| read_json(&self.dir.join(id).join(CONSENT_META)))
            .flatten()
    }

    /// Every stored consent, oldest first.
    pub(crate) fn consents(&self) -> Vec<StoredConsent> {
        self.entries(CONSENT_PREFIX, CONSENT_META, |c: &StoredConsent| {
            (c.created_at, c.id.clone())
        })
    }

    /// Give the consent `id` the name `name`, `None` when the store holds none.
    pub(crate) fn rename_consent(
        &self,
        id: &str,
        name: String,
    ) -> anyhow::Result<Option<StoredConsent>> {
        let Some(mut consent) = self.consent(id) else {
            return Ok(None);
        };
        consent.name = name;
        self.rewrite(id, CONSENT_META, &serde_json::to_vec_pretty(&consent)?)?;
        Ok(Some(consent))
    }

    /// Delete the consent `id` and its recording, `false` when the store holds none.
    pub(crate) fn delete_consent(&self, id: &str) -> std::io::Result<bool> {
        self.delete(id, CONSENT_PREFIX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(name: &str) -> VoiceStore {
        let dir = crate::inference::generic_transformer::class_load::tests::scratch(name);
        let _ = std::fs::remove_dir_all(&dir);
        VoiceStore::at(dir)
    }

    fn new_voice() -> NewVoice {
        NewVoice {
            name: "calm".to_string(),
            kind: "prompt",
            prompt: Some("A calm voice.".to_string()),
            consent: None,
            checkpoint: "org/talker".to_string(),
        }
    }

    #[test]
    fn an_id_is_its_prefix_and_32_hexadecimal_digits() {
        let id = new_id(VOICE_PREFIX);
        assert!(is_id(&id, VOICE_PREFIX), "{id}");
        assert!(!is_id(&id, CONSENT_PREFIX));
        assert_ne!(id, new_id(VOICE_PREFIX));
        assert!(!is_id("voice_../../etc", VOICE_PREFIX));
        assert!(!is_id(&format!("{id}0"), VOICE_PREFIX));
        assert!(!is_id(&id.to_uppercase(), VOICE_PREFIX));
    }

    #[test]
    fn a_voice_is_created_listed_read_and_deleted() {
        let store = store("voice-store-roundtrip");
        let speaker: Vec<f32> = (0..8).map(|i| i as f32 * 0.5 - 1.0).collect();
        let made = store.create_voice(new_voice(), &speaker).unwrap();
        assert_eq!(made.width, 8);
        assert_eq!(store.voice(&made.id), Some(made.clone()));
        assert_eq!(store.speaker(&made.id).unwrap(), speaker);
        let second = store.create_voice(new_voice(), &speaker).unwrap();
        let ids: Vec<String> = store.voices().into_iter().map(|v| v.id).collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&made.id) && ids.contains(&second.id));
        assert!(store.delete_voice(&made.id).unwrap());
        assert!(!store.delete_voice(&made.id).unwrap());
        assert_eq!(store.voice(&made.id), None);
        assert_eq!(store.voices().len(), 1);
        let _ = std::fs::remove_dir_all(store.dir());
    }

    #[test]
    fn an_unknown_or_malformed_id_names_nothing() {
        let store = store("voice-store-unknown");
        assert_eq!(store.voice(&new_id(VOICE_PREFIX)), None);
        assert_eq!(store.voice("voice_../x"), None);
        assert!(store.speaker("alloy").is_err());
        assert!(!store.delete_voice("voice_1234").unwrap());
        assert_eq!(store.consent(&new_id(CONSENT_PREFIX)), None);
        assert!(store.voices().is_empty());
    }

    #[test]
    fn a_staged_entry_is_not_listed() {
        let store = store("voice-store-staging");
        let id = new_id(VOICE_PREFIX);
        let staging = store.dir().join(format!("{STAGING}{id}"));
        std::fs::create_dir_all(&staging).unwrap();
        std::fs::write(
            staging.join(VOICE_META),
            serde_json::to_vec(&StoredVoice {
                id: id.clone(),
                name: "half".to_string(),
                created_at: 0,
                kind: "prompt".to_string(),
                prompt: None,
                consent: None,
                checkpoint: "c".to_string(),
                width: 1,
            })
            .unwrap(),
        )
        .unwrap();
        assert!(store.voices().is_empty());
        assert_eq!(store.voice(&id), None);
        let made = store.create_voice(new_voice(), &[1.0]).unwrap();
        assert!(!store.dir().join(format!("{STAGING}{}", made.id)).exists());
        let _ = std::fs::remove_dir_all(store.dir());
    }

    #[test]
    fn a_consent_is_created_renamed_and_deleted_with_its_recording() {
        let store = store("voice-store-consent");
        let c = store
            .create_consent("Jane".to_string(), "en-US".to_string(), b"RIFF", "wav")
            .unwrap();
        assert!(is_id(&c.id, CONSENT_PREFIX));
        assert_eq!(
            std::fs::read(store.dir().join(&c.id).join(&c.recording)).unwrap(),
            b"RIFF"
        );
        let renamed = store
            .rename_consent(&c.id, "Jane D".to_string())
            .unwrap()
            .unwrap();
        assert_eq!(renamed.name, "Jane D");
        assert_eq!(store.consent(&c.id), Some(renamed));
        assert_eq!(store.consents().len(), 1);
        assert!(store.voices().is_empty());
        assert!(store.delete_consent(&c.id).unwrap());
        assert!(!store.dir().join(&c.id).exists());
        assert_eq!(store.rename_consent(&c.id, "x".to_string()).unwrap(), None);
        let _ = std::fs::remove_dir_all(store.dir());
    }

    #[test]
    fn the_store_sits_beside_the_hub_directory() {
        assert_eq!(VoiceStore::of("/s/hf/hub").dir(), Path::new("/s/hf/voices"));
        assert_eq!(VoiceStore::of("/s/hf").dir(), Path::new("/s/hf/voices"));
    }
}
