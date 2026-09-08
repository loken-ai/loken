//! Audio API family: transcription/translation/speech/voices,
//! TTS + ASR helpers, wav/pcm/symphonia/resample codecs, and the
//! /v1/conversation + /v1/voice orchestration endpoints (they share the
//! intent router and conv_ok/conv_err response helpers with /v1/voice).

use super::*;

// ------------------------------------------------------------
// Unified multimodal conversation endpoint (Stage A: rule-based routing + context
// bridge). One /conversation call inspects the latest user turn and dispatches to
// the right EXISTING engine (chat / vision / image-gen / TTS), returning the
// assistant turn to append - so a caller keeps one continuous multimodal history
// without choosing a model/endpoint per turn. Reuses the multi-model residency
// registry (auto-load + LRU-evict) and the separate media engines.
// ------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConvRoute {
    Chat,
    Vision,
    ImageGen,
    Tts,
    SoundGen,
}

mod conversation;
pub(crate) use conversation::*;
mod voice;
pub(crate) use voice::*;
mod whisper;
pub(crate) use whisper::*;
mod wav;
pub(crate) use wav::*;
mod transcribe;
use transcribe::*;
mod decode;
pub use decode::*;

#[cfg(test)]
mod tests;
