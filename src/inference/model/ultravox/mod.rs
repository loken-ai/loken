//! The Ultravox speech tower: a Whisper-large-v3-turbo encoder and a gated projector that
//! maps its frames into the Llama embedding space.
//!
//! Read by [`crate::inference::model::audio_in::AudioTower`], from a checkpoint directory
//! (`audio_tower.*` and `multi_modal_projector.*` tensors, the configuration in
//! `config.json`) or from an `mmproj` GGUF (`clip.projector_type = ultravox`). The encoder
//! is the Voxtral one. The projector stacks `stack_factor` frames (the last group padded
//! with zeros), then:
//!
//! ```text
//! RMSNorm -> linear (no bias) -> SwiGLU (value, gate = halves; silu(gate) * value)
//!   -> RMSNorm -> linear (no bias) -> [ceil(F / S), text width]
//! ```
//!
//! The clip is padded to a whole hop only, windows of 3000 frames are encoded alone and the
//! last one is as short as its audio.
//!
//! The parity gate holds the tower against the reference implementation at toy size.

#[cfg(test)]
mod parity;
