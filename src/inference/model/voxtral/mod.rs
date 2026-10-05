//! The Voxtral speech tower: a Whisper-large-v3 encoder and a two-layer GELU projector that
//! maps its frames into the Ministral embedding space.
//!
//! Read by [`crate::inference::model::audio_in::AudioTower`], from the llama.cpp
//! `mmproj-*.gguf` (`clip.projector_type = voxtral`) or from the checkpoint's
//! `audio_tower.*` and `multi_modal_projector.*` tensors. Layout:
//!
//! ```text
//! mel [1, n_mels, T] -> conv(k3, s1) + gelu -> conv(k3, s2) + gelu -> + positions
//!   -> N x [LayerNorm, MHA, +, LayerNorm, FFN(gelu), +] -> LayerNorm -> [T/2, width]
//! stack S frames -> [T/2S, S x width] -> linear -> gelu -> linear -> [T/2S, projection]
//! ```
//!
//! The clip is padded to whole 30 s windows and each window is encoded alone, with its
//! positions starting at zero; a window yields `1500 / S` embeddings.
//!
//! The parity gate holds the tower against the reference implementation at toy size.

#[cfg(all(test, feature = "audio"))]
mod parity;
