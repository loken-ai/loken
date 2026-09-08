//! One directory per model FAMILY, and nothing else.
//!
//! A family with a single part still gets its directory: how many files a model needs
//! today is not something a reader should have to know, and the answer changes. What used
//! to sit here and is not a model - an STFT, a GIF encoder, a tokeniser, an ODE solver -
//! moved to codec/, media/, token/ and sample/, because every family draws on those.

pub mod acestep;
/// The attention every encoder-decoder in the tree runs.
pub mod attention;
pub mod block;
pub mod boogu;
pub mod clip;
pub mod embedding;
pub mod ezaudio;
/// What every flow-matching sampler does to its time axis.
pub mod flow_match;
pub mod flux;
#[cfg(feature = "image")]
pub mod flux2;
pub mod gptoss;
pub mod kyutai;
pub mod lfm2_moe;
pub mod mixformer;
pub mod moondream;
pub mod nemotron_h;
#[cfg(feature = "audio")]
pub mod parler;
/// Cutting an image into patches, and putting it back.
pub mod patches;
#[cfg(feature = "audio")]
pub mod piper;
pub mod pixtral;
#[cfg(feature = "audio")]
pub mod pocket_tts;
pub mod qwen25;
pub mod qwen3;
pub mod qwen35;
pub mod qwen3vl;
pub mod qwen_image;
pub mod reranker;
pub mod rope;
pub mod sdxl;
#[cfg(feature = "audio")]
pub mod stable_audio;
pub mod t5;
#[cfg(feature = "audio")]
pub mod ultravox;
pub mod umt5;
/// The blocks every 2-D autoencoder in the tree shares.
pub mod vae_blocks;
/// The block every vision tower in the tree is made of.
pub mod vit;
pub mod voxtral;
pub mod wan;
#[cfg(feature = "audio")]
pub mod whisper;
pub mod zimage;
