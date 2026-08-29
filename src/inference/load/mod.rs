//! Getting weights off disk: the quantised formats, the adapters, and the
//! managers that resolve a model name to files.
//!
//! Callers name a module through this directory - `crate::inference::load::<module>` - so the path says which
//! part of the system a file belongs to, which is the whole reason the directory exists.
pub mod awq;
pub mod awq_loader;
pub mod fp8_scaled;
pub mod huggingface_manager;
pub mod model_manager;
pub mod ollama_manager;
pub mod onnx;
#[cfg(feature = "image")]
pub mod onnx_exec;
// Low-rank adapters: reading them off disk is a loading concern, and applying them is
// the projection's, so the type lives with the layers and the reader lives here.
pub mod lora;
