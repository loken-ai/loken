// Native tensor ops, our own replacement for candle_nn.
// Native weight loader (VarBuilder + builders + Conv1d) for
// the image/audio/vision inference models. Safetensors-backed; the training
// path (VarMap/autograd) is deliberately not covered.
#[cfg(feature = "gpu")]
pub mod moe_cuda;
#[cfg(feature = "gpu")]
pub mod quantized_cuda;
// CPU expert-GEMM fallback, ALSO compiled into the CUDA build so the single
// binary serves MoE arches on CPU (`serve --cpu` / `--num-gpu 0`). The cuda
// moe_cuda expert kernels reject CPU tensors (`as_cuda_device`), so lfm2/nemotron
// MoE branch to `moe_cpu` at their call sites when the activation is on CPU.
#[cfg(feature = "gpu")]
#[path = "moe_cuda_cpu/mod.rs"]
pub mod moe_cpu;
#[cfg(not(feature = "gpu"))]
#[path = "moe_cuda_cpu/mod.rs"]
pub mod moe_cuda;
#[cfg(not(feature = "gpu"))]
pub use moe_cuda as moe_cpu;

pub mod engine;
pub mod generic_transformer;
pub mod media_facts;
pub mod model_facts;

// Re-export engine submodules directly
pub use engine::llm_engine;

pub use engine::{InferenceEngine, LlmEngine};
pub mod cache;
pub mod codec;
pub mod kernel;
pub mod load;
pub mod media;
pub mod model;
pub mod offload;
pub mod place;
pub mod sample;
pub mod serve;
pub mod token;
