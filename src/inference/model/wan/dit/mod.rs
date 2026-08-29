//! Wan 2.1 video - WanModel DiT (full-Rust, `crate::tensor`), Stage 3.
//!
//! The DiT is the spatiotemporal diffusion transformer that predicts the flow-match
//! velocity for a latent video `[16, F, H, W]`, conditioned on a sinusoidal timestep
//! and the umT5 text context `[S, 4096]`. Source of truth for the architecture is the
//! reference `github.com/Wan-Video/Wan2.1` `wan/modules/model.py` (t2v 1.3B):
//! dim 1536, 30 layers, 12 heads, head_dim 128, ffn 8960, eps 1e-6, in/out channels 16,
//! text_dim 4096, freq_dim 256, patch (1,2,2).
//!
//! ## Variants (identical architecture, different config + weight format)
//! The SAME forward runs both checkpoints - only `dim`/`n_heads`/`n_layers` and the linear
//! op type change (see [`WanVariant`]):
//!  * **1.3B** (default): F32 safetensors -> F16 device weights, packs onto ONE GPU.
//!  * **14B**: dim 5120, 40 blocks, 40 heads, ffn 13824. Every weight MATRIX is Q8_0 in the
//!    GGUF -> kept quantized on device (`QMatMul`, dequant-on-the-fly, same as the ACE-Step
//!    DiT); every bias/norm/modulation is the separate F32 GGUF tensor. ~16 GB Q8 -> the 40
//!    blocks are split across GPUs by the unified HeteroPlan (per-block device placement),
//!    exactly like the ACE-Step XL DiT. The inter-block activation is the only tensor that
//!    crosses devices.
//!
//! Net-new vs our 1D/2D DiTs (ACE-Step / EZAudio):
//!  * conv3d patch embed - kernel/stride (1,2,2): temporal kernel 1 ⟹ a per-frame 2x2/2
//!    spatial `conv2d`, then the (F,H/2,W/2) grid is flattened (w fastest) to a token seq.
//!  * 3D-RoPE - head_dim 128 -> 64 complex pairs split 22/21/21 across the (temporal,
//!    height, width) axes; each token's pair `j` is rotated by its grid coordinate on
//!    that axis. INTERLEAVED pairing `(2j, 2j+1)` (the reference's `view_as_complex`),
//!    so we drive the substrate's `rope_i` with per-token cos/sin tables.
//!  * full dense 3D self-attention over all F.H/2.W/2 tokens (no causal/window mask).
//!  * AdaLN-Zero: the time projection (6.dim) is ADDED to each block's learned
//!    `modulation [1,6,dim]`, then split into (shift,scale,gate)x(self-attn,ffn); the
//!    self-attn and ffn branches are gated, the cross-attn branch is a plain residual.
//!  * qk-RMSNorm on q,k (both self- and cross-attn). NOTE: the checkpoint's
//!    `norm_q/norm_k.weight` are `[dim]`, so the RMSNorm is over the FULL projection (all
//!    heads jointly) BEFORE the head reshape - not a per-head-dim norm.
//!
//! 1.3B weights load from `diffusion_pytorch_model.safetensors` (F32 on disk) kept resident
//! as F16; the forward computes in F32 (each F16/Q8 weight is up-converted per op), which
//! keeps every substrate kernel on its robust F32 path regardless of CPU/CUDA placement. The
//! flow-match Euler + CFG loop is Stage 4 (not here): this module loads + runs ONE forward.

use crate::tensor::layer::qlinear::{QLinear, Weight};
use crate::tensor::layer::Linear;

use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan};
use crate::tensor::quantized::QVarBuilder;
use crate::tensor::safetensors_io::SafeTensorsLoader;
use crate::tensor::{DType, Device, Result, Tensor};

/// Which Wan 2.1 T2V checkpoint to load. The architecture is identical; only the config
/// constants and the weight format/placement differ. Default is [`WanVariant::B1_3`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WanVariant {
    /// 1.3B (dim 1536, 30 blocks, 12 heads): F32 safetensors -> F16 device weights, packs
    /// onto one GPU. The backward-compatible default.
    B1_3,
    /// 14B (dim 5120, 40 blocks, 40 heads, ffn 13824): Q8_0 GGUF -> QMatMul on-device
    /// weights, multi-GPU layer split. Selected via `WAN_MODEL=14b` (wan_render).
    B14,
}

mod config;
pub use config::*;
mod layers;
pub use layers::*;
