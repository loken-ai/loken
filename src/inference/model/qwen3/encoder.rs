//! The standard Qwen3 pre-norm encoder block, and the GGUF loading it needs.
//!
//! One layer is: `input_layernorm` -> GQA self-attention with per-head q/k RMS norms
//! and RoPE -> residual -> `post_attention_layernorm` -> SwiGLU -> residual. No
//! adaptive-layernorm conditioning and no causal mask baked in - the caller decides
//! whether attention is bidirectional or masked, which is the only difference between
//! the encoders built on it.
//!
//! Several stacks share this block: a text encoder producing hidden states for a
//! conditioner, an embedding model pooling them into a vector, an audio
//! de-tokenizer turning quantized codes back into latents. It is written once here
//! so those stay one implementation rather than three that drift.

use crate::tensor::{Device, Tensor};

/// The substrate's, re-exported. This copy multiplied with `matmul_t`, which is a cuBLAS
/// N-T flag rather than a materialised transpose - the shared type stores the transpose once
/// at construction and reaches the same GEMM, and hands back the published orientation for
/// the callers below that want it.
pub use crate::tensor::layer::Linear;

/// One pre-norm block: two norms, the four attention projections with their q/k
/// norms, and the three SwiGLU projections.
pub struct Layer {
    pub input_ln: Tensor,
    pub post_ln: Tensor,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

/// RoPE cos/sin tables in NEOX layout: `cos[p,j] = cos(p.θ^(-2j/D))`.
pub(crate) fn rope_tables(s: usize, d: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = d / 2;
    let (mut cos, mut sin) = (vec![0f32; s * half], vec![0f32; s * half]);
    for p in 0..s {
        for j in 0..half {
            let a = (p as f32) * theta.powf(-2.0 * (j as f32) / (d as f32));
            cos[p * half + j] = a.cos();
            sin[p * half + j] = a.sin();
        }
    }
    (cos, sin)
}

/// Load one GGUF tensor as a dense F32 tensor on `device`.
///
/// The GGUF reader is the compatibility stack and runs once, at load; the values are
/// rebuilt as a native tensor so the forward path stays entirely on the native stack.
pub(crate) fn load_t(
    c: &crate::tensor::quantized::gguf_file::Content,
    f: &mut std::fs::File,
    name: &str,
    device: &Device,
) -> crate::tensor::Result<Tensor> {
    use crate::tensor::{DType as CDType, Device as CDevice};
    let dq = c
        .tensor(f, name, &CDevice::Cpu)?
        .dequantize(&CDevice::Cpu)?
        .to_dtype(CDType::F32)?;
    let dims = dq.dims().to_vec();
    let v = dq.flatten_all()?.to_vec1::<f32>()?;
    Ok(Tensor::from_vec_f32(v, dims)?.to_device(device)?)
}

/// Load a linear from `<prefix>.weight` and, when asked for, `<prefix>.bias`.
pub(crate) fn lin(
    c: &crate::tensor::quantized::gguf_file::Content,
    f: &mut std::fs::File,
    prefix: &str,
    bias: bool,
    device: &Device,
) -> crate::tensor::Result<Linear> {
    let w = load_t(c, f, &format!("{prefix}.weight"), device)?;
    let b = if bias {
        Some(load_t(c, f, &format!("{prefix}.bias"), device)?)
    } else {
        None
    };
    Linear::new(w, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_tables_are_the_neox_angles() {
        let (cos, sin) = rope_tables(4, 8, 10000.0);
        assert_eq!(cos.len(), 4 * 4);
        // Position 0 has angle 0 in every pair, so the rotation is the identity.
        assert!(cos[..4].iter().all(|&c| (c - 1.0).abs() < 1e-6));
        assert!(sin[..4].iter().all(|&s| s.abs() < 1e-6));
        // Pair j at position p is the angle p.theta^(-2j/D), and cos/sin agree with it.
        for p in 0..4 {
            for j in 0..4 {
                let a = (p as f32) * 10000f32.powf(-2.0 * (j as f32) / 8.0);
                assert!((cos[p * 4 + j] - a.cos()).abs() < 1e-6);
                assert!((sin[p * 4 + j] - a.sin()).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn a_linear_applies_its_bias() {
        let w = Tensor::from_vec_f32(vec![1.0, 0.0, 0.0, 1.0], (2, 2)).unwrap();
        let x = Tensor::from_vec_f32(vec![3.0, 5.0], (1, 2)).unwrap();
        let no_bias = Linear::new(w.clone(), None).unwrap();
        assert_eq!(no_bias.forward(&x).unwrap().to_vec_f32(), vec![3.0, 5.0]);
        let b = Tensor::from_vec_f32(vec![10.0, 20.0], 2).unwrap();
        let with_bias = Linear::new(w, Some(b)).unwrap();
        assert_eq!(
            with_bias.forward(&x).unwrap().to_vec_f32(),
            vec![13.0, 25.0]
        );
    }
}
