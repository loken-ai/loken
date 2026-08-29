//! A projection over a quantised weight, or a dense one.
//!
//! `QMatMul` is what a layer holds when it does not know which it got. The dense arms are not
//! a fallback for failure - a checkpoint may simply carry an unquantised tensor.

use super::*;

/// A projection over a quantised weight, or a dense one.
#[derive(Debug, Clone)]
pub enum QMatMul {
    QTensor(Arc<QTensor>),
    Tensor(Tensor),
    TensorF16(Tensor),
}

impl QMatMul {
    pub fn from_arc(qt: Arc<QTensor>) -> Result<Self> {
        // Mirror the fork: dense float "quantized" tensors dequantize to
        // a plain weight (the QTensor kernel path is for block quants).
        match qt.dtype() {
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => {
                let t = qt.dequantize(qt.device())?;
                Ok(Self::Tensor(t))
            }
            _ => Ok(Self::QTensor(qt)),
        }
    }

    /// Where the weight lives.
    ///
    /// A caller attaching something alongside a projection - an adapter delta - has
    /// to build it on THIS device, and the bias is not a reliable stand-in: a
    /// projection without one would send it to the host and the matmul would then
    /// span two devices.
    pub fn device(&self) -> Device {
        match self {
            Self::QTensor(t) => t.device().clone(),
            Self::Tensor(t) | Self::TensorF16(t) => t.device().clone(),
        }
    }

    /// fork-compat: dequantize to F16 on the weight's device.
    pub fn dequantize_f16(&self) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => t.dequantize_f16(t.device()),
            Self::Tensor(t) => t.to_dtype(crate::tensor::DType::F16),
            Self::TensorF16(t) => Ok(t.clone()),
        }
    }

    pub fn from_qtensor(qt: QTensor) -> Result<Self> {
        Self::from_arc(Arc::new(qt))
    }

    /// `x @ w^T` for a 2-D `[n, k]` dense weight: flatten x's batch into
    /// rows, one transb=T gemm, reshape back, cast to `out_dtype`.
    fn dense_matmul_t(x: &Tensor, w: &Tensor, out_dtype: crate::tensor::DType) -> Result<Tensor> {
        let xdims = x.dims().to_vec();
        let k = *xdims
            .last()
            .ok_or_else(|| Error::msg("QMatMul: rank-0 input"))?;
        let (n, k2) = (w.dims()[0], w.dims()[1]);
        if k != k2 {
            return Err(Error::msg(format!(
                "QMatMul dense: input k {k} != weight k {k2}"
            )));
        }
        let rows = x.elem_count() / k;
        let x2 = x.reshape((rows, k))?;
        let y = x2.matmul_t(w)?;
        let mut odims = xdims;
        *odims.last_mut().unwrap() = n;
        y.reshape(odims)?.to_dtype(out_dtype)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::QTensor(qt) => {
                // `x` is already a concrete tensor::Tensor: the native
                // QMatMul forward handles its own layout/offset (any
                // zero-copy narrow view is encoded in the native tensor).
                Ok(qt.native_qmm()?.forward(x)?)
            }
            // dense fallbacks: w is [n, k] -> x @ w^T.
            // DIRECT transb=T gemm on the untransposed weight - `w.t()`
            // here materialized a full permute of the weight EVERY call
            // (the AWQ F16 lm_head = a ~19 ms 1.5 GB device copy per
            // decoded token, 59% of AWQ GPU time), and matmul_t
            // never read it.
            Self::Tensor(w) => Self::dense_matmul_t(x, w, x.dtype()),
            Self::TensorF16(w) => {
                let xf = x.to_dtype(crate::tensor::DType::F16)?;
                Self::dense_matmul_t(&xf, w, x.dtype())
            }
        }
    }

    /// Fused dense SwiGLU decode `silu(self.x)*(up.x)` for block-quantized
    /// CPU weights + F16 activation (delegates to the native kernel).
    /// `Ok(None)` for any non-fast-path case -> caller runs the unfused path.
    pub fn gate_up_silu_f16(&self, up: &QMatMul, x: &Tensor) -> Result<Option<Tensor>> {
        match (self, up) {
            (Self::QTensor(g), Self::QTensor(u)) => {
                g.native_qmm()?.gate_up_silu_f16(u.native_qmm()?, x)
            }
            _ => Ok(None),
        }
    }

    /// zero-alloc CPU decode projection: delegates to the native
    /// QMatMul slice path for block-quantized weights. `Ok(false)` for dense
    /// (Tensor/F16) weights -> caller falls back to `forward`.
    pub fn forward_slice_cpu(&self, x: &[f32], out: &mut [f32]) -> Result<bool> {
        match self {
            Self::QTensor(qt) => qt.native_qmm()?.forward_slice_cpu(x, out),
            _ => Ok(false),
        }
    }

    /// Several projections against ONE activation, quantised once.
    /// `Ok(false)` and nothing written when any weight is not block-quantized,
    /// so the caller falls back to the per-weight calls.
    pub fn forward_slice_cpu_shared(
        x: &[f32],
        mats: &[&Self],
        outs: &mut [&mut [f32]],
    ) -> Result<bool> {
        let mut inner = Vec::with_capacity(mats.len());
        for m in mats {
            match m {
                Self::QTensor(qt) => inner.push(qt.native_qmm()?),
                _ => return Ok(false),
            }
        }
        QKernelMatMul::forward_slice_cpu_shared(x, &inner, outs)
    }

    /// Output dim (`n`) - to size decode-arena buffers.
    pub fn out_dim(&self) -> Result<usize> {
        match self {
            Self::QTensor(qt) => Ok(qt.native_qmm()?.out_dim()),
            Self::Tensor(w) | Self::TensorF16(w) => Ok(w.dims()[0]),
        }
    }

    /// Dense BF16 copy of the weight, dequantized ONCE (see
    /// [`QKernelMatMul::dequant_weight_bf16`]). `Ok(None)` for dense/CPU weights.
    #[cfg(feature = "cuda")]
    pub fn dequant_weight_bf16(&self) -> Result<Option<Tensor>> {
        match self {
            Self::QTensor(qt) => qt.native_qmm()?.dequant_weight_bf16(),
            _ => Ok(None),
        }
    }

    /// Raw CPU quantized weight `(dtype, k, n, bytes)` for fused multi-weight
    /// regions (MoE flat `matmul_bytes_multi`). Forwards to the kernel matmul
    /// (the borrow lives as long as `self` - `native_qmm` caches a reference).
    pub fn cpu_raw(&self) -> Option<(GgmlDType, usize, usize, &[u8])> {
        match self {
            Self::QTensor(qt) => qt.native_qmm().ok()?.cpu_raw(),
            _ => None,
        }
    }
}

impl crate::tensor::Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> tensor::Result<Tensor> {
        QMatMul::forward(self, xs)
    }
}
