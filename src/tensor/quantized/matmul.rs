//! A projection over a quantised weight, or a dense one.
//!
//! `QMatMul` is what a layer holds when it does not know which it got. The dense arms are not
//! a fallback for failure - a checkpoint may simply carry an unquantised tensor.

use super::*;

/// Whether a product of an F16 or BF16 matrix over several rows on a card may read the rows in
/// the matrix's format.
/// The server sets it from its fidelity level at start and before each load.
/// It allows it until set.
static HALF_ROWS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Say whether a product of an F16 or BF16 matrix over several rows on a card may read the rows
/// in the matrix's format.
pub fn set_half_rows(allowed: bool) {
    HALF_ROWS.store(allowed, std::sync::atomic::Ordering::Relaxed);
}

fn half_rows() -> bool {
    HALF_ROWS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether `qt` is an F16 or BF16 weight on the host whose blocks are a view of a file mapping.
/// Its products read the mapping in place, with the numerics of a half weight held as stored.
fn host_half_view(qt: &QTensor) -> bool {
    qt.device().is_cpu()
        && matches!(qt.dtype(), GgmlDType::F16 | GgmlDType::BF16)
        && qt.native_qtensor().is_view()
}

/// The dense dtype of a half weight format.
fn half_dtype(dtype: GgmlDType) -> crate::tensor::DType {
    match dtype {
        GgmlDType::F16 => crate::tensor::DType::F16,
        _ => crate::tensor::DType::BF16,
    }
}

/// A projection over a quantised weight, or a dense one.
#[derive(Debug, Clone)]
pub enum QMatMul {
    QTensor(Arc<QTensor>),
    /// An F32 weight.
    Tensor(Tensor),
    /// An F16 weight held as stored, on the host or a card.
    TensorF16(Tensor),
    /// A BF16 weight held as stored, on the host or a card.
    Bf16(Tensor),
}

impl QMatMul {
    /// A float weight is held in its stored format on the device it was read to; a block
    /// format keeps its blocks for the quantised kernels.
    pub fn from_arc(qt: Arc<QTensor>) -> Result<Self> {
        let dev = qt.device().clone();
        // A half weight on the host that is a view of the file is read in place.
        if host_half_view(&qt) {
            return Ok(Self::QTensor(qt));
        }
        match qt.dtype() {
            GgmlDType::BF16 => Self::bf16_onto(&qt, &dev),
            GgmlDType::F16 => Self::f16_onto(&qt, &dev),
            // CONVERSION: none of value: the decode of F32 blocks is the F32 tensor itself.
            GgmlDType::F32 => Ok(Self::Tensor(qt.dequantize(&dev)?)),
            _ => Ok(Self::QTensor(qt)),
        }
    }

    /// The weight as a dense tensor, when this projection holds one: what a fused launch reads
    /// without going through a matrix product.
    pub fn dense(&self) -> Option<&Tensor> {
        match self {
            Self::Tensor(w) => Some(w),
            _ => None,
        }
    }

    /// The first `n` rows of a quantised weight on a card, as a view of the same blocks: no
    /// copy, no memory. `None` for a weight held otherwise.
    pub fn row_prefix(&self, n: usize) -> Result<Option<Self>> {
        match self {
            Self::QTensor(q) => Ok(q.row_prefix(n)?.map(|t| Self::QTensor(Arc::new(t)))),
            _ => Ok(None),
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
            Self::Tensor(t) | Self::TensorF16(t) | Self::Bf16(t) => t.device().clone(),
        }
    }

    /// The BF16 weight `qt` held as stored on `dev`: its bits decoded on the host and placed
    /// once. A counting device counts it at two bytes a weight.
    pub fn bf16_onto(qt: &QTensor, dev: &Device) -> Result<Self> {
        if dev.is_dry() {
            return Ok(Self::Bf16(Tensor::dry(
                dev,
                crate::tensor::DType::BF16,
                qt.shape().clone(),
            )?));
        }
        // CONVERSION: none of value: the host decode of BF16 blocks is the BF16 tensor itself.
        Ok(Self::Bf16(qt.dequantize(&Device::Cpu)?.to_device(dev)?))
    }

    /// The F16 weight `qt` held as stored on `dev`, as [`Self::bf16_onto`] holds a BF16 one.
    pub fn f16_onto(qt: &QTensor, dev: &Device) -> Result<Self> {
        if dev.is_dry() {
            return Ok(Self::TensorF16(Tensor::dry(
                dev,
                crate::tensor::DType::F16,
                qt.shape().clone(),
            )?));
        }
        // CONVERSION: none of value: the host decode of F16 blocks is the F16 tensor itself.
        Ok(Self::TensorF16(
            qt.dequantize(&Device::Cpu)?.to_device(dev)?,
        ))
    }

    /// The stored float format this projection holds its weight in, `None` for block formats.
    pub fn float_format(&self) -> Option<GgmlDType> {
        match self {
            Self::QTensor(q) if host_half_view(q) => Some(q.dtype()),
            Self::QTensor(_) => None,
            Self::Tensor(_) => Some(GgmlDType::F32),
            Self::TensorF16(_) => Some(GgmlDType::F16),
            Self::Bf16(_) => Some(GgmlDType::BF16),
        }
    }

    /// The BF16 weight a card holds, when this projection is one.
    pub fn bf16(&self) -> Option<&Tensor> {
        match self {
            Self::Bf16(w) if w.device().runs_as_card() => Some(w),
            _ => None,
        }
    }

    /// fork-compat: dequantize to F16 on the weight's device.
    pub fn dequantize_f16(&self) -> Result<Tensor> {
        match self {
            // CONVERSION: the F16 dequantisation is what the caller asked for.
            Self::QTensor(t) => t.dequantize_f16(t.device()),
            Self::Tensor(t) => t.to_dtype(crate::tensor::DType::F16),
            Self::TensorF16(t) => Ok(t.clone()),
            Self::Bf16(t) => t.to_dtype(crate::tensor::DType::F16),
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

    /// `x @ w^T` for the F32 rows `x` `[rows, k]` and a half matrix `w` `[n, k]`, the rows read in
    /// F32 and the products summed in F32.
    /// The matrix is widened to F32 a slice of its rows at a time.
    /// A widened slice takes no more memory than the product's output.
    fn widened_product(x: &Tensor, w: &Tensor) -> Result<Tensor> {
        let (rows, n, k) = (x.dims()[0], w.dims()[0], w.dims()[1]);
        let slice = (rows * n / k.max(1)).clamp(1, n);
        let mut parts = Vec::with_capacity(n.div_ceil(slice));
        let mut at = 0;
        while at < n {
            let m = slice.min(n - at);
            // CONVERSION: none of value: a half matrix widens to F32 exactly.
            let wide = w.narrow(0, at, m)?.to_dtype(crate::tensor::DType::F32)?;
            parts.push(x.matmul_t(&wide)?);
            at += m;
        }
        match parts.len() {
            1 => Ok(parts.remove(0)),
            _ => Tensor::cat(&parts.iter().collect::<Vec<_>>(), 1),
        }
    }

    /// `x @ w^T` on the host for a weight held as stored in F16 or BF16: the activation read as
    /// F32, the weight widened row by row inside the product, summed in F32.
    fn host_half(x: &Tensor, w: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let k = *dims
            .last()
            .ok_or_else(|| Error::msg("QMatMul: rank-0 input"))?;
        let rows = x.elem_count() / k;
        let x2 = x.to_dtype(crate::tensor::DType::F32)?.reshape((rows, k))?;
        // A weight held as a view of a larger storage takes the F32 product of its widened copy.
        let y = match x2.matmul_t_half_f32(w)? {
            Some(y) => y,
            None => x2.matmul_t(&w.to_dtype(crate::tensor::DType::F32)?)?,
        };
        let mut odims = dims;
        *odims.last_mut().unwrap() = w.dims()[0];
        y.reshape(odims)?.to_dtype(x.dtype())
    }

    /// [`Self::host_half`] over a half weight read in place from its file mapping.
    /// The product is the one a copy of the weight held as stored gives.
    fn host_half_mapped(x: &Tensor, qt: &QTensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let k = *dims
            .last()
            .ok_or_else(|| Error::msg("QMatMul: rank-0 input"))?;
        let n = qt.shape().dims()[0];
        let rows = x.elem_count() / k;
        let x2 = x.to_dtype(crate::tensor::DType::F32)?.reshape((rows, k))?;
        let bytes = qt.native_qtensor().data();
        let y = match x2.matmul_t_half_bytes_f32(bytes, half_dtype(qt.dtype()), n)? {
            Some(y) => y,
            // CONVERSION: none of value: a half matrix widens to F32 exactly.
            None => x2.matmul_t(
                &qt.dequantize(&Device::Cpu)?
                    .to_dtype(crate::tensor::DType::F32)?,
            )?,
        };
        let mut odims = dims;
        *odims.last_mut().unwrap() = n;
        y.reshape(odims)?.to_dtype(x.dtype())
    }

    /// The rows one call of a float weight on a card takes while a verify holds the decode
    /// order: the most rows one launch of the float mat-vec kernel takes.
    /// `None` for a weight that is not a float weight on a card.
    fn decode_rows(&self) -> Option<usize> {
        match self {
            Self::Tensor(w) | Self::TensorF16(w) | Self::Bf16(w) if w.device().is_cuda() => {
                Some(crate::inference::moe_cuda::FLOAT_MATVEC_ROWS)
            }
            _ => None,
        }
    }

    /// `x @ w^T` for a float weight `w` on a card through the float mat-vec kernel.
    /// It takes up to `few` rows, or the rows of a verify while it holds the decode order.
    /// A decode step and a verify row take the same kernel, so a verify row gives its step.
    /// `None` where the kernel does not take the call.
    fn card_matvec(x: &Tensor, w: &Tensor, few: usize) -> Result<Option<Tensor>> {
        let mut dims = x.dims().to_vec();
        let k = *dims
            .last()
            .ok_or_else(|| Error::msg("QMatMul: rank-0 input"))?;
        let rows = x.elem_count() / k.max(1);
        if !w.device().is_cuda() || (rows > few && !DecodeOrder::active()) {
            return Ok(None);
        }
        let Some(y) = crate::inference::moe_cuda::float_matvec(&x.reshape((rows, k))?, w)? else {
            return Ok(None);
        };
        *dims.last_mut().expect("checked above") = w.dims()[0];
        Ok(Some(y.reshape(dims)?.to_dtype(x.dtype())?))
    }

    /// Whether the rows of a verify through the float weight `w` on a card take a one-row call
    /// each, the float mat-vec kernel having declined them.
    fn verify_rows_apart(x: &Tensor, w: &Tensor) -> bool {
        let k = x.dims().last().copied().unwrap_or(0);
        k > 0 && x.elem_count() / k > 1 && DecodeOrder::active() && w.device().is_cuda()
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // A verify's rows through a float weight on a card go a launch's worth at a time.
        if let Some(per_call) = self.decode_rows().filter(|_| DecodeOrder::active()) {
            let k = x.dims().last().copied().unwrap_or(0);
            if k > 0 && x.elem_count() / k > per_call {
                return DecodeOrder::by_rows(x, per_call, |rows| self.forward(rows));
            }
        }
        match self {
            Self::QTensor(qt) if host_half_view(qt) => Self::host_half_mapped(x, qt),
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
            // An F32 weight takes the activation read as F32, a half one included.
            // The output follows the activation's dtype.
            // On a card, one row and the rows of a verify take the float mat-vec kernel.
            Self::Tensor(w) => {
                if let Some(y) = Self::card_matvec(x, w, 1)? {
                    return Ok(y);
                }
                if Self::verify_rows_apart(x, w) {
                    return DecodeOrder::by_rows(x, 1, |row| self.forward(row));
                }
                Self::dense_matmul_t(&x.to_dtype(crate::tensor::DType::F32)?, w, x.dtype())
            }
            Self::TensorF16(w) if w.device().is_cpu() => Self::host_half(x, w),
            Self::Bf16(w) if w.device().is_cpu() => Self::host_half(x, w),
            Self::TensorF16(w) => {
                // One row and the rows of a verify take the float mat-vec kernel.
                // Where it declines, one row takes the F16-weight product summed in F32.
                // More rows take the library's F16 product with the activation rounded to F16,
                // summed in F32 where the card offers it, where the level allows it.
                if let Some(y) = Self::card_matvec(x, w, 1)? {
                    return Ok(y);
                }
                if Self::verify_rows_apart(x, w) {
                    return DecodeOrder::by_rows(x, 1, |row| self.forward(row));
                }
                let k = *x
                    .dims()
                    .last()
                    .ok_or_else(|| Error::msg("QMatMul: rank-0 input"))?;
                let mut odims = x.dims().to_vec();
                *odims.last_mut().unwrap() = w.dims()[0];
                if x.elem_count() == k {
                    let row = x.to_dtype(crate::tensor::DType::F32)?.reshape((1, k))?;
                    if let Some(y) = row.gemv_f16w(w, None)? {
                        return y.reshape(odims)?.to_dtype(x.dtype());
                    }
                }
                // Where rows may not be read in F16, they stay F32 as the single row does.
                if !half_rows() {
                    let rows = x
                        .to_dtype(crate::tensor::DType::F32)?
                        .reshape((x.elem_count() / k, k))?;
                    let y = Self::widened_product(&rows, w)?;
                    return y.reshape(odims)?.to_dtype(x.dtype());
                }
                let xf = x.to_dtype(crate::tensor::DType::F16)?;
                let rows = xf.reshape((x.elem_count() / k, k))?;
                if let Some(y) = rows.matmul_t_f16_f32(w)? {
                    return y.reshape(odims)?.to_dtype(x.dtype());
                }
                Self::dense_matmul_t(&xf, w, x.dtype())
            }
            Self::Bf16(w) => {
                // A few rows take the float mat-vec kernel, summed in F32, the decode's and a
                // verify's rows alike.
                // More take the library's BF16 product with the activation rounded to BF16,
                // summed in F32, where the level allows it.
                if let Some(y) =
                    Self::card_matvec(x, w, crate::inference::moe_cuda::BF16_MATVEC_ROWS)?
                {
                    return Ok(y);
                }
                if Self::verify_rows_apart(x, w) {
                    return DecodeOrder::by_rows(x, 1, |row| self.forward(row));
                }
                let dims = x.dims().to_vec();
                let k = *dims
                    .last()
                    .ok_or_else(|| Error::msg("QMatMul: rank-0 input"))?;
                let rows = x.elem_count() / k;
                let x2 = x.reshape((rows, k))?;
                // Where rows may not be read in BF16, they stay F32 as a few rows do.
                let y = if !half_rows() {
                    Self::widened_product(&x2.to_dtype(crate::tensor::DType::F32)?, w)?
                } else {
                    #[cfg(feature = "gpu")]
                    let y = x2
                        .to_dtype(crate::tensor::DType::BF16)?
                        .matmul_t_bf16_f32(w)?;
                    #[cfg(not(feature = "gpu"))]
                    let y: Option<Tensor> = None;
                    match y {
                        Some(y) => y,
                        None => x2
                            .to_dtype(crate::tensor::DType::F32)?
                            .matmul_t(&w.to_dtype(crate::tensor::DType::F32)?)?,
                    }
                };
                let mut odims = dims;
                *odims.last_mut().unwrap() = w.dims()[0];
                y.reshape(odims)?.to_dtype(x.dtype())
            }
        }
    }

    /// Fused dense SwiGLU decode `silu(self.x)*(up.x)` for block-quantized
    /// CPU weights + F16 activation (delegates to the native kernel).
    /// `Ok(None)` for any non-fast-path case -> caller runs the unfused path.
    pub fn gate_up_silu_f16(&self, up: &QMatMul, x: &Tensor) -> Result<Option<Tensor>> {
        match (self, up) {
            (Self::QTensor(g), Self::QTensor(u)) if !host_half_view(g) && !host_half_view(u) => {
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
            Self::QTensor(qt) if !host_half_view(qt) => qt.native_qmm()?.forward_slice_cpu(x, out),
            _ => Ok(false),
        }
    }

    /// [`Self::forward_slice_cpu`] over several rows at once, each row bit-identical to a
    /// one-row call. `Ok(false)` where the one-row call declines.
    pub fn forward_rows_cpu(&self, x: &[f32], out: &mut [f32]) -> Result<bool> {
        match self {
            Self::QTensor(qt) if !host_half_view(qt) => qt.native_qmm()?.forward_rows_cpu(x, out),
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
                Self::QTensor(qt) if !host_half_view(qt) => inner.push(qt.native_qmm()?),
                _ => return Ok(false),
            }
        }
        QKernelMatMul::forward_slice_cpu_shared(x, &inner, outs)
    }

    /// Output dim (`n`) - to size decode-arena buffers.
    pub fn out_dim(&self) -> Result<usize> {
        match self {
            Self::QTensor(qt) if host_half_view(qt) => Ok(qt.shape().dims()[0]),
            Self::QTensor(qt) => Ok(qt.native_qmm()?.out_dim()),
            Self::Tensor(w) | Self::TensorF16(w) | Self::Bf16(w) => Ok(w.dims()[0]),
        }
    }

    /// Dense BF16 copy of the weight, dequantized ONCE (see
    /// [`QKernelMatMul::dequant_weight_bf16`]). `Ok(None)` for dense/CPU weights.
    #[cfg(feature = "gpu")]
    pub fn dequant_weight_bf16(&self) -> Result<Option<Tensor>> {
        match self {
            Self::QTensor(qt) => qt.native_qmm()?.dequant_weight_bf16(),
            _ => Ok(None),
        }
    }

    /// The blocks of a quantised weight on a card, as stored: `(blocks, type, n, k)`.
    #[cfg(feature = "gpu")]
    pub(crate) fn card_blocks(
        &self,
    ) -> Option<(
        &crate::tensor::gpu_runtime::driver::CudaSlice<u8>,
        GgmlDType,
        usize,
        usize,
    )> {
        match self {
            Self::QTensor(qt) => qt.native_qmm().ok()?.card_blocks(),
            _ => None,
        }
    }

    /// The kernel matmul `forward` runs, `None` for a dense weight.
    pub(crate) fn kernel(&self) -> Option<&super::QKernelMatMul> {
        match self {
            Self::QTensor(qt) if !host_half_view(qt) => qt.native_qmm().ok(),
            _ => None,
        }
    }

    /// Raw CPU quantized weight `(dtype, k, n, bytes)` for fused multi-weight
    /// regions (MoE flat `matmul_bytes_multi`). Forwards to the kernel matmul
    /// (the borrow lives as long as `self` - `native_qmm` caches a reference).
    pub fn cpu_raw(&self) -> Option<(GgmlDType, usize, usize, &[u8])> {
        match self {
            Self::QTensor(qt) if !host_half_view(qt) => qt.native_qmm().ok()?.cpu_raw(),
            _ => None,
        }
    }
}

/// `x @ w^T` for the rows `x` `[rows, k]` and the float head `w` `[n, k]` held in F16 on a card.
/// One row, and the rows of a verify while it holds the decode order, take the float mat-vec
/// kernel: the product summed and written in F32.
/// A card that takes that kernel through its tensor cores multiplies each row as 2 F16 parts:
/// its nearest F16 value and what that misses.
/// Another card reads the rows in F32.
/// Where that kernel declines, each of them takes the product of more rows alone.
/// More rows are rounded to F16, and their product is written in F16.
pub fn half_head(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let k = x.dims().last().copied().unwrap_or(0);
    let rows = x.elem_count() / k.max(1);
    let half_rows = |x: &Tensor| x.to_dtype(crate::tensor::DType::F16)?.matmul_t(w);
    if rows > 1 && !DecodeOrder::active() {
        return half_rows(x);
    }
    let x = x.to_dtype(crate::tensor::DType::F32)?;
    let per_call = crate::inference::moe_cuda::FLOAT_MATVEC_ROWS;
    DecodeOrder::by_rows(
        &x,
        per_call,
        |part| match crate::inference::moe_cuda::float_matvec(part, w)? {
            Some(y) => Ok(y),
            None => DecodeOrder::by_rows(part, 1, half_rows),
        },
    )
}

impl crate::tensor::Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> tensor::Result<Tensor> {
        QMatMul::forward(self, xs)
    }
}
