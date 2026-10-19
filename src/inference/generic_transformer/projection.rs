//! The quantised projection a transformer layer multiplies through, and the AWQ weight
//! behind one of its forms.
//!
//! Two ways a weight arrives quantised, and they are not interchangeable: GGUF block formats,
//! which the CPU and mmvq kernels read directly, and AWQ - 4-bit with a per-group zero point,
//! which reaches a tensor core only after a repack into Marlin's tile order. [`QMatMul`] is
//! the one type a layer holds either way, so the layer never branches on which it got.
//!
//! Carried here from a file named after the Mistral3 model. That model's implementation was
//! superseded by the generic transformer and is gone; what stayed is this, which every
//! architecture the loader builds goes through.

use crate::tensor::quantized::QTensor;
use crate::tensor::Module;
use crate::tensor::{DType, Device, Error, Result, Tensor};

/// Slot type for the Marlin W4A16 layer. Arc'd so `Clone` shares the
/// device-resident repacked weights; a unit stub keeps the field (and the
/// struct literals that initialize it) identical on non-CUDA builds.
#[cfg(feature = "gpu")]
pub type MarlinSlot = std::sync::Arc<crate::tensor::marlin::MarlinAwqLayer>;
#[cfg(not(feature = "gpu"))]
pub type MarlinSlot = std::sync::Arc<()>;

/// Which layouts an AWQ linear keeps resident.
///
/// `Dual` keeps both: the dp4a layout serves M=1 decode, where it is marginally faster and
/// natively f32, and the repacked Marlin layout serves the small-M speculative-verify regime
/// the tensor cores win. That doubles the per-linear weight VRAM, so a build that would not
/// leave enough headroom falls back to `Off` for that layer - which is the only way this is
/// chosen. `Only` drops the dp4a layout wherever the repack applies and is VRAM-neutral.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AwqMarlinPolicy {
    Off,
    Dual,
    Only,
}

/// One layout resident: the repacked one, which serves every row count through the
/// tensor-core GEMM, and the dp4a layout only for a shape the repack does not cover. A
/// second copy of the weights would take the room the first prefill needs.
fn awq_marlin_policy() -> AwqMarlinPolicy {
    AwqMarlinPolicy::Only
}

/// Prefill-sized forwards run on the Marlin M-loop rather than dequant-then-GEMM: it
/// re-streams the 4-bit weights per 32-row chunk but skips materialising the whole f16
/// dequantisation, promoting it to f32, and a cuBLAS pass over it. Measured faster.
fn awq_marlin_prefill() -> bool {
    true
}

/// AWQ (uniform 4-bit) resident weight for one linear layer. Holds the raw
/// AutoAWQ GEMM tensors on-device; `forward` runs the custom GEMV kernel at decode
/// (seq=1) and a dequant->cuBLAS f16 GEMM at prefill (seq>1). See `inference::load::awq`
/// (CPU reference, bit-exact vs vLLM) and `cuda/awq_gemv.cu`.
#[derive(Debug)]
pub struct AwqWeight {
    /// `[K, N/8]` i32 - 8 output nibbles per i32 (AWQ order).
    pub qweight: Tensor,
    /// `[K/gs, N/8]` i32 - packed zero-points.
    pub qzeros: Tensor,
    /// `[K/gs, N]` f16 - per (group, output) scale.
    pub scales: Tensor,
    /// Optional `[N]` bias (qwen2 q/k/v).
    pub bias: Option<Tensor>,
    pub k: usize,
    pub n: usize,
    pub group_size: usize,
    /// Output-major repacked tensors `(qw_t[N,K/8]i32, scales_t[N,K/gs]f16,
    /// zeros_t[N,K/gs]u8)` for the multi-row no-atomic decode GEMV (mr4). Built
    /// once (eagerly in `from_awq` on CUDA) and reused; `None` = use the K-major
    /// split-K kernel. Lazy slot so it survives Arc clones.
    pub repacked: std::sync::OnceLock<Option<(Tensor, Tensor, Tensor)>>,
    /// Marlin W4A16 tensor-core layer for the small-M (2..=32)
    /// speculative-verify regime (and, per policy, M=1 decode + prefill).
    /// Built once at load per [`awq_marlin_policy`]; `None` = unsupported
    /// shape / policy off / VRAM headroom exhausted.
    pub marlin: std::sync::OnceLock<Option<MarlinSlot>>,
}

impl Clone for AwqWeight {
    fn clone(&self) -> Self {
        let repacked = std::sync::OnceLock::new();
        if let Some(v) = self.repacked.get() {
            let _ = repacked.set(v.clone());
        }
        let marlin = std::sync::OnceLock::new();
        if let Some(v) = self.marlin.get() {
            let _ = marlin.set(v.clone());
        }
        Self {
            qweight: self.qweight.clone(),
            qzeros: self.qzeros.clone(),
            scales: self.scales.clone(),
            bias: self.bias.clone(),
            k: self.k,
            n: self.n,
            group_size: self.group_size,
            repacked,
            marlin,
        }
    }
}

impl AwqWeight {
    fn add_bias(&self, y: Tensor) -> Result<Tensor> {
        match &self.bias {
            Some(b) => y.broadcast_add(&b.to_dtype(y.dtype())?),
            None => Ok(y),
        }
    }

    /// One-time device->host round-trip of the raw AutoAWQ tensors, shared by
    /// the dp4a and Marlin load-time repacks.
    #[cfg(feature = "gpu")]
    fn host_tensors(&self) -> Result<(Vec<i32>, Vec<i32>, Vec<half::f16>)> {
        let qweight = self.qweight.flatten_all()?.to_vec1::<i32>()?;
        let qzeros = self.qzeros.flatten_all()?.to_vec1::<i32>()?;
        let scales_f32 = self
            .scales
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let scales: Vec<half::f16> = scales_f32.into_iter().map(half::f16::from_f32).collect();
        Ok((qweight, qzeros, scales))
    }

    /// Build the output-major repacked tensors for the dp4a decode kernel from
    /// host copies of the raw tensors (`awq::repack_awq`). Resident on the
    /// weight's device.
    #[cfg(feature = "gpu")]
    fn build_repacked_host(
        &self,
        qweight: &[i32],
        qzeros: &[i32],
        scales: &[half::f16],
    ) -> Result<(Tensor, Tensor, Tensor)> {
        use crate::inference::load::awq::{repack_awq, AwqTensor};
        let dev = self.qweight.device();
        let t = AwqTensor::new(
            self.k,
            self.n,
            self.group_size,
            qweight.to_vec(),
            qzeros.to_vec(),
            scales.to_vec(),
        );
        // CONVERSION: the AWQ weight is repacked output-major for the dp4a decode kernel.
        let r = repack_awq(&t);
        let ng = self.k / self.group_size;
        let qw_t = Tensor::from_vec(r.qw_t, (self.n, self.k / 8), &dev)?;
        let scales_t = Tensor::from_vec(r.scales_t, (self.n, ng), &dev)?;
        let zeros_t = Tensor::from_vec(r.zeros_t, (self.n, ng), &dev)?;
        Ok((qw_t, scales_t, zeros_t))
    }

    /// Build the Marlin W4A16 layer from host copies. `Ok(None)` =
    /// skipped because the dual-layout VRAM headroom would drop below the
    /// reserve (the per-layer graceful-degradation path); `Err` = real failure.
    #[cfg(feature = "gpu")]
    fn build_marlin(
        &self,
        qweight: &[i32],
        qzeros: &[i32],
        scales: &[half::f16],
        check_headroom: bool,
        keep_free: usize,
    ) -> Result<Option<crate::tensor::marlin::MarlinAwqLayer>> {
        use crate::tensor::marlin;
        let dev = self.qweight.device();
        let r =
            // CONVERSION: the AWQ weight is repacked into the tensor-core layout, built only while room stays free.
            marlin::repack_awq_to_marlin(qweight, qzeros, scales, self.k, self.n, self.group_size)
                .map_err(|e| crate::tensor::Error::msg(e.0))?;
        if check_headroom {
            let free = crate::inference::place::room::free_now(&dev).unwrap_or(0) as usize;
            if free < r.device_bytes() + keep_free {
                return Ok(None);
            }
        }
        let ndev = dev.as_cuda_device()?.native().clone();
        let layer =
            marlin::MarlinAwqLayer::new(&ndev, &r).map_err(|e| crate::tensor::Error::msg(e.0))?;
        Ok(Some(layer))
    }

    /// Marlin forward for any M: one f32->f16 activation convert, the Marlin
    /// GEMM (M-looped in chunks of 32 inside the launcher), one f16->f32 output
    /// convert (downstream consumers are f32, like the dp4a/GEMM paths).
    #[cfg(feature = "gpu")]
    fn forward_marlin(
        &self,
        xs: &Tensor,
        layer: &crate::tensor::marlin::MarlinAwqLayer,
        tokens: usize,
        out_shape: Vec<usize>,
    ) -> Result<Tensor> {
        use crate::tensor::cuda_ext;
        let dev = xs.device();
        let ndev = dev.as_cuda_device()?.native().clone();
        let x = xs
            .reshape((tokens, self.k))?
            .to_dtype(DType::F16)?
            .contiguous()?;
        let xv = cuda_ext::f16_slice_of(&x)?;
        let view = xv.view()?;
        let c = layer
            .forward_view(&ndev, &view, tokens)
            .map_err(|e| crate::tensor::Error::msg(format!("awq marlin: {}", e.0)))?;
        let y = cuda_ext::tensor_from_f16_slice(c, (tokens, self.n), &dev)?.to_dtype(DType::F32)?;
        let y = self.add_bias(y)?;
        y.reshape(out_shape)
    }

    #[cfg(feature = "gpu")]
    fn forward_cuda(&self, xs: &Tensor) -> Result<Tensor> {
        use crate::tensor::cuda_ext::tensor_from_cuda_storage;
        let dims = xs.dims();
        let kin = *dims.last().unwrap();
        if kin != self.k {
            crate::tensor::bail!("AWQ forward: input dim {kin} != K {}", self.k);
        }
        let tokens: usize = dims[..dims.len() - 1].iter().product();
        let dev = xs.device();
        let mut out_shape = dims[..dims.len() - 1].to_vec();
        out_shape.push(self.n);

        let marlin = self.marlin.get().and_then(|o| o.as_ref());

        if tokens == 1 {
            // Decode: GEMV kernel. x must be f32, contiguous, length K.
            // Prefer the dp4a kernel (q8_1 activation x int4 weight) over the
            // repacked layout when available: mmvq-class ~833 GB/s vs the K-major
            // split-K's ~600 in-forward. Falls back to split-K otherwise.
            // Under the Marlin-only policy the dp4a layout is absent and the
            // Marlin GEMM serves M=1 too (bandwidth-parity with dp4a).
            if let Some((qw_t, scales_t, zeros_t)) = self.repacked.get().and_then(|o| o.as_ref()) {
                let x = xs.reshape((self.k,))?.to_dtype(DType::F32)?.contiguous()?;
                let st = crate::inference::quantized_cuda::awq_gemv_dp4a_storage(
                    qw_t,
                    scales_t,
                    zeros_t,
                    &x,
                    self.n,
                    self.k,
                    self.group_size,
                )
                .map_err(|e| crate::tensor::Error::msg(format!("awq dp4a gemv: {e}")))?;
                let y = tensor_from_cuda_storage(st, (self.n,))?;
                let y = self.add_bias(y)?;
                return y.reshape(out_shape);
            }
            if let Some(m) = marlin {
                return self.forward_marlin(xs, m, 1, out_shape);
            }
            let x = xs.reshape((self.k,))?.to_dtype(DType::F32)?.contiguous()?;
            let st = crate::inference::quantized_cuda::awq_gemv_storage(
                &self.qweight,
                &self.qzeros,
                &self.scales,
                &x,
                self.n,
                self.k,
                self.group_size,
            )
            .map_err(|e| crate::tensor::Error::msg(format!("awq gemv: {e}")))?;
            let y = tensor_from_cuda_storage(st, (self.n,))?;
            let y = self.add_bias(y)?;
            return y.reshape(out_shape);
        }
        // Small-M (2..=32): the speculative-verify regime (PLD `forward_all`
        // drafts; EAGLE later). One Marlin tensor-core GEMM keeps near-flat
        // µs/call over M where the dp4a loop scales linearly and dequant+GEMM
        // costs a full weight materialization (M=8 48µs vs 183
        // dp4a-loop / 3495 dequant+GEMM at the gate/up shape).
        if tokens <= 32 {
            if let Some(m) = marlin {
                return self.forward_marlin(xs, m, tokens, out_shape);
            }
        }
        // Small-M without Marlin (off-policy / unsupported shape, M<=8): loop
        // the mmvq-class dp4a GEMV per row instead of a full dequant+GEMM.
        // A PLD draft is 2-4 rows -> 2-4 cheap dp4a calls beat
        // dequantizing the whole [K,N] weight.
        if tokens <= 8 {
            if let Some((qw_t, scales_t, zeros_t)) = self.repacked.get().and_then(|o| o.as_ref()) {
                let x2 = xs
                    .reshape((tokens, self.k))?
                    .to_dtype(DType::F32)?
                    .contiguous()?;
                let mut rows: Vec<Tensor> = Vec::with_capacity(tokens);
                for m in 0..tokens {
                    let xm = x2.narrow(0, m, 1)?.reshape((self.k,))?.contiguous()?;
                    let st = crate::inference::quantized_cuda::awq_gemv_dp4a_storage(
                        qw_t,
                        scales_t,
                        zeros_t,
                        &xm,
                        self.n,
                        self.k,
                        self.group_size,
                    )
                    .map_err(|e| crate::tensor::Error::msg(format!("awq dp4a small-M: {e}")))?;
                    let y = tensor_from_cuda_storage(st, (1, self.n))?;
                    rows.push(y);
                }
                let y = Tensor::cat(&rows.iter().collect::<Vec<_>>(), 0)?;
                let y = self.add_bias(y)?;
                let _ = dev;
                return y.reshape(out_shape);
            }
        }
        // Prefill (M>32): Marlin M-loop when enabled (default - skips the
        // full-weight f16 dequant + f32 promotion + cuBLAS pass), otherwise
        // dequant+GEMM below.
        if let Some(m) = marlin {
            if awq_marlin_prefill() || self.repacked.get().and_then(|o| o.as_ref()).is_none() {
                return self.forward_marlin(xs, m, tokens, out_shape);
            }
        }
        // Prefill: dequantize to f16 [K, N], then GEMM x[tokens,K] @ W[K,N] in
        // F32. We do NOT cast the activation to F16 - Qwen2 "massive activations"
        // exceed the F16 range (±65504) in deeper layers, which would overflow to
        // inf/NaN. The weight is dequantized as F16 (compact) then promoted to F32
        // for the matmul; the activation stays F32.
        let st = match self.repacked.get().and_then(|o| o.as_ref()) {
            // Repacked-only weight (K-major dropped to save VRAM): dequant from
            // the output-major layout.
            Some((qw_t, scales_t, zeros_t)) => {
                crate::inference::quantized_cuda::awq_dequant_f16_repacked_storage(
                    qw_t,
                    scales_t,
                    zeros_t,
                    self.n,
                    self.k,
                    self.group_size,
                )
                .map_err(|e| crate::tensor::Error::msg(format!("awq dequant rep: {e}")))?
            }
            None => crate::inference::quantized_cuda::awq_dequant_f16_storage(
                &self.qweight,
                &self.qzeros,
                &self.scales,
                self.n,
                self.k,
                self.group_size,
            )
            .map_err(|e| crate::tensor::Error::msg(format!("awq dequant: {e}")))?,
        };
        let w_kn = tensor_from_cuda_storage(st, (self.k, self.n))?.to_dtype(DType::F32)?;
        let x2 = xs.reshape((tokens, self.k))?.to_dtype(DType::F32)?;
        let y = x2.matmul(&w_kn)?;
        let y = self.add_bias(y)?;
        let _ = dev;
        y.reshape(out_shape)
    }

    /// CPU fallback (CPU-only build, or AWQ tensors landed on CPU): dequantize via
    /// the `awq` reference into an f32 `[K, N]` weight and matmul. Slow - AWQ is a
    /// GPU decode lever - but keeps the non-cuda build correct.
    fn forward_cpu(&self, xs: &Tensor) -> Result<Tensor> {
        use crate::inference::load::awq::AwqTensor;
        let qweight = self.qweight.flatten_all()?.to_vec1::<i32>()?;
        let qzeros = self.qzeros.flatten_all()?.to_vec1::<i32>()?;
        let scales_f32 = self
            .scales
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let scales: Vec<half::f16> = scales_f32.into_iter().map(half::f16::from_f32).collect();
        let t = AwqTensor::new(self.k, self.n, self.group_size, qweight, qzeros, scales);
        let w = t.dequant(); // [K, N] row-major = W^T
        let dev = xs.device();
        let w_kn = Tensor::from_vec(w, (self.k, self.n), &dev)?;
        let dims = xs.dims();
        let tokens: usize = dims[..dims.len() - 1].iter().product();
        let mut out_shape = dims[..dims.len() - 1].to_vec();
        out_shape.push(self.n);
        let x2 = xs.reshape((tokens, self.k))?.to_dtype(DType::F32)?;
        let y = x2.matmul(&w_kn)?;
        let y = self.add_bias(y)?;
        y.reshape(out_shape)
    }
}

// QMatMul wrapper. Originally a thin newtype over the reference quantized::QMatMul;
// now enum-backed to also carry an AWQ (uniform 4-bit) weight so the
// generic transformer path can serve AWQ checkpoints unchanged - the variant is
// invisible to callers, which only use `forward`/`qtensor`/`from_*`.
#[derive(Debug, Clone)]
enum QMatMulKind {
    Gguf(crate::tensor::quantized::QMatMul),
    Awq(std::sync::Arc<AwqWeight>),
}

#[derive(Debug, Clone)]
pub struct QMatMul {
    inner: QMatMulKind,
    /// Adapters attached after the checkpoint was read, and their fusion into one pair.
    /// Empty is the weight exactly as the file wrote it.
    lora: Vec<crate::tensor::lora::LoraDelta>,
    lora_fused: Option<(Tensor, Tensor)>,
}

impl QMatMul {
    /// The format the weight is resident in, by name, and the bytes it owns beyond the
    /// file's own blocks (zero for a weight kept as stored, the dense size for a widened one).
    #[cfg(test)]
    pub(crate) fn resident_format(&self) -> (String, u64) {
        match &self.inner {
            QMatMulKind::Gguf(crate::tensor::quantized::QMatMul::QTensor(q)) => {
                (q.dtype().gguf_name().to_string(), 0)
            }
            QMatMulKind::Gguf(
                crate::tensor::quantized::QMatMul::Tensor(t)
                | crate::tensor::quantized::QMatMul::TensorF16(t)
                | crate::tensor::quantized::QMatMul::Bf16(t),
            ) => (
                t.dtype().as_str().to_string(),
                (t.elem_count() * t.dtype().size_in_bytes()) as u64,
            ),
            QMatMulKind::Awq(_) => ("awq_int4".to_string(), 0),
        }
    }

    /// The weight as a dense tensor when it is held as one and carries no adapter: what a fused
    /// launch reads in place of the projection.
    pub fn dense(&self) -> Option<&Tensor> {
        match &self.inner {
            QMatMulKind::Gguf(inner) if self.lora.is_empty() => inner.dense(),
            _ => None,
        }
    }

    /// The BF16 weight a card holds, when it carries no adapter.
    pub fn bf16(&self) -> Option<&Tensor> {
        match &self.inner {
            QMatMulKind::Gguf(inner) if self.lora.is_empty() => inner.bf16(),
            _ => None,
        }
    }

    pub fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = crate::tensor::quantized::QMatMul::from_qtensor(qtensor)?;
        Ok(Self {
            inner: QMatMulKind::Gguf(inner),
            lora: Vec::new(),
            lora_fused: None,
        })
    }

    /// The BF16 weight `qt` held as stored on the card `dev`.
    pub fn bf16_onto(qt: &QTensor, dev: &Device) -> Result<Self> {
        Ok(Self {
            inner: QMatMulKind::Gguf(crate::tensor::quantized::QMatMul::bf16_onto(qt, dev)?),
            lora: Vec::new(),
            lora_fused: None,
        })
    }

    /// Create from an AWQ resident weight. On CUDA, eagerly builds the
    /// output-major repacked layout so decode uses the dp4a kernel (q8_1
    /// activation x int4 weight, mmvq-class ~833 GB/s vs split-K's ~600 ).
    /// (Currently doubles weight VRAM by keeping the K-major copy for prefill;
    /// the single-layout follow-up replaces prefill with a repacked dequant.)
    /// The f32 mr4 repacked path was a measured dead end (-19%); dp4a wins
    /// because it removes the f32 dequant-per-weight via int8 dp4a.
    ///
    /// `keep_free` is VRAM the rest of the load and the model's work still need on this
    /// card: a second layout is built only while that stays free beside it.
    pub fn from_awq(mut w: AwqWeight, keep_free: usize) -> Self {
        // dp4a decode: forward is FAST in-server (12ms < split-K 14ms). The
        // earlier regression was PLD `forward_all` hitting the slow dequant - now
        // small-M forward_all loops dp4a (no dequant, see forward_cuda). Eager-build
        // the repacked layout; drop K-major -> no 2x VRAM.
        // Marlin W4A16: also eager-built per `awq_marlin_policy` - the
        // tensor-core small-M GEMM that removes the speculative-verify cliff.
        #[cfg(feature = "gpu")]
        if w.qweight.device().is_cuda()
            && w.k.is_multiple_of(8)
            && w.k.is_multiple_of(w.group_size.max(1))
        {
            match w.host_tensors() {
                Ok((qw, qz, sc)) => {
                    use crate::tensor::marlin;
                    let policy = awq_marlin_policy();
                    let mut marlin_built = false;
                    // A card without matrix instructions has no build of the kernel, and the
                    // dp4a path serves the layer.
                    if policy != AwqMarlinPolicy::Off
                        && w.qweight
                            .device()
                            .as_cuda_device()
                            .is_ok_and(|d| marlin::runs_on(d.native()))
                        && w.k.is_multiple_of(marlin::MARLIN_TILE)
                        && w.n.is_multiple_of(marlin::MIN_THREAD_N)
                        && w.group_size == marlin::MARLIN_GROUP_SIZE
                    {
                        // In dual mode the Marlin copy is the second layout ->
                        // guard the build on free-VRAM headroom; in only mode
                        // it is the single layout -> always build.
                        let check_headroom = policy == AwqMarlinPolicy::Dual;
                        match w.build_marlin(&qw, &qz, &sc, check_headroom, keep_free) {
                            Ok(Some(layer)) => {
                                let _ = w.marlin.set(Some(std::sync::Arc::new(layer)));
                                marlin_built = true;
                            }
                            Ok(None) => {
                                static WARNED: std::sync::Once = std::sync::Once::new();
                                WARNED.call_once(|| {
                                    tracing::warn!(
                                        "AWQ Marlin: free-VRAM headroom exhausted; \
                                     remaining layers stay dp4a-only"
                                    )
                                });
                            }
                            Err(e) => {
                                tracing::warn!("AWQ Marlin build failed ({e}); dp4a path kept")
                            }
                        }
                    }
                    let mut dp4a_built = false;
                    if !(marlin_built && policy == AwqMarlinPolicy::Only) {
                        match w.build_repacked_host(&qw, &qz, &sc) {
                            Ok(r) => {
                                let _ = w.repacked.set(Some(r));
                                dp4a_built = true;
                            }
                            Err(e) => {
                                tracing::warn!("AWQ repack failed ({e}); using split-K decode")
                            }
                        }
                    }
                    // Drop the K-major originals once any repacked layout can
                    // serve every M (no 2x VRAM for the raw copy).
                    if dp4a_built || marlin_built {
                        let dev = w.qweight.device().clone();
                        if let Ok(d) = Tensor::zeros_on((1, 1), DType::I32, &dev) {
                            w.qweight = d.clone();
                            w.qzeros = d;
                        }
                        if let Ok(d) = Tensor::zeros_on((1, 1), DType::F16, &dev) {
                            w.scales = d;
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("AWQ host round-trip failed ({e}); using split-K decode");
                }
            }
        }
        Self {
            inner: QMatMulKind::Awq(std::sync::Arc::new(w)),
            lora: Vec::new(),
            lora_fused: None,
        }
    }

    /// zero-alloc CPU decode projection: delegates to the inner GGUF
    /// QMatMul's slice path. `Ok(false)` for AWQ / unsupported -> caller falls
    /// back to the Tensor `forward`.
    pub fn forward_slice_cpu(&self, x: &[f32], out: &mut [f32]) -> Result<bool> {
        // An attached adapter is not part of the quantised weight, so this path cannot
        // produce it. Declining sends the caller to `forward`, which applies it; running
        // here would return the base weight's answer and call it the adapted one.
        if self.lora_fused.is_some() {
            return Ok(false);
        }
        match &self.inner {
            QMatMulKind::Gguf(inner) => inner.forward_slice_cpu(x, out),
            _ => Ok(false),
        }
    }

    /// Several projections against ONE activation, quantised once.
    ///
    /// Delegates to the GGUF slice path's shared entry point when every weight
    /// is on it; `Ok(false)` otherwise, and nothing is written, so the caller
    /// falls back to the per-weight calls. Bit-identical to those calls - the
    /// activation goes through the same quantisation routine, just once.
    pub fn forward_slice_cpu_shared(
        x: &[f32],
        mats: &[&Self],
        outs: &mut [&mut [f32]],
    ) -> Result<bool> {
        let mut inners = Vec::with_capacity(mats.len());
        for m in mats {
            if m.lora_fused.is_some() {
                return Ok(false);
            }
            match &m.inner {
                QMatMulKind::Gguf(inner) => inners.push(inner),
                _ => return Ok(false),
            }
        }
        crate::tensor::quantized::QMatMul::forward_slice_cpu_shared(x, &inners, outs)
    }

    /// Raw CPU quantized weight `(dtype, k, n, bytes)` for fused multi-projection
    /// regions (same gate as `forward_slice_cpu`). `None` off the CPU fast path.
    pub fn cpu_raw(&self) -> Option<(crate::tensor::quantized::GgmlDType, usize, usize, &[u8])> {
        // Same reason as the slice path: a caller reading the raw weight would compute
        // without the adapter and never know it was there.
        if self.lora_fused.is_some() {
            return None;
        }
        match &self.inner {
            QMatMulKind::Gguf(inner) => inner.cpu_raw(),
            _ => None,
        }
    }

    /// Output dim (`n`) - to size decode-arena buffers. 0 = unknown (AWQ).
    pub fn out_dim(&self) -> usize {
        match &self.inner {
            QMatMulKind::Gguf(inner) => inner.out_dim().unwrap_or(0),
            _ => 0,
        }
    }

    /// The projection, plus any adapter attached to it.
    ///
    /// Wrapping the base rather than editing it: the quantised forward has several return
    /// paths - a CPU k-quant one and more than one CUDA kernel - and adding the delta at each
    /// is how a family silently ends up without its adapter.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let base = self.forward_base(xs)?;
        let Some((down, up)) = &self.lora_fused else {
            return Ok(base);
        };
        let xdims = xs.dims().to_vec();
        let k = *xdims
            .last()
            .ok_or_else(|| Error("lora: rank-0 input".into()))?;
        let rows = xs.elem_count() / k;
        // The adapters are dense f32 while the activation may be a half carrier, so the
        // conversion happens for THIS term only - the quantised path is untouched.
        let xf = xs
            .to_dtype(crate::tensor::DType::F32)?
            .reshape(vec![1, rows, k])?;
        let acc = xf.matmul(down)?.matmul(up)?;
        let mut odims = xdims;
        *odims.last_mut().unwrap() = up.dims()[2];
        base.add(&acc.reshape(odims)?.to_dtype(base.dtype())?)
    }

    /// Attach a low-rank correction on top of this projection.
    ///
    /// `down` is `[in, r]` and `up` is `[r, out]`. The shapes are checked against each other
    /// and, where the weight can say what it is, against the weight.
    pub fn add_lora(&mut self, delta: crate::tensor::lora::LoraDelta) -> Result<()> {
        let (din, r) = delta.down.shape().dims2()?;
        let (r2, dout) = delta.up.shape().dims2()?;
        if r != r2 {
            return Err(Error(format!("lora: down rank {r} against up rank {r2}")));
        }
        let n = self.out_dim();
        if n != 0 && dout != n {
            return Err(Error(format!(
                "lora: up projects to {dout} on a weight producing {n}"
            )));
        }
        self.lora.push(delta);
        self.lora_fused = crate::tensor::lora::fuse_loras(&self.lora, din, dout);
        if self.lora_fused.is_none() {
            self.lora.clear();
            return Err(Error("lora: adapters did not fuse".into()));
        }
        Ok(())
    }

    /// Drop every attached adapter, returning this projection to the checkpoint.
    pub fn clear_lora(&mut self) {
        self.lora.clear();
        self.lora_fused = None;
    }

    pub fn lora_count(&self) -> usize {
        self.lora.len()
    }

    #[inline(always)]
    fn forward_base(&self, xs: &Tensor) -> Result<Tensor> {
        match &self.inner {
            // Q4_K IMMA M=8 fast path was wired here briefly - REVERTED (5x slower
            // for typical QKV/out_proj shapes; the per-call quantize + extra launch
            // dominates the small matmul).
            QMatMulKind::Gguf(inner) => inner.forward(xs),
            QMatMulKind::Awq(w) => {
                #[cfg(feature = "gpu")]
                if xs.device().is_cuda() {
                    return w.forward_cuda(xs);
                }
                w.forward_cpu(xs)
            }
        }
    }

    /// The device this weight sits on.
    ///
    /// A model split across cards holds each layer on its own, so an adapter has to be moved
    /// to the weight it corrects before it can be attached to it.
    pub fn device(&self) -> Device {
        match &self.inner {
            QMatMulKind::Gguf(crate::tensor::quantized::QMatMul::QTensor(t)) => t.device().clone(),
            QMatMulKind::Gguf(
                crate::tensor::quantized::QMatMul::Tensor(t)
                | crate::tensor::quantized::QMatMul::Bf16(t),
            ) => t.device().clone(),
            QMatMulKind::Awq(w) => w.qweight.device().clone(),
            _ => Device::Cpu,
        }
    }

    /// `silu(gate . x) * (up . x)` for one row `x` `[1, k]` on a card, in the one launch the
    /// card has for a k-quant pair after the row's quantisation: `[1, n]` F32. `None` where the
    /// pair or the row is not what that launch takes; the caller runs the two projections.
    #[cfg(feature = "gpu")]
    pub fn fused_silu_pair(gate: &QMatMul, up: &QMatMul, x: &Tensor) -> Result<Option<Tensor>> {
        use crate::tensor::quantized::{GgmlDType, QStorage};
        // The first pair declined says why, once.
        static DECLINED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let declined = |why: String| {
            if !DECLINED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::info!("fused gate and up pair declined: {why}");
            }
        };
        let (Some(gq), Some(uq)) = (gate.qtensor(), up.qtensor()) else {
            declined("a projection holds no quantised tensor".into());
            return Ok(None);
        };
        let fused_formats = [
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
            GgmlDType::Q8_0,
            GgmlDType::Q4_0,
            GgmlDType::Q4_1,
            GgmlDType::Q5_0,
            GgmlDType::Q5_1,
        ];
        if !fused_formats.contains(&gq.dtype())
            || !fused_formats.contains(&uq.dtype())
            || !x.device().is_cuda()
            || x.dims().last().is_none_or(|&k| x.elem_count() != k)
            || !gate.lora.is_empty()
            || !up.lora.is_empty()
        {
            declined(format!(
                "gate {:?} up {:?} x {:?} {:?} on {:?}, adapters {} and {}",
                gq.dtype(),
                uq.dtype(),
                x.dtype(),
                x.dims(),
                x.device().location(),
                gate.lora.len(),
                up.lora.len()
            ));
            return Ok(None);
        }
        let (QStorage::Cuda(gcuda), QStorage::Cuda(ucuda)) = (gq.storage(), uq.storage()) else {
            return Ok(None);
        };
        let x = x.contiguous()?;
        let (storage, layout) = x.storage_and_layout();
        let crate::tensor::StorageView::Cuda(rhs) = &*storage else {
            return Ok(None);
        };
        let shape = crate::tensor::Shape::from_dims(uq.shape().dims());
        match crate::tensor::quantized::fast_mmvq::try_fused_silu(
            ucuda, gcuda, &shape, rhs, layout,
        )? {
            Some((out, out_shape)) => Ok(Some(crate::tensor::cuda_ext::tensor_from_cuda_storage(
                out, out_shape,
            )?)),
            None => {
                declined(format!(
                    "the launch declined gate {:?} up {:?} {:?} x {:?} {:?}",
                    gq.dtype(),
                    uq.dtype(),
                    uq.shape().dims(),
                    x.dtype(),
                    x.dims()
                ));
                Ok(None)
            }
        }
    }

    /// Borrow the underlying QTensor if this matmul stores its weight as
    /// quantized GGUF data. Returns None for AWQ or dequantized weights.
    pub fn qtensor(&self) -> Option<&std::sync::Arc<crate::tensor::quantized::QTensor>> {
        match &self.inner {
            QMatMulKind::Gguf(crate::tensor::quantized::QMatMul::QTensor(t)) => Some(t),
            _ => None,
        }
    }

    /// The kernel matmul `forward` runs when nothing is added on top of it.
    #[cfg(feature = "gpu")]
    fn bare_kernel(&self) -> Option<&crate::tensor::quantized::QKernelMatMul> {
        if self.lora_fused.is_some() {
            return None;
        }
        self.qtensor()?.native_qmm().ok()
    }

    /// `xs` quantised once for several projections of the same input: see
    /// `QKernelMatMul::quantize_for_matvec`.
    #[cfg(feature = "gpu")]
    pub fn quantize_for_matvec(
        &self,
        xs: &Tensor,
    ) -> Result<Option<crate::tensor::quantized::QuantizedActivation>> {
        match self.bare_kernel() {
            Some(k) => k.quantize_for_matvec(xs),
            None => Ok(None),
        }
    }

    /// The projections `ws` of one input `xs`, each as its own `forward` gives it, with the
    /// input quantised once when every weight takes the mat-vec kernels.
    #[cfg(feature = "gpu")]
    pub fn forward_shared(ws: &[&QMatMul], xs: &Tensor) -> Result<Vec<Tensor>> {
        if xs.device().is_cuda() && ws.len() > 1 {
            if let Some(a) = ws[0].quantize_for_matvec(&xs.contiguous()?)? {
                let shared: Result<Vec<Option<Tensor>>> =
                    ws.iter().map(|w| w.forward_quantized(&a)).collect();
                let shared = shared?;
                if shared.iter().all(Option::is_some) {
                    return Ok(shared.into_iter().flatten().collect());
                }
            }
        }
        ws.iter().map(|w| w.forward(xs)).collect()
    }

    /// [`Self::forward_shared`] on a build with no cards: each projection on its own.
    #[cfg(not(feature = "gpu"))]
    pub fn forward_shared(ws: &[&QMatMul], xs: &Tensor) -> Result<Vec<Tensor>> {
        ws.iter().map(|w| w.forward(xs)).collect()
    }

    /// `forward` over an activation [`Self::quantize_for_matvec`] produced, bit-identical to
    /// `forward` on it; `None` where `forward` would take another path.
    #[cfg(feature = "gpu")]
    pub fn forward_quantized(
        &self,
        a: &crate::tensor::quantized::QuantizedActivation,
    ) -> Result<Option<Tensor>> {
        match self.bare_kernel() {
            Some(k) => k.forward_quantized(a),
            None => Ok(None),
        }
    }

    /// [`Self::forward_quantized`] with `bias` added in the mat-vec's store, the same bits as
    /// the product followed by a bias add; `None` where the row takes another path.
    #[cfg(feature = "gpu")]
    pub fn forward_quantized_bias(
        &self,
        a: &crate::tensor::quantized::QuantizedActivation,
        bias: &Tensor,
    ) -> Result<Option<Tensor>> {
        match self.bare_kernel() {
            Some(k) => k.forward_quantized_bias(a, bias),
            None => Ok(None),
        }
    }

    /// `forward(xs) + bias`: one launch for a decode row the mat-vec serves, the product and
    /// a broadcast add otherwise.
    pub fn forward_bias(&self, xs: &Tensor, bias: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "gpu")]
        if xs.device().is_cuda() {
            if let Some(a) = self.quantize_for_matvec(&xs.contiguous()?)? {
                if let Some(y) = self.forward_quantized_bias(&a, bias)? {
                    return Ok(y);
                }
            }
        }
        self.forward(xs)?.broadcast_add(bias)
    }
}

#[derive(Debug, Clone)]
pub struct Mlp {
    pub feed_forward_w1: QMatMul,
    pub feed_forward_w2: QMatMul,
    pub feed_forward_w3: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        self.feed_forward_w2
            .forward(&(crate::tensor::ops::silu(&w1)? * w3)?)
    }
}

#[cfg(test)]
#[cfg(feature = "gpu")]
mod marlin_parity {
    use super::*;

    /// The INT4 tensor-core GEMM as the layer reaches it, judged against the host path on the
    /// same weights.
    ///
    /// This one covers the wiring: that a linear built from AWQ weights actually resolves to
    /// the tensor-core kernel and returns its answer. The arithmetic itself has a sharper judge
    /// in `tensor::marlin`, which compares against weights it generated and demands equality.
    ///
    /// The reference here is not a dequantised one - sizing a tolerance for that would need the
    /// sum of the term magnitudes before they cancel, and that is some thirty times the result.
    /// It is the CPU forward over the SAME packed tensors: the AWQ nibble order and the group
    /// scales are common to both paths and drop out, leaving the accumulation order and the f16
    /// carrier.
    ///
    /// Three row counts because three kernels answer them, and a zeroed write inside Marlin
    /// says which: at eight rows and at sixty-four the test fails, at one row it does not. So
    /// M=1 is the dp4a GEMV - what the dual policy documents - and this covers Marlin from the
    /// verify regime upward. The single row stays as the third path's own check.
    ///
    /// Two output widths because the work split is decided by how many tiles there are against
    /// how many multiprocessors the card has, and a narrow output never has enough to reach it.
    #[test]
    fn the_int4_tensor_core_gemm_agrees_with_the_host_on_the_same_weights() {
        use crate::tensor::{cuda::CudaDevice, DType, Device, Tensor};
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the Marlin kernels are NOT covered by this run");
            return;
        };
        // Below Ampere the kernel accumulates in f16: each of the k/16 MMA steps rounds the
        // running sum by up to half an ulp, 2^-11 of it, so the bound grows with k. From
        // Ampere on the accumulation is f32 and the f16 carrier alone sets the bound.
        let f16_accumulation = !dev.has_ampere_tensor_cores();
        let gpu = Device::Cuda(dev);

        // Two shapes, because the kernel divides its work differently in each. A narrow
        // output has fewer tiles than the card has multiprocessors, so every block takes one
        // whole column of tiles and the split-work arithmetic is never reached; the wide one
        // has more tiles than that, which is the case where a block computes part of a tile
        // and hands it to the next through the lock. The narrow shape alone left that
        // arithmetic - the largest single passage still taken from upstream - unjudged.
        for &(k, n, gs) in &[(256usize, 128usize, 128usize), (1024, 16384, 128)] {
            let mix = |i: usize, salt: u32| -> u32 {
                (i as u32)
                    .wrapping_mul(2_246_822_519)
                    .wrapping_add(salt)
                    .rotate_left(13)
            };
            // Eight output nibbles per i32, so the packed rows are N/8 wide.
            let qweight: Vec<i32> = (0..k * (n / 8))
                .map(|i| mix(i, 374_761_393) as i32)
                .collect();
            let qzeros: Vec<i32> = (0..(k / gs) * (n / 8))
                .map(|i| (mix(i, 668_265_263) & 0x7777_7777) as i32)
                .collect();
            // Scales small enough that a product of 256 terms stays well inside f16.
            let scales: Vec<f32> = (0..(k / gs) * n)
                .map(|i| ((mix(i, 2_654_435_761) >> 20) as f32 / 4096.0) * 0.02 + 0.001)
                .collect();

            let build = |device: &Device| -> AwqWeight {
                let qw = Tensor::from_vec(qweight.clone(), vec![k, n / 8], device).unwrap();
                let qz = Tensor::from_vec(qzeros.clone(), vec![k / gs, n / 8], device).unwrap();
                let sc = Tensor::from_vec_f32(scales.clone(), vec![k / gs, n])
                    .unwrap()
                    .to_dtype(DType::F16)
                    .unwrap()
                    .to_device(device)
                    .unwrap();
                AwqWeight {
                    qweight: qw,
                    qzeros: qz,
                    scales: sc,
                    bias: None,
                    k,
                    n,
                    group_size: gs,
                    repacked: std::sync::OnceLock::new(),
                    marlin: std::sync::OnceLock::new(),
                }
            };

            // The layer has to exist before the comparison means anything: `build_marlin` declines
            // on an unsupported shape or when VRAM is tight, and a declined build would leave this
            // comparing the dp4a path against the host and calling it Marlin coverage.
            let on_gpu = QMatMul::from_awq(build(&gpu), 0);
            let marlin_built = match &on_gpu.inner {
                QMatMulKind::Awq(w) => matches!(w.marlin.get(), Some(Some(_))),
                _ => false,
            };
            assert!(
                marlin_built,
                "the Marlin layer was not built for K={k} N={n} group={gs}, so this test would \
             cover the dp4a path instead"
            );
            let on_cpu = QMatMul::from_awq(build(&Device::Cpu), 0);

            for rows in [1usize, 8, 64] {
                let xv: Vec<f32> = (0..rows * k)
                    .map(|i| ((i % 71) as f32) * 0.013 - 0.45)
                    .collect();
                let x_gpu = Tensor::from_vec_f32(xv.clone(), vec![rows, k])
                    .unwrap()
                    .to_dtype(DType::F16)
                    .unwrap()
                    .to_device(&gpu)
                    .unwrap();
                let x_cpu = Tensor::from_vec_f32(xv, vec![rows, k]).unwrap();

                let got = on_gpu.forward(&x_gpu).unwrap().to_vec_f32();
                // CPU-FALLBACK-OK: not a fallback - this IS the judge. The host path is what
                // the device result is being compared against, so running it is the point.
                let want = on_cpu.forward(&x_cpu).unwrap().to_vec_f32();
                assert_eq!(got.len(), rows * n, "rows={rows}: wrong output length");

                // The row's own largest output is the scale: a column that cancelled to near zero
                // cannot be held to a relative bound, and the GPU path carries f16.
                let bound = if f16_accumulation {
                    ((k / 16) as f64 * 2f64.powi(-11)).max(8e-3)
                } else {
                    8e-3
                };
                for r in 0..rows {
                    let scale = (0..n).map(|c| want[r * n + c].abs()).fold(0f32, f32::max) as f64;
                    for c in 0..n {
                        let (a, b) = (got[r * n + c] as f64, want[r * n + c] as f64);
                        assert!(
                        (a - b).abs() <= bound * scale,
                        "rows={rows} r={r} col {c}: gpu {a} against the host's {b} on the same \
                         packed weights (row scale {scale}, bound {bound})"
                    );
                    }
                }
                eprintln!("K={k} N={n}, {rows} rows: the INT4 GEMM agrees with the host");
            }
        }
    }
}

#[cfg(test)]
mod lora_tests {
    use super::*;
    use crate::tensor::lora::LoraDelta;
    use crate::tensor::quantized::{GgmlDType, QTensor};

    fn weight(k: usize, n: usize) -> QMatMul {
        let w = Tensor::from_vec_f32(
            (0..n * k).map(|i| ((i % 17) as f32 - 8.0) / 16.0).collect(),
            vec![n, k],
        )
        .expect("weight");
        let q = QTensor::quantize(&w, GgmlDType::Q8_0).expect("quantize");
        QMatMul::from_qtensor(q).expect("matmul")
    }

    /// An attached adapter changes the answer, and dropping it restores the checkpoint's
    /// exactly. Without the second half a no-op attach would pass the first.
    #[test]
    fn an_adapter_changes_the_answer_and_detaching_restores_it() {
        let (k, n, r) = (64usize, 32usize, 4usize);
        let mut w = weight(k, n);
        let x = Tensor::from_vec_f32(
            (0..2 * k).map(|i| ((i % 7) as f32 - 3.0) / 8.0).collect(),
            vec![2, k],
        )
        .expect("x");
        let base = w.forward(&x).expect("base").to_vec_f32();

        w.add_lora(LoraDelta {
            down: Tensor::from_vec_f32(
                (0..k * r).map(|i| ((i % 5) as f32 - 2.0) / 10.0).collect(),
                vec![k, r],
            )
            .expect("down"),
            up: Tensor::from_vec_f32(
                (0..r * n).map(|i| ((i % 3) as f32 - 1.0) / 10.0).collect(),
                vec![r, n],
            )
            .expect("up"),
            scale: 1.0,
        })
        .expect("attach");
        assert_eq!(w.lora_count(), 1);
        let adapted = w.forward(&x).expect("adapted").to_vec_f32();
        let moved = base
            .iter()
            .zip(&adapted)
            .filter(|(b, a)| (*b - *a).abs() > 1e-6)
            .count();
        assert!(moved > 0, "the adapter changed nothing");

        w.clear_lora();
        assert_eq!(w.lora_count(), 0);
        let restored = w.forward(&x).expect("restored").to_vec_f32();
        for (b, r) in base.iter().zip(&restored) {
            assert_eq!(b, r, "detaching must return the checkpoint's own answer");
        }
    }

    /// An adapter at strength zero is a request to disable it, and must leave the weight
    /// bit-identical rather than adding a zero block.
    #[test]
    fn an_adapter_at_zero_strength_is_not_applied() {
        let (k, n, r) = (32usize, 16usize, 2usize);
        let mut w = weight(k, n);
        let x = Tensor::from_vec_f32(
            (0..1 * k).map(|i| ((i % 7) as f32 - 3.0) / 8.0).collect(),
            vec![1, k],
        )
        .expect("x");
        let base = w.forward(&x).expect("base").to_vec_f32();
        let err = w
            .add_lora(LoraDelta {
                down: Tensor::from_vec_f32(
                    (0..k * r).map(|i| ((i % 5) as f32 - 2.0) / 10.0).collect(),
                    vec![k, r],
                )
                .expect("down"),
                up: Tensor::from_vec_f32(
                    (0..r * n).map(|i| ((i % 3) as f32 - 1.0) / 10.0).collect(),
                    vec![r, n],
                )
                .expect("up"),
                scale: 0.0,
            })
            .is_err();
        assert!(
            err,
            "nothing live to fuse is refused rather than half-applied"
        );
        for (b, a) in base.iter().zip(&w.forward(&x).expect("after").to_vec_f32()) {
            assert_eq!(b, a);
        }
    }

    /// A shape that cannot belong to this projection is refused, rather than fused into
    /// something that would fail much later inside a matmul.
    #[test]
    fn a_delta_of_the_wrong_shape_is_refused() {
        let mut w = weight(64, 32);
        let bad = w.add_lora(LoraDelta {
            down: Tensor::from_vec_f32(vec![0.1; 64 * 4], vec![64, 4]).expect("down"),
            up: Tensor::from_vec_f32(vec![0.1; 4 * 48], vec![4, 48]).expect("up"),
            scale: 1.0,
        });
        assert!(
            bad.is_err(),
            "48 outputs on a 32-output weight was accepted"
        );
        assert_eq!(w.lora_count(), 0);
    }

    /// The zero-allocation CPU decode path cannot see an adapter, so it must decline once one
    /// is attached. Returning its answer would silently be the base weight's.
    #[test]
    fn the_slice_path_declines_once_an_adapter_is_attached() {
        let (k, n, r) = (64usize, 32usize, 4usize);
        let mut w = weight(k, n);
        let x = vec![0.1f32; k];
        let mut out = vec![0f32; n];
        assert!(
            w.forward_slice_cpu(&x, &mut out).expect("slice"),
            "the fast path must handle a plain weight, or this test proves nothing"
        );
        w.add_lora(LoraDelta {
            down: Tensor::from_vec_f32(
                (0..k * r).map(|i| ((i % 5) as f32 - 2.0) / 10.0).collect(),
                vec![k, r],
            )
            .expect("down"),
            up: Tensor::from_vec_f32(
                (0..r * n).map(|i| ((i % 3) as f32 - 1.0) / 10.0).collect(),
                vec![r, n],
            )
            .expect("up"),
            scale: 1.0,
        })
        .expect("attach");
        assert!(!w.forward_slice_cpu(&x, &mut out).expect("slice"));
        assert!(
            w.cpu_raw().is_none(),
            "the raw weight would omit the adapter"
        );
    }
}

#[cfg(all(test, feature = "gpu"))]
mod shared_input_tests {
    use super::*;
    use crate::tensor::quantized::GgmlDType;
    use crate::tensor::Device;

    fn weight(n: usize, k: usize, salt: usize, dev: &Device) -> QMatMul {
        let w: Vec<f32> = (0..n * k)
            .map(|i| (((i * 2654435761usize + salt * 131) >> 6) % 2001) as f32 / 4000.0 - 0.25)
            .collect();
        let w = Tensor::from_vec(w, (n, k), &Device::Cpu).unwrap();
        QMatMul::from_qtensor(QTensor::quantize_onto(&w, GgmlDType::Q4K, dev).unwrap()).unwrap()
    }

    fn bits(t: &Tensor) -> Vec<u32> {
        t.to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    /// Projections of one input over one quantisation of it are bit-identical to each
    /// projection on its own, in the input's own type.
    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn projections_over_one_quantisation_match_their_own_forwards() {
        let dev = Device::new_cuda(0).expect("a card for the card test");
        let k = 512;
        let ws = [
            weight(256, k, 1, &dev),
            weight(64, k, 2, &dev),
            weight(32, k, 3, &dev),
            weight(8, k, 4, &dev),
        ];
        let refs: Vec<&QMatMul> = ws.iter().collect();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for rows in [1usize, 3] {
                let x: Vec<f32> = (0..rows * k)
                    .map(|i| (((i * 40503usize) >> 3) % 1999) as f32 / 1000.0 - 1.0)
                    .collect();
                let x = Tensor::from_vec(x, (1, rows, k), &Device::Cpu)
                    .unwrap()
                    .to_device(&dev)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                let shared = QMatMul::forward_shared(&refs, &x).unwrap();
                for (w, got) in ws.iter().zip(&shared) {
                    let own = w.forward(&x).unwrap();
                    assert_eq!(got.dtype(), own.dtype(), "{dtype:?} x {rows}: type");
                    assert_eq!(bits(got), bits(&own), "{dtype:?} x {rows}: values");
                }
            }
        }
    }
}

#[cfg(all(test, feature = "gpu"))]
mod fused_pair_tests {
    use super::*;
    use crate::tensor::quantized::{GgmlDType, QTensor};
    use crate::tensor::{DType, Device};

    /// A gate and up pair in every two of the formats the fused launch serves gives, for one
    /// row of either width, the two projections activated and multiplied.
    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn the_fused_pair_is_the_two_projections_activated() {
        let Ok(card) = crate::tensor::cuda::CudaDevice::new(0).map(Device::Cuda) else {
            return;
        };
        let (n, k) = (64usize, 2560usize);
        let weight = |seed: u32, scale: f32| -> Tensor {
            let v: Vec<f32> = (0..n * k)
                .map(|i| {
                    let h = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
                    ((h >> 8) as f32 / (1 << 24) as f32 - 0.5) * scale
                })
                .collect();
            Tensor::from_vec_f32(v, vec![n, k]).unwrap()
        };
        let x = Tensor::from_vec_f32(
            (0..k).map(|i| ((i % 53) as f32) * 0.02 - 0.5).collect(),
            vec![1, 1, k],
        )
        .unwrap()
        .to_device(&card)
        .unwrap();
        let formats = [
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
            GgmlDType::Q8_0,
        ];
        for (gd, ud) in formats
            .iter()
            .flat_map(|g| formats.iter().map(move |u| (*g, *u)))
        {
            let dtype = (gd, ud);
            let gate =
                QMatMul::from_qtensor(QTensor::quantize_onto(&weight(7, 0.14), gd, &card).unwrap())
                    .unwrap();
            let up =
                QMatMul::from_qtensor(QTensor::quantize_onto(&weight(1, 0.5), ud, &card).unwrap())
                    .unwrap();
            for half in [false, true] {
                let x = if half {
                    x.to_dtype(DType::F16).unwrap()
                } else {
                    x.clone()
                };
                let got = QMatMul::fused_silu_pair(&gate, &up, &x)
                    .unwrap()
                    .unwrap_or_else(|| panic!("{dtype:?} half {half}: the fused pair declined"));
                let g = gate.forward(&x).unwrap().to_dtype(DType::F32).unwrap();
                let u = up.forward(&x).unwrap().to_dtype(DType::F32).unwrap();
                let want = (crate::tensor::ops::silu(&g).unwrap() * u).unwrap();
                let flat = |t: &Tensor| -> Vec<f32> {
                    t.to_device(&Device::Cpu)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                };
                let (got, want) = (flat(&got), flat(&want));
                assert_eq!(got.len(), want.len(), "{dtype:?}");
                let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
                let gap = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    gap / scale < 2e-2,
                    "{dtype:?} half {half}: gap {gap} of {scale}"
                );
            }
        }
    }
}
