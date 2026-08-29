//! The production matmul over a quantised weight: which kernel runs, and why that one.
//!
//! There is no single quantised matmul. Which of them serves a call depends on the shape and
//! the dtype - a one-row decode wants a mat-vec, a prefill wants a tiled matmul, some dtypes
//! have an interleaved repack that beats both - and every one of them can decline, in which
//! case the next takes it. The dequantised matmul at the end never declines, which is what
//! makes the whole chain safe: a workspace that cannot be allocated degrades instead of
//! failing the request.

use super::mmq::{mmq_launcher, mmq_qk, mmq_quantize_launcher, mmq_supports, MMQ_WORKSPACES};
use super::*;

mod cpu;
mod kat;
mod mmq;
mod repack;

use kat::kat_bad;
pub use kat::kernel_known_answer_test;

/// Quantized matmul over the production kernels: holds the GGML block
/// weight on-device and computes `x [.., k] -> [.., n]`. Decode rows (the
/// hot path) run the mmvq GEMV, batches the tiled MMQ; on CPU (or for
/// unsupported quant types) a lazily-dequantized F32 matmul serves as the
/// correctness path.

pub struct QKernelMatMul {
    /// Low-rank adapters applied ON TOP of the quantised weight.
    ///
    /// This is why LoRA is applied rather than merged. Folding a delta into a quantised
    /// weight means dequantise, add, requantise: a dense copy in memory and a rounding
    /// pass over every affected tensor - on weights whose entire purpose is to stay
    /// small and exact. As a separate term the blob is untouched, several adapters
    /// compose by addition, and unloading one is a drop rather than a reload.
    ///
    /// Living HERE covers every family that keeps its weights quantised - Wan, Flux,
    /// Qwen-Image, Boogu - through one code path instead of four.
    pub(super) lora: Vec<crate::tensor::lora::LoraDelta>,
    /// The attached adapters fused into one pair - see `lora::fuse_loras`.
    pub(super) lora_fused: Option<(crate::tensor::Tensor, crate::tensor::Tensor)>,
    pub(super) dtype: GgmlDType,
    /// Output features (weight rows).
    pub(super) n: usize,
    /// Input features (weight cols).
    pub(super) k: usize,
    /// Dequantized-weight fallback (built lazily on first use; lives on the
    /// compute device).
    pub(super) dequant: std::sync::OnceLock<crate::tensor::Tensor>,
    pub(super) host: std::sync::Arc<QHostTensor>,
    pub(super) device: crate::tensor::Device,
    /// CPU Q4_K prefill: block-interleaved `q4_Kx8` repack of the weight, built
    /// lazily on the first multi-row (M>=4) CPU matmul and cached. The repacked
    /// GEMM reuses each loaded weight across a tile of 4 activation rows, which
    /// turns prompt processing from per-row weight re-streaming into a compute-
    /// bound GEMM (~5x on this hardware). Only allocated when a Q4_K weight is
    /// actually hit by a CPU prefill, so decode-only / CUDA weights pay nothing.
    pub(super) cpu_repack_q4k:
        std::sync::OnceLock<Vec<crate::tensor::quant_cpu::repack_q4k::BlockQ4Kx8>>,
    /// CPU Q6_K prefill: 8-column repack in the packed 6-bit planes, same
    /// weight-reuse story as `cpu_repack_q4k`. Built lazily on the first CPU
    /// prefill that hits a Q6_K weight (e.g. the Q6_K `ffn_down` rows).
    pub(super) cpu_repack_q6k:
        std::sync::OnceLock<Vec<crate::tensor::quant_cpu::repack_q6k::BlockQ6Kx8>>,
    /// CPU Q5_0 prefill: 8-column block-major repack, same weight-reuse story as
    /// the K-quant repacks. Built lazily on the first CPU prefill that hits a
    /// Q5_0 weight (the projections of the Q5-quant models).
    pub(super) cpu_repack_q5_0: std::sync::OnceLock<Vec<crate::tensor::quant_cpu::BlockQ5_0>>,
    /// CPU Q5_K prefill: 8-column repack keeping the packed 4-bit + 1-bit planes,
    /// same shape as the Q6_K repack. Built lazily on the first CPU prefill that
    /// hits a Q5_K weight (qwen3next, deepseek-r1's Q5_K rows).
    pub(super) cpu_repack_q5k:
        std::sync::OnceLock<Vec<crate::tensor::quant_cpu::repack_q5k::BlockQ5Kx8>>,
    /// CPU Q4_0 decode: row-grouped (8-wide) repack for the amortised decode
    /// GEMV - Q4_0 is the only quant whose decode otherwise re-streams the
    /// activation per output column (mistral-nemo, the fleet's dense Q4_0).
    /// Built lazily on the first CPU decode, same story as `cpu_repack_q4k`.
    pub(super) cpu_repack_q4_0: std::sync::OnceLock<Vec<crate::tensor::quant_cpu::BlockQ4_0>>,
    /// Column-interleaved MXFP4 repack (`block_mxfp4x8`) for the denser prefill
    /// GEMM (8 columns in the SIMD lanes, no per-column hsum). Built lazily.
    pub(super) cpu_repack_mxfp4_x8:
        std::sync::OnceLock<Vec<crate::tensor::quant_cpu::repack_mxfp4_x8::BlockMxFp4x8>>,
    /// Column-interleaved Q8_0 repack (`block_q8_0x8`) - same denser prefill GEMM
    /// generalized to Q8_0 (fleet-wide: many small models + attention). Lazy.
    pub(super) cpu_repack_q8_0_x8:
        std::sync::OnceLock<Vec<crate::tensor::quant_cpu::repack_q8_0_x8::BlockQ8_0x8>>,
    /// Arc-shared so the compat shim's `QCudaStorage` projection
    /// and this QKernelMatMul reuse ONE padded device upload per weight.
    #[cfg(feature = "cuda")]
    pub(super) blob: Option<std::sync::Arc<cudarc::driver::CudaSlice<u8>>>,
    /// The room a placement on a counting device would give this weight.
    ///
    /// A dry device stands in for a card, and what a card gives up for a quantised
    /// weight is the padded blob the constructor uploads. The entry is taken at the
    /// moment the upload would happen and released by the drop that would have freed
    /// it, so a weight costs the ledger exactly as long as it costs the card. `None`
    /// on every real placement, where the blob itself is the record.
    ///
    /// Held, never read: the entry is made by its constructor and withdrawn by its
    /// drop, so the value carries nothing a caller would want.
    #[allow(dead_code)]
    pub(super) dry: Option<crate::tensor::dry::DryStorage>,
}

/// Per-device kernel families that failed their boot known-answer test. A family in the
/// set is never dispatched on that device; its call sites fall through to the dequant +
/// dense matmul path, which is slower and correct. The failure this exists for is an
/// arch-mismatched kernel returning garbage without an error - twice now that garbage
/// was NaN and cost a day each time; wrong-but-finite would be worse.
#[cfg(feature = "cuda")]
static KAT_BAD_MMVQ: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<usize>>> =
    std::sync::OnceLock::new();
#[cfg(feature = "cuda")]
static KAT_BAD_MMQ: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<usize>>> =
    std::sync::OnceLock::new();

impl QKernelMatMul {
    /// Place the weight on `device` (CUDA: blob uploaded for the kernel
    /// paths; CPU: host blocks kept for the dequant path).
    pub fn from_qtensor_on(
        qt: std::sync::Arc<QHostTensor>,
        device: &crate::tensor::Device,
    ) -> Result<Self> {
        #[cfg(feature = "cuda")]
        let blob = match device {
            crate::tensor::Device::Cuda(dev) => {
                // The MMQ/MMVQ kernels assume row lengths padded to
                // MATRIX_ROW_PADDING (512) elements and may read past the
                // logical end of the LAST row when k % 512 != 0 (flux:
                // img_in k=64, time_in k=256). Upload into a zeroed
                // allocation with a tail margin covering the worst-case
                // over-read for every block dtype.
                let data = qt.data();
                let stream = dev.stream();
                let mut padded = crate::tensor::cuda::with_oom_retry(dev, "QKernelMatMul", || {
                    stream.alloc_zeros::<u8>(data.len() + BLOB_TAIL_PAD_BYTES)
                })?;
                stream
                    .memcpy_htod(data, &mut padded.slice_mut(0..data.len()))
                    .map_err(|e| Error(format!("QKernelMatMul upload: {e}")))?;
                Some(std::sync::Arc::new(padded))
            }
            _ => None,
        };
        if qt.dims.len() != 2 {
            return Err(Error(format!(
                "QKernelMatMul: weight must be 2-D, got {:?}",
                qt.dims
            )));
        }
        let (n, k) = (qt.dims[0], qt.dims[1]);
        // This constructor is where the upload happens, so it is where a counting
        // device is charged for it.
        let dry = dry_blob_room(&qt, device);
        let qmm = Self {
            lora: Vec::new(),
            lora_fused: None,
            dtype: qt.dtype,
            n,
            k,
            dequant: std::sync::OnceLock::new(),
            cpu_repack_q4k: std::sync::OnceLock::new(),
            cpu_repack_q6k: std::sync::OnceLock::new(),
            cpu_repack_q5_0: std::sync::OnceLock::new(),
            cpu_repack_q5k: std::sync::OnceLock::new(),
            cpu_repack_q4_0: std::sync::OnceLock::new(),
            cpu_repack_mxfp4_x8: std::sync::OnceLock::new(),
            cpu_repack_q8_0_x8: std::sync::OnceLock::new(),
            host: qt,
            device: device.clone(),
            #[cfg(feature = "cuda")]
            blob,
            dry,
        };
        // In-place repack at LOAD: CPU Q4_K weights build their unified interleaved
        // copy up front (off the first token), and every matmul reads it - the
        // canonical bytes stay for correctness paths only.
        #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
        if matches!(qmm.device, crate::tensor::Device::Cpu) {
            let _ = qmm.q4k_x8();
        }
        Ok(qmm)
    }

    #[cfg(feature = "cuda")]
    pub fn from_qtensor(
        qt: QHostTensor,
        dev: std::sync::Arc<crate::tensor::cuda::CudaDevice>,
    ) -> Result<Self> {
        Self::from_qtensor_on(std::sync::Arc::new(qt), &crate::tensor::Device::Cuda(dev))
    }

    /// Build over an EXISTING padded device blob (compat shim -  the
    /// blob was already uploaded for the `QCudaStorage` projection; sharing it
    /// avoids a second per-weight VRAM copy). The blob must be the tail-padded
    /// upload of `qt.data()`. `blob = None` on CPU placements.
    #[cfg(feature = "cuda")]
    pub fn from_qtensor_on_with_blob(
        qt: std::sync::Arc<QHostTensor>,
        device: &crate::tensor::Device,
        blob: Option<std::sync::Arc<cudarc::driver::CudaSlice<u8>>>,
    ) -> Result<Self> {
        if qt.dims.len() != 2 {
            return Err(Error(format!(
                "QKernelMatMul: weight must be 2-D, got {:?}",
                qt.dims
            )));
        }
        if matches!(device, crate::tensor::Device::Cuda(_)) && blob.is_none() {
            // CUDA placement without a shared blob: fall back to the
            // uploading constructor.
            return Self::from_qtensor_on(qt, device);
        }
        let (n, k) = (qt.dims[0], qt.dims[1]);
        let qmm = Self {
            lora: Vec::new(),
            lora_fused: None,
            dtype: qt.dtype,
            n,
            k,
            dequant: std::sync::OnceLock::new(),
            cpu_repack_q4k: std::sync::OnceLock::new(),
            cpu_repack_q6k: std::sync::OnceLock::new(),
            cpu_repack_q5_0: std::sync::OnceLock::new(),
            cpu_repack_q5k: std::sync::OnceLock::new(),
            cpu_repack_q4_0: std::sync::OnceLock::new(),
            cpu_repack_mxfp4_x8: std::sync::OnceLock::new(),
            cpu_repack_q8_0_x8: std::sync::OnceLock::new(),
            host: qt,
            device: device.clone(),
            blob,
            // No entry: this constructor uploads nothing. The blob it is handed was
            // allocated by whoever owns it, and on a counting device that owner - the
            // [`QTensor`] holding these same blocks - is already charged for it.
            // Charging again here would count one weight twice.
            dry: None,
        };
        // In-place repack at LOAD (see the sibling constructor).
        #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
        if matches!(qmm.device, crate::tensor::Device::Cpu) {
            let _ = qmm.q4k_x8();
        }
        Ok(qmm)
    }

    fn kernel_tag(&self) -> Result<&'static str> {
        Ok(match self.dtype {
            GgmlDType::Q4_0 => "q4_0",
            GgmlDType::Q4_1 => "q4_1",
            GgmlDType::Q5_0 => "q5_0",
            GgmlDType::Q5_1 => "q5_1",
            GgmlDType::Q2K => "q2_k",
            GgmlDType::Q3K => "q3_k",
            GgmlDType::Q4K => "q4_k",
            GgmlDType::Q5K => "q5_k",
            GgmlDType::Q6K => "q6_k",
            GgmlDType::Q8_0 => "q8_0",
            other => {
                return Err(Error(format!(
                    "QKernelMatMul: no mmvq kernel for {other:?}"
                )))
            }
        })
    }

    /// `x`: [.., k] on the same device -> [.., n].
    /// Force an exact F32 matmul: dequantize the weight per call and accumulate in F32. The
    /// quantized MMQ/MMVQ path quantizes the activation and can overflow fp16 at very large
    /// activation magnitudes; the Qwen-Image DiT legitimately produces those (AdaLN scale ~477),
    /// so its blocks route through here. Slow (per-call dequant, no VRAM cache) - offline render.
    pub fn forward_dequant_f32(&self, x: &crate::tensor::Tensor) -> Result<crate::tensor::Tensor> {
        // The dense copy is the largest thing this call holds, and it is held for the
        // length of the matmul. On a counting device it is charged and never built:
        // decoding a weight into a host vector to answer a question about placement is
        // the one allocation a dry run promises not to make.
        let wt = if x.device().is_dry() {
            crate::tensor::Tensor::dry(
                &x.device(),
                crate::tensor::DType::F32,
                (self.n, self.k),
            )?
        } else {
            let w = self.host.dequantize_f32()?; // flat [n.k], weight rows [n, k]
            crate::tensor::Tensor::from_vec_f32(w, (self.n, self.k))?.to_device(&x.device())?
        };
        let x32 = x.to_dtype(crate::tensor::DType::F32)?;
        x32.matmul_t(&wt)
    }

    /// GPU-dequant weight -> dense F32 on-device -> parallel cuBLAS F32 GEMM. High GPU util AND F32
    /// accumulation (holds the DiT's ~1e9 activations that overflow the quantized MMQ path). No
    /// weight cache -> no OOM; the dequant buffer is freed after the matmul. Falls back to the CPU
    /// dequant for non-K-quant or CPU tensors. Q4_K_M DiT = Q4_K/Q5_K/Q6_K.
    /// Dequantize this weight ONCE into a dense BF16 device tensor `[n, k]`, for
    /// callers that run several GEMMs against the same weight (e.g. a token-chunked
    /// MLP). `Ok(None)` when the weight is not a GPU K-quant/Q8_0 blob - the caller
    /// then falls back to per-call `forward*`. BF16 is the DiT's reference GEMM
    /// precision (holds ~1e9 activations, cuBLAS accumulates in F32).
    #[cfg(feature = "cuda")]
    pub fn dequant_weight_bf16(&self) -> Result<Option<crate::tensor::Tensor>> {
        let (crate::tensor::Device::Cuda(qdev), Some(blob)) = (&self.device, self.blob.as_ref())
        else {
            return Ok(None);
        };
        let total = self.n * self.k;
        let w = match self.dtype {
            GgmlDType::Q4K => Some(crate::tensor::cuda::dequantize_kquant_f32(
                qdev,
                "dequantize_block_q4_K_f32",
                blob,
                total,
                32,
            )?),
            GgmlDType::Q5K => Some(crate::tensor::cuda::dequantize_kquant_f32(
                qdev,
                "dequantize_block_q5_K_f32",
                blob,
                total,
                64,
            )?),
            GgmlDType::Q6K => Some(crate::tensor::cuda::dequantize_kquant_f32(
                qdev,
                "dequantize_block_q6_K_f32",
                blob,
                total,
                64,
            )?),
            GgmlDType::Q8_0 => Some(crate::tensor::cuda::dequantize_q8_0_f32(qdev, blob, total)?),
            _ => None,
        };
        match w {
            Some(w) => {
                let wt = crate::tensor::Tensor::from_cuda_storage(
                    crate::tensor::cuda::CudaStorage::F32(w),
                    qdev.clone(),
                    vec![self.n, self.k],
                )?;
                Ok(Some(wt.to_dtype(crate::tensor::DType::BF16)?))
            }
            None => Ok(None),
        }
    }

    pub fn forward_dequant_gpu(&self, x: &crate::tensor::Tensor) -> Result<crate::tensor::Tensor> {
        #[cfg(feature = "cuda")]
        if let (crate::tensor::Device::Cuda(qdev), Some(blob)) = (&self.device, self.blob.as_ref())
        {
            let total = self.n * self.k;
            // GPU-dequant the weight to a dense F32 device buffer. K-quants use the 256-superblock
            // kernels; Q8_0 (32-wide blocks) uses its own launcher. Anything unsupported yields None
            // and falls through to the CPU path below - callers on the GPU hot path MUST use a dtype
            // handled here, else the "GPU" matmul silently runs on CPU.
            let w = match self.dtype {
                GgmlDType::Q4K => Some(crate::tensor::cuda::dequantize_kquant_f32(
                    qdev,
                    "dequantize_block_q4_K_f32",
                    blob,
                    total,
                    32,
                )?),
                GgmlDType::Q5K => Some(crate::tensor::cuda::dequantize_kquant_f32(
                    qdev,
                    "dequantize_block_q5_K_f32",
                    blob,
                    total,
                    64,
                )?),
                GgmlDType::Q6K => Some(crate::tensor::cuda::dequantize_kquant_f32(
                    qdev,
                    "dequantize_block_q6_K_f32",
                    blob,
                    total,
                    64,
                )?),
                GgmlDType::Q8_0 => {
                    Some(crate::tensor::cuda::dequantize_q8_0_f32(qdev, blob, total)?)
                }
                _ => None,
            };
            if let Some(w) = w {
                let wt = crate::tensor::Tensor::from_cuda_storage(
                    crate::tensor::cuda::CudaStorage::F32(w),
                    qdev.clone(),
                    vec![self.n, self.k],
                )?;
                // BF16 on the tensor cores: its eight-bit exponent holds the DiT's ~1e9
                // activations where f16 overflows, cuBLAS accumulates the products in F32 anyway,
                // and it is four to eight times an F32 GEMM.
                if x.dtype() == crate::tensor::DType::BF16 {
                    return x.matmul_t(&wt.to_dtype(crate::tensor::DType::BF16)?);
                }
                let xb = x.to_dtype(crate::tensor::DType::BF16)?;
                let y = xb.matmul_t(&wt.to_dtype(crate::tensor::DType::BF16)?)?;
                return y.to_dtype(crate::tensor::DType::F32);
            }
        }
        self.forward_dequant_f32(x)
    }

    /// Attach a low-rank correction on top of the quantised weight.
    pub fn add_lora(&mut self, delta: crate::tensor::lora::LoraDelta) -> Result<()> {
        let (din, r) = delta.down.shape().dims2()?;
        let (r2, dout) = delta.up.shape().dims2()?;
        if din != self.k || dout != self.n || r != r2 {
            return Err(Error(format!(
                "lora: down {din}x{r} up {r2}x{dout} does not fit a {}x{} projection",
                self.k, self.n
            )));
        }
        self.lora.push(delta);
        self.lora_fused = crate::tensor::lora::fuse_loras(&self.lora, self.k, self.n);
        Ok(())
    }

    /// Drop every attached adapter, returning this weight to the checkpoint.
    pub fn clear_lora(&mut self) {
        self.lora.clear();
        self.lora_fused = None;
    }

    pub fn lora_count(&self) -> usize {
        self.lora.len()
    }

    /// Quantised matmul, plus any attached low-rank terms.
    ///
    /// Wrapping the base rather than editing it: the quantised forward has several
    /// return paths (a CPU k-quant fast path and multiple CUDA kernels), and adding the
    /// delta at each one is how a family silently ends up without its adapter.
    pub fn forward(&self, x: &crate::tensor::Tensor) -> Result<crate::tensor::Tensor> {
        let base = self.forward_base(x)?;
        let Some((down, up)) = &self.lora_fused else {
            return Ok(base);
        };
        let xdims = x.dims().to_vec();
        let k = *xdims
            .last()
            .ok_or_else(|| Error("lora: rank-0 input".into()))?;
        let rows = x.elem_count() / k;
        // The adapters are dense f32 while the activation may be a half carrier, so the
        // conversion happens for THIS term only - the quantised path is untouched.
        let xf = x
            .to_dtype(crate::tensor::DType::F32)?
            .reshape(vec![1, rows, k])?;
        let acc = Some(xf.matmul(down)?.matmul(up)?);
        let mut odims = xdims;
        *odims.last_mut().unwrap() = self.n;
        let delta = acc.unwrap().reshape(odims)?.to_dtype(base.dtype())?;
        base.add(&delta)
    }

    fn forward_base(&self, x: &crate::tensor::Tensor) -> Result<crate::tensor::Tensor> {
        let xdims = x.dims().to_vec();
        let k = *xdims
            .last()
            .ok_or_else(|| Error("QKernelMatMul: rank-0 input".into()))?;
        if k != self.k {
            return Err(Error(format!(
                "QKernelMatMul: input k {k} != weight k {}",
                self.k
            )));
        }
        let rows = x.elem_count() / k;
        let mut odims = xdims.clone();
        *odims.last_mut().unwrap() = self.n;

        // A weight that lives on a ledger. Every path below - the host dot engine, the
        // mat-vec kernels, the tiled one, the dequantised fallback - leaves the same
        // thing behind at this layer: one buffer of `odims` carrying the activation's
        // dtype. So which of them would have served the call does not change the count,
        // and it cannot be decided here anyway: a dry device reports no card, and a
        // kernel choice made from that would be the choice of a machine that is not the
        // one the placement is for.
        //
        // What this does NOT see is what the kernels take below the tensor layer: the
        // q8_1 staging the activation is quantised into, and the tiled path's shared
        // per-device workspace. Both are real device memory and neither passes through
        // here - see the module doc on what the count is a lower bound of.
        if self.device.is_dry() {
            if !x.device().same_device(&self.device) {
                return Err(Error(
                    "QKernelMatMul: a counted weight cannot serve an activation from \
                     another device"
                        .into(),
                ));
            }
            let odtype = match x.dtype() {
                crate::tensor::DType::F16 | crate::tensor::DType::BF16 => x.dtype(),
                _ => crate::tensor::DType::F32,
            };
            return crate::tensor::Tensor::dry(&self.device, odtype, odims);
        }

        // CPU fast path: the lifted k_quants dot engine - the
        // activation is Q8-quantized once per row and the output columns run
        // rayon-parallel, bit-exact with the fork's CPU matmul. This replaces
        // the dequant-f32 fallback for every supported dtype (without it the
        // facade flip would regress CPU/hybrid decode 10-100x).
        if matches!(self.device, crate::tensor::Device::Cpu)
            && crate::tensor::quant_cpu::supports(self.dtype)
            && self.k % self.dtype.block_size() == 0
        {
            // Output carries the INPUT's dtype (
            // same as the CUDA branches below): a half-carrier model's
            // residual stream must come back half, or the caller's
            // `residual + h` add sees a mixed f16/f32 pair and errors.
            // f16 activations run the direct f16 kernel - no f32 staging
            // buffer in and no whole-output convert pass back out.
            if x.dtype() == crate::tensor::DType::F16 {
                if let Ok(lhs) = x.cpu_f16_data() {
                    // Prefill routes through the interleaved repack GEMM, which reuses a
                    // loaded weight tile across four activation rows; decode (rows<4) and
                    // the formats without a repack decline and take the GEMV below.
                    if let Some(out) = self.try_repack_gemm(lhs, rows, &odims) {
                        return out;
                    }
                    let mut dst = vec![half::f16::ZERO; rows * self.n];
                    crate::tensor::quant_cpu::matmul_f16_bytes(
                        self.dtype,
                        (rows, self.k, self.n),
                        lhs,
                        self.host.data(),
                        &mut dst,
                    )?;
                    return crate::tensor::Tensor::from_storage(
                        crate::tensor::CpuStorage::F16(dst),
                        odims,
                    );
                }
            }
            // Borrow contiguous f32 activations in place; only convert
            // (one staging copy) for the remaining dtypes.
            let staged;
            let lhs: &[f32] = match x.cpu_f32_data() {
                Ok(s) => s,
                Err(_) => {
                    staged = x.to_vec_f32();
                    &staged
                }
            };
            let mut dst = vec![0f32; rows * self.n];
            // CPU Q4_K prefill: the block-interleaved tiled GEMM reuses each
            // loaded weight across 4 activation rows (~5x vs the per-row
            // weight re-stream of `matmul_bytes`). Decode (rows<4) keeps the
            // bandwidth-bound per-column path. Repack is built once, cached.
            let nb = self.k / 256;
            // The tiled Q4_K prefill GEMM applies only when the mmap'd weight is
            // exactly n*nb aligned whole Q4_K blocks (GGUF Q4_K is unpadded and
            // 2-byte aligned); otherwise fall through to the per-column path.
            // Q4_K serves EVERY row count from the unified repack (M=1 interleaved
            // GEMV, M>=2 tiled GEMM) - the in-place storage path.
            let q4k_prefill = self.dtype == GgmlDType::Q4K
                && self.n % 8 == 0
                && self.k % 256 == 0
                && crate::tensor::quant_cpu::cast_blocks::<crate::tensor::quant_cpu::BlockQ4K>(
                    self.host.data(),
                )
                .map(|b| b.len() == self.n * nb)
                .unwrap_or(false);
            // CPU Q4_0 decode (rows<4): the per-column dot re-streams the
            // activation and serialises on one fmadd chain; the row-grouped
            // repack amortises both 8-way (the measured mistral-nemo CPU gap  - 
            // Q4_0 was the only quant without an amortised decode path).
            let nb40 = self.k / 32;
            let q4_0_decode = self.dtype == GgmlDType::Q4_0
                && rows < 4
                && self.n % 8 == 0
                && self.k % 32 == 0
                && crate::tensor::quant_cpu::cast_blocks::<crate::tensor::quant_cpu::BlockQ4_0>(
                    self.host.data(),
                )
                .map(|b| b.len() == self.n * nb40)
                .unwrap_or(false);
            // Q4_0 (mistral-nemo, moondream) gets the same tiled weight-reuse at
            // prefill; its per-column path re-streams every 4-bit weight block
            // once per prompt row. Decode (rows<4) keeps the row-grouped GEMV.
            let q4_0_prefill = self.dtype == GgmlDType::Q4_0
                && rows >= 4
                && self.n % 8 == 0
                && self.k % 32 == 0
                && crate::tensor::quant_cpu::cast_blocks::<crate::tensor::quant_cpu::BlockQ4_0>(
                    self.host.data(),
                )
                .map(|b| b.len() == self.n * nb40)
                .unwrap_or(false);
            // Q6_K (e.g. ffn_down) gets the same tiled weight-reuse as Q4_K on
            // the F32-activation path; without it the per-column path re-decodes
            // each weight once per prompt row, the bulk of its prefill cost.
            let q6k_prefill = self.dtype == GgmlDType::Q6K
                && rows >= 4
                && self.n % 8 == 0
                && self.k % 256 == 0
                && crate::tensor::quant_cpu::cast_blocks::<crate::tensor::quant_cpu::BlockQ6K>(
                    self.host.data(),
                )
                .map(|b| b.len() == self.n * nb)
                .unwrap_or(false);
            // Q5_0 (the projections of the Q5-quant models) gets the same tiled
            // weight-reuse; its per-column path is doubly penalised (no reuse and
            // a 32-weight block, so the high-bit decode is paid far more often).
            let q5_0_prefill = self.dtype == GgmlDType::Q5_0
                && rows >= 4
                && self.n % 8 == 0
                && self.k % 32 == 0
                && crate::tensor::quant_cpu::cast_blocks::<crate::tensor::quant_cpu::BlockQ5_0>(
                    self.host.data(),
                )
                .map(|b| b.len() == self.n * nb40)
                .unwrap_or(false);
            let q5k_prefill = self.dtype == GgmlDType::Q5K
                && rows >= 4
                && self.n % 8 == 0
                && self.k % 256 == 0
                && crate::tensor::quant_cpu::cast_blocks::<crate::tensor::quant_cpu::BlockQ5K>(
                    self.host.data(),
                )
                .map(|b| b.len() == self.n * nb)
                .unwrap_or(false);
            // Q8_0 (the 8-bit small models: llama3.2:1b, ernie4-5, falcon3) gets
            // the same tiled weight-reuse; its per-column path re-streams every
            // 1-byte weight block once per prompt row.
            let q8_0_prefill = self.dtype == GgmlDType::Q8_0
                && rows >= 4
                && self.n % 8 == 0
                && self.k % 32 == 0
                && crate::tensor::quant_cpu::cast_blocks::<crate::tensor::quant_cpu::BlockQ8_0>(
                    self.host.data(),
                )
                .map(|b| b.len() == self.n * nb40)
                .unwrap_or(false);
            // MXFP4 (gpt-oss, whole model - attention proj + MoE experts) gets the
            // same tiled weight-reuse; its per-column path re-streams and re-decodes
            // every 4-bit E2M1 block once per prompt row, the bulk of prefill cost.
            let mxfp4_prefill = self.dtype == GgmlDType::MxFp4
                && rows >= 4
                && self.n % 8 == 0
                && self.k % 32 == 0
                && crate::tensor::quant_cpu::cast_blocks::<crate::tensor::quant_cpu::BlockMxFp4>(
                    self.host.data(),
                )
                .map(|b| b.len() == self.n * nb40)
                .unwrap_or(false);
            if q4k_prefill {
                let x8 = self
                    .q4k_x8()
                    .expect("q4k_prefill guard implies eligibility");
                crate::tensor::quant_cpu::matmul_q4k_plain(
                    (rows, self.k, self.n),
                    lhs,
                    x8,
                    &mut dst,
                )?;
            } else if q6k_prefill {
                let x8 = self.cpu_repack_q6k.get_or_init(|| {
                    let blocks = crate::tensor::quant_cpu::cast_blocks::<
                        crate::tensor::quant_cpu::BlockQ6K,
                    >(self.host.data())
                    .expect("q6k_prefill guard verified the cast");
                    crate::tensor::quant_cpu::repack_q6k::repack(blocks, self.n, nb)
                });
                crate::tensor::quant_cpu::matmul_q6k_repacked_tiled(
                    (rows, self.k, self.n),
                    lhs,
                    x8,
                    &mut dst,
                )?;
            } else if q5k_prefill {
                let x8 = self.cpu_repack_q5k.get_or_init(|| {
                    let blocks = crate::tensor::quant_cpu::cast_blocks::<
                        crate::tensor::quant_cpu::BlockQ5K,
                    >(self.host.data())
                    .expect("q5k_prefill guard verified the cast");
                    crate::tensor::quant_cpu::repack_q5k::repack(blocks, self.n, nb)
                });
                crate::tensor::quant_cpu::matmul_q5k_repacked_tiled(
                    (rows, self.k, self.n),
                    lhs,
                    x8,
                    &mut dst,
                )?;
            } else if q5_0_prefill {
                let g8 = self.cpu_repack_q5_0.get_or_init(|| {
                    let blocks = crate::tensor::quant_cpu::cast_blocks::<
                        crate::tensor::quant_cpu::BlockQ5_0,
                    >(self.host.data())
                    .expect("q5_0_prefill guard verified the cast");
                    crate::tensor::quant_cpu::repack_q5_0::repack(blocks, self.n, nb40)
                });
                crate::tensor::quant_cpu::matmul_q5_0_repacked_tiled(
                    (rows, self.k, self.n),
                    lhs,
                    g8,
                    &mut dst,
                )?;
            } else if q8_0_prefill {
                let x8 = self.cpu_repack_q8_0_x8.get_or_init(|| {
                    let blocks = crate::tensor::quant_cpu::cast_blocks::<
                        crate::tensor::quant_cpu::BlockQ8_0,
                    >(self.host.data())
                    .expect("q8_0_prefill guard verified the cast");
                    crate::tensor::quant_cpu::repack_q8_0_x8::repack(blocks, self.n, nb40)
                });
                crate::tensor::quant_cpu::matmul_q8_0_x8_tiled(
                    (rows, self.k, self.n),
                    lhs,
                    x8,
                    &mut dst,
                )?;
            } else if q4_0_prefill {
                let g8 = self.cpu_repack_q4_0.get_or_init(|| {
                    let blocks = crate::tensor::quant_cpu::cast_blocks::<
                        crate::tensor::quant_cpu::BlockQ4_0,
                    >(self.host.data())
                    .expect("q4_0_prefill guard verified the cast");
                    crate::tensor::quant_cpu::repack_q4_0::repack(blocks, self.n, nb40)
                });
                crate::tensor::quant_cpu::matmul_q4_0_repacked_tiled(
                    (rows, self.k, self.n),
                    lhs,
                    g8,
                    &mut dst,
                )?;
            } else if mxfp4_prefill {
                let x8 = self.cpu_repack_mxfp4_x8.get_or_init(|| {
                    let blocks = crate::tensor::quant_cpu::cast_blocks::<
                        crate::tensor::quant_cpu::BlockMxFp4,
                    >(self.host.data())
                    .expect("mxfp4_prefill guard verified the cast");
                    crate::tensor::quant_cpu::repack_mxfp4_x8::repack(blocks, self.n, nb40)
                });
                crate::tensor::quant_cpu::matmul_mxfp4_x8_tiled(
                    (rows, self.k, self.n),
                    lhs,
                    x8,
                    &mut dst,
                )?;
            } else if q4_0_decode {
                let g8 = self.cpu_repack_q4_0.get_or_init(|| {
                    let blocks = crate::tensor::quant_cpu::cast_blocks::<
                        crate::tensor::quant_cpu::BlockQ4_0,
                    >(self.host.data())
                    .expect("q4_0_decode guard verified the cast");
                    crate::tensor::quant_cpu::repack_q4_0::repack(blocks, self.n, nb40)
                });
                crate::tensor::quant_cpu::matmul_q4_0_repacked_gemv(
                    (rows, self.k, self.n),
                    lhs,
                    g8,
                    &mut dst,
                )?;
            } else {
                crate::tensor::quant_cpu::matmul_bytes(
                    self.dtype,
                    (rows, self.k, self.n),
                    lhs,
                    self.host.data(),
                    &mut dst,
                )?;
            }
            let y = crate::tensor::Tensor::from_vec_f32(dst, odims)?;
            return match x.dtype() {
                crate::tensor::DType::F16 | crate::tensor::DType::BF16 => y.to_dtype(x.dtype()),
                _ => Ok(y),
            };
        }

        #[cfg(feature = "cuda")]
        if let (crate::tensor::Device::Cuda(qdev), Some(blob)) = (&self.device, self.blob.as_ref())
        {
            let xdt = x.dtype();
            // MMVQ (GEMV-family) vs MMQ (tiled GEMM) crossover. MMVQ wins for very
            // small batches; MMQ's weight-reuse wins once the batch is wide enough
            // to amortize the load - an A/B (mistral-nemo, CB mid-concurrency)
            // showed rows>=6 gain ~+25% on MMQ (N=8 270->337 tok/s), rows<=5 stay on
            // MMVQ. But ONLY drop to 5 when MMQ can actually run this quant - else
            // rows 6-8 would fall through to the slow dequant-F32 path, so keep the
            // old MMVQ-through-8 boundary for non-MMQ quants (no regression).
            let mmq_capable = mmq_supports(self.dtype) && self.k % mmq_qk(self.dtype) == 0;
            let mmvq_max = if mmq_capable { 5 } else { 8 };
            if (1..=mmvq_max).contains(&rows)
                && !kat_bad(&KAT_BAD_MMVQ, qdev.ordinal())
                && self.kernel_tag().is_ok()
                && matches!(
                    xdt,
                    crate::tensor::DType::F32
                        | crate::tensor::DType::F16
                        | crate::tensor::DType::BF16
                )
            {
                // hot decode path (rows=1) + small batches (rows 2..=8  - 
                // forward_all/PLD/short prefill): production batched MMVQ,
                // the same dispatch boundary as the fork's fast_mmvq
                // (MMVQ_MAX_BATCH=8). MMQ only beyond that, like the facade.
                // Activation dtype dispatch mirrors fast_mmvq::try_fwd:
                // F32/F16/BF16 each quantize directly and produce output in
                // the SAME dtype (f16-carrier models - qwen3.5 DeltaNet  - 
                // stay in half end-to-end, no f32 detour).
                let tag = self.kernel_tag()?;
                match xdt {
                    crate::tensor::DType::F32 => {
                        let (x_slice, dev) = x.cuda_f32_slice()?;
                        if dev.ordinal() != qdev.ordinal() {
                            return Err(Error(
                                "QKernelMatMul: input on a different cuda device".into(),
                            ));
                        }
                        let q81 = crate::tensor::cuda::quantize_q8_1_rows(qdev, x_slice, k, rows)?;
                        // smallk (4-rows/block decode kernel) - same ggml dispatch as the
                        // F16 path above: decode-only (rows==1), n%4==0, K-quant with a
                        // smallk entry, small K ((k/256) < 4*vdr). Tiny-model GEMVs (small K)
                        // otherwise run the 1-row/block kernel at only ~27% BW peak.
                        let smallk_vdr: Option<i32> = match self.dtype {
                            GgmlDType::Q4K | GgmlDType::Q5K => Some(2),
                            GgmlDType::Q6K => Some(1),
                            _ => None,
                        };
                        let smallk = rows == 1
                            && self.n % 4 == 0
                            && smallk_vdr.is_some_and(|vdr| (k as i32 / 256) < 4 * vdr);
                        let y = crate::tensor::cuda::mmvq_f32(
                            qdev, tag, blob, &q81, k, self.n, rows, smallk,
                        )?;
                        return crate::tensor::Tensor::from_cuda_storage(
                            crate::tensor::cuda::CudaStorage::F32(y),
                            qdev.clone(),
                            odims,
                        );
                    }
                    crate::tensor::DType::F16 => {
                        let (x_slice, dev) = x.cuda_f16_slice()?;
                        if dev.ordinal() != qdev.ordinal() {
                            return Err(Error(
                                "QKernelMatMul: input on a different cuda device".into(),
                            ));
                        }
                        let q81 =
                            crate::tensor::cuda::quantize_q8_1_rows_f16(qdev, x_slice, k, rows)?;
                        // ggml small_k dispatch (fork fast_mmvq.rs): decode-only
                        // (b_size==1), K-quants with a smallk entry, nrows%4==0,
                        // and (k/256) < 4*vdr.
                        let smallk_vdr: Option<i32> = match self.dtype {
                            GgmlDType::Q4K | GgmlDType::Q5K => Some(2),
                            GgmlDType::Q6K => Some(1),
                            _ => None,
                        };
                        let smallk = rows == 1
                            && self.n % 4 == 0
                            && smallk_vdr.is_some_and(|vdr| (k as i32 / 256) < 4 * vdr);
                        let y = crate::tensor::cuda::mmvq_f16(
                            qdev, tag, blob, &q81, k, self.n, rows, smallk,
                        )?;
                        return crate::tensor::Tensor::from_cuda_storage(
                            crate::tensor::cuda::CudaStorage::F16(y),
                            qdev.clone(),
                            odims,
                        );
                    }
                    crate::tensor::DType::BF16 => {
                        let (x_slice, dev) = x.cuda_bf16_slice()?;
                        if dev.ordinal() != qdev.ordinal() {
                            return Err(Error(
                                "QKernelMatMul: input on a different cuda device".into(),
                            ));
                        }
                        let q81 =
                            crate::tensor::cuda::quantize_q8_1_rows_bf16(qdev, x_slice, k, rows)?;
                        let y =
                            crate::tensor::cuda::mmvq_bf16(qdev, tag, blob, &q81, k, self.n, rows)?;
                        return crate::tensor::Tensor::from_cuda_storage(
                            crate::tensor::cuda::CudaStorage::BF16(y),
                            qdev.clone(),
                            odims,
                        );
                    }
                    _ => unreachable!("gated by the matches! above"),
                }
            }

            // batch>8: tiled quantized matmul (the production prefill MMQ
            // kernel family) when the dtype supports it... MMQ quantizes from
            // f32: half inputs take a device cast in and back out, exactly
            // like the fork's fast_mmq::try_fwd.
            if rows > 1
                && mmq_supports(self.dtype)
                && self.k % mmq_qk(self.dtype) == 0
                && !kat_bad(&KAT_BAD_MMQ, qdev.ordinal())
            {
                if matches!(xdt, crate::tensor::DType::F16 | crate::tensor::DType::BF16) {
                    let x32 = x.to_dtype(crate::tensor::DType::F32)?;
                    let (x_slice, dev) = x32.cuda_f32_slice()?;
                    if dev.ordinal() != qdev.ordinal() {
                        return Err(Error(
                            "QKernelMatMul: input on a different cuda device".into(),
                        ));
                    }
                    // MMQ is the FAST path, not the only one: the dequantised matmul
                    // below needs no per-call workspace. So a workspace that cannot be
                    // had - even after the reclaim - degrades to it instead of failing the
                    // request. An OOM must never reach the caller while a slower path is
                    // still available.
                    match self.mmq_forward(qdev, blob, x_slice, rows) {
                        Ok(y) => {
                            let out = crate::tensor::Tensor::from_cuda_storage(
                                crate::tensor::cuda::CudaStorage::F32(y),
                                qdev.clone(),
                                odims,
                            )?;
                            return out.to_dtype(xdt);
                        }
                        Err(e) if e.is_oom() => {
                            tracing::warn!(
                                "mmq unavailable ({e}); this matmul takes the dequantised \
                                 path instead of failing"
                            );
                        }
                        Err(e) => return Err(e),
                    }
                }
                if xdt == crate::tensor::DType::F32 {
                    let (x_slice, dev) = x.cuda_f32_slice()?;
                    if dev.ordinal() != qdev.ordinal() {
                        return Err(Error(
                            "QKernelMatMul: input on a different cuda device".into(),
                        ));
                    }
                    // MMQ is the FAST path, not the only one: the dequantised matmul
                    // below needs no per-call workspace. So a workspace that cannot be
                    // had - even after the reclaim - degrades to it instead of failing the
                    // request. An OOM must never reach the caller while a slower path is
                    // still available.
                    match self.mmq_forward(qdev, blob, x_slice, rows) {
                        Ok(y) => {
                            return crate::tensor::Tensor::from_cuda_storage(
                                crate::tensor::cuda::CudaStorage::F32(y),
                                qdev.clone(),
                                odims,
                            );
                        }
                        Err(e) if e.is_oom() => {
                            tracing::warn!(
                                "mmq unavailable ({e}); this matmul takes the dequantised \
                                 path instead of failing"
                            );
                        }
                        Err(e) => return Err(e),
                    }
                }
            }
        }

        // ...else dequantized-weight matmul (built once, cached on the
        // compute device - the CPU path and the unsupported-dtype path)
        let w = match self.dequant.get() {
            Some(w) => w.clone(),
            None => {
                let wf = self.host.dequantize_f32()?;
                let wt = crate::tensor::Tensor::from_vec_f32(wf, vec![self.n, self.k])?
                    .transpose(0, 1)? // [k, n] so x @ w works row-major
                    .to_device(&self.device)?;
                let _ = self.dequant.set(wt);
                self.dequant.get().unwrap().clone()
            }
        };
        // The cached dequant weight is F32; a half-carrier activation (e.g. the
        // nemotron shared-expert reached during prefill, where the Q8 weight is
        // not MMQ-eligible) must be promoted to F32 for the matmul, then carried
        // back to the input dtype - same "output follows input dtype" contract as
        // the mmvq paths above. Without this, `x2.matmul` errors
        // "expected f32 cuda storage, got f16" on longer prompts.
        let xc = if x.dtype() == crate::tensor::DType::F32 {
            x.clone()
        } else {
            x.to_dtype(crate::tensor::DType::F32)?
        };
        let x2 = xc.reshape(vec![1, rows, k])?;
        let w2 = w.reshape(vec![1, self.k, self.n])?;
        let y = x2.matmul(&w2)?.reshape(odims)?;
        match x.dtype() {
            crate::tensor::DType::F16 | crate::tensor::DType::BF16 => y.to_dtype(x.dtype()),
            _ => Ok(y),
        }
    }

    /// Zero-alloc CPU decode projection: single f32 row `x` `[k]` ->
    /// `out` `[n]`, via the SAME `matmul_bytes` the CPU `forward` path uses
    /// (bit-identical), but writing into a caller buffer instead of allocating
    /// a Tensor. Returns `Ok(false)` if this weight isn't on the supported CPU
    /// quantized fast path (caller falls back to `forward`).
    /// Unified in-place-style Q4_K repack: the single interleaved copy that serves
    /// EVERY CPU matmul over this weight (M=1 decode GEMV + tiled prefill GEMM).
    /// Built once (eagerly at load for CPU weights, lazily otherwise); the canonical
    /// bytes then serve only correctness paths (dequantize / device upload), so
    /// mmap-backed originals go cold and the repack becomes the resident copy.
    pub(crate) fn q4k_x8(&self) -> Option<&Vec<crate::tensor::quant_cpu::repack_q4k::BlockQ4Kx8>> {
        if self.dtype != GgmlDType::Q4K
            || !matches!(self.device, crate::tensor::Device::Cpu)
            || self.n % 8 != 0
            || self.k % 256 != 0
        {
            return None;
        }
        let nb = self.k / 256;
        let blocks = crate::tensor::quant_cpu::cast_blocks::<crate::tensor::quant_cpu::BlockQ4K>(
            self.host.data(),
        )
        .ok()?;
        if blocks.len() != self.n * nb {
            return None;
        }
        Some(
            self.cpu_repack_q4k
                .get_or_init(|| crate::tensor::quant_cpu::repack_q4k::repack(blocks, self.n, nb)),
        )
    }

    /// Output dim (`n`) - needed to size decode-arena buffers.
    pub fn out_dim(&self) -> usize {
        self.n
    }

    /// Fused dense SwiGLU decode: `silu(self.x) * (up.x)` for an F16 activation
    /// row on CPU, in ONE pass over both weights (quantizes `x` once, no
    /// intermediate tensors) - vs two `forward` calls + f32/silu/mul/f16 tensor
    /// ops that each re-quantize `x`. `Ok(None)` if the fast path doesn't apply
    /// (non-CPU, dtype/shape mismatch, m>1, or non-F16 activation) - caller
    /// falls back to the unfused path. Bit-close: same F32 silu.mul, F16 store.
    pub fn gate_up_silu_f16(
        &self,
        up: &QKernelMatMul,
        x: &crate::tensor::Tensor,
    ) -> Result<Option<crate::tensor::Tensor>> {
        if !matches!(self.device, crate::tensor::Device::Cpu)
            || self.dtype != up.dtype
            || self.k != up.k
            || self.n != up.n
            || !crate::tensor::quant_cpu::supports(self.dtype)
            || self.k % self.dtype.block_size() != 0
            || x.dtype() != crate::tensor::DType::F16
        {
            return Ok(None);
        }
        let xdims = x.dims();
        let k = match xdims.last() {
            Some(&k) => k,
            None => return Ok(None),
        };
        if k != self.k || x.elem_count() != k {
            // Only the m==1 decode row is fused here.
            return Ok(None);
        }
        let Ok(lhs) = x.cpu_f16_data() else {
            return Ok(None);
        };
        let mut dst = vec![half::f16::ZERO; self.n];
        crate::tensor::quant_cpu::matmul_f16_gate_up_silu_bytes(
            self.dtype,
            (1, self.k, self.n),
            lhs,
            self.host.data(),
            up.host.data(),
            &mut dst,
        )?;
        let mut odims = xdims.to_vec();
        *odims.last_mut().unwrap() = self.n;
        Ok(Some(crate::tensor::Tensor::from_storage(
            crate::tensor::CpuStorage::F16(dst),
            odims,
        )?))
    }
    /// Input dim (`k`).
    pub fn in_dim(&self) -> usize {
        self.k
    }

    /// Decode-path forward over a zero-copy narrow VIEW of `src`: quantizes
    /// the activation straight from `[offset, offset + rows*k)` of the source
    /// buffer (no materialization). `Ok(None)` = shape/dtype/batch outside
    /// the MMVQ decode window - the caller materializes and uses `forward`.
    #[cfg(feature = "cuda")]
    pub fn forward_view(
        &self,
        src: &crate::tensor::Tensor,
        offset: usize,
        shape: &crate::tensor::Shape,
    ) -> Result<Option<crate::tensor::Tensor>> {
        let xdims = shape.dims().to_vec();
        let Some(&k) = xdims.last() else {
            return Ok(None);
        };
        if k != self.k {
            return Err(Error(format!(
                "QKernelMatMul: input k {k} != weight k {}",
                self.k
            )));
        }
        let rows = shape.elem_count() / k;
        let mut odims = xdims.clone();
        *odims.last_mut().unwrap() = self.n;
        let (crate::tensor::Device::Cuda(qdev), Some(blob)) = (&self.device, self.blob.as_ref())
        else {
            return Ok(None);
        };
        let xdt = src.dtype();
        if !(1..=8).contains(&rows)
            || self.kernel_tag().is_err()
            || !matches!(
                xdt,
                crate::tensor::DType::F32 | crate::tensor::DType::F16 | crate::tensor::DType::BF16
            )
        {
            return Ok(None);
        }
        let tag = self.kernel_tag()?;
        let n_in = rows * k;
        match xdt {
            crate::tensor::DType::F32 => {
                let (x_slice, dev) = src.cuda_f32_slice()?;
                if dev.ordinal() != qdev.ordinal() {
                    return Err(Error(
                        "QKernelMatMul: input on a different cuda device".into(),
                    ));
                }
                let xv = x_slice.slice(offset..offset + n_in);
                let q81 = crate::tensor::cuda::quantize_q8_1_rows_f32_view(qdev, &xv, k, rows)?;
                let y =
                    crate::tensor::cuda::mmvq_f32(qdev, tag, blob, &q81, k, self.n, rows, false)?;
                Ok(Some(crate::tensor::Tensor::from_cuda_storage(
                    crate::tensor::cuda::CudaStorage::F32(y),
                    qdev.clone(),
                    odims,
                )?))
            }
            crate::tensor::DType::F16 => {
                let (x_slice, dev) = src.cuda_f16_slice()?;
                if dev.ordinal() != qdev.ordinal() {
                    return Err(Error(
                        "QKernelMatMul: input on a different cuda device".into(),
                    ));
                }
                let xv = x_slice.slice(offset..offset + n_in);
                let q81 = crate::tensor::cuda::quantize_q8_1_rows_f16_view(qdev, &xv, k, rows)?;
                // same small_k dispatch as `forward`
                let smallk_vdr: Option<i32> = match self.dtype {
                    GgmlDType::Q4K | GgmlDType::Q5K => Some(2),
                    GgmlDType::Q6K => Some(1),
                    _ => None,
                };
                let smallk = rows == 1
                    && self.n % 4 == 0
                    && smallk_vdr.is_some_and(|vdr| (k as i32 / 256) < 4 * vdr);
                let y =
                    crate::tensor::cuda::mmvq_f16(qdev, tag, blob, &q81, k, self.n, rows, smallk)?;
                Ok(Some(crate::tensor::Tensor::from_cuda_storage(
                    crate::tensor::cuda::CudaStorage::F16(y),
                    qdev.clone(),
                    odims,
                )?))
            }
            crate::tensor::DType::BF16 => {
                let (x_slice, dev) = src.cuda_bf16_slice()?;
                if dev.ordinal() != qdev.ordinal() {
                    return Err(Error(
                        "QKernelMatMul: input on a different cuda device".into(),
                    ));
                }
                let xv = x_slice.slice(offset..offset + n_in);
                let q81 = crate::tensor::cuda::quantize_q8_1_rows_bf16_view(qdev, &xv, k, rows)?;
                let y = crate::tensor::cuda::mmvq_bf16(qdev, tag, blob, &q81, k, self.n, rows)?;
                Ok(Some(crate::tensor::Tensor::from_cuda_storage(
                    crate::tensor::cuda::CudaStorage::BF16(y),
                    qdev.clone(),
                    odims,
                )?))
            }
            _ => Ok(None),
        }
    }
}
