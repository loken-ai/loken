//! The production matmul over a quantised weight: which kernel runs, and why that one.
//!
//! There is no single quantised matmul. Which of them serves a call depends on the shape and
//! the dtype - a one-row decode wants a mat-vec, a prefill wants a tiled matmul, some dtypes
//! have an interleaved repack that beats both - and every one of them can decline, in which
//! case the next takes it. The dequantised matmul at the end never declines, which is what
//! makes the whole chain safe: a workspace that cannot be allocated degrades instead of
//! failing the request.

#[cfg(feature = "gpu")]
use super::mmq::{mmq_launcher, mmq_qk, mmq_quantize_launcher, mmq_supports, MMQ_WORKSPACES};
use super::*;

mod cpu;
#[cfg(feature = "gpu")]
mod kat;
mod mmq;
mod repack;

#[cfg(feature = "gpu")]
use kat::kat_bad;

/// The most rows one launch of the matvec kernels takes for a `[.., k]` weight of `dtype`:
/// the crossover past which the tiled GEMM's weight reuse wins, where that GEMM can run this
/// storage, and the matvec family's own bound where it cannot.
#[cfg(feature = "gpu")]
pub fn matvec_rows(dtype: GgmlDType, k: usize) -> usize {
    if mmq_supports(dtype) && k.is_multiple_of(mmq_qk(dtype)) {
        5
    } else {
        8
    }
}
thread_local! {
    static DECODE_ORDER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// While alive, every quantized matmul on this thread computes each activation row
/// bit-identically to a one-row call over that row: the rows take the mat-vec kernels in
/// launches that sum like a single column, never the tiled GEMM. A draft verify holds it,
/// so a verified row reproduces the decode step it stands for.
pub struct DecodeOrder {
    previous: bool,
}

impl DecodeOrder {
    pub fn enter() -> Self {
        Self {
            previous: DECODE_ORDER.with(|d| d.replace(true)),
        }
    }

    pub fn active() -> bool {
        DECODE_ORDER.with(std::cell::Cell::get)
    }
}

impl Drop for DecodeOrder {
    fn drop(&mut self) {
        DECODE_ORDER.with(|d| d.set(self.previous));
    }
}

#[cfg(feature = "gpu")]
pub use kat::kernel_known_answer_test;

/// An activation quantised to q8_1 by [`QKernelMatMul::quantize_for_matvec`], for every
/// mat-vec over the same `k` on the same card.
#[cfg(feature = "gpu")]
pub struct QuantizedActivation {
    q81: crate::tensor::gpu_runtime::driver::CudaSlice<u8>,
    k: usize,
    rows: usize,
    ordinal: usize,
    /// The activation's dimensions before `k`.
    lead: Vec<usize>,
    /// The type the activation was in, and so the type each product comes back in.
    dtype: crate::tensor::DType,
}

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
    #[cfg(feature = "gpu")]
    pub(super) blob: Option<std::sync::Arc<crate::tensor::gpu_runtime::driver::CudaSlice<u8>>>,
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
#[cfg(feature = "gpu")]
static KAT_BAD_MMVQ: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<usize>>> =
    std::sync::OnceLock::new();
#[cfg(feature = "gpu")]
static KAT_BAD_MMQ: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<usize>>> =
    std::sync::OnceLock::new();

impl QKernelMatMul {
    /// Place the weight on `device` (CUDA: blob uploaded for the kernel
    /// paths; CPU: host blocks kept for the dequant path).
    pub fn from_qtensor_on(
        qt: std::sync::Arc<QHostTensor>,
        device: &crate::tensor::Device,
    ) -> Result<Self> {
        #[cfg(feature = "gpu")]
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
            #[cfg(feature = "gpu")]
            blob,
            dry,
        };
        // In-place repack at LOAD: CPU Q4_K weights build their unified interleaved
        // copy up front (off the first token), and every matmul reads it - the
        // canonical bytes stay for correctness paths only.
        #[cfg(target_arch = "x86_64")]
        if matches!(qmm.device, crate::tensor::Device::Cpu) && crate::cpu::dispatch_avx2() {
            let _ = qmm.q4k_x8();
        }
        Ok(qmm)
    }

    #[cfg(feature = "gpu")]
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
    #[cfg(feature = "gpu")]
    pub fn from_qtensor_on_with_blob(
        qt: std::sync::Arc<QHostTensor>,
        device: &crate::tensor::Device,
        blob: Option<std::sync::Arc<crate::tensor::gpu_runtime::driver::CudaSlice<u8>>>,
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
        #[cfg(target_arch = "x86_64")]
        if matches!(qmm.device, crate::tensor::Device::Cpu) && crate::cpu::dispatch_avx2() {
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
    /// The weight as a flat host `[n * k]` F32 vector: decoded from the host blocks, or from
    /// the card's copy when the card holds the only one.
    fn host_f32(&self) -> Result<Vec<f32>> {
        #[cfg(feature = "gpu")]
        if self.host.is_card_only() {
            let blob = self
                .blob
                .as_ref()
                .ok_or_else(|| Error("a card-only weight has no blob".into()))?;
            let crate::tensor::Device::Cuda(dev) = &self.device else {
                return Err(Error("a card-only weight is not on a card".into()));
            };
            let bytes = dev
                .stream()
                .clone_dtoh(&blob.slice(0..self.host.byte_len()))
                .map_err(|e| Error(format!("QKernelMatMul download: {e}")))?;
            let mut out = vec![0f32; self.n * self.k];
            crate::tensor::quant_cpu::to_float_bytes(self.dtype, &bytes, &mut out)?;
            return Ok(out);
        }
        // CONVERSION: an exact F32 matmul was asked for; the weight is decoded for the call.
        self.host.dequantize_f32()
    }

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
            crate::tensor::Tensor::dry(&x.device(), crate::tensor::DType::F32, (self.n, self.k))?
        } else {
            let w = self.host_f32()?; // flat [n.k], weight rows [n, k]
            crate::tensor::Tensor::from_vec_f32(w, (self.n, self.k))?.to_device(&x.device())?
        };
        let x32 = x.to_dtype(crate::tensor::DType::F32)?;
        x32.matmul_t(&wt)
    }

    /// The whole weight dequantised on its card to a fresh dense F32 buffer of `n * k` values.
    /// `None` for a weight that is not a GPU K-quant (Q4_K, Q5_K, Q6_K) or Q8_0 blob.
    #[cfg(feature = "gpu")]
    pub(crate) fn dequant_f32_gpu(
        &self,
    ) -> Result<Option<crate::tensor::gpu_runtime::driver::CudaSlice<f32>>> {
        let (crate::tensor::Device::Cuda(qdev), Some(blob)) = (&self.device, self.blob.as_ref())
        else {
            return Ok(None);
        };
        let total = self.n * self.k;
        // K-quants use the 256-value superblock kernels; Q8_0 (32-value blocks) has its own
        // launcher.
        let kquant = |kernel: &str, threads: u32| {
            crate::tensor::cuda::dequantize_kquant_f32(qdev, kernel, blob, total, threads)
        };
        Ok(match self.dtype {
            GgmlDType::Q4K => Some(kquant("dequantize_block_q4_K_f32", 32)?),
            GgmlDType::Q5K => Some(kquant("dequantize_block_q5_K_f32", 64)?),
            GgmlDType::Q6K => Some(kquant("dequantize_block_q6_K_f32", 64)?),
            GgmlDType::Q8_0 => Some(crate::tensor::cuda::dequantize_q8_0_f32(qdev, blob, total)?),
            _ => None,
        })
    }

    /// Dequantize this weight ONCE into a dense BF16 device tensor `[n, k]`, for
    /// callers that run several GEMMs against the same weight (e.g. a token-chunked
    /// MLP). `Ok(None)` when the weight is not a GPU K-quant/Q8_0 blob - the caller
    /// then falls back to per-call `forward*`. BF16 is the DiT's reference GEMM
    /// precision (holds ~1e9 activations, cuBLAS accumulates in F32).
    #[cfg(feature = "gpu")]
    pub fn dequant_weight_bf16(&self) -> Result<Option<crate::tensor::Tensor>> {
        let (crate::tensor::Device::Cuda(qdev), Some(blob)) = (&self.device, self.blob.as_ref())
        else {
            return Ok(None);
        };
        // Straight to BF16 where the dtype has a BF16 dequantiser: the same values as the F32
        // route cast to BF16, without the F32 copy.
        if let Some(dq) = crate::tensor::cuda::DequantBf16::resolve(qdev, self.dtype) {
            let w = dq.to_new(qdev, blob, self.n * self.k)?;
            return crate::tensor::Tensor::from_cuda_storage(
                crate::tensor::cuda::CudaStorage::BF16(w),
                qdev.clone(),
                vec![self.n, self.k],
            )
            .map(Some);
        }
        match self.dequant_f32_gpu()? {
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

    /// `x [.., k] @ W^T` with the weight dequantised on its card and one cuBLAS BF16 GEMM
    /// accumulating in F32: the wide calls of a DiT, whose activations overflow the quantised
    /// tile path. The output is BF16 for a BF16 activation and F32 otherwise.
    ///
    /// The weight is dequantised straight to BF16 into the stream's reused scratch where that
    /// route applies ([`Self::forward_dequant_bf16`]), else to a fresh F32 copy cast to BF16, else
    /// on the host for a dtype or device no GPU dequantiser serves. Callers on the GPU hot path
    /// must use a dtype one of the GPU routes serves, else the matmul silently runs on the CPU.
    pub fn forward_dequant_gpu(&self, x: &crate::tensor::Tensor) -> Result<crate::tensor::Tensor> {
        #[cfg(feature = "gpu")]
        {
            if let Some(y) = self.forward_dequant_bf16(x)? {
                return Ok(y);
            }
            if let Some(y) = self.forward_dequant_gpu_f32(x)? {
                return Ok(y);
            }
        }
        self.forward_dequant_f32(x)
    }

    /// The dequantised GEMM over a BF16 copy of the weight written by the dtype's BF16
    /// dequantiser into the scratch of the device's stream, reused across calls.
    ///
    /// The GEMM reads BF16 operands, sums in F32 and writes F32 directly for an F32 or F16
    /// activation, so neither the weight nor the output is cast. The weight's values are the
    /// F32 route's rounded once to BF16, so the two routes differ only by the F32 route's
    /// rounding of its output to BF16. `None` where the route does not apply: no BF16
    /// dequantiser for the dtype, a card without BF16 matrix units, an activation that is not
    /// on the weight's card, a dry run, or a stream recording a graph.
    #[cfg(feature = "gpu")]
    pub(crate) fn forward_dequant_bf16(
        &self,
        x: &crate::tensor::Tensor,
    ) -> Result<Option<crate::tensor::Tensor>> {
        use crate::tensor::DType;
        let (crate::tensor::Device::Cuda(qdev), Some(blob)) = (&self.device, self.blob.as_ref())
        else {
            return Ok(None);
        };
        let crate::tensor::Device::Cuda(xdev) = x.device() else {
            return Ok(None);
        };
        let dims = x.dims().to_vec();
        let rows = x.elem_count() / self.k.max(1);
        if xdev.ordinal() != qdev.ordinal()
            || dims.last() != Some(&self.k)
            || rows == 0
            || !qdev.has_bf16_matrix_cores()
            || crate::tensor::cuda_ext::capture_active_on(qdev.stream())
        {
            return Ok(None);
        }
        let Some(dq) = crate::tensor::cuda::DequantBf16::resolve(qdev, self.dtype) else {
            return Ok(None);
        };
        let (n, k) = (self.n, self.k);
        let bf16_out = x.dtype() == DType::BF16;
        let xb = x.to_dtype(DType::BF16)?.contiguous()?;
        let (xs, _) = xb.cuda_bf16_slice()?;
        let storage = crate::tensor::cuda::with_dequant_scratch_bf16(qdev, n * k, |w| {
            dq.launch(qdev, blob, w, n * k)?;
            Ok(if bf16_out {
                crate::tensor::cuda::CudaStorage::BF16(crate::tensor::cuda::matmul_nt_bf16(
                    qdev, xs, w, 1, rows, k, n,
                )?)
            } else {
                crate::tensor::cuda::CudaStorage::F32(crate::tensor::cuda::matmul_nt_bf16_f32(
                    qdev, xs, w, 1, rows, k, n,
                )?)
            })
        })?;
        let mut odims = dims;
        if let Some(last) = odims.last_mut() {
            *last = n;
        }
        crate::tensor::Tensor::from_cuda_storage(storage, qdev.clone(), odims).map(Some)
    }

    /// The dequantised GEMM over a fresh F32 copy of the weight, cast to BF16 with the
    /// activation, the output cast back to F32 for a non-BF16 activation. `None` for a weight
    /// [`Self::dequant_f32_gpu`] does not serve.
    #[cfg(feature = "gpu")]
    pub(crate) fn forward_dequant_gpu_f32(
        &self,
        x: &crate::tensor::Tensor,
    ) -> Result<Option<crate::tensor::Tensor>> {
        let crate::tensor::Device::Cuda(qdev) = &self.device else {
            return Ok(None);
        };
        let Some(w) = self.dequant_f32_gpu()? else {
            return Ok(None);
        };
        let wt = crate::tensor::Tensor::from_cuda_storage(
            crate::tensor::cuda::CudaStorage::F32(w),
            qdev.clone(),
            vec![self.n, self.k],
        )?;
        // BF16 on the tensor cores: its eight-bit exponent holds the DiT's ~1e9
        // activations where f16 overflows, cuBLAS accumulates the products in F32 anyway,
        // and it is four to eight times an F32 GEMM.
        if x.dtype() == crate::tensor::DType::BF16 {
            return x
                .matmul_t(&wt.to_dtype(crate::tensor::DType::BF16)?)
                .map(Some);
        }
        let xb = x.to_dtype(crate::tensor::DType::BF16)?;
        let y = xb.matmul_t(&wt.to_dtype(crate::tensor::DType::BF16)?)?;
        y.to_dtype(crate::tensor::DType::F32).map(Some)
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
    /// The F32 mat-vec over `rows` activation rows already quantised to q8_1.
    #[cfg(feature = "gpu")]
    fn mmvq_f32_rows(
        &self,
        qdev: &std::sync::Arc<crate::tensor::cuda::CudaDevice>,
        blob: &crate::tensor::gpu_runtime::driver::CudaSlice<u8>,
        tag: &str,
        q81: &crate::tensor::gpu_runtime::driver::CudaSlice<u8>,
        rows: usize,
        odims: Vec<usize>,
    ) -> Result<crate::tensor::Tensor> {
        let k = self.k;
        // A one-row call with a small K takes the small-K kernels, which serve several rows a
        // block (`smallk_single`).
        let y = if rows > 1 && DecodeOrder::active() {
            crate::tensor::cuda::mmvq_decode_order::<f32>(
                qdev,
                tag,
                "f32",
                blob,
                q81,
                k,
                self.n,
                rows,
                self.smallk_single(k),
            )?
        } else {
            let smallk = rows == 1 && self.smallk_single(k);
            crate::tensor::cuda::mmvq_f32(qdev, tag, blob, q81, k, self.n, rows, smallk)?
        };
        crate::tensor::Tensor::from_cuda_storage(
            crate::tensor::cuda::CudaStorage::F32(y),
            qdev.clone(),
            odims,
        )
    }

    /// `x` (F32, `[.., k]`) quantised to q8_1 the way `forward` quantises it for the
    /// mat-vec kernels, so several weights over the same activation quantise it once.
    /// `None` where `forward` would not take those kernels.
    #[cfg(feature = "gpu")]
    pub fn quantize_for_matvec(
        &self,
        x: &crate::tensor::Tensor,
    ) -> Result<Option<QuantizedActivation>> {
        let crate::tensor::Device::Cuda(qdev) = &self.device else {
            return Ok(None);
        };
        let rows = x.elem_count() / self.k;
        let dtype = x.dtype();
        if !self.takes_matvec(qdev, rows)
            || !matches!(
                dtype,
                crate::tensor::DType::F32 | crate::tensor::DType::F16 | crate::tensor::DType::BF16
            )
            || x.dims().last() != Some(&self.k)
        {
            return Ok(None);
        }
        let q81 = match dtype {
            crate::tensor::DType::F32 => {
                let (x_slice, dev) = x.cuda_f32_slice()?;
                if dev.ordinal() != qdev.ordinal() {
                    return Ok(None);
                }
                crate::tensor::cuda::quantize_q8_1_rows(qdev, x_slice, self.k, rows)?
            }
            crate::tensor::DType::F16 => {
                let (x_slice, dev) = x.cuda_f16_slice()?;
                if dev.ordinal() != qdev.ordinal() {
                    return Ok(None);
                }
                crate::tensor::cuda::quantize_q8_1_rows_f16(qdev, x_slice, self.k, rows)?
            }
            _ => {
                let (x_slice, dev) = x.cuda_bf16_slice()?;
                if dev.ordinal() != qdev.ordinal() {
                    return Ok(None);
                }
                crate::tensor::cuda::quantize_q8_1_rows_bf16(qdev, x_slice, self.k, rows)?
            }
        };
        let lead = x.dims()[..x.dims().len() - 1].to_vec();
        Ok(Some(QuantizedActivation {
            q81,
            k: self.k,
            rows,
            ordinal: qdev.ordinal(),
            lead,
            dtype,
        }))
    }

    /// `forward` over an activation [`Self::quantize_for_matvec`] produced: bit-identical
    /// to `forward` on the F32 rows. `None` where `forward` would take another path.
    #[cfg(feature = "gpu")]
    pub fn forward_quantized(
        &self,
        a: &QuantizedActivation,
    ) -> Result<Option<crate::tensor::Tensor>> {
        let (crate::tensor::Device::Cuda(qdev), Some(blob)) = (&self.device, self.blob.as_ref())
        else {
            return Ok(None);
        };
        if a.k != self.k
            || a.ordinal != qdev.ordinal()
            || !self.takes_matvec(qdev, a.rows)
            || self.lora_fused.is_some()
        {
            return Ok(None);
        }
        let mut odims = a.lead.clone();
        odims.push(self.n);
        let tag = self.kernel_tag()?;
        let decode_order = a.rows > 1 && DecodeOrder::active();
        let smallk = a.rows == 1 && self.smallk_single(self.k);
        let (k, n, rows) = (self.k, self.n, a.rows);
        let (storage, device) = match a.dtype {
            crate::tensor::DType::F16 => {
                let y = if decode_order {
                    crate::tensor::cuda::mmvq_decode_order::<half::f16>(
                        qdev,
                        tag,
                        "f16",
                        blob,
                        &a.q81,
                        k,
                        n,
                        rows,
                        self.smallk_single(k),
                    )?
                } else {
                    crate::tensor::cuda::mmvq_f16(qdev, tag, blob, &a.q81, k, n, rows, smallk)?
                };
                (crate::tensor::cuda::CudaStorage::F16(y), qdev.clone())
            }
            crate::tensor::DType::BF16 => {
                let y = if decode_order {
                    crate::tensor::cuda::mmvq_decode_order::<half::bf16>(
                        qdev, tag, "bf16", blob, &a.q81, k, n, rows, false,
                    )?
                } else {
                    crate::tensor::cuda::mmvq_bf16(qdev, tag, blob, &a.q81, k, n, rows)?
                };
                (crate::tensor::cuda::CudaStorage::BF16(y), qdev.clone())
            }
            _ => {
                return self
                    .mmvq_f32_rows(qdev, blob, tag, &a.q81, a.rows, odims)
                    .map(Some)
            }
        };
        crate::tensor::Tensor::from_cuda_storage(storage, device, odims).map(Some)
    }

    /// `forward_quantized` of this weight and of `other` over one activation row, in one
    /// launch: each output has the bits of its own `forward_quantized`. `None` where the pair
    /// is not two weights of one format the one-row kernel serves.
    #[cfg(feature = "gpu")]
    pub fn forward_quantized_pair(
        &self,
        other: &Self,
        a: &QuantizedActivation,
    ) -> Result<Option<(crate::tensor::Tensor, crate::tensor::Tensor)>> {
        let (
            crate::tensor::Device::Cuda(qdev),
            Some(blob0),
            crate::tensor::Device::Cuda(odev),
            Some(blob1),
        ) = (
            &self.device,
            self.blob.as_ref(),
            &other.device,
            other.blob.as_ref(),
        )
        else {
            return Ok(None);
        };
        if a.k != self.k
            || other.k != self.k
            || a.rows != 1
            || a.dtype != crate::tensor::DType::F32
            || self.dtype != other.dtype
            || qdev.ordinal() != odev.ordinal()
            || a.ordinal != qdev.ordinal()
            || !self.takes_matvec(qdev, 1)
            || !other.takes_matvec(qdev, 1)
            || self.lora_fused.is_some()
            || other.lora_fused.is_some()
            || self.smallk_single(self.k)
            || other.smallk_single(self.k)
        {
            return Ok(None);
        }
        let tag = self.kernel_tag()?;
        let (y0, y1) = crate::tensor::cuda::mmvq_f32_dual(
            qdev, tag, blob0, blob1, &a.q81, self.k, self.n, other.n,
        )?;
        let mut d0 = a.lead.clone();
        d0.push(self.n);
        let mut d1 = a.lead.clone();
        d1.push(other.n);
        let t0 = crate::tensor::Tensor::from_cuda_storage(
            crate::tensor::cuda::CudaStorage::F32(y0),
            qdev.clone(),
            d0,
        )?;
        let t1 = crate::tensor::Tensor::from_cuda_storage(
            crate::tensor::cuda::CudaStorage::F32(y1),
            qdev.clone(),
            d1,
        )?;
        Ok(Some((t0, t1)))
    }

    /// [`Self::forward_quantized`] over one F32 row with `bias` (`[n]` F32 on the same card)
    /// added to each output in the kernel's store: the same bits as the product followed by a
    /// bias add, one launch fewer. `None` where another path serves the row.
    #[cfg(feature = "gpu")]
    pub fn forward_quantized_bias(
        &self,
        a: &QuantizedActivation,
        bias: &crate::tensor::Tensor,
    ) -> Result<Option<crate::tensor::Tensor>> {
        let (crate::tensor::Device::Cuda(qdev), Some(blob)) = (&self.device, self.blob.as_ref())
        else {
            return Ok(None);
        };
        if a.k != self.k
            || a.rows != 1
            || a.dtype != crate::tensor::DType::F32
            || a.ordinal != qdev.ordinal()
            || !self.takes_matvec(qdev, 1)
            || self.lora_fused.is_some()
            || bias.dtype() != crate::tensor::DType::F32
            || bias.elem_count() != self.n
            || !bias.is_contiguous()
        {
            return Ok(None);
        }
        let (b_slice, bdev) = bias.cuda_f32_slice()?;
        if bdev.ordinal() != qdev.ordinal() {
            return Ok(None);
        }
        let offset = bias.layout().start_offset();
        let b_view = b_slice.slice(offset..offset + self.n);
        let tag = self.kernel_tag()?;
        let y = crate::tensor::cuda::mmvq_f32_bias(
            qdev,
            tag,
            blob,
            &a.q81,
            self.k,
            self.n,
            self.smallk_single(self.k),
            &b_view,
        )?;
        let mut odims = a.lead.clone();
        odims.push(self.n);
        crate::tensor::Tensor::from_cuda_storage(
            crate::tensor::cuda::CudaStorage::F32(y),
            qdev.clone(),
            odims,
        )
        .map(Some)
    }

    /// Whether `forward` serves `rows` F32 activation rows with the mat-vec kernels.
    #[cfg(feature = "gpu")]
    fn takes_matvec(&self, qdev: &crate::tensor::cuda::CudaDevice, rows: usize) -> bool {
        self.blob.is_some()
            && ((1..=matvec_rows(self.dtype, self.k)).contains(&rows)
                || (rows > 1 && DecodeOrder::active()))
            && !kat_bad(&KAT_BAD_MMVQ, qdev.ordinal())
            && self.kernel_tag().is_ok()
    }

    /// Whether a one-row call over this weight takes a small-K mat-vec: a K whose row the
    /// one-row kernel's lanes cover in one pass, and a row count the kernel's rows per block
    /// divide ([`crate::tensor::cuda::mmvq_smallk_warps`]).
    #[cfg(feature = "gpu")]
    fn smallk_single(&self, k: usize) -> bool {
        self.kernel_tag()
            .is_ok_and(|tag| crate::tensor::cuda::mmvq_smallk_warps(tag, k, self.n).is_some())
    }

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
        let acc = xf.matmul(down)?.matmul(up)?;
        let mut odims = xdims;
        *odims.last_mut().unwrap() = self.n;
        let delta = acc.reshape(odims)?.to_dtype(base.dtype())?;
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

        // Rows that must each reproduce a one-row call: the decode engine over all of them
        // where it serves the one-row call, else one call per row.
        if rows > 1
            && DecodeOrder::active()
            && matches!(self.device, crate::tensor::Device::Cpu)
            && crate::tensor::quant_cpu::supports(self.dtype)
            && self.k.is_multiple_of(self.dtype.block_size())
        {
            if x.dtype() == crate::tensor::DType::F32
                && crate::cpu::dispatch_avx2()
                && self.dtype != GgmlDType::Q4_0
            {
                if let Ok(lhs) = x.cpu_f32_data() {
                    let mut dst = crate::tensor::quant_cpu::zeroed_vec_f32(rows * self.n);
                    if self.forward_rows_cpu(lhs, &mut dst)? {
                        return crate::tensor::Tensor::from_vec_f32(dst, odims);
                    }
                }
            }
            let flat = x.reshape((rows, self.k))?;
            let ys = (0..rows)
                .map(|r| self.forward_base(&flat.narrow(0, r, 1)?))
                .collect::<Result<Vec<_>>>()?;
            return crate::tensor::Tensor::cat(&ys.iter().collect::<Vec<_>>(), 0)?.reshape(odims);
        }

        // CPU fast path: the lifted k_quants dot engine - the
        // activation is Q8-quantized once per row and the output columns run
        // rayon-parallel, bit-exact with the fork's CPU matmul. This replaces
        // the dequant-f32 fallback for every supported dtype (without it the
        // facade flip would regress CPU/hybrid decode 10-100x).
        if matches!(self.device, crate::tensor::Device::Cpu)
            && crate::tensor::quant_cpu::supports(self.dtype)
            && self.k.is_multiple_of(self.dtype.block_size())
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
            let mut dst = crate::tensor::quant_cpu::zeroed_vec_f32(rows * self.n);
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
                && self.n.is_multiple_of(8)
                && self.k.is_multiple_of(256)
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
                && self.n.is_multiple_of(8)
                && self.k.is_multiple_of(32)
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
                && self.n.is_multiple_of(8)
                && self.k.is_multiple_of(32)
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
                && self.n.is_multiple_of(8)
                && self.k.is_multiple_of(256)
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
                && self.n.is_multiple_of(8)
                && self.k.is_multiple_of(32)
                && crate::tensor::quant_cpu::cast_blocks::<crate::tensor::quant_cpu::BlockQ5_0>(
                    self.host.data(),
                )
                .map(|b| b.len() == self.n * nb40)
                .unwrap_or(false);
            let q5k_prefill = self.dtype == GgmlDType::Q5K
                && rows >= 4
                && self.n.is_multiple_of(8)
                && self.k.is_multiple_of(256)
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
                && self.n.is_multiple_of(8)
                && self.k.is_multiple_of(32)
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
                && self.n.is_multiple_of(8)
                && self.k.is_multiple_of(32)
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
                    // CONVERSION: x8 interleaved layout the host dot kernels read; the stored blocks stay.
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
                    // CONVERSION: x8 interleaved layout the host dot kernels read; the stored blocks stay.
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
                    // CONVERSION: x8 interleaved layout the host dot kernels read; the stored blocks stay.
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
                    // CONVERSION: x8 interleaved layout the host dot kernels read; the stored blocks stay.
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
                    // CONVERSION: x8 interleaved layout the host dot kernels read; the stored blocks stay.
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
                    // CONVERSION: x8 interleaved layout the host dot kernels read; the stored blocks stay.
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
                    // CONVERSION: x8 interleaved layout the host dot kernels read; the stored blocks stay.
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

        #[cfg(feature = "gpu")]
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
            let mmvq_max = matvec_rows(self.dtype, self.k);
            // Rows that must each reproduce a one-row call stay on the mat-vec kernels
            // whatever their count.
            let decode_order = rows > 1 && DecodeOrder::active();
            if ((1..=mmvq_max).contains(&rows) || decode_order)
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
                        return self.mmvq_f32_rows(qdev, blob, tag, &q81, rows, odims);
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
                        // The small-K kernels for a one-row call (`smallk_single`).
                        let smallk = rows == 1 && self.smallk_single(k);
                        let y = if decode_order {
                            crate::tensor::cuda::mmvq_decode_order::<half::f16>(
                                qdev,
                                tag,
                                "f16",
                                blob,
                                &q81,
                                k,
                                self.n,
                                rows,
                                self.smallk_single(k),
                            )?
                        } else {
                            crate::tensor::cuda::mmvq_f16(
                                qdev, tag, blob, &q81, k, self.n, rows, smallk,
                            )?
                        };
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
                        let y = if decode_order {
                            crate::tensor::cuda::mmvq_decode_order::<half::bf16>(
                                qdev, tag, "bf16", blob, &q81, k, self.n, rows, false,
                            )?
                        } else {
                            crate::tensor::cuda::mmvq_bf16(qdev, tag, blob, &q81, k, self.n, rows)?
                        };
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
                && self.k.is_multiple_of(mmq_qk(self.dtype))
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
                let wf = self.host_f32()?;
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
            || !self.n.is_multiple_of(8)
            || !self.k.is_multiple_of(256)
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
                // CONVERSION: x8 interleaved layout the host dot kernels read; the stored blocks stay.
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
            || !self.k.is_multiple_of(self.dtype.block_size())
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
    #[cfg(feature = "gpu")]
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
                let smallk = rows == 1 && self.smallk_single(k);
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
