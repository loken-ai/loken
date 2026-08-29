//! Part of `impl Tensor`, split out of the parent module.
//!
//! Rust lets one inherent impl live in several modules of a crate, so this is the
//! same impl - only the file changed. Methods that were private are `pub(super)`
//! here, which is the reach they had when they sat beside their callers.

use super::*;

impl Tensor {
    /// RMS norm over the last dim: `x / sqrt(mean(x²) + eps) * weight`.
    /// On CUDA storage this dispatches to the production fused kernel.
    pub fn rms_norm(&self, weight: &Self, eps: f32) -> Result<Self> {
        if let Some(t) = self.dry_out(self.shape.clone(), self.dtype()) {
            return Ok(t);
        }
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() {
            let w = if matches!(weight.dtype(), DType::F16 | DType::BF16) {
                weight.to_dtype(DType::F32)?
            } else {
                weight.clone()
            };
            return self.f16_unary_via_f32(|t| t.rms_norm(&w, eps));
        }
        #[cfg(feature = "cuda")]
        if let Storage::Cuda { data, dev } = self.storage().as_ref() {
            let (w_slice, w_dev) = weight.cuda_f32_slice().map_err(|_| {
                Error("rms_norm: weight must be on the same cuda device as x".into())
            })?;
            if w_dev.ordinal() != dev.ordinal() {
                return Err(Error("rms_norm: weight on a different cuda device".into()));
            }
            let last = *self
                .dims()
                .last()
                .ok_or_else(|| Error("rms_norm on rank-0".into()))?;
            let rows = self.elem_count() / last;
            let out = crate::tensor::cuda::rms_norm_f32(
                dev,
                data.as_f32_slice()?,
                w_slice,
                rows,
                last,
                eps,
            )?;
            return Self::from_cuda_storage(
                crate::tensor::cuda::CudaStorage::F32(out),
                dev.clone(),
                self.shape.clone(),
            );
        }
        let a = self.f32_data()?;
        let w = weight.f32_data()?;
        let last = *self
            .dims()
            .last()
            .ok_or_else(|| Error("rms_norm on rank-0".into()))?;
        if w.len() != last {
            return Err(Error(format!(
                "rms_norm: weight len {} != last dim {last}",
                w.len()
            )));
        }
        let rows = self.elem_count() / last;
        let mut out = vec![0.0f32; a.len()];
        for r in 0..rows {
            let src = &a[r * last..][..last];
            let dst = &mut out[r * last..][..last];
            let ss: f32 = src.iter().map(|&x| x * x).sum();
            let inv = 1.0 / (ss / last as f32 + eps).sqrt();
            for ((d, &s), &wv) in dst.iter_mut().zip(src).zip(w) {
                *d = s * inv * wv;
            }
        }
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(CpuStorage::F32(out))),
            self.shape.clone(),
        ))
    }

    /// What a kernel launch reads off a tensor: the raw device slice, contiguous
    /// strides, and a start offset. ZERO-COPY on narrow views - the RAW storage is
    /// handed out with the view offset projected through `Layout::start_offset()`,
    /// which the launch sites index the buffer with.
    pub fn storage_and_layout(
        &self,
    ) -> (
        crate::tensor::kernel_ffi::StorageHandle,
        crate::tensor::kernel_ffi::Layout,
    ) {
        use crate::tensor::kernel_ffi as kf;
        let layout = kf::Layout::contiguous_with_offset(self.shape().clone(), self.offset);
        let storage = match self.storage_raw.as_ref() {
            Storage::Cpu(_) | Storage::Dry(_) => kf::StorageView::Cpu(kf::CpuStorageRef {
                storage: self.storage_raw.clone(),
            }),
            #[cfg(feature = "cuda")]
            Storage::Cuda { dev, .. } => kf::StorageView::Cuda(kf::CudaStorage {
                storage: self.storage_raw.clone(),
                device: crate::tensor::kernel_ffi::CudaDevice(dev.clone()),
            }),
        };
        (kf::StorageHandle(storage), layout)
    }

    /// matmul that broadcasts the lower-rank operand's leading (batch) dims first - the
    /// broadcasting API the inference layer calls. Equal-batch shapes go straight
    /// to `matmul`.
    pub fn broadcast_matmul(&self, rhs: &Self) -> Result<Self> {
        let (a, b) = (self.rank(), rhs.rank());
        if a == b && self.dims()[..a - 2] == rhs.dims()[..b - 2] {
            return self.matmul(rhs);
        }
        if a < b {
            let mut dims = rhs.dims()[..b - 2].to_vec();
            dims.extend_from_slice(self.dims());
            return self.broadcast_as(dims)?.matmul(rhs);
        }
        let mut dims = self.dims()[..a - 2].to_vec();
        dims.extend_from_slice(rhs.dims());
        self.matmul(&rhs.broadcast_as(dims)?)
    }

    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        let ldims = self.dims();
        let rdims = rhs.dims();
        if ldims.len() < 2 || rdims.len() != ldims.len() {
            return Err(Error(format!("matmul: ranks {ldims:?} x {rdims:?}")));
        }
        let (m, k) = (ldims[ldims.len() - 2], ldims[ldims.len() - 1]);
        let (k2, n) = (rdims[rdims.len() - 2], rdims[rdims.len() - 1]);
        let lbatch: usize = ldims[..ldims.len() - 2].iter().product();
        let rbatch: usize = rdims[..rdims.len() - 2].iter().product();
        if k != k2 || lbatch != rbatch {
            return Err(Error(format!("matmul: {ldims:?} x {rdims:?} incompatible")));
        }
        if self.is_dry() || rhs.is_dry() {
            let mut odims = ldims[..ldims.len() - 2].to_vec();
            odims.push(m);
            odims.push(n);
            let dtype = self.dtype();
            if let Some(t) = self.dry_out(odims.clone(), dtype) {
                return Ok(t);
            }
            if let Some(t) = rhs.dry_out(odims, dtype) {
                return Ok(t);
            }
        }
        #[cfg(feature = "cuda")]
        if let (
            Storage::Cuda { data, dev },
            Storage::Cuda {
                data: rd,
                dev: rdev,
            },
        ) = (self.storage().as_ref(), rhs.storage().as_ref())
        {
            if dev.ordinal() != rdev.ordinal() {
                return Err(Error("matmul: operands on different cuda devices".into()));
            }
            let mut odims = ldims[..ldims.len() - 2].to_vec();
            odims.push(m);
            odims.push(n);
            let cuda_run = || -> Result<Self> {
                // bf16 pair -> bf16 gemm
                if self.dtype() == DType::BF16 && rhs.dtype() == DType::BF16 {
                    let out = crate::tensor::cuda::matmul_bf16(
                        dev,
                        data.as_bf16_slice()?,
                        rd.as_bf16_slice()?,
                        lbatch,
                        m,
                        k,
                        n,
                    )?;
                    return Self::from_cuda_storage(
                        crate::tensor::cuda::CudaStorage::BF16(out),
                        dev.clone(),
                        odims.clone(),
                    );
                }
                // f16 pair -> hgemm; otherwise the f32 path
                if self.dtype() == DType::F16 && rhs.dtype() == DType::F16 {
                    let out = crate::tensor::cuda::matmul_f16(
                        dev,
                        data.as_f16_slice()?,
                        rd.as_f16_slice()?,
                        lbatch,
                        m,
                        k,
                        n,
                    )?;
                    return Self::from_cuda_storage(
                        crate::tensor::cuda::CudaStorage::F16(out),
                        dev.clone(),
                        odims.clone(),
                    );
                }
                let out = crate::tensor::cuda::matmul_f32(
                    dev,
                    data.as_f32_slice()?,
                    rd.as_f32_slice()?,
                    lbatch,
                    m,
                    k,
                    n,
                )?;
                Self::from_cuda_storage(
                    crate::tensor::cuda::CudaStorage::F32(out),
                    dev.clone(),
                    odims.clone(),
                )
            };
            return match cuda_run() {
                // Device truly out of memory even after the allocator's
                // reclaim-and-retry: fall back to a host matmul rather than
                // failing the request. Slow but correct - the no-OOM
                // guarantee outranks throughput here.
                Err(e) if e.is_oom() => {
                    crate::tensor::bounce::note_pressure_bounce();
                    tracing::warn!(
                        "matmul {ldims:?} x {rdims:?}: GPU{} OOM after reclaim; bouncing to CPU",
                        dev.ordinal()
                    );
                    let _b = crate::tensor::bounce::start("oom_bounce_matmul");
                    let a = self.to_device(&Device::Cpu)?;
                    let b = rhs.to_device(&Device::Cpu)?;
                    a.matmul(&b)?.to_device(&self.device())
                }
                r => r,
            };
        }
        // CPU half-precision route: f32 compute + round back (the gemm-crate
        // half kernels also accumulate in f32); output keeps the
        // input dtype like the reference matmul.
        if matches!(self.dtype(), DType::F16 | DType::BF16) && rhs.dtype() == self.dtype() {
            let dt = self.dtype();
            // f16 x f16: stream the f16 rhs through gemm_f16w_f32_cpu's
            // L2-resident per-tile convert instead of materializing full f32
            // copies of BOTH operands every call (per-token cost: CPU decode
            // attention `p.v` with an F16 KV store re-upcast the whole V
            // window each token/layer - write k.n.4B + re-read, vs a 2B/elem
            // direct stream). rhs there is the KV cache (changes every token),
            // so the upcast is not cacheable tensor-side. Bit-identical:
            // f16->f32 is exact and the accumulation loops mirror gemm_f32_cpu
            // (see gemm_f16w_matches_upcast_path test). bf16 keeps the f32
            // round-trip (mirrors matmul_t's f16/bf16 split).
            if dt == DType::F16 {
                if let (Storage::Cpu(CpuStorage::F16(af)), Storage::Cpu(CpuStorage::F16(bf))) =
                    (self.storage().as_ref(), rhs.storage().as_ref())
                {
                    use half::slice::HalfFloatSliceExt;
                    // activations convert once (m.k - small next to the k.n rhs)
                    let mut a = vec![0f32; lbatch * m * k];
                    af.convert_to_f32_slice(&mut a);
                    // gemm fully overwrites out (see the F32 path note below)
                    let mut out = Vec::with_capacity(lbatch * m * n);
                    #[allow(clippy::uninit_vec)]
                    unsafe {
                        out.set_len(lbatch * m * n)
                    };
                    let per = m * k * n;
                    // Two parallelism sources, and they are mutually exclusive
                    // below: the batch/head dim (`lbatch` ways) or the gemm's own
                    // n-stripes (`n/NT` ways). Pick whichever offers MORE, rather
                    // than letting any `n > NT` claim the work: attention prefill
                    // is `[1, n_head, seq, ..]`, so batch gives n_head ways while
                    // q.kᵀ's stripes give only kv_len/NT - once the context grows
                    // past one stripe the old test handed a 32-way region to a
                    // 2..5-way one and left most of the pool idle.
                    let inner_ways = n.div_ceil(512);
                    let inner_parallel = per >= (1 << 20) && n > 512 && inner_ways > lbatch;
                    if lbatch > 1 && !inner_parallel && lbatch * per >= (1 << 20) && lbatch <= 16 {
                        // Small-head models (tiny models, <=16 heads): run the batched
                        // F16 attention GEMM on the prefill spin-pool instead of rayon.
                        // This takes rayon OFF the forward hot path -> its 20 workers
                        // park instead of idle-spinning (and stealing SMT cycles) during
                        // the quantized-GEMM regions - the two-pool contention that
                        // dominates tiny-model prefill. Large-head models keep rayon
                        // (its work-stealing balances their bigger attention better).
                        crate::tensor::quant_cpu::pool_par_chunks_mut_prefill(
                            &mut out,
                            m * n,
                            &|bi, ob| {
                                let ab = &a[bi * m * k..][..m * k];
                                let bb = &bf[bi * k * n..][..k * n];
                                gemm_f16w_f32_cpu(ab, bb, m, k, n, ob);
                            },
                        );
                    } else if lbatch > 1 && !inner_parallel && lbatch * per >= (1 << 20) {
                        use rayon::prelude::*;
                        out.par_chunks_mut(m * n).enumerate().for_each(|(bi, ob)| {
                            let ab = &a[bi * m * k..][..m * k];
                            let bb = &bf[bi * k * n..][..k * n];
                            gemm_f16w_f32_cpu(ab, bb, m, k, n, ob);
                        });
                    } else {
                        for bi in 0..lbatch {
                            let ab = &a[bi * m * k..][..m * k];
                            let bb = &bf[bi * k * n..][..k * n];
                            let ob = &mut out[bi * m * n..][..m * n];
                            gemm_f16w_f32_cpu(ab, bb, m, k, n, ob);
                        }
                    }
                    let mut odims = ldims[..ldims.len() - 2].to_vec();
                    odims.push(m);
                    odims.push(n);
                    // round back through to_dtype (same half::f16::from_f32 as
                    // the old f32-materializing route)
                    return Self::from_packed(
                        Arc::new(Storage::Cpu(CpuStorage::F32(out))),
                        Shape::from(odims),
                    )
                    .to_dtype(dt);
                }
            }
            return self
                .to_dtype(DType::F32)?
                .matmul(&rhs.to_dtype(DType::F32)?)?
                .to_dtype(dt);
        }
        let a = self.f32_data()?;
        let b = rhs.f32_data()?;
        // gemm_f32_cpu fully overwrites `out` (its column stripes partition
        // 0..n and every stripe writes all m rows via copy_from_slice), so
        // skip the zero-fill: `vec![0.0; ..]` memset/calloc'd a fresh buffer
        // per matmul, in the per-token CPU decode attention path.
        let mut out: Vec<f32> = Vec::with_capacity(lbatch * m * n);
        #[allow(clippy::uninit_vec)]
        unsafe {
            out.set_len(lbatch * m * n)
        };
        // `gemm_f32_cpu` self-parallelizes (over column stripes) only when a
        // single matmul is large enough. Batched attention at decode is the
        // opposite shape: many small per-head GEMVs (m=1, n=kv_len) that each
        // stay below that threshold, so a serial batch loop ran all heads on
        // ONE core while the rest idled - the dominant wall-time cost of
        // long-context CPU decode (attention is serial while the weight matmuls
        // are parallel). Parallelize the batch (heads) when the inner gemm
        // won't, and there's enough total work to amortize.
        let per = m * k * n;
        let inner_parallel = per >= (1 << 20) && n > 512;
        if lbatch > 1 && !inner_parallel && lbatch * per >= (1 << 20) {
            use rayon::prelude::*;
            out.par_chunks_mut(m * n).enumerate().for_each(|(bi, ob)| {
                let ab = &a[bi * m * k..][..m * k];
                let bb = &b[bi * k * n..][..k * n];
                gemm_f32_cpu(ab, bb, m, k, n, ob);
            });
        } else {
            for bi in 0..lbatch {
                let ab = &a[bi * m * k..][..m * k];
                let bb = &b[bi * k * n..][..k * n];
                let ob = &mut out[bi * m * n..][..m * n];
                gemm_f32_cpu(ab, bb, m, k, n, ob);
            }
        }
        let mut odims = ldims[..ldims.len() - 2].to_vec();
        odims.push(m);
        odims.push(n);
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(CpuStorage::F32(out))),
            Shape::from(odims),
        ))
    }

    /// `self [.., m, k] @ rhs[.., n, k]^T -> [.., m, n]` with rhs given
    /// UNtransposed - issues the same `transa=T` cuBLAS call the tensor-op
    /// facade makes for `x.matmul(&w.t()?)` (bit-exactness across the flip;
    /// it also skips the transpose materialization). CPU/non-CUDA
    /// falls back to materializing.
    pub fn matmul_t(&self, rhs: &Self) -> Result<Self> {
        if self.is_dry() || rhs.is_dry() {
            let (ldims, rdims) = (self.dims(), rhs.dims());
            if ldims.len() < 2 || rdims.len() != ldims.len() {
                return Err(Error(format!("matmul_t: ranks {ldims:?} x {rdims:?}")));
            }
            let m = ldims[ldims.len() - 2];
            let n = rdims[rdims.len() - 2];
            let mut odims = ldims[..ldims.len() - 2].to_vec();
            odims.push(m);
            odims.push(n);
            let dtype = self.dtype();
            if let Some(t) = self.dry_out(odims.clone(), dtype) {
                return Ok(t);
            }
            if let Some(t) = rhs.dry_out(odims, dtype) {
                return Ok(t);
            }
        }
        #[cfg(feature = "cuda")]
        {
            let ldims = self.dims();
            let rdims = rhs.dims();
            if ldims.len() >= 2 && rdims.len() == ldims.len() {
                let (m, k) = (ldims[ldims.len() - 2], ldims[ldims.len() - 1]);
                let (n, k2) = (rdims[rdims.len() - 2], rdims[rdims.len() - 1]);
                let lbatch: usize = ldims[..ldims.len() - 2].iter().product();
                let rbatch: usize = rdims[..rdims.len() - 2].iter().product();
                if k == k2 && lbatch == rbatch && self.dtype() == rhs.dtype() {
                    if let (
                        Storage::Cuda { data, dev },
                        Storage::Cuda {
                            data: rd,
                            dev: rdev,
                        },
                    ) = (self.storage().as_ref(), rhs.storage().as_ref())
                    {
                        if dev.ordinal() != rdev.ordinal() {
                            return Err(Error(
                                "matmul_t: operands on different cuda devices".into(),
                            ));
                        }
                        let mut odims = ldims[..ldims.len() - 2].to_vec();
                        odims.push(m);
                        odims.push(n);
                        let storage = match self.dtype() {
                            DType::F32 => crate::tensor::cuda::CudaStorage::F32(
                                crate::tensor::cuda::matmul_nt_f32(
                                    dev,
                                    data.as_f32_slice()?,
                                    rd.as_f32_slice()?,
                                    lbatch,
                                    m,
                                    k,
                                    n,
                                )?,
                            ),
                            DType::F16 => crate::tensor::cuda::CudaStorage::F16(
                                crate::tensor::cuda::matmul_nt_f16(
                                    dev,
                                    data.as_f16_slice()?,
                                    rd.as_f16_slice()?,
                                    lbatch,
                                    m,
                                    k,
                                    n,
                                )?,
                            ),
                            DType::BF16 => crate::tensor::cuda::CudaStorage::BF16(
                                crate::tensor::cuda::matmul_nt_bf16(
                                    dev,
                                    data.as_bf16_slice()?,
                                    rd.as_bf16_slice()?,
                                    lbatch,
                                    m,
                                    k,
                                    n,
                                )?,
                            ),
                            other => {
                                return Err(Error(format!("matmul_t: unsupported dtype {other}")))
                            }
                        };
                        return Self::from_cuda_storage(storage, dev.clone(), odims);
                    }
                }
            }
        }
        // CPU path: direct NT (dot-product) gemm - both operand rows are
        // contiguous, so no transpose materialization (the per-element gather
        // transpose dominated lfm2 CPU decode under the flip). bf16 routes
        // through f32 once (gemm-crate-style f32 accumulation) and rounds
        // back; f16 pairs get a direct kernel below.
        if !self.is_on_cuda() && !rhs.is_on_cuda() {
            if self.dtype() == DType::BF16 && rhs.dtype() == DType::BF16 {
                let dt = self.dtype();
                return self
                    .to_dtype(DType::F32)?
                    .matmul_t(&rhs.to_dtype(DType::F32)?)?
                    .to_dtype(dt);
            }
            let ldims = self.dims();
            let rdims = rhs.dims();
            if ldims.len() >= 2 && rdims.len() == ldims.len() {
                let (m, k) = (ldims[ldims.len() - 2], ldims[ldims.len() - 1]);
                let (n, k2) = (rdims[rdims.len() - 2], rdims[rdims.len() - 1]);
                let lbatch: usize = ldims[..ldims.len() - 2].iter().product();
                let rbatch: usize = rdims[..rdims.len() - 2].iter().product();
                // f16 x f16: stream the f16 weight ONCE, converting row
                // chunks in registers with f32 accumulation. The previous
                // route materialized f32 copies of BOTH operands every call
                // (for a dense-weight linear that is a whole-matrix convert
                // + 2x the read traffic per token) and round-tripped the
                // output. Column-sharded on the persistent GEMV pool.
                if k == k2
                    && lbatch == rbatch
                    && self.dtype() == DType::F16
                    && rhs.dtype() == DType::F16
                {
                    if let (Storage::Cpu(CpuStorage::F16(a)), Storage::Cpu(CpuStorage::F16(b))) =
                        (self.storage().as_ref(), rhs.storage().as_ref())
                    {
                        use half::slice::HalfFloatSliceExt;
                        let pool = crate::tensor::quant_cpu::gemv_pool::pool();
                        // Activations convert once (small next to the weight).
                        let mut a32 = vec![0f32; lbatch * m * k];
                        a.convert_to_f32_slice(&mut a32);
                        let mut out = vec![half::f16::ZERO; lbatch * m * n];
                        let rows_total = lbatch * m;
                        let (chunk_cols, n_chunks) =
                            crate::tensor::quant_cpu::gemv_grid(rows_total, n, pool.threads);
                        let chunks_per_row = n.div_ceil(chunk_cols);
                        struct MutPtr(*mut half::f16);
                        unsafe impl Send for MutPtr {}
                        unsafe impl Sync for MutPtr {}
                        let out_ptr = MutPtr(out.as_mut_ptr());
                        pool.run(n_chunks, &|chunk| {
                            let out_ptr = &out_ptr;
                            let r = chunk / chunks_per_row;
                            let bi = r / m;
                            let col0 = (chunk % chunks_per_row) * chunk_cols;
                            let col1 = (col0 + chunk_cols).min(n);
                            let a_row = &a32[r * k..(r + 1) * k];
                            // SAFETY: chunks address disjoint [r, col0..col1]
                            // ranges of `out`.
                            let out_row = unsafe {
                                std::slice::from_raw_parts_mut(
                                    out_ptr.0.add(r * n + col0),
                                    col1 - col0,
                                )
                            };
                            let mut buf = [0f32; 128];
                            for (j, o) in out_row.iter_mut().enumerate() {
                                let col = col0 + j;
                                let b_row = &b[(bi * n + col) * k..][..k];
                                // 8-lane accumulators so the inner loop
                                // vectorizes to FMA despite strict FP order.
                                let mut acc = [0f32; 8];
                                let mut tail = 0f32;
                                for (bc, ac) in b_row.chunks(128).zip(a_row.chunks(128)) {
                                    let bl = &mut buf[..bc.len()];
                                    bc.convert_to_f32_slice(bl);
                                    let bv = bl.chunks_exact(8);
                                    let av = ac.chunks_exact(8);
                                    for (x, y) in av.clone().zip(bv.clone()) {
                                        for l in 0..8 {
                                            acc[l] += x[l] * y[l];
                                        }
                                    }
                                    for (x, y) in av.remainder().iter().zip(bv.remainder()) {
                                        tail += x * y;
                                    }
                                }
                                let s = acc.iter().sum::<f32>() + tail;
                                *o = half::f16::from_f32(s);
                            }
                        });
                        let mut odims = ldims[..ldims.len() - 2].to_vec();
                        odims.push(m);
                        odims.push(n);
                        return Ok(Self::from_packed(
                            Arc::new(Storage::Cpu(CpuStorage::F16(out))),
                            Shape::from(odims),
                        ));
                    }
                }
                if self.dtype() == DType::F16 && rhs.dtype() == DType::F16 {
                    // non-CPU-contiguous f16 falls back to the f32 round-trip.
                    let dt = self.dtype();
                    return self
                        .to_dtype(DType::F32)?
                        .matmul_t(&rhs.to_dtype(DType::F32)?)?
                        .to_dtype(dt);
                }
                if k == k2 && lbatch == rbatch && self.dtype() == DType::F32 {
                    let a = self.f32_data()?;
                    let b = rhs.f32_data()?;
                    let mut out = vec![0.0f32; lbatch * m * n];
                    for bi in 0..lbatch {
                        gemm_nt_f32_cpu(
                            &a[bi * m * k..][..m * k],
                            &b[bi * n * k..][..n * k],
                            m,
                            k,
                            n,
                            &mut out[bi * m * n..][..m * n],
                        );
                    }
                    let mut odims = ldims[..ldims.len() - 2].to_vec();
                    odims.push(m);
                    odims.push(n);
                    return Ok(Self::from_packed(
                        Arc::new(Storage::Cpu(CpuStorage::F32(out))),
                        Shape::from(odims),
                    ));
                }
            }
        }
        // fallback: materialize the transpose
        let rank = rhs.dims().len();
        self.matmul(&rhs.transpose(rank - 2, rank - 1)?)
    }

    /// Standard-normal tensor matching `self`'s shape/device (Box-Muller
    /// over the process RNG; generated host-side, uploaded once - the VAE
    /// encode-path reparameterization sample).
    pub fn randn_like(&self) -> Result<Self> {
        let n = self.elem_count();
        let mut v = Vec::with_capacity(n + 1);
        // Box-Muller over the process stream: seeded when a caller asked for a
        // reproducible draw (see `set_global_seed`), OS-random otherwise.
        while v.len() < n {
            let u1: f32 = crate::tensor::rng::next_f32().max(1e-12);
            let u2: f32 = crate::tensor::rng::next_f32();
            let r = (-2.0f32 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            v.push(r * theta.cos());
            v.push(r * theta.sin());
        }
        v.truncate(n);
        Self::from_vec_f32(v, self.dims().to_vec())?.to_device(&self.device())
    }

    /// Numerically-stable softmax over the last dim.
    pub fn softmax_last_dim(&self) -> Result<Self> {
        if let Some(t) = self.dry_out(self.shape.clone(), self.dtype()) {
            return Ok(t);
        }
        // BF16 on CUDA runs natively: the kernel widens to f32 internally, so the
        // arithmetic is the f32 one and only the store is bf16. Routing it through
        // f32 tensors instead would cost two full passes over what is, in attention,
        // the largest buffer of the forward.
        #[cfg(feature = "cuda")]
        if self.dtype() == DType::BF16 {
            if let Storage::Cuda { data, dev } = self.storage().as_ref() {
                let last = *self
                    .dims()
                    .last()
                    .ok_or_else(|| Error("softmax on rank-0".into()))?;
                let rows = self.elem_count() / last;
                let out = crate::tensor::cuda::softmax_lastdim_bf16(
                    dev,
                    data.as_bf16_slice()?,
                    rows,
                    last,
                )?;
                return Self::from_cuda_storage(
                    crate::tensor::cuda::CudaStorage::BF16(out),
                    dev.clone(),
                    self.shape.clone(),
                );
            }
        }
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() {
            return self.f16_unary_via_f32(|t| t.softmax_last_dim());
        }
        #[cfg(feature = "cuda")]
        if let Storage::Cuda { data, dev } = self.storage().as_ref() {
            let last = *self
                .dims()
                .last()
                .ok_or_else(|| Error("softmax on rank-0".into()))?;
            let rows = self.elem_count() / last;
            let out =
                crate::tensor::cuda::softmax_lastdim_f32(dev, data.as_f32_slice()?, rows, last)?;
            return Self::from_cuda_storage(
                crate::tensor::cuda::CudaStorage::F32(out),
                dev.clone(),
                self.shape.clone(),
            );
        }
        let a = self.f32_data()?;
        let dims = self.dims();
        let last = *dims
            .last()
            .ok_or_else(|| Error("softmax on rank-0".into()))?;
        let rows = self.elem_count() / last;
        let mut out = vec![0.0f32; a.len()];
        let row = |src: &[f32], dst: &mut [f32]| {
            let max = src.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for (d, &s) in dst.iter_mut().zip(src) {
                let e = (s - max).exp();
                *d = e;
                sum += e;
            }
            for d in dst.iter_mut() {
                *d /= sum;
            }
        };
        if a.len() >= PAR_CPU_MIN && rows > 1 {
            use rayon::prelude::*;
            out.par_chunks_mut(last)
                .zip(a.par_chunks(last))
                .for_each(|(dst, src)| row(src, dst));
        } else {
            for r in 0..rows {
                row(&a[r * last..][..last], &mut out[r * last..][..last]);
            }
        }
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(CpuStorage::F32(out))),
            self.shape.clone(),
        ))
    }

    // -- The LLM decode-path ops (plan a.1 histogram) --

    /// Collapse to rank-1 (packed storage: pure reshape).
    pub fn flatten_all(&self) -> Result<Self> {
        self.reshape(vec![self.elem_count()])
    }

    /// Strict-dtype u32 readback (token ids / sort indices).
    pub fn to_vec_u32(&self) -> Result<Vec<u32>> {
        match self.host_storage()?.as_ref() {
            CpuStorage::U32(v) => Ok(v.clone()),
            other => Err(Error(format!("to_vec_u32 on {} tensor", other.dtype()))),
        }
    }

    /// Strict-dtype i64 readback (kv positions).
    pub fn to_vec_i64(&self) -> Result<Vec<i64>> {
        match self.host_storage()?.as_ref() {
            CpuStorage::I64(v) => Ok(v.clone()),
            other => Err(Error(format!("to_vec_i64 on {} tensor", other.dtype()))),
        }
    }

    /// Rank-checked accessors (the tensor-op `to_vec1/to_vec2/to_scalar` shapes).
    pub fn to_vec1_f32(&self) -> Result<Vec<f32>> {
        if self.rank() != 1 {
            return Err(Error(format!("to_vec1 on rank-{} tensor", self.rank())));
        }
        Ok(self.to_vec_f32())
    }

    pub fn to_vec1_u32(&self) -> Result<Vec<u32>> {
        if self.rank() != 1 {
            return Err(Error(format!("to_vec1 on rank-{} tensor", self.rank())));
        }
        self.to_vec_u32()
    }

    pub fn to_vec1_i64(&self) -> Result<Vec<i64>> {
        if self.rank() != 1 {
            return Err(Error(format!("to_vec1 on rank-{} tensor", self.rank())));
        }
        self.to_vec_i64()
    }

    pub fn to_vec2_f32(&self) -> Result<Vec<Vec<f32>>> {
        let (r, c) = self.shape.dims2()?;
        let v = self.to_vec_f32();
        Ok((0..r).map(|i| v[i * c..(i + 1) * c].to_vec()).collect())
    }

    pub fn to_scalar_f32(&self) -> Result<f32> {
        if self.elem_count() != 1 {
            return Err(Error(format!("to_scalar on {:?} tensor", self.dims())));
        }
        Ok(self.to_vec_f32()[0])
    }

    pub fn to_scalar_u32(&self) -> Result<u32> {
        if self.elem_count() != 1 {
            return Err(Error(format!("to_scalar on {:?} tensor", self.dims())));
        }
        Ok(self.to_vec_u32()?[0])
    }

    pub fn tanh(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() {
            return self.f16_unary_via_f32(|t| t.tanh());
        }
        #[cfg(feature = "cuda")]
        if let Some(t) = self.unary_cuda("native_tanh_f32")? {
            return Ok(t);
        }
        self.unary_f32(f32::tanh)
    }

    pub fn abs(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() {
            return self.f16_unary_via_f32(|t| t.abs());
        }
        #[cfg(feature = "cuda")]
        if let Some(t) = self.unary_cuda("native_abs_f32")? {
            return Ok(t);
        }
        self.unary_f32(f32::abs)
    }

    pub fn recip(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() {
            return self.f16_unary_via_f32(|t| t.recip());
        }
        #[cfg(feature = "cuda")]
        if let Some(t) = self.unary_cuda("native_recip_f32")? {
            return Ok(t);
        }
        self.unary_f32(|x| 1.0 / x)
    }

    pub fn sqrt(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() {
            return self.f16_unary_via_f32(|t| t.sqrt());
        }
        #[cfg(feature = "cuda")]
        if let Some(t) = self.unary_cuda("native_sqrt_f32")? {
            return Ok(t);
        }
        self.unary_f32(f32::sqrt)
    }
}
