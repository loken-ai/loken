//! Part of `impl Tensor`, split out of the parent module.
//!
//! Rust lets one inherent impl live in several modules of a crate, so this is the
//! same impl - only the file changed. Methods that were private are `pub(super)`
//! here, which is the reach they had when they sat beside their callers.

use super::*;

impl Tensor {
    /// Select index `i` along `dim` and drop that dim (narrow + squeeze).
    pub fn get_on_dim<I: Dim>(&self, dim: I, i: usize) -> Result<Self> {
        let d = dim.to_index(&self.shape, "get_on_dim")?;
        self.narrow(d, i, 1)?.squeeze(d)
    }

    /// LayerNorm over the last dim: mean-subtract, variance-normalize, affine.
    pub fn layer_norm(&self, weight: &Self, bias: Option<&Self>, eps: f32) -> Result<Self> {
        if let Some(t) = self.dry_out(self.shape.clone(), self.dtype()) {
            return Ok(t);
        }
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() {
            let w = weight.to_dtype(DType::F32)?;
            let b = bias.map(|b| b.to_dtype(DType::F32)).transpose()?;
            return self.f16_unary_via_f32(|t| t.layer_norm(&w, b.as_ref(), eps));
        }
        #[cfg(feature = "cuda")]
        if let Storage::Cuda { data, dev } = self.storage().as_ref() {
            let last = *self
                .dims()
                .last()
                .ok_or_else(|| Error("layer_norm on rank-0".into()))?;
            if weight.elem_count() != last {
                return Err(Error(format!(
                    "layer_norm: weight len {} != last dim {last}",
                    weight.elem_count()
                )));
            }
            let rows = self.elem_count() / last;
            // coerce params onto this device / f32 (model paths place them
            // device-side at load; this covers stragglers cheaply)
            let dev_t = self.device();
            let w = weight.to_device(&dev_t)?.to_dtype(DType::F32)?;
            let b = match bias {
                Some(b) => Some(b.to_device(&dev_t)?.to_dtype(DType::F32)?),
                None => None,
            };
            let (w_slice, _) = w.cuda_f32_slice()?;
            let b_slice = match &b {
                Some(b) => Some(b.cuda_f32_slice()?.0),
                None => None,
            };
            let out = crate::tensor::cuda::layer_norm_f32(
                dev,
                data.as_f32_slice()?,
                w_slice,
                b_slice,
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
            .ok_or_else(|| Error("layer_norm on rank-0".into()))?;
        if w.len() != last {
            return Err(Error(format!(
                "layer_norm: weight len {} != last dim {last}",
                w.len()
            )));
        }
        let bv = bias.map(|b| b.f32_data()).transpose()?;
        let rows = self.elem_count() / last;
        let mut out = vec![0.0f32; a.len()];
        for r in 0..rows {
            let src = &a[r * last..][..last];
            let dst = &mut out[r * last..][..last];
            let mean: f32 = src.iter().sum::<f32>() / last as f32;
            let var: f32 = src.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / last as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for (j, (d, &s)) in dst.iter_mut().zip(src).enumerate() {
                *d = (s - mean) * inv * w[j] + bv.map_or(0.0, |b| b[j]);
            }
        }
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(CpuStorage::F32(out))),
            self.shape.clone(),
        ))
    }

    pub fn relu(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() {
            return self.f16_unary_via_f32(|t| t.relu());
        }
        #[cfg(feature = "cuda")]
        if let Some(t) = self.unary_cuda("native_relu_f32")? {
            return Ok(t);
        }
        self.unary_f32(|x| x.max(0.0))
    }

    /// erf-based GELU (Abramowitz-Stegun 7.1.26 erf; the served encoders use
    /// the tanh form `gelu()` - this exists for config-completeness).
    pub fn gelu_erf(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.is_on_cuda() {
            // F32 on-device via the erff kernel (no CPU round-trip). Other dtypes fall through to
            // the F32-compute-on-CPU path below (rare; big-tensor F32 is the hot case, e.g. DiT MLP).
            if self.dtype() == DType::F32 {
                if let Some(t) = self.unary_cuda("native_gelu_erf_f32")? {
                    return Ok(t);
                }
            }
            let back = self.dtype();
            let cpu = self.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
            return cpu.gelu_erf()?.to_dtype(back)?.to_device(&self.device());
        }
        self.unary_f32(|x| {
            let z = x / std::f32::consts::SQRT_2;
            let t = 1.0 / (1.0 + 0.3275911 * z.abs());
            let poly = t
                * (0.254_829_6
                    + t * (-0.284_496_72
                        + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
            let erf = 1.0 - poly * (-z * z).exp();
            let erf = if z < 0.0 { -erf } else { erf };
            0.5 * x * (1.0 + erf)
        })
    }

    pub fn silu(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.dtype() == DType::F16 {
            // half-precision silu, reference-kernel parity (x / (1 + hexp(-x)))
            if let Storage::Cuda { data, dev } = self.storage().as_ref() {
                let out = crate::tensor::cuda::unary_f16(
                    dev,
                    "native_silu_f16",
                    data.as_f16_slice()?,
                    self.elem_count(),
                )?;
                return Self::from_cuda_storage(
                    crate::tensor::cuda::CudaStorage::F16(out),
                    dev.clone(),
                    self.shape.clone(),
                );
            }
        }
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() {
            return self.f16_unary_via_f32(|t| t.silu());
        }
        #[cfg(feature = "cuda")]
        if let Some(t) = self.unary_cuda("native_silu_f32")? {
            return Ok(t);
        }
        self.unary_f32(|x| x / (1.0 + (-x).exp()))
    }

    /// GELU, tanh approximation (the `gelu` the transformer MLPs use).
    pub fn gelu(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.dtype() == DType::F16 {
            // half-precision gelu chain, reference-kernel parity
            if let Storage::Cuda { data, dev } = self.storage().as_ref() {
                let out = crate::tensor::cuda::unary_f16(
                    dev,
                    "native_gelu_f16",
                    data.as_f16_slice()?,
                    self.elem_count(),
                )?;
                return Self::from_cuda_storage(
                    crate::tensor::cuda::CudaStorage::F16(out),
                    dev.clone(),
                    self.shape.clone(),
                );
            }
        }
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() {
            return self.f16_unary_via_f32(|t| t.gelu());
        }
        #[cfg(feature = "cuda")]
        if let Some(t) = self.unary_cuda("native_gelu_f32")? {
            return Ok(t);
        }
        self.unary_f32(|x| {
            0.5 * x
                * (1.0
                    + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
        })
    }

    /// Rotary embedding, non-interleaved (NeoX half-split): pairs `(i, i+d/2)`.
    /// `x`: [b, h, seq, d]; `cos`/`sin`: [seq, d/2].
    pub fn rope(&self, cos: &Self, sin: &Self) -> Result<Self> {
        self.rope_impl(cos, sin, false)
    }

    /// Rotary embedding, interleaved: pairs `(2i, 2i+1)`.
    pub fn rope_i(&self, cos: &Self, sin: &Self) -> Result<Self> {
        self.rope_impl(cos, sin, true)
    }

    pub(super) fn rope_impl(&self, cos: &Self, sin: &Self, interleaved: bool) -> Result<Self> {
        let (b, h, seq, d) = self.shape.dims4()?;
        // One buffer, the input's shape and dtype, on every device that runs this.
        // The rotation is a question about values; what it costs is not, and the
        // tables it reads were charged where they were built. Placed before the
        // half-precision route below so a counted half activation is not sent through
        // two casts a card never performs.
        if self.is_dry() {
            let dtype = self.dtype();
            let _ = (cos, sin, interleaved, b, h, seq, d);
            if let Some(t) = self.dry_out(self.shape.clone(), dtype) {
                return Ok(t);
            }
        }
        // CPU half-precision route: f32 compute + round back (see binary_f32).
        if matches!(self.dtype(), DType::F16 | DType::BF16) && !self.is_on_cuda() {
            let dt = self.dtype();
            let c = if cos.dtype() != DType::F32 {
                cos.to_dtype(DType::F32)?
            } else {
                cos.clone()
            };
            let s = if sin.dtype() != DType::F32 {
                sin.to_dtype(DType::F32)?
            } else {
                sin.clone()
            };
            return self
                .to_dtype(DType::F32)?
                .rope_impl(&c, &s, interleaved)?
                .to_dtype(dt);
        }
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() {
            // Preserve the INPUT half dtype - a BF16 input must come back BF16,
            // not F16. Hardcoding F16 here silently turned BF16 activations
            // into F16, so a later op mixing this with an untouched BF16 tensor
            // (e.g. rope(q) vs v in attention) failed with "expected f32 cuda
            // storage, got f16". (Surfaced by the z-image text encoder on GPU.)
            let back = self.dtype();
            let c = if matches!(cos.dtype(), DType::F16 | DType::BF16) {
                cos.to_dtype(DType::F32)?
            } else {
                cos.clone()
            };
            let sn = if matches!(sin.dtype(), DType::F16 | DType::BF16) {
                sin.to_dtype(DType::F32)?
            } else {
                sin.clone()
            };
            return self
                .to_dtype(DType::F32)?
                .rope_impl(&c, &sn, interleaved)?
                .to_dtype(back);
        }
        #[cfg(feature = "cuda")]
        if let Storage::Cuda { data, dev } = self.storage().as_ref() {
            let (c_slice, _) = cos
                .cuda_f32_slice()
                .map_err(|_| Error("rope: cos/sin must be on the same cuda device as x".into()))?;
            let (s_slice, _) = sin
                .cuda_f32_slice()
                .map_err(|_| Error("rope: cos/sin must be on the same cuda device as x".into()))?;
            let out = crate::tensor::cuda::rope_f32(
                dev,
                data.as_f32_slice()?,
                c_slice,
                s_slice,
                b * h,
                seq,
                d,
                interleaved,
            )?;
            return Self::from_cuda_storage(
                crate::tensor::cuda::CudaStorage::F32(out),
                dev.clone(),
                self.shape.clone(),
            );
        }
        let half = d / 2;
        let c = cos.f32_data()?;
        let s = sin.f32_data()?;
        if c.len() < seq * half || s.len() < seq * half {
            return Err(Error(format!(
                "rope: cos/sin too small for seq {seq} half {half}"
            )));
        }
        let x = self.f32_data()?;
        let mut out = vec![0.0f32; x.len()];
        for bi in 0..b * h {
            for t in 0..seq {
                let xrow = &x[(bi * seq + t) * d..][..d];
                let orow = &mut out[(bi * seq + t) * d..][..d];
                let crow = &c[t * half..][..half];
                let srow = &s[t * half..][..half];
                if interleaved {
                    for i in 0..half {
                        let (x0, x1) = (xrow[2 * i], xrow[2 * i + 1]);
                        orow[2 * i] = x0 * crow[i] - x1 * srow[i];
                        orow[2 * i + 1] = x0 * srow[i] + x1 * crow[i];
                    }
                } else {
                    for i in 0..half {
                        let (x0, x1) = (xrow[i], xrow[i + half]);
                        orow[i] = x0 * crow[i] - x1 * srow[i];
                        orow[i + half] = x0 * srow[i] + x1 * crow[i];
                    }
                }
            }
        }
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(CpuStorage::F32(out))),
            self.shape.clone(),
        ))
    }

    pub(super) fn reduce_dim<I: Dim>(
        &self,
        dim: I,
        keepdim: bool,
        op: &'static str,
        init: f32,
        f: impl Fn(f32, f32) -> f32,
        finish: impl Fn(f32, usize) -> f32,
    ) -> Result<Self> {
        let d = dim.to_index(&self.shape, op)?;
        // One buffer with the reduced axis gone or flattened to one - the same shape
        // the device arm below computes, from the same two lines. What the reduction
        // COMPUTES needs values; what it leaves behind does not.
        if self.is_dry() {
            let mut odims = self.dims().to_vec();
            if keepdim {
                odims[d] = 1;
            } else {
                odims.remove(d);
            }
            let dtype = self.dtype();
            if let Some(t) = self.dry_out(odims, dtype) {
                return Ok(t);
            }
        }
        #[cfg(feature = "cuda")]
        if self.is_on_cuda() {
            // On-device reduction for the known ops (sum/mean/max) - the generic closure path below
            // can't run on the GPU, so it used to round-trip the whole tensor to the CPU and back
            // (a hidden PCIe cost per call). F32 CUDA storage only; anything else falls through.
            let dims = self.dims();
            let red = dims[d];
            let opscale = match op {
                "sum" | "sum_keepdim" => Some((0i32, 1.0f32)),
                "mean_keepdim" => Some((0i32, 1.0 / red as f32)),
                "max_keepdim" => Some((1i32, 1.0f32)),
                _ => None,
            };
            if let (
                Some((opcode, scale)),
                Storage::Cuda {
                    data: crate::tensor::cuda::CudaStorage::F32(s),
                    dev,
                },
            ) = (opscale, self.storage().as_ref())
            {
                let outer: usize = dims[..d].iter().product();
                let inner: usize = dims[d + 1..].iter().product();
                let out =
                    crate::tensor::cuda::reduce_dim_f32(dev, s, outer, red, inner, opcode, scale)?;
                let mut odims = dims.to_vec();
                if keepdim {
                    odims[d] = 1;
                } else {
                    odims.remove(d);
                }
                return Self::from_cuda_storage(
                    crate::tensor::cuda::CudaStorage::F32(out),
                    dev.clone(),
                    odims,
                );
            }
            let dev = self.device();
            let cpu = self.to_device(&Device::Cpu)?;
            return cpu
                .reduce_dim(d, keepdim, op, init, f, finish)?
                .to_device(&dev);
        }
        let dims = self.dims();
        let outer: usize = dims[..d].iter().product();
        let red = dims[d];
        let inner: usize = dims[d + 1..].iter().product();
        let a = self.f32_data()?;
        let mut out = vec![init; outer * inner];
        for o in 0..outer {
            for r in 0..red {
                let base = (o * red + r) * inner;
                for i in 0..inner {
                    let slot = &mut out[o * inner + i];
                    *slot = f(*slot, a[base + i]);
                }
            }
        }
        for v in out.iter_mut() {
            *v = finish(*v, red);
        }
        let mut odims = dims.to_vec();
        if keepdim {
            odims[d] = 1;
        } else {
            odims.remove(d);
        }
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(CpuStorage::F32(out))),
            Shape::from(odims),
        ))
    }

    pub fn sum_keepdim<I: Dim>(&self, dim: I) -> Result<Self> {
        self.reduce_dim(dim, true, "sum_keepdim", 0.0, |a, b| a + b, |v, _| v)
    }

    pub fn sum<I: Dim>(&self, dim: I) -> Result<Self> {
        self.reduce_dim(dim, false, "sum", 0.0, |a, b| a + b, |v, _| v)
    }

    pub fn mean_keepdim<I: Dim>(&self, dim: I) -> Result<Self> {
        self.reduce_dim(
            dim,
            true,
            "mean_keepdim",
            0.0,
            |a, b| a + b,
            |v, n| v / n as f32,
        )
    }

    pub fn max_keepdim<I: Dim>(&self, dim: I) -> Result<Self> {
        self.reduce_dim(
            dim,
            true,
            "max_keepdim",
            f32::NEG_INFINITY,
            f32::max,
            |v, _| v,
        )
    }

    /// Convert element type (f32 <-> f16/bf16; integer widths preserved).
    #[track_caller]
    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        if self.dtype() == dtype {
            return Ok(self.clone());
        }
        // The cast copies nobody counts. A projection run at the weight's width and
        // read back at the activation's allocates a full second copy of the tensor
        // on each side, and at a denoiser's sequence length that is the difference
        // between a model that fits a card and one that does not.
        if let Some(t) = self.dry_out(self.shape.clone(), dtype) {
            return Ok(t);
        }
        #[cfg(feature = "cuda")]
        if let Storage::Cuda { data, dev } = self.storage().as_ref() {
            let _b = crate::tensor::bounce::start("dev_cast");
            let n = self.elem_count();
            match (self.dtype(), dtype) {
                (DType::F32, DType::F16) => {
                    let out = crate::tensor::cuda::cast_f32_to_f16(dev, data.as_f32_slice()?, n)?;
                    return Self::from_cuda_storage(
                        crate::tensor::cuda::CudaStorage::F16(out),
                        dev.clone(),
                        self.shape.clone(),
                    );
                }
                (DType::F16, DType::F32) => {
                    let out = crate::tensor::cuda::cast_f16_to_f32(dev, data.as_f16_slice()?, n)?;
                    return Self::from_cuda_storage(
                        crate::tensor::cuda::CudaStorage::F32(out),
                        dev.clone(),
                        self.shape.clone(),
                    );
                }
                (DType::F32, DType::BF16) => {
                    let out = crate::tensor::cuda::cast_f32_to_bf16(dev, data.as_f32_slice()?, n)?;
                    return Self::from_cuda_storage(
                        crate::tensor::cuda::CudaStorage::BF16(out),
                        dev.clone(),
                        self.shape.clone(),
                    );
                }
                (DType::BF16, DType::F32) => {
                    let out = crate::tensor::cuda::cast_bf16_to_f32(dev, data.as_bf16_slice()?, n)?;
                    return Self::from_cuda_storage(
                        crate::tensor::cuda::CudaStorage::F32(out),
                        dev.clone(),
                        self.shape.clone(),
                    );
                }
                // half <-> half: route through f32 on device (two casts,
                // no host roundtrip)
                (DType::F16, DType::BF16) | (DType::BF16, DType::F16) => {
                    return self.to_dtype(DType::F32)?.to_dtype(dtype);
                }
                _ => return self.host_bounce(|cpu| cpu.to_dtype(dtype)),
            }
        }
        // integer targets: direct typed casts (token ids / kv positions - the
        // f32 route would clip the i64 range)
        if matches!(dtype, DType::U8 | DType::U32 | DType::I64) {
            let c = self.cpu_storage_ref()?;
            let vi: Vec<i64> = match c {
                CpuStorage::U8(v) => v.iter().map(|&x| x as i64).collect(),
                CpuStorage::U32(v) => v.iter().map(|&x| x as i64).collect(),
                CpuStorage::I64(v) => v.clone(),
                other => other.to_f32_vec().iter().map(|&x| x as i64).collect(),
            };
            let storage = match dtype {
                DType::U8 => CpuStorage::U8(vi.iter().map(|&x| x as u8).collect()),
                DType::U32 => CpuStorage::U32(vi.iter().map(|&x| x as u32).collect()),
                _ => CpuStorage::I64(vi),
            };
            return Ok(Self::from_packed(
                Arc::new(Storage::Cpu(storage)),
                self.shape.clone(),
            ));
        }
        // Fused half↔half narrowing on the CPU: convert bf16↔f16 in a SINGLE parallel
        // pass, skipping the full f32 intermediate Vec (22 GB for umT5-XXL's 5.5B bf16
        // elements - roughly half the CPU-staged cast cost). `x.to_f32()` then `from_f32`
        // per element is exactly what to_f32_vec()->map did, so the result is bit-identical.
        if let Storage::Cpu(c) = self.storage().as_ref() {
            match (c, dtype) {
                (CpuStorage::BF16(v), DType::F16) => {
                    let f = half::slice::HalfFloatSliceExt::to_f32_vec(v.as_slice());
                    return Ok(Self::from_packed(
                        Arc::new(Storage::Cpu(CpuStorage::F16(narrow_from_f32::<half::f16>(
                            &f,
                        )))),
                        self.shape.clone(),
                    ));
                }
                (CpuStorage::F16(v), DType::BF16) => {
                    let f = half::slice::HalfFloatSliceExt::to_f32_vec(v.as_slice());
                    return Ok(Self::from_packed(
                        Arc::new(Storage::Cpu(CpuStorage::BF16(
                            narrow_from_f32::<half::bf16>(&f),
                        ))),
                        self.shape.clone(),
                    ));
                }
                // F32 -> half: narrow DIRECTLY from the f32 slice. The generic path
                // below calls `to_f32_vec()` first, which CLONEs the whole f32 buffer
                // (2.68 GB for a 131072x5120 lm_head) before the map - pure waste when
                // the source is already f32. Bit-identical, just no clone.
                (CpuStorage::F32(v), DType::F16) => {
                    return Ok(Self::from_packed(
                        Arc::new(Storage::Cpu(CpuStorage::F16(narrow_from_f32::<half::f16>(
                            v,
                        )))),
                        self.shape.clone(),
                    ));
                }
                (CpuStorage::F32(v), DType::BF16) => {
                    return Ok(Self::from_packed(
                        Arc::new(Storage::Cpu(CpuStorage::BF16(
                            narrow_from_f32::<half::bf16>(v),
                        ))),
                        self.shape.clone(),
                    ));
                }
                _ => {}
            }
        }
        let f = match self.storage().as_ref() {
            Storage::Cpu(c) => c.to_f32_vec(),
            #[cfg(feature = "cuda")]
            Storage::Cuda { .. } => {
                return Err(Error("to_dtype on cuda not implemented yet".into()))
            }
            Storage::Dry(_) => return Err(Error("to_dtype: unreachable on a dry tensor".into())),
        };
        // Parallel narrowing pass: the per-element f32->{f16,bf16} map runs over every
        // weight element (billions for a big encoder like umT5-XXL) and was the dominant
        // cost of a CPU-staged dtype cast. rayon's indexed collect preserves order, so
        // the result is bit-identical to the sequential map.
        use rayon::prelude::*;
        let storage = match dtype {
            DType::F32 => CpuStorage::F32(f),
            DType::F16 => CpuStorage::F16(narrow_from_f32::<half::f16>(&f)),
            DType::BF16 => CpuStorage::BF16(narrow_from_f32::<half::bf16>(&f)),
            DType::F64 => CpuStorage::F64(f.par_iter().map(|&x| x as f64).collect()),
            other => return Err(Error(format!("to_dtype: unsupported target {other}"))),
        };
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(storage)),
            self.shape.clone(),
        ))
    }

    /// 1-D convolution `[b, c_in, l] * [c_out, c_in/groups, k] -> [b, c_out, l_out]`
    /// via im2col + matmul (CPU; CUDA tensors host-bounce until a kernel is
    /// warranted - the conv consumers run once per request).
    pub fn conv1d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.is_on_cuda() {
            // device path: im2col + cuBLAS per batch (groups==1, f32; the
            // general cases host-bounce)
            if groups == 1 && self.dtype() == DType::F32 && kernel.dtype() == DType::F32 {
                if let (
                    Storage::Cuda { data, dev },
                    Storage::Cuda {
                        data: kd,
                        dev: kdev,
                    },
                ) = (self.storage().as_ref(), kernel.storage().as_ref())
                {
                    if dev.ordinal() == kdev.ordinal() {
                        let (b, c_in, l) = self.shape.dims3()?;
                        let (c_out, c_in_g, k) = kernel.shape().dims3()?;
                        if c_in_g != c_in {
                            return Err(Error("conv1d: kernel/input c_in mismatch".into()));
                        }
                        let l_out = (l + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
                        let x = data.as_f32_slice()?;
                        let w = kd.as_f32_slice()?;
                        let st = dev.stream();
                        let cuda_run = || -> Result<Self> {
                            let mut out =
                                crate::tensor::cuda::with_oom_retry(dev, "conv1d out", || {
                                    st.alloc_zeros::<f32>(b * c_out * l_out)
                                })?;
                            // Bound the im2col transient: long sequences (audio
                            // decode) would otherwise need c_in*k*l_out floats in
                            // one allocation - multi-GB and an OOM class. Tile
                            // the output columns to a fixed budget instead.
                            const COL_BUDGET_ELEMS: usize = 32 << 20; // 128 MB f32
                            let tile = (COL_BUDGET_ELEMS / (c_in * k).max(1))
                                .max(1)
                                .min(l_out.max(1));
                            for bi in 0..b {
                                let xv = x.slice(bi * c_in * l..(bi + 1) * c_in * l);
                                let mut lo0 = 0usize;
                                while lo0 < l_out {
                                    let lt = tile.min(l_out - lo0);
                                    let col = crate::tensor::cuda::im2col1d_f32(
                                        dev, &xv, c_in, l, k, l_out, lo0, lt, padding, stride,
                                        dilation,
                                    )?;
                                    let y = crate::tensor::cuda::matmul_f32(
                                        dev,
                                        w,
                                        &col,
                                        1,
                                        c_out,
                                        c_in * k,
                                        lt,
                                    )?;
                                    if lt == l_out {
                                        // Single tile: one contiguous copy.
                                        let mut dst = out.slice_mut(
                                            bi * c_out * l_out..(bi + 1) * c_out * l_out,
                                        );
                                        st.memcpy_dtod(&y, &mut dst)
                                            .map_err(|e| Error(format!("conv1d copy: {e}")))?;
                                    } else {
                                        // Strided landing: y is [c_out, lt], out rows are l_out apart.
                                        for co in 0..c_out {
                                            let src = y.slice(co * lt..(co + 1) * lt);
                                            let base = bi * c_out * l_out + co * l_out + lo0;
                                            let mut dst = out.slice_mut(base..base + lt);
                                            st.memcpy_dtod(&src, &mut dst)
                                                .map_err(|e| Error(format!("conv1d copy: {e}")))?;
                                        }
                                    }
                                    lo0 += lt;
                                }
                            }
                            Self::from_cuda_storage(
                                crate::tensor::cuda::CudaStorage::F32(out),
                                dev.clone(),
                                vec![b, c_out, l_out],
                            )
                        };
                        return match cuda_run() {
                            // Truly out of VRAM even tiled + after reclaim:
                            // host-bounce (no-OOM guarantee over throughput).
                            Err(e) if e.is_oom() => {
                                crate::tensor::bounce::note_pressure_bounce();
                                tracing::warn!(
                                    "conv1d [{b},{c_in},{l}]: GPU{} OOM after reclaim; bouncing to CPU",
                                    dev.ordinal()
                                );
                                let kc = kernel.to_device(&Device::Cpu)?;
                                self.host_bounce(|cpu| {
                                    cpu.conv1d(&kc, padding, stride, dilation, groups)
                                })
                            }
                            r => r,
                        };
                    }
                }
            }
            let kc = kernel.to_device(&Device::Cpu)?;
            return self.host_bounce(|cpu| cpu.conv1d(&kc, padding, stride, dilation, groups));
        }
        let (b, c_in, l) = self.shape.dims3()?;
        let (c_out, c_in_g, k) = kernel.shape().dims3()?;
        if c_in_g * groups != c_in {
            return Err(Error(format!(
                "conv1d: kernel c_in {c_in_g}x{groups} != input c_in {c_in}"
            )));
        }
        let l_out = (l + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
        let x = self.f32_data()?;
        let w = kernel.f32_data()?;
        let co_g = c_out / groups;
        let mut out = vec![0f32; b * c_out * l_out];
        for bi in 0..b {
            for g in 0..groups {
                // im2col for this group: [c_in_g * k, l_out]
                let mut col = vec![0f32; c_in_g * k * l_out];
                for ci in 0..c_in_g {
                    let src = &x[(bi * c_in + g * c_in_g + ci) * l..][..l];
                    for kk in 0..k {
                        for lo in 0..l_out {
                            let pos = lo * stride + kk * dilation;
                            let v = if pos < padding || pos - padding >= l {
                                0.0
                            } else {
                                src[pos - padding]
                            };
                            col[(ci * k + kk) * l_out + lo] = v;
                        }
                    }
                }
                // [co_g, c_in_g*k] x [c_in_g*k, l_out]
                let wg = &w[g * co_g * c_in_g * k..][..co_g * c_in_g * k];
                for co in 0..co_g {
                    let orow = &mut out[(bi * c_out + g * co_g + co) * l_out..][..l_out];
                    let wrow = &wg[co * c_in_g * k..][..c_in_g * k];
                    for (ki, &wv) in wrow.iter().enumerate() {
                        if wv == 0.0 {
                            continue;
                        }
                        let crow = &col[ki * l_out..][..l_out];
                        for lo in 0..l_out {
                            orow[lo] += wv * crow[lo];
                        }
                    }
                }
            }
        }
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(CpuStorage::F32(out))),
            Shape::from(vec![b, c_out, l_out]),
        ))
    }

    /// 2-D convolution `[b, c_in, h, w] * [c_out, c_in/groups, kh, kw]`.
    /// Elements the 2-D convolution keeps live in its im2col column buffer.
    ///
    /// The column matrix of a 3x3 convolution over a megapixel map is multi-gigabyte, so
    /// `conv2d` windows it instead of materialising it. Anything SIZING a convolution has to
    /// charge this bound rather than the full column matrix: the VAE decode estimate charged
    /// the full one and made a 1024^2 decode look like 5.9 GB when it needs under 2, so the
    /// most common render size tiled its decode on every request for no reason.
    pub const CONV2D_COL_BUDGET_ELEMS: usize = 48 << 20;

    pub fn conv2d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.is_on_cuda() {
            if groups == 1 && self.dtype() == DType::F32 && kernel.dtype() == DType::F32 {
                if let (
                    Storage::Cuda { data, dev },
                    Storage::Cuda {
                        data: kd,
                        dev: kdev,
                    },
                ) = (self.storage().as_ref(), kernel.storage().as_ref())
                {
                    if dev.ordinal() == kdev.ordinal() {
                        let (b, c_in, h, w_in) = self.shape.dims4()?;
                        let (c_out, c_in_g, kh, kw) = kernel.shape().dims4()?;
                        if c_in_g != c_in {
                            return Err(Error("conv2d: kernel/input c_in mismatch".into()));
                        }
                        let h_out = (h + 2 * padding - dilation * (kh - 1) - 1) / stride + 1;
                        let w_out = (w_in + 2 * padding - dilation * (kw - 1) - 1) / stride + 1;
                        let hw_out = h_out * w_out;
                        let patch = c_in * kh * kw;
                        let x = data.as_f32_slice()?;
                        let w = kd.as_f32_slice()?;
                        let st = dev.stream();
                        let cuda_run = || -> Result<Self> {
                            // every (bi) slice is overwritten by the dtod copies below
                            let mut out = crate::tensor::cuda::with_oom_retry(
                                dev,
                                "conv2d out",
                                || unsafe { st.alloc::<f32>(b * c_out * hw_out) },
                            )?;
                            // Bound the im2col transient: megapixel feature maps with 3x3
                            // kernels would otherwise ask for multi-GB col buffers and OOM
                            // next to resident models.
                            let max_cols = (Self::CONV2D_COL_BUDGET_ELEMS / patch.max(1)).max(1);
                            for bi in 0..b {
                                let xv = x.slice(bi * c_in * h * w_in..(bi + 1) * c_in * h * w_in);
                                let mut j0 = 0;
                                while j0 < hw_out {
                                    let cl = max_cols.min(hw_out - j0);
                                    let col = crate::tensor::cuda::im2col2d_f32(
                                        dev, &xv, c_in, h, w_in, kh, kw, w_out, j0, cl, padding,
                                        stride, dilation,
                                    )?;
                                    // TF32 tensor-core gemm: conv feeds image
                                    // pixels; ~1e-3 rel error is invisible at u8
                                    // output scale.
                                    let y = crate::tensor::cuda::matmul_f32_tf32(
                                        dev, w, &col, 1, c_out, patch, cl,
                                    )?;
                                    if cl == hw_out {
                                        let mut dst = out.slice_mut(
                                            bi * c_out * hw_out..(bi + 1) * c_out * hw_out,
                                        );
                                        st.memcpy_dtod(&y, &mut dst)
                                            .map_err(|e| Error(format!("conv2d copy: {e}")))?;
                                    } else {
                                        for co in 0..c_out {
                                            let src = y.slice(co * cl..(co + 1) * cl);
                                            let base = (bi * c_out + co) * hw_out + j0;
                                            let mut dst = out.slice_mut(base..base + cl);
                                            st.memcpy_dtod(&src, &mut dst).map_err(|e| {
                                                Error(format!("conv2d chunk copy: {e}"))
                                            })?;
                                        }
                                    }
                                    j0 += cl;
                                }
                            }
                            Self::from_cuda_storage(
                                crate::tensor::cuda::CudaStorage::F32(out),
                                dev.clone(),
                                vec![b, c_out, h_out, w_out],
                            )
                        };
                        return match cuda_run() {
                            Err(e) if e.is_oom() => {
                                crate::tensor::bounce::note_pressure_bounce();
                                tracing::warn!(
                                    "conv2d [{b},{c_in},{h},{w_in}]: GPU{} OOM after reclaim; bouncing to CPU",
                                    dev.ordinal()
                                );
                                let kc = kernel.to_device(&Device::Cpu)?;
                                self.host_bounce(|cpu| {
                                    cpu.conv2d(&kc, padding, stride, dilation, groups)
                                })
                            }
                            r => r,
                        };
                    }
                }
            }
            let kc = kernel.to_device(&Device::Cpu)?;
            return self.host_bounce(|cpu| cpu.conv2d(&kc, padding, stride, dilation, groups));
        }
        let (b, c_in, h, w_in) = self.shape.dims4()?;
        let (c_out, c_in_g, kh, kw) = kernel.shape().dims4()?;
        if c_in_g * groups != c_in {
            return Err(Error(format!(
                "conv2d: kernel c_in {c_in_g}x{groups} != input c_in {c_in}"
            )));
        }
        let h_out = (h + 2 * padding - dilation * (kh - 1) - 1) / stride + 1;
        let w_out = (w_in + 2 * padding - dilation * (kw - 1) - 1) / stride + 1;
        let x = self.f32_data()?;
        let wk = kernel.f32_data()?;
        let co_g = c_out / groups;
        let hw_out = h_out * w_out;
        let mut out = vec![0f32; b * c_out * hw_out];
        for bi in 0..b {
            for g in 0..groups {
                let patch = c_in_g * kh * kw;
                let mut col = vec![0f32; patch * hw_out];
                {
                    use rayon::prelude::*;
                    col.par_chunks_mut(kh * kw * hw_out)
                        .enumerate()
                        .for_each(|(ci, cslab)| {
                            let src = &x[(bi * c_in + g * c_in_g + ci) * h * w_in..][..h * w_in];
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let prow = ki * kw + kj;
                                    for ho in 0..h_out {
                                        let hi = ho * stride + ki * dilation;
                                        for wo in 0..w_out {
                                            let wi = wo * stride + kj * dilation;
                                            let v = if hi < padding
                                                || wi < padding
                                                || hi - padding >= h
                                                || wi - padding >= w_in
                                            {
                                                0.0
                                            } else {
                                                src[(hi - padding) * w_in + (wi - padding)]
                                            };
                                            cslab[prow * hw_out + ho * w_out + wo] = v;
                                        }
                                    }
                                }
                            }
                        });
                }
                let wg = &wk[g * co_g * patch..][..co_g * patch];
                // [co_g, patch] . [patch, hw_out] through the blocked parallel
                // gemm - the VAE CPU fallback runs megapixel feature maps here.
                gemm_f32_cpu(
                    wg,
                    &col,
                    co_g,
                    patch,
                    hw_out,
                    &mut out[(bi * c_out + g * co_g) * hw_out..][..co_g * hw_out],
                );
            }
        }
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(CpuStorage::F32(out))),
            Shape::from(vec![b, c_out, h_out, w_out]),
        ))
    }

    /// 1-D transposed convolution `[b, c_in, l] * [c_in, c_out/groups, k]`
    /// (stride upsampling). CUDA: direct gather kernel (dilation 1, groups 1
    /// - the audio-codec shape); CPU: scatter-accumulate, rayon over output
    /// channels.
    pub fn conv_transpose1d(
        &self,
        kernel: &Self,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.is_on_cuda() {
            if groups == 1
                && dilation == 1
                && self.dtype() == DType::F32
                && kernel.dtype() == DType::F32
            {
                if let (
                    Storage::Cuda { data, dev },
                    Storage::Cuda {
                        data: kd,
                        dev: kdev,
                    },
                ) = (self.storage().as_ref(), kernel.storage().as_ref())
                {
                    if dev.ordinal() == kdev.ordinal() {
                        let (b, c_in, l_in) = self.shape.dims3()?;
                        let (c_in_k, c_out, k) = kernel.shape().dims3()?;
                        if c_in_k != c_in {
                            return Err(Error(format!(
                                "conv_transpose1d: c_in {c_in_k} != {c_in}"
                            )));
                        }
                        let l_full = (l_in - 1) * stride + (k - 1) + output_padding + 1;
                        if l_full <= 2 * padding {
                            return Err(Error(
                                "conv_transpose1d: output smaller than padding".into(),
                            ));
                        }
                        let l_out = l_full - 2 * padding;
                        let x = data.as_f32_slice()?;
                        let w = kd.as_f32_slice()?;
                        let st = dev.stream();
                        let cuda_run = || -> Result<Self> {
                            let mut out =
                                crate::tensor::cuda::with_oom_retry(dev, "convt out", || unsafe {
                                    st.alloc::<f32>(b * c_out * l_out)
                                })?;
                            for bi in 0..b {
                                let xv = x.slice(bi * c_in * l_in..(bi + 1) * c_in * l_in);
                                let y = crate::tensor::cuda::convt1d_f32(
                                    dev, &xv, w, c_in, c_out, l_in, l_out, k, stride, padding,
                                )?;
                                let mut dst =
                                    out.slice_mut(bi * c_out * l_out..(bi + 1) * c_out * l_out);
                                st.memcpy_dtod(&y, &mut dst)
                                    .map_err(|e| Error(format!("convt copy: {e}")))?;
                            }
                            Self::from_cuda_storage(
                                crate::tensor::cuda::CudaStorage::F32(out),
                                dev.clone(),
                                vec![b, c_out, l_out],
                            )
                        };
                        return match cuda_run() {
                            Err(e) if e.is_oom() => {
                                crate::tensor::bounce::note_pressure_bounce();
                                tracing::warn!(
                                    "conv_transpose1d [{b},{c_in},{l_in}]: GPU{} OOM after reclaim; bouncing to CPU",
                                    dev.ordinal()
                                );
                                let kc = kernel.to_device(&Device::Cpu)?;
                                self.host_bounce(|cpu| {
                                    cpu.conv_transpose1d(
                                        &kc,
                                        padding,
                                        output_padding,
                                        stride,
                                        dilation,
                                        groups,
                                    )
                                })
                            }
                            r => r,
                        };
                    }
                }
            }
            let kc = kernel.to_device(&Device::Cpu)?;
            return self.host_bounce(|cpu| {
                cpu.conv_transpose1d(&kc, padding, output_padding, stride, dilation, groups)
            });
        }
        let (b, c_in, l) = self.shape.dims3()?;
        let (c_in_k, co_g, k) = kernel.shape().dims3()?;
        if c_in_k != c_in {
            return Err(Error(format!("conv_transpose1d: c_in {c_in_k} != {c_in}")));
        }
        let c_out = co_g * groups;
        let ci_g = c_in / groups;
        let l_out = (l - 1) * stride + dilation * (k - 1) + output_padding + 1;
        if l_out <= 2 * padding {
            return Err(Error(
                "conv_transpose1d: output smaller than padding".into(),
            ));
        }
        let l_out = l_out - 2 * padding;
        let x = self.f32_data()?;
        let w = kernel.f32_data()?;
        let mut out = vec![0f32; b * c_out * l_out];
        for bi in 0..b {
            for g in 0..groups {
                for ci in 0..ci_g {
                    let src = &x[(bi * c_in + g * ci_g + ci) * l..][..l];
                    for co in 0..co_g {
                        let wrow = &w[((g * ci_g + ci) * co_g + co) * k..][..k];
                        let orow = &mut out[(bi * c_out + g * co_g + co) * l_out..][..l_out];
                        for (li, &xv) in src.iter().enumerate() {
                            if xv == 0.0 {
                                continue;
                            }
                            for (kk, &wv) in wrow.iter().enumerate() {
                                let pos = li * stride + kk * dilation;
                                if pos >= padding && pos - padding < l_out {
                                    orow[pos - padding] += xv * wv;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(CpuStorage::F32(out))),
            Shape::from(vec![b, c_out, l_out]),
        ))
    }

    /// GroupNorm over `[b, c, ...spatial]`: normalize each (batch, group)
    /// slab of `c/groups` channels x spatial, then per-channel affine.
    pub fn group_norm(
        &self,
        num_groups: usize,
        weight: &Self,
        bias: &Self,
        eps: f32,
    ) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() {
            let w = weight.to_dtype(DType::F32)?;
            let bs = bias.to_dtype(DType::F32)?;
            return self.f16_unary_via_f32(|t| t.group_norm(num_groups, &w, &bs, eps));
        }
        #[cfg(feature = "cuda")]
        if let Storage::Cuda { data, dev } = self.storage().as_ref() {
            let dims = self.dims();
            if dims.len() < 2 {
                return Err(Error("group_norm needs [b, c, ...]".into()));
            }
            let (b, c) = (dims[0], dims[1]);
            if c % num_groups != 0 {
                return Err(Error(format!(
                    "group_norm: c {c} % groups {num_groups} != 0"
                )));
            }
            let spatial: usize = dims[2..].iter().product();
            // weight/bias off-device -> correctness fallback through host.
            if let (Ok((w_slice, w_dev)), Ok((b_slice, b_dev))) =
                (weight.cuda_f32_slice(), bias.cuda_f32_slice())
            {
                if w_dev.ordinal() == dev.ordinal() && b_dev.ordinal() == dev.ordinal() {
                    let out = crate::tensor::cuda::group_norm_f32(
                        dev,
                        data.as_f32_slice()?,
                        w_slice,
                        b_slice,
                        b,
                        num_groups,
                        c / num_groups,
                        spatial,
                        eps,
                    )?;
                    return Self::from_cuda_storage(
                        crate::tensor::cuda::CudaStorage::F32(out),
                        dev.clone(),
                        self.shape.clone(),
                    );
                }
            }
            let w = weight.to_device(&Device::Cpu)?;
            let bs = bias.to_device(&Device::Cpu)?;
            return self.host_bounce(|cpu| cpu.group_norm(num_groups, &w, &bs, eps));
        }
        let dims = self.dims().to_vec();
        if dims.len() < 2 {
            return Err(Error("group_norm needs [b, c, ...]".into()));
        }
        let (b, c) = (dims[0], dims[1]);
        if c % num_groups != 0 {
            return Err(Error(format!(
                "group_norm: c {c} % groups {num_groups} != 0"
            )));
        }
        let spatial: usize = dims[2..].iter().product();
        let cg = c / num_groups;
        let x = self.f32_data()?;
        let w = weight.f32_data()?;
        let bv = bias.f32_data()?;
        if w.len() != c || bv.len() != c {
            return Err(Error("group_norm: affine params must be [c]".into()));
        }
        let mut out = vec![0f32; x.len()];
        let slab = cg * spatial;
        let one = |bg: usize, dslab: &mut [f32]| {
            let g = bg % num_groups;
            let src = &x[bg * slab..][..slab];
            let mean: f32 = src.iter().sum::<f32>() / slab as f32;
            let var: f32 = src.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / slab as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for ci in 0..cg {
                let ch = g * cg + ci;
                let (wv, bvv) = (w[ch], bv[ch]);
                let s = &src[ci * spatial..][..spatial];
                let d = &mut dslab[ci * spatial..][..spatial];
                for (dd, &sv) in d.iter_mut().zip(s) {
                    *dd = (sv - mean) * inv * wv + bvv;
                }
            }
        };
        if x.len() >= PAR_CPU_MIN && b * num_groups > 1 {
            use rayon::prelude::*;
            out.par_chunks_mut(slab)
                .enumerate()
                .for_each(|(bg, dslab)| one(bg, dslab));
        } else {
            for bg in 0..b * num_groups {
                one(bg, &mut out[bg * slab..][..slab]);
            }
        }
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(CpuStorage::F32(out))),
            self.shape.clone(),
        ))
    }

    /// Snake activation over `[b, c, l]`: `x + inv_alpha[c] * sin(alpha[c]*x)²`
    /// (fused - the audio-codec per-channel form; `inv_alpha` is precomputed
    /// at load as `1/(alpha+eps)`).
    pub fn snake1d(&self, alpha: &Self, inv_alpha: &Self) -> Result<Self> {
        let (_b, c, l) = self.shape().dims3()?;
        if alpha.elem_count() != c || inv_alpha.elem_count() != c {
            return Err(Error(format!(
                "snake1d: alpha {} / inv {} != channels {c}",
                alpha.elem_count(),
                inv_alpha.elem_count()
            )));
        }
        #[cfg(feature = "cuda")]
        if let Storage::Cuda { data, dev } = self.storage().as_ref() {
            if let (Ok((a_s, a_d)), Ok((i_s, i_d))) =
                (alpha.cuda_f32_slice(), inv_alpha.cuda_f32_slice())
            {
                if a_d.ordinal() == dev.ordinal() && i_d.ordinal() == dev.ordinal() {
                    let out = crate::tensor::cuda::snake1d_f32(
                        dev,
                        data.as_f32_slice()?,
                        a_s,
                        i_s,
                        self.elem_count(),
                        l,
                        c,
                    )?;
                    return Self::from_cuda_storage(
                        crate::tensor::cuda::CudaStorage::F32(out),
                        dev.clone(),
                        self.shape.clone(),
                    );
                }
            }
            let a = alpha.to_device(&Device::Cpu)?;
            let i = inv_alpha.to_device(&Device::Cpu)?;
            return self.host_bounce(|cpu| cpu.snake1d(&a, &i));
        }
        let x = self.f32_data()?;
        let av = alpha.f32_data()?;
        let iv = inv_alpha.f32_data()?;
        let mut out = vec![0f32; x.len()];
        let one = |row: usize, d: &mut [f32], s: &[f32]| {
            let ch = row % c;
            let (a, inv) = (av[ch], iv[ch]);
            for (dd, &sv) in d.iter_mut().zip(s) {
                let sn = (a * sv).sin();
                *dd = sv + inv * sn * sn;
            }
        };
        if x.len() >= PAR_CPU_MIN {
            use rayon::prelude::*;
            out.par_chunks_mut(l)
                .zip(x.par_chunks(l))
                .enumerate()
                .for_each(|(row, (d, s))| one(row, d, s));
        } else {
            for (row, (d, s)) in out.chunks_mut(l).zip(x.chunks(l)).enumerate() {
                one(row, d, s);
            }
        }
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(CpuStorage::F32(out))),
            self.shape.clone(),
        ))
    }
}
