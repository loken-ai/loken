//! Part of `impl Tensor`, split out of the parent module.
//!
//! Rust lets one inherent impl live in several modules of a crate, so this is the
//! same impl - only the file changed. Methods that were private are `pub(super)`
//! here, which is the reach they had when they sat beside their callers.

use super::*;

impl Tensor {
    /// `x^e` elementwise.
    pub fn powf(&self, e: f32) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() {
            return self.f16_unary_via_f32(|t| t.powf(e));
        }
        #[cfg(feature = "cuda")]
        if let Storage::Cuda { data, dev } = self.storage().as_ref() {
            let out =
                crate::tensor::cuda::pow_f32(dev, data.as_f32_slice()?, e, self.elem_count())?;
            return Self::from_cuda_storage(
                crate::tensor::cuda::CudaStorage::F32(out),
                dev.clone(),
                self.shape.clone(),
            );
        }
        self.unary_f32(|x| x.powf(e))
    }

    pub fn broadcast_div(&self, rhs: &Self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.is_cuda_f16() {
            let r = if matches!(rhs.dtype(), DType::F16 | DType::BF16) {
                rhs.to_dtype(DType::F32)?
            } else {
                rhs.clone()
            };
            return self.f16_unary_via_f32(|t| t.broadcast_div(&r));
        }
        self.broadcast_dispatch(rhs, "broadcast_div", "native_bdiv_tail_f32", 2, |x, y| {
            x / y
        })
    }

    /// Materialize `self` broadcast to `shape` (the packed substrate copies  -
    /// no stride views).
    pub fn broadcast_as<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let target = shape.into();
        let sd = self.dims();
        let td = target.dims().to_vec();
        if td.len() < sd.len() {
            return Err(Error(format!("broadcast_as: {sd:?} -> {td:?} lowers rank")));
        }
        if td == sd {
            return Ok(self.clone());
        }
        // right-aligned axes: each must match the target or be 1
        let pad = td.len() - sd.len();
        let sstride = self.shape.stride_contiguous();
        let mut eff = vec![0usize; td.len()];
        for (i, &t) in td.iter().enumerate() {
            if i < pad {
                continue;
            }
            let s = sd[i - pad];
            if s == t {
                eff[i] = sstride[i - pad];
            } else if s != 1 {
                return Err(Error(format!(
                    "broadcast_as: cannot broadcast {sd:?} to {td:?} (axis {i}: {s} vs {t})"
                )));
            }
        }
        // Element count unchanged ⟹ every broadcast axis is size 1 - the
        // packed data is identical, so this is a zero-cost reshape (the
        // `broadcast_left(1)` / leading-unsqueeze pattern; the gather copy
        // below duplicated full weight matrices per call on the CPU path).
        if target.elem_count() == self.elem_count() {
            return self.reshape(target);
        }
        // Past that reshape this is a real copy on every device - an attention mask
        // grown to the head count is one of the larger buffers a prefill holds - and
        // the shape it copies into has already been computed and checked above.
        if self.is_dry() {
            let dtype = self.dtype();
            if let Some(t) = self.dry_out(target.clone(), dtype) {
                return Ok(t);
            }
        }
        #[cfg(feature = "cuda")]
        if self.is_on_cuda() {
            if self.dtype() == DType::F32 {
                let Storage::Cuda { data, dev } = self.storage().as_ref() else {
                    unreachable!()
                };
                // stride-gather through the permute kernel (0-stride on the
                // broadcast axes)
                let odims_i: Vec<i32> = td.iter().map(|&d| d as i32).collect();
                let eff_i: Vec<i32> = eff.iter().map(|&s| s as i32).collect();
                let out = crate::tensor::cuda::permute_f32(
                    dev,
                    data.as_f32_slice()?,
                    &odims_i,
                    &eff_i,
                    target.elem_count(),
                )?;
                return Self::from_cuda_storage(
                    crate::tensor::cuda::CudaStorage::F32(out),
                    dev.clone(),
                    target,
                );
            }
            // any other dtype: width-generic device gather
            let Storage::Cuda { data, dev } = self.storage().as_ref() else {
                unreachable!()
            };
            let odims_i: Vec<i32> = td.iter().map(|&d| d as i32).collect();
            let eff_i: Vec<i32> = eff.iter().map(|&s| s as i32).collect();
            let out = crate::tensor::cuda::permute_storage(
                dev,
                data,
                &odims_i,
                &eff_i,
                target.elem_count(),
            )?;
            return Self::from_cuda_storage(out, dev.clone(), target);
        }
        let c = self.cpu_storage_ref()?;
        let ostride = target.stride_contiguous();
        let n = target.elem_count();
        let storage = map_cpu!(c, |v, wrap| {
            let mut out = Vec::with_capacity(n);
            for oi in 0..n {
                let mut rem = oi;
                let mut ii = 0usize;
                for (ax, &os) in ostride.iter().enumerate() {
                    let idx = rem / os;
                    rem %= os;
                    ii += idx * eff[ax];
                }
                out.push(v[ii]);
            }
            wrap(out)
        });
        Ok(Self::from_packed(Arc::new(Storage::Cpu(storage)), target))
    }

    /// Gather along `dim`: output takes `indexes`' shape (same rank as
    /// `self`, other dims no larger). Indexes are u32 or i64.
    pub fn gather<I: Dim>(&self, indexes: &Self, dim: I) -> Result<Self> {
        let d = dim.to_index(&self.shape, "gather")?;
        let sd = self.dims().to_vec();
        let id = indexes.dims().to_vec();
        if id.len() != sd.len()
            || id
                .iter()
                .zip(&sd)
                .enumerate()
                .any(|(i, (a, b))| i != d && a > b)
        {
            return Err(Error(format!(
                "gather: indexes {id:?} incompatible with {sd:?}"
            )));
        }
        // One buffer of the INDEXES' shape - which rows are picked is a question about
        // values, how many is not.
        if self.is_dry() {
            let dtype = self.dtype();
            if let Some(t) = self.dry_out(id.clone(), dtype) {
                return Ok(t);
            }
        }
        #[cfg(feature = "cuda")]
        {
            // device path: non-dim dims equal (the decode-path form), idx on
            // the same device
            let dims_eq = id
                .iter()
                .zip(&sd)
                .enumerate()
                .all(|(i, (a, b))| i == d || a == b);
            if let (
                Storage::Cuda { data, dev },
                Storage::Cuda {
                    data: ix,
                    dev: idev,
                },
            ) = (self.storage().as_ref(), indexes.storage().as_ref())
            {
                if dims_eq
                    && dev.ordinal() == idev.ordinal()
                    && matches!(ix.dtype(), DType::U32 | DType::I64)
                {
                    let inner: usize = sd[d + 1..].iter().product();
                    let out = crate::tensor::cuda::gather_storage(
                        dev,
                        data,
                        ix,
                        indexes.elem_count(),
                        inner.max(1),
                        id[d],
                        sd[d],
                    )?;
                    return Self::from_cuda_storage(out, dev.clone(), id);
                }
            }
            if self.is_on_cuda() || indexes.is_on_cuda() {
                let ids_cpu = indexes.to_device(&Device::Cpu)?;
                return self.host_bounce(move |cpu| cpu.gather(&ids_cpu, d));
            }
        }
        let idx = Self::index_values(indexes)?;
        let c = self.cpu_storage_ref()?;
        let istride = Shape::from(id.clone()).stride_contiguous();
        let sstride = self.shape.stride_contiguous();
        let storage = map_cpu!(c, |v, wrap| {
            let mut out = Vec::with_capacity(idx.len());
            for (oi, &t) in idx.iter().enumerate() {
                if t >= sd[d] {
                    return Err(Error(format!("gather: index {t} out of range {}", sd[d])));
                }
                let mut rem = oi;
                let mut src = t * sstride[d];
                for (ax, &is) in istride.iter().enumerate() {
                    let coord = rem / is;
                    rem %= is;
                    if ax != d {
                        src += coord * sstride[ax];
                    }
                }
                out.push(v[src]);
            }
            wrap(out)
        });
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(storage)),
            Shape::from(id),
        ))
    }

    /// `self` is the condition (u8/u32/i64; non-zero = true): element-wise
    /// select between `on_true` and `on_false` (all three the same shape).
    pub fn where_cond(&self, on_true: &Self, on_false: &Self) -> Result<Self> {
        if self.dims() != on_true.dims() || self.dims() != on_false.dims() {
            return Err(Error(format!(
                "where_cond: shape mismatch {:?} / {:?} / {:?}",
                self.dims(),
                on_true.dims(),
                on_false.dims()
            )));
        }
        // One buffer of the operands' shape, whichever way each element goes: the
        // selection is a question about values and the cost is not. Charged in the
        // dtype the chosen branches carry, which is `on_true`'s - the shapes were
        // checked above, and the two arms agree on dtype or the real op would fail.
        if self.is_dry() || on_true.is_dry() || on_false.is_dry() {
            let dtype = on_true.dtype();
            let dims = self.dims().to_vec();
            for t in [self, on_true, on_false] {
                if let Some(out) = t.dry_out(dims.clone(), dtype) {
                    return Ok(out);
                }
            }
        }
        #[cfg(feature = "cuda")]
        {
            if on_true.is_cuda_f16() && on_false.is_cuda_f16() && self.is_on_cuda() {
                let back = on_true.dtype();
                let t = on_true.to_dtype(DType::F32)?;
                let f = on_false.to_dtype(DType::F32)?;
                return self.where_cond(&t, &f)?.to_dtype(back);
            }
            if let (
                Storage::Cuda { data: cd, dev },
                Storage::Cuda {
                    data: td,
                    dev: tdev,
                },
                Storage::Cuda {
                    data: fd,
                    dev: fdev,
                },
            ) = (
                self.storage().as_ref(),
                on_true.storage().as_ref(),
                on_false.storage().as_ref(),
            ) {
                if dev.ordinal() == tdev.ordinal()
                    && dev.ordinal() == fdev.ordinal()
                    && matches!(cd.dtype(), DType::U8 | DType::U32)
                    && td.dtype() == DType::F32
                    && fd.dtype() == DType::F32
                {
                    let out = crate::tensor::cuda::where_f32(
                        dev,
                        cd,
                        td.as_f32_slice()?,
                        fd.as_f32_slice()?,
                        self.elem_count(),
                    )?;
                    return Self::from_cuda_storage(
                        crate::tensor::cuda::CudaStorage::F32(out),
                        dev.clone(),
                        self.shape.clone(),
                    );
                }
            }
            if self.is_on_cuda() || on_true.is_on_cuda() || on_false.is_on_cuda() {
                let dev = on_true.device();
                let c = self.to_device(&Device::Cpu)?;
                let t = on_true.to_device(&Device::Cpu)?;
                let f = on_false.to_device(&Device::Cpu)?;
                return c.where_cond(&t, &f)?.to_device(&dev);
            }
        }
        let cond: Vec<bool> = match self.cpu_storage_ref()? {
            CpuStorage::U8(v) => v.iter().map(|&x| x != 0).collect(),
            CpuStorage::U32(v) => v.iter().map(|&x| x != 0).collect(),
            CpuStorage::I64(v) => v.iter().map(|&x| x != 0).collect(),
            other => {
                return Err(Error(format!(
                    "where_cond: condition must be integer, got {}",
                    other.dtype()
                )))
            }
        };
        let storage = map_cpu2!(
            "where_cond",
            on_true.cpu_storage_ref()?,
            on_false.cpu_storage_ref()?,
            |tv, fv, wrap| {
                let out = cond
                    .iter()
                    .enumerate()
                    .map(|(i, &c)| if c { tv[i] } else { fv[i] })
                    .collect();
                wrap(out)
            }
        );
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(storage)),
            self.shape.clone(),
        ))
    }

    /// Sort each row of the last dim; returns `(values, u32 indices)`.
    /// Stable, NaN-greater comparator (matches the previous substrate).
    pub fn sort_last_dim(&self, asc: bool) -> Result<(Self, Self)> {
        #[cfg(feature = "cuda")]
        if self.is_on_cuda() {
            // host fallback - the in-server MoE routing already falls back to
            // a host sort for large pair counts; small sorts go the same way.
            let dev = self.device();
            let (v, i) = self.to_device(&Device::Cpu)?.sort_last_dim(asc)?;
            return Ok((v.to_device(&dev)?, i.to_device(&dev)?));
        }
        let last = *self
            .dims()
            .last()
            .ok_or_else(|| Error("sort on rank-0".into()))?;
        let rows = self.elem_count() / last.max(1);
        let c = self.cpu_storage_ref()?;
        let (values, indices) = map_cpu!(c, |v, wrap| {
            let mut vals = Vec::with_capacity(v.len());
            let mut idxs = Vec::with_capacity(v.len());
            for r in 0..rows {
                let row = &v[r * last..][..last];
                let mut ix: Vec<u32> = (0..last as u32).collect();
                if asc {
                    ix.sort_by(|&i, &j| {
                        row[i as usize]
                            .partial_cmp(&row[j as usize])
                            .unwrap_or(std::cmp::Ordering::Greater)
                    });
                } else {
                    ix.sort_by(|&j, &i| {
                        row[i as usize]
                            .partial_cmp(&row[j as usize])
                            .unwrap_or(std::cmp::Ordering::Greater)
                    });
                }
                vals.extend(ix.iter().map(|&i| row[i as usize]));
                idxs.extend_from_slice(&ix);
            }
            (wrap(vals), CpuStorage::U32(idxs))
        });
        Ok((
            Self::from_packed(Arc::new(Storage::Cpu(values)), self.shape.clone()),
            Self::from_packed(Arc::new(Storage::Cpu(indices)), self.shape.clone()),
        ))
    }

    /// Sort indices only (u32), same comparator as `sort_last_dim`.
    pub fn arg_sort_last_dim(&self, asc: bool) -> Result<Self> {
        Ok(self.sort_last_dim(asc)?.1)
    }

    /// Index of the max along `dim` (first occurrence), u32, dim removed.
    pub fn argmax<I: Dim>(&self, dim: I) -> Result<Self> {
        self.argmax_impl(dim, false)
    }

    pub fn argmax_keepdim<I: Dim>(&self, dim: I) -> Result<Self> {
        self.argmax_impl(dim, true)
    }

    pub(super) fn argmax_impl<I: Dim>(&self, dim: I, keepdim: bool) -> Result<Self> {
        let d = dim.to_index(&self.shape, "argmax")?;
        #[cfg(feature = "cuda")]
        if self.is_on_cuda() {
            return self.host_bounce(move |cpu| cpu.argmax_impl(d, keepdim));
        }
        let dims = self.dims();
        let outer: usize = dims[..d].iter().product();
        let red = dims[d];
        let inner: usize = dims[d + 1..].iter().product();
        if red == 0 {
            return Err(Error("argmax over an empty dim".into()));
        }
        let c = self.cpu_storage_ref()?;
        let out: Vec<u32> = map_cpu!(c, |v, _wrap| {
            let mut out = vec![0u32; outer * inner];
            for o in 0..outer {
                for i in 0..inner {
                    let mut best = 0usize;
                    for r in 1..red {
                        let x = &v[(o * red + r) * inner + i];
                        let b = &v[(o * red + best) * inner + i];
                        if matches!(x.partial_cmp(b), Some(std::cmp::Ordering::Greater)) {
                            best = r;
                        }
                    }
                    out[o * inner + i] = best as u32;
                }
            }
            out
        });
        let mut odims = dims.to_vec();
        if keepdim {
            odims[d] = 1;
        } else {
            odims.remove(d);
        }
        Ok(Self::from_packed(
            Arc::new(Storage::Cpu(CpuStorage::U32(out))),
            Shape::from(odims),
        ))
    }

    /// Copy `src` into `self` at `offset` along `dim`, IN PLACE: the write is
    /// visible through every tensor sharing this storage (the O(1) KV-cache
    /// append / CUDA-graph buffer update - same `&self` mutation semantics as
    /// the previous substrate). `self` must not share storage with `src`, and
    /// the caller must not read the written region concurrently from another
    /// thread (the decode path mutates caches from one model thread).
    pub fn slice_set<I: Dim>(&self, src: &Self, dim: I, offset: usize) -> Result<()> {
        let d = dim.to_index(&self.shape, "slice_set")?;
        // In-place writes into a narrow VIEW (offset or prefix) are rejected: a view shares
        // its parent's storage, so the write would reach past what the caller named. No
        // correct caller asks for it.
        if !self.spans_whole() {
            return Err(Error("slice_set: destination is a narrow view".into()));
        }
        // `src.storage()` below materializes a view src, so overlap is only
        // possible when both are the same whole allocation.
        if Arc::ptr_eq(&self.storage_raw, src.storage()) {
            return Err(Error("slice_set: self and src share storage".into()));
        }
        if self.dtype() != src.dtype() {
            return Err(Error(format!(
                "slice_set: dtype mismatch {} vs {}",
                self.dtype(),
                src.dtype()
            )));
        }
        let dims = self.dims().to_vec();
        let sdims = src.dims().to_vec();
        if dims.len() != sdims.len() {
            return Err(Error(format!(
                "slice_set: rank mismatch {dims:?} vs {sdims:?}"
            )));
        }
        for (i, (&dv, &sv)) in dims.iter().zip(&sdims).enumerate() {
            if i == d && sv + offset > dv {
                return Err(Error(format!(
                    "slice_set: src {sv} + offset {offset} > dst {dv} on dim {d}"
                )));
            }
            if i != d && dv != sv {
                return Err(Error(format!(
                    "slice_set: shape mismatch on dim {i}, {dv} vs {sv}"
                )));
            }
        }
        if src.elem_count() == 0 {
            return Ok(());
        }
        // Writing into a buffer that is already counted. Nothing is allocated and
        // nothing is moved, so the ledger has nothing to record - and every check
        // above still ran, which is what keeps a counted forward from passing a
        // write the real one would reject. Refused across ledgers for the reason the
        // CUDA arm refuses across cards: a write whose two sides live in different
        // places is not a write the hardware would perform.
        if self.is_dry() || src.is_dry() {
            if !self.device().same_device(&src.device()) {
                return Err(Error("slice_set: operands on different devices".into()));
            }
            return Ok(());
        }
        let outer: usize = dims[..d].iter().product();
        let inner: usize = dims[d + 1..].iter().product();
        let (dst_d, src_d) = (dims[d], sdims[d]);
        match (self.storage().as_ref(), src.storage().as_ref()) {
            #[cfg(feature = "cuda")]
            (
                Storage::Cuda { data: dd, dev },
                Storage::Cuda {
                    data: sd,
                    dev: sdev,
                },
            ) => {
                if dev.ordinal() != sdev.ordinal() {
                    return Err(Error(
                        "slice_set: operands on different cuda devices".into(),
                    ));
                }
                let esize = self.dtype().size_in_bytes();
                crate::tensor::cuda::slice_set_storage(
                    dev,
                    dd,
                    sd,
                    outer.max(1),
                    dst_d * inner * esize,
                    src_d * inner * esize,
                    offset * inner * esize,
                )
            }
            (Storage::Cpu(dc), Storage::Cpu(sc)) => {
                map_cpu2!("slice_set", dc, sc, |dv, sv, _wrap| {
                    let n = src_d * inner;
                    for o in 0..outer {
                        let db = (o * dst_d + offset) * inner;
                        let sb = o * n;
                        // SAFETY: in-place write through the shared storage  -
                        // bounds checked above, src/dst storages distinct, no
                        // live typed borrow of the destination region; the
                        // single-writer requirement is documented on the fn.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                sv.as_ptr().add(sb),
                                dv.as_ptr().cast_mut().add(db),
                                n,
                            );
                        }
                    }
                    Ok(())
                })
            }
            _ => Err(Error("slice_set: operands on different devices".into())),
        }
    }

    /// `self[.., indexes[e], ..] = source[e]` along `dim`, IN PLACE (the
    /// CUDA-graph KV write). `indexes` (u32/i64) has `source`'s shape; all
    /// dims but `dim` must match `self`. Same in-place semantics and
    /// single-writer requirement as `slice_set`.
    pub fn scatter_set<I: Dim>(&self, indexes: &Self, source: &Self, dim: I) -> Result<()> {
        let d = dim.to_index(&self.shape, "scatter_set")?;
        // Same view-destination rejection as `slice_set`.
        if !self.spans_whole() {
            return Err(Error("scatter_set: destination is a narrow view".into()));
        }
        if Arc::ptr_eq(&self.storage_raw, source.storage()) {
            return Err(Error("scatter_set: self and source share storage".into()));
        }
        if self.dtype() != source.dtype() {
            return Err(Error(format!(
                "scatter_set: dtype mismatch {} vs {}",
                self.dtype(),
                source.dtype()
            )));
        }
        let dims = self.dims().to_vec();
        let sdims = source.dims().to_vec();
        if indexes.dims() != sdims.as_slice() {
            return Err(Error(format!(
                "scatter_set: indexes {:?} != source {sdims:?}",
                indexes.dims()
            )));
        }
        if dims.len() != sdims.len()
            || dims
                .iter()
                .zip(&sdims)
                .enumerate()
                .any(|(i, (a, b))| i != d && a != b)
        {
            return Err(Error(format!(
                "scatter_set: shape mismatch {dims:?} vs {sdims:?}"
            )));
        }
        if source.elem_count() == 0 {
            return Ok(());
        }
        // In place, into a counted buffer - see `slice_set` for why this allocates
        // nothing and why it still refuses two ledgers.
        if self.is_dry() || source.is_dry() {
            if !self.device().same_device(&source.device()) {
                return Err(Error("scatter_set: operands on different devices".into()));
            }
            return Ok(());
        }
        let inner: usize = sdims[d + 1..].iter().product();
        let (dst_d, src_d) = (dims[d], sdims[d]);
        match (
            self.storage().as_ref(),
            source.storage().as_ref(),
            indexes.storage().as_ref(),
        ) {
            #[cfg(feature = "cuda")]
            (
                Storage::Cuda { data: dd, dev },
                Storage::Cuda {
                    data: sd,
                    dev: sdev,
                },
                Storage::Cuda {
                    data: ix,
                    dev: idev,
                },
            ) => {
                if dev.ordinal() != sdev.ordinal() || dev.ordinal() != idev.ordinal() {
                    return Err(Error(
                        "scatter_set: operands on different cuda devices".into(),
                    ));
                }
                crate::tensor::cuda::scatter_set_storage(
                    dev,
                    dd,
                    sd,
                    ix,
                    inner.max(1),
                    src_d,
                    dst_d,
                )
            }
            (Storage::Cpu(dc), Storage::Cpu(sc), Storage::Cpu(_)) => {
                let idx = Self::index_values(indexes)?;
                map_cpu2!("scatter_set", dc, sc, |dv, sv, _wrap| {
                    let dp = dv.as_ptr().cast_mut();
                    let inner = inner.max(1);
                    for (e, &t) in idx.iter().enumerate() {
                        if t >= dst_d {
                            return Err(Error(format!(
                                "scatter_set: index {t} out of range {dst_d}"
                            )));
                        }
                        let ii = e % inner;
                        let o = e / (inner * src_d);
                        // SAFETY: same in-place contract as `slice_set`; the
                        // destination element index is bounds-checked above.
                        unsafe {
                            *dp.add((o * dst_d + t) * inner + ii) = sv[e];
                        }
                    }
                    Ok(())
                })
            }
            _ => Err(Error("scatter_set: operands on different devices".into())),
        }
    }

    // -- the shape and arithmetic surface the inference layer calls -----------

    // -- dims accessors (delegate to Shape) --
    pub fn dims1(&self) -> Result<usize> {
        self.shape.dims1()
    }
    pub fn dims2(&self) -> Result<(usize, usize)> {
        self.shape.dims2()
    }
    pub fn dims3(&self) -> Result<(usize, usize, usize)> {
        self.shape.dims3()
    }
    pub fn dims4(&self) -> Result<(usize, usize, usize, usize)> {
        self.shape.dims4()
    }
    pub fn dims5(&self) -> Result<(usize, usize, usize, usize, usize)> {
        self.shape.dims5()
    }

    // -- contiguity (native tensors are contiguous by construction; a narrow
    //    view is a contiguous RANGE of its raw storage, so this stays true) --
    pub fn is_contiguous(&self) -> bool {
        true
    }
    pub fn force_contiguous(&self) -> Result<Self> {
        self.contiguous()
    }
    pub fn detach(&self) -> Self {
        self.clone()
    }
}
