//! The expert projections of a MoE prefill through the expert-batched tiled matmul.
//!
//! Grouped by expert, every expert's routed rows are one contiguous run, and every expert's
//! weights are one slab of the stack. One launch per projection multiplies each expert against
//! its run on the int8 tensor cores (`launch_mmq_moe_gguf_*`), reading the activation
//! quantised once into the q8_1 tiles that kernel takes. The gate and up rows stay in sorted order; the
//! down projection writes each row back to its (token, choice) slot, and the fold sums a
//! token's slots in slot order, so the result does not depend on scheduling.

use super::*;
use crate::tensor::cuda_ext::tensor_from_f32_slice;
use crate::tensor::quantized::{
    mmq_moe_launcher, mmq_moe_qk, mmq_moe_quantize_launcher, mmq_moe_supports, GgmlDType, QTensor,
};
use cudarc::driver::CudaSlice;

cuda_launchers! {
    loken_moe_mmq_plan(
        experts: *const i32, slots: *const i32, n: i32, topk: i32, n_experts: i32,
        bounds: *mut i32, slots_out: *mut i32, src_rows: *mut i32);
    loken_moe_mmq_route(
        choices: *const i32, n: i32, topk: i32, n_experts: i32, counts: *mut i32,
        bounds: *mut i32, slots: *mut i32, src_rows: *mut i32);
    loken_moe_mmq_silu_mul(gu: *const f32, out: *mut f32, rows: i64, width: i32);
    loken_moe_mmq_swiglu_oai(
        gu: *const f32, out: *mut f32, rows: i64, width: i32, bounds: *const i32,
        n_experts: i32, gate_bias: *const f32, up_bias: *const f32, alpha: f32, limit: f32);
    loken_moe_mmq_fold(
        pairs: *const f32, weights: *const f32, choices: *const i32, down_bias: *const f32,
        out: *mut f32, tokens: i64, topk: i32, cols: i32);
}

/// Values in one q8_1 tile block of the activation, and the bytes it takes.
const Q8_1_MMQ_VALUES: usize = 128;
const Q8_1_MMQ_BYTES: usize = Q8_1_MMQ_VALUES + 16;
/// The widest tile the kernels are built for, in activation columns: the quantised activation
/// is padded by this many so the last tile can read past the last row.
const MMQ_X_MAX: usize = 128;
/// The narrowest tile, which sets how many column tiles the grid can need.
const MMQ_X_MIN: usize = 8;
/// The row padding the quantiser lays a row out to.
const MMQ_ROW_PADDING: usize = 512;
/// Grid limit on the column-tile axis.
const GRID_Y_MAX: usize = 65535;

/// Routed rows per expert from which the expert-batched matmul is at least as fast as the
/// decode kernels: measured by `expert_batched_matmul_timings` on a 128-expert, top-8 layer,
/// where the two break even at one row per expert and the tile kernels pull ahead above it.
const MIN_ROWS_PER_EXPERT: usize = 1;

/// Whether `rows` routed rows over `experts` experts are enough for the expert-batched matmul
/// to beat the decode kernels. Never while a draft verify holds [`DecodeOrder`]: its rows
/// must sum exactly as the decode steps they stand for.
///
/// [`DecodeOrder`]: crate::tensor::quantized::DecodeOrder
pub fn moe_mmq_pays(rows: usize, experts: usize) -> bool {
    rows >= MIN_ROWS_PER_EXPERT * experts.max(1) && !crate::tensor::quantized::DecodeOrder::active()
}

/// Whether the expert-batched matmul serves a stack of `dtype` reducing over `k`.
///
/// A pass of the tile kernel covers more of k than one block, so a row whose length is not a
/// whole number of passes is read past its end: into the next row, against activation values
/// the quantiser padded with zeros, and past the last row into the zeroed tail every device
/// blob carries (`BLOB_TAIL_PAD_BYTES`).
pub fn moe_mmq_serves(dtype: GgmlDType, k: usize) -> bool {
    mmq_moe_supports(dtype) && k.is_multiple_of(mmq_moe_qk(dtype))
}

/// The routing, in the form the expert-batched launches read it.
pub struct MmqRouting {
    /// `[experts + 1]`: where each expert's run of sorted rows starts.
    bounds: CudaSlice<i32>,
    /// `[rows]`: the flattened (token, choice) pair each sorted row is built from.
    slots: CudaSlice<i32>,
    /// `[rows]`: the token each sorted row is built from.
    src_rows: CudaSlice<i32>,
    device: crate::tensor::Device,
    /// `[tokens, topk]` u32: the router's choices, when the routing was built from them.
    choices: Option<Tensor>,
    tokens: usize,
    topk: usize,
    experts: usize,
    /// A forced tile width, 0 for the launcher's own choice.
    mmq_x: i32,
}

impl MmqRouting {
    fn alloc(
        device: &crate::tensor::Device,
        rows: usize,
        topk: usize,
        n_experts: usize,
    ) -> Result<Self> {
        if rows % topk != 0 {
            crate::tensor::bail!("moe mmq: {rows} routed rows is not a multiple of topk={topk}");
        }
        let dev = device.as_cuda_device()?;
        Ok(Self {
            bounds: dev.alloc_zeros::<i32>(n_experts + 1)?,
            slots: dev.alloc_zeros::<i32>(rows)?,
            src_rows: dev.alloc_zeros::<i32>(rows)?,
            device: device.clone(),
            choices: None,
            tokens: rows / topk,
            topk,
            experts: n_experts,
            mmq_x: 0,
        })
    }

    /// Build from the stably sorted routing: `experts[i]` the expert of sorted row `i`,
    /// `slots[i]` the flattened (token, choice) pair it came from.
    pub fn new(slots: &Tensor, experts: &Tensor, topk: usize, n_experts: usize) -> Result<Self> {
        let rows = slots.dim(0)?;
        let mut plan = Self::alloc(&slots.device(), rows, topk, n_experts)?;
        let dev = slots.device().as_cuda_device()?;
        unsafe {
            loken_moe_mmq_plan(
                dev_ptr::<u32>(experts, "experts")? as *const i32,
                dev_ptr::<u32>(slots, "slots")? as *const i32,
                rows as i32,
                topk as i32,
                n_experts as i32,
                slice_ptr(&mut plan.bounds),
                slice_ptr(&mut plan.slots),
                slice_ptr(&mut plan.src_rows),
                dev.cuda_stream().cu_stream() as i64,
            );
        }
        Ok(plan)
    }

    /// Build straight from the router's choices `[tokens, topk]`, grouping the rows by expert
    /// on the device. Within an expert the rows come in no particular order, which changes
    /// nothing downstream: a row's result does not depend on which tile column holds it, and
    /// the fold reads the rows back by slot.
    pub fn from_choices(choices: &Tensor, n_experts: usize) -> Result<Self> {
        let (tokens, topk) = choices.dims2()?;
        let choices = choices.to_dtype(DType::U32)?.contiguous()?;
        let rows = tokens * topk;
        let mut plan = Self::alloc(&choices.device(), rows, topk, n_experts)?;
        let dev = choices.device().as_cuda_device()?;
        let mut counts = dev.alloc_zeros::<i32>(n_experts)?;
        unsafe {
            loken_moe_mmq_route(
                dev_ptr::<u32>(&choices, "choices")? as *const i32,
                rows as i32,
                topk as i32,
                n_experts as i32,
                slice_ptr(&mut counts),
                slice_ptr(&mut plan.bounds),
                slice_ptr(&mut plan.slots),
                slice_ptr(&mut plan.src_rows),
                dev.cuda_stream().cu_stream() as i64,
            );
        }
        drop(counts);
        plan.choices = Some(choices);
        Ok(plan)
    }

    /// Force the tile width (a measurement knob; 0 restores the launcher's choice).
    pub fn with_tile_width(mut self, mmq_x: i32) -> Self {
        self.mmq_x = mmq_x;
        self
    }

    fn rows(&self) -> usize {
        self.tokens * self.topk
    }

    /// `silu(gate . x) * (up . x)` for every routed row, in sorted order: `[rows, n]`.
    ///
    /// `up` is `None` when `gate` is a fused gate‖up stack of `2n` rows per expert.
    pub fn gate_up_silu(
        &self,
        xs: &Tensor,
        gate: &QTensor,
        up: Option<&QTensor>,
    ) -> Result<Tensor> {
        self.gate_up(xs, gate, up, Glu::Silu)
    }

    /// The two projections for every routed row, in sorted order, combined by `glu`:
    /// `[rows, n]`.
    pub fn gate_up(
        &self,
        xs: &Tensor,
        gate: &QTensor,
        up: Option<&QTensor>,
        glu: Glu<'_>,
    ) -> Result<Tensor> {
        let (tokens, k) = xs.dims2()?;
        if tokens != self.tokens {
            crate::tensor::bail!("moe mmq: {tokens} tokens against a plan of {}", self.tokens);
        }
        let (e, n_gate, gk) = stack_dims(gate)?;
        let width = match up {
            Some(u) => {
                if u.shape().dims() != gate.shape().dims() || u.dtype() != gate.dtype() {
                    crate::tensor::bail!("moe mmq: the gate and up stacks differ");
                }
                n_gate
            }
            None => n_gate / 2,
        };
        if gk != k || e != self.experts {
            crate::tensor::bail!("moe mmq: stack [{e}, {n_gate}, {gk}] against k={k}");
        }
        let dev = xs.device().as_cuda_device()?;
        let rows = self.rows();
        let y = self.quantize(xs, Some(&self.src_rows), gate.dtype(), k)?;
        let gu = unsafe { dev.native().stream().alloc::<f32>(rows * 2 * width) }
            .map_err(|e| crate::tensor::Error::msg(format!("moe mmq: {e}")))?;
        let gu_ptr = gu.device_ptr(gu.stream()).0;
        match up {
            Some(u) => {
                self.matmul(gate, &y, k, gu_ptr, 2 * width, None)?;
                let up_ptr = gu_ptr + (width * std::mem::size_of::<f32>()) as u64;
                self.matmul(u, &y, k, up_ptr, 2 * width, None)?;
            }
            None => self.matmul(gate, &y, k, gu_ptr, 2 * width, None)?,
        }
        let out = unsafe { dev.native().stream().alloc::<f32>(rows * width) }
            .map_err(|e| crate::tensor::Error::msg(format!("moe mmq: {e}")))?;
        let out_ptr = out.device_ptr(out.stream()).0 as *mut f32;
        let stream = dev.cuda_stream().cu_stream() as i64;
        match glu {
            Glu::Silu => unsafe {
                loken_moe_mmq_silu_mul(
                    gu_ptr as *const f32,
                    out_ptr,
                    rows as i64,
                    width as i32,
                    stream,
                );
            },
            Glu::SwigluOai {
                gate_bias,
                up_bias,
                alpha,
                limit,
            } => {
                let bias = |b: Option<&Tensor>, what: &str| -> Result<Option<Tensor>> {
                    b.map(|b| {
                        let b = b.to_dtype(DType::F32)?.contiguous()?;
                        if b.elem_count() != self.experts * width {
                            crate::tensor::bail!("moe mmq: {what} is not [experts, {width}]");
                        }
                        Ok(b)
                    })
                    .transpose()
                };
                let gb = bias(gate_bias, "gate bias")?;
                let ub = bias(up_bias, "up bias")?;
                let addr = |b: &Option<Tensor>, what: &str| -> Result<*const f32> {
                    Ok(match b {
                        Some(b) => dev_ptr::<f32>(b, what)? as *const f32,
                        None => std::ptr::null(),
                    })
                };
                unsafe {
                    loken_moe_mmq_swiglu_oai(
                        gu_ptr as *const f32,
                        out_ptr,
                        rows as i64,
                        width as i32,
                        self.bounds.device_ptr(self.bounds.stream()).0 as *const i32,
                        self.experts as i32,
                        addr(&gb, "gate bias")?,
                        addr(&ub, "up bias")?,
                        alpha,
                        limit,
                        stream,
                    );
                }
                drop((gb, ub));
            }
        }
        drop(y);
        drop(gu);
        tensor_from_f32_slice(out, (rows, width), &xs.device())
    }

    /// The down projection of sorted rows `hs` `[rows, k]`, each row scaled by its router
    /// weight `topk_weights` `[tokens * topk]` and folded into its token in slot order, on
    /// top of `residual` when there is one: `[tokens, hidden]`.
    pub fn down_fold(
        &self,
        hs: &Tensor,
        down: &QTensor,
        topk_weights: &Tensor,
        residual: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.down_fold_biased(hs, down, topk_weights, residual, None)
    }

    /// [`Self::down_fold`] with a per-expert down bias `[experts, hidden]` added to each
    /// routed row before its router weight scales it. The bias needs the router's choices, so
    /// the routing must come from [`Self::from_choices`].
    pub fn down_fold_biased(
        &self,
        hs: &Tensor,
        down: &QTensor,
        topk_weights: &Tensor,
        residual: Option<&Tensor>,
        down_bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (rows, k) = hs.dims2()?;
        let (e, hidden, dk) = stack_dims(down)?;
        if rows != self.rows() || dk != k || e != self.experts {
            crate::tensor::bail!(
                "moe mmq: {rows} rows of k={k} against a down stack [{e}, {hidden}, {dk}]"
            );
        }
        let dev = hs.device().as_cuda_device()?;
        let y = self.quantize(hs, None, down.dtype(), k)?;
        let pairs = unsafe { dev.native().stream().alloc::<f32>(rows * hidden) }
            .map_err(|e| crate::tensor::Error::msg(format!("moe mmq: {e}")))?;
        let pairs_ptr = pairs.device_ptr(pairs.stream()).0;
        let slots = self.slots.device_ptr(self.slots.stream()).0 as *const i32;
        self.matmul(down, &y, k, pairs_ptr, hidden, Some(slots))?;
        let out = match residual {
            // A fresh buffer holding the residual: the fold adds into it in place.
            Some(r) => r
                .reshape((self.tokens, hidden))?
                .to_dtype(DType::F32)?
                .affine(1.0, 0.0)?,
            None => Tensor::zeros_on((self.tokens, hidden), DType::F32, &hs.device())?,
        };
        let weights = topk_weights.to_dtype(DType::F32)?.contiguous()?;
        if weights.elem_count() != rows {
            crate::tensor::bail!(
                "moe mmq: {} router weights for {rows} rows",
                weights.elem_count()
            );
        }
        let bias = down_bias
            .map(|b| b.to_dtype(DType::F32)?.contiguous())
            .transpose()?;
        let (choices_ptr, bias_ptr) = match (&bias, &self.choices) {
            (None, _) => (std::ptr::null(), std::ptr::null()),
            (Some(b), Some(c)) => {
                if b.elem_count() != self.experts * hidden {
                    crate::tensor::bail!("moe mmq: the down bias is not [experts, {hidden}]");
                }
                (
                    dev_ptr::<u32>(c, "choices")? as *const i32,
                    dev_ptr::<f32>(b, "down bias")? as *const f32,
                )
            }
            (Some(_), None) => {
                crate::tensor::bail!("moe mmq: a down bias needs the routing built from choices")
            }
        };
        unsafe {
            loken_moe_mmq_fold(
                pairs_ptr as *const f32,
                dev_ptr::<f32>(&weights, "topk_weights")? as *const f32,
                choices_ptr,
                bias_ptr,
                dev_ptr::<f32>(&out, "out")? as *mut f32,
                self.tokens as i64,
                self.topk as i32,
                hidden as i32,
                dev.cuda_stream().cu_stream() as i64,
            );
        }
        drop(y);
        drop(pairs);
        Ok(out)
    }

    /// The rows of `x` `[_, k]` quantised into q8_1 tiles for a `dtype` weight, in sorted
    /// order: row `i` of the result is row `gather[i]` of `x`, or row `i` without a gather.
    fn quantize(
        &self,
        x: &Tensor,
        gather: Option<&CudaSlice<i32>>,
        dtype: GgmlDType,
        k: usize,
    ) -> Result<CudaSlice<u8>> {
        let rows = self.rows();
        let k_padded = k
            .next_multiple_of(MMQ_ROW_PADDING)
            .next_multiple_of(Q8_1_MMQ_VALUES);
        let tile_ints = rows * (k_padded / Q8_1_MMQ_VALUES) * Q8_1_MMQ_BYTES / 4;
        if tile_ints + MMQ_X_MAX * Q8_1_MMQ_BYTES > i32::MAX as usize {
            crate::tensor::bail!("moe mmq: {rows} rows of k={k} overflow the tile indexing");
        }
        let bytes =
            rows * (k_padded / Q8_1_MMQ_VALUES) * Q8_1_MMQ_BYTES + MMQ_X_MAX * Q8_1_MMQ_BYTES;
        let x = x.to_dtype(DType::F32)?.contiguous()?;
        let dev = x.device().as_cuda_device()?;
        let stream = dev.native().stream();
        let y = unsafe { stream.alloc::<u8>(bytes) }
            .map_err(|e| crate::tensor::Error::msg(format!("moe mmq: {e}")))?;
        let ids = match gather {
            Some(g) => g.device_ptr(g.stream()).0 as *const i32,
            None => std::ptr::null(),
        };
        unsafe {
            mmq_moe_quantize_launcher(dtype)?(
                dev_ptr::<f32>(&x, "activation")? as *const c_void,
                ids,
                y.device_ptr(y.stream()).0 as *mut c_void,
                0,
                k as i64,
                k as i64,
                0,
                0,
                k_padded as i64,
                rows as i64,
                1,
                1,
                stream.cu_stream() as *mut c_void,
            );
        }
        Ok(y)
    }

    /// One expert-batched launch of `stack` against the quantised sorted rows `y`, writing
    /// sorted row `i` to destination row `ids_dst[i]` (or `i`) of a `dst` whose rows are
    /// `stride` floats apart.
    fn matmul(
        &self,
        stack: &QTensor,
        y: &CudaSlice<u8>,
        k: usize,
        dst: u64,
        stride: usize,
        ids_dst: Option<*const i32>,
    ) -> Result<()> {
        let (e, n, sk) = stack_dims(stack)?;
        let dtype = stack.dtype();
        if !moe_mmq_serves(dtype, sk) || sk != k {
            crate::tensor::bail!("moe mmq: no expert-batched kernel for {dtype:?} at k={sk}");
        }
        let rows = self.rows();
        let blocks_per_row = k / dtype.block_size();
        let expert_bytes = n * blocks_per_row * dtype.type_size();
        let cuda = self.device.as_cuda_device()?;
        let info = cuda.native().mmq_device_info()?;
        let ncols_hint = rows.div_ceil(self.experts.max(1)).max(1);
        if self.tokens.div_ceil(MMQ_X_MIN) > GRID_Y_MAX {
            crate::tensor::bail!("moe mmq: {} tokens exceed the launch grid", self.tokens);
        }
        let launched = unsafe {
            mmq_moe_launcher(dtype)?(
                stack.device_ptr()? as *const c_void,
                y.device_ptr(y.stream()).0 as *const c_void,
                dst as *mut c_void,
                self.bounds.device_ptr(self.bounds.stream()).0 as *const i32,
                ids_dst.unwrap_or(std::ptr::null()),
                expert_bytes as i64,
                k as i64,
                n as i64,
                rows as i64,
                blocks_per_row as i64,
                stride as i64,
                e as i32,
                self.tokens as i64,
                ncols_hint as i64,
                self.mmq_x,
                info.cc,
                info.smpbo,
                info.warp_size,
                cuda.cuda_stream().cu_stream() as *mut c_void,
            )
        };
        if launched == 0 {
            crate::tensor::bail!("moe mmq: no tile width fits this card's shared memory");
        }
        Ok(())
    }
}

/// How the gate and up rows combine into what the down projection reads.
#[derive(Clone, Copy)]
pub enum Glu<'a> {
    /// `silu(gate) * up`.
    Silu,
    /// gpt-oss: `x = min(gate + gb, limit)`, `x * sigmoid(alpha * x) * (1 + clamp(up + ub))`.
    SwigluOai {
        gate_bias: Option<&'a Tensor>,
        up_bias: Option<&'a Tensor>,
        alpha: f32,
        limit: f32,
    },
}

/// A device buffer's address, for a kernel that writes it.
fn slice_ptr(s: &mut CudaSlice<i32>) -> *mut i32 {
    s.device_ptr(s.stream()).0 as *mut i32
}

/// `[experts, rows, k]` of an expert stack.
fn stack_dims(stack: &QTensor) -> Result<(usize, usize, usize)> {
    match stack.shape().dims() {
        [e, n, k] => Ok((*e, *n, *k)),
        s => crate::tensor::bail!("moe mmq: an expert stack is 3D, got {s:?}"),
    }
}

/// Where a tensor's first element sits on the device. Tensors are packed, so a view is its
/// storage from an element offset on.
fn dev_ptr<T: crate::tensor::kernel_ffi::CudaDType>(t: &Tensor, what: &str) -> Result<u64> {
    let (storage, layout) = t.storage_and_layout();
    match &*storage {
        StorageView::Cuda(c) => {
            let slice = c.as_cuda_slice::<T>()?;
            let (addr, _read) = slice.device_ptr(slice.stream());
            Ok(addr + (layout.start_offset() * std::mem::size_of::<T>()) as u64)
        }
        _ => crate::tensor::bail!("moe mmq: {what} must be on a CUDA device"),
    }
}
