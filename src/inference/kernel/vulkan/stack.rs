//! A run of dense decoder layers on one Vulkan device: their weights, their f16 KV caches, and
//! the command buffers a step replays.
//!
//! A step takes up to [`MAX_TOKENS`] rows of the residual stream. Its command buffer is recorded
//! once for each (rows, layer range) and replayed: the position of the first row travels in a
//! host-mapped step buffer, so a replay needs no recording. A longer prompt runs in chunks.

use super::{f32_bytes, Buffer, Context};
use crate::tensor::quantized::GgmlDType;
use ash::vk;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

/// Rows of the residual stream one step carries.
pub const MAX_TOKENS: usize = 8;
/// Rows of a mat-vec grid's x dimension; the rest go to y.
const GRID_X: u32 = 32768;

/// Rows one mat-vec workgroup runs.
const ROWS_PER_GROUP: u32 = 2;

/// The workgroup grid of a mat-vec over `rows` rows.
fn matvec_groups(rows: u32) -> [u32; 3] {
    let groups = rows.div_ceil(ROWS_PER_GROUP);
    [groups.min(GRID_X), groups.div_ceil(GRID_X), 1]
}

/// The shape of the decoder: what every layer of the stack shares.
#[derive(Debug, Clone)]
pub struct DenseSpec {
    pub hidden: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub eps: f32,
    /// Lanes of each head RoPE rotates; the rest pass through.
    pub rope_dim: usize,
    /// RoPE pairs adjacent lanes; otherwise lanes half the rotated width apart.
    pub rope_adjacent: bool,
    pub attn_scale: f32,
    /// Positions each layer's cache holds.
    pub ctx: usize,
}

/// A weight as the file stores it.
pub struct RawTensor<'a> {
    pub dtype: GgmlDType,
    /// Rows (outputs) then columns (inputs).
    pub rows: usize,
    pub cols: usize,
    pub bytes: Cow<'a, [u8]>,
}

/// One layer's tensors, read from the file.
pub struct LayerTensors<'a> {
    pub attn_norm: Vec<f32>,
    pub q: RawTensor<'a>,
    pub k: RawTensor<'a>,
    pub v: RawTensor<'a>,
    pub q_bias: Option<Vec<f32>>,
    pub k_bias: Option<Vec<f32>>,
    pub v_bias: Option<Vec<f32>>,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub o: RawTensor<'a>,
    pub ffn_norm: Vec<f32>,
    pub gate: RawTensor<'a>,
    pub up: RawTensor<'a>,
    pub down: RawTensor<'a>,
}

/// Whether the stack runs mat-vecs of `dtype`.
pub fn runs_format(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q4K | GgmlDType::Q6K | GgmlDType::Q8_0 | GgmlDType::F16
    )
}

struct Matrix {
    buf: Buffer,
    kernel: &'static str,
    rows: usize,
    cols: usize,
}

impl Matrix {
    fn upload(ctx: &Arc<Context>, t: &RawTensor) -> anyhow::Result<Self> {
        let block = t.dtype.block_size();
        let want = t.rows * t.cols / block * t.dtype.type_size();
        if !t.cols.is_multiple_of(block) || t.bytes.len() != want {
            anyhow::bail!(
                "vulkan: a {} matrix of {}x{} holds {} bytes, expected {want}",
                t.dtype.gguf_name(),
                t.rows,
                t.cols,
                t.bytes.len()
            );
        }
        if !t.cols.is_multiple_of(32) {
            anyhow::bail!("vulkan: {} columns is not a multiple of 32", t.cols);
        }
        let (kernel, bytes): (&'static str, Cow<[u8]>) = match t.dtype {
            GgmlDType::Q4K => ("mv_q4k", Cow::Borrowed(&t.bytes)),
            GgmlDType::F16 => ("mv_f16", Cow::Borrowed(&t.bytes)),
            GgmlDType::Q6K => ("mv_q6k", Cow::Owned(repack(&t.bytes, 210, 208, 212))),
            GgmlDType::Q8_0 => ("mv_q8_0", Cow::Owned(repack_q8_0(&t.bytes))),
            other => anyhow::bail!("vulkan: no mat-vec for {}", other.gguf_name()),
        };
        Ok(Self {
            buf: ctx.upload(&bytes)?,
            kernel,
            rows: t.rows,
            cols: t.cols,
        })
    }
}

/// Blocks of `from` bytes whose last two bytes are the f16 scale, written as blocks of `to`
/// bytes with the scale moved to the aligned offset `at`.
fn repack(bytes: &[u8], from: usize, at: usize, to: usize) -> Vec<u8> {
    let mut out = vec![0u8; bytes.len() / from * to];
    for (src, dst) in bytes.chunks_exact(from).zip(out.chunks_exact_mut(to)) {
        dst[..from - 2].copy_from_slice(&src[..from - 2]);
        dst[at..at + 2].copy_from_slice(&src[from - 2..]);
    }
    out
}

/// Q8_0 blocks (the f16 scale first, then 32 int8) written with the scale in a word of its own.
fn repack_q8_0(bytes: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; bytes.len() / 34 * 36];
    for (src, dst) in bytes.chunks_exact(34).zip(out.chunks_exact_mut(36)) {
        dst[..2].copy_from_slice(&src[..2]);
        dst[4..].copy_from_slice(&src[2..]);
    }
    out
}

struct Layer {
    attn_norm: Buffer,
    q: Matrix,
    k: Matrix,
    v: Matrix,
    q_bias: Option<Buffer>,
    k_bias: Option<Buffer>,
    v_bias: Option<Buffer>,
    q_norm: Option<Buffer>,
    k_norm: Option<Buffer>,
    o: Matrix,
    ffn_norm: Buffer,
    gate: Matrix,
    up: Matrix,
    down: Matrix,
    k_cache: Buffer,
    v_cache: Buffer,
}

/// The final norm and the output projection, when the stack ends the model.
struct Head {
    norm: Buffer,
    proj: Matrix,
    logits: Buffer,
    out: Buffer,
}

/// One dispatch of a step.
struct Op {
    kernel: &'static str,
    bufs: [vk::Buffer; super::BINDINGS as usize],
    args: [u32; 16],
    groups: [u32; 3],
    /// Waits for every earlier write of the step before it runs.
    barrier: bool,
}

/// A recorded step: its command buffer and the descriptor sets it binds.
struct Program {
    pool: vk::CommandPool,
    descriptors: vk::DescriptorPool,
    cmd: vk::CommandBuffer,
}

/// Layers of a dense decoder resident on a Vulkan device.
pub struct VulkanStack {
    ctx: Arc<Context>,
    spec: DenseSpec,
    layers: Vec<Layer>,
    head: Option<Head>,
    cos: Buffer,
    sin: Buffer,
    table_half: usize,
    table_len: usize,
    x: Buffer,
    xn: Buffer,
    q_raw: Buffer,
    k_raw: Buffer,
    v_raw: Buffer,
    q: Buffer,
    attn: Buffer,
    gate: Buffer,
    up: Buffer,
    scores: Buffer,
    io: Buffer,
    step: Buffer,
    dummy: Buffer,
    programs: HashMap<(usize, usize, usize, bool), Program>,
}

impl std::fmt::Debug for VulkanStack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "VulkanStack({} layers on {:?})",
            self.layers.len(),
            self.ctx
        )
    }
}

impl VulkanStack {
    /// Uploads `layers` to Vulkan device `device`. `cos` and `sin` are the RoPE tables, rows of
    /// `table_half` values per position.
    pub fn new(
        device: usize,
        spec: DenseSpec,
        layers: Vec<LayerTensors<'_>>,
        cos: &[f32],
        sin: &[f32],
        table_half: usize,
    ) -> anyhow::Result<Self> {
        let ctx = super::context(device)?;
        let s = &spec;
        if s.head_dim > 512
            || !s.head_dim.is_multiple_of(2)
            || s.n_kv == 0
            || !s.n_head.is_multiple_of(s.n_kv)
        {
            anyhow::bail!(
                "vulkan: heads of {} lanes in {}/{} groups",
                s.head_dim,
                s.n_head,
                s.n_kv
            );
        }
        if s.rope_dim > s.head_dim || s.rope_dim / 2 > table_half {
            anyhow::bail!(
                "vulkan: RoPE over {} lanes with tables of {}",
                s.rope_dim,
                table_half
            );
        }
        let f32buf = |v: &[f32]| ctx.upload(&f32_bytes(v));
        let opt = |v: &Option<Vec<f32>>| v.as_deref().map(f32buf).transpose();
        let cache_bytes = (s.ctx * s.n_kv * s.head_dim * 2) as u64;
        let mut built = Vec::with_capacity(layers.len());
        for l in &layers {
            built.push(Layer {
                attn_norm: f32buf(&l.attn_norm)?,
                q: Matrix::upload(&ctx, &l.q)?,
                k: Matrix::upload(&ctx, &l.k)?,
                v: Matrix::upload(&ctx, &l.v)?,
                q_bias: opt(&l.q_bias)?,
                k_bias: opt(&l.k_bias)?,
                v_bias: opt(&l.v_bias)?,
                q_norm: opt(&l.q_norm)?,
                k_norm: opt(&l.k_norm)?,
                o: Matrix::upload(&ctx, &l.o)?,
                ffn_norm: f32buf(&l.ffn_norm)?,
                gate: Matrix::upload(&ctx, &l.gate)?,
                up: Matrix::upload(&ctx, &l.up)?,
                down: Matrix::upload(&ctx, &l.down)?,
                k_cache: ctx.buffer(cache_bytes, false)?,
                v_cache: ctx.buffer(cache_bytes, false)?,
            });
        }
        let card = |elems: usize| ctx.buffer((elems * 4) as u64, false);
        let t = MAX_TOKENS;
        let qd = s.n_head * s.head_dim;
        let kvd = s.n_kv * s.head_dim;
        Ok(Self {
            layers: built,
            head: None,
            cos: f32buf(cos)?,
            sin: f32buf(sin)?,
            table_half,
            table_len: cos.len() / table_half.max(1),
            x: card(t * s.hidden)?,
            xn: card(t * s.hidden)?,
            q_raw: card(t * qd)?,
            k_raw: card(t * kvd)?,
            v_raw: card(t * kvd)?,
            q: card(t * qd)?,
            attn: card(t * qd)?,
            gate: card(t * s.ffn)?,
            up: card(t * s.ffn)?,
            scores: card(t * s.n_head * s.ctx)?,
            io: ctx.buffer((t * s.hidden * 4) as u64, true)?,
            step: ctx.buffer(16, true)?,
            dummy: ctx.buffer(16, false)?,
            programs: HashMap::new(),
            ctx,
            spec,
        })
    }

    /// Puts the model's final norm and output projection on the device, so a step that ends
    /// the stack hands back logits.
    pub fn attach_head(&mut self, norm: &[f32], proj: &RawTensor) -> anyhow::Result<()> {
        let proj = Matrix::upload(&self.ctx, proj)?;
        if proj.cols != self.spec.hidden {
            anyhow::bail!("vulkan: an output head of {} columns", proj.cols);
        }
        let rows = proj.rows;
        self.head = Some(Head {
            norm: self.ctx.upload(&f32_bytes(norm))?,
            logits: self.ctx.buffer((rows * 4) as u64, false)?,
            out: self.ctx.buffer((rows * 4) as u64, true)?,
            proj,
        });
        self.programs.clear();
        Ok(())
    }

    /// Whether the stack holds the output head.
    pub fn has_head(&self) -> bool {
        self.head.is_some()
    }

    /// Layers the stack holds.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Positions a cache holds.
    pub fn context(&self) -> usize {
        self.spec.ctx
    }

    /// Runs `rows` (`[tokens, hidden]`, the first at position `index_pos`) through the stack's
    /// layers `[from, to)` and returns the rows that leave the last one.
    pub fn forward(
        &mut self,
        rows: &[f32],
        index_pos: usize,
        from: usize,
        to: usize,
    ) -> anyhow::Result<Vec<f32>> {
        Ok(self.run(rows, index_pos, from, to, false)?.0)
    }

    /// [`Self::forward`] through the head as well: the rows and the logits of the last row.
    pub fn forward_to_logits(
        &mut self,
        rows: &[f32],
        index_pos: usize,
        from: usize,
    ) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        if self.head.is_none() {
            anyhow::bail!("vulkan: the stack holds no output head");
        }
        let to = self.layers.len();
        let (rows, logits) = self.run(rows, index_pos, from, to, true)?;
        Ok((rows, logits.unwrap_or_default()))
    }

    fn run(
        &mut self,
        rows: &[f32],
        index_pos: usize,
        from: usize,
        to: usize,
        head: bool,
    ) -> anyhow::Result<(Vec<f32>, Option<Vec<f32>>)> {
        let h = self.spec.hidden;
        let tokens = rows.len() / h;
        if rows.len() != tokens * h || tokens == 0 {
            anyhow::bail!("vulkan: {} values are not rows of {h}", rows.len());
        }
        if to > self.layers.len() || from > to {
            anyhow::bail!(
                "vulkan: layers {from}..{to} of a stack of {}",
                self.layers.len()
            );
        }
        let end = index_pos + tokens;
        if end > self.spec.ctx || end > self.table_len {
            anyhow::bail!(
                "vulkan: position {end} is past the cache of {} positions",
                self.spec.ctx.min(self.table_len)
            );
        }
        let mut out = Vec::with_capacity(rows.len());
        let mut logits = None;
        let mut start = 0;
        while start < tokens {
            let n = (tokens - start).min(MAX_TOKENS);
            let last = start + n == tokens;
            let with_head = head && last;
            self.io
                .write(0, &f32_bytes(&rows[start * h..(start + n) * h]));
            self.step
                .write(0, &((index_pos + start) as u32).to_le_bytes());
            let key = (n, from, to, with_head);
            if !self.programs.contains_key(&key) {
                let program = self.record(n, from, to, with_head)?;
                self.programs.insert(key, program);
            }
            self.ctx.run(self.programs[&key].cmd)?;
            let mut chunk = vec![0u8; n * h * 4];
            self.io.read(0, &mut chunk);
            out.extend(
                chunk
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            );
            if with_head {
                let head = self.head.as_ref().expect("checked");
                let mut bytes = vec![0u8; head.proj.rows * 4];
                head.out.read(0, &mut bytes);
                logits = Some(
                    bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect(),
                );
            }
            start += n;
        }
        Ok((out, logits))
    }

    fn matvec(
        &self,
        w: &Matrix,
        x: &Buffer,
        y: &Buffer,
        bias: Option<&Buffer>,
        accumulate: bool,
        tokens: usize,
        barrier: bool,
    ) -> Op {
        let rows = w.rows as u32;
        let flags = u32::from(accumulate) | (u32::from(bias.is_some()) << 1);
        let mut args = [0u32; 16];
        args[..8].copy_from_slice(&[
            rows,
            w.cols as u32,
            tokens as u32,
            0,
            0,
            flags,
            w.cols as u32,
            rows,
        ]);
        self.op(
            w.kernel,
            &[&w.buf, x, y, bias.unwrap_or(&self.dummy)],
            args,
            matvec_groups(rows),
            barrier,
        )
    }

    /// The products `ws` of `x`: the first waits for `x`, the others run beside it.
    fn products(
        &self,
        ops: &mut Vec<Op>,
        x: &Buffer,
        tokens: usize,
        ws: &[(&Matrix, &Buffer, Option<&Buffer>, bool)],
    ) {
        for (i, (w, y, bias, accumulate)) in ws.iter().enumerate() {
            ops.push(self.matvec(w, x, y, *bias, *accumulate, tokens, i == 0));
        }
    }

    fn op(
        &self,
        kernel: &'static str,
        bufs: &[&Buffer],
        args: [u32; 16],
        groups: [u32; 3],
        barrier: bool,
    ) -> Op {
        let mut all = [self.dummy.buf; super::BINDINGS as usize];
        for (slot, b) in all.iter_mut().zip(bufs) {
            *slot = b.buf;
        }
        Op {
            kernel,
            bufs: all,
            args,
            groups,
            barrier,
        }
    }

    fn norm(&self, x: &Buffer, w: &Buffer, y: &Buffer, rows: usize, x_off: usize) -> Op {
        let mut args = [0u32; 16];
        args[..6].copy_from_slice(&[
            rows as u32,
            self.spec.hidden as u32,
            x_off as u32,
            0,
            self.spec.eps.to_bits(),
            0,
        ]);
        self.op("rms_norm", &[x, w, y], args, [rows as u32, 1, 1], true)
    }

    /// The dispatches of a step of `tokens` rows through layers `[from, to)`.
    fn ops(&self, tokens: usize, from: usize, to: usize, head: bool) -> Vec<Op> {
        let s = &self.spec;
        let t = tokens as u32;
        let mut ops = Vec::new();
        for l in &self.layers[from..to] {
            ops.push(self.norm(&self.x, &l.attn_norm, &self.xn, tokens, 0));
            self.products(
                &mut ops,
                &self.xn,
                tokens,
                &[
                    (&l.q, &self.q_raw, l.q_bias.as_ref(), false),
                    (&l.k, &self.k_raw, l.k_bias.as_ref(), false),
                    (&l.v, &self.v_raw, l.v_bias.as_ref(), false),
                ],
            );
            let prep = |heads: usize, norm: Option<&Buffer>| {
                let mut a = [0u32; 16];
                a[..8].copy_from_slice(&[
                    heads as u32,
                    s.head_dim as u32,
                    u32::from(norm.is_some()),
                    s.eps.to_bits(),
                    s.rope_dim as u32,
                    u32::from(s.rope_adjacent),
                    s.n_kv as u32,
                    self.table_half as u32,
                ]);
                a
            };
            ops.push(self.op(
                "prep_q",
                &[
                    &self.q_raw,
                    l.q_norm.as_ref().unwrap_or(&self.dummy),
                    &self.q,
                    &self.dummy,
                    &self.dummy,
                    &self.step,
                    &self.cos,
                    &self.sin,
                ],
                prep(s.n_head, l.q_norm.as_ref()),
                [s.n_head as u32, t, 1],
                true,
            ));
            ops.push(self.op(
                "prep_kv",
                &[
                    &self.k_raw,
                    l.k_norm.as_ref().unwrap_or(&self.dummy),
                    &l.k_cache,
                    &self.v_raw,
                    &l.v_cache,
                    &self.step,
                    &self.cos,
                    &self.sin,
                ],
                prep(s.n_kv, l.k_norm.as_ref()),
                [s.n_kv as u32, t, 1],
                false,
            ));
            let mut a = [0u32; 16];
            a[..5].copy_from_slice(&[
                s.n_head as u32,
                s.n_kv as u32,
                s.head_dim as u32,
                s.attn_scale.to_bits(),
                s.ctx as u32,
            ]);
            ops.push(self.op(
                "attention",
                &[
                    &self.q,
                    &l.k_cache,
                    &l.v_cache,
                    &self.attn,
                    &self.scores,
                    &self.step,
                ],
                a,
                [s.n_head as u32, t, 1],
                true,
            ));
            self.products(&mut ops, &self.attn, tokens, &[(&l.o, &self.x, None, true)]);
            ops.push(self.norm(&self.x, &l.ffn_norm, &self.xn, tokens, 0));
            self.products(
                &mut ops,
                &self.xn,
                tokens,
                &[
                    (&l.gate, &self.gate, None, false),
                    (&l.up, &self.up, None, false),
                ],
            );
            let n = (tokens * s.ffn) as u32;
            let mut a = [0u32; 16];
            a[0] = n;
            ops.push(self.op(
                "silu_mul",
                &[&self.gate, &self.up],
                a,
                [n.div_ceil(256), 1, 1],
                true,
            ));
            self.products(
                &mut ops,
                &self.gate,
                tokens,
                &[(&l.down, &self.x, None, true)],
            );
        }
        if let Some(hd) = self.head.as_ref().filter(|_| head) {
            ops.push(self.norm(&self.x, &hd.norm, &self.xn, 1, (tokens - 1) * s.hidden));
            self.products(
                &mut ops,
                &self.xn,
                1,
                &[(&hd.proj, &hd.logits, None, false)],
            );
        }
        ops
    }

    fn record(&self, tokens: usize, from: usize, to: usize, head: bool) -> anyhow::Result<Program> {
        let ops = self.ops(tokens, from, to, head);
        let dev = self.ctx.device();
        let sets = ops.len().max(1) as u32;
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(sets * super::BINDINGS)];
        let descriptors = unsafe {
            dev.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(sets)
                    .pool_sizes(&sizes),
                None,
            )?
        };
        let layouts = vec![self.ctx.set_layout(); ops.len()];
        let sets = if ops.is_empty() {
            Vec::new()
        } else {
            unsafe {
                dev.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptors)
                        .set_layouts(&layouts),
                )?
            }
        };
        for (op, &set) in ops.iter().zip(&sets) {
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = op
                .bufs
                .iter()
                .map(|&b| {
                    [vk::DescriptorBufferInfo::default()
                        .buffer(b)
                        .range(vk::WHOLE_SIZE)]
                })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                })
                .collect();
            unsafe { dev.update_descriptor_sets(&writes, &[]) };
        }
        let pool = self.ctx.command_pool()?;
        let cmd = self.ctx.command_buffer(pool)?;
        let bytes = (tokens * self.spec.hidden * 4) as u64;
        let barrier = |src: vk::AccessFlags,
                       dst: vk::AccessFlags,
                       from: vk::PipelineStageFlags,
                       to: vk::PipelineStageFlags| unsafe {
            let mb = [vk::MemoryBarrier::default()
                .src_access_mask(src)
                .dst_access_mask(dst)];
            dev.cmd_pipeline_barrier(cmd, from, to, vk::DependencyFlags::empty(), &mb, &[], &[]);
        };
        let rw = vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE;
        let compute = vk::PipelineStageFlags::COMPUTE_SHADER;
        let transfer = vk::PipelineStageFlags::TRANSFER;
        unsafe {
            dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;
            barrier(rw, vk::AccessFlags::TRANSFER_WRITE, compute, transfer);
            let region = [vk::BufferCopy::default().size(bytes)];
            dev.cmd_copy_buffer(cmd, self.io.buf, self.x.buf, &region);
            barrier(vk::AccessFlags::TRANSFER_WRITE, rw, transfer, compute);
            for (op, &set) in ops.iter().zip(&sets) {
                if op.barrier {
                    barrier(rw, rw, compute, compute);
                }
                dev.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.ctx.pipeline(op.kernel)?,
                );
                dev.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.ctx.pipeline_layout(),
                    0,
                    &[set],
                    &[],
                );
                let words: Vec<u8> = op.args.iter().flat_map(|a| a.to_le_bytes()).collect();
                dev.cmd_push_constants(
                    cmd,
                    self.ctx.pipeline_layout(),
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &words,
                );
                dev.cmd_dispatch(cmd, op.groups[0], op.groups[1], op.groups[2]);
            }
            barrier(rw, vk::AccessFlags::TRANSFER_READ, compute, transfer);
            dev.cmd_copy_buffer(cmd, self.x.buf, self.io.buf, &region);
            if let Some(hd) = self.head.as_ref().filter(|_| head) {
                let region = [vk::BufferCopy::default().size(hd.logits.size)];
                dev.cmd_copy_buffer(cmd, hd.logits.buf, hd.out.buf, &region);
            }
            barrier(
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::HOST_READ,
                transfer,
                vk::PipelineStageFlags::HOST,
            );
            dev.end_command_buffer(cmd)?;
        }
        Ok(Program {
            pool,
            descriptors,
            cmd,
        })
    }
}

#[cfg(test)]
impl VulkanStack {
    /// Microseconds each kernel takes over one step of `tokens` rows at position `index_pos`
    /// through every layer and the head, summed by kernel.
    pub fn profile(
        &self,
        tokens: usize,
        index_pos: usize,
    ) -> anyhow::Result<Vec<(&'static str, f64)>> {
        let ops = self.ops(tokens, 0, self.layers.len(), self.head.is_some());
        let dev = self.ctx.device();
        let n = ops.len() as u32 + 1;
        let pool = unsafe {
            dev.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(n),
                None,
            )?
        };
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(ops.len() as u32 * super::BINDINGS)];
        let descriptors = unsafe {
            dev.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(ops.len() as u32)
                    .pool_sizes(&sizes),
                None,
            )?
        };
        let layouts = vec![self.ctx.set_layout(); ops.len()];
        let sets = unsafe {
            dev.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptors)
                    .set_layouts(&layouts),
            )?
        };
        for (op, &set) in ops.iter().zip(&sets) {
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = op
                .bufs
                .iter()
                .map(|&b| {
                    [vk::DescriptorBufferInfo::default()
                        .buffer(b)
                        .range(vk::WHOLE_SIZE)]
                })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                })
                .collect();
            unsafe { dev.update_descriptor_sets(&writes, &[]) };
        }
        self.step.write(0, &(index_pos as u32).to_le_bytes());
        let mut pipes = Vec::new();
        for op in &ops {
            pipes.push(self.ctx.pipeline(op.kernel)?);
        }
        self.ctx.submit_once(|cmd| unsafe {
            dev.cmd_reset_query_pool(cmd, pool, 0, n);
            let mb = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, pool, 0);
            for (i, (op, &set)) in ops.iter().zip(&sets).enumerate() {
                // Every op serialised, so each interval is that op alone.
                dev.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &mb,
                    &[],
                    &[],
                );
                dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipes[i]);
                dev.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.ctx.pipeline_layout(),
                    0,
                    &[set],
                    &[],
                );
                let words: Vec<u8> = op.args.iter().flat_map(|a| a.to_le_bytes()).collect();
                dev.cmd_push_constants(
                    cmd,
                    self.ctx.pipeline_layout(),
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &words,
                );
                dev.cmd_dispatch(cmd, op.groups[0], op.groups[1], op.groups[2]);
                dev.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    pool,
                    i as u32 + 1,
                );
            }
        })?;
        let mut stamps = vec![0u64; n as usize];
        unsafe {
            dev.get_query_pool_results(
                pool,
                0,
                &mut stamps,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
            dev.destroy_query_pool(pool, None);
            dev.destroy_descriptor_pool(descriptors, None);
        }
        let mut by: Vec<(&'static str, f64)> = Vec::new();
        for (i, op) in ops.iter().enumerate() {
            let us = (stamps[i + 1] - stamps[i]) as f64 * self.ctx.timestamp_period as f64 / 1e3;
            let name = if op.kernel.starts_with("mv_")
                && op.groups[0] as usize * op.groups[1] as usize > 25_000
            {
                "head"
            } else {
                op.kernel
            };
            match by.iter_mut().find(|(k, _)| *k == name) {
                Some(e) => e.1 += us,
                None => by.push((name, us)),
            }
        }
        Ok(by)
    }
}

impl Drop for VulkanStack {
    fn drop(&mut self) {
        let dev = self.ctx.device();
        unsafe {
            let _ = dev.device_wait_idle();
            for (_, p) in self.programs.drain() {
                dev.destroy_command_pool(p.pool, None);
                dev.destroy_descriptor_pool(p.descriptors, None);
            }
        }
    }
}

/// `W x` for `tokens` rows of `x` through the mat-vec kernel of `w`'s format, on device
/// `device`: the kernel alone, for the checks against the host.
#[cfg(test)]
pub(super) fn matvec_probe(
    device: usize,
    w: &RawTensor,
    x: &[f32],
    tokens: usize,
) -> anyhow::Result<Vec<f32>> {
    let ctx = super::context(device)?;
    let m = Matrix::upload(&ctx, w)?;
    let xb = ctx.upload(&f32_bytes(x))?;
    let yb = ctx.buffer((tokens * m.rows * 4) as u64, false)?;
    let out = ctx.buffer((tokens * m.rows * 4) as u64, true)?;
    let dummy = ctx.buffer(16, false)?;
    let dev = ctx.device();
    let sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(super::BINDINGS)];
    let pool = unsafe {
        dev.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&sizes),
            None,
        )?
    };
    let layouts = [ctx.set_layout()];
    let set = unsafe {
        dev.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts),
        )?[0]
    };
    let mut bufs = [dummy.buf; super::BINDINGS as usize];
    bufs[..3].copy_from_slice(&[m.buf.buf, xb.buf, yb.buf]);
    let infos: Vec<[vk::DescriptorBufferInfo; 1]> = bufs
        .iter()
        .map(|&b| {
            [vk::DescriptorBufferInfo::default()
                .buffer(b)
                .range(vk::WHOLE_SIZE)]
        })
        .collect();
    let writes: Vec<vk::WriteDescriptorSet> = infos
        .iter()
        .enumerate()
        .map(|(i, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(info)
        })
        .collect();
    unsafe { dev.update_descriptor_sets(&writes, &[]) };
    let rows = m.rows as u32;
    let args: [u32; 16] = [
        rows,
        m.cols as u32,
        tokens as u32,
        0,
        0,
        0,
        m.cols as u32,
        rows,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ];
    let words: Vec<u8> = args.iter().flat_map(|a| a.to_le_bytes()).collect();
    let pipeline = ctx.pipeline(m.kernel)?;
    ctx.submit_once(|cmd| unsafe {
        dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        dev.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            ctx.pipeline_layout(),
            0,
            &[set],
            &[],
        );
        dev.cmd_push_constants(
            cmd,
            ctx.pipeline_layout(),
            vk::ShaderStageFlags::COMPUTE,
            0,
            &words,
        );
        let [gx, gy, gz] = matvec_groups(rows);
        dev.cmd_dispatch(cmd, gx, gy, gz);
        let mb = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
        dev.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &mb,
            &[],
            &[],
        );
        let region = [vk::BufferCopy::default().size(yb.size)];
        dev.cmd_copy_buffer(cmd, yb.buf, out.buf, &region);
    })?;
    unsafe { dev.destroy_descriptor_pool(pool, None) };
    let mut bytes = vec![0u8; tokens * m.rows * 4];
    out.read(0, &mut bytes);
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}
