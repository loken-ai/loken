//! Dense decoder layers on an OpenCL device, run from the blocks of a GGUF file.
//!
//! A stack holds a run of layers: their weights stay in the file's quantised blocks and are
//! read by the mat-vec kernels as stored, the caches are positional f32, and a step takes the
//! rows on the host and gives the rows back there. With the output head attached, the last
//! row's logits come back as well. A prompt goes through in chunks of [`CHUNK`] rows, each
//! chunk reading every weight once.

#![allow(clippy::too_many_arguments)]

use anyhow::{anyhow, bail, Result};
use opencl3::{
    command_queue::CommandQueue,
    context::Context,
    device::Device,
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, CL_MEM_COPY_HOST_PTR, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE},
    program::Program,
    types::{cl_device_id, CL_BLOCKING},
};
use std::collections::HashMap;
use std::ptr::null_mut;

use crate::tensor::quantized::GgmlDType;

/// Rows a prompt chunk holds: the row count the wide kernels are built for.
pub const CHUNK: usize = 32;

/// Positions each layer's cache starts with; it doubles up to the context.
const FIRST_CACHE: usize = 1024;

/// Threads of a mat-vec workgroup, the lanes that share a group of a row's values, and the
/// slices of a row's groups worked on at once.
const GROUP_THREADS: usize = 64;
const LANES: usize = 8;
const SLICES: usize = 4;
/// Rows a thread takes together, so that each activation load serves that many weight rows.
const ROW_BLOCK: usize = 4;

const FLAG_BIAS: u32 = 1;
const FLAG_RESIDUAL: u32 = 2;

/// The shape of the layers.
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
    /// Positions each layer's cache may hold.
    pub ctx: usize,
}

/// A weight as the file stores it.
pub struct RawTensor {
    pub dtype: GgmlDType,
    /// Rows (outputs) then columns (inputs).
    pub rows: usize,
    pub cols: usize,
    pub bytes: Vec<u8>,
}

/// One layer's tensors, read from the file.
pub struct LayerTensors {
    pub attn_norm: Vec<f32>,
    pub q: RawTensor,
    pub k: RawTensor,
    pub v: RawTensor,
    pub q_bias: Option<Vec<f32>>,
    pub k_bias: Option<Vec<f32>>,
    pub v_bias: Option<Vec<f32>>,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub o: RawTensor,
    pub ffn_norm: Vec<f32>,
    pub gate: RawTensor,
    pub up: RawTensor,
    pub down: RawTensor,
}

/// Whether the mat-vec kernels read `dtype`.
pub fn runs_format(dtype: GgmlDType) -> bool {
    format_of(dtype).is_some()
}

/// The mat-vec family of a stored format.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fmt {
    Q8_0,
    Q4_0,
    Q5_0,
    F16,
    F32,
    Q4K,
    Q6K,
}

fn format_of(dtype: GgmlDType) -> Option<Fmt> {
    Some(match dtype {
        GgmlDType::Q8_0 => Fmt::Q8_0,
        GgmlDType::Q4_0 => Fmt::Q4_0,
        GgmlDType::Q5_0 => Fmt::Q5_0,
        GgmlDType::F16 => Fmt::F16,
        GgmlDType::F32 => Fmt::F32,
        GgmlDType::Q4K => Fmt::Q4K,
        GgmlDType::Q6K => Fmt::Q6K,
        _ => return None,
    })
}

impl Fmt {
    /// The number the single-launch layer kernel knows the format by.
    fn id(self) -> u32 {
        match self {
            Fmt::Q8_0 => 0,
            Fmt::Q4_0 => 1,
            Fmt::Q5_0 => 2,
            Fmt::F16 => 3,
            Fmt::F32 => 4,
            Fmt::Q4K => 5,
            Fmt::Q6K => 6,
        }
    }

    fn mat_vec(self) -> &'static str {
        match self {
            Fmt::Q8_0 => "mv_q8_0",
            Fmt::Q4_0 => "mv_q4_0",
            Fmt::Q5_0 => "mv_q5_0",
            Fmt::F16 => "mv_f16",
            Fmt::F32 => "mv_f32",
            Fmt::Q4K => "mv_q4_k",
            Fmt::Q6K => "mv_q6_k",
        }
    }

    fn gate_up(self) -> &'static str {
        match self {
            Fmt::Q8_0 => "gu_q8_0",
            Fmt::Q4_0 => "gu_q4_0",
            Fmt::Q5_0 => "gu_q5_0",
            Fmt::F16 => "gu_f16",
            Fmt::F32 => "gu_f32",
            Fmt::Q4K => "gu_q4_k",
            Fmt::Q6K => "gu_q6_k",
        }
    }

    /// Values one lane takes per step along a row.
    fn unit_elems(self) -> usize {
        match self {
            Fmt::Q4K => 64,
            _ => 32,
        }
    }
}

const KERNELS: [&str; 19] = [
    "mv_q8_0",
    "mv_q4_0",
    "mv_q5_0",
    "mv_f16",
    "mv_f32",
    "mv_q4_k",
    "mv_q6_k",
    "gu_q8_0",
    "gu_q4_0",
    "gu_q5_0",
    "gu_f16",
    "gu_f32",
    "gu_q4_k",
    "gu_q6_k",
    "rms_norm_rows",
    "qkv_post",
    "attention_rows",
    "silu_mul",
    "quant_rows",
];

/// The kernels of one program, built for a row count.
struct Kernels {
    by_name: HashMap<&'static str, Kernel>,
    /// Output rows of one mat-vec workgroup, as the program was built.
    rows_per_group: usize,
}

impl Kernels {
    fn build(ctx: &Context, rows: usize) -> Result<Self> {
        Self::build_from(
            ctx,
            rows,
            include_str!("../../kernels/dense_stack.cl"),
            SLICES,
            ROW_BLOCK,
        )
    }

    /// The program for a prompt chunk: a thread takes one row, its registers go to the chunk.
    fn wide(ctx: &Context) -> Result<Self> {
        Self::build_from(
            ctx,
            CHUNK,
            include_str!("../../kernels/dense_stack.cl"),
            SLICES,
            1,
        )
    }

    fn build_from(
        ctx: &Context,
        rows: usize,
        source: &str,
        slices: usize,
        row_block: usize,
    ) -> Result<Self> {
        let defines = format!("-DR={rows} -DSLICES={slices} -DRB={row_block}");
        // OpenCL C 3.0 exposes the integer dot product where the device has it; a compiler that
        // does not know the option builds the same source with the scalar fallback.
        let program =
            Program::create_and_build_from_source(ctx, source, &format!("-cl-std=CL3.0 {defines}"))
                .or_else(|_| Program::create_and_build_from_source(ctx, source, &defines))
                .map_err(|e| anyhow!("OpenCL program build error: {e}"))?;
        let mut by_name = HashMap::new();
        for name in KERNELS {
            by_name.insert(name, Kernel::create(&program, name)?);
        }
        Ok(Self {
            by_name,
            rows_per_group: GROUP_THREADS / (LANES * slices) * row_block,
        })
    }

    fn get(&self, name: &str) -> &Kernel {
        &self.by_name[name]
    }
}

/// A weight matrix on the device.
struct Matrix {
    buf: Buffer<u8>,
    fmt: Fmt,
    rows: usize,
    cols: usize,
    /// Blocks of the format in the matrix.
    total: usize,
}

impl Matrix {
    fn upload(ctx: &Context, t: &RawTensor, max_alloc: usize) -> Result<Self> {
        let fmt = format_of(t.dtype)
            .ok_or_else(|| anyhow!("OpenCL mat-vec does not read {}", t.dtype.gguf_name()))?;
        let block = t.dtype.block_size();
        if t.cols % block != 0 || t.cols % fmt.unit_elems() != 0 {
            bail!("a {} matrix of {} columns", t.dtype.gguf_name(), t.cols);
        }
        if matches!(fmt, Fmt::Q4K | Fmt::Q6K) && t.cols % 256 != 0 {
            bail!("a {} matrix of {} columns", t.dtype.gguf_name(), t.cols);
        }
        let total = t.rows * t.cols / block;
        if t.bytes.len() != total * t.dtype.type_size() {
            bail!(
                "a {} matrix of {}x{} holds {} bytes, expected {}",
                t.dtype.gguf_name(),
                t.rows,
                t.cols,
                t.bytes.len(),
                total * t.dtype.type_size()
            );
        }
        let device_bytes = repack(fmt, &t.bytes, total);
        if device_bytes.len() > max_alloc {
            bail!(
                "a {} matrix of {} bytes exceeds the device's allocation limit of {max_alloc}",
                t.dtype.gguf_name(),
                device_bytes.len()
            );
        }
        let buf = upload_bytes(ctx, &device_bytes)?;
        Ok(Self {
            buf,
            fmt,
            rows: t.rows,
            cols: t.cols,
            total,
        })
    }
}

/// The bytes of `total` blocks of `fmt` as the kernels read them. The formats whose blocks are
/// not a multiple of 16 bytes are split into one array per field, each aligned for wide loads:
/// the quants of every block, then the other fields in the order the kernel reads them.
fn repack(fmt: Fmt, src: &[u8], total: usize) -> Vec<u8> {
    let (size, fields): (usize, &[(usize, usize)]) = match fmt {
        // (offset in the block, bytes) of each field, quants first.
        Fmt::Q8_0 => (34, &[(2, 32), (0, 2)]),
        Fmt::Q4_0 => (18, &[(2, 16), (0, 2)]),
        Fmt::Q5_0 => (22, &[(6, 16), (2, 4), (0, 2)]),
        Fmt::Q6K => (210, &[(0, 128), (128, 64), (192, 16), (208, 2)]),
        _ => return src.to_vec(),
    };
    let mut out = Vec::with_capacity(src.len());
    for &(offset, len) in fields {
        for block in src.chunks_exact(size).take(total) {
            out.extend_from_slice(&block[offset..offset + len]);
        }
    }
    out
}

fn upload_bytes(ctx: &Context, bytes: &[u8]) -> Result<Buffer<u8>> {
    unsafe {
        Ok(Buffer::<u8>::create(
            ctx,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            bytes.len().max(4),
            bytes.as_ptr() as *mut std::ffi::c_void,
        )?)
    }
}

fn upload_f32(ctx: &Context, v: &[f32]) -> Result<Buffer<f32>> {
    let zero = [0.0f32];
    let v = if v.is_empty() { &zero[..] } else { v };
    unsafe {
        Ok(Buffer::<f32>::create(
            ctx,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            v.len(),
            v.as_ptr() as *mut std::ffi::c_void,
        )?)
    }
}

fn scratch(ctx: &Context, n: usize) -> Result<Buffer<f32>> {
    unsafe {
        Ok(Buffer::<f32>::create(
            ctx,
            CL_MEM_READ_WRITE,
            n.max(1),
            null_mut(),
        )?)
    }
}

/// Threads of a workgroup of the single-launch layer kernel, and the workgroups of its grid:
/// all of them must be resident at once for the barrier between phases to be crossed.
const MEGA_THREADS: usize = 256;
const MEGA_GROUPS: [usize; 8] = [128, 96, 64, 48, 32, 24, 16, 8];

/// The single-launch decode layer: a grid of workgroups that crosses a barrier between the
/// phases of the layer, in place of a launch for each phase.
struct Mega {
    layer: Kernel,
    head: Kernel,
    /// Arrival counters, eight words to a layer, and the count of barriers given up on.
    counters: Buffer<f32>,
    fail: Buffer<f32>,
    groups: usize,
}

impl Mega {
    /// The program, if the device can run it: it needs acquire/release atomics across the
    /// device, and a grid of the size that is resident together.
    fn new(ctx: &Context, spec: &DenseSpec, layers: &[Layer]) -> Result<Self> {
        let q_dim = spec.n_head * spec.head_dim;
        let widest = spec.hidden.max(q_dim).max(spec.ffn).div_ceil(256) * 256;
        if widest > 8192
            || spec.head_dim > 256
            || spec.hidden % 32 != 0
            || spec.n_head % spec.n_kv != 0
        {
            bail!("the shape of the layers");
        }
        // The floats of an activation stay in local memory only for the rows the float
        // formats read, which are the hidden state's.
        let float_in = |f: Fmt| matches!(f, Fmt::F16 | Fmt::F32);
        if layers
            .iter()
            .any(|l| float_in(l.o.fmt) || float_in(l.down.fmt) || l.gate.fmt != l.up.fmt)
        {
            bail!("a format the kernel does not read from local memory");
        }
        let source = format!(
            "{}\n{}",
            include_str!("../../kernels/dense_stack.cl"),
            include_str!("../../kernels/dense_mega.cl")
        );
        let program = Program::create_and_build_from_source(
            ctx,
            &source,
            &format!(
                "-cl-std=CL3.0 -DMEGA -DR=1 -DSLICES={SLICES} -DRB=1 -DMAXC={widest} -DMAXH={}",
                spec.hidden
            ),
        )
        .map_err(|e| anyhow!("{e}"))?;
        Ok(Self {
            layer: Kernel::create(&program, "layer_step")?,
            head: Kernel::create(&program, "head_step")?,
            counters: scratch(ctx, (layers.len() + 1) * 8)?,
            fail: scratch(ctx, 4)?,
            groups: MEGA_GROUPS[0],
        })
    }

    /// Zeroes the counters of `layers` layers and the failure count.
    fn clear(&mut self, queue: &CommandQueue, layers: usize) -> Result<()> {
        let zero = [0f32];
        unsafe {
            queue.enqueue_fill_buffer(
                &mut self.counters,
                &zero,
                0,
                (layers.max(1) * 8) * 4,
                &[],
            )?;
            queue.enqueue_fill_buffer(&mut self.fail, &zero, 0, 4 * 4, &[])?;
        }
        Ok(())
    }

    /// Barriers some workgroup gave up waiting at since the last clear.
    fn failures(&self, queue: &CommandQueue) -> Result<u32> {
        let mut n = [0f32; 1];
        unsafe {
            queue.enqueue_read_buffer(&self.fail, CL_BLOCKING, 0, &mut n, &[])?;
        }
        Ok(n[0].to_bits())
    }
}

/// An activation row quantised for the integer formats: the codes and scales the kernels read,
/// for up to `CHUNK` rows of `cols` values.
struct Quant {
    qk: Buffer<u8>,
    qz: Buffer<u8>,
    dk: Buffer<f32>,
    /// Integer sums, held in float-sized words.
    sk: Buffer<f32>,
    dz: Buffer<f32>,
}

impl Quant {
    fn new(ctx: &Context, cols: usize) -> Result<Self> {
        let rows = CHUNK;
        let bytes = |n: usize| -> Result<Buffer<u8>> {
            unsafe {
                Ok(Buffer::<u8>::create(
                    ctx,
                    CL_MEM_READ_WRITE,
                    n.max(4),
                    null_mut(),
                )?)
            }
        };
        Ok(Self {
            qk: bytes(rows * cols)?,
            qz: bytes(rows * cols)?,
            dk: scratch(ctx, rows * cols.div_ceil(256))?,
            sk: scratch(ctx, rows * cols.div_ceil(32))?,
            dz: scratch(ctx, rows * cols.div_ceil(32))?,
        })
    }
}

/// Activation rows as floats and as codes.
#[derive(Clone, Copy)]
struct Act<'a> {
    f: &'a Buffer<f32>,
    q: &'a Quant,
}

struct Layer {
    attn_norm: Buffer<f32>,
    q: Matrix,
    k: Matrix,
    v: Matrix,
    q_bias: Option<Buffer<f32>>,
    k_bias: Option<Buffer<f32>>,
    v_bias: Option<Buffer<f32>>,
    q_norm: Option<Buffer<f32>>,
    k_norm: Option<Buffer<f32>>,
    o: Matrix,
    ffn_norm: Buffer<f32>,
    gate: Matrix,
    up: Matrix,
    down: Matrix,
    cache_k: Buffer<f32>,
    cache_v: Buffer<f32>,
    cache_cap: usize,
}

struct Head {
    norm: Buffer<f32>,
    proj: Matrix,
    logits: Buffer<f32>,
}

/// Layers on one OpenCL device.
pub struct OpenClStack {
    ctx: Context,
    queue: CommandQueue,
    wide: Kernels,
    narrow: Kernels,
    spec: DenseSpec,
    layers: Vec<Layer>,
    cos: Buffer<f32>,
    sin: Buffer<f32>,
    table_positions: usize,
    half: usize,
    head: Option<Head>,
    unused: Buffer<f32>,
    // Activation buffers for one chunk.
    a: Buffer<f32>,
    b: Buffer<f32>,
    xn: Buffer<f32>,
    /// The codes of `xn`, of the attention output and of the FFN activation.
    qn: Quant,
    qa: Quant,
    qf: Quant,
    q: Buffer<f32>,
    k: Buffer<f32>,
    v: Buffer<f32>,
    att: Buffer<f32>,
    act: Buffer<f32>,
    act2: Buffer<f32>,
    max_alloc: usize,
    /// The single-launch decode layer, where the device runs it, and whether it is in use.
    mega: Option<Mega>,
    mega_on: bool,
}

// The handles are raw pointers into a driver that serialises access through the one queue,
// which only `&mut self` methods use.
unsafe impl Send for OpenClStack {}
unsafe impl Sync for OpenClStack {}

/// Queues the quantisation of `rows` rows of `cols` values of `x` into `q`.
fn launch_quant(
    queue: &CommandQueue,
    kernels: &Kernels,
    x: &Buffer<f32>,
    q: &Quant,
    rows: usize,
    cols: usize,
) -> Result<opencl3::event::Event> {
    let blocks = rows * cols.div_ceil(256);
    unsafe {
        Ok(ExecuteKernel::new(kernels.get("quant_rows"))
            .set_arg(x)
            .set_arg(&q.qk)
            .set_arg(&q.qz)
            .set_arg(&q.dk)
            .set_arg(&q.sk)
            .set_arg(&q.dz)
            .set_arg(&(cols as u32))
            .set_arg(&0u32)
            .set_global_work_size(blocks * 256)
            .set_local_work_size(256)
            .enqueue_nd_range(queue)?)
    }
}

/// Queues the mat-vec of `m` over `rows` activation rows `x`. The residual may be the
/// output itself: each output is read and written by the one thread that owns it.
fn launch_mat_vec(
    queue: &CommandQueue,
    kernels: &Kernels,
    unused: &Buffer<f32>,
    m: &Matrix,
    x: Act,
    out: &Buffer<f32>,
    rows: usize,
    bias: Option<&Buffer<f32>>,
    residual: Option<&Buffer<f32>>,
) -> Result<opencl3::event::Event> {
    let flags = if bias.is_some() { FLAG_BIAS } else { 0 }
        | if residual.is_some() { FLAG_RESIDUAL } else { 0 };
    let groups = m.rows.div_ceil(kernels.rows_per_group);
    unsafe {
        Ok(ExecuteKernel::new(kernels.get(m.fmt.mat_vec()))
            .set_arg(&m.buf)
            .set_arg(x.f)
            .set_arg(&x.q.qk)
            .set_arg(&x.q.qz)
            .set_arg(&x.q.dk)
            .set_arg(&x.q.sk)
            .set_arg(&x.q.dz)
            .set_arg(out)
            .set_arg(&(m.cols as u32))
            .set_arg(&(m.rows as u32))
            .set_arg(&(rows as u32))
            .set_arg(bias.unwrap_or(unused))
            .set_arg(residual.unwrap_or(unused))
            .set_arg(&flags)
            .set_arg(&(m.total as u64))
            .set_global_work_size(groups * 64)
            .set_local_work_size(64)
            .enqueue_nd_range(queue)?)
    }
}

impl OpenClStack {
    /// A stack of `layers` on the `device`-th OpenCL GPU. `cos` and `sin` are the RoPE
    /// tables, `half` values per position.
    #[allow(deprecated)]
    pub fn new(
        device: usize,
        spec: DenseSpec,
        layers: Vec<LayerTensors>,
        cos: &[f32],
        sin: &[f32],
        half: usize,
    ) -> Result<Self> {
        let ids = super::opencl::enumerate_opencl_devices();
        let id = *ids
            .get(device)
            .ok_or_else(|| anyhow!("no OpenCL device {device}"))?;
        let dev = Device::new(id);
        let max_alloc = dev.max_mem_alloc_size().unwrap_or(0) as usize;
        let ctx = Context::from_device(&dev)?;
        let queue = CommandQueue::create_default(&ctx, 0)?;
        if spec.head_dim > 256 || spec.head_dim % 4 != 0 {
            bail!("a head of {} lanes", spec.head_dim);
        }
        if half == 0 || cos.len() != sin.len() || cos.len() % half != 0 {
            bail!("RoPE tables of {} values, {half} per position", cos.len());
        }
        let wide = Kernels::wide(&ctx)?;
        let narrow = Kernels::build(&ctx, 1)?;
        let q_dim = spec.n_head * spec.head_dim;
        let kv_dim = spec.n_kv * spec.head_dim;
        let first_cap = FIRST_CACHE.min(spec.ctx).max(1);
        let mut out = Vec::with_capacity(layers.len());
        for l in &layers {
            let m = |t: &RawTensor, rows: usize, cols: usize| -> Result<Matrix> {
                if t.rows != rows || t.cols != cols {
                    bail!("a {}x{} matrix where {rows}x{cols} belongs", t.rows, t.cols);
                }
                Matrix::upload(&ctx, t, max_alloc)
            };
            let opt = |v: &Option<Vec<f32>>| -> Result<Option<Buffer<f32>>> {
                v.as_ref().map(|v| upload_f32(&ctx, v)).transpose()
            };
            out.push(Layer {
                attn_norm: upload_f32(&ctx, &l.attn_norm)?,
                q: m(&l.q, q_dim, spec.hidden)?,
                k: m(&l.k, kv_dim, spec.hidden)?,
                v: m(&l.v, kv_dim, spec.hidden)?,
                q_bias: opt(&l.q_bias)?,
                k_bias: opt(&l.k_bias)?,
                v_bias: opt(&l.v_bias)?,
                q_norm: opt(&l.q_norm)?,
                k_norm: opt(&l.k_norm)?,
                o: m(&l.o, spec.hidden, q_dim)?,
                ffn_norm: upload_f32(&ctx, &l.ffn_norm)?,
                gate: m(&l.gate, spec.ffn, spec.hidden)?,
                up: m(&l.up, spec.ffn, spec.hidden)?,
                down: m(&l.down, spec.hidden, spec.ffn)?,
                cache_k: scratch(&ctx, first_cap * kv_dim)?,
                cache_v: scratch(&ctx, first_cap * kv_dim)?,
                cache_cap: first_cap,
            });
        }
        let mega = match Mega::new(&ctx, &spec, &out) {
            Ok(m) => Some(m),
            Err(e) => {
                tracing::info!("OpenCL layers run a launch to a phase: {e}");
                None
            }
        };
        let mut stack = Self {
            mega_on: mega.is_some(),
            mega,
            cos: upload_f32(&ctx, cos)?,
            sin: upload_f32(&ctx, sin)?,
            table_positions: cos.len() / half,
            half,
            head: None,
            unused: scratch(&ctx, 4)?,
            a: scratch(&ctx, CHUNK * spec.hidden)?,
            b: scratch(&ctx, CHUNK * spec.hidden)?,
            xn: scratch(&ctx, CHUNK * spec.hidden)?,
            qn: Quant::new(&ctx, spec.hidden)?,
            qa: Quant::new(&ctx, q_dim)?,
            qf: Quant::new(&ctx, spec.ffn)?,
            q: scratch(&ctx, CHUNK * q_dim)?,
            k: scratch(&ctx, CHUNK * kv_dim)?,
            v: scratch(&ctx, CHUNK * kv_dim)?,
            att: scratch(&ctx, CHUNK * q_dim)?,
            act: scratch(&ctx, CHUNK * spec.ffn)?,
            act2: scratch(&ctx, CHUNK * spec.ffn)?,
            max_alloc,
            ctx,
            queue,
            wide,
            narrow,
            spec,
            layers: out,
        };
        stack.settle_grid();
        if let Some(m) = &stack.mega {
            tracing::debug!(
                "single-launch layer: {} workgroups of {MEGA_THREADS}, {} bytes of local memory",
                m.groups,
                m.layer.get_local_mem_size(id).unwrap_or(0),
            );
        }
        Ok(stack)
    }

    /// Finds the largest grid of workgroups for the single-launch layer that is resident all at
    /// once, by running a layer on it; without one the layer runs a launch to a phase.
    fn settle_grid(&mut self) {
        if self.mega.is_none() || self.layers.is_empty() {
            return;
        }
        for groups in MEGA_GROUPS {
            if let Some(m) = self.mega.as_mut() {
                m.groups = groups;
                if m.clear(&self.queue, 1).is_err() {
                    break;
                }
            }
            let ran = self.layer_single(0, 0, 0, true).and_then(|_| {
                self.queue.finish()?;
                self.mega
                    .as_ref()
                    .map_or(Ok(1), |m| m.failures(&self.queue))
            });
            if let Ok(0) = ran {
                return;
            }
        }
        tracing::info!(
            "OpenCL layers run a launch to a phase: no grid of workgroups is resident together"
        );
        self.mega = None;
        self.mega_on = false;
    }

    /// Puts the output head on the device: the final norm's weight and the projection.
    pub fn attach_head(&mut self, norm: &[f32], proj: &RawTensor) -> Result<()> {
        if proj.cols != self.spec.hidden || norm.len() != self.spec.hidden {
            bail!("an output head of {} columns", proj.cols);
        }
        let proj_m = Matrix::upload(&self.ctx, proj, self.max_alloc)?;
        self.head = Some(Head {
            norm: upload_f32(&self.ctx, norm)?,
            logits: scratch(&self.ctx, proj.rows)?,
            proj: proj_m,
        });
        Ok(())
    }

    /// Whether the output head is on the device.
    pub fn has_head(&self) -> bool {
        self.head.is_some()
    }

    /// Layers in the stack.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Runs `rows` (`n * hidden` values, the first at `index_pos`) through layers `[from, to)`
    /// and returns the rows after them.
    pub fn forward(
        &mut self,
        rows: &[f32],
        index_pos: usize,
        from: usize,
        to: usize,
    ) -> Result<Vec<f32>> {
        self.run(rows, index_pos, from, to, false).map(|(r, _)| r)
    }

    /// [`forward`](Self::forward) through the last layer, and the logits of the last row.
    pub fn forward_to_logits(
        &mut self,
        rows: &[f32],
        index_pos: usize,
        from: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let to = self.layers.len();
        let (rows, logits) = self.run(rows, index_pos, from, to, true)?;
        Ok((
            rows,
            logits.ok_or_else(|| anyhow!("the stack holds no output head"))?,
        ))
    }

    fn run(
        &mut self,
        rows: &[f32],
        index_pos: usize,
        from: usize,
        to: usize,
        logits: bool,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        let hidden = self.spec.hidden;
        if to > self.layers.len() || from > to || rows.len() % hidden != 0 || rows.is_empty() {
            bail!("rows {} layers {from}..{to}", rows.len());
        }
        let n = rows.len() / hidden;
        let need = index_pos + n;
        if need > self.spec.ctx || need > self.table_positions {
            bail!("position {need} is past the context of {}", self.spec.ctx);
        }
        for li in from..to {
            self.ensure_cache(li, need)?;
        }
        let mut out = vec![0f32; rows.len()];
        let mut last_logits = None;
        let mut done = 0;
        while done < n {
            let c = CHUNK.min(n - done);
            let chunk = &rows[done * hidden..(done + c) * hidden];
            let mut single = c == 1 && self.mega_on && self.mega.is_some();
            // Layers alternate their input between the two buffers.
            let mut in_a;
            loop {
                unsafe {
                    self.queue
                        .enqueue_write_buffer(&mut self.a, CL_BLOCKING, 0, chunk, &[])?;
                }
                in_a = true;
                if single {
                    if let Some(mega) = self.mega.as_mut() {
                        mega.clear(&self.queue, to - from)?;
                    }
                }
                for li in from..to {
                    if single {
                        self.layer_single(li, from, index_pos + done, in_a)?;
                    } else {
                        self.layer(li, c, index_pos + done, in_a)?;
                    }
                    in_a = !in_a;
                }
                if !single {
                    break;
                }
                // A grid that was not resident gave up at a barrier: the step runs again, a
                // launch to a phase from here on.
                self.queue.finish()?;
                let failed = self
                    .mega
                    .as_ref()
                    .map_or(0, |m| m.failures(&self.queue).unwrap_or(1));
                if failed == 0 {
                    break;
                }
                tracing::warn!("the single-launch layer was not resident ({failed} barriers given up); running a launch to a phase");
                self.mega_on = false;
                single = false;
            }
            let last_chunk = done + c == n;
            let cur = if in_a { &self.a } else { &self.b };
            if logits && last_chunk && to == self.layers.len() {
                if single {
                    self.head_single(cur)?;
                } else {
                    self.head_logits(cur, c)?;
                }
                let h = self
                    .head
                    .as_ref()
                    .ok_or_else(|| anyhow!("no output head"))?;
                let mut l = vec![0f32; h.proj.rows];
                unsafe {
                    self.queue
                        .enqueue_read_buffer(&h.logits, CL_BLOCKING, 0, &mut l, &[])?;
                }
                last_logits = Some(l);
            }
            let cur = if in_a { &self.a } else { &self.b };
            unsafe {
                self.queue.enqueue_read_buffer(
                    cur,
                    CL_BLOCKING,
                    0,
                    &mut out[done * hidden..(done + c) * hidden],
                    &[],
                )?;
            }
            done += c;
        }
        Ok((out, last_logits))
    }

    /// Grows layer `li`'s cache to hold `need` positions.
    fn ensure_cache(&mut self, li: usize, need: usize) -> Result<()> {
        let kv_dim = self.spec.n_kv * self.spec.head_dim;
        let cap = self.layers[li].cache_cap;
        if need <= cap {
            return Ok(());
        }
        let mut new_cap = cap;
        while new_cap < need {
            new_cap *= 2;
        }
        let new_cap = new_cap.min(self.spec.ctx).max(need);
        if new_cap * kv_dim * 4 > self.max_alloc {
            bail!("a cache of {new_cap} positions exceeds the device's allocation limit");
        }
        let mut nk = scratch(&self.ctx, new_cap * kv_dim)?;
        let mut nv = scratch(&self.ctx, new_cap * kv_dim)?;
        let l = &mut self.layers[li];
        unsafe {
            self.queue
                .enqueue_copy_buffer(&l.cache_k, &mut nk, 0, 0, cap * kv_dim * 4, &[])?;
            self.queue
                .enqueue_copy_buffer(&l.cache_v, &mut nv, 0, 0, cap * kv_dim * 4, &[])?;
            self.queue.finish()?;
        }
        l.cache_k = nk;
        l.cache_v = nv;
        l.cache_cap = new_cap;
        Ok(())
    }

    fn kernels(&self, rows: usize) -> &Kernels {
        if rows == 1 {
            &self.narrow
        } else {
            &self.wide
        }
    }

    fn mat_vec(
        &self,
        m: &Matrix,
        x: Act,
        out: &Buffer<f32>,
        rows: usize,
        bias: Option<&Buffer<f32>>,
        residual: Option<&Buffer<f32>>,
    ) -> Result<()> {
        launch_mat_vec(
            &self.queue,
            self.kernels(rows),
            &self.unused,
            m,
            x,
            out,
            rows,
            bias,
            residual,
        )?;
        Ok(())
    }

    /// Quantises `rows` rows of `cols` values of `x` into `q`.
    fn quantize(&self, x: &Buffer<f32>, q: &Quant, rows: usize, cols: usize) -> Result<()> {
        launch_quant(&self.queue, self.kernels(rows), x, q, rows, cols)?;
        Ok(())
    }

    fn rms_norm(
        &self,
        x: &Buffer<f32>,
        w: &Buffer<f32>,
        out: &Buffer<f32>,
        rows: usize,
        x_off: usize,
        kernels_rows: usize,
    ) -> Result<()> {
        unsafe {
            ExecuteKernel::new(self.kernels(kernels_rows).get("rms_norm_rows"))
                .set_arg(x)
                .set_arg(w)
                .set_arg(out)
                .set_arg(&(self.spec.hidden as u32))
                .set_arg(&self.spec.eps)
                .set_arg(&(x_off as u32))
                .set_global_work_size(rows * 128)
                .set_local_work_size(128)
                .enqueue_nd_range(&self.queue)?
        };
        Ok(())
    }

    /// One layer over `c` rows starting at `pos`. The input is in `a` when `in_a`, else `b`;
    /// the output lands in the other.
    fn layer(&self, li: usize, c: usize, pos: usize, in_a: bool) -> Result<()> {
        let s = &self.spec;
        let l = &self.layers[li];
        let (cur, nxt) = if in_a {
            (&self.a, &self.b)
        } else {
            (&self.b, &self.a)
        };
        let kern = self.kernels(c);
        let xn = Act {
            f: &self.xn,
            q: &self.qn,
        };
        self.rms_norm(cur, &l.attn_norm, &self.xn, c, 0, c)?;
        self.quantize(&self.xn, &self.qn, c, s.hidden)?;
        self.mat_vec(&l.q, xn, &self.q, c, l.q_bias.as_ref(), None)?;
        self.mat_vec(&l.k, xn, &self.k, c, l.k_bias.as_ref(), None)?;
        self.mat_vec(&l.v, xn, &self.v, c, l.v_bias.as_ref(), None)?;
        unsafe {
            ExecuteKernel::new(kern.get("qkv_post"))
                .set_arg(&self.q)
                .set_arg(&self.k)
                .set_arg(&self.v)
                .set_arg(&l.cache_k)
                .set_arg(&l.cache_v)
                .set_arg(&self.cos)
                .set_arg(&self.sin)
                .set_arg(l.q_norm.as_ref().unwrap_or(&self.unused))
                .set_arg(l.k_norm.as_ref().unwrap_or(&self.unused))
                .set_arg(&(s.n_head as u32))
                .set_arg(&(s.n_kv as u32))
                .set_arg(&(s.head_dim as u32))
                .set_arg(&(s.rope_dim as u32))
                .set_arg(&(self.half as u32))
                .set_arg(&(s.rope_adjacent as u32))
                .set_arg(&(l.q_norm.is_some() as u32))
                .set_arg(&(l.k_norm.is_some() as u32))
                .set_arg(&s.eps)
                .set_arg(&(pos as u32))
                .set_global_work_size(c * (s.n_head + 2 * s.n_kv) * 128)
                .set_local_work_size(128)
                .enqueue_nd_range(&self.queue)?;
            ExecuteKernel::new(kern.get("attention_rows"))
                .set_arg(&self.q)
                .set_arg(&l.cache_k)
                .set_arg(&l.cache_v)
                .set_arg(&self.att)
                .set_arg(&(s.n_head as u32))
                .set_arg(&(s.n_kv as u32))
                .set_arg(&(s.head_dim as u32))
                .set_arg(&s.attn_scale)
                .set_arg(&(pos as u32))
                .set_global_work_size(c * s.n_head * 128)
                .set_local_work_size(128)
                .enqueue_nd_range(&self.queue)?;
        }
        self.quantize(&self.att, &self.qa, c, s.n_head * s.head_dim)?;
        let att = Act {
            f: &self.att,
            q: &self.qa,
        };
        self.mat_vec(&l.o, att, nxt, c, None, Some(cur))?;
        self.rms_norm(nxt, &l.ffn_norm, &self.xn, c, 0, c)?;
        self.quantize(&self.xn, &self.qn, c, s.hidden)?;
        if l.gate.fmt == l.up.fmt {
            let groups = l.gate.rows.div_ceil(kern.rows_per_group);
            unsafe {
                ExecuteKernel::new(kern.get(l.gate.fmt.gate_up()))
                    .set_arg(&l.gate.buf)
                    .set_arg(&l.up.buf)
                    .set_arg(&self.xn)
                    .set_arg(&self.qn.qk)
                    .set_arg(&self.qn.qz)
                    .set_arg(&self.qn.dk)
                    .set_arg(&self.qn.sk)
                    .set_arg(&self.qn.dz)
                    .set_arg(&self.act)
                    .set_arg(&(l.gate.cols as u32))
                    .set_arg(&(l.gate.rows as u32))
                    .set_arg(&(c as u32))
                    .set_arg(&(l.gate.total as u64))
                    .set_global_work_size(groups * 64)
                    .set_local_work_size(64)
                    .enqueue_nd_range(&self.queue)?;
            }
        } else {
            self.mat_vec(&l.gate, xn, &self.act, c, None, None)?;
            self.mat_vec(&l.up, xn, &self.act2, c, None, None)?;
            let n = c * s.ffn;
            unsafe {
                ExecuteKernel::new(kern.get("silu_mul"))
                    .set_arg(&self.act)
                    .set_arg(&self.act2)
                    .set_arg(&self.act)
                    .set_arg(&(n as u32))
                    .set_global_work_size(n.div_ceil(64) * 64)
                    .set_local_work_size(64)
                    .enqueue_nd_range(&self.queue)?;
            }
        }
        self.quantize(&self.act, &self.qf, c, s.ffn)?;
        let act = Act {
            f: &self.act,
            q: &self.qf,
        };
        self.mat_vec(&l.down, act, nxt, c, None, Some(nxt))?;
        Ok(())
    }

    /// Chooses whether decode steps run each layer in one launch, where the device allows it.
    pub fn set_single_launch(&mut self, on: bool) {
        self.mega_on = on && self.mega.is_some();
    }

    /// Whether decode steps run each layer in one launch.
    pub fn single_launch(&self) -> bool {
        self.mega_on
    }

    /// The workgroups of a single-launch layer, and a way to ask for another count.
    pub fn single_launch_groups(&self) -> usize {
        self.mega.as_ref().map_or(0, |m| m.groups)
    }

    pub fn set_single_launch_groups(&mut self, groups: usize) {
        if let Some(m) = self.mega.as_mut() {
            m.groups = groups;
        }
    }

    /// Layer `li` of a decode step in one launch; `first` is the first layer of the step.
    fn layer_single(&self, li: usize, first: usize, pos: usize, in_a: bool) -> Result<()> {
        let mega = self
            .mega
            .as_ref()
            .ok_or_else(|| anyhow!("no single-launch layer"))?;
        let s = &self.spec;
        let l = &self.layers[li];
        let (cur, nxt) = if in_a {
            (&self.a, &self.b)
        } else {
            (&self.b, &self.a)
        };
        let unused = &self.unused;
        let flags = l.q_bias.is_some() as u32
            | (l.k_bias.is_some() as u32) << 1
            | (l.v_bias.is_some() as u32) << 2
            | (l.q_norm.is_some() as u32) << 3
            | (l.k_norm.is_some() as u32) << 4
            | (s.rope_adjacent as u32) << 5;
        unsafe {
            let mut e = ExecuteKernel::new(&mega.layer);
            e.set_arg(cur)
                .set_arg(nxt)
                .set_arg(&self.q)
                .set_arg(&self.k)
                .set_arg(&self.v)
                .set_arg(&self.att)
                .set_arg(&self.act)
                .set_arg(&l.q.buf)
                .set_arg(&l.k.buf)
                .set_arg(&l.v.buf)
                .set_arg(&l.o.buf)
                .set_arg(&l.gate.buf)
                .set_arg(&l.up.buf)
                .set_arg(&l.down.buf)
                .set_arg(&l.q.fmt.id())
                .set_arg(&l.k.fmt.id())
                .set_arg(&l.v.fmt.id())
                .set_arg(&l.o.fmt.id())
                .set_arg(&l.gate.fmt.id())
                .set_arg(&l.up.fmt.id())
                .set_arg(&l.down.fmt.id())
                .set_arg(&(l.q.total as u64))
                .set_arg(&(l.k.total as u64))
                .set_arg(&(l.v.total as u64))
                .set_arg(&(l.o.total as u64))
                .set_arg(&(l.gate.total as u64))
                .set_arg(&(l.up.total as u64))
                .set_arg(&(l.down.total as u64))
                .set_arg(&l.attn_norm)
                .set_arg(&l.ffn_norm)
                .set_arg(l.q_bias.as_ref().unwrap_or(unused))
                .set_arg(l.k_bias.as_ref().unwrap_or(unused))
                .set_arg(l.v_bias.as_ref().unwrap_or(unused))
                .set_arg(l.q_norm.as_ref().unwrap_or(unused))
                .set_arg(l.k_norm.as_ref().unwrap_or(unused))
                .set_arg(&l.cache_k)
                .set_arg(&l.cache_v)
                .set_arg(&self.cos)
                .set_arg(&self.sin)
                .set_arg(&(s.hidden as u32))
                .set_arg(&(s.n_head as u32))
                .set_arg(&(s.n_kv as u32))
                .set_arg(&(s.head_dim as u32))
                .set_arg(&(s.ffn as u32))
                .set_arg(&(s.rope_dim as u32))
                .set_arg(&(self.half as u32))
                .set_arg(&flags)
                .set_arg(&s.eps)
                .set_arg(&s.attn_scale)
                .set_arg(&(pos as u32));
            // The layer's own eight counters are picked by its place in the step.
            e.set_arg(&mega.counters)
                .set_arg(&((li - first) as u32))
                .set_arg(&mega.fail)
                .set_global_work_size(mega.groups * MEGA_THREADS)
                .set_local_work_size(MEGA_THREADS)
                .enqueue_nd_range(&self.queue)?;
        }
        Ok(())
    }

    /// The head of a decode step in one launch.
    fn head_single(&self, cur: &Buffer<f32>) -> Result<()> {
        let mega = self
            .mega
            .as_ref()
            .ok_or_else(|| anyhow!("no single-launch layer"))?;
        let h = self
            .head
            .as_ref()
            .ok_or_else(|| anyhow!("no output head"))?;
        unsafe {
            ExecuteKernel::new(&mega.head)
                .set_arg(cur)
                .set_arg(&h.norm)
                .set_arg(&h.proj.buf)
                .set_arg(&h.logits)
                .set_arg(&h.proj.fmt.id())
                .set_arg(&(h.proj.total as u64))
                .set_arg(&(h.proj.rows as u32))
                .set_arg(&(self.spec.hidden as u32))
                .set_arg(&self.spec.eps)
                .set_global_work_size(mega.groups * MEGA_THREADS)
                .set_local_work_size(MEGA_THREADS)
                .enqueue_nd_range(&self.queue)?;
        }
        Ok(())
    }

    /// The head over the last of `c` rows in `cur`.
    fn head_logits(&self, cur: &Buffer<f32>, c: usize) -> Result<()> {
        let h = self
            .head
            .as_ref()
            .ok_or_else(|| anyhow!("no output head"))?;
        self.rms_norm(cur, &h.norm, &self.xn, 1, (c - 1) * self.spec.hidden, 1)?;
        self.quantize(&self.xn, &self.qn, 1, self.spec.hidden)?;
        let xn = Act {
            f: &self.xn,
            q: &self.qn,
        };
        self.mat_vec(&h.proj, xn, &h.logits, 1, None, None)
    }
}

/// A device-wide barrier among work-groups, the way the persistent kernels of the layers use
/// one: each group counts itself in and spins until all have, so every group must make progress
/// while the others wait. A device that schedules fewer groups at once than were launched hangs
/// it, and one whose memory model does not publish the writes reads stale values.
const GRID_BARRIER_SRC: &str = "
              inline void gsync(volatile __global atomic_uint* ctr, uint target, __global uint* fail) {
                  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
                  if (get_local_id(0) == 0) {
                      atomic_fetch_add_explicit(ctr, 1u, memory_order_acq_rel, memory_scope_device);
                      uint spins = 0;
                      while (atomic_load_explicit(ctr, memory_order_acquire, memory_scope_device) < target) {
                          if (++spins > 300000u) { atomic_fetch_add_explicit((volatile __global atomic_uint*)fail, 1u, memory_order_relaxed, memory_scope_device); break; }
                      }
                  }
                  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
              }
              __kernel void sync_test(volatile __global atomic_uint* ctr, __global uint* data, __global uint* fail, uint rounds) {
                  uint nwg = get_num_groups(0), wg = get_group_id(0);
                  uint t = get_local_id(0);
                  for (uint r = 0; r < rounds; r++) {
                      data[wg * 256 + t] = r + 1;
                      gsync(ctr, (2 * r + 1) * nwg, fail);
                      uint other = data[((wg + 1) % nwg) * 256 + t];
                      if (other != r + 1) atomic_fetch_add_explicit((volatile __global atomic_uint*)(fail + 1), 1u, memory_order_relaxed, memory_scope_device);
                      gsync(ctr, (2 * r + 2) * nwg, fail);
                  }
              }";

/// Whether the layers may run on `device`: not on Mesa's platform, and only where the barrier
/// above holds with the group counts the layers launch. Decided once per device and remembered.
pub(crate) fn runs_layers(device: cl_device_id) -> bool {
    !on_mesa_platform(device) && barrier_probe_holds(device)
}

/// Whether `device` belongs to Mesa's OpenCL platform. Its Intel driver calls itself incomplete,
/// and the layers measured wrong there (a first layer 8 times further from the reference than
/// the host's, a text of one repeated token) while Intel's own runtime gives the host's text.
fn on_mesa_platform(device: cl_device_id) -> bool {
    Device::new(device)
        .platform()
        .ok()
        .and_then(|p| opencl3::platform::Platform::new(p).vendor().ok())
        .is_some_and(|v| v.to_lowercase().contains("mesa"))
}

fn barrier_probe_holds(device: cl_device_id) -> bool {
    static KNOWN: std::sync::OnceLock<std::sync::Mutex<HashMap<usize, bool>>> =
        std::sync::OnceLock::new();
    let known = KNOWN.get_or_init(Default::default);
    if let Some(&held) = known
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&(device as usize))
    {
        return held;
    }
    let held = probe_grid_barrier(device).unwrap_or(false);
    if !held {
        tracing::warn!(
            "an OpenCL device fails the device-wide barrier the layers rely on; it is not used for layers"
        );
    }
    known
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(device as usize, held);
    held
}

fn probe_grid_barrier(device: cl_device_id) -> Result<bool> {
    let dev = Device::new(device);
    let ctx = Context::from_device(&dev)?;
    #[allow(deprecated)]
    let queue = CommandQueue::create_default(&ctx, 0)?;
    let program = Program::create_and_build_from_source(&ctx, GRID_BARRIER_SRC, "-cl-std=CL3.0")
        .map_err(|e| anyhow!("{e}"))?;
    let kernel = Kernel::create(&program, "sync_test")?;
    for (nwg, wgs) in [(32usize, 64usize), (64, 256)] {
        let (mut ctr, data, mut fail) = (
            scratch(&ctx, 4)?,
            scratch(&ctx, nwg * 256)?,
            scratch(&ctx, 4)?,
        );
        let zero = [f32::from_bits(0); 4];
        let rounds = 50u32;
        unsafe {
            queue.enqueue_write_buffer(&mut ctr, CL_BLOCKING, 0, &zero, &[])?;
            queue.enqueue_write_buffer(&mut fail, CL_BLOCKING, 0, &zero, &[])?;
            ExecuteKernel::new(&kernel)
                .set_arg(&ctr)
                .set_arg(&data)
                .set_arg(&fail)
                .set_arg(&rounds)
                .set_global_work_size(nwg * wgs)
                .set_local_work_size(wgs)
                .enqueue_nd_range(&queue)?;
            queue.finish()?;
            let mut out = [0f32; 4];
            queue.enqueue_read_buffer(&fail, CL_BLOCKING, 0, &mut out, &[])?;
            if out[0].to_bits() != 0 || out[1].to_bits() != 0 {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rate the device reads a large buffer at, and the host reads memory at: the bound
    /// on a decode step, which reads each weight once.
    #[test]
    #[ignore = "needs an OpenCL device"]
    fn memory_bandwidth() {
        for (i, id) in super::super::opencl::enumerate_opencl_devices()
            .into_iter()
            .enumerate()
        {
            let dev = Device::new(id);
            let ctx = Context::from_device(&dev).unwrap();
            #[allow(deprecated)]
            let queue = CommandQueue::create_default(&ctx, 0).unwrap();
            let program = Program::create_and_build_from_source(
                &ctx,
                "__kernel void sum(__global const float4* x, __global float* out, uint n) {
                     float4 s = 0;
                     for (uint i = get_global_id(0); i < n; i += get_global_size(0)) s += x[i];
                     out[get_global_id(0)] = s.x + s.y + s.z + s.w;
                 }
                 __kernel void nop(__global float* out) { out[get_global_id(0)] = 1.0f; }",
                "",
            )
            .unwrap();
            let bytes = 512usize << 20;
            let buf = scratch(&ctx, bytes / 4).unwrap();
            let out = scratch(&ctx, 1 << 16).unwrap();
            let sum = Kernel::create(&program, "sum").unwrap();
            let nop = Kernel::create(&program, "nop").unwrap();
            let n = (bytes / 16) as u32;
            let run = |global: usize, local: usize, reps: usize| -> f64 {
                let start = std::time::Instant::now();
                for _ in 0..reps {
                    unsafe {
                        ExecuteKernel::new(&sum)
                            .set_arg(&buf)
                            .set_arg(&out)
                            .set_arg(&n)
                            .set_global_work_size(global)
                            .set_local_work_size(local)
                            .enqueue_nd_range(&queue)
                            .unwrap();
                    }
                }
                queue.finish().unwrap();
                (bytes * reps) as f64 / start.elapsed().as_secs_f64() / 1e9
            };
            run(1 << 14, 64, 2);
            for (g, l) in [(1 << 12, 64), (1 << 14, 64), (1 << 15, 128), (1 << 16, 256)] {
                println!(
                    "device {i} {}: read {g} items x {l}: {:.1} GB/s",
                    dev.name().unwrap_or_default(),
                    run(g, l, 8)
                );
            }
            let start = std::time::Instant::now();
            let launches = 2000;
            for _ in 0..launches {
                unsafe {
                    ExecuteKernel::new(&nop)
                        .set_arg(&out)
                        .set_global_work_size(64)
                        .set_local_work_size(64)
                        .enqueue_nd_range(&queue)
                        .unwrap();
                }
            }
            queue.finish().unwrap();
            println!(
                "device {i}: {:.1} us per queued launch",
                start.elapsed().as_secs_f64() * 1e6 / launches as f64
            );
        }
        let data = vec![1.0f32; 256 << 20];
        for threads in [1usize, 4, 8, 16, 22] {
            let chunk = data.len() / threads;
            let start = std::time::Instant::now();
            let total: f32 = std::thread::scope(|s| {
                (0..threads)
                    .map(|t| {
                        let part = &data[t * chunk..(t + 1) * chunk];
                        s.spawn(move || part.iter().sum::<f32>())
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|h| h.join().unwrap())
                    .sum()
            });
            println!(
                "host: {threads} threads read {:.1} GB/s ({total})",
                (data.len() * 4) as f64 / start.elapsed().as_secs_f64() / 1e9
            );
        }
    }

    /// Quantised random weights of `dtype`.
    fn random_matrix(dtype: GgmlDType, rows: usize, cols: usize) -> (RawTensor, Vec<f32>) {
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        let data: Vec<f32> = (0..rows * cols).map(|_| next()).collect();
        let t = crate::tensor::Tensor::from_vec(data, (rows, cols), &crate::tensor::Device::Cpu)
            .unwrap();
        let (bytes, deq) = match dtype {
            GgmlDType::F32 => (
                t.flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
                t.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            ),
            GgmlDType::F16 => {
                let h: Vec<half::f16> = t
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
                    .iter()
                    .map(|&v| half::f16::from_f32(v))
                    .collect();
                (
                    h.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    h.iter().map(|v| v.to_f32()).collect(),
                )
            }
            _ => {
                let q = crate::tensor::quantized::QTensor::quantize(&t, dtype).unwrap();
                let bytes = q.data().unwrap().into_owned();
                let deq = q
                    .dequantize(&crate::tensor::Device::Cpu)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                (bytes, deq)
            }
        };
        (
            RawTensor {
                dtype,
                rows,
                cols,
                bytes,
            },
            deq,
        )
    }

    /// The activation rows as the host sees them in a dot product with weights of `dtype`:
    /// quantised the way the host quantises them, then read back.
    fn host_quantised(dtype: GgmlDType, x: &[f32], cols: usize) -> Vec<f32> {
        use crate::tensor::quant_cpu::{BlockFormat, BlockQ8K, BlockQ8_0};
        match dtype {
            GgmlDType::F16 | GgmlDType::F32 => x.to_vec(),
            GgmlDType::Q4K | GgmlDType::Q6K => {
                let mut out = vec![0f32; x.len()];
                for (row, o) in x.chunks(cols).zip(out.chunks_mut(cols)) {
                    let mut blocks = vec![BlockQ8K::zeros(); cols / 256];
                    BlockQ8K::quantize(row, &mut blocks);
                    BlockQ8K::dequantize(&blocks, o);
                }
                out
            }
            _ => {
                let mut out = vec![0f32; x.len()];
                for (row, o) in x.chunks(cols).zip(out.chunks_mut(cols)) {
                    let mut blocks = vec![BlockQ8_0::zeros(); cols / 32];
                    BlockQ8_0::quantize(row, &mut blocks);
                    BlockQ8_0::dequantize(&blocks, o);
                }
                out
            }
        }
    }

    /// The device quantiser makes the host's codes and scales.
    #[test]
    #[ignore = "needs an OpenCL device"]
    fn quantiser_matches_the_host() {
        use crate::tensor::quant_cpu::{BlockFormat, BlockQ8K, BlockQ8_0};
        for (i, id) in super::super::opencl::enumerate_opencl_devices()
            .into_iter()
            .enumerate()
        {
            let dev = Device::new(id);
            let ctx = Context::from_device(&dev).unwrap();
            #[allow(deprecated)]
            let queue = CommandQueue::create_default(&ctx, 0).unwrap();
            let kernels = Kernels::build(&ctx, 1).unwrap();
            // Four blocks of 256 and a ragged end: the k-quant half covers the first four.
            let (rows, cols) = (3usize, 1056usize);
            let x: Vec<f32> = (0..rows * cols)
                .map(|k| (((k * 7919 % 1013) as f32) / 1013.0 - 0.5) * (1.0 + (k % 9) as f32))
                .collect();
            let xb = upload_f32(&ctx, &x).unwrap();
            let q = Quant::new(&ctx, cols).unwrap();
            launch_quant(&queue, &kernels, &xb, &q, rows, cols).unwrap();
            queue.finish().unwrap();
            let mut qz = vec![0u8; rows * cols];
            let mut qk = vec![0u8; rows * cols];
            let mut dz = vec![0f32; rows * cols / 32];
            let mut dk = vec![0f32; rows * (cols / 256)];
            let mut sk = vec![0f32; rows * cols / 32];
            unsafe {
                queue
                    .enqueue_read_buffer(&q.qz, CL_BLOCKING, 0, &mut qz, &[])
                    .unwrap();
                queue
                    .enqueue_read_buffer(&q.qk, CL_BLOCKING, 0, &mut qk, &[])
                    .unwrap();
                queue
                    .enqueue_read_buffer(&q.dz, CL_BLOCKING, 0, &mut dz, &[])
                    .unwrap();
                queue
                    .enqueue_read_buffer(&q.dk, CL_BLOCKING, 0, &mut dk, &[])
                    .unwrap();
                queue
                    .enqueue_read_buffer(&q.sk, CL_BLOCKING, 0, &mut sk, &[])
                    .unwrap();
            }
            let (mut flips, mut total) = (0, 0);
            for r in 0..rows {
                let row = &x[r * cols..(r + 1) * cols];
                let mut z = vec![BlockQ8_0::zeros(); cols / 32];
                BlockQ8_0::quantize(row, &mut z);
                for (b, blk) in z.iter().enumerate() {
                    assert_eq!(
                        blk.d.to_f32(),
                        dz[r * (cols / 32) + b],
                        "q8_0 scale {r}/{b}"
                    );
                    for (j, &c) in blk.qs.iter().enumerate() {
                        total += 1;
                        flips += (qz[r * cols + b * 32 + j] as i8 != c) as usize;
                    }
                }
                let full = cols / 256;
                let mut kb = vec![BlockQ8K::zeros(); full];
                BlockQ8K::quantize(&row[..full * 256], &mut kb);
                for (b, blk) in kb.iter().enumerate() {
                    let got = dk[r * full + b];
                    assert!(
                        (blk.d - got).abs() <= blk.d.abs() * 4e-7,
                        "q8_K scale {r}/{b}: {} against {got}",
                        blk.d
                    );
                    for (j, &c) in blk.qs.iter().enumerate() {
                        total += 1;
                        flips += (qk[r * cols + b * 256 + j] as i8 != c) as usize;
                    }
                    for s in 0..8 {
                        let want: i32 =
                            blk.qs[s * 32..(s + 1) * 32].iter().map(|&c| c as i32).sum();
                        assert_eq!(
                            want,
                            sk[r * (cols / 32) + b * 8 + s].to_bits() as i32,
                            "q8_K sum {r}/{b}/{s}"
                        );
                    }
                }
            }
            println!("device {i}: {flips} of {total} codes differ from the host's");
            assert!(flips * 1000 <= total, "{flips} of {total}");
        }
    }

    const FORMATS: [GgmlDType; 7] = [
        GgmlDType::Q8_0,
        GgmlDType::Q4_0,
        GgmlDType::Q5_0,
        GgmlDType::F16,
        GgmlDType::F32,
        GgmlDType::Q4K,
        GgmlDType::Q6K,
    ];

    /// Every format's mat-vec, narrow and wide, against the dequantised weights on the host.
    #[test]
    #[ignore = "needs an OpenCL device"]
    fn mat_vec_matches_the_dequantised_weights() {
        for (i, id) in super::super::opencl::enumerate_opencl_devices()
            .into_iter()
            .enumerate()
        {
            let dev = Device::new(id);
            let ctx = Context::from_device(&dev).unwrap();
            #[allow(deprecated)]
            let queue = CommandQueue::create_default(&ctx, 0).unwrap();
            let kernels = [
                Kernels::build(&ctx, 1).unwrap(),
                Kernels::wide(&ctx).unwrap(),
            ];
            let unused = scratch(&ctx, 4).unwrap();
            for dtype in FORMATS {
                let (rows, cols) = (70, 768);
                let (raw, deq) = random_matrix(dtype, rows, cols);
                let m = Matrix::upload(&ctx, &raw, usize::MAX).unwrap();
                for nr in [1, 3, CHUNK] {
                    let x: Vec<f32> = (0..nr * cols)
                        .map(|k| (((k * 31 % 17) as f32 - 8.0) / 8.0) * (1.0 + (k % 5) as f32))
                        .collect();
                    let xb = upload_f32(&ctx, &x).unwrap();
                    let qx = Quant::new(&ctx, cols).unwrap();
                    let out = scratch(&ctx, nr * rows).unwrap();
                    let k = &kernels[(nr > 1) as usize];
                    launch_quant(&queue, k, &xb, &qx, nr, cols).unwrap();
                    let act = Act { f: &xb, q: &qx };
                    launch_mat_vec(&queue, k, &unused, &m, act, &out, nr, None, None).unwrap();
                    let mut got = vec![0f32; nr * rows];
                    unsafe {
                        queue
                            .enqueue_read_buffer(&out, CL_BLOCKING, 0, &mut got, &[])
                            .unwrap();
                    }
                    let seen = host_quantised(dtype, &x, cols);
                    let mut worst = 0f32;
                    for r in 0..nr {
                        for o in 0..rows {
                            let want: f32 = (0..cols)
                                .map(|c| deq[o * cols + c] * seen[r * cols + c])
                                .sum();
                            worst = worst.max((got[r * rows + o] - want).abs());
                        }
                    }
                    println!(
                        "device {i} {} {nr} rows: worst error {worst:.2e}",
                        dtype.gguf_name()
                    );
                    assert!(worst < 5e-3, "{}: {worst}", dtype.gguf_name());
                }
            }
        }
    }

    /// Time per launch and rate of every format's mat-vec, and of the gate-and-up kernel, on
    /// the sizes the small models use. A `dense_stack.cl` in the working directory replaces
    /// the built-in source, to try a kernel without rebuilding.
    #[test]
    #[ignore = "needs an OpenCL device"]
    fn mat_vec_rates() {
        for (i, id) in super::super::opencl::enumerate_opencl_devices()
            .into_iter()
            .enumerate()
        {
            let dev = Device::new(id);
            let ctx = Context::from_device(&dev).unwrap();
            #[allow(deprecated)]
            let queue = CommandQueue::create_default(&ctx, 0).unwrap();
            let kernels = Kernels::build(&ctx, 1).unwrap();
            let unused = scratch(&ctx, 4).unwrap();
            let cases = [
                (GgmlDType::Q4K, 8, 1024),
                (GgmlDType::Q8_0, 8, 1024),
                (GgmlDType::F32, 8, 1024),
                (GgmlDType::Q4K, 1024, 1024),
                (GgmlDType::Q4K, 2048, 1024),
                (GgmlDType::Q4K, 3072, 1024),
                (GgmlDType::Q4K, 1024, 3072),
                (GgmlDType::Q4K, 32768, 1024),
                (GgmlDType::Q6K, 1024, 3072),
                (GgmlDType::Q6K, 151936, 1024),
                (GgmlDType::Q8_0, 4864, 896),
                (GgmlDType::F16, 1024, 1024),
                (GgmlDType::F16, 4096, 1024),
                (GgmlDType::F32, 4096, 1024),
                (GgmlDType::Q8_0, 4096, 1024),
            ];
            for (dtype, rows, cols) in cases {
                let (raw, _) = random_matrix(dtype, rows.min(8192), cols);
                let mut raw = raw;
                if rows > 8192 {
                    // Rows past the sample repeat it: the rate does not depend on the values.
                    let reps = rows / 8192;
                    let one = raw.bytes.clone();
                    raw.bytes = one.iter().cycle().take(one.len() * reps).copied().collect();
                    raw.rows = 8192 * reps;
                }
                let rows = raw.rows;
                let m = Matrix::upload(&ctx, &raw, usize::MAX).unwrap();
                let x: Vec<f32> = (0..cols).map(|k| (k % 7) as f32 - 3.0).collect();
                let xb = upload_f32(&ctx, &x).unwrap();
                let qx = Quant::new(&ctx, cols).unwrap();
                launch_quant(&queue, &kernels, &xb, &qx, 1, cols).unwrap();
                let out = scratch(&ctx, rows).unwrap();
                let go = |reps: usize| {
                    for _ in 0..reps {
                        launch_mat_vec(
                            &queue,
                            &kernels,
                            &unused,
                            &m,
                            Act { f: &xb, q: &qx },
                            &out,
                            1,
                            None,
                            None,
                        )
                        .unwrap();
                    }
                    queue.finish().unwrap();
                };
                // Warm up for long enough that the device has left its idle clock.
                let warm = std::time::Instant::now();
                while warm.elapsed().as_millis() < 400 {
                    go(20);
                }
                let reps = 400;
                let start = std::time::Instant::now();
                go(reps);
                let us = start.elapsed().as_secs_f64() * 1e6 / reps as f64;
                println!(
                    "device {i}: {} {rows}x{cols}: {us:.1} us per launch, {:.1} GB/s",
                    dtype.gguf_name(),
                    raw.bytes.len() as f64 / us / 1e3
                );
            }
        }
    }

    /// What a launch costs on its own, and one that reads what the last one wrote.
    #[test]
    #[ignore = "needs an OpenCL device"]
    fn launch_costs() {
        for (i, id) in super::super::opencl::enumerate_opencl_devices()
            .into_iter()
            .enumerate()
        {
            let dev = Device::new(id);
            let ctx = Context::from_device(&dev).unwrap();
            #[allow(deprecated)]
            let queue = CommandQueue::create_default(&ctx, 0).unwrap();
            let program = Program::create_and_build_from_source(
                &ctx,
                "__kernel void step(__global const float* a, __global float* b, uint n) {
                     uint i = get_global_id(0);
                     if (i < n) b[i] = a[i] + 1.0f;
                 }",
                "",
            )
            .unwrap();
            let kernel = Kernel::create(&program, "step").unwrap();
            let n = 1024u32;
            let bufs: Vec<Buffer<f32>> =
                (0..4).map(|_| scratch(&ctx, n as usize).unwrap()).collect();
            let launches = 3000;
            let run = |dependent: bool, items: usize| -> f64 {
                let start = std::time::Instant::now();
                for l in 0..launches {
                    let (a, b) = if dependent {
                        (&bufs[l % 2], &bufs[(l + 1) % 2])
                    } else {
                        (&bufs[2], &bufs[3])
                    };
                    unsafe {
                        ExecuteKernel::new(&kernel)
                            .set_arg(a)
                            .set_arg(b)
                            .set_arg(&n)
                            .set_global_work_size(items)
                            .set_local_work_size(64)
                            .enqueue_nd_range(&queue)
                            .unwrap();
                    }
                }
                queue.finish().unwrap();
                start.elapsed().as_secs_f64() * 1e6 / launches as f64
            };
            run(true, 1024);
            println!(
                "device {i}: {:.1} us per independent launch, {:.1} us per dependent one; {:.1} us with 64 items",
                run(false, 1024),
                run(true, 1024),
                run(true, 64)
            );
        }
    }

    /// The probe placement relies on says, device by device, whether the barrier holds.
    #[test]
    #[ignore = "needs an OpenCL device"]
    fn the_layers_are_offered_only_where_they_run() {
        for (i, id) in super::super::opencl::enumerate_opencl_devices()
            .into_iter()
            .enumerate()
        {
            println!("device {i}: the layers run here: {}", runs_layers(id));
        }
    }

    /// A device-wide barrier between the workgroups of one launch: its cost, and whether data
    /// written before it is seen after it.
    #[test]
    #[ignore = "needs an OpenCL device"]
    fn grid_barrier() {
        for (i, id) in super::super::opencl::enumerate_opencl_devices()
            .into_iter()
            .enumerate()
        {
            let dev = Device::new(id);
            let ctx = Context::from_device(&dev).unwrap();
            #[allow(deprecated)]
            let queue = CommandQueue::create_default(&ctx, 0).unwrap();
            let src = GRID_BARRIER_SRC;
            let Ok(program) = Program::create_and_build_from_source(&ctx, src, "-cl-std=CL3.0")
            else {
                println!("device {i}: no device-scope atomics");
                continue;
            };
            let kernel = Kernel::create(&program, "sync_test").unwrap();
            for (nwg, wgs) in [
                (16usize, 64usize),
                (32, 64),
                (64, 64),
                (128, 64),
                (32, 256),
                (64, 256),
                (96, 256),
            ] {
                let ctr = scratch(&ctx, 4).unwrap();
                let data = scratch(&ctx, nwg * 256).unwrap();
                let fail = scratch(&ctx, 4).unwrap();
                let zero = [0u32; 4];
                let rounds = 500u32;
                unsafe {
                    let mut c = ctr;
                    let mut f = fail;
                    queue
                        .enqueue_write_buffer(
                            &mut c,
                            CL_BLOCKING,
                            0,
                            &zero.map(|v| f32::from_bits(v)),
                            &[],
                        )
                        .unwrap();
                    queue
                        .enqueue_write_buffer(
                            &mut f,
                            CL_BLOCKING,
                            0,
                            &zero.map(|v| f32::from_bits(v)),
                            &[],
                        )
                        .unwrap();
                    let start = std::time::Instant::now();
                    ExecuteKernel::new(&kernel)
                        .set_arg(&c)
                        .set_arg(&data)
                        .set_arg(&f)
                        .set_arg(&rounds)
                        .set_global_work_size(nwg * wgs)
                        .set_local_work_size(wgs)
                        .enqueue_nd_range(&queue)
                        .unwrap();
                    queue.finish().unwrap();
                    let us = start.elapsed().as_secs_f64() * 1e6 / (2.0 * rounds as f64);
                    let mut out = [0f32; 4];
                    queue
                        .enqueue_read_buffer(&f, CL_BLOCKING, 0, &mut out, &[])
                        .unwrap();
                    println!(
                        "device {i}: {nwg} groups of {wgs}: {us:.1} us per barrier, {} timeouts, {} stale reads",
                        out[0].to_bits(),
                        out[1].to_bits()
                    );
                }
            }
        }
    }
}
