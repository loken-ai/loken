//! The kernels of a batched decode step that are not a GEMM, for a batch of rows and no model.
//! The decode graph of the continuous-batching worker replays the whole layer stack as one
//! graph. Each non-GEMM kernel of one layer is called here through the wrapper the layer loop
//! uses, on synthetic F16 tensors at the real shapes, and timed alone under the host clock
//! around a stream synchronisation.
//!
//! Each kernel is set against the floor of the bytes it must at least move at the card's peak
//! memory bandwidth. The kernels of one layer are then captured in one graph, whose replay gives
//! the in-graph cost the individual medians overstate by their launch and synchronisation.
//!
//! The once-per-step kernels follow: the embedding gather, the first norm, the copy of the
//! hidden state out of the graph, the head matmul and the batched argmax. The input upload
//! between 2 steps is the `decode-inputs` bench.

use super::decode_layout::{batch_width, blocks_per_row};
use super::marlin_rows::{DEFAULT_HEADS, DEFAULT_INTER, DEFAULT_KV_HEADS};
use super::Options;
use crate::tensor::quantized::GgmlDType;

/// The layers of the stack when `--layers` is absent.
pub const DEFAULT_LAYERS: usize = 36;

/// The vocabulary of the head matmul and the argmax when `--vocab` is absent.
pub const DEFAULT_VOCAB: usize = 131072;

/// The format of the embedding table the gather decodes.
const EMBED_DTYPE: GgmlDType = GgmlDType::Q8_0;

/// Bytes of one element of an activation row, F16.
const HALF: u64 = 2;

/// Bytes of one element of a norm weight or of a gathered embedding row, F32.
const FULL: u64 = 4;

/// The shapes of one decode step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerShape {
    /// Rows of the graph: the batch rounded up to its width.
    pub rows: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub context: usize,
    pub block_size: usize,
    pub vocab: usize,
}

impl LayerShape {
    /// The shapes the options give, with `hidden` from the shapes of the run.
    pub fn from_options(o: &Options, hidden: usize) -> Result<Self, String> {
        let shape = Self {
            rows: batch_width(o.batch),
            hidden,
            heads: o.heads.unwrap_or(DEFAULT_HEADS),
            kv_heads: o.kv_heads.unwrap_or(DEFAULT_KV_HEADS),
            head_dim: o.rope_head_dim(),
            ffn: o.inter.unwrap_or(DEFAULT_INTER),
            context: o.context,
            block_size: o.block_size,
            vocab: o.vocab,
        };
        let sizes = [
            o.batch,
            hidden,
            shape.heads,
            shape.kv_heads,
            shape.head_dim,
            shape.ffn,
            shape.context,
            shape.block_size,
            shape.vocab,
            o.layers,
        ];
        if sizes.contains(&0) {
            return Err(
                "--batch, --hidden, --heads, --kv-heads, --head-dim, --inter, --context, --block-size, --vocab and --layers must be above 0"
                    .to_string(),
            );
        }
        if !shape.heads.is_multiple_of(shape.kv_heads) {
            return Err(format!(
                "heads {} must be a multiple of the kv heads {}",
                shape.heads, shape.kv_heads
            ));
        }
        if !shape.head_dim.is_multiple_of(2) {
            return Err("--head-dim must be even".to_string());
        }
        if !hidden.is_multiple_of(EMBED_DTYPE.block_size()) {
            return Err(format!(
                "--hidden must be a multiple of {}, the block of the embedding table",
                EMBED_DTYPE.block_size()
            ));
        }
        Ok(shape)
    }

    /// Elements of the query rows of a token.
    pub fn q_width(&self) -> usize {
        self.heads * self.head_dim
    }

    /// Elements of the key or value row of a token.
    pub fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// Blocks of a row's block table.
    pub fn blocks(&self) -> usize {
        blocks_per_row(self.context, self.block_size)
    }
}

/// Bytes of one row of the embedding table.
pub fn embed_row_bytes(hidden: usize) -> u64 {
    (hidden / EMBED_DTYPE.block_size() * EMBED_DTYPE.type_size()) as u64
}

/// What makes a kernel part of a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    /// Every layer runs it.
    Always,
    /// Only a layer with a norm on the heads of q and k runs it.
    QkNorm,
    /// Only a layer whose projections are fused runs it: a copy cuts a part out of the fused
    /// output.
    FusedProjection,
}

/// One non-GEMM kernel of the decode step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    QCopy,
    KCopy,
    VCopy,
    QNorm,
    KNorm,
    RopeK,
    RopeQ,
    KvWrite,
    Attention,
    AddNormAttention,
    GateCopy,
    UpCopy,
    SiluMul,
    AddNormFfn,
    EmbedGather,
    EmbedCast,
    FirstNorm,
    HiddenCopy,
    LmHead,
    Argmax,
}

/// The kernels of one layer, in the order the layer loop launches them.
pub const LAYER_KERNELS: [Kernel; 14] = [
    Kernel::QCopy,
    Kernel::KCopy,
    Kernel::VCopy,
    Kernel::QNorm,
    Kernel::KNorm,
    Kernel::RopeK,
    Kernel::RopeQ,
    Kernel::KvWrite,
    Kernel::Attention,
    Kernel::AddNormAttention,
    Kernel::GateCopy,
    Kernel::UpCopy,
    Kernel::SiluMul,
    Kernel::AddNormFfn,
];

/// The kernels that run once a step, in the order of the step.
pub const STEP_KERNELS: [Kernel; 6] = [
    Kernel::EmbedGather,
    Kernel::EmbedCast,
    Kernel::FirstNorm,
    Kernel::HiddenCopy,
    Kernel::LmHead,
    Kernel::Argmax,
];

/// The bytes of a copy that cuts `width` elements of each of `rows` rows out of a fused row.
/// A single row is a view and copies nothing.
fn cut_bytes(rows: u64, width: u64) -> u64 {
    if rows > 1 {
        2 * rows * width * HALF
    } else {
        0
    }
}

impl Kernel {
    /// The name the table and the record carry.
    pub fn name(self) -> &'static str {
        match self {
            Self::QCopy => "q copy out of the fused qkv (narrow)",
            Self::KCopy => "k copy out of the fused qkv (narrow)",
            Self::VCopy => "v copy out of the fused qkv (narrow)",
            Self::QNorm => "q head norm (fused_rmsnorm_f16)",
            Self::KNorm => "k head norm (fused_rmsnorm_f16)",
            Self::RopeK => "k rope (rope_apply)",
            Self::RopeQ => "q rope (rope_apply)",
            Self::KvWrite => "k and v write to the paged store (paged_kv_write)",
            Self::Attention => "paged flash decode (paged_flash_decode)",
            Self::AddNormAttention => "attention output add and ffn norm (fused_add_rmsnorm_f16)",
            Self::GateCopy => "gate copy out of the fused gate and up (narrow)",
            Self::UpCopy => "up copy out of the fused gate and up (narrow)",
            Self::SiluMul => "silu mul (silu_mul_f16)",
            Self::AddNormFfn => "ffn output add and next layer norm (fused_add_rmsnorm_f16)",
            Self::EmbedGather => "embedding gather (embed_rows_gguf)",
            Self::EmbedCast => "embedding cast to F16 (to_dtype)",
            Self::FirstNorm => "first layer norm (fused_rmsnorm_f16)",
            Self::HiddenCopy => "hidden state copy out of the graph (slice_set)",
            Self::LmHead => "head matmul (matmul_t)",
            Self::Argmax => "batched argmax (batched_argmax)",
        }
    }

    /// Whether every layer runs it, in place of once a step.
    pub fn per_layer(self) -> bool {
        LAYER_KERNELS.contains(&self)
    }

    /// Calls in a step of `layers` layers.
    pub fn calls(self, layers: usize) -> usize {
        if self.per_layer() {
            layers
        } else {
            1
        }
    }

    /// Which layers run it.
    pub fn group(self) -> Group {
        match self {
            Self::QNorm | Self::KNorm => Group::QkNorm,
            Self::QCopy | Self::KCopy | Self::VCopy | Self::GateCopy | Self::UpCopy => {
                Group::FusedProjection
            }
            _ => Group::Always,
        }
    }

    /// The bytes the kernel must at least read and write at `s`: each input and output once.
    pub fn bytes(self, s: &LayerShape) -> u64 {
        let (b, hidden) = (s.rows as u64, s.hidden as u64);
        let (hd, ffn) = (s.head_dim as u64, s.ffn as u64);
        let (qd, kvd) = (s.q_width() as u64, s.kv_width() as u64);
        let norm = |width: u64, weight: u64| 2 * b * width * HALF + weight * FULL;
        match self {
            Self::QCopy => cut_bytes(b, qd),
            Self::KCopy | Self::VCopy => cut_bytes(b, kvd),
            Self::QNorm => norm(qd, hd),
            Self::KNorm => norm(kvd, hd),
            Self::RopeK => 2 * b * kvd * HALF + 2 * b * hd,
            Self::RopeQ => 2 * b * qd * HALF + 2 * b * hd,
            Self::KvWrite => 4 * b * kvd * HALF + b * 4,
            Self::Attention => {
                let kv = 2 * b * s.context as u64 * kvd * HALF;
                let q_and_out = 2 * b * qd * HALF;
                let index = b * (s.blocks() as u64 + 1) * 4;
                kv + q_and_out + index
            }
            Self::AddNormAttention | Self::AddNormFfn => 4 * b * hidden * HALF + hidden * FULL,
            Self::GateCopy | Self::UpCopy => cut_bytes(b, ffn),
            Self::SiluMul => 3 * b * ffn * HALF,
            Self::EmbedGather => b * embed_row_bytes(s.hidden) + b * 4 + b * hidden * FULL,
            Self::EmbedCast => b * hidden * (FULL + HALF),
            Self::FirstNorm => norm(hidden, hidden),
            Self::HiddenCopy => 2 * b * hidden * HALF,
            Self::LmHead => {
                let vocab = s.vocab as u64;
                vocab * hidden * HALF + b * hidden * HALF + b * vocab * HALF
            }
            Self::Argmax => b * s.vocab as u64 * HALF + b * 4,
        }
    }
}

/// The peak memory bandwidth in GB/s from the driver's memory clock in kHz and bus width in
/// bits: a double-rate clock over the width. 0 when either is unknown.
pub fn peak_gbs(mem_clock_khz: i64, bus_bits: i64) -> f64 {
    if mem_clock_khz <= 0 || bus_bits <= 0 {
        return 0.0;
    }
    mem_clock_khz as f64 * 1e3 * (bus_bits as f64 / 8.0) * 2.0 / 1e9
}

/// The time in microseconds that `bytes` take at `peak` GB/s. `None` without bytes or a peak.
pub fn floor_us(bytes: u64, peak: f64) -> Option<f64> {
    (bytes > 0 && peak > 0.0).then(|| bytes as f64 / (peak * 1e3))
}

/// One kernel's measure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Entry {
    pub kernel: Kernel,
    pub median_us: f64,
    pub bytes: u64,
    pub floor_us: Option<f64>,
}

impl Entry {
    /// The median over the floor, when there is a floor.
    pub fn ratio(&self) -> Option<f64> {
        self.floor_us
            .filter(|f| *f > 0.0)
            .map(|f| self.median_us / f)
    }

    /// Microseconds a call costs above its floor.
    pub fn excess_us(&self) -> f64 {
        self.median_us - self.floor_us.unwrap_or(0.0)
    }

    /// Milliseconds a step of `layers` layers loses above the floor of this kernel.
    pub fn excess_ms(&self, layers: usize) -> f64 {
        self.excess_us() * self.kernel.calls(layers) as f64 / 1e3
    }

    /// The same with the cost of an empty launch taken from the median first.
    pub fn net_excess_ms(&self, layers: usize, launch_us: f64) -> f64 {
        let net = (self.median_us - launch_us).max(0.0) - self.floor_us.unwrap_or(0.0);
        net.max(0.0) * self.kernel.calls(layers) as f64 / 1e3
    }
}

/// The indices of `entries` from the largest excess over the floor in a step to the smallest.
pub fn ranking(entries: &[Entry], layers: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.sort_by(|&a, &b| {
        entries[b]
            .excess_ms(layers)
            .total_cmp(&entries[a].excess_ms(layers))
    });
    order
}

/// The sums of the medians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Totals {
    /// The kernels of one layer, in microseconds.
    pub layer_us: f64,
    /// Of those, the norms on the heads of q and k.
    pub qk_norm_us: f64,
    /// Of those, the copies that cut the fused projections.
    pub fused_us: f64,
    /// The layer sum over all the layers, in milliseconds.
    pub layers_ms: f64,
    /// The once-per-step kernels, in milliseconds.
    pub once_ms: f64,
    /// The non-GEMM time of a step, in milliseconds.
    pub step_ms: f64,
}

/// The sums of the medians of `entries` for a step of `layers` layers.
pub fn totals(entries: &[Entry], layers: usize) -> Totals {
    let layer = |keep: &dyn Fn(Group) -> bool| -> f64 {
        entries
            .iter()
            .filter(|e| e.kernel.per_layer() && keep(e.kernel.group()))
            .map(|e| e.median_us)
            .sum()
    };
    let layer_us = layer(&|_| true);
    let once_ms: f64 = entries
        .iter()
        .filter(|e| !e.kernel.per_layer())
        .map(|e| e.median_us)
        .sum::<f64>()
        / 1e3;
    let layers_ms = layer_us * layers as f64 / 1e3;
    Totals {
        layer_us,
        qk_norm_us: layer(&|g| g == Group::QkNorm),
        fused_us: layer(&|g| g == Group::FusedProjection),
        layers_ms,
        once_ms,
        step_ms: layers_ms + once_ms,
    }
}

/// The note of a kernel's row: its floor, its ratio to it and its share of the peak.
pub fn row_note(e: &Entry, peak: f64) -> String {
    match (e.floor_us, e.ratio()) {
        (Some(floor), Some(ratio)) => format!(
            "floor {floor:.1} us, {ratio:.2}x the floor, {:.0}% of the peak {peak:.0} GB/s",
            100.0 / ratio
        ),
        _ if e.bytes == 0 => "moves no bytes at this shape".to_string(),
        _ => "no floor: the card reports no memory bandwidth".to_string(),
    }
}

/// The ranking table: the kernels by what a step loses above their floor.
pub fn ranking_table(entries: &[Entry], layers: usize, launch_us: f64) -> String {
    let mut out = format!(
        "{:>4}  {:<56}  {:>6}  {:>10}  {:>9}  {:>8}  {:>10}  {:>14}\n",
        "rank", "kernel", "calls", "median us", "floor us", "x floor", "excess ms", "net of launch"
    );
    for (rank, &i) in ranking(entries, layers).iter().enumerate() {
        let e = &entries[i];
        let floor = e.floor_us.map_or("-".to_string(), |f| format!("{f:.1}"));
        let ratio = e.ratio().map_or("-".to_string(), |r| format!("{r:.2}"));
        out.push_str(&format!(
            "{:>4}  {:<56}  {:>6}  {:>10.1}  {:>9}  {:>8}  {:>10.3}  {:>14.3}\n",
            rank + 1,
            e.kernel.name(),
            e.kernel.calls(layers),
            e.median_us,
            floor,
            ratio,
            e.excess_ms(layers),
            e.net_excess_ms(layers, launch_us)
        ));
    }
    out
}

/// The final lines: the layer and the step totals by the individual medians, and by the graph
/// replay of a layer when it ran.
pub fn totals_lines(t: &Totals, layers: usize, graph_layer_us: Option<f64>) -> String {
    let mut out = format!(
        "one layer, sum of the medians: {:.1} us, of which {:.1} us q and k head norms and {:.1} us fused projection copies\n\
         {layers} layers: {:.3} ms a step\n\
         once a step: {:.3} ms\n\
         non-GEMM time of a step, individual medians: {:.3} ms\n",
        t.layer_us, t.qk_norm_us, t.fused_us, t.layers_ms, t.once_ms, t.step_ms
    );
    if let Some(g) = graph_layer_us {
        let layers_ms = g * layers as f64 / 1e3;
        out.push_str(&format!(
            "one layer, graph replay: {g:.1} us, {:.1} us under the sum of the medians\n\
             {layers} layers: {layers_ms:.3} ms a step\n\
             non-GEMM time of a step, graph replay of the layers: {:.3} ms\n",
            t.layer_us - g,
            layers_ms + t.once_ms
        ));
    }
    out
}

#[cfg(feature = "gpu")]
mod device {
    use super::*;
    use crate::inference::cache::paged_attention::{rope_apply, rope_cos_sin};
    use crate::inference::cache::paged_kv::PagedKvStore;
    use crate::inference::kernel::fused::{
        fused_add_rmsnorm_f16, fused_rmsnorm_f16, paged_flash_decode, paged_kv_write, silu_mul_f16,
    };
    use crate::inference::moe_cuda::{batched_argmax, embed_rows_gguf};
    use crate::microbench::card::Spin;
    use crate::microbench::decode_layout::padded_block_table;
    use crate::microbench::marlin_rows::copies_for;
    use crate::microbench::stats::{run_for, timed, Summary};
    use crate::microbench::{Context, Fail, Measured};
    use crate::tensor::cuda_ext;
    use crate::tensor::gpu_runtime::driver::{sys, CudaStream};
    use crate::tensor::quantized::QTensor;
    use crate::tensor::{DType, Device, Tensor};
    use std::sync::Arc;
    use std::time::Duration;

    /// The time of each call of `work`, followed by a stream synchronisation, in microseconds.
    fn time_calls(
        target: Duration,
        stream: &Arc<CudaStream>,
        fail: &Fail,
        work: &mut dyn FnMut() -> crate::tensor::Result<()>,
    ) -> Vec<f64> {
        run_for(
            target,
            timed(|| {
                fail.ok(work());
                fail.ok(stream.synchronize());
            }),
        )
    }

    pub fn run(cx: &Context) -> Result<Vec<Measured>, String> {
        let card = cx.card.ok_or("decode-layer needs a card")?;
        let o = cx.options;
        let shape = LayerShape::from_options(o, cx.shapes.hidden)?;
        let LayerShape {
            rows: b,
            hidden,
            heads: nh,
            kv_heads: nkv,
            head_dim: hd,
            ffn,
            context: ctx,
            block_size: bs,
            vocab,
        } = shape;
        let (qd, kvd) = (shape.q_width(), shape.kv_width());
        let max_blocks = shape.blocks();
        let layers = o.layers;
        let dev = Device::Cuda(card.dev.clone());
        let stream = card.dev.stream().clone();
        let target = Duration::from_secs_f64(o.seconds);
        let tensor = |e: crate::tensor::Error| format!("{e:?}");
        let fail = Fail::default();

        let attribute = |a: sys::CUdevice_attribute| {
            card.dev
                .context()
                .attribute(a)
                .map(i64::from)
                .map_err(|e| format!("card attribute {a:?}: {e:?}"))
        };
        let peak = peak_gbs(
            attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)?,
            attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)?,
        );
        let l2 =
            attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)?.max(0) as usize;

        // Synthetic tensors at the real shapes. Contiguous ones feed the kernels alone, and the
        // fused outputs feed the copies that cut them.
        let half = |dims: &[usize]| -> Result<Tensor, String> {
            Tensor::randn(0.0, 1.0, dims, &dev)
                .and_then(|t| t.to_dtype(DType::F16))
                .map_err(tensor)
        };
        let weight = |len: usize| -> Result<Tensor, String> {
            Tensor::randn(0.0, 1.0, &[len][..], &dev).map_err(tensor)
        };
        let qkv = half(&[b, qd + 2 * kvd])?;
        let gu = half(&[b, 2 * ffn])?;
        let (gate, up) = (half(&[b, ffn])?, half(&[b, ffn])?);
        let (q_heads, k_heads) = (half(&[b, nh, hd])?, half(&[b, nkv, hd])?);
        let (k_rot, v_new) = (half(&[b, kvd])?, half(&[b, kvd])?);
        let (attn_out, ffn_out, residual, x) = (
            half(&[b, hidden])?,
            half(&[b, hidden])?,
            half(&[b, hidden])?,
            half(&[b, hidden])?,
        );
        let (w_hidden, w_head) = (weight(hidden)?, weight(hd)?);
        let eps = f32::EPSILON;
        let scale = 1.0 / (hd as f32).sqrt();

        // Every row decodes its last token at the full context.
        let last = ctx - 1;
        let positions = vec![last; b];
        let (cos, sin) =
            rope_cos_sin(&positions, hd, o.rope_base, &dev, DType::F16).map_err(tensor)?;
        let tables: Vec<Vec<u32>> = (0..b)
            .map(|row| {
                (0..max_blocks)
                    .map(|j| (row * max_blocks + j) as u32)
                    .collect()
            })
            .collect();
        let slots: Vec<i32> = tables
            .iter()
            .map(|t| (t[last / bs] as usize * bs + last % bs) as i32)
            .collect();
        let slot_dev = Tensor::from_vec(slots, &[b][..], &dev).map_err(tensor)?;
        let seq_lens = Tensor::from_vec(vec![ctx as i32; b], &[b][..], &dev).map_err(tensor)?;
        let block_table = Tensor::from_vec(
            padded_block_table(&tables, max_blocks),
            &[b, max_blocks][..],
            &dev,
        )
        .map_err(tensor)?;

        // Copies of the paged store, together larger than twice the second-level cache, are
        // read in turn so that the attention streams the cache from card memory.
        let store_bytes = 2 * b * max_blocks * bs * kvd * HALF as usize;
        let stores = (0..copies_for(l2, store_bytes))
            .map(|_| {
                PagedKvStore::new(b * max_blocks, bs, nkv, hd, DType::F16, &dev).map_err(tensor)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // The once-per-step inputs.
        let ids =
            Tensor::from_vec((0..b as u32).collect::<Vec<_>>(), &[b][..], &dev).map_err(tensor)?;
        let table = Tensor::randn(0.0, 1.0, &[b, hidden][..], &Device::Cpu)
            // CONVERSION: synthetic embedding table of a microbench, no stored weight.
            .and_then(|t| QTensor::quantize_onto(&t, EMBED_DTYPE, &dev))
            .map_err(tensor)?;
        let gathered = Tensor::randn(0.0, 1.0, &[b, hidden][..], &dev).map_err(tensor)?;
        let hidden_buf = Tensor::zeros_on(&[b, hidden][..], DType::F16, &dev).map_err(tensor)?;
        let head = Tensor::zeros_on(&[vocab, hidden][..], DType::F16, &dev).map_err(tensor)?;
        let logits = x.matmul_t(&head).map_err(tensor)?;

        let declined = |what: &str| crate::tensor::Error::msg(format!("{what} declined"));
        let attention = |store: &PagedKvStore| -> crate::tensor::Result<()> {
            let out = paged_flash_decode(
                &q_heads,
                store.k(),
                store.v(),
                &block_table,
                &seq_lens,
                scale,
                b,
                nh,
                nkv,
                hd,
                bs,
                max_blocks,
            )?;
            std::hint::black_box(out.ok_or_else(|| declined("paged_flash_decode"))?);
            Ok(())
        };
        let mut turn = 0usize;
        let mut call = |k: Kernel| -> crate::tensor::Result<()> {
            match k {
                Kernel::QCopy => drop(std::hint::black_box(qkv.narrow(1, 0, qd)?)),
                Kernel::KCopy => drop(std::hint::black_box(qkv.narrow(1, qd, kvd)?)),
                Kernel::VCopy => drop(std::hint::black_box(qkv.narrow(1, qd + kvd, kvd)?)),
                Kernel::QNorm => drop(std::hint::black_box(fused_rmsnorm_f16(
                    &q_heads, &w_head, eps,
                )?)),
                Kernel::KNorm => drop(std::hint::black_box(fused_rmsnorm_f16(
                    &k_heads, &w_head, eps,
                )?)),
                Kernel::RopeK => drop(std::hint::black_box(rope_apply(
                    &k_heads, &cos, &sin, false,
                )?)),
                Kernel::RopeQ => drop(std::hint::black_box(rope_apply(
                    &q_heads, &cos, &sin, false,
                )?)),
                Kernel::KvWrite => paged_kv_write(
                    &k_rot,
                    &v_new,
                    stores[0].k(),
                    stores[0].v(),
                    &slot_dev,
                    b,
                    kvd,
                )?,
                Kernel::Attention => {
                    turn += 1;
                    attention(&stores[turn % stores.len()])?
                }
                Kernel::AddNormAttention => drop(std::hint::black_box(fused_add_rmsnorm_f16(
                    &attn_out, &residual, &w_hidden, eps,
                )?)),
                Kernel::GateCopy => drop(std::hint::black_box(gu.narrow(1, 0, ffn)?)),
                Kernel::UpCopy => drop(std::hint::black_box(gu.narrow(1, ffn, ffn)?)),
                Kernel::SiluMul => drop(std::hint::black_box(
                    silu_mul_f16(&gate, &up)?.ok_or_else(|| declined("silu_mul_f16"))?,
                )),
                Kernel::AddNormFfn => drop(std::hint::black_box(fused_add_rmsnorm_f16(
                    &ffn_out, &residual, &w_hidden, eps,
                )?)),
                Kernel::EmbedGather => {
                    // Safety: the gather writes every element of the buffer.
                    let out = unsafe { Tensor::empty((1, b, hidden), DType::F32, &dev)? };
                    embed_rows_gguf(&table, &ids, &out)?;
                    drop(std::hint::black_box(out));
                }
                Kernel::EmbedCast => drop(std::hint::black_box(gathered.to_dtype(DType::F16)?)),
                Kernel::FirstNorm => {
                    drop(std::hint::black_box(fused_rmsnorm_f16(&x, &w_hidden, eps)?))
                }
                Kernel::HiddenCopy => hidden_buf.slice_set(&x, 0, 0)?,
                Kernel::LmHead => drop(std::hint::black_box(x.matmul_t(&head)?)),
                Kernel::Argmax => drop(std::hint::black_box(batched_argmax(&logits)?)),
            }
            Ok(())
        };

        // One layer's kernels in the order of the layer loop, on the store `store`.
        let layer_once = |store: &PagedKvStore| -> crate::tensor::Result<()> {
            let (q, k, v) = (
                qkv.narrow(1, 0, qd)?,
                qkv.narrow(1, qd, kvd)?,
                qkv.narrow(1, qd + kvd, kvd)?,
            );
            let qh = fused_rmsnorm_f16(&q.reshape(&[b, nh, hd][..])?.contiguous()?, &w_head, eps)?;
            let kh = fused_rmsnorm_f16(&k.reshape(&[b, nkv, hd][..])?.contiguous()?, &w_head, eps)?;
            let k_new = rope_apply(&kh, &cos, &sin, false)?.reshape(&[b, kvd][..])?;
            let q_new = rope_apply(&qh, &cos, &sin, false)?;
            paged_kv_write(&k_new, &v, store.k(), store.v(), &slot_dev, b, kvd)?;
            let a = paged_flash_decode(
                &q_new.reshape(&[b, nh, hd][..])?,
                store.k(),
                store.v(),
                &block_table,
                &seq_lens,
                scale,
                b,
                nh,
                nkv,
                hd,
                bs,
                max_blocks,
            )?
            .ok_or_else(|| declined("paged_flash_decode"))?;
            std::hint::black_box(&a);
            let (sum, norm) = fused_add_rmsnorm_f16(&attn_out, &residual, &w_hidden, eps)?;
            std::hint::black_box(&norm);
            let (g, u) = (gu.narrow(1, 0, ffn)?, gu.narrow(1, ffn, ffn)?);
            let act = silu_mul_f16(&g, &u)?.ok_or_else(|| declined("silu_mul_f16"))?;
            std::hint::black_box(&act);
            let next = fused_add_rmsnorm_f16(&ffn_out, &sum, &w_hidden, eps)?;
            std::hint::black_box(&next);
            Ok(())
        };

        let mut measured: Vec<Measured> = Vec::new();
        let mut entries: Vec<Entry> = Vec::new();

        // The cost of a launch that does nothing, timed as every kernel is.
        let spin = Spin::lasting(card, 0.0)?;
        let variant = "floor: empty kernel, launch and synchronise".to_string();
        let samples = time_calls(target, &stream, &fail, &mut || {
            spin.launch(&stream).map_err(crate::tensor::Error::msg)
        });
        fail.check(&variant)?;
        let launch_us = Summary::of(&samples).median_us;
        measured.push(Measured {
            variant,
            bytes: None,
            samples_us: samples,
            note: None,
        });

        for kernel in LAYER_KERNELS.into_iter().chain(STEP_KERNELS) {
            let place = if kernel.per_layer() { "layer" } else { "step" };
            let variant = format!("{place}: {}", kernel.name());
            let samples = time_calls(target, &stream, &fail, &mut || call(kernel));
            fail.check(&variant)?;
            let bytes = kernel.bytes(&shape);
            let entry = Entry {
                kernel,
                median_us: Summary::of(&samples).median_us,
                bytes,
                floor_us: floor_us(bytes, peak),
            };
            measured.push(Measured {
                variant,
                bytes: (bytes > 0).then_some(bytes),
                samples_us: samples,
                note: Some(row_note(&entry, peak)),
            });
            entries.push(entry);
        }

        // One graph of the layer's kernels for each copy of the store, replayed in turn.
        let eager = layer_once(&stores[0]);
        fail.ok(eager);
        fail.ok(stream.synchronize());
        fail.check("graph warm-up")?;
        let mut graphs = Vec::new();
        for store in &stores {
            cuda_ext::begin_capture(&stream).map_err(|e| format!("graph capture: {e:?}"))?;
            let ran = layer_once(store);
            let graph =
                cuda_ext::end_capture(&stream).map_err(|e| format!("graph capture: {e:?}"))?;
            ran.map_err(tensor)?;
            graphs.push(graph.ok_or("graph capture: no graph")?);
        }
        let mut next = 0usize;
        let variant = "graph: the 14 kernels of one layer, 1 replay, 1 synchronise".to_string();
        let samples = run_for(
            target,
            timed(|| {
                fail.ok(graphs[next % graphs.len()].launch());
                fail.ok(stream.synchronize());
                next += 1;
            }),
        );
        fail.check(&variant)?;
        let replay_us = Summary::of(&samples).median_us;
        let sum_us = totals(&entries, layers).layer_us;
        measured.push(Measured {
            variant,
            bytes: None,
            samples_us: samples,
            note: Some(format!(
                "{:.1} us under the sum of the individual medians, {:.1} us of which an empty launch costs",
                sum_us - replay_us,
                launch_us
            )),
        });

        let variant = format!("graph: {layers} replays back to back, 1 synchronise");
        let samples = run_for(
            target,
            timed(|| {
                for i in 0..layers {
                    fail.ok(graphs[i % graphs.len()].launch());
                }
                fail.ok(stream.synchronize());
            }),
        );
        fail.check(&variant)?;
        let chain_us = Summary::of(&samples).median_us / layers as f64;
        measured.push(Measured {
            variant,
            bytes: None,
            samples_us: samples,
            note: Some(format!("{chain_us:.1} us a layer")),
        });

        let t = totals(&entries, layers);
        println!(
            "empty launch and synchronise: {launch_us:.1} us. Peak memory bandwidth: {peak:.0} GB/s. Rows: {b}. Context: {ctx}. Store copies read in turn: {}.\n",
            stores.len()
        );
        print!("{}", ranking_table(&entries, layers, launch_us));
        println!();
        print!("{}", totals_lines(&t, layers, Some(chain_us)));
        println!();
        Ok(measured)
    }
}

#[cfg(feature = "gpu")]
pub use device::run;

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn opts(args: &[&str]) -> Options {
        let mut all = vec!["microbench", "decode-layer"];
        all.extend_from_slice(args);
        Options::parse_from(all)
    }

    fn shape() -> LayerShape {
        LayerShape {
            rows: 64,
            hidden: 4096,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            ffn: 12288,
            context: 512,
            block_size: 16,
            vocab: 131072,
        }
    }

    fn entry(kernel: Kernel, median_us: f64, floor: Option<f64>) -> Entry {
        Entry {
            kernel,
            median_us,
            bytes: 1,
            floor_us: floor,
        }
    }

    #[test]
    fn the_layer_lists_14_kernels_and_the_step_6_in_order() {
        assert_eq!(LAYER_KERNELS.len(), 14);
        assert_eq!(STEP_KERNELS.len(), 6);
        assert!(LAYER_KERNELS.iter().all(|k| k.per_layer()));
        assert!(STEP_KERNELS.iter().all(|k| !k.per_layer()));
        let rope_k = LAYER_KERNELS.iter().position(|k| *k == Kernel::RopeK);
        let write = LAYER_KERNELS.iter().position(|k| *k == Kernel::KvWrite);
        let attention = LAYER_KERNELS.iter().position(|k| *k == Kernel::Attention);
        assert!(rope_k < write && write < attention);
        assert_eq!(Kernel::Attention.calls(36), 36);
        assert_eq!(Kernel::Argmax.calls(36), 1);
    }

    #[test]
    fn the_options_give_the_shapes_and_refuse_the_impossible() {
        let s = LayerShape::from_options(&opts(&[]), 4096).unwrap();
        assert_eq!((s.rows, s.heads, s.kv_heads, s.head_dim), (64, 32, 8, 128));
        assert_eq!((s.ffn, s.context, s.block_size), (12288, 512, 16));
        assert_eq!((s.q_width(), s.kv_width(), s.blocks()), (4096, 1024, 32));
        let s = LayerShape::from_options(&opts(&["--batch", "50"]), 4096).unwrap();
        assert_eq!(s.rows, 64);
        assert!(LayerShape::from_options(&opts(&["--layers", "0"]), 4096).is_err());
        assert!(LayerShape::from_options(&opts(&["--vocab", "0"]), 4096).is_err());
        assert!(LayerShape::from_options(&opts(&["--kv-heads", "5"]), 4096).is_err());
        assert!(LayerShape::from_options(&opts(&["--head-dim", "127"]), 4096).is_err());
        assert!(LayerShape::from_options(&opts(&[]), 4100).is_err());
    }

    #[test]
    fn the_bytes_are_the_inputs_and_outputs_at_the_shape() {
        let s = shape();
        assert_eq!(Kernel::QCopy.bytes(&s), 2 * 64 * 4096 * 2);
        assert_eq!(Kernel::VCopy.bytes(&s), 2 * 64 * 1024 * 2);
        assert_eq!(Kernel::GateCopy.bytes(&s), 2 * 64 * 12288 * 2);
        assert_eq!(Kernel::QNorm.bytes(&s), 2 * 64 * 4096 * 2 + 128 * 4);
        assert_eq!(Kernel::RopeK.bytes(&s), 2 * 64 * 1024 * 2 + 2 * 64 * 128);
        assert_eq!(Kernel::KvWrite.bytes(&s), 4 * 64 * 1024 * 2 + 64 * 4);
        assert_eq!(Kernel::SiluMul.bytes(&s), 3 * 64 * 12288 * 2);
        assert_eq!(Kernel::AddNormFfn.bytes(&s), 4 * 64 * 4096 * 2 + 4096 * 4);
        assert_eq!(Kernel::HiddenCopy.bytes(&s), 2 * 64 * 4096 * 2);
        assert_eq!(Kernel::Argmax.bytes(&s), 64 * 131072 * 2 + 64 * 4);
    }

    #[test]
    fn the_attention_reads_the_context_of_every_row_once() {
        let s = shape();
        let kv = 2 * 64 * 512 * 1024 * 2;
        let rest = 2 * 64 * 4096 * 2 + 64 * (32 + 1) * 4;
        assert_eq!(Kernel::Attention.bytes(&s), kv + rest);
    }

    #[test]
    fn a_single_row_copies_nothing_when_a_fused_output_is_cut() {
        let s = LayerShape { rows: 1, ..shape() };
        assert_eq!(Kernel::QCopy.bytes(&s), 0);
        assert_eq!(Kernel::UpCopy.bytes(&s), 0);
        assert!(Kernel::QNorm.bytes(&s) > 0);
        let empty = Entry {
            bytes: 0,
            ..entry(Kernel::QCopy, 5.0, None)
        };
        assert!(row_note(&empty, 900.0).contains("no bytes"));
    }

    #[test]
    fn the_head_reads_the_vocabulary_matrix_once() {
        let s = shape();
        let bytes = Kernel::LmHead.bytes(&s);
        assert!(bytes > 131072 * 4096 * 2);
        assert_eq!(bytes, 131072 * 4096 * 2 + 64 * 4096 * 2 + 64 * 131072 * 2);
    }

    #[test]
    fn the_embedding_row_is_the_blocks_of_the_table_format() {
        assert_eq!(embed_row_bytes(4096), 4096 / 32 * 34);
        let s = shape();
        assert_eq!(
            Kernel::EmbedGather.bytes(&s),
            64 * embed_row_bytes(4096) + 64 * 4 + 64 * 4096 * 4
        );
    }

    #[test]
    fn the_peak_is_the_double_rate_clock_over_the_bus() {
        assert!((peak_gbs(10_501_000, 384) - 1008.096).abs() < 1e-3);
        assert_eq!(peak_gbs(0, 384), 0.0);
        assert_eq!(peak_gbs(10_501_000, 0), 0.0);
    }

    #[test]
    fn the_floor_is_the_bytes_at_the_peak() {
        assert!((floor_us(1_000_000_000, 1000.0).unwrap() - 1000.0).abs() < 1e-9);
        assert!(floor_us(0, 1000.0).is_none());
        assert!(floor_us(10, 0.0).is_none());
    }

    #[test]
    fn the_excess_counts_a_layer_kernel_for_every_layer() {
        let layer = entry(Kernel::RopeK, 10.0, Some(4.0));
        let once = entry(Kernel::Argmax, 10.0, Some(4.0));
        assert!((layer.excess_ms(36) - 0.216).abs() < 1e-12);
        assert!((once.excess_ms(36) - 0.006).abs() < 1e-12);
        assert!((layer.ratio().unwrap() - 2.5).abs() < 1e-12);
        assert!((layer.net_excess_ms(36, 6.0) - 0.0).abs() < 1e-12);
        assert!((layer.net_excess_ms(36, 2.0) - 0.144).abs() < 1e-12);
        assert!(entry(Kernel::RopeK, 10.0, None).ratio().is_none());
    }

    #[test]
    fn the_ranking_puts_the_largest_loss_of_a_step_first() {
        let entries = [
            entry(Kernel::RopeK, 10.0, Some(9.0)),
            entry(Kernel::Argmax, 500.0, Some(100.0)),
            entry(Kernel::Attention, 50.0, Some(10.0)),
            entry(Kernel::SiluMul, 6.0, Some(1.0)),
        ];
        assert_eq!(ranking(&entries, 36), vec![2, 1, 3, 0]);
        let table = ranking_table(&entries, 36, 5.0);
        let first = table.lines().nth(1).unwrap();
        assert!(first.contains("paged flash decode"));
    }

    #[test]
    fn the_totals_sum_a_layer_then_the_stack_then_the_step() {
        let entries = [
            entry(Kernel::QCopy, 10.0, None),
            entry(Kernel::QNorm, 20.0, None),
            entry(Kernel::Attention, 100.0, None),
            entry(Kernel::EmbedGather, 30.0, None),
            entry(Kernel::Argmax, 70.0, None),
        ];
        let t = totals(&entries, 10);
        assert_eq!(t.layer_us, 130.0);
        assert_eq!(t.qk_norm_us, 20.0);
        assert_eq!(t.fused_us, 10.0);
        assert!((t.layers_ms - 1.3).abs() < 1e-12);
        assert!((t.once_ms - 0.1).abs() < 1e-12);
        assert!((t.step_ms - 1.4).abs() < 1e-12);
        let lines = totals_lines(&t, 10, Some(100.0));
        assert!(lines.contains("1.300 ms a step"));
        assert!(lines.contains("individual medians: 1.400 ms"));
        assert!(lines.contains("30.0 us under the sum"));
        assert!(lines.contains("graph replay of the layers: 1.100 ms"));
        assert!(!totals_lines(&t, 10, None).contains("graph replay"));
    }

    #[test]
    fn the_note_gives_the_floor_the_ratio_and_the_share_of_the_peak() {
        let note = row_note(&entry(Kernel::RopeK, 10.0, Some(5.0)), 1000.0);
        assert_eq!(
            note,
            "floor 5.0 us, 2.00x the floor, 50% of the peak 1000 GB/s"
        );
        let note = row_note(&entry(Kernel::RopeK, 10.0, None), 0.0);
        assert!(note.contains("no floor"));
    }
}
