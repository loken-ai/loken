//! The input upload between two decode steps of the continuous-batching worker, for a batch of
//! rows and no model. The worker builds the per-row arrays on the host, uploads each with
//! `Tensor::from_vec`, copies each into the stable buffer the graph reads with `slice_set`, and
//! synchronises the stream around the replay. The same sequence without the 2 explicit
//! synchronisations, one packed copy from page-locked memory, and the floors of a
//! synchronisation and of a trivial graph replay stand beside it.

use super::card::{device_bytes, Card, Pinned, Spin};
use super::decode_layout::{
    batch_width, blocks_per_row, padded_block_table, source_row, Layout, Section,
};
use super::stats::{run_for, timed, Summary};
use super::{Context, Fail, Measured, Options};
use crate::inference::cache::paged_attention::rope_cos_sin;
use crate::inference::model::rope::inverse_frequencies;
use crate::tensor::cuda_ext;
use crate::tensor::gpu_runtime::driver::CudaEvent;
use crate::tensor::{DType, Device, Tensor};
use std::time::Duration;

/// The variants whose time is compared with the first one's: the upload sequences.
const SEQUENCES: usize = 3;

/// The synthetic rows of a step and the stable device buffers the graph would read.
struct Step {
    device: Device,
    toks: Vec<u32>,
    pos: Vec<usize>,
    slots: Vec<usize>,
    ctx: Vec<usize>,
    tables: Vec<Vec<u32>>,
    width: usize,
    max_blocks: usize,
    head_dim: usize,
    base: f32,
    ids: Tensor,
    cos: Tensor,
    sin: Tensor,
    slot: Tensor,
    seq_lens: Tensor,
    block_table: Tensor,
}

/// The blocks per row the options give: `--blocks-per-row`, else the blocks of `--context`
/// tokens at `--block-size` tokens a block.
fn max_blocks_of(o: &Options) -> usize {
    o.blocks_per_row
        .unwrap_or_else(|| blocks_per_row(o.context, o.block_size))
}

impl Step {
    fn new(card: &Card, o: &Options) -> Result<Self, String> {
        let device = Device::Cuda(card.dev.clone());
        let live = o.batch;
        let width = batch_width(live);
        let max_blocks = max_blocks_of(o);
        let half = o.rope_head_dim() / 2;
        let last = o.context - 1;
        let tables: Vec<Vec<u32>> = (0..live)
            .map(|row| {
                (0..max_blocks)
                    .map(|j| (row * max_blocks + j) as u32)
                    .collect()
            })
            .collect();
        let slots = tables
            .iter()
            .map(|t| {
                let block = t[(last / o.block_size).min(max_blocks - 1)] as usize;
                block * o.block_size + last % o.block_size
            })
            .collect();
        let zeros = |dims: &[usize], dt: DType| {
            Tensor::zeros_on(dims, dt, &device).map_err(|e| format!("stable buffer: {e:?}"))
        };
        Ok(Self {
            toks: (0..live as u32).collect(),
            pos: vec![last; live],
            slots,
            ctx: vec![o.context; live],
            width,
            max_blocks,
            head_dim: o.rope_head_dim(),
            base: o.rope_base,
            ids: zeros(&[width], DType::U32)?,
            cos: zeros(&[width, 1, half], DType::F16)?,
            sin: zeros(&[width, 1, half], DType::F16)?,
            slot: zeros(&[width], DType::I32)?,
            seq_lens: zeros(&[width], DType::I32)?,
            block_table: zeros(&[width, max_blocks], DType::I32)?,
            tables,
            device,
        })
    }

    /// The worker's sequence: pad the rows to the graph's width, upload each array with
    /// `Tensor::from_vec`, then copy each into its stable buffer with `slice_set`.
    fn upload(&self) -> crate::tensor::Result<()> {
        let b = self.width;
        let pad = b - self.toks.len();
        let dev = &self.device;
        let toks: Vec<u32> = self
            .toks
            .iter()
            .copied()
            .chain(std::iter::repeat_n(self.toks[0], pad))
            .collect();
        let pos: Vec<usize> = self
            .pos
            .iter()
            .copied()
            .chain(std::iter::repeat_n(self.pos[0], pad))
            .collect();
        let slots: Vec<usize> = self
            .slots
            .iter()
            .copied()
            .chain(std::iter::repeat_n(self.slots[0], pad))
            .collect();
        let ctx: Vec<usize> = self
            .ctx
            .iter()
            .copied()
            .chain(std::iter::repeat_n(self.ctx[0], pad))
            .collect();
        let tables: Vec<Vec<u32>> = self
            .tables
            .iter()
            .cloned()
            .chain(std::iter::repeat_n(self.tables[0].clone(), pad))
            .collect();
        let ids = Tensor::from_vec(toks, &[b][..], dev)?;
        let (cos, sin) = rope_cos_sin(&pos, self.head_dim, self.base, dev, DType::F16)?;
        let slot = Tensor::from_vec(
            slots.iter().map(|&s| s as i32).collect::<Vec<_>>(),
            &[b][..],
            dev,
        )?;
        let seq_lens = Tensor::from_vec(
            ctx.iter().map(|&c| c as i32).collect::<Vec<_>>(),
            &[b][..],
            dev,
        )?;
        let block_table = Tensor::from_vec(
            padded_block_table(&tables, self.max_blocks),
            &[b, self.max_blocks][..],
            dev,
        )?;
        self.ids.slice_set(&ids, 0, 0)?;
        self.cos.slice_set(&cos, 0, 0)?;
        self.sin.slice_set(&sin, 0, 0)?;
        self.slot.slice_set(&slot, 0, 0)?;
        self.seq_lens.slice_set(&seq_lens, 0, 0)?;
        self.block_table.slice_set(&block_table, 0, 0)?;
        Ok(())
    }

    /// Write every array of the padded step into `buffer` at the sections of `layout`.
    fn pack(&self, layout: &Layout, buffer: &mut [u8]) {
        let half = self.head_dim / 2;
        let inv_freq = inverse_frequencies(self.head_dim, self.base);
        for row in 0..self.width {
            let s = source_row(row, self.toks.len());
            layout.tokens.put(buffer, row, &self.toks[s].to_ne_bytes());
            let slot = self.slots[s] as i32;
            layout.slots.put(buffer, row, &slot.to_ne_bytes());
            let seq_len = self.ctx[s] as i32;
            layout.seq_lens.put(buffer, row, &seq_len.to_ne_bytes());
            for (j, &block) in self.tables[s].iter().take(self.max_blocks).enumerate() {
                let at = row * self.max_blocks + j;
                layout
                    .block_table
                    .put(buffer, at, &(block as i32).to_ne_bytes());
            }
            for (i, f) in inv_freq.iter().enumerate() {
                let (sn, cs) = (self.pos[s] as f32 * f).sin_cos();
                let at = row * half + i;
                layout
                    .cos
                    .put(buffer, at, &half::f16::from_f32(cs).to_ne_bytes());
                layout
                    .sin
                    .put(buffer, at, &half::f16::from_f32(sn).to_ne_bytes());
            }
        }
    }
}

/// The note of each of the first `count` rows: its median over the first row's.
fn note_ratios(rows: &mut [Measured], count: usize) {
    let first = Summary::of(&rows[0].samples_us).median_us;
    for r in rows.iter_mut().take(count) {
        let ratio = Summary::of(&r.samples_us).median_us / first;
        r.note = Some(format!("{ratio:.3} of the first variant's median"));
    }
}

pub fn run(cx: &Context) -> Result<Vec<Measured>, String> {
    let o = cx.options;
    if o.batch == 0 || o.context == 0 || o.block_size == 0 || o.blocks_per_row == Some(0) {
        return Err("--batch, --context, --block-size and --blocks-per-row must be above 0".into());
    }
    if o.rope_head_dim() < 2 || o.rope_head_dim() % 2 != 0 {
        return Err("--head-dim must be even".to_string());
    }
    let card = cx.card.ok_or("decode-inputs needs a card")?;
    let dev = &card.dev;
    let stream = dev.stream().clone();
    let target = Duration::from_secs_f64(o.seconds);
    let step = Step::new(card, o)?;
    let layout = Layout::new(step.width, step.max_blocks, o.rope_head_dim() / 2);
    let mut pinned = Pinned::new(card, layout.total)?;
    let mut packed = device_bytes(&stream, layout.total)?;
    let wait: CudaEvent = dev.new_event().map_err(|e| format!("event: {e:?}"))?;
    let spin = Spin::lasting(card, 0.0)?;
    let fail = Fail::default();

    // One graph holding one kernel that returns at once.
    stream.synchronize().map_err(|e| format!("stream: {e:?}"))?;
    cuda_ext::begin_capture(&stream).map_err(|e| format!("graph capture: {e:?}"))?;
    let launched = spin.launch(&stream);
    let graph = cuda_ext::end_capture(&stream).map_err(|e| format!("graph capture: {e:?}"))?;
    launched?;
    let graph = graph.ok_or("graph capture: no graph")?;

    let mut rows = Vec::new();
    let mut measure = |variant: String, bytes: Option<u64>, body: &mut dyn FnMut()| {
        let samples = run_for(target, timed(body));
        fail.check(&variant)?;
        rows.push(Measured {
            variant,
            bytes,
            samples_us: samples,
            note: None,
        });
        Ok::<(), String>(())
    };

    // Each of the 6 `from_vec` is a host-to-card copy followed by a stream synchronisation.
    // Each of the 6 `slice_set` is a card-to-card copy.
    let bytes = Some(layout.total as u64);
    measure(
        "as the worker does: 6 h2d, 6 d2d, 8 syncs".to_string(),
        bytes,
        &mut || {
            fail.ok(step.upload());
            fail.ok(stream.synchronize());
            fail.ok(stream.synchronize());
        },
    )?;
    measure(
        "no explicit sync: 6 h2d, 6 d2d, 6 syncs, 1 event wait".to_string(),
        bytes,
        &mut || {
            fail.ok(step.upload());
            fail.ok(wait.record(&stream));
            fail.ok(wait.synchronize());
        },
    )?;
    measure(
        "packed: 1 h2d, 0 d2d, 0 syncs, 1 event wait".to_string(),
        bytes,
        &mut || {
            step.pack(&layout, pinned.as_mut_slice());
            fail.ok(stream.memcpy_htod(pinned.as_slice(), &mut packed));
            let views: Vec<_> = layout
                .sections()
                .iter()
                .map(|s: &Section| packed.slice(s.range()))
                .collect();
            std::hint::black_box(&views);
            fail.ok(wait.record(&stream));
            fail.ok(wait.synchronize());
        },
    )?;
    measure(
        "floor: stream synchronise, nothing queued".to_string(),
        None,
        &mut || fail.ok(stream.synchronize()),
    )?;
    measure(
        "floor: event record and wait, nothing queued".to_string(),
        None,
        &mut || {
            fail.ok(wait.record(&stream));
            fail.ok(wait.synchronize());
        },
    )?;
    measure(
        "floor: graph replay of a trivial kernel, 1 sync".to_string(),
        None,
        &mut || {
            fail.ok(graph.launch());
            fail.ok(stream.synchronize());
        },
    )?;
    measure(
        "floor: graph replay of a trivial kernel, 1 event wait".to_string(),
        None,
        &mut || {
            fail.ok(graph.launch());
            fail.ok(wait.record(&stream));
            fail.ok(wait.synchronize());
        },
    )?;
    note_ratios(&mut rows, SEQUENCES);
    Ok(rows)
}
