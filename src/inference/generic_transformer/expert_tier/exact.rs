//! A cached layer's routed experts computed in the steps of [`ExactDot`] wherever they run: the
//! card's slots, the transit room and the host give each routed pair the same bits, and the pairs
//! of a row are summed in the router's order. Which experts the cache holds changes from one
//! request to the next; with this, the answer does not change with it.
//!
//! A pair's row is `w * down(swiglu(gate . x, up . x))`: the input row as q8_K blocks, the gate
//! and up products, the SwiGLU of `offload::experts`, its row as q8_K blocks, the down product,
//! the weight. The card's kernels are `tx_*` in `requant.cu`.
use super::*;
use crate::tensor::gpu_runtime::driver::{LaunchConfig, PushKernelArg};
use crate::tensor::quant_cpu::{BlockFormat, BlockQ2K, BlockQ4K, BlockQ6K, BlockQ8K, ExactDot};

/// A format the exact kernels read.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Fmt {
    Q2K,
    Q4K,
    Q6K,
}

impl Fmt {
    fn of(dtype: GgmlDType) -> Option<Self> {
        match dtype {
            GgmlDType::Q2K => Some(Self::Q2K),
            GgmlDType::Q4K => Some(Self::Q4K),
            GgmlDType::Q6K => Some(Self::Q6K),
            _ => None,
        }
    }

    /// The suffix of the kernels' entry points.
    fn name(self) -> &'static str {
        match self {
            Self::Q2K => "q2_k",
            Self::Q4K => "q4_k",
            Self::Q6K => "q6_k",
        }
    }
}

/// The formats and widths of a layer whose experts the exact kernels serve.
#[derive(Clone, Copy, Debug)]
pub(super) struct Exact {
    pub(super) gate: Fmt,
    pub(super) down: Fmt,
    pub(super) dim: usize,
    pub(super) inter: usize,
}

/// Values of a q8_K block, the unit both widths are counted in.
const BLOCK: usize = crate::tensor::quant_cpu::QK_K;

impl Exact {
    /// The exact form of the stacks `[gate, up, down]`, `None` when a format has no exact kernel
    /// or a width is not whole blocks.
    pub(super) fn of(host: &[Arc<QTensor>; 3]) -> Option<Self> {
        let gate = Fmt::of(host[0].dtype())?;
        if host[1].dtype() != host[0].dtype() {
            return None;
        }
        let down = Fmt::of(host[2].dtype())?;
        let (g, d) = (host[0].shape().dims(), host[2].shape().dims());
        let (inter, dim) = match (g, d) {
            ([_, i, h], [_, h2, i2]) if h == h2 && i == i2 => (*i, *h),
            _ => return None,
        };
        (dim % BLOCK == 0 && inter % BLOCK == 0).then_some(Self {
            gate,
            down,
            dim,
            inter,
        })
    }

    /// Bytes of `rows` rows of `len` values as q8_K blocks.
    pub(super) fn q8k_bytes(rows: usize, len: usize) -> usize {
        rows * (len / BLOCK) * std::mem::size_of::<BlockQ8K>()
    }
}

/// Where a launch finds its experts: slot `s` below `base_slots` at `base[stack] + s * bytes`,
/// past them in the stacks whose addresses the card reads at `extra` (gate, up, down; `0` when
/// there are none).
pub(super) struct Stacks {
    pub(super) base: [u64; 3],
    pub(super) extra: u64,
    pub(super) base_slots: usize,
    pub(super) bytes: [usize; 3],
}

/// The buffers on the card a launch reads and writes: the input rows as q8_K, the rows entering
/// the down projections `[pairs, inter]` and their q8_K blocks, and the pairs' rows
/// `[pairs, dim]`.
#[derive(Clone, Copy)]
pub(super) struct Rooms {
    pub(super) xq: u64,
    pub(super) h: u64,
    pub(super) hq: u64,
    pub(super) out: u64,
}

fn launched<T, E: std::fmt::Debug>(what: &str, r: std::result::Result<T, E>) -> Result<()> {
    r.map(|_| ())
        .map_err(|e| crate::tensor::Error::msg(format!("{what} launch: {e:?}")))
}

/// A warp per element of `grid`.
fn warps(grid: (u32, u32)) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (grid.0, grid.1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Load the kernels of `exact` on `dev` before a recorded decode can wait on the host: a kernel
/// loaded at its first launch, while the card waits on the tier worker, holds the driver until the
/// card is idle, and the worker's copies queue behind it.
pub(super) fn preload(dev: &Device, exact: &Exact) -> Result<()> {
    let cd = dev.as_cuda_device()?;
    for name in [
        "cx_quantize_q8_k".to_string(),
        format!("tx_h_{}", exact.gate.name()),
        "tx_quantize_h".to_string(),
        format!("tx_down_{}", exact.down.name()),
        "tx_sources".to_string(),
        "tx_sum".to_string(),
    ] {
        cd.0.quantized_fn(&name)?;
    }
    Ok(())
}

/// The rows `[rows, dim]` f32 at `x` as q8_K rows at `xq`.
pub(super) fn quantize_rows(dev: &Device, x: u64, xq: u64, rows: usize, dim: usize) -> Result<()> {
    let cd = dev.as_cuda_device()?;
    let f = cd.0.quantized_fn("cx_quantize_q8_k")?;
    let stream = cd.0.stream();
    let blocks = (rows * dim / BLOCK) as i64;
    let mut b = stream.launch_builder(&f);
    b.arg(&x);
    b.arg(&xq);
    b.arg(&blocks);
    launched("cx_quantize_q8_k", unsafe {
        b.launch(warps((blocks as u32, 1)))
    })
}

/// The rows of `n` instances: instance `i` is the pair `pairs[i]` (`i` when `pairs` is `0`), its
/// expert in slot `slots[i]` of `stacks` (all ones: skipped), its input row `pair / k` of
/// `rooms.xq`. Writes `rooms.out[pair] = weights[pair] * product`.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_pairs(
    dev: &Device,
    exact: &Exact,
    stacks: &Stacks,
    slots: u64,
    pairs: u64,
    n: usize,
    k: usize,
    weights: u64,
    rooms: Rooms,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let cd = dev.as_cuda_device()?;
    let stream = cd.0.stream();
    let (dim, inter) = (exact.dim as i32, exact.inter as i32);
    let base_slots = stacks.base_slots as i32;
    let k = k as i32;
    let name = format!("tx_h_{}", exact.gate.name());
    let f = cd.0.quantized_fn(&name)?;
    let gate_bytes = stacks.bytes[0] as i64;
    let mut b = stream.launch_builder(&f);
    b.arg(&stacks.base[0]);
    b.arg(&stacks.base[1]);
    b.arg(&stacks.extra);
    b.arg(&base_slots);
    b.arg(&gate_bytes);
    b.arg(&slots);
    b.arg(&pairs);
    b.arg(&k);
    b.arg(&rooms.xq);
    b.arg(&rooms.h);
    b.arg(&dim);
    b.arg(&inter);
    launched(&name, unsafe { b.launch(warps((inter as u32, n as u32))) })?;
    let f = cd.0.quantized_fn("tx_quantize_h")?;
    let mut b = stream.launch_builder(&f);
    b.arg(&rooms.h);
    b.arg(&slots);
    b.arg(&pairs);
    b.arg(&rooms.hq);
    b.arg(&inter);
    launched("tx_quantize_h", unsafe {
        b.launch(warps(((exact.inter / BLOCK) as u32, n as u32)))
    })?;
    let name = format!("tx_down_{}", exact.down.name());
    let f = cd.0.quantized_fn(&name)?;
    let down_bytes = stacks.bytes[2] as i64;
    let mut b = stream.launch_builder(&f);
    b.arg(&stacks.base[2]);
    b.arg(&stacks.extra);
    b.arg(&base_slots);
    b.arg(&down_bytes);
    b.arg(&slots);
    b.arg(&pairs);
    b.arg(&weights);
    b.arg(&rooms.hq);
    b.arg(&rooms.out);
    b.arg(&inter);
    b.arg(&dim);
    launched(&name, unsafe { b.launch(warps((dim as u32, n as u32))) })
}

/// After a recorded decode's answer: the transit room's pairs at `late_slots` and the source of
/// every pair's row at `src` (see `tx_sources`).
#[allow(clippy::too_many_arguments)]
pub(super) fn sources(
    dev: &Device,
    card_slots: u64,
    counter: u64,
    live: u64,
    late_pairs: u64,
    late_places: u64,
    late_slots: u64,
    src: u64,
    pairs: usize,
) -> Result<()> {
    let cd = dev.as_cuda_device()?;
    let f = cd.0.quantized_fn("tx_sources")?;
    let stream = cd.0.stream();
    let n = pairs as i32;
    let mut b = stream.launch_builder(&f);
    b.arg(&card_slots);
    b.arg(&counter);
    b.arg(&live);
    b.arg(&late_pairs);
    b.arg(&late_places);
    b.arg(&late_slots);
    b.arg(&src);
    b.arg(&n);
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    launched("tx_sources", unsafe { b.launch(cfg) })
}

/// `out[row]`, `rows` rows of `hidden`: the row's `k` pairs summed in their order, each read
/// from `card` or, where `src` names one, from that row of `host`.
#[allow(clippy::too_many_arguments)]
pub(super) fn sum(
    dev: &Device,
    card: u64,
    host: u64,
    src: u64,
    out: u64,
    rows: usize,
    k: usize,
    hidden: usize,
) -> Result<()> {
    let cd = dev.as_cuda_device()?;
    let f = cd.0.quantized_fn("tx_sum")?;
    let stream = cd.0.stream();
    let (r, kk, h) = (rows as i32, k as i32, hidden as i32);
    let mut b = stream.launch_builder(&f);
    b.arg(&card);
    b.arg(&host);
    b.arg(&src);
    b.arg(&out);
    b.arg(&r);
    b.arg(&kk);
    b.arg(&h);
    let threads = 256u32;
    let cfg = LaunchConfig {
        grid_dim: (((rows * hidden) as u32).div_ceil(threads), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    launched("tx_sum", unsafe { b.launch(cfg) })
}

/// The host's rows of `pairs`, each `(input row, expert, weight)` over the rows `x` of
/// `exact.dim`: `[pairs, dim]`, the bits the card's kernels give the same pairs.
pub(super) fn host_pairs(
    exact: &Exact,
    host: &[Arc<QTensor>; 3],
    x: &[f32],
    pairs: &[(usize, u32, f32)],
) -> Result<Vec<f32>> {
    fn typed<T: BlockFormat>(t: &QTensor) -> &[T] {
        let bytes = t.native_qtensor().data();
        // Safety: the stack holds blocks of `T`, its format, laid out as the file stores them.
        unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const T,
                bytes.len() / std::mem::size_of::<T>(),
            )
        }
    }
    fn run<G: ExactDot, D: ExactDot>(
        exact: &Exact,
        host: &[Arc<QTensor>; 3],
        x: &[f32],
        pairs: &[(usize, u32, f32)],
    ) -> Vec<f32> {
        let (gate, up, down) = (
            typed::<G>(&host[0]),
            typed::<G>(&host[1]),
            typed::<D>(&host[2]),
        );
        host_rows::<G, D>(gate, up, down, exact.dim, exact.inter, x, pairs)
    }
    Ok(match (exact.gate, exact.down) {
        (Fmt::Q2K, Fmt::Q2K) => run::<BlockQ2K, BlockQ2K>(exact, host, x, pairs),
        (Fmt::Q2K, Fmt::Q4K) => run::<BlockQ2K, BlockQ4K>(exact, host, x, pairs),
        (Fmt::Q2K, Fmt::Q6K) => run::<BlockQ2K, BlockQ6K>(exact, host, x, pairs),
        (Fmt::Q4K, Fmt::Q2K) => run::<BlockQ4K, BlockQ2K>(exact, host, x, pairs),
        (Fmt::Q4K, Fmt::Q4K) => run::<BlockQ4K, BlockQ4K>(exact, host, x, pairs),
        (Fmt::Q4K, Fmt::Q6K) => run::<BlockQ4K, BlockQ6K>(exact, host, x, pairs),
        (Fmt::Q6K, Fmt::Q2K) => run::<BlockQ6K, BlockQ2K>(exact, host, x, pairs),
        (Fmt::Q6K, Fmt::Q4K) => run::<BlockQ6K, BlockQ4K>(exact, host, x, pairs),
        (Fmt::Q6K, Fmt::Q6K) => run::<BlockQ6K, BlockQ6K>(exact, host, x, pairs),
    })
}

/// [`host_pairs`] over typed stacks: gate and up `[experts, inter, dim]`, down
/// `[experts, dim, inter]`.
fn host_rows<G: ExactDot, D: ExactDot>(
    gate: &[G],
    up: &[G],
    down: &[D],
    dim: usize,
    inter: usize,
    x: &[f32],
    pairs: &[(usize, u32, f32)],
) -> Vec<f32> {
    let (nb, nbi) = (dim / BLOCK, inter / BLOCK);
    let rows = x.len() / dim;
    let mut xq = vec![BlockQ8K::zeros(); rows * nb];
    for (r, q) in xq.chunks_exact_mut(nb).enumerate() {
        if pairs.iter().any(|p| p.0 == r) {
            BlockQ8K::quantize(&x[r * dim..(r + 1) * dim], q);
        }
    }
    let n = pairs.len();
    if n == 0 {
        return Vec::new();
    }
    // The pairs by expert, in their order within one: each unit of work is one expert and a run
    // of its weight rows, and each row meets every pair of the expert before the next is read,
    // so an expert's blocks are read once for all the pairs that route to it.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&p| pairs[p].1);
    let mut groups: Vec<(usize, usize)> = Vec::new();
    for (j, &p) in order.iter().enumerate() {
        match groups.last_mut() {
            Some(g) if pairs[order[g.0]].1 == pairs[p].1 => g.1 = j + 1,
            _ => groups.push((j, j + 1)),
        }
    }
    let units = |len: usize| -> Vec<(usize, usize)> {
        groups
            .iter()
            .enumerate()
            .flat_map(|(g, _)| (0..len.div_ceil(ROWS_A_UNIT)).map(move |c| (g, c * ROWS_A_UNIT)))
            .collect()
    };
    let pool = crate::tensor::quant_cpu::gemv_pool::pool();
    let mut h = vec![0f32; n * inter];
    {
        let at_h = Shared(h.as_mut_ptr());
        let work = units(inter);
        pool.run(work.len(), &|u| {
            let at_h = &at_h;
            let (g, r0) = work[u];
            let (lo, hi) = groups[g];
            let e = pairs[order[lo]].1 as usize;
            for r in r0..(r0 + ROWS_A_UNIT).min(inter) {
                let at = (e * inter + r) * nb;
                let (gw, uw) = (&gate[at..at + nb], &up[at..at + nb]);
                for &p in &order[lo..hi] {
                    let row = pairs[p].0;
                    let a = &xq[row * nb..(row + 1) * nb];
                    let v = crate::inference::offload::experts::swiglu(
                        G::dot_exact(gw, a),
                        G::dot_exact(uw, a),
                        0.0,
                    );
                    // SAFETY: each (pair, row) is written by the one unit that holds the row of
                    // the pair's expert.
                    unsafe { *at_h.0.add(p * inter + r) = v };
                }
            }
        });
    }
    let mut hq = vec![BlockQ8K::zeros(); n * nbi];
    for (p, q) in hq.chunks_exact_mut(nbi).enumerate() {
        BlockQ8K::quantize(&h[p * inter..(p + 1) * inter], q);
    }
    let mut y = vec![0f32; n * dim];
    {
        let at_y = Shared(y.as_mut_ptr());
        let work = units(dim);
        pool.run(work.len(), &|u| {
            let at_y = &at_y;
            let (g, c0) = work[u];
            let (lo, hi) = groups[g];
            let e = pairs[order[lo]].1 as usize;
            for c in c0..(c0 + ROWS_A_UNIT).min(dim) {
                let at = (e * dim + c) * nbi;
                let dw = &down[at..at + nbi];
                for &p in &order[lo..hi] {
                    let v = pairs[p].2 * D::dot_exact(dw, &hq[p * nbi..(p + 1) * nbi]);
                    // SAFETY: each (pair, column) is written by the one unit that holds the
                    // column of the pair's expert.
                    unsafe { *at_y.0.add(p * dim + c) = v };
                }
            }
        });
    }
    y
}

/// Weight rows of one expert a unit of [`host_rows`] takes.
const ROWS_A_UNIT: usize = 16;

/// A pointer the units of [`host_rows`] write their disjoint outputs through.
struct Shared(*mut f32);
// SAFETY: the units write disjoint elements, and the pool joins them before the buffer is read.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

/// The rows of a token's pairs summed as `tx_sum` sums them: in the router's order from -0.
#[cfg(test)]
pub(super) fn sum_in_order(pairs: &[f32], k: usize, hidden: usize) -> Vec<f32> {
    let rows = pairs.len() / (k * hidden).max(1);
    let mut out = vec![-0.0f32; rows * hidden];
    for r in 0..rows {
        for j in 0..k {
            let p = r * k + j;
            for (o, v) in out[r * hidden..(r + 1) * hidden]
                .iter_mut()
                .zip(&pairs[p * hidden..(p + 1) * hidden])
            {
                *o += v;
            }
        }
    }
    out
}

/// What one routed pair stands for in an eager decode once the pairs are partitioned: its
/// expert, its weight, and the slot the card holds the expert in (`None` when it misses).
pub(super) struct Partition<'a> {
    pub(super) ids: &'a [u32],
    pub(super) weights: &'a [f32],
    pub(super) slots: &'a [Option<u32>],
    /// The missing experts copied into the transit room, in the order they land there; the
    /// other missing experts are the host's.
    pub(super) transit: &'a [u32],
}

/// The buffers a recorded decode's exact launches keep on the card, for `rows_max` rows of `k`
/// pairs: the rooms of [`Rooms`], the slot of each pair on the card (all ones: missing) and in
/// the transit room, and where each pair's row is read from.
pub(super) struct GraphRooms {
    xq: Tensor,
    h: Tensor,
    hq: Tensor,
    rows: Tensor,
    pub(super) card_slots: Tensor,
    late_slots: Tensor,
    src: Tensor,
}

impl GraphRooms {
    pub(super) fn new(dev: &Device, exact: &Exact, rows_max: usize, k: usize) -> Result<Self> {
        let pairs = rows_max * k;
        let f32s = |n: usize| Tensor::zeros_on(n, crate::tensor::DType::F32, dev);
        let u32s = |n: usize| Tensor::zeros_on(n, crate::tensor::DType::U32, dev);
        Ok(Self {
            xq: f32s(Exact::q8k_bytes(rows_max, exact.dim) / 4)?,
            h: f32s(pairs * exact.inter)?,
            hq: f32s(Exact::q8k_bytes(pairs, exact.inter) / 4)?,
            rows: f32s(pairs * exact.dim)?,
            card_slots: u32s(pairs)?,
            late_slots: u32s(pairs)?,
            src: u32s(pairs)?,
        })
    }

    fn rooms(&self) -> Result<Rooms> {
        Ok(Rooms {
            xq: address_of::<f32>(&self.xq)?,
            h: address_of::<f32>(&self.h)?,
            hq: address_of::<f32>(&self.hq)?,
            out: address_of::<f32>(&self.rows)?,
        })
    }
}

/// A slot no expert of the launch is in.
const SKIP: u32 = u32::MAX;

impl LayerTier {
    /// The cache's stacks as an exact launch reads them: the extra stacks through the tier's
    /// table, so that a recorded launch follows them when they are lent and come back.
    fn exact_card_stacks(&self) -> Result<Stacks> {
        Ok(Stacks {
            base: [
                self.card[0].device_ptr()? as u64,
                self.card[1].device_ptr()? as u64,
                self.card[2].device_ptr()? as u64,
            ],
            extra: self.extra_table.as_ref().map_or(0, |t| t.addr),
            base_slots: self.base_slots,
            bytes: self.expert_bytes,
        })
    }

    /// The transit room's stacks as an exact launch reads them, an expert's place for its slot.
    fn exact_transit_stacks(&self) -> Option<Stacks> {
        use crate::tensor::gpu_runtime::driver::DevicePtr;
        let stage = self.stage.as_ref()?;
        let arenas = &stage.stager.buffers[stage.buffer].arenas;
        Some(Stacks {
            base: [0, 1, 2].map(|i| arenas[i].device_ptr(arenas[i].stream()).0),
            extra: 0,
            base_slots: stage.stager.transit,
            bytes: self.expert_bytes,
        })
    }

    /// An eager decode's expert block in the exact steps: the card's pairs launched first, the
    /// copies into the transit room started, the host's pairs computed while they run, the
    /// transit room's pairs launched once they land, and every row's pairs summed in order.
    /// `xs` `[rows, hidden]` and the router's weights `w_card` (in pair order) are on the card,
    /// `x_host` holds the same rows. Returns `[rows, hidden]` on the card.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn exact_eager(
        &self,
        exact: &Exact,
        scratch: &Scratch,
        xs: &Tensor,
        x_host: &[f32],
        w_card: &Tensor,
        part: Partition<'_>,
        laps: &mut Laps,
    ) -> Result<Tensor> {
        let host = Device::Cpu;
        let dev = xs.device().clone();
        let stream = dev.as_cuda_device()?.0.stream().clone();
        let (hidden, pairs) = (exact.dim, part.ids.len());
        let rows = x_host.len() / hidden;
        let k = pairs / rows.max(1);
        let x = xs.to_dtype(crate::tensor::DType::F32)?.contiguous()?;
        let f32s = |n: usize| Tensor::zeros_on(n, crate::tensor::DType::F32, &dev);
        let (xq, h, hq, pair_rows) = (
            f32s(Exact::q8k_bytes(rows, exact.dim) / 4)?,
            f32s(pairs * exact.inter)?,
            f32s(Exact::q8k_bytes(pairs, exact.inter) / 4)?,
            f32s(pairs * hidden)?,
        );
        let rooms = Rooms {
            xq: address_of::<f32>(&xq)?,
            h: address_of::<f32>(&h)?,
            hq: address_of::<f32>(&hq)?,
            out: address_of::<f32>(&pair_rows)?,
        };
        let w = address_of::<f32>(w_card)?;
        let upload = |v: Vec<u32>| Tensor::from_vec(v, pairs, &host)?.to_device(&dev);
        quantize_rows(&dev, address_of::<f32>(&x)?, rooms.xq, rows, exact.dim)?;
        let card_slots: Vec<u32> = part.slots.iter().map(|s| s.unwrap_or(SKIP)).collect();
        let card_slots = if part.slots.iter().any(Option::is_some) {
            let t = upload(card_slots)?;
            run_pairs(
                &dev,
                exact,
                &self.exact_card_stacks()?,
                address_of::<u32>(&t)?,
                0,
                pairs,
                k,
                w,
                rooms,
            )?;
            Some(t)
        } else {
            None
        };
        laps.mark("tier exact: card launch");
        if !part.transit.is_empty() {
            self.copy_to_transit(scratch, part.transit)?;
        }
        // The host's pairs, each read from its row of `host_rows` by the sum.
        let mut src = vec![u32::MAX; pairs];
        let mut mine: Vec<(usize, u32, f32)> = Vec::new();
        for p in 0..pairs {
            if part.slots[p].is_none() && !part.transit.contains(&part.ids[p]) {
                src[p] = mine.len() as u32;
                mine.push((p / k, part.ids[p], part.weights[p]));
            }
        }
        let t_host = std::time::Instant::now();
        let host_rows = if mine.is_empty() {
            None
        } else {
            let y = host_pairs(exact, &self.host, x_host, &mine)?;
            let mut experts: Vec<u32> = mine.iter().map(|p| p.1).collect();
            experts.sort_unstable();
            experts.dedup();
            blend(
                &self.host_us[pace_class(rows)],
                t_host.elapsed().as_secs_f64() * 1e6 / experts.len() as f64,
            );
            Some(Tensor::from_vec(y, mine.len() * hidden, &host)?.to_device(&dev)?)
        };
        laps.mark("tier exact: host experts");
        let transit_slots = if part.transit.is_empty() {
            None
        } else {
            stream
                .wait(&scratch.upload_end)
                .map_err(|e| crate::tensor::Error::msg(format!("wait: {e:?}")))?;
            let places: Vec<u32> = (0..pairs)
                .map(|p| match part.slots[p] {
                    Some(_) => SKIP,
                    None => part
                        .transit
                        .iter()
                        .position(|&e| e == part.ids[p])
                        .map_or(SKIP, |at| at as u32),
                })
                .collect();
            let t = upload(places)?;
            let stacks = self
                .exact_transit_stacks()
                .expect("a copy to transit has a stage");
            run_pairs(
                &dev,
                exact,
                &stacks,
                address_of::<u32>(&t)?,
                0,
                pairs,
                k,
                w,
                rooms,
            )?;
            Some(t)
        };
        laps.mark("tier exact: transit launch");
        let src = upload(src)?;
        let out = f32s(rows * hidden)?;
        sum(
            &dev,
            rooms.out,
            host_rows.as_ref().map_or(Ok(0), |t| address_of::<f32>(t))?,
            address_of::<u32>(&src)?,
            address_of::<f32>(&out)?,
            rows,
            k,
            hidden,
        )?;
        // The buffers the launches read live until the launches are queued behind them.
        drop((
            card_slots,
            transit_slots,
            host_rows,
            xq,
            h,
            hq,
            pair_rows,
            src,
        ));
        out.reshape((rows, hidden))
    }

    /// The first half of a recorded decode's expert block in the exact steps, after the plan:
    /// the card's pairs, their rows in the rooms of `g`.
    pub(super) fn exact_graph_begin(
        &self,
        exact: &Exact,
        g: &GraphRooms,
        x: &Tensor,
        weights: &Tensor,
        rows: usize,
        k: usize,
    ) -> Result<()> {
        let rooms = g.rooms()?;
        quantize_rows(&self.dev, address_of::<f32>(x)?, rooms.xq, rows, exact.dim)?;
        run_pairs(
            &self.dev,
            exact,
            &self.exact_card_stacks()?,
            address_of::<u32>(&g.card_slots)?,
            0,
            rows * k,
            k,
            address_of::<f32>(weights)?,
            rooms,
        )
    }

    /// The second half, after the wait for the host's answer: the transit room's pairs the host
    /// named, then every row's pairs summed in order, the host's read from the mailbox's
    /// `product`. Writes `out` `[rows, hidden]`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn exact_graph_end(
        &self,
        exact: &Exact,
        g: &GraphRooms,
        mail: Mail,
        weights: &Tensor,
        rows: usize,
        k: usize,
        out: u64,
    ) -> Result<()> {
        let pairs = rows * k;
        let rooms = g.rooms()?;
        sources(
            &self.dev,
            address_of::<u32>(&g.card_slots)?,
            mail.counter,
            mail.live,
            mail.late_pairs,
            mail.late_places,
            address_of::<u32>(&g.late_slots)?,
            address_of::<u32>(&g.src)?,
            pairs,
        )?;
        if let Some(stacks) = self.exact_transit_stacks() {
            run_pairs(
                &self.dev,
                exact,
                &stacks,
                address_of::<u32>(&g.late_slots)?,
                0,
                pairs,
                k,
                address_of::<f32>(weights)?,
                rooms,
            )?;
        }
        sum(
            &self.dev,
            rooms.out,
            mail.product,
            address_of::<u32>(&g.src)?,
            out,
            rows,
            k,
            exact.dim,
        )
    }
}

/// The card's addresses of what a recorded decode's second half reads of the mailbox: the
/// layer's counter, the count of late pairs, the late pairs and their places in the transit
/// room, and the host's rows.
pub(super) struct Mail {
    pub(super) counter: u64,
    pub(super) live: u64,
    pub(super) late_pairs: u64,
    pub(super) late_places: u64,
    pub(super) product: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stack of `experts` matrices `[rows, cols]` drawn from `seed`, quantised on the host.
    fn stack(
        experts: usize,
        rows: usize,
        cols: usize,
        seed: u32,
        dtype: GgmlDType,
    ) -> Arc<QTensor> {
        let values: Vec<f32> = (0..experts * rows * cols)
            .map(|i| {
                (((i as u32)
                    .wrapping_add(seed.wrapping_mul(0x9E37_79B9))
                    .wrapping_mul(2_654_435_761)
                    .rotate_left(9)
                    >> 8) as f32
                    / (1 << 24) as f32
                    - 0.5)
                    * 0.2
            })
            .collect();
        let w = Tensor::from_vec(values, (experts, rows, cols), &Device::Cpu).unwrap();
        Arc::new(QTensor::quantize_onto(&w, dtype, &Device::Cpu).unwrap())
    }

    /// The host's pairs of a synthetic prompt, each expert read once for all the rows that route
    /// to it, are the bits each pair gets computed alone as a decoded token's would be, and the
    /// rows they sum to in the router's order are the same.
    #[test]
    fn a_prompt_batched_by_expert_matches_its_pairs_one_by_one() {
        let (experts, used, dim, inter, rows) = (6usize, 3usize, 512usize, 256usize, 11usize);
        for (gate_dt, down_dt) in [
            (GgmlDType::Q2K, GgmlDType::Q4K),
            (GgmlDType::Q4K, GgmlDType::Q6K),
        ] {
            let host = [
                stack(experts, inter, dim, 1, gate_dt),
                stack(experts, inter, dim, 2, gate_dt),
                stack(experts, dim, inter, 3, down_dt),
            ];
            let exact = Exact::of(&host).expect("formats with exact kernels");
            let x: Vec<f32> = (0..rows * dim)
                .map(|i| ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 4.0)
                .collect();
            // Row r routes to experts r, r + 1 and r + 3 (mod experts): each expert serves several
            // rows, adjacent and not.
            let pairs: Vec<(usize, u32, f32)> = (0..rows)
                .flat_map(|r| {
                    [0usize, 1, 3]
                        .map(|d| (r, ((r + d) % experts) as u32, 0.2 + 0.03 * (r + d) as f32))
                })
                .collect();
            assert_eq!(pairs.len(), rows * used);
            let together = host_pairs(&exact, &host, &x, &pairs).unwrap();
            let mut alone = Vec::with_capacity(together.len());
            for p in &pairs {
                alone.extend(host_pairs(&exact, &host, &x, std::slice::from_ref(p)).unwrap());
            }
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&together), bits(&alone), "{gate_dt:?} / {down_dt:?}");
            assert_eq!(
                bits(&sum_in_order(&together, used, dim)),
                bits(&sum_in_order(&alone, used, dim))
            );
        }
    }
}
