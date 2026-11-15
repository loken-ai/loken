//! A cached layer's routed experts computed in the steps of [`ExactDot`] wherever they run: the
//! card's slots, the transit room and the host give each routed pair the same bits, and the pairs
//! of a row are summed in the router's order. Which experts the cache holds changes from one
//! request to the next; with this, the answer does not change with it.
//!
//! A pair's row is `w * down(swiglu(gate . x, up . x))`: the input row as the activation blocks
//! of the gate's format, the gate and up products, the SwiGLU of `bounded_swiglu`, its row as the
//! activation blocks of the down projection's format, the down product, the weight. The card's
//! kernels are `tx_*` in `requant.cu`.
use super::*;
use crate::tensor::gpu_runtime::driver::{LaunchConfig, PushKernelArg};
use crate::tensor::quant_cpu::{
    BlockFormat, BlockIq2S, BlockIq3S, BlockIq3Xxs, BlockIq4Nl, BlockQ2K, BlockQ2_0, BlockQ4K,
    BlockQ5_0, BlockQ6K, BlockQ8K, BlockQ8K32, BlockQ8_0, ExactDot,
};

/// A format the exact kernels read.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) enum Fmt {
    Q2K,
    Q4K,
    Q6K,
    Iq2S,
    Iq3Xxs,
    Iq3S,
    Iq4Nl,
    Q2_0,
    Q8_0,
    Q5_0,
}

impl Fmt {
    fn of(dtype: GgmlDType) -> Option<Self> {
        match dtype {
            GgmlDType::Q2K => Some(Self::Q2K),
            GgmlDType::Q4K => Some(Self::Q4K),
            GgmlDType::Q6K => Some(Self::Q6K),
            GgmlDType::Iq2S => Some(Self::Iq2S),
            GgmlDType::Iq3Xxs => Some(Self::Iq3Xxs),
            GgmlDType::Iq3S => Some(Self::Iq3S),
            GgmlDType::Iq4Nl => Some(Self::Iq4Nl),
            GgmlDType::Q2_0 => Some(Self::Q2_0),
            GgmlDType::Q8_0 => Some(Self::Q8_0),
            GgmlDType::Q5_0 => Some(Self::Q5_0),
            _ => None,
        }
    }

    /// The suffix of the kernels' entry points.
    fn name(self) -> &'static str {
        match self {
            Self::Q2K => "q2_k",
            Self::Q4K => "q4_k",
            Self::Q6K => "q6_k",
            Self::Iq2S => "iq2_s",
            Self::Iq3Xxs => "iq3_xxs",
            Self::Iq3S => "iq3_s",
            Self::Iq4Nl => "iq4_nl",
            Self::Q2_0 => "q2_0",
            Self::Q8_0 => "q8_0",
            Self::Q5_0 => "q5_0",
        }
    }

    /// Whether the format's dot reads q8_K32 rows and folds a block run by run, rather than
    /// reading q8_K rows.
    fn runs(self) -> bool {
        matches!(
            self,
            Self::Iq2S
                | Self::Iq3Xxs
                | Self::Iq3S
                | Self::Iq4Nl
                | Self::Q2_0
                | Self::Q8_0
                | Self::Q5_0
        )
    }

    /// Bytes of an activation block the format's dot reads.
    fn act_block_bytes(self) -> usize {
        if self.runs() {
            std::mem::size_of::<BlockQ8K32>()
        } else {
            std::mem::size_of::<BlockQ8K>()
        }
    }

    /// Whether a weight row of `len` values in format `dtype` can be read: whole blocks of 256
    /// values for a q8_K format, whole runs of 32 and whole blocks of the format for a run format.
    fn reads_rows_of(self, dtype: GgmlDType, len: usize) -> bool {
        let unit = if self.runs() {
            RUN.max(dtype.block_size())
        } else {
            BLOCK
        };
        len > 0 && len.is_multiple_of(unit)
    }
}

/// Values of a run: the unit a run format's row is counted in.
const RUN: usize = 32;

/// The formats and widths of a layer whose experts the exact kernels serve.
#[derive(Clone, Copy, Debug)]
pub(super) struct Exact {
    pub(super) gate: Fmt,
    pub(super) down: Fmt,
    pub(super) dim: usize,
    pub(super) inter: usize,
    /// The bound of the SwiGLU, `0` for none: the `limit` of `bounded_swiglu::swiglu`.
    pub(super) limit: f32,
}

/// Values of an activation block, the unit both widths are counted in.
const BLOCK: usize = crate::tensor::quant_cpu::QK_K;

impl Exact {
    /// The exact form of the stacks `[gate, up, down]` under the SwiGLU bound `limit`, `None`
    /// when a format has no exact kernel, its blocks do not divide an activation block or a row
    /// is not whole units of its format (see `Fmt::reads_rows_of`).
    pub(super) fn of(host: &[Arc<QTensor>; 3], limit: f32) -> Option<Self> {
        let gate = Fmt::of(host[0].dtype())?;
        if host[1].dtype() != host[0].dtype() {
            return None;
        }
        let down = Fmt::of(host[2].dtype())?;
        if host
            .iter()
            .any(|t| !BLOCK.is_multiple_of(t.dtype().block_size()))
        {
            return None;
        }
        let (g, d) = (host[0].shape().dims(), host[2].shape().dims());
        let (inter, dim) = match (g, d) {
            ([_, i, h], [_, h2, i2]) if h == h2 && i == i2 => (*i, *h),
            _ => return None,
        };
        let rows_read =
            gate.reads_rows_of(host[0].dtype(), dim) && down.reads_rows_of(host[2].dtype(), inter);
        rows_read.then_some(Self {
            gate,
            down,
            dim,
            inter,
            limit,
        })
    }

    /// Bytes of `rows` rows of `len` values as the activation blocks `fmt`'s dot reads, the last
    /// block of a row padded.
    pub(super) fn act_bytes(fmt: Fmt, rows: usize, len: usize) -> usize {
        rows * len.div_ceil(BLOCK) * fmt.act_block_bytes()
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

/// The buffers on the card a launch reads and writes: the input rows as the gate's activation
/// blocks, the rows entering the down projections `[pairs, inter]` and their blocks as the down
/// projection's, and the pairs' rows `[pairs, dim]`.
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

/// The rows `[rows, dim]` f32 at `x` as the activation blocks `fmt`'s dot reads, at `xq`.
pub(super) fn quantize_rows(
    dev: &Device,
    fmt: Fmt,
    x: u64,
    xq: u64,
    rows: usize,
    dim: usize,
) -> Result<()> {
    let cd = dev.as_cuda_device()?;
    let stream = cd.0.stream();
    let blocks = (rows * dim.div_ceil(BLOCK)) as i64;
    let name = if fmt.runs() {
        "tx_quantize_q8_k32"
    } else {
        "cx_quantize_q8_k"
    };
    let f = cd.0.quantized_fn(name)?;
    let len = dim as i32;
    let mut b = stream.launch_builder(&f);
    b.arg(&x);
    b.arg(&xq);
    b.arg(&blocks);
    if fmt.runs() {
        b.arg(&len);
    }
    launched(name, unsafe { b.launch(warps((blocks as u32, 1))) })
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
    b.arg(&exact.limit);
    launched(&name, unsafe { b.launch(warps((inter as u32, n as u32))) })?;
    let name = if exact.down.runs() {
        "tx_quantize_h_q8_k32"
    } else {
        "tx_quantize_h"
    };
    let f = cd.0.quantized_fn(name)?;
    let mut b = stream.launch_builder(&f);
    b.arg(&rooms.h);
    b.arg(&slots);
    b.arg(&pairs);
    b.arg(&rooms.hq);
    b.arg(&inter);
    launched(name, unsafe {
        b.launch(warps((exact.inter.div_ceil(BLOCK) as u32, n as u32)))
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
        host_rows::<G, D>(gate, up, down, exact, x, pairs)
    }
    fn with_down<G: ExactDot>(
        exact: &Exact,
        host: &[Arc<QTensor>; 3],
        x: &[f32],
        pairs: &[(usize, u32, f32)],
    ) -> Vec<f32> {
        match exact.down {
            Fmt::Q2K => run::<G, BlockQ2K>(exact, host, x, pairs),
            Fmt::Q4K => run::<G, BlockQ4K>(exact, host, x, pairs),
            Fmt::Q6K => run::<G, BlockQ6K>(exact, host, x, pairs),
            Fmt::Iq2S => run::<G, BlockIq2S>(exact, host, x, pairs),
            Fmt::Iq3Xxs => run::<G, BlockIq3Xxs>(exact, host, x, pairs),
            Fmt::Iq3S => run::<G, BlockIq3S>(exact, host, x, pairs),
            Fmt::Iq4Nl => run::<G, BlockIq4Nl>(exact, host, x, pairs),
            Fmt::Q2_0 => run::<G, BlockQ2_0>(exact, host, x, pairs),
            Fmt::Q8_0 => run::<G, BlockQ8_0>(exact, host, x, pairs),
            Fmt::Q5_0 => run::<G, BlockQ5_0>(exact, host, x, pairs),
        }
    }
    Ok(match exact.gate {
        Fmt::Q2K => with_down::<BlockQ2K>(exact, host, x, pairs),
        Fmt::Q4K => with_down::<BlockQ4K>(exact, host, x, pairs),
        Fmt::Q6K => with_down::<BlockQ6K>(exact, host, x, pairs),
        Fmt::Iq2S => with_down::<BlockIq2S>(exact, host, x, pairs),
        Fmt::Iq3Xxs => with_down::<BlockIq3Xxs>(exact, host, x, pairs),
        Fmt::Iq3S => with_down::<BlockIq3S>(exact, host, x, pairs),
        Fmt::Iq4Nl => with_down::<BlockIq4Nl>(exact, host, x, pairs),
        Fmt::Q2_0 => with_down::<BlockQ2_0>(exact, host, x, pairs),
        Fmt::Q8_0 => with_down::<BlockQ8_0>(exact, host, x, pairs),
        Fmt::Q5_0 => with_down::<BlockQ5_0>(exact, host, x, pairs),
    })
}

/// [`host_pairs`] over typed stacks: gate and up `[experts, inter, dim]`, down
/// `[experts, dim, inter]`.
fn host_rows<G: ExactDot, D: ExactDot>(
    gate: &[G],
    up: &[G],
    down: &[D],
    exact: &Exact,
    x: &[f32],
    pairs: &[(usize, u32, f32)],
) -> Vec<f32> {
    let (dim, inter, limit) = (exact.dim, exact.inter, exact.limit);
    // The activation blocks of an input row and of a row entering the down projection.
    let (nb, nbi) = (dim.div_ceil(BLOCK), inter.div_ceil(BLOCK));
    // The weight blocks of a gate or up row and of a down row.
    let (gb, db) = (dim / G::BLOCK_LEN, inter / D::BLOCK_LEN);
    let rows = x.len() / dim;
    let mut xq = vec![G::Act::zeros(); rows * nb];
    for (r, q) in xq.chunks_exact_mut(nb).enumerate() {
        if pairs.iter().any(|p| p.0 == r) {
            quantize_padded(&x[r * dim..(r + 1) * dim], q);
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
                let at = (e * inter + r) * gb;
                let (gw, uw) = (&gate[at..at + gb], &up[at..at + gb]);
                for &p in &order[lo..hi] {
                    let row = pairs[p].0;
                    let a = &xq[row * nb..(row + 1) * nb];
                    let v = crate::tensor::ops::bounded_swiglu::swiglu(
                        G::dot_exact(gw, a),
                        G::dot_exact(uw, a),
                        limit,
                    );
                    // SAFETY: each (pair, row) is written by the one unit that holds the row of
                    // the pair's expert.
                    unsafe { *at_h.0.add(p * inter + r) = v };
                }
            }
        });
    }
    let mut hq = vec![D::Act::zeros(); n * nbi];
    for (p, q) in hq.chunks_exact_mut(nbi).enumerate() {
        quantize_padded(&h[p * inter..(p + 1) * inter], q);
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
                let at = (e * dim + c) * db;
                let dw = &down[at..at + db];
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

/// `row` as the activation blocks `out`, the last block padded with zeros as the card pads it.
fn quantize_padded<A: BlockFormat>(row: &[f32], out: &mut [A]) {
    if row.len() == out.len() * A::BLOCK_LEN {
        A::quantize(row, out);
    } else {
        let mut padded = row.to_vec();
        padded.resize(out.len() * A::BLOCK_LEN, 0.0);
        A::quantize(&padded, out);
    }
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
            xq: f32s(Exact::act_bytes(exact.gate, rows_max, exact.dim) / 4)?,
            h: f32s(pairs * exact.inter)?,
            hq: f32s(Exact::act_bytes(exact.down, pairs, exact.inter) / 4)?,
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
            f32s(Exact::act_bytes(exact.gate, rows, exact.dim) / 4)?,
            f32s(pairs * exact.inter)?,
            f32s(Exact::act_bytes(exact.down, pairs, exact.inter) / 4)?,
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
        quantize_rows(
            &dev,
            exact.gate,
            address_of::<f32>(&x)?,
            rooms.xq,
            rows,
            exact.dim,
        )?;
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
            host_rows.as_ref().map_or(Ok(0), address_of::<f32>)?,
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
        quantize_rows(
            &self.dev,
            exact.gate,
            address_of::<f32>(x)?,
            rooms.xq,
            rows,
            exact.dim,
        )?;
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
            let exact = Exact::of(&host, 0.0).expect("formats with exact kernels");
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

    /// The pairs of one row worked out one value at a time: the gate and up products of
    /// `ExactDot`, the clamp and SwiGLU of `bounded_swiglu::swiglu` at `limit`, the row as
    /// activation blocks, the down product, the weight.
    fn pairs_one_value_at_a_time<G: ExactDot, D: ExactDot>(
        host: &[Arc<QTensor>; 3],
        dim: usize,
        inter: usize,
        limit: f32,
        x: &[f32],
        pairs: &[(usize, u32, f32)],
    ) -> (Vec<f32>, usize) {
        fn typed<T: BlockFormat>(t: &QTensor) -> &[T] {
            let bytes = t.native_qtensor().data();
            // Safety: the stack holds blocks of `T`, its format.
            unsafe {
                std::slice::from_raw_parts(
                    bytes.as_ptr() as *const T,
                    bytes.len() / std::mem::size_of::<T>(),
                )
            }
        }
        let (gate, up, down) = (
            typed::<G>(&host[0]),
            typed::<G>(&host[1]),
            typed::<D>(&host[2]),
        );
        let (nb, nbi) = (dim / BLOCK, inter / BLOCK);
        let mut out = Vec::with_capacity(pairs.len() * dim);
        let mut bound = 0usize;
        for &(row, e, w) in pairs {
            let e = e as usize;
            let mut xq = vec![G::Act::zeros(); nb];
            G::Act::quantize(&x[row * dim..(row + 1) * dim], &mut xq);
            let h: Vec<f32> = (0..inter)
                .map(|r| {
                    let at = (e * inter + r) * nb;
                    let g = G::dot_exact(&gate[at..at + nb], &xq);
                    let u = G::dot_exact(&up[at..at + nb], &xq);
                    bound += usize::from(g > limit || u.abs() > limit);
                    crate::tensor::ops::bounded_swiglu::swiglu(g, u, limit)
                })
                .collect();
            let mut hq = vec![D::Act::zeros(); nbi];
            D::Act::quantize(&h, &mut hq);
            for c in 0..dim {
                let at = (e * dim + c) * nbi;
                out.push(w * D::dot_exact(&down[at..at + nbi], &hq));
            }
        }
        (out, bound)
    }

    /// The rows of `pairs` in the steps of each format's own dot: the input row and the row
    /// entering the down projection quantised to the format's activation blocks.
    fn pairs_through_the_formats_own_dots<G: BlockFormat, D: BlockFormat>(
        host: &[Arc<QTensor>; 3],
        dim: usize,
        inter: usize,
        x: &[f32],
        pairs: &[(usize, u32, f32)],
    ) -> Vec<f32> {
        fn typed<T: BlockFormat>(t: &QTensor) -> &[T] {
            let bytes = t.native_qtensor().data();
            // Safety: the stack holds blocks of `T`, its format.
            unsafe {
                std::slice::from_raw_parts(
                    bytes.as_ptr() as *const T,
                    bytes.len() / std::mem::size_of::<T>(),
                )
            }
        }
        fn quantized<A: BlockFormat>(v: &[f32]) -> Vec<A> {
            let mut out = vec![A::zeros(); v.len() / A::BLOCK_LEN];
            A::quantize(v, &mut out);
            out
        }
        let (gate, up, down) = (
            typed::<G>(&host[0]),
            typed::<G>(&host[1]),
            typed::<D>(&host[2]),
        );
        let (gb, db) = (dim / G::BLOCK_LEN, inter / D::BLOCK_LEN);
        let mut out = Vec::with_capacity(pairs.len() * dim);
        for &(row, e, w) in pairs {
            let e = e as usize;
            let xa = quantized::<G::ActivationBlock>(&x[row * dim..(row + 1) * dim]);
            let h: Vec<f32> = (0..inter)
                .map(|r| {
                    let at = (e * inter + r) * gb;
                    let g = G::dot(&gate[at..at + gb], &xa);
                    let u = G::dot(&up[at..at + gb], &xa);
                    crate::tensor::ops::bounded_swiglu::swiglu(g, u, 0.0)
                })
                .collect();
            let ha = quantized::<D::ActivationBlock>(&h);
            for c in 0..dim {
                let at = (e * dim + c) * db;
                out.push(w * D::dot(&down[at..at + db], &ha));
            }
        }
        out
    }

    /// The rows of `pairs` from the dequantised weights in f64: the float reference.
    fn pairs_in_floats(
        host: &[Arc<QTensor>; 3],
        dim: usize,
        inter: usize,
        x: &[f32],
        pairs: &[(usize, u32, f32)],
    ) -> Vec<f32> {
        let floats = |t: &QTensor| -> Vec<f32> {
            t.dequantize(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let (gate, up, down) = (floats(&host[0]), floats(&host[1]), floats(&host[2]));
        let dot = |w: &[f32], v: &[f32]| -> f64 {
            w.iter().zip(v).map(|(&a, &b)| a as f64 * b as f64).sum()
        };
        let mut out = Vec::with_capacity(pairs.len() * dim);
        for &(row, e, w) in pairs {
            let e = e as usize;
            let xr = &x[row * dim..(row + 1) * dim];
            let h: Vec<f32> = (0..inter)
                .map(|r| {
                    let at = (e * inter + r) * dim;
                    let g = dot(&gate[at..at + dim], xr);
                    let u = dot(&up[at..at + dim], xr);
                    (g / (1.0 + (-g).exp()) * u) as f32
                })
                .collect();
            for c in 0..dim {
                let at = (e * dim + c) * inter;
                out.push((w as f64 * dot(&down[at..at + inter], &h)) as f32);
            }
        }
        out
    }

    /// The largest and the root mean square distance of `got` from `want`, and the largest
    /// magnitude of `want`.
    fn distance(got: &[f32], want: &[f32]) -> (f32, f32, f32) {
        let worst = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let squares: f64 = got
            .iter()
            .zip(want)
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum();
        let rms = (squares / want.len().max(1) as f64).sqrt() as f32;
        (
            worst,
            rms,
            want.iter().map(|v| v.abs()).fold(0f32, f32::max),
        )
    }

    /// The exact steps are no farther from the float reference than each format's own dot, but
    /// for a rounding margin of 1e-4 of the output's peak: the layer's output summed in the
    /// router's order, the weights quantised from values.
    ///
    /// The exact steps read the codes and scales of the format's own activation.
    /// They differ from its own dot by the order of the float steps only.
    /// An output takes at most `dim / 32 + inter / 32 + 4` rounded steps, each off by 2^-24 of
    /// the terms it sums, about 2e-6 of their absolute sum here.
    /// The margin holds that while the terms cancel by less than 50 times.
    #[test]
    fn the_exact_steps_are_as_accurate_as_the_formats_own_dots() {
        fn compare<G: ExactDot, D: ExactDot>(gate_dt: GgmlDType, down_dt: GgmlDType) {
            compare_at::<G, D>(gate_dt, down_dt, 512, 512);
        }
        fn compare_at<G: ExactDot, D: ExactDot>(
            gate_dt: GgmlDType,
            down_dt: GgmlDType,
            dim: usize,
            inter: usize,
        ) {
            let (experts, used, rows) = (8usize, 4usize, 8usize);
            let host = [
                stack(experts, inter, dim, 1, gate_dt),
                stack(experts, inter, dim, 2, gate_dt),
                stack(experts, dim, inter, 3, down_dt),
            ];
            let exact = Exact::of(&host, 0.0).expect("formats with exact kernels");
            let x: Vec<f32> = (0..rows * dim)
                .map(|i| ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 4.0)
                .collect();
            let pairs: Vec<(usize, u32, f32)> = (0..rows)
                .flat_map(|r| {
                    [0usize, 1, 3, 6]
                        .map(|d| (r, ((r + d) % experts) as u32, 0.1 + 0.05 * d as f32))
                })
                .collect();
            let reference =
                sum_in_order(&pairs_in_floats(&host, dim, inter, &x, &pairs), used, dim);
            let exact_rows =
                sum_in_order(&host_pairs(&exact, &host, &x, &pairs).unwrap(), used, dim);
            let own = sum_in_order(
                &pairs_through_the_formats_own_dots::<G, D>(&host, dim, inter, &x, &pairs),
                used,
                dim,
            );
            let (exact_off, exact_rms, peak) = distance(&exact_rows, &reference);
            let (own_off, own_rms, _) = distance(&own, &reference);
            eprintln!(
                "{gate_dt:?} / {down_dt:?} at {dim} by {inter}: peak {peak:.4e}, exact steps off by \
                 {:.4}% (rms {:.4}%), own dots off by {:.4}% (rms {:.4}%)",
                100.0 * exact_off / peak,
                100.0 * exact_rms / peak,
                100.0 * own_off / peak,
                100.0 * own_rms / peak
            );
            assert!(
                exact_off <= own_off + 1e-4 * peak,
                "{gate_dt:?} / {down_dt:?}: the exact steps are off by {exact_off}, the own dots \
                 by {own_off}, of {peak}"
            );
        }
        compare::<BlockQ4K, BlockQ6K>(GgmlDType::Q4K, GgmlDType::Q6K);
        compare::<BlockIq2S, BlockQ6K>(GgmlDType::Iq2S, GgmlDType::Q6K);
        compare::<BlockQ6K, BlockIq2S>(GgmlDType::Q6K, GgmlDType::Iq2S);
        compare::<BlockIq3Xxs, BlockQ6K>(GgmlDType::Iq3Xxs, GgmlDType::Q6K);
        compare::<BlockQ6K, BlockIq3Xxs>(GgmlDType::Q6K, GgmlDType::Iq3Xxs);
        compare::<BlockIq3S, BlockQ6K>(GgmlDType::Iq3S, GgmlDType::Q6K);
        compare::<BlockQ6K, BlockIq3S>(GgmlDType::Q6K, GgmlDType::Iq3S);
        compare::<BlockIq4Nl, BlockQ6K>(GgmlDType::Iq4Nl, GgmlDType::Q6K);
        compare::<BlockQ6K, BlockIq4Nl>(GgmlDType::Q6K, GgmlDType::Iq4Nl);
        compare::<BlockIq2S, BlockIq4Nl>(GgmlDType::Iq2S, GgmlDType::Iq4Nl);
        compare::<BlockIq3S, BlockIq4Nl>(GgmlDType::Iq3S, GgmlDType::Iq4Nl);
        compare::<BlockQ2_0, BlockQ6K>(GgmlDType::Q2_0, GgmlDType::Q6K);
        compare::<BlockQ6K, BlockQ2_0>(GgmlDType::Q6K, GgmlDType::Q2_0);
        compare::<BlockIq3Xxs, BlockQ2_0>(GgmlDType::Iq3Xxs, GgmlDType::Q2_0);
        compare::<BlockQ8_0, BlockQ8_0>(GgmlDType::Q8_0, GgmlDType::Q8_0);
        compare::<BlockQ4K, BlockQ8_0>(GgmlDType::Q4K, GgmlDType::Q8_0);
        compare::<BlockQ5_0, BlockQ5_0>(GgmlDType::Q5_0, GgmlDType::Q5_0);
        compare::<BlockQ4K, BlockQ5_0>(GgmlDType::Q4K, GgmlDType::Q5_0);
        // The widths of a released file whose expert width is whole runs, not whole blocks.
        compare_at::<BlockQ4K, BlockQ8_0>(GgmlDType::Q4K, GgmlDType::Q8_0, 2560, 640);
        compare_at::<BlockQ4K, BlockQ5_0>(GgmlDType::Q4K, GgmlDType::Q5_0, 2560, 640);
    }

    /// The card's q8_K32 rows are the bytes of the host's, zeros, ties and extremes included.
    #[test]
    fn the_card_quantises_q8_k32_rows_as_the_host() {
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let blocks = 12usize;
        let mut x: Vec<f32> = (0..blocks * BLOCK)
            .map(|i| {
                let h = (i as u32).wrapping_mul(2_654_435_761).rotate_left(11);
                ((h >> 8) as f32 / (1 << 24) as f32 - 0.5) * (1.0 + (i % 7) as f32)
            })
            .collect();
        // A run of zeros, a run of one value, and a run whose values sit on rounding ties.
        x[..32].fill(0.0);
        x[32..64].fill(-3.25);
        for (j, v) in x[64..96].iter_mut().enumerate() {
            *v = (j as f32 - 16.0) * 0.5;
        }
        x[96] = 1e30;
        let mut want = vec![BlockQ8K32::zeros(); blocks];
        BlockQ8K32::quantize(&x, &mut want);
        // Safety: the blocks are `repr(C)` plain data, read as the bytes they occupy.
        let want = unsafe {
            std::slice::from_raw_parts(want.as_ptr() as *const u8, std::mem::size_of_val(&want[..]))
        }
        .to_vec();
        let xs = Tensor::from_vec(x, blocks * BLOCK, &card).unwrap();
        let out = Tensor::zeros_on(want.len() / 4, crate::tensor::DType::F32, &card).unwrap();
        let fmt = Fmt::Iq2S;
        quantize_rows(
            &card,
            fmt,
            address_of::<f32>(&xs).unwrap(),
            address_of::<f32>(&out).unwrap(),
            blocks,
            BLOCK,
        )
        .unwrap();
        let got: Vec<u8> = out
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        assert_eq!(got, want);
        // Rows of 640 values: 2 whole blocks and a last block of 4 runs, padded with zeros.
        let (rows, len) = (3usize, 640usize);
        let x: Vec<f32> = (0..rows * len)
            .map(|i| ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 3.0)
            .collect();
        let per_row = len.div_ceil(BLOCK);
        let mut want = vec![BlockQ8K32::zeros(); rows * per_row];
        for (r, q) in want.chunks_exact_mut(per_row).enumerate() {
            quantize_padded(&x[r * len..(r + 1) * len], q);
        }
        // Safety: the blocks are `repr(C)` plain data, read as the bytes they occupy.
        let want = unsafe {
            std::slice::from_raw_parts(want.as_ptr() as *const u8, std::mem::size_of_val(&want[..]))
        }
        .to_vec();
        let xs = Tensor::from_vec(x, rows * len, &card).unwrap();
        let out = Tensor::zeros_on(want.len() / 4, crate::tensor::DType::F32, &card).unwrap();
        quantize_rows(
            &card,
            fmt,
            address_of::<f32>(&xs).unwrap(),
            address_of::<f32>(&out).unwrap(),
            rows,
            len,
        )
        .unwrap();
        let got: Vec<u8> = out
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        assert_eq!(got, want, "rows of {len} values");
    }

    /// The card's time per pair of `run_pairs` for formats of about the same size, read through
    /// q8_K and through q8_K32. Prints the times; asserts nothing.
    #[test]
    #[ignore = "a measure: run alone on an idle card"]
    fn time_the_exact_kernels_per_pair() {
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let (experts, dim, inter, k) = (8usize, 2048usize, 1024usize, 8usize);
        for dt in [
            GgmlDType::Q2K,
            GgmlDType::Iq2S,
            GgmlDType::Iq3Xxs,
            GgmlDType::Iq3S,
            GgmlDType::Iq4Nl,
            GgmlDType::Q2_0,
            GgmlDType::Q8_0,
            GgmlDType::Q5_0,
        ] {
            let host = [
                stack(experts, inter, dim, 1, dt),
                stack(experts, inter, dim, 2, dt),
                stack(experts, dim, inter, 3, dt),
            ];
            let exact = Exact::of(&host, 0.0).unwrap();
            let on_card: Vec<QTensor> = host
                .iter()
                .map(|t| experts_range(t, 0, experts, &card).unwrap())
                .collect();
            let stacks = Stacks {
                base: [0, 1, 2].map(|i| on_card[i].device_ptr().unwrap() as u64),
                extra: 0,
                base_slots: experts,
                bytes: [0, 1, 2].map(|i| host[i].native_qtensor().data().len() / experts),
            };
            let f32s = |n: usize| Tensor::zeros_on(n, crate::tensor::DType::F32, &card).unwrap();
            let x = Tensor::from_vec(
                (0..dim)
                    .map(|i| (i % 13) as f32 * 0.1 - 0.6)
                    .collect::<Vec<f32>>(),
                dim,
                &card,
            )
            .unwrap();
            let (xq, h, hq, out) = (
                f32s(Exact::act_bytes(exact.gate, 1, dim) / 4),
                f32s(k * inter),
                f32s(Exact::act_bytes(exact.down, k, inter) / 4),
                f32s(k * dim),
            );
            let rooms = Rooms {
                xq: address_of::<f32>(&xq).unwrap(),
                h: address_of::<f32>(&h).unwrap(),
                hq: address_of::<f32>(&hq).unwrap(),
                out: address_of::<f32>(&out).unwrap(),
            };
            let slots = Tensor::from_vec((0..k as u32).collect::<Vec<u32>>(), k, &Device::Cpu)
                .unwrap()
                .to_device(&card)
                .unwrap();
            let w = f32s(k);
            quantize_rows(
                &card,
                exact.gate,
                address_of::<f32>(&x).unwrap(),
                rooms.xq,
                1,
                dim,
            )
            .unwrap();
            let stream = card.as_cuda_device().unwrap().0.stream().clone();
            let launch = || {
                run_pairs(
                    &card,
                    &exact,
                    &stacks,
                    address_of::<u32>(&slots).unwrap(),
                    0,
                    k,
                    k,
                    address_of::<f32>(&w).unwrap(),
                    rooms,
                )
                .unwrap()
            };
            for _ in 0..20 {
                launch();
            }
            stream.synchronize().unwrap();
            let rounds = 200;
            let t = std::time::Instant::now();
            for _ in 0..rounds {
                launch();
            }
            stream.synchronize().unwrap();
            let per_pair = t.elapsed().as_secs_f64() * 1e6 / (rounds * k) as f64;
            eprintln!(
                "{dt:?}: {per_pair:.2} us per pair, dim {dim}, inter {inter}, {k} pairs a launch"
            );
        }
    }

    /// Under a SwiGLU bound the host's pairs are the bits of the bounded SwiGLU of
    /// `bounded_swiglu::swiglu` taken one value at a time, and differ from the unbounded ones.
    #[test]
    fn a_bounded_swiglu_gives_the_bits_of_the_offload_swiglu() {
        let (experts, dim, inter, rows) = (4usize, 512usize, 256usize, 3usize);
        let host = [
            stack(experts, inter, dim, 1, GgmlDType::Q4K),
            stack(experts, inter, dim, 2, GgmlDType::Q4K),
            stack(experts, dim, inter, 3, GgmlDType::Q6K),
        ];
        let x: Vec<f32> = (0..rows * dim)
            .map(|i| ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 4.0)
            .collect();
        let pairs: Vec<(usize, u32, f32)> = (0..rows)
            .flat_map(|r| {
                [0usize, 2].map(|d| (r, ((r + d) % experts) as u32, 0.3 + 0.1 * d as f32))
            })
            .collect();
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        let free = host_pairs(&Exact::of(&host, 0.0).unwrap(), &host, &x, &pairs).unwrap();
        // A bound below the largest products, so that the clamp moves some of them.
        let limit = 0.5f32;
        let exact = Exact::of(&host, limit).expect("formats with exact kernels");
        let got = host_pairs(&exact, &host, &x, &pairs).unwrap();
        let (want, bound) =
            pairs_one_value_at_a_time::<BlockQ4K, BlockQ6K>(&host, dim, inter, limit, &x, &pairs);
        assert!(bound > 0, "the bound moves no product");
        assert_eq!(bits(&got), bits(&want));
        assert_ne!(bits(&got), bits(&free), "the bound changes nothing");
    }
}
