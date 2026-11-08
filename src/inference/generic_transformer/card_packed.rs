//! A projection held on a card in the form its file stores.
//!
//! The card keeps the packed bytes, never a dense F32 copy:
//! - Q4_K and Q6_K blocks have 3 products, and the card measures which one it runs, see below,
//! - the other GGUF blocks go to the card's quantised matmul, which reads them in place,
//! - block-scaled fp8 and fp4 keep their bytes and ue8m0 scales and expand a tile of rows at a
//!   time into a scratch buffer that one matrix product consumes,
//! - bf16 keeps its 16-bit values and expands a tile the same way,
//! - a dense weight keeps its F32 values, which are its stored form.
//!
//! The scratch of an expansion is bounded by the offload's weight tile.
//!
//! The 3 products of Q4_K and Q6_K blocks (`Route`):
//! - the rows in q8_K blocks and the host's dot steps,
//! - the rows in q8_1 blocks and the card's quantised matmul,
//! - the rows in F32 and tiles of the blocks expanded to F32.
//!
//! A unit is a block type and a shape on one card, shared by every weight of a model that has it.
//! The first product of a unit measures, on its own rows, the error of the host and of each
//! product against the exact one: the blocks decoded to F32 times the F32 rows.
//! A product is admissible when it is no farther from the exact product than the host, up to the
//! rounding of an F32 sum (`no_farther`).
//! The first product of each power of 2 of rows times the admissible products on the card.
//! The fastest admissible product serves that size from then on (`choose`).
//! Products whose times differ by no more than their measured noise are tied.
//! Products are also tied when the faster one does not win the pairs of samples taken back to
//! back beyond chance (`tied`).
//! A tie falls to the product nearest the exact one, so the timing noise never picks the bits.

use crate::tensor::blockscaled::fp8::ScaleFormat;
use crate::tensor::blockscaled::projection::Projection;
use crate::tensor::cuda::{
    alloc_f32, cast_bf16_to_f32, copy_f32, matmul_nt_f32, put_cols, upload_f32, CudaDevice,
    CudaStorage, QUANT_LINEAR_TILE_BYTES,
};
use crate::tensor::gpu_runtime::driver::{CudaEvent, CudaSlice, LaunchConfig, PushKernelArg};
use crate::tensor::quantized::{GgmlDType, QMatMul};
use crate::tensor::{Device, Error, Result, Tensor};

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, MutexGuard};

/// How the card holds the values of a weight.
enum Form {
    /// F32 values.
    Dense(CudaSlice<f32>),
    /// GGUF blocks behind the card's quantised matmul.
    Blocks(Box<QMatMul>),
    /// Q4_K or Q6_K blocks, with the suffix of their mat-vec over q8_K rows.
    Exact(Box<QMatMul>, &'static str),
    /// e4m3 values of the rows of a view, with the ue8m0 scales of the whole weight's tiles.
    Fp8 {
        bytes: CudaSlice<u8>,
        scales: CudaSlice<u8>,
        block: usize,
        row0: usize,
    },
    /// e2m1 nibbles with a ue8m0 scale per `block` values along the input.
    Fp4 {
        nibbles: CudaSlice<u8>,
        scales: CudaSlice<u8>,
        block: usize,
    },
    /// bf16 values.
    Bf16(CudaSlice<half::bf16>),
}

/// A weight `[out, inp]` on a card, in its packed form.
pub(super) struct PackedDev {
    dev: Arc<CudaDevice>,
    form: Form,
    out: usize,
    inp: usize,
    /// Bytes the card holds for the weight.
    bytes: usize,
    /// Bytes of F32 values a scratch tile holds.
    tile_bytes: usize,
    /// The unit whose measured product a Q4_K or Q6_K weight runs.
    unit: Option<Arc<Unit>>,
    /// The host weight, which gives the host's product and the exact one of a measurement.
    host: Option<Projection>,
}

/// A product of Q4_K or Q6_K blocks on a card.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(super) enum Route {
    /// Rows in q8_K blocks, the host's steps.
    HostSteps,
    /// The card's quantised matmul, rows in q8_1 blocks.
    Q81,
    /// Blocks expanded to F32 a tile at a time, rows unquantised.
    F32,
}

impl Route {
    /// Every route, in the order a tie between equal measurements falls to.
    pub(super) const ALL: [Route; 3] = [Route::HostSteps, Route::Q81, Route::F32];

    /// The name a log line gives the route.
    pub(super) fn name(self) -> &'static str {
        match self {
            Route::HostSteps => "q8_k",
            Route::Q81 => "q8_1",
            Route::F32 => "f32",
        }
    }
}

/// The relative rounding of an F32 sum of `terms` terms.
pub(super) fn f32_rounding(terms: usize) -> f64 {
    terms as f64 * f32::EPSILON as f64
}

/// Whether a product at relative L2 distance `card` from the exact product is no farther than
/// the host at `host`, up to the rounding of an F32 sum of `terms` terms.
pub(super) fn no_farther(host: f64, card: f64, terms: usize) -> bool {
    card <= host + f32_rounding(terms)
}

/// The relative L2 distance of `got` to `exact`.
fn relative_l2(exact: &[f32], got: &[f32]) -> f64 {
    let (mut dd, mut ww) = (0f64, 0f64);
    for (&w, &g) in exact.iter().zip(got) {
        let (w, g) = (w as f64, g as f64);
        dd += (w - g) * (w - g);
        ww += w * w;
    }
    (dd / ww).sqrt()
}

/// The middle value of `v`, the lower of the 2 middle ones for an even count.
///
/// The lower one keeps a slow first sample, a kernel's first load, out of a median of 2.
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[(s.len() - 1) / 2]
}

/// The median distance of the samples `v` to their median, 0 for no sample.
///
/// A sample far off, a scratch buffer's first allocation, moves it by at most 1 rank.
pub(super) fn spread(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let m = median(v);
    let off: Vec<f64> = v.iter().map(|s| (s - m).abs()).collect();
    median(&off)
}

/// The measured time of one route over one size of rows.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Timing {
    pub(super) route: Route,
    /// Median microseconds of the samples.
    pub(super) median: f64,
    /// Median distance of the samples to their median, in microseconds.
    pub(super) spread: f64,
    /// Samples taken.
    pub(super) samples: usize,
    /// The samples in microseconds, in the order taken.
    ///
    /// Sample `i` of every route of a size is taken in the same round, back to back.
    pub(super) values: Vec<f64>,
}

impl Timing {
    /// The timing of `route` from its samples in microseconds, in the order taken.
    pub(super) fn of(route: Route, samples: &[f64]) -> Self {
        Self {
            route,
            median: median(samples),
            spread: spread(samples),
            samples: samples.len(),
            values: samples.to_vec(),
        }
    }
}

/// The timing noise between 2 measured routes: the clock's `tick` plus the spread of each.
pub(super) fn noise(a: &Timing, b: &Timing, tick: f64) -> f64 {
    tick + a.spread + b.spread
}

/// The pairs of samples taken back to back where `a` is faster than `b` by more than the clock's
/// `tick`, and the pairs compared.
///
/// The first round is left out: it holds each product's first run, a kernel's first load and a
/// scratch buffer's first allocation.
/// A timing of 1 sample is its own pair.
pub(super) fn pairs_won(a: &Timing, b: &Timing, tick: f64) -> (usize, usize) {
    let n = a.values.len().min(b.values.len());
    let skip = usize::from(n > 1);
    let won = a
        .values
        .iter()
        .zip(&b.values)
        .skip(skip)
        .filter(|(x, y)| **x + tick < **y)
        .count();
    (won, n - skip)
}

/// Whether `won` pairs of `n` are more than equal routes give by chance.
///
/// Equal routes win each pair with a chance of a half.
/// The count is beyond chance when equal routes win at least as many pairs with a chance below
/// 1 in `n`: the chance that a given one of the `n` pairs is the slowest.
pub(super) fn beyond_chance(won: usize, n: usize) -> bool {
    if n == 0 || won == 0 {
        return false;
    }
    // The chance of at least `won` wins of `n`, each at a half.
    let mut term = 0.5f64.powi(n as i32);
    let mut tail = 0.0;
    for j in 0..=n {
        if j >= won {
            tail += term;
        }
        term *= (n - j) as f64 / (j + 1) as f64;
    }
    tail * (n as f64) < 1.0
}

/// Whether route `t` is tied with the fastest route `lead`.
///
/// They are tied when their medians differ by no more than their `noise`, or when `lead` does
/// not win its pairs with `t` beyond chance.
pub(super) fn tied(lead: &Timing, t: &Timing, tick: f64) -> bool {
    let (won, n) = pairs_won(lead, t, tick);
    t.median - lead.median <= noise(lead, t, tick) || !beyond_chance(won, n)
}

/// The errors of a unit's products against the exact product, over rows of its first product.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Errors {
    /// Relative L2 distance of the host's product.
    pub(super) host: f64,
    /// Relative L2 distance of each route's product.
    pub(super) routes: Vec<(Route, f64)>,
    /// Rows the distances are taken over.
    pub(super) rows: usize,
    /// Terms of each output's sum: the input width.
    pub(super) terms: usize,
}

impl Errors {
    /// The routes no farther from the exact product than the host.
    pub(super) fn admissible(&self) -> Vec<Route> {
        self.routes
            .iter()
            .filter(|(_, e)| no_farther(self.host, *e, self.terms))
            .map(|(r, _)| *r)
            .collect()
    }

    fn of(&self, route: Route) -> f64 {
        self.routes
            .iter()
            .find(|(r, _)| *r == route)
            .map_or(f64::INFINITY, |(_, e)| *e)
    }
}

/// The route of one size of rows, and the timings behind it.
#[derive(Clone, Debug)]
pub(super) struct Decision {
    pub(super) route: Route,
    pub(super) times: Vec<Timing>,
    /// The clock's tick in microseconds the times settled within, 0 when nothing was timed.
    pub(super) tick: f64,
}

/// The route `errors` and `times` give: the admissible route nearest the exact product among
/// those tied with the fastest admissible one.
///
/// A route is [`tied`] with the fastest when its median exceeds the fastest's by no more than the
/// `noise` of the 2, the clock's `tick` included, or when the fastest does not win the pairs of
/// samples taken back to back beyond chance.
/// Equal errors fall to the order of `Route::ALL`.
/// With no admissible route, the smallest error wins.
/// A route without a time counts as the slowest, which only matters when it is alone.
pub(super) fn choose(errors: &Errors, times: &[Timing], tick: f64) -> Route {
    let time = |r: Route| times.iter().find(|t| t.route == r);
    let by_error = |a: &Route, b: &Route| errors.of(*a).total_cmp(&errors.of(*b)).then(a.cmp(b));
    let admissible = errors.admissible();
    if admissible.is_empty() {
        return errors
            .routes
            .iter()
            .map(|(r, _)| *r)
            .min_by(by_error)
            .unwrap_or(Route::HostSteps);
    }
    let median = |r: Route| time(r).map_or(f64::INFINITY, |t| t.median);
    let fastest = admissible
        .iter()
        .copied()
        .min_by(|a, b| median(*a).total_cmp(&median(*b)).then(by_error(a, b)))
        .unwrap_or(Route::HostSteps);
    let Some(lead) = time(fastest) else {
        return fastest;
    };
    admissible
        .into_iter()
        .filter(|&r| time(r).is_some_and(|t| tied(lead, t, tick)))
        .min_by(by_error)
        .unwrap_or(fastest)
}

/// What a unit has measured.
#[derive(Default)]
struct Measured {
    errors: Option<Errors>,
    /// The decision of each power of 2 of rows.
    sizes: BTreeMap<usize, Decision>,
}

/// A block type and a shape on one card: the weights that share a measured product.
pub(super) struct Unit {
    dtype: GgmlDType,
    measured: Mutex<Measured>,
}

impl Unit {
    fn measured(&self) -> MutexGuard<'_, Measured> {
        self.measured.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The units of the card weights of one model.
#[derive(Default)]
pub(super) struct Choices {
    units: Mutex<HashMap<(usize, GgmlDType, usize, usize), Arc<Unit>>>,
}

impl Choices {
    /// The unit of `dtype` blocks `[out, inp]` on card `ordinal`.
    fn unit(&self, ordinal: usize, dtype: GgmlDType, out: usize, inp: usize) -> Arc<Unit> {
        let mut units = self.units.lock().unwrap_or_else(|e| e.into_inner());
        units
            .entry((ordinal, dtype, out, inp))
            .or_insert_with(|| {
                Arc::new(Unit {
                    dtype,
                    measured: Mutex::default(),
                })
            })
            .clone()
    }
}

/// Microseconds between 2 timed events on `dev`'s stream around `work`.
fn timed(
    dev: &CudaDevice,
    (start, end): (&CudaEvent, &CudaEvent),
    work: impl FnOnce() -> Result<()>,
) -> Result<f64> {
    let event = |e| Error::msg(format!("product timing: {e:?}"));
    start.record(dev.stream()).map_err(event)?;
    work()?;
    end.record(dev.stream()).map_err(event)?;
    end.synchronize().map_err(event)?;
    Ok(start.elapsed_ms(end).map_err(event)? as f64 * 1e3)
}

/// The shortest time in microseconds the events of `dev` report between 2 records with nothing
/// in between.
///
/// Rounds of doubling size run until one does not lower it.
fn clock_tick(dev: &CudaDevice, events: (&CudaEvent, &CudaEvent)) -> Result<f64> {
    let mut least = f64::INFINITY;
    let mut round = 1usize;
    loop {
        let mut lowered = false;
        for _ in 0..round {
            let t = timed(dev, events, || Ok(()))?;
            if t < least {
                least = t;
                lowered = true;
            }
        }
        if !lowered {
            return Ok(least);
        }
        round *= 2;
    }
}

/// Whether medians `now` after a round, `before` it, are settled within the clock's `tick`.
///
/// Settled means every median moved by no more than `tick`, or the fastest one sits below every
/// other by more than both their moves and `tick`.
fn settled(before: &[f64], now: &[f64], tick: f64) -> bool {
    let moved: Vec<f64> = before.iter().zip(now).map(|(a, b)| (a - b).abs()).collect();
    if moved.iter().all(|&m| m <= tick) {
        return true;
    }
    let lead = (0..now.len())
        .min_by(|&a, &b| now[a].total_cmp(&now[b]))
        .unwrap_or(0);
    (0..now.len()).all(|k| k == lead || now[k] - now[lead] > moved[k] + moved[lead] + tick)
}

fn upload_u8(dev: &CudaDevice, what: &str, v: &[u8]) -> Result<CudaSlice<u8>> {
    dev.stream()
        .clone_htod(v)
        .map_err(|e| Error::msg(format!("{what} upload: {e}")))
}

fn launch_err(name: &str, e: crate::tensor::gpu_runtime::driver::DriverError) -> Error {
    Error::msg(format!("{name} launch: {e}"))
}

/// Values of a q8_K block.
const Q8K_VALUES: usize = 256;
/// Bytes of a q8_K block: the f32 scale, the codes and the sums of 16 codes.
const Q8K_BYTES: usize = 4 + Q8K_VALUES + Q8K_VALUES / 8;
/// Activation rows a mat-vec warp serves with one read of a weight block.
const EXACT_ROWS: usize = 4;

/// The suffix of the mat-vec kernel that takes the host's steps for `dtype` over rows of `inp`
/// values, `None` for a type the card's quantised matmul serves.
fn exact_kernel(dtype: GgmlDType, inp: usize) -> Option<&'static str> {
    if !inp.is_multiple_of(Q8K_VALUES) {
        return None;
    }
    match dtype {
        GgmlDType::Q4K => Some("px_matvec_q4_k"),
        GgmlDType::Q6K => Some("px_matvec_q6_k"),
        _ => None,
    }
}

/// The kernel that expands `dtype` blocks to F32 values and the threads it takes a block with,
/// `None` for a type the F32 route does not serve.
fn f32_kernel(dtype: GgmlDType) -> Option<(&'static str, u32)> {
    match dtype {
        GgmlDType::Q4K => Some(("dequantize_block_q4_K_f32", 32)),
        GgmlDType::Q6K => Some(("dequantize_block_q6_K_f32", 64)),
        _ => None,
    }
}

fn grid(total: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (total.div_ceil(256).max(1) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl PackedDev {
    /// `p` uploaded as stored, a Q4_K or Q6_K weight in its unit of `choices`.
    pub(super) fn new(dev: &Arc<CudaDevice>, p: &Projection, choices: &Choices) -> Result<Self> {
        let (out, inp) = match p.dims().as_slice() {
            [out, inp] => (*out, *inp),
            _ => return Err(Error::msg("a card mixer weight is a matrix")),
        };
        let mut unit = None;
        let (form, bytes) = match p {
            Projection::Quant(q) => {
                let on_card = q.to_device(&Device::Cuda(dev.clone()))?;
                let qmm = Box::new(QMatMul::from_arc(Arc::new(on_card))?);
                let form = match (exact_kernel(q.dtype(), inp), f32_kernel(q.dtype())) {
                    (Some(kernel), Some((expand, _))) => {
                        dev.quantized_fn("cx_quantize_q8_k")?;
                        dev.quantized_fn(kernel)?;
                        dev.quantized_fn(expand)?;
                        unit = Some(choices.unit(dev.ordinal(), q.dtype(), out, inp));
                        Form::Exact(qmm, kernel)
                    }
                    _ => Form::Blocks(qmm),
                };
                (form, p.bytes())
            }
            Projection::Fp8(f) if f.scale_format == ScaleFormat::E8m0 => {
                // The scales of the tile rows the view touches, addressed from the first of them.
                let per_row = f.inp.div_ceil(f.block);
                let first = f.row0() / f.block;
                let last = (f.row0() + f.out).div_ceil(f.block);
                let scales = &f.scales.as_slice()[first * per_row..last * per_row];
                (
                    Form::Fp8 {
                        bytes: upload_u8(dev, "fp8 weight", f.view_bytes())?,
                        scales: upload_u8(dev, "fp8 scales", scales)?,
                        block: f.block,
                        row0: f.row0() - first * f.block,
                    },
                    f.view_bytes().len() + scales.len(),
                )
            }
            Projection::Fp4(f) => (
                Form::Fp4 {
                    nibbles: upload_u8(dev, "fp4 weight", f.nibbles.as_slice())?,
                    scales: upload_u8(dev, "fp4 scales", f.scales.as_slice())?,
                    block: inp / (f.scales.len() / out),
                },
                p.bytes(),
            ),
            Projection::Bf16(f) => {
                let values: Vec<half::bf16> = f
                    .bytes
                    .as_slice()
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                let up = dev
                    .stream()
                    .clone_htod(&values)
                    .map_err(|e| Error::msg(format!("bf16 weight upload: {e}")))?;
                (Form::Bf16(up), p.bytes())
            }
            // The values of a dense weight, or of fp8 scales stored as F32, are F32 already.
            _ => {
                let values = p.dense_f32()?;
                (
                    Form::Dense(upload_f32(dev, "mixer weight", &values)?),
                    values.len() * std::mem::size_of::<f32>(),
                )
            }
        };
        let host = unit.is_some().then(|| p.clone());
        Ok(Self {
            dev: dev.clone(),
            form,
            out,
            inp,
            bytes,
            tile_bytes: QUANT_LINEAR_TILE_BYTES,
            unit,
            host,
        })
    }

    /// The weight with scratch tiles of `bytes`.
    #[cfg(test)]
    pub(super) fn with_tile_bytes(mut self, bytes: usize) -> Self {
        self.tile_bytes = bytes;
        self
    }

    pub(super) fn out(&self) -> usize {
        self.out
    }

    /// Bytes the card holds for the weight.
    pub(super) fn bytes(&self) -> usize {
        self.bytes
    }

    /// Rows of the weight a scratch tile holds.
    fn tile_rows(&self) -> usize {
        (self.tile_bytes / (self.inp * std::mem::size_of::<f32>()).max(1)).clamp(1, self.out)
    }

    /// Rows `r0..r0 + rows` of the weight expanded to F32, `[rows, inp]`.
    fn expand(&self, r0: usize, rows: usize) -> Result<CudaSlice<f32>> {
        let dev: &CudaDevice = &self.dev;
        let inp = self.inp;
        let total = rows * inp;
        match &self.form {
            Form::Fp8 {
                bytes,
                scales,
                block,
                row0,
            } => {
                let w = alloc_f32(dev, "fp8 tile", total)?;
                let func = dev.quantized_fn("dequant_fp8_tiles")?;
                let args = [
                    inp as i64,
                    *block as i64,
                    (row0 + r0) as i64,
                    inp.div_ceil(*block) as i64,
                    total as i64,
                ];
                let view = bytes.slice(r0 * inp..(r0 + rows) * inp);
                let mut b = dev.stream().launch_builder(&func);
                b.arg(&view);
                b.arg(scales);
                b.arg(&w);
                for a in &args {
                    b.arg(a);
                }
                unsafe { b.launch(grid(total)) }.map_err(|e| launch_err("dequant_fp8_tiles", e))?;
                Ok(w)
            }
            Form::Fp4 {
                nibbles,
                scales,
                block,
            } => {
                let w = alloc_f32(dev, "fp4 tile", total)?;
                let func = dev.quantized_fn("dequant_fp4_e8m0")?;
                let per_row = inp / block;
                let nib = nibbles.slice(r0 * inp / 2..(r0 + rows) * inp / 2);
                let sc = scales.slice(r0 * per_row..(r0 + rows) * per_row);
                let (inp_i, block_i, total_i) = (inp as i64, *block as i64, total as i64);
                let mut b = dev.stream().launch_builder(&func);
                b.arg(&nib);
                b.arg(&sc);
                b.arg(&w);
                b.arg(&inp_i);
                b.arg(&block_i);
                b.arg(&total_i);
                unsafe { b.launch(grid(total)) }.map_err(|e| launch_err("dequant_fp4_e8m0", e))?;
                Ok(w)
            }
            Form::Bf16(values) => {
                let tile = values.slice(r0 * inp..(r0 + rows) * inp);
                let own = {
                    let mut c = dev
                        .stream()
                        .alloc_zeros::<half::bf16>(total)
                        .map_err(|e| Error::msg(format!("bf16 tile: {e}")))?;
                    dev.stream()
                        .memcpy_dtod(&tile, &mut c.slice_mut(0..total))
                        .map_err(|e| Error::msg(format!("bf16 tile copy: {e}")))?;
                    c
                };
                cast_bf16_to_f32(dev, &own, total)
            }
            Form::Exact(qmm, _) => {
                let (blob, dtype, _, _) = qmm
                    .card_blocks()
                    .ok_or_else(|| Error::msg("a card weight holds its blocks"))?;
                let (kernel, threads) = f32_kernel(dtype)
                    .ok_or_else(|| Error::msg("no F32 expansion for this type"))?;
                // A row of the weight is its blocks along the input, contiguous in the blob.
                let row_bytes = inp / dtype.block_size() * dtype.type_size();
                let view = blob.slice(r0 * row_bytes..(r0 + rows) * row_bytes);
                let w = alloc_f32(dev, "k-quant tile", total)?;
                let func = dev.quantized_fn(kernel)?;
                let mut b = dev.stream().launch_builder(&func);
                b.arg(&view);
                b.arg(&w);
                let cfg = LaunchConfig {
                    grid_dim: ((total / dtype.block_size()) as u32, 1, 1),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { b.launch(cfg) }.map_err(|e| launch_err(kernel, e))?;
                Ok(w)
            }
            Form::Dense(_) | Form::Blocks(_) => Err(Error::msg("this form expands no tile")),
        }
    }

    /// `x` `[rows, inp]` through Q4_K or Q6_K blocks the way the host does it.
    ///
    /// The rows become q8_K blocks with the host's quantiser, then each output is the host's
    /// dot of a weight row against a row: the integer sums of a block, the fixed float step that
    /// folds them, and the sum over the blocks in order.
    fn apply_exact(
        &self,
        qmm: &QMatMul,
        kernel: &str,
        x: &CudaSlice<f32>,
        rows: usize,
    ) -> Result<CudaSlice<f32>> {
        let dev: &CudaDevice = &self.dev;
        let (out, inp) = (self.out, self.inp);
        let (blob, _, n, k) = qmm
            .card_blocks()
            .ok_or_else(|| Error::msg("a card weight holds its blocks"))?;
        if (n, k) != (out, inp) {
            return Err(Error::msg("the card blocks are not the weight's shape"));
        }
        let nsb = inp / Q8K_VALUES;
        let blocks = rows * nsb;
        let xq = dev
            .stream()
            .alloc_zeros::<u8>(blocks * Q8K_BYTES)
            .map_err(|e| Error::msg(format!("q8_K rows: {e}")))?;
        let quantize = dev.quantized_fn("cx_quantize_q8_k")?;
        let xv = x.slice(0..rows * inp);
        let count = blocks as i64;
        let mut b = dev.stream().launch_builder(&quantize);
        b.arg(&xv);
        b.arg(&xq);
        b.arg(&count);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(cfg) }.map_err(|e| launch_err("cx_quantize_q8_k", e))?;

        let y = alloc_f32(dev, "mixer product", rows * out)?;
        let func = dev.quantized_fn(kernel)?;
        let (nsb_i, n_i, rows_i) = (nsb as i32, out as i32, rows as i32);
        let mut b = dev.stream().launch_builder(&func);
        b.arg(blob);
        b.arg(&xq);
        b.arg(&y);
        b.arg(&nsb_i);
        b.arg(&n_i);
        b.arg(&rows_i);
        let cfg = LaunchConfig {
            grid_dim: (out as u32, rows.div_ceil(EXACT_ROWS) as u32, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(cfg) }.map_err(|e| launch_err(kernel, e))?;
        Ok(y)
    }

    /// The product of the card's quantised matmul, which quantises the rows in q8_1 blocks.
    pub(super) fn apply_q8_1(&self, x: &CudaSlice<f32>, rows: usize) -> Result<CudaSlice<f32>> {
        let dev: &CudaDevice = &self.dev;
        let (out, inp) = (self.out, self.inp);
        let (Form::Blocks(qmm) | Form::Exact(qmm, _)) = &self.form else {
            return Err(Error::msg("this form has no quantised matmul"));
        };
        let xt = Tensor::from_cuda_storage(
            CudaStorage::F32(copy_f32(dev, "mixer input", &x.slice(0..rows * inp))?),
            self.dev.clone(),
            (rows, inp),
        )?;
        let y = qmm.forward(&xt)?;
        let (ys, _) = y.cuda_f32_slice()?;
        copy_f32(dev, "mixer product", &ys.slice(0..rows * out))
    }

    /// `x` `[rows, inp]` through tiles of the weight expanded to F32, one cuBLAS product a tile.
    fn apply_tiles(&self, x: &CudaSlice<f32>, rows: usize) -> Result<CudaSlice<f32>> {
        let dev: &CudaDevice = &self.dev;
        let (out, inp) = (self.out, self.inp);
        let tile = self.tile_rows();
        if tile >= out {
            let w = self.expand(0, out)?;
            return matmul_nt_f32(dev, x, &w, 1, rows, inp, out);
        }
        let mut y = alloc_f32(dev, "mixer product", rows * out)?;
        for r0 in (0..out).step_by(tile) {
            let n = tile.min(out - r0);
            let w = self.expand(r0, n)?;
            let part = matmul_nt_f32(dev, x, &w, 1, rows, inp, n)?;
            put_cols(dev, &part, &mut y, (rows, out), (r0, n))?;
        }
        Ok(y)
    }

    /// `x` `[rows, inp]` through Q4_K or Q6_K blocks by `route`.
    fn product(
        &self,
        route: Route,
        qmm: &QMatMul,
        kernel: &str,
        x: &CudaSlice<f32>,
        rows: usize,
    ) -> Result<CudaSlice<f32>> {
        match route {
            Route::HostSteps => self.apply_exact(qmm, kernel, x, rows),
            Route::Q81 => self.apply_q8_1(x, rows),
            Route::F32 => self.apply_tiles(x, rows),
        }
    }

    /// The errors of the host and of every route against the exact product, over rows of `x`.
    ///
    /// The rows are taken evenly spread, 1 then 2, 4 and so on, then all `rows`.
    /// The sample stops growing when no error moves by more than the rounding of an F32 sum
    /// between 2 sizes.
    fn measure_errors(
        &self,
        qmm: &QMatMul,
        kernel: &str,
        x: &CudaSlice<f32>,
        rows: usize,
    ) -> Result<Errors> {
        let (out, inp) = (self.out, self.inp);
        let host = self
            .host
            .as_ref()
            .ok_or_else(|| Error::msg("a measured weight keeps its host form"))?;
        let all = super::card_mixers::download(&self.dev, x, rows * inp)?;
        let exact = Projection::Dense(Tensor::from_vec(
            host.dense_f32()?,
            (out, inp),
            &Device::Cpu,
        )?);
        let flat = |t: Tensor| -> Result<Vec<f32>> { t.flatten_all()?.to_vec1::<f32>() };
        let mut before: Option<Vec<f64>> = None;
        let mut size = 1usize;
        loop {
            let take = size.min(rows);
            let mut xs = Vec::with_capacity(take * inp);
            for i in 0..take {
                let r = i * rows / take;
                xs.extend_from_slice(&all[r * inp..(r + 1) * inp]);
            }
            let xt = Tensor::from_vec(xs.clone(), (take, inp), &Device::Cpu)?;
            let truth = flat(exact.apply_here(&xt)?)?;
            let mut now = vec![relative_l2(&truth, &flat(host.apply_here(&xt)?)?)];
            let up = upload_f32(&self.dev, "measured rows", &xs)?;
            for route in Route::ALL {
                let y = self.product(route, qmm, kernel, &up, take)?;
                let got = super::card_mixers::download(&self.dev, &y, take * out)?;
                now.push(relative_l2(&truth, &got));
            }
            let stable = before.as_ref().is_some_and(|b| {
                b.iter()
                    .zip(&now)
                    .all(|(a, n)| (a - n).abs() <= f32_rounding(inp))
            });
            if stable || take == rows {
                return Ok(Errors {
                    host: now[0],
                    routes: Route::ALL
                        .into_iter()
                        .zip(now[1..].iter().copied())
                        .collect(),
                    rows: take,
                    terms: inp,
                });
            }
            before = Some(now);
            size *= 2;
        }
    }

    /// The median microseconds of each of `routes` over `rows` rows of `x` and its samples, and
    /// the clock's tick.
    ///
    /// The routes are sampled in turn, in rounds that double the samples, until the medians
    /// are `settled` within the clock's tick.
    /// Sample `i` of every route is taken back to back.
    fn time_routes(
        &self,
        qmm: &QMatMul,
        kernel: &str,
        routes: &[Route],
        x: &CudaSlice<f32>,
        rows: usize,
    ) -> Result<(Vec<Timing>, f64)> {
        let dev: &CudaDevice = &self.dev;
        let (start, end) = (dev.new_timed_event()?, dev.new_timed_event()?);
        let events = (&start, &end);
        let tick = clock_tick(dev, events)?;
        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); routes.len()];
        let mut before: Option<Vec<f64>> = None;
        let mut round = 1usize;
        loop {
            for _ in 0..round {
                for (k, &route) in routes.iter().enumerate() {
                    let t = timed(dev, events, || {
                        self.product(route, qmm, kernel, x, rows).map(drop)
                    })?;
                    samples[k].push(t);
                }
            }
            let now: Vec<f64> = samples.iter().map(|s| median(s)).collect();
            if before.as_ref().is_some_and(|b| settled(b, &now, tick)) {
                let times = routes
                    .iter()
                    .zip(&samples)
                    .map(|(&r, s)| Timing::of(r, s))
                    .collect();
                return Ok((times, tick));
            }
            before = Some(now);
            round = samples[0].len();
        }
    }

    /// The route of `rows` rows of a Q4_K or Q6_K weight: the unit's decision for their size,
    /// measured on `x` at the first product of that size.
    fn route(&self, qmm: &QMatMul, kernel: &str, x: &CudaSlice<f32>, rows: usize) -> Result<Route> {
        #[cfg(test)]
        if let Some(r) = route::forced() {
            return Ok(r);
        }
        let Some(unit) = &self.unit else {
            return Ok(Route::HostSteps);
        };
        let size = rows.next_power_of_two();
        let mut measured = unit.measured();
        if let Some(d) = measured.sizes.get(&size) {
            return Ok(d.route);
        }
        let started = std::time::Instant::now();
        let errors = match &measured.errors {
            Some(e) => e.clone(),
            None => {
                let e = self.measure_errors(qmm, kernel, x, rows)?;
                measured.errors = Some(e.clone());
                e
            }
        };
        let admissible = errors.admissible();
        let (times, tick) = if admissible.len() > 1 {
            self.time_routes(qmm, kernel, &admissible, x, rows)?
        } else {
            (Vec::new(), 0.0)
        };
        let route = choose(&errors, &times, tick);
        let shown_errors = errors
            .routes
            .iter()
            .map(|(r, e)| format!("{} {e:.3e}", r.name()))
            .collect::<Vec<_>>()
            .join(", ");
        let decision = Decision { route, times, tick };
        let shown_times = if decision.times.is_empty() {
            "not timed".to_string()
        } else {
            let each = decision
                .times
                .iter()
                .map(|t| {
                    format!(
                        "{} {:.1} us +- {:.1} ({})",
                        t.route.name(),
                        t.median,
                        t.spread,
                        t.samples
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("median time {each}, clock tick {:.1} us", decision.tick)
        };
        tracing::info!(
            "card mixer {} [{}, {}] on card {}, {rows} rows: {} chosen; error to the exact product over {} rows: host {:.3e}, {shown_errors} (bound {:.1e}); {shown_times}; measured in {:.3} s",
            unit.dtype.gguf_name(),
            self.out,
            self.inp,
            self.dev.ordinal(),
            route.name(),
            errors.rows,
            errors.host,
            f32_rounding(errors.terms),
            started.elapsed().as_secs_f64(),
        );
        measured.sizes.insert(size, decision);
        Ok(route)
    }

    /// The errors and the decision of the unit for `rows` rows, `None` before they are measured.
    #[cfg(test)]
    pub(super) fn decision(&self, rows: usize) -> Option<(Errors, Decision)> {
        let measured = self.unit.as_ref()?.measured();
        let d = measured.sizes.get(&rows.next_power_of_two())?.clone();
        Some((measured.errors.clone()?, d))
    }

    /// Whether `self` and `other` share a unit.
    #[cfg(test)]
    pub(super) fn shares_unit(&self, other: &PackedDev) -> bool {
        match (&self.unit, &other.unit) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }

    /// `x` `[rows, inp]` through the weight: `[rows, out]`.
    pub(super) fn apply(&self, x: &CudaSlice<f32>, rows: usize) -> Result<CudaSlice<f32>> {
        let dev: &CudaDevice = &self.dev;
        let (out, inp) = (self.out, self.inp);
        if x.len() < rows * inp {
            return Err(Error::msg(format!(
                "linear: {} inputs for [{rows}, {inp}] x [{out}, {inp}]",
                x.len()
            )));
        }
        match &self.form {
            Form::Dense(w) => matmul_nt_f32(dev, x, w, 1, rows, inp, out),
            Form::Blocks(_) => self.apply_q8_1(x, rows),
            Form::Exact(qmm, kernel) => {
                let route = self.route(qmm, kernel, x, rows)?;
                self.product(route, qmm, kernel, x, rows)
            }
            _ => self.apply_tiles(x, rows),
        }
    }
}

/// The route a test forces on every card weight held in Q4_K or Q6_K blocks, in place of the
/// measured one.
#[cfg(test)]
pub(super) mod route {
    pub(in crate::inference::generic_transformer) use super::Route;
    use std::sync::atomic::{AtomicU8, Ordering};

    /// 0 when no route is forced, else 1 plus the index of the route in `Route::ALL`.
    static FORCED: AtomicU8 = AtomicU8::new(0);

    /// Forces `r`.
    pub(in crate::inference::generic_transformer) fn set(r: Route) {
        let at = Route::ALL.iter().position(|&x| x == r).unwrap_or(0);
        FORCED.store(at as u8 + 1, Ordering::SeqCst);
    }

    /// Gives the measured route back.
    pub(in crate::inference::generic_transformer) fn clear() {
        FORCED.store(0, Ordering::SeqCst);
    }

    pub(super) fn forced() -> Option<Route> {
        match FORCED.load(Ordering::SeqCst) {
            0 => None,
            n => Route::ALL.get(n as usize - 1).copied(),
        }
    }
}
