//! A float weight times a few F32 rows on a card, each weight row read once for all of them.
//!
//! One kernel serves a decode step and the rows of a verify: a row's result does not depend on
//! the rows beside it.
//!
//! A card with the tensor-core path offers several variants of it, which give the same bits.
//! While a model warms up, each weight shape measures them and keeps the fastest for a decode
//! step and for the rows of a verify.

use super::*;
use crate::tensor::kernel_ffi::CudaDType;
use core::ffi::c_void;
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// The most rows one launch of the float product takes: the rows of a verify.
pub const FLOAT_MATVEC_ROWS: usize = crate::inference::engine::decode_step::VERIFY_ROWS;
/// Rows a BF16 product takes through [`float_matvec`] outside a verify.
pub const BF16_MATVEC_ROWS: usize = 8;
/// Weights the kernel reads at a time: a row's length is a multiple of it.
const PER_LOAD: usize = 8;

fn address<T: CudaDType>(storage: &StorageView, what: &str) -> Result<u64> {
    let slice = match storage {
        StorageView::Cuda(c) => c.as_cuda_slice::<T>()?,
        _ => crate::tensor::bail!("float product: {what} must be on CUDA"),
    };
    Ok(slice.device_ptr(slice.stream()).0)
}

/// `x @ w^T` for `x` `[rows, k]` and a BF16, F16 or F32 weight `w` `[n, k]` on a card:
/// `[rows, n]` F32, summed in F32.
/// A card that takes half weights through its tensor cores multiplies each row as 2 parts of the
/// weight's type: its nearest value and what that misses.
/// A row's result does not depend on the rows beside it.
/// `None` for more rows than [`FLOAT_MATVEC_ROWS`], a row length the kernel does not take, or a
/// weight held otherwise.
pub fn float_matvec(x: &Tensor, w: &Tensor) -> Result<Option<Tensor>> {
    product(x, w, None)
}

/// [`float_matvec`] through variant `variant` when given, else through the variant the weight's
/// shape takes.
fn product(x: &Tensor, w: &Tensor, variant: Option<i32>) -> Result<Option<Tensor>> {
    if !w.device().is_cuda() || w.dims().len() != 2 {
        return Ok(None);
    }
    let (n, k) = (w.dims()[0], w.dims()[1]);
    let xd = x.dims();
    if xd.len() != 2 || xd[1] != k || !k.is_multiple_of(PER_LOAD) {
        return Ok(None);
    }
    let rows = xd[0];
    if rows == 0 || rows > FLOAT_MATVEC_ROWS {
        return Ok(None);
    }
    let dev = w.device().as_cuda_device()?;
    let x = x.to_dtype(DType::F32)?.contiguous()?;
    let w = w.contiguous()?;
    let Some(card) = Card::of(&w)? else {
        return Ok(None);
    };
    let variant = match variant {
        Some(v) => v.clamp(0, card.variants - 1),
        None if card.variants > 1 => variants_for(&card, &x, &w)?.of(rows),
        None => 0,
    };
    let y = card.product(&x, variant)?;
    Ok(Some(tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(y, dev.clone()),
        (rows, n),
    )?))
}

/// A float weight on a card, ready for its launches.
struct Card {
    w_ptr: *const c_void,
    dtype: DType,
    n: usize,
    k: usize,
    sms: i32,
    cc: i32,
    ordinal: usize,
    variants: i32,
    device: crate::tensor::Device,
}

impl Card {
    /// The weight `w`, contiguous on a card, or `None` for a weight the kernel does not take.
    fn of(w: &Tensor) -> Result<Option<Self>> {
        let dev = w.device().as_cuda_device()?;
        let (w_st, w_lo) = w.storage_and_layout();
        let w_ptr = match w.dtype() {
            DType::BF16 => {
                address::<half::bf16>(&w_st, "weight")? as usize + w_lo.start_offset() * 2
            }
            DType::F16 => address::<half::f16>(&w_st, "weight")? as usize + w_lo.start_offset() * 2,
            DType::F32 => address::<f32>(&w_st, "weight")? as usize + w_lo.start_offset() * 4,
            _ => return Ok(None),
        } as *const c_void;
        if !(w_ptr as usize).is_multiple_of(16) {
            return Ok(None);
        }
        // The launch follows the card's multiprocessors and compute capability.
        let (sms, cc) = dev.0.mmq_device_info().map_or((1, 0), |i| (i.nsm, i.cc));
        let variants = match w.dtype() {
            DType::F32 => 1,
            _ => unsafe { loken_float_matvec_variants(cc) },
        };
        let (n, k) = (w.dims()[0], w.dims()[1]);
        Ok(Some(Self {
            w_ptr,
            dtype: w.dtype(),
            n,
            k,
            sms,
            cc,
            ordinal: dev.0.ordinal(),
            variants,
            device: w.device(),
        }))
    }

    /// `x @ w^T` for the contiguous F32 rows `x` through variant `variant`, or through the
    /// CUDA-core kernel for a variant below 0.
    fn product(
        &self,
        x: &Tensor,
        variant: i32,
    ) -> Result<crate::tensor::gpu_runtime::driver::CudaSlice<f32>> {
        let dev = self.device.as_cuda_device()?;
        let rows = x.dims()[0];
        let (x_st, x_lo) = x.storage_and_layout();
        let x_ptr = (address::<f32>(&x_st, "activation")? as usize + x_lo.start_offset() * 4)
            as *const c_void;
        if !(x_ptr as usize).is_multiple_of(16) {
            crate::tensor::bail!("float product: activations not 16-byte aligned");
        }
        let y = unsafe { dev.alloc::<f32>(rows * self.n) }?;
        let y_ptr = y.device_ptr(y.stream()).0 as *mut c_void;
        let stream = dev.cuda_stream().cu_stream() as i64;
        let (k, n, rows) = (self.k as i32, self.n as i32, rows as i32);
        let (sms, cc) = (self.sms, self.cc);
        unsafe {
            match self.dtype {
                DType::BF16 => loken_bf16_matvec(
                    x_ptr, self.w_ptr, y_ptr, k, n, rows, sms, cc, variant, stream,
                ),
                DType::F16 => loken_f16_matvec(
                    x_ptr, self.w_ptr, y_ptr, k, n, rows, sms, cc, variant, stream,
                ),
                _ => loken_f32_matvec(x_ptr, self.w_ptr, y_ptr, k, n, rows, sms, cc, stream),
            }
        }
        Ok(y)
    }
}

thread_local! {
    /// Whether this thread measures the variants of the float product for a weight shape it has
    /// not measured yet.
    static CHOOSING: Cell<bool> = const { Cell::new(false) };
}

/// While held, the float product of a weight shape that offers several variants measures them
/// on its first call on this thread, and keeps its choice for every later call.
/// A model's warm-up holds it, so no request pays for the timing.
/// Outside it, a shape not measured yet takes variant 0.
pub struct ChooseFloatVariants {
    previous: bool,
}

thread_local! {
    /// The weight shapes this thread measured while it chose.
    static MEASURED: std::cell::RefCell<Vec<Shape>> = const { std::cell::RefCell::new(Vec::new()) };
}

impl ChooseFloatVariants {
    pub fn enter() -> Self {
        MEASURED.with(|m| m.borrow_mut().clear());
        Self {
            previous: CHOOSING.with(|c| c.replace(true)),
        }
    }

    /// Forgets the choices measured while held, so the next warm-up measures them again: their
    /// timing ran beside another model's work on the card.
    pub fn forget(&self) {
        let shapes = MEASURED.with(|m| std::mem::take(&mut *m.borrow_mut()));
        let mut chosen = chosen().lock().unwrap_or_else(|e| e.into_inner());
        for shape in shapes {
            chosen.remove(&shape);
        }
    }
}

impl Drop for ChooseFloatVariants {
    fn drop(&mut self) {
        CHOOSING.with(|c| c.set(self.previous));
    }
}

/// Whether the float product of a half weight on `device` offers several variants, which a
/// warm-up measures.
pub fn measures_float_variants(device: &crate::tensor::Device) -> bool {
    let Ok(dev) = device.as_cuda_device() else {
        return false;
    };
    dev.0
        .mmq_device_info()
        .is_ok_and(|i| unsafe { loken_float_matvec_variants(i.cc) } > 1)
}

/// A weight shape on a card: the card's ordinal, the weight's rows, its row length and type.
type Shape = (usize, usize, usize, DType);

/// Activation rows a launch takes in 1 group: more launch the kernel that takes a second group.
const GROUP_ROWS: usize = 8;

/// The variants a weight shape takes by the rows of a launch.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FloatVariants {
    /// For 1 row.
    pub one: i32,
    /// For 2 rows up to [`GROUP_ROWS`].
    pub group: i32,
    /// For more rows.
    pub more: i32,
}

impl FloatVariants {
    /// The variant a launch of `rows` rows takes.
    fn of(&self, rows: usize) -> i32 {
        match rows {
            1 => self.one,
            r if r <= GROUP_ROWS => self.group,
            _ => self.more,
        }
    }
}

/// The variants measured for each weight shape.
fn chosen() -> &'static Mutex<HashMap<Shape, FloatVariants>> {
    static CHOSEN: OnceLock<Mutex<HashMap<Shape, FloatVariants>>> = OnceLock::new();
    CHOSEN.get_or_init(Default::default)
}

/// The variants measured for the shape of the weight `w` on its card.
pub fn measured_float_variant(w: &Tensor) -> Option<FloatVariants> {
    let ordinal = w.device().as_cuda_device().ok()?.0.ordinal();
    let (n, k) = w.dims2().ok()?;
    chosen()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&(ordinal, n, k, w.dtype()))
        .copied()
}

/// The variants the weight `w` takes on `card`: the ones measured for its shape, or measured now
/// with the first row of `x` while this thread chooses and the card records no graph, or
/// variant 0.
///
/// The variants give the same bits, so the choice only moves time.
/// Each run is timed as the microbench times it, over copies of a weight smaller than twice the
/// card's second-level cache, after as many calls that leave the card as the run itself does.
/// A measurement that fails, for want of memory on a full card, keeps the default variants.
/// For 1 row, the fastest variant at 1 row is kept.
/// For 2 rows up to [`GROUP_ROWS`], timed at [`GROUP_ROWS`], and for more, timed at the rows of a
/// verify, the variant is the fastest among those not slower than the library product at those
/// rows, or one with more rows a warp that gives up less time to it than it saves on the
/// library; with none not slower, the fastest.
/// The rows of a verify launch 1 build only, so they time 1 variant of each row count.
/// The CUDA-core kernel at 1 row is timed beside them, for the log.
fn variants_for(card: &Card, x: &Tensor, w: &Tensor) -> Result<FloatVariants> {
    let shape = (card.ordinal, card.n, card.k, card.dtype);
    if let Some(v) = chosen()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&shape)
    {
        return Ok(*v);
    }
    if !CHOOSING.with(Cell::get) || crate::tensor::cuda_ext::capture_active(&card.device) {
        return Ok(FloatVariants::default());
    }
    // A measurement that fails, for want of memory on a full card, takes the default variants
    // and keeps no choice: the next warm-up measures again.
    let chosen_now = match measure_variants(card, x, w) {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!(
                "float product {}x{} {:?} on card {}: variants unmeasured, the default taken: {e}",
                card.n,
                card.k,
                card.dtype,
                card.ordinal
            );
            return Ok(FloatVariants::default());
        }
    };
    chosen()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(shape, chosen_now);
    MEASURED.with(|m| m.borrow_mut().push(shape));
    Ok(chosen_now)
}

/// The measurement of [`variants_for`].
fn measure_variants(card: &Card, x: &Tensor, w: &Tensor) -> Result<FloatVariants> {
    let started = Instant::now();
    use crate::tensor::gpu_runtime::driver::sys::CUdevice_attribute;
    let l2 = card
        .device
        .as_cuda_device()?
        .0
        .context()
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
        .map_err(|e| crate::tensor::Error::msg(format!("card cache size: {e:?}")))?
        .max(0) as usize;
    // A weight smaller than twice the cache gets copies that together pass it, freed when the
    // measurement ends, and the calls turn over them.
    // A copy the card has no room for is left out, and the calls turn over fewer.
    let bytes = card.n * card.k * card.dtype.size_in_bytes();
    let mut weights = vec![w.clone()];
    for _ in 1..(2 * l2).div_ceil(bytes.max(1)) {
        match w.affine(1.0, 0.0) {
            Ok(copy) => weights.push(copy),
            Err(e) => {
                tracing::debug!(
                    "float product {}x{} {:?} on card {}: fewer copies to time over: {e}",
                    card.n,
                    card.k,
                    card.dtype,
                    card.ordinal
                );
                break;
            }
        }
    }
    let mut cards = Vec::with_capacity(weights.len());
    for copy in &weights {
        match Card::of(copy)? {
            Some(c) => cards.push(c),
            None => return Ok(FloatVariants::default()),
        }
    }
    let turn = Cell::new(0usize);
    let next = || {
        let i = turn.get();
        turn.set(i + 1);
        i % weights.len()
    };
    let (cards, weights, next) = (&cards, &weights, &next);
    let one = x.narrow(0, 0, 1)?.contiguous()?;
    let group = Tensor::cat(&[&one; GROUP_ROWS], 0)?.contiguous()?;
    let verify = Tensor::cat(&[&one; FLOAT_MATVEC_ROWS], 0)?.contiguous()?;
    let (one, group, verify) = (&one, &group, &verify);
    let variants = card.variants as usize;
    // The rows a warp of each variant, and for each row count the variant a verify launches
    // with it.
    let tiles: Vec<usize> = (0..variants as i32)
        .map(|v| unsafe { loken_float_matvec_tiles(v) }.max(1) as usize)
        .collect();
    let mut verify_variants: Vec<usize> = Vec::new();
    for v in 0..variants {
        match verify_variants.last_mut() {
            Some(last) if tiles[*last] == tiles[v] => *last = v,
            _ => verify_variants.push(v),
        }
    }
    let counts = verify_variants.len();
    // Timed in each round: the CUDA-core kernel at 1 row, each variant at 1 row and at a group's
    // rows, each row count at the verify rows, then the library at a group's rows and at the
    // verify rows.
    let mut runs: Vec<Box<dyn Fn() -> Result<()> + '_>> =
        vec![Box::new(move || cards[next()].product(one, -1).map(drop))];
    for v in 0..variants as i32 {
        runs.push(Box::new(move || cards[next()].product(one, v).map(drop)));
    }
    for v in 0..variants as i32 {
        runs.push(Box::new(move || cards[next()].product(group, v).map(drop)));
    }
    for &v in &verify_variants {
        runs.push(Box::new(move || {
            cards[next()].product(verify, v as i32).map(drop)
        }));
    }
    // The library's product pays for its activations' cast to the weight's type, as a launch
    // that takes it does.
    let half = card.dtype;
    runs.push(Box::new(move || {
        group.to_dtype(half)?.matmul_t(&weights[next()]).map(drop)
    }));
    runs.push(Box::new(move || {
        verify.to_dtype(half)?.matmul_t(&weights[next()]).map(drop)
    }));
    let at_one = |v: usize| 1 + v;
    let at_group = |v: usize| 1 + variants + v;
    let at_verify = |c: usize| 1 + 2 * variants + c;
    let (group_library, verify_library) = (1 + 2 * variants + counts, 2 + 2 * variants + counts);
    let fastest =
        |t: &Timings, at: &dyn Fn(usize) -> usize, of: &mut dyn Iterator<Item = usize>| {
            of.min_by(|a, b| t.median[at(*a)].total_cmp(&t.median[at(*b)]))
        };
    // Among the runs `of` not slower than the library run `library`, the fastest, or a run with
    // more rows a warp that gives up less time to the fastest than it saves on the library.
    // More rows a warp read fewer activations, the traffic that holds a card at its power limit
    // under a long run of verifies.
    // With none not slower than the library, the fastest.
    let choose = |t: &Timings,
                  at: &dyn Fn(usize) -> usize,
                  of: usize,
                  library: usize,
                  rows: &dyn Fn(usize) -> usize| {
        let kept: Vec<usize> = (0..of).filter(|v| !t.slower(at(*v), library)).collect();
        let Some(lead) = fastest(t, at, &mut kept.iter().copied()) else {
            return fastest(t, at, &mut (0..of)).unwrap_or(0);
        };
        let worth = |v: &usize| {
            rows(*v) > rows(lead)
                && t.median[at(*v)] - t.median[at(lead)] <= t.median[library] - t.median[at(*v)]
        };
        let most = kept.iter().copied().filter(worth).map(rows).max();
        match most {
            Some(r) => {
                fastest(t, at, &mut kept.iter().copied().filter(|v| rows(*v) == r)).unwrap_or(lead)
            }
            None => lead,
        }
    };
    // The group's variant, and the verify's row count.
    let group_variant = |t: &Timings| choose(t, &at_group, variants, group_library, &|v| tiles[v]);
    let verify_count = |t: &Timings| choose(t, &at_verify, counts, verify_library, &|c| c);
    // Every variant against the library, and the chosen one against the others, at each width.
    let decided = |t: &Timings| {
        let lead = fastest(t, &at_one, &mut (0..variants)).unwrap_or(0);
        let (g, c) = (group_variant(t), verify_count(t));
        (0..variants).all(|v| {
            t.settled(at_one(lead), at_one(v))
                && t.settled(at_group(v), group_library)
                && t.settled(at_group(g), at_group(v))
        }) && (0..counts).all(|v| {
            t.settled(at_verify(v), verify_library) && t.settled(at_verify(c), at_verify(v))
        })
    };
    let t = timings(&runs, weights.len(), || card.device.synchronize(), decided)?;
    let chosen_now = FloatVariants {
        one: fastest(&t, &at_one, &mut (0..variants)).unwrap_or(0) as i32,
        group: group_variant(&t) as i32,
        more: verify_variants[verify_count(&t)] as i32,
    };
    tracing::debug!(
        "float product {}x{} {:?} on card {}: {chosen_now:?}; faster than the CUDA-core kernel \
         {:?}; medians {:?} us, spreads {:?} us, after {} rounds in {} ms",
        card.n,
        card.k,
        card.dtype,
        card.ordinal,
        (0..variants)
            .filter(|v| t.slower(0, at_one(*v)))
            .collect::<Vec<_>>(),
        t.median
            .iter()
            .map(|m| (m * 10.0).round() / 10.0)
            .collect::<Vec<_>>(),
        t.spread
            .iter()
            .map(|m| (m * 100.0).round() / 100.0)
            .collect::<Vec<_>>(),
        t.rounds,
        started.elapsed().as_millis()
    );
    Ok(chosen_now)
}

/// The median of `v` and the median of its distances to that median.
fn median_and_spread(v: &[f64]) -> (f64, f64) {
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        let h = v.len() / 2;
        if v.len() % 2 == 1 {
            v[h]
        } else {
            (v[h - 1] + v[h]) / 2.0
        }
    };
    let m = median(&mut v.to_vec());
    let s = median(&mut v.iter().map(|x| (x - m).abs()).collect());
    (m, s)
}

/// The median times of some runs, in microseconds, with their standard errors.
struct Timings {
    median: Vec<f64>,
    /// The median distance of a run's times to its median: the spread of a single call.
    spread: Vec<f64>,
    error: Vec<f64>,
    /// The spread of an empty wait on the card: what the host clock resolves.
    floor: f64,
    rounds: usize,
}

impl Timings {
    /// The standard error of the difference between runs `a` and `b`.
    fn gap_error(&self, a: usize, b: usize) -> f64 {
        (self.error[a].powi(2) + self.error[b].powi(2)).sqrt()
    }

    /// Whether run `a` is slower than run `b` by twice the standard error of their difference.
    fn slower(&self, a: usize, b: usize) -> bool {
        self.median[a] - self.median[b] > 2.0 * self.gap_error(a, b)
    }

    /// Whether `a` and `b` are told apart, or tied: their difference is known to within the
    /// spread of a single call of either, or to within what the host clock resolves.
    fn settled(&self, a: usize, b: usize) -> bool {
        let known = 2.0 * self.gap_error(a, b);
        self.slower(a, b)
            || self.slower(b, a)
            || known <= self.spread[a].min(self.spread[b])
            || self.gap_error(a, b) <= self.floor
    }
}

/// The timings of `runs`, taken until `done` holds.
///
/// A round calls each run `2 * burst` times in a row and times the last `burst` calls, then
/// times an empty wait on the card.
/// The first calls leave the card as the run's own earlier calls do.
/// A time spans a call and the wait for it.
/// From the third round on, each median gets a standard error from its spread, and `done`
/// decides whether the rounds stop.
fn timings(
    runs: &[Box<dyn Fn() -> Result<()> + '_>],
    burst: usize,
    sync: impl Fn() -> Result<()>,
    done: impl Fn(&Timings) -> bool,
) -> Result<Timings> {
    // A median's standard error over its median distance, for normal draws, times the root of
    // the sample count.
    const MEDIAN_ERROR: f64 = 1.4826 * 1.2533;
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); runs.len()];
    let mut waits = Vec::new();
    sync()?;
    loop {
        for (run, s) in runs.iter().zip(samples.iter_mut()) {
            for call in 0..2 * burst {
                let at = Instant::now();
                run()?;
                sync()?;
                if call >= burst {
                    s.push(at.elapsed().as_secs_f64() * 1e6);
                }
            }
        }
        let at = Instant::now();
        sync()?;
        waits.push(at.elapsed().as_secs_f64() * 1e6);
        let rounds = waits.len();
        if rounds < 3 {
            continue;
        }
        let stats: Vec<(f64, f64)> = samples.iter().map(|s| median_and_spread(s)).collect();
        let count = (rounds * burst) as f64;
        let t = Timings {
            median: stats.iter().map(|s| s.0).collect(),
            spread: stats.iter().map(|s| s.1).collect(),
            error: stats
                .iter()
                .map(|s| MEDIAN_ERROR * s.1 / count.sqrt())
                .collect(),
            floor: median_and_spread(&waits).1,
            rounds,
        };
        if done(&t) {
            return Ok(t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Device;

    fn values(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let h = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
                (h >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn a_row_equals_its_own_launch_every_variant_and_the_f32_sum() {
        // An odd output count, and a row length that ends inside a step of 32.
        for (n, k) in [(37, 2560), (40, 1000)] {
            for dtype in [DType::BF16, DType::F16, DType::F32] {
                rows_equal_their_own_launch(dtype, n, k);
            }
        }
    }

    /// [`a_row_equals_its_own_launch_every_variant_and_the_f32_sum`] for a weight of `dtype`
    /// `[n, k]`.
    fn rows_equal_their_own_launch(dtype: DType, n: usize, k: usize) {
        let dev = Device::new_cuda(0).expect("a card for the card test");
        let w = Tensor::from_vec(values(n * k, 1), (n, k), &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let wide: Vec<f32> = w
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let x: Vec<f32> = values(FLOAT_MATVEC_ROWS * k, 2);
        let w_card = w.to_device(&dev).unwrap();
        // Each variant, and the variant the shape takes outside a warm-up: variant 0.
        // Every variant gives the bits of the first.
        let count = Card::of(&w_card).unwrap().map_or(1, |c| c.variants);
        let mut first: Option<Vec<Vec<f32>>> = None;
        for variant in (0..count).map(Some).chain([None]) {
            let alone: Vec<Vec<f32>> = (0..FLOAT_MATVEC_ROWS)
                .map(|r| {
                    let xr =
                        Tensor::from_vec(x[r * k..(r + 1) * k].to_vec(), (1, k), &dev).unwrap();
                    let y = product(&xr, &w_card, variant.or(Some(0)))
                        .unwrap()
                        .expect("the kernel takes one row");
                    y.flatten_all().unwrap().to_vec1().unwrap()
                })
                .collect();
            let first = first.get_or_insert_with(|| alone.clone());
            for (r, (a, b)) in alone.iter().zip(first.iter()).enumerate() {
                let same = a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(
                    same,
                    "{dtype:?}: variant {variant:?}, row {r} against variant 0"
                );
            }
            for rows in 1..=FLOAT_MATVEC_ROWS {
                let xs = Tensor::from_vec(x[..rows * k].to_vec(), (rows, k), &dev).unwrap();
                let y: Vec<f32> = product(&xs, &w_card, variant)
                    .unwrap()
                    .expect("the kernel takes these rows")
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap();
                for r in 0..rows {
                    assert_eq!(
                        &y[r * n..(r + 1) * n],
                        &alone[r][..],
                        "{dtype:?}: variant {variant:?}, row {r} of {rows}"
                    );
                }
            }
            for r in 0..FLOAT_MATVEC_ROWS {
                for o in 0..n {
                    let want: f64 = (0..k)
                        .map(|i| f64::from(wide[o * k + i]) * f64::from(x[r * k + i]))
                        .sum();
                    assert!(
                        (f64::from(alone[r][o]) - want).abs() < 1e-3,
                        "{dtype:?}: variant {variant:?}, row {r} output {o}"
                    );
                }
            }
        }
        let over = Tensor::from_vec(
            values((FLOAT_MATVEC_ROWS + 1) * k, 3),
            (FLOAT_MATVEC_ROWS + 1, k),
            &dev,
        )
        .unwrap();
        assert!(float_matvec(&over, &w_card).unwrap().is_none());
    }

    /// Spins for `us` microseconds.
    fn spin(us: f64) -> Result<()> {
        let at = Instant::now();
        while at.elapsed().as_secs_f64() * 1e6 < us {}
        Ok(())
    }

    #[test]
    fn timings_tell_a_slower_run_apart() {
        let runs: Vec<Box<dyn Fn() -> Result<()>>> =
            vec![Box::new(|| spin(600.0)), Box::new(|| spin(200.0))];
        let t = timings(&runs, 1, || Ok(()), |t| t.settled(0, 1)).unwrap();
        assert!(t.slower(0, 1));
        assert!(!t.slower(1, 0));
        assert!(t.rounds >= 3);
    }

    #[test]
    fn timings_of_equal_runs_end_tied() {
        let runs: Vec<Box<dyn Fn() -> Result<()>>> =
            vec![Box::new(|| spin(50.0)), Box::new(|| spin(50.0))];
        let t = timings(&runs, 1, || Ok(()), |t| t.settled(0, 1)).unwrap();
        assert!(t.rounds >= 3);
    }

    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn the_library_product_sums_bf16_operands_in_f32() {
        let dev = Device::new_cuda(0).expect("a card for the card test");
        let (m, n, k) = (300, 37, 2560);
        let bf16 = |v: Vec<f32>, rows| {
            Tensor::from_vec(v, (rows, k), &Device::Cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let (x, w) = (bf16(values(m * k, 4), m), bf16(values(n * k, 5), n));
        let wide = |t: &Tensor| -> Vec<f32> {
            t.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap()
        };
        let (xs, ws) = (wide(&x), wide(&w));
        let y = x
            .to_device(&dev)
            .unwrap()
            .matmul_t_bf16_f32(&w.to_device(&dev).unwrap())
            .unwrap()
            .expect("two BF16 matrices on one card");
        assert_eq!(y.dtype(), DType::F32);
        assert_eq!(y.dims(), &[m, n]);
        let y: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        for r in 0..m {
            for o in 0..n {
                let want: f64 = (0..k)
                    .map(|i| f64::from(xs[r * k + i]) * f64::from(ws[o * k + i]))
                    .sum();
                assert!(
                    (f64::from(y[r * n + o]) - want).abs() < 1e-3,
                    "row {r} output {o}"
                );
            }
        }
    }
}
