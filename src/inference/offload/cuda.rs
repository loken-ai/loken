//! The heavy steps of a forward on a card: every projection product, whatever its storage, and
//! sparse attention, each dequantised or gathered on the device and computed at full precision.
//! The weights every token reads stay on the card once sent; routed experts are kept as the
//! room allows and never traded for one another.

use super::experts::{Expert, ExpertOffload};
use super::projection::Projection;
use super::room::Room;
use super::{AttnBlock, Offload};
use crate::tensor::cuda::{
    gpu_expert_row, gpu_expert_rows_grouped, gpu_expert_rows_grouped_multi, gpu_fp4_linear,
    gpu_fp8_linear, gpu_index_scores, gpu_quant_linear, gpu_sparse_attn, iq2_xxs_tables,
    CudaDevice,
};
use crate::tensor::ops::rms_norm;
use crate::tensor::ops::softmax_last_dim;
use crate::tensor::quantized::{matvec_rows, GgmlDType, QMatMul, QTensor};
use crate::tensor::{Device, Error, Result, Tensor};
use cudarc::driver::CudaSlice;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

const F32: usize = std::mem::size_of::<f32>();

/// Rope the last `rd` elements of each row of `x` [n, hd] on the device, interleaved (GPT-J) as the
/// reference does, leaving the leading `hd - rd` (nope) untouched. `cos`/`sin` are [rd/2] for this
/// position. The same float operations as the host `rope_partial`, composed from tensor ops.
fn rope_partial_dev(x: &Tensor, cos: &Tensor, sin: &Tensor, rd: usize) -> Result<Tensor> {
    let (n, hd) = x.dims2()?;
    let half = rd / 2;
    let nope = hd - rd;
    let head = x.narrow(1, 0, nope)?;
    let tail = x.narrow(1, nope, rd)?.reshape((n, half, 2))?;
    let even = tail.narrow(2, 0, 1)?.reshape((n, half))?;
    let odd = tail.narrow(2, 1, 1)?.reshape((n, half))?;
    let cos = cos.reshape((1, half))?;
    let sin = sin.reshape((1, half))?;
    let ne = even
        .broadcast_mul(&cos)?
        .sub(&odd.broadcast_mul(&sin)?)?
        .reshape((n, half, 1))?;
    let no = even
        .broadcast_mul(&sin)?
        .add(&odd.broadcast_mul(&cos)?)?
        .reshape((n, half, 1))?;
    let rot = Tensor::cat(&[&ne, &no], 2)?.reshape((n, rd))?;
    Tensor::cat(&[&head, &rot], 1)
}

/// The weights a card keeps, by where their bytes live on the host: a grouped projection hands
/// out a fresh view of the same weight at every call, so a key made of the view's identity is
/// new every time and the weight crosses the bus again. The address and length of the bytes are
/// the same view after view.
type Resident = HashMap<(usize, usize), Arc<QMatMul>>;

/// One routed expert's three weights on the card, and when it was last asked for.
struct ExpertOnCard {
    // Behind an Arc so an expert about to run is held by a cheap handle clone, not a
    // device-to-device copy of its ten megabytes - `CudaSlice`'s own `Clone` copies the whole
    // buffer, which turned every resident expert of every layer into a fresh allocation and a
    // memcpy, the cost that made a card full of the hot set slower than the host.
    gate: Arc<CudaSlice<u8>>,
    up: Arc<CudaSlice<u8>>,
    down: Arc<CudaSlice<u8>>,
    used: u64,
    /// How many times a token has routed to this expert while it was kept. A card sized below the
    /// working set keeps the most FREQUENTLY routed, not the most recent: a traced generation
    /// missed twice as often under recency, because the hot experts recur across the whole answer
    /// while recency lets a burst of one-off experts push them out.
    freq: u64,
    bytes: usize,
}

/// How many times an expert must be seen before a slot on a full card is spent on it: a card the
/// working set overflows keeps only the experts a conversation keeps returning to, and a third
/// sighting is the evidence it will be asked again rather than a one-off tail. The one-off experts
/// stay on the host, so the slots settle on the hottest of the set.
const ADMIT_AFTER: u32 = 3;

type ExpertKeys = (usize, usize);

#[derive(Default)]
struct ExpertTier {
    kept: HashMap<ExpertKeys, ExpertOnCard>,
    /// How many times each expert has been asked for. Sending one over costs more than the host's
    /// own product, so a card sized below the working set waits for a few requests before it
    /// spends a slot: it fills with the experts a conversation keeps returning to and leaves the
    /// long tail a decode routes to once or twice on the host.
    seen: HashMap<ExpertKeys, u32>,
    clock: u64,
}

/// A projection's blocks, when it is quantised in the format asked for.
fn quant_blocks(p: &Projection, want: GgmlDType) -> Option<std::borrow::Cow<'_, [u8]>> {
    match p {
        Projection::Quant(q) if q.dtype() == want => q.data().ok(),
        _ => None,
    }
}

/// Whether an error is the card saying it has no room, in any of the spellings the driver and
/// the libraries it calls use for it.
fn is_out_of_memory(e: &Error) -> bool {
    let s = format!("{e}");
    s.contains("[oom]") || s.contains("OUT_OF_MEMORY") || s.contains("ALLOC_FAILED")
}

/// One card stream and what it keeps.
pub struct Card {
    dev: Arc<CudaDevice>,
    /// Which device's room this card draws on. Several cards point at the same device - the
    /// steps' and the expert lanes' - and a ceiling held by each would let them fill it
    /// together and past it.
    ordinal: usize,
    room: &'static Mutex<Room>,
    /// The quantised weights every token reads, kept once sent: on the host they are memory
    /// bandwidth, on the card a matvec over memory an order of magnitude faster.
    resident: Mutex<Resident>,
    /// The dense always-read weights - a router's gate - by their storage.
    dense: Mutex<HashMap<crate::tensor::ops::traits::TensorId, Arc<QMatMul>>>,
    experts: Mutex<ExpertTier>,
    /// The clock ticks one expert block spans a token - the model's layer count. An expert used
    /// within this many ticks is this token's working set and is never evicted, so a decode
    /// adapts the card toward its own hot experts across tokens without trading out, at every
    /// layer, the ones the same token still needs.
    protect: u64,
    /// The IQ2_XXS decode tables, built on first use and kept beside the weights.
    tables: OnceLock<(CudaSlice<u8>, CudaSlice<u8>)>,
    /// Set when a step failed for want of memory. From then on a step is tried only when the
    /// card has the room for it, read from the driver, until one succeeds: a transient of a
    /// prefill can fill the card for a moment, and a card written off at that moment would
    /// run nothing for the rest of the session, while one retried blindly would pay the
    /// failure at every projection of every token.
    declined: AtomicBool,
    /// Whether a refusal on the always-read path has been said.
    said: AtomicBool,
}

impl Card {
    pub fn new(dev: Arc<CudaDevice>, room: &'static Mutex<Room>, protect: u64) -> Self {
        Self {
            ordinal: dev.ordinal(),
            dev,
            room,
            resident: Mutex::new(HashMap::new()),
            dense: Mutex::new(HashMap::new()),
            experts: Mutex::new(ExpertTier::default()),
            protect: protect.max(1),
            tables: OnceLock::new(),
            declined: AtomicBool::new(false),
            said: AtomicBool::new(false),
        }
    }

    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// The expert tier, whole even after a lane panicked while holding it.
    fn tier(&self) -> std::sync::MutexGuard<'_, ExpertTier> {
        self.experts.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn free_now(&self) -> Option<usize> {
        crate::tensor::cuda_ext::mem_get_info(&Device::Cuda(self.dev.clone()))
            .ok()
            .map(|(free, _)| free)
    }

    /// Whether the device has `bytes` free right now: what a step in flight may take, the
    /// working room being what the ceiling left free for exactly these steps.
    fn has_free(&self, bytes: usize) -> bool {
        self.free_now().is_some_and(|free| free > bytes)
    }

    /// Whether the device has `bytes` free right now beyond the working room and the reserve:
    /// what a weight to be kept may take.
    fn fits_now(&self, bytes: usize) -> bool {
        self.free_now().is_some_and(|free| {
            self.room
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .leaves(free, bytes)
        })
    }

    fn take_room(&self, bytes: usize, always_read: bool) -> bool {
        self.room
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take(bytes, always_read)
    }

    fn give_room(&self, bytes: usize) {
        self.room
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .give(bytes);
    }

    /// `w` on this card as a plain device tensor, kept for the next call, keyed by its storage.
    /// `None` when the card has no room for it, which leaves the host to run it as before.
    fn kept_dense(&self, w: &Tensor) -> Option<Arc<QMatMul>> {
        let key = w.id();
        if let Some(m) = self
            .dense
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
        {
            return Some(m.clone());
        }
        let bytes = w.elem_count() * F32;
        if !self.take_room(bytes, true) {
            return None;
        }
        let sent = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let q = QTensor::quantize_onto(w, GgmlDType::F32, &Device::Cuda(self.dev.clone()))?;
            QMatMul::from_arc(Arc::new(q))
        }));
        let Ok(Ok(m)) = sent else {
            self.give_room(bytes);
            self.say_once("a dense always-read weight could not go over");
            return None;
        };
        let m = Arc::new(m);
        self.dense
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, m.clone());
        Some(m)
    }

    /// `q` on this card, kept for the next call. `None` when it could not go over, which leaves
    /// the caller to upload per call as before.
    /// A weight already resident in the always-read tier, or `None` - never an upload. A step that
    /// must not admit (a fused block declining unless the whole set is already here, so it never
    /// forces a routed expert into the always-read tier nor churns the working set) asks through
    /// this rather than `kept`.
    fn kept_resident(&self, q: &Arc<QTensor>) -> Option<Arc<QMatMul>> {
        let bytes = q.data().ok()?;
        let key = (bytes.as_ptr() as usize, bytes.len());
        self.resident
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
            .cloned()
    }

    fn kept(&self, q: &Arc<QTensor>) -> Option<Arc<QMatMul>> {
        let bytes = q.data().ok()?;
        let key = (bytes.as_ptr() as usize, bytes.len());
        if let Some(m) = self
            .resident
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
        {
            return Some(m.clone());
        }
        if !self.take_room(key.1, true) {
            self.say_once(&format!(
                "no room in its accounting for an always-read weight of {} MB: {}",
                key.1 / 1_000_000,
                self.room
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .summary()
            ));
            return None;
        }
        let sent = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let on_card = q.to_device(&Device::Cuda(self.dev.clone()))?;
            QMatMul::from_arc(Arc::new(on_card))
        }));
        let m = match sent {
            Ok(Ok(m)) => Arc::new(m),
            Ok(Err(e)) => {
                self.give_room(key.1);
                self.say_once(&format!("an always-read weight could not go over: {e}"));
                return None;
            }
            Err(_) => {
                self.give_room(key.1);
                self.say_once("an always-read weight could not go over: the driver panicked");
                return None;
            }
        };
        self.resident
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, m.clone());
        Some(m)
    }

    /// `run`, a step that needs about `need` bytes of the card beside what it keeps, unless the
    /// card is refusing and has not the room for it. A memory failure inside makes the card
    /// refuse from here on, said once, and the host runs the step.
    fn attempt<T>(
        &self,
        what: &str,
        need: usize,
        run: impl FnOnce() -> Result<T>,
    ) -> Option<Result<T>> {
        if self.declined.load(Ordering::Relaxed) && !self.has_free(need) {
            return None;
        }
        // The driver panics on some allocations it cannot make rather than erring; a step that
        // panics is a step the card declined, and the host runs it.
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(run));
        match ran {
            Err(_) => {
                self.decline(what, &Error::msg("the driver panicked inside this step"));
                None
            }
            Ok(Ok(v)) => {
                self.declined.store(false, Ordering::Relaxed);
                Some(Ok(v))
            }
            Ok(Err(e)) if is_out_of_memory(&e) => {
                self.decline(what, &e);
                None
            }
            Ok(Err(e)) => Some(Err(e)),
        }
    }

    fn decline(&self, what: &str, e: &Error) {
        if !self.declined.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                "card {}: no memory for {what} ({e}); the host runs what the card has no room for",
                self.ordinal
            );
        }
    }

    /// A refusal on the always-read path, said once per card: a card that silently keeps
    /// nothing looks exactly like one that is not worth using.
    fn say_once(&self, what: &str) {
        if !self.said.swap(true, Ordering::Relaxed) {
            tracing::warn!("card {}: {what}", self.ordinal);
        }
    }

    /// Whether this card already holds `e`, with no upload owed. Read only: it takes the tier
    /// lock to look, changes nothing, and is what lets the host start its own experts the moment
    /// the cards start theirs rather than after the block returns.
    pub fn holds(&self, e: &Expert) -> bool {
        let Some(gate) = quant_blocks(&e.w1, GgmlDType::Iq2Xxs) else {
            return false;
        };
        let key = (gate.as_ptr() as usize, gate.len());
        self.tier().kept.contains_key(&key)
    }

    /// The key under which this card holds `e`, taking it on when the evidence says it will be
    /// asked again: a second request, or `proven` by the caller - a prompt's batch routing it
    /// by several tokens at once, which is what a decode learns over several tokens. `None`
    /// when the expert is not in the formats the kernels read, or the card is not taking it.
    pub fn admit(&self, e: &Expert, proven: bool) -> Option<ExpertKeys> {
        let gate = quant_blocks(&e.w1, GgmlDType::Iq2Xxs)?;
        let up = quant_blocks(&e.w3, GgmlDType::Iq2Xxs)?;
        let down = quant_blocks(&e.w2, GgmlDType::Q2K)?;
        let key = (gate.as_ptr() as usize, gate.len());
        let total = gate.len() + up.len() + down.len();
        let n = {
            let mut tier = self.tier();
            if tier.kept.contains_key(&key) {
                return Some(key);
            }
            let c = tier.seen.entry(key).or_insert(0);
            *c += 1;
            *c
        };
        // A slot goes to an expert the conversation keeps returning to, not to whoever reached the
        // card first: the one-off tail stays on the host rather than churning the hot set out.
        if n < ADMIT_AFTER && !proven {
            return None;
        }
        // How hot the one asking in is: it evicts only experts colder than itself, so a resident
        // hot expert is never traded for a colder newcomer and the working set settles.
        let incoming = if proven { u64::MAX } else { n as u64 };
        // Room for one more, or a colder resident traded for it: the card holds far fewer than the
        // working set, so the slots must move toward the hottest, but only ever downhill in heat,
        // or the same experts would upload and evict each other every token.
        // Room for one more, or a colder resident the last token did not touch traded for it: the
        // card holds far fewer experts than the working set, so the slots drift toward the hottest,
        // but the protected recent set (see `evict_to_fit`) keeps a token from trading out what it
        // still needs, so the drift is one upload per genuinely new hot expert, not a churn.
        let mut evicted = false;
        if !self.take_room(total, false) {
            if !self.evict_to_fit(total, incoming) || !self.take_room(total, false) {
                return None;
            }
            evicted = true;
        }
        // A full card fails `fits_now` even when the budget had room - the accounting ceiling is
        // above what the device physically holds once the always-read weights and the prior are
        // on it. Without evicting here the eviction never fires (the budget rarely fills), so a
        // card that has taken budget but has no physical room trades a colder expert for this one;
        // the eviction frees a real block the upload's pool then serves.
        if !evicted && !self.fits_now(total) && self.evict_to_fit(total, incoming) {
            evicted = true;
        }
        // Into memory the card has free right now, beyond what its own transient work needs:
        // the accounting knows what is kept, not what a step in flight is holding. After an
        // eviction the freed blocks sit in the device's memory pool, which the driver's free
        // count does not report until it is trimmed, so this check would refuse an upload the
        // pool can serve; trust the pool there and let the upload's own OOM guard decline.
        if !evicted && !self.fits_now(total) {
            self.give_room(total);
            return None;
        }
        // Sent outside the tier's lock, and a driver that panics on the allocation rather than
        // erring is a card that declined: the room goes back and the host runs the expert.
        let stream = self.dev.stream();
        let sent = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Some((
                stream.clone_htod(gate.as_ref()).ok()?,
                stream.clone_htod(up.as_ref()).ok()?,
                stream.clone_htod(down.as_ref()).ok()?,
            ))
        }));
        let Ok(Some((gate_d, up_d, down_d))) = sent else {
            self.give_room(total);
            self.decline(
                "an expert upload",
                &Error::msg("the card could not take it"),
            );
            return None;
        };
        let mut tier = self.tier();
        if tier.kept.contains_key(&key) {
            // Another lane sent it meanwhile: one copy is kept, this one's room goes back.
            self.give_room(total);
            return Some(key);
        }
        let used = tier.clock;
        tier.kept.insert(
            key,
            ExpertOnCard {
                gate: Arc::new(gate_d),
                up: Arc::new(up_d),
                down: Arc::new(down_d),
                used,
                // The heat that earned the slot, not one: inserted at one, a just-admitted expert
                // is the coldest resident and the next miss evicts it, so the last slots churn -
                // uploading and evicting the same borderline experts every token. Carrying the
                // count it was seen means it is only traded for one demonstrably hotter, and the
                // set converges instead of thrashing.
                freq: n as u64,
                bytes: total,
            },
        );
        Some(key)
    }

    /// Free at least `need` bytes by dropping the LEAST-FREQUENTLY-routed kept experts (never one
    /// routed to at the current clock), so an adaptive card converges to the hottest experts the
    /// conversation keeps returning to and lets the one-off ones give way. Evicted slices live
    /// until any in-flight compute holding a clone drops them; the room goes back at once and the
    /// caller's `fits_now` check still guards the allocation.
    fn evict_to_fit(&self, need: usize, incoming: u64) -> bool {
        let mut freed = Vec::new();
        {
            let mut tier = self.tier();
            let now = tier.clock;
            // Drop the coldest kept expert (lowest count, ties to least-recently-used) that is
            // colder than the one asking in, one at a time until `need` is free. Found by a scan,
            // not a full sort, since a decode frees one expert's worth at a time and the tier
            // holds thousands. The colder-than-incoming rule is what stops the churn: a resident
            // hot expert is never traded for a newcomer routed to fewer times, so once the working
            // set is on the card the one-off tail runs on the host and the uploads stop.
            while freed.iter().sum::<usize>() < need {
                // Never the current token's own experts: one used within a layer-count of ticks is
                // still needed this token, and trading it out would upload it again at the next
                // layer. Only an expert the last token did not touch, and colder than the one
                // asking in, is traded - so the card drifts toward the decode's hot set across
                // tokens without churning within one.
                let victim = tier
                    .kept
                    .iter()
                    .filter(|(_, v)| v.used + self.protect < now && v.freq < incoming)
                    .min_by_key(|(_, v)| (v.freq, v.used))
                    .map(|(k, v)| (*k, v.bytes));
                match victim {
                    Some((k, bytes)) => {
                        // Its heat outlives the slot. A resident's count grows as it runs but its
                        // `seen` does not (an admitted expert returns before the tally), so without
                        // this an evicted expert the conversation still returns to comes back at a
                        // stale low count, the coldest slot, and is traded straight out again. Kept
                        // in `seen`, it re-admits at the heat it earned and the last slots settle.
                        if let Some(v) = tier.kept.remove(&k) {
                            let s = tier.seen.entry(k).or_insert(0);
                            *s = (*s).max(v.freq.min(u32::MAX as u64) as u32);
                        }
                        freed.push(bytes);
                    }
                    None => break,
                }
            }
        }
        let total: usize = freed.iter().sum();
        for bytes in &freed {
            self.give_room(*bytes);
        }
        total >= need
    }

    /// One expert's row with its weights kept on this card: two launches and one crossing each
    /// way. `None` when the expert is not kept here.
    fn expert_row(
        &self,
        e: &Expert,
        xs: &[f32],
        limit: f32,
        timings: &[AtomicU64; 3],
    ) -> Option<Result<(Vec<f32>, Vec<f32>)>> {
        let hold_started = std::time::Instant::now();
        let dims = (e.w1.dims()[1], e.w1.dims()[0]);
        let key = self.admit(e, false)?;
        let mut tier = self.tier();
        tier.clock += 1;
        let now = tier.clock;
        let on = tier.kept.get_mut(&key)?;
        on.used = now;
        on.freq += 1;
        let (gate_d, up_d, down_d) = (on.gate.clone(), on.up.clone(), on.down.clone());
        drop(tier);
        timings[0].fetch_add(hold_started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let kernels_started = std::time::Instant::now();
        if self.tables.get().is_none() {
            let _ = self.tables.set(iq2_xxs_tables(&self.dev).ok()?);
        }
        let tables = self.tables.get()?;
        let need = (xs.len() + 2 * dims.1 + dims.0) * F32;
        let r = self.attempt("an expert row", need, || {
            gpu_expert_row(
                &self.dev,
                gate_d.as_ref(),
                up_d.as_ref(),
                down_d.as_ref(),
                (&tables.0, &tables.1),
                dims,
                xs,
                limit,
            )
        });
        timings[1].fetch_add(
            kernels_started.elapsed().as_nanos() as u64,
            Ordering::Relaxed,
        );
        if r.is_some() {
            timings[2].fetch_add(1, Ordering::Relaxed);
        }
        r
    }

    /// Several experts that share one input row - a decode's active experts of a layer - run on
    /// this card in one block: each resident expert's kernels are queued, then their outputs come
    /// back together. `es[i]` answers `Some` when it was resident and ran here, `None` when it was
    /// not and the host must run it. Bit-exact with `expert_row` called on each.
    fn expert_rows(
        &self,
        es: &[&Expert],
        xs: &[f32],
        limit: f32,
        timings: &[AtomicU64; 3],
    ) -> Vec<Option<Result<(Vec<f32>, Vec<f32>)>>> {
        let mut out: Vec<Option<Result<(Vec<f32>, Vec<f32>)>>> =
            (0..es.len()).map(|_| None).collect();
        let Some(&first) = es.first() else {
            return out;
        };
        let dims = (first.w1.dims()[1], first.w1.dims()[0]);
        if self.tables.get().is_none() {
            match iq2_xxs_tables(&self.dev) {
                Ok(t) => {
                    let _ = self.tables.set(t);
                }
                Err(_) => return out,
            }
        }
        let Some(tables) = self.tables.get() else {
            return out;
        };
        // The resident subset, admitted (second-sighting or with room) as `expert_row` does, each
        // keeping its own hold on the card's weights so an eviction cannot pull them mid-block.
        let hold_started = std::time::Instant::now();
        let mut idx = Vec::new();
        let mut held: Vec<(Arc<CudaSlice<u8>>, Arc<CudaSlice<u8>>, Arc<CudaSlice<u8>>)> =
            Vec::new();
        for (i, e) in es.iter().enumerate() {
            let Some(key) = self.admit(e, false) else {
                continue;
            };
            let mut tier = self.tier();
            tier.clock += 1;
            let now = tier.clock;
            if let Some(on) = tier.kept.get_mut(&key) {
                on.used = now;
                on.freq += 1;
                held.push((on.gate.clone(), on.up.clone(), on.down.clone()));
                idx.push(i);
            }
        }
        timings[0].fetch_add(hold_started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if held.is_empty() {
            return out;
        }
        let batch: Vec<(&CudaSlice<u8>, &CudaSlice<u8>, &CudaSlice<u8>)> = held
            .iter()
            .map(|(g, u, d)| (g.as_ref(), u.as_ref(), d.as_ref()))
            .collect();
        let need = (xs.len() + held.len() * (2 * dims.1 + dims.0)) * F32;
        let kernels_started = std::time::Instant::now();
        let ran = self.attempt("expert rows", need, || {
            gpu_expert_rows_grouped(&self.dev, &batch, (&tables.0, &tables.1), dims, xs, limit)
        });
        timings[1].fetch_add(
            kernels_started.elapsed().as_nanos() as u64,
            Ordering::Relaxed,
        );
        // A whole-block failure leaves every one of these to the host: `None` stands.
        if let Some(Ok(rows)) = ran {
            timings[2].fetch_add(rows.len() as u64, Ordering::Relaxed);
            for (&i, row) in idx.iter().zip(rows) {
                out[i] = Some(Ok(row));
            }
        }
        out
    }

    /// One activation row per instance through the expert it names, for the instances whose expert
    /// this card already holds. `insts[i]` is run over `xrows[i]`; an instance whose expert is not
    /// resident here gets `None` and the caller runs it on the host. Nothing is uploaded: a verify
    /// uses only what a decode already warmed, so it never churns the set.
    fn expert_rows_multi(
        &self,
        insts: &[&Expert],
        xrows: &[&[f32]],
        limit: f32,
        timings: &[AtomicU64; 3],
    ) -> Vec<Option<Result<Vec<f32>>>> {
        let mut out: Vec<Option<Result<Vec<f32>>>> = (0..insts.len()).map(|_| None).collect();
        let Some(&first) = insts.first() else {
            return out;
        };
        let dims = (first.w1.dims()[1], first.w1.dims()[0]);
        if self.tables.get().is_none() {
            match iq2_xxs_tables(&self.dev) {
                Ok(t) => {
                    let _ = self.tables.set(t);
                }
                Err(_) => return out,
            }
        }
        let Some(tables) = self.tables.get() else {
            return out;
        };
        let hold_started = std::time::Instant::now();
        let mut idx = Vec::new();
        let mut held: Vec<(Arc<CudaSlice<u8>>, Arc<CudaSlice<u8>>, Arc<CudaSlice<u8>>)> =
            Vec::new();
        let mut x_multi: Vec<f32> = Vec::new();
        for (i, e) in insts.iter().enumerate() {
            let Some(gate) = quant_blocks(&e.w1, GgmlDType::Iq2Xxs) else {
                continue;
            };
            let key = (gate.as_ptr() as usize, gate.len());
            let mut tier = self.tier();
            tier.clock += 1;
            let now = tier.clock;
            if let Some(on) = tier.kept.get_mut(&key) {
                on.used = now;
                on.freq += 1;
                held.push((on.gate.clone(), on.up.clone(), on.down.clone()));
                idx.push(i);
                x_multi.extend_from_slice(xrows[i]);
            }
        }
        timings[0].fetch_add(hold_started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if held.is_empty() {
            return out;
        }
        let batch: Vec<(&CudaSlice<u8>, &CudaSlice<u8>, &CudaSlice<u8>)> = held
            .iter()
            .map(|(g, u, d)| (g.as_ref(), u.as_ref(), d.as_ref()))
            .collect();
        let need = (x_multi.len() + held.len() * (2 * dims.1 + dims.0)) * F32;
        let kernels_started = std::time::Instant::now();
        let ran = self.attempt("expert rows multi", need, || {
            gpu_expert_rows_grouped_multi(
                &self.dev,
                &batch,
                (&tables.0, &tables.1),
                dims,
                &x_multi,
                limit,
            )
        });
        timings[1].fetch_add(
            kernels_started.elapsed().as_nanos() as u64,
            Ordering::Relaxed,
        );
        if let Some(Ok(rows)) = ran {
            timings[2].fetch_add(rows.len() as u64, Ordering::Relaxed);
            for (&i, row) in idx.iter().zip(rows) {
                out[i] = Some(Ok(row));
            }
        }
        out
    }

    /// `xs` rows through `m`, a weight kept on this card.
    fn through_kept(
        &self,
        what: &str,
        m: &QMatMul,
        xs: &[f32],
        inp: usize,
        out: usize,
    ) -> Option<Result<Vec<f32>>> {
        let rows = xs.len() / inp;
        self.attempt(what, (xs.len() + rows * out) * F32, || {
            let x = Tensor::from_vec(xs.to_vec(), (rows, inp), &Device::Cuda(self.dev.clone()))?;
            m.forward(&x)?.flatten_all()?.to_vec1::<f32>()
        })
    }
}

impl Offload for Card {
    fn projection(&self, p: &Projection, xs: &[f32]) -> Option<Result<Vec<f32>>> {
        let dev = &self.dev;
        let dims = p.dims();
        let (out, inp) = (dims.first().copied()?, dims.get(1).copied()?);
        if inp == 0 || xs.len() % inp != 0 {
            return None;
        }
        let rows = xs.len() / inp;
        let one_row = rows == 1;
        // A weight is kept for the calls a decode makes: one row, or the few rows a compressor
        // pools at a group boundary. A prompt's batch of hundreds reads each weight once and
        // is not kept for; the bound is the width one launch of the resident matvec takes.
        let always_read = matches!(p, Projection::Quant(q) if matches!(q.dtype(), GgmlDType::Q6K | GgmlDType::Q8_0));
        if let (true, Projection::Quant(q)) = (always_read, p) {
            // Kept for the decode's rows of an always-read weight, and nothing else. The
            // routed experts are read once each and there are thousands, so keeping one this
            // way buys nothing and spends the room the next always-read weight needs. Kept for
            // a batch, too, they cost: a prefill reads each weight once, and holding them made
            // it a third slower by taking the room its own products need.
            if rows <= matvec_rows(q.dtype(), inp) {
                if let Some(m) = self.kept(q) {
                    return self.through_kept("a resident weight", &m, xs, inp, out);
                }
            }
        }
        // A router's gate is the one dense weight every token reads: kept as the quantised
        // always-read weights are, keyed by its storage, answered the same way.
        if let (true, Projection::Dense(w)) = (rows <= matvec_rows(GgmlDType::F32, inp), p) {
            if let Some(m) = self.kept_dense(w) {
                return self.through_kept("a resident dense weight", &m, xs, inp, out);
            }
        }
        // A weight already on the card answers a single row ten to twenty times faster than
        // the host does, activation sent and result read back included. A weight that would
        // have to cross for this one row does not: there the host wins, so it declines.
        if one_row {
            return None;
        }
        let need = p.bytes() + (xs.len() + rows * out) * F32;
        match p {
            Projection::Fp4(w) => self.attempt("an fp4 weight", need, || {
                gpu_fp4_linear(
                    dev,
                    w.nibbles.as_slice(),
                    w.scales.as_slice(),
                    (w.out, w.inp),
                    xs,
                )
            }),
            Projection::Fp8(w) => self.attempt("an fp8 weight", need, || {
                gpu_fp8_linear(
                    dev,
                    w.view_bytes(),
                    w.scales.as_slice(),
                    (w.out, w.inp, w.block, w.row0()),
                    xs,
                )
            }),
            Projection::Quant(q) => {
                let dtype = q.dtype();
                if !matches!(dtype, GgmlDType::Iq2Xxs | GgmlDType::Q2K | GgmlDType::Q8_0) {
                    return None;
                }
                self.attempt("a quantised weight", need, || {
                    let bytes = q.data()?;
                    gpu_quant_linear(dev, dtype, &bytes, (out, inp), xs)
                })
            }
            Projection::Dense(w) => {
                if dims.len() != 2 {
                    return None;
                }
                self.attempt("a dense weight", need * 2, || {
                    let wv = w.flatten_all()?.to_vec1::<f32>()?;
                    let mut y = vec![0f32; rows * out];
                    let wt: Vec<f32> = (0..inp * out)
                        .map(|i| wv[(i % out) * inp + i / out])
                        .collect();
                    crate::tensor::cuda::gpu_matmul_host(dev, xs, &wt, &mut y, (rows, inp, out))?;
                    Ok(y)
                })
            }
            Projection::Bf16(_) => None,
        }
    }

    fn sparse_attention(
        &self,
        q: &[f32],
        kv: &[f32],
        sink: &[f32],
        idxs: &[i32],
        dims: (usize, usize, usize, usize),
        scale: f32,
    ) -> Option<Result<Vec<f32>>> {
        // The launcher takes the queries in chunks sized by what the card has free, so what a
        // call needs beside the keys and its answer is its own to size.
        let need = (q.len() + kv.len() + idxs.len()) * F32 + q.len() * F32;
        self.attempt("sparse attention", need, || {
            gpu_sparse_attn(&self.dev, q, kv, sink, idxs, dims, scale)
        })
    }

    fn index_scores(
        &self,
        q: &[f32],
        k: &[f32],
        weights: &[f32],
        dims: (usize, usize, usize, usize),
        ratio: usize,
        scale: f32,
    ) -> Option<Result<Vec<f32>>> {
        let need = (q.len() + k.len() + weights.len() + dims.0 * dims.3) * F32;
        self.attempt("the indexer scores", need, || {
            gpu_index_scores(&self.dev, q, k, weights, dims, ratio, scale)
        })
    }

    fn attn_out(
        &self,
        wo_a: &Projection,
        wo_b: &Projection,
        o: &[f32],
        o_groups: usize,
        p: usize,
        o_lora: usize,
        dim: usize,
    ) -> Option<Result<Vec<f32>>> {
        // Only the always-read k-quant path is kept on a card; anything else runs on the host.
        let wob_q = match wo_b {
            Projection::Quant(q) => q,
            _ => return None,
        };
        if o.len() != o_groups * p {
            return None;
        }
        // Each group's row-view of wo_a and wo_b, all kept on THIS card or the chain is declined so
        // the co-located weights answer from one place; the views are what a group projection keeps
        // on the host path too, so nothing new crosses.
        let mut groups = Vec::with_capacity(o_groups);
        for g in 0..o_groups {
            let rv = match wo_a.rows(g * o_lora, o_lora) {
                Ok(Projection::Quant(q)) => q,
                _ => return None,
            };
            groups.push(self.kept(&rv)?);
        }
        let wob = self.kept(wob_q)?;
        let need = (o.len() + o_groups * o_lora + dim) * F32;
        self.attempt("attn grouped out", need, || {
            let o_dev =
                Tensor::from_vec(o.to_vec(), (o_groups, p), &Device::Cuda(self.dev.clone()))?;
            let mut parts = Vec::with_capacity(o_groups);
            for (g, m) in groups.iter().enumerate() {
                let xg = o_dev.narrow(0, g, 1)?;
                parts.push(m.forward(&xg)?);
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            let cat = Tensor::cat(&refs, 1)?;
            wob.forward(&cat)?.flatten_all()?.to_vec1::<f32>()
        })
    }

    fn attn_block(&self, p: &AttnBlock) -> Option<Result<(Vec<f32>, Vec<f32>)>> {
        // Every projection of the block must be kept on THIS card or it declines to the host chain,
        // so the activation stays on one device from the layer input to the block output.
        let keep = |proj: &Projection| -> Option<Arc<QMatMul>> {
            match proj {
                Projection::Quant(q) => self.kept(q),
                _ => None,
            }
        };
        let m_qa = keep(p.wq_a)?;
        let m_qb = keep(p.wq_b)?;
        let m_kv = keep(p.wkv)?;
        let mut wo_groups = Vec::with_capacity(p.o_groups);
        for g in 0..p.o_groups {
            let rv = p.wo_a.rows(g * p.o_lora, p.o_lora).ok()?;
            wo_groups.push(keep(&rv)?);
        }
        let wob = keep(p.wo_b)?;
        let (h, hd, rd) = (p.n_heads, p.head_dim, p.rope_head_dim);
        let dev = Device::Cuda(self.dev.clone());
        let w = if hd > 0 { p.window.len() / hd } else { 0 };
        let need = (p.x.len() + (w + 1) * hd + h * hd * 4 + p.dim) * F32;
        self.attempt("attn block", need, || {
            let x = Tensor::from_vec(p.x.to_vec(), (1, p.dim), &dev)?;
            let cos = Tensor::from_vec(p.cos.to_vec(), (1, rd / 2), &dev)?;
            let sin = Tensor::from_vec(p.sin.to_vec(), (1, rd / 2), &dev)?;
            let q_norm = p.q_norm.to_device(&dev)?;
            let kv_norm = p.kv_norm.to_device(&dev)?;
            let sink = p.sink.to_device(&dev)?.reshape((h, 1))?;
            // Query: wq_a -> q_norm -> wq_b -> partial rope, all on the card.
            let qr = rms_norm(&m_qa.forward(&x)?, &q_norm, p.eps)?;
            let qf = m_qb.forward(&qr)?.reshape((h, hd))?;
            let qrot = rope_partial_dev(&qf, &cos, &sin, rd)?;
            // The token's compressed key latent: wkv -> kv_norm -> partial rope.
            let kvn = rms_norm(&m_kv.forward(&x)?, &kv_norm, p.eps)?.reshape((1, hd))?;
            let kvrot = rope_partial_dev(&kvn, &cos, &sin, rd)?;
            let kv_row: Vec<f32> = kvrot.flatten_all()?.to_vec1::<f32>()?;
            // Windowed attention over the past keys plus this token's, with a per-head sink that
            // competes in the softmax but attends to nothing.
            let winkv = if w == 0 {
                kvrot.clone()
            } else {
                let win = Tensor::from_vec(p.window.to_vec(), (w, hd), &dev)?;
                Tensor::cat(&[&win, &kvrot], 0)?
            };
            let keys = winkv.dim(0)?;
            let scores = qrot.matmul(&winkv.t()?)?.affine(p.scale, 0.0)?;
            let logits = Tensor::cat(&[&scores, &sink], 1)?;
            let probs = softmax_last_dim(&logits)?.narrow(1, 0, keys)?;
            let o = probs.matmul(&winkv)?;
            // Output rope (negated sin), then the grouped low-rank output projection.
            let sin_neg = sin.affine(-1.0, 0.0)?;
            let orot = rope_partial_dev(&o, &cos, &sin_neg, rd)?;
            let p2 = h * hd / p.o_groups;
            let og = orot.reshape((p.o_groups, p2))?;
            let mut parts = Vec::with_capacity(p.o_groups);
            for (g, m) in wo_groups.iter().enumerate() {
                parts.push(m.forward(&og.narrow(0, g, 1)?)?);
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            let cat = Tensor::cat(&refs, 1)?;
            let out = wob.forward(&cat)?.flatten_all()?.to_vec1::<f32>()?;
            Ok((out, kv_row))
        })
    }

    fn expert_dev(
        &self,
        w1: &Projection,
        w3: &Projection,
        w2: &Projection,
        x: &[f32],
        limit: f32,
    ) -> Option<Result<Vec<f32>>> {
        let quant = |p: &Projection| match p {
            Projection::Quant(q) => Some(q.clone()),
            _ => None,
        };
        let m1 = self.kept(&quant(w1)?)?;
        let m3 = self.kept(&quant(w3)?)?;
        let m2 = self.kept(&quant(w2)?)?;
        let d1 = w1.dims();
        let (inter, dim) = (d1[0], d1[1]);
        if dim == 0 || x.len() % dim != 0 {
            return None;
        }
        let rows = x.len() / dim;
        let dev = Device::Cuda(self.dev.clone());
        let need = (x.len() + rows * (2 * inter + dim)) * F32;
        self.attempt("shared expert", need, || {
            let xd = Tensor::from_vec(x.to_vec(), (rows, dim), &dev)?;
            let gate = m1.forward(&xd)?;
            let up = m3.forward(&xd)?;
            // SwiGLU with the reference clamps: the gate branch from above, the up branch both sides.
            let h = if limit > 0.0 {
                let hi = Tensor::full(limit, 1, &dev)?;
                let lo = Tensor::full(-limit, 1, &dev)?;
                let g = gate.broadcast_minimum(&hi)?;
                let u = up.broadcast_maximum(&lo)?.broadcast_minimum(&hi)?;
                g.silu()?.mul(&u)?
            } else {
                gate.silu()?.mul(&up)?
            };
            m2.forward(&h)?.flatten_all()?.to_vec1::<f32>()
        })
    }

    fn moe_decode(
        &self,
        active: &[(&Projection, &Projection, &Projection)],
        weights: &[f32],
        shared: Option<(&Projection, &Projection, &Projection)>,
        x: &[f32],
        limit: f32,
        dim: usize,
    ) -> Option<Result<Vec<f32>>> {
        let quant = |p: &Projection| match p {
            Projection::Quant(q) => Some(q.clone()),
            _ => None,
        };
        // Every active expert (and the shared) must ALREADY be resident on THIS card, or the block
        // declines so the streamed path runs it. Resident-only, never an upload: a routed expert
        // lives in the evictable expert tier with its CPU overlap, so admitting it into the
        // always-read tier here would force a per-token crossing and churn the working set - the
        // regression this fused block existed to avoid.
        let mut ms: Vec<(Arc<QMatMul>, Arc<QMatMul>, Arc<QMatMul>)> =
            Vec::with_capacity(active.len());
        for (w1, w3, w2) in active {
            ms.push((
                self.kept_resident(&quant(w1)?)?,
                self.kept_resident(&quant(w3)?)?,
                self.kept_resident(&quant(w2)?)?,
            ));
        }
        let sh = match shared {
            Some((w1, w3, w2)) => Some((
                self.kept_resident(&quant(w1)?)?,
                self.kept_resident(&quant(w3)?)?,
                self.kept_resident(&quant(w2)?)?,
            )),
            None => None,
        };
        let inter = active.first().map(|(w1, _, _)| w1.dims()[0]).unwrap_or(0);
        let dev = Device::Cuda(self.dev.clone());
        let need = (x.len() + (active.len() + 1) * (2 * inter + dim)) * F32;
        self.attempt("moe decode", need, || {
            let xd = Tensor::from_vec(x.to_vec(), (1, dim), &dev)?;
            let hi = Tensor::full(limit, 1, &dev)?;
            let lo = Tensor::full(-limit, 1, &dev)?;
            let run = |m1: &QMatMul, m3: &QMatMul, m2: &QMatMul| -> Result<Tensor> {
                let gate = m1.forward(&xd)?;
                let up = m3.forward(&xd)?;
                let h = if limit > 0.0 {
                    let g = gate.broadcast_minimum(&hi)?;
                    let u = up.broadcast_maximum(&lo)?.broadcast_minimum(&hi)?;
                    g.silu()?.mul(&u)?
                } else {
                    gate.silu()?.mul(&up)?
                };
                m2.forward(&h)
            };
            // The shared expert unweighted, then each routed expert scaled by its routing weight.
            let mut out = match &sh {
                Some((m1, m3, m2)) => run(m1, m3, m2)?,
                None => Tensor::zeros_on((1, dim), crate::tensor::DType::F32, &dev)?,
            };
            for (i, (m1, m3, m2)) in ms.iter().enumerate() {
                out = out.add(&run(m1, m3, m2)?.affine(weights[i], 0.0)?)?;
            }
            out.flatten_all()?.to_vec1::<f32>()
        })
    }
}

/// The cards of one placement behind the trait: a weight kept on any of them answers from
/// there, one not yet kept goes to the first card with room for it, and the transient steps
/// run on the first card, the fastest.
pub struct Cards(pub Vec<Arc<Card>>);

impl Offload for Cards {
    fn projection(&self, p: &Projection, xs: &[f32]) -> Option<Result<Vec<f32>>> {
        self.0.iter().find_map(|c| c.projection(p, xs))
    }

    fn sparse_attention(
        &self,
        q: &[f32],
        kv: &[f32],
        sink: &[f32],
        idxs: &[i32],
        dims: (usize, usize, usize, usize),
        scale: f32,
    ) -> Option<Result<Vec<f32>>> {
        self.0
            .iter()
            .find_map(|c| c.sparse_attention(q, kv, sink, idxs, dims, scale))
    }

    fn index_scores(
        &self,
        q: &[f32],
        k: &[f32],
        weights: &[f32],
        dims: (usize, usize, usize, usize),
        ratio: usize,
        scale: f32,
    ) -> Option<Result<Vec<f32>>> {
        self.0
            .iter()
            .find_map(|c| c.index_scores(q, k, weights, dims, ratio, scale))
    }

    fn attn_out(
        &self,
        wo_a: &Projection,
        wo_b: &Projection,
        o: &[f32],
        o_groups: usize,
        p: usize,
        o_lora: usize,
        dim: usize,
    ) -> Option<Result<Vec<f32>>> {
        self.0
            .iter()
            .find_map(|c| c.attn_out(wo_a, wo_b, o, o_groups, p, o_lora, dim))
    }

    fn attn_block(&self, p: &AttnBlock) -> Option<Result<(Vec<f32>, Vec<f32>)>> {
        self.0.iter().find_map(|c| c.attn_block(p))
    }

    fn expert_dev(
        &self,
        w1: &Projection,
        w3: &Projection,
        w2: &Projection,
        x: &[f32],
        limit: f32,
    ) -> Option<Result<Vec<f32>>> {
        self.0
            .iter()
            .find_map(|c| c.expert_dev(w1, w3, w2, x, limit))
    }

    fn moe_decode(
        &self,
        active: &[(&Projection, &Projection, &Projection)],
        weights: &[f32],
        shared: Option<(&Projection, &Projection, &Projection)>,
        x: &[f32],
        limit: f32,
        dim: usize,
    ) -> Option<Result<Vec<f32>>> {
        self.0
            .iter()
            .find_map(|c| c.moe_decode(active, weights, shared, x, limit, dim))
    }
}

/// Routed experts run on `cards`, a lane per card stream: the gate and up products there, the
/// activation between them here, the down product there.
pub fn lanes(cards: Vec<Arc<Card>>) -> ExpertOffload {
    let cards = Arc::new(cards);
    let timings: Arc<[AtomicU64; 3]> = Arc::new(Default::default());
    let shared = timings.clone();
    let for_warm = cards.clone();
    let for_oncard = cards.clone();
    let batch_cards = cards.clone();
    let batch_timings = timings.clone();
    let multi_cards = cards.clone();
    let multi_timings = timings.clone();
    ExpertOffload {
        lanes: cards.len(),
        timings,
        warm: Some(Box::new(move |id, expert| {
            let on = &for_warm[id % for_warm.len()];
            on.admit(expert, true).is_some()
        })),
        run_batch: Some(Box::new(move |active, fetch, x, limit| {
            let ncards = batch_cards.len().max(1);
            let mut out: Vec<Option<Result<(Vec<f32>, Vec<f32>)>>> =
                (0..active.len()).map(|_| None).collect();
            // Each expert runs on the card its number names, the same home as the per-expert
            // path, so a card keeps the ones that keep coming back to it. The cards' blocks run
            // at once, on a thread each: a block is short and one card's kernels and its host
            // work must not wait on the other's, or the second card's answer arrives a whole
            // block late every layer.
            let mut by_card: Vec<Vec<usize>> = (0..ncards).map(|_| Vec::new()).collect();
            for (i, &e) in active.iter().enumerate() {
                by_card[e % ncards].push(i);
            }
            let slots = std::sync::Mutex::new(&mut out);
            std::thread::scope(|scope| {
                for (c, positions) in by_card.iter().enumerate() {
                    if positions.is_empty() {
                        continue;
                    }
                    let (cards, timings, slots) = (&batch_cards, &batch_timings, &slots);
                    scope.spawn(move || {
                        let mut refs_owned: Vec<Arc<Expert>> = Vec::with_capacity(positions.len());
                        let mut kept: Vec<usize> = Vec::with_capacity(positions.len());
                        for &p in positions {
                            if let Ok(e) = fetch(active[p]) {
                                refs_owned.push(e);
                                kept.push(p);
                            }
                        }
                        if refs_owned.is_empty() {
                            return;
                        }
                        let refs: Vec<&Expert> = refs_owned.iter().map(|e| e.as_ref()).collect();
                        let rows = cards[c].expert_rows(&refs, x, limit, timings);
                        let mut g = slots.lock().unwrap_or_else(|e| e.into_inner());
                        for (&p, r) in kept.iter().zip(rows) {
                            g[p] = r;
                        }
                    });
                }
            });
            out
        })),
        run_multi: Some(Box::new(move |eidx, xrows, fetch, limit| {
            let ncards = multi_cards.len().max(1);
            let mut out: Vec<Option<Result<Vec<f32>>>> = (0..eidx.len()).map(|_| None).collect();
            let mut by_card: Vec<Vec<usize>> = (0..ncards).map(|_| Vec::new()).collect();
            for (i, &e) in eidx.iter().enumerate() {
                by_card[e % ncards].push(i);
            }
            let slots = std::sync::Mutex::new(&mut out);
            std::thread::scope(|scope| {
                for (c, positions) in by_card.iter().enumerate() {
                    if positions.is_empty() {
                        continue;
                    }
                    let (cards, timings, slots) = (&multi_cards, &multi_timings, &slots);
                    scope.spawn(move || {
                        let mut refs_owned: Vec<Arc<Expert>> = Vec::with_capacity(positions.len());
                        let mut kept: Vec<usize> = Vec::with_capacity(positions.len());
                        for &p in positions {
                            if let Ok(e) = fetch(eidx[p]) {
                                refs_owned.push(e);
                                kept.push(p);
                            }
                        }
                        if refs_owned.is_empty() {
                            return;
                        }
                        let refs: Vec<&Expert> = refs_owned.iter().map(|e| e.as_ref()).collect();
                        let xs: Vec<&[f32]> = kept.iter().map(|&p| xrows[p]).collect();
                        let rows = cards[c].expert_rows_multi(&refs, &xs, limit, timings);
                        let mut g = slots.lock().unwrap_or_else(|e| e.into_inner());
                        for (&p, r) in kept.iter().zip(rows) {
                            g[p] = r;
                        }
                    });
                }
            });
            out
        })),
        run: Box::new(move |lane, tokens, expert, rows, limit| {
            let on = &cards[lane % cards.len()];
            // A prompt's batch fills the cards before any token is decoded: an expert several
            // of its tokens routed to goes over now, in the prefill's own time, and the decode
            // that follows finds it there instead of paying the crossing then.
            if tokens > 1 && rows.len() >= 2 * expert.w1.dims()[1] {
                let _ = on.admit(expert, true);
            }
            // A decode asks for one row of one token: the whole expert runs on the card, its
            // weights kept there. A prompt's batch does not take this path even for the
            // experts a single one of its tokens routed to - it would fill the card with
            // weights that batch never reads again.
            if tokens == 1 && rows.len() == expert.w1.dims()[1] {
                if let Some(r) = on.expert_row(expert, rows, limit, &shared) {
                    return Some(r);
                }
            }
            let (Some(gate), Some(up)) = (
                on.projection(&expert.w1, rows),
                on.projection(&expert.w3, rows),
            ) else {
                return None;
            };
            let run = || -> Result<(Vec<f32>, Vec<f32>)> {
                let (gate, up) = (gate?, up?);
                // The SwiGLU with its training clamps, as `Expert::forward` applies them.
                let h: Vec<f32> = gate
                    .iter()
                    .zip(&up)
                    .map(|(&g, &u)| {
                        let (g, u) = if limit > 0.0 {
                            (g.min(limit), u.clamp(-limit, limit))
                        } else {
                            (g, u)
                        };
                        g / (1.0 + (-g).exp()) * u
                    })
                    .collect();
                let out = match on.projection(&expert.w2, &h) {
                    Some(r) => r?,
                    None => {
                        // The card declined the down product after taking the two before it.
                        // The rows entering it are here already, so the host finishes the
                        // expert rather than the step failing.
                        let inp = expert.w2.dims()[1];
                        let hx = Tensor::from_vec(h.clone(), (h.len() / inp, inp), &Device::Cpu)?;
                        expert.w2.apply(&hx)?.flatten_all()?.to_vec1::<f32>()?
                    }
                };
                Ok((h, out))
            };
            Some(run())
        }),
        on_card: Some(Box::new(move |e| for_oncard.iter().any(|c| c.holds(e)))),
    }
}
