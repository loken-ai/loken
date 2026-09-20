//! The heavy steps of a forward on a card: every projection product, whatever its storage, and
//! sparse attention, each dequantised or gathered on the device and computed at full precision.
//! The weights every token reads stay on the card once sent; routed experts are kept as the
//! room allows and never traded for one another.

use super::experts::{Expert, ExpertOffload};
use super::projection::Projection;
use super::room::Room;
use super::Offload;
use crate::tensor::cuda::{
    gpu_expert_row, gpu_fp4_linear, gpu_fp8_linear, gpu_index_scores, gpu_quant_linear,
    gpu_sparse_attn, iq2_xxs_tables, CudaDevice,
};
use crate::tensor::quantized::{matvec_rows, GgmlDType, QMatMul, QTensor};
use crate::tensor::{Device, Error, Result, Tensor};
use cudarc::driver::CudaSlice;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

const F32: usize = std::mem::size_of::<f32>();

/// The weights a card keeps, by where their bytes live on the host: a grouped projection hands
/// out a fresh view of the same weight at every call, so a key made of the view's identity is
/// new every time and the weight crosses the bus again. The address and length of the bytes are
/// the same view after view.
type Resident = HashMap<(usize, usize), Arc<QMatMul>>;

/// One routed expert's three weights on the card, and when it was last asked for.
struct ExpertOnCard {
    gate: CudaSlice<u8>,
    up: CudaSlice<u8>,
    down: CudaSlice<u8>,
    used: u64,
}

type ExpertKeys = (usize, usize);

#[derive(Default)]
struct ExpertTier {
    kept: HashMap<ExpertKeys, ExpertOnCard>,
    /// Experts asked for once. Sending one over costs more than the host's own product, so it
    /// goes over on its second request, not its first: a decode of a few dozen tokens reads
    /// hundreds of experts exactly once, and paying for each of those is what made a short
    /// answer slower rather than faster.
    seen: HashSet<ExpertKeys>,
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
    pub fn new(dev: Arc<CudaDevice>, room: &'static Mutex<Room>) -> Self {
        Self {
            ordinal: dev.ordinal(),
            dev,
            room,
            resident: Mutex::new(HashMap::new()),
            dense: Mutex::new(HashMap::new()),
            experts: Mutex::new(ExpertTier::default()),
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
        {
            let mut tier = self.tier();
            if tier.kept.contains_key(&key) {
                return Some(key);
            }
            if tier.seen.insert(key) && !proven {
                return None;
            }
        }
        // Room for one more, or the host runs it. Nothing is evicted to make room: a miss that
        // uploads ten megabytes costs more than the host's own product, so a card fills once
        // and keeps what it holds rather than trading one expert for another at every token.
        if !self.take_room(total, false) {
            return None;
        }
        // Into memory the card has free right now, beyond what its own transient work needs:
        // the accounting knows what is kept, not what a step in flight is holding.
        if !self.fits_now(total) {
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
                gate: gate_d,
                up: up_d,
                down: down_d,
                used,
            },
        );
        Some(key)
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
                &gate_d,
                &up_d,
                &down_d,
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
}

/// Routed experts run on `cards`, a lane per card stream: the gate and up products there, the
/// activation between them here, the down product there.
pub fn lanes(cards: Vec<Arc<Card>>) -> ExpertOffload {
    let cards = Arc::new(cards);
    let timings: Arc<[AtomicU64; 3]> = Arc::new(Default::default());
    let shared = timings.clone();
    let for_warm = cards.clone();
    ExpertOffload {
        lanes: cards.len(),
        timings,
        warm: Some(Box::new(move |id, expert| {
            let on = &for_warm[id % for_warm.len()];
            on.admit(expert, true).is_some()
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
    }
}
