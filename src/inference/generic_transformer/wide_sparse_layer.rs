//! A layer of a wide-residual mixture model whose residual mixes through Sinkhorn projections
//! around latent attention, on the host.
//!
//! The residual is `streams` copies of the hidden state.
//! A sublayer reads the copies collapsed by a pre mix, normalised.
//! Its output goes back to every copy through the post and comb mixes of its own Sinkhorn mixer.
//!
//! The pre mix a sublayer collapses with is the one the sublayer before it computed:
//! - the token mixer collapses with the pre mix of the previous layer's feed-forward,
//! - the feed-forward collapses with the pre mix its own layer's token mixer computed,
//! - the first layer of a forward collapses onto stream 0,
//! - the final norm collapses with the pre mix of the last layer's feed-forward.
//!
//! The pre mixes pass from layer to layer through a [`PreMixCarry`] the model shares.
//!
//! The rows of a verify attend 1 at a time, each as the decode step of its row.
//! Their feed-forward runs over every row at once.
//!
//! An engram layer adds hashed n-gram rows to its residual before its token mixer.
//! The hashes read the token ids of the sequence from a [`TokenCarry`] the model shares.
//! The rows of a verify take their engram 1 row at a time, a prompt's all at once.
//!
//! A layer a block draft reads adds its input, after the engram, to the draft's capture.

use super::block_draft::{BlockDraft, LayerCapture};
use super::block_scaled_experts::BlockScaledExpert;
use super::compressed_band::BandCell;
use super::hyper::sinkhorn::SinkhornMixer;
use super::moe_weights::{
    sqrt_softplus_choice, sqrt_softplus_truncated_choice, sqrt_softplus_weigh, Truncation,
};
use super::ngram::engram::{EngramBlock, EngramMeta};
use super::wide::staged;
use super::wide_expert_card::CardExperts;
use super::*;
use crate::tensor::blockscaled::projection::Projection;
use crate::tensor::ops::rms_norm;
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// The pre mix of a row that collapses onto stream 0 of `streams`.
pub(super) fn onto_first(streams: usize) -> Vec<f32> {
    let mut row = vec![0f32; streams];
    if let Some(w) = row.first_mut() {
        *w = 1.0;
    }
    row
}

/// The pre mix one layer left for the rows `first..first + rows`.
struct Carried {
    first: usize,
    rows: usize,
    pre: Vec<Vec<f32>>,
}

/// Where each layer of a forward finds the pre mix its token mixer collapses with, and leaves
/// the one its feed-forward computed.
#[derive(Default)]
pub struct PreMixCarry(std::sync::Mutex<Option<Carried>>);

impl PreMixCarry {
    fn with<R>(&self, f: impl FnOnce(&mut Option<Carried>) -> R) -> R {
        let mut held = self.0.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut held)
    }

    /// Opens the forward over the rows `first..first + rows`: every row collapses onto
    /// stream 0 of `streams`.
    pub fn open(&self, first: usize, rows: usize, streams: usize) {
        let pre = (0..rows).map(|_| onto_first(streams)).collect();
        self.with(|c| *c = Some(Carried { first, rows, pre }));
    }

    /// Leaves `pre` for the next layer of the forward over the rows `first..first + rows`.
    pub(super) fn put(&self, first: usize, rows: usize, pre: Vec<Vec<f32>>) {
        self.with(|c| *c = Some(Carried { first, rows, pre }));
    }

    /// The pre mix left for the forward over the rows `first..first + rows`.
    pub(super) fn get(&self, first: usize, rows: usize) -> Result<Vec<Vec<f32>>> {
        self.with(|c| match c {
            Some(c) if c.first == first && c.rows == rows => Ok(c.pre.clone()),
            _ => Err(crate::tensor::Error::msg(format!(
                "no pre mix for the {rows} rows from position {first}: the layer before has not run"
            ))),
        })
    }

    /// The pre mix the last layer left, for a forward over `rows` rows.
    fn last(&self, rows: usize) -> Result<Vec<Vec<f32>>> {
        self.with(|c| match c {
            Some(c) if c.rows == rows => Ok(c.pre.clone()),
            _ => Err(crate::tensor::Error::msg(format!(
                "no pre mix for the {rows} rows of the head: the last layer has not run"
            ))),
        })
    }
}

/// The token ids of a sequence by position.
///
/// A forward from position `first` keeps the ids before `first` and writes its own after them.
///
/// A log on the far side of a cut of the stack holds instead the compressed ids it received.
/// They are the ids of the positions the engram reads for the rows of 1 forward.
#[derive(Debug, Default, Clone)]
pub struct TokenLog {
    /// The ids from position 0.
    ids: Vec<u32>,
    /// The position of the first id received and the compressed ids from there.
    received: Option<(usize, Vec<u32>)>,
}

impl TokenLog {
    /// Writes `tokens` at the positions `first..`, after the ids of the positions before `first`.
    ///
    /// Refused, with nothing written, when the log holds fewer than `first` ids.
    pub(super) fn open(&mut self, first: usize, tokens: &[u32]) -> Result<()> {
        if self.ids.len() < first {
            crate::tensor::bail!(
                "the token log holds {} positions but the forward starts at {first}",
                self.ids.len()
            );
        }
        self.received = None;
        self.ids.truncate(first);
        self.ids.extend_from_slice(tokens);
        Ok(())
    }

    /// Holds the compressed ids `compressed` of the positions `from..` in place of any id.
    pub(super) fn receive(&mut self, from: usize, compressed: Vec<u32>) {
        self.ids.clear();
        self.received = Some((from, compressed));
    }

    /// The first position whose id an engram row at `first` reads.
    pub(super) fn window_start(meta: &EngramMeta, first: usize) -> usize {
        (first + 1).saturating_sub(meta.max_ngram)
    }

    /// The compressed ids of the positions `window_start(first)..first + rows`, oldest first.
    ///
    /// They are every id the engram rows of the positions `first..first + rows` read.
    pub(super) fn window(&self, meta: &EngramMeta, first: usize, rows: usize) -> Result<Vec<u32>> {
        let start = Self::window_start(meta, first);
        match &self.received {
            Some((from, compressed)) => {
                if start < *from || first + rows > from + compressed.len() {
                    crate::tensor::bail!(
                        "the token log received {} ids from position {from}, not those of the {rows} rows from position {first}",
                        compressed.len()
                    );
                }
                Ok(compressed[start - from..first + rows - from].to_vec())
            }
            None => {
                if self.ids.len() < first + rows {
                    crate::tensor::bail!(
                        "the token log holds {} positions, not the {rows} from position {first}",
                        self.ids.len()
                    );
                }
                Ok(self.ids[start..first + rows]
                    .iter()
                    .map(|&t| meta.compress(t))
                    .collect())
            }
        }
    }

    /// The table rows engram layer `which` reads for the positions `first..first + rows`.
    ///
    /// A position hashes the compressed ids of the last `max_ngram` positions up to it, oldest
    /// first.
    pub(super) fn engram_hashes(
        &self,
        meta: &EngramMeta,
        which: usize,
        first: usize,
        rows: usize,
    ) -> Result<Vec<Vec<u64>>> {
        let window = self.window(meta, first, rows)?;
        let start = Self::window_start(meta, first);
        Ok((first..first + rows)
            .map(|p| {
                let from = Self::window_start(meta, p) - start;
                meta.hash(which, &window[from..=p - start])
            })
            .collect())
    }
}

/// The token log of the sequence the model's layers hold, shared by the layers that read it.
#[derive(Default)]
pub struct TokenCarry(std::sync::Mutex<TokenLog>);

impl TokenCarry {
    fn with<R>(&self, f: impl FnOnce(&mut TokenLog) -> R) -> R {
        let mut held = self.0.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut held)
    }

    /// Opens the forward of `tokens` at the positions `first..`, as [`TokenLog::open`] does.
    pub fn open(&self, first: usize, tokens: &[u32]) -> Result<()> {
        self.with(|log| log.open(first, tokens))
    }

    /// Holds `tokens` as the whole sequence.
    pub fn reset(&self, tokens: &[u32]) {
        self.with(|log| {
            *log = TokenLog {
                ids: tokens.to_vec(),
                received: None,
            }
        });
    }

    /// Holds the compressed ids `compressed` of the positions `from..`, as
    /// [`TokenLog::receive`] does.
    pub(super) fn receive(&self, from: usize, compressed: Vec<u32>) {
        self.with(|log| log.receive(from, compressed));
    }

    /// The compressed ids the engram rows of the positions `first..first + rows` read, as
    /// [`TokenLog::window`] gives them.
    pub(super) fn window(&self, meta: &EngramMeta, first: usize, rows: usize) -> Result<Vec<u32>> {
        self.with(|log| log.window(meta, first, rows))
    }

    /// Exchanges the log held with `other`.
    fn exchange(&self, other: &mut TokenLog) {
        self.with(|log| std::mem::swap(log, other));
    }
}

/// The engram a layer adds to its residual before its token mixer.
pub struct LayerEngram {
    pub block: EngramBlock,
    /// The hash layout of every engram layer of the model.
    pub meta: Arc<EngramMeta>,
    /// The layer's index among the engram layers.
    pub which: usize,
    /// Shared by every layer of the model and its head.
    pub tokens: Arc<TokenCarry>,
}

impl LayerEngram {
    /// The residual rows `r` `[1, rows, streams * hidden]` at the positions `first..` with the
    /// engram added, hashed from the ids `log` holds.
    pub(super) fn add(&self, r: &Tensor, log: &TokenLog, first: usize) -> Result<Tensor> {
        let rows = r.dims3()?.1;
        let hashes = log.engram_hashes(&self.meta, self.which, first, rows)?;
        self.block.forward(r, &hashes)
    }
}

/// The residual rows `r` at the positions `first..` with the layer's engram added, hashed from
/// the model's token log.
///
/// `r` itself for a layer without an engram.
/// A layer a block draft reads adds the result to the draft's capture.
fn with_engram(block: &SinkhornBlock, r: &Tensor, first: usize) -> Result<Tensor> {
    let r = match block.engram.as_ref() {
        Some(e) => e.tokens.with(|log| e.add(r, log, first))?,
        None => r.clone(),
    };
    if let Some(c) = block.capture.as_ref() {
        c.log.add(c.slot, first, &r)?;
    }
    Ok(r)
}

/// Routing that gives up exactness for speed, and what it saved.
///
/// The default routes exactly.
#[derive(Default)]
pub struct LossyRouting {
    pub truncation: Truncation,
    /// The experts counted as in memory.
    ///
    /// `None` counts none: the layer reads every routed expert from its mapping on use.
    pub resident: Option<HashSet<usize>>,
    /// Slots of experts not in memory that the truncation removed or replaced, since last
    /// taken.
    pub saved: AtomicU64,
    /// Routes a measurement imposes on the next single-row forwards, 1 per forward, oldest first.
    ///
    /// Empty in service.
    pub script: std::sync::Mutex<std::collections::VecDeque<Vec<usize>>>,
    /// Routes `script` holds, read before the lock so that a layer with none pays no lock.
    pub scripted: AtomicUsize,
}

impl LossyRouting {
    /// The next route a measurement imposes, `None` when none is queued.
    fn next_scripted(&self) -> Option<Vec<usize>> {
        if self.scripted.load(Ordering::Relaxed) == 0 {
            return None;
        }
        let route = self
            .script
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop_front();
        if route.is_some() {
            self.scripted.fetch_sub(1, Ordering::Relaxed);
        }
        route
    }
}

/// The routed experts of a layer in every file of the checkpoint that stores them.
///
/// Empty for a model read from 1 file.
#[derive(Default)]
pub struct ExpertVariants {
    /// Per variant, per expert.
    ///
    /// Variant 0 is the model's own file.
    pub stored: Vec<Vec<BlockScaledExpert>>,
    /// Per expert, the variant the layer reads.
    pub chosen: Vec<usize>,
}

/// What a routed feed-forward tells its watcher at each forward.
///
/// `None` with the block's input rows, `[rows, hidden]` row-major, once per forward.
/// `Some(e)` once per expert `e` the forward routed to, in id order.
/// The 2nd argument is then the rows routed to `e`, in row order.
/// The 3rd is the rows its down projection took, each times its route weight.
pub type ExpertObserver = dyn Fn(Option<usize>, &[f32], &[f32]) + Send + Sync;

/// The watcher of a routed feed-forward, `None` in service.
#[derive(Default)]
pub struct ExpertWatch(std::sync::RwLock<Option<Arc<ExpertObserver>>>);

impl ExpertWatch {
    /// Sets the watcher, `None` removes it.
    pub fn set(&self, observer: Option<Arc<ExpertObserver>>) {
        *self.0.write().unwrap_or_else(|e| e.into_inner()) = observer;
    }

    fn get(&self) -> Option<Arc<ExpertObserver>> {
        self.0.read().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// The feed-forward of a layer: a sqrt-softplus router over routed experts, beside a shared
/// expert.
pub struct RoutedFfn {
    /// `[experts, hidden]`, dense F32.
    pub router: Projection,
    /// Added to the scores for the choice only, `[experts]`.
    pub bias: Vec<f32>,
    /// The experts the layer reads, by id.
    ///
    /// Each one is the variant `variants` chose, when the experts have variants.
    pub experts: Vec<BlockScaledExpert>,
    pub shared: Option<BlockScaledExpert>,
    /// Experts a row is routed to.
    pub used: usize,
    pub temperature: f32,
    /// The chosen weights are divided by their sum.
    pub normalize: bool,
    /// Multiplies every chosen weight.
    pub scale: f32,
    /// The bound of the SwiGLU, `0` for none.
    pub limit: f32,
    pub lossy: LossyRouting,
    pub variants: ExpertVariants,
    /// The experts the cards keep, each the variant `variants` chose.
    pub card: CardExperts,
    /// A watched forward runs every routed expert on the host.
    pub watch: ExpertWatch,
    /// The layer's share of the host tier, `None` when the host keeps no tier.
    pub host: Option<Arc<super::wide_sparse_host_tier::LayerKeep>>,
}

impl RoutedFfn {
    /// Bytes of expert `e` as variant `v` stores it, `None` when there is no such variant.
    ///
    /// Variant 0 of a layer without variants is the expert the layer reads.
    pub fn variant_bytes(&self, e: usize, v: usize) -> Option<u64> {
        let expert = match self.variants.stored.is_empty() {
            true => self.experts.get(e).filter(|_| v == 0),
            false => self.variants.stored.get(v).and_then(|s| s.get(e)),
        }?;
        Some((expert.gate.bytes() + expert.up.bytes() + expert.down.bytes()) as u64)
    }

    /// Reads expert `e` from variant `v` from now on.
    ///
    /// False when there is no such variant, and the layer reads what it read.
    pub fn choose_variant(&mut self, e: usize, v: usize) -> bool {
        let stored = self.variants.stored.get(v).and_then(|s| s.get(e)).cloned();
        match (
            stored,
            self.experts.get_mut(e),
            self.variants.chosen.get_mut(e),
        ) {
            (Some(expert), Some(slot), Some(chosen)) => {
                *slot = expert;
                *chosen = v;
                true
            }
            _ => v == 0 && self.variants.stored.is_empty(),
        }
    }

    /// Each row's experts and their weights before the scale.
    ///
    /// A row takes the truncated choice when the truncation gives up exactness, the plain
    /// choice otherwise.
    pub(super) fn route(&self, logits: &[Vec<f32>]) -> Vec<(Vec<usize>, Vec<f32>)> {
        // A queued route stands in for the choice of a single row, weighed by the gate's row.
        if let [row] = logits {
            if let Some(chosen) = self.lossy.next_scripted() {
                return vec![sqrt_softplus_weigh(
                    row,
                    chosen,
                    self.temperature,
                    self.normalize,
                )];
            }
        }
        let t = self.lossy.truncation;
        if !t.active() {
            return logits
                .par_iter()
                .map(|row| {
                    sqrt_softplus_choice(
                        row,
                        Some(self.bias.as_slice()),
                        self.used,
                        self.temperature,
                        self.normalize,
                    )
                })
                .collect();
        }
        let resident = |e: usize| {
            self.lossy
                .resident
                .as_ref()
                .is_some_and(|set| set.contains(&e))
        };
        let picks: Vec<(Vec<usize>, Vec<f32>, u64)> = logits
            .par_iter()
            .map(|row| {
                sqrt_softplus_truncated_choice(
                    row,
                    Some(self.bias.as_slice()),
                    self.used,
                    self.temperature,
                    self.normalize,
                    t,
                    &resident,
                )
            })
            .collect();
        let saved: u64 = picks.iter().map(|p| p.2).sum();
        self.lossy.saved.fetch_add(saved, Ordering::Relaxed);
        picks.into_iter().map(|(ids, w, _)| (ids, w)).collect()
    }

    /// Notes in the process's expert counters the pairs of `assign`: those of the experts in
    /// `on_card` served by the card, those of `on_host` computed by the host, and the experts a
    /// single decoded row routed to.
    ///
    /// `kept[i]` says whether the host tier keeps `on_host[i]`.
    /// The bytes of the experts it does not keep are read from the file mapping.
    fn note_served(
        &self,
        assign: &[Vec<(usize, f32)>],
        on_card: &[usize],
        on_host: &[usize],
        kept: &[bool],
        n: usize,
    ) {
        let counters = crate::inference::expert_counters::global();
        if !counters.on() {
            return;
        }
        let layer = self as *const Self as usize;
        let pairs = |ids: &[usize]| ids.iter().map(|&e| assign[e].len() as u64).sum::<u64>();
        let routed = on_card.iter().chain(on_host);
        counters.routed(
            layer,
            routed.flat_map(|&e| std::iter::repeat_n(e, assign[e].len())),
        );
        counters.served(pairs(on_card), pairs(on_host));
        let mut by_card: Vec<(usize, u64)> = Vec::new();
        for &e in on_card {
            let Some(ordinal) = self.card.card_of(e) else {
                continue;
            };
            let p = assign[e].len() as u64;
            match by_card.iter_mut().find(|(o, _)| *o == ordinal) {
                Some((_, n)) => *n += p,
                None => by_card.push((ordinal, p)),
            }
        }
        for (ordinal, p) in by_card {
            counters.card(ordinal, p);
        }
        let bytes = |e: usize| {
            let x = &self.experts[e];
            (x.gate.bytes() + x.up.bytes() + x.down.bytes()) as u64
        };
        let kept_at = |i: usize| kept.get(i).copied().unwrap_or(false);
        let tier: Vec<usize> = (0..on_host.len()).filter(|&i| kept_at(i)).collect();
        counters.host(
            on_host.iter().map(|&e| bytes(e)).sum(),
            tier.iter().map(|&i| assign[on_host[i]].len() as u64).sum(),
        );
        let read: u64 = (0..on_host.len())
            .filter(|&i| !kept_at(i))
            .map(|i| bytes(on_host[i]))
            .sum();
        counters.read(false, read);
        if n == 1 {
            let mut now: Vec<usize> = on_card.iter().chain(on_host).copied().collect();
            now.sort_unstable();
            counters.decoded(layer, &now);
        }
    }

    /// The block over the rows of `x` `[.., hidden]` F32 on the host, in `x`'s shape.
    ///
    /// Each row starts from its shared expert's output.
    /// The routed experts run by id, each once over the rows routed to it, in row order.
    /// A row adds the outputs of its experts in that order, each times its weight.
    ///
    /// A card that keeps the router takes a single row's logits, with the host's bits.
    /// A card that keeps the shared expert runs it beside the host's experts.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_routed(x, None)
    }

    /// [`Self::forward`] over a single row whose router logits `logits` the card gave, with the
    /// host's bits.
    ///
    /// `None` runs the router as [`Self::forward`] does.
    pub(super) fn forward_routed(&self, x: &Tensor, logits: Option<Vec<f32>>) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let dim = dims.last().copied().unwrap_or(0);
        let n = x.elem_count() / dim.max(1);
        let x2 = x.reshape((n, dim))?;
        let rows = x2.flatten_all()?.to_vec1::<f32>()?;
        let host = Device::Cpu;
        let logits = match logits.filter(|_| n == 1) {
            Some(row) => vec![row],
            None => staged("experts: router", &host, false, || {
                Ok(
                    match (n == 1).then(|| self.card.router_logits(&rows)).flatten() {
                        Some(row) => vec![row],
                        None => self.router.apply_here(&x2)?.to_vec2::<f32>()?,
                    },
                )
            })?,
        };
        let picks = self.route(&logits);
        // Per expert, the rows routed to it in row order, with their weights.
        let mut assign: Vec<Vec<(usize, f32)>> = vec![Vec::new(); self.experts.len()];
        for (t, (chosen, w)) in picks.into_iter().enumerate() {
            for (&e, &wi) in chosen.iter().zip(&w) {
                match assign.get_mut(e) {
                    Some(rows) => rows.push((t, wi * self.scale)),
                    None => crate::tensor::bail!("routed to expert {e} of {}", self.experts.len()),
                }
            }
        }
        let active: Vec<usize> = (0..assign.len())
            .filter(|&e| !assign[e].is_empty())
            .collect();
        let watch = self.watch.get();
        // The experts a card keeps run there, the others on the host's cores.
        let (on_card, on_host): (Vec<usize>, Vec<usize>) = active
            .iter()
            .partition(|&&e| watch.is_none() && self.card.holds(e));
        let tier = self.host.as_deref();
        // Every host expert's bytes are asked for at once, on the tier's reader threads where it
        // has them, so the reads overlap the bookkeeping and the compute.
        let asked = std::time::Instant::now();
        let reads = tier.and_then(|t| t.read_now(&on_host));
        if reads.is_none() {
            on_host
                .par_iter()
                .for_each(|&e| self.experts[e].will_need());
        }
        let mut asked_ns = asked.elapsed().as_nanos() as u64;
        // The host tier scores the guess the layer before made and says what it keeps.
        let prepared = std::time::Instant::now();
        if let Some(t) = tier {
            t.opens(&active, n);
        }
        // Per host expert, the last row routed to it.
        let last: Vec<usize> = on_host
            .iter()
            .map(|&e| assign[e].last().map_or(0, |&(t, _)| t))
            .collect();
        let kept = tier.map_or_else(
            || vec![false; on_host.len()],
            |t| {
                let routes: Vec<(usize, usize)> = on_host
                    .iter()
                    .zip(&last)
                    .map(|(&e, &l)| (l, assign[e].len()))
                    .collect();
                t.kept(&on_host, &routes, &on_card, n)
            },
        );
        self.note_served(&assign, &on_card, &on_host, &kept, n);
        crate::inference::stage_prof::prof_record(
            "experts: tier",
            prepared.elapsed().as_nanos() as u64,
        );
        let ahead = std::time::Instant::now();
        // A decoded row reads ahead what the next layer is expected to route to.
        if let Some(t) = tier.filter(|_| n == 1) {
            t.read_ahead(&x2);
        }
        asked_ns += ahead.elapsed().as_nanos() as u64;
        crate::inference::stage_prof::prof_record("experts: reads asked", asked_ns);
        let shared_on_card = self.shared.is_some() && self.card.keeps_shared();
        let host_shared = |s: &BlockScaledExpert| -> Result<Vec<Vec<f32>>> {
            s.forward(&x2, self.limit)?.to_vec2::<f32>()
        };
        let mut y: Vec<Vec<f32>> = match &self.shared {
            Some(s) if !shared_on_card => {
                staged("experts: shared on the host", &host, false, || {
                    host_shared(s)
                })?
            }
            _ => vec![vec![0f32; dim]; n],
        };
        if let Some(w) = &watch {
            w(None, &rows, &[]);
        }
        let picked = |e: usize| -> Vec<f32> {
            assign[e]
                .iter()
                .flat_map(|&(t, _)| rows[t * dim..(t + 1) * dim].iter().copied())
                .collect()
        };
        // Per expert, its output rows and, in a watched forward, the rows its down projection
        // took.
        let host_run = |ids: &[usize]| -> Result<Vec<(Vec<f32>, Vec<f32>)>> {
            ids.par_iter()
                .map(|&e| -> Result<(Vec<f32>, Vec<f32>)> {
                    let xe = Tensor::from_vec(picked(e), (assign[e].len(), dim), &Device::Cpu)?;
                    let seen = std::cell::RefCell::new(Vec::new());
                    let keep = |h: &[f32]| *seen.borrow_mut() = h.to_vec();
                    let out = self.experts[e]
                        .forward_seen(
                            &xe,
                            self.limit,
                            watch.as_ref().map(|_| &keep as &dyn Fn(&[f32])),
                        )?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    Ok((out, seen.into_inner()))
                })
                .collect()
        };
        // The cards and the cores run at once: a scoped thread drives the cards, 1 thread a card.
        // The card that keeps the shared expert runs it first, then its experts.
        let card_rows: Vec<Vec<usize>> = on_card
            .iter()
            .map(|&e| assign[e].iter().map(|&(t, _)| t).collect())
            .collect();
        let (shared_out, card_outs, host_outs) = std::thread::scope(|scope| {
            let card = (!on_card.is_empty() || shared_on_card).then(|| {
                scope.spawn(|| {
                    self.card.run_cards(
                        &on_card,
                        &card_rows,
                        &rows,
                        dim,
                        self.limit,
                        shared_on_card,
                    )
                })
            });
            let started = std::time::Instant::now();
            let host = host_run(&on_host);
            let ran = started.elapsed();
            let (shared, experts) = card
                .and_then(|h| h.join().ok())
                .unwrap_or_else(|| (None, vec![None; on_card.len()]));
            crate::inference::stage_prof::prof_record("experts: host", ran.as_nanos() as u64);
            crate::inference::stage_prof::prof_record(
                "experts: card wait",
                (started.elapsed() - ran).as_nanos() as u64,
            );
            (shared, experts, host)
        });
        // The reads of this forward not yet made are no longer asked.
        drop(reads);
        let host_outs = host_outs?;
        // The tier admits or lets go what the host computed, by the kind of rows.
        if let Some(t) = tier {
            t.computed(&on_host, &last, n);
        }
        // A card that failed leaves the shared expert to the host.
        if let Some(s) = self.shared.as_ref().filter(|_| shared_on_card) {
            y = match shared_out {
                Some(flat) => flat.chunks(dim.max(1)).map(<[f32]>::to_vec).collect(),
                None => host_shared(s)?,
            };
        }
        // A card that failed leaves its experts to the host, which gives the same bits.
        let failed: Vec<usize> = (0..on_card.len())
            .filter(|&i| card_outs[i].is_none())
            .collect();
        let mut again = host_run(&failed.iter().map(|&i| on_card[i]).collect::<Vec<_>>())?
            .into_iter()
            .map(|(out, _)| out);
        let card_outs: Vec<Vec<f32>> = card_outs
            .into_iter()
            .map(|out| out.or_else(|| again.next()).unwrap_or_default())
            .collect();
        // Each active expert's outputs, in the order of `active`.
        let mut by_id: std::collections::HashMap<usize, Vec<f32>> =
            on_card.into_iter().zip(card_outs).collect();
        let mut seen: std::collections::HashMap<usize, Vec<f32>> = Default::default();
        for (e, (out, h)) in on_host.into_iter().zip(host_outs) {
            by_id.insert(e, out);
            seen.insert(e, h);
        }
        if let Some(w) = &watch {
            for &e in &active {
                let h = seen.remove(&e).unwrap_or_default();
                let inter = h.len() / assign[e].len().max(1);
                let weighted: Vec<f32> = h
                    .chunks(inter.max(1))
                    .zip(&assign[e])
                    .flat_map(|(row, &(_, wt))| row.iter().map(move |v| v * wt))
                    .collect();
                w(Some(e), &picked(e), &weighted);
            }
        }
        let outs: Vec<Vec<f32>> = active
            .iter()
            .map(|e| by_id.remove(e).unwrap_or_default())
            .collect();
        for (out, &e) in outs.iter().zip(&active) {
            for (r, &(t, w)) in assign[e].iter().enumerate() {
                for (a, b) in y[t].iter_mut().zip(&out[r * dim..(r + 1) * dim]) {
                    *a += w * b;
                }
            }
        }
        let flat: Vec<f32> = y.into_iter().flatten().collect();
        Tensor::from_vec(flat, (n, dim), &Device::Cpu)?.reshape(dims)
    }
}

/// What a layer holds around its token mixer: the 2 Sinkhorn mixers, the 2 norms, the token
/// mixer's output projection and the feed-forward.
pub struct SinkhornBlock {
    pub attn_mix: SinkhornMixer,
    pub ffn_mix: SinkhornMixer,
    /// The norm of the token mixer's input, `[hidden]` F32.
    pub attn_norm: Tensor,
    /// The norm of the feed-forward's input, `[hidden]` F32.
    pub ffn_norm: Tensor,
    pub eps: f32,
    /// `[hidden, mixer output]`: what the token mixer's output goes through.
    pub output: Projection,
    pub ffn: RoutedFfn,
    /// Shared by every layer of the model and its head.
    pub carry: Arc<PreMixCarry>,
    /// What an engram layer adds to the residual before the token mixer reads it.
    pub engram: Option<LayerEngram>,
    /// The 2 mixers and the 2 norms on a card, when the load keeps them there.
    pub card_mix: super::wide_sparse_card_stream::CardSinkhorn,
    /// Where a layer a block draft reads adds its input.
    pub capture: Option<LayerCapture>,
}

/// The token mixer's half of a layer over the residual rows `r` `[1, seq, streams * hidden]` at
/// positions `pos..`, collapsed by `pre`.
///
/// Returns the residual after the half, the pre mix the feed-forward collapses with, and the
/// token mixer's input.
pub(super) fn attention_half(
    block: &SinkhornBlock,
    mixer: &mut mixer::Mixer,
    r: &Tensor,
    pre: &[Vec<f32>],
    pos: usize,
) -> Result<(Tensor, Vec<Vec<f32>>, Tensor)> {
    let host = Device::Cpu;
    let (mixes, input) = staged("mix_token", &host, false, || {
        let mixes = block.attn_mix.mixes(r)?;
        let input = rms_norm(&block.attn_mix.pre(r, pre)?, &block.attn_norm, block.eps)?;
        Ok((mixes, input))
    })?;
    let out = token_mixer_out(block, mixer, &input, pos)?;
    let next = staged("combine", &host, false, || {
        block.attn_mix.post(&out, r, &mixes.post, &mixes.comb)
    })?;
    Ok((next, mixes.pre, input))
}

/// The token mixer's output `[1, seq, hidden]` for its input rows `input` at positions `pos..`,
/// through the layer's output projection.
pub(super) fn token_mixer_out(
    block: &SinkhornBlock,
    mixer: &mut mixer::Mixer,
    input: &Tensor,
    pos: usize,
) -> Result<Tensor> {
    let host = Device::Cpu;
    let mixed = staged("token mixer", &host, false, || mixer.forward(input, pos))?;
    let (b, seq, width) = mixed.dims3()?;
    // A card that runs the output projection after the mixer gives its rows already projected.
    match mixer.projects_output() {
        true => Ok(mixed),
        false => staged("output projection", &host, false, || {
            block
                .output
                .apply_here(&mixed.reshape((b * seq, width))?)?
                .reshape((b, seq, block.attn_mix.dim()))
        }),
    }
}

/// The residual `r` of the row of sequence `seq` at `pos` with the layer's engram added, `r`
/// itself for a layer without an engram.
pub(super) fn engram_row(
    block: &SinkhornBlock,
    r: Tensor,
    seq: &SequenceMixers,
    pos: usize,
) -> Result<Tensor> {
    match block.engram.as_ref() {
        Some(e) => staged("engram", &Device::Cpu, false, || {
            e.add(&r, &seq.tokens, pos)
        }),
        None => Ok(r),
    }
}

/// `run` over the mixer of layer `il` holding the state of sequence `seq` and its band cell.
///
/// The mixer and the cell hold their own states again once `run` returns.
pub(super) fn in_sequence<R>(
    mixer: &mut mixer::Mixer,
    il: usize,
    seq: &mut SequenceMixers,
    cell: &BandCell,
    run: impl FnOnce(&mut mixer::Mixer) -> Result<R>,
) -> Result<R> {
    staged("exchange", &Device::Cpu, false, || {
        exchange_state(mixer, &mut seq.states[il])?;
        cell.exchange(&seq.cell);
        Ok(())
    })?;
    let out = run(mixer);
    staged("exchange", &Device::Cpu, false, || {
        cell.exchange(&seq.cell);
        exchange_state(mixer, &mut seq.states[il])
    })?;
    out
}

/// The feed-forward's half of a layer over the residual rows `mid` `[1, seq, streams * hidden]`,
/// collapsed by `attn_pre`: the residual after the half and the pre mix the next layer collapses
/// with.
pub(super) fn ffn_half(
    block: &SinkhornBlock,
    mid: &Tensor,
    attn_pre: &[Vec<f32>],
) -> Result<(Tensor, Vec<Vec<f32>>)> {
    let (mixes, input) = staged("mix_ffn", &Device::Cpu, false, || {
        let mixes = block.ffn_mix.mixes(mid)?;
        let input = rms_norm(
            &block.ffn_mix.pre(mid, attn_pre)?,
            &block.ffn_norm,
            block.eps,
        )?;
        Ok((mixes, input))
    })?;
    let out = staged("experts", &Device::Cpu, false, || block.ffn.forward(&input))?;
    let next = staged("combine", &Device::Cpu, false, || {
        block.ffn_mix.post(&out, mid, &mixes.post, &mixes.comb)
    })?;
    Ok((next, mixes.pre))
}

impl GenericTransformerLayer {
    /// One layer over the residual rows `x` `[1, seq, streams * hidden]` at positions
    /// `index_pos..`, on the host.
    pub(super) fn forward_sinkhorn(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let r = x
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let seq = r.dims3()?.1;
        if self.spec_record {
            self.spec_begin(&r, None, index_pos)?;
        }
        // The rows of a verify each attend as the decode step of their row.
        let stepped = seq > 1 && index_pos > 0 && crate::tensor::quantized::DecodeOrder::active();
        let (mid, attn_pre, input) = {
            let block = self
                .sinkhorn
                .as_ref()
                .ok_or_else(|| crate::tensor::Error::msg("a Sinkhorn layer without its block"))?;
            let pre = block.carry.get(index_pos, seq)?;
            if stepped {
                let (mut mids, mut pres, mut inputs) = (Vec::new(), Vec::new(), Vec::new());
                for j in 0..seq {
                    let rj = r.narrow(1, j, 1)?.contiguous()?;
                    let rj = with_engram(block, &rj, index_pos + j)?;
                    let (m, p, i) =
                        attention_half(block, &mut self.mixer, &rj, &pre[j..j + 1], index_pos + j)?;
                    mids.push(m);
                    pres.extend(p);
                    inputs.push(i);
                }
                let mids: Vec<&Tensor> = mids.iter().collect();
                let inputs: Vec<&Tensor> = inputs.iter().collect();
                (Tensor::cat(&mids, 1)?, pres, Tensor::cat(&inputs, 1)?)
            } else {
                let r = with_engram(block, &r, index_pos)?;
                attention_half(block, &mut self.mixer, &r, &pre, index_pos)?
            }
        };
        self.spec_input(&input);
        let block = self
            .sinkhorn
            .as_ref()
            .ok_or_else(|| crate::tensor::Error::msg("a Sinkhorn layer without its block"))?;
        let mixes = block.ffn_mix.mixes(&mid)?;
        let input = rms_norm(
            &block.ffn_mix.pre(&mid, &attn_pre)?,
            &block.ffn_norm,
            block.eps,
        )?;
        let out = staged("experts", &Device::Cpu, false, || block.ffn.forward(&input))?;
        let next = block.ffn_mix.post(&out, &mid, &mixes.post, &mixes.comb)?;
        block.carry.put(index_pos, seq, mixes.pre);
        Ok(next)
    }
}

/// What a model whose layers mix their residual through Sinkhorn projections holds around them:
/// the state the layers pass to each other, the final norm and the head.
pub struct WideSparseHead {
    /// Shared with every layer.
    pub carry: Arc<PreMixCarry>,
    /// Shared with every compressed band layer.
    pub cell: Arc<BandCell>,
    /// Parallel copies of the hidden state.
    pub streams: usize,
    /// The final norm, `[hidden]` F32.
    pub norm: Tensor,
    pub eps: f32,
    /// `[vocab, hidden]`.
    pub head: Projection,
    /// Shared with every engram layer, `None` for a model without one.
    pub tokens: Option<Arc<TokenCarry>>,
    /// The hash layout of every engram layer, `None` for a model without one.
    pub engram: Option<Arc<EngramMeta>>,
    /// The block draft, `None` for a model without one.
    pub draft: Option<BlockDraft>,
    /// The description the layers were built from.
    pub meta: Arc<super::wide_sparse_config::WideSparseMeta>,
    /// Per layer of the stack, the experts from the most to the least routed to.
    ///
    /// The file's own, or a companion file's when the file carries none.
    /// Empty lists without a prior.
    pub prior: Vec<Vec<usize>>,
    /// Per layer of the stack, the experts ranked by a routing profile, the prior ordering the
    /// ones it never counted.
    ///
    /// `None` without a profile that counted a route.
    pub heat: Option<Vec<Vec<usize>>>,
    /// Per card ordinal, the bytes the routed experts may take on it, cards in ordinal order.
    ///
    /// Empty when no card keeps routed experts.
    pub card_rooms: Vec<(usize, u64)>,
    /// Per card the plan keeps a cache on, the part of the cache's room the routed experts
    /// borrow until a sequence needs it.
    pub lent: Vec<super::wide_sparse_card_tier::LentRoom>,
    /// Per card the load placed weights on, what a byte of each kept there saves.
    pub savings: Vec<super::wide_always_read::CardSaving>,
    /// Per layer of the stack, per expert, the share of the tokens a routing profile routed to
    /// it.
    ///
    /// `None` without a profile that counted a route, and for a layer it never counted.
    pub shares: Option<Vec<Option<Vec<f64>>>>,
    /// The head on a card, when the load keeps it there.
    pub card_head: super::wide_always_read::CardProjection,
    /// Whether a decoded row's final collapse runs on the card that keeps the head.
    pub head_collapse: super::wide_sparse_card_stream::HeadCollapse,
}

impl WideSparseHead {
    /// The residual the rows of `hidden` `[1, seq, hidden]` at `index_pos..` start from: 1 copy
    /// of each row per stream.
    ///
    /// The carry and the band cell open for the forward over those rows.
    /// The token log and the block draft's capture take `input_ids`, the ids of those rows.
    pub fn widen(&self, hidden: &Tensor, input_ids: &Tensor, index_pos: usize) -> Result<Tensor> {
        let seq = hidden.dim(1)?;
        if self.tokens.is_some() || self.draft.is_some() {
            let ids = input_ids
                .flatten_all()?
                .to_dtype(DType::U32)?
                .to_vec1::<u32>()?;
            if let Some(tokens) = &self.tokens {
                tokens.open(index_pos, &ids)?;
            }
            if let Some(draft) = &self.draft {
                draft.capture.open(index_pos, &ids);
            }
        }
        self.carry.open(index_pos, seq, self.streams);
        self.cell.open(index_pos, seq);
        super::wide::widen_embedding(hidden, self.streams)
    }

    /// The logits of the residual rows `hidden` `[1, seq, streams * hidden]`: every row's
    /// `[1, seq, vocab]` with `all_positions`, else the last row's `[1, vocab]`.
    ///
    /// `collapse` is a Sinkhorn mixer of the model.
    /// The rows collapse with the pre mix the last layer left.
    pub fn logits(
        &self,
        collapse: &SinkhornMixer,
        hidden: &Tensor,
        all_positions: bool,
    ) -> Result<Tensor> {
        let r = hidden.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let (b, seq, _) = r.dims3()?;
        let pre = self.carry.last(seq)?;
        let (rows, pre) = match all_positions {
            true => (r, pre),
            false => (
                r.narrow(1, seq - 1, 1)?.contiguous()?,
                pre[seq - 1..].to_vec(),
            ),
        };
        let n = rows.dim(1)?;
        let h = rms_norm(&collapse.pre(&rows, &pre)?, &self.norm, self.eps)?;
        let h = h.reshape((b * n, collapse.dim()))?;
        // A card that keeps the head runs it, else the host.
        let logits = match self.card_head.apply(&h) {
            Some(logits) => logits,
            None => self.head.apply_here(&h)?,
        };
        match all_positions {
            true => {
                let vocab = logits.dim(1)?;
                logits.reshape((b, n, vocab))
            }
            false => Ok(logits),
        }
    }
}

impl GenericHeteroTransformer {
    /// Bytes of the weights every token of a Sinkhorn wide-residual model reads, as held.
    ///
    /// Per layer: the token mixer's resident weights, its output projection, the router with
    /// its bias, the shared expert and the engram projection.
    /// Then the head.
    /// The routed experts are not counted.
    /// `None` for a model of another kind.
    pub fn resident_path_bytes(&self) -> Option<u64> {
        let mut bytes = self.wide_sparse.as_ref()?.head.bytes();
        for layer in &self.layers {
            let block = layer.sinkhorn.as_ref()?;
            bytes += match &layer.mixer {
                mixer::Mixer::LatentMqa(m) => m.resident_bytes(),
                mixer::Mixer::CompressedBand(b) => b.resident_bytes(),
                _ => return None,
            };
            let ffn = &block.ffn;
            let shared = ffn
                .shared
                .as_ref()
                .map_or(0, |s| s.gate.bytes() + s.up.bytes() + s.down.bytes());
            let bias = ffn.bias.len() * std::mem::size_of::<f32>();
            let engram = block.engram.as_ref().map_or(0, |e| e.block.wkv_bytes());
            // A mixer on a card that runs the output projection counts it among its own.
            let output = match layer.mixer.projects_output() {
                true => 0,
                false => block.output.bytes(),
            };
            bytes += output + ffn.router.bytes() + bias + shared + engram;
        }
        Some(bytes as u64)
    }

    /// Bytes of mixer state 1 sequence keeps once it holds `tokens` positions, summed over the
    /// layers.
    ///
    /// `None` for a model of another kind.
    pub fn sequence_state_bytes(&self, tokens: usize) -> Option<u64> {
        let meta = &self.wide_sparse.as_ref()?.meta;
        Some(
            (0..self.layers.len())
                .map(|il| meta.cache_bytes(il, tokens))
                .sum(),
        )
    }

    /// The routed feed-forward of every layer that has one, by layer index.
    fn routed_ffns(&mut self) -> impl Iterator<Item = (usize, &mut RoutedFfn)> {
        self.layers
            .iter_mut()
            .enumerate()
            .filter_map(|(il, l)| l.sinkhorn.as_mut().map(|b| (il, &mut b.ffn)))
    }

    /// Routed experts a token passes through beside the shared one, `0` without a routed
    /// feed-forward.
    pub fn routed_per_token(&self) -> usize {
        self.layers
            .iter()
            .find_map(|l| l.sinkhorn.as_ref())
            .map_or(0, |b| b.ffn.used)
    }

    /// Sets every routed feed-forward's truncation.
    pub fn set_truncation(&mut self, t: Truncation) {
        for (_, ffn) in self.routed_ffns() {
            ffn.lossy.truncation = t;
        }
    }

    /// The slots the truncation saved since last asked, summed over the layers.
    pub fn take_saved(&self) -> u64 {
        self.layers
            .iter()
            .filter_map(|l| l.sinkhorn.as_ref())
            .map(|b| b.ffn.lossy.saved.swap(0, Ordering::Relaxed))
            .sum()
    }

    /// Counts, per layer index, the given experts as the ones in memory.
    ///
    /// `None`, or a layer past the end of `per_layer`, counts none.
    pub fn set_resident_override(&mut self, per_layer: Option<&[HashSet<usize>]>) {
        for (il, ffn) in self.routed_ffns() {
            ffn.lossy.resident = per_layer.and_then(|p| p.get(il).cloned());
        }
    }

    /// The band cell the layers of a Sinkhorn wide residual share.
    fn shared_cell(&self) -> Result<Arc<BandCell>> {
        match self.wide_sparse.as_ref() {
            Some(ws) => Ok(Arc::clone(&ws.cell)),
            None => crate::tensor::bail!("a batch of sequences needs a Sinkhorn wide residual"),
        }
    }

    /// What a sequence keeps before its first token: no state in any layer.
    ///
    /// Refused for a model that is not a Sinkhorn wide residual over latent and band layers.
    pub fn sequence_mixers(&self) -> Result<SequenceMixers> {
        self.shared_cell()?;
        let states = self
            .layers
            .iter()
            .map(|l| match &l.mixer {
                mixer::Mixer::LatentMqa(_) => Ok(mixer::MixerState::LatentMqa(None)),
                mixer::Mixer::CompressedBand(_) => Ok(mixer::MixerState::CompressedBand(None)),
                _ => Err(crate::tensor::Error::msg(
                    "a batch of sequences keeps latent and band states only",
                )),
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(SequenceMixers {
            states,
            cell: BandCell::default(),
            tokens: TokenLog::default(),
        })
    }

    /// True when the token mixer of a layer runs on a card.
    pub fn mixers_on_card(&self) -> bool {
        #[cfg(feature = "gpu")]
        return self.layers.iter().any(|l| match &l.mixer {
            mixer::Mixer::LatentMqa(m) => m.card.is_some(),
            mixer::Mixer::CompressedBand(b) => b.base.card.is_some(),
            _ => false,
        });
        #[cfg(not(feature = "gpu"))]
        false
    }

    /// The token log the engram layers share, `None` for a model without one.
    fn shared_tokens(&self) -> Option<Arc<TokenCarry>> {
        self.wide_sparse.as_ref().and_then(|ws| ws.tokens.clone())
    }

    /// Exchanges every layer's mixer state with the one `seq` holds for that layer.
    fn exchange_states(&mut self, seq: &mut SequenceMixers) -> Result<()> {
        if seq.states.len() != self.layers.len() {
            crate::tensor::bail!(
                "a sequence keeps {} layer states for {} layers",
                seq.states.len(),
                self.layers.len()
            );
        }
        for (layer, held) in self.layers.iter_mut().zip(seq.states.iter_mut()) {
            exchange_state(&mut layer.mixer, held)?;
        }
        Ok(())
    }

    /// The prompt `tokens` of the sequence `seq` from position 0: the logits of its last row,
    /// `[1, vocab]`.
    ///
    /// The model runs its serial forward over the sequence's own states.
    /// It holds its own states again once the call returns.
    pub fn prefill_sequence(&mut self, tokens: &[u32], seq: &mut SequenceMixers) -> Result<Tensor> {
        self.with_sequence(seq, |model| {
            let ids = Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), &Device::Cpu)?;
            model.forward(&ids, 0)
        })
    }

    /// What `run` gives with the model over the state of the sequence `seq`: every layer's
    /// mixer state, the band cell and the token log.
    ///
    /// The model holds its own states again once the call returns.
    pub fn with_sequence<R>(
        &mut self,
        seq: &mut SequenceMixers,
        run: impl FnOnce(&mut Self) -> Result<R>,
    ) -> Result<R> {
        let cell = self.shared_cell()?;
        let log = self.shared_tokens();
        self.exchange_states(seq)?;
        cell.exchange(&seq.cell);
        if let Some(log) = &log {
            log.exchange(&mut seq.tokens);
        }
        let out = run(self);
        if let Some(log) = &log {
            log.exchange(&mut seq.tokens);
        }
        cell.exchange(&seq.cell);
        self.exchange_states(seq)?;
        out
    }

    /// 1 decode step of each sequence of `seqs`: token `tokens[j]` at `positions[j]` of
    /// sequence `j`, the logits of every row, `[seqs, vocab]`.
    ///
    /// The layers run in order, each over every row before the next.
    /// The token mixer runs per sequence, over the sequence's own mixer state and band cell.
    /// Each row keeps its own residual and the pre mix its next sublayer collapses with.
    /// Every step of a row runs the operations of the serial decode of its token, on the same
    /// values, so each row gives the bits of its sequence decoded alone.
    pub fn decode_sequences(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        seqs: &mut [&mut SequenceMixers],
    ) -> Result<Tensor> {
        let rows_in = tokens.len();
        if positions.len() != rows_in || seqs.len() != rows_in {
            crate::tensor::bail!(
                "{rows_in} tokens for {} positions and {} sequences",
                positions.len(),
                seqs.len()
            );
        }
        let (streams, carry) = match self.wide_sparse.as_ref() {
            Some(ws) => (ws.streams, Arc::clone(&ws.carry)),
            None => crate::tensor::bail!("a batch of sequences needs a Sinkhorn wide residual"),
        };
        let cell = self.shared_cell()?;
        let hashed = self.shared_tokens().is_some();
        if seqs.iter().any(|s| s.states.len() != self.layers.len()) {
            crate::tensor::bail!("a sequence keeps states for another model");
        }
        crate::tensor::quant_cpu::set_small_model_prefill(false);
        self.take_back_card_room(positions.iter().map(|p| p + 1).max().unwrap_or(0));

        // Each row's residual as the embedding stage of its decode starts it.
        // A sequence's own token log takes the row's token where an engram layer reads it.
        let emb_device = self.embeddings.embeddings().device();
        let mut rows = Vec::with_capacity(rows_in);
        for (seq, (&t, &pos)) in seqs.iter_mut().zip(tokens.iter().zip(positions)) {
            if hashed {
                seq.tokens.open(pos, &[t])?;
            }
            let ids = Tensor::from_vec(vec![t], (1, 1), &emb_device)?;
            let hidden = self.add_positions(self.embeddings.forward(&ids)?, pos)?;
            let hidden = match self.config.embed_scale {
                Some(scale) => (hidden * scale)?,
                None => hidden,
            };
            rows.push(super::wide::widen_embedding(&hidden, streams)?);
            seq.cell.open(pos, 1);
        }
        let mut pres: Vec<Vec<Vec<f32>>> =
            (0..rows_in).map(|_| vec![onto_first(streams)]).collect();

        // A single row keeps its residual on the cards through the layers that keep their
        // Sinkhorn mixers there.
        #[cfg(feature = "gpu")]
        if rows_in == 1 && self.layers_keep_sinkhorn_on_cards() {
            let row = rows
                .pop()
                .ok_or_else(|| crate::tensor::Error::msg("no row to decode"))?;
            use super::wide_sparse_card_stream::gpu::Decoded;
            let decoded = self.decode_row_on_cards(row, positions[0], &mut *seqs[0], &cell)?;
            return match decoded {
                Decoded::Rows(row, pre) => {
                    carry.put(positions[0], 1, vec![pre]);
                    staged("head", &Device::Cpu, false, || {
                        self.output_stage(row, false, 1, false)
                    })
                }
                Decoded::Logits(logits, pre) => {
                    carry.put(positions[0], 1, vec![pre]);
                    self.finish_logits(logits)
                }
            };
        }

        for il in 0..self.layers.len() {
            let layer = &mut self.layers[il];
            let mut mids = Vec::with_capacity(rows_in);
            let mut attn_pres = Vec::with_capacity(rows_in);
            for (j, seq) in seqs.iter_mut().enumerate() {
                let r = rows[j]
                    .to_device(&Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .contiguous()?;
                let block = layer.sinkhorn.as_ref().ok_or_else(|| {
                    crate::tensor::Error::msg("a Sinkhorn layer without its block")
                })?;
                let r = engram_row(block, r, seq, positions[j])?;
                let (mid, pre, _) = in_sequence(&mut layer.mixer, il, seq, &cell, |mixer| {
                    attention_half(block, mixer, &r, &pres[j], positions[j])
                })?;
                mids.push(mid);
                attn_pres.push(pre);
            }
            let block = layer
                .sinkhorn
                .as_ref()
                .ok_or_else(|| crate::tensor::Error::msg("a Sinkhorn layer without its block"))?;
            let each = rows
                .iter_mut()
                .zip(pres.iter_mut())
                .zip(mids.iter().zip(&attn_pres));
            for ((row, pre), (mid, attn_pre)) in each {
                (*row, *pre) = ffn_half(block, mid, attn_pre)?;
            }
        }

        // Each row's head collapses with the pre mix its last layer left.
        let mut logits = Vec::with_capacity(rows_in);
        for (j, row) in rows.into_iter().enumerate() {
            carry.put(positions[j], 1, std::mem::take(&mut pres[j]));
            logits.push(staged("head", &Device::Cpu, false, || {
                self.output_stage(row, false, 1, false)
            })?);
        }
        Tensor::cat(&logits.iter().collect::<Vec<_>>(), 0)
    }

    /// The prompts `prompts`, each from position 0, run layer after layer: every prompt crosses a
    /// layer before any prompt goes on to the next.
    ///
    /// The token mixer runs per prompt, over the prompt's own mixer state and band cell.
    /// The feed-forward of a layer runs once over the rows of every prompt, in prompt order.
    /// `progress` is told the layers done and the layers in all after each layer.
    /// No logits are computed.
    pub fn prefill_by_layer(
        &mut self,
        prompts: &[Vec<u32>],
        progress: &dyn Fn(usize, usize),
    ) -> Result<()> {
        let streams = match self.wide_sparse.as_ref() {
            Some(ws) => ws.streams,
            None => crate::tensor::bail!("a prefill by layer needs a Sinkhorn wide residual"),
        };
        let cell = self.shared_cell()?;
        let hashed = self.shared_tokens().is_some();
        let longest = prompts.iter().map(Vec::len).max().unwrap_or(0);
        self.give_back_for_prompt(0, longest);
        self.take_back_card_room(longest);
        let mut seqs = prompts
            .iter()
            .map(|_| self.sequence_mixers())
            .collect::<Result<Vec<_>>>()?;

        // Each prompt's residual as the embedding stage of its forward starts it.
        let emb_device = self.embeddings.embeddings().device().clone();
        let mut rows = Vec::with_capacity(prompts.len());
        let mut pres: Vec<Vec<Vec<f32>>> = Vec::with_capacity(prompts.len());
        for (seq, prompt) in seqs.iter_mut().zip(prompts) {
            if hashed {
                seq.tokens.open(0, prompt)?;
            }
            let ids = Tensor::from_vec(prompt.clone(), (1, prompt.len()), &emb_device)?;
            let hidden = self.add_positions(self.embeddings.forward(&ids)?, 0)?;
            let hidden = match self.config.embed_scale {
                Some(scale) => (hidden * scale)?,
                None => hidden,
            };
            rows.push(super::wide::widen_embedding(&hidden, streams)?);
            seq.cell.open(0, prompt.len());
            pres.push((0..prompt.len()).map(|_| onto_first(streams)).collect());
        }

        let n_layers = self.layers.len();
        for il in 0..n_layers {
            let layer = &mut self.layers[il];
            let mut mids = Vec::with_capacity(prompts.len());
            let mut attn_pre = Vec::new();
            for (j, seq) in seqs.iter_mut().enumerate() {
                let r = rows[j]
                    .to_device(&Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .contiguous()?;
                let block = layer.sinkhorn.as_ref().ok_or_else(|| {
                    crate::tensor::Error::msg("a Sinkhorn layer without its block")
                })?;
                let r = match block.engram.as_ref() {
                    Some(e) => e.add(&r, &seq.tokens, 0)?,
                    None => r,
                };
                exchange_state(&mut layer.mixer, &mut seq.states[il])?;
                cell.exchange(&seq.cell);
                let half = attention_half(block, &mut layer.mixer, &r, &pres[j], 0);
                cell.exchange(&seq.cell);
                exchange_state(&mut layer.mixer, &mut seq.states[il])?;
                let (mid, pre, _) = half?;
                mids.push(mid);
                attn_pre.extend(pre);
            }
            let block = layer
                .sinkhorn
                .as_ref()
                .ok_or_else(|| crate::tensor::Error::msg("a Sinkhorn layer without its block"))?;
            let mid = Tensor::cat(&mids.iter().collect::<Vec<_>>(), 1)?;
            let mixes = block.ffn_mix.mixes(&mid)?;
            let input = rms_norm(
                &block.ffn_mix.pre(&mid, &attn_pre)?,
                &block.ffn_norm,
                block.eps,
            )?;
            let out = staged("experts", &Device::Cpu, false, || block.ffn.forward(&input))?;
            let next = block.ffn_mix.post(&out, &mid, &mixes.post, &mixes.comb)?;
            let mut at = 0;
            for (j, prompt) in prompts.iter().enumerate() {
                let s = prompt.len();
                rows[j] = next.narrow(1, at, s)?.contiguous()?;
                pres[j] = mixes.pre[at..at + s].to_vec();
                at += s;
            }
            progress(il + 1, n_layers);
        }
        Ok(())
    }

    /// Sets the watcher of every routed feed-forward: `observe` is told the layer's index
    /// first, then what [`ExpertObserver`] is told.
    ///
    /// `None` removes every watcher.
    pub fn observe_experts(
        &self,
        observe: Option<Arc<dyn Fn(usize, Option<usize>, &[f32], &[f32]) + Send + Sync>>,
    ) {
        for (il, layer) in self.layers.iter().enumerate() {
            let Some(block) = layer.sinkhorn.as_ref() else {
                continue;
            };
            let at = self.first_layer + il;
            block.ffn.watch.set(observe.clone().map(|f| {
                Arc::new(move |e: Option<usize>, x: &[f32], h: &[f32]| f(at, e, x, h))
                    as Arc<ExpertObserver>
            }));
        }
    }
}

/// What one sequence of a batch keeps between its steps on a Sinkhorn wide-residual model.
///
/// The model's layers hold 1 sequence's state at a time.
/// A batch brings each sequence's state into its layer for that sequence's rows.
pub struct SequenceMixers {
    /// Each layer's mixer state, in layer order.
    states: Vec<mixer::MixerState>,
    /// What the sequence's band sources leave for its band readers within a step.
    cell: BandCell,
    /// The sequence's token ids, which its engram layers hash.
    tokens: TokenLog,
}

/// Exchanges the state `mixer` holds with `held`.
///
/// The exchange counts as a replaced state.
/// A card buffer stays while the state it copies comes back under the same name.
///
/// Refused, with nothing exchanged, when `held` is not a state of `mixer`'s kind.
pub(super) fn exchange_state(mixer: &mut mixer::Mixer, held: &mut mixer::MixerState) -> Result<()> {
    match (mixer, held) {
        (mixer::Mixer::LatentMqa(m), mixer::MixerState::LatentMqa(s)) => {
            std::mem::swap(&mut m.state, s);
            m.moves += 1;
        }
        (mixer::Mixer::CompressedBand(m), mixer::MixerState::CompressedBand(s)) => {
            std::mem::swap(&mut m.state, s);
            m.moves += 1;
        }
        _ => crate::tensor::bail!("a sequence's state is not of its layer's mixer kind"),
    }
    Ok(())
}
