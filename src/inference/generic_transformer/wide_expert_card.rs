//! The routed experts of a wide-residual layer that a card keeps.
//!
//! A card keeps an expert's 3 weights in the blocks the file stores.
//! It runs them with the expert kernels that give the host's bits (`gpu_experts_exact`).
//! Only the formats those kernels read are kept: gate and up in `IQ2_XXS` with down in `Q2_K`,
//! or gate and up in `IQ3_XXS` with down in `Q3_K`.
//!
//! A layer's experts on 1 card run in 1 call.
//! The layer's rows go up once, every (expert, row) pair runs, and the outputs come back once.
//! A layer whose experts sit on several cards makes 1 call per card, each card on its own thread.
//! A card that keeps routed experts only takes the rows and gives back its outputs, nothing more.
//!
//! The card that runs the layer's token mixer may also keep 2 weights every token reads:
//! - the shared expert, in the blocks the file stores, its products measured per unit as the
//!   mixers' are (`card_packed.rs`),
//! - the router's F32 values, which a decoded row takes with the host's bits.

use super::block_scaled_experts::BlockScaledExpert;
use crate::tensor::Result;

/// The routed experts of 1 layer that the cards keep, by expert id, and the layer's shared
/// expert and router when a card keeps them.
///
/// Empty on the host and in a build without cards.
#[derive(Default)]
pub struct CardExperts {
    #[cfg(feature = "gpu")]
    sets: Vec<gpu::CardSet>,
    #[cfg(feature = "gpu")]
    shared: Option<gpu::SharedCard>,
    #[cfg(feature = "gpu")]
    router: Option<gpu::RouterCard>,
    /// The ids of the experts the cards keep, in the order they were kept: the hottest first.
    #[cfg(feature = "gpu")]
    order: Vec<usize>,
    /// (expert, row) pairs the cards ran since the load.
    ran: std::sync::atomic::AtomicU64,
    /// Per card whose run failed since the last take, the most bytes a failed run asked for.
    #[cfg(feature = "gpu")]
    short: std::sync::Mutex<Vec<(usize, u64)>>,
}

impl CardExperts {
    /// Whether no card keeps an expert of the layer.
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Experts the cards keep.
    pub fn count(&self) -> usize {
        #[cfg(feature = "gpu")]
        {
            self.sets.iter().map(|s| s.experts.len()).sum()
        }
        #[cfg(not(feature = "gpu"))]
        0
    }

    /// Whether a card keeps expert `e`.
    pub fn holds(&self, e: usize) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.sets.iter().any(|s| s.experts.contains_key(&e))
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = e;
            false
        }
    }

    /// The ordinal of the card that keeps expert `e`, `None` when no card keeps it.
    pub fn card_of(&self, e: usize) -> Option<usize> {
        #[cfg(feature = "gpu")]
        {
            self.sets
                .iter()
                .find(|s| s.experts.contains_key(&e))
                .map(|s| s.ordinal)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = e;
            None
        }
    }

    /// Per card whose run of routed experts failed since the last call, the most bytes a failed
    /// run asked for.
    ///
    /// The notes are emptied.
    pub fn take_short(&self) -> Vec<(usize, u64)> {
        #[cfg(feature = "gpu")]
        {
            std::mem::take(&mut *self.short.lock().unwrap_or_else(|e| e.into_inner()))
        }
        #[cfg(not(feature = "gpu"))]
        Vec::new()
    }

    /// Experts card `ordinal` keeps.
    pub fn count_on(&self, ordinal: usize) -> usize {
        #[cfg(feature = "gpu")]
        {
            self.sets
                .iter()
                .filter(|s| s.ordinal == ordinal)
                .map(|s| s.experts.len())
                .sum()
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = ordinal;
            0
        }
    }

    /// Lets card `ordinal` drop the last expert it kept, the coldest: the bytes it freed, `None`
    /// when the card keeps none.
    ///
    /// The host runs the expert from then on, with the same bits.
    pub fn release_coldest(&mut self, ordinal: usize) -> Option<u64> {
        #[cfg(feature = "gpu")]
        {
            let set = self.sets.iter_mut().find(|s| s.ordinal == ordinal)?;
            let at = self
                .order
                .iter()
                .rposition(|e| set.experts.contains_key(e))?;
            let e = self.order.remove(at);
            let freed = set.release(e);
            self.sets.retain(|s| !s.experts.is_empty());
            freed
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = ordinal;
            None
        }
    }

    /// The expert card `ordinal` would drop first, the last it kept: `None` when it keeps none.
    pub fn coldest_on(&self, ordinal: usize) -> Option<usize> {
        #[cfg(feature = "gpu")]
        {
            let set = self.sets.iter().find(|s| s.ordinal == ordinal)?;
            self.order
                .iter()
                .rev()
                .find(|e| set.experts.contains_key(e))
                .copied()
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = ordinal;
            None
        }
    }

    /// Whether card `ordinal` keeps the layer's shared expert.
    pub fn keeps_shared_on(&self, ordinal: usize) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.shared.as_ref().is_some_and(|s| s.ordinal() == ordinal)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = ordinal;
            false
        }
    }

    /// Whether card `ordinal` keeps the layer's router.
    pub fn keeps_router_on(&self, ordinal: usize) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.router.as_ref().is_some_and(|r| r.ordinal() == ordinal)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = ordinal;
            false
        }
    }

    /// Drops the shared expert from the card: the bytes it held, `None` when no card keeps it.
    ///
    /// The host runs it from then on.
    pub fn release_shared(&mut self) -> Option<u64> {
        #[cfg(feature = "gpu")]
        {
            self.shared.take().map(|s| s.bytes() as u64)
        }
        #[cfg(not(feature = "gpu"))]
        None
    }

    /// Drops the router from the card: the bytes it held, `None` when no card keeps it.
    ///
    /// The host runs it from then on.
    pub fn release_router(&mut self) -> Option<u64> {
        #[cfg(feature = "gpu")]
        {
            self.router.take().map(|r| r.bytes() as u64)
        }
        #[cfg(not(feature = "gpu"))]
        None
    }

    /// (expert, row) pairs the cards ran since the load.
    pub fn pairs_run(&self) -> u64 {
        self.ran.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Per card ordinal, the bytes of experts kept there, with the shared expert and the router.
    pub fn bytes_by_card(&self) -> Vec<(usize, u64)> {
        #[cfg(feature = "gpu")]
        {
            let mut held: Vec<(usize, u64)> =
                self.sets.iter().map(|s| (s.ordinal, s.bytes)).collect();
            let always = self
                .shared
                .iter()
                .map(|s| (s.ordinal(), s.bytes() as u64))
                .chain(self.router.iter().map(|r| (r.ordinal(), r.bytes() as u64)));
            for (ordinal, bytes) in always {
                match held.iter_mut().find(|(o, _)| *o == ordinal) {
                    Some((_, b)) => *b += bytes,
                    None => held.push((ordinal, bytes)),
                }
            }
            held
        }
        #[cfg(not(feature = "gpu"))]
        Vec::new()
    }

    /// Whether a card keeps the layer's shared expert.
    pub fn keeps_shared(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.shared.is_some()
        }
        #[cfg(not(feature = "gpu"))]
        false
    }

    /// Whether a card keeps the layer's router.
    pub fn keeps_router(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.router.is_some()
        }
        #[cfg(not(feature = "gpu"))]
        false
    }

    /// The shared expert's outputs over the rows `x` `[n, dim]`, `[n * dim]` row-major, from the
    /// card that keeps it.
    ///
    /// `None` without one, or when the card call fails: the caller runs it on the host.
    pub fn run_shared(&self, x: &[f32], n: usize, limit: f32) -> Option<Vec<f32>> {
        #[cfg(feature = "gpu")]
        {
            let shared = self.shared.as_ref()?;
            match shared.run(x, n, limit) {
                Ok(y) => Some(y),
                Err(e) => {
                    tracing::warn!(
                        "card {}: shared expert not run, the host runs it: {e}",
                        shared.ordinal()
                    );
                    None
                }
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (x, n, limit);
            None
        }
    }

    /// The router's logits of the single row `x`, from the card that keeps it, with the host's
    /// bits.
    ///
    /// `None` without one, or when the card call fails: the caller runs it on the host.
    pub fn router_logits(&self, x: &[f32]) -> Option<Vec<f32>> {
        #[cfg(feature = "gpu")]
        {
            let router = self.router.as_ref()?;
            match router.logits(x) {
                Ok(y) => Some(y),
                Err(e) => {
                    tracing::warn!(
                        "card {}: router not run, the host runs it: {e}",
                        router.ordinal()
                    );
                    None
                }
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = x;
            None
        }
    }

    /// The router a card keeps, when it is card `ordinal`.
    #[cfg(feature = "gpu")]
    pub(super) fn router_on(&self, ordinal: usize) -> Option<&gpu::RouterCard> {
        self.router.as_ref().filter(|r| r.ordinal() == ordinal)
    }

    /// Keeps `shared`, the layer's shared expert, on card `ordinal`, its products in their units
    /// of `choices`.
    #[cfg(feature = "gpu")]
    pub(super) fn keep_shared(
        &mut self,
        ordinal: usize,
        shared: &BlockScaledExpert,
        choices: &super::card_packed::Choices,
    ) -> Result<()> {
        self.shared = Some(gpu::SharedCard::new(ordinal, shared, choices)?);
        Ok(())
    }

    /// Keeps `router`, the layer's router, on card `ordinal` as F32 values.
    #[cfg(feature = "gpu")]
    pub(super) fn keep_router(
        &mut self,
        ordinal: usize,
        router: &crate::tensor::blockscaled::projection::Projection,
    ) -> Result<()> {
        self.router = Some(gpu::RouterCard::new(ordinal, router)?);
        Ok(())
    }

    /// The outputs of the experts `ids`, kept on the cards, over their rows.
    ///
    /// `rows[i]` are the row indices of `x` `[n, dim]` that expert `ids[i]` runs over.
    /// Answers per expert its outputs `[rows[i].len(), dim]` row-major, in the order of `ids`.
    /// `None` when a card call fails: the caller runs them on the host.
    pub fn run(
        &self,
        ids: &[usize],
        rows: &[Vec<usize>],
        x: &[f32],
        dim: usize,
        limit: f32,
    ) -> Option<Vec<Vec<f32>>> {
        let (_, outs) = self.run_cards(ids, rows, x, dim, limit, false);
        outs.into_iter().collect()
    }

    /// The outputs of the experts `ids` kept on the cards over their rows, and, when `shared`
    /// and a card keeps it, the shared expert's outputs over every row of `x`.
    ///
    /// `rows[i]` are the row indices of `x` `[n, dim]` that expert `ids[i]` runs over.
    /// Each card with work runs on a thread of its own, the calling thread drives the first.
    /// The card that keeps the shared expert runs it before its routed experts.
    /// Answers the shared expert's outputs `[n * dim]` row-major, `None` when it did not run.
    /// Answers per expert its outputs `[rows[i].len(), dim]` row-major, in the order of `ids`.
    /// An expert no card keeps, or whose card call failed, answers `None`: the caller runs it on
    /// the host.
    pub fn run_cards(
        &self,
        ids: &[usize],
        rows: &[Vec<usize>],
        x: &[f32],
        dim: usize,
        limit: f32,
        shared: bool,
    ) -> (Option<Vec<f32>>, Vec<Option<Vec<f32>>>) {
        #[cfg(feature = "gpu")]
        {
            let n = x.len() / dim.max(1);
            let shared = self.shared.as_ref().filter(|_| shared);
            // Per card with work, its set and the indices of `ids` it keeps.
            let mut work: Vec<(usize, Option<&gpu::CardSet>, Vec<usize>)> = self
                .sets
                .iter()
                .map(|set| {
                    let mine: Vec<usize> = (0..ids.len())
                        .filter(|&i| set.experts.contains_key(&ids[i]))
                        .collect();
                    (set.ordinal, Some(set), mine)
                })
                .filter(|(_, _, mine)| !mine.is_empty())
                .collect();
            if let Some(s) = shared {
                if !work.iter().any(|w| w.0 == s.ordinal()) {
                    work.push((s.ordinal(), None, Vec::new()));
                }
            }
            // The shared expert's card stands first.
            work.sort_by_key(|w| Some(w.0) != shared.map(|s| s.ordinal()));
            let one = |(ordinal, set, mine): &(usize, Option<&gpu::CardSet>, Vec<usize>)| {
                let shared_out = shared.filter(|s| s.ordinal() == *ordinal).and_then(|s| {
                    match s.run(x, n, limit) {
                        Ok(y) => Some(y),
                        Err(e) => {
                            tracing::warn!(
                                "card {ordinal}: shared expert not run, the host runs it: {e}"
                            );
                            None
                        }
                    }
                });
                let outs = set.and_then(|set| {
                    let ids_here: Vec<usize> = mine.iter().map(|&i| ids[i]).collect();
                    let rows_here: Vec<Vec<usize>> =
                        mine.iter().map(|&i| rows[i].clone()).collect();
                    let pairs: usize = rows_here.iter().map(Vec::len).sum();
                    match set.run(&ids_here, &rows_here, x, dim, limit) {
                        Ok(ys) => {
                            self.ran
                                .fetch_add(pairs as u64, std::sync::atomic::Ordering::Relaxed);
                            Some(ys)
                        }
                        Err(e) => {
                            tracing::warn!(
                                "card {ordinal}: routed experts not run, the host runs them: {e}"
                            );
                            self.note_short(*ordinal, set.run_bytes(pairs, n));
                            None
                        }
                    }
                });
                (shared_out, outs)
            };
            let one = &one;
            let ran: Vec<(Option<Vec<f32>>, Option<Vec<Vec<f32>>>)> = std::thread::scope(|scope| {
                let others: Vec<_> = work
                    .iter()
                    .skip(1)
                    .map(|w| scope.spawn(move || one(w)))
                    .collect();
                let first = work.first().map(one);
                first
                    .into_iter()
                    .chain(others.into_iter().map(|h| h.join().unwrap_or((None, None))))
                    .collect()
            });
            let mut out: Vec<Option<Vec<f32>>> = vec![None; ids.len()];
            let mut shared_out = None;
            for ((_, _, mine), (sh, outs)) in work.iter().zip(ran) {
                shared_out = shared_out.or(sh);
                for (&i, y) in mine.iter().zip(outs.into_iter().flatten()) {
                    out[i] = Some(y);
                }
            }
            (shared_out, out)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (rows, x, dim, limit, shared);
            (None, vec![None; ids.len()])
        }
    }

    /// Notes that a run on card `ordinal` that asked for `bytes` failed.
    #[cfg(feature = "gpu")]
    fn note_short(&self, ordinal: usize, bytes: u64) {
        let mut short = self.short.lock().unwrap_or_else(|e| e.into_inner());
        match short.iter_mut().find(|(o, _)| *o == ordinal) {
            Some((_, b)) => *b = (*b).max(bytes),
            None => short.push((ordinal, bytes)),
        }
    }

    /// Keeps `experts` (id and weights) on card `ordinal`, beside what the layer keeps.
    ///
    /// Refused when their format has no kernel that gives the host's bits, or the upload fails.
    #[cfg(feature = "gpu")]
    pub fn keep(&mut self, ordinal: usize, experts: &[(usize, &BlockScaledExpert)]) -> Result<()> {
        let set = gpu::CardSet::new(ordinal, experts)?;
        self.order.extend(experts.iter().map(|&(e, _)| e));
        match self.sets.iter_mut().find(|s| s.ordinal == ordinal) {
            Some(held) => held.absorb(set),
            None => self.sets.push(set),
        }
        Ok(())
    }
}

/// Whether the cards have a kernel that gives the host's bits for `expert`'s format and shape.
pub fn served(expert: &BlockScaledExpert) -> bool {
    #[cfg(feature = "gpu")]
    {
        gpu::format_of(expert).is_some()
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = expert;
        false
    }
}

/// The host's outputs of `experts` over the rows `x` `[n, dim]`, each over all the rows.
///
/// The experts run at once, as a layer's forward runs them.
pub fn host_outputs(
    experts: &[&BlockScaledExpert],
    x: &crate::tensor::Tensor,
    limit: f32,
) -> Result<Vec<Vec<f32>>> {
    use rayon::prelude::*;
    experts
        .par_iter()
        .map(|e| e.forward(x, limit)?.flatten_all()?.to_vec1::<f32>())
        .collect()
}

#[cfg(feature = "gpu")]
pub(super) mod gpu {
    use super::super::card_packed::{Choices, PackedDev};
    use super::BlockScaledExpert;
    use crate::tensor::blockscaled::projection::Projection;
    use crate::tensor::cuda::{
        alloc_f32, expert_tables, gpu_experts_exact, q8k32_rows_to_card, swiglu_limit, upload_f32,
        CudaDevice, ExpertFormat, ExpertTables,
    };
    use crate::tensor::gpu_runtime::driver::{CudaSlice, CudaView, LaunchConfig, PushKernelArg};
    use crate::tensor::quant_cpu::QK_K;
    use crate::tensor::{Device, Error, Result};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    /// The kernels' format of `expert`, `None` when no kernel reads it or its shape.
    ///
    /// The kernels read the rows as q8_K blocks: `dim` and `inter` are whole blocks of it.
    pub(crate) fn format_of(expert: &BlockScaledExpert) -> Option<ExpertFormat> {
        let format = ExpertFormat::of(
            expert.gate.quant_dtype()?,
            expert.up.quant_dtype()?,
            expert.down.quant_dtype()?,
        )?;
        // The gate is `[inter, dim]`.
        let dims = expert.gate.dims();
        let (inter, dim) = (*dims.first()?, *dims.get(1)?);
        (dim % QK_K == 0 && inter % QK_K == 0).then_some(format)
    }

    /// The stored blocks of a projection in a GGUF format.
    fn blocks(p: &Projection) -> Result<std::borrow::Cow<'_, [u8]>> {
        match p {
            Projection::Quant(q) => q.data(),
            _ => Err(Error::msg("a card expert is stored in GGUF blocks")),
        }
    }

    /// Experts of 1 format kept on 1 card.
    pub(crate) struct CardSet {
        pub(crate) ordinal: usize,
        dev: Arc<CudaDevice>,
        tables: Arc<ExpertTables>,
        /// `(dim, inter)`.
        dims: (usize, usize),
        /// Per expert id, gate, up and down blocks.
        pub(crate) experts: BTreeMap<usize, [CudaSlice<u8>; 3]>,
        /// Bytes of the blocks kept.
        pub(crate) bytes: u64,
    }

    impl CardSet {
        /// `experts` uploaded to card `ordinal`.
        ///
        /// Refused when they are not all of 1 format and shape the kernels read.
        pub(crate) fn new(ordinal: usize, experts: &[(usize, &BlockScaledExpert)]) -> Result<Self> {
            let Some(&(_, first)) = experts.first() else {
                return Err(Error::msg("no expert to keep on a card"));
            };
            let format = format_of(first)
                .ok_or_else(|| Error::msg("no card kernel reads these experts' format"))?;
            let dims = first.gate.dims();
            let dims = (dims[1], dims[0]);
            let Device::Cuda(dev) = Device::new_cuda(ordinal)? else {
                return Err(Error::msg("card experts need a CUDA device"));
            };
            let tables = Arc::new(expert_tables(&dev, format)?);
            let mut set = Self {
                ordinal,
                dev,
                tables,
                dims,
                experts: BTreeMap::new(),
                bytes: 0,
            };
            for &(id, e) in experts {
                if format_of(e) != Some(format) || (e.gate.dims()[1], e.gate.dims()[0]) != dims {
                    return Err(Error::msg(format!(
                        "expert {id} differs in format or shape from the others on card {ordinal}"
                    )));
                }
                let stream = set.dev.stream();
                let mut up = |p: &Projection| -> Result<CudaSlice<u8>> {
                    let b = blocks(p)?;
                    set.bytes += b.len() as u64;
                    stream
                        .clone_htod(b.as_ref())
                        .map_err(|err| Error::msg(format!("expert {id} upload: {err}")))
                };
                let weights = [up(&e.gate)?, up(&e.up)?, up(&e.down)?];
                set.experts.insert(id, weights);
            }
            Ok(set)
        }

        /// Takes the experts of `other`, kept on the same card.
        pub(crate) fn absorb(&mut self, other: CardSet) {
            self.bytes += other.bytes;
            self.experts.extend(other.experts);
        }

        /// Drops expert `e` from the card: the bytes its blocks held, `None` when the card does
        /// not keep it.
        pub(crate) fn release(&mut self, e: usize) -> Option<u64> {
            let blocks = self.experts.remove(&e)?;
            let freed: u64 = blocks.iter().map(|b| b.len() as u64).sum();
            self.bytes = self.bytes.saturating_sub(freed);
            Some(freed)
        }

        /// Bytes a run of `pairs` (expert, row) pairs over `n` rows takes on the card: the rows,
        /// each pair's intermediate values and each pair's outputs, as F32.
        pub(crate) fn run_bytes(&self, pairs: usize, n: usize) -> u64 {
            let (dim, inter) = self.dims;
            let floats = n * dim + pairs * (inter + dim);
            (floats * std::mem::size_of::<f32>()) as u64
        }

        /// The outputs of `ids` over their `rows` of `x` `[n, dim]`, in the order of `ids`.
        pub(crate) fn run(
            &self,
            ids: &[usize],
            rows: &[Vec<usize>],
            x: &[f32],
            dim: usize,
            limit: f32,
        ) -> Result<Vec<Vec<f32>>> {
            if dim != self.dims.0 {
                return Err(Error::msg(format!(
                    "rows of {dim} for experts of {}",
                    self.dims.0
                )));
            }
            let weights = ids
                .iter()
                .map(|e| {
                    self.experts
                        .get(e)
                        .map(|[g, u, d]| (g, u, d))
                        .ok_or_else(|| Error::msg(format!("expert {e} is not on the card")))
                })
                .collect::<Result<Vec<_>>>()?;
            // Instances of 1 expert stand together, its rows in row order.
            let instances: Vec<(usize, usize)> = rows
                .iter()
                .enumerate()
                .flat_map(|(i, r)| r.iter().map(move |&t| (i, t)))
                .collect();
            if instances.is_empty() {
                return Ok(vec![Vec::new(); ids.len()]);
            }
            let xq = q8k32_rows_to_card(&self.dev, x)?;
            let (_h, y) = gpu_experts_exact(
                &self.dev,
                &weights,
                &self.tables,
                self.dims,
                &xq,
                &instances,
                limit,
            )?;
            let flat = self
                .dev
                .stream()
                .clone_dtoh(&y)
                .map_err(|e| Error::msg(format!("card expert outputs: {e}")))?;
            let mut at = 0;
            Ok(rows
                .iter()
                .map(|r| {
                    let n = r.len() * dim;
                    let part = flat[at..at + n].to_vec();
                    at += n;
                    part
                })
                .collect())
        }
    }

    /// A layer's shared expert on the card that runs its token mixer.
    pub(crate) struct SharedCard {
        ordinal: usize,
        dev: Arc<CudaDevice>,
        gate: PackedDev,
        up: PackedDev,
        down: PackedDev,
    }

    impl SharedCard {
        /// `expert` uploaded to card `ordinal` as stored, its products in their units of
        /// `choices`.
        pub(crate) fn new(
            ordinal: usize,
            expert: &BlockScaledExpert,
            choices: &Choices,
        ) -> Result<Self> {
            let Device::Cuda(dev) = Device::new_cuda(ordinal)? else {
                return Err(Error::msg("a card shared expert needs a CUDA device"));
            };
            Ok(Self {
                ordinal,
                gate: PackedDev::new(&dev, &expert.gate, choices)?,
                up: PackedDev::new(&dev, &expert.up, choices)?,
                down: PackedDev::new(&dev, &expert.down, choices)?,
                dev,
            })
        }

        pub(crate) fn ordinal(&self) -> usize {
            self.ordinal
        }

        /// Bytes the card holds for the 3 weights.
        pub(crate) fn bytes(&self) -> usize {
            self.gate.bytes() + self.up.bytes() + self.down.bytes()
        }

        /// The outputs over the rows `x` `[n, dim]`, `[n * dim]` row-major.
        ///
        /// The rows go up once, the gate and up products, the SwiGLU and the down product run on
        /// the card, and the outputs come back once.
        pub(crate) fn run(&self, x: &[f32], n: usize, limit: f32) -> Result<Vec<f32>> {
            let dev: &CudaDevice = &self.dev;
            let rows = upload_f32(dev, "shared expert rows", x)?;
            let inter = self.gate.out();
            let g = self.gate.apply(&rows, n)?;
            let u = self.up.apply(&rows, n)?;
            let h = swiglu_limit(dev, &g.slice(0..n * inter), &u.slice(0..n * inter), limit)?;
            let y = self.down.apply(&h, n)?;
            super::super::card_mixers::download(dev, &y, n * self.down.out())
        }
    }

    /// A layer's router on the card that runs its token mixer, as the F32 values the host
    /// holds.
    pub(crate) struct RouterCard {
        ordinal: usize,
        dev: Arc<CudaDevice>,
        /// `[out, inp]` row-major.
        values: CudaSlice<f32>,
        out: usize,
        inp: usize,
    }

    impl RouterCard {
        /// `router`'s values uploaded to card `ordinal`.
        pub(crate) fn new(ordinal: usize, router: &Projection) -> Result<Self> {
            let (out, inp) = match router.dims().as_slice() {
                [out, inp] => (*out, *inp),
                _ => return Err(Error::msg("a router is a matrix")),
            };
            let Device::Cuda(dev) = Device::new_cuda(ordinal)? else {
                return Err(Error::msg("a card router needs a CUDA device"));
            };
            dev.quantized_fn("seq_row_f32")?;
            let values = upload_f32(&dev, "router", &router.dense_f32()?)?;
            Ok(Self {
                ordinal,
                dev,
                values,
                out,
                inp,
            })
        }

        pub(crate) fn ordinal(&self) -> usize {
            self.ordinal
        }

        /// Bytes the card holds for the values.
        pub(crate) fn bytes(&self) -> usize {
            self.out * self.inp * std::mem::size_of::<f32>()
        }

        /// The logits of the single row `x`.
        ///
        /// Each is the fold from -0, in input order, of the rounded products of a weight row
        /// and the row: the steps the host takes for 1 row.
        pub(crate) fn logits(&self, x: &[f32]) -> Result<Vec<f32>> {
            let row = upload_f32(&self.dev, "router row", x)?;
            let y = self.logits_dev(&row.slice(0..x.len()))?;
            super::super::card_mixers::download(&self.dev, &y, self.out)
        }

        /// The logits of the single row `x` on the card, left there, by the steps of
        /// [`Self::logits`].
        pub(crate) fn logits_dev(&self, x: &CudaView<'_, f32>) -> Result<CudaSlice<f32>> {
            if x.len() != self.inp {
                return Err(Error::msg(format!(
                    "a row of {} values for a router of {}",
                    x.len(),
                    self.inp
                )));
            }
            let dev: &CudaDevice = &self.dev;
            let y = alloc_f32(dev, "router logits", self.out)?;
            let func = dev.quantized_fn("seq_row_f32")?;
            let inp = self.inp as i32;
            let mut b = dev.stream().launch_builder(&func);
            b.arg(&self.values);
            b.arg(x);
            b.arg(&y);
            b.arg(&inp);
            // A warp per output.
            let cfg = LaunchConfig {
                grid_dim: (self.out as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { b.launch(cfg) }.map_err(|e| Error::msg(format!("seq_row_f32 launch: {e}")))?;
            Ok(y)
        }

        /// The logits' count.
        pub(crate) fn out(&self) -> usize {
            self.out
        }

        /// The card's seconds for the logits of the row `x` already on the card, as `timed`
        /// times them: the kernel and a wait for it, no crossing.
        pub(crate) fn time_on_card(
            &self,
            x: &[f32],
            timed: &mut dyn FnMut(
                &mut dyn FnMut() -> Result<()>,
            ) -> std::result::Result<f64, String>,
        ) -> std::result::Result<f64, String> {
            let text = |e: Error| e.to_string();
            let row = upload_f32(&self.dev, "router row", x).map_err(text)?;
            let view = row.slice(0..x.len());
            timed(&mut || {
                self.logits_dev(&view)?;
                self.dev
                    .stream()
                    .synchronize()
                    .map_err(|e| Error::msg(format!("router wait: {e}")))
            })
        }
    }
}
