//! The weights every token of a Sinkhorn wide-residual model reads, kept on the cards when the
//! load measures that a byte of them saves more host time than a byte of routed experts.
//!
//! The candidates of a card, for each held layer whose token mixer runs there:
//! - the output projection, which the mixer's card runs right after the mixer,
//! - the shared expert, which the card runs while the host runs its routed experts,
//! - the router, which the card runs for a decoded row with the host's bits,
//! - the engram's key and value projection, which the card runs on the gathered table rows,
//! - the 2 Sinkhorn mixers and the 2 norms, which keep a decoded row's residual on the card.
//!
//! The head is a candidate of the card of the last layer of the stack, when the model holds it.
//!
//! The load times 1 weight of each kind over 1 row on the host, and on the card with the row's
//! crossing.
//! A weight saves the difference at every token.
//! A routed expert saves the difference of its own times at the tokens routed to it, its share:
//! - the routing profile's count of the expert over the tokens it counted, when a profile primed
//!   the heat,
//! - else the largest share its heat rank allows: the k-th hottest of a layer is routed by at
//!   most `used / k` of the tokens.
//!
//! The candidates and the experts are walked from the largest saving per byte down until the
//! card's room is spent (`split_room`).
//! The experts get the room the candidates leave.

use super::GenericHeteroTransformer;
use crate::inference::place::expert_variants::VariantChoice;

/// A kind of weight every token reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AlwaysRead {
    /// A layer's output projection, after its token mixer.
    Output,
    /// A layer's shared expert.
    Shared,
    /// A layer's router.
    Router,
    /// An engram layer's key and value projection.
    Engram,
    /// The model's head.
    Head,
    /// A layer's 2 Sinkhorn mixers and its 2 norms, which keep a decoded row's residual on the
    /// card.
    Sinkhorn,
}

impl AlwaysRead {
    /// Every kind, in the order the log lists them.
    pub const ALL: [AlwaysRead; 6] = [
        AlwaysRead::Output,
        AlwaysRead::Shared,
        AlwaysRead::Router,
        AlwaysRead::Engram,
        AlwaysRead::Head,
        AlwaysRead::Sinkhorn,
    ];

    /// The name a log line gives the kind.
    pub fn name(self) -> &'static str {
        match self {
            AlwaysRead::Output => "output projection",
            AlwaysRead::Shared => "shared expert",
            AlwaysRead::Router => "router",
            AlwaysRead::Engram => "engram projection",
            AlwaysRead::Head => "head",
            AlwaysRead::Sinkhorn => "Sinkhorn mixers",
        }
    }
}

/// What the load measured of 1 weight of a kind on 1 card.
#[derive(Clone, Debug, PartialEq)]
pub struct KindMeasure {
    pub kind: AlwaysRead,
    /// Bytes the weight holds on the card.
    pub bytes: u64,
    /// Seconds the host took for the weight over 1 row.
    pub host_secs: f64,
    /// Seconds the card took for the weight over 1 row: the row up, the kernels, the row down.
    pub card_secs: f64,
}

impl KindMeasure {
    /// Host seconds a byte of the weight on the card saves at each token.
    ///
    /// 0 or less when the card is not faster than the host.
    pub fn saving_per_byte(&self) -> f64 {
        (self.host_secs - self.card_secs) / self.bytes.max(1) as f64
    }
}

/// What the load measured on 1 card, kept for the cache's take-back.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CardSaving {
    pub ordinal: usize,
    /// Per kind measured, the host seconds a byte of the weight on the card saves at each token.
    pub kinds: Vec<(AlwaysRead, f64)>,
    /// Host seconds a routed expert on the card saves at a token routed to it.
    pub per_expert: f64,
}

impl CardSaving {
    /// Host seconds a byte of a weight of `kind` saves at each token, 0 when unmeasured.
    pub fn of(&self, kind: AlwaysRead) -> f64 {
        self.kinds
            .iter()
            .find(|(k, _)| *k == kind)
            .map_or(0.0, |&(_, s)| s)
    }
}

/// Something that may take room on a card: its bytes and the host seconds a byte of it saves at
/// each token.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Offer {
    pub bytes: u64,
    pub saving: f64,
}

/// Which of `weights` a room of `room` bytes takes beside `experts`, and the bytes of `experts`
/// the room takes.
///
/// Both lists are walked together from the largest saving per byte down.
/// A weight goes before an expert that saves as much per byte.
/// An offer that does not fit what is left of the room is passed over.
/// An offer that saves nothing is never taken.
pub fn split_room(room: u64, weights: &[Offer], experts: &[Offer]) -> (Vec<bool>, u64) {
    // Per offer: its saving, its bytes, and the index of the weight it is, `None` for an expert.
    let mut all: Vec<(f64, u64, Option<usize>)> = weights
        .iter()
        .enumerate()
        .map(|(i, o)| (o.saving, o.bytes, Some(i)))
        .chain(experts.iter().map(|o| (o.saving, o.bytes, None)))
        .collect();
    all.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| b.2.is_some().cmp(&a.2.is_some()))
            .then_with(|| a.2.cmp(&b.2))
    });
    let mut taken = vec![false; weights.len()];
    let mut left = room;
    let mut expert_bytes = 0u64;
    for (saving, bytes, weight) in all {
        if saving <= 0.0 || bytes > left {
            continue;
        }
        left -= bytes;
        match weight {
            Some(i) => taken[i] = true,
            None => expert_bytes += bytes,
        }
    }
    (taken, expert_bytes)
}

/// The share of the tokens that route to the expert of heat rank `rank` (0 the hottest) of a
/// layer whose tokens each route to `used` experts, `count_share` when a profile counted it.
pub fn expert_share(count_share: Option<f64>, rank: usize, used: usize) -> f64 {
    match count_share {
        Some(s) => s,
        None => (used as f64 / (rank + 1) as f64).min(1.0),
    }
}

/// Per expert of a layer, the share of the tokens a profile's `counts` routed to it, each token
/// routing to `used` experts.
///
/// `None` for a layer the profile never counted.
pub fn count_shares(counts: &[u32], used: usize) -> Option<Vec<f64>> {
    let total: u64 = counts.iter().map(|&c| c as u64).sum();
    if total == 0 {
        return None;
    }
    let tokens = total as f64 / used.max(1) as f64;
    Some(counts.iter().map(|&c| c as f64 / tokens).collect())
}

/// A projection on a card, when the load keeps it there: the head of a Sinkhorn wide-residual
/// model, or an engram layer's key and value projection.
#[derive(Default)]
pub struct CardProjection {
    #[cfg(feature = "gpu")]
    held: Option<(
        std::sync::Arc<crate::tensor::cuda::CudaDevice>,
        super::card_packed::PackedDev,
    )>,
}

impl CardProjection {
    /// The ordinal of the card that keeps the projection, `None` when the host keeps it.
    pub fn ordinal(&self) -> Option<usize> {
        #[cfg(feature = "gpu")]
        {
            self.held.as_ref().map(|(dev, _)| dev.ordinal())
        }
        #[cfg(not(feature = "gpu"))]
        None
    }

    /// Bytes the card holds for the projection.
    pub fn bytes(&self) -> usize {
        #[cfg(feature = "gpu")]
        {
            self.held.as_ref().map_or(0, |(_, p)| p.bytes())
        }
        #[cfg(not(feature = "gpu"))]
        0
    }

    /// The row `x` on the card of the projection through it, left on that card, with its width.
    ///
    /// `None` when no card keeps the projection.
    #[cfg(feature = "gpu")]
    pub(super) fn apply_dev(
        &self,
        x: &crate::tensor::gpu_runtime::driver::CudaSlice<f32>,
    ) -> crate::tensor::Result<Option<(crate::tensor::gpu_runtime::driver::CudaSlice<f32>, usize)>>
    {
        let Some((_, packed)) = self.held.as_ref() else {
            return Ok(None);
        };
        Ok(Some((packed.apply(x, 1)?, packed.out())))
    }

    /// The rows `h` `[rows, inp]` on the host through the projection, `[rows, out]`, from the
    /// card.
    ///
    /// `None` when no card keeps the projection, or when the card call fails: the caller runs it
    /// on the host.
    pub fn apply(&self, h: &crate::tensor::Tensor) -> Option<crate::tensor::Tensor> {
        #[cfg(feature = "gpu")]
        {
            let (dev, head) = self.held.as_ref()?;
            let run = || -> crate::tensor::Result<crate::tensor::Tensor> {
                let rows = h.dim(0)?;
                let up = crate::tensor::cuda::upload_f32(
                    dev,
                    "projection rows",
                    &h.flatten_all()?.to_vec1::<f32>()?,
                )?;
                let y = head.apply(&up, rows)?;
                let out = head.out();
                let host = super::card_mixers::download(dev, &y, rows * out)?;
                crate::tensor::Tensor::from_vec(host, (rows, out), &crate::tensor::Device::Cpu)
            };
            match run() {
                Ok(t) => Some(t),
                Err(e) => {
                    tracing::warn!(
                        "card {}: projection not run, the host runs it: {e}",
                        dev.ordinal()
                    );
                    None
                }
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = h;
            None
        }
    }

    /// Drops the projection from its card: the bytes it held, `None` when the host keeps it.
    ///
    /// The host runs it from then on.
    pub fn release(&mut self) -> Option<u64> {
        let bytes = self.bytes() as u64;
        self.ordinal()?;
        #[cfg(feature = "gpu")]
        {
            self.held = None;
        }
        Some(bytes)
    }

    /// Keeps `projection` on card `ordinal`, its product in its unit of `choices`.
    #[cfg(feature = "gpu")]
    pub(super) fn keep(
        &mut self,
        ordinal: usize,
        projection: &crate::tensor::blockscaled::projection::Projection,
        choices: &super::card_packed::Choices,
    ) -> crate::tensor::Result<()> {
        let crate::tensor::Device::Cuda(dev) = crate::tensor::Device::new_cuda(ordinal)? else {
            return Err(crate::tensor::Error::msg(
                "a card projection needs a CUDA device",
            ));
        };
        let packed = super::card_packed::PackedDev::new(&dev, projection, choices)?;
        self.held = Some((dev, packed));
        Ok(())
    }
}

impl GenericHeteroTransformer {
    /// Per card ordinal, the held layers whose token mixer runs there, and the held index of
    /// the layer whose card would keep the head.
    ///
    /// The head has a card only when the model holds the last layer of the stack.
    fn always_read_homes(&self) -> (Vec<(usize, Vec<usize>)>, Option<usize>) {
        let mut cards: Vec<(usize, Vec<usize>)> = Vec::new();
        for (il, layer) in self.layers.iter().enumerate() {
            if layer.sinkhorn.is_none() {
                continue;
            }
            let Some(ordinal) = layer.mixer.card_ordinal() else {
                continue;
            };
            match cards.iter_mut().find(|(o, _)| *o == ordinal) {
                Some((_, layers)) => layers.push(il),
                None => cards.push((ordinal, vec![il])),
            }
        }
        let stack = self.wide_sparse.as_ref().map_or(0, |ws| ws.meta.n_layers);
        let last = self.layers.len().checked_sub(1);
        let head = last.filter(|&il| {
            self.first_layer + il + 1 == stack
                && self.layers[il].sinkhorn.is_some()
                && self.layers[il].mixer.card_ordinal().is_some()
        });
        (cards, head)
    }

    /// Per held layer, its routed experts from the hottest down, each with the share of the
    /// tokens routed to it and its bytes.
    fn expert_shares(&self) -> Vec<(f64, u64)> {
        let Some(ws) = self.wide_sparse.as_ref() else {
            return Vec::new();
        };
        let used = self.routed_per_token();
        let ranks = self.held_heat();
        let mut held = 0usize;
        let mut out = Vec::new();
        for (il, layer) in self.layers.iter().enumerate() {
            let Some(block) = layer.sinkhorn.as_ref() else {
                continue;
            };
            let ffn = &block.ffn;
            let counted = ws
                .shares
                .as_ref()
                .and_then(|s| s.get(self.first_layer + il))
                .and_then(Option::as_ref);
            let ids =
                crate::inference::place::expert_variants::order(&ranks, held, ffn.experts.len());
            for (rank, e) in ids.into_iter().enumerate() {
                let share = expert_share(counted.and_then(|c| c.get(e).copied()), rank, used);
                let bytes = ffn.variant_bytes(e, 0).unwrap_or(0);
                out.push((share, bytes));
            }
            held += 1;
        }
        out
    }

    /// The weights every token reads that card `ordinal` keeps, by kind and held layer, in the
    /// order the load offers them.
    pub(super) fn always_read_on(&self, ordinal: usize) -> Vec<(AlwaysRead, usize)> {
        let mut out = Vec::new();
        for (il, layer) in self.layers.iter().enumerate() {
            let Some(block) = layer.sinkhorn.as_ref() else {
                continue;
            };
            if layer.mixer.card_ordinal() == Some(ordinal) && layer.mixer.projects_output() {
                out.push((AlwaysRead::Output, il));
            }
            if block.ffn.card.keeps_shared_on(ordinal) {
                out.push((AlwaysRead::Shared, il));
            }
            if block.ffn.card.keeps_router_on(ordinal) {
                out.push((AlwaysRead::Router, il));
            }
            let wkv = block.engram.as_ref().and_then(|e| e.block.wkv_on_card());
            if wkv.is_some_and(|(o, _)| o == ordinal) {
                out.push((AlwaysRead::Engram, il));
            }
            if block.card_mix.ordinal() == Some(ordinal) {
                out.push((AlwaysRead::Sinkhorn, il));
            }
        }
        let head = self
            .wide_sparse
            .as_ref()
            .is_some_and(|ws| ws.card_head.ordinal() == Some(ordinal));
        if let (true, Some(il)) = (head, self.layers.len().checked_sub(1)) {
            out.push((AlwaysRead::Head, il));
        }
        out
    }

    /// Drops the weight of `kind` of held layer `il` from its card: the bytes it held.
    ///
    /// `None` when no card keeps it, or when a forward in flight shares the mixer's card.
    /// The host runs it from then on, as it runs a weight the load leaves on the host.
    pub(super) fn release_always_read(&mut self, kind: AlwaysRead, il: usize) -> Option<u64> {
        match kind {
            AlwaysRead::Head => self.wide_sparse.as_mut()?.card_head.release(),
            AlwaysRead::Output => self.layers.get_mut(il)?.mixer.release_output_on_card(),
            AlwaysRead::Shared | AlwaysRead::Router | AlwaysRead::Engram | AlwaysRead::Sinkhorn => {
                let block = self.layers.get_mut(il)?.sinkhorn.as_mut()?;
                match kind {
                    AlwaysRead::Shared => block.ffn.card.release_shared(),
                    AlwaysRead::Router => block.ffn.card.release_router(),
                    AlwaysRead::Engram => block.engram.as_mut()?.block.release_wkv_on_card(),
                    _ => block.card_mix.release(),
                }
            }
        }
    }

    /// Keeps on each card that has a room the weights every token reads that save more host
    /// time per byte there than the routed experts, and leaves the experts the rest of the room.
    ///
    /// The load log says what was measured and kept.
    /// A model of another kind, or on the host, is left as it is.
    ///
    /// The experts' saving is measured in the variant the card tier keeps under `choice`.
    pub fn place_always_read(&mut self, choice: VariantChoice) {
        #[cfg(feature = "gpu")]
        self.place_always_read_on_cards(choice);
        #[cfg(not(feature = "gpu"))]
        let _ = choice;
    }

    #[cfg(feature = "gpu")]
    fn place_always_read_on_cards(&mut self, choice: VariantChoice) {
        let rooms = self.wide_card_rooms().to_vec();
        if rooms.is_empty() {
            return;
        }
        let (homes, head_layer) = self.always_read_homes();
        // The experts from the largest share per byte down, the same order on every card.
        let mut experts: Vec<(f64, u64)> = self
            .expert_shares()
            .into_iter()
            .filter(|&(_, b)| b > 0)
            .collect();
        experts.sort_by(|a, b| (b.0 / b.1 as f64).total_cmp(&(a.0 / a.1 as f64)));
        let mut cursor = 0usize;
        let choices = super::card_packed::Choices::default();
        let mut left_rooms = Vec::with_capacity(rooms.len());
        let mut savings: Vec<CardSaving> = Vec::new();
        for (ordinal, room) in rooms {
            let layers: Vec<usize> = homes
                .iter()
                .find(|(o, _)| *o == ordinal)
                .map(|(_, l)| l.clone())
                .unwrap_or_default();
            let head_here =
                head_layer.filter(|&il| self.layers[il].mixer.card_ordinal() == Some(ordinal));
            // The candidates of the card, by kind and held layer.
            let mut candidates: Vec<(AlwaysRead, usize)> = Vec::new();
            for &il in &layers {
                let Some(block) = self.layers[il].sinkhorn.as_ref() else {
                    continue;
                };
                candidates.push((AlwaysRead::Output, il));
                if block.ffn.shared.is_some() {
                    candidates.push((AlwaysRead::Shared, il));
                }
                candidates.push((AlwaysRead::Router, il));
                if block.engram.is_some() {
                    candidates.push((AlwaysRead::Engram, il));
                }
                candidates.push((AlwaysRead::Sinkhorn, il));
            }
            if let Some(il) = head_here {
                candidates.push((AlwaysRead::Head, il));
            }
            if candidates.is_empty() {
                left_rooms.push((ordinal, room));
                continue;
            }
            // 1 weight of each kind measured on the first layer that has it.
            let mut measures: Vec<KindMeasure> = Vec::new();
            for kind in AlwaysRead::ALL {
                let Some(&(_, il)) = candidates.iter().find(|(k, _)| *k == kind) else {
                    continue;
                };
                match self.measure_always_read(ordinal, kind, il) {
                    Ok(m) => {
                        let count = candidates.iter().filter(|(k, _)| *k == kind).count();
                        tracing::info!(
                            "card {ordinal}: always-read {}: {count} of {:.1} MB, 1 row in {:.3} ms on the host, {:.3} ms on the card: saves {:.2} us a token per MB",
                            kind.name(),
                            m.bytes as f64 / 1e6,
                            m.host_secs * 1e3,
                            m.card_secs * 1e3,
                            m.saving_per_byte() * 1e12
                        );
                        measures.push(m);
                    }
                    Err(why) => {
                        tracing::info!(
                            "card {ordinal}: always-read {} stays on the host: {why}",
                            kind.name()
                        )
                    }
                }
            }
            // The experts' saving on this card, by the measure the card tier takes.
            let per_expert = match self.measure_wide_card(ordinal, choice) {
                Ok(m) if m.experts > 0 => (m.host_secs - m.card_secs) / m.experts as f64,
                _ => 0.0,
            };
            savings.push(CardSaving {
                ordinal,
                kinds: measures
                    .iter()
                    .map(|m| (m.kind, m.saving_per_byte()))
                    .collect(),
                per_expert,
            });
            let measured: Vec<(AlwaysRead, usize, &KindMeasure)> = candidates
                .iter()
                .filter_map(|&(k, il)| measures.iter().find(|m| m.kind == k).map(|m| (k, il, m)))
                .collect();
            let offers: Vec<Offer> = measured
                .iter()
                .map(|(_, _, m)| Offer {
                    bytes: m.bytes,
                    saving: m.saving_per_byte(),
                })
                .collect();
            let expert_offers: Vec<Offer> = experts[cursor.min(experts.len())..]
                .iter()
                .map(|&(share, bytes)| Offer {
                    bytes,
                    saving: share * per_expert / bytes as f64,
                })
                .collect();
            let best_expert = expert_offers.first().map_or(0.0, |o| o.saving);
            let (taken, expert_bytes) = split_room(room, &offers, &expert_offers);
            // The cursor passes the experts this card's room takes.
            let mut passed = 0u64;
            while cursor < experts.len() && passed < expert_bytes {
                passed += experts[cursor].1;
                cursor += 1;
            }
            let mut kept_bytes = 0u64;
            let mut kept: Vec<(AlwaysRead, usize)> = Vec::new();
            for ((kind, il, _), take) in measured.iter().zip(&taken) {
                if !*take {
                    continue;
                }
                match self.keep_always_read(ordinal, *kind, *il, &choices) {
                    Ok(bytes) => {
                        kept_bytes += bytes;
                        kept.push((*kind, *il));
                    }
                    Err(e) => {
                        tracing::warn!(
                            "card {ordinal}: always-read {} of layer {} not kept, the rest stays on the host: {e}",
                            kind.name(),
                            self.first_layer + il
                        );
                        break;
                    }
                }
            }
            let listed = AlwaysRead::ALL
                .iter()
                .map(|&k| (k, kept.iter().filter(|(x, _)| *x == k).count()))
                .filter(|&(_, n)| n > 0)
                .map(|(k, n)| format!("{} {n}", k.name()))
                .collect::<Vec<_>>()
                .join(", ");
            let left = room.saturating_sub(kept_bytes);
            tracing::info!(
                "card {ordinal}: always-read weights keep {} MB of its {} MB ({}); the routed experts save at most {:.2} us a token per MB: {} MB left to them",
                kept_bytes / 1_000_000,
                room / 1_000_000,
                if listed.is_empty() { "none" } else { listed.as_str() },
                best_expert * 1e12,
                left / 1_000_000
            );
            left_rooms.push((ordinal, left));
        }
        if let Some(ws) = self.wide_sparse.as_mut() {
            ws.savings = savings;
        }
        self.set_wide_card_rooms(left_rooms);
    }

    /// Keeps the weight of `kind` of held layer `il` on card `ordinal`: its bytes there.
    #[cfg(feature = "gpu")]
    pub(super) fn keep_always_read(
        &mut self,
        ordinal: usize,
        kind: AlwaysRead,
        il: usize,
        choices: &super::card_packed::Choices,
    ) -> crate::tensor::Result<u64> {
        let no_block = || crate::tensor::Error::msg("a wide layer without its block");
        match kind {
            AlwaysRead::Output => {
                let layer = &mut self.layers[il];
                let block = layer.sinkhorn.as_ref().ok_or_else(no_block)?;
                if !layer.mixer.keep_output_on_card(&block.output, choices)? {
                    crate::tensor::bail!("the mixer's card is in use");
                }
                Ok(block.output.bytes() as u64)
            }
            AlwaysRead::Shared => {
                let ffn = &mut self.layers[il].sinkhorn.as_mut().ok_or_else(no_block)?.ffn;
                let Some(shared) = ffn.shared.as_ref() else {
                    return Ok(0);
                };
                ffn.card.keep_shared(ordinal, shared, choices)?;
                Ok(shared_bytes(shared))
            }
            AlwaysRead::Router => {
                let ffn = &mut self.layers[il].sinkhorn.as_mut().ok_or_else(no_block)?.ffn;
                ffn.card.keep_router(ordinal, &ffn.router)?;
                let values: usize = ffn.router.dims().iter().product();
                Ok((values * std::mem::size_of::<f32>()) as u64)
            }
            AlwaysRead::Engram => {
                let block = self.layers[il].sinkhorn.as_mut().ok_or_else(no_block)?;
                let Some(engram) = block.engram.as_mut() else {
                    return Ok(0);
                };
                engram.block.keep_wkv_on_card(ordinal, choices)?;
                Ok(engram.block.wkv().bytes() as u64)
            }
            AlwaysRead::Head => {
                let ws = self
                    .wide_sparse
                    .as_mut()
                    .ok_or_else(|| crate::tensor::Error::msg("no wide residual"))?;
                ws.card_head.keep(ordinal, &ws.head, choices)?;
                Ok(ws.card_head.bytes() as u64)
            }
            AlwaysRead::Sinkhorn => {
                let block = self.layers[il].sinkhorn.as_mut().ok_or_else(no_block)?;
                let held = super::wide_sparse_card_stream::CardSinkhorn::keep(ordinal, block)?;
                block.card_mix = held;
                Ok(block.card_mix.bytes() as u64)
            }
        }
    }

    /// Times 1 weight of `kind`, of held layer `il`, over 1 row on the host and on card
    /// `ordinal`.
    ///
    /// The row is the layer's feed-forward norm, repeated to the weight's input width.
    /// A first pass reads the bytes in and loads the kernels, the second is timed.
    /// The card copy is dropped after: it is measured in a unit of its own.
    #[cfg(feature = "gpu")]
    pub(super) fn measure_always_read(
        &self,
        ordinal: usize,
        kind: AlwaysRead,
        il: usize,
    ) -> std::result::Result<KindMeasure, String> {
        use super::card_packed::{Choices, PackedDev};
        use super::wide_expert_card::gpu::{RouterCard, SharedCard};
        use crate::tensor::{Device, Tensor};
        let text = |e: crate::tensor::Error| e.to_string();
        let block = self.layers[il].sinkhorn.as_ref().ok_or("no wide layer")?;
        let ws = self.wide_sparse.as_ref().ok_or("no wide residual")?;
        let norm = block
            .ffn_norm
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(text)?;
        let row = |width: usize| -> Vec<f32> { norm.iter().copied().cycle().take(width).collect() };
        let Device::Cuda(dev) = Device::new_cuda(ordinal).map_err(text)? else {
            return Err("not a CUDA card".into());
        };
        let probe = Choices::default();
        let timed =
            |f: &mut dyn FnMut() -> crate::tensor::Result<()>| -> std::result::Result<f64, String> {
                f().map_err(text)?;
                let started = std::time::Instant::now();
                f().map_err(text)?;
                Ok(started.elapsed().as_secs_f64())
            };
        let projection = |p: &crate::tensor::blockscaled::projection::Projection| -> std::result::Result<KindMeasure, String> {
            let width = p.dims().get(1).copied().ok_or("not a matrix")?;
            let x = row(width);
            let xt = Tensor::from_vec(x.clone(), (1, width), &Device::Cpu).map_err(text)?;
            let host_secs = timed(&mut || p.apply_here(&xt).map(drop))?;
            let packed = PackedDev::new(&dev, p, &probe).map_err(text)?;
            let card_secs = timed(&mut || {
                let up = crate::tensor::cuda::upload_f32(&dev, "probe row", &x)?;
                let y = packed.apply(&up, 1)?;
                super::card_mixers::download(&dev, &y, packed.out()).map(drop)
            })?;
            Ok(KindMeasure {
                kind,
                bytes: packed.bytes() as u64,
                host_secs,
                card_secs,
            })
        };
        match kind {
            AlwaysRead::Output => projection(&block.output),
            AlwaysRead::Head => projection(&ws.head),
            AlwaysRead::Engram => {
                let engram = block.engram.as_ref().ok_or("no engram")?;
                projection(engram.block.wkv())
            }
            AlwaysRead::Shared => {
                let shared = block.ffn.shared.as_ref().ok_or("no shared expert")?;
                let width = shared.gate.dims().get(1).copied().ok_or("not a matrix")?;
                let x = row(width);
                let xt = Tensor::from_vec(x.clone(), (1, width), &Device::Cpu).map_err(text)?;
                let limit = block.ffn.limit;
                let host_secs = timed(&mut || shared.forward(&xt, limit).map(drop))?;
                let card = SharedCard::new(ordinal, shared, &probe).map_err(text)?;
                let card_secs = timed(&mut || card.run(&x, 1, limit).map(drop))?;
                Ok(KindMeasure {
                    kind,
                    bytes: card.bytes() as u64,
                    host_secs,
                    card_secs,
                })
            }
            AlwaysRead::Router => {
                let router = &block.ffn.router;
                let width = router.dims().get(1).copied().ok_or("not a matrix")?;
                let x = row(width);
                let xt = Tensor::from_vec(x.clone(), (1, width), &Device::Cpu).map_err(text)?;
                let host = router
                    .apply_here(&xt)
                    .and_then(|t| t.flatten_all()?.to_vec1::<f32>())
                    .map_err(text)?;
                let host_secs = timed(&mut || router.apply_here(&xt).map(drop))?;
                let card = RouterCard::new(ordinal, router).map_err(text)?;
                let got = card.logits(&x).map_err(text)?;
                let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
                if bits(&host) != bits(&got) {
                    return Err("the card's logits differ from the host's bits".into());
                }
                // A decoded row on the card stream finds its input on the card and takes the
                // logits down with it.
                let card_secs = card.time_on_card(&x, &mut |f| timed(f))?;
                Ok(KindMeasure {
                    kind,
                    bytes: card.bytes() as u64,
                    host_secs,
                    card_secs,
                })
            }
            AlwaysRead::Sinkhorn => {
                let width = ws.streams * norm.len();
                let x = row(width);
                let card = super::wide_sparse_card_stream::CardSinkhorn::keep(ordinal, block)
                    .map_err(text)?;
                let (host_secs, card_secs) = card.time_layer(block, &x, &mut |f| timed(f))?;
                Ok(KindMeasure {
                    kind,
                    bytes: card.bytes() as u64,
                    host_secs,
                    card_secs,
                })
            }
        }
    }
}

/// Bytes of a shared expert's 3 weights as stored.
#[cfg(feature = "gpu")]
fn shared_bytes(e: &super::block_scaled_experts::BlockScaledExpert) -> u64 {
    (e.gate.bytes() + e.up.bytes() + e.down.bytes()) as u64
}
