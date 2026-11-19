//! What a model keeps on its cards for later requests and can give back, card by card, for
//! another component: see `place::reclaim`.

use super::{GenericHeteroTransformer, LayerDevice};

impl GenericHeteroTransformer {
    /// The ordinals of the cards the model runs on.
    pub fn card_ordinals(&self) -> Vec<usize> {
        let layers = self.layer_devs.iter().filter_map(|d| match d {
            LayerDevice::Cuda(card) => Some(*card),
            LayerDevice::Cpu => None,
        });
        #[cfg(feature = "gpu")]
        let layers = layers.chain(
            self.cuda_devices
                .values()
                .filter_map(|d| d.as_cuda_device().ok().map(|c| c.0.ordinal())),
        );
        let mut cards: Vec<usize> = layers.collect();
        cards.sort_unstable();
        cards.dedup();
        cards
    }

    /// The card memory each card of the model takes as its quantised caches grow to `planned`
    /// positions, the context the load planned: what the plan counted and the cards do not
    /// hold yet.
    #[cfg(feature = "gpu")]
    pub fn kv_growth_by_card(&self, planned: usize) -> Vec<(usize, u64)> {
        let mut out: Vec<(usize, u64)> = Vec::new();
        for (layer, dev) in self.layers.iter().zip(&self.layer_devs) {
            let (LayerDevice::Cuda(card), Some(cache)) = (dev, layer.quant_kv.as_ref()) else {
                continue;
            };
            let bytes = cache.growth_bytes(planned);
            match out.iter_mut().find(|(c, _)| c == card) {
                Some(entry) => entry.1 += bytes,
                None => out.push((*card, bytes)),
            }
        }
        out
    }

    /// The card memory each card of the model takes as its quantised caches grow: none without a
    /// card runtime.
    #[cfg(not(feature = "gpu"))]
    pub fn kv_growth_by_card(&self, _planned: usize) -> Vec<(usize, u64)> {
        Vec::new()
    }

    /// Gives back the extra expert slots of the layers on card `card`, the coldest first, until
    /// `bytes` of them are freed or none is left. True when a layer gave its slots back.
    /// The next request's prefill decides again what they hold.
    #[cfg(feature = "gpu")]
    pub fn lend_extra_slots_on(&self, card: usize, bytes: u64) -> bool {
        let mut held: Vec<_> = self
            .layers
            .iter()
            .zip(&self.layer_devs)
            .filter(|(_, d)| **d == LayerDevice::Cuda(card))
            .filter_map(|(l, _)| l.moe.as_ref().and_then(|m| m.tier.as_ref()))
            .filter(|t| t.has_extra())
            .map(|t| (t.extra_heat(), t))
            .collect();
        held.sort_by(|a, b| a.0.total_cmp(&b.0));
        let before = crate::inference::place::reclaim::free_on_card(card);
        let mut lent = false;
        for (_, tier) in held {
            if lent
                && crate::inference::place::reclaim::free_on_card(card).saturating_sub(before)
                    >= bytes
            {
                break;
            }
            lent |= tier.lend_extra();
        }
        if lent {
            self.hand_back_pools();
        }
        lent
    }

    /// Without a card no layer keeps expert slots on one.
    #[cfg(not(feature = "gpu"))]
    pub fn lend_extra_slots_on(&self, _card: usize, _bytes: u64) -> bool {
        false
    }
}
