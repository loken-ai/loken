//! The streamed placement: a model too large for the cards and the host together runs on the
//! host from its mapping, and the cards keep what every token reads and as many routed experts
//! as they have room for. Opened from what the model declares, under the engine's own
//! switches: told to serve on the host, or past every split on the pressure ladder, it opens
//! nothing; otherwise it takes the cards fastest first, as every other placement does.

use super::experts::{Expert, ExpertOffload};
use super::Offload;
use std::sync::Arc;

/// What a model tells the placement about itself.
pub struct Demand<'a> {
    /// Bytes every token reads: kept on the cards ahead of any routed expert.
    pub always_read: usize,
    /// The most a step asks a card for beside the weights it keeps: a batch's products.
    pub transient: usize,
    /// Experts a token routes to at once: the lanes opened.
    pub concurrency: usize,
    /// The model's layer count: how many expert blocks a token runs, and so the clock span within
    /// which a card holds an expert against eviction as this token's own.
    pub layers: usize,
    /// Per layer, experts ranked by how often the calibration routed to them.
    pub prior: &'a [Vec<usize>],
    /// Expert `id` of layer `layer`, for warming the cards.
    pub fetch: &'a dyn Fn(usize, usize) -> Option<Arc<Expert>>,
}

/// The cards behind a model's forward: `offload` for its steps, `lanes` for its routed experts.
pub struct Streamed {
    pub offload: Arc<dyn Offload>,
    pub lanes: Arc<ExpertOffload>,
}

impl Streamed {
    /// The placement for `demand`, or `None` when the model runs on the host alone.
    pub fn open(demand: &Demand) -> Option<Self> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = demand;
            None
        }
        #[cfg(feature = "cuda")]
        Self::open_cuda(demand)
    }

    #[cfg(feature = "cuda")]
    fn open_cuda(demand: &Demand) -> Option<Self> {
        use super::cuda::{lanes, Card, Cards};
        use super::room;
        use crate::inference::place::vram_manager;
        use crate::tensor::cuda::CudaDevice;
        if crate::gpu::force_cpu() || vram_manager::vram_force_cpu() {
            return None;
        }
        // Fastest first, through the engine's own probe: what it says is free is what every
        // placement is allowed to believe.
        let mut rooms = Vec::new();
        for (ordinal, _, dev) in vram_manager::probe(0) {
            let Ok((free, total)) = crate::tensor::cuda_ext::mem_get_info(&dev) else {
                tracing::warn!("card {ordinal}: no memory reading; not used");
                continue;
            };
            // From the card's total, not from what is free at this moment: what is free now
            // is what this placement is about to fill.
            let r = room::open(ordinal, total, demand.transient);
            tracing::info!(
                "card {ordinal}: keeping up to {} MB of weights, of {} in all ({} free now)",
                r.lock().unwrap().ceiling() / 1_000_000,
                total / 1_000_000,
                free / 1_000_000
            );
            rooms.push((ordinal, r));
        }
        if rooms.is_empty() {
            return None;
        }
        // The weights every token reads are set aside first, on the cards in order: as much as
        // the first has room for, the rest on the next.
        let mut left = demand.always_read;
        for (_, r) in &rooms {
            left = r.lock().unwrap().reserve(left);
        }
        if left > 0 {
            tracing::warn!(
                "{} MB of the weights every token reads have no card to go to; the host reads them",
                left / 1_000_000
            );
        }
        // Each card handle is its own stream: the steps' handles, one per card, and the lanes',
        // one per expert a token routes to, spread over the cards.
        let protect = demand.layers as u64;
        let open = |ordinal: usize, r: &'static std::sync::Mutex<room::Room>| {
            CudaDevice::new(ordinal)
                .ok()
                .map(|d| Arc::new(Card::new(d, r, protect)))
        };
        let cards: Vec<Arc<Card>> = rooms.iter().filter_map(|&(o, r)| open(o, r)).collect();
        // One lane per device, not one per expert a token routes to. A decode is latency-bound,
        // not throughput-bound: the cards and cores sit near-idle while the step waits on the
        // per-lane launch-and-download syncs, so a token's experts on one device run in a single
        // block with one sync rather than one per lane. It also keeps that device's residency in
        // one tier instead of fragmenting the hot set across `concurrency` of them.
        let _ = demand.concurrency;
        let lane_cards: Vec<Arc<Card>> =
            rooms.iter().filter_map(|&(o, r)| open(o, r)).collect();
        if cards.is_empty() || lane_cards.is_empty() {
            return None;
        }
        let lanes = Arc::new(lanes(lane_cards));
        warm(&lanes, demand);
        vram_manager::residency_changed();
        Some(Self {
            offload: Arc::new(Cards(cards)),
            lanes,
        })
    }
}

/// Fill the lanes' cards from the routing prior before any request: rank by rank across the
/// layers, so every layer gets its most routed-to experts first, until the cards take no more.
/// A short request would otherwise spend its first tokens filling them.
fn warm(lanes: &ExpertOffload, demand: &Demand) {
    let Some(warm) = &lanes.warm else { return };
    let deepest = demand.prior.iter().map(|l| l.len()).max().unwrap_or(0);
    let started = std::time::Instant::now();
    let mut taken = 0usize;
    for rank in 0..deepest {
        let mut any = false;
        for (layer, ids) in demand.prior.iter().enumerate() {
            let Some(&id) = ids.get(rank) else { continue };
            let Some(expert) = (demand.fetch)(layer, id) else {
                continue;
            };
            if warm(id, &expert) {
                any = true;
                taken += 1;
            }
        }
        if !any {
            break;
        }
    }
    if taken > 0 {
        tracing::info!(
            "{taken} experts kept on the cards from the routing prior in {:.1}s",
            started.elapsed().as_secs_f32()
        );
    }
}
