//! The host tier of a KV cache: the pages of a context longer than the cards hold.
//!
//! The cache is cut into pages of a fixed number of positions.
//! Host memory keeps every page of an attending layer.
//! The card keeps a fixed number of page slots per layer.
//!
//! 2 pages are always on the card:
//! the first page, which holds the attention sink positions,
//! and the newest page, which the decode writes.
//!
//! Each step, every other page is ranked by an upper bound of its attention score.
//! The bound comes from the per-channel minimum and maximum of the page's keys.
//! The best pages fill the remaining slots.
//! A page missing from the card is fetched from host memory before the layer attends.
//! The pages fetched per step are capped by what the host link copies under one layer's compute.
//!
//! The attention over the pages on the card is exact.
//! With every page attended, the output equals the attention over the whole cache.
//! The pages are then read a card's slots at a time, and the rounds merge their softmax.
//! A context the cards hold never engages the tier.

use super::paged_attention::{host_attn_rows, HostAttnShape};

/// Pages always held on the card: the first page and the newest page.
pub const KEPT_PAGES: usize = 2;

/// The geometry of one layer's pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageShape {
    /// Key and value heads of the layer.
    pub n_kv: usize,
    /// Channels of one head.
    pub head_dim: usize,
    /// Positions of one page.
    pub page: usize,
}

impl PageShape {
    /// Values one position holds in K, and again in V.
    pub fn row(&self) -> usize {
        self.n_kv * self.head_dim
    }

    /// Bytes of one page of one layer, K and V, at `elem_bytes` per value.
    pub fn page_bytes(&self, elem_bytes: usize) -> u64 {
        2 * (self.page * self.row() * elem_bytes) as u64
    }
}

/// The pages of one attending layer held in host memory, with the key bounds of each.
///
/// K and V are `[positions, n_kv, head_dim]`.
/// The bounds of page `p` are `[n_kv, head_dim]` each.
/// A page that is not yet full keeps the bounds of the positions it holds.
#[derive(Debug, Clone)]
pub struct HostPages {
    shape: PageShape,
    k: Vec<f32>,
    v: Vec<f32>,
    lo: Vec<f32>,
    hi: Vec<f32>,
    len: usize,
}

impl HostPages {
    pub fn new(shape: PageShape) -> Self {
        assert!(shape.page > 0 && shape.row() > 0, "a page holds positions");
        Self {
            shape,
            k: Vec::new(),
            v: Vec::new(),
            lo: Vec::new(),
            hi: Vec::new(),
            len: 0,
        }
    }

    pub fn shape(&self) -> PageShape {
        self.shape
    }

    /// Positions held.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Pages held, the last one possibly partial.
    pub fn pages(&self) -> usize {
        self.len.div_ceil(self.shape.page)
    }

    /// The positions of page `p`.
    pub fn span(&self, p: usize) -> std::ops::Range<usize> {
        let start = p * self.shape.page;
        start..(start + self.shape.page).min(self.len)
    }

    /// Appends whole positions: `k` and `v` are `[positions, n_kv, head_dim]`.
    pub fn append(&mut self, k: &[f32], v: &[f32]) {
        let row = self.shape.row();
        assert!(
            k.len() == v.len() && k.len().is_multiple_of(row),
            "K and V carry whole positions"
        );
        for (kr, vr) in k.chunks_exact(row).zip(v.chunks_exact(row)) {
            if self.len.is_multiple_of(self.shape.page) {
                self.lo.extend_from_slice(kr);
                self.hi.extend_from_slice(kr);
            } else {
                let p = self.len / self.shape.page;
                let lo = &mut self.lo[p * row..][..row];
                let hi = &mut self.hi[p * row..][..row];
                for ((l, h), x) in lo.iter_mut().zip(hi.iter_mut()).zip(kr) {
                    *l = l.min(*x);
                    *h = h.max(*x);
                }
            }
            self.k.extend_from_slice(kr);
            self.v.extend_from_slice(vr);
            self.len += 1;
        }
    }

    /// Keeps the first `len` positions and the bounds of the pages they fill.
    pub fn truncate(&mut self, len: usize) {
        if len >= self.len {
            return;
        }
        let row = self.shape.row();
        self.k.truncate(len * row);
        self.v.truncate(len * row);
        self.len = len;
        let pages = self.pages();
        self.lo.truncate(pages * row);
        self.hi.truncate(pages * row);
        // The last page lost positions: its bounds are read again from the ones it keeps.
        if !len.is_multiple_of(self.shape.page) {
            let p = pages - 1;
            let span = self.span(p);
            let (lo, hi) = (
                &mut self.lo[p * row..][..row],
                &mut self.hi[p * row..][..row],
            );
            lo.copy_from_slice(&self.k[span.start * row..][..row]);
            hi.copy_from_slice(&self.k[span.start * row..][..row]);
            for t in span.start + 1..span.end {
                for ((l, h), x) in lo
                    .iter_mut()
                    .zip(hi.iter_mut())
                    .zip(&self.k[t * row..][..row])
                {
                    *l = l.min(*x);
                    *h = h.max(*x);
                }
            }
        }
    }

    /// The keys of page `p`, `[positions, n_kv, head_dim]`.
    pub fn page_keys(&self, p: usize) -> &[f32] {
        let row = self.shape.row();
        let span = self.span(p);
        &self.k[span.start * row..span.end * row]
    }

    /// The values of page `p`, `[positions, n_kv, head_dim]`.
    pub fn page_values(&self, p: usize) -> &[f32] {
        let row = self.shape.row();
        let span = self.span(p);
        &self.v[span.start * row..span.end * row]
    }

    /// The largest score any query head of `q` (`[n_head, head_dim]`) can give a key of page `p`.
    ///
    /// For each channel the product with the query is largest at the minimum or the maximum
    /// key of the page, so the sum of those products bounds every key's dot product.
    pub fn bound(&self, p: usize, q: &[f32], n_head: usize) -> f32 {
        let (nkv, hd) = (self.shape.n_kv, self.shape.head_dim);
        let group = n_head / nkv;
        let row = self.shape.row();
        let (lo, hi) = (&self.lo[p * row..][..row], &self.hi[p * row..][..row]);
        let mut best = f32::NEG_INFINITY;
        for h in 0..n_head {
            let kvh = h / group;
            let qh = &q[h * hd..][..hd];
            let (l, u) = (&lo[kvh * hd..][..hd], &hi[kvh * hd..][..hd]);
            let s: f32 = qh
                .iter()
                .zip(l.iter().zip(u))
                .map(|(x, (a, b))| (x * a).max(x * b))
                .sum();
            best = best.max(s);
        }
        best
    }
}

/// How many pages a step attends besides the kept ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Selection {
    /// Every page: the output is the attention over the whole cache.
    Every,
    /// The pages with the highest bounds, this many at most.
    Top(usize),
}

/// The pages one step of one layer attends.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Pick {
    /// The first and the newest page, ascending.
    pub kept: Vec<usize>,
    /// The other pages chosen, highest bound first.
    pub ranked: Vec<usize>,
}

impl Pick {
    /// Every page picked, ascending.
    pub fn all(&self) -> Vec<usize> {
        let mut all: Vec<usize> = self.kept.iter().chain(&self.ranked).copied().collect();
        all.sort_unstable();
        all
    }
}

/// The pages the query `q` (`[n_head, head_dim]`) attends under `selection`.
pub fn select(pages: &HostPages, q: &[f32], n_head: usize, selection: Selection) -> Pick {
    let n = pages.pages();
    let mut kept: Vec<usize> = [0, n.saturating_sub(1)].into_iter().take(n).collect();
    kept.dedup();
    let mut others: Vec<(f32, usize)> = (0..n)
        .filter(|p| !kept.contains(p))
        .map(|p| (pages.bound(p, q, n_head), p))
        .collect();
    others.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let take = match selection {
        Selection::Every => others.len(),
        Selection::Top(t) => t.min(others.len()),
    };
    Pick {
        kept,
        ranked: others[..take].iter().map(|(_, p)| *p).collect(),
    }
}

/// The keys and values of `picked` pages, in the order given, `[positions, n_kv, head_dim]`.
fn gather(pages: &HostPages, picked: &[usize]) -> (Vec<f32>, Vec<f32>, usize) {
    let mut k = Vec::new();
    let mut v = Vec::new();
    for &p in picked {
        k.extend_from_slice(pages.page_keys(p));
        v.extend_from_slice(pages.page_values(p));
    }
    let keys = k.len() / pages.shape().row();
    (k, v, keys)
}

/// The decode attention of `q` (`[n_head, head_dim]`) over the `picked` pages.
///
/// Keys carry their rotation already, so the attention reads them in any order.
/// A windowed layer keeps its own cache and does not reach this tier.
pub fn attend(pages: &HostPages, picked: &[usize], q: &[f32], shape: &HostAttnShape) -> Vec<f32> {
    assert!(shape.window.is_none(), "a windowed layer holds its window");
    let (k, v, keys) = gather(pages, picked);
    host_attn_rows(q, &k, &v, &[keys], shape)
}

/// The running softmax of one query head over the keys seen so far.
#[derive(Debug, Clone)]
struct Running {
    max: f32,
    sum: f32,
    acc: Vec<f32>,
}

/// The decode attention of `q` over `picked` pages read `slots` pages at a time.
///
/// Each round attends the pages the card holds at once, and the rounds merge by their running
/// maxima and sums.
/// It equals [`attend`] over the same pages within float rounding.
/// It is the attention of every page when the card holds fewer slots than the context has pages.
pub fn attend_in_rounds(
    pages: &HostPages,
    picked: &[usize],
    slots: usize,
    q: &[f32],
    shape: &HostAttnShape,
) -> Vec<f32> {
    assert!(shape.window.is_none(), "a windowed layer holds its window");
    let (nh, hd) = (shape.n_head, shape.head_dim);
    let group = nh / shape.n_kv_head;
    let row = pages.shape().row();
    let mut heads: Vec<Running> = (0..nh)
        .map(|h| Running {
            max: shape.sinks.map_or(f32::NEG_INFINITY, |s| s[h]),
            sum: 0.0,
            acc: vec![0.0; hd],
        })
        .collect();
    let mut scores = Vec::new();
    for round in picked.chunks(slots.max(1)) {
        let (k, v, keys) = gather(pages, round);
        for (h, run) in heads.iter_mut().enumerate() {
            let kvh = h / group;
            let qh = &q[h * hd..][..hd];
            scores.clear();
            let mut mx = run.max;
            for t in 0..keys {
                let kt = &k[t * row + kvh * hd..][..hd];
                let sc = qh.iter().zip(kt).map(|(a, b)| a * b).sum::<f32>() * shape.scale;
                mx = mx.max(sc);
                scores.push(sc);
            }
            let carry = if run.max == f32::NEG_INFINITY {
                0.0
            } else {
                (run.max - mx).exp()
            };
            run.sum *= carry;
            run.acc.iter_mut().for_each(|a| *a *= carry);
            for (t, sc) in scores.iter().enumerate() {
                let p = (sc - mx).exp();
                run.sum += p;
                let vt = &v[t * row + kvh * hd..][..hd];
                for (a, x) in run.acc.iter_mut().zip(vt) {
                    *a += p * x;
                }
            }
            run.max = mx;
        }
    }
    let mut out = vec![0f32; nh * hd];
    for (h, run) in heads.iter().enumerate() {
        let sum = run.sum + shape.sinks.map_or(0.0, |s| (s[h] - run.max).exp());
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for (o, a) in out[h * hd..][..hd].iter_mut().zip(&run.acc) {
            *o = a * inv;
        }
    }
    out
}

/// A copy of one page from host memory to one card slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fetch {
    pub page: usize,
    pub slot: usize,
}

/// Which page each card slot of one layer holds.
#[derive(Debug, Clone, Default)]
pub struct CardSlots {
    held: Vec<Option<usize>>,
}

impl CardSlots {
    pub fn new(slots: usize) -> Self {
        Self {
            held: vec![None; slots],
        }
    }

    pub fn slots(&self) -> usize {
        self.held.len()
    }

    pub fn slot_of(&self, page: usize) -> Option<usize> {
        self.held.iter().position(|h| *h == Some(page))
    }

    /// The pages on the card, ascending.
    pub fn resident(&self) -> Vec<usize> {
        let mut pages: Vec<usize> = self.held.iter().flatten().copied().collect();
        pages.sort_unstable();
        pages
    }

    /// Forgets every page, as a reset of the sequence does.
    pub fn clear(&mut self) {
        self.held.iter_mut().for_each(|h| *h = None);
    }

    /// Forgets the pages from `first` on, as a trim of the sequence does.
    pub fn drop_from(&mut self, first: usize) {
        for h in self.held.iter_mut() {
            if h.is_some_and(|p| p >= first) {
                *h = None;
            }
        }
    }

    /// Brings the pages of `pick` onto the card and returns the copies that takes.
    ///
    /// Every kept page is admitted.
    /// At most `cap` ranked pages missing from the card are admitted, highest bound first.
    /// A missing page takes a free slot, else the slot of a page `pick` does not name.
    /// A page not named stays on the card while its slot is not needed.
    pub fn admit(&mut self, pick: &Pick, cap: usize) -> Vec<Fetch> {
        let named: Vec<usize> = pick.kept.iter().chain(&pick.ranked).copied().collect();
        let missing_kept = pick.kept.iter().filter(|p| self.slot_of(**p).is_none());
        let missing_ranked = pick
            .ranked
            .iter()
            .filter(|p| self.slot_of(**p).is_none())
            .take(cap);
        let wanted: Vec<usize> = missing_kept.chain(missing_ranked).copied().collect();
        let mut fetches = Vec::with_capacity(wanted.len());
        for page in wanted {
            let free = self.held.iter().position(Option::is_none).or_else(|| {
                self.held
                    .iter()
                    .position(|h| h.is_some_and(|p| !named.contains(&p)))
            });
            let Some(slot) = free else { break };
            self.held[slot] = Some(page);
            fetches.push(Fetch { page, slot });
        }
        fetches
    }
}

/// What sizes the tier of one load.
#[derive(Debug, Clone, Copy)]
pub struct TierInputs {
    /// The context the sequence reaches.
    pub context: u64,
    /// The context the plan holds on the cards.
    pub held_by_cards: u64,
    /// Card bytes the plan leaves for page slots, over every attending layer.
    pub card_room: u64,
    /// Attending layers on the cards.
    pub layers: usize,
    /// Bytes of one page of one layer, K and V.
    pub page_bytes: u64,
    /// Weight bytes one layer reads per token.
    pub layer_bytes: u64,
    /// The card's memory bandwidth, GB/s.
    pub card_gbs: f64,
    /// The card's link to the host, GB/s.
    pub link_gbs: f64,
    /// The level asks for the attention of every page.
    pub exact: bool,
}

/// The tier of one load: the slots each layer keeps and the copies a step may make.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TierPlan {
    /// Page slots each attending layer keeps on the card.
    pub slots: usize,
    /// Pages a layer may fetch per step while the copy stays under its compute.
    pub fetch_per_step: usize,
    /// The pages a step attends besides the kept ones.
    pub selection: Selection,
}

/// The tier `inputs` call for, `None` when the cards hold the context or hold no page to select.
///
/// The slots are the card room shared over the layers, a page per slot.
/// A layer's compute is the time the card reads its weights and its slots.
/// The pages fetched per step are what the link copies in that time.
pub fn plan(inputs: &TierInputs) -> Option<TierPlan> {
    if inputs.context <= inputs.held_by_cards || inputs.layers == 0 || inputs.page_bytes == 0 {
        return None;
    }
    let slots = (inputs.card_room / (inputs.layers as u64 * inputs.page_bytes)) as usize;
    if slots <= KEPT_PAGES {
        return None;
    }
    let read = inputs.layer_bytes as f64 + (slots as u64 * inputs.page_bytes) as f64;
    let fetch_per_step = match inputs.card_gbs > 0.0 && inputs.link_gbs > 0.0 {
        true => (read * inputs.link_gbs / inputs.card_gbs / inputs.page_bytes as f64) as usize,
        false => 0,
    };
    let selection = match inputs.exact {
        true => Selection::Every,
        false => Selection::Top(slots - KEPT_PAGES),
    };
    Some(TierPlan {
        slots,
        fetch_per_step,
        selection,
    })
}

/// The context a load reaches with the tier: `full`, cut where host memory runs out.
///
/// Host memory keeps every page: `host_room` bytes at `host_bytes_per_token` a position over
/// every attending layer.
/// The tier never cuts below what the cards hold.
pub fn reach(full: u64, held_by_cards: u64, host_room: u64, host_bytes_per_token: u64) -> u64 {
    if full <= held_by_cards {
        return full;
    }
    let host_positions = host_room
        .checked_div(host_bytes_per_token)
        .unwrap_or(u64::MAX);
    full.min(host_positions.max(held_by_cards))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reproducible stream of values in `[-1, 1)`.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }

        fn fill(&mut self, n: usize, amp: f32) -> Vec<f32> {
            (0..n).map(|_| self.next() * amp).collect()
        }
    }

    const SHAPE: PageShape = PageShape {
        n_kv: 2,
        head_dim: 32,
        page: 16,
    };
    const N_HEAD: usize = 4;

    fn attn_shape() -> HostAttnShape<'static> {
        HostAttnShape {
            n_head: N_HEAD,
            n_kv_head: SHAPE.n_kv,
            head_dim: SHAPE.head_dim,
            scale: 1.0 / (SHAPE.head_dim as f32).sqrt(),
            window: None,
            sinks: None,
        }
    }

    /// `pages` pages of noise keys, the keys of the `planted` pages lying along the query.
    /// The planted strengths fall with their order, so their true masses are ranked.
    fn planted(seed: u64, pages: usize, planted: &[usize]) -> (HostPages, Vec<f32>) {
        let mut rng = Lcg(seed);
        let row = SHAPE.row();
        let q = rng.fill(N_HEAD * SHAPE.head_dim, 1.0);
        let mut k = rng.fill(pages * SHAPE.page * row, 0.2);
        let v = rng.fill(pages * SHAPE.page * row, 1.0);
        let group = N_HEAD / SHAPE.n_kv;
        for (rank, &p) in planted.iter().enumerate() {
            let strength = 4.0 - 0.5 * rank as f32;
            for t in p * SHAPE.page..(p + 1) * SHAPE.page {
                for kvh in 0..SHAPE.n_kv {
                    let qh = &q[kvh * group * SHAPE.head_dim..][..SHAPE.head_dim];
                    let kt = &mut k[t * row + kvh * SHAPE.head_dim..][..SHAPE.head_dim];
                    for (x, y) in kt.iter_mut().zip(qh) {
                        *x += y * strength;
                    }
                }
            }
        }
        let mut store = HostPages::new(SHAPE);
        store.append(&k, &v);
        (store, q)
    }

    /// The exact softmax mass each page takes, summed over the query heads.
    fn true_mass(pages: &HostPages, q: &[f32]) -> Vec<f64> {
        let shape = attn_shape();
        let (hd, row) = (SHAPE.head_dim, SHAPE.row());
        let group = N_HEAD / SHAPE.n_kv;
        let all: Vec<usize> = (0..pages.pages()).collect();
        let (k, _, keys) = gather(pages, &all);
        let mut mass = vec![0f64; pages.pages()];
        for h in 0..N_HEAD {
            let qh = &q[h * hd..][..hd];
            let kvh = h / group;
            let sc: Vec<f64> = (0..keys)
                .map(|t| {
                    let kt = &k[t * row + kvh * hd..][..hd];
                    qh.iter().zip(kt).map(|(a, b)| (a * b) as f64).sum::<f64>() * shape.scale as f64
                })
                .collect();
            let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = sc.iter().map(|s| (s - mx).exp()).sum();
            for (t, s) in sc.iter().enumerate() {
                mass[t / SHAPE.page] += (s - mx).exp() / z;
            }
        }
        mass
    }

    #[test]
    fn the_bound_holds_every_key_of_its_page() {
        let (pages, q) = planted(7, 6, &[3]);
        let row = SHAPE.row();
        let group = N_HEAD / SHAPE.n_kv;
        for p in 0..pages.pages() {
            let bound = pages.bound(p, &q, N_HEAD);
            let k = pages.page_keys(p);
            for t in 0..k.len() / row {
                for h in 0..N_HEAD {
                    let kt = &k[t * row + (h / group) * SHAPE.head_dim..][..SHAPE.head_dim];
                    let qh = &q[h * SHAPE.head_dim..][..SHAPE.head_dim];
                    let dot: f32 = qh.iter().zip(kt).map(|(a, b)| a * b).sum();
                    assert!(dot <= bound + 1e-4, "page {p} key {t}: {dot} > {bound}");
                }
            }
        }
    }

    #[test]
    fn the_bounds_pick_the_pages_with_the_most_attention_mass() {
        let planted_pages = [9, 4, 22, 15];
        let (pages, q) = planted(11, 32, &planted_pages);
        let pick = select(&pages, &q, N_HEAD, Selection::Top(planted_pages.len()));
        assert_eq!(pick.kept, vec![0, 31]);
        assert_eq!(pick.ranked, planted_pages.to_vec());
        // The true masses rank the pages the same way, the kept ones aside.
        let mass = true_mass(&pages, &q);
        let mut by_mass: Vec<usize> = (1..31).collect();
        by_mass.sort_by(|a, b| mass[*b].total_cmp(&mass[*a]));
        assert_eq!(&by_mass[..planted_pages.len()], &planted_pages);
    }

    #[test]
    fn every_page_attended_gives_the_full_attention() {
        let (pages, q) = planted(13, 40, &[5, 17, 33]);
        let shape = attn_shape();
        let row = SHAPE.row();
        let all = select(&pages, &q, N_HEAD, Selection::Every).all();
        assert_eq!(all, (0..pages.pages()).collect::<Vec<_>>());
        let full_k = &pages.k[..pages.len() * row];
        let full_v = &pages.v[..pages.len() * row];
        let full = host_attn_rows(&q, full_k, full_v, &[pages.len()], &shape);
        assert_eq!(attend(&pages, &all, &q, &shape), full);
        // Read through 3 slots at a time, the rounds merge to the same output.
        let rounds = attend_in_rounds(&pages, &all, 3, &q, &shape);
        for (a, b) in rounds.iter().zip(&full) {
            assert!((a - b).abs() <= 1e-5 * b.abs().max(1.0), "{a} vs {b}");
        }
    }

    #[test]
    fn a_selected_subset_stays_within_the_mass_it_leaves_out() {
        let planted_pages = [8, 21, 30, 44, 52];
        let (pages, q) = planted(17, 64, &planted_pages);
        let shape = attn_shape();
        let hd = SHAPE.head_dim;
        let row = SHAPE.row();
        let full = host_attn_rows(
            &q,
            &pages.k[..pages.len() * row],
            &pages.v[..pages.len() * row],
            &[pages.len()],
            &shape,
        );
        let pick = select(&pages, &q, N_HEAD, Selection::Top(planted_pages.len() + 3));
        let picked = pick.all();
        let sub = attend(&pages, &picked, &q, &shape);
        // The full output mixes the subset's output and the rest's by the mass m left out:
        // their gap is at most 2 m times the largest value.
        let vmax = pages.v.iter().fold(0f32, |m, x| m.max(x.abs()));
        let group = N_HEAD / SHAPE.n_kv;
        let all: Vec<usize> = (0..pages.pages()).collect();
        let (k, _, keys) = gather(&pages, &all);
        for h in 0..N_HEAD {
            let qh = &q[h * hd..][..hd];
            let kvh = h / group;
            let sc: Vec<f64> = (0..keys)
                .map(|t| {
                    let kt = &k[t * row + kvh * hd..][..hd];
                    qh.iter().zip(kt).map(|(a, b)| (a * b) as f64).sum::<f64>() * shape.scale as f64
                })
                .collect();
            let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = sc.iter().map(|s| (s - mx).exp()).sum();
            let left_out: f64 = sc
                .iter()
                .enumerate()
                .filter(|(t, _)| !picked.contains(&(t / SHAPE.page)))
                .map(|(_, s)| (s - mx).exp() / z)
                .sum();
            // A head the planted keys lie along loses less mass than the share of pages left out.
            if h % group == 0 {
                let pages_left_out = 1.0 - picked.len() as f64 / pages.pages() as f64;
                assert!(left_out < pages_left_out, "head {h} leaves out {left_out}");
            }
            let tol = 2.0 * left_out as f32 * vmax + 1e-5;
            for c in 0..hd {
                let gap = (sub[h * hd + c] - full[h * hd + c]).abs();
                assert!(gap <= tol, "head {h} channel {c}: {gap} > {tol}");
            }
        }
    }

    #[test]
    fn truncation_reads_the_bounds_of_the_last_page_again() {
        let (mut pages, q) = planted(19, 4, &[2]);
        let mut fresh = HostPages::new(SHAPE);
        let cut = 2 * SHAPE.page + 3;
        let row = SHAPE.row();
        fresh.append(&pages.k[..cut * row], &pages.v[..cut * row]);
        pages.truncate(cut);
        assert_eq!(pages.len(), cut);
        assert_eq!(pages.pages(), 3);
        for p in 0..3 {
            assert_eq!(pages.bound(p, &q, N_HEAD), fresh.bound(p, &q, N_HEAD));
        }
    }

    #[test]
    fn slots_keep_what_they_hold_and_cap_the_copies() {
        let mut slots = CardSlots::new(5);
        let pick = Pick {
            kept: vec![0, 9],
            ranked: vec![4, 6, 2],
        };
        let first = slots.admit(&pick, 2);
        // The kept pages are always admitted, the ranked ones up to the cap.
        let pages: Vec<usize> = first.iter().map(|f| f.page).collect();
        assert_eq!(pages, vec![0, 9, 4, 6]);
        assert_eq!(slots.resident(), vec![0, 4, 6, 9]);
        // The next step names 2 new pages.
        // 7 takes the free slot and 2 takes the slot of 6, which the step does not name.
        let next = Pick {
            kept: vec![0, 9],
            ranked: vec![7, 4, 2],
        };
        let second = slots.admit(&next, 2);
        assert_eq!(
            second.iter().map(|f| f.page).collect::<Vec<_>>(),
            vec![7, 2]
        );
        assert_eq!(slots.resident(), vec![0, 2, 4, 7, 9]);
        // A new kept page takes the slot of 9, which the step does not name.
        let third = Pick {
            kept: vec![0, 10],
            ranked: vec![2, 4, 7],
        };
        let copies = slots.admit(&third, 3);
        assert_eq!(copies.len(), 1);
        assert_eq!(copies[0].page, 10);
        assert_eq!(slots.resident(), vec![0, 2, 4, 7, 10]);
        slots.drop_from(7);
        assert_eq!(slots.resident(), vec![0, 2, 4]);
    }

    fn inputs(context: u64, held_by_cards: u64) -> TierInputs {
        TierInputs {
            context,
            held_by_cards,
            card_room: 64 * SHAPE.page_bytes(2) * 10,
            layers: 10,
            page_bytes: SHAPE.page_bytes(2),
            layer_bytes: 200 * SHAPE.page_bytes(2),
            card_gbs: 900.0,
            link_gbs: 25.0,
            exact: false,
        }
    }

    #[test]
    fn a_context_the_cards_hold_never_engages_the_tier() {
        assert_eq!(plan(&inputs(8192, 8192)), None);
        assert_eq!(plan(&inputs(4096, 8192)), None);
        assert_eq!(reach(4096, 8192, u64::MAX, 1), 4096);
        assert_eq!(reach(8192, 8192, 0, 1), 8192);
    }

    #[test]
    fn the_plan_shares_the_card_room_and_paces_copies_by_the_link() {
        let tier = plan(&inputs(131072, 8192)).expect("a longer context engages the tier");
        assert_eq!(tier.slots, 64);
        assert_eq!(tier.selection, Selection::Top(64 - KEPT_PAGES));
        // A layer reads 200 pages of weights and 64 of slots: the link copies 25/900 of that.
        assert_eq!(tier.fetch_per_step, (264.0 * 25.0 / 900.0) as usize);
        let exact = plan(&TierInputs {
            exact: true,
            ..inputs(131072, 8192)
        })
        .unwrap();
        assert_eq!(exact.selection, Selection::Every);
        // Room for the kept pages alone selects nothing.
        let narrow = TierInputs {
            card_room: KEPT_PAGES as u64 * 10 * SHAPE.page_bytes(2),
            ..inputs(131072, 8192)
        };
        assert_eq!(plan(&narrow), None);
    }

    #[test]
    fn host_memory_bounds_the_reach_past_the_cards() {
        // 1 MB a position: 100 GB of host room holds 100k positions.
        let per_token = 1 << 20;
        let room = 100_000u64 << 20;
        assert_eq!(reach(262144, 8192, room, per_token), 100_000);
        assert_eq!(reach(65536, 8192, room, per_token), 65536);
        // Less host room than the cards hold keeps what the cards hold.
        assert_eq!(reach(262144, 8192, 1 << 20, per_token), 8192);
    }
}
