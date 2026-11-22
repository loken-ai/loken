//! Card memory a loaded model keeps only for its later requests, and gives back when another
//! component on the same card needs the room.
//!
//! A model registers itself as a [`Holder`] once loaded.
//! The registry holds it weakly: a model dropped leaves a dead entry, pruned at the next walk.
//! A placement that finds a card short asks the holders on that card, the cheapest kind first,
//! before it unloads a model or plans further off the card.
//! A holder serving a request gives back only what that request does not use.
//! A component loading onto the cards is listed as an arrival, a holder of nothing on every
//! card, until it is loaded.
//! A holder knows which of its cards another owner's holder runs on.

use std::sync::{Arc, Mutex, Weak};

/// What a holder keeps, cheapest to give back first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// The paged pool a stopped batch worker left with its model.
    ParkedPool,
    /// The paged pool of a batch worker that serves no request.
    IdlePool,
    /// Decode graphs recorded by an earlier request, kept for the next one.
    KeptGraphs,
    /// Conversation states copied aside for a later turn.
    Snapshots,
    /// Expert cache slots past the ones the load placed.
    ExpertSlots,
}

/// Every kind, in the order they are asked for.
pub const KINDS: [Kind; 5] = [
    Kind::ParkedPool,
    Kind::IdlePool,
    Kind::KeptGraphs,
    Kind::Snapshots,
    Kind::ExpertSlots,
];

/// A model that keeps card memory for its later requests.
pub trait Holder: Send + Sync {
    /// The model's name: a caller never asks itself.
    fn owner(&self) -> &str;
    /// The ordinals of the cards the model runs on.
    fn cards(&self) -> Vec<usize>;
    /// Gives back what the model keeps of `kind` on `card`, at least `bytes` of it when it holds
    /// that much. True when anything was given back.
    fn give_back(&self, kind: Kind, card: usize, bytes: u64) -> bool;
    /// The bytes of `kind` the model keeps on `card` and would give back, counted without giving
    /// anything back.
    /// A holder that cannot count a kind without dropping it counts 0.
    fn keeps(&self, kind: Kind, card: usize) -> u64 {
        let _ = (kind, card);
        0
    }
    /// Brings what the holder declared on its cards up to date, when it can without waiting.
    /// `shared` lists its cards another owner's holder runs on.
    fn refresh(&self, shared: &[usize]) {
        let _ = shared;
    }
}

fn holders() -> &'static Mutex<Vec<Weak<dyn Holder>>> {
    static HOLDERS: std::sync::OnceLock<Mutex<Vec<Weak<dyn Holder>>>> = std::sync::OnceLock::new();
    HOLDERS.get_or_init(Default::default)
}

/// Lists `holder` until it is dropped.
/// Every live holder then brings its declarations up to date.
pub fn register(holder: &Arc<dyn Holder>) {
    {
        let mut list = holders().lock().unwrap_or_else(|e| e.into_inner());
        list.retain(|w| w.strong_count() > 0);
        list.push(Arc::downgrade(holder));
    }
    refresh();
}

/// Has every live holder bring its declarations up to date, before a placement reads them.
pub fn refresh() {
    let live: Vec<Arc<dyn Holder>> = {
        let mut list = holders().lock().unwrap_or_else(|e| e.into_inner());
        list.retain(|w| w.strong_count() > 0);
        list.iter().filter_map(Weak::upgrade).collect()
    };
    let cards: Vec<Vec<usize>> = live.iter().map(|h| h.cards()).collect();
    for (i, holder) in live.iter().enumerate() {
        let shared: Vec<usize> = cards[i]
            .iter()
            .copied()
            .filter(|card| {
                live.iter()
                    .zip(&cards)
                    .any(|(other, on)| other.owner() != holder.owner() && on.contains(card))
            })
            .collect();
        holder.refresh(&shared);
    }
}

/// A component loading onto `cards`: it holds nothing and gives nothing back.
struct Arriving {
    owner: String,
    cards: Vec<usize>,
}

impl Holder for Arriving {
    fn owner(&self) -> &str {
        &self.owner
    }

    fn cards(&self) -> Vec<usize> {
        self.cards.clone()
    }

    fn give_back(&self, _kind: Kind, _card: usize, _bytes: u64) -> bool {
        false
    }
}

/// Lists a component as loading until it drops; the other holders then bring their
/// declarations up to date.
pub struct Arrival(Option<Arc<dyn Holder>>);

impl Drop for Arrival {
    fn drop(&mut self) {
        if self.0.take().is_some() {
            refresh();
        }
    }
}

/// Lists `owner` as loading onto `cards` until the returned arrival drops.
pub fn arrive_on(owner: &str, cards: Vec<usize>) -> Arrival {
    let holder: Arc<dyn Holder> = Arc::new(Arriving {
        owner: owner.to_string(),
        cards,
    });
    register(&holder);
    Arrival(Some(holder))
}

/// Lists `owner` as loading onto any card of the machine until the returned arrival drops.
pub fn arrive(owner: &str) -> Arrival {
    let cards = super::device_probe::probe_cuda_gpus()
        .iter()
        .map(|g| g.index)
        .collect();
    arrive_on(owner, cards)
}

/// The live holders other than `except` that run on `card`.
fn on_card(card: usize, except: &str) -> Vec<Arc<dyn Holder>> {
    let mut list = holders().lock().unwrap_or_else(|e| e.into_inner());
    list.retain(|w| w.strong_count() > 0);
    list.iter()
        .filter_map(Weak::upgrade)
        .filter(|h| h.owner() != except && h.cards().contains(&card))
        .collect()
}

/// Asks the holders other than `except` on `card` for their memory, the cheapest kind first,
/// until `free` reads `need` bytes or more on the card or nothing is left to ask.
/// `free` reads the card as the caller plans against it.
/// True when anything was given back.
pub fn make_room(card: usize, need: u64, except: &str, free: impl Fn() -> u64) -> bool {
    let mut given = false;
    for kind in KINDS {
        for holder in on_card(card, except) {
            // A kind is asked again while it gives: what a card reads freed can fall short of
            // what the holder counted.
            loop {
                let have = free();
                if have >= need {
                    return given;
                }
                if !holder.give_back(kind, card, need - have) {
                    tracing::debug!(
                        "card {card}: {} keeps no {kind:?} it can give",
                        holder.owner()
                    );
                    break;
                }
                given = true;
                super::vram_manager::residency_changed();
                tracing::info!(
                    "card {card}: {} gave back its {kind:?} for {except}, {} MB free of {} MB wanted",
                    holder.owner(),
                    free() >> 20,
                    need >> 20
                );
            }
        }
    }
    given
}

/// The bytes [`make_room`] would have the holders other than `except` on `card` give back for
/// `need` bytes on a card that has `have` free, counted from what each holder keeps, without
/// asking any of them.
/// The kinds are counted in the order [`make_room`] asks for them, each whole.
pub fn would_give(card: usize, need: u64, except: &str, have: u64) -> u64 {
    let mut given = 0u64;
    for kind in KINDS {
        for holder in on_card(card, except) {
            if have.saturating_add(given) >= need {
                return given;
            }
            given = given.saturating_add(holder.keeps(kind, card));
        }
    }
    given
}

/// What a load short of room takes back from the other models on the cards before it places
/// its model.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Reclaimed {
    /// The idle models the load unloads, the least recently used first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<String>,
    /// The bytes each card gets back, cards in ordinal order.
    pub cards: Vec<CardBack>,
}

/// The bytes one card gets back.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CardBack {
    pub index: usize,
    pub bytes: u64,
}

impl Reclaimed {
    /// Adds `bytes` given back on `card`.
    pub fn give(&mut self, card: usize, bytes: u64) {
        if bytes == 0 {
            return;
        }
        match self.cards.iter_mut().find(|c| c.index == card) {
            Some(c) => c.bytes = c.bytes.saturating_add(bytes),
            None => {
                self.cards.push(CardBack { index: card, bytes });
                self.cards.sort_by_key(|c| c.index);
            }
        }
    }

    /// The bytes each card gets back, as `(ordinal, bytes)`.
    pub fn given_back(&self) -> Vec<(usize, u64)> {
        self.cards.iter().map(|c| (c.index, c.bytes)).collect()
    }

    /// Whether the load takes nothing back.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty() && self.cards.is_empty()
    }
}

/// [`make_room`] on `card` for `need` bytes, reading the card as `except` is offered it.
pub fn make_room_for(card: usize, need: u64, except: &str) -> bool {
    make_room(card, need, except, || super::room::offered(card, except))
}

/// Asks every holder other than `except` for everything it keeps, on every card: for a request
/// or a load that ran out of memory once the pools were returned. True when anything was
/// given back.
pub fn give_back_all(except: &str) -> bool {
    let cards: Vec<usize> = super::device_probe::probe_cuda_gpus()
        .iter()
        .map(|g| g.index)
        .collect();
    let mut given = false;
    for card in cards {
        given |= make_room(card, u64::MAX, except, || 0);
    }
    given
}

/// What `card` has free now, the pools handed back.
pub fn free_on_card(card: usize) -> u64 {
    super::vram_manager::free_on(card).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A card no machine has: the holders of other tests never run on it.
    const CARD: usize = usize::MAX;

    /// A holder of `held` bytes of each kind on [`CARD`], which frees them into `free`.
    struct Toy {
        name: &'static str,
        held: Mutex<Vec<(Kind, u64)>>,
        free: Arc<AtomicU64>,
        asked: Mutex<Vec<Kind>>,
    }

    impl Holder for Toy {
        fn owner(&self) -> &str {
            self.name
        }
        fn cards(&self) -> Vec<usize> {
            vec![CARD]
        }
        fn give_back(&self, kind: Kind, _card: usize, _bytes: u64) -> bool {
            self.asked.lock().unwrap().push(kind);
            let mut held = self.held.lock().unwrap();
            let Some(i) = held.iter().position(|(k, _)| *k == kind) else {
                return false;
            };
            let (_, bytes) = held.remove(i);
            self.free.fetch_add(bytes, Ordering::Relaxed);
            true
        }
    }

    fn toy(name: &'static str, free: &Arc<AtomicU64>) -> Arc<dyn Holder> {
        Arc::new(Toy {
            name,
            held: Mutex::new(KINDS.iter().map(|k| (*k, 10)).collect()),
            free: free.clone(),
            asked: Mutex::default(),
        })
    }

    #[test]
    fn the_cheapest_kinds_go_first_and_the_asker_keeps_its_own() {
        let free = Arc::new(AtomicU64::new(0));
        let mine = toy("reclaim-test-mine", &free);
        let other = toy("reclaim-test-other", &free);
        register(&mine);
        register(&other);
        let read = || free.load(Ordering::Relaxed);
        assert!(make_room(CARD, 15, "reclaim-test-mine", read));
        assert_eq!(
            read(),
            20,
            "the pool kinds went, the graphs and states stayed"
        );
        assert!(!make_room(CARD, 15, "reclaim-test-mine", read));
        assert!(make_room(CARD, 55, "reclaim-test-mine", read));
        assert_eq!(read(), 50, "the other holder gave back all it had");
        let same = |a: &Arc<dyn Holder>, b: &Arc<dyn Holder>| {
            std::ptr::addr_eq(Arc::as_ptr(a), Arc::as_ptr(b))
        };
        assert!(on_card(CARD, "reclaim-test-other")
            .iter()
            .any(|h| same(h, &mine)));
        drop(other);
        assert!(on_card(CARD, "reclaim-test-mine").is_empty());
    }

    /// A card no machine has, apart from [`CARD`].
    const COUNTED: usize = usize::MAX - 1;

    /// A holder of `held` bytes of snapshots on [`COUNTED`] that counts them.
    struct Counted {
        name: &'static str,
        held: u64,
        given: Mutex<bool>,
    }

    impl Holder for Counted {
        fn owner(&self) -> &str {
            self.name
        }
        fn cards(&self) -> Vec<usize> {
            vec![COUNTED]
        }
        fn give_back(&self, _kind: Kind, _card: usize, _bytes: u64) -> bool {
            *self.given.lock().unwrap() = true;
            true
        }
        fn keeps(&self, kind: Kind, _card: usize) -> u64 {
            match kind {
                Kind::Snapshots => self.held,
                _ => 0,
            }
        }
    }

    /// The count reads what the other holders keep, stops once the card has the room, and
    /// asks no holder for anything.
    #[test]
    fn the_count_reads_what_the_others_keep_and_takes_nothing() {
        let other = Arc::new(Counted {
            name: "reclaim-count-other",
            held: 30,
            given: Mutex::default(),
        });
        let mine = Arc::new(Counted {
            name: "reclaim-count-mine",
            held: 70,
            given: Mutex::default(),
        });
        let (o, m): (Arc<dyn Holder>, Arc<dyn Holder>) = (other.clone(), mine.clone());
        register(&o);
        register(&m);
        assert_eq!(would_give(COUNTED, 100, "reclaim-count-mine", 50), 30);
        assert_eq!(would_give(COUNTED, 100, "reclaim-count-mine", 100), 0);
        assert!(!*other.given.lock().unwrap());
        assert!(!*mine.given.lock().unwrap());
        let mut r = Reclaimed::default();
        r.give(3, 10);
        r.give(1, 5);
        r.give(3, 2);
        r.give(2, 0);
        assert_eq!(r.given_back(), vec![(1, 5), (3, 12)]);
    }
}
