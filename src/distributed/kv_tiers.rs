//! Where a KV block lives when it is not on the card, and what has to be true when it comes back.
//!
//! A block evicted from VRAM is not necessarily dead: on a long context, moving it to host RAM
//! and back is cheaper than re-prefilling the tokens it holds, and moving it to a peer is what
//! lets a recovering request find its cache instead of rebuilding it. So eviction becomes a
//! move between tiers - VRAM, host, disk, peer - rather than a discard.
//!
//! The whole idea rests on one property: a block that comes back must be the block that left.
//! Not close, not statistically indistinguishable - identical. A KV cache is read as exact
//! state, and a single flipped byte produces fluent text from a context that never existed,
//! which no coherence check can catch. So every tier round trip is verified against a digest
//! taken before the move, and a mismatch is an error rather than a warning.
//!
//! What this module is: the tier bookkeeping and the integrity rule, with an in-memory backing
//! so both are testable without a card, a disk or a peer. What it is not: the transfers
//! themselves. Those belong to the allocator and the data plane, and wiring them to a tier map
//! that has not been proven exact would be building on the assumption this exists to check.

use std::collections::HashMap;

/// Where a block currently is, cheapest to reach first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// On the card that will read it.
    Vram,
    /// Host memory: a copy over the bus away.
    Host,
    /// Local storage: survives a model unload, costs a read.
    Disk,
    /// Another node's memory: costs a link crossing, and the only tier that can fail because
    /// something else died.
    Peer,
}

/// A block's identity and where it sits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockLocation {
    pub tier: Tier,
    /// Digest of the bytes when they were last written. What makes a bad round trip loud.
    pub digest: u64,
    pub bytes: usize,
}

/// Cheap, deterministic content digest.
///
/// Not cryptographic and not meant to be: it defends against a truncated read, a stale buffer
/// or a partial transfer, not against an adversary. What it must be is exact and stable, so a
/// mismatch always means the bytes changed.
pub fn digest(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

/// What a tier can do, so the pool can be exercised without a card or a network.
pub trait TierStore {
    fn put(&mut self, key: u64, bytes: &[u8]) -> Result<(), String>;
    fn get(&self, key: u64) -> Result<Vec<u8>, String>;
    fn drop_block(&mut self, key: u64);
}

/// A tier backed by memory, for tests and for the host tier itself.
#[derive(Default)]
pub struct MemoryTier {
    blocks: HashMap<u64, Vec<u8>>,
    /// Set to make every read come back wrong, so the integrity check can be shown to fire
    /// rather than assumed to.
    pub corrupt_on_read: bool,
    /// Set to make the tier unreachable, as a dead peer is.
    pub unavailable: bool,
}

impl TierStore for MemoryTier {
    fn put(&mut self, key: u64, bytes: &[u8]) -> Result<(), String> {
        if self.unavailable {
            return Err("tier unavailable".into());
        }
        self.blocks.insert(key, bytes.to_vec());
        Ok(())
    }
    fn get(&self, key: u64) -> Result<Vec<u8>, String> {
        if self.unavailable {
            return Err("tier unavailable".into());
        }
        let mut v = self.blocks.get(&key).cloned().ok_or("no such block")?;
        if self.corrupt_on_read && !v.is_empty() {
            v[0] ^= 0x01;
        }
        Ok(v)
    }
    fn drop_block(&mut self, key: u64) {
        self.blocks.remove(&key);
    }
}

/// Which tier holds what, and the rule that a block never comes back changed.
#[derive(Default)]
pub struct TieredKv {
    where_is: HashMap<u64, BlockLocation>,
}

impl TieredKv {
    /// Record a block as resident on the card.
    pub fn admit(&mut self, key: u64, bytes: &[u8]) {
        self.where_is.insert(
            key,
            BlockLocation {
                tier: Tier::Vram,
                digest: digest(bytes),
                bytes: bytes.len(),
            },
        );
    }

    pub fn locate(&self, key: u64) -> Option<&BlockLocation> {
        self.where_is.get(&key)
    }

    /// Move a block down a tier. The digest recorded at admission travels with it and is not
    /// recomputed here - recomputing would make the check agree with whatever was written,
    /// which is the one thing it must not do.
    pub fn demote(
        &mut self,
        key: u64,
        bytes: &[u8],
        to: Tier,
        store: &mut dyn TierStore,
    ) -> Result<(), String> {
        let loc = self.where_is.get_mut(&key).ok_or("unknown block")?;
        if to <= loc.tier {
            return Err(format!("{to:?} is not below {:?}", loc.tier));
        }
        store.put(key, bytes)?;
        loc.tier = to;
        Ok(())
    }

    /// Bring a block back, and refuse it if it changed on the way.
    pub fn promote(&mut self, key: u64, store: &dyn TierStore) -> Result<Vec<u8>, String> {
        let loc = self.where_is.get(&key).ok_or("unknown block")?.clone();
        let bytes = store.get(key)?;
        let seen = digest(&bytes);
        if seen != loc.digest {
            // Loud, and it must stay loud. A KV block is read as exact state: a flipped byte
            // yields fluent text from a context that never existed, and nothing downstream can
            // tell that apart from a correct answer.
            return Err(format!(
                "block {key} came back changed from {:?}: digest {:#x} != {:#x}",
                loc.tier, seen, loc.digest
            ));
        }
        self.where_is.insert(
            key,
            BlockLocation {
                tier: Tier::Vram,
                digest: loc.digest,
                bytes: bytes.len(),
            },
        );
        Ok(bytes)
    }

    /// Blocks currently below the card, cheapest to reach first - what a rebalancer wants.
    pub fn evicted(&self) -> Vec<(u64, Tier)> {
        let mut v: Vec<(u64, Tier)> = self
            .where_is
            .iter()
            .filter(|(_, l)| l.tier != Tier::Vram)
            .map(|(k, l)| (*k, l.tier))
            .collect();
        v.sort_by_key(|(k, t)| (*t, *k));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(seed: u8) -> Vec<u8> {
        (0..4096u32).map(|i| (i as u8) ^ seed).collect()
    }

    /// The plan's invariant: a block migrated between tiers and read back is bit-exact.
    #[test]
    fn a_block_survives_every_tier_round_trip_bit_exact() {
        for tier in [Tier::Host, Tier::Disk, Tier::Peer] {
            let mut kv = TieredKv::default();
            let mut store = MemoryTier::default();
            let original = block(0x5A);
            kv.admit(1, &original);
            kv.demote(1, &original, tier, &mut store).unwrap();
            assert_eq!(kv.locate(1).unwrap().tier, tier);

            let back = kv.promote(1, &store).unwrap();
            assert_eq!(back, original, "{tier:?} round trip changed the bytes");
            assert_eq!(kv.locate(1).unwrap().tier, Tier::Vram);
        }
    }

    /// And the check has to be able to fire, or it proves nothing. One flipped byte in four
    /// kilobytes must be refused.
    #[test]
    fn a_single_flipped_byte_is_refused() {
        let mut kv = TieredKv::default();
        let mut store = MemoryTier::default();
        let original = block(0x11);
        kv.admit(2, &original);
        kv.demote(2, &original, Tier::Disk, &mut store).unwrap();

        store.corrupt_on_read = true;
        let err = kv.promote(2, &store).unwrap_err();
        assert!(err.contains("came back changed"), "got: {err}");
    }

    /// A tier that cannot answer is an error, not an empty block. Returning nothing would let
    /// a caller continue against a context it no longer has.
    #[test]
    fn an_unreachable_tier_fails_rather_than_returning_nothing() {
        let mut kv = TieredKv::default();
        let mut store = MemoryTier::default();
        let original = block(0x22);
        kv.admit(3, &original);
        kv.demote(3, &original, Tier::Peer, &mut store).unwrap();

        store.unavailable = true; // the peer died
        let err = kv.promote(3, &store).unwrap_err();
        assert!(err.contains("unavailable"), "got: {err}");
        assert_eq!(
            kv.locate(3).unwrap().tier,
            Tier::Peer,
            "still recorded where it was"
        );
    }

    /// Tiers are ordered, and a block only ever moves down by demotion. Allowing a "demote"
    /// upward would make the map disagree with where the bytes actually are.
    #[test]
    fn a_demotion_cannot_move_a_block_upward() {
        let mut kv = TieredKv::default();
        let mut store = MemoryTier::default();
        let original = block(0x33);
        kv.admit(4, &original);
        kv.demote(4, &original, Tier::Disk, &mut store).unwrap();
        let err = kv.demote(4, &original, Tier::Host, &mut store).unwrap_err();
        assert!(err.contains("not below"), "got: {err}");
    }

    /// A rebalancer asks what is off-card and gets it cheapest-first, so it can weigh a host
    /// copy against a peer crossing without re-deriving the order.
    #[test]
    fn evicted_blocks_are_listed_cheapest_tier_first() {
        let mut kv = TieredKv::default();
        let mut store = MemoryTier::default();
        for (k, tier) in [(10u64, Tier::Peer), (11, Tier::Host), (12, Tier::Disk)] {
            let b = block(k as u8);
            kv.admit(k, &b);
            kv.demote(k, &b, tier, &mut store).unwrap();
        }
        kv.admit(13, &block(13)); // stays on the card
        assert_eq!(
            kv.evicted(),
            vec![(11, Tier::Host), (12, Tier::Disk), (10, Tier::Peer)]
        );
    }

    /// The digest travels with the block rather than being recomputed on arrival: recomputing
    /// would make the check agree with whatever was written, which is no check at all.
    #[test]
    fn the_digest_is_taken_before_the_move_not_after() {
        let mut kv = TieredKv::default();
        let mut store = MemoryTier::default();
        let original = block(0x44);
        kv.admit(5, &original);
        let recorded = kv.locate(5).unwrap().digest;
        assert_eq!(recorded, digest(&original));

        // A tier that stores something else entirely must still be caught.
        kv.demote(5, &original, Tier::Host, &mut store).unwrap();
        store.put(5, &block(0x99)).unwrap();
        let err = kv.promote(5, &store).unwrap_err();
        assert!(err.contains("came back changed"), "got: {err}");
    }
}
