//! A caching allocator for the buffers the runtime would otherwise allocate and free one by
//! one.
//!
//! `hipFree` waits for every stream of the card, and a layer's worth of temporaries frees tens
//! of buffers, so a prompt's forward pass spent most of its time with the card idle. A buffer
//! released here goes to a free list keyed by its card, its stream and its size class, and the
//! next allocation of that class on that stream takes it back without calling the runtime.
//!
//! Invariants: a cached buffer is reused only on the stream it was allocated for, so the stream
//! order that protected the old buffer protects the reused one; a buffer is in the live table
//! from its allocation to its release, and a pointer the table does not know is freed outright;
//! the cache of a card holds at most an eighth of the card's memory, and an allocation the
//! runtime refuses empties it and tries once more. The cached bytes count as free memory in
//! `mem_get_info`.

use std::collections::HashMap;
use std::ffi::{c_int, c_void};
use std::sync::{Mutex, MutexGuard, OnceLock};

use super::result::{self, DriverError};
use super::sys::{self, hip};

/// Granularity of a size class: a sixty-fourth of the largest power of two the size reaches,
/// at least this many bytes.
const MIN_QUANTUM: usize = 256;
const CLASS_STEPS: usize = 64;

/// The fraction of a card's memory its cache may hold.
const CACHE_SHARE: usize = 8;

#[derive(Default)]
struct Card {
    cap: usize,
    cached: usize,
}

#[derive(Default)]
struct State {
    /// Size class of every buffer handed out and not yet released.
    live: HashMap<usize, usize>,
    free: HashMap<(c_int, usize, usize), Vec<usize>>,
    cards: HashMap<c_int, Card>,
}

fn state() -> MutexGuard<'static, State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// The size class of a request: the request rounded up to a multiple of its quantum.
fn class_of(bytes: usize) -> usize {
    let top = 1usize << (usize::BITS - 1 - bytes.max(1).leading_zeros());
    let quantum = (top / CLASS_STEPS).max(MIN_QUANTUM);
    bytes.div_ceil(quantum) * quantum
}

/// Frees the cached buffers of `card` (every card when `None`). The card's memory is given back
/// to the runtime; `hipFree` waits for the work still using it.
pub(crate) fn trim(card: Option<c_int>) {
    let blocks: Vec<(c_int, usize)> = {
        let mut s = state();
        let keys: Vec<_> = s
            .free
            .keys()
            .filter(|k| card.is_none_or(|c| k.0 == c))
            .copied()
            .collect();
        let mut out = Vec::new();
        for k in keys {
            for p in s.free.remove(&k).unwrap_or_default() {
                out.push((k.0, p));
            }
        }
        for (c, st) in s.cards.iter_mut() {
            if card.is_none_or(|x| x == *c) {
                st.cached = 0;
            }
        }
        out
    };
    free_blocks(blocks);
}

fn free_blocks(blocks: Vec<(c_int, usize)>) {
    let mut bound: Option<c_int> = None;
    for (card, p) in blocks {
        if bound != Some(card) {
            let _ = unsafe { hip!(hipSetDevice(card)) };
            bound = Some(card);
        }
        let _ = unsafe { result::free_sync(p as sys::CUdeviceptr) };
    }
}

/// The bytes the cache of `card` holds.
pub(crate) fn cached_bytes(card: c_int) -> usize {
    state().cards.get(&card).map_or(0, |c| c.cached)
}

/// Takes a buffer of at least `bytes` for `stream` on `card`, the current device.
pub(crate) fn alloc(
    card: c_int,
    stream: usize,
    bytes: usize,
) -> Result<sys::CUdeviceptr, DriverError> {
    let class = class_of(bytes);
    {
        let mut s = state();
        if let Some(p) = s.free.get_mut(&(card, stream, class)).and_then(Vec::pop) {
            if let Some(c) = s.cards.get_mut(&card) {
                c.cached = c.cached.saturating_sub(class);
            }
            s.live.insert(p, class);
            drop(s);
            crate::tensor::cuda::prequant::forget(p);
            return Ok(p as sys::CUdeviceptr);
        }
        if !s.cards.contains_key(&card) {
            // PLACEMENT-EXEMPT: the substrate sizing its own cache from the card's total.
            let cap = result::mem_get_info().map_or(0, |(_, total)| total / CACHE_SHARE);
            s.cards.insert(card, Card { cap, cached: 0 });
        }
    }
    let ptr = match unsafe { result::malloc_sync(class) } {
        Ok(p) => p,
        Err(_) => {
            trim(Some(card));
            unsafe { result::malloc_sync(class) }?
        }
    };
    state().live.insert(ptr as usize, class);
    crate::tensor::cuda::prequant::forget(ptr as usize);
    Ok(ptr)
}

/// Releases `ptr`: into the cache of `card` for `stream` when it fits, to the runtime otherwise.
pub(crate) fn release(card: c_int, stream: usize, ptr: sys::CUdeviceptr) {
    crate::tensor::cuda::prequant::forget(ptr as usize);
    let overflow = {
        let mut s = state();
        match s.live.remove(&(ptr as usize)) {
            None => Some(ptr as usize),
            Some(class) => {
                let st = s.cards.entry(card).or_default();
                if st.cached + class <= st.cap {
                    st.cached += class;
                    s.free
                        .entry((card, stream, class))
                        .or_default()
                        .push(ptr as usize);
                    None
                } else {
                    Some(ptr as usize)
                }
            }
        }
    };
    if let Some(p) = overflow {
        let _ = unsafe { result::free_sync(p as sys::CUdeviceptr) };
    }
}

/// Frees `ptr` outright, forgetting it when the cache handed it out.
pub(crate) fn free_now(ptr: *mut c_void) {
    crate::tensor::cuda::prequant::forget(ptr as usize);
    state().live.remove(&(ptr as usize));
    let _ = unsafe { hip!(hipFree(ptr)) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_class_covers_its_request_within_two_percent() {
        for bytes in [
            1usize,
            255,
            256,
            257,
            4096,
            1 << 20,
            (1 << 20) + 1,
            57_000_001,
        ] {
            let c = class_of(bytes);
            assert!(c >= bytes);
            assert!(c - bytes <= (bytes / 50).max(MIN_QUANTUM));
        }
    }

    /// A released buffer comes back to the next request of its class on the same stream, and
    /// not to another stream's.
    #[test]
    fn a_released_buffer_is_reused_on_its_own_stream_only() {
        let Ok(ctx) = super::super::CudaContext::new(0) else {
            return;
        };
        let (a, b) = (ctx.new_stream().unwrap(), ctx.new_stream().unwrap());
        let first = unsafe { a.alloc::<u8>(3 << 20) }.unwrap();
        let ptr = first.cu_device_ptr;
        drop(first);
        let other = unsafe { b.alloc::<u8>(3 << 20) }.unwrap();
        assert_ne!(other.cu_device_ptr, ptr);
        let again = unsafe { a.alloc::<u8>(3 << 20) }.unwrap();
        assert_eq!(again.cu_device_ptr, ptr);
    }
}
