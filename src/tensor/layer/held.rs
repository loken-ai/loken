//! The formats the projections built on this thread are held in, while a load collects them.
//!
//! A projection notes its format when it is built.
//! A note outside a collection is dropped.

use std::cell::RefCell;

thread_local! {
    /// The distinct formats noted since the collection on this thread started, in the order
    /// first met. `None` outside a collection.
    static HELD: RefCell<Option<Vec<String>>> = const { RefCell::new(None) };
}

/// A collection running on this thread.
///
/// It restores the previous collection when dropped, also on an unwind.
pub struct Collection {
    prev: Option<Vec<String>>,
}

impl Collection {
    /// The distinct formats noted so far, in the order first met, joined with `+`.
    ///
    /// `None` when none was noted.
    pub fn formats(&self) -> Option<String> {
        HELD.with(|h| {
            h.borrow()
                .as_ref()
                .filter(|f| !f.is_empty())
                .map(|f| f.join("+"))
        })
    }
}

impl Drop for Collection {
    fn drop(&mut self) {
        HELD.with(|h| *h.borrow_mut() = self.prev.take());
    }
}

/// Starts an empty collection on this thread, until the returned guard drops.
#[must_use = "the formats are collected only while the guard is alive"]
pub fn collect() -> Collection {
    Collection {
        prev: HELD.with(|h| h.borrow_mut().replace(Vec::new())),
    }
}

/// Notes the format of a projection built on this thread.
pub fn note(format: &str) {
    HELD.with(|h| {
        if let Some(f) = h.borrow_mut().as_mut() {
            if !f.iter().any(|x| x == format) {
                f.push(format.to_string());
            }
        }
    });
}

/// The place a build starts at in the collection running on this thread.
pub fn mark() -> usize {
    HELD.with(|h| h.borrow().as_ref().map_or(0, Vec::len))
}

/// Drops the formats noted on this thread since `mark`.
///
/// A build that failed and is tried again holds none of the projections it noted.
pub fn rewind(mark: usize) {
    HELD.with(|h| {
        if let Some(f) = h.borrow_mut().as_mut() {
            f.truncate(mark);
        }
    });
}

/// Runs `f` with the collection on this thread suspended.
///
/// The projections a probe builds and drops are not noted.
pub fn aside<T>(f: impl FnOnce() -> T) -> T {
    let held = HELD.with(|h| h.borrow_mut().take());
    let out = f();
    HELD.with(|h| *h.borrow_mut() = held);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_collection_joins_the_distinct_formats_in_the_order_first_met() {
        note("f32");
        let held = collect();
        assert_eq!(held.formats(), None);
        note("f16");
        note("bf16");
        note("f16");
        assert_eq!(held.formats().as_deref(), Some("f16+bf16"));
        aside(|| note("f32"));
        assert_eq!(held.formats().as_deref(), Some("f16+bf16"));
        let start = mark();
        note("f32");
        note("f16");
        rewind(start);
        assert_eq!(held.formats().as_deref(), Some("f16+bf16"));
        note("q8_0");
        assert_eq!(held.formats().as_deref(), Some("f16+bf16+q8_0"));
    }

    #[test]
    fn a_nested_collection_gives_the_outer_one_back() {
        let outer = collect();
        note("f32");
        {
            let inner = collect();
            note("f16");
            assert_eq!(inner.formats().as_deref(), Some("f16"));
        }
        assert_eq!(outer.formats().as_deref(), Some("f32"));
    }
}
