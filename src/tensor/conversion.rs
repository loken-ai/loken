//! The ledger of declared representation changes.
//!
//! A load that widens, requantises or repacks a weight changes what the file holds into
//! something else, and the cost shows up as resident bytes or as numerics, never as an error.
//! Every such change a loader decides on records one entry here: the class of tensor, the
//! format it had and the one it took, the bytes on each side and the reason. The load then
//! closes the ledger, logs one summary line, and compares the resident bytes it measured with
//! what the file plus the declared conversions explain.
//!
//! Invariant: an entry is recorded only for a conversion that was performed, with `before`
//! the bytes of the stored form and `after` the bytes of the form it was turned into.
//! `kept` says whether the result stays resident (adds to the footprint) or is dropped when
//! the load ends (adds to the peak only).

use std::sync::Mutex;

/// One declared change of a tensor's representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conversion {
    /// The kind of tensor: `embedding`, `head`, `attention`, `ffn`, `experts`, `norm`,
    /// `mixer`, `per_layer_table`, `tower`, `kv`, `state`, ...
    pub class: &'static str,
    /// The stored format, as its checkpoint spells it.
    pub from: String,
    /// The format it was turned into.
    pub to: String,
    /// Bytes of the stored form.
    pub before: u64,
    /// Bytes of the converted form.
    pub after: u64,
    /// Whether the converted form stays resident after the load.
    pub kept: bool,
    /// The reason, in a few words.
    pub reason: &'static str,
}

static LEDGER: Mutex<Vec<Conversion>> = Mutex::new(Vec::new());

fn ledger() -> std::sync::MutexGuard<'static, Vec<Conversion>> {
    LEDGER.lock().unwrap_or_else(|e| e.into_inner())
}

/// Record a conversion that was performed.
pub fn record(c: Conversion) {
    ledger().push(c);
}

/// Record a resident conversion between two formats by name.
pub fn record_kept(
    class: &'static str,
    from: impl std::fmt::Display,
    to: impl std::fmt::Display,
    before: u64,
    after: u64,
    reason: &'static str,
) {
    record(Conversion {
        class,
        from: from.to_string(),
        to: to.to_string(),
        before,
        after,
        kept: true,
        reason,
    });
}

/// Record an owned copy of a weight that stays in its stored format: bytes the file's own
/// mapping did not hold in process memory.
pub fn record_copy(
    class: &'static str,
    format: impl std::fmt::Display,
    bytes: u64,
    reason: &'static str,
) {
    record_kept(
        class,
        format.to_string(),
        format.to_string(),
        0,
        bytes,
        reason,
    );
}

/// Record a conversion whose result is dropped when the load ends.
pub fn record_transient(
    class: &'static str,
    from: impl std::fmt::Display,
    to: impl std::fmt::Display,
    before: u64,
    after: u64,
    reason: &'static str,
) {
    record(Conversion {
        class,
        from: from.to_string(),
        to: to.to_string(),
        before,
        after,
        kept: false,
        reason,
    });
}

/// Forget the entries of a previous load.
pub fn open() {
    ledger().clear();
}

/// What the entries recorded since [`open`] add up to.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Summary {
    /// The entries, grouped by (class, from, to, reason) in the order first seen.
    pub groups: Vec<Group>,
    /// Resident bytes the kept conversions add over the stored forms (saturating at zero per entry).
    pub inflation: u64,
    /// Resident bytes the kept conversions remove from the stored forms.
    pub shrink: u64,
    /// Peak-only bytes the transient conversions hold at their largest.
    pub transient_peak: u64,
}

/// Entries of one kind, merged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Group {
    pub class: &'static str,
    pub from: String,
    pub to: String,
    pub reason: &'static str,
    pub kept: bool,
    pub count: u64,
    pub before: u64,
    pub after: u64,
}

/// Summarise a set of entries.
pub fn summarise(entries: &[Conversion]) -> Summary {
    let mut s = Summary::default();
    for e in entries {
        if e.kept {
            s.inflation += e.after.saturating_sub(e.before);
            s.shrink += e.before.saturating_sub(e.after);
        } else {
            s.transient_peak = s.transient_peak.max(e.after);
        }
        match s.groups.iter_mut().find(|g| {
            g.class == e.class
                && g.from == e.from
                && g.to == e.to
                && g.reason == e.reason
                && g.kept == e.kept
        }) {
            Some(g) => {
                g.count += 1;
                g.before += e.before;
                g.after += e.after;
            }
            None => s.groups.push(Group {
                class: e.class,
                from: e.from.clone(),
                to: e.to.clone(),
                reason: e.reason,
                kept: e.kept,
                count: 1,
                before: e.before,
                after: e.after,
            }),
        }
    }
    s
}

/// The entries recorded since [`open`].
pub fn entries() -> Vec<Conversion> {
    ledger().clone()
}

/// The one line a load logs: the net resident change and every group of conversions.
pub fn summary_line(s: &Summary) -> String {
    if s.groups.is_empty() {
        return "conversions: none, resident forms are the stored forms".to_string();
    }
    let list = s
        .groups
        .iter()
        .map(|g| {
            format!(
                "{} {}->{} x{} {}->{} bytes{} ({})",
                g.class,
                g.from,
                g.to,
                g.count,
                g.before,
                g.after,
                if g.kept { "" } else { " transient" },
                g.reason
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    format!(
        "conversions: resident +{} -{} bytes, transient peak {} bytes: {list}",
        s.inflation, s.shrink, s.transient_peak
    )
}

/// How far resident bytes may exceed the file plus the declared inflation before the load
/// warns, as a fraction of the file: allocator rounding, alignment padding, scratch.
pub const UNEXPLAINED_SLACK: f64 = 0.03;

/// Bytes by which `resident` exceeds what `file_bytes` and the declared conversions explain,
/// beyond the slack. Zero when the footprint is explained.
pub fn unexplained(file_bytes: u64, resident: u64, s: &Summary) -> u64 {
    let explained = file_bytes.saturating_sub(s.shrink) + s.inflation;
    let slack = (file_bytes as f64 * UNEXPLAINED_SLACK) as u64;
    resident.saturating_sub(explained + slack)
}

/// Close the ledger at the end of a load: log the summary line, and warn when `resident`
/// (the bytes the load holds on every device) exceeds what the file and the ledger explain.
/// `resident` is `None` when the caller has no measurement.
pub fn close(file_bytes: u64, resident: Option<u64>) -> Summary {
    let s = summarise(&entries());
    tracing::info!("{}", summary_line(&s));
    if let Some(r) = resident {
        let extra = unexplained(file_bytes, r, &s);
        if extra > 0 {
            tracing::warn!(
                "resident {r} bytes exceed the file ({file_bytes}) plus declared conversions by {extra} bytes: an undeclared conversion or a second copy"
            );
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(class: &'static str, before: u64, after: u64, kept: bool) -> Conversion {
        Conversion {
            class,
            from: "q5_0".into(),
            to: "q8_0".into(),
            before,
            after,
            kept,
            reason: "test",
        }
    }

    #[test]
    fn inflation_counts_only_what_stays_resident() {
        let s = summarise(&[
            entry("experts", 100, 160, true),
            entry("head", 10, 40, false),
        ]);
        assert_eq!(s.inflation, 60);
        assert_eq!(s.transient_peak, 40);
        assert_eq!(s.groups.len(), 2);
    }

    #[test]
    fn like_entries_merge_into_one_group() {
        let s = summarise(&[
            entry("experts", 100, 160, true),
            entry("experts", 100, 160, true),
        ]);
        assert_eq!(s.groups.len(), 1);
        assert_eq!(s.groups[0].count, 2);
        assert_eq!(s.groups[0].after, 320);
    }

    #[test]
    fn a_footprint_the_ledger_explains_does_not_warn() {
        let s = summarise(&[entry("experts", 100, 160, true)]);
        assert_eq!(unexplained(1000, 1060, &s), 0);
        assert!(unexplained(1000, 1200, &s) > 0);
    }

    #[test]
    fn an_empty_ledger_says_so() {
        assert!(summary_line(&Summary::default()).contains("none"));
    }
}
