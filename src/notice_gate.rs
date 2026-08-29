//! Hold `NOTICE.md` to the measurement instead of to memory.
//!
//! The borrowed-code table is driven by `scripts/provenance/manifest.tsv`, the committed
//! output of the provenance measurement (exact body-line intersection against the reference
//! checkouts). The contract has a deliberate hysteresis:
//!
//!   - a file measured at or above 30% MUST have a row in NOTICE - high enough that only a
//!     real derivation trips it;
//!   - a file measured below 10% MUST NOT have a row - its entry leaves as the debt is paid,
//!     which is how NOTICE shrinks toward empty; a self-declared port below that carries its
//!     attribution inline, at the definition it applies to;
//!   - between the two, judgment: an entry may stay while a rename has moved the number more
//!     than the provenance.
//!
//! Two supporting checks: every CUDA translation unit must appear in the manifest at all
//! (this is where verbatim vendored code arrives, and a file the measurement has never seen
//! is a file nobody looked at - regenerate the manifest when adding one), and every in-repo
//! path NOTICE names must still exist (a rename severs the attribution silently; twenty-five
//! entries had rotted that way once).

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::Path;

    /// Prefixes that mean "a file in this repository". Anything else in backticks is a
    /// reference to another project and is not ours to resolve.
    const OURS: [&str; 5] = ["src/", "inference/", "tensor/", "cuda/", "scripts/"];

    /// A row at or above this share of matched body lines must be recorded.
    const MUST_LIST: f64 = 30.0;
    /// A row below this share must leave - the table only shrinks with the measurement.
    const MUST_LEAVE: f64 = 10.0;

    /// Files an external scanner still attributes, whatever the local measurement says.
    ///
    /// The measurement here compares normalised lines and needs them identical. A scanner that
    /// fingerprints snippets recognises a copy that was reformatted, and the two disagree by a
    /// lot: whisper's decoder measured 7.9% here against 53% there. Letting the local floor
    /// retire those entries would have the gate deleting attributions that are still owed,
    /// which is the opposite of what it exists for.
    fn externally_attested() -> Vec<(String, f64)> {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/provenance/externally_attested.tsv");
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!("the external attestation keeps NOTICE.md honest past the local floor: {path:?}: {e}")
        });
        text.lines()
            .filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
            .filter_map(|l| {
                let mut c = l.split('\t');
                let path = c.next()?.trim().to_string();
                let share = c.nth(1)?.trim().trim_end_matches('%').parse::<f64>().ok()?;
                Some((path, share))
            })
            .collect()
    }

    fn notice() -> String {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("NOTICE.md");
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("NOTICE.md is part of the build's contract: {path:?}: {e}"))
    }

    /// path -> (body-match %, matched body lines), from the committed measurement.
    fn manifest() -> HashMap<String, (f64, usize)> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/provenance/manifest.tsv");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("the provenance manifest drives NOTICE.md: {path:?}: {e}"));
        let mut out = HashMap::new();
        for line in text.lines() {
            let mut cols = line.split('\t');
            let (Some(file), Some(pct), Some(lines)) = (cols.next(), cols.next(), cols.next())
            else {
                panic!("manifest row with fewer than three columns: {line:?}");
            };
            let pct: f64 = pct
                .trim_end_matches('%')
                .parse()
                .unwrap_or_else(|e| panic!("manifest share {pct:?} for {file}: {e}"));
            let lines: usize = lines
                .parse()
                .unwrap_or_else(|e| panic!("manifest line count for {file}: {e}"));
            out.insert(file.to_string(), (pct, lines));
        }
        assert!(
            out.len() > 100,
            "the manifest holds only {} rows - regenerate it, this gate is running blind",
            out.len()
        );
        out
    }

    /// Every backticked span, so a path is recognised wherever it appears - prose, table cell
    /// or list item - rather than only in the shapes that happen to exist today.
    fn quoted(text: &str) -> Vec<&str> {
        let mut out = Vec::new();
        let mut rest = text;
        while let Some(open) = rest.find('`') {
            rest = &rest[open + 1..];
            match rest.find('`') {
                Some(close) => {
                    out.push(&rest[..close]);
                    rest = &rest[close + 1..];
                }
                None => break,
            }
        }
        out
    }

    /// Every `.cu`, `.cuh`, `.h`, `.hpp` in the tree, relative to the crate root.
    fn cuda_units(dir: &Path, root: &Path, out: &mut Vec<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                cuda_units(&path, root, out);
            } else if path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| matches!(e, "cu" | "cuh" | "h" | "hpp"))
            {
                out.push(
                    path.strip_prefix(root)
                        .unwrap_or(&path)
                        .to_string_lossy()
                        .into_owned(),
                );
            }
        }
    }

    /// A CUDA file the measurement has never seen is a CUDA file nobody looked at.
    #[test]
    fn every_cuda_translation_unit_is_in_the_manifest() {
        let manifest = manifest();
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut units = Vec::new();
        cuda_units(&root.join("cuda"), root, &mut units);
        cuda_units(&root.join("src"), root, &mut units);
        assert!(
            units.len() > 40,
            "found only {} CUDA units - this walk is looking in the wrong place",
            units.len()
        );
        let unseen: Vec<&String> = units
            .iter()
            .filter(|u| !manifest.contains_key(*u))
            .collect();
        assert!(
            unseen.is_empty(),
            "CUDA files absent from scripts/provenance/manifest.tsv. This is where verbatim \
             vendored code arrives; measure the tree and regenerate the manifest (a clean \
             file gets a 0% row - the point is that someone measured it):\n  {}",
            unseen
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("\n  ")
        );
    }

    /// The borrowed-code table matches the measurement in both directions.
    #[test]
    fn notice_lists_exactly_what_the_measurement_says_is_borrowed() {
        let text = notice();
        let manifest = manifest();
        let attested = externally_attested();

        // Above the threshold and not recorded: the debt is real and NOTICE must say so.
        let unlisted: Vec<&String> = manifest
            .iter()
            .filter(|(file, (pct, _))| *pct >= MUST_LIST && !text.contains(&format!("`{file}`")))
            .map(|(file, _)| file)
            .collect();
        assert!(
            unlisted.is_empty(),
            "files measured at >= {MUST_LIST}% with no row in NOTICE.md - record them:\n  {}",
            unlisted
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("\n  ")
        );

        // Below the floor and still recorded: the entry must leave with the debt, or the
        // table never shrinks and stops meaning anything.
        // Rows of the table only. The prose names files too - it names this one - and a
        // sentence about the gate is not an attribution the measurement can retire.
        let stale: Vec<String> = text
            .lines()
            .filter(|l| l.starts_with("| `"))
            .flat_map(quoted)
            .filter(|span| OURS.iter().any(|p| span.starts_with(p)) && !span.contains('{'))
            .filter(|span| !attested.iter().any(|(a, _)| a == span))
            .filter(|span| {
                manifest
                    .get(*span)
                    .is_some_and(|(pct, _)| *pct < MUST_LEAVE)
            })
            .map(|s| s.to_string())
            .collect();
        // The other half of the contract: a file the external scan names owes a row, whatever
        // the local measurement makes of it. Without this the table records only what one
        // instrument can see.
        let unrecorded: Vec<&str> = attested
            .iter()
            .map(|(a, _)| a.as_str())
            .filter(|a| !text.contains(&format!("`{a}`")))
            .collect();
        assert!(
            unrecorded.is_empty(),
            "files an external scan attributes with no row in NOTICE.md - the attribution is \
             owed whatever the local measure says:\n  {}",
            unrecorded.join("\n  ")
        );

        assert!(
            stale.is_empty(),
            "NOTICE.md still lists files the measurement puts under {MUST_LEAVE}% - the debt \
             is paid, the entry leaves (inline attribution stays where a port declares \
             itself):\n  {}",
            stale.join("\n  ")
        );
    }

    /// A row that states a share is a row that can go stale - a table updated from memory of
    /// what was rewritten will claim 93% for a file the same manifest puts at 72%, which is
    /// the failure the manifest exists to prevent.
    #[test]
    fn the_shares_notice_md_states_are_the_measured_ones() {
        let text = notice();
        let manifest = manifest();
        let attested = externally_attested();
        let mut wrong = Vec::new();
        let mut checked = 0;

        for line in text.lines() {
            let cells: Vec<&str> = line.split('|').map(str::trim).collect();
            // `| `path` | 72% | 742 | upstream | licence |`
            if cells.len() < 5 || !cells[1].starts_with('`') {
                continue;
            }
            let file = cells[1].trim_matches('`');
            // Two instruments, and a row states the LARGER of what they see. Neither is a
            // bound on the other: the local one compares normalised lines and misses a copy
            // that was reformatted, the external one fingerprints snippets and reads a
            // different denominator. A match seen by either is a match, so the attribution
            // follows whichever saw more. The line count stays the local one, or a dash where
            // there is none to state.
            if let Some((_, external)) = attested.iter().find(|(a, _)| a == file) {
                checked += 1;
                let local = manifest.get(file).map_or(0.0, |(p, _)| *p);
                let owed = external.max(local);
                let said = cells[2].strip_suffix('%').and_then(|p| p.parse::<f64>().ok());
                if said.is_none_or(|v| (v - owed).abs() > 1.0) {
                    wrong.push(format!(
                        "{file}: NOTICE says {}, the larger of the two measurements is {owed:.0}%                          (local {local:.1}%, external {external}%)",
                        cells[2]
                    ));
                }
                continue;
            }
            let Some((pct, lines)) = manifest.get(file) else {
                continue;
            };
            let Some(said_pct) = cells[2]
                .strip_suffix('%')
                .and_then(|p| p.parse::<f64>().ok())
            else {
                continue;
            };
            let Ok(said_lines) = cells[3].parse::<usize>() else {
                continue;
            };
            checked += 1;
            // The table rounds; the manifest does not.
            if (said_pct - pct).abs() > 1.0 || said_lines != *lines {
                wrong.push(format!(
                    "{file}: NOTICE says {said_pct}% / {said_lines} lines, \
                     the measurement says {pct:.1}% / {lines}"
                ));
            }
        }

        // The guard is that every row the table HAS was parsed, not that the table is big.
        // A fixed floor made sense while the debt was large; it turns paying the debt into a
        // failure, which is the one outcome this gate must not punish.
        // What this guard must catch is the parser reading NOTHING while the table still has
        // rows - that is how a gate goes silently inert. It must NOT punish the table for
        // getting short, because getting short is the goal.
        let share_rows = text
            .lines()
            .filter(|l| l.trim_start().starts_with("| `"))
            .filter(|l| l.split('|').count() >= 6)
            .count();
        assert!(
            share_rows == 0 || checked > 0,
            "the table has {share_rows} rows and this gate could read none of them - its \
             shape has changed and the gate is going inert"
        );
        assert!(
            wrong.is_empty(),
            "NOTICE.md states shares the measurement does not. Regenerate the manifest and \
             copy its figures; never edit one without the other:\n  {}",
            wrong.join("\n  ")
        );
    }

    #[test]
    fn every_file_notice_md_attributes_still_exists() {
        let text = notice();
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut missing = Vec::new();
        let mut checked = 0;

        for span in quoted(&text) {
            if !OURS.iter().any(|p| span.starts_with(p)) {
                continue;
            }
            // A brace group names several siblings at once: `quantized/{a.rs,b.rs}`.
            let (base, leaves) = match span.split_once('{') {
                Some((base, tail)) => (base, tail.trim_end_matches('}').split(',').collect()),
                None => (span, vec![""]),
            };
            for leaf in leaves {
                let rel = format!("{base}{leaf}");
                // A path may be written from the crate root or from `src/`; both are how the
                // file already spells them, and both are unambiguous.
                let found = root.join(&rel).exists() || root.join("src").join(&rel).exists();
                checked += 1;
                if !found {
                    missing.push(rel);
                }
            }
        }

        // Same rule as the share gate: the floor is what the file names, not a number chosen
        // when the table was long. Every path an external scan still attributes must be among
        // them, which is what keeps this from going vacuous as the table empties.
        let attested = externally_attested();
        assert!(
            checked >= attested.len(),
            "the path scan checked {checked} paths but {} files are still externally \
             attributed - the convention in this file's header has probably changed, and the \
             gate is going inert",
            attested.len()
        );
        assert!(
            missing.is_empty(),
            "NOTICE.md attributes files that no longer exist. A rename does not remove the \
             obligation, it only hides where it applies - repoint these, do not delete them:\n  {}",
            missing.join("\n  ")
        );
    }
}
