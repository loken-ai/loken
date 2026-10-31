# Provenance manifest

`manifest.tsv` holds 1 row per file that shares body lines with the reference checkouts.
It also holds a 0% row for every CUDA translation unit.
Each row has 4 columns: path, body-match percentage, body lines matched, closest upstream file.

`measure.py` generates it by exact-line, body-only intersection.
`LOKEN_PROVENANCE_REFS` names the directory that holds the reference checkouts.
Regenerate the manifest after any de-vendoring batch and commit it with the change.

`externally_attested.tsv` lists files that an external snippet scanner still attributes, whatever the local measurement says.

`src/notice_gate.rs` reads the manifest and holds `NOTICE.md` to it:

- A file at or above `MUST_LIST` has a row in `NOTICE.md`.
- A row of `NOTICE.md` leaves when its file falls below `MUST_LEAVE`, unless `externally_attested.tsv` lists the file.
- Every CUDA file has a row in the manifest.
- Every in-repository path `NOTICE.md` names exists.

`NOTICE.md` shrinks as the measurement does and never ahead of it.
