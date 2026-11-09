//! Recorded outputs of the retired V4.1 path that the generic route must still give.
//!
//! Each file of `tests/vectors/deepseek_v41_frozen/` holds 1 record a line: a key, a tab, a value.
//! `MANIFEST` names the tests that read each file and the commit whose code wrote it.
//! A value is a digest of bytes with their length, or a small value written as text.
//! A digest compares bit for bit.
//! A value the host kernels give differently at each dispatch level carries the level in its key.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use sha2::{Digest, Sha256};

/// The directory of the record files.
fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/deepseek_v41_frozen")
}

/// The records of `file`, read once a process.
fn table(file: &str) -> Arc<BTreeMap<String, String>> {
    static TABLES: OnceLock<Mutex<HashMap<String, Arc<BTreeMap<String, String>>>>> =
        OnceLock::new();
    let mut tables = TABLES
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    tables
        .entry(file.to_string())
        .or_insert_with(|| {
            let path = dir().join(format!("{file}.txt"));
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let lines = text.lines().filter(|l| !l.is_empty());
            let pairs = lines.map(|l| {
                let (k, v) = l
                    .split_once('\t')
                    .unwrap_or_else(|| panic!("{file}: a line without a tab: {l}"));
                (k.to_string(), v.to_string())
            });
            Arc::new(pairs.collect())
        })
        .clone()
}

/// The recorded value of `key` in `file`.
///
/// A value that depends on the level the host kernels dispatch on is recorded once per level,
/// under the key followed by `@` and the level.
pub(crate) fn recorded(file: &str, key: &str) -> String {
    let t = table(file);
    let level = format!("{key}@{:?}", crate::cpu::cpu_level());
    t.get(key)
        .or_else(|| t.get(&level))
        .unwrap_or_else(|| panic!("{file}: no record {key} nor {level}"))
        .clone()
}

/// Asserts `got` is the recorded value of `key` in `file`.
pub(crate) fn expect(file: &str, key: &str, got: &str) {
    assert_eq!(got, recorded(file, key), "{file}: {key}");
}

/// The length and the SHA-256 of `bytes`, as a record holds them.
pub(crate) fn digest(bytes: &[u8]) -> String {
    let sum: String = Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    format!("{} {sum}", bytes.len())
}

/// [`digest`] of the bits of `values`, little-endian.
pub(crate) fn digest_f32(values: &[f32]) -> String {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    digest(&bytes)
}

/// [`digest`] of the file at `path`.
pub(crate) fn digest_file(path: &std::path::Path) -> String {
    digest(&std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
}

/// Asserts the bytes of `bytes` are the recorded ones.
pub(crate) fn expect_bytes(file: &str, key: &str, bytes: &[u8]) {
    expect(file, key, &digest(bytes));
}

/// Asserts the bits of `values` are the recorded ones.
pub(crate) fn expect_f32(file: &str, key: &str, values: &[f32]) {
    expect(file, key, &digest_f32(values));
}

/// Asserts the value written by `Debug` is the recorded one.
pub(crate) fn expect_debug(file: &str, key: &str, value: &impl std::fmt::Debug) {
    expect(file, key, &format!("{value:?}"));
}
