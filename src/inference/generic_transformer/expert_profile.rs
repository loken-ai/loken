//! The routing counts of a model's layers as a file: how often each expert of each layer was
//! routed to over the decodes of a run. A load reads one to fill the expert caches with the
//! experts the routing favours before any prompt has routed, and a run writes one with what
//! its decodes counted.
//!
//! Without a configured path, each model keeps its profile in `profiles/` of the model store,
//! under a name derived from the model id. A profile read at the load seeds the counts the
//! next save writes, so the file accumulates the routing of every run. A count never wraps:
//! when one would pass `u32::MAX`, every count of the layer is halved first, which keeps the
//! ranking and ages the history.

use std::path::{Path, PathBuf};

use crate::inference::cache::container;
use crate::tensor::{Error, Result};

/// Starts a profile written before the shared cache format: the magic, a version word of 1,
/// the layer and expert counts, then the counts.
const LEGACY_MAGIC: &[u8; 4] = b"LKXP";
const VERSION: u16 = 1;

/// Routing counts, `[layers * experts]`, layer by layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertProfile {
    pub layers: usize,
    pub experts: usize,
    pub counts: Vec<u32>,
}

impl ExpertProfile {
    /// The counts of layer `layer`.
    pub fn layer(&self, layer: usize) -> &[u32] {
        &self.counts[layer * self.experts..(layer + 1) * self.experts]
    }

    /// The profile at `path`, `None` when no file is there.
    pub fn read_if_present(path: &Path) -> Result<Option<Self>> {
        if !path.exists() {
            return Ok(None);
        }
        Self::read(path).map(Some)
    }

    /// The file at `path`, refused when it is not a profile or holds fewer counts than its
    /// header says. A profile written before the shared cache format is read as it was.
    pub fn read(path: &Path) -> Result<Self> {
        let refused = |why: &str| Error::msg(format!("expert profile {}: {why}", path.display()));
        let file = std::fs::read(path).map_err(|e| refused(&e.to_string()))?;
        let body = if file.starts_with(LEGACY_MAGIC) {
            let mut body = Vec::with_capacity(file.len());
            body.extend_from_slice(&file[8..]);
            if file.get(4..8) != Some(&1u32.to_le_bytes()[..]) {
                return Err(refused("not a profile this build reads"));
            }
            body
        } else {
            let (version, body) = container::open(container::EXPERT_PROFILE, file)
                .map_err(|_| refused("not a profile this build reads"))?;
            if version != VERSION {
                return Err(refused("not a profile this build reads"));
            }
            body
        };
        let mut r = container::Reader(&body);
        let (layers, experts) = match (r.u32(), r.u32()) {
            (Some(l), Some(e)) => (l as usize, e as usize),
            _ => return Err(refused("truncated")),
        };
        let counts = (0..layers * experts)
            .map(|_| r.u32().ok_or_else(|| refused("truncated")))
            .collect::<Result<Vec<u32>>>()?;
        Ok(Self {
            layers,
            experts,
            counts,
        })
    }

    /// The profile written at `path`, whole or not at all: a reader never sees a half file.
    pub fn write(&self, path: &Path) -> Result<()> {
        let mut w = container::Writer::default();
        w.u32(self.layers as u32);
        w.u32(self.experts as u32);
        for c in &self.counts {
            w.u32(*c);
        }
        let file = container::seal(container::EXPERT_PROFILE, VERSION, &w.0, true);
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)
                .map_err(|e| Error::msg(format!("expert profile {}: {e}", dir.display())))?;
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &file)
            .and_then(|()| std::fs::rename(&tmp, path))
            .map_err(|e| Error::msg(format!("expert profile {}: {e}", path.display())))
    }
}

/// Longest run of the model id kept in a profile's file name.
const NAME_PREFIX_MAX: usize = 96;

/// The file name of the profile of `model_id`: the id with every character outside
/// `[A-Za-z0-9._-]` replaced by `_`, cut to `NAME_PREFIX_MAX` bytes, then a digest of the
/// whole id, so two ids give two names even when their safe forms meet.
pub fn file_name(model_id: &str) -> String {
    use sha2::{Digest, Sha256};
    let safe: String = model_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .take(NAME_PREFIX_MAX)
        .collect();
    let digest = Sha256::digest(model_id.as_bytes());
    let tag: String = digest[..8].iter().map(|b| format!("{b:02x}")).collect();
    format!("{safe}-{tag}.profile")
}

/// Where the profile of `model_id` lives in the model store `store`.
pub fn default_path(store: &Path, model_id: &str) -> PathBuf {
    store.join("profiles").join(file_name(model_id))
}

/// Halve every count of `counts`.
fn halve(counts: &mut [u32]) {
    counts.iter_mut().for_each(|c| *c /= 2);
}

/// One more route to `expert`, halving every count first when it would wrap.
pub fn count_route(counts: &mut [u32], expert: usize) {
    if counts[expert] == u32::MAX {
        halve(counts);
    }
    counts[expert] += 1;
}

/// `extra` added to `counts` expert by expert, halving every count until the sums fit.
pub fn add_counts(counts: &mut [u32], extra: &[u32]) {
    let mut sums: Vec<u64> = counts
        .iter()
        .zip(extra)
        .map(|(&a, &b)| u64::from(a) + u64::from(b))
        .collect();
    while sums.iter().any(|&s| s > u64::from(u32::MAX)) {
        sums.iter_mut().for_each(|s| *s /= 2);
    }
    for (c, s) in counts.iter_mut().zip(sums) {
        *c = s as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A profile read back is the profile written, and a file that is not one is refused.
    #[test]
    fn a_profile_survives_the_file_and_a_stranger_is_refused() {
        let dir = std::env::temp_dir().join(format!("loken-expert-profile-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("profile.bin");
        let profile = ExpertProfile {
            layers: 2,
            experts: 3,
            counts: vec![5, 0, 7, 1, 2, 3],
        };
        profile.write(&path).unwrap();
        assert_eq!(ExpertProfile::read(&path).unwrap(), profile);
        assert_eq!(ExpertProfile::read(&path).unwrap().layer(1), &[1, 2, 3]);
        std::fs::write(&path, b"LKXP\x02\x00\x00\x00").unwrap();
        assert!(ExpertProfile::read(&path).is_err());
        let mut legacy = b"LKXP".to_vec();
        for word in [1u32, 2, 3, 5, 0, 7, 1, 2, 3] {
            legacy.extend_from_slice(&word.to_le_bytes());
        }
        std::fs::write(&path, legacy).unwrap();
        assert_eq!(ExpertProfile::read(&path).unwrap(), profile);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Two ids give two files, and every name is safe in a directory.
    #[test]
    fn each_model_has_its_own_safe_file() {
        let a = file_name("org/name:tag");
        let b = file_name("org_name_tag");
        assert_ne!(a, b);
        assert!(a.starts_with("org_name_tag-") && a.ends_with(".profile"));
        let odd = file_name("../x\\y z\u{e9}:*?\"<>|");
        assert!(odd
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')));
        assert!(!odd.contains('/'));
        assert!(file_name(&"x".repeat(1000)).len() < 255);
        let store = Path::new("store");
        assert_eq!(
            default_path(store, "m"),
            store.join("profiles").join(file_name("m"))
        );
    }

    /// A missing file is no profile and no error; a stranger is still an error.
    #[test]
    fn a_missing_profile_is_skipped() {
        let dir = std::env::temp_dir().join(format!("loken-expert-skip-{}", std::process::id()));
        let path = dir.join("profiles").join("p.profile");
        assert_eq!(ExpertProfile::read_if_present(&path).unwrap(), None);
        let profile = ExpertProfile {
            layers: 1,
            experts: 2,
            counts: vec![3, 4],
        };
        profile.write(&path).unwrap();
        assert_eq!(
            ExpertProfile::read_if_present(&path).unwrap(),
            Some(profile)
        );
        std::fs::write(&path, b"nope").unwrap();
        assert!(ExpertProfile::read_if_present(&path).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A seeded history plus a short run is what the next save writes, and no count wraps.
    #[test]
    fn counts_accumulate_without_wrapping() {
        let mut counts = vec![0u32; 3];
        add_counts(&mut counts, &[100, 0, 40]);
        count_route(&mut counts, 1);
        count_route(&mut counts, 2);
        assert_eq!(counts, vec![100, 1, 41]);
        let mut full = vec![u32::MAX, 10, 3];
        count_route(&mut full, 0);
        assert_eq!(full, vec![u32::MAX / 2 + 1, 5, 1]);
        let mut big = vec![u32::MAX - 1, 8];
        add_counts(&mut big, &[4, 2]);
        assert_eq!(big, vec![((u64::from(u32::MAX) + 3) / 2) as u32, 5]);
    }
}
