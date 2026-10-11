//! The routing counts of a model's layers as a file: how often each expert of each layer was
//! routed to over the decodes of a run. A load reads one to fill the expert caches with the
//! experts the routing favours before any prompt has routed, and a run writes one with what
//! its decodes counted.

use std::path::Path;

use crate::tensor::{Error, Result};

const MAGIC: &[u8; 4] = b"LKXP";
const VERSION: u32 = 1;

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

    /// The file at `path`, refused when it is not a profile or holds fewer counts than its
    /// header says.
    pub fn read(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| Error::msg(format!("expert profile {}: {e}", path.display())))?;
        let word = |at: usize| -> Result<u32> {
            bytes
                .get(at..at + 4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .ok_or_else(|| Error::msg(format!("expert profile {}: truncated", path.display())))
        };
        if bytes.get(..4) != Some(MAGIC.as_slice()) || word(4)? != VERSION {
            return Err(Error::msg(format!(
                "expert profile {}: not a profile this build reads",
                path.display()
            )));
        }
        let (layers, experts) = (word(8)? as usize, word(12)? as usize);
        let counts = (0..layers * experts)
            .map(|i| word(16 + 4 * i))
            .collect::<Result<Vec<u32>>>()?;
        Ok(Self {
            layers,
            experts,
            counts,
        })
    }

    /// The profile written at `path`, whole or not at all: a reader never sees a half file.
    pub fn write(&self, path: &Path) -> Result<()> {
        let mut bytes = Vec::with_capacity(16 + 4 * self.counts.len());
        bytes.extend_from_slice(MAGIC);
        for word in [VERSION, self.layers as u32, self.experts as u32] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        for c in &self.counts {
            bytes.extend_from_slice(&c.to_le_bytes());
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &bytes)
            .and_then(|()| std::fs::rename(&tmp, path))
            .map_err(|e| Error::msg(format!("expert profile {}: {e}", path.display())))
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
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
