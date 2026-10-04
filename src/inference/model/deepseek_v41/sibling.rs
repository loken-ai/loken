//! Companion GGUF files beside a model file, found by the metadata keys they declare and never by
//! name: the routing prior a model that carries none borrows, and the draft it does not hold.
//! Scanning reads only each candidate's key table, so a directory of large model files costs a
//! few header reads, not their weights.

use crate::tensor::gguf_write::write_gguf_with_metadata;
use crate::tensor::quantized::gguf_file::{open_mapped_unadvised, Value};
use crate::tensor::{Error, Result};
use std::io::{BufReader, Read, Seek};
use std::path::{Path, PathBuf};

/// The largest key a header can plausibly declare; anything longer is not a GGUF key table.
const MAX_KEY_BYTES: u64 = 1 << 16;

fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn skip<R: Seek>(r: &mut R, bytes: u64) -> std::io::Result<()> {
    r.seek_relative(i64::try_from(bytes).map_err(|_| std::io::ErrorKind::InvalidData)?)
}

/// Step over one metadata value of GGUF type `ty`.
fn skip_value<R: Read + Seek>(r: &mut R, ty: u32) -> std::io::Result<()> {
    match ty {
        0 | 1 | 7 => skip(r, 1),
        2 | 3 => skip(r, 2),
        4..=6 => skip(r, 4),
        10..=12 => skip(r, 8),
        8 => {
            let n = read_u64(r)?;
            skip(r, n)
        }
        9 => {
            let element = read_u32(r)?;
            let count = read_u64(r)?;
            match element {
                8 | 9 => {
                    for _ in 0..count {
                        skip_value(r, element)?;
                    }
                    Ok(())
                }
                0 | 1 | 7 => skip(r, count),
                2 | 3 => skip(r, count * 2),
                4..=6 => skip(r, count * 4),
                10..=12 => skip(r, count * 8),
                _ => Err(std::io::ErrorKind::InvalidData.into()),
            }
        }
        _ => Err(std::io::ErrorKind::InvalidData.into()),
    }
}

/// Whether the GGUF at `path` declares a metadata key starting with `prefix`; false for a file
/// that is not a GGUF or cannot be read.
pub fn declares(path: &Path, prefix: &str) -> bool {
    let scan = || -> std::io::Result<bool> {
        let mut r = BufReader::new(std::fs::File::open(path)?);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            return Ok(false);
        }
        let _version = read_u32(&mut r)?;
        let _tensors = read_u64(&mut r)?;
        let keys = read_u64(&mut r)?;
        for _ in 0..keys {
            let n = read_u64(&mut r)?;
            if n > MAX_KEY_BYTES {
                return Ok(false);
            }
            let mut key = vec![0u8; n as usize];
            r.read_exact(&mut key)?;
            if key.starts_with(prefix.as_bytes()) {
                return Ok(true);
            }
            let ty = read_u32(&mut r)?;
            skip_value(&mut r, ty)?;
        }
        r.stream_position()?;
        Ok(false)
    };
    scan().unwrap_or(false)
}

/// The files in `model`'s directory, other than `model` itself, whose metadata declares a key
/// starting with `prefix`, in name order.
pub fn find_beside(model: &Path, prefix: &str) -> Vec<PathBuf> {
    let Some(dir) = model.parent() else {
        return Vec::new();
    };
    let me = std::fs::canonicalize(model).ok();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .filter(|p| std::fs::canonicalize(p).ok() != me)
        .filter(|p| declares(p, prefix))
        .collect();
    found.sort();
    found
}

fn prior_key(layer: usize) -> String {
    format!("{}.hot_experts.{layer}", super::ARCH)
}

/// The routing prior a companion file beside `model` declares: for each of `n_layers` layers its
/// experts from the most routed-to down, every id below `n_routed`. `None` when no companion
/// declares one that fits this model; a companion that does not fit is passed over.
pub fn prior_beside(model: &Path, n_layers: usize, n_routed: usize) -> Option<Vec<Vec<usize>>> {
    let key = format!("{}.hot_experts.", super::ARCH);
    for path in find_beside(model, &key) {
        let Ok(file) = open_mapped_unadvised(&path) else {
            continue;
        };
        let md = &file.content.metadata;
        let lists: Option<Vec<Vec<usize>>> = (0..n_layers)
            .map(|l| {
                md.get(&prior_key(l))?
                    .to_vec()
                    .ok()?
                    .iter()
                    .map(|v| v.to_u32().ok().map(|x| x as usize))
                    .collect::<Option<Vec<usize>>>()
            })
            .collect();
        match lists {
            Some(lists) if lists.iter().flatten().all(|&e| e < n_routed) => {
                tracing::info!(
                    "routing prior taken from {} ({n_layers} layers)",
                    path.display()
                );
                return Some(lists);
            }
            _ => tracing::warn!(
                "{} declares a routing prior that does not fit this model; passed over",
                path.display()
            ),
        }
    }
    None
}

/// Write a companion file holding only a routing prior: `lists[l]` ranks layer `l`'s experts.
pub fn write_prior(path: &Path, lists: &[Vec<u32>]) -> Result<()> {
    let mut md = vec![(
        "general.architecture".to_string(),
        Value::String(super::ARCH.to_string()),
    )];
    md.extend(super::convert::hot_experts_metadata(lists));
    write_gguf_with_metadata(path, &md, &[]).map_err(|e| Error::msg(format!("{e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("loken-sibling-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn lists(layers: usize, experts: u32) -> Vec<Vec<u32>> {
        (0..layers)
            .map(|l| {
                (0..experts)
                    .rev()
                    .map(|e| (e + l as u32) % experts)
                    .collect()
            })
            .collect()
    }

    /// A companion is found by what its metadata declares: a decoy with other keys, a file that is
    /// no GGUF, and the model itself are passed over, whatever the names.
    #[test]
    fn a_prior_is_found_beside_the_model_by_its_keys() {
        let d = dir("found");
        std::fs::write(d.join("model.bin"), b"not a gguf at all").unwrap();
        std::fs::write(d.join("a-note"), b"GGUF but short").unwrap();
        write_gguf_with_metadata(
            &d.join("decoy"),
            &[("general.architecture".into(), Value::String("other".into()))],
            &[],
        )
        .unwrap();
        write_prior(&d.join("zz-side"), &lists(3, 8)).unwrap();
        let got = prior_beside(&d.join("model.bin"), 3, 8).expect("a prior");
        assert_eq!(got.len(), 3);
        assert_eq!(got[0], vec![7, 6, 5, 4, 3, 2, 1, 0]);
        assert_eq!(got[1], vec![0, 7, 6, 5, 4, 3, 2, 1]);
        // The companion is not its own sibling.
        assert!(prior_beside(&d.join("zz-side"), 3, 8).is_none());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A prior of another shape is passed over silently, and a moved one is simply not there.
    #[test]
    fn a_prior_that_does_not_fit_or_is_gone_gives_none() {
        let d = dir("fit");
        std::fs::write(d.join("model.bin"), b"x").unwrap();
        write_prior(&d.join("side"), &lists(3, 8)).unwrap();
        assert!(
            prior_beside(&d.join("model.bin"), 4, 8).is_none(),
            "layer count"
        );
        assert!(
            prior_beside(&d.join("model.bin"), 3, 6).is_none(),
            "expert count"
        );
        std::fs::remove_file(d.join("side")).unwrap();
        assert!(prior_beside(&d.join("model.bin"), 3, 8).is_none());
        let _ = std::fs::remove_dir_all(&d);
    }
}
