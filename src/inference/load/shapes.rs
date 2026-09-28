//! A checkpoint's shape, read from its headers and never from its weights.
//!
//! A loader builds its config from what the file holds: widths from the tensors that carry
//! them, block counts from the indices the tensors are numbered with, and, in a GGUF, the
//! values its metadata states. Reading a header costs nothing next to the weights, so a
//! planner may ask before anything is loaded.

use crate::tensor::quantized::gguf_file;
use crate::tensor::{Error, Result};
use std::collections::HashMap;

/// Every tensor's shape in a checkpoint, row-major, with a GGUF's metadata beside it.
#[derive(Debug, Clone, Default)]
pub struct Shapes {
    tensors: HashMap<String, Vec<usize>>,
    meta: HashMap<String, gguf_file::Value>,
}

fn is_gguf(path: &std::path::Path) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
}

impl Shapes {
    /// The shapes of one checkpoint, whole or in shards: safetensors headers, or a GGUF's
    /// tensor directory and metadata.
    pub fn read<P: AsRef<std::path::Path>>(paths: &[P]) -> Result<Self> {
        let mut out = Self::default();
        for p in paths {
            let p = p.as_ref();
            if is_gguf(p) {
                let content = gguf_file::open_header(p)?;
                for (name, info) in content.tensor_infos {
                    out.tensors.insert(name, info.shape.dims().to_vec());
                }
                out.meta.extend(content.metadata);
            } else {
                // SAFETY: the map lives only for this read and nothing writes the file.
                let st = unsafe { crate::tensor::safetensors_io::SafeTensorsLoader::multi(&[p])? };
                for name in st.names() {
                    if let Some(dims) = st.shape_of(name) {
                        out.tensors.insert(name.to_string(), dims);
                    }
                }
            }
        }
        if out.tensors.is_empty() {
            return Err(Error("checkpoint header lists no tensor".to_string()));
        }
        Ok(out)
    }

    /// Shapes over names as they are given, for a caller that already holds them.
    pub fn from_dims(tensors: HashMap<String, Vec<usize>>) -> Self {
        Self {
            tensors,
            meta: HashMap::new(),
        }
    }

    /// The same shapes under the names a quantized loader keys them by, wrapper prefixes
    /// such as `model.` stripped.
    pub fn canonical(self) -> Self {
        let tensors = self
            .tensors
            .into_iter()
            .map(|(k, v)| {
                (
                    crate::inference::load::fp8_scaled::canonical_name(&k).to_string(),
                    v,
                )
            })
            .collect();
        Self {
            tensors,
            meta: self.meta,
        }
    }

    /// The shape of `name`; an error names the tensor that is missing.
    pub fn dims(&self, name: &str) -> Result<&[usize]> {
        self.tensors
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| Error(format!("checkpoint has no tensor {name}")))
    }

    /// Extent `axis` of `name`.
    pub fn dim(&self, name: &str, axis: usize) -> Result<usize> {
        let dims = self.dims(name)?;
        dims.get(axis)
            .copied()
            .ok_or_else(|| Error(format!("{name} has {} axes, not {}", dims.len(), axis + 1)))
    }

    /// Whether the checkpoint holds `name`.
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Blocks numbered under `prefix`: one past the highest `i` of any `{prefix}.{i}.` name.
    pub fn blocks(&self, prefix: &str) -> usize {
        let head = format!("{prefix}.");
        self.tensors
            .keys()
            .filter_map(|k| k.strip_prefix(&head))
            .filter_map(|rest| rest.split('.').next()?.parse::<usize>().ok())
            .max()
            .map_or(0, |i| i + 1)
    }

    /// A GGUF metadata value as an integer, keyed under the file's own architecture.
    pub fn meta_usize(&self, key: &str) -> Result<usize> {
        self.meta_value(key)?.to_u64().map(|n| n as usize)
    }

    /// A GGUF metadata array of integers, keyed under the file's own architecture.
    pub fn meta_usizes(&self, key: &str) -> Result<Vec<usize>> {
        self.meta_value(key)?
            .to_vec()?
            .iter()
            .map(|v| v.to_u64().map(|n| n as usize))
            .collect()
    }

    /// A GGUF metadata value as a float, keyed under the file's own architecture.
    pub fn meta_f32(&self, key: &str) -> Result<f32> {
        self.meta_value(key)?.to_f32()
    }

    fn meta_value(&self, key: &str) -> Result<&gguf_file::Value> {
        let arch = self
            .meta
            .get("general.architecture")
            .ok_or_else(|| Error("GGUF names no architecture".to_string()))?
            .to_string()?;
        let full = format!("{arch}.{key}");
        self.meta
            .get(&full)
            .ok_or_else(|| Error(format!("GGUF metadata has no {full}")))
    }
}

/// The `config.json` a diffusers or transformers checkpoint keeps beside its weights, when
/// there is one.
pub fn config_beside(weights: &std::path::Path) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(weights.parent()?.join("config.json")).ok()?;
    serde_json::from_str(&text).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shapes(names: &[(&str, &[usize])]) -> Shapes {
        Shapes::from_dims(
            names
                .iter()
                .map(|(n, d)| (n.to_string(), d.to_vec()))
                .collect(),
        )
    }

    #[test]
    fn blocks_are_counted_from_their_highest_index() {
        let s = shapes(&[
            ("layers.0.q.weight", &[4, 4]),
            ("layers.11.q.weight", &[4, 4]),
            ("layers.3.k.weight", &[4, 4]),
            ("layersx.40.q.weight", &[4, 4]),
            ("norm.weight", &[4]),
        ]);
        assert_eq!(s.blocks("layers"), 12);
        assert_eq!(s.blocks("absent"), 0);
    }

    #[test]
    fn a_missing_tensor_is_named() {
        let s = shapes(&[("a.weight", &[2, 3])]);
        assert_eq!(s.dim("a.weight", 1).unwrap(), 3);
        assert!(s
            .dim("b.weight", 0)
            .unwrap_err()
            .to_string()
            .contains("b.weight"));
        assert!(s.dim("a.weight", 2).is_err());
    }

    #[test]
    fn wrapper_prefixes_are_stripped_for_a_quantized_loader() {
        let s = shapes(&[("model.layers.0.q.weight", &[2, 2])]).canonical();
        assert!(s.contains("layers.0.q.weight"));
    }
}
