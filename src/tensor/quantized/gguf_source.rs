//! One handle over a model's GGUF file or files.
//!
//! A large model ships split (`gguf-split`): several files, each a complete GGUF whose header
//! lists only the tensors it carries, plus `split.count` and `split.no`. Nothing below the
//! architecture loader should care which part a tensor sits in, so `SplitGguf` presents the set as
//! one source: metadata from the first part, every tensor found in the part that holds it. The two
//! traits are what a loader asks of a source - what the header says, and the tensor bytes - so the
//! same loader reads a single mapped file and a split set.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::gguf_file::{
    advise_if_it_fits, open_mapped_unadvised, Content, MappedGguf, TensorInfo, Value,
};
use super::QTensor;
use crate::tensor::quant_view::gguf_mmap_view;
use crate::tensor::{Device, Error, Result};

/// What a header says: the metadata and the tensor directory.
pub trait GgufMeta {
    fn metadata(&self) -> &HashMap<String, Value>;
    fn info(&self, name: &str) -> Option<&TensorInfo>;
    fn tensor_names(&self) -> Vec<String>;
}

/// A header plus the bytes behind it.
pub trait GgufSource: GgufMeta {
    /// One tensor, pinned by its mapping.
    fn tensor(&self, name: &str, device: &Device) -> Result<QTensor>;
    /// A zero-copy view of one tensor in its mapping, `None` when its dtype has no view path.
    fn mmap_view(&self, name: &str) -> Result<Option<QTensor>>;
    /// Where one tensor's bytes are: the mapping, the offset into it, the length. For a layout
    /// only its own reader understands, which the kernels have no format for.
    fn mapped_range(
        &self,
        name: &str,
    ) -> Result<Option<(std::sync::Arc<memmap2::Mmap>, usize, usize)>>;
    /// The file one tensor's mapping is of, for page-cache advice addressed to the file
    /// rather than to a mapping of it; `None` where the source is not a mapped file.
    fn mapped_file(&self, _name: &str) -> Option<std::sync::Arc<std::fs::File>> {
        None
    }
}

/// The byte range `name` occupies in `mapped`, from the directory the file carries.
fn range_in(
    mapped: &MappedGguf,
    name: &str,
) -> Result<Option<(std::sync::Arc<memmap2::Mmap>, usize, usize)>> {
    let Some(info) = mapped.content.tensor_infos.get(name) else {
        return Ok(None);
    };
    let dtype = info.ggml_dtype;
    let len = info.elem_count() / dtype.block_size() * dtype.type_size();
    let offset = mapped.content.tensor_data_offset as usize + info.offset as usize;
    Ok(Some((mapped.mmap.clone(), offset, len)))
}

impl GgufMeta for Content {
    fn metadata(&self) -> &HashMap<String, Value> {
        &self.metadata
    }
    fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_infos.get(name)
    }
    fn tensor_names(&self) -> Vec<String> {
        self.tensor_infos.keys().cloned().collect()
    }
}

impl GgufMeta for MappedGguf {
    fn metadata(&self) -> &HashMap<String, Value> {
        &self.content.metadata
    }
    fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.content.tensor_infos.get(name)
    }
    fn tensor_names(&self) -> Vec<String> {
        self.content.tensor_infos.keys().cloned().collect()
    }
}

impl GgufSource for MappedGguf {
    fn tensor(&self, name: &str, device: &Device) -> Result<QTensor> {
        MappedGguf::tensor(self, name, device)
    }
    fn mmap_view(&self, name: &str) -> Result<Option<QTensor>> {
        gguf_mmap_view(&self.content, &self.mmap, name)
    }
    fn mapped_range(
        &self,
        name: &str,
    ) -> Result<Option<(std::sync::Arc<memmap2::Mmap>, usize, usize)>> {
        range_in(self, name)
    }
    fn mapped_file(&self, _name: &str) -> Option<std::sync::Arc<std::fs::File>> {
        Some(self.file.clone())
    }
}

/// A split model: every part mapped, and each tensor name resolved to the part that carries it.
pub struct SplitGguf {
    parts: Vec<MappedGguf>,
    owner: HashMap<String, usize>,
}

/// `<stem>-NNNNN-of-MMMMM.gguf` taken apart: the stem, this part's number, the count, the digit
/// width. `None` for a name that is not a split part.
fn split_name(path: &Path) -> Option<(String, usize, usize, usize)> {
    let name = path.file_name()?.to_str()?;
    let stem = name.strip_suffix(".gguf")?;
    let (head, count) = stem.rsplit_once("-of-")?;
    let (stem, no) = head.rsplit_once('-')?;
    if no.is_empty() || no.len() != count.len() || !no.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((
        stem.to_string(),
        no.parse().ok()?,
        count.parse().ok()?,
        no.len(),
    ))
}

fn declared(part: &MappedGguf, key: &str) -> Option<usize> {
    part.content
        .metadata
        .get(key)
        .and_then(|v| v.to_u32().ok())
        .map(|v| v as usize)
}

impl SplitGguf {
    /// Open the file at `path` and, when it is one part of a split, its siblings. A single file is
    /// a set of one. The parts are checked against what they declare: the count in the name must
    /// be the `split.count` in the header, and each part must be the number its name says.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let first = open_mapped_unadvised(path)?;
        let count = declared(&first, "split.count").unwrap_or(1);
        let mut parts = vec![first];
        if count > 1 {
            let (stem, no, named, width) = split_name(path).ok_or_else(|| {
                Error::msg(format!(
                    "{}: header declares split.count {count} but the name is not a split part",
                    path.display()
                ))
            })?;
            if named != count {
                return Err(Error::msg(format!(
                    "{}: name says {named} parts, header says {count}",
                    path.display()
                )));
            }
            if no != 1 {
                return Err(Error::msg(format!(
                    "{}: open a split by its first part",
                    path.display()
                )));
            }
            let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
            for i in 2..=count {
                let p: PathBuf = dir.join(format!("{stem}-{i:0width$}-of-{count:0width$}.gguf"));
                parts.push(open_mapped_unadvised(&p)?);
            }
        }
        for (i, part) in parts.iter().enumerate() {
            let no = declared(part, "split.no").unwrap_or(0);
            if no != i {
                return Err(Error::msg(format!(
                    "split part {}: header says part {no}, expected part {i}",
                    i + 1
                )));
            }
        }
        let mut owner = HashMap::new();
        for (i, part) in parts.iter().enumerate() {
            for name in part.content.tensor_infos.keys() {
                if let Some(prev) = owner.insert(name.clone(), i) {
                    return Err(Error::msg(format!(
                        "tensor {name} appears in parts {} and {}",
                        prev + 1,
                        i + 1
                    )));
                }
            }
        }
        advise_if_it_fits(&parts.iter().map(|p| &*p.mmap).collect::<Vec<_>>());
        Ok(Self { parts, owner })
    }

    pub fn parts(&self) -> &[MappedGguf] {
        &self.parts
    }

    fn part_of(&self, name: &str) -> Result<&MappedGguf> {
        self.owner
            .get(name)
            .map(|&i| &self.parts[i])
            .ok_or_else(|| Error::msg(format!("gguf: no tensor `{name}` in any part")))
    }
}

impl GgufMeta for SplitGguf {
    fn metadata(&self) -> &HashMap<String, Value> {
        &self.parts[0].content.metadata
    }
    fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.owner
            .get(name)
            .and_then(|&i| self.parts[i].content.tensor_infos.get(name))
    }
    fn tensor_names(&self) -> Vec<String> {
        self.owner.keys().cloned().collect()
    }
}

impl GgufSource for SplitGguf {
    fn tensor(&self, name: &str, device: &Device) -> Result<QTensor> {
        self.part_of(name)?.tensor(name, device)
    }
    fn mmap_view(&self, name: &str) -> Result<Option<QTensor>> {
        let p = self.part_of(name)?;
        gguf_mmap_view(&p.content, &p.mmap, name)
    }
    fn mapped_range(
        &self,
        name: &str,
    ) -> Result<Option<(std::sync::Arc<memmap2::Mmap>, usize, usize)>> {
        range_in(self.part_of(name)?, name)
    }
    fn mapped_file(&self, name: &str) -> Option<std::sync::Arc<std::fs::File>> {
        self.part_of(name).ok().map(|p| p.file.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_names_are_taken_apart() {
        let p = Path::new("/m/DSV41-mixedq2-00002-of-00005.gguf");
        assert_eq!(split_name(p), Some(("DSV41-mixedq2".to_string(), 2, 5, 5)));
        assert_eq!(split_name(Path::new("/m/model.gguf")), None);
        assert_eq!(split_name(Path::new("/m/a-of-b.gguf")), None);
    }
}
