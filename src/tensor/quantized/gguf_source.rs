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

/// The files after the first of the set `path` opens, for a header that declares `count` parts.
/// Empty for a single file. The count in the name must be the header's, and the set is opened by
/// its first part.
fn sibling_paths(path: &Path, count: usize) -> Result<Vec<PathBuf>> {
    if count <= 1 {
        return Ok(Vec::new());
    }
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
    Ok((2..=count)
        .map(|i| dir.join(format!("{stem}-{i:0width$}-of-{count:0width$}.gguf")))
        .collect())
}

/// The elements every tensor of the set `path` belongs to holds: the model's parameter count. A
/// part lists only the tensors it carries, so the set is counted across its parts. Only the
/// headers are read: no part is mapped, so this is safe to ask of a listing.
pub fn set_parameter_count<P: AsRef<Path>>(path: P) -> Result<u64> {
    let path = path.as_ref();
    let first = super::gguf_file::open_header(path)?;
    let count = first
        .metadata
        .get("split.count")
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(1) as usize;
    let mut total: u64 = first
        .tensor_infos
        .values()
        .map(|t| t.elem_count() as u64)
        .sum();
    for sibling in sibling_paths(path, count)? {
        let part = super::gguf_file::open_header(&sibling)?;
        total += part
            .tensor_infos
            .values()
            .map(|t| t.elem_count() as u64)
            .sum::<u64>();
    }
    Ok(total)
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
        for sibling in sibling_paths(path, count)? {
            parts.push(open_mapped_unadvised(&sibling)?);
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

    /// The set's header as one: the metadata of the first part - a split set writes it once,
    /// there - and the union of every part's tensor directory.
    ///
    /// A tensor keeps the offset it has in its own part, so its *bytes* still have to be read
    /// from `part_of`; this is the header a loader asks *about* the model - its architecture,
    /// its shapes, what each block carries - and it answers for the whole set. `mmap_owner` is
    /// left unset: no single mapping backs this directory, so a reader that wants zero-copy
    /// views takes them from the part it reads.
    pub fn merged_content(&self) -> Content {
        let first = &self.parts[0].content;
        let tensor_infos = self
            .owner
            .iter()
            .filter_map(|(name, &i)| {
                self.parts[i]
                    .content
                    .tensor_infos
                    .get(name)
                    .map(|info| (name.clone(), info.clone()))
            })
            .collect();
        Content {
            magic: first.magic,
            metadata: first.metadata.clone(),
            tensor_infos,
            tensor_data_offset: first.tensor_data_offset,
            mmap_owner: None,
        }
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

    /// A set's header as one: the metadata of the first part - where a split set writes it - and
    /// the union of every part's tensor directory, so a loader asks about the whole model. The
    /// offsets stay per part, so a read still goes to the part that carries the tensor; a reader
    /// that took the bytes from the merged header would read the wrong file's bytes.
    #[test]
    fn a_set_reads_as_one_header() {
        use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
        use crate::tensor::quantized::gguf_file::Value;
        use crate::tensor::quantized::GgmlDType;

        let dir = std::env::temp_dir().join(format!("loken-split-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (one, two) = (
            dir.join("set-00001-of-00002.gguf"),
            dir.join("set-00002-of-00002.gguf"),
        );
        let entry = |name: &str| GgufEntry {
            name: name.to_string(),
            dims: vec![4, 4],
            dtype: GgmlDType::F32,
            data: [0f32.to_le_bytes(); 16].concat(),
        };
        let md1 = vec![
            ("split.count".to_string(), Value::U32(2)),
            ("split.no".to_string(), Value::U32(0)),
            (
                "general.architecture".to_string(),
                Value::String("qwen3moe".into()),
            ),
        ];
        let md2 = vec![
            ("split.count".to_string(), Value::U32(2)),
            ("split.no".to_string(), Value::U32(1)),
        ];
        write_gguf_with_metadata(&one, &md1, &[entry("blk.0.attn_norm.weight")]).unwrap();
        write_gguf_with_metadata(&two, &md2, &[entry("blk.1.attn_norm.weight")]).unwrap();

        let split = SplitGguf::open(&one).unwrap();
        let merged = split.merged_content();
        assert_eq!(
            merged
                .metadata
                .get("general.architecture")
                .and_then(|v| v.to_string().ok())
                .map(std::string::String::as_str),
            Some("qwen3moe"),
            "the metadata is the first part's"
        );
        assert!(merged.tensor_infos.contains_key("blk.0.attn_norm.weight"));
        assert!(
            merged.tensor_infos.contains_key("blk.1.attn_norm.weight"),
            "the directory is the union of the parts"
        );
        assert!(
            merged.mmap_owner.is_none(),
            "no single mapping backs the union"
        );
        // The count is the set's, not the first part's: each part holds one 4x4 tensor.
        assert_eq!(set_parameter_count(&one).unwrap(), 32);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
