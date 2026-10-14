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

use super::gguf_file::{open_mapped_unadvised, Content, MappedGguf, TensorInfo, Value};
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
    /// Tensors whose stored values are one less than the values read.
    plus_one: std::collections::HashSet<String>,
    /// Tensors held in another format than the one stored, quantised as they are read: the
    /// format and the stored rows of each leading index the listed rows are taken from.
    held: HashMap<String, (super::GgmlDType, usize)>,
}

/// A tensor a part carries under another name: rows of a stored tensor, listed as the reader
/// asks for it.
pub struct Relisted {
    /// The name the reader asks for.
    pub name: String,
    /// The stored tensor it is in.
    pub source: String,
    /// The row it starts at in each matrix of the stored tensor, and the rows it takes there;
    /// `None` for all of them. A selection in a stack of matrices is held: its rows are not
    /// contiguous in the file.
    pub rows: Option<(usize, usize)>,
    /// Its stored values are one less than the values read.
    pub plus_one: bool,
    /// The format it is held in, when not the stored one.
    pub held: Option<super::GgmlDType>,
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

/// The draft head that sits beside the model at `path`, `<stem>.mtp.gguf` with the stem of a
/// split's parts, if there is one.
pub fn draft_head_beside(path: &Path) -> Option<PathBuf> {
    let stem = match split_name(path) {
        Some((stem, ..)) => stem,
        None => path
            .file_name()?
            .to_str()?
            .strip_suffix(".gguf")?
            .to_string(),
    };
    let head = path.parent()?.join(format!("{stem}.mtp.gguf"));
    head.is_file().then_some(head)
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
        Self::open_with(path, &[])
    }

    /// `open`, then the files of `extra` as parts of their own: tensors the set does not carry,
    /// read beside it under names it does not use. They are not parts of the split the name and
    /// header describe, so they declare no number.
    pub fn open_with<P: AsRef<Path>>(path: P, extra: &[PathBuf]) -> Result<Self> {
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
        for file in extra {
            parts.push(open_mapped_unadvised(file)?);
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
        Ok(Self {
            parts,
            owner,
            plus_one: Default::default(),
            held: Default::default(),
        })
    }

    /// List the tensors of part `part` under the names of `listing`, each over the bytes of its
    /// source: the rows it takes of it, the dimensions of one dropped. The part's own names leave
    /// the set. The bytes stay where they are.
    pub fn relist(&mut self, part: usize, listing: &[Relisted]) -> Result<()> {
        let content = &self.parts[part].content;
        let mut infos = HashMap::new();
        for r in listing {
            let src = content.tensor_infos.get(&r.source).ok_or_else(|| {
                Error::msg(format!(
                    "{}: no tensor {} to list as {}",
                    part + 1,
                    r.source,
                    r.name
                ))
            })?;
            let mut dims: Vec<usize> = src.shape.dims().to_vec();
            let mut offset = src.offset;
            let stored_rows = dims.len().checked_sub(2).map_or(1, |i| dims[i]);
            if let Some((first, rows)) = r.rows {
                let cols = *dims.last().unwrap_or(&1);
                let dtype = src.ggml_dtype;
                let leading: usize = dims[..dims.len().saturating_sub(2)].iter().product();
                if dims.len() < 2
                    || first + rows > stored_rows
                    || !cols.is_multiple_of(dtype.block_size())
                    || (leading > 1 && r.held.is_none())
                {
                    return Err(Error::msg(format!(
                        "{}: rows {first}..{} of {:?} as {}",
                        r.source,
                        first + rows,
                        dims,
                        r.name
                    )));
                }
                offset += (first * cols / dtype.block_size() * dtype.type_size()) as u64;
                let at = dims.len() - 2;
                dims[at] = rows;
            }
            dims.retain(|&d| d != 1);
            if dims.is_empty() {
                dims.push(1);
            }
            infos.insert(
                r.name.clone(),
                TensorInfo {
                    ggml_dtype: src.ggml_dtype,
                    shape: dims.into(),
                    offset,
                },
            );
            if r.plus_one {
                self.plus_one.insert(r.name.clone());
            }
            if let Some(f) = r.held {
                self.held.insert(r.name.clone(), (f, stored_rows));
            }
        }
        self.owner.retain(|_, &mut i| i != part);
        for name in infos.keys() {
            if let Some(prev) = self.owner.insert(name.clone(), part) {
                return Err(Error::msg(format!(
                    "tensor {name} appears in parts {} and {}",
                    prev + 1,
                    part + 1
                )));
            }
        }
        self.parts[part].content.tensor_infos = infos;
        Ok(())
    }

    /// Bytes the set holds: every tensor in the format it is held in.
    pub fn held_bytes(&self) -> u64 {
        self.merged_content()
            .tensor_infos
            .values()
            .map(|t| t.size_in_bytes() as u64)
            .sum()
    }

    /// The format `name` is held in when not the one stored, and the stored rows of each
    /// leading index its rows are taken from.
    pub fn held(&self, name: &str) -> Option<(super::GgmlDType, usize)> {
        self.held.get(name).copied()
    }

    /// Whether the stored values of `name` are one less than the values read.
    pub fn stored_one_less(&self, name: &str) -> bool {
        self.plus_one.contains(name)
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
    /// views takes them from the part it reads. A tensor held in another format than stored is
    /// listed in the held one: what the cards will hold.
    pub fn merged_content(&self) -> Content {
        let first = &self.parts[0].content;
        let tensor_infos = self
            .owner
            .iter()
            .filter_map(|(name, &i)| {
                self.parts[i].content.tensor_infos.get(name).map(|info| {
                    let mut info = info.clone();
                    if let Some(&(f, _)) = self.held.get(name) {
                        info.ggml_dtype = f;
                    }
                    (name.clone(), info)
                })
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
    #[test]
    fn a_draft_head_is_found_by_the_stem_of_the_model_or_of_its_split() {
        let dir = std::env::temp_dir().join(format!("draft-head-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let head = dir.join("model-Q4.mtp.gguf");
        std::fs::write(&head, b"x").unwrap();
        assert_eq!(
            draft_head_beside(&dir.join("model-Q4.gguf")),
            Some(head.clone())
        );
        assert_eq!(
            draft_head_beside(&dir.join("model-Q4-00001-of-00002.gguf")),
            Some(head)
        );
        assert_eq!(draft_head_beside(&dir.join("other.gguf")), None);
        std::fs::remove_dir_all(&dir).ok();
    }

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
