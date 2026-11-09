//! Completing a converted file of a wide sparse model without converting it again.
//!
//! Every tensor and key of an existing GGUF is carried byte for byte.
//! The block draft of a block-scaled checkpoint is added when the file carries none.
//! The whole is laid out for the reads a token makes.
//!
//! The layout, front to back:
//! - the path every token reads: embeddings, head, each layer's attention, shared expert, gate,
//!   norms, engram projections, then the draft's own always-read tensors;
//! - the draft's routed experts, then each layer's, 1 unit per expert: its gate, up and down side
//!   by side, so an expert is 1 contiguous read;
//! - the engram tables, read a row at a time.
//!
//! A layer's experts follow the order its `hot_experts` list ranks them in, the others after it
//! by id.
//! Every tensor starts on the largest block the device reads.
//! Every expert unit and engram table starts on the device's preferred I/O size.
//! A stack of experts is carried as its experts' slices: only their order and placement change.
//!
//! The draft alone is also written as a companion file to keep beside a model that carries none.

use super::block_draft::{dense_names, expert_prefix, BlockDraftMeta};
use super::wide_sparse_names::{expert_unit_name, ARCH, ENGRAM_TABLE, EXPERT_STACKS};
use super::BlockScaledCheckpoint;
use crate::inference::load::block_scaled_convert::{bytes_per, written_format, Formats};
use crate::inference::load::model_dirs::ensure_room;
use crate::tensor::gguf_write::{
    encode_value, planned_len, GgufStreamWriter, Placement, PlannedEntry,
};
use crate::tensor::quantized::gguf_file::{layout, MappedGguf, TensorInfo, Value};
use crate::tensor::quantized::GgmlDType;
use crate::tensor::{Error, Result};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Where tensors start: every tensor on `alignment`, expert units and engram tables on `unit`, a
/// multiple of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub alignment: u64,
    pub unit: u64,
}

/// The `queue` directory of the block device holding `path`, or of the disk when the device is
/// a partition.
#[cfg(target_os = "linux")]
fn queue_dir(path: &Path) -> Option<PathBuf> {
    use std::os::unix::fs::MetadataExt;
    let existing = if path.exists() {
        path.to_path_buf()
    } else {
        path.parent()?.to_path_buf()
    };
    let dev = std::fs::metadata(existing).ok()?.dev();
    let sys = std::fs::canonicalize(format!(
        "/sys/dev/block/{}:{}",
        libc::major(dev),
        libc::minor(dev)
    ))
    .ok()?;
    [sys.join("queue"), sys.parent()?.join("queue")]
        .into_iter()
        .find(|q| q.is_dir())
}

#[cfg(target_os = "linux")]
fn queue_value(queue: &Path, name: &str) -> Option<u64> {
    std::fs::read_to_string(queue.join(name))
        .ok()?
        .trim()
        .parse()
        .ok()
}

impl Layout {
    /// The layout for the devices holding `paths`.
    ///
    /// Tensors start on the largest of the memory page and each device's physical block and
    /// minimum I/O size.
    /// Units start on the largest optimal I/O size a device declares that is a multiple of
    /// that, the tensor alignment otherwise.
    pub fn for_paths(paths: &[&Path]) -> Self {
        let page =
            crate::tensor::mapped::page_size().unwrap_or(layout::DEFAULT_ALIGNMENT as usize) as u64;
        let mut alignment = page.max(layout::DEFAULT_ALIGNMENT);
        let mut optimal = Vec::new();
        #[cfg(target_os = "linux")]
        for q in paths.iter().filter_map(|p| queue_dir(p)) {
            for name in [
                "physical_block_size",
                "minimum_io_size",
                "logical_block_size",
            ] {
                if let Some(v) = queue_value(&q, name).filter(|v| v.is_power_of_two()) {
                    alignment = alignment.max(v);
                }
            }
            if let Some(v) = queue_value(&q, "optimal_io_size").filter(|&v| v > 0) {
                optimal.push(v);
            }
        }
        #[cfg(not(target_os = "linux"))]
        let _ = paths;
        let unit = optimal
            .into_iter()
            .filter(|v| v % alignment == 0)
            .max()
            .unwrap_or(alignment);
        Self { alignment, unit }
    }
}

/// Where one tensor's bytes come from.
enum Bytes {
    /// A source tensor, whole.
    Carried(String),
    /// Expert `e` of a source stack.
    Slice(String, usize),
    /// A checkpoint tensor, decoded and written in the planned format.
    Draft(String),
}

struct Item {
    entry: PlannedEntry,
    /// The boundary this tensor starts on.
    start: u64,
    bytes: Bytes,
}

/// `blk.{layer}.{stack}` for one of the expert stacks, taken apart.
fn stack_of(name: &str) -> Option<(usize, &'static str)> {
    let rest = name.strip_prefix("blk.")?;
    let (l, rest) = rest.split_once('.')?;
    let rest = rest.strip_suffix(".weight").unwrap_or(rest);
    let (stack, _) = EXPERT_STACKS.iter().find(|(s, _)| *s == rest)?;
    Some((l.parse().ok()?, stack))
}

/// `blk.{layer}.{stack}.{e}`, an expert already stored as a unit, taken apart.
fn unit_of(name: &str) -> Option<(usize, &'static str, usize)> {
    let (head, e) = name.rsplit_once('.')?;
    let e = e.parse().ok()?;
    let (l, stack) = stack_of(head)?;
    Some((l, stack, e))
}

fn is_engram_table(name: &str) -> bool {
    name.contains("engram_embd") || name == ENGRAM_TABLE
}

/// A layer's experts in the order its hot list ranks them, the others after it by id.
fn hot_order(md: &HashMap<String, Value>, layer: usize, n: usize) -> Vec<usize> {
    let arch = md
        .get("general.architecture")
        .and_then(|v| v.to_string().ok().cloned())
        .unwrap_or_else(|| ARCH.to_string());
    let listed: Vec<usize> = md
        .get(&format!("{arch}.hot_experts.{layer}"))
        .and_then(|v| v.to_vec().ok())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.to_u32().ok().map(|x| x as usize))
                .collect()
        })
        .unwrap_or_default();
    let mut seen = HashSet::new();
    let mut order: Vec<usize> = listed
        .into_iter()
        .filter(|&e| e < n && seen.insert(e))
        .collect();
    order.extend((0..n).filter(|e| !seen.contains(e)));
    order
}

/// The format the draft's tensors are written in when the caller names none: the one the file
/// already holds its head in.
pub fn always_read_format(src: &MappedGguf) -> Option<GgmlDType> {
    ["output", "output.weight", "token_embd", "token_embd.weight"]
        .iter()
        .find_map(|n| src.content.tensor_infos.get(*n))
        .map(|i| i.ggml_dtype)
        .filter(|d| *d != GgmlDType::F32)
}

/// The format a routed expert of the draft is written in: `always_read` when its rows tile into
/// that format's blocks, F32 otherwise.
///
/// A calibration covers the model's experts only.
fn draft_expert_format(shape: &[usize], always_read: GgmlDType) -> GgmlDType {
    if shape.len() == 2 && shape[1].is_multiple_of(always_read.block_size()) {
        always_read
    } else {
        GgmlDType::F32
    }
}

/// The draft's tensors as the file will carry them: its always-read tensors, then its routed
/// experts, 1 unit per expert.
fn draft_items(
    ckpt: &BlockScaledCheckpoint,
    meta: &BlockDraftMeta,
    format: GgmlDType,
    layout: Layout,
) -> Result<(Vec<Item>, Vec<Item>)> {
    let shape = |name: &str| {
        ckpt.stored_shape(name)
            .ok_or_else(|| Error::msg(format!("checkpoint: no tensor {name}")))
    };
    let item = |name: &str, dtype: GgmlDType, start: u64| -> Result<Item> {
        let dims = shape(name)?;
        let elems: usize = dims.iter().product();
        Ok(Item {
            entry: PlannedEntry {
                name: name.to_string(),
                dims,
                dtype,
                byte_len: bytes_per(dtype, elems),
            },
            start,
            bytes: Bytes::Draft(name.to_string()),
        })
    };
    let dense = dense_names(meta.stages)
        .iter()
        .map(|n| item(n, written_format(&shape(n)?, format), layout.alignment))
        .collect::<Result<Vec<_>>>()?;
    let mut experts = Vec::new();
    for s in 0..meta.stages {
        for e in 0..meta.routed {
            for (i, (_, w)) in EXPERT_STACKS.iter().enumerate() {
                let name = format!("{}.{e}.{w}.weight", expert_prefix(s));
                let dtype = draft_expert_format(&shape(&name)?, format);
                let start = if i == 0 {
                    layout.unit
                } else {
                    layout.alignment
                };
                experts.push(item(&name, dtype, start)?);
            }
        }
    }
    Ok((dense, experts))
}

/// A checkpoint tensor's bytes in `dtype`.
fn draft_bytes(ckpt: &BlockScaledCheckpoint, name: &str, dtype: GgmlDType) -> Result<Vec<u8>> {
    let values = ckpt.stored_values(name)?;
    if dtype == GgmlDType::F32 {
        return Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect());
    }
    crate::tensor::quant_cpu::from_float_bytes(dtype, &values)
}

/// Drops a range of a mapped file from the page cache once it has been read.
fn release_range(src: &MappedGguf, offset: usize, len: usize) {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        crate::tensor::mapped::advise_span(
            src.mmap.as_ptr() as usize + offset,
            len,
            libc::MADV_DONTNEED,
            true,
        );
        // Safety: a valid descriptor and a range inside the file; advice only.
        unsafe {
            libc::posix_fadvise(
                src.file.as_raw_fd(),
                offset as libc::off_t,
                len as libc::off_t,
                libc::POSIX_FADV_DONTNEED,
            );
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (src, offset, len);
}

/// The absolute byte range of a source tensor.
fn range(src: &MappedGguf, name: &str) -> Result<(usize, usize)> {
    let info = src
        .content
        .tensor_infos
        .get(name)
        .ok_or_else(|| Error::msg(format!("source: no tensor {name}")))?;
    Ok((
        (src.content.tensor_data_offset + info.offset) as usize,
        info.size_in_bytes(),
    ))
}

/// The bytes written between 2 page-cache releases: a sixteenth of the memory available now.
fn release_every() -> u64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    (sys.available_memory() / 16).max(1)
}

/// Writes `out`: every tensor and key of `src`, carried byte for byte and laid out by `layout`.
///
/// The block draft of `draft` is added when `src` declares none.
/// The draft's tensors are written in `format`, else in the file's head format.
/// Returns the SHA-256 of the file written.
pub fn complete(
    src: &MappedGguf,
    draft: Option<&BlockScaledCheckpoint>,
    out: &Path,
    layout: Layout,
    format: Option<GgmlDType>,
    progress: &(dyn Fn(usize, usize, &str) + Sync),
) -> Result<String> {
    let md = &src.content.metadata;
    let mut keys: Vec<&String> = md.keys().collect();
    keys.sort();
    let mut metadata: Vec<(String, Value)> = keys
        .into_iter()
        .filter(|k| *k != layout::ALIGNMENT_KEY)
        .map(|k| (k.clone(), md[k].clone()))
        .collect();

    // Source tensors in file order, sorted into the 3 groups.
    let mut names: Vec<&String> = src.content.tensor_infos.keys().collect();
    names.sort_by_key(|n| src.content.tensor_infos[*n].offset);
    let info = |n: &str| &src.content.tensor_infos[n];
    let mut front = Vec::new();
    let mut carried_draft = Vec::new();
    let mut tail = Vec::new();
    // Per layer and stack: the stack's name, or the experts already stored as units.
    let mut stacks: HashMap<(usize, &'static str), String> = HashMap::new();
    let mut units: HashMap<(usize, &'static str, usize), String> = HashMap::new();
    for name in names {
        let i = info(name);
        let carried = |start: u64| Item {
            entry: PlannedEntry {
                name: name.clone(),
                dims: i.shape.dims().to_vec(),
                dtype: i.ggml_dtype,
                byte_len: i.size_in_bytes() as u64,
            },
            start,
            bytes: Bytes::Carried(name.clone()),
        };
        if let Some(key) = stack_of(name).filter(|_| i.shape.dims().len() == 3) {
            stacks.insert(key, name.clone());
        } else if let Some(key) = unit_of(name) {
            units.insert(key, name.clone());
        } else if is_engram_table(name) {
            tail.push(carried(layout.unit));
        } else if name.starts_with("mtp.") {
            carried_draft.push(carried(layout.alignment));
        } else {
            front.push(carried(layout.alignment));
        }
    }

    // The draft, from the checkpoint, when the file declares none.
    let mut draft_dense = Vec::new();
    let mut draft_experts = Vec::new();
    let draft_ckpt = match draft {
        Some(ckpt) if BlockDraftMeta::from_metadata(md)?.is_none() => {
            let meta = BlockDraftMeta::from_config_json(&ckpt.dir)?;
            let format = format
                .or_else(|| always_read_format(src))
                .unwrap_or(Formats::default().always_read);
            (draft_dense, draft_experts) = draft_items(ckpt, &meta, format, layout)?;
            metadata.extend(meta.to_metadata(ARCH));
            Some(ckpt)
        }
        _ => None,
    };

    // Each layer's experts as units, hot ones first.
    let mut layers: Vec<usize> = stacks
        .keys()
        .map(|(l, _)| *l)
        .chain(units.keys().map(|(l, _, _)| *l))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    layers.sort();
    let mut expert_items = Vec::new();
    for layer in layers {
        let count = |stack: &'static str| -> usize {
            stacks
                .get(&(layer, stack))
                .map(|n| info(n).shape.dims()[0])
                .unwrap_or_else(|| {
                    units
                        .keys()
                        .filter(|(l, s, _)| *l == layer && *s == stack)
                        .count()
                })
        };
        let n = count(EXPERT_STACKS[0].0);
        for (stack, _) in EXPERT_STACKS {
            if count(stack) != n {
                return Err(Error::msg(format!(
                    "layer {layer}: {stack} holds {} experts, {} holds {n}",
                    count(stack),
                    EXPERT_STACKS[0].0
                )));
            }
        }
        for e in hot_order(md, layer, n) {
            for (i, (stack, _)) in EXPERT_STACKS.iter().enumerate() {
                let start = if i == 0 {
                    layout.unit
                } else {
                    layout.alignment
                };
                let item = match stacks.get(&(layer, *stack)) {
                    Some(s) => {
                        let si = info(s);
                        Item {
                            entry: PlannedEntry {
                                name: expert_unit_name(layer, stack, e),
                                byte_len: (si.size_in_bytes() / n) as u64,
                                dims: si.shape.dims()[1..].to_vec(),
                                dtype: si.ggml_dtype,
                            },
                            start,
                            bytes: Bytes::Slice(s.clone(), e),
                        }
                    }
                    None => {
                        let u = units.get(&(layer, *stack, e)).ok_or_else(|| {
                            Error::msg(format!("layer {layer}: no {stack} for expert {e}"))
                        })?;
                        let ui = info(u);
                        Item {
                            entry: PlannedEntry {
                                name: u.clone(),
                                dims: ui.shape.dims().to_vec(),
                                dtype: ui.ggml_dtype,
                                byte_len: ui.size_in_bytes() as u64,
                            },
                            start,
                            bytes: Bytes::Carried(u.clone()),
                        }
                    }
                };
                expert_items.push(item);
            }
        }
    }

    let items: Vec<Item> = front
        .into_iter()
        .chain(carried_draft)
        .chain(draft_dense)
        .chain(draft_experts)
        .chain(expert_items)
        .chain(tail)
        .collect();
    let (entries, placement) = planned(&items, layout);
    let mut writer = GgufStreamWriter::create_placed(out, &metadata, &entries, &placement)?;

    let total = items.len();
    let every = release_every();
    let mut since_release = 0u64;
    let mut i = 0;
    while i < total {
        // A run of draft tensors is produced in parallel, a batch at a time, and written in order.
        if let Bytes::Draft(_) = items[i].bytes {
            let ckpt =
                draft_ckpt.ok_or_else(|| Error::msg("draft tensors without a checkpoint"))?;
            let batch = rayon::current_num_threads().max(1);
            let run: Vec<&Item> = items[i..]
                .iter()
                .take_while(|it| matches!(it.bytes, Bytes::Draft(_)))
                .take(batch)
                .collect();
            for (it, bytes) in run.iter().zip(produce(ckpt, &run)) {
                let bytes = bytes?;
                since_release += bytes.len() as u64;
                writer.append(&bytes)?;
                i += 1;
                progress(i, total, &it.entry.name);
            }
        } else {
            let it = &items[i];
            let (offset, len) = match &it.bytes {
                Bytes::Carried(name) => range(src, name)?,
                Bytes::Slice(stack, e) => {
                    let (start, whole) = range(src, stack)?;
                    let one = it.entry.byte_len as usize;
                    if one * info(stack).shape.dims()[0] != whole {
                        return Err(Error::msg(format!(
                            "{stack}: {whole} bytes do not split into experts of {one}"
                        )));
                    }
                    (start + e * one, one)
                }
                Bytes::Draft(_) => unreachable!("handled above"),
            };
            // A tensor larger than the release interval goes out in pieces of it.
            let piece = every as usize;
            if len <= piece {
                writer.append(&src.mmap[offset..offset + len])?;
            } else {
                writer.append_parts((0..len).step_by(piece).map(|at| {
                    let n = piece.min(len - at);
                    Ok(src.mmap[offset + at..offset + at + n].to_vec())
                }))?;
            }
            release_range(src, offset, len);
            since_release += len as u64;
            i += 1;
            progress(i, total, &it.entry.name);
        }
        if since_release >= every {
            writer.release_written()?;
            since_release = 0;
        }
    }
    writer.finish_with_digest()
}

/// The entries and the starts of `items`, in order.
fn planned(items: &[Item], layout: Layout) -> (Vec<PlannedEntry>, Placement) {
    let entries = items
        .iter()
        .map(|it| PlannedEntry {
            name: it.entry.name.clone(),
            dims: it.entry.dims.clone(),
            dtype: it.entry.dtype,
            byte_len: it.entry.byte_len,
        })
        .collect();
    let placement = Placement {
        alignment: layout.alignment,
        starts: items.iter().map(|it| it.start).collect(),
    };
    (entries, placement)
}

/// The bytes of a run of draft tensors, produced in parallel, in the run's order.
fn produce(ckpt: &BlockScaledCheckpoint, run: &[&Item]) -> Vec<Result<Vec<u8>>> {
    run.par_iter()
        .map(|it| match &it.bytes {
            Bytes::Draft(name) => draft_bytes(ckpt, name, it.entry.dtype),
            _ => Err(Error::msg("a draft run holds draft tensors only")),
        })
        .collect()
}

/// The draft's file before a byte of it is written: its keys and its tensors, where each starts.
struct DraftPlan {
    metadata: Vec<(String, Value)>,
    items: Vec<Item>,
    entries: Vec<PlannedEntry>,
    placement: Placement,
}

impl DraftPlan {
    fn new(ckpt: &BlockScaledCheckpoint, layout: Layout, format: GgmlDType) -> Result<Self> {
        let meta = BlockDraftMeta::from_config_json(&ckpt.dir)?;
        let (dense, experts) = draft_items(ckpt, &meta, format, layout)?;
        let mut metadata = vec![(
            "general.architecture".to_string(),
            Value::String(ARCH.to_string()),
        )];
        metadata.extend(meta.to_metadata(ARCH));
        let items: Vec<Item> = dense.into_iter().chain(experts).collect();
        let (entries, placement) = planned(&items, layout);
        Ok(Self {
            metadata,
            items,
            entries,
            placement,
        })
    }
}

/// Whether the checkpoint holds a block draft.
pub fn has_draft(ckpt: &BlockScaledCheckpoint) -> bool {
    ckpt.stored_shape("mtp.0.main_proj.weight").is_some()
}

/// The bytes [`write_draft`] writes for the same arguments.
pub fn draft_bytes_planned(
    ckpt: &BlockScaledCheckpoint,
    out: &Path,
    layout: Layout,
    format: GgmlDType,
) -> Result<u64> {
    let draft = DraftPlan::new(ckpt, layout, format)?;
    planned_len(out, &draft.metadata, &draft.entries, &draft.placement)
}

/// Writes the block draft of the checkpoint alone, as a file of its own to keep beside a model
/// that carries none.
///
/// The file holds the architecture key, the draft's keys and its tensors laid out as
/// [`complete`] lays them.
/// Each tensor is in `format` where its rows tile into that format's blocks.
/// Returns the SHA-256 of the file written.
pub fn write_draft(
    ckpt: &BlockScaledCheckpoint,
    out: &Path,
    layout: Layout,
    format: GgmlDType,
    progress: &(dyn Fn(usize, usize, &str) + Sync),
) -> Result<String> {
    let draft = DraftPlan::new(ckpt, layout, format)?;
    let need = planned_len(out, &draft.metadata, &draft.entries, &draft.placement)?;
    let room = out.parent().unwrap_or(Path::new("."));
    ensure_room("the draft", room, need).map_err(Error::msg)?;
    let DraftPlan {
        metadata,
        items,
        entries,
        placement,
    } = draft;
    let mut writer = GgufStreamWriter::create_placed(out, &metadata, &entries, &placement)?;
    let total = items.len();
    let batch = rayon::current_num_threads().max(1);
    let mut i = 0;
    while i < total {
        let run: Vec<&Item> = items[i..].iter().take(batch).collect();
        for (it, bytes) in run.iter().zip(produce(ckpt, &run)) {
            writer.append(&bytes?)?;
            i += 1;
            progress(i, total, &it.entry.name);
        }
        writer.release_written()?;
    }
    writer.finish_with_digest()
}

/// What [`verify_carried`] compared.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CarryCheck {
    /// Metadata keys of the source, each found with the same value.
    pub keys: usize,
    /// Source tensors found whole under their own name.
    pub whole: usize,
    /// Source stacks found as 1 tensor per expert.
    pub split: usize,
    /// Bytes compared.
    pub bytes: u64,
}

/// Checks that `out` carries every metadata key of `src` with the same value, and every tensor
/// of `src` with the same bytes, type and shape, whatever their offsets.
///
/// A tensor is found under its own name.
/// A stack of experts is found as 1 tensor per expert, `{stack}.{e}`, each the stack's slice.
pub fn verify_carried(
    src: &Path,
    out: &Path,
    progress: &(dyn Fn(usize, usize, &str) + Sync),
) -> Result<CarryCheck> {
    let a = crate::tensor::quantized::gguf_file::open_mapped_unadvised(src)?;
    let b = crate::tensor::quantized::gguf_file::open_mapped_unadvised(out)?;
    let mut check = CarryCheck::default();
    for (k, v) in &a.content.metadata {
        let w = b
            .content
            .metadata
            .get(k)
            .ok_or_else(|| Error::msg(format!("{}: no key {k}", out.display())))?;
        if encode_value(v) != encode_value(w) {
            return Err(Error::msg(format!("{}: key {k} differs", out.display())));
        }
        check.keys += 1;
    }
    let same = |(ao, al): (usize, usize), (bo, bl): (usize, usize), what: &str| -> Result<()> {
        if al != bl {
            return Err(Error::msg(format!("{what}: {al} bytes against {bl}")));
        }
        // Compared in pieces of a few thousand pages across the cores.
        // Each piece is dropped from the page cache after.
        let page = crate::tensor::mapped::page_size().unwrap_or(layout::DEFAULT_ALIGNMENT as usize);
        let piece = page << 12;
        let equal = (0..al)
            .step_by(piece)
            .collect::<Vec<_>>()
            .par_iter()
            .all(|&at| {
                let n = piece.min(al - at);
                let eq = a.mmap[ao + at..ao + at + n] == b.mmap[bo + at..bo + at + n];
                release_range(&a, ao + at, n);
                release_range(&b, bo + at, n);
                eq
            });
        if !equal {
            return Err(Error::msg(format!("{what}: bytes differ")));
        }
        Ok(())
    };
    let mut names: Vec<&String> = a.content.tensor_infos.keys().collect();
    names.sort_by_key(|n| a.content.tensor_infos[*n].offset);
    let total = names.len();
    for (i, name) in names.into_iter().enumerate() {
        let ai = &a.content.tensor_infos[name];
        let describe = |bi: &TensorInfo, dims: &[usize], what: &str| -> Result<()> {
            if bi.ggml_dtype != ai.ggml_dtype || bi.shape.dims() != dims {
                return Err(Error::msg(format!(
                    "{what}: {:?} {:?} against {:?} {:?}",
                    bi.ggml_dtype,
                    bi.shape.dims(),
                    ai.ggml_dtype,
                    dims
                )));
            }
            Ok(())
        };
        match b.content.tensor_infos.get(name) {
            Some(bi) => {
                describe(bi, ai.shape.dims(), name)?;
                same(range(&a, name)?, range(&b, name)?, name)?;
                check.whole += 1;
            }
            None => {
                let dims = ai.shape.dims();
                if dims.len() != 3 {
                    return Err(Error::msg(format!("{}: no tensor {name}", out.display())));
                }
                let (start, whole) = range(&a, name)?;
                let one = whole / dims[0];
                for e in 0..dims[0] {
                    let unit = format!("{name}.{e}");
                    let bi = b.content.tensor_infos.get(&unit).ok_or_else(|| {
                        Error::msg(format!("{}: neither {name} nor {unit}", out.display()))
                    })?;
                    describe(bi, &dims[1..], &unit)?;
                    same((start + e * one, one), range(&b, &unit)?, &unit)?;
                }
                check.split += 1;
            }
        }
        check.bytes += ai.size_in_bytes() as u64;
        progress(i + 1, total, name);
    }
    Ok(check)
}
