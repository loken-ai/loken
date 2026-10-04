//! Completing a converted file without converting it again: every tensor and key of an existing
//! GGUF carried byte for byte, the DSpark draft added from the released checkpoint, and the whole
//! laid out for the reads a token makes.
//!
//! The layout, front to back:
//! - the path every token reads - embeddings, head, each layer's attention, shared expert, gate,
//!   norms, engram projections - then the draft's own always-read tensors;
//! - the draft's routed experts, then each layer's, one unit per expert: its gate, up and down
//!   side by side, so an expert is one contiguous read. A layer's experts follow the order its
//!   `hot_experts` list ranks them in, the others after it by id, so warming a layer's hot set
//!   reads one run of the file;
//! - the engram tables, read a row at a time.
//!
//! Every tensor starts on the largest block the device reads, so no read straddles a block it
//! does not need; every expert unit and engram table starts on the device's preferred I/O size.
//! A stack of experts is carried as its experts' slices: the bytes are the source's, only their
//! order and placement change.

use super::convert::{bytes_per, draft_expert_format, written_format};
use super::dspark::{expert_prefix, tensor_names, DsparkConfig};
use super::safetensors_source::SafeTensorsSource;
use super::source::{expert_unit_name, WeightSource, EXPERT_STACKS};
use crate::tensor::gguf_write::{encode_value, GgufStreamWriter, Placement, PlannedEntry};
use crate::tensor::quantized::gguf_file::{layout, MappedGguf};
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
    /// The layout for the devices holding `paths`: tensors on the largest of the memory page and
    /// each device's physical block and minimum I/O size, units on the largest optimal I/O size a
    /// device declares that is a multiple of that, the tensor alignment otherwise.
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
    /// A checkpoint tensor, dequantised and written in the planned format.
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
    name.contains("engram_embd")
}

/// A layer's experts in the order its hot list ranks them, the others after it by id.
fn hot_order(
    md: &HashMap<String, crate::tensor::quantized::gguf_file::Value>,
    layer: usize,
    n: usize,
) -> Vec<usize> {
    let listed: Vec<usize> = md
        .get(&format!("{}.hot_experts.{layer}", super::ARCH))
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
/// already holds its head in, which the converter writes without calibration.
pub fn always_read_format(src: &MappedGguf) -> Option<GgmlDType> {
    ["output", "output.weight", "token_embd", "token_embd.weight"]
        .iter()
        .find_map(|n| src.content.tensor_infos.get(*n))
        .map(|i| i.ggml_dtype)
        .filter(|d| *d != GgmlDType::F32)
}

/// The draft's tensors as the file will carry them: its always-read tensors and its routed
/// experts, one unit per expert, each in `format` when its rows tile into that format's blocks.
fn draft_items(
    ckpt: &SafeTensorsSource,
    cfg: &DsparkConfig,
    format: GgmlDType,
    layout: Layout,
) -> Result<(Vec<Item>, Vec<Item>)> {
    let shape = |name: &str| {
        ckpt.shape(name)
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
    let dense = tensor_names(cfg.n_stages)
        .iter()
        .map(|n| item(n, written_format(&shape(n)?, format), layout.alignment))
        .collect::<Result<Vec<_>>>()?;
    let mut experts = Vec::new();
    for s in 0..cfg.n_stages {
        for e in 0..cfg.n_routed {
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
fn draft_bytes(ckpt: &SafeTensorsSource, name: &str, dtype: GgmlDType) -> Result<Vec<u8>> {
    let values = ckpt.dense_f32(name)?.flatten_all()?.to_vec1::<f32>()?;
    if dtype == GgmlDType::F32 {
        return Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect());
    }
    crate::tensor::quant_cpu::from_float_bytes(dtype, &values)
}

/// Drop a range of a mapped file from the page cache once it has been read.
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

/// The bytes written between two page-cache releases: a sixteenth of the memory available now,
/// so the copy never holds more than a small share of it.
fn release_every() -> u64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    (sys.available_memory() / 16).max(1)
}

/// Write `out`: every tensor and key of `src`, carried byte for byte and laid out by `layout`,
/// plus the DSpark draft from `draft` - the released checkpoint and its directory - when `src`
/// carries none. The draft's tensors are written in `format`, the file's head format when `None`.
/// Returns the SHA-256 of the file written.
pub fn complete(
    src: &MappedGguf,
    draft: Option<(&SafeTensorsSource, &Path)>,
    out: &Path,
    layout: Layout,
    format: Option<GgmlDType>,
    progress: &(dyn Fn(usize, usize, &str) + Sync),
) -> Result<String> {
    let md = &src.content.metadata;
    let mut keys: Vec<&String> = md.keys().collect();
    keys.sort();
    let mut metadata: Vec<(String, crate::tensor::quantized::gguf_file::Value)> = keys
        .into_iter()
        .filter(|k| *k != layout::ALIGNMENT_KEY)
        .map(|k| (k.clone(), md[k].clone()))
        .collect();

    // Source tensors in file order, sorted into the three groups.
    let mut names: Vec<&String> = src.content.tensor_infos.keys().collect();
    names.sort_by_key(|n| src.content.tensor_infos[*n].offset);
    let info = |n: &str| &src.content.tensor_infos[n];
    let mut front = Vec::new();
    let mut carried_draft = Vec::new();
    let mut tail = Vec::new();
    // Per layer, per stack: the stack's name, or the experts already stored as units.
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

    // The draft, from the checkpoint, when the file carries none.
    let mut draft_dense = Vec::new();
    let mut draft_experts = Vec::new();
    let draft_ckpt = match draft {
        Some((ckpt, dir)) if DsparkConfig::from_metadata(md)?.is_none() => {
            let cfg = DsparkConfig::from_config_json(dir)?;
            let format = format
                .or_else(|| always_read_format(src))
                .unwrap_or(super::convert::Formats::default().always_read);
            (draft_dense, draft_experts) = draft_items(ckpt, &cfg, format, layout)?;
            metadata.extend(cfg.to_metadata());
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
                        let dims = si.shape.dims()[1..].to_vec();
                        Item {
                            entry: PlannedEntry {
                                name: expert_unit_name(layer, stack, e),
                                byte_len: (si.size_in_bytes() / n) as u64,
                                dims,
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
    let entries: Vec<PlannedEntry> = items
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
            let produced: Vec<Result<Vec<u8>>> = run
                .par_iter()
                .map(|it| match &it.bytes {
                    Bytes::Draft(name) => draft_bytes(ckpt, name, it.entry.dtype),
                    _ => unreachable!("the run holds draft tensors only"),
                })
                .collect();
            for (it, bytes) in run.iter().zip(produced) {
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

/// What `verify_carried` compared.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CarryCheck {
    /// Metadata keys of the source, each found with the same value.
    pub keys: usize,
    /// Source tensors found whole under their own name.
    pub whole: usize,
    /// Source stacks found as one tensor per expert.
    pub split: usize,
    /// Bytes compared.
    pub bytes: u64,
}

/// Check that `out` carries every metadata key of `src` with the same value and every tensor of
/// `src` with the same bytes, type and shape, whatever their offsets: a tensor under its own
/// name, or a stack of experts as one tensor per expert, `{stack}.{e}`, each the stack's slice.
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
        // Compared in pieces of a few thousand pages across the cores, each piece dropped from
        // the page cache after.
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
        let describe = |bi: &crate::tensor::quantized::gguf_file::TensorInfo,
                        dims: &[usize],
                        what: &str|
         -> Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
    use crate::tensor::quantized::gguf_file::{open_mapped, Value};
    use safetensors::tensor::TensorView;
    use safetensors::Dtype;

    const STAGES: usize = 2;
    const DRAFT_EXPERTS: usize = 3;

    /// A small released checkpoint: the configuration `DsparkConfig` reads and every draft
    /// tensor, f32, in one shard.
    fn checkpoint(dir: &Path) {
        std::fs::create_dir_all(dir.join("inference")).unwrap();
        std::fs::write(dir.join("inference/config.json"), "{}").unwrap();
        let cfg = serde_json::json!({
            "quantization_config": { "weight_block_size": [32, 32] },
            "text_config": {
                "num_nextn_predict_layers": STAGES, "dspark_block_size": 5,
                "dspark_noise_token_id": 7, "dspark_target_layer_ids": [0, 1],
                "dspark_markov_rank": 4, "dspark_n_routed_experts": DRAFT_EXPERTS,
                "dspark_num_experts_per_tok": 2, "sliding_window": 16, "hidden_size": 8,
                "hc_mult": 2, "num_attention_heads": 2, "head_dim": 4, "qk_rope_head_dim": 2,
                "o_groups": 1, "o_lora_rank": 4, "vocab_size": 12, "swiglu_limit": 10.0,
                "routed_scaling_factor": 1.5, "rms_norm_eps": 1e-6, "rope_theta": 10000,
                "rope_scaling": { "factor": 4, "original_max_position_embeddings": 64 }
            }
        });
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
        let mut names = tensor_names(STAGES);
        for s in 0..STAGES {
            for e in 0..DRAFT_EXPERTS {
                for w in ["w1", "w2", "w3"] {
                    names.push(format!("{}.{e}.{w}.weight", expert_prefix(s)));
                }
            }
        }
        let data: Vec<(String, Vec<u8>)> = names
            .iter()
            .enumerate()
            .map(|(i, n)| {
                let v: Vec<u8> = (0..16)
                    .flat_map(|j| ((i * 16 + j) as f32 * 0.01).to_le_bytes())
                    .collect();
                (n.clone(), v)
            })
            .collect();
        let views: Vec<(String, TensorView)> = data
            .iter()
            .map(|(n, v)| {
                (
                    n.clone(),
                    TensorView::new(Dtype::F32, vec![4, 4], v).unwrap(),
                )
            })
            .collect();
        safetensors::serialize_to_file(views, None, &dir.join("model.safetensors")).unwrap();
    }

    /// A converted file: dense tensors, two layers of expert stacks - one f32, one q8_0 - a hot
    /// list for the first, and an engram table with its scales.
    fn converted(path: &Path) {
        let bytes = |n: usize, seed: usize| -> Vec<u8> {
            (0..n).map(|i| ((i * 31 + seed * 7) % 251) as u8).collect()
        };
        let f32s = |n: usize, seed: usize| -> Vec<u8> {
            (0..n)
                .flat_map(|i| ((i + seed) as f32 * 0.5).to_le_bytes())
                .collect()
        };
        let md = vec![
            (
                "general.architecture".to_string(),
                Value::String(super::super::ARCH.into()),
            ),
            ("deepseek_v41.block_count".to_string(), Value::U32(2)),
            (
                "deepseek_v41.hot_experts.0".to_string(),
                Value::Array(vec![Value::U32(2), Value::U32(0)]),
            ),
            (
                "tokenizer.ggml.tokens".to_string(),
                Value::Array(vec![Value::String("a".into()), Value::String("b".into())]),
            ),
        ];
        let q8 = GgmlDType::Q8_0;
        let q8_len = |rows: usize, cols: usize| rows * cols / q8.block_size() * q8.type_size();
        let mut entries = vec![
            GgufEntry {
                name: "token_embd".into(),
                dims: vec![12, 8],
                dtype: GgmlDType::F32,
                data: f32s(96, 1),
            },
            GgufEntry {
                name: "blk.0.attn_norm".into(),
                dims: vec![8],
                dtype: GgmlDType::F32,
                data: f32s(8, 2),
            },
        ];
        for (i, (stack, _)) in EXPERT_STACKS.iter().enumerate() {
            entries.push(GgufEntry {
                name: format!("blk.0.{stack}"),
                dims: vec![3, 4, 8],
                dtype: GgmlDType::F32,
                data: f32s(96, 10 + i),
            });
        }
        entries.push(GgufEntry {
            name: "blk.1.engram_embd.weight".into(),
            dims: vec![5, 16],
            dtype: GgmlDType::I8,
            data: bytes(80, 3),
        });
        entries.push(GgufEntry {
            name: "blk.1.engram_embd.scale".into(),
            dims: vec![5, 2],
            dtype: GgmlDType::I8,
            data: bytes(10, 4),
        });
        for (i, (stack, _)) in EXPERT_STACKS.iter().enumerate() {
            entries.push(GgufEntry {
                name: format!("blk.1.{stack}"),
                dims: vec![3, 2, 32],
                dtype: q8,
                data: bytes(3 * q8_len(2, 32), 20 + i),
            });
        }
        write_gguf_with_metadata(path, &md, &entries).unwrap();
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("loken-complete-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Where the draft comes from, in order: a companion file beside the model that declares it,
    /// then the model file itself; a companion that has moved is not missed, and with neither the
    /// model has no draft.
    #[test]
    fn the_draft_is_taken_from_a_companion_then_the_model_file() {
        use super::super::dspark::Dspark;
        let dir = scratch("locate");
        let ckpt_dir = dir.join("ckpt");
        checkpoint(&ckpt_dir);
        let ckpt = SafeTensorsSource::open(&ckpt_dir).unwrap();
        let layout = Layout {
            alignment: 64,
            unit: 256,
        };
        let plain = dir.join("models").join("plain");
        std::fs::create_dir_all(plain.parent().unwrap()).unwrap();
        converted(&plain);
        let carrying = dir.join("models").join("carrying");
        let src = open_mapped(&plain).unwrap();
        complete(
            &src,
            Some((&ckpt, ckpt_dir.as_path())),
            &carrying,
            layout,
            Some(GgmlDType::Q8_0),
            &|_, _, _| {},
        )
        .unwrap();
        let locate = |model: &Path| {
            let g = crate::tensor::quantized::gguf_source::SplitGguf::open(model).unwrap();
            Dspark::locate(model, &g).map(|d| d.stages.len())
        };

        // The model file carries the draft, the companion is gone: the model file's.
        let apart = dir.join("apart");
        std::fs::create_dir_all(&apart).unwrap();
        std::fs::copy(&carrying, apart.join("model")).unwrap();
        assert_eq!(locate(&apart.join("model")), Some(STAGES));

        // The model carries none and a companion beside it does: the companion's.
        assert_eq!(locate(&plain), Some(STAGES));

        // The companion moves away: the model has no draft of its own and takes none.
        std::fs::rename(&carrying, dir.join("elsewhere")).unwrap();
        with_no_hub(|| assert_eq!(locate(&plain), None));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Run `f` with the hub cache pointed at nothing, so a test of the other sources cannot be
    /// answered by the released checkpoint on the machine running it.
    fn with_no_hub(f: impl FnOnce()) {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved: Vec<(&str, Option<std::ffi::OsString>)> =
            ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "HOME"]
                .into_iter()
                .map(|k| (k, std::env::var_os(k)))
                .collect();
        let empty = std::env::temp_dir().join(format!("loken-no-hub-{}", std::process::id()));
        std::fs::create_dir_all(&empty).unwrap();
        for k in ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "HOME"] {
            std::env::set_var(k, &empty);
        }
        f();
        for (k, v) in saved {
            match v {
                Some(v) => std::env::set_var(k, v),
                None => std::env::remove_var(k),
            }
        }
    }

    /// A completed file carries every key and every tensor of the one it completes, byte for
    /// byte - the stacks as one tensor per expert - lays each on the boundary asked for, puts a
    /// layer's hot experts first, and carries the draft the reader loads with the checkpoint's
    /// own configuration.
    #[test]
    fn a_completed_file_carries_its_source_and_the_draft() {
        let dir = scratch("carry");
        let ckpt_dir = dir.join("ckpt");
        checkpoint(&ckpt_dir);
        let src_path = dir.join("src.gguf");
        converted(&src_path);
        let out = dir.join("out.gguf");
        let layout = Layout {
            alignment: 64,
            unit: 256,
        };
        let src = open_mapped(&src_path).unwrap();
        let ckpt = SafeTensorsSource::open(&ckpt_dir).unwrap();
        let digest = complete(
            &src,
            Some((&ckpt, ckpt_dir.as_path())),
            &out,
            layout,
            Some(GgmlDType::Q8_0),
            &|_, _, _| {},
        )
        .unwrap();
        let written = std::fs::read(&out).unwrap();
        let want: String = {
            use sha2::Digest;
            sha2::Sha256::digest(&written)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect()
        };
        assert_eq!(digest, want);

        let check = verify_carried(&src_path, &out, &|_, _, _| {}).unwrap();
        assert_eq!(check.keys, src.content.metadata.len());
        assert_eq!(check.split, 6);
        assert_eq!(check.whole + check.split, src.content.tensor_infos.len());

        let g = open_mapped(&out).unwrap();
        let at = |n: &str| g.content.tensor_data_offset + g.content.tensor_infos[n].offset;
        for n in g.content.tensor_infos.keys() {
            assert_eq!(at(n) % layout.alignment, 0, "{n} is not aligned");
        }
        for n in ["blk.0.ffn_gate_exps.1", "blk.1.engram_embd.weight"] {
            assert_eq!(at(n) % layout.unit, 0, "{n} does not start a unit");
        }
        // Hot experts first, in the list's order, then the rest; each unit gate, up, down.
        let order = |a: &str, b: &str| assert!(at(a) < at(b), "{a} is not before {b}");
        order("blk.0.ffn_gate_exps.2", "blk.0.ffn_up_exps.2");
        order("blk.0.ffn_up_exps.2", "blk.0.ffn_down_exps.2");
        order("blk.0.ffn_down_exps.2", "blk.0.ffn_gate_exps.0");
        order("blk.0.ffn_down_exps.0", "blk.0.ffn_gate_exps.1");
        order("token_embd", "mtp.0.attn.wq_a.weight");
        order("mtp.1.ffn.experts.2.w2.weight", "blk.0.ffn_gate_exps.2");
        order("blk.1.ffn_down_exps.2", "blk.1.engram_embd.weight");

        // The draft loads from the file alone, with the configuration the checkpoint gives.
        let cfg = DsparkConfig::from_config_json(&ckpt_dir).unwrap();
        let d = super::super::dspark::Dspark::from_file(&g)
            .unwrap()
            .unwrap();
        assert_eq!(d.cfg, cfg);
        assert_eq!(d.stages.len(), STAGES);
        assert_eq!(d.stages[1].moe.experts.count(), DRAFT_EXPERTS);
        let e = d.stages[1].moe.experts.get(2).unwrap();
        assert_eq!(e.w1.dims(), vec![4, 4]);

        // The layers' experts read back as units, the same bytes the stacks held.
        let loader = g.experts(1, 3).unwrap();
        let unit = loader.build(2).unwrap();
        let (s_off, s_len) = range(&src, "blk.1.ffn_down_exps").unwrap();
        let one = s_len / 3;
        match &unit.w2 {
            crate::inference::offload::projection::Projection::Quant(q) => {
                assert_eq!(
                    &*q.data().unwrap(),
                    &src.mmap[s_off + 2 * one..s_off + 3 * one]
                );
            }
            _ => panic!("a q8_0 unit is read in place"),
        }
        #[cfg(unix)]
        assert!(
            unit.storage_unit().is_some(),
            "a unit's three tensors lie together"
        );

        // A carried byte changed is found.
        let mut bad = written.clone();
        let (o_off, _) = range(&g, "blk.0.ffn_up_exps.1").unwrap();
        bad[o_off] ^= 1;
        let bad_path = dir.join("bad.gguf");
        std::fs::write(&bad_path, &bad).unwrap();
        assert!(verify_carried(&src_path, &bad_path, &|_, _, _| {}).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Completing a completed file carries its units and its draft as they are.
    #[test]
    fn completing_twice_carries_the_units_and_the_draft() {
        let dir = scratch("twice");
        let ckpt_dir = dir.join("ckpt");
        checkpoint(&ckpt_dir);
        let src_path = dir.join("src.gguf");
        converted(&src_path);
        let layout = Layout {
            alignment: 64,
            unit: 128,
        };
        let ckpt = SafeTensorsSource::open(&ckpt_dir).unwrap();
        let once = dir.join("once.gguf");
        let src = open_mapped(&src_path).unwrap();
        let draft = Some((&ckpt, ckpt_dir.as_path()));
        complete(&src, draft, &once, layout, None, &|_, _, _| {}).unwrap();
        let twice = dir.join("twice.gguf");
        let first = open_mapped(&once).unwrap();
        complete(&first, draft, &twice, layout, None, &|_, _, _| {}).unwrap();
        let check = verify_carried(&once, &twice, &|_, _, _| {}).unwrap();
        assert_eq!(check.split, 0);
        assert_eq!(check.whole, first.content.tensor_infos.len());
        let second = open_mapped(&twice).unwrap();
        assert_eq!(
            second.content.tensor_infos.len(),
            first.content.tensor_infos.len()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
