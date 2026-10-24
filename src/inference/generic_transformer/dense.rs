//! A dense safetensors checkpoint presented as the tensors of a GGUF file.
//!
//! The checkpoint's shards are mapped and each tensor is listed under the name the generic
//! loader asks for (`blk.N.attn_q.weight`, `token_embd.weight`) with its stored width: F32, F16
//! or BF16 are read in place, never widened. The loader then builds the model exactly as it
//! builds one from a GGUF file. What differs is carried by the checkpoint itself: its rows are
//! in the order the framework wrote them (see `hf_rows` in the loader), and a long-context rope
//! is declared in the configuration where a GGUF file stores the table, so the table is
//! computed here.
//!
//! The class of the checkpoint names a layout (`dense_layouts.rs`): which stored tensor is which
//! of the loader's, and how the ones that differ in shape or content are made. A tensor the
//! loader reads as stored is listed in place. One it reads otherwise (a fused projection split
//! by head, experts stacked, a norm shifted by one) is an [`Assembly`], built from the stored
//! bytes when the loader asks for it and held nowhere else.

use super::dense_layouts::{layout, Layout, Rule};
use crate::inference::load::awq_loader::{frequency_factors, AwqConfigJson};
use crate::inference::load::hf_config::{hf_class, lookup_b, lookup_u, LayoutKind};
use crate::tensor::quantized::gguf_file::{Content, TensorInfo, Value, VersionedMagic};
use crate::tensor::quantized::GgmlDType;
use crate::tensor::{Error, Result, Shape};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// One mapped shard, its tensors listed under loader names.
pub(super) struct DensePart {
    pub content: Content,
    pub mmap: Arc<memmap2::Mmap>,
}

/// A tensor the loader reads that no stored tensor is as it is: the bytes it is made of and
/// how they are put together.
pub(super) struct Assembly {
    dtype: GgmlDType,
    dims: Vec<usize>,
    /// The stored tensors it is made from: the part holding each, and its listing.
    srcs: Vec<(usize, TensorInfo)>,
    op: Op,
}

enum Op {
    /// Each source is `slabs` matrices of `rows` by `cols`, each transposed when `transpose`
    /// says; the rows in `pick` of every matrix, in order, one source after another.
    Gather {
        slabs: usize,
        rows: usize,
        cols: usize,
        transpose: bool,
        pick: Vec<usize>,
    },
    /// MXFP4 rows made of the blocks (first source) and block scales (second): `slabs`
    /// matrices of `rows` rows of `blocks` blocks, the rows in `pick`.
    Blocks {
        slabs: usize,
        rows: usize,
        blocks: usize,
        pick: Vec<usize>,
    },
    /// Every value through a function, written F32.
    Map(fn(f32) -> f32),
}

/// The shards of a checkpoint, and the header the loader reads the model from.
pub struct DenseCheckpoint {
    pub(super) parts: Vec<DensePart>,
    /// Bytes of tensors the checkpoint does not store but the loader reads, with the part that
    /// lists them.
    pub(super) computed: (Content, Vec<u8>),
    /// The tensors made from stored ones, by the loader's name.
    pub(super) assembled: HashMap<String, Assembly>,
    /// The metadata of `base` and every tensor of every part.
    pub content: Content,
    /// Rotary lanes are paired as a GGUF file pairs them (adjacent), which a vendor's native
    /// layout does; the framework's layout pairs each lane with the one half a head away.
    pub adjacent_rope: bool,
}

/// One tensor of a shard: its name, the shard, stored type, dimensions and start in the data.
struct Listed {
    name: String,
    part: usize,
    dtype: String,
    shape: Vec<usize>,
    start: u64,
}

/// The tensors a layout makes of a listing: those read in place and those assembled.
#[derive(Default)]
struct Translation {
    direct: HashMap<String, (usize, TensorInfo)>,
    assembled: HashMap<String, Assembly>,
}

fn dtype_of(s: &str) -> Option<GgmlDType> {
    Some(match s {
        "F32" => GgmlDType::F32,
        "F16" => GgmlDType::F16,
        "BF16" => GgmlDType::BF16,
        // Raw bytes: the blocks and scales of a quantised stack.
        "U8" => GgmlDType::I8,
        _ => return None,
    })
}

fn float_dtype(l: &Listed) -> Result<GgmlDType> {
    match dtype_of(&l.dtype) {
        Some(d) if d != GgmlDType::I8 => Ok(d),
        _ => Err(Error(format!(
            "{} is {}; dense checkpoints are F32, F16 or BF16",
            l.name, l.dtype
        ))),
    }
}

/// The layout a checkpoint's class names, the vendor's own for a native one.
fn layout_of(cfg: &AwqConfigJson) -> Result<&'static Layout> {
    let kind = if cfg.native {
        LayoutKind::Native
    } else {
        match hf_class(&cfg.hf_arch) {
            Some(c) => c.layout,
            None => {
                return Err(Error(format!(
                    "no tensor layout for class {:?}",
                    cfg.hf_arch
                )))
            }
        }
    };
    Ok(layout(kind))
}

/// Whether `part` is the part of a block `template` names; `{e}` stands for an expert's index,
/// which is returned.
fn matches(template: &str, part: &str) -> Option<Option<usize>> {
    match template.split_once("{e}") {
        None => (template == part).then_some(None),
        Some((pre, post)) => part
            .strip_prefix(pre)?
            .strip_suffix(post)?
            .parse::<usize>()
            .ok()
            .map(Some),
    }
}

/// What the configuration says about the shapes a rule needs.
struct Shapes {
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    experts: usize,
    groups: usize,
    delta: (usize, usize, usize, usize),
    grouped_qkv: bool,
}

impl Shapes {
    fn of(cfg: &AwqConfigJson) -> Self {
        let r = &cfg.root;
        let u = |keys: &[&str]| lookup_u(r, keys);
        Self {
            n_head: cfg.n_head,
            n_kv: cfg.n_kv_head,
            head_dim: cfg.head_dim.unwrap_or(cfg.hidden_size / cfg.n_head.max(1)),
            experts: u(&[
                "num_experts",
                "num_local_experts",
                "n_routed_experts",
                "moe_num_experts",
                "ffn_config.moe_num_experts",
                "moe.num_experts",
            ])
            .unwrap_or(0),
            // ARCH-NAME-EXEMPT: the keys of a configuration that name a state-space model's groups.
            groups: u(&["n_groups", "mamba_n_groups"]).unwrap_or(1),
            delta: (
                u(&["linear_num_key_heads"]).unwrap_or(0),
                u(&["linear_key_head_dim"]).unwrap_or(0),
                u(&["linear_num_value_heads"]).unwrap_or(0),
                u(&["linear_value_head_dim"]).unwrap_or(0),
            ),
            grouped_qkv: lookup_b(r, &["new_decoder_architecture"]) == Some(true)
                || lookup_b(r, &["multi_query"]) == Some(false),
        }
    }
}

/// The rows of a fused projection grouped by key-value head, as query rows, key rows and
/// value rows in turn.
fn grouped_rows(n_head: usize, n_kv: usize, head_dim: usize) -> Vec<usize> {
    let group = n_head / n_kv.max(1);
    let size = group + 2;
    let block = |kv: usize, j: usize| (kv * size + j) * head_dim..(kv * size + j + 1) * head_dim;
    let mut rows = Vec::with_capacity((n_head + 2 * n_kv) * head_dim);
    for kv in 0..n_kv {
        for j in 0..group {
            rows.extend(block(kv, j));
        }
    }
    for j in [group, group + 1] {
        for kv in 0..n_kv {
            rows.extend(block(kv, j));
        }
    }
    rows
}

/// The rows of the delta rule's `qkvz` projection, grouped by key head (its query, key, values
/// and gate in turn): the `qkv` rows of every head, and the gate's.
fn delta_rows(shapes: (usize, usize, usize, usize)) -> (Vec<usize>, Vec<usize>) {
    let (kh, kd, vh, vd) = shapes;
    let per = vh / kh.max(1) * vd;
    let group = 2 * kd + 2 * per;
    let mut qkv = Vec::new();
    for (offset, len) in [(0, kd), (kd, kd), (2 * kd, per)] {
        for g in 0..kh {
            qkv.extend(g * group + offset..g * group + offset + len);
        }
    }
    let z = (0..kh)
        .flat_map(|g| g * group + 2 * kd + per..(g + 1) * group)
        .collect();
    (qkv, z)
}

/// The blocks a stored `_blocks` tensor holds, and the name of its scales.
fn scales_of(name: &str) -> String {
    name.replace("_blocks", "_scales")
}

impl Translation {
    fn put_view(&mut self, name: String, l: &Listed, dims: Vec<usize>) -> Result<()> {
        let dtype = float_dtype(l)?;
        let info = TensorInfo {
            ggml_dtype: dtype,
            shape: Shape::from(dims),
            offset: l.start,
        };
        self.direct.insert(name, (l.part, info));
        Ok(())
    }

    fn put_assembly(&mut self, name: String, a: Assembly) {
        self.assembled.insert(name, a);
    }

    fn source(l: &Listed) -> Result<(usize, TensorInfo)> {
        Ok((
            l.part,
            TensorInfo {
                ggml_dtype: dtype_of(&l.dtype).ok_or_else(|| {
                    Error(format!("{} is {}, which no rule reads", l.name, l.dtype))
                })?,
                shape: Shape::from(l.shape.clone()),
                offset: l.start,
            },
        ))
    }

    /// `rule` applied to the stored tensor `l`, its results named `name(tail)`.
    fn apply(
        &mut self,
        rule: Rule,
        l: &Listed,
        by_name: &HashMap<&str, &Listed>,
        shapes: &Shapes,
        name: &dyn Fn(&str) -> String,
    ) -> Result<()> {
        let dims = &l.shape;
        let cols = if dims.len() >= 2 {
            dims[dims.len() - 1]
        } else {
            1
        };
        let bad = |what: &str| {
            Error(format!(
                "{}: stored {:?} is not {what} of the checkpoint's configuration",
                l.name, l.shape
            ))
        };
        let gather = |dtype: GgmlDType, out: Vec<usize>, op: Op| Assembly {
            dtype,
            dims: out,
            srcs: vec![Self::source(l).expect("a gathered source has a dtype")],
            op,
        };
        match rule {
            Rule::Same(t) => self.put_view(name(t), l, dims.clone())?,
            Rule::PlusOne(t) => {
                float_dtype(l)?;
                self.put_assembly(
                    name(t),
                    gather(GgmlDType::F32, dims.clone(), Op::Map(|x| x + 1.0)),
                );
            }
            Rule::NegExp(t) | Rule::NegExpColumn(t) => {
                float_dtype(l)?;
                let mut out = dims.clone();
                if matches!(rule, Rule::NegExpColumn(_)) {
                    out.push(1);
                }
                self.put_assembly(name(t), gather(GgmlDType::F32, out, Op::Map(|x| -x.exp())));
            }
            Rule::Squeeze(t) => {
                if dims.len() != 3 || dims[1] != 1 {
                    return Err(bad("a stack of one-wide filters"));
                }
                self.put_view(name(t), l, vec![dims[0], dims[2]])?;
            }
            Rule::Flatten(t) => {
                if dims.len() != 2 || dims[0] != 1 {
                    return Err(bad("a single row"));
                }
                self.put_view(name(t), l, vec![dims[1]])?;
            }
            Rule::Column(t) => self.put_view(name(t), l, vec![dims[0], 1])?,
            Rule::Groups(t) => {
                let g = shapes.groups.max(1);
                if dims.len() != 1 || !dims[0].is_multiple_of(g) {
                    return Err(bad("a vector of whole groups"));
                }
                self.put_view(name(t), l, vec![g, dims[0] / g])?;
            }
            Rule::Experts(t) => {
                let e = shapes.experts;
                if e == 0 || dims.len() != 2 || !dims[0].is_multiple_of(e) {
                    return Err(bad("a flat stack of experts"));
                }
                self.put_view(name(t), l, vec![e, dims[0] / e, dims[1]])?;
            }
            Rule::Qkv(t) | Rule::Grouped(t) => {
                let grouped = matches!(rule, Rule::Grouped(_)) || shapes.grouped_qkv;
                if !grouped {
                    return self.put_view(name(t), l, dims.clone());
                }
                let want = (shapes.n_head + 2 * shapes.n_kv) * shapes.head_dim;
                if dims[0] != want {
                    return Err(bad("a fused projection of the heads"));
                }
                let pick = grouped_rows(shapes.n_head, shapes.n_kv, shapes.head_dim);
                let op = Op::Gather {
                    slabs: 1,
                    rows: dims[0],
                    cols,
                    transpose: false,
                    pick,
                };
                self.put_assembly(name(t), gather(float_dtype(l)?, dims.clone(), op));
            }
            Rule::Qkvz => {
                let (kh, kd, vh, vd) = shapes.delta;
                let (qkv, z) = delta_rows(shapes.delta);
                if dims[0] != 2 * kh * kd + 2 * vh * vd {
                    return Err(bad("a fused projection of the delta rule"));
                }
                for (t, pick) in [("attn_qkv.weight", qkv), ("attn_gate.weight", z)] {
                    let mut out = dims.clone();
                    out[0] = pick.len();
                    let op = Op::Gather {
                        slabs: 1,
                        rows: dims[0],
                        cols,
                        transpose: false,
                        pick,
                    };
                    self.put_assembly(name(t), gather(float_dtype(l)?, out, op));
                }
            }
            Rule::Half(t, second) => {
                if dims.len() != 3 || !dims[1].is_multiple_of(2) {
                    return Err(bad("a stack of experts of two halves"));
                }
                let half = dims[1] / 2;
                let pick = (0..half)
                    .map(|r| r + if second { half } else { 0 })
                    .collect();
                let op = Op::Gather {
                    slabs: dims[0],
                    rows: dims[1],
                    cols,
                    transpose: false,
                    pick,
                };
                self.put_assembly(
                    name(t),
                    gather(float_dtype(l)?, vec![dims[0], half, cols], op),
                );
            }
            Rule::Alternating(t, odd) => {
                if dims.len() != 3 || !dims[2].is_multiple_of(2) {
                    return Err(bad("a stack of experts of alternating columns"));
                }
                let half = dims[2] / 2;
                let pick = (0..half).map(|r| 2 * r + usize::from(odd)).collect();
                let op = Op::Gather {
                    slabs: dims[0],
                    rows: dims[1],
                    cols: dims[2],
                    transpose: true,
                    pick,
                };
                self.put_assembly(
                    name(t),
                    gather(float_dtype(l)?, vec![dims[0], half, dims[1]], op),
                );
            }
            Rule::AlternatingBias(t, odd) => {
                if dims.len() != 2 || !dims[1].is_multiple_of(2) {
                    return Err(bad("a bias of alternating columns"));
                }
                let (e, half) = (dims[0], dims[1] / 2);
                let pick = (0..e)
                    .flat_map(|x| (0..half).map(move |f| x * dims[1] + 2 * f + usize::from(odd)))
                    .collect();
                let op = Op::Gather {
                    slabs: 1,
                    rows: e * dims[1],
                    cols: 1,
                    transpose: false,
                    pick,
                };
                self.put_assembly(name(t), gather(float_dtype(l)?, vec![e, half], op));
            }
            Rule::Transposed(t) => {
                if dims.len() != 3 {
                    return Err(bad("a stack of matrices"));
                }
                let op = Op::Gather {
                    slabs: dims[0],
                    rows: dims[1],
                    cols: dims[2],
                    transpose: true,
                    pick: (0..dims[2]).collect(),
                };
                self.put_assembly(
                    name(t),
                    gather(float_dtype(l)?, vec![dims[0], dims[2], dims[1]], op),
                );
            }
            Rule::ExpertsTransposed(t) => {
                let e = shapes.experts;
                if e == 0 || dims.len() != 2 || !dims[0].is_multiple_of(e) {
                    return Err(bad("a flat stack of experts"));
                }
                let op = Op::Gather {
                    slabs: e,
                    rows: dims[0] / e,
                    cols: dims[1],
                    transpose: true,
                    pick: (0..dims[1]).collect(),
                };
                self.put_assembly(
                    name(t),
                    gather(float_dtype(l)?, vec![e, dims[1], dims[0] / e], op),
                );
            }
            Rule::AlternatingBlocks(t, _) | Rule::Blocks(t) => {
                let odd = matches!(rule, Rule::AlternatingBlocks(_, true));
                let scales = by_name
                    .get(scales_of(&l.name).as_str())
                    .ok_or_else(|| Error(format!("{} has no scales", l.name)))?;
                if l.shape.len() != 4 || l.shape[3] != 16 {
                    return Err(bad("MXFP4 blocks"));
                }
                let (slabs, rows, blocks) = (l.shape[0], l.shape[1], l.shape[2]);
                let alternating = matches!(rule, Rule::AlternatingBlocks(..));
                let pick: Vec<usize> = if alternating {
                    (0..rows / 2).map(|f| 2 * f + usize::from(odd)).collect()
                } else {
                    (0..rows).collect()
                };
                let out = vec![slabs, pick.len(), blocks * 32];
                self.put_assembly(
                    name(t),
                    Assembly {
                        dtype: GgmlDType::MxFp4,
                        dims: out,
                        srcs: vec![Self::source(l)?, Self::source(scales)?],
                        op: Op::Blocks {
                            slabs,
                            rows,
                            blocks,
                            pick,
                        },
                    },
                );
            }
            Rule::Stack(_) => unreachable!("stacks are built from their group"),
        }
        Ok(())
    }
}

/// What a layout makes of the tensors listed, or why it cannot.
fn translate(items: &[Listed], layout: &Layout, cfg: &AwqConfigJson) -> Result<Translation> {
    let by_name: HashMap<&str, &Listed> = items.iter().map(|l| (l.name.as_str(), l)).collect();
    let shapes = Shapes::of(cfg);
    // The model's prefix is where its embedding table is.
    let table = layout
        .embed
        .iter()
        .flat_map(|e| {
            items
                .iter()
                .filter(move |l| l.name.ends_with(e) && (!layout.rooted || l.name == *e))
                .map(move |l| (l, *e))
        })
        .min_by_key(|(l, _)| l.name.len())
        .ok_or_else(|| {
            Error(format!(
                "the checkpoint has no {}",
                layout.embed.first().copied().unwrap_or("embedding table")
            ))
        })?;
    let model = if layout.rooted {
        String::new()
    } else {
        table.0.name.trim_end_matches(table.1).to_string()
    };
    let wrapper = model
        .trim_end_matches('.')
        .rsplit_once('.')
        .map_or(String::new(), |(w, _)| format!("{w}."));
    let mut out = Translation::default();
    out.put_view("token_embd.weight".into(), table.0, table.0.shape.clone())?;
    let plain = |t: &str| t.to_string();
    for (part, rule) in layout.top {
        if let Some(l) = by_name.get(format!("{model}{part}").as_str()) {
            out.apply(*rule, l, &by_name, &shapes, &plain)?;
        }
    }
    for (part, rule) in layout.head {
        let found = [wrapper.as_str(), model.as_str(), ""]
            .iter()
            .find_map(|p| by_name.get(format!("{p}{part}").as_str()));
        if let Some(l) = found {
            out.apply(*rule, l, &by_name, &shapes, &plain)?;
        }
    }
    let root = format!("{model}{}.", layout.layers);
    // Experts stored one tensor each, gathered by the tensor they stack into.
    let mut stacks: BTreeMap<String, BTreeMap<usize, &Listed>> = BTreeMap::new();
    for l in items {
        let Some(rest) = l.name.strip_prefix(root.as_str()) else {
            continue;
        };
        let Some((n, part)) = rest.split_once('.') else {
            continue;
        };
        let Ok(n) = n.parse::<usize>() else { continue };
        let block = |t: &str| format!("blk.{n}.{t}");
        let hits = layout.layer.iter().find_map(|table| {
            let hits: Vec<(Rule, Option<usize>)> = table
                .iter()
                .filter_map(|(template, rule)| matches(template, part).map(|e| (*rule, e)))
                .collect();
            (!hits.is_empty()).then_some(hits)
        });
        for (rule, expert) in hits.unwrap_or_default() {
            match (rule, expert) {
                (Rule::Stack(t), Some(e)) => {
                    stacks.entry(block(t)).or_default().insert(e, l);
                }
                _ => out.apply(rule, l, &by_name, &shapes, &block)?,
            }
        }
    }
    for (name, experts) in stacks {
        let count = experts.len();
        if experts.keys().copied().ne(0..count) {
            return Err(Error(format!("{name}: experts are not numbered from 0")));
        }
        let first = experts.values().next().expect("a stack has an expert");
        let dtype = float_dtype(first)?;
        let mut srcs = Vec::new();
        for l in experts.values() {
            if l.shape != first.shape || float_dtype(l)? != dtype {
                return Err(Error(format!("{name}: experts differ in shape or type")));
            }
            srcs.push(Translation::source(l)?);
        }
        let rows = first.shape[0];
        let cols = first.shape.get(1).copied().unwrap_or(1);
        let mut dims = vec![count];
        dims.extend(first.shape.iter());
        out.put_assembly(
            name,
            Assembly {
                dtype,
                dims,
                srcs,
                op: Op::Gather {
                    slabs: 1,
                    rows,
                    cols,
                    transpose: false,
                    pick: (0..rows).collect(),
                },
            },
        );
    }
    Ok(out)
}

fn listing_of(mapped: &[(Arc<memmap2::Mmap>, Vec<RawTensor>, u64)]) -> Vec<Listed> {
    mapped
        .iter()
        .enumerate()
        .flat_map(|(part, (_, tensors, _))| {
            tensors.iter().map(move |t| Listed {
                name: t.0.clone(),
                part,
                dtype: t.1.clone(),
                shape: t.2.clone(),
                start: t.3,
            })
        })
        .collect()
}

/// Bytes the text model's blocks take as stored, and bytes its head takes uploaded as F16 (the
/// embedding table standing in for an absent head), for a checkpoint's mapped shards. Tensors
/// of other parts of a composite model (a speech tower) count for neither.
pub fn text_bytes(
    st: &crate::tensor::safetensors::MmapedSafetensors,
    cfg: &AwqConfigJson,
) -> Option<(u64, u64)> {
    let sizes: HashMap<String, usize> = st.byte_sizes().into_iter().collect();
    let items: Vec<Listed> = st
        .names()
        .iter()
        .filter_map(|n| {
            let shape = st.shape_of(n)?;
            let elems: usize = shape.iter().product::<usize>().max(1);
            let dtype = match sizes.get(*n)? / elems {
                4 => "F32",
                2 => "BF16",
                _ => "U8",
            };
            Some(Listed {
                name: n.to_string(),
                part: 0,
                dtype: dtype.to_string(),
                shape,
                start: 0,
            })
        })
        .collect();
    let t = translate(&items, layout_of(cfg).ok()?, cfg).ok()?;
    let size = |dtype: GgmlDType, dims: &[usize]| {
        (dims.iter().product::<usize>() / dtype.block_size() * dtype.type_size()) as u64
    };
    let block_bytes: u64 = t
        .direct
        .iter()
        .map(|(n, (_, i))| (n, size(i.ggml_dtype, i.shape.dims())))
        .chain(t.assembled.iter().map(|(n, a)| (n, size(a.dtype, &a.dims))))
        .filter(|(n, _)| n.starts_with("blk."))
        .map(|(_, b)| b)
        .sum();
    let head = t
        .direct
        .get("output.weight")
        .or_else(|| t.direct.get("token_embd.weight"))?;
    let head_bytes = head.1.shape.elem_count() as u64 * 2;
    Some((block_bytes, head_bytes))
}

/// The tensors of one shard: name, stored dtype, dimensions and start in the data.
type RawTensor = (String, String, Vec<usize>, u64);

fn read_header(mmap: &memmap2::Mmap, path: &str) -> Result<(Vec<RawTensor>, u64)> {
    let bad = |m: &str| Error(format!("{path}: {m}"));
    let len = mmap.get(..8).ok_or_else(|| bad("shorter than a header"))?;
    let len = u64::from_le_bytes(len.try_into().map_err(|_| bad("header length"))?) as usize;
    let json = mmap
        .get(8..8 + len)
        .ok_or_else(|| bad("header past the end"))?;
    let header: serde_json::Value =
        serde_json::from_slice(json).map_err(|e| bad(&format!("header: {e}")))?;
    let mut out = Vec::new();
    for (name, v) in header
        .as_object()
        .ok_or_else(|| bad("header is not an object"))?
    {
        if name == "__metadata__" {
            continue;
        }
        let dtype = v
            .get("dtype")
            .and_then(|d| d.as_str())
            .unwrap_or("")
            .to_string();
        let shape = v
            .get("shape")
            .and_then(|s| s.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|d| d.as_u64().map(|d| d as usize))
                    .collect()
            })
            .unwrap_or_default();
        let start = v
            .get("data_offsets")
            .and_then(|o| o.get(0))
            .and_then(|o| o.as_u64())
            .ok_or_else(|| bad("a tensor has no data_offsets"))?;
        out.push((name.clone(), dtype, shape, start));
    }
    Ok((out, 8 + len as u64))
}

impl Assembly {
    /// The assembled tensor's bytes in its own type, `parts` being the shards the sources are in.
    fn bytes(&self, parts: &[DensePart]) -> Result<Vec<u8>> {
        let sources: Vec<&[u8]> = self
            .srcs
            .iter()
            .map(|(p, info)| {
                let part = &parts[*p];
                let at = (part.content.tensor_data_offset + info.offset) as usize;
                part.mmap
                    .get(at..at + info.size_in_bytes())
                    .ok_or_else(|| Error("a tensor reaches past its shard".to_string()))
            })
            .collect::<Result<_>>()?;
        let width = self.srcs[0].1.ggml_dtype.type_size();
        Ok(match &self.op {
            Op::Gather {
                slabs,
                rows,
                cols,
                transpose,
                pick,
            } => {
                let matrix = rows * cols * width;
                let mut out = Vec::with_capacity(sources.len() * slabs * pick.len() * cols * width);
                for src in &sources {
                    for slab in 0..*slabs {
                        let m = &src[slab * matrix..(slab + 1) * matrix];
                        for &p in pick {
                            if *transpose {
                                for i in 0..*rows {
                                    let at = (i * cols + p) * width;
                                    out.extend_from_slice(&m[at..at + width]);
                                }
                            } else {
                                out.extend_from_slice(&m[p * cols * width..(p + 1) * cols * width]);
                            }
                        }
                    }
                }
                out
            }
            Op::Blocks {
                slabs,
                rows,
                blocks,
                pick,
            } => {
                let (data, scales) = (sources[0], sources[1]);
                let mut out = Vec::with_capacity(slabs * pick.len() * blocks * 17);
                for slab in 0..*slabs {
                    for &p in pick {
                        for b in 0..*blocks {
                            let at = (slab * rows + p) * blocks + b;
                            out.push(scales[at]);
                            let packed = &data[at * 16..(at + 1) * 16];
                            // Stored: byte i holds values 2i (low) and 2i + 1 (high); the
                            // block holds value j (low) and j + 16 (high) in byte j.
                            let value = |k: usize| (packed[k / 2] >> (4 * (k % 2))) & 0xF;
                            out.extend((0..16).map(|j| value(j) | (value(j + 16) << 4)));
                        }
                    }
                }
                out
            }
            Op::Map(f) => {
                let dtype = self.srcs[0].1.ggml_dtype;
                let mut out = Vec::with_capacity(sources[0].len() / width * 4);
                for v in sources[0].chunks_exact(width) {
                    let x = match dtype {
                        GgmlDType::F32 => f32::from_le_bytes([v[0], v[1], v[2], v[3]]),
                        GgmlDType::F16 => half::f16::from_le_bytes([v[0], v[1]]).to_f32(),
                        _ => half::bf16::from_le_bytes([v[0], v[1]]).to_f32(),
                    };
                    out.extend_from_slice(&f(x).to_le_bytes());
                }
                out
            }
        })
    }
}

impl DenseCheckpoint {
    /// The tensor assembled under `name`, as `(type, dimensions, bytes)`; `None` when the
    /// checkpoint reads no such tensor that way.
    pub(super) fn assemble(&self, name: &str) -> Result<Option<(GgmlDType, Vec<usize>, Vec<u8>)>> {
        let Some(a) = self.assembled.get(name) else {
            return Ok(None);
        };
        Ok(Some((a.dtype, a.dims.clone(), a.bytes(&self.parts)?)))
    }

    /// Map `shards` and list their tensors under the loader's names. `base` carries the
    /// metadata the model is configured from; `cfg` the rope scaling the table is computed for.
    pub fn open(base: &Content, shards: &[String], cfg: &AwqConfigJson) -> Result<Self> {
        let mut mapped = Vec::new();
        for path in shards {
            let file = std::fs::File::open(path).map_err(|e| Error(format!("{path}: {e}")))?;
            let mmap =
                unsafe { memmap2::Mmap::map(&file) }.map_err(|e| Error(format!("{path}: {e}")))?;
            let (listing, data_start) = read_header(&mmap, path)?;
            mapped.push((Arc::new(mmap), listing, data_start));
        }
        let items = listing_of(&mapped);
        let translation = translate(&items, layout_of(cfg)?, cfg)?;
        let mut parts = Vec::new();
        let mut union: HashMap<String, TensorInfo> = HashMap::new();
        for (index, (mmap, _, data_start)) in mapped.into_iter().enumerate() {
            let mut infos = HashMap::new();
            for (name, (part, info)) in &translation.direct {
                if *part == index {
                    union.insert(name.clone(), info_clone(info));
                    infos.insert(name.clone(), info_clone(info));
                }
            }
            let owner: Arc<dyn std::any::Any + Send + Sync> = mmap.clone();
            parts.push(DensePart {
                content: Content {
                    magic: VersionedMagic::GgufV3,
                    metadata: HashMap::new(),
                    tensor_infos: infos,
                    tensor_data_offset: data_start,
                    mmap_owner: Some(owner),
                },
                mmap,
            });
        }
        // The rope table a GGUF file stores, computed from the configuration's scaling, and
        // the listing of the tensors made from stored ones.
        let mut computed_infos = HashMap::new();
        let mut bytes = Vec::new();
        let head_dim = cfg.head_dim.unwrap_or(cfg.hidden_size / cfg.n_head);
        let factors = match cfg.frequency_scaling {
            Some(scaling) => Some(frequency_factors(head_dim, cfg.rope_theta, scaling)),
            None => crate::inference::load::hf_config::proportional_factors(
                &cfg.root,
                crate::inference::load::hf_config::global_head_dim(&cfg.root).unwrap_or(head_dim),
            ),
        };
        if let Some(factors) = factors {
            bytes = factors.iter().flat_map(|f| f.to_le_bytes()).collect();
            let info = TensorInfo {
                ggml_dtype: GgmlDType::F32,
                shape: Shape::from(vec![factors.len()]),
                offset: 0,
            };
            union.insert("rope_freqs.weight".into(), info_clone(&info));
            computed_infos.insert("rope_freqs.weight".to_string(), info);
        }
        for (name, a) in &translation.assembled {
            let info = TensorInfo {
                ggml_dtype: a.dtype,
                shape: Shape::from(a.dims.clone()),
                offset: 0,
            };
            union.insert(name.clone(), info_clone(&info));
            computed_infos.insert(name.clone(), info);
        }
        let computed = Content {
            magic: VersionedMagic::GgufV3,
            metadata: HashMap::new(),
            tensor_infos: computed_infos,
            tensor_data_offset: 0,
            mmap_owner: None,
        };
        let mut content = Content {
            magic: base.magic,
            metadata: base.metadata.clone(),
            tensor_infos: union,
            tensor_data_offset: 0,
            mmap_owner: None,
        };
        fix_metadata(&mut content, cfg);
        let adjacent_rope = cfg.native || hf_class(&cfg.hf_arch).is_some_and(|c| c.adjacent_rope);
        Ok(Self {
            parts,
            computed: (computed, bytes),
            assembled: translation.assembled,
            content,
            adjacent_rope,
        })
    }
}

/// The header keys the tensors say better than the configuration: a hybrid's per-block key-value
/// heads, the width of a routed expert, the window of a short convolution.
fn fix_metadata(content: &mut Content, cfg: &AwqConfigJson) {
    use super::mixer::{layer_mixer_kind, MixerKind};
    let a = cfg.gguf_arch.clone();
    let dim = |content: &Content, tail: &str, axis: usize| {
        (0..cfg.n_layers).find_map(|il| {
            content
                .tensor_infos
                .get(&format!("blk.{il}.{tail}"))
                .and_then(|t| t.shape.dims().get(axis).copied())
        })
    };
    let kinds: Vec<MixerKind> = (0..cfg.n_layers)
        .map(|il| layer_mixer_kind(content, il))
        .collect();
    if kinds.contains(&MixerKind::Attention) && kinds.iter().any(|k| *k != MixerKind::Attention) {
        let heads = Value::Array(
            kinds
                .iter()
                .map(|k| {
                    Value::U32(if *k == MixerKind::Attention {
                        cfg.n_kv_head as u32
                    } else {
                        0
                    })
                })
                .collect(),
        );
        content
            .metadata
            .insert(format!("{a}.attention.head_count_kv"), heads);
    }
    let experts = dim(content, "ffn_up_exps.weight", 1)
        .or_else(|| dim(content, "ffn_gate_exps.weight", 1))
        .or_else(|| dim(content, "ffn_gate_up_exps.weight", 1).map(|rows| rows / 2));
    if let Some(width) = experts {
        content.metadata.insert(
            format!("{a}.expert_feed_forward_length"),
            Value::U32(width as u32),
        );
    }
    if let Some(l_cache) = dim(content, "shortconv.conv.weight", 1) {
        content
            .metadata
            .insert(format!("{a}.shortconv.l_cache"), Value::U32(l_cache as u32));
    }
}

fn info_clone(i: &TensorInfo) -> TensorInfo {
    TensorInfo {
        ggml_dtype: i.ggml_dtype,
        shape: i.shape.clone(),
        offset: i.offset,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn listed(names: &[(&str, &[usize])]) -> Vec<Listed> {
        names
            .iter()
            .map(|(n, s)| Listed {
                name: n.to_string(),
                part: 0,
                dtype: "F32".into(),
                shape: s.to_vec(),
                start: 0,
            })
            .collect()
    }

    fn config(class: &str) -> AwqConfigJson {
        config_with(class, "")
    }

    fn config_with(class: &str, extra: &str) -> AwqConfigJson {
        let text = format!(
            r#"{{"architectures": ["{class}"], "hidden_size": 8, "num_hidden_layers": 2,
            "num_attention_heads": 2, "vocab_size": 4, "intermediate_size": 8{extra}}}"#
        );
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("loken-dense-names-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), text).unwrap();
        let cfg = crate::inference::load::awq_loader::parse_config_json(dir.to_str().unwrap());
        let _ = std::fs::remove_dir_all(&dir);
        cfg.unwrap()
    }

    #[test]
    fn checkpoint_names_become_loader_names() {
        let cfg = config("LlamaForCausalLM");
        let l = layout_of(&cfg).unwrap();
        let t = translate(
            &listed(&[
                ("model.embed_tokens.weight", &[4, 8]),
                ("model.norm.weight", &[8]),
                ("lm_head.weight", &[4, 8]),
                ("model.layers.3.self_attn.o_proj.weight", &[8, 8]),
                ("model.layers.3.mlp.up_proj.weight", &[8, 8]),
                ("audio_tower.conv1.weight", &[8, 4]),
            ]),
            l,
            &cfg,
        )
        .unwrap();
        let mut names: Vec<&str> = t.direct.keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "blk.3.attn_output.weight",
                "blk.3.ffn_up.weight",
                "output.weight",
                "output_norm.weight",
                "token_embd.weight"
            ]
        );
        // A wrapper's prefix is the embedding table's, the head sits beside the wrapper.
        let t = translate(
            &listed(&[
                ("language_model.model.embed_tokens.weight", &[4, 8]),
                ("language_model.model.norm.weight", &[8]),
                ("language_model.lm_head.weight", &[4, 8]),
            ]),
            l,
            &cfg,
        )
        .unwrap();
        assert!(t.direct.contains_key("output.weight"));
        assert!(t.direct.contains_key("output_norm.weight"));
    }

    /// A fused projection is read in place unless the configuration says its rows are grouped by
    /// key-value head.
    #[test]
    fn the_configuration_says_how_a_fused_projection_is_laid_out() {
        let names = [
            ("transformer.word_embeddings.weight", &[4usize, 8][..]),
            (
                "transformer.h.0.self_attention.query_key_value.weight",
                &[16, 8],
            ),
        ];
        let tensors = listed(&names);
        let common = r#", "num_kv_heads": 1, "head_dim": 4"#;
        for (extra, grouped) in [
            (r#", "multi_query": true"#, false),
            (
                r#", "multi_query": true, "new_decoder_architecture": true"#,
                true,
            ),
        ] {
            let cfg = config_with("FalconForCausalLM", &format!("{common}{extra}"));
            let t = translate(&tensors, layout_of(&cfg).unwrap(), &cfg).unwrap();
            let name = "blk.0.attn_qkv.weight";
            assert_eq!(t.assembled.contains_key(name), grouped, "{extra}");
            assert_eq!(t.direct.contains_key(name), !grouped, "{extra}");
        }
    }

    /// Stored rows become the loader's: a fused projection grouped by key-value head is read
    /// query rows, key rows, value rows; per-expert tensors are stacked; a norm scale is shifted.
    #[test]
    fn assembled_tensors_hold_the_bytes_the_rules_say() {
        let rows = grouped_rows(4, 2, 3);
        // Two key-value heads of two query heads: q0 q1 k v | q2 q3 k v, three rows each.
        let block = |i: usize| (3 * i..3 * i + 3).collect::<Vec<_>>();
        let want: Vec<usize> = [0, 1, 4, 5, 2, 6, 3, 7]
            .iter()
            .flat_map(|i| block(*i))
            .collect();
        assert_eq!(rows, want);
        let (qkv, z) = delta_rows((2, 2, 4, 3));
        // Per key head: query 2, key 2, values and gate 6 each.
        assert_eq!(qkv.len(), 2 * 2 + 2 * 2 + 12);
        assert_eq!(z.len(), 12);
        assert_eq!(qkv[0..2], [0, 1]);
        assert_eq!(qkv[2..4], [16, 17]);
        assert_eq!(z[0..6], [10, 11, 12, 13, 14, 15]);
        assert_eq!(
            matches(
                "mlp.experts.{e}.up_proj.weight",
                "mlp.experts.12.up_proj.weight"
            ),
            Some(Some(12))
        );
        assert_eq!(
            matches("mlp.experts.{e}.up_proj.weight", "mlp.up_proj.weight"),
            None
        );
    }

    #[test]
    fn a_long_context_rope_keeps_short_wavelengths_and_divides_long_ones() {
        let f = frequency_factors(64, 500_000.0, (32.0, 1.0, 4.0, 8192.0));
        assert_eq!(f.len(), 32);
        assert_eq!(f[0], 1.0);
        assert_eq!(f[31], 32.0);
        assert!(f.windows(2).all(|w| w[0] <= w[1]));
    }

    /// A seeded toy checkpoint against the reference's logits: the model built from the
    /// checkpoint's own tensors, in the width they are stored, gives the logits of the last
    /// position. The CPU path keeps its cache quantised, so agreement is to a few percent of the
    /// logits' range; a wrong rope pairing is off by more than the range itself.
    fn logits_match(dir: &str, tolerance: f32) {
        use crate::inference::load::awq_loader as awql;
        use crate::inference::place::layer_executor::HeteroPlan;
        use crate::tensor::safetensors::MmapedSafetensors;
        use crate::tensor::{Device, Tensor};
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/vectors/dense_llama")
            .join(dir);
        let cfg = awql::parse_config_json(root.to_str().unwrap()).unwrap();
        assert!(!cfg.quantized);
        let shards = vec![root.join("model.safetensors").to_string_lossy().to_string()];
        let header = awql::synth_gguf_content(&cfg, awql::AwqTensorPresence::default());
        let ck = DenseCheckpoint::open(&header, &shards, &cfg).unwrap();
        let plan = HeteroPlan::calculate_with_kv(cfg.n_layers, 1 << 20, &[], &[], 1.0, 0);
        let mut model =
            crate::inference::generic_transformer::GenericHeteroTransformer::from_dense_checkpoint(
                &ck,
                &HashMap::new(),
                &plan,
                crate::inference::engine::llm_engine::KvQuant::Off,
                None,
                false,
                false,
            )
            .unwrap();
        let vectors = unsafe { MmapedSafetensors::new(root.join("vectors.safetensors")) }.unwrap();
        let ids: Vec<u32> = vectors
            .load("ids", &Device::Cpu)
            .unwrap()
            .to_dtype(crate::tensor::DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|v| *v as u32)
            .collect();
        let want = vectors.load("logits", &Device::Cpu).unwrap();
        let vocab = want.dim(1).unwrap();
        let rows = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let scale = rows.iter().map(|v| v.abs()).fold(0f32, f32::max);
        // Every prefix of the prompt: a fresh prefill of the first n tokens gives row n - 1.
        let mut report = Vec::new();
        // The CPU decode path keeps its cache in Q8 blocks of 32 values, which a head of 8 cannot
        // form, so only the whole prompt is prefilled.
        for n in ids.len()..=ids.len() {
            let input = Tensor::from_vec(ids[..n].to_vec(), (1usize, n), &Device::Cpu).unwrap();
            let got = model
                .forward(&input, 0)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(got.len(), vocab);
            let want = &rows[(n - 1) * vocab..n * vocab];
            let worst = got
                .iter()
                .zip(want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            report.push((n, worst / scale));
        }
        assert!(
            report.iter().all(|(_, e)| *e <= tolerance),
            "{dir}: error per prefix length {report:?}"
        );
    }

    /// The rope table: the frequencies the model rotates by are the reference's, whichever
    /// scaling the configuration declares.
    #[test]
    fn the_rope_frequencies_are_the_references() {
        use crate::inference::load::awq_loader as awql;
        use crate::tensor::safetensors::MmapedSafetensors;
        use crate::tensor::Device;
        for dir in ["f32", "plain"] {
            let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/vectors/dense_llama")
                .join(dir);
            let cfg = awql::parse_config_json(root.to_str().unwrap()).unwrap();
            let want = unsafe { MmapedSafetensors::new(root.join("vectors.safetensors")) }
                .unwrap()
                .load("inv_freq", &Device::Cpu)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let head_dim = cfg.head_dim.unwrap();
            let factors = match cfg.frequency_scaling {
                Some(s) => awql::frequency_factors(head_dim, cfg.rope_theta, s),
                None => vec![1.0; head_dim / 2],
            };
            for (i, (w, f)) in want.iter().zip(&factors).enumerate() {
                let got = 1.0 / cfg.rope_theta.powf(2.0 * i as f32 / head_dim as f32) / f;
                assert!(
                    ((got - w) / w).abs() < 1e-5,
                    "{dir} pair {i}: {got} against {w}"
                );
            }
            assert_eq!(want.len(), factors.len());
        }
    }

    #[test]
    fn a_dense_f32_checkpoint_gives_the_reference_logits() {
        logits_match("f32", 6e-2);
    }

    /// The same weights under an unscaled rope: the frequency table is the only difference.
    #[test]
    fn a_dense_checkpoint_without_rope_scaling_gives_the_reference_logits() {
        logits_match("plain", 6e-2);
    }

    #[test]
    fn a_dense_bf16_checkpoint_gives_the_reference_logits() {
        logits_match("bf16", 8e-2);
    }
}
