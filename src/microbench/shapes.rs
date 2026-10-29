//! The shapes a bench runs at: a decode row and one routed expert's projections, from the
//! options or from a GGUF's metadata and tensor table, read without loading a weight.

use super::Options;
use crate::tensor::quantized::gguf_file::Content;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// The names of a routed expert's projections, in the order a file stores them.
const PROJECTIONS: [&str; 3] = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];

/// One projection of an expert, as stored.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Projection {
    pub name: String,
    /// The bytes one expert's copy occupies.
    pub bytes: u64,
    /// Where each expert's copy starts in the file, by expert. Empty without a file.
    pub offsets: Vec<u64>,
    /// The stored format, when a file says.
    pub dtype: Option<String>,
}

/// The shapes resolved for a run.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Shapes {
    /// Where the shapes came from: `options` or the file.
    pub source: String,
    /// The file whose offsets the projections list.
    pub file: Option<PathBuf>,
    pub layer: Option<usize>,
    /// The expert whose copy the file-reading benches start at.
    pub expert: usize,
    pub hidden: usize,
    /// The hyper-connection multiple: rows the residual stream carries per token.
    pub hc: usize,
    pub element_bytes: usize,
    pub projections: Vec<Projection>,
}

impl Shapes {
    /// The bytes of one token's residual stream: every hyper-connection row of the hidden size.
    pub fn row_bytes(&self) -> usize {
        self.hidden * self.hc * self.element_bytes
    }

    /// The bytes of one expert, its projections together.
    pub fn expert_bytes(&self) -> u64 {
        self.projections.iter().map(|p| p.bytes).sum()
    }

    /// The experts the file offsets list, 0 without a file.
    pub fn experts_listed(&self) -> usize {
        self.projections
            .iter()
            .map(|p| p.offsets.len())
            .min()
            .unwrap_or(0)
    }

    /// One line naming the shapes.
    pub fn describe(&self) -> String {
        let projections: Vec<String> = self
            .projections
            .iter()
            .map(|p| match &p.dtype {
                Some(d) => format!("{} {} B {d}", p.name, p.bytes),
                None => format!("{} {} B", p.name, p.bytes),
            })
            .collect();
        let layer = self.layer.map_or(String::new(), |l| format!(", layer {l}"));
        format!(
            "hidden {} x hc {} x {} B = row {} B{layer}, expert {} = {} B ({}), from {}",
            self.hidden,
            self.hc,
            self.element_bytes,
            self.row_bytes(),
            self.expert,
            self.expert_bytes(),
            projections.join(", "),
            self.source
        )
    }
}

/// The shapes the options name: the file's when `--shapes-from` is given, the options' otherwise.
pub fn resolve(opts: &Options) -> Result<Shapes, String> {
    match &opts.shapes_from {
        Some(path) => {
            let content = crate::tensor::quantized::gguf_file::open_header(path)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            from_content(&content, path, opts)
        }
        None => from_options(opts),
    }
}

/// The shapes from the options alone: each projection is `hidden x inter` weights at
/// `--weight-bits`, or the bytes `--projection-bytes` states.
pub fn from_options(opts: &Options) -> Result<Shapes, String> {
    let bytes: Vec<u64> = match opts.projection_bytes.len() {
        0 => {
            let b = (opts.hidden as f64 * opts.inter as f64 * opts.weight_bits / 8.0).round();
            vec![b as u64; PROJECTIONS.len()]
        }
        1 => vec![opts.projection_bytes[0]; PROJECTIONS.len()],
        n if n == PROJECTIONS.len() => opts.projection_bytes.clone(),
        n => {
            return Err(format!(
                "--projection-bytes takes one value or {}, not {n}",
                PROJECTIONS.len()
            ))
        }
    };
    if opts.hidden == 0 || opts.hc == 0 || opts.element_bytes == 0 {
        return Err("--hidden, --hc and --element-bytes must be above zero".to_string());
    }
    Ok(Shapes {
        source: "options".to_string(),
        file: None,
        layer: None,
        expert: opts.expert,
        hidden: opts.hidden,
        hc: opts.hc,
        element_bytes: opts.element_bytes,
        projections: PROJECTIONS
            .iter()
            .zip(bytes)
            .map(|(name, bytes)| Projection {
                name: name.to_string(),
                bytes,
                offsets: Vec::new(),
                dtype: None,
            })
            .collect(),
    })
}

/// The shapes of `content`, the header of the GGUF at `path`: the hidden size and the
/// hyper-connection multiple from the metadata, the expert's projections from the tensor table.
/// A stacked projection (`blk.N.ffn_gate_exps.weight` over every expert) is cut per expert; a
/// file with one tensor per expert (`blk.N.ffn_gate_exps.E`) lists each. `--tensor` names one
/// tensor in place of the expert.
pub fn from_content(content: &Content, path: &Path, opts: &Options) -> Result<Shapes, String> {
    let arch = content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .cloned()
        .ok_or_else(|| "the file names no general.architecture".to_string())?;
    let meta_u = |key: &str| -> Option<u64> {
        content
            .metadata
            .get(&format!("{arch}.{key}"))
            .and_then(|v| v.to_u64().ok())
    };
    let hidden = meta_u("embedding_length")
        .ok_or_else(|| format!("the file names no {arch}.embedding_length"))?
        as usize;
    let hc = ["hyper_connection.count", "hyper_connection_mult"]
        .iter()
        .find_map(|k| meta_u(k))
        .unwrap_or(1) as usize;
    let data = content.tensor_data_offset;
    let mut shapes = Shapes {
        source: path.display().to_string(),
        file: Some(path.to_path_buf()),
        layer: None,
        expert: opts.expert,
        hidden,
        hc,
        element_bytes: opts.element_bytes,
        projections: Vec::new(),
    };
    if let Some(name) = &opts.tensor {
        let info = content
            .tensor_infos
            .get(name)
            .ok_or_else(|| format!("the file has no tensor `{name}`"))?;
        shapes.expert = 0;
        shapes.projections.push(Projection {
            name: name.clone(),
            bytes: info.size_in_bytes() as u64,
            offsets: vec![data + info.offset],
            dtype: Some(info.ggml_dtype.gguf_name().to_string()),
        });
        return Ok(shapes);
    }
    let experts = meta_u("expert_count")
        .ok_or_else(|| format!("the file names no {arch}.expert_count"))?
        as usize;
    let layer = match opts.layer {
        Some(l) => l,
        None => content
            .tensor_infos
            .keys()
            .filter_map(|n| expert_layer(n))
            .min()
            .ok_or_else(|| "the file has no routed expert tensors".to_string())?,
    };
    if opts.expert >= experts {
        return Err(format!("expert {} is not below {experts}", opts.expert));
    }
    for name in PROJECTIONS {
        let stacked = format!("blk.{layer}.{name}.weight");
        let projection = if let Some(info) = content.tensor_infos.get(&stacked) {
            let bytes = info.size_in_bytes() as u64 / experts as u64;
            Projection {
                name: name.to_string(),
                bytes,
                offsets: (0..experts as u64)
                    .map(|e| data + info.offset + e * bytes)
                    .collect(),
                dtype: Some(info.ggml_dtype.gguf_name().to_string()),
            }
        } else {
            let unit = |e: usize| {
                [
                    format!("blk.{layer}.{name}.{e}"),
                    format!("blk.{layer}.{name}.{e}.weight"),
                ]
                .into_iter()
                .find_map(|n| content.tensor_infos.get(&n))
            };
            let first = unit(0).ok_or_else(|| {
                format!("layer {layer} has neither `{stacked}` nor `blk.{layer}.{name}.0`")
            })?;
            let offsets = (0..experts)
                .map(|e| {
                    unit(e)
                        .map(|i| data + i.offset)
                        .ok_or_else(|| format!("layer {layer} has no `{name}` for expert {e}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Projection {
                name: name.to_string(),
                bytes: first.size_in_bytes() as u64,
                offsets,
                dtype: Some(first.ggml_dtype.gguf_name().to_string()),
            }
        };
        shapes.projections.push(projection);
    }
    shapes.layer = Some(layer);
    Ok(shapes)
}

/// The layer of a routed expert tensor's name, `None` for any other tensor.
fn expert_layer(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("blk.")?;
    let (layer, tail) = rest.split_once('.')?;
    PROJECTIONS
        .iter()
        .any(|p| tail.starts_with(p))
        .then(|| layer.parse().ok())
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quantized::gguf_file::{TensorInfo, Value, VersionedMagic};
    use crate::tensor::quantized::GgmlDType;
    use crate::tensor::Shape;
    use clap::Parser;
    use std::collections::HashMap;

    fn options(args: &[&str]) -> Options {
        Options::parse_from(std::iter::once("microbench").chain(args.iter().copied()))
    }

    fn content(stacked: bool) -> Content {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            Value::String("arch".to_string()),
        );
        metadata.insert("arch.embedding_length".to_string(), Value::U32(64));
        metadata.insert("arch.expert_count".to_string(), Value::U32(4));
        metadata.insert("arch.hyper_connection.count".to_string(), Value::U32(3));
        let mut tensor_infos = HashMap::new();
        let mut offset = 0u64;
        let mut put = |name: String, dims: Vec<usize>| {
            let info = TensorInfo {
                ggml_dtype: GgmlDType::Q8_0,
                shape: Shape::from(dims),
                offset,
            };
            offset += info.size_in_bytes() as u64;
            tensor_infos.insert(name, info);
        };
        put("blk.0.attn_q.weight".to_string(), vec![64, 64]);
        for layer in [2usize, 3] {
            for name in PROJECTIONS {
                if stacked {
                    put(format!("blk.{layer}.{name}.weight"), vec![4, 32, 64]);
                } else {
                    for e in 0..4 {
                        put(format!("blk.{layer}.{name}.{e}"), vec![32, 64]);
                    }
                }
            }
        }
        Content {
            magic: VersionedMagic::GgufV3,
            metadata,
            tensor_infos,
            tensor_data_offset: 1000,
            mmap_owner: None,
        }
    }

    #[test]
    fn the_options_alone_give_a_row_and_three_projections() {
        let s = from_options(&options(&[
            "crossing",
            "--hidden",
            "8",
            "--hc",
            "2",
            "--inter",
            "16",
            "--weight-bits",
            "4",
        ]))
        .unwrap();
        assert_eq!(s.row_bytes(), 8 * 2 * 4);
        assert_eq!(s.projections.len(), 3);
        assert_eq!(s.projections[0].bytes, 64);
        assert_eq!(s.expert_bytes(), 192);
        assert_eq!(s.experts_listed(), 0);
        let s = from_options(&options(&["crossing", "--projection-bytes", "1,2,3"])).unwrap();
        assert_eq!(s.expert_bytes(), 6);
        assert!(from_options(&options(&["crossing", "--projection-bytes", "1,2"])).is_err());
    }

    #[test]
    fn a_stacked_file_is_cut_per_expert_and_the_first_expert_layer_is_taken() {
        let c = content(true);
        let s = from_content(&c, Path::new("m.gguf"), &options(&["expert-read"])).unwrap();
        assert_eq!((s.hidden, s.hc, s.layer), (64, 3, Some(2)));
        let per = 32 * 64 / 32 * 34;
        assert_eq!(s.projections[0].bytes, per as u64);
        assert_eq!(s.projections[0].offsets.len(), 4);
        let base = c.tensor_data_offset + c.tensor_infos["blk.2.ffn_gate_exps.weight"].offset;
        assert_eq!(s.projections[0].offsets[1], base + per as u64);
        assert_eq!(s.projections[0].dtype.as_deref(), Some("q8_0"));
        assert_eq!(s.experts_listed(), 4);
    }

    #[test]
    fn a_file_with_one_tensor_per_expert_lists_each() {
        let c = content(false);
        let s = from_content(
            &c,
            Path::new("m.gguf"),
            &options(&["expert-read", "--layer", "3", "--expert", "2"]),
        )
        .unwrap();
        assert_eq!(s.layer, Some(3));
        let want = c.tensor_data_offset + c.tensor_infos["blk.3.ffn_up_exps.2"].offset;
        assert_eq!(s.projections[1].offsets[2], want);
        assert!(from_content(&c, Path::new("m.gguf"), &options(&["x", "--expert", "4"])).is_err());
        assert!(from_content(&c, Path::new("m.gguf"), &options(&["x", "--layer", "0"])).is_err());
    }

    #[test]
    fn a_named_tensor_stands_for_the_expert() {
        let c = content(true);
        let s = from_content(
            &c,
            Path::new("m.gguf"),
            &options(&["expert-upload", "--tensor", "blk.0.attn_q.weight"]),
        )
        .unwrap();
        assert_eq!(s.projections.len(), 1);
        assert_eq!(s.projections[0].offsets, vec![1000]);
        assert!(s.describe().contains("blk.0.attn_q.weight"));
    }
}
