//! The draft head a checkpoint ships beside its layers, read off the hub and written as a GGUF.
//!
//! A released checkpoint carries `mtp.*` tensors that the served quantisations leave out: the
//! two projections that fuse a token's embedding with the hidden state before it, and one more
//! layer of the model's own kind. This reads exactly those tensors with range requests, so the
//! shards they sit in are never downloaded whole, quantises them, and writes them under the
//! names the layer loader reads: `blk.N.*` for the layer, N being the count of the model's own
//! layers, and `mtp.*` for the rest.

use crate::tensor::gguf_write::{GgufStreamWriter, PlannedEntry};
use crate::tensor::quantized::gguf_file::Value;
use crate::tensor::quantized::{GgmlDType, QTensor};
use crate::tensor::{Device, Error, Result, Tensor};
use std::collections::BTreeMap;
use std::path::Path;

/// Experts read and quantised per request.
const EXPERTS_PER_REQUEST: usize = 8;
/// Attempts per request before the conversion gives up.
const ATTEMPTS: usize = 4;

/// How a tensor reaches the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    /// Gates read in full precision.
    Exact,
    /// Norm scales, written as the model's own file holds them: one more than the checkpoint's.
    Norm,
    /// Quantised by its row length.
    Matrix,
    /// Rows `from..to` of the released tensor.
    Rows(usize, usize),
    /// Experts stacked on the first axis, half of the second axis: the first half (gate) or the
    /// second (up) of a fused projection.
    FusedHalf { second: bool },
    /// Experts stacked on the first axis, quantised whole.
    Stack,
}

/// A tensor of the file: where it comes from, what it is called, how it is written.
#[derive(Clone, Debug)]
struct Wanted {
    source: String,
    name: String,
    role: Role,
}

fn wanted(source: &str, name: &str, role: Role) -> Wanted {
    Wanted {
        source: source.to_string(),
        name: name.to_string(),
        role,
    }
}

/// Every tensor of the head, from the checkpoint's names to the loader's. `q_rows` and `k_rows`
/// split the indexer's fused projection.
fn plan(index: usize, q_rows: usize, k_rows: usize) -> Vec<Wanted> {
    let layer = "mtp.layers.0.";
    let blk = format!("blk.{index}");
    let mut out = vec![
        wanted(
            "mtp.fc_embedding.weight",
            "mtp.fc_embedding.weight",
            Role::Matrix,
        ),
        wanted("mtp.fc_hidden.weight", "mtp.fc_hidden.weight", Role::Matrix),
        wanted(
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_embedding.weight",
            Role::Norm,
        ),
        wanted(
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.pre_fc_norm_hidden.weight",
            Role::Norm,
        ),
        wanted(
            "mtp.hyper_connection_mixer.hc_norm.weight",
            "mtp.out_hc_norm.weight",
            Role::Norm,
        ),
        wanted(
            "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
            "mtp.out_hc_down.weight",
            Role::Matrix,
        ),
        wanted(
            "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
            "mtp.out_hc_up.weight",
            Role::Matrix,
        ),
    ];
    for (hf, gguf) in [("attn", "attn"), ("mlp", "ffn")] {
        let from = format!("{layer}{hf}_hyper_connection.");
        out.push(wanted(
            &format!("{from}block_inject_weight.weight"),
            &format!("{blk}.hc_{gguf}_inject.weight"),
            Role::Matrix,
        ));
        out.push(wanted(
            &format!("{from}hc_norm.weight"),
            &format!("{blk}.hc_{gguf}_norm.weight"),
            Role::Norm,
        ));
        out.push(wanted(
            &format!("{from}input_mix_weight_down.weight"),
            &format!("{blk}.hc_{gguf}_down.weight"),
            Role::Matrix,
        ));
        out.push(wanted(
            &format!("{from}input_mix_weight_up.weight"),
            &format!("{blk}.hc_{gguf}_up.weight"),
            Role::Matrix,
        ));
    }
    for (hf, gguf, role) in [
        ("q_proj", "attn_q", Role::Matrix),
        ("k_proj", "attn_k", Role::Matrix),
        ("v_proj", "attn_v", Role::Matrix),
        ("o_proj", "attn_output", Role::Matrix),
        ("q_norm", "attn_q_norm", Role::Norm),
        ("k_norm", "attn_k_norm", Role::Norm),
    ] {
        out.push(wanted(
            &format!("{layer}self_attn.{hf}.weight"),
            &format!("{blk}.{gguf}.weight"),
            role,
        ));
    }
    let indexer = format!("{layer}self_attn.indexer.");
    out.push(wanted(
        &format!("{indexer}index_qk_proj.weight"),
        &format!("{blk}.indexer.q_proj.weight"),
        Role::Rows(0, q_rows),
    ));
    out.push(wanted(
        &format!("{indexer}index_qk_proj.weight"),
        &format!("{blk}.indexer.k_proj.weight"),
        Role::Rows(q_rows, q_rows + k_rows),
    ));
    out.push(wanted(
        &format!("{indexer}q_layernorm.weight"),
        &format!("{blk}.indexer.q_norm.weight"),
        Role::Norm,
    ));
    out.push(wanted(
        &format!("{indexer}k_layernorm.weight"),
        &format!("{blk}.indexer.k_norm.weight"),
        Role::Norm,
    ));
    out.push(wanted(
        &format!("{layer}mlp.gate.weight"),
        &format!("{blk}.ffn_gate_inp.weight"),
        Role::Exact,
    ));
    out.push(wanted(
        &format!("{layer}mlp.shared_expert_gate.weight"),
        &format!("{blk}.ffn_gate_inp_shexp.weight"),
        Role::Exact,
    ));
    for (hf, gguf) in [
        ("gate_proj", "gate"),
        ("up_proj", "up"),
        ("down_proj", "down"),
    ] {
        out.push(wanted(
            &format!("{layer}mlp.shared_expert.{hf}.weight"),
            &format!("{blk}.ffn_{gguf}_shexp.weight"),
            Role::Matrix,
        ));
    }
    out.push(wanted(
        &format!("{layer}mlp.experts.gate_up_proj"),
        &format!("{blk}.ffn_gate_exps.weight"),
        Role::FusedHalf { second: false },
    ));
    out.push(wanted(
        &format!("{layer}mlp.experts.gate_up_proj"),
        &format!("{blk}.ffn_up_exps.weight"),
        Role::FusedHalf { second: true },
    ));
    out.push(wanted(
        &format!("{layer}mlp.experts.down_proj"),
        &format!("{blk}.ffn_down_exps.weight"),
        Role::Stack,
    ));
    out
}

/// Where a tensor of a shard sits.
#[derive(Clone, Debug)]
struct Located {
    shard: String,
    /// Absolute offset of the tensor's first byte in the shard.
    start: u64,
    bytes: u64,
    dtype: String,
    shape: Vec<usize>,
}

/// The hub as range requests read it.
struct Hub {
    client: reqwest::blocking::Client,
    base: String,
}

impl Hub {
    fn new(endpoint: &str, repo: &str, revision: &str) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(|e| Error(format!("http client: {e}")))?;
        Ok(Self {
            client,
            base: format!(
                "{}/{repo}/resolve/{revision}/",
                endpoint.trim_end_matches('/')
            ),
        })
    }

    fn get(&self, file: &str, range: Option<(u64, u64)>) -> Result<Vec<u8>> {
        let url = format!("{}{file}", self.base);
        let mut last = String::new();
        for attempt in 0..ATTEMPTS {
            let mut req = self.client.get(&url);
            if let Some((a, b)) = range {
                req = req.header("Range", format!("bytes={a}-{}", b - 1));
            }
            match req
                .send()
                .and_then(|r| r.error_for_status())
                .and_then(|r| r.bytes())
            {
                Ok(bytes) => {
                    if let Some((a, b)) = range {
                        if bytes.len() as u64 != b - a {
                            last = format!("{} bytes of {}", bytes.len(), b - a);
                            continue;
                        }
                    }
                    return Ok(bytes.to_vec());
                }
                Err(e) => last = e.to_string(),
            }
            std::thread::sleep(std::time::Duration::from_secs(1 << attempt));
        }
        Err(Error(format!("reading {url}: {last}")))
    }

    /// The checkpoint's tensor-to-shard map.
    fn weight_map(&self) -> Result<serde_json::Map<String, serde_json::Value>> {
        let index: serde_json::Value =
            serde_json::from_slice(&self.get("model.safetensors.index.json", None)?)
                .map_err(|e| Error(format!("checkpoint index: {e}")))?;
        index["weight_map"]
            .as_object()
            .cloned()
            .ok_or_else(|| Error("checkpoint index has no weight_map".into()))
    }

    /// The tensors named `names`, found through the checkpoint's map and the shard headers.
    fn locate(
        &self,
        map: &serde_json::Map<String, serde_json::Value>,
        names: &[&str],
    ) -> Result<BTreeMap<String, Located>> {
        let mut by_shard: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for name in names {
            let shard = map
                .get(*name)
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error(format!("the checkpoint has no tensor {name}")))?;
            by_shard.entry(shard).or_default().push(name);
        }
        let mut found = BTreeMap::new();
        for (shard, wanted) in by_shard {
            let len = u64::from_le_bytes(
                self.get(shard, Some((0, 8)))?
                    .try_into()
                    .map_err(|_| Error("short safetensors header".into()))?,
            );
            let header: serde_json::Value =
                serde_json::from_slice(&self.get(shard, Some((8, 8 + len)))?)
                    .map_err(|e| Error(format!("{shard} header: {e}")))?;
            for name in wanted {
                let t = &header[name];
                let offsets = t["data_offsets"]
                    .as_array()
                    .filter(|o| o.len() == 2)
                    .ok_or_else(|| Error(format!("{shard} has no offsets for {name}")))?;
                let (a, b) = (
                    offsets[0].as_u64().unwrap_or(0),
                    offsets[1].as_u64().unwrap_or(0),
                );
                found.insert(
                    name.to_string(),
                    Located {
                        shard: shard.to_string(),
                        start: 8 + len + a,
                        bytes: b - a,
                        dtype: t["dtype"].as_str().unwrap_or("").to_string(),
                        shape: t["shape"]
                            .as_array()
                            .map(|s| {
                                s.iter()
                                    .filter_map(|d| d.as_u64())
                                    .map(|d| d as usize)
                                    .collect()
                            })
                            .unwrap_or_default(),
                    },
                );
            }
        }
        Ok(found)
    }
}

/// The released values as `f32`: the checkpoint stores 16-bit floats.
fn floats(dtype: &str, bytes: &[u8]) -> Result<Vec<f32>> {
    match dtype {
        "BF16" => Ok(bytes
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect()),
        "F16" => Ok(bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect()),
        "F32" => Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()),
        other => Err(Error(format!("tensor dtype {other} is not read"))),
    }
}

/// The format a matrix is written in: the widest k-quant its rows take whole blocks of, else the
/// widest legacy format they do.
fn matrix_dtype(cols: usize) -> GgmlDType {
    if cols.is_multiple_of(GgmlDType::Q4K.block_size()) {
        GgmlDType::Q4K
    } else {
        GgmlDType::Q5_0
    }
}

fn quantized_bytes(
    values: Vec<f32>,
    rows: usize,
    cols: usize,
    dtype: GgmlDType,
) -> Result<Vec<u8>> {
    let t = Tensor::from_vec(values, (rows, cols), &Device::Cpu)?;
    // CONVERSION: the draft-head converter encodes each released tensor once, offline, in the format the loader reads.
    let q = QTensor::quantize(&t, dtype)?;
    Ok(q.data()?.into_owned())
}

fn dtype_bytes(dtype: GgmlDType, elements: usize) -> u64 {
    (elements / dtype.block_size() * dtype.type_size()) as u64
}

/// What a planned tensor is written as: its dims (row-major), format and size.
fn entry_of(w: &Wanted, from: &Located) -> Result<PlannedEntry> {
    let cols = *from.shape.last().unwrap_or(&1);
    let (dims, dtype) = match w.role {
        Role::Exact | Role::Norm => {
            // A gate stored as one row is a vector.
            let dims: Vec<usize> = from.shape.iter().copied().filter(|&d| d != 1).collect();
            (if dims.is_empty() { vec![1] } else { dims }, GgmlDType::F32)
        }
        Role::Matrix => (from.shape.clone(), matrix_dtype(cols)),
        Role::Rows(a, b) => (vec![b - a, cols], GgmlDType::BF16),
        Role::FusedHalf { .. } => {
            let [experts, rows, cols] = from.shape[..] else {
                return Err(Error(format!("{}: fused experts are not 3-D", w.source)));
            };
            (vec![experts, rows / 2, cols], matrix_dtype(cols))
        }
        Role::Stack => (from.shape.clone(), matrix_dtype(cols)),
    };
    let elements: usize = dims.iter().product();
    Ok(PlannedEntry {
        name: w.name.clone(),
        byte_len: dtype_bytes(dtype, elements),
        dims,
        dtype,
    })
}

/// How many layers the model has besides the head: one more than the highest `.layers.N.` of a
/// tensor that is not the head's.
fn layer_count(map: &serde_json::Map<String, serde_json::Value>) -> Result<usize> {
    map.keys()
        .filter(|k| !k.starts_with("mtp."))
        .filter_map(|k| {
            let rest = k.split(".layers.").nth(1)?;
            rest.split('.').next()?.parse::<usize>().ok()
        })
        .max()
        .map(|m| m + 1)
        .ok_or_else(|| Error("the checkpoint names no layers".into()))
}

/// The parts of a stack, produced a window of them at a time on the worker threads and yielded
/// in order.
fn in_order<F>(chunks: usize, make: F) -> impl Iterator<Item = Result<Vec<u8>>>
where
    F: Fn(usize) -> Result<Vec<u8>> + Sync + Send,
{
    use rayon::prelude::*;
    let window = rayon::current_num_threads().max(1);
    (0..chunks).step_by(window).flat_map(move |first| {
        (first..(first + window).min(chunks))
            .into_par_iter()
            .map(&make)
            .collect::<Vec<_>>()
    })
}

/// Read the draft head of `repo` at `revision` from `endpoint` and write it to `out`.
pub fn fetch_mtp_head(endpoint: &str, repo: &str, revision: &str, out: &Path) -> Result<()> {
    let hub = Hub::new(endpoint, repo, revision)?;
    // The indexer's fused projection is split by the widths the layer's own indexer has, which
    // the shard says only as a total: the query side is the larger four fifths.
    let sources: Vec<String> = plan(0, 0, 0).into_iter().map(|w| w.source).collect();
    let names: Vec<&str> = sources.iter().map(String::as_str).collect();
    let map = hub.weight_map()?;
    let index = layer_count(&map)?;
    let located = hub.locate(&map, &names)?;
    let fused = located
        .get("mtp.layers.0.self_attn.indexer.index_qk_proj.weight")
        .ok_or_else(|| Error("the indexer projection is missing".into()))?;
    let total = fused.shape[0];
    let k_rows = total / 5;
    let wanted = plan(index, total - k_rows, k_rows);
    let entries: Vec<PlannedEntry> = wanted
        .iter()
        .map(|w| entry_of(w, &located[&w.source]))
        .collect::<Result<_>>()?;
    let metadata = vec![
        (
            "general.architecture".to_string(),
            Value::String("qwen4exp".to_string()),
        ),
        (
            "general.name".to_string(),
            Value::String("draft head".to_string()),
        ),
    ];
    let mut writer = GgufStreamWriter::create(out, &metadata, &entries)?;
    for (w, entry) in wanted.iter().zip(&entries) {
        let from = &located[&w.source];
        tracing::info!("draft head: {} ({} bytes)", w.name, entry.byte_len);
        match w.role {
            Role::Exact | Role::Norm | Role::Matrix | Role::Rows(..) => {
                let bytes = hub.get(&from.shard, Some((from.start, from.start + from.bytes)))?;
                let cols = *from.shape.last().unwrap_or(&1);
                let (bytes, rows) = match w.role {
                    Role::Rows(a, b) => {
                        let row = cols * 2;
                        (bytes[a * row..b * row].to_vec(), b - a)
                    }
                    _ => (bytes, from.bytes as usize / (cols * 2).max(1)),
                };
                let data = match entry.dtype {
                    GgmlDType::BF16 => bytes,
                    GgmlDType::F32 => {
                        let plus = if w.role == Role::Norm { 1.0 } else { 0.0 };
                        floats(&from.dtype, &bytes)?
                            .iter()
                            .flat_map(|v| (v + plus).to_le_bytes())
                            .collect()
                    }
                    dtype => quantized_bytes(floats(&from.dtype, &bytes)?, rows, cols, dtype)?,
                };
                writer.append(&data)?;
            }
            Role::FusedHalf { second } => {
                let [experts, rows, cols] = from.shape[..] else {
                    unreachable!("entry_of checked the shape")
                };
                let half = rows / 2;
                let per_expert = (rows * cols * 2) as u64;
                let half_bytes = (half * cols * 2) as u64;
                let offset = if second { half_bytes } else { 0 };
                let dtype = entry.dtype;
                let parts = in_order(experts.div_ceil(EXPERTS_PER_REQUEST), |chunk| {
                    let first = chunk * EXPERTS_PER_REQUEST;
                    let last = (first + EXPERTS_PER_REQUEST).min(experts);
                    let mut data = Vec::new();
                    for e in first..last {
                        let a = from.start + e as u64 * per_expert + offset;
                        let bytes = hub.get(&from.shard, Some((a, a + half_bytes)))?;
                        data.extend(quantized_bytes(
                            floats(&from.dtype, &bytes)?,
                            half,
                            cols,
                            dtype,
                        )?);
                    }
                    Ok(data)
                });
                writer.append_parts(parts)?;
            }
            Role::Stack => {
                let [experts, rows, cols] = from.shape[..] else {
                    return Err(Error(format!("{}: experts are not 3-D", w.source)));
                };
                let per_expert = (rows * cols * 2) as u64;
                let dtype = entry.dtype;
                let parts = in_order(experts.div_ceil(EXPERTS_PER_REQUEST), |chunk| {
                    let first = chunk * EXPERTS_PER_REQUEST;
                    let last = (first + EXPERTS_PER_REQUEST).min(experts);
                    let a = from.start + first as u64 * per_expert;
                    let b = from.start + last as u64 * per_expert;
                    let bytes = hub.get(&from.shard, Some((a, b)))?;
                    quantized_bytes(
                        floats(&from.dtype, &bytes)?,
                        (last - first) * rows,
                        cols,
                        dtype,
                    )
                });
                writer.append_parts(parts)?;
            }
        }
    }
    writer.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_name_of_the_head_is_planned_once() {
        let plan = plan(48, 512, 128);
        let mut names: Vec<_> = plan.iter().map(|w| w.name.as_str()).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), plan.len());
        // 31 released tensors: the indexer's fused projection becomes two and the fused experts
        // two, so the file has 33.
        assert_eq!(plan.len(), 33);
    }

    #[test]
    fn a_row_length_without_whole_k_blocks_takes_a_legacy_format() {
        assert_eq!(matrix_dtype(2560), GgmlDType::Q4K);
        assert_eq!(matrix_dtype(640), GgmlDType::Q5_0);
        assert_eq!(matrix_dtype(320), GgmlDType::Q5_0);
    }

    #[test]
    fn a_fused_half_is_sized_by_half_the_second_axis() {
        let w = wanted("x", "y", Role::FusedHalf { second: true });
        let from = Located {
            shard: "s".into(),
            start: 0,
            bytes: 0,
            dtype: "BF16".into(),
            shape: vec![4, 1280, 2560],
        };
        let e = entry_of(&w, &from).unwrap();
        assert_eq!(e.dims, vec![4, 640, 2560]);
        assert_eq!(e.dtype, GgmlDType::Q4K);
        assert_eq!(e.byte_len, 4 * 640 * 2560 / 256 * 144);
    }

    #[test]
    fn a_gate_stored_as_one_row_is_written_as_a_vector() {
        let w = wanted("x", "y", Role::Exact);
        let from = Located {
            shard: "s".into(),
            start: 0,
            bytes: 0,
            dtype: "BF16".into(),
            shape: vec![1, 2560],
        };
        assert_eq!(entry_of(&w, &from).unwrap().dims, vec![2560]);
    }

    #[test]
    fn the_layer_count_skips_the_head_and_the_non_layer_tensors() {
        let map: serde_json::Map<String, serde_json::Value> = [
            "model.language_model.layers.0.mlp.gate.weight",
            "model.language_model.layers.47.self_attn.q_proj.weight",
            "model.language_model.embed_tokens.weight",
            "mtp.layers.0.mlp.gate.weight",
        ]
        .iter()
        .map(|k| (k.to_string(), serde_json::Value::Null))
        .collect();
        assert_eq!(layer_count(&map).unwrap(), 48);
        assert!(layer_count(&serde_json::Map::new()).is_err());
    }
}
