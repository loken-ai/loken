//! The draft head a checkpoint ships beside its layers, read off the hub and written as released.
//!
//! A released checkpoint carries `mtp.*` tensors that the served quantisations leave out: the
//! two projections that fuse a token's embedding with the hidden state before it, and one more
//! layer of the model's own kind. This reads exactly those tensors with range requests, so the
//! shards they sit in are never downloaded whole, and writes them into a GGUF file as they are:
//! the checkpoint's names, its widths, its bytes. The loader reads that file under its own names
//! ([`checkpoint_listing`]) and gives the head the form it runs in.

use crate::tensor::gguf_write::{GgufStreamWriter, PlannedEntry};
use crate::tensor::quantized::gguf_file::Value;
use crate::tensor::quantized::GgmlDType;
use crate::tensor::{Error, Result};
use std::collections::BTreeMap;
use std::path::Path;

/// Bytes of a released tensor read per request.
const BYTES_PER_REQUEST: u64 = 64 << 20;
/// Attempts per request before the conversion gives up.
const ATTEMPTS: usize = 4;

/// How the loader reads a released tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    /// As stored: a router's scores.
    Exact,
    /// A matrix or a stack of them, held in the block format its rows take.
    Matrix,
    /// A norm scale, stored one less than the model's own file holds it.
    Norm,
    /// Rows `from..to` of the released tensor.
    Rows(usize, usize),
    /// The first or the second half of the rows of each matrix of a stack, held as a matrix.
    Half { second: bool },
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
    // Each expert's gate rows, then its up rows.
    out.push(wanted(
        &format!("{layer}mlp.experts.gate_up_proj"),
        &format!("{blk}.ffn_gate_exps.weight"),
        Role::Half { second: false },
    ));
    out.push(wanted(
        &format!("{layer}mlp.experts.gate_up_proj"),
        &format!("{blk}.ffn_up_exps.weight"),
        Role::Half { second: true },
    ));
    out.push(wanted(
        &format!("{layer}mlp.experts.down_proj"),
        &format!("{blk}.ffn_down_exps.weight"),
        Role::Matrix,
    ));
    out
}

/// How a head file carrying the checkpoint's own names and widths reads under the loader's: each
/// tensor the loader asks for over the stored tensor it is in, and the format it is held in. The
/// fused experts read as their gate and up halves; the indexer's fused projection as its query
/// rows then its `key_rows` key rows; norms are stored one less than the loader reads them. `None` when `stored` does not carry the checkpoint's names. `index` is the count of the
/// model's own layers.
pub fn checkpoint_listing(
    stored: &crate::tensor::quantized::gguf_file::Content,
    index: usize,
    key_rows: usize,
) -> Option<Vec<crate::tensor::quantized::gguf_source::Relisted>> {
    use crate::tensor::quantized::gguf_source::Relisted;
    let fused = stored
        .tensor_infos
        .get("mtp.layers.0.self_attn.indexer.index_qk_proj.weight")?;
    let total = *fused.shape.dims().first()?;
    let q_rows = total.checked_sub(key_rows)?;
    let mut out = Vec::new();
    for w in plan(index, q_rows, key_rows) {
        let dims = stored
            .tensor_infos
            .get(&w.source)
            .map(|t| t.shape.dims().to_vec())
            .unwrap_or_default();
        let cols = dims.last().copied().unwrap_or(0);
        let half = dims.len().checked_sub(2).map_or(0, |i| dims[i] / 2);
        out.push(Relisted {
            held: match w.role {
                Role::Half { .. } => stack_format(cols),
                Role::Matrix => matrix_format(cols),
                _ => None,
            },
            name: w.name,
            source: w.source,
            rows: match w.role {
                Role::Rows(a, b) => Some((a, b - a)),
                Role::Half { second } => Some((if second { half } else { 0 }, half)),
                _ => None,
            },
            plus_one: w.role == Role::Norm,
        });
    }
    out.iter()
        .all(|r| stored.tensor_infos.contains_key(&r.source))
        .then_some(out)
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

/// The format the head holds a matrix of rows of `cols` in: the widest k-quant the rows take
/// whole blocks of, else the widest legacy format they do. Its guesses are verified, so their
/// precision only sets how often one is kept: two bits on the down stack or the dense matrices
/// lose more guesses than the room they free wins back, and BF16 guessed no better than these.
fn matrix_format(cols: usize) -> Option<GgmlDType> {
    [GgmlDType::Q4K, GgmlDType::Q5_0]
        .into_iter()
        .find(|f| cols > 0 && cols.is_multiple_of(f.block_size()))
}

/// The format the head holds its experts' gate and up stacks in: two bits on the plain grid where
/// the rows take whole blocks of it, the room they free on the card worth more than the guesses
/// they cost; their down stack and the dense matrices keep [`matrix_format`].
fn stack_format(cols: usize) -> Option<GgmlDType> {
    match cols > 0 && cols.is_multiple_of(GgmlDType::Q2_0.block_size()) {
        true => Some(GgmlDType::Q2_0),
        false => matrix_format(cols),
    }
}

/// The GGUF width of a released dtype.
fn stored_dtype(dtype: &str) -> Result<GgmlDType> {
    match dtype {
        "BF16" => Ok(GgmlDType::BF16),
        "F16" => Ok(GgmlDType::F16),
        "F32" => Ok(GgmlDType::F32),
        other => Err(Error(format!("tensor dtype {other} is not read"))),
    }
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

/// Read the draft head of `repo` at `revision` from `endpoint` and write it to `out` as
/// released: each tensor under its checkpoint name, in its stored width, its bytes as they are.
pub fn fetch_mtp_head(endpoint: &str, repo: &str, revision: &str, out: &Path) -> Result<()> {
    let hub = Hub::new(endpoint, repo, revision)?;
    let mut sources: Vec<String> = plan(0, 0, 0).into_iter().map(|w| w.source).collect();
    sources.sort_unstable();
    sources.dedup();
    let names: Vec<&str> = sources.iter().map(String::as_str).collect();
    let located = hub.locate(&hub.weight_map()?, &names)?;
    let entries: Vec<PlannedEntry> = sources
        .iter()
        .map(|s| {
            let from = &located[s];
            Ok(PlannedEntry {
                name: s.clone(),
                byte_len: from.bytes,
                dims: from.shape.clone(),
                dtype: stored_dtype(&from.dtype)?,
            })
        })
        .collect::<Result<_>>()?;
    let metadata = vec![
        (
            "general.name".to_string(),
            Value::String("draft head".to_string()),
        ),
        (
            "general.source.huggingface.repository".to_string(),
            Value::String(repo.to_string()),
        ),
    ];
    let mut writer = GgufStreamWriter::create(out, &metadata, &entries)?;
    for source in &sources {
        let from = &located[source];
        tracing::info!("draft head: {source} ({} bytes)", from.bytes);
        let end = from.start + from.bytes;
        let parts = in_order(from.bytes.div_ceil(BYTES_PER_REQUEST) as usize, |i| {
            let a = from.start + i as u64 * BYTES_PER_REQUEST;
            hub.get(&from.shard, Some((a, (a + BYTES_PER_REQUEST).min(end))))
        });
        writer.append_parts(parts)?;
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
        // 31 released tensors: the indexer's fused projection and the fused experts each read as
        // two, so the loader asks for 33.
        assert_eq!(plan.len(), 33);
    }

    /// A head under the checkpoint's names lists every tensor the loader reads: the fused
    /// experts once, as one stack, the indexer's projection as its query and key rows, the norms
    /// stored one less.
    #[test]
    fn a_head_under_the_checkpoint_s_names_lists_under_the_loader_s() {
        use crate::tensor::quantized::gguf_file::{Content, TensorInfo};
        let mut stored = Content {
            magic: crate::tensor::quantized::gguf_file::VersionedMagic::GgufV3,
            metadata: Default::default(),
            tensor_infos: Default::default(),
            tensor_data_offset: 0,
            mmap_owner: None,
        };
        for w in plan(0, 0, 0) {
            let dims = if w.source.ends_with("index_qk_proj.weight") {
                vec![640, 2560]
            } else if w.source.ends_with("gate_up_proj") {
                vec![512, 1280, 2560]
            } else {
                vec![1]
            };
            stored.tensor_infos.insert(
                w.source,
                TensorInfo {
                    ggml_dtype: GgmlDType::BF16,
                    shape: dims.into(),
                    offset: 0,
                },
            );
        }
        let listing = checkpoint_listing(&stored, 48, 128).expect("the checkpoint's names");
        assert_eq!(listing.len(), plan(48, 512, 128).len());
        let find = |name: &str| listing.iter().find(|r| r.name == name).expect(name);
        assert_eq!(find("blk.48.indexer.q_proj.weight").rows, Some((0, 512)));
        assert_eq!(find("blk.48.indexer.k_proj.weight").rows, Some((512, 128)));
        let up = find("blk.48.ffn_up_exps.weight");
        assert_eq!(up.source, "mtp.layers.0.mlp.experts.gate_up_proj");
        assert_eq!(up.rows, Some((640, 640)));
        assert_eq!(up.held, Some(GgmlDType::Q2_0));
        assert_eq!(find("blk.48.ffn_gate_exps.weight").rows, Some((0, 640)));
        assert_eq!(find("blk.48.ffn_gate_inp.weight").held, None);
        assert!(find("mtp.pre_fc_norm_hidden.weight").plus_one);
        assert!(!find("blk.48.attn_q.weight").plus_one);
        stored.tensor_infos.clear();
        assert!(checkpoint_listing(&stored, 48, 128).is_none());
    }
}
