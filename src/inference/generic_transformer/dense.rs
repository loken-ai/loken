//! A dense safetensors checkpoint presented as the tensors of a GGUF file.
//!
//! The checkpoint's shards are mapped and each tensor is listed under the name the generic
//! loader asks for (`blk.N.attn_q.weight`, `token_embd.weight`) with its stored width: F32, F16
//! or BF16 are read in place, never widened. The loader then builds the model exactly as it
//! builds one from a GGUF file. What differs is carried by the checkpoint itself: its rows are
//! in the order the framework wrote them (see `hf_rows` in the loader), and a long-context rope
//! is declared in the configuration where a GGUF file stores the table, so the table is
//! computed here.

use crate::inference::load::awq_loader::{frequency_factors, AwqConfigJson};
use crate::tensor::quantized::gguf_file::{Content, TensorInfo, VersionedMagic};
use crate::tensor::quantized::GgmlDType;
use crate::tensor::{Error, Result, Shape};
use std::collections::HashMap;
use std::sync::Arc;

/// One mapped shard, its tensors listed under loader names.
pub(super) struct DensePart {
    pub content: Content,
    pub mmap: Arc<memmap2::Mmap>,
}

/// The shards of a checkpoint, and the header the loader reads the model from.
pub struct DenseCheckpoint {
    pub(super) parts: Vec<DensePart>,
    /// Bytes of tensors the checkpoint does not store but the loader reads, with the part that
    /// lists them.
    pub(super) computed: (Content, Vec<u8>),
    /// The metadata of `base` and every tensor of every part.
    pub content: Content,
    /// Rotary lanes are paired as a GGUF file pairs them (adjacent), which a vendor's native
    /// layout does; the framework's layout pairs each lane with the one half a head away.
    pub adjacent_rope: bool,
}

/// The loader's name for a layer tensor of the checkpoint's `part` (`self_attn.q_proj.weight`).
fn layer_name(part: &str) -> Option<&'static str> {
    Some(match part {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_proj.bias" => "attn_q.bias",
        "self_attn.k_proj.bias" => "attn_k.bias",
        "self_attn.v_proj.bias" => "attn_v.bias",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        _ => return None,
    })
}

/// How a checkpoint names its text model's tensors.
enum Naming {
    /// The framework's names, under `model` (`model.`, or that behind a wrapper's prefix) and
    /// with the head under `head`.
    Framework { model: String, head: String },
    /// The vendor's own: `tok_embeddings.weight`, `layers.N.attention.wq.weight`.
    Native,
}

/// The loader's name for a layer tensor of a native layout's `part` (`attention.wq.weight`).
fn native_layer_name(part: &str) -> Option<&'static str> {
    Some(match part {
        "attention_norm.weight" => "attn_norm.weight",
        "ffn_norm.weight" => "ffn_norm.weight",
        "attention.wq.weight" => "attn_q.weight",
        "attention.wk.weight" => "attn_k.weight",
        "attention.wv.weight" => "attn_v.weight",
        "attention.wo.weight" => "attn_output.weight",
        "attention.wq.bias" => "attn_q.bias",
        "attention.wk.bias" => "attn_k.bias",
        "attention.wv.bias" => "attn_v.bias",
        "feed_forward.w1.weight" => "ffn_gate.weight",
        "feed_forward.w3.weight" => "ffn_up.weight",
        "feed_forward.w2.weight" => "ffn_down.weight",
        _ => return None,
    })
}

/// Where the checkpoint keeps its text model: for the framework's names, the prefix of
/// `embed_tokens.weight` and the prefix its head would have.
fn naming(names: &[String], native: bool) -> Result<Naming> {
    if native {
        return Ok(Naming::Native);
    }
    let embed = names
        .iter()
        .find(|n| n.ends_with("embed_tokens.weight"))
        .ok_or_else(|| Error("the checkpoint has no embed_tokens.weight".to_string()))?;
    let model = embed.trim_end_matches("embed_tokens.weight").to_string();
    let head = model.strip_suffix("model.").unwrap_or(&model).to_string();
    Ok(Naming::Framework { model, head })
}

fn canonical(naming: &Naming, name: &str) -> Option<String> {
    let (model, head) = match naming {
        Naming::Native => {
            return match name {
                // The vendor's multimodal release keeps the table in the tower's namespace.
                "tok_embeddings.weight" | "mm_whisper_embeddings.tok_embeddings.weight" => {
                    Some("token_embd.weight".into())
                }
                "norm.weight" => Some("output_norm.weight".into()),
                "output.weight" => Some("output.weight".into()),
                _ => {
                    let (n, part) = name.strip_prefix("layers.")?.split_once('.')?;
                    n.parse::<usize>().ok()?;
                    Some(format!("blk.{n}.{}", native_layer_name(part)?))
                }
            };
        }
        Naming::Framework { model, head } => (model, head),
    };
    if name == format!("{model}embed_tokens.weight") {
        return Some("token_embd.weight".into());
    }
    if name == format!("{model}norm.weight") {
        return Some("output_norm.weight".into());
    }
    if name == format!("{head}lm_head.weight") {
        return Some("output.weight".into());
    }
    let rest = name.strip_prefix(model.as_str())?.strip_prefix("layers.")?;
    let (n, part) = rest.split_once('.')?;
    n.parse::<usize>().ok()?;
    Some(format!("blk.{n}.{}", layer_name(part)?))
}

/// Bytes the text model's blocks take as stored, and bytes its head takes uploaded as F16 (the
/// embedding table standing in for an absent head), for a checkpoint's mapped shards. Tensors
/// of other parts of a composite model (a speech tower) count for neither.
pub fn text_bytes(
    st: &crate::tensor::safetensors::MmapedSafetensors,
    native: bool,
) -> Option<(u64, u64)> {
    let names: Vec<String> = st.names().iter().map(|n| n.to_string()).collect();
    let naming = naming(&names, native).ok()?;
    let block_bytes: u64 = st
        .byte_sizes()
        .iter()
        .filter(|(n, _)| canonical(&naming, n).is_some_and(|c| c.starts_with("blk.")))
        .map(|(_, b)| *b as u64)
        .sum();
    let holder = |want: &str| {
        names
            .iter()
            .find(|n| canonical(&naming, n).as_deref() == Some(want))
    };
    let head = holder("output.weight").or_else(|| holder("token_embd.weight"))?;
    let head_bytes = st
        .shape_of(head)
        .map_or(0, |d| d.iter().product::<usize>() as u64 * 2);
    Some((block_bytes, head_bytes))
}

fn dtype_of(s: &str) -> Option<GgmlDType> {
    Some(match s {
        "F32" => GgmlDType::F32,
        "F16" => GgmlDType::F16,
        "BF16" => GgmlDType::BF16,
        _ => return None,
    })
}

/// The tensors of one shard: name, stored dtype, dimensions and `[start, end)` in the data.
type Listing = Vec<(String, String, Vec<usize>, u64)>;

fn read_header(mmap: &memmap2::Mmap, path: &str) -> Result<(Listing, u64)> {
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

impl DenseCheckpoint {
    /// Map `shards` and list their tensors under the loader's names. `base` carries the
    /// metadata the model is configured from; `cfg` the rope scaling the table is computed for.
    pub fn open(base: &Content, shards: &[String], cfg: &AwqConfigJson) -> Result<Self> {
        let mut mapped = Vec::new();
        let mut all_names = Vec::new();
        for path in shards {
            let file = std::fs::File::open(path).map_err(|e| Error(format!("{path}: {e}")))?;
            let mmap =
                unsafe { memmap2::Mmap::map(&file) }.map_err(|e| Error(format!("{path}: {e}")))?;
            let (listing, data_start) = read_header(&mmap, path)?;
            all_names.extend(listing.iter().map(|l| l.0.clone()));
            mapped.push((Arc::new(mmap), listing, data_start));
        }
        let naming = naming(&all_names, cfg.native)?;
        let mut parts = Vec::new();
        let mut union: HashMap<String, TensorInfo> = HashMap::new();
        for ((mmap, listing, data_start), path) in mapped.into_iter().zip(shards) {
            let mut infos = HashMap::new();
            for (name, dtype, shape, start) in listing {
                let Some(canon) = canonical(&naming, &name) else {
                    continue;
                };
                let ggml_dtype = dtype_of(&dtype).ok_or_else(|| {
                    Error(format!(
                        "{path}: {name} is {dtype}; dense checkpoints are F32, F16 or BF16"
                    ))
                })?;
                let info = TensorInfo {
                    ggml_dtype,
                    shape: Shape::from(shape),
                    offset: start,
                };
                union.insert(canon.clone(), info_clone(&info));
                infos.insert(canon, info);
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
        // The rope table a GGUF file stores, computed from the configuration's scaling.
        let mut computed_infos = HashMap::new();
        let mut bytes = Vec::new();
        if let Some(scaling) = cfg.frequency_scaling {
            let head_dim = cfg.head_dim.unwrap_or(cfg.hidden_size / cfg.n_head);
            let factors = frequency_factors(head_dim, cfg.rope_theta, scaling);
            bytes = factors.iter().flat_map(|f| f.to_le_bytes()).collect();
            let info = TensorInfo {
                ggml_dtype: GgmlDType::F32,
                shape: Shape::from(vec![factors.len()]),
                offset: 0,
            };
            union.insert("rope_freqs.weight".into(), info_clone(&info));
            computed_infos.insert("rope_freqs.weight".to_string(), info);
        }
        let computed = Content {
            magic: VersionedMagic::GgufV3,
            metadata: HashMap::new(),
            tensor_infos: computed_infos,
            tensor_data_offset: 0,
            mmap_owner: None,
        };
        let content = Content {
            magic: base.magic,
            metadata: base.metadata.clone(),
            tensor_infos: union,
            tensor_data_offset: 0,
            mmap_owner: None,
        };
        Ok(Self {
            parts,
            computed: (computed, bytes),
            content,
            adjacent_rope: cfg.native,
        })
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

    #[test]
    fn checkpoint_names_become_loader_names() {
        let fw = Naming::Framework {
            model: "model.".into(),
            head: String::new(),
        };
        let c = |n: &str| canonical(&fw, n);
        assert_eq!(
            c("model.embed_tokens.weight").as_deref(),
            Some("token_embd.weight")
        );
        assert_eq!(
            c("model.layers.3.self_attn.o_proj.weight").as_deref(),
            Some("blk.3.attn_output.weight")
        );
        assert_eq!(
            c("model.layers.3.mlp.up_proj.weight").as_deref(),
            Some("blk.3.ffn_up.weight")
        );
        assert_eq!(c("lm_head.weight").as_deref(), Some("output.weight"));
        assert_eq!(c("audio_tower.conv1.weight"), None);
        let wrapped = Naming::Framework {
            model: "language_model.model.".into(),
            head: "language_model.".into(),
        };
        assert_eq!(
            canonical(&wrapped, "language_model.lm_head.weight").as_deref(),
            Some("output.weight")
        );
        assert_eq!(
            canonical(&wrapped, "language_model.model.norm.weight").as_deref(),
            Some("output_norm.weight")
        );
        let n = |name: &str| canonical(&Naming::Native, name);
        assert_eq!(
            n("tok_embeddings.weight").as_deref(),
            Some("token_embd.weight")
        );
        assert_eq!(n("output.weight").as_deref(), Some("output.weight"));
        assert_eq!(
            n("layers.4.attention.wo.weight").as_deref(),
            Some("blk.4.attn_output.weight")
        );
        assert_eq!(
            n("layers.4.feed_forward.w3.weight").as_deref(),
            Some("blk.4.ffn_up.weight")
        );
        assert_eq!(
            n("mm_whisper_embeddings.tok_embeddings.weight").as_deref(),
            Some("token_embd.weight")
        );
        assert_eq!(
            n("mm_whisper_embeddings.audio_language_projection.0.weight"),
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
