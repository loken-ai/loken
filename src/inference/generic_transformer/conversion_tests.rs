//! What a load keeps, against what the file stores.
//!
//! A synthetic dense model is written once per dtype the loader supports, loaded on the host,
//! and every tensor class is read back: the resident format must be the stored one, unless a
//! conversion for that class and format is declared in `DECLARED` with its reason. A load that
//! widens, requantises or copies a class the table does not name fails here.

use super::*;
use crate::tensor::quantized::gguf_file::Value;
use crate::tensor::quantized::GgmlDType;

const HIDDEN: usize = 256;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const FFN: usize = 512;
const VOCAB: usize = 64;
const BLOCKS: usize = 2;

/// Every conversion a dense host load that fits performs: `(class, stored, resident, reason)`.
/// A stored format a class keeps does not appear: a float checkpoint that fits keeps its
/// projections and its output head as stored. The reason is what the loader's ledger records
/// for the same change.
const DECLARED: &[(&str, &str, &str, &str)] = &[];

/// The reason the ledger records for a float projection rebuilt to fit.
const REQUANT_REASON: &str =
    "float projection rebuilt as a block format for the block-quant kernels and fused projections";

/// Every conversion a load performs on a float checkpoint that does not fit as stored:
/// `(class, stored, resident)`, each recorded with [`REQUANT_REASON`].
const REQUANTISED: &[(&str, &str, &str)] = &[
    ("projection", "f32", "q8_0"),
    ("projection", "f16", "q8_0"),
    ("projection", "bf16", "q8_0"),
    ("projection", "f32", "q4_K"),
    ("projection", "f16", "q4_K"),
    ("projection", "bf16", "q4_K"),
    ("projection", "f32", "q6_K"),
    ("projection", "f16", "q6_K"),
    ("projection", "bf16", "q6_K"),
];

/// The formats the loader accepts for a weight, by the name GGUF spells them with.
const STORED_FORMATS: [GgmlDType; 14] = [
    GgmlDType::F32,
    GgmlDType::F16,
    GgmlDType::BF16,
    GgmlDType::Q4_0,
    GgmlDType::Q4_1,
    GgmlDType::Q5_0,
    GgmlDType::Q5_1,
    GgmlDType::Q8_0,
    GgmlDType::Q2K,
    GgmlDType::Q3K,
    GgmlDType::Q4K,
    GgmlDType::Q5K,
    GgmlDType::Q6K,
    GgmlDType::MxFp4,
];

fn values(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut x = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * amp
        })
        .collect()
}

/// The tensors of the dense model of two blocks, by their GGUF names: dimensions outer first
/// and values.
fn dense_tensors() -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let mut tensors: Vec<(String, Vec<usize>)> = vec![
        ("token_embd.weight".into(), vec![VOCAB, HIDDEN]),
        ("output_norm.weight".into(), vec![HIDDEN]),
        ("output.weight".into(), vec![VOCAB, HIDDEN]),
    ];
    for il in 0..BLOCKS {
        tensors.push((format!("blk.{il}.attn_norm.weight"), vec![HIDDEN]));
        tensors.push((
            format!("blk.{il}.attn_q.weight"),
            vec![HEADS * HEAD_DIM, HIDDEN],
        ));
        tensors.push((
            format!("blk.{il}.attn_k.weight"),
            vec![KV_HEADS * HEAD_DIM, HIDDEN],
        ));
        tensors.push((
            format!("blk.{il}.attn_v.weight"),
            vec![KV_HEADS * HEAD_DIM, HIDDEN],
        ));
        tensors.push((
            format!("blk.{il}.attn_output.weight"),
            vec![HIDDEN, HEADS * HEAD_DIM],
        ));
        tensors.push((format!("blk.{il}.ffn_norm.weight"), vec![HIDDEN]));
        tensors.push((format!("blk.{il}.ffn_gate.weight"), vec![FFN, HIDDEN]));
        tensors.push((format!("blk.{il}.ffn_up.weight"), vec![FFN, HIDDEN]));
        tensors.push((format!("blk.{il}.ffn_down.weight"), vec![HIDDEN, FFN]));
    }
    tensors
        .into_iter()
        .enumerate()
        .map(|(i, (name, dims))| {
            let n: usize = dims.iter().product();
            let vals: Vec<f32> = if is_norm(&name) {
                values(n, i as u64, 0.1)
                    .into_iter()
                    .map(|x| 1.0 + x)
                    .collect()
            } else {
                values(n, i as u64 + 1, 1.5 / (*dims.last().unwrap() as f32).sqrt())
            };
            (name, dims, vals)
        })
        .collect()
}

/// Whether the tensor `name` is a norm scale, which every file stores F32.
fn is_norm(name: &str) -> bool {
    name.ends_with("norm.weight")
}

/// A dense model of two blocks whose every weight is stored as `dtype`; the norms are F32.
fn write_dense(path: &std::path::Path, dtype: GgmlDType) -> Result<()> {
    write_dense_values_in(path, dtype, dtype)
}

/// [`write_dense`] with the values projections stored in `values`.
fn write_dense_values_in(
    path: &std::path::Path,
    dtype: GgmlDType,
    values: GgmlDType,
) -> Result<()> {
    let meta = vec![
        (
            "general.architecture".to_string(),
            Value::String("llama".into()),
        ),
        (
            "llama.attention.head_count".to_string(),
            Value::U32(HEADS as u32),
        ),
        (
            "llama.attention.head_count_kv".to_string(),
            Value::U32(KV_HEADS as u32),
        ),
        (
            "llama.attention.layer_norm_rms_epsilon".to_string(),
            Value::F32(1e-5),
        ),
        ("llama.block_count".to_string(), Value::U32(BLOCKS as u32)),
        (
            "llama.embedding_length".to_string(),
            Value::U32(HIDDEN as u32),
        ),
        (
            "llama.feed_forward_length".to_string(),
            Value::U32(FFN as u32),
        ),
        ("llama.context_length".to_string(), Value::U32(512)),
    ];
    let mut entries = Vec::new();
    for (name, dims, vals) in dense_tensors() {
        let stored = if is_norm(&name) {
            GgmlDType::F32
        } else if name.ends_with("attn_v.weight") {
            values
        } else {
            dtype
        };
        let data = crate::tensor::quant_cpu::from_float_bytes(stored, &vals)?;
        entries.push(crate::tensor::gguf_write::GgufEntry {
            name,
            dims,
            dtype: stored,
            data,
        });
    }
    crate::tensor::gguf_write::write_gguf_with_metadata(path, &meta, &entries)
}

fn load(path: &std::path::Path) -> Result<GenericHeteroTransformer> {
    let file = std::fs::File::open(path).unwrap();
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file).unwrap() });
    let content =
        gguf_file::Content::read_mapped(&mut std::io::Cursor::new(&mmap[..]), mmap.clone())
            .unwrap();
    let plan = crate::inference::place::layer_executor::HeteroPlan::forced_gpu(BLOCKS, 0, 0);
    GenericHeteroTransformer::from_gguf(content, &mmap[..], &HashMap::new(), &plan)
}

fn stored_bytes(dtype: GgmlDType, elems: usize) -> u64 {
    (elems / dtype.block_size() * dtype.type_size()) as u64
}

#[test]
fn a_load_keeps_every_class_in_its_stored_format_unless_declared() {
    let dir = std::env::temp_dir().join(format!("loken-conv-matrix-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut problems = Vec::new();
    for dtype in STORED_FORMATS {
        let path = dir.join(format!("{}.gguf", dtype.gguf_name()));
        if write_dense(&path, dtype).is_err() {
            problems.push(format!(
                "{}: the synthetic file could not be written",
                dtype.gguf_name()
            ));
            continue;
        }
        let m = match load(&path) {
            Ok(m) => m,
            Err(e) => {
                problems.push(format!("{}: load failed: {e}", dtype.gguf_name()));
                continue;
            }
        };
        let stored = dtype.gguf_name();
        // The embedding is read from the mapping and owns nothing.
        if !m.embeddings.is_stored() || m.embeddings.owned_bytes() != 0 {
            problems.push(format!("{stored}: the embedding table is widened"));
        }
        let l = &m.layers[0];
        let classes: Vec<(&str, &str, &QMatMul)> = [
            ("attn_q", l.attn_q.as_ref()),
            ("attn_k", l.attn_k.as_ref()),
            ("attn_v", l.attn_v.as_ref()),
            ("attn_qkv", l.attn_qkv.as_ref()),
            ("attn_output", Some(&l.attn_output)),
            ("ffn_gate", l.ffn_gate.as_ref()),
            ("ffn_up", Some(&l.ffn_up)),
            ("ffn_down", Some(&l.ffn_down)),
        ]
        .into_iter()
        .filter_map(|(name, qm)| qm.map(|q| ("projection", name, q)))
        .collect();
        assert!(
            classes.len() >= 3,
            "{stored}: the projections were not found: {:?}",
            classes.iter().map(|c| c.1).collect::<Vec<_>>()
        );
        let head_declared = |got: &str| {
            DECLARED.iter().any(|(c, from, to, why)| {
                *c == "output" && *from == stored && *to == got && !why.is_empty()
            })
        };
        let Some(host_head) = m.output_proj.as_ref() else {
            panic!("{stored}: the host copy of the output head is missing");
        };
        match host_head {
            crate::tensor::quantized::QMatMul::QTensor(q) if q.dtype() == dtype => {}
            crate::tensor::quantized::QMatMul::Tensor(t)
            | crate::tensor::quantized::QMatMul::TensorF16(t)
            | crate::tensor::quantized::QMatMul::Bf16(t)
                if t.dtype().as_str() == stored || head_declared(t.dtype().as_str()) => {}
            other => problems.push(format!(
                "{stored}: the output head is resident as {other:?}"
            )),
        }
        for (class, name, qm) in classes {
            let (got, _owned) = qm.resident_format();
            if got == stored {
                continue;
            }
            let declared = DECLARED.iter().any(|(c, from, to, why)| {
                *c == class && *from == stored && *to == got && !why.is_empty()
            });
            if !declared {
                problems.push(format!(
                    "{stored}: {name} is resident as {got}, an undeclared conversion"
                ));
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// The float formats a checkpoint is served in as stored.
const FLOAT_FORMATS: [GgmlDType; 3] = [GgmlDType::F32, GgmlDType::F16, GgmlDType::BF16];

/// The logits of a prefill of four tokens and of one decoded after them.
fn prefill_and_step(m: &mut GenericHeteroTransformer) -> Vec<f32> {
    let prompt = Tensor::new(&[3u32, 17, 42, 5][..], &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let next = Tensor::new(&[9u32][..], &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let mut out = Vec::new();
    for (ids, pos) in [(prompt, 0), (next, 4)] {
        let logits = m.forward(&ids, pos).unwrap();
        let logits = logits
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap();
        out.extend(logits.to_vec1::<f32>().unwrap());
    }
    out
}

/// Fails unless `got` is finite and within the rounding of the half formats of `want`, the
/// logits of the F32 checkpoint on the host.
///
/// The half formats round each weight to 11 or 8 significant bits.
fn assert_near_the_f32_checkpoint(got: &[f32], want: &[f32], what: &str) {
    assert!(
        got.iter().all(|v| v.is_finite()),
        "{what}: a logit is not finite"
    );
    assert_eq!(got.len(), want.len(), "{what}: logit count");
    let scale = want.iter().fold(0f32, |a, v| a.max(v.abs()));
    let err = got
        .iter()
        .zip(want)
        .fold(0f32, |a, (g, w)| a.max((g - w).abs()));
    assert!(
        err <= scale * 0.05,
        "{what}: logits off by {err} of {scale} from the f32 checkpoint"
    );
}

#[test]
fn a_float_checkpoint_that_fits_is_served_as_stored() {
    let dir = std::env::temp_dir().join(format!("loken-conv-stored-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut reference: Option<Vec<f32>> = None;
    for dtype in FLOAT_FORMATS {
        let path = dir.join(format!("{}.gguf", dtype.gguf_name()));
        write_dense(&path, dtype).unwrap();
        let mut m = load(&path).unwrap();
        assert_eq!(
            m.running_format().as_deref(),
            Some(dtype.gguf_name()),
            "{} checkpoint",
            dtype.gguf_name()
        );
        let got = prefill_and_step(&mut m);
        match &reference {
            None => {
                assert_near_the_f32_checkpoint(&got, &got, "f32");
                reference = Some(got);
            }
            Some(want) => assert_near_the_f32_checkpoint(&got, want, dtype.gguf_name()),
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The logits of the F32 checkpoint written in `dir`, every layer on the host.
#[cfg(any(feature = "gpu", feature = "vulkan"))]
fn host_reference(dir: &std::path::Path) -> Vec<f32> {
    let path = dir.join("reference-f32.gguf");
    write_dense(&path, GgmlDType::F32).unwrap();
    prefill_and_step(&mut load(&path).unwrap())
}

/// The model in `path` with its layers placed as `plan` says, on `cards`.
#[cfg(any(feature = "gpu", feature = "vulkan"))]
fn load_placed(
    path: &std::path::Path,
    cards: &HashMap<usize, Device>,
    plan: &HeteroPlan,
) -> Result<GenericHeteroTransformer> {
    let file = std::fs::File::open(path).unwrap();
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file).unwrap() });
    let content =
        gguf_file::Content::read_mapped(&mut std::io::Cursor::new(&mmap[..]), mmap.clone())
            .unwrap();
    GenericHeteroTransformer::from_gguf(content, &mmap[..], cards, plan)
}

/// The checkpoint name of a tensor [`dense_tensors`] names.
#[cfg(any(feature = "gpu", feature = "vulkan"))]
fn checkpoint_name(name: &str) -> String {
    match name {
        "token_embd.weight" => "model.embed_tokens.weight".into(),
        "output.weight" => "lm_head.weight".into(),
        "output_norm.weight" => "model.norm.weight".into(),
        _ => {
            let (il, rest) = name
                .strip_prefix("blk.")
                .and_then(|r| r.split_once('.'))
                .unwrap();
            let hf = match rest {
                "attn_norm.weight" => "input_layernorm.weight",
                "ffn_norm.weight" => "post_attention_layernorm.weight",
                "attn_q.weight" => "self_attn.q_proj.weight",
                "attn_k.weight" => "self_attn.k_proj.weight",
                "attn_v.weight" => "self_attn.v_proj.weight",
                "attn_output.weight" => "self_attn.o_proj.weight",
                "ffn_gate.weight" => "mlp.gate_proj.weight",
                "ffn_up.weight" => "mlp.up_proj.weight",
                "ffn_down.weight" => "mlp.down_proj.weight",
                other => panic!("no checkpoint name for {other}"),
            };
            format!("model.layers.{il}.{hf}")
        }
    }
}

/// The dense model as a Llama checkpoint directory `dir` of one safetensors file, every weight
/// stored as `dtype` after rounding to `through` when it is given. The norms are F32.
#[cfg(any(feature = "gpu", feature = "vulkan"))]
fn write_checkpoint(
    dir: &std::path::Path,
    dtype: GgmlDType,
    through: Option<GgmlDType>,
) -> std::path::PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let round = |v: f32| match through {
        Some(GgmlDType::BF16) => half::bf16::from_f32(v).to_f32(),
        Some(GgmlDType::F16) => half::f16::from_f32(v).to_f32(),
        _ => v,
    };
    let stored: Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)> = dense_tensors()
        .into_iter()
        .map(|(name, dims, vals)| {
            let norm = is_norm(&name);
            let vals = vals.into_iter().map(|v| if norm { v } else { round(v) });
            let (st, bytes): (_, Vec<u8>) = match dtype {
                _ if norm => (
                    safetensors::Dtype::F32,
                    vals.flat_map(f32::to_le_bytes).collect(),
                ),
                GgmlDType::BF16 => (
                    safetensors::Dtype::BF16,
                    vals.flat_map(|v| half::bf16::from_f32(v).to_le_bytes())
                        .collect(),
                ),
                GgmlDType::F16 => (
                    safetensors::Dtype::F16,
                    vals.flat_map(|v| half::f16::from_f32(v).to_le_bytes())
                        .collect(),
                ),
                _ => (
                    safetensors::Dtype::F32,
                    vals.flat_map(f32::to_le_bytes).collect(),
                ),
            };
            (checkpoint_name(&name), st, dims, bytes)
        })
        .collect();
    let views: Vec<_> = stored
        .iter()
        .map(|(name, st, dims, bytes)| {
            let view = safetensors::tensor::TensorView::new(*st, dims.clone(), bytes).unwrap();
            (name.clone(), view)
        })
        .collect();
    safetensors::serialize_to_file(views, None, &dir.join("model.safetensors")).unwrap();
    let config = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": HIDDEN,
        "intermediate_size": FFN,
        "num_hidden_layers": BLOCKS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "max_position_embeddings": 512,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": false
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    dir.to_path_buf()
}

/// The checkpoint directory `dir` loaded as the server loads one, its layers placed as `plan`
/// says, on `cards`.
#[cfg(any(feature = "gpu", feature = "vulkan"))]
fn load_checkpoint(
    dir: &std::path::Path,
    cards: &HashMap<usize, Device>,
    plan: &HeteroPlan,
) -> Result<GenericHeteroTransformer> {
    let (_, shards) = crate::inference::load::awq_loader::checkpoint_shards(dir)
        .map_err(|e| crate::tensor::Error::msg(e.to_string()))?;
    crate::inference::engine::llm_engine::GenericWeights::Awq {
        dir: &dir.to_string_lossy(),
        shards: &shards,
    }
    .build(
        cards,
        plan,
        crate::inference::engine::llm_engine::KvQuant::Off,
        None,
        false,
        false,
    )
}

/// The largest distance between two runs' logits.
#[cfg(any(feature = "gpu", feature = "vulkan"))]
fn max_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// The F32, F16 and BF16 checkpoint directories of the dense model, placed as `plan` says on
/// `cards`, run in their stored format and stay within the half formats' rounding of the host.
///
/// The BF16 one's logits are as close to the host's logits of the same BF16 weights as those
/// of its F16 copy, or closer.
/// Equal is equal to one F32 rounding of the largest logit: the two runs sum in their own order.
/// `placed` says whether the device under test holds the layers.
#[cfg(any(feature = "gpu", feature = "vulkan"))]
fn checkpoints_run_as_stored(
    dir: &std::path::Path,
    cards: &HashMap<usize, Device>,
    plan: &HeteroPlan,
    what: &str,
    placed: &dyn Fn(&GenericHeteroTransformer) -> bool,
) {
    let host = HeteroPlan::forced_gpu(BLOCKS, 0, 0);
    let on_host = |d: std::path::PathBuf| {
        prefill_and_step(&mut load_checkpoint(&d, &HashMap::new(), &host).unwrap())
    };
    let on_device = |d: std::path::PathBuf, format: &str| {
        let mut m = load_checkpoint(&d, cards, plan).unwrap();
        assert!(
            placed(&m),
            "{what} {format} directory: a layer is not on the device"
        );
        let running = m.running_format();
        eprintln!("{what}: {format} directory runs as {running:?}");
        assert_eq!(
            running.as_deref(),
            Some(format),
            "{what} {format} directory"
        );
        prefill_and_step(&mut m)
    };
    let want = on_host(write_checkpoint(&dir.join("st-ref"), GgmlDType::F32, None));
    for dtype in FLOAT_FORMATS {
        let name = dtype.gguf_name();
        let got = on_device(
            write_checkpoint(&dir.join(format!("st-{name}")), dtype, None),
            name,
        );
        assert_near_the_f32_checkpoint(&got, &want, &format!("{what} {name} directory"));
    }
    let bf16 = write_checkpoint(&dir.join("st-bf16"), GgmlDType::BF16, None);
    let exact = on_host(bf16.clone());
    let stored = max_distance(&on_device(bf16, "bf16"), &exact);
    let f16_copy = write_checkpoint(
        &dir.join("st-f16-of-bf16"),
        GgmlDType::F16,
        Some(GgmlDType::BF16),
    );
    let copied = max_distance(&on_device(f16_copy, "f16"), &exact);
    eprintln!("{what}: bf16 weights off the host by {stored} as bf16, {copied} as f16");
    let ulp = exact.iter().fold(0f32, |a, v| a.max(v.abs())) * f32::EPSILON;
    assert!(
        stored <= copied + ulp,
        "{what}: bf16 weights run as bf16 are off the host by {stored}, as f16 by {copied}"
    );
}

/// A float checkpoint whose every layer a card holds runs there in its stored format, and its
/// logits stay within the half formats' rounding of the host's.
///
/// It covers GGUF files and checkpoint directories.
/// It runs on card 0 of the runtime the build carries: CUDA, or HIP for a ROCm build.
#[cfg(feature = "gpu")]
#[test]
fn a_float_checkpoint_on_a_card_runs_as_stored() {
    let Ok(card) = Device::new_cuda(0) else {
        eprintln!("skipping: no card");
        return;
    };
    let cards: HashMap<usize, Device> = [(0usize, card)].into_iter().collect();
    let total = crate::inference::place::device_probe::probe_cuda_gpus()
        .first()
        .map_or(0, |g| g.total >> 20);
    eprintln!("card 0 holds {total} MiB");
    let dir = std::env::temp_dir().join(format!("loken-conv-card-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let want = host_reference(&dir);
    let plan = HeteroPlan::forced_gpu(BLOCKS, BLOCKS, 0);
    for dtype in FLOAT_FORMATS {
        let path = dir.join(format!("{}.gguf", dtype.gguf_name()));
        write_dense(&path, dtype).unwrap();
        let mut m = load_placed(&path, &cards, &plan).unwrap();
        assert!(
            m.layers.iter().all(|l| l.attn_output.device().is_cuda()),
            "{}: a layer is not on the card",
            dtype.gguf_name()
        );
        let running = m.running_format();
        eprintln!("card: {} checkpoint runs as {running:?}", dtype.gguf_name());
        assert_eq!(
            running.as_deref(),
            Some(dtype.gguf_name()),
            "{} checkpoint on the card",
            dtype.gguf_name()
        );
        let got = prefill_and_step(&mut m);
        assert_near_the_f32_checkpoint(&got, &want, &format!("card {}", dtype.gguf_name()));
    }
    checkpoints_run_as_stored(&dir, &cards, &plan, "card", &|m| {
        m.layers.iter().all(|l| l.attn_output.device().is_cuda())
    });
    let _ = std::fs::remove_dir_all(&dir);
}

/// A float projection rebuilt to fit is rebuilt onto the card that runs it, in the block
/// format asked, and the ledger records it.
#[cfg(feature = "gpu")]
#[test]
fn a_float_projection_rebuilt_to_fit_is_rebuilt_on_the_card() {
    let Ok(card) = Device::new_cuda(0) else {
        eprintln!("skipping: no card");
        return;
    };
    let w = Tensor::from_vec(values(4 * HIDDEN, 3, 0.1), (4, HIDDEN), &Device::Cpu).unwrap();
    for (class, from, to) in REQUANTISED {
        let find = |name: &str| {
            STORED_FORMATS
                .into_iter()
                .find(|d| d.gguf_name() == name)
                .unwrap()
        };
        let qt = QTensor::quantize(&w, find(from)).unwrap();
        let out = super::loader::requant_float_weight(qt, find(to), &card, class).unwrap();
        assert_eq!(out.dtype(), find(to), "{from}->{to}");
        assert!(out.device().is_cuda(), "{from}->{to}: not on the card");
        assert!(
            crate::tensor::conversion::entries()
                .iter()
                .any(|e| e.class == *class
                    && e.from == *from
                    && e.to == *to
                    && e.reason == REQUANT_REASON),
            "no ledger entry for {class} {from}->{to}"
        );
    }
}

/// A float checkpoint whose every layer a Vulkan device runs is read there in its stored
/// format, on every Vulkan GPU the loader lists, and its logits stay within the half formats'
/// rounding of the host's.
///
/// It covers GGUF files and checkpoint directories.
#[cfg(feature = "vulkan")]
#[test]
fn a_float_checkpoint_on_vulkan_runs_as_stored() {
    use crate::inference::place::layer_executor::HeteroSegment;
    let devices = crate::inference::kernel::vulkan::devices();
    if devices.is_empty() {
        eprintln!("skipping: no Vulkan device");
        return;
    }
    let dir = std::env::temp_dir().join(format!("loken-conv-vulkan-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let want = host_reference(&dir);
    for d in devices {
        let plan = HeteroPlan {
            segments: vec![HeteroSegment {
                kind: DeviceKind::Vulkan(d.index),
                layer_start: 0,
                layer_end: BLOCKS,
                free_memory_bytes: u64::MAX,
            }],
            tp_cards: None,
            total_layers: BLOCKS,
            host_experts_from: None,
        };
        for dtype in FLOAT_FORMATS {
            let what = format!("vulkan {} {}", d.name, dtype.gguf_name());
            let path = dir.join(format!("{}.gguf", dtype.gguf_name()));
            write_dense(&path, dtype).unwrap();
            let mut m = load_placed(&path, &HashMap::new(), &plan).unwrap();
            assert!(!m.vulkan.is_empty(), "{what}: no layer on the device");
            let running = m.running_format();
            eprintln!("{what}: runs as {running:?}");
            assert_eq!(running.as_deref(), Some(dtype.gguf_name()), "{what}");
            let got = prefill_and_step(&mut m);
            assert_near_the_f32_checkpoint(&got, &want, &what);
        }
        let what = format!("vulkan {}", d.name);
        checkpoints_run_as_stored(&dir, &HashMap::new(), &plan, &what, &|m| {
            !m.vulkan.is_empty()
        });
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_requantised_projection_is_in_the_ledger() {
    let w = Tensor::from_vec(values(4 * HIDDEN, 3, 0.1), (4, HIDDEN), &Device::Cpu).unwrap();
    for (class, from, to) in REQUANTISED {
        let stored = STORED_FORMATS
            .into_iter()
            .find(|d| d.gguf_name() == *from)
            .unwrap();
        let target = STORED_FORMATS
            .into_iter()
            .find(|d| d.gguf_name() == *to)
            .unwrap();
        let qt = QTensor::quantize(&w, stored).unwrap();
        let out = super::loader::requant_float_weight(qt, target, &Device::Cpu, class).unwrap();
        assert_eq!(out.dtype(), target);
        let entries = crate::tensor::conversion::entries();
        assert!(
            entries.iter().any(|e| e.class == *class
                && e.from == *from
                && e.to == *to
                && e.reason == REQUANT_REASON),
            "no ledger entry for {class} {from}->{to}"
        );
    }
}

#[test]
fn the_stored_embedding_decodes_the_rows_it_is_asked_for() {
    let dir = std::env::temp_dir().join(format!("loken-conv-rows-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for dtype in STORED_FORMATS {
        let path = dir.join(format!("{}.gguf", dtype.gguf_name()));
        write_dense(&path, dtype).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file).unwrap() });
        let content =
            gguf_file::Content::read_mapped(&mut std::io::Cursor::new(&mmap[..]), mmap.clone())
                .unwrap();
        let qt = content
            .tensor_infos
            .get("token_embd.weight")
            .unwrap()
            .read(
                &mut std::io::Cursor::new(&mmap[..]),
                content.tensor_data_offset,
                &Device::Cpu,
            )
            .unwrap();
        let dense = qt
            .dequantize(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let emb = Embedding::from_stored(Arc::new(qt)).unwrap();
        let ids = Tensor::new(&[5u32, 0, 63, 5][..], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let got = emb.forward(&ids).unwrap();
        assert_eq!(got.dims(), &[1, 4, HIDDEN]);
        let want = dense.index_select(&ids.flatten_all().unwrap(), 0).unwrap();
        let g = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let w = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(
            g,
            w,
            "{}: a stored row decodes differently from the dense table",
            dtype.gguf_name()
        );
        assert_eq!(
            stored_bytes(dtype, VOCAB * HIDDEN),
            (VOCAB * HIDDEN / dtype.block_size() * dtype.type_size()) as u64
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_host_expert_stack_is_used_as_the_file_stores_it_when_a_dot_kernel_reads_it() {
    use crate::inference::generic_transformer::loader::moe_stack_served_as_stored;
    for dtype in [
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q8_0,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::MxFp4,
        GgmlDType::F32,
    ] {
        assert!(
            moe_stack_served_as_stored(dtype, false),
            "{dtype:?} host expert stack would be rebuilt"
        );
    }
}

/// A float projection a Vulkan or OpenCL device reads from the file takes the block format its
/// layer holds. A layer that holds it as stored, and a tensor outside the layers, keep the
/// stored format.
#[test]
fn a_projection_read_for_another_runtime_takes_its_layer_format() {
    use super::loader::in_layer_format;
    let dir = std::env::temp_dir().join(format!("loken-conv-layer-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let w = Tensor::from_vec(values(FFN * HIDDEN, 5, 0.1), (FFN, HIDDEN), &Device::Cpu).unwrap();
    let stored = || QTensor::quantize(&w, GgmlDType::BF16).unwrap();
    for held in [GgmlDType::Q8_0, GgmlDType::Q4K, GgmlDType::BF16] {
        let path = dir.join(format!("{}.gguf", held.gguf_name()));
        write_dense(&path, held).unwrap();
        let m = load(&path).unwrap();
        for name in [
            "blk.0.attn_q.weight",
            "blk.1.ffn_up.weight",
            "blk.1.ffn_gate.weight",
        ] {
            let got = in_layer_format(&m.layers, name, stored()).unwrap();
            assert_eq!(got.dtype(), held, "{name} in a {held:?} model");
        }
        for name in [
            "output.weight",
            "blk.0.attn_norm.weight",
            "blk.9.attn_q.weight",
        ] {
            let got = in_layer_format(&m.layers, name, stored()).unwrap();
            assert_eq!(got.dtype(), GgmlDType::BF16, "{name} in a {held:?} model");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Each of the query, key and value projections read for another runtime takes the format its
/// own projection holds: values stored apart from the queries keep their own format.
#[test]
fn values_read_for_another_runtime_keep_their_own_format() {
    use super::loader::in_layer_format;
    let dir = std::env::temp_dir().join(format!("loken-conv-values-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("mixed.gguf");
    write_dense_values_in(&path, GgmlDType::Q4K, GgmlDType::F16).unwrap();
    let m = load(&path).unwrap();
    let w = Tensor::from_vec(
        values(HIDDEN * HIDDEN, 5, 0.1),
        (HIDDEN, HIDDEN),
        &Device::Cpu,
    )
    .unwrap();
    let held = |q: &Option<QMatMul>| {
        q.as_ref()
            .and_then(|q| q.qtensor())
            .map(|t| t.dtype())
            .filter(|d| !matches!(d, GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::F32))
    };
    let l = &m.layers[0];
    for (name, own) in [
        ("blk.0.attn_q.weight", held(&l.attn_q)),
        ("blk.0.attn_v.weight", held(&l.attn_v)),
    ] {
        let stored = QTensor::quantize(&w, GgmlDType::F16).unwrap();
        let got = in_layer_format(&m.layers, name, stored).unwrap();
        assert_eq!(
            got.dtype(),
            own.unwrap_or(GgmlDType::F16),
            "{name} of a model whose values are stored in F16"
        );
    }
    assert_ne!(
        held(&l.attn_q),
        held(&l.attn_v),
        "the layer holds both alike"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
