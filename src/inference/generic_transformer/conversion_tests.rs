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

/// Every conversion a dense host load performs: `(class, stored, resident, reason)`.
/// A stored format a class keeps does not appear. The reason is what the loader's ledger
/// records for the same change.
const DECLARED: &[(&str, &str, &str, &str)] = &[
    (
        "projection",
        "f32",
        "q4_K",
        "float projection rebuilt as a block format for the block-quant kernels and fused projections",
    ),
    (
        "projection",
        "f16",
        "q4_K",
        "float projection rebuilt as a block format for the block-quant kernels and fused projections",
    ),
    (
        "projection",
        "bf16",
        "q4_K",
        "float projection rebuilt as a block format for the block-quant kernels and fused projections",
    ),
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

/// A dense model of two blocks whose every weight is stored as `dtype`; the norms are F32.
fn write_dense(path: &std::path::Path, dtype: GgmlDType) -> Result<()> {
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
    let mut entries = Vec::new();
    for (i, (name, dims)) in tensors.into_iter().enumerate() {
        let n: usize = dims.iter().product();
        let is_norm = name.ends_with("norm.weight");
        let vals: Vec<f32> = if is_norm {
            values(n, i as u64, 0.1)
                .into_iter()
                .map(|x| 1.0 + x)
                .collect()
        } else {
            values(n, i as u64 + 1, 1.5 / (*dims.last().unwrap() as f32).sqrt())
        };
        let stored = if is_norm { GgmlDType::F32 } else { dtype };
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
        match &m.output_proj {
            crate::tensor::quantized::QMatMul::QTensor(q) if q.dtype() == dtype => {}
            crate::tensor::quantized::QMatMul::Tensor(t)
            | crate::tensor::quantized::QMatMul::TensorF16(t)
                if t.dtype().as_str() == stored => {}
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

#[test]
fn a_declared_conversion_is_in_the_ledger() {
    let dir = std::env::temp_dir().join(format!("loken-conv-ledger-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("f16.gguf");
    write_dense(&path, GgmlDType::F16).unwrap();
    let _m = load(&path).unwrap();
    let entries = crate::tensor::conversion::entries();
    let _ = std::fs::remove_dir_all(&dir);
    for (class, from, to, why) in DECLARED.iter().filter(|d| d.1 == "f16") {
        assert!(
            entries
                .iter()
                .any(|e| e.class == *class && e.from == *from && e.to == *to && e.reason == *why),
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
    ] {
        assert!(
            moe_stack_served_as_stored(dtype, false),
            "{dtype:?} host expert stack would be rebuilt"
        );
    }
}
