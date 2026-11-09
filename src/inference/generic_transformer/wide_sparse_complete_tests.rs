//! The completion and the draft companion of the generic route against the V4.1 writers.
//!
//! The checkpoint is the one of the block-scaled reader tests with a block draft of 2 stages.
//! Its draft holds fp8, fp4, bf16 and f32 tensors under the official names.
//! Both writers read the same directory and write with the same layout and format.
//! The files they write are compared byte for byte.
//!
//! Only a host build is compared, as for the converter.

use std::path::{Path, PathBuf};

use super::block_draft::{dense_names, expert_prefix, BlockDraftMeta};
use super::block_scaled_oracle_tests::{
    checkpoint_of, entries, entry, Entry, Format, DIM, STREAMS,
};
use super::wide_sparse_complete::{
    complete, draft_bytes_planned, has_draft, verify_carried, write_draft, Layout,
};
use super::wide_sparse_names::{ARCH, EXPERT_STACKS};
use super::BlockScaledCheckpoint;
use crate::inference::model::deepseek_v41::complete as v41;
use crate::inference::model::deepseek_v41::safetensors_source::SafeTensorsSource;
use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
use crate::tensor::quantized::gguf_file::{open_mapped, Value};
use crate::tensor::quantized::GgmlDType;

/// Stages of the draft.
const STAGES: usize = 2;
/// Routed experts of each stage.
const DRAFT_EXPERTS: usize = 3;
/// Inputs of a draft expert's down projection: a multiple of the q8_0 block, not of the q6_k one.
const DRAFT_INTER: usize = 96;
/// Rows of a Sinkhorn mixing function: pre and post per stream, and the stream by stream mix.
const MIX: usize = (2 + STREAMS) * STREAMS;
/// The layouts both writers are asked for.
const LAYOUTS: [(u64, u64); 2] = [(64, 256), (4096, 8192)];

/// The draft's tensors, by the official names.
fn draft_entries() -> Vec<Entry> {
    use Format::*;
    let mut out = Vec::new();
    let mut put = |name: String, format: Format, shape: &[usize]| {
        out.push(entry(name.clone(), name, format, shape));
    };
    for s in 0..STAGES {
        let p = |rest: &str| format!("mtp.{s}.{rest}");
        put(p("attn.wq_a.weight"), Fp8, &[64, DIM]);
        put(p("attn.q_norm.weight"), F32, &[64]);
        put(p("attn.wq_b.weight"), Fp8, &[512, 64]);
        put(p("attn.wkv.weight"), Bf16, &[128, DIM]);
        put(p("attn.kv_norm.weight"), F32, &[128]);
        put(p("attn.attn_sink"), F32, &[4]);
        put(p("attn.wo_a.weight"), Fp8, &[256, 128]);
        put(p("attn.wo_b.weight"), Fp8, &[DIM, 256]);
        put(p("attn_norm.weight"), F32, &[DIM]);
        put(p("ffn_norm.weight"), F32, &[DIM]);
        put(p("ffn.gate.weight"), F32, &[DRAFT_EXPERTS, DIM]);
        put(p("ffn.gate.bias"), F32, &[DRAFT_EXPERTS]);
        put(p("ffn.shared_experts.w1.weight"), Fp8, &[DRAFT_INTER, DIM]);
        put(p("ffn.shared_experts.w2.weight"), Fp8, &[DIM, DRAFT_INTER]);
        put(p("ffn.shared_experts.w3.weight"), Fp8, &[DRAFT_INTER, DIM]);
        for part in ["attn", "ffn"] {
            put(p(&format!("hc_{part}_fn")), F32, &[MIX, STREAMS * DIM]);
            put(p(&format!("hc_{part}_scale")), F32, &[3]);
            put(p(&format!("hc_{part}_base")), F32, &[MIX]);
        }
        for e in 0..DRAFT_EXPERTS {
            let stem = format!("{}.{e}", expert_prefix(s));
            put(format!("{stem}.w1.weight"), Fp4, &[DRAFT_INTER, DIM]);
            put(format!("{stem}.w2.weight"), Fp4, &[DIM, DRAFT_INTER]);
            put(format!("{stem}.w3.weight"), Fp4, &[DRAFT_INTER, DIM]);
        }
    }
    let last = STAGES - 1;
    // Enough values for the always-read format.
    put("mtp.0.main_proj.weight".into(), Fp8, &[512, 2048]);
    put("mtp.0.main_norm.weight".into(), F32, &[512]);
    put(format!("mtp.{last}.norm.weight"), F32, &[DIM]);
    put(
        format!("mtp.{last}.markov_head.embed.weight"),
        Bf16,
        &[48, 32],
    );
    put(
        format!("mtp.{last}.markov_head.head.weight"),
        Fp8,
        &[64, 32],
    );
    out
}

/// The checkpoint of the block-scaled reader tests with the draft's tensors and its shape in
/// `config.json`.
fn draft_checkpoint(tag: &str) -> PathBuf {
    let mut all = entries();
    all.extend(draft_entries());
    let dir = checkpoint_of(tag, &all);
    let path = dir.join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    config["text_config"] = serde_json::json!({
        "num_nextn_predict_layers": STAGES, "dspark_block_size": 5,
        "dspark_noise_token_id": 7, "dspark_target_layer_ids": [0, 2],
        "dspark_markov_rank": 32, "dspark_n_routed_experts": DRAFT_EXPERTS,
        "dspark_num_experts_per_tok": 2, "sliding_window": 16, "hidden_size": DIM,
        "hc_mult": STREAMS, "num_attention_heads": 4, "head_dim": 128, "qk_rope_head_dim": 32,
        "o_groups": 4, "o_lora_rank": 64, "vocab_size": 48, "swiglu_limit": 10.0,
        "routed_scaling_factor": 1.5, "rms_norm_eps": 1e-6, "rope_theta": 10000,
        "rope_scaling": { "factor": 16, "original_max_position_embeddings": 65536 }
    });
    std::fs::write(&path, config.to_string()).unwrap();
    dir
}

/// A converted file: dense tensors, 2 layers of expert stacks, 1 f32 and 1 q8_0, a hot list for
/// the first, and an engram table with its scales.
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
            Value::String(ARCH.into()),
        ),
        (format!("{ARCH}.block_count"), Value::U32(2)),
        (
            format!("{ARCH}.hot_experts.0"),
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

/// The V4.1 layout of the same boundaries.
fn v41_layout(l: Layout) -> v41::Layout {
    v41::Layout {
        alignment: l.alignment,
        unit: l.unit,
    }
}

/// Asserts the files at `a` and `b` hold the same bytes.
fn same_file(a: &Path, b: &Path, what: &str) {
    let (x, y) = (std::fs::read(a).unwrap(), std::fs::read(b).unwrap());
    assert_eq!(x.len(), y.len(), "{what}: file lengths");
    if let Some(at) = x.iter().zip(&y).position(|(p, q)| p != q) {
        panic!("{what}: the files first differ at byte {at}");
    }
}

/// For each format and layout, the generic writer of the draft companion writes the bytes of the
/// V4.1 writer, and plans the length it writes.
/// The file declares the shape the checkpoint gives the draft.
#[test]
fn the_draft_companion_is_the_bytes_of_the_v41_writer() {
    let dir = draft_checkpoint("draft-companion");
    let ours = BlockScaledCheckpoint::open(&dir).unwrap();
    let old = SafeTensorsSource::open(&dir).unwrap();
    assert!(has_draft(&ours));
    for format in [GgmlDType::Q8_0, GgmlDType::Q6K] {
        for (alignment, unit) in LAYOUTS {
            let layout = Layout { alignment, unit };
            let what = format!("{format:?} {alignment}/{unit}");
            let old_out = dir.join("draft-v41.gguf");
            let new_out = dir.join("draft-generic.gguf");
            let old_digest = v41::write_draft(
                &old,
                &dir,
                &old_out,
                v41_layout(layout),
                format,
                &|_, _, _| {},
            )
            .unwrap();
            let new_digest = write_draft(&ours, &new_out, layout, format, &|_, _, _| {}).unwrap();
            same_file(&old_out, &new_out, &what);
            assert_eq!(old_digest, new_digest, "{what}: digests");
            let planned = draft_bytes_planned(&ours, &new_out, layout, format).unwrap();
            let old_planned =
                v41::draft_bytes_planned(&old, &dir, &old_out, v41_layout(layout), format).unwrap();
            assert_eq!(planned, old_planned, "{what}: planned lengths");
            assert_eq!(
                planned,
                std::fs::metadata(&new_out).unwrap().len(),
                "{what}"
            );

            // Both formats are reached: a q6_k down projection of the draft stays f32.
            let g = open_mapped(&new_out).unwrap();
            let dtype = |n: &str| g.content.tensor_infos[n].ggml_dtype;
            assert_eq!(dtype("mtp.0.main_proj.weight"), format, "{what}");
            assert_eq!(dtype("mtp.0.attn_norm.weight"), GgmlDType::F32, "{what}");
            let down = format!("{}.0.w2.weight", expert_prefix(1));
            let want = match format {
                GgmlDType::Q6K => GgmlDType::F32,
                f => f,
            };
            assert_eq!(dtype(&down), want, "{what}");
            let tensors = dense_names(STAGES).len() + STAGES * DRAFT_EXPERTS * 3;
            assert_eq!(g.content.tensor_infos.len(), tensors, "{what}");
            let declared = BlockDraftMeta::from_metadata(&g.content.metadata)
                .unwrap()
                .unwrap();
            assert_eq!(declared, BlockDraftMeta::from_config_json(&dir).unwrap());
            assert_eq!(
                (declared.rope_factor, declared.original_context),
                (16.0, 65536)
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Completes `src` with both writers into `{tag}-v41.gguf` and `{tag}-generic.gguf`, asserts
/// the 2 files, their digests and their carry checks are equal, and returns the generic file.
fn both_complete(
    dir: &Path,
    src: &Path,
    with_draft: bool,
    format: Option<GgmlDType>,
    layout: Layout,
    tag: &str,
) -> PathBuf {
    let ours = BlockScaledCheckpoint::open(dir).unwrap();
    let old = SafeTensorsSource::open(dir).unwrap();
    let g = open_mapped(src).unwrap();
    let old_out = dir.join(format!("{tag}-v41.gguf"));
    let new_out = dir.join(format!("{tag}-generic.gguf"));
    let old_draft = with_draft.then_some((&old, dir));
    let old_digest = v41::complete(
        &g,
        old_draft,
        &old_out,
        v41_layout(layout),
        format,
        &|_, _, _| {},
    )
    .unwrap();
    let new_draft = with_draft.then_some(&ours);
    let new_digest = complete(&g, new_draft, &new_out, layout, format, &|_, _, _| {}).unwrap();
    same_file(&old_out, &new_out, tag);
    assert_eq!(old_digest, new_digest, "{tag}: digests");
    let old_check = v41::verify_carried(src, &old_out, &|_, _, _| {}).unwrap();
    let new_check = verify_carried(src, &new_out, &|_, _, _| {}).unwrap();
    assert_eq!(
        (
            old_check.keys,
            old_check.whole,
            old_check.split,
            old_check.bytes
        ),
        (
            new_check.keys,
            new_check.whole,
            new_check.split,
            new_check.bytes
        ),
        "{tag}: carry checks"
    );
    new_out
}

/// The generic completion writes the bytes of the V4.1 completion: with the checkpoint's draft
/// in a named format and in the file's own, without a checkpoint, and completing a completed
/// file again.
/// A carried byte changed is found by the generic check.
#[test]
fn completing_writes_the_bytes_of_the_v41_completion() {
    let dir = draft_checkpoint("complete");
    let src = dir.join("src.gguf");
    converted(&src);
    let [small, large] = LAYOUTS.map(|(alignment, unit)| Layout { alignment, unit });
    let once = both_complete(&dir, &src, true, Some(GgmlDType::Q8_0), small, "named");
    both_complete(&dir, &src, true, None, large, "own-format");
    both_complete(&dir, &src, false, None, small, "no-draft");
    let twice = both_complete(&dir, &once, true, None, small, "twice");

    // Completing again carries the units and the draft the first completion wrote.
    let check = verify_carried(&once, &twice, &|_, _, _| {}).unwrap();
    let first = open_mapped(&once).unwrap();
    assert_eq!(
        (check.split, check.whole),
        (0, first.content.tensor_infos.len())
    );
    // The completed file declares the checkpoint's draft and carries every tensor of it.
    let declared = BlockDraftMeta::from_metadata(&first.content.metadata)
        .unwrap()
        .unwrap();
    assert_eq!(declared, BlockDraftMeta::from_config_json(&dir).unwrap());
    for n in dense_names(STAGES) {
        assert!(first.content.tensor_infos.contains_key(&n), "{n}");
    }

    let mut bad = std::fs::read(&once).unwrap();
    let info = &first.content.tensor_infos["blk.0.ffn_up_exps.1"];
    bad[(first.content.tensor_data_offset + info.offset) as usize] ^= 1;
    let bad_path = dir.join("bad.gguf");
    std::fs::write(&bad_path, &bad).unwrap();
    assert!(verify_carried(&src, &bad_path, &|_, _, _| {}).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}
