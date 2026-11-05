//! The block-scaled checkpoint reader of the generic route against the V4.1 source, on a small
//! checkpoint written with the official naming.
//!
//! The checkpoint holds fp8 matrices with ue8m0 tile scales, fp4 routed experts with ue8m0 row
//! scales, bf16 and f32 tensors, in 2 shards.
//! Both readers open the same directory.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use safetensors::tensor::TensorView;
use safetensors::Dtype;

use super::wide_sparse_oracle_tests::{
    bit_difference, record_generic, record_v41, Record, RECORDS,
};
use super::*;
use crate::inference::model::deepseek_v41::model::DeepseekV41Model;
use crate::inference::model::deepseek_v41::safetensors_source::SafeTensorsSource;
use crate::inference::model::deepseek_v41::source::WeightSource;
use crate::inference::model::deepseek_v41::DeepseekV41Config;
use crate::inference::offload::store::ExpertLoader;
use crate::tensor::blockscaled::projection::Projection;

pub(super) const LAYERS: usize = 3;
pub(super) const DIM: usize = 256;
const HEADS: usize = 4;
const HEAD_DIM: usize = 128;
const Q_LORA: usize = 64;
const O_LORA: usize = 64;
const O_GROUPS: usize = 4;
const EXPERTS: usize = 8;
const INTER: usize = 256;
pub(super) const STREAMS: usize = 4;
/// Rows of a Sinkhorn mixing function: pre and post per stream, and the stream by stream mix.
const MIX: usize = (2 + STREAMS) * STREAMS;
const INDEX_HEADS: usize = 4;
const INDEX_HEAD_DIM: usize = 64;
pub(super) const VOCAB: usize = 48;
/// Side of the square tiles of an fp8 matrix.
const FP8_BLOCK: usize = 32;
/// Inputs of a row that share 1 fp4 scale.
const FP4_BLOCK: usize = 32;
/// Compress ratio per layer: layer 1 is the band's kv-source and index-source.
const RATIOS: [usize; LAYERS] = [0, 2, 0];
const BAND: usize = 1;

/// How a tensor is stored.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Format {
    F32,
    Bf16,
    /// e4m3 values, ue8m0 scales per `FP8_BLOCK` square tile.
    Fp8,
    /// e2m1 values 2 a byte, ue8m0 scales per `FP4_BLOCK` inputs of a row.
    Fp4,
    /// e4m3 rows, ue8m0 scales per run of the given count of values of a row.
    Fp8Rows(usize),
}

/// One tensor of the checkpoint: its official name, the loader's name for it, its storage and
/// its logical shape.
pub(super) struct Entry {
    pub(super) reference: String,
    loader: String,
    format: Format,
    pub(super) shape: Vec<usize>,
}

pub(super) fn entry(reference: String, loader: String, format: Format, shape: &[usize]) -> Entry {
    Entry {
        reference,
        loader,
        format,
        shape: shape.to_vec(),
    }
}

/// Every tensor of the checkpoint, the top tensors first, then layer after layer.
pub(super) fn entries() -> Vec<Entry> {
    use Format::*;
    let mut out = vec![
        entry(
            "embed.weight".into(),
            "token_embd".into(),
            Bf16,
            &[VOCAB, DIM],
        ),
        entry("norm.weight".into(), "output_norm".into(), F32, &[DIM]),
        entry("head.weight".into(), "output".into(), Fp8, &[VOCAB, DIM]),
    ];
    for l in 0..LAYERS {
        let mut put = |reference: &str, loader: &str, format: Format, shape: &[usize]| {
            out.push(entry(
                format!("layers.{l}.{reference}"),
                format!("blk.{l}.{loader}"),
                format,
                shape,
            ));
        };
        put("attn_norm.weight", "attn_norm", F32, &[DIM]);
        put("ffn_norm.weight", "ffn_norm", F32, &[DIM]);
        for part in ["attn", "ffn"] {
            put(
                &format!("hc_{part}_fn"),
                &format!("hc_{part}_fn"),
                F32,
                &[MIX, STREAMS * DIM],
            );
            put(
                &format!("hc_{part}_scale"),
                &format!("hc_{part}_scale"),
                F32,
                &[3],
            );
            put(
                &format!("hc_{part}_base"),
                &format!("hc_{part}_base"),
                F32,
                &[MIX],
            );
        }
        put("attn.wq_a.weight", "attn_q_a", Fp8, &[Q_LORA, DIM]);
        put("attn.q_norm.weight", "attn_q_a_norm", F32, &[Q_LORA]);
        put(
            "attn.wq_b.weight",
            "attn_q_b",
            Fp8,
            &[HEADS * HEAD_DIM, Q_LORA],
        );
        put("attn.wkv.weight", "attn_kv", Bf16, &[HEAD_DIM, DIM]);
        put("attn.kv_norm.weight", "attn_kv_a_norm", F32, &[HEAD_DIM]);
        put("attn.attn_sink", "attn_sinks", F32, &[HEADS]);
        put(
            "attn.wo_a.weight",
            "attn_output_a",
            Fp8,
            &[O_GROUPS * O_LORA, HEADS * HEAD_DIM / O_GROUPS],
        );
        put(
            "attn.wo_b.weight",
            "attn_output_b",
            Fp8,
            &[DIM, O_GROUPS * O_LORA],
        );
        put("ffn.gate.weight", "ffn_gate_inp", F32, &[EXPERTS, DIM]);
        put("ffn.gate.bias", "exp_probs_b", F32, &[EXPERTS]);
        put(
            "ffn.shared_experts.w1.weight",
            "ffn_gate_shexp",
            Fp8,
            &[INTER, DIM],
        );
        put(
            "ffn.shared_experts.w2.weight",
            "ffn_down_shexp",
            Fp8,
            &[DIM, INTER],
        );
        put(
            "ffn.shared_experts.w3.weight",
            "ffn_up_shexp",
            Fp8,
            &[INTER, DIM],
        );
        for e in 0..EXPERTS {
            let matrices = [
                ("w1", "ffn_gate_exps", [INTER, DIM]),
                ("w2", "ffn_down_exps", [DIM, INTER]),
                ("w3", "ffn_up_exps", [INTER, DIM]),
            ];
            for (w, stack, shape) in matrices {
                let reference = format!("ffn.experts.{e}.{w}.weight");
                put(&reference, &format!("{stack}.{e}"), Fp4, &shape);
            }
        }
        if l == BAND {
            put(
                "attn.compressor.norm.weight",
                "attn_compressor_norm",
                F32,
                &[HEAD_DIM],
            );
            put(
                "attn.compressor.wkv.weight",
                "attn_compressor_kv",
                Fp8,
                &[HEAD_DIM, DIM],
            );
            put(
                "attn.compressor.wgate.weight",
                "attn_compressor_gate",
                Bf16,
                &[HEAD_DIM, DIM],
            );
            put(
                "attn.indexer.wq_b.weight",
                "indexer.attn_q_b",
                Fp8,
                &[INDEX_HEADS * INDEX_HEAD_DIM, Q_LORA],
            );
            put(
                "attn.indexer.weights_proj.weight",
                "indexer.proj",
                Bf16,
                &[INDEX_HEADS, DIM],
            );
            put(
                "attn.indexer.wk.weight",
                "indexer.attn_k",
                F32,
                &[INDEX_HEAD_DIM, HEAD_DIM],
            );
            put(
                "attn.indexer.k_norm.weight",
                "indexer.k_norm",
                F32,
                &[INDEX_HEAD_DIM],
            );
        }
    }
    out
}

/// A small value of element `i` of tensor `seed`.
fn value(i: usize, seed: usize) -> f32 {
    ((i * 37 + seed * 101) % 211) as f32 * 0.0005 - 0.05
}

/// The stored tensors of `e`, tensor `seed`: its values and, when block-scaled, its scales.
///
/// An e4m3 byte keeps an exponent field below 7, so its value stays under 1.
/// A ue8m0 scale byte below 127 is a negative power of 2.
fn stored(e: &Entry, seed: usize) -> Vec<(String, Dtype, Vec<usize>, Vec<u8>)> {
    let n: usize = e.shape.iter().product();
    let scale_name = || {
        let stem = e
            .reference
            .strip_suffix(".weight")
            .expect("a block-scaled matrix");
        format!("{stem}.scale")
    };
    match e.format {
        Format::F32 => {
            let bytes = (0..n).flat_map(|i| value(i, seed).to_le_bytes()).collect();
            vec![(e.reference.clone(), Dtype::F32, e.shape.clone(), bytes)]
        }
        Format::Bf16 => {
            let bytes = (0..n)
                .flat_map(|i| ((value(i, seed).to_bits() >> 16) as u16).to_le_bytes())
                .collect();
            vec![(e.reference.clone(), Dtype::BF16, e.shape.clone(), bytes)]
        }
        Format::Fp8 => {
            let (rows, cols) = (e.shape[0], e.shape[1]);
            let bytes = (0..n)
                .map(|i| {
                    let magnitude = ((i * 13 + seed * 7) % 0x38) as u8;
                    let sign = if (i + seed).is_multiple_of(3) {
                        0x80
                    } else {
                        0
                    };
                    sign | magnitude
                })
                .collect();
            let tiles = vec![rows.div_ceil(FP8_BLOCK), cols.div_ceil(FP8_BLOCK)];
            let scales = (0..tiles[0] * tiles[1])
                .map(|t| 120 + ((t + seed) % 6) as u8)
                .collect();
            vec![
                (e.reference.clone(), Dtype::F8_E4M3, e.shape.clone(), bytes),
                (scale_name(), Dtype::F8_E8M0, tiles, scales),
            ]
        }
        Format::Fp4 => {
            let (rows, cols) = (e.shape[0], e.shape[1]);
            let bytes = (0..n / 2)
                .map(|i| ((i * 29 + seed * 11) % 256) as u8)
                .collect();
            let runs = vec![rows, cols / FP4_BLOCK];
            let scales = (0..runs[0] * runs[1])
                .map(|r| 118 + ((r + seed) % 5) as u8)
                .collect();
            vec![
                (e.reference.clone(), Dtype::I8, vec![rows, cols / 2], bytes),
                (scale_name(), Dtype::F8_E8M0, runs, scales),
            ]
        }
        Format::Fp8Rows(run) => {
            let (rows, cols) = (e.shape[0], e.shape[1]);
            let bytes = (0..n)
                .map(|i| {
                    let magnitude = ((i * 13 + seed * 7) % 0x38) as u8;
                    let sign = if (i + seed).is_multiple_of(3) {
                        0x80
                    } else {
                        0
                    };
                    sign | magnitude
                })
                .collect();
            let runs = vec![rows, cols / run];
            let scales = (0..runs[0] * runs[1])
                .map(|r| 120 + ((r + seed) % 6) as u8)
                .collect();
            vec![
                (e.reference.clone(), Dtype::F8_E4M3, e.shape.clone(), bytes),
                (scale_name(), Dtype::F8_E8M0, runs, scales),
            ]
        }
    }
}

/// The reference arguments of the checkpoint's model.
pub(super) fn reference_config() -> serde_json::Value {
    serde_json::json!({
        "n_layers": LAYERS, "dim": DIM, "vocab_size": VOCAB, "norm_eps": 1e-20,
        "n_heads": HEADS, "head_dim": HEAD_DIM, "rope_head_dim": 32,
        "q_lora_rank": Q_LORA, "o_lora_rank": O_LORA, "o_groups": O_GROUPS, "window_size": 32,
        "n_routed_experts": EXPERTS, "n_shared_experts": 1, "n_activated_experts": 2,
        "moe_inter_dim": INTER, "route_scale": 1.5, "swiglu_limit": 10.0,
        "norm_topk_prob": true, "compress_ratios": RATIOS, "kv_source_layers": [BAND],
        "index_source_layers": [BAND], "index_n_heads": INDEX_HEADS,
        "index_head_dim": INDEX_HEAD_DIM, "index_topk": 16, "hc_mult": STREAMS,
        "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, "rope_theta": 10000.0,
        "compress_rope_theta": 160000.0, "rope_factor": 16.0, "original_seq_len": 65536
    })
}

/// A new directory for one test's checkpoint.
fn scratch(tag: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "loken-block-scaled-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("inference")).unwrap();
    dir
}

/// Writes the checkpoint into a new directory: the top tensors and layer 0 in the first shard,
/// the other layers in the second.
pub(super) fn checkpoint(tag: &str) -> PathBuf {
    checkpoint_of(tag, &entries())
}

/// Writes the checkpoint of `entries` into a new directory, as `checkpoint` writes its own.
pub(super) fn checkpoint_of(tag: &str, entries: &[Entry]) -> PathBuf {
    checkpoint_with(tag, entries, &reference_config())
}

/// Writes the checkpoint of `entries` described by the reference arguments `reference` into a
/// new directory, as `checkpoint` writes its own.
pub(super) fn checkpoint_with(
    tag: &str,
    entries: &[Entry],
    reference: &serde_json::Value,
) -> PathBuf {
    let dir = scratch(tag);
    let config = serde_json::json!({
        "quantization_config": { "weight_block_size": [FP8_BLOCK, FP8_BLOCK] }
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    std::fs::write(
        dir.join("inference").join("config.json"),
        reference.to_string(),
    )
    .unwrap();
    let mut shards: [Vec<(String, Dtype, Vec<usize>, Vec<u8>)>; 2] = [Vec::new(), Vec::new()];
    for (seed, e) in entries.iter().enumerate() {
        let second = e.reference.starts_with("layers.") && !e.reference.starts_with("layers.0.");
        shards[usize::from(second)].extend(stored(e, seed));
    }
    for (i, tensors) in shards.iter().enumerate() {
        let views: Vec<(String, TensorView)> = tensors
            .iter()
            .map(|(name, dtype, shape, bytes)| {
                (
                    name.clone(),
                    TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
                )
            })
            .collect();
        let path = dir.join(format!("model-0000{}-of-00002.safetensors", i + 1));
        safetensors::serialize_to_file(views, None, &path).unwrap();
    }
    dir
}

fn bits(t: &Tensor) -> Vec<u32> {
    let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    v.iter().map(|x| x.to_bits()).collect()
}

/// A projection as stored: its kind, its shape, its bytes, its scale bytes and its block.
fn layout(p: &Projection) -> (String, Vec<usize>, Vec<u8>, Vec<u8>, usize) {
    match p {
        Projection::Fp8(f) => (
            format!("fp8 {:?}", f.scale_format),
            vec![f.out, f.inp],
            f.bytes.as_slice().to_vec(),
            f.scales.as_slice().to_vec(),
            f.block,
        ),
        Projection::Fp4(f) => (
            "fp4".into(),
            vec![f.out, f.inp],
            f.nibbles.as_slice().to_vec(),
            f.scales.as_slice().to_vec(),
            f.block,
        ),
        Projection::Bf16(f) => (
            "bf16".into(),
            vec![f.out, f.inp],
            f.bytes.as_slice().to_vec(),
            Vec::new(),
            0,
        ),
        Projection::Dense(t) => (
            "dense".into(),
            t.dims().to_vec(),
            bits(t).iter().flat_map(|b| b.to_le_bytes()).collect(),
            Vec::new(),
            0,
        ),
        Projection::Quant(_) => panic!("a safetensors checkpoint holds no GGUF blocks"),
    }
}

/// The projection the V4.1 loader builds of `name`: in place when the source offers it, dense
/// F32 otherwise.
fn v41_projection(src: &SafeTensorsSource, name: &str) -> Projection {
    match src.projection(name).unwrap() {
        Some(p) => p,
        None => Projection::Dense(src.dense_f32(name).unwrap()),
    }
}

/// Every tensor of the checkpoint reads to the same bytes through both readers: its dense F32
/// values, its projection in place with the same values, scales and block, and each routed
/// expert's 3 matrices as the V4.1 expert loader builds them.
#[test]
fn the_block_scaled_reader_hands_out_the_bytes_of_the_v41_source() {
    let dir = checkpoint("reads");
    let v41 = SafeTensorsSource::open(&dir).unwrap();
    let ours = BlockScaledCheckpoint::open(&dir).unwrap();
    let mut differ = Vec::new();
    for e in entries() {
        assert!(ours.has(&e.loader), "{} is not listed", e.loader);
        let want = WeightSource::dense_f32(&v41, &e.reference).unwrap();
        let got = ours.dense_f32(&e.loader).unwrap();
        if want.dims() != got.dims() || bits(&want) != bits(&got) {
            differ.push(format!("{}: dense values", e.reference));
        }
        if e.shape.len() == 2 {
            let want = layout(&v41_projection(&v41, &e.reference));
            let got = layout(&ours.projection(&e.loader).unwrap());
            if want != got {
                differ.push(format!(
                    "{}: projection {} != {}",
                    e.reference, got.0, want.0
                ));
            }
        }
    }
    for l in 0..LAYERS {
        let loader = v41.experts(l, EXPERTS).unwrap();
        for x in 0..EXPERTS {
            let built = loader.build(x).unwrap();
            let matrices = [
                (&built.w1, "ffn_gate_exps"),
                (&built.w3, "ffn_up_exps"),
                (&built.w2, "ffn_down_exps"),
            ];
            for (w, stack) in matrices {
                let name = format!("blk.{l}.{stack}.{x}");
                if layout(w) != layout(&ours.projection(&name).unwrap()) {
                    differ.push(format!("{name}: expert matrix"));
                }
                if !matches!(w, Projection::Fp4(_)) {
                    differ.push(format!("{name}: the V4.1 expert is not an fp4 view"));
                }
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        differ.is_empty(),
        "the generic reader differs from the V4.1 source:\n{}",
        differ.join("\n")
    );
}

/// The tokens of the checkpoint test: a prompt longer than the verified block and the greedy
/// prompt.
pub(super) fn tokens() -> Vec<u32> {
    (0..12).map(|i| ((i * 7 + 3) % VOCAB) as u32).collect()
}

/// The V4.1 model of the checkpoint, read through its own source.
pub(super) fn v41_model(dir: &Path) -> DeepseekV41Model {
    let src = SafeTensorsSource::open(dir).unwrap();
    let cfg = DeepseekV41Config::from_reference_config(&src.inference_config, None).unwrap();
    DeepseekV41Model::load(&src, &cfg).unwrap()
}

/// The generic model of the checkpoint, every layer on the host.
pub(super) fn generic_model(dir: &Path) -> GenericHeteroTransformer {
    let ck = BlockScaledCheckpoint::open(dir).unwrap();
    let plan = HeteroPlan::calculate_with_kv(LAYERS, 1 << 20, &[], &[], 1.0, 0);
    GenericHeteroTransformer::from_block_scaled(
        &ck,
        &plan,
        crate::inference::engine::llm_engine::KvQuant::Off,
        None,
    )
    .unwrap()
}

/// The generic model built from the block-scaled reader gives the bits of the V4.1 model read
/// from the same checkpoint: the prefill, the decode, the verify, the rewound verify and the
/// decode after it, and the greedy run.
#[test]
fn a_model_read_from_the_block_scaled_checkpoint_gives_the_bits_of_the_v41_path() {
    let dir = checkpoint("model");
    let v41 = v41_model(&dir);
    let mut generic = generic_model(&dir);
    let tokens = tokens();
    let (mut want, mut got): (BTreeMap<String, Record>, BTreeMap<String, Record>) =
        (BTreeMap::new(), BTreeMap::new());
    record_v41(&v41, &tokens, 0, &mut want);
    record_generic(&mut generic, &tokens, 0, &mut got);
    let _ = std::fs::remove_dir_all(&dir);
    let mut differ = Vec::new();
    for record in RECORDS {
        let name = format!("l0_{record}");
        match (want.get(&name), got.get(&name)) {
            (Some(w), Some(g)) => {
                if let Some(d) = bit_difference(w, g) {
                    differ.push(format!("{name}: {d}"));
                }
            }
            _ => differ.push(format!("{name}: missing")),
        }
    }
    assert!(
        differ.is_empty(),
        "the generic model of the checkpoint differs from the V4.1 path:\n{}",
        differ.join("\n")
    );
}
