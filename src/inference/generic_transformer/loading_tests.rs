//! One loading test per architecture: a tiny synthetic checkpoint of the architecture is
//! written in each format the loader reads for it, loaded through the entry point and the
//! placement the server uses, and run through one forward pass of a few tokens.
//!
//! The loader decides everything from tensors and metadata, so an architecture is a recipe:
//! the tensors one block carries and the header keys around them. `docs/ARCHITECTURES.md`
//! lists every architecture with the test that covers it.

use super::*;
use crate::inference::engine::llm_engine::GenericWeights;
use crate::inference::place::layer_executor::HeteroPlan;
use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
use crate::tensor::quantized::gguf_file::Value;
use crate::tensor::quantized::GgmlDType;
use std::collections::BTreeMap;

const HIDDEN: usize = 64;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 32;
const FFN: usize = 128;
const VOCAB: usize = 96;
/// Rows of a learned position table: the context the tiny files declare.
const POSITIONS: usize = 256;
const EXPERTS: usize = 4;
const EXPERTS_USED: usize = 2;
const EXPERT_FFN: usize = 64;
const CONV_KERNEL: usize = 4;
/// Channels of the recurrent mixers' state: inner width, state size, groups, value heads.
const SSM_INNER: usize = 64;
const SSM_STATE: usize = 16;
const SSM_GROUPS: usize = 2;
const SSM_HEADS: usize = 4;
const MAMBA_HEADS: usize = 8;
const PROMPT: [u32; 5] = [1, 17, 42, 8, 63];

/// What mixes a block's tokens.
#[derive(Clone, Copy, PartialEq)]
enum Mix {
    Attention,
    ShortConv,
    Mamba,
    /// Gated delta rule with `alpha` and `beta` projections apart.
    Delta,
    /// Gated delta rule with them packed in one `ssm_ba`.
    DeltaPacked,
    /// A feed-forward alone.
    Nothing,
}

/// The block's feed-forward.
#[derive(Clone, Copy, PartialEq)]
enum Ffn {
    Gated,
    /// Gate and up in one `ffn_up` of twice the width.
    FusedGateUp,
    /// `down(act(up(x)))`.
    Plain,
    /// Routed experts with a gate and an up projection each.
    Moe,
    /// Routed experts with an up projection only.
    MoeUngated,
    None,
}

/// The tensors and header of one architecture's tiny checkpoint.
#[derive(Clone)]
struct Recipe {
    arch: &'static str,
    mixers: Vec<Mix>,
    /// The feed-forward of every block; blocks listed in `dense_blocks` keep a dense one.
    ffn: Ffn,
    dense_blocks: Vec<usize>,
    /// The blocks that carry a feed-forward; every block when empty.
    ffn_blocks: Vec<usize>,
    fused_qkv: bool,
    qkv_bias: bool,
    out_bias: bool,
    qk_norm: bool,
    /// Norms after attention and after the feed-forward.
    post_norms: bool,
    /// The only norm ahead of the feed-forward is `post_attention_norm`.
    post_attention_as_ffn_norm: bool,
    /// The feed-forward norm is absent: one norm feeds attention and feed-forward.
    parallel: bool,
    layer_norm: bool,
    ffn_bias: bool,
    tied_head: bool,
    /// Query projection carries an output gate per head.
    gated_query: bool,
    shared_expert: bool,
    sigmoid_router: bool,
    /// Attention output tensor is named `attn_out`, with sinks and expert biases.
    sinks: bool,
    /// Per-layer output scale, window pattern and the second feed-forward norm pair.
    local_global: bool,
    /// The embedding norm stands for the output norm.
    embedding_norm: bool,
    sliding_window: bool,
    /// Lanes of a head the rotation covers, when not all.
    rotary: Option<usize>,
    /// The Q and K norms span every head's projection, not one head.
    full_qk_norm: bool,
    /// No norm precedes attention or the feed-forward.
    no_pre_norms: bool,
    /// The scale keys of a family that multiplies embeddings, residuals, scores and logits.
    scales: bool,
    /// Alternating sliding-window layers, with the frequency of the local ones.
    window: bool,
    /// Biases of the routed experts.
    expert_bias: bool,
    /// A learned table of absolute positions in place of the rotation.
    positions: bool,
}

impl Recipe {
    fn dense(arch: &'static str) -> Self {
        Self {
            arch,
            mixers: vec![Mix::Attention; 2],
            ffn: Ffn::Gated,
            dense_blocks: Vec::new(),
            ffn_blocks: Vec::new(),
            fused_qkv: false,
            qkv_bias: false,
            out_bias: false,
            qk_norm: false,
            post_norms: false,
            post_attention_as_ffn_norm: false,
            parallel: false,
            layer_norm: false,
            ffn_bias: false,
            tied_head: false,
            gated_query: false,
            shared_expert: false,
            sigmoid_router: false,
            sinks: false,
            local_global: false,
            embedding_norm: false,
            sliding_window: false,
            rotary: None,
            full_qk_norm: false,
            no_pre_norms: false,
            scales: false,
            window: false,
            expert_bias: false,
            positions: false,
        }
    }

    fn positions(mut self) -> Self {
        self.positions = true;
        self
    }

    fn bias(mut self) -> Self {
        self.qkv_bias = true;
        self
    }

    fn qk(mut self) -> Self {
        self.qk_norm = true;
        self
    }

    fn post(mut self) -> Self {
        self.post_norms = true;
        self
    }

    fn fused(mut self) -> Self {
        self.fused_qkv = true;
        self
    }

    fn ffn(mut self, ffn: Ffn) -> Self {
        self.ffn = ffn;
        self
    }

    /// Layer norms with biases and a feed-forward with biases, as the older families write.
    fn classic(mut self) -> Self {
        self.layer_norm = true;
        self.ffn_bias = true;
        self.out_bias = true;
        self.ffn = Ffn::Plain;
        self
    }

    fn parallel(mut self) -> Self {
        self.parallel = true;
        self
    }

    fn tied(mut self) -> Self {
        self.tied_head = true;
        self
    }

    fn moe(self) -> Self {
        self.ffn(Ffn::Moe)
    }

    fn heads_per_block(&self) -> Vec<usize> {
        self.mixers
            .iter()
            .map(|m| if *m == Mix::Attention { KV_HEADS } else { 0 })
            .collect()
    }

    fn ffn_of(&self, il: usize) -> Ffn {
        if !self.ffn_blocks.is_empty() && !self.ffn_blocks.contains(&il) {
            Ffn::None
        } else if self.dense_blocks.contains(&il) && matches!(self.ffn, Ffn::Moe | Ffn::MoeUngated)
        {
            Ffn::Gated
        } else {
            self.ffn
        }
    }

    fn metadata(&self) -> Vec<(String, Value)> {
        let a = self.arch;
        let u = |k: &str, v: usize| (format!("{a}.{k}"), Value::U32(v as u32));
        let mut meta = vec![
            ("general.architecture".to_string(), Value::String(a.into())),
            u("block_count", self.mixers.len()),
            u("embedding_length", HIDDEN),
            u("feed_forward_length", FFN),
            u("attention.head_count", HEADS),
            u("attention.key_length", HEAD_DIM),
            u("attention.value_length", HEAD_DIM),
            u("context_length", 256),
            (
                format!("{a}.attention.layer_norm_rms_epsilon"),
                Value::F32(1e-5),
            ),
            (format!("{a}.rope.freq_base"), Value::F32(10000.0)),
        ];
        if self.mixers.iter().any(|m| *m != Mix::Attention) {
            meta.push((
                format!("{a}.attention.head_count_kv"),
                Value::Array(
                    self.heads_per_block()
                        .into_iter()
                        .map(|n| Value::U32(n as u32))
                        .collect(),
                ),
            ));
        } else {
            meta.push(u("attention.head_count_kv", KV_HEADS));
        }
        if self.mixers.contains(&Mix::ShortConv) {
            meta.push(u("shortconv.l_cache", CONV_KERNEL));
        }
        if self
            .mixers
            .iter()
            .any(|m| matches!(m, Mix::Mamba | Mix::Delta | Mix::DeltaPacked))
        {
            meta.push(u("ssm.conv_kernel", CONV_KERNEL));
            meta.push(u("ssm.inner_size", SSM_INNER));
            meta.push(u("ssm.state_size", SSM_STATE));
            meta.push(u("ssm.group_count", SSM_GROUPS));
            let heads = if self.mixers.contains(&Mix::Mamba) {
                MAMBA_HEADS
            } else {
                SSM_HEADS
            };
            meta.push(u("ssm.time_step_rank", heads));
        }
        if matches!(self.ffn, Ffn::Moe | Ffn::MoeUngated) {
            meta.push(u("expert_count", EXPERTS));
            meta.push(u("expert_used_count", EXPERTS_USED));
            meta.push(u("expert_feed_forward_length", EXPERT_FFN));
            if self.sigmoid_router {
                meta.push(u("expert_gating_func", 2));
            }
        }
        if self.sliding_window {
            meta.push(u("attention.sliding_window", 8));
        }
        if let Some(lanes) = self.rotary {
            meta.push(u("rope.dimension_count", lanes));
        }
        if self.scales {
            for (key, v) in [
                ("embedding_scale", 12.0f32),
                ("residual_scale", 0.22),
                ("attention.scale", 0.015625),
                ("logit_scale", 8.0),
            ] {
                meta.push((format!("{a}.{key}"), Value::F32(v)));
            }
        }
        if self.window {
            meta.push(u("attention.sliding_window", 8));
            meta.push((
                format!("{a}.attention.sliding_window_pattern"),
                Value::Array(
                    (0..self.mixers.len())
                        .map(|il| Value::Bool(il % 2 == 0))
                        .collect(),
                ),
            ));
            meta.push((format!("{a}.rope.freq_base_swa"), Value::F32(10000.0)));
        }
        if self.local_global {
            meta.push(u("attention.sliding_window", 8));
            meta.push((
                format!("{a}.attention.sliding_window_pattern"),
                Value::Array(
                    (0..self.mixers.len())
                        .map(|il| Value::Bool(il % 2 == 0))
                        .collect(),
                ),
            ));
            meta.push((format!("{a}.rope.freq_base_swa"), Value::F32(10000.0)));
        }
        meta
    }

    /// The tensors of the checkpoint, name and dimensions in the GGUF layout.
    fn tensors(&self) -> Vec<(String, Vec<usize>)> {
        let mut t: Vec<(String, Vec<usize>)> = Vec::new();
        let mut put = |name: &str, dims: &[usize]| t.push((name.to_string(), dims.to_vec()));
        put("token_embd.weight", &[VOCAB, HIDDEN]);
        if self.positions {
            put("position_embd.weight", &[POSITIONS, HIDDEN]);
        }
        if self.embedding_norm {
            put("token_embd_norm.weight", &[HIDDEN]);
        } else {
            put("output_norm.weight", &[HIDDEN]);
            if self.layer_norm {
                put("output_norm.bias", &[HIDDEN]);
            }
        }
        if !self.tied_head {
            put("output.weight", &[VOCAB, HIDDEN]);
        }
        let q_rows = HEADS * HEAD_DIM * if self.gated_query { 2 } else { 1 };
        let kv_rows = KV_HEADS * HEAD_DIM;
        for (il, mix) in self.mixers.iter().enumerate() {
            let p = format!("blk.{il}");
            let mut put = |name: &str, dims: &[usize]| {
                t.push((format!("{p}.{name}"), dims.to_vec()));
            };
            if !self.no_pre_norms {
                put("attn_norm.weight", &[HIDDEN]);
                if self.layer_norm {
                    put("attn_norm.bias", &[HIDDEN]);
                }
            }
            match mix {
                Mix::Attention => {
                    if self.fused_qkv {
                        put("attn_qkv.weight", &[q_rows + 2 * kv_rows, HIDDEN]);
                        if self.qkv_bias {
                            put("attn_qkv.bias", &[q_rows + 2 * kv_rows]);
                        }
                    } else {
                        put("attn_q.weight", &[q_rows, HIDDEN]);
                        put("attn_k.weight", &[kv_rows, HIDDEN]);
                        put("attn_v.weight", &[kv_rows, HIDDEN]);
                        if self.qkv_bias {
                            put("attn_q.bias", &[q_rows]);
                            put("attn_k.bias", &[kv_rows]);
                            put("attn_v.bias", &[kv_rows]);
                        }
                    }
                    if self.qk_norm && self.full_qk_norm {
                        put("attn_q_norm.weight", &[HEADS * HEAD_DIM]);
                        put("attn_k_norm.weight", &[kv_rows]);
                    } else if self.qk_norm {
                        put("attn_q_norm.weight", &[HEAD_DIM]);
                        put("attn_k_norm.weight", &[HEAD_DIM]);
                    }
                    let out = if self.sinks {
                        "attn_out"
                    } else {
                        "attn_output"
                    };
                    put(&format!("{out}.weight"), &[HIDDEN, HEADS * HEAD_DIM]);
                    if self.out_bias {
                        put(&format!("{out}.bias"), &[HIDDEN]);
                    }
                    if self.sinks {
                        put("attn_sinks", &[HEADS]);
                    }
                }
                Mix::ShortConv => {
                    put("shortconv.in_proj.weight", &[3 * HIDDEN, HIDDEN]);
                    put("shortconv.conv.weight", &[HIDDEN, CONV_KERNEL]);
                    put("shortconv.out_proj.weight", &[HIDDEN, HIDDEN]);
                }
                Mix::Mamba => {
                    let xbc = SSM_INNER + 2 * SSM_GROUPS * SSM_STATE;
                    put("ssm_in.weight", &[SSM_INNER + xbc + MAMBA_HEADS, HIDDEN]);
                    put("ssm_conv1d.weight", &[xbc, CONV_KERNEL]);
                    put("ssm_a", &[MAMBA_HEADS, 1]);
                    put("ssm_d", &[MAMBA_HEADS, 1]);
                    put("ssm_dt.bias", &[MAMBA_HEADS]);
                    put("ssm_norm.weight", &[SSM_GROUPS, SSM_INNER / SSM_GROUPS]);
                    put("ssm_out.weight", &[HIDDEN, SSM_INNER]);
                }
                Mix::Delta | Mix::DeltaPacked => {
                    let qkv = 2 * SSM_GROUPS * SSM_STATE + SSM_INNER;
                    put("attn_qkv.weight", &[qkv, HIDDEN]);
                    put("attn_gate.weight", &[SSM_INNER, HIDDEN]);
                    if *mix == Mix::DeltaPacked {
                        put("ssm_ba.weight", &[2 * SSM_HEADS, HIDDEN]);
                    } else {
                        put("ssm_alpha.weight", &[SSM_HEADS, HIDDEN]);
                        put("ssm_beta.weight", &[SSM_HEADS, HIDDEN]);
                    }
                    put("ssm_conv1d.weight", &[qkv, CONV_KERNEL]);
                    put("ssm_a", &[SSM_HEADS]);
                    put("ssm_dt", &[SSM_HEADS]);
                    put("ssm_norm.weight", &[SSM_INNER / SSM_HEADS]);
                    put("ssm_out.weight", &[HIDDEN, SSM_INNER]);
                }
                Mix::Nothing => {}
            }
            let ffn = self.ffn_of(il);
            if ffn != Ffn::None && !self.parallel && !self.no_pre_norms {
                let stem = if self.post_attention_as_ffn_norm {
                    "post_attention_norm"
                } else {
                    "ffn_norm"
                };
                put(&format!("{stem}.weight"), &[HIDDEN]);
                if self.layer_norm {
                    put(&format!("{stem}.bias"), &[HIDDEN]);
                }
            }
            if self.post_norms {
                put("post_attention_norm.weight", &[HIDDEN]);
                put("post_ffw_norm.weight", &[HIDDEN]);
            }
            if self.local_global {
                put("layer_output_scale.weight", &[1]);
            }
            match ffn {
                Ffn::Gated => {
                    put("ffn_gate.weight", &[FFN, HIDDEN]);
                    put("ffn_up.weight", &[FFN, HIDDEN]);
                    put("ffn_down.weight", &[HIDDEN, FFN]);
                }
                Ffn::FusedGateUp => {
                    put("ffn_up.weight", &[2 * FFN, HIDDEN]);
                    put("ffn_down.weight", &[HIDDEN, FFN]);
                }
                Ffn::Plain => {
                    put("ffn_up.weight", &[FFN, HIDDEN]);
                    put("ffn_down.weight", &[HIDDEN, FFN]);
                    if self.ffn_bias {
                        put("ffn_up.bias", &[FFN]);
                        put("ffn_down.bias", &[HIDDEN]);
                    }
                }
                Ffn::Moe | Ffn::MoeUngated => {
                    put("ffn_gate_inp.weight", &[EXPERTS, HIDDEN]);
                    if self.sinks {
                        put("ffn_gate_inp.bias", &[EXPERTS]);
                    }
                    if self.sigmoid_router {
                        put("exp_probs_b.bias", &[EXPERTS]);
                    }
                    if ffn == Ffn::Moe {
                        put("ffn_gate_exps.weight", &[EXPERTS, EXPERT_FFN, HIDDEN]);
                    }
                    put("ffn_up_exps.weight", &[EXPERTS, EXPERT_FFN, HIDDEN]);
                    put("ffn_down_exps.weight", &[EXPERTS, HIDDEN, EXPERT_FFN]);
                    if self.expert_bias {
                        put("ffn_gate_exps.bias", &[EXPERTS, EXPERT_FFN]);
                        put("ffn_up_exps.bias", &[EXPERTS, EXPERT_FFN]);
                        put("ffn_down_exps.bias", &[EXPERTS, HIDDEN]);
                    }
                    if self.shared_expert {
                        if ffn == Ffn::Moe {
                            put("ffn_gate_inp_shexp.weight", &[HIDDEN]);
                            put("ffn_gate_shexp.weight", &[EXPERT_FFN, HIDDEN]);
                        }
                        put("ffn_up_shexp.weight", &[EXPERT_FFN, HIDDEN]);
                        put("ffn_down_shexp.weight", &[HIDDEN, EXPERT_FFN]);
                    }
                }
                Ffn::None => {}
            }
        }
        t
    }
}

/// Deterministic values: norms near one, decays negative, projections scaled by their fan-in.
fn values(name: &str, dims: &[usize], seed: u64) -> Vec<f32> {
    let n: usize = dims.iter().product();
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut next = move || {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    let fan_in = *dims.last().unwrap() as f32;
    (0..n)
        .map(|_| {
            let r = next();
            if name.ends_with("norm.weight") {
                1.0 + 0.1 * r
            } else if name.ends_with("ssm_a") {
                -(0.6 + 0.3 * r)
            } else if name.ends_with("ssm_d") || name.ends_with("layer_output_scale.weight") {
                1.0
            } else if name.ends_with(".bias") || name.ends_with("ssm_dt") || name.ends_with("sinks")
            {
                0.1 * r
            } else {
                r * 1.5 / fan_in.sqrt()
            }
        })
        .collect()
}

fn scratch(tag: &str) -> std::path::PathBuf {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("loken-loading-{tag}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A host-only plan over `layers` blocks: the placement every test shares.
fn host_plan(layers: usize) -> HeteroPlan {
    HeteroPlan::calculate_with_kv(layers, 1 << 20, &[], &[], 1.0, 0)
}

/// Asserts that `model` runs the prompt to finite logits of the vocabulary's size.
fn assert_runs(label: &str, model: &mut GenericHeteroTransformer, vocab: usize) -> Vec<f32> {
    let input = Tensor::from_vec(PROMPT.to_vec(), (1usize, PROMPT.len()), &Device::Cpu).unwrap();
    let logits = model
        .forward(&input, 0)
        .unwrap_or_else(|e| panic!("{label}: forward failed: {e}"))
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(
        logits.len(),
        vocab,
        "{label}: logits are one row of the vocabulary"
    );
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "{label}: logits are finite"
    );
    assert!(
        logits.iter().any(|v| *v != logits[0]),
        "{label}: logits are not constant"
    );
    logits
}

/// The recipe written as a GGUF file and loaded as the server loads one.
fn load_gguf(recipe: &Recipe) {
    let built = build_gguf(recipe, None);
    let mut model = built.unwrap_or_else(|e| panic!("{}: load failed: {e}", recipe.arch));
    assert_runs(recipe.arch, &mut model, VOCAB);
}

/// The recipe's file built into a model, with the tensor `without` left out of the file.
fn build_gguf(recipe: &Recipe, without: Option<&str>) -> Result<GenericHeteroTransformer> {
    build_gguf_with(recipe, without, &BTreeMap::new())
}

/// A tensor of the file in a format and bytes of its own.
type Replaced = BTreeMap<String, (GgmlDType, Vec<u8>)>;

/// [`build_gguf`] with the tensors in `replaced` written as given.
fn build_gguf_with(
    recipe: &Recipe,
    without: Option<&str>,
    replaced: &Replaced,
) -> Result<GenericHeteroTransformer> {
    let dir = scratch(&format!(
        "{}-{}-{}",
        recipe.arch,
        without.is_some(),
        replaced.len()
    ));
    let path = dir.join("m.gguf");
    let entries: Vec<GgufEntry> = recipe
        .tensors()
        .into_iter()
        .enumerate()
        .filter(|(_, (name, _))| Some(name.as_str()) != without)
        .map(|(i, (name, dims))| {
            if let Some((dtype, data)) = replaced.get(&name) {
                return GgufEntry {
                    name,
                    dims,
                    dtype: *dtype,
                    data: data.clone(),
                };
            }
            let data = values(&name, &dims, i as u64 + 1)
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            GgufEntry {
                name,
                dims,
                dtype: GgmlDType::F32,
                data,
            }
        })
        .collect();
    write_gguf_with_metadata(&path, &recipe.metadata(), &entries).unwrap();
    let file = std::fs::File::open(&path).unwrap();
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file).unwrap() });
    let bytes: &[u8] = &mmap[..];
    // The file says what the recipe meant: each block's mixer is the one written.
    let content =
        gguf_file::Content::read_mapped(&mut std::io::Cursor::new(bytes), mmap.clone()).unwrap();
    for (il, mix) in recipe.mixers.iter().enumerate() {
        use super::mixer::{layer_mixer_kind, MixerKind};
        let want = match mix {
            Mix::Attention => MixerKind::Attention,
            Mix::ShortConv => MixerKind::ShortConv,
            Mix::Mamba => MixerKind::Ssm,
            Mix::Delta | Mix::DeltaPacked => MixerKind::DeltaNet,
            Mix::Nothing => MixerKind::Absent,
        };
        assert_eq!(
            layer_mixer_kind(&content, il),
            want,
            "{} block {il}",
            recipe.arch
        );
    }
    let weights = GenericWeights::Gguf {
        mmap_bytes: bytes,
        mmap: &mmap,
    };
    let layers = recipe.mixers.len();
    let built = weights.build(
        &HashMap::new(),
        &host_plan(layers),
        crate::inference::engine::llm_engine::KvQuant::Off,
        None,
        false,
        false,
    );
    let _ = std::fs::remove_dir_all(&dir);
    built
}

/// The wide-residual architecture, whose indexer, n-gram features, delta-rule blocks and
/// routed mixture a recipe does not write, loaded from its seeded toy file.
#[test]
fn qwen4exp_loads_from_gguf() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/qwen4exp/toy.gguf");
    let file = std::fs::File::open(path).unwrap();
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file).unwrap() });
    let bytes: &[u8] = &mmap[..];
    let content =
        gguf_file::Content::read_mapped(&mut std::io::Cursor::new(bytes), mmap.clone()).unwrap();
    let layers = content.metadata["qwen4exp.block_count"].to_u32().unwrap() as usize;
    let vocab = content.tensor_infos["token_embd.weight"].shape.dims()[0];
    let mut model = GenericWeights::Gguf {
        mmap_bytes: bytes,
        mmap: &mmap,
    }
    .build(
        &HashMap::new(),
        &host_plan(layers),
        crate::inference::engine::llm_engine::KvQuant::Off,
        None,
        false,
        false,
    )
    .unwrap_or_else(|e| panic!("qwen4exp: load failed: {e}"));
    assert_runs("qwen4exp", &mut model, vocab);
}

/// The tests above can fail: a file missing a tensor its blocks need is refused.
#[test]
fn a_file_missing_a_tensor_of_its_block_is_refused() {
    for (arch, missing) in [
        ("llama", "blk.1.ffn_down.weight"),
        ("qwen35", "blk.0.ssm_out.weight"),
        ("nemotron_h_moe", "blk.0.ssm_conv1d.weight"),
    ] {
        let recipe = all_recipes().into_iter().find(|r| r.arch == arch).unwrap();
        assert!(
            build_gguf(&recipe, Some(missing)).is_err(),
            "{arch} without {missing} loads"
        );
    }
}

/// Every architecture the loader holds facts for has a recipe above.
#[test]
fn every_architecture_has_a_loading_test() {
    let tested: std::collections::BTreeSet<&str> = all_recipes().iter().map(|r| r.arch).collect();
    let known: std::collections::BTreeSet<&str> =
        crate::inference::model_facts::architectures().collect();
    // One architecture's structure is too wide for a recipe: its test loads a seeded toy file.
    let own_file: std::collections::BTreeSet<&str> = ["qwen4exp"].into_iter().collect();
    let covered = tested.union(&own_file).copied().collect();
    assert_eq!(known, covered);
    safetensors_coverage(&known);
}

macro_rules! gguf_loads {
    ($($test:ident: $recipe:expr;)*) => {
        $(
            #[test]
            fn $test() {
                load_gguf(&$recipe);
            }
        )*

        /// Every recipe the tests above load, in the order they are listed.
        fn all_recipes() -> Vec<Recipe> {
            vec![$($recipe),*]
        }
    };
}

fn dense(arch: &'static str) -> Recipe {
    Recipe::dense(arch)
}

fn hybrid(arch: &'static str, mixers: Vec<Mix>) -> Recipe {
    Recipe {
        mixers,
        ..Recipe::dense(arch)
    }
}

fn gemma() -> Recipe {
    dense("gemma").post().tied()
}

gguf_loads! {
    llama_loads_from_gguf: dense("llama");
    mistral_loads_from_gguf: dense("mistral");
    mistral3_loads_from_gguf: dense("mistral3");
    smollm3_loads_from_gguf: Recipe { mixers: vec![Mix::Attention; 4], ..dense("smollm3") };
    ernie4_5_loads_from_gguf: dense("ernie4_5").tied();
    granite_loads_from_gguf: Recipe { scales: true, ..dense("granite") };
    granite3_loads_from_gguf: dense("granite3");
    granitemoe_loads_from_gguf: dense("granitemoe").moe();
    yi_loads_from_gguf: dense("yi");
    internlm2_loads_from_gguf: dense("internlm2");
    starcoder_loads_from_gguf: dense("starcoder").classic().fused().bias().positions();
    starcoder2_loads_from_gguf: dense("starcoder2").classic().bias();
    qwen2_loads_from_gguf: dense("qwen2").bias();
    qwen3_loads_from_gguf: dense("qwen3").qk();
    qwen3moe_loads_from_gguf: dense("qwen3moe").qk().moe();
    phi_loads_from_gguf: dense("phi").classic().parallel().fused().bias();
    phi2_loads_from_gguf: Recipe { rotary: Some(16), ..dense("phi2").classic().parallel().fused().bias() };
    phi3_loads_from_gguf: dense("phi3").fused().ffn(Ffn::FusedGateUp);
    phi4_loads_from_gguf: dense("phi4").fused().ffn(Ffn::FusedGateUp);
    chatglm_loads_from_gguf: Recipe { rotary: Some(16), ..dense("chatglm").fused().bias().ffn(Ffn::FusedGateUp) };
    glm4_loads_from_gguf: Recipe { rotary: Some(16), ..dense("glm4").fused().bias().post().ffn(Ffn::FusedGateUp) };
    stablelm_loads_from_gguf: Recipe { ffn_bias: false, ..dense("stablelm").classic().ffn(Ffn::Gated) };
    olmo2_loads_from_gguf: Recipe { post_norms: true, full_qk_norm: true, no_pre_norms: true, ..dense("olmo2").qk() };
    olmoe_loads_from_gguf: Recipe { full_qk_norm: true, ..dense("olmoe").qk().moe() };
    moondream_loads_from_gguf: dense("moondream").classic().parallel().fused().bias();
    falcon_loads_from_gguf: Recipe { ffn_bias: false, out_bias: false, ..dense("falcon").classic().parallel().fused() };
    falcon3_loads_from_gguf: dense("falcon3");
    dbrx_loads_from_gguf: dense("dbrx").fused().moe();
    orion_loads_from_gguf: Recipe { ffn_bias: false, out_bias: false, ..dense("orion").classic().ffn(Ffn::Gated) };
    mpt_loads_from_gguf: Recipe { ffn_bias: false, out_bias: false, ..dense("mpt").classic().fused() };
    gemma_loads_from_gguf: Recipe { post_norms: false, ..gemma() };
    gemma2_loads_from_gguf: Recipe { arch: "gemma2", ..gemma() };
    gemma3_loads_from_gguf: Recipe { arch: "gemma3", window: true, ..gemma().qk() };
    gemma4_loads_from_gguf: Recipe { arch: "gemma4", local_global: true, ..gemma().qk() };
    gptoss_loads_from_gguf: Recipe { sinks: true, out_bias: true, qkv_bias: true, sliding_window: true, expert_bias: true, ..dense("gptoss").moe() };
    lfm2_loads_from_gguf: Recipe { embedding_norm: true, tied_head: true, ..hybrid("lfm2", vec![Mix::ShortConv, Mix::Attention, Mix::ShortConv, Mix::Attention]).qk() };
    lfm2moe_loads_from_gguf: Recipe {
        embedding_norm: true,
        tied_head: true,
        sigmoid_router: true,
        dense_blocks: vec![0],
        ..hybrid("lfm2moe", vec![Mix::ShortConv, Mix::Attention, Mix::ShortConv]).qk().moe()
    };
    qwen35_loads_from_gguf: Recipe {
        gated_query: true,
        post_attention_as_ffn_norm: true,
        ..hybrid("qwen35", vec![Mix::Delta, Mix::Delta, Mix::Delta, Mix::Attention]).qk()
    };
    qwen35moe_loads_from_gguf: Recipe {
        gated_query: true,
        post_attention_as_ffn_norm: true,
        shared_expert: true,
        ..hybrid("qwen35moe", vec![Mix::Delta, Mix::Attention]).qk().moe()
    };
    qwen3next_loads_from_gguf: Recipe {
        gated_query: true,
        post_attention_as_ffn_norm: true,
        shared_expert: true,
        ..hybrid("qwen3next", vec![Mix::DeltaPacked, Mix::Attention]).qk().moe()
    };
    nemotron_h_moe_loads_from_gguf: Recipe {
        ffn_blocks: vec![1],
        ffn: Ffn::MoeUngated,
        parallel: true,
        shared_expert: true,
        sigmoid_router: true,
        ..hybrid("nemotron_h_moe", vec![Mix::Mamba, Mix::Nothing, Mix::Attention])
    };
}

/// How a checkpoint directory names and shapes its model.
#[derive(Clone, Copy)]
struct Checkpoint {
    /// `architectures[0]` of `config.json`; unused by a native layout.
    class: &'static str,
    /// The text model sits in a wrapper (`language_model.`) with its fields in `text_config`.
    wrapped: bool,
    /// The weights are the vendor's own: `params.json` and `consolidated.safetensors`.
    native: bool,
    /// The embedding table is stored in the speech tower's namespace (native layout only).
    table_in_tower: bool,
    qkv_bias: bool,
    qk_norm: bool,
    tied: bool,
    bf16: bool,
}

impl Checkpoint {
    const fn plain(class: &'static str) -> Self {
        Self {
            class,
            wrapped: false,
            native: false,
            table_in_tower: false,
            qkv_bias: false,
            qk_norm: false,
            tied: false,
            bf16: false,
        }
    }

    /// The checkpoint's tensors under the names its layout gives them, dimensions as stored.
    fn tensors(&self) -> Vec<(String, Vec<usize>)> {
        let kv = KV_HEADS * HEAD_DIM;
        let q = HEADS * HEAD_DIM;
        let (model, head) = match (self.native, self.wrapped) {
            (true, _) => (String::new(), String::new()),
            (false, true) => (
                "language_model.model.".to_string(),
                "language_model.".to_string(),
            ),
            (false, false) => ("model.".to_string(), String::new()),
        };
        let mut t: Vec<(String, Vec<usize>)> = Vec::new();
        let table = match (self.native, self.table_in_tower) {
            (true, true) => "mm_whisper_embeddings.tok_embeddings.weight".to_string(),
            (true, false) => "tok_embeddings.weight".to_string(),
            _ => format!("{model}embed_tokens.weight"),
        };
        t.push((table, vec![VOCAB, HIDDEN]));
        t.push((
            if self.native {
                "norm.weight".to_string()
            } else {
                format!("{model}norm.weight")
            },
            vec![HIDDEN],
        ));
        if !self.tied {
            t.push((
                if self.native {
                    "output.weight".to_string()
                } else {
                    format!("{head}lm_head.weight")
                },
                vec![VOCAB, HIDDEN],
            ));
        }
        for il in 0..2 {
            let parts: Vec<(&str, &str, Vec<usize>)> = vec![
                (
                    "input_layernorm.weight",
                    "attention_norm.weight",
                    vec![HIDDEN],
                ),
                (
                    "post_attention_layernorm.weight",
                    "ffn_norm.weight",
                    vec![HIDDEN],
                ),
                (
                    "self_attn.q_proj.weight",
                    "attention.wq.weight",
                    vec![q, HIDDEN],
                ),
                (
                    "self_attn.k_proj.weight",
                    "attention.wk.weight",
                    vec![kv, HIDDEN],
                ),
                (
                    "self_attn.v_proj.weight",
                    "attention.wv.weight",
                    vec![kv, HIDDEN],
                ),
                (
                    "self_attn.o_proj.weight",
                    "attention.wo.weight",
                    vec![HIDDEN, q],
                ),
                (
                    "mlp.gate_proj.weight",
                    "feed_forward.w1.weight",
                    vec![FFN, HIDDEN],
                ),
                (
                    "mlp.up_proj.weight",
                    "feed_forward.w3.weight",
                    vec![FFN, HIDDEN],
                ),
                (
                    "mlp.down_proj.weight",
                    "feed_forward.w2.weight",
                    vec![HIDDEN, FFN],
                ),
            ];
            for (hf, native, dims) in parts {
                t.push((
                    if self.native {
                        format!("layers.{il}.{native}")
                    } else {
                        format!("{model}layers.{il}.{hf}")
                    },
                    dims,
                ));
            }
            if self.qkv_bias {
                for (name, rows) in [("q", q), ("k", kv), ("v", kv)] {
                    t.push((
                        format!("{model}layers.{il}.self_attn.{name}_proj.bias"),
                        vec![rows],
                    ));
                }
            }
            if self.qk_norm {
                for name in ["q", "k"] {
                    t.push((
                        format!("{model}layers.{il}.self_attn.{name}_norm.weight"),
                        vec![HEAD_DIM],
                    ));
                }
            }
        }
        if self.wrapped {
            // The speech tower of a composite release shares the shard and is not the text model's.
            t.push(("audio_tower.conv1.weight".to_string(), vec![8, 4]));
        }
        t
    }

    fn config_json(&self) -> String {
        let text = format!(
            r#""hidden_size": {HIDDEN}, "intermediate_size": {FFN}, "num_hidden_layers": 2,
            "num_attention_heads": {HEADS}, "num_key_value_heads": {KV_HEADS},
            "head_dim": {HEAD_DIM}, "vocab_size": {VOCAB}, "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5, "tie_word_embeddings": {}"#,
            self.tied
        );
        if self.wrapped {
            format!(
                r#"{{"architectures": ["{}"], "vocab_size": {VOCAB}, "text_config": {{{text}}}}}"#,
                self.class
            )
        } else {
            format!(r#"{{"architectures": ["{}"], {text}}}"#, self.class)
        }
    }

    fn params_json(&self) -> String {
        format!(
            r#"{{"dim": {HIDDEN}, "n_layers": 2, "head_dim": {HEAD_DIM}, "hidden_dim": {FFN},
            "n_heads": {HEADS}, "n_kv_heads": {KV_HEADS}, "rope_theta": 10000.0,
            "norm_eps": 1e-5, "vocab_size": {VOCAB}, "tied_embeddings": {}}}"#,
            self.tied
        )
    }
}

/// The values of a tensor in the stored width's bytes.
fn stored(bf16: bool, vals: &[f32]) -> Vec<u8> {
    if bf16 {
        vals.iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    } else {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

/// One tensor of a safetensors file: name, stored type, dimensions and bytes.
type Stored = (String, &'static str, Vec<usize>, Vec<u8>);

/// A safetensors file of `tensors`, written in the format's own layout.
fn write_stored(path: &std::path::Path, tensors: &[Stored]) {
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, dtype, dims, bytes) in tensors {
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": dtype,
                "shape": dims,
                "data_offsets": [data.len(), data.len() + bytes.len()],
            }),
        );
        data.extend(bytes);
    }
    let mut json = serde_json::to_vec(&header).unwrap();
    while !json.len().is_multiple_of(8) {
        json.push(b' ');
    }
    let mut out = (json.len() as u64).to_le_bytes().to_vec();
    out.extend(json);
    out.extend(data);
    std::fs::write(path, out).unwrap();
}

/// A safetensors file of `tensors` as floats, in `bf16` or F32 (the speech tower stays F32).
fn write_safetensors(path: &std::path::Path, tensors: &[(String, Vec<usize>)], bf16: bool) {
    let stored_tensors: Vec<Stored> = tensors
        .iter()
        .enumerate()
        .map(|(i, (name, dims))| {
            let wide = bf16 && !name.starts_with("audio_tower");
            let vals = values(name, dims, i as u64 + 1);
            (
                name.clone(),
                if wide { "BF16" } else { "F32" },
                dims.clone(),
                stored(wide, &vals),
            )
        })
        .collect();
    write_stored(path, &stored_tensors);
}

/// The checkpoint written as a directory and loaded as the server loads one.
fn load_checkpoint(tag: &str, ck: Checkpoint) {
    use crate::inference::load::awq_loader as awql;
    let dir = scratch(tag);
    let shard = if ck.native {
        "consolidated.safetensors"
    } else {
        "model.safetensors"
    };
    write_safetensors(&dir.join(shard), &ck.tensors(), ck.bf16);
    if ck.native {
        std::fs::write(dir.join("params.json"), ck.params_json()).unwrap();
    } else {
        std::fs::write(dir.join("config.json"), ck.config_json()).unwrap();
    }
    std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    let dir_str = dir.to_string_lossy().to_string();
    let found = awql::find_dense_model_dir(&dir_str);
    assert_eq!(
        found.as_deref(),
        Some(dir_str.as_str()),
        "{tag}: the directory is a dense checkpoint"
    );
    let (native, shards) = awql::checkpoint_shards(&dir).unwrap();
    assert_eq!(native, ck.native, "{tag}: layout");
    let built = (|| {
        let (header, _) = crate::inference::engine::llm_engine::awq_checkpoint(&dir_str)
            .map_err(|e| crate::tensor::Error::msg(e.to_string()))?;
        let tag_of = |c: &str| match c {
            "Qwen2ForCausalLM" => "qwen2",
            "Qwen3ForCausalLM" => "qwen3",
            _ => "llama",
        };
        let arch = header
            .metadata
            .get("general.architecture")
            .map(|v| v.to_string().unwrap().clone());
        assert_eq!(
            arch.as_deref(),
            Some(tag_of(ck.class)),
            "the header names the text model's architecture"
        );
        GenericWeights::Awq {
            dir: &dir_str,
            shards: &shards,
        }
        .build(
            &HashMap::new(),
            &host_plan(2),
            crate::inference::engine::llm_engine::KvQuant::Off,
            None,
            false,
            false,
        )
    })();
    let _ = std::fs::remove_dir_all(&dir);
    let mut model = built.unwrap_or_else(|e| panic!("{tag}: load failed: {e}"));
    assert_runs(tag, &mut model, VOCAB);
}

macro_rules! checkpoint_loads {
    ($($test:ident: $ck:expr;)*) => {
        $(
            #[test]
            fn $test() {
                load_checkpoint(stringify!($test), $ck);
            }
        )*
    };
}

checkpoint_loads! {
    llama_loads_from_safetensors: Checkpoint::plain("LlamaForCausalLM");
    mistral_loads_from_safetensors: Checkpoint { bf16: true, ..Checkpoint::plain("MistralForCausalLM") };
    mistral3_loads_from_safetensors: Checkpoint { tied: true, ..Checkpoint::plain("Mistral3ForCausalLM") };
    qwen2_loads_from_safetensors: Checkpoint { qkv_bias: true, ..Checkpoint::plain("Qwen2ForCausalLM") };
    qwen3_loads_from_safetensors: Checkpoint { qk_norm: true, ..Checkpoint::plain("Qwen3ForCausalLM") };
    mistral3_composite_loads_from_safetensors: Checkpoint { wrapped: true, ..Checkpoint::plain("MistralForConditionalGeneration") };
    voxtral_loads_from_safetensors: Checkpoint { wrapped: true, bf16: true, ..Checkpoint::plain("VoxtralForConditionalGeneration") };
    llama_loads_from_native_layout: Checkpoint { native: true, ..Checkpoint::plain("") };
    mistral_loads_from_native_layout: Checkpoint { native: true, tied: true, bf16: true, ..Checkpoint::plain("") };
    voxtral_loads_from_native_layout: Checkpoint { native: true, table_in_tower: true, ..Checkpoint::plain("") };
}

/// Group size of the quantised checkpoints: divides every reduction of the tiny model.
const AWQ_GROUP: usize = 32;

/// A 4-bit AWQ checkpoint of the framework's layout: every projection is packed as
/// `qweight`, `qzeros` and `scales`; norms, biases, the table and the head stay float.
fn write_awq(dir: &std::path::Path, class: &'static str, qkv_bias: bool, qk_norm: bool) {
    let ck = Checkpoint {
        qkv_bias,
        qk_norm,
        ..Checkpoint::plain(class)
    };
    let mut out: Vec<Stored> = Vec::new();
    for (i, (name, dims)) in ck.tensors().into_iter().enumerate() {
        let vals = values(&name, &dims, i as u64 + 1);
        match name.strip_suffix(".weight") {
            Some(stem) if stem.ends_with("_proj") => {
                let (n, k) = (dims[0], dims[1]);
                let words = |count: usize, f: &mut dyn FnMut(usize) -> u32| -> Vec<u8> {
                    (0..count).flat_map(|w| f(w).to_le_bytes()).collect()
                };
                // Nibbles 1..=7 around a zero point of 8: small signed weights.
                let mut nibbles = |w: usize| -> u32 {
                    (0..8).fold(0u32, |acc, j| {
                        acc | ((1 + ((w * 8 + j + i) % 15)) as u32) << (4 * j)
                    })
                };
                out.push((
                    format!("{stem}.qweight"),
                    "I32",
                    vec![k, n / 8],
                    words(k * n / 8, &mut nibbles),
                ));
                out.push((
                    format!("{stem}.qzeros"),
                    "I32",
                    vec![k / AWQ_GROUP, n / 8],
                    words(k / AWQ_GROUP * n / 8, &mut |_| 0x8888_8888),
                ));
                let scales: Vec<u8> = (0..k / AWQ_GROUP * n)
                    .flat_map(|j| half::f16::from_f32(0.02 + 0.01 * (j % 3) as f32).to_le_bytes())
                    .collect();
                out.push((
                    format!("{stem}.scales"),
                    "F16",
                    vec![k / AWQ_GROUP, n],
                    scales,
                ));
            }
            _ => out.push((name, "F32", dims, stored(false, &vals))),
        }
    }
    write_stored(&dir.join("model.safetensors"), &out);
    let config = ck.config_json();
    let config = format!(
        r#"{}, "quantization_config": {{"quant_method": "awq", "bits": 4, "group_size": {AWQ_GROUP}}}}}"#,
        config.trim_end().trim_end_matches('}')
    );
    std::fs::write(dir.join("config.json"), config).unwrap();
    std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
}

fn load_awq(tag: &str, class: &'static str, qkv_bias: bool, qk_norm: bool) {
    use crate::inference::load::awq_loader as awql;
    let dir = scratch(tag);
    write_awq(&dir, class, qkv_bias, qk_norm);
    let dir_str = dir.to_string_lossy().to_string();
    let found = awql::find_awq_model_dir(&dir_str);
    assert_eq!(
        found.as_deref(),
        Some(dir_str.as_str()),
        "{tag}: the directory is an AWQ checkpoint"
    );
    let (_, shards) = awql::checkpoint_shards(&dir).unwrap();
    let built = GenericWeights::Awq {
        dir: &dir_str,
        shards: &shards,
    }
    .build(
        &HashMap::new(),
        &host_plan(2),
        crate::inference::engine::llm_engine::KvQuant::Off,
        None,
        false,
        false,
    );
    let _ = std::fs::remove_dir_all(&dir);
    let mut model = built.unwrap_or_else(|e| panic!("{tag}: load failed: {e}"));
    assert_runs(tag, &mut model, VOCAB);
}

#[test]
fn llama_loads_from_awq_safetensors() {
    load_awq("llama_awq", "LlamaForCausalLM", false, false);
}

#[test]
fn mistral_loads_from_awq_safetensors() {
    load_awq("mistral_awq", "MistralForCausalLM", false, false);
}

#[test]
fn qwen2_loads_from_awq_safetensors() {
    load_awq("qwen2_awq", "Qwen2ForCausalLM", true, false);
}

#[test]
fn qwen3_loads_from_awq_safetensors() {
    load_awq("qwen3_awq", "Qwen3ForCausalLM", false, true);
}

// ---------------------------------------------------------------------------------------------
// The checkpoints publishers ship. Each test writes the weights of an architecture's recipe in
// the publisher's tensor names and layout, loads the directory as the server does, and compares
// its logits with the same weights loaded from the recipe's GGUF file.
// ---------------------------------------------------------------------------------------------

type Weights = BTreeMap<String, (Vec<usize>, Vec<f32>)>;

/// The recipe's tensors with the values its GGUF file holds.
fn recipe_weights(recipe: &Recipe) -> Weights {
    recipe
        .tensors()
        .into_iter()
        .enumerate()
        .map(|(i, (name, dims))| {
            let vals = values(&name, &dims, i as u64 + 1);
            (name, (dims, vals))
        })
        .collect()
}

/// How a publisher's rows pair a head's rotary lanes against the recipe's file.
#[derive(Clone, Copy, PartialEq)]
enum Pairing {
    Same,
    /// The file pairs adjacent lanes, the publisher's rows pair lane `i` with `i + rot / 2`.
    AdjacentToHalf,
    /// The reverse.
    HalfToAdjacent,
}

/// What the publisher stores for a tensor of the recipe.
#[derive(Clone, Copy)]
enum Xf {
    Id,
    /// A norm scale stored less one.
    Minus1,
    /// `[c, k]` stored `[c, 1, k]`.
    Unsqueeze,
    /// A decay stored as `ln(-a)`, flat.
    LogNeg,
    Flat,
    /// `[n]` stored `[1, n]`.
    Row,
}

/// How the publisher stores the attention projections.
#[derive(Clone, Copy)]
enum Qkv {
    Split,
    /// One tensor of query, key and value rows in turn.
    Contiguous(&'static str),
    /// One tensor of rows grouped by key-value head.
    Grouped(&'static str),
}

/// How the publisher stores routed experts; the names are relative to a block.
#[derive(Clone, Copy)]
enum Experts {
    /// One tensor per expert and projection (`{e}` is the index); an ungated stack has no gate.
    PerExpert {
        gate: &'static str,
        up: &'static str,
        down: &'static str,
    },
    /// `[experts, 2 * rows, cols]` with the gate rows, then the up rows; the down as is.
    FusedHalves {
        gate_up: &'static str,
        down: &'static str,
    },
    /// Flat `[experts * rows, cols]` stacks; the down stored transposed per expert.
    Flat {
        gate: &'static str,
        up: &'static str,
        down: &'static str,
    },
    /// `[experts, cols, 2 * rows]` with gate and up columns alternating, biases beside it,
    /// and the down transposed per expert.
    Alternating {
        gate_up: &'static str,
        down: &'static str,
    },
}

/// A publisher's naming and layout.
#[derive(Clone, Copy)]
struct Style {
    pairing: Pairing,
    /// Lanes of a head the rotation covers.
    rot: usize,
    /// Prefix of a block, `{l}` standing for its index.
    layer: &'static str,
    /// The recipe's tensor (a block's by its tail) against the publisher's name (a block's
    /// relative to its prefix) and what it stores.
    names: &'static [&'static [(&'static str, &'static str, Xf)]],
    qkv: Qkv,
    experts: Option<Experts>,
    /// The delta rule's `qkvz` projection, one tensor grouped by key head.
    delta_packed: Option<&'static str>,
}

use Xf::{Flat, Id, LogNeg, Minus1, Row, Unsqueeze};

/// The names the llama family uses for what a block holds.
const LLAMA_BLOCK: [(&str, &str, Xf); 14] = [
    ("attn_norm.weight", "input_layernorm.weight", Id),
    ("ffn_norm.weight", "post_attention_layernorm.weight", Id),
    ("attn_q.weight", "self_attn.q_proj.weight", Id),
    ("attn_k.weight", "self_attn.k_proj.weight", Id),
    ("attn_v.weight", "self_attn.v_proj.weight", Id),
    ("attn_output.weight", "self_attn.o_proj.weight", Id),
    ("attn_q.bias", "self_attn.q_proj.bias", Id),
    ("attn_k.bias", "self_attn.k_proj.bias", Id),
    ("attn_v.bias", "self_attn.v_proj.bias", Id),
    ("attn_q_norm.weight", "self_attn.q_norm.weight", Id),
    ("attn_k_norm.weight", "self_attn.k_norm.weight", Id),
    ("ffn_gate.weight", "mlp.gate_proj.weight", Id),
    ("ffn_up.weight", "mlp.up_proj.weight", Id),
    ("ffn_down.weight", "mlp.down_proj.weight", Id),
];

/// Layer norms with biases and the output bias of the classic families.
const LLAMA_BIASES: [(&str, &str, Xf); 3] = [
    ("attn_norm.bias", "input_layernorm.bias", Id),
    ("ffn_norm.bias", "post_attention_layernorm.bias", Id),
    ("attn_output.bias", "self_attn.o_proj.bias", Id),
];

/// The recipe's tensor rows in the order the publisher pairs lanes.
fn repair(vals: &[f32], dims: &[usize], rot: usize, pairing: Pairing) -> Vec<f32> {
    if pairing == Pairing::Same {
        return vals.to_vec();
    }
    let rows = dims[0];
    let cols = vals.len() / rows;
    let heads = rows / HEAD_DIM.min(rows);
    let hd = rows / heads;
    let mut out = vals.to_vec();
    let at = |r: usize| r * cols..(r + 1) * cols;
    for h in 0..heads {
        for i in 0..rot / 2 {
            let (a, b) = (h * hd + i, h * hd + rot / 2 + i);
            let (x, y) = (h * hd + 2 * i, h * hd + 2 * i + 1);
            match pairing {
                Pairing::AdjacentToHalf => {
                    out[at(a)].copy_from_slice(&vals[at(x)]);
                    out[at(b)].copy_from_slice(&vals[at(y)]);
                }
                _ => {
                    out[at(x)].copy_from_slice(&vals[at(a)]);
                    out[at(y)].copy_from_slice(&vals[at(b)]);
                }
            }
        }
    }
    out
}

/// Rows of a matrix of `cols` columns.
fn take_rows(vals: &[f32], cols: usize, rows: std::ops::Range<usize>) -> Vec<f32> {
    vals[rows.start * cols..rows.end * cols].to_vec()
}

/// A tensor of the checkpoint: name, dimensions, values.
type Out = (String, Vec<usize>, Vec<f32>);

/// The recipe's weights as the publisher stores them.
fn hf_tensors(recipe: &Recipe, st: &Style, w: &Weights) -> Vec<Out> {
    let mut out: Vec<Out> = Vec::new();
    let mut done: std::collections::HashSet<String> = std::collections::HashSet::new();
    let store = |xf: Xf, dims: &[usize], vals: &[f32]| -> (Vec<usize>, Vec<f32>) {
        match xf {
            Id => (dims.to_vec(), vals.to_vec()),
            Minus1 => (dims.to_vec(), vals.iter().map(|v| v - 1.0).collect()),
            Unsqueeze => (vec![dims[0], 1, dims[1]], vals.to_vec()),
            LogNeg => (vec![vals.len()], vals.iter().map(|v| (-v).ln()).collect()),
            Flat => (vec![vals.len()], vals.to_vec()),
            Row => (vec![1, vals.len()], vals.to_vec()),
        }
    };
    let name_of = |layer: Option<usize>, tail: &str| -> (String, Xf) {
        let (_, part, xf) = st
            .names
            .iter()
            .flat_map(|table| table.iter())
            .find(|(g, _, _)| *g == tail)
            .unwrap_or_else(|| panic!("no publisher name for {tail}"));
        let name = match layer {
            Some(l) => format!("{}{part}", st.layer.replace("{l}", &l.to_string())),
            None => part.to_string(),
        };
        (name, *xf)
    };
    let block_prefix = |l: usize| st.layer.replace("{l}", &l.to_string());
    for (il, mix) in recipe.mixers.iter().enumerate() {
        let b = |tail: &str| format!("blk.{il}.{tail}");
        let get = |tail: &str| w.get(&b(tail));
        if *mix == Mix::Attention {
            for suffix in ["weight", "bias"] {
                let (q, k, v) = if let Some((dims, vals)) = get(&format!("attn_qkv.{suffix}")) {
                    let cols = if suffix == "weight" { dims[1] } else { 1 };
                    let (qn, kn) = (HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
                    let rows = |r: std::ops::Range<usize>| take_rows(vals, cols, r);
                    (rows(0..qn), rows(qn..qn + kn), rows(qn + kn..qn + 2 * kn))
                } else if let (Some(q), Some(k), Some(v)) = (
                    get(&format!("attn_q.{suffix}")),
                    get(&format!("attn_k.{suffix}")),
                    get(&format!("attn_v.{suffix}")),
                ) {
                    (q.1.clone(), k.1.clone(), v.1.clone())
                } else {
                    continue;
                };
                for t in ["attn_qkv", "attn_q", "attn_k", "attn_v"] {
                    done.insert(b(&format!("{t}.{suffix}")));
                }
                let cols = q.len() / (HEADS * HEAD_DIM * if recipe.gated_query { 2 } else { 1 });
                let q = repair(&q, &[q.len() / cols.max(1), cols], st.rot, st.pairing);
                let k = repair(&k, &[k.len() / cols.max(1), cols], st.rot, st.pairing);
                let dims_of = |rows: usize| {
                    if suffix == "weight" {
                        vec![rows, cols]
                    } else {
                        vec![rows]
                    }
                };
                match st.qkv {
                    Qkv::Split => {
                        for (tail, vals) in [("attn_q", q), ("attn_k", k), ("attn_v", v)] {
                            let (name, _) = name_of(Some(il), &format!("{tail}.{suffix}"));
                            out.push((name, dims_of(vals.len() / cols), vals));
                        }
                    }
                    Qkv::Contiguous(n) => {
                        let vals: Vec<f32> = [q, k, v].concat();
                        out.push((
                            format!("{}{n}.{suffix}", block_prefix(il)),
                            dims_of(vals.len() / cols),
                            vals,
                        ));
                    }
                    Qkv::Grouped(n) => {
                        let group = HEADS / KV_HEADS;
                        let mut vals = Vec::new();
                        for kv in 0..KV_HEADS {
                            for j in 0..group {
                                let h = kv * group + j;
                                vals.extend(take_rows(&q, cols, h * HEAD_DIM..(h + 1) * HEAD_DIM));
                            }
                            vals.extend(take_rows(&k, cols, kv * HEAD_DIM..(kv + 1) * HEAD_DIM));
                            vals.extend(take_rows(&v, cols, kv * HEAD_DIM..(kv + 1) * HEAD_DIM));
                        }
                        out.push((
                            format!("{}{n}.{suffix}", block_prefix(il)),
                            dims_of(vals.len() / cols),
                            vals,
                        ));
                    }
                }
            }
        }
        if let (Some(name), Some((_, qkv)), Some((_, z))) = (
            st.delta_packed.filter(|_| matches!(mix, Mix::DeltaPacked)),
            get("attn_qkv.weight"),
            get("attn_gate.weight"),
        ) {
            // Per key head: its query rows, key rows, value rows and gate rows.
            let (dk, per) = (SSM_STATE, SSM_HEADS / SSM_GROUPS * (SSM_INNER / SSM_HEADS));
            let key_total = SSM_GROUPS * dk;
            let mut vals = Vec::new();
            for g in 0..SSM_GROUPS {
                vals.extend(take_rows(qkv, HIDDEN, g * dk..(g + 1) * dk));
                vals.extend(take_rows(
                    qkv,
                    HIDDEN,
                    key_total + g * dk..key_total + (g + 1) * dk,
                ));
                vals.extend(take_rows(
                    qkv,
                    HIDDEN,
                    2 * key_total + g * per..2 * key_total + (g + 1) * per,
                ));
                vals.extend(take_rows(z, HIDDEN, g * per..(g + 1) * per));
            }
            out.push((
                format!("{}{name}", block_prefix(il)),
                vec![vals.len() / HIDDEN, HIDDEN],
                vals,
            ));
            done.insert(b("attn_qkv.weight"));
            done.insert(b("attn_gate.weight"));
        }
        if let Some(experts) = st.experts.filter(|_| get("ffn_up_exps.weight").is_some()) {
            let prefix = block_prefix(il);
            let (_, gate) = get("ffn_gate_exps.weight").map_or((vec![], vec![]), Clone::clone);
            let (_, up) = get("ffn_up_exps.weight").cloned().unwrap();
            let (_, down) = get("ffn_down_exps.weight").cloned().unwrap();
            let (f, h) = (EXPERT_FFN, HIDDEN);
            match experts {
                Experts::PerExpert {
                    gate: gn,
                    up: un,
                    down: dn,
                } => {
                    for (t, vals, dims) in
                        [(gn, &gate, [f, h]), (un, &up, [f, h]), (dn, &down, [h, f])]
                    {
                        if vals.is_empty() || t.is_empty() {
                            continue;
                        }
                        let per = vals.len() / EXPERTS;
                        for e in 0..EXPERTS {
                            out.push((
                                format!("{prefix}{}", t.replace("{e}", &e.to_string())),
                                dims.to_vec(),
                                vals[e * per..(e + 1) * per].to_vec(),
                            ));
                        }
                    }
                }
                Experts::FusedHalves { gate_up, down: dn } => {
                    let mut vals = Vec::new();
                    for e in 0..EXPERTS {
                        vals.extend(&gate[e * f * h..(e + 1) * f * h]);
                        vals.extend(&up[e * f * h..(e + 1) * f * h]);
                    }
                    out.push((format!("{prefix}{gate_up}"), vec![EXPERTS, 2 * f, h], vals));
                    out.push((format!("{prefix}{dn}"), vec![EXPERTS, h, f], down.clone()));
                }
                Experts::Flat {
                    gate: gn,
                    up: un,
                    down: dn,
                } => {
                    out.push((format!("{prefix}{gn}"), vec![EXPERTS * f, h], gate.clone()));
                    out.push((format!("{prefix}{un}"), vec![EXPERTS * f, h], up.clone()));
                    let mut vals = Vec::new();
                    for e in 0..EXPERTS {
                        for r in 0..f {
                            for c in 0..h {
                                vals.push(down[e * h * f + c * f + r]);
                            }
                        }
                    }
                    out.push((format!("{prefix}{dn}"), vec![EXPERTS * f, h], vals));
                }
                Experts::Alternating { gate_up, down: dn } => {
                    let mut vals = Vec::new();
                    for e in 0..EXPERTS {
                        for c in 0..h {
                            for r in 0..f {
                                vals.push(gate[e * f * h + r * h + c]);
                                vals.push(up[e * f * h + r * h + c]);
                            }
                        }
                    }
                    out.push((format!("{prefix}{gate_up}"), vec![EXPERTS, h, 2 * f], vals));
                    let mut vals = Vec::new();
                    for e in 0..EXPERTS {
                        for c in 0..f {
                            for r in 0..h {
                                vals.push(down[e * h * f + r * f + c]);
                            }
                        }
                    }
                    out.push((format!("{prefix}{dn}"), vec![EXPERTS, f, h], vals));
                    if let (Some(g), Some(u)) = (get("ffn_gate_exps.bias"), get("ffn_up_exps.bias"))
                    {
                        let vals: Vec<f32> =
                            (0..EXPERTS * f).flat_map(|i| [g.1[i], u.1[i]]).collect();
                        out.push((
                            format!("{prefix}{gate_up}_bias"),
                            vec![EXPERTS, 2 * f],
                            vals,
                        ));
                        let d = get("ffn_down_exps.bias").unwrap();
                        out.push((format!("{prefix}{dn}_bias"), d.0.clone(), d.1.clone()));
                    }
                }
            }
            for t in [
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
                "ffn_gate_exps.bias",
                "ffn_up_exps.bias",
                "ffn_down_exps.bias",
            ] {
                done.insert(b(t));
            }
        }
    }
    for (name, (dims, vals)) in w {
        if done.contains(name) {
            continue;
        }
        let (layer, tail) = match name.strip_prefix("blk.") {
            Some(rest) => {
                let (l, tail) = rest.split_once('.').unwrap();
                (Some(l.parse::<usize>().unwrap()), tail.to_string())
            }
            None => (None, name.clone()),
        };
        let (published, xf) = name_of(layer, &tail);
        let norm = tail == "attn_q_norm.weight" || tail == "attn_k_norm.weight";
        let vals = if norm {
            repair(vals, dims, st.rot, st.pairing)
        } else {
            vals.clone()
        };
        let (dims, vals) = store(xf, dims, &vals);
        out.push((published, dims, vals));
    }
    out
}

/// A publisher: the recipe it writes, the class and configuration of its directory, and how
/// it names and lays out the tensors.
struct Publisher {
    recipe: Recipe,
    class: &'static str,
    /// `config.json`, or `params.json` of a native layout.
    config: String,
    style: Style,
    /// The vendor's own layout: `params.json` and `consolidated.safetensors`.
    native: bool,
    /// Every tensor is stored BF16.
    bf16: bool,
    /// The experts are stored as MXFP4 blocks and scales, in the GGUF file as MXFP4 blocks.
    mx: bool,
    /// Every tensor is stored F16.
    f16: bool,
}

/// The keys the llama family's configurations carry, with the norm epsilon under `eps_key`
/// and `extra` (a fragment starting with a comma) after them.
fn llama_fields(recipe: &Recipe, eps_key: &str, extra: &str) -> String {
    format!(
        r#""hidden_size": {HIDDEN}, "intermediate_size": {FFN}, "num_hidden_layers": {},
        "num_attention_heads": {HEADS}, "num_key_value_heads": {KV_HEADS}, "head_dim": {HEAD_DIM},
        "vocab_size": {VOCAB}, "rope_theta": 10000.0, "{eps_key}": 1e-5,
        "tie_word_embeddings": {}{extra}"#,
        recipe.mixers.len(),
        recipe.tied_head,
    )
}

/// A directory's configuration with `fields` at the root.
fn root_config(class: &str, fields: &str) -> String {
    format!(r#"{{"architectures": ["{class}"], {fields}}}"#)
}

/// A composite model's configuration: the text model's `fields` under `text_config`.
fn wrapped_config(class: &str, fields: &str) -> String {
    format!(
        r#"{{"architectures": ["{class}"], "vocab_size": {VOCAB}, "text_config": {{{fields}}}}}"#
    )
}

fn llama_publisher(recipe: Recipe, class: &'static str, pairing: Pairing) -> Publisher {
    let config = root_config(class, &llama_fields(&recipe, "rms_norm_eps", ""));
    Publisher {
        recipe,
        class,
        config,
        style: Style {
            pairing,
            rot: HEAD_DIM,
            layer: "model.layers.{l}.",
            names: &LLAMA_NAMES,
            qkv: Qkv::Split,
            experts: None,
            delta_packed: None,
        },
        native: false,
        bf16: false,
        mx: false,
        f16: false,
    }
}

/// The names of the tensors outside the blocks, and the sinks, in the llama family's layout.
const LLAMA_GLOBAL: [(&str, &str, Xf); 5] = [
    ("token_embd.weight", "model.embed_tokens.weight", Id),
    ("output_norm.weight", "model.norm.weight", Id),
    ("output_norm.bias", "model.norm.bias", Id),
    ("output.weight", "lm_head.weight", Id),
    ("attn_sinks", "self_attn.sinks", Id),
];

/// The llama family's names.
const LLAMA_NAMES: [&[(&str, &str, Xf)]; 3] = [&LLAMA_GLOBAL, &LLAMA_BLOCK, &LLAMA_BIASES];

fn recipe_of(arch: &str) -> Recipe {
    all_recipes()
        .into_iter()
        .find(|r| r.arch == arch)
        .unwrap_or_else(|| panic!("no recipe for {arch}"))
}

/// A llama-family publisher with `names` in place of the llama family's, and the
/// configuration's norm epsilon under `eps_key` and `extra` after the common keys.
fn styled(
    arch: &str,
    class: &'static str,
    pairing: Pairing,
    eps_key: &str,
    extra: &str,
    names: &'static [&'static [(&'static str, &'static str, Xf)]],
) -> Publisher {
    let recipe = recipe_of(arch);
    let config = root_config(class, &llama_fields(&recipe, eps_key, extra));
    Publisher {
        recipe,
        class,
        config,
        style: Style {
            pairing,
            rot: HEAD_DIM,
            layer: "model.layers.{l}.",
            names,
            qkv: Qkv::Split,
            experts: None,
            delta_packed: None,
        },
        native: false,
        bf16: false,
        mx: false,
        f16: false,
    }
}

const MOE_KEYS: &str = r#", "num_experts": 4, "num_local_experts": 4, "num_experts_per_tok": 2,
    "moe_intermediate_size": 64"#;

const ROUTER_MLP_GATE: [(&str, &str, Xf); 1] = [("ffn_gate_inp.weight", "mlp.gate.weight", Id)];

const PER_EXPERT_PROJ: Experts = Experts::PerExpert {
    gate: "mlp.experts.{e}.gate_proj.weight",
    up: "mlp.experts.{e}.up_proj.weight",
    down: "mlp.experts.{e}.down_proj.weight",
};

fn smollm3() -> Publisher {
    styled(
        "smollm3",
        "SmolLM3ForCausalLM",
        Pairing::AdjacentToHalf,
        "rms_norm_eps",
        "",
        &LLAMA_NAMES,
    )
}

fn ernie4_5() -> Publisher {
    styled(
        "ernie4_5",
        "Ernie4_5ForCausalLM",
        Pairing::Same,
        "rms_norm_eps",
        "",
        &LLAMA_NAMES,
    )
}

fn granite() -> Publisher {
    styled(
        "granite",
        "GraniteForCausalLM",
        Pairing::AdjacentToHalf,
        "rms_norm_eps",
        r#", "embedding_multiplier": 12.0, "residual_multiplier": 0.22,
        "attention_multiplier": 0.015625, "logits_scaling": 8.0"#,
        &LLAMA_NAMES,
    )
}

fn granitemoe() -> Publisher {
    const NAMES: [&[(&str, &str, Xf)]; 4] = [
        &[(
            "ffn_gate_inp.weight",
            "block_sparse_moe.router.layer.weight",
            Id,
        )],
        &LLAMA_GLOBAL,
        &LLAMA_BLOCK,
        &LLAMA_BIASES,
    ];
    let mut p = styled(
        "granitemoe",
        "GraniteMoeForCausalLM",
        Pairing::AdjacentToHalf,
        "rms_norm_eps",
        MOE_KEYS,
        &NAMES,
    );
    p.style.experts = Some(Experts::FusedHalves {
        gate_up: "block_sparse_moe.input_linear.weight",
        down: "block_sparse_moe.output_linear.weight",
    });
    p
}

fn yi() -> Publisher {
    const NAMES: [&[(&str, &str, Xf)]; 4] = [
        &[
            ("attn_norm.weight", "ln1.weight", Id),
            ("ffn_norm.weight", "ln2.weight", Id),
        ],
        &LLAMA_GLOBAL,
        &LLAMA_BLOCK,
        &LLAMA_BIASES,
    ];
    styled(
        "yi",
        "YiForCausalLM",
        Pairing::AdjacentToHalf,
        "rms_norm_eps",
        "",
        &NAMES,
    )
}

fn internlm2() -> Publisher {
    const NAMES: [&[(&str, &str, Xf)]; 1] = [&[
        ("token_embd.weight", "model.tok_embeddings.weight", Id),
        ("output_norm.weight", "model.norm.weight", Id),
        ("output.weight", "output.weight", Id),
        ("attn_norm.weight", "attention_norm.weight", Id),
        ("ffn_norm.weight", "ffn_norm.weight", Id),
        ("attn_output.weight", "attention.wo.weight", Id),
        ("ffn_gate.weight", "feed_forward.w1.weight", Id),
        ("ffn_up.weight", "feed_forward.w3.weight", Id),
        ("ffn_down.weight", "feed_forward.w2.weight", Id),
    ]];
    let mut p = styled(
        "internlm2",
        "InternLM2ForCausalLM",
        Pairing::AdjacentToHalf,
        "rms_norm_eps",
        "",
        &NAMES,
    );
    p.style.qkv = Qkv::Grouped("attention.wqkv");
    p
}

fn starcoder() -> Publisher {
    const NAMES: [&[(&str, &str, Xf)]; 1] = [&[
        ("token_embd.weight", "transformer.wte.weight", Id),
        ("position_embd.weight", "transformer.wpe.weight", Id),
        ("output_norm.weight", "transformer.ln_f.weight", Id),
        ("output_norm.bias", "transformer.ln_f.bias", Id),
        ("output.weight", "lm_head.weight", Id),
        ("attn_norm.weight", "ln_1.weight", Id),
        ("attn_norm.bias", "ln_1.bias", Id),
        ("ffn_norm.weight", "ln_2.weight", Id),
        ("ffn_norm.bias", "ln_2.bias", Id),
        ("attn_output.weight", "attn.c_proj.weight", Id),
        ("attn_output.bias", "attn.c_proj.bias", Id),
        ("ffn_up.weight", "mlp.c_fc.weight", Id),
        ("ffn_up.bias", "mlp.c_fc.bias", Id),
        ("ffn_down.weight", "mlp.c_proj.weight", Id),
        ("ffn_down.bias", "mlp.c_proj.bias", Id),
    ]];
    let recipe = recipe_of("starcoder");
    let config = root_config(
        "GPTBigCodeForCausalLM",
        &format!(
            r#""n_embd": {HIDDEN}, "n_head": {HEADS}, "n_layer": {}, "n_inner": {FFN},
            "head_dim": {HEAD_DIM}, "vocab_size": {VOCAB}, "n_positions": 256,
            "layer_norm_epsilon": 1e-5, "multi_query": true, "num_key_value_heads": {KV_HEADS}"#,
            recipe.mixers.len()
        ),
    );
    Publisher {
        recipe,
        class: "GPTBigCodeForCausalLM",
        config,
        style: Style {
            pairing: Pairing::Same,
            rot: HEAD_DIM,
            layer: "transformer.h.{l}.",
            names: &NAMES,
            qkv: Qkv::Contiguous("attn.c_attn"),
            experts: None,
            delta_packed: None,
        },
        native: false,
        bf16: false,
        mx: false,
        f16: false,
    }
}

fn starcoder2() -> Publisher {
    const NAMES: [&[(&str, &str, Xf)]; 4] = [
        &[
            ("ffn_up.weight", "mlp.c_fc.weight", Id),
            ("ffn_up.bias", "mlp.c_fc.bias", Id),
            ("ffn_down.weight", "mlp.c_proj.weight", Id),
            ("ffn_down.bias", "mlp.c_proj.bias", Id),
        ],
        &LLAMA_GLOBAL,
        &LLAMA_BLOCK,
        &LLAMA_BIASES,
    ];
    styled(
        "starcoder2",
        "Starcoder2ForCausalLM",
        Pairing::Same,
        "norm_epsilon",
        "",
        &NAMES,
    )
}

fn qwen3moe() -> Publisher {
    const NAMES: [&[(&str, &str, Xf)]; 4] =
        [&ROUTER_MLP_GATE, &LLAMA_GLOBAL, &LLAMA_BLOCK, &LLAMA_BIASES];
    let mut p = styled(
        "qwen3moe",
        "Qwen3MoeForCausalLM",
        Pairing::Same,
        "rms_norm_eps",
        &format!("{MOE_KEYS}, \"norm_topk_prob\": true"),
        &NAMES,
    );
    p.style.experts = Some(PER_EXPERT_PROJ);
    p
}

fn phi2() -> Publisher {
    const NAMES: [&[(&str, &str, Xf)]; 4] = [
        &[
            ("output_norm.weight", "model.final_layernorm.weight", Id),
            ("output_norm.bias", "model.final_layernorm.bias", Id),
            ("attn_norm.weight", "input_layernorm.weight", Id),
            ("attn_norm.bias", "input_layernorm.bias", Id),
            ("attn_output.weight", "self_attn.dense.weight", Id),
            ("attn_output.bias", "self_attn.dense.bias", Id),
            ("ffn_up.weight", "mlp.fc1.weight", Id),
            ("ffn_up.bias", "mlp.fc1.bias", Id),
            ("ffn_down.weight", "mlp.fc2.weight", Id),
            ("ffn_down.bias", "mlp.fc2.bias", Id),
        ],
        &LLAMA_GLOBAL,
        &LLAMA_BLOCK,
        &LLAMA_BIASES,
    ];
    styled(
        "phi2",
        "PhiForCausalLM",
        Pairing::Same,
        "layer_norm_eps",
        r#", "partial_rotary_factor": 0.5"#,
        &NAMES,
    )
}

fn phi3() -> Publisher {
    const NAMES: [&[(&str, &str, Xf)]; 4] = [
        &[("ffn_up.weight", "mlp.gate_up_proj.weight", Id)],
        &LLAMA_GLOBAL,
        &LLAMA_BLOCK,
        &LLAMA_BIASES,
    ];
    let mut p = styled(
        "phi3",
        "Phi3ForCausalLM",
        Pairing::Same,
        "rms_norm_eps",
        "",
        &NAMES,
    );
    p.style.qkv = Qkv::Contiguous("self_attn.qkv_proj");
    p
}

fn chatglm() -> Publisher {
    const NAMES: [&[(&str, &str, Xf)]; 1] = [&[
        (
            "token_embd.weight",
            "transformer.embedding.word_embeddings.weight",
            Id,
        ),
        (
            "output_norm.weight",
            "transformer.encoder.final_layernorm.weight",
            Id,
        ),
        ("output.weight", "transformer.output_layer.weight", Id),
        ("attn_norm.weight", "input_layernorm.weight", Id),
        ("ffn_norm.weight", "post_attention_layernorm.weight", Id),
        ("attn_output.weight", "self_attention.dense.weight", Id),
        ("ffn_up.weight", "mlp.dense_h_to_4h.weight", Id),
        ("ffn_down.weight", "mlp.dense_4h_to_h.weight", Id),
    ]];
    let recipe = recipe_of("chatglm");
    let config = root_config(
        "ChatGLMModel",
        &format!(
            r#""hidden_size": {HIDDEN}, "num_layers": {}, "num_attention_heads": {HEADS},
            "multi_query_attention": true, "multi_query_group_num": {KV_HEADS},
            "kv_channels": {HEAD_DIM}, "ffn_hidden_size": {FFN}, "padded_vocab_size": {VOCAB},
            "layernorm_epsilon": 1e-5, "rope_ratio": 1, "seq_length": 256, "add_qkv_bias": true"#,
            recipe.mixers.len()
        ),
    );
    Publisher {
        recipe,
        class: "ChatGLMModel",
        config,
        style: Style {
            pairing: Pairing::HalfToAdjacent,
            rot: 16,
            layer: "transformer.encoder.layers.{l}.",
            names: &NAMES,
            qkv: Qkv::Contiguous("self_attention.query_key_value"),
            experts: None,
            delta_packed: None,
        },
        native: false,
        bf16: false,
        mx: false,
        f16: false,
    }
}

fn glm4() -> Publisher {
    const NAMES: [&[(&str, &str, Xf)]; 4] = [
        &[
            (
                "post_attention_norm.weight",
                "post_self_attn_layernorm.weight",
                Id,
            ),
            ("post_ffw_norm.weight", "post_mlp_layernorm.weight", Id),
            ("ffn_up.weight", "mlp.gate_up_proj.weight", Id),
        ],
        &LLAMA_GLOBAL,
        &LLAMA_BLOCK,
        &LLAMA_BIASES,
    ];
    let mut p = styled(
        "glm4",
        "Glm4ForCausalLM",
        Pairing::HalfToAdjacent,
        "rms_norm_eps",
        r#", "partial_rotary_factor": 0.5"#,
        &NAMES,
    );
    p.style.rot = 16;
    p
}

fn stablelm() -> Publisher {
    styled(
        "stablelm",
        "StableLmForCausalLM",
        Pairing::Same,
        "layer_norm_eps",
        "",
        &LLAMA_NAMES,
    )
}

fn olmo2() -> Publisher {
    const NAMES: [&[(&str, &str, Xf)]; 4] = [
        &[
            (
                "post_attention_norm.weight",
                "post_attention_layernorm.weight",
                Id,
            ),
            (
                "post_ffw_norm.weight",
                "post_feedforward_layernorm.weight",
                Id,
            ),
        ],
        &LLAMA_GLOBAL,
        &LLAMA_BLOCK,
        &LLAMA_BIASES,
    ];
    styled(
        "olmo2",
        "Olmo2ForCausalLM",
        Pairing::Same,
        "rms_norm_eps",
        "",
        &NAMES,
    )
}

fn olmoe() -> Publisher {
    const NAMES: [&[(&str, &str, Xf)]; 4] =
        [&ROUTER_MLP_GATE, &LLAMA_GLOBAL, &LLAMA_BLOCK, &LLAMA_BIASES];
    let mut p = styled(
        "olmoe",
        "OlmoeForCausalLM",
        Pairing::Same,
        "rms_norm_eps",
        &format!("{MOE_KEYS}, \"norm_topk_prob\": false"),
        &NAMES,
    );
    p.style.experts = Some(PER_EXPERT_PROJ);
    p
}

fn falcon() -> Publisher {
    const NAMES: [&[(&str, &str, Xf)]; 1] = [&[
        (
            "token_embd.weight",
            "transformer.word_embeddings.weight",
            Id,
        ),
        ("output_norm.weight", "transformer.ln_f.weight", Id),
        ("output_norm.bias", "transformer.ln_f.bias", Id),
        ("output.weight", "lm_head.weight", Id),
        ("attn_norm.weight", "input_layernorm.weight", Id),
        ("attn_norm.bias", "input_layernorm.bias", Id),
        ("attn_output.weight", "self_attention.dense.weight", Id),
        ("ffn_up.weight", "mlp.dense_h_to_4h.weight", Id),
        ("ffn_down.weight", "mlp.dense_4h_to_h.weight", Id),
    ]];
    let recipe = recipe_of("falcon");
    let config = root_config(
        "FalconForCausalLM",
        &format!(
            r#""hidden_size": {HIDDEN}, "num_hidden_layers": {}, "num_attention_heads": {HEADS},
            "num_kv_heads": {KV_HEADS}, "head_dim": {HEAD_DIM}, "ffn_hidden_size": {FFN},
            "multi_query": true,
            "parallel_attn": true, "new_decoder_architecture": false, "vocab_size": {VOCAB},
            "layer_norm_epsilon": 1e-5, "rope_theta": 10000.0, "bias": false"#,
            recipe.mixers.len()
        ),
    );
    Publisher {
        recipe,
        class: "FalconForCausalLM",
        config,
        style: Style {
            pairing: Pairing::Same,
            rot: HEAD_DIM,
            layer: "transformer.h.{l}.",
            names: &NAMES,
            qkv: Qkv::Contiguous("self_attention.query_key_value"),
            experts: None,
            delta_packed: None,
        },
        native: false,
        bf16: false,
        mx: false,
        f16: false,
    }
}

fn dbrx() -> Publisher {
    const NAMES: [&[(&str, &str, Xf)]; 1] = [&[
        ("token_embd.weight", "transformer.wte.weight", Id),
        ("output_norm.weight", "transformer.norm_f.weight", Id),
        ("output.weight", "lm_head.weight", Id),
        ("attn_norm.weight", "norm_attn_norm.norm_1.weight", Id),
        ("ffn_norm.weight", "norm_attn_norm.norm_2.weight", Id),
        (
            "attn_output.weight",
            "norm_attn_norm.attn.out_proj.weight",
            Id,
        ),
        ("ffn_gate_inp.weight", "ffn.router.layer.weight", Id),
    ]];
    let recipe = recipe_of("dbrx");
    let config = root_config(
        "DbrxForCausalLM",
        &format!(
            r#""d_model": {HIDDEN}, "n_heads": {HEADS}, "n_layers": {}, "head_dim": {HEAD_DIM},
            "vocab_size": {VOCAB}, "max_seq_len": 256,
            "attn_config": {{"kv_n_heads": {KV_HEADS}, "rope_theta": 10000.0}},
            "ffn_config": {{"ffn_hidden_size": {EXPERT_FFN}, "moe_num_experts": {EXPERTS},
                "moe_top_k": {EXPERTS_USED}, "moe_normalize_expert_weights": 1.0}}"#,
            recipe.mixers.len()
        ),
    );
    Publisher {
        recipe,
        class: "DbrxForCausalLM",
        config,
        style: Style {
            pairing: Pairing::Same,
            rot: HEAD_DIM,
            layer: "transformer.blocks.{l}.",
            names: &NAMES,
            qkv: Qkv::Contiguous("norm_attn_norm.attn.Wqkv"),
            experts: Some(Experts::Flat {
                gate: "ffn.experts.mlp.w1",
                up: "ffn.experts.mlp.v1",
                down: "ffn.experts.mlp.w2",
            }),
            delta_packed: None,
        },
        native: false,
        bf16: false,
        mx: false,
        f16: false,
    }
}

fn orion() -> Publisher {
    styled(
        "orion",
        "OrionForCausalLM",
        Pairing::Same,
        "rms_norm_eps",
        "",
        &LLAMA_NAMES,
    )
}

fn mpt() -> Publisher {
    const NAMES: [&[(&str, &str, Xf)]; 1] = [&[
        ("token_embd.weight", "transformer.wte.weight", Id),
        ("output_norm.weight", "transformer.norm_f.weight", Id),
        ("output_norm.bias", "transformer.norm_f.bias", Id),
        ("output.weight", "lm_head.weight", Id),
        ("attn_norm.weight", "norm_1.weight", Id),
        ("attn_norm.bias", "norm_1.bias", Id),
        ("ffn_norm.weight", "norm_2.weight", Id),
        ("ffn_norm.bias", "norm_2.bias", Id),
        ("attn_output.weight", "attn.out_proj.weight", Id),
        ("ffn_up.weight", "ffn.up_proj.weight", Id),
        ("ffn_down.weight", "ffn.down_proj.weight", Id),
    ]];
    let recipe = recipe_of("mpt");
    let config = root_config(
        "MPTForCausalLM",
        &format!(
            r#""d_model": {HIDDEN}, "n_heads": {HEADS}, "n_layers": {}, "head_dim": {HEAD_DIM},
            "expansion_ratio": 2, "vocab_size": {VOCAB}, "max_seq_len": 256,
            "layer_norm_epsilon": 1e-5, "attn_config": {{"kv_n_heads": {KV_HEADS}}}"#,
            recipe.mixers.len()
        ),
    );
    Publisher {
        recipe,
        class: "MPTForCausalLM",
        config,
        style: Style {
            pairing: Pairing::Same,
            rot: HEAD_DIM,
            layer: "transformer.blocks.{l}.",
            names: &NAMES,
            qkv: Qkv::Contiguous("attn.Wqkv"),
            experts: None,
            delta_packed: None,
        },
        native: false,
        bf16: false,
        mx: false,
        f16: false,
    }
}

/// The norm scales Gemma stores less one, in the llama family's names.
const GEMMA_SHIFTED: [(&str, &str, Xf); 5] = [
    ("output_norm.weight", "model.norm.weight", Minus1),
    ("attn_norm.weight", "input_layernorm.weight", Minus1),
    ("ffn_norm.weight", "post_attention_layernorm.weight", Minus1),
    ("attn_q_norm.weight", "self_attn.q_norm.weight", Minus1),
    ("attn_k_norm.weight", "self_attn.k_norm.weight", Minus1),
];

fn gemma_publisher() -> Publisher {
    const NAMES: [&[(&str, &str, Xf)]; 4] =
        [&GEMMA_SHIFTED, &LLAMA_GLOBAL, &LLAMA_BLOCK, &LLAMA_BIASES];
    styled(
        "gemma",
        "GemmaForCausalLM",
        Pairing::Same,
        "rms_norm_eps",
        "",
        &NAMES,
    )
}

const GEMMA_FOUR_NORMS: [(&str, &str, Xf); 3] = [
    (
        "post_attention_norm.weight",
        "post_attention_layernorm.weight",
        Minus1,
    ),
    (
        "ffn_norm.weight",
        "pre_feedforward_layernorm.weight",
        Minus1,
    ),
    (
        "post_ffw_norm.weight",
        "post_feedforward_layernorm.weight",
        Minus1,
    ),
];

fn gemma2() -> Publisher {
    const NAMES: [&[(&str, &str, Xf)]; 5] = [
        &GEMMA_FOUR_NORMS,
        &GEMMA_SHIFTED,
        &LLAMA_GLOBAL,
        &LLAMA_BLOCK,
        &LLAMA_BIASES,
    ];
    styled(
        "gemma2",
        "Gemma2ForCausalLM",
        Pairing::Same,
        "rms_norm_eps",
        "",
        &NAMES,
    )
}

/// The names of a Gemma 3 release of the composite class: the text model under `language_model`.
const GEMMA3_NAMES: [&[(&str, &str, Xf)]; 6] = [
    &[
        (
            "token_embd.weight",
            "language_model.model.embed_tokens.weight",
            Id,
        ),
        (
            "output_norm.weight",
            "language_model.model.norm.weight",
            Minus1,
        ),
        ("output.weight", "language_model.lm_head.weight", Id),
    ],
    &GEMMA_FOUR_NORMS,
    &GEMMA_SHIFTED,
    &LLAMA_BLOCK,
    &LLAMA_BIASES,
    &LLAMA_GLOBAL,
];

fn gemma3() -> Publisher {
    let recipe = recipe_of("gemma3");
    let fields = llama_fields(
        &recipe,
        "rms_norm_eps",
        r#", "sliding_window": 8, "rope_local_base_freq": 10000.0,
        "layer_types": ["sliding_attention", "full_attention"]"#,
    );
    Publisher {
        config: wrapped_config("Gemma3ForConditionalGeneration", &fields),
        recipe,
        class: "Gemma3ForConditionalGeneration",
        style: Style {
            pairing: Pairing::Same,
            rot: HEAD_DIM,
            layer: "language_model.model.layers.{l}.",
            names: &GEMMA3_NAMES,
            qkv: Qkv::Split,
            experts: None,
            delta_packed: None,
        },
        native: false,
        bf16: false,
        mx: false,
        f16: false,
    }
}

fn gptoss() -> Publisher {
    const NAMES: [&[(&str, &str, Xf)]; 4] = [
        &[
            ("ffn_gate_inp.weight", "mlp.router.weight", Id),
            ("ffn_gate_inp.bias", "mlp.router.bias", Id),
            ("attn_out.weight", "self_attn.o_proj.weight", Id),
            ("attn_out.bias", "self_attn.o_proj.bias", Id),
        ],
        &LLAMA_GLOBAL,
        &LLAMA_BLOCK,
        &LLAMA_BIASES,
    ];
    let mut p = styled(
        "gptoss",
        "GptOssForCausalLM",
        Pairing::Same,
        "rms_norm_eps",
        r#", "num_local_experts": 4, "num_experts_per_tok": 2, "sliding_window": 8,
        "layer_types": ["sliding_attention", "full_attention"]"#,
        &NAMES,
    );
    p.style.experts = Some(Experts::Alternating {
        gate_up: "mlp.experts.gate_up_proj",
        down: "mlp.experts.down_proj",
    });
    p
}

/// The MXFP4 release: the experts as blocks and scales, and the `quantization_config` that
/// names them, which the publisher ships with them.
fn gptoss_mxfp4() -> Publisher {
    let p = gptoss();
    let config = p.config.replacen(
        '{',
        r#"{"quantization_config": {"quant_method": "mxfp4",
        "modules_to_not_convert": ["model.layers.*.self_attn", "lm_head"]}, "#,
        1,
    );
    Publisher {
        mx: true,
        config,
        ..p
    }
}

const LFM2_NAMES: [(&str, &str, Xf); 14] = [
    ("token_embd_norm.weight", "model.embedding_norm.weight", Id),
    ("attn_norm.weight", "operator_norm.weight", Id),
    ("ffn_norm.weight", "ffn_norm.weight", Id),
    ("shortconv.in_proj.weight", "conv.in_proj.weight", Id),
    ("shortconv.conv.weight", "conv.conv.weight", Unsqueeze),
    ("shortconv.out_proj.weight", "conv.out_proj.weight", Id),
    ("attn_output.weight", "self_attn.out_proj.weight", Id),
    ("attn_q_norm.weight", "self_attn.q_layernorm.weight", Id),
    ("attn_k_norm.weight", "self_attn.k_layernorm.weight", Id),
    ("ffn_gate.weight", "feed_forward.w1.weight", Id),
    ("ffn_up.weight", "feed_forward.w3.weight", Id),
    ("ffn_down.weight", "feed_forward.w2.weight", Id),
    ("ffn_gate_inp.weight", "feed_forward.gate.weight", Id),
    ("exp_probs_b.bias", "feed_forward.expert_bias", Id),
];

fn lfm2_fields(recipe: &Recipe, extra: &str) -> String {
    llama_fields(
        recipe,
        "norm_eps",
        &format!(
            r#", "conv_L_cache": 4,
        "layer_types": [{}]{extra}"#,
            recipe
                .mixers
                .iter()
                .map(|m| if *m == Mix::Attention {
                    "\"full_attention\""
                } else {
                    "\"conv\""
                })
                .collect::<Vec<_>>()
                .join(", ")
        ),
    )
}

fn lfm2() -> Publisher {
    const NAMES: [&[(&str, &str, Xf)]; 4] =
        [&LFM2_NAMES, &LLAMA_GLOBAL, &LLAMA_BLOCK, &LLAMA_BIASES];
    let mut p = styled(
        "lfm2",
        "Lfm2ForCausalLM",
        Pairing::Same,
        "norm_eps",
        "",
        &NAMES,
    );
    p.config = root_config("Lfm2ForCausalLM", &lfm2_fields(&p.recipe, ""));
    p
}

fn lfm2moe() -> Publisher {
    const NAMES: [&[(&str, &str, Xf)]; 4] =
        [&LFM2_NAMES, &LLAMA_GLOBAL, &LLAMA_BLOCK, &LLAMA_BIASES];
    let mut p = styled(
        "lfm2moe",
        "Lfm2MoeForCausalLM",
        Pairing::Same,
        "norm_eps",
        "",
        &NAMES,
    );
    p.config = root_config(
        "Lfm2MoeForCausalLM",
        &lfm2_fields(
            &p.recipe,
            r#", "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 64,
            "num_dense_layers": 1, "norm_topk_prob": true, "use_expert_bias": true"#,
        ),
    );
    p.style.experts = Some(Experts::PerExpert {
        gate: "feed_forward.experts.{e}.w1.weight",
        up: "feed_forward.experts.{e}.w3.weight",
        down: "feed_forward.experts.{e}.w2.weight",
    });
    p
}

/// The delta rule's tensors in the names of its two releases, the norms stored less one.
const DELTA_COMMON: [(&str, &str, Xf); 14] = [
    (
        "token_embd.weight",
        "model.language_model.embed_tokens.weight",
        Id,
    ),
    (
        "output_norm.weight",
        "model.language_model.norm.weight",
        Minus1,
    ),
    ("output.weight", "lm_head.weight", Id),
    ("attn_norm.weight", "input_layernorm.weight", Minus1),
    (
        "post_attention_norm.weight",
        "post_attention_layernorm.weight",
        Minus1,
    ),
    ("attn_q_norm.weight", "self_attn.q_norm.weight", Minus1),
    ("attn_k_norm.weight", "self_attn.k_norm.weight", Minus1),
    ("ssm_conv1d.weight", "linear_attn.conv1d.weight", Unsqueeze),
    ("ssm_a", "linear_attn.A_log", LogNeg),
    ("ssm_dt", "linear_attn.dt_bias", Id),
    ("ssm_norm.weight", "linear_attn.norm.weight", Id),
    ("ssm_out.weight", "linear_attn.out_proj.weight", Id),
    ("ffn_gate_inp.weight", "mlp.gate.weight", Id),
    (
        "ffn_gate_inp_shexp.weight",
        "mlp.shared_expert_gate.weight",
        Row,
    ),
];

const DELTA_SPLIT: [(&str, &str, Xf); 4] = [
    ("attn_qkv.weight", "linear_attn.in_proj_qkv.weight", Id),
    ("attn_gate.weight", "linear_attn.in_proj_z.weight", Id),
    ("ssm_beta.weight", "linear_attn.in_proj_b.weight", Id),
    ("ssm_alpha.weight", "linear_attn.in_proj_a.weight", Id),
];

const SHARED_EXPERT: [(&str, &str, Xf); 3] = [
    (
        "ffn_gate_shexp.weight",
        "mlp.shared_expert.gate_proj.weight",
        Id,
    ),
    (
        "ffn_up_shexp.weight",
        "mlp.shared_expert.up_proj.weight",
        Id,
    ),
    (
        "ffn_down_shexp.weight",
        "mlp.shared_expert.down_proj.weight",
        Id,
    ),
];

fn delta_fields(recipe: &Recipe, extra: &str) -> String {
    let kinds: Vec<&str> = recipe
        .mixers
        .iter()
        .map(|m| match m {
            Mix::Attention => "\"full_attention\"",
            _ => "\"linear_attention\"",
        })
        .collect();
    llama_fields(
        recipe,
        "rms_norm_eps",
        &format!(
            r#", "layer_types": [{}], "linear_conv_kernel_dim": {CONV_KERNEL},
        "linear_key_head_dim": {SSM_STATE}, "linear_num_key_heads": {SSM_GROUPS},
        "linear_num_value_heads": {SSM_HEADS}, "linear_value_head_dim": {},
        "num_experts": {EXPERTS}, "num_experts_per_tok": {EXPERTS_USED},
        "moe_intermediate_size": {EXPERT_FFN}{extra}"#,
            kinds.join(", "),
            SSM_INNER / SSM_HEADS
        ),
    )
}

fn delta_publisher(arch: &str, class: &'static str, composite: bool) -> Publisher {
    const NAMES: [&[(&str, &str, Xf)]; 6] = [
        &DELTA_COMMON,
        &DELTA_SPLIT,
        &SHARED_EXPERT,
        &LLAMA_GLOBAL,
        &LLAMA_BLOCK,
        &LLAMA_BIASES,
    ];
    let recipe = recipe_of(arch);
    let fields = delta_fields(&recipe, "");
    let config = if composite {
        wrapped_config(class, &fields)
    } else {
        root_config(class, &fields)
    };
    Publisher {
        recipe,
        class,
        config,
        style: Style {
            pairing: Pairing::Same,
            rot: HEAD_DIM,
            layer: "model.language_model.layers.{l}.",
            names: &NAMES,
            qkv: Qkv::Split,
            experts: Some(Experts::FusedHalves {
                gate_up: "mlp.experts.gate_up_proj",
                down: "mlp.experts.down_proj",
            }),
            delta_packed: None,
        },
        native: false,
        bf16: false,
        mx: false,
        f16: false,
    }
}

fn qwen35() -> Publisher {
    delta_publisher("qwen35", "Qwen3_5ForConditionalGeneration", true)
}

fn qwen35moe() -> Publisher {
    delta_publisher("qwen35moe", "Qwen3_5MoeForConditionalGeneration", true)
}

/// The packed release: one `qkvz` projection, one `ba`, the experts a tensor each.
fn qwen3next() -> Publisher {
    const PACKED: [(&str, &str, Xf); 3] = [
        ("ssm_ba.weight", "linear_attn.in_proj_ba.weight", Id),
        ("token_embd.weight", "model.embed_tokens.weight", Id),
        ("output_norm.weight", "model.norm.weight", Minus1),
    ];
    const NAMES: [&[(&str, &str, Xf)]; 7] = [
        &PACKED,
        &DELTA_COMMON,
        &DELTA_SPLIT,
        &SHARED_EXPERT,
        &LLAMA_GLOBAL,
        &LLAMA_BLOCK,
        &LLAMA_BIASES,
    ];
    let mut p = delta_publisher("qwen3next", "Qwen3NextForCausalLM", false);
    p.style.names = &NAMES;
    p.style.layer = "model.layers.{l}.";
    p.style.experts = Some(PER_EXPERT_PROJ);
    p.style.delta_packed = Some("linear_attn.in_proj_qkvz.weight");
    p
}

fn nemotron() -> Publisher {
    const NAMES: [&[(&str, &str, Xf)]; 1] = [&[
        ("token_embd.weight", "backbone.embeddings.weight", Id),
        ("output_norm.weight", "backbone.norm_f.weight", Id),
        ("output.weight", "lm_head.weight", Id),
        ("attn_norm.weight", "norm.weight", Id),
        ("ssm_in.weight", "mixer.in_proj.weight", Id),
        ("ssm_conv1d.weight", "mixer.conv1d.weight", Unsqueeze),
        ("ssm_a", "mixer.A_log", LogNeg),
        ("ssm_d", "mixer.D", Flat),
        ("ssm_dt.bias", "mixer.dt_bias", Id),
        ("ssm_norm.weight", "mixer.norm.weight", Flat),
        ("ssm_out.weight", "mixer.out_proj.weight", Id),
        ("attn_q.weight", "mixer.q_proj.weight", Id),
        ("attn_k.weight", "mixer.k_proj.weight", Id),
        ("attn_v.weight", "mixer.v_proj.weight", Id),
        ("attn_output.weight", "mixer.o_proj.weight", Id),
        ("ffn_gate_inp.weight", "mixer.gate.weight", Id),
        ("exp_probs_b.bias", "mixer.gate.e_score_correction_bias", Id),
        (
            "ffn_up_shexp.weight",
            "mixer.shared_experts.up_proj.weight",
            Id,
        ),
        (
            "ffn_down_shexp.weight",
            "mixer.shared_experts.down_proj.weight",
            Id,
        ),
    ]];
    let recipe = recipe_of("nemotron_h_moe");
    let config = root_config(
        "NemotronHForCausalLM",
        &format!(
            r#""hidden_size": {HIDDEN}, "num_hidden_layers": {}, "num_attention_heads": {HEADS},
            "num_key_value_heads": {KV_HEADS}, "head_dim": {HEAD_DIM}, "vocab_size": {VOCAB},
            "layer_norm_epsilon": 1e-5, "mamba_num_heads": {MAMBA_HEADS},
            "mamba_head_dim": {}, "n_groups": {SSM_GROUPS}, "ssm_state_size": {SSM_STATE},
            "conv_kernel": {CONV_KERNEL}, "n_routed_experts": {EXPERTS},
            "num_experts_per_tok": {EXPERTS_USED}, "moe_intermediate_size": {EXPERT_FFN},
            "intermediate_size": {FFN}, "tie_word_embeddings": false"#,
            recipe.mixers.len(),
            SSM_INNER / MAMBA_HEADS
        ),
    );
    Publisher {
        recipe,
        class: "NemotronHForCausalLM",
        config,
        style: Style {
            pairing: Pairing::Same,
            rot: HEAD_DIM,
            layer: "backbone.layers.{l}.",
            names: &NAMES,
            qkv: Qkv::Split,
            experts: Some(Experts::PerExpert {
                gate: "",
                up: "mixer.experts.{e}.up_proj.weight",
                down: "mixer.experts.{e}.down_proj.weight",
            }),
            delta_packed: None,
        },
        native: false,
        bf16: false,
        mx: false,
        f16: false,
    }
}

const MIXTRAL_EXPERTS: Experts = Experts::PerExpert {
    gate: "block_sparse_moe.experts.{e}.w1.weight",
    up: "block_sparse_moe.experts.{e}.w3.weight",
    down: "block_sparse_moe.experts.{e}.w2.weight",
};

fn mixtral() -> Publisher {
    const NAMES: [&[(&str, &str, Xf)]; 4] = [
        &[("ffn_gate_inp.weight", "block_sparse_moe.gate.weight", Id)],
        &LLAMA_GLOBAL,
        &LLAMA_BLOCK,
        &LLAMA_BIASES,
    ];
    let recipe = Recipe {
        arch: "llama",
        ..dense("llama").moe()
    };
    let config = root_config(
        "MixtralForCausalLM",
        &llama_fields(&recipe, "rms_norm_eps", MOE_KEYS),
    );
    Publisher {
        recipe,
        class: "MixtralForCausalLM",
        config,
        style: Style {
            pairing: Pairing::AdjacentToHalf,
            rot: HEAD_DIM,
            layer: "model.layers.{l}.",
            names: &NAMES,
            qkv: Qkv::Split,
            experts: Some(MIXTRAL_EXPERTS),
            delta_packed: None,
        },
        native: false,
        bf16: false,
        mx: false,
        f16: false,
    }
}

/// Mixtral in the vendor's own layout: the rows keep the file's adjacent pairing.
fn mixtral_native() -> Publisher {
    const NAMES: [&[(&str, &str, Xf)]; 1] = [&[
        ("token_embd.weight", "tok_embeddings.weight", Id),
        ("output_norm.weight", "norm.weight", Id),
        ("output.weight", "output.weight", Id),
        ("attn_norm.weight", "attention_norm.weight", Id),
        ("ffn_norm.weight", "ffn_norm.weight", Id),
        ("attn_q.weight", "attention.wq.weight", Id),
        ("attn_k.weight", "attention.wk.weight", Id),
        ("attn_v.weight", "attention.wv.weight", Id),
        ("attn_output.weight", "attention.wo.weight", Id),
        ("ffn_gate_inp.weight", "feed_forward.gate.weight", Id),
    ]];
    let recipe = Recipe {
        arch: "llama",
        ..dense("llama").moe()
    };
    let params = format!(
        r#"{{"dim": {HIDDEN}, "n_layers": {}, "head_dim": {HEAD_DIM}, "hidden_dim": {EXPERT_FFN},
        "n_heads": {HEADS}, "n_kv_heads": {KV_HEADS}, "rope_theta": 10000.0, "norm_eps": 1e-5,
        "vocab_size": {VOCAB}, "moe": {{"num_experts": {EXPERTS}, "num_experts_per_tok": {EXPERTS_USED}}}}}"#,
        recipe.mixers.len()
    );
    Publisher {
        recipe,
        class: "",
        config: params,
        style: Style {
            pairing: Pairing::Same,
            rot: HEAD_DIM,
            layer: "layers.{l}.",
            names: &NAMES,
            qkv: Qkv::Split,
            experts: Some(Experts::PerExpert {
                gate: "feed_forward.experts.{e}.w1.weight",
                up: "feed_forward.experts.{e}.w3.weight",
                down: "feed_forward.experts.{e}.w2.weight",
            }),
            delta_packed: None,
        },
        native: true,
        bf16: false,
        mx: false,
        f16: false,
    }
}

/// The experts of the recipe's blocks as MXFP4: codes and scales chosen per row, written as the
/// publisher stores them (a block's values in pairs per byte, the rows of gate and up
/// alternating) and as a GGUF file holds them (one scale byte, then value `j` low and value
/// `j + 16` high in byte `j`).
fn mxfp4_experts(recipe: &Recipe, st: &Style) -> (Vec<Stored>, Replaced) {
    let Some(Experts::Alternating {
        gate_up,
        down: down_name,
    }) = st.experts
    else {
        panic!("MXFP4 experts are stored as an alternating stack");
    };
    let mut hf: Vec<Stored> = Vec::new();
    let mut gguf = Replaced::new();
    let codes = |seed: usize, rows: usize, cols: usize| -> Vec<u8> {
        (0..EXPERTS * rows * cols)
            .map(|i| ((i * 7 + seed * 5 + i / 13) % 16) as u8)
            .collect()
    };
    let scale = |row: usize| 122 + (row % 3) as u8;
    // `[experts, rows, cols]` codes as a file's blocks.
    let file_blocks = |c: &[u8], rows: usize, cols: usize| -> Vec<u8> {
        let mut out = Vec::new();
        for r in 0..EXPERTS * rows {
            for b in 0..cols / 32 {
                let v = &c[r * cols + b * 32..r * cols + (b + 1) * 32];
                out.push(scale(r));
                out.extend((0..16).map(|j| v[j] | (v[j + 16] << 4)));
            }
        }
        out
    };
    // One published row: blocks of paired values and their scales.
    let row = |c: &[u8], r: usize, cols: usize, blocks: &mut Vec<u8>, scales: &mut Vec<u8>| {
        for b in 0..cols / 32 {
            let v = &c[r * cols + b * 32..r * cols + (b + 1) * 32];
            blocks.extend((0..16).map(|i| v[2 * i] | (v[2 * i + 1] << 4)));
            scales.push(scale(r));
        }
    };
    let (f, h) = (EXPERT_FFN, HIDDEN);
    for il in 0..recipe.mixers.len() {
        let prefix = st.layer.replace("{l}", &il.to_string());
        let (gate, up, down) = (codes(il, f, h), codes(il + 7, f, h), codes(il + 13, h, f));
        for (name, c, rows, cols) in [
            ("ffn_gate_exps.weight", &gate, f, h),
            ("ffn_up_exps.weight", &up, f, h),
            ("ffn_down_exps.weight", &down, h, f),
        ] {
            gguf.insert(
                format!("blk.{il}.{name}"),
                (GgmlDType::MxFp4, file_blocks(c, rows, cols)),
            );
        }
        let (mut blocks, mut scales) = (Vec::new(), Vec::new());
        for r in 0..EXPERTS * f {
            for c in [&gate, &up] {
                row(c, r, h, &mut blocks, &mut scales);
            }
        }
        hf.push((
            format!("{prefix}{gate_up}_blocks"),
            "U8",
            vec![EXPERTS, 2 * f, h / 32, 16],
            blocks,
        ));
        hf.push((
            format!("{prefix}{gate_up}_scales"),
            "U8",
            vec![EXPERTS, 2 * f, h / 32],
            scales,
        ));
        let (mut blocks, mut scales) = (Vec::new(), Vec::new());
        for r in 0..EXPERTS * h {
            row(&down, r, f, &mut blocks, &mut scales);
        }
        hf.push((
            format!("{prefix}{down_name}_blocks"),
            "U8",
            vec![EXPERTS, h, f / 32, 16],
            blocks,
        ));
        hf.push((
            format!("{prefix}{down_name}_scales"),
            "U8",
            vec![EXPERTS, h, f / 32],
            scales,
        ));
    }
    (hf, gguf)
}

/// The publisher's checkpoint written to a directory, with the tensor `without` left out, and
/// loaded as the server loads one; and the replacements its GGUF reference needs.
fn publisher_model(
    tag: &str,
    p: &Publisher,
    without: Option<&str>,
) -> (crate::tensor::Result<GenericHeteroTransformer>, Replaced) {
    use crate::inference::load::awq_loader as awql;
    let weights = recipe_weights(&p.recipe);
    let mut tensors = hf_tensors(&p.recipe, &p.style, &weights);
    let mut replaced = Replaced::new();
    let mut quantised: Vec<Stored> = Vec::new();
    if p.mx {
        tensors.retain(|(n, _, _)| {
            !(n.ends_with("mlp.experts.gate_up_proj") || n.ends_with("mlp.experts.down_proj"))
        });
        (quantised, replaced) = mxfp4_experts(&p.recipe, &p.style);
    }
    tensors.retain(|(n, _, _)| Some(n.as_str()) != without);
    let dir = scratch(tag);
    let dtype = match (p.bf16, p.f16) {
        (true, _) => "BF16",
        (_, true) => "F16",
        _ => "F32",
    };
    let bytes = |vals: &[f32]| match p.f16 {
        true => vals
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect(),
        false => stored(p.bf16, vals),
    };
    let mut files: Vec<Stored> = tensors
        .iter()
        .map(|(name, dims, vals)| (name.clone(), dtype, dims.clone(), bytes(vals)))
        .collect();
    files.extend(quantised);
    let (shard, config) = if p.native {
        ("consolidated.safetensors", "params.json")
    } else {
        ("model.safetensors", "config.json")
    };
    write_stored(&dir.join(shard), &files);
    std::fs::write(dir.join(config), &p.config).unwrap();
    std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    let dir_str = dir.to_string_lossy().to_string();
    assert_eq!(
        awql::find_dense_model_dir(&dir_str).as_deref(),
        Some(dir_str.as_str()),
        "{tag}: the directory is a dense checkpoint"
    );
    let (native, shards) = awql::checkpoint_shards(&dir).unwrap();
    assert_eq!(native, p.native, "{tag}: layout");
    let layers = p.recipe.mixers.len();
    let built = (|| {
        let (header, _) = crate::inference::engine::llm_engine::awq_checkpoint(&dir_str)
            .map_err(|e| crate::tensor::Error::msg(e.to_string()))?;
        let arch = header
            .metadata
            .get("general.architecture")
            .map(|v| v.to_string().unwrap().clone());
        assert_eq!(
            arch.as_deref(),
            Some(p.recipe.arch),
            "{tag}: the class {:?} is read as the recipe's architecture",
            p.class
        );
        GenericWeights::Awq {
            dir: &dir_str,
            shards: &shards,
        }
        .build(
            &HashMap::new(),
            &host_plan(layers),
            crate::inference::engine::llm_engine::KvQuant::Off,
            None,
            false,
            false,
        )
    })();
    let _ = std::fs::remove_dir_all(&dir);
    (built, replaced)
}

/// The relative distance between the logits of the publisher's checkpoint, loaded as the server
/// loads one, and of the same weights in the recipe's GGUF file.
fn publisher_error(tag: &str, p: &Publisher) -> f32 {
    let (built, replaced) = publisher_model(tag, p, None);
    let mut model = built.unwrap_or_else(|e| panic!("{tag}: load failed: {e}"));
    let got = assert_runs(tag, &mut model, VOCAB);
    let mut reference = build_gguf_with(&p.recipe, None, &replaced)
        .unwrap_or_else(|e| panic!("{tag}: the recipe's GGUF load failed: {e}"));
    let want = assert_runs(&format!("{tag} (GGUF)"), &mut reference, VOCAB);
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let worst = got
        .iter()
        .zip(&want)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    worst / scale
}

/// The checkpoint loads and gives the logits of the recipe's GGUF file: to rounding when it is
/// stored in half width, and the GGUF file holds the weights in full.
fn load_publisher(tag: &str, p: &Publisher) {
    let error = publisher_error(tag, p);
    let tolerance = if p.bf16 || p.f16 { 5e-2 } else { 2e-3 };
    assert!(
        error < tolerance,
        "{tag}: the checkpoint's logits are {error} (of their range) from the GGUF file's"
    );
}

macro_rules! publisher_loads {
    ($($test:ident ($arch:literal): $publisher:expr;)*) => {
        $(
            #[test]
            fn $test() {
                load_publisher(stringify!($test), &$publisher);
            }
        )*

        /// The architectures the tests above load from a publisher's safetensors.
        fn publisher_archs() -> Vec<&'static str> {
            vec![$($arch),*]
        }
    };
}

publisher_loads! {
    smollm3_loads_from_safetensors ("smollm3"): smollm3();
    ernie4_5_loads_from_safetensors ("ernie4_5"): ernie4_5();
    granite_loads_from_safetensors ("granite"): granite();
    granitemoe_loads_from_safetensors ("granitemoe"): granitemoe();
    yi_loads_from_safetensors ("yi"): yi();
    internlm2_loads_from_safetensors ("internlm2"): internlm2();
    starcoder_loads_from_safetensors ("starcoder"): starcoder();
    starcoder2_loads_from_safetensors ("starcoder2"): starcoder2();
    qwen3moe_loads_from_safetensors ("qwen3moe"): qwen3moe();
    qwen35_loads_from_safetensors ("qwen35"): qwen35();
    qwen35moe_loads_from_safetensors ("qwen35moe"): qwen35moe();
    qwen3next_loads_from_safetensors ("qwen3next"): qwen3next();
    phi2_loads_from_safetensors ("phi2"): phi2();
    phi3_loads_from_safetensors ("phi3"): phi3();
    chatglm_loads_from_safetensors ("chatglm"): chatglm();
    glm4_loads_from_safetensors ("glm4"): glm4();
    stablelm_loads_from_safetensors ("stablelm"): stablelm();
    olmo2_loads_from_safetensors ("olmo2"): olmo2();
    olmoe_loads_from_safetensors ("olmoe"): olmoe();
    falcon_loads_from_safetensors ("falcon"): falcon();
    dbrx_loads_from_safetensors ("dbrx"): dbrx();
    orion_loads_from_safetensors ("orion"): orion();
    mpt_loads_from_safetensors ("mpt"): mpt();
    gemma_loads_from_safetensors ("gemma"): gemma_publisher();
    gemma2_loads_from_safetensors ("gemma2"): gemma2();
    gemma3_loads_from_safetensors ("gemma3"): gemma3();
    gptoss_loads_from_safetensors ("gptoss"): gptoss();
    lfm2_loads_from_safetensors ("lfm2"): lfm2();
    lfm2moe_loads_from_safetensors ("lfm2moe"): lfm2moe();
    nemotron_h_moe_loads_from_safetensors ("nemotron_h_moe"): nemotron();
    gptoss_loads_from_mxfp4_safetensors ("gptoss"): gptoss_mxfp4();
    gptoss_loads_from_bf16_safetensors ("gptoss"): Publisher { bf16: true, ..gptoss() };
    qwen3next_loads_from_bf16_safetensors ("qwen3next"): Publisher { bf16: true, ..qwen3next() };
    gemma3_loads_from_bf16_safetensors ("gemma3"): Publisher { bf16: true, ..gemma3() };
    starcoder_loads_from_f16_safetensors ("starcoder"): Publisher { f16: true, ..starcoder() };
    llama_safetensors_equal_gguf ("llama"): llama_publisher(recipe_of("llama"), "LlamaForCausalLM", Pairing::AdjacentToHalf);
    qwen2_safetensors_equal_gguf ("qwen2"): llama_publisher(recipe_of("qwen2"), "Qwen2ForCausalLM", Pairing::Same);
    qwen3_safetensors_equal_gguf ("qwen3"): llama_publisher(recipe_of("qwen3"), "Qwen3ForCausalLM", Pairing::Same);
    mixtral_loads_from_safetensors ("llama"): mixtral();
    mixtral_loads_from_native_layout ("llama"): mixtral_native();
}

/// The families whose safetensors tests are the older, hand-written ones above.
const FAMILY_TESTED: [&str; 5] = ["llama", "mistral", "mistral3", "qwen2", "qwen3"];

/// Architectures whose publisher ships safetensors under a class the loader reads as another
/// tag: the architecture, the tag its class gives, and the class.
const READ_AS: [(&str, &str, &str); 4] = [
    ("granite3", "granite", "GraniteForCausalLM"),
    ("phi", "phi2", "PhiForCausalLM"),
    ("phi4", "phi3", "Phi3ForCausalLM"),
    ("falcon3", "llama", "LlamaForCausalLM"),
];

/// Architectures whose publisher ships no safetensors the loader can read, and why.
const NO_SAFETENSORS: [(&str, &str); 3] = [
    (
        "moondream",
        "custom code only, tensor names that change with each revision, no transformers class",
    ),
    (
        "qwen4exp",
        "no transformers class and no published checkpoint; read from GGUF only",
    ),
    (
        "gemma4",
        "the per-layer input, shared key-value and key-as-value tensors of its checkpoint are not known well enough to translate",
    ),
];

/// Every architecture with publisher safetensors has a test that loads them, every other one
/// says why not, and `docs/ARCHITECTURES.md` says the same.
fn safetensors_coverage(known: &std::collections::BTreeSet<&str>) {
    use crate::inference::load::hf_config::{hf_architectures, CLASSES};
    let tested: std::collections::BTreeSet<&str> =
        publisher_archs().into_iter().chain(FAMILY_TESTED).collect();
    let read_as: std::collections::BTreeSet<&str> = READ_AS.iter().map(|r| r.0).collect();
    let none: std::collections::BTreeSet<&str> = NO_SAFETENSORS.iter().map(|r| r.0).collect();
    for arch in known {
        let ways = [
            tested.contains(arch),
            read_as.contains(arch),
            none.contains(arch),
        ];
        assert_eq!(
            ways.iter().filter(|w| **w).count(),
            1,
            "{arch}: a safetensors test, a class it is read as, or a reason there is none"
        );
    }
    for arch in tested.iter().chain(&read_as).chain(&none) {
        assert!(known.contains(arch), "{arch} is not an architecture");
    }
    for arch in hf_architectures() {
        assert!(
            tested.contains(arch),
            "a checkpoint class is read as {arch}, which has no safetensors test"
        );
    }
    for (arch, tag, class) in READ_AS {
        assert!(
            CLASSES.iter().any(|c| c.class == class && c.arch == tag),
            "{arch}: {class} is not read as {tag}"
        );
    }
    let docs = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/ARCHITECTURES.md"),
    )
    .unwrap();
    let row = |arch: &str| {
        docs.lines()
            .find(|l| l.starts_with(&format!("| `{arch}` |")))
            .unwrap_or_else(|| panic!("{arch} has no row in docs/ARCHITECTURES.md"))
    };
    for arch in known {
        let line = row(arch);
        if tested.contains(arch) {
            let test = format!("{arch}_loads_from_safetensors");
            assert!(line.contains(&test), "the row of {arch} lacks {test}");
        }
    }
    for (arch, tag, class) in READ_AS {
        let line = row(arch);
        assert!(
            line.contains(&format!("`{class}`, read as `{tag}`")),
            "the row of {arch} does not say its class is read as {tag}"
        );
    }
    for (arch, reason) in NO_SAFETENSORS {
        let line = row(arch);
        assert!(
            line.contains(&format!("none: {reason}")),
            "the row of {arch} does not give the reason: {reason}"
        );
    }
    for c in CLASSES {
        assert!(
            docs.contains(&format!("`{}`", c.class)),
            "docs/ARCHITECTURES.md does not name {}",
            c.class
        );
    }
}

/// A checkpoint whose layout is read wrongly gives other logits: the comparison with the GGUF
/// file can fail.
#[test]
fn a_wrong_layout_is_caught() {
    let mut pairing = llama_publisher(recipe_of("llama"), "LlamaForCausalLM", Pairing::Same);
    pairing.style.pairing = Pairing::Same;
    let mut grouped = internlm2();
    grouped.style.qkv = Qkv::Contiguous("attention.wqkv");
    let mut shifted = gemma_publisher();
    const UNSHIFTED: [(&str, &str, Xf); 1] = [("output_norm.weight", "model.norm.weight", Id)];
    const NAMES: [&[(&str, &str, Xf)]; 4] =
        [&UNSHIFTED, &LLAMA_GLOBAL, &LLAMA_BLOCK, &LLAMA_BIASES];
    shifted.style.names = &NAMES;
    let mut experts = granitemoe();
    experts.style.pairing = Pairing::Same;
    for (tag, p) in [
        ("rows left in the file's pairing", pairing),
        ("a fused projection read in the wrong order", grouped),
        ("norm scales not shifted", shifted),
        ("granite rows left in the file's pairing", experts),
    ] {
        let error = publisher_error(&format!("wrong-{}", tag.len()), &p);
        assert!(error > 2e-2, "{tag}: the logits are only {error} away");
    }
}

/// A checkpoint missing a tensor of its blocks is refused, whichever way the tensor is stored.
#[test]
fn a_checkpoint_missing_a_tensor_is_refused() {
    for (p, without) in [
        (qwen3moe(), "model.layers.1.mlp.experts.2.down_proj.weight"),
        (
            granitemoe(),
            "model.layers.0.block_sparse_moe.output_linear.weight",
        ),
        (internlm2(), "model.layers.1.attention.wqkv.weight"),
        (
            qwen3next(),
            "model.layers.0.linear_attn.in_proj_qkvz.weight",
        ),
        (smollm3(), "model.layers.1.mlp.down_proj.weight"),
    ] {
        let (built, _) = publisher_model("refused", &p, Some(without));
        assert!(built.is_err(), "{} without {without} loads", p.class);
    }
}

/// A serialised SentencePiece `ModelProto`: each piece with its score and type, and the
/// normaliser's `add_dummy_prefix`.
fn spm_model_bytes(pieces: &[(&str, f32, i32)], add_dummy_prefix: bool) -> Vec<u8> {
    fn varint(out: &mut Vec<u8>, mut v: u64) {
        while v >= 0x80 {
            out.push((v as u8) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }
    fn message(out: &mut Vec<u8>, field: u64, body: &[u8]) {
        varint(out, field << 3 | 2);
        varint(out, body.len() as u64);
        out.extend_from_slice(body);
    }
    let mut out = Vec::new();
    for (text, score, ptype) in pieces {
        let mut piece = Vec::new();
        message(&mut piece, 1, text.as_bytes());
        varint(&mut piece, 2 << 3 | 5);
        piece.extend_from_slice(&score.to_le_bytes());
        varint(&mut piece, 3 << 3);
        varint(&mut piece, *ptype as u64);
        message(&mut out, 1, &piece);
    }
    let mut spec = Vec::new();
    message(&mut spec, 1, b"identity");
    varint(&mut spec, 3 << 3);
    varint(&mut spec, add_dummy_prefix as u64);
    message(&mut out, 3, &spec);
    out
}

/// A checkpoint that ships a SentencePiece `tokenizer.model` and no `tokenizer.json` takes its
/// vocabulary from the model file: the prompt is split into the pieces it holds whole.
#[test]
fn a_checkpoint_without_tokenizer_json_reads_its_sentencepiece_model() {
    const PIECES: [(&str, f32, i32); 22] = [
        ("<unk>", 0.0, 2),
        ("<s>", 0.0, 3),
        ("</s>", 0.0, 3),
        ("\u{2581}", -1.0, 1),
        ("h", -2.0, 1),
        ("e", -2.0, 1),
        ("l", -2.0, 1),
        ("o", -2.0, 1),
        ("w", -2.0, 1),
        ("r", -2.0, 1),
        ("d", -2.0, 1),
        ("he", -3.0, 1),
        ("ll", -3.1, 1),
        ("llo", -3.2, 1),
        ("hello", -3.3, 1),
        ("\u{2581}hello", -3.4, 1),
        ("wo", -3.5, 1),
        ("wor", -3.6, 1),
        ("worl", -3.7, 1),
        ("world", -3.8, 1),
        ("\u{2581}world", -3.9, 1),
        ("<|im_start|>", 0.0, 4),
    ];
    let p = internlm2();
    let weights = recipe_weights(&p.recipe);
    let files: Vec<Stored> = hf_tensors(&p.recipe, &p.style, &weights)
        .iter()
        .map(|(name, dims, vals)| (name.clone(), "F32", dims.clone(), stored(false, vals)))
        .collect();
    let dir = scratch("spm-tokenizer");
    write_stored(&dir.join("model.safetensors"), &files);
    std::fs::write(dir.join("config.json"), &p.config).unwrap();
    std::fs::write(dir.join("tokenizer.model"), spm_model_bytes(&PIECES, true)).unwrap();
    let header = crate::inference::engine::llm_engine::awq_checkpoint(&dir.to_string_lossy());
    let _ = std::fs::remove_dir_all(&dir);
    let (header, _) = header.unwrap_or_else(|e| panic!("the checkpoint is refused: {e}"));
    let tokenizer = crate::inference::engine::llm_engine::build_tokenizer_from_gguf(&header)
        .unwrap_or_else(|e| panic!("no tokenizer from the model file: {e}"));
    let encoding = tokenizer.encode("hello world", false).unwrap();
    assert_eq!(encoding.get_ids(), &[15, 20]);
    assert_eq!(tokenizer.decode(&[15, 20], false).unwrap(), " hello world");
}

/// A feed-forward norm stored with a bias is a LayerNorm: its bias reaches the logits.
#[test]
fn a_feed_forward_layer_norm_applies_its_bias() {
    let recipe = recipe_of("starcoder2");
    let shifted: Replaced = (0..recipe.mixers.len())
        .map(|il| {
            let bytes = (0..HIDDEN).flat_map(|_| 0.5f32.to_le_bytes()).collect();
            (format!("blk.{il}.ffn_norm.bias"), (GgmlDType::F32, bytes))
        })
        .collect();
    let mut plain = build_gguf(&recipe, None).unwrap();
    let mut moved = build_gguf_with(&recipe, None, &shifted).unwrap();
    let a = assert_runs("starcoder2", &mut plain, VOCAB);
    let b = assert_runs("starcoder2, other bias", &mut moved, VOCAB);
    let worst = a
        .iter()
        .zip(&b)
        .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    assert!(
        worst > 1e-3,
        "the feed-forward norm's bias changes nothing ({worst})"
    );
}

/// A learned table of absolute positions is added to the token embeddings, and the heads of a
/// model that has one are not rotated.
#[test]
fn a_learned_position_table_places_the_tokens() {
    let recipe = recipe_of("starcoder");
    let table = (0..POSITIONS * HIDDEN).flat_map(|i| (i as f32 * 0.1).sin().to_le_bytes());
    let moved: Replaced = [(
        "position_embd.weight".to_string(),
        (GgmlDType::F32, table.collect()),
    )]
    .into_iter()
    .collect();
    let mut plain = build_gguf(&recipe, None).unwrap();
    assert!(
        plain.layers.iter().all(|l| l.no_rope),
        "a model with learned positions rotates its heads"
    );
    let mut other = build_gguf_with(&recipe, None, &moved).unwrap();
    let a = assert_runs("starcoder", &mut plain, VOCAB);
    let b = assert_runs("starcoder, other positions", &mut other, VOCAB);
    let worst = a
        .iter()
        .zip(&b)
        .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    assert!(worst > 1e-3, "the position table changes nothing ({worst})");
}

/// A prompt of several tokens does not go through the one-row buffers a decode step keeps for
/// its graph: they are present after a graph was captured, and a longer verify or prefill
/// follows.
#[cfg(feature = "gpu")]
#[test]
fn a_prompt_of_several_tokens_skips_the_one_row_graph_buffers() {
    let recipe = dense("llama");
    let mut model = build_gguf(&recipe, None).unwrap();
    for layer in model.layers.iter_mut() {
        let i = layer.flags.intermediate_size;
        let row = || Tensor::zeros_on((1usize, 1usize, i), DType::F32, &Device::Cpu).unwrap();
        layer.graph_ffn_up_buffer = Some(row());
        layer.graph_ffn_activated_buffer = Some(row());
    }
    assert_runs("llama with the graph buffers set", &mut model, VOCAB);
}
