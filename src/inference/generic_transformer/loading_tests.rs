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

const HIDDEN: usize = 64;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 32;
const FFN: usize = 128;
const VOCAB: usize = 96;
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
        }
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
            put("attn_norm.weight", &[HIDDEN]);
            if self.layer_norm {
                put("attn_norm.bias", &[HIDDEN]);
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
                    if self.qk_norm {
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
            if ffn != Ffn::None && !self.parallel {
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
    let dir = std::env::temp_dir().join(format!("loken-loading-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A host-only plan over `layers` blocks: the placement every test shares.
fn host_plan(layers: usize) -> HeteroPlan {
    HeteroPlan::calculate_with_kv(layers, 1 << 20, &[], &[], 1.0, 0)
}

/// Asserts that `model` runs the prompt to finite logits of the vocabulary's size.
fn assert_runs(label: &str, model: &mut GenericHeteroTransformer, vocab: usize) {
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
}

/// The recipe written as a GGUF file and loaded as the server loads one.
fn load_gguf(recipe: &Recipe) {
    let built = build_gguf(recipe, None);
    let mut model = built.unwrap_or_else(|e| panic!("{}: load failed: {e}", recipe.arch));
    assert_runs(recipe.arch, &mut model, VOCAB);
}

/// The recipe's file built into a model, with the tensor `without` left out of the file.
fn build_gguf(recipe: &Recipe, without: Option<&str>) -> Result<GenericHeteroTransformer> {
    let dir = scratch(&format!("{}-{}", recipe.arch, without.is_some()));
    let path = dir.join("m.gguf");
    let entries: Vec<GgufEntry> = recipe
        .tensors()
        .into_iter()
        .enumerate()
        .filter(|(_, (name, _))| Some(name.as_str()) != without)
        .map(|(i, (name, dims))| {
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
    granite_loads_from_gguf: dense("granite");
    granite3_loads_from_gguf: dense("granite3");
    granitemoe_loads_from_gguf: dense("granitemoe").moe();
    yi_loads_from_gguf: dense("yi");
    internlm2_loads_from_gguf: dense("internlm2");
    starcoder_loads_from_gguf: dense("starcoder").classic().fused().bias();
    starcoder2_loads_from_gguf: dense("starcoder2").classic().bias();
    qwen2_loads_from_gguf: dense("qwen2").bias();
    qwen3_loads_from_gguf: dense("qwen3").qk();
    qwen3moe_loads_from_gguf: dense("qwen3moe").qk().moe();
    phi_loads_from_gguf: dense("phi").classic().parallel().fused().bias();
    phi2_loads_from_gguf: dense("phi2").classic().parallel().fused().bias();
    phi3_loads_from_gguf: dense("phi3").fused().ffn(Ffn::FusedGateUp);
    phi4_loads_from_gguf: dense("phi4").fused().ffn(Ffn::FusedGateUp);
    chatglm_loads_from_gguf: dense("chatglm").fused().bias().ffn(Ffn::FusedGateUp);
    glm4_loads_from_gguf: dense("glm4").fused().bias().post().ffn(Ffn::FusedGateUp);
    stablelm_loads_from_gguf: Recipe { ffn_bias: false, ..dense("stablelm").classic().ffn(Ffn::Gated) };
    olmo2_loads_from_gguf: Recipe { post_norms: true, ..dense("olmo2").qk() };
    olmoe_loads_from_gguf: dense("olmoe").qk().moe();
    moondream_loads_from_gguf: dense("moondream").classic().parallel().fused().bias();
    falcon_loads_from_gguf: Recipe { ffn_bias: false, out_bias: false, ..dense("falcon").classic().parallel().fused() };
    falcon3_loads_from_gguf: dense("falcon3");
    dbrx_loads_from_gguf: dense("dbrx").fused().moe();
    orion_loads_from_gguf: Recipe { ffn_bias: false, out_bias: false, ..dense("orion").classic().ffn(Ffn::Gated) };
    mpt_loads_from_gguf: Recipe { ffn_bias: false, out_bias: false, ..dense("mpt").classic().fused() };
    gemma_loads_from_gguf: Recipe { post_norms: false, ..gemma() };
    gemma2_loads_from_gguf: Recipe { arch: "gemma2", ..gemma() };
    gemma3_loads_from_gguf: Recipe { arch: "gemma3", ..gemma().qk() };
    gemma4_loads_from_gguf: Recipe { arch: "gemma4", local_global: true, ..gemma().qk() };
    gptoss_loads_from_gguf: Recipe { sinks: true, out_bias: true, qkv_bias: true, sliding_window: true, ..dense("gptoss").moe() };
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
