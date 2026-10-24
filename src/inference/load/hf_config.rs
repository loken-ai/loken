//! What a Hugging Face `config.json` says, as the header keys the loader reads.
//!
//! A checkpoint's `architectures[0]` selects a row of [`CLASSES`]: the tag of the architecture
//! its tensors are read as, the layout that names them (`generic_transformer/dense_layouts.rs`),
//! and the few conventions the class fixes. Everything else is a key of the configuration, read
//! under the names the publishers use for it.

use super::awq_loader::AwqConfigJson;
use crate::tensor::quantized::gguf_file::Value;
use serde_json::Value as Json;

/// The tensor layouts the classes share; `dense_layouts.rs` says what each names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutKind {
    Llama,
    Native,
    Mixtral,
    QwenMoe,
    GraniteMoe,
    Yi,
    InternLm2,
    BigCode,
    Starcoder2,
    Phi,
    Phi3,
    ChatGlm,
    Glm4,
    Olmo2,
    Falcon,
    Dbrx,
    Mpt,
    Gemma,
    Gemma2,
    Gemma4,
    GptOss,
    Lfm2,
    Lfm2Moe,
    Qwen3Next,
    Qwen35,
    Qwen35Moe,
    NemotronH,
}

/// One class a checkpoint's configuration can name.
#[derive(Debug, Clone, Copy)]
pub struct HfClass {
    pub class: &'static str,
    /// The architecture tag the model is built as.
    pub arch: &'static str,
    /// The tensor layout of the class.
    pub layout: LayoutKind,
    /// The class rotates adjacent lanes of a head, which the loader's pairing then follows
    /// whatever the tag's GGUF files hold.
    pub adjacent_rope: bool,
    /// An AWQ checkpoint of the class is read.
    pub awq: bool,
    /// The share of a head the class rotates when its configuration does not say.
    pub partial_rotary: Option<f32>,
    /// Period of the sliding-window layers when the configuration gives a window and no list.
    pub window_period: Option<usize>,
    /// The router scores experts by sigmoid, which the configuration does not say.
    pub sigmoid_router: bool,
}

const fn class(class: &'static str, arch: &'static str, layout: LayoutKind) -> HfClass {
    HfClass {
        class,
        arch,
        layout,
        adjacent_rope: false,
        awq: false,
        partial_rotary: None,
        window_period: None,
        sigmoid_router: false,
    }
}

const fn awq(c: HfClass) -> HfClass {
    HfClass { awq: true, ..c }
}

const fn adjacent(c: HfClass) -> HfClass {
    HfClass {
        adjacent_rope: true,
        ..c
    }
}

const fn windowed(c: HfClass, period: usize) -> HfClass {
    HfClass {
        window_period: Some(period),
        ..c
    }
}

const fn sigmoid(c: HfClass) -> HfClass {
    HfClass {
        sigmoid_router: true,
        ..c
    }
}

const fn partial(c: HfClass, share: f32) -> HfClass {
    HfClass {
        partial_rotary: Some(share),
        ..c
    }
}

/// The classes a checkpoint can name, and what each is read as.
// ARCH-NAME-EXEMPT: the `architectures` classes a checkpoint's configuration declares, and
// the tags they are translated to.
pub const CLASSES: &[HfClass] = &[
    awq(class("LlamaForCausalLM", "llama", LayoutKind::Llama)),
    awq(class("MistralForCausalLM", "llama", LayoutKind::Llama)),
    awq(class("Mistral3ForCausalLM", "llama", LayoutKind::Llama)),
    awq(class(
        "MistralForConditionalGeneration",
        "llama",
        LayoutKind::Llama,
    )),
    awq(class(
        "Mistral3ForConditionalGeneration",
        "llama",
        LayoutKind::Llama,
    )),
    awq(class(
        "VoxtralForConditionalGeneration",
        "llama",
        LayoutKind::Llama,
    )),
    class("MixtralForCausalLM", "llama", LayoutKind::Mixtral),
    awq(class("Qwen2ForCausalLM", "qwen2", LayoutKind::Llama)),
    awq(class("Qwen3ForCausalLM", "qwen3", LayoutKind::Llama)),
    class("Qwen3MoeForCausalLM", "qwen3moe", LayoutKind::QwenMoe),
    class("Qwen3NextForCausalLM", "qwen3next", LayoutKind::Qwen3Next),
    class("Qwen3_5ForCausalLM", "qwen35", LayoutKind::Qwen35),
    class(
        "Qwen3_5ForConditionalGeneration",
        "qwen35",
        LayoutKind::Qwen35,
    ),
    class("Qwen3_5MoeForCausalLM", "qwen35moe", LayoutKind::Qwen35Moe),
    class(
        "Qwen3_5MoeForConditionalGeneration",
        "qwen35moe",
        LayoutKind::Qwen35Moe,
    ),
    class("SmolLM3ForCausalLM", "smollm3", LayoutKind::Llama),
    adjacent(class("Ernie4_5ForCausalLM", "ernie4_5", LayoutKind::Llama)),
    class("GraniteForCausalLM", "granite", LayoutKind::Llama),
    class(
        "GraniteMoeForCausalLM",
        "granitemoe",
        LayoutKind::GraniteMoe,
    ),
    class("YiForCausalLM", "yi", LayoutKind::Yi),
    class("InternLM2ForCausalLM", "internlm2", LayoutKind::InternLm2),
    adjacent(class(
        "GPTBigCodeForCausalLM",
        "starcoder",
        LayoutKind::BigCode,
    )),
    class(
        "Starcoder2ForCausalLM",
        "starcoder2",
        LayoutKind::Starcoder2,
    ),
    class("PhiForCausalLM", "phi2", LayoutKind::Phi),
    class("Phi3ForCausalLM", "phi3", LayoutKind::Phi3),
    partial(
        adjacent(class("ChatGLMModel", "chatglm", LayoutKind::ChatGlm)),
        0.5,
    ),
    partial(
        adjacent(class(
            "ChatGLMForConditionalGeneration",
            "chatglm",
            LayoutKind::ChatGlm,
        )),
        0.5,
    ),
    adjacent(class("Glm4ForCausalLM", "glm4", LayoutKind::Glm4)),
    class("StableLmForCausalLM", "stablelm", LayoutKind::Llama),
    class("Olmo2ForCausalLM", "olmo2", LayoutKind::Olmo2),
    class("OlmoeForCausalLM", "olmoe", LayoutKind::QwenMoe),
    class("FalconForCausalLM", "falcon", LayoutKind::Falcon),
    class("DbrxForCausalLM", "dbrx", LayoutKind::Dbrx),
    class("OrionForCausalLM", "orion", LayoutKind::Llama),
    class("MPTForCausalLM", "mpt", LayoutKind::Mpt),
    class("MptForCausalLM", "mpt", LayoutKind::Mpt),
    class("GemmaForCausalLM", "gemma", LayoutKind::Gemma),
    windowed(class("Gemma2ForCausalLM", "gemma2", LayoutKind::Gemma2), 2),
    windowed(class("Gemma3ForCausalLM", "gemma3", LayoutKind::Gemma2), 6),
    windowed(
        class(
            "Gemma3ForConditionalGeneration",
            "gemma3",
            LayoutKind::Gemma2,
        ),
        6,
    ),
    class("Gemma4ForCausalLM", "gemma4", LayoutKind::Gemma4),
    class(
        "Gemma4ForConditionalGeneration",
        "gemma4",
        LayoutKind::Gemma4,
    ),
    class("GptOssForCausalLM", "gptoss", LayoutKind::GptOss),
    class("Lfm2ForCausalLM", "lfm2", LayoutKind::Lfm2),
    sigmoid(class("Lfm2MoeForCausalLM", "lfm2moe", LayoutKind::Lfm2Moe)),
    class(
        "NemotronHForCausalLM",
        "nemotron_h_moe",
        LayoutKind::NemotronH,
    ),
];

/// The row of `name`, `None` for a class the loader does not read.
pub fn hf_class(name: &str) -> Option<&'static HfClass> {
    CLASSES.iter().find(|c| c.class == name)
}

/// The architectures a checkpoint of some class is read as.
pub fn hf_architectures() -> impl Iterator<Item = &'static str> {
    CLASSES.iter().map(|c| c.arch)
}

/// A value of the configuration under a dotted path, `None` for an absent or null one.
fn path<'a>(v: &'a Json, key: &str) -> Option<&'a Json> {
    key.split('.')
        .try_fold(v, |v, k| v.get(k))
        .filter(|x| !x.is_null())
}

/// The first of `keys` the configuration holds: the text model's fields of a composite
/// checkpoint first, then the file's own.
pub fn lookup<'a>(root: &'a Json, keys: &[&str]) -> Option<&'a Json> {
    let text = root.get("text_config").filter(|t| t.is_object());
    keys.iter()
        .find_map(|k| text.and_then(|t| path(t, k)).or_else(|| path(root, k)))
}

pub fn lookup_u(root: &Json, keys: &[&str]) -> Option<usize> {
    lookup(root, keys)
        .and_then(Json::as_u64)
        .map(|x| x as usize)
}

pub fn lookup_f(root: &Json, keys: &[&str]) -> Option<f32> {
    lookup(root, keys).and_then(Json::as_f64).map(|x| x as f32)
}

pub fn lookup_b(root: &Json, keys: &[&str]) -> Option<bool> {
    lookup(root, keys).and_then(Json::as_bool)
}

/// The configuration's rope block: `rope_scaling`, or `rope_parameters` of a newer one; of a
/// block per layer kind, the global layers'.
pub fn rope_block(root: &Json) -> Option<&Json> {
    lookup(root, &["rope_scaling", "rope_parameters"])
        .filter(|b| b.is_object())
        .map(|b| {
            b.get("full_attention")
                .filter(|g| g.is_object())
                .unwrap_or(b)
        })
}

/// The head width of the global layers when it differs from the configuration's `head_dim`:
/// `global_head_dim`, or the widest of a per-layer configuration.
pub fn global_head_dim(root: &Json) -> Option<usize> {
    lookup_u(root, &["global_head_dim"]).or_else(|| {
        lookup(root, &["per_layer_config"])
            .and_then(Json::as_object)
            .and_then(|layers| {
                layers
                    .values()
                    .filter_map(|l| l.get("head_dim").and_then(Json::as_u64))
                    .max()
            })
            .map(|d| d as usize)
    })
}

/// The rotary frequency factors of a proportional rotation: the first `share` of a head's
/// frequencies turn at the base frequencies over `factor`, the rest stand still. `None` for
/// another rotation.
pub fn proportional_factors(root: &Json, head_dim: usize) -> Option<Vec<f32>> {
    let block = rope_block(root)?;
    if block.get("rope_type").and_then(Json::as_str) != Some("proportional") {
        return None;
    }
    let share = block
        .get("partial_rotary_factor")
        .and_then(Json::as_f64)
        .unwrap_or(1.0);
    let factor = block.get("factor").and_then(Json::as_f64).unwrap_or(1.0) as f32;
    let turning = (share * head_dim as f64) as usize / 2;
    Some(
        (0..head_dim / 2)
            .map(|i| if i < turning { factor } else { f32::INFINITY })
            .collect(),
    )
}

/// The scale of a YaRN rotation's cos and sin tables: `attention_factor`, else the ratio of
/// `0.1 * m * ln(factor) + 1` at `mscale` over the same at `mscale_all_dim` when both are given
/// and not zero, else the same at one. One for a factor of at most one.
fn yarn_attention_factor(block: &Json) -> Option<f32> {
    let get = |k: &str| block.get(k).and_then(Json::as_f64);
    if let Some(f) = get("attention_factor") {
        return Some(f as f32);
    }
    let factor = get("factor")?;
    let scale = |m: f64| {
        if factor <= 1.0 {
            1.0
        } else {
            0.1 * m * factor.ln() + 1.0
        }
    };
    let f = match (get("mscale"), get("mscale_all_dim")) {
        (Some(m), Some(all)) if m != 0.0 && all != 0.0 => scale(m) / scale(all),
        _ => scale(1.0),
    };
    Some(f as f32)
}

/// The sliding-window layers of a model, `true` for a windowed one: its own list of layer
/// types, else the class's period. `None` when no layer is windowed.
fn window_pattern(cfg: &AwqConfigJson, class: Option<&HfClass>) -> Option<Vec<bool>> {
    let root = &cfg.root;
    let types = lookup(root, &["layer_types"]).and_then(Json::as_array);
    let pattern: Vec<bool> = match types {
        Some(types) => types
            .iter()
            .map(|t| t.as_str().is_some_and(|s| s.starts_with("sliding")))
            .collect(),
        None => {
            let period = lookup_u(root, &["sliding_window_pattern"])
                .or(class.and_then(|c| c.window_period))?;
            (0..cfg.n_layers)
                .map(|i| i % period < period.saturating_sub(1))
                .collect()
        }
    };
    pattern.iter().any(|w| *w).then_some(pattern)
}

/// The header keys the configuration gives beyond the core ones, under `{arch}.`.
pub fn hf_metadata(cfg: &AwqConfigJson) -> Vec<(String, Value)> {
    let a = &cfg.gguf_arch;
    let root = &cfg.root;
    let class = hf_class(&cfg.hf_arch);
    let mut out: Vec<(String, Value)> = Vec::new();
    let mut put = |k: &str, v: Value| out.push((format!("{a}.{k}"), v));
    let head_dim = cfg.head_dim.unwrap_or(cfg.hidden_size / cfg.n_head.max(1));

    let share = lookup_f(
        root,
        &[
            "partial_rotary_factor",
            "rope_parameters.partial_rotary_factor",
            "rope_scaling.partial_rotary_factor",
            "rotary_pct",
            "rope_pct",
        ],
    )
    .or(class.and_then(|c| c.partial_rotary));
    if let Some(share) = share.filter(|s| *s < 1.0) {
        let dims = ((head_dim as f32 * share) as usize) & !1;
        put("rope.dimension_count", Value::U32(dims as u32));
    }

    if let Some(n) = lookup_u(
        root,
        &[
            "num_experts",
            "num_local_experts",
            "n_routed_experts",
            "moe_num_experts",
            "ffn_config.moe_num_experts",
            "moe.num_experts",
        ],
    )
    .filter(|n| *n > 0)
    {
        put("expert_count", Value::U32(n as u32));
        let used = lookup_u(
            root,
            &[
                "num_experts_per_tok",
                "top_k_experts",
                "experts_per_token",
                "moe_top_k",
                "ffn_config.moe_top_k",
                "num_experts_per_token",
                "moe.num_experts_per_tok",
            ],
        );
        put("expert_used_count", Value::U32(used.unwrap_or(2) as u32));
        let gating = lookup(root, &["scoring_func"]).and_then(Json::as_str);
        match gating {
            Some("sigmoid") => put("expert_gating_func", Value::U32(2)),
            Some("softmax") => put("expert_gating_func", Value::U32(1)),
            _ if class.is_some_and(|c| c.sigmoid_router) => {
                put("expert_gating_func", Value::U32(2));
            }
            _ => {}
        }
        let normalised = lookup_b(root, &["norm_topk_prob"]).or_else(|| {
            lookup_f(root, &["ffn_config.moe_normalize_expert_weights"]).map(|p| p != 0.0)
        });
        if let Some(norm) = normalised {
            put("norm_top_k_prob", Value::Bool(norm));
        }
        if let Some(s) = lookup_f(root, &["routed_scaling_factor"]) {
            put("expert_weights_scale", Value::F32(s));
        }
    }

    let window = lookup_u(root, &["sliding_window"]);
    if let (Some(window), Some(pattern)) = (window, window_pattern(cfg, class)) {
        put("attention.sliding_window", Value::U32(window as u32));
        put(
            "attention.sliding_window_pattern",
            Value::Array(pattern.into_iter().map(Value::Bool).collect()),
        );
        if let Some(local) = lookup_f(
            root,
            &[
                "rope_local_base_freq",
                "rope_parameters.sliding_attention.rope_theta",
            ],
        ) {
            put("rope.freq_base_swa", Value::F32(local));
        }
    }

    for (key, names) in [
        ("embedding_scale", &["embedding_multiplier"][..]),
        ("residual_scale", &["residual_multiplier"]),
        ("attention.scale", &["attention_multiplier"]),
        ("logit_scale", &["logits_scaling"]),
        ("attn_logit_softcapping", &["attn_logit_softcapping"]),
        ("final_logit_softcapping", &["final_logit_softcapping"]),
    ] {
        if let Some(v) = lookup_f(root, names).filter(|v| *v != 0.0) {
            put(key, Value::F32(v));
        }
    }
    if lookup_b(root, &["attn_config.alibi", "alibi"]) == Some(true) {
        let bias = lookup_f(root, &["attn_config.alibi_bias_max", "alibi_bias_max"]);
        put("attention.max_alibi_bias", Value::F32(bias.unwrap_or(8.0)));
    }
    if let Some(q) = lookup_f(root, &["query_pre_attn_scalar"]).filter(|q| *q > 0.0) {
        put("attention.scale", Value::F32(1.0 / q.sqrt()));
    }

    // The query scale `1 + beta * ln(1 + floor(position / length))` of the rope block.
    if let Some(block) = rope_block(root) {
        // ARCH-NAME-EXEMPT: the keys of the library's query scale.
        let beta = block.get("llama_4_scaling_beta").and_then(Json::as_f64);
        // ARCH-NAME-EXEMPT: the keys of the library's query scale.
        let length = ["llama_4_scaling_length", "original_max_position_embeddings"]
            .iter()
            .find_map(|k| block.get(*k).and_then(Json::as_f64));
        if let (Some(beta), Some(length)) = (beta.filter(|b| *b > 0.0), length) {
            put("attention.temperature_scale", Value::F32(beta as f32));
            put("attention.temperature_length", Value::U32(length as u32));
        }
    }

    if cfg.yarn.is_some() {
        if let Some(block) = rope_block(root) {
            for (key, name) in [
                ("rope.scaling.beta_fast", "beta_fast"),
                ("rope.scaling.beta_slow", "beta_slow"),
            ] {
                if let Some(v) = block.get(name).and_then(Json::as_f64) {
                    put(key, Value::F32(v as f32));
                }
            }
            if let Some(f) = yarn_attention_factor(block) {
                put("rope.scaling.attn_factor", Value::F32(f));
            }
        }
    }

    // The global layers' head width, the layers that read another's keys and values, and the
    // width of the per-layer inputs.
    if let Some(global) = global_head_dim(root).filter(|g| *g != head_dim) {
        put("attention.key_length", Value::U32(global as u32));
        put("attention.value_length", Value::U32(global as u32));
    }
    if let Some(n) = lookup_u(root, &["num_kv_shared_layers"]).filter(|n| *n > 0) {
        put("attention.shared_kv_layers", Value::U32(n as u32));
    }
    if let Some(w) = lookup_u(root, &["hidden_size_per_layer_input"]).filter(|w| *w > 0) {
        put("embedding_length_per_layer_input", Value::U32(w as u32));
    }

    // The recurrent mixers' shapes: the delta rule's, then the state-space model's.
    let conv = lookup_u(root, &["linear_conv_kernel_dim", "conv_kernel", "d_conv"]);
    let delta_values = lookup_u(root, &["linear_num_value_heads"]);
    // ARCH-NAME-EXEMPT: the keys of a state-space model's configuration.
    let ssm_heads = lookup_u(root, &["mamba_num_heads"]);
    if let Some(heads) = delta_values.or(ssm_heads) {
        let inner = match (delta_values, ssm_heads) {
            (Some(h), _) => h * lookup_u(root, &["linear_value_head_dim"]).unwrap_or(0),
            // ARCH-NAME-EXEMPT: a key of a state-space model's configuration.
            (None, Some(h)) => h * lookup_u(root, &["mamba_head_dim"]).unwrap_or(0),
            _ => 0,
        };
        put("ssm.inner_size", Value::U32(inner as u32));
        put("ssm.time_step_rank", Value::U32(heads as u32));
        put("ssm.conv_kernel", Value::U32(conv.unwrap_or(4) as u32));
        let state = lookup_u(
            root,
            &["linear_key_head_dim", "ssm_state_size", "state_size"],
        );
        put("ssm.state_size", Value::U32(state.unwrap_or(0) as u32));
        // ARCH-NAME-EXEMPT: the keys of a configuration that name the delta rule's or the
        // state-space model's groups.
        let groups = lookup_u(
            root,
            &["linear_num_key_heads", "n_groups", "mamba_n_groups"],
        );
        put("ssm.group_count", Value::U32(groups.unwrap_or(1) as u32));
    }
    out
}
