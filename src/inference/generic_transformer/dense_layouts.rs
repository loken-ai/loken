//! The tensor names each Hugging Face class writes, and what turns them into the loader's.
//!
//! A layout lists, per part of a block, the rule that gives the loader's tensor (`blk.N.attn_q.weight`)
//! from the checkpoint's (`self_attn.q_proj.weight`). Most rules rename; the others say how a
//! stored tensor is reshaped, shifted, split, stacked or transposed to become what a GGUF file
//! of the architecture holds. A part of a block is looked up in the layout's tables in order,
//! and the first table that names it gives every rule for it.

/// How one tensor of a checkpoint becomes tensors the loader reads.
#[derive(Debug, Clone, Copy)]
pub(super) enum Rule {
    /// Read in place under this name.
    Same(&'static str),
    /// A norm scale the checkpoint stores less one: the loader's is the stored value plus one.
    PlusOne(&'static str),
    /// In place, the stored axis of length one after the first dropped (`[c, 1, k]` to `[c, k]`).
    Squeeze(&'static str),
    /// In place, a leading axis of length one dropped (`[1, n]` to `[n]`).
    Flatten(&'static str),
    /// In place, a vector as a column (`[n]` to `[n, 1]`).
    Column(&'static str),
    /// A vector `x` stored as the loader's `-exp(x)`.
    NegExp(&'static str),
    /// `-exp(x)` as a column.
    NegExpColumn(&'static str),
    /// A vector in rows of one group (`[n]` to `[groups, n / groups]`).
    Groups(&'static str),
    /// A fused attention projection whose layout the configuration says: rows grouped by
    /// key-value head when the model declares a decoder of that kind or has no shared head,
    /// else query rows, then key rows, then value rows.
    Qkv(&'static str),
    /// A fused attention projection whose rows are grouped by key-value head.
    Grouped(&'static str),
    /// The delta rule's `qkvz` projection, grouped by key head: the loader's `attn_qkv` and
    /// `attn_gate` are read from it.
    Qkvz,
    /// Per-expert tensors (`{e}` in the part) stacked into one `[experts, rows, cols]`.
    Stack(&'static str),
    /// Rows of each expert's `[2 * rows, cols]`: the first half, or the second when `true`.
    Half(&'static str, bool),
    /// An expert stack whose rows alternate between two projections, `[experts, cols, 2 * rows]`
    /// with the second axis the input: one projection's rows, `true` for the odd ones.
    Alternating(&'static str, bool),
    /// The bias of an alternating stack, `[experts, 2 * rows]`: one projection's.
    AlternatingBias(&'static str, bool),
    /// An alternating stack stored as MXFP4 blocks and scales (`{part}_blocks`, `{part}_scales`).
    AlternatingBlocks(&'static str, bool),
    /// MXFP4 blocks and scales as one stack.
    Blocks(&'static str),
    /// Each expert's `[a, b]` as `[b, a]`.
    Transposed(&'static str),
    /// A flat `[experts * rows, cols]` as `[experts, rows, cols]`, in place.
    Experts(&'static str),
    /// A flat `[experts * rows, cols]` as `[experts, cols, rows]`.
    ExpertsTransposed(&'static str),
}

pub(super) type Table = &'static [(&'static str, Rule)];

/// A class's tensors.
pub(super) struct Layout {
    /// Names of the embedding table, under the model's prefix.
    pub embed: &'static [&'static str],
    /// Where the blocks are, under the model's prefix (`layers`).
    pub layers: &'static str,
    /// The tensors outside the blocks, under the model's prefix.
    pub top: Table,
    /// The output head, under the prefix of the wrapper around the model (the model's own
    /// when the head is inside it), else at the root.
    pub head: Table,
    /// A block's parts.
    pub layer: &'static [Table],
    /// The names are the file's own, without a prefix.
    pub rooted: bool,
}

use crate::inference::load::hf_config::LayoutKind;
use Rule::*;

const NORMS_AND_ATTENTION: Table = &[
    ("input_layernorm.weight", Same("attn_norm.weight")),
    ("input_layernorm.bias", Same("attn_norm.bias")),
    ("post_attention_layernorm.weight", Same("ffn_norm.weight")),
    ("post_attention_layernorm.bias", Same("ffn_norm.bias")),
    ("self_attn.q_proj.weight", Same("attn_q.weight")),
    ("self_attn.k_proj.weight", Same("attn_k.weight")),
    ("self_attn.v_proj.weight", Same("attn_v.weight")),
    ("self_attn.o_proj.weight", Same("attn_output.weight")),
    ("self_attn.q_proj.bias", Same("attn_q.bias")),
    ("self_attn.k_proj.bias", Same("attn_k.bias")),
    ("self_attn.v_proj.bias", Same("attn_v.bias")),
    ("self_attn.o_proj.bias", Same("attn_output.bias")),
    ("self_attn.q_norm.weight", Same("attn_q_norm.weight")),
    ("self_attn.k_norm.weight", Same("attn_k_norm.weight")),
];

const GATED_MLP: Table = &[
    ("mlp.gate_proj.weight", Same("ffn_gate.weight")),
    ("mlp.up_proj.weight", Same("ffn_up.weight")),
    ("mlp.down_proj.weight", Same("ffn_down.weight")),
];

const FINAL_NORM: Table = &[
    ("norm.weight", Same("output_norm.weight")),
    ("norm.bias", Same("output_norm.bias")),
];

const HEAD: Table = &[
    ("lm_head.weight", Same("output.weight")),
    ("lm_head.bias", Same("output.bias")),
];

const LLAMA: Layout = Layout {
    embed: &["embed_tokens.weight"],
    layers: "layers",
    top: FINAL_NORM,
    head: HEAD,
    layer: &[NORMS_AND_ATTENTION, GATED_MLP],
    rooted: false,
};

/// The vendor's own layout of the llama family, with a routed feed-forward when it has one.
const NATIVE: Layout = Layout {
    embed: &[
        "tok_embeddings.weight",
        "mm_whisper_embeddings.tok_embeddings.weight",
    ],
    layers: "layers",
    top: &[("norm.weight", Same("output_norm.weight"))],
    head: &[("output.weight", Same("output.weight"))],
    layer: &[&[
        ("attention_norm.weight", Same("attn_norm.weight")),
        ("ffn_norm.weight", Same("ffn_norm.weight")),
        ("attention.wq.weight", Same("attn_q.weight")),
        ("attention.wk.weight", Same("attn_k.weight")),
        ("attention.wv.weight", Same("attn_v.weight")),
        ("attention.wo.weight", Same("attn_output.weight")),
        ("attention.wq.bias", Same("attn_q.bias")),
        ("attention.wk.bias", Same("attn_k.bias")),
        ("attention.wv.bias", Same("attn_v.bias")),
        ("feed_forward.w1.weight", Same("ffn_gate.weight")),
        ("feed_forward.w3.weight", Same("ffn_up.weight")),
        ("feed_forward.w2.weight", Same("ffn_down.weight")),
        ("feed_forward.gate.weight", Same("ffn_gate_inp.weight")),
        (
            "feed_forward.experts.{e}.w1.weight",
            Stack("ffn_gate_exps.weight"),
        ),
        (
            "feed_forward.experts.{e}.w3.weight",
            Stack("ffn_up_exps.weight"),
        ),
        (
            "feed_forward.experts.{e}.w2.weight",
            Stack("ffn_down_exps.weight"),
        ),
    ]],
    rooted: true,
};

const MIXTRAL: Layout = Layout {
    layer: &[
        &[
            ("block_sparse_moe.gate.weight", Same("ffn_gate_inp.weight")),
            (
                "block_sparse_moe.experts.{e}.w1.weight",
                Stack("ffn_gate_exps.weight"),
            ),
            (
                "block_sparse_moe.experts.{e}.w3.weight",
                Stack("ffn_up_exps.weight"),
            ),
            (
                "block_sparse_moe.experts.{e}.w2.weight",
                Stack("ffn_down_exps.weight"),
            ),
        ],
        NORMS_AND_ATTENTION,
    ],
    ..LLAMA
};

/// Routed experts one tensor each, a router named `mlp.gate`.
const QWEN_MOE: Layout = Layout {
    layer: &[
        &[
            ("mlp.gate.weight", Same("ffn_gate_inp.weight")),
            (
                "mlp.experts.{e}.gate_proj.weight",
                Stack("ffn_gate_exps.weight"),
            ),
            (
                "mlp.experts.{e}.up_proj.weight",
                Stack("ffn_up_exps.weight"),
            ),
            (
                "mlp.experts.{e}.down_proj.weight",
                Stack("ffn_down_exps.weight"),
            ),
        ],
        NORMS_AND_ATTENTION,
        GATED_MLP,
    ],
    ..LLAMA
};

const GRANITEMOE: Layout = Layout {
    layer: &[
        &[
            (
                "block_sparse_moe.router.layer.weight",
                Same("ffn_gate_inp.weight"),
            ),
            (
                "block_sparse_moe.input_linear.weight",
                Half("ffn_gate_exps.weight", false),
            ),
            (
                "block_sparse_moe.input_linear.weight",
                Half("ffn_up_exps.weight", true),
            ),
            (
                "block_sparse_moe.output_linear.weight",
                Same("ffn_down_exps.weight"),
            ),
        ],
        NORMS_AND_ATTENTION,
    ],
    ..LLAMA
};

/// The first release's names of the two norms of a block.
const YI: Layout = Layout {
    layer: &[
        &[
            ("ln1.weight", Same("attn_norm.weight")),
            ("ln2.weight", Same("ffn_norm.weight")),
        ],
        NORMS_AND_ATTENTION,
        GATED_MLP,
    ],
    ..LLAMA
};

const INTERNLM2: Layout = Layout {
    embed: &["tok_embeddings.weight"],
    top: &[("norm.weight", Same("output_norm.weight"))],
    head: &[("output.weight", Same("output.weight"))],
    layer: &[&[
        ("attention_norm.weight", Same("attn_norm.weight")),
        ("ffn_norm.weight", Same("ffn_norm.weight")),
        ("attention.wqkv.weight", Grouped("attn_qkv.weight")),
        ("attention.wqkv.bias", Grouped("attn_qkv.bias")),
        ("attention.wo.weight", Same("attn_output.weight")),
        ("attention.wo.bias", Same("attn_output.bias")),
        ("feed_forward.w1.weight", Same("ffn_gate.weight")),
        ("feed_forward.w3.weight", Same("ffn_up.weight")),
        ("feed_forward.w2.weight", Same("ffn_down.weight")),
    ]],
    ..LLAMA
};

const BIGCODE: Layout = Layout {
    embed: &["wte.weight"],
    layers: "h",
    top: &[
        ("wpe.weight", Same("position_embd.weight")),
        ("ln_f.weight", Same("output_norm.weight")),
        ("ln_f.bias", Same("output_norm.bias")),
    ],
    head: HEAD,
    layer: &[&[
        ("ln_1.weight", Same("attn_norm.weight")),
        ("ln_1.bias", Same("attn_norm.bias")),
        ("attn.c_attn.weight", Qkv("attn_qkv.weight")),
        ("attn.c_attn.bias", Qkv("attn_qkv.bias")),
        ("attn.c_proj.weight", Same("attn_output.weight")),
        ("attn.c_proj.bias", Same("attn_output.bias")),
        ("ln_2.weight", Same("ffn_norm.weight")),
        ("ln_2.bias", Same("ffn_norm.bias")),
        ("mlp.c_fc.weight", Same("ffn_up.weight")),
        ("mlp.c_fc.bias", Same("ffn_up.bias")),
        ("mlp.c_proj.weight", Same("ffn_down.weight")),
        ("mlp.c_proj.bias", Same("ffn_down.bias")),
    ]],
    rooted: false,
};

const STARCODER2: Layout = Layout {
    layer: &[
        &[
            ("mlp.c_fc.weight", Same("ffn_up.weight")),
            ("mlp.c_fc.bias", Same("ffn_up.bias")),
            ("mlp.c_proj.weight", Same("ffn_down.weight")),
            ("mlp.c_proj.bias", Same("ffn_down.bias")),
        ],
        NORMS_AND_ATTENTION,
    ],
    ..LLAMA
};

const PHI: Layout = Layout {
    top: &[
        ("final_layernorm.weight", Same("output_norm.weight")),
        ("final_layernorm.bias", Same("output_norm.bias")),
    ],
    layer: &[
        &[
            ("self_attn.dense.weight", Same("attn_output.weight")),
            ("self_attn.dense.bias", Same("attn_output.bias")),
            ("mlp.fc1.weight", Same("ffn_up.weight")),
            ("mlp.fc1.bias", Same("ffn_up.bias")),
            ("mlp.fc2.weight", Same("ffn_down.weight")),
            ("mlp.fc2.bias", Same("ffn_down.bias")),
        ],
        NORMS_AND_ATTENTION,
    ],
    ..LLAMA
};

const PHI3: Layout = Layout {
    layer: &[
        &[
            ("self_attn.qkv_proj.weight", Same("attn_qkv.weight")),
            ("mlp.gate_up_proj.weight", Same("ffn_up.weight")),
        ],
        NORMS_AND_ATTENTION,
        GATED_MLP,
    ],
    ..LLAMA
};

const CHATGLM: Layout = Layout {
    embed: &["embedding.word_embeddings.weight"],
    top: &[("encoder.final_layernorm.weight", Same("output_norm.weight"))],
    head: &[("output_layer.weight", Same("output.weight"))],
    layers: "encoder.layers",
    layer: &[&[
        ("input_layernorm.weight", Same("attn_norm.weight")),
        ("post_attention_layernorm.weight", Same("ffn_norm.weight")),
        (
            "self_attention.query_key_value.weight",
            Same("attn_qkv.weight"),
        ),
        ("self_attention.query_key_value.bias", Same("attn_qkv.bias")),
        ("self_attention.dense.weight", Same("attn_output.weight")),
        ("mlp.dense_h_to_4h.weight", Same("ffn_up.weight")),
        ("mlp.dense_4h_to_h.weight", Same("ffn_down.weight")),
    ]],
    rooted: false,
};

/// The four norms of a block: before and after each of attention and the feed-forward.
const GLM4: Layout = Layout {
    layer: &[
        &[
            (
                "post_self_attn_layernorm.weight",
                Same("post_attention_norm.weight"),
            ),
            ("post_mlp_layernorm.weight", Same("post_ffw_norm.weight")),
            ("mlp.gate_up_proj.weight", Same("ffn_up.weight")),
        ],
        NORMS_AND_ATTENTION,
        GATED_MLP,
    ],
    ..LLAMA
};

/// The only norms of a block follow attention and the feed-forward.
const OLMO2: Layout = Layout {
    layer: &[
        &[
            (
                "post_attention_layernorm.weight",
                Same("post_attention_norm.weight"),
            ),
            (
                "post_feedforward_layernorm.weight",
                Same("post_ffw_norm.weight"),
            ),
        ],
        NORMS_AND_ATTENTION,
        GATED_MLP,
    ],
    ..LLAMA
};

const FALCON: Layout = Layout {
    embed: &["word_embeddings.weight"],
    layers: "h",
    top: &[
        ("ln_f.weight", Same("output_norm.weight")),
        ("ln_f.bias", Same("output_norm.bias")),
    ],
    head: HEAD,
    layer: &[&[
        ("input_layernorm.weight", Same("attn_norm.weight")),
        ("input_layernorm.bias", Same("attn_norm.bias")),
        ("post_attention_layernorm.weight", Same("ffn_norm.weight")),
        ("post_attention_layernorm.bias", Same("ffn_norm.bias")),
        (
            "self_attention.query_key_value.weight",
            Qkv("attn_qkv.weight"),
        ),
        ("self_attention.query_key_value.bias", Qkv("attn_qkv.bias")),
        ("self_attention.dense.weight", Same("attn_output.weight")),
        ("self_attention.dense.bias", Same("attn_output.bias")),
        ("mlp.dense_h_to_4h.weight", Same("ffn_up.weight")),
        ("mlp.dense_h_to_4h.bias", Same("ffn_up.bias")),
        ("mlp.dense_4h_to_h.weight", Same("ffn_down.weight")),
        ("mlp.dense_4h_to_h.bias", Same("ffn_down.bias")),
    ]],
    rooted: false,
};

const DBRX: Layout = Layout {
    embed: &["wte.weight"],
    layers: "blocks",
    top: &[("norm_f.weight", Same("output_norm.weight"))],
    head: HEAD,
    layer: &[&[
        ("norm_attn_norm.norm_1.weight", Same("attn_norm.weight")),
        ("norm_attn_norm.norm_2.weight", Same("ffn_norm.weight")),
        ("norm_attn_norm.attn.Wqkv.weight", Same("attn_qkv.weight")),
        (
            "norm_attn_norm.attn.out_proj.weight",
            Same("attn_output.weight"),
        ),
        ("ffn.router.layer.weight", Same("ffn_gate_inp.weight")),
        ("ffn.experts.mlp.w1", Experts("ffn_gate_exps.weight")),
        ("ffn.experts.mlp.v1", Experts("ffn_up_exps.weight")),
        (
            "ffn.experts.mlp.w2",
            ExpertsTransposed("ffn_down_exps.weight"),
        ),
    ]],
    rooted: false,
};

const MPT: Layout = Layout {
    embed: &["wte.weight"],
    layers: "blocks",
    top: &[
        ("norm_f.weight", Same("output_norm.weight")),
        ("norm_f.bias", Same("output_norm.bias")),
    ],
    head: HEAD,
    layer: &[&[
        ("norm_1.weight", Same("attn_norm.weight")),
        ("norm_1.bias", Same("attn_norm.bias")),
        ("norm_2.weight", Same("ffn_norm.weight")),
        ("norm_2.bias", Same("ffn_norm.bias")),
        ("attn.Wqkv.weight", Same("attn_qkv.weight")),
        ("attn.Wqkv.bias", Same("attn_qkv.bias")),
        ("attn.out_proj.weight", Same("attn_output.weight")),
        ("attn.out_proj.bias", Same("attn_output.bias")),
        ("ffn.up_proj.weight", Same("ffn_up.weight")),
        ("ffn.up_proj.bias", Same("ffn_up.bias")),
        ("ffn.down_proj.weight", Same("ffn_down.weight")),
        ("ffn.down_proj.bias", Same("ffn_down.bias")),
    ]],
    rooted: false,
};

/// Norm scales stored less one.
const GEMMA_NORMS: Table = &[
    ("input_layernorm.weight", PlusOne("attn_norm.weight")),
    (
        "post_attention_layernorm.weight",
        PlusOne("ffn_norm.weight"),
    ),
    ("self_attn.q_norm.weight", PlusOne("attn_q_norm.weight")),
    ("self_attn.k_norm.weight", PlusOne("attn_k_norm.weight")),
];

const GEMMA_FINAL: Table = &[("norm.weight", PlusOne("output_norm.weight"))];

const GEMMA: Layout = Layout {
    top: GEMMA_FINAL,
    layer: &[GEMMA_NORMS, NORMS_AND_ATTENTION, GATED_MLP],
    ..LLAMA
};

/// Four norms around attention and the feed-forward.
const GEMMA_FOUR_NORMS: Table = &[
    (
        "post_attention_layernorm.weight",
        PlusOne("post_attention_norm.weight"),
    ),
    (
        "pre_feedforward_layernorm.weight",
        PlusOne("ffn_norm.weight"),
    ),
    (
        "post_feedforward_layernorm.weight",
        PlusOne("post_ffw_norm.weight"),
    ),
];

const GEMMA2: Layout = Layout {
    top: GEMMA_FINAL,
    layer: &[
        GEMMA_FOUR_NORMS,
        GEMMA_NORMS,
        NORMS_AND_ATTENTION,
        GATED_MLP,
    ],
    ..LLAMA
};

/// Norm scales stored as they are applied, four around attention and the feed-forward, and
/// the per-layer input: its gate, projection and norm, and the block's output scale.
const GEMMA4_BLOCK: Table = &[
    (
        "post_attention_layernorm.weight",
        Same("post_attention_norm.weight"),
    ),
    ("pre_feedforward_layernorm.weight", Same("ffn_norm.weight")),
    (
        "post_feedforward_layernorm.weight",
        Same("post_ffw_norm.weight"),
    ),
    ("per_layer_input_gate.weight", Same("inp_gate.weight")),
    ("per_layer_projection.weight", Same("proj.weight")),
    ("post_per_layer_input_norm.weight", Same("post_norm.weight")),
    ("layer_scalar", Same("layer_output_scale.weight")),
];

const GEMMA4: Layout = Layout {
    top: &[
        ("norm.weight", Same("output_norm.weight")),
        (
            "embed_tokens_per_layer.weight",
            Same("per_layer_token_embd.weight"),
        ),
        (
            "per_layer_model_projection.weight",
            Same("per_layer_model_proj.weight"),
        ),
        (
            "per_layer_projection_norm.weight",
            Same("per_layer_proj_norm.weight"),
        ),
    ],
    layer: &[GEMMA4_BLOCK, NORMS_AND_ATTENTION, GATED_MLP],
    ..LLAMA
};

const GPTOSS: Layout = Layout {
    layer: &[
        &[
            ("self_attn.sinks", Same("attn_sinks")),
            ("mlp.router.weight", Same("ffn_gate_inp.weight")),
            ("mlp.router.bias", Same("ffn_gate_inp.bias")),
            (
                "mlp.experts.gate_up_proj",
                Alternating("ffn_gate_exps.weight", false),
            ),
            (
                "mlp.experts.gate_up_proj",
                Alternating("ffn_up_exps.weight", true),
            ),
            (
                "mlp.experts.gate_up_proj_bias",
                AlternatingBias("ffn_gate_exps.bias", false),
            ),
            (
                "mlp.experts.gate_up_proj_bias",
                AlternatingBias("ffn_up_exps.bias", true),
            ),
            ("mlp.experts.down_proj", Transposed("ffn_down_exps.weight")),
            ("mlp.experts.down_proj_bias", Same("ffn_down_exps.bias")),
            (
                "mlp.experts.gate_up_proj_blocks",
                AlternatingBlocks("ffn_gate_exps.weight", false),
            ),
            (
                "mlp.experts.gate_up_proj_blocks",
                AlternatingBlocks("ffn_up_exps.weight", true),
            ),
            (
                "mlp.experts.down_proj_blocks",
                Blocks("ffn_down_exps.weight"),
            ),
        ],
        NORMS_AND_ATTENTION,
    ],
    ..LLAMA
};

const LFM2: Layout = Layout {
    top: &[("embedding_norm.weight", Same("token_embd_norm.weight"))],
    layer: &[&[
        ("operator_norm.weight", Same("attn_norm.weight")),
        ("ffn_norm.weight", Same("ffn_norm.weight")),
        ("conv.in_proj.weight", Same("shortconv.in_proj.weight")),
        ("conv.conv.weight", Squeeze("shortconv.conv.weight")),
        ("conv.out_proj.weight", Same("shortconv.out_proj.weight")),
        ("self_attn.q_proj.weight", Same("attn_q.weight")),
        ("self_attn.k_proj.weight", Same("attn_k.weight")),
        ("self_attn.v_proj.weight", Same("attn_v.weight")),
        ("self_attn.out_proj.weight", Same("attn_output.weight")),
        ("self_attn.q_layernorm.weight", Same("attn_q_norm.weight")),
        ("self_attn.k_layernorm.weight", Same("attn_k_norm.weight")),
        ("feed_forward.w1.weight", Same("ffn_gate.weight")),
        ("feed_forward.w3.weight", Same("ffn_up.weight")),
        ("feed_forward.w2.weight", Same("ffn_down.weight")),
    ]],
    ..LLAMA
};

const LFM2MOE: Layout = Layout {
    layer: &[
        &[
            ("feed_forward.gate.weight", Same("ffn_gate_inp.weight")),
            ("feed_forward.expert_bias", Same("exp_probs_b.bias")),
            (
                "feed_forward.experts.{e}.w1.weight",
                Stack("ffn_gate_exps.weight"),
            ),
            (
                "feed_forward.experts.{e}.w3.weight",
                Stack("ffn_up_exps.weight"),
            ),
            (
                "feed_forward.experts.{e}.w2.weight",
                Stack("ffn_down_exps.weight"),
            ),
        ],
        LFM2.layer[0],
    ],
    ..LFM2
};

/// The delta rule's tensors, shared by the hybrids that carry it.
const DELTA: Table = &[
    ("linear_attn.conv1d.weight", Squeeze("ssm_conv1d.weight")),
    ("linear_attn.A_log", NegExp("ssm_a")),
    ("linear_attn.dt_bias", Same("ssm_dt")),
    ("linear_attn.norm.weight", Same("ssm_norm.weight")),
    ("linear_attn.out_proj.weight", Same("ssm_out.weight")),
];

/// Norms stored less one, the feed-forward's named for what it follows.
const QWEN_NEXT_NORMS: Table = &[
    ("input_layernorm.weight", PlusOne("attn_norm.weight")),
    (
        "post_attention_layernorm.weight",
        PlusOne("post_attention_norm.weight"),
    ),
    ("self_attn.q_norm.weight", PlusOne("attn_q_norm.weight")),
    ("self_attn.k_norm.weight", PlusOne("attn_k_norm.weight")),
];

const QWEN_NEXT_FINAL: Table = &[("norm.weight", PlusOne("output_norm.weight"))];

const DELTA_SPLIT: Table = &[
    ("linear_attn.in_proj_qkv.weight", Same("attn_qkv.weight")),
    ("linear_attn.in_proj_z.weight", Same("attn_gate.weight")),
    ("linear_attn.in_proj_b.weight", Same("ssm_beta.weight")),
    ("linear_attn.in_proj_a.weight", Same("ssm_alpha.weight")),
];

const DELTA_PACKED: Table = &[
    ("linear_attn.in_proj_qkvz.weight", Qkvz),
    ("linear_attn.in_proj_ba.weight", Same("ssm_ba.weight")),
];

const SHARED_EXPERT: Table = &[
    ("mlp.gate.weight", Same("ffn_gate_inp.weight")),
    (
        "mlp.shared_expert.gate_proj.weight",
        Same("ffn_gate_shexp.weight"),
    ),
    (
        "mlp.shared_expert.up_proj.weight",
        Same("ffn_up_shexp.weight"),
    ),
    (
        "mlp.shared_expert.down_proj.weight",
        Same("ffn_down_shexp.weight"),
    ),
    (
        "mlp.shared_expert_gate.weight",
        Flatten("ffn_gate_inp_shexp.weight"),
    ),
];

/// Routed experts, one tensor each or fused as the newer releases write them.
const EXPERTS: Table = &[
    (
        "mlp.experts.{e}.gate_proj.weight",
        Stack("ffn_gate_exps.weight"),
    ),
    (
        "mlp.experts.{e}.up_proj.weight",
        Stack("ffn_up_exps.weight"),
    ),
    (
        "mlp.experts.{e}.down_proj.weight",
        Stack("ffn_down_exps.weight"),
    ),
    (
        "mlp.experts.gate_up_proj",
        Half("ffn_gate_exps.weight", false),
    ),
    ("mlp.experts.gate_up_proj", Half("ffn_up_exps.weight", true)),
    ("mlp.experts.down_proj", Same("ffn_down_exps.weight")),
];

const QWEN3NEXT: Layout = Layout {
    top: QWEN_NEXT_FINAL,
    layer: &[
        QWEN_NEXT_NORMS,
        DELTA,
        DELTA_PACKED,
        SHARED_EXPERT,
        EXPERTS,
        NORMS_AND_ATTENTION,
        GATED_MLP,
    ],
    ..LLAMA
};

const QWEN35: Layout = Layout {
    top: QWEN_NEXT_FINAL,
    layer: &[
        QWEN_NEXT_NORMS,
        DELTA,
        DELTA_SPLIT,
        NORMS_AND_ATTENTION,
        GATED_MLP,
    ],
    ..LLAMA
};

const QWEN35MOE: Layout = Layout {
    top: QWEN_NEXT_FINAL,
    layer: &[
        QWEN_NEXT_NORMS,
        DELTA,
        DELTA_SPLIT,
        SHARED_EXPERT,
        EXPERTS,
        NORMS_AND_ATTENTION,
        GATED_MLP,
    ],
    ..LLAMA
};

const NEMOTRON_H: Layout = Layout {
    embed: &["embeddings.weight"],
    top: &[("norm_f.weight", Same("output_norm.weight"))],
    head: HEAD,
    layer: &[&[
        ("norm.weight", Same("attn_norm.weight")),
        ("mixer.in_proj.weight", Same("ssm_in.weight")),
        ("mixer.conv1d.weight", Squeeze("ssm_conv1d.weight")),
        ("mixer.conv1d.bias", Same("ssm_conv1d.bias")),
        ("mixer.A_log", NegExpColumn("ssm_a")),
        ("mixer.D", Column("ssm_d")),
        ("mixer.dt_bias", Same("ssm_dt.bias")),
        ("mixer.norm.weight", Groups("ssm_norm.weight")),
        ("mixer.out_proj.weight", Same("ssm_out.weight")),
        ("mixer.q_proj.weight", Same("attn_q.weight")),
        ("mixer.k_proj.weight", Same("attn_k.weight")),
        ("mixer.v_proj.weight", Same("attn_v.weight")),
        ("mixer.o_proj.weight", Same("attn_output.weight")),
        ("mixer.up_proj.weight", Same("ffn_up.weight")),
        ("mixer.down_proj.weight", Same("ffn_down.weight")),
        ("mixer.gate.weight", Same("ffn_gate_inp.weight")),
        (
            "mixer.gate.e_score_correction_bias",
            Same("exp_probs_b.bias"),
        ),
        (
            "mixer.experts.{e}.up_proj.weight",
            Stack("ffn_up_exps.weight"),
        ),
        (
            "mixer.experts.{e}.down_proj.weight",
            Stack("ffn_down_exps.weight"),
        ),
        (
            "mixer.shared_experts.up_proj.weight",
            Same("ffn_up_shexp.weight"),
        ),
        (
            "mixer.shared_experts.down_proj.weight",
            Same("ffn_down_shexp.weight"),
        ),
    ]],
    ..LLAMA
};

/// The layout of a class's tensors.
pub(super) fn layout(kind: LayoutKind) -> &'static Layout {
    match kind {
        LayoutKind::Llama => &LLAMA,
        LayoutKind::Native => &NATIVE,
        LayoutKind::Mixtral => &MIXTRAL,
        LayoutKind::QwenMoe => &QWEN_MOE,
        LayoutKind::GraniteMoe => &GRANITEMOE,
        LayoutKind::Yi => &YI,
        LayoutKind::InternLm2 => &INTERNLM2,
        LayoutKind::BigCode => &BIGCODE,
        LayoutKind::Starcoder2 => &STARCODER2,
        LayoutKind::Phi => &PHI,
        LayoutKind::Phi3 => &PHI3,
        LayoutKind::ChatGlm => &CHATGLM,
        LayoutKind::Glm4 => &GLM4,
        LayoutKind::Olmo2 => &OLMO2,
        LayoutKind::Falcon => &FALCON,
        LayoutKind::Dbrx => &DBRX,
        LayoutKind::Mpt => &MPT,
        LayoutKind::Gemma => &GEMMA,
        LayoutKind::Gemma2 => &GEMMA2,
        LayoutKind::Gemma4 => &GEMMA4,
        LayoutKind::GptOss => &GPTOSS,
        LayoutKind::Lfm2 => &LFM2,
        LayoutKind::Lfm2Moe => &LFM2MOE,
        LayoutKind::Qwen3Next => &QWEN3NEXT,
        LayoutKind::Qwen35 => &QWEN35,
        LayoutKind::Qwen35Moe => &QWEN35MOE,
        LayoutKind::NemotronH => &NEMOTRON_H,
    }
}
