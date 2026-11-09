//! The tensor names of the DeepSeek V4.1 family.
//!
//! The released checkpoint names a tensor `embed.weight`, `layers.3.attn.wq_a.weight`,
//! `layers.3.ffn.experts.17.w1.weight` and so on: the reference names.
//! A GGUF file names the same tensor under the spellings its converter chose.
//! The tables here translate a reference name into every spelling a file may use.
//! They also give the name llama.cpp's `deepseek41` loader reads a tensor under.

/// The GGUF metadata prefix of the in-tree scheme of this family.
pub const ARCH: &str = "deepseek_v41";

/// The reference name of block 0's stream mix: no other family carries it.
pub const HYPER_CONNECTION_MARKER: &str = "layers.0.hc_attn_fn";

/// The GGUF spellings of a reference name below `layers.N.`: the bare converter name, then the
/// legacy name the synthetic gates write. A mainline conversion appends `.weight` or `.bias` to
/// the bare name; that spelling is tried too.
fn gguf_block_names(rest: &str) -> Option<(&'static str, &'static str, &'static str)> {
    // (bare name, suffix a mainline conversion appends, legacy name)
    Some(match rest {
        "attn_norm.weight" => ("attn_norm", ".weight", "attn_norm.weight"),
        "ffn_norm.weight" => ("ffn_norm", ".weight", "ffn_norm.weight"),
        "hc_attn_fn" => ("hc_attn_fn", ".weight", "hc_attn_fn.weight"),
        "hc_attn_scale" => ("hc_attn_scale", ".weight", "hc_attn_scale"),
        "hc_attn_base" => ("hc_attn_base", ".weight", "hc_attn_base"),
        "hc_ffn_fn" => ("hc_ffn_fn", ".weight", "hc_ffn_fn.weight"),
        "hc_ffn_scale" => ("hc_ffn_scale", ".weight", "hc_ffn_scale"),
        "hc_ffn_base" => ("hc_ffn_base", ".weight", "hc_ffn_base"),
        "attn.wq_a.weight" => ("attn_q_a", ".weight", "attn_q_a.weight"),
        "attn.q_norm.weight" => ("attn_q_a_norm", ".weight", "attn_q_a_norm.weight"),
        "attn.wq_b.weight" => ("attn_q_b", ".weight", "attn_q_b.weight"),
        "attn.wkv.weight" => ("attn_kv", ".weight", "attn_kv.weight"),
        "attn.kv_norm.weight" => ("attn_kv_a_norm", ".weight", "attn_kv_norm.weight"),
        "attn.attn_sink" => ("attn_sinks", ".weight", "attn_sink"),
        "attn.wo_a.weight" => ("attn_output_a", ".weight", "attn_o_a.weight"),
        "attn.wo_b.weight" => ("attn_output_b", ".weight", "attn_o_b.weight"),
        "attn.compressor.norm.weight" => (
            "attn_compressor_norm",
            ".weight",
            "attn_compressor_norm.weight",
        ),
        "attn.compressor.wkv.weight" => {
            ("attn_compressor_kv", ".weight", "attn_compressor_kv.weight")
        }
        "attn.compressor.wgate.weight" => (
            "attn_compressor_gate",
            ".weight",
            "attn_compressor_gate.weight",
        ),
        "attn.indexer.wq_b.weight" => ("indexer.attn_q_b", ".weight", "attn_indexer_q_b.weight"),
        "attn.indexer.weights_proj.weight" => {
            ("indexer.proj", ".weight", "attn_indexer_weights.weight")
        }
        "attn.indexer.wk.weight" => ("indexer.attn_k", ".weight", "attn_indexer_k.weight"),
        "attn.indexer.k_norm.weight" => ("indexer.k_norm", ".weight", "attn_indexer_k_norm.weight"),
        "ffn.gate.weight" => ("ffn_gate_inp", ".weight", "ffn_gate_inp.weight"),
        "ffn.gate.bias" => ("exp_probs_b", ".bias", "ffn_gate_inp.bias"),
        "ffn.shared_experts.w1.weight" => ("ffn_gate_shexp", ".weight", "ffn_gate_shexp.weight"),
        "ffn.shared_experts.w2.weight" => ("ffn_down_shexp", ".weight", "ffn_down_shexp.weight"),
        "ffn.shared_experts.w3.weight" => ("ffn_up_shexp", ".weight", "ffn_up_shexp.weight"),
        "engram.embed.weight" => ("engram_embd", ".weight", "engram_embd.weight"),
        "engram.wkv.weight" => ("engram_wkv", ".weight", "engram_wkv.weight"),
        "engram.q_weight" => ("engram_q", ".weight", "engram_q.weight"),
        "engram.k_weight" => ("engram_k", ".weight", "engram_k.weight"),
        // The query and key weights multiplied together: not a checkpoint tensor, a file's.
        "engram.gate.weight" => ("engram_gate", ".weight", "engram_gate.weight"),
        _ => return None,
    })
}

/// Every reference name below `layers.N.` that [`gguf_block_names`] spells, the routed experts
/// and the engram table aside.
pub const BLOCK_TENSORS: &[&str] = &[
    "attn_norm.weight",
    "ffn_norm.weight",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_attn_base",
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "attn.wq_a.weight",
    "attn.q_norm.weight",
    "attn.wq_b.weight",
    "attn.wkv.weight",
    "attn.kv_norm.weight",
    "attn.attn_sink",
    "attn.wo_a.weight",
    "attn.wo_b.weight",
    "attn.compressor.norm.weight",
    "attn.compressor.wkv.weight",
    "attn.compressor.wgate.weight",
    "attn.indexer.wq_b.weight",
    "attn.indexer.weights_proj.weight",
    "attn.indexer.wk.weight",
    "attn.indexer.k_norm.weight",
    "ffn.gate.weight",
    "ffn.gate.bias",
    "ffn.shared_experts.w1.weight",
    "ffn.shared_experts.w2.weight",
    "ffn.shared_experts.w3.weight",
    "engram.wkv.weight",
    "engram.q_weight",
    "engram.k_weight",
    "engram.gate.weight",
];

/// The one table llama.cpp's layout holds every engram layer's rows in, layer after layer; the
/// bucket offsets the file declares index it whole.
pub const ENGRAM_TABLE: &str = "per_layer_token_embd.weight";

/// llama.cpp's spelling of a block tensor whose name there is not the bare converter name.
fn llamacpp_block_name(rest: &str) -> Option<&'static str> {
    Some(match rest {
        "engram.wkv.weight" => "engram_kv.weight",
        _ => return None,
    })
}

/// The name llama.cpp's `deepseek41` loader reads a reference tensor under, `None` for a tensor
/// it has no place for. Every engram table maps to [`ENGRAM_TABLE`]; an engram layer's query and
/// key weights have no name there, only their product, `engram.gate.weight`.
pub fn llamacpp_name(name: &str) -> Option<String> {
    match name {
        "embed.weight" => return Some("token_embd.weight".into()),
        "norm.weight" => return Some("output_norm.weight".into()),
        "head.weight" => return Some("output.weight".into()),
        _ => {}
    }
    let (l, rest) = layer_of(name)?;
    if matches!(rest, "engram.q_weight" | "engram.k_weight") {
        return None;
    }
    if let Some((_, w)) = expert_of(rest) {
        return gguf_expert_stack(w).map(|s| format!("blk.{l}.{s}.weight"));
    }
    if rest == "engram.embed.weight" {
        return Some(ENGRAM_TABLE.into());
    }
    if let Some(n) = llamacpp_block_name(rest) {
        return Some(format!("blk.{l}.{n}"));
    }
    let (bare, suffix, _) = gguf_block_names(rest)?;
    Some(format!("blk.{l}.{bare}{suffix}"))
}

/// The GGUF stack a routed expert's projection sits in.
fn gguf_expert_stack(w: &str) -> Option<&'static str> {
    Some(match w {
        "w1" => "ffn_gate_exps",
        "w2" => "ffn_down_exps",
        "w3" => "ffn_up_exps",
        _ => return None,
    })
}

/// The GGUF stacks of a layer's routed experts, in the order an expert unit stores them:
/// gate, up, down, with the reference projection each holds.
pub const EXPERT_STACKS: [(&str, &str); 3] = [
    ("ffn_gate_exps", "w1"),
    ("ffn_up_exps", "w3"),
    ("ffn_down_exps", "w2"),
];

/// One expert of stack `stack` in layer `layer`, stored as a tensor of its own.
pub fn expert_unit_name(layer: usize, stack: &str, expert: usize) -> String {
    format!("blk.{layer}.{stack}.{expert}")
}

/// `layers.{l}.{rest}` taken apart.
pub(crate) fn layer_of(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("layers.")?;
    let (l, rest) = rest.split_once('.')?;
    Some((l.parse().ok()?, rest))
}

/// `ffn.experts.{e}.{w}.weight` taken apart.
pub(crate) fn expert_of(rest: &str) -> Option<(usize, &str)> {
    let rest = rest.strip_prefix("ffn.experts.")?;
    let (e, rest) = rest.split_once('.')?;
    let w = rest.strip_suffix(".weight")?;
    Some((e.parse().ok()?, w))
}

/// Every GGUF spelling of a reference name, most likely first.
pub fn gguf_names(name: &str) -> Vec<String> {
    match name {
        "embed.weight" => return vec!["token_embd".into(), "token_embd.weight".into()],
        "norm.weight" => return vec!["output_norm".into(), "output_norm.weight".into()],
        "head.weight" => return vec!["output".into(), "output.weight".into()],
        _ => {}
    }
    let Some((l, rest)) = layer_of(name) else {
        return vec![name.to_string()];
    };
    if let Some((e, w)) = expert_of(rest) {
        // The stack, in which the expert is addressed separately; then the expert's own tensor,
        // for a file that stores each expert as a unit.
        return match gguf_expert_stack(w) {
            Some(s) => vec![
                format!("blk.{l}.{s}"),
                format!("blk.{l}.{s}.weight"),
                expert_unit_name(l, s, e),
            ],
            None => vec![],
        };
    }
    match gguf_block_names(rest) {
        Some((bare, suffix, legacy)) => {
            let mut names = vec![
                format!("blk.{l}.{bare}"),
                format!("blk.{l}.{bare}{suffix}"),
                format!("blk.{l}.{legacy}"),
            ];
            if let Some(n) = llamacpp_block_name(rest) {
                names.push(format!("blk.{l}.{n}"));
            }
            if rest == "engram.embed.weight" {
                names.push(ENGRAM_TABLE.to_string());
            }
            names
        }
        None => vec![name.to_string()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every reference name the loaders use has GGUF spellings, bare first, and an expert's
    /// name maps to its stack.
    #[test]
    fn reference_names_translate_to_gguf_spellings() {
        assert_eq!(
            gguf_names("embed.weight"),
            vec!["token_embd", "token_embd.weight"]
        );
        assert_eq!(
            gguf_names("layers.3.attn.kv_norm.weight"),
            vec![
                "blk.3.attn_kv_a_norm",
                "blk.3.attn_kv_a_norm.weight",
                "blk.3.attn_kv_norm.weight"
            ]
        );
        assert_eq!(
            gguf_names("layers.3.ffn.gate.bias"),
            vec![
                "blk.3.exp_probs_b",
                "blk.3.exp_probs_b.bias",
                "blk.3.ffn_gate_inp.bias"
            ]
        );
        assert_eq!(
            gguf_names("layers.3.ffn.experts.17.w2.weight"),
            vec![
                "blk.3.ffn_down_exps",
                "blk.3.ffn_down_exps.weight",
                "blk.3.ffn_down_exps.17"
            ]
        );
        assert_eq!(gguf_names("layers.3.attn.attn_sink")[2], "blk.3.attn_sink");
        assert!(gguf_names("layers.3.ffn.experts.17.w9.weight").is_empty());
    }
}
