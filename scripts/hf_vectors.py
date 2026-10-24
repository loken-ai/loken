#!/usr/bin/env python3
"""Tiny random checkpoints and their reference logits, one folder per architecture.

Each model is built from its `transformers` configuration class with random weights, saved with
`save_pretrained` as float32 safetensors, and run in float32 on the CPU over a fixed prompt. The
folder holds `config.json`, `model.safetensors` and `reference.json`: the logits of the last
position of the prompt, the greedy ids of the next tokens with the logits each was chosen from
and the gap between their two top values.

Usage: hf_vectors.py OUT_DIR [ARCH ...]
"""

import json
import copy
import math
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import transformers as tf

PROMPT = [1, 17, 42, 99, 5, 73]
STEPS = 4
VOCAB = 128
HIDDEN = 32

BASE = dict(
    vocab_size=VOCAB,
    hidden_size=HIDDEN,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=32,
    max_position_embeddings=256,
)


def base(**kw):
    out = dict(BASE)
    out.update(kw)
    return out


def legacy(**kw):
    """A configuration whose head width is the hidden size over the heads."""
    out = base(num_attention_heads=4, num_key_value_heads=2, **kw)
    del out["head_dim"]
    return out


MOE = dict(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=HIDDEN)

DELTA = dict(
    num_hidden_layers=4,
    linear_conv_kernel_dim=4,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
)

# Folder name: (config class, model class, configuration keys).
ARCHS = {
    "llama": ("LlamaConfig", "LlamaForCausalLM", base()),
    "mistral": ("MistralConfig", "MistralForCausalLM", base(sliding_window=None)),
    "mixtral": (
        "MixtralConfig",
        "MixtralForCausalLM",
        base(intermediate_size=HIDDEN, num_local_experts=4, num_experts_per_tok=2, sliding_window=None),
    ),
    "qwen2": ("Qwen2Config", "Qwen2ForCausalLM", base()),
    "qwen3": ("Qwen3Config", "Qwen3ForCausalLM", base()),
    "qwen3moe": (
        "Qwen3MoeConfig",
        "Qwen3MoeForCausalLM",
        base(norm_topk_prob=True, **MOE),
    ),
    "qwen3next": (
        "Qwen3NextConfig",
        "Qwen3NextForCausalLM",
        base(**DELTA, **MOE, shared_expert_intermediate_size=HIDDEN, norm_topk_prob=True),
    ),
    "qwen35": ("Qwen3_5TextConfig", "Qwen3_5ForCausalLM", base(**DELTA)),
    "qwen35moe": (
        "Qwen3_5MoeTextConfig",
        "Qwen3_5MoeForCausalLM",
        base(**DELTA, **MOE, shared_expert_intermediate_size=HIDDEN),
    ),
    "smollm3": ("SmolLM3Config", "SmolLM3ForCausalLM", base(num_hidden_layers=4, pad_token_id=0)),
    "ernie4_5": ("Ernie4_5Config", "Ernie4_5ForCausalLM", base()),
    "granite": (
        "GraniteConfig",
        "GraniteForCausalLM",
        base(embedding_multiplier=12.0, residual_multiplier=0.22, attention_multiplier=0.0625, logits_scaling=8.0),
    ),
    "granitemoe": (
        "GraniteMoeConfig",
        "GraniteMoeForCausalLM",
        base(
            intermediate_size=HIDDEN,
            num_local_experts=4,
            num_experts_per_tok=2,
            embedding_multiplier=12.0,
            residual_multiplier=0.22,
            attention_multiplier=0.0625,
            logits_scaling=8.0,
        ),
    ),
    "starcoder": (
        "GPTBigCodeConfig",
        "GPTBigCodeForCausalLM",
        dict(vocab_size=VOCAB, n_embd=HIDDEN, n_layer=2, n_head=4, n_inner=64, n_positions=64, multi_query=True),
    ),
    "starcoder2": ("Starcoder2Config", "Starcoder2ForCausalLM", base(sliding_window=None)),
    "phi2": ("PhiConfig", "PhiForCausalLM", base(partial_rotary_factor=0.5)),
    "phi3": ("Phi3Config", "Phi3ForCausalLM", base(pad_token_id=0)),
    "glm4": ("Glm4Config", "Glm4ForCausalLM", base(pad_token_id=0)),
    "stablelm": ("StableLmConfig", "StableLmForCausalLM", legacy(use_qkv_bias=True, partial_rotary_factor=0.5)),
    "olmo2": ("Olmo2Config", "Olmo2ForCausalLM", base()),
    "olmoe": ("OlmoeConfig", "OlmoeForCausalLM", legacy(intermediate_size=HIDDEN, num_experts=4, num_experts_per_tok=2)),
    "falcon": (
        "FalconConfig",
        "FalconForCausalLM",
        dict(
            vocab_size=VOCAB,
            hidden_size=HIDDEN,
            num_hidden_layers=2,
            num_attention_heads=4,
            multi_query=True,
            parallel_attn=True,
            new_decoder_architecture=False,
            bias=False,
        ),
    ),
    "dbrx": (
        "DbrxConfig",
        "DbrxForCausalLM",
        dict(
            vocab_size=VOCAB,
            d_model=HIDDEN,
            n_heads=4,
            n_layers=2,
            max_seq_len=256,
            attn_config=dict(kv_n_heads=2, rope_theta=10000.0, clip_qkv=8.0),
            ffn_config=dict(ffn_hidden_size=HIDDEN, moe_num_experts=4, moe_top_k=2),
        ),
    ),
    "mpt": (
        "MptConfig",
        "MptForCausalLM",
        dict(vocab_size=VOCAB, d_model=HIDDEN, n_heads=4, n_layers=2, expansion_ratio=2, max_seq_len=64),
    ),
    "gemma": ("GemmaConfig", "GemmaForCausalLM", base()),
    "gemma2": (
        "Gemma2Config",
        "Gemma2ForCausalLM",
        base(sliding_window=4, query_pre_attn_scalar=8, layer_types=["sliding_attention", "full_attention"]),
    ),
    "gemma3": (
        "Gemma3TextConfig",
        "Gemma3ForCausalLM",
        base(sliding_window=4, query_pre_attn_scalar=8, layer_types=["sliding_attention", "full_attention"]),
    ),
    "gemma4": (
        "Gemma4TextConfig",
        "Gemma4ForCausalLM",
        base(
            num_hidden_layers=4,
            
            global_head_dim=64,
            sliding_window=4,
            layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
            vocab_size_per_layer_input=VOCAB,
            hidden_size_per_layer_input=32,
            num_kv_shared_layers=2,
            use_double_wide_mlp=True,
            final_logit_softcapping=30.0,
        ),
    ),
    "gemma4_keqv": (
        "Gemma4TextConfig",
        "Gemma4ForCausalLM",
        base(
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            global_head_dim=64,
            num_global_key_value_heads=1,
            attention_k_eq_v=True,
            sliding_window=4,
            layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
            hidden_size_per_layer_input=0,
            final_logit_softcapping=30.0,
        ),
    ),
    "gptoss": (
        "GptOssConfig",
        "GptOssForCausalLM",
        base(
            intermediate_size=HIDDEN,
            
            num_local_experts=4,
            num_experts_per_tok=2,
            sliding_window=4,
            layer_types=["sliding_attention", "full_attention"],
        ),
    ),
    "lfm2": (
        "Lfm2Config",
        "Lfm2ForCausalLM",
        base(
            num_hidden_layers=4,
            block_auto_adjust_ff_dim=False,
            layer_types=["conv", "full_attention", "conv", "full_attention"],
        ),
    ),
    "lfm2moe": (
        "Lfm2MoeConfig",
        "Lfm2MoeForCausalLM",
        base(
            num_hidden_layers=4,
            num_dense_layers=1,
            layer_types=["conv", "full_attention", "conv", "full_attention"],
            **MOE,
        ),
    ),
    "nemotron_h_moe": (
        "NemotronHConfig",
        "NemotronHForCausalLM",
        base(
            num_hidden_layers=4,
            layers_block_type=["mamba", "moe", "attention", "moe"],
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=HIDDEN,
            moe_shared_expert_intermediate_size=HIDDEN,
            mamba_num_heads=4,
            mamba_head_dim=8,
            ssm_state_size=8,
            n_groups=2,
            expand=1,
            chunk_size=8,
            use_mamba_kernels=False,
        ),
    ),
}


def seed_of(name):
    return sum((i + 1) * ord(c) for i, c in enumerate(name))


def randomise(model, seed):
    """Weights large enough that the logits spread: projections scaled by their fan-in, norm
    scales and biases perturbed. Convolutions and the state-space parameters keep their init."""
    g = torch.Generator().manual_seed(seed)

    def normal(t, std):
        return torch.randn(t.shape, generator=g, dtype=t.dtype) * std

    with torch.no_grad():
        for module in model.modules():
            kind = type(module).__name__
            if isinstance(module, torch.nn.Conv1d):
                continue
            for name, p in module.named_parameters(recurse=False):
                if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)) and name == "weight":
                    p.copy_(normal(p, 1.0 / math.sqrt(p.shape[-1])))
                elif "Norm" in kind and name == "weight":
                    p.add_(normal(p, 0.1))
                elif "bias" in name:
                    p.copy_(normal(p, 0.1))
                elif p.ndim >= 2:
                    p.copy_(normal(p, 1.0 / math.sqrt(p.shape[-1])))


def margin(logits):
    top = torch.topk(logits, 2).values
    return float(top[0] - top[1])


def experts_used(config):
    """The number of experts a token is routed to, `None` for a dense model."""
    for key in ("num_experts_per_tok", "top_k_experts"):
        if getattr(config, key, None):
            return getattr(config, key)
    ffn = getattr(config, "ffn_config", None)
    return getattr(ffn, "moe_top_k", None) if ffn is not None else None


def route_gap(model, ids, used):
    """The smallest gap, over every token and router, between the score of the last expert a
    token is routed to and the next one: under the rounding of the weights to 8 bits, a gap
    near zero can pick another expert."""
    gaps = []

    def hook(module, args):
        x = args[0].reshape(-1, args[0].shape[-1]).float()
        scores = x @ module.weight.float().T
        if getattr(module, "bias", None) is not None:
            scores = scores + module.bias.float()
        top = torch.sort(scores, dim=-1, descending=True).values
        if top.shape[-1] > used:
            gaps.append(float((top[:, used - 1] - top[:, used]).min()))

    handles = [
        m.register_forward_pre_hook(hook)
        for n, m in model.named_modules()
        if n.endswith((".gate", ".router", ".router.layer"))
        and getattr(m, "weight", None) is not None
        and m.weight.ndim == 2
    ]
    with torch.no_grad():
        model(torch.tensor([ids]))
    for h in handles:
        h.remove()
    return min(gaps) if gaps else math.inf


def q8(t):
    """`t` rounded as 8-bit blocks of 32 along its last axis, each with its own scale."""
    if t.ndim < 1 or t.shape[-1] % 32:
        return t
    blocks = t.reshape(-1, 32)
    scale = blocks.abs().amax(-1, keepdim=True) / 127
    scale[scale == 0] = 1
    return (torch.round(blocks / scale) * scale).reshape(t.shape)


def rounding_cosine(model, ids, steps):
    """The smallest cosine, over the steps, between the logits of each step and those of a copy
    of the model whose projections, and the inputs of its linear layers, are rounded to 8 bits
    as the loader's block format rounds them."""
    rounded = copy.deepcopy(model)
    # An output head tied to the embedding table is read in full, as the table is.
    tables = {id(m.weight) for m in rounded.modules() if isinstance(m, torch.nn.Embedding)}
    with torch.no_grad():
        for module in rounded.modules():
            if isinstance(module, (torch.nn.Embedding, torch.nn.Conv1d)) or "Norm" in type(module).__name__:
                continue
            if id(getattr(module, "weight", None)) in tables:
                continue
            for _, p in module.named_parameters(recurse=False):
                if p.ndim >= 2:
                    p.copy_(q8(p))
            if isinstance(module, torch.nn.Linear):
                forward = module.forward
                module.forward = lambda x, forward=forward: forward(q8(x))
        cosines = []
        for i, want in enumerate(steps):
            got = rounded(torch.tensor([ids[: len(PROMPT) + i]])).logits[0, -1].float()
            cosines.append(float(torch.nn.functional.cosine_similarity(got, want, dim=0)))
    return min(cosines)


# A seed is kept when every routing decision is clear of a tie by this much, and when the
# logits of the model rounded to 8 bits keep this cosine with its own.
ROUTE_GAP = 0.05
ROUNDING_COSINE = 0.9995
ATTEMPTS = 64


def run(name, attempt):
    config_class, model_class, keys = ARCHS[name]
    seed = seed_of(name) + 7919 * attempt
    torch.manual_seed(seed)
    config = getattr(tf, config_class)(**keys)
    config._attn_implementation = "eager"
    model = getattr(tf, model_class)(config).to(torch.float32).eval()
    randomise(model, seed)
    ids = list(PROMPT)
    steps = []
    nexts, margins = [], []
    with torch.no_grad():
        for _ in range(STEPS):
            logits = model(torch.tensor([ids])).logits[0, -1].float()
            steps.append(logits)
            nexts.append(int(torch.argmax(logits)))
            margins.append(margin(logits))
            ids.append(nexts[-1])
    used = experts_used(config)
    gap = route_gap(model, ids, used) if used else math.inf
    rounding = rounding_cosine(model, ids, steps)
    return model, steps, nexts, margins, gap, rounding


def score(got):
    """How far a seed stays from both limits, in units of each: at least one when both hold."""
    gap, rounding = got[4], got[5]
    return min(gap / ROUTE_GAP, (1 - ROUNDING_COSINE) / max(1 - rounding, 1e-9))


def build(name, out_root):
    """The first seed that holds both limits, else the best of `ATTEMPTS`."""
    best = None
    for attempt in range(ATTEMPTS):
        got = run(name, attempt)
        if best is None or score(got) > score(best):
            best = got
        if score(got) >= 1:
            break
    model, steps, nexts, margins, gap, rounding = best
    out = out_root / name
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp, safe_serialization=True)
        out.mkdir(parents=True, exist_ok=True)
        for f in ("config.json", "model.safetensors"):
            shutil.copy(Path(tmp) / f, out / f)
    reference = {
        "class": ARCHS[name][1],
        "prompt": PROMPT,
        "logits": [float(x) for x in steps[0]],
        "next_ids": nexts,
        "step_logits": [[float(x) for x in step] for step in steps],
        "next_margins": margins,
    }
    if gap != math.inf:
        reference["route_gap"] = gap
    reference["rounding_cosine"] = rounding
    (out / "reference.json").write_text(json.dumps(reference, indent=1) + "\n")
    size = sum(f.stat().st_size for f in out.iterdir())
    print(f"{name}: {size} bytes, next {nexts}, margins {[round(m, 3) for m in margins]}, route gap {gap:.3f}, rounding {rounding:.5f}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out_root = Path(sys.argv[1])
    names = sys.argv[2:] or list(ARCHS)
    for name in names:
        build(name, out_root)


if __name__ == "__main__":
    main()
