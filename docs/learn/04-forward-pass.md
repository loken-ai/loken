# One layer, then 40 of them

A transformer is one block repeated. The whole model is that block applied 28 or
92 times, and the next lessons (what is read per token, what is kept between tokens)
fall out of it.

## The idea

An id becomes a vector by a table lookup, the **embedding**, of width `d` (1024 on
qwen3:0.6b). From then on every token is one such vector, and a layer is a function from a
vector to a corrected vector. The correction is added to the input, the **residual**, so a
layer only has to learn what to change.

![One decoder layer](../img/learn-layer.svg)

2 sub-layers, each behind a **norm** (RMSNorm divides by the root-mean-square of the
vector and scales by a learned weight):

- **Attention** lets a token read the others. 3 projections give a query, a key and a
  value per token. The query is scored against every earlier key, the scores go through a
  softmax, and the token's output is the values weighted by those scores, then one more
  projection. Before the scoring, **RoPE** rotates query and key by an angle that grows with
  the position, which is how the model knows distance. Heads run this in parallel on slices
  of the vector. In **grouped-query attention** several query heads share one key and value
  head (16 against 8 on qwen3), which halves what has to be kept per token. **Latent
  attention** (DeepSeek) goes further and keeps one compressed vector per token, decompressed
  when read.
- **Feed-forward** is 2 matrices with a gate in between: `silu(x W_gate) * (x W_up)`,
  then `W_down`. It is where most of a dense model's bytes are. A **mixture of experts**
  replaces it with many such blocks and a router: a small projection scores the experts, the
  top `k` are run (8 of 128, say) and their outputs summed with the router's weights. Only
  those experts' bytes are read for that token.

After the last layer, a final norm and the output projection give one **logit** per
vocabulary entry. Lesson 7 turns that row into a token.

What a layer reads: its weights, once per pass, whatever the number of tokens in the pass.
What it keeps: the key and value of every token it has seen, lesson 6.

## What loken does

- **Architectures**: one decoder implementation serves the dense families (Qwen2 and 3, Gemma,
  Phi and the others). It adapts to the checkpoint: split or fused QKV, q and k norms, extra
  norms, sliding-window layers, a mixture of experts.
- **Positions**: RoPE, with YaRN scaling so a model answers past its trained length.
- **Attention variants**: grouped-query, and latent attention for DeepSeek.
- **Mixtures**: a router picks the top experts per token and only those are read.
- **Ports**: a family with its own design, such as DeepSeek V4.1, has a dedicated port.

## What was measured

**Judging a port without running the original (2026-09-12 to 14).** The DeepSeek V4.1 port
was gated against the authors' own reference implementation, unmodified, run on the CPU: their
kernel module was replaced by a pure-torch one, which works because their linear layer
dispatches on the weight's dtype and bf16 never reaches a compiled kernel.
Each phase of the reference dumps its inputs and outputs, and the port replays the phase on
the same inputs. The gates: attention within 1.3e-2 relative L2,
the mixture within 3.8e-3, the hyper-connections under 5e-3, the fp8 and fp4 dequantisers
bit-exact.

2 pitfalls, both finite and wrong, neither visible as an error: casting the reference's
rotary table to fp32 drops the imaginary half, so RoPE applies half its rotation, and a
lazily evaluated closure bound its loop variable late, so every expert came from the last
layer (1 layer matched, 2 gave cosine 0.92, an hour to find).

**Gating a mixture is not gating a matrix (2026-09-13).** The router's top-k is a discrete
choice, so any precision gap between an f32 port and a bf16 reference flips the occasional
borderline token to another expert. 1 flipped token out of 64 moved the block's relative
L2 from 1e-2 to 6e-2 while every other token agreed and the cosine stayed at 0.998
(attention sub-layer 1.5e-2, stream after the hyper-connection 9.8e-3, block output 5.8e-2:
the jump is one token). A tight global threshold fails correct code. A loose one hides real
bugs. The gate that separates them: cosine over the whole tensor, a bounded count of
tokens over a per-token threshold, and the pre-routing stages gated tightly on their own.

## Try it

Read the shape of a model from its metadata:

```sh
curl -s localhost:11435/api/show -d '{"model":"qwen3:0.6b"}' | python3 -c "
import sys, json
d = json.load(sys.stdin)['model_info']
for k in ('block_count', 'embedding_length', 'attention.head_count',
          'attention.head_count_kv', 'feed_forward_length'):
    print(k, d.get('qwen3.' + k))"
```

Expected: 28 blocks, a width of 1024, 16 query heads against 8 key and value heads, so 2
query heads share each key head. Repeat on a mixture model and look for the expert count keys.
