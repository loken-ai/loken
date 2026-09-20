# From logits to a token

The model's output is a row of numbers, one per vocabulary entry. Everything that turns
that row into the next id is a policy, chosen per request, and it decides whether two runs
of the same prompt agree.

## The idea

- **Logits to a draw**: the row of **logits** becomes probabilities through a softmax.
  **Temperature** divides the logits first: below one it sharpens the distribution, at zero it
  is a maximum, **greedy**. Then the candidate set is cut: **top-k** keeps the k largest,
  **top-p** keeps the smallest set whose probabilities sum to p, **min-p** keeps every entry
  above a fraction of the largest. A **repetition penalty** scales down the logits of tokens
  already emitted; a **logit bias** adds to chosen entries; **logprobs** report what the row
  said about the token chosen and its runners-up. Then one draw, in proportion to what is left.
- **Constrained decoding**: a mask on the row: a grammar (a JSON schema compiled to one)
  says which tokens can continue a valid output at this position, and every other logit is
  removed before the draw.
- **Determinism**: a property of the whole path. Greedy on the same logits gives the same
  token; the same logits require the same arithmetic, in the same order, which the next
  lesson's cache reuse and a batched kernel can both change. A seed fixes the draw, not the
  logits.

## In loken

| File | What it decides |
|---|---|
| `src/inference/sample/token_sampling.rs` | The two steps: which tokens are candidates, then one draw. The draw is host-side, so the logits come back from the device once per token. |
| `src/inference/engine/llm_engine/gguf.rs` (`build_sampling_from`) and `sampling_tests.rs` | How a request's `temperature`, `top_k`, `top_p` and the model's own defaults combine; the tests read as the specification (a `top_p` without a `top_k` is nucleus sampling over the whole vocabulary). |
| `src/inference/engine/decode_step.rs` | The per-step order: penalty, then sampling, on the shared decode path. |
| `cuda/sampling.cu` | The device-side sampler used when the draw can stay on the card. |
| `src/inference/engine/llm_engine/mod.rs` | The grammar parser factory: a JSON schema or a Lark grammar becomes a per-token mask through llguidance. `response_format` on the OpenAI surface and `format` on the Ollama one reach it. |
| `src/distributed/replay.rs` | Resuming a generation on another node with the same tokens, which holds only if sampling depends on position and seed alone. |
| `scripts/determinism.py` | Runs a prompt repeatedly and diffs the answers. |

## What was measured

**A greedy decode that did not repeat itself (2026-09-12, desktop node).** A campaign of
sixteen cells published two comparisons. Two cells were blanked because one engine averaged
96 tokens against the other's 128 on gemma4:31b, and a rate over a different length is a
different measurement. Ninety-six was the mean of three iterations that ran 128, 80 and 80,
on a greedy decode that should have produced the same text three times.

**Reuse only on a chunk boundary.** A cold run prefills in fixed chunks from position zero;
a warm run re-prefills the tail from the reused prefix. Off a chunk boundary the tail runs
matrix shapes the cold run never used at those positions, and the reassociated arithmetic
flips a greedy choice wherever the two best candidates are close. The reused prefix is
therefore rounded down to a chunk boundary (the comment above `kv_reuse_start` in
`src/inference/engine/llm_engine/load.rs` states the rule), and
[`../STATUS.md`](../STATUS.md#context-shift) keeps the window shift off by default for the
same reason: a re-phased key is not the cold key, bit for bit.

## Try it

Twice, greedy:

```sh
for i in 1 2; do curl -s localhost:11435/api/chat -d '{"model":"qwen3:0.6b","stream":false,
  "think":false,"messages":[{"role":"user","content":"Name three colours."}],
  "options":{"temperature":0,"num_predict":40}}' | python3 -c "
import sys, json; print(repr(json.load(sys.stdin)['message']['content']))"; done
```

Expected: the same string twice (observed: `Three colors are: **blue, red, and green**.`).
Drop `temperature` and the model's own default applies (`/api/show` lists it under
`parameters`: 0.6 for qwen3), and the two answers may differ.

A schema:

```sh
curl -s localhost:11435/v1/chat/completions -d '{"model":"qwen3:0.6b",
  "messages":[{"role":"user","content":"Give the capital of France and its population."}],
  "response_format":{"type":"json_schema","json_schema":{"name":"capital","schema":{
    "type":"object","properties":{"city":{"type":"string"},"population":{"type":"integer"}},
    "required":["city","population"],"additionalProperties":false}}}}'
```

Expected: `content` is a JSON object with exactly those two keys (observed: `Paris`,
`1900000`), and `usage.completion_tokens` counts only what the mask let through.
