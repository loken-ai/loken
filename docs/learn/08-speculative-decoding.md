# More than one token per weight read

Decode reads the whole model to produce one token and leaves the arithmetic units mostly
idle. Speculative decoding spends that idle arithmetic on checking several guessed tokens in
the pass that would have produced one.

## The idea

A cheap **drafter** proposes k tokens. The target model runs one forward over the prompt
plus those k tokens, which costs about one decode step in bytes since the weights are read
once, and yields the target's own prediction at each of the k positions. The first draft the
target disagrees with is replaced by the target's choice, and everything after it is
dropped. What survives is exactly what the target would have generated on its own, token for
token under greedy decoding, so the output is unchanged.

The gain is accepted drafts per pass; the cost is drafting plus a verification pass slightly
heavier than a plain step. Three sources of drafts:

- a **separate small model** of the same family, with the same vocabulary;
- a **trained head** on the target's own last hidden state (EAGLE), a few percent of a layer,
  with acceptance rates near ninety percent;
- the **prompt itself**: an n-gram that already occurred is continued the way it continued
  before, which costs nothing and works on repetitive text such as code.

It pays where verification is far cheaper per token than generation: a model that
spills to the host crosses the bus once for k tokens instead of once per token. On a model
that fits a card, verification costs about what drafting saves.

## In loken

| File | What it decides |
|---|---|
| `src/inference/serve/eagle.rs` | The EAGLE-1 head on the native substrate: one decoder layer over the target's feature and the next token's embedding, reusing the target's embedding table and output head. The module comment is a complete account of the method. |
| `src/inference/serve/prompt_lookup.rs` | Prompt-lookup drafting from a rolling buffer of prompt and generated tokens. |
| `src/inference/engine/llm_engine/spec.rs` | Drafting with a separate model and verifying in one target pass; the comment states the greedy restriction and what temperature would need. |
| `src/inference/serve/speculative_config.rs` | Whether drafting is worth it, decided from measured device timings rather than assumed. |
| `src/inference/serve/spec_kv_cache.rs` | A cache that can drop the rejected tail. |
| `src/inference/kernel/flash_decode_tc.rs` | The tensor-core decode attention, which packs the verify tokens into the matrix dimension the grouped-query heads already fill. |

The configuration and the attach route are in [`../guides/drafter.md`](../guides/drafter.md).

## What was measured

**One pair, on a spilled target (2026-09-03, desktop node).** deepseek-r1:70b at Q4_K_M, 26
of its 80 layers on the host, with llama3.2:1b drafting: 2.21 tokens a second at 41.0 J a
token against 1.49 and 51.8 alone, the same answer token for token, with 18% of drafts
accepted. A better-matched drafter would accept more
([`../STATUS.md`](../STATUS.md#placement-measured-separately)).

**Two loops, one to go.** The plain stream carries a speculative loop gated per step by a
calibrator; the attach route and a configured drafter drive another. On the same target and
drafter the first gave 1.68 tokens a second and the second 2.21. The measurement says which
should remain ([`../STATUS.md`](../STATUS.md#open)).

**The head, when it was built.** The EAGLE head reached 1.71 to 1.82 times the plain rate on
deepcoder, held out, on an earlier substrate; the figure is the module's, dated by it, and
not re-measured on the native one.

## Try it

This exercise needs a target that spills, which is a 70B on two 16 GB cards. With the pair
above in `config.toml`:

```toml
[inference]
model_id = "deepseek-r1:70b"
draft_model = "llama3.2:1b"
draft_device_index = 0
```

Run one prompt of a few hundred generated tokens with the drafter and without it, and
compare `eval_count / eval_duration` on the two replies. Expected: the same `response` text
both times, and the drafted rate above the plain one by the accepted-draft share. Without
a restart, `POST /api/draft/attach` with `{"model":"llama3.2:1b"}` attaches the drafter to
the resident model and `GET /api/draft/status` lists the pair (an empty `pairs` array means
none is attached).
