# From logits to a token

A sampling policy, chosen per request, turns the logit row into the next id.  
It decides whether 2 runs of the same prompt agree.

## The idea

| Step | What it does |
|---|---|
| **Logits** | The row of numbers, 1 per vocabulary entry. A softmax turns them into probabilities. |
| **Temperature** | Divides the logits first. Below 1 it sharpens the distribution. At 0 it takes the maximum, which is **greedy** decoding. |
| **Top-k** | Keeps the k largest entries. |
| **Top-p** | Keeps the smallest set whose probabilities sum to p. |
| **Min-p** | Keeps every entry above a fraction of the largest. |
| **Repetition penalty** | Scales down the logits of tokens already emitted. |
| **Logit bias** | Adds to chosen entries. |
| **Logprobs** | Report what the row said about the chosen token and its runners-up. |
| **Draw** | One draw, in proportion to what is left. |

**Constrained decoding** is a mask on the row.
A grammar says which tokens can continue a valid output at this position.

A JSON schema compiles to such a grammar.
Every other logit is removed before the draw.

**Determinism** is a property of the whole path.
Greedy decoding on the same logits gives the same token, and the same logits require the same arithmetic in the same order.

Cache reuse ([lesson 6](06-kv-cache.md)) and a batched kernel can both change that order.  
A seed fixes the draw, not the logits.

## What loken does

- loken takes `temperature`, `top_k`, `top_p` and the other sampling options from the request.
- A value the request omits falls back to the model's own default, listed by `/api/show` under `parameters`.
- A `top_p` without a `top_k` is nucleus sampling over the whole vocabulary.
- The repetition penalty applies first, then the candidates are cut, then one draw is made.
- The draw runs on the card when it can stay there.
- A JSON schema or a Lark grammar becomes a per-token mask. `response_format` on the OpenAI routes and `format` on the Ollama routes reach it.
- A generation resumed on another node with the same tokens gives the same continuation, because sampling depends on position and seed alone.

## What was measured

### A greedy decode that did not repeat itself

Setup: 2026-09-12, desktop node, a campaign of 16 cells that published 2 comparisons.

2 cells were blanked.  
On gemma4:31b, 1 engine averaged 96 tokens and the other 128.
A rate over a different length is a different measurement.

The 96 was the mean of 3 iterations that ran 128, 80 and 80 tokens.  
A greedy decode should have produced the same text 3 times.

### Reuse only on a chunk boundary

- A cold run prefills in fixed chunks from position 0.
- A warm run re-prefills the tail from the reused prefix.
- Off a chunk boundary, the tail runs matrix shapes that the cold run never used at those positions.
- The reassociated arithmetic flips a greedy choice wherever the 2 best candidates are close.

The reused prefix is therefore rounded down to a chunk boundary.

The window shift has the same cause: a re-phased key is not the cold key, bit for bit.  
`fidelity = "reference"` turns the shift off ([Status](../STATUS.md#context-shift)).

## Try it

<!-- site: widget steps id=sampling-steps -->
1. Run a greedy request twice.

   ```sh
   for i in 1 2; do curl -s localhost:11435/api/chat -d '{"model":"qwen3:0.6b","stream":false,
     "think":false,"messages":[{"role":"user","content":"Name three colours."}],
     "options":{"temperature":0,"num_predict":40}}' | python3 -c "
   import sys, json; print(repr(json.load(sys.stdin)['message']['content']))"; done
   ```

   Expected: the same string twice, for example `Three colors are: **blue, red, and green**.`

2. Drop `temperature` and run again.  
   The model's own default applies, and the 2 answers may differ.
   `/api/show` lists the default under `parameters`: 0.6 for qwen3.

3. Request a schema.

   ```sh
   curl -s localhost:11435/v1/chat/completions -d '{"model":"qwen3:0.6b",
     "messages":[{"role":"user","content":"Give the capital of France and its population."}],
     "response_format":{"type":"json_schema","json_schema":{"name":"capital","schema":{
       "type":"object","properties":{"city":{"type":"string"},"population":{"type":"integer"}},
       "required":["city","population"],"additionalProperties":false}}}}'
   ```

4. Read `content`.  
   It is a JSON object with exactly 2 keys (observed: `Paris`, `1900000`).  
   `usage.completion_tokens` counts only what the mask let through.
<!-- site: end -->

## Check yourself

<!-- site: widget quiz answer=1 id=sampling-quiz-1 -->
What does `temperature` 0 do?

1. It takes the maximum of the logits, which is greedy decoding
2. It disables the softmax
3. It cuts the candidates to the 0 largest

Temperature divides the logits first. At 0 the draw is a maximum.
<!-- site: end -->

<!-- site: widget quiz answer=3 id=sampling-quiz-2 -->
What does a seed fix?

1. The logits
2. The candidate set
3. The draw

A seed fixes the draw, not the logits. The same logits need the same arithmetic in the same order.
<!-- site: end -->

<!-- site: widget quiz answer=2 id=sampling-quiz-3 -->
Why is the reused prefix rounded down to a chunk boundary?

1. To save memory
2. Off a boundary, the tail runs matrix shapes the cold run never used, and the reassociated arithmetic can flip a greedy choice
3. To keep the snapshot small

A cold run prefills in fixed chunks from position 0. A warm run must use the same shapes to give the same tokens.
<!-- site: end -->

<!-- site: widget flashcards id=sampling-terms -->
| Term | Meaning |
|---|---|
| Logits | The model's output row, 1 number per vocabulary entry. |
| Temperature | Divides the logits before the softmax. 0 is greedy. |
| Top-k | Keeps the k largest entries. |
| Top-p | Keeps the smallest set whose probabilities sum to p. |
| Min-p | Keeps every entry above a fraction of the largest. |
| Repetition penalty | Scales down the logits of tokens already emitted. |
| Constrained decoding | A grammar mask that removes every token that cannot continue a valid output. |
| Determinism | Same tokens from the same logits, which needs the same arithmetic in the same order. |
<!-- site: end -->
