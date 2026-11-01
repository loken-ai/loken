# 2 regimes, one model

A request runs the same layers in 2 regimes: once over the whole prompt, then once per generated token.  
Prefill is bound by arithmetic. Decode is bound by bytes.

## The idea

![Prefill against decode](../img/learn-prefill-decode.svg)

- **Prefill** takes the n prompt tokens through every layer in one pass.
  Each projection multiplies n rows, so every weight byte read serves n tokens and the card's multiply speed bounds the pass.

  Attention adds a term that grows with n squared, because every token scores against every earlier one.
  At long prompts it takes most of the time.
- **Decode** generates 1 token per pass.
  Each projection multiplies 1 row, so every weight byte read serves 1 token.
  The token rate is close to memory bandwidth divided by model bytes.

  A model on a card reads at the card's rate.  
  A layer on the host reads at the host's rate, an order of magnitude slower.  
  A layer streamed from disk reads at the disk's rate, another order of magnitude slower.
- **Chunks** cut prefill so that its activations fit beside the weights.
  A chunk's peak memory scales with the chunk length times the widest feed-forward.  
  Chunks also let a long prompt yield to other requests.
- **Scheduler**: a decode step reads the weights anyway, so several sequences can share the read.
  At every step the scheduler decides which requests prefill and which decode.  
  It admits new requests when cache blocks free up, and preempts under pressure.

Orders of magnitude on an RTX 5070 Ti:

| Case | Result |
|---|---|
| qwen3:0.6b prefill, 1 782 tokens | 50 ms |
| qwen3:0.6b decode | about 730 tokens a second |
| 70B at Q4, 26 of 80 layers on the host, decode | 1.5 tokens a second, pinned to the host's 36 GB/s |

At 730 tokens a second the 522 MB of qwen3:0.6b weights stream well under the card's bandwidth.
Fixed per-step costs take the rest: launches and the sampling round trip.  
Capturing a decode step as one CUDA graph reduces them.

## What loken does

| Part | Behavior |
|---|---|
| Prefill chunk | Derived from the free VRAM and the widest feed-forward, not a constant. An out-of-memory error in a chunk retries with a smaller one. |
| Scheduler | Plans at every step who prefills and who decodes, under a prefill token budget. Under pressure it preempts a sequence without leaving it half allocated. |
| Decode on the host | A token served from the host allocates nothing. |
| Timers | `/api/layer_perf` and `/api/stage_perf` report per-layer and per-stage times. They are off by default because each stage costs a device synchronisation. |

## What was measured

### Where a long prefill's time goes

Setup: 2026-09-11, RTX 5070 Ti and 5060 Ti, qwen3-coder 30b-a3b, chunk 512, Q4 KV.  
Prefill time was fitted on 3 prompts of real source, non-streamed and cache-busted, to about 1%.

| Build | prefill(n) in ms |
|---|---|
| before | `1.369 n + 1.655e-4 n^2` |
| after | `1.372 n + 2.048e-5 n^2` |

The quadratic term is 8 times cheaper, the linear term unchanged.
A full 131 072-token cold prefill went from 50.4 to 8.9 minutes.

3 changes made the difference:

- Attention runs band by band with a running maximum, instead of holding every score at once.
- 6 passes became 1 fused kernel.
- The whole-context causal mask is gone. It was built on the host and copied to the card on every pass.

Share of time per stage at 10 151 tokens:

| Stage | Share |
|---|---|
| Experts | 76% (gate and up 37.5%, down 36.7%) |
| Attention | 21% |
| Router | 1.6% |

At 44 363 tokens attention takes 43%, being the quadratic half of the cost.

### 3 leads measured and rejected

Same day and setup.

| Lead | Result |
|---|---|
| Hand-written scalar flash-attention kernel | 83.6 s, against 26.8 s for cuBLAS, which has the tensor cores |
| Prefill through the tensor-core expert path, instead of the fused decode-shaped kernel | 7 119 ms, against 6 271 ms |
| Chunk of 2 048 instead of 512 | 19.8 s, against 16.2 s, because the score band shrinks with the chunk |

### A misreported measurement

Until 2026-09-11, the log line of a streamed request reported a word-count estimate of the prompt, not the tokenizer's count.
On source code the estimate is nearly half the real count.
Every streamed measurement read as twice as slow per token.

The usage chunk was always right, and the fit above used it.

## Try it

<!-- site: widget steps id=prefill-decode-steps -->
1. Switch the timers on.

   ```sh
   curl -s 'localhost:11435/api/stage_perf?enable=1' > /dev/null
   ```

2. Run the prompt.

   ```sh
   curl -s localhost:11435/api/generate -d @prompt.json > /dev/null
   ```

3. Read the table.

   ```sh
   curl -s localhost:11435/api/stage_perf | python3 -c "
   import sys, json
   d = json.load(sys.stdin); print(d['layer_calls'], 'layer calls', d['total_us'], 'us')
   for s in d['stages']:
       if s['total_us']: print(f\"{s['stage']:10} {s['share']:.2f}\")"
   ```

4. Switch the timers off.

   ```sh
   curl -s 'localhost:11435/api/stage_perf?enable=0' > /dev/null
   ```
<!-- site: end -->

Observed with a 715-token prompt on qwen3:0.6b:

| Stage | Share |
|---|---|
| Attention | 0.83 |
| Feed-forward, reported under `experts` | 0.13 |
| Norms and residual adds | the rest |

The `prompt_eval_duration` of the reply is a little higher while the timers are on.
That is the synchronisation.  
A mixture fills the `router`, `gate_up` and `down` rows instead.

## Check yourself

<!-- site: widget quiz answer=2 id=prefill-decode-quiz-1 -->
What bounds the speed of decode?

1. The card's multiply speed
2. The speed at which weight bytes arrive
3. The size of the prompt

Each decode projection multiplies 1 row, so every weight byte read serves 1 token. The token rate is close to memory bandwidth divided by model bytes.
<!-- site: end -->

<!-- site: widget quiz answer=3 id=prefill-decode-quiz-2 -->
Which term takes most of a very long prefill?

1. The router
2. The linear term in n
3. The attention term that grows with n squared

1 782 tokens prefill in 50 ms, but at 44 363 tokens attention is 43% of the time.
<!-- site: end -->

<!-- site: widget quiz answer=1 id=prefill-decode-quiz-3 -->
Why are `/api/stage_perf` timers off by default?

1. Each stage costs a device synchronisation
2. They allocate cache blocks
3. They only work on mixtures

The synchronisation makes the numbers real and raises `prompt_eval_duration` while the timers are on.
<!-- site: end -->

<!-- site: widget flashcards id=prefill-decode-terms -->
| Term | Meaning |
|---|---|
| Prefill | One pass of the n prompt tokens through every layer. Bound by arithmetic. |
| Decode | One pass per generated token. Bound by memory bandwidth. |
| Chunk | A slice of the prompt prefilled in 1 pass, sized so activations fit beside the weights. |
| Scheduler | Decides at every step which requests prefill and which decode. |
| CUDA graph | A decode step captured as 1 launch, to cut fixed per-step costs. |
| `/api/stage_perf` | Route reporting the time share per stage. Off until `enable=1`. |
<!-- site: end -->
