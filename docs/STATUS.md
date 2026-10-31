# Status

What is measured, what is slower, what has never run. Revised 2026-11-01.

Every figure comes from 1 machine: RTX 5070 Ti + RTX 5060 Ti (16 GB each), Linux, CUDA 13.3, models on a USB disk reading at 460 MB/s.

## Decode rate vs ollama

The [benchmarks](https://github.com/loken-ai/benchmarks) repository holds the decode rates against ollama and vLLM, per card, per backend and per context size.
[Its site](https://loken-ai.github.io/benchmarks/) publishes them.
How a cell is measured is in its [method](https://github.com/loken-ai/benchmarks/blob/main/docs/METHOD.md).

The figures quoted below come from the campaign of 2026-10-03 on the machine named above:

- Both cards available to every engine.
- Idle machine.
- Greedy, streamed, short prompt, 4096 ctx.
- Median of 3 iterations.

2 mixtures lost in that campaign.
Their harness had moved into its own repository and started the engine without its configuration.
Measured configured on 2026-09-02, olmoe is +17.4% and granite3-moe +0.2%.

The mixtures that do lose are the ones too large for the cards.
The placement used to place a layer whole.
A spilled mixture then ran attention on the host for the sake of expert weights, which are 97% of its bytes and are read a few at a time.
Spilling only the experts and keeping everything else on the cards changed the rates:

| Model | Before | After | ollama |
|---|---:|---:|---:|
| qwen3next | 15.6 tok/s | 34.7 tok/s | 33 tok/s |
| qwen3-coder-next | 10.7 tok/s | 30.8 tok/s | 26 to 28 tok/s |

A mixture that already fits is untouched.

## Placement, measured separately

Given both cards, ollama splits models that fit on 1 card.
loken keeps them on the fast card and pays nothing for the second card being present.
Measured 2026-09-03 on olmoe:

| Setting | Rate |
|---|---:|
| Card 0 alone | 464 tok/s |
| Both cards visible | 462 tok/s |
| Card 1 alone | 252 tok/s |

The placer picks the fast card.
An earlier revision of this page reported a 390 against 496 penalty, and it is gone.

A model that fits on neither card is limited by the bus.
deepseek-r1:70b at Q4_K_M holds 26 of its 80 layers on the host.
The host reads each of them in full per token at ~36 GB/s, this machine's DRAM ceiling.

| Run | loken | ollama |
|---|---:|---:|
| 2026-09, host path unaided | 1.6 tok/s | 1.5 tok/s |
| 2026-10-03 campaign | 3.1 tok/s | 1.6 tok/s |
| 2026-09, `llama3.2:1b` drafter | 2.0 tok/s | |

In the 2026-10-03 campaign, loken picks the drafter itself for a model that spills.
The drafter proposes tokens and the 70B verifies them.

A variant that fits is the other way out.
The FFN-at-Q3 requantisations (`deepseek-r1:70b-ffnq3`, `-ffnq3d2`, `-q3ks`) are among the measured runs.

A checkpoint that ships split is read as 1 model across its parts.
unsloth/Qwen3-235B-A22B-GGUF, a mixture of 235B parameters, runs on the 2 cards and the host at 7.0 tok/s prefill and 4.7 tok/s decode (4096 ctx, short prompt, streamed).

## Context shift

A prompt longer than the window is refused on every route.
ollama would cut it, and a cut prompt can lose its system prompt or its tool instructions and still read as a plausible answer.

Before refusing, loken reloads the model with a wider window when the model and the request's `num_ctx` allow it.

A prompt that fits but leaves too little room for the answer keeps its first token and its tail.
Once a conversation reaches that point, the prefix it shares with the resident KV is that first token alone.
A cold run then re-prefills the whole window every turn.

loken keeps the tail instead:

- It pulls the resident keys down to their new positions.
- It re-phases them in place. The rotary phase of the shift is read off the layer's own table, so YaRN and factored frequencies need nothing special.
- It prefills only the new tokens.

Measured 2026-09-04, on the 2nd turn of a conversation 500 tokens past the window:

| Model | Window | Prefill cold | Prefill with shift | Dropped | Reused |
|---|---:|---:|---:|---:|---:|
| qwen3:8b | 4,096 | 1,256 ms | 156 ms | 508 | 3,519 |
| deepseek-r1:70b, 26 layers on the host | 2,048 | 153 s | 38 s | 430 | 1,554 |

The answer is coherent both ways, and the text differs.
A re-phased key is not the key a cold prefill would have produced, bit for bit.

The shift runs by itself wherever the cache supports it.
It falls back to a cold prefill in 3 cases:

| Case | Reason |
|---|---|
| Q4 KV cache | A block groups 32 positions of 1 channel |
| Sliding-window layers | The buffer index stops being the position after a slide |
| Recurrent hybrids | The state holds no positions to shift |

## Large mixtures streamed from disk

DeepSeek V4.1 is served from its own GGUF, which may be split.
The routed experts are read from the file as routing asks for them.
Its runs are on the [benchmarks site](https://loken-ai.github.io/benchmarks/).
The other engines could not run it.

<!-- site: widget table id=status-streamed-pieces -->
| Piece | State |
|---|---|
| Prefill on the cards | A prompt's attention, residual stream, expert products and routing run on the cards. Each layer reads only the experts the prompt routes to, in runs, into a page-locked slab, and sends them on a copy stream. See the benchmark rows of the model |
| Read-ahead, reader threads, expert tier | Derived from measurement at load and while decoding, see [Configuration](CONFIGURATION.md#streamed-mixtures-deepseek-v41) |
| Decode-state cache | In host memory, or on disk when `state_cache_dir` is set. A stored prefix always pays. A short remainder still reads nearly every expert, which bounds how little it costs |
| `quality` | `exact` unless `fidelity = "fast"`. The other levels change the output. The budget is derived from the model's own noise and chosen against a teacher-forced run of a bundled text, not against a task score |
| Sharding across nodes | Unmeasured at 2 nodes, see [The cluster](#the-cluster) |
<!-- site: end -->

A transformer with recurrent layers (a convolution, state-space or delta-rule mixer beside attention) cannot trim its state back to a shorter prefix.
Its cache therefore stores the states its prefill passes.
It restores the longest one that a prompt extends, in host memory or, with `state_cache_dir` set, on disk.
No row of the benchmark tables measures that cache.

The quality harness scores a setting against the exact run on held-out text.
It reports 3 figures:

- Top-1 agreement.
- Divergence over the exact run's top 20.
- The log-probability given to the exact choice.

It runs by hand on a real checkpoint.

## Audio in

Voxtral and Ultravox towers are read from an `mmproj` GGUF or a checkpoint directory.
Every dimension is taken from the file, and the tower output is spliced into a chat turn.
The mel features and the towers match the reference implementation at rounding level on toy towers.

An Ultravox checkpoint serves alone.
Its text model is the safetensors checkpoint its config names, found in the hub cache or pulled with it.

A Voxtral model is prompted in the bracket instruction format, whatever template its file carries.
The published GGUF holds another model's template, whose default system prompt made the model answer the clip instead of transcribing it.

Real checkpoints were measured against the reference implementation (transformers 4.57.1, F32) on 2026-10-19:

- The input is the 20 clips of the LibriSpeech set kept in the benchmarks repository (183 s).
- Each clip is sent as 1 chat turn of an audio part and a text part, greedy, at most 256 tokens.
- Both sides run on the CPU of an i9-10900K shared with other jobs, so the real-time factor (wall time over clip time) is indicative.
- Words are lowercased with punctuation removed on both sides.
- "On task" leaves out the clips where either side answers or comments instead of transcribing (errors at least the clip's word count).

<!-- site: widget table id=status-audio-in -->
| Model | Prompt | Side | WER | Errors per clip | WER on task | Clips on task | RTF |
|---|---|---|---:|---:|---:|---:|---:|
| Voxtral Mini 3B | Transcribe this audio | reference | 20.4% | 5.05 | 7.4% | 18 | 3.96 |
| Voxtral Mini 3B | Transcribe this audio | loken | 33.8% | 8.35 | 1.6% | 18 | 5.82 |
| Ultravox 0.5 1B | Repeat exactly what the speaker says. | reference | 22.9% | 5.65 | 2.6% | 16 | 1.64 |
| Ultravox 0.5 1B | Repeat exactly what the speaker says. | loken | 41.1% | 10.15 | 3.6% | 16 | 1.33 |
| Ultravox 0.5 1B | Transcribe this audio | reference | 356% | 88.0 | - | 1 | 6.74 |
| Ultravox 0.5 1B | Transcribe this audio | loken | 370% | 91.4 | - | 1 | 2.77 |
<!-- site: end -->

On the clips that both sides transcribe, loken is within a quarter of a word per clip of the reference, or better.

The whole-set gap comes from 1 clip per model.
On that clip both sides answer instead of transcribing, and loken's answer is longer:

| Model | Clip | Errors, loken | Errors, reference |
|---|---|---:|---:|
| Voxtral | 2830-3979-0000 | 159 | 13 |
| Ultravox | 1995-1826-0000 | 141 | 27 |

loken runs the published Q4_K_M Voxtral file and holds the Ultravox text model in Q4_K.
The reference runs both in F32.
An off-task answer therefore takes another course.

Ultravox answers "Transcribe this audio" with commentary on nearly every clip, on both sides.
That prompt leaves nothing to compare but the length of the commentary.

A Voxtral request spends 29 to 74 s (median 36 s) between its arrival and its prefill.
In that interval the clip is decoded and the speech tower runs on its 30 s window.
It is most of loken's time per clip.

## OpenCL

The layers of a language model run on an Intel GPU through Intel's OpenCL runtime, and give the host's text (qwen3 0.6b, greedy).
On the laptop's processor they are slower than the host: 51 tokens per second against 74.

Devices of Mesa's OpenCL platform are not offered for layers, because the layers measured wrong on them.
The image models' OpenCL block still differs from the host's by 15% on a block, and is not shipped.

## Forecasting

The forecast route serves Chronos-2 and the 4 Timestral checkpoints.
The fixed set of the Timestral gate holds 8 series with their held-out continuation.
On that set, the mean MASE of the median forecast is:

| Forecaster | Mean MASE |
|---|---:|
| Chronos-2 | 0.85 |
| Timestral checkpoints | 1.15 to 1.29 |
| Seasonal naive | 2.71 |

2 tests fail when a model stops beating the seasonal naive forecast.

## The qwen4exp family

3 files of the architecture load through the generic route on the 2-card machine:

- A 160-expert Q4_K_M (28.4 GB).
- A 256-expert IQ1_M in 2 parts (58.4 GB).
- A 512-expert UD-Q4_K_XL in 4 parts (111 GB).

Speed is in the runs of the [benchmarks site](https://loken-ai.github.io/benchmarks/).
This page quotes none.

<!-- site: widget table id=status-qwen4exp-state -->
| What | State |
|---|---|
| Card against host numerics | The card paths agree with the host to a cosine of 0.9997 per layer on the 160-expert file |
| Low-bit expert formats | Run on the card in decode and prefill, judged against the host decoder |
| Expert cache | Hot experts are held on each layer's card. A missing expert is copied into the card's transit room or computed on the host, split by measured rates. A wide batch streams a layer's experts through 2 buffers when that costs less than its rows. A card with room for all its experts holds them and runs its layers as loaded ones |
| 256 experts, decode | On a 4.0k token prompt, 1 card decodes 21 to 23 tok/s and 2 cards 30 to 33. On 14k tokens, 20 and 28 to 30. Before the cache plan used the second card: 16 to 17 on 4.0k and 12 on 14k |
| Decode graph | Captured for files without an n-gram table or indexed attention. The others, and every file under the cache plan, decode eagerly |
| 160 experts, 1 card | Measured on 3.2k and 12.9k token prompts |
| 160 experts, 2 cards | Measured on a 4.5k token prompt |
| 256 experts, 1 card | Answers a 72-token prompt coherently. A 3.2k token prefill ends the first request after 1 token, and a second request fails out of memory |
| 256 experts, 2 cards | Measured on a 72-token prompt before the cache plan. The 3.2k-token prompt fails as on 1 card |
| 512 experts | Loads. Not measured on either envelope |
<!-- site: end -->

Quality is the mean negative log-likelihood per token (nats) of held-out texts from this repository, scored by the quality harness.
The code text holds 1500 tokens and the docs text 2200.
The chunk is 512, on 2 cards.

<!-- site: widget chart type=bar id=status-qwen4exp-nll -->
| File | Code | Docs |
|---|---:|---:|
| 160 experts, Q4_K_M | 2.18 | 3.37 |
| 256 experts, IQ1_M | 1.23 | 1.81 |
| 512 experts, UD-Q4_K_XL | 1.03 | 1.76 |
<!-- site: end -->

For the 160-expert file, the placements differ by at most 0.01 nat:

| Placement | Code | Docs |
|---|---:|---:|
| Host only | 2.176 | 3.359 |
| 2 cards | 2.181 | 3.370 |

Host against cards, top-1 agreement is 0.94 on code and 0.92 on docs, at a mean KL of 0.017 and 0.021.
Each cell is 1 run.

Limits:

- **Out of memory in prefill.** The 256-expert file on a 3.2k token prompt fails with a card reported nearly full (27 to 59 MB free), while the chunk accounting counted several GB free. The engine halves the chunk, and the request ends with 1 token or an error. The cause is unknown. A 2.7k token prompt completes.
- **Prefill is compute-bound** in the delta-rule projections, convolution, scan and the wide mixers. The expert copy is hidden behind it.
- **The first 128 decoded tokens run while the expert cache learns.** A decode rate over fewer tokens understates the steady state.
- **The 512-expert file** holds 77 GB of experts against 56 GB of host memory in the scope measured, and that case is untried. Only 3 quarters of the memory the kernel still has free is page-locked. The rest copies through the driver's staging at a lower rate.
- **2 cards** under the cache plan cut the layers by link. The card on the slower link (x4 here) holds the layers it can hold whole, and the other card holds the rest with a cache. The decode of a card with a cache is bound by the host launching its kernels 1 by 1, since the decode graph is off for these files.
- **Multi-token prediction** runs as a draft head that `loken draft-head` writes beside the model from the checkpoint's `mtp.*` tensors. The decode drafts with it only when `fidelity = "fast"` is set and the decode graph is recorded.
- A test of the q4 KV decode parity fails now and then when GPU tests run in parallel, and passes alone.

## Costs

- **~870 MiB per load, any model.** CUDA contexts, cuBLAS workspace and preloaded module images take it. It is fixed, not proportional: 893 and 867 MiB on models that differ 2x in weights. The placement planner does not know about it.
- **The first answer after a load is slower.** A mixture pays ~0.5 s when a request arrives before the background repack finishes, if it fits. A spilled mixture pays far more: qwen3next decodes its first request at 2.0 tok/s and the next at 34.7. The benchmark table takes the median over its iterations, so that cost no longer reaches a published rate. A user's first answer still pays it.
- **Load time is the disk.** 24 GB over USB: 52 s reading, 6 s to the cards.

### Host memory of a spilled model

A spilled model used to be held twice in host RAM.
The file was prefetched whole before placement.
The 28 GB already on the cards stayed cached beside the 14 GB the host reads, and a 64 GB box swapped 16 GB during the decode being timed.

The advice now follows the plan: `DontNeed` for what the cards hold, `WillNeed` for what the host reads.
Each layer's pages are dropped the moment the layer lands on a card.
Waiting for the end of the load was too late: the cache had reached 44 GB and 12 to 15 GB of the process had gone to zram.
The same cell then decoded anywhere between 1.5 and 2.2 tok/s from 1 run to the next.
Dropped per layer, the cell measures 2.25 tok/s, 3 times.

Those figures predate 2 changes:

- A weight the loader built (fused q/k/v and gate/up, a rebuilt float projection) and that now sits on a card keeps no host copy.
- The token tables stay as the file stores them instead of widening to F32.

The load logs 1 ledger line with every conversion it made and its bytes before and after.
It warns when the cards grow by more than the file.
The load still reads the pages of host layers once their repack exists.
No figure here measures the peak after the changes.

## The cluster

2 halves live in [Cluster](CLUSTER.md), and only 1 of them runs.

### Replicated serving

Replicated serving runs.
Any node is an entry point, and an entry point holds nothing.
A client talks to whichever node it knows.
That node forwards the whole HTTP request to the best holder and relays the stream back.

Membership is SWIM gossip with a phi-accrual detector.
Multicast announcement or a `join` list seeds it.
`/api/cluster/state`, `/api/cluster/peers` and `/api/cluster/prefix` report it.
There is no consensus protocol: routing to a dead node costs a retry, not a corruption.

A hand-over is priced on rates that each node measures from its own completed generations.
Hardware nameplates play no part, because what a card could do is not what this build achieves on this model at this quantisation.

### Sharding for DeepSeek V4.1

Sharding runs for DeepSeek V4.1, unmeasured at 2 nodes.
At load, each node measures its own profile.
The planner compares a split against local serving.
When a split pays, the head forms a pipeline of any length over the binary data plane.
A 3-node loopback chain decodes bit-identically to 1 node.

### Sharding for dense transformers

Sharding runs for dense generic transformers, measured at 2 nodes on one machine.

1. Each clustered node times a layer on its cards and on its host.
2. Each node publishes that time with the bytes its cards carry of the model.
3. The node that loads the model 2nd plans the cut from those figures and the gossip round trip.
4. It asks the other node to reload its layers through its own placement ladder, and reloads its own the same way.
5. It sends the boundary of every prefill chunk and decode step over the binary frames.

Each node keeps the KV cache of its layers.
The tail node gives up its own copy of the model while it serves a segment.
The gossip round trip alone prices the crossing, not the logits that come back.

Measured on 2 loopback servers, 1 card each, with DeepSeek R1 32B (19.9 GB, 64 layers, does not fit 1 16 GB card):

<!-- site: widget table id=status-cluster-r1-32b -->
| Setup | Context | Prefill | Decode |
|---|---|---:|---:|
| 2 nodes, pipeline | 2k | 1170 tok/s | 26 tok/s |
| 2 nodes, pipeline | 6k | 1060 tok/s | 21 tok/s |
| 1 card, 21 layers on the host | not stated | 43 tok/s | 3.3 tok/s |
| 2 cards of 1 node, tensor-parallel | not stated | 711 tok/s | 27 tok/s |
<!-- site: end -->

Greedy output matches the 1-card node for 64 tokens on 2 prompts, and for a 6k prompt with its continuation.
The 1-card node itself varies between runs past about 25 tokens.
In the tests, a 2-way and a 4-way split are bit-identical to 1 part on the host.

Sharding refuses a model, with the reason, when its layers:

- Take per-layer inputs.
- Share KV.
- Keep a recurrent state.
- Hold experts in host memory.
- Carry an adapter or an image tower.

Verify batches are refused too.

The link-cost matrix, the tiered KV, the resume path and the layer scheduler are reached by no code path.

### Capacity defect

A node published a capacity derived from 1 generation's decode rate multiplied by its lane count.
Its generations serialise behind the model lock.
2 comparable machines therefore priced each other an order of magnitude apart.
The cluster handed over a few requests where it should have split the work roughly in half.

The meter now publishes what a window of real completions produced.
The 2nd machine left the bench before the fix could be measured, so **the 2-node gain is unmeasured**.
Trace replay judges the correction, not a cluster.

### Known defects

- A peer that has never answered is priced from a floor rather than from its catalogue. A cold node therefore looks worse than an idle one that holds nothing.
- A replica set keyed on a placeholder digest, which models cached from Hugging Face carry, groups every such model into 1 set.

## Written, wired to nothing

- **Layer scheduler:** no code calls it, since the engine that used to build a plan by hand was removed. It was judged over 21 placement and 119 naming cases, with 2 perturbations that prove the judge can fail.
- **Reserve pass:** runs the forward on a device that allocates nothing, to replace the per-model formulas. The Z-Image, FLUX and other image loaders call it. Every language-model placement is budgeted by formula.

## Shipped as source, never compiled

The CUDA kernels carry AMD arms: MFMA on CDNA and WMMA on RDNA3+, guarded and complete.
No hipcc build ran on the measurement machine, so none has been through a compiler.
Treat AMD as source, not support.

## Judged partially

- **Mat-vec core:** checked against an independent reference for 2 formats of 10. The production entry point does not reach the core.
- **Z-Image bends its schedule twice.** `new` derives timesteps from bent sigmas, and `set_timesteps` from the pre-bend ramp. At 9 steps the first gives `[1.0, 0.960129, 0.913349, ...]` and the second `[1.0, 0.96, 0.913043, ...]`, for 1 bend. Only a render says which the checkpoint expects.
- **Image parity is a hash**, one prompt and seed per family, and a property of the placement as well as the code. The placer reads free VRAM at load, and 4 runs of 1 binary gave 4 hashes. Run it on a quiet machine.

## Measured and refused

A 3-bit KV cache through a randomised Hadamard rotation.
The rotation is orthogonal and does spread outliers: excess kurtosis falls from 35.7 to -0.6.
Against f16 it delivers what it promises.

Against the Q4_0 cache already in place it does not.
Token-axis grouping removes the outliers first, so rotating changes the error by under 0.5%.
Q4_0 is 2.6x more accurate for 1 more bit.
The harness stays in the code, wired to nothing.

## Borrowed code

1 file shares body lines with upstream, at 58%, recorded in [NOTICE](../NOTICE.md).
Its largest matched range is the GGUF k-quant bit specification, which any correct reader reproduces.

## Other measurements

- The 870 MiB per load is unbudgeted, and the reserve pass does not reach language models.
- olmoe's deficit is closed. The harness caused it, not the engine.
- granite3-moe:1b loses 1 to 5% streamed and wins 7 to 20% not streamed, over 6 runs across 3 days. At ~3 ms a token, the stream's per-token cost is of the same order as the token. That cost is the chunk, its JSON and the client's read loop. The gap lives there, not in the kernels.
- 1 speculative loop learns its draft length from the cycles. On deepseek-r1:70b spilling to the host, drafted for by llama3.2:1b, greedy, it decodes 1.49 tok/s alone and 2.72 drafted, with the same text.
