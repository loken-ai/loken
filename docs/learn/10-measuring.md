# Measuring an engine without fooling yourself

A rate is a ratio, and both halves can be wrong.
This lesson shows how the rates in the earlier lessons are taken, and which defects each measurement in place could not see.

## The idea

**The numerator** is the token count.

- Which tokens: the prompt as the tokenizer counts it, or as a word count estimates it.
- Over what span: from the first generated token, or from the request.
- Timed by whom: the client's clock, or the engine's own bookkeeping, which leaves out whatever it does not time.

**The denominator** is the tokens the model produced.
A model that stops at 30 tokens is averaged over 30 steps and carries the request's fixed cost.

**The conditions** also change a rate:

- a cold first iteration
- a machine doing something else
- a model evicted to make room for the next
- a prompt sent raw where the model expects its template
- a mean over 3 iterations, 1 of which loaded the model

No rate says whether the output was good, or a loop the gate did not read far enough to see.

The discipline that came out of 9 months of measuring:

- Measure from the client, the same way for every engine.
- Take the median. Report the first request apart from the later ones: cold and warm.
- Publish the token count beside the rate.
- Say which build, which machine, which day.
- Profile a representative slice before a run of hours.
- Let nothing else run.
- Treat a log line as a smell, not a test.

## What loken does

loken reports the joules of a request, summed from the host and the cards over the request window.
It also reports the time spent in each layer.

The protocol and its tools do the rest:

- The [method](https://github.com/loken-ai/benchmarks/blob/main/docs/METHOD.md) defines the cell, the prompts, the cards each engine may use, what a cell with no published rate means, and the `Tokens` column beside the rate.
- The [rate columns](https://github.com/loken-ai/benchmarks/blob/main/docs/METHOD.md#the-rate-columns) define prefill as prompt tokens over the client's time to first token, and decode as the client's span. Both are medians. [Cold and warm](https://github.com/loken-ai/benchmarks/blob/main/docs/METHOD.md#cold-and-warm) figures stay apart.
- The [benchmarks site](https://loken-ai.github.io/benchmarks/) shows the runs of every machine, 1 page per hardware, rebuilt from the records.
- [lokprof](https://github.com/loken-ai/lokprof) drives each engine through its own API, separates prefill from decode, and records energy beside speed.

## What was measured

### The engines' own prefill rates

Date: 2026-08.
On a 15-token prompt, 1 engine reported 652 tokens/s where its own time to first token implied 35.
Another reported 49 373 where its own implied 728.
Side by side, those figures reversed the verdict on 3 cells of 4.
Both columns now use the client's clock.

### Mean against median

Date: 2026-09.
A mixture too large to hold decoded at 3.8, 15.2 and 15.2 tokens/s over 3 iterations.
The first iteration was the cold load.
The published mean was 11.4 for a model that decodes at 15.2.

| Quantity | Published | Steady state |
|---|---|---|
| Decode rate, tokens/s | 11.4 | 15.2 |
| Energy against the other engine | +383.6% | +149.9% |

The energy figure flattered this engine: the first iteration's joules include loading the weights.

### A gate that read 120 characters

Date: 2026-09-12.
The coherence gate that blanks a looping answer judged a preview, the first 120 characters of an answer running past 500.
One answer scored 0.139 distinct words over its full text and 0.870 over the preview, so the campaign blamed the wrong engine.
A second, stricter gate written that day read the same preview and was retracted the same day.
The 4 defects of lesson 1 lived below the same gate for months.

### A sweep under eviction

Date: 2026-09-13.
The sweep covered 65 models, with 1 question and 1 tool call each and 1 model resident at a time.

| Outcome | Models |
|---|---|
| Clean | 16 |
| Catalogue entries naming no loadable file | 11 |
| Reasoning models fixed by reading the generation prompt | 3 |
| Load errors that looked like deep defects | 3 |

All 3 load errors were contention.
Retested cold and alone, each loaded and answered.
A sweep is reliable for a deterministic defect and useless for a load error.

### Nothing compiles during a cell

Date: 2026-09-03.
The same 70B cell gave 2.15, then 1.64, then 1.58 tokens/s.
From the second run, a build on 20 threads took half the DRAM bandwidth that the host layers were reading at.

<!-- site: widget chart type=bar id=measuring-build -->
| Run | Tokens per second |
|---|---|
| Quiet machine | 2.15 |
| Build running, run 2 | 1.64 |
| Build running, run 3 | 1.58 |
<!-- site: end -->

### Profile a slice first

Date: 2026-09-17.
9 machine-hours of full runs on the 552B mixture found bottlenecks that a 2-layer slice finds in minutes.
On 4 layers and 4 096 tokens, 7 changes took 1 batched prefill from 78.7 s to 18.7 s.
A stage timer measured each change.
The whole model then ran once to confirm.

### What a record identifies

Date: 2026-09-12.
A stored result named its engine by the binary's modification time and its model by a size and a quantisation label.
Every digest read as zeros.
The fairness protocol lived in shell, so a stranger's file looked like a protocol run.

Records are now versioned and self-describing.
They are compared only within an equivalence class: same schema, same protocol profile, same declared flags.
A submission is quarantined until reproduced.

## Try it

This exercise runs 1 cell with the bench tool against the daemon, over 3 iterations.

<!-- site: widget checklist id=measuring-prereq -->
- [ ] [`lokprof`](https://github.com/loken-ai/lokprof) is cloned and built.
- [ ] The loken daemon answers on `http://localhost:11435`.
- [ ] qwen3:0.6b is in the store.
- [ ] Nothing else runs on the machine.
<!-- site: end -->

<!-- site: widget steps id=measuring-try -->
1. Build `lokprof` once:

   ```sh
   git clone https://github.com/loken-ai/lokprof
   cargo build --release --manifest-path lokprof/Cargo.toml   # lokprof/target/release/lokprof
   ```

2. Run the cell with that binary as `lokprof`:

   ```sh
   lokprof --loken http://localhost:11435 --models qwen3:0.6b --num-ctx 4096 \
         --prompts short --stream
   ```

3. Read 3 things in the output: the rates are medians with their spread beside them, the token count says how many steps the decode rate averages over, and the preview is the whole of what the gate judged.
4. Run the cell again with a build running in another terminal and compare.
<!-- site: end -->

Observed on qwen3:0.6b, RTX 5070 Ti:

| Quantity | Value |
|---|---|
| Prompt rate | 676 tokens/s |
| Completion rate | 724 tokens/s |
| Spread | over `n=3` |
| Tokens generated per iteration | 128 |
| Energy | 0.39 J/token on the card |
| Coherence gate | `PASS`, over a preview of the answer's first 120 characters |

## Check yourself

<!-- site: widget quiz answer=2 id=measuring-quiz-1 -->
Why does a mean over 3 iterations mislead when the first one loaded the model?

1. The mean ignores the token count.
2. The cold load drags the mean below the steady rate.
3. The median is always lower.

The mean published 11.4 tokens/s for a model that decodes at 15.2.
The median ignores the cold iteration, and cold and warm are reported apart.
<!-- site: end -->

<!-- site: widget quiz answer=3 id=measuring-quiz-2 -->
Whose clock times a rate in the benchmarks?

1. The engine's own bookkeeping.
2. The tokenizer's.
3. The client's, the same way for every engine.

Engine-reported prefill rates reversed the verdict on 3 cells of 4.
<!-- site: end -->

<!-- site: widget quiz answer=1 id=measuring-quiz-3 -->
What should a 3-model load error in a sweep make you do?

1. Retest cold and alone.
2. Report a deep defect.
3. Raise the iteration count.

All 3 errors were contention, and each model loaded when retested alone.
<!-- site: end -->

<!-- site: widget flashcards id=measuring-terms -->
| Term | Meaning |
|---|---|
| Cold and warm | The first request after a load, and the later ones. Reported apart. |
| Median | The middle value of the iterations, which a cold first run does not drag. |
| Coherence gate | A check that blanks a looping answer. |
| Equivalence class | Same schema, same protocol profile and same declared flags. Records compare only within one. |
| lokprof | The bench tool that drives each engine through its API and records energy beside speed. |
<!-- site: end -->
