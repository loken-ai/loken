# Measuring an engine without fooling yourself

This lesson covers how the rates in the earlier lessons are taken, and the defects that each
measurement in place could not see.

## The idea

- **A rate is a ratio**, and both halves can be wrong. The numerator: which tokens were
  counted (the prompt as the tokenizer sees it, or as a word count estimates it), over what
  span (from the first generated token, or from the request), by whom (the client's clock, or
  the engine's own bookkeeping, which leaves out whatever it does not time). The denominator:
  how many tokens the model actually produced, since a model that stops at 30 tokens is
  averaged over 30 steps and carries the request's fixed cost.
- **The conditions**: a cold first iteration, a machine doing something else, a model that had
  to be evicted to make room for the next, a prompt sent raw where the model expects its
  template. Then the summary: a mean over 3 iterations one of which loaded the model. And
  the question no rate answers: was the output any good, or a loop the gate did not read far
  enough to see.

The discipline that came out of 9 months of this:

- Measure from the client, the same way for every engine.
- Take the median, and report the first request apart from the later ones: cold and warm.
- Publish the token count beside the rate.
- Say which build, which machine, which day.
- Profile a representative slice before a run of hours.
- Let nothing else run.
- Treat a log line as a smell, not as a test.

## What loken does

loken reports the joules of a request, summed from the host and the cards over the request window, and the time spent in each layer.

The measurement side is the protocol and its tools:

- The [method](https://github.com/loken-ai/benchmarks/blob/main/docs/METHOD.md) states the cell, the prompts, which cards each engine may use, what a cell that publishes no rate means, and the `Tokens` column beside the rate.
- The [rate columns](https://github.com/loken-ai/benchmarks/blob/main/docs/METHOD.md#the-rate-columns) explain why prefill is prompt tokens over the client's time to first token, why decode is the client's span, and why both are medians. [Cold and warm](https://github.com/loken-ai/benchmarks/blob/main/docs/METHOD.md#cold-and-warm) figures are kept apart.
- The [benchmarks site](https://loken-ai.github.io/benchmarks/) shows the runs of every machine, one page per hardware, rebuilt from the records.
- [lokprof](https://github.com/loken-ai/lokprof) drives each engine through its own API, separates prefill from decode, and records energy beside speed.

## What was measured

**The engines' own prefill rates (2026-08).** On a 15-token prompt one engine reported
652 tokens a second where its own time to first token implied 35, and this engine reported
49 373 where its own implied 728. Side by side those reversed the verdict on 3 cells of
4. Both columns are now the client's clock.

**Mean against median (2026-09).** A mixture too large to hold measured 3.8, 15.2 and 15.2
tokens a second over 3 iterations, the first being the cold load, and the mean published
11.4 for a model that decodes at 15.2. Energy was worse and flattered this engine: the first
iteration's joules include loading the weights, and one cell published +383.6% where the
steady state is +149.9%.

**A gate that read 120 characters (2026-09-12).** The coherence gate that blanks a looping
answer judged a preview, the first 120 characters of an answer running past 500. One answer
scored 0.139 distinct words over its full text and 0.870 over the preview it was given, so
the campaign blamed the wrong engine. A second, stricter gate written that day read the same
preview and was retracted the same day. The 4 defects of lesson 1 lived below the same
gate for months.

**A sweep under eviction (2026-09-13).** 65 models, 1 question and 1 tool call
each, 1 model resident at a time. 16 clean, 11 catalogue entries that named no
loadable file, 3 reasoning models fixed by reading the generation prompt, and 3 load
errors that looked like deep defects. All 3 were contention: retested cold and alone,
each loaded and answered. A sweep is reliable for a deterministic defect and useless for a
load error.

**Nothing compiles during a cell (2026-09-03).** The same 70B cell gave 2.15 tokens a second,
then 1.64, then 1.58 once a build on 20 threads took half the DRAM bandwidth the host
layers were reading at.

**Profile a slice first (2026-09-17).** 9 machine-hours of full runs on the 552B mixture
found bottlenecks that a 2-layer slice finds in minutes: on 4 layers and 4 096 tokens,
7 changes took 1 batched prefill from 78.7 s to 18.7 s, each measured by a stage timer
before the whole model was run once to confirm.

**What a record identifies (2026-09-12).** A stored result named its engine by the binary's
modification time, its model by a size and a quantisation label, and every digest read as
zeros. The fairness protocol was real but lived in shell, so a stranger's file looked like a
protocol run. Records are now versioned and self-describing, compared only within an
equivalence class (same schema, same protocol profile, same declared flags), and a
submission is quarantined until reproduced.

## Try it

One cell with the bench tool, against the daemon, 3 iterations.

Prerequisite: [`lokprof`](https://github.com/loken-ai/lokprof), the bench tool. Build it once
and call the binary it produces:

```sh
git clone https://github.com/loken-ai/lokprof
cargo build --release --manifest-path lokprof/Cargo.toml   # lokprof/target/release/lokprof
```

Then, with that binary as `lokprof`:

```sh
lokprof --loken http://localhost:11435 --models qwen3:0.6b --num-ctx 4096 \
      --prompts short --stream
```

Observed on qwen3:0.6b, RTX 5070 Ti: prompt 676 tokens a second and completion 724, each
with its spread over `n=3`, 128 tokens generated on every iteration, 0.39 J a token on the
card, and a coherence gate that reads `PASS` above a preview of the answer's first 120
characters. Read 3 things: the rates are medians and the spread is printed beside them,
the token count says how many steps the decode rate averages over, and the preview is the
whole of what the gate judged. Then run the cell again with a build running in another
terminal and compare.
