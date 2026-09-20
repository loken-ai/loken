# Measuring an engine without fooling yourself

This lesson covers how the rates in the earlier lessons are taken, and the defects that each
measurement in place could not see.

## The idea

- **A rate is a ratio**, and both halves can be wrong. The numerator: which tokens were
  counted (the prompt as the tokenizer sees it, or as a word count estimates it), over what
  span (from the first generated token, or from the request), by whom (the client's clock, or
  the engine's own bookkeeping, which leaves out whatever it does not time). The denominator:
  how many tokens the model actually produced, since a model that stops at thirty tokens is
  averaged over thirty steps and carries the request's fixed cost.
- **The conditions**: a cold first iteration, a machine doing something else, a model that had
  to be evicted to make room for the next, a prompt sent raw where the model expects its
  template. Then the summary: a mean over three iterations one of which loaded the model. And
  the question no rate answers: was the output any good, or a loop the gate did not read far
  enough to see.

The discipline that came out of nine months of this:

- Measure from the client, the same way for every engine.
- Take the median.
- Publish the token count beside the rate.
- Say which build, which machine, which day.
- Profile a representative slice before a run of hours.
- Let nothing else run.
- Treat a log line as a smell, not as a test.

## In loken

| File | What it decides |
|---|---|
| [`../BENCHMARKS.md`](../BENCHMARKS.md#method) | The protocol: the cell, the prompts, which cards each engine may use, what a cell that publishes no rate means, the `Tokens` column beside the rate. |
| [`../BENCHMARKS.md`](../BENCHMARKS.md#the-two-rate-columns) | Why prefill is prompt tokens over the client's time to first token, why decode is the client's span, and why both are medians. |
| `scripts/campaign.sh`, `scripts/bench_row.py`, `scripts/scoreboard.py` | The harness: refuses a stale binary, stamps the build into every row, writes the rows into the page. |
| `scripts/figures.py` | Regenerates the table in `STATUS.md` and the figures from one parse of `BENCHMARKS.md`, because the two pages once disagreed. |
| `scripts/determinism.py` | The same prompt repeated, and a diff of the answers. |
| `src/inference/place/layer_perf.rs`, `src/energy.rs` | The per-layer timers of lesson 5, and the joules: host RAPL and card NVML summed over the request window. |
| [assay](https://github.com/loken-ai/assay) | The bench tool: drives each engine through its own API, separates prefill from decode, records energy beside speed, makes the engines comparable before comparing them. |

## What was measured

**The engines' own prefill rates (2026-08).** On a 15-token prompt one engine reported
652 tokens a second where its own time to first token implied 35, and this engine reported
49 373 where its own implied 728. Side by side those reversed the verdict on three cells of
four. Both columns are now the client's clock.

**Mean against median (2026-09).** A mixture too large to hold measured 3.8, 15.2 and 15.2
tokens a second over three iterations, the first being the cold load, and the mean published
11.4 for a model that decodes at 15.2. Energy was worse and flattered this engine: the first
iteration's joules include loading the weights, and one cell published +383.6% where the
steady state is +149.9%.

**A gate that read 120 characters (2026-09-12).** The coherence gate that blanks a looping
answer judged a preview, the first 120 characters of an answer running past 500. One answer
scored 0.139 distinct words over its full text and 0.870 over the preview it was given, so
the campaign blamed the wrong engine; a second, stricter gate written that day read the same
preview and was retracted the same day. The four defects of lesson 1 lived below the same
gate for months.

**A sweep under eviction (2026-09-13).** Sixty-five models, one question and one tool call
each, one model resident at a time. Sixteen clean, eleven catalogue entries that named no
loadable file, three reasoning models fixed by reading the generation prompt, and three load
errors that looked like deep defects. All three were contention: retested cold and alone,
each loaded and answered. A sweep is reliable for a deterministic defect and useless for a
load error.

**Nothing compiles during a cell (2026-09-03).** The same 70B cell gave 2.15 tokens a second,
then 1.64, then 1.58 once a build on twenty threads took half the DRAM bandwidth the host
layers were reading at.

**Profile a slice first (2026-09-17).** Nine machine-hours of full runs on the 552B mixture
found bottlenecks that a two-layer slice finds in minutes: on four layers and 4 096 tokens,
seven changes took one batched prefill from 78.7 s to 18.7 s, each measured by a stage timer
before the whole model was run once to confirm.

**What a record identifies (2026-09-12).** A stored result named its engine by the binary's
modification time, its model by a size and a quantisation label, and every digest read as
zeros; the fairness protocol was real but lived in shell, so a stranger's file looked like a
protocol run. Records are now versioned and self-describing, compared only within an
equivalence class (same schema, same protocol profile, same declared flags), and a
submission is quarantined until reproduced.

## Try it

One cell with the bench tool, against the daemon, three iterations:

```sh
assay --loken http://localhost:11435 --models qwen3:0.6b --num-ctx 4096 \
      --prompts short --stream
```

Observed on qwen3:0.6b, RTX 5070 Ti: prompt 676 tokens a second and completion 724, each
with its spread over `n=3`, 128 tokens generated on every iteration, 0.39 J a token on the
card, and a coherence gate that reads `PASS` above a preview of the answer's first 120
characters. Read three things: the rates are medians and the spread is printed beside them;
the token count says how many steps the decode rate averages over; and the preview is the
whole of what the gate judged. Then run the cell again with a `cargo build` in another
terminal and compare.
