# Benchmarks

One page per hardware configuration, written from the store by `scripts/report.py`; nothing
below the index is typed. A page names the runs, dates and loken builds behind its rows, and
a figure is only ever compared with another on the same page. The
procedure that produces the runs is [BENCHMARKING.md](BENCHMARKING.md).

## Hardware

<!-- benchmarks:index -->

| Hardware | Backend | Id | Runs from | Runs | Rows |
|---|---|---|---|---|---|
| [NVIDIA GeForce RTX 5060 Ti + NVIDIA GeForce RTX 5070 Ti on Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz](benchmarks/rtx5060ti-rtx5070ti-i910900k.md) | unstated | 91853bdab0a9d057 | 2026-09-03 to 2026-10-05 | 711 | 1366 |
| [NVIDIA GeForce RTX 5060 Ti + NVIDIA GeForce RTX 5070 Ti on AMD Ryzen 9 9950X3D 16-Core Processor](benchmarks/rtx5060ti-rtx5070ti-ryzen99950x3d16coreprocessor-cuda.md) | cuda | 45ed3ef839a07686 | 2026-10-18 to 2026-10-19 | 157 | 144 |
| [NVIDIA GeForce GTX 1650 on Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz](benchmarks/gtx1650-i910900k.md) | unstated | b4fa5b0e3e6190aa | 2026-10-18 to 2026-10-21 | 74 | 144 |

<!-- /benchmarks:index -->

## Method

One cell is `(model, context, prompt, mode, device, engine)`, measured by
[assay](https://github.com/loken-ai/assay): 3 iterations, greedy, 128 tokens.

`Ctx` is the window each engine was asked to reserve, not a window the prompt fills: the
longest prompt is about 300 tokens. Ollama allocates the whole window up front, and at 131072
that sends layers of the larger models to the host; this engine plans its quantised cache at
4096 tokens and grows it on demand, so it reserved less. The 131072 column compares those two
policies on short prompts, not two engines at a full context.
Engines run alone, others stopped, cards back at idle - a peer process changes where a model
lands. Ollama runs without Vulkan, `num_gpu 0` for CPU rows.

**Prompts** are the harness's three, byte-identical to every engine. Only the prefill differs;
the 128-token cap keeps the decode column comparable.

| Prompt | What it is | Size |
|---|---|---|
| `short` | a story opening, cut mid-sentence | 13 words |
| `medium` | a paragraph on computing history, likewise | 34 words |
| `long` | a code review over a 22-line Rust function | 139 words |

Vision rows use the parallel `vision_*` prompts.

**Cards.** Ollama is pinned to one card only while the weights fit it - above that it would get
one card plus host spill while the others get the machine. Rows before 2026-08-14 04:00 on
models larger than one card were taken under the older rule and read as wins that were not.

**Prompts reach every engine unwrapped.** `/api/generate` applies the chat template; vLLM is
benched on `/v1/completions`, which does not. Rows before 2026-08-14 08:00 had two engines
answering a story opening as a question while the third continued it.

**Cells that publish no rate.**

| Marked | Meaning |
|---|---|
| `incoherent` | the answer repeats itself. Excluded both ways: it would flatter whichever engine produced it |
| `no answer` | one token back, no decode interval to divide by. Prefill, latency and joules stand |
| `impossible` | the engine was given the model and could not run it: an architecture it does not know, or weights it cannot hold. The result file carries its reason |
| `failed` | the engine took the model and no iteration survived: a request the bench gave up on, or one the engine refused. The result file carries each iteration's error |

**Read `Tokens` before the rate.** A model that stops early is averaged over a few dozen steps
and carries the request's fixed cost - the same engine measured 398 tok/s over 30 tokens and 417
over 128. Short rows are published; their Δ is blank.

The `incoherent` gemma4 cells are a tracked defect here, not a verdict on the models: the same
weights answer correctly under ollama on several of those rows.

**Δ** compares loken to the best other engine, cold with cold and warm with warm, blank when
token counts differ. A positive Δ means loken is better: faster for prefill and decode, a
shorter time to first token, fewer joules. **Energy** sums
NVML GPU draw with the RAPL package and DRAM domains. **vLLM rows read a different file** - an
AWQ 4-bit checkpoint against the GGUF Q4_K_M the other two share - so every vLLM comparison
carries that caveat, and their prefill column was not collected.

## Reading the tables

Check the `Date` and `Run` of a row before comparing two rows: a page can hold cells from
several runs and builds, and the newest measurement of a cell wins. `Device` says `GPU` or
`CPU`: it separates card from host, not one card from two, and two cards flatter this engine
because ollama splits models that fit on one (olmoe measured 496 tok/s on a single card, 390
given the machine).

Sections are the architecture each checkpoint declares under `general.architecture`, not the
name it is published as. That is why `deepseek-r1:70b` sits under `llama` and
`deepseek-r1:32b` under `qwen2`: they are distils, and what the kernels run is the
architecture, not the brand.

## Cold and warm

Every cell runs its prompt three times, the same prompt each time. An engine that keeps a
prompt cache answers the second and third request from it, so one number per cell would
mostly measure the cache. Each rate column therefore shows two figures, `cold / warm`.

**Cold** is the first iteration of the cell: the first request, which no prompt cache has
served. **Warm** is the median of the following iterations, served from the cache by an
engine that keeps one. A cell of a single iteration shows its cold figure and a dash.

The columns that split are Prefill, TTFT and Decode, and so are their three Δ columns, each
printed `cold / warm`. A Δ is blank for a half that cannot be compared. Joules, E2E and the
token count are taken over all iterations, as before. Non-stream rows have no time to first
token, so their Prefill and TTFT are dashes. The scoreboard (`scripts/scoreboard.py`) prints
one line for each: `N/M cells win on both axes`, throughput being the decode Δ of that half
and the other axis the energy Δ.

## The rate columns

**Prefill** is prompt tokens over the time to first token, measured by the bench, the
same way for every engine. **TTFT** is that time, in milliseconds.

It used to be each engine's own `prompt_tok_s`, which is that engine's timer minus whatever it
does not count - and they do not leave out the same things. On a 15-token prompt ollama
reported 652 tok/s where its own time to first token implied 35, and this engine 49 373 where
its own implied 728. Side by side those reversed the verdict on three cells of four, and at the
long prompt the exaggeration was worse, not better.

**Decode** is the client-observed span from the first generated token to the last, also the
same for every engine. vLLM publishes no prefill/decode split, so scoring two engines by their
own bookkeeping and the third by the clock would compare the bookkeeping. Cells measured from
2026-09-02 also carry `server_decode_tok_s` in the result file, the rate the server reports of
itself, recorded so the two can be seen to disagree - at 2 ms a token the client's read loop
costs about what the token does.

**The joules are a median over all the iterations of a cell** (the rates are split cold and
warm, above). The first iteration of a cell is a cold load. Prefill took the median for that reason from the start;
decode and energy took the mean, on the assumption that they did not carry the outlier. On a
mixture too large to hold they do: qwen3next measured 3.8, 15.2, 15.2 tok/s and the mean
published 11.4 for a model that decodes at 15.2. Energy was worse and flattered this engine -
the first iteration's joules include loading the weights, and both engines inflate, not by the
same factor: qwen3:0.6b published +383.6% where the steady state is +149.9%. Cells whose
iterations are flat are unchanged by the median, which is every cell but the large mixtures.
