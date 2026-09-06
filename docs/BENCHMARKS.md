# Benchmarks

> **Every figure on this page is out of date.** They were taken during a phase of very active
> development - kernels, placement and the rate meter all changed under them - so a row says
> what one build did on one day, not what this engine does. A fresh campaign is needed before
> any of it is quoted. The rows are kept because a dated measurement is still evidence of what
> was tried; they are not kept as a claim.

## Hardware

| | |
|---|---|
| GPU 0 | NVIDIA GeForce RTX 5070 Ti - 16 GB, 300 W |
| GPU 1 | NVIDIA GeForce RTX 5060 Ti - 16 GB, 180 W |
| CPU | Intel Core i9-10900K - 10 cores / 20 threads, 3.7 GHz base |

## Method

One cell is `(model, context, prompt, mode, device, engine)`, measured by
[assay](https://github.com/loken-ai/assay): 3 iterations, greedy, 128 tokens, two builds of the engine.

The GPU cells at 4096 and the short prompt at 131072 were measured by the build of
2026-09-03 15:05 (tree `55682a0`), every other cell by the build of 2026-09-04 07:11
(`9f6e6fe`). Two commits separate them: `7ae8c54` releases a layer's file pages while the
model loads, which touches the cold first iteration the medians drop; `9f6e6fe` reuses the
resident KV once a conversation outgrows its window, which no cell here does.
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
| empty row | attempted, nothing survived. Kept so the matrix shows what was tried |

**Read `Tokens` before the rate.** A model that stops early is averaged over a few dozen steps
and carries the request's fixed cost - the same engine measured 398 tok/s over 30 tokens and 417
over 128. Short rows are published; their Δ is blank.

The `incoherent` gemma4 cells are a tracked defect here, not a verdict on the models: the same
weights answer correctly under ollama on several of those rows.

**Δ** compares loken to the best other engine, blank when token counts differ. **Energy** sums
NVML GPU draw with the RAPL package and DRAM domains. **vLLM rows read a different file** - an
AWQ 4-bit checkpoint against the GGUF Q4_K_M the other two share - so every vLLM comparison
carries that caveat, and their prefill column was not collected.

## Results

Check the date before comparing two rows.

- **2026-08-18** - full matrix, both cards available to every engine.
- **2026-08-27** - narrower run, most rows on **one card** (`GPUS=0`) to compare kernels without
  placement in the way. `Device` says `GPU` either way: it separates card from host, not one
  card from two.

Two cards flatter this engine: ollama splits models that fit on one. olmoe measured 496 tok/s on
a single card, 390 given the machine.

Sections are the architecture each checkpoint declares under `general.architecture`, not the
name it is published as. That is why `deepseek-r1:70b` sits under `llama` and
`deepseek-r1:32b` under `qwen2`: they are distils, and what the kernels run is the
architecture, not the brand. Each figure carries every cell of its section that both engines
measured, and says how many of the section's cells that is.

## The two rate columns

**Prefill** is prompt tokens over the median time to first token, measured by the bench, the
same way for every engine.

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

**Both rates, and the joules, are medians over the iterations of a cell.** The first
iteration of a cell is a cold load. Prefill took the median for that reason from the start;
decode and energy took the mean, on the assumption that they did not carry the outlier. On a
mixture too large to hold they do: qwen3next measured 3.8, 15.2, 15.2 tok/s and the mean
published 11.4 for a model that decodes at 15.2. Energy was worse and flattered this engine -
the first iteration's joules include loading the weights, and both engines inflate, not by the
same factor: qwen3:0.6b published +383.6% where the steady state is +149.9%. Cells whose
iterations are flat are unchanged by the median, which is every cell but the large mixtures.

### ernie4_5

![ernie4_5](img/family-ernie4_5.svg)

| Model           | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token   | Δ decode   | Δ energy   | Date             |
|-----------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|-----------|------------|------------|------------------|
| ernie4-5:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 314.2        | 128     | 1 557     | 49      | 0.385     |            |            | 2026-09-04 00:14 |
| ernie4-5:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **514.4**    | **128** | **278**   | **41**  | **0.322** | **+63.7%** | **+19.5%** | 2026-09-04 00:14 |
| ernie4-5:latest | 4096   | long   | stream     | CPU    | Ollama 0.32.6   | 1 385.3       | 54.0         | 128     | 3 246     | 64      | 0.499     |            |            | 2026-09-05 05:50 |
| ernie4-5:latest | 4096   | long   | stream     | CPU    | **loken 0.1.0** | **654.2**     | **49.7**     | **128** | **3 003** | **177** | **1.383** | -7.9%      | -63.9%     | 2026-09-05 05:50 |
| ernie4-5:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 2 693.4       | 406.3        | 128     | 1 590     | 45      | 0.350     |            |            | 2026-09-03 22:36 |
| ernie4-5:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **6 203.7**   | **545.9**    | **128** | **299**   | **37**  | **0.285** | **+34.4%** | **+22.5%** | 2026-09-03 22:36 |
| ernie4-5:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 304.5        | 80      | 1 462     | 29      | 0.367     |            |            | 2026-09-03 20:58 |
| ernie4-5:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **388.7**    | **60**  | **172**   | **17**  | **0.280** |            |            | 2026-09-03 20:58 |
| ernie4-5:latest | 4096   | medium | stream     | CPU    | Ollama 0.32.6   | 236.4         | 56.7         | 22      | 1 192     | 18      | 0.832     |            |            | 2026-09-05 00:47 |
| ernie4-5:latest | 4096   | medium | stream     | CPU    | **loken 0.1.0** | **223.5**     | **53.5**     | **128** | **2 584** | **152** | **1.188** |            |            | 2026-09-05 00:47 |
| ernie4-5:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 319.5         | 420.1        | 80      | 1 469     | 34      | 0.421     |            |            | 2026-09-03 19:14 |
| ernie4-5:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **2 536.8**   | **587.3**    | **60**  | **152**   | **19**  | **0.320** |            |            | 2026-09-03 19:14 |
| ernie4-5:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 316.8        | 128     | 1 560     | 44      | 0.342     |            |            | 2026-09-03 17:36 |
| ernie4-5:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **483.0**    | **128** | **277**   | **32**  | **0.250** | **+52.5%** | **+36.8%** | 2026-09-03 17:36 |
| ernie4-5:latest | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 87.7          | 56.5         | 128     | 3 043     | 57      | 0.443     |            |            | 2026-09-04 20:12 |
| ernie4-5:latest | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **266.2**     | **54.6**     | **8**   | **278**   | **16**  | **1.991** |            |            | 2026-09-04 20:12 |
| ernie4-5:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 164.2         | 419.0        | 128     | 1 553     | 40      | 0.313     |            |            | 2026-09-03 16:00 |
| ernie4-5:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **1 120.1**   | **600.4**    | **128** | **262**   | **42**  | **0.331** | **+43.3%** | -5.6%      | 2026-09-03 16:00 |
| ernie4-5:latest | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 313.2        | 128     | 1 730     | 39      | 0.305     |            |            | 2026-09-04 15:57 |
| ernie4-5:latest | 131072 | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **534.7**    | **128** | **271**   | **41**  | **0.320** | **+70.7%** | -4.6%      | 2026-09-04 15:57 |
| ernie4-5:latest | 131072 | long   | stream     | CPU    | Ollama 0.32.6   | 1 480.0       | 54.1         | 128     | 3 319     | 62      | 0.482     |            |            | 2026-09-06 02:36 |
| ernie4-5:latest | 131072 | long   | stream     | CPU    | **loken 0.1.0** | **663.8**     | **50.6**     | **128** | **2 960** | **178** | **1.389** | -6.5%      | -65.3%     | 2026-09-06 02:36 |
| ernie4-5:latest | 131072 | long   | stream     | GPU    | Ollama 0.32.6   | 2 989.2       | 404.8        | 128     | 1 703     | 40      | 0.313     |            |            | 2026-09-04 13:55 |
| ernie4-5:latest | 131072 | long   | stream     | GPU    | **loken 0.1.0** | **13 584.3**  | **598.9**    | **128** | **274**   | **34**  | **0.267** | **+47.9%** | **+17.4%** | 2026-09-04 13:55 |
| ernie4-5:latest | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 308.7        | 80      | 1 596     | 32      | 0.399     |            |            | 2026-09-04 12:05 |
| ernie4-5:latest | 131072 | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **375.8**    | **60**  | **173**   | **17**  | **0.286** |            |            | 2026-09-04 12:05 |
| ernie4-5:latest | 131072 | medium | stream     | CPU    | Ollama 0.32.6   | 251.6         | 56.7         | 22      | 1 269     | 18      | 0.809     |            |            | 2026-09-05 23:42 |
| ernie4-5:latest | 131072 | medium | stream     | CPU    | **loken 0.1.0** | **207.6**     | **53.5**     | **128** | **2 613** | **158** | **1.230** |            |            | 2026-09-05 23:42 |
| ernie4-5:latest | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 477.2         | 419.9        | 80      | 1 608     | 29      | 0.363     |            |            | 2026-09-04 10:13 |
| ernie4-5:latest | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **2 509.3**   | **577.1**    | **60**  | **160**   | **19**  | **0.312** |            |            | 2026-09-04 10:13 |
| ernie4-5:latest | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 321.5        | 128     | 1 728     | 45      | 0.353     |            |            | 2026-09-04 08:12 |
| ernie4-5:latest | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **464.3**    | **128** | **289**   | **31**  | **0.242** | **+44.4%** | **+46.1%** | 2026-09-04 08:12 |
| ernie4-5:latest | 131072 | short  | stream     | CPU    | Ollama 0.32.6   | 75.3          | 56.5         | 128     | 3 144     | 60      | 0.469     |            |            | 2026-09-05 20:49 |
| ernie4-5:latest | 131072 | short  | stream     | CPU    | **loken 0.1.0** | **156.7**     | **54.5**     | **8**   | **266**   | **18**  | **2.279** |            |            | 2026-09-05 20:49 |
| ernie4-5:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 169.1         | 418.7        | 128     | 1 717     | 40      | 0.316     |            |            | 2026-09-04 02:14 |
| ernie4-5:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **1 061.7**   | **604.1**    | **128** | **262**   | **38**  | **0.296** | **+44.3%** | **+6.9%**  | 2026-09-04 02:14 |

### gemma4

![gemma4](img/family-gemma4.svg)

| Model         | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms      | J/req     | J/token    | Δ decode    | Δ energy   | Date             |
|---------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|-------------|-----------|------------|-------------|------------|------------------|
| gemma4:12b    | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 42.8         | 128     | 5 171       | 501       | 3.915      |             |            | 2026-09-04 00:16 |
| gemma4:12b    | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **67.4**     | **128** | **1 932**   | **431**   | **3.369**  | **+57.7%**  | **+16.2%** | 2026-09-04 00:16 |
| gemma4:12b    | 4096   | long   | stream     | CPU    | Ollama 0.32.6   | 223.7         | 3.0          | 128     | 48 409      | 1 029     | 8.041      |             |            | 2026-09-05 05:58 |
| gemma4:12b    | 4096   | long   | stream     | CPU    | **loken 0.1.0** | **42.2**      | **2.8**      | **128** | **51 938**  | **1 183** | **9.245**  | -4.4%       | -13.0%     | 2026-09-05 05:58 |
| gemma4:12b    | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 468.0         | 53.6         | 128     | 5 139       | 503       | 3.931      |             |            | 2026-09-03 22:39 |
| gemma4:12b    | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **2 482.9**   | **69.9**     | **128** | **1 976**   | **446**   | **3.484**  | **+30.4%**  | **+12.8%** | 2026-09-03 22:39 |
| gemma4:12b    | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -          |  -        |  -         | incoherent  |            | 2026-09-03 21:00 |
| gemma4:12b    | 4096   | medium | non-stream | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -          |  -        |  -         | incoherent  |            | 2026-09-03 21:00 |
| gemma4:12b    | 4096   | medium | stream     | CPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -          |  -        |  -         | incoherent  |            | 2026-09-05 00:55 |
| gemma4:12b    | 4096   | medium | stream     | CPU    | **loken 0.1.0** |  -            |  -           | **128** |  -          |  -        |  -         | incoherent  |            | 2026-09-05 00:55 |
| gemma4:12b    | 4096   | medium | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 74      |  -          |  -        |  -         | incoherent  |            | 2026-09-03 19:17 |
| gemma4:12b    | 4096   | medium | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -          |  -        |  -         | incoherent  |            | 2026-09-03 19:17 |
| gemma4:12b    | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 27.3         | 61      | 3 805       | 292       | 4.757      |             |            | 2026-09-03 17:38 |
| gemma4:12b    | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **74.7**     | **128** | **1 745**   | **402**   | **3.139**  |             |            | 2026-09-03 17:38 |
| gemma4:12b    | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 13.3          | 3.0          | 128     | 45 788      | 1 008     | 7.878      |             |            | 2026-09-04 20:20 |
| gemma4:12b    | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **17.3**      | **2.9**      | **128** | **45 228**  | **1 031** | **8.051**  | -4.1%       | -2.2%      | 2026-09-04 20:20 |
| gemma4:12b    | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 30.3          | 54.6         | 61      | 3 798       | 291       | 4.751      |             |            | 2026-09-03 16:02 |
| gemma4:12b    | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **184.3**     | **73.9**     | **128** | **1 784**   | **400**   | **3.128**  |             |            | 2026-09-03 16:02 |
| gemma4:12b    | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 42.8         | 128     | 5 296       | 506       | 3.954      |             |            | 2026-09-04 16:00 |
| gemma4:12b    | 131072 | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **67.2**     | **128** | **1 937**   | **430**   | **3.363**  | **+57.2%**  | **+17.6%** | 2026-09-04 16:00 |
| gemma4:12b    | 131072 | long   | stream     | CPU    | Ollama 0.32.6   | 222.5         | 3.0          | 128     | 48 668      | 1 031     | 8.057      |             |            | 2026-09-06 02:44 |
| gemma4:12b    | 131072 | long   | stream     | CPU    | **loken 0.1.0** | **42.4**      | **2.9**      | **128** | **51 304**  | **1 171** | **9.147**  | -3.2%       | -11.9%     | 2026-09-06 02:44 |
| gemma4:12b    | 131072 | long   | stream     | GPU    | Ollama 0.32.6   | 465.6         | 53.4         | 128     | 5 276       | 494       | 3.863      |             |            | 2026-09-04 13:58 |
| gemma4:12b    | 131072 | long   | stream     | GPU    | **loken 0.1.0** | **2 417.6**   | **69.9**     | **128** | **1 968**   | **448**   | **3.497**  | **+30.8%**  | **+10.5%** | 2026-09-04 13:58 |
| gemma4:12b    | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -          |  -        |  -         | incoherent  |            | 2026-09-04 12:08 |
| gemma4:12b    | 131072 | medium | non-stream | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -          |  -        |  -         | incoherent  |            | 2026-09-04 12:08 |
| gemma4:12b    | 131072 | medium | stream     | CPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -          |  -        |  -         | incoherent  |            | 2026-09-05 23:50 |
| gemma4:12b    | 131072 | medium | stream     | CPU    | **loken 0.1.0** |  -            |  -           | **128** |  -          |  -        |  -         | incoherent  |            | 2026-09-05 23:50 |
| gemma4:12b    | 131072 | medium | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 74      |  -          |  -        |  -         | incoherent  |            | 2026-09-04 10:16 |
| gemma4:12b    | 131072 | medium | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -          |  -        |  -         | incoherent  |            | 2026-09-04 10:16 |
| gemma4:12b    | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 26.6         | 61      | 3 975       | 298       | 4.863      |             |            | 2026-09-04 08:14 |
| gemma4:12b    | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **73.7**     | **128** | **1 759**   | **395**   | **3.085**  |             |            | 2026-09-04 08:14 |
| gemma4:12b    | 131072 | short  | stream     | CPU    | Ollama 0.32.6   | 13.5          | 3.0          | 128     | 45 733      | 1 004     | 7.847      |             |            | 2026-09-05 20:57 |
| gemma4:12b    | 131072 | short  | stream     | CPU    | **loken 0.1.0** | **17.7**      | **2.9**      | **128** | **44 763**  | **1 019** | **7.960**  | -3.4%       | -1.4%      | 2026-09-05 20:57 |
| gemma4:12b    | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 27.9          | 54.5         | 61      | 3 992       | 316       | 5.155      |             |            | 2026-09-04 02:17 |
| gemma4:12b    | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **332.1**     | **73.9**     | **128** | **1 786**   | **401**   | **3.130**  |             |            | 2026-09-04 02:17 |
| gemma4:26b    | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -          |  -        |  -         | incoherent  |            | 2026-09-04 00:19 |
| gemma4:26b    | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **83.2**     | **128** | **1 666**   | **265**   | **2.068**  |             |            | 2026-09-04 00:19 |
| gemma4:26b    | 4096   | long   | stream     | CPU    | Ollama 0.32.6   | 315.6         | 7.5          | 128     | 23 098      | 407       | 3.177      |             |            | 2026-09-05 06:25 |
| gemma4:26b    | 4096   | long   | stream     | CPU    | **loken 0.1.0** |  -            |  -           | **128** |  -          |  -        |  -         | incoherent  |            | 2026-09-05 06:25 |
| gemma4:26b    | 4096   | long   | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -          |  -        |  -         | incoherent  |            | 2026-09-03 22:41 |
| gemma4:26b    | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **1 145.1**   | **95.4**     | **128** | **1 699**   | **266**   | **2.074**  |             |            | 2026-09-03 22:41 |
| gemma4:26b    | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -          |  -        |  -         | incoherent  |            | 2026-09-03 21:02 |
| gemma4:26b    | 4096   | medium | non-stream | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -          |  -        |  -         | incoherent  |            | 2026-09-03 21:02 |
| gemma4:26b    | 4096   | medium | stream     | CPU    | Ollama 0.32.6   |  -            |  -           | 98      |  -          |  -        |  -         | incoherent  |            | 2026-09-05 01:21 |
| gemma4:26b    | 4096   | medium | stream     | CPU    | **loken 0.1.0** |  -            |  -           | **128** |  -          |  -        |  -         | incoherent  |            | 2026-09-05 01:21 |
| gemma4:26b    | 4096   | medium | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -          |  -        |  -         | incoherent  |            | 2026-09-03 19:19 |
| gemma4:26b    | 4096   | medium | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -          |  -        |  -         | incoherent  |            | 2026-09-03 19:19 |
| gemma4:26b    | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -          |  -        |  -         | incoherent  |            | 2026-09-03 17:41 |
| gemma4:26b    | 4096   | short  | non-stream | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -          |  -        |  -         | incoherent  |            | 2026-09-03 17:41 |
| gemma4:26b    | 4096   | short  | stream     | CPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -          |  -        |  -         | incoherent  |            | 2026-09-04 20:45 |
| gemma4:26b    | 4096   | short  | stream     | CPU    | **loken 0.1.0** |  -            |  -           | **128** |  -          |  -        |  -         | incoherent  |            | 2026-09-04 20:45 |
| gemma4:26b    | 4096   | short  | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 93      |  -          |  -        |  -         | incoherent  |            | 2026-09-03 16:05 |
| gemma4:26b    | 4096   | short  | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -          |  -        |  -         | incoherent  |            | 2026-09-03 16:05 |
| gemma4:26b    | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -          |  -        |  -         | incoherent  |            | 2026-09-04 16:02 |
| gemma4:26b    | 131072 | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **82.7**     | **128** | **1 664**   | **258**   | **2.018**  |             |            | 2026-09-04 16:02 |
| gemma4:26b    | 131072 | long   | stream     | CPU    | Ollama 0.32.6   | 312.1         | 7.5          | 128     | 23 323      | 413       | 3.224      |             |            | 2026-09-06 03:15 |
| gemma4:26b    | 131072 | long   | stream     | CPU    | **loken 0.1.0** |  -            |  -           | **128** |  -          |  -        |  -         | incoherent  |            | 2026-09-06 03:15 |
| gemma4:26b    | 131072 | long   | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -          |  -        |  -         | incoherent  |            | 2026-09-04 14:00 |
| gemma4:26b    | 131072 | long   | stream     | GPU    | **loken 0.1.0** | **1 126.9**   | **95.3**     | **128** | **1 696**   | **263**   | **2.057**  |             |            | 2026-09-04 14:00 |
| gemma4:26b    | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -          |  -        |  -         | incoherent  |            | 2026-09-04 12:10 |
| gemma4:26b    | 131072 | medium | non-stream | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -          |  -        |  -         | incoherent  |            | 2026-09-04 12:10 |
| gemma4:26b    | 131072 | medium | stream     | CPU    | Ollama 0.32.6   |  -            |  -           | 98      |  -          |  -        |  -         | incoherent  |            | 2026-09-06 00:18 |
| gemma4:26b    | 131072 | medium | stream     | CPU    | **loken 0.1.0** |  -            |  -           | **128** |  -          |  -        |  -         | incoherent  |            | 2026-09-06 00:18 |
| gemma4:26b    | 131072 | medium | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -          |  -        |  -         | incoherent  |            | 2026-09-04 10:18 |
| gemma4:26b    | 131072 | medium | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -          |  -        |  -         | incoherent  |            | 2026-09-04 10:18 |
| gemma4:26b    | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -          |  -        |  -         | incoherent  |            | 2026-09-04 08:16 |
| gemma4:26b    | 131072 | short  | non-stream | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -          |  -        |  -         | incoherent  |            | 2026-09-04 08:16 |
| gemma4:26b    | 131072 | short  | stream     | CPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -          |  -        |  -         | incoherent  |            | 2026-09-05 21:22 |
| gemma4:26b    | 131072 | short  | stream     | CPU    | **loken 0.1.0** |  -            |  -           | **128** |  -          |  -        |  -         | incoherent  |            | 2026-09-05 21:22 |
| gemma4:26b    | 131072 | short  | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 93      |  -          |  -        |  -         | incoherent  |            | 2026-09-04 02:19 |
| gemma4:26b    | 131072 | short  | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -          |  -        |  -         | incoherent  |            | 2026-09-04 02:19 |
| gemma4:31b    | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           |  -      |  -          |  -        |  -         | incoherent  |            | 2026-09-04 00:21 |
| gemma4:31b    | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **22.3**     | **128** | **5 824**   | **1 219** | **9.527**  |             |            | 2026-09-04 00:21 |
| gemma4:31b    | 4096   | long   | stream     | CPU    | Ollama 0.32.6   |  -            |  -           | 53      |  -          |  -        |  -         | incoherent  |            | 2026-09-05 06:36 |
| gemma4:31b    | 4096   | long   | stream     | CPU    | **loken 0.1.0** | **17.1**      | **1.1**      | **128** | **130 751** | **2 909** | **22.730** |             |            | 2026-09-05 06:36 |
| gemma4:31b    | 4096   | long   | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 44      |  -          |  -        |  -         | incoherent  |            | 2026-09-03 22:44 |
| gemma4:31b    | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **828.2**     | **23.4**     | **128** | **5 885**   | **1 222** | **9.547**  |             |            | 2026-09-03 22:44 |
| gemma4:31b    | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           |  -      | 5 747       |  -        |  -         |             |            | 2026-09-03 21:05 |
| gemma4:31b    | 4096   | medium | non-stream | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -          |  -        |  -         | incoherent  |            | 2026-09-03 21:05 |
| gemma4:31b    | 4096   | medium | stream     | CPU    | Ollama 0.32.6   | 13.4          | 1.2          | 75      | 72 207      | 1 584     | 21.214     |             |            | 2026-09-05 01:32 |
| gemma4:31b    | 4096   | medium | stream     | CPU    | **loken 0.1.0** | **12.3**      | **1.2**      | **128** | **115 692** | **2 593** | **20.258** |             |            | 2026-09-05 01:32 |
| gemma4:31b    | 4096   | medium | stream     | GPU    | Ollama 0.32.6   |  -            | 25.6         | 48      | 5 798       | 452       | 9.414      |             |            | 2026-09-03 19:22 |
| gemma4:31b    | 4096   | medium | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -          |  -        |  -         | incoherent  |            | 2026-09-03 19:22 |
| gemma4:31b    | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 8.3          | 128     | 7 619       | 1 831     | 14.304     |             |            | 2026-09-03 17:43 |
| gemma4:31b    | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **24.8**     | **128** | **5 262**   | **1 097** | **8.570**  | **+197.4%** | **+66.9%** | 2026-09-03 17:43 |
| gemma4:31b    | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 4.9           | 1.2          | 121     | 110 769     | 2 416     | 19.911     |             |            | 2026-09-04 20:58 |
| gemma4:31b    | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **7.8**       | **1.2**      | **128** | **113 423** | **2 518** | **19.670** |             |            | 2026-09-04 20:58 |
| gemma4:31b    | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 26.2          | 25.7         | 96      | 7 588       | 866       | 9.019      |             |            | 2026-09-03 16:07 |
| gemma4:31b    | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **188.9**     | **24.9**     | **128** | **5 314**   | **1 098** | **8.579**  |             |            | 2026-09-03 16:07 |
| gemma4:31b    | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           |  -      |  -          |  -        |  -         | incoherent  |            | 2026-09-04 16:05 |
| gemma4:31b    | 131072 | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **22.2**     | **128** | **5 842**   | **1 255** | **9.802**  |             |            | 2026-09-04 16:05 |
| gemma4:31b    | 131072 | long   | stream     | CPU    | Ollama 0.32.6   |  -            |  -           | 53      |  -          |  -        |  -         | incoherent  |            | 2026-09-06 03:27 |
| gemma4:31b    | 131072 | long   | stream     | CPU    | **loken 0.1.0** | **16.9**      | **1.2**      | **128** | **130 286** | **2 885** | **22.541** |             |            | 2026-09-06 03:27 |
| gemma4:31b    | 131072 | long   | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 41      |  -          |  -        |  -         | incoherent  |            | 2026-09-04 14:03 |
| gemma4:31b    | 131072 | long   | stream     | GPU    | **loken 0.1.0** | **824.4**     | **23.4**     | **128** | **5 871**   | **1 243** | **9.708**  |             |            | 2026-09-04 14:03 |
| gemma4:31b    | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           |  -      | 22 092      |  -        |  -         |             |            | 2026-09-04 12:14 |
| gemma4:31b    | 131072 | medium | non-stream | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -          |  -        |  -         | incoherent  |            | 2026-09-04 12:14 |
| gemma4:31b    | 131072 | medium | stream     | CPU    | Ollama 0.32.6   | 13.0          | 1.2          | 75      | 73 127      | 1 576     | 21.105     |             |            | 2026-09-06 00:29 |
| gemma4:31b    | 131072 | medium | stream     | CPU    | **loken 0.1.0** | **12.8**      | **1.2**      | **128** | **115 016** | **2 559** | **19.993** |             |            | 2026-09-06 00:29 |
| gemma4:31b    | 131072 | medium | stream     | GPU    | Ollama 0.32.6   |  -            | 3.7          | 82      | 33 104      | 2 729     | 33.285     |             |            | 2026-09-04 10:22 |
| gemma4:31b    | 131072 | medium | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -          |  -        |  -         | incoherent  |            | 2026-09-04 10:22 |
| gemma4:31b    | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           |  -      | 18 183      |  -        |  -         |             |            | 2026-09-04 08:19 |
| gemma4:31b    | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **24.8**     | **128** | **5 253**   | **1 098** | **8.576**  |             |            | 2026-09-04 08:19 |
| gemma4:31b    | 131072 | short  | stream     | CPU    | Ollama 0.32.6   | 5.2           | 1.2          | 121     | 111 351     | 2 416     | 19.915     |             |            | 2026-09-05 21:35 |
| gemma4:31b    | 131072 | short  | stream     | CPU    | **loken 0.1.0** | **7.5**       | **1.2**      | **128** | **113 337** | **2 534** | **19.794** |             |            | 2026-09-05 21:35 |
| gemma4:31b    | 131072 | short  | stream     | GPU    | Ollama 0.32.6   |  -            | 5.1          | 63      | 18 065      | 1 195     | 18.974     |             |            | 2026-09-04 02:25 |
| gemma4:31b    | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **190.7**     | **24.9**     | **128** | **5 284**   | **1 118** | **8.734**  |             |            | 2026-09-04 02:25 |
| gemma4:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 66.6         | 128     | 3 954       | 258       | 2.016      |             |            | 2026-09-04 00:23 |
| gemma4:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **119.5**    | **128** | **1 121**   | **202**   | **1.582**  | **+79.3%**  | **+27.5%** | 2026-09-04 00:23 |
| gemma4:latest | 4096   | long   | stream     | CPU    | Ollama 0.32.6   | 351.1         | 7.1          | 128     | 21 528      | 437       | 3.418      |             |            | 2026-09-05 06:40 |
| gemma4:latest | 4096   | long   | stream     | CPU    | **loken 0.1.0** | **97.4**      | **6.7**      | **128** | **22 399**  | **513**   | **4.006**  | -5.7%       | -14.7%     | 2026-09-05 06:40 |
| gemma4:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 544.4         | 89.5         | 128     | 3 973       | 254       | 1.982      |             |            | 2026-09-03 22:45 |
| gemma4:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **3 852.7**   | **125.0**    | **128** | **1 125**   | **196**   | **1.534**  | **+39.7%**  | **+29.2%** | 2026-09-03 22:45 |
| gemma4:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 67.2         | 128     | 3 927       | 243       | 1.900      |             |            | 2026-09-03 21:06 |
| gemma4:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **128.1**    | **128** | **1 026**   | **189**   | **1.479**  | **+90.6%**  | **+28.5%** | 2026-09-03 21:06 |
| gemma4:latest | 4096   | medium | stream     | CPU    | Ollama 0.32.6   | 56.5          | 7.2          | 128     | 20 635      | 432       | 3.373      |             |            | 2026-09-05 01:35 |
| gemma4:latest | 4096   | medium | stream     | CPU    | **loken 0.1.0** | **80.4**      | **7.0**      | **128** | **19 197**  | **437**   | **3.410**  | -2.2%       | -1.1%      | 2026-09-05 01:35 |
| gemma4:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 87.9          | 90.3         | 128     | 3 923       | 251       | 1.958      |             |            | 2026-09-03 19:23 |
| gemma4:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **540.5**     | **131.8**    | **128** | **1 072**   | **185**   | **1.445**  | **+46.0%**  | **+35.5%** | 2026-09-03 19:23 |
| gemma4:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 67.5         | 128     | 3 903       | 247       | 1.926      |             |            | 2026-09-03 17:45 |
| gemma4:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **128.3**    | **128** | **1 023**   | **175**   | **1.368**  | **+90.1%**  | **+40.8%** | 2026-09-03 17:45 |
| gemma4:latest | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 19.6          | 7.2          | 128     | 20 567      | 429       | 3.354      |             |            | 2026-09-04 21:01 |
| gemma4:latest | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **42.5**      | **7.1**      | **128** | **18 846**  | **428**   | **3.347**  | -1.6%       | **+0.2%**  | 2026-09-04 21:01 |
| gemma4:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 31.9          | 91.4         | 128     | 3 936       | 243       | 1.899      |             |            | 2026-09-03 16:09 |
| gemma4:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **339.4**     | **133.1**    | **128** | **1 048**   | **183**   | **1.430**  | **+45.7%**  | **+32.8%** | 2026-09-03 16:09 |
| gemma4:latest | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 66.9         | 128     | 4 042       | 253       | 1.978      |             |            | 2026-09-04 16:07 |
| gemma4:latest | 131072 | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **118.5**    | **128** | **1 098**   | **198**   | **1.550**  | **+77.0%**  | **+27.7%** | 2026-09-04 16:07 |
| gemma4:latest | 131072 | long   | stream     | CPU    | Ollama 0.32.6   | 338.9         | 7.1          | 128     | 21 645      | 433       | 3.384      |             |            | 2026-09-06 03:30 |
| gemma4:latest | 131072 | long   | stream     | CPU    | **loken 0.1.0** | **97.1**      | **6.7**      | **128** | **22 348**  | **510**   | **3.984**  | -5.4%       | -15.1%     | 2026-09-06 03:30 |
| gemma4:latest | 131072 | long   | stream     | GPU    | Ollama 0.32.6   | 558.5         | 89.5         | 128     | 4 037       | 255       | 1.994      |             |            | 2026-09-04 14:05 |
| gemma4:latest | 131072 | long   | stream     | GPU    | **loken 0.1.0** | **3 554.5**   | **124.3**    | **128** | **1 146**   | **199**   | **1.556**  | **+38.9%**  | **+28.2%** | 2026-09-04 14:05 |
| gemma4:latest | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 67.2         | 128     | 3 977       | 259       | 2.020      |             |            | 2026-09-04 12:15 |
| gemma4:latest | 131072 | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **126.4**    | **128** | **1 026**   | **188**   | **1.468**  | **+88.2%**  | **+37.7%** | 2026-09-04 12:15 |
| gemma4:latest | 131072 | medium | stream     | CPU    | Ollama 0.32.6   | 54.9          | 7.2          | 128     | 20 808      | 426       | 3.331      |             |            | 2026-09-06 00:32 |
| gemma4:latest | 131072 | medium | stream     | CPU    | **loken 0.1.0** | **58.7**      | **7.0**      | **128** | **19 228**  | **439**   | **3.431**  | -1.8%       | -2.9%      | 2026-09-06 00:32 |
| gemma4:latest | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 52.5          | 68.4         | 128     | 6 780       | 303       | 2.370      |             |            | 2026-09-04 10:24 |
| gemma4:latest | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **315.5**     | **75.6**     | **128** | **1 962**   | **244**   | **1.908**  | **+10.5%**  | **+24.2%** | 2026-09-04 10:24 |
| gemma4:latest | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 67.8         | 128     | 4 022       | 248       | 1.941      |             |            | 2026-09-04 08:21 |
| gemma4:latest | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **128.7**    | **128** | **1 020**   | **184**   | **1.439**  | **+90.0%**  | **+34.8%** | 2026-09-04 08:21 |
| gemma4:latest | 131072 | short  | stream     | CPU    | Ollama 0.32.6   | 19.2          | 7.2          | 128     | 20 680      | 433       | 3.379      |             |            | 2026-09-05 21:38 |
| gemma4:latest | 131072 | short  | stream     | CPU    | **loken 0.1.0** | **30.1**      | **7.1**      | **128** | **18 869**  | **431**   | **3.371**  | -1.6%       | **+0.3%**  | 2026-09-05 21:38 |
| gemma4:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 32.8          | 91.2         | 128     | 4 024       | 252       | 1.967      |             |            | 2026-09-04 02:27 |
| gemma4:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **331.0**     | **133.4**    | **128** | **1 060**   | **181**   | **1.413**  | **+46.2%**  | **+39.3%** | 2026-09-04 02:27 |

### gptoss

![gptoss](img/family-gptoss.svg)

| Model       | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode    | Δ energy    | Date             |
|-------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|-------------|-------------|------------------|
| gpt-oss:20b | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 71.7         | 128     | 4 380      | 252     | 1.966     |             |             | 2026-09-04 00:35 |
| gpt-oss:20b | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **205.2**    | **128** | **888**    | **130** | **1.012** | **+186.4%** | **+94.3%**  | 2026-09-04 00:35 |
| gpt-oss:20b | 4096   | long   | stream     | CPU    | Ollama 0.32.6   | 361.2         | 6.1          | 128     | 24 697     | 503     | 3.931     |             |             | 2026-09-05 06:43 |
| gpt-oss:20b | 4096   | long   | stream     | CPU    | **loken 0.1.0** | **1 406.8**   | **7.3**      | **128** | **20 048** | **416** | **3.252** | **+20.7%**  | **+20.9%**  | 2026-09-05 06:43 |
| gpt-oss:20b | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 575.4         | 96.1         | 128     | 4 435      | 251     | 1.958     |             |             | 2026-09-03 22:57 |
| gpt-oss:20b | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **10 647.6**  | **210.0**    | **128** | **913**    | **128** | **1.004** | **+118.5%** | **+95.1%**  | 2026-09-03 22:57 |
| gpt-oss:20b | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 71.3         | 128     | 4 377      | 245     | 1.914     |             |             | 2026-09-03 21:18 |
| gpt-oss:20b | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **200.6**    | **128** | **828**    | **123** | **0.960** | **+181.2%** | **+99.5%**  | 2026-09-03 21:18 |
| gpt-oss:20b | 4096   | medium | stream     | CPU    | Ollama 0.32.6   | 50.7          | 6.1          | 128     | 24 032     | 509     | 3.979     |             |             | 2026-09-05 01:39 |
| gpt-oss:20b | 4096   | medium | stream     | CPU    | **loken 0.1.0** | **268.5**     | **7.5**      | **128** | **18 588** | **407** | **3.183** | **+21.5%**  | **+25.0%**  | 2026-09-05 01:39 |
| gpt-oss:20b | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 99.5          | 93.8         | 128     | 4 350      | 250     | 1.956     |             |             | 2026-09-03 19:35 |
| gpt-oss:20b | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 232.4**   | **212.5**    | **128** | **836**    | **123** | **0.962** | **+126.4%** | **+103.3%** | 2026-09-03 19:35 |
| gpt-oss:20b | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 72.3         | 128     | 4 281      | 245     | 1.913     |             |             | 2026-09-03 17:57 |
| gpt-oss:20b | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **197.1**    | **128** | **805**    | **128** | **1.001** | **+172.6%** | **+91.1%**  | 2026-09-03 17:57 |
| gpt-oss:20b | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 18.3          | 6.1          | 128     | 23 835     | 505     | 3.943     |             |             | 2026-09-04 21:04 |
| gpt-oss:20b | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **95.5**      | **7.5**      | **128** | **18 266** | **405** | **3.160** | **+21.7%**  | **+24.7%**  | 2026-09-04 21:04 |
| gpt-oss:20b | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 35.4          | 94.6         | 128     | 4 317      | 245     | 1.913     |             |             | 2026-09-03 16:21 |
| gpt-oss:20b | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **449.4**     | **211.8**    | **128** | **820**    | **124** | **0.969** | **+123.8%** | **+97.3%**  | 2026-09-03 16:21 |
| gpt-oss:20b | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 74.1         | 128     | 4 477      | 246     | 1.925     |             |             | 2026-09-04 16:19 |
| gpt-oss:20b | 131072 | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **198.1**    | **128** | **901**    | **139** | **1.085** | **+167.4%** | **+77.3%**  | 2026-09-04 16:19 |
| gpt-oss:20b | 131072 | long   | stream     | CPU    | Ollama 0.32.6   | 369.8         | 6.1          | 128     | 25 070     | 504     | 3.936     |             |             | 2026-09-06 03:34 |
| gpt-oss:20b | 131072 | long   | stream     | CPU    | **loken 0.1.0** | **1 837.9**   | **7.3**      | **128** | **20 024** | **412** | **3.221** | **+21.0%**  | **+22.2%**  | 2026-09-06 03:34 |
| gpt-oss:20b | 131072 | long   | stream     | GPU    | Ollama 0.32.6   | 571.6         | 96.3         | 128     | 4 492      | 249     | 1.948     |             |             | 2026-09-04 14:17 |
| gpt-oss:20b | 131072 | long   | stream     | GPU    | **loken 0.1.0** | **5 023.7**   | **210.2**    | **128** | **918**    | **131** | **1.024** | **+118.2%** | **+90.2%**  | 2026-09-04 14:17 |
| gpt-oss:20b | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 72.5         | 128     | 4 526      | 252     | 1.968     |             |             | 2026-09-04 12:27 |
| gpt-oss:20b | 131072 | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **210.1**    | **128** | **798**    | **118** | **0.923** | **+189.7%** | **+113.3%** | 2026-09-04 12:27 |
| gpt-oss:20b | 131072 | medium | stream     | CPU    | Ollama 0.32.6   | 50.5          | 6.1          | 128     | 24 238     | 501     | 3.913     |             |             | 2026-09-06 00:36 |
| gpt-oss:20b | 131072 | medium | stream     | CPU    | **loken 0.1.0** | **243.4**     | **7.5**      | **128** | **18 606** | **406** | **3.174** | **+21.9%**  | **+23.3%**  | 2026-09-06 00:36 |
| gpt-oss:20b | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 97.3          | 94.0         | 128     | 4 524      | 252     | 1.971     |             |             | 2026-09-04 10:37 |
| gpt-oss:20b | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **1 449.9**   | **212.8**    | **128** | **831**    | **122** | **0.957** | **+126.4%** | **+106.0%** | 2026-09-04 10:37 |
| gpt-oss:20b | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 72.6         | 128     | 4 463      | 249     | 1.946     |             |             | 2026-09-04 08:33 |
| gpt-oss:20b | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **197.1**    | **128** | **812**    | **125** | **0.979** | **+171.6%** | **+98.7%**  | 2026-09-04 08:33 |
| gpt-oss:20b | 131072 | short  | stream     | CPU    | Ollama 0.32.6   | 18.1          | 6.1          | 128     | 24 053     | 508     | 3.972     |             |             | 2026-09-05 21:42 |
| gpt-oss:20b | 131072 | short  | stream     | CPU    | **loken 0.1.0** | **111.7**     | **7.5**      | **128** | **18 292** | **404** | **3.159** | **+22.1%**  | **+25.7%**  | 2026-09-05 21:42 |
| gpt-oss:20b | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 35.7          | 94.5         | 128     | 4 452      | 245     | 1.910     |             |             | 2026-09-04 02:39 |
| gpt-oss:20b | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **511.1**     | **212.8**    | **128** | **824**    | **115** | **0.899** | **+125.1%** | **+112.5%** | 2026-09-04 02:39 |

### granite

![granite](img/family-granite.svg)

| Model               | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode   | Δ energy   | Date             |
|---------------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|------------|------------|------------------|
| granite3.1-dense:2b | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 200.9        | 128     | 1 862      | 102     | 0.799     |            |            | 2026-09-04 00:37 |
| granite3.1-dense:2b | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **254.2**    | **128** | **537**    | **115** | **0.899** | **+26.5%** | -11.1%     | 2026-09-04 00:37 |
| granite3.1-dense:2b | 4096   | long   | stream     | CPU    | Ollama 0.32.6   | 1 187.9       | 13.6         | 128     | 10 932     | 227     | 1.771     |            |            | 2026-09-05 06:47 |
| granite3.1-dense:2b | 4096   | long   | stream     | CPU    | **loken 0.1.0** | **168.8**     | **11.9**     | **128** | **12 674** | **402** | **3.141** | -12.5%     | -43.6%     | 2026-09-05 06:47 |
| granite3.1-dense:2b | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 2 773.6       | 223.0        | 128     | 1 874      | 120     | 0.935     |            |            | 2026-09-03 22:59 |
| granite3.1-dense:2b | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **8 602.9**   | **273.3**    | **128** | **546**    | **114** | **0.888** | **+22.6%** | **+5.3%**  | 2026-09-03 22:59 |
| granite3.1-dense:2b | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 201.1        | 128     | 1 872      | 99      | 0.777     |            |            | 2026-09-03 21:20 |
| granite3.1-dense:2b | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **273.3**    | **128** | **497**    | **109** | **0.849** | **+35.9%** | -8.5%      | 2026-09-03 21:20 |
| granite3.1-dense:2b | 4096   | medium | stream     | CPU    | Ollama 0.32.6   | 217.3         | 13.8         | 128     | 10 350     | 223     | 1.740     |            |            | 2026-09-05 01:42 |
| granite3.1-dense:2b | 4096   | medium | stream     | CPU    | **loken 0.1.0** | **103.8**     | **13.6**     | **128** | **10 086** | **438** | **3.423** | -2.1%      | -49.2%     | 2026-09-05 01:42 |
| granite3.1-dense:2b | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 689.1         | 224.2        | 128     | 1 890      | 100     | 0.778     |            |            | 2026-09-03 19:37 |
| granite3.1-dense:2b | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **3 550.4**   | **290.2**    | **128** | **487**    | **110** | **0.863** | **+29.4%** | -9.8%      | 2026-09-03 19:37 |
| granite3.1-dense:2b | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 201.3        | 128     | 1 869      | 104     | 0.810     |            |            | 2026-09-03 17:59 |
| granite3.1-dense:2b | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **262.3**    | **128** | **496**    | **103** | **0.803** | **+30.3%** | **+0.8%**  | 2026-09-03 17:59 |
| granite3.1-dense:2b | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 69.9          | 13.9         | 128     | 10 246     | 220     | 1.715     |            |            | 2026-09-04 21:07 |
| granite3.1-dense:2b | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **47.2**      | **13.2**     | **128** | **10 139** | **419** | **3.274** | -4.8%      | -47.6%     | 2026-09-04 21:07 |
| granite3.1-dense:2b | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 158.7         | 224.6        | 128     | 1 879      | 112     | 0.873     |            |            | 2026-09-03 16:23 |
| granite3.1-dense:2b | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **943.5**     | **293.1**    | **128** | **486**    | **109** | **0.848** | **+30.5%** | **+2.9%**  | 2026-09-03 16:23 |
| granite3.1-dense:2b | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 195.0        | 128     | 2 010      | 113     | 0.884     |            |            | 2026-09-04 16:21 |
| granite3.1-dense:2b | 131072 | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **253.0**    | **128** | **539**    | **112** | **0.875** | **+29.7%** | **+1.0%**  | 2026-09-04 16:21 |
| granite3.1-dense:2b | 131072 | long   | stream     | CPU    | Ollama 0.32.6   | 1 146.3       | 13.6         | 128     | 11 556     | 227     | 1.772     |            |            | 2026-09-06 03:37 |
| granite3.1-dense:2b | 131072 | long   | stream     | CPU    | **loken 0.1.0** | **168.6**     | **11.9**     | **128** | **12 704** | **396** | **3.091** | -12.4%     | -42.7%     | 2026-09-06 03:37 |
| granite3.1-dense:2b | 131072 | long   | stream     | GPU    | Ollama 0.32.6   | 1 831.6       | 174.9        | 128     | 3 471      | 140     | 1.091     |            |            | 2026-09-04 14:19 |
| granite3.1-dense:2b | 131072 | long   | stream     | GPU    | **loken 0.1.0** | **5 756.6**   | **246.0**    | **128** | **615**    | **122** | **0.950** | **+40.6%** | **+14.8%** | 2026-09-04 14:19 |
| granite3.1-dense:2b | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 199.7        | 128     | 2 125      | 102     | 0.798     |            |            | 2026-09-04 12:29 |
| granite3.1-dense:2b | 131072 | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **259.7**    | **128** | **507**    | **108** | **0.844** | **+30.1%** | -5.5%      | 2026-09-04 12:29 |
| granite3.1-dense:2b | 131072 | medium | stream     | CPU    | Ollama 0.32.6   | 209.0         | 13.9         | 128     | 10 878     | 221     | 1.727     |            |            | 2026-09-06 00:39 |
| granite3.1-dense:2b | 131072 | medium | stream     | CPU    | **loken 0.1.0** | **103.0**     | **13.6**     | **128** | **10 045** | **429** | **3.352** | -1.9%      | -48.5%     | 2026-09-06 00:39 |
| granite3.1-dense:2b | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 687.6         | 223.8        | 128     | 2 100      | 111     | 0.869     |            |            | 2026-09-04 10:38 |
| granite3.1-dense:2b | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **3 468.5**   | **289.6**    | **128** | **488**    | **109** | **0.855** | **+29.4%** | **+1.7%**  | 2026-09-04 10:38 |
| granite3.1-dense:2b | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 196.8        | 128     | 2 152      | 117     | 0.915     |            |            | 2026-09-04 08:35 |
| granite3.1-dense:2b | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **262.6**    | **128** | **504**    | **97**  | **0.759** | **+33.4%** | **+20.6%** | 2026-09-04 08:35 |
| granite3.1-dense:2b | 131072 | short  | stream     | CPU    | Ollama 0.32.6   | 86.5          | 14.0         | 128     | 10 695     | 221     | 1.728     |            |            | 2026-09-05 21:45 |
| granite3.1-dense:2b | 131072 | short  | stream     | CPU    | **loken 0.1.0** | **47.5**      | **13.3**     | **128** | **10 125** | **438** | **3.423** | -4.9%      | -49.5%     | 2026-09-05 21:45 |
| granite3.1-dense:2b | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 224.3         | 224.6        | 128     | 2 054      | 105     | 0.824     |            |            | 2026-09-04 02:41 |
| granite3.1-dense:2b | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **1 128.3**   | **292.7**    | **128** | **484**    | **104** | **0.815** | **+30.3%** | **+1.2%**  | 2026-09-04 02:41 |

### granitemoe

![granitemoe](img/family-granitemoe.svg)

| Model           | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token   | Δ decode   | Δ energy   | Date             |
|-----------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|-----------|------------|------------|------------------|
| granite3-moe:1b | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 280.5        | 128     | 529       | 55      | 0.428     |            |            | 2026-09-04 00:36 |
| granite3-moe:1b | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **279.4**    | **128** | **503**   | **57**  | **0.443** | -0.4%      | -3.5%      | 2026-09-04 00:36 |
| granite3-moe:1b | 4096   | long   | stream     | CPU    | Ollama 0.32.6   | 1 643.3       | 68.8         | 128     | 2 871     | 49      | 0.385     |            |            | 2026-09-05 06:45 |
| granite3-moe:1b | 4096   | long   | stream     | CPU    | **loken 0.1.0** | **447.9**     | **50.7**     | **128** | **3 190** | **194** | **1.518** | -26.4%     | -74.6%     | 2026-09-05 06:45 |
| granite3-moe:1b | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 1 955.1       | 328.6        | 128     | 534       | 54      | 0.422     |            |            | 2026-09-03 22:58 |
| granite3-moe:1b | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **4 288.6**   | **319.6**    | **128** | **500**   | **63**  | **0.492** | -2.7%      | -14.2%     | 2026-09-03 22:58 |
| granite3-moe:1b | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 270.2        | 128     | 500       | 46      | 0.363     |            |            | 2026-09-03 21:19 |
| granite3-moe:1b | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **291.5**    | **128** | **450**   | **48**  | **0.374** | **+7.9%**  | -3.0%      | 2026-09-03 21:19 |
| granite3-moe:1b | 4096   | medium | stream     | CPU    | Ollama 0.32.6   | 340.5         | 71.8         | 128     | 2 658     | 48      | 0.377     |            |            | 2026-09-05 01:40 |
| granite3-moe:1b | 4096   | medium | stream     | CPU    | **loken 0.1.0** | **204.4**     | **58.5**     | **128** | **2 431** | **147** | **1.145** | -18.5%     | -67.1%     | 2026-09-05 01:40 |
| granite3-moe:1b | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 695.7         | 337.9        | 128     | 513       | 49      | 0.383     |            |            | 2026-09-03 19:36 |
| granite3-moe:1b | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **3 152.2**   | **327.3**    | **128** | **462**   | **46**  | **0.360** | -3.1%      | **+6.3%**  | 2026-09-03 19:36 |
| granite3-moe:1b | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 266.4        | 128     | 513       | 46      | 0.359     |            |            | 2026-09-03 17:58 |
| granite3-moe:1b | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **319.9**    | **128** | **440**   | **41**  | **0.321** | **+20.1%** | **+11.9%** | 2026-09-03 17:58 |
| granite3-moe:1b | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 104.8         | 72.7         | 128     | 2 643     | 49      | 0.380     |            |            | 2026-09-04 21:06 |
| granite3-moe:1b | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **109.5**     | **59.7**     | **128** | **2 276** | **134** | **1.044** | -17.8%     | -63.6%     | 2026-09-04 21:06 |
| granite3-moe:1b | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 248.5         | 339.1        | 128     | 517       | 47      | 0.367     |            |            | 2026-09-03 16:22 |
| granite3-moe:1b | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **1 045.3**   | **323.0**    | **128** | **476**   | **48**  | **0.376** | -4.7%      | -2.3%      | 2026-09-03 16:22 |
| granite3-moe:1b | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 280.9        | 128     | 516       | 50      | 0.394     |            |            | 2026-09-04 16:20 |
| granite3-moe:1b | 131072 | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **281.5**    | **128** | **493**   | **66**  | **0.519** | **+0.2%**  | -24.1%     | 2026-09-04 16:20 |
| granite3-moe:1b | 131072 | long   | stream     | CPU    | Ollama 0.32.6   | 1 774.0       | 68.9         | 128     | 2 817     | 50      | 0.394     |            |            | 2026-09-06 03:35 |
| granite3-moe:1b | 131072 | long   | stream     | CPU    | **loken 0.1.0** | **434.8**     | **50.6**     | **128** | **3 183** | **192** | **1.502** | -26.5%     | -73.8%     | 2026-09-06 03:35 |
| granite3-moe:1b | 131072 | long   | stream     | GPU    | Ollama 0.32.6   | 4 300.5       | 279.3        | 128     | 566       | 55      | 0.427     |            |            | 2026-09-04 14:18 |
| granite3-moe:1b | 131072 | long   | stream     | GPU    | **loken 0.1.0** | **3 447.5**   | **176.8**    | **128** | **901**   | **80**  | **0.628** | -36.7%     | -31.9%     | 2026-09-04 14:18 |
| granite3-moe:1b | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 271.0        | 128     | 516       | 50      | 0.390     |            |            | 2026-09-04 12:28 |
| granite3-moe:1b | 131072 | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **303.4**    | **128** | **464**   | **53**  | **0.418** | **+12.0%** | -6.6%      | 2026-09-04 12:28 |
| granite3-moe:1b | 131072 | medium | stream     | CPU    | Ollama 0.32.6   | 271.5         | 71.9         | 128     | 2 662     | 47      | 0.367     |            |            | 2026-09-06 00:37 |
| granite3-moe:1b | 131072 | medium | stream     | CPU    | **loken 0.1.0** | **223.1**     | **58.4**     | **128** | **2 422** | **139** | **1.088** | -18.7%     | -66.3%     | 2026-09-06 00:37 |
| granite3-moe:1b | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 707.6         | 336.8        | 128     | 487       | 55      | 0.428     |            |            | 2026-09-04 10:38 |
| granite3-moe:1b | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **2 846.4**   | **326.9**    | **128** | **449**   | **41**  | **0.318** | -3.0%      | **+34.7%** | 2026-09-04 10:38 |
| granite3-moe:1b | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 280.2        | 128     | 534       | 46      | 0.360     |            |            | 2026-09-04 08:34 |
| granite3-moe:1b | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **296.3**    | **128** | **472**   | **49**  | **0.380** | **+5.7%**  | -5.3%      | 2026-09-04 08:34 |
| granite3-moe:1b | 131072 | short  | stream     | CPU    | Ollama 0.32.6   | 95.2          | 73.0         | 128     | 2 645     | 48      | 0.376     |            |            | 2026-09-05 21:43 |
| granite3-moe:1b | 131072 | short  | stream     | CPU    | **loken 0.1.0** | **108.1**     | **60.0**     | **128** | **2 295** | **136** | **1.061** | -17.8%     | -64.6%     | 2026-09-05 21:43 |
| granite3-moe:1b | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 252.3         | 334.9        | 128     | 518       | 50      | 0.390     |            |            | 2026-09-04 02:40 |
| granite3-moe:1b | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **1 028.0**   | **324.2**    | **128** | **468**   | **40**  | **0.313** | -3.2%      | **+24.7%** | 2026-09-04 02:40 |

### lfm2

![lfm2](img/family-lfm2.svg)

| Model                  | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token   | Δ decode   | Δ energy    | Date             |
|------------------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|-----------|------------|-------------|------------------|
| lfm2.5-thinking:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 315.7        | 128     | 1 615     | 69      | 0.538     |            |             | 2026-09-04 00:38 |
| lfm2.5-thinking:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **561.7**    | **128** | **430**   | **53**  | **0.411** | **+77.9%** | **+31.1%**  | 2026-09-04 00:38 |
| lfm2.5-thinking:latest | 4096   | long   | stream     | CPU    | Ollama 0.32.6   | 1 224.7       | 29.8         | 128     | 5 391     | 108     | 0.841     |            |             | 2026-09-05 06:48 |
| lfm2.5-thinking:latest | 4096   | long   | stream     | CPU    | **loken 0.1.0** | **2 341.2**   | **29.5**     | **128** | **4 777** | **262** | **2.044** | -0.9%      | -58.8%      | 2026-09-05 06:48 |
| lfm2.5-thinking:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 2 358.6       | 416.9        | 128     | 1 653     | 59      | 0.463     |            |             | 2026-09-03 23:00 |
| lfm2.5-thinking:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **33 797.8**  | **664.9**    | **128** | **435**   | **32**  | **0.254** | **+59.5%** | **+82.3%**  | 2026-09-03 23:00 |
| lfm2.5-thinking:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 306.6        | 128     | 1 646     | 61      | 0.479     |            |             | 2026-09-03 21:21 |
| lfm2.5-thinking:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **538.0**    | **128** | **395**   | **50**  | **0.389** | **+75.5%** | **+23.2%**  | 2026-09-03 21:21 |
| lfm2.5-thinking:latest | 4096   | medium | stream     | CPU    | Ollama 0.32.6   | 196.6         | 30.0         | 128     | 5 198     | 107     | 0.836     |            |             | 2026-09-05 01:43 |
| lfm2.5-thinking:latest | 4096   | medium | stream     | CPU    | **loken 0.1.0** | **408.9**     | **30.0**     | **128** | **4 535** | **260** | **2.033** | **+0.0%**  | -58.9%      | 2026-09-05 01:43 |
| lfm2.5-thinking:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 436.3         | 394.2        | 128     | 1 668     | 55      | 0.429     |            |             | 2026-09-03 19:38 |
| lfm2.5-thinking:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **5 135.2**   | **668.5**    | **128** | **414**   | **30**  | **0.231** | **+69.6%** | **+85.7%**  | 2026-09-03 19:38 |
| lfm2.5-thinking:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 305.0        | 128     | 1 651     | 52      | 0.409     |            |             | 2026-09-03 18:00 |
| lfm2.5-thinking:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **529.7**    | **128** | **399**   | **48**  | **0.373** | **+73.7%** | **+9.4%**   | 2026-09-03 18:00 |
| lfm2.5-thinking:latest | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 72.6          | 30.0         | 128     | 5 176     | 108     | 0.843     |            |             | 2026-09-04 21:09 |
| lfm2.5-thinking:latest | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **140.7**     | **30.6**     | **128** | **4 406** | **253** | **1.980** | **+2.0%**  | -57.4%      | 2026-09-04 21:09 |
| lfm2.5-thinking:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 116.1         | 397.6        | 128     | 1 665     | 73      | 0.572     |            |             | 2026-09-03 16:24 |
| lfm2.5-thinking:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **606.8**     | **666.9**    | **128** | **430**   | **52**  | **0.403** | **+67.7%** | **+41.7%**  | 2026-09-03 16:24 |
| lfm2.5-thinking:latest | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 317.4        | 128     | 1 712     | 61      | 0.474     |            |             | 2026-09-04 16:22 |
| lfm2.5-thinking:latest | 131072 | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **528.3**    | **128** | **441**   | **46**  | **0.362** | **+66.4%** | **+31.0%**  | 2026-09-04 16:22 |
| lfm2.5-thinking:latest | 131072 | long   | stream     | CPU    | Ollama 0.32.6   | 1 125.9       | 29.9         | 128     | 5 439     | 108     | 0.840     |            |             | 2026-09-06 03:38 |
| lfm2.5-thinking:latest | 131072 | long   | stream     | CPU    | **loken 0.1.0** | **2 608.5**   | **29.5**     | **128** | **4 796** | **258** | **2.016** | -1.3%      | -58.3%      | 2026-09-06 03:38 |
| lfm2.5-thinking:latest | 131072 | long   | stream     | GPU    | Ollama 0.32.6   | 2 027.7       | 404.4        | 128     | 2 256     | 67      | 0.523     |            |             | 2026-09-04 14:20 |
| lfm2.5-thinking:latest | 131072 | long   | stream     | GPU    | **loken 0.1.0** | **32 590.7**  | **656.1**    | **128** | **430**   | **29**  | **0.229** | **+62.2%** | **+128.9%** | 2026-09-04 14:20 |
| lfm2.5-thinking:latest | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 281.3        | 128     | 1 705     | 70      | 0.545     |            |             | 2026-09-04 12:30 |
| lfm2.5-thinking:latest | 131072 | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **548.4**    | **128** | **419**   | **50**  | **0.390** | **+95.0%** | **+39.9%**  | 2026-09-04 12:30 |
| lfm2.5-thinking:latest | 131072 | medium | stream     | CPU    | Ollama 0.32.6   | 219.4         | 30.0         | 128     | 5 260     | 104     | 0.815     |            |             | 2026-09-06 00:40 |
| lfm2.5-thinking:latest | 131072 | medium | stream     | CPU    | **loken 0.1.0** | **404.7**     | **30.7**     | **128** | **4 443** | **251** | **1.961** | **+2.0%**  | -58.4%      | 2026-09-06 00:40 |
| lfm2.5-thinking:latest | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 457.4         | 381.5        | 128     | 1 714     | 74      | 0.575     |            |             | 2026-09-04 10:39 |
| lfm2.5-thinking:latest | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **4 881.4**   | **669.2**    | **128** | **390**   | **51**  | **0.397** | **+75.4%** | **+45.0%**  | 2026-09-04 10:39 |
| lfm2.5-thinking:latest | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 272.5        | 128     | 1 835     | 70      | 0.548     |            |             | 2026-09-04 08:36 |
| lfm2.5-thinking:latest | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **526.1**    | **128** | **400**   | **46**  | **0.357** | **+93.1%** | **+53.7%**  | 2026-09-04 08:36 |
| lfm2.5-thinking:latest | 131072 | short  | stream     | CPU    | Ollama 0.32.6   | 76.1          | 30.1         | 128     | 5 239     | 106     | 0.829     |            |             | 2026-09-05 21:46 |
| lfm2.5-thinking:latest | 131072 | short  | stream     | CPU    | **loken 0.1.0** | **164.9**     | **30.5**     | **128** | **4 432** | **246** | **1.922** | **+1.2%**  | -56.9%      | 2026-09-05 21:46 |
| lfm2.5-thinking:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 161.7         | 391.6        | 128     | 1 713     | 70      | 0.547     |            |             | 2026-09-04 02:42 |
| lfm2.5-thinking:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **1 330.1**   | **667.4**    | **128** | **386**   | **34**  | **0.268** | **+70.4%** | **+104.3%** | 2026-09-04 02:42 |

### lfm2moe

![lfm2moe](img/family-lfm2moe.svg)

| Model       | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode   | Δ energy   | Date             |
|-------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|------------|------------|------------------|
| lfm2:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 196.6        | 128     | 3 147      | 118     | 0.918     |            |            | 2026-09-04 00:39 |
| lfm2:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **302.8**    | **128** | **703**    | **80**  | **0.627** | **+54.0%** | **+46.4%** | 2026-09-04 00:39 |
| lfm2:latest | 4096   | long   | stream     | CPU    | Ollama 0.32.6   | 782.2         | 14.8         | 128     | 13 277     | 215     | 1.682     |            |            | 2026-09-05 06:51 |
| lfm2:latest | 4096   | long   | stream     | CPU    | **loken 0.1.0** | **2 037.9**   | **14.3**     | **128** | **12 909** | **218** | **1.704** | -3.7%      | -1.3%      | 2026-09-05 06:51 |
| lfm2:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 2 416.2       | 234.5        | 128     | 3 111      | 118     | 0.923     |            |            | 2026-09-03 23:02 |
| lfm2:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **19 531.2**  | **316.7**    | **128** | **676**    | **94**  | **0.733** | **+35.1%** | **+25.8%** | 2026-09-03 23:02 |
| lfm2:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 197.4        | 128     | 3 118      | 116     | 0.905     |            |            | 2026-09-03 21:23 |
| lfm2:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **135.1**    | **4**   | **249**    | **13**  | **3.328** |            |            | 2026-09-03 21:23 |
| lfm2:latest | 4096   | medium | stream     | CPU    | Ollama 0.32.6   | 152.2         | 14.8         | 5       | 4 724      | 15      | 2.998     |            |            | 2026-09-05 01:45 |
| lfm2:latest | 4096   | medium | stream     | CPU    | **loken 0.1.0** |  -            |  -           | **40**  |  -         |  -      |  -        | incoherent |            | 2026-09-05 01:45 |
| lfm2:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 409.1         | 234.7        | 128     | 3 117      | 122     | 0.951     |            |            | 2026-09-03 19:39 |
| lfm2:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **2 673.4**   | **308.0**    | **4**   | **240**    | **10**  | **2.469** |            |            | 2026-09-03 19:39 |
| lfm2:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent |            | 2026-09-03 18:01 |
| lfm2:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent |            | 2026-09-03 18:01 |
| lfm2:latest | 4096   | short  | stream     | CPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent |            | 2026-09-04 21:11 |
| lfm2:latest | 4096   | short  | stream     | CPU    | **loken 0.1.0** |  -            |  -           | **84**  |  -         |  -      |  -        | incoherent |            | 2026-09-04 21:11 |
| lfm2:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent |            | 2026-09-03 16:25 |
| lfm2:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent |            | 2026-09-03 16:25 |
| lfm2:latest | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 196.7        | 128     | 725        | 116     | 0.905     |            |            | 2026-09-04 16:23 |
| lfm2:latest | 131072 | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **296.6**    | **128** | **689**    | **93**  | **0.726** | **+50.7%** | **+24.7%** | 2026-09-04 16:23 |
| lfm2:latest | 131072 | long   | stream     | CPU    | Ollama 0.32.6   | 764.8         | 14.9         | 128     | 13 332     | 214     | 1.670     |            |            | 2026-09-06 03:41 |
| lfm2:latest | 131072 | long   | stream     | CPU    | **loken 0.1.0** | **1 876.3**   | **14.3**     | **128** | **12 923** | **215** | **1.682** | -3.9%      | -0.8%      | 2026-09-06 03:41 |
| lfm2:latest | 131072 | long   | stream     | GPU    | Ollama 0.32.6   | 2 007.8       | 213.7        | 128     | 838        | 111     | 0.866     |            |            | 2026-09-04 14:21 |
| lfm2:latest | 131072 | long   | stream     | GPU    | **loken 0.1.0** | **16 831.4**  | **314.5**    | **128** | **761**    | **84**  | **0.655** | **+47.2%** | **+32.2%** | 2026-09-04 14:21 |
| lfm2:latest | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           |  -      |  -         |  -      |  -        | incoherent |            | 2026-09-04 12:31 |
| lfm2:latest | 131072 | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **141.8**    | **4**   | **270**    | **11**  | **2.747** |            |            | 2026-09-04 12:31 |
| lfm2:latest | 131072 | medium | stream     | CPU    | Ollama 0.32.6   | 130.7         | 14.8         | 5       | 4 838      | 17      | 3.407     |            |            | 2026-09-06 00:42 |
| lfm2:latest | 131072 | medium | stream     | CPU    | **loken 0.1.0** |  -            |  -           | **40**  |  -         |  -      |  -        | incoherent |            | 2026-09-06 00:42 |
| lfm2:latest | 131072 | medium | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 33      |  -         |  -      |  -        | incoherent |            | 2026-09-04 10:41 |
| lfm2:latest | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **3 033.3**   | **195.8**    | **4**   | **269**    | **10**  | **2.475** |            |            | 2026-09-04 10:41 |
| lfm2:latest | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent |            | 2026-09-04 08:37 |
| lfm2:latest | 131072 | short  | non-stream | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent |            | 2026-09-04 08:37 |
| lfm2:latest | 131072 | short  | stream     | CPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent |            | 2026-09-05 21:48 |
| lfm2:latest | 131072 | short  | stream     | CPU    | **loken 0.1.0** |  -            |  -           | **84**  |  -         |  -      |  -        | incoherent |            | 2026-09-05 21:48 |
| lfm2:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent |            | 2026-09-04 02:43 |
| lfm2:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent |            | 2026-09-04 02:43 |

### llama

![llama](img/family-llama.svg)

| Model                | Ctx    | Prompt | Mode       | Device | Engine                        | Prefill tok/s | Decode tok/s | Tokens  | E2E ms        | J/req      | J/token       | Δ decode     | Δ energy    | Date             |
|----------------------|--------|--------|------------|--------|-------------------------------|---------------|--------------|---------|---------------|------------|---------------|--------------|-------------|------------------|
| deepseek-r1:70b      | 4096   | long   | non-stream | GPU    | Ollama 0.32.6                 |  -            | 1.5          | 128     | 88 699        | 6 585      | 51.449        |              |             | 2026-09-03 23:54 |
| deepseek-r1:70b      | 4096   | long   | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **1.4**      | **128** | **94 368**    | **6 763**  | **52.838**    | -9.5%        | -2.6%       | 2026-09-03 23:54 |
| deepseek-r1:70b      | 4096   | long   | stream     | CPU    | Ollama 0.32.6                 | 107.0         | 0.5          | 128     | 265 836       | 5 587      | 43.648        |              |             | 2026-09-05 05:12 |
| deepseek-r1:70b      | 4096   | long   | stream     | CPU    | **loken 0.1.0**               | ** - **       | **0.0**      | **24**  | **1 855 036** | **40 342** | **1 657.902** |              |             | 2026-09-05 05:12 |
| deepseek-r1:70b      | 4096   | long   | stream     | GPU    | Ollama 0.32.6                 | 248.4         | 1.5          | 128     | 88 730        | 6 609      | 51.632        |              |             | 2026-09-03 22:17 |
| deepseek-r1:70b      | 4096   | long   | stream     | GPU    | **loken 0.1.0**               | **24.0**      | **1.6**      | **128** | **89 227**    | **6 328**  | **49.439**    | **+5.3%**    | **+4.4%**   | 2026-09-03 22:17 |
| deepseek-r1:70b      | 4096   | long   | stream     | GPU    | **loken 0.1.0 + llama3.2:1b** | **23.5**      | **2.0**      | **128** | **80 310**    | **6 143**  | **47.995**    | **+31.5%**   | **+7.6%**   | 2026-09-04 17:16 |
| deepseek-r1:70b      | 4096   | medium | non-stream | GPU    | Ollama 0.32.6                 |  -            | 1.5          | 128     | 88 182        | 6 516      | 50.903        |              |             | 2026-09-03 20:38 |
| deepseek-r1:70b      | 4096   | medium | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **1.4**      | **128** | **93 379**    | **6 966**  | **54.423**    | -9.0%        | -6.5%       | 2026-09-03 20:38 |
| deepseek-r1:70b      | 4096   | medium | stream     | CPU    | Ollama 0.32.6                 | 19.8          | 0.5          | 128     | 268 065       | 5 511      | 43.056        |              |             | 2026-09-05 00:01 |
| deepseek-r1:70b      | 4096   | medium | stream     | CPU    | **loken 0.1.0**               | ** - **       | **0.0**      | **29**  | **1 514 824** | **40 314** | **1 390.129** |              |             | 2026-09-05 00:01 |
| deepseek-r1:70b      | 4096   | medium | stream     | GPU    | Ollama 0.32.6                 | 46.1          | 1.6          | 128     | 88 248        | 6 556      | 51.222        |              |             | 2026-09-03 18:55 |
| deepseek-r1:70b      | 4096   | medium | stream     | GPU    | **loken 0.1.0**               | **20.8**      | **1.5**      | **128** | **91 081**    | **6 851**  | **53.520**    | -5.3%        | -4.3%       | 2026-09-03 18:55 |
| deepseek-r1:70b      | 4096   | medium | stream     | GPU    | **loken 0.1.0 + llama3.2:1b** | **20.1**      | **2.4**      | **128** | **67 571**    | **4 738**  | **37.019**    | **+57.0%**   | **+38.4%**  | 2026-09-04 17:09 |
| deepseek-r1:70b      | 4096   | short  | non-stream | GPU    | Ollama 0.32.6                 |  -            | 1.6          | 128     | 87 920        | 6 570      | 51.329        |              |             | 2026-09-03 17:17 |
| deepseek-r1:70b      | 4096   | short  | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **1.5**      | **128** | **89 715**    | **6 798**  | **53.108**    | -5.3%        | -3.3%       | 2026-09-03 17:17 |
| deepseek-r1:70b      | 4096   | short  | stream     | CPU    | Ollama 0.32.6                 | 7.2           | 0.5          | 128     | 252 787       | 5 513      | 43.071        |              |             | 2026-09-04 19:24 |
| deepseek-r1:70b      | 4096   | short  | stream     | CPU    | **loken 0.1.0**               | ** - **       | **0.0**      | **32**  | **1 866 753** | **47 956** | **1 498.622** |              |             | 2026-09-04 19:24 |
| deepseek-r1:70b      | 4096   | short  | stream     | GPU    | Ollama 0.32.6                 | 17.0          | 1.6          | 128     | 87 816        | 6 546      | 51.138        |              |             | 2026-09-03 15:41 |
| deepseek-r1:70b      | 4096   | short  | stream     | GPU    | **loken 0.1.0**               | **13.7**      | **1.5**      | **128** | **90 246**    | **6 687**  | **52.244**    | -3.6%        | -2.1%       | 2026-09-03 15:41 |
| deepseek-r1:70b      | 4096   | short  | stream     | GPU    | **loken 0.1.0 + llama3.2:1b** | **13.5**      | **2.2**      | **128** | **70 934**    | **5 246**  | **40.981**    | **+42.4%**   | **+24.8%**  | 2026-09-04 17:04 |
| deepseek-r1:70b      | 131072 | long   | non-stream | GPU    | Ollama 0.32.6                 |  -            | 0.5          | 128     | 278 682       | 18 130     | 141.638       |              |             | 2026-09-04 15:38 |
| deepseek-r1:70b      | 131072 | long   | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **0.6**      | **128** | **247 674**   | **13 959** | **109.052**   | **+19.9%**   | **+29.9%**  | 2026-09-04 15:38 |
| deepseek-r1:70b      | 131072 | long   | stream     | GPU    | Ollama 0.32.6                 | 155.6         | 0.8          | 128     | 166 393       | 11 111     | 86.801        |              |             | 2026-09-04 13:30 |
| deepseek-r1:70b      | 131072 | long   | stream     | GPU    | **loken 0.1.0**               | **24.4**      | **1.6**      | **128** | **93 195**    | **6 752**  | **52.754**    | **+99.6%**   | **+64.5%**  | 2026-09-04 13:30 |
| deepseek-r1:70b      | 131072 | medium | non-stream | GPU    | Ollama 0.32.6                 |  -            | 0.8          | 128     | 166 036       | 11 177     | 87.320        |              |             | 2026-09-04 11:40 |
| deepseek-r1:70b      | 131072 | medium | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **1.4**      | **128** | **98 527**    | **7 168**  | **56.003**    | **+71.6%**   | **+55.9%**  | 2026-09-04 11:40 |
| deepseek-r1:70b      | 131072 | medium | stream     | GPU    | Ollama 0.32.6                 | 28.2          | 0.8          | 128     | 166 643       | 11 135     | 86.993        |              |             | 2026-09-04 09:42 |
| deepseek-r1:70b      | 131072 | medium | stream     | GPU    | **loken 0.1.0**               | **21.1**      | **1.5**      | **128** | **95 919**    | **6 943**  | **54.246**    | **+79.2%**   | **+60.4%**  | 2026-09-04 09:42 |
| deepseek-r1:70b      | 131072 | short  | non-stream | GPU    | Ollama 0.32.6                 |  -            | 0.7          | 128     | 188 180       | 12 028     | 93.969        |              |             | 2026-09-04 07:45 |
| deepseek-r1:70b      | 131072 | short  | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **0.9**      | **128** | **143 827**   | **9 863**  | **77.053**    | **+24.6%**   | **+22.0%**  | 2026-09-04 07:45 |
| deepseek-r1:70b      | 131072 | short  | stream     | GPU    | Ollama 0.32.6                 | 9.9           | 0.6          | 128     | 223 697       | 14 437     | 112.788       |              |             | 2026-09-04 01:45 |
| deepseek-r1:70b      | 131072 | short  | stream     | GPU    | **loken 0.1.0**               | **12.2**      | **1.5**      | **128** | **105 678**   | **7 000**  | **54.689**    | **+148.0%**  | **+106.2%** | 2026-09-04 01:45 |
| deepseek-r1:70b-q3ks | 4096   | long   | non-stream | GPU    | Ollama 0.32.6                 |  -            | 6.8          | 128     | 23 460        | 2 757      | 21.537        |              |             | 2026-09-03 23:59 |
| deepseek-r1:70b-q3ks | 4096   | long   | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **18.6**     | **128** | **7 187**     | **2 192**  | **17.122**    | **+172.9%**  | **+25.8%**  | 2026-09-03 23:59 |
| deepseek-r1:70b-q3ks | 4096   | long   | stream     | CPU    | Ollama 0.32.6                 | 144.4         | 0.7          | 128     | 197 724       | 3 955      | 30.899        |              |             | 2026-09-05 05:36 |
| deepseek-r1:70b-q3ks | 4096   | long   | stream     | CPU    | **loken 0.1.0**               | **3.7**       | **0.7**      | **128** | **245 040**   | **5 587**  | **43.650**    | -7.2%        | -29.2%      | 2026-09-05 05:36 |
| deepseek-r1:70b-q3ks | 4096   | long   | stream     | GPU    | Ollama 0.32.6                 | 577.9         | 6.9          | 128     | 23 502        | 2 769      | 21.635        |              |             | 2026-09-03 22:21 |
| deepseek-r1:70b-q3ks | 4096   | long   | stream     | GPU    | **loken 0.1.0**               | **296.0**     | **20.5**     | **128** | **7 224**     | **2 200**  | **17.185**    | **+196.4%**  | **+25.9%**  | 2026-09-03 22:21 |
| deepseek-r1:70b-q3ks | 4096   | medium | non-stream | GPU    | Ollama 0.32.6                 |  -            | 6.9          | 128     | 23 242        | 2 706      | 21.138        |              |             | 2026-09-03 20:42 |
| deepseek-r1:70b-q3ks | 4096   | medium | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **20.7**     | **128** | **6 482**     | **2 022**  | **15.796**    | **+202.4%**  | **+33.8%**  | 2026-09-03 20:42 |
| deepseek-r1:70b-q3ks | 4096   | medium | stream     | CPU    | Ollama 0.32.6                 | 17.8          | 0.5          | 128     | 270 552       | 8 798      | 68.731        |              |             | 2026-09-05 00:29 |
| deepseek-r1:70b-q3ks | 4096   | medium | stream     | CPU    | **loken 0.1.0**               | **3.4**       | **0.7**      | **128** | **209 091**   | **4 431**  | **34.617**    | **+43.6%**   | **+98.6%**  | 2026-09-05 00:29 |
| deepseek-r1:70b-q3ks | 4096   | medium | stream     | GPU    | Ollama 0.32.6                 | 96.2          | 6.9          | 128     | 23 266        | 2 725      | 21.291        |              |             | 2026-09-03 18:59 |
| deepseek-r1:70b-q3ks | 4096   | medium | stream     | GPU    | **loken 0.1.0**               | **229.2**     | **21.2**     | **128** | **6 406**     | **2 019**  | **15.776**    | **+205.3%**  | **+35.0%**  | 2026-09-03 18:59 |
| deepseek-r1:70b-q3ks | 4096   | short  | non-stream | GPU    | Ollama 0.32.6                 |  -            | 6.8          | 128     | 23 222        | 2 727      | 21.301        |              |             | 2026-09-03 17:21 |
| deepseek-r1:70b-q3ks | 4096   | short  | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **21.0**     | **128** | **6 449**     | **1 999**  | **15.615**    | **+207.5%**  | **+36.4%**  | 2026-09-03 17:21 |
| deepseek-r1:70b-q3ks | 4096   | short  | stream     | CPU    | Ollama 0.32.6                 | 9.7           | 0.7          | 128     | 178 714       | 3 957      | 30.912        |              |             | 2026-09-04 19:44 |
| deepseek-r1:70b-q3ks | 4096   | short  | stream     | CPU    | **loken 0.1.0**               | **2.9**       | **0.7**      | **128** | **185 414**   | **4 251**  | **33.215**    | -4.5%        | -6.9%       | 2026-09-04 19:44 |
| deepseek-r1:70b-q3ks | 4096   | short  | stream     | GPU    | Ollama 0.32.6                 | 37.3          | 6.9          | 128     | 23 315        | 2 749      | 21.475        |              |             | 2026-09-03 15:45 |
| deepseek-r1:70b-q3ks | 4096   | short  | stream     | GPU    | **loken 0.1.0**               | **133.3**     | **21.3**     | **128** | **6 403**     | **2 014**  | **15.737**    | **+206.5%**  | **+36.5%**  | 2026-09-03 15:45 |
| deepseek-r1:70b-q3ks | 131072 | long   | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **18.6**     | **128** | **7 152**     | **2 164**  | **16.909**    |              |             | 2026-09-04 15:41 |
| deepseek-r1:70b-q3ks | 131072 | long   | stream     | GPU    | Ollama 0.32.6                 | 205.4         | 1.1          | 128     | 119 483       | 7 454      | 58.235        |              |             | 2026-09-04 13:39 |
| deepseek-r1:70b-q3ks | 131072 | long   | stream     | GPU    | **loken 0.1.0**               | **294.9**     | **20.5**     | **128** | **7 199**     | **2 194**  | **17.139**    | **+1679.4%** | **+239.8%** | 2026-09-04 13:39 |
| deepseek-r1:70b-q3ks | 131072 | medium | non-stream | GPU    | Ollama 0.32.6                 |  -            | 1.2          | 128     | 117 532       | 7 721      | 60.319        |              |             | 2026-09-04 11:49 |
| deepseek-r1:70b-q3ks | 131072 | medium | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **20.7**     | **128** | **6 477**     | **2 016**  | **15.750**    | **+1695.9%** | **+283.0%** | 2026-09-04 11:49 |
| deepseek-r1:70b-q3ks | 131072 | medium | stream     | GPU    | Ollama 0.32.6                 | 33.8          | 0.8          | 128     | 176 646       | 11 153     | 87.135        |              |             | 2026-09-04 09:54 |
| deepseek-r1:70b-q3ks | 131072 | medium | stream     | GPU    | **loken 0.1.0**               | **224.5**     | **20.9**     | **128** | **6 907**     | **2 108**  | **16.467**    | **+2642.4%** | **+429.2%** | 2026-09-04 09:54 |
| deepseek-r1:70b-q3ks | 131072 | short  | non-stream | GPU    | Ollama 0.32.6                 |  -            | 1.2          | 128     | 124 769       | 7 981      | 62.351        |              |             | 2026-09-04 07:55 |
| deepseek-r1:70b-q3ks | 131072 | short  | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **21.1**     | **128** | **6 385**     | **2 013**  | **15.728**    | **+1688.2%** | **+296.4%** | 2026-09-04 07:55 |
| deepseek-r1:70b-q3ks | 131072 | short  | stream     | GPU    | Ollama 0.32.6                 | 10.2          | 1.0          | 128     | 135 561       | 8 555      | 66.839        |              |             | 2026-09-04 01:57 |
| deepseek-r1:70b-q3ks | 131072 | short  | stream     | GPU    | **loken 0.1.0**               | **82.2**      | **20.2**     | **128** | **7 082**     | **2 045**  | **15.979**    | **+1909.9%** | **+318.3%** | 2026-09-04 01:57 |
| devstral:24b         | 4096   | long   | non-stream | GPU    | Ollama 0.32.6                 |  -            |  -           | 1       | 2 824         | 31         |  -            | no answer    |             | 2026-09-04 00:13 |
| devstral:24b         | 4096   | medium | non-stream | GPU    | Ollama 0.32.6                 |  -            | 34.0         | 128     | 6 203         | 773        | 6.038         |              |             | 2026-09-03 20:57 |
| devstral:24b         | 4096   | medium | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **52.6**     | **128** | **2 494**     | **668**    | **5.216**     | **+55.0%**   | **+15.8%**  | 2026-09-03 20:57 |
| devstral:24b         | 4096   | medium | stream     | CPU    | Ollama 0.32.6                 | 48.6          | 1.6          | 128     | 84 967        | 1 830      | 14.294        |              |             | 2026-09-05 00:47 |
| devstral:24b         | 4096   | medium | stream     | CPU    | **loken 0.1.0**               | **20.2**      | **1.6**      | **128** | **84 223**    | **1 903**  | **14.866**    | -0.4%        | -3.8%       | 2026-09-05 00:47 |
| devstral:24b         | 4096   | medium | stream     | GPU    | Ollama 0.32.6                 | 170.7         | 36.2         | 128     | 6 173         | 771        | 6.024         |              |             | 2026-09-03 19:14 |
| devstral:24b         | 4096   | medium | stream     | GPU    | **loken 0.1.0**               | **912.2**     | **53.1**     | **128** | **2 511**     | **666**    | **5.205**     | **+46.5%**   | **+15.7%**  | 2026-09-03 19:14 |
| devstral:24b         | 4096   | short  | non-stream | GPU    | Ollama 0.32.6                 |  -            | 33.7         | 128     | 6 294         | 767        | 5.994         |              |             | 2026-09-03 17:35 |
| devstral:24b         | 4096   | short  | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **51.6**     | **128** | **2 544**     | **674**    | **5.268**     | **+53.1%**   | **+13.8%**  | 2026-09-03 17:35 |
| devstral:24b         | 4096   | short  | stream     | CPU    | Ollama 0.32.6                 | 15.8          | 1.1          | 128     | 119 255       | 2 993      | 23.386        |              |             | 2026-09-04 20:11 |
| devstral:24b         | 4096   | short  | stream     | CPU    | **loken 0.1.0**               | **11.7**      | **1.6**      | **128** | **84 113**    | **1 872**  | **14.628**    | **+44.9%**   | **+59.9%**  | 2026-09-04 20:11 |
| devstral:24b         | 4096   | short  | stream     | GPU    | Ollama 0.32.6                 | 58.6          | 36.3         | 128     | 6 221         | 782        | 6.108         |              |             | 2026-09-03 15:59 |
| devstral:24b         | 4096   | short  | stream     | GPU    | **loken 0.1.0**               | **246.3**     | **52.0**     | **128** | **2 574**     | **689**    | **5.384**     | **+43.3%**   | **+13.4%**  | 2026-09-03 15:59 |
| devstral:24b         | 131072 | long   | non-stream | GPU    | Ollama 0.32.6                 |  -            |  -           | 1       | 3 745         | 37         |  -            | no answer    |             | 2026-09-04 15:57 |
| devstral:24b         | 131072 | medium | non-stream | GPU    | Ollama 0.32.6                 |  -            | 6.8          | 128     | 22 032        | 1 603      | 12.520        |              |             | 2026-09-04 12:05 |
| devstral:24b         | 131072 | medium | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **52.4**     | **128** | **2 498**     | **672**    | **5.250**     | **+670.0%**  | **+138.5%** | 2026-09-04 12:05 |
| devstral:24b         | 131072 | medium | stream     | CPU    | Ollama 0.32.6                 | 49.4          | 1.6          | 128     | 85 065        | 1 818      | 14.205        |              |             | 2026-09-05 23:41 |
| devstral:24b         | 131072 | medium | stream     | CPU    | **loken 0.1.0**               | **20.8**      | **1.6**      | **128** | **82 822**    | **1 859**  | **14.520**    | -0.3%        | -2.2%       | 2026-09-05 23:41 |
| devstral:24b         | 131072 | medium | stream     | GPU    | Ollama 0.32.6                 | 100.6         | 6.9          | 128     | 22 122        | 1 581      | 12.351        |              |             | 2026-09-04 10:12 |
| devstral:24b         | 131072 | medium | stream     | GPU    | **loken 0.1.0**               | **693.8**     | **52.8**     | **128** | **2 532**     | **676**    | **5.280**     | **+664.3%**  | **+133.9%** | 2026-09-04 10:12 |
| devstral:24b         | 131072 | short  | non-stream | GPU    | Ollama 0.32.6                 |  -            | 6.8          | 128     | 21 939        | 1 614      | 12.606        |              |             | 2026-09-04 08:11 |
| devstral:24b         | 131072 | short  | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **53.5**     | **128** | **2 439**     | **668**    | **5.218**     | **+684.4%**  | **+141.6%** | 2026-09-04 08:11 |
| devstral:24b         | 131072 | short  | stream     | CPU    | Ollama 0.32.6                 | 17.5          | 1.6          | 128     | 84 015        | 1 804      | 14.096        |              |             | 2026-09-05 20:48 |
| devstral:24b         | 131072 | short  | stream     | CPU    | **loken 0.1.0**               | **15.6**      | **1.6**      | **128** | **82 115**    | **1 836**  | **14.343**    | -1.3%        | -1.7%       | 2026-09-05 20:48 |
| devstral:24b         | 131072 | short  | stream     | GPU    | Ollama 0.32.6                 | 37.7          | 6.9          | 128     | 21 920        | 1 611      | 12.585        |              |             | 2026-09-04 02:14 |
| devstral:24b         | 131072 | short  | stream     | GPU    | **loken 0.1.0**               | **352.5**     | **51.9**     | **128** | **2 563**     | **694**    | **5.423**     | **+647.4%**  | **+132.1%** | 2026-09-04 02:14 |
| falcon3:latest       | 4096   | long   | non-stream | GPU    | Ollama 0.32.6                 |  -            | 159.5        | 128     | 2 101         | 110        | 0.856         |              |             | 2026-09-04 00:15 |
| falcon3:latest       | 4096   | long   | stream     | CPU    | Ollama 0.32.6                 | 1 046.1       | 14.2         | 128     | 10 350        | 220        | 1.716         |              |             | 2026-09-05 05:51 |
| falcon3:latest       | 4096   | long   | stream     | GPU    | Ollama 0.32.6                 | 1 920.7       | 208.8        | 128     | 2 091         | 111        | 0.868         |              |             | 2026-09-03 22:37 |
| falcon3:latest       | 4096   | medium | non-stream | GPU    | Ollama 0.32.6                 |  -            | 159.5        | 128     | 2 105         | 107        | 0.839         |              |             | 2026-09-03 20:59 |
| falcon3:latest       | 4096   | medium | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **365.1**    | **128** | **364**       | **65**     | **0.504**     | **+128.9%**  | **+66.4%**  | 2026-09-03 20:59 |
| falcon3:latest       | 4096   | medium | stream     | CPU    | Ollama 0.32.6                 | 136.7         | 14.4         | 128     | 9 958         | 213        | 1.665         |              |             | 2026-09-05 00:49 |
| falcon3:latest       | 4096   | medium | stream     | CPU    | **loken 0.1.0**               | **127.8**     | **14.4**     | **128** | **9 303**     | **432**    | **3.376**     | -0.4%        | -50.7%      | 2026-09-05 00:49 |
| falcon3:latest       | 4096   | medium | stream     | GPU    | Ollama 0.32.6                 | 256.5         | 211.2        | 128     | 2 072         | 103        | 0.803         |              |             | 2026-09-03 19:15 |
| falcon3:latest       | 4096   | medium | stream     | GPU    | **loken 0.1.0**               | **1 879.9**   | **402.7**    | **128** | **365**       | **75**     | **0.588**     | **+90.7%**   | **+36.5%**  | 2026-09-03 19:15 |
| falcon3:latest       | 4096   | short  | non-stream | GPU    | Ollama 0.32.6                 |  -            | 168.7        | 128     | 2 121         | 106        | 0.830         |              |             | 2026-09-03 17:37 |
| falcon3:latest       | 4096   | short  | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **401.3**    | **128** | **357**       | **56**     | **0.437**     | **+137.8%**  | **+89.9%**  | 2026-09-03 17:37 |
| falcon3:latest       | 4096   | short  | stream     | CPU    | Ollama 0.32.6                 | 56.3          | 14.5         | 128     | 9 894         | 211        | 1.650         |              |             | 2026-09-04 20:14 |
| falcon3:latest       | 4096   | short  | stream     | CPU    | **loken 0.1.0**               | **68.5**      | **14.2**     | **128** | **9 005**     | **432**    | **3.376**     | -1.9%        | -51.1%      | 2026-09-04 20:14 |
| falcon3:latest       | 4096   | short  | stream     | GPU    | Ollama 0.32.6                 | 102.1         | 211.5        | 128     | 2 074         | 104        | 0.815         |              |             | 2026-09-03 16:01 |
| falcon3:latest       | 4096   | short  | stream     | GPU    | **loken 0.1.0**               | **593.3**     | **403.9**    | **128** | **368**       | **78**     | **0.609**     | **+91.0%**   | **+33.8%**  | 2026-09-03 16:01 |
| falcon3:latest       | 131072 | long   | non-stream | GPU    | Ollama 0.32.6                 |  -            | 167.6        | 128     | 779           | 109        | 0.854         |              |             | 2026-09-04 15:58 |
| falcon3:latest       | 131072 | long   | stream     | CPU    | Ollama 0.32.6                 | 988.5         | 14.3         | 128     | 10 331        | 216        | 1.688         |              |             | 2026-09-06 02:38 |
| falcon3:latest       | 131072 | long   | stream     | GPU    | Ollama 0.32.6                 | 1 956.7       | 208.8        | 128     | 782           | 101        | 0.787         |              |             | 2026-09-04 13:56 |
| falcon3:latest       | 131072 | medium | non-stream | GPU    | Ollama 0.32.6                 |  -            | 167.6        | 128     | 788           | 106        | 0.825         |              |             | 2026-09-04 12:06 |
| falcon3:latest       | 131072 | medium | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **365.5**    | **128** | **374**       | **72**     | **0.563**     | **+118.1%**  | **+46.4%**  | 2026-09-04 12:06 |
| falcon3:latest       | 131072 | medium | stream     | CPU    | Ollama 0.32.6                 | 149.1         | 14.5         | 128     | 9 905         | 214        | 1.671         |              |             | 2026-09-05 23:44 |
| falcon3:latest       | 131072 | medium | stream     | CPU    | **loken 0.1.0**               | **116.7**     | **14.4**     | **128** | **9 024**     | **444**    | **3.472**     | -0.9%        | -51.9%      | 2026-09-05 23:44 |
| falcon3:latest       | 131072 | medium | stream     | GPU    | Ollama 0.32.6                 | 274.3         | 210.2        | 128     | 783           | 99         | 0.776         |              |             | 2026-09-04 10:14 |
| falcon3:latest       | 131072 | medium | stream     | GPU    | **loken 0.1.0**               | **1 659.0**   | **402.3**    | **128** | **371**       | **71**     | **0.556**     | **+91.4%**   | **+39.7%**  | 2026-09-04 10:14 |
| falcon3:latest       | 131072 | short  | non-stream | GPU    | Ollama 0.32.6                 |  -            | 167.9        | 128     | 770           | 105        | 0.823         |              |             | 2026-09-04 08:13 |
| falcon3:latest       | 131072 | short  | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **399.4**    | **128** | **354**       | **68**     | **0.534**     | **+137.9%**  | **+54.0%**  | 2026-09-04 08:13 |
| falcon3:latest       | 131072 | short  | stream     | CPU    | Ollama 0.32.6                 | 58.2          | 14.5         | 128     | 9 831         | 212        | 1.652         |              |             | 2026-09-05 20:51 |
| falcon3:latest       | 131072 | short  | stream     | CPU    | **loken 0.1.0**               | **63.1**      | **14.2**     | **128** | **8 992**     | **453**    | **3.539**     | -2.1%        | -53.3%      | 2026-09-05 20:51 |
| falcon3:latest       | 131072 | short  | stream     | GPU    | Ollama 0.32.6                 | 100.5         | 211.5        | 128     | 768           | 105        | 0.820         |              |             | 2026-09-04 02:15 |
| falcon3:latest       | 131072 | short  | stream     | GPU    | **loken 0.1.0**               | **734.1**     | **403.8**    | **128** | **366**       | **77**     | **0.605**     | **+90.9%**   | **+35.5%**  | 2026-09-04 02:15 |
| llama3.2:1b          | 4096   | long   | non-stream | GPU    | Ollama 0.32.6                 |  -            | 172.4        | 128     | 2 083         | 102        | 0.800         |              |             | 2026-09-04 00:40 |
| llama3.2:1b          | 4096   | long   | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **368.3**    | **128** | **356**       | **62**     | **0.483**     | **+113.7%**  | **+65.7%**  | 2026-09-04 00:40 |
| llama3.2:1b          | 4096   | long   | stream     | CPU    | Ollama 0.32.6                 | 679.9         | 16.2         | 128     | 9 191         | 196        | 1.529         |              |             | 2026-09-05 06:52 |
| llama3.2:1b          | 4096   | long   | stream     | CPU    | **loken 0.1.0**               | **288.6**     | **21.2**     | **128** | **6 883**     | **404**    | **3.155**     | **+30.7%**   | -51.5%      | 2026-09-05 06:52 |
| llama3.2:1b          | 4096   | long   | stream     | GPU    | Ollama 0.32.6                 | 854.0         | 243.9        | 128     | 2 090         | 102        | 0.795         |              |             | 2026-09-03 23:03 |
| llama3.2:1b          | 4096   | long   | stream     | GPU    | **loken 0.1.0**               | **7 748.2**   | **420.5**    | **128** | **370**       | **63**     | **0.491**     | **+72.4%**   | **+61.8%**  | 2026-09-03 23:03 |
| llama3.2:1b          | 4096   | medium | non-stream | GPU    | Ollama 0.32.6                 |  -            | 160.7        | 128     | 2 108         | 102        | 0.801         |              |             | 2026-09-03 21:24 |
| llama3.2:1b          | 4096   | medium | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **400.7**    | **128** | **323**       | **57**     | **0.447**     | **+149.4%**  | **+79.1%**  | 2026-09-03 21:24 |
| llama3.2:1b          | 4096   | medium | stream     | CPU    | Ollama 0.32.6                 | 130.4         | 16.3         | 128     | 9 173         | 201        | 1.571         |              |             | 2026-09-05 01:47 |
| llama3.2:1b          | 4096   | medium | stream     | CPU    | **loken 0.1.0**               | **138.2**     | **16.7**     | **128** | **8 007**     | **458**    | **3.577**     | **+2.3%**    | -56.1%      | 2026-09-05 01:47 |
| llama3.2:1b          | 4096   | medium | stream     | GPU    | Ollama 0.32.6                 | 170.7         | 237.9        | 128     | 2 107         | 102        | 0.798         |              |             | 2026-09-03 19:41 |
| llama3.2:1b          | 4096   | medium | stream     | GPU    | **loken 0.1.0**               | **2 022.5**   | **463.6**    | **128** | **324**       | **56**     | **0.439**     | **+94.9%**   | **+81.8%**  | 2026-09-03 19:41 |
| llama3.2:1b          | 4096   | short  | non-stream | GPU    | Ollama 0.32.6                 |  -            | 165.2        | 128     | 2 093         | 102        | 0.799         |              |             | 2026-09-03 18:02 |
| llama3.2:1b          | 4096   | short  | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **407.7**    | **128** | **326**       | **61**     | **0.473**     | **+146.8%**  | **+69.1%**  | 2026-09-03 18:02 |
| llama3.2:1b          | 4096   | short  | stream     | CPU    | Ollama 0.32.6                 | 48.3          | 16.3         | 128     | 8 984         | 196        | 1.532         |              |             | 2026-09-04 21:13 |
| llama3.2:1b          | 4096   | short  | stream     | CPU    | **loken 0.1.0**               | **65.9**      | **16.6**     | **128** | **8 005**     | **463**    | **3.614**     | **+1.7%**    | -57.6%      | 2026-09-04 21:13 |
| llama3.2:1b          | 4096   | short  | stream     | GPU    | Ollama 0.32.6                 | 71.3          | 232.7        | 128     | 2 068         | 104        | 0.814         |              |             | 2026-09-03 16:26 |
| llama3.2:1b          | 4096   | short  | stream     | GPU    | **loken 0.1.0**               | **611.3**     | **466.6**    | **128** | **323**       | **69**     | **0.536**     | **+100.6%**  | **+51.9%**  | 2026-09-03 16:26 |
| llama3.2:1b          | 131072 | long   | non-stream | GPU    | Ollama 0.32.6                 |  -            | 166.7        | 128     | 2 259         | 100        | 0.784         |              |             | 2026-09-04 16:24 |
| llama3.2:1b          | 131072 | long   | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **403.6**    | **128** | **339**       | **50**     | **0.389**     | **+142.1%**  | **+101.6%** | 2026-09-04 16:24 |
| llama3.2:1b          | 131072 | long   | stream     | CPU    | Ollama 0.32.6                 | 669.3         | 16.3         | 128     | 9 332         | 194        | 1.519         |              |             | 2026-09-06 03:43 |
| llama3.2:1b          | 131072 | long   | stream     | CPU    | **loken 0.1.0**               | **289.3**     | **21.3**     | **128** | **6 896**     | **407**    | **3.180**     | **+30.8%**   | -52.2%      | 2026-09-06 03:43 |
| llama3.2:1b          | 131072 | long   | stream     | GPU    | Ollama 0.32.6                 | 1 021.5       | 238.2        | 128     | 2 287         | 105        | 0.822         |              |             | 2026-09-04 14:22 |
| llama3.2:1b          | 131072 | long   | stream     | GPU    | **loken 0.1.0**               | **8 720.0**   | **412.8**    | **128** | **368**       | **58**     | **0.456**     | **+73.3%**   | **+80.3%**  | 2026-09-04 14:22 |
| llama3.2:1b          | 131072 | medium | non-stream | GPU    | Ollama 0.32.6                 |  -            | 162.0        | 128     | 2 278         | 103        | 0.807         |              |             | 2026-09-04 12:32 |
| llama3.2:1b          | 131072 | medium | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **427.6**    | **128** | **318**       | **68**     | **0.529**     | **+164.0%**  | **+52.6%**  | 2026-09-04 12:32 |
| llama3.2:1b          | 131072 | medium | stream     | CPU    | Ollama 0.32.6                 | 118.3         | 16.4         | 128     | 9 276         | 195        | 1.523         |              |             | 2026-09-06 00:44 |
| llama3.2:1b          | 131072 | medium | stream     | CPU    | **loken 0.1.0**               | **134.3**     | **16.7**     | **128** | **8 002**     | **462**    | **3.610**     | **+1.4%**    | -57.8%      | 2026-09-06 00:44 |
| llama3.2:1b          | 131072 | medium | stream     | GPU    | Ollama 0.32.6                 | 190.1         | 222.1        | 128     | 2 293         | 97         | 0.757         |              |             | 2026-09-04 10:42 |
| llama3.2:1b          | 131072 | medium | stream     | GPU    | **loken 0.1.0**               | **2 054.1**   | **463.4**    | **128** | **330**       | **65**     | **0.507**     | **+108.7%**  | **+49.2%**  | 2026-09-04 10:42 |
| llama3.2:1b          | 131072 | short  | non-stream | GPU    | Ollama 0.32.6                 |  -            | 156.8        | 128     | 2 312         | 100        | 0.779         |              |             | 2026-09-04 08:38 |
| llama3.2:1b          | 131072 | short  | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **401.9**    | **128** | **329**       | **61**     | **0.480**     | **+156.4%**  | **+62.3%**  | 2026-09-04 08:38 |
| llama3.2:1b          | 131072 | short  | stream     | CPU    | Ollama 0.32.6                 | 43.9          | 16.5         | 128     | 9 223         | 197        | 1.536         |              |             | 2026-09-05 21:50 |
| llama3.2:1b          | 131072 | short  | stream     | CPU    | **loken 0.1.0**               | **62.2**      | **16.7**     | **128** | **7 955**     | **461**    | **3.600**     | **+1.2%**    | -57.3%      | 2026-09-05 21:50 |
| llama3.2:1b          | 131072 | short  | stream     | GPU    | Ollama 0.32.6                 | 66.6          | 229.4        | 128     | 2 385         | 98         | 0.768         |              |             | 2026-09-04 02:44 |
| llama3.2:1b          | 131072 | short  | stream     | GPU    | **loken 0.1.0**               | **672.9**     | **465.7**    | **128** | **323**       | **65**     | **0.510**     | **+103.0%**  | **+50.7%**  | 2026-09-04 02:44 |
| magistral:latest     | 4096   | long   | non-stream | GPU    | Ollama 0.32.6                 |  -            |  -           | 1       | 2 800         | 33         |  -            | no answer    |             | 2026-09-04 00:42 |
| magistral:latest     | 4096   | medium | non-stream | GPU    | Ollama 0.32.6                 |  -            | 33.9         | 128     | 6 314         | 781        | 6.098         |              |             | 2026-09-03 21:26 |
| magistral:latest     | 4096   | medium | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **52.3**     | **128** | **2 523**     | **667**    | **5.213**     | **+54.1%**   | **+17.0%**  | 2026-09-03 21:26 |
| magistral:latest     | 4096   | medium | stream     | CPU    | Ollama 0.32.6                 | 47.5          | 1.6          | 128     | 86 349        | 1 932      | 15.094        |              |             | 2026-09-05 01:57 |
| magistral:latest     | 4096   | medium | stream     | CPU    | **loken 0.1.0**               | **19.8**      | **1.6**      | **128** | **82 569**    | **1 873**  | **14.636**    | **+3.3%**    | **+3.1%**   | 2026-09-05 01:57 |
| magistral:latest     | 4096   | medium | stream     | GPU    | Ollama 0.32.6                 | 167.2         | 36.3         | 128     | 6 302         | 809        | 6.323         |              |             | 2026-09-03 19:43 |
| magistral:latest     | 4096   | medium | stream     | GPU    | **loken 0.1.0**               | **843.2**     | **52.6**     | **128** | **2 552**     | **676**    | **5.282**     | **+44.9%**   | **+19.7%**  | 2026-09-03 19:43 |
| magistral:latest     | 4096   | short  | non-stream | GPU    | Ollama 0.32.6                 |  -            | 34.0         | 128     | 6 336         | 776        | 6.062         |              |             | 2026-09-03 18:04 |
| magistral:latest     | 4096   | short  | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **52.2**     | **128** | **2 532**     | **674**    | **5.269**     | **+53.4%**   | **+15.1%**  | 2026-09-03 18:04 |
| magistral:latest     | 4096   | short  | stream     | CPU    | Ollama 0.32.6                 | 17.6          | 1.6          | 128     | 85 523        | 1 901      | 14.850        |              |             | 2026-09-04 21:23 |
| magistral:latest     | 4096   | short  | stream     | CPU    | **loken 0.1.0**               | **12.8**      | **1.6**      | **128** | **85 671**    | **1 938**  | **15.141**    | -2.2%        | -1.9%       | 2026-09-04 21:23 |
| magistral:latest     | 4096   | short  | stream     | GPU    | Ollama 0.32.6                 | 61.9          | 36.3         | 128     | 6 342         | 783        | 6.118         |              |             | 2026-09-03 16:28 |
| magistral:latest     | 4096   | short  | stream     | GPU    | **loken 0.1.0**               | **359.4**     | **52.5**     | **128** | **2 541**     | **663**    | **5.177**     | **+44.6%**   | **+18.2%**  | 2026-09-03 16:28 |
| magistral:latest     | 131072 | long   | non-stream | GPU    | Ollama 0.32.6                 |  -            |  -           | 1       | 2 883         | 27         |  -            | no answer    |             | 2026-09-04 16:26 |
| magistral:latest     | 131072 | medium | non-stream | GPU    | Ollama 0.32.6                 |  -            | 33.8         | 128     | 6 255         | 811        | 6.337         |              |             | 2026-09-04 12:34 |
| magistral:latest     | 131072 | medium | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **51.9**     | **128** | **2 526**     | **667**    | **5.215**     | **+53.8%**   | **+21.5%**  | 2026-09-04 12:34 |
| magistral:latest     | 131072 | medium | stream     | CPU    | Ollama 0.32.6                 | 49.8          | 1.6          | 128     | 86 805        | 1 906      | 14.889        |              |             | 2026-09-06 00:54 |
| magistral:latest     | 131072 | medium | stream     | CPU    | **loken 0.1.0**               | **19.8**      | **1.7**      | **128** | **81 830**    | **1 850**  | **14.452**    | **+4.0%**    | **+3.0%**   | 2026-09-06 00:54 |
| magistral:latest     | 131072 | medium | stream     | GPU    | Ollama 0.32.6                 | 172.3         | 36.3         | 128     | 6 230         | 807        | 6.303         |              |             | 2026-09-04 10:44 |
| magistral:latest     | 131072 | medium | stream     | GPU    | **loken 0.1.0**               | **875.8**     | **52.6**     | **128** | **2 536**     | **672**    | **5.247**     | **+45.0%**   | **+20.1%**  | 2026-09-04 10:44 |
| magistral:latest     | 131072 | short  | non-stream | GPU    | Ollama 0.32.6                 |  -            | 33.5         | 128     | 6 277         | 779        | 6.090         |              |             | 2026-09-04 08:40 |
| magistral:latest     | 131072 | short  | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **52.1**     | **128** | **2 522**     | **667**    | **5.210**     | **+55.6%**   | **+16.9%**  | 2026-09-04 08:40 |
| magistral:latest     | 131072 | short  | stream     | CPU    | Ollama 0.32.6                 | 17.4          | 1.6          | 128     | 86 230        | 1 907      | 14.899        |              |             | 2026-09-05 22:00 |
| magistral:latest     | 131072 | short  | stream     | CPU    | **loken 0.1.0**               | **14.4**      | **1.6**      | **128** | **85 311**    | **1 943**  | **15.176**    | -1.7%        | -1.8%       | 2026-09-05 22:00 |
| magistral:latest     | 131072 | short  | stream     | GPU    | Ollama 0.32.6                 | 61.2          | 36.3         | 128     | 6 241         | 778        | 6.079         |              |             | 2026-09-04 02:46 |
| magistral:latest     | 131072 | short  | stream     | GPU    | **loken 0.1.0**               | **366.1**     | **52.5**     | **128** | **2 555**     | **675**    | **5.272**     | **+44.5%**   | **+15.3%**  | 2026-09-04 02:46 |
| mistral-nemo:latest  | 4096   | long   | non-stream | GPU    | Ollama 0.32.6                 |  -            | 60.4         | 128     | 3 918         | 366        | 2.857         |              |             | 2026-09-04 00:43 |
| mistral-nemo:latest  | 4096   | long   | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **99.9**     | **128** | **1 324**     | **306**    | **2.389**     | **+65.3%**   | **+19.6%**  | 2026-09-04 00:43 |
| mistral-nemo:latest  | 4096   | long   | stream     | CPU    | Ollama 0.32.6                 | 394.9         | 3.4          | 128     | 41 689        | 906        | 7.080         |              |             | 2026-09-05 07:00 |
| mistral-nemo:latest  | 4096   | long   | stream     | CPU    | **loken 0.1.0**               | **21.9**      | **3.9**      | **128** | **44 409**    | **1 029**  | **8.040**     | **+14.9%**   | -11.9%      | 2026-09-05 07:00 |
| mistral-nemo:latest  | 4096   | long   | stream     | GPU    | Ollama 0.32.6                 | 1 031.3       | 68.8         | 128     | 3 902         | 372        | 2.909         |              |             | 2026-09-03 23:06 |
| mistral-nemo:latest  | 4096   | long   | stream     | GPU    | **loken 0.1.0**               | **3 323.7**   | **105.2**    | **128** | **1 348**     | **310**    | **2.423**     | **+52.9%**   | **+20.0%**  | 2026-09-03 23:06 |
| mistral-nemo:latest  | 4096   | medium | non-stream | GPU    | Ollama 0.32.6                 |  -            | 60.7         | 128     | 3 926         | 354        | 2.766         |              |             | 2026-09-03 21:27 |
| mistral-nemo:latest  | 4096   | medium | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **102.5**    | **128** | **1 284**     | **295**    | **2.306**     | **+69.0%**   | **+19.9%**  | 2026-09-03 21:27 |
| mistral-nemo:latest  | 4096   | medium | stream     | CPU    | Ollama 0.32.6                 | 75.0          | 3.4          | 128     | 40 203        | 893        | 6.978         |              |             | 2026-09-05 02:02 |
| mistral-nemo:latest  | 4096   | medium | stream     | CPU    | **loken 0.1.0**               | **17.6**      | **3.3**      | **128** | **41 844**    | **958**    | **7.483**     | -2.7%        | -6.7%       | 2026-09-05 02:02 |
| mistral-nemo:latest  | 4096   | medium | stream     | GPU    | Ollama 0.32.6                 | 184.9         | 68.0         | 128     | 3 957         | 359        | 2.805         |              |             | 2026-09-03 19:44 |
| mistral-nemo:latest  | 4096   | medium | stream     | GPU    | **loken 0.1.0**               | **1 326.0**   | **106.0**    | **128** | **1 284**     | **287**    | **2.240**     | **+55.8%**   | **+25.2%**  | 2026-09-03 19:44 |
| mistral-nemo:latest  | 4096   | short  | non-stream | GPU    | Ollama 0.32.6                 |  -            | 60.9         | 128     | 3 911         | 356        | 2.785         |              |             | 2026-09-03 18:05 |
| mistral-nemo:latest  | 4096   | short  | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **103.6**    | **128** | **1 266**     | **283**    | **2.210**     | **+70.3%**   | **+26.0%**  | 2026-09-03 18:05 |
| mistral-nemo:latest  | 4096   | short  | stream     | CPU    | Ollama 0.32.6                 | 26.4          | 3.5          | 128     | 38 871        | 860        | 6.717         |              |             | 2026-09-04 21:28 |
| mistral-nemo:latest  | 4096   | short  | stream     | CPU    | **loken 0.1.0**               | **14.1**      | **3.4**      | **128** | **39 866**    | **913**    | **7.134**     | -4.3%        | -5.8%       | 2026-09-04 21:28 |
| mistral-nemo:latest  | 4096   | short  | stream     | GPU    | Ollama 0.32.6                 | 65.3          | 68.2         | 128     | 3 918         | 355        | 2.770         |              |             | 2026-09-03 16:29 |
| mistral-nemo:latest  | 4096   | short  | stream     | GPU    | **loken 0.1.0**               | **514.4**     | **107.2**    | **128** | **1 270**     | **284**    | **2.220**     | **+57.1%**   | **+24.8%**  | 2026-09-03 16:29 |
| mistral-nemo:latest  | 131072 | long   | non-stream | GPU    | Ollama 0.32.6                 |  -            | 61.2         | 128     | 3 982         | 374        | 2.921         |              |             | 2026-09-04 16:27 |
| mistral-nemo:latest  | 131072 | long   | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **99.9**     | **128** | **1 338**     | **315**    | **2.461**     | **+63.4%**   | **+18.7%**  | 2026-09-04 16:27 |
| mistral-nemo:latest  | 131072 | long   | stream     | CPU    | Ollama 0.32.6                 | 407.2         | 3.4          | 128     | 43 575        | 897        | 7.011         |              |             | 2026-09-06 03:51 |
| mistral-nemo:latest  | 131072 | long   | stream     | CPU    | **loken 0.1.0**               | **21.9**      | **4.0**      | **128** | **43 870**    | **1 013**  | **7.916**     | **+15.9%**   | -11.4%      | 2026-09-06 03:51 |
| mistral-nemo:latest  | 131072 | long   | stream     | GPU    | Ollama 0.32.6                 | 1 047.9       | 68.8         | 128     | 3 984         | 371        | 2.900         |              |             | 2026-09-04 14:26 |
| mistral-nemo:latest  | 131072 | long   | stream     | GPU    | **loken 0.1.0**               | **2 375.5**   | **103.3**    | **128** | **1 392**     | **308**    | **2.409**     | **+50.1%**   | **+20.4%**  | 2026-09-04 14:26 |
| mistral-nemo:latest  | 131072 | medium | non-stream | GPU    | Ollama 0.32.6                 |  -            | 60.6         | 128     | 4 016         | 356        | 2.785         |              |             | 2026-09-04 12:36 |
| mistral-nemo:latest  | 131072 | medium | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **102.4**    | **128** | **1 288**     | **285**    | **2.226**     | **+69.2%**   | **+25.1%**  | 2026-09-04 12:36 |
| mistral-nemo:latest  | 131072 | medium | stream     | CPU    | Ollama 0.32.6                 | 75.7          | 3.5          | 128     | 41 127        | 864        | 6.750         |              |             | 2026-09-06 00:59 |
| mistral-nemo:latest  | 131072 | medium | stream     | CPU    | **loken 0.1.0**               | **18.3**      | **3.5**      | **128** | **39 558**    | **911**    | **7.119**     | -0.3%        | -5.2%       | 2026-09-06 00:59 |
| mistral-nemo:latest  | 131072 | medium | stream     | GPU    | Ollama 0.32.6                 | 158.0         | 68.1         | 128     | 4 008         | 358        | 2.797         |              |             | 2026-09-04 10:45 |
| mistral-nemo:latest  | 131072 | medium | stream     | GPU    | **loken 0.1.0**               | **1 139.9**   | **106.0**    | **128** | **1 296**     | **280**    | **2.187**     | **+55.7%**   | **+27.9%**  | 2026-09-04 10:45 |
| mistral-nemo:latest  | 131072 | short  | non-stream | GPU    | Ollama 0.32.6                 |  -            | 50.2         | 128     | 5 229         | 392        | 3.061         |              |             | 2026-09-04 08:42 |
| mistral-nemo:latest  | 131072 | short  | non-stream | GPU    | **loken 0.1.0**               | ** - **       | **101.5**    | **128** | **1 358**     | **284**    | **2.217**     | **+102.2%**  | **+38.0%**  | 2026-09-04 08:42 |
| mistral-nemo:latest  | 131072 | short  | stream     | CPU    | Ollama 0.32.6                 | 27.0          | 3.5          | 128     | 40 605        | 866        | 6.768         |              |             | 2026-09-05 22:05 |
| mistral-nemo:latest  | 131072 | short  | stream     | CPU    | **loken 0.1.0**               | **15.5**      | **3.5**      | **128** | **38 290**    | **879**    | **6.870**     | -0.8%        | -1.5%       | 2026-09-05 22:05 |
| mistral-nemo:latest  | 131072 | short  | stream     | GPU    | Ollama 0.32.6                 | 64.1          | 68.2         | 128     | 4 021         | 352        | 2.747         |              |             | 2026-09-04 02:47 |
| mistral-nemo:latest  | 131072 | short  | stream     | GPU    | **loken 0.1.0**               | **496.1**     | **107.0**    | **128** | **1 276**     | **281**    | **2.196**     | **+56.8%**   | **+25.1%**  | 2026-09-04 02:47 |

### mistral3

![mistral3](img/family-mistral3.svg)

| Model                   | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms      | J/req     | J/token    | Δ decode     | Δ energy    | Date             |
|-------------------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|-------------|-----------|------------|--------------|-------------|------------------|
| devstral-small-2:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 33.0         | 128     | 6 524       | 764       | 5.968      |              |             | 2026-09-04 00:01 |
| devstral-small-2:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **52.3**     | **128** | **2 661**   | **748**   | **5.842**  | **+58.5%**   | **+2.1%**   | 2026-09-04 00:01 |
| devstral-small-2:latest | 4096   | long   | stream     | CPU    | Ollama 0.32.6   | 251.3         | 1.6          | 128     | 87 507      | 1 823     | 14.239     |              |             | 2026-09-05 05:46 |
| devstral-small-2:latest | 4096   | long   | stream     | CPU    | **loken 0.1.0** | **23.9**      | **1.8**      | **128** | **85 186**  | **1 901** | **14.848** | **+11.2%**   | -4.1%       | 2026-09-05 05:46 |
| devstral-small-2:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 893.5         | 35.6         | 128     | 6 543       | 764       | 5.966      |              |             | 2026-09-03 22:24 |
| devstral-small-2:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **997.6**     | **58.0**     | **128** | **2 668**   | **750**   | **5.858**  | **+63.1%**   | **+1.8%**   | 2026-09-03 22:24 |
| devstral-small-2:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 32.6         | 128     | 6 531       | 760       | 5.941      |              |             | 2026-09-03 20:45 |
| devstral-small-2:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **58.0**     | **128** | **2 399**   | **692**   | **5.405**  | **+77.7%**   | **+9.9%**   | 2026-09-03 20:45 |
| devstral-small-2:latest | 4096   | medium | stream     | CPU    | Ollama 0.32.6   | 48.2          | 1.6          | 128     | 85 815      | 1 847     | 14.432     |              |             | 2026-09-05 00:37 |
| devstral-small-2:latest | 4096   | medium | stream     | CPU    | **loken 0.1.0** | **19.1**      | **3.7**      | **128** | **40 048**  | **841**   | **6.572**  | **+132.2%**  | **+119.6%** | 2026-09-05 00:37 |
| devstral-small-2:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 154.3         | 35.4         | 128     | 6 538       | 765       | 5.974      |              |             | 2026-09-03 19:01 |
| devstral-small-2:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **705.7**     | **59.6**     | **128** | **2 404**   | **695**   | **5.433**  | **+68.6%**   | **+10.0%**  | 2026-09-03 19:01 |
| devstral-small-2:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 33.2         | 128     | 6 497       | 754       | 5.887      |              |             | 2026-09-03 17:23 |
| devstral-small-2:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **59.4**     | **128** | **2 358**   | **682**   | **5.327**  | **+79.1%**   | **+10.5%**  | 2026-09-03 17:23 |
| devstral-small-2:latest | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 17.3          | 1.6          | 128     | 83 095      | 1 823     | 14.242     |              |             | 2026-09-04 19:55 |
| devstral-small-2:latest | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **12.4**      | **1.2**      | **128** | **106 807** | **2 566** | **20.047** | -28.5%       | -29.0%      | 2026-09-04 19:55 |
| devstral-small-2:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 56.8          | 35.5         | 128     | 6 530       | 763       | 5.960      |              |             | 2026-09-03 15:47 |
| devstral-small-2:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **321.8**     | **59.9**     | **128** | **2 383**   | **663**   | **5.180**  | **+68.6%**   | **+15.1%**  | 2026-09-03 15:47 |
| devstral-small-2:latest | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 5.3          | 128     | 27 979      | 1 887     | 14.738     |              |             | 2026-09-04 15:45 |
| devstral-small-2:latest | 131072 | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **51.8**     | **128** | **2 668**   | **772**   | **6.030**  | **+885.8%**  | **+144.4%** | 2026-09-04 15:45 |
| devstral-small-2:latest | 131072 | long   | stream     | CPU    | Ollama 0.32.6   | 260.1         | 1.6          | 128     | 88 854      | 1 820     | 14.218     |              |             | 2026-09-06 02:33 |
| devstral-small-2:latest | 131072 | long   | stream     | CPU    | **loken 0.1.0** | **24.1**      | **1.8**      | **128** | **84 167**  | **1 882** | **14.702** | **+12.2%**   | -3.3%       | 2026-09-06 02:33 |
| devstral-small-2:latest | 131072 | long   | stream     | GPU    | Ollama 0.32.6   | 489.3         | 5.3          | 128     | 27 871      | 1 871     | 14.614     |              |             | 2026-09-04 13:42 |
| devstral-small-2:latest | 131072 | long   | stream     | GPU    | **loken 0.1.0** | **1 006.9**   | **58.1**     | **128** | **2 677**   | **768**   | **6.002**  | **+992.9%**  | **+143.5%** | 2026-09-04 13:42 |
| devstral-small-2:latest | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 5.3          | 128     | 27 898      | 1 898     | 14.831     |              |             | 2026-09-04 11:52 |
| devstral-small-2:latest | 131072 | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **58.6**     | **128** | **2 389**   | **697**   | **5.443**  | **+1014.9%** | **+172.5%** | 2026-09-04 11:52 |
| devstral-small-2:latest | 131072 | medium | stream     | CPU    | Ollama 0.32.6   | 49.1          | 1.6          | 128     | 85 350      | 1 812     | 14.157     |              |             | 2026-09-05 23:32 |
| devstral-small-2:latest | 131072 | medium | stream     | CPU    | **loken 0.1.0** | **20.9**      | **3.8**      | **128** | **38 308**  | **830**   | **6.487**  | **+131.8%**  | **+118.2%** | 2026-09-05 23:32 |
| devstral-small-2:latest | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 55.1          | 1.8          | 128     | 66 652      | 4 878     | 38.110     |              |             | 2026-09-04 09:59 |
| devstral-small-2:latest | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **706.5**     | **59.4**     | **128** | **2 431**   | **733**   | **5.725**  | **+3268.9%** | **+565.6%** | 2026-09-04 09:59 |
| devstral-small-2:latest | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 5.3          | 128     | 27 688      | 1 881     | 14.698     |              |             | 2026-09-04 07:58 |
| devstral-small-2:latest | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **58.5**     | **128** | **2 396**   | **705**   | **5.505**  | **+1010.0%** | **+167.0%** | 2026-09-04 07:58 |
| devstral-small-2:latest | 131072 | short  | stream     | CPU    | Ollama 0.32.6   | 16.9          | 1.6          | 128     | 84 405      | 1 821     | 14.224     |              |             | 2026-09-05 20:38 |
| devstral-small-2:latest | 131072 | short  | stream     | CPU    | **loken 0.1.0** | **14.0**      | **1.6**      | **128** | **83 896**  | **1 841** | **14.384** | -3.3%        | -1.1%       | 2026-09-05 20:38 |
| devstral-small-2:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 31.4          | 5.0          | 128     | 29 326      | 2 032     | 15.876     |              |             | 2026-09-04 02:00 |
| devstral-small-2:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **165.1**     | **56.0**     | **128** | **2 887**   | **754**   | **5.890**  | **+1030.5%** | **+169.5%** | 2026-09-04 02:00 |
| mistral-small3.2:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 34.1         | 128     | 6 402       | 782       | 6.113      |              |             | 2026-09-04 00:45 |
| mistral-small3.2:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **52.9**     | **128** | **2 626**   | **758**   | **5.925**  | **+55.2%**   | **+3.2%**   | 2026-09-04 00:45 |
| mistral-small3.2:latest | 4096   | long   | stream     | CPU    | Ollama 0.32.6   | 257.8         | 1.6          | 128     | 88 700      | 1 876     | 14.654     |              |             | 2026-09-05 07:11 |
| mistral-small3.2:latest | 4096   | long   | stream     | CPU    | **loken 0.1.0** | **25.3**      | **1.7**      | **128** | **84 949**  | **1 927** | **15.051** | **+9.1%**    | -2.6%       | 2026-09-05 07:11 |
| mistral-small3.2:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 894.1         | 36.5         | 128     | 6 425       | 769       | 6.005      |              |             | 2026-09-03 23:08 |
| mistral-small3.2:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **1 024.1**   | **58.2**     | **128** | **2 635**   | **754**   | **5.888**  | **+59.6%**   | **+2.0%**   | 2026-09-03 23:08 |
| mistral-small3.2:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 33.8         | 128     | 6 388       | 773       | 6.040      |              |             | 2026-09-03 21:29 |
| mistral-small3.2:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **58.1**     | **128** | **2 401**   | **719**   | **5.614**  | **+72.0%**   | **+7.6%**   | 2026-09-03 21:29 |
| mistral-small3.2:latest | 4096   | medium | stream     | CPU    | Ollama 0.32.6   | 48.2          | 1.6          | 128     | 84 212      | 1 847     | 14.431     |              |             | 2026-09-05 02:12 |
| mistral-small3.2:latest | 4096   | medium | stream     | CPU    | **loken 0.1.0** | **18.8**      | **1.6**      | **128** | **85 787**  | **1 943** | **15.180** | -3.0%        | -4.9%       | 2026-09-05 02:12 |
| mistral-small3.2:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 154.5         | 36.3         | 128     | 6 387       | 771       | 6.027      |              |             | 2026-09-03 19:46 |
| mistral-small3.2:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **676.8**     | **59.7**     | **128** | **2 475**   | **714**   | **5.578**  | **+64.6%**   | **+8.1%**   | 2026-09-03 19:46 |
| mistral-small3.2:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 33.4         | 128     | 6 409       | 773       | 6.039      |              |             | 2026-09-03 18:07 |
| mistral-small3.2:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **58.2**     | **128** | **2 392**   | **705**   | **5.504**  | **+74.1%**   | **+9.7%**   | 2026-09-03 18:07 |
| mistral-small3.2:latest | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 17.7          | 1.6          | 128     | 83 788      | 1 859     | 14.526     |              |             | 2026-09-04 21:38 |
| mistral-small3.2:latest | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **12.7**      | **1.5**      | **128** | **86 025**  | **1 952** | **15.254** | -4.3%        | -4.8%       | 2026-09-04 21:38 |
| mistral-small3.2:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 55.5          | 36.3         | 128     | 6 384       | 779       | 6.089      |              |             | 2026-09-03 16:31 |
| mistral-small3.2:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **198.2**     | **59.8**     | **128** | **2 392**   | **713**   | **5.570**  | **+64.7%**   | **+9.3%**   | 2026-09-03 16:31 |
| mistral-small3.2:latest | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 5.2          | 128     | 28 368      | 1 911     | 14.933     |              |             | 2026-09-04 16:30 |
| mistral-small3.2:latest | 131072 | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **52.8**     | **128** | **2 624**   | **765**   | **5.976**  | **+909.0%**  | **+149.9%** | 2026-09-04 16:30 |
| mistral-small3.2:latest | 131072 | long   | stream     | CPU    | Ollama 0.32.6   | 262.6         | 1.6          | 128     | 89 336      | 1 845     | 14.413     |              |             | 2026-09-06 04:01 |
| mistral-small3.2:latest | 131072 | long   | stream     | CPU    | **loken 0.1.0** | **25.5**      | **1.7**      | **128** | **85 042**  | **1 908** | **14.909** | **+7.8%**    | -3.3%       | 2026-09-06 04:01 |
| mistral-small3.2:latest | 131072 | long   | stream     | GPU    | Ollama 0.32.6   | 390.6         | 2.3          | 128     | 57 532      | 3 850     | 30.077     |              |             | 2026-09-04 14:31 |
| mistral-small3.2:latest | 131072 | long   | stream     | GPU    | **loken 0.1.0** | **674.5**     | **46.7**     | **128** | **3 716**   | **826**   | **6.450**  | **+1931.1%** | **+366.3%** | 2026-09-04 14:31 |
| mistral-small3.2:latest | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 5.3          | 128     | 27 935      | 1 901     | 14.850     |              |             | 2026-09-04 12:39 |
| mistral-small3.2:latest | 131072 | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **58.5**     | **128** | **2 396**   | **713**   | **5.570**  | **+1000.7%** | **+166.6%** | 2026-09-04 12:39 |
| mistral-small3.2:latest | 131072 | medium | stream     | CPU    | Ollama 0.32.6   | 48.1          | 1.6          | 128     | 85 827      | 1 825     | 14.255     |              |             | 2026-09-06 01:09 |
| mistral-small3.2:latest | 131072 | medium | stream     | CPU    | **loken 0.1.0** | **19.7**      | **1.6**      | **128** | **85 781**  | **1 925** | **15.039** | -3.6%        | -5.2%       | 2026-09-06 01:09 |
| mistral-small3.2:latest | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 89.2          | 5.3          | 128     | 28 342      | 1 914     | 14.951     |              |             | 2026-09-04 10:48 |
| mistral-small3.2:latest | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **653.9**     | **59.7**     | **128** | **2 440**   | **717**   | **5.598**  | **+1021.5%** | **+167.1%** | 2026-09-04 10:48 |
| mistral-small3.2:latest | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 5.1          | 128     | 37 114      | 1 988     | 15.531     |              |             | 2026-09-04 08:47 |
| mistral-small3.2:latest | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **59.1**     | **128** | **2 741**   | **714**   | **5.578**  | **+1054.2%** | **+178.4%** | 2026-09-04 08:47 |
| mistral-small3.2:latest | 131072 | short  | stream     | CPU    | Ollama 0.32.6   | 17.3          | 1.6          | 128     | 85 158      | 1 836     | 14.342     |              |             | 2026-09-05 22:15 |
| mistral-small3.2:latest | 131072 | short  | stream     | CPU    | **loken 0.1.0** | **14.3**      | **1.6**      | **128** | **85 781**  | **1 916** | **14.971** | -4.8%        | -4.2%       | 2026-09-05 22:15 |
| mistral-small3.2:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 32.0          | 5.3          | 128     | 28 089      | 1 934     | 15.111     |              |             | 2026-09-04 02:50 |
| mistral-small3.2:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **314.8**     | **59.8**     | **128** | **2 409**   | **700**   | **5.467**  | **+1020.4%** | **+176.4%** | 2026-09-04 02:50 |

### nemotron_h_moe

![nemotron_h_moe](img/family-nemotron_h_moe.svg)

| Model                  | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode    | Δ energy   | Date             |
|------------------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|-------------|------------|------------------|
| nemotron-3-nano:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 104.0        | 128     | 4 731      | 187     | 1.460     |             |            | 2026-09-04 00:48 |
| nemotron-3-nano:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **143.6**    | **128** | **1 306**  | **169** | **1.317** | **+38.1%**  | **+10.9%** | 2026-09-04 00:48 |
| nemotron-3-nano:latest | 4096   | long   | stream     | CPU    | Ollama 0.32.6   | 431.9         | 8.3          | 128     | 20 361     | 373     | 2.914     |             |            | 2026-09-05 07:15 |
| nemotron-3-nano:latest | 4096   | long   | stream     | CPU    | **loken 0.1.0** | **3 986.6**   | **7.5**      | **128** | **21 388** | **407** | **3.180** | -9.7%       | -8.4%      | 2026-09-05 07:15 |
| nemotron-3-nano:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 845.8         | 135.3        | 128     | 4 698      | 186     | 1.453     |             |            | 2026-09-03 23:11 |
| nemotron-3-nano:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **15 315.6**  | **149.2**    | **128** | **1 330**  | **167** | **1.305** | **+10.2%**  | **+11.4%** | 2026-09-03 23:11 |
| nemotron-3-nano:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 103.1        | 128     | 4 660      | 177     | 1.380     |             |            | 2026-09-03 21:32 |
| nemotron-3-nano:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **149.2**    | **128** | **1 205**  | **164** | **1.280** | **+44.8%**  | **+7.8%**  | 2026-09-03 21:32 |
| nemotron-3-nano:latest | 4096   | medium | stream     | CPU    | Ollama 0.32.6   | 81.3          | 8.3          | 128     | 19 502     | 375     | 2.928     |             |            | 2026-09-05 02:17 |
| nemotron-3-nano:latest | 4096   | medium | stream     | CPU    | **loken 0.1.0** | **678.5**     | **7.7**      | **128** | **17 467** | **391** | **3.057** | -6.5%       | -4.2%      | 2026-09-05 02:17 |
| nemotron-3-nano:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 137.6         | 136.1        | 128     | 4 633      | 177     | 1.382     |             |            | 2026-09-03 19:49 |
| nemotron-3-nano:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **2 098.0**   | **150.3**    | **128** | **1 203**  | **168** | **1.314** | **+10.4%**  | **+5.2%**  | 2026-09-03 19:49 |
| nemotron-3-nano:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 101.2        | 128     | 4 640      | 183     | 1.426     |             |            | 2026-09-03 18:10 |
| nemotron-3-nano:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **150.7**    | **128** | **1 164**  | **161** | **1.261** | **+48.9%**  | **+13.0%** | 2026-09-03 18:10 |
| nemotron-3-nano:latest | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 28.1          | 8.3          | 127     | 19 139     | 369     | 2.903     |             |            | 2026-09-04 21:43 |
| nemotron-3-nano:latest | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **203.4**     | **7.8**      | **128** | **16 936** | **390** | **3.047** | -6.4%       | -5.4%      | 2026-09-04 21:43 |
| nemotron-3-nano:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 46.5          | 133.9        | 128     | 4 669      | 186     | 1.456     |             |            | 2026-09-03 16:34 |
| nemotron-3-nano:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **753.3**     | **149.8**    | **128** | **1 170**  | **165** | **1.288** | **+11.9%**  | **+13.0%** | 2026-09-03 16:34 |
| nemotron-3-nano:latest | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 102.1        | 128     | 4 803      | 178     | 1.390     |             |            | 2026-09-04 16:33 |
| nemotron-3-nano:latest | 131072 | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **149.4**    | **128** | **1 320**  | **166** | **1.296** | **+46.3%**  | **+7.3%**  | 2026-09-04 16:33 |
| nemotron-3-nano:latest | 131072 | long   | stream     | CPU    | Ollama 0.32.6   | 434.1         | 8.3          | 128     | 19 892     | 371     | 2.902     |             |            | 2026-09-06 04:06 |
| nemotron-3-nano:latest | 131072 | long   | stream     | CPU    | **loken 0.1.0** | **8 220.0**   | **7.6**      | **128** | **21 177** | **401** | **3.131** | -9.2%       | -7.3%      | 2026-09-06 04:06 |
| nemotron-3-nano:latest | 131072 | long   | stream     | GPU    | Ollama 0.32.6   | 887.9         | 127.1        | 128     | 6 677      | 193     | 1.507     |             |            | 2026-09-04 14:34 |
| nemotron-3-nano:latest | 131072 | long   | stream     | GPU    | **loken 0.1.0** | **5 458.4**   | **135.1**    | **128** | **1 525**  | **185** | **1.446** | **+6.3%**   | **+4.2%**  | 2026-09-04 14:34 |
| nemotron-3-nano:latest | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 102.6        | 128     | 4 740      | 179     | 1.401     |             |            | 2026-09-04 12:42 |
| nemotron-3-nano:latest | 131072 | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **148.9**    | **128** | **1 180**  | **160** | **1.249** | **+45.1%**  | **+12.2%** | 2026-09-04 12:42 |
| nemotron-3-nano:latest | 131072 | medium | stream     | CPU    | Ollama 0.32.6   | 85.1          | 8.3          | 128     | 18 962     | 372     | 2.908     |             |            | 2026-09-06 01:14 |
| nemotron-3-nano:latest | 131072 | medium | stream     | CPU    | **loken 0.1.0** | **629.5**     | **7.8**      | **128** | **17 248** | **387** | **3.021** | -6.4%       | -3.7%      | 2026-09-06 01:14 |
| nemotron-3-nano:latest | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 114.9         | 136.4        | 128     | 4 769      | 191     | 1.489     |             |            | 2026-09-04 10:51 |
| nemotron-3-nano:latest | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **983.6**     | **149.4**    | **128** | **1 219**  | **164** | **1.285** | **+9.6%**   | **+15.9%** | 2026-09-04 10:51 |
| nemotron-3-nano:latest | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 72.1         | 128     | 7 841      | 217     | 1.697     |             |            | 2026-09-04 08:51 |
| nemotron-3-nano:latest | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **150.2**    | **128** | **1 236**  | **159** | **1.239** | **+108.3%** | **+36.9%** | 2026-09-04 08:51 |
| nemotron-3-nano:latest | 131072 | short  | stream     | CPU    | Ollama 0.32.6   | 26.8          | 8.4          | 127     | 18 604     | 371     | 2.925     |             |            | 2026-09-05 22:20 |
| nemotron-3-nano:latest | 131072 | short  | stream     | CPU    | **loken 0.1.0** | **204.9**     | **7.9**      | **128** | **16 693** | **387** | **3.024** | -6.0%       | -4.0%      | 2026-09-05 22:20 |
| nemotron-3-nano:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 47.4          | 134.1        | 128     | 4 733      | 174     | 1.361     |             |            | 2026-09-04 02:54 |
| nemotron-3-nano:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **902.0**     | **150.5**    | **128** | **1 226**  | **163** | **1.274** | **+12.3%**  | **+6.8%**  | 2026-09-04 02:54 |

### olmo2

![olmo2](img/family-olmo2.svg)

| Model    | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode   | Δ energy   | Date             |
|----------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|------------|------------|------------------|
| olmo2:7b | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 89.3         | 128     | 1 460      | 268     | 2.095     |            |            | 2026-09-04 00:49 |
| olmo2:7b | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **111.1**    | **128** | **1 209**  | **273** | **2.130** | **+24.4%** | -1.6%      | 2026-09-04 00:49 |
| olmo2:7b | 4096   | long   | stream     | CPU    | Ollama 0.32.6   | 589.4         | 5.1          | 128     | 28 152     | 603     | 4.710     |            |            | 2026-09-05 07:19 |
| olmo2:7b | 4096   | long   | stream     | CPU    | **loken 0.1.0** | **71.5**      | **6.3**      | **128** | **24 358** | **558** | **4.360** | **+23.9%** | **+8.0%**  | 2026-09-05 07:19 |
| olmo2:7b | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 1 764.8       | 98.4         | 128     | 1 474      | 272     | 2.127     |            |            | 2026-09-03 23:12 |
| olmo2:7b | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **3 703.3**   | **116.5**    | **128** | **1 199**  | **272** | **2.122** | **+18.4%** | **+0.2%**  | 2026-09-03 23:12 |
| olmo2:7b | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 89.4         | 128     | 1 456      | 265     | 2.071     |            |            | 2026-09-03 21:34 |
| olmo2:7b | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **134.1**    | **128** | **993**    | **228** | **1.777** | **+49.9%** | **+16.5%** | 2026-09-03 21:34 |
| olmo2:7b | 4096   | medium | stream     | CPU    | Ollama 0.32.6   | 106.1         | 5.2          | 128     | 26 505     | 584     | 4.559     |            |            | 2026-09-05 02:21 |
| olmo2:7b | 4096   | medium | stream     | CPU    | **loken 0.1.0** | **52.8**      | **4.7**      | **128** | **28 757** | **659** | **5.146** | -10.3%     | -11.4%     | 2026-09-05 02:21 |
| olmo2:7b | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 240.6         | 100.8        | 128     | 1 454      | 272     | 2.124     |            |            | 2026-09-03 19:50 |
| olmo2:7b | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 394.3**   | **136.9**    | **128** | **1 016**  | **229** | **1.788** | **+35.8%** | **+18.8%** | 2026-09-03 19:50 |
| olmo2:7b | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 89.5         | 128     | 1 439      | 265     | 2.071     |            |            | 2026-09-03 18:11 |
| olmo2:7b | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **131.5**    | **128** | **1 025**  | **237** | **1.853** | **+47.0%** | **+11.8%** | 2026-09-03 18:11 |
| olmo2:7b | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 37.8          | 5.2          | 128     | 26 280     | 580     | 4.532     |            |            | 2026-09-04 21:47 |
| olmo2:7b | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **28.7**      | **4.7**      | **128** | **28 091** | **641** | **5.007** | -9.6%      | -9.5%      | 2026-09-04 21:47 |
| olmo2:7b | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 88.5          | 101.0        | 128     | 1 426      | 267     | 2.082     |            |            | 2026-09-03 16:35 |
| olmo2:7b | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **603.3**     | **137.6**    | **128** | **998**    | **232** | **1.809** | **+36.3%** | **+15.1%** | 2026-09-03 16:35 |
| olmo2:7b | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 89.9         | 128     | 1 459      | 269     | 2.102     |            |            | 2026-09-04 16:34 |
| olmo2:7b | 131072 | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **111.6**    | **128** | **1 199**  | **269** | **2.099** | **+24.1%** | **+0.1%**  | 2026-09-04 16:34 |
| olmo2:7b | 131072 | long   | stream     | CPU    | Ollama 0.32.6   | 569.4         | 5.1          | 128     | 28 134     | 602     | 4.707     |            |            | 2026-09-06 04:10 |
| olmo2:7b | 131072 | long   | stream     | CPU    | **loken 0.1.0** | **66.4**      | **6.4**      | **128** | **24 021** | **545** | **4.258** | **+27.1%** | **+10.5%** | 2026-09-06 04:10 |
| olmo2:7b | 131072 | long   | stream     | GPU    | Ollama 0.32.6   | 1 544.3       | 97.7         | 128     | 1 473      | 269     | 2.101     |            |            | 2026-09-04 14:35 |
| olmo2:7b | 131072 | long   | stream     | GPU    | **loken 0.1.0** | **3 462.8**   | **116.3**    | **128** | **1 198**  | **277** | **2.164** | **+19.0%** | -2.9%      | 2026-09-04 14:35 |
| olmo2:7b | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 89.0         | 128     | 1 454      | 267     | 2.086     |            |            | 2026-09-04 12:43 |
| olmo2:7b | 131072 | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **134.5**    | **128** | **986**    | **228** | **1.781** | **+51.1%** | **+17.1%** | 2026-09-04 12:43 |
| olmo2:7b | 131072 | medium | stream     | CPU    | Ollama 0.32.6   | 106.2         | 5.2          | 128     | 26 399     | 579     | 4.525     |            |            | 2026-09-06 01:18 |
| olmo2:7b | 131072 | medium | stream     | CPU    | **loken 0.1.0** | **42.5**      | **4.8**      | **128** | **28 419** | **648** | **5.061** | -8.8%      | -10.6%     | 2026-09-06 01:18 |
| olmo2:7b | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 221.0         | 100.8        | 128     | 1 457      | 268     | 2.096     |            |            | 2026-09-04 10:52 |
| olmo2:7b | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **1 375.5**   | **137.3**    | **128** | **1 001**  | **228** | **1.778** | **+36.2%** | **+17.9%** | 2026-09-04 10:52 |
| olmo2:7b | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 90.1         | 128     | 1 443      | 264     | 2.060     |            |            | 2026-09-04 08:52 |
| olmo2:7b | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **132.6**    | **128** | **1 005**  | **240** | **1.876** | **+47.2%** | **+9.8%**  | 2026-09-04 08:52 |
| olmo2:7b | 131072 | short  | stream     | CPU    | Ollama 0.32.6   | 36.4          | 5.3          | 128     | 26 219     | 572     | 4.469     |            |            | 2026-09-05 22:24 |
| olmo2:7b | 131072 | short  | stream     | CPU    | **loken 0.1.0** | **21.5**      | **4.8**      | **128** | **27 913** | **636** | **4.968** | -8.4%      | -10.1%     | 2026-09-05 22:24 |
| olmo2:7b | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 89.2          | 101.0        | 128     | 1 434      | 261     | 2.041     |            |            | 2026-09-04 02:55 |
| olmo2:7b | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **621.8**     | **137.2**    | **128** | **1 006**  | **229** | **1.787** | **+35.8%** | **+14.2%** | 2026-09-04 02:55 |

### olmoe

![olmoe](img/family-olmoe.svg)

| Model        | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token   | Δ decode   | Δ energy   | Date             |
|--------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|-----------|------------|------------|------------------|
| olmoe:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 321.9        | 128     | 435       | 69      | 0.540     |            |            | 2026-09-04 00:51 |
| olmoe:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **335.1**    | **128** | **438**   | **83**  | **0.647** | **+4.1%**  | -16.6%     | 2026-09-04 00:51 |
| olmoe:latest | 4096   | long   | stream     | CPU    | Ollama 0.32.6   | 1 390.4       | 27.0         | 128     | 6 642     | 117     | 0.916     |            |            | 2026-09-05 07:21 |
| olmoe:latest | 4096   | long   | stream     | CPU    | **loken 0.1.0** | **203.3**     | **24.9**     | **128** | **6 684** | **152** | **1.187** | -7.7%      | -22.9%     | 2026-09-05 07:21 |
| olmoe:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 4 187.7       | 383.2        | 128     | 438       | 66      | 0.513     |            |            | 2026-09-03 23:13 |
| olmoe:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **2 818.2**   | **415.2**    | **128** | **442**   | **84**  | **0.658** | **+8.4%**  | -22.1%     | 2026-09-03 23:13 |
| olmoe:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 317.8        | 128     | 434       | 66      | 0.519     |            |            | 2026-09-03 21:35 |
| olmoe:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **412.9**    | **122** | **354**   | **44**  | **0.363** |            |            | 2026-09-03 21:35 |
| olmoe:latest | 4096   | medium | stream     | CPU    | Ollama 0.32.6   | 266.4         | 28.2         | 128     | 6 171     | 113     | 0.880     |            |            | 2026-09-05 02:22 |
| olmoe:latest | 4096   | medium | stream     | CPU    | **loken 0.1.0** | **110.9**     | **25.9**     | **118** | **5 068** | **166** | **1.410** |            |            | 2026-09-05 02:22 |
| olmoe:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 231.0         | 289.4        | 128     | 681       | 72      | 0.560     |            |            | 2026-09-03 19:52 |
| olmoe:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 941.9**   | **463.6**    | **128** | **343**   | **43**  | **0.338** | **+60.2%** | **+65.4%** | 2026-09-03 19:52 |
| olmoe:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 321.2        | 128     | 420       | 62      | 0.488     |            |            | 2026-09-03 18:13 |
| olmoe:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **372.3**    | **128** | **367**   | **54**  | **0.420** | **+15.9%** | **+16.2%** | 2026-09-03 18:13 |
| olmoe:latest | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 75.5          | 28.4         | 128     | 6 075     | 113     | 0.879     |            |            | 2026-09-04 21:48 |
| olmoe:latest | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **44.4**      | **26.4**     | **128** | **5 474** | **175** | **1.370** | -6.9%      | -35.8%     | 2026-09-04 21:48 |
| olmoe:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 143.3         | 389.1        | 128     | 450       | 69      | 0.536     |            |            | 2026-09-03 16:37 |
| olmoe:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **400.7**     | **455.1**    | **128** | **360**   | **56**  | **0.441** | **+17.0%** | **+21.6%** | 2026-09-03 16:37 |
| olmoe:latest | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 321.5        | 128     | 449       | 64      | 0.503     |            |            | 2026-09-04 16:36 |
| olmoe:latest | 131072 | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **332.0**    | **128** | **422**   | **78**  | **0.608** | **+3.3%**  | -17.3%     | 2026-09-04 16:36 |
| olmoe:latest | 131072 | long   | stream     | CPU    | Ollama 0.32.6   | 1 576.7       | 27.0         | 128     | 6 565     | 117     | 0.918     |            |            | 2026-09-06 04:11 |
| olmoe:latest | 131072 | long   | stream     | CPU    | **loken 0.1.0** | **202.1**     | **24.9**     | **128** | **6 502** | **216** | **1.685** | -7.8%      | -45.5%     | 2026-09-06 04:11 |
| olmoe:latest | 131072 | long   | stream     | GPU    | **loken 0.1.0** | **2 700.8**   | **413.8**    | **128** | **455**   | **76**  | **0.590** |            |            | 2026-09-04 14:36 |
| olmoe:latest | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 319.0        | 128     | 432       | 64      | 0.502     |            |            | 2026-09-04 12:44 |
| olmoe:latest | 131072 | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **399.7**    | **122** | **364**   | **49**  | **0.403** |            |            | 2026-09-04 12:44 |
| olmoe:latest | 131072 | medium | stream     | CPU    | Ollama 0.32.6   | 239.1         | 28.3         | 128     | 6 121     | 113     | 0.880     |            |            | 2026-09-06 01:20 |
| olmoe:latest | 131072 | medium | stream     | CPU    | **loken 0.1.0** | **128.3**     | **26.0**     | **118** | **5 032** | **116** | **0.986** |            |            | 2026-09-06 01:20 |
| olmoe:latest | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 577.2         | 389.9        | 128     | 440       | 68      | 0.533     |            |            | 2026-09-04 10:54 |
| olmoe:latest | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **1 042.9**   | **453.9**    | **128** | **367**   | **59**  | **0.462** | **+16.4%** | **+15.4%** | 2026-09-04 10:54 |
| olmoe:latest | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 298.6        | 128     | 428       | 70      | 0.544     |            |            | 2026-09-04 08:53 |
| olmoe:latest | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **423.7**    | **128** | **337**   | **50**  | **0.390** | **+41.9%** | **+39.6%** | 2026-09-04 08:53 |
| olmoe:latest | 131072 | short  | stream     | CPU    | Ollama 0.32.6   | 81.9          | 28.5         | 128     | 6 072     | 110     | 0.860     |            |            | 2026-09-05 22:26 |
| olmoe:latest | 131072 | short  | stream     | CPU    | **loken 0.1.0** | **43.0**      | **26.5**     | **128** | **5 444** | **177** | **1.381** | -7.2%      | -37.7%     | 2026-09-05 22:26 |
| olmoe:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 189.7         | 390.3        | 128     | 430       | 61      | 0.480     |            |            | 2026-09-04 02:56 |
| olmoe:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **1 153.5**   | **462.2**    | **128** | **334**   | **48**  | **0.375** | **+18.4%** | **+27.8%** | 2026-09-04 02:56 |

### phi2

![phi2](img/family-phi2.svg)

| Model            | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token   | Δ decode    | Δ energy   | Date             |
|------------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|-----------|-------------|------------|------------------|
| moondream:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 1       | 117       | 12      |  -        | no answer   |            | 2026-09-04 00:46 |
| moondream:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 276.0        | 128     | 451       | 63      | 0.492     |             |            | 2026-09-03 21:30 |
| moondream:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **475.3**    | **128** | **262**   | **47**  | **0.366** | **+72.2%**  | **+34.4%** | 2026-09-03 21:30 |
| moondream:latest | 4096   | medium | stream     | CPU    | Ollama 0.32.6   | 219.5         | 27.6         | 128     | 5 533     | 115     | 0.899     |             |            | 2026-09-05 02:13 |
| moondream:latest | 4096   | medium | stream     | CPU    | **loken 0.1.0** | **108.7**     | **20.8**     | **128** | **6 562** | **322** | **2.513** | -24.8%      | -64.2%     | 2026-09-05 02:13 |
| moondream:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 411.5         | 370.6        | 128     | 458       | 67      | 0.520     |             |            | 2026-09-03 19:47 |
| moondream:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **2 759.4**   | **605.8**    | **128** | **248**   | **48**  | **0.374** | **+63.5%**  | **+39.2%** | 2026-09-03 19:47 |
| moondream:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 306.3        | 128     | 448       | 61      | 0.475     |             |            | 2026-09-03 18:08 |
| moondream:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **549.0**    | **128** | **253**   | **47**  | **0.364** | **+79.2%**  | **+30.5%** | 2026-09-03 18:08 |
| moondream:latest | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 94.4          | 27.9         | 128     | 5 466     | 112     | 0.874     |             |            | 2026-09-04 21:39 |
| moondream:latest | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **52.6**      | **21.2**     | **128** | **6 310** | **370** | **2.887** | -23.9%      | -69.7%     | 2026-09-04 21:39 |
| moondream:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 112.6         | 374.5        | 128     | 463       | 61      | 0.476     |             |            | 2026-09-03 16:32 |
| moondream:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **363.2**     | **589.0**    | **128** | **258**   | **48**  | **0.374** | **+57.3%**  | **+27.3%** | 2026-09-03 16:32 |
| moondream:latest | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 1       | 91        | 6       |  -        | no answer   |            | 2026-09-04 16:31 |
| moondream:latest | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 276.3        | 128     | 463       | 62      | 0.482     |             |            | 2026-09-04 12:40 |
| moondream:latest | 131072 | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **541.1**    | **128** | **252**   | **47**  | **0.370** | **+95.9%**  | **+30.3%** | 2026-09-04 12:40 |
| moondream:latest | 131072 | medium | stream     | CPU    | Ollama 0.32.6   | 213.9         | 27.6         | 128     | 5 551     | 113     | 0.884     |             |            | 2026-09-06 01:11 |
| moondream:latest | 131072 | medium | stream     | CPU    | **loken 0.1.0** | **92.2**      | **20.7**     | **128** | **6 596** | **382** | **2.985** | -24.8%      | -70.4%     | 2026-09-06 01:11 |
| moondream:latest | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 509.8         | 370.7        | 128     | 458       | 69      | 0.537     |             |            | 2026-09-04 10:49 |
| moondream:latest | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **2 644.2**   | **607.0**    | **128** | **242**   | **46**  | **0.360** | **+63.7%**  | **+49.3%** | 2026-09-04 10:49 |
| moondream:latest | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 208.7        | 128     | 615       | 74      | 0.581     |             |            | 2026-09-04 08:48 |
| moondream:latest | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **560.8**    | **128** | **256**   | **45**  | **0.352** | **+168.8%** | **+65.2%** | 2026-09-04 08:48 |
| moondream:latest | 131072 | short  | stream     | CPU    | Ollama 0.32.6   | 91.3          | 27.9         | 128     | 5 452     | 110     | 0.861     |             |            | 2026-09-05 22:17 |
| moondream:latest | 131072 | short  | stream     | CPU    | **loken 0.1.0** | **56.2**      | **21.1**     | **128** | **6 322** | **355** | **2.774** | -24.2%      | -69.0%     | 2026-09-05 22:17 |
| moondream:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 189.3         | 372.8        | 128     | 458       | 60      | 0.467     |             |            | 2026-09-04 02:51 |
| moondream:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **1 357.8**   | **614.2**    | **128** | **244**   | **50**  | **0.388** | **+64.8%**  | **+20.4%** | 2026-09-04 02:51 |

### qwen2

![qwen2](img/family-qwen2.svg)

| Model           | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms      | J/req     | J/token    | Δ decode     | Δ energy    | Date             |
|-----------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|-------------|-----------|------------|--------------|-------------|------------------|
| deepcoder:14b   | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 49.7         | 128     | 4 550       | 500       | 3.908      |              |             | 2026-09-03 23:37 |
| deepcoder:14b   | 4096   | long   | non-stream | GPU    | vLLM 0.22.0     |  -            | 84.7         | 128     | 1 565       | 323       | 2.522      |              |             | 2026-09-03 23:37 |
| deepcoder:14b   | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **75.6**     | **128** | **1 722**   | **440**   | **3.437**  | -10.8%       | -26.6%      | 2026-09-03 23:37 |
| deepcoder:14b   | 4096   | long   | stream     | CPU    | Ollama 0.32.6   | 367.0         | 2.6          | 128     | 54 486      | 1 174     | 9.172      |              |             | 2026-09-05 03:09 |
| deepcoder:14b   | 4096   | long   | stream     | CPU    | **loken 0.1.0** | **39.3**      | **2.6**      | **128** | **56 938**  | **1 281** | **10.009** | -0.0%        | -8.4%       | 2026-09-05 03:09 |
| deepcoder:14b   | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 1 163.8       | 53.4         | 128     | 4 571       | 496       | 3.877      |              |             | 2026-09-03 22:00 |
| deepcoder:14b   | 4096   | long   | stream     | GPU    | vLLM 0.22.0     | 10 364.5      | 85.1         | 128     | 1 569       | 329       | 2.573      |              |             | 2026-09-03 22:00 |
| deepcoder:14b   | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **2 413.2**   | **79.2**     | **128** | **1 728**   | **439**   | **3.431**  | -6.9%        | -25.0%      | 2026-09-03 22:00 |
| deepcoder:14b   | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 49.0         | 128     | 4 588       | 509       | 3.974      |              |             | 2026-09-03 20:21 |
| deepcoder:14b   | 4096   | medium | non-stream | GPU    | vLLM 0.22.0     |  -            | 83.1         | 128     | 1 557       | 321       | 2.509      |              |             | 2026-09-03 20:21 |
| deepcoder:14b   | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **81.8**     | **128** | **1 572**   | **396**   | **3.092**  | -1.6%        | -18.8%      | 2026-09-03 20:21 |
| deepcoder:14b   | 4096   | medium | stream     | CPU    | Ollama 0.32.6   | 68.0          | 2.6          | 128     | 52 412      | 1 155     | 9.026      |              |             | 2026-09-04 22:41 |
| deepcoder:14b   | 4096   | medium | stream     | CPU    | **loken 0.1.0** | **28.5**      | **2.6**      | **120** | **48 760**  | **1 105** | **9.207**  |              |             | 2026-09-04 22:41 |
| deepcoder:14b   | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 208.0         | 53.0         | 128     | 4 584       | 499       | 3.895      |              |             | 2026-09-03 18:38 |
| deepcoder:14b   | 4096   | medium | stream     | GPU    | vLLM 0.22.0     | 2 203.0       | 85.0         | 128     | 1 549       | 315       | 2.461      |              |             | 2026-09-03 18:38 |
| deepcoder:14b   | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **672.9**     | **84.2**     | **128** | **1 589**   | **398**   | **3.109**  | -1.0%        | -20.8%      | 2026-09-03 18:38 |
| deepcoder:14b   | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 49.0         | 128     | 4 543       | 509       | 3.974      |              |             | 2026-09-03 17:00 |
| deepcoder:14b   | 4096   | short  | non-stream | GPU    | vLLM 0.22.0     |  -            | 83.4         | 128     | 1 561       | 337       | 2.637      |              |             | 2026-09-03 17:00 |
| deepcoder:14b   | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **81.6**     | **128** | **1 586**   | **409**   | **3.198**  | -2.1%        | -17.5%      | 2026-09-03 17:00 |
| deepcoder:14b   | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 24.7          | 2.6          | 128     | 52 075      | 1 127     | 8.806      |              |             | 2026-09-04 17:22 |
| deepcoder:14b   | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **19.1**      | **2.7**      | **128** | **49 670**  | **1 102** | **8.607**  | **+3.3%**    | **+2.3%**   | 2026-09-04 17:22 |
| deepcoder:14b   | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 74.9          | 53.0         | 128     | 4 620       | 513       | 4.004      |              |             | 2026-09-03 15:24 |
| deepcoder:14b   | 4096   | short  | stream     | GPU    | vLLM 0.22.0     | 718.6         | 85.1         | 128     | 1 553       | 318       | 2.488      |              |             | 2026-09-03 15:24 |
| deepcoder:14b   | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **417.3**     | **81.7**     | **128** | **1 618**   | **402**   | **3.144**  | -4.0%        | -20.9%      | 2026-09-03 15:24 |
| deepcoder:14b   | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 12.5         | 128     | 12 928      | 929       | 7.255      |              |             | 2026-09-04 15:04 |
| deepcoder:14b   | 131072 | long   | non-stream | GPU    | vLLM 0.22.0     |  -            | 84.5         | 128     | 1 572       | 311       | 2.430      |              |             | 2026-09-04 15:04 |
| deepcoder:14b   | 131072 | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **75.6**     | **128** | **1 721**   | **443**   | **3.461**  | -10.5%       | -29.8%      | 2026-09-04 15:04 |
| deepcoder:14b   | 131072 | long   | stream     | CPU    | Ollama 0.32.6   | 368.0         | 2.6          | 128     | 56 776      | 1 161     | 9.070      |              |             | 2026-09-06 02:08 |
| deepcoder:14b   | 131072 | long   | stream     | CPU    | **loken 0.1.0** | **38.9**      | **2.6**      | **128** | **56 507**  | **1 286** | **10.048** | **+0.6%**    | -9.7%       | 2026-09-06 02:08 |
| deepcoder:14b   | 131072 | long   | stream     | GPU    | Ollama 0.32.6   | 750.7         | 12.7         | 128     | 13 006      | 936       | 7.310      |              |             | 2026-09-04 13:08 |
| deepcoder:14b   | 131072 | long   | stream     | GPU    | vLLM 0.22.0     | 4 523.1       | 85.2         | 128     | 1 600       | 342       | 2.671      |              |             | 2026-09-04 13:08 |
| deepcoder:14b   | 131072 | long   | stream     | GPU    | **loken 0.1.0** | **2 370.7**   | **79.2**     | **128** | **1 732**   | **448**   | **3.498**  | -7.0%        | -23.6%      | 2026-09-04 13:08 |
| deepcoder:14b   | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 12.5         | 128     | 12 987      | 928       | 7.250      |              |             | 2026-09-04 11:18 |
| deepcoder:14b   | 131072 | medium | non-stream | GPU    | vLLM 0.22.0     |  -            | 84.1         | 128     | 1 553       | 328       | 2.565      |              |             | 2026-09-04 11:18 |
| deepcoder:14b   | 131072 | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **82.0**     | **128** | **1 571**   | **398**   | **3.112**  | -2.6%        | -17.6%      | 2026-09-04 11:18 |
| deepcoder:14b   | 131072 | medium | stream     | CPU    | Ollama 0.32.6   | 65.1          | 2.6          | 128     | 54 704      | 1 154     | 9.014      |              |             | 2026-09-05 23:10 |
| deepcoder:14b   | 131072 | medium | stream     | CPU    | **loken 0.1.0** | **26.2**      | **2.6**      | **120** | **48 331**  | **1 089** | **9.076**  |              |             | 2026-09-05 23:10 |
| deepcoder:14b   | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 137.5         | 12.9         | 128     | 12 801      | 900       | 7.033      |              |             | 2026-09-04 09:19 |
| deepcoder:14b   | 131072 | medium | stream     | GPU    | vLLM 0.22.0     | 2 007.8       | 85.0         | 128     | 1 551       | 317       | 2.478      |              |             | 2026-09-04 09:19 |
| deepcoder:14b   | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **634.3**     | **84.2**     | **128** | **1 589**   | **400**   | **3.123**  | -0.9%        | -20.7%      | 2026-09-04 09:19 |
| deepcoder:14b   | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 6.3          | 128     | 17 326      | 1 602     | 12.515     |              |             | 2026-09-04 07:15 |
| deepcoder:14b   | 131072 | short  | non-stream | GPU    | vLLM 0.22.0     |  -            | 84.4         | 128     | 1 573       | 325       | 2.542      |              |             | 2026-09-04 07:15 |
| deepcoder:14b   | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **78.2**     | **128** | **1 638**   | **414**   | **3.231**  | -7.3%        | -21.3%      | 2026-09-04 07:15 |
| deepcoder:14b   | 131072 | short  | stream     | CPU    | Ollama 0.32.6   | 23.5          | 2.6          | 128     | 54 611      | 1 158     | 9.047      |              |             | 2026-09-05 08:11 |
| deepcoder:14b   | 131072 | short  | stream     | CPU    | **loken 0.1.0** | **19.7**      | **2.7**      | **128** | **49 536**  | **1 122** | **8.769**  | **+3.6%**    | **+3.2%**   | 2026-09-05 08:11 |
| deepcoder:14b   | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 47.2          | 12.8         | 128     | 12 948      | 932       | 7.279      |              |             | 2026-09-04 01:16 |
| deepcoder:14b   | 131072 | short  | stream     | GPU    | vLLM 0.22.0     | 823.5         | 85.0         | 128     | 1 548       | 324       | 2.532      |              |             | 2026-09-04 01:16 |
| deepcoder:14b   | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **468.4**     | **82.4**     | **128** | **1 602**   | **399**   | **3.120**  | -3.0%        | -18.9%      | 2026-09-04 01:16 |
| deepseek-r1:32b | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 25.0         | 128     | 8 176       | 1 068     | 8.342      |              |             | 2026-09-03 23:40 |
| deepseek-r1:32b | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **36.5**     | **128** | **3 773**   | **1 074** | **8.392**  | **+45.8%**   | -0.6%       | 2026-09-03 23:40 |
| deepseek-r1:32b | 4096   | long   | stream     | CPU    | Ollama 0.32.6   | 217.0         | 1.2          | 128     | 120 780     | 2 513     | 19.635     |              |             | 2026-09-05 03:23 |
| deepseek-r1:32b | 4096   | long   | stream     | CPU    | **loken 0.1.0** | **17.1**      | **1.2**      | **128** | **123 092** | **2 750** | **21.485** | **+3.4%**    | -8.6%       | 2026-09-05 03:23 |
| deepseek-r1:32b | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 1 001.9       | 25.9         | 128     | 8 135       | 1 076     | 8.408      |              |             | 2026-09-03 22:02 |
| deepseek-r1:32b | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **673.9**     | **40.0**     | **128** | **3 812**   | **1 079** | **8.431**  | **+54.2%**   | -0.3%       | 2026-09-03 22:02 |
| deepseek-r1:32b | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 24.9         | 128     | 8 297       | 1 070     | 8.362      |              |             | 2026-09-03 20:24 |
| deepseek-r1:32b | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **40.1**     | **128** | **3 500**   | **1 003** | **7.839**  | **+60.9%**   | **+6.7%**   | 2026-09-03 20:24 |
| deepseek-r1:32b | 4096   | medium | stream     | CPU    | Ollama 0.32.6   | 38.8          | 1.2          | 128     | 116 627     | 2 571     | 20.086     |              |             | 2026-09-04 22:54 |
| deepseek-r1:32b | 4096   | medium | stream     | CPU    | **loken 0.1.0** | **14.4**      | **1.2**      | **128** | **115 701** | **2 584** | **20.190** | **+0.5%**    | -0.5%       | 2026-09-04 22:54 |
| deepseek-r1:32b | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 177.3         | 26.0         | 128     | 8 232       | 1 077     | 8.417      |              |             | 2026-09-03 18:41 |
| deepseek-r1:32b | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **374.9**     | **41.1**     | **128** | **3 485**   | **1 002** | **7.831**  | **+58.2%**   | **+7.5%**   | 2026-09-03 18:41 |
| deepseek-r1:32b | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 24.9         | 128     | 8 053       | 1 065     | 8.323      |              |             | 2026-09-03 17:03 |
| deepseek-r1:32b | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **40.6**     | **128** | **3 431**   | **965**   | **7.540**  | **+63.2%**   | **+10.4%**  | 2026-09-03 17:03 |
| deepseek-r1:32b | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 14.2          | 1.2          | 128     | 115 938     | 2 529     | 19.755     |              |             | 2026-09-04 17:35 |
| deepseek-r1:32b | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **9.1**       | **1.2**      | **128** | **111 029** | **2 459** | **19.214** | **+3.7%**    | **+2.8%**   | 2026-09-04 17:35 |
| deepseek-r1:32b | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 65.7          | 25.9         | 128     | 8 073       | 1 087     | 8.496      |              |             | 2026-09-03 15:26 |
| deepseek-r1:32b | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **231.9**     | **41.2**     | **128** | **3 412**   | **974**   | **7.612**  | **+59.3%**   | **+11.6%**  | 2026-09-03 15:26 |
| deepseek-r1:32b | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 1.5          | 128     | 87 335      | 6 026     | 47.082     |              |             | 2026-09-04 15:11 |
| deepseek-r1:32b | 131072 | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **26.5**     | **128** | **4 800**   | **1 156** | **9.028**  | **+1720.5%** | **+421.5%** | 2026-09-04 15:11 |
| deepseek-r1:32b | 131072 | long   | stream     | CPU    | Ollama 0.32.6   | 214.0         | 1.2          | 128     | 123 990     | 2 530     | 19.765     |              |             | 2026-09-06 02:22 |
| deepseek-r1:32b | 131072 | long   | stream     | CPU    | **loken 0.1.0** | **16.7**      | **1.2**      | **128** | **125 925** | **2 743** | **21.430** | **+3.0%**    | -7.8%       | 2026-09-06 02:22 |
| deepseek-r1:32b | 131072 | long   | stream     | GPU    | Ollama 0.32.6   | 362.7         | 2.4          | 128     | 57 941      | 3 861     | 30.165     |              |             | 2026-09-04 13:14 |
| deepseek-r1:32b | 131072 | long   | stream     | GPU    | **loken 0.1.0** | **693.3**     | **39.7**     | **128** | **3 826**   | **1 089** | **8.512**  | **+1531.9%** | **+254.4%** | 2026-09-04 13:14 |
| deepseek-r1:32b | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 2.5          | 128     | 57 402      | 3 806     | 29.737     |              |             | 2026-09-04 11:23 |
| deepseek-r1:32b | 131072 | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **39.8**     | **128** | **3 489**   | **1 023** | **7.993**  | **+1520.7%** | **+272.0%** | 2026-09-04 11:23 |
| deepseek-r1:32b | 131072 | medium | stream     | CPU    | Ollama 0.32.6   | 40.2          | 1.2          | 128     | 119 800     | 2 559     | 19.990     |              |             | 2026-09-05 23:24 |
| deepseek-r1:32b | 131072 | medium | stream     | CPU    | **loken 0.1.0** | **14.3**      | **1.2**      | **128** | **117 892** | **2 573** | **20.098** | **+0.9%**    | -0.5%       | 2026-09-05 23:24 |
| deepseek-r1:32b | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 66.1          | 2.4          | 128     | 58 100      | 3 784     | 29.561     |              |             | 2026-09-04 09:25 |
| deepseek-r1:32b | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **462.5**     | **40.8**     | **128** | **3 507**   | **1 013** | **7.916**  | **+1574.9%** | **+273.4%** | 2026-09-04 09:25 |
| deepseek-r1:32b | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 2.0          | 128     | 96 298      | 4 644     | 36.279     |              |             | 2026-09-04 07:23 |
| deepseek-r1:32b | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **27.8**     | **128** | **4 726**   | **1 082** | **8.453**  | **+1281.2%** | **+329.2%** | 2026-09-04 07:23 |
| deepseek-r1:32b | 131072 | short  | stream     | CPU    | Ollama 0.32.6   | 13.8          | 1.2          | 128     | 118 305     | 2 534     | 19.796     |              |             | 2026-09-05 08:25 |
| deepseek-r1:32b | 131072 | short  | stream     | CPU    | **loken 0.1.0** | **8.7**       | **1.2**      | **128** | **114 417** | **2 453** | **19.164** | **+2.5%**    | **+3.3%**   | 2026-09-05 08:25 |
| deepseek-r1:32b | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 23.5          | 2.0          | 128     | 73 370      | 4 701     | 36.729     |              |             | 2026-09-04 01:26 |
| deepseek-r1:32b | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **163.6**     | **41.1**     | **128** | **3 453**   | **1 011** | **7.902**  | **+1994.7%** | **+364.8%** | 2026-09-04 01:26 |
| qwen2.5:0.5b    | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 219.8        | 128     | 1 869       | 56        | 0.434      |              |             | 2026-09-04 00:51 |
| qwen2.5:0.5b    | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **319.0**    | **128** | **409**     | **45**    | **0.348**  | **+45.1%**   | **+24.8%**  | 2026-09-04 00:51 |
| qwen2.5:0.5b    | 4096   | long   | stream     | CPU    | Ollama 0.32.6   | 881.0         | 51.7         | 128     | 3 476       | 66        | 0.518      |              |             | 2026-09-05 07:22 |
| qwen2.5:0.5b    | 4096   | long   | stream     | CPU    | **loken 0.1.0** | **314.6**     | **49.9**     | **128** | **3 273**   | **198**   | **1.544**  | -3.4%        | -66.4%      | 2026-09-05 07:22 |
| qwen2.5:0.5b    | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 1 272.3       | 325.2        | 128     | 1 885       | 62        | 0.484      |              |             | 2026-09-03 23:14 |
| qwen2.5:0.5b    | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **8 275.0**   | **370.8**    | **128** | **399**     | **50**    | **0.392**  | **+14.0%**   | **+23.4%**  | 2026-09-03 23:14 |
| qwen2.5:0.5b    | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 223.1        | 128     | 1 826       | 58        | 0.452      |              |             | 2026-09-03 21:36 |
| qwen2.5:0.5b    | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **325.4**    | **128** | **385**     | **42**    | **0.326**  | **+45.8%**   | **+38.8%**  | 2026-09-03 21:36 |
| qwen2.5:0.5b    | 4096   | medium | stream     | CPU    | Ollama 0.32.6   | 169.4         | 52.8         | 128     | 3 321       | 67        | 0.523      |              |             | 2026-09-05 02:24 |
| qwen2.5:0.5b    | 4096   | medium | stream     | CPU    | **loken 0.1.0** | **150.8**     | **51.8**     | **128** | **2 758**   | **164**   | **1.283**  | -1.9%        | -59.2%      | 2026-09-05 02:24 |
| qwen2.5:0.5b    | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 206.0         | 288.8        | 128     | 2 274       | 65        | 0.508      |              |             | 2026-09-03 19:53 |
| qwen2.5:0.5b    | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **790.9**     | **245.6**    | **128** | **604**     | **65**    | **0.506**  | -15.0%       | **+0.3%**   | 2026-09-03 19:53 |
| qwen2.5:0.5b    | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 215.5        | 128     | 1 867       | 59        | 0.461      |              |             | 2026-09-03 18:14 |
| qwen2.5:0.5b    | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **351.5**    | **128** | **379**     | **45**    | **0.353**  | **+63.1%**   | **+30.3%**  | 2026-09-03 18:14 |
| qwen2.5:0.5b    | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 55.3          | 53.1         | 128     | 3 322       | 67        | 0.524      |              |             | 2026-09-04 21:49 |
| qwen2.5:0.5b    | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **71.8**      | **52.2**     | **128** | **2 684**   | **161**   | **1.259**  | -1.7%        | -58.4%      | 2026-09-04 21:49 |
| qwen2.5:0.5b    | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 82.3          | 306.5        | 128     | 1 896       | 56        | 0.441      |              |             | 2026-09-03 16:38 |
| qwen2.5:0.5b    | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **722.4**     | **382.2**    | **128** | **379**     | **47**    | **0.370**  | **+24.7%**   | **+19.2%**  | 2026-09-03 16:38 |
| qwen2.5:0.5b    | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 224.9        | 128     | 622         | 60        | 0.468      |              |             | 2026-09-04 16:37 |
| qwen2.5:0.5b    | 131072 | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **359.0**    | **128** | **378**     | **49**    | **0.381**  | **+59.6%**   | **+22.9%**  | 2026-09-04 16:37 |
| qwen2.5:0.5b    | 131072 | long   | stream     | CPU    | Ollama 0.32.6   | 997.3         | 51.7         | 128     | 3 529       | 66        | 0.519      |              |             | 2026-09-06 04:13 |
| qwen2.5:0.5b    | 131072 | long   | stream     | CPU    | **loken 0.1.0** | **335.6**     | **51.4**     | **128** | **3 184**   | **190**   | **1.485**  | -0.7%        | -65.0%      | 2026-09-06 04:13 |
| qwen2.5:0.5b    | 131072 | long   | stream     | GPU    | Ollama 0.32.6   | 1 171.8       | 322.1        | 128     | 651         | 63        | 0.492      |              |             | 2026-09-04 14:37 |
| qwen2.5:0.5b    | 131072 | long   | stream     | GPU    | **loken 0.1.0** | **6 656.5**   | **335.9**    | **128** | **447**     | **50**    | **0.394**  | **+4.3%**    | **+24.8%**  | 2026-09-04 14:37 |
| qwen2.5:0.5b    | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 205.6        | 128     | 643         | 59        | 0.457      |              |             | 2026-09-04 12:45 |
| qwen2.5:0.5b    | 131072 | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **335.2**    | **128** | **376**     | **47**    | **0.369**  | **+63.0%**   | **+23.9%**  | 2026-09-04 12:45 |
| qwen2.5:0.5b    | 131072 | medium | stream     | CPU    | Ollama 0.32.6   | 168.4         | 52.8         | 128     | 3 421       | 67        | 0.526      |              |             | 2026-09-06 01:21 |
| qwen2.5:0.5b    | 131072 | medium | stream     | CPU    | **loken 0.1.0** | **151.0**     | **52.1**     | **128** | **2 783**   | **163**   | **1.273**  | -1.3%        | -58.6%      | 2026-09-06 01:21 |
| qwen2.5:0.5b    | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 222.4         | 321.2        | 128     | 620         | 62        | 0.487      |              |             | 2026-09-04 10:54 |
| qwen2.5:0.5b    | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **1 903.3**   | **380.1**    | **128** | **381**     | **43**    | **0.333**  | **+18.4%**   | **+46.5%**  | 2026-09-04 10:54 |
| qwen2.5:0.5b    | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 173.3        | 128     | 775         | 72        | 0.560      |              |             | 2026-09-04 08:54 |
| qwen2.5:0.5b    | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **349.2**    | **128** | **391**     | **49**    | **0.383**  | **+101.5%**  | **+46.4%**  | 2026-09-04 08:54 |
| qwen2.5:0.5b    | 131072 | short  | stream     | CPU    | Ollama 0.32.6   | 54.6          | 53.1         | 128     | 3 416       | 65        | 0.511      |              |             | 2026-09-05 22:27 |
| qwen2.5:0.5b    | 131072 | short  | stream     | CPU    | **loken 0.1.0** | **68.1**      | **52.4**     | **128** | **2 677**   | **165**   | **1.289**  | -1.3%        | -60.4%      | 2026-09-05 22:27 |
| qwen2.5:0.5b    | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 82.9          | 306.4        | 128     | 642         | 56        | 0.438      |              |             | 2026-09-04 02:57 |
| qwen2.5:0.5b    | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **683.3**     | **375.6**    | **128** | **391**     | **49**    | **0.379**  | **+22.6%**   | **+15.4%**  | 2026-09-04 02:57 |

### qwen3

![qwen3](img/family-qwen3.svg)

| Model      | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req     | J/token    | Δ decode    | Δ energy    | Date             |
|------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|-----------|------------|-------------|-------------|------------------|
| qwen3:0.6b | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -        |  -         | incoherent  |             | 2026-09-04 01:05 |
| qwen3:0.6b | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **542.4**    | **128** | **267**    | **36**    | **0.282**  |             |             | 2026-09-04 01:05 |
| qwen3:0.6b | 4096   | long   | stream     | CPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -        |  -         | incoherent  |             | 2026-09-05 07:47 |
| qwen3:0.6b | 4096   | long   | stream     | CPU    | **loken 0.1.0** | **443.9**     | **47.0**     | **128** | **3 255**  | **196**   | **1.535**  |             |             | 2026-09-05 07:47 |
| qwen3:0.6b | 4096   | long   | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -        |  -         | incoherent  |             | 2026-09-03 23:26 |
| qwen3:0.6b | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **7 965.8**   | **594.1**    | **128** | **283**    | **42**    | **0.325**  |             |             | 2026-09-03 23:26 |
| qwen3:0.6b | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 286.7        | 128     | 1 702      | 59        | 0.464      |             |             | 2026-09-03 21:50 |
| qwen3:0.6b | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **650.6**    | **128** | **241**    | **28**    | **0.220**  | **+127.0%** | **+110.4%** | 2026-09-03 21:50 |
| qwen3:0.6b | 4096   | medium | stream     | CPU    | Ollama 0.32.6   | 155.7         | 48.6         | 128     | 3 643      | 71        | 0.555      |             |             | 2026-09-05 02:47 |
| qwen3:0.6b | 4096   | medium | stream     | CPU    | **loken 0.1.0** | **226.4**     | **49.3**     | **128** | **2 825**  | **164**   | **1.284**  | **+1.4%**   | -56.8%      | 2026-09-05 02:47 |
| qwen3:0.6b | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 226.5         | 486.6        | 128     | 1 741      | 62        | 0.484      |             |             | 2026-09-03 20:09 |
| qwen3:0.6b | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 803.3**   | **736.8**    | **128** | **248**    | **23**    | **0.182**  | **+51.4%**  | **+165.9%** | 2026-09-03 20:09 |
| qwen3:0.6b | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 281.2        | 128     | 1 712      | 61        | 0.479      |             |             | 2026-09-03 18:27 |
| qwen3:0.6b | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **665.1**    | **128** | **233**    | **30**    | **0.236**  | **+136.5%** | **+103.3%** | 2026-09-03 18:27 |
| qwen3:0.6b | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 58.2          | 49.1         | 128     | 3 588      | 68        | 0.531      |             |             | 2026-09-04 22:15 |
| qwen3:0.6b | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **119.8**     | **51.2**     | **128** | **2 709**  | **154**   | **1.206**  | **+4.1%**   | -55.9%      | 2026-09-04 22:15 |
| qwen3:0.6b | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 80.2          | 485.9        | 128     | 1 733      | 63        | 0.491      |             |             | 2026-09-03 16:50 |
| qwen3:0.6b | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **252.9**     | **720.9**    | **128** | **261**    | **36**    | **0.281**  | **+48.4%**  | **+74.9%**  | 2026-09-03 16:50 |
| qwen3:0.6b | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -        |  -         | incoherent  |             | 2026-09-04 16:51 |
| qwen3:0.6b | 131072 | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **466.5**    | **128** | **281**    | **41**    | **0.319**  |             |             | 2026-09-04 16:51 |
| qwen3:0.6b | 131072 | long   | stream     | CPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -        |  -         | incoherent  |             | 2026-09-06 04:37 |
| qwen3:0.6b | 131072 | long   | stream     | CPU    | **loken 0.1.0** | **440.2**     | **45.6**     | **128** | **3 403**  | **192**   | **1.502**  |             |             | 2026-09-06 04:37 |
| qwen3:0.6b | 131072 | long   | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -        |  -         | incoherent  |             | 2026-09-04 14:51 |
| qwen3:0.6b | 131072 | long   | stream     | GPU    | **loken 0.1.0** | **6 459.6**   | **590.5**    | **128** | **287**    | **37**    | **0.290**  |             |             | 2026-09-04 14:51 |
| qwen3:0.6b | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 283.3        | 128     | 1 782      | 62        | 0.483      |             |             | 2026-09-04 12:57 |
| qwen3:0.6b | 131072 | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **610.2**    | **128** | **242**    | **28**    | **0.221**  | **+115.4%** | **+118.5%** | 2026-09-04 12:57 |
| qwen3:0.6b | 131072 | medium | stream     | CPU    | Ollama 0.32.6   | 161.9         | 48.3         | 128     | 3 877      | 72        | 0.560      |             |             | 2026-09-06 01:45 |
| qwen3:0.6b | 131072 | medium | stream     | CPU    | **loken 0.1.0** | **183.9**     | **48.0**     | **128** | **2 896**  | **176**   | **1.373**  | -0.7%       | -59.2%      | 2026-09-06 01:45 |
| qwen3:0.6b | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 231.7         | 484.1        | 128     | 1 797      | 60        | 0.470      |             |             | 2026-09-04 11:07 |
| qwen3:0.6b | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **1 651.5**   | **736.5**    | **128** | **249**    | **39**    | **0.306**  | **+52.1%**  | **+53.7%**  | 2026-09-04 11:07 |
| qwen3:0.6b | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 159.8        | 128     | 2 620      | 82        | 0.637      |             |             | 2026-09-04 09:07 |
| qwen3:0.6b | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **560.7**    | **128** | **244**    | **32**    | **0.246**  | **+251.0%** | **+158.7%** | 2026-09-04 09:07 |
| qwen3:0.6b | 131072 | short  | stream     | CPU    | Ollama 0.32.6   | 55.4          | 49.2         | 128     | 3 823      | 68        | 0.534      |             |             | 2026-09-05 22:49 |
| qwen3:0.6b | 131072 | short  | stream     | CPU    | **loken 0.1.0** | **100.9**     | **48.5**     | **128** | **2 842**  | **163**   | **1.276**  | -1.3%       | -58.1%      | 2026-09-05 22:49 |
| qwen3:0.6b | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 81.3          | 486.7        | 128     | 1 796      | 58        | 0.453      |             |             | 2026-09-04 03:11 |
| qwen3:0.6b | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **376.9**     | **735.2**    | **128** | **258**    | **39**    | **0.303**  | **+51.1%**  | **+49.3%**  | 2026-09-04 03:11 |
| qwen3:8b   | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 78.4         | 128     | 3 656      | 295       | 2.301      |             |             | 2026-09-04 01:08 |
| qwen3:8b   | 4096   | long   | non-stream | GPU    | vLLM 0.22.0     |  -            | 137.7        | 128     | 994        | 194       | 1.519      |             |             | 2026-09-04 01:08 |
| qwen3:8b   | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **140.0**    | **128** | **959**    | **233**   | **1.822**  | **+1.7%**   | -16.6%      | 2026-09-04 01:08 |
| qwen3:8b   | 4096   | long   | stream     | CPU    | Ollama 0.32.6   | 487.4         | 4.5          | 128     | 31 637     | 679       | 5.303      |             |             | 2026-09-05 07:51 |
| qwen3:8b   | 4096   | long   | stream     | CPU    | **loken 0.1.0** | **72.0**      | **5.0**      | **128** | **29 254** | **673**   | **5.254**  | **+12.3%**  | **+0.9%**   | 2026-09-05 07:51 |
| qwen3:8b   | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 1 235.5       | 96.6         | 128     | 3 194      | 294       | 2.298      |             |             | 2026-09-03 23:29 |
| qwen3:8b   | 4096   | long   | stream     | GPU    | vLLM 0.22.0     | 14 949.1      | 139.4        | 128     | 967        | 177       | 1.384      |             |             | 2026-09-03 23:29 |
| qwen3:8b   | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **3 312.2**   | **148.0**    | **128** | **969**    | **235**   | **1.839**  | **+6.2%**   | -24.7%      | 2026-09-03 23:29 |
| qwen3:8b   | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 85.3         | 128     | 3 158      | 272       | 2.124      |             |             | 2026-09-03 21:52 |
| qwen3:8b   | 4096   | medium | non-stream | GPU    | vLLM 0.22.0     |  -            | 138.5        | 128     | 959        | 179       | 1.401      |             |             | 2026-09-03 21:52 |
| qwen3:8b   | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **139.3**    | **128** | **939**    | **235**   | **1.835**  | **+0.6%**   | -23.7%      | 2026-09-03 21:52 |
| qwen3:8b   | 4096   | medium | stream     | CPU    | Ollama 0.32.6   | 88.1          | 4.5          | 128     | 30 424     | 675       | 5.277      |             |             | 2026-09-05 02:51 |
| qwen3:8b   | 4096   | medium | stream     | CPU    | **loken 0.1.0** | **49.6**      | **4.7**      | **128** | **28 317** | **656**   | **5.124**  | **+5.0%**   | **+3.0%**   | 2026-09-05 02:51 |
| qwen3:8b   | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 214.6         | 96.1         | 128     | 3 214      | 291       | 2.270      |             |             | 2026-09-03 20:12 |
| qwen3:8b   | 4096   | medium | stream     | GPU    | vLLM 0.22.0     | 3 021.2       | 139.4        | 128     | 964        | 179       | 1.402      |             |             | 2026-09-03 20:12 |
| qwen3:8b   | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **982.8**     | **142.6**    | **128** | **997**    | **234**   | **1.828**  | **+2.2%**   | -23.3%      | 2026-09-03 20:12 |
| qwen3:8b   | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 84.5         | 128     | 3 737      | 296       | 2.313      |             |             | 2026-09-03 18:30 |
| qwen3:8b   | 4096   | short  | non-stream | GPU    | vLLM 0.22.0     |  -            | 137.7        | 128     | 970        | 187       | 1.461      |             |             | 2026-09-03 18:30 |
| qwen3:8b   | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **147.6**    | **128** | **916**    | **224**   | **1.750**  | **+7.2%**   | -16.5%      | 2026-09-03 18:30 |
| qwen3:8b   | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 27.6          | 3.4          | 128     | 43 598     | 866       | 6.764      |             |             | 2026-09-04 22:21 |
| qwen3:8b   | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **18.2**      | **2.1**      | **128** | **51 656** | **1 392** | **10.875** | -37.9%      | -37.8%      | 2026-09-04 22:21 |
| qwen3:8b   | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 77.3          | 97.5         | 128     | 3 161      | 282       | 2.206      |             |             | 2026-09-03 16:52 |
| qwen3:8b   | 4096   | short  | stream     | GPU    | vLLM 0.22.0     | 863.9         | 139.3        | 128     | 969        | 185       | 1.449      |             |             | 2026-09-03 16:52 |
| qwen3:8b   | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **561.0**     | **153.1**    | **128** | **901**    | **222**   | **1.733**  | **+9.9%**   | -16.4%      | 2026-09-03 16:52 |
| qwen3:8b   | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 84.9         | 128     | 3 189      | 292       | 2.285      |             |             | 2026-09-04 16:53 |
| qwen3:8b   | 131072 | long   | non-stream | GPU    | vLLM 0.22.0     |  -            | 138.4        | 128     | 971        | 181       | 1.414      |             |             | 2026-09-04 16:53 |
| qwen3:8b   | 131072 | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **139.9**    | **128** | **947**    | **234**   | **1.829**  | **+1.1%**   | -22.7%      | 2026-09-04 16:53 |
| qwen3:8b   | 131072 | long   | stream     | CPU    | Ollama 0.32.6   | 488.1         | 4.5          | 128     | 31 852     | 678       | 5.294      |             |             | 2026-09-06 04:42 |
| qwen3:8b   | 131072 | long   | stream     | CPU    | **loken 0.1.0** | **73.1**      | **5.1**      | **128** | **29 174** | **660**   | **5.159**  | **+12.9%**  | **+2.6%**   | 2026-09-06 04:42 |
| qwen3:8b   | 131072 | long   | stream     | GPU    | Ollama 0.32.6   | 1 237.8       | 95.2         | 128     | 3 259      | 289       | 2.260      |             |             | 2026-09-04 14:54 |
| qwen3:8b   | 131072 | long   | stream     | GPU    | vLLM 0.22.0     | 9 180.9       | 139.4        | 128     | 977        | 195       | 1.527      |             |             | 2026-09-04 14:54 |
| qwen3:8b   | 131072 | long   | stream     | GPU    | **loken 0.1.0** | **2 094.9**   | **142.9**    | **128** | **1 061**  | **243**   | **1.895**  | **+2.5%**   | -19.4%      | 2026-09-04 14:54 |
| qwen3:8b   | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 85.1         | 128     | 3 219      | 290       | 2.263      |             |             | 2026-09-04 13:00 |
| qwen3:8b   | 131072 | medium | non-stream | GPU    | vLLM 0.22.0     |  -            | 137.5        | 128     | 966        | 178       | 1.390      |             |             | 2026-09-04 13:00 |
| qwen3:8b   | 131072 | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **138.8**    | **128** | **950**    | **234**   | **1.825**  | **+0.9%**   | -23.8%      | 2026-09-04 13:00 |
| qwen3:8b   | 131072 | medium | stream     | CPU    | Ollama 0.32.6   | 88.1          | 4.5          | 128     | 30 753     | 672       | 5.247      |             |             | 2026-09-06 01:49 |
| qwen3:8b   | 131072 | medium | stream     | CPU    | **loken 0.1.0** | **48.9**      | **4.8**      | **128** | **28 199** | **644**   | **5.033**  | **+5.9%**   | **+4.3%**   | 2026-09-06 01:49 |
| qwen3:8b   | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 213.7         | 97.3         | 128     | 3 202      | 288       | 2.251      |             |             | 2026-09-04 11:10 |
| qwen3:8b   | 131072 | medium | stream     | GPU    | vLLM 0.22.0     | 1 059.0       | 139.6        | 128     | 975        | 201       | 1.567      |             |             | 2026-09-04 11:10 |
| qwen3:8b   | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **1 446.0**   | **145.6**    | **128** | **946**    | **230**   | **1.794**  | **+4.3%**   | -12.6%      | 2026-09-04 11:10 |
| qwen3:8b   | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 68.8         | 128     | 4 344      | 312       | 2.440      |             |             | 2026-09-04 09:10 |
| qwen3:8b   | 131072 | short  | non-stream | GPU    | vLLM 0.22.0     |  -            | 135.0        | 128     | 985        | 199       | 1.553      |             |             | 2026-09-04 09:10 |
| qwen3:8b   | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **148.9**    | **128** | **905**    | **202**   | **1.577**  | **+10.3%**  | -1.5%       | 2026-09-04 09:10 |
| qwen3:8b   | 131072 | short  | stream     | CPU    | Ollama 0.32.6   | 31.0          | 4.5          | 128     | 30 451     | 673       | 5.257      |             |             | 2026-09-05 22:53 |
| qwen3:8b   | 131072 | short  | stream     | CPU    | **loken 0.1.0** | **25.1**      | **4.8**      | **128** | **27 666** | **635**   | **4.958**  | **+6.4%**   | **+6.0%**   | 2026-09-05 22:53 |
| qwen3:8b   | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 75.5          | 97.3         | 128     | 3 177      | 288       | 2.247      |             |             | 2026-09-04 03:13 |
| qwen3:8b   | 131072 | short  | stream     | GPU    | vLLM 0.22.0     | 697.9         | 139.6        | 128     | 967        | 189       | 1.474      |             |             | 2026-09-04 03:13 |
| qwen3:8b   | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **243.3**     | **152.8**    | **128** | **924**    | **230**   | **1.793**  | **+9.4%**   | -17.8%      | 2026-09-04 03:13 |

### qwen35

![qwen35](img/family-qwen35.svg)

| Model          | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode    | Δ energy   | Date             |
|----------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|-------------|------------|------------------|
| qwen3.5:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 62.0         | 128     | 4 212      | 350     | 2.731     |             |            | 2026-09-04 01:04 |
| qwen3.5:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **113.7**    | **128** | **1 327**  | **261** | **2.038** | **+83.4%**  | **+34.1%** | 2026-09-04 01:04 |
| qwen3.5:latest | 4096   | long   | stream     | CPU    | Ollama 0.32.6   | 407.7         | 4.3          | 128     | 33 542     | 715     | 5.584     |             |            | 2026-09-05 07:46 |
| qwen3.5:latest | 4096   | long   | stream     | CPU    | **loken 0.1.0** | **2 526.2**   | **4.2**      | **128** | **33 086** | **717** | **5.599** | -2.4%       | -0.3%      | 2026-09-05 07:46 |
| qwen3.5:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 639.6         | 75.8         | 128     | 4 221      | 350     | 2.734     |             |            | 2026-09-03 23:26 |
| qwen3.5:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **7 894.9**   | **114.2**    | **128** | **1 384**  | **261** | **2.035** | **+50.6%**  | **+34.3%** | 2026-09-03 23:26 |
| qwen3.5:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 61.0         | 128     | 4 244      | 341     | 2.662     |             |            | 2026-09-03 21:49 |
| qwen3.5:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **120.7**    | **128** | **1 264**  | **258** | **2.013** | **+97.9%**  | **+32.2%** | 2026-09-03 21:49 |
| qwen3.5:latest | 4096   | medium | stream     | CPU    | Ollama 0.32.6   | 61.7          | 4.3          | 128     | 32 397     | 719     | 5.614     |             |            | 2026-09-05 02:46 |
| qwen3.5:latest | 4096   | medium | stream     | CPU    | **loken 0.1.0** | **272.5**     | **4.2**      | **128** | **31 557** | **722** | **5.637** | -2.5%       | -0.4%      | 2026-09-05 02:46 |
| qwen3.5:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 101.9         | 76.3         | 128     | 4 208      | 349     | 2.726     |             |            | 2026-09-03 20:09 |
| qwen3.5:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 204.9**   | **123.4**    | **128** | **1 287**  | **263** | **2.053** | **+61.7%**  | **+32.8%** | 2026-09-03 20:09 |
| qwen3.5:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 59.3         | 128     | 5 011      | 358     | 2.797     |             |            | 2026-09-03 18:26 |
| qwen3.5:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **122.8**    | **128** | **1 224**  | **253** | **1.975** | **+107.0%** | **+41.6%** | 2026-09-03 18:26 |
| qwen3.5:latest | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 21.7          | 4.3          | 128     | 32 412     | 717     | 5.604     |             |            | 2026-09-04 22:11 |
| qwen3.5:latest | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **212.1**     | **4.1**      | **128** | **35 305** | **963** | **7.520** | -4.4%       | -25.5%     | 2026-09-04 22:11 |
| qwen3.5:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 34.6          | 75.4         | 128     | 4 228      | 332     | 2.597     |             |            | 2026-09-03 16:49 |
| qwen3.5:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **217.6**     | **126.3**    | **128** | **1 280**  | **262** | **2.046** | **+67.6%**  | **+27.0%** | 2026-09-03 16:49 |
| qwen3.5:latest | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 61.8         | 128     | 4 367      | 356     | 2.784     |             |            | 2026-09-04 16:50 |
| qwen3.5:latest | 131072 | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **111.3**    | **128** | **1 351**  | **263** | **2.052** | **+80.0%**  | **+35.7%** | 2026-09-04 16:50 |
| qwen3.5:latest | 131072 | long   | stream     | CPU    | Ollama 0.32.6   | 377.5         | 4.3          | 128     | 33 966     | 716     | 5.591     |             |            | 2026-09-06 04:36 |
| qwen3.5:latest | 131072 | long   | stream     | CPU    | **loken 0.1.0** | **2 362.9**   | **4.2**      | **128** | **33 110** | **717** | **5.599** | -2.3%       | -0.2%      | 2026-09-06 04:36 |
| qwen3.5:latest | 131072 | long   | stream     | GPU    | Ollama 0.32.6   | 645.4         | 74.4         | 128     | 4 329      | 358     | 2.794     |             |            | 2026-09-04 14:50 |
| qwen3.5:latest | 131072 | long   | stream     | GPU    | **loken 0.1.0** | **7 696.3**   | **113.6**    | **128** | **1 385**  | **273** | **2.133** | **+52.7%**  | **+31.0%** | 2026-09-04 14:50 |
| qwen3.5:latest | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 60.3         | 128     | 4 395      | 335     | 2.617     |             |            | 2026-09-04 12:56 |
| qwen3.5:latest | 131072 | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **121.1**    | **128** | **1 238**  | **249** | **1.946** | **+100.9%** | **+34.5%** | 2026-09-04 12:56 |
| qwen3.5:latest | 131072 | medium | stream     | CPU    | Ollama 0.32.6   | 59.6          | 4.3          | 128     | 32 638     | 713     | 5.570     |             |            | 2026-09-06 01:43 |
| qwen3.5:latest | 131072 | medium | stream     | CPU    | **loken 0.1.0** | **355.7**     | **4.2**      | **128** | **31 718** | **716** | **5.592** | -3.2%       | -0.4%      | 2026-09-06 01:43 |
| qwen3.5:latest | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 97.6          | 76.6         | 128     | 4 265      | 335     | 2.615     |             |            | 2026-09-04 11:06 |
| qwen3.5:latest | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **1 203.2**   | **123.7**    | **128** | **1 290**  | **257** | **2.011** | **+61.6%**  | **+30.1%** | 2026-09-04 11:06 |
| qwen3.5:latest | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 47.9         | 128     | 6 358      | 378     | 2.954     |             |            | 2026-09-04 09:06 |
| qwen3.5:latest | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **125.0**    | **128** | **1 210**  | **254** | **1.986** | **+161.3%** | **+48.7%** | 2026-09-04 09:06 |
| qwen3.5:latest | 131072 | short  | stream     | CPU    | Ollama 0.32.6   | 21.6          | 4.3          | 128     | 32 682     | 716     | 5.591     |             |            | 2026-09-05 22:48 |
| qwen3.5:latest | 131072 | short  | stream     | CPU    | **loken 0.1.0** | **118.5**     | **4.2**      | **128** | **31 282** | **715** | **5.584** | -2.1%       | **+0.1%**  | 2026-09-05 22:48 |
| qwen3.5:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 33.3          | 75.3         | 128     | 4 301      | 353     | 2.760     |             |            | 2026-09-04 03:10 |
| qwen3.5:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **475.4**     | **124.8**    | **128** | **1 299**  | **242** | **1.888** | **+65.8%**  | **+46.2%** | 2026-09-04 03:10 |

### qwen35moe

![qwen35moe](img/family-qwen35moe.svg)

| Model       | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode    | Δ energy   | Date             |
|-------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|-------------|------------|------------------|
| qwen3.5:35b | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 84.3         | 128     | 19 547     | 202     | 1.580     |             |            | 2026-09-04 01:03 |
| qwen3.5:35b | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **125.1**    | **128** | **1 468**  | **176** | **1.371** | **+48.4%**  | **+15.3%** | 2026-09-04 01:03 |
| qwen3.5:35b | 4096   | long   | stream     | CPU    | Ollama 0.32.6   | 402.0         | 9.3          | 128     | 20 046     | 338     | 2.644     |             |            | 2026-09-05 07:42 |
| qwen3.5:35b | 4096   | long   | stream     | CPU    | **loken 0.1.0** | **2 731.2**   | **8.7**      | **128** | **31 494** | **353** | **2.754** | -7.0%       | -4.0%      | 2026-09-05 07:42 |
| qwen3.5:35b | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 633.9         | 117.6        | 128     | 5 254      | 196     | 1.531     |             |            | 2026-09-03 23:24 |
| qwen3.5:35b | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **7 134.7**   | **127.3**    | **128** | **1 499**  | **169** | **1.317** | **+8.3%**   | **+16.3%** | 2026-09-03 23:24 |
| qwen3.5:35b | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 80.6         | 128     | 19 881     | 207     | 1.617     |             |            | 2026-09-03 21:47 |
| qwen3.5:35b | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **131.3**    | **128** | **1 342**  | **165** | **1.288** | **+62.8%**  | **+25.5%** | 2026-09-03 21:47 |
| qwen3.5:35b | 4096   | medium | stream     | CPU    | Ollama 0.32.6   | 67.3          | 9.4          | 128     | 19 494     | 340     | 2.653     |             |            | 2026-09-05 02:41 |
| qwen3.5:35b | 4096   | medium | stream     | CPU    | **loken 0.1.0** | **449.9**     | **8.5**      | **128** | **20 471** | **359** | **2.807** | -9.0%       | -5.5%      | 2026-09-05 02:41 |
| qwen3.5:35b | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 87.3          | 112.8        | 128     | 20 303     | 205     | 1.603     |             |            | 2026-09-03 20:07 |
| qwen3.5:35b | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **654.2**     | **133.8**    | **128** | **1 399**  | **169** | **1.321** | **+18.7%**  | **+21.4%** | 2026-09-03 20:07 |
| qwen3.5:35b | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 83.3         | 128     | 9 355      | 197     | 1.537     |             |            | 2026-09-03 18:24 |
| qwen3.5:35b | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **117.0**    | **128** | **1 638**  | **175** | **1.366** | **+40.5%**  | **+12.5%** | 2026-09-03 18:24 |
| qwen3.5:35b | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 24.2          | 9.3          | 128     | 19 491     | 340     | 2.656     |             |            | 2026-09-04 22:06 |
| qwen3.5:35b | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **134.4**     | **8.7**      | **128** | **19 703** | **353** | **2.757** | -7.1%       | -3.7%      | 2026-09-04 22:06 |
| qwen3.5:35b | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 23.0          | 115.0        | 128     | 5 046      | 201     | 1.573     |             |            | 2026-09-03 16:48 |
| qwen3.5:35b | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **469.1**     | **132.9**    | **128** | **1 359**  | **167** | **1.305** | **+15.6%**  | **+20.6%** | 2026-09-03 16:48 |
| qwen3.5:35b | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 85.8         | 128     | 20 313     | 204     | 1.596     |             |            | 2026-09-04 16:48 |
| qwen3.5:35b | 131072 | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **122.9**    | **128** | **1 477**  | **175** | **1.366** | **+43.3%**  | **+16.9%** | 2026-09-04 16:48 |
| qwen3.5:35b | 131072 | long   | stream     | CPU    | Ollama 0.32.6   | 421.6         | 9.4          | 128     | 20 210     | 337     | 2.633     |             |            | 2026-09-06 04:32 |
| qwen3.5:35b | 131072 | long   | stream     | CPU    | **loken 0.1.0** | **3 473.8**   | **8.6**      | **128** | **33 191** | **351** | **2.744** | -7.9%       | -4.1%      | 2026-09-06 04:32 |
| qwen3.5:35b | 131072 | long   | stream     | GPU    | **loken 0.1.0** | **8 744.8**   | **124.9**    | **128** | **1 473**  | **177** | **1.386** |             |            | 2026-09-04 14:49 |
| qwen3.5:35b | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 80.5         | 128     | 5 180      | 207     | 1.620     |             |            | 2026-09-04 12:55 |
| qwen3.5:35b | 131072 | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **128.7**    | **128** | **1 347**  | **165** | **1.290** | **+59.8%**  | **+25.6%** | 2026-09-04 12:55 |
| qwen3.5:35b | 131072 | medium | stream     | CPU    | Ollama 0.32.6   | 65.1          | 9.4          | 128     | 19 697     | 338     | 2.638     |             |            | 2026-09-06 01:39 |
| qwen3.5:35b | 131072 | medium | stream     | CPU    | **loken 0.1.0** | **352.6**     | **8.6**      | **128** | **29 173** | **358** | **2.799** | -8.4%       | -5.7%      | 2026-09-06 01:39 |
| qwen3.5:35b | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 91.9          | 113.3        | 128     | 13 630     | 206     | 1.608     |             |            | 2026-09-04 11:05 |
| qwen3.5:35b | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **1 188.7**   | **134.2**    | **128** | **1 382**  | **171** | **1.334** | **+18.5%**  | **+20.6%** | 2026-09-04 11:05 |
| qwen3.5:35b | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 53.2         | 128     | 7 250      | 243     | 1.896     |             |            | 2026-09-04 09:05 |
| qwen3.5:35b | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **135.9**    | **128** | **1 312**  | **153** | **1.199** | **+155.3%** | **+58.2%** | 2026-09-04 09:05 |
| qwen3.5:35b | 131072 | short  | stream     | CPU    | Ollama 0.32.6   | 23.9          | 9.4          | 128     | 19 552     | 338     | 2.643     |             |            | 2026-09-05 22:44 |
| qwen3.5:35b | 131072 | short  | stream     | CPU    | **loken 0.1.0** | **193.0**     | **8.7**      | **128** | **19 453** | **352** | **2.752** | -7.4%       | -4.0%      | 2026-09-05 22:44 |
| qwen3.5:35b | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 32.7          | 114.6        | 128     | 20 057     | 200     | 1.563     |             |            | 2026-09-04 03:09 |
| qwen3.5:35b | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **379.1**     | **134.2**    | **128** | **1 325**  | **166** | **1.297** | **+17.1%**  | **+20.5%** | 2026-09-04 03:09 |

### qwen3moe

![qwen3moe](img/family-qwen3moe.svg)

| Model           | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode   | Δ energy   | Date             |
|-----------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|------------|------------|------------------|
| qwen3-coder:30b | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent |            | 2026-09-04 01:00 |
| qwen3-coder:30b | 4096   | long   | non-stream | GPU    | vLLM 0.22.0     |  -            | 124.4        | 128     | 1 081      | 164     | 1.282     |            |            | 2026-09-04 01:00 |
| qwen3-coder:30b | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **118.8**    | **128** | **1 429**  | **242** | **1.888** | -4.5%      | -32.1%     | 2026-09-04 01:00 |
| qwen3-coder:30b | 4096   | long   | stream     | CPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent |            | 2026-09-05 07:37 |
| qwen3-coder:30b | 4096   | long   | stream     | CPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent |            | 2026-09-05 07:37 |
| qwen3-coder:30b | 4096   | long   | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent |            | 2026-09-03 23:22 |
| qwen3-coder:30b | 4096   | long   | stream     | GPU    | vLLM 0.22.0     | 10 261.2      | 134.2        | 128     | 1 036      | 164     | 1.278     |            |            | 2026-09-03 23:22 |
| qwen3-coder:30b | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **631.5**     | **177.7**    | **128** | **1 416**  | **242** | **1.890** | **+32.4%** | -32.4%     | 2026-09-03 23:22 |
| qwen3-coder:30b | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 121.5        | 128     | 3 877      | 159     | 1.246     |            |            | 2026-09-03 21:44 |
| qwen3-coder:30b | 4096   | medium | non-stream | GPU    | vLLM 0.22.0     |  -            | 124.6        | 128     | 1 062      | 170     | 1.332     |            |            | 2026-09-03 21:44 |
| qwen3-coder:30b | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **158.5**    | **128** | **1 110**  | **167** | **1.303** | **+27.2%** | -4.4%      | 2026-09-03 21:44 |
| qwen3-coder:30b | 4096   | medium | stream     | CPU    | Ollama 0.32.6   | 129.6         | 10.9         | 128     | 16 766     | 293     | 2.287     |            |            | 2026-09-05 02:37 |
| qwen3-coder:30b | 4096   | medium | stream     | CPU    | **loken 0.1.0** | **32.0**      | **10.5**     | **128** | **18 920** | **323** | **2.527** | -3.5%      | -9.5%      | 2026-09-05 02:37 |
| qwen3-coder:30b | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 188.5         | 148.1        | 128     | 3 947      | 176     | 1.374     |            |            | 2026-09-03 20:04 |
| qwen3-coder:30b | 4096   | medium | stream     | GPU    | vLLM 0.22.0     | 1 938.7       | 126.4        | 128     | 1 051      | 172     | 1.340     |            |            | 2026-09-03 20:04 |
| qwen3-coder:30b | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **459.7**     | **184.1**    | **128** | **1 116**  | **168** | **1.312** | **+24.3%** | **+2.2%**  | 2026-09-03 20:04 |
| qwen3-coder:30b | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 122.7        | 128     | 3 866      | 164     | 1.285     |            |            | 2026-09-03 18:22 |
| qwen3-coder:30b | 4096   | short  | non-stream | GPU    | vLLM 0.22.0     |  -            | 134.2        | 128     | 991        | 163     | 1.275     |            |            | 2026-09-03 18:22 |
| qwen3-coder:30b | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **165.5**    | **128** | **1 052**  | **157** | **1.226** | **+23.3%** | **+4.0%**  | 2026-09-03 18:22 |
| qwen3-coder:30b | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 38.9          | 11.0         | 128     | 16 735     | 288     | 2.247     |            |            | 2026-09-04 22:03 |
| qwen3-coder:30b | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **18.5**      | **10.6**     | **128** | **18 275** | **305** | **2.380** | -3.6%      | -5.6%      | 2026-09-04 22:03 |
| qwen3-coder:30b | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 62.8          | 148.3        | 128     | 3 918      | 173     | 1.355     |            |            | 2026-09-03 16:46 |
| qwen3-coder:30b | 4096   | short  | stream     | GPU    | vLLM 0.22.0     | 633.6         | 144.9        | 128     | 969        | 164     | 1.278     |            |            | 2026-09-03 16:46 |
| qwen3-coder:30b | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **264.5**     | **184.8**    | **128** | **1 070**  | **160** | **1.253** | **+24.6%** | **+2.0%**  | 2026-09-03 16:46 |
| qwen3-coder:30b | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 70.5         | 128     | 5 454      | 216     | 1.689     |            |            | 2026-09-04 16:45 |
| qwen3-coder:30b | 131072 | long   | non-stream | GPU    | vLLM 0.22.0     |  -            | 124.3        | 128     | 1 084      | 159     | 1.241     |            |            | 2026-09-04 16:45 |
| qwen3-coder:30b | 131072 | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **119.0**    | **128** | **1 393**  | **248** | **1.937** | -4.3%      | -36.0%     | 2026-09-04 16:45 |
| qwen3-coder:30b | 131072 | long   | stream     | CPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent |            | 2026-09-06 04:27 |
| qwen3-coder:30b | 131072 | long   | stream     | CPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent |            | 2026-09-06 04:27 |
| qwen3-coder:30b | 131072 | long   | stream     | GPU    | Ollama 0.32.6   | 12.1          | 81.3         | 128     | 20 918     | 1 823   | 14.242    |            |            | 2026-09-04 14:47 |
| qwen3-coder:30b | 131072 | long   | stream     | GPU    | vLLM 0.22.0     | 10 138.2      | 126.3        | 128     | 1 075      | 172     | 1.341     |            |            | 2026-09-04 14:47 |
| qwen3-coder:30b | 131072 | long   | stream     | GPU    | **loken 0.1.0** | **565.1**     | **165.8**    | **128** | **1 558**  | **246** | **1.920** | **+31.3%** | -30.2%     | 2026-09-04 14:47 |
| qwen3-coder:30b | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 74.4         | 128     | 5 393      | 207     | 1.618     |            |            | 2026-09-04 12:53 |
| qwen3-coder:30b | 131072 | medium | non-stream | GPU    | vLLM 0.22.0     |  -            | 124.5        | 128     | 1 073      | 168     | 1.312     |            |            | 2026-09-04 12:53 |
| qwen3-coder:30b | 131072 | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **159.0**    | **128** | **1 109**  | **164** | **1.284** | **+27.7%** | **+2.2%**  | 2026-09-04 12:53 |
| qwen3-coder:30b | 131072 | medium | stream     | CPU    | Ollama 0.32.6   | 120.7         | 10.9         | 128     | 17 882     | 282     | 2.203     |            |            | 2026-09-06 01:35 |
| qwen3-coder:30b | 131072 | medium | stream     | CPU    | **loken 0.1.0** | **32.0**      | **10.6**     | **128** | **18 668** | **315** | **2.464** | -3.3%      | -10.6%     | 2026-09-06 01:35 |
| qwen3-coder:30b | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 164.8         | 82.1         | 128     | 5 485      | 215     | 1.678     |            |            | 2026-09-04 11:02 |
| qwen3-coder:30b | 131072 | medium | stream     | GPU    | vLLM 0.22.0     | 1 957.5       | 126.3        | 128     | 1 052      | 175     | 1.366     |            |            | 2026-09-04 11:02 |
| qwen3-coder:30b | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **453.2**     | **183.5**    | **128** | **1 105**  | **167** | **1.309** | **+45.3%** | **+4.4%**  | 2026-09-04 11:02 |
| qwen3-coder:30b | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 72.0         | 128     | 5 437      | 207     | 1.617     |            |            | 2026-09-04 09:02 |
| qwen3-coder:30b | 131072 | short  | non-stream | GPU    | vLLM 0.22.0     |  -            | 129.8        | 128     | 1 045      | 173     | 1.353     |            |            | 2026-09-04 09:02 |
| qwen3-coder:30b | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **174.6**    | **128** | **1 063**  | **134** | **1.046** | **+34.5%** | **+29.3%** | 2026-09-04 09:02 |
| qwen3-coder:30b | 131072 | short  | stream     | CPU    | Ollama 0.32.6   | 44.4          | 11.0         | 128     | 17 879     | 280     | 2.191     |            |            | 2026-09-05 22:40 |
| qwen3-coder:30b | 131072 | short  | stream     | CPU    | **loken 0.1.0** | **18.2**      | **10.6**     | **128** | **18 230** | **302** | **2.356** | -3.4%      | -7.0%      | 2026-09-05 22:40 |
| qwen3-coder:30b | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 73.1          | 84.1         | 128     | 5 367      | 207     | 1.615     |            |            | 2026-09-04 03:05 |
| qwen3-coder:30b | 131072 | short  | stream     | GPU    | vLLM 0.22.0     | 640.6         | 134.3        | 128     | 1 010      | 168     | 1.313     |            |            | 2026-09-04 03:05 |
| qwen3-coder:30b | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **260.2**     | **184.9**    | **128** | **1 076**  | **157** | **1.223** | **+37.7%** | **+7.3%**  | 2026-09-04 03:05 |

### qwen3next

![qwen3next](img/family-qwen3next.svg)

| Model                   | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms      | J/req   | J/token   | Δ decode   | Δ energy    | Date             |
|-------------------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|-------------|---------|-----------|------------|-------------|------------------|
| qwen3-coder-next:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 26.2         | 128     | 11 032      | 431     | 3.370     |            |             | 2026-09-04 00:56 |
| qwen3-coder-next:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **28.7**     | **128** | **18 940**  | **440** | **3.439** | **+9.6%**  | -2.0%       | 2026-09-04 00:56 |
| qwen3-coder-next:latest | 4096   | long   | stream     | CPU    | Ollama 0.32.6   | 458.5         | 7.6          | 128     | 45 517      | 408     | 3.185     |            |             | 2026-09-05 07:34 |
| qwen3-coder-next:latest | 4096   | long   | stream     | CPU    | **loken 0.1.0** | **50.9**      | **6.4**      | **128** | **123 711** | **576** | **4.502** | -15.5%     | -29.3%      | 2026-09-05 07:34 |
| qwen3-coder-next:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 592.8         | 28.1         | 128     | 11 080      | 430     | 3.359     |            |             | 2026-09-03 23:18 |
| qwen3-coder-next:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **4 661.8**   | **28.8**     | **128** | **12 205**  | **438** | **3.418** | **+2.4%**  | -1.7%       | 2026-09-03 23:18 |
| qwen3-coder-next:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 25.8         | 128     | 10 621      | 431     | 3.364     |            |             | 2026-09-03 21:40 |
| qwen3-coder-next:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **29.3**     | **128** | **10 608**  | **424** | **3.313** | **+13.6%** | **+1.5%**   | 2026-09-03 21:40 |
| qwen3-coder-next:latest | 4096   | medium | stream     | CPU    | Ollama 0.32.6   | 77.3          | 7.6          | 128     | 49 071      | 412     | 3.220     |            |             | 2026-09-05 02:34 |
| qwen3-coder-next:latest | 4096   | medium | stream     | CPU    | **loken 0.1.0** | **140.2**     | **6.6**      | **128** | **102 586** | **470** | **3.670** | -13.4%     | -12.2%      | 2026-09-05 02:34 |
| qwen3-coder-next:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 88.9          | 23.6         | 128     | 17 032      | 786     | 6.140     |            |             | 2026-09-03 19:58 |
| qwen3-coder-next:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **670.0**     | **18.1**     | **128** | **20 743**  | **618** | **4.826** | -23.4%     | **+27.2%**  | 2026-09-03 19:58 |
| qwen3-coder-next:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 25.6         | 128     | 10 397      | 434     | 3.389     |            |             | 2026-09-03 18:18 |
| qwen3-coder-next:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **29.3**     | **128** | **10 239**  | **424** | **3.316** | **+14.6%** | **+2.2%**   | 2026-09-03 18:18 |
| qwen3-coder-next:latest | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 27.8          | 7.6          | 128     | 43 122      | 411     | 3.209     |            |             | 2026-09-04 21:59 |
| qwen3-coder-next:latest | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **46.5**      | **6.7**      | **128** | **95 749**  | **454** | **3.550** | -11.2%     | -9.6%       | 2026-09-04 21:59 |
| qwen3-coder-next:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 31.7          | 27.9         | 128     | 10 418      | 434     | 3.393     |            |             | 2026-09-03 16:41 |
| qwen3-coder-next:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **303.9**     | **30.2**     | **128** | **9 760**   | **417** | **3.256** | **+8.2%**  | **+4.2%**   | 2026-09-03 16:41 |
| qwen3-coder-next:latest | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 23.8         | 128     | 11 348      | 457     | 3.572     |            |             | 2026-09-04 16:41 |
| qwen3-coder-next:latest | 131072 | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **29.5**     | **128** | **11 788**  | **427** | **3.340** | **+23.9%** | **+7.0%**   | 2026-09-04 16:41 |
| qwen3-coder-next:latest | 131072 | long   | stream     | CPU    | Ollama 0.32.6   | 455.9         | 7.6          | 128     | 45 707      | 408     | 3.191     |            |             | 2026-09-06 04:23 |
| qwen3-coder-next:latest | 131072 | long   | stream     | CPU    | **loken 0.1.0** | **269.2**     | **6.6**      | **128** | **110 591** | **478** | **3.736** | -13.7%     | -14.6%      | 2026-09-06 04:23 |
| qwen3-coder-next:latest | 131072 | long   | stream     | GPU    | **loken 0.1.0** | **7 280.4**   | **23.9**     | **128** | **15 042**  | **490** | **3.831** |            |             | 2026-09-04 14:41 |
| qwen3-coder-next:latest | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 23.6         | 128     | 10 809      | 462     | 3.606     |            |             | 2026-09-04 12:49 |
| qwen3-coder-next:latest | 131072 | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **29.1**     | **128** | **10 605**  | **435** | **3.402** | **+23.1%** | **+6.0%**   | 2026-09-04 12:49 |
| qwen3-coder-next:latest | 131072 | medium | stream     | CPU    | Ollama 0.32.6   | 86.9          | 7.6          | 128     | 46 688      | 405     | 3.167     |            |             | 2026-09-06 01:31 |
| qwen3-coder-next:latest | 131072 | medium | stream     | CPU    | **loken 0.1.0** | **69.4**      | **6.6**      | **128** | **102 502** | **470** | **3.670** | -14.0%     | -13.7%      | 2026-09-06 01:31 |
| qwen3-coder-next:latest | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 91.2          | 25.7         | 128     | 10 912      | 454     | 3.547     |            |             | 2026-09-04 10:58 |
| qwen3-coder-next:latest | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **844.6**     | **30.3**     | **128** | **10 186**  | **419** | **3.275** | **+18.1%** | **+8.3%**   | 2026-09-04 10:58 |
| qwen3-coder-next:latest | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 22.3         | 128     | 10 901      | 471     | 3.681     |            |             | 2026-09-04 08:58 |
| qwen3-coder-next:latest | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **28.0**     | **128** | **11 739**  | **432** | **3.378** | **+25.6%** | **+9.0%**   | 2026-09-04 08:58 |
| qwen3-coder-next:latest | 131072 | short  | stream     | CPU    | Ollama 0.32.6   | 29.0          | 7.6          | 128     | 49 830      | 406     | 3.175     |            |             | 2026-09-05 22:37 |
| qwen3-coder-next:latest | 131072 | short  | stream     | CPU    | **loken 0.1.0** | **33.7**      | **6.7**      | **128** | **89 728**  | **458** | **3.581** | -12.4%     | -11.3%      | 2026-09-05 22:37 |
| qwen3-coder-next:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 33.9          | 25.6         | 128     | 10 611      | 453     | 3.538     |            |             | 2026-09-04 03:01 |
| qwen3-coder-next:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **328.2**     | **30.4**     | **128** | **9 781**   | **418** | **3.265** | **+18.6%** | **+8.4%**   | 2026-09-04 03:01 |
| qwen3next:latest        | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 29.5         | 128     | 10 163      | 394     | 3.080     |            |             | 2026-09-04 01:12 |
| qwen3next:latest        | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **32.8**     | **128** | **10 439**  | **397** | **3.103** | **+11.2%** | -0.7%       | 2026-09-04 01:12 |
| qwen3next:latest        | 4096   | long   | stream     | CPU    | Ollama 0.32.6   | 448.6         | 7.5          | 128     | 53 950      | 406     | 3.171     |            |             | 2026-09-05 08:03 |
| qwen3next:latest        | 4096   | long   | stream     | CPU    | **loken 0.1.0** | **279.7**     | **6.5**      | **128** | **112 715** | **480** | **3.748** | -14.3%     | -15.4%      | 2026-09-05 08:03 |
| qwen3next:latest        | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 629.9         | 32.5         | 128     | 10 173      | 385     | 3.009     |            |             | 2026-09-03 23:33 |
| qwen3next:latest        | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **4 675.9**   | **32.9**     | **128** | **10 553**  | **397** | **3.100** | **+1.4%**  | -2.9%       | 2026-09-03 23:33 |
| qwen3next:latest        | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 30.0         | 128     | 9 894       | 388     | 3.027     |            |             | 2026-09-03 21:56 |
| qwen3next:latest        | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **33.4**     | **128** | **9 488**   | **389** | **3.038** | **+11.5%** | -0.4%       | 2026-09-03 21:56 |
| qwen3next:latest        | 4096   | medium | stream     | CPU    | Ollama 0.32.6   | 86.1          | 7.6          | 128     | 29 201      | 408     | 3.188     |            |             | 2026-09-05 03:00 |
| qwen3next:latest        | 4096   | medium | stream     | CPU    | **loken 0.1.0** | **64.1**      | **6.5**      | **128** | **98 725**  | **479** | **3.745** | -14.4%     | -14.9%      | 2026-09-05 03:00 |
| qwen3next:latest        | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 22.0          | 21.1         | 128     | 16 724      | 1 039   | 8.120     |            |             | 2026-09-03 20:17 |
| qwen3next:latest        | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 190.2**   | **18.9**     | **128** | **16 327**  | **591** | **4.615** | -10.2%     | **+75.9%**  | 2026-09-03 20:17 |
| qwen3next:latest        | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 28.9         | 128     | 9 765       | 404     | 3.153     |            |             | 2026-09-03 18:34 |
| qwen3next:latest        | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **33.5**     | **128** | **12 043**  | **396** | **3.093** | **+15.8%** | **+1.9%**   | 2026-09-03 18:34 |
| qwen3next:latest        | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 28.3          | 7.1          | 128     | 69 549      | 428     | 3.347     |            |             | 2026-09-04 22:33 |
| qwen3next:latest        | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **31.7**      | **6.7**      | **128** | **86 817**  | **458** | **3.578** | -6.5%      | -6.5%       | 2026-09-04 22:33 |
| qwen3next:latest        | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 33.8          | 32.8         | 128     | 9 490       | 388     | 3.030     |            |             | 2026-09-03 16:56 |
| qwen3next:latest        | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **309.8**     | **33.8**     | **128** | **8 941**   | **380** | **2.971** | **+2.8%**  | **+2.0%**   | 2026-09-03 16:56 |
| qwen3next:latest        | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 26.4         | 128     | 10 648      | 425     | 3.322     |            |             | 2026-09-04 16:57 |
| qwen3next:latest        | 131072 | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **32.7**     | **128** | **10 579**  | **400** | **3.128** | **+23.9%** | **+6.2%**   | 2026-09-04 16:57 |
| qwen3next:latest        | 131072 | long   | stream     | CPU    | Ollama 0.32.6   | 449.8         | 7.6          | 128     | 54 732      | 403     | 3.145     |            |             | 2026-09-06 04:53 |
| qwen3next:latest        | 131072 | long   | stream     | CPU    | **loken 0.1.0** | **232.9**     | **6.5**      | **128** | **110 380** | **478** | **3.736** | -14.9%     | -15.8%      | 2026-09-06 04:53 |
| qwen3next:latest        | 131072 | long   | stream     | GPU    | Ollama 0.32.6   | 537.3         | 12.9         | 128     | 20 312      | 1 499   | 11.711    |            |             | 2026-09-04 14:59 |
| qwen3next:latest        | 131072 | long   | stream     | GPU    | **loken 0.1.0** | **4 329.9**   | **20.7**     | **128** | **17 620**  | **560** | **4.377** | **+60.3%** | **+167.6%** | 2026-09-04 14:59 |
| qwen3next:latest        | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 26.3         | 128     | 10 125      | 428     | 3.341     |            |             | 2026-09-04 13:04 |
| qwen3next:latest        | 131072 | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **33.6**     | **128** | **9 450**   | **382** | **2.985** | **+27.7%** | **+11.9%**  | 2026-09-04 13:04 |
| qwen3next:latest        | 131072 | medium | stream     | CPU    | Ollama 0.32.6   | 86.6          | 7.6          | 128     | 53 167      | 405     | 3.162     |            |             | 2026-09-06 01:59 |
| qwen3next:latest        | 131072 | medium | stream     | CPU    | **loken 0.1.0** | **109.0**     | **6.6**      | **128** | **99 662**  | **462** | **3.607** | -12.9%     | -12.3%      | 2026-09-06 01:59 |
| qwen3next:latest        | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 96.0          | 28.1         | 128     | 10 227      | 433     | 3.380     |            |             | 2026-09-04 11:13 |
| qwen3next:latest        | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **817.0**     | **33.8**     | **128** | **9 410**   | **392** | **3.061** | **+20.4%** | **+10.4%**  | 2026-09-04 11:13 |
| qwen3next:latest        | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 17.0         | 128     | 19 935      | 599     | 4.680     |            |             | 2026-09-04 09:15 |
| qwen3next:latest        | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **10.7**     | **128** | **18 049**  | **958** | **7.481** | -37.1%     | -37.4%      | 2026-09-04 09:15 |
| qwen3next:latest        | 131072 | short  | stream     | CPU    | Ollama 0.32.6   | 30.1          | 7.6          | 128     | 29 160      | 407     | 3.178     |            |             | 2026-09-05 23:02 |
| qwen3next:latest        | 131072 | short  | stream     | CPU    | **loken 0.1.0** | **73.0**      | **6.7**      | **128** | **78 355**  | **451** | **3.521** | -12.5%     | -9.8%       | 2026-09-05 23:02 |
| qwen3next:latest        | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 33.6          | 28.4         | 128     | 10 087      | 418     | 3.267     |            |             | 2026-09-04 03:17 |
| qwen3next:latest        | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **300.5**     | **33.5**     | **128** | **9 016**   | **389** | **3.040** | **+18.0%** | **+7.5%**   | 2026-09-04 03:17 |

### smollm3

![smollm3](img/family-smollm3.svg)

| Model          | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode   | Δ energy   | Date             |
|----------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|------------|------------|------------------|
| smollm3:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 127.3        | 128     | 2 419      | 154     | 1.207     |            |            | 2026-09-04 01:13 |
| smollm3:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **246.3**    | **128** | **549**    | **118** | **0.925** | **+93.5%** | **+30.5%** | 2026-09-04 01:13 |
| smollm3:latest | 4096   | long   | stream     | CPU    | Ollama 0.32.6   | 646.3         | 11.2         | 128     | 13 269     | 276     | 2.158     |            |            | 2026-09-05 08:05 |
| smollm3:latest | 4096   | long   | stream     | CPU    | **loken 0.1.0** | **162.0**     | **10.4**     | **128** | **14 029** | **348** | **2.721** | -6.6%      | -20.7%     | 2026-09-05 08:05 |
| smollm3:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 1 012.1       | 172.9        | 128     | 2 384      | 154     | 1.206     |            |            | 2026-09-03 23:34 |
| smollm3:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **3 818.9**   | **272.9**    | **128** | **558**    | **115** | **0.896** | **+57.8%** | **+34.6%** | 2026-09-03 23:34 |
| smollm3:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 127.2        | 128     | 2 433      | 154     | 1.202     |            |            | 2026-09-03 21:57 |
| smollm3:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **239.8**    | **128** | **567**    | **121** | **0.944** | **+88.4%** | **+27.3%** | 2026-09-03 21:57 |
| smollm3:latest | 4096   | medium | stream     | CPU    | Ollama 0.32.6   | 110.9         | 11.4         | 128     | 12 664     | 273     | 2.136     |            |            | 2026-09-05 03:02 |
| smollm3:latest | 4096   | medium | stream     | CPU    | **loken 0.1.0** | **91.1**      | **10.7**     | **128** | **12 639** | **387** | **3.026** | -6.1%      | -29.4%     | 2026-09-05 03:02 |
| smollm3:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 147.4         | 169.7        | 128     | 2 719      | 151     | 1.183     |            |            | 2026-09-03 20:18 |
| smollm3:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **741.4**     | **260.0**    | **128** | **578**    | **128** | **0.999** | **+53.2%** | **+18.4%** | 2026-09-03 20:18 |
| smollm3:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 109.8        | 128     | 3 346      | 170     | 1.327     |            |            | 2026-09-03 18:35 |
| smollm3:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **152.8**    | **128** | **899**    | **133** | **1.043** | **+39.2%** | **+27.2%** | 2026-09-03 18:35 |
| smollm3:latest | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 39.8          | 11.4         | 128     | 12 665     | 268     | 2.094     |            |            | 2026-09-04 22:35 |
| smollm3:latest | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **35.6**      | **10.8**     | **128** | **12 586** | **375** | **2.929** | -5.0%      | -28.5%     | 2026-09-04 22:35 |
| smollm3:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 63.4          | 172.0        | 128     | 2 390      | 162     | 1.264     |            |            | 2026-09-03 16:57 |
| smollm3:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **257.4**     | **266.8**    | **128** | **567**    | **122** | **0.955** | **+55.2%** | **+32.4%** | 2026-09-03 16:57 |
| smollm3:latest | 131072 | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 132.6        | 128     | 2 486      | 156     | 1.220     |            |            | 2026-09-04 16:58 |
| smollm3:latest | 131072 | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **246.2**    | **128** | **551**    | **115** | **0.898** | **+85.7%** | **+35.8%** | 2026-09-04 16:58 |
| smollm3:latest | 131072 | long   | stream     | CPU    | Ollama 0.32.6   | 559.3         | 11.3         | 128     | 13 483     | 273     | 2.129     |            |            | 2026-09-06 04:55 |
| smollm3:latest | 131072 | long   | stream     | CPU    | **loken 0.1.0** | **152.6**     | **10.5**     | **128** | **13 947** | **351** | **2.740** | -6.9%      | -22.3%     | 2026-09-06 04:55 |
| smollm3:latest | 131072 | long   | stream     | GPU    | Ollama 0.32.6   | 461.9         | 132.2        | 128     | 3 768      | 189     | 1.478     |            |            | 2026-09-04 15:00 |
| smollm3:latest | 131072 | long   | stream     | GPU    | **loken 0.1.0** | **3 147.1**   | **168.9**    | **128** | **890**    | **140** | **1.098** | **+27.8%** | **+34.7%** | 2026-09-04 15:00 |
| smollm3:latest | 131072 | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 131.9        | 128     | 2 513      | 157     | 1.230     |            |            | 2026-09-04 13:05 |
| smollm3:latest | 131072 | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **251.9**    | **128** | **551**    | **111** | **0.871** | **+91.0%** | **+41.3%** | 2026-09-04 13:05 |
| smollm3:latest | 131072 | medium | stream     | CPU    | Ollama 0.32.6   | 105.6         | 11.4         | 128     | 13 013     | 268     | 2.095     |            |            | 2026-09-06 02:01 |
| smollm3:latest | 131072 | medium | stream     | CPU    | **loken 0.1.0** | **88.7**      | **10.7**     | **128** | **12 606** | **380** | **2.967** | -6.2%      | -29.4%     | 2026-09-06 02:01 |
| smollm3:latest | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 152.7         | 171.3        | 128     | 2 477      | 163     | 1.273     |            |            | 2026-09-04 11:14 |
| smollm3:latest | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **1 605.8**   | **261.5**    | **128** | **556**    | **104** | **0.812** | **+52.6%** | **+56.7%** | 2026-09-04 11:14 |
| smollm3:latest | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 133.1        | 128     | 2 505      | 160     | 1.247     |            |            | 2026-09-04 09:16 |
| smollm3:latest | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **239.2**    | **128** | **556**    | **120** | **0.938** | **+79.6%** | **+33.0%** | 2026-09-04 09:16 |
| smollm3:latest | 131072 | short  | stream     | CPU    | Ollama 0.32.6   | 39.2          | 11.5         | 128     | 12 956     | 267     | 2.085     |            |            | 2026-09-05 23:04 |
| smollm3:latest | 131072 | short  | stream     | CPU    | **loken 0.1.0** | **43.1**      | **10.8**     | **128** | **12 397** | **383** | **2.990** | -5.7%      | -30.3%     | 2026-09-05 23:04 |
| smollm3:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 64.3          | 172.6        | 128     | 2 490      | 158     | 1.235     |            |            | 2026-09-04 03:18 |
| smollm3:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **727.6**     | **269.2**    | **128** | **550**    | **113** | **0.880** | **+56.0%** | **+40.4%** | 2026-09-04 03:18 |
