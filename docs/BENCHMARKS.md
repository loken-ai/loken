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
[assay](https://github.com/loken-ai/assay): 3 iterations, greedy, 128 tokens, one engine build.
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






































































## Reading the prefill column

On a short prompt it does not compare the engines. The short prompt is 15 tokens, and each
engine reports a prefill rate from its own timer: on gpt-oss:20b ollama reported 652 tok/s
where its own time to first token implies 35, and this engine reported 49 373 where its time to
first token implies 728. Both exclude more than they include at that length, and they exclude
different things.

It does not recover at the medium prompt either. Over the first four models measured there,
each engine's self-reported rate against what its own time to first token implies:

| model | ollama says | ollama's TTFT implies | this engine says | its TTFT implies |
|---|---:|---:|---:|---:|
| deepcoder:14b | 2 304 | 205 | 1 635 | 831 |
| deepseek-r1:32b | 1 131 | 201 | 585 | 478 |
| deepseek-r1:70b-q3ks | 298 | 91 | 252 | 230 |
| deepseek-r1:70b | 67 | 46 | 21 | 21 |

On three of the four the column reverses the verdict: it reads as an ollama win while the time
to first token is two to four times better here. Read the TTFT column instead. The prefill
figures are each engine's own bookkeeping, and only the times are measured by the bench.

































































































### ernie4_5

![ernie4_5](img/family-ernie4_5.svg)

| Model           | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token   | Δ decode    | Δ energy | Date             |
|-----------------|------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|-----------|-------------|----------|------------------|
| ernie4-5:latest | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 87 143.4      | 481.2        | 128     | 1 408     | 116     | 0.907     |             |          | 2026-08-18 01:14 |
| ernie4-5:latest | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **219 434.0** | **519.7**    | **98**  | **242**   | **33**  | **0.336** |             |          | 2026-08-18 01:14 |
| ernie4-5:latest | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 84 524.2      | 483.0        | 128     | 1 896     | 148     | 1.156     |             |          | 2026-08-18 01:14 |
| ernie4-5:latest | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **187 373.1** | **512.7**    | **98**  | **245**   | **32**  | **0.327** |             |          | 2026-08-18 01:14 |
| ernie4-5:latest | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 11 680.2      | 207.7        | 80      | 1 462     |  -      |  -        |             |          | 2026-09-01 14:30 |
| ernie4-5:latest | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **22 700.7**  | **483.5**    | **128** | **273**   | ** - ** | ** - **   |             |          | 2026-09-01 14:30 |
| ernie4-5:latest | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 11 943.1      | 405.6        | 80      | 1 501     |  -      |  -        |             |          | 2026-09-01 12:52 |
| ernie4-5:latest | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **9 633.0**   | **570.7**    | **128** | **287**   | ** - ** | ** - **   |             |          | 2026-09-01 12:52 |
| ernie4-5:latest | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 4 151.8       | 226.2        | 128     | 1 588     |  -      |  -        |             |          | 2026-09-01 11:13 |
| ernie4-5:latest | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **6 060.6**   | **496.0**    | **128** | **264**   | ** - ** | ** - **   | **+119.3%** |          | 2026-09-01 11:13 |
| ernie4-5:latest | 4096 | short  | stream     | CPU    | Ollama 0.32.6   | 3 766.7       | 52.7         | 128     | 3 614     | 165     | 1.286     |             |          | 2026-08-18 01:14 |
| ernie4-5:latest | 4096 | short  | stream     | CPU    | **loken 0.1.0** | **3 228.9**   | **50.2**     | **104** | **2 180** | **95**  | **0.915** |             |          | 2026-08-18 01:14 |
| ernie4-5:latest | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 4 129.1       | 418.9        | 128     | 1 578     |  -      |  -        |             |          | 2026-09-01 09:35 |
| ernie4-5:latest | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **2 840.4**   | **600.1**    | **128** | **267**   | ** - ** | ** - **   | **+43.2%**  |          | 2026-09-01 09:35 |

### gemma4

![gemma4](img/family-gemma4.svg)

| Model         | Ctx   | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode    | Δ energy    | Date             |
|---------------|-------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|-------------|-------------|------------------|
| gemma4:12b    | 4096  | long   | non-stream | GPU    | Ollama 0.32.6   | 2 184.5       | 72.7         | 128     | 4 637      | 581     | 4.540     |             |             | 2026-08-18 01:14 |
| gemma4:12b    | 4096  | long   | non-stream | GPU    | **loken 0.1.0** | **12 322.1**  | **74.8**     | **128** | **1 789**  | **420** | **3.283** | **+2.8%**   | **+38.3%**  | 2026-08-18 01:14 |
| gemma4:12b    | 4096  | long   | stream     | GPU    | Ollama 0.32.6   | 2 082.2       | 72.1         | 128     | 4 817      | 592     | 4.624     |             |             | 2026-08-18 01:14 |
| gemma4:12b    | 4096  | long   | stream     | GPU    | **loken 0.1.0** | **12 532.8**  | **72.9**     | **128** | **1 822**  | **422** | **3.296** | **+1.1%**   | **+40.3%**  | 2026-08-18 01:14 |
| gemma4:12b    | 4096  | medium | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent  |             | 2026-09-01 14:33 |
| gemma4:12b    | 4096  | medium | non-stream | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent  |             | 2026-09-01 14:33 |
| gemma4:12b    | 4096  | medium | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 74      |  -         |  -      |  -        | incoherent  |             | 2026-09-01 12:54 |
| gemma4:12b    | 4096  | medium | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent  |             | 2026-09-01 12:54 |
| gemma4:12b    | 4096  | short  | non-stream | GPU    | Ollama 0.32.6   | 288.1         | 22.4         | 61      | 3 886      |  -      |  -        |             |             | 2026-09-01 11:16 |
| gemma4:12b    | 4096  | short  | non-stream | GPU    | **loken 0.1.0** | **517.3**     | **71.8**     | **128** | **1 784**  | ** - ** | ** - **   |             |             | 2026-09-01 11:16 |
| gemma4:12b    | 4096  | short  | stream     | GPU    | Ollama 0.32.6   | 289.3         | 42.6         | 61      | 5 331      |  -      |  -        |             |             | 2026-09-01 09:37 |
| gemma4:12b    | 4096  | short  | stream     | GPU    | **loken 0.1.0** | **432.3**     | **71.9**     | **128** | **1 837**  | ** - ** | ** - **   |             |             | 2026-09-01 09:37 |
| gemma4:26b    | 4096  | long   | non-stream | GPU    | Ollama 0.32.6   | 2 184.8       | 98.5         | 128     | 5 135      | 480     | 3.750     |             |             | 2026-08-18 01:14 |
| gemma4:26b    | 4096  | long   | non-stream | GPU    | **loken 0.1.0** | **1 169.0**   | **88.3**     | **128** | **1 754**  | **274** | **2.139** | -10.3%      | **+75.3%**  | 2026-08-18 01:14 |
| gemma4:26b    | 4096  | long   | stream     | GPU    | Ollama 0.32.6   | 2 158.9       | 97.2         | 128     | 5 137      | 485     | 3.786     |             |             | 2026-08-18 01:14 |
| gemma4:26b    | 4096  | long   | stream     | GPU    | **loken 0.1.0** | **1 173.7**   | **85.8**     | **128** | **1 803**  | **274** | **2.137** | -11.7%      | **+77.2%**  | 2026-08-18 01:14 |
| gemma4:26b    | 4096  | medium | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent  |             | 2026-09-01 14:35 |
| gemma4:26b    | 4096  | medium | non-stream | GPU    | **loken 0.1.0** | **490.5**     | **85.4**     | **128** | **1 519**  | ** - ** | ** - **   |             |             | 2026-09-01 14:35 |
| gemma4:26b    | 4096  | medium | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent  |             | 2026-09-01 12:57 |
| gemma4:26b    | 4096  | medium | stream     | GPU    | **loken 0.1.0** | **469.9**     | **95.4**     | **128** | **1 565**  | ** - ** | ** - **   |             |             | 2026-09-01 12:57 |
| gemma4:26b    | 4096  | short  | non-stream | GPU    | Ollama 0.32.6   | 84.5          | 11.4         | 128     | 4 582      |  -      |  -        |             |             | 2026-09-01 11:18 |
| gemma4:26b    | 4096  | short  | non-stream | GPU    | **loken 0.1.0** | **307.8**     | **88.4**     | **128** | **1 463**  | ** - ** | ** - **   | **+676.5%** |             | 2026-09-01 11:18 |
| gemma4:26b    | 4096  | short  | stream     | GPU    | Ollama 0.32.6   | 5.1           | 101.3        | 93      | 5 661      |  -      |  -        |             |             | 2026-09-01 09:40 |
| gemma4:26b    | 4096  | short  | stream     | GPU    | **loken 0.1.0** | **245.5**     | **95.5**     | **128** | **1 519**  | ** - ** | ** - **   |             |             | 2026-09-01 09:40 |
| gemma4:31b    | 4096  | long   | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           |  -      |  -         |  -      |  -        | incoherent  |             | 2026-08-18 01:14 |
| gemma4:31b    | 4096  | long   | non-stream | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent  |             | 2026-08-18 01:14 |
| gemma4:31b    | 4096  | long   | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 44      |  -         |  -      |  -        | incoherent  |             | 2026-08-18 01:14 |
| gemma4:31b    | 4096  | long   | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent  |             | 2026-08-18 01:14 |
| gemma4:31b    | 4096  | medium | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           |  -      | 5 832      |  -      |  -        |             |             | 2026-09-01 14:37 |
| gemma4:31b    | 4096  | medium | non-stream | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent  |             | 2026-09-01 14:37 |
| gemma4:31b    | 4096  | medium | stream     | GPU    | Ollama 0.32.6   |  -            | 25.4         | 48      | 5 908      |  -      |  -        |             |             | 2026-09-01 12:59 |
| gemma4:31b    | 4096  | medium | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent  |             | 2026-09-01 12:59 |
| gemma4:31b    | 4096  | short  | non-stream | GPU    | Ollama 0.32.6   | 76.1          | 8.3          | 128     | 7 646      |  -      |  -        |             |             | 2026-09-01 11:21 |
| gemma4:31b    | 4096  | short  | non-stream | GPU    | **loken 0.1.0** | **194.1**     | **23.9**     | **128** | **5 370**  | ** - ** | ** - **   | **+188.0%** |             | 2026-09-01 11:21 |
| gemma4:31b    | 4096  | short  | stream     | GPU    | Ollama 0.32.6   | 83.3          | 25.7         | 96      | 7 652      |  -      |  -        |             |             | 2026-09-01 09:42 |
| gemma4:31b    | 4096  | short  | stream     | GPU    | **loken 0.1.0** | **193.8**     | **24.4**     | **128** | **5 407**  | ** - ** | ** - **   |             |             | 2026-09-01 09:42 |
| gemma4:latest | 4096  | long   | non-stream | GPU    | Ollama 0.32.6   | 8 157.1       | 115.9        | 128     | 3 841      | 380     | 2.968     |             |             | 2026-08-18 01:14 |
| gemma4:latest | 4096  | long   | stream     | GPU    | Ollama 0.32.6   | 8 168.3       | 115.3        | 128     | 3 973      | 396     | 3.096     |             |             | 2026-08-18 01:14 |
| gemma4:latest | 4096  | long   | stream     | GPU    | **loken 0.1.0** | **17 599.2**  | **128.9**    | **128** | **1 064**  | **194** | **1.515** | **+11.8%**  | **+104.4%** | 2026-08-18 01:14 |
| gemma4:latest | 4096  | medium | non-stream | GPU    | Ollama 0.32.6   | 1 328.2       | 49.5         | 128     | 4 096      |  -      |  -        |             |             | 2026-09-01 14:39 |
| gemma4:latest | 4096  | medium | non-stream | GPU    | **loken 0.1.0** | **1 251.9**   | **119.3**    | **128** | **1 074**  | ** - ** | ** - **   | **+141.0%** |             | 2026-09-01 14:39 |
| gemma4:latest | 4096  | medium | stream     | GPU    | Ollama 0.32.6   | 1 332.9       | 90.2         | 128     | 3 965      |  -      |  -        |             |             | 2026-09-01 13:01 |
| gemma4:latest | 4096  | medium | stream     | GPU    | **loken 0.1.0** | **876.9**     | **127.1**    | **128** | **1 113**  | ** - ** | ** - **   | **+40.9%**  |             | 2026-09-01 13:01 |
| gemma4:latest | 4096  | short  | non-stream | GPU    | Ollama 0.32.6   | 487.4         | 50.1         | 128     | 3 965      |  -      |  -        |             |             | 2026-09-01 11:22 |
| gemma4:latest | 4096  | short  | non-stream | GPU    | **loken 0.1.0** | **439.9**     | **121.8**    | **128** | **1 051**  | ** - ** | ** - **   | **+142.9%** |             | 2026-09-01 11:22 |
| gemma4:latest | 4096  | short  | stream     | CPU    | Ollama 0.32.6   | 82.9          | 7.0          | 128     | 21 777     | 765     | 5.980     |             |             | 2026-08-18 01:14 |
| gemma4:latest | 4096  | short  | stream     | CPU    | **loken 0.1.0** | **79.9**      | **6.7**      | **128** | **19 608** | **652** | **5.090** | -4.6%       | **+17.5%**  | 2026-08-18 01:14 |
| gemma4:latest | 4096  | short  | stream     | GPU    | Ollama 0.32.6   | 488.7         | 91.1         | 128     | 3 981      |  -      |  -        |             |             | 2026-09-01 09:44 |
| gemma4:latest | 4096  | short  | stream     | GPU    | **loken 0.1.0** | **544.6**     | **128.4**    | **128** | **1 081**  | ** - ** | ** - **   | **+40.9%**  |             | 2026-09-01 09:44 |
| gemma4:latest | 16384 | long   | stream     | GPU    | Ollama 0.32.6   | 8 418.1       | 119.0        | 128     | 3 811      | 349     | 2.726     |             |             | 2026-08-18 01:14 |
| gemma4:latest | 16384 | long   | stream     | GPU    | **loken 0.1.0** | **14 816.7**  | **124.8**    | **128** | **1 124**  | **194** | **1.515** | **+4.9%**   | **+79.9%**  | 2026-08-18 01:14 |
| gemma4:latest | 16384 | short  | stream     | GPU    | Ollama 0.32.6   | 976.8         | 118.7        | 128     | 3 806      | 345     | 2.697     |             |             | 2026-08-18 01:14 |
| gemma4:latest | 16384 | short  | stream     | GPU    | **loken 0.1.0** | **1 600.3**   | **131.0**    | **128** | **1 006**  | **180** | **1.410** | **+10.4%**  | **+91.3%**  | 2026-08-18 01:14 |
| gemma4:latest | 32768 | long   | stream     | GPU    | Ollama 0.32.6   | 8 410.6       | 119.0        | 128     | 3 785      | 341     | 2.667     |             |             | 2026-08-18 01:14 |
| gemma4:latest | 32768 | long   | stream     | GPU    | **loken 0.1.0** | **11 025.6**  | **126.1**    | **128** | **1 095**  | **199** | **1.552** | **+6.0%**   | **+71.9%**  | 2026-08-18 01:14 |

### gptoss

![gptoss](img/family-gptoss.svg)

| Model       | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms  | J/req   | J/token   | Δ decode    | Δ energy    | Date             |
|-------------|------|--------|------------|--------|-----------------|---------------|--------------|---------|---------|---------|-----------|-------------|-------------|------------------|
| gpt-oss:20b | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 15 571.4      | 131.7        | 128     | 3 620   | 359     | 2.802     |             |             | 2026-08-18 01:14 |
| gpt-oss:20b | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **555 945.1** | **195.0**    | **128** | **919** | **172** | **1.344** | **+48.1%**  | **+108.5%** | 2026-08-18 01:14 |
| gpt-oss:20b | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 15 731.1      | 130.4        | 128     | 3 627   | 369     | 2.880     |             |             | 2026-08-18 01:14 |
| gpt-oss:20b | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **173 878.9** | **189.7**    | **128** | **929** | **172** | **1.343** | **+45.5%**  | **+114.4%** | 2026-08-18 01:14 |
| gpt-oss:20b | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 1 250.0       | 52.3         | 128     | 4 408   |  -      |  -        |             |             | 2026-09-01 14:51 |
| gpt-oss:20b | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **58 988.3**  | **172.1**    | **128** | **822** | ** - ** | ** - **   | **+229.1%** |             | 2026-09-01 14:51 |
| gpt-oss:20b | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 1 238.5       | 93.5         | 128     | 4 434   |  -      |  -        |             |             | 2026-09-01 13:13 |
| gpt-oss:20b | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **43 843.2**  | **212.4**    | **128** | **831** | ** - ** | ** - **   | **+127.3%** |             | 2026-09-01 13:13 |
| gpt-oss:20b | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 456.7         | 52.7         | 128     | 4 385   |  -      |  -        |             |             | 2026-09-01 11:34 |
| gpt-oss:20b | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **39 356.6**  | **175.1**    | **128** | **809** | ** - ** | ** - **   | **+232.2%** |             | 2026-09-01 11:34 |
| gpt-oss:20b | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 461.9         | 94.1         | 128     | 4 472   |  -      |  -        |             |             | 2026-09-01 09:56 |
| gpt-oss:20b | 4096 | short  | stream     | GPU    | vLLM 0.22.0     |  -            | 146.9        | 128     | 887     | 189     | 1.476     |             |             | 2026-08-10 07:13 |
| gpt-oss:20b | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **17 015.8**  | **211.9**    | **128** | **826** | ** - ** | ** - **   | **+44.2%**  |             | 2026-09-01 09:56 |

### granite

![granite](img/family-granite.svg)

| Model               | Ctx   | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode   | Δ energy    | Date             |
|---------------------|-------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|------------|-------------|------------------|
| granite3.1-dense:2b | 4096  | long   | non-stream | GPU    | Ollama 0.32.6   | 52 637.4      | 288.2        | 128     | 1 828      | 193     | 1.505     |            |             | 2026-08-18 01:14 |
| granite3.1-dense:2b | 4096  | long   | non-stream | GPU    | **loken 0.1.0** | **23 014.2**  | **271.1**    | **128** | **533**    | **101** | **0.788** | -5.9%      | **+91.0%**  | 2026-08-18 01:14 |
| granite3.1-dense:2b | 4096  | long   | stream     | GPU    | Ollama 0.32.6   | 52 970.2      | 287.4        | 128     | 1 924      | 203     | 1.587     |            |             | 2026-08-18 01:14 |
| granite3.1-dense:2b | 4096  | long   | stream     | GPU    | **loken 0.1.0** | **20 648.2**  | **268.3**    | **128** | **533**    | **113** | **0.885** | -6.7%      | **+79.4%**  | 2026-08-18 01:14 |
| granite3.1-dense:2b | 4096  | medium | non-stream | GPU    | Ollama 0.32.6   | 7 339.3       | 142.8        | 128     | 1 891      |  -      |  -        |            |             | 2026-09-01 14:53 |
| granite3.1-dense:2b | 4096  | medium | non-stream | GPU    | **loken 0.1.0** | **1 718.0**   | **243.1**    | **128** | **529**    | ** - ** | ** - **   | **+70.3%** |             | 2026-09-01 14:53 |
| granite3.1-dense:2b | 4096  | medium | stream     | GPU    | Ollama 0.32.6   | 7 354.5       | 223.8        | 128     | 1 918      |  -      |  -        |            |             | 2026-09-01 13:14 |
| granite3.1-dense:2b | 4096  | medium | stream     | GPU    | **loken 0.1.0** | **1 868.8**   | **277.1**    | **128** | **519**    | ** - ** | ** - **   | **+23.8%** |             | 2026-09-01 13:14 |
| granite3.1-dense:2b | 4096  | short  | non-stream | GPU    | Ollama 0.32.6   | 2 563.6       | 143.2        | 128     | 1 891      |  -      |  -        |            |             | 2026-09-01 11:36 |
| granite3.1-dense:2b | 4096  | short  | non-stream | GPU    | **loken 0.1.0** | **1 108.9**   | **248.8**    | **128** | **522**    | ** - ** | ** - **   | **+73.7%** |             | 2026-09-01 11:36 |
| granite3.1-dense:2b | 4096  | short  | stream     | CPU    | Ollama 0.32.6   | 606.7         | 13.3         | 128     | 11 258     | 418     | 3.263     |            |             | 2026-08-18 01:14 |
| granite3.1-dense:2b | 4096  | short  | stream     | CPU    | **loken 0.1.0** | **149.2**     | **12.2**     | **128** | **10 850** | **501** | **3.917** | -8.3%      | -16.7%      | 2026-08-18 01:14 |
| granite3.1-dense:2b | 4096  | short  | stream     | GPU    | Ollama 0.32.6   | 2 541.6       | 223.8        | 128     | 1 886      |  -      |  -        |            |             | 2026-09-01 09:58 |
| granite3.1-dense:2b | 4096  | short  | stream     | GPU    | **loken 0.1.0** | **535.2**     | **278.5**    | **128** | **522**    | ** - ** | ** - **   | **+24.5%** |             | 2026-09-01 09:58 |
| granite3.1-dense:2b | 16384 | long   | stream     | GPU    | Ollama 0.32.6   | 60 165.8      | 288.7        | 128     | 1 922      | 195     | 1.525     |            |             | 2026-08-18 01:14 |
| granite3.1-dense:2b | 16384 | long   | stream     | GPU    | **loken 0.1.0** | **20 130.6**  | **267.7**    | **128** | **530**    | **104** | **0.812** | -7.3%      | **+87.8%**  | 2026-08-18 01:14 |
| granite3.1-dense:2b | 32768 | long   | stream     | GPU    | Ollama 0.32.6   | 59 867.9      | 288.0        | 128     | 1 934      | 199     | 1.555     |            |             | 2026-08-18 01:14 |
| granite3.1-dense:2b | 32768 | long   | stream     | GPU    | **loken 0.1.0** | **5 549.3**   | **266.8**    | **128** | **555**    | **98**  | **0.763** | -7.4%      | **+103.8%** | 2026-08-18 01:14 |

### granitemoe

![granitemoe](img/family-granitemoe.svg)

| Model           | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token   | Δ decode   | Δ energy    | Date             |
|-----------------|------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|-----------|------------|-------------|------------------|
| granite3-moe:1b | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 54 756.8      | 317.4        | 128     | 2 361     | 178     | 1.390     |            |             | 2026-08-18 01:14 |
| granite3-moe:1b | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **519.4**     | **368.4**    | **128** | **996**   | **82**  | **0.643** | **+16.1%** | **+116.3%** | 2026-08-18 01:14 |
| granite3-moe:1b | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 55 811.8      | 309.5        | 128     | 2 502     | 190     | 1.485     |            |             | 2026-08-18 01:14 |
| granite3-moe:1b | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **524.0**     | **363.4**    | **128** | **992**   | **91**  | **0.713** | **+17.4%** | **+108.4%** | 2026-08-18 01:14 |
| granite3-moe:1b | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 9 247.2       | 250.0        | 128     | 524       |  -      |  -        |            |             | 2026-09-01 14:52 |
| granite3-moe:1b | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **2 478.3**   | **271.9**    | **128** | **483**   | ** - ** | ** - **   | **+8.8%**  |             | 2026-09-01 14:52 |
| granite3-moe:1b | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 9 088.8       | 331.3        | 128     | 512       |  -      |  -        |            |             | 2026-09-01 13:14 |
| granite3-moe:1b | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **1 272.6**   | **320.1**    | **128** | **505**   | ** - ** | ** - **   | -3.4%      |             | 2026-09-01 13:14 |
| granite3-moe:1b | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 3 202.4       | 258.0        | 128     | 511       |  -      |  -        |            |             | 2026-09-01 11:35 |
| granite3-moe:1b | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **824.4**     | **274.3**    | **128** | **474**   | ** - ** | ** - **   | **+6.3%**  |             | 2026-09-01 11:35 |
| granite3-moe:1b | 4096 | short  | stream     | CPU    | Ollama 0.32.6   | 2 464.5       | 68.7         | 128     | 3 165     | 152     | 1.184     |            |             | 2026-08-18 01:14 |
| granite3-moe:1b | 4096 | short  | stream     | CPU    | **loken 0.1.0** | **133.2**     | **55.7**     | **128** | **2 593** | **158** | **1.234** | -18.9%     | -4.1%       | 2026-08-18 01:14 |
| granite3-moe:1b | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 3 262.5       | 333.8        | 128     | 516       |  -      |  -        |            |             | 2026-09-01 09:57 |
| granite3-moe:1b | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **754.6**     | **323.5**    | **128** | **471**   | ** - ** | ** - **   | -3.1%      |             | 2026-09-01 09:57 |

### lfm2

![lfm2](img/family-lfm2.svg)

| Model                  | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s   | Decode tok/s | Tokens  | E2E ms  | J/req   | J/token   | Δ decode    | Δ energy    | Date             |
|------------------------|------|--------|------------|--------|-----------------|-----------------|--------------|---------|---------|---------|-----------|-------------|-------------|------------------|
| lfm2.5-thinking:latest | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 59 674.8        | 533.2        | 128     | 1 790   | 165     | 1.285     |             |             | 2026-08-18 01:14 |
| lfm2.5-thinking:latest | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **1 066 715.1** | **660.0**    | **128** | **355** | **55**  | **0.428** | **+23.8%**  | **+200.2%** | 2026-08-18 01:14 |
| lfm2.5-thinking:latest | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 59 376.1        | 529.3        | 128     | 1 851   | 165     | 1.290     |             |             | 2026-08-18 01:14 |
| lfm2.5-thinking:latest | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **214 095.4**   | **653.6**    | **128** | **350** | **50**  | **0.393** | **+23.5%**  | **+228.3%** | 2026-08-18 01:14 |
| lfm2.5-thinking:latest | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 7 819.7         | 213.2        | 128     | 1 688   |  -      |  -        |             |             | 2026-09-01 14:54 |
| lfm2.5-thinking:latest | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **217 436.5**   | **463.0**    | **128** | **404** | ** - ** | ** - **   | **+117.1%** |             | 2026-09-01 14:54 |
| lfm2.5-thinking:latest | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 7 839.8         | 396.0        | 128     | 1 713   |  -      |  -        |             |             | 2026-09-01 13:15 |
| lfm2.5-thinking:latest | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **86 908.3**    | **668.5**    | **128** | **414** | ** - ** | ** - **   | **+68.8%**  |             | 2026-09-01 13:15 |
| lfm2.5-thinking:latest | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 2 821.0         | 215.5        | 128     | 1 672   |  -      |  -        |             |             | 2026-09-01 11:37 |
| lfm2.5-thinking:latest | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **56 031.6**    | **442.8**    | **128** | **402** | ** - ** | ** - **   | **+105.4%** |             | 2026-09-01 11:37 |
| lfm2.5-thinking:latest | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 2 848.9         | 393.7        | 128     | 1 679   |  -      |  -        |             |             | 2026-09-01 09:59 |
| lfm2.5-thinking:latest | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **29 101.6**    | **636.3**    | **128** | **440** | ** - ** | ** - **   | **+61.6%**  |             | 2026-09-01 09:59 |

### lfm2moe

![lfm2moe](img/family-lfm2moe.svg)

| Model       | Ctx   | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode   | Δ energy    | Date             |
|-------------|-------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|------------|-------------|------------------|
| lfm2:latest | 4096  | long   | non-stream | GPU    | Ollama 0.32.6   | 22 420.3      | 299.0        | 128     | 2 556      | 244     | 1.904     |            |             | 2026-08-18 01:14 |
| lfm2:latest | 4096  | long   | non-stream | GPU    | **loken 0.1.0** | **845 164.9** | **308.1**    | **128** | **658**    | **108** | **0.841** | **+3.0%**  | **+126.3%** | 2026-08-18 01:14 |
| lfm2:latest | 4096  | long   | stream     | GPU    | Ollama 0.32.6   | 22 439.5      | 297.4        | 128     | 2 616      | 253     | 1.974     |            |             | 2026-08-18 01:14 |
| lfm2:latest | 4096  | long   | stream     | GPU    | **loken 0.1.0** | **346 867.9** | **304.7**    | **128** | **668**    | **117** | **0.912** | **+2.4%**  | **+116.5%** | 2026-08-18 01:14 |
| lfm2:latest | 4096  | medium | non-stream | GPU    | Ollama 0.32.6   | 2 840.1       | 136.0        | 128     | 3 136      |  -      |  -        |            |             | 2026-09-01 14:55 |
| lfm2:latest | 4096  | medium | non-stream | GPU    | **loken 0.1.0** | **20 936.2**  | **70.4**     | **4**   | **257**    | ** - ** | ** - **   |            |             | 2026-09-01 14:55 |
| lfm2:latest | 4096  | medium | stream     | GPU    | Ollama 0.32.6   | 2 922.4       | 233.5        | 128     | 3 125      |  -      |  -        |            |             | 2026-09-01 13:17 |
| lfm2:latest | 4096  | medium | stream     | GPU    | **loken 0.1.0** | **53 764.4**  | **238.1**    | **4**   | **226**    | ** - ** | ** - **   |            |             | 2026-09-01 13:17 |
| lfm2:latest | 4096  | short  | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent |             | 2026-09-01 11:39 |
| lfm2:latest | 4096  | short  | non-stream | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent |             | 2026-09-01 11:39 |
| lfm2:latest | 4096  | short  | stream     | CPU    | Ollama 0.32.6   | 140.4         | 14.4         | 128     | 13 330     | 477     | 3.724     |            |             | 2026-08-18 01:14 |
| lfm2:latest | 4096  | short  | stream     | CPU    | **loken 0.1.0** | **730.7**     | **14.0**     | **128** | **12 797** | **572** | **4.472** | -3.4%      | -16.7%      | 2026-08-18 01:14 |
| lfm2:latest | 4096  | short  | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent |             | 2026-09-01 10:00 |
| lfm2:latest | 4096  | short  | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent |             | 2026-09-01 10:00 |
| lfm2:latest | 16384 | long   | stream     | GPU    | Ollama 0.32.6   | 23 402.9      | 297.3        | 128     | 2 672      | 242     | 1.894     |            |             | 2026-08-18 01:14 |
| lfm2:latest | 16384 | long   | stream     | GPU    | **loken 0.1.0** | **355 793.0** | **307.0**    | **128** | **655**    | **111** | **0.869** | **+3.3%**  | **+117.9%** | 2026-08-18 01:14 |
| lfm2:latest | 32768 | long   | stream     | GPU    | Ollama 0.32.6   | 23 412.8      | 297.3        | 128     | 2 659      | 242     | 1.888     |            |             | 2026-08-18 01:14 |
| lfm2:latest | 32768 | long   | stream     | GPU    | **loken 0.1.0** | **352 400.1** | **307.6**    | **128** | **630**    | **124** | **0.973** | **+3.5%**  | **+94.1%**  | 2026-08-18 01:14 |

### llama

![llama](img/family-llama.svg)

| Model                | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req     | J/token    | Δ decode    | Δ energy    | Date             |
|----------------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|-----------|------------|-------------|-------------|------------------|
| deepseek-r1:70b      | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 264.2         | 1.5          | 128     | 91 787     | 7 258     | 56.701     |             |             | 2026-08-18 01:14 |
| deepseek-r1:70b      | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **197.5**     | **1.5**      | **128** | **98 181** | **7 508** | **58.657** | -2.0%       | -3.3%       | 2026-08-18 01:14 |
| deepseek-r1:70b      | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 264.8         | 1.5          | 128     | 91 793     | 7 272     | 56.809     |             |             | 2026-08-18 01:14 |
| deepseek-r1:70b      | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **188.8**     | **1.6**      | **128** | **90 756** | **6 958** | **54.360** | **+5.3%**   | **+4.5%**   | 2026-08-18 01:14 |
| deepseek-r1:70b      | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 51.2          | 1.5          | 128     | 88 444     |  -        |  -         |             |             | 2026-09-01 14:11 |
| deepseek-r1:70b      | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **14.6**      | **1.3**      | **128** | **99 211** | ** - **   | ** - **    | -10.6%      |             | 2026-09-01 14:11 |
| deepseek-r1:70b      | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 51.4          | 1.6          | 128     | 88 181     |  -        |  -         |             |             | 2026-09-01 12:32 |
| deepseek-r1:70b      | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **14.5**      | **1.5**      | **128** | **96 647** | ** - **   | ** - **    | -6.0%       |             | 2026-09-01 12:32 |
| deepseek-r1:70b      | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 20.4          | 1.5          | 128     | 88 064     |  -        |  -         |             |             | 2026-09-01 10:54 |
| deepseek-r1:70b      | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **8.7**       | **1.4**      | **128** | **95 279** | ** - **   | ** - **    | -7.2%       |             | 2026-09-01 10:54 |
| deepseek-r1:70b      | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 20.3          | 1.5          | 128     | 88 302     |  -        |  -         |             |             | 2026-09-01 09:15 |
| deepseek-r1:70b      | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **8.7**       | **1.5**      | **128** | **91 938** | ** - **   | ** - **    | -0.9%       |             | 2026-09-01 09:15 |
| deepseek-r1:70b      | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 10.5          | 0.8          | 128     | 167 239    | 11 682    | 91.268     |             |             | 2026-08-18 01:14 |
| deepseek-r1:70b      | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **14.1**      | **1.5**      | **128** | **92 449** | **7 193** | **56.196** | **+88.3%**  | **+62.4%**  | 2026-08-18 01:14 |
| deepseek-r1:70b-q3ks | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 1 138.7       | 6.9          | 128     | 23 859     | 3 168     | 24.746     |             |             | 2026-08-18 01:14 |
| deepseek-r1:70b-q3ks | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **272.1**     | **18.2**     | **128** | **7 983**  | **2 526** | **19.734** | **+166.0%** | **+25.4%**  | 2026-08-18 01:14 |
| deepseek-r1:70b-q3ks | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 1 136.4       | 6.8          | 128     | 24 021     | 3 180     | 24.845     |             |             | 2026-08-18 01:14 |
| deepseek-r1:70b-q3ks | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **269.9**     | **17.8**     | **128** | **8 005**  | **2 545** | **19.882** | **+162.5%** | **+25.0%**  | 2026-08-18 01:14 |
| deepseek-r1:70b-q3ks | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 227.6         | 5.9          | 128     | 23 402     |  -        |  -         |             |             | 2026-09-01 14:15 |
| deepseek-r1:70b-q3ks | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **179.7**     | **19.8**     | **128** | **6 480**  | ** - **   | ** - **    | **+238.5%** |             | 2026-09-01 14:15 |
| deepseek-r1:70b-q3ks | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 227.1         | 6.9          | 128     | 23 352     |  -        |  -         |             |             | 2026-09-01 12:37 |
| deepseek-r1:70b-q3ks | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **179.4**     | **21.1**     | **128** | **6 493**  | ** - **   | ** - **    | **+204.8%** |             | 2026-09-01 12:37 |
| deepseek-r1:70b-q3ks | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 85.7          | 5.9          | 128     | 23 310     |  -        |  -         |             |             | 2026-09-01 10:58 |
| deepseek-r1:70b-q3ks | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **101.8**     | **20.2**     | **128** | **6 367**  | ** - **   | ** - **    | **+243.5%** |             | 2026-09-01 10:58 |
| deepseek-r1:70b-q3ks | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 86.0          | 6.9          | 128     | 23 287     |  -        |  -         |             |             | 2026-09-01 09:20 |
| deepseek-r1:70b-q3ks | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **103.6**     | **21.2**     | **128** | **6 401**  | ** - **   | ** - **    | **+205.7%** |             | 2026-09-01 09:20 |
| devstral:24b         | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 7 753.5       |  -           | 1       | 2 347      | 174       |  -         | no answer   |             | 2026-08-18 01:14 |
| devstral:24b         | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 1 269.5       | 26.5         | 128     | 6 243      |  -        |  -         |             |             | 2026-09-01 14:29 |
| devstral:24b         | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **813.2**     | **50.1**     | **128** | **2 559**  | ** - **   | ** - **    | **+89.0%**  |             | 2026-09-01 14:29 |
| devstral:24b         | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 1 269.8       | 36.2         | 128     | 6 240      |  -        |  -         |             |             | 2026-09-01 12:51 |
| devstral:24b         | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **759.8**     | **52.1**     | **128** | **2 564**  | ** - **   | ** - **    | **+43.9%**  |             | 2026-09-01 12:51 |
| devstral:24b         | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 459.1         | 26.5         | 128     | 6 244      |  -        |  -         |             |             | 2026-09-01 11:13 |
| devstral:24b         | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **408.3**     | **49.3**     | **128** | **2 598**  | ** - **   | ** - **    | **+86.2%**  |             | 2026-09-01 11:13 |
| devstral:24b         | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 459.0         | 36.3         | 128     | 6 288      |  -        |  -         |             |             | 2026-09-01 09:34 |
| devstral:24b         | 4096   | short  | stream     | GPU    | vLLM 0.22.0     |  -            | 50.6         | 128     | 2 541      | 580       | 4.534      |             |             | 2026-08-10 07:40 |
| devstral:24b         | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **326.9**     | **51.0**     | **128** | **2 611**  | ** - **   | ** - **    | **+0.6%**   |             | 2026-09-01 09:34 |
| falcon3:latest       | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 54 289.6      | 270.2        | 128     | 2 167      | 193       | 1.506      |             |             | 2026-08-18 01:14 |
| falcon3:latest       | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 54 055.3      | 267.1        | 128     | 2 283      | 198       | 1.546      |             |             | 2026-08-18 01:14 |
| falcon3:latest       | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 6 179.3       | 121.1        | 128     | 2 087      |  -        |  -         |             |             | 2026-09-01 14:31 |
| falcon3:latest       | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **2 416.8**   | **342.3**    | **128** | **378**    | ** - **   | ** - **    | **+182.7%** |             | 2026-09-01 14:31 |
| falcon3:latest       | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 6 238.4       | 211.0        | 128     | 2 113      |  -        |  -         |             |             | 2026-09-01 12:53 |
| falcon3:latest       | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 618.0**   | **389.5**    | **128** | **394**    | ** - **   | ** - **    | **+84.6%**  |             | 2026-09-01 12:53 |
| falcon3:latest       | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 2 229.4       | 120.8        | 128     | 2 117      |  -        |  -         |             |             | 2026-09-01 11:14 |
| falcon3:latest       | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **915.0**     | **342.4**    | **128** | **378**    | ** - **   | ** - **    | **+183.6%** |             | 2026-09-01 11:14 |
| falcon3:latest       | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 1 035.9       | 14.8         | 21      | 3 106      | 151       | 7.206      |             |             | 2026-08-18 01:14 |
| falcon3:latest       | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **158.6**     | **14.1**     | **79**  | **5 967**  | **327**   | **4.118**  |             |             | 2026-08-18 01:14 |
| falcon3:latest       | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 2 232.3       | 211.2        | 128     | 2 121      |  -        |  -         |             |             | 2026-09-01 09:36 |
| falcon3:latest       | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **764.1**     | **391.9**    | **128** | **385**    | ** - **   | ** - **    | **+85.6%**  |             | 2026-09-01 09:36 |
| llama3.2:1b          | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 43 978.8      | 316.1        | 128     | 2 219      | 195       | 1.520      |             |             | 2026-08-18 01:14 |
| llama3.2:1b          | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **44 166.7**  | **469.3**    | **128** | **321**    | **53**    | **0.414**  | **+48.5%**  | **+267.4%** | 2026-08-18 01:14 |
| llama3.2:1b          | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 43 681.4      | 316.7        | 128     | 2 277      | 198       | 1.550      |             |             | 2026-08-18 01:14 |
| llama3.2:1b          | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **40 994.5**  | **463.1**    | **128** | **328**    | **54**    | **0.420**  | **+46.2%**  | **+268.9%** | 2026-08-18 01:14 |
| llama3.2:1b          | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 6 731.6       | 120.9        | 128     | 2 136      |  -        |  -         |             |             | 2026-09-01 14:56 |
| llama3.2:1b          | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **2 178.6**   | **383.0**    | **128** | **336**    | ** - **   | ** - **    | **+216.8%** |             | 2026-09-01 14:56 |
| llama3.2:1b          | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 6 612.9       | 237.5        | 128     | 2 166      |  -        |  -         |             |             | 2026-09-01 13:18 |
| llama3.2:1b          | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 980.5**   | **447.5**    | **128** | **346**    | ** - **   | ** - **    | **+88.4%**  |             | 2026-09-01 13:18 |
| llama3.2:1b          | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 2 382.3       | 116.2        | 128     | 2 187      |  -        |  -         |             |             | 2026-09-01 11:40 |
| llama3.2:1b          | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **808.5**     | **376.7**    | **128** | **344**    | ** - **   | ** - **    | **+224.1%** |             | 2026-09-01 11:40 |
| llama3.2:1b          | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 538.1         | 16.6         | 128     | 9 415      | 357       | 2.791      |             |             | 2026-08-18 01:14 |
| llama3.2:1b          | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **327.6**     | **15.9**     | **128** | **8 239**  | **414**   | **3.235**  | -3.8%       | -13.7%      | 2026-08-18 01:14 |
| llama3.2:1b          | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 2 400.0       | 232.2        | 128     | 2 157      |  -        |  -         |             |             | 2026-09-01 10:01 |
| llama3.2:1b          | 4096   | short  | stream     | GPU    | vLLM 0.22.0     |  -            | 292.6        | 128     | 446        | 88        | 0.685      |             |             | 2026-08-10 07:11 |
| llama3.2:1b          | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **737.8**     | **451.0**    | **128** | **344**    | ** - **   | ** - **    | **+54.1%**  |             | 2026-09-01 10:01 |
| magistral:latest     | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 7 722.8       |  -           | 1       | 2 377      | 180       |  -         | no answer   |             | 2026-08-18 01:14 |
| magistral:latest     | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 1 263.3       | 26.3         | 128     | 6 283      |  -        |  -         |             |             | 2026-09-01 14:58 |
| magistral:latest     | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **803.9**     | **49.6**     | **128** | **2 586**  | ** - **   | ** - **    | **+88.1%**  |             | 2026-09-01 14:58 |
| magistral:latest     | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 1 218.9       | 36.3         | 128     | 6 296      |  -        |  -         |             |             | 2026-09-01 13:20 |
| magistral:latest     | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 130.1**   | **53.3**     | **128** | **2 500**  | ** - **   | ** - **    | **+46.8%**  |             | 2026-09-01 13:20 |
| magistral:latest     | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 419.9         | 26.3         | 128     | 6 422      |  -        |  -         |             |             | 2026-09-01 11:42 |
| magistral:latest     | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **407.2**     | **49.6**     | **128** | **2 585**  | ** - **   | ** - **    | **+88.6%**  |             | 2026-09-01 11:42 |
| magistral:latest     | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 427.9         | 36.3         | 128     | 6 343      |  -        |  -         |             |             | 2026-09-01 10:03 |
| magistral:latest     | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **322.3**     | **51.3**     | **128** | **2 605**  | ** - **   | ** - **    | **+41.5%**  |             | 2026-09-01 10:03 |
| mistral-nemo:latest  | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 15 103.7      | 96.6         | 128     | 3 214      | 428       | 3.340      |             |             | 2026-08-18 01:14 |
| mistral-nemo:latest  | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **20 996.2**  | **107.2**    | **128** | **1 270**  | **296**   | **2.312**  | **+11.0%**  | **+44.5%**  | 2026-08-18 01:14 |
| mistral-nemo:latest  | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 15 153.6      | 95.9         | 128     | 3 306      | 428       | 3.345      |             |             | 2026-08-18 01:14 |
| mistral-nemo:latest  | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **20 034.2**  | **106.3**    | **128** | **1 300**  | **305**   | **2.379**  | **+10.8%**  | **+40.6%**  | 2026-08-18 01:14 |
| mistral-nemo:latest  | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 2 149.8       | 45.8         | 128     | 3 994      |  -        |  -         |             |             | 2026-09-01 15:00 |
| mistral-nemo:latest  | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **1 746.4**   | **95.5**     | **128** | **1 342**  | ** - **   | ** - **    | **+108.4%** |             | 2026-09-01 15:00 |
| mistral-nemo:latest  | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 2 186.8       | 67.9         | 128     | 4 009      |  -        |  -         |             |             | 2026-09-01 13:21 |
| mistral-nemo:latest  | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 591.6**   | **100.9**    | **128** | **1 359**  | ** - **   | ** - **    | **+48.7%**  |             | 2026-09-01 13:21 |
| mistral-nemo:latest  | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 792.3         | 45.9         | 128     | 3 998      |  -        |  -         |             |             | 2026-09-01 11:43 |
| mistral-nemo:latest  | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **313.7**     | **95.1**     | **128** | **1 349**  | ** - **   | ** - **    | **+107.3%** |             | 2026-09-01 11:43 |
| mistral-nemo:latest  | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 713.5         | 68.1         | 128     | 5 862      |  -        |  -         |             |             | 2026-09-01 10:05 |
| mistral-nemo:latest  | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **262.8**     | **102.2**    | **128** | **1 356**  | ** - **   | ** - **    | **+50.0%**  |             | 2026-09-01 10:05 |

### mistral3

![mistral3](img/family-mistral3.svg)

| Model                   | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token   | Δ decode    | Δ energy   | Date             |
|-------------------------|------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|-----------|-------------|------------|------------------|
| devstral-small-2:latest | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 3 459.6       | 19.2         | 128     | 9 543     | 1 060   | 8.278     |             |            | 2026-08-18 01:14 |
| devstral-small-2:latest | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **979.4**     | **57.3**     | **128** | **2 641** | **772** | **6.028** | **+198.1%** | **+37.3%** | 2026-08-18 01:14 |
| devstral-small-2:latest | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 3 461.8       | 19.0         | 128     | 9 674     | 1 071   | 8.367     |             |            | 2026-08-18 01:14 |
| devstral-small-2:latest | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **973.6**     | **56.8**     | **128** | **2 649** | **764** | **5.970** | **+198.7%** | **+40.2%** | 2026-08-18 01:14 |
| devstral-small-2:latest | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 1 122.9       | 25.4         | 128     | 6 592     |  -      |  -        |             |            | 2026-09-01 14:17 |
| devstral-small-2:latest | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **554.0**     | **53.9**     | **128** | **2 412** | ** - ** | ** - **   | **+112.4%** |            | 2026-09-01 14:17 |
| devstral-small-2:latest | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 1 140.5       | 35.2         | 128     | 6 593     |  -      |  -        |             |            | 2026-09-01 12:39 |
| devstral-small-2:latest | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **622.8**     | **59.3**     | **128** | **2 411** | ** - ** | ** - **   | **+68.2%**  |            | 2026-09-01 12:39 |
| devstral-small-2:latest | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 403.0         | 25.6         | 128     | 6 571     |  -      |  -        |             |            | 2026-09-01 11:00 |
| devstral-small-2:latest | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **301.3**     | **54.8**     | **128** | **2 368** | ** - ** | ** - **   | **+113.9%** |            | 2026-09-01 11:00 |
| devstral-small-2:latest | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 410.3         | 35.4         | 128     | 6 556     |  -      |  -        |             |            | 2026-09-01 09:22 |
| devstral-small-2:latest | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **281.2**     | **59.5**     | **128** | **2 399** | ** - ** | ** - **   | **+67.8%**  |            | 2026-09-01 09:22 |
| mistral-small3.2:latest | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 3 444.2       | 19.4         | 128     | 9 542     | 1 058   | 8.265     |             |            | 2026-08-18 01:14 |
| mistral-small3.2:latest | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **973.4**     | **57.8**     | **128** | **2 574** | **752** | **5.871** | **+198.2%** | **+40.8%** | 2026-08-18 01:14 |
| mistral-small3.2:latest | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 3 504.3       | 19.8         | 128     | 9 276     | 1 023   | 7.995     |             |            | 2026-08-18 01:14 |
| mistral-small3.2:latest | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **972.7**     | **57.1**     | **128** | **2 583** | **753** | **5.881** | **+188.9%** | **+35.9%** | 2026-08-18 01:14 |
| mistral-small3.2:latest | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 1 193.8       | 26.2         | 128     | 6 418     |  -      |  -        |             |            | 2026-09-01 15:02 |
| mistral-small3.2:latest | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **629.0**     | **53.1**     | **128** | **2 467** | ** - ** | ** - **   | **+103.0%** |            | 2026-09-01 15:02 |
| mistral-small3.2:latest | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 1 195.7       | 36.3         | 128     | 6 424     |  -      |  -        |             |            | 2026-09-01 13:23 |
| mistral-small3.2:latest | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **618.8**     | **59.2**     | **128** | **2 470** | ** - ** | ** - **   | **+63.1%**  |            | 2026-09-01 13:23 |
| mistral-small3.2:latest | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 459.8         | 26.0         | 128     | 6 429     |  -      |  -        |             |            | 2026-09-01 11:45 |
| mistral-small3.2:latest | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **249.2**     | **54.6**     | **128** | **2 372** | ** - ** | ** - **   | **+110.5%** |            | 2026-09-01 11:45 |
| mistral-small3.2:latest | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 434.8         | 36.3         | 128     | 6 459     |  -      |  -        |             |            | 2026-09-01 10:07 |
| mistral-small3.2:latest | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **300.5**     | **59.4**     | **128** | **2 388** | ** - ** | ** - **   | **+63.9%**  |            | 2026-09-01 10:07 |

### nemotron_h_moe

![nemotron_h_moe](img/family-nemotron_h_moe.svg)

| Model                  | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token   | Δ decode   | Δ energy   | Date             |
|------------------------|------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|-----------|------------|------------|------------------|
| nemotron-3-nano:latest | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 3 236.5       | 134.7        | 128     | 4 813     | 452     | 3.530     |            |            | 2026-08-18 01:14 |
| nemotron-3-nano:latest | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **424 483.6** | **118.2**    | **128** | **1 604** | **244** | **1.908** | -12.2%     | **+85.1%** | 2026-08-18 01:14 |
| nemotron-3-nano:latest | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 3 239.8       | 133.6        | 128     | 4 810     | 444     | 3.471     |            |            | 2026-08-18 01:14 |
| nemotron-3-nano:latest | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **218 674.2** | **119.1**    | **128** | **1 532** | **238** | **1.858** | -10.9%     | **+86.8%** | 2026-08-18 01:14 |
| nemotron-3-nano:latest | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 606.4         | 73.0         | 128     | 4 689     |  -      |  -        |            |            | 2026-09-01 15:05 |
| nemotron-3-nano:latest | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **128 303.2** | **120.4**    | **128** | **1 214** | ** - ** | ** - **   | **+65.0%** |            | 2026-09-01 15:05 |
| nemotron-3-nano:latest | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 604.8         | 136.0        | 128     | 4 713     |  -      |  -        |            |            | 2026-09-01 13:26 |
| nemotron-3-nano:latest | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **18 725.6**  | **144.4**    | **128** | **1 199** | ** - ** | ** - **   | **+6.2%**  |            | 2026-09-01 13:26 |
| nemotron-3-nano:latest | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 210.9         | 70.3         | 128     | 4 742     |  -      |  -        |            |            | 2026-09-01 11:48 |
| nemotron-3-nano:latest | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **17 399.8**  | **120.1**    | **128** | **1 211** | ** - ** | ** - **   | **+70.7%** |            | 2026-09-01 11:48 |
| nemotron-3-nano:latest | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 207.2         | 133.2        | 128     | 4 724     |  -      |  -        |            |            | 2026-09-01 10:10 |
| nemotron-3-nano:latest | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **28 150.1**  | **145.0**    | **128** | **1 204** | ** - ** | ** - **   | **+8.9%**  |            | 2026-09-01 10:10 |

### olmo2

![olmo2](img/family-olmo2.svg)

| Model    | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode   | Δ energy   | Date             |
|----------|------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|------------|------------|------------------|
| olmo2:7b | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 20 028.7      | 136.4        | 128     | 2 840      | 354     | 2.767     |            |            | 2026-08-18 01:14 |
| olmo2:7b | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **34 901.7**  | **116.7**    | **128** | **1 169**  | **280** | **2.190** | -14.4%     | **+26.3%** | 2026-08-18 01:14 |
| olmo2:7b | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 19 953.2      | 134.6        | 128     | 2 898      | 356     | 2.782     |            |            | 2026-08-18 01:14 |
| olmo2:7b | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **28 548.3**  | **115.6**    | **128** | **1 168**  | **271** | **2.118** | -14.1%     | **+31.3%** | 2026-08-18 01:14 |
| olmo2:7b | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 2 957.5       | 87.5         | 128     | 1 464      |  -      |  -        |            |            | 2026-09-01 15:06 |
| olmo2:7b | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **3 065.5**   | **125.4**    | **128** | **1 025**  | ** - ** | ** - **   | **+43.4%** |            | 2026-09-01 15:06 |
| olmo2:7b | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 2 949.3       | 100.7        | 128     | 1 442      |  -      |  -        |            |            | 2026-09-01 13:27 |
| olmo2:7b | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **1 878.7**   | **134.2**    | **128** | **1 038**  | ** - ** | ** - **   | **+33.3%** |            | 2026-09-01 13:27 |
| olmo2:7b | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 1 056.8       | 89.2         | 128     | 1 436      |  -      |  -        |            |            | 2026-09-01 11:49 |
| olmo2:7b | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **716.7**     | **126.3**    | **128** | **1 017**  | ** - ** | ** - **   | **+41.6%** |            | 2026-09-01 11:49 |
| olmo2:7b | 4096 | short  | stream     | CPU    | Ollama 0.32.6   | 202.1         | 5.2          | 128     | 27 396     | 978     | 7.638     |            |            | 2026-08-18 01:14 |
| olmo2:7b | 4096 | short  | stream     | CPU    | **loken 0.1.0** | **60.4**      | **6.7**      | **91**  | **15 762** | **508** | **5.606** |            |            | 2026-08-18 01:14 |
| olmo2:7b | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 1 026.8       | 100.6        | 128     | 1 450      |  -      |  -        |            |            | 2026-09-01 10:11 |
| olmo2:7b | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **704.3**     | **133.9**    | **128** | **1 032**  | ** - ** | ** - **   | **+33.2%** |            | 2026-09-01 10:11 |

### olmoe

![olmoe](img/family-olmoe.svg)

| Model        | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token   | Δ decode   | Δ energy | Date             |
|--------------|------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|-----------|------------|----------|------------------|
| olmoe:latest | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 82 498.5      | 483.8        | 128     | 2 013     | 182     | 1.424     |            |          | 2026-08-18 01:14 |
| olmoe:latest | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **236.6**     | **441.1**    | **128** | **3 105** | **227** | **1.777** | -8.8%      | -19.9%   | 2026-08-18 01:14 |
| olmoe:latest | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 82 280.0      | 479.1        | 128     | 2 012     | 178     | 1.390     |            |          | 2026-08-18 01:14 |
| olmoe:latest | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **230.8**     | **417.1**    | **128** | **3 079** | **235** | **1.837** | -12.9%     | -24.3%   | 2026-08-18 01:14 |
| olmoe:latest | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 10 699.6      | 287.2        | 128     | 454       |  -      |  -        |            |          | 2026-09-01 15:07 |
| olmoe:latest | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **1 466.5**   | **330.0**    | **128** | **415**   | ** - ** | ** - **   | **+14.9%** |          | 2026-09-01 15:07 |
| olmoe:latest | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 10 655.9      | 390.2        | 128     | 443       |  -      |  -        |            |          | 2026-09-01 13:29 |
| olmoe:latest | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **1 148.9**   | **428.9**    | **128** | **410**   | ** - ** | ** - **   | **+9.9%**  |          | 2026-09-01 13:29 |
| olmoe:latest | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 3 651.5       | 296.4        | 128     | 436       |  -      |  -        |            |          | 2026-09-01 11:50 |
| olmoe:latest | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **590.4**     | **312.1**    | **128** | **458**   | ** - ** | ** - **   | **+5.3%**  |          | 2026-09-01 11:50 |
| olmoe:latest | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 3 628.7       | 389.0        | 128     | 422       |  -      |  -        |            |          | 2026-09-01 10:12 |
| olmoe:latest | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **361.6**     | **440.7**    | **128** | **392**   | ** - ** | ** - **   | **+13.3%** |          | 2026-09-01 10:12 |

### phi2

![phi2](img/family-phi2.svg)

| Model            | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms  | J/req   | J/token | Δ decode   | Δ energy | Date             |
|------------------|------|--------|------------|--------|-----------------|---------------|--------------|---------|---------|---------|---------|------------|----------|------------------|
| moondream:latest | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 35 453.0      |  -           | 1       | 1 614   | 119     |  -      | no answer  |          | 2026-08-18 01:14 |
| moondream:latest | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 10 333.5      | 293.0        | 128     | 437     |  -      |  -      |            |          | 2026-09-01 15:02 |
| moondream:latest | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **2 747.6**   | **467.5**    | **128** | **280** | ** - ** | ** - ** | **+59.5%** |          | 2026-09-01 15:02 |
| moondream:latest | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 10 354.2      | 370.2        | 128     | 450     |  -      |  -      |            |          | 2026-09-01 13:24 |
| moondream:latest | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **3 543.0**   | **591.5**    | **128** | **250** | ** - ** | ** - ** | **+59.8%** |          | 2026-09-01 13:24 |
| moondream:latest | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 3 658.1       | 289.7        | 128     | 443     |  -      |  -      |            |          | 2026-09-01 11:46 |
| moondream:latest | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **1 399.5**   | **508.6**    | **128** | **254** | ** - ** | ** - ** | **+75.5%** |          | 2026-09-01 11:46 |
| moondream:latest | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 3 604.0       | 370.9        | 128     | 447     |  -      |  -      |            |          | 2026-09-01 10:08 |
| moondream:latest | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **1 647.6**   | **598.1**    | **128** | **245** | ** - ** | ** - ** | **+61.2%** |          | 2026-09-01 10:08 |

### qwen2

![qwen2](img/family-qwen2.svg)

| Model            | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req     | J/token    | Δ decode     | Δ energy    | Date             |
|------------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|-----------|------------|--------------|-------------|------------------|
| deepcoder:14b    | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 11 443.0      | 75.5         | 128     | 3 630      | 563       | 4.395      |              |             | 2026-08-18 01:14 |
| deepcoder:14b    | 4096   | long   | non-stream | GPU    | vLLM 0.22.0     |  -            | 81.9         | 128     | 1 566      | 335       | 2.621      |              |             | 2026-08-18 01:14 |
| deepcoder:14b    | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **13 046.1**  | **77.9**     | **128** | **1 716**  | **436**   | **3.408**  | -4.9%        | -23.1%      | 2026-08-18 01:14 |
| deepcoder:14b    | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 8 371.4       | 53.3         | 128     | 4 605      |  -        |  -         |              |             | 2026-09-01 15:31 |
| deepcoder:14b    | 4096   | long   | stream     | GPU    | vLLM 0.22.0     |  -            | 85.0         | 128     | 1 569      |  -        |  -         |              |             | 2026-09-01 15:31 |
| deepcoder:14b    | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **3 214.1**   | **75.7**     | **128** | **1 813**  | ** - **   | ** - **    | -10.9%       |             | 2026-09-01 15:31 |
| deepcoder:14b    | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 1 707.8       | 37.7         | 128     | 4 596      |  -        |  -         |              |             | 2026-09-01 13:55 |
| deepcoder:14b    | 4096   | medium | non-stream | GPU    | vLLM 0.22.0     |  -            | 82.4         | 128     | 1 555      |  -        |  -         |              |             | 2026-09-01 13:55 |
| deepcoder:14b    | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **1 309.1**   | **77.3**     | **128** | **1 658**  | ** - **   | ** - **    | -6.2%        |             | 2026-09-01 13:55 |
| deepcoder:14b    | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 1 709.3       | 52.9         | 128     | 4 636      |  -        |  -         |              |             | 2026-09-01 12:17 |
| deepcoder:14b    | 4096   | medium | stream     | GPU    | vLLM 0.22.0     |  -            | 85.0         | 128     | 1 550      |  -        |  -         |              |             | 2026-09-01 12:17 |
| deepcoder:14b    | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 239.0**   | **80.4**     | **128** | **1 668**  | ** - **   | ** - **    | -5.5%        |             | 2026-09-01 12:17 |
| deepcoder:14b    | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 620.7         | 37.6         | 128     | 4 607      |  -        |  -         |              |             | 2026-09-01 10:39 |
| deepcoder:14b    | 4096   | short  | non-stream | GPU    | vLLM 0.22.0     |  -            | 82.7         | 128     | 1 549      |  -        |  -         |              |             | 2026-09-01 10:39 |
| deepcoder:14b    | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **465.6**     | **76.6**     | **128** | **1 671**  | ** - **   | ** - **    | -7.4%        |             | 2026-09-01 10:39 |
| deepcoder:14b    | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 44.7          | 2.6          | 128     | 52 337     | 1 718     | 13.418     |              |             | 2026-08-18 01:14 |
| deepcoder:14b    | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **23.4**      | **2.6**      | **128** | **50 305** | **1 636** | **12.782** | **+1.2%**    | **+5.0%**   | 2026-08-18 01:14 |
| deepcoder:14b    | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 642.2         | 52.6         | 128     | 4 639      |  -        |  -         |              |             | 2026-09-01 08:59 |
| deepcoder:14b    | 4096   | short  | stream     | GPU    | vLLM 0.22.0     |  -            | 85.1         | 128     | 1 531      |  -        |  -         |              |             | 2026-09-01 08:59 |
| deepcoder:14b    | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **427.1**     | **78.8**     | **128** | **1 694**  | ** - **   | ** - **    | -7.4%        |             | 2026-09-01 08:59 |
| deepcoder:14b    | 16384  | long   | stream     | GPU    | Ollama 0.32.6   | 11 703.7      | 75.4         | 128     | 3 681      | 557       | 4.351      |              |             | 2026-08-18 01:14 |
| deepcoder:14b    | 16384  | long   | stream     | GPU    | **loken 0.1.0** | **11 401.0**  | **77.3**     | **128** | **1 721**  | **441**   | **3.448**  | **+2.5%**    | **+26.2%**  | 2026-08-18 01:14 |
| deepcoder:14b    | 32768  | long   | stream     | GPU    | Ollama 0.32.6   | 11 776.2      | 75.4         | 128     | 3 704      | 547       | 4.276      |              |             | 2026-08-18 01:14 |
| deepcoder:14b    | 32768  | long   | stream     | GPU    | **loken 0.1.0** | **8 435.7**   | **77.3**     | **128** | **1 731**  | **430**   | **3.359**  | **+2.5%**    | **+27.3%**  | 2026-08-18 01:14 |
| deepcoder:14b    | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 57.0          | 4.4          | 128     | 33 043     | 1 934     | 15.110     |              |             | 2026-08-18 01:14 |
| deepcoder:14b    | 131072 | short  | stream     | GPU    | vLLM 0.22.0     |  -            | 85.0         | 128     | 1 550      | 341       | 2.663      |              |             | 2026-08-18 01:14 |
| deepcoder:14b    | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **1 050.6**   | **80.2**     | **128** | **1 638**  | **404**   | **3.159**  | -5.6%        | -15.7%      | 2026-08-18 01:14 |
| deepcoder:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 1 106.8       | 74.4         | 128     | 3 644      | 544       | 4.250      |              |             | 2026-08-18 01:14 |
| deepcoder:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **1 206.0**   | **86.0**     | **128** | **1 532**  | **394**   | **3.081**  | **+15.7%**   | **+38.0%**  | 2026-08-18 01:14 |
| deepseek-r1:32b  | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 4 219.3       | 25.9         | 128     | 8 211      | 1 307     | 10.213     |              |             | 2026-08-18 01:14 |
| deepseek-r1:32b  | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **615.7**     | **39.7**     | **128** | **3 743**  | **1 062** | **8.299**  | **+53.4%**   | **+23.1%**  | 2026-08-18 01:14 |
| deepseek-r1:32b  | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 4 215.1       | 25.9         | 128     | 8 094      |  -        |  -         |              |             | 2026-09-01 15:33 |
| deepseek-r1:32b  | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **559.4**     | **39.7**     | **128** | **3 796**  | ** - **   | ** - **    | **+53.4%**   |             | 2026-09-01 15:33 |
| deepseek-r1:32b  | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 864.6         | 19.7         | 128     | 8 089      |  -        |  -         |              |             | 2026-09-01 13:57 |
| deepseek-r1:32b  | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **352.2**     | **37.5**     | **128** | **3 452**  | ** - **   | ** - **    | **+90.4%**   |             | 2026-09-01 13:57 |
| deepseek-r1:32b  | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 893.8         | 25.9         | 128     | 8 158      |  -        |  -         |              |             | 2026-09-01 12:19 |
| deepseek-r1:32b  | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **397.3**     | **40.8**     | **128** | **3 473**  | ** - **   | ** - **    | **+57.7%**   |             | 2026-09-01 12:19 |
| deepseek-r1:32b  | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 317.6         | 19.6         | 128     | 8 267      |  -        |  -         |              |             | 2026-09-01 10:41 |
| deepseek-r1:32b  | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **230.7**     | **38.4**     | **128** | **3 364**  | ** - **   | ** - **    | **+96.2%**   |             | 2026-09-01 10:41 |
| deepseek-r1:32b  | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 335.8         | 25.8         | 128     | 8 179      |  -        |  -         |              |             | 2026-09-01 09:01 |
| deepseek-r1:32b  | 4096   | short  | stream     | GPU    | vLLM 0.22.0     |  -            | 36.9         | 128     | 3 490      | 811       | 6.334      |              |             | 2026-08-10 07:42 |
| deepseek-r1:32b  | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **234.4**     | **41.0**     | **128** | **3 432**  | ** - **   | ** - **    | **+11.2%**   |             | 2026-09-01 09:01 |
| deepseek-r1:32b  | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 31.3          | 2.4          | 128     | 57 917     | 4 184     | 32.686     |              |             | 2026-08-18 01:14 |
| deepseek-r1:32b  | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **368.7**     | **40.5**     | **128** | **3 389**  | **1 007** | **7.863**  | **+1566.4%** | **+315.7%** | 2026-08-18 01:14 |
| qwen2.5:0.5b     | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 43 488.0      | 376.1        | 128     | 2 016      | 157       | 1.225      |              |             | 2026-08-18 01:14 |
| qwen2.5:0.5b     | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **38 142.7**  | **375.1**    | **128** | **376**    | **52**    | **0.404**  | -0.3%        | **+203.3%** | 2026-08-18 01:14 |
| qwen2.5:0.5b     | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 43 653.6      | 373.1        | 128     | 2 001      | 154       | 1.204      |              |             | 2026-08-18 01:14 |
| qwen2.5:0.5b     | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **34 434.4**  | **363.4**    | **128** | **393**    | **53**    | **0.415**  | -2.6%        | **+190.5%** | 2026-08-18 01:14 |
| qwen2.5:0.5b     | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 8 422.9       | 151.9        | 128     | 1 945      |  -        |  -         |              |             | 2026-09-01 15:08 |
| qwen2.5:0.5b     | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **3 814.6**   | **321.5**    | **128** | **408**    | ** - **   | ** - **    | **+111.7%**  |             | 2026-09-01 15:08 |
| qwen2.5:0.5b     | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 8 215.8       | 317.3        | 128     | 1 927      |  -        |  -         |              |             | 2026-09-01 13:29 |
| qwen2.5:0.5b     | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **2 008.6**   | **365.2**    | **128** | **414**    | ** - **   | ** - **    | **+15.1%**   |             | 2026-09-01 13:29 |
| qwen2.5:0.5b     | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 2 854.2       | 153.4        | 128     | 1 951      |  -        |  -         |              |             | 2026-09-01 11:51 |
| qwen2.5:0.5b     | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **1 298.3**   | **332.6**    | **128** | **390**    | ** - **   | ** - **    | **+116.7%**  |             | 2026-09-01 11:51 |
| qwen2.5:0.5b     | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 1 594.1       | 49.7         | 35      | 2 181      | 121       | 3.433      |              |             | 2026-08-18 01:14 |
| qwen2.5:0.5b     | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **769.5**     | **49.1**     | **42**  | **989**    | **63**    | **1.489**  |              |             | 2026-08-18 01:14 |
| qwen2.5:0.5b     | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 2 829.9       | 303.6        | 128     | 1 923      |  -        |  -         |              |             | 2026-09-01 08:57 |
| qwen2.5:0.5b     | 4096   | short  | stream     | GPU    | vLLM 0.22.0     |  -            | 502.5        | 128     | 273        | 51        | 0.398      |              |             | 2026-08-10 07:11 |
| qwen2.5:0.5b     | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **784.7**     | **366.6**    | **128** | **415**    | ** - **   | ** - **    | -27.0%       |             | 2026-09-01 08:57 |

### qwen3

![qwen3](img/family-qwen3.svg)

| Model        | Ctx   | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode    | Δ energy    | Date             |
|--------------|-------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|-------------|-------------|------------------|
| qwen3:0.6b   | 4096  | long   | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent  |             | 2026-08-18 01:14 |
| qwen3:0.6b   | 4096  | long   | non-stream | GPU    | **loken 0.1.0** | **26 432.8**  | **652.0**    | **128** | **262**    | **37**  | **0.289** |             |             | 2026-08-18 01:14 |
| qwen3:0.6b   | 4096  | long   | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent  |             | 2026-08-18 01:14 |
| qwen3:0.6b   | 4096  | long   | stream     | GPU    | **loken 0.1.0** | **24 191.2**  | **649.5**    | **128** | **268**    | **35**  | **0.276** |             |             | 2026-08-18 01:14 |
| qwen3:0.6b   | 4096  | medium | non-stream | GPU    | Ollama 0.32.6   | 12 423.7      | 198.7        | 128     | 1 776      |  -      |  -        |             |             | 2026-09-01 15:20 |
| qwen3:0.6b   | 4096  | medium | non-stream | GPU    | **loken 0.1.0** | **2 260.7**   | **520.1**    | **128** | **257**    | ** - ** | ** - **   | **+161.7%** |             | 2026-09-01 15:20 |
| qwen3:0.6b   | 4096  | medium | stream     | GPU    | Ollama 0.32.6   | 12 016.5      | 455.9        | 128     | 1 766      |  -      |  -        |             |             | 2026-09-01 13:44 |
| qwen3:0.6b   | 4096  | medium | stream     | GPU    | **loken 0.1.0** | **1 759.8**   | **718.1**    | **128** | **265**    | ** - ** | ** - **   | **+57.5%**  |             | 2026-09-01 13:44 |
| qwen3:0.6b   | 4096  | short  | non-stream | GPU    | Ollama 0.32.6   | 4 336.4       | 190.0        | 128     | 1 792      |  -      |  -        |             |             | 2026-09-01 12:06 |
| qwen3:0.6b   | 4096  | short  | non-stream | GPU    | **loken 0.1.0** | **671.3**     | **500.5**    | **128** | **268**    | ** - ** | ** - **   | **+163.4%** |             | 2026-09-01 12:06 |
| qwen3:0.6b   | 4096  | short  | stream     | CPU    | Ollama 0.32.6   | 1 154.6       | 45.6         | 128     | 4 286      | 186     | 1.451     |             |             | 2026-08-18 01:14 |
| qwen3:0.6b   | 4096  | short  | stream     | CPU    | **loken 0.1.0** | **178.7**     | **46.8**     | **128** | **2 904**  | **178** | **1.393** | **+2.7%**   | **+4.1%**   | 2026-08-18 01:14 |
| qwen3:0.6b   | 4096  | short  | stream     | GPU    | Ollama 0.32.6   | 4 326.7       | 469.9        | 128     | 1 739      |  -      |  -        |             |             | 2026-09-01 10:26 |
| qwen3:0.6b   | 4096  | short  | stream     | GPU    | vLLM 0.22.0     |  -            | 473.0        | 128     | 280        | 55      | 0.430     |             |             | 2026-08-10 07:10 |
| qwen3:0.6b   | 4096  | short  | stream     | GPU    | **loken 0.1.0** | **663.3**     | **708.9**    | **128** | **262**    | ** - ** | ** - **   | **+49.9%**  |             | 2026-09-01 10:26 |
| qwen3:0.6b   | 16384 | long   | stream     | GPU    | Ollama 0.32.6   | 65 512.2      | 599.1        | 128     | 2 021      | 154     | 1.201     |             |             | 2026-08-18 01:14 |
| qwen3:0.6b   | 16384 | long   | stream     | GPU    | **loken 0.1.0** | **16 523.9**  | **646.5**    | **128** | **270**    | **48**  | **0.378** | **+7.9%**   | **+217.4%** | 2026-08-18 01:14 |
| qwen3:0.6b   | 16384 | short  | stream     | GPU    | Ollama 0.32.6   | 11 388.5      | 528.7        | 93      | 1 835      | 441     | 4.741     |             |             | 2026-08-18 01:14 |
| qwen3:0.6b   | 16384 | short  | stream     | GPU    | **loken 0.1.0** | **2 105.9**   | **670.3**    | **111** | **249**    | **32**  | **0.285** |             |             | 2026-08-18 01:14 |
| qwen3:0.6b   | 32768 | long   | stream     | GPU    | Ollama 0.32.6   | 68 185.4      | 604.4        | 128     | 2 023      | 153     | 1.196     |             |             | 2026-08-18 01:14 |
| qwen3:0.6b   | 32768 | long   | stream     | GPU    | **loken 0.1.0** | **16 473.2**  | **651.9**    | **128** | **258**    | **47**  | **0.364** | **+7.9%**   | **+228.9%** | 2026-08-18 01:14 |
| qwen3:1.7b   | 4096  | short  | stream     | GPU    | Ollama 0.32.6   | 3 843.4       | 407.7        | 128     | 534        |  -      |  -        |             |             | 2026-08-27 14:34 |
| qwen3:1.7b   | 4096  | short  | stream     | GPU    | **loken 0.1.0** | **524.1**     | **427.8**    | **128** | **375**    | ** - ** | ** - **   | **+4.9%**   |             | 2026-08-27 14:34 |
| qwen3:8b     | 4096  | long   | non-stream | GPU    | Ollama 0.32.6   | 19 399.7      | 138.0        | 128     | 2 995      | 385     | 3.010     |             |             | 2026-08-18 01:14 |
| qwen3:8b     | 4096  | long   | non-stream | GPU    | vLLM 0.22.0     |  -            | 132.5        | 128     | 970        | 197     | 1.542     |             |             | 2026-08-18 01:14 |
| qwen3:8b     | 4096  | long   | non-stream | GPU    | **loken 0.1.0** | **20 485.7**  | **144.4**    | **128** | **955**    | **236** | **1.847** | **+4.7%**   | -16.5%      | 2026-08-18 01:14 |
| qwen3:8b     | 4096  | long   | stream     | GPU    | Ollama 0.32.6   | 19 559.7      | 136.7        | 128     | 3 031      | 373     | 2.915     |             |             | 2026-08-18 01:14 |
| qwen3:8b     | 4096  | long   | stream     | GPU    | vLLM 0.22.0     |  -            | 139.4        | 128     | 971        | 195     | 1.525     |             |             | 2026-08-18 01:14 |
| qwen3:8b     | 4096  | long   | stream     | GPU    | **loken 0.1.0** | **22 262.6**  | **143.0**    | **128** | **965**    | **234** | **1.830** | **+2.6%**   | -16.7%      | 2026-08-18 01:14 |
| qwen3:8b     | 4096  | medium | non-stream | GPU    | Ollama 0.32.6   | 2 856.4       | 63.2         | 128     | 3 211      |  -      |  -        |             |             | 2026-09-01 15:23 |
| qwen3:8b     | 4096  | medium | non-stream | GPU    | vLLM 0.22.0     |  -            | 133.2        | 128     | 964        |  -      |  -        |             |             | 2026-09-01 15:23 |
| qwen3:8b     | 4096  | medium | non-stream | GPU    | **loken 0.1.0** | **2 323.8**   | **131.2**    | **128** | **978**    | ** - ** | ** - **   | -1.5%       |             | 2026-09-01 15:23 |
| qwen3:8b     | 4096  | medium | stream     | GPU    | Ollama 0.32.6   | 2 827.9       | 96.7         | 128     | 3 241      |  -      |  -        |             |             | 2026-09-01 13:46 |
| qwen3:8b     | 4096  | medium | stream     | GPU    | vLLM 0.22.0     |  -            | 139.6        | 128     | 970        |  -      |  -        |             |             | 2026-09-01 13:46 |
| qwen3:8b     | 4096  | medium | stream     | GPU    | **loken 0.1.0** | **1 226.4**   | **138.7**    | **128** | **1 014**  | ** - ** | ** - **   | -0.6%       |             | 2026-09-01 13:46 |
| qwen3:8b     | 4096  | short  | non-stream | GPU    | Ollama 0.32.6   | 1 033.8       | 63.3         | 128     | 3 185      |  -      |  -        |             |             | 2026-09-01 12:08 |
| qwen3:8b     | 4096  | short  | non-stream | GPU    | vLLM 0.22.0     |  -            | 133.8        | 128     | 959        |  -      |  -        |             |             | 2026-09-01 12:08 |
| qwen3:8b     | 4096  | short  | non-stream | GPU    | **loken 0.1.0** | **698.4**     | **137.5**    | **128** | **933**    | ** - ** | ** - **   | **+2.8%**   |             | 2026-09-01 12:08 |
| qwen3:8b     | 4096  | short  | stream     | CPU    | Ollama 0.32.6   | 111.7         | 4.6          | 128     | 30 746     | 1 045   | 8.166     |             |             | 2026-08-18 01:14 |
| qwen3:8b     | 4096  | short  | stream     | CPU    | **loken 0.1.0** | **71.4**      | **5.0**      | **128** | **26 755** | **875** | **6.836** | **+8.9%**   | **+19.4%**  | 2026-08-18 01:14 |
| qwen3:8b     | 4096  | short  | stream     | GPU    | Ollama 0.32.6   | 1 027.4       | 96.7         | 128     | 3 221      |  -      |  -        |             |             | 2026-09-01 10:30 |
| qwen3:8b     | 4096  | short  | stream     | GPU    | vLLM 0.22.0     |  -            | 139.6        | 128     | 12 064     |  -      |  -        |             |             | 2026-09-01 10:30 |
| qwen3:8b     | 4096  | short  | stream     | GPU    | **loken 0.1.0** | **542.7**     | **145.4**    | **128** | **954**    | ** - ** | ** - **   | **+4.2%**   |             | 2026-09-01 10:30 |
| qwen3:8b     | 16384 | long   | stream     | GPU    | Ollama 0.32.6   | 20 813.5      | 136.3        | 128     | 3 127      | 368     | 2.871     |             |             | 2026-08-18 01:14 |
| qwen3:8b     | 16384 | long   | stream     | GPU    | **loken 0.1.0** | **21 874.2**  | **140.2**    | **128** | **981**    | **236** | **1.846** | **+2.8%**   | **+55.5%**  | 2026-08-18 01:14 |
| qwen3:8b     | 16384 | short  | stream     | GPU    | Ollama 0.32.6   | 2 904.4       | 138.9        | 128     | 2 991      | 354     | 2.765     |             |             | 2026-08-18 01:14 |
| qwen3:8b     | 16384 | short  | stream     | GPU    | **loken 0.1.0** | **1 724.6**   | **149.6**    | **128** | **936**    | **220** | **1.717** | **+7.7%**   | **+61.0%**  | 2026-08-18 01:14 |
| qwen3:8b     | 32768 | long   | stream     | GPU    | Ollama 0.32.6   | 20 850.4      | 135.9        | 128     | 3 119      | 364     | 2.844     |             |             | 2026-08-18 01:14 |
| qwen3:8b     | 32768 | long   | stream     | GPU    | **loken 0.1.0** | **17 655.2**  | **140.3**    | **128** | **981**    | **238** | **1.860** | **+3.3%**   | **+52.9%**  | 2026-08-18 01:14 |
| qwen3:latest | 4096  | short  | stream     | GPU    | **loken 0.1.0** | **2 024.3**   | **141.1**    | **128** | **1 024**  | **220** | **1.716** |             |             | 2026-08-18 01:14 |

### qwen35

![qwen35](img/family-qwen35.svg)

| Model          | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token   | Δ decode    | Δ energy   | Date             |
|----------------|------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|-----------|-------------|------------|------------------|
| qwen3.5:latest | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 4 346.3       | 102.8        | 128     | 3 976     | 466     | 3.642     |             |            | 2026-08-18 01:14 |
| qwen3.5:latest | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **393 160.0** | **112.5**    | **128** | **1 317** | **287** | **2.242** | **+9.5%**   | **+62.5%** | 2026-08-18 01:14 |
| qwen3.5:latest | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 4 329.8       | 102.6        | 128     | 3 981     | 446     | 3.481     |             |            | 2026-08-18 01:14 |
| qwen3.5:latest | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **230 400.8** | **111.9**    | **128** | **1 331** | **287** | **2.243** | **+9.1%**   | **+55.2%** | 2026-08-18 01:14 |
| qwen3.5:latest | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 615.9         | 46.2         | 128     | 4 211     |  -      |  -        |             |            | 2026-09-01 15:19 |
| qwen3.5:latest | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **98 697.8**  | **108.5**    | **128** | **1 242** | ** - ** | ** - **   | **+135.0%** |            | 2026-09-01 15:19 |
| qwen3.5:latest | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 572.7         | 76.3         | 128     | 4 281     |  -      |  -        |             |            | 2026-09-01 13:43 |
| qwen3.5:latest | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **43 852.0**  | **123.7**    | **128** | **1 287** | ** - ** | ** - **   | **+62.2%**  |            | 2026-09-01 13:43 |
| qwen3.5:latest | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 198.6         | 45.5         | 128     | 4 292     |  -      |  -        |             |            | 2026-09-01 12:05 |
| qwen3.5:latest | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **38 476.4**  | **107.3**    | **128** | **1 272** | ** - ** | ** - **   | **+135.7%** |            | 2026-09-01 12:05 |
| qwen3.5:latest | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 213.8         | 75.2         | 128     | 4 250     |  -      |  -        |             |            | 2026-09-01 10:25 |
| qwen3.5:latest | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **25 974.3**  | **126.1**    | **128** | **1 267** | ** - ** | ** - **   | **+67.6%**  |            | 2026-09-01 10:25 |

### qwen35moe

![qwen35moe](img/family-qwen35moe.svg)

| Model       | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token   | Δ decode   | Δ energy    | Date             |
|-------------|------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|-----------|------------|-------------|------------------|
| qwen3.5:35b | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 2 740.5       | 115.2        | 128     | 22 116    | 1 517   | 11.849    |            |             | 2026-08-18 01:14 |
| qwen3.5:35b | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **356 035.2** | **87.7**     | **128** | **1 885** | **269** | **2.099** | -23.8%     | **+464.4%** | 2026-08-18 01:14 |
| qwen3.5:35b | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 2 726.2       | 114.9        | 128     | 22 474    | 1 444   | 11.282    |            |             | 2026-08-18 01:14 |
| qwen3.5:35b | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **212 543.3** | **90.0**     | **128** | **1 895** | **260** | **2.031** | -21.6%     | **+455.4%** | 2026-08-18 01:14 |
| qwen3.5:35b | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 477.3         | 58.7         | 128     | 5 103     |  -      |  -        |            |             | 2026-09-01 15:18 |
| qwen3.5:35b | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **19 057.3**  | **107.0**    | **128** | **1 355** | ** - ** | ** - **   | **+82.3%** |             | 2026-09-01 15:18 |
| qwen3.5:35b | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 481.7         | 113.0        | 128     | 20 424    |  -      |  -        |            |             | 2026-09-01 13:42 |
| qwen3.5:35b | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **35 323.6**  | **133.2**    | **128** | **1 366** | ** - ** | ** - **   | **+17.8%** |             | 2026-09-01 13:42 |
| qwen3.5:35b | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 166.6         | 55.9         | 128     | 20 500    |  -      |  -        |            |             | 2026-09-01 12:03 |
| qwen3.5:35b | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **20 501.2**  | **108.7**    | **128** | **1 325** | ** - ** | ** - **   | **+94.4%** |             | 2026-09-01 12:03 |
| qwen3.5:35b | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 165.6         | 114.1        | 128     | 5 100     |  -      |  -        |            |             | 2026-09-01 10:24 |
| qwen3.5:35b | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **13 220.2**  | **134.3**    | **128** | **1 308** | ** - ** | ** - **   | **+17.6%** |             | 2026-09-01 10:24 |

### qwen3moe

![qwen3moe](img/family-qwen3moe.svg)

| Model           | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token   | Δ decode   | Δ energy | Date             |
|-----------------|------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|-----------|------------|----------|------------------|
| qwen3-coder:30b | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 17 683.9      | 144.7        | 128     | 4 106     | 397     | 3.099     |            |          | 2026-08-18 01:14 |
| qwen3-coder:30b | 4096 | long   | non-stream | GPU    | vLLM 0.22.0     |  -            | 125.4        | 128     | 1 032     | 166     | 1.297     |            |          | 2026-08-18 01:14 |
| qwen3-coder:30b | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **4 087.9**   | **141.5**    | **128** | **1 440** | **206** | **1.610** | -2.2%      | -19.5%   | 2026-08-18 01:14 |
| qwen3-coder:30b | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 17 987.0      | 143.6        | 128     | 4 071     | 398     | 3.108     |            |          | 2026-08-18 01:14 |
| qwen3-coder:30b | 4096 | long   | stream     | GPU    | vLLM 0.22.0     |  -            | 129.6        | 128     | 1 066     | 179     | 1.401     |            |          | 2026-08-18 01:14 |
| qwen3-coder:30b | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **4 050.2**   | **142.5**    | **128** | **1 432** | **217** | **1.698** | -0.8%      | -17.5%   | 2026-08-18 01:14 |
| qwen3-coder:30b | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 4 172.8       | 85.3         | 128     | 3 983     |  -      |  -        |            |          | 2026-09-01 15:16 |
| qwen3-coder:30b | 4096 | medium | non-stream | GPU    | vLLM 0.22.0     |  -            | 127.6        | 128     | 1 009     |  -      |  -        |            |          | 2026-09-01 15:16 |
| qwen3-coder:30b | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **344.2**     | **127.8**    | **128** | **1 201** | ** - ** | ** - **   | **+0.1%**  |          | 2026-09-01 15:16 |
| qwen3-coder:30b | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 4 133.8       | 146.5        | 128     | 3 988     |  -      |  -        |            |          | 2026-09-01 13:38 |
| qwen3-coder:30b | 4096 | medium | stream     | GPU    | vLLM 0.22.0     |  -            | 138.7        | 128     | 995       |  -      |  -        |            |          | 2026-09-01 13:38 |
| qwen3-coder:30b | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **383.8**     | **177.4**    | **128** | **1 211** | ** - ** | ** - **   | **+21.1%** |          | 2026-09-01 13:38 |
| qwen3-coder:30b | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 1 435.7       | 85.9         | 128     | 3 943     |  -      |  -        |            |          | 2026-09-01 12:00 |
| qwen3-coder:30b | 4096 | short  | non-stream | GPU    | vLLM 0.22.0     |  -            | 129.7        | 128     | 996       |  -      |  -        |            |          | 2026-09-01 12:00 |
| qwen3-coder:30b | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **248.7**     | **134.8**    | **128** | **1 170** | ** - ** | ** - **   | **+3.9%**  |          | 2026-09-01 12:00 |
| qwen3-coder:30b | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 1 430.0       | 147.4        | 128     | 3 967     |  -      |  -        |            |          | 2026-09-01 10:22 |
| qwen3-coder:30b | 4096 | short  | stream     | GPU    | vLLM 0.22.0     |  -            | 141.1        | 128     | 968       |  -      |  -        |            |          | 2026-09-01 10:22 |
| qwen3-coder:30b | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **263.9**     | **178.8**    | **128** | **1 178** | ** - ** | ** - **   | **+21.3%** |          | 2026-09-01 10:22 |

### qwen3next

![qwen3next](img/family-qwen3next.svg)

| Model                   | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms      | J/req     | J/token    | Δ decode | Δ energy | Date             |
|-------------------------|------|--------|------------|--------|-----------------|---------------|--------------|---------|-------------|-----------|------------|----------|----------|------------------|
| qwen3-coder-next:latest | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 965.5         | 26.7         | 128     | 11 500      | 917       | 7.163      |          |          | 2026-08-18 01:14 |
| qwen3-coder-next:latest | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **3 235.7**   | **5.7**      | **128** | **98 838**  | **4 334** | **33.861** | -78.6%   | -78.8%   | 2026-08-18 01:14 |
| qwen3-coder-next:latest | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 1 051.0       | 26.7         | 128     | 11 459      | 915       | 7.145      |          |          | 2026-08-18 01:14 |
| qwen3-coder-next:latest | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **5 049.3**   | **5.7**      | **128** | **102 055** | **4 526** | **35.357** | -78.6%   | -79.8%   | 2026-08-18 01:14 |
| qwen3-coder-next:latest | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 216.7         | 19.4         | 128     | 10 592      |  -        |  -         |          |          | 2026-09-01 15:12 |
| qwen3-coder-next:latest | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **594.5**     | **8.2**      | **21**  | **11 599**  | ** - **   | ** - **    |          |          | 2026-09-01 15:12 |
| qwen3-coder-next:latest | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 215.2         | 28.4         | 128     | 10 639      |  -        |  -         |          |          | 2026-09-01 13:34 |
| qwen3-coder-next:latest | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **1 011.2**   | **9.1**      | **128** | **23 020**  | ** - **   | ** - **    | -68.1%   |          | 2026-09-01 13:34 |
| qwen3-coder-next:latest | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 81.9          | 19.4         | 128     | 10 435      |  -        |  -         |          |          | 2026-09-01 11:56 |
| qwen3-coder-next:latest | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **409.3**     | **9.1**      | **128** | **23 265**  | ** - **   | ** - **    | -53.0%   |          | 2026-09-01 11:56 |
| qwen3-coder-next:latest | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 80.1          | 28.1         | 128     | 10 488      |  -        |  -         |          |          | 2026-09-01 10:17 |
| qwen3-coder-next:latest | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **305.3**     | **9.2**      | **128** | **23 157**  | ** - **   | ** - **    | -67.1%   |          | 2026-09-01 10:17 |
| qwen3next:latest        | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 1 042.6       | 30.4         | 128     | 10 612      | 850       | 6.644      |          |          | 2026-08-18 01:14 |
| qwen3next:latest        | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **6 038.2**   | **6.4**      | **128** | **65 338**  | **3 086** | **24.112** | -79.1%   | -72.4%   | 2026-08-18 01:14 |
| qwen3next:latest        | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 1 136.1       | 30.2         | 128     | 10 706      | 854       | 6.674      |          |          | 2026-08-18 01:14 |
| qwen3next:latest        | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **5 519.9**   | **6.3**      | **128** | **68 790**  | **3 339** | **26.088** | -79.0%   | -74.4%   | 2026-08-18 01:14 |
| qwen3next:latest        | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 231.7         | 22.2         | 128     | 9 826       |  -        |  -         |          |          | 2026-09-01 15:27 |
| qwen3next:latest        | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **1 239.8**   | **9.5**      | **128** | **21 099**  | ** - **   | ** - **    | -57.1%   |          | 2026-09-01 15:27 |
| qwen3next:latest        | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 228.8         | 32.5         | 128     | 9 935       |  -        |  -         |          |          | 2026-09-01 13:51 |
| qwen3next:latest        | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **1 201.3**   | **9.5**      | **128** | **23 900**  | ** - **   | ** - **    | -70.7%   |          | 2026-09-01 13:51 |
| qwen3next:latest        | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 86.4          | 22.2         | 128     | 9 696       |  -        |  -         |          |          | 2026-09-01 12:13 |
| qwen3next:latest        | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **318.9**     | **9.6**      | **128** | **21 282**  | ** - **   | ** - **    | -56.6%   |          | 2026-09-01 12:13 |
| qwen3next:latest        | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 86.4          | 32.9         | 128     | 9 659       |  -        |  -         |          |          | 2026-09-01 10:34 |
| qwen3next:latest        | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **187.8**     | **9.9**      | **128** | **20 982**  | ** - **   | ** - **    | -70.0%   |          | 2026-09-01 10:34 |

### smollm3

![smollm3](img/family-smollm3.svg)

| Model          | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode    | Δ energy   | Date             |
|----------------|------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|-------------|------------|------------------|
| smollm3:latest | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 28 723.2      | 220.5        | 128     | 2 528      | 252     | 1.965     |             |            | 2026-08-18 01:14 |
| smollm3:latest | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **29 576.9**  | **237.0**    | **128** | **649**    | **135** | **1.055** | **+7.5%**   | **+86.3%** | 2026-08-18 01:14 |
| smollm3:latest | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 29 317.1      | 222.7        | 128     | 2 449      | 257     | 2.007     |             |            | 2026-08-18 01:14 |
| smollm3:latest | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **27 845.1**  | **233.8**    | **128** | **657**    | **133** | **1.037** | **+5.0%**   | **+93.5%** | 2026-08-18 01:14 |
| smollm3:latest | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 4 742.2       | 93.7         | 128     | 2 497      |  -      |  -        |             |            | 2026-09-01 15:28 |
| smollm3:latest | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **2 952.8**   | **224.4**    | **128** | **581**    | ** - ** | ** - **   | **+139.4%** |            | 2026-09-01 15:28 |
| smollm3:latest | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 4 768.2       | 169.6        | 128     | 2 500      |  -      |  -        |             |            | 2026-09-01 13:52 |
| smollm3:latest | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **915.8**     | **251.8**    | **128** | **616**    | ** - ** | ** - **   | **+48.5%**  |            | 2026-09-01 13:52 |
| smollm3:latest | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 1 676.7       | 96.3         | 128     | 2 461      |  -      |  -        |             |            | 2026-09-01 12:14 |
| smollm3:latest | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **350.8**     | **214.0**    | **128** | **610**    | ** - ** | ** - **   | **+122.2%** |            | 2026-09-01 12:14 |
| smollm3:latest | 4096 | short  | stream     | CPU    | Ollama 0.32.6   | 693.1         | 11.0         | 128     | 13 711     | 504     | 3.935     |             |            | 2026-08-18 01:14 |
| smollm3:latest | 4096 | short  | stream     | CPU    | **loken 0.1.0** | **140.4**     | **10.6**     | **128** | **12 517** | **558** | **4.358** | -3.5%       | -9.7%      | 2026-08-18 01:14 |
| smollm3:latest | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 1 687.1       | 170.9        | 128     | 2 491      |  -      |  -        |             |            | 2026-09-01 10:36 |
| smollm3:latest | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **694.5**     | **256.7**    | **128** | **578**    | ** - ** | ** - **   | **+50.2%**  |            | 2026-09-01 10:36 |
