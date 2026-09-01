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

Time to first token is the honest figure for short prompts - 20.6 ms against 429.6 on that
cell. The prefill column becomes a comparison again at the medium and long prompt lengths,
where the work outgrows what either timer leaves out.



### ernie4_5

![ernie4_5](img/family-ernie4_5.svg)

| Model           | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token   | Δ decode    | Δ energy | Date             |
|-----------------|------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|-----------|-------------|----------|------------------|
| ernie4-5:latest | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 87 143.4      | 481.2        | 128     | 1 408     | 116     | 0.907     |             |          | 2026-08-18 01:14 |
| ernie4-5:latest | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **219 434.0** | **519.7**    | **98**  | **242**   | **33**  | **0.336** |             |          | 2026-08-18 01:14 |
| ernie4-5:latest | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 84 524.2      | 483.0        | 128     | 1 896     | 148     | 1.156     |             |          | 2026-08-18 01:14 |
| ernie4-5:latest | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **187 373.1** | **512.7**    | **98**  | **245**   | **32**  | **0.327** |             |          | 2026-08-18 01:14 |
| ernie4-5:latest | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 13 799.7      | 458.7        | 22      | 1 199     | 93      | 4.209     |             |          | 2026-08-18 01:14 |
| ernie4-5:latest | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **51 462.4**  | **580.3**    | **128** | **263**   | **38**  | **0.295** |             |          | 2026-08-18 01:14 |
| ernie4-5:latest | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 14 202.6      | 470.3        | 22      | 1 200     | 93      | 4.237     |             |          | 2026-08-18 01:14 |
| ernie4-5:latest | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **46 364.9**  | **529.5**    | **128** | **283**   | **38**  | **0.301** |             |          | 2026-08-18 01:14 |
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
| gemma4:12b    | 4096  | medium | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent  |             | 2026-08-18 01:14 |
| gemma4:12b    | 4096  | medium | non-stream | GPU    | **loken 0.1.0** | **2 242.0**   | **79.6**     | **128** | **1 661**  | **393** | **3.070** |             |             | 2026-08-18 01:14 |
| gemma4:12b    | 4096  | medium | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent  |             | 2026-08-18 01:14 |
| gemma4:12b    | 4096  | medium | stream     | GPU    | **loken 0.1.0** | **1 488.6**   | **77.5**     | **128** | **1 703**  | **393** | **3.069** |             |             | 2026-08-18 01:14 |
| gemma4:12b    | 4096  | short  | non-stream | GPU    | Ollama 0.32.6   | 288.1         | 22.4         | 61      | 3 886      |  -      |  -        |             |             | 2026-09-01 11:16 |
| gemma4:12b    | 4096  | short  | non-stream | GPU    | **loken 0.1.0** | **517.3**     | **71.8**     | **128** | **1 784**  | ** - ** | ** - **   |             |             | 2026-09-01 11:16 |
| gemma4:12b    | 4096  | short  | stream     | GPU    | Ollama 0.32.6   | 289.3         | 42.6         | 61      | 5 331      |  -      |  -        |             |             | 2026-09-01 09:37 |
| gemma4:12b    | 4096  | short  | stream     | GPU    | **loken 0.1.0** | **432.3**     | **71.9**     | **128** | **1 837**  | ** - ** | ** - **   |             |             | 2026-09-01 09:37 |
| gemma4:26b    | 4096  | long   | non-stream | GPU    | Ollama 0.32.6   | 2 184.8       | 98.5         | 128     | 5 135      | 480     | 3.750     |             |             | 2026-08-18 01:14 |
| gemma4:26b    | 4096  | long   | non-stream | GPU    | **loken 0.1.0** | **1 169.0**   | **88.3**     | **128** | **1 754**  | **274** | **2.139** | -10.3%      | **+75.3%**  | 2026-08-18 01:14 |
| gemma4:26b    | 4096  | long   | stream     | GPU    | Ollama 0.32.6   | 2 158.9       | 97.2         | 128     | 5 137      | 485     | 3.786     |             |             | 2026-08-18 01:14 |
| gemma4:26b    | 4096  | long   | stream     | GPU    | **loken 0.1.0** | **1 173.7**   | **85.8**     | **128** | **1 803**  | **274** | **2.137** | -11.7%      | **+77.2%**  | 2026-08-18 01:14 |
| gemma4:26b    | 4096  | medium | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent  |             | 2026-08-18 01:14 |
| gemma4:26b    | 4096  | medium | non-stream | GPU    | **loken 0.1.0** | **524.4**     | **91.3**     | **128** | **1 570**  | **230** | **1.795** |             |             | 2026-08-18 01:14 |
| gemma4:26b    | 4096  | medium | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent  |             | 2026-08-18 01:14 |
| gemma4:26b    | 4096  | medium | stream     | GPU    | **loken 0.1.0** | **573.6**     | **89.6**     | **128** | **1 607**  | **229** | **1.789** |             |             | 2026-08-18 01:14 |
| gemma4:26b    | 4096  | short  | non-stream | GPU    | Ollama 0.32.6   | 84.5          | 11.4         | 128     | 4 582      |  -      |  -        |             |             | 2026-09-01 11:18 |
| gemma4:26b    | 4096  | short  | non-stream | GPU    | **loken 0.1.0** | **307.8**     | **88.4**     | **128** | **1 463**  | ** - ** | ** - **   | **+676.5%** |             | 2026-09-01 11:18 |
| gemma4:26b    | 4096  | short  | stream     | GPU    | Ollama 0.32.6   | 5.1           | 101.3        | 93      | 5 661      |  -      |  -        |             |             | 2026-09-01 09:40 |
| gemma4:26b    | 4096  | short  | stream     | GPU    | **loken 0.1.0** | **245.5**     | **95.5**     | **128** | **1 519**  | ** - ** | ** - **   |             |             | 2026-09-01 09:40 |
| gemma4:31b    | 4096  | long   | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           |  -      |  -         |  -      |  -        | incoherent  |             | 2026-08-18 01:14 |
| gemma4:31b    | 4096  | long   | non-stream | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent  |             | 2026-08-18 01:14 |
| gemma4:31b    | 4096  | long   | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 44      |  -         |  -      |  -        | incoherent  |             | 2026-08-18 01:14 |
| gemma4:31b    | 4096  | long   | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent  |             | 2026-08-18 01:14 |
| gemma4:31b    | 4096  | medium | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           |  -      | 5 810      |  -      |  -        |             |             | 2026-08-18 01:14 |
| gemma4:31b    | 4096  | medium | non-stream | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent  |             | 2026-08-18 01:14 |
| gemma4:31b    | 4096  | medium | stream     | GPU    | Ollama 0.32.6   |  -            | 25.5         | 48      | 5 817      | 689     | 14.352    |             |             | 2026-08-18 01:14 |
| gemma4:31b    | 4096  | medium | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent  |             | 2026-08-18 01:14 |
| gemma4:31b    | 4096  | short  | non-stream | GPU    | Ollama 0.32.6   | 76.1          | 8.3          | 128     | 7 646      |  -      |  -        |             |             | 2026-09-01 11:21 |
| gemma4:31b    | 4096  | short  | non-stream | GPU    | **loken 0.1.0** | **194.1**     | **23.9**     | **128** | **5 370**  | ** - ** | ** - **   | **+188.0%** |             | 2026-09-01 11:21 |
| gemma4:31b    | 4096  | short  | stream     | GPU    | Ollama 0.32.6   | 83.3          | 25.7         | 96      | 7 652      |  -      |  -        |             |             | 2026-09-01 09:42 |
| gemma4:31b    | 4096  | short  | stream     | GPU    | **loken 0.1.0** | **193.8**     | **24.4**     | **128** | **5 407**  | ** - ** | ** - **   |             |             | 2026-09-01 09:42 |
| gemma4:latest | 4096  | long   | non-stream | GPU    | Ollama 0.32.6   | 8 157.1       | 115.9        | 128     | 3 841      | 380     | 2.968     |             |             | 2026-08-18 01:14 |
| gemma4:latest | 4096  | long   | stream     | GPU    | Ollama 0.32.6   | 8 168.3       | 115.3        | 128     | 3 973      | 396     | 3.096     |             |             | 2026-08-18 01:14 |
| gemma4:latest | 4096  | long   | stream     | GPU    | **loken 0.1.0** | **17 599.2**  | **128.9**    | **128** | **1 064**  | **194** | **1.515** | **+11.8%**  | **+104.4%** | 2026-08-18 01:14 |
| gemma4:latest | 4096  | medium | non-stream | GPU    | Ollama 0.32.6   | 1 749.3       | 118.2        | 128     | 3 926      | 355     | 2.776     |             |             | 2026-08-18 01:14 |
| gemma4:latest | 4096  | medium | non-stream | GPU    | **loken 0.1.0** | **1 776.1**   | **123.5**    | **9**   | **127**    | **17**  | **1.927** |             |             | 2026-08-18 01:14 |
| gemma4:latest | 4096  | medium | stream     | GPU    | Ollama 0.32.6   | 1 763.2       | 118.6        | 128     | 3 753      | 379     | 2.961     |             |             | 2026-08-18 01:14 |
| gemma4:latest | 4096  | medium | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent  |             | 2026-08-18 01:14 |
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
| gpt-oss:20b | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 1 668.9       | 131.2        | 128     | 3 617   | 355     | 2.772     |             |             | 2026-08-18 01:14 |
| gpt-oss:20b | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **90 783.2**  | **199.5**    | **128** | **818** | **145** | **1.136** | **+52.1%**  | **+144.1%** | 2026-08-18 01:14 |
| gpt-oss:20b | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 1 664.4       | 130.0        | 128     | 3 590   | 355     | 2.771     |             |             | 2026-08-18 01:14 |
| gpt-oss:20b | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **59 658.7**  | **197.5**    | **128** | **825** | **143** | **1.116** | **+51.9%**  | **+148.3%** | 2026-08-18 01:14 |
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
| granite3.1-dense:2b | 4096  | medium | non-stream | GPU    | Ollama 0.32.6   | 9 262.4       | 293.2        | 128     | 1 815      | 192     | 1.500     |            |             | 2026-08-18 01:14 |
| granite3.1-dense:2b | 4096  | medium | non-stream | GPU    | **loken 0.1.0** | **4 100.7**   | **296.8**    | **128** | **497**    | **99**  | **0.771** | **+1.2%**  | **+94.6%**  | 2026-08-18 01:14 |
| granite3.1-dense:2b | 4096  | medium | stream     | GPU    | Ollama 0.32.6   | 9 397.9       | 290.7        | 128     | 1 818      | 193     | 1.508     |            |             | 2026-08-18 01:14 |
| granite3.1-dense:2b | 4096  | medium | stream     | GPU    | **loken 0.1.0** | **3 530.3**   | **292.3**    | **128** | **484**    | **108** | **0.847** | **+0.6%**  | **+78.0%**  | 2026-08-18 01:14 |
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
| granite3-moe:1b | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 9 978.4       | 363.2        | 128     | 2 411     | 179     | 1.399     |            |             | 2026-08-18 01:14 |
| granite3-moe:1b | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **341.3**     | **369.0**    | **128** | **656**   | **65**  | **0.507** | **+1.6%**  | **+176.3%** | 2026-08-18 01:14 |
| granite3-moe:1b | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 11 039.6      | 370.9        | 128     | 417       |  -      |  -        |            |             | 2026-08-27 20:06 |
| granite3-moe:1b | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **2 174.9**   | **323.9**    | **128** | **422**   | ** - ** | ** - **   | -12.7%     |             | 2026-08-27 20:06 |
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
| lfm2.5-thinking:latest | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 9 832.1         | 501.5        | 128     | 1 852   | 166     | 1.297     |             |             | 2026-08-18 01:14 |
| lfm2.5-thinking:latest | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **32 654.0**    | **630.6**    | **128** | **345** | **50**  | **0.391** | **+25.7%**  | **+231.6%** | 2026-08-18 01:14 |
| lfm2.5-thinking:latest | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 10 056.2        | 500.3        | 128     | 1 841   | 161     | 1.258     |             |             | 2026-08-18 01:14 |
| lfm2.5-thinking:latest | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **68 776.3**    | **662.7**    | **128** | **335** | **60**  | **0.466** | **+32.5%**  | **+169.9%** | 2026-08-18 01:14 |
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
| lfm2:latest | 4096  | medium | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           |  -      |  -         |  -      |  -        | incoherent |             | 2026-08-18 01:14 |
| lfm2:latest | 4096  | medium | non-stream | GPU    | **loken 0.1.0** | **81 337.3**  | **312.2**    | **128** | **596**    | **92**  | **0.717** |            |             | 2026-08-18 01:14 |
| lfm2:latest | 4096  | medium | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 33      |  -         |  -      |  -        | incoherent |             | 2026-08-18 01:14 |
| lfm2:latest | 4096  | medium | stream     | GPU    | **loken 0.1.0** | **49 241.3**  | **310.0**    | **128** | **588**    | **88**  | **0.691** |            |             | 2026-08-18 01:14 |
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
| deepseek-r1:70b      | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 51.1          | 1.5          | 128     | 89 035     | 7 009     | 54.759     |             |             | 2026-08-18 01:14 |
| deepseek-r1:70b      | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **37.5**      | **1.4**      | **128** | **99 722** | **7 674** | **59.953** | -7.8%       | -8.7%       | 2026-08-18 01:14 |
| deepseek-r1:70b      | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 51.1          | 1.5          | 128     | 88 449     | 6 961     | 54.386     |             |             | 2026-08-18 01:14 |
| deepseek-r1:70b      | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **38.2**      | **1.5**      | **128** | **95 625** | **7 382** | **57.674** | -5.7%       | -5.7%       | 2026-08-18 01:14 |
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
| deepseek-r1:70b-q3ks | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 225.4         | 7.0          | 128     | 23 345     | 3 088     | 24.121     |             |             | 2026-08-18 01:14 |
| deepseek-r1:70b-q3ks | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **237.7**     | **18.6**     | **128** | **7 318**  | **2 352** | **18.374** | **+165.4%** | **+31.3%**  | 2026-08-18 01:14 |
| deepseek-r1:70b-q3ks | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 226.8         | 6.9          | 128     | 23 345     | 3 074     | 24.015     |             |             | 2026-08-18 01:14 |
| deepseek-r1:70b-q3ks | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **245.6**     | **18.4**     | **128** | **7 292**  | **2 378** | **18.579** | **+165.1%** | **+29.3%**  | 2026-08-18 01:14 |
| deepseek-r1:70b-q3ks | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 85.7          | 5.9          | 128     | 23 310     |  -        |  -         |             |             | 2026-09-01 10:58 |
| deepseek-r1:70b-q3ks | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **101.8**     | **20.2**     | **128** | **6 367**  | ** - **   | ** - **    | **+243.5%** |             | 2026-09-01 10:58 |
| deepseek-r1:70b-q3ks | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 86.0          | 6.9          | 128     | 23 287     |  -        |  -         |             |             | 2026-09-01 09:20 |
| deepseek-r1:70b-q3ks | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **103.6**     | **21.2**     | **128** | **6 401**  | ** - **   | ** - **    | **+205.7%** |             | 2026-09-01 09:20 |
| devstral:24b         | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 7 753.5       |  -           | 1       | 2 347      | 174       |  -         | no answer   |             | 2026-08-18 01:14 |
| devstral:24b         | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 1 774.6       | 52.4         | 128     | 4 734      | 797       | 6.228      |             |             | 2026-08-18 01:14 |
| devstral:24b         | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **1 097.4**   | **53.3**     | **128** | **2 512**  | **660**   | **5.157**  | **+1.7%**   | **+20.8%**  | 2026-08-18 01:14 |
| devstral:24b         | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 1 767.8       | 52.0         | 128     | 4 709      | 810       | 6.328      |             |             | 2026-08-18 01:14 |
| devstral:24b         | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 271.4**   | **52.9**     | **128** | **2 513**  | **673**   | **5.257**  | **+1.6%**   | **+20.4%**  | 2026-08-18 01:14 |
| devstral:24b         | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 459.1         | 26.5         | 128     | 6 244      |  -        |  -         |             |             | 2026-09-01 11:13 |
| devstral:24b         | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **408.3**     | **49.3**     | **128** | **2 598**  | ** - **   | ** - **    | **+86.2%**  |             | 2026-09-01 11:13 |
| devstral:24b         | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 459.0         | 36.3         | 128     | 6 288      |  -        |  -         |             |             | 2026-09-01 09:34 |
| devstral:24b         | 4096   | short  | stream     | GPU    | vLLM 0.22.0     |  -            | 50.6         | 128     | 2 541      | 580       | 4.534      |             |             | 2026-08-10 07:40 |
| devstral:24b         | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **326.9**     | **51.0**     | **128** | **2 611**  | ** - **   | ** - **    | **+0.6%**   |             | 2026-09-01 09:34 |
| falcon3:latest       | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 54 289.6      | 270.2        | 128     | 2 167      | 193       | 1.506      |             |             | 2026-08-18 01:14 |
| falcon3:latest       | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 54 055.3      | 267.1        | 128     | 2 283      | 198       | 1.546      |             |             | 2026-08-18 01:14 |
| falcon3:latest       | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 8 134.7       | 276.5        | 128     | 2 181      | 194       | 1.513      |             |             | 2026-08-18 01:14 |
| falcon3:latest       | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **8 007.8**   | **411.3**    | **128** | **360**    | **75**    | **0.583**  | **+48.8%**  | **+159.3%** | 2026-08-18 01:14 |
| falcon3:latest       | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 8 244.8       | 275.4        | 128     | 2 219      | 201       | 1.574      |             |             | 2026-08-18 01:14 |
| falcon3:latest       | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **6 153.3**   | **391.4**    | **128** | **372**    | **79**    | **0.617**  | **+42.1%**  | **+155.3%** | 2026-08-18 01:14 |
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
| llama3.2:1b          | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 8 892.5       | 310.7        | 128     | 2 199      | 191       | 1.495      |             |             | 2026-08-18 01:14 |
| llama3.2:1b          | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **6 384.6**   | **454.5**    | **128** | **327**    | **60**    | **0.467**  | **+46.3%**  | **+220.1%** | 2026-08-18 01:14 |
| llama3.2:1b          | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 8 861.9       | 308.9        | 128     | 2 195      | 197       | 1.536      |             |             | 2026-08-18 01:14 |
| llama3.2:1b          | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **3 701.6**   | **449.6**    | **128** | **336**    | **55**    | **0.427**  | **+45.5%**  | **+259.7%** | 2026-08-18 01:14 |
| llama3.2:1b          | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 3 146.6       | 302.6        | 128     | 2 396      | 203       | 1.585      |             |             | 2026-08-18 01:14 |
| llama3.2:1b          | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **1 518.0**   | **452.0**    | **128** | **332**    | **52**    | **0.408**  | **+49.4%**  | **+288.9%** | 2026-08-18 01:14 |
| llama3.2:1b          | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 538.1         | 16.6         | 128     | 9 415      | 357       | 2.791      |             |             | 2026-08-18 01:14 |
| llama3.2:1b          | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **327.6**     | **15.9**     | **128** | **8 239**  | **414**   | **3.235**  | -3.8%       | -13.7%      | 2026-08-18 01:14 |
| llama3.2:1b          | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 2 400.0       | 232.2        | 128     | 2 157      |  -        |  -         |             |             | 2026-09-01 10:01 |
| llama3.2:1b          | 4096   | short  | stream     | GPU    | vLLM 0.22.0     |  -            | 292.6        | 128     | 446        | 88        | 0.685      |             |             | 2026-08-10 07:11 |
| llama3.2:1b          | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **737.8**     | **451.0**    | **128** | **344**    | ** - **   | ** - **    | **+54.1%**  |             | 2026-09-01 10:01 |
| magistral:latest     | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 7 722.8       |  -           | 1       | 2 377      | 180       |  -         | no answer   |             | 2026-08-18 01:14 |
| magistral:latest     | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 1 774.3       | 52.4         | 128     | 4 760      | 823       | 6.430      |             |             | 2026-08-18 01:14 |
| magistral:latest     | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **1 110.6**   | **53.8**     | **128** | **2 549**  | **661**   | **5.165**  | **+2.8%**   | **+24.5%**  | 2026-08-18 01:14 |
| magistral:latest     | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 1 769.2       | 51.9         | 128     | 4 744      | 822       | 6.420      |             |             | 2026-08-18 01:14 |
| magistral:latest     | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 064.8**   | **53.3**     | **128** | **2 546**  | **656**   | **5.129**  | **+2.7%**   | **+25.2%**  | 2026-08-18 01:14 |
| magistral:latest     | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 636.5         | 52.6         | 128     | 4 801      | 814       | 6.362      |             |             | 2026-08-18 01:14 |
| magistral:latest     | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **477.4**     | **52.6**     | **128** | **2 583**  | **677**   | **5.288**  | **+0.1%**   | **+20.3%**  | 2026-08-18 01:14 |
| magistral:latest     | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 427.9         | 36.3         | 128     | 6 343      |  -        |  -         |             |             | 2026-09-01 10:03 |
| magistral:latest     | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **322.3**     | **51.3**     | **128** | **2 605**  | ** - **   | ** - **    | **+41.5%**  |             | 2026-09-01 10:03 |
| mistral-nemo:latest  | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 15 103.7      | 96.6         | 128     | 3 214      | 428       | 3.340      |             |             | 2026-08-18 01:14 |
| mistral-nemo:latest  | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **20 996.2**  | **107.2**    | **128** | **1 270**  | **296**   | **2.312**  | **+11.0%**  | **+44.5%**  | 2026-08-18 01:14 |
| mistral-nemo:latest  | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 15 153.6      | 95.9         | 128     | 3 306      | 428       | 3.345      |             |             | 2026-08-18 01:14 |
| mistral-nemo:latest  | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **20 034.2**  | **106.3**    | **128** | **1 300**  | **305**   | **2.379**  | **+10.8%**  | **+40.6%**  | 2026-08-18 01:14 |
| mistral-nemo:latest  | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 3 007.5       | 95.2         | 128     | 3 215      | 429       | 3.353      |             |             | 2026-08-18 01:14 |
| mistral-nemo:latest  | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **3 071.3**   | **103.9**    | **128** | **1 294**  | **298**   | **2.329**  | **+9.2%**   | **+43.9%**  | 2026-08-18 01:14 |
| mistral-nemo:latest  | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 2 933.6       | 94.6         | 128     | 3 232      | 426       | 3.328      |             |             | 2026-08-18 01:14 |
| mistral-nemo:latest  | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **3 755.7**   | **103.1**    | **128** | **1 303**  | **299**   | **2.338**  | **+9.0%**   | **+42.3%**  | 2026-08-18 01:14 |
| mistral-nemo:latest  | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 1 062.6       | 96.5         | 128     | 3 231      | 422       | 3.300      |             |             | 2026-08-18 01:14 |
| mistral-nemo:latest  | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **682.6**     | **104.1**    | **128** | **1 298**  | **297**   | **2.320**  | **+7.9%**   | **+42.2%**  | 2026-08-18 01:14 |
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
| devstral-small-2:latest | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 670.4         | 19.6         | 128     | 9 382     | 1 036   | 8.091     |             |            | 2026-08-18 01:14 |
| devstral-small-2:latest | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **942.2**     | **59.4**     | **128** | **2 350** | **696** | **5.440** | **+202.4%** | **+48.7%** | 2026-08-18 01:14 |
| devstral-small-2:latest | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 665.6         | 19.4         | 128     | 9 412     | 1 034   | 8.076     |             |            | 2026-08-18 01:14 |
| devstral-small-2:latest | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **1 305.3**   | **58.7**     | **128** | **2 360** | **705** | **5.504** | **+202.9%** | **+46.7%** | 2026-08-18 01:14 |
| devstral-small-2:latest | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 403.0         | 25.6         | 128     | 6 571     |  -      |  -        |             |            | 2026-09-01 11:00 |
| devstral-small-2:latest | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **301.3**     | **54.8**     | **128** | **2 368** | ** - ** | ** - **   | **+113.9%** |            | 2026-09-01 11:00 |
| devstral-small-2:latest | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 410.3         | 35.4         | 128     | 6 556     |  -      |  -        |             |            | 2026-09-01 09:22 |
| devstral-small-2:latest | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **281.2**     | **59.5**     | **128** | **2 399** | ** - ** | ** - **   | **+67.8%**  |            | 2026-09-01 09:22 |
| mistral-small3.2:latest | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 3 444.2       | 19.4         | 128     | 9 542     | 1 058   | 8.265     |             |            | 2026-08-18 01:14 |
| mistral-small3.2:latest | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **973.4**     | **57.8**     | **128** | **2 574** | **752** | **5.871** | **+198.2%** | **+40.8%** | 2026-08-18 01:14 |
| mistral-small3.2:latest | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 3 504.3       | 19.8         | 128     | 9 276     | 1 023   | 7.995     |             |            | 2026-08-18 01:14 |
| mistral-small3.2:latest | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **972.7**     | **57.1**     | **128** | **2 583** | **753** | **5.881** | **+188.9%** | **+35.9%** | 2026-08-18 01:14 |
| mistral-small3.2:latest | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 677.9         | 19.8         | 128     | 9 380     | 1 022   | 7.981     |             |            | 2026-08-18 01:14 |
| mistral-small3.2:latest | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **927.0**     | **59.2**     | **128** | **2 367** | **702** | **5.484** | **+198.4%** | **+45.5%** | 2026-08-18 01:14 |
| mistral-small3.2:latest | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 680.2         | 19.8         | 128     | 9 285     | 1 021   | 7.979     |             |            | 2026-08-18 01:14 |
| mistral-small3.2:latest | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **839.5**     | **58.8**     | **128** | **2 370** | **698** | **5.454** | **+196.7%** | **+46.3%** | 2026-08-18 01:14 |
| mistral-small3.2:latest | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 250.7         | 20.0         | 128     | 9 363     | 1 018   | 7.956     |             |            | 2026-08-18 01:14 |
| mistral-small3.2:latest | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **558.6**     | **59.7**     | **128** | **2 343** | **688** | **5.378** | **+198.9%** | **+47.9%** | 2026-08-18 01:14 |
| mistral-small3.2:latest | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 434.8         | 36.3         | 128     | 6 459     |  -      |  -        |             |            | 2026-09-01 10:07 |
| mistral-small3.2:latest | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **300.5**     | **59.4**     | **128** | **2 388** | ** - ** | ** - **   | **+63.9%**  |            | 2026-09-01 10:07 |

### nemotron_h_moe

![nemotron_h_moe](img/family-nemotron_h_moe.svg)

| Model                  | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token   | Δ decode  | Δ energy    | Date             |
|------------------------|------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|-----------|-----------|-------------|------------------|
| nemotron-3-nano:latest | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 3 236.5       | 134.7        | 128     | 4 813     | 452     | 3.530     |           |             | 2026-08-18 01:14 |
| nemotron-3-nano:latest | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **424 483.6** | **118.2**    | **128** | **1 604** | **244** | **1.908** | -12.2%    | **+85.1%**  | 2026-08-18 01:14 |
| nemotron-3-nano:latest | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 3 239.8       | 133.6        | 128     | 4 810     | 444     | 3.471     |           |             | 2026-08-18 01:14 |
| nemotron-3-nano:latest | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **218 674.2** | **119.1**    | **128** | **1 532** | **238** | **1.858** | -10.9%    | **+86.8%**  | 2026-08-18 01:14 |
| nemotron-3-nano:latest | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 600.2         | 136.0        | 128     | 4 767     | 443     | 3.460     |           |             | 2026-08-18 01:14 |
| nemotron-3-nano:latest | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **68 273.2**  | **122.7**    | **128** | **1 415** | **202** | **1.575** | -9.8%     | **+119.7%** | 2026-08-18 01:14 |
| nemotron-3-nano:latest | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 607.0         | 135.2        | 128     | 4 763     | 441     | 3.443     |           |             | 2026-08-18 01:14 |
| nemotron-3-nano:latest | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **57 503.3**  | **119.4**    | **128** | **1 432** | **210** | **1.639** | -11.7%    | **+110.0%** | 2026-08-18 01:14 |
| nemotron-3-nano:latest | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 209.3         | 134.7        | 128     | 4 684     | 435     | 3.401     |           |             | 2026-08-18 01:14 |
| nemotron-3-nano:latest | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **20 834.4**  | **121.5**    | **128** | **1 347** | **202** | **1.576** | -9.8%     | **+115.8%** | 2026-08-18 01:14 |
| nemotron-3-nano:latest | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 207.2         | 133.2        | 128     | 4 724     |  -      |  -        |           |             | 2026-09-01 10:10 |
| nemotron-3-nano:latest | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **28 150.1**  | **145.0**    | **128** | **1 204** | ** - ** | ** - **   | **+8.9%** |             | 2026-09-01 10:10 |

### olmo2

![olmo2](img/family-olmo2.svg)

| Model    | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode   | Δ energy   | Date             |
|----------|------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|------------|------------|------------------|
| olmo2:7b | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 20 028.7      | 136.4        | 128     | 2 840      | 354     | 2.767     |            |            | 2026-08-18 01:14 |
| olmo2:7b | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **34 901.7**  | **116.7**    | **128** | **1 169**  | **280** | **2.190** | -14.4%     | **+26.3%** | 2026-08-18 01:14 |
| olmo2:7b | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 19 953.2      | 134.6        | 128     | 2 898      | 356     | 2.782     |            |            | 2026-08-18 01:14 |
| olmo2:7b | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **28 548.3**  | **115.6**    | **128** | **1 168**  | **271** | **2.118** | -14.1%     | **+31.3%** | 2026-08-18 01:14 |
| olmo2:7b | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 4 137.9       | 139.9        | 128     | 2 824      | 345     | 2.692     |            |            | 2026-08-18 01:14 |
| olmo2:7b | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **3 314.2**   | **133.7**    | **128** | **1 030**  | **244** | **1.908** | -4.4%      | **+41.1%** | 2026-08-18 01:14 |
| olmo2:7b | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 4 119.3       | 138.5        | 128     | 2 871      | 356     | 2.779     |            |            | 2026-08-18 01:14 |
| olmo2:7b | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **5 447.3**   | **132.5**    | **128** | **1 023**  | **247** | **1.931** | -4.4%      | **+43.9%** | 2026-08-18 01:14 |
| olmo2:7b | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 1 451.9       | 140.3        | 128     | 2 828      | 348     | 2.721     |            |            | 2026-08-18 01:14 |
| olmo2:7b | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **1 822.5**   | **137.0**    | **128** | **991**    | **226** | **1.762** | -2.4%      | **+54.5%** | 2026-08-18 01:14 |
| olmo2:7b | 4096 | short  | stream     | CPU    | Ollama 0.32.6   | 202.1         | 5.2          | 128     | 27 396     | 978     | 7.638     |            |            | 2026-08-18 01:14 |
| olmo2:7b | 4096 | short  | stream     | CPU    | **loken 0.1.0** | **60.4**      | **6.7**      | **91**  | **15 762** | **508** | **5.606** |            |            | 2026-08-18 01:14 |
| olmo2:7b | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 1 026.8       | 100.6        | 128     | 1 450      |  -      |  -        |            |            | 2026-09-01 10:11 |
| olmo2:7b | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **704.3**     | **133.9**    | **128** | **1 032**  | ** - ** | ** - **   | **+33.2%** |            | 2026-09-01 10:11 |

### olmoe

![olmoe](img/family-olmoe.svg)

| Model        | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token   | Δ decode   | Δ energy   | Date             |
|--------------|------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|-----------|------------|------------|------------------|
| olmoe:latest | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 82 498.5      | 483.8        | 128     | 2 013     | 182     | 1.424     |            |            | 2026-08-18 01:14 |
| olmoe:latest | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **236.6**     | **441.1**    | **128** | **3 105** | **227** | **1.777** | -8.8%      | -19.9%     | 2026-08-18 01:14 |
| olmoe:latest | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 82 280.0      | 479.1        | 128     | 2 012     | 178     | 1.390     |            |            | 2026-08-18 01:14 |
| olmoe:latest | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **230.8**     | **417.1**    | **128** | **3 079** | **235** | **1.837** | -12.9%     | -24.3%     | 2026-08-18 01:14 |
| olmoe:latest | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 13 403.6      | 495.6        | 128     | 2 033     | 171     | 1.338     |            |            | 2026-08-18 01:14 |
| olmoe:latest | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **115.3**     | **471.2**    | **127** | **2 793** | **200** | **1.569** | -4.9%      | -14.3%     | 2026-08-18 01:14 |
| olmoe:latest | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 13 295.7      | 498.2        | 128     | 374       |  -      |  -        |            |            | 2026-08-27 20:06 |
| olmoe:latest | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **1 275.9**   | **441.4**    | **122** | **319**   | ** - ** | ** - **   |            |            | 2026-08-27 20:06 |
| olmoe:latest | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 4 586.5       | 496.2        | 128     | 1 980     | 174     | 1.362     |            |            | 2026-08-18 01:14 |
| olmoe:latest | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **48.0**      | **475.3**    | **128** | **1 663** | **137** | **1.067** | -4.2%      | **+27.7%** | 2026-08-18 01:14 |
| olmoe:latest | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 3 628.7       | 389.0        | 128     | 422       |  -      |  -        |            |            | 2026-09-01 10:12 |
| olmoe:latest | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **361.6**     | **440.7**    | **128** | **392**   | ** - ** | ** - **   | **+13.3%** |            | 2026-09-01 10:12 |

### phi2

![phi2](img/family-phi2.svg)

| Model            | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms  | J/req   | J/token   | Δ decode   | Δ energy    | Date             |
|------------------|------|--------|------------|--------|-----------------|---------------|--------------|---------|---------|---------|-----------|------------|-------------|------------------|
| moondream:latest | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 35 453.0      |  -           | 1       | 1 614   | 119     |  -        | no answer  |             | 2026-08-18 01:14 |
| moondream:latest | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 13 143.8      | 471.2        | 128     | 1 923   | 166     | 1.295     |            |             | 2026-08-18 01:14 |
| moondream:latest | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **4 265.2**   | **624.0**    | **128** | **270** | **46**  | **0.360** | **+32.4%** | **+259.6%** | 2026-08-18 01:14 |
| moondream:latest | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 12 904.2      | 456.7        | 128     | 1 951   | 173     | 1.350     |            |             | 2026-08-18 01:14 |
| moondream:latest | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **1 200.5**   | **585.3**    | **128** | **292** | **50**  | **0.392** | **+28.1%** | **+244.2%** | 2026-08-18 01:14 |
| moondream:latest | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 4 379.8       | 469.2        | 128     | 1 917   | 173     | 1.348     |            |             | 2026-08-18 01:14 |
| moondream:latest | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **1 840.9**   | **617.9**    | **128** | **261** | **50**  | **0.387** | **+31.7%** | **+248.4%** | 2026-08-18 01:14 |
| moondream:latest | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 3 604.0       | 370.9        | 128     | 447     |  -      |  -        |            |             | 2026-09-01 10:08 |
| moondream:latest | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **1 647.6**   | **598.1**    | **128** | **245** | ** - ** | ** - **   | **+61.2%** |             | 2026-09-01 10:08 |

### qwen2

![qwen2](img/family-qwen2.svg)

| Model            | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req     | J/token    | Δ decode     | Δ energy    | Date             |
|------------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|-----------|------------|--------------|-------------|------------------|
| deepcoder:14b    | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 11 443.0      | 75.5         | 128     | 3 630      | 563       | 4.395      |              |             | 2026-08-18 01:14 |
| deepcoder:14b    | 4096   | long   | non-stream | GPU    | vLLM 0.22.0     |  -            | 81.9         | 128     | 1 566      | 335       | 2.621      |              |             | 2026-08-18 01:14 |
| deepcoder:14b    | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **13 046.1**  | **77.9**     | **128** | **1 716**  | **436**   | **3.408**  | -4.9%        | -23.1%      | 2026-08-18 01:14 |
| deepcoder:14b    | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 11 553.8      | 75.0         | 128     | 3 664      | 563       | 4.400      |              |             | 2026-08-18 01:14 |
| deepcoder:14b    | 4096   | long   | stream     | GPU    | vLLM 0.22.0     |  -            | 84.9         | 128     | 1 571      | 346       | 2.706      |              |             | 2026-08-18 01:14 |
| deepcoder:14b    | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **12 567.4**  | **77.2**     | **128** | **1 725**  | **434**   | **3.387**  | -9.2%        | -20.1%      | 2026-08-18 01:14 |
| deepcoder:14b    | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 2 388.2       | 75.0         | 128     | 3 631      | 546       | 4.269      |              |             | 2026-08-18 01:14 |
| deepcoder:14b    | 4096   | medium | non-stream | GPU    | vLLM 0.22.0     |  -            | 81.0         | 128     | 1 584      | 333       | 2.605      |              |             | 2026-08-18 01:14 |
| deepcoder:14b    | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **1 854.1**   | **82.6**     | **128** | **1 611**  | **404**   | **3.153**  | **+2.0%**    | -17.4%      | 2026-08-18 01:14 |
| deepcoder:14b    | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 2 394.6       | 74.5         | 128     | 3 642      | 548       | 4.278      |              |             | 2026-08-18 01:14 |
| deepcoder:14b    | 4096   | medium | stream     | GPU    | vLLM 0.22.0     |  -            | 85.0         | 128     | 1 569      | 339       | 2.648      |              |             | 2026-08-18 01:14 |
| deepcoder:14b    | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **2 462.6**   | **81.7**     | **128** | **1 615**  | **398**   | **3.113**  | -3.9%        | -14.9%      | 2026-08-18 01:14 |
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
| deepseek-r1:32b  | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 4 224.7       | 25.7         | 128     | 8 263      | 1 318     | 10.293     |              |             | 2026-08-18 01:14 |
| deepseek-r1:32b  | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **591.0**     | **39.3**     | **128** | **3 881**  | **1 087** | **8.491**  | **+52.9%**   | **+21.2%**  | 2026-08-18 01:14 |
| deepseek-r1:32b  | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 898.0         | 26.1         | 128     | 8 053      | 1 281     | 10.008     |              |             | 2026-08-18 01:14 |
| deepseek-r1:32b  | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **506.0**     | **40.8**     | **128** | **3 424**  | **993**   | **7.760**  | **+56.6%**   | **+29.0%**  | 2026-08-18 01:14 |
| deepseek-r1:32b  | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 900.4         | 25.9         | 128     | 8 122      | 1 293     | 10.099     |              |             | 2026-08-18 01:14 |
| deepseek-r1:32b  | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **524.6**     | **40.5**     | **128** | **3 439**  | **991**   | **7.741**  | **+56.4%**   | **+30.5%**  | 2026-08-18 01:14 |
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
| qwen2.5:0.5b     | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 9 940.0       | 368.1        | 128     | 2 031      | 154       | 1.204      |              |             | 2026-08-18 01:14 |
| qwen2.5:0.5b     | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **3 510.8**   | **378.6**    | **128** | **392**    | **47**    | **0.366**  | **+2.9%**    | **+229.1%** | 2026-08-18 01:14 |
| qwen2.5:0.5b     | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 10 082.4      | 371.9        | 128     | 2 038      | 153       | 1.195      |              |             | 2026-08-18 01:14 |
| qwen2.5:0.5b     | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **2 512.4**   | **387.1**    | **128** | **383**    | **51**    | **0.397**  | **+4.1%**    | **+200.8%** | 2026-08-18 01:14 |
| qwen2.5:0.5b     | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 3 355.3       | 363.4        | 128     | 2 059      | 163       | 1.270      |              |             | 2026-08-18 01:14 |
| qwen2.5:0.5b     | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **1 192.4**   | **371.2**    | **128** | **386**    | **56**    | **0.438**  | **+2.2%**    | **+190.1%** | 2026-08-18 01:14 |
| qwen2.5:0.5b     | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 1 594.1       | 49.7         | 35      | 2 181      | 121       | 3.433      |              |             | 2026-08-18 01:14 |
| qwen2.5:0.5b     | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **769.5**     | **49.1**     | **42**  | **989**    | **63**    | **1.489**  |              |             | 2026-08-18 01:14 |
| qwen2.5:0.5b     | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 2 829.9       | 303.6        | 128     | 1 923      |  -        |  -         |              |             | 2026-09-01 08:57 |
| qwen2.5:0.5b     | 4096   | short  | stream     | GPU    | vLLM 0.22.0     |  -            | 502.5        | 128     | 273        | 51        | 0.398      |              |             | 2026-08-10 07:11 |
| qwen2.5:0.5b     | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **784.7**     | **366.6**    | **128** | **415**    | ** - **   | ** - **    | -27.0%       |             | 2026-09-01 08:57 |

### qwen3

![qwen3](img/family-qwen3.svg)

| Model        | Ctx   | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode   | Δ energy    | Date             |
|--------------|-------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|------------|-------------|------------------|
| qwen3:0.6b   | 4096  | long   | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent |             | 2026-08-18 01:14 |
| qwen3:0.6b   | 4096  | long   | non-stream | GPU    | **loken 0.1.0** | **26 432.8**  | **652.0**    | **128** | **262**    | **37**  | **0.289** |            |             | 2026-08-18 01:14 |
| qwen3:0.6b   | 4096  | long   | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent |             | 2026-08-18 01:14 |
| qwen3:0.6b   | 4096  | long   | stream     | GPU    | **loken 0.1.0** | **24 191.2**  | **649.5**    | **128** | **268**    | **35**  | **0.276** |            |             | 2026-08-18 01:14 |
| qwen3:0.6b   | 4096  | medium | non-stream | GPU    | Ollama 0.32.6   | 15 632.2      | 600.7        | 128     | 1 882      | 145     | 1.133     |            |             | 2026-08-18 01:14 |
| qwen3:0.6b   | 4096  | medium | non-stream | GPU    | **loken 0.1.0** | **3 528.5**   | **685.0**    | **128** | **269**    | **37**  | **0.289** | **+14.0%** | **+292.0%** | 2026-08-18 01:14 |
| qwen3:0.6b   | 4096  | medium | stream     | GPU    | Ollama 0.32.6   | 15 411.6      | 594.4        | 128     | 1 963      | 156     | 1.219     |            |             | 2026-08-18 01:14 |
| qwen3:0.6b   | 4096  | medium | stream     | GPU    | **loken 0.1.0** | **3 864.7**   | **679.1**    | **128** | **260**    | **40**  | **0.316** | **+14.2%** | **+285.6%** | 2026-08-18 01:14 |
| qwen3:0.6b   | 4096  | short  | non-stream | GPU    | Ollama 0.32.6   | 5 523.2       | 598.6        | 128     | 1 898      | 152     | 1.188     |            |             | 2026-08-18 01:14 |
| qwen3:0.6b   | 4096  | short  | non-stream | GPU    | **loken 0.1.0** | **1 340.7**   | **677.2**    | **128** | **251**    | **37**  | **0.288** | **+13.1%** | **+312.9%** | 2026-08-18 01:14 |
| qwen3:0.6b   | 4096  | short  | stream     | CPU    | Ollama 0.32.6   | 1 154.6       | 45.6         | 128     | 4 286      | 186     | 1.451     |            |             | 2026-08-18 01:14 |
| qwen3:0.6b   | 4096  | short  | stream     | CPU    | **loken 0.1.0** | **178.7**     | **46.8**     | **128** | **2 904**  | **178** | **1.393** | **+2.7%**  | **+4.1%**   | 2026-08-18 01:14 |
| qwen3:0.6b   | 4096  | short  | stream     | GPU    | Ollama 0.32.6   | 4 326.7       | 469.9        | 128     | 1 739      |  -      |  -        |            |             | 2026-09-01 10:26 |
| qwen3:0.6b   | 4096  | short  | stream     | GPU    | vLLM 0.22.0     |  -            | 473.0        | 128     | 280        | 55      | 0.430     |            |             | 2026-08-10 07:10 |
| qwen3:0.6b   | 4096  | short  | stream     | GPU    | **loken 0.1.0** | **663.3**     | **708.9**    | **128** | **262**    | ** - ** | ** - **   | **+49.9%** |             | 2026-09-01 10:26 |
| qwen3:0.6b   | 16384 | long   | stream     | GPU    | Ollama 0.32.6   | 65 512.2      | 599.1        | 128     | 2 021      | 154     | 1.201     |            |             | 2026-08-18 01:14 |
| qwen3:0.6b   | 16384 | long   | stream     | GPU    | **loken 0.1.0** | **16 523.9**  | **646.5**    | **128** | **270**    | **48**  | **0.378** | **+7.9%**  | **+217.4%** | 2026-08-18 01:14 |
| qwen3:0.6b   | 16384 | short  | stream     | GPU    | Ollama 0.32.6   | 11 388.5      | 528.7        | 93      | 1 835      | 441     | 4.741     |            |             | 2026-08-18 01:14 |
| qwen3:0.6b   | 16384 | short  | stream     | GPU    | **loken 0.1.0** | **2 105.9**   | **670.3**    | **111** | **249**    | **32**  | **0.285** |            |             | 2026-08-18 01:14 |
| qwen3:0.6b   | 32768 | long   | stream     | GPU    | Ollama 0.32.6   | 68 185.4      | 604.4        | 128     | 2 023      | 153     | 1.196     |            |             | 2026-08-18 01:14 |
| qwen3:0.6b   | 32768 | long   | stream     | GPU    | **loken 0.1.0** | **16 473.2**  | **651.9**    | **128** | **258**    | **47**  | **0.364** | **+7.9%**  | **+228.9%** | 2026-08-18 01:14 |
| qwen3:1.7b   | 4096  | short  | stream     | GPU    | Ollama 0.32.6   | 3 843.4       | 407.7        | 128     | 534        |  -      |  -        |            |             | 2026-08-27 14:34 |
| qwen3:1.7b   | 4096  | short  | stream     | GPU    | **loken 0.1.0** | **524.1**     | **427.8**    | **128** | **375**    | ** - ** | ** - **   | **+4.9%**  |             | 2026-08-27 14:34 |
| qwen3:8b     | 4096  | long   | non-stream | GPU    | Ollama 0.32.6   | 19 399.7      | 138.0        | 128     | 2 995      | 385     | 3.010     |            |             | 2026-08-18 01:14 |
| qwen3:8b     | 4096  | long   | non-stream | GPU    | vLLM 0.22.0     |  -            | 132.5        | 128     | 970        | 197     | 1.542     |            |             | 2026-08-18 01:14 |
| qwen3:8b     | 4096  | long   | non-stream | GPU    | **loken 0.1.0** | **20 485.7**  | **144.4**    | **128** | **955**    | **236** | **1.847** | **+4.7%**  | -16.5%      | 2026-08-18 01:14 |
| qwen3:8b     | 4096  | long   | stream     | GPU    | Ollama 0.32.6   | 19 559.7      | 136.7        | 128     | 3 031      | 373     | 2.915     |            |             | 2026-08-18 01:14 |
| qwen3:8b     | 4096  | long   | stream     | GPU    | vLLM 0.22.0     |  -            | 139.4        | 128     | 971        | 195     | 1.525     |            |             | 2026-08-18 01:14 |
| qwen3:8b     | 4096  | long   | stream     | GPU    | **loken 0.1.0** | **22 262.6**  | **143.0**    | **128** | **965**    | **234** | **1.830** | **+2.6%**  | -16.7%      | 2026-08-18 01:14 |
| qwen3:8b     | 4096  | medium | non-stream | GPU    | Ollama 0.32.6   | 4 006.3       | 139.5        | 128     | 3 004      | 369     | 2.886     |            |             | 2026-08-18 01:14 |
| qwen3:8b     | 4096  | medium | non-stream | GPU    | vLLM 0.22.0     |  -            | 133.0        | 128     | 966        | 200     | 1.559     |            |             | 2026-08-18 01:14 |
| qwen3:8b     | 4096  | medium | non-stream | GPU    | **loken 0.1.0** | **4 068.9**   | **139.4**    | **128** | **989**    | **239** | **1.868** | -0.1%      | -16.5%      | 2026-08-18 01:14 |
| qwen3:8b     | 4096  | medium | stream     | GPU    | Ollama 0.32.6   | 4 012.6       | 139.1        | 128     | 2 922      | 366     | 2.858     |            |             | 2026-08-18 01:14 |
| qwen3:8b     | 4096  | medium | stream     | GPU    | vLLM 0.22.0     |  -            | 139.7        | 128     | 1 012      | 204     | 1.592     |            |             | 2026-08-18 01:14 |
| qwen3:8b     | 4096  | medium | stream     | GPU    | **loken 0.1.0** | **3 845.2**   | **138.4**    | **128** | **984**    | **240** | **1.879** | -1.0%      | -15.3%      | 2026-08-18 01:14 |
| qwen3:8b     | 4096  | short  | non-stream | GPU    | Ollama 0.32.6   | 1 444.7       | 140.2        | 128     | 2 904      | 369     | 2.880     |            |             | 2026-08-18 01:14 |
| qwen3:8b     | 4096  | short  | non-stream | GPU    | vLLM 0.22.0     |  -            | 133.1        | 128     | 964        | 201     | 1.574     |            |             | 2026-08-18 01:14 |
| qwen3:8b     | 4096  | short  | non-stream | GPU    | **loken 0.1.0** | **812.6**     | **145.9**    | **128** | **942**    | **232** | **1.812** | **+4.1%**  | -13.2%      | 2026-08-18 01:14 |
| qwen3:8b     | 4096  | short  | stream     | CPU    | Ollama 0.32.6   | 111.7         | 4.6          | 128     | 30 746     | 1 045   | 8.166     |            |             | 2026-08-18 01:14 |
| qwen3:8b     | 4096  | short  | stream     | CPU    | **loken 0.1.0** | **71.4**      | **5.0**      | **128** | **26 755** | **875** | **6.836** | **+8.9%**  | **+19.4%**  | 2026-08-18 01:14 |
| qwen3:8b     | 4096  | short  | stream     | GPU    | Ollama 0.32.6   | 1 027.4       | 96.7         | 128     | 3 221      |  -      |  -        |            |             | 2026-09-01 10:30 |
| qwen3:8b     | 4096  | short  | stream     | GPU    | vLLM 0.22.0     |  -            | 139.6        | 128     | 12 064     |  -      |  -        |            |             | 2026-09-01 10:30 |
| qwen3:8b     | 4096  | short  | stream     | GPU    | **loken 0.1.0** | **542.7**     | **145.4**    | **128** | **954**    | ** - ** | ** - **   | **+4.2%**  |             | 2026-09-01 10:30 |
| qwen3:8b     | 16384 | long   | stream     | GPU    | Ollama 0.32.6   | 20 813.5      | 136.3        | 128     | 3 127      | 368     | 2.871     |            |             | 2026-08-18 01:14 |
| qwen3:8b     | 16384 | long   | stream     | GPU    | **loken 0.1.0** | **21 874.2**  | **140.2**    | **128** | **981**    | **236** | **1.846** | **+2.8%**  | **+55.5%**  | 2026-08-18 01:14 |
| qwen3:8b     | 16384 | short  | stream     | GPU    | Ollama 0.32.6   | 2 904.4       | 138.9        | 128     | 2 991      | 354     | 2.765     |            |             | 2026-08-18 01:14 |
| qwen3:8b     | 16384 | short  | stream     | GPU    | **loken 0.1.0** | **1 724.6**   | **149.6**    | **128** | **936**    | **220** | **1.717** | **+7.7%**  | **+61.0%**  | 2026-08-18 01:14 |
| qwen3:8b     | 32768 | long   | stream     | GPU    | Ollama 0.32.6   | 20 850.4      | 135.9        | 128     | 3 119      | 364     | 2.844     |            |             | 2026-08-18 01:14 |
| qwen3:8b     | 32768 | long   | stream     | GPU    | **loken 0.1.0** | **17 655.2**  | **140.3**    | **128** | **981**    | **238** | **1.860** | **+3.3%**  | **+52.9%**  | 2026-08-18 01:14 |
| qwen3:latest | 4096  | short  | stream     | GPU    | **loken 0.1.0** | **2 024.3**   | **141.1**    | **128** | **1 024**  | **220** | **1.716** |            |             | 2026-08-18 01:14 |

### qwen35

![qwen35](img/family-qwen35.svg)

| Model          | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token   | Δ decode   | Δ energy   | Date             |
|----------------|------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|-----------|------------|------------|------------------|
| qwen3.5:latest | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 4 346.3       | 102.8        | 128     | 3 976     | 466     | 3.642     |            |            | 2026-08-18 01:14 |
| qwen3.5:latest | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **393 160.0** | **112.5**    | **128** | **1 317** | **287** | **2.242** | **+9.5%**  | **+62.5%** | 2026-08-18 01:14 |
| qwen3.5:latest | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 4 329.8       | 102.6        | 128     | 3 981     | 446     | 3.481     |            |            | 2026-08-18 01:14 |
| qwen3.5:latest | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **230 400.8** | **111.9**    | **128** | **1 331** | **287** | **2.243** | **+9.1%**  | **+55.2%** | 2026-08-18 01:14 |
| qwen3.5:latest | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 747.0         | 104.3        | 128     | 3 954     | 449     | 3.508     |            |            | 2026-08-18 01:14 |
| qwen3.5:latest | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **70 159.8**  | **120.6**    | **128** | **1 225** | **272** | **2.121** | **+15.6%** | **+65.4%** | 2026-08-18 01:14 |
| qwen3.5:latest | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 704.8         | 104.4        | 128     | 3 895     | 452     | 3.528     |            |            | 2026-08-18 01:14 |
| qwen3.5:latest | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **5 343.4**   | **119.6**    | **128** | **1 269** | **267** | **2.083** | **+14.5%** | **+69.4%** | 2026-08-18 01:14 |
| qwen3.5:latest | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 248.6         | 103.6        | 128     | 3 939     | 440     | 3.438     |            |            | 2026-08-18 01:14 |
| qwen3.5:latest | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **6 422.6**   | **120.8**    | **128** | **1 221** | **274** | **2.143** | **+16.7%** | **+60.4%** | 2026-08-18 01:14 |
| qwen3.5:latest | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 213.8         | 75.2         | 128     | 4 250     |  -      |  -        |            |            | 2026-09-01 10:25 |
| qwen3.5:latest | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **25 974.3**  | **126.1**    | **128** | **1 267** | ** - ** | ** - **   | **+67.6%** |            | 2026-09-01 10:25 |

### qwen35moe

![qwen35moe](img/family-qwen35moe.svg)

| Model       | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token   | Δ decode   | Δ energy    | Date             |
|-------------|------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|-----------|------------|-------------|------------------|
| qwen3.5:35b | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 2 740.5       | 115.2        | 128     | 22 116    | 1 517   | 11.849    |            |             | 2026-08-18 01:14 |
| qwen3.5:35b | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **356 035.2** | **87.7**     | **128** | **1 885** | **269** | **2.099** | -23.8%     | **+464.4%** | 2026-08-18 01:14 |
| qwen3.5:35b | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 2 726.2       | 114.9        | 128     | 22 474    | 1 444   | 11.282    |            |             | 2026-08-18 01:14 |
| qwen3.5:35b | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **212 543.3** | **90.0**     | **128** | **1 895** | **260** | **2.031** | -21.6%     | **+455.4%** | 2026-08-18 01:14 |
| qwen3.5:35b | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 475.8         | 112.1        | 128     | 22 151    | 1 426   | 11.141    |            |             | 2026-08-18 01:14 |
| qwen3.5:35b | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **52 192.0**  | **90.9**     | **128** | **1 736** | **231** | **1.806** | -18.9%     | **+517.0%** | 2026-08-18 01:14 |
| qwen3.5:35b | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 481.9         | 113.1        | 128     | 22 665    | 1 457   | 11.386    |            |             | 2026-08-18 01:14 |
| qwen3.5:35b | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **45 239.8**  | **91.8**     | **128** | **1 776** | **235** | **1.833** | -18.8%     | **+521.0%** | 2026-08-18 01:14 |
| qwen3.5:35b | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 170.2         | 115.3        | 128     | 11 772    | 860     | 6.716     |            |             | 2026-08-18 01:14 |
| qwen3.5:35b | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **12 007.7**  | **90.7**     | **128** | **1 719** | **227** | **1.777** | -21.4%     | **+277.9%** | 2026-08-18 01:14 |
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
| qwen3-coder:30b | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 4 119.2       | 145.5        | 128     | 4 160     | 401     | 3.132     |            |          | 2026-08-18 01:14 |
| qwen3-coder:30b | 4096 | medium | non-stream | GPU    | vLLM 0.22.0     |  -            | 122.4        | 128     | 1 047     | 178     | 1.389     |            |          | 2026-08-18 01:14 |
| qwen3-coder:30b | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **766.0**     | **144.6**    | **128** | **1 334** | **183** | **1.426** | -0.6%      | -2.6%    | 2026-08-18 01:14 |
| qwen3-coder:30b | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 4 154.1       | 146.9        | 128     | 4 004     | 380     | 2.970     |            |          | 2026-08-18 01:14 |
| qwen3-coder:30b | 4096 | medium | stream     | GPU    | vLLM 0.22.0     |  -            | 129.1        | 128     | 1 051     | 174     | 1.361     |            |          | 2026-08-18 01:14 |
| qwen3-coder:30b | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **626.3**     | **145.1**    | **128** | **1 329** | **186** | **1.452** | -1.2%      | -6.2%    | 2026-08-18 01:14 |
| qwen3-coder:30b | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 1 419.7       | 149.3        | 128     | 3 960     | 379     | 2.960     |            |          | 2026-08-18 01:14 |
| qwen3-coder:30b | 4096 | short  | non-stream | GPU    | vLLM 0.22.0     |  -            | 130.9        | 128     | 983       | 169     | 1.320     |            |          | 2026-08-18 01:14 |
| qwen3-coder:30b | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **173.8**     | **150.9**    | **128** | **1 297** | **179** | **1.401** | **+1.1%**  | -5.8%    | 2026-08-18 01:14 |
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
| qwen3-coder-next:latest | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 211.1         | 28.2         | 128     | 10 761      | 850       | 6.638      |          |          | 2026-08-18 01:14 |
| qwen3-coder-next:latest | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **312.9**     | **5.7**      | **128** | **91 632**  | **4 050** | **31.641** | -79.6%   | -79.0%   | 2026-08-18 01:14 |
| qwen3-coder-next:latest | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 215.4         | 28.3         | 128     | 10 669      | 847       | 6.619      |          |          | 2026-08-18 01:14 |
| qwen3-coder-next:latest | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **495.4**     | **5.9**      | **128** | **82 882**  | **3 809** | **29.761** | -79.2%   | -77.8%   | 2026-08-18 01:14 |
| qwen3-coder-next:latest | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 81.9          | 28.7         | 128     | 10 342      | 813       | 6.350      |          |          | 2026-08-18 01:14 |
| qwen3-coder-next:latest | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **194.3**     | **6.1**      | **128** | **69 677**  | **3 354** | **26.201** | -78.7%   | -75.8%   | 2026-08-18 01:14 |
| qwen3-coder-next:latest | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 80.1          | 28.1         | 128     | 10 488      |  -        |  -         |          |          | 2026-09-01 10:17 |
| qwen3-coder-next:latest | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **305.3**     | **9.2**      | **128** | **23 157**  | ** - **   | ** - **    | -67.1%   |          | 2026-09-01 10:17 |
| qwen3next:latest        | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 1 042.6       | 30.4         | 128     | 10 612      | 850       | 6.644      |          |          | 2026-08-18 01:14 |
| qwen3next:latest        | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **6 038.2**   | **6.4**      | **128** | **65 338**  | **3 086** | **24.112** | -79.1%   | -72.4%   | 2026-08-18 01:14 |
| qwen3next:latest        | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 1 136.1       | 30.2         | 128     | 10 706      | 854       | 6.674      |          |          | 2026-08-18 01:14 |
| qwen3next:latest        | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **5 519.9**   | **6.3**      | **128** | **68 790**  | **3 339** | **26.088** | -79.0%   | -74.4%   | 2026-08-18 01:14 |
| qwen3next:latest        | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 224.9         | 23.2         | 128     | 16 220      | 1 196     | 9.348      |          |          | 2026-08-18 01:14 |
| qwen3next:latest        | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **1 050.9**   | **6.4**      | **128** | **64 210**  | **3 113** | **24.319** | -72.6%   | -61.6%   | 2026-08-18 01:14 |
| qwen3next:latest        | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 230.0         | 33.1         | 128     | 9 879       | 788       | 6.153      |          |          | 2026-08-18 01:14 |
| qwen3next:latest        | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **856.1**     | **6.7**      | **128** | **67 547**  | **3 182** | **24.856** | -79.9%   | -75.2%   | 2026-08-18 01:14 |
| qwen3next:latest        | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 86.2          | 33.1         | 128     | 9 673       | 767       | 5.995      |          |          | 2026-08-18 01:14 |
| qwen3next:latest        | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **128.4**     | **6.8**      | **128** | **63 253**  | **3 047** | **23.805** | -79.5%   | -74.8%   | 2026-08-18 01:14 |
| qwen3next:latest        | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 86.4          | 32.9         | 128     | 9 659       |  -        |  -         |          |          | 2026-09-01 10:34 |
| qwen3next:latest        | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **187.8**     | **9.9**      | **128** | **20 982**  | ** - **   | ** - **    | -70.0%   |          | 2026-09-01 10:34 |

### smollm3

![smollm3](img/family-smollm3.svg)

| Model          | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode   | Δ energy    | Date             |
|----------------|------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|------------|-------------|------------------|
| smollm3:latest | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 28 723.2      | 220.5        | 128     | 2 528      | 252     | 1.965     |            |             | 2026-08-18 01:14 |
| smollm3:latest | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **29 576.9**  | **237.0**    | **128** | **649**    | **135** | **1.055** | **+7.5%**  | **+86.3%**  | 2026-08-18 01:14 |
| smollm3:latest | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 29 317.1      | 222.7        | 128     | 2 449      | 257     | 2.007     |            |             | 2026-08-18 01:14 |
| smollm3:latest | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **27 845.1**  | **233.8**    | **128** | **657**    | **133** | **1.037** | **+5.0%**  | **+93.5%**  | 2026-08-18 01:14 |
| smollm3:latest | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 6 320.1       | 220.5        | 128     | 2 496      | 249     | 1.946     |            |             | 2026-08-18 01:14 |
| smollm3:latest | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **5 361.0**   | **250.6**    | **128** | **616**    | **116** | **0.910** | **+13.7%** | **+113.9%** | 2026-08-18 01:14 |
| smollm3:latest | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 6 243.0       | 220.2        | 128     | 2 507      | 248     | 1.941     |            |             | 2026-08-18 01:14 |
| smollm3:latest | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **3 044.6**   | **246.9**    | **128** | **636**    | **123** | **0.965** | **+12.1%** | **+101.2%** | 2026-08-18 01:14 |
| smollm3:latest | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 2 227.1       | 225.9        | 128     | 2 451      | 246     | 1.918     |            |             | 2026-08-18 01:14 |
| smollm3:latest | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **358.0**     | **250.4**    | **128** | **639**    | **132** | **1.028** | **+10.8%** | **+86.5%**  | 2026-08-18 01:14 |
| smollm3:latest | 4096 | short  | stream     | CPU    | Ollama 0.32.6   | 693.1         | 11.0         | 128     | 13 711     | 504     | 3.935     |            |             | 2026-08-18 01:14 |
| smollm3:latest | 4096 | short  | stream     | CPU    | **loken 0.1.0** | **140.4**     | **10.6**     | **128** | **12 517** | **558** | **4.358** | -3.5%      | -9.7%       | 2026-08-18 01:14 |
| smollm3:latest | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 1 687.1       | 170.9        | 128     | 2 491      |  -      |  -        |            |             | 2026-09-01 10:36 |
| smollm3:latest | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **694.5**     | **256.7**    | **128** | **578**    | ** - ** | ** - **   | **+50.2%** |             | 2026-09-01 10:36 |
