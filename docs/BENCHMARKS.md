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
to first token is two to four times better here.

Nor at the long prompt, where it gets worse. At 245 tokens ollama reported 15 745 tok/s on
mistral-nemo against a time to first token implying 968, and 8 622 on mistral-small3.2 against
886 - a factor of ten to sixteen. This engine reported 6 464 against 2 967, and 1 134 against
1 038. Both cells read as an ollama win on the column and as a win here on the clock.

So the column is not a comparison at any prompt length measured. Read the TTFT column. The
prefill figures are each engine's own bookkeeping; only the times are measured by the bench.





























































































































































































### ernie4_5

![ernie4_5](img/family-ernie4_5.svg)

| Model           | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token   | Δ decode    | Δ energy | Date             |
|-----------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|-----------|-------------|----------|------------------|
| ernie4-5:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 74 060.6      | 213.1        | 128     | 1 591     |  -      |  -        |             |          | 2026-09-01 17:44 |
| ernie4-5:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **51 966.9**  | **375.0**    | **75**  | **208**   | ** - ** | ** - **   |             |          | 2026-09-01 17:44 |
| ernie4-5:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 73 484.7      | 406.5        | 128     | 1 585     |  -      |  -        |             |          | 2026-09-01 16:06 |
| ernie4-5:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **44 888.3**  | **570.5**    | **75**  | **210**   | ** - ** | ** - **   |             |          | 2026-09-01 16:06 |
| ernie4-5:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 11 680.2      | 207.7        | 80      | 1 462     |  -      |  -        |             |          | 2026-09-01 14:30 |
| ernie4-5:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **22 700.7**  | **483.5**    | **128** | **273**   | ** - ** | ** - **   |             |          | 2026-09-01 14:30 |
| ernie4-5:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 11 943.1      | 405.6        | 80      | 1 501     |  -      |  -        |             |          | 2026-09-01 12:52 |
| ernie4-5:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **9 633.0**   | **570.7**    | **128** | **287**   | ** - ** | ** - **   |             |          | 2026-09-01 12:52 |
| ernie4-5:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 4 151.8       | 226.2        | 128     | 1 588     |  -      |  -        |             |          | 2026-09-01 11:13 |
| ernie4-5:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **6 060.6**   | **496.0**    | **128** | **264**   | ** - ** | ** - **   | **+119.3%** |          | 2026-09-01 11:13 |
| ernie4-5:latest | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 3 766.7       | 52.7         | 128     | 3 614     | 165     | 1.286     |             |          | 2026-08-18 01:14 |
| ernie4-5:latest | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **3 228.9**   | **50.2**     | **104** | **2 180** | **95**  | **0.915** |             |          | 2026-08-18 01:14 |
| ernie4-5:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 4 129.1       | 418.9        | 128     | 1 578     |  -      |  -        |             |          | 2026-09-01 09:35 |
| ernie4-5:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **2 840.4**   | **600.1**    | **128** | **267**   | ** - ** | ** - **   | **+43.2%**  |          | 2026-09-01 09:35 |
| ernie4-5:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 4 000.2       | 418.5        | 128     | 1 749     |  -      |  -        |             |          | 2026-09-01 19:36 |
| ernie4-5:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **2 950.3**   | **579.9**    | **128** | **274**   | ** - ** | ** - **   | **+38.6%**  |          | 2026-09-01 19:36 |

### gemma4

![gemma4](img/family-gemma4.svg)

| Model         | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode    | Δ energy   | Date             |
|---------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|-------------|------------|------------------|
| gemma4:12b    | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 1 690.5       | 32.9         | 128     | 5 234      |  -      |  -        |             |            | 2026-09-01 17:46 |
| gemma4:12b    | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **2 779.5**   | **65.3**     | **128** | **1 961**  | ** - ** | ** - **   | **+98.5%**  |            | 2026-09-01 17:46 |
| gemma4:12b    | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 1 640.0       | 53.3         | 128     | 5 236      |  -      |  -        |             |            | 2026-09-01 16:09 |
| gemma4:12b    | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **2 507.8**   | **67.9**     | **128** | **2 033**  | ** - ** | ** - **   | **+27.4%**  |            | 2026-09-01 16:09 |
| gemma4:12b    | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent  |            | 2026-09-01 14:33 |
| gemma4:12b    | 4096   | medium | non-stream | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent  |            | 2026-09-01 14:33 |
| gemma4:12b    | 4096   | medium | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 74      |  -         |  -      |  -        | incoherent  |            | 2026-09-01 12:54 |
| gemma4:12b    | 4096   | medium | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent  |            | 2026-09-01 12:54 |
| gemma4:12b    | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 288.1         | 22.4         | 61      | 3 886      |  -      |  -        |             |            | 2026-09-01 11:16 |
| gemma4:12b    | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **517.3**     | **71.8**     | **128** | **1 784**  | ** - ** | ** - **   |             |            | 2026-09-01 11:16 |
| gemma4:12b    | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 289.3         | 42.6         | 61      | 5 331      |  -      |  -        |             |            | 2026-09-01 09:37 |
| gemma4:12b    | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **432.3**     | **71.9**     | **128** | **1 837**  | ** - ** | ** - **   |             |            | 2026-09-01 09:37 |
| gemma4:12b    | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 298.1         | 54.4         | 61      | 4 027      |  -      |  -        |             |            | 2026-09-01 19:38 |
| gemma4:12b    | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **397.7**     | **71.8**     | **128** | **1 835**  | ** - ** | ** - **   |             |            | 2026-09-01 19:38 |
| gemma4:26b    | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 2 209.7       | 49.7         | 128     | 5 032      |  -      |  -        |             |            | 2026-09-01 17:48 |
| gemma4:26b    | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **1 044.7**   | **73.1**     | **128** | **1 773**  | ** - ** | ** - **   | **+46.9%**  |            | 2026-09-01 17:48 |
| gemma4:26b    | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 2 233.7       | 99.2         | 128     | 5 022      |  -      |  -        |             |            | 2026-09-01 16:11 |
| gemma4:26b    | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **1 040.6**   | **90.7**     | **128** | **1 770**  | ** - ** | ** - **   | -8.6%       |            | 2026-09-01 16:11 |
| gemma4:26b    | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent  |            | 2026-09-01 14:35 |
| gemma4:26b    | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **490.5**     | **85.4**     | **128** | **1 519**  | ** - ** | ** - **   |             |            | 2026-09-01 14:35 |
| gemma4:26b    | 4096   | medium | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent  |            | 2026-09-01 12:57 |
| gemma4:26b    | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **469.9**     | **95.4**     | **128** | **1 565**  | ** - ** | ** - **   |             |            | 2026-09-01 12:57 |
| gemma4:26b    | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 84.5          | 11.4         | 128     | 4 582      |  -      |  -        |             |            | 2026-09-01 11:18 |
| gemma4:26b    | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **307.8**     | **88.4**     | **128** | **1 463**  | ** - ** | ** - **   | **+676.5%** |            | 2026-09-01 11:18 |
| gemma4:26b    | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 5.1           | 101.3        | 93      | 5 661      |  -      |  -        |             |            | 2026-09-01 09:40 |
| gemma4:26b    | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **245.5**     | **95.5**     | **128** | **1 519**  | ** - ** | ** - **   |             |            | 2026-09-01 09:40 |
| gemma4:26b    | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 69.5          | 101.4        | 93      | 4 697      |  -      |  -        |             |            | 2026-09-01 19:40 |
| gemma4:26b    | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **257.6**     | **96.2**     | **128** | **1 533**  | ** - ** | ** - **   |             |            | 2026-09-01 19:40 |
| gemma4:31b    | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           |  -      |  -         |  -      |  -        | incoherent  |            | 2026-09-01 17:51 |
| gemma4:31b    | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **803.7**     | **21.6**     | **128** | **5 921**  | ** - ** | ** - **   |             |            | 2026-09-01 17:51 |
| gemma4:31b    | 4096   | long   | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 44      |  -         |  -      |  -        | incoherent  |            | 2026-09-01 16:14 |
| gemma4:31b    | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **775.4**     | **22.9**     | **128** | **5 975**  | ** - ** | ** - **   |             |            | 2026-09-01 16:14 |
| gemma4:31b    | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           |  -      | 5 832      |  -      |  -        |             |            | 2026-09-01 14:37 |
| gemma4:31b    | 4096   | medium | non-stream | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent  |            | 2026-09-01 14:37 |
| gemma4:31b    | 4096   | medium | stream     | GPU    | Ollama 0.32.6   |  -            | 25.4         | 48      | 5 908      |  -      |  -        |             |            | 2026-09-01 12:59 |
| gemma4:31b    | 4096   | medium | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent  |            | 2026-09-01 12:59 |
| gemma4:31b    | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 76.1          | 8.3          | 128     | 7 646      |  -      |  -        |             |            | 2026-09-01 11:21 |
| gemma4:31b    | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **194.1**     | **23.9**     | **128** | **5 370**  | ** - ** | ** - **   | **+188.0%** |            | 2026-09-01 11:21 |
| gemma4:31b    | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 83.3          | 25.7         | 96      | 7 652      |  -      |  -        |             |            | 2026-09-01 09:42 |
| gemma4:31b    | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **193.8**     | **24.4**     | **128** | **5 407**  | ** - ** | ** - **   |             |            | 2026-09-01 09:42 |
| gemma4:31b    | 131072 | short  | stream     | GPU    | Ollama 0.32.6   |  -            | 5.1          | 63      | 18 287     |  -      |  -        |             |            | 2026-09-01 19:43 |
| gemma4:31b    | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **190.8**     | **24.3**     | **128** | **5 404**  | ** - ** | ** - **   |             |            | 2026-09-01 19:43 |
| gemma4:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 6 523.8       | 49.0         | 128     | 4 060      |  -      |  -        |             |            | 2026-09-01 17:53 |
| gemma4:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **3 868.8**   | **111.8**    | **128** | **1 145**  | ** - ** | ** - **   | **+128.0%** |            | 2026-09-01 17:53 |
| gemma4:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 6 518.1       | 89.6         | 128     | 4 029      |  -      |  -        |             |            | 2026-09-01 16:15 |
| gemma4:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **4 284.1**   | **121.9**    | **128** | **1 163**  | ** - ** | ** - **   | **+36.1%**  |            | 2026-09-01 16:15 |
| gemma4:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 1 328.2       | 49.5         | 128     | 4 096      |  -      |  -        |             |            | 2026-09-01 14:39 |
| gemma4:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **1 251.9**   | **119.3**    | **128** | **1 074**  | ** - ** | ** - **   | **+141.0%** |            | 2026-09-01 14:39 |
| gemma4:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 1 332.9       | 90.2         | 128     | 3 965      |  -      |  -        |             |            | 2026-09-01 13:01 |
| gemma4:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **876.9**     | **127.1**    | **128** | **1 113**  | ** - ** | ** - **   | **+40.9%**  |            | 2026-09-01 13:01 |
| gemma4:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 487.4         | 50.1         | 128     | 3 965      |  -      |  -        |             |            | 2026-09-01 11:22 |
| gemma4:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **439.9**     | **121.8**    | **128** | **1 051**  | ** - ** | ** - **   | **+142.9%** |            | 2026-09-01 11:22 |
| gemma4:latest | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 82.9          | 7.0          | 128     | 21 777     | 765     | 5.980     |             |            | 2026-08-18 01:14 |
| gemma4:latest | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **79.9**      | **6.7**      | **128** | **19 608** | **652** | **5.090** | -4.6%       | **+17.5%** | 2026-08-18 01:14 |
| gemma4:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 488.7         | 91.1         | 128     | 3 981      |  -      |  -        |             |            | 2026-09-01 09:44 |
| gemma4:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **544.6**     | **128.4**    | **128** | **1 081**  | ** - ** | ** - **   | **+40.9%**  |            | 2026-09-01 09:44 |
| gemma4:latest | 16384  | long   | stream     | GPU    | Ollama 0.32.6   | 8 418.1       | 119.0        | 128     | 3 811      | 349     | 2.726     |             |            | 2026-08-18 01:14 |
| gemma4:latest | 16384  | long   | stream     | GPU    | **loken 0.1.0** | **14 816.7**  | **124.8**    | **128** | **1 124**  | **194** | **1.515** | **+4.9%**   | **+79.9%** | 2026-08-18 01:14 |
| gemma4:latest | 16384  | short  | stream     | GPU    | Ollama 0.32.6   | 976.8         | 118.7        | 128     | 3 806      | 345     | 2.697     |             |            | 2026-08-18 01:14 |
| gemma4:latest | 16384  | short  | stream     | GPU    | **loken 0.1.0** | **1 600.3**   | **131.0**    | **128** | **1 006**  | **180** | **1.410** | **+10.4%**  | **+91.3%** | 2026-08-18 01:14 |
| gemma4:latest | 32768  | long   | stream     | GPU    | Ollama 0.32.6   | 8 410.6       | 119.0        | 128     | 3 785      | 341     | 2.667     |             |            | 2026-08-18 01:14 |
| gemma4:latest | 32768  | long   | stream     | GPU    | **loken 0.1.0** | **11 025.6**  | **126.1**    | **128** | **1 095**  | **199** | **1.552** | **+6.0%**   | **+71.9%** | 2026-08-18 01:14 |
| gemma4:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 489.1         | 91.0         | 128     | 4 048      |  -      |  -        |             |            | 2026-09-01 19:45 |
| gemma4:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **246.8**     | **129.2**    | **128** | **1 099**  | ** - ** | ** - **   | **+42.1%**  |            | 2026-09-01 19:45 |

### gptoss

![gptoss](img/family-gptoss.svg)

| Model       | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms  | J/req   | J/token | Δ decode    | Δ energy | Date             |
|-------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|---------|---------|---------|-------------|----------|------------------|
| gpt-oss:20b | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 11 829.6      | 53.3         | 128     | 4 430   |  -      |  -      |             |          | 2026-09-01 18:05 |
| gpt-oss:20b | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **808 694.5** | **166.0**    | **128** | **899** | ** - ** | ** - ** | **+211.3%** |          | 2026-09-01 18:05 |
| gpt-oss:20b | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 11 792.5      | 95.2         | 128     | 4 387   |  -      |  -      |             |          | 2026-09-01 16:27 |
| gpt-oss:20b | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **350 903.3** | **210.1**    | **128** | **905** | ** - ** | ** - ** | **+120.7%** |          | 2026-09-01 16:27 |
| gpt-oss:20b | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 1 250.0       | 52.3         | 128     | 4 408   |  -      |  -      |             |          | 2026-09-01 14:51 |
| gpt-oss:20b | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **58 988.3**  | **172.1**    | **128** | **822** | ** - ** | ** - ** | **+229.1%** |          | 2026-09-01 14:51 |
| gpt-oss:20b | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 1 238.5       | 93.5         | 128     | 4 434   |  -      |  -      |             |          | 2026-09-01 13:13 |
| gpt-oss:20b | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **43 843.2**  | **212.4**    | **128** | **831** | ** - ** | ** - ** | **+127.3%** |          | 2026-09-01 13:13 |
| gpt-oss:20b | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 456.7         | 52.7         | 128     | 4 385   |  -      |  -      |             |          | 2026-09-01 11:34 |
| gpt-oss:20b | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **39 356.6**  | **175.1**    | **128** | **809** | ** - ** | ** - ** | **+232.2%** |          | 2026-09-01 11:34 |
| gpt-oss:20b | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 461.9         | 94.1         | 128     | 4 472   |  -      |  -      |             |          | 2026-09-01 09:56 |
| gpt-oss:20b | 4096   | short  | stream     | GPU    | vLLM 0.22.0     |  -            | 146.9        | 128     | 887     | 189     | 1.476   |             |          | 2026-08-10 07:13 |
| gpt-oss:20b | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **17 015.8**  | **211.9**    | **128** | **826** | ** - ** | ** - ** | **+44.2%**  |          | 2026-09-01 09:56 |
| gpt-oss:20b | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 467.0         | 94.6         | 128     | 4 557   |  -      |  -      |             |          | 2026-09-01 19:57 |
| gpt-oss:20b | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **17 227.1**  | **212.2**    | **128** | **825** | ** - ** | ** - ** | **+124.4%** |          | 2026-09-01 19:57 |

### granite

![granite](img/family-granite.svg)

| Model               | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode   | Δ energy    | Date             |
|---------------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|------------|-------------|------------------|
| granite3.1-dense:2b | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 40 711.3      | 143.9        | 128     | 1 922      |  -      |  -        |            |             | 2026-09-01 18:06 |
| granite3.1-dense:2b | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **8 585.3**   | **225.6**    | **128** | **570**    | ** - ** | ** - **   | **+56.8%** |             | 2026-09-01 18:06 |
| granite3.1-dense:2b | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 41 559.5      | 221.8        | 128     | 1 887      |  -      |  -        |            |             | 2026-09-01 16:29 |
| granite3.1-dense:2b | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **6 536.4**   | **260.0**    | **128** | **576**    | ** - ** | ** - **   | **+17.2%** |             | 2026-09-01 16:29 |
| granite3.1-dense:2b | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 7 339.3       | 142.8        | 128     | 1 891      |  -      |  -        |            |             | 2026-09-01 14:53 |
| granite3.1-dense:2b | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **1 718.0**   | **243.1**    | **128** | **529**    | ** - ** | ** - **   | **+70.3%** |             | 2026-09-01 14:53 |
| granite3.1-dense:2b | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 7 354.5       | 223.8        | 128     | 1 918      |  -      |  -        |            |             | 2026-09-01 13:14 |
| granite3.1-dense:2b | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 868.8**   | **277.1**    | **128** | **519**    | ** - ** | ** - **   | **+23.8%** |             | 2026-09-01 13:14 |
| granite3.1-dense:2b | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 2 563.6       | 143.2        | 128     | 1 891      |  -      |  -        |            |             | 2026-09-01 11:36 |
| granite3.1-dense:2b | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **1 108.9**   | **248.8**    | **128** | **522**    | ** - ** | ** - **   | **+73.7%** |             | 2026-09-01 11:36 |
| granite3.1-dense:2b | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 606.7         | 13.3         | 128     | 11 258     | 418     | 3.263     |            |             | 2026-08-18 01:14 |
| granite3.1-dense:2b | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **149.2**     | **12.2**     | **128** | **10 850** | **501** | **3.917** | -8.3%      | -16.7%      | 2026-08-18 01:14 |
| granite3.1-dense:2b | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 2 541.6       | 223.8        | 128     | 1 886      |  -      |  -        |            |             | 2026-09-01 09:58 |
| granite3.1-dense:2b | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **535.2**     | **278.5**    | **128** | **522**    | ** - ** | ** - **   | **+24.5%** |             | 2026-09-01 09:58 |
| granite3.1-dense:2b | 16384  | long   | stream     | GPU    | Ollama 0.32.6   | 60 165.8      | 288.7        | 128     | 1 922      | 195     | 1.525     |            |             | 2026-08-18 01:14 |
| granite3.1-dense:2b | 16384  | long   | stream     | GPU    | **loken 0.1.0** | **20 130.6**  | **267.7**    | **128** | **530**    | **104** | **0.812** | -7.3%      | **+87.8%**  | 2026-08-18 01:14 |
| granite3.1-dense:2b | 32768  | long   | stream     | GPU    | Ollama 0.32.6   | 59 867.9      | 288.0        | 128     | 1 934      | 199     | 1.555     |            |             | 2026-08-18 01:14 |
| granite3.1-dense:2b | 32768  | long   | stream     | GPU    | **loken 0.1.0** | **5 549.3**   | **266.8**    | **128** | **555**    | **98**  | **0.763** | -7.4%      | **+103.8%** | 2026-08-18 01:14 |
| granite3.1-dense:2b | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 2 485.6       | 220.2        | 128     | 2 053      |  -      |  -        |            |             | 2026-09-01 19:59 |
| granite3.1-dense:2b | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **649.1**     | **279.3**    | **128** | **516**    | ** - ** | ** - **   | **+26.8%** |             | 2026-09-01 19:59 |

### granitemoe

![granitemoe](img/family-granitemoe.svg)

| Model           | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token   | Δ decode  | Δ energy | Date             |
|-----------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|-----------|-----------|----------|------------------|
| granite3-moe:1b | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 51 193.4      | 254.9        | 128     | 517       |  -      |  -        |           |          | 2026-09-01 18:06 |
| granite3-moe:1b | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **4 622.4**   | **256.4**    | **128** | **513**   | ** - ** | ** - **   | **+0.6%** |          | 2026-09-01 18:06 |
| granite3-moe:1b | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 51 110.1      | 323.7        | 128     | 534       |  -      |  -        |           |          | 2026-09-01 16:28 |
| granite3-moe:1b | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **4 079.3**   | **327.4**    | **128** | **499**   | ** - ** | ** - **   | **+1.1%** |          | 2026-09-01 16:28 |
| granite3-moe:1b | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 9 247.2       | 250.0        | 128     | 524       |  -      |  -        |           |          | 2026-09-01 14:52 |
| granite3-moe:1b | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **2 478.3**   | **271.9**    | **128** | **483**   | ** - ** | ** - **   | **+8.8%** |          | 2026-09-01 14:52 |
| granite3-moe:1b | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 9 088.8       | 331.3        | 128     | 512       |  -      |  -        |           |          | 2026-09-01 13:14 |
| granite3-moe:1b | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 272.6**   | **320.1**    | **128** | **505**   | ** - ** | ** - **   | -3.4%     |          | 2026-09-01 13:14 |
| granite3-moe:1b | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 3 202.4       | 258.0        | 128     | 511       |  -      |  -        |           |          | 2026-09-01 11:35 |
| granite3-moe:1b | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **824.4**     | **274.3**    | **128** | **474**   | ** - ** | ** - **   | **+6.3%** |          | 2026-09-01 11:35 |
| granite3-moe:1b | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 2 464.5       | 68.7         | 128     | 3 165     | 152     | 1.184     |           |          | 2026-08-18 01:14 |
| granite3-moe:1b | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **133.2**     | **55.7**     | **128** | **2 593** | **158** | **1.234** | -18.9%    | -4.1%    | 2026-08-18 01:14 |
| granite3-moe:1b | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 3 262.5       | 333.8        | 128     | 516       |  -      |  -        |           |          | 2026-09-01 09:57 |
| granite3-moe:1b | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **754.6**     | **323.5**    | **128** | **471**   | ** - ** | ** - **   | -3.1%     |          | 2026-09-01 09:57 |
| granite3-moe:1b | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 3 175.6       | 331.9        | 128     | 519       |  -      |  -        |           |          | 2026-09-01 19:58 |
| granite3-moe:1b | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **694.3**     | **325.7**    | **128** | **464**   | ** - ** | ** - **   | -1.9%     |          | 2026-09-01 19:58 |

### lfm2

![lfm2](img/family-lfm2.svg)

| Model                  | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms  | J/req   | J/token | Δ decode    | Δ energy | Date             |
|------------------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|---------|---------|---------|-------------|----------|------------------|
| lfm2.5-thinking:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 48 403.9      | 224.3        | 128     | 1 660   |  -      |  -      |             |          | 2026-09-01 18:07 |
| lfm2.5-thinking:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **968 715.4** | **438.3**    | **128** | **441** | ** - ** | ** - ** | **+95.4%**  |          | 2026-09-01 18:07 |
| lfm2.5-thinking:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 48 596.2      | 415.5        | 128     | 1 668   |  -      |  -      |             |          | 2026-09-01 16:30 |
| lfm2.5-thinking:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **248 397.1** | **644.0**    | **128** | **443** | ** - ** | ** - ** | **+55.0%**  |          | 2026-09-01 16:30 |
| lfm2.5-thinking:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 7 819.7       | 213.2        | 128     | 1 688   |  -      |  -      |             |          | 2026-09-01 14:54 |
| lfm2.5-thinking:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **217 436.5** | **463.0**    | **128** | **404** | ** - ** | ** - ** | **+117.1%** |          | 2026-09-01 14:54 |
| lfm2.5-thinking:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 7 839.8       | 396.0        | 128     | 1 713   |  -      |  -      |             |          | 2026-09-01 13:15 |
| lfm2.5-thinking:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **86 908.3**  | **668.5**    | **128** | **414** | ** - ** | ** - ** | **+68.8%**  |          | 2026-09-01 13:15 |
| lfm2.5-thinking:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 2 821.0       | 215.5        | 128     | 1 672   |  -      |  -      |             |          | 2026-09-01 11:37 |
| lfm2.5-thinking:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **56 031.6**  | **442.8**    | **128** | **402** | ** - ** | ** - ** | **+105.4%** |          | 2026-09-01 11:37 |
| lfm2.5-thinking:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 2 848.9       | 393.7        | 128     | 1 679   |  -      |  -      |             |          | 2026-09-01 09:59 |
| lfm2.5-thinking:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **29 101.6**  | **636.3**    | **128** | **440** | ** - ** | ** - ** | **+61.6%**  |          | 2026-09-01 09:59 |
| lfm2.5-thinking:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 2 732.8       | 386.2        | 128     | 1 894   |  -      |  -      |             |          | 2026-09-01 20:00 |
| lfm2.5-thinking:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **8 435.0**   | **669.1**    | **128** | **422** | ** - ** | ** - ** | **+73.3%**  |          | 2026-09-01 20:00 |

### lfm2moe

![lfm2moe](img/family-lfm2moe.svg)

| Model       | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode   | Δ energy    | Date             |
|-------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|------------|-------------|------------------|
| lfm2:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 16 874.9      | 133.1        | 128     | 3 169      |  -      |  -        |            |             | 2026-09-01 18:09 |
| lfm2:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **718 749.8** | **237.2**    | **128** | **684**    | ** - ** | ** - **   | **+78.2%** |             | 2026-09-01 18:09 |
| lfm2:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 16 897.6      | 233.8        | 128     | 3 112      |  -      |  -        |            |             | 2026-09-01 16:31 |
| lfm2:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **222 013.4** | **308.5**    | **128** | **693**    | ** - ** | ** - **   | **+31.9%** |             | 2026-09-01 16:31 |
| lfm2:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 2 840.1       | 136.0        | 128     | 3 136      |  -      |  -        |            |             | 2026-09-01 14:55 |
| lfm2:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **20 936.2**  | **70.4**     | **4**   | **257**    | ** - ** | ** - **   |            |             | 2026-09-01 14:55 |
| lfm2:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 2 922.4       | 233.5        | 128     | 3 125      |  -      |  -        |            |             | 2026-09-01 13:17 |
| lfm2:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **53 764.4**  | **238.1**    | **4**   | **226**    | ** - ** | ** - **   |            |             | 2026-09-01 13:17 |
| lfm2:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent |             | 2026-09-01 11:39 |
| lfm2:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent |             | 2026-09-01 11:39 |
| lfm2:latest | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 140.4         | 14.4         | 128     | 13 330     | 477     | 3.724     |            |             | 2026-08-18 01:14 |
| lfm2:latest | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **730.7**     | **14.0**     | **128** | **12 797** | **572** | **4.472** | -3.4%      | -16.7%      | 2026-08-18 01:14 |
| lfm2:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent |             | 2026-09-01 10:00 |
| lfm2:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent |             | 2026-09-01 10:00 |
| lfm2:latest | 16384  | long   | stream     | GPU    | Ollama 0.32.6   | 23 402.9      | 297.3        | 128     | 2 672      | 242     | 1.894     |            |             | 2026-08-18 01:14 |
| lfm2:latest | 16384  | long   | stream     | GPU    | **loken 0.1.0** | **355 793.0** | **307.0**    | **128** | **655**    | **111** | **0.869** | **+3.3%**  | **+117.9%** | 2026-08-18 01:14 |
| lfm2:latest | 32768  | long   | stream     | GPU    | Ollama 0.32.6   | 23 412.8      | 297.3        | 128     | 2 659      | 242     | 1.888     |            |             | 2026-08-18 01:14 |
| lfm2:latest | 32768  | long   | stream     | GPU    | **loken 0.1.0** | **352 400.1** | **307.6**    | **128** | **630**    | **124** | **0.973** | **+3.5%**  | **+94.1%**  | 2026-08-18 01:14 |
| lfm2:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent |             | 2026-09-01 20:01 |
| lfm2:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent |             | 2026-09-01 20:01 |

### llama

![llama](img/family-llama.svg)

| Model                | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms      | J/req   | J/token   | Δ decode     | Δ energy | Date             |
|----------------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|-------------|---------|-----------|--------------|----------|------------------|
| deepseek-r1:70b      | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 273.5         | 1.5          | 128     | 88 767      |  -      |  -        |              |          | 2026-09-01 17:24 |
| deepseek-r1:70b      | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **18.6**      | **1.3**      | **128** | **99 377**  | ** - ** | ** - **   | -10.4%       |          | 2026-09-01 17:24 |
| deepseek-r1:70b      | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 272.3         | 1.5          | 128     | 88 777      |  -      |  -        |              |          | 2026-09-01 15:47 |
| deepseek-r1:70b      | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **18.4**      | **1.5**      | **128** | **100 193** | ** - ** | ** - **   | -1.6%        |          | 2026-09-01 15:47 |
| deepseek-r1:70b      | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 51.2          | 1.5          | 128     | 88 444      |  -      |  -        |              |          | 2026-09-01 14:11 |
| deepseek-r1:70b      | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **14.6**      | **1.3**      | **128** | **99 211**  | ** - ** | ** - **   | -10.6%       |          | 2026-09-01 14:11 |
| deepseek-r1:70b      | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 51.4          | 1.6          | 128     | 88 181      |  -      |  -        |              |          | 2026-09-01 12:32 |
| deepseek-r1:70b      | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **14.5**      | **1.5**      | **128** | **96 647**  | ** - ** | ** - **   | -6.0%        |          | 2026-09-01 12:32 |
| deepseek-r1:70b      | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 20.4          | 1.5          | 128     | 88 064      |  -      |  -        |              |          | 2026-09-01 10:54 |
| deepseek-r1:70b      | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **8.7**       | **1.4**      | **128** | **95 279**  | ** - ** | ** - **   | -7.2%        |          | 2026-09-01 10:54 |
| deepseek-r1:70b      | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 20.3          | 1.5          | 128     | 88 302      |  -      |  -        |              |          | 2026-09-01 09:15 |
| deepseek-r1:70b      | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **8.7**       | **1.5**      | **128** | **91 938**  | ** - ** | ** - **   | -0.9%        |          | 2026-09-01 09:15 |
| deepseek-r1:70b      | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 10.7          | 0.8          | 128     | 165 115     |  -      |  -        |              |          | 2026-09-01 19:10 |
| deepseek-r1:70b      | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **8.8**       | **1.5**      | **128** | **92 457**  | ** - ** | ** - **   | **+86.8%**   |          | 2026-09-01 19:10 |
| deepseek-r1:70b-q3ks | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 1 151.4       | 5.8          | 128     | 23 520      |  -      |  -        |              |          | 2026-09-01 17:29 |
| deepseek-r1:70b-q3ks | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **247.9**     | **17.8**     | **128** | **7 245**   | ** - ** | ** - **   | **+204.0%**  |          | 2026-09-01 17:29 |
| deepseek-r1:70b-q3ks | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 1 154.4       | 6.9          | 128     | 23 527      |  -      |  -        |              |          | 2026-09-01 15:51 |
| deepseek-r1:70b-q3ks | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **254.7**     | **20.4**     | **128** | **7 185**   | ** - ** | ** - **   | **+196.2%**  |          | 2026-09-01 15:51 |
| deepseek-r1:70b-q3ks | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 227.6         | 5.9          | 128     | 23 402      |  -      |  -        |              |          | 2026-09-01 14:15 |
| deepseek-r1:70b-q3ks | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **179.7**     | **19.8**     | **128** | **6 480**   | ** - ** | ** - **   | **+238.5%**  |          | 2026-09-01 14:15 |
| deepseek-r1:70b-q3ks | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 227.1         | 6.9          | 128     | 23 352      |  -      |  -        |              |          | 2026-09-01 12:37 |
| deepseek-r1:70b-q3ks | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **179.4**     | **21.1**     | **128** | **6 493**   | ** - ** | ** - **   | **+204.8%**  |          | 2026-09-01 12:37 |
| deepseek-r1:70b-q3ks | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 85.7          | 5.9          | 128     | 23 310      |  -      |  -        |              |          | 2026-09-01 10:58 |
| deepseek-r1:70b-q3ks | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **101.8**     | **20.2**     | **128** | **6 367**   | ** - ** | ** - **   | **+243.5%**  |          | 2026-09-01 10:58 |
| deepseek-r1:70b-q3ks | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 86.0          | 6.9          | 128     | 23 287      |  -      |  -        |              |          | 2026-09-01 09:20 |
| deepseek-r1:70b-q3ks | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **103.6**     | **21.2**     | **128** | **6 401**   | ** - ** | ** - **   | **+205.7%**  |          | 2026-09-01 09:20 |
| deepseek-r1:70b-q3ks | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 14.5          | 1.2          | 128     | 117 268     |  -      |  -        |              |          | 2026-09-01 19:19 |
| deepseek-r1:70b-q3ks | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **103.5**     | **21.2**     | **128** | **6 392**   | ** - ** | ** - **   | **+1712.7%** |          | 2026-09-01 19:19 |
| devstral:24b         | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 5 774.2       |  -           | 1       | 2 790       |  -      |  -        | no answer    |          | 2026-09-01 17:43 |
| devstral:24b         | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 1 269.5       | 26.5         | 128     | 6 243       |  -      |  -        |              |          | 2026-09-01 14:29 |
| devstral:24b         | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **813.2**     | **50.1**     | **128** | **2 559**   | ** - ** | ** - **   | **+89.0%**   |          | 2026-09-01 14:29 |
| devstral:24b         | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 1 269.8       | 36.2         | 128     | 6 240       |  -      |  -        |              |          | 2026-09-01 12:51 |
| devstral:24b         | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **759.8**     | **52.1**     | **128** | **2 564**   | ** - ** | ** - **   | **+43.9%**   |          | 2026-09-01 12:51 |
| devstral:24b         | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 459.1         | 26.5         | 128     | 6 244       |  -      |  -        |              |          | 2026-09-01 11:13 |
| devstral:24b         | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **408.3**     | **49.3**     | **128** | **2 598**   | ** - ** | ** - **   | **+86.2%**   |          | 2026-09-01 11:13 |
| devstral:24b         | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 459.0         | 36.3         | 128     | 6 288       |  -      |  -        |              |          | 2026-09-01 09:34 |
| devstral:24b         | 4096   | short  | stream     | GPU    | vLLM 0.22.0     |  -            | 50.6         | 128     | 2 541       | 580     | 4.534     |              |          | 2026-08-10 07:40 |
| devstral:24b         | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **326.9**     | **51.0**     | **128** | **2 611**   | ** - ** | ** - **   | **+0.6%**    |          | 2026-09-01 09:34 |
| devstral:24b         | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 91.1          | 6.9          | 128     | 22 203      |  -      |  -        |              |          | 2026-09-01 19:35 |
| devstral:24b         | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **324.4**     | **50.7**     | **128** | **2 628**   | ** - ** | ** - **   | **+629.3%**  |          | 2026-09-01 19:35 |
| falcon3:latest       | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 41 564.2      | 119.5        | 128     | 2 129       |  -      |  -        |              |          | 2026-09-01 17:45 |
| falcon3:latest       | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 41 746.9      | 208.9        | 128     | 2 120       |  -      |  -        |              |          | 2026-09-01 16:07 |
| falcon3:latest       | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 6 179.3       | 121.1        | 128     | 2 087       |  -      |  -        |              |          | 2026-09-01 14:31 |
| falcon3:latest       | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **2 416.8**   | **342.3**    | **128** | **378**     | ** - ** | ** - **   | **+182.7%**  |          | 2026-09-01 14:31 |
| falcon3:latest       | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 6 238.4       | 211.0        | 128     | 2 113       |  -      |  -        |              |          | 2026-09-01 12:53 |
| falcon3:latest       | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 618.0**   | **389.5**    | **128** | **394**     | ** - ** | ** - **   | **+84.6%**   |          | 2026-09-01 12:53 |
| falcon3:latest       | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 2 229.4       | 120.8        | 128     | 2 117       |  -      |  -        |              |          | 2026-09-01 11:14 |
| falcon3:latest       | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **915.0**     | **342.4**    | **128** | **378**     | ** - ** | ** - **   | **+183.6%**  |          | 2026-09-01 11:14 |
| falcon3:latest       | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 1 035.9       | 14.8         | 21      | 3 106       | 151     | 7.206     |              |          | 2026-08-18 01:14 |
| falcon3:latest       | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **158.6**     | **14.1**     | **79**  | **5 967**   | **327** | **4.118** |              |          | 2026-08-18 01:14 |
| falcon3:latest       | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 2 232.3       | 211.2        | 128     | 2 121       |  -      |  -        |              |          | 2026-09-01 09:36 |
| falcon3:latest       | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **764.1**     | **391.9**    | **128** | **385**     | ** - ** | ** - **   | **+85.6%**   |          | 2026-09-01 09:36 |
| falcon3:latest       | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 2 217.4       | 212.0        | 128     | 776         |  -      |  -        |              |          | 2026-09-01 19:37 |
| falcon3:latest       | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **780.9**     | **391.4**    | **128** | **384**     | ** - ** | ** - **   | **+84.7%**   |          | 2026-09-01 19:37 |
| llama3.2:1b          | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 34 177.4      | 120.4        | 128     | 2 142       |  -      |  -        |              |          | 2026-09-01 18:10 |
| llama3.2:1b          | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **9 809.2**   | **344.2**    | **128** | **375**     | ** - ** | ** - **   | **+185.7%**  |          | 2026-09-01 18:10 |
| llama3.2:1b          | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 33 528.7      | 244.0        | 128     | 2 152       |  -      |  -        |              |          | 2026-09-01 16:32 |
| llama3.2:1b          | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **8 677.9**   | **414.2**    | **128** | **374**     | ** - ** | ** - **   | **+69.8%**   |          | 2026-09-01 16:32 |
| llama3.2:1b          | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 6 731.6       | 120.9        | 128     | 2 136       |  -      |  -        |              |          | 2026-09-01 14:56 |
| llama3.2:1b          | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **2 178.6**   | **383.0**    | **128** | **336**     | ** - ** | ** - **   | **+216.8%**  |          | 2026-09-01 14:56 |
| llama3.2:1b          | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 6 612.9       | 237.5        | 128     | 2 166       |  -      |  -        |              |          | 2026-09-01 13:18 |
| llama3.2:1b          | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 980.5**   | **447.5**    | **128** | **346**     | ** - ** | ** - **   | **+88.4%**   |          | 2026-09-01 13:18 |
| llama3.2:1b          | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 2 382.3       | 116.2        | 128     | 2 187       |  -      |  -        |              |          | 2026-09-01 11:40 |
| llama3.2:1b          | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **808.5**     | **376.7**    | **128** | **344**     | ** - ** | ** - **   | **+224.1%**  |          | 2026-09-01 11:40 |
| llama3.2:1b          | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 538.1         | 16.6         | 128     | 9 415       | 357     | 2.791     |              |          | 2026-08-18 01:14 |
| llama3.2:1b          | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **327.6**     | **15.9**     | **128** | **8 239**   | **414** | **3.235** | -3.8%        | -13.7%   | 2026-08-18 01:14 |
| llama3.2:1b          | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 2 400.0       | 232.2        | 128     | 2 157       |  -      |  -        |              |          | 2026-09-01 10:01 |
| llama3.2:1b          | 4096   | short  | stream     | GPU    | vLLM 0.22.0     |  -            | 292.6        | 128     | 446         | 88      | 0.685     |              |          | 2026-08-10 07:11 |
| llama3.2:1b          | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **737.8**     | **451.0**    | **128** | **344**     | ** - ** | ** - **   | **+54.1%**   |          | 2026-09-01 10:01 |
| llama3.2:1b          | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 2 415.0       | 228.0        | 128     | 2 291       |  -      |  -        |              |          | 2026-09-01 20:02 |
| llama3.2:1b          | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **478.6**     | **451.9**    | **128** | **348**     | ** - ** | ** - **   | **+98.2%**   |          | 2026-09-01 20:02 |
| magistral:latest     | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 5 351.1       |  -           | 1       | 2 827       |  -      |  -        | no answer    |          | 2026-09-01 18:12 |
| magistral:latest     | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 1 263.3       | 26.3         | 128     | 6 283       |  -      |  -        |              |          | 2026-09-01 14:58 |
| magistral:latest     | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **803.9**     | **49.6**     | **128** | **2 586**   | ** - ** | ** - **   | **+88.1%**   |          | 2026-09-01 14:58 |
| magistral:latest     | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 1 218.9       | 36.3         | 128     | 6 296       |  -      |  -        |              |          | 2026-09-01 13:20 |
| magistral:latest     | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 130.1**   | **53.3**     | **128** | **2 500**   | ** - ** | ** - **   | **+46.8%**   |          | 2026-09-01 13:20 |
| magistral:latest     | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 419.9         | 26.3         | 128     | 6 422       |  -      |  -        |              |          | 2026-09-01 11:42 |
| magistral:latest     | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **407.2**     | **49.6**     | **128** | **2 585**   | ** - ** | ** - **   | **+88.6%**   |          | 2026-09-01 11:42 |
| magistral:latest     | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 427.9         | 36.3         | 128     | 6 343       |  -      |  -        |              |          | 2026-09-01 10:03 |
| magistral:latest     | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **322.3**     | **51.3**     | **128** | **2 605**   | ** - ** | ** - **   | **+41.5%**   |          | 2026-09-01 10:03 |
| magistral:latest     | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 423.5         | 36.3         | 128     | 6 285       |  -      |  -        |              |          | 2026-09-01 20:04 |
| magistral:latest     | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **321.8**     | **51.3**     | **128** | **2 624**   | ** - ** | ** - **   | **+41.4%**   |          | 2026-09-01 20:04 |
| mistral-nemo:latest  | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 11 213.4      | 46.5         | 128     | 3 964       |  -      |  -        |              |          | 2026-09-01 18:13 |
| mistral-nemo:latest  | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **4 988.6**   | **91.3**     | **128** | **1 406**   | ** - ** | ** - **   | **+96.4%**   |          | 2026-09-01 18:13 |
| mistral-nemo:latest  | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 11 186.4      | 68.8         | 128     | 3 980       |  -      |  -        |              |          | 2026-09-01 16:36 |
| mistral-nemo:latest  | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **4 744.9**   | **100.1**    | **128** | **1 403**   | ** - ** | ** - **   | **+45.4%**   |          | 2026-09-01 16:36 |
| mistral-nemo:latest  | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 2 149.8       | 45.8         | 128     | 3 994       |  -      |  -        |              |          | 2026-09-01 15:00 |
| mistral-nemo:latest  | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **1 746.4**   | **95.5**     | **128** | **1 342**   | ** - ** | ** - **   | **+108.4%**  |          | 2026-09-01 15:00 |
| mistral-nemo:latest  | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 2 186.8       | 67.9         | 128     | 4 009       |  -      |  -        |              |          | 2026-09-01 13:21 |
| mistral-nemo:latest  | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 591.6**   | **100.9**    | **128** | **1 359**   | ** - ** | ** - **   | **+48.7%**   |          | 2026-09-01 13:21 |
| mistral-nemo:latest  | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 792.3         | 45.9         | 128     | 3 998       |  -      |  -        |              |          | 2026-09-01 11:43 |
| mistral-nemo:latest  | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **313.7**     | **95.1**     | **128** | **1 349**   | ** - ** | ** - **   | **+107.3%**  |          | 2026-09-01 11:43 |
| mistral-nemo:latest  | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 713.5         | 68.1         | 128     | 5 862       |  -      |  -        |              |          | 2026-09-01 10:05 |
| mistral-nemo:latest  | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **262.8**     | **102.2**    | **128** | **1 356**   | ** - ** | ** - **   | **+50.0%**   |          | 2026-09-01 10:05 |
| mistral-nemo:latest  | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 785.9         | 68.2         | 128     | 4 069       |  -      |  -        |              |          | 2026-09-01 20:06 |
| mistral-nemo:latest  | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **614.6**     | **102.1**    | **128** | **1 338**   | ** - ** | ** - **   | **+49.8%**   |          | 2026-09-01 20:06 |

### mistral3

![mistral3](img/family-mistral3.svg)

| Model                   | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token | Δ decode     | Δ energy | Date             |
|-------------------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|---------|--------------|----------|------------------|
| devstral-small-2:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 6 066.6       | 25.8         | 128     | 6 544     |  -      |  -      |              |          | 2026-09-01 17:31 |
| devstral-small-2:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **873.6**     | **48.5**     | **128** | **2 673** | ** - ** | ** - ** | **+88.3%**   |          | 2026-09-01 17:31 |
| devstral-small-2:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 6 026.4       | 35.5         | 128     | 6 577     |  -      |  -      |              |          | 2026-09-01 15:53 |
| devstral-small-2:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **861.6**     | **57.6**     | **128** | **2 671** | ** - ** | ** - ** | **+62.3%**   |          | 2026-09-01 15:53 |
| devstral-small-2:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 1 122.9       | 25.4         | 128     | 6 592     |  -      |  -      |              |          | 2026-09-01 14:17 |
| devstral-small-2:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **554.0**     | **53.9**     | **128** | **2 412** | ** - ** | ** - ** | **+112.4%**  |          | 2026-09-01 14:17 |
| devstral-small-2:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 1 140.5       | 35.2         | 128     | 6 593     |  -      |  -      |              |          | 2026-09-01 12:39 |
| devstral-small-2:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **622.8**     | **59.3**     | **128** | **2 411** | ** - ** | ** - ** | **+68.2%**   |          | 2026-09-01 12:39 |
| devstral-small-2:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 403.0         | 25.6         | 128     | 6 571     |  -      |  -      |              |          | 2026-09-01 11:00 |
| devstral-small-2:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **301.3**     | **54.8**     | **128** | **2 368** | ** - ** | ** - ** | **+113.9%**  |          | 2026-09-01 11:00 |
| devstral-small-2:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 410.3         | 35.4         | 128     | 6 556     |  -      |  -      |              |          | 2026-09-01 09:22 |
| devstral-small-2:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **281.2**     | **59.5**     | **128** | **2 399** | ** - ** | ** - ** | **+67.8%**   |          | 2026-09-01 09:22 |
| devstral-small-2:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 67.9          | 5.3          | 128     | 28 146    |  -      |  -      |              |          | 2026-09-01 19:22 |
| devstral-small-2:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **303.5**     | **59.6**     | **128** | **2 382** | ** - ** | ** - ** | **+1020.1%** |          | 2026-09-01 19:22 |
| mistral-small3.2:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 6 185.7       | 26.4         | 128     | 6 413     |  -      |  -      |              |          | 2026-09-01 18:15 |
| mistral-small3.2:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **835.5**     | **49.2**     | **128** | **2 632** | ** - ** | ** - ** | **+86.7%**   |          | 2026-09-01 18:15 |
| mistral-small3.2:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 6 100.7       | 36.4         | 128     | 6 508     |  -      |  -      |              |          | 2026-09-01 16:38 |
| mistral-small3.2:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **861.2**     | **57.8**     | **128** | **2 638** | ** - ** | ** - ** | **+58.5%**   |          | 2026-09-01 16:38 |
| mistral-small3.2:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 1 193.8       | 26.2         | 128     | 6 418     |  -      |  -      |              |          | 2026-09-01 15:02 |
| mistral-small3.2:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **629.0**     | **53.1**     | **128** | **2 467** | ** - ** | ** - ** | **+103.0%**  |          | 2026-09-01 15:02 |
| mistral-small3.2:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 1 195.7       | 36.3         | 128     | 6 424     |  -      |  -      |              |          | 2026-09-01 13:23 |
| mistral-small3.2:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **618.8**     | **59.2**     | **128** | **2 470** | ** - ** | ** - ** | **+63.1%**   |          | 2026-09-01 13:23 |
| mistral-small3.2:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 459.8         | 26.0         | 128     | 6 429     |  -      |  -      |              |          | 2026-09-01 11:45 |
| mistral-small3.2:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **249.2**     | **54.6**     | **128** | **2 372** | ** - ** | ** - ** | **+110.5%**  |          | 2026-09-01 11:45 |
| mistral-small3.2:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 434.8         | 36.3         | 128     | 6 459     |  -      |  -      |              |          | 2026-09-01 10:07 |
| mistral-small3.2:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **300.5**     | **59.4**     | **128** | **2 388** | ** - ** | ** - ** | **+63.9%**   |          | 2026-09-01 10:07 |
| mistral-small3.2:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 71.0          | 5.5          | 128     | 27 569    |  -      |  -      |              |          | 2026-09-01 20:09 |
| mistral-small3.2:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **302.2**     | **59.4**     | **128** | **2 381** | ** - ** | ** - ** | **+985.4%**  |          | 2026-09-01 20:09 |

### nemotron_h_moe

![nemotron_h_moe](img/family-nemotron_h_moe.svg)

| Model                  | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token | Δ decode   | Δ energy | Date             |
|------------------------|------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|---------|------------|----------|------------------|
| nemotron-3-nano:latest | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 3 245.8       | 72.7         | 128     | 4 804     |  -      |  -      |            |          | 2026-09-01 18:18 |
| nemotron-3-nano:latest | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **518 454.9** | **116.4**    | **128** | **1 322** | ** - ** | ** - ** | **+60.2%** |          | 2026-09-01 18:18 |
| nemotron-3-nano:latest | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 3 182.9       | 134.2        | 128     | 4 820     |  -      |  -      |            |          | 2026-09-01 16:41 |
| nemotron-3-nano:latest | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **260 325.1** | **144.4**    | **128** | **1 324** | ** - ** | ** - ** | **+7.7%**  |          | 2026-09-01 16:41 |
| nemotron-3-nano:latest | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 606.4         | 73.0         | 128     | 4 689     |  -      |  -      |            |          | 2026-09-01 15:05 |
| nemotron-3-nano:latest | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **128 303.2** | **120.4**    | **128** | **1 214** | ** - ** | ** - ** | **+65.0%** |          | 2026-09-01 15:05 |
| nemotron-3-nano:latest | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 604.8         | 136.0        | 128     | 4 713     |  -      |  -      |            |          | 2026-09-01 13:26 |
| nemotron-3-nano:latest | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **18 725.6**  | **144.4**    | **128** | **1 199** | ** - ** | ** - ** | **+6.2%**  |          | 2026-09-01 13:26 |
| nemotron-3-nano:latest | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 210.9         | 70.3         | 128     | 4 742     |  -      |  -      |            |          | 2026-09-01 11:48 |
| nemotron-3-nano:latest | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **17 399.8**  | **120.1**    | **128** | **1 211** | ** - ** | ** - ** | **+70.7%** |          | 2026-09-01 11:48 |
| nemotron-3-nano:latest | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 207.2         | 133.2        | 128     | 4 724     |  -      |  -      |            |          | 2026-09-01 10:10 |
| nemotron-3-nano:latest | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **28 150.1**  | **145.0**    | **128** | **1 204** | ** - ** | ** - ** | **+8.9%**  |          | 2026-09-01 10:10 |

### olmo2

![olmo2](img/family-olmo2.svg)

| Model    | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode   | Δ energy | Date             |
|----------|------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|------------|----------|------------------|
| olmo2:7b | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 14 726.3      | 87.8         | 128     | 1 460      |  -      |  -        |            |          | 2026-09-01 18:19 |
| olmo2:7b | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **9 156.6**   | **104.3**    | **128** | **1 232**  | ** - ** | ** - **   | **+18.8%** |          | 2026-09-01 18:19 |
| olmo2:7b | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 14 856.5      | 98.2         | 128     | 1 468      |  -      |  -        |            |          | 2026-09-01 16:42 |
| olmo2:7b | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **5 235.3**   | **114.8**    | **128** | **1 235**  | ** - ** | ** - **   | **+16.9%** |          | 2026-09-01 16:42 |
| olmo2:7b | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 2 957.5       | 87.5         | 128     | 1 464      |  -      |  -        |            |          | 2026-09-01 15:06 |
| olmo2:7b | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **3 065.5**   | **125.4**    | **128** | **1 025**  | ** - ** | ** - **   | **+43.4%** |          | 2026-09-01 15:06 |
| olmo2:7b | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 2 949.3       | 100.7        | 128     | 1 442      |  -      |  -        |            |          | 2026-09-01 13:27 |
| olmo2:7b | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **1 878.7**   | **134.2**    | **128** | **1 038**  | ** - ** | ** - **   | **+33.3%** |          | 2026-09-01 13:27 |
| olmo2:7b | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 1 056.8       | 89.2         | 128     | 1 436      |  -      |  -        |            |          | 2026-09-01 11:49 |
| olmo2:7b | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **716.7**     | **126.3**    | **128** | **1 017**  | ** - ** | ** - **   | **+41.6%** |          | 2026-09-01 11:49 |
| olmo2:7b | 4096 | short  | stream     | CPU    | Ollama 0.32.6   | 202.1         | 5.2          | 128     | 27 396     | 978     | 7.638     |            |          | 2026-08-18 01:14 |
| olmo2:7b | 4096 | short  | stream     | CPU    | **loken 0.1.0** | **60.4**      | **6.7**      | **91**  | **15 762** | **508** | **5.606** |            |          | 2026-08-18 01:14 |
| olmo2:7b | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 1 026.8       | 100.6        | 128     | 1 450      |  -      |  -        |            |          | 2026-09-01 10:11 |
| olmo2:7b | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **704.3**     | **133.9**    | **128** | **1 032**  | ** - ** | ** - **   | **+33.2%** |          | 2026-09-01 10:11 |

### olmoe

![olmoe](img/family-olmoe.svg)

| Model        | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms  | J/req   | J/token | Δ decode   | Δ energy | Date             |
|--------------|------|--------|------------|--------|-----------------|---------------|--------------|---------|---------|---------|---------|------------|----------|------------------|
| olmoe:latest | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 64 531.8      | 281.4        | 128     | 466     |  -      |  -      |            |          | 2026-09-01 18:20 |
| olmoe:latest | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **2 548.8**   | **278.1**    | **128** | **473** | ** - ** | ** - ** | -1.2%      |          | 2026-09-01 18:20 |
| olmoe:latest | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 64 612.8      | 381.1        | 128     | 448     |  -      |  -      |            |          | 2026-09-01 16:43 |
| olmoe:latest | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **2 018.5**   | **395.2**    | **128** | **566** | ** - ** | ** - ** | **+3.7%**  |          | 2026-09-01 16:43 |
| olmoe:latest | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 10 699.6      | 287.2        | 128     | 454     |  -      |  -      |            |          | 2026-09-01 15:07 |
| olmoe:latest | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **1 466.5**   | **330.0**    | **128** | **415** | ** - ** | ** - ** | **+14.9%** |          | 2026-09-01 15:07 |
| olmoe:latest | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 10 655.9      | 390.2        | 128     | 443     |  -      |  -      |            |          | 2026-09-01 13:29 |
| olmoe:latest | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **1 148.9**   | **428.9**    | **128** | **410** | ** - ** | ** - ** | **+9.9%**  |          | 2026-09-01 13:29 |
| olmoe:latest | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 3 651.5       | 296.4        | 128     | 436     |  -      |  -      |            |          | 2026-09-01 11:50 |
| olmoe:latest | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **590.4**     | **312.1**    | **128** | **458** | ** - ** | ** - ** | **+5.3%**  |          | 2026-09-01 11:50 |
| olmoe:latest | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 3 628.7       | 389.0        | 128     | 422     |  -      |  -      |            |          | 2026-09-01 10:12 |
| olmoe:latest | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **361.6**     | **440.7**    | **128** | **392** | ** - ** | ** - ** | **+13.3%** |          | 2026-09-01 10:12 |

### phi2

![phi2](img/family-phi2.svg)

| Model            | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms  | J/req   | J/token | Δ decode   | Δ energy | Date             |
|------------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|---------|---------|---------|------------|----------|------------------|
| moondream:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 34 459.4      |  -           | 1       | 112     |  -      |  -      | no answer  |          | 2026-09-01 18:16 |
| moondream:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 10 333.5      | 293.0        | 128     | 437     |  -      |  -      |            |          | 2026-09-01 15:02 |
| moondream:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **2 747.6**   | **467.5**    | **128** | **280** | ** - ** | ** - ** | **+59.5%** |          | 2026-09-01 15:02 |
| moondream:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 10 354.2      | 370.2        | 128     | 450     |  -      |  -      |            |          | 2026-09-01 13:24 |
| moondream:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **3 543.0**   | **591.5**    | **128** | **250** | ** - ** | ** - ** | **+59.8%** |          | 2026-09-01 13:24 |
| moondream:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 3 658.1       | 289.7        | 128     | 443     |  -      |  -      |            |          | 2026-09-01 11:46 |
| moondream:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **1 399.5**   | **508.6**    | **128** | **254** | ** - ** | ** - ** | **+75.5%** |          | 2026-09-01 11:46 |
| moondream:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 3 604.0       | 370.9        | 128     | 447     |  -      |  -      |            |          | 2026-09-01 10:08 |
| moondream:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **1 647.6**   | **598.1**    | **128** | **245** | ** - ** | ** - ** | **+61.2%** |          | 2026-09-01 10:08 |
| moondream:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 3 641.9       | 372.3        | 128     | 433     |  -      |  -      |            |          | 2026-09-01 20:09 |
| moondream:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **1 560.8**   | **574.5**    | **128** | **249** | ** - ** | ** - ** | **+54.3%** |          | 2026-09-01 20:09 |

### qwen2

![qwen2](img/family-qwen2.svg)

| Model            | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req     | J/token    | Δ decode     | Δ energy   | Date             |
|------------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|-----------|------------|--------------|------------|------------------|
| deepcoder:14b    | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 8 368.9       | 38.0         | 128     | 4 618      |  -        |  -         |              |            | 2026-09-01 17:08 |
| deepcoder:14b    | 4096   | long   | non-stream | GPU    | vLLM 0.22.0     |  -            | 80.6         | 128     | 1 593      |  -        |  -         |              |            | 2026-09-01 17:08 |
| deepcoder:14b    | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **3 593.7**   | **71.1**     | **128** | **1 801**  | ** - **   | ** - **    | -11.8%       |            | 2026-09-01 17:08 |
| deepcoder:14b    | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 8 371.4       | 53.3         | 128     | 4 605      |  -        |  -         |              |            | 2026-09-01 15:31 |
| deepcoder:14b    | 4096   | long   | stream     | GPU    | vLLM 0.22.0     |  -            | 85.0         | 128     | 1 569      |  -        |  -         |              |            | 2026-09-01 15:31 |
| deepcoder:14b    | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **3 214.1**   | **75.7**     | **128** | **1 813**  | ** - **   | ** - **    | -10.9%       |            | 2026-09-01 15:31 |
| deepcoder:14b    | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 1 707.8       | 37.7         | 128     | 4 596      |  -        |  -         |              |            | 2026-09-01 13:55 |
| deepcoder:14b    | 4096   | medium | non-stream | GPU    | vLLM 0.22.0     |  -            | 82.4         | 128     | 1 555      |  -        |  -         |              |            | 2026-09-01 13:55 |
| deepcoder:14b    | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **1 309.1**   | **77.3**     | **128** | **1 658**  | ** - **   | ** - **    | -6.2%        |            | 2026-09-01 13:55 |
| deepcoder:14b    | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 1 709.3       | 52.9         | 128     | 4 636      |  -        |  -         |              |            | 2026-09-01 12:17 |
| deepcoder:14b    | 4096   | medium | stream     | GPU    | vLLM 0.22.0     |  -            | 85.0         | 128     | 1 550      |  -        |  -         |              |            | 2026-09-01 12:17 |
| deepcoder:14b    | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 239.0**   | **80.4**     | **128** | **1 668**  | ** - **   | ** - **    | -5.5%        |            | 2026-09-01 12:17 |
| deepcoder:14b    | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 620.7         | 37.6         | 128     | 4 607      |  -        |  -         |              |            | 2026-09-01 10:39 |
| deepcoder:14b    | 4096   | short  | non-stream | GPU    | vLLM 0.22.0     |  -            | 82.7         | 128     | 1 549      |  -        |  -         |              |            | 2026-09-01 10:39 |
| deepcoder:14b    | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **465.6**     | **76.6**     | **128** | **1 671**  | ** - **   | ** - **    | -7.4%        |            | 2026-09-01 10:39 |
| deepcoder:14b    | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 44.7          | 2.6          | 128     | 52 337     | 1 718     | 13.418     |              |            | 2026-08-18 01:14 |
| deepcoder:14b    | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **23.4**      | **2.6**      | **128** | **50 305** | **1 636** | **12.782** | **+1.2%**    | **+5.0%**  | 2026-08-18 01:14 |
| deepcoder:14b    | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 642.2         | 52.6         | 128     | 4 639      |  -        |  -         |              |            | 2026-09-01 08:59 |
| deepcoder:14b    | 4096   | short  | stream     | GPU    | vLLM 0.22.0     |  -            | 85.1         | 128     | 1 531      |  -        |  -         |              |            | 2026-09-01 08:59 |
| deepcoder:14b    | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **427.1**     | **78.8**     | **128** | **1 694**  | ** - **   | ** - **    | -7.4%        |            | 2026-09-01 08:59 |
| deepcoder:14b    | 16384  | long   | stream     | GPU    | Ollama 0.32.6   | 11 703.7      | 75.4         | 128     | 3 681      | 557       | 4.351      |              |            | 2026-08-18 01:14 |
| deepcoder:14b    | 16384  | long   | stream     | GPU    | **loken 0.1.0** | **11 401.0**  | **77.3**     | **128** | **1 721**  | **441**   | **3.448**  | **+2.5%**    | **+26.2%** | 2026-08-18 01:14 |
| deepcoder:14b    | 32768  | long   | stream     | GPU    | Ollama 0.32.6   | 11 776.2      | 75.4         | 128     | 3 704      | 547       | 4.276      |              |            | 2026-08-18 01:14 |
| deepcoder:14b    | 32768  | long   | stream     | GPU    | **loken 0.1.0** | **8 435.7**   | **77.3**     | **128** | **1 731**  | **430**   | **3.359**  | **+2.5%**    | **+27.3%** | 2026-08-18 01:14 |
| deepcoder:14b    | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 163.7         | 12.8         | 128     | 13 140     |  -        |  -         |              |            | 2026-09-01 18:47 |
| deepcoder:14b    | 131072 | short  | stream     | GPU    | vLLM 0.22.0     |  -            | 85.0         | 128     | 1 551      |  -        |  -         |              |            | 2026-09-01 18:47 |
| deepcoder:14b    | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **271.9**     | **78.9**     | **128** | **1 700**  | ** - **   | ** - **    | -7.2%        |            | 2026-09-01 18:47 |
| deepcoder:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 1 106.8       | 74.4         | 128     | 3 644      | 544       | 4.250      |              |            | 2026-08-18 01:14 |
| deepcoder:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **1 206.0**   | **86.0**     | **128** | **1 532**  | **394**   | **3.081**  | **+15.7%**   | **+38.0%** | 2026-08-18 01:14 |
| deepseek-r1:32b  | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 4 200.5       | 19.6         | 128     | 8 218      |  -        |  -         |              |            | 2026-09-01 17:10 |
| deepseek-r1:32b  | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **581.2**     | **34.5**     | **128** | **3 733**  | ** - **   | ** - **    | **+76.5%**   |            | 2026-09-01 17:10 |
| deepseek-r1:32b  | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 4 215.1       | 25.9         | 128     | 8 094      |  -        |  -         |              |            | 2026-09-01 15:33 |
| deepseek-r1:32b  | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **559.4**     | **39.7**     | **128** | **3 796**  | ** - **   | ** - **    | **+53.4%**   |            | 2026-09-01 15:33 |
| deepseek-r1:32b  | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 864.6         | 19.7         | 128     | 8 089      |  -        |  -         |              |            | 2026-09-01 13:57 |
| deepseek-r1:32b  | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **352.2**     | **37.5**     | **128** | **3 452**  | ** - **   | ** - **    | **+90.4%**   |            | 2026-09-01 13:57 |
| deepseek-r1:32b  | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 893.8         | 25.9         | 128     | 8 158      |  -        |  -         |              |            | 2026-09-01 12:19 |
| deepseek-r1:32b  | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **397.3**     | **40.8**     | **128** | **3 473**  | ** - **   | ** - **    | **+57.7%**   |            | 2026-09-01 12:19 |
| deepseek-r1:32b  | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 317.6         | 19.6         | 128     | 8 267      |  -        |  -         |              |            | 2026-09-01 10:41 |
| deepseek-r1:32b  | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **230.7**     | **38.4**     | **128** | **3 364**  | ** - **   | ** - **    | **+96.2%**   |            | 2026-09-01 10:41 |
| deepseek-r1:32b  | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 335.8         | 25.8         | 128     | 8 179      |  -        |  -         |              |            | 2026-09-01 09:01 |
| deepseek-r1:32b  | 4096   | short  | stream     | GPU    | vLLM 0.22.0     |  -            | 36.9         | 128     | 3 490      | 811       | 6.334      |              |            | 2026-08-10 07:42 |
| deepseek-r1:32b  | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **234.4**     | **41.0**     | **128** | **3 432**  | ** - **   | ** - **    | **+11.2%**   |            | 2026-09-01 09:01 |
| deepseek-r1:32b  | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 32.3          | 2.5          | 128     | 56 889     |  -        |  -         |              |            | 2026-09-01 18:53 |
| deepseek-r1:32b  | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **185.6**     | **40.8**     | **128** | **3 440**  | ** - **   | ** - **    | **+1543.6%** |            | 2026-09-01 18:53 |
| qwen2.5:0.5b     | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   | 37 214.8      | 157.8        | 128     | 1 923      |  -        |  -         |              |            | 2026-09-01 18:21 |
| qwen2.5:0.5b     | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | **15 610.1**  | **312.7**    | **128** | **413**    | ** - **   | ** - **    | **+98.2%**   |            | 2026-09-01 18:21 |
| qwen2.5:0.5b     | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 36 052.9      | 312.3        | 128     | 1 900      |  -        |  -         |              |            | 2026-09-01 16:44 |
| qwen2.5:0.5b     | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **11 027.2**  | **352.9**    | **128** | **430**    | ** - **   | ** - **    | **+13.0%**   |            | 2026-09-01 16:44 |
| qwen2.5:0.5b     | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   | 8 422.9       | 151.9        | 128     | 1 945      |  -        |  -         |              |            | 2026-09-01 15:08 |
| qwen2.5:0.5b     | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | **3 814.6**   | **321.5**    | **128** | **408**    | ** - **   | ** - **    | **+111.7%**  |            | 2026-09-01 15:08 |
| qwen2.5:0.5b     | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 8 215.8       | 317.3        | 128     | 1 927      |  -        |  -         |              |            | 2026-09-01 13:29 |
| qwen2.5:0.5b     | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **2 008.6**   | **365.2**    | **128** | **414**    | ** - **   | ** - **    | **+15.1%**   |            | 2026-09-01 13:29 |
| qwen2.5:0.5b     | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   | 2 854.2       | 153.4        | 128     | 1 951      |  -        |  -         |              |            | 2026-09-01 11:51 |
| qwen2.5:0.5b     | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | **1 298.3**   | **332.6**    | **128** | **390**    | ** - **   | ** - **    | **+116.7%**  |            | 2026-09-01 11:51 |
| qwen2.5:0.5b     | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 1 594.1       | 49.7         | 35      | 2 181      | 121       | 3.433      |              |            | 2026-08-18 01:14 |
| qwen2.5:0.5b     | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **769.5**     | **49.1**     | **42**  | **989**    | **63**    | **1.489**  |              |            | 2026-08-18 01:14 |
| qwen2.5:0.5b     | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 2 829.9       | 303.6        | 128     | 1 923      |  -        |  -         |              |            | 2026-09-01 08:57 |
| qwen2.5:0.5b     | 4096   | short  | stream     | GPU    | vLLM 0.22.0     |  -            | 502.5        | 128     | 273        | 51        | 0.398      |              |            | 2026-08-10 07:11 |
| qwen2.5:0.5b     | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **784.7**     | **366.6**    | **128** | **415**    | ** - **   | ** - **    | -27.0%       |            | 2026-09-01 08:57 |

### qwen3

![qwen3](img/family-qwen3.svg)

| Model        | Ctx   | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode    | Δ energy    | Date             |
|--------------|-------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|-------------|-------------|------------------|
| qwen3:0.6b   | 4096  | long   | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent  |             | 2026-09-01 18:36 |
| qwen3:0.6b   | 4096  | long   | non-stream | GPU    | **loken 0.1.0** | **10 044.2**  | **439.6**    | **128** | **295**    | ** - ** | ** - **   |             |             | 2026-09-01 18:36 |
| qwen3:0.6b   | 4096  | long   | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent  |             | 2026-09-01 16:57 |
| qwen3:0.6b   | 4096  | long   | stream     | GPU    | **loken 0.1.0** | **10 151.3**  | **603.8**    | **128** | **298**    | ** - ** | ** - **   |             |             | 2026-09-01 16:57 |
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
| qwen3:8b     | 4096  | long   | non-stream | GPU    | Ollama 0.32.6   | 14 265.3      | 63.2         | 128     | 3 223      |  -      |  -        |             |             | 2026-09-01 18:38 |
| qwen3:8b     | 4096  | long   | non-stream | GPU    | vLLM 0.22.0     |  -            | 131.5        | 128     | 979        |  -      |  -        |             |             | 2026-09-01 18:38 |
| qwen3:8b     | 4096  | long   | non-stream | GPU    | **loken 0.1.0** | **6 740.9**   | **128.9**    | **128** | **996**    | ** - ** | ** - **   | -2.0%       |             | 2026-09-01 18:38 |
| qwen3:8b     | 4096  | long   | stream     | GPU    | Ollama 0.32.6   | 14 288.6      | 95.9         | 128     | 3 241      |  -      |  -        |             |             | 2026-09-01 16:59 |
| qwen3:8b     | 4096  | long   | stream     | GPU    | vLLM 0.22.0     |  -            | 139.4        | 128     | 970        |  -      |  -        |             |             | 2026-09-01 16:59 |
| qwen3:8b     | 4096  | long   | stream     | GPU    | **loken 0.1.0** | **4 539.6**   | **141.6**    | **128** | **1 012**  | ** - ** | ** - **   | **+1.5%**   |             | 2026-09-01 16:59 |
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

| Model          | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token | Δ decode    | Δ energy | Date             |
|----------------|------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|---------|-------------|----------|------------------|
| qwen3.5:latest | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 3 351.3       | 46.1         | 128     | 4 285     |  -      |  -      |             |          | 2026-09-01 18:35 |
| qwen3.5:latest | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **596 729.7** | **99.3**     | **128** | **1 355** | ** - ** | ** - ** | **+115.1%** |          | 2026-09-01 18:35 |
| qwen3.5:latest | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 3 482.4       | 76.0         | 128     | 4 262     |  -      |  -      |             |          | 2026-09-01 16:56 |
| qwen3.5:latest | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **61 267.8**  | **114.7**    | **128** | **1 405** | ** - ** | ** - ** | **+50.9%**  |          | 2026-09-01 16:56 |
| qwen3.5:latest | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 615.9         | 46.2         | 128     | 4 211     |  -      |  -      |             |          | 2026-09-01 15:19 |
| qwen3.5:latest | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **98 697.8**  | **108.5**    | **128** | **1 242** | ** - ** | ** - ** | **+135.0%** |          | 2026-09-01 15:19 |
| qwen3.5:latest | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 572.7         | 76.3         | 128     | 4 281     |  -      |  -      |             |          | 2026-09-01 13:43 |
| qwen3.5:latest | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **43 852.0**  | **123.7**    | **128** | **1 287** | ** - ** | ** - ** | **+62.2%**  |          | 2026-09-01 13:43 |
| qwen3.5:latest | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 198.6         | 45.5         | 128     | 4 292     |  -      |  -      |             |          | 2026-09-01 12:05 |
| qwen3.5:latest | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **38 476.4**  | **107.3**    | **128** | **1 272** | ** - ** | ** - ** | **+135.7%** |          | 2026-09-01 12:05 |
| qwen3.5:latest | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 213.8         | 75.2         | 128     | 4 250     |  -      |  -      |             |          | 2026-09-01 10:25 |
| qwen3.5:latest | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **25 974.3**  | **126.1**    | **128** | **1 267** | ** - ** | ** - ** | **+67.6%**  |          | 2026-09-01 10:25 |

### qwen35moe

![qwen35moe](img/family-qwen35moe.svg)

| Model       | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token | Δ decode   | Δ energy | Date             |
|-------------|------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|---------|------------|----------|------------------|
| qwen3.5:35b | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 2 780.1       | 57.7         | 128     | 20 325    |  -      |  -      |            |          | 2026-09-01 18:34 |
| qwen3.5:35b | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **388 582.0** | **102.2**    | **128** | **1 475** | ** - ** | ** - ** | **+77.2%** |          | 2026-09-01 18:34 |
| qwen3.5:35b | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 2 773.7       | 116.9        | 128     | 4 980     |  -      |  -      |            |          | 2026-09-01 16:55 |
| qwen3.5:35b | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **147 800.7** | **125.2**    | **128** | **1 484** | ** - ** | ** - ** | **+7.1%**  |          | 2026-09-01 16:55 |
| qwen3.5:35b | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 477.3         | 58.7         | 128     | 5 103     |  -      |  -      |            |          | 2026-09-01 15:18 |
| qwen3.5:35b | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **19 057.3**  | **107.0**    | **128** | **1 355** | ** - ** | ** - ** | **+82.3%** |          | 2026-09-01 15:18 |
| qwen3.5:35b | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 481.7         | 113.0        | 128     | 20 424    |  -      |  -      |            |          | 2026-09-01 13:42 |
| qwen3.5:35b | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **35 323.6**  | **133.2**    | **128** | **1 366** | ** - ** | ** - ** | **+17.8%** |          | 2026-09-01 13:42 |
| qwen3.5:35b | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 166.6         | 55.9         | 128     | 20 500    |  -      |  -      |            |          | 2026-09-01 12:03 |
| qwen3.5:35b | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **20 501.2**  | **108.7**    | **128** | **1 325** | ** - ** | ** - ** | **+94.4%** |          | 2026-09-01 12:03 |
| qwen3.5:35b | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 165.6         | 114.1        | 128     | 5 100     |  -      |  -      |            |          | 2026-09-01 10:24 |
| qwen3.5:35b | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **13 220.2**  | **134.3**    | **128** | **1 308** | ** - ** | ** - ** | **+17.6%** |          | 2026-09-01 10:24 |

### qwen3moe

![qwen3moe](img/family-qwen3moe.svg)

| Model           | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token | Δ decode   | Δ energy | Date             |
|-----------------|------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|---------|------------|----------|------------------|
| qwen3-coder:30b | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 17 921.0      | 83.3         | 128     | 4 047     |  -      |  -      |            |          | 2026-09-01 18:30 |
| qwen3-coder:30b | 4096 | long   | non-stream | GPU    | vLLM 0.22.0     |  -            | 129.0        | 128     | 1 004     |  -      |  -      |            |          | 2026-09-01 18:30 |
| qwen3-coder:30b | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **492.4**     | **97.1**     | **128** | **1 495** | ** - ** | ** - ** | -24.7%     |          | 2026-09-01 18:30 |
| qwen3-coder:30b | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 17 997.0      | 145.3        | 128     | 3 986     |  -      |  -      |            |          | 2026-09-01 16:52 |
| qwen3-coder:30b | 4096 | long   | stream     | GPU    | vLLM 0.22.0     |  -            | 138.7        | 128     | 1 017     |  -      |  -      |            |          | 2026-09-01 16:52 |
| qwen3-coder:30b | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **482.1**     | **173.8**    | **128** | **1 524** | ** - ** | ** - ** | **+19.6%** |          | 2026-09-01 16:52 |
| qwen3-coder:30b | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 4 172.8       | 85.3         | 128     | 3 983     |  -      |  -      |            |          | 2026-09-01 15:16 |
| qwen3-coder:30b | 4096 | medium | non-stream | GPU    | vLLM 0.22.0     |  -            | 127.6        | 128     | 1 009     |  -      |  -      |            |          | 2026-09-01 15:16 |
| qwen3-coder:30b | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **344.2**     | **127.8**    | **128** | **1 201** | ** - ** | ** - ** | **+0.1%**  |          | 2026-09-01 15:16 |
| qwen3-coder:30b | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 4 133.8       | 146.5        | 128     | 3 988     |  -      |  -      |            |          | 2026-09-01 13:38 |
| qwen3-coder:30b | 4096 | medium | stream     | GPU    | vLLM 0.22.0     |  -            | 138.7        | 128     | 995       |  -      |  -      |            |          | 2026-09-01 13:38 |
| qwen3-coder:30b | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **383.8**     | **177.4**    | **128** | **1 211** | ** - ** | ** - ** | **+21.1%** |          | 2026-09-01 13:38 |
| qwen3-coder:30b | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 1 435.7       | 85.9         | 128     | 3 943     |  -      |  -      |            |          | 2026-09-01 12:00 |
| qwen3-coder:30b | 4096 | short  | non-stream | GPU    | vLLM 0.22.0     |  -            | 129.7        | 128     | 996       |  -      |  -      |            |          | 2026-09-01 12:00 |
| qwen3-coder:30b | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **248.7**     | **134.8**    | **128** | **1 170** | ** - ** | ** - ** | **+3.9%**  |          | 2026-09-01 12:00 |
| qwen3-coder:30b | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 1 430.0       | 147.4        | 128     | 3 967     |  -      |  -      |            |          | 2026-09-01 10:22 |
| qwen3-coder:30b | 4096 | short  | stream     | GPU    | vLLM 0.22.0     |  -            | 141.1        | 128     | 968       |  -      |  -      |            |          | 2026-09-01 10:22 |
| qwen3-coder:30b | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **263.9**     | **178.8**    | **128** | **1 178** | ** - ** | ** - ** | **+21.3%** |          | 2026-09-01 10:22 |

### qwen3next

![qwen3next](img/family-qwen3next.svg)

| Model                   | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token | Δ decode | Δ energy | Date             |
|-------------------------|------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|---------|----------|----------|------------------|
| qwen3-coder-next:latest | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 1 066.2       | 19.4         | 128     | 10 999     |  -      |  -      |          |          | 2026-09-01 18:26 |
| qwen3-coder-next:latest | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **6 668.9**   | **8.8**      | **128** | **25 960** | ** - ** | ** - ** | -54.7%   |          | 2026-09-01 18:26 |
| qwen3-coder-next:latest | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 1 061.4       | 28.4         | 128     | 11 030     |  -      |  -      |          |          | 2026-09-01 16:49 |
| qwen3-coder-next:latest | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **7 588.6**   | **9.5**      | **128** | **22 871** | ** - ** | ** - ** | -66.6%   |          | 2026-09-01 16:49 |
| qwen3-coder-next:latest | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 216.7         | 19.4         | 128     | 10 592     |  -      |  -      |          |          | 2026-09-01 15:12 |
| qwen3-coder-next:latest | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **594.5**     | **8.2**      | **21**  | **11 599** | ** - ** | ** - ** |          |          | 2026-09-01 15:12 |
| qwen3-coder-next:latest | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 215.2         | 28.4         | 128     | 10 639     |  -      |  -      |          |          | 2026-09-01 13:34 |
| qwen3-coder-next:latest | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **1 011.2**   | **9.1**      | **128** | **23 020** | ** - ** | ** - ** | -68.1%   |          | 2026-09-01 13:34 |
| qwen3-coder-next:latest | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 81.9          | 19.4         | 128     | 10 435     |  -      |  -      |          |          | 2026-09-01 11:56 |
| qwen3-coder-next:latest | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **409.3**     | **9.1**      | **128** | **23 265** | ** - ** | ** - ** | -53.0%   |          | 2026-09-01 11:56 |
| qwen3-coder-next:latest | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 80.1          | 28.1         | 128     | 10 488     |  -      |  -      |          |          | 2026-09-01 10:17 |
| qwen3-coder-next:latest | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **305.3**     | **9.2**      | **128** | **23 157** | ** - ** | ** - ** | -67.1%   |          | 2026-09-01 10:17 |
| qwen3next:latest        | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 1 144.9       | 22.2         | 128     | 10 157     |  -      |  -      |          |          | 2026-09-01 18:43 |
| qwen3next:latest        | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **5 944.3**   | **9.3**      | **128** | **23 860** | ** - ** | ** - ** | -58.0%   |          | 2026-09-01 18:43 |
| qwen3next:latest        | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 1 148.4       | 32.6         | 128     | 10 165     |  -      |  -      |          |          | 2026-09-01 17:04 |
| qwen3next:latest        | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **8 076.2**   | **9.7**      | **128** | **22 664** | ** - ** | ** - ** | -70.4%   |          | 2026-09-01 17:04 |
| qwen3next:latest        | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 231.7         | 22.2         | 128     | 9 826      |  -      |  -      |          |          | 2026-09-01 15:27 |
| qwen3next:latest        | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **1 239.8**   | **9.5**      | **128** | **21 099** | ** - ** | ** - ** | -57.1%   |          | 2026-09-01 15:27 |
| qwen3next:latest        | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 228.8         | 32.5         | 128     | 9 935      |  -      |  -      |          |          | 2026-09-01 13:51 |
| qwen3next:latest        | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **1 201.3**   | **9.5**      | **128** | **23 900** | ** - ** | ** - ** | -70.7%   |          | 2026-09-01 13:51 |
| qwen3next:latest        | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 86.4          | 22.2         | 128     | 9 696      |  -      |  -      |          |          | 2026-09-01 12:13 |
| qwen3next:latest        | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **318.9**     | **9.6**      | **128** | **21 282** | ** - ** | ** - ** | -56.6%   |          | 2026-09-01 12:13 |
| qwen3next:latest        | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 86.4          | 32.9         | 128     | 9 659      |  -      |  -      |          |          | 2026-09-01 10:34 |
| qwen3next:latest        | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **187.8**     | **9.9**      | **128** | **20 982** | ** - ** | ** - ** | -70.0%   |          | 2026-09-01 10:34 |

### smollm3

![smollm3](img/family-smollm3.svg)

| Model          | Ctx  | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode    | Δ energy | Date             |
|----------------|------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|-------------|----------|------------------|
| smollm3:latest | 4096 | long   | non-stream | GPU    | Ollama 0.32.6   | 22 998.1      | 96.6         | 128     | 2 454      |  -      |  -        |             |          | 2026-09-01 18:44 |
| smollm3:latest | 4096 | long   | non-stream | GPU    | **loken 0.1.0** | **14 774.0**  | **225.2**    | **128** | **581**    | ** - ** | ** - **   | **+133.2%** |          | 2026-09-01 18:44 |
| smollm3:latest | 4096 | long   | stream     | GPU    | Ollama 0.32.6   | 23 057.4      | 171.4        | 128     | 2 460      |  -      |  -        |             |          | 2026-09-01 17:05 |
| smollm3:latest | 4096 | long   | stream     | GPU    | **loken 0.1.0** | **9 457.0**   | **262.9**    | **128** | **584**    | ** - ** | ** - **   | **+53.4%**  |          | 2026-09-01 17:05 |
| smollm3:latest | 4096 | medium | non-stream | GPU    | Ollama 0.32.6   | 4 742.2       | 93.7         | 128     | 2 497      |  -      |  -        |             |          | 2026-09-01 15:28 |
| smollm3:latest | 4096 | medium | non-stream | GPU    | **loken 0.1.0** | **2 952.8**   | **224.4**    | **128** | **581**    | ** - ** | ** - **   | **+139.4%** |          | 2026-09-01 15:28 |
| smollm3:latest | 4096 | medium | stream     | GPU    | Ollama 0.32.6   | 4 768.2       | 169.6        | 128     | 2 500      |  -      |  -        |             |          | 2026-09-01 13:52 |
| smollm3:latest | 4096 | medium | stream     | GPU    | **loken 0.1.0** | **915.8**     | **251.8**    | **128** | **616**    | ** - ** | ** - **   | **+48.5%**  |          | 2026-09-01 13:52 |
| smollm3:latest | 4096 | short  | non-stream | GPU    | Ollama 0.32.6   | 1 676.7       | 96.3         | 128     | 2 461      |  -      |  -        |             |          | 2026-09-01 12:14 |
| smollm3:latest | 4096 | short  | non-stream | GPU    | **loken 0.1.0** | **350.8**     | **214.0**    | **128** | **610**    | ** - ** | ** - **   | **+122.2%** |          | 2026-09-01 12:14 |
| smollm3:latest | 4096 | short  | stream     | CPU    | Ollama 0.32.6   | 693.1         | 11.0         | 128     | 13 711     | 504     | 3.935     |             |          | 2026-08-18 01:14 |
| smollm3:latest | 4096 | short  | stream     | CPU    | **loken 0.1.0** | **140.4**     | **10.6**     | **128** | **12 517** | **558** | **4.358** | -3.5%       | -9.7%    | 2026-08-18 01:14 |
| smollm3:latest | 4096 | short  | stream     | GPU    | Ollama 0.32.6   | 1 687.1       | 170.9        | 128     | 2 491      |  -      |  -        |             |          | 2026-09-01 10:36 |
| smollm3:latest | 4096 | short  | stream     | GPU    | **loken 0.1.0** | **694.5**     | **256.7**    | **128** | **578**    | ** - ** | ** - **   | **+50.2%**  |          | 2026-09-01 10:36 |
