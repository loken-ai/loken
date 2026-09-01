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


### ernie4_5

![ernie4_5](img/family-ernie4_5.svg)

| Model           | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token   | Δ decode    | Δ energy | Date             |
|-----------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|-----------|-------------|----------|------------------|
| ernie4-5:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 213.1        | 128     | 1 591     |  -      |  -        |             |          | 2026-09-01 17:44 |
| ernie4-5:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **375.0**    | **75**  | **208**   | ** - ** | ** - **   |             |          | 2026-09-01 17:44 |
| ernie4-5:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 3 029.2       | 406.5        | 128     | 1 585     |  -      |  -        |             |          | 2026-09-01 16:06 |
| ernie4-5:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **19 333.0**  | **570.5**    | **75**  | **210**   | ** - ** | ** - **   |             |          | 2026-09-01 16:06 |
| ernie4-5:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 207.7        | 80      | 1 462     |  -      |  -        |             |          | 2026-09-01 14:30 |
| ernie4-5:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **483.5**    | **128** | **273**   | ** - ** | ** - **   |             |          | 2026-09-01 14:30 |
| ernie4-5:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 459.8         | 405.6        | 80      | 1 501     |  -      |  -        |             |          | 2026-09-01 12:52 |
| ernie4-5:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **5 233.5**   | **570.7**    | **128** | **287**   | ** - ** | ** - **   |             |          | 2026-09-01 12:52 |
| ernie4-5:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 226.2        | 128     | 1 588     |  -      |  -        |             |          | 2026-09-01 11:13 |
| ernie4-5:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **496.0**    | **128** | **264**   | ** - ** | ** - **   | **+119.3%** |          | 2026-09-01 11:13 |
| ernie4-5:latest | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 5 196.2       | 52.7         | 128     | 3 614     | 165     | 1.286     |             |          | 2026-08-18 01:14 |
| ernie4-5:latest | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **4 224.4**   | **50.2**     | **104** | **2 180** | **95**  | **0.915** |             |          | 2026-08-18 01:14 |
| ernie4-5:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 157.2         | 418.9        | 128     | 1 578     |  -      |  -        |             |          | 2026-09-01 09:35 |
| ernie4-5:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **2 226.9**   | **600.1**    | **128** | **267**   | ** - ** | ** - **   | **+43.2%**  |          | 2026-09-01 09:35 |
| ernie4-5:latest | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 465.8         | 403.5        | 80      | 1 660     |  -      |  -        |             |          | 2026-09-02 00:02 |
| ernie4-5:latest | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **9 004.7**   | **538.8**    | **128** | **294**   | ** - ** | ** - **   |             |          | 2026-09-02 00:02 |
| ernie4-5:latest | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 221.3        | 128     | 1 737     |  -      |  -        |             |          | 2026-09-01 21:30 |
| ernie4-5:latest | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **501.5**    | **128** | **262**   | ** - ** | ** - **   | **+126.6%** |          | 2026-09-01 21:30 |
| ernie4-5:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 158.2         | 418.5        | 128     | 1 749     |  -      |  -        |             |          | 2026-09-01 19:36 |
| ernie4-5:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **2 278.6**   | **579.9**    | **128** | **274**   | ** - ** | ** - **   | **+38.6%**  |          | 2026-09-01 19:36 |

### gemma4

![gemma4](img/family-gemma4.svg)

| Model         | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode    | Δ energy   | Date             |
|---------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|-------------|------------|------------------|
| gemma4:12b    | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 32.9         | 128     | 5 234      |  -      |  -        |             |            | 2026-09-01 17:46 |
| gemma4:12b    | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **65.3**     | **128** | **1 961**  | ** - ** | ** - **   | **+98.5%**  |            | 2026-09-01 17:46 |
| gemma4:12b    | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 451.3         | 53.3         | 128     | 5 236      |  -      |  -        |             |            | 2026-09-01 16:09 |
| gemma4:12b    | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **2 126.9**   | **67.9**     | **128** | **2 033**  | ** - ** | ** - **   | **+27.4%**  |            | 2026-09-01 16:09 |
| gemma4:12b    | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent  |            | 2026-09-01 14:33 |
| gemma4:12b    | 4096   | medium | non-stream | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent  |            | 2026-09-01 14:33 |
| gemma4:12b    | 4096   | medium | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 74      |  -         |  -      |  -        | incoherent  |            | 2026-09-01 12:54 |
| gemma4:12b    | 4096   | medium | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent  |            | 2026-09-01 12:54 |
| gemma4:12b    | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 22.4         | 61      | 3 886      |  -      |  -        |             |            | 2026-09-01 11:16 |
| gemma4:12b    | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **71.8**     | **128** | **1 784**  | ** - ** | ** - **   |             |            | 2026-09-01 11:16 |
| gemma4:12b    | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 29.1          | 42.6         | 61      | 5 331      |  -      |  -        |             |            | 2026-09-01 09:37 |
| gemma4:12b    | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **237.8**     | **71.9**     | **128** | **1 837**  | ** - ** | ** - **   |             |            | 2026-09-01 09:37 |
| gemma4:12b    | 131072 | medium | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 74      |  -         |  -      |  -        | incoherent  |            | 2026-09-02 00:05 |
| gemma4:12b    | 131072 | medium | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent  |            | 2026-09-02 00:05 |
| gemma4:12b    | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 22.3         | 61      | 4 038      |  -      |  -        |             |            | 2026-09-01 21:32 |
| gemma4:12b    | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **71.3**     | **128** | **1 795**  | ** - ** | ** - **   |             |            | 2026-09-01 21:32 |
| gemma4:12b    | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 29.6          | 54.4         | 61      | 4 027      |  -      |  -        |             |            | 2026-09-01 19:38 |
| gemma4:12b    | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **253.2**     | **71.8**     | **128** | **1 835**  | ** - ** | ** - **   |             |            | 2026-09-01 19:38 |
| gemma4:26b    | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 49.7         | 128     | 5 032      |  -      |  -        |             |            | 2026-09-01 17:48 |
| gemma4:26b    | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **73.1**     | **128** | **1 773**  | ** - ** | ** - **   | **+46.9%**  |            | 2026-09-01 17:48 |
| gemma4:26b    | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 499.3         | 99.2         | 128     | 5 022      |  -      |  -        |             |            | 2026-09-01 16:11 |
| gemma4:26b    | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **1 100.4**   | **90.7**     | **128** | **1 770**  | ** - ** | ** - **   | -8.6%       |            | 2026-09-01 16:11 |
| gemma4:26b    | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent  |            | 2026-09-01 14:35 |
| gemma4:26b    | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **85.4**     | **128** | **1 519**  | ** - ** | ** - **   |             |            | 2026-09-01 14:35 |
| gemma4:26b    | 4096   | medium | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent  |            | 2026-09-01 12:57 |
| gemma4:26b    | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **399.0**     | **95.4**     | **128** | **1 565**  | ** - ** | ** - **   |             |            | 2026-09-01 12:57 |
| gemma4:26b    | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 11.4         | 128     | 4 582      |  -      |  -        |             |            | 2026-09-01 11:18 |
| gemma4:26b    | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **88.4**     | **128** | **1 463**  | ** - ** | ** - **   | **+676.5%** |            | 2026-09-01 11:18 |
| gemma4:26b    | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 32.0          | 101.3        | 93      | 5 661      |  -      |  -        |             |            | 2026-09-01 09:40 |
| gemma4:26b    | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **200.3**     | **95.5**     | **128** | **1 519**  | ** - ** | ** - **   |             |            | 2026-09-01 09:40 |
| gemma4:26b    | 131072 | medium | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent  |            | 2026-09-02 00:07 |
| gemma4:26b    | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **399.8**     | **96.1**     | **128** | **1 544**  | ** - ** | ** - **   |             |            | 2026-09-02 00:07 |
| gemma4:26b    | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 11.1         | 128     | 4 682      |  -      |  -        |             |            | 2026-09-01 21:34 |
| gemma4:26b    | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **87.8**     | **128** | **1 475**  | ** - ** | ** - **   | **+690.6%** |            | 2026-09-01 21:34 |
| gemma4:26b    | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 29.5          | 101.4        | 93      | 4 697      |  -      |  -        |             |            | 2026-09-01 19:40 |
| gemma4:26b    | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **206.0**     | **96.2**     | **128** | **1 533**  | ** - ** | ** - **   |             |            | 2026-09-01 19:40 |
| gemma4:31b    | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           |  -      |  -         |  -      |  -        | incoherent  |            | 2026-09-01 17:51 |
| gemma4:31b    | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **21.6**     | **128** | **5 921**  | ** - ** | ** - **   |             |            | 2026-09-01 17:51 |
| gemma4:31b    | 4096   | long   | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 44      |  -         |  -      |  -        | incoherent  |            | 2026-09-01 16:14 |
| gemma4:31b    | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **803.2**     | **22.9**     | **128** | **5 975**  | ** - ** | ** - **   |             |            | 2026-09-01 16:14 |
| gemma4:31b    | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           |  -      | 5 832      |  -      |  -        |             |            | 2026-09-01 14:37 |
| gemma4:31b    | 4096   | medium | non-stream | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent  |            | 2026-09-01 14:37 |
| gemma4:31b    | 4096   | medium | stream     | GPU    | Ollama 0.32.6   |  -            | 25.4         | 48      | 5 908      |  -      |  -        |             |            | 2026-09-01 12:59 |
| gemma4:31b    | 4096   | medium | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent  |            | 2026-09-01 12:59 |
| gemma4:31b    | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 8.3          | 128     | 7 646      |  -      |  -        |             |            | 2026-09-01 11:21 |
| gemma4:31b    | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **23.9**     | **128** | **5 370**  | ** - ** | ** - **   | **+188.0%** |            | 2026-09-01 11:21 |
| gemma4:31b    | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 24.8          | 25.7         | 96      | 7 652      |  -      |  -        |             |            | 2026-09-01 09:42 |
| gemma4:31b    | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **169.3**     | **24.4**     | **128** | **5 407**  | ** - ** | ** - **   |             |            | 2026-09-01 09:42 |
| gemma4:31b    | 131072 | medium | stream     | GPU    | Ollama 0.32.6   |  -            | 5.0          | 82      | 22 180     |  -      |  -        |             |            | 2026-09-02 00:11 |
| gemma4:31b    | 131072 | medium | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent  |            | 2026-09-02 00:11 |
| gemma4:31b    | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           |  -      | 18 389     |  -      |  -        |             |            | 2026-09-01 21:38 |
| gemma4:31b    | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **23.9**     | **128** | **5 359**  | ** - ** | ** - **   |             |            | 2026-09-01 21:38 |
| gemma4:31b    | 131072 | short  | stream     | GPU    | Ollama 0.32.6   |  -            | 5.1          | 63      | 18 287     |  -      |  -        |             |            | 2026-09-01 19:43 |
| gemma4:31b    | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **163.7**     | **24.3**     | **128** | **5 404**  | ** - ** | ** - **   |             |            | 2026-09-01 19:43 |
| gemma4:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 49.0         | 128     | 4 060      |  -      |  -        |             |            | 2026-09-01 17:53 |
| gemma4:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **111.8**    | **128** | **1 145**  | ** - ** | ** - **   | **+128.0%** |            | 2026-09-01 17:53 |
| gemma4:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 550.1         | 89.6         | 128     | 4 029      |  -      |  -        |             |            | 2026-09-01 16:15 |
| gemma4:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **3 280.1**   | **121.9**    | **128** | **1 163**  | ** - ** | ** - **   | **+36.1%**  |            | 2026-09-01 16:15 |
| gemma4:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 49.5         | 128     | 4 096      |  -      |  -        |             |            | 2026-09-01 14:39 |
| gemma4:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **119.3**    | **128** | **1 074**  | ** - ** | ** - **   | **+141.0%** |            | 2026-09-01 14:39 |
| gemma4:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 89.1          | 90.2         | 128     | 3 965      |  -      |  -        |             |            | 2026-09-01 13:01 |
| gemma4:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **503.3**     | **127.1**    | **128** | **1 113**  | ** - ** | ** - **   | **+40.9%**  |            | 2026-09-01 13:01 |
| gemma4:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 50.1         | 128     | 3 965      |  -      |  -        |             |            | 2026-09-01 11:22 |
| gemma4:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **121.8**    | **128** | **1 051**  | ** - ** | ** - **   | **+142.9%** |            | 2026-09-01 11:22 |
| gemma4:latest | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 91.4          | 7.0          | 128     | 21 777     | 765     | 5.980     |             |            | 2026-08-18 01:14 |
| gemma4:latest | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **113.9**     | **6.7**      | **128** | **19 608** | **652** | **5.090** | -4.6%       | **+17.5%** | 2026-08-18 01:14 |
| gemma4:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 31.8          | 91.1         | 128     | 3 981      |  -      |  -        |             |            | 2026-09-01 09:44 |
| gemma4:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **275.5**     | **128.4**    | **128** | **1 081**  | ** - ** | ** - **   | **+40.9%**  |            | 2026-09-01 09:44 |
| gemma4:latest | 16384  | long   | stream     | GPU    | Ollama 0.32.6   | 10 891.8      | 119.0        | 128     | 3 811      | 349     | 2.726     |             |            | 2026-08-18 01:14 |
| gemma4:latest | 16384  | long   | stream     | GPU    | **loken 0.1.0** | **11 723.1**  | **124.8**    | **128** | **1 124**  | **194** | **1.515** | **+4.9%**   | **+79.9%** | 2026-08-18 01:14 |
| gemma4:latest | 16384  | short  | stream     | GPU    | Ollama 0.32.6   | 1 282.3       | 118.7        | 128     | 3 806      | 345     | 2.697     |             |            | 2026-08-18 01:14 |
| gemma4:latest | 16384  | short  | stream     | GPU    | **loken 0.1.0** | **2 067.0**   | **131.0**    | **128** | **1 006**  | **180** | **1.410** | **+10.4%**  | **+91.3%** | 2026-08-18 01:14 |
| gemma4:latest | 32768  | long   | stream     | GPU    | Ollama 0.32.6   | 10 903.8      | 119.0        | 128     | 3 785      | 341     | 2.667     |             |            | 2026-08-18 01:14 |
| gemma4:latest | 32768  | long   | stream     | GPU    | **loken 0.1.0** | **5 961.5**   | **126.1**    | **128** | **1 095**  | **199** | **1.552** | **+6.0%**   | **+71.9%** | 2026-08-18 01:14 |
| gemma4:latest | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 87.6          | 89.8         | 128     | 4 065      |  -      |  -        |             |            | 2026-09-02 00:13 |
| gemma4:latest | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **739.4**     | **126.5**    | **128** | **1 113**  | ** - ** | ** - **   | **+40.9%**  |            | 2026-09-02 00:13 |
| gemma4:latest | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 50.4         | 128     | 4 029      |  -      |  -        |             |            | 2026-09-01 21:40 |
| gemma4:latest | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **122.2**    | **128** | **1 049**  | ** - ** | ** - **   | **+142.6%** |            | 2026-09-01 21:40 |
| gemma4:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 31.8          | 91.0         | 128     | 4 048      |  -      |  -        |             |            | 2026-09-01 19:45 |
| gemma4:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **181.5**     | **129.2**    | **128** | **1 099**  | ** - ** | ** - **   | **+42.1%**  |            | 2026-09-01 19:45 |

### gptoss

![gptoss](img/family-gptoss.svg)

| Model       | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms  | J/req   | J/token | Δ decode    | Δ energy | Date             |
|-------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|---------|---------|---------|-------------|----------|------------------|
| gpt-oss:20b | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 53.3         | 128     | 4 430   |  -      |  -      |             |          | 2026-09-01 18:05 |
| gpt-oss:20b | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **166.0**    | **128** | **899** | ** - ** | ** - ** | **+211.3%** |          | 2026-09-01 18:05 |
| gpt-oss:20b | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 541.3         | 95.2         | 128     | 4 387   |  -      |  -      |             |          | 2026-09-01 16:27 |
| gpt-oss:20b | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **10 676.9**  | **210.1**    | **128** | **905** | ** - ** | ** - ** | **+120.7%** |          | 2026-09-01 16:27 |
| gpt-oss:20b | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 52.3         | 128     | 4 408   |  -      |  -      |             |          | 2026-09-01 14:51 |
| gpt-oss:20b | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **172.1**    | **128** | **822** | ** - ** | ** - ** | **+229.1%** |          | 2026-09-01 14:51 |
| gpt-oss:20b | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 98.5          | 93.5         | 128     | 4 434   |  -      |  -      |             |          | 2026-09-01 13:13 |
| gpt-oss:20b | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 349.8**   | **212.4**    | **128** | **831** | ** - ** | ** - ** | **+127.3%** |          | 2026-09-01 13:13 |
| gpt-oss:20b | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 52.7         | 128     | 4 385   |  -      |  -      |             |          | 2026-09-01 11:34 |
| gpt-oss:20b | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **175.1**    | **128** | **809** | ** - ** | ** - ** | **+232.2%** |          | 2026-09-01 11:34 |
| gpt-oss:20b | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 34.9          | 94.1         | 128     | 4 472   |  -      |  -      |             |          | 2026-09-01 09:56 |
| gpt-oss:20b | 4096   | short  | stream     | GPU    | vLLM 0.22.0     | 667.3         | 146.9        | 128     | 887     | 189     | 1.476   |             |          | 2026-08-10 07:13 |
| gpt-oss:20b | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **460.8**     | **211.9**    | **128** | **826** | ** - ** | ** - ** | **+44.2%**  |          | 2026-09-01 09:56 |
| gpt-oss:20b | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 100.4         | 93.5         | 128     | 4 648   |  -      |  -      |             |          | 2026-09-02 00:25 |
| gpt-oss:20b | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **1 795.4**   | **212.8**    | **128** | **851** | ** - ** | ** - ** | **+127.6%** |          | 2026-09-02 00:25 |
| gpt-oss:20b | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 52.7         | 128     | 4 515   |  -      |  -      |             |          | 2026-09-01 21:52 |
| gpt-oss:20b | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **168.9**    | **128** | **828** | ** - ** | ** - ** | **+220.6%** |          | 2026-09-01 21:52 |
| gpt-oss:20b | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 34.7          | 94.6         | 128     | 4 557   |  -      |  -      |             |          | 2026-09-01 19:57 |
| gpt-oss:20b | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **448.1**     | **212.2**    | **128** | **825** | ** - ** | ** - ** | **+124.4%** |          | 2026-09-01 19:57 |

### granite

![granite](img/family-granite.svg)

| Model               | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode   | Δ energy    | Date             |
|---------------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|------------|-------------|------------------|
| granite3.1-dense:2b | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 143.9        | 128     | 1 922      |  -      |  -        |            |             | 2026-09-01 18:06 |
| granite3.1-dense:2b | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **225.6**    | **128** | **570**    | ** - ** | ** - **   | **+56.8%** |             | 2026-09-01 18:06 |
| granite3.1-dense:2b | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 3 365.4       | 221.8        | 128     | 1 887      |  -      |  -        |            |             | 2026-09-01 16:29 |
| granite3.1-dense:2b | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **4 810.1**   | **260.0**    | **128** | **576**    | ** - ** | ** - **   | **+17.2%** |             | 2026-09-01 16:29 |
| granite3.1-dense:2b | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 142.8        | 128     | 1 891      |  -      |  -        |            |             | 2026-09-01 14:53 |
| granite3.1-dense:2b | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **243.1**    | **128** | **529**    | ** - ** | ** - **   | **+70.3%** |             | 2026-09-01 14:53 |
| granite3.1-dense:2b | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 623.4         | 223.8        | 128     | 1 918      |  -      |  -        |            |             | 2026-09-01 13:14 |
| granite3.1-dense:2b | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 755.0**   | **277.1**    | **128** | **519**    | ** - ** | ** - **   | **+23.8%** |             | 2026-09-01 13:14 |
| granite3.1-dense:2b | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 143.2        | 128     | 1 891      |  -      |  -        |            |             | 2026-09-01 11:36 |
| granite3.1-dense:2b | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **248.8**    | **128** | **522**    | ** - ** | ** - **   | **+73.7%** |             | 2026-09-01 11:36 |
| granite3.1-dense:2b | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 822.3         | 13.3         | 128     | 11 258     | 418     | 3.263     |            |             | 2026-08-18 01:14 |
| granite3.1-dense:2b | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **178.4**     | **12.2**     | **128** | **10 850** | **501** | **3.917** | -8.3%      | -16.7%      | 2026-08-18 01:14 |
| granite3.1-dense:2b | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 230.6         | 223.8        | 128     | 1 886      |  -      |  -        |            |             | 2026-09-01 09:58 |
| granite3.1-dense:2b | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **452.0**     | **278.5**    | **128** | **522**    | ** - ** | ** - **   | **+24.5%** |             | 2026-09-01 09:58 |
| granite3.1-dense:2b | 16384  | long   | stream     | GPU    | Ollama 0.32.6   | 87 644.1      | 288.7        | 128     | 1 922      | 195     | 1.525     |            |             | 2026-08-18 01:14 |
| granite3.1-dense:2b | 16384  | long   | stream     | GPU    | **loken 0.1.0** | **25 834.3**  | **267.7**    | **128** | **530**    | **104** | **0.812** | -7.3%      | **+87.8%**  | 2026-08-18 01:14 |
| granite3.1-dense:2b | 32768  | long   | stream     | GPU    | Ollama 0.32.6   | 87 503.3      | 288.0        | 128     | 1 934      | 199     | 1.555     |            |             | 2026-08-18 01:14 |
| granite3.1-dense:2b | 32768  | long   | stream     | GPU    | **loken 0.1.0** | **7 213.2**   | **266.8**    | **128** | **555**    | **98**  | **0.763** | -7.4%      | **+103.8%** | 2026-08-18 01:14 |
| granite3.1-dense:2b | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 637.2         | 223.2        | 128     | 2 153      |  -      |  -        |            |             | 2026-09-02 00:27 |
| granite3.1-dense:2b | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **1 641.1**   | **276.6**    | **128** | **520**    | ** - ** | ** - **   | **+23.9%** |             | 2026-09-02 00:27 |
| granite3.1-dense:2b | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 142.3        | 128     | 2 047      |  -      |  -        |            |             | 2026-09-01 21:54 |
| granite3.1-dense:2b | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **236.1**    | **128** | **545**    | ** - ** | ** - **   | **+66.0%** |             | 2026-09-01 21:54 |
| granite3.1-dense:2b | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 222.3         | 220.2        | 128     | 2 053      |  -      |  -        |            |             | 2026-09-01 19:59 |
| granite3.1-dense:2b | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **611.2**     | **279.3**    | **128** | **516**    | ** - ** | ** - **   | **+26.8%** |             | 2026-09-01 19:59 |

### granitemoe

![granitemoe](img/family-granitemoe.svg)

| Model           | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token   | Δ decode  | Δ energy | Date             |
|-----------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|-----------|-----------|----------|------------------|
| granite3-moe:1b | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 254.9        | 128     | 517       |  -      |  -        |           |          | 2026-09-01 18:06 |
| granite3-moe:1b | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **256.4**    | **128** | **513**   | ** - ** | ** - **   | **+0.6%** |          | 2026-09-01 18:06 |
| granite3-moe:1b | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 2 743.7       | 323.7        | 128     | 534       |  -      |  -        |           |          | 2026-09-01 16:28 |
| granite3-moe:1b | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **4 817.6**   | **327.4**    | **128** | **499**   | ** - ** | ** - **   | **+1.1%** |          | 2026-09-01 16:28 |
| granite3-moe:1b | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 250.0        | 128     | 524       |  -      |  -        |           |          | 2026-09-01 14:52 |
| granite3-moe:1b | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **271.9**    | **128** | **483**   | ** - ** | ** - **   | **+8.8%** |          | 2026-09-01 14:52 |
| granite3-moe:1b | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 674.9         | 331.3        | 128     | 512       |  -      |  -        |           |          | 2026-09-01 13:14 |
| granite3-moe:1b | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **936.0**     | **320.1**    | **128** | **505**   | ** - ** | ** - **   | -3.4%     |          | 2026-09-01 13:14 |
| granite3-moe:1b | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 258.0        | 128     | 511       |  -      |  -        |           |          | 2026-09-01 11:35 |
| granite3-moe:1b | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **274.3**    | **128** | **474**   | ** - ** | ** - **   | **+6.3%** |          | 2026-09-01 11:35 |
| granite3-moe:1b | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 3 383.2       | 68.7         | 128     | 3 165     | 152     | 1.184     |           |          | 2026-08-18 01:14 |
| granite3-moe:1b | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **118.5**     | **55.7**     | **128** | **2 593** | **158** | **1.234** | -18.9%    | -4.1%    | 2026-08-18 01:14 |
| granite3-moe:1b | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 249.1         | 333.8        | 128     | 516       |  -      |  -        |           |          | 2026-09-01 09:57 |
| granite3-moe:1b | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **815.8**     | **323.5**    | **128** | **471**   | ** - ** | ** - **   | -3.1%     |          | 2026-09-01 09:57 |
| granite3-moe:1b | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 670.1         | 335.8        | 128     | 505       |  -      |  -        |           |          | 2026-09-02 00:26 |
| granite3-moe:1b | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **924.8**     | **321.2**    | **128** | **466**   | ** - ** | ** - **   | -4.3%     |          | 2026-09-02 00:26 |
| granite3-moe:1b | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 254.7        | 128     | 524       |  -      |  -        |           |          | 2026-09-01 21:53 |
| granite3-moe:1b | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **279.1**    | **128** | **471**   | ** - ** | ** - **   | **+9.6%** |          | 2026-09-01 21:53 |
| granite3-moe:1b | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 254.2         | 331.9        | 128     | 519       |  -      |  -        |           |          | 2026-09-01 19:58 |
| granite3-moe:1b | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **734.5**     | **325.7**    | **128** | **464**   | ** - ** | ** - **   | -1.9%     |          | 2026-09-01 19:58 |

### lfm2

![lfm2](img/family-lfm2.svg)

| Model                  | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms  | J/req   | J/token | Δ decode    | Δ energy | Date             |
|------------------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|---------|---------|---------|-------------|----------|------------------|
| lfm2.5-thinking:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 224.3        | 128     | 1 660   |  -      |  -      |             |          | 2026-09-01 18:07 |
| lfm2.5-thinking:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **438.3**    | **128** | **441** | ** - ** | ** - ** | **+95.4%**  |          | 2026-09-01 18:07 |
| lfm2.5-thinking:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 2 719.4       | 415.5        | 128     | 1 668   |  -      |  -      |             |          | 2026-09-01 16:30 |
| lfm2.5-thinking:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **9 414.2**   | **644.0**    | **128** | **443** | ** - ** | ** - ** | **+55.0%**  |          | 2026-09-01 16:30 |
| lfm2.5-thinking:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 213.2        | 128     | 1 688   |  -      |  -      |             |          | 2026-09-01 14:54 |
| lfm2.5-thinking:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **463.0**    | **128** | **404** | ** - ** | ** - ** | **+117.1%** |          | 2026-09-01 14:54 |
| lfm2.5-thinking:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 455.9         | 396.0        | 128     | 1 713   |  -      |  -      |             |          | 2026-09-01 13:15 |
| lfm2.5-thinking:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **3 580.4**   | **668.5**    | **128** | **414** | ** - ** | ** - ** | **+68.8%**  |          | 2026-09-01 13:15 |
| lfm2.5-thinking:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 215.5        | 128     | 1 672   |  -      |  -      |             |          | 2026-09-01 11:37 |
| lfm2.5-thinking:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **442.8**    | **128** | **402** | ** - ** | ** - ** | **+105.4%** |          | 2026-09-01 11:37 |
| lfm2.5-thinking:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 161.0         | 393.7        | 128     | 1 679   |  -      |  -      |             |          | 2026-09-01 09:59 |
| lfm2.5-thinking:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **563.6**     | **636.3**    | **128** | **440** | ** - ** | ** - ** | **+61.6%**  |          | 2026-09-01 09:59 |
| lfm2.5-thinking:latest | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 427.8         | 381.3        | 128     | 1 759   |  -      |  -      |             |          | 2026-09-02 00:28 |
| lfm2.5-thinking:latest | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **3 538.2**   | **669.8**    | **128** | **416** | ** - ** | ** - ** | **+75.7%**  |          | 2026-09-02 00:28 |
| lfm2.5-thinking:latest | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 213.7        | 128     | 1 740   |  -      |  -      |             |          | 2026-09-01 21:54 |
| lfm2.5-thinking:latest | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **475.4**    | **128** | **393** | ** - ** | ** - ** | **+122.4%** |          | 2026-09-01 21:54 |
| lfm2.5-thinking:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 161.3         | 386.2        | 128     | 1 894   |  -      |  -      |             |          | 2026-09-01 20:00 |
| lfm2.5-thinking:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **1 014.0**   | **669.1**    | **128** | **422** | ** - ** | ** - ** | **+73.3%**  |          | 2026-09-01 20:00 |

### lfm2moe

![lfm2moe](img/family-lfm2moe.svg)

| Model       | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode   | Δ energy    | Date             |
|-------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|------------|-------------|------------------|
| lfm2:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 133.1        | 128     | 3 169      |  -      |  -        |            |             | 2026-09-01 18:09 |
| lfm2:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **237.2**    | **128** | **684**    | ** - ** | ** - **   | **+78.2%** |             | 2026-09-01 18:09 |
| lfm2:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 2 617.3       | 233.8        | 128     | 3 112      |  -      |  -        |            |             | 2026-09-01 16:31 |
| lfm2:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **9 146.5**   | **308.5**    | **128** | **693**    | ** - ** | ** - **   | **+31.9%** |             | 2026-09-01 16:31 |
| lfm2:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 136.0        | 128     | 3 136      |  -      |  -        |            |             | 2026-09-01 14:55 |
| lfm2:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **70.4**     | **4**   | **257**    | ** - ** | ** - **   |            |             | 2026-09-01 14:55 |
| lfm2:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 404.6         | 233.5        | 128     | 3 125      |  -      |  -        |            |             | 2026-09-01 13:17 |
| lfm2:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **3 666.3**   | **238.1**    | **4**   | **226**    | ** - ** | ** - **   |            |             | 2026-09-01 13:17 |
| lfm2:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent |             | 2026-09-01 11:39 |
| lfm2:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent |             | 2026-09-01 11:39 |
| lfm2:latest | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 180.3         | 14.4         | 128     | 13 330     | 477     | 3.724     |            |             | 2026-08-18 01:14 |
| lfm2:latest | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **1 094.5**   | **14.0**     | **128** | **12 797** | **572** | **4.472** | -3.4%      | -16.7%      | 2026-08-18 01:14 |
| lfm2:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent |             | 2026-09-01 10:00 |
| lfm2:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent |             | 2026-09-01 10:00 |
| lfm2:latest | 16384  | long   | stream     | GPU    | Ollama 0.32.6   | 34 014.5      | 297.3        | 128     | 2 672      | 242     | 1.894     |            |             | 2026-08-18 01:14 |
| lfm2:latest | 16384  | long   | stream     | GPU    | **loken 0.1.0** | **512 707.6** | **307.0**    | **128** | **655**    | **111** | **0.869** | **+3.3%**  | **+117.9%** | 2026-08-18 01:14 |
| lfm2:latest | 32768  | long   | stream     | GPU    | Ollama 0.32.6   | 33 887.8      | 297.3        | 128     | 2 659      | 242     | 1.888     |            |             | 2026-08-18 01:14 |
| lfm2:latest | 32768  | long   | stream     | GPU    | **loken 0.1.0** | **528 241.9** | **307.6**    | **128** | **630**    | **124** | **0.973** | **+3.5%**  | **+94.1%**  | 2026-08-18 01:14 |
| lfm2:latest | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent |             | 2026-09-01 21:56 |
| lfm2:latest | 131072 | short  | non-stream | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent |             | 2026-09-01 21:56 |
| lfm2:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent |             | 2026-09-01 20:01 |
| lfm2:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** |  -            |  -           | **128** |  -         |  -      |  -        | incoherent |             | 2026-09-01 20:01 |

### llama

![llama](img/family-llama.svg)

| Model                | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms      | J/req   | J/token   | Δ decode     | Δ energy | Date             |
|----------------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|-------------|---------|-----------|--------------|----------|------------------|
| deepseek-r1:70b      | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 1.5          | 128     | 88 767      |  -      |  -        |              |          | 2026-09-01 17:24 |
| deepseek-r1:70b      | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **1.3**      | **128** | **99 377**  | ** - ** | ** - **   | -10.4%       |          | 2026-09-01 17:24 |
| deepseek-r1:70b      | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 250.8         | 1.5          | 128     | 88 777      |  -      |  -        |              |          | 2026-09-01 15:47 |
| deepseek-r1:70b      | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **24.1**      | **1.5**      | **128** | **100 193** | ** - ** | ** - **   | -1.6%        |          | 2026-09-01 15:47 |
| deepseek-r1:70b      | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 1.5          | 128     | 88 444      |  -      |  -        |              |          | 2026-09-01 14:11 |
| deepseek-r1:70b      | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **1.3**      | **128** | **99 211**  | ** - ** | ** - **   | -10.6%       |          | 2026-09-01 14:11 |
| deepseek-r1:70b      | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 45.9          | 1.6          | 128     | 88 181      |  -      |  -        |              |          | 2026-09-01 12:32 |
| deepseek-r1:70b      | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **20.1**      | **1.5**      | **128** | **96 647**  | ** - ** | ** - **   | -6.0%        |          | 2026-09-01 12:32 |
| deepseek-r1:70b      | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 1.5          | 128     | 88 064      |  -      |  -        |              |          | 2026-09-01 10:54 |
| deepseek-r1:70b      | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **1.4**      | **128** | **95 279**  | ** - ** | ** - **   | -7.2%        |          | 2026-09-01 10:54 |
| deepseek-r1:70b      | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 16.9          | 1.5          | 128     | 88 302      |  -      |  -        |              |          | 2026-09-01 09:15 |
| deepseek-r1:70b      | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **12.2**      | **1.5**      | **128** | **91 938**  | ** - ** | ** - **   | -0.9%        |          | 2026-09-01 09:15 |
| deepseek-r1:70b      | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 28.4          | 0.8          | 128     | 165 802     |  -      |  -        |              |          | 2026-09-01 23:33 |
| deepseek-r1:70b      | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **20.9**      | **1.5**      | **128** | **97 642**  | ** - ** | ** - **   | **+77.4%**   |          | 2026-09-01 23:33 |
| deepseek-r1:70b      | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 0.8          | 128     | 165 268     |  -      |  -        |              |          | 2026-09-01 21:03 |
| deepseek-r1:70b      | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **1.3**      | **128** | **98 169**  | ** - ** | ** - **   | **+70.0%**   |          | 2026-09-01 21:03 |
| deepseek-r1:70b      | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 10.4          | 0.8          | 128     | 165 115     |  -      |  -        |              |          | 2026-09-01 19:10 |
| deepseek-r1:70b      | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **12.0**      | **1.5**      | **128** | **92 457**  | ** - ** | ** - **   | **+86.8%**   |          | 2026-09-01 19:10 |
| deepseek-r1:70b-q3ks | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 5.8          | 128     | 23 520      |  -      |  -        |              |          | 2026-09-01 17:29 |
| deepseek-r1:70b-q3ks | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **17.8**     | **128** | **7 245**   | ** - ** | ** - **   | **+204.0%**  |          | 2026-09-01 17:29 |
| deepseek-r1:70b-q3ks | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 550.5         | 6.9          | 128     | 23 527      |  -      |  -        |              |          | 2026-09-01 15:51 |
| deepseek-r1:70b-q3ks | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **299.1**     | **20.4**     | **128** | **7 185**   | ** - ** | ** - **   | **+196.2%**  |          | 2026-09-01 15:51 |
| deepseek-r1:70b-q3ks | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 5.9          | 128     | 23 402      |  -      |  -        |              |          | 2026-09-01 14:15 |
| deepseek-r1:70b-q3ks | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **19.8**     | **128** | **6 480**   | ** - ** | ** - **   | **+238.5%**  |          | 2026-09-01 14:15 |
| deepseek-r1:70b-q3ks | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 91.4          | 6.9          | 128     | 23 352      |  -      |  -        |              |          | 2026-09-01 12:37 |
| deepseek-r1:70b-q3ks | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **224.3**     | **21.1**     | **128** | **6 493**   | ** - ** | ** - **   | **+204.8%**  |          | 2026-09-01 12:37 |
| deepseek-r1:70b-q3ks | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 5.9          | 128     | 23 310      |  -      |  -        |              |          | 2026-09-01 10:58 |
| deepseek-r1:70b-q3ks | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **20.2**     | **128** | **6 367**   | ** - ** | ** - **   | **+243.5%**  |          | 2026-09-01 10:58 |
| deepseek-r1:70b-q3ks | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 35.9          | 6.9          | 128     | 23 287      |  -      |  -        |              |          | 2026-09-01 09:20 |
| deepseek-r1:70b-q3ks | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **112.7**     | **21.2**     | **128** | **6 401**   | ** - ** | ** - **   | **+205.7%**  |          | 2026-09-01 09:20 |
| deepseek-r1:70b-q3ks | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 36.1          | 1.2          | 128     | 118 535     |  -      |  -        |              |          | 2026-09-01 23:45 |
| deepseek-r1:70b-q3ks | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **208.3**     | **21.1**     | **128** | **6 505**   | ** - ** | ** - **   | **+1727.4%** |          | 2026-09-01 23:45 |
| deepseek-r1:70b-q3ks | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 1.1          | 128     | 117 852     |  -      |  -        |              |          | 2026-09-01 21:13 |
| deepseek-r1:70b-q3ks | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **20.1**     | **128** | **6 401**   | ** - ** | ** - **   | **+1736.3%** |          | 2026-09-01 21:13 |
| deepseek-r1:70b-q3ks | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 13.6          | 1.2          | 128     | 117 268     |  -      |  -        |              |          | 2026-09-01 19:19 |
| deepseek-r1:70b-q3ks | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **112.5**     | **21.2**     | **128** | **6 392**   | ** - ** | ** - **   | **+1712.7%** |          | 2026-09-01 19:19 |
| devstral:24b         | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 1       | 2 790       |  -      |  -        | no answer    |          | 2026-09-01 17:43 |
| devstral:24b         | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 26.5         | 128     | 6 243       |  -      |  -        |              |          | 2026-09-01 14:29 |
| devstral:24b         | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **50.1**     | **128** | **2 559**   | ** - ** | ** - **   | **+89.0%**   |          | 2026-09-01 14:29 |
| devstral:24b         | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 170.9         | 36.2         | 128     | 6 240       |  -      |  -        |              |          | 2026-09-01 12:51 |
| devstral:24b         | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **745.4**     | **52.1**     | **128** | **2 564**   | ** - ** | ** - **   | **+43.9%**   |          | 2026-09-01 12:51 |
| devstral:24b         | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 26.5         | 128     | 6 244       |  -      |  -        |              |          | 2026-09-01 11:13 |
| devstral:24b         | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **49.3**     | **128** | **2 598**   | ** - ** | ** - **   | **+86.2%**   |          | 2026-09-01 11:13 |
| devstral:24b         | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 58.8          | 36.3         | 128     | 6 288       |  -      |  -        |              |          | 2026-09-01 09:34 |
| devstral:24b         | 4096   | short  | stream     | GPU    | vLLM 0.22.0     | 478.1         | 50.6         | 128     | 2 541       | 580     | 4.534     |              |          | 2026-08-10 07:40 |
| devstral:24b         | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **296.7**     | **51.0**     | **128** | **2 611**   | ** - ** | ** - **   | **+0.6%**    |          | 2026-09-01 09:34 |
| devstral:24b         | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 104.4         | 6.9          | 128     | 22 323      |  -      |  -        |              |          | 2026-09-02 00:01 |
| devstral:24b         | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **599.1**     | **51.8**     | **128** | **2 583**   | ** - ** | ** - **   | **+648.8%**  |          | 2026-09-02 00:01 |
| devstral:24b         | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 6.0          | 128     | 22 139      |  -      |  -        |              |          | 2026-09-01 21:29 |
| devstral:24b         | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **49.1**     | **128** | **2 609**   | ** - ** | ** - **   | **+713.4%**  |          | 2026-09-01 21:29 |
| devstral:24b         | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 36.1          | 6.9          | 128     | 22 203      |  -      |  -        |              |          | 2026-09-01 19:35 |
| devstral:24b         | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **295.4**     | **50.7**     | **128** | **2 628**   | ** - ** | ** - **   | **+629.3%**  |          | 2026-09-01 19:35 |
| falcon3:latest       | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 119.5        | 128     | 2 129       |  -      |  -        |              |          | 2026-09-01 17:45 |
| falcon3:latest       | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 1 891.9       | 208.9        | 128     | 2 120       |  -      |  -        |              |          | 2026-09-01 16:07 |
| falcon3:latest       | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 121.1        | 128     | 2 087       |  -      |  -        |              |          | 2026-09-01 14:31 |
| falcon3:latest       | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **342.3**    | **128** | **378**     | ** - ** | ** - **   | **+182.7%**  |          | 2026-09-01 14:31 |
| falcon3:latest       | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 271.4         | 211.0        | 128     | 2 113       |  -      |  -        |              |          | 2026-09-01 12:53 |
| falcon3:latest       | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **724.7**     | **389.5**    | **128** | **394**     | ** - ** | ** - **   | **+84.6%**   |          | 2026-09-01 12:53 |
| falcon3:latest       | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 120.8        | 128     | 2 117       |  -      |  -        |              |          | 2026-09-01 11:14 |
| falcon3:latest       | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **342.4**    | **128** | **378**     | ** - ** | ** - **   | **+183.6%**  |          | 2026-09-01 11:14 |
| falcon3:latest       | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 1 433.5       | 14.8         | 21      | 3 106       | 151     | 7.206     |              |          | 2026-08-18 01:14 |
| falcon3:latest       | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **181.6**     | **14.1**     | **79**  | **5 967**   | **327** | **4.118** |              |          | 2026-08-18 01:14 |
| falcon3:latest       | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 79.7          | 211.2        | 128     | 2 121       |  -      |  -        |              |          | 2026-09-01 09:36 |
| falcon3:latest       | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **512.9**     | **391.9**    | **128** | **385**     | ** - ** | ** - **   | **+85.6%**   |          | 2026-09-01 09:36 |
| falcon3:latest       | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 271.2         | 209.1        | 128     | 781         |  -      |  -        |              |          | 2026-09-02 00:03 |
| falcon3:latest       | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **1 449.1**   | **386.7**    | **128** | **388**     | ** - ** | ** - **   | **+85.0%**   |          | 2026-09-02 00:03 |
| falcon3:latest       | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 165.7        | 128     | 773         |  -      |  -        |              |          | 2026-09-01 21:31 |
| falcon3:latest       | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **335.9**    | **128** | **384**     | ** - ** | ** - **   | **+102.8%**  |          | 2026-09-01 21:31 |
| falcon3:latest       | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 92.7          | 212.0        | 128     | 776         |  -      |  -        |              |          | 2026-09-01 19:37 |
| falcon3:latest       | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **504.7**     | **391.4**    | **128** | **384**     | ** - ** | ** - **   | **+84.7%**   |          | 2026-09-01 19:37 |
| llama3.2:1b          | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 120.4        | 128     | 2 142       |  -      |  -        |              |          | 2026-09-01 18:10 |
| llama3.2:1b          | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **344.2**    | **128** | **375**     | ** - ** | ** - **   | **+185.7%**  |          | 2026-09-01 18:10 |
| llama3.2:1b          | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 910.1         | 244.0        | 128     | 2 152       |  -      |  -        |              |          | 2026-09-01 16:32 |
| llama3.2:1b          | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **6 454.3**   | **414.2**    | **128** | **374**     | ** - ** | ** - **   | **+69.8%**   |          | 2026-09-01 16:32 |
| llama3.2:1b          | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 120.9        | 128     | 2 136       |  -      |  -        |              |          | 2026-09-01 14:56 |
| llama3.2:1b          | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **383.0**    | **128** | **336**     | ** - ** | ** - **   | **+216.8%**  |          | 2026-09-01 14:56 |
| llama3.2:1b          | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 189.6         | 237.5        | 128     | 2 166       |  -      |  -        |              |          | 2026-09-01 13:18 |
| llama3.2:1b          | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 159.7**   | **447.5**    | **128** | **346**     | ** - ** | ** - **   | **+88.4%**   |          | 2026-09-01 13:18 |
| llama3.2:1b          | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 116.2        | 128     | 2 187       |  -      |  -        |              |          | 2026-09-01 11:40 |
| llama3.2:1b          | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **376.7**    | **128** | **344**     | ** - ** | ** - **   | **+224.1%**  |          | 2026-09-01 11:40 |
| llama3.2:1b          | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 655.2         | 16.6         | 128     | 9 415       | 357     | 2.791     |              |          | 2026-08-18 01:14 |
| llama3.2:1b          | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **321.0**     | **15.9**     | **128** | **8 239**   | **414** | **3.235** | -3.8%        | -13.7%   | 2026-08-18 01:14 |
| llama3.2:1b          | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 70.9          | 232.2        | 128     | 2 157       |  -      |  -        |              |          | 2026-09-01 10:01 |
| llama3.2:1b          | 4096   | short  | stream     | GPU    | vLLM 0.22.0     | 1 318.4       | 292.6        | 128     | 446         | 88      | 0.685     |              |          | 2026-08-10 07:11 |
| llama3.2:1b          | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **500.0**     | **451.0**    | **128** | **344**     | ** - ** | ** - **   | **+54.1%**   |          | 2026-09-01 10:01 |
| llama3.2:1b          | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 114.7        | 128     | 2 409       |  -      |  -        |              |          | 2026-09-01 21:57 |
| llama3.2:1b          | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **388.0**    | **128** | **332**     | ** - ** | ** - **   | **+238.4%**  |          | 2026-09-01 21:57 |
| llama3.2:1b          | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 71.3          | 228.0        | 128     | 2 291       |  -      |  -        |              |          | 2026-09-01 20:02 |
| llama3.2:1b          | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **249.7**     | **451.9**    | **128** | **348**     | ** - ** | ** - **   | **+98.2%**   |          | 2026-09-01 20:02 |
| magistral:latest     | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 1       | 2 827       |  -      |  -        | no answer    |          | 2026-09-01 18:12 |
| magistral:latest     | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 26.3         | 128     | 6 283       |  -      |  -        |              |          | 2026-09-01 14:58 |
| magistral:latest     | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **49.6**     | **128** | **2 586**   | ** - ** | ** - **   | **+88.1%**   |          | 2026-09-01 14:58 |
| magistral:latest     | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 168.4         | 36.3         | 128     | 6 296       |  -      |  -        |              |          | 2026-09-01 13:20 |
| magistral:latest     | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **689.9**     | **53.3**     | **128** | **2 500**   | ** - ** | ** - **   | **+46.8%**   |          | 2026-09-01 13:20 |
| magistral:latest     | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 26.3         | 128     | 6 422       |  -      |  -        |              |          | 2026-09-01 11:42 |
| magistral:latest     | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **49.6**     | **128** | **2 585**   | ** - ** | ** - **   | **+88.6%**   |          | 2026-09-01 11:42 |
| magistral:latest     | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 60.6          | 36.3         | 128     | 6 343       |  -      |  -        |              |          | 2026-09-01 10:03 |
| magistral:latest     | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **303.7**     | **51.3**     | **128** | **2 605**   | ** - ** | ** - **   | **+41.5%**   |          | 2026-09-01 10:03 |
| magistral:latest     | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 26.5         | 128     | 6 264       |  -      |  -        |              |          | 2026-09-01 21:59 |
| magistral:latest     | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **49.4**     | **128** | **2 597**   | ** - ** | ** - **   | **+86.2%**   |          | 2026-09-01 21:59 |
| magistral:latest     | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 58.1          | 36.3         | 128     | 6 285       |  -      |  -        |              |          | 2026-09-01 20:04 |
| magistral:latest     | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **311.5**     | **51.3**     | **128** | **2 624**   | ** - ** | ** - **   | **+41.4%**   |          | 2026-09-01 20:04 |
| mistral-nemo:latest  | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 46.5         | 128     | 3 964       |  -      |  -        |              |          | 2026-09-01 18:13 |
| mistral-nemo:latest  | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **91.3**     | **128** | **1 406**   | ** - ** | ** - **   | **+96.4%**   |          | 2026-09-01 18:13 |
| mistral-nemo:latest  | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 967.7         | 68.8         | 128     | 3 980       |  -      |  -        |              |          | 2026-09-01 16:36 |
| mistral-nemo:latest  | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **2 967.7**   | **100.1**    | **128** | **1 403**   | ** - ** | ** - **   | **+45.4%**   |          | 2026-09-01 16:36 |
| mistral-nemo:latest  | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 45.8         | 128     | 3 994       |  -      |  -        |              |          | 2026-09-01 15:00 |
| mistral-nemo:latest  | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **95.5**     | **128** | **1 342**   | ** - ** | ** - **   | **+108.4%**  |          | 2026-09-01 15:00 |
| mistral-nemo:latest  | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 179.6         | 67.9         | 128     | 4 009       |  -      |  -        |              |          | 2026-09-01 13:21 |
| mistral-nemo:latest  | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **941.9**     | **100.9**    | **128** | **1 359**   | ** - ** | ** - **   | **+48.7%**   |          | 2026-09-01 13:21 |
| mistral-nemo:latest  | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 45.9         | 128     | 3 998       |  -      |  -        |              |          | 2026-09-01 11:43 |
| mistral-nemo:latest  | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **95.1**     | **128** | **1 349**   | ** - ** | ** - **   | **+107.3%**  |          | 2026-09-01 11:43 |
| mistral-nemo:latest  | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 64.1          | 68.1         | 128     | 5 862       |  -      |  -        |              |          | 2026-09-01 10:05 |
| mistral-nemo:latest  | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **217.0**     | **102.2**    | **128** | **1 356**   | ** - ** | ** - **   | **+50.0%**   |          | 2026-09-01 10:05 |
| mistral-nemo:latest  | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 45.8         | 128     | 4 089       |  -      |  -        |              |          | 2026-09-01 22:00 |
| mistral-nemo:latest  | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **96.8**     | **128** | **1 324**   | ** - ** | ** - **   | **+111.6%**  |          | 2026-09-01 22:00 |
| mistral-nemo:latest  | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 64.6          | 68.2         | 128     | 4 069       |  -      |  -        |              |          | 2026-09-01 20:06 |
| mistral-nemo:latest  | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **374.4**     | **102.1**    | **128** | **1 338**   | ** - ** | ** - **   | **+49.8%**   |          | 2026-09-01 20:06 |

### mistral3

![mistral3](img/family-mistral3.svg)

| Model                   | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token | Δ decode     | Δ energy | Date             |
|-------------------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|---------|--------------|----------|------------------|
| devstral-small-2:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 25.8         | 128     | 6 544     |  -      |  -      |              |          | 2026-09-01 17:31 |
| devstral-small-2:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **48.5**     | **128** | **2 673** | ** - ** | ** - ** | **+88.3%**   |          | 2026-09-01 17:31 |
| devstral-small-2:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 872.2         | 35.5         | 128     | 6 577     |  -      |  -      |              |          | 2026-09-01 15:53 |
| devstral-small-2:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **999.4**     | **57.6**     | **128** | **2 671** | ** - ** | ** - ** | **+62.3%**   |          | 2026-09-01 15:53 |
| devstral-small-2:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 25.4         | 128     | 6 592     |  -      |  -      |              |          | 2026-09-01 14:17 |
| devstral-small-2:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **53.9**     | **128** | **2 412** | ** - ** | ** - ** | **+112.4%**  |          | 2026-09-01 14:17 |
| devstral-small-2:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 152.7         | 35.2         | 128     | 6 593     |  -      |  -      |              |          | 2026-09-01 12:39 |
| devstral-small-2:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **655.5**     | **59.3**     | **128** | **2 411** | ** - ** | ** - ** | **+68.2%**   |          | 2026-09-01 12:39 |
| devstral-small-2:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 25.6         | 128     | 6 571     |  -      |  -      |              |          | 2026-09-01 11:00 |
| devstral-small-2:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **54.8**     | **128** | **2 368** | ** - ** | ** - ** | **+113.9%**  |          | 2026-09-01 11:00 |
| devstral-small-2:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 55.1          | 35.4         | 128     | 6 556     |  -      |  -      |              |          | 2026-09-01 09:22 |
| devstral-small-2:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **285.9**     | **59.5**     | **128** | **2 399** | ** - ** | ** - ** | **+67.8%**   |          | 2026-09-01 09:22 |
| devstral-small-2:latest | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 82.1          | 5.3          | 128     | 28 321    |  -      |  -      |              |          | 2026-09-01 23:48 |
| devstral-small-2:latest | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **664.9**     | **59.3**     | **128** | **2 416** | ** - ** | ** - ** | **+1015.5%** |          | 2026-09-01 23:48 |
| devstral-small-2:latest | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 4.7          | 128     | 28 108    |  -      |  -      |              |          | 2026-09-01 21:16 |
| devstral-small-2:latest | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **54.4**     | **128** | **2 391** | ** - ** | ** - ** | **+1055.9%** |          | 2026-09-01 21:16 |
| devstral-small-2:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 30.6          | 5.3          | 128     | 28 146    |  -      |  -      |              |          | 2026-09-01 19:22 |
| devstral-small-2:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **290.5**     | **59.6**     | **128** | **2 382** | ** - ** | ** - ** | **+1020.1%** |          | 2026-09-01 19:22 |
| mistral-small3.2:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 26.4         | 128     | 6 413     |  -      |  -      |              |          | 2026-09-01 18:15 |
| mistral-small3.2:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **49.2**     | **128** | **2 632** | ** - ** | ** - ** | **+86.7%**   |          | 2026-09-01 18:15 |
| mistral-small3.2:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 885.7         | 36.4         | 128     | 6 508     |  -      |  -      |              |          | 2026-09-01 16:38 |
| mistral-small3.2:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **1 037.8**   | **57.8**     | **128** | **2 638** | ** - ** | ** - ** | **+58.5%**   |          | 2026-09-01 16:38 |
| mistral-small3.2:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 26.2         | 128     | 6 418     |  -      |  -      |              |          | 2026-09-01 15:02 |
| mistral-small3.2:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **53.1**     | **128** | **2 467** | ** - ** | ** - ** | **+103.0%**  |          | 2026-09-01 15:02 |
| mistral-small3.2:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 152.1         | 36.3         | 128     | 6 424     |  -      |  -      |              |          | 2026-09-01 13:23 |
| mistral-small3.2:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **675.8**     | **59.2**     | **128** | **2 470** | ** - ** | ** - ** | **+63.1%**   |          | 2026-09-01 13:23 |
| mistral-small3.2:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 26.0         | 128     | 6 429     |  -      |  -      |              |          | 2026-09-01 11:45 |
| mistral-small3.2:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **54.6**     | **128** | **2 372** | ** - ** | ** - ** | **+110.5%**  |          | 2026-09-01 11:45 |
| mistral-small3.2:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 50.0          | 36.3         | 128     | 6 459     |  -      |  -      |              |          | 2026-09-01 10:07 |
| mistral-small3.2:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **299.4**     | **59.4**     | **128** | **2 388** | ** - ** | ** - ** | **+63.9%**   |          | 2026-09-01 10:07 |
| mistral-small3.2:latest | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 4.8          | 128     | 27 859    |  -      |  -      |              |          | 2026-09-01 22:03 |
| mistral-small3.2:latest | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **54.9**     | **128** | **2 362** | ** - ** | ** - ** | **+1053.5%** |          | 2026-09-01 22:03 |
| mistral-small3.2:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 31.5          | 5.5          | 128     | 27 569    |  -      |  -      |              |          | 2026-09-01 20:09 |
| mistral-small3.2:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **296.9**     | **59.4**     | **128** | **2 381** | ** - ** | ** - ** | **+985.4%**  |          | 2026-09-01 20:09 |

### nemotron_h_moe

![nemotron_h_moe](img/family-nemotron_h_moe.svg)

| Model                  | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token | Δ decode   | Δ energy | Date             |
|------------------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|---------|------------|----------|------------------|
| nemotron-3-nano:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 72.7         | 128     | 4 804     |  -      |  -      |            |          | 2026-09-01 18:18 |
| nemotron-3-nano:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **116.4**    | **128** | **1 322** | ** - ** | ** - ** | **+60.2%** |          | 2026-09-01 18:18 |
| nemotron-3-nano:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 876.7         | 134.2        | 128     | 4 820     |  -      |  -      |            |          | 2026-09-01 16:41 |
| nemotron-3-nano:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **14 928.2**  | **144.4**    | **128** | **1 324** | ** - ** | ** - ** | **+7.7%**  |          | 2026-09-01 16:41 |
| nemotron-3-nano:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 73.0         | 128     | 4 689     |  -      |  -      |            |          | 2026-09-01 15:05 |
| nemotron-3-nano:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **120.4**    | **128** | **1 214** | ** - ** | ** - ** | **+65.0%** |          | 2026-09-01 15:05 |
| nemotron-3-nano:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 124.7         | 136.0        | 128     | 4 713     |  -      |  -      |            |          | 2026-09-01 13:26 |
| nemotron-3-nano:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **2 235.5**   | **144.4**    | **128** | **1 199** | ** - ** | ** - ** | **+6.2%**  |          | 2026-09-01 13:26 |
| nemotron-3-nano:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 70.3         | 128     | 4 742     |  -      |  -      |            |          | 2026-09-01 11:48 |
| nemotron-3-nano:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **120.1**    | **128** | **1 211** | ** - ** | ** - ** | **+70.7%** |          | 2026-09-01 11:48 |
| nemotron-3-nano:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 48.4          | 133.2        | 128     | 4 724     |  -      |  -      |            |          | 2026-09-01 10:10 |
| nemotron-3-nano:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **800.5**     | **145.0**    | **128** | **1 204** | ** - ** | ** - ** | **+8.9%**  |          | 2026-09-01 10:10 |
| nemotron-3-nano:latest | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 72.3         | 128     | 4 845     |  -      |  -      |            |          | 2026-09-01 22:06 |
| nemotron-3-nano:latest | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **121.5**    | **128** | **1 186** | ** - ** | ** - ** | **+68.1%** |          | 2026-09-01 22:06 |
| nemotron-3-nano:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 42.5          | 132.5        | 128     | 4 856     |  -      |  -      |            |          | 2026-09-01 20:12 |
| nemotron-3-nano:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **629.2**     | **144.6**    | **128** | **1 210** | ** - ** | ** - ** | **+9.1%**  |          | 2026-09-01 20:12 |

### olmo2

![olmo2](img/family-olmo2.svg)

| Model    | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode   | Δ energy | Date             |
|----------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|------------|----------|------------------|
| olmo2:7b | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 87.8         | 128     | 1 460      |  -      |  -        |            |          | 2026-09-01 18:19 |
| olmo2:7b | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **104.3**    | **128** | **1 232**  | ** - ** | ** - **   | **+18.8%** |          | 2026-09-01 18:19 |
| olmo2:7b | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 1 514.2       | 98.2         | 128     | 1 468      |  -      |  -        |            |          | 2026-09-01 16:42 |
| olmo2:7b | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **3 032.2**   | **114.8**    | **128** | **1 235**  | ** - ** | ** - **   | **+16.9%** |          | 2026-09-01 16:42 |
| olmo2:7b | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 87.5         | 128     | 1 464      |  -      |  -        |            |          | 2026-09-01 15:06 |
| olmo2:7b | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **125.4**    | **128** | **1 025**  | ** - ** | ** - **   | **+43.4%** |          | 2026-09-01 15:06 |
| olmo2:7b | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 234.1         | 100.7        | 128     | 1 442      |  -      |  -        |            |          | 2026-09-01 13:27 |
| olmo2:7b | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 097.9**   | **134.2**    | **128** | **1 038**  | ** - ** | ** - **   | **+33.3%** |          | 2026-09-01 13:27 |
| olmo2:7b | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 89.2         | 128     | 1 436      |  -      |  -        |            |          | 2026-09-01 11:49 |
| olmo2:7b | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **126.3**    | **128** | **1 017**  | ** - ** | ** - **   | **+41.6%** |          | 2026-09-01 11:49 |
| olmo2:7b | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 270.5         | 5.2          | 128     | 27 396     | 978     | 7.638     |            |          | 2026-08-18 01:14 |
| olmo2:7b | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **85.6**      | **6.7**      | **91**  | **15 762** | **508** | **5.606** |            |          | 2026-08-18 01:14 |
| olmo2:7b | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 91.2          | 100.6        | 128     | 1 450      |  -      |  -        |            |          | 2026-09-01 10:11 |
| olmo2:7b | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **427.6**     | **133.9**    | **128** | **1 032**  | ** - ** | ** - **   | **+33.2%** |          | 2026-09-01 10:11 |
| olmo2:7b | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 89.3         | 128     | 1 433      |  -      |  -        |            |          | 2026-09-01 22:07 |
| olmo2:7b | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **126.0**    | **128** | **1 019**  | ** - ** | ** - **   | **+41.0%** |          | 2026-09-01 22:07 |
| olmo2:7b | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 84.9          | 100.9        | 128     | 1 421      |  -      |  -        |            |          | 2026-09-01 20:13 |
| olmo2:7b | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **383.8**     | **133.9**    | **128** | **1 028**  | ** - ** | ** - **   | **+32.7%** |          | 2026-09-01 20:13 |

### olmoe

![olmoe](img/family-olmoe.svg)

| Model        | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms  | J/req   | J/token | Δ decode   | Δ energy | Date             |
|--------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|---------|---------|---------|------------|----------|------------------|
| olmoe:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 281.4        | 128     | 466     |  -      |  -      |            |          | 2026-09-01 18:20 |
| olmoe:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **278.1**    | **128** | **473** | ** - ** | ** - ** | -1.2%      |          | 2026-09-01 18:20 |
| olmoe:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 4 193.6       | 381.1        | 128     | 448     |  -      |  -      |            |          | 2026-09-01 16:43 |
| olmoe:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **2 332.0**   | **395.2**    | **128** | **566** | ** - ** | ** - ** | **+3.7%**  |          | 2026-09-01 16:43 |
| olmoe:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 287.2        | 128     | 454     |  -      |  -      |            |          | 2026-09-01 15:07 |
| olmoe:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **330.0**    | **128** | **415** | ** - ** | ** - ** | **+14.9%** |          | 2026-09-01 15:07 |
| olmoe:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 540.6         | 390.2        | 128     | 443     |  -      |  -      |            |          | 2026-09-01 13:29 |
| olmoe:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 293.3**   | **428.9**    | **128** | **410** | ** - ** | ** - ** | **+9.9%**  |          | 2026-09-01 13:29 |
| olmoe:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 296.4        | 128     | 436     |  -      |  -      |            |          | 2026-09-01 11:50 |
| olmoe:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **312.1**    | **128** | **458** | ** - ** | ** - ** | **+5.3%**  |          | 2026-09-01 11:50 |
| olmoe:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 181.9         | 389.0        | 128     | 422     |  -      |  -      |            |          | 2026-09-01 10:12 |
| olmoe:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **380.7**     | **440.7**    | **128** | **392** | ** - ** | ** - ** | **+13.3%** |          | 2026-09-01 10:12 |
| olmoe:latest | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 303.0        | 128     | 425     |  -      |  -      |            |          | 2026-09-01 22:09 |
| olmoe:latest | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **339.3**    | **128** | **415** | ** - ** | ** - ** | **+12.0%** |          | 2026-09-01 22:09 |
| olmoe:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 187.1         | 388.7        | 128     | 422     |  -      |  -      |            |          | 2026-09-01 20:14 |
| olmoe:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **475.7**     | **436.6**    | **128** | **385** | ** - ** | ** - ** | **+12.3%** |          | 2026-09-01 20:14 |

### phi2

![phi2](img/family-phi2.svg)

| Model            | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms  | J/req   | J/token | Δ decode   | Δ energy | Date             |
|------------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|---------|---------|---------|------------|----------|------------------|
| moondream:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 1       | 112     |  -      |  -      | no answer  |          | 2026-09-01 18:16 |
| moondream:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 293.0        | 128     | 437     |  -      |  -      |            |          | 2026-09-01 15:02 |
| moondream:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **467.5**    | **128** | **280** | ** - ** | ** - ** | **+59.5%** |          | 2026-09-01 15:02 |
| moondream:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 513.9         | 370.2        | 128     | 450     |  -      |  -      |            |          | 2026-09-01 13:24 |
| moondream:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **2 721.0**   | **591.5**    | **128** | **250** | ** - ** | ** - ** | **+59.8%** |          | 2026-09-01 13:24 |
| moondream:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 289.7        | 128     | 443     |  -      |  -      |            |          | 2026-09-01 11:46 |
| moondream:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **508.6**    | **128** | **254** | ** - ** | ** - ** | **+75.5%** |          | 2026-09-01 11:46 |
| moondream:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 172.0         | 370.9        | 128     | 447     |  -      |  -      |            |          | 2026-09-01 10:08 |
| moondream:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **1 085.0**   | **598.1**    | **128** | **245** | ** - ** | ** - ** | **+61.2%** |          | 2026-09-01 10:08 |
| moondream:latest | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 293.3        | 128     | 437     |  -      |  -      |            |          | 2026-09-01 22:04 |
| moondream:latest | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **534.5**    | **128** | **242** | ** - ** | ** - ** | **+82.3%** |          | 2026-09-01 22:04 |
| moondream:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 171.0         | 372.3        | 128     | 433     |  -      |  -      |            |          | 2026-09-01 20:09 |
| moondream:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **958.1**     | **574.5**    | **128** | **249** | ** - ** | ** - ** | **+54.3%** |          | 2026-09-01 20:09 |

### qwen2

![qwen2](img/family-qwen2.svg)

| Model            | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req     | J/token    | Δ decode     | Δ energy   | Date             |
|------------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|-----------|------------|--------------|------------|------------------|
| deepcoder:14b    | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 38.0         | 128     | 4 618      |  -        |  -         |              |            | 2026-09-01 17:08 |
| deepcoder:14b    | 4096   | long   | non-stream | GPU    | vLLM 0.22.0     |  -            | 80.6         | 128     | 1 593      |  -        |  -         |              |            | 2026-09-01 17:08 |
| deepcoder:14b    | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **71.1**     | **128** | **1 801**  | ** - **   | ** - **    | -11.8%       |            | 2026-09-01 17:08 |
| deepcoder:14b    | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 1 164.7       | 53.3         | 128     | 4 605      |  -        |  -         |              |            | 2026-09-01 15:31 |
| deepcoder:14b    | 4096   | long   | stream     | GPU    | vLLM 0.22.0     | 12 210.2      | 85.0         | 128     | 1 569      |  -        |  -         |              |            | 2026-09-01 15:31 |
| deepcoder:14b    | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **2 195.8**   | **75.7**     | **128** | **1 813**  | ** - **   | ** - **    | -10.9%       |            | 2026-09-01 15:31 |
| deepcoder:14b    | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 37.7         | 128     | 4 596      |  -        |  -         |              |            | 2026-09-01 13:55 |
| deepcoder:14b    | 4096   | medium | non-stream | GPU    | vLLM 0.22.0     |  -            | 82.4         | 128     | 1 555      |  -        |  -         |              |            | 2026-09-01 13:55 |
| deepcoder:14b    | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **77.3**     | **128** | **1 658**  | ** - **   | ** - **    | -6.2%        |            | 2026-09-01 13:55 |
| deepcoder:14b    | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 204.8         | 52.9         | 128     | 4 636      |  -        |  -         |              |            | 2026-09-01 12:17 |
| deepcoder:14b    | 4096   | medium | stream     | GPU    | vLLM 0.22.0     | 2 425.3       | 85.0         | 128     | 1 550      |  -        |  -         |              |            | 2026-09-01 12:17 |
| deepcoder:14b    | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **777.9**     | **80.4**     | **128** | **1 668**  | ** - **   | ** - **    | -5.5%        |            | 2026-09-01 12:17 |
| deepcoder:14b    | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 37.6         | 128     | 4 607      |  -        |  -         |              |            | 2026-09-01 10:39 |
| deepcoder:14b    | 4096   | short  | non-stream | GPU    | vLLM 0.22.0     |  -            | 82.7         | 128     | 1 549      |  -        |  -         |              |            | 2026-09-01 10:39 |
| deepcoder:14b    | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **76.6**     | **128** | **1 671**  | ** - **   | ** - **    | -7.4%        |            | 2026-09-01 10:39 |
| deepcoder:14b    | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 52.4          | 2.6          | 128     | 52 337     | 1 718     | 13.418     |              |            | 2026-08-18 01:14 |
| deepcoder:14b    | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **30.2**      | **2.6**      | **128** | **50 305** | **1 636** | **12.782** | **+1.2%**    | **+5.0%**  | 2026-08-18 01:14 |
| deepcoder:14b    | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 70.5          | 52.6         | 128     | 4 639      |  -        |  -         |              |            | 2026-09-01 08:59 |
| deepcoder:14b    | 4096   | short  | stream     | GPU    | vLLM 0.22.0     | 894.2         | 85.1         | 128     | 1 531      |  -        |  -         |              |            | 2026-09-01 08:59 |
| deepcoder:14b    | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **211.7**     | **78.8**     | **128** | **1 694**  | ** - **   | ** - **    | -7.4%        |            | 2026-09-01 08:59 |
| deepcoder:14b    | 16384  | long   | stream     | GPU    | Ollama 0.32.6   | 16 383.4      | 75.4         | 128     | 3 681      | 557       | 4.351      |              |            | 2026-08-18 01:14 |
| deepcoder:14b    | 16384  | long   | stream     | GPU    | **loken 0.1.0** | **14 168.0**  | **77.3**     | **128** | **1 721**  | **441**   | **3.448**  | **+2.5%**    | **+26.2%** | 2026-08-18 01:14 |
| deepcoder:14b    | 32768  | long   | stream     | GPU    | Ollama 0.32.6   | 16 559.7      | 75.4         | 128     | 3 704      | 547       | 4.276      |              |            | 2026-08-18 01:14 |
| deepcoder:14b    | 32768  | long   | stream     | GPU    | **loken 0.1.0** | **5 334.8**   | **77.3**     | **128** | **1 731**  | **430**   | **3.359**  | **+2.5%**    | **+27.3%** | 2026-08-18 01:14 |
| deepcoder:14b    | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 134.1         | 12.8         | 128     | 13 127     |  -        |  -         |              |            | 2026-09-01 23:03 |
| deepcoder:14b    | 131072 | medium | stream     | GPU    | vLLM 0.22.0     | 2 457.5       | 85.0         | 128     | 1 549      |  -        |  -         |              |            | 2026-09-01 23:03 |
| deepcoder:14b    | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **851.6**     | **79.6**     | **128** | **1 679**  | ** - **   | ** - **    | -6.4%        |            | 2026-09-01 23:03 |
| deepcoder:14b    | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 10.5         | 128     | 13 199     |  -        |  -         |              |            | 2026-09-01 20:39 |
| deepcoder:14b    | 131072 | short  | non-stream | GPU    | vLLM 0.22.0     |  -            | 82.2         | 128     | 1 559      |  -        |  -         |              |            | 2026-09-01 20:39 |
| deepcoder:14b    | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **76.3**     | **128** | **1 677**  | ** - **   | ** - **    | -7.2%        |            | 2026-09-01 20:39 |
| deepcoder:14b    | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 47.7          | 12.8         | 128     | 13 140     |  -        |  -         |              |            | 2026-09-01 18:47 |
| deepcoder:14b    | 131072 | short  | stream     | GPU    | vLLM 0.22.0     | 742.7         | 85.0         | 128     | 1 551      |  -        |  -         |              |            | 2026-09-01 18:47 |
| deepcoder:14b    | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **221.1**     | **78.9**     | **128** | **1 700**  | ** - **   | ** - **    | -7.2%        |            | 2026-09-01 18:47 |
| deepcoder:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 1 478.3       | 74.4         | 128     | 3 644      | 544       | 4.250      |              |            | 2026-08-18 01:14 |
| deepcoder:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **1 515.0**   | **86.0**     | **128** | **1 532**  | **394**   | **3.081**  | **+15.7%**   | **+38.0%** | 2026-08-18 01:14 |
| deepseek-r1:32b  | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 19.6         | 128     | 8 218      |  -        |  -         |              |            | 2026-09-01 17:10 |
| deepseek-r1:32b  | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **34.5**     | **128** | **3 733**  | ** - **   | ** - **    | **+76.5%**   |            | 2026-09-01 17:10 |
| deepseek-r1:32b  | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 1 052.7       | 25.9         | 128     | 8 094      |  -        |  -         |              |            | 2026-09-01 15:33 |
| deepseek-r1:32b  | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **668.7**     | **39.7**     | **128** | **3 796**  | ** - **   | ** - **    | **+53.4%**   |            | 2026-09-01 15:33 |
| deepseek-r1:32b  | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 19.7         | 128     | 8 089      |  -        |  -         |              |            | 2026-09-01 13:57 |
| deepseek-r1:32b  | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **37.5**     | **128** | **3 452**  | ** - **   | ** - **    | **+90.4%**   |            | 2026-09-01 13:57 |
| deepseek-r1:32b  | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 180.1         | 25.9         | 128     | 8 158      |  -        |  -         |              |            | 2026-09-01 12:19 |
| deepseek-r1:32b  | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **455.1**     | **40.8**     | **128** | **3 473**  | ** - **   | ** - **    | **+57.7%**   |            | 2026-09-01 12:19 |
| deepseek-r1:32b  | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 19.6         | 128     | 8 267      |  -        |  -         |              |            | 2026-09-01 10:41 |
| deepseek-r1:32b  | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **38.4**     | **128** | **3 364**  | ** - **   | ** - **    | **+96.2%**   |            | 2026-09-01 10:41 |
| deepseek-r1:32b  | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 63.3          | 25.8         | 128     | 8 179      |  -        |  -         |              |            | 2026-09-01 09:01 |
| deepseek-r1:32b  | 4096   | short  | stream     | GPU    | vLLM 0.22.0     | 347.6         | 36.9         | 128     | 3 490      | 811       | 6.334      |              |            | 2026-08-10 07:42 |
| deepseek-r1:32b  | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **250.2**     | **41.0**     | **128** | **3 432**  | ** - **   | ** - **    | **+11.2%**   |            | 2026-09-01 09:01 |
| deepseek-r1:32b  | 131072 | medium | stream     | GPU    | Ollama 0.32.6   | 68.6          | 2.4          | 128     | 58 163     |  -        |  -         |              |            | 2026-09-01 23:13 |
| deepseek-r1:32b  | 131072 | medium | stream     | GPU    | **loken 0.1.0** | **454.8**     | **40.8**     | **128** | **3 472**  | ** - **   | ** - **    | **+1577.7%** |            | 2026-09-01 23:13 |
| deepseek-r1:32b  | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 2.2          | 128     | 57 944     |  -        |  -         |              |            | 2026-09-01 20:45 |
| deepseek-r1:32b  | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **38.1**     | **128** | **3 385**  | ** - **   | ** - **    | **+1601.7%** |            | 2026-09-01 20:45 |
| deepseek-r1:32b  | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 24.0          | 2.5          | 128     | 56 889     |  -        |  -         |              |            | 2026-09-01 18:53 |
| deepseek-r1:32b  | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **159.8**     | **40.8**     | **128** | **3 440**  | ** - **   | ** - **    | **+1543.6%** |            | 2026-09-01 18:53 |
| qwen2.5:0.5b     | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 157.8        | 128     | 1 923      |  -        |  -         |              |            | 2026-09-01 18:21 |
| qwen2.5:0.5b     | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **312.7**    | **128** | **413**    | ** - **   | ** - **    | **+98.2%**   |            | 2026-09-01 18:21 |
| qwen2.5:0.5b     | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 1 282.5       | 312.3        | 128     | 1 900      |  -        |  -         |              |            | 2026-09-01 16:44 |
| qwen2.5:0.5b     | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **6 541.7**   | **352.9**    | **128** | **430**    | ** - **   | ** - **    | **+13.0%**   |            | 2026-09-01 16:44 |
| qwen2.5:0.5b     | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 151.9        | 128     | 1 945      |  -        |  -         |              |            | 2026-09-01 15:08 |
| qwen2.5:0.5b     | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **321.5**    | **128** | **408**    | ** - **   | ** - **    | **+111.7%**  |            | 2026-09-01 15:08 |
| qwen2.5:0.5b     | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 203.9         | 317.3        | 128     | 1 927      |  -        |  -         |              |            | 2026-09-01 13:29 |
| qwen2.5:0.5b     | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 070.0**   | **365.2**    | **128** | **414**    | ** - **   | ** - **    | **+15.1%**   |            | 2026-09-01 13:29 |
| qwen2.5:0.5b     | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 153.4        | 128     | 1 951      |  -        |  -         |              |            | 2026-09-01 11:51 |
| qwen2.5:0.5b     | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **332.6**    | **128** | **390**    | ** - **   | ** - **    | **+116.7%**  |            | 2026-09-01 11:51 |
| qwen2.5:0.5b     | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 2 126.1       | 49.7         | 35      | 2 181      | 121       | 3.433      |              |            | 2026-08-18 01:14 |
| qwen2.5:0.5b     | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **932.4**     | **49.1**     | **42**  | **989**    | **63**    | **1.489**  |              |            | 2026-08-18 01:14 |
| qwen2.5:0.5b     | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 80.0          | 302.2        | 128     | 1 920      |  -        |  -         |              |            | 2026-09-02 00:33 |
| qwen2.5:0.5b     | 4096   | short  | stream     | GPU    | vLLM 0.22.0     | 1 451.3       | 502.5        | 128     | 273        | 51        | 0.398      |              |            | 2026-08-10 07:11 |
| qwen2.5:0.5b     | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **498.1**     | **369.5**    | **128** | **394**    | ** - **   | ** - **    | -26.5%       |            | 2026-09-02 00:33 |
| qwen2.5:0.5b     | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 197.1        | 128     | 663        |  -        |  -         |              |            | 2026-09-01 22:10 |
| qwen2.5:0.5b     | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **324.5**    | **128** | **403**    | ** - **   | ** - **    | **+64.7%**   |            | 2026-09-01 22:10 |
| qwen2.5:0.5b     | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 82.2          | 302.0        | 128     | 661        |  -        |  -         |              |            | 2026-09-01 20:15 |
| qwen2.5:0.5b     | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **414.2**     | **362.9**    | **128** | **415**    | ** - **   | ** - **    | **+20.2%**   |            | 2026-09-01 20:15 |

### qwen3

![qwen3](img/family-qwen3.svg)

| Model        | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode    | Δ energy    | Date             |
|--------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|-------------|-------------|------------------|
| qwen3:0.6b   | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent  |             | 2026-09-01 18:36 |
| qwen3:0.6b   | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **439.6**    | **128** | **295**    | ** - ** | ** - **   |             |             | 2026-09-01 18:36 |
| qwen3:0.6b   | 4096   | long   | stream     | GPU    | Ollama 0.32.6   |  -            |  -           | 128     |  -         |  -      |  -        | incoherent  |             | 2026-09-01 16:57 |
| qwen3:0.6b   | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **5 864.6**   | **603.8**    | **128** | **298**    | ** - ** | ** - **   |             |             | 2026-09-01 16:57 |
| qwen3:0.6b   | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 198.7        | 128     | 1 776      |  -      |  -        |             |             | 2026-09-01 15:20 |
| qwen3:0.6b   | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **520.1**    | **128** | **257**    | ** - ** | ** - **   | **+161.7%** |             | 2026-09-01 15:20 |
| qwen3:0.6b   | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 226.1         | 455.9        | 128     | 1 766      |  -      |  -        |             |             | 2026-09-01 13:44 |
| qwen3:0.6b   | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 173.6**   | **718.1**    | **128** | **265**    | ** - ** | ** - **   | **+57.5%**  |             | 2026-09-01 13:44 |
| qwen3:0.6b   | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 190.0        | 128     | 1 792      |  -      |  -        |             |             | 2026-09-01 12:06 |
| qwen3:0.6b   | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **500.5**    | **128** | **268**    | ** - ** | ** - **   | **+163.4%** |             | 2026-09-01 12:06 |
| qwen3:0.6b   | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 1 419.7       | 45.6         | 128     | 4 286      | 186     | 1.451     |             |             | 2026-08-18 01:14 |
| qwen3:0.6b   | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **207.3**     | **46.8**     | **128** | **2 904**  | **178** | **1.393** | **+2.7%**   | **+4.1%**   | 2026-08-18 01:14 |
| qwen3:0.6b   | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 79.2          | 469.9        | 128     | 1 739      |  -      |  -        |             |             | 2026-09-01 10:26 |
| qwen3:0.6b   | 4096   | short  | stream     | GPU    | vLLM 0.22.0     | 1 428.1       | 473.0        | 128     | 280        | 55      | 0.430     |             |             | 2026-08-10 07:10 |
| qwen3:0.6b   | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **469.3**     | **708.9**    | **128** | **262**    | ** - ** | ** - **   | **+49.9%**  |             | 2026-09-01 10:26 |
| qwen3:0.6b   | 16384  | long   | stream     | GPU    | Ollama 0.32.6   | 96 986.1      | 599.1        | 128     | 2 021      | 154     | 1.201     |             |             | 2026-08-18 01:14 |
| qwen3:0.6b   | 16384  | long   | stream     | GPU    | **loken 0.1.0** | **7 681.2**   | **646.5**    | **128** | **270**    | **48**  | **0.378** | **+7.9%**   | **+217.4%** | 2026-08-18 01:14 |
| qwen3:0.6b   | 16384  | short  | stream     | GPU    | Ollama 0.32.6   | 16 774.9      | 528.7        | 93      | 1 835      | 441     | 4.741     |             |             | 2026-08-18 01:14 |
| qwen3:0.6b   | 16384  | short  | stream     | GPU    | **loken 0.1.0** | **2 774.4**   | **670.3**    | **111** | **249**    | **32**  | **0.285** |             |             | 2026-08-18 01:14 |
| qwen3:0.6b   | 32768  | long   | stream     | GPU    | Ollama 0.32.6   | 99 721.9      | 604.4        | 128     | 2 023      | 153     | 1.196     |             |             | 2026-08-18 01:14 |
| qwen3:0.6b   | 32768  | long   | stream     | GPU    | **loken 0.1.0** | **22 113.3**  | **651.9**    | **128** | **258**    | **47**  | **0.364** | **+7.9%**   | **+228.9%** | 2026-08-18 01:14 |
| qwen3:0.6b   | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 198.3        | 128     | 1 849      |  -      |  -        |             |             | 2026-09-01 22:41 |
| qwen3:0.6b   | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **545.7**    | **128** | **246**    | ** - ** | ** - **   | **+175.3%** |             | 2026-09-01 22:41 |
| qwen3:0.6b   | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 79.1          | 454.4        | 128     | 1 858      |  -      |  -        |             |             | 2026-09-01 20:28 |
| qwen3:0.6b   | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **474.8**     | **713.9**    | **128** | **258**    | ** - ** | ** - **   | **+57.1%**  |             | 2026-09-01 20:28 |
| qwen3:1.7b   | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 64.8          | 407.7        | 128     | 534        |  -      |  -        |             |             | 2026-08-27 14:34 |
| qwen3:1.7b   | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **357.3**     | **427.8**    | **128** | **375**    | ** - ** | ** - **   | **+4.9%**   |             | 2026-08-27 14:34 |
| qwen3:8b     | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 63.2         | 128     | 3 223      |  -      |  -        |             |             | 2026-09-01 18:38 |
| qwen3:8b     | 4096   | long   | non-stream | GPU    | vLLM 0.22.0     |  -            | 131.5        | 128     | 979        |  -      |  -        |             |             | 2026-09-01 18:38 |
| qwen3:8b     | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **128.9**    | **128** | **996**    | ** - ** | ** - **   | -2.0%       |             | 2026-09-01 18:38 |
| qwen3:8b     | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 1 222.9       | 95.9         | 128     | 3 241      |  -      |  -        |             |             | 2026-09-01 16:59 |
| qwen3:8b     | 4096   | long   | stream     | GPU    | vLLM 0.22.0     | 13 099.3      | 139.4        | 128     | 970        |  -      |  -        |             |             | 2026-09-01 16:59 |
| qwen3:8b     | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **2 941.1**   | **141.6**    | **128** | **1 012**  | ** - ** | ** - **   | **+1.5%**   |             | 2026-09-01 16:59 |
| qwen3:8b     | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 63.2         | 128     | 3 211      |  -      |  -        |             |             | 2026-09-01 15:23 |
| qwen3:8b     | 4096   | medium | non-stream | GPU    | vLLM 0.22.0     |  -            | 133.2        | 128     | 964        |  -      |  -        |             |             | 2026-09-01 15:23 |
| qwen3:8b     | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **131.2**    | **128** | **978**    | ** - ** | ** - **   | -1.5%       |             | 2026-09-01 15:23 |
| qwen3:8b     | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 169.0         | 96.7         | 128     | 3 241      |  -      |  -        |             |             | 2026-09-01 13:46 |
| qwen3:8b     | 4096   | medium | stream     | GPU    | vLLM 0.22.0     | 1 234.0       | 139.6        | 128     | 970        |  -      |  -        |             |             | 2026-09-01 13:46 |
| qwen3:8b     | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **583.7**     | **138.7**    | **128** | **1 014**  | ** - ** | ** - **   | -0.6%       |             | 2026-09-01 13:46 |
| qwen3:8b     | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 63.3         | 128     | 3 185      |  -      |  -        |             |             | 2026-09-01 12:08 |
| qwen3:8b     | 4096   | short  | non-stream | GPU    | vLLM 0.22.0     |  -            | 133.8        | 128     | 959        |  -      |  -        |             |             | 2026-09-01 12:08 |
| qwen3:8b     | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **137.5**    | **128** | **933**    | ** - ** | ** - **   | **+2.8%**   |             | 2026-09-01 12:08 |
| qwen3:8b     | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 141.9         | 4.6          | 128     | 30 746     | 1 045   | 8.166     |             |             | 2026-08-18 01:14 |
| qwen3:8b     | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **98.7**      | **5.0**      | **128** | **26 755** | **875** | **6.836** | **+8.9%**   | **+19.4%**  | 2026-08-18 01:14 |
| qwen3:8b     | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 73.4          | 96.7         | 128     | 3 221      |  -      |  -        |             |             | 2026-09-01 10:30 |
| qwen3:8b     | 4096   | short  | stream     | GPU    | vLLM 0.22.0     | 535.1         | 139.6        | 128     | 12 064     |  -      |  -        |             |             | 2026-09-01 10:30 |
| qwen3:8b     | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **355.0**     | **145.4**    | **128** | **954**    | ** - ** | ** - **   | **+4.2%**   |             | 2026-09-01 10:30 |
| qwen3:8b     | 16384  | long   | stream     | GPU    | Ollama 0.32.6   | 30 059.9      | 136.3        | 128     | 3 127      | 368     | 2.871     |             |             | 2026-08-18 01:14 |
| qwen3:8b     | 16384  | long   | stream     | GPU    | **loken 0.1.0** | **29 441.2**  | **140.2**    | **128** | **981**    | **236** | **1.846** | **+2.8%**   | **+55.5%**  | 2026-08-18 01:14 |
| qwen3:8b     | 16384  | short  | stream     | GPU    | Ollama 0.32.6   | 4 162.8       | 138.9        | 128     | 2 991      | 354     | 2.765     |             |             | 2026-08-18 01:14 |
| qwen3:8b     | 16384  | short  | stream     | GPU    | **loken 0.1.0** | **2 503.4**   | **149.6**    | **128** | **936**    | **220** | **1.717** | **+7.7%**   | **+61.0%**  | 2026-08-18 01:14 |
| qwen3:8b     | 32768  | long   | stream     | GPU    | Ollama 0.32.6   | 29 980.9      | 135.9        | 128     | 3 119      | 364     | 2.844     |             |             | 2026-08-18 01:14 |
| qwen3:8b     | 32768  | long   | stream     | GPU    | **loken 0.1.0** | **21 105.0**  | **140.3**    | **128** | **981**    | **238** | **1.860** | **+3.3%**   | **+52.9%**  | 2026-08-18 01:14 |
| qwen3:8b     | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 62.4         | 128     | 3 260      |  -      |  -        |             |             | 2026-09-01 22:47 |
| qwen3:8b     | 131072 | short  | non-stream | GPU    | vLLM 0.22.0     |  -            | 132.1        | 128     | 971        |  -      |  -        |             |             | 2026-09-01 22:47 |
| qwen3:8b     | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **137.4**    | **128** | **935**    | ** - ** | ** - **   | **+4.0%**   |             | 2026-09-01 22:47 |
| qwen3:8b     | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 64.2          | 96.8         | 128     | 3 274      |  -      |  -        |             |             | 2026-09-01 20:30 |
| qwen3:8b     | 131072 | short  | stream     | GPU    | vLLM 0.22.0     | 758.9         | 139.4        | 128     | 967        |  -      |  -        |             |             | 2026-09-01 20:30 |
| qwen3:8b     | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **369.5**     | **145.7**    | **128** | **953**    | ** - ** | ** - **   | **+4.5%**   |             | 2026-09-01 20:30 |
| qwen3:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **2 378.0**   | **141.1**    | **128** | **1 024**  | **220** | **1.716** |             |             | 2026-08-18 01:14 |

### qwen35

![qwen35](img/family-qwen35.svg)

| Model          | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token | Δ decode    | Δ energy | Date             |
|----------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|---------|-------------|----------|------------------|
| qwen3.5:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 46.1         | 128     | 4 285     |  -      |  -      |             |          | 2026-09-01 18:35 |
| qwen3.5:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **99.3**     | **128** | **1 355** | ** - ** | ** - ** | **+115.1%** |          | 2026-09-01 18:35 |
| qwen3.5:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 635.0         | 76.0         | 128     | 4 262     |  -      |  -      |             |          | 2026-09-01 16:56 |
| qwen3.5:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **6 934.5**   | **114.7**    | **128** | **1 405** | ** - ** | ** - ** | **+50.9%**  |          | 2026-09-01 16:56 |
| qwen3.5:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 46.2         | 128     | 4 211     |  -      |  -      |             |          | 2026-09-01 15:19 |
| qwen3.5:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **108.5**    | **128** | **1 242** | ** - ** | ** - ** | **+135.0%** |          | 2026-09-01 15:19 |
| qwen3.5:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 96.4          | 76.3         | 128     | 4 281     |  -      |  -      |             |          | 2026-09-01 13:43 |
| qwen3.5:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 212.0**   | **123.7**    | **128** | **1 287** | ** - ** | ** - ** | **+62.2%**  |          | 2026-09-01 13:43 |
| qwen3.5:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 45.5         | 128     | 4 292     |  -      |  -      |             |          | 2026-09-01 12:05 |
| qwen3.5:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **107.3**    | **128** | **1 272** | ** - ** | ** - ** | **+135.7%** |          | 2026-09-01 12:05 |
| qwen3.5:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 34.5          | 75.2         | 128     | 4 250     |  -      |  -      |             |          | 2026-09-01 10:25 |
| qwen3.5:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **390.1**     | **126.1**    | **128** | **1 267** | ** - ** | ** - ** | **+67.6%**  |          | 2026-09-01 10:25 |
| qwen3.5:latest | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 45.3         | 128     | 4 421     |  -      |  -      |             |          | 2026-09-01 22:40 |
| qwen3.5:latest | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **108.0**    | **128** | **1 254** | ** - ** | ** - ** | **+138.6%** |          | 2026-09-01 22:40 |
| qwen3.5:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 33.2          | 75.0         | 128     | 4 370     |  -      |  -      |             |          | 2026-09-01 20:27 |
| qwen3.5:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **427.6**     | **125.9**    | **128** | **1 270** | ** - ** | ** - ** | **+67.8%**  |          | 2026-09-01 20:27 |

### qwen35moe

![qwen35moe](img/family-qwen35moe.svg)

| Model       | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token | Δ decode   | Δ energy | Date             |
|-------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|---------|------------|----------|------------------|
| qwen3.5:35b | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 57.7         | 128     | 20 325    |  -      |  -      |            |          | 2026-09-01 18:34 |
| qwen3.5:35b | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **102.2**    | **128** | **1 475** | ** - ** | ** - ** | **+77.2%** |          | 2026-09-01 18:34 |
| qwen3.5:35b | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 637.1         | 116.9        | 128     | 4 980     |  -      |  -      |            |          | 2026-09-01 16:55 |
| qwen3.5:35b | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **7 044.0**   | **125.2**    | **128** | **1 484** | ** - ** | ** - ** | **+7.1%**  |          | 2026-09-01 16:55 |
| qwen3.5:35b | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 58.7         | 128     | 5 103     |  -      |  -      |            |          | 2026-09-01 15:18 |
| qwen3.5:35b | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **107.0**    | **128** | **1 355** | ** - ** | ** - ** | **+82.3%** |          | 2026-09-01 15:18 |
| qwen3.5:35b | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 91.0          | 113.0        | 128     | 20 424    |  -      |  -      |            |          | 2026-09-01 13:42 |
| qwen3.5:35b | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **1 066.1**   | **133.2**    | **128** | **1 366** | ** - ** | ** - ** | **+17.8%** |          | 2026-09-01 13:42 |
| qwen3.5:35b | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 55.9         | 128     | 20 500    |  -      |  -      |            |          | 2026-09-01 12:03 |
| qwen3.5:35b | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **108.7**    | **128** | **1 325** | ** - ** | ** - ** | **+94.4%** |          | 2026-09-01 12:03 |
| qwen3.5:35b | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 33.1          | 114.1        | 128     | 5 100     |  -      |  -      |            |          | 2026-09-01 10:24 |
| qwen3.5:35b | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **390.4**     | **134.3**    | **128** | **1 308** | ** - ** | ** - ** | **+17.6%** |          | 2026-09-01 10:24 |
| qwen3.5:35b | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 58.9         | 128     | 5 097     |  -      |  -      |            |          | 2026-09-01 22:34 |
| qwen3.5:35b | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **111.2**    | **128** | **1 279** | ** - ** | ** - ** | **+88.9%** |          | 2026-09-01 22:34 |
| qwen3.5:35b | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 32.8          | 113.7        | 128     | 5 120     |  -      |  -      |            |          | 2026-09-01 20:26 |
| qwen3.5:35b | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **221.2**     | **132.6**    | **128** | **1 374** | ** - ** | ** - ** | **+16.6%** |          | 2026-09-01 20:26 |

### qwen3moe

![qwen3moe](img/family-qwen3moe.svg)

| Model           | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms    | J/req   | J/token | Δ decode   | Δ energy | Date             |
|-----------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|-----------|---------|---------|------------|----------|------------------|
| qwen3-coder:30b | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 83.3         | 128     | 4 047     |  -      |  -      |            |          | 2026-09-01 18:30 |
| qwen3-coder:30b | 4096   | long   | non-stream | GPU    | vLLM 0.22.0     |  -            | 129.0        | 128     | 1 004     |  -      |  -      |            |          | 2026-09-01 18:30 |
| qwen3-coder:30b | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **97.1**     | **128** | **1 495** | ** - ** | ** - ** | -24.7%     |          | 2026-09-01 18:30 |
| qwen3-coder:30b | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 1 262.5       | 145.3        | 128     | 3 986     |  -      |  -      |            |          | 2026-09-01 16:52 |
| qwen3-coder:30b | 4096   | long   | stream     | GPU    | vLLM 0.22.0     | 10 217.4      | 138.7        | 128     | 1 017     |  -      |  -      |            |          | 2026-09-01 16:52 |
| qwen3-coder:30b | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **611.6**     | **173.8**    | **128** | **1 524** | ** - ** | ** - ** | **+19.6%** |          | 2026-09-01 16:52 |
| qwen3-coder:30b | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 85.3         | 128     | 3 983     |  -      |  -      |            |          | 2026-09-01 15:16 |
| qwen3-coder:30b | 4096   | medium | non-stream | GPU    | vLLM 0.22.0     |  -            | 127.6        | 128     | 1 009     |  -      |  -      |            |          | 2026-09-01 15:16 |
| qwen3-coder:30b | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **127.8**    | **128** | **1 201** | ** - ** | ** - ** | **+0.1%**  |          | 2026-09-01 15:16 |
| qwen3-coder:30b | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 218.4         | 146.5        | 128     | 3 988     |  -      |  -      |            |          | 2026-09-01 13:38 |
| qwen3-coder:30b | 4096   | medium | stream     | GPU    | vLLM 0.22.0     | 1 901.8       | 138.7        | 128     | 995       |  -      |  -      |            |          | 2026-09-01 13:38 |
| qwen3-coder:30b | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **451.1**     | **177.4**    | **128** | **1 211** | ** - ** | ** - ** | **+21.1%** |          | 2026-09-01 13:38 |
| qwen3-coder:30b | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 85.9         | 128     | 3 943     |  -      |  -      |            |          | 2026-09-01 12:00 |
| qwen3-coder:30b | 4096   | short  | non-stream | GPU    | vLLM 0.22.0     |  -            | 129.7        | 128     | 996       |  -      |  -      |            |          | 2026-09-01 12:00 |
| qwen3-coder:30b | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **134.8**    | **128** | **1 170** | ** - ** | ** - ** | **+3.9%**  |          | 2026-09-01 12:00 |
| qwen3-coder:30b | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 65.1          | 147.4        | 128     | 3 967     |  -      |  -      |            |          | 2026-09-01 10:22 |
| qwen3-coder:30b | 4096   | short  | stream     | GPU    | vLLM 0.22.0     | 635.8         | 141.1        | 128     | 968       |  -      |  -      |            |          | 2026-09-01 10:22 |
| qwen3-coder:30b | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **267.1**     | **178.8**    | **128** | **1 178** | ** - ** | ** - ** | **+21.3%** |          | 2026-09-01 10:22 |
| qwen3-coder:30b | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 52.2         | 128     | 5 448     |  -      |  -      |            |          | 2026-09-01 22:24 |
| qwen3-coder:30b | 131072 | short  | non-stream | GPU    | vLLM 0.22.0     |  -            | 131.1        | 128     | 992       |  -      |  -      |            |          | 2026-09-01 22:24 |
| qwen3-coder:30b | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **139.4**    | **128** | **1 057** | ** - ** | ** - ** | **+6.3%**  |          | 2026-09-01 22:24 |
| qwen3-coder:30b | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 61.7          | 84.2         | 128     | 5 453     |  -      |  -      |            |          | 2026-09-01 20:23 |
| qwen3-coder:30b | 131072 | short  | stream     | GPU    | vLLM 0.22.0     | 638.5         | 138.3        | 128     | 1 003     |  -      |  -      |            |          | 2026-09-01 20:23 |
| qwen3-coder:30b | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **159.6**     | **178.8**    | **128** | **1 201** | ** - ** | ** - ** | **+29.3%** |          | 2026-09-01 20:23 |

### qwen3next

![qwen3next](img/family-qwen3next.svg)

| Model                   | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token | Δ decode | Δ energy | Date             |
|-------------------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|---------|----------|----------|------------------|
| qwen3-coder-next:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 19.4         | 128     | 10 999     |  -      |  -      |          |          | 2026-09-01 18:26 |
| qwen3-coder-next:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **8.8**      | **128** | **25 960** | ** - ** | ** - ** | -54.7%   |          | 2026-09-01 18:26 |
| qwen3-coder-next:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 581.1         | 28.4         | 128     | 11 030     |  -      |  -      |          |          | 2026-09-01 16:49 |
| qwen3-coder-next:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **4 562.3**   | **9.5**      | **128** | **22 871** | ** - ** | ** - ** | -66.6%   |          | 2026-09-01 16:49 |
| qwen3-coder-next:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 19.4         | 128     | 10 592     |  -      |  -      |          |          | 2026-09-01 15:12 |
| qwen3-coder-next:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **8.2**      | **21**  | **11 599** | ** - ** | ** - ** |          |          | 2026-09-01 15:12 |
| qwen3-coder-next:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 92.0          | 28.4         | 128     | 10 639     |  -      |  -      |          |          | 2026-09-01 13:34 |
| qwen3-coder-next:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **853.2**     | **9.1**      | **128** | **23 020** | ** - ** | ** - ** | -68.1%   |          | 2026-09-01 13:34 |
| qwen3-coder-next:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 19.4         | 128     | 10 435     |  -      |  -      |          |          | 2026-09-01 11:56 |
| qwen3-coder-next:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **9.1**      | **128** | **23 265** | ** - ** | ** - ** | -53.0%   |          | 2026-09-01 11:56 |
| qwen3-coder-next:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 33.4          | 28.1         | 128     | 10 488     |  -      |  -      |          |          | 2026-09-01 10:17 |
| qwen3-coder-next:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **237.7**     | **9.2**      | **128** | **23 157** | ** - ** | ** - ** | -67.1%   |          | 2026-09-01 10:17 |
| qwen3-coder-next:latest | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 18.0         | 128     | 10 552     |  -      |  -      |          |          | 2026-09-01 22:14 |
| qwen3-coder-next:latest | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **9.0**      | **128** | **23 632** | ** - ** | ** - ** | -49.9%   |          | 2026-09-01 22:14 |
| qwen3-coder-next:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 34.8          | 26.0         | 128     | 10 552     |  -      |  -      |          |          | 2026-09-01 20:19 |
| qwen3-coder-next:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **257.8**     | **9.2**      | **128** | **23 862** | ** - ** | ** - ** | -64.6%   |          | 2026-09-01 20:19 |
| qwen3next:latest        | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 22.2         | 128     | 10 157     |  -      |  -      |          |          | 2026-09-01 18:43 |
| qwen3next:latest        | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **9.3**      | **128** | **23 860** | ** - ** | ** - ** | -58.0%   |          | 2026-09-01 18:43 |
| qwen3next:latest        | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 625.2         | 32.6         | 128     | 10 165     |  -      |  -      |          |          | 2026-09-01 17:04 |
| qwen3next:latest        | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **5 573.9**   | **9.7**      | **128** | **22 664** | ** - ** | ** - ** | -70.4%   |          | 2026-09-01 17:04 |
| qwen3next:latest        | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 22.2         | 128     | 9 826      |  -      |  -      |          |          | 2026-09-01 15:27 |
| qwen3next:latest        | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **9.5**      | **128** | **21 099** | ** - ** | ** - ** | -57.1%   |          | 2026-09-01 15:27 |
| qwen3next:latest        | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 98.7          | 32.5         | 128     | 9 935      |  -      |  -      |          |          | 2026-09-01 13:51 |
| qwen3next:latest        | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **570.5**     | **9.5**      | **128** | **23 900** | ** - ** | ** - ** | -70.7%   |          | 2026-09-01 13:51 |
| qwen3next:latest        | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 22.2         | 128     | 9 696      |  -      |  -      |          |          | 2026-09-01 12:13 |
| qwen3next:latest        | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **9.6**      | **128** | **21 282** | ** - ** | ** - ** | -56.6%   |          | 2026-09-01 12:13 |
| qwen3next:latest        | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 36.1          | 32.9         | 128     | 9 659      |  -      |  -      |          |          | 2026-09-01 10:34 |
| qwen3next:latest        | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **169.3**     | **9.9**      | **128** | **20 982** | ** - ** | ** - ** | -70.0%   |          | 2026-09-01 10:34 |
| qwen3next:latest        | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 19.8         | 128     | 9 968      |  -      |  -      |          |          | 2026-09-01 22:51 |
| qwen3next:latest        | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **9.7**      | **128** | **19 529** | ** - ** | ** - ** | -50.9%   |          | 2026-09-01 22:51 |
| qwen3next:latest        | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 37.6          | 28.4         | 128     | 10 049     |  -      |  -      |          |          | 2026-09-01 20:35 |
| qwen3next:latest        | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **178.0**     | **9.9**      | **128** | **20 589** | ** - ** | ** - ** | -65.3%   |          | 2026-09-01 20:35 |

### smollm3

![smollm3](img/family-smollm3.svg)

| Model          | Ctx    | Prompt | Mode       | Device | Engine          | Prefill tok/s | Decode tok/s | Tokens  | E2E ms     | J/req   | J/token   | Δ decode    | Δ energy | Date             |
|----------------|--------|--------|------------|--------|-----------------|---------------|--------------|---------|------------|---------|-----------|-------------|----------|------------------|
| smollm3:latest | 4096   | long   | non-stream | GPU    | Ollama 0.32.6   |  -            | 96.6         | 128     | 2 454      |  -      |  -        |             |          | 2026-09-01 18:44 |
| smollm3:latest | 4096   | long   | non-stream | GPU    | **loken 0.1.0** | ** - **       | **225.2**    | **128** | **581**    | ** - ** | ** - **   | **+133.2%** |          | 2026-09-01 18:44 |
| smollm3:latest | 4096   | long   | stream     | GPU    | Ollama 0.32.6   | 1 027.5       | 171.4        | 128     | 2 460      |  -      |  -        |             |          | 2026-09-01 17:05 |
| smollm3:latest | 4096   | long   | stream     | GPU    | **loken 0.1.0** | **4 820.7**   | **262.9**    | **128** | **584**    | ** - ** | ** - **   | **+53.4%**  |          | 2026-09-01 17:05 |
| smollm3:latest | 4096   | medium | non-stream | GPU    | Ollama 0.32.6   |  -            | 93.7         | 128     | 2 497      |  -      |  -        |             |          | 2026-09-01 15:28 |
| smollm3:latest | 4096   | medium | non-stream | GPU    | **loken 0.1.0** | ** - **       | **224.4**    | **128** | **581**    | ** - ** | ** - **   | **+139.4%** |          | 2026-09-01 15:28 |
| smollm3:latest | 4096   | medium | stream     | GPU    | Ollama 0.32.6   | 161.2         | 169.6        | 128     | 2 500      |  -      |  -        |             |          | 2026-09-01 13:52 |
| smollm3:latest | 4096   | medium | stream     | GPU    | **loken 0.1.0** | **635.8**     | **251.8**    | **128** | **616**    | ** - ** | ** - **   | **+48.5%**  |          | 2026-09-01 13:52 |
| smollm3:latest | 4096   | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 96.3         | 128     | 2 461      |  -      |  -        |             |          | 2026-09-01 12:14 |
| smollm3:latest | 4096   | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **214.0**    | **128** | **610**    | ** - ** | ** - **   | **+122.2%** |          | 2026-09-01 12:14 |
| smollm3:latest | 4096   | short  | stream     | CPU    | Ollama 0.32.6   | 961.9         | 11.0         | 128     | 13 711     | 504     | 3.935     |             |          | 2026-08-18 01:14 |
| smollm3:latest | 4096   | short  | stream     | CPU    | **loken 0.1.0** | **192.4**     | **10.6**     | **128** | **12 517** | **558** | **4.358** | -3.5%       | -9.7%    | 2026-08-18 01:14 |
| smollm3:latest | 4096   | short  | stream     | GPU    | Ollama 0.32.6   | 55.8          | 170.9        | 128     | 2 491      |  -      |  -        |             |          | 2026-09-01 10:36 |
| smollm3:latest | 4096   | short  | stream     | GPU    | **loken 0.1.0** | **461.2**     | **256.7**    | **128** | **578**    | ** - ** | ** - **   | **+50.2%**  |          | 2026-09-01 10:36 |
| smollm3:latest | 131072 | short  | non-stream | GPU    | Ollama 0.32.6   |  -            | 95.7         | 128     | 2 539      |  -      |  -        |             |          | 2026-09-01 22:54 |
| smollm3:latest | 131072 | short  | non-stream | GPU    | **loken 0.1.0** | ** - **       | **223.4**    | **128** | **589**    | ** - ** | ** - **   | **+133.3%** |          | 2026-09-01 22:54 |
| smollm3:latest | 131072 | short  | stream     | GPU    | Ollama 0.32.6   | 53.7          | 171.6        | 128     | 2 531      |  -      |  -        |             |          | 2026-09-01 20:36 |
| smollm3:latest | 131072 | short  | stream     | GPU    | **loken 0.1.0** | **415.5**     | **256.5**    | **128** | **578**    | ** - ** | ** - **   | **+49.5%**  |          | 2026-09-01 20:36 |
