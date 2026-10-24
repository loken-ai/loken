# Benchmarks - Intel(R) Arc(TM) Graphics on Intel(R) Core(TM) Ultra 7 155H

12 run(s), 2026-10-24 to 2026-10-24. loken builds measured: 0.3.1-dev. A row's `Date` and `Run` name the run behind it.

## Hardware

| | |
|---|---|
| GPU 0 | Intel(R) Arc(TM) Graphics |
| CPU | Intel(R) Core(TM) Ultra 7 155H - 22 threads |
| RAM | 30 GB |
| Backend | opencl |
| Kernel | 7.1.8+deb13-amd64 |

## Results

Method and column definitions: [BENCHMARKS.md](../BENCHMARKS.md). Each rate reads `cold / warm`: the first request of a cell, then the median of the later ones ([why](../BENCHMARKS.md#cold-and-warm)).

No reference engine runs a language model through OpenCL on a GPU (ollama and vLLM do not), so this page holds loken alone.

### qwen3

![qwen3](../img/family-qwen3.svg)

| Model      | Ctx    | Prompt | Mode       | Device | Engine              | Prefill tok/s   | TTFT ms       | Decode tok/s    | Tokens  | E2E ms    | J/req     | J/token    | Δ prefill | Δ TTFT | Δ decode   | Δ energy | Date             | Run      |
|------------|--------|--------|------------|--------|---------------------|-----------------|---------------|-----------------|---------|-----------|-----------|------------|-----------|--------|------------|----------|------------------|----------|
| qwen3:0.6b | 4096   | long   | non-stream | GPU    | **loken 0.3.1-dev** |  -              |  -            |  -              | **128** |  -        |  -        |  -         |           |        | incoherent |          | 2026-10-24 19:52 | 99d0f653 |
| qwen3:0.6b | 4096   | long   | stream     | GPU    | **loken 0.3.1-dev** |  -              |  -            |  -              | **128** |  -        |  -        |  -         |           |        | incoherent |          | 2026-10-24 19:51 | e191a09c |
| qwen3:0.6b | 4096   | medium | non-stream | GPU    | **loken 0.3.1-dev** |  -              |  -            | **22.9 / 26.4** | **128** | **5 091** | **1 422** | **11.111** |           |        |            |          | 2026-10-24 19:49 | 6519f266 |
| qwen3:0.6b | 4096   | medium | stream     | GPU    | **loken 0.3.1-dev** |  -              |  -            |  -              | **128** |  -        |  -        |  -         |           |        | incoherent |          | 2026-10-24 19:48 | 1e1ad579 |
| qwen3:0.6b | 4096   | short  | non-stream | GPU    | **loken 0.3.1-dev** |  -              |  -            |  -              | **128** |  -        |  -        |  -         |           |        | incoherent |          | 2026-10-24 19:47 | e3d73024 |
| qwen3:0.6b | 4096   | short  | stream     | GPU    | **loken 0.3.1-dev** | **21.0 / 22.4** | **714 / 669** | **34.3 / 45.6** | **128** | **3 775** | **879**   | **6.870**  |           |        |            |          | 2026-10-24 19:46 | 2b9c14ba |
| qwen3:0.6b | 131072 | long   | non-stream | GPU    | **loken 0.3.1-dev** |  -              |  -            |  -              | **128** |  -        |  -        |  -         |           |        | incoherent |          | 2026-10-24 20:00 | e5cd931a |
| qwen3:0.6b | 131072 | long   | stream     | GPU    | **loken 0.3.1-dev** |  -              |  -            |  -              | **128** |  -        |  -        |  -         |           |        | incoherent |          | 2026-10-24 19:58 | b4b2b892 |
| qwen3:0.6b | 131072 | medium | non-stream | GPU    | **loken 0.3.1-dev** |  -              |  -            | **23.0 / 26.8** | **128** | **5 047** | **1 351** | **10.554** |           |        |            |          | 2026-10-24 19:57 | 0fca2395 |
| qwen3:0.6b | 131072 | medium | stream     | GPU    | **loken 0.3.1-dev** |  -              |  -            |  -              | **128** |  -        |  -        |  -         |           |        | incoherent |          | 2026-10-24 19:55 | cd5945e5 |
| qwen3:0.6b | 131072 | short  | non-stream | GPU    | **loken 0.3.1-dev** |  -              |  -            |  -              | **128** |  -        |  -        |  -         |           |        | incoherent |          | 2026-10-24 19:54 | 3a7917a9 |
| qwen3:0.6b | 131072 | short  | stream     | GPU    | **loken 0.3.1-dev** | **21.1 / 22.3** | **712 / 672** | **34.6 / 45.8** | **128** | **3 759** | **992**   | **7.751**  |           |        |            |          | 2026-10-24 19:53 | 3c0869e3 |
