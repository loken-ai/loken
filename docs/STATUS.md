# Status

What is measured, what is slower, what has never run. Revised 2026-08-29.

Every figure comes from one machine: RTX 5070 Ti + RTX 5060 Ti (16 GB each), Linux, CUDA 13.3,
models on a USB disk reading at 460 MB/s.

## Decode rate vs ollama

**Superseded, and kept as evidence rather than as a claim.** These rows were measured on
2026-08-27, during a phase of very active development: kernels, placement and the rate meter
have all changed since. A fresh campaign is needed before any of this is quoted. Conditions
were one card for both engines, idle machine, cold, greedy, streamed, short prompt, 4096 ctx,
best of three, decode tokens/s.

![Decode rate against ollama](img/decode-vs-ollama.svg)

<!-- table:decode -->
| model | ollama | loken | |
|---|---:|---:|---|
| gpt-oss:20b | 131.5 | **206.5** | +57% |
| llama3.2:1b | 297.7 | **450.1** | +51% |
| qwen2.5:0.5b | 302.0 | **368.4** | +22% |
| nemotron-3-nano:latest | 133.5 | **144.5** | +8% |
| qwen3:1.7b | 407.7 | **427.8** | +5% |
| olmoe:latest | **485.7** | 438.7 | **-10%** |
| granite3-moe:1b | **373.2** | 323.4 | **-13%** |
<!-- /table:decode -->

Both mixtures lose. olmoe by 10 percent, granite3-moe by 13, and eight of olmoe's ten points
predate this year's kernel work - a build from before it measured 455.2 - so its deficit is
standing and unexplained. That two mixtures and no dense model lose is the clue nobody has
followed.

The table and the figure are regenerated together from `docs/BENCHMARKS.md` by
`scripts/figures.py`. They were transcribed by hand once, and the page ended up quoting a
parity for granite3-moe that appears nowhere in the measurements.

## Placement, measured separately

Given both cards, ollama splits models that fit on one:

| model | ollama, 2 cards | loken, 2 cards |
|---|---:|---:|
| olmoe | 390.4 | **447.7** |
| llama3.2:1b | 231.3 | **453.2** |

The first table is the kernels. This one is the placement. Only the first says anything about
arithmetic.

## Costs

- **~870 MiB per load, any model.** CUDA contexts, cuBLAS workspace, preloaded module images.
  Fixed, not proportional - 893 and 867 MiB on models differing 2x in weights. The placement
  planner does not know about it.
- **First answer after load is slower.** A mixture pays ~0.5 s if a request arrives before the
  background repack finishes.
- **Load time is your disk.** 24 GB over USB: 52 s reading, 6 s to the cards.

## Written, wired to nothing

- **Layer scheduler** (`src/distributed/layer_scheduler.rs`) - one caller: itself. The
  distributed engine builds its plan by hand. Judged over 21 placement and 119 naming cases, with
  two perturbations proving the judge can fail.
- **Reserve pass** (`src/tensor/dry.rs`) - runs the forward on a device that allocates nothing,
  to replace the per-model formulas. Called from one site, in Z-Image. Every language-model
  placement is still budgeted by formula.

## Shipped as source, never compiled

The CUDA kernels carry AMD arms - MFMA on CDNA, WMMA on RDNA3+ - guarded and complete. There is
no hipcc build here, so none has been through a compiler. Treat AMD as source, not support.

## Next

A Vulkan backend. One generic compute backend reaches AMD, Intel and NVIDIA; HIP reaches one
vendor and a subset of its cards.

## Judged partially

- **Mat-vec core** - checked against an independent reference for 2 formats of 10. The production
  entry point does not reach the core.
- **Z-Image bends its schedule twice.** `new` derives timesteps from bent sigmas,
  `set_timesteps` from the pre-bend ramp. At 9 steps: `[1.0, 0.960129, 0.913349, ...]` against
  `[1.0, 0.96, 0.913043, ...]` for one bend. Only a render says which the checkpoint expects.
- **Image parity is a hash**, one prompt and seed per family - and not a property of the code
  alone. The placer reads free VRAM at load, and four runs of one binary gave four hashes. Run it
  on a quiet machine.

## Measured and refused

A 3-bit KV cache through a randomised Hadamard rotation. The rotation is orthogonal and does
spread outliers (excess kurtosis 35.7 -> -0.6); against f16 it delivers what it promises. Against
the Q4_0 cache already here it does not: token-axis grouping removes the outliers first, so
rotating changes the error by under 0.5%, and Q4_0 is 2.6x more accurate for one more bit. The
harness stays, wired to nothing.

## Borrowed code

One file shares body lines with upstream, at 58%, recorded in [`NOTICE.md`](../NOTICE.md). Its
largest matched range is the GGUF k-quant bit specification, which any correct reader reproduces.

## Open

olmoe's deficit, the 870 MiB nobody budgets, and the reserve pass reaching language models. Each
has a measurement waiting.
