# Status

What is measured, what is slower, what has never run. Revised 2026-09-03.

Every figure comes from one machine: RTX 5070 Ti + RTX 5060 Ti (16 GB each), Linux, CUDA 13.3,
models on a USB disk reading at 460 MB/s.

## Decode rate vs ollama

**Out of date, and kept as evidence rather than as a claim.** These rows were measured on
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

The two mixtures that lose here do not lose. Both rows came from a harness that had moved
into its own repository and started the engine without its configuration; measured configured
on 2026-09-02, olmoe is +17.4% and granite3-moe +0.2%. The clue was real, though, and pointed
elsewhere: the mixtures that do lose are the ones too large for the cards. A layer was placed
whole, so a spilled mixture ran attention on the host for the sake of expert weights that are
97% of its bytes and read a few at a time. Spilling the experts and keeping everything else on
the cards took qwen3next from 15.6 to 34.7 tok/s against ollama's 33, and qwen3-coder-next from
10.7 to 30.8 against 26-28. A mixture that already fits is untouched by it.

The table and the figure are regenerated together from `docs/BENCHMARKS.md` by
`scripts/figures.py`. They were transcribed by hand once, and the page ended up quoting a
parity for granite3-moe that appears nowhere in the measurements.

## Placement, measured separately

Given both cards, ollama splits models that fit on one; this engine keeps them on the fast
card, and pays nothing for having the second one present. Measured 2026-09-03, olmoe: 464 tok/s
on card 0 alone, 462 with both cards visible, 252 on card 1 alone - the placer picks the fast
card, and the 390-against-496 penalty an earlier revision of this page reported is gone.

For a model that fits on neither card the question is different, and the answer is the bus.
deepseek-r1:70b at Q4_K_M holds 26 of its 80 layers on the host, every one of them read in
full per token at ~36 GB/s, which is what this machine's DRAM gives: 1.51 tok/s here, 1.55
under ollama, both engines pinned to the same ceiling. The way out of that cell is a variant
that fits, not a faster host path - the FFN-at-Q3 requantisation the store already names but
never received weights.

The first table is the kernels. This one is the placement. Only the first says anything about
arithmetic.

## Costs

- **~870 MiB per load, any model.** CUDA contexts, cuBLAS workspace, preloaded module images.
  Fixed, not proportional - 893 and 867 MiB on models differing 2x in weights. The placement
  planner does not know about it.
- **First answer after load is slower.** A mixture pays ~0.5 s if a request arrives before the
  background repack finishes - when it fits. A spilled one pays far more: qwen3next decodes its
  first request at 2.0 tok/s and the next at 34.7. The benchmark table takes the median over
  its iterations, so that cost no longer reaches a published rate; a user's first answer still
  pays it.
- **A spilled model was double-held in host RAM.** The file was prefetched whole before
  placement, so the 28 GB already on the cards stayed cached beside the 14 GB the host actually
  reads, and a 64 GB box swapped 16 GB during the decode being timed. The advice now follows
  the plan - DontNeed for what the cards hold, WillNeed for what the host reads - and the
  process keeps its resident pages. What remains resident and could go: the token embedding
  dequantised to F32 on the host (4 GB for a 0.6 GB Q4_K table), and the file pages of host
  layers once their repack exists.
- **Load time is your disk.** 24 GB over USB: 52 s reading, 6 s to the cards.

## The cluster

Two halves live in [`CLUSTER.md`](CLUSTER.md), and only one of them runs.

**Replicated serving runs.** Any node is an entry point and an entry point holds nothing: a
client talks to whichever node it knows, that node forwards the whole HTTP request to the best
holder and relays the stream back. Membership is SWIM gossip with a phi-accrual detector,
seeded by multicast announcement or by a `join` list. `/api/cluster/state`, `/api/cluster/peers`
and `/api/cluster/prefix` report it. No consensus protocol: routing to a dead node costs a
retry, not a corruption.

A hand-over is priced on rates each node measures from its own completed generations, never
from hardware nameplates - what a card could do is not what this build achieves on this model
at this quantisation.

**Sharding does not.** Pipeline parallelism across hosts, the binary data plane it needs, the
link-cost matrix, the tiered KV, the resume path and the layer scheduler are all written and
reached by nothing. That is not a claim from reading: `distributed::wiring_gate` records the
state of every module in that directory and fails both when one of them is finally reached and
when one that was reached falls silent. Cross-host layer execution refuses loudly rather than
returning a placeholder.

**One defect found, and its fix unverified at two nodes.** A node published a capacity derived
from one generation's decode rate multiplied by its lane count, while its generations serialise
behind the model lock. Two comparable machines therefore priced each other an order of
magnitude apart, and the cluster handed over a few requests where it should have split the work
roughly in half. The meter now publishes what a window of real completions produced. The second
machine left the bench before the fix could be measured, so **the two-node gain is currently
unmeasured** - the correction is judged by trace replay, not by a cluster.

**Open.** A peer that has never answered is priced from a floor rather than from its catalogue,
so a cold node looks worse than an idle one that holds nothing. And a replica set keyed on a
placeholder digest - what models cached from Hugging Face carry - groups every such model into
one set.

## Written, wired to nothing

- **Layer scheduler** (`src/distributed/layer_scheduler.rs`) - one caller: itself, since the
  engine that used to build a plan by hand was removed. Judged over 21 placement and 119 naming
  cases, with two perturbations proving the judge can fail.
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

The 870 MiB nobody budgets, and the reserve pass reaching language models. Each has a
measurement waiting. olmoe's deficit is closed: it was the harness, not the engine.
