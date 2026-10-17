# Roadmap

What the next releases bring, feature by feature. [`CHANGELOG.md`](CHANGELOG.md) lists what is
done; [`docs/STATUS.md`](docs/STATUS.md) says what is measured. A feature is done when its
acceptance is met and measured; until then it stays here with its state.

## 0.3.0

| Feature | Acceptance | State |
|---|---|---|
| One model across several machines, cut by layer | A model no single machine serves well, cut across machines from each one's measured layer cost, faster than the best machine alone | Done: three machines, peers measured on a sample of layers |
| Tensor parallelism in the generic path | A layer's dense projections split across cards inside the one placement path, any number of cards, mixtures included | Done: dense and mixture layers on any number of cards, recorded in the decode graph, chosen from measured sums and products; GELU, fused or clamped experts stay layered |
| Expert parallelism | A mixture layer's experts split across cards where measured links make it pay | Done for experts that would otherwise sit on the host |
| Pipeline parallelism across machines | A chain of any length, the prompt-lookup verify crossing every cut | Done |
| Data parallelism | Copies of a model served side by side, requests spread over them | Done on one machine; across machines through the cluster's routing |
| Context parallelism | A long context's attention split across cards | Done: each attention layer's cache cut by ranges of positions over the cards, the parts combined on the layer's card, the cut layers outside the recorded step; a request reserves its ranges before its prefill; a 68k-token prompt served on two cards, prefill 55 s against 33 and decode 5.9 tok/s against 10 for one cache |
| Prefill and decode on different instances | A prompt read on one instance and decoded on another, its state handed over | Done: the KV cache handed over exactly for attention models; recurrent and windowed models hand over their whole decode state; split layers and cut caches still read their prompt where they decode |
| CUDA on the published cudarc crate | No patched fork, graphs still recorded and replayed | Done |
| A Linux binary | One archive that starts on a machine without CUDA libraries and serves on the CPU, and a second that adds them for the cards | Done |
| CPU kernels chosen at run time | One binary serving AVX2 and AVX-512 machines, the level capped by configuration | Done |
| Qwen3.8 at full speed | The IQ1_M and UD-Q4_K_XL Qwen3.8 files at or above the reference engine on one and two cards | In progress: IQ1_M on two cards 72-97 tok/s against 78-85, on one card 54-77 tok/s (median 58) against 70-82; UD-Q4_K_XL on one card warm 15-18 tok/s against 16-18, prefill 15-18 s against 20; on two cards 14-15 tok/s, where the reference cannot load it (its expert arena needs 72 GiB of RAM); after a full prefill on one card 16-18 tok/s against 16-18 with its most routed experts held in host memory, the first request 11 |
| A coding agent on large models | verve completing a multi-step task on the IQ1_M and UD-Q4_K_XL Qwen3.8 files and on DeepSeek V4.1, tool calls and reasoning kept apart | Done |
| A faster test suite | The checks a change must pass in a fraction of today's time | Done: about a quarter, the rebuild in place and the slowest tests' setup shared or parallel |

## 0.4.0

Major, by priority:

| Feature | Acceptance | State |
|---|---|---|
| ROCm and Vulkan backends | One model of each family served through ROCm on an AMD card and through Vulkan on an NVIDIA card, an AMD card and an Intel GPU that shares the host's memory, by the same placement path, measured against the CUDA path and the reference engine on the same cards | Planned; an OpenCL path exists for the image models |
| Coding agents beyond verve | opencode, Claude Code and Cline completing the verve task on a Qwen3.8 file and on DeepSeek V4.1, each through its own protocol | Planned |
| Forecasting and tabular models | Time series forecasts and tabular predictions served beside the language models, measured against naive baselines | Planned |

Smaller, by priority:

| Feature | Acceptance | State |
|---|---|---|
| AVX-512 host kernels for the IQ formats | The IQ2_S, IQ3_XXS and IQ4_NL expert products faster on an AVX-512 host than its AVX2 path, measured on the same host | Planned |
| The OpenCL path brought back | A language and an image model served on an Intel GPU that shares the host's memory through the OpenCL path, that memory counted once by the placement, measured against the host | Partial: OpenCL kernels for the image models and a speculative driver, untouched since September |
| A routing profile kept per model | The profile of each model written and read back without configuration, so a restart starts with a warm expert cache | Planned |
| Qwen3.8 IQ1_M on one card at the reference | The resumed-prompt decode at or above the reference engine's on one card | Planned |
| Context parallelism inside the recorded step | The decode of a context cut over the cards replayed as a graph, within a tenth of an uncut cache's speed | Planned |

## 0.5.0

Major, by priority:

| Feature | Acceptance | State |
|---|---|---|
| Throughput serving | Continuous batching with chunked prefill and multi-step scheduling: a concurrent load of many requests served at the reference server's throughput on the same card | Planned |
| Video input | A video turn read by a model's vision tower, frames sampled by the model's own rule, answers matching the reference implementation | Planned |
| Speculation for every model | A draft model or head found or built for any served model, its windows verified as a tree, faster than plain decode on each family | Planned |

Smaller, by priority:

| Feature | Acceptance | State |
|---|---|---|
| Sleep mode | A model unloaded to host memory on idle and back on the next request, the reload time logged | Planned |
| Per-request metrics | Queue, prefill, decode and cache figures per request on the metrics endpoint | Planned |
| Beam search | n beams with length penalty, the result equal to the reference implementation's | Planned |
| Prompt embeddings | A request that hands embeddings instead of tokens, answered as for the tokens they stand for | Planned |

## 0.6.0

Major, by priority:

| Feature | Acceptance | State |
|---|---|---|
| A cluster that survives a node | A node lost mid-request re-planned onto the others and the request finished, the loss reported | Planned |
| Tensor parallelism across machines | One layer split over cards of several machines where the measured links pay, faster than the layer cut alone | Planned |
| More weight formats | GPTQ, AWQ and BitsAndBytes checkpoints loaded through the same placement, outputs matching the reference implementation | Planned |

Smaller, by priority:

| Feature | Acceptance | State |
|---|---|---|
| Adapters per request | Several text adapters served over one base model, each request naming its own | Planned |
| Pooling and classification | The classify, score and pooling routes over the loaded models | Planned |
| Encoder-decoder models | A sequence-to-sequence family served for text and speech, matching the reference implementation | Planned |

## 0.7.0

Major, by priority:

| Feature | Acceptance | State |
|---|---|---|
| A native Windows server | The server built, installed and run as a service on Windows with CUDA, at the Linux build's speed on the same card | Planned |
| Quantisation from the command line | A checkpoint quantised to a chosen format with an importance matrix, its quality scored against the original | Planned |

Smaller, by priority:

| Feature | Acceptance | State |
|---|---|---|
| Web search and fetch tools | Built-in tools a request may enable, their calls and results logged as tool turns | Planned |

## 0.8.0

Major, by priority:

| Feature | Acceptance | State |
|---|---|---|
| Fine-tuning of adapters | A text adapter trained on the local cards from a dataset, served at once, its loss and quality measured | Planned |
| loken as a tool server | loken's models and media pipelines offered to agents as MCP tools, each call within the server's limits | Planned |

Smaller, by priority:

| Feature | Acceptance | State |
|---|---|---|
| Energy-aware placement | A placement that trades speed for watts when asked, both measured | Planned |
| Quotas per key | Token and request budgets per API key, enforced and reported | Planned |

## Waiting for hardware

Features that cannot be measured on the machines at hand; each moves into a version once a machine
to measure it on is available.

| Feature | Acceptance | Needs |
|---|---|---|
| A backend for Apple silicon | One model of each family served through Metal on an M-series Mac by the same placement path, measured against the reference engine on that Mac | An M-series Mac |
