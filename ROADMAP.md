# Roadmap

What the next releases bring, feature by feature. [`CHANGELOG.md`](CHANGELOG.md) lists what is
done; [`docs/STATUS.md`](docs/STATUS.md) says what is measured. A feature is done when its
acceptance is met and measured; until then it stays here with its state.

A state is one word: Done, Partial (what is left is a feature of a later version), Started,
Planned, or Waiting (for the hardware to measure it on).

## 0.3.0

| Feature | Category | Acceptance | State |
|---|---|---|---|
| One model across several machines, cut by layer | Cluster | A model no single machine serves well, cut across machines from each one's measured layer cost, faster than the best machine alone | Done |
| Tensor parallelism in the generic path | Parallelism | A layer's dense projections split across cards inside the one placement path, any number of cards, mixtures included | Done |
| Expert parallelism | Parallelism | A mixture layer's experts split across cards where measured links make it pay | Partial |
| Pipeline parallelism across machines | Cluster | A chain of any length, the prompt-lookup verify crossing every cut | Done |
| Data parallelism | Parallelism | Copies of a model served side by side, requests spread over them | Done |
| Context parallelism | Parallelism | A long context's attention split across cards | Done |
| Prefill and decode on different instances | Serving | A prompt read on one instance and decoded on another, its state handed over | Partial |
| CUDA on the published cudarc crate | Backends | No patched fork, graphs still recorded and replayed | Done |
| A Linux binary | Packaging | One archive that starts on a machine without CUDA libraries and serves on the CPU, and a second that adds them for the cards | Done |
| CPU kernels chosen at run time | CPU | One binary serving AVX2 and AVX-512 machines, the level capped by configuration | Done |
| Qwen3.8 at full speed | Performance | The IQ1_M and UD-Q4_K_XL Qwen3.8 files at or above the reference engine on one and on two cards of a machine where the reference runs, both measured the same day | Partial |
| A coding agent on large models | Agents | verve completing a multi-step task on the IQ1_M and UD-Q4_K_XL Qwen3.8 files and on DeepSeek V4.1, tool calls and reasoning kept apart | Done |
| A faster test suite | Tests | The checks a change must pass in a fraction of today's time | Done |

## 0.4.0

Major, by priority:

| Feature | Category | Acceptance | State |
|---|---|---|---|
| ROCm and Vulkan backends | Backends | One model of each family served through ROCm on an AMD card and through Vulkan on an NVIDIA card, an AMD card and an Intel GPU that shares the host's memory, by the same placement path, measured against the CUDA path and the reference engine on the same cards | Started |
| Coding agents beyond verve | Agents | opencode, Claude Code and Cline completing the verve task on a Qwen3.8 file and on DeepSeek V4.1, each through its own protocol | Planned |
| Forecasting and tabular models | Models | Time series forecasts and tabular predictions served beside the language models, measured against naive baselines | Planned |

Smaller, by priority:

| Feature | Category | Acceptance | State |
|---|---|---|---|
| AVX-512 host kernels for the IQ formats | CPU | The IQ2_S, IQ3_XXS and IQ4_NL expert products faster on an AVX-512 host than its AVX2 path, measured on the same host | Planned |
| The OpenCL path brought back | Backends | A language and an image model served on an Intel GPU that shares the host's memory through the OpenCL path, that memory counted once by the placement, measured against the host | Partial |
| A routing profile kept per model | Performance | The profile of each model written and read back without configuration, so a restart starts with a warm expert cache | Done |
| Qwen3.8 IQ1_M on one card at the reference | Performance | The reference engine's own bench (128-token replies, 3k and 13k prompts) at or above it on one card it runs on, measured the same day: decode and first token | Planned |
| Context parallelism inside the recorded step | Parallelism | The decode of a context cut over the cards replayed as a graph, within a tenth of an uncut cache's speed | Planned |
| Audio input proven on real checkpoints | Audio | Voxtral and Ultravox transcripts of a 20-clip set at or below the reference's word error rate, with their real-time factor, committed | Planned |
| Ultravox on its own | Audio | One pull or checkpoint directory serves an audio turn with no GGUF supplied beside it | Planned |
| Media input limits | API | Per-request caps on images, pixels and audio seconds, stated in the docs, a request past one answered 400 with the reason | Done |
| Images by URL on the Messages API | API | An image block with a URL source fetched within the host allowlist and answered | Done |
| Multimodal parts on the Responses API | API | Image and audio input items answered as on chat completions, covered by a test | Done |
| Generated audio as mp3 and flac | Audio | The audio generations route answers the format asked, mp3 and flac besides wav, as the speech route does | Planned |
| MIDI at the tempo asked | Music | A MIDI render carries the tempo the request names instead of a fixed one | Planned |
| Audio models in half precision | Audio | The Stable Audio, EzAudio and ACE-Step weights held in F16 rather than widened to F32, Stable Audio resident in 2.5 GB at most | Planned |

## 0.5.0

Major, by priority:

| Feature | Category | Acceptance | State |
|---|---|---|---|
| Throughput serving | Serving | Continuous batching with chunked prefill and multi-step scheduling: a concurrent load of many requests served at the reference server's throughput on the same card | Planned |
| Video input | Multimodal input | A video turn read by a model's vision tower, frames sampled by the model's own rule, answers matching the reference implementation | Planned |
| Speculation for every model | Speculation | A draft model or head found or built for any served model, its windows verified as a tree, faster than plain decode on each family | Planned |

Smaller, by priority:

| Feature | Category | Acceptance | State |
|---|---|---|---|
| Sleep mode | Serving | A model unloaded to host memory on idle and back on the next request, the reload time logged | Planned |
| Per-request metrics | Operations | Queue, prefill, decode and cache figures per request on the metrics endpoint | Planned |
| Beam search | Decoding | n beams with length penalty, the result equal to the reference implementation's | Planned |
| Prompt embeddings | API | A request that hands embeddings instead of tokens, answered as for the tokens they stand for | Planned |
| Streamed speech | Audio | The first audio chunk of a speech request before a tenth of its render time, opus, flac and aac besides wav and mp3 | Planned |
| Audio from a chat turn | Audio | A chat completion asking for text and audio modalities returns the audio in its message | Planned |
| Concurrent media requests | Serving | Two image or speech requests on one card finish sooner together than one after the other | Planned |
| Streamed music | Music | A streamed music request's first audio bytes arrive before the render ends, within a fifth of its time | Planned |
| Several takes at once | Music | n renders with different seeds in less than n times one render's time, each listed in the answer | Planned |
| Audio-conditioned sound | Audio | A reference clip conditions Stable Audio and EzAudio, a repaint keeping the clip outside its span (SNR above 30 dB) | Planned |
| MIDI in | Music | A MIDI file given as the prompt continues for the bars asked, in its tempo and key | Planned |
| Expert parallelism for resident experts | Parallelism | A mixture layer whose experts all fit on the cards split across them where measured links make it pay, faster than the layer on one card | Planned |
| Prefill hand-over for split layers and cut caches | Serving | A layer split over cards and a cache cut by positions handed over from the prefill instance, the decode instance reading no prompt | Planned |
| Image and text embeddings | API | A CLIP or SigLIP checkpoint served alone on the embeddings route for text and images, its vectors matching the reference implementation, measured by the bench | Planned |
| Voice conversion | Audio | An OpenVoice checkpoint turning a recording into a reference speaker's voice through a documented route, measured by the bench | Planned |
| Embeddings measured by the bench | Bench | Every model with the embedding capability measured by assay, inputs per second and latency, against the peers that serve it | Planned |
| ACE-Step at the reference speed | Performance | A song of the length asked rendered by each ACE-Step 1.5 variant within 1.2x of the reference implementation's time on the same card, each stage's time logged | Planned |
| Music editing | Music | Repaint, extend, cover, lego, extract and complete on the audio generations route, each matching the reference implementation's task | Planned |
| ACE-Step adapters | Models | A LoRA trained for ACE-Step applied at load or per request, its style heard against the base render | Planned |

## 0.6.0

Major, by priority:

| Feature | Category | Acceptance | State |
|---|---|---|---|
| A cluster that survives a node | Cluster | A node lost mid-request re-planned onto the others and the request finished, the loss reported | Planned |
| A cluster across networks | Cluster | Nodes on different networks, behind NAT and without a shared LAN, joined over an authenticated and encrypted link, the placement pricing the measured wide-area links | Planned |
| Tensor parallelism across machines | Cluster | One layer split over cards of several machines where the measured links pay, faster than the layer cut alone | Planned |
| More weight formats | Models | GPTQ, AWQ and BitsAndBytes checkpoints loaded through the same placement, outputs matching the reference implementation | Planned |

Smaller, by priority:

| Feature | Category | Acceptance | State |
|---|---|---|---|
| Adapters per request | Models | Several text adapters served over one base model, each request naming its own | Planned |
| Pooling and classification | API | The classify, score and pooling routes over the loaded models | Planned |
| Encoder-decoder models | Models | A sequence-to-sequence family served for text and speech, matching the reference implementation | Planned |
| Vision and video speed | Performance | A one-megapixel image's first token within 1.2x of the reference engines, a video clip's time within 1.2x with its decoder on the card | Planned |
| Image and sound quality measured | Quality | Image families scored by PSNR or LPIPS against their reference, music and sound families by a committed metric set | Planned |
| Longer sound | Audio | A sound or Stable Audio request past today's caps renders its whole length without an audible seam | Planned |
| Another music family | Music | One more open music family rendering through the audio generations route on the placement ladder | Planned |
| Audio speed published | Performance | The real-time factor of each music and sound family on a named card in the benchmarks | Planned |

## 0.7.0

Major, by priority:

| Feature | Category | Acceptance | State |
|---|---|---|---|
| A native Windows server | Packaging | The server built, installed and run as a service on Windows with CUDA, at the Linux build's speed on the same card | Planned |
| Quantisation from the command line | Quantisation | A checkpoint quantised to a chosen format with an importance matrix, its quality scored against the original | Planned |

Smaller, by priority:

| Feature | Category | Acceptance | State |
|---|---|---|---|
| Web search and fetch tools | Tools | Built-in tools a request may enable, their calls and results logged as tool turns | Planned |
| A build with vision input alone | Packaging | A build that reads images in chat without the generation pipelines | Planned |

## 0.8.0

Major, by priority:

| Feature | Category | Acceptance | State |
|---|---|---|---|
| Fine-tuning of adapters | Models | A text adapter trained on the local cards from a dataset, served at once, its loss and quality measured | Planned |
| loken as a tool server | Tools | loken's models and media pipelines offered to agents as MCP tools, each call within the server's limits | Planned |

Smaller, by priority:

| Feature | Category | Acceptance | State |
|---|---|---|---|
| Energy-aware placement | Operations | A placement that trades speed for watts when asked, both measured | Planned |
| Quotas per key | Operations | Token and request budgets per API key, enforced and reported | Planned |

## Waiting for hardware

Features that cannot be measured on the machines at hand; each moves into a version once a machine
to measure it on is available.

| Feature | Acceptance | Needs |
|---|---|---|---|
| A backend for Apple silicon | Backends | One model of each family served through Metal on an M-series Mac by the same placement path, measured against the reference engine on that Mac | Waiting |
