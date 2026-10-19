# Roadmap

What the next releases bring, feature by feature. [`CHANGELOG.md`](CHANGELOG.md) lists what is
done; [`docs/STATUS.md`](docs/STATUS.md) says what is measured. A feature is done when
what its "Done when" column says is true and measured; until then it stays here with its state.

A state is one word: Done, Partial (what is left is a feature of a later version), Started,
Planned, or Waiting (for the hardware to measure it on).

## 0.3.0

| Feature | Category | Done when | State |
|---|---|---|---|
| One model across several machines, cut by layer | Cluster | A model too big for one machine runs across several, faster than on the best machine alone. | Done |
| Tensor parallelism in the generic path | Parallelism | A model runs split across any number of cards, MoE models included. | Done |
| Expert parallelism | Parallelism | A MoE model's experts are spread across cards when that is faster. | Partial |
| Pipeline parallelism across machines | Cluster | A chain of any number of machines serves one model, with speculative decoding still working. | Done |
| Data parallelism | Parallelism | Several copies of a model share the incoming requests. | Done |
| Context parallelism | Parallelism | A long context's attention is split across cards. | Done |
| Prefill and decode on different instances | Serving | One instance reads the prompt and another one writes the answer. | Partial |
| CUDA on the published cudarc crate | Backends | loken builds on the published cudarc crate, with no patched fork, and graphs still work. | Done |
| A Linux binary | Packaging | One Linux archive runs on the CPU anywhere; a second one adds the CUDA libraries. | Done |
| CPU kernels chosen at run time | CPU | One binary picks AVX2 or AVX-512 on the machine it runs on. | Done |
| Qwen3.8 at full speed | Performance | Qwen3.8 IQ1_M and UD-Q4_K_XL run at least as fast as the reference engine, on one and two cards. | Partial |
| A coding agent on large models | Agents | verve finishes a multi-step task on Qwen3.8 and DeepSeek V4.1. | Done |
| A faster test suite | Tests | The tests a change must pass run in a fraction of the old time. | Done |

## 0.4.0

Major, by priority:

| Feature | Category | Done when | State |
|---|---|---|---|
| One binary for NVIDIA and AMD cards | Backends | One binary uses NVIDIA and AMD cards at the same time, and one model can run across both. | Started |
| Everything CUDA runs, on AMD | Backends | Every CUDA test passes on an AMD card, the NVIDIA-only kernels ported. | Started |
| A Vulkan backend | Backends | One model of each family runs through Vulkan on an NVIDIA, an AMD and an Intel card. | Planned |
| Backends measured | Backends | Each backend is measured against the reference engine on the same card. | Started |
| Coding agents beyond verve | Agents | opencode, Claude Code and Cline finish the verve task on Qwen3.8 and DeepSeek V4.1. | Planned |
| Forecasting and tabular models | Models | loken forecasts time series and beats naive forecasts on a fixed set. | Partial |
| Mistral Timestral forecasts | Models | The four Timestral checkpoints answer on the forecast route with their 21 quantiles and covariates, matching Mistral's library on a fixed set. | Planned |

Smaller, by priority:

| Feature | Category | Done when | State |
|---|---|---|---|
| AVX-512 host kernels for the IQ formats | CPU | IQ2_S, IQ3_XXS and IQ4_NL experts run faster with AVX-512 than with AVX2 on the same machine. | Done |
| The OpenCL path brought back | Backends | A language model and an image model run through OpenCL on an Intel GPU, faster than on its CPU. | Partial |
| A routing profile kept per model | Performance | After a restart, a model's expert cache is already warm, with no setting needed. | Done |
| Qwen3.8 IQ1_M on one card at the reference | Performance | Qwen3.8 IQ1_M on one card is at least as fast as the reference engine, with 3k and 13k prompts. | Planned |
| Context parallelism inside the recorded step | Parallelism | A context split across cards decodes with recorded graphs, within 10% of an unsplit cache. | Planned |
| Audio input proven on real checkpoints | Audio | Voxtral and Ultravox transcribe a 20-clip set as well as the reference implementation. | Planned |
| Ultravox on its own | Audio | An Ultravox checkpoint answers audio with no extra GGUF beside it. | Done |
| Media input limits | API | A request with too many images, pixels or audio seconds gets a 400 that names the limit. | Done |
| Images by URL on the Messages API | API | The Messages API reads an image from a URL on an allowed host. | Done |
| Multimodal parts on the Responses API | API | The Responses API reads images and audio as chat completions does. | Done |
| Generated audio as mp3 and flac | Audio | Generated audio comes back as wav, mp3 or flac, as asked. | Done |
| MIDI at the tempo asked | Music | A MIDI render has the tempo the request asks for. | Done |
| Audio models in half precision | Audio | Stable Audio, EzAudio and ACE-Step keep their weights in F16, and Stable Audio fits in 2.5 GB. | Done |

## 0.5.0

Major, by priority:

| Feature | Category | Done when | State |
|---|---|---|---|
| Throughput serving | Serving | Many requests at once run at the reference server's throughput on the same card. | Planned |
| Video input | Multimodal input | A vision model answers about a video as the reference implementation does. | Planned |
| Speculation for every model | Speculation | Every model family decodes faster with a draft than without. | Planned |
| DeepSeek V4.1 for coding agents | Performance | DeepSeek V4.1 reads a 10k-token agent prompt in under a minute and writes 500 tokens in under a minute, on two 16 GB cards and the host | Planned |

Smaller, by priority:

| Feature | Category | Done when | State |
|---|---|---|---|
| Sleep mode | Serving | An idle model moves to host memory and comes back on the next request. | Planned |
| Per-request metrics | Operations | The metrics endpoint shows queue, prefill, decode and cache figures for each request. | Planned |
| Beam search | Decoding | Beam search gives the same result as the reference implementation. | Planned |
| Prompt embeddings | API | A request can send embeddings instead of tokens and gets the same answer. | Planned |
| Streamed speech | Audio | Speech starts playing before a tenth of its render time, in opus, flac, aac, wav or mp3. | Planned |
| Audio from a chat turn | Audio | A chat completion can answer with text and audio. | Planned |
| Concurrent media requests | Serving | Two image or speech requests on one card finish sooner together than one after the other. | Planned |
| Streamed music | Music | Streamed music starts within a fifth of its render time. | Planned |
| Several takes at once | Music | n music takes with different seeds take less than n times one take. | Planned |
| Audio-conditioned sound | Audio | A reference clip guides Stable Audio and EzAudio, and a repaint leaves the rest of the clip intact. | Planned |
| MIDI in | Music | A MIDI file given as the prompt continues in its tempo and key. | Planned |
| Expert parallelism for resident experts | Parallelism | A MoE model whose experts fit on the cards runs faster spread across them than on one card. | Planned |
| Prefill hand-over for split layers and cut caches | Serving | Prompt hand-over also works when the model or its cache is split across cards. | Planned |
| Batched expert reads in a prompt | Performance | Reading a prompt loads each streamed expert once for all the tokens that route to it, and prompt speed rises accordingly | Planned |
| Prompt lookup while experts stream | Speculation | Prompt-lookup drafting runs on models whose experts stream from disk, faster than plain decode on code that repeats its context | Planned |
| DeepSeek V4.1 native draft | Speculation | DeepSeek V4.1 drafts with its own draft head and decodes faster than without it, with the same greedy text | Planned |
| Experts fetched one layer ahead | Performance | The experts of the next layer load while the current layer runs, and decode gets faster with the same text | Planned |
| Image and text embeddings | API | A CLIP or SigLIP model turns texts and images into vectors on the embeddings route, as the original code does. | Planned |
| Voice conversion | Audio | An OpenVoice model makes a recording sound like another speaker. | Planned |
| Embeddings measured by the bench | Bench | The bench measures every embedding model against the other servers that run it. | Planned |
| ACE-Step at the reference speed | Performance | Each ACE-Step 1.5 variant renders a full song at most 1.2x slower than the original code. | Planned |
| Music editing | Music | The music route can repaint, extend, cover, add a track, split tracks and complete a song. | Planned |
| ACE-Step adapters | Models | A style LoRA for ACE-Step changes the sound of a render. | Planned |
| Tabular models | Models | loken predicts rows of a table and beats a simple baseline on a fixed data set. | Planned |

## 0.6.0

Major, by priority:

| Feature | Category | Done when | State |
|---|---|---|---|
| A cluster that survives a node | Cluster | When a machine drops out mid-request, the others finish the request. | Planned |
| A cluster across networks | Cluster | Machines on different networks, even behind NAT, form one cluster over an encrypted link. | Planned |
| Tensor parallelism across machines | Cluster | A layer split across the cards of several machines runs faster than the layer cut alone. | Planned |
| More weight formats | Models | GPTQ, AWQ and BitsAndBytes checkpoints load and answer as the reference implementation does. | Planned |

Smaller, by priority:

| Feature | Category | Done when | State |
|---|---|---|---|
| Adapters per request | Models | Several text adapters share one base model, and each request picks its own. | Planned |
| Pooling and classification | API | The classify, score and pooling routes work on the loaded models. | Planned |
| Encoder-decoder models | Models | An encoder-decoder family serves text and speech as the reference implementation does. | Planned |
| Vision and video speed | Performance | An image or a video clip is read at most 1.2x slower than the reference engines. | Planned |
| Image and sound quality measured | Quality | Every image, music and sound family has a quality score against its reference. | Planned |
| Longer sound | Audio | A long sound renders in one piece, with no audible seam. | Planned |
| Another music family | Music | One more open music family renders through the music route. | Planned |
| Audio speed published | Performance | The benchmarks show each music and sound family's speed on a named card. | Planned |

## 0.7.0

Major, by priority:

| Feature | Category | Done when | State |
|---|---|---|---|
| A native Windows server | Packaging | The server runs as a Windows service with CUDA, as fast as on Linux. | Planned |
| Quantisation from the command line | Quantisation | One command quantises a checkpoint, and its quality is scored against the original. | Planned |

Smaller, by priority:

| Feature | Category | Done when | State |
|---|---|---|---|
| Web search and fetch tools | Tools | A request can turn on web search and fetch, and each call is logged. | Planned |
| A build with vision input alone | Packaging | A smaller build reads images in chat without the generation pipelines. | Planned |

## 0.8.0

Major, by priority:

| Feature | Category | Done when | State |
|---|---|---|---|
| Fine-tuning of adapters | Models | A text adapter trains on the local cards and is served right away, its quality measured. | Planned |
| loken as a tool server | Tools | Agents can call loken's models and media pipelines as MCP tools. | Planned |

Smaller, by priority:

| Feature | Category | Done when | State |
|---|---|---|---|
| Energy-aware placement | Operations | On request, the placement saves watts at some cost in speed, both measured. | Planned |
| Quotas per key | Operations | Each API key has a token and request budget that loken enforces. | Planned |

## 0.9.0

Major, by priority:

| Feature | Category | Done when | State |
|---|---|---|---|
| Routing between families | Models | One model name sends each request to the best member, beats each member alone, and costs less than running them all. | Planned |
| A supermodel from several families | Models | Models of different families choose each word together and beat each of them alone. | Planned |
| A mixture of experts built from several families | Models | Models of different families join one trained mixture of experts that beats each of them alone. | Planned |

Smaller, by priority:

| Feature | Category | Done when | State |
|---|---|---|---|
| A supermodel distilled into one | Models | One model trained to answer like the group scores close to the group on its own. | Planned |

## Waiting for hardware

Features that cannot be measured on the machines at hand; each moves into a version once a machine
to measure it on is available.

| Feature | Done when | Needs |
|---|---|---|---|
| A backend for Apple silicon | Backends | One model of each family runs through Metal on an Apple Silicon Mac. | Waiting |
