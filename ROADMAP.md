# Roadmap

The next work, by version. [`CHANGELOG.md`](CHANGELOG.md) lists what shipped. [`docs/STATUS.md`](docs/STATUS.md) says what is measured.

A row stays until its `Done when` is true and measured.

| State | Meaning |
|---|---|
| Started | Work is under way |
| Partial | Part shipped, the rest is open |
| Planned | Not started |
| Waiting | Needs hardware to measure on |

## 0.3.0

Run 1 model across several cards and machines.

| Feature | Category | Done when | State |
|---|---|---|---|
| Expert parallelism | Parallelism | A MoE model's experts are spread across cards when that is faster. | Partial |
| Prefill and decode on different instances | Serving | One instance reads the prompt and another one writes the answer. | Partial |
| Qwen3.8 at full speed | Performance | Qwen3.8 IQ1_M and UD-Q4_K_XL run at least as fast as the reference engine, on 1 and 2 cards. | Partial |
| A coding agent on large models | Agents | lokode finishes a multi-step task on Qwen3.8 and DeepSeek V4.1. | Done |

## 0.4.0

Every backend and every format loken reads.

Major, by priority:

| Feature | Category | Done when | State |
|---|---|---|---|
| Coding agents beyond lokode | Agents | opencode, Claude Code and Cline finish the lokode task on Qwen3.8 and DeepSeek V4.1. | Done |

## 0.4.x

Patch releases that finish or speed up what 0.4.0 ships:

| Feature | Category | Done when | State |
|---|---|---|---|
| Vulkan at the speed of CUDA and ROCm | Performance | Vulkan decodes and reads a prompt at the speed of CUDA on the NVIDIA card and of ROCm on the AMD card, for 1 model of each family. | Planned |
| Vulkan for every layer kind | Backends | A model whose layers the Vulkan backend refuses today (recurrent mixers, shared experts, other expert routing) runs through Vulkan and answers as the host does. | Planned |
| OpenCL faster than the CPU | Performance | A language model and an image model run through OpenCL on an Intel GPU faster than on its CPU. | Planned |
| Backends at the reference speed | Performance | Each backend is at least as fast as the reference engine on the same card. | Planned |

## 0.5.0

Decode fast when the experts do not fit on the card, and serve many requests at once.

Major, by priority:

| Feature | Category | Done when | State |
|---|---|---|---|
| Throughput serving | Serving | Many requests at once run at the reference server's throughput on the same card. | Started |
| Video input | Multimodal input | A vision model answers about a video as the reference implementation does. | Started |
| Speculation for every model | Speculation | Every model family decodes faster with a draft than without. | Started |
| DeepSeek V4.1 for coding agents | Performance | DeepSeek V4.1 reads a 10k-token agent prompt in under a minute and writes 500 tokens in under a minute, on 2 16 GB cards and the host | Planned |
| Experts chosen across quantised files | Performance | One model served from 2 GGUF files of one checkpoint picks each expert's file by memory budget and routing heat: DeepSeek V4.1 from its IQ2 and IQ3 files decodes within 10% of IQ2 alone and stays closer to the official model than IQ2 alone. | Partial |
| Qwen3.8 IQ1_M on 1 card at the reference | Performance | Qwen3.8 IQ1_M on 1 card is at least as fast as the reference engine, with 3k and 13k prompts. | Planned |
| Context parallelism inside the recorded step | Parallelism | A context split across cards decodes with recorded graphs, within 10% of an unsplit cache. | Planned |

Smaller, by priority:

| Feature | Category | Done when | State |
|---|---|---|---|
| ACE-Step at the reference speed | Performance | Each ACE-Step 1.5 variant renders a full song at most 1.2x slower than the original code. | Planned |
| Embeddings measured by the bench | Bench | The bench measures every embedding model against the other servers that run it. | Planned |
| Sleep mode | Serving | An idle model moves to host memory and comes back on the next request. | Planned |
| Per-request figures in the answer | Operations | A chat answer on the OpenAI and Anthropic routes carries its own queue, prefill and decode times, its energy with how it was shared, its place in the cache and its accepted drafts, and a long prompt reports its reading as it goes. | Partial |
| Beam search | Decoding | Beam search gives the same result as the reference implementation. | Partial |
| Concurrent media requests | Serving | 2 image or speech requests on 1 card finish sooner together than one after the other. | Started |
| Expert parallelism for resident experts | Parallelism | A MoE model whose experts fit on the cards runs faster spread across them than on 1 card. | Planned |
| Prefill hand-over for split layers and cut caches | Serving | Prompt hand-over also works when the model or its cache is split across cards. | Planned |
| Reuse or recompute | Serving | A prompt hand-over takes the KV when moving it is faster than reading the prompt again, judged on the link speed and the prompt speed loken measured, and it works in both directions. | Planned |
| Batched expert reads in a prompt | Performance | Reading a prompt loads each streamed expert once for all the tokens that route to it, and prompt speed rises accordingly | Planned |
| Prompt lookup while experts stream | Speculation | Prompt-lookup drafting runs on models whose experts stream from disk, faster than plain decode on code that repeats its context | Planned |
| DeepSeek V4.1 native draft | Speculation | DeepSeek V4.1 drafts with its own draft head and decodes faster than without it, with the same greedy text | Planned |
| Experts fetched one layer ahead | Performance | The experts of the next layer load while the current layer runs, and decode gets faster with the same text | Planned |
| Checkpoints served as stored | Models | A float checkpoint is served in its stored format whenever it fits, on the host and on every card family, it is requantised only to fit, and the log, `/api/ps` and the bench records say which format runs. | Partial |
| One setting from fidelity to speed | Serving | One configuration setting, with a few named levels, chooses every trade between fidelity to the published model and speed: the weight format, the cache format, the drafts that are not bit for bit, the shifted context. | Started |
| Speculation sized by expert cost | Speculation | The draft length accounts for the experts a draft adds that are not in memory, and a streamed model decodes faster with it than with the length chosen by acceptance alone. | Planned |
| Routing toward resident experts | Performance | At the fast level the router favours experts already in memory, and the answers stay inside the bounds of the quality harness. | Planned |
| Expert skipping | Performance | At the fast level a token skips the routed experts that add little, which removes their reads, and the answers stay inside the bounds of the quality harness. | Planned |
| Lower precision on a miss | Performance | An expert that is not in memory and weighs little in the routing is read from a smaller copy, and the answers stay inside the bounds of the quality harness. | Planned |
| Expert deferral | Performance | At the fast level the host's expert work overlaps the next layer's attention, decode is faster than without, and the answers stay inside the bounds of the quality harness. | Planned |
| Server-side thinking budget | API | A request's reasoning budget ends the reasoning with the closing marker of the chat template once it is spent. | Planned |
| Batch invariance measured | Tests | A test compares a sequence's logits alone and inside a batch, and reports the largest difference. | Planned |
| Load profiles in the bench | Bench | The bench sends requests at a constant or a random rate, and finds the rate at which the queue of the engine starts to grow without bound. | Planned |
| Replay of agent sessions | Bench | The bench replays a recorded session with its shared prefixes and its timing, and the record names the trace by its digest. | Planned |
| Quality next to speed | Bench | A text record also holds how far each answer departs from a reference run, on the engines that expose the data. | Planned |
| Records that carry their recipe | Bench | A record holds the command, the binary hash, the model file hash and the configuration digest, and a signature so that a copy cannot change under the same name. | Planned |

Fixes, by priority:

| Feature | Category | Done when | State |
|---|---|---|---|
| Stream errors reach the client as errors | API | An engine failure during a stream on the OpenAI chat and Responses routes ends the stream with an error and no finish chunk, as the Ollama and Messages routes do. | Started |
| A missing token count stays missing | Bench | An iteration with no token count is left out of the totals and says so, instead of counting 0. | Done |
| Same first-token time on every engine | Bench | The bench starts the first-token time at the first text of the stream on every engine. | Done |
| Failures visible in concurrent cells | Bench | A failed request in a cell with concurrent requests is recorded in the errors of the cell. | Done |
| Streams that cannot hang | Bench | The measuring client times out a silent stream and refuses an https address instead of connecting in plain text. | Done |
| A cut stream is reported as cut | API | A Responses stream whose upstream ends without its final marker ends as incomplete, not as completed. | Started |
| Warm-up failures visible | Bench | A failed warm-up request is reported for every modality. | Done |
| Measured settings recorded | Bench | The image size, seed, duration, voice and keep-alive of a cell appear in the record, and so does the request timeout. | Partial |
| A memory fallback that does not stay | Performance | A matrix product that runs out of card memory falls back for that call only, and the fallback copy is not kept. | Done |
| Sampling without per-token allocation | Performance | A sampled token allocates no buffer on the card, and uploads no penalty list when the penalty is off. | Partial |
| Honest back-pressure | API | A refused request carries a Retry-After computed from the queue and the measured rate, and a request waiting on a model load reports the load as it goes. | Partial |

## 0.6.0

A cluster that survives failures and can be trusted, more weight formats, and a plan before a load.

Major, by priority:

| Feature | Category | Done when | State |
|---|---|---|---|
| A cluster that survives a node | Cluster | When a machine drops out mid-request, the others finish the request. | Planned |
| Routing across loken nodes | Cluster | One model name sends each request to the loken node that serves it best by load, speed and energy, to the node that already holds its conversation's KV, and to another node when one fails. | Planned |
| A cluster across networks | Cluster | Machines on different networks, even behind NAT, form one cluster over an encrypted link. | Planned |
| Tensor parallelism across machines | Cluster | A layer split across the cards of several machines runs faster than the layer cut alone. | Planned |
| More weight formats | Models | GPTQ, AWQ and BitsAndBytes checkpoints load and answer as the reference implementation does. | Planned |
| Mistral 4 Large | Models | The published Mistral 4 Large checkpoint loads and answers as the reference implementation does. | Planned |

Smaller, by priority:

| Feature | Category | Done when | State |
|---|---|---|---|
| Decision route | Models | A decision route takes a block of state and typed questions (choice, score, yes or no) and returns one calibrated probability per option in one pass, and several questions on one state cost one reading. Strands Decider 2B matches its reference on a fixed set, on the host and on a card. | Planned |
| Adapters per request | Models | Several text adapters share one base model, and each request picks its own. | Planned |
| Pooling and classification | API | The classify, score and pooling routes work on the loaded models. | Planned |
| Encoder-decoder models | Models | An encoder-decoder family serves text and speech as the reference implementation does. | Planned |
| Vision and video speed | Performance | An image or a video clip is read at most 1.2x slower than the reference engines. | Planned |
| Image and sound quality measured | Quality | Every image, music and sound family has a quality score against its reference. | Planned |
| Longer sound | Audio | A long sound renders in one piece, with no audible seam. | Planned |
| Another music family | Music | One more open music family renders through the music route. | Planned |
| Audio speed published | Performance | The benchmarks show each music and sound family's speed on a named card. | Planned |
| Placement preview | API | `loken fit` and a plan route tell, for a model file, the layers per device, the host share and the context that fits, from the code a real load runs. | Planned |
| Authenticated cluster | Cluster | Nodes sign their announcements and their peer calls with a shared secret, and a node without it is refused. | Planned |
| Model copy between nodes | Cluster | A node fetches a model file from a peer that holds it, checks its digest and resumes after a cut. | Planned |
| Host tier for the KV cache | Serving | The KV of a long context lives in host memory, a working set stays on the card, and the pages a query selects are fetched, so a long context is no longer bounded by card memory. | Planned |
| Suffix drafting across requests | Speculation | A draft index built from earlier answers of the same model, kept in memory and cleared on unload, drafts faster than prompt lookup alone on repeated agent work. | Planned |
| Tree drafting | Speculation | The verification pass checks a tree of candidates, and the width of the tree follows the cost of the experts it adds. | Planned |
| Chunked pipeline prefill | Cluster | The chunks of a prompt cross a cut pipeline one behind the other, so every stage works at once, and a long prompt reaches its first token sooner than with each chunk waiting for the whole chain. | Planned |
| Several sequences through a cut pipeline | Cluster | A pipeline across machines serves several sequences at once, each with its own KV on every stage, and keeps speculative verification working. | Planned |
| Disk-streamed layers in the cut plan | Cluster | A node whose share of layers does not fit in its memory streams them from disk while the other stages work, and the cut plan counts that stream as capacity. | Planned |
| Cut boundary sized by the link | Cluster | The activations that cross a cut are sent in the narrowest form the link needs, chosen from the link speed loken measured, and answers stay inside the bounds of the quality harness. | Planned |
| Load-driven re-cut | Cluster | A cut pipeline moves layers between nodes when the load or the speed of a node changes, without dropping the requests in flight. | Planned |
| Peer memory as an expert tier | Cluster | The memory of a peer holds experts for a node whose own memory is short, fetched over the link, and a model decodes faster than with those experts read from disk. | Planned |
| Encoder on its own card or node | Serving | The vision or audio encoder of a model runs on a card or a node other than the one that decodes, and a request that carries media decodes sooner than with both on 1 card. | Planned |
| Weight types NVFP4, IQ1 and ternary | Models | A GGUF file that holds an NVFP4, an IQ1 or a ternary tensor loads and answers as the reference implementation does. | Planned |
| Block cache for diffusion transformers | Performance | An image or video model reuses the output of its blocks from one step to the next, renders faster than with whole steps skipped, and the quality score does not fall. | Planned |
| Multi-vector embeddings | API | The embeddings and rerank routes return one vector for each token and score a query against a document by late interaction, for text and for page images. | Planned |
| Embeddings as integers and bits | API | The embeddings routes return a vector as int8 or as binary on request, and a client ranks with them as well as with the float vector. | Planned |
| Word timestamps and long recordings | Audio | The transcription route returns the time of each word, and a recording of any length is read in windows cut at silences. | Planned |
| Several vision embeddings cached | Serving | The embeddings of several images are kept between requests, and a conversation that comes back to an image does not read it again. | Planned |

Fixes, by priority:

| Feature | Category | Done when | State |
|---|---|---|---|
| Cluster polling in parallel | Operations | lokpit asks all nodes at the same time, so a slow node delays only its own reading, and a failed endpoint marks the node as degraded. | Planned |
| Each tensor parsed once | Performance | Loading a safetensors checkpoint parses the header of each file once. | Planned |
| The disk cache lock held briefly | Performance | Saving a prompt to the disk cache never delays a lookup. | Planned |
| lokpit export serves | Operations | `lokpit export` serves on the address it is given, or the option is removed. | Planned |

## 0.7.0

Reach and hardening: Windows, packaging, typed answers, streamed audio, and exact decoding.

Major, by priority:

| Feature | Category | Done when | State |
|---|---|---|---|
| A native Windows server | Packaging | The server runs as a Windows service with CUDA, as fast as on Linux. | Planned |
| Typed answers on the language route | API | A request with a schema always gets an answer that fits it, or an explicit error, never a malformed one. | Planned |
| Quantisation from the command line | Quantisation | One command quantises a checkpoint, and its quality is scored against the original. | Planned |

Smaller, by priority:

| Feature | Category | Done when | State |
|---|---|---|---|
| Open decision models | Models | Clef 27B, Clef-flash 9B, Clef-omni 30B, Perplexity's Decider, Liquid's d1-3B and d1-omni-600M each match their reference on a fixed set on the decision route, and read the images, audio or video they were trained on. Closed ones (Jev, Microsoft, OpenAI, Liquid d1-ultra) are not served. | Planned |
| System One extraction models | Models | Once its weights are published, Clairic returns numbers with an 80% range, typed values and extracts with their positions in one pass, matching the reference on a fixed set. | Planned |
| Streamed speech | Audio | Speech starts playing before a tenth of its render time, in opus, flac, aac, wav or mp3. | Planned |
| Audio from a chat turn | Audio | A chat completion can answer with text and audio. | Planned |
| Streamed music | Music | Streamed music starts within a fifth of its render time. | Planned |
| Several takes at once | Music | n music takes with different seeds take less than n times one take. | Planned |
| Audio-conditioned sound | Audio | A reference clip guides Stable Audio and EzAudio, and a repaint leaves the rest of the clip intact. | Planned |
| MIDI in | Music | A MIDI file given as the prompt continues in its tempo and key. | Planned |
| Voice conversion | Audio | An OpenVoice model makes a recording sound like another speaker. | Planned |
| Web search and fetch tools | Tools | A request can turn on web search and fetch, and each call is logged. | Planned |
| A build with vision input alone | Packaging | A smaller build reads images in chat without the generation pipelines. | Planned |
| A service profile | Packaging | A systemd unit with a dedicated user and access to the devices the node uses, and a container recipe, run loken on NVIDIA and AMD nodes. | Planned |
| Signed releases | Packaging | Every release archive is signed and published with its checksums and a software bill of materials. | Planned |
| Realtime voice session | Audio | A WebSocket session streams audio in and out with turn detection. | Planned |
| Job callbacks | API | A render or video job calls a URL on an allowed host, or streams its events, when it ends. | Planned |
| More samplers | API | typical_p, mirostat, XTC and DRY are accepted on every route that takes sampling options, and match the reference results. | Planned |
| Deterministic mode | Decoding | At the reference level a sequence's tokens do not depend on the batch around it. | Planned |
| FP4 cache measured | Quantisation | A 4-bit float KV cache is scored against the 4-bit cache already here, and kept only if it wins. | Planned |
| Half-precision sums for split layers | Parallelism | Split layers sum their partial results in half precision, and a model split across cards decodes faster than with the sums in F32, inside the bounds of the quality harness. | Planned |
| Tensor parallelism across NVIDIA and AMD cards | Parallelism | One layer split across an NVIDIA card and an AMD card runs, and faster than the layer cut alone. | Planned |
| Wake a node on demand | Cluster | A node that sleeps is woken over the network when a model it holds is asked for, and sleeps again when it is idle. | Planned |
| Per-tensor quantisation within a size | Quantisation | The converter picks the format of each tensor by its importance, and the file is smaller than with one format for all at the same quality score. | Planned |
| Speaker labels | Audio | The transcription route says which speaker said each part. | Planned |
| Transducer speech recognition | Audio | A transducer speech model transcribes through the existing ONNX executor as its reference does. | Planned |
| 4-bit weights and activations | Quantisation | A diffusion model runs with 4-bit weights and activations plus a low-rank branch in 16 bits, inside the quality score of the 8-bit model. | Planned |
| Lossless compressed BF16 weights | Quantisation | A BF16 checkpoint is stored entropy-coded, decoded on the card as it is read, and gives the same bits in less memory. | Planned |
| Video diffusion with a wider VAE | Models | A video model whose VAE compresses 16 by 16 in space and 4 in time loads and renders as its reference does. | Planned |

Fixes, by priority:

| Feature | Category | Done when | State |
|---|---|---|---|
| lokraft starts or says why | Client | lokraft reports an error when its HTTP client cannot be built, and carries no unused items. | Planned |

## 0.8.0

Agents and operations: tools, keys, traces, cost and what an operator can see.

Major, by priority:

| Feature | Category | Done when | State |
|---|---|---|---|
| Fine-tuning of adapters | Models | A text adapter trains on the local cards and is served right away, its quality measured. | Planned |
| loken as a tool server | Tools | Agents can call loken's models and media pipelines as MCP tools. | Planned |

Smaller, by priority:

| Feature | Category | Done when | State |
|---|---|---|---|
| Music editing | Music | The music route can repaint, extend, cover, add a track, split tracks and complete a song. | Planned |
| ACE-Step adapters | Models | A style LoRA for ACE-Step changes the sound of a render. | Planned |
| Tabular models | Models | loken predicts rows of a table and beats a simple baseline on a fixed data set. | Planned |
| Energy-aware placement | Operations | On request, the placement saves watts at some cost in speed, both measured. | Planned |
| Quotas per key | Operations | Each API key has a token and request budget that loken enforces, names the models, tools and media it may use, and every request leaves a record of the key, model and tools. | Planned |
| Scopes per key | Operations | A key is limited to inference or to administration. | Planned |
| Traces and metrics export | Operations | loken exports traces and the standard generation metrics, follows an incoming trace across cluster hops, and carries no prompt text. | Planned |
| Cost and carbon ledger | Operations | loken keeps counters per key and model of requests, tokens and energy, with the carbon the configured carbon intensity gives. | Planned |
| Warm prefixes | Serving | A client names a prefix that loken loads into the cache tiers and keeps against eviction, within the memory budget. | Planned |
| Tool loop on the server | Tools | loken runs a bounded loop of tool calls itself and returns the final answer. | Planned |
| Cluster history and alerts | Operations | lokpit records snapshots, replays them, and calls a command when a finding appears or clears. | Planned |
| Side-by-side answers | Client | lokraft sends one prompt to 2 models or 2 fidelity levels and shows both answers with their speed and energy. | Planned |
| KV reuse outside the prefix | Serving | A chunk of text seen before is reused at another position with a share of its tokens recomputed, and the answers stay close to a cold read. | Planned |
| Attention and feed-forward on different nodes | Cluster | The attention of a layer and its feed-forward run on different machines, and a model too large for the best machine decodes faster than with its layers cut. | Planned |
| Multi-path placement | Cluster | A model is placed over a cluster by the speed of its links and the memory of its nodes, with a layer served by more than one path, and decodes faster than with 1 path. | Planned |
| Visual token pruning | Performance | A request can drop the image tokens that matter least to its question, and a vision model answers faster with the answer inside the bounds of the quality harness. | Planned |
| Sampling recipe carried by an adapter | Models | A few-step adapter brings the sampler and the step count it needs, and a render uses them without a setting. | Planned |

## 0.9.0

Several model families answering together.

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

Features that no available machine can measure. Each moves into a version when a machine to measure it on is available.

| Feature | Category | Done when | State |
|---|---|---|---|
| A backend for Apple silicon | Backends | 1 model of each family runs through Metal on an Apple Silicon Mac. | Waiting |
| A native backend for Intel GPUs | Backends | 1 model of each family runs through Intel's Level Zero, with the matrix engines of an Arc card, faster than through Vulkan and OpenCL on the same card. | Waiting |
| Disk reads straight to the card | Performance | Experts stream from the SSD to a card without passing through host memory, faster and with fewer joules per byte than the current read. | Waiting |
