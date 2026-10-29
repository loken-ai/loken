# A map of lokend

The layer stack runs up the middle, layer 0 at the bottom. HeteroPlan cuts it into runs, one device for each run. The rest of the daemon is cut into groups, each in one region of the drawing.

## Overview

![The groups of lokend and the way of a request](img/anatomy-overview.svg)

The way of one request:

1. **Received**: the request arrives on an API surface, its template applied and its text turned into tokens.
2. **Prefilled**: the prompt's tokens climb the layers together, a long prompt in chunks, and each token keeps its KV.
3. **Forwarded**: each step climbs the layer stack, run after run.
4. **Remembered**: the step reads and extends the KV cache.
5. **Sampled**: the next token is drawn, batched with the steps of other requests.
6. **Streamed**: the answer streams out as it is decoded.

Steps 3 to 6 repeat for every token.

## Layer stack

![Layer stack](img/anatomy-stack.svg)

| Level | In the plan | Groups that read it |
|---|---|---|
| Output head (`holds_head`) | the output head sits on the card of the last layer | Layer stack and placement, Generation loop |
| Tensor parallel (`tp_cards`) | each dense layer split across a group of cards | Layer stack and placement, Generation loop |
| Runs of layers (`segments`) | a run of layers on one device | Layer stack and placement, Generation loop, Vision and audio input, Experts and streaming, Models and formats, Backends, Media generation, Cluster |
| Expert spill (`host_experts_from`) | routed experts on the host from this layer on | Layer stack and placement, Generation loop, Experts and streaming |
| Pipeline split (`shifted`) | one node's share of a stack that goes on elsewhere | Layer stack and placement, Generation loop, Cluster |
| Placement ladder (`ladder`) | the Demand placed rung after rung until it holds | Layer stack and placement, Generation loop, KV and state, Requests and responses, Experts and streaming, Models and formats, Media generation |

A layer split across two cards:

1. The token climbs layers 0 to 4 on the host.
2. Layers 5 to 16 are split across two CUDA cards. Card 1 gets a copy of the token.
3. Each card computes its share of the attention heads. A card's share follows its read rate.
4. The two partial outputs are summed onto both cards.
5. Each card computes its share of the feed-forward.
6. The partial outputs are summed again. Both cards hold the output of the layer and go on to the next one.
7. Card 0 hands the token to the ROCm card, which runs layers 17 to 23. The output head on that card draws the next token.

Steps 3 to 6 repeat for every split layer.

The sums cross the link between two cards of one machine, twice per layer. Between machines each sum would be a network round trip. The cluster splits the stack by layers instead.

## Requests, input and KV

![Requests, input and KV](img/anatomy-head.svg)

An image in a chat, then the next turn:

1. A chat request with an image arrives. The template builds the prompt and the tokenizer turns its text into ids.
2. The vision encoder turns the image into embeddings. They are spliced into the text at a run of placeholder tokens.
3. The prompt is read through the layers. Its KV is kept in fixed blocks.
4. After the answer, the blocks are written to disk under a hash chained over their tokens.
5. The next turn starts with the same conversation. Its blocks are found again by their hash.
6. Only the new message is read through the layers.

## Generation, experts and models

![Generation, experts and models](img/anatomy-trunk.svg)

Tokens guessed ahead, then checked:

1. The decode step climbs the layers and the output head draws a token.
2. The draft head, one more layer of the model's own kind, guesses the next tokens. The loop tunes how far it guesses.
3. The model checks every guess in one pass through the layers, replayed as a graph.
4. The guesses the model agrees with are kept, with one token of its own. The layers are cut back to the kept tokens.

A long prompt read in chunks:

1. A long prompt arrives. It is cut into chunks sized from the free memory of the tightest card.
2. Each chunk climbs the layers in one pass. Each of its tokens attends to every token before it.
3. The chunk's tokens add their K and V to the cache.
4. A chunk that runs a card out of memory is halved. Reading goes on from that chunk, and the tokens before it keep their cache.
5. Only the last chunk's logits are kept. They give the first token.

Steps 2 and 3 repeat for every chunk.

A token through a mixture of experts:

1. A token reaches a mixture layer. The router picks its experts.
2. The experts that have a slot on the card run there.
3. An expert without a slot runs on the host cores, beside the card. The measured rates can send it to the card instead.
4. Their products are gathered and the token goes on.
5. The counts show that expert is now hotter than the coldest slot. A worker copies it in without holding up a token.

A model loaded through the plan:

1. The catalogue finds the model in one of the model directories.
2. HeteroPlan measures the devices and gives each run of layers to one of them.
3. The loaders read each layer's weights into the device of its run.
4. The parsed checkpoint stays in host memory. Loading the model again skips the disk.

## Backends and media

![Backends and media](img/anatomy-limbs.svg)

An image generated:

1. The images route takes the prompt.
2. The text encoder reads the prompt.
3. The denoiser clears a latent image step by step. HeteroPlan places its layers like a language model's.
4. The decoder turns the latent into pixels, tile by tile.
5. The image leaves as a PNG, or as JPEG or WebP when the request asks.

Step 3 repeats for every denoising step.

## Cluster

![Cluster](img/anatomy-cluster.svg)

## Groups

### Layer stack and placement

Every layer of the model and where it runs. HeteroPlan cuts the stack into runs, one device for each run, and the ladder places them until they fit.

| Code | Reads | Job | Features |
|---|---|---|---|
| `heteroplan`<br>[`src/inference/place/layer_executor.rs`](../src/inference/place/layer_executor.rs) | `segments`, `tp_cards`, `host_experts_from`, `holds_head`, `shifted` | The plan: runs of layers, one device for each run, the head on the last. | Placement measured across cards and the host (own) |
| `generic_transformer`<br>[`src/inference/generic_transformer/loader.rs`](../src/inference/generic_transformer/loader.rs)<br>[`src/inference/generic_transformer/hetero.rs`](../src/inference/generic_transformer/hetero.rs)<br>[`src/inference/generic_transformer/tp_forward.rs`](../src/inference/generic_transformer/tp_forward.rs) | `segments`, `tp_cards`, `host_experts_from`, `holds_head` | Builds and runs the layers of every text model on the devices the plan names. |  |
| `mixers`<br>[`src/inference/generic_transformer/mixer.rs`](../src/inference/generic_transformer/mixer.rs)<br>[`src/inference/generic_transformer/qsa.rs`](../src/inference/generic_transformer/qsa.rs)<br>[`src/inference/generic_transformer/layer_attn.rs`](../src/inference/generic_transformer/layer_attn.rs)<br>[`src/inference/generic_transformer/wide.rs`](../src/inference/generic_transformer/wide.rs)<br>[`src/inference/generic_transformer/ngram.rs`](../src/inference/generic_transformer/ngram.rs) |  | Mixes a layer's tokens: softmax attention, sparse attention or a recurrent state. | Hybrid SSM (Mamba / DeltaNet) (full)<br>Sparse attention through an indexer (own) |
| `model_families`<br>[`src/inference/model`](../src/inference/model) | `segments` | One directory per model family, each loading its blocks through the plan. | Encoder-decoder (seq2seq) (partial)<br>Diffusion image models (full)<br>Audio / video models (full) |
| `placement_llm`<br>[`src/inference/place/llm.rs`](../src/inference/place/llm.rs) | `segments` | The cards a language model is offered and the context its plan can hold. |  |
| `placement_shared`<br>[`src/inference/place/plan.rs`](../src/inference/place/plan.rs) | `segments` | The placement helpers every native model port shares. |  |
| `dry_plan`<br>[`src/inference/place/dry_plan.rs`](../src/inference/place/dry_plan.rs) | `segments`, `tp_cards`, `host_experts_from` | Chooses a placement by measuring candidates instead of estimating one. | Placement chosen by measuring candidate plans (own) |
| `tensor_parallel`<br>[`src/inference/place/tensor_parallel.rs`](../src/inference/place/tensor_parallel.rs) | `segments`, `tp_cards` | Splits every dense layer's heads and feed-forward across a group of cards. | Tensor parallel (TP) (partial) |
| `expert_parallel`<br>[`src/inference/place/expert_parallel.rs`](../src/inference/place/expert_parallel.rs) |  | Splits a mixture layer's routed experts by id range across cards. | Expert parallel (EP) (partial) |
| `ladder`<br>[`src/inference/place/ladder.rs`](../src/inference/place/ladder.rs) | `ladder`, `segments`, `tp_cards`, `host_experts_from` | Places a component again on the next rung when a card runs out of memory. | Recovery ladder when a card runs out of memory (own) |
| `runtime_demand`<br>[`src/inference/place/runtime_demand.rs`](../src/inference/place/runtime_demand.rs)<br>[`src/inference/place/audio_demand.rs`](../src/inference/place/audio_demand.rs)<br>[`src/inference/place/video_plan.rs`](../src/inference/place/video_plan.rs) | `ladder` | What one generation needs, derived from the request and the architecture. |  |
| `fit_report`<br>[`src/inference/place/fit_report.rs`](../src/inference/place/fit_report.rs) |  | Says whether a model fits these cards and what the answer costs. |  |
| `device_probe`<br>[`src/inference/place/device_probe.rs`](../src/inference/place/device_probe.rs)<br>[`src/gpu`](../src/gpu) | `segments` | Lists the cards and what each one offers. | Apple Metal (partial) |
| `vram_manager`<br>[`src/inference/place/vram_manager.rs`](../src/inference/place/vram_manager.rs) | `segments` | The one place placement reads card memory from. |  |
| `room`<br>[`src/inference/place/room.rs`](../src/inference/place/room.rs) | `ladder` | What the cards offer right now, for a load decision. | Host budgets kept inside a container memory cap (own) |
| `layer_perf`<br>[`src/inference/place/layer_perf.rs`](../src/inference/place/layer_perf.rs) |  | Tracks each layer's measured time. |  |
| `cpu_topology`<br>[`src/cpu`](../src/cpu) |  | Detects the cores and the SIMD level the host kernels run at. |  |
| `recover`<br>[`src/inference/place/recover.rs`](../src/inference/place/recover.rs) | `ladder` | Shrinks a size until it fits once a component is placed. |  |

### Generation loop

Admits requests, reads the prompt in chunks, then gives a token to every running sequence at each step: batching, sampling, graph replay and speculation.

| Code | Reads | Job | Features |
|---|---|---|---|
| `llm_engine`<br>[`src/inference/engine/llm_engine/mod.rs`](../src/inference/engine/llm_engine/mod.rs)<br>[`src/inference/engine/llm_engine/loading.rs`](../src/inference/engine/llm_engine/loading.rs)<br>[`src/inference/engine/llm_engine/generate.rs`](../src/inference/engine/llm_engine/generate.rs) | `segments`, `tp_cards`, `host_experts_from`, `holds_head`, `shifted`, `ladder` | Loads a language model through the plan and runs its generations. | Chat and text completion (full) |
| `model_manager`<br>[`src/api/handlers/model_manager.rs`](../src/api/handlers/model_manager.rs) | `ladder` | Keeps the loaded models, their replicas and their keep-alive. | Data parallel (DP) (partial) |
| `decode_step`<br>[`src/inference/engine/decode_step.rs`](../src/inference/engine/decode_step.rs)<br>[`src/inference/engine/model_backend.rs`](../src/inference/engine/model_backend.rs) |  | One decode step, the same for every architecture. |  |
| `graph_capture`<br>[`src/tensor/cuda/captured_graph.rs`](../src/tensor/cuda/captured_graph.rs)<br>[`src/inference/generic_transformer/graph.rs`](../src/inference/generic_transformer/graph.rs) | `ladder` | Captures the decode step once and replays it. | CUDA graph / graph capture (full) |
| `tuner`<br>[`src/inference/offload/tuner.rs`](../src/inference/offload/tuner.rs) |  | Moves one decode setting by what the loop measures. |  |
| `sampler`<br>[`src/inference/sample/token_sampling.rs`](../src/inference/sample/token_sampling.rs)<br>[`src/api/handlers/sampling.rs`](../src/api/handlers/sampling.rs) |  | Chooses the next token from a row of logits. | Seed (full)<br>best_of (full)<br>n (multiple choices per request) (full)<br>Log-probabilities (full) |
| `token_rules`<br>[`src/inference/sample/token_rules.rs`](../src/inference/sample/token_rules.rs) |  | The masks a request puts on the logits before the draw. | logit_bias (full)<br>Grammar / regex / choice constrained decoding (full)<br>Structured output (JSON schema) (full) |
| `scheduler`<br>[`src/inference/serve/scheduler.rs`](../src/inference/serve/scheduler.rs)<br>[`src/inference/serve/continuous_batch.rs`](../src/inference/serve/continuous_batch.rs)<br>[`src/inference/serve/continuous_serve.rs`](../src/inference/serve/continuous_serve.rs)<br>[`src/inference/engine/llm_engine/batching.rs`](../src/inference/engine/llm_engine/batching.rs) |  | Batches the running sequences step by step. | Continuous batching (full) |
| `admission`<br>[`src/api/handlers/admission.rs`](../src/api/handlers/admission.rs)<br>[`src/api/gate.rs`](../src/api/gate.rs) | `ladder` | Gets a generation onto an engine, by priority. |  |
| `cancel_progress`<br>[`src/inference/serve/cancel.rs`](../src/inference/serve/cancel.rs)<br>[`src/inference/serve/progress.rs`](../src/inference/serve/progress.rs) |  | Stops a long request and says what it is doing. |  |
| `prefill`<br>[`src/inference/kernel/fused/prefill_gqa.rs`](../src/inference/kernel/fused/prefill_gqa.rs)<br>[`cuda/flash_prefill_gqa.cu`](../cuda/flash_prefill_gqa.cu)<br>[`cuda/flash_prefill_f16.cu`](../cuda/flash_prefill_f16.cu)<br>[`src/inference/engine/llm_engine/sequential.rs`](../src/inference/engine/llm_engine/sequential.rs) |  | Reads a prompt through the layers in chunks sized for the tightest card. | Chunked prefill (partial) |
| `speculation`<br>[`src/inference/serve/speculative_config.rs`](../src/inference/serve/speculative_config.rs)<br>[`src/inference/serve/prompt_lookup.rs`](../src/inference/serve/prompt_lookup.rs)<br>[`src/inference/serve/eagle.rs`](../src/inference/serve/eagle.rs) |  | Drafts tokens ahead and tunes how far to draft. | Speculative decoding (full) |
| `draft_head`<br>[`src/inference/generic_transformer/draft_head.rs`](../src/inference/generic_transformer/draft_head.rs)<br>[`src/inference/load/mtp_head.rs`](../src/inference/load/mtp_head.rs) |  | One more layer of the model's own kind that drafts the next tokens. | Draft head shipped with the checkpoint (own) |
| `verify`<br>[`src/inference/generic_transformer/verify_window.rs`](../src/inference/generic_transformer/verify_window.rs)<br>[`src/inference/generic_transformer/speculate.rs`](../src/inference/generic_transformer/speculate.rs)<br>[`src/inference/generic_transformer/layer_rows_cpu.rs`](../src/inference/generic_transformer/layer_rows_cpu.rs) |  | Checks a draft in one pass and cuts a recurrent state back to what was kept. | Draft verify replayed as a graph (own) |

### KV and state

What a sequence has seen: on the card, across cards, in host memory and on disk.

| Code | Reads | Job | Features |
|---|---|---|---|
| `kv_cache`<br>[`src/inference/cache/kv_core.rs`](../src/inference/cache/kv_core.rs)<br>[`src/inference/cache/quant_kv.rs`](../src/inference/cache/quant_kv.rs)<br>[`src/inference/cache/q8_kv.rs`](../src/inference/cache/q8_kv.rs)<br>[`src/inference/cache/q4_kv.rs`](../src/inference/cache/q4_kv.rs) | `ladder` | The device KV cache in every precision, its size per layer known to the ladder. | Context shift / past-window reuse (full)<br>Quantised KV cache (full) |
| `paged_kv`<br>[`src/inference/cache/paged_kv.rs`](../src/inference/cache/paged_kv.rs)<br>[`src/inference/cache/paged_attention.rs`](../src/inference/cache/paged_attention.rs)<br>[`src/inference/place/paged.rs`](../src/inference/place/paged.rs) |  | Fixed blocks of KV shared by the sequences of a batch. | Paged attention / paged KV (full) |
| `context_parallel`<br>[`src/inference/generic_transformer/context_parallel.rs`](../src/inference/generic_transformer/context_parallel.rs) |  | Cuts each attention layer's cache over every card, one range of positions each. | Context parallel (CP) (partial) |
| `kv_disk`<br>[`src/inference/cache/kv_disk.rs`](../src/inference/cache/kv_disk.rs)<br>[`src/inference/cache/state_disk.rs`](../src/inference/cache/state_disk.rs)<br>[`src/inference/cache/container.rs`](../src/inference/cache/container.rs) |  | KV and decode state written to disk and found again by prefix. | Automatic prefix caching / KV reuse (full)<br>KV cache offloading (CPU / disk) (full) |
| `state_cache`<br>[`src/inference/generic_transformer/state_cache.rs`](../src/inference/generic_transformer/state_cache.rs)<br>[`src/inference/cache/state_tune.rs`](../src/inference/cache/state_tune.rs) |  | The decode state of a recurrent hybrid or a large mixture, kept by prefix. | Decode-state cache for large mixtures and recurrent hybrids (own) |

### Requests and responses

The API surfaces and coding agents, templates and tokens, tool calls and reasoning, streamed answers, embeddings, scores and forecasts.

| Code | Reads | Job | Features |
|---|---|---|---|
| `openai_surface`<br>[`src/api/handlers/openai`](../src/api/handlers/openai)<br>[`src/api/types`](../src/api/types) |  | The OpenAI routes, with the fields vLLM adds, files and batches included. | OpenAI /v1/chat/completions (full)<br>OpenAI /v1/completions (full)<br>OpenAI /v1/responses (full)<br>OpenAI /v1/models (full)<br>OpenAI files and batches (full) |
| `ollama_surface`<br>[`src/api/handlers/ollama`](../src/api/handlers/ollama) |  | The Ollama routes. | Ollama /api/chat and /api/generate (full)<br>Ollama /api/embed and /api/embeddings (full)<br>Ollama model management (pull/create/copy/delete/show/tags/ps) (full) |
| `anthropic_surface`<br>[`src/api/handlers/anthropic_api.rs`](../src/api/handlers/anthropic_api.rs)<br>[`src/api/anthropic.rs`](../src/api/anthropic.rs) |  | The Anthropic Messages routes. | Anthropic /v1/messages (full)<br>Anthropic token counting (full)<br>Message batches on /v1/messages/batches (own) |
| `agents`<br>[`src/api/launch.rs`](../src/api/launch.rs) |  | What a coding agent needs to reach a lokend, in the form each agent reads. | Coding agents launched on the daemon (own) |
| `prompt_format`<br>[`src/api/handlers/prompt_format.rs`](../src/api/handlers/prompt_format.rs)<br>[`src/api/handlers/chat_prompt.rs`](../src/api/handlers/chat_prompt.rs) |  | Builds the prompt a chat request prefills, once for every dialect. | Reasoning levels / effort (full) |
| `tokenizer`<br>[`src/inference/token`](../src/inference/token) |  | Text to ids and back. |  |
| `prompt_embeds`<br>[`src/inference/prompt_embeds`](../src/inference/prompt_embeds) |  | A prompt sent as embeddings instead of text. | Prompt embeddings (prompt_embeds) (partial) |
| `pooling`<br>[`src/api/handlers/openai/pooling.rs`](../src/api/handlers/openai/pooling.rs)<br>[`src/api/handlers/openai/embed_items.rs`](../src/api/handlers/openai/embed_items.rs)<br>[`src/api/handlers/openai/moderation.rs`](../src/api/handlers/openai/moderation.rs)<br>[`src/api/handlers/ollama/embed.rs`](../src/api/handlers/ollama/embed.rs) |  | Embeddings, scores and moderation from a model's hidden states. | OpenAI /v1/embeddings (full)<br>OpenAI /v1/moderations (full)<br>Pooling APIs (classify / score / pooling) (partial)<br>Cohere embed / rerank APIs (partial)<br>Embeddings (full)<br>Reranking / cross-encoder scoring (full)<br>Classification / moderation (full) |
| `forecast`<br>[`src/inference/engine/forecast_engine.rs`](../src/inference/engine/forecast_engine.rs)<br>[`src/api/handlers/openai/forecast.rs`](../src/api/handlers/openai/forecast.rs)<br>[`src/inference/model/chronos2`](../src/inference/model/chronos2)<br>[`src/inference/model/timestral`](../src/inference/model/timestral) | `ladder` | Time series quantile forecasts, loaded through the ladder on first request. | Time series forecasts on /v1/forecast (own) |
| `reasoning_split`<br>[`src/api/thinking.rs`](../src/api/thinking.rs) |  | Separates a model's reasoning from its answer. | Reasoning returned apart from the answer (full) |
| `tool_calls`<br>[`src/api/tool_calls.rs`](../src/api/tool_calls.rs) |  | Reads tool calls out of the generated text. | Tool calling (function calling) (full)<br>Parallel tool calls (full)<br>tool_choice (auto/required/none/named) (full) |
| `stream_out`<br>[`src/inference/engine/llm_engine/stream.rs`](../src/inference/engine/llm_engine/stream.rs) |  | Streams the answer as it is decoded. | Streaming responses (full) |

### Vision and audio input

Images, video frames and sound turned into the model's input.

| Code | Reads | Job | Features |
|---|---|---|---|
| `hearing`<br>[`src/api/handlers/audio/transcribe.rs`](../src/api/handlers/audio/transcribe.rs)<br>[`src/api/handlers/audio/whisper.rs`](../src/api/handlers/audio/whisper.rs)<br>[`src/inference/codec`](../src/inference/codec)<br>[`src/inference/model/audio_in`](../src/inference/model/audio_in) | `segments` | Audio in a chat turn, transcription and translation. | Audio input in a chat turn (full)<br>Speech-to-text (transcription) (full)<br>Speech translation (full) |
| `vision_in`<br>[`src/inference/generic_transformer/splice.rs`](../src/inference/generic_transformer/splice.rs)<br>[`src/inference/model/mmproj.rs`](../src/inference/model/mmproj.rs)<br>[`src/inference/media/video_io.rs`](../src/inference/media/video_io.rs) |  | Image embeddings spliced into the text stream. | Vision-language models (full)<br>Vision (image input) (full) |

### Experts and streaming

Routed experts and weights moved between the disk, the host and the cards, and the setting that trades exactness for speed.

| Code | Reads | Job | Features |
|---|---|---|---|
| `streamed_offload`<br>[`src/inference/offload/streamed.rs`](../src/inference/offload/streamed.rs)<br>[`src/inference/offload/store.rs`](../src/inference/offload/store.rs)<br>[`src/inference/offload/readahead.rs`](../src/inference/offload/readahead.rs)<br>[`src/inference/offload/cuda.rs`](../src/inference/offload/cuda.rs) | `ladder` | Runs a model too large for the cards and the host together by streaming it. | Mixtures larger than the host streamed from disk (own) |
| `expert_tier`<br>[`src/inference/generic_transformer/expert_tier.rs`](../src/inference/generic_transformer/expert_tier.rs)<br>[`src/inference/offload/admission.rs`](../src/inference/offload/admission.rs)<br>[`src/inference/generic_transformer/expert_profile.rs`](../src/inference/generic_transformer/expert_profile.rs) | `host_experts_from`, `segments` | Keeps routed experts in host memory with an adaptive cache of them on the cards. | Mixture-of-Experts (MoE) (full)<br>Experts spilled across cards and host for mixtures that do not fit (own)<br>Experts cached on the cards by use, with a routing profile per model (own) |
| `moe_dispatch`<br>[`src/inference/offload/moe_dispatch.rs`](../src/inference/offload/moe_dispatch.rs)<br>[`src/inference/place/expert_variants.rs`](../src/inference/place/expert_variants.rs) |  | Splits one block's routed experts between the cards and the cores. |  |
| `quality`<br>[`src/inference/model/deepseek_v41/calibrate.rs`](../src/inference/model/deepseek_v41/calibrate.rs)<br>[`src/inference/engine/llm_engine/quality.rs`](../src/inference/engine/llm_engine/quality.rs)<br>[`src/inference/media/precision.rs`](../src/inference/media/precision.rs) |  | Chooses how much exactness to give up for speed, and scores a setting against the exact run. | Speed and fidelity policy (exact, balanced, fast) (own) |
| `cpu_experts`<br>[`src/inference/offload/cpu_experts.rs`](../src/inference/offload/cpu_experts.rs) | `host_experts_from` | Routed experts computed on the processor cores, one thread-pool region for a block. | Routed experts computed on the processor cores (own) |

### Models and formats

The model stores and pulls, every weight format, conversion on the card, adapters and the parsed checkpoints kept for a quick swap.

| Code | Reads | Job | Features |
|---|---|---|---|
| `weight_cache`<br>[`src/inference/cache/qvb.rs`](../src/inference/cache/qvb.rs) |  | Parsed checkpoints kept in host memory so a model swap is quick. |  |
| `model_store`<br>[`src/inference/load/model_dirs.rs`](../src/inference/load/model_dirs.rs)<br>[`src/inference/load/ollama_manager.rs`](../src/inference/load/ollama_manager.rs)<br>[`src/inference/cache/hf.rs`](../src/inference/cache/hf.rs)<br>[`src/api/handlers/catalogue.rs`](../src/api/handlers/catalogue.rs)<br>[`src/download`](../src/download) |  | The model directories, the Ollama store and the HF cache read as one catalogue, and the pulls. | Model listing with declared capabilities (full)<br>Ollama model store (manifests / blobs) (full)<br>Custom model creation (full)<br>Model catalogue across several disks (own) |
| `loaders`<br>[`src/inference/load`](../src/inference/load)<br>[`src/tensor/blockscaled`](../src/tensor/blockscaled)<br>[`src/tensor/safetensors_io.rs`](../src/tensor/safetensors_io.rs)<br>[`src/tensor/quantized`](../src/tensor/quantized) | `segments` | Reads every weight format into the layers the plan places. | Hugging Face hub / safetensors (full)<br>safetensors / HF transformers (full)<br>GGUF (k-quants) (full)<br>AWQ (full)<br>FP8 (full)<br>MXFP4 / NVFP4 (full) |
| `requantize`<br>[`src/inference/load/requantize.rs`](../src/inference/load/requantize.rs)<br>[`src/inference/load/calibration.rs`](../src/inference/load/calibration.rs)<br>[`src/inference/load/compensate.rs`](../src/inference/load/compensate.rs)<br>[`src/api/handlers/convert.rs`](../src/api/handlers/convert.rs) | `ladder` | Requantises or converts a checkpoint into a new tag, on the card. | Requantise a model into a new tag (full)<br>Conversion and requantisation on the card, weighted by importance (own) |
| `adapters`<br>[`src/inference/load/lora.rs`](../src/inference/load/lora.rs)<br>[`src/tensor/lora.rs`](../src/tensor/lora.rs) |  | Low-rank adapters for the image and text models. | Image adapters (LoRA for diffusion) (full)<br>LoRA adapters (text) (partial) |

### Backends

The kernels of each device kind: CUDA, ROCm, Vulkan, OpenCL and the CPU.

| Code | Reads | Job | Features |
|---|---|---|---|
| `cuda_kernels`<br>[`src/inference/kernel`](../src/inference/kernel)<br>[`src/inference/quantized_cuda`](../src/inference/quantized_cuda)<br>[`src/inference/moe_cuda`](../src/inference/moe_cuda)<br>[`src/tensor/cuda`](../src/tensor/cuda)<br>[`cuda`](../cuda) |  | Fused, quantised and expert kernels, flash attention and the decode graph on NVIDIA cards. | NVIDIA CUDA (full) |
| `rocm_runtime`<br>[`src/tensor/gpu_runtime/hip`](../src/tensor/gpu_runtime/hip)<br>[`cuda/hip`](../cuda/hip)<br>[`cuda/flash_prefill_gqa_amd.cu`](../cuda/flash_prefill_gqa_amd.cu)<br>[`cuda/flash_dit_amd.cu`](../cuda/flash_dit_amd.cu) |  | The HIP runtime and the matrix-core kernels of AMD cards. | AMD ROCm (partial) |
| `gpu_dual`<br>[`src/tensor/gpu_runtime/mod.rs`](../src/tensor/gpu_runtime/mod.rs)<br>[`src/tensor/gpu_runtime/dual`](../src/tensor/gpu_runtime/dual) |  | One binary that drives NVIDIA and AMD cards at once. | Layers on NVIDIA and AMD cards from one binary (own) |
| `vulkan_layers`<br>[`src/inference/generic_transformer/vulkan_layers.rs`](../src/inference/generic_transformer/vulkan_layers.rs)<br>[`src/inference/kernel/vulkan`](../src/inference/kernel/vulkan) | `segments` | The layers a plan gives a Vulkan device, and their shaders. | Vulkan (partial) |
| `opencl_layers`<br>[`src/inference/generic_transformer/opencl_layers.rs`](../src/inference/generic_transformer/opencl_layers.rs)<br>[`src/inference/kernel/opencl.rs`](../src/inference/kernel/opencl.rs)<br>[`src/inference/kernel/opencl_stack.rs`](../src/inference/kernel/opencl_stack.rs)<br>[`src/kernels`](../src/kernels) | `segments` | The layers a plan gives an OpenCL device, and their kernels. | Intel GPU (XPU / OpenCL) (partial) |
| `host_kernels`<br>[`src/inference/kernel/cpu_decode_exec.rs`](../src/inference/kernel/cpu_decode_exec.rs)<br>[`src/tensor/quant_cpu`](../src/tensor/quant_cpu) |  | The host decode, AVX-512 kernels where the processor has them. | CPU (host) (full)<br>AVX-512 host kernels chosen at start (own) |

### Media generation

Images, video, music, sound effects, speech and stems.

| Code | Reads | Job | Features |
|---|---|---|---|
| `voice`<br>[`src/api/handlers/audio/voice.rs`](../src/api/handlers/audio/voice.rs) |  | Speech synthesis over HTTP. | Text-to-speech (full)<br>OpenAI audio API (full) |
| `media_engines`<br>[`src/inference/engine/image_engine`](../src/inference/engine/image_engine)<br>[`src/inference/engine/flux2_engine.rs`](../src/inference/engine/flux2_engine.rs)<br>[`src/inference/engine/qwen_image_engine.rs`](../src/inference/engine/qwen_image_engine.rs)<br>[`src/inference/engine/boogu_engine.rs`](../src/inference/engine/boogu_engine.rs)<br>[`src/inference/engine/tts_engine.rs`](../src/inference/engine/tts_engine.rs)<br>[`src/inference/engine/audio_engine.rs`](../src/inference/engine/audio_engine.rs) | `segments`, `ladder` | Loads each media model through the ladder and runs its pipeline. | Image generation (full)<br>Video generation (full)<br>Music and sound-effect generation (full)<br>MIDI generation (own) |
| `media_routes`<br>[`src/api/handlers/media`](../src/api/handlers/media)<br>[`src/api/handlers/separate.rs`](../src/api/handlers/separate.rs) |  | The image, video, music and stem routes. | OpenAI images API (full)<br>Image editing and variations (full)<br>Source separation (stems) (full) |
| `media_out`<br>[`src/inference/media`](../src/inference/media)<br>[`src/inference/sample/flow_unipc.rs`](../src/inference/sample/flow_unipc.rs) |  | The diffusion samplers and the formats an image, a clip or a sound leaves as. |  |

### Cluster modules

Several machines serving as one: membership, routing, cut points, KV and weights handed between nodes.

| Code | Reads | Job | Features |
|---|---|---|---|
| `cluster`<br>[`src/distributed/cluster.rs`](../src/distributed/cluster.rs)<br>[`src/distributed/membership.rs`](../src/distributed/membership.rs)<br>[`src/distributed/discovery.rs`](../src/distributed/discovery.rs)<br>[`src/distributed/cluster_runtime.rs`](../src/distributed/cluster_runtime.rs) |  | Who is in the cluster and what each node publishes. | Multi-node / cluster (full) |
| `routing`<br>[`src/distributed/routing.rs`](../src/distributed/routing.rs)<br>[`src/distributed/rate_meter.rs`](../src/distributed/rate_meter.rs) |  | Which node answers a request, by prediction. | Request / model routing to peers (full) |
| `cut_plan`<br>[`src/distributed/cut_plan.rs`](../src/distributed/cut_plan.rs)<br>[`src/distributed/auto_tune.rs`](../src/distributed/auto_tune.rs)<br>[`src/distributed/link_matrix.rs`](../src/distributed/link_matrix.rs) | `segments` | Where to cut a layer stack across nodes, priced by each link. | Pipeline parallel (PP) (partial) |
| `pipeline_segment`<br>[`src/inference/generic_transformer/segment.rs`](../src/inference/generic_transformer/segment.rs)<br>[`src/inference/serve/pipeline_segment.rs`](../src/inference/serve/pipeline_segment.rs)<br>[`src/inference/engine/pipeline_head.rs`](../src/inference/engine/pipeline_head.rs)<br>[`src/inference/serve/pipeline_serve.rs`](../src/inference/serve/pipeline_serve.rs) | `shifted` | A node running its share of a stack another node holds the rest of. | Sharding one model across machines (partial) |
| `wire`<br>[`src/distributed/wire.rs`](../src/distributed/wire.rs)<br>[`src/distributed/wire_link.rs`](../src/distributed/wire_link.rs)<br>[`src/distributed/wire_mux.rs`](../src/distributed/wire_mux.rs) |  | The binary frames activations travel in. |  |
| `kv_handover`<br>[`src/distributed/kv_handover.rs`](../src/distributed/kv_handover.rs)<br>[`src/distributed/kv_tiers.rs`](../src/distributed/kv_tiers.rs)<br>[`src/distributed/replay.rs`](../src/distributed/replay.rs)<br>[`src/inference/engine/llm_engine/handover.rs`](../src/inference/engine/llm_engine/handover.rs) |  | A prompt read by one instance and decoded by another, a generation resumed elsewhere. | Disaggregated prefill / decode (partial) |
| `weight_stream`<br>[`src/inference/offload/weight_stream.rs`](../src/inference/offload/weight_stream.rs)<br>[`src/inference/offload/weight_cache.rs`](../src/inference/offload/weight_cache.rs) |  | Weights streamed from the node that holds them. | Prompts and weights handed between nodes (own) |

### Security and operations

The API key, rate and upload limits, user content kept out of logs, metrics and energy per request.

| Code | Reads | Job | Features |
|---|---|---|---|
| `energy`<br>[`src/energy.rs`](../src/energy.rs)<br>[`src/energy_report.rs`](../src/energy_report.rs) |  | Energy and carbon of a decode window, in everyday terms. | Energy accounting per request (full) |
| `auth`<br>[`src/api/handlers/mod.rs`](../src/api/handlers/mod.rs) |  | Checks the API key when one is configured. | Authentication (API key) (full) |
| `privacy`<br>[`src/privacy.rs`](../src/privacy.rs) |  | Keeps user content out of every log. | User content kept out of every log (own) |
| `notice_gate`<br>[`src/notice_gate.rs`](../src/notice_gate.rs) |  | Holds NOTICE.md to the measurement. |  |
| `request_metrics`<br>[`src/request_metrics.rs`](../src/request_metrics.rs)<br>[`src/api/handlers/metrics.rs`](../src/api/handlers/metrics.rs)<br>[`src/stats.rs`](../src/stats.rs)<br>[`src/api/handlers/system.rs`](../src/api/handlers/system.rs) |  | What each request spent, OpenMetrics over it, the health and device routes. | Per-request metrics (partial)<br>Per-stage and per-kernel timing on /api/stage_perf (own) |
| `limits`<br>[`src/api/rate_limiter.rs`](../src/api/rate_limiter.rs)<br>[`src/api/media_limits.rs`](../src/api/media_limits.rs) |  | A token bucket per client, caps on the media a request hands the model. | Rate limiting (full)<br>Upload, body and image-URL limits (own) |
