# Changelog

All notable changes to loken. The format follows [Keep a Changelog](https://keepachangelog.com/1.1.0/).

## [Unreleased]

### Changed

- Experts: give a mixture's first request the whole cache of its cards.
- Experts: lock a mixture's host pages before its load ends, not during a first prompt.
- Experts: compute a mixture's missing Q5_1 experts on the host.
- Experts: compute a mixture's missing Q3_K experts on the host.
- Host: decode an AWQ model faster and read Q5_0 prompts about 4 times faster.
- Loading: answer a request that widens the context sooner after its reload.

### Fixed

- Serving: leave a load in progress its room when a batch worker starts.
- Placement: keep a model's out-of-memory escalation to its own placements.
- Serving: give the card back the cuBLAS workspace of a model that leaves.
- Serving: keep a model's graphs whole while another model shares its card.
- Serving: give a batch worker the room serial models keep on its card.
- API: keep tool calls when a message quotes a thinking tag.
- API: say that a non-streamed answer ended on its token budget when it did.
- Generation: rotate positions as the model was trained when a file omits its rope keys.
- Generation: apply the linear rope scaling a file declares.
- Tokenizer: split a file that names no pre-tokenizer as its architecture was trained.
- Generation: give the same greedy tokens streamed or not when the host verifies drafts.
- Generation: keep a long answer going when a model's attention window slides.
- Generation: answer a long prompt when a card runs out of room for its cache.
- Loading: load a quantised checkpoint on the host without running out of memory.
- Generation: keep a quantised key cache accurate on models with outlier channels.
- Generation: give the same greedy tokens on a card whether drafts ran or not.
- Generation: keep an AWQ model's key cache at the configured fidelity on a card.
- Generation: keep a quantised key cache accurate on models without a key norm.
- ROCm: keep a mixture serving when a card refuses a scratch allocation.
- Loading: serve a hybrid model whose experts do not all fit on its card.

## [0.5.1] - 2026-11-17

### Changed

- Sampling: apply no repeat penalty when neither the request nor the model sets one.
- Experts: decode a mixture's cached experts faster on a card.

### Fixed

- Experts: read a prompt without copying the experts its rows do not use.
- Serving: answer lone requests serially until a burst has measured the worker.
- Experts: keep a token from hanging while another request loads kernels on its card.
- Serving: weigh a lone request's prompt on both decode paths, whatever its length.
- Generation: end on every end token a checkpoint declares, whatever its format.
- Speculation: keep a drafter from taking the card room its target needs.
- API: leave a stored draft out of the model lists and serve its model when asked for it.
- API: show a listed safetensors checkpoint with its window, format and capabilities.
- Serving: log and report every runtime and media feature the build has.
- OpenCL: offer a card when the backends leave its native runtime out.
- Vulkan: offer a GPU when the backends leave its native runtime out.
- OpenCL: build the layers in the OpenCL C version the device declares.
- Release: carry the Vulkan backend in the Linux archive.
- Experts: give the same greedy tokens whatever a card's cache holds, in every format.
- CUDA: decode Q4_0, Q4_1, Q5_0 and Q5_1 weights as precisely as the host does.
- Generation: never advance a recurrent state twice when a card runs out of memory.
- Experts: leave a decode its room when a card's cache takes its slots back.
- API: load a model on an empty request whose keep-alive is negative.

## [0.5.0] - 2026-11-15

### Added

- Serving: batch mixtures, attention sinks and stretched rotations on a card.
- Serving: keep an idle model in host memory instead of unloading it, and load it from there on the next request.
- Experts: favour experts in memory at the fast level.
- Speculation: try prompt lookup on models whose experts stream, kept while its rounds win.
- Speculation: size the draft by the experts it adds that no card holds.
- Serving: hand a prompt's cache over for a model split or cut across cards.
- Speculation: cut a recurrent drafter back to the tokens its target kept.
- Speculation: draft DeepSeek V4.1 tokens with the model's own draft head when its files carry one.
- Metrics: count the expert bytes missing from memory by asker, and the experts a token shares with the last.
- Metrics: publish each request's queue, prefill, decode and cache figures.
- Serving: batch the image and speech requests queued behind a running one.
- API: take a completion's prompt as embeddings, in loken's checksummed format or as torch.save tensors.
- API: take vLLM's request fields and serve its extra routes.
- API: tell a client refused by the rate limit or for lack of a slot when to retry.
- API: close a reasoning at its token budget on chat completions and Messages.
- Models: embed texts and images in 1 space with CLIP and SigLIP checkpoints, on the embeddings routes.
- Config: list several model directories, on several disks, as 1 catalogue read in the order listed.
- Models: serve a float checkpoint in its stored format when it fits.
- API: answer use_beam_search with the n best beams, scored as transformers does.
- Serving: log the pace of batched steps once a second.
- Vision: read the Qwen3.5 vision tower from a separate projector file.
- Vision: answer about a video clip sent as a video_url part.
- API: report each chat answer's times, energy share, cache place and drafts, summed over its generations.
- API: stream the reading of a long prompt as comment lines.
- Models: convert DeepSeek V4.1 and its draft into files ollama and llama.cpp load.
- Docs: map the code of lokend for contributors.
- Config: choose fidelity to the published model or speed with 1 setting.
- API: show the progress of a model load in the process list.
- Cluster: read a prompt here when that is faster than taking its cache from a peer.
- API: answer speech in FLAC, Opus and AAC.
- Speech: speak with Qwen3-TTS in a voice described in words or named by the checkpoint.
- Speech: create a voice from a clip or a description and reuse it.
- Vulkan: run layers with residual scales, no pre-norm, softcap, sinks, biases, Q5_0 or MXFP4 weights, and hybrid models.
- OpenCL: run the layers of a model whose RoPE has frequency factors.

### Changed

- MoE: copy the next layer's likely experts while the current layer runs, when the hit rate pays.
- MoE: read a long prompt layer by layer when its experts sit in host memory, each read once.
- Serving: read the prompts that hit the prefix cache together in 1 pass.
- Serving: keep an AWQ model's activations in f16 between its projections.
- Serving: batch a burst from its first request and draft for a lone request.
- OpenCL: run the Z-Image projections faster than the host on an Intel GPU.
- OpenCL: read a prompt and decode faster on a GPU.
- Vulkan: decode at pace on an idle host and read a prompt in larger steps, experts included.
- Experts: split a mixture that 1 card holds across cards when the measured links make that faster.
- Server: replace --cpu by --backends, which names the backends to compute on.
- Speech: hold the Parler-TTS decoder in F16 on a card unless fidelity is reference.
- Tokenizer: follow the file's add_bos_token on a bare prompt, as the reference does.
- Experts: keep more of them in host memory, sized while serving, with read-ahead taken only when it runs faster.
- API: leave the stop sequence out of the text returned by every route unless asked to keep it.
- Serving: decode and prefill more requests in 1 batch, sharing the cache and fusing the attention.
- Inference: run the host attention of a long prompt over tiles of keys and rows.
- Inference: read a q4_K or q5_K weight once for all the rows of a small batch on a card.
- Speculative decoding: record the draft verify and the drafter step as graphs, and cost a cycle by its replays.
- Speech: speak faster with pocket-tts by keeping its cache across frames.
- DeepSeek V4.1: serve it through the generic route, on the host and the cards.
- Sampling: allocate nothing on the card for a sampled token.
- Sampling: build and upload no repeat penalty list when the penalty is 1.0.

### Fixed

- Speech: speak a long text with pocket-tts in its reference's chunks, the voice held across them.
- Conversations: keep the state of a short answer in a state cache in host memory.
- OpenCL: give each block of 32 activations of a product its own scale.
- OpenCL: read the subnormal half scales of a model on every device.
- OpenCL: offer no card that CUDA or ROCm already drives.
- OpenCL: load and render Z-Image on an Intel GPU without host copies or a stalled queue.
- Image and video: hold SDXL, Z-Image and Wan projections in their narrowest exact form.
- Z-Image: build the caption encoder on the host only for a prompt it reads.
- OpenCL: build the image kernels with Intel's OpenCL runtime.
- Cluster: let 2 instances each be the prefill peer of the other.
- Cluster: hand a prompt over from a peer whose model sits in its batch worker.
- API: answer about an image sent to the generate route of a Qwen vision model.
- API: count images, cached tokens and batched answers in the prompt tokens of every route.
- Placement: keep a model on 1 card when an unasked context would spread it.
- Vulkan: run the layers that take no position rotation without one.
- Vulkan: load a model whose layers Vulkan refuses on the host.
- Vulkan: count the layers a Vulkan card runs as on that card, not on the host.
- Vulkan: cache keys and values at the nearest half, not rounded toward 0.
- Models: take a stored model's draft and routing prior from its own manifest.
- Vision: keep a Qwen answer about an image on track past its first token.
- API: accept audio and image uploads up to the limit of their route.
- API: open the window a load request's num_ctx asks for.
- Speech: decode Parler with its checkpoint's sampling, length limit, end rule and 1 seed.
- Delete: unload a model after releasing the table of loaded models.
- API: return 1 log-probability per token, streamed, drafted or batched, to its own request.
- Embeddings: scale a vector cut to fewer dimensions back to unit length.
- Matmul: dequantise on the card for 1 call when its workspace is short, keeping no copy.
- API: keep the prompt caches, stored responses, files, batches and renders of each API key to itself.
- Renders: stop a render by its X-Request-Id, a transcription or separation sooner, a cancelled SDXL row at once.
- Memory: give the allocator's free memory back to the host once it passes the headroom.
- Sessions: keep no token list for each session id, and count the resident conversation.
- Loaded models: list them without holding the table through each engine's wait.
- Chat: accept a turn with media or a tool result and no text.
- Responses: report a stream cut upstream as incomplete.
- Streams: end a failed OpenAI stream with an error frame, not as text.
- API: stop listing 2 multi-device routes that do not exist.
- Messages: refuse a request without max_tokens, hold it to the usual bounds and check them before fetching images.
- KV disk cache: write a manifest through a temporary file.
- Chat: answer a failed preload with an error, not as a load.
- Generate: send the prompt as given to a package that has no template layer.
- ROCm: run the hardware 8-bit dot product on RDNA3 and RDNA4 cards.
- Serving: honour stop sequences on the grammar path.
- CUDA: run the kernels on a driver older than the bundled NVRTC.
- Serving: answer requests batched together on an NVIDIA card.
- Serving: send the requests queued at the gate to the batch worker together.
- Serving: continue a preempted request where it stopped.
- Loading: give the loader's worker threads the main thread's stack.
- Speculative decoding: prefill the drafter in chunks, as its target is.
- Speculative decoding: accept a drafter whose vocabulary is padded to fewer rows.
- Speculative decoding: draft 1 token first and lengthen drafts a row at a time.
- Speculative decoding: verify a Qwen3 draft's rows exactly as their decode steps.
- Serving: go on from the failed chunk when a long prompt runs out of card memory.
- Inference: size the attention scores of a prompt on a card from its free memory.
- Vision: place the text after a tall image as the reference does.
- Models: load the model of an Ollama manifest that lists its projector first.
- Serving: keep the images of a vision model when requests arrive together.
- Inference: attend over the full-width cache on the host when the cache is not quantised.
- Models: list a released checkpoint that loken serves in place.
- API: end a chat stream on an engine error with the error alone.
- Serving: answer with a model split over several cards.
- OpenCL: offer a GPU that shares the host's memory only what the host can spare.
- Placement: plan no card when the driver sees none, though the machine has some.
- Speculation: try a waiting prompt lookup again when the answer copies the prompt.
- Speculation: cut a hybrid model's recurrent state back to the drafted tokens it kept.
- ROCm: give a drafted token the logits of its decode step.
- Whisper: keep the words at the 30 s window edges of a long recording.
- MoE: give a drafted token the router logits of its decode step.
- Speech: serve the Piper voices in a build without the image and video families.
- Cluster: read here a streamed prompt that carries a Qwen image or video.
- API: report the prompt time of an answer the batch worker writes.
- Serving: end a batched answer that a failed step cuts with an error.
- Wan: keep every stored value of the norms and the patch embedding.
- Kernels: refuse the inputs and launches that failed without an error.
- State cache: load when the disk probe measures no rate.
- MoE: drop and read ahead the expert pages of a model loaded from 1 file.
- Vulkan: give each block of 32 activations its own scale, the K-quant mat-mat apart.
- Serving: read a long prompt on the serial path in the rows the cards afford.
- Speculation: judge a prompt lookup trial after a hold on its own cost.
- Hub: pull the merges file of a tokenizer that ships no tokenizer.json.
- Models: load a 16-bit float checkpoint without 32-bit copies on the host.
- Models: end a multimodal checkpoint's answer at its end token, not at "!".
- Stats: report the memory of a ROCm card in the periodic stats line.
- Models: load and answer when host memory is short instead of being killed.

### Removed

- API: derive window, cache format and priority instead of loken headers and options.
- Config: stop reading 17 keys and the [cpu] table, which loken decides itself.

## [0.4.2] - 2026-10-28

### Changed

- Server: log every error answer with its reason.

### Fixed

- Inference: open the configured context window for a model on the host alone.
- Inference: serve a model whose BF16 output head runs on the host.

## [0.4.1] - 2026-10-26

### Added

- Tensor parallelism: split the layers of an AWQ checkpoint across cards.

### Changed

- Images: add the Qwen-Image feed-forward bias on the card.
- Images: run Qwen-Image attention in the flash kernel.
- Quantised projections: dequantise straight to BF16 for many rows.
- AWQ: read a projection left on the host as eight-bit blocks, unpacked once.
- Placement: choose the context by how fast a token reads its weights.
- Placement: hold a model on one card at a smaller context before spreading it.
- Decode: attend over a long context with the split-K kernel only.

### Fixed

- Devices: wait for a cached kernel constant to reach the card.
- ROCm: copy every row of a large strided copy, so Qwen-Image renders.
- Loading: send a layer's fused weights to the card once.
- Images: decode a Qwen-Image in smaller tiles on a card before the host.
- Images: run the feed-forward of a large Qwen-Image F32 checkpoint.
- Images: run the DiT flash attention only on the cards the build has it for.
- Placement: after a card runs out of memory, take blocks off that card alone.
- Placement: keep the context a request asks for.
- Tensor parallelism: open one copy of NCCL and sum outside the decode graph.
- Devices: time the half-precision product and say why a sum is not weighed.
- Images: place the Flux VAE on the fastest card that has room for it.
- Devices: rank the cards by a timed matrix product.
- Placement: count the packed weights of an AWQ checkpoint.
- Images: reserve for one frame on a Qwen-Image generation and two on an edit.

### Removed

- Bench: move the campaign, its method, its reports and its results to the benchmarks repository.

## [0.4.0] - 2026-10-24

### Added

- Devices: list the OpenCL GPUs that placement offers.
- Bench: publish the pages of the ROCm, Vulkan and OpenCL campaigns.
- Models: load the routed experts of Gemma 4 from its safetensors.
- Models: check every architecture against reference logits.
- Models: load Gemma 4 from its safetensors.
- Models: apply ALiBi to MPT on the host.
- Devices: list the Vulkan GPUs that placement offers.
- Bench: measure a card through its Vulkan or OpenCL backend.
- Models: read the vocabulary of a checkpoint from its tokenizer.model.
- Models: load architectures from the safetensors their publisher ships.
- Docs: list the architectures the loader reads and the formats of each.
- API: name the backend of each device in the device list.
- OpenCL: run the layers of a language model on an Intel GPU.
- Vulkan: run attention over a sliding window.
- Vulkan: run attention heads up to 256 wide.
- Vulkan: run feed-forward layers that use GELU.
- Vulkan: run routed mixtures of experts.
- Chat: store the decode states a cache writes to disk compressed.
- Serve: reuse the whole prompt when the same prompt comes back.
- Voxtral: load the Hugging Face safetensors checkpoint in both layouts.
- Audio: transcribe with Voxtral on the transcription route, in its own mode.
- Vulkan: run mat-vec on int8 activations with the integer dot product.
- Vulkan: read prompts through tiled mat-mat and attention kernels.
- Vulkan: run the layers of a dense decoder on Vulkan devices.
- Forecast: serve the Timestral checkpoints, with 21 quantiles and covariates.
- Release: ship one Linux archive that serves NVIDIA and AMD cards.
- Models: keep the routing profile of a DeepSeek V4.1 model across restarts.
- Convert: encode q4_K, q5_K and q6_K on the card.
- Convert: write iq3_s and iq2_s, weighted by importance, on the card.
- Convert: write iq4_nl and iq4_xs, weighted by importance, on the card.
- Convert: weight q4_0, q4_1, q5_0 and q5_1 by importance.
- Requantize: write q2_0.
- Models: serve a mixture from several files of its checkpoint, expert by expert.
- Offload: run IQ3_XXS and Q3_K routed experts on the cards.
- Convert: write IQ3_XXS weighted by each expert's importance.
- Convert: encode IQ3_XXS and Q3_K on the card, error compensation included.
- API: time each streamed token's parts in /api/stage_perf.
- ROCm: run the grouped-query prefill attention on AMD matrix cores.
- ROCm: run the DiT flash attention on AMD matrix cores.
- ROCm: run the MoE prefill over quantised experts on AMD matrix cores.
- ROCm: run the IMMA Q4_K and Q5_K matrix products on AMD matrix cores.
- ROCm: run the AWQ W4A16 GEMM on AMD matrix cores.
- ROCm: replay the recorded decode step on AMD cards.
- GPU: drive NVIDIA and AMD cards from one binary built with cuda and rocm.
- Audio: keep Stable Audio, EzAudio and ACE-Step weights in half precision.
- ROCm: read a prompt through the WMMA flash prefill on AMD cards.
- Bench: measure AMD cards against ollama's ROCm build.
- ROCm: serve language models on AMD cards through HIP.
- Forecast: serve time series quantile forecasts on POST /v1/forecast.
- Ultravox: serve a checkpoint alone and pull the text model it names.
- MIDI: render at the tempo the request names in bpm.
- Audio: answer generated clips as mp3 or flac by response_format.
- Messages API: read images by URL from allowed hosts.
- API: cap images, pixels and audio seconds per request.
- Experts: keep a routing profile per model in the store, read at load.
- Bench: measure Whisper against vLLM on a golden recording.
- Responses API: read image and audio input parts as chat completions does.

### Changed

- Bench: fetch no peer engine to measure the OpenCL backend.
- Serve: read only the tokens a conversation's next turn adds.
- Models: stop the context where a model starts scaling its queries.
- Models: keep the feed-forward output in 6 bits when float weights take 4.
- Models: rebuild float weights in 8 bits when the checkpoint fits its memory.
- Serve: read the chat format once when a model loads.
- Bench: write the records under the loken schema.
- Bench: find the measurement tool under its new name lokprof.
- Cache: write every cache file in one binary format.
- Chat: write the state cache index less often.
- Scripts: take the vLLM environment from the caller.
- Bench: keep the engines it installs inside the repository's results folder.
- ROCm: launch fewer kernels per decoded token, and take split-K attention for MHA models.
- Serve: keep a losing prompt lookup waiting across requests.
- ROCm: split the MoE router gemv across more blocks.
- Serve: warm the prefill path when a model loads.
- Bench: report prefill, time to first token and decode cold and warm.
- ROCm: cache device buffers instead of freeing each one.
- Convert: keep the cards busy through a conversion, about seven times faster.
- GPU: verify the draft rows of an MHA model in one launch per kernel.
- Offload: read a streamed expert not in memory in full read-ahead windows.
- Offload: read a short prompt on the cards with their kept weights and overlap card and host experts.
- Offload: stream a prompt's weights with direct reads and drop pages off the compute thread.
- CUDA: serve a one-row mat-vec whose row fits one pass of a block at 4 to 16 rows a block.
- CUDA: fuse a dense decode's gate, up and silu in one launch for 32-value and mixed pairs.
- CUDA: write a decode token's Q8 keys and values in one launch at any head count.
- CUDA: add the QKV bias in the mat-vec that projects a decode row.
- MoE: read each host expert once for all the prompt rows that route to it.
- Bench: sweep the whole matrix with FULL=1, one fair-run per prompt, in place of campaign.sh.
- Whisper: transcribe with half-precision weights and a reserved decoder cache.
- Placement: shrink a context nobody asked for until the cards hold a tensor split.
- Bench: say in an error cell whether loken or lokprof lacks the model.
- Bench: keep the store in bench/store, outside the server's configuration.

### Fixed
- Models: keep a model loaded while a request runs on it after its keep-alive ends.
- Responses: report the prompt tokens a kept state served.
- Models: read the text model of Mistral 3 and Voxtral checkpoints.
- Models: keep a model with an ALiBi bias off the cards.
- Vulkan: cut the cache window to what the card holds beside the weights.
- Models: give a multi-query Falcon one key-value head.
- Models: read the layer norms of DBRX and MPT without a bias.
- Models: size the sliding window of local attention layers as declared.
- Build: link the tensor-core kernels in a build for older cards.
- Models: skip the fused feed-forward norm of a layer that has no such norm.
- Models: keep a prompt of several tokens out of the one-row graph buffers.
- Models: detect a checkpoint whose weights are MXFP4.
- Chat: keep the prompt of a base model out of chat turns.
- Models: check draft rows of a model with one query head per key head.
- Models: read the learned positions of a GPT BigCode checkpoint.
- Models: apply the feed-forward norm bias of a LayerNorm model.
- Build: link the image attention kernel in a build for older cards.
- Models: read the fused QKV bias of a GGUF file.
- Models: serve a request past the declared window without reloading.
- Image: run the Z-Image blocks on OpenCL and read the result back.
- Conversations: keep the states of a model without a card in host memory.
- Models: read the header of a model file through a buffer.
- ROCm: stop pinning file mappings the driver refused once.
- Chat: give back cached states when the host memory runs short.
- Chat: save a decoded state without copying it in memory.
- Memory: keep host budgets inside a container or service memory cap.

- Chat: count a prompt restored from the state cache as already read.
- Chat: keep a long prompt whole when the client allows a huge reply.
- Tier: wait longer for a worker the processors are not running.
- Serve: read only the tokens a turn adds on a model with a recurrent state.
- Prefill: give back the kept conversation states before a chunk runs out of memory.
- Serve: keep each conversation's state while others take turns, until a set lease ends.
- Chat: count a streamed answer's time to first token from the request.
- State cache: keep prompt states in memory when no directory is set.
- KV cache: keep both buffers whole when a growth runs out of memory.
- Show: describe the models held in the Hugging Face store.
- Chat: send the arguments of an Ollama tool call as an object, and read them back.
- Messages: fold a system message sent after a turn into the system prompt.
- Pull: answer a media checkpoint pulled without streaming instead of failing its load.
- Offload: run card experts of two formats as two batches in one block.
- ROCm: build the matrix kernels for every AMD target, with a fallback on RDNA2.
- ROCm: answer the first prompt a model reads after it loads on an AMD card.
- ROCm: keep two models decoding on one AMD card from hanging.
- GPU: verify draft rows on a card as their decode steps.
- Convert: read a layer span as first to last, both included.
- SDXL: render with a bf16 UNet on a card.
- GPU: keep two models decoding on one card from mixing their activations.
- Tokenizer: lead a bare prompt with the BOS a model's chat template opens with.
- GPU: release a card's context once no loaded model uses it, so the card cools.
- GPU: list the cards at start without opening a context, so an idle server leaves them cold.
- CUDA: bind the card before a page-locked staging buffer is made on a new thread.
- Attention: give a long prompt the same greedy tokens however full the cards are.
- CPU: give each row of a draft verify the arithmetic of its decode step.
- ROCm: stage copies from read-only mappings so the driver never tries to pin them.
- ROCm: keep the rounding intrinsics uncontracted so cached experts match the host.
- ROCm: take lent expert blobs outside the stream-ordered pool on AMD cards.
- ROCm: keep AMD builds of a kernel off the paths chosen by PTX version.
- Indexer: fit the WMMA attention of prompt rows in 64 KB of shared memory.
- Experts: take the narrowest tile the card builds at or above a requested width.
- API: read a checkpoint's declared context once, not on each request past it.
- Prefill: stop probing and trimming the cards on each request to a dense model.
- Experts: give the same greedy tokens whichever experts a cached layer holds.
- Experts: finish moving locked experts before a decode waits on the tier worker.
- CUDA: record graphs on a stream of their own, out of other threads' work on the card.
- Experts: give the same greedy tokens whichever experts the cards hold.
- Audio: prompt Voxtral in its own instruction format, not the file's template.
- Bench: never stamp an earlier result with a build that measured nothing.

### Removed

- Bench: drop the cross-hardware table; each hardware keeps its own page.

## [0.3.1] - 2026-10-19

### Fixed

- Placement: keep every network on the host when the server runs with --cpu.
- Attention: count each key once when the flash prefill splits a long history.
- Experts: fail a request whose tier worker stalls instead of hanging the card.
- Experts: lock and unlock host pages without holding the tier worker back.
- Audio: decode a FLAC file whose last frame is short.

## [0.3.0] - 2026-10-18

### Added

- Experts: a file larger than the host keeps its most routed experts of every layer in host memory.
- Context parallelism: context_parallel cuts each attention layer's cache over every card.
- Benchmarks: one page per hardware configuration, written from a store of runs.
- Benchmarks: a cross-hardware table of the cells two machines measured alike.
- Benchmarks: archived runs are held in the store and left out of the tables.
- Benchmarks: a page per hardware and backend, the backend read from the server.
- Disaggregation: prefill_peer has another instance read long prompts and hand their KV over.
- Disaggregation: recurrent and windowed models hand over their whole decode state.
- Expert parallelism: a layer's experts split across cards where measured links make it pay.
- CPU: AVX-512 kernels for attention, the KV cache and elementwise ops.
- CPU: AVX-512 kernels for the quantized GEMMs and dots.
- CPU: the SIMD level is chosen at start and capped by the cpu level setting.
- Build: LOKEN_CUDA_ARCHS builds the kernels for the named architectures only.
- Prefill: a long prompt crosses each layer once, chunks sized for the tightest card.
- Data parallelism: `replicas` serves a model whole on several cards, requests spread.
- API: Cohere /v2/rerank and /v2/embed and vLLM /score over the existing models.
- Completions: best_of samples that many completions and returns the n most probable.
- Tensor parallelism: dense and mixture layers split across cards when measured costs pay.
- Profile: /api/stage_perf times each graph, stage and kernel on the card.
- Draft head: a model's own head drafts, read as released, its verify replayed as a graph.
- Speculation: a verify runs its rows together, recurrent models included, while it pays.
- Decode: a step replays as CUDA graphs, indexed attention and n-gram layers included.
- Experts on the host: every card caches them, a miss runs on the card or the host.
- Cluster: a dense model is cut across nodes from measured cost, peers sampled for theirs.
- Build: scripts/build-portable.sh packages Linux archives, with or without CUDA.
- Sparse attention: the indexer selects and attends on the card, decode and prefill.
- Experts: tensor-core kernels for the IQ formats and q2_0, integer sums on the host.

### Changed

- KV cache: a quantised cache snapshots its packed codes, and a copy that would not fit is skipped.
- Speculative decoding: a verify window replays the kept rows instead of storing their states.
- Prefill: the gated delta rule runs a chunk's rows as blocked matrix products.
- Prefill: the delta-net convolution spreads a chunk's rows over the card.
- Prefill: indexed attention over a prompt's rows runs on tensor cores, three times faster.
- Decode: kept graphs outlive a prefill that lends the expert slots; a replay's room is measured apart.
- Decode: the recorded step and windows are kept for the next request while nothing they read moved.
- Prefill: the experts' products of a chunk run in launches sized for their longest run.
- CPU: the IQ4_NL expert products decode each block once for every row.
- CPU: AVX-512 kernels for the IQ2_S, IQ3_XXS and IQ4_NL expert products.
- CPU: the IQ2_S and IQ3_XXS expert products decode each block once for every row.
- Experts: read cold prefill experts directly, sparing the page cache.
- Context parallelism: each range is attended by the card's flash kernels, read in place.
- Context parallelism: a group of query heads reads its keys and values once.
- Experts: the cache forgets a route within a few tens of tokens and follows the text in hand.
- Tests: the gate profile rebuilds a change in place, the suite runs in half the time.
- CPU: the prompt's KV cache rows are quantized on the thread pool.
- CPU: large matmul outputs are cleared on the thread pool.
- CPU: the build targets x86-64-v3 and refuses an older CPU with a message.
- BF16 weights stay BF16 on a card instead of widened to F32 at the load.
- Wide residual: mixers in half precision for a prompt, int8 rows for a step.
- Decode: norms, rotations, shared experts and mixers run in fewer fused launches.
- Linux: CUDA and OpenCL load at run time; without them the server runs on the host.
- Prefill: a request is sized from free memory, not from cached blocks.
- Build: a CUDA kernel recompiles only when its own sources change.
- Build: CUDA runs on the published cudarc crate, graph capture handled by loken.

### Fixed

- Decode: layers with RoPE factors, attention sinks or no q/k norm run the fused attention step.
- Prefill: a prompt over an F16 cache attends through the flash kernel, not whole score matrices.
- Serving: the move to batched serving is made under the serial gate, never mid-answer.
- DeepSeek V4.1: a long prompt is prefilled in one pass, not decoded past its first chunk.
- Tool calls: DeepSeek V4.1 sees its tool results and past calls as its encoder writes them.
- Tensors: a forced contiguous copy owns its storage instead of sharing its source.
- Attention: a decode with normed heads and a partial rotation no longer reads past its tables.
- Tool calls: a call that names its tool under a tool key is read as a call.
- Tools: a call that closes a value with its key's tag, or tags its arguments, is read.
- Server: a request that ends no longer stops another one served beside it.
- Placement: a model the cards hold whole keeps room for whole prefill chunks.
- Build: the cpu-only configuration compiles again.
- Server: the [server] port of config.toml is used when --port is not given.
- Config: a partial file is read, the settings it leaves out take their defaults.
- N-gram embedding: a forward run again after a failure finds the history where it was.
- OpenAI and Anthropic APIs: a penalty applies only as the request names it.
- Chat: a repository with a folder per quantisation is prompted with its own template.
- Load: a file with a rotary scaling and no context key takes its scaled window.
- Chat: a grammar-constrained answer is content, not reasoning.
- Tools: a call written as JSON inside a function tag is read as a call.

## [0.2.1] - 2026-10-05

### Fixed

- Load: an undeclared context window no longer takes the weights' room.
- Bench: the model filter applies to the run, not only to the printed plan.

## [0.2.0] - 2026-10-05

### Added

- API: a Responses endpoint over chat completions.
- API: a continuation store for the Responses endpoint.
- OpenAI: the n, logprobs, logit_bias, suffix and echo parameters.
- OpenAI: reasoning_effort and parallel_tool_calls.
- OpenAI: response headers for context window, KV format and priority.
- Messages: count_tokens and the models list.
- Messages: thinking, tool choice and structured output.
- Ollama: version, embed dims and truncate, pull progress.
- Ollama: tool choice and reasoning.
- Ollama: legacy embeddings.
- Ollama: copy, delete, create, tags and show endpoints.
- API: files, batches and moderations.
- API: message batches.
- Multimodal: image input.
- Multimodal: image results by URL.
- Multimodal: MP3 speech output.
- Multimodal: audio input to a chat turn through a model's speech tower.
- Reasoning: returned apart from the answer.
- Reasoning: per-token log-probs.
- Tools: parsing for gpt-oss Harmony.
- Tools: parsing for DeepSeek calls.
- Tools: parsing for qwen3-coder XML calls.
- Tools: parsing from chat templates.
- Tools: DeepSeek DSML calls.
- Tools: a generic tool-calls block fallback.
- KV: context-shift reuse of the resident cache past the window.
- KV: snapshots resume a conversation from its cache.
- KV: snapshots stored on disk as well as in memory.
- Offload: a mixture's experts spilled across cards and host.
- Offload: a streamed mixture reads the next layer's experts ahead of its gate.
- Offload: read-ahead depth, reader threads and promotion are derived.
- Offload: host experts cached on the cards by use.
- Offload: a wide batch's experts streamed through two card buffers.
- Offload: a cache plan for a file with no room to prefill.
- Placement: one ladder over the measured cards for every language model.
- Placement: the same ladder for the image and speech stacks.
- Placement: a ledger of every conversion a load makes.
- Prefill: a wide mixture's prompt runs its attention and experts on the cards.
- Prefill: a layer's experts staged on a copy stream.
- Cache: decode states stored on disk by prefix for the large mixture.
- Cache: decode states stored on disk for transformers with recurrent layers.
- Cache: block size, spacing and budget derived from the model.
- Quality: a policy (exact, balanced, fast) bounds lossy routing, default exact.
- Quality: a teacher-forced harness scores a setting against the exact run.
- DeepSeek: V4.1 Flash served from its own GGUF, experts streamed from disk.
- Requant: a model into a new tag, on the GPU, as detached work.
- Loading: a split GGUF as one source.
- Loading: a BPE or Unigram tokenizer from a GGUF.
- Loading: a split GGUF set on the generic route, read part by part.
- Models: the qwen4exp architecture on the generic route.
- Models: a sparsely read n-gram table for the qwen4exp family.
- Formats: IQ4_NL, IQ4_XS, IQ3_XXS, IQ3_S, IQ2_S and Q2_0 read.
- Formats: the same formats run on a card for experts.
- CLI: pull owner/repo:SELECTOR fetches one quantisation of a GGUF repo.
- CLI: a pull stops after eight times the longest progress gap.
- Tests: a gate makes every stored-weight conversion state its reason.
- Cluster: routing for every surface, handing a model or render to a peer.
- Cluster: the model listing shows a loading model and per-layer placement.
- Cluster: a peers endpoint.
- Cluster: a scrape endpoint behind a feature.
- Adapters: rank-decomposition adapters through the layer-swap route.
- Speculative: a configured draft model.
- Speculative: per-position logits for verify.
- Speculative: a resident mixture verifies a prompt-lookup block at once.
- Agents: loken launch for a coding agent.
- Perf: per-layer timing on demand.
- Perf: a streamed decode reports where its time goes, by stage, on demand.
- Docs: every config key and route.
- Docs: ten lessons.
- CI: a cargo-deny job.

### Changed

- Attention: prefill accumulates over a band of keys in one pass.
- Cluster: yielding work goes to a peer, not the node holding a conversation.
- Perf: wide RMSNorm decode kernel.
- Perf: prefill chunk priced once.
- Perf: pages advised.
- KV: num_ctx clamped to the checkpoint.
- KV: the quantised cache grows to context.
- API: the version endpoint names the build commit.
- Cluster: media requests count as in flight.
- Cluster: the pool returns between jobs.
- Offload: the expert host-tier sizing is logged at load.
- Build: a commit or checkout no longer recompiles the CUDA kernels.
- Docs: condensed to lead with the point, detail in tables.
- Loading: token tables and text-encoder embeddings stay in the stored format.
- Loading: a card weight built by the loader keeps no host copy.
- Loading: host expert stacks keep the file's format when the host reads it.

### Fixed

- Loading: a card allocation failure reports how full the card was.
- Attention: the sparse layer's scores are bounded to a block of rows.
- Attention: the sparse layer's cache is held at half width on a card.
- Placement: the KV cache is priced by the cache the layers build.
- Catalogue: a repository with a model and projector resolves to the model.
- Tests: the demand-ledger tests no longer fail when run in parallel.
- Catalogue: a GGUF in the Hugging Face store is listed as one.
- Catalogue: the parameter count of a split set is the model's own.
- Client: a load the daemon refuses is reported, not printed as a success.
- CLI: pull follows the daemon's streamed progress, with no total deadline.
- Loading: a part of a split set on the single-file route is refused.
- Reasoning: markers survive decode, so the thought is not the answer.
- Reasoning: a prompt-opened block is read as reasoning (deepcoder, qwen3.5).
- Tools: calls no longer leak into the answer (XML, arrays, failed templates).
- Streaming: no dropped or repeated characters.
- Streaming: no resent settled text.
- KV: prefix reuse prefills from the cache's real position.
- Precision: attention and rotary run at full precision, so no misspelling.
- Reload: a failed widening or format reload restores the model.
- Cluster: a member/address lock-order inversion no longer deadlocks a node.
- Generation: never returns an empty answer when one was asked for.
- Offload: a reopened placement resets its per-card accounting.
- Build: the host-only and no-AVX2 builds work again.
- Build: newer clippy lints pass.
- Bench: empty cells no longer render as literal asterisks.

### Security

- Deps: patched the rustls TLS handshake advisory (RUSTSEC-2026-0285).
- Logs: user text kept out; only lengths and counts are logged.
- Limits: upload names, body sizes, batch counts and image-URL hosts bounded.
- Cluster: a cross-node replay refused on a placeholder digest.

## [0.1.0] - 2026-08-29

First public release. What it does not yet do, and where it is known to be slower, is in
[docs/STATUS.md](docs/STATUS.md) rather than left for you to find.

### Added

- Server: OpenAI- and Ollama-compatible for GGUF and safetensors LLMs.
- Image: generation and editing across several families.
- Audio: music, sound, speech, transcription, and separation.
- Video: generation, text and image conditioned.
- Placement: one rule per family, weights charged at what the pool keeps.
- Energy: per-request accounting beside the timings.
- Backends: CUDA and OpenCL.
- Features: a Cargo feature per media family; uncompiled answers unknown model.
- Cluster: multi-node, a node announces itself and work is routed.
- Bench: a harness against other engines under one protocol.
- Provenance: a measure and a gate holding the attribution file to it.
