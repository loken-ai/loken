# Changelog

All notable changes to loken. The format follows [Keep a Changelog](https://keepachangelog.com/1.1.0/).

## [Unreleased]

### Added

- Sparse attention: the indexer selects and attends on the card, decode and prefill.
- Experts: tensor-core kernels for the IQ formats and q2_0, integer sums on the host.

### Changed

- Prefill: a request is sized from free memory, not from cached blocks.
- Build: a CUDA kernel recompiles only when its own sources change.
- Build: CUDA runs on the published cudarc crate, graph capture handled by loken.


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
