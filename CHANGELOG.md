# Changelog

All notable changes to loken. The format follows [Keep a Changelog](https://keepachangelog.com/1.1.0/).

## [Unreleased]

### Added

- API: a Responses endpoint over chat completions, with a continuation store.
- OpenAI: n, reasoning_effort, parallel_tool_calls, logprobs, logit_bias, suffix, echo.
- Messages: count_tokens, models list, thinking, tool choice, structured output.
- Ollama: version, embed dims/truncate, pull progress, tool choice, reasoning.
- API: files, batches, message batches, and moderations.
- Ollama: legacy embeddings, copy, delete, create, tags and show.
- Multimodal: image input, image results by URL, and MP3 speech output.
- Multimodal: audio input to a chat turn, for models with a speech tower beside the text GGUF.
- Reasoning: returned apart from the answer, with per-token log-probs.
- Tools: parsing for gpt-oss Harmony, DeepSeek, qwen3-coder XML, and templates.
- OpenAI: headers for the context window, KV format, and background priority.
- KV: context-shift reuse of the resident cache past the window.
- KV: snapshots resume a conversation from its cache, in memory or on disk.
- Offload: a mixture's experts spilled across cards and host.
- Placement: one ladder over the measured cards places every language model and the image and
  speech stacks, with a ledger of every conversion a load makes.
- Offload: a streamed mixture reads the next layer's experts into page-locked memory ahead of
  its gate; read-ahead depth, reader threads and the sightings that move an expert to a card
  are derived from measurement.
- Prefill: a wide mixture's prompt runs its attention, residual stream, expert products and
  routing on the cards, a layer's experts staged on a copy stream.
- Cache: decode states stored on disk by prefix for the large mixture and for transformers
  with recurrent layers; block size, spacing, budget and restore-or-prefill derived.
- Quality: `quality_budget` picks lossy expert routing from a stated divergence; `expert_deferral`
  lets a share of host experts finish a layer late. Both default to exact.
- Quality: a teacher-forced harness scores a setting against the exact run.
- DeepSeek: V4.1 Flash served from its own GGUF, experts streamed from disk.
- Tools: DeepSeek DSML calls, and a generic tool-calls block fallback.
- Requant: a model into a new tag, on the GPU, as detached work.
- Loading: a split GGUF as one source; a BPE or Unigram tokenizer from it.
- Loading: a split GGUF set on the generic route, read as one model part by part.
- Models: the qwen4exp architecture on the generic route: wide residual, delta-rule and indexed
  attention layers, a sparsely read n-gram table, 160 to 512 routed experts.
- Formats: IQ4_NL, IQ4_XS, IQ3_XXS, IQ3_S, IQ2_S and Q2_0 read, and run on a card for experts.
- Offload: host experts cached on the cards by use, a wide batch's experts streamed through two
  buffers, a cache plan for a file with no room to prefill.
- CLI: `loken pull owner/repo:SELECTOR` fetches one quantisation of a GGUF repository.
- Tests: a gate requires every conversion of a stored weight to state its reason.
- CLI: `loken pull --stall-factor`, the multiple of the longest frame gap that stops a pull.
- Cluster: routing for every surface, handing a model or render to a peer.
- Cluster: the model listing shows a loading model and per-layer placement.
- Cluster: a peers endpoint, and a scrape endpoint behind a feature.
- Adapters: rank-decomposition adapters through the layer-swap route.
- Speculative: a configured draft model, and per-position logits for verify.
- Speculative: a mixture verifies a prompt-lookup block at once when resident.
- Agents: loken launch for a coding agent; per-layer timing on demand.
- Perf: a streamed decode reports where its time goes, by stage, on demand.
- Docs: every config key and route, ten lessons, and CI with cargo-deny.

### Changed

- Attention: prefill accumulates over a band of keys in one pass.
- Cluster: yielding work goes to a peer, not the node holding a conversation.
- Perf: wide RMSNorm decode kernel, prefill chunk priced once, pages advised.
- KV: num_ctx clamped to the checkpoint; the quantised cache grows to context.
- API: the version endpoint names the build commit.
- Cluster: media requests count as in flight; the pool returns between jobs.
- Offload: the expert host-tier sizing is logged at load.
- Build: a commit or checkout no longer recompiles the CUDA kernels.
- Docs: condensed to lead with the point, detail in tables.
- Loading: token tables and text-encoder embeddings stay in the stored format; a card weight
  built by the loader keeps no host copy.
- Loading: host expert stacks are used in the format the file stores when the host kernels
  read it.

### Fixed

- Loading: a card allocation failure reports how full the card was.
- Attention: the sparse layer's scores are bounded to a block of rows, its cache held at half
  width on a card.
- Placement: the KV cache is priced by the cache the layers build, not a byte per element.
- Catalogue: a repository holding a model and its projector resolves to the model file.
- Tests: the demand-ledger tests take turns, so a module's parallelism no longer fails them.
- Catalogue: a GGUF in the Hugging Face store is listed as one, with the model's own parameter
  count, split sets included.
- Client: a load the daemon refuses is reported, not printed as a success.
- CLI: pull follows the daemon's streamed progress, with no total deadline.
- Loading: a part of a split set on the single-file route is refused, not built as pass-through
  blocks.
- Reasoning: markers survive decode, so the thought is not the answer.
- Reasoning: a prompt-opened block is read as reasoning (deepcoder, qwen3.5).
- Tools: calls no longer leak into the answer (XML, arrays, failed templates).
- Streaming: no dropped or repeated characters, no resent settled text.
- KV: prefix reuse prefills from the cache's real position.
- Precision: attention and rotary run at full precision, so no misspelling.
- Reload: a failed widening or format reload restores the model.
- Cluster: a member/address lock-order inversion no longer deadlocks a node.
- Generation: never returns an empty answer when one was asked for.
- Offload: a reopened placement resets its per-card accounting.
- Build: the host-only and no-AVX2 builds, and newer clippy lints.
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
