<p align="center">
  <img src="https://raw.githubusercontent.com/loken-ai/.github/main/brand/png/lockup.png" alt="LOKEN - Local . Multimodal . Green" width="420">
</p>

# loken

A local inference server for language **and** generative-media models, written in Rust. One
process serves chat, images, audio, speech, transcription and video, over an
OpenAI-compatible API, an Ollama-compatible one and the Messages API.

```sh
cp config.toml.example config.toml   # the daemon reads this from its working directory
cargo build --release                # target/release/{lokend,loken}
./target/release/lokend serve        # the daemon, on port 11435
./target/release/loken  list         # and the client that talks to it
```

It speaks the OpenAI API, so anything that already talks to one works unchanged:

```sh
curl -s localhost:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3:1.7b",
  "messages": [{"role": "user", "content": "Say hello in six words."}]
}'
```

**Inference never leaves the machine.** No account, no API key, no hosted fallback. What does
reach the network: fetching a model you asked for, and a cluster node announcing itself if you
configure one.

## What it does

| Capability | Route | Configuration | State |
|---|---|---|---|
| Chat and completion, streamed or whole | `/v1/chat/completions`, `/v1/completions`, `/api/chat`, `/api/generate`, `/v1/messages` | `model_id`, `context_length` | Measured against ollama, [STATUS](docs/STATUS.md#decode-rate-vs-ollama) |
| Responses API | `/v1/responses` | | Runs |
| Tools, JSON schema, grammar-constrained output | the chat routes, `tool_choice`, `response_format` | | Runs |
| Reasoning apart from the answer, Harmony included | the chat routes, `think`, `reasoning_effort` | | Runs |
| Vision and speech in a chat turn | the chat routes, image and audio content | | Runs |
| Log-probabilities, logit bias, seeds | the chat and completion routes | | Runs |
| Embeddings and reranking | `/v1/embeddings`, `/api/embed`, `/v1/rerank` | | Runs |
| Placement across cards and the host | automatic | `max_gpu_memory_fraction`, `kv_quant` | Measured, [STATUS](docs/STATUS.md#placement-measured-separately) |
| Expert spilling for mixtures that do not fit | automatic | | Measured |
| Continuous batching | | `continuous_batching` | Runs, cards only |
| Speculative decoding | `/api/draft/attach` | `draft_model` | Measured on one pair, [guide](docs/guides/drafter.md) |
| KV reuse across conversations, on disk, past the window | automatic | `kv_snapshots`, `kv_disk_dir`, `kv_shift_reuse` | Runs, [guide](docs/guides/kv-memory.md) |
| Files, batches, message batches | `/v1/files`, `/v1/batches`, `/v1/messages/batches` | `files_dir` | Runs |
| Moderation with a classifier | `/v1/moderations` | `moderation_model` | Runs, five families |
| Images, edits, variations, adapters | `/v1/images/*`, `/v1/loras` | `lora_dir` | Judged partially, [STATUS](docs/STATUS.md#judged-partially) |
| Video | `/v1/video/generations` | | Runs |
| Transcription and translation | `/v1/audio/transcriptions`, `/v1/audio/translations` | | Runs |
| Speech, music, sound effects, separation | `/v1/audio/speech`, `/v1/audio/generations`, `/v1/audio/separate` | | Runs |
| Energy per request | every answer | `[energy]` | Measured |
| Replicated serving across machines | `/api/cluster/*` | `[cluster]` | Runs at one node, unmeasured at two |
| Sharding one model across machines | | | Written, wired to nothing |
| Scrape endpoint | `/metrics` | feature `metrics` | Runs |
| Authentication and rate limit | every route | `require_auth`, `api_keys`, `rate_limit_per_minute` | Runs, off by default |

Each media family is a Cargo feature - `image`, `audio`, `video`, `midi`, or `media` for all. A
family that is not compiled reports its models as unknown rather than dispatching to something
absent. The families served are in [`docs/MODELS.md`](docs/MODELS.md); one page per task is in
[`docs/guides/`](docs/guides/).

## Where it is careful

- **Placement is measured.** The forward is walked without allocating, each card is asked what
  it holds, and one rule decides for every family. Weights are charged at what the pool keeps,
  not what the files weigh.
- **Out of memory is a bug, not a condition.** What does not fit spreads across cards; only what
  no card holds reaches the host.
- **Energy is reported beside speed**, per request.

## Building

The default build wants a CUDA toolkit and OpenCL headers; without them, build the host path:

```sh
cargo build --release --no-default-features --features cpu
```

Toolchains, features, the daemon's flags and what to do when a build or a load fails are in
[`docs/BUILDING.md`](docs/BUILDING.md).

## Running it on a network

The server **has no authentication and no rate limit unless you configure them**, and its API
can delete models. That suits one person on one machine. Turn both on in `config.toml` before it
listens on anything but localhost: `require_auth`, `api_keys` and `rate_limit_per_minute` under
`[server]`.

## Documentation

[`docs/README.md`](docs/README.md) says what to read in which order. The short list:

- [`docs/guides/`](docs/guides/) - one page per task, with the configuration and one request
- [`docs/MODELS.md`](docs/MODELS.md) - the families served and how a request names them
- [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) - every key of `config.toml` and its default
- [`docs/API.md`](docs/API.md) - every route on the three surfaces
- [`docs/STATUS.md`](docs/STATUS.md) - **what is measured, what is known to be slower, what is
  written but not wired, and what has never run.** Read this one before deciding whether the
  engine suits you
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) - measurements against other engines and the
  protocol that makes the comparison fair, including the cells where this engine loses
- [`docs/CLUSTER.md`](docs/CLUSTER.md) - spreading work across more than one machine

## Contributing and reporting

[`CONTRIBUTING.md`](CONTRIBUTING.md) - what a change should bring with it.
[`SECURITY.md`](SECURITY.md) - reporting a vulnerability, and what the unauthenticated default
assumes.

## History

The git history was rewritten before publication. The project began as experiments early in
2026; what is published starts on 2026-08-30 with the tree as it stood then, laid out as
chapters, and carries the work from that point on. The earlier commits are not part of this
repository.

## Licence

MIT OR Apache-2.0, at your option. Third-party attributions are in [`NOTICE.md`](NOTICE.md).
Model weights are not distributed here and carry their own terms.
