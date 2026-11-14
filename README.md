<p align="center">
  <img src="https://raw.githubusercontent.com/loken-ai/.github/main/brand/png/banner.png" alt="LOKEN" width="680">
</p>

# loken

A local inference server for language and generative-media models, written in Rust.  
One process serves chat, images, audio, speech, transcription and video over an OpenAI-compatible API, an Ollama-compatible API and the Messages API.

```sh
cp config.toml.example config.toml   # the daemon reads this from its working directory
cargo build --release                # target/release/{lokend,loken}
./target/release/lokend serve        # the daemon, on port 11435
./target/release/loken  list         # and the client that talks to it
```

Any OpenAI client works unchanged:

```sh
curl -s localhost:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3:1.7b",
  "messages": [{"role": "user", "content": "Say hello in 6 words."}]
}'
```

Inference stays on the machine. loken needs no account and no API key, and has no hosted fallback.
It reaches the network for 4 things, each configured or asked for:

- to fetch a model you asked for
- to fetch an image a request gives by URL, from a host you listed in `image_fetch_hosts`
- to announce a cluster node you configured
- to send a request or a long prompt to a peer you configured

## What it does

- Chat, completion, embeddings and reranking, with tools, JSON schema and grammar-constrained output.
- Images, video, transcription, speech with named, described or cloned voices, music and sound effects.
- Time-series forecasting.
- Placement of a model across cards and the host, with experts streamed from disk when a mixture does not fit.
- 1 setting, `fidelity`, that trades fidelity to the published model for speed.
- Continuous batching, speculative decoding and KV reuse across conversations.
- Coding agents served by the daemon: `loken launch claude`, `loken launch cline`.
- Replicated serving, sharding across machines, and a prompt read by another instance.
- Energy, times and cache place reported per request.

[The overview](https://loken-ai.github.io/loken/overview/) holds the full table: each capability, its route, its configuration key and its state.

## Expose it on a network

The server has no authentication and no rate limit unless you configure them, and its API can delete models.  
Set `require_auth`, `api_keys` and `rate_limit_per_minute` under `[server]` in `config.toml` before it listens on anything but localhost.

## Documentation

| Page | Read it to |
|---|---|
| [Install](https://loken-ai.github.io/loken/install/) | Build and install loken |
| [Guides](https://loken-ai.github.io/loken/) | Do 1 task, with the smallest configuration and 1 request |
| [Learn](https://loken-ai.github.io/loken/course/) | Learn the fundamentals of inference in 10 lessons |
| [`docs/API.md`](docs/API.md) | Look up a route on the 3 surfaces |
| [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) | Look up a key of `config.toml` and its default |
| [`docs/STATUS.md`](docs/STATUS.md) | See what is measured, what is slower, and what has never run |
| [`ROADMAP.md`](ROADMAP.md) | See the next work |

[`docs/README.md`](docs/README.md) indexes every page.

## Contributing and reporting

- [`CONTRIBUTING.md`](CONTRIBUTING.md): what a change brings with it.
- [`SECURITY.md`](SECURITY.md): reporting a vulnerability, and what the unauthenticated default assumes.

## History

The git history was rewritten before publication.  
The project began as experiments early in 2026.
The published history starts on 2026-08-30 with the tree as it stood then, laid out as chapters, and carries the work from that point on.

## Licence

MIT OR Apache-2.0, at your option.  
[`NOTICE.md`](NOTICE.md) holds the third-party attributions.  
Model weights are not distributed here and carry their own terms.
