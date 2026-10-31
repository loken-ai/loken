# loken

A local inference server for language and generative-media models. One process serves chat, images, audio, speech, transcription and video over an OpenAI-compatible API, an Ollama-compatible API and the Messages API.

> [!NOTE]
> Inference stays on the machine. loken needs no account and no API key, and has no hosted fallback. It reaches the network to fetch a model you asked for, and to announce a cluster node you configured.

## Run it

```sh
lokend serve
loken pull qwen3:1.7b
curl -s localhost:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3:1.7b",
  "messages": [{"role": "user", "content": "Say hello in 6 words."}]
}'
```

Any client written for the OpenAI API, the Ollama API or the Messages API works unchanged. [Install](install.md) covers the binaries, and [First run](first-run.md) covers the first model.

## What it does

<!-- site: widget table id=overview-capabilities -->
| Capability | Interface | State |
|---|---|---|
| Chat and completion, streamed or whole | `/v1/chat/completions`, `/v1/completions`, `/api/chat`, `/api/generate`, `/v1/messages` | Measured, [decode rate](../STATUS.md#decode-rate) |
| Responses API | `/v1/responses` | Runs |
| Coding agents on the daemon | `loken launch claude`, `loken launch cline` | Runs, [guide](../guides/agents.md) |
| Tools, JSON schema, grammar-constrained output | the chat routes, `tool_choice`, `response_format` | Runs |
| Reasoning apart from the answer, Harmony included | the chat routes, `think`, `reasoning_effort` | Runs |
| Vision and speech in a chat turn | the chat routes, image and audio content | Runs |
| Log-probabilities, logit bias, seeds | the chat and completion routes | Runs |
| Embeddings and reranking | `/v1/embeddings`, `/api/embed`, `/v1/rerank` | Runs, [guide](../guides/embeddings.md) |
| Placement across cards and the host | automatic | Measured, [placement](../STATUS.md#placement-measured-separately) |
| Expert spilling for mixtures that do not fit, streamed from disk when the file exceeds the host | automatic | Measured, [large mixtures](../STATUS.md#large-mixtures-streamed-from-disk) |
| Continuous batching | automatic when requests overlap | Runs, cards only |
| Speculative decoding | automatic for a model that spills to the host, `/api/draft/attach` | Measured on 1 pair, [guide](../guides/drafter.md) |
| KV reuse across conversations, on disk, past the window | automatic, `kv_disk_dir` | Runs, [guide](../guides/kv-memory.md) |
| Decode-state cache for the large mixture and recurrent hybrids | automatic in memory, on disk once `state_cache_dir` is set | Runs, [configuration](../CONFIGURATION.md#decode-state-cache-deepseek-v41-and-recurrent-hybrids) |
| Lossy expert routing for speed, within the model's own noise | `fidelity = "fast"` | DeepSeek V4.1 only, off by default, [configuration](../CONFIGURATION.md#streamed-mixtures-deepseek-v41) |
| Files, batches, message batches | `/v1/files`, `/v1/batches`, `/v1/messages/batches`, `files_dir` | Runs |
| Moderation with a classifier | `/v1/moderations`, `moderation_model` | Runs, 5 families |
| Images, edits, variations, adapters | `/v1/images/*`, `/v1/loras`, `lora_dir` | Judged partially, [images](../guides/images.md) |
| Video | `/v1/video/generations` | Runs |
| Transcription and translation | `/v1/audio/transcriptions`, `/v1/audio/translations` | Runs, [guide](../guides/speech.md) |
| Speech, music, sound effects, separation | `/v1/audio/speech`, `/v1/audio/generations`, `/v1/audio/separate` | Runs, [guide](../guides/speech.md) |
| Energy per request | every answer, `[energy]` | Measured |
| Replicated serving across machines | `/api/cluster/*`, `[cluster]` | Runs at 1 node, unmeasured at 2 |
| Sharding one model across machines | automatic at load, `[cluster]` | DeepSeek V4.1 only, unmeasured at 2 nodes |
| Scrape endpoint | `/metrics`, feature `metrics` | Runs |
| Authentication and rate limit | every route, `require_auth`, `api_keys`, `rate_limit_per_minute` | Runs, off by default |
<!-- site: end -->

Each media family is a build feature: `image`, `audio`, `video`, `midi`, or `media` for all. A family that is not built reports its models as unknown. [Models](../MODELS.md) lists the families served.

## Where it is careful

- **Placement is measured.** loken walks the forward pass without allocating, asks each card what it holds, and applies 1 rule to every family. Weights are charged at what the pool keeps, not at what the files weigh.
- **Out of memory is a bug.** What does not fit spreads across cards. Only what no card holds reaches the host.
- **Energy is reported beside speed**, for each request.

[Status](../STATUS.md) lists what is measured, what is slower than other engines, and what has never run. Read it before deciding whether loken suits you.

## Before you expose it

> [!WARNING]
> The server has no authentication and no rate limit until you configure them, and its API can delete models. Set `require_auth`, `api_keys` and `rate_limit_per_minute` under `[server]` before it listens on anything but localhost.

## Next

- [Install](install.md) and [First run](first-run.md)
- [Guides](../guides/openai-sdk.md): 1 page per task
- [Learn](../learn/README.md): the fundamentals of inference in 10 lessons
- [API](../API.md) and [Configuration](../CONFIGURATION.md)
