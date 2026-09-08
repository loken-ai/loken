# API

One process, three surfaces: the Ollama API under `/api`, the OpenAI API under `/v1`, and
the Messages API under `/v1/messages`. A client written for any of the three works unchanged.
Bodies are read as JSON whatever `Content-Type` says, and every surface answers a malformed
body in its own error envelope.

Every route but `/health` demands a key when `[server] require_auth` is on, as
`Authorization: Bearer <key>` or `X-API-Key: <key>`.

## Ollama surface

| Route | What it does |
|---|---|
| `GET /` | Liveness, as Ollama answers it. |
| `GET /api/version` | The Ollama API level served, with the crate version beside it. |
| `GET /api/tags` | The models on this machine, with the architecture, quantisation and parameter count each header declares. In a cluster, the models the peers hold follow, each under `node`; a request naming one is forwarded there. |
| `POST /api/show` | A model's details, with the Modelfile synthesised from its own blobs. `verbose` adds the vocabulary arrays. |
| `GET /api/ps` | The models resident in memory. |
| `POST /api/pull` | Fetch a model. Streams progress unless `stream: false`. |
| `POST /api/push` | Refused: nothing is pushed to a registry. |
| `POST /api/create` | Build a model from blobs uploaded through `/api/blobs`, with template, system, parameters, license and messages as layers of their own. Streams NDJSON. |
| `HEAD /api/blobs/{digest}`, `POST /api/blobs/{digest}` | Check for and upload a blob. |
| `POST /api/copy` | Copy a manifest under a new name, blobs shared. |
| `DELETE /api/delete` | Remove a manifest and every blob no manifest names. |
| `POST /api/chat` | Chat. Takes `tools`, `tool_choice`, `think`, `format`, `logprobs`, `top_logprobs`, images. Returns `thinking` apart from `content`, `done_reason`, and the engine's timings. |
| `POST /api/generate` | Completion. Renders `system` and `template`, takes `suffix`, returns `context`. |
| `POST /api/embed` | Embeddings. Honours `dimensions` and `truncate`. |
| `POST /api/embeddings` | The legacy one-prompt shape. |

## OpenAI surface

### Language

| Route | What it does |
|---|---|
| `POST /v1/chat/completions` | Chat. Takes content parts and images, `tools`, `tool_choice`, `parallel_tool_calls`, `response_format`, `reasoning_effort`, `n`, `logprobs`, `logit_bias`, `max_completion_tokens`. Returns `reasoning_content` apart from `content`, `finish_reason`, and reasoning tokens in `usage`. |
| `POST /v1/completions` | Text completion, with `suffix` as fill-in-the-middle, `echo`, and an array of prompts. |
| `POST /v1/responses` | The Responses API, lowered to a chat completion and lifted back into output items. `GET` and `DELETE /v1/responses/{id}` read and drop a stored response; `previous_response_id` continues one. |
| `POST /v1/embeddings` | Embeddings, loading the model on demand. |
| `POST /v1/rerank`, `POST /rerank` | Score documents against a query with a cross-encoder. |
| `GET /v1/models` | The models on this machine, then those the peers hold with the holder as `owned_by`. Answers in the Messages API's shape to a client sending its version header. |
| `GET /v1/models/{id}`, `DELETE /v1/models/{id}` | One model, ids with slashes included. |
| `POST /v1/moderations` | Category scores from the configured classifier, with the family's own labels under `hazards`. 501 until `moderation_model` is set. |

### Files and batches

| Route | What it does |
|---|---|
| `POST /v1/files`, `GET /v1/files` | Upload and list. `GET`, `DELETE /v1/files/{id}` and `GET /v1/files/{id}/content` read one. |
| `POST /v1/batches`, `GET /v1/batches` | Run a JSONL of requests through the handlers a client would call, answers written to an output file. `GET /v1/batches/{id}` and `POST /v1/batches/{id}/cancel` follow one. |

### Media

Each block exists when its Cargo feature is compiled; a family that is not reports its
models as unknown.

| Route | Feature | What it does |
|---|---|---|
| `POST /v1/audio/transcriptions`, `POST /v1/audio/translations` | `audio` | Speech to text. |
| `POST /v1/audio/speech` | `audio` | Text to speech, as wav or mp3, whole or streamed per sentence. `GET /v1/audio/voices` lists the voices. |
| `POST /v1/audio/separate` | `audio` | Split a mix into stems. |
| `POST /v1/audio/generations` | `audio` | Music and sound effects from a description. |
| `POST /voice`, `POST /conversation` | `audio` | Speech in, speech out, through the chat model. |
| `POST /v1/images/generations`, `/edits`, `/variations` | `image` | Image generation. `response_format: url` stores the picture as a file and answers its URL on `public_url` or the request's `Host`. |
| `GET /v1/loras` | `image` | The adapters this server can apply, by name. |
| `POST /v1/video/generations` | `video` | Text- and image-conditioned clips. `POST /api/video/plan` says what a clip of a given length will cost before rendering it. |
| `GET /v1/renders`, `POST /v1/renders/{id}/cancel` | `image` | Renders in flight, and cancelling one. |

## Messages API surface

| Route | What it does |
|---|---|
| `POST /v1/messages` | Messages. Takes content blocks and images, `tools`, `tool_choice`, `disable_parallel_tool_use`, `thinking`, `output_format`, `top_k`, `seed`, `stop_sequences`. Returns a thinking block before the text, `stop_reason` with the sequence matched, and cache counts in `usage`. Streams as the official event stream, with `error` events. |
| `POST /v1/messages/count_tokens` | Count the prompt with the model's own tokenizer. |
| `POST /v1/messages/batches`, `GET /v1/messages/batches/{id}` | Batches in the Messages API's shape. `/cancel` stops one; `/results` serves its JSONL. |

A 401, 429 or 404 under `/v1/messages` carries the Messages API envelope, and the request id
comes back in the `request-id` header. A part the server cannot take, such as a document
block or an image by URL, is refused with the reason.

## Server, models and cluster

| Route | What it does |
|---|---|
| `GET /health` | Uptime and time. The one route exempt from authentication. |
| `GET /metrics` | OpenMetrics, with the `metrics` feature only: uptime, resident models, gate depth, in-flight generations, decode rates, devices, energy totals. |
| `GET /api/models`, `GET /api/models/loaded` | The catalogue, and what is resident with its placement per device. |
| `POST /api/models/validate`, `POST /api/models/repair` | Check a model's files, and fetch what is missing. |
| `POST /api/swap` | Replace the resident model. |
| `POST /api/layers/swap` | Attach or detach adapters on a loaded model without reloading it. The body is the set the model should end up with. |
| `POST /api/draft/attach`, `POST /api/draft/detach`, `GET /api/draft/status` | Speculative decoding for the resident model, after a tokenizer compatibility check. |
| `GET /api/inflight` | Requests in flight and queued, by priority. |
| `GET /api/layer_perf`, `GET /api/stage_perf` | Where a decode step's time goes, per layer and per stage. Stage timing is switched on with `?enable=1` and costs a device synchronisation per stage. |
| `GET /api/distributed/devices` | The compute devices and their state. |
| `GET /api/distributed/stats`, `GET /api/distributed/recommend` | What this node runs with, and the largest model in the catalogue that fits what is free. |
| `GET /api/multi-device/status` | How the resident model is spread across devices. |
| `GET /api/cluster/state`, `GET /api/cluster/peers` | What a node publishes about itself, and who it sees. |
| `POST /api/cluster/prefix` | How much of a prompt this node already holds in its KV. |
