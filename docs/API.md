# API

One process, three surfaces: the Ollama API under `/api`, the OpenAI API under `/v1`, and
the Messages API under `/v1/messages`. A client written for any of the three works unchanged.
Bodies are read as JSON whatever `Content-Type` says, and every surface answers a malformed
body in its own error envelope.

Every route but `/health` and `/ping` demands a key when `[server] require_auth` is on, as
`Authorization: Bearer <key>` or `X-API-Key: <key>`.

## Ollama surface

| Route | What it does |
|---|---|
| `GET /` | Liveness, as Ollama answers it. |
| `GET /api/version` | The Ollama API level served, with the crate version beside it. |
| `GET /api/tags` | The models on this machine, with the architecture, quantisation and parameter count each header declares, read once before the port opens and again only for a file that changed. In a cluster, the models the peers hold follow, each under `node`; a request naming one is forwarded there. |
| `POST /api/show` | A model's details, with the Modelfile synthesised from its own blobs. `verbose` adds the vocabulary arrays. |
| `GET /api/ps` | The models resident in memory. |
| `POST /api/pull` | Fetch a model. Streams progress unless `stream: false`. |
| `POST /api/push` | Refused: nothing is pushed to a registry. |
| `POST /api/create` | Build a model from blobs uploaded through `/api/blobs`, with template, system, parameters, license and messages as layers of their own. Streams NDJSON. |
| `HEAD /api/blobs/{digest}`, `POST /api/blobs/{digest}` | Check for and upload a blob. |
| `POST /api/copy` | Copy a manifest under a new name, blobs shared. |
| `DELETE /api/delete` | Remove a manifest and every blob no manifest names. |
| `POST /api/chat` | Chat. Takes `tools`, `tool_choice`, `think`, `format`, `logprobs`, `top_logprobs`, images, and `audios` (base64 clips) on a user message for a model with the `audio` capability. Returns `thinking` apart from `content`, `done_reason`, and the engine's timings. |
| `POST /api/generate` | Completion. Renders `system` and `template`, takes `suffix`, returns `context`. |
| `POST /api/embed` | Embeddings. Honours `dimensions` and `truncate`. |
| `POST /api/embeddings` | The legacy one-prompt shape. |

## OpenAI surface

### Language

| Route | What it does |
|---|---|
| `POST /v1/chat/completions` | Chat. Takes content parts, images and `input_audio` parts (`{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}`, any container the server decodes) for a model with the `audio` capability, `tools`, `tool_choice`, `parallel_tool_calls`, `response_format`, `reasoning_effort`, `n`, `logprobs`, `logit_bias`, `max_completion_tokens`, and vLLM's extensions (below). Returns `reasoning_content` apart from `content`, `finish_reason`, and in `usage` the reasoning tokens, the speculation's `accepted_prediction_tokens` and `rejected_prediction_tokens`, and the [`loken` object](#per-request-figures). A stop sequence is not part of the text. |
| `POST /v1/chat/completions/batch` | vLLM's batch of conversations: `messages` is a list of conversations, each answered as its own chat completion, one choice per conversation. No `stream`, `tools` or `n` above 1. |
| `POST /v1/completions` | Text completion, with `suffix` as fill-in-the-middle, `echo`, an array of prompts, `logit_bias`, `response_format`, and vLLM's extensions (below). `prompt_embeds` sends the prompt as embedding rows of the model's width instead of text, in one of the forms under [Prompt embeddings](#prompt-embeddings); not served while the model batches concurrent requests. |
| `POST /v1/responses` | The Responses API, lowered to a chat completion and lifted back into output items. `GET` and `DELETE /v1/responses/{id}` read and drop a stored response; `previous_response_id` continues one. `input_image` (`image_url` as a base64 data URL, or `file_id` of an upload) and `input_audio` (`data`, `format`) parts are ingested as the `image_url` and `input_audio` parts of a chat completion; another part type answers 400 naming it. |
| `POST /v1/embeddings` | Embeddings, loading the model on demand, as `float` or `base64` (`encoding_format`), cut to `dimensions`. `input` is a string, strings, or one conversation; `messages` is one conversation or a list of them, as vLLM takes them. A conversation is one vector: its text parts joined by newlines, or its one `image_url` part (a base64 data URL, or an http(s) URL from a host in `image_fetch_hosts`) for a CLIP or SigLIP model; text beside an image is refused. A CLIP or SigLIP checkpoint is named by its Hugging Face id, e.g. `openai/clip-vit-large-patch14`; see [the embeddings guide](guides/embeddings.md). |
| `POST /v1/rerank`, `POST /rerank` | Score documents against a query with a cross-encoder. |
| `POST /v2/rerank` | The same in Cohere's shape: `documents` as strings or `{text}` objects, `top_n`, `return_documents`; answers `{id, results, meta}`. |
| `POST /v2/embed` | Embeddings in Cohere's shape: exactly one of `texts`, `images` (data URIs) or `inputs` (`content` lists of `text` and `image_url` parts), `output_dimension`, answered as `embeddings.float` and `embeddings.base64` as `embedding_types` asks (`float` by default). Images need a CLIP or SigLIP model. |
| `POST /score`, `POST /v1/score` | vLLM's pair scoring: `text_1` against `text_2`, one score per pair, `data[].score`. |
| `POST /pooling` | vLLM's pooling shape over the embeddings route: `data[].data` is the pooled vector. |
| `POST /tokenize` | vLLM's tokenizer route: a `prompt`, or chat `messages` rendered by the model's template (`add_generation_prompt`, `continue_final_message`, `chat_template_kwargs`, `tools`), with `add_special_tokens` and `return_token_strs`. Answers `count`, `max_model_len`, `tokens`, `token_strs`. |
| `POST /detokenize` | `tokens` back to `prompt`, special tokens kept. |
| `POST /invocations` | SageMaker's route, as vLLM serves it: a body with `messages` goes to chat, one with `prompt` to completions, one with `input` to embeddings. |
| `POST /v1/forecast` | Continue a time series with a forecaster checkpoint: `model`, `series` (numbers, `null` for a missing point), `horizon` (steps, up to what the checkpoint answers), optional `quantiles` (levels, read between the checkpoint's own by a straight line). A checkpoint that reads covariates also takes `past_covariates` (rows as long as `series`), `known_covariates` (rows covering `series` and the horizon), and their `categorical_` forms, whose values are any strings or numbers; each field is an array of rows or an object of named rows. Answers `median` and `quantiles[].{level, values}`, one value per future step. Runs on a card when one has room, else on the host. |
| `GET /v1/models` | The models on this machine, with `max_model_len`, the widest window each takes, then those the peers hold with the holder as `owned_by`. A request is served with the window its prompt and `max_tokens` need, up to that one; a window that must grow for it at least doubles. Answers in the Messages API's shape to a client sending its version header. |
| `GET /v1/models/{id}`, `DELETE /v1/models/{id}` | One model, ids with slashes included. |
| `POST /v1/moderations` | Category scores from the configured classifier, with the family's own labels under `hazards`. 501 until `moderation_model` is set. |

### Prompt embeddings

`prompt_embeds` on `/v1/completions` takes a base64 string, an array of base64 strings (one
prompt each, answered as an array of prompts), or an array of rows of numbers. A base64 payload
is read by its first bytes:

| Form | First bytes | Width | Dtypes |
|---|---|---|---|
| loken's format | `00 00 C0 FF 4C 4B 45 4D` | in the header | float32, float16, bfloat16 |
| `torch.save` of one tensor, as vLLM clients send it | `PK 03 04` | the tensor's last dimension | float32, float16, bfloat16, float64 |
| bare little-endian f32, as `/v1/embeddings` returns them | anything else | the model's | float32 |

Every form is refused with a 400 naming the fault, before a value is converted, when the rows
are not the model's width or fill the served context window; and after, when a value is not
finite.

#### loken's format

Little-endian throughout:

| Offset | Bytes | Field |
|---|---|---|
| 0 | 8 | magic `00 00 C0 FF 4C 4B 45 4D`; its first four bytes are an f32 NaN, which a bare f32 payload may not carry, so the two never collide |
| 8 | 2 | version, `1` |
| 10 | 1 | dtype: `0` float32, `1` float16, `2` bfloat16 |
| 11 | 1 | reserved, `0` |
| 12 | 4 | rows (u32) |
| 16 | 4 | width (u32) |
| 20 | 4 | reserved, `0` |
| 24 | rows x width x dtype size | values, row after row |
| end - 4 | 4 | CRC-32 of every byte before it (zlib's `crc32`) |

A payload whose length is not exactly 28 bytes plus the values is refused as truncated or
padded; one whose CRC-32 does not match, header included, as corrupted. Half precision takes half
the bytes of f32: a 256 MiB request body carries about 24,000 rows of width 4096 in float16 or
bfloat16, 12,000 in float32. A request body of raw bytes is not offered: base64 costs a third
more bytes and one decode pass, small next to the prefill of those rows.

```python
import struct, zlib
import numpy as np

def encode_loken(rows: np.ndarray) -> bytes:  # float32 or float16, shape (rows, width)
    code = {np.dtype("<f4"): 0, np.dtype("<f2"): 1}[rows.dtype.newbyteorder("<")]
    head = b"\x00\x00\xc0\xffLKEM" + struct.pack("<HBBIII", 1, code, 0, *rows.shape, 0)
    body = head + np.ascontiguousarray(rows, rows.dtype.newbyteorder("<")).tobytes()
    return body + struct.pack("<I", zlib.crc32(body))
```

bfloat16 is the top half of each float32: `(rows.astype("<f4").view("<u4") >> 16).astype("<u2")`
with code `2`.

#### torch.save

The archive `torch.save(tensor, buffer)` writes, base64-encoded: shape `(tokens, hidden_size)`
or `(1, tokens, hidden_size)`, any strides and storage offset, on any device. It is read without
running the pickle: only the opcodes, globals (`torch._utils._rebuild_tensor_v2`,
`collections.OrderedDict`, `torch.FloatStorage`, `HalfStorage`, `BFloat16Storage`,
`DoubleStorage`) and persistent references of one plain tensor are accepted, every entry used is
checked against its CRC-32, and anything else is refused. Compressed or big-endian archives,
torch's legacy format, sparse and quantized tensors, and tensors carrying attributes or hooks are
refused.

### vLLM's extensions

`/v1/completions` and `/v1/chat/completions` read the fields vLLM's server adds to OpenAI's.

| Field | What it does |
|---|---|
| `repetition_penalty` | A multiplicative penalty, in (0, 4], on every token already in the prompt and the answer. It wins over `frequency_penalty` and `presence_penalty`. |
| `min_p` | Drops the tokens less probable than this share of the most probable one. |
| `allowed_token_ids` | The only tokens that may be drawn. |
| `bad_words` | Words never written: the last token of each is banned after the tokens before it, at the start of a text and after a space. |
| `min_tokens` | Tokens written before an end token, a stop token or a stop sequence may end the answer. |
| `ignore_eos` | The model's end tokens and its default stop sequences end nothing; the answer runs to `max_tokens`. |
| `stop_token_ids` | Tokens that end the answer besides the model's own. |
| `include_stop_str_in_output` | Keeps the stop sequence the answer ended on in the text. |
| `skip_special_tokens` | `false` keeps the special tokens drawn in the text. |
| `truncate_prompt_tokens`, `truncation_side` | Reads only this many prompt tokens, the last ones (`left`, the default) or the first ones (`right`); `-1` is the model's window. |
| `return_tokens_as_token_ids` | Writes each log-probability token as `token_id:<id>`. |
| `structured_outputs` | One constraint: `json` (a schema, as an object or a string), `regex`, `choice` (a list of strings), `grammar` (Lark, or llama.cpp's GBNF), `json_object`; with `disable_any_whitespace`, `disable_additional_properties` and `whitespace_pattern` for a JSON one. `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar` and `guided_whitespace_pattern` are read the same way. `structural_tag` answers 400. |
| `cache_salt` | The request reuses no cached prompt prefix and leaves none for another. |
| `use_beam_search`, `length_penalty`, `early_stopping` | Answers the `n` best beams of a beam search of width `n`, best first (see below). |
| `echo` (chat) | Writes the final assistant message in front of the answer. |
| `add_generation_prompt` (chat) | `false` renders the conversation without opening the assistant's turn, with a Jinja template. |
| `continue_final_message` (chat) | The answer continues the final assistant message instead of opening a new turn. |
| `chat_template_kwargs`, `documents` (chat) | Variables handed to the model's template; `enable_thinking` is also the thinking preference. |
| `include_reasoning` (chat) | `false` leaves the reasoning out of the answer. |

`min_p`, `allowed_token_ids`, `bad_words`, `min_tokens`, `skip_special_tokens: false`,
`cache_salt` and a structured output decode one token per step, without prompt-lookup drafts,
captured graphs or prefix reuse. `prompt_logprobs`, a request
`chat_template` and `continue_final_message` with `add_generation_prompt: true` answer 400.
`priority`, `request_id`, `spaces_between_special_tokens`,
`add_special_tokens`, `stream_interval` and the disaggregation fields are accepted without effect.

#### Beam search

`use_beam_search: true` scores and stops as transformers' `generate(num_beams=n,
length_penalty=..., early_stopping=...)` does:

- Each step ranks every continuation of the running beams by cumulative log-probability, read
  from the raw logits. Of the first `n`, those ending on an end token, and on the last step all
  of them, finish. The first `n` that do not end run on.
- A finished beam scores its cumulative log-probability over its generated length, the end
  token counted, raised to `length_penalty` (default 1).
- `early_stopping`: `false` (default) stops once `n` beams are finished and the best running
  one cannot beat the worst of them at its current length; `true` stops once `n` are finished;
  `"never"` scores the running beam at `max_tokens` when `length_penalty` is positive.
- Ties rank the earlier beam, then the lower token id, first.

The choices are the beams, best first. The text leaves out the end token; `logprobs` carry
every generated token, the end token included, so they sum to the beam's cumulative
log-probability. The model's end tokens and `stop_token_ids` end a beam; `ignore_eos` runs every
beam to `max_tokens`. Temperature, the sampling knobs, penalties, stop sequences and token rules
are not read. `stream`, `prompt_embeds`, `best_of` above `n`, a structured output and a required
tool call answer 400. A model whose KV cache cannot be cut back to a shorter sequence (a
recurrent or sliding-window layer, a backend other than the generic transformer) answers 500.

vLLM's own beam search differs from transformers': it divides by the prompt and generated
length without the end token, finishes every end token among a beam's best `2n` candidates and
always runs to `max_tokens`.

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
| `POST /v1/audio/generations` | `audio` | Music and sound effects from a description, as wav, mp3 or flac by `response_format` (wav when absent; another name answers 400). A MIDI model answers a `.mid` file whose tempo is `bpm`, 20 to 300, 120 when absent; a tempo out of range answers 400. |
| `POST /conversation` | `audio` | One turn of a conversation, routed by what the prompt asks for: chat, vision on an attached image, image generation, speech or sound. Rules decide the clear cases and a small classifier model the rest; the reply carries `route`, `routed_by` and the picture or clip beside the message. Takes `stream` for events, `model` as the chat model to prefer, `image_model` and `tts_model` to name the renderers. |
| `POST /voice` | `audio` | Speech in, speech out: transcription, a chat turn, then the reply spoken. |
| `POST /v1/images/generations`, `/edits`, `/variations` | `image` | Image generation. `response_format: url` stores the picture as a file and answers its URL on `public_url` or the request's `Host`. |
| `GET /v1/loras` | `image` | The adapters this server can apply, by name. |
| `POST /v1/video/generations` | `video` | Text- and image-conditioned clips. `POST /api/video/plan` says what a clip of a given length will cost before rendering it. |
| `GET /v1/renders`, `POST /v1/renders/{id}/cancel` | `image` | Renders in flight, and cancelling one. |

A streamed render on any of these routes sends events with `status`, `phase`, `phase_label`, `step`, `total` and `elapsed_ms`, and every event carries `node`: the name of the node that renders, so a client behind a hand-over sees where its work is, or null when the server runs alone.

### Media input limits

Every chat route (`/api/chat`, `/api/generate`, `/v1/chat/completions`, `/v1/responses`,
`/v1/messages`) holds a request to `max_images` images, `max_image_pixels` pixels per image and
`max_audio_seconds` seconds of audio, set under `[inference]` in
[CONFIGURATION.md](CONFIGURATION.md#media-input). A request past one is answered 400 with the
cap's key and the value it carried, before the image is decoded.

## Messages API surface

| Route | What it does |
|---|---|
| `POST /v1/messages` | Messages. Takes content blocks and images, `tools`, `tool_choice`, `disable_parallel_tool_use`, `thinking`, `output_format`, `top_k`, `seed`, `stop_sequences`. Returns a thinking block before the text, `stop_reason` with the sequence matched, and cache counts and the [`loken` object](#per-request-figures) in `usage`. Streams as the official event stream, with `error` events. |
| `POST /v1/messages/count_tokens` | Count the prompt with the model's own tokenizer. |
| `POST /v1/messages/batches`, `GET /v1/messages/batches/{id}` | Batches in the Messages API's shape. `/cancel` stops one; `/results` serves its JSONL. |

A 401, 429 or 404 under `/v1/messages` carries the Messages API envelope, and the request id
comes back in the `request-id` header. An image block with a URL source
(`{"type": "image", "source": {"type": "url", "url": ...}}`) is fetched from a host in
`image_fetch_hosts` and held to the media caps; a host outside it is answered 400 naming the host.
A part the server cannot take, such as a document block, is refused with the reason.

## Per-request figures

A chat answer on `/v1/chat/completions` and `/v1/messages` reports its own figures in a `loken`
object inside its final usage:

| Route | Where |
|---|---|
| `/v1/chat/completions` | `usage` of a whole answer. The usage chunk of a stream, sent when `stream_options.include_usage` is set. |
| `/v1/messages` | `usage` of a whole answer. `usage` of the `message_delta` event of a stream. |

```json
"loken": {
  "queue_ms": 1.2, "prefill_ms": 840.5, "decode_ms": 2210.0,
  "energy": {"joules": 95.3, "gpu_joules": 80.1, "cpu_joules": 15.2, "estimated": false, "share": "work"},
  "kv": {"used_tokens": 1830, "capacity_tokens": 32768}
}
```

| Field | Meaning |
|---|---|
| `queue_ms` | From the request's arrival at the model to the start of its prompt. |
| `prefill_ms` | Reading the prompt, to the first token. |
| `decode_ms` | Writing the answer after the first token. |
| `energy.joules` | The domains read, summed. |
| `energy.gpu_joules` | From the GPU energy counter. Absent when no counter is read. |
| `energy.cpu_joules` | CPU package and DRAM. Read from RAPL, or the configured `cpu_tdp_w` over the time when RAPL is not readable. `null` when neither reads it. |
| `energy.estimated` | `true` when the CPU part is not read from RAPL. |
| `energy.share` | `alone` when no other request was active during the request. `work` when others were. |
| `kv.used_tokens` | The request's sequence in the KV cache when it ended. |
| `kv.capacity_tokens` | The tokens the cache of the loaded model holds. |

The share rule: the energy meter is read at every start and end of a request. Each reading
between two of them is split among the requests active over it by the tokens each computed
over it, its prompt read and its answer generated. A reading over which no request computed a
token is split evenly. A request's energy is the sum of its parts.

A field loken does not measure is left out. `energy` is absent when `[energy] enabled` is off or no domain is read.
`kv` is absent when the backend does not report its cache. An answer that gathers several
generations (`n` above 1, a beam search) carries no `loken` object.

`completion_tokens_details.accepted_prediction_tokens` counts the drafted tokens the model kept
and `rejected_prediction_tokens` the ones it refused. Both stay at zero on a model served by
continuous batching, whose worker does not report its speculation.

A streamed answer whose prompt is still being read after 250 ms reports the reading as an SSE
comment line about every 250 ms until the first token, on both routes:

```
: loken-prefill {"done": 2048, "total": 8192}
```

`done` is the prompt tokens read so far and `total` the tokens to read, the cached prefix left
out. A prompt read in under 250 ms sends no comment line. Neither does a model served by
continuous batching, whose worker reads several prompts at once. A client that follows the SSE
specification ignores comment lines.

## Server, models and cluster

| Route | What it does |
|---|---|
| `GET /health` | Uptime and time. Exempt from authentication, as `GET /ping` and `POST /ping` (an empty 200, for SageMaker and vLLM clients). |
| `GET /version` | The server's version, in vLLM's shape. |
| `GET /load` | Requests running and queued, as vLLM's `server_load`. |
| `POST /reset_prefix_cache` | Every loaded language model drops its session records, its resident KV and the logits of its last prefill. The KV snapshots, the disk tier and the batch worker's prefix cache stay. |
| `GET /metrics` | OpenMetrics, with the `metrics` feature only: uptime, resident models, gate depth, in-flight generations, decode rates, devices, energy totals, and per model a histogram over requests of the wait at the gate, the wait for the model, the prompt's reading, the answer's writing, the prompt and answer tokens and the prompt tokens served from the cache. |
| `GET /api/models`, `GET /api/models/loaded` | The catalogue, and what is resident with its placement per device. Every family lists its layers by device: a text or image model as one entry, a voice, transcription, sound or separation model and what the video pipeline stages on the host as one entry per part, named `model (part)`. A render in progress is listed the same way, one entry per part it has loaded, with a `status` of the form `rendering sound: codes 12/300`; before its first part loads, the render is one entry with the status alone. The free memory published beside these is what the driver has: what a finished job freed is returned to it when the job ends. |
| `POST /api/models/validate`, `POST /api/models/repair` | Check a model's files, and fetch what is missing. |
| `POST /api/swap` | Replace the resident model. |
| `POST /api/layers/swap` | Attach or detach adapters on a loaded model without reloading it. The body is the set the model should end up with. |
| `POST /api/draft/attach`, `POST /api/draft/detach`, `GET /api/draft/status` | Speculative decoding for the resident model, after a tokenizer compatibility check. |
| `GET /api/inflight` | Requests in flight and queued, by priority. The priority comes from the request: a completion with a `suffix` is an editor filling in the middle and goes first, the lines of the batch APIs yield the cards to whoever is waiting, the rest is interactive. |
| `GET /api/layer_perf`, `GET /api/stage_perf` | Where a decode step's time goes, per layer and per stage. Both are recorded only while measurement is on: `?enable=1` on either switches it on and zeroes the counters, `?enable=0` switches it off, and each reply says whether it is on. It costs a device synchronisation per stage, which is why it is off by default. A layer's figure is what issuing it costs the calling thread, which is what tells a layer on a card from one on the host. A streamed decode also times each token's iteration and its parts (`token: prepare`, `token: launch`, `token: sample`, `token: stream`) under `offload_stages`. |
| `GET /api/distributed/devices` | The compute devices and their state. |
| `GET /api/distributed/stats`, `GET /api/distributed/recommend` | What this node runs with, and the largest model in the catalogue that fits what is free. |
| `GET /api/multi-device/status` | How the resident model is spread across devices. |
| `GET /api/cluster/state`, `GET /api/cluster/peers` | What a node publishes about itself, and who it sees. |
| `POST /api/cluster/prefix` | How much of a prompt this node already holds in its KV. |
