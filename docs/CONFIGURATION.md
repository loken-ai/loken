# Configuration

The daemon reads `config.toml` from its working directory, then from beside the executable
(walking up through a build tree), then from the user configuration directory:
`~/.config/loken/config.toml`, or `%APPDATA%\loken\config.toml` on Windows.
`config.toml.example` at the root of the repository is a working starting point.

Every key below is optional except `[inference] model_id`. A key left out takes the default
shown.

Logs carry lengths and counts, never a prompt, a message, a transcript or a generated text,
at any level.

A component that falls back to the host is admitted against the memory the host can give
without swapping, a quarter of it kept as headroom. Past that, the request is refused with the
figures rather than run in swap. A health probe or a model listing never waits on a load or a
render: it reports what was last known.

## Top level

| Key | Default | What it does |
|---|---|---|
| `ollama_models_dir` | `~/.ollama/models` | The Ollama-style store: manifests and blobs. `OLLAMA_MODELS` in the environment overrides it. |
| `huggingface_models_dir` | `~/.cache/huggingface/hub` | The Hugging Face store. A bare GGUF placed at its root, or under its `hub/`, is found by file name. |
| `lora_dir` | `loras` beside the Ollama store | Where adapters live. Requests name an adapter, never a path. |

## `[server]`

| Key | Default | What it does |
|---|---|---|
| `host` | `"127.0.0.1"` | Listening address. The address is the security boundary: see `SECURITY.md`. |
| `port` | `11435` | Listening port. |
| `require_auth` | `false` | Demand an API key on every request but `/health`. Turn it on before the daemon listens on anything but localhost. |
| `api_keys` | `[]` | Accepted keys. Sent as `Authorization: Bearer <key>` or `X-API-Key: <key>`. Empty with `require_auth = true` refuses every request. |
| `allowed_origins` | `[]` | Browser origins allowed when authentication is on. Empty means none. |
| `rate_limit_per_minute` | `0` | Requests per minute per client. `0` is unlimited. |
| `rate_limit_burst` | `10` | Requests that may arrive back to back before the per-minute rate applies. |

## `[inference]`

### Model

| Key | Default | What it does |
|---|---|---|
| `model_id` | required | The model loaded when a request names none. `loken list` prints the tags on this machine. |

### Decided by loken

Nothing below is configured; a key of that name in an older `config.toml` loads and is reported as no longer read.

| Decision | Where it comes from |
|---|---|
| Sampling (`temperature`, `top_p`, `top_k`, repetition penalty, stop sequences) | The request, else the model's own recommendation (the `params` layer of an Ollama manifest, `generation_config.json` of a checkpoint), else the Ollama API defaults. |
| Seed | The request's, else a fresh one per request. |
| Generation length | The request's cap, else the end of the context window. |
| Context window | The request's `num_ctx`, else what the checkpoint declares; a quantised cache grows into it on demand, an F16 cache is sized by the placement plan. |
| Card a model loads on | The fastest card that holds it, from the probe. |
| Memory each card offers | The measured free memory; the load keeps its own runtime reserve and backs off after an out-of-memory. |
| Layers on cards, host threads | The placement plan, and the physical (or performance) cores. |
| KV cache format | Q8 for every model whose head shape the quantised kernels serve, Q4 for qwen3moe; full width for a head shape they do not cover, capped attention scores (Gemma2) and gemma4. A request may still ask for one with `options.kv_quant`. |
| Conversation snapshots | Each finished sequence is copied aside so a later request on it resumes from its own KV; how many are kept is bounded by the free device memory, leaving the cache its working window. |
| Batched serving | A model serves requests one at a time until they overlap several times in a row, then batches them in its paged worker (cards only); as many lone arrivals in a row take it back to serial decode, the faster path for one stream. |
| Speculative decoding | A model that spills layers to the host is drafted for by the smallest model in the store with the same ordinary vocabulary, loaded on the fastest card before the target; the draft length is learned from measured acceptance and cycle time. A model held on the cards gets no drafter. `POST /api/draft/attach` still attaches one by hand. |
| Resident models | As many language models stay loaded as the cards hold; the least recently used is unloaded only when an incoming model would otherwise be pushed off the cards. |
| Context shift | Once a conversation outgrows the window, the resident tail is kept and re-phased instead of prefilling the whole window, wherever the cache supports it (not Q4 caches, sliding-window layers or recurrent hybrids). |

### KV reuse across requests

| Key | Default | What it does |
|---|---|---|
| `kv_disk_dir` | none | Directory of the disk tier under the snapshots: token blocks shared between sequences by prefix, one manifest per sequence, surviving a restart. |
| `kv_disk_budget_gb` | `0` | Bytes the disk tier may hold, in GiB. `0` is no limit. Least recently used sequences go first. |

### Decode-state cache (DeepSeek V4.1 and recurrent hybrids)

Off unless `state_cache_dir` is set. A DeepSeek V4.1 prompt costs one pass over the routed experts,
read from disk; this cache stores the model's decode state after a prompt and after each finished
turn, so a later prompt that begins with the same tokens restores it and prefills only the rest.

The same keys open the same store for a generic-route model that carries a recurrent state beside
its KV (a convolution, state-space or delta-rule mixer interleaved with attention). Such a state
cannot be cut back to a shorter prefix, so no prefix of an earlier prompt can be reused by trimming
and every turn re-prefills; with the store open, each state the prefill passes is kept and a later
prompt restores the longest one it extends. A dense model keeps its disk tier (`kv_disk`) and is
unchanged.

The files hold state derived from prompts: KV tensors, hidden rows of the filling compression group
and compressed token-class ids. They hold no text and no token ids, and names and keys are hashes,
but the state is as private as the prompt it came from. Keep the directory where the prompts
could live, and leave the key unset where nothing about an exchange may reach the disk.

| Key | Default | What it does |
|---|---|---|
| `state_cache_dir` | none | Directory of the cache. |
| `state_cache_budget_gb` | derived | Bytes the cache may hold, in GiB. Unset or `0` derives the limit: what the cache holds plus the free space of the directory's filesystem, less room for one more state of the largest size written. Least recently used first; in each family of states sharing a first block the one restored most, the longest among equals, goes last. |
| `state_cache_block_tokens` | derived | Tokens per block of the hash chain. Unset or `0` derives one per model, a multiple of what the model's state divides into whole groups by, large enough that a block's transfer time covers the directory's measured per-file cost (see below). Fixing it applies to every model and orphans what is stored. |
| `state_cache_salt` | empty | Mixed into every state's identity; states written under another salt are never read. |

What the cache works out for itself, from what the model, the disk and the engine report, and logs when it changes:

| Quantity | Derived from |
|---|---|
| Block size | The bytes a token adds to the blocks, the model's granule, and the directory's per-file cost and write throughput, measured once on the directory with synced probe files; kept in `geometry.json` so a restart finds the chain it wrote. |
| Spacing of kept states | The measured write throughput, the size of a manifest and the measured prefill time per token: a state is kept no closer than the disk can finish writing it within the prefill of the tokens in between. It is the prefill's own chunk until both are measured, and a multiple of the block size and of the chunk. |
| Whether to restore | A stored state is read only while its measured read-and-decode cost per byte, times its bytes, is below the measured prefill time of the tokens it covers; tried while either is unmeasured. |
| Writes | One at a time, queued behind earlier ones only while the disk drains the queue within the prefill time of the prompt that produced the state; otherwise skipped with a log line. |

A DeepSeek V4.1 state costs about 2.7 MB of window per snapshot (the last 128 tokens of every layer, kept as the
model keeps them) and about 1.1 KB per token of compressed KV, stored once in block files that every
snapshot of a conversation shares. A prompt restores the longest stored prefix and prefills the rest
in one layer-major pass onto it, so a stored prefix always pays. A pass over a few hundred tokens still
routes to nearly every expert of every layer and so reads about as many expert bytes as a whole prompt:
a remainder of 250 tokens costs a little over half a cold 2000-token prompt. The batch worker and the serial path use the cache; a pipelined model does not, because half of its state is on
the tail node.

A state is also keyed by the model files (length, modification time and header) and `expert_deferral`,
so a different file or setting starts from nothing. A damaged, truncated or foreign file is ignored
with a log line.

For a recurrent hybrid a stored state is a manifest and blocks. The manifest holds every recurrent
mixer's state, every windowed layer's resident rows and the global layers' rows past the last whole
block; its size depends on the model and not on the prompt. Blocks hold the global layers' rows for
`state_cache_block_tokens` tokens, once per shared prefix, as f16 where every value is exactly an f16
and as f32 otherwise, so a restore returns the stored bits. States are kept at multiples of the
largest whole number of blocks that fits a prefill chunk, and at the end of every prompt of at least
one block. A state is keyed by the file, `kv_quant`, the salt, the layer kinds, the row geometry and
which layers run on a card, so another placement or KV format starts from nothing. What a restore
removes is the prefill of the covered tokens: the recurrent state and the global KV are read from
disk instead of recomputed, so the gain grows with the covered length and with how slow a prefill is
on the machine and placement in use, and is bounded by the read speed of the cache directory against
the state size. A restore always leaves at least one token to prefill, to produce the logits.

### Files, moderation and URLs

| Key | Default | What it does |
|---|---|---|
| `files_dir` | `files` beside the Ollama store | Where `/v1/files` keeps uploads, batch inputs and batch outputs. |
| `moderation_model` | none | The classifier `/v1/moderations` asks, by tag. Recognised by name: Llama Guard, Shieldstral, ShieldGemma, Granite Guardian and gpt-oss Safeguard. Without it the endpoint answers 501. |
| `public_url` | request `Host` | The origin clients reach this server at, for the URLs it hands out, such as image files answered as `url`. Set it behind a proxy. |

## `[energy]`

| Key | Default | What it does |
|---|---|---|
| `enabled` | `true` | Measure and report energy on every request, from the card's own counter. |
| `carbon_intensity` | `50` | Grid intensity in gCO2eq per kWh, for the CO2 estimate. |
| `cpu_tdp_w` | `0` | Package TDP used to estimate host energy when RAPL is unreadable. `0` reports measured domains only. |
| `water_l_per_kwh` | `1.8` | Water footprint of generation, in litres per kWh. |

## `[cluster]`

Absent, the node runs alone and every cluster path is skipped. `docs/CLUSTER.md` explains the
model; the keys are:

| Key | Default | What it does |
|---|---|---|
| `name` | `"default"` | Nodes announcing another name are ignored. |
| `node_id` | hostname | How this node calls itself. |
| `advertise` | empty | The URL peers reach this node at. Empty stays a client of the others. |
| `join` | `[]` | Seeds for peers multicast cannot reach. |
| `gossip_interval_ms` | `1000` | How often peers exchange state. |
| `min_speedup` | `1.15` | Refuse to hand a request over below this predicted speedup. |

## Build features

| Feature | Default | What it adds |
|---|---|---|
| `cuda`, `opencl` | on | The device backends. |
| `image`, `audio`, `video`, `midi`, or `media` for all | on | The generative-media families and their routes. |
| `metrics` | off | `GET /metrics` in OpenMetrics, behind the same authentication as every other route. |
| `cpu` | off | Language models on the host only. |
