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

### Model and sampling

| Key | Default | What it does |
|---|---|---|
| `model_id` | required | The model loaded when a request names none. `loken list` prints the tags on this machine. |
| `model_source` | detected | `"ollama"` or `"huggingface"`, when a name exists in both stores. |
| `max_tokens` | `2048` | Generation cap when a request sets none. |
| `context_length` | `4096` | The window a model is loaded with. A request whose `num_ctx` exceeds it reloads the model at that context, clamped to what the checkpoint declares. |
| `temperature`, `top_p`, `top_k`, `seed` | `0.15`, `0.9`, `50`, `42` | Sampling defaults, each overridable per request. |

### Placement

| Key | Default | What it does |
|---|---|---|
| `device_index` | first card | The card a single-device model loads on. |
| `max_gpu_memory_fraction` | `0.9` | Share of each card's memory the placement budget may plan against. |
| `use_quantized_gpu` | `true` | Run quantized matmuls on the cards. Off sends quantized layers to the host. |
| `force_gpu_layers` | planned | Fix the number of layers placed on cards instead of measuring what fits. |
| `cpu_threads` | `0` | Threads for the host path. `0` derives the count from the physical cores. |
| `kv_quant` | `"off"` | KV cache storage: `"off"`, `"q8"` or `"q4"`. Read at load time only. |
| `continuous_batching` | `false` | Paged KV and batched decode for several concurrent clients. Cards only. |
| `disable_arc_layers` | `false` | Test switch: skip the OpenCL path so a run can be pinned to CUDA or the host. |

### Speculative decoding

| Key | Default | What it does |
|---|---|---|
| `draft_model` | none | A smaller model, by tag, that proposes tokens the loaded model verifies. Pays where verification is far dearer than drafting, such as a model spilling to the host. |
| `draft_device_index` | `0` | The card the drafter loads on, whole. |

`POST /api/draft/attach` does the same for one session without a restart.

### KV reuse across requests

| Key | Default | What it does |
|---|---|---|
| `kv_shift_reuse` | `false` | Once a conversation outgrows the window, keep the resident tail and re-phase it in place instead of prefilling the whole window. Refused, with a cold prefill, on Q4 caches, sliding-window layers and recurrent hybrids. |
| `kv_snapshots` | `0` | Resident sequences copied aside after a request, so a later request on another conversation resumes from its own KV. Each entry costs one KV where the layers live. `0` keeps the single resident KV. |
| `kv_snapshot_budget_gb` | `0` | Device memory the snapshots may hold, in GiB, least recently used evicted first. `0` derives it from the free memory at snapshot time, leaving the cache its working window. |
| `kv_disk_dir` | none | Directory of the disk tier under the snapshots: token blocks shared between sequences by prefix, one manifest per sequence, surviving a restart. Needs `kv_snapshots > 0`. |
| `kv_disk_budget_gb` | `0` | Bytes the disk tier may hold, in GiB. `0` is no limit. Least recently used sequences go first. |

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
