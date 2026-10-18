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
| Conversation snapshots | Each finished sequence is copied aside, a quantised cache in its packed form, so a later request on it resumes from its own KV; how many are kept is bounded by the free device memory, leaving the cache its working window and the next prefill the memory it sizes its chunk from, and a copy that would not fit is not taken. |
| Batched serving | A model serves requests one at a time until they overlap several times in a row, then batches them in its paged worker (cards only); as many lone arrivals in a row take it back to serial decode, the faster path for one stream. |
| Speculative decoding | A model that spills layers to the host is drafted for by the smallest model in the store with the same ordinary vocabulary, loaded on the fastest card before the target; the draft length is learned from measured acceptance and cycle time. A model held on the cards gets no drafter. `POST /api/draft/attach` still attaches one by hand. |
| Resident models | As many language models stay loaded as the cards hold; the least recently used is unloaded only when an incoming model would otherwise be pushed off the cards. |
| Context shift | Once a conversation outgrows the window, the resident tail is kept and re-phased instead of prefilling the whole window, wherever the cache supports it (not Q4 caches, sliding-window layers or recurrent hybrids). |

### Copies of a model

| Key | Default | What it does |
|---|---|---|
| `replicas` | `1` | Copies of a model to serve side by side, each whole on a card of its own, so concurrent requests run in parallel. A request goes to the least busy copy, and to the one that served the last request while that one is free, so a conversation keeps its cache. A model that does not fit one card is served once whatever this says, and a copy that cannot be placed whole on its card is dropped. A reload for a wider window or another cache format brings the model back as one copy until its next cold load. |

### KV reuse across requests

| Key | Default | What it does |
|---|---|---|
| `kv_disk_dir` | none | Directory of the disk tier under the snapshots: token blocks shared between sequences by prefix, one manifest per sequence, surviving a restart. |
| `kv_disk_budget_gb` | `0` | Bytes the disk tier may hold, in GiB. `0` is no limit. Least recently used sequences go first. |

### Prompts read by another instance

An instance with `prefill_peer` set has its long prompts read by that peer, which serves the same
model: the peer prefills the prompt and answers its KV cache over `/api/cluster/prefill`, and this
instance imports it as a snapshot and decodes. The prefix reuse restores it in whole prefill
chunks and reads the rest of the prompt here, so a prompt within a chunk of what the cache holds
never leaves. A model with recurrent or windowed layers hands over its whole decode state instead
of rows: every recurrent state, the windowed layers' rows, the global layers' rows and the logits
to resume from, checked against this instance's layer kinds and row geometry before it is read.
A model with a split layer, a cache cut over the cards, or a peer that cannot answer reads the
prompt here. The rows travel as f32, about four times the cache's own size.

| Key | Default | What it does |
|---|---|---|
| `prefill_peer` | none | URL of the instance that reads this one's long prompts. |

### A context cut over the cards

With `context_parallel` set, the cache of each attention layer is cut into ranges of positions,
one on each card the machine shows: the layer's own card first, then the others. Before its
prefill a request reserves the positions it needs, each card's room past the work of a prefill
shared by the layers; a decode past them grows a range a working window at a time. A query attends to every range on the range's card and the
parts are combined on the layer's card, so a context no single card holds is served whole. The
ranges hold F16 rows; the layers so cut run outside the recorded decode step, and their prefixes
are not kept as snapshots. Layers with a sliding window, attention sinks, indexed attention, a
cache other layers share, or split across cards keep their own cache.

| Key | Default | What it does |
|---|---|---|
| `context_parallel` | `false` | Cut attention layers' caches over every card. |

### Decode-state cache (DeepSeek V4.1 and recurrent hybrids)

In host memory for the life of the process, or on disk when `state_cache_dir` is set. A DeepSeek
V4.1 prompt costs one pass over the routed experts, read from disk; this cache stores the model's
decode state after a prompt and after each finished turn, so a later prompt that begins with the
same tokens restores it and prefills only the rest.

The same keys open the same store for a generic-route model that carries a recurrent state beside
its KV (a convolution, state-space or delta-rule mixer interleaved with attention). Such a state
cannot be cut back to a shorter prefix, so no prefix of an earlier prompt can be reused by trimming
and every turn re-prefills; with the store open, each state the prefill passes is kept and a later
prompt restores the longest one it extends. A dense model keeps its disk tier (`kv_disk`) and is
unchanged.

In memory the states are kept as the KV snapshots are: never written, gone with the process. The
files hold state derived from prompts: KV tensors, hidden rows of the filling compression group
and compressed token-class ids. They hold no text and no token ids, and names and keys are hashes,
but the state is as private as the prompt it came from. Keep the directory where the prompts
could live, and leave the key unset where nothing about an exchange may reach the disk.

| Key | Default | What it does |
|---|---|---|
| `state_cache_dir` | none | Directory of the cache; none keeps it in host memory. |
| `state_cache_budget_gb` | derived | Bytes the cache may hold, in GiB. Unset or `0` derives the limit: what the cache holds plus the free space of the directory's filesystem, or in memory the host memory a component may take, less room for one more state of the largest size written. Least recently used first; in each family of states sharing a first block the one restored most, the longest among equals, goes last. |

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

A state is also keyed by the model files (length, modification time and header),
so a different file starts from nothing. A damaged, truncated or foreign file is ignored
with a log line.

For a recurrent hybrid a stored state is a manifest and blocks. The manifest holds every recurrent
mixer's state, every windowed layer's resident rows and the global layers' rows past the last whole
block; its size depends on the model and not on the prompt. Blocks hold the global layers' rows for
one block size of tokens (derived, see above), once per shared prefix, as f16 where every value is exactly an f16
and as f32 otherwise, so a restore returns the stored bits. States are kept at multiples of the
largest whole number of blocks that fits a prefill chunk, and at the end of every prompt of at least
one block. A state is keyed by the file, `kv_quant`, the layer kinds, the row geometry and
which layers run on a card, so another placement or KV format starts from nothing. What a restore
removes is the prefill of the covered tokens: the recurrent state and the global KV are read from
disk instead of recomputed, so the gain grows with the covered length and with how slow a prefill is
on the machine and placement in use, and is bounded by the read speed of the cache directory against
the state size. A restore always leaves at least one token to prefill, to produce the logits.

### Streamed mixtures (DeepSeek V4.1)

`quality` applies to a model whose routed experts stream from disk, and it is the only knob: the
engine measures the rest.

| Key | Default | What it does |
|---|---|---|
| `quality` | `exact` | How much exactness the output may give up for speed. `exact`: identical to the full model. `balanced`: differences no larger than the model's own run-to-run noise, the divergence between its exact runs on the card's kernels and on the host's. `fast`: more speed for small, measured differences, three times that noise. Any other value is a startup error. |

At load the engine runs a bundled text teacher-forced on the card path and on the host path to
measure the noise, then once under each candidate way of giving up experts, and applies the
candidate that saves the most disk reads within the level's budget. With no card and host to
compare, `balanced` and `fast` run exact and say why in the log.

The candidates derive from `K`, the number of experts a token routes to: keep the `K-1` or `K-2`
heaviest, or replace a light expert that sits in no card tier when its weight share is under
`0.5/K`, `0.75/K` or `1/K`. The measured table is cached per model file and plan in
`quality-calibration.json`, and the noise in `quality-noise.json`, under `loken/` in the user cache
directory, so a different level reuses them. A file with no tokenizer in its metadata runs exact.

What the offload derives for itself, from measurement, and logs:

| Quantity | Derived from |
|---|---|
| Read-ahead depth | A controller that times decode tokens in windows, tries each neighbouring depth for one window, and keeps it only when its mean token time beats the incumbent's by more than the noise, judged against the incumbent's windows on both sides. Bounded by the hardware and the plan. |
| Reader threads | The smallest count that reaches the device's read rate, from a probe at load. |
| Sightings before a card takes an expert | The measured upload cost against what a use saves. |
| Experts kept per layer | What the host memory allows. |

During a prompt's prefill, the experts of the next layer are read from the file into one of two
page-locked slabs while the dense work of the current layer runs.

### Expert cache of a mixture held in the file's mapping

A mixture too large for the cards keeps its routed experts in the file's mapping, and the cache
on each layer's card is sized by the load. Two keys carry what a run learned to the next one:

| Key | Default | What it does |
|---|---|---|
| `expert_profile` | `profiles/` of the model store | A routing profile written by `expert_profile_save`; at the load it fills the slots the prefill does not borrow with the most routed experts, and gives every expert the heat its count earns. A missing file is skipped. With neither key set, each model reads its own profile from `profiles/` of the model store, named after the model id. |
| `expert_profile_save` | `profiles/` of the model store | Where the routing counts are written after each request: those of the profile read at the load plus the decodes since, halved together when one would pass the 32-bit limit. With neither key set, each model writes its own profile where it reads it. |


| Quantity | Derived from |
|---|---|
| Experts a layer's cache holds | The budget the plan gave the card, less what the load took of it, the room for copies and the working memory of a chunk, divided by the bytes of one more expert in every layer the card holds. When the budget covers every expert of every layer, they are all on the card and the layers run as loaded ones, with no cache |
| Which card holds which layers | Cards on equal links are cut in proportion to their room beside the output head. On unequal links, read from the driver as the generation and width of each card's PCIe link, the slower card holds the layers it can hold whole, every expert on the card, and the card on the fastest link takes the rest |
| Where a missing expert runs | On the card, copied into a transit room and run with the cached ones, or on the host. The split is the one at which both finish together, from the time a copy and a host computation have each taken on the layer. A transited expert hotter than the coldest slot not in use stays in that slot |
| Which expert a slot holds | Counts of each expert's use, scaled down every few tens of tokens through a layer, so the cache follows the text in hand. A missing expert replaces the coldest slot not in use only when its count is higher by a margin |
| Bytes page-locked for copies to the cards | Three quarters of the memory the kernel still has free, within the cap of the process's control group |
| How long a recorded token waits on the host | 64 times the longest request the card's tier worker has answered, or the longest a layer's widest request takes at the measured paces of a host product and a copy. A request left unanswered that long while the worker does not move releases the card, logs the layer and the worker's step at error level, and fails the request; the layers serve again once the worker answers |
| A wide batch streams a layer's experts | Only when copying what the card does not hold, at the host-to-card rate measured, takes less than running the rows through the cache at the pace tokens have decoded |
| Host memory admitted for a spill | The bytes the process owns; pages viewed in place from the mapping are not counted, and the log gives the bytes read from the mapping with a warning when they exceed the host's memory |

### Files, moderation and URLs

| Key | Default | What it does |
|---|---|---|
| `files_dir` | `files` beside the Ollama store | Where `/v1/files` keeps uploads, batch inputs and batch outputs. |
| `moderation_model` | none | The classifier `/v1/moderations` asks, by tag. Recognised by name: Llama Guard, Shieldstral, ShieldGemma, Granite Guardian and gpt-oss Safeguard. Without it the endpoint answers 501. |
| `public_url` | request `Host` | The origin clients reach this server at, for the URLs it hands out, such as image files answered as `url`. Set it behind a proxy. |

### Media input

Caps every chat route holds a request to before its media reach the model. A request past one
is answered 400 naming the key and the value it carried.

| Key | Default | What it does |
|---|---|---|
| `max_images` | 16, the most one message carries | Images one request hands the vision encoder. |
| `max_image_pixels` | 134217728, the decoder's 512 MiB budget at RGBA8 | Pixels (width times height) an image may declare in its header, checked before it is decoded. An image fetched by URL may also be at most four bytes per allowed pixel. |
| `max_audio_seconds` | 1398, the same budget for a stereo clip at 48 kHz in f32 | Seconds of the audio clip, read from the container's header where it states them and from the decoded frames otherwise. |
| `image_fetch_hosts` | none | Hosts, or `host:port` pairs, an image given by URL on the Messages API is fetched from; redirects are followed only to them. Empty fetches nothing. |
| `image_fetch_timeout_secs` | 30 | Seconds a fetch of an image by URL may take, connection to last byte. |

## `[cpu]`

| Key | Default | Meaning |
|---|---|---|
| `level` | the best the CPU has | Highest SIMD level the host kernels may use: `generic` (scalar reference code), `avx2` or `avx512`. It only lowers the level, never raises it above what the CPU supports. Used to compare paths and for hosts where a wider one measures slower. |

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

## Command line and environment

| Setting | Default | What it does |
|---|---|---|
| `LOKEN_API_KEY` | unset | Read by `loken launch`: the key handed to the agent when the daemon requires one. |
| `OLLAMA_MODELS`, `OLLAMA_KEEP_ALIVE` | | The Ollama store, overriding `ollama_models_dir`; the keep-alive of a loaded model. |
| `HF_TOKEN` | unset | Sent when a pull fetches from Hugging Face. |

## Build features

| Feature | Default | What it adds |
|---|---|---|
| `cuda`, `opencl` | on | The device backends. |
| `image`, `audio`, `video`, `midi`, or `media` for all | on | The generative-media families and their routes. |
| `metrics` | off | `GET /metrics` in OpenMetrics, behind the same authentication as every other route. |
| `cpu` | off | Language models on the host only. |
