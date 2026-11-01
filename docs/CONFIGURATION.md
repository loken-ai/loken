# Configuration

The daemon reads `config.toml` from the first of these places:

1. its working directory
2. beside the executable, walking up through a build tree
3. the user configuration directory: `~/.config/loken/config.toml`, or `%APPDATA%\loken\config.toml` on Windows

`config.toml.example` at the root of the repository is a working starting point.

Every key is optional. A key left out takes the default shown.

Logs carry lengths and counts, never a prompt, a message, a transcript or a generated text, at any level.

A component that falls back to the host is admitted against the memory the host can give without swapping, minus a quarter kept as headroom.
Past that, the request is refused with the figures instead of running in swap.
A health probe or a model listing never waits on a load or a render. It reports what was last known.

## Top level

| Key | Default | What it does |
|---|---|---|
| `ollama_models_dir` | `~/.ollama/models` | The Ollama-style store of manifests and blobs. A string, or a list: see [Model directories on several disks](#model-directories-on-several-disks). |
| `huggingface_models_dir` | `~/.cache/huggingface/hub` | The Hugging Face store, as a string or a list. A bare GGUF at its root, or under its `hub/`, is found by file name. |
| `lora_dir` | `loras` beside the first Hugging Face store | Where adapters live. Requests name an adapter, never a path. |

### Model directories on several disks

```toml
ollama_models_dir = ["/mnt/nvme/ollama", "/mnt/hdd/ollama"]
huggingface_models_dir = ["/mnt/nvme/huggingface", "/mnt/hdd/huggingface"]
```

The directories of a key form 1 catalogue: listings and loads see the models of all of them.
List order is the order of preference, so put the fastest disk first.

| Case | What happens |
|---|---|
| A model held by several directories | Listed once, read from the first of them in the list. The log says once which copy is read. |
| A pull | Lands on the directory already holding the model (its manifest or weights blob, its repository) when that directory has room for the bytes it lacks. Otherwise lands on the first directory with room for the whole model. With no room anywhere, it fails and names each directory's free space. |
| A pull that replaces an Ollama model held elsewhere | The manifest in the other directory is removed. Its blobs stay until a delete prunes them. |
| A model built here (`/api/create`, `/api/blobs`, conversions) | Written to the first directory listed. A manifest goes to the first directory holding every blob it names. |
| A copy (`/api/copy`) | Stays in the directory of its source, beside the blobs it shares. |
| A delete | Removes every copy. |

A manifest and every blob it names stay in 1 directory.
A Hugging Face repository stays whole in 1 directory.

A single string reads as a list of 1.
Some media families read checkpoints from fixed paths under the store: `boogu/`, `wan/`, `flux-kontext-q4km.gguf` and `qwen-image-edit/`.
These are looked up in the first Hugging Face directory only.

## `[server]`

<!-- site: widget table id=config-server -->
| Key | Default | What it does |
|---|---|---|
| `host` | `"127.0.0.1"` | Listening address. The address is the security boundary: see [SECURITY.md](../SECURITY.md). |
| `port` | `11435` | Listening port. |
| `require_auth` | `false` | Demand an API key on every request but the health probes (`/health`, `/healthz`, `/api/health`, `/ping`). Turn it on before the daemon listens on anything but localhost. |
| `api_keys` | `[]` | Accepted keys, sent as `Authorization: Bearer <key>` or `X-API-Key: <key>`. Empty with `require_auth = true` refuses every request. |
| `allowed_origins` | `[]` | Browser origins allowed when authentication is on. Empty allows none. |
| `rate_limit_per_minute` | `0` | Requests per minute per client. `0` is unlimited. |
| `rate_limit_burst` | `10` | Requests that may arrive back to back before the per-minute rate applies. |
<!-- site: end -->

## `[inference]`

### Model

| Key | Default | What it does |
|---|---|---|
| `model_id` | `"ollama:qwen3:0.6b"` | The model loaded when a request names none. `loken list` prints the tags on this machine. |

### Fidelity to the published model

| Key | Default | What it does |
|---|---|---|
| `fidelity` | `balanced` | How closely the output follows the published model, against speed and memory: `reference`, `balanced` or `fast`. Any other value is a startup error. The server logs the level and what it chose at start. |

What each level chooses:

<!-- site: widget table id=config-fidelity-levels -->
| Choice | `reference` | `balanced` | `fast` |
|---|---|---|---|
| Float projections of an F32, F16 or BF16 checkpoint | As stored (BF16 held as F16 on a card) | Q8_0 when the stack sits on one kind of memory, Q4_K (Q6_K for the feed-forward output) when it is split between cards and host | Q4_K (Q6_K for the feed-forward output) |
| KV cache format | Full width | Q8 for a model the quantised kernels serve | Q8 for a model the quantised kernels serve |
| Float GEMMs on a CUDA card | IEEE F32 reduction | TF32 and fast BF16 | TF32 and fast BF16 |
| Context shift | Off | On | On |
| Drafts verified in recorded windows | No | No | Yes |
| Exactness the experts streamed from disk may give up | `exact` | `exact` | `fast` |
| File each expert is read from, for a model served from several files | `largest` | `plan` | `smallest` |
<!-- site: end -->

The level decides these choices alone. The file names none of them.
A quantised checkpoint (Q4_K, Q8_0 and the like) is read as stored at every level.
Its products quantise the activation to 8 bits, as llama.cpp does.

The level does not choose these, which are the same at all 3:

| Choice | Behaviour |
|---|---|
| Prompt lookup, and a drafter attached to a model that spills | Every draft is verified by the target's own sampler. A short verify runs the decode step's numerics row by row. |
| A prompt longer than the window | Refused with an error on every route. A cut prompt could lose its system prompt or tool instructions and still read as a plausible answer. |
| Batched serving of overlapping requests | Runs the residual stream in F16 on the cards. No level turns it off. |
| Approximate kernels (fast exp, fused attention in F16 with F32 sums) | Built into the kernels, within float rounding of their precise forms. |

### Decided by loken

None of these is configured.
A key of that name in an older `config.toml` loads and is reported as no longer read.

| Decision | Where it comes from |
|---|---|
| Sampling (`temperature`, `top_p`, `top_k`, repetition penalty, stop sequences) | The request, else the model's own recommendation (the `params` layer of an Ollama manifest, the checkpoint's generation config), else the Ollama API defaults. |
| Seed | The request's, else a fresh one per request. |
| Generation length | The request's cap, else the end of the context window. |
| Context window | The request's `num_ctx`, else what the checkpoint declares. A quantised cache grows into it on demand. An F16 cache is sized by the placement plan. |
| Card a model loads on | The fastest card that holds it, from the probe. |
| Memory each card offers | The measured free memory. The load keeps its own runtime reserve and backs off after an out-of-memory. |
| Layers on cards, host threads | The placement plan, and the physical (or performance) cores. |
| KV cache format | Q8 for every model whose head shape the quantised kernels serve, at `balanced` and `fast`. Full width at `reference`, for a head shape they do not cover, and for capped attention scores (Gemma2), attention sinks and gemma4. |
| Conversation snapshots | Each finished sequence is copied aside, a quantised cache in its packed form, so a later request on it resumes from its own KV. The count kept is bounded by the free device memory, leaving the cache its working window and the next prefill the memory it sizes its chunk from. A copy that would not fit is not taken. |
| Batched serving | A model serves requests 1 at a time until they overlap several times in a row, then batches them in its paged worker (cards only). As many lone arrivals in a row return it to serial decode, the faster path for 1 stream. |
| Speculative decoding | A model that spills layers to the host is drafted for by the smallest model in the store with the same ordinary vocabulary, loaded on the fastest card before the target. The draft length is learned from measured acceptance and cycle time. A cycle that recorded a verify window, or the drafter's step, counts its acceptance alone. The drafter's decode step is recorded where the model allows it and kept between requests while nothing it reads has moved. A model held on the cards gets no drafter. `POST /api/draft/attach` attaches one by hand. |
| Resident models | As many language models stay loaded as the cards hold. The least recently used is unloaded only when an incoming model would otherwise be pushed off the cards. |
| Context shift | At `balanced` and `fast`, once a conversation outgrows the window, the resident tail is kept and re-phased instead of prefilling the whole window, wherever the cache supports it (not Q4 caches, sliding-window layers or recurrent hybrids). |

### Copies of a model

| Key | Default | What it does |
|---|---|---|
| `replicas` | `1` | Copies of a model served side by side, each whole on a card of its own, so concurrent requests run in parallel. A request goes to the least busy copy, or to the one that served the last request while it is free, so a conversation keeps its cache. A model that does not fit 1 card is served once whatever this says. A copy that cannot be placed whole on its card is dropped. A reload for a wider window or another cache format brings the model back as 1 copy until its next cold load. |
| `vulkan_devices` | none | Vulkan devices placement may give layers to, by index or by a part of their name. Empty offers those no other runtime of the build drives. Needs the `vulkan` feature. |

### KV reuse across requests

| Key | Default | What it does |
|---|---|---|
| `kv_disk_dir` | none | Directory of the disk tier under the snapshots. Holds token blocks shared between sequences by prefix, and 1 manifest per sequence. Survives a restart. |

The next turn of a conversation keeps the rows it shares with the turn before, up to the last turn marker of the chat template:

- It reads only the answer and the new turn: a conversation of 7000 tokens reads 30 at its second turn.
- Several conversations are kept side by side, each restored when its turn comes, as long as the lease holds.
- The usage of a reply names the tokens a kept state served (`cached_tokens`), on the chat route and on the Responses route, where `previous_response_id` names the conversation.
- A prompt with no turn marker keeps what a cold read of it would have produced, in whole blocks.

### Prompts read by another instance

An instance with `prefill_peer` set has its long prompts read by that peer, which serves the same model:

1. The peer prefills the prompt and answers its KV cache over `/api/cluster/prefill`.
2. This instance imports it as a snapshot and decodes.
3. The prefix reuse restores it in whole prefill chunks and reads the rest of the prompt here, so a prompt within a chunk of what the cache holds never leaves.

A model with recurrent or windowed layers hands over its whole decode state instead of rows.
That state holds every recurrent state, the windowed layers' rows, the global layers' rows and the logits to resume from.
It is checked against this instance's layer kinds and row geometry before it is read.

The prompt is read here when:

- the model has a split layer
- the cache is cut over the cards
- the peer cannot answer

The rows travel as f32, about 4 times the cache's own size.

| Key | Default | What it does |
|---|---|---|
| `prefill_peer` | none | URL of the instance that reads this one's long prompts. |

### A context cut over the cards

With `context_parallel` set, the cache of each attention layer is cut into ranges of positions, 1 on each card the machine shows: the layer's own card first, then the others.

- Before its prefill, a request reserves the positions it needs, each card's room past the work of a prefill shared by the layers.
- A decode past them grows a range a working window at a time.
- A query attends to every range on the range's card, and the parts are combined on the layer's card, so a context no single card holds is served whole.

The ranges hold F16 rows.
The layers so cut run outside the recorded decode step, and their prefixes are not kept as snapshots.
These layers keep their own cache: layers with a sliding window, attention sinks or indexed attention, layers sharing a cache with others, and layers split across cards.

| Key | Default | What it does |
|---|---|---|
| `context_parallel` | `false` | Cut attention layers' caches over every card. |

### One directory for each disk cache

`kv_disk_dir` and `state_cache_dir` are separate keys because the caches ask different things of their disk.

The KV disk tier writes blocks of 256 tokens, shared between sequences by prefix.
It writes and reads at the speed of its disk and measures nothing.

The decode-state cache writes whole decode states, far larger than a block.
It measures the write speed, the cost per file and the restore time of its disk.
It writes a state only when the disk can drain it within the prefill time of the prompt it came from.
It restores a state only when reading it costs less than prefilling the prompt again.
It sizes its files from the cost per file.

A machine with a fast disk and a large one can put each cache on the disk that suits it, the KV tier on the fast disk and the states on the large one.
Both keys unset keep nothing on disk.

### Decode-state cache (DeepSeek V4.1 and recurrent hybrids)

The cache lives in host memory for the life of the process, or on disk when `state_cache_dir` is set.
A DeepSeek V4.1 prompt costs 1 pass over the routed experts, read from disk.
The cache stores the model's decode state after a prompt and after each finished turn.
A later prompt that begins with the same tokens restores it and prefills only the rest.

Every file a cache keeps on disk (the state cache, the KV disk tier, the expert routing profile) has the same format:

- A 20-byte header names the kind of file and its version, then comes a binary body.
- A body is stored compressed when that makes it clearly smaller, and read back whole. A store in memory is never compressed.
- A file of another kind, another version or cut short is refused, and the cache rebuilds what it lost.
- Files hold floating-point state and token ids, no text, and are not encrypted.

The same keys open the same store for a generic-route model that carries a recurrent state beside its KV (a convolution, state-space or delta-rule mixer interleaved with attention).
Such a state cannot be cut back to a shorter prefix, so trimming never reuses an earlier prompt and every turn re-prefills.
With the store open, each state the prefill passes is kept, and a later prompt restores the longest one it extends.
A dense model keeps its disk tier (`kv_disk_dir`) and is unchanged.

In memory, states are kept as the KV snapshots are: never written, gone with the process.
On disk, the files hold state derived from prompts: KV tensors, hidden rows of the filling compression group and compressed token-class ids.
They hold no text and no token ids, and names and keys are hashes.
The state is as private as the prompt it came from.
Keep the directory where the prompts could live, and leave the key unset where nothing about an exchange may reach the disk.

| Key | Default | What it does |
|---|---|---|
| `state_cache_dir` | none | Directory of the cache. None keeps it in host memory. |

The cache works out these quantities from what the model, the disk and the engine report, and logs when they change:

| Quantity | Derived from |
|---|---|
| Block size | The bytes a token adds to the blocks, the model's granule, and the directory's per-file cost and write throughput. These are measured once on the directory with synced probe files and kept, so a restart finds the chain it wrote. |
| Spacing of kept states | The measured write throughput, the size of a manifest and the measured prefill time per token. A state is kept no closer than the disk can finish writing it within the prefill of the tokens in between. Until both are measured, the spacing is the prefill's own chunk. It is a multiple of the block size and of the chunk. |
| Whether to restore | A stored state is read only while its measured read-and-decode cost per byte, times its bytes, is below the measured prefill time of the tokens it covers. It is tried while either is unmeasured. |
| Writes | 1 at a time. A write queues behind earlier ones only while the disk drains the queue within the prefill time of the prompt that produced the state. Otherwise it is skipped with a log line. |

A DeepSeek V4.1 state costs:

- about 2.7 MB of window per snapshot (the last 128 tokens of every layer, kept as the model keeps them)
- about 1.1 KB per token of compressed KV, stored once in block files that every snapshot of a conversation shares

A prompt restores the longest stored prefix and prefills the rest in 1 layer-major pass onto it, so a stored prefix always pays.
A pass over a few 100 tokens still routes to nearly every expert of every layer, so it reads about as many expert bytes as a whole prompt.
A remainder of 250 tokens costs a little over half a cold 2000-token prompt.
The batch worker and the serial path use the cache.
A pipelined model does not, because half of its state is on the tail node.

A state is keyed by the model files (length, modification time and header), so a different file starts from nothing.
A damaged, truncated or foreign file is ignored with a log line.

For a recurrent hybrid, a stored state is a manifest and blocks:

- The manifest holds every recurrent mixer's state, every windowed layer's resident rows and the global layers' rows past the last whole block. Its size depends on the model, not on the prompt.
- Blocks hold the global layers' rows for 1 block size of tokens (derived, see above), once per shared prefix. A block is stored as f16 where every value is exactly an f16, and as f32 otherwise, so a restore returns the stored bits.
- States are kept at multiples of the largest whole number of blocks that fits a prefill chunk, and at the end of every prompt of at least 1 block.
- A state is keyed by the file, `kv_quant`, the layer kinds, the row geometry and which layers run on a card, so another placement or KV format starts from nothing.

A restore removes the prefill of the covered tokens: the recurrent state and the global KV are read from disk instead of recomputed.
The gain grows with the covered length and with how slow a prefill is on the machine and placement in use.
The read speed of the cache directory against the state size bounds it.
A restore always leaves at least 1 token to prefill, to produce the logits.

### Streamed mixtures (DeepSeek V4.1)

The `fidelity` level decides both choices of a model whose routed experts stream from disk. The file names neither.

At `reference` and `balanced`, the output is identical to the full model, the same tokens whichever experts the cards hold, because a routed expert on a card gives the bits the host gives.
At `fast`, the output differs by small, measured amounts, 3 times the model's own run-to-run noise, the divergence between its exact runs on the card's kernels and on the host's.

For a model whose manifest names further files of its checkpoint ([MODELS.md](MODELS.md)), the level also chooses the file each routed expert is read from.
`balanced` reads the experts the cards keep, hottest first, from the most precise file while the cards have room, and those the host tier keeps or reads from disk from the smallest.

At load, the engine runs a bundled text teacher-forced on the card path and on the host path to measure the noise.
It then runs the text once under each candidate way of giving up experts, and applies the candidate that saves the most disk reads within the level's budget.
With no card and host to compare, `balanced` and `fast` run exact and say why in the log.

The candidates derive from `K`, the number of experts a token routes to.
Each candidate either keeps the `K-1` or `K-2` heaviest experts, or replaces a light expert that sits in no card tier when its weight share is under `0.5/K`, `0.75/K` or `1/K`.
The measured table is cached per model file and plan, and the noise per model file, under `loken/` in the user cache directory, so a different level reuses them.
A file with no tokenizer in its metadata runs exact.

The offload derives these quantities from measurement and logs them:

| Quantity | Derived from |
|---|---|
| Read-ahead depth | A controller that times decode tokens in windows and tries each neighbouring depth for 1 window. It keeps a depth only when its mean token time beats the incumbent's by more than the noise, judged against the incumbent's windows on both sides. Bounded by the hardware and the plan. |
| Reader threads | The smallest count that reaches the device's read rate, from a probe at load. |
| Sightings before a card takes an expert | The measured upload cost against what a use saves. |
| Experts kept per layer | What the host memory allows. |
| Where a prompt's expert runs | The measured time of a copy to the card per byte against the measured time of a host row, per expert of the layer. |

During a prompt's prefill, each layer reads only the experts the prompt routes to, once its router has named them:

- The cards' copies are read from the file into a page-locked slab in runs of adjacent experts, around the page cache.
- Meanwhile the cards run the experts they keep and the host runs its own.
- A copy made for the layer is let go after it unless the card has the room free to keep it. It never takes a kept expert's place.
- When no card has room for the prompt's rows, the whole next layer is read ahead into the slab while the current layer runs.

### Expert cache of a mixture held in the file's mapping

A mixture too large for the cards keeps its routed experts in the file's mapping.
The load sizes the cache on each layer's card.
| Quantity | Derived from |
|---|---|
| Experts a layer's cache holds | The budget the plan gave the card, less what the load took of it, the room for copies and the working memory of a chunk, divided by the bytes of 1 more expert in every layer the card holds. When the budget covers every expert of every layer, they are all on the card and the layers run as loaded ones, with no cache. |
| Which card holds which layers | Cards on equal links are cut in proportion to their room beside the output head. On unequal links, read from the driver as the generation and width of each card's PCIe link, the slower card holds the layers it can hold whole, every expert on the card, and the card on the fastest link takes the rest. |
| Where a missing expert runs | On the card, copied into a transit room and run with the cached ones, or on the host. The split is the one at which both finish together, from the time a copy and a host computation have each taken on the layer. A transited expert hotter than the coldest slot not in use stays in that slot. |
| Which expert a slot holds | Counts of each expert's use, scaled down every few tens of tokens through a layer, so the cache follows the text in hand. A missing expert replaces the coldest slot not in use only when its count is higher by a margin. |
| Bytes page-locked for copies to the cards | 3 quarters of the memory the kernel still has free, within the cap of the process's control group. |
| How long a recorded token waits on the host | 64 times the longest request the card's tier worker has answered, or the longest a layer's widest request takes at the measured paces of a host product and a copy. A request left unanswered that long while the worker does not move releases the card, logs the layer and the worker's step at error level, and fails the request. The layers serve again once the worker answers. |
| A wide batch streams a layer's experts | Only when copying what the card does not hold, at the measured host-to-card rate, takes less than running the rows through the cache at the pace tokens have decoded. |
| Host memory admitted for a spill | The bytes the process owns. Pages viewed in place from the mapping are not counted. The log gives the bytes read from the mapping, with a warning when they exceed the host's memory. |

### Files, moderation and URLs

| Key | Default | What it does |
|---|---|---|
| `files_dir` | `files` under the first Ollama store | Where `/v1/files` keeps uploads, batch inputs and batch outputs. |
| `moderation_model` | none | The classifier `/v1/moderations` asks, by tag. Recognised by name: Llama Guard, Shieldstral, ShieldGemma, Granite Guardian and gpt-oss Safeguard. Without it, the endpoint answers 501. |
| `public_url` | request `Host` | The origin clients reach this server at, used in the URLs it hands out, such as image files answered as `url`. Set it behind a proxy. |

### Media input

Every chat route holds a request to caps before its media reach the model.
The caps follow the decoder's allocation budget: a request past 1 is answered 400 naming the cap and the value it carried.

<!-- site: widget table id=config-media-input -->
| Key | Default | What it does |
|---|---|---|
| `image_fetch_hosts` | none | Hosts, or `host:port` pairs, an image given by URL on the Messages API or `/v1/embeddings` is fetched from. Redirects are followed only to them. Empty fetches nothing. |
<!-- site: end -->

## `[energy]`

<!-- site: widget table id=config-energy -->
| Key | Default | What it does |
|---|---|---|
| `carbon_intensity` | `50` | Grid intensity in gCO2eq per kWh, for the CO2 estimate. |
<!-- site: end -->

## `[cluster]`

Without this table, the node runs alone and every cluster path is skipped.
See [CLUSTER.md](CLUSTER.md) for the model.

<!-- site: widget table id=config-cluster -->
| Key | Default | What it does |
|---|---|---|
| `name` | `"default"` | Cluster name. Nodes announcing another name are ignored. |
| `node_id` | hostname | How this node calls itself. |
| `advertise` | empty | The URL peers reach this node at. Empty stays a client of the others. |
| `join` | `[]` | Seeds for peers multicast cannot reach. |
| `gossip_interval_ms` | `1000` | How often peers exchange state. |
| `min_speedup` | `1.15` | Predicted speedup below which a request is not handed over. |
<!-- site: end -->

## Command line and environment

| Setting | Default | What it does |
|---|---|---|
| `LOKEN_API_KEY` | unset | Read by `loken launch`: the key handed to the agent when the daemon requires one. |
| `OLLAMA_KEEP_ALIVE` | unset | The keep-alive of a loaded model. |
| `HF_TOKEN` | unset | Sent when a pull fetches from Hugging Face. |

## Build features

<!-- site: widget table id=config-build-features -->
| Feature | Default | What it adds |
|---|---|---|
| `cuda`, `opencl` | on | The device backends. |
| `rocm` | off | AMD cards through HIP. Alone with `--no-default-features --features rocm,cpu`, or beside `cuda` with `--features cuda,rocm`, where 1 binary drives both vendors' cards. |
| `image`, `audio`, `video`, `midi`, or `media` for all | on | The generative-media families and their routes. |
| `vulkan` | off | Dense decoder layers on Vulkan GPUs: those no other runtime drives, or those `vulkan_devices` names. |
| `metrics` | off | `GET /metrics` in OpenMetrics, behind the same authentication as every other route. |
| `cpu` | off | Language models on the host only. |
<!-- site: end -->
