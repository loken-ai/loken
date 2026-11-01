# Troubleshoot

Each table maps a message of the daemon, `lokend`, to its cause and fix.  
Messages are quoted as printed, and `...` stands for varying text.

## Before you start

<!-- site: widget checklist id=operate-troubleshoot-prereq -->
- [ ] You can read the output of `lokend serve`.
  Add `--verbose` for more lines.
- [ ] You know the name of the model and the address of the daemon, `localhost:11435` by default.
<!-- site: end -->

## Daemon will not start

| Symptom | Cause | Fix |
|---|---|---|
| `error: this build needs an x86-64-v3 CPU; this one lacks ...` | The CPU lacks AVX2, FMA, F16C, BMI1, BMI2, LZCNT or MOVBE. The message names the missing ones. | Run on a CPU that has them. |
| `Address already in use` | Another process listens on the port. | Run `lokend serve --port <other>`, or stop the other process. |
| `Failed to load config.toml: config.toml not found in CWD, exe dir, or user config dir. Using defaults.` | No `config.toml` in the working directory, beside the executable or in the user config directory. | Copy `config.toml.example` to `config.toml` in the working directory. |
| `Failed to load config.toml: Failed to parse config file: ...` | `config.toml` is not valid TOML, or a key has the wrong type. The daemon runs on defaults. | Fix the line the message names. |
| `CUDA libraries not found: ...` | The CUDA runtime is not on the library path. Every layer runs on the host. | Install the CUDA runtime, or ignore the message on a machine without a card. |
| `nvcc not found at ...; set CUDA_PATH` at build time | The build cannot find the CUDA toolkit. | Set `CUDA_PATH` to the toolkit, or build without CUDA ([Building](https://github.com/loken-ai/loken/blob/main/docs/BUILDING.md)). |

## A model will not load or does not fit

| Symptom | Cause | Fix |
|---|---|---|
| `404` with `Model not found` | Neither the Ollama store nor the Hugging Face store holds the name. | `loken pull <name> --source ollama` or `--source huggingface`. A Hugging Face id needs its owner. |
| `404` with `Model '...' not found in remote registry` | The registry chosen by `--source` does not know the name. | Check the tag on the registry, or try the other source. |
| `404` with `Model '...' not found. Pull it first with POST /api/pull.` | The forecast route found no such model on this machine. | Pull the model. |
| `404` with `Model '...' could not be loaded: ...` | The files exist and the load failed. The text after the colon names the reason. | Read the reason. Run `loken pull` again when a file is missing or damaged. |
| `503` with `Model '...' exists but the machine has no room for it right now: ... Retry once the current work finishes.` | The cards and the host are full with other models or work. | Wait for the work to finish, or unload a model. |
| `400` with `the prompt is N tokens and the model's context window is M; shorten the conversation or load the model with a larger num_ctx` | The prompt exceeds the window the model loaded with. | Shorten the prompt, or send a larger `num_ctx`. |
| `400` with `` `model` '...' contains invalid characters `` | The name holds a space, a control character, a backslash, `..`, or starts with `/` or `.`. | Use the name `GET /api/tags` lists. |
| The log says `...: placement failed (...); trying the host` | The cards could not take the model. It loads on the host, which decodes slower. | Free card memory, or pick a smaller quantisation. |

## Out of memory

| Symptom | Cause | Fix |
|---|---|---|
| `alloc failed: out of memory` | A card ran out after the placement had spread the model. | Lower the `num_ctx` of the request, or pick a smaller quantisation. |
| The log says `...: out of card memory; retrying at pressure level N` | A step ran out of card memory. The daemon frees its pools and retries. | No action when the request then succeeds. Lower `num_ctx` when it repeats. |
| The log says `...: out of card memory; retrying at N` | The decode chunk of an audio model did not fit. The daemon halves the chunk and retries. | No action when the request then succeeds. |
| The log says `...prefill: out of memory at N (...); retrying at M` | A prompt chunk did not fit. The daemon halves the chunk and goes on from the chunk that failed. A model with recurrent layers, or whose cache cannot be cut back, reads the prompt again from where the reading began. | No action when the request then succeeds. |
| `503` with `Model '...' exists but the machine has no room for it right now` | The load ran out of memory after every fallback of the placement. | Wait for other work to finish, or unload a model. |
| The log says `...: not whole on one card, so served once` | `replicas` is above 1 and the model does not fit 1 card. | Lower `replicas` to 1, or use a smaller quantisation. |

## Slow decode

| Symptom | Cause | Fix |
|---|---|---|
| The first answer after a load is slow | The load, the repack of a mixture of experts, and on a fresh node the CUDA kernel compilation are paid once. | Send it again. |
| The log says `...: placement failed (...); trying the host` | Layers run on the host. | Free card memory, shrink `num_ctx`, or use a smaller quantisation. |
| `CUDA libraries not found: ...` | Every layer runs on the host. | Install the CUDA runtime. |
| The log says `stays off tensor parallelism on N cards: ...` | A layer of the model cannot be split across N cards. | No action. The reason follows the colon. |
| The log says `stays layer by layer on N cards: ...` | Splitting every layer is predicted slower than whole layers, for a token or for a prompt chunk. | No action. |
| The daemon was started with `--cpu` | `--cpu` places every layer on the host. | Start without `--cpu`. |

## Requests refused

| Symptom | Cause | Fix |
|---|---|---|
| `401` with `missing or invalid API key` | `require_auth` is on and the key is absent or unknown. | Send `Authorization: Bearer <key>` or `X-API-Key: <key>` with a key from `api_keys`. |
| `429` with `rate limit exceeded` | The client used its burst and its rate. | Wait the seconds of the `Retry-After` header, or raise `rate_limit_per_minute`. |
| `503` with `no inference slot became available within ...s (...); retry later` | Every slot of the model stayed busy for the whole wait. | Retry later, or raise `replicas` when a second copy fits. |
| `409` with `this node is holding a conversation and has nowhere to place work that yields` | A request of batch priority found no peer to take it. | Send it without the batch priority. |
| `404` with `Endpoint not found: ...` | The path is not a route of the daemon. The body lists the routes. | Fix the path ([API](../API.md)). |

Authentication and rate limits: [Expose loken on a network](network.md).

## Cluster peers not found

| Symptom | Cause | Fix |
|---|---|---|
| `/api/cluster/peers` answers `"clustered": false` | The `[cluster]` section is absent. | Add it to `config.toml` and restart. |
| The peers list holds only this node | The peer is unreachable, runs another `name`, or answers `401`. | Open the port, give both nodes the same `name`, turn `require_auth` off on both. |
| The log says `cluster: no multicast discovery (...); seeds only` | The network offers no multicast. | List the peers in `join`. |
| The log says `cluster: serving locally: single node` | `join` is empty on this node. | Put the address of 1 peer in `join`. |
| The log says `cluster: serving locally: no local price for this model yet` | The node has no measured rate for this model. | Nothing. The first request stays local and becomes the measurement. |
| The log says `cluster: serving locally: no peer published anything usable` | No usable node: no peer has published its state, or every peer is excluded. | Check `/api/cluster/peers`, then the `advertise` address of each peer. |
| The log says `cluster: serving locally: no address known for ...` | The best peer has no address on record. | Read the `endpoint` of the peer in `/api/cluster/peers`, then set `join` or `advertise`. |
| The log says `cluster: serving locally: peer ... is only ...x faster, below the ...x a hand-over has to earn` | The peer is not faster than `min_speedup` requires. | Lower `min_speedup`, or accept that this node serves the request. |
| The log says `cluster: hand-over to ... failed (...) - serving here instead` | The peer did not answer the forwarded request. | Check the peer. The failed peer is not chosen again until its penalty expires. |
| The log says `cluster auto-tune ...: not enough measured nodes to plan a split` | Too few nodes published a cost for the model. | Load the model on each node, or wait for the next gossip rounds. |
| The log says `cluster auto-tune ...: pipeline not formed (...); one node` | A tail node refused its share. | Read the reason. The tail needs the model file locally. |
