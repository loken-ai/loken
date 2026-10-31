# Keep conversations in memory between requests

A request that continues the previous one reuses the resident KV cache and prefills only its new tokens. 2 more tiers extend that to several conversations and across a restart.

## Before you start

<!-- site: widget checklist id=kv-memory-prereq -->
- [ ] Requests of one conversation repeat the earlier turns as a prefix.
- [ ] For the disk tier, a directory the daemon can write.
<!-- site: end -->

## Smallest configuration

The snapshot tier needs no configuration. The disk tier needs a directory:

```toml
[inference]
kv_disk_dir = "/var/lib/loken/kv"
kv_disk_budget_gb = 20
```

Keys are described in [the configuration page](../CONFIGURATION.md#kv-reuse-across-requests).

## Tiers

| Tier | Switch | What it does |
|---|---|---|
| Snapshots | automatic | After a request, the resident KV is copied aside under its tokens. A prompt that shares more of a snapshot than of the resident KV makes that snapshot resident and prefills the rest. The free device memory bounds the number kept, and the cache keeps its working window. |
| Disk | `kv_disk_dir` | Full token blocks are written to disk off the request path. Conversations share them by prefix. A later request or a later run of the daemon reads them back. |
| Past the window | `fidelity` | Once a conversation outgrows the context, the resident tail is kept and re-phased instead of re-prefilled. |

Past the window applies at `fidelity = "balanced"` (the default) and `"fast"`.
The re-phased keys differ from a cold run's in the last bits.
Q4 caches, sliding-window layers and recurrent hybrids prefill cold.
At `fidelity = "reference"` the tail is re-prefilled cold.

## Run it

<!-- site: widget steps id=kv-memory-run -->
1. Send a first turn.

   ```sh
   curl -s localhost:11435/api/chat -H 'Content-Type: application/json' -d '{"model": "MODEL", "stream": false, "messages": [
     {"role": "user", "content": "Hello"}]}'
   ```

2. Send the same messages, the answer and a new user turn.

   ```sh
   curl -s localhost:11435/api/chat -H 'Content-Type: application/json' -d '{"model": "MODEL", "stream": false, "messages": [
     {"role": "user", "content": "Hello"},
     {"role": "assistant", "content": "ANSWER"},
     {"role": "user", "content": "And then?"}]}'
   ```
<!-- site: end -->

## Expected result

`prompt_eval_count` counts the tokens actually prefilled. The second request shows only its new tokens.
The Messages API reports the cache counts in `usage`.
A snapshot costs 1 KV, held where the layers live.
A quantised cache restores on its own grid: Q8 exactly, Q4 within a step.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| 400 `the prompt is N tokens and the model's context window is M` | The prompt is longer than the window. Past the window does not apply to a single prompt. | Shorten the conversation, or load the model with a larger `num_ctx`. |

## Next

- [Configuration](../CONFIGURATION.md#kv-reuse-across-requests)
- [The KV cache lesson](../learn/06-kv-cache.md)
