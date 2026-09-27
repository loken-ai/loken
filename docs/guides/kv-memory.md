# Keep conversations in memory between requests

A request that continues the previous one reuses the resident KV and prefills only its new
tokens. Two more tiers extend that to several conversations and across a restart.

## Snapshots

Automatic. After a request the resident KV is copied aside under its tokens. A prompt that
shares more of a snapshot than of the resident KV makes that snapshot resident and prefills
the rest. How many are kept is bounded by the free device memory, leaving the cache its
working window.

## Disk

```toml
[inference]
kv_disk_dir = "/var/lib/loken/kv"
kv_disk_budget_gb = 20
```

Full token blocks are written to disk, shared between conversations by prefix, and read back
by a later request or a later run of the daemon. Writing happens off the request path.

## Past the window

Automatic. Once a conversation outgrows the context, the resident tail is kept and
re-phased instead of re-prefilled. The re-phased keys are not the cold run's bit for bit.
Refused on Q4 caches, sliding-window layers and recurrent hybrids, which prefill cold.

## What to expect

`prompt_eval_count` counts the tokens actually prefilled, so a served prefix shows there; the
Messages API reports the cache counts in `usage`. Snapshots cost one KV each where the layers
live, and a quantised cache restores on its own grid: Q8 exactly, Q4 within a step.
