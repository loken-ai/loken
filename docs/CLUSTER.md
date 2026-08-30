# Clustering

Serving one model across many machines: high throughput, tolerant of a node
dying mid-generation, and - like the single-node placement rule it extends - **measured, not
declared**. No operator is asked what their network is. The cluster finds out.

## Where it stands today

**Replicated serving works end to end.** A node discovers its peers, gossips what it holds and
what it measures, and either serves a request or forwards it whole and relays the stream back.

**What works today:** peer discovery and membership, failure detection with eviction, whole-request
forwarding with the response streamed back to the client, cache-aware node selection, deterministic
recovery when a node dies mid-generation, the binary transport between nodes, the tiered KV pool,
and cluster-wide energy reporting.

**What does not work:** serving a model that no single node can hold. Cutting a model into
pipeline stages across nodes has a planner but no data path - nothing executes such a plan. If
the model does not fit on at least one node, the cluster cannot serve it.

**One module to stay away from.** `DistributedEngine`, in `src/inference/serve/distributed_engine.rs`,
is a scaffold: it exposes an API for distributing execution across nodes that is not implemented.
No generation goes through it. Two status endpoints do - `/api/distributed/stats` and
`/api/distributed/recommend` build one per call to read a device inventory - so treat those two
as reporting, not as a path to build on. The collective operations beneath it (`all_reduce`,
`all_gather`, `all_to_all`, `barrier`) are exact for a single rank and return an explicit error
above one, rather than a plausible-looking result nobody computed.

What runs: `discovery.rs` (multicast announce + seed union), `membership.rs` (phi-accrual, with
an `acceptable_pause_ms` a metronomic node cannot trip), `routing.rs` (completion-time estimate
per node), `cluster.rs` + `cluster_runtime.rs` (the decision, the gossip loop, the hand-over),
`rate_meter.rs` (this node's own measured rates, published to peers), `wire.rs` / `wire_link.rs`
/ `wire_mux.rs` (binary framed transport, multiplexed), `link_matrix.rs` (per-link cost via
NVML, so it reads the same on Linux and Windows), `cut_plan.rs`, `kv_tiers.rs`, `replay.rs`
(position-seeded sampling), `evidence.rs`.

Two properties worth stating because they are what make a cluster usable rather than merely
present. A node holding **no weights at all** serves anything the cluster can serve - it
forwards. And a peer that dies stops receiving requests without any request hanging on it: the
detector evicts it and the arithmetic simply stops finding it.

The single-node substrate underneath is sound and the cluster is built **on** it, not beside
it: `continuous_serve.rs` (continuous batching over paged KV with CUDA graphs),
`tp_decode.rs` (TP=2, validated bit-exact across the PCIe pair), `dry_plan.rs` (placement
measured by walking the forward without allocating).

## Running two nodes

Each node needs a `[cluster]` block in its own `config.toml`; everything else it learns.

```toml
[cluster]
name = "home"                       # nodes announcing another name are ignored
node_id = ""                        # empty = the hostname
advertise = "http://192.0.2.10:11435"   # how peers reach THIS node
join = []                           # seeds, for peers multicast cannot reach
```

`advertise` is the one field that cannot be derived: a machine has several addresses and only
the operator knows which one peers should use. Leave it empty and the node still uses the
cluster without being usable by it - correct behind NAT, wrong if you meant to share the card.

On one network nothing else is needed: the nodes hear each other. Across subnets, put each
other's addresses in `join`; discovery adds to that list rather than replacing it.

Two things to expect on a fresh node. The first request pays a one-off NVRTC compilation of the
CUDA kernels before any token appears - the model load itself is seconds, the JIT is not. And
two instances on one host cannot both bind the discovery port, by design: the second falls back
to seeds, so give it the first one's address in `join` when running both on one machine.

`GET /api/cluster/state` shows what a node publishes about itself; `POST /api/cluster/prefix`
asks it how much of a given prompt it already holds.

## The principle

Two subsystems already answer a "should we?" question by measuring rather than assuming, and
the cluster planner is the third instance of the same pattern.

`dry_plan.rs` does not ask what a card can hold; it walks the forward without allocating and
finds out. `speculative_config.rs` does not assume speculative decoding pays; it collects a
profile (`record_baseline`, `record_draft`, `record_transfer`, `record_acceptance`), applies a
cost model (`speedup_ratio`), and `decide()` returns a verdict carrying a
human-readable reason - including when the answer is no:

> `Devices too balanced: verify=12ms vs draft=9ms (need >1.5x ratio)`

A cluster planner that required its fabric to be declared would be a regression against both.
Where the single-node plan asks each card what it can hold, the cluster plan asks each **link**
what it can carry, and decides from the answer.

## The cost model

Start from the cheapest thing the cluster can do, because it sets the bar every other
topology has to clear. For a 1000-token prompt and 500 generated tokens, over a link of
round-trip time `RTT`:

| mode | exchanges | bytes on the wire | added latency |
|---|---|---|---|
| whole-request offload | 1 per request | ~6 KB, once | 1 RTT |
| pipeline (PP) | 1 per node boundary **per token** | ~4 MB | 500 x RTT |
| tensor parallel (TP) | 2 per layer **per token** | ~260 MB | 32 000 x RTT |

Three orders of magnitude separate offloading a whole request from splitting a model across
the same link. So the planner's question is not *how do I split this model* - it is **is there
any reason not to send the entire request to one node?** Splitting is the fallback when no
single node holds the model. It is never the goal.

Per generated token, for the split modes:

```
pipeline (PP)        n_cuts x RTT                  ~ 1 RTT per node boundary
tensor parallel (TP) 2 x n_layers x RTT            ~ 64 RTT for a 32-layer model
```

TP synchronises twice per layer - the row-parallel outputs of `o_proj` and `down` are
all-reduced, while column->row pairs need no exchange (`tp_decode.rs`). PP exchanges one
activation per node boundary: `[1, hidden]` in f16 is 8 KiB at hidden=4096, so the cost is
latency, not bandwidth.

Two orders of magnitude separate them, which usually makes the verdict obvious - TP inside a
node, PP between nodes. **Usually is not always.** Over NVLink or RDMA the RTT falls far
enough that inter-node TP becomes profitable again. A fixed rule would forbid that case; a
measured decision finds it, and says why it took it.

These formulas are the planner's objective function. They are not the planner's conclusion.

## Architecture

One binary, `lokend`, with the role composed by configuration rather than chosen by a flag.
`lokend serve` takes `--port`, `--models-dir`, `--verbose`, `--keep-alive` and `--cpu`; there is
no cluster flag. A node joins by way of the `[cluster]` block of its `config.toml` shown above  - 
`name` decides which cluster it belongs to, `advertise` how peers reach it, and `join` lists the
seeds for peers multicast cannot reach.

```
                       client
                          |
                          v
              +-----------------------+   control plane: SWIM gossip,
              |        router         |   phi-accrual detector, and an
              +-----------------------+   eventually-consistent table
                 |         |         |
       +---------+         |         +---------+
       v                   v                   v
 +-----------+       +-----------+       +-----------+
 |  node A   |       |  node B   |       |  node C   |   full replica
 |   TP=2    |       |   TP=2    |       |   CPU     |
 +-----------+       +-----------+       +-----------+

 A model is sharded across nodes only when no single node holds it, and only
 where the measured link cost says it pays.
```

**No consensus protocol, initially.** In replicated mode the routing table need not be
linearizable: routing to a dead node costs a retry, not a corruption. SWIM gossip with a
phi-accrual failure detector suffices, adds no external dependency, and keeps consensus off
the critical path. Exclusive shard ownership - needed only in sharded mode - is served by a
lease, not by Raft.

**Two planes, separately designed.** Control is HTTP/JSON over the existing `axum` stack:
registration, heartbeat, topology, plans. Data is a persistent binary transport, because the
older distributed protocol serialised its tensor payload through `serde_json` - a decimal ASCII
array, roughly 4x the bytes plus parsing, on the per-token hot path.

**Any node is an entry point, and an entry point holds nothing.** A node that only routes
needs no weights: the client talks to whichever node it knows, that node forwards the whole
request to the best holder and relays the stream back. No dedicated router process, and no
single point of failure in the routing layer.

This also means the replicated cluster needs none of the binary data plane. A forwarded
request is the existing OpenAI/Ollama HTTP request and the response is already SSE - so
replicated serving rides entirely on the current stack. The binary transport becomes a
prerequisite only for sharding.

## Where the weights live

Three separate questions, routinely conflated.

**Who needs weights.** Only nodes that execute, and only for the layers they execute. Weights
must be resident and mapped; streaming them per token is not on the table. But under PP a node
holds only its assigned segment - a 70B split four ways is a quarter of the bytes on each
disk. Sharding cuts the disk footprint as much as the VRAM.

**How weights arrive.** The Ollama-compatible API already serves `HEAD` and `POST` on
`/api/blobs/{digest}` against a real content-addressed store in Ollama's `blobs/sha256-<hex>`
layout, and the downloader already fetches in parallel. A joining node can therefore ask its peers who
holds a digest and pull from the fastest one rather than from upstream - and the link cost
matrix says which peer that is. Peer-to-peer weight distribution is nearly free here.

**What a replica actually is.** This one is correctness, not convenience. Two nodes announcing
`qwen3:8b` must hold *the same bytes*: otherwise the same request answered by either returns
different text, and deterministic recovery collapses - replaying on another node
with the same seed would produce a different continuation.

> A replica set is identified by **digest**, never by model name. Two quantisations of one
> model are two disjoint sets, which the digest handles by construction.

One gap to close before this can bear weight: the digest is real only where an Ollama manifest
supplies it. Models discovered by scanning the Hugging Face cache get a
placeholder digest of `sha256:0000...` from the model manager. Those need a real computed digest
before replica identity can rest on it, so a cluster mixing manifest-backed and cache-discovered
copies of one model cannot yet be trusted to treat them as the same replica set.

## How the pieces fit

Each part below names one capability, what it delivers, and the property a test holds it to.
All of them are in the tree.

### Node discovery

`join` is a seed list, not discovery: on its own it works only when every address is known in
advance and rewritten whenever a machine moves. For a handful of machines on one network, coming
and going, that is the wrong shape - so a node also announces itself on a multicast group and
listens for the others. Multicast rather than broadcast: broadcast reaches every host on the segment whether it cares
or not and is filtered on many networks, while a group is scoped and joined only by those
interested. The announcement carries a node id and an endpoint, and nothing else.

Two hazards, both silent when they happen:

- **Two clusters on one network.** A development machine and a production node sharing a switch
  would merge and route each other's requests. The announcement carries a cluster name and a
  mismatch is ignored, not negotiated.
- **A node discovering itself.** Every node hears its own announcement; one that adds itself as
  a peer compares against itself and can forward to its own address. Filtered on the node id,
  since a machine has several addresses and announces one.

Discovery ADDS to the seeds rather than replacing them: multicast does not cross routers, so a
cluster spanning subnets needs both and must end with the union.

*Invariant:* a foreign cluster, a node's own voice, and a malformed datagram all leave the peer
book unchanged; a peer that moves is the same peer at a new address, not a second one.

### Membership and failure detection

SWIM gossip with a phi-accrual detector - no fixed timeout, which is either too slow or a
false-positive generator under load. Each node publishes its devices (`DeviceInfo`), load,
loaded models, its row of the link matrix, and the prefix block hashes it holds in cache.

*Invariant:* a killed node is evicted from every peer's routing table within a bounded number
of gossip rounds; a paused-then-resumed node is not.

### Binary data plane

Persistent TCP, length-prefixed framing, fixed binary header (request id, layer id, dtype,
shape) followed by the raw tensor payload with no copy (`bytes::Bytes`). Many requests
multiplexed per connection, explicit backpressure. Control traffic stays on HTTP/JSON.

The quantised KV caches (`q8_kv_cache.rs`, `q4_kv_cache.rs`) already carry the code to halve
a payload at a measurable error; the same applies to transported activations, as an option.

*Invariant:* a `[1, hidden]` f16 round trip between two processes is bit-exact, and p99
latency stays under a threshold pinned in the test.

### Fabric profiler

On join, and periodically after, every node pair measures its link: idle RTT, RTT under load,
sustained bandwidth, jitter. The result is a **link cost matrix**, not a scalar - a generic
cluster is heterogeneous down to its wiring, and one plan may span NVLink, a shared switch,
and a slow uplink at once.

This is the transposition of `dry_plan`: where the single-node plan asks each card what it can
hold, this asks each link what it can carry.

*Invariant:* the matrix is symmetric within measurement noise, and a link degraded at runtime
is reflected within one gossip round.

### Whole-request offload and cache-aware routing

The bulk of the value, fault-tolerant by construction, and - since a forwarded request is just
the existing HTTP request - buildable before the binary data plane exists.

Any node receiving a request either serves it or forwards it whole, streaming the response
back. Node selection is measured, not merely availability-based: forward when
`predicted_completion(B) + RTT < predicted_completion(A)`, which covers the node that holds the
model but is saturated, and the node that holds it but is simply slower. Because loken already
reports joules per request, the same arbitration can prefer the most efficient node that still
meets the latency target.

The paged KV allocator already computes per-block hashes and holds a prefix cache.
Publishing those hashes through gossip lets the selection maximise shared prefix against load,
with power-of-two-choices to avoid herding. Plus admission control, circuit breaking on
failure, and slow-start reentry.

*Invariants:* a node holding no weights can serve any request the cluster can serve. A request
whose prefix is cached on exactly one live node routes there, unless that node's load exceeds
the configured margin.

The figure the whole comparison rests on is each node's sustained throughput, and it is measured
over a window - tokens emitted divided by the time taken - never as a per-request rate times a
width. Where generations are serialised behind the model lock, one request runs at full speed
while the others wait, so that product counts waiting requests as concurrent ones. Measured on
two nodes of equal capacity it published 13338 tok/s for a machine emitting 623, and the router
kept 22 requests out of 24 that it should have shared.

### Deterministic recovery

A request already carries its own seed and the native sampler seeds its
`StdRng` from that, so determinism is within reach. When a node dies mid-generation, the router
re-routes with `prompt ++ tokens_already_emitted` and the same seed. The client sees a pause;
the cost is a re-prefill, not a loss.

**One change is required first.** A sequential `StdRng` carries state: after N tokens it has
consumed N draws, and replaying from the start reproduces the sequence only if the RNG is
advanced by exactly N - fragile, and broken as soon as draws per token vary (top-k, top-p,
rejection). Seed per position instead:

```rust
StdRng::seed_from_u64(hash(request_seed, position))
```

The RNG becomes stateless, recovery is exact at any point, and - a welcome side effect  - 
sampling becomes reproducible under continuous batching, where a request's position within
the batch varies run to run.

Replay must land on a node whose weights carry the **same digest**, per "Where the weights
live" above - a same-named replica holding different bytes would continue the generation
differently.

*Invariant:* a generation interrupted at token N and resumed on another node of the same
replica set produces a token stream identical to the uninterrupted one, at any temperature.

### Hierarchical KV

A tiered pool above `PagedKvAllocator`: VRAM -> host RAM -> NVMe -> peer node. Enables KV
migration - cheaper than a re-prefill on long contexts - for both recovery and live
rebalancing. Prefill/decode disaggregation belongs after this, and only if the binary
transport is fast enough to pay for itself.

*Invariant:* a KV block migrated between tiers and read back is bit-exact.

### Evidence

Extend the energy report across the cluster: joules per request summed over every node that
served it. Nobody publishes this at cluster scale, and the measurement grows more interesting
with node count, not less.

Chaos benches: kill a node mid-run and record p99, error rate, tokens lost, reconvergence
time. Which topology the cluster chose, and why, belongs in the report - a benchmark that does
not say what it measured is not reproducible.

## Configuration

The fabric is discovered, so it is not configured. What remains is policy:

```toml
[cluster]
# Name of this cluster. Nodes announcing another name are ignored, so two clusters
# can share a network without merging.
name = "default"

# Seeds to contact on start, for peers multicast cannot reach (another subnet).
# Discovery adds to this list; it does not replace it. Empty + no peers heard = single node.
join = []

# Recovery when a node dies mid-generation:
#   "replay"  - re-route with prompt ++ emitted tokens (needs nothing from the fabric)
#   "migrate" - transfer the KV blocks (hierarchical KV; faster on long contexts)
#   "fail"    - surface the error to the client
recovery = "replay"

# Refuse to distribute below this predicted speedup over the best single node.
min_speedup = 1.15
```

`recovery = "replay"` is the default because it demands nothing of the infrastructure.
`min_speedup` mirrors the equivalent threshold used by the speculative-decoding decision, and
governs every hand-over: a peer that is not this much faster keeps the request where it is. It
is also what a hand-over sequence converges to, since each one charges the peer a queue it has
not reported yet - so the threshold, not the capacity gap, is where sharing stops.

## Non-goals

- **Inter-node TP as a design target.** It is reachable when the measured RTT justifies it,
  and a measured planner would take it. It is not something to build toward.
- **Consensus for the routing table.** Revisit only if exclusive shard ownership outgrows
  leases.
- **Elastic autoscaling.** Nodes join and leave; deciding when they should is a layer above.
