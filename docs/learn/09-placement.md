# Fitting a model on the machine you have

Where each part of a model lives decides its speed more than any kernel does. This lesson covers
the decision: cards, host, disk, and the other machine, and what a wrong choice costs.

## The idea

A card holds what fits in its memory and reads it at its own bandwidth. What does not fit
goes to the host and is read over the bus at the host's rate, an order of magnitude lower;
what does not fit the host either is read from disk on demand, another order lower. The
rule: fill the fastest card first, then the next, and let the host take
only what no card can hold.

Three things make it hard. The **weights are not the whole cost**: a request needs
activations, a cache for every token of its window, workspace for the kernels, and a fixed
overhead per loaded model, none of which is in the file. The **allocator is not the
program**: a card reports its pool's reservation, which depends on the order of
allocations and frees, not on the peak the program held. And a **mixture is not a dense
model**: its experts are nearly all of its bytes and a token touches a few of them, so
spilling a whole layer, attention included, to keep expert bytes together is the wrong cut.

![Two ways to spill a mixture](../img/learn-placement.svg)

When the working set does not fit the host either, it is read from disk, and the
metric becomes bytes read per token: which experts a token needs, how many of them the
last token needed too, what the operating system's page cache keeps, and whether reading
overlaps computing. Across machines the same rule applies: a node
forwards a request only when the other node's predicted completion, plus the trip, beats
its own.

## In loken

| File | What it decides |
|---|---|
| `src/inference/place/plan.rs` | The heterogeneous plan every block stack uses: pack the fastest card first, spill card to card to host, never out of memory. |
| `src/inference/place/vram_manager.rs` | The one place placement reads a card's state from: pools trimmed before the probe, a reclaim registry for idle media residents, headroom for a hot component before it loads. |
| `src/tensor/dry.rs`, `src/tensor/heap.rs`, `src/inference/place/dry_plan.rs` | The circularity break: a request's peak is measured by running the forward on a device that allocates nothing and keeps a high-water mark, then replaying the allocator's chunk and hole behaviour on top of it. |
| `src/inference/place/placement_invariants.rs` | The fleet rule as tests: the cards decide, not the budgets; no placement decides on a name or an index; every hot component can span n cards. |
| `src/inference/place/layer_executor.rs` | The host budget as a share of the machine, not a constant. |
| `src/inference/offload/streamed.rs`, `room.rs`, `cuda.rs` | The streamed placement for a model larger than the cards and the host together: the weights every token reads and as many routed experts as the room allows kept on the cards, fastest first, under the same switches as every other placement; the room derived from the card's memory and the model's declared transient, never a constant. |
| `src/inference/offload/store.rs` | The hot tier over streamed experts: views into the mapping, least-recently-routed eviction, cold experts dropped from the page cache as soon as they have run. |
| `src/distributed/routing.rs` | Which node answers, by predicted completion rather than by load. [`../CLUSTER.md`](../CLUSTER.md) has the cost model. |

## What was measured

**Spill the experts, not the layer (2026-09-02 to 03, desktop node).** With a layer placed
whole, a spilled mixture ran its attention on the host for the sake of expert weights that
are 97% of its bytes and are read a few at a time. Spilling only the experts took qwen3next
from 15.6 to 34.7 tokens a second (ollama: 33) and qwen3-coder-next from 10.7 to 30.8
(26 to 28). A mixture that already fits is untouched
([`../STATUS.md`](../STATUS.md#decode-rate-vs-ollama)). The same page prices the costs the
planner does not know: about 870 MiB per loaded model whatever its size, and a spilled
model that was held twice in host memory until the page advice followed the plan.

**A 552B mixture on a USB disk (2026-09-14 to 17).** DeepSeek V4.1 Flash, 340 GB on a disk
that reads 440 MB/s with one reader and 461 to 464 with two to sixteen: parallel readers
buy nothing on that link, only fewer bytes, shared reads and overlap count. Routing has
locality: 32% of a token's experts were the previous token's, 50% within the last four,
67% within the last sixteen. A managed hot tier on top of the page cache was measured in a
controlled alternation (6.29, 5.91, 6.06, 6.04 s a token with it off, on, off, on): about
5% of the bytes and 2% of the time, because the kernel's own eviction already does the job
at that horizon. Two things that made it worse, not to be retried: anonymous copies of hot
experts (41 GB copied, 22 GB into compressed swap, 10.7 s a token), and `MADV_RANDOM` on
the expert stacks, which cut the read rate from 342 to 46 MB/s for the same access pattern
and had been set with the opposite intent. With the prompt prefilled as one batch on the
cards, a 1 024-token prompt reads 141.6 GB of experts in 348 s at 407 MB/s against a
320 s floor at the disk's rate: the path is disk-bound, and the levers left are the bytes.

**Two nodes, no hand-over (2026-09-13).** Fifty concurrent requests on qwen3:0.6b, a
desktop with an RTX 5070 Ti and a laptop with an 8 GB card: the desktop chose to serve
locally 554 times and to forward 0 times. Not a defect: the router forwards only when the
peer lowers the predicted completion, and the fast card drains its queue faster than a hop
plus a slower card. The laptop's value is holding a model the desktop does not
([`../STATUS.md`](../STATUS.md#the-cluster)).

## Try it

```sh
curl -s localhost:11435/api/models/loaded | python3 -m json.tool
```

Observed with qwen3:0.6b: one entry, `num_layers: 28`, one `layer_distribution` element on
`CUDA` device 0 covering layers 0 to 27, `memory_bytes: 522640096`. Then ask for a model
larger than one card (qwen3-coder:30b at Q4_K_M is 18 GB on 16 GB cards) with one
`/api/generate` request, and read the same route again: two elements, one per card, with
the layer ranges the planner chose. A model larger than both cards adds a `CPU` element,
and its decode rate falls to the host's bandwidth over the bytes it holds.
