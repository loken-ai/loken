# Fitting a model on the machine you have

Where each part of a model sits decides its speed more than any kernel does.
The fastest card fills first, then the next card.
The host takes only what no card holds.
Disk sits below the host, another order of magnitude slower.

3 facts make placement hard:

- **Weights are not the whole cost.** A model also needs activations, a KV cache for every token of the window, kernel workspace, and about 870 MiB fixed per loaded model. None of it is in the file.
- **The allocator is not the program.** A card reports its pool's reservation. That figure depends on the order of allocations and frees, not on the peak the program held.
- **A mixture is not a dense model.** Its experts hold about 97% of its bytes and a token touches a few of them. Spilling a whole layer, attention included, to keep experts together is the wrong cut.

Past the host, the metric is bytes read per token.
It depends on which experts a token needs, how many the last token needed too, what the page cache kept, and whether reading overlaps computing.

Across machines, forward a request only when the peer's predicted completion plus the trip beats serving it locally.

![2 ways to spill a mixture](../img/learn-placement.svg)

## What loken does

loken packs the fastest card first, then spills from card to card, then to the host. It does not run out of memory.
The host budget is a share of the machine.
No placement depends on a card's name or index.
Every hot component can span n cards.

It reads a card's state from 1 place.
It measures a request's peak by running the forward on a device that allocates nothing, then replays the allocator's behaviour on top.

A model larger than the cards and the host together is streamed.
The weights every token reads stay on the cards, with as many routed experts as fit, fastest card first.
The other experts are read from disk, through a hot tier that evicts the least recently routed.

Across nodes, a request goes to the node with the best predicted completion, not the lowest load.
[The cluster page](../CLUSTER.md) describes how a request is routed.

## What was measured

### Spill the experts, not the layer

Dates: 2026-09-02/03.
Placing a layer whole ran a spilled mixture's attention on the host to save expert weights.
Those weights are 97% of the layer's bytes and are read a few at a time.
Spilling only the experts changed the rate, in tokens per second:

<!-- site: widget chart type=bar id=placement-spill -->
| Model | Layer spilled | Experts spilled |
|---|---|---|
| qwen3next | 15.6 | 34.7 |
| qwen3-coder-next | 10.7 | 30.8 |
<!-- site: end -->

A mixture that already fits is untouched.
The planner cannot see 2 costs: about 870 MiB per loaded model whatever its size, and a spilled model held twice in host memory until the page advice followed the plan.
See the [decode rate](../STATUS.md#decode-rate).

### A 552B mixture on a USB disk

Dates: 2026-09-14/17.
The model is DeepSeek V4.1 Flash, 340 GB, on a disk that reads 440 MB/s with 1 reader.
2 to 16 readers reach 461 to 464 MB/s, so only fewer bytes and overlap count.

Routing has locality:

| Window | Share of a token's experts also used before |
|---|---|
| Previous token | 32% |
| Last 4 tokens | 50% |
| Last 16 tokens | 67% |

A managed hot tier over the page cache saved about 5% of the bytes and 2% of the time.
The run alternated off, on, off, on: 6.29, 5.91, 6.06 and 6.04 s per token.
The kernel's own eviction already does the job at that horizon.

2 changes made it worse. Do not retry them:

- Anonymous copies of hot experts: 41 GB copied, 22 GB in compressed swap, 10.7 s per token.
- `MADV_RANDOM` on the expert stacks: reads fell from 342 to 46 MB/s.

Prefilled as 1 batch on the cards, a 1024-token prompt reads 141.6 GB of experts in 348 s at 407 MB/s.
The disk floor is 320 s.
The run is disk-bound, so only fewer bytes help.

### 2 nodes, no hand-over

Date: 2026-09-13.
Setup: 50 concurrent requests on qwen3:0.6b, a desktop (RTX 5070 Ti) and a laptop (8 GB card).
The desktop served 554 locally and forwarded 0.

This is expected.
The router forwards only when the peer lowers the predicted completion, and the fast card drains its queue faster than a hop to a slower one.
The laptop's value is holding a model the desktop does not.
See [the cluster](../STATUS.md#the-cluster).

## Try it

<!-- site: widget steps id=placement-try -->
1. Read the route of a loaded model:

   ```sh
   curl -s localhost:11435/api/models/loaded | python3 -m json.tool
   ```

2. With qwen3:0.6b, expect 1 entry with `num_layers: 28` and 1 `layer_distribution` element on `CUDA` 0 covering layers 0 to 27.
3. Load a model larger than 1 card, such as qwen3-coder:30b at Q4_K_M (18 GB on 16 GB cards), and read the route again. Expect 2 elements, 1 per card.
4. Load a model larger than both cards. The route gains a `CPU` element, and the decode rate falls to the host's bandwidth over the bytes it holds.
<!-- site: end -->

## Check yourself

<!-- site: widget quiz answer=1 id=placement-quiz-1 -->
For a spilled mixture, what does loken spill?

1. Only the experts.
2. The whole layer, attention included.
3. Only the attention.

Experts hold about 97% of the bytes and a token touches a few.
Attention then stays on the card.
<!-- site: end -->

<!-- site: widget quiz answer=3 id=placement-quiz-2 -->
Why did the desktop forward 0 of 554 requests to the laptop?

1. The laptop was offline.
2. The router balances by lowest load.
3. The peer never lowered the predicted completion.

The router forwards only when the peer's predicted completion plus the trip beats local serving.
<!-- site: end -->

<!-- site: widget quiz answer=2 id=placement-quiz-3 -->
What limits a model streamed from disk?

1. Kernel speed.
2. Bytes read per token.
3. The number of readers.

Parallel readers bought nothing on the USB disk.
<!-- site: end -->

<!-- site: widget flashcards id=placement-terms -->
| Term | Meaning |
|---|---|
| Spill | Placing part of a model on the next slower tier: another card, the host, then disk. |
| Layer distribution | The list of devices and layer ranges a loaded model occupies. |
| Hot tier | The routed experts kept in memory, evicting the least recently routed. |
| Streaming | Reading the experts that fit nowhere from disk at each token. |
| Predicted completion | The cost model loken uses to pick a node for a request. |
<!-- site: end -->
