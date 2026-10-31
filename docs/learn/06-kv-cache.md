# What the model remembers between tokens

Attention lets a token read every earlier one, which would mean recomputing every earlier
key and value at each step. The cache keeps them instead.

## The idea

For every token seen, every layer keeps its key and value vectors, one row each per key
head. On qwen3:0.6b that is 28 layers times 8 heads times 128 values times 2, 57 344
numbers a token, 112 KB at F16. A 4 096-token window is 460 MB, close to the weights. On a
larger model the cache is where the memory goes, and the per-token bytes are what
grouped-query and latent attention were invented to cut.

The cache is also **state**: a conversation's next turn is the same prompt plus a little,
and a server that keeps the cache prefills only the little. This drives a series of
designs:

- **paging**: the cache is a pool of fixed blocks and a sequence holds a block table, so
  several sequences share one pool without reserving a full window each, and 2 prompts
  with a common prefix can share its blocks.
- **quantised caches**: keys and values at 8 bits, or at 4 with the key scale taken
  per channel (a key's channels have systematic outliers, so a per-token scale loses them).
- **sliding windows**: some layers attend only to the last n positions and keep no more.
- **snapshots**: after a request the resident cache is copied aside under its tokens, so
  another conversation can take the cards and this one can come back.
- **a disk tier**: full token blocks written after a response, keyed by a hash chained
  over the tokens, so a shared prefix is written once and survives a restart.
- **reuse past the window**: when a conversation outgrows the window, keep the tail and
  re-phase its positions rather than prefill the window again.

![Where a conversation's KV lives](../img/learn-kv-tiers.svg)

2 rules follow:

- **One live cache** per model, so 2 conversations served in alternation evict each other
  unless a snapshot holds the other.
- **A window too small** is slower than a large one: compaction rewrites the history from
  near its start, and a rewritten prefix matches nothing that is held.

## What loken does

loken keeps one live cache per model, paged into fixed blocks, at F16, 8 bits or 4 bits with a per-channel key scale. It keeps a snapshot of a finished conversation so that an alternated one comes back without a full prefill.

A new prompt reuses the longest prefix it finds, in this order: the resident cache, a snapshot that covers more, a disk sequence that covers more still. The reused prefix is rounded down to a prefill chunk.

The disk tier is on when `kv_disk_dir` is set, with a size budget. Snapshots need no key. `fidelity` set to `reference` turns off the shift past the window. The guide is [`../guides/kv-memory.md`](../guides/kv-memory.md).

## What was measured

**Snapshots (2026-09-06, desktop node).** 2 conversations of 200 lines each,
alternated, at an 8 192-token window: the prefill of an alternated turn went from about
5 195 tokens to about 1 100 on gemma4 (1 170 ms to 300 ms). Extended the same day to
quantised caches, captured by dequantising a layer to F16 and restored by re-quantising it
on the same grid: llama3.2 at Q8 4 553 to 457 tokens with bit-identical output, qwen3 at Q4
5 233 to 1 137 within one quantisation step. The disk tier persists only prefix-uniform
snapshots, with a sliding-window model's windowed layers in a side blob. On gemma4, 1 248 of 3 296
tokens were re-prefilled after a restart. Tried and removed: deflate on the blocks (395 MB in
15 s against 433 MB in 5 s) and a second on-disk format at 8 bits. The disk holds the
native dtype, and disk and memory hold the same values bit for bit.

**3 bits, measured and refused.** A randomised Hadamard rotation does spread the
outliers (excess kurtosis 35.7 to -0.6) and beats F16 as promised. Against the 4-bit cache
already here it does not: per-channel grouping removes the outliers first, so rotating
changes the error by under 0.5%, and Q4_0 is 2.6 times more accurate for 1 more bit
([`../STATUS.md`](../STATUS.md#measured-and-refused)).

**One cache per model (2026-09-11).** An agent's speculative read ran at batch priority on
the same model as its conversation, so every interactive step re-prefilled from nothing:
137 s where a reuse costs 1.2 s. Work that yields now goes to a peer that can serve the
model or is declined, and the pricing cannot see this cost, since it falls on a different
request than the one being priced. On the same day, the same question at 2 windows:
301 s at 32 768 tokens against 24 s at 131 072, with later turns at 250 ms against 60 s,
which is why the agent's default window is the large one.

## Try it

The reused prefix is rounded down to a prefill chunk, and on a small model with a free card
the chunk is 2 048 tokens, so the conversation has to be longer than that to
show anything. Send a first turn of about
4 400 tokens on conversation A, then its second turn, then a first turn of another
conversation B, then A's second turn again, each with `"options":{"num_ctx":8192}`, and read
`prompt_eval_count` and `prompt_eval_duration` on each reply.

Observed on qwen3:0.6b, RTX 5070 Ti:

| turn | `prompt_eval_count` | `prompt_eval_duration` |
|---|---|---|
| A, first turn, cold | 4422 | 592 ms |
| A, second turn | 381 | 24 ms |
| B, first turn, cold | 4522 | 472 ms |
| A, second turn again | 381 | 26 ms |

The second turn computed 381 tokens: 2 chunks of the 4 477 shared were reused and the
remainder recomputed. The last line came from a snapshot, and the daemon's log says so:
`kv snapshot restored: 4477 tokens shared with the prompt, 4506 resident`. With turns shorter than a chunk
nothing is reused at all, and the counts say so.
