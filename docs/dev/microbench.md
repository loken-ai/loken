# Microbench

A microbench times one mechanism of a decode on synthetic data of a real model's shapes.
It loads no weight and starts no server.  
It runs its variants side by side in one process, so one run settles one hypothesis.

Mechanisms of this size cost a few microseconds to a few milliseconds each: a row crossing the bus, an expert uploaded to a card, an expert read from the disk.

Measuring a mechanism on a served model costs a load of the whole file per hypothesis.

The benches are the bench target `benches/microbench.rs`, built by the `microbench` feature.
No release build enables it, so nothing reaches the daemon.

## Run

```sh
cargo bench --features cuda,microbench --bench microbench -- list
cargo bench --features cuda,microbench --bench microbench -- crossing --shapes-from /models/large.gguf
cargo bench --features cuda,microbench --bench microbench -- expert-upload --shapes-from /models/large.gguf --json results/upload.json
cargo bench --features cuda,microbench --bench microbench -- expert-ahead --shapes-from /models/large.gguf --guess-hit 0,0.5,1 --json results/expert-ahead.json
cargo bench --features cuda,microbench --bench microbench -- expert-read --shapes-from /models/large.gguf --layer 10
cargo bench --features cuda,microbench --bench microbench -- expert-pairs --shapes-from /models/large.gguf --layer 3 --block 1,9 --json results/expert-pairs.json
cargo bench --features cuda,microbench --bench microbench -- host-pairs --shapes-from /models/large.gguf --layer 3 --block 1,9 --json results/host-pairs.json
cargo bench --features cuda,microbench --bench microbench -- prefill-layer --shapes-from /models/dense.gguf --json results/prefill-layer.json
cargo bench --features cuda,microbench --bench microbench -- shortconv-prompt --shapes-from /models/hybrid.gguf --rows 1000
cargo bench --features cuda,microbench --bench microbench -- mamba2-prompt --shapes-from /models/mamba.gguf --rows 1000
cargo bench --features cuda,microbench --bench microbench -- marlin-rows --json results/marlin-rows.json
cargo bench --features cuda,microbench --bench microbench -- decode-inputs --batch 64 --context 512
cargo bench --features cuda,microbench --bench microbench -- decode-layer --batch 64 --context 512 --json results/decode-layer.json
cargo bench --features cuda,microbench --bench microbench -- media-overlap --rows 1024 --hidden 1280 --layers 4
cargo bench --features cuda,microbench --bench microbench -- parler-step --hidden 1024 --inter 4096 --layers 24 --heads 16 --vocab 1088
cargo bench --features microbench --bench microbench -- schedule-host --batch 64 --context 512
cargo bench --features microbench --bench microbench -- host-prompt --hidden 4096 --inter 12288 --rows 108
```

A ROCm build takes `rocm` in place of `cuda`.  
A build without a GPU runtime omits `crossing`, `decode-inputs`, `decode-layer`, `dequant-gemm`, `expert-ahead`, `expert-pairs`, `expert-upload`, `host-pairs`, `marlin-rows`, `media-overlap`, `prefill-layer`, `shortconv-prompt`, `mamba2-prompt` and `parler-step`.
`parler-step` also needs the `audio` feature, which `media` brings.

Arguments after `--`:

| Argument | What it sets | Default |
|---|---|---|
| `name` | The bench, or `list`. | required |
| `--shapes-from` | A GGUF whose metadata and tensor table give the shapes. No weight is loaded. | none |
| `--layer` | The layer whose routed experts give the expert shapes. | the first layer with routed experts |
| `--expert` | The expert the file-reading variants start at. | 0 |
| `--tensor` | One tensor of the file in place of the layer's experts. | none |
| `--hidden` | Elements of one activation row, without a file. | 4096 |
| `--hc` | The hyper-connection multiple, without a file. | 1 |
| `--element-bytes` | Bytes per element of a row. | 4 |
| `--inter` | The feed-forward length, without a file. | 2048 for an expert, 12288 for `marlin-rows` |
| `--weight-bits` | Bits per stored weight, without a file. 4.5 for q4_k, 8.5 for q8_0. | 4.5 |
| `--projection-bytes` | The bytes of each projection, 1 value or 3, in place of `--inter` and `--weight-bits`. | derived |
| `--gate-format` | The format of an expert's gate and up weights, without a file (`host-expert`, `expert-pairs`). | iq2_xxs |
| `--down-format` | The format of an expert's down weights, without a file (`host-expert`, `expert-pairs`). | q2_k |
| `--rows` | Rows of the prompt batch one layer of `prefill-layer` moves, or the prompt rows of `host-prompt`, `shortconv-prompt` and `mamba2-prompt`. | 130 |
| `--heads` | Query heads, without a file (`prefill-layer`, `marlin-rows`, `decode-layer`). | the file's for `prefill-layer`, 32 for `marlin-rows` |
| `--kv-heads` | Key-value heads, without a file (`prefill-layer`, `marlin-rows`, `decode-layer`). | the file's, else the heads for `prefill-layer`, 8 for `marlin-rows` |
| `--head-dim` | Elements of one head, without a file. It sizes the rope rows of `schedule-host` and `decode-inputs`, and the heads of `decode-layer`. | the file's, else the hidden size over the heads for `prefill-layer`, 128 for the others |
| `--pack` | Rows the packed variants of `crossing` move in one copy. | 8 |
| `--file` | The file `expert-read` reads, or the recorded trace `cache-policy` replays. | the shapes file for `expert-read` |
| `--offset` | Where the first expert read starts, when the file is not the shapes file. | 0 |
| `--threads` | Threads of the threaded variants. | the processors the host reports |
| `--card` | The card the card benches run on. | 0 |
| `--seconds` | Wall time of each variant. A warm-up of a tenth of it comes first. | 1.0 |
| `--batch` | Running sequences, 1 decode row each (`schedule-host`, `decode-inputs`, `decode-layer`). | 64 |
| `--context` | Tokens of context per sequence (`schedule-host`, `decode-inputs`, `decode-layer`). | 512 |
| `--prompt-tokens` | Tokens of each sequence's prompt (`schedule-host`). | 53 |
| `--block-size` | Tokens per block of the paged cache (`schedule-host`, `decode-inputs`, `decode-layer`). | 16 |
| `--layers` | Layers of the stack a step runs (`decode-layer`, `expert-ahead`). | 36 |
| `--vocab` | Entries of the vocabulary the head matmul and the argmax run over (`decode-layer`). | 131072 |
| `--codebooks` | The codebooks of the audio decoder (`parler-step`). | 9 |
| `--blocks-per-row` | The width of a block table (`decode-inputs`). | `--context` over `--block-size`, rounded up: 32 |
| `--rope-base` | The rope base of the rope rows (`schedule-host`, `decode-inputs`). | 10000 |
| `--json` | Where the JSON record is written. | none |
| `--profile` | The expert profile `cache-policy` draws its routing from. | none |
| `--card-slots` | The hottest experts per layer the cards hold, which the host tier never serves (`cache-policy`). | 58 |
| `--slots` | Slots per layer the host tier is tried with, 1 variant each (`cache-policy`). | 94,130,170 |
| `--guess-hit` | The share of a token's experts the read-ahead guess names, 1 variant each (`cache-policy`, `expert-ahead`). | 1.0 |
| `--block` | Tokens verified together, a draft's block. The block reads its experts once. 1 variant each (`cache-policy`). The rows of a verify, 1 variant each (`expert-pairs`). | 1 |
| `--accept` | The share of a draft token accepted once the token before it was (`cache-policy`). | 0.8 |
| `--tokens` | Tokens routed per variant (`cache-policy`). | 20000 |
| `--group-size` | Weights per quantisation group, only 128 runs (`marlin-rows`). | 128 |
| `--rows-list` | Row counts of the GEMM, 1 variant each (`marlin-rows`). | 1,8,16,32,33,64,65,130,260,512 |
| `--compare-rows` | The row count whose layer time is set against the time at 1 tile of rows (`marlin-rows`). | 130 |
| `--disk-gbs` | The disk read rate in GB/s, which turns misses into time (`cache-policy`). | 6.0 |

With `--shapes-from`, the row is the file's hidden size times its hyper-connection multiple.  
The expert is the layer's 3 routed projections.

The projections are cut from a stacked tensor or taken from one tensor per expert, with the offset of each expert's copy in the file.

Without a file, `--hidden`, `--hc`, `--element-bytes` and `--inter` or `--projection-bytes` give the row and the projections' bytes.

## Benches

### crossing

`crossing` moves one activation row from the host to the card and back.
It measures 1 host-card crossing, the unit behind a layer's count of crossings.  
Its variants:

- 1 copy each way from pageable memory with a stream synchronisation, as a decode step crosses.
- The same from page-locked memory, queued and waited on with an event.
- The round trips, pageable and page-locked.
- `--pack` rows in separate copies, against the same rows packed in 1 copy each way.
- A stream synchronisation and an event wait with nothing queued.

### expert-pairs

`expert-pairs` times the routed pairs of 1 layer on a card, for a decoded token and for the rows of a verify.
It runs at the widths and formats of the layer `--layer` of the shapes file, with the pairs a row its `expert_used_count` states.
Without a file, `--hidden`, `--inter`, `--gate-format` and `--down-format` set the layer, and a row routes to 8 pairs.

The rows of each variant are the row counts of `--block`, 1 variant each.
Each row routes to distinct experts drawn from a fixed seed, as a router's choice.
Consecutive calls take different routings, enough of them that the weights read between 2 uses of one exceed the card's cache.
A call reads its weights from the card's memory, as a token's layer does after the layers before it.

The stacks hold the layer's routed experts, so the pairs read their weights from the card's memory as a served model does.
Without a file they hold the pairs of every row.
They take 3 projections of each expert on the card, a few gigabytes for a large layer.
A card whose free memory holds fewer keeps what it holds, and the fused gate and up row says how many.

The weights are synthetic bytes of the stacks' formats, with a small finite scale in front of each block.
The times do not depend on the values.

Its variants, for each row count:

- `fused gate and up`: the expert kernel of a layer without exact kernels, gate and up with the SiLU in 1 launch, its quantisation of the input included.
- `fused down, pairs`: that layer's down kernel writing each pair's row, as a recorded decode runs it.
- `fused down, reduced`: the same, each row's pairs folded into it.
- `fused, gate and up then down reduced`: the 2 fused launches in sequence, as a token of a layer without exact kernels runs them.
- `exact quantise x`, `exact gate and up`, `exact quantise h`, `exact down` and `exact sum`: the launches of the exact kernels 1 at a time.
- `exact, all 5`: those 5 in sequence, as 1 token of a recorded decode runs them on the card.

A layer whose stacks have no exact form gets a note in place of the exact rows.

Each call is timed between 2 events of the card's stream.
The bandwidth is the weight bytes the pairs read, which the GEMM rows and the totals carry.

Read `exact gate and up` against `fused gate and up`, `exact down` against `fused down, pairs`, and `exact, all 5` against `fused, gate and up then down reduced`.
The quantisations and the sum are what the exact path adds around them.

### expert-upload

`expert-upload` copies the 3 projections of one routed expert from the host to the card.
It measures a synchronous upload inside a token and what overlap buys.  
Its variants cross 3 choices:

- The compute stream, a copy stream, or a copy stream ordered after an event recorded on the compute stream, which is how the uploader orders its allocation.
- Pageable or page-locked memory.
- The card idle, or a kernel holding the compute stream.

The kernel runs as long as 1 idle upload from page-locked memory on the compute stream.  
Each busy variant gets a second row: the kernel's own time beside that upload, against its time alone.

### decode-inputs

`decode-inputs` uploads the per-step inputs of a batched decode, as the continuous-batching worker does between 2 steps.
It loads no model and replays no decode graph.

The inputs are the token ids, the slot of each row, the sequence lengths, the block tables padded to `--blocks-per-row`, and the rotary cos and sin rows in half precision.
The blocks per row default to the blocks of `--context` tokens at `--block-size` tokens a block, 512 over 16 giving 32.
Every row is at full context, the worst case for the block tables.
The hidden state the worker also copies into its stable buffers is left out, since it is produced on the card.

The variants run the real `Tensor::from_vec`, `rope_cos_sin` and `slice_set`.
Each `from_vec` to a card is a host-to-card copy followed by a stream synchronisation.
Its variants:

- As the worker does: 6 `from_vec`, 6 `slice_set` into stable buffers, then 2 stream synchronisations.
  The worker puts the graph replay between the 2, so here the second waits for nothing.
- The same without the 2 explicit synchronisations, ending with 1 event wait.
  The 6 synchronisations inside `from_vec` stay.
- Packed: all the arrays written back to back into 1 page-locked buffer, each aligned to its element size, then 1 asynchronous copy to 1 card buffer, then 1 view per array, then 1 event wait.
- The floors: a synchronisation and an event wait with nothing queued, and the replay of a graph holding a kernel that returns at once, waited on with a synchronisation and with an event.

The 3 upload variants include the host work that builds the arrays, so the packed variant pays for writing the cos and sin rows too.
The variant name carries its host-to-card copies (`h2d`), card-to-card copies (`d2d`) and synchronisations (`syncs`).
The note of each upload variant gives its median over the first variant's.
The views are byte ranges of the card buffer, and no kernel reads them.

### decode-layer

`decode-layer` times the kernels of a batched decode step that are not a GEMM.
It answers where the time goes once the Marlin GEMMs are taken out, and which kernel sits far above the bytes it must move.
It loads no model and replays no decode graph of a model.

Each kernel is called through the wrapper the layer loop of the decode graph uses, on synthetic F16 tensors at the real shapes.
The shapes come from `--hidden`, `--heads`, `--kv-heads`, `--head-dim`, `--inter` and `--vocab`, the defaults of `marlin-rows` for the others.
The rows are `--batch` rounded up to its graph width, each at `--context` tokens, in blocks of `--block-size`.
The paged store has a block table and a slot for every row.
Copies of the store, together larger than twice the second-level cache, are read in turn, so the attention streams the cache from card memory.

The kernels of a layer, in the order of the layer loop:

- 3 copies that cut q, k and v out of the fused projection (`narrow`).
- The norms on the heads of q and k (`fused_rmsnorm_f16`).
- The rotation of k, then q (`rope_apply`).
- The write of k and v into the paged store (`paged_kv_write`).
- The paged flash decode (`paged_flash_decode`).
- The add of the attention output and the feed-forward norm (`fused_add_rmsnorm_f16`).
- 2 copies that cut the gate and the up out of the fused projection (`narrow`).
- `silu_mul_f16`.
- The add of the feed-forward output and the next layer's norm (`fused_add_rmsnorm_f16`).

The kernels of a step:

- The embedding gather (`embed_rows_gguf`) from a Q8_0 table, and the cast of its rows to F16.
- The first layer's norm.
- The copy of the hidden state out of the graph (`slice_set`).
- The head matmul (`matmul_t`) of a float head, and the batched argmax over its logits.

The upload of the step's inputs is `decode-inputs`.
A model without a norm on the heads skips the 2 norms, and a model with separate projections skips the 5 copies.
The record and the final lines give the sums that include them and the part they hold.

Each kernel is one variant.
Its time is the host clock around the call and a stream synchronisation, so it includes the launch of the kernel and the wait.
The first variant is an empty kernel timed the same way, the cost to read the others against.

The bytes of a kernel are its inputs and its outputs once each, and the key and value rows of every row's context for the attention.
The floor is those bytes at the peak memory bandwidth, the double-rate memory clock over the bus width the driver reports.
Each row's note gives the floor, the median over the floor, and the share of the peak.
Small kernels read from the second-level cache, so their bandwidth can pass the peak.

Then a graph:

- The 14 kernels of 1 layer captured in 1 graph for each copy of the store.
- 1 replay and 1 synchronise, against the sum of the individual medians.
  The difference is the launch and the wait the individual numbers include.
- `--layers` replays back to back and 1 synchronise, divided by the layers.
  This is the in-graph cost of a layer.

2 groups of variants set the loops of a layer against the kernels that replace them, on the same tensors:

- `q/k/v stage, old: 3 copies, 2 norms, 2 rope, kv write: 8 launches` against `q/k/v stage, new: one fused kernel`.
  The new variant is `decode_rope_kv_write_f16`, which reads q, k and v in place from the fused projection and writes k and v into the store.
- `MLP activation, old: 2 copies and silu_mul` against `MLP activation, new: one kernel`.
  The new variant is `silu_mul_gu_f16`, which reads the gate and the up in place.

3 groups of variants time a kernel against its replacement inside a graph.
Each variant captures the kernel back to back, once for each layer or twice for the add and norm, and divides one replay by the calls.
The launch and the synchronisation of the individual timing are not in these numbers.

- `add and norm in a graph`: `fused_add_rmsnorm_f16` with the row cached in shared memory, against the kernel that holds one 16-byte vector of the row in registers for each thread.
- `q/k/v stage in a graph`: `decode_rope_kv_write_f16` with one block for each head, against `decode_q_rope_kv_write_f16` with one warp for each head and no shared memory.
- `MLP activation in a graph`: `silu_mul_gu_f16` with 2 columns a thread, against the kernel with 8 columns a thread and the fast exponential.

The replacements sum in another order or use the fast math units, so a value can differ from the old kernel in its last bits.
The card tests of `cuda_reduction_order_tests.rs` bound the gap.

A variant the kernel declines is a row without samples, and its note says so.
The bench times the fused stage with the head norms, as the old loop runs them.
The decode loop of a model uses the fused stage only for a layer without head norms.

When both kernels fit, a second graph holds the layer with them, 5 kernels in place of 14:

- `graph, new kernels` rows give the replay and the back-to-back replays, each noted against the graph of the old kernels.

The final lines:

- A table ranking the kernels by the time a step loses above their floor, `(median - floor)` times the calls of a step.
  A last column does the same with the median of the empty kernel taken off first.
- The sum of the medians of a layer, the layers over `--layers`, the once-a-step kernels, and the non-GEMM time of a step.
- The same with the graph replay of a layer in place of the sum of its medians.

The head matmul reads the whole vocabulary matrix, so it sets the memory the bench needs: `--vocab` times `--hidden` times 2 bytes.

### expert-ahead

`expert-ahead` copies the experts of the next layer to the card while the current layer runs.
It measures what a copy queued one layer ahead takes off the token, and what a wrong guess costs.

A token runs `--layers` layers.
In every layer all the routed experts are missing from the card, and the layer takes them through the transit room, 2 buffers that consecutive layers take turns over.
The copies come from page-locked memory on a copy stream.
The card runs a kernel that holds the compute stream, then answers its router on the host, then reads the experts from the transit room.
The read is a copy within card memory of the bytes the layer reads, so no expert arithmetic runs.

The sweep crosses 2 axes, each a column of the table:

- The experts missing per layer: 1, 2, 4 and 8.
- The card time of a layer before its router answers, as 0.5, 1 and 2 times the idle copy time of the layer's missing experts.

Its variants for each pair:

- No prefetch: the copies start once the router of the layer has answered.
- The last token's set, copied from the start of the layer before.
- The last token's set, copied once the layer before has crossed to the host.
- The early router: the same, with a router kernel run on the card for the next layer in the crossing of this one.

A guess names each routed expert with the probability `--guess-hit`, 1 variant each.
A wrong name copies an expert no layer reads.
The layer copies the experts the guess missed after its router, behind the guess on the copy stream.

The time of a call is the host clock around a whole token and its final synchronisation.
Each prefetching variant carries its median over the median without prefetch.
The bytes of a call are the copies a token queues, so the bandwidth shows the load a wrong guess puts on the bus.

The bench does not know how often a real guess is right.
It gives the hit rate from which a prefetch pays.
The hit rate itself comes from the routing of a served model.

### expert-read

`expert-read` reads the bytes of one expert from a file on the disk under test.  
`expert-read` drops their pages from the page cache before every call, and each call reads the next expert of the layer.

`expert-read` measures how best to read a missing expert.
Its variants:

- Page faults on first touch, by 1 thread and by `--threads` threads.
- A read-ahead advice on each projection followed by its touch, as the decode advises.
- The advice over the whole expert before any touch.
- A populating advice.
- A buffered read.
- A direct read into an aligned buffer, which bypasses the page cache.

### host-pairs

`host-pairs` times the routed pairs of 1 layer on the host cores, for a decoded token and for the rows of a verify.
It measures what a pair the host takes costs through the exact dots against the formats' own dots.  
It runs at the layer `expert-pairs` reads, with the same synthetic stacks and the same draw of experts.
It needs no card.

The stacks hold the layer's routed experts in the process's memory.
Consecutive calls take different routings, as many as the layer has experts, so a call reads its weights from memory.

Its variants:

- `1 pair` and each row count of `--block`, through the formats' own dots: the calls the tier makes for a layer without exact kernels, the quantisation of the input included.
- The same pairs through the exact dots, as the tier computes a cached layer's pairs.
- The gate of 1 expert and its down, each on 1 thread, through the format's own dot and through its exact dot.

Read `exact dots` against `own dots` at the same pairs.
The 1-thread rows set the dots of 1 format against each other, without the pool.


`host-expert` computes routed experts on the host cores.

`host-expert` measures what a missing expert costs once its pages are in memory.  
It runs at the dimensions of the shapes file, or of `--hidden` and `--inter`, in the formats `--gate-format` and `--down-format`.

`host-expert` touches no disk and no card.
Its variants:

- 1 expert, the same weights every call, so they stay in the caches.
- 4 experts per call out of 96 distinct ones, far more than the caches hold, so the weights come from memory.
  Each runs through the thread pool in 3 regions, 1 thread per expert, and one after the other on 1 thread.
- The same 4 experts on weights in the page cache of a mapped file, where a served model keeps them.

The formats are `iq2_xxs`, `q2_k`, `q4_1`, `q4_k`, `q5_0`, `q6_k` and `q8_0`.

### host-prompt

`host-prompt` runs the rows of a prompt through 1 Q5_0 projection on the host cores.
It runs `--rows` rows against `--hidden` inputs and `--inter` outputs.  
It touches no disk and no card.
Its variants:

- The tiled GEMM over the Q5_0 blocks, repacked 8 columns a group.
- The tiled Q8_0 GEMM over the same values held as Q8_0 blocks, which the served prompt runs.

The note of the second variant counts the outputs that differ from the first's.

### shortconv-prompt

`shortconv-prompt` runs the gated short convolution of one layer over `--rows` prompt rows on a card.
The width is the file's `embedding_length` and the taps its `shortconv.l_cache`, so it needs `--shapes-from`.
The rows are F16, the card's activation format.
Its variants:

- The token steps: 1 launch of the token kernel per row, then the rows concatenated.
- The whole sequence: 1 launch for every row, with the same rows and the same state after them.

### mamba2-prompt

`mamba2-prompt` runs the Mamba2 block of one layer over `--rows` prompt rows on a card, from the rows its input projection gives.
The dimensions are the file's `ssm.inner_size`, `ssm.state_size`, `ssm.group_count`, `ssm.conv_kernel` and `ssm.time_step_rank`, so it needs `--shapes-from`.
A file whose routed experts the shapes do not read takes `--tensor` with any tensor of the file, such as `token_embd.weight`.
Its variants:

- The token steps: the convolution, the time step, the recurrence and the gated norm launched for each row, then the rows concatenated.
- The whole sequence: 1 launch of each of the 4 kernels for every row, with the same rows and the same state after them.

### prefill-layer

`prefill-layer` runs the data movement of one layer of a prefill on synthetic F16 tensors.
It measures what the separate launches of the layer loop cost against 1 fused launch.  
It needs a card, and the heads from `--shapes-from` or from `--heads`, `--kv-heads` and `--head-dim`.

One call is one layer: the copies of q, k and v out of the projections, the norms of q and k, the rotation of both, and the write of k and v into a paged store.  
A call ends with a synchronisation of the card, so its time is the launches and the work together.

Each variant name ends with its count of launches, read off the code path.  
Its variants:

- Old, separate projections, with the norms of q and k: 10 launches.
- Old, one fused projection cut into q, k and v by copies, with the norms: 13 launches.
- Old, separate projections, without norms: 4 launches.
- New, separate projections, with the norms: 1 launch.
- New, one fused projection read in place, with the norms: 1 launch.
- New, separate projections, without norms: 1 launch.

Each new variant carries in its note its speed against the old variant of the same inputs, by the median.  
A new variant that does not fit the card or the widths says so in place of its figures.

2 more groups time the rest of the layer on the same tensors.

The activation group runs the feed-forward activation of a fused gate and up projection:

- Old, 2 narrow copies and `silu_mul_f16`: 3 launches.
- New, 1 kernel that reads the fused buffer: 1 launch.

The attention group runs the stage from the fused input launch to the attention output, over 1 sequence of all the rows:

- Old, the fused stage, the cast of q, the copy of q to heads first, the flash kernel and the cast of its output: 5 launches.
- New, the fused stage writes q in the flash kernel's layout and the flash kernel stores f16: 2 launches.

Each new variant carries its speed against the old one by the median.
It says so when the fused launch or the flash kernel does not fit the card or the widths.

The rotation base only sets the angles and does not change the time.  
Without a file it is 1.

### cache-policy

`cache-policy` replays a routing through the policies of a host tier and counts the experts each policy misses.
The disk then reads those experts.

`cache-policy` needs `--profile`, the routing counts a served model writes, and an expert size from `--shapes-from` or `--projection-bytes`.
It touches no model, no disk and no card.  
Its variants are the slots per layer in `--slots`, each under 3 policies:

- Static: the slots hold the experts the profile counts most and never change.
- Recency: the least recently routed expert makes room.
- Frequency: the least often routed expert makes room.

The cards hold the `--card-slots` hottest experts of each layer, and the tier is never asked for them.

Each token's routing is drawn from the counts alone.
The routing carries heat and no order, so a recency policy is judged without the locality a real answer adds.

The time per token is the misses read at `--disk-gbs`.

With `--file`, `cache-policy` replays a recorded trace in place of draws from a profile.
The trace is the list of events `/api/stage_perf` returns under `host_tier_trace` while the measure runs, gathered over the reads.
It holds what each layer kept when its trace began, then every forward in order with the experts the cards served and those the host computed.
A request starts at a prompt that follows a decode.
Its variants are the bound the trace recorded and each of `--slots`, under 5 rankings of the kept experts:

- recency, as the host tier ranks them,
- frequency, the rows routed to an expert since the trace began,
- decoded frequency, the decoded rows routed to it since the trace began,
- decoded first, an expert a prompt used last before one a decoded row used last,
- farthest next use, which reads the future: the floor no ranking passes.

Every expert a forward computes is admitted, then the lowest ranked give way, never one of a decoded row's own.
Each variant reports per request the decoded rows' misses a token, their time at `--disk-gbs` and the prompt's misses.
A first row reports what the trace recorded: the misses a token and the bytes not in memory when a decoded row asked for its experts.

### schedule-host

`schedule-host` times the host work of the continuous-batching worker between 2 decode steps, for `--batch` running sequences at `--context` tokens.
The sequences hold `--prompt-tokens` of prompt each, in blocks of `--block-size`, and sample as a client of the API does: temperature, top-k, top-p and a repeat penalty over a window.
It touches no model, no disk and no card.

Its variants are:

- Schedule and commit: `Scheduler::schedule` and `Scheduler::on_step_done` for the decoding rows. A sample is `--block-size` consecutive steps on a fresh scheduler, divided.
- Step items with clones: the items the engine builds from the plan, with the block table, the repeat window, the token and slot vectors and the sampling options copied.
- Step items with borrows: the same rows as references into the plan and the request, in a type local to the bench. A model taking them would need its signature changed.
- Decode inputs, indices: the vectors of `GhBatched::step` and of `decode_graph` up to the first card call. Token ids, positions, slots, sequence lengths, the batch padded to a power of 2 and the block tables padded and flattened.
- Decode inputs, rope rows: the cosine and sine rows of `rope_cos_sin` in f32 for the padded batch, sized by `--head-dim` and `--rope-base`.
- Host stretch: the sum of the medians of schedule and commit, items with clones and both decode inputs. The card is idle for this long between 2 decode steps.
- Engine step: `ContinuousBatchEngine::step_stream` over a model that answers at once, a cross-check of the sum above without the decode inputs.

### dequant-gemm

`dequant-gemm` finds the row count from which dequantising a 4-bit AWQ weight to f16 once and running a dense f16 GEMM beats the Marlin GEMM.
It runs on the 4 projections of one layer, with the weights, the shapes and the copies of `marlin-rows`.

Its variants, for each projection and each row count of `--rows-list`, are:

- `marlin`: `MarlinAwqLayer::forward_into`, the call `marlin-rows` times.
- `dequantise only`: `awq_dequant_f16_storage`, the AWQ prefill dequantisation, on the weight in the checkpoint layout. It writes a transient f16 weight `[k, n]` that is freed inside the timed call. It does not depend on the rows, so it is timed once per projection.
- `dense f16 only`: `Tensor::matmul` of an f16 `[rows, k]` input and the f16 `[k, n]` weight, with the weight already present. The output allocation is timed.
- `dequantise + dense f16`: the 2 above in sequence, including the transient allocation, as a prefill path runs them.

Each call is timed with the host clock around a stream synchronisation before and after it.

A table per projection gives the median time of each variant, the speed of `dequantise + dense f16` relative to `marlin`, and the TFLOPS of the dense GEMM, `2 * rows * k * n` over its time.
A speed above 1 means the pair is faster.

The last rows of the record and the final lines sum the 4 medians of a layer at each row count for `marlin` and for `dequantise + dense f16`.
The last line gives the row count from which the pair stays faster than Marlin at every larger row count measured, or `none up to <max>`.

### marlin-rows

`marlin-rows` times the 4-bit AWQ Marlin GEMM on the 4 projections of one layer, for a series of row counts.
It measures how the time of a call grows with the rows.

The launcher cuts a batch into tiles of `MAX_M_PER_CHUNK` rows.
Each tile reads the weights again, so a call makes `ceil(rows / MAX_M_PER_CHUNK)` passes over them.

The weights are synthetic, of the byte sizes the loader produces, repacked by the loader's own routine.
The call is the wrapper a served layer uses, on the stream of the card.  
Copies of each weight, together larger than twice the card's second-level cache, are called in turn, so every call streams its weights from card memory.

The projections come from `--hidden` or the shapes file, `--inter`, `--heads`, `--kv-heads` and `--head-dim`:

- Fused query, key and value: `hidden` to `(heads + 2 * kv_heads) * head_dim`.
- Output: `heads * head_dim` to `hidden`.
- Fused gate and up: `hidden` to `2 * inter`.
- Down: `inter` to `hidden`.

Its variants are the pairs of a projection and a row count of `--rows-list`.
A card event pair times each call.

Each variant moves the weight bytes times the passes, so its bandwidth is the rate at which the weights stream.
A table per projection gives the rows, the passes, the median time, the time per pass and that rate.

The last rows of the record sum the 4 medians of a layer at each row count.
The final line gives those sums and the ratio of the layer time at `--compare-rows` to the layer time at `MAX_M_PER_CHUNK` rows.
A ratio near the ratio of the passes says the time follows the passes over the weights.

### media-overlap

`media-overlap` times the denoise steps of 2 image requests on 1 card.
It answers whether 2 requests finish sooner together than one after the other.
It loads no model.

A step is `--layers` blocks of 4 dense f16 GEMMs on `--rows` tokens: the fused q, k and v projection, the output projection, and the feed-forward up and down.
`--hidden` sets the width and `--inter` the feed-forward length, 4 times the width when absent.
Each variant is timed from the launch of its work to the end of all of it:

- 1 step, the reference.
- 2 steps one after the other on the card's stream.
- 2 steps on 2 streams, from 2 devices opened on the card, each with its own weights and launched layer by layer in turn.
- 1 step of `2 * --rows` tokens, the 2 requests batched.

The outputs of a step are freed after the last synchronisation, so no free waits for the card inside the timed run.
The note of the last 2 variants gives the speed against the 2 steps one after the other.
Run it on an idle card: a second process on the card changes every figure.

On a GTX 1650 with `--hidden 1280`, the 2 streams finish in 0.93 to 0.97 of the serial time and the batched step in 0.97 to 0.99, from 128 to 4096 rows.
A step that fills the card leaves no room for the other.

### parler-step

`parler-step` times one decode step of a Parler-TTS decoder with its projections held in F32 and in F16.
It answers how much of a step the bytes of the weights decide.
It loads no model: the weights are deterministic values of the shapes given.

A step runs the embeddings, every layer, the final norm and the heads.
The logits row of each codebook is read back to the host, as a generation reads them.
A sample is 1 prompt pass, not timed, then timed steps whose median is reported per step.

`--hidden` sets the width, `--inter` the feed-forward length, `--layers` the layer count, `--heads` the attention heads, `--vocab` the audio tokens per codebook and `--codebooks` the codebooks.
`--prompt-tokens` sets the text prefix the first pass carries.
The note of each variant gives the rate at which the projections one step reads stream.

## Report

The heading names the bench, the build, the host, the card, and the shapes with their source.

Each variant gets 1 line:

- the timed calls
- the median time per call
- the 10th to 90th percentile
- the minimum and the maximum
- the outliers
- the bandwidth

An outlier is a call above `q3 + 1.5 iqr`.  
Outliers are counted and stay in every figure, so a variant whose maximum is far above its median has a stall worth a look.

The bandwidth is the bytes 1 call moves over the median call.  
A variant that could not run says why in place of its figures.

The JSON record carries the build, the machine, the card, the options as parsed, the shapes as resolved and the summary of every row.
2 records with the same options compare 2 commits.

Run a card bench on a card that serves nothing else.  
Run `expert-read` while no model is being read from the same disk.

## Add a bench

A bench is a module under `src/microbench/` with 1 function:

```rust
pub fn run(cx: &Context) -> Result<Vec<Measured>, String>
```

`cx.options` is the command line, `cx.shapes` the row and the expert, and `cx.card` the card opened when the bench asks for one.

The function builds its buffers once.  
For each variant it calls `stats::run_for` with the target wall time and a closure that answers the time of 1 call in microseconds.

`stats::timed` wraps a closure under the host clock.  
A variant whose work is on the card times it with events, as `expert-upload` does for its kernel.

Each variant is 1 `Measured`: a name, the bytes 1 call moves when it moves any, the samples, and a note when something is left out.

A new bench joins the list in `benches()` of `src/microbench/mod.rs`, with its name, what it measures and whether it needs a card.  
The card modules compile only with a GPU runtime, so a bench that needs none stays outside that condition.
