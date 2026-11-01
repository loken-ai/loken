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
cargo bench --features cuda,microbench --bench microbench -- expert-read --shapes-from /models/large.gguf --layer 10
```

A ROCm build takes `rocm` in place of `cuda`.  
A build without a GPU runtime omits `crossing` and `expert-upload`.

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
| `--inter` | The expert's feed-forward length, without a file. | 2048 |
| `--weight-bits` | Bits per stored weight, without a file. 4.5 for q4_k, 8.5 for q8_0. | 4.5 |
| `--projection-bytes` | The bytes of each projection, 1 value or 3, in place of `--inter` and `--weight-bits`. | derived |
| `--gate-format` | The format of an expert's gate and up weights, for `host-expert`. | iq2_xxs |
| `--down-format` | The format of an expert's down weights, for `host-expert`. | q2_k |
| `--pack` | Rows the packed variants of `crossing` move in one copy. | 8 |
| `--file` | The file `expert-read` reads. | the shapes file |
| `--offset` | Where the first expert read starts, when the file is not the shapes file. | 0 |
| `--threads` | Threads of the threaded variants. | the processors the host reports |
| `--card` | The card the card benches run on. | 0 |
| `--seconds` | Wall time of each variant. A warm-up of a tenth of it comes first. | 1.0 |
| `--json` | Where the JSON record is written. | none |
| `--profile` | The expert profile `cache-policy` draws its routing from. | none |
| `--card-slots` | The hottest experts per layer the cards hold, which the host tier never serves (`cache-policy`). | 58 |
| `--slots` | Slots per layer the host tier is tried with, 1 variant each (`cache-policy`). | 94,130,170 |
| `--guess-hit` | The share of a token's experts the read-ahead guess names, 1 variant each (`cache-policy`). | 1.0 |
| `--block` | Tokens verified together, a draft's block. The block reads its experts once. 1 variant each (`cache-policy`). | 1 |
| `--accept` | The share of a draft token accepted once the token before it was (`cache-policy`). | 0.8 |
| `--tokens` | Tokens routed per variant (`cache-policy`). | 20000 |
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

### expert-upload

`expert-upload` copies the 3 projections of one routed expert from the host to the card.
It measures a synchronous upload inside a token and what overlap buys.  
Its variants cross 3 choices:

- The compute stream, a copy stream, or a copy stream ordered after an event recorded on the compute stream, which is how the uploader orders its allocation.
- Pageable or page-locked memory.
- The card idle, or a kernel holding the compute stream.

The kernel runs as long as 1 idle upload from page-locked memory on the compute stream.  
Each busy variant gets a second row: the kernel's own time beside that upload, against its time alone.

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

### host-expert

`host-expert` computes routed experts on the host cores.

`host-expert` measures what a missing expert costs once its pages are in memory, and what the decode's host path adds to the bare products.  
It runs at the dimensions of the shapes file, or of `--hidden` and `--inter`, in the formats `--gate-format` and `--down-format`.

`host-expert` touches no disk and no card.
Its variants:

- 1 expert, the same weights every call, so they stay in the caches.
- 4 experts per call out of 96 distinct ones, far more than the caches hold, so the weights come from memory.
  Each runs through the thread pool in 3 regions, 1 thread per expert, and one after the other on 1 thread.
- The same 4 experts on weights in the page cache of a mapped file, where a served model keeps them.
- 4 and 6 experts per call through the function a decode calls for its host experts, on experts built as the store builds them.

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
