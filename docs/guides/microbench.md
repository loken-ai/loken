# Time one mechanism with a microbench

A decode loses time in mechanisms that are each a few microseconds to a few milliseconds: a
row crossing the bus, an expert uploaded to a card, an expert read from the disk. Measuring one
of them on a served model costs a load of the whole file for every hypothesis. A microbench
isolates one mechanism instead. It runs on synthetic data with the real shapes, loads no
weight and starts no server, and runs its variants side by side in the same process, so one
run settles one hypothesis.

The benches are a bench target of the crate, built only with the `microbench` feature. No
release build enables it and nothing of it reaches the daemon.

## Run

```sh
cargo bench --features cuda,microbench --bench microbench -- list
cargo bench --features cuda,microbench --bench microbench -- crossing --shapes-from /models/large.gguf
cargo bench --features cuda,microbench --bench microbench -- expert-upload --shapes-from /models/large.gguf --json results/upload.json
cargo bench --features cuda,microbench --bench microbench -- expert-read --shapes-from /models/large.gguf --layer 10
```

A ROCm build takes `rocm` in place of `cuda`. A build without a runtime lists `expert-read`
alone.

The arguments after `--`:

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
| `--projection-bytes` | The bytes of each projection, one value or three, in place of the two above. | derived |
| `--pack` | Rows the packed variants of `crossing` move in one copy. | 8 |
| `--file` | The file `expert-read` reads. | the shapes file |
| `--offset` | Where the first expert read starts, when the file is not the shapes file. | 0 |
| `--threads` | Threads of the threaded variants. | the processors the host reports |
| `--card` | The card the card benches run on. | 0 |
| `--seconds` | Wall time each variant runs for, after a warm-up of a tenth of it. | 1.0 |
| `--json` | Where the JSON record is written. | none |

With `--shapes-from`, the row is the file's hidden size times its hyper-connection multiple.
The expert is the layer's three routed projections, cut from a stacked tensor or taken from
one tensor per expert, with the offset of every expert's copy in the file. Without a file, the
options give the row and the projections' bytes.

A bench runs for a few seconds: one target wall time per variant, plus the warm-up.

## The benches

`crossing` moves one activation row host to card and back. It answers what one host-card
crossing costs, the unit behind a layer's count of crossings. Its variants:

- One copy each way from pageable memory with a stream synchronisation, as a decode step
  crosses.
- The same from page-locked memory, queued and waited on with an event.
- The round trips, pageable and page-locked.
- `--pack` rows in separate copies, against the same rows packed in one copy each way.
- A stream synchronisation and an event wait with nothing queued.

`expert-upload` copies one routed expert's three projections host to card. It answers the
cost of a synchronous upload inside a token and what overlap buys. Its variants cross three
choices:

- The compute stream, a copy stream, or a copy stream ordered after an event recorded on the
  compute stream, which is how the uploader orders its allocation.
- Pageable or page-locked memory.
- The card idle, or a kernel holding the compute stream.

The kernel runs as long as one idle upload from page-locked memory on the compute stream.
For each busy variant a second row gives the kernel's own time beside that upload, against
its time alone.

`expert-read` reads one expert's bytes from a file on the disk under test, their pages
dropped from the page cache before every call. Each call reads the next expert of the layer.
It answers how a missing expert is best read. Its variants:

- Page faults on first touch, by one thread and by `--threads` threads.
- A read-ahead advice on each projection followed by its touch, as the decode advises.
- The advice over the whole expert before any touch.
- A populating advice.
- A buffered read.
- A direct read into an aligned buffer, which bypasses the page cache.

`cache-policy` replays a routing through the policies of a host tier and counts the experts each
misses, which the disk then has to read. It needs `--profile`, the routing counts a served model
writes, and an expert size from `--shapes-from` or `--projection-bytes`. It touches no model, no
disk and no card. Its variants are the slots per layer in `--slots`, each under three policies:

- Static: the slots hold the experts the profile counts most and never change.
- Recency: the least recently routed expert makes room.
- Frequency: the least often routed expert makes room.

The experts the cards hold (`--card-slots`, the hottest of each layer) are never asked of the tier.
The routing is drawn for each token from the counts alone, so it carries the heat of the experts and
no order: a recency policy is judged without the locality a real answer adds. The time per token is
the misses read at `--disk-gbs`.

## Read the report

The heading names the bench, the build, the host and the card, then the shapes and where they
came from.

One line per variant: the timed calls, the median time per call, the 10th to 90th percentile,
the minimum, the maximum, the outliers and the bandwidth. An outlier is a call above
`q3 + 1.5 iqr`. Outliers are counted and stay in every figure, so a variant whose maximum is
far above its median has a stall worth a look. The bandwidth is the bytes one call moves over
the median call. A variant that could not run says why in place of its figures.

The JSON record carries the build, the machine, the card, the options as parsed, the shapes
as resolved and every row's summary. Two records with the same options compare two commits.

Run a card bench on a card that serves nothing else. Run `expert-read` while no model is being
read from the same disk.

## Add a bench

A bench is a module under `src/microbench/` with one function:

```rust
pub fn run(cx: &Context) -> Result<Vec<Measured>, String>
```

`cx.options` is the command line, `cx.shapes` the row and the expert, and `cx.card` the card
opened when the bench asks for one. The function builds its buffers once, then for each
variant calls `stats::run_for` with the target wall time and a closure that answers one call's
time in microseconds. `stats::timed` wraps a closure under the host clock. A variant whose work
is on the card times it with events, as `expert-upload` does for its kernel. Each variant is
one `Measured`: a name, the bytes one call moves when it moves any, the samples, and a note
when something is left out.

A new bench joins the list in `benches()` of `src/microbench/mod.rs`, with its name, what it
answers and whether it needs a card. The card modules are compiled with a GPU runtime alone,
so a bench that needs none stays outside that condition.
