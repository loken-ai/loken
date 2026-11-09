# Layer bench

The layer bench replays decode steps through a few layers of a model file.
It places them as the server places the whole model.  
It reports the stage profile the server keeps, with the counters a hypothesis needs.

Time lost in a served decode shows at the scale of 1 layer: an expert uploaded on the compute stream, a read advised 1 projection at a time, an activation crossing the bus once per projection.

Finding that time on the whole model costs a build and a load of the whole file per hypothesis.

The bench is `benches/layerbench.rs`, behind the `microbench` feature.
No release build carries it.  
It starts no server and loads no other model.

## Run

```sh
cargo bench --features cuda,microbench --bench layerbench -- /models/large.gguf --layers 10..13 --tokens 64 --json results/layers-10-13.json
```

| Argument | What it sets | Default |
|---|---|---|
| `file` | The model file. The parts of a split file are found beside it. | required |
| `--layers` | The layers held, `from..to` with the end excluded, or 1 index. | required |
| `--tokens` | Decode steps timed. | 32 |
| `--warm` | Decode steps run before the timed ones. | 4 |
| `--placement` | `auto` as served, `host` with no card. `cards` is refused: the generic route keeps the host tier a served load keeps. | `auto` |
| `--page-cache` | `warm` reads the held layers' experts in first, `cold` drops them first. | `warm` |
| `--readahead` | Refused: the generic route has no read-ahead depth to pin for the experts. | none |
| `--card-experts` | Routed experts per held layer the first card keeps, as the room it gives them. | what the card has free |
| `--guess-hit` | The share of a token's experts its read-ahead guess names, 0 to 1. Unused: the generic route makes no guess. | 1.0 |
| `--seed` | The seed of the synthetic stream and of the routing drawn. | 1 |
| `--expert-profile` | A profile whose counts give the routing heat. | the file's prior |
| `--json` | Where the JSON record is written. | none |

A range must start at a layer whose attention reads no state from a layer below it.
Those layers are layer 0, a layer without compression, and a layer that publishes the compressed keys.

A range that starts elsewhere is refused, and the refusal lists the layers a range may start at.

## What runs

The layers are loaded through the generic route, as the engine loads a share of the stack.
The file is read through its mapping, so the experts of other layers are never read.
The embedding, the final norm and the output head are loaded whatever the range, so a range that ends the stack runs to logits.

### Placement

The placement is the one the engine gives a model:

- The mixers of the held layers are the blocks the cards hold, as the plan of the engine sizes them.
- The routed experts stay on the host.
  After the load, each card that pays keeps the hottest experts in the room it has left, as in service.
- `--placement host` opens no card.

The routing prior is the file's, or the one a companion file beside it declares.
A profile given with `--expert-profile` ranks the experts in its place.

### Steps

Each step starts from a synthetic stream when the range starts inside the stack.
It starts from a token drawn from the vocabulary when the range starts at layer 0.

The routing is scripted.
For every step and layer, the experts are drawn from the file's routing prior, or from the profile's counts when one is given.
The router takes them in place of its own choice, and its weights stay the router's.

## Report

The table starts with the run:

- the layers held
- the placement obtained
- the experts the cards hold per layer, and the host tier's slots (0 when the host holds every expert)
- where the routing heat came from

Then it gives the time per token over the timed steps.

The stages are the server's stage profile, by time: milliseconds per token, calls per token, and the share of the stages' total.
A stage name is the one `/api/stage_perf` reports.
The generic route reports the token mixer and the experts of each layer.

The record has a section for the steps at the card seam, by kind.
The generic route has no such seam, so the section is empty.

The counters, over the timed steps:

- the expert uploads to the cards (a stage the generic route does not record)
- the host tier's hits, misses and experts let go
- the major page faults
- the bytes the process read from disk

The JSON record carries the build (version and commit), the machine (host, processor, memory, cards), the controls, the model as held and placed, the time of every token, the stages and the counters.
2 records with the same controls compare 2 commits.

## Compare 2 commits

1. Build each commit.
2. Run the same command with `--json` on each.
3. Read the 2 records side by side.

The seed fixes the routing and the stream, so both commits read the same experts.  
Run the warm page cache before the cold one when you want both, because a cold run leaves the experts it read in memory.
