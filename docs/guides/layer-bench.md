# Measure a decode on a few layers with the layer bench

A served decode of a large model loses time in places that show at the scale of one layer: an
expert uploaded on the compute stream, a read advised one projection at a time, an activation
crossing the bus once per projection. Finding them on the whole model costs a build and a load
of the whole file for every hypothesis. The layer bench holds a few layers of the file instead,
places them as the server places the whole, replays decode steps through them and reports the
same stage profile the server keeps, with the counters a hypothesis needs.

The bench is a cargo bench behind the `microbench` feature, so no release build carries it. It starts no server and loads no other model.

## Run

```sh
cargo bench --features cuda,microbench --bench layerbench -- /models/large.gguf --layers 10..13 --tokens 64 --json results/layers-10-13.json
```

The arguments:

| Argument | What it sets | Default |
|---|---|---|
| `file` | The model file. The parts of a split file are found beside it. | required |
| `--layers` | The layers held, `from..to` with the end excluded, or one index. | required |
| `--tokens` | Decode steps timed. | 32 |
| `--warm` | Decode steps run before the timed ones. | 4 |
| `--placement` | `auto` as served, `host` with no card, `cards` with no host tier. | `auto` |
| `--page-cache` | `warm` reads the held layers' experts in first, `cold` drops them first. | `warm` |
| `--readahead` | The expert read-ahead depth in layers. Without it the model's controller sets it. | controller |
| `--card-experts` | Routed experts per held layer the cards keep. Without it each layer gets the share a served layer gets. | served share |
| `--guess-hit` | The share of a token's experts its read-ahead guess names, 0 to 1. | 1.0 |
| `--seed` | The seed of the synthetic stream and of the routing drawn. | 1 |
| `--expert-profile` | A profile whose counts give the routing heat, in place of the file's prior. | none |
| `--json` | Where the JSON record is written. | none |

A range starts at a layer whose attention reads no state from a layer below it: layer 0, a
layer without compression, or a layer that publishes the compressed keys. A range that starts
elsewhere is refused, and the refusal lists the layers a range may start at.

## What runs

The layers are read from the file's mapping alone. The embedding, the final norm and the
output head are loaded whatever the range, so a range that ends the stack runs to logits.

The placement is the one a served model gets. The weights every token reads are set aside on
the cards as if the whole stack were held. The room for routed experts is the held layers'
share of what the whole stack's experts would get, so a held layer keeps on the cards what it
keeps in service. The host tier is sized as the server sizes it, the memory left divided over
every layer of the stack. `--placement host` opens no card. `--placement cards` keeps no host
tier, so an expert off the cards is read from the file.

Each step starts from a synthetic stream when the range starts inside the stack, or from a
token drawn from the vocabulary when it starts at layer 0. The routing is scripted: for every
step and layer, the experts are drawn from the file's routing prior, or from the profile's
counts when one is given, and the gate takes them in place of its own choice. The weights stay
the gate's. The read-ahead guess of the next layer is scripted the same way, and `--guess-hit`
sets how many of the token's experts it names. The read-ahead, the host tier, the card lanes
and the deferral of host experts run as they do in service.

## Read the report

The table starts with the run: the layers held, the placement obtained, the experts the cards
hold per layer and the host tier's slots, and where the routing heat came from. Then the time
per token over the timed steps.

The stages are the server's stage profile, by time: milliseconds per token, calls per token,
and the share of the stages' total. A stage name is the one `/api/stage_perf` reports.

The steps at the card seam are what the layers offered a card, by kind. A step taken ran on the
card and returned its answer before the layer went on, so each taken step is one wait on the
card's stream. A declined step ran on the host. The bytes are those handed to the card and
those read back, per token.

The counters are the expert uploads to the cards, the host tier's hits, misses and evictions,
the major page faults, and the bytes read from disk by the process, over the timed steps.

The JSON record carries the build (version and commit), the machine (host, processor, memory,
cards), the controls, the model as held and placed, the time of every token, the stages and the
counters. 2 records with the same controls compare 2 commits.

## Compare 2 commits

Build each commit, run the same command with `--json` on each, and read the 2 records side
by side. The seed fixes the routing and the stream, so the experts read are the same on both.
Run the warm page cache before the cold one when both are wanted: a cold run leaves the experts
it read in memory.
