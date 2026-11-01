# Documentation

The same pages are served at <https://loken-ai.github.io/loken/>.

| Page | Read it to |
|---|---|
| [`start/overview.md`](start/overview.md) | See what loken serves and in which state |
| [`start/install.md`](start/install.md) | Build the 2 binaries |
| [`start/first-run.md`](start/first-run.md) | Start the daemon, pull a model and get an answer |
| [`start/choose-a-model.md`](start/choose-a-model.md) | Pick a family, a size and a quantisation that fit your memory |
| [`start/concepts.md`](start/concepts.md) | Learn the terms of the guides and how a request reaches a model |
| [`guides/openai-sdk.md`](guides/openai-sdk.md) | Serve a chat model to an OpenAI client |
| [`guides/ollama-clients.md`](guides/ollama-clients.md) | Point an Ollama client at loken |
| [`guides/agents.md`](guides/agents.md) | Run Claude Code or Cline on loken |
| [`guides/embeddings.md`](guides/embeddings.md) | Embed texts and images |
| [`guides/images.md`](guides/images.md) | Generate an image |
| [`guides/speech.md`](guides/speech.md) | Transcribe and synthesise speech |
| [`guides/drafter.md`](guides/drafter.md) | Speed up a model that spills to the host with a drafter |
| [`guides/kv-memory.md`](guides/kv-memory.md) | Keep conversations in memory between requests |
| [`operate/cluster.md`](operate/cluster.md) | Serve from 2 machines |
| [`operate/network.md`](operate/network.md) | Expose loken on a network with a key and a rate limit |
| [`operate/troubleshooting.md`](operate/troubleshooting.md) | Find the cause of a failure by its message |
| [`inside/map.md`](inside/map.md) | See how lokend serves a request, part by part, one figure each |
| [`learn/README.md`](learn/README.md) | Follow the series of 10 lessons, one exercise each |
| [`MODELS.md`](MODELS.md) | Find the families served, their stores, and how a request names them |
| [`CONFIGURATION.md`](CONFIGURATION.md) | Look up a key of `config.toml` and its default |
| [`API.md`](API.md) | Look up a route on the 3 surfaces |
| [`BUILDING.md`](BUILDING.md) | Pick a toolchain, build without CUDA, read the daemon's flags, fix a failed build |
| [`CLUSTER.md`](CLUSTER.md) | Look up the `[cluster]` keys, the routes and how a request is routed |
| [`STATUS.md`](STATUS.md) | See what is measured, what is slower, and what has never run |
| [`ARCHITECTURES.md`](ARCHITECTURES.md) | List the architectures the loader accepts and the formats of each |
| [`REFERENCES.md`](REFERENCES.md) | Find the papers and implementations each model was written against |
| [`dev/layer-bench.md`](dev/layer-bench.md) | Measure a decode on a few layers of a large model |
| [`dev/microbench.md`](dev/microbench.md) | Time one mechanism at a model's shapes, variants side by side |
| [`dev/code-map.md`](dev/code-map.md) | Find the code behind each part of lokend and the plan levels it reads |

The [benchmarks](https://github.com/loken-ai/benchmarks) repository holds the campaign, its method and its results, published at <https://loken-ai.github.io/benchmarks/>.
