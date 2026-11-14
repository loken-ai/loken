# Documentation

The same pages are served at <https://loken-ai.github.io/loken/>.

| Page | Read it to |
|---|---|
| [Overview](https://loken-ai.github.io/loken/overview/) | See what loken serves and in which state |
| [Install](https://loken-ai.github.io/loken/install/) | Build the 2 binaries |
| [First run](https://loken-ai.github.io/loken/first-run/) | Start the daemon, pull a model and get an answer |
| [Choose a model](https://loken-ai.github.io/loken/choose-a-model/) | Pick a family, a size and a quantisation that fit your memory |
| [Concepts](https://loken-ai.github.io/loken/concepts/) | Learn the terms of the guides and how a request reaches a model |
| [Serve a chat model to an OpenAI client](https://loken-ai.github.io/loken/openai-sdk/) | Serve a chat model to an OpenAI client |
| [Point an Ollama client at loken](https://loken-ai.github.io/loken/ollama-clients/) | Point an Ollama client at loken |
| [Run a coding agent on loken](https://loken-ai.github.io/loken/agents/) | Run Claude Code or Cline on loken |
| [Embed texts and images](https://loken-ai.github.io/loken/embeddings/) | Embed texts and images |
| [Generate an image](https://loken-ai.github.io/loken/images/) | Generate an image |
| [Transcribe and synthesise speech](https://loken-ai.github.io/loken/speech/) | Transcribe and synthesise speech |
| [Speed up a model that spills to the host with a drafter](https://loken-ai.github.io/loken/drafter/) | Speed up a model that spills to the host with a drafter |
| [Keep conversations in memory between requests](https://loken-ai.github.io/loken/kv-memory/) | Keep conversations in memory between requests |
| [Run 2 machines](https://loken-ai.github.io/loken/cluster/) | Serve from 2 machines |
| [Expose loken on a network](https://loken-ai.github.io/loken/network/) | Expose loken on a network with a key and a rate limit |
| [Troubleshoot](https://loken-ai.github.io/loken/troubleshooting/) | Find the cause of a failure by its message |
| [Map of lokend](https://loken-ai.github.io/loken/map/) | See how lokend serves a request, part by part, one figure each |
| [Course: how inference works](https://loken-ai.github.io/loken/course/) | Follow the series of 10 lessons, one exercise each |
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
| [`inside/engines.md`](inside/engines.md) | Compare how loken, vLLM and Ollama place a model, cards, machines and requests |
| [`inside/streamed.md`](inside/streamed.md) | Follow a mixture larger than the server from the disk to the card |

The [benchmarks](https://github.com/loken-ai/benchmarks) repository holds the campaign, its method and its results, published at <https://loken-ai.github.io/benchmarks/>.
