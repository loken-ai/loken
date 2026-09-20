# Documentation

Read in this order.

1. The [feature matrix](../README.md#what-it-does) in the README: each capability, its route,
   its configuration key and its state.
2. [`guides/`](guides/): one page per task, each with the smallest configuration, one request
   and what to expect.
3. [`learn/`](learn/): the fundamentals of inference in ten lessons, each with what this
   tree does about it, what was measured, and one thing to run.
4. [`MODELS.md`](MODELS.md): the families served, the stores they are read from, how a request
   names them.
5. [`CONFIGURATION.md`](CONFIGURATION.md): every key of `config.toml` and its default.
6. [`API.md`](API.md): every route on the three surfaces.
7. [`BUILDING.md`](BUILDING.md): the toolchains, the builds without CUDA, the daemon's flags,
   troubleshooting.
8. [`STATUS.md`](STATUS.md): what is measured, what is slower, what is written and wired to
   nothing. Read it before relying on a feature.
9. [`BENCHMARKS.md`](BENCHMARKS.md): the measurements and the protocol behind them.
10. [`CLUSTER.md`](CLUSTER.md): serving from more than one machine.
11. [`REFERENCES.md`](REFERENCES.md): the papers and implementations this work draws on.

## Learn

| Lesson | Question |
|---|---|
| [`learn/README.md`](learn/README.md) | The series: ten lessons, ten minutes each, one exercise per lesson |
| [`learn/slides.html`](learn/slides.html) | The same series as a deck |

## Guides

| Guide | Task |
|---|---|
| [`guides/openai-sdk.md`](guides/openai-sdk.md) | Serve a chat model to an OpenAI client |
| [`guides/ollama-clients.md`](guides/ollama-clients.md) | Point an Ollama client at loken |
| [`guides/agents.md`](guides/agents.md) | Run Claude Code or Cline on loken |
| [`guides/images.md`](guides/images.md) | Generate an image |
| [`guides/speech.md`](guides/speech.md) | Transcribe and synthesise speech |
| [`guides/drafter.md`](guides/drafter.md) | Speed up a model that spills to the host with a drafter |
| [`guides/kv-memory.md`](guides/kv-memory.md) | Keep conversations in memory between requests |
| [`guides/cluster.md`](guides/cluster.md) | Two machines |
