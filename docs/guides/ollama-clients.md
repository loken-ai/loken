# Point an Ollama client at loken

loken serves the Ollama API on its own port, so a client that talks to Ollama talks to loken by
changing the address.

```sh
export OLLAMA_HOST=http://localhost:11435
ollama list
ollama run qwen3:0.6b
```

`loken pull` and `ollama pull` fill the same store layout, so a model pulled by either is
served by loken:

```sh
loken pull qwen3:0.6b --source ollama
```

## What matches

`/api/chat`, `/api/generate`, `/api/embed`, `/api/tags`, `/api/show`, `/api/ps`, `/api/pull`,
`/api/create`, `/api/copy`, `/api/delete` answer in Ollama's shapes, `keep_alive` included. A
model's template, system prompt and parameters come from its manifest, as in Ollama. `think`
returns the reasoning apart from the answer, and `/api/version` names the API level served.

## What differs

`/api/push` is refused: nothing is pushed to a registry. A `num_ctx` above the configured
context reloads the model at that context, clamped to what the checkpoint declares. Energy per
request is reported beside the timings.

The routes and their fields are in [`../API.md`](../API.md#ollama-surface).
