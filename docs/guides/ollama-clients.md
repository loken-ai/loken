# Point an Ollama client at loken

An Ollama client talks to loken by changing the address.

## Before you start

<!-- site: widget checklist id=ollama-clients-prereq -->
- [ ] `lokend` is running.
- [ ] An Ollama client is installed: the `ollama` command, or an application that speaks the Ollama API.
- [ ] A model is in the store. `loken pull` and `ollama pull` fill the same layout, so a model pulled by either is served.
<!-- site: end -->

## Smallest configuration

Set the address of the daemon in the client.

```sh
export OLLAMA_HOST=http://localhost:11435
```

## Run it

<!-- site: widget tabs group=client id=ollama-clients-run -->
### ollama

```sh
ollama list
ollama run qwen3:0.6b
```

### curl

```sh
curl -s localhost:11435/api/chat -d '{
  "model": "qwen3:0.6b",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": false
}'
```
<!-- site: end -->

Pull a model with loken when the store is empty:

```sh
loken pull qwen3:0.6b --source ollama
```

## Expected result

`ollama list` prints the models of the store. `ollama run` answers in the terminal.

## What matches

| Route | Behaviour |
|---|---|
| `/api/chat`, `/api/generate`, `/api/embed`, `/api/tags`, `/api/show`, `/api/ps`, `/api/pull`, `/api/create`, `/api/copy`, `/api/delete` | Answer in Ollama's shapes, `keep_alive` included. |
| `/api/version` | Names the API level served. |

A model's template, system prompt and parameters come from its manifest.
`think` returns the reasoning apart from the answer.

## What differs

| Request | loken |
|---|---|
| `/api/push` | Refused with 403. |
| `num_ctx` above the configured context | Reloads the model at that context, clamped to what the checkpoint declares. |
| Timings | Energy per request is reported beside them. |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| 403 `loken does not support pushing models to a remote registry` | `/api/push` is refused. | Drop the push step. loken keeps models on the machine that holds them. |
| 401 `missing or invalid API key` | `require_auth` is on. | Send an accepted key from `api_keys` as `Authorization: Bearer <key>`. |
| 404 `Model '<name>' not found in remote registry` | `ollama pull` or `/api/pull` named a model the registry does not hold. | Check the name and tag. |

## Next

- [Routes and fields of the Ollama surface](../API.md#ollama-surface)
- [Serve a chat model to an OpenAI client](openai-sdk.md)
