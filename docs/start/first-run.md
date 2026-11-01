# First run

Start the daemon, pull a model and get one answer back.

## Before you start

<!-- site: widget checklist id=first-run-prereq -->
- [ ] `lokend` and `loken` are built, see [Install](install.md).
- [ ] The machine can reach the model registry to fetch a model.
<!-- site: end -->

## Run it

<!-- site: widget steps id=first-run-steps -->
1. Start the daemon.
   It listens on port 11435, set by `[server] port` or `--port`.

   ```sh
   lokend serve
   ```

2. Check it is up from another terminal.
   `/health` answers without a key.

   ```sh
   curl -s localhost:11435/health
   ```

3. Pull a model.
   A name without `/` comes from the Ollama registry, one with `/` from Hugging Face.

   ```sh
   loken pull qwen3:0.6b
   ```

4. List the models the daemon holds.

   ```sh
   loken list
   ```

5. Send 1 request.
   The first one loads the model.

   ```sh
   loken chat --model qwen3:0.6b "Say hello in 6 words."
   ```
<!-- site: end -->

## Expected result

<!-- site: widget replay id=first-run-replay -->
```
$ curl -s localhost:11435/health
{"status":"healthy","service":"loken","loaded_models":0, ...}
$ loken list
Downloaded models:
  qwen3:0.6b - <size> (modified: <date>)
$ loken chat --model qwen3:0.6b "Say hello in 6 words."
Assistant: <the answer>

[Tokens: <count>]
```
<!-- site: end -->

`/health` also reports `version`, `uptime_seconds` and the engines loaded.  
`loken ps` lists the models resident in memory.

An OpenAI client reaches the same model at `http://localhost:11435/v1`, see [Serve a chat model to an OpenAI client](../guides/openai-sdk.md).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Make sure the server is running` from `loken load` or `loken ps`, or a connection error from another command | No daemon at the address `loken` uses, `http://localhost:11435` by default. | Start `lokend serve`, or pass `--server <url>` to `loken`. |
| `Address already in use` from `lokend` | Another process holds the port. | `lokend serve --port <other>`, and pass the same address to `loken --server`. |
| `Model not found` | Neither store holds the name. | `loken pull <name> --source ollama` or `--source huggingface`. A Hugging Face id needs its owner. |
| `Model '...' not found in remote registry` | The registry chosen by `--source` does not know the name. | Check the tag on that registry, or use the other source. |
| 401 `missing or invalid API key` | `require_auth` is on. | Send `Authorization: Bearer <key>` or `X-API-Key` with a key from `api_keys`. |
| `loken launch` fails to authenticate | The daemon needs a key. | Set `LOKEN_API_KEY` before `loken launch`. |
