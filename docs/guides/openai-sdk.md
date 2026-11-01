# Serve a chat model to an OpenAI client

An OpenAI SDK or `curl` talks to loken by changing the base URL.

## Before you start

<!-- site: widget checklist id=openai-sdk-prereq -->
- [ ] `lokend` and `loken` are built.
- [ ] An OpenAI SDK, or `curl`, is installed on the client.
<!-- site: end -->

## Smallest configuration

```toml
[server]
host = "127.0.0.1"
port = 11435

[inference]
model_id = "qwen3:0.6b"
```

`model_id` is the model a request gets when it names none.

## Run it

<!-- site: widget steps id=openai-sdk-start -->
1. Start the daemon.

   ```sh
   lokend serve
   ```

2. Fetch the model from another terminal.

   ```sh
   loken pull qwen3:0.6b --source ollama
   ```
<!-- site: end -->

Send a chat completion.
The API key is any string until `require_auth` is on.

<!-- site: widget tabs group=client id=openai-sdk-client -->
### curl

```sh
curl -s localhost:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3:0.6b",
  "messages": [{"role": "user", "content": "Say hello in 6 words."}]
}'
```

### Python

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11435/v1", api_key="unused")
r = client.chat.completions.create(model="qwen3:0.6b",
                                   messages=[{"role": "user", "content": "Hello"}])
print(r.choices[0].message.content)
```
<!-- site: end -->

## Expected result

The answer has the OpenAI shape: `choices[0].message.content`, `finish_reason` and `usage`.  
A reasoning model returns its thinking under `reasoning_content`, apart from `content`.

The first request loads the model.
`GET /api/ps` lists what is resident.  
Every answer on the Ollama surface carries the engine's timings (`eval_count`, `eval_duration`) for the decode rate.

## Supported features

| Feature | Detail |
|---|---|
| Streaming, tools, `response_format`, `n`, `logprobs`, `logit_bias` | Work as the API documents them. |
| `/v1/responses` | Serves the Responses API, the default of the newer SDKs. |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| 401 `missing or invalid API key` | `require_auth` is on and the request carries no accepted key, or `api_keys` is empty. | Add the key to `api_keys`. Send it as `Authorization: Bearer <key>` or `X-API-Key: <key>`. |

## Next

- [Routes and fields of the OpenAI surface](../API.md#openai-surface)
- [Server keys, `require_auth` and `api_keys`](../CONFIGURATION.md)
- [Point an Ollama client at loken](ollama-clients.md)
