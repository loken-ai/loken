# Serve a chat model to an OpenAI client

## Configuration

```toml
[server]
host = "127.0.0.1"
port = 11435

[inference]
model_id = "qwen3:0.6b"
```

`model_id` is the model a request gets when it names none. Fetch it first:

```sh
loken pull qwen3:0.6b --source ollama
lokend serve
```

## One request

```sh
curl -s localhost:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3:0.6b",
  "messages": [{"role": "user", "content": "Say hello in six words."}]
}'
```

The answer has the OpenAI shape: `choices[0].message.content`, `finish_reason`, `usage`. A
reasoning model returns its thinking under `reasoning_content`, apart from `content`.

## With an SDK

Point the client at the daemon; the key is anything until `require_auth` is on.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11435/v1", api_key="unused")
r = client.chat.completions.create(model="qwen3:0.6b",
                                   messages=[{"role": "user", "content": "Hello"}])
print(r.choices[0].message.content)
```

Streaming, tools, `response_format`, `n`, `logprobs` and `logit_bias` work as the API documents
them; `/v1/responses` serves the Responses API the newer SDKs default to. The routes and what
each takes are in [`../API.md`](../API.md#openai-surface).

## What to expect

The first request pays the load. `GET /api/ps` shows what is resident, and every answer carries
the engine's timings on the Ollama surface (`eval_count`, `eval_duration`) if you want the
decode rate.
