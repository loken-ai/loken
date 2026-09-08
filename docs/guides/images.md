# Generate an image

Needs a build with the `image` feature, which the default build has.

## Files

The `model` of a request names the family by substring, and the loader reads that family's
files from the Hugging Face store. For Z-Image:

```sh
loken pull Tongyi-MAI/Z-Image-Turbo --source huggingface
```

The families and their files are listed in [`../MODELS.md`](../MODELS.md#image).

## One request

```sh
curl -s localhost:11435/v1/images/generations -H 'Content-Type: application/json' -d '{
  "model": "z-image",
  "prompt": "a lighthouse at dusk, oil painting",
  "size": "1024x1024"
}'
```

The picture comes back base64 under `data[0].b64_json`. `response_format: "url"` stores it as
a file and answers a URL on the request's `Host`, or on `public_url` when configured. `n`,
`seed`, `num_steps`, `guidance`, `negative_prompt`, `loras`, `sampler` and `scheduler` are
read; `stream: true` sends per-step progress as events.

Edits and variations take the source image as multipart on `/v1/images/edits` and
`/v1/images/variations`; adapters are named through `GET /v1/loras` and resolved inside
`lora_dir`, never by path.

## What to expect

The placement measures what each card holds before loading; a family that does not fit spreads
across cards and, only when no card holds a piece, reaches the host. `GET /v1/renders` shows
what is in flight, and `POST /v1/renders/{id}/cancel` stops one.
