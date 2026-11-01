# Generate an image

`POST /v1/images/generations` renders a picture from a prompt and returns it as a URL or as base64.

## Before you start

<!-- site: widget checklist id=images-prereq -->
- [ ] The build has the `image` feature, which the default build has.
- [ ] `lokend` is running.
- [ ] The model files are in the Hugging Face store.
<!-- site: end -->

The `model` of a request names the family by substring.
The families and their files are in [the image models](../MODELS.md#image). For Z-Image:

```sh
loken pull Tongyi-MAI/Z-Image-Turbo --source huggingface
```

## Smallest configuration

None. A request with no `model` renders with `z-image-turbo`.
These keys apply to specific cases:

| Key | Effect |
|---|---|
| `public_url` | The origin of the URLs the server hands out. Set it behind a proxy. |
| `lora_dir` | Where adapters live. |

## Run it

```sh
curl -s localhost:11435/v1/images/generations -H 'Content-Type: application/json' -d '{
  "model": "z-image",
  "prompt": "a lighthouse at dusk, oil painting",
  "size": "1024x1024"
}'
```

Add `"response_format": "b64_json"` to get the picture inline.

## Expected result

Without `response_format`, or with `url`, the picture is stored as a file and the answer holds `data[0].url`, built on the `Host` of the request or on `public_url`.
With `b64_json`, the picture comes back base64 under `data[0].b64_json`.
The answer also holds `created` and `render_ms`.

```json
{"created": 1700000000, "data": [{"url": "http://localhost:11435/v1/files/FILE_ID/content"}], "render_ms": 9000}
```

## Options

| Field | Effect |
|---|---|
| `size` | `WIDTHxHEIGHT`, each side a multiple of 16. |
| `n` | 1 to 10 pictures. |
| `seed`, `num_steps`, `guidance`, `negative_prompt`, `sampler`, `scheduler` | Read by the renderer. |
| `loras` | Adapters, named through `GET /v1/loras` and resolved inside `lora_dir`, never by path. |
| `output_format` | `png`, `jpeg` or `webp`. |
| `stream: true` | Sends per-step progress as events. |

`POST /v1/images/edits` and `POST /v1/images/variations` take the source image as multipart.

## Placement

The placement measures what each card holds before loading.
A family that does not fit spreads across cards, and reaches the host only when no card holds a piece.
`GET /v1/renders` lists the renders in flight. `POST /v1/renders/{id}/cancel` stops one.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| 400 `prompt must not be empty` | The prompt is blank. | Send a prompt. |
| 400 `size ... must be a multiple of 16 (VAE stride)` | A side of `size` is not a multiple of 16. | Use a size such as `1024x1024`. |
| 400 `n must be <= 10` or `n must be >= 1` | `n` is out of range. | Send 1 to 10. |
| 400 `response_format '...' not supported; use 'url' or 'b64_json'` | `response_format` has another value. | Send `url` or `b64_json`. |
| 400 `num_steps must be <= ...` or `num_steps must be >= 1` | `num_steps` is out of range. | Send a smaller value. |

## Next

- [Image families and their files](../MODELS.md#image)
- [Routes and fields](../API.md)
