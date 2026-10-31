# Embed texts and images

Post texts, or images with a CLIP or SigLIP model, to `/v1/embeddings` and get 1 L2-normalised vector per item.

An embedding model is a Qwen3-Embedding or BERT-shaped checkpoint, or a chat model as a fallback.
A CLIP or SigLIP checkpoint embeds images as well, in the same space as its texts. A caption and its picture land near each other, so the cosine ranks images against a text or texts against an image.
The cosine is the dot product.

## Before you start

<!-- site: widget checklist id=embeddings-prereq -->
- [ ] An embedding model is in the store. A CLIP or SigLIP checkpoint is found by its Hugging Face repository id, for example `openai/clip-vit-large-patch14`, under `huggingface_models_dir` ([configuration](../CONFIGURATION.md)).
- [ ] For images by URL, `image_fetch_hosts` lists the hosts to fetch from. A base64 data URL needs no host.
- [ ] For an image model, the checkpoint folder holds safetensors, `config.json`, `tokenizer.json` and `preprocessor_config.json`.
<!-- site: end -->

## Smallest configuration

Texts and data URLs need none. Images by URL need:

```toml
[inference]
image_fetch_hosts = ["images.example.com"]
```

## Run it

<!-- site: widget tabs group=client id=embeddings-client -->
### OpenAI

`input` is a string or a list of strings.

```sh
curl localhost:11435/v1/embeddings -d '{
  "model": "openai/clip-vit-large-patch14",
  "input": ["a photo of a cat", "a photo of a dog"]
}'
```

### vLLM

1 conversation per vector. Its single part is an `image_url`: a base64 data URL, or an http(s) URL from a host listed in `image_fetch_hosts`. `messages` holds 1 conversation, or a list of them for several vectors.

```sh
curl localhost:11435/v1/embeddings -d '{
  "model": "openai/clip-vit-large-patch14",
  "messages": [[{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0..."}}]}]],
  "encoding_format": "base64"
}'
```

### Cohere

`images` holds data URIs. `inputs` holds `content` lists. `texts` holds strings.

```sh
curl localhost:11435/v2/embed -d '{
  "model": "google/siglip-base-patch16-224",
  "images": ["data:image/png;base64,iVBORw0..."],
  "embedding_types": ["float", "base64"]
}'
```
<!-- site: end -->

`/api/embed` is the Ollama route for texts.

| Option | Effect |
|---|---|
| `encoding_format` (`embedding_types` on `/v2/embed`) | `float` is a JSON array. `base64` is the little-endian f32 bytes, base64-encoded. |
| `dimensions` (`output_dimension` on `/v2/embed`) | Keeps the first coordinates of each vector. Must be 1 or more. |

## Expected result

1 vector per item, in the order sent. The OpenAI form answers one `data` entry per vector with its `embedding`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| 400 `model 'x' embeds texts only; an image needs a CLIP or SigLIP model` | An image was sent to a text model. | Name a CLIP or SigLIP checkpoint. |
| 400 `a conversation is embedded as text or as one image, not both` | A conversation holds text and an image. | Send them as 2 items. |
| 400 `a conversation is embedded as one image; send each image apart` | A conversation holds several images. | Send 1 image per conversation. |
| 400 `a conversation to embed holds no text` | The conversation is empty. | Send text or an `image_url` part. |
| 400 `exactly one of texts, images or inputs must be given` | `/v2/embed` got 0 or several of them. | Send 1 field. |
| 400 `dimensions must be >= 1` | `dimensions` is below 1. | Raise it. |
| 404 `Model 'x' not loaded` | The model is not in the store. | Pull it with `POST /api/pull`, or check the repository id. |

## Next

- [`/v1/embeddings` in the API reference](../API.md)
- [Models](../MODELS.md)
