# Embed texts and images

`/v1/embeddings`, `/api/embed` and `/v2/embed` turn texts into vectors with an embedding model
(a Qwen3-Embedding or BERT-shaped checkpoint, or a chat model as a fallback). A CLIP or SigLIP
checkpoint embeds images as well, in the same space as its texts: a caption and its picture
land near each other, so one cosine ranks images against a text or texts against an image.
Every vector is L2-normalised; the cosine is the dot product.

## Find the model

A text-image model is a Hugging Face checkpoint directory, found by its repository id as any
Hugging Face model is: `openai/clip-vit-large-patch14` is looked up as
`models--openai--clip-vit-large-patch14/snapshots/<revision>` under `<models dir>/huggingface/hub`,
the models directory itself, and `huggingface_models_dir`
([configuration](../CONFIGURATION.md)). The first snapshot holding safetensors and a
`config.json` is taken, and it is served as a text-image model when it also holds
`tokenizer.json`, `preprocessor_config.json`, a `text_config` and a `vision_config`, and the
tensors of one of the two layouts:

| Layout | Tensors | Text vector | Image vector |
|---|---|---|---|
| CLIP | `text_projection`, `visual_projection` | causal tower, read at the end-of-text token, projected | class token through `post_layernorm`, projected |
| SigLIP | `text_model.head`, `vision_model.head` | bidirectional tower over a row padded to the position table, read at its last position through the head | attention-pool head |

The heads, norm epsilon and activation come from the tower configs, with transformers'
defaults for what a config leaves out. Texts longer than the position table are cut, keeping
the end-of-text token. Images are resized, cropped, rescaled and normalised as
`preprocessor_config.json` states, with Pillow's resampling filter of the same number. The
model runs on the host in f32 and loads on the first request that names it.

## Texts

```sh
curl localhost:11435/v1/embeddings -d '{
  "model": "openai/clip-vit-large-patch14",
  "input": ["a photo of a cat", "a photo of a dog"]
}'
```

## Images

vLLM's form: one conversation per vector, its single part an `image_url`, a base64 data URL or
an http(s) URL from a host listed in `image_fetch_hosts`. `messages` holds one conversation, or
a list of them for several vectors.

```sh
curl localhost:11435/v1/embeddings -d '{
  "model": "openai/clip-vit-large-patch14",
  "messages": [[{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0..."}}]}]],
  "encoding_format": "base64"
}'
```

Cohere's form: `images` as data URIs, or `inputs` as `content` lists.

```sh
curl localhost:11435/v2/embed -d '{
  "model": "google/siglip-base-patch16-224",
  "images": ["data:image/png;base64,iVBORw0..."],
  "embedding_types": ["float", "base64"]
}'
```

A conversation holding text and an image is refused: the model embeds one or the other. A text
model asked for an image answers 400.

## Encodings

`float` is a JSON array; `base64` is the little-endian f32 bytes, base64-encoded.
`dimensions` (`output_dimension` on `/v2/embed`) keeps the first coordinates of each vector.
