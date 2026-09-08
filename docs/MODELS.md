# Models

What loken can load, where it looks for the files, and how a request names them.

## Formats and stores

| Store | Layout | Formats | Named as |
|---|---|---|---|
| Ollama | `manifests/` and `blobs/` under `ollama_models_dir` | GGUF, with the template, system prompt and parameters of the manifest | `name:tag`, as in `qwen3:0.6b` |
| Hugging Face | the cache layout under `huggingface_models_dir`, or its `hub/` subdirectory | safetensors, and the `.pth` archives some media families ship | `owner/repo`, as in `Qwen/Qwen3-0.6B` |
| A bare file | a GGUF placed at the root of the Hugging Face store | GGUF | its file name, with or without a tag |

`loken pull <name> --source ollama|huggingface` fetches into the matching store: an Ollama
tag whole, a Hugging Face repository's safetensors, GGUF, bin and JSON files. A `.pth` archive a
media family needs is placed in the store by hand. `loken list` prints what both stores hold;
`loken delete` removes a model. A request that names a model not yet
loaded loads it, and a model stays resident for the keep-alive the daemon was started with.

Quantisations are read as the file carries them. On the cards the block formats run through
their own kernels; on the host they need AVX2 (see [`BUILDING.md`](BUILDING.md)).

## Language models

One generic transformer reads its shape from the checkpoint, so a family is a row in a table
rather than a port. The architectures that table names:

| Line | Architectures as the GGUF header declares them |
|---|---|
| Llama and its distils | `llama`, `mistral`, `mistral3`, `yi`, `falcon`, `falcon3`, `internlm2`, `neox`, `stablelm` |
| Qwen | `qwen2`, `qwen3`, `qwen3moe`, `qwen35`, `qwen35moe`, `qwen3next` |
| Gemma | `gemma`, `gemma2`, `gemma3`, `gemma4` |
| Phi | `phi`, `phi2`, `phi3`, `phi4` |
| Granite | `granite`, `granite3`, `granitemoe` |
| OLMo | `olmo2`, `olmoe` |
| Others | `gptoss`, `lfm2`, `lfm2moe`, `nemotron_h_moe`, `smollm3`, `starcoder`, `starcoder2`, `glm4`, `chatglm`, `ernie4_5`, `dbrx` |

Mixtures route their experts on the card, and spill experts alone when the model does not fit;
the recurrent hybrids (`lfm2moe`, `nemotron_h_moe`, `qwen3next`) carry their state-space blocks
beside attention. Reasoning models return their thinking apart from the answer on every
surface; gpt-oss speaks Harmony and is read as such.

Vision: `moondream`, `pixtral`, the Qwen vision lines (`qwen25` vision, `qwen3vl`). Images travel
base64 in the request on all three surfaces. Speech in: `voxtral`, `ultravox` take audio in a
chat turn.

Embeddings and reranking: BERT-shaped encoders (`nomic`, `bge` and the like) answer
`/v1/embeddings` and `/api/embed`; a Qwen3-Reranker checkpoint answers `/v1/rerank`.

## Image

The `model` of an image request names the family by substring; the loader then reads that
family's files from the Hugging Face store.

| Family | `model` contains | Files |
|---|---|---|
| FLUX.1 | `flux` | `black-forest-labs/FLUX.1-schnell`, with the T5 and CLIP encoders it references |
| FLUX.2 klein | `flux2` | the klein checkpoint and its Qwen text encoder |
| Z-Image | `z-image` | `Tongyi-MAI/Z-Image-Turbo` |
| Qwen-Image | `qwen-image` | the Qwen-Image checkpoint and its text encoder |
| SDXL | `sdxl` | an SDXL checkpoint in safetensors, adapters through `/v1/loras` |
| Boogu | `boogu` | the Boogu checkpoint |

A loader that lacks a file says which one it looked for. Image parity across machines is a hash
of one render per family and depends on the placement, see [`STATUS.md`](STATUS.md).

## Video

`wan` clips, text- and image-conditioned, with the umT5 encoder the family ships. Ask
`POST /api/video/plan` what a clip of a given length will cost before rendering it.

## Audio

| Task | Family | `model` |
|---|---|---|
| Transcription | Whisper | `whisper-1`, `whisper-small`, `whisper-medium`, `whisper-large-v3`, distil and faster-whisper checkpoints by their repository |
| Speech | Parler-TTS | `tts-1`, `tts-1-hd`, `parler-tts/parler-tts-mini-v1`, `parler-tts/parler-tts-large-v1` |
| Speech | Piper | `piper/<voice>`, as in `piper/fr_FR-tom-medium` |
| Speech | Pocket-TTS, Kyutai | `pocket-tts`, `kyutai` |
| Music | ACE-Step, Stable Audio | `acestep`, `stable-audio` |
| Sound effects | EzAudio | `ezaudio` |
| Separation | Mel-Band RoFormer | `/v1/audio/separate` |

The OpenAI voice names (`alloy`, `echo`, `fable`, `onyx` and the rest) are accepted and mapped;
`GET /v1/audio/voices` lists what the resident model offers.

## What a request may not ask for

A family that is not compiled reports its models as unknown. A media request for a family the
build lacks is refused with the feature to enable, never served by another family.
