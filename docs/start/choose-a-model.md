# Choose a model

Pick a family for the task, then a size and quantisation that fit your memory.

## Before you start

<!-- site: widget checklist id=choose-model-prereq -->
- [ ] You know the task: chat, vision, speech, images, video, audio or forecasting.
- [ ] You know the memory of each card and of the host.
<!-- site: end -->

## Families

A request names the model as in [Concepts](concepts.md#model-stores-and-names). The table lists what is served. [Models](../MODELS.md) has the full architecture lists.

<!-- site: widget table id=choose-model-families -->
| Task | Family | Request names it as | Notes |
|---|---|---|---|
| Chat | Llama, Mistral, Yi, Falcon and other distils | `name:tag` or `owner/repo` | One generic route reads the shape from the file |
| Chat | Qwen (`qwen2`, `qwen3`, `qwen3moe`, `qwen35`, `qwen3next`, `qwen4exp`) | `name:tag` or `owner/repo` | Mixtures and recurrent hybrids included |
| Chat | Gemma (`gemma` to `gemma4`) | `name:tag` or `owner/repo` | |
| Chat | Phi, Granite, OLMo, gpt-oss, LFM2, Nemotron-H, SmolLM3, StarCoder, GLM, ERNIE, DBRX | `name:tag` or `owner/repo` | gpt-oss speaks Harmony |
| Chat | DeepSeek V4.1 | its GGUF, as stored | Experts read from disk as routing asks. May be split. The only model sharded across machines |
| Chat with thinking | Reasoning models | `name:tag` | Thinking returned apart from the answer on every surface |
| Chat with images | `moondream`, `pixtral`, `qwen25` vision, `qwen3vl` | `name:tag` | Images travel base64 in the request |
| Chat with audio | `voxtral`, `ultravox` | `name:tag` or `owner/repo` | Capability `audio`. Clips over 30 s are encoded window by window |
| Embeddings | BERT-shaped encoders (`nomic`, `bge`), CLIP, SigLIP | `name:tag` or `owner/repo` | `/v1/embeddings`, `/api/embed` |
| Reranking | Qwen3-Reranker | `owner/repo` | `/v1/rerank` |
| Images | FLUX.1, FLUX.2 klein, Z-Image, Qwen-Image, SDXL, Boogu | `flux`, `flux2`, `z-image`, `qwen-image`, `sdxl`, `boogu` in `model` | Files from the Hugging Face store |
| Video | Wan | `wan` | `POST /api/video/plan` prices a clip before rendering |
| Transcription | Whisper | `whisper-1`, `whisper-small`, `whisper-medium`, `whisper-large-v3` | Distil and faster-whisper by repository |
| Speech | Parler-TTS, Piper, Pocket-TTS, Kyutai | `tts-1`, `piper/<voice>`, `pocket-tts`, `kyutai` | |
| Music and effects | ACE-Step, Stable Audio, EzAudio | `acestep`, `stable-audio`, `ezaudio` | |
| Separation | Mel-Band RoFormer | `/v1/audio/separate` | |
| Forecasting | Chronos-2, Timestral | `autogluon/chronos-2-small`, `mistralai/Timestral-5M` | `POST /v1/forecast` |
<!-- site: end -->

Image, video, audio and music families are build features. A family that is not built reports its models as unknown, see [Installation](install.md).

## Size and memory

The weights take parameters x bits per weight / 8 bytes. The bits per weight come from the file.

<!-- site: widget table id=choose-model-bits -->
| Format | Bits per weight | Weights of 8B parameters | Weights of 70B parameters |
|---|---:|---:|---:|
| F16, BF16 | 16 | 16 GB | 140 GB |
| Q8_0 | 8.5 | 8.5 GB | 74 GB |
| Q6_K | 6.56 | 6.6 GB | 57 GB |
| Q4_K, Q4_0 | 4.5 | 4.5 GB | 39 GB |
| IQ2_XXS | 2.06 | 2.1 GB | 18 GB |
<!-- site: end -->

The bits per weight are those of [Fewer bits per weight](../learn/03-quantisation.md). The 2 size columns are that arithmetic, not measurements.

Add these to the weights:

| Cost | Amount |
|---|---|
| Per loaded model, whatever its size | About 870 MiB, which the placement does not budget |
| KV cache | Grows with the context window, see [KV cache](../learn/06-kv-cache.md) |
| Host memory for a spill | The daemon admits a component on the host against the memory the host can give without swapping, with a quarter kept as headroom |

## What fits where

| Your memory | Result |
|---|---|
| The weights and cache fit on one card | The model loads on the fastest card |
| They fit on the cards together | The layers spread across cards |
| They fit on the cards and the host | The remainder spills to the host. Decode reads every host layer per token, so the host memory bandwidth bounds the rate |
| A mixture larger than both | Only the experts spill, and are read from disk. The disk bounds the load and the rate |

[Placement](../learn/09-placement.md) shows the measured cases. 2 examples from [Status](../STATUS.md):

| Model | Result on 2 cards of 16 GB and the host |
|---|---|
| deepseek-r1:70b at Q4_K_M | 26 of 80 layers on the host |
| unsloth/Qwen3-235B-A22B-GGUF | Runs on the 2 cards and the host |

## Choose a quantisation

| Goal | Pick |
|---|---|
| Fit a model that is slightly too large | The next smaller format of the same model |
| Keep quality | Q8_0 or F16 when memory allows |
| Run on the host | Any block format, which needs AVX2, see [Building](../BUILDING.md) |

A file in a format loken does not read is refused with the format's id. [Models](../MODELS.md) lists the formats read.

## Run it

```sh
loken pull qwen3:1.7b --source ollama
loken list
loken ps
```

`loken list` shows the model in the store. `loken ps` shows it after a first request loads it.

## Next

- [First run](first-run.md)
- [Models](../MODELS.md) and [Status](../STATUS.md)
- [Placement](../learn/09-placement.md)
