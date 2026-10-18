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

| Tensor formats a GGUF may carry | |
|---|---|
| Float | `F32`, `F16`, `BF16` |
| Legacy blocks | `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1` |
| K-quants | `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K` |
| Codebook | `IQ2_XXS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`, `Q2_0` |
| Others | `MXFP4` |

The codebook formats run on a card for the experts, in decode and prefill, and on the host
through integer dot products that need AVX2. A dense projection stored in a codebook format is
requantised once at load to Q8_0 for a card; the experts keep their format.

A file that stores a tensor in another format is refused with the format's id. Token and
per-layer embedding tables stay as stored and a lookup decodes the rows it needs. A load that
rebuilds a weight in another representation logs one ledger line with the bytes before and
after.

`loken pull owner/repo:SELECTOR` keeps the GGUF files of a repository whose directory, or
dash-delimited file-name token, equals SELECTOR (case-insensitive), with the repository's
configuration files; without a selector every file is pulled. Each file is checked against its
listed size and its sha256.

A pull follows the daemon's streamed progress and stops when no frame arrives for a multiple of
the longest gap seen (eight times). A repository
holding a model GGUF and its `mmproj` resolves to the model file; the projector is its companion.

A GGUF split into parts by `gguf-split` is read as one model on the generic route: the first
part's metadata, the union of the parts' tensor directories, each tensor read from the part
that holds it. It is named by its first part, and a part whose siblings are missing is refused.
A set is placed layer by layer, never tensor-parallel. A family's vision tower reads from the
first part only; a set whose `v.*` tensors sit in another part is served as text, and the log
says why.

A checkpoint may ship a draft head that the served file leaves out: the multi-token-prediction
layer of the wide-residual family. It is read from a companion file, `<stem>.mtp.gguf` beside
the model, the stem being that of a split's parts. `loken draft-head <repo> <out.gguf>` writes it
from the checkpoint's `mtp.*` tensors by range requests, as released: the checkpoint's names,
shapes and bytes. This layout is loken's own, and no GGUF specification names it. The loader
lists the head under the model's names, its layer `blk.N.*` with N the model's layer count, splits
the fused indexer projection and expert stacks by rows, reads each norm scale one more than
stored, and quantises a matrix as it reads it. The file carries `general.architecture` and
`general.name` and nothing else. A model without the companion decodes without a head.

## Language models

One generic transformer reads its shape from the checkpoint, so a family is a row in a table
rather than a port. The architectures that table names:

| Line | Architectures as the GGUF header declares them |
|---|---|
| Llama and its distils | `llama`, `mistral`, `mistral3`, `yi`, `falcon`, `falcon3`, `internlm2`, `neox`, `stablelm` |
| Qwen | `qwen2`, `qwen3`, `qwen3moe`, `qwen35`, `qwen35moe`, `qwen3next`, `qwen4exp` |
| Gemma | `gemma`, `gemma2`, `gemma3`, `gemma4` |
| Phi | `phi`, `phi2`, `phi3`, `phi4` |
| Granite | `granite`, `granite3`, `granitemoe` |
| OLMo | `olmo2`, `olmoe` |
| Others | `gptoss`, `lfm2`, `lfm2moe`, `nemotron_h_moe`, `smollm3`, `starcoder`, `starcoder2`, `glm4`, `chatglm`, `ernie4_5`, `dbrx` |

Mixtures route their experts on the card, and spill experts alone when the model does not fit;
the recurrent hybrids (`lfm2moe`, `nemotron_h_moe`, `qwen3next`) carry their state-space blocks
beside attention. Reasoning models return their thinking apart from the answer on every
surface; gpt-oss speaks Harmony and is read as such.

DeepSeek V4.1 has a loader of its own, chosen by what the file or the checkpoint directory
contains (hyper-connection tensors), not by name. Its routed experts stay in the file, which may
be split, and are read as routing asks for them; the always-read weights and the busiest experts
sit on the cards. A routing prior the file lacks is borrowed from a companion GGUF beside it,
found by the metadata it declares. Sharding across machines runs for this model only, see
[`CLUSTER.md`](CLUSTER.md).

Vision: `moondream`, `pixtral`, the Qwen vision lines (`qwen25` vision, `qwen3vl`). Images travel
base64 in the request on all three surfaces.

Speech in: `voxtral` and `ultravox` take audio in a chat turn. A model whose files carry a speech
tower lists `audio` among its capabilities. The tower is either the `mmproj*.gguf` beside a text
GGUF (`clip.projector_type` names the family), or a checkpoint directory whose `config.json`
declares an `audio_config`: Ultravox ships only the tower and the projector, and names its text
model by `text_model_id`, which is looked up in the hub cache by id, else by name under any owner.
Its safetensors are served through the dense checkpoint route below, so no GGUF is needed. Pulling
the checkpoint pulls that text model too when the cache holds no copy of it; a gated id needs a
token, or a copy pulled under another owner.
The tower is placed after the text model, on a card the text model does not use when there is one.
A clip arrives base64 as an `input_audio` part of `/v1/chat/completions` or in `audios` of an
`/api/chat` message, in any container the server decodes, and is mixed down and resampled to the
tower's rate. Clips over 30 s are encoded window by window. Its embeddings take the place of one
marker in the prompt, so the clip counts against the context window and in `prompt_tokens` /
`prompt_eval_count`. A request carrying a clip is not served from the prompt cache and takes no
grammar.

Dense checkpoints: a directory of `config.json`, a tokenizer and F32, F16 or BF16 safetensors
shards of a Llama, Mistral or Qwen text model is served through the same generic route as a GGUF
file. Shapes come from the configuration, the tensors are read in their stored width, and a
long-context rope declared there is turned into the frequency table a GGUF file would carry.

Embeddings and reranking: BERT-shaped encoders (`nomic`, `bge` and the like) answer
`/v1/embeddings` and `/api/embed`; a Qwen3-Reranker checkpoint answers `/v1/rerank`.

### The qwen4exp family

One generic transformer reads the architecture from the file: a model of this family is not a
loader of its own. What the file declares and the loader reads:

| Element | Read from the file |
|---|---|
| Wide residual (hyper-connections) | The stream count and the low-rank width, with the mixers that replace the layer norms and the output mixer. Mixers and tables stored in BF16 or a block format are read as stored |
| Layer pattern | One indexed-attention layer after every `full_attention_interval` layers, the others delta-rule (three delta-rule layers, then one attention layer, in the files measured) |
| Indexed attention | The indexer's head count, key length and top-k, and the per-layer compress ratio |
| Experts | Routed and used counts, expert width, a shared expert with a scalar gate. 160, 256 and 512 routed experts have been loaded |
| N-gram table | One table at the layer `ple.layers` names, 16 heads over bigrams and trigrams. It is read by row from the mapping and never staged on a card |

The files measured, all loading through the generic route:

| File | Experts | Parts | Size |
|---|---|---|---|
| Q4_K_M | 160 | 1 | 28.4 GB |
| IQ1_M | 256 | 2 | 58.4 GB, 28.8 GB of it the n-gram table |
| UD-Q4_K_XL | 512 | 4 | 111 GB |

The experts stay in the file's mapping. A card holds a cache of the hot ones per layer and the
rest run on the host beside it; a wide prompt batch streams the layer's experts to the card
through two buffers when that costs less than running its rows one by one. Host memory is
admitted against the bytes the process owns. Limits and measurements are in
[`STATUS.md`](STATUS.md#the-qwen4exp-family).

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
