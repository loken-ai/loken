# Architectures

Every architecture the loader accepts, the formats it reads, and the test that loads it.

The loader decides from tensors and metadata. The key `general.architecture` selects the row of
`src/inference/model_facts.rs`. A checkpoint's `architectures[0]` selects a GGUF tag in
`src/inference/load/awq_loader.rs`. This table lists the keys of the first and the classes of the second.

Each test lives in `src/inference/generic_transformer/loading_tests.rs`. It writes a tiny synthetic
checkpoint of the architecture in the format, loads it through `GenericWeights::build` (the call
the server makes) on a host-only `HeteroPlan`, runs a forward pass of 5 tokens, and asserts finite,
non-constant logits of the vocabulary's size. A safetensors test writes the weights of the
architecture's GGUF recipe in the publisher's tensor names and layout (fused or split projections,
the publisher's rotary row order, stacked or per-expert tensors, shifted norms), and asserts that
its logits equal those of the same weights loaded from the GGUF file, so a wrong permutation or
split fails. `every_architecture_has_a_loading_test` fails when the facts table holds an
architecture without a test, and when an architecture whose publisher ships safetensors has no
safetensors test or no row here saying so.

`src/inference/generic_transformer/hf_reference_tests.rs` checks the layout against the reference
implementation. `scripts/hf_vectors.py` builds a tiny random model from each class of the Hugging Face
`transformers` library, saves it with the library's `save_pretrained` as float32 safetensors under
`tests/vectors/hf/<folder>`, and records the logits of the last position of a 6-token prompt and of
the next 4 greedy tokens, run in float32 on the CPU. The test loads each folder through the same
entry point on a host-only plan, feeds the reference's ids back in turn, and asserts a cosine of at
least 0.998 with the reference's logits at every step and the same argmax wherever the 2 top
reference logits are further apart than twice the measured rounding error (float weights are rebuilt
in 8 bits). Every hybrid keeps one block of each kind. A seed is kept only when its routing decisions
and its logits stay clear of that rounding. `every_reference_folder_has_a_test` fails when a folder
and the list of tests differ. A composite release whose vendor also ships a native layout gets
`params.json` and `consolidated.safetensors` of the same weights beside it, checked by the library's
own converter of that layout. Its native tests assert the reference's logits and those of the
composite.

The `Publisher safetensors` column names the `architectures[0]` classes of the publisher's
`config.json` the loader reads (the class table in `src/inference/load/hf_config.rs`), `read as`
when the class loads under another architecture's tag, and `none:` with the reason when the publisher
ships no checkpoint the loader can read. The `Verified` column says how the safetensors layout is
checked: `reference` (the folders whose logits it matches), `synthetic` (equality with the GGUF
recipe), `real checkpoint` (a published checkpoint checked by hand).

| Architecture | Publisher safetensors | Formats read | Test names | Verified |
|---|---|---|---|---|
| `llama` | `LlamaForCausalLM`, `MixtralForCausalLM`, native layout, dense and mixture-of-experts | GGUF, HF safetensors, native layout, AWQ safetensors | `llama_loads_from_gguf`, `llama_loads_from_safetensors`, `llama_loads_from_native_layout`, `llama_loads_from_awq_safetensors`, `llama_safetensors_equal_gguf`, `mixtral_loads_from_safetensors`, `mixtral_loads_from_native_layout` | reference (`llama`, `mixtral`), synthetic |
| `mistral` | `MistralForCausalLM` (loaded under the `llama` tag) | GGUF, HF safetensors, native layout, AWQ safetensors | `mistral_loads_from_gguf`, `mistral_loads_from_safetensors`, `mistral_loads_from_native_layout`, `mistral_loads_from_awq_safetensors` | reference (`mistral`), synthetic |
| `mistral3` | `Mistral3ForCausalLM`, `MistralForConditionalGeneration`, `Mistral3ForConditionalGeneration` (loaded under the `llama` tag), YaRN with its table scale, the query scale by position caps the context | GGUF, HF safetensors composite, native layout | `mistral3_loads_from_gguf`, `mistral3_loads_from_safetensors`, `mistral3_composite_loads_from_safetensors`, `mistral3_matches_reference`, `mistral3_native_matches_reference`, `mistral3_native_matches_composite` | reference (`mistral3`), synthetic |
| `smollm3` | `SmolLM3ForCausalLM` | GGUF, HF safetensors | `smollm3_loads_from_gguf`, `smollm3_loads_from_safetensors` | reference, synthetic, real checkpoint |
| `ernie4_5` | `Ernie4_5ForCausalLM` | GGUF, HF safetensors | `ernie4_5_loads_from_gguf`, `ernie4_5_loads_from_safetensors` | reference, synthetic, real checkpoint |
| `granite` | `GraniteForCausalLM` | GGUF, HF safetensors | `granite_loads_from_gguf`, `granite_loads_from_safetensors` | reference, synthetic, real checkpoint |
| `granite3` | `GraniteForCausalLM`, read as `granite` | GGUF | `granite3_loads_from_gguf` | through `granite` |
| `granitemoe` | `GraniteMoeForCausalLM` | GGUF, HF safetensors | `granitemoe_loads_from_gguf`, `granitemoe_loads_from_safetensors` | reference, synthetic, real checkpoint |
| `yi` | `YiForCausalLM` (the first release, later ones ship `LlamaForCausalLM`) | GGUF, HF safetensors | `yi_loads_from_gguf`, `yi_loads_from_safetensors` | synthetic, no transformers class (`YiForCausalLM` is custom code) |
| `internlm2` | `InternLM2ForCausalLM` | GGUF, HF safetensors | `internlm2_loads_from_gguf`, `internlm2_loads_from_safetensors` | synthetic, real checkpoint, no transformers class (custom code) |
| `starcoder` | `GPTBigCodeForCausalLM` | GGUF, HF safetensors | `starcoder_loads_from_gguf`, `starcoder_loads_from_safetensors` | reference, synthetic, real checkpoint (santacoder) |
| `starcoder2` | `Starcoder2ForCausalLM` | GGUF, HF safetensors | `starcoder2_loads_from_gguf`, `starcoder2_loads_from_safetensors` | reference, synthetic, real checkpoint |
| `qwen2` | `Qwen2ForCausalLM` | GGUF, HF safetensors, AWQ safetensors | `qwen2_loads_from_gguf`, `qwen2_loads_from_safetensors`, `qwen2_loads_from_awq_safetensors`, `qwen2_safetensors_equal_gguf` | reference, synthetic, real checkpoint |
| `qwen3` | `Qwen3ForCausalLM` | GGUF, HF safetensors, AWQ safetensors | `qwen3_loads_from_gguf`, `qwen3_loads_from_safetensors`, `qwen3_loads_from_awq_safetensors`, `qwen3_safetensors_equal_gguf` | reference, synthetic, real checkpoint |
| `qwen3moe` | `Qwen3MoeForCausalLM` | GGUF, HF safetensors | `qwen3moe_loads_from_gguf`, `qwen3moe_loads_from_safetensors` | reference, synthetic |
| `qwen35` | `Qwen3_5ForCausalLM`, `Qwen3_5ForConditionalGeneration` | GGUF, HF safetensors | `qwen35_loads_from_gguf`, `qwen35_loads_from_safetensors` | reference, synthetic |
| `qwen35moe` | `Qwen3_5MoeForCausalLM`, `Qwen3_5MoeForConditionalGeneration` | GGUF, HF safetensors | `qwen35moe_loads_from_gguf`, `qwen35moe_loads_from_safetensors` | reference, synthetic |
| `qwen3next` | `Qwen3NextForCausalLM` | GGUF, HF safetensors | `qwen3next_loads_from_gguf`, `qwen3next_loads_from_safetensors`, `qwen3next_loads_from_bf16_safetensors` | reference, synthetic |
| `qwen4exp` | none: no transformers class and no published checkpoint, read from GGUF only | GGUF | `qwen4exp_loads_from_gguf` | GGUF toy file against its reference (`wide_tests.rs`), no transformers class |
| `phi` | `PhiForCausalLM`, read as `phi2` | GGUF | `phi_loads_from_gguf` | through `phi2` |
| `phi2` | `PhiForCausalLM` | GGUF, HF safetensors | `phi2_loads_from_gguf`, `phi2_loads_from_safetensors` | reference, synthetic, real checkpoint |
| `phi3` | `Phi3ForCausalLM` | GGUF, HF safetensors | `phi3_loads_from_gguf`, `phi3_loads_from_safetensors` | reference, synthetic, real checkpoint |
| `phi4` | `Phi3ForCausalLM`, read as `phi3` | GGUF | `phi4_loads_from_gguf` | through `phi3` |
| `chatglm` | `ChatGLMModel`, `ChatGLMForConditionalGeneration` | GGUF, HF safetensors | `chatglm_loads_from_gguf`, `chatglm_loads_from_safetensors` | synthetic, no transformers class (custom code) |
| `glm4` | `Glm4ForCausalLM` | GGUF, HF safetensors | `glm4_loads_from_gguf`, `glm4_loads_from_safetensors` | reference, synthetic |
| `stablelm` | `StableLmForCausalLM` | GGUF, HF safetensors | `stablelm_loads_from_gguf`, `stablelm_loads_from_safetensors` | reference, synthetic, real checkpoint |
| `olmo2` | `Olmo2ForCausalLM` | GGUF, HF safetensors | `olmo2_loads_from_gguf`, `olmo2_loads_from_safetensors` | reference, synthetic, real checkpoint |
| `olmoe` | `OlmoeForCausalLM` | GGUF, HF safetensors | `olmoe_loads_from_gguf`, `olmoe_loads_from_safetensors` | reference, synthetic, real checkpoint |
| `moondream` | none: custom code only, tensor names that change with each revision, no transformers class | GGUF | `moondream_loads_from_gguf` | no transformers class (custom code) |
| `falcon` | `FalconForCausalLM` | GGUF, HF safetensors | `falcon_loads_from_gguf`, `falcon_loads_from_safetensors` | reference (multi-query decoder), synthetic |
| `falcon3` | `LlamaForCausalLM`, read as `llama` | GGUF | `falcon3_loads_from_gguf` | through `llama` |
| `dbrx` | `DbrxForCausalLM` | GGUF, HF safetensors | `dbrx_loads_from_gguf`, `dbrx_loads_from_safetensors` | reference, synthetic |
| `orion` | `OrionForCausalLM` | GGUF, HF safetensors | `orion_loads_from_gguf`, `orion_loads_from_safetensors` | synthetic, no transformers class (custom code) |
| `mpt` | `MPTForCausalLM`, `MptForCausalLM` | GGUF, HF safetensors | `mpt_loads_from_gguf`, `mpt_loads_from_safetensors` | reference, synthetic |
| `gemma` | `GemmaForCausalLM` | GGUF, HF safetensors | `gemma_loads_from_gguf`, `gemma_loads_from_safetensors` | reference, synthetic |
| `gemma2` | `Gemma2ForCausalLM` | GGUF, HF safetensors | `gemma2_loads_from_gguf`, `gemma2_loads_from_safetensors` | reference, synthetic, real checkpoint |
| `gemma3` | `Gemma3ForCausalLM`, `Gemma3ForConditionalGeneration` | GGUF, HF safetensors | `gemma3_loads_from_gguf`, `gemma3_loads_from_safetensors`, `gemma3_loads_from_bf16_safetensors` | reference, synthetic, real checkpoint |
| `gemma4` | `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`, per-layer inputs, shared key-value layers, keys read as values, routed experts beside the dense feed-forward | GGUF, HF safetensors | `gemma4_loads_from_gguf`, `gemma4_matches_reference`, `gemma4_keqv_matches_reference`, `gemma4_moe_matches_reference` | reference (`gemma4`, `gemma4_keqv`, `gemma4_moe`) |
| `gptoss` | `GptOssForCausalLM` (MXFP4 blocks, or the dequantised experts) | GGUF, HF safetensors | `gptoss_loads_from_gguf`, `gptoss_loads_from_safetensors`, `gptoss_loads_from_mxfp4_safetensors`, `gptoss_loads_from_bf16_safetensors` | reference, synthetic, real checkpoint |
| `lfm2` | `Lfm2ForCausalLM` | GGUF, HF safetensors | `lfm2_loads_from_gguf`, `lfm2_loads_from_safetensors` | reference, synthetic, real checkpoint |
| `lfm2moe` | `Lfm2MoeForCausalLM` | GGUF, HF safetensors | `lfm2moe_loads_from_gguf`, `lfm2moe_loads_from_safetensors` | reference, synthetic |
| `nemotron_h_moe` | `NemotronHForCausalLM` | GGUF, HF safetensors | `nemotron_h_moe_loads_from_gguf`, `nemotron_h_moe_loads_from_safetensors` | reference, synthetic |
| `llama` text model of a speech release | `VoxtralForConditionalGeneration`, native layout with the table in the tower's namespace | HF safetensors composite, native layout | `voxtral_loads_from_safetensors`, `voxtral_loads_from_native_layout`, `voxtral_matches_reference`, `voxtral_native_matches_reference`, `voxtral_native_matches_composite` | reference (`voxtral`), synthetic |

## Notes

- Formats are GGUF (one file. A split set reads through the same tensor path, covered once in `loader.rs`), dense safetensors in the framework's layout (F32, F16, BF16. MXFP4 blocks for the experts of `gptoss`), AWQ safetensors, and the vendor's native layout (`params.json` and `consolidated*.safetensors`), dense or with routed experts.
- A checkpoint's `architectures[0]` selects a row of the class table in `src/inference/load/hf_config.rs`: the architecture tag, the tensor layout (`src/inference/generic_transformer/dense_layouts.rs`), and the conventions the class fixes. The layouts are data: a rule per part of a block says whether a stored tensor is read in place, reshaped, shifted by one (norm scales stored less one), split by head, stacked from per-expert tensors, halved, transposed or repacked (MXFP4). The config keys are read under the names each publisher uses (`hf_config.rs`). Widths that the tensors state (the dense and the expert feed-forward, the short convolution, which blocks attend) are read from the tensors.
- Rotary rows: a class pairs lanes as the framework's rotation does (lane `i` with `i + n/2`) unless its row says adjacent. A GGUF converter reorders the rows of the tags whose GGUF files pair adjacent lanes. The checkpoint's rows are not reordered, the loader's pairing follows the class.
- A fused attention projection whose rows are grouped by key-value head (`InternLM2ForCausalLM`. `FalconForCausalLM` of a grouped decoder, `GPTBigCodeForCausalLM` without a shared head) is split into query, key and value rows when read. The others are read in place.
- The recipes write the tensor set each architecture's GGUF carries: fused or split projections, biases, QK norms, post norms, routed experts with shared experts, short convolution, Mamba2 and delta-rule blocks. Every recipe is written in F32. `tiny_model::write_quant` exercises K-quant weights for one architecture.
- `qwen4exp` loads the seeded toy file `tests/vectors/qwen4exp/toy.gguf`, since its wide residual, indexer and n-gram features are not part of a recipe. `wide_tests.rs` checks that model against the reference.
- A test checks the file says what the recipe meant (each block's mixer) and that a file missing a tensor of its block is refused (`a_file_missing_a_tensor_of_its_block_is_refused`). `a_wrong_layout_is_caught` loads checkpoints laid out wrongly (rows left in the file's pairing, a fused projection read in the wrong order, norms not shifted) and asserts their logits differ.
- The synthetic tests check that a model loads and runs, and that a publisher's checkpoint equals its GGUF recipe. The reference tests check that it answers as the reference does. `mpt` positions come from ALiBi (`attention.max_alibi_bias`), which only the unfused host attention adds to the scores: the load offers such a model no card, Vulkan or OpenCL device and logs why, and a card's attention refuses such a layer (`an_alibi_model_stays_off_the_cards`, run with a card).
- A query scale by position (`llama_4_scaling_beta`, `attention.temperature_scale` and `attention.temperature_length`) is not applied: the context stops at the length before which the scale is one, and the load logs it.
- Not read, as from a GGUF file: LongRoPE factors (`phi3`), the linear rope scaling of the global layers (`gemma3`), `clip_qkv` (`dbrx`), the per-head query and key norms of the 12B `stablelm`. The reference vectors hold none of these. The grouped `falcon` decoder's 2 input norms are not mapped, so such a checkpoint is refused for a missing tensor. A quantised checkpoint other than AWQ of the llama, mistral and qwen families is refused.
- Architectures with their own model code (image, speech, video, forecasting, DeepSeek V4.1) are not read by the generic transformer and are not in this table.

## Where the lessons live in the code

The lessons in `learn/` describe behaviour. This section lists, lesson by lesson, the source files that carry it and the checks that verify it.

### Lesson 1: Tokens

File table removed from "In loken":

| File | What it decides |
|---|---|
| `src/inference/engine/llm_engine/gguf.rs` | Builds the tokenizer from GGUF metadata (`tokenizer.ggml.model`, `.tokens`, `.merges`, `.token_type`, `.pre`). Synthesises merges for a SentencePiece vocabulary that ships none. |
| `src/inference/token/sentencepiece.rs` | The unigram tokenizer from scratch: `ModelProto` wire format, Viterbi segmentation, the space meta-symbol, byte fallback. |
| `src/api/handlers/prompt_format.rs` | Renders the checkpoint's `tokenizer.chat_template` through a Jinja engine. |
| `src/api/thinking.rs` | Splits reasoning from the answer (`<think>`, Harmony channels, the `thought` variant), surviving a chunk boundary mid-stream. `SPLIT_MARKERS` are registered as ordinary text so a decoder that drops special tokens keeps them. |
| `src/api/tool_calls.rs` | Reads a tool call out of the raw token stream, on every family's spelling. |

Checks removed: none. The "Try it" curl exercises stay.

### Lesson 2: What a model file holds

File table removed from "In loken":

| File | What it decides |
|---|---|
| `src/tensor/quantized/gguf_file.rs` | The one statement of the container: `mod layout` holds the magic, the field order, the dimension reversal, the value-type ids and the alignment rule, so the reader and the writer cannot disagree. `read_facts` reads what a catalogue lists and stops at the first `tokenizer.*` key. |
| `src/tensor/quantized/gguf_source.rs` | A checkpoint split into parts, presented as one source. |
| `src/tensor/safetensors_io.rs` | The safetensors reader: header, then views over the mapping. |
| `src/tensor/quant_view.rs` | A typed view into blocks owned by a parent tensor or by a mapped file, so a layer can be read in place. |
| `src/inference/cache/qvb.rs` | The parsed checkpoint kept on the host between model swaps, so switching back does not re-read gigabytes. |
| `src/api/handlers/catalogue.rs` | What `/api/tags` and `/v1/models` list, and the header facts they are built from. |

Document links removed from the text:

- How a request names a model, and which stores are searched, is in `../MODELS.md`.
- Load time costs ("A 24 GB checkpoint over a USB disk: 52 s reading, 6 s uploading to the cards") were sourced from `../STATUS.md#costs`, which also records what a spilled model used to hold twice in host memory, and what the loader still uploads twice.

Checks removed: none.

### Lesson 3: Fewer bits per weight, and what it costs

File table removed from "In loken":

| File | What it decides |
|---|---|
| `src/tensor/quant_cpu/format/q4_0.rs`, `q4_k.rs`, `iq2_xxs.rs` | One file per format, holding layout, dequantiser and quantiser together, because a sub-block count changed in one and not the others is a silent defect. Start with `q4_0.rs`: the bias that makes an unsigned nibble signed, and why the 2 nibbles of a byte are not neighbours. |
| `src/tensor/quant_cpu/scale.rs` | The scale searches: signed, with offset, and the second stage that quantises the scales themselves. |
| `src/tensor/quant_cpu/avx/mod.rs` | The AVX2 dot products: the 4-step shape every kernel follows and the unsigned-by-signed constraint of the multiply-add. |
| `src/tensor/quantized/kernel_matmul.rs` | Which matmul serves a call: mat-vec for decode, tiled for prefill, a repack when the format has one, each able to decline, and the dequantised matmul at the end that never does. |
| `src/tensor/oracle_parity.rs` | Every dequantiser pinned against ggml's scalar output, since a format compared only with itself proves nothing. |
| `src/inference/load/requantize.rs`, `calibration.rs`, `compensate.rs` | Re-quantising per tensor rather than per model, the importance record, the error carried into later columns. |
| `src/tensor/blockscaled/` | fp8, fp4 and bf16 with the checkpoint's block scales, read in place by any model that ships them. |
| `cuda/marlin/DESIGN.md`, `cuda/README.md` | The 4-bit tensor-core GEMM explained as a design, and which kernels are built by nvcc against which are compiled at run time. |
| `src/inference/load/awq.rs` | Why a second 4-bit format is worth carrying, in bytes a weight. |

Checks removed from "Try it":

```sh
cargo test --lib oracle_parity
```

- Expected: 14 tests, one per format plus `every_format_reconstructs_about_as_well_as_ggml` and the importance-matrix one, all passing.
- It verifies every dequantiser against ggml's scalar output.
- Reading `src/tensor/quant_cpu/format/q4_0.rs` confirms the arithmetic: 32 values, a 16-bit scale, 4 bits each, 18 bytes, 4.5 bits a weight.
- `q8_0.rs` gives 34 bytes for 32, 8.5 bits a weight.

### Lesson 4: One layer, then 40 of them

File table removed from "In loken":

| File | What it decides |
|---|---|
| `src/inference/generic_transformer/mod.rs` | One layer struct for every `blk.{i}.*` decoder: the optional pieces (split or fused QKV, q/k norms, post-attention norms, a mixture) are detected from the checkpoint's tensors, so Qwen2/3, Gemma, Phi and the others share the code. |
| `src/inference/generic_transformer/layer_attn.rs`, `layer_ffn.rs` | The 2 sub-layers, with the sliding-window variant and the fused paths. |
| `src/inference/model/rope.rs` | The rotary table, and YaRN: high-frequency dimensions kept, low-frequency ones interpolated, so a model trained at one length answers past it. |
| `src/tensor/ops/attention.rs` | The attention op itself: scaled dot product, the grouped-query repeat, the 2 rotary pairings, the fused prefill. |
| `src/inference/fused_moe.rs` | The router and the batched expert projections it feeds. |
| `src/inference/model/deepseek_v41/attention.rs` | Latent attention with a single shared key head, an fp8 round trip and a learned attention sink. The `model/` directory holds one such directory per family. |
| `src/inference/engine/model_backend.rs` | The trait each architecture implements, which replaced a match over every variant. |

Other references removed:

- The papers behind each family are listed in `../REFERENCES.md`.
- The DeepSeek V4.1 port was gated against the authors' own `model.py`. Each phase of the reference dumps its inputs and outputs into `tests/vectors/`, and a test replays the phase on the same inputs.

Checks removed from "Try it":

```sh
ls tests/vectors/
cargo test --lib the_rope_frequencies_are_the_references
cargo test --lib the_selected_attention_matches_the_reference_block
cargo test --lib deepseek_v41::moe
```

- `the_rope_frequencies_are_the_references` checks the rotary frequencies against the reference's `inv_freq`.
- `the_selected_attention_matches_the_reference_block` checks an attention block against the reference's block output.
- Exercise: find the tolerance the second test asserts against the scale of the output, divide it by 10 and run again. The assertion names the measured difference, which is what a parity test is for.
- `deepseek_v41::moe` runs the streamed experts against the resident ones.

### Lesson 5: 2 regimes, one model

File table removed from "In loken":

| File | What it decides |
|---|---|
| `src/inference/engine/model_backend.rs` | The prefill chunk, derived from the free VRAM and the widest feed-forward rather than a constant, and the chunked prefill with its out-of-memory ladder. |
| `src/inference/serve/scheduler.rs` | The iteration-level scheduler: a plan per step (who prefills, who decodes, with their block tables), a prefill token budget, preemption that never half-allocates. Pure bookkeeping, tested without a card. |
| `src/inference/serve/continuous_serve.rs` | The worker thread that owns the model and the paged cache and is the only caller of the forward. |
| `src/inference/serve/batched_forward.rs` | The batched forward as an f32 specification, and the batch-invariance contract a real backend has to meet. |
| `src/inference/kernel/cpu_decode_exec.rs` | The host decode path with a preallocated arena, so a token on the host allocates nothing. |
| `src/inference/place/layer_perf.rs`, `src/api/handlers/system.rs` | The per-layer and per-stage timers behind `/api/layer_perf` and `/api/stage_perf`. Off by default because each stage costs a device synchronisation. |

Other references removed: the 70B host-placement figure was sourced from `../STATUS.md#placement-measured-separately`.

Checks removed: none. The stage timer exercise stays.

### Lesson 6: What the model remembers between tokens

The removed table, verbatim:

| File | What it decides |
|---|---|
| `src/inference/cache/mod.rs` | Everything that remembers between tokens: the index of this directory. |
| `src/inference/cache/paged_kv.rs`, `paged_attention.rs` | The block allocator and block tables, pure bookkeeping tested without a card, and the attention that reads through them. |
| `src/inference/cache/q4_kv.rs`, `q8_kv.rs` | The quantised caches: why a per-token 4-bit key failed and the per-channel scheme that replaced it, and the 8-bit layout that appends one row per step with one copy. |
| `src/inference/cache/kv_disk.rs` | The disk tier: blocks, the chained hash, one manifest per sequence, writes off the request path. |
| `src/inference/cache/turboquant/mod.rs` | A 3-bit cache written as a host reference so that the measurement deciding whether to write the kernel exists first. Read `measure` before the rest. |
| `src/inference/engine/llm_engine/load.rs` (`kv_reuse_start`) | What a new prompt reuses: the resident prefix, a snapshot that covers more of it, a disk sequence that covers more still, each rounded down to a prefill chunk. |
| `src/inference/engine/llm_engine/params.rs` | The keys: `kv_disk_dir` and its budget. Snapshots need none. `fidelity` turns the shift past the window off at `reference`. The guide is [`../guides/kv-memory.md`](guides/kv-memory.md). |

Removed reference in the Try it text: the prefill chunk size comes from `adaptive_prefill_chunk` in `src/inference/engine/model_backend.rs`.

Removed checks: none.

### Lesson 7: From logits to a token

The removed table, verbatim:

| File | What it decides |
|---|---|
| `src/inference/sample/token_sampling.rs` | The 2 steps: which tokens are candidates, then one draw. The draw is host-side, so the logits come back from the device once per token. |
| `src/inference/engine/llm_engine/gguf.rs` (`build_sampling_from`) and `sampling_tests.rs` | How a request's `temperature`, `top_k`, `top_p` and the model's own defaults combine. The tests read as the specification (a `top_p` without a `top_k` is nucleus sampling over the whole vocabulary). |
| `src/inference/engine/decode_step.rs` | The per-step order: penalty, then sampling, on the shared decode path. |
| `cuda/sampling.cu` | The device-side sampler used when the draw can stay on the card. |
| `src/inference/engine/llm_engine/mod.rs` | The grammar parser factory: a JSON schema or a Lark grammar becomes a per-token mask through llguidance. `response_format` on the OpenAI surface and `format` on the Ollama one reach it. |
| `src/distributed/replay.rs` | Resuming a generation on another node with the same tokens, which holds only if sampling depends on position and seed alone. |
| `scripts/determinism.py` | Runs a prompt repeatedly and diffs the answers. |

Removed reference in the measured section: the comment above `kv_reuse_start` in `src/inference/engine/llm_engine/load.rs` states the chunk boundary rule.

Removed checks: none. The `scripts/determinism.py` runner was only listed, never run in an exercise.

### Lesson 8: More than one token per weight read

The removed table, verbatim:

| File | What it decides |
|---|---|
| `src/inference/serve/eagle.rs` | The EAGLE-1 head on the native substrate: one decoder layer over the target's feature and the next token's embedding, reusing the target's embedding table and output head. The module comment is a complete account of the method. |
| `src/inference/serve/prompt_lookup.rs` | Prompt-lookup drafting from a rolling buffer of prompt and generated tokens. |
| `src/inference/engine/llm_engine/spec.rs` | Drafting with a separate model and verifying in one target pass. The comment states the greedy restriction and what temperature would need. |
| `src/inference/serve/speculative_config.rs` | Whether drafting is worth it, decided from measured device timings rather than assumed. |
| `src/inference/serve/spec_kv_cache.rs` | A cache that can drop the rejected tail. |

Removed reference in the measured section: the EAGLE figure of 1.71 to 1.82 times was "the module's, dated by it", from the `eagle.rs` module comment.

Removed checks: none.

### Lesson 9: Fitting a model on the machine you have

The removed table, verbatim:

| File | What it decides |
|---|---|
| `src/inference/place/plan.rs` | The heterogeneous plan every block stack uses: pack the fastest card first, spill card to card to host, never out of memory. |
| `src/inference/place/vram_manager.rs` | The one place placement reads a card's state from: pools trimmed before the probe, a reclaim registry for idle media residents, headroom for a hot component before it loads. |
| `src/tensor/dry.rs`, `src/tensor/heap.rs`, `src/inference/place/dry_plan.rs` | The circularity break: measure a request's peak by running the forward on a device that allocates nothing, then replay the allocator's chunk and hole behaviour on top of it. |
| `src/inference/place/placement_invariants.rs` | The fleet rule as tests: the cards decide, not the budgets. No placement decides on a name or an index. Every hot component can span n cards. |
| `src/inference/place/layer_executor.rs` | The host budget as a share of the machine, not a constant. |
| `src/inference/offload/streamed.rs`, `room.rs`, `cuda.rs` | The streamed placement for a model larger than the cards and host together: the weights every token reads and as many routed experts as the room allows, kept on the cards fastest first, under the same switches as every other placement. |
| `src/inference/offload/store.rs` | The hot tier over streamed experts: views into the mapping, least-recently-routed eviction, cold experts dropped from the page cache once they have run while the host's memory is short, and left there otherwise. |
| `src/distributed/routing.rs` | Which node answers, by predicted completion rather than by load. [`CLUSTER.md`](CLUSTER.md) has the cost model. |

Removed checks: none. The invariants in `placement_invariants.rs` are tests (the cards decide, not the budgets, no decision on a name or an index, every hot component can span n cards), but the lesson gave no command to run them.

### Lesson 10: Measuring an engine without fooling yourself

The removed table, verbatim:

| File | What it decides |
|---|---|
| [Method](https://github.com/loken-ai/benchmarks/blob/main/docs/METHOD.md) | The protocol: the cell, the prompts, which cards each engine may use, what a cell that publishes no rate means, the `Tokens` column beside the rate. |
| [Rate columns](https://github.com/loken-ai/benchmarks/blob/main/docs/METHOD.md#the-rate-columns) | Why prefill is prompt tokens over the client's time to first token, why decode is the client's span, and why both are medians. Cold and warm figures: [cold and warm](https://github.com/loken-ai/benchmarks/blob/main/docs/METHOD.md#cold-and-warm). |
| [`scripts/bench`](https://github.com/loken-ai/benchmarks/blob/main/scripts/bench) in the benchmarks repository | The harness: starts the releases named in its settings, stamps the build into every row, files each run. |
| [Benchmarks site](https://loken-ai.github.io/benchmarks/) | The runs of every machine, one page per hardware, rebuilt from the records so that no page is typed. |
| `scripts/determinism.py` | The same prompt repeated, and a diff of the answers. |
| `src/inference/place/layer_perf.rs`, `src/energy.rs` | The per-layer timers of lesson 5, and the joules: host RAPL and card NVML summed over the request window. |
| [lokprof](https://github.com/loken-ai/lokprof) | The bench tool: drives each engine through its own API, separates prefill from decode, records energy beside speed, makes the engines comparable before comparing them. |

Removed wording in Try it: lokprof "is not in this tree", and a `cargo build` running in another terminal (now "a build").

Removed checks: none.

## README

Removed: "Lessons 3 and 4 run a `cargo test`", "shows where it lives in this tree", and the path `../img/learn-*.svg` of the diagrams.
