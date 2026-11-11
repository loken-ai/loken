# Architectures

Every architecture the loader accepts, the formats it reads, and the tests that load it.

## Selection

The loader decides from tensors and metadata:

- `general.architecture` selects a row of `src/inference/model_facts.rs`.
- A checkpoint's `architectures[0]` selects a GGUF tag through the class table in `src/inference/load/hf_config.rs`.
  AWQ checkpoints are read by `src/inference/load/awq_loader.rs`.

## Tests

Each loading test lives in `src/inference/generic_transformer/loading_tests.rs`.  
A test writes a tiny synthetic checkpoint of the architecture in the format.

The test loads the checkpoint through `GenericWeights::build`, the call the server makes, on a host-only `HeteroPlan`.
It runs a forward pass of 5 tokens and asserts finite, non-constant logits of the vocabulary's size.

A safetensors test also writes the weights of the architecture's GGUF recipe in the publisher's tensor names and layout:

- fused or split projections
- the publisher's rotary row order
- stacked or per-expert tensors
- shifted norms

The safetensors test asserts that its logits equal those of the same weights loaded from the GGUF file.
A wrong permutation or split fails.

`every_architecture_has_a_loading_test` fails in 2 cases:

- The facts table holds an architecture without a test.
- An architecture whose publisher ships safetensors has no safetensors test or no row here saying so.

### Reference tests

`src/inference/generic_transformer/hf_reference_tests.rs` checks the layout against the reference implementation.

`scripts/hf_vectors.py` builds the vectors:

1. It builds a tiny random model from each class of the Hugging Face `transformers` library.
2. It saves the model with `save_pretrained` as float32 safetensors under `tests/vectors/hf/<folder>`.
3. It records, in float32 on the CPU, the logits of the last position of a 6-token prompt and of the next 4 greedy tokens.

The test loads each folder through the same entry point on a host-only plan and feeds the reference's ids back in turn.  
At every step it asserts:

- a cosine of at least 0.998 with the reference's logits
- the same argmax wherever the 2 top reference logits are further apart than twice the measured rounding error (float weights are rebuilt in 8 bits)

Every hybrid keeps 1 block of each kind.  
A seed is kept only when its routing decisions and its logits stay clear of that rounding.  
`every_reference_folder_has_a_test` fails when the folders and the list of tests differ.

A composite release whose vendor also ships a native layout gets `params.json` and `consolidated.safetensors` of the same weights beside it.
The library's own converter of that layout checks them.  
The native tests assert the reference's logits and those of the composite.

## Architecture table

The `Publisher safetensors` column lists:

- the `architectures[0]` classes of the publisher's `config.json` that the loader reads (the class table in `src/inference/load/hf_config.rs`)
- `read as` when the class loads under another architecture's tag
- `none:` and the reason when the publisher ships no checkpoint the loader can read

The `Verified` column says how the safetensors layout is checked:

| Value | Meaning |
|---|---|
| `reference` | its logits match the named reference folders |
| `synthetic` | its logits equal the GGUF recipe |
| `real checkpoint` | a published checkpoint, checked by hand |

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
| `glm_moe_dsa` | `GlmMoeDsaForCausalLM` (latent attention, sparse indexer shared across layers, sigmoid router with a selection bias, shared expert, dense first blocks, fp8 weights with 128 by 128 tile scales) | GGUF, HF safetensors | `glm_moe_dsa_loads_from_gguf`, `glm_moe_dsa_matches_reference`, `glm_moe_dsa_dense_matches_reference`, `glm_moe_dsa_sparse_matches_reference`, `glm_moe_dsa_fp8_matches_reference`, `glm_moe_dsa_layers_match_reference` | reference (`glm_moe_dsa`, `glm_moe_dsa_dense`, `glm_moe_dsa_sparse`, `glm_moe_dsa_fp8`), synthetic |
| `qwen3` talker and code predictor of a speech release | `Qwen3TTSForConditionalGeneration` with its fields under `talker_config`, `Qwen3TTSTalkerCodePredictorModelForConditionalGeneration` for the code predictor nested in the same checkpoint with its fields under `talker_config.code_predictor_config` | HF safetensors composite | `a_class_reads_its_text_model_under_its_own_key`, `the_published_configuration_gives_both_transformers_geometry`, `a_speech_checkpoint_reads_as_a_talker_and_a_code_predictor`, `the_published_speech_header_reads_as_a_talker_and_a_code_predictor`, `a_nested_class_loads_and_a_layout_without_a_head_gives_none` | synthetic, real checkpoint configuration and header |
| `llama` transformer of a speech codec's decoder | `Qwen3TTSTokenizerV2DecoderTransformerModel` nested in the codec's own configuration with its fields under `decoder_config`, every layer windowed, a scale per channel on both residual branches | HF safetensors, the codec's file | `a_windowed_scaled_class_matches_a_plain_host_run`, `a_window_wider_than_the_rows_is_the_causal_run`, `the_card_windowed_scaled_class_matches_a_plain_host_run` | synthetic |
| `llama` transformer of Mimi's decoder | `ProjectedTransformer`, its configuration made from the codec's tensors, no token table, no final norm, layer norms with a bias, one Q, K and V projection, adjacent rotary lanes, the error-function GELU of an ungated feed-forward, every layer windowed, a scale per channel on both residual branches | the codec's own file | `a_tableless_layer_normed_class_matches_a_plain_host_run`, `a_prefill_longer_than_the_cache_is_the_windowed_run`, `the_card_tableless_layer_normed_class_matches_a_plain_host_run` | synthetic |
| `llama` transformer of Mimi's encoder | `ProjectedTransformer`, its blocks under the path the caller gives, the Q, K and V matrix stored as a parameter or as the weight of a linear | the codec's own file | `an_encoder_spelled_class_matches_a_plain_host_run`, `the_card_encoder_spelled_class_matches_a_plain_host_run` | synthetic |
| `llama` text model of a speech release | `VoxtralForConditionalGeneration`, native layout with the table in the tower's namespace | HF safetensors composite, native layout | `voxtral_loads_from_safetensors`, `voxtral_loads_from_native_layout`, `voxtral_matches_reference`, `voxtral_native_matches_reference`, `voxtral_native_matches_composite` | reference (`voxtral`), synthetic |

## Notes

### Formats

The loader reads:

- GGUF, 1 file.
  A split set reads through the same tensor path, covered once in `loader.rs`.
- Dense safetensors in the framework's layout: F32, F16, BF16, and MXFP4 blocks for the experts of `gptoss`.
- AWQ safetensors.
- The vendor's native layout (`params.json` and `consolidated*.safetensors`), dense or with routed experts.

### Layout rules

A checkpoint's `architectures[0]` selects a row of the class table in `src/inference/load/hf_config.rs`.
The row fixes the architecture tag, the tensor layout (`src/inference/generic_transformer/dense_layouts.rs`) and the conventions of the class.

The layouts are data.
A rule per part of a block says how a stored tensor is read:

- in place
- reshaped
- shifted by 1 (norm scales stored less 1)
- split by head
- stacked from per-expert tensors
- halved
- transposed
- repacked (MXFP4)

`hf_config.rs` reads the config keys under the names each publisher uses.
A class reads its text model's fields under the key its row names, `text_config` unless the row says otherwise.
A model nested in a checkpoint has a class row of its own.
It is read from the checkpoint's configuration with `architectures` set to that class (`nested_config`).
A class row names the route that serves its checkpoints, the text route unless the row says otherwise.
The text route resolves no checkpoint of a class it does not serve.
A request for one fails with the kind of model and the route that serves it.

The embedding table a layout names gives the model's prefix.
The talker of `Qwen3TTSForConditionalGeneration` reads its codec table as the token table and its codec head as the head.
Its code predictor reads the tensors under its own prefix, its first codec table as the token table, and has no head.
The transformer of the codec's decoder reads the tensors under `decoder.pre_transformer.`, its output projection as the token table, and has no head.
A layout with no head gives the model none: its forward gives the normed rows and no logits.
A layout with no token table gives the model a table of 1 zero row, which no lookup reads: the model is fed by embeddings alone.
Its prefix is where its first block is.
A layout that reads no tensor outside the blocks gives the model no final norm: its rows are the last block's.
A class row can say that the ungated feed-forward runs the error-function GELU, which a GGUF file's runs in its tanh form.
`class_load::load_class` loads the transformer of one class of a dense checkpoint, the checkpoint's own or a class nested in it.
`class_load::load_class_from` loads it from a configuration and weights files given apart, on the cards it is given.
`class_load::load_class_held` also takes the path of the blocks from its caller, and can have a card hold the stored 16-bit float weights at F32, as the host does.
It places the blocks, the head, the cache and the work reserve through `HeteroPlan` and the ladder, as any dense load.
The cache of a class windowed in every layer is priced at the window and a prefill chunk, whatever the context.

### Windows and scales

A class row can window every layer when the configuration gives a `sliding_window` and no list of layer kinds.
Each windowed layer keeps the last keys of its window and a prefill chunk, and drops the older ones.
A row attends to itself and the keys of the window before it.

A layer can scale each residual branch by channel: `attn_layer_scale` multiplies the attention output, `ffn_layer_scale` the feed-forward output, before the residual add.
The scales are read when the layer's tensors hold them.
A layer with a scale runs the layer's forward on the host and on a card.
The paths that do not apply a scale step aside for it: the host decode executors, the recorded decode step, the continuous batch, tensor parallel splits, Vulkan and OpenCL.
Vulkan and OpenCL also step aside for the error-function GELU.
The tensors state some widths, and the loader reads those from the tensors: the dense and the expert feed-forward, the short convolution, and which blocks attend.

### Rotary rows

A class pairs lanes as the framework's rotation does (lane `i` with `i + n/2`), unless its row says adjacent.

A GGUF converter reorders the rows of the tags whose GGUF files pair adjacent lanes.
The loader leaves the checkpoint's rows in place and pairs lanes by class.

### Fused attention projections

A fused attention projection whose rows are grouped by key-value head is split into query, key and value rows when read.
This applies to:

- `InternLM2ForCausalLM`
- `FalconForCausalLM` of a grouped decoder
- `GPTBigCodeForCausalLM` without a shared head

The loader reads the other projections in place.

### Recipes

The recipes write the tensor set each architecture's GGUF carries:

- fused or split projections
- biases, QK norms and post norms
- routed experts with shared experts
- short convolution
- Mamba2 and delta-rule blocks

Every recipe is written in F32.
`tiny_model::write_quant` exercises K-quant weights for 1 architecture.

`qwen4exp` loads the seeded toy file `tests/vectors/qwen4exp/toy.gguf`, because its wide residual, indexer and n-gram features are not part of a recipe.
`wide_tests.rs` checks that model against the reference.

### What the tests check

| Test | Checks |
|---|---|
| Synthetic loading tests | A model loads and runs, and a publisher's checkpoint equals its GGUF recipe. |
| Reference tests | A model answers as the reference does. |
| `a_file_missing_a_tensor_of_its_block_is_refused` | The file says what the recipe meant (each block's mixer), and a file missing a tensor of its block is refused. |
| `a_wrong_layout_is_caught` | Checkpoints laid out wrongly (rows left in the file's pairing, a fused projection read in the wrong order, norms not shifted) produce different logits. |

### ALiBi

`mpt` positions come from ALiBi (`attention.max_alibi_bias`).
Only the unfused host attention adds ALiBi to the scores.

The load offers such a model no card, Vulkan or OpenCL device, and logs why.
A card's attention refuses such a layer.

`an_alibi_model_stays_off_the_cards` checks this and runs with a card.

### Query scale by position

The loader does not apply a query scale by position (`llama_4_scaling_beta`, `attention.temperature_scale` and `attention.temperature_length`).
The context stops at the length before which the scale is 1, and the load logs it.

### Not read

The loader does not read these, as from a GGUF file:

- LongRoPE factors (`phi3`)
- the linear rope scaling of the global layers (`gemma3`)
- `clip_qkv` (`dbrx`)
- the per-head query and key norms of the 12B `stablelm`

The reference vectors hold none of these.

The loader does not map the 2 input norms of the grouped `falcon` decoder, so it refuses such a checkpoint for a missing tensor.  
It refuses a quantised checkpoint other than AWQ of the llama, mistral and qwen families.

Architectures with their own model code (image, speech, video, forecasting) are outside this table, and the generic transformer does not read them.
DeepSeek V4.1 is read by the generic transformer as a wide-residual model.

## Embeddings towers

A text-image checkpoint is served as such when it holds:

- `tokenizer.json`
- `preprocessor_config.json`
- a `text_config` and a `vision_config`
- the tensors of 1 of 2 layouts

`src/inference/model/clip/pair.rs` makes the check.

| Layout | Tensors | Text vector | Image vector |
|---|---|---|---|
| CLIP | `text_projection`, `visual_projection` | Causal tower, read at the end-of-text token, projected. | Class token through `post_layernorm`, projected. |
| SigLIP | `text_model.head`, `vision_model.head` | Bidirectional tower over a row padded to the position table, read at its last position through the head. | Attention-pool head. |

- Heads, norm epsilon and activation come from the tower configs, with the defaults of `transformers` for what a config leaves out.
- A text longer than the position table is cut, keeping the end-of-text token.
- Images are resized, cropped, rescaled and normalised as `preprocessor_config.json` states, with the Pillow resampling filter of the same number (`src/inference/model/clip/preprocess.rs`).
- The model runs on the host in f32.

## Decode step from an embedding

`GenericHeteroTransformer::forward_embedding_step` runs the decode step of 1 row fed by an embedding instead of a token id.

- The row enters the layers as a token's table row does, with no embedding scale and no position row.
- The layers' caches advance as for a token step. Position 0 empties them first.
- The step gives back the row after the final norm, as the head reads it, and the logits when asked.
- A model whose layers take more than the embedding row is refused: a wide residual, per-layer inputs, learned positions.

`forward_embeddings_prefill` runs the prefill of several rows fed by embeddings from position 0.
It gives back the last row after the final norm, and its logits when asked.
`forward_embeddings_rows` runs the same prefill and gives back every row after the final norm, in order, with no head.
Its logits are those of a splice of the same rows with no text around them.
`token_rows` gives the token table's rows of some ids, as a token step reads them before any embedding scale.

`embedding_step_tests.rs` holds the step to 3 references, on the host and on a card:

- a plain host run of the same layers in f64 sums, within a tolerance derived from the float type and the depth
- the token step fed the same table row, bit for bit
- a fresh model after position 0, bit for bit
- for the prefill, the last row of the plain host run, and the splice of the same rows bit for bit

## Speech talker

`src/inference/model/qwen3_tts/` drives the talker of a Qwen3-TTS checkpoint frame by frame.
The talker is a transformer of the generic route, loaded by `class_load::load_class`.

The talker's cache holds the context a language model takes for a window nobody asked for, through `llm::context_the_cards_hold`:

- it starts at the positions of the talker's configuration
- it halves while its plan sends a layer to the host, spreads over more cards or reads a token's weights more slowly
- it stops at the working window
- with no card it keeps the positions of the configuration

The cards are read under the fleet's pressure, so a higher pressure level shrinks the cache.
The cache is unquantised, of the width the route gives an unquantised cache of the architecture.
The frames of a call are bounded by the rows the cache holds and by the checkpoint's frame limit.

The prompt holds rows fed as embeddings (`prompt.rs`):

- the design turn and the opening of the speech turn, each id a text row
- the control codes, each beside the pad text row, the last beside the start of text
- each text id beside the codec pad, and the end of text beside it
- the pad text row beside the start of codes

A text row is a row of the text table through the text projection.
A codec row is a row of the talker's token table.
The control codes name the language when the request gives one.

Each frame, the talker draws the code of group 0 from its last logits (`sampling.rs`):

- a code already drawn is penalised once
- the ids past the codebook are banned, but the end of speech
- the end of speech is banned before `MIN_FRAMES_BEFORE_EOS` frames
- the temperature, the top-k and the top-p apply, then the draw

The other codes of the frame come from a source of sub-codes, the code predictor (`predictor.rs`).
The next talker row sums the frame's code rows and the pad text row.

The code predictor is a second transformer of the generic route with no head, loaded by `class_load::load_class` with its own demand.
Its cache restarts at position 0 every frame.
It holds the 2 rows of the first pass and a row for each later pass, read from the group count of the configuration.
Its first pass takes 2 rows: the talker's normed row and the talker's codec row of group 0.
Each later pass takes the code predictor's row of the code the pass before drew.
Every row it takes goes through its input projection first.
The head of pass `s` gives the logits of group `s + 1`, drawn with the code predictor's own sampling values.
The input projection, the heads and the talker's text projection sit beside its last layer, counted in its demand.
The loop stops at the end of speech, at the frame limit of `generation_config.json`, or when the cache is full.
Every id comes from `config.json`, and every sampling value from `generation_config.json`.
The tokenizer is built from the `vocab.json`, `merges.txt` and `tokenizer_config.json` the checkpoint ships.
The tables stay mapped on the host and are read by row.

`src/inference/model/qwen3_tts/tests.rs` tests the talker on a toy checkpoint, and on the published checkpoint when the test store holds it.
The comparison to the reference runs outside the test suite, so no reference file lives in the repository.
The tests check:

- the layout of the prompt, the suppressed ids, the end of speech and the repetition penalty
- a prompt row as the sum of its text and codec parts, and the row after a frame as the sum of its code rows and the pad row
- a seeded run giving the same frames twice, on the toy and on the published checkpoint
- the loop stopping where the cache is full, and a streamed text spoken to its end
- the published checkpoint's tokenizer wrapping both turns
- the blocks of the published checkpoint summing to the tensors they load

### The Base variant

`tts_model_type` names the variant: `voice_design` speaks a described voice, `base` a voice cloned from a clip.
A Base checkpoint loads the same talker and code predictor, then its speaker encoder (`speaker.rs`).
A Base request carries a clone prompt and no design text.
A request without a clone prompt is refused with what the variant needs.

The speaker encoder turns a mono clip into the speaker embedding, as wide as a talker row:

- a clip at another rate than `speaker_encoder_config.sample_rate` is resampled first
- the mel front end mirrors the clip, windows it, and pools the magnitudes by a Slaney filter bank to their log
- the bands are the input channels of the first convolution
- an ECAPA network of convolutions, squeeze-excitation Res2Net blocks and an attentive statistics pooling follows
- its widths and kernels are read from the tensors, its dilations and the front end's sizes are constants of the reference code

The encoder has no layers and is placed whole through `ladder::load_whole`, its tensors in F32.

The clone prompt has 2 modes (`prompt::clone_rows`):

- the opening of the speech turn, each id a text row
- the control codes and the speaker embedding, each beside the pad text row
- the codec pad beside the start of text
- with the speaker embedding only: each text id and the end of text beside the codec pad, then the pad text row beside the start of codes
- in context: the reference text ids, the text ids and the end of text beside the codec pad, then the start of codes and each reference frame beside the pad text row

A reference frame row sums the frame's code rows and the pad text row, as the row after a drawn frame does.
The decoder takes the reference frames before the frames drawn, and their samples are cut from the waveform.
The engine entries `qwen3_speaker_embedding`, `qwen3_clone_blocking` and `qwen3_clone_streamed` in `tts_engine.rs` take a clone prompt.
No request field carries one yet.

## Speech codec decoder

`src/inference/model/qwen3_tts/decoder/` turns frames of codes into the waveform of a Qwen3-TTS checkpoint.
It reads `speech_tokenizer/config.json` and the `decoder.` tensors of `speech_tokenizer/model.safetensors`.
The transformer's layers and final norm are a transformer of the generic route, class `Qwen3TTSTokenizerV2DecoderTransformerModel`, loaded by `class_load::load_class_from` with its own demand.
The rest is 1 block placed whole by `ladder::load_whole`, in F32 as stored.
The rest holds the transformer's input and output projections, which run on its device.

The decode runs in 2 parts:

- the front: the dequantisation of the 2 quantizers, a causal conv and the transformer, at the frame rate
- the back: the upsample stages, the first conv, the snake blocks, the last snake and conv, at the rates up to the sample rate

The transformer layer is pre-norm with RMSNorm, separate q k v, a rotary embedding over the 2 halves of each head, a sliding causal window, SwiGLU and LayerScale.
Each pass of the front runs the layers afresh, its frames at positions from 0.
The layers give every normed row (`forward_embeddings_rows`).
An upsample stage is a causal transposed conv and a ConvNeXt block.
A snake block is SnakeBeta, a causal transposed conv and residual units of growing dilation.
SnakeBeta runs on `snake1d`, its 2 inputs computed at load from `alpha` and `beta`.
The waveform is clamped to the reference's bound.

The RVQ dequantisation and the causal convs are shared with Kyutai's Mimi in `src/inference/model/codec_blocks.rs`.

Each part decodes in chunks whose left context is its receptive field, so a chunked decode gives the samples of a whole one:

- `config::left_context` sums the reach of every stage in samples, then rounds up to whole frames
- the front reaches back through the first conv and the window of every layer
- the back reaches back through every kernel, stride and dilation
- a chunk holds the most frames whose working set fits the reserve the block is placed with
- the layers run a front pass in prefill chunks, their cache held to the window

A streamed decode (`DecodeStream`) takes frames as the talker draws them:

- the decoder runs on a thread of its own, beside the frame loop
- each time it is free, it takes every frame drawn since its last call
- the front of a call runs over the new frames and the front's left context before them
- the back reads the front rows of its left context, kept from the calls before
- the samples equal those of a whole decode within the rounding of F32
- a listener that is gone stops the decoder, and the frame loop at its next frame

The values that only the reference code holds are named in the code: `RVQ_EPS`, `CONVNEXT_LN_EPS`, `SNAKE_EPS`, `RESIDUAL_DILATION_BASE` and `WAVEFORM_BOUND`.

On a card, the depthwise conv of the ConvNeXt blocks runs on the card: a grouped conv is 1 strided GEMM whose batch is the groups.

`src/inference/model/qwen3_tts/decoder/tests.rs` tests the decoder on toy chains, and on the published checkpoint when the test store holds it.
The comparison to the reference runs outside the test suite.
The tests check, on the host and on a card when the ladder places it there:

- the transformer on the generic layers against a plain host run in F64 of the reference's layer, on a toy and on the published checkpoint, within the rounding of the residual's float type
- the waveform of 8 frames: its sample count, its bound and its repeat
- a decode in small chunks against a whole decode, for each part, within the rounding of F32 through the part
- a decode fed frames in uneven groups against a whole decode, within the rounding of F32 through the decoder
- the left context of toy chains of stages

## Mimi decoder

`src/inference/model/kyutai/mimi.rs` turns the codes of Kyutai's TTS into the waveform.
It reads the codec's own file, which holds no configuration.
The decoder-transformer's configuration is made from the shapes of its tensors and the settings Mimi's reference fixes: `ATTENTION_HEADS`, `ATTENTION_WINDOW`, `ROPE_THETA` and `LN_EPS`.
Its layers are a transformer of the generic route, class `ProjectedTransformer`, loaded by `class_load::load_class_from` with their own demand.
They sit on the cards the ladder sees when the decoder sits on a card, on the host otherwise.
The rest of the decoder sits on the device it is given: the dequantisation, the upsample and the SEANet decoder.

A decode runs every frame at once, from position 0, and keeps no state between calls.
The layers run the rows in prefill chunks, their cache held to the window and a chunk.

`generic_transformer::mimi_class_tests` holds the layers to a plain host run in F64 of Mimi's layer on a toy, within the rounding of F32, on the host and on a card.
The tests of the decoder check, on the published checkpoint when the test store holds it:

- the dry plan of the layers' own demand on a card that holds them, a card smaller than their blocks, and the host
- the waveform of a fixed code list: its sample count, its finite values and its repeat, on the host and on a card

## Drafter verification

A verify step runs several rows at once (`src/inference/generic_transformer/verify_window.rs`).
On a card, its kernels can settle a near-tie between 2 tokens the other way than a 1-row step does.

## Where the lessons live in the code

Each lesson in `learn/` describes a behaviour, and the tables below list the source files that carry it.

### Lesson 1: Tokens

[Lesson](https://loken-ai.github.io/loken/tokens/)

| File | Role |
|---|---|
| `src/inference/engine/llm_engine/gguf.rs` | Builds the tokenizer from the GGUF keys `tokenizer.ggml.model`, `.tokens`, `.merges`, `.token_type`, `.pre`. Synthesises merges for a SentencePiece vocabulary that ships none. |
| `src/inference/token/sentencepiece.rs` | Unigram tokenizer: `ModelProto` wire format, Viterbi segmentation, space meta-symbol, byte fallback. |
| `src/api/handlers/prompt_format.rs` | Renders `tokenizer.chat_template` through a Jinja engine. |
| `src/api/thinking.rs` | Splits reasoning from the answer (`<think>`, Harmony channels, the `thought` variant) across chunk boundaries. Registers `SPLIT_MARKERS` as ordinary text, so a decoder that drops special tokens keeps them. |
| `src/api/tool_calls.rs` | Reads a tool call out of the raw token stream, in every family's spelling. |

### Lesson 2: What a model file holds

[Lesson](https://loken-ai.github.io/loken/model-file/)

| File | Role |
|---|---|
| `src/tensor/quantized/gguf_file.rs` | States the container once in `mod layout`: magic, field order, dimension reversal, value-type ids, alignment rule. Reader and writer share it. `read_facts` reads what a catalogue lists and stops at the first `tokenizer.*` key. |
| `src/tensor/quantized/gguf_source.rs` | Presents a checkpoint split into parts as 1 source. |
| `src/tensor/safetensors_io.rs` | Safetensors reader: header, then views over the mapping. |
| `src/tensor/quant_view.rs` | Typed view into blocks owned by a parent tensor or a mapped file. A layer is read in place. |
| `src/inference/cache/qvb.rs` | Keeps the parsed checkpoint of an image, video or audio model on the host between model swaps. |
| `src/api/handlers/catalogue.rs` | Lists models on `/api/tags` and `/v1/models` from the header facts. |

- [`MODELS.md`](MODELS.md) says how a request names a model and which stores are searched.
- [`STATUS.md`](STATUS.md#costs) holds the load-time costs and what the loader still uploads twice.

### Lesson 3: Fewer bits per weight, and what it costs

[Lesson](https://loken-ai.github.io/loken/quantisation/)

| File | Role |
|---|---|
| `src/tensor/quant_cpu/format/q4_0.rs`, `q4_k.rs`, `iq2_xxs.rs` | 1 file per format, with layout, dequantiser and quantiser together. A sub-block count changed in only 1 of them is a silent defect. Read `q4_0.rs` first. |
| `src/tensor/quant_cpu/scale.rs` | Scale searches: signed, with offset, and the second stage that quantises the scales. |
| `src/tensor/quant_cpu/avx/mod.rs` | AVX2 dot products: the 4-step shape of every kernel and the unsigned-by-signed constraint of the multiply-add. |
| `src/tensor/quantized/kernel_matmul.rs` | Picks the matmul for a call: mat-vec for decode, tiled for prefill, a repack when the format has one. Each can decline. The dequantised matmul at the end never does. |
| `src/tensor/oracle_parity.rs` | Pins every dequantiser against the scalar output of ggml. |
| `src/inference/load/requantize.rs`, `calibration.rs`, `compensate.rs` | Re-quantises per tensor, records importance, carries the error into later columns. |
| `src/tensor/blockscaled/` | fp8, fp4 and bf16 with the checkpoint's block scales, read in place. |
| `cuda/marlin/DESIGN.md`, `cuda/README.md` | The 4-bit tensor-core GEMM design, and which kernels `nvcc` builds and which NVRTC compiles at run time. |
| `src/inference/load/awq.rs` | The second 4-bit format, in bytes per weight. |

```sh
cargo test --lib oracle_parity
```

- The command runs 14 tests: 1 per format, `every_format_reconstructs_about_as_well_as_ggml`, and the importance-matrix test.
- The command checks every dequantiser against the scalar output of ggml.
- `q4_0.rs` shows the arithmetic: 32 values, a 16-bit scale, 4 bits each, 18 bytes, 4.5 bits per weight.
- `q8_0.rs` gives 34 bytes for 32 values, 8.5 bits per weight.

### Lesson 4: One layer, then 40 of them

[Lesson](https://loken-ai.github.io/loken/forward-pass/)

| File | Role |
|---|---|
| `src/inference/generic_transformer/mod.rs` | 1 layer struct for every `blk.{i}.*` decoder. It detects the optional pieces from the tensors: split or fused QKV, q/k norms, post-attention norms, a mixture. |
| `src/inference/generic_transformer/layer_attn.rs`, `layer_ffn.rs` | The 2 sub-layers, with the sliding-window variant and the fused paths. |
| `src/inference/model/rope.rs` | Rotary table and YaRN: high-frequency dimensions kept, low-frequency ones interpolated. |
| `src/tensor/ops/attention.rs` | Attention op: scaled dot product, grouped-query repeat, the 2 rotary pairings, fused prefill. |
| `src/inference/generic_transformer/moe_weights.rs` | Routes a mixture and runs the batched expert projections. |
| `src/inference/generic_transformer/latent_mqa.rs` | Latent attention with 1 shared key head, an fp8 round trip and a learned attention sink. |
| `src/inference/engine/model_backend.rs` | The trait each architecture implements. |

- [`REFERENCES.md`](REFERENCES.md) lists the papers behind each family.
- The DeepSeek V4.1 port is gated against the authors' `model.py`.
  Each phase of the reference dumps its inputs and outputs into `tests/vectors/`, and a test replays the phase on the same inputs.

```sh
ls tests/vectors/
cargo test --lib the_rope_frequencies_are_the_references
cargo test --lib the_selected_attention_matches_the_reference_block
cargo test --lib wide_sparse_oracle_tests
```

| Test | Checks |
|---|---|
| `the_rope_frequencies_are_the_references` | Rotary frequencies against the reference's `inv_freq`. |
| `the_selected_attention_matches_the_reference_block` | An attention block against the reference's block output. |
| `wide_sparse_oracle_tests` | The generic route against the frozen V4.1 oracle, and against the bits the V4.1 path gave. |

Exercise: divide the tolerance of the second test by 10 and run it again.
The assertion names the measured difference.

### Lesson 5: 2 regimes, one model

[Lesson](https://loken-ai.github.io/loken/prefill-and-decode/)

| File | Role |
|---|---|
| `src/inference/engine/model_backend.rs` | Derives the prefill chunk from the free VRAM and the widest feed-forward. Runs the chunked prefill with its out-of-memory ladder. |
| `src/inference/serve/scheduler.rs` | Iteration-level scheduler. Plans each step: who prefills, who decodes, their block tables. Applies a prefill token budget. Preempts without half-allocating. Pure bookkeeping, tested without a card. |
| `src/inference/serve/continuous_serve.rs` | Worker thread that owns the model and the paged cache. The only caller of the forward. |
| `src/inference/serve/batched_forward.rs` | The batched forward as an f32 specification, and the batch-invariance contract a backend meets. |
| `src/inference/kernel/cpu_decode_exec.rs` | Host decode path with a preallocated arena. A token on the host allocates nothing. |
| `src/inference/place/layer_perf.rs`, `src/api/handlers/system.rs` | Per-layer and per-stage timers behind `/api/layer_perf` and `/api/stage_perf`. Off by default, because each stage costs a device synchronisation. |

The 70B host-placement figure comes from [`STATUS.md`](STATUS.md#placement-measured-separately).

### Lesson 6: What the model remembers between tokens

[Lesson](https://loken-ai.github.io/loken/kv-cache/)

| File | Role |
|---|---|
| `src/inference/cache/mod.rs` | Index of everything that remembers between tokens. |
| `src/inference/cache/paged_kv.rs`, `paged_attention.rs` | Block allocator and block tables, tested without a card. The attention that reads through them. |
| `src/inference/cache/q4_kv.rs`, `q8_kv.rs` | Quantised caches. A per-token 4-bit key failed and a per-channel scheme replaced it. The 8-bit layout appends 1 row per step with 1 copy. |
| `src/inference/cache/kv_disk.rs` | Disk tier: blocks, chained hash, 1 manifest per sequence, writes off the request path. |
| `src/inference/cache/turboquant/mod.rs` | 3-bit cache as a host reference. The measurement that decides whether to write the kernel comes first: read `measure.rs` before the rest. |
| `src/inference/engine/llm_engine/load.rs` (`kv_reuse_start`) | What a new prompt reuses: the resident prefix, a snapshot that covers more, a disk sequence that covers more still. Each is rounded down to a prefill chunk. |
| `src/inference/engine/llm_engine/params.rs` | Keys `kv_disk_dir` (and its budget) and `fidelity`. Snapshots need no key. `fidelity = reference` turns the shift past the window off. |

- The prefill chunk size comes from `adaptive_prefill_chunk` in `src/inference/engine/model_backend.rs`.
- [`guides/kv-memory.md`](https://loken-ai.github.io/loken/kv-memory/) is the user guide.

### Lesson 7: From logits to a token

[Lesson](https://loken-ai.github.io/loken/sampling/)

| File | Role |
|---|---|
| `src/inference/sample/token_sampling.rs` | 2 steps: pick the candidates, then draw 1. The draw runs on the host, so the logits return from the device once per token. |
| `src/inference/engine/llm_engine/gguf.rs` (`build_sampling_from`), `sampling_tests.rs` | Combines a request's `temperature`, `top_k`, `top_p` with the model's defaults. The tests are the specification: a `top_p` without a `top_k` is nucleus sampling over the whole vocabulary. |
| `src/inference/engine/decode_step.rs` | Per-step order on the shared decode path: penalty, then sampling. |
| `cuda/sampling.cu` | Device-side sampler for draws that can stay on the card. |
| `src/inference/engine/llm_engine/mod.rs` | Grammar parser factory. A JSON schema or a Lark grammar becomes a per-token mask through llguidance. `response_format` (OpenAI) and `format` (Ollama) reach it. |
| `src/distributed/replay.rs` | Resumes a generation on another node with the same tokens. This holds only if sampling depends on position and seed alone. |
| `scripts/determinism.py` | Runs a prompt repeatedly and diffs the answers. |

The comment above `kv_reuse_start` in `src/inference/engine/llm_engine/load.rs` states the chunk boundary rule.

### Lesson 8: More than one token per weight read

[Lesson](https://loken-ai.github.io/loken/speculative-decoding/)

| File | Role |
|---|---|
| `src/inference/generic_transformer/draft_head.rs` | The draft head a checkpoint ships: 1 more layer of the model's own kind over the hidden state and the next token's embedding. The decode drafts with it under `fidelity = "fast"`. |
| `src/inference/generic_transformer/block_draft.rs` | The block draft of a Sinkhorn wide-residual model: stages of latent attention and routed experts that guess a block of tokens in 1 pass. It reads the input of the target layers its file names, and keeps each stage's keys in a ring. The decode drafts with it whenever the model carries it. |
| `src/inference/serve/eagle.rs` | EAGLE-1 head: 1 decoder layer over the target's feature and the next token's embedding. Reuses the target's embedding table and output head. No request path builds it. The module comment describes the method in full. |
| `src/inference/serve/prompt_lookup.rs` | Prompt-lookup drafting from a rolling buffer of prompt and generated tokens. |
| `src/inference/engine/llm_engine/mod.rs` | `draft_k` drafts k tokens with a separate model. The target samples its own token at every position. |
| `src/inference/serve/speculative_config.rs` | Decides from measured device timings whether drafting pays. |
| `src/inference/serve/spec_kv_cache.rs` | Cache that drops the rejected tail. |

The EAGLE speedup of 1.71 to 1.82 times comes from the `eagle.rs` module comment, dated there.

### Lesson 9: Fitting a model on the machine you have

[Lesson](https://loken-ai.github.io/loken/placement/)

| File | Role |
|---|---|
| `src/inference/place/plan.rs` | Heterogeneous plan of every block stack: pack the fastest card first, spill card to card to host, never run out of memory. |
| `src/inference/place/tensor_parallel.rs` | Splits every layer across a group of cards, a mixture's experts by range, when the model does not fit 1 card and the measured reads, links and sums predict it faster. |
| `src/inference/place/vram_manager.rs` | The one place placement reads card state from. Trims pools before the probe, keeps a reclaim registry for idle media residents, reserves headroom for a hot component before it loads. |
| `src/tensor/dry.rs`, `src/tensor/heap.rs`, `src/inference/place/dry_plan.rs` | Break the circularity. They measure a request's peak by running the forward on a device that allocates nothing, then replay the allocator's chunk and hole behaviour on it. |
| `src/inference/place/placement_invariants.rs` | The fleet rule as tests. The cards decide, not the budgets. No placement decides on a name or an index. Every hot component can span n cards. |
| `src/inference/place/layer_executor.rs` | Sets the host budget as a share of the machine. |
| `src/inference/generic_transformer/wide_sparse_card_tier.rs`, `wide_expert_card.rs` | The card tier of a model larger than the cards and host together. Keeps the hottest routed experts on the cards when a card measures faster than the host. Every card the load was offered has a room, a card outside the plan for routed experts only. Each card runs its experts on a thread of its own. |
| `src/inference/generic_transformer/wide_sparse_host_tier.rs` | The host tier of the same experts. Keeps the routed experts of each layer that rank highest in the page cache, ranked by the decoded rows that routed to them or by recency, whichever a shadow of each layer measures to miss less, within what the host has left beside a burst, a layer's experts in flight and a sequence's state. Drops every other expert's pages once computed, so a prompt never pushes the decode's experts out, and reads ahead the next layer's guessed experts. |
| `src/inference/generic_transformer/wide_always_read.rs` | The weights every token of a wide-residual model reads. Splits each card's room between them and the routed experts by the host time a byte saves, measured at load. |
| `src/inference/generic_transformer/wide_sparse_card_stream.rs` | The residual of a single decoded row of a wide-residual model on a card. The Sinkhorn mixes, the collapse with its norm and the post of a layer whose mixers the load keeps on the card run there. A token mixer that runs its output projection on the same card reads its input there and leaves its output there. Only its key row and the rows its host state keeps come down. A router on the same card reads the feed-forward's input there. Its logits come down with that input and with the rows the token mixer's host state keeps. The feed-forward's output goes up. A residual that ends on the card that keeps the head is collapsed and normed there, and only the logits come down. A layer is admitted at its first decoded row when each card product is no farther from the exact one than the host's. |
| `src/inference/generic_transformer/expert_tier.rs`, `expert_readahead.rs` | Expert tiers over streamed experts: views into the mapping, a cache on the cards by use, reads ahead of the layer. Drops used experts from the page cache while the host's memory is short. |
| `src/distributed/routing.rs` | Picks the node that answers by predicted completion. [`CLUSTER.md`](CLUSTER.md) describes the routing. |

### Lesson 10: Measuring an engine without fooling yourself

[Lesson](https://loken-ai.github.io/loken/measuring/)

| Resource | Role |
|---|---|
| [Method](https://github.com/loken-ai/benchmarks/blob/main/docs/METHOD.md) | The protocol: the cell, the prompts, the cards each engine may use, what a cell without a published rate means, the `Tokens` column. |
| [Rate columns](https://github.com/loken-ai/benchmarks/blob/main/docs/METHOD.md#the-rate-columns) | Prefill is prompt tokens over the client's time to first token. Decode is the client's span. Both are medians. [Cold and warm](https://github.com/loken-ai/benchmarks/blob/main/docs/METHOD.md#cold-and-warm) figures are separate. |
| [`scripts/bench`](https://github.com/loken-ai/benchmarks/blob/main/scripts/bench) (benchmarks repository) | Harness: starts the releases named in its settings, stamps the build into every row, files each run. |
| [Benchmarks site](https://loken-ai.github.io/benchmarks/) | The runs of every machine, 1 page per hardware, rebuilt from the records. |
| `scripts/determinism.py` | Repeats a prompt and diffs the answers. |
| `src/inference/place/layer_perf.rs`, `src/energy.rs` | Per-layer timers (lesson 5) and joules: host RAPL and card NVML summed over the request window. |
| [lokprof](https://github.com/loken-ai/lokprof) | Bench tool: drives each engine through its own API, separates prefill from decode, records energy beside speed. |
