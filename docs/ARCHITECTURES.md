# Architectures

Every architecture the loader accepts, the formats it reads, and the test that loads it.

The loader decides from tensors and metadata. The key `general.architecture` selects the row of
`src/inference/model_facts.rs`; a checkpoint's `architectures[0]` selects a GGUF tag in
`src/inference/load/awq_loader.rs`. This table lists the keys of the first and the classes of the second.

Each test lives in `src/inference/generic_transformer/loading_tests.rs`. It writes a tiny synthetic
checkpoint of the architecture in the format, loads it through `GenericWeights::build` (the call
the server makes) on a host-only `HeteroPlan`, runs a forward pass of five tokens, and asserts finite,
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
`tests/vectors/hf/<folder>`, and records the logits of the last position of a six-token prompt and of
the next four greedy tokens, run in float32 on the CPU. The test loads each folder through the same
entry point on a host-only plan, feeds the reference's ids back in turn, and asserts a cosine of at
least 0.998 with the reference's logits at every step and the same argmax wherever the two top
reference logits are further apart than twice the measured rounding error (float weights are rebuilt
in 8 bits). Every hybrid keeps one block of each kind; a seed is kept only when its routing decisions
and its logits stay clear of that rounding. `every_reference_folder_has_a_test` fails when a folder
and the list of tests differ. A composite release whose vendor also ships a native layout gets
`params.json` and `consolidated.safetensors` of the same weights beside it, checked by the library's
own converter of that layout; its native tests assert the reference's logits and those of the
composite.

The `Publisher safetensors` column names the `architectures[0]` classes of the publisher's
`config.json` the loader reads (the class table in `src/inference/load/hf_config.rs`), `read as`
when the class loads under another architecture's tag, and `none:` with the reason when the publisher
ships no checkpoint the loader can read. The `Verified` column says how the safetensors layout is
checked: `reference` (the folders whose logits it matches), `synthetic` (equality with the GGUF
recipe), `real checkpoint` (a published checkpoint checked by hand).

| Architecture | Publisher safetensors | Formats read | Test names | Verified |
|---|---|---|---|---|
| `llama` | `LlamaForCausalLM`, `MixtralForCausalLM`; native layout, dense and mixture-of-experts | GGUF; HF safetensors; native layout; AWQ safetensors | `llama_loads_from_gguf`, `llama_loads_from_safetensors`, `llama_loads_from_native_layout`, `llama_loads_from_awq_safetensors`, `llama_safetensors_equal_gguf`, `mixtral_loads_from_safetensors`, `mixtral_loads_from_native_layout` | reference (`llama`, `mixtral`); synthetic |
| `mistral` | `MistralForCausalLM` (loaded under the `llama` tag) | GGUF; HF safetensors; native layout; AWQ safetensors | `mistral_loads_from_gguf`, `mistral_loads_from_safetensors`, `mistral_loads_from_native_layout`, `mistral_loads_from_awq_safetensors` | reference (`mistral`); synthetic |
| `mistral3` | `Mistral3ForCausalLM`, `MistralForConditionalGeneration`, `Mistral3ForConditionalGeneration` (loaded under the `llama` tag); YaRN with its table scale, the query scale by position caps the context | GGUF; HF safetensors composite; native layout | `mistral3_loads_from_gguf`, `mistral3_loads_from_safetensors`, `mistral3_composite_loads_from_safetensors`, `mistral3_matches_reference`, `mistral3_native_matches_reference`, `mistral3_native_matches_composite` | reference (`mistral3`); synthetic |
| `smollm3` | `SmolLM3ForCausalLM` | GGUF; HF safetensors | `smollm3_loads_from_gguf`, `smollm3_loads_from_safetensors` | reference; synthetic; real checkpoint |
| `ernie4_5` | `Ernie4_5ForCausalLM` | GGUF; HF safetensors | `ernie4_5_loads_from_gguf`, `ernie4_5_loads_from_safetensors` | reference; synthetic; real checkpoint |
| `granite` | `GraniteForCausalLM` | GGUF; HF safetensors | `granite_loads_from_gguf`, `granite_loads_from_safetensors` | reference; synthetic; real checkpoint |
| `granite3` | `GraniteForCausalLM`, read as `granite` | GGUF | `granite3_loads_from_gguf` | through `granite` |
| `granitemoe` | `GraniteMoeForCausalLM` | GGUF; HF safetensors | `granitemoe_loads_from_gguf`, `granitemoe_loads_from_safetensors` | reference; synthetic; real checkpoint |
| `yi` | `YiForCausalLM` (the first release; later ones ship `LlamaForCausalLM`) | GGUF; HF safetensors | `yi_loads_from_gguf`, `yi_loads_from_safetensors` | synthetic; no transformers class (`YiForCausalLM` is custom code) |
| `internlm2` | `InternLM2ForCausalLM` | GGUF; HF safetensors | `internlm2_loads_from_gguf`, `internlm2_loads_from_safetensors` | synthetic; real checkpoint; no transformers class (custom code) |
| `starcoder` | `GPTBigCodeForCausalLM` | GGUF; HF safetensors | `starcoder_loads_from_gguf`, `starcoder_loads_from_safetensors` | reference; synthetic; real checkpoint (santacoder) |
| `starcoder2` | `Starcoder2ForCausalLM` | GGUF; HF safetensors | `starcoder2_loads_from_gguf`, `starcoder2_loads_from_safetensors` | reference; synthetic; real checkpoint |
| `qwen2` | `Qwen2ForCausalLM` | GGUF; HF safetensors; AWQ safetensors | `qwen2_loads_from_gguf`, `qwen2_loads_from_safetensors`, `qwen2_loads_from_awq_safetensors`, `qwen2_safetensors_equal_gguf` | reference; synthetic; real checkpoint |
| `qwen3` | `Qwen3ForCausalLM` | GGUF; HF safetensors; AWQ safetensors | `qwen3_loads_from_gguf`, `qwen3_loads_from_safetensors`, `qwen3_loads_from_awq_safetensors`, `qwen3_safetensors_equal_gguf` | reference; synthetic; real checkpoint |
| `qwen3moe` | `Qwen3MoeForCausalLM` | GGUF; HF safetensors | `qwen3moe_loads_from_gguf`, `qwen3moe_loads_from_safetensors` | reference; synthetic |
| `qwen35` | `Qwen3_5ForCausalLM`, `Qwen3_5ForConditionalGeneration` | GGUF; HF safetensors | `qwen35_loads_from_gguf`, `qwen35_loads_from_safetensors` | reference; synthetic |
| `qwen35moe` | `Qwen3_5MoeForCausalLM`, `Qwen3_5MoeForConditionalGeneration` | GGUF; HF safetensors | `qwen35moe_loads_from_gguf`, `qwen35moe_loads_from_safetensors` | reference; synthetic |
| `qwen3next` | `Qwen3NextForCausalLM` | GGUF; HF safetensors | `qwen3next_loads_from_gguf`, `qwen3next_loads_from_safetensors`, `qwen3next_loads_from_bf16_safetensors` | reference; synthetic |
| `qwen4exp` | none: no transformers class and no published checkpoint; read from GGUF only | GGUF | `qwen4exp_loads_from_gguf` | GGUF toy file against its reference (`wide_tests.rs`); no transformers class |
| `phi` | `PhiForCausalLM`, read as `phi2` | GGUF | `phi_loads_from_gguf` | through `phi2` |
| `phi2` | `PhiForCausalLM` | GGUF; HF safetensors | `phi2_loads_from_gguf`, `phi2_loads_from_safetensors` | reference; synthetic; real checkpoint |
| `phi3` | `Phi3ForCausalLM` | GGUF; HF safetensors | `phi3_loads_from_gguf`, `phi3_loads_from_safetensors` | reference; synthetic; real checkpoint |
| `phi4` | `Phi3ForCausalLM`, read as `phi3` | GGUF | `phi4_loads_from_gguf` | through `phi3` |
| `chatglm` | `ChatGLMModel`, `ChatGLMForConditionalGeneration` | GGUF; HF safetensors | `chatglm_loads_from_gguf`, `chatglm_loads_from_safetensors` | synthetic; no transformers class (custom code) |
| `glm4` | `Glm4ForCausalLM` | GGUF; HF safetensors | `glm4_loads_from_gguf`, `glm4_loads_from_safetensors` | reference; synthetic |
| `stablelm` | `StableLmForCausalLM` | GGUF; HF safetensors | `stablelm_loads_from_gguf`, `stablelm_loads_from_safetensors` | reference; synthetic; real checkpoint |
| `olmo2` | `Olmo2ForCausalLM` | GGUF; HF safetensors | `olmo2_loads_from_gguf`, `olmo2_loads_from_safetensors` | reference; synthetic; real checkpoint |
| `olmoe` | `OlmoeForCausalLM` | GGUF; HF safetensors | `olmoe_loads_from_gguf`, `olmoe_loads_from_safetensors` | reference; synthetic; real checkpoint |
| `moondream` | none: custom code only, tensor names that change with each revision, no transformers class | GGUF | `moondream_loads_from_gguf` | no transformers class (custom code) |
| `falcon` | `FalconForCausalLM` | GGUF; HF safetensors | `falcon_loads_from_gguf`, `falcon_loads_from_safetensors` | reference (multi-query decoder); synthetic |
| `falcon3` | `LlamaForCausalLM`, read as `llama` | GGUF | `falcon3_loads_from_gguf` | through `llama` |
| `dbrx` | `DbrxForCausalLM` | GGUF; HF safetensors | `dbrx_loads_from_gguf`, `dbrx_loads_from_safetensors` | reference; synthetic |
| `orion` | `OrionForCausalLM` | GGUF; HF safetensors | `orion_loads_from_gguf`, `orion_loads_from_safetensors` | synthetic; no transformers class (custom code) |
| `mpt` | `MPTForCausalLM`, `MptForCausalLM` | GGUF; HF safetensors | `mpt_loads_from_gguf`, `mpt_loads_from_safetensors` | reference; synthetic |
| `gemma` | `GemmaForCausalLM` | GGUF; HF safetensors | `gemma_loads_from_gguf`, `gemma_loads_from_safetensors` | reference; synthetic |
| `gemma2` | `Gemma2ForCausalLM` | GGUF; HF safetensors | `gemma2_loads_from_gguf`, `gemma2_loads_from_safetensors` | reference; synthetic; real checkpoint |
| `gemma3` | `Gemma3ForCausalLM`, `Gemma3ForConditionalGeneration` | GGUF; HF safetensors | `gemma3_loads_from_gguf`, `gemma3_loads_from_safetensors`, `gemma3_loads_from_bf16_safetensors` | reference; synthetic; real checkpoint |
| `gemma4` | `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`; per-layer inputs, shared key-value layers, keys read as values, routed experts beside the dense feed-forward | GGUF; HF safetensors | `gemma4_loads_from_gguf`, `gemma4_matches_reference`, `gemma4_keqv_matches_reference`, `gemma4_moe_matches_reference` | reference (`gemma4`, `gemma4_keqv`, `gemma4_moe`) |
| `gptoss` | `GptOssForCausalLM` (MXFP4 blocks, or the dequantised experts) | GGUF; HF safetensors | `gptoss_loads_from_gguf`, `gptoss_loads_from_safetensors`, `gptoss_loads_from_mxfp4_safetensors`, `gptoss_loads_from_bf16_safetensors` | reference; synthetic; real checkpoint |
| `lfm2` | `Lfm2ForCausalLM` | GGUF; HF safetensors | `lfm2_loads_from_gguf`, `lfm2_loads_from_safetensors` | reference; synthetic; real checkpoint |
| `lfm2moe` | `Lfm2MoeForCausalLM` | GGUF; HF safetensors | `lfm2moe_loads_from_gguf`, `lfm2moe_loads_from_safetensors` | reference; synthetic |
| `nemotron_h_moe` | `NemotronHForCausalLM` | GGUF; HF safetensors | `nemotron_h_moe_loads_from_gguf`, `nemotron_h_moe_loads_from_safetensors` | reference; synthetic |
| `llama` text model of a speech release | `VoxtralForConditionalGeneration`; native layout with the table in the tower's namespace | HF safetensors composite; native layout | `voxtral_loads_from_safetensors`, `voxtral_loads_from_native_layout`, `voxtral_matches_reference`, `voxtral_native_matches_reference`, `voxtral_native_matches_composite` | reference (`voxtral`); synthetic |

## Notes

- Formats are GGUF (one file; a split set reads through the same tensor path, covered once in `loader.rs`), dense safetensors in the framework's layout (F32, F16, BF16; MXFP4 blocks for the experts of `gptoss`), AWQ safetensors, and the vendor's native layout (`params.json` and `consolidated*.safetensors`), dense or with routed experts.
- A checkpoint's `architectures[0]` selects a row of the class table in `src/inference/load/hf_config.rs`: the architecture tag, the tensor layout (`src/inference/generic_transformer/dense_layouts.rs`), and the conventions the class fixes. The layouts are data: a rule per part of a block says whether a stored tensor is read in place, reshaped, shifted by one (norm scales stored less one), split by head, stacked from per-expert tensors, halved, transposed or repacked (MXFP4). The config keys are read under the names each publisher uses (`hf_config.rs`); widths that the tensors state (the dense and the expert feed-forward, the short convolution, which blocks attend) are read from the tensors.
- Rotary rows: a class pairs lanes as the framework's rotation does (lane `i` with `i + n/2`) unless its row says adjacent. A GGUF converter reorders the rows of the tags whose GGUF files pair adjacent lanes; the checkpoint's rows are not reordered, the loader's pairing follows the class.
- A fused attention projection whose rows are grouped by key-value head (`InternLM2ForCausalLM`; `FalconForCausalLM` of a grouped decoder, `GPTBigCodeForCausalLM` without a shared head) is split into query, key and value rows when read; the others are read in place.
- The recipes write the tensor set each architecture's GGUF carries: fused or split projections, biases, QK norms, post norms, routed experts with shared experts, short convolution, Mamba2 and delta-rule blocks. Every recipe is written in F32; `tiny_model::write_quant` exercises K-quant weights for one architecture.
- `qwen4exp` loads the seeded toy file `tests/vectors/qwen4exp/toy.gguf`, since its wide residual, indexer and n-gram features are not part of a recipe. `wide_tests.rs` checks that model against the reference.
- A test checks the file says what the recipe meant (each block's mixer) and that a file missing a tensor of its block is refused (`a_file_missing_a_tensor_of_its_block_is_refused`). `a_wrong_layout_is_caught` loads checkpoints laid out wrongly (rows left in the file's pairing, a fused projection read in the wrong order, norms not shifted) and asserts their logits differ.
- The synthetic tests check that a model loads and runs, and that a publisher's checkpoint equals its GGUF recipe; the reference tests check that it answers as the reference does. `mpt` positions come from ALiBi (`attention.max_alibi_bias`), which only the unfused host attention adds to the scores: the load offers such a model no card, Vulkan or OpenCL device and logs why, and a card's attention refuses such a layer (`an_alibi_model_stays_off_the_cards`, run with a card).
- A query scale by position (`llama_4_scaling_beta`, `attention.temperature_scale` and `attention.temperature_length`) is not applied: the context stops at the length before which the scale is one, and the load logs it.
- Not read, as from a GGUF file: LongRoPE factors (`phi3`), the linear rope scaling of the global layers (`gemma3`), `clip_qkv` (`dbrx`), the per-head query and key norms of the 12B `stablelm`. The reference vectors hold none of these. The grouped `falcon` decoder's two input norms are not mapped, so such a checkpoint is refused for a missing tensor. A quantised checkpoint other than AWQ of the llama, mistral and qwen families is refused.
- Architectures with their own model code (image, speech, video, forecasting, DeepSeek V4.1) are not read by the generic transformer and are not in this table.
