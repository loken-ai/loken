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

The `Publisher safetensors` column names the `architectures[0]` classes of the publisher's
`config.json` the loader reads (the class table in `src/inference/load/hf_config.rs`), `read as`
when the class loads under another architecture's tag, and `none:` with the reason when the publisher
ships no checkpoint the loader can read.

| Architecture | Publisher safetensors | Formats read | Test names |
|---|---|---|---|
| `llama` | `LlamaForCausalLM`, `MixtralForCausalLM`; native layout, dense and mixture-of-experts | GGUF; HF safetensors; native layout; AWQ safetensors | `llama_loads_from_gguf`, `llama_loads_from_safetensors`, `llama_loads_from_native_layout`, `llama_loads_from_awq_safetensors`, `llama_safetensors_equal_gguf`, `mixtral_loads_from_safetensors`, `mixtral_loads_from_native_layout` |
| `mistral` | `MistralForCausalLM` (loaded under the `llama` tag) | GGUF; HF safetensors; native layout; AWQ safetensors | `mistral_loads_from_gguf`, `mistral_loads_from_safetensors`, `mistral_loads_from_native_layout`, `mistral_loads_from_awq_safetensors` |
| `mistral3` | `Mistral3ForCausalLM`, `MistralForConditionalGeneration` (loaded under the `llama` tag) | GGUF; HF safetensors | `mistral3_loads_from_gguf`, `mistral3_loads_from_safetensors`, `mistral3_composite_loads_from_safetensors` |
| `smollm3` | `SmolLM3ForCausalLM` | GGUF; HF safetensors | `smollm3_loads_from_gguf`, `smollm3_loads_from_safetensors` |
| `ernie4_5` | `Ernie4_5ForCausalLM` | GGUF; HF safetensors | `ernie4_5_loads_from_gguf`, `ernie4_5_loads_from_safetensors` |
| `granite` | `GraniteForCausalLM` | GGUF; HF safetensors | `granite_loads_from_gguf`, `granite_loads_from_safetensors` |
| `granite3` | `GraniteForCausalLM`, read as `granite` | GGUF | `granite3_loads_from_gguf` |
| `granitemoe` | `GraniteMoeForCausalLM` | GGUF; HF safetensors | `granitemoe_loads_from_gguf`, `granitemoe_loads_from_safetensors` |
| `yi` | `YiForCausalLM` (the first release; later ones ship `LlamaForCausalLM`) | GGUF; HF safetensors | `yi_loads_from_gguf`, `yi_loads_from_safetensors` |
| `internlm2` | `InternLM2ForCausalLM` | GGUF; HF safetensors | `internlm2_loads_from_gguf`, `internlm2_loads_from_safetensors` |
| `starcoder` | `GPTBigCodeForCausalLM` | GGUF; HF safetensors | `starcoder_loads_from_gguf`, `starcoder_loads_from_safetensors` |
| `starcoder2` | `Starcoder2ForCausalLM` | GGUF; HF safetensors | `starcoder2_loads_from_gguf`, `starcoder2_loads_from_safetensors` |
| `qwen2` | `Qwen2ForCausalLM` | GGUF; HF safetensors; AWQ safetensors | `qwen2_loads_from_gguf`, `qwen2_loads_from_safetensors`, `qwen2_loads_from_awq_safetensors`, `qwen2_safetensors_equal_gguf` |
| `qwen3` | `Qwen3ForCausalLM` | GGUF; HF safetensors; AWQ safetensors | `qwen3_loads_from_gguf`, `qwen3_loads_from_safetensors`, `qwen3_loads_from_awq_safetensors`, `qwen3_safetensors_equal_gguf` |
| `qwen3moe` | `Qwen3MoeForCausalLM` | GGUF; HF safetensors | `qwen3moe_loads_from_gguf`, `qwen3moe_loads_from_safetensors` |
| `qwen35` | `Qwen3_5ForCausalLM`, `Qwen3_5ForConditionalGeneration` | GGUF; HF safetensors | `qwen35_loads_from_gguf`, `qwen35_loads_from_safetensors` |
| `qwen35moe` | `Qwen3_5MoeForCausalLM`, `Qwen3_5MoeForConditionalGeneration` | GGUF; HF safetensors | `qwen35moe_loads_from_gguf`, `qwen35moe_loads_from_safetensors` |
| `qwen3next` | `Qwen3NextForCausalLM` | GGUF; HF safetensors | `qwen3next_loads_from_gguf`, `qwen3next_loads_from_safetensors`, `qwen3next_loads_from_bf16_safetensors` |
| `qwen4exp` | none: no transformers class and no published checkpoint; read from GGUF only | GGUF | `qwen4exp_loads_from_gguf` |
| `phi` | `PhiForCausalLM`, read as `phi2` | GGUF | `phi_loads_from_gguf` |
| `phi2` | `PhiForCausalLM` | GGUF; HF safetensors | `phi2_loads_from_gguf`, `phi2_loads_from_safetensors` |
| `phi3` | `Phi3ForCausalLM` | GGUF; HF safetensors | `phi3_loads_from_gguf`, `phi3_loads_from_safetensors` |
| `phi4` | `Phi3ForCausalLM`, read as `phi3` | GGUF | `phi4_loads_from_gguf` |
| `chatglm` | `ChatGLMModel`, `ChatGLMForConditionalGeneration` | GGUF; HF safetensors | `chatglm_loads_from_gguf`, `chatglm_loads_from_safetensors` |
| `glm4` | `Glm4ForCausalLM` | GGUF; HF safetensors | `glm4_loads_from_gguf`, `glm4_loads_from_safetensors` |
| `stablelm` | `StableLmForCausalLM` | GGUF; HF safetensors | `stablelm_loads_from_gguf`, `stablelm_loads_from_safetensors` |
| `olmo2` | `Olmo2ForCausalLM` | GGUF; HF safetensors | `olmo2_loads_from_gguf`, `olmo2_loads_from_safetensors` |
| `olmoe` | `OlmoeForCausalLM` | GGUF; HF safetensors | `olmoe_loads_from_gguf`, `olmoe_loads_from_safetensors` |
| `moondream` | none: custom code only, tensor names that change with each revision, no transformers class | GGUF | `moondream_loads_from_gguf` |
| `falcon` | `FalconForCausalLM` | GGUF; HF safetensors | `falcon_loads_from_gguf`, `falcon_loads_from_safetensors` |
| `falcon3` | `LlamaForCausalLM`, read as `llama` | GGUF | `falcon3_loads_from_gguf` |
| `dbrx` | `DbrxForCausalLM` | GGUF; HF safetensors | `dbrx_loads_from_gguf`, `dbrx_loads_from_safetensors` |
| `orion` | `OrionForCausalLM` | GGUF; HF safetensors | `orion_loads_from_gguf`, `orion_loads_from_safetensors` |
| `mpt` | `MPTForCausalLM`, `MptForCausalLM` | GGUF; HF safetensors | `mpt_loads_from_gguf`, `mpt_loads_from_safetensors` |
| `gemma` | `GemmaForCausalLM` | GGUF; HF safetensors | `gemma_loads_from_gguf`, `gemma_loads_from_safetensors` |
| `gemma2` | `Gemma2ForCausalLM` | GGUF; HF safetensors | `gemma2_loads_from_gguf`, `gemma2_loads_from_safetensors` |
| `gemma3` | `Gemma3ForCausalLM`, `Gemma3ForConditionalGeneration` | GGUF; HF safetensors | `gemma3_loads_from_gguf`, `gemma3_loads_from_safetensors`, `gemma3_loads_from_bf16_safetensors` |
| `gemma4` | none: the per-layer input, shared key-value and key-as-value tensors of its checkpoint are not known well enough to translate | GGUF | `gemma4_loads_from_gguf` |
| `gptoss` | `GptOssForCausalLM` (MXFP4 blocks, or the dequantised experts) | GGUF; HF safetensors | `gptoss_loads_from_gguf`, `gptoss_loads_from_safetensors`, `gptoss_loads_from_mxfp4_safetensors`, `gptoss_loads_from_bf16_safetensors` |
| `lfm2` | `Lfm2ForCausalLM` | GGUF; HF safetensors | `lfm2_loads_from_gguf`, `lfm2_loads_from_safetensors` |
| `lfm2moe` | `Lfm2MoeForCausalLM` | GGUF; HF safetensors | `lfm2moe_loads_from_gguf`, `lfm2moe_loads_from_safetensors` |
| `nemotron_h_moe` | `NemotronHForCausalLM` | GGUF; HF safetensors | `nemotron_h_moe_loads_from_gguf`, `nemotron_h_moe_loads_from_safetensors` |
| `llama` text model of a speech release | `VoxtralForConditionalGeneration`; native layout with the table in the tower's namespace | HF safetensors composite; native layout | `voxtral_loads_from_safetensors`, `voxtral_loads_from_native_layout` |

## Notes

- Formats are GGUF (one file; a split set reads through the same tensor path, covered once in `loader.rs`), dense safetensors in the framework's layout (F32, F16, BF16; MXFP4 blocks for the experts of `gptoss`), AWQ safetensors, and the vendor's native layout (`params.json` and `consolidated*.safetensors`), dense or with routed experts.
- A checkpoint's `architectures[0]` selects a row of the class table in `src/inference/load/hf_config.rs`: the architecture tag, the tensor layout (`src/inference/generic_transformer/dense_layouts.rs`), and the conventions the class fixes. The layouts are data: a rule per part of a block says whether a stored tensor is read in place, reshaped, shifted by one (norm scales stored less one), split by head, stacked from per-expert tensors, halved, transposed or repacked (MXFP4). The config keys are read under the names each publisher uses (`hf_config.rs`); widths that the tensors state (the dense and the expert feed-forward, the short convolution, which blocks attend) are read from the tensors.
- Rotary rows: a class pairs lanes as the framework's rotation does (lane `i` with `i + n/2`) unless its row says adjacent. A GGUF converter reorders the rows of the tags whose GGUF files pair adjacent lanes; the checkpoint's rows are not reordered, the loader's pairing follows the class.
- A fused attention projection whose rows are grouped by key-value head (`InternLM2ForCausalLM`; `FalconForCausalLM` of a grouped decoder, `GPTBigCodeForCausalLM` without a shared head) is split into query, key and value rows when read; the others are read in place.
- The recipes write the tensor set each architecture's GGUF carries: fused or split projections, biases, QK norms, post norms, routed experts with shared experts, short convolution, Mamba2 and delta-rule blocks. Every recipe is written in F32; `tiny_model::write_quant` exercises K-quant weights for one architecture.
- `qwen4exp` loads the seeded toy file `tests/vectors/qwen4exp/toy.gguf`, since its wide residual, indexer and n-gram features are not part of a recipe. `wide_tests.rs` checks that model against the reference.
- A test checks the file says what the recipe meant (each block's mixer) and that a file missing a tensor of its block is refused (`a_file_missing_a_tensor_of_its_block_is_refused`). `a_wrong_layout_is_caught` loads checkpoints laid out wrongly (rows left in the file's pairing, a fused projection read in the wrong order, norms not shifted) and asserts their logits differ.
- The tests check that a model loads and runs, and that a publisher's checkpoint equals its GGUF recipe, not that it answers as the publisher's does. `mpt` positions come from ALiBi in the publisher's model; the loader has no ALiBi and rotates the heads, so the logits of a real `mpt` file are not the reference's. `starcoder` has learned positions the loader does not read.
- Not read, as from a GGUF file: LongRoPE factors (`phi3`), the linear rope scaling of the global layers (`gemma3`), `clip_qkv` (`dbrx`), the per-head query and key norms of the 12B `stablelm`. The grouped `falcon` decoder's two input norms are not mapped, so such a checkpoint is refused for a missing tensor. A quantised checkpoint other than AWQ of the llama, mistral and qwen families is refused.
- Architectures with their own model code (image, speech, video, forecasting, DeepSeek V4.1) are not read by the generic transformer and are not in this table.
