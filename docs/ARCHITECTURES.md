# Architectures

Every architecture the loader accepts, the formats it reads, and the test that loads it.

The loader decides from tensors and metadata. The key `general.architecture` selects the row of
`src/inference/model_facts.rs`; a checkpoint's `architectures[0]` selects a GGUF tag in
`src/inference/load/awq_loader.rs`. This table lists the keys of the first and the classes of the second.

Each test lives in `src/inference/generic_transformer/loading_tests.rs`. It writes a tiny synthetic
checkpoint of the architecture in the format, loads it through `GenericWeights::build` (the call
the server makes) on a host-only `HeteroPlan`, runs a forward pass of five tokens, and asserts finite,
non-constant logits of the vocabulary's size. `every_architecture_has_a_loading_test` fails when the facts table
holds an architecture without a test.

| Architecture | Formats read | Loading test | Test names |
|---|---|---|---|
| `llama` | GGUF; HF safetensors (`LlamaForCausalLM`); native layout; AWQ safetensors | yes | `llama_loads_from_gguf`, `llama_loads_from_safetensors`, `llama_loads_from_native_layout`, `llama_loads_from_awq_safetensors` |
| `mistral` | GGUF; HF safetensors (`MistralForCausalLM`); native layout; AWQ safetensors | yes | `mistral_loads_from_gguf`, `mistral_loads_from_safetensors`, `mistral_loads_from_native_layout`, `mistral_loads_from_awq_safetensors` |
| `mistral3` | GGUF; HF safetensors (`Mistral3ForCausalLM`, `MistralForConditionalGeneration`) | yes | `mistral3_loads_from_gguf`, `mistral3_loads_from_safetensors`, `mistral3_composite_loads_from_safetensors` |
| `smollm3` | GGUF | yes | `smollm3_loads_from_gguf` |
| `ernie4_5` | GGUF | yes | `ernie4_5_loads_from_gguf` |
| `granite` | GGUF | yes | `granite_loads_from_gguf` |
| `granite3` | GGUF | yes | `granite3_loads_from_gguf` |
| `granitemoe` | GGUF | yes | `granitemoe_loads_from_gguf` |
| `yi` | GGUF | yes | `yi_loads_from_gguf` |
| `internlm2` | GGUF | yes | `internlm2_loads_from_gguf` |
| `starcoder` | GGUF | yes | `starcoder_loads_from_gguf` |
| `starcoder2` | GGUF | yes | `starcoder2_loads_from_gguf` |
| `qwen2` | GGUF; HF safetensors (`Qwen2ForCausalLM`); AWQ safetensors | yes | `qwen2_loads_from_gguf`, `qwen2_loads_from_safetensors`, `qwen2_loads_from_awq_safetensors` |
| `qwen3` | GGUF; HF safetensors (`Qwen3ForCausalLM`); AWQ safetensors | yes | `qwen3_loads_from_gguf`, `qwen3_loads_from_safetensors`, `qwen3_loads_from_awq_safetensors` |
| `qwen3moe` | GGUF | yes | `qwen3moe_loads_from_gguf` |
| `qwen35` | GGUF | yes | `qwen35_loads_from_gguf` |
| `qwen35moe` | GGUF | yes | `qwen35moe_loads_from_gguf` |
| `qwen3next` | GGUF | yes | `qwen3next_loads_from_gguf` |
| `qwen4exp` | GGUF | yes | `qwen4exp_loads_from_gguf` |
| `phi` | GGUF | yes | `phi_loads_from_gguf` |
| `phi2` | GGUF | yes | `phi2_loads_from_gguf` |
| `phi3` | GGUF | yes | `phi3_loads_from_gguf` |
| `phi4` | GGUF | yes | `phi4_loads_from_gguf` |
| `chatglm` | GGUF | yes | `chatglm_loads_from_gguf` |
| `glm4` | GGUF | yes | `glm4_loads_from_gguf` |
| `stablelm` | GGUF | yes | `stablelm_loads_from_gguf` |
| `olmo2` | GGUF | yes | `olmo2_loads_from_gguf` |
| `olmoe` | GGUF | yes | `olmoe_loads_from_gguf` |
| `moondream` | GGUF | yes | `moondream_loads_from_gguf` |
| `falcon` | GGUF | yes | `falcon_loads_from_gguf` |
| `falcon3` | GGUF | yes | `falcon3_loads_from_gguf` |
| `dbrx` | GGUF | yes | `dbrx_loads_from_gguf` |
| `orion` | GGUF | yes | `orion_loads_from_gguf` |
| `mpt` | GGUF | yes | `mpt_loads_from_gguf` |
| `gemma` | GGUF | yes | `gemma_loads_from_gguf` |
| `gemma2` | GGUF | yes | `gemma2_loads_from_gguf` |
| `gemma3` | GGUF | yes | `gemma3_loads_from_gguf` |
| `gemma4` | GGUF | yes | `gemma4_loads_from_gguf` |
| `gptoss` | GGUF | yes | `gptoss_loads_from_gguf` |
| `lfm2` | GGUF | yes | `lfm2_loads_from_gguf` |
| `lfm2moe` | GGUF | yes | `lfm2moe_loads_from_gguf` |
| `nemotron_h_moe` | GGUF | yes | `nemotron_h_moe_loads_from_gguf` |
| `llama` text model of a speech release | HF safetensors composite (`VoxtralForConditionalGeneration`); native layout with the table in the tower's namespace | yes | `voxtral_loads_from_safetensors`, `voxtral_loads_from_native_layout` |

## Notes

- Formats are GGUF (one file; a split set reads through the same tensor path, covered once in `loader.rs`), dense safetensors in the framework's layout (F32, F16, BF16), AWQ safetensors, and the vendor's native layout (`params.json` and `consolidated*.safetensors`).
- Only the llama, mistral, mistral3, qwen2 and qwen3 families load from safetensors; the loader maps no other class (`map_hf_arch`). A mixture-of-experts native layout is refused.
- The recipes write the tensor set each architecture's GGUF carries: fused or split projections, biases, QK norms, post norms, routed experts with shared experts, short convolution, Mamba2 and delta-rule blocks. Every recipe is written in F32; `tiny_model::write_quant` exercises K-quant weights for one architecture.
- `qwen4exp` loads the seeded toy file `tests/vectors/qwen4exp/toy.gguf`, since its wide residual, indexer and n-gram features are not part of a recipe. `wide_tests.rs` checks that model against the reference.
- A test checks the file says what the recipe meant (each block's mixer) and that a file missing a tensor of its block is refused (`a_file_missing_a_tensor_of_its_block_is_refused`).
- The tests check that a model loads and runs, not that it answers as the publisher's does. `mpt` positions come from ALiBi in the publisher's model; the loader has no ALiBi and rotates the heads, so the logits of a real `mpt` file are not the reference's.
- Architectures with their own model code (image, speech, video, forecasting, DeepSeek V4.1) are not read by the generic transformer and are not in this table.
