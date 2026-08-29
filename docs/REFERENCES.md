# References the model implementations were written against

This is attribution of ARCHITECTURES, not of code: a checkpoint dictates its tensor names,
shapes and operation order, and an implementation that departs from them does not load. The
code in these modules is this project's; each row names the reference it was validated
against. Borrowed code - files that still share measured body lines with an upstream - is
listed in `NOTICE.md` instead, driven by `scripts/provenance/manifest.tsv`.

A model checkpoint dictates its own architecture: the tensor names, the shapes and the order of
operations are fixed by the file, and an implementation that departs from them does not load at
all. The modules below implement those architectures. Each names the reference it was written
against, and the code is this project's.

If you maintain one of the projects named here and read this as under-crediting your work, open
an issue and it will be corrected.

| Module | What the reference supplied | Upstream model / authors | License |
|--------|-----------------------------|--------------------------|---------|
| `inference/model/flux/`, `inference/model/flux/facade.rs` | model architecture and weight layout | [black-forest-labs/flux](https://github.com/black-forest-labs/flux) | Apache-2.0 (code) |
| `inference/model/moondream/vision.rs`, `quantized_moondream.rs` | CLIP-style vision tower; layout verified against Ollama's moondream blob | [vikhyat/moondream](https://github.com/vikhyat/moondream) | Apache-2.0 |
| `inference/model/nemotron_h/` | hybrid Mamba2 + attention + MoE; SSM math from candle `mamba2` | NVIDIA Nemotron-H | see NVIDIA model license |
| `inference/fused_moe.rs` (MoE layer layout) | adapted from `vllm.rs` `models/layers/moe.rs` | [guoqingbao/vllm.rs](https://github.com/guoqingbao/vllm.rs) | MIT OR Apache-2.0 |
| `inference/load/awq.rs`, `inference/quantized_cuda/awq.rs` | AWQ 4-bit format/dequant conventions | [casper-hansen/AutoAWQ](https://github.com/casper-hansen/AutoAWQ) / [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq) | MIT |
| `inference/serve/eagle.rs` | EAGLE-1 speculative draft head, independent Rust reimplementation validated against the reference | [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) | Apache-2.0 |
| `inference/model/acestep/` | Rust port; the DWT scalers and the APG-guidance step follow `acestep.cpp` (`dwt-haar.h`, `dit-sampler.h`) as well as the Python reference, and the FSQ quantiser follows `vector_quantize_pytorch` | [ace-step/ACE-Step](https://github.com/ace-step/ACE-Step) / [acestep.cpp](https://github.com/ace-step/acestep.cpp) / [lucidrains/vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) | Apache-2.0 (code); MIT (vector-quantize-pytorch); model weights under their own license |
| `inference/model/wan/` | independent Rust reimplementation validated against the reference | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) | Apache-2.0 |
| `inference/model/ezaudio/` | independent Rust reimplementation validated against the reference | [haidog-yaqub/EzAudio](https://github.com/haidog-yaqub/EzAudio) | MIT |
| `inference/model/pocket_tts/`, `native_kyutai_*.rs` | independent Rust reimplementation validated against the reference (parity 1.0) | [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts) / [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) | MIT (code); some voice/weight assets CC-BY - see upstream |
| `inference/media/midi.rs` (AMT token scheme) | arrival-time MIDI token format | [slseanwu MIDI-LLM](https://huggingface.co/slseanwu) / [jthickstun/anticipation](https://github.com/jthickstun/anticipation) | Apache-2.0 (anticipation); MIDI-LLM weights under the Llama 3.2 license |
| `inference/model/qwen25/vision.rs` | vision tower layout per the reference implementation | [QwenLM/Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) | Apache-2.0 |
| `inference/cache/paged_attention.rs`, `continuous_batch*.rs` (paged-KV/continuous-batching design) | algorithmic design after the vLLM paper/implementation | [vllm-project/vllm](https://github.com/vllm-project/vllm) | Apache-2.0 |
| `inference/model/qwen35/moe.rs` (gated delta rule) | recurrent form ported from vLLM's `recurrent_gated_delta_rule` | [vllm-project/vllm](https://github.com/vllm-project/vllm) | Apache-2.0 |
| `inference/model/qwen35/vision.rs` | 2D rope and vision-tower layout ported from Ollama's Go implementation and llama.cpp's `qwen3vl.cpp` | [ollama/ollama](https://github.com/ollama/ollama) / [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) / [QwenLM](https://github.com/QwenLM) | MIT |
| `inference/model/sdxl/` | SDXL text conditioning, UNet block layout and VAE name mapping, implemented against the checkpoint; the noise schedules and DPM++ solvers transcribe the published formulations (Karras et al. 2022; DPM-Solver++) | [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models) / [crowsonkb/k-diffusion](https://github.com/crowsonkb/k-diffusion) | MIT (k-diffusion) |
| `inference/model/flux2/` | ported against the `diffusers` reference (`transformer_flux2.py`), including the empirical `mu` fit | [huggingface/diffusers](https://github.com/huggingface/diffusers) / [black-forest-labs](https://github.com/black-forest-labs) | Apache-2.0 |
| `inference/model/qwen_image/` | config and block/attention/rope spec captured from the `diffusers` reference (`transformer_qwenimage.py`) | [huggingface/diffusers](https://github.com/huggingface/diffusers) / [QwenLM Qwen-Image](https://github.com/QwenLM) | Apache-2.0 |
| `inference/model/boogu/dit.rs` | dual-stream MMDiT following the OmniGen2 and Lumina2 architectures the checkpoint implements | [VectorSpaceLab/OmniGen2](https://github.com/VectorSpaceLab/OmniGen2) / [Alpha-VLLM/Lumina-Image-2.0](https://github.com/Alpha-VLLM/Lumina-Image-2.0) | Apache-2.0 |
| `inference/sample/flow_unipc.rs` | port of `FlowUniPCMultistepScheduler`, after the UniPC solver (Zhao et al. 2023) | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) / [wl-zhao/UniPC](https://github.com/wl-zhao/UniPC) | Apache-2.0 |
| `inference/model/stable_audio/`, `inference/model/ezaudio/vae.rs` | DiT and VAE key schemes and weight-norm folding per the reference | [Stability-AI/stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) | MIT |
| `inference/codec/dac.rs` | Descript Audio Codec encoder/decoder | [descriptinc/descript-audio-codec](https://github.com/descriptinc/descript-audio-codec) | MIT |
| `inference/model/piper/`, `inference/load/onnx.rs` | VITS forward pass and the ONNX tensor map Piper ships its voices in | [rhasspy/piper](https://github.com/rhasspy/piper) / [jaywalnut310/vits](https://github.com/jaywalnut310/vits) | MIT |
| `inference/codec/melband.rs` | Mel-Band RoFormer architecture and the chunked overlap-add `demix` loop | [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer) / [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) | MIT |
| `inference/token/sentencepiece.rs` | unigram tokenizer and the `ModelProto` wire format | [google/sentencepiece](https://github.com/google/sentencepiece) | Apache-2.0 |
| `inference/model/clip/`, `inference/model/clip/vision.rs` | CLIP text/vision towers | [openai/CLIP](https://github.com/openai/CLIP) | MIT |
| `inference/model/t5/encoder.rs`, `inference/model/umt5/encoder.rs` | T5/UMT5 encoders, including a port of `_relative_position_bucket` | [huggingface/transformers](https://github.com/huggingface/transformers) / [google-research/text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer) | Apache-2.0 |
| `inference/model/ultravox/` | projector and stacking factor per `ultravox_model.py` | [fixie-ai/ultravox](https://github.com/fixie-ai/ultravox) | MIT |
| `inference/model/pixtral/`, `inference/model/voxtral/` | vision/audio towers read from the llama.cpp `mmproj` GGUF layout | [mistralai](https://huggingface.co/mistralai) via [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | Apache-2.0 (models); MIT (llama.cpp) |

Model *weights* are not distributed with this repository. Downloading and
using any model is subject to that model's own license and terms of use
(e.g. the FLUX.1 weights and the Z-Image weights carry their own,
sometimes non-commercial, weight licenses distinct from the code license).
