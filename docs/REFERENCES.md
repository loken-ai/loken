# References

Each row names the reference a model implementation was validated against.  
This page attributes architectures, not code.
A checkpoint dictates its tensor names, shapes and operation order.

The code in these modules is this project's.  
Borrowed code is code in files that still share measured body lines with an upstream.
`NOTICE.md` lists it, driven by `scripts/provenance/manifest.tsv`.

If you maintain a project named here and read this page as under-crediting your work, open an issue.
The page will be corrected.

| Module | What the reference supplied | Upstream model or authors | License |
|---|---|---|---|
| `inference/model/flux/` | model architecture and weight layout | [black-forest-labs/flux](https://github.com/black-forest-labs/flux) | Apache-2.0 (code) |
| `inference/model/moondream/vision.rs`, `inference/model/moondream/quantized.rs` | CLIP-style vision tower, layout checked against Ollama's moondream blob | [vikhyat/moondream](https://github.com/vikhyat/moondream) | Apache-2.0 |
| `inference/generic_transformer/mixer.rs` (Mamba2 mixer) | hybrid Mamba2, attention and MoE, SSM math from candle `mamba2` | NVIDIA Nemotron-H | see NVIDIA model license |
| `inference/generic_transformer/moe_weights.rs` (MoE layer layout) | adapted from `models/layers/moe.rs` of vllm.rs | [guoqingbao/vllm.rs](https://github.com/guoqingbao/vllm.rs) | MIT OR Apache-2.0 |
| `inference/load/awq.rs`, `inference/quantized_cuda/awq.rs` | AWQ 4-bit format and dequantisation conventions | [casper-hansen/AutoAWQ](https://github.com/casper-hansen/AutoAWQ) / [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq) | MIT |
| `inference/serve/eagle.rs` | EAGLE-1 speculative draft head, independent Rust reimplementation validated against the reference | [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) | Apache-2.0 |
| `inference/model/acestep/` | Rust port. The DWT scalers and the APG-guidance step follow `acestep.cpp` (`dwt-haar.h`, `dit-sampler.h`) and the Python reference. The FSQ quantiser follows `vector_quantize_pytorch` | [ace-step/ACE-Step](https://github.com/ace-step/ACE-Step) / [acestep.cpp](https://github.com/ace-step/acestep.cpp) / [lucidrains/vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) | Apache-2.0 (code), MIT (vector-quantize-pytorch), model weights under their own licenses |
| `inference/model/wan/` | independent Rust reimplementation validated against the reference | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) | Apache-2.0 |
| `inference/model/ezaudio/` | independent Rust reimplementation validated against the reference | [haidog-yaqub/EzAudio](https://github.com/haidog-yaqub/EzAudio) | MIT |
| `inference/model/pocket_tts/`, `inference/model/kyutai/`, `inference/model/codec_blocks.rs` | independent Rust reimplementation validated against the reference | [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts) / [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) | MIT (code), some voice/weight assets CC-BY, see upstream |
| `inference/media/midi.rs` (AMT token scheme) | arrival-time MIDI token format | [slseanwu MIDI-LLM](https://huggingface.co/slseanwu) / [jthickstun/anticipation](https://github.com/jthickstun/anticipation) | Apache-2.0 (anticipation), MIDI-LLM weights under the Llama 3.2 license |
| `inference/model/qwen3_tts/` | prompt layout, frame loop and sampling of the reference generation, the clone prompt, the named-voice prompt and its dialect language, the speaker encoder with its mel front end, and the layers of the codec decoder, validated against its fixtures | [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | Apache-2.0 |
| `inference/model/qwen25/vision.rs` | vision tower layout per the reference implementation | [QwenLM/Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) | Apache-2.0 |
| `inference/cache/paged_attention.rs`, `inference/serve/continuous_batch.rs` (paged KV and continuous batching) | algorithmic design after the vLLM paper and implementation | [vllm-project/vllm](https://github.com/vllm-project/vllm) | Apache-2.0 |
| `inference/generic_transformer/mixer.rs`, `inference/moe_cuda/dense/gdn.rs` (gated delta rule) | recurrent form ported from `recurrent_gated_delta_rule` of vLLM | [vllm-project/vllm](https://github.com/vllm-project/vllm) | Apache-2.0 |
| `inference/model/qwen35/vision.rs` | 2D rope and vision-tower layout ported from Ollama's Go implementation and llama.cpp's `qwen3vl.cpp` | [ollama/ollama](https://github.com/ollama/ollama) / [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) / [QwenLM](https://github.com/QwenLM) | MIT |
| `inference/model/sdxl/` | SDXL text conditioning, UNet block layout and VAE name mapping, implemented against the checkpoint. The noise schedules and DPM++ solvers transcribe the published formulations (Karras et al. 2022, DPM-Solver++) | [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models) / [crowsonkb/k-diffusion](https://github.com/crowsonkb/k-diffusion) | MIT (k-diffusion) |
| `inference/model/flux2/` | ported against the `diffusers` reference (`transformer_flux2.py`), including the empirical `mu` fit | [huggingface/diffusers](https://github.com/huggingface/diffusers) / [black-forest-labs](https://github.com/black-forest-labs) | Apache-2.0 |
| `inference/model/qwen_image/` | config and block, attention and rope spec captured from the `diffusers` reference (`transformer_qwenimage.py`) | [huggingface/diffusers](https://github.com/huggingface/diffusers) / [QwenLM Qwen-Image](https://github.com/QwenLM) | Apache-2.0 |
| `inference/model/boogu/dit.rs` | dual-stream MMDiT following the OmniGen2 and Lumina2 architectures the checkpoint implements | [VectorSpaceLab/OmniGen2](https://github.com/VectorSpaceLab/OmniGen2) / [Alpha-VLLM/Lumina-Image-2.0](https://github.com/Alpha-VLLM/Lumina-Image-2.0) | Apache-2.0 |
| `inference/sample/flow_unipc.rs` | port of `FlowUniPCMultistepScheduler`, after the UniPC solver (Zhao et al. 2023) | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) / [wl-zhao/UniPC](https://github.com/wl-zhao/UniPC) | Apache-2.0 |
| `inference/model/stable_audio/`, `inference/model/ezaudio/vae.rs` | DiT and VAE key schemes and weight-norm folding per the reference | [Stability-AI/stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) | MIT |
| `inference/codec/dac.rs` | Descript Audio Codec encoder and decoder | [descriptinc/descript-audio-codec](https://github.com/descriptinc/descript-audio-codec) | MIT |
| `inference/model/piper/`, `inference/load/onnx.rs` | VITS forward pass and the ONNX tensor map Piper ships its voices in | [rhasspy/piper](https://github.com/rhasspy/piper) / [jaywalnut310/vits](https://github.com/jaywalnut310/vits) | MIT |
| `inference/codec/melband.rs` | Mel-Band RoFormer architecture and the chunked overlap-add `demix` loop | [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer) / [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) | MIT |
| `inference/token/sentencepiece.rs` | unigram tokenizer and the `ModelProto` wire format | [google/sentencepiece](https://github.com/google/sentencepiece) | Apache-2.0 |
| `inference/model/clip/` | CLIP and SigLIP text and vision towers, the SigLIP attention-pool head, and the image processor's resize, crop and normalisation | [openai/CLIP](https://github.com/openai/CLIP) / [huggingface/transformers](https://github.com/huggingface/transformers) | MIT (CLIP), Apache-2.0 (transformers) |
| `inference/model/t5/encoder.rs`, `inference/model/umt5/encoder.rs` | T5 and UMT5 encoders, including a port of `_relative_position_bucket` | [huggingface/transformers](https://github.com/huggingface/transformers) / [google-research/text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer) | Apache-2.0 |
| `inference/model/ultravox/` | projector and stacking factor per `ultravox_model.py` | [fixie-ai/ultravox](https://github.com/fixie-ai/ultravox) | MIT |
| `inference/model/pixtral/`, `inference/model/voxtral/` | vision and audio towers read from the llama.cpp `mmproj` GGUF layout | [mistralai](https://huggingface.co/mistralai) via [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | Apache-2.0 (models), MIT (llama.cpp) |
| `inference/generic_transformer/`, `inference/load/hf_config.rs` (layouts of the class table) | tensor layout, rotary pairing and operation order of each class, held to the logits of the library's classes in `tests/vectors/hf/` | [huggingface/transformers](https://github.com/huggingface/transformers) | Apache-2.0 |
| `inference/model/whisper/` | encoder and decoder layout, and the log-mel front end ported from whisper.cpp's audio code | [openai/whisper](https://github.com/openai/whisper) / [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp) | MIT |
| `inference/model/zimage/` | the `ZImageTransformer2DModel` blocks, attention and rope, and the diffusers export's tensor names | [huggingface/diffusers](https://github.com/huggingface/diffusers) / [Tongyi-MAI](https://huggingface.co/Tongyi-MAI) | Apache-2.0 (code), model weights under their own license |
| `inference/model/parler/` | the decoder over delayed codebooks and the stopping rule of the reference processor | [huggingface/parler-tts](https://github.com/huggingface/parler-tts) | Apache-2.0 |

Model weights are not distributed with this repository.  
Each model's license and terms of use govern its download and use.
The FLUX.1 and Z-Image weights, for example, carry weight licenses distinct from the code license, some non-commercial.
