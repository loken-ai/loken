# Third-Party Notices

`loken` is licensed under **MIT OR Apache-2.0**. See [`LICENSE-MIT`](https://github.com/loken-ai/loken/blob/main/LICENSE-MIT) and [`LICENSE-APACHE`](https://github.com/loken-ai/loken/blob/main/LICENSE-APACHE).

`loken` builds on, vendors or links against the third-party works listed here. Each keeps its own license, and this file collects the required notices. The authoritative license of a crate dependency is the one shipped in its source tree, resolved in `Cargo.lock`.

## Dependency

| Component | Upstream | License |
|-----------|----------|---------|
| `cudarc` | [chelsea0x3b/cudarc](https://github.com/chelsea0x3b/cudarc) | MIT OR Apache-2.0 |
| `ash` | [ash-rs/ash](https://github.com/ash-rs/ash) | MIT OR Apache-2.0 |
| `naga` | [gfx-rs/wgpu](https://github.com/gfx-rs/wgpu) | MIT OR Apache-2.0 |

## Redistributed binaries

The `-cuda-libs` release archive carries NVIDIA's NVRTC, cuBLAS and cuBLASLt, redistributed under
the CUDA Toolkit EULA (`LICENSE-CUDA.txt` in the archive, Attachment A lists them), and NCCL
under its own licence (`LICENSE-NCCL.txt`) when the build machine provides it. The light archive
carries none of them.

## Bench recordings

The recordings the speech recognition cells and the audio-input check transcribe are kept, with
their credits, in the [benchmarks](https://github.com/loken-ai/benchmarks) repository.

## Borrowed code, measured

Each row states what a file still shares, verbatim, with the upstream it names.
The figures come from `scripts/provenance/manifest.tsv`, measured against real checkouts.
`src/notice_gate.rs` holds the table to them:

- At or above 30%, a file must have a row.
- Below 10%, a row must go.
- Between the 2, a row is a judgement call.

The goal is an empty table.

2 caveats:

- A snippet scanner sees reformatted copies that the line-for-line measure cannot, so the larger of the 2 figures is recorded.
- The largest range left in `mmq_gguf.cuh` is the GGUF k-quant bit specification: masks, shifts and strides that any correct reader reproduces. 2 restructurings left the arithmetic identical and the share slightly higher. This is convergent design over a published format, not residual copying.

The IQ2_XXS row is the same case in one file. Every matched line is the format's codebook, the point grid, the sign patterns and the mask that a reader of that format holds byte for byte. The dequantiser around them is ours.

**Chain of title.** The `cuda/mmq_gguf/` and `cuda/moe/` kernels reached this tree through candle, whose own copies are llama.cpp's. Candle matches `ggml/src/ggml-cuda` at 100%, 94%, 63% and 55% across those files. Those are candle's shares, not ours. They establish that llama.cpp (MIT) governs the kernels even where candle is the immediate upstream.
`cuda/marlin/` descends from [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin) via vLLM.
Files under the listing threshold that remain ports carry their attribution inline.

| Path | Share | Lines | Immediate upstream | License |
|------|-------|-------|--------------------|---------|
| `cuda/moe/gated_delta_net.cu` | 13% | 5 | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `ggml/src/ggml-cuda/gated_delta_net.cu` | MIT |
| `cuda/mmq_gguf/mmq_gguf.cuh` | 58% | - | [ggerganov/ggml](https://github.com/ggerganov/ggml) `ggml/src/ggml-cuda/mmq.cuh` | MIT |
| `src/tensor/quant_cpu/format/iq2_xxs.rs` | 53% | 64 | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `ggml/src/ggml-common.h` | MIT |

## Algorithms followed

The IQ3_XXS encoder in `src/tensor/quant_cpu/format/iq3_xxs.rs` follows the search of `quantize_row_iq3_xxs_impl` in [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `ggml/src/ggml-quants.c` (MIT). It follows the scale candidates, the weighting by importance and the nearest-point fallback for patterns off the grid. The code is written for loken.

The encoders below follow the same file's searches in the same way, written for loken:

| Path | Follows |
|------|---------|
| `src/tensor/quant_cpu/format/iq3_s.rs` | `quantize_row_iq3_s_impl` |
| `src/tensor/quant_cpu/format/iq2_s.rs` | `quantize_row_iq2_s_impl` |
| `src/tensor/quant_cpu/format/iq_codebook.rs` | the neighbour lists of `iq2xs_init_impl` and `iq3xs_init_impl` |
| `src/tensor/quant_cpu/format/iq4_nl.rs`, `iq4_xs.rs` | `quantize_row_iq4_nl_impl`, `quantize_iq4_xs` |
| `src/tensor/quant_cpu/format/legacy_guided.rs` | the importance weighting of `quantize_row_q4_0_impl` and its kin |

The card kernels for these formats in `src/inference/cuda/quantized/requant.cu` run the same searches as the host encoders above.

## Model weights

Model *weights* are not distributed with this repository. Downloading and using a model is subject to that model's own license and terms of use.
The references each model implementation was written against are architecture attribution, not borrowed code. They are listed in [`docs/REFERENCES.md`](https://github.com/loken-ai/loken/blob/main/docs/REFERENCES.md).

## Notable dependencies and their licenses

Permissive (MIT, Apache-2.0, BSD), among others: `tokio`, `axum`, `serde`, `reqwest`, `tokenizers`, `hf-hub`, `nvml-wrapper`, `opencl3`, `llguidance`, `toktrie_hf_tokenizers`, `image`, `rayon`, `half`, `safetensors`, `bytemuck`.

Weak copyleft (file-level), used unmodified as a dependency:

- **`symphonia`** (audio decoding: MP3, FLAC, OGG, AAC): **MPL-2.0**. Compatible with this project's license. It requires that modifications to the symphonia source files themselves remain under MPL-2.0. `loken` does not modify symphonia, so no additional obligation applies beyond this notice.

`cargo license` lists the complete, version-pinned dependency license set.
