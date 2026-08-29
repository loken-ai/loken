# Third-Party Notices

`loken` is licensed under the terms of **MIT OR Apache-2.0** (see
[`LICENSE-MIT`](LICENSE-MIT) and [`LICENSE-APACHE`](LICENSE-APACHE)).

It builds on, vendors, or links against the third-party works listed below. Each remains
under its own license; this file collects the required notices. The authoritative license for
any crate dependency is the one shipped in its own source tree (resolved in `Cargo.lock`).

## Patched dependency

| Component | Upstream | License |
|-----------|----------|---------|
| `cudarc` | [chelsea0x3b/cudarc](https://github.com/chelsea0x3b/cudarc) | MIT OR Apache-2.0 |

A small delta over `cudarc` lives at [loken-ai/cudarc](https://github.com/loken-ai/cudarc),
which carries the patched sources and their upstream licence files, and states what has to
happen for it to stop existing.

## Borrowed code, measured

Each row states what a file still shares, verbatim, with the upstream it names. Figures come
from `scripts/provenance/manifest.tsv`, measured against real checkouts. `src/notice_gate.rs`
holds the table to them: at or above 30% a file must have a row, below 10% a row must go, and
between the two is a judgement call. The goal is an empty table.

Two caveats. A snippet scanner sees reformatted copies our line-for-line measure cannot, so the
larger of the two figures is what is recorded. And the largest range left in `mmq_gguf.cuh` is
the GGUF k-quant bit specification - masks, shifts and strides any correct reader reproduces  - 
so two restructurings left the arithmetic identical and the share slightly higher. Convergent
design over a published format, not residual copying.

**Chain of title.** The `cuda/mmq_gguf/` and `cuda/moe/` kernels reached this tree through
candle, whose own copies are llama.cpp's - candle matches `ggml/src/ggml-cuda` at 100%, 94%, 63%
and 55% across those files. Those are candle's shares, not ours; they establish that llama.cpp
(MIT) governs the kernels even where candle is the immediate upstream. `cuda/marlin/` descends
from [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin) via vLLM. Files under the listing
threshold that remain ports carry their attribution inline.

| Path | Share | Lines | Immediate upstream | License |
|------|-------|-------|--------------------|---------|
| `cuda/moe/gated_delta_net.cu` | 13% | 5 | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `ggml/src/ggml-cuda/gated_delta_net.cu` | MIT |
| `cuda/mmq_gguf/mmq_gguf.cuh` | 58% | - | [ggerganov/ggml](https://github.com/ggerganov/ggml) `ggml/src/ggml-cuda/mmq.cuh` | MIT |

Model *weights* are not distributed with this repository. Downloading and using any model is
subject to that model's own license and terms of use. The references each model
implementation was written against - architecture attribution, not borrowed code - are listed
in [`docs/REFERENCES.md`](docs/REFERENCES.md).

## Notable dependencies and their licenses

Permissive (MIT / Apache-2.0 / BSD), among others: `tokio`, `axum`, `serde`, `reqwest`,
`tokenizers`, `hf-hub`, `nvml-wrapper`, `opencl3`, `llguidance`, `toktrie_hf_tokenizers`,
`image`, `rayon`, `half`, `safetensors`, `bytemuck`.

Weak copyleft (file-level), used unmodified as a dependency:

- **`symphonia`** (audio decoding: MP3/FLAC/OGG/AAC) - **MPL-2.0**. Compatible with this
  project's license; it requires that modifications to the symphonia source files themselves
  remain under MPL-2.0. `loken` does not modify symphonia, so no additional obligation
  applies beyond this notice.

For the complete, version-pinned dependency license set: `cargo license`.
