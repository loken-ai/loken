# CUDA sources

2 compilation models exist. The directory of a `.cu` file says which one applies.

| Directory | Compiled | By | Reached through |
|---|---|---|---|
| `cuda/` | at build time | `nvcc`, driven by `build.rs` | `extern "C"` launchers called from Rust |
| `src/inference/cuda/` | at run time | NVRTC | the source text, embedded with `include_str!` and handed to the driver on first use |

## `cuda/`

`build.rs` turns each file into an object and links it.
Each kernel carries SASS for every architecture in `portable_gencodes()`.
A card gets machine code, and the driver compiles nothing on load.

## `src/inference/cuda/`

These files are Rust source data.
Each path resolves relative to the `.rs` file that embeds it, so they live under `src/`.
They cost nothing to build.
They suit a kernel whose shape depends on values known only once a model is loaded.

## The file in both

`cuda/moe/gguf.cuh` is compiled by `nvcc` for the MoE kernels.
`src/inference/quantized_cuda/mod.rs` also prepends it to the mat-vec NVRTC unit.
It states the GGUF block layouts and their dot products, so neither model restates them.

NVRTC compiles from a string with no system headers.
The loader prepends `src/inference/cuda/nvrtc_compat.h`, which defines the same types.
The header therefore guards its include:

```c
#ifndef __CUDACC_RTC__
#include <stdint.h>
#endif
```

## Add a file

`src/notice_gate.rs` fails the test suite when a CUDA file has no row in `scripts/provenance/manifest.tsv`.
Run the measurement and commit the regenerated manifest, see `scripts/provenance/README.md`.
A file measured at or above the gate's `MUST_LIST` share also needs a row in `NOTICE.md`.
