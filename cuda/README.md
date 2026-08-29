# Which CUDA source lives where

Two compilation models, and the directory a `.cu` sits in says which one it belongs to.

**`cuda/**` - compiled at build time by `nvcc`.** `build.rs` turns each of these into an
object and links it; the kernels are reached from Rust through `extern "C"` launchers. They
carry real SASS for every architecture in `portable_gencodes()`, so a card gets machine code
rather than something the driver has to compile on the way in.

**`src/inference/cuda/*.cu` - compiled at run time by NVRTC.** These are `include_str!`-ed
into the binary and handed to the driver as text on first use. They live under `src/` because
that is what they are: Rust source data, with paths resolved relative to the `.rs` file that
embeds them. They cost nothing to build and are the right home for a kernel whose shape
depends on values only known once a model is loaded.

## The one file in both

`cuda/moe/gguf.cuh` is compiled by `nvcc` for the MoE kernels *and* prepended to the mat-vec
NVRTC unit by `src/inference/quantized_cuda/mod.rs`. It states the GGUF block layouts and
their dot products, which both models need and neither should restate.

That dual use is why it carries

```c
#ifndef __CUDACC_RTC__
#include <stdint.h>
#endif
```

 -  NVRTC compiles from a string with no system headers, and the loader prepends its own
`nvrtc_compat.h` defining the same types, so including `<stdint.h>` there is a hard error.

## Adding a file

`notice_gate.rs` fails the test suite for any `.cu`/`.cuh` under `cuda/` with no row in
`NOTICE.md`. This is where verbatim vendored code has historically arrived, so a new file
states what it is - measured against its upstream, not remembered.
