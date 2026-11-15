# Building

Build `lokend` and `loken` from source.
A release archive needs no build, see [Install](https://loken-ai.github.io/loken/install/).  
First start: [First run](https://loken-ai.github.io/loken/first-run/).

```sh
cargo build --release   # target/release/{lokend,loken}
```

## Build profiles

| Target | Command | Needs |
|---|---|---|
| NVIDIA, default | `cargo build --release` | CUDA toolkit |
| AMD (ROCm) | `cargo build --release --no-default-features --features rocm,cpu` | ROCm `hipcc` |
| NVIDIA and AMD in 1 binary | `cargo build --release --features cuda,rocm` | Both toolkits |
| Vulkan beside CUDA and OpenCL | `cargo build --release --features vulkan` | CUDA toolkit |
| Vulkan alone | `cargo build --release --no-default-features --features vulkan,cpu` | Nothing |
| Host only, language models | `cargo build --release --no-default-features --features cpu` | Nothing |
| Host only, with every media family | `cargo build --release --no-default-features --features cpu,media` | Nothing |

`image`, `audio`, `video` and `midi` can each be added on their own.  
`metrics` adds the scrape endpoint.  
The feature table is in [Configuration](CONFIGURATION.md#build-features).

## The default build

The default features are `cuda`, `opencl` and `media`.

The build needs a **CUDA toolkit**.  
It is built and tested against **CUDA 13.3**.
No older toolkit has been tried.

OpenCL needs nothing at build time.
The server opens the OpenCL loader when it starts.

The kernels are compiled for a fixed list of architectures from Turing to Blackwell, not probed against the toolkit.
A toolkit that does not know `sm_120a` fails the build.

| Variable | Effect |
|---|---|
| `CUDA_PATH`, `CUDA_HOME` | Locates the toolkit when it is not where the build looks |
| `LOKEN_CUDA_ARCHS` | Narrows the list of NVIDIA architectures the kernels are built for |
| `LOKEN_STATIC_CXX` | Links the C++ runtime statically |

## Release archive

`scripts/build-portable.sh` builds the Linux x86-64 archives in a container and packages them in `dist/`.
`scripts/build-portable.sh --host` builds with the toolchain of this machine instead.

It produces 2 archives:

| Archive | Holds |
|---|---|
| Light | The binaries. Starts on any machine |
| CUDA libraries | `lib/` with NVRTC, cuBLAS and cuBLASLt, and NCCL when the build machine has it |

Unpacked over the light archive, the second one runs on a machine that has only the NVIDIA driver.

| Variable | Effect |
|---|---|
| `FEATURES` | The cargo features. Default: the crate's defaults, plus `rocm` when `hipcc` is found |
| `CPU_BASELINE` | The `-C target-cpu` of the CPU kernels. Default `x86-64-v3` |
| `MAX_GLIBC` | The highest glibc symbol version the binaries may need. Default 2.34 |
| `DIST` | The output directory. Default `dist` |

The script header lists the other variables.

## AMD cards (ROCm)

```sh
cargo build --release --no-default-features --features rocm,cpu
```

This build serves language models only.
`--features rocm,cpu,media` adds the speech, image and video routes.

The build needs ROCm's `hipcc` on the `PATH` or under `ROCM_PATH`.  
The kernels are compiled for the AMD targets from CDNA (gfx908) to RDNA4 (gfx1201).
`LOKEN_ROCM_ARCHS` narrows the list.

The server opens the HIP runtime, hipBLAS, hipRTC and RCCL when it starts.
The binary therefore starts on a machine without them and lists no AMD card.

These operations run on AMD's matrix instructions, on CDNA and from RDNA3 on:

- The W4A16 GEMM.
- The integer matrix products.
- The MoE prefill over quantised experts.
- The DiT and grouped-query prefill attention.

An RDNA2 card takes their callers' other paths.

## NVIDIA and AMD cards in one binary

```sh
cargo build --release --features cuda,rocm
```

The build needs both toolkits.
It lists the NVIDIA cards through CUDA, then the AMD cards through HIP, as 1 set of card numbers.

The placement plan, the ladder and the probes see every card, so a model can be split across a GeForce and a Radeon.

| Behaviour | Detail |
|---|---|
| Transfers between vendors | Through host memory, as between cards without peer access |
| Runtime libraries | Opened at server start, the HIP runtime included. A machine with 1 vendor's stack lists only that vendor's cards |
| Tensor-parallel group | Sums through NCCL among NVIDIA cards and RCCL among AMD cards. A group of both vendors moves its activations through host memory, and is chosen only over a plan that leaves layers on the host |
| Replayed decode step | Stays on cards of 1 vendor |
| Layer on an AMD card | Takes the kernels the ROCm build runs |

## OpenCL GPUs

`opencl` is a default feature, so `cargo build --release` already carries it.
The build needs no OpenCL package.
The server opens the OpenCL loader when it starts.
A machine without a loader lists no OpenCL device.
A driver that does not answer when asked what it has counts as no device.

A card that CUDA or ROCm drives is told by its PCI address and is not offered for OpenCL layers.
A card is theirs only while `--backends` keeps their runtime.
Devices of Mesa's OpenCL platform are not offered for layers.
The layers of a language model and the blocks of Z-Image run through Intel's OpenCL runtime on an Intel GPU, see [Status](STATUS.md#opencl).
A GPU that shares the host's memory offers what the host can spare at load.
`/api/distributed/devices` reports that figure as the device's free memory.

`lokend serve --backends opencl,cpu` serves on OpenCL and the host only.

## Vulkan cards

```sh
cargo build --release --no-default-features --features vulkan,cpu
```

The server opens the Vulkan loader when it starts, and compiles the kernels for the device when it opens.

Placement offers the Vulkan GPUs that no other runtime of the build drives.  
They come after the CUDA or HIP cards and before the host.  
The build above has no other runtime, so every GPU is offered.
`--features vulkan` on the default build adds Vulkan beside CUDA and OpenCL, and needs the CUDA toolkit.

The layers of a decoder in `Q4_K`, `Q6_K`, `Q8_0`, `Q5_0`, `MXFP4`, `BF16`, `F16` or `F32` run on Vulkan, with the output head when the last layer does.
The routed experts of a mixture run there when the router is a softmax over experts held as 1 stack each of gate, up and down.
A recurrent layer of a hybrid model runs on the host between them.
Their weights count in the `size_vram` that `/api/ps` reports.
A model of another shape stays off Vulkan and loads on the host.
The context reads the storage binding limit and the allocation limit of the device.
A buffer above either limit is refused with an error that names its size and both limits.

## Without CUDA

```sh
cargo build --release --no-default-features --features cpu        # language models, host only
cargo build --release --no-default-features --features cpu,media  # and every media family
```

A CUDA build also serves without a card: `lokend serve --backends cpu` places every layer and every tower on the host.

## CPU level

`.cargo/config.toml` builds for x86-64-v3 (AVX2, FMA, BMI1/2, F16C, LZCNT, MOVBE), not for the host CPU.
A CPU without them is refused at start, with a message that names what is missing.

The daemon picks the quantized kernels at start from what the CPU has, and logs the level.

To let the compiler use the build host's whole instruction set outside the kernels, build with `RUSTFLAGS="-C target-cpu=native"`.
That binary does not run on other hosts.

### AVX-512 level

At the `avx512` level (AVX-512 F/BW/VL/DQ/VNNI), these use 512-bit registers:

- The prefill GEMMs of `Q4_K`, `Q5_K`, `Q6_K`, `Q4_0`, `Q5_0`, `Q8_0` and `MXFP4`.
- The dots of the k-quants, of the 32-weight formats and of f32, f16 and bf16.
- The bf16 and fp8 row dots, and the q8_K quantiser.
- The attention dot and axpy, the f16 and q8 KV helpers, the activations, the softmax exp, RMSNorm and RoPE.
- The expert products of `IQ2_S`, `IQ3_XXS` and `IQ4_NL`.

Each kernel returns the bits of its AVX2 twin.
The bf16 and fp8 row dots are the exception: they sum in another order, and agree to the rounding of the sum.

The other IQ formats, the fp4 chunk dot, the argmax and the decode GEMVs of the repacked layouts stay on their AVX2 form.

## Daemon options

```
lokend serve [--port PORT] [--models-dir DIR] [--keep-alive DURATION] [--backends LIST] [--verbose]
```

| Option | Effect |
|---|---|
| `--port` | Overrides `[server] port` of `config.toml`. The port is 11435 when neither names one |
| `--models-dir` | Names the Ollama store |
| `--keep-alive` | Sets how long a model stays resident after its last request. `-1` keeps it for ever. `OLLAMA_KEEP_ALIVE` is read too |
| `--backends` | Names the backends to compute on, among `cuda`, `rocm`, `vulkan`, `opencl` and `cpu`. A backend left out is not probed and receives no layer. `cpu` alone places every layer and every tower on the host. Without the option every backend the machine offers is used |
| `--verbose` | Logs more |

`config.toml` holds everything else.  
The daemon finds it in the working directory, beside the executable, or in the user configuration directory.

## Troubleshooting

| What you see | What it means |
|---|---|
| `nvcc not found at ...; set CUDA_PATH` | The build cannot find the toolkit. Point `CUDA_PATH` at it, or build without CUDA |
| `config.toml not found in CWD, exe dir, or user config dir` | The daemon runs with defaults. Copy `config.toml.example` next to it |
| `Model not found` | Neither store holds the name. Run `loken pull <name> --source ollama` or `--source huggingface`. A Hugging Face id needs its owner |
| `Model '...' not found in remote registry` | The registry chosen by `--source` does not know the name. Check the tag on the registry, or try the other source |
| `alloc failed: out of memory` | A card ran out while the placement had already spread the model. Lower the request's `num_ctx`, or pick a smaller quantisation |
| `401` on every request but the health paths | `require_auth` is on. Send `Authorization: Bearer <key>` or `X-API-Key` with a key from `api_keys`. `/health`, `/healthz`, `/api/health` and `/ping` need no key |
| `Address already in use` | Another daemon holds the port. Run `lokend serve --port <other>`, or stop it |
| The first answer after a load is slow | The load is paid once, and so is the background repack of a mixture. The second request runs at the model's rate |
