# Building

Build `lokend` and `loken` from source.  
A release archive needs no build, see [Install](start/install.md).  
First start: [First run](start/first-run.md).

```sh
cargo build --release   # target/release/{lokend,loken}
```

## Build profiles

| Target | Command | Needs |
|---|---|---|
| NVIDIA, default | `cargo build --release` | CUDA toolkit, OpenCL headers |
| AMD (ROCm) | `cargo build --release --no-default-features --features rocm,cpu` | ROCm `hipcc` |
| NVIDIA and AMD in 1 binary | `cargo build --release --features cuda,rocm` | Both toolkits |
| Vulkan | `cargo build --release --features vulkan` | The crates only |
| Host only, language models | `cargo build --release --no-default-features --features cpu` | Nothing |
| Host only, with every media family | `cargo build --release --no-default-features --features cpu,media` | Nothing |

`image`, `audio`, `video` and `midi` can each be added on their own.
`metrics` adds the scrape endpoint.  
The feature table is in [Configuration](CONFIGURATION.md#build-features).

## The default build

The default features are `cuda`, `opencl` and `media`.

The build needs a **CUDA toolkit** and **OpenCL headers**.
It is built and tested against **CUDA 13.3**.
No older toolkit has been tried.

The kernels are compiled for a fixed list of architectures from Turing to Blackwell, not probed against the toolkit.
A toolkit that does not know `sm_120a` fails the build.

| Variable | Effect |
|---|---|
| `CUDA_PATH`, `CUDA_HOME` | Locates the toolkit when it is not where the build looks |
| `CUDNN_LIB` | Locates cuDNN |

## AMD cards (ROCm)

```sh
cargo build --release --no-default-features --features rocm,cpu
```

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
| Tensor-parallel group, replayed decode step | Stay on cards of 1 vendor |
| Layer on an AMD card | Takes the kernels the ROCm build runs |

## Vulkan cards

```sh
cargo build --release --features vulkan
```

The server opens the Vulkan loader when it starts, and compiles the kernels for the device when it opens.

Placement offers the Vulkan GPUs that no other runtime of the build drives.  
They come after the CUDA or HIP cards and before the host.
A build with `vulkan` alone has no other runtime, so every GPU is offered.

The layers of a dense decoder in `Q4_K`, `Q6_K`, `Q8_0` or `F16` run on Vulkan, with the output head when the last layer does.
A model of another shape stays off Vulkan.

## Without CUDA

```sh
cargo build --release --no-default-features --features cpu        # language models, host only
cargo build --release --no-default-features --features cpu,media  # and every media family
```

A CUDA build also serves without a card: `lokend serve --cpu` places every layer and every tower on the host.

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
lokend serve [--port PORT] [--models-dir DIR] [--keep-alive DURATION] [--cpu] [--verbose]
```

| Option | Effect |
|---|---|
| `--port` | Overrides `[server] port` of `config.toml`. The port is 11435 when neither names one |
| `--models-dir` | Names the Ollama store |
| `--keep-alive` | Sets how long a model stays resident after its last request. `-1` keeps it for ever. `OLLAMA_KEEP_ALIVE` is read too |
| `--cpu` | Places every layer and every tower on the host |
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
| `401` on every request but `/health` | `require_auth` is on. Send `Authorization: Bearer <key>` or `X-API-Key` with a key from `api_keys` |
| `Address already in use` | Another daemon holds the port. Run `lokend serve --port <other>`, or stop it |
| The first answer after a load is slow | The load is paid once, and so is the background repack of a mixture. The second request runs at the model's rate |
