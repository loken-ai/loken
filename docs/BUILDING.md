# Building

```sh
cp config.toml.example config.toml   # the daemon reads this from its working directory
cargo build --release                # target/release/{lokend,loken}
```

## The default build

The default features are `cuda`, `opencl` and `media`, so the build wants a **CUDA toolkit**
and **OpenCL headers**. It is built and tested against **CUDA 13.3**; no older toolkit has been
tried. The kernels are compiled for a fixed list of architectures from Turing to Blackwell, and
that list is not probed against your toolkit: one that does not know `sm_120a` fails the build
rather than skipping it. `CUDA_PATH` or `CUDA_HOME` says where the toolkit is if it is not where
the build looks; `CUDNN_LIB` does the same for cuDNN.

## AMD cards (ROCm)

```sh
cargo build --release --no-default-features --features rocm,cpu
```

The build wants ROCm's `hipcc` on the `PATH` or under `ROCM_PATH`. The kernels are compiled for
the AMD targets from CDNA (gfx908) to RDNA4 (gfx1201); `LOKEN_ROCM_ARCHS` narrows the list. The
HIP runtime, hipBLAS, hipRTC and RCCL are opened when the server starts, so the binary starts on
a machine without them and lists no AMD card. Not yet on AMD: the replayed decode step and the
kernels written in NVIDIA's own matrix instructions, whose callers take their other paths.

## NVIDIA and AMD cards in one binary

```sh
cargo build --release --features cuda,rocm
```

The build wants both toolkits. It lists the NVIDIA cards through CUDA, then the AMD cards through
HIP, as one set of card numbers: the placement plan, the ladder and the probes see every card,
and a model can be split across a GeForce and a Radeon. Tensors move between cards of different
vendors through host memory, as between cards without peer access. Each runtime's libraries are
opened when the server starts, the HIP runtime included, so the binary starts on a machine with
one vendor's stack and lists only that vendor's cards. A tensor-parallel group and a replayed
decode step stay on cards of one vendor; a layer on an AMD card takes the kernels the ROCm build
runs.

## Without CUDA

```sh
cargo build --release --no-default-features --features cpu        # language models, host only
cargo build --release --no-default-features --features cpu,media  # and every media family
```

`image`, `audio`, `video` and `midi` can each be added on their own; `metrics` adds the scrape
endpoint. The feature table is in [`CONFIGURATION.md`](CONFIGURATION.md#build-features).

## The binary and the machine it runs on

`.cargo/config.toml` builds for x86-64-v3 (AVX2, FMA, BMI1/2, F16C, LZCNT, MOVBE), not for the
host CPU. A CPU without them is refused at start with a message naming what is missing. The
quantized kernels are chosen at start from what the CPU has, and the daemon logs the level.
`[cpu] level` in `config.toml` caps it, and `generic` selects the scalar reference code.

At the `avx512` level (AVX-512 F/BW/VL/DQ/VNNI) the prefill GEMMs of Q4_K, Q5_K, Q6_K, Q4_0, Q5_0,
Q8_0 and MXFP4, the dots of the k-quants, of the 32-weight formats and of f32, f16 and bf16, the
bf16 and fp8 row dots and the q8_K quantiser use 512-bit registers. Each returns the bits of its
AVX2 twin, except the bf16 and fp8 row dots, which sum in another order and agree to the rounding
of the sum.

The attention dot and axpy, the f16 and q8 KV helpers, the activations, the softmax exp, RMSNorm
and RoPE use 512-bit registers too, with the AVX2 results. So do the expert products of IQ2_S,
IQ3_XXS and IQ4_NL, to the AVX2 bits. The other IQ formats, the fp4 chunk dot, the argmax and the
decode GEMVs of the repacked layouts stay on their AVX2 form.

To let the compiler use the build host's whole instruction set outside the kernels, build with
`RUSTFLAGS="-C target-cpu=native"`; that binary does not run on other hosts.

A CUDA build serves without a card too: `lokend serve --cpu` places every layer and every tower on the host.

## Running the daemon

```
lokend serve [--port PORT] [--models-dir DIR] [--keep-alive DURATION] [--cpu] [--verbose]
```

`--port` overrides the `[server] port` of `config.toml` (11435 when neither names one);
`--models-dir` names the Ollama store; `--keep-alive` is how long a model stays resident after
its last request, `-1` for ever, also read from `OLLAMA_KEEP_ALIVE`. Everything else comes from
`config.toml`, found in the working directory, beside the executable, or in the user
configuration directory.

## Troubleshooting

| What you see | What it means |
|---|---|
| `nvcc not found at ...; set CUDA_PATH` | The build cannot find the toolkit. Point `CUDA_PATH` at it, or build without CUDA. |
| `config.toml not found in CWD, exe dir, or user config dir` | The daemon runs with defaults. Copy `config.toml.example` next to it. |
| `Model not found` | Neither store holds the name. `loken pull <name> --source ollama` or `--source huggingface`; a Hugging Face id needs its owner. |
| `Model '...' not found in remote registry` | The registry chosen by `--source` does not know the name. Check the tag on the registry, or the other source. |
| `alloc failed: out of memory` | A card ran out while the placement had already spread the model. Lower the request's `num_ctx`, or pick a smaller quantisation. |
| `401` on every request but `/health` | `require_auth` is on. Send `Authorization: Bearer <key>` or `X-API-Key` with a key from `api_keys`. |
| `Address already in use` | Another daemon holds the port. `lokend serve --port <other>`, or stop it. |
| The first answer after a load is slow | The load and, for a mixture, the background repack are paid once. The second request runs at the model's rate. |
