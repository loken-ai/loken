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

## Without CUDA

```sh
cargo build --release --no-default-features --features cpu        # language models, host only
cargo build --release --no-default-features --features cpu,media  # and every media family
```

`image`, `audio`, `video` and `midi` can each be added on their own; `metrics` adds the scrape
endpoint. The feature table is in [`CONFIGURATION.md`](CONFIGURATION.md#build-features).

## The binary and the machine it runs on

`.cargo/config.toml` builds for the host CPU so the quantized kernels use whatever SIMD it has.
That binary is not portable: drop `target-cpu=native` when building for another machine. The
host path needs AVX2 either way: a build for a CPU without it compiles, and refuses the
quantized matmuls at run time.

A CUDA build serves without a card too: `lokend serve --cpu` places every layer on the host.

## Running the daemon

```
lokend serve [--port 11435] [--models-dir DIR] [--keep-alive DURATION] [--cpu] [--verbose]
```

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
| `alloc failed: out of memory` | A card ran out while the placement had already spread the model. Lower `context_length` or `max_gpu_memory_fraction`, or pick a smaller quantisation. |
| `matmul_q4k_plain: avx2 build required` | This binary was built for a CPU without AVX2 and cannot run quantized weights on the host. Build on the target machine, or serve on a card. |
| `401` on every request but `/health` | `require_auth` is on. Send `Authorization: Bearer <key>` or `X-API-Key` with a key from `api_keys`. |
| `Address already in use` | Another daemon holds the port. `lokend serve --port <other>`, or stop it. |
| The first answer after a load is slow | The load and, for a mixture, the background repack are paid once. The second request runs at the model's rate. |
