# Install

loken builds from source into 2 binaries, `lokend` (the daemon) and `loken` (the client).

## Before you start

<!-- site: widget checklist id=install-prereq -->
- [ ] A Rust toolchain with `cargo`, and the repository checked out.
- [ ] For the default build: a CUDA toolkit and OpenCL headers.
- [ ] For the CPU build: nothing beyond the toolchain.
- [ ] A CPU with AVX2, FMA, BMI1/2, F16C, LZCNT and MOVBE.
  Older CPUs are refused at start with a message naming what is missing.
<!-- site: end -->

## Build

<!-- site: widget tabs group=install id=install-method -->
### With CUDA

```sh
cargo build --release
```

The default features are `cuda`, `opencl` and `media`.  
The build looks for the toolkit through `CUDA_PATH` or `CUDA_HOME`.
Without a toolkit it fails with `nvcc not found at ...; set CUDA_PATH`.

### Without CUDA

```sh
cargo build --release --no-default-features --features cpu
```

Language models run on the host.  
Add `media` for every media family: `--features cpu,media`.

### AMD, NVIDIA and Vulkan

```sh
cargo build --release --no-default-features --features rocm,cpu
cargo build --release --features cuda,rocm
cargo build --release --features vulkan
```

ROCm needs `hipcc` on the `PATH` or under `ROCM_PATH`.  
The feature table and the per-vendor requirements are in [Building](https://github.com/loken-ai/loken/blob/main/docs/BUILDING.md).
<!-- site: end -->

The binaries are `target/release/lokend` and `target/release/loken`.

loken publishes no prebuilt archive and no package.

## Add a configuration

```sh
cp config.toml.example config.toml
```

The daemon reads `config.toml` from the first of these places:

1. The working directory.
2. The directory of the executable.
3. The user configuration directory, `~/.config/loken/config.toml` on Linux.

Without a file the daemon runs on its defaults.  
Every key is in [Configuration](../CONFIGURATION.md).

## Verify

<!-- site: widget checklist id=install-verify -->
- [ ] `target/release/loken --version` prints a version.
- [ ] `target/release/lokend serve` logs `Server listening on http://127.0.0.1:11435`.
- [ ] From another terminal, `target/release/loken list` answers instead of failing to connect.
<!-- site: end -->

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `nvcc not found at ...; set CUDA_PATH` | The build cannot find the CUDA toolkit. | Set `CUDA_PATH` to it, or build without CUDA. |
| `config.toml not found in CWD, exe dir, or user config dir` | No configuration file in the 3 places. | Copy `config.toml.example` next to the daemon. |
| `Address already in use` | Another process holds the port. | `lokend serve --port <other>`, or stop that process. |

## Next

- [First run](first-run.md)
- [Building](https://github.com/loken-ai/loken/blob/main/docs/BUILDING.md)
- [Models](../MODELS.md)
