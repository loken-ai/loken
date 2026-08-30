<p align="center">
  <img src="https://raw.githubusercontent.com/loken-ai/.github/main/brand/png/lockup.png" alt="LOKEN - Local . Multimodal . Green" width="420">
</p>

# loken

A local inference server for language **and** generative-media models, written in Rust. One
process serves chat, images, audio, speech, transcription and video, over an
OpenAI-compatible API and an Ollama-compatible one.

```sh
cp config.toml.example config.toml   # the daemon reads this from its working directory
cargo build --release                # target/release/{lokend,loken}
./target/release/lokend serve        # the daemon, on port 11435
./target/release/loken  list         # and the client that talks to it
```

It speaks the OpenAI API, so anything that already talks to one works unchanged:

```sh
curl -s localhost:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3:1.7b",
  "messages": [{"role": "user", "content": "Say hello in six words."}]
}'
```

**Inference never leaves the machine.** No account, no API key, no hosted fallback. What does
reach the network: fetching a model you asked for, and a cluster node announcing itself if you
configure one.

## What it serves

- **Language models** - GGUF and safetensors, quantized or not, with continuous batching,
  paged KV, speculative decoding and prompt-prefix reuse.
- **Images** - several families, including editing.
- **Audio and speech** - music, sound effects, text to speech, transcription and source
  separation.
- **Video** - text and image conditioned clips.

Each family is a Cargo feature - `image`, `audio`, `video`, `midi`, or `media` for all. A
family that is not compiled reports its models as unknown rather than dispatching to something
absent. Text-only is roughly half the source.

## Where it is careful

- **Placement is measured.** The forward is walked without allocating, each card is asked what
  it holds, and one rule decides for every family. Weights are charged at what the pool keeps,
  not what the files weigh.
- **Out of memory is a bug, not a condition.** What does not fit spreads across cards; only what
  no card holds reaches the host.
- **Energy is reported beside speed**, per request.

## Building

The default build wants a **CUDA toolkit** and **OpenCL headers**, because CUDA and OpenCL are
both on by default along with every media family. It is built and tested against **CUDA 13.3**;
no older toolkit has been tried. The kernels are compiled for a fixed list of architectures from
Turing to Blackwell, and that list is not probed against your toolkit - one that does not know
`sm_120a` (added in 12.8) will fail the build rather than skip it. `CUDA_PATH` or `CUDA_HOME`
says where the toolkit is if it is not where the build looks; `CUDNN_LIB` does the same for
cuDNN. Without any of it:

```sh
cargo build --release --no-default-features --features cpu        # language models, CPU only
cargo build --release --no-default-features --features cpu,media  # and every media family
```

`image`, `audio`, `video` and `midi` can each be added on their own. `.cargo/config.toml`
builds for the host CPU so the quantized kernels can use whatever SIMD it has - which makes
the binary non-portable, so drop that flag if you are building for another machine.

## Running it on a network

The server **has no authentication and no rate limit unless you configure them**, and its API
can delete models. That suits one person on one machine. Turn both on in `config.toml` before it
listens on anything but localhost.

## Documentation

- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) - measurements against other engines and the
  protocol that makes the comparison fair, including the cells where this engine loses. Every
  figure there is out of date: it was taken during a phase of very active development and a
  fresh campaign is needed before any of it is quoted
- [`docs/CLUSTER.md`](docs/CLUSTER.md) - spreading work across more than one machine
- [`docs/STATUS.md`](docs/STATUS.md) - **what is measured, what is known to be slower, what is
  written but not wired, and what has never run.** Read this one before deciding whether the
  engine suits you

## Contributing and reporting

[`CONTRIBUTING.md`](CONTRIBUTING.md) - what a change should bring with it.
[`SECURITY.md`](SECURITY.md) - reporting a vulnerability, and what the unauthenticated default
assumes.

## Licence

MIT OR Apache-2.0, at your option. Third-party attributions are in [`NOTICE.md`](NOTICE.md).
Model weights are not distributed here and carry their own terms.
