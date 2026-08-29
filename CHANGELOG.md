# Changelog

All notable changes to `loken`. The format follows [Keep a Changelog](https://keepachangelog.com/1.1.0/).

## [0.1.0] - 2026-08-29

First public release. What it does not yet do, and where it is known to be slower, is in
[`docs/STATUS.md`](docs/STATUS.md) rather than left for you to find.

### Added

- An OpenAI-compatible and Ollama-compatible HTTP server for language models: GGUF and
  safetensors, quantized or not, with continuous batching, paged KV, speculative decoding
  and prompt-prefix reuse.
- Image generation and editing across several model families.
- Audio: music and sound generation, text to speech, transcription, and source separation.
- Video generation, text and image conditioned.
- One placement rule for every family: the forward is walked without allocating, each card
  is asked what it holds, and the plan follows from a list of weights and a list of free
  VRAM. Weights are charged at what the memory pool keeps rather than what the files weigh.
- Per-request energy accounting beside the timings.
- CUDA and OpenCL backends.
- Cargo features per media family - `image`, `audio`, `video`, `midi` - carried through the
  modules, the engines, the routes and the model catalogue. A family that is not compiled
  answers "unknown model" rather than dispatching to something that is not there, and a
  text-only build is roughly half the source.
- Multi-node operation: a node announces itself to its peers and work is routed across them.
- A measurement harness that runs the same benchmark against other engines under one protocol,
  reporting the cells this engine loses as well as the ones it wins.
- A provenance measure and a gate that holds the attribution file to it, so a borrowed line is
  recorded by measurement rather than from memory.
