# Course: how inference works

10 lessons for a developer who has never opened an inference engine.
Each lesson takes about 10 minutes.

Each lesson states 1 fundamental, shows what loken does about it, reports what was measured, and ends with a request or command to run against a local daemon.

## Prerequisites

<!-- site: widget checklist id=learn-prereq -->
- [ ] A running `lokend` serving a small model, `qwen3:0.6b` in the exercises (see [BUILDING](../BUILDING.md))
- [ ] `curl`
- [ ] `python3`
- [ ] [`lokprof`](https://github.com/loken-ai/lokprof), for lesson 10 only
<!-- site: end -->

## Lessons

| Lesson | Question |
|---|---|
| [Tokens](01-tokens.md) | How does text become ids, and ids become the JSON a client reads? |
| [What a model file holds](02-model-file.md) | What is in a checkpoint, and why does the catalogue read a few kilobytes of it? |
| [Fewer bits per weight](03-quantisation.md) | How are bits traded for bytes, and how is the trade judged? |
| [One layer, then 40 of them](04-forward-pass.md) | What does one layer read, compute and keep? |
| [Prefill and decode](05-prefill-and-decode.md) | Why is the prompt bound by arithmetic and the answer by bytes? |
| [The KV cache](06-kv-cache.md) | What is kept between tokens, and between requests? |
| [Sampling](07-sampling.md) | How does a row of logits become 1 token, and when is that deterministic? |
| [Speculative decoding](08-speculative-decoding.md) | How can a decode step commit more than 1 token? |
| [Placement](09-placement.md) | Where does each part of a model live, and what does a wrong choice cost? |
| [Measuring](10-measuring.md) | How is a rate taken so that it means something? |

## Progress

<!-- site: widget progress id=learn-progress -->
The progress of the reader appears here.
<!-- site: end -->

## Common thread

Every defect reported in the lessons was invisible to the measurement in place when it happened:

- a coherence gate that read 120 characters
- a prompt counted in words
- a token rate
- a sweep under eviction
- a verification against a cache that already carried the defect

## About the figures

- Every figure carries a date and the machine it came from.
- The machine is the one [STATUS](../STATUS.md) describes, unless a lesson says otherwise.
- The figures in the [benchmarks](https://github.com/loken-ai/benchmarks) repository are the protocol's.  
  The ones in the lessons are the experiments'.
- The diagrams are hand-written SVG.
