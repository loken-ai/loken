# The fundamentals of inference, in ten lessons

Ten lessons for a developer who has never opened an inference engine. Each takes about ten
minutes, states one fundamental, shows where it lives in this tree, reports what was
measured about it here, and ends with one request or command to run against a local daemon.

Prerequisites: a running `lokend` with a small model (the exercises use `qwen3:0.6b`; see
[`../BUILDING.md`](../BUILDING.md)), `curl` and `python3`. Lessons 3 and 4 run a `cargo test`.

| Lesson | Question |
|---|---|
| [`01-tokens.md`](01-tokens.md) | How does text become ids, and ids become the JSON a client reads? |
| [`02-model-file.md`](02-model-file.md) | What is in a checkpoint, and why does the catalogue read a few kilobytes of it? |
| [`03-quantisation.md`](03-quantisation.md) | How are bits traded for bytes, and how is the trade judged? |
| [`04-forward-pass.md`](04-forward-pass.md) | What does one layer read, compute and keep? |
| [`05-prefill-and-decode.md`](05-prefill-and-decode.md) | Why is the prompt bound by arithmetic and the answer by bytes? |
| [`06-kv-cache.md`](06-kv-cache.md) | What is kept between tokens, and between requests? |
| [`07-sampling.md`](07-sampling.md) | How does a row of logits become one token, and when is that deterministic? |
| [`08-speculative-decoding.md`](08-speculative-decoding.md) | How can a decode step commit more than one token? |
| [`09-placement.md`](09-placement.md) | Where does each part of a model live, and what does a wrong choice cost? |
| [`10-measuring.md`](10-measuring.md) | How is a rate taken so that it means something? |

[`slides.html`](slides.html) is the same series as a deck, one file, to open in a browser.

The thread through the ten: every defect reported in these pages was invisible to the
measurement in place when it happened. A coherence gate that read 120 characters, a prompt
counted in words, a token rate, a sweep under eviction, a verification against a cache that
already carried the defect.

Every figure carries a date and the machine it came from; the machine is the one
[`../STATUS.md`](../STATUS.md) describes unless a lesson says otherwise. The figures in
[`../BENCHMARKS.md`](../BENCHMARKS.md) are the protocol's; the ones here are the
experiments'. The diagrams are hand-written SVG in `../img/learn-*.svg`.
