# Contributing

Issues and pull requests welcome. A few things first.

## What a change should bring

- **A measurement, not an argument.** Performance claims need a number and its conditions  - 
  which card, which model, what else was running. `docs/BENCHMARKS.md` has the protocol,
  `scripts/` has the harness.
- **A test that can fail.** Kernel and arithmetic changes want a reference computed a *different*
  way - a host reference, not a second GPU path - and seen to go red. Defects have shipped here
  because a kernel had no such test; one produced wrong answers for two days.
- **A derived tolerance.** Say what rounding steps the two paths differ by and size the bound
  from that. A tolerance picked to make a test pass will pass when the code is wrong.

## Build

CUDA toolkit and OpenCL headers, or `cargo build --release --no-default-features --features cpu`
for neither. Tested against CUDA 13.3. `cp config.toml.example config.toml` before running the
daemon.

No CI: the suite needs a CUDA GPU, so it runs where you are. `cargo test --lib --release` before
opening a PR. Tests marked `#[ignore]` need model weights on disk  - 
`config.test.toml.example` says how to point them at yours.

## Style

Comments say what the code does and what invariant it holds - not how it was found, not what it
replaced. Commit messages: two paragraphs at most.

Porting an architecture: take it from the architecture and the checkpoint's tensor names. Copied
code is measured and listed in `NOTICE.md`, and that table should be empty.

## Reporting

Model, prompt, hardware, and what you expected. If the output is wrong rather than slow, say
whether it is wrong the same way each time - consistent and random need different
investigations.
