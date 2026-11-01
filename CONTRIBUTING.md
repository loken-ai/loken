# Contributing

Issues and pull requests are welcome.

## What a change brings

| Change | It brings |
|---|---|
| A performance claim | A number and its conditions: card, model, what else was running. The [benchmarks](https://github.com/loken-ai/benchmarks) repository holds the protocol and the harness. |
| A kernel or arithmetic change | A test that can fail, against a reference computed another way: a host reference, not a second GPU path. The test is seen to go red. |
| A tolerance in a test | A bound derived from the rounding steps where the 2 paths differ. A bound picked to make a test pass also passes when the code is wrong. |

A kernel without such a test has shipped wrong answers for 2 days.

## Build

| Setup | Command |
|---|---|
| CUDA toolkit and OpenCL headers | `cargo build --release` |
| Neither | `cargo build --release --no-default-features --features cpu` |

Copy `config.toml.example` to `config.toml` before running the daemon.  
CUDA 13.3 is the tested toolkit.

## Test

Run `cargo test --lib --profile gate` before opening a pull request.

| Where | What runs |
|---|---|
| CI | The suite on the host |
| Your machine | The card tests |

Tests marked `#[ignore]` need model weights on disk or several cards.
`config.test.toml.example` shows how to point them at your weights.

## Style

- A comment says what the code does and what invariant it holds.
- A commit message has 2 paragraphs at most.
- A ported architecture comes from the architecture and the checkpoint's tensor names.
- Copied code is measured and listed in `NOTICE.md`.
  The goal is an empty table.

## Reporting a problem

Give the model, the prompt, the hardware and the expected output.  
For a wrong output, say whether it is wrong the same way each time.
