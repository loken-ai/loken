# Benchmarking loken

How to run a reproducible bench campaign, and where every input comes from. This page is the
procedure; there is no other. `BENCHMARKS.md` holds the results it produces.

## The shape of it

A campaign compares loken against ollama and vLLM on the models loken serves, across a matrix of
context, prompt, streaming and device, and writes the numbers into `docs/BENCHMARKS.md`. Two layers:

- **assay** (`../assay`) is the measurement tool. It benches one model against one engine and emits
  a result JSON. It measures every modality natively - text, vision, image generation, TTS, ASR -
  selected with `--modality`; it never guesses the modality from a model name.
- **`scripts/bench`** is the driver. It reads the inputs from their single sources, provisions the
  peer engines, drives assay through the isolation protocol, and regenerates the report.

Nothing about a campaign is retyped or maintained by hand:

| input | where it comes from |
|---|---|
| store paths | `config.toml` (`ollama_models_dir`, `huggingface_models_dir`) - one source, no split-brain |
| model list | loken's own catalogue, `GET /api/tags` - no scan, no list to maintain |
| matrix axes + vLLM peers | `bench/matrix.toml` - the only curated data, public HF repo ids only |
| ollama / vLLM binaries | installed by the campaign into a bench-only compartment (see below) |

## Run it

```
scripts/bench            # plan: print the cells a campaign would run, no side effects
scripts/bench run        # run the campaign
```

Plan first. It prints the store it will use, the models loken is serving, the cell count, and each
model's vLLM peer - so what a campaign will do is visible before it does it. `run` is the deliberate
step: it monopolizes the machine for hours.

`bench` needs loken already serving (it asks loken for the catalogue) and `python3`/`curl` present.

## The bench engine compartment

The campaign measures against the *current* ollama and vLLM, installed into a compartment that is
the bench's alone - never your `~/vllm` venv or whatever `ollama` is on `PATH`. They live under the
bench working tree (`~/bench-logs/engines/`), a fresh vLLM venv and a versioned ollama, and their
versions are recorded in the run record. Isolated so the comparison is reproducible and touches no
other environment.

## Modalities

Each model is measured by the clock its modality warrants; the driver picks the modality from what
loken declares and passes `--modality` to assay:

| modality | endpoint | headline metric | gate |
|---|---|---|---|
| text | `/api/generate`, `/v1/completions` | decode tok/s | answer not degenerate |
| vision | `/api/generate` + image | decode tok/s | as text |
| image | `/v1/images/generations` | megapixels/s | fixed-seed determinism |
| tts | `/v1/audio/speech` | real-time factor | produced audio, not silence |
| asr | `/v1/audio/transcriptions` | word error rate | produced a transcript |

A model loken lists but cannot measure (say an AWQ checkpoint whose tokenizer the loader does not
read) is an **error cell**, visible in the report - never a silent skip.

## What the driver guards (and why)

- **Binary currency** - refuses to measure if any `src/**/*.rs` is newer than
  `target/release/lokend`; the same mtime stamps each result, so a rebuild re-measures every cell.
- **Engine versions** - installs and records ollama and vLLM; a stale peer is not measured silently.
- **Store consistency** - ollama and loken read the same store because both derive from
  `config.toml`; the old default (`~/.ollama`, nearly empty) is gone.
- **Peer presence** - a vLLM peer checkpoint absent from the cache is caught before vLLM is asked to
  serve it, so a campaign never downloads gigabytes mid-run.
- **Thermal floor / one engine at a time** - the isolation protocol lives in
  `../assay/scripts/fair-run.sh`: peers' processes are killed, cards return to an idle floor, and
  each engine is measured alone. The driver does not reimplement it.

## Status

Being built as an approved redesign (see the plan). Done: all five modalities native in assay;
`bench` plan core (store from config, catalogue from loken, matrix from the manifest). Landing next:
the `run` path (engine provisioning + the fair-run/assay sweep with resume), `bench_row.py` made
modality-aware, and the removal of the old `campaign.sh` / `full_matrix.sh` / `fair_bench.sh` /
`image_parity.sh` once their responsibilities have moved here.
