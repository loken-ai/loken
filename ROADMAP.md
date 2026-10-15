# Roadmap

What the next releases bring, feature by feature. [`CHANGELOG.md`](CHANGELOG.md) lists what is
done; [`docs/STATUS.md`](docs/STATUS.md) says what is measured. A feature is done when its
acceptance is met and measured; until then it stays here with its state.

## 0.3.0

| Feature | Acceptance | State |
|---|---|---|
| One model across several machines, cut by layer | A model no single machine serves well, cut across machines from each one's measured layer cost, faster than the best machine alone | Done: three machines, peers measured on a sample of layers |
| Tensor parallelism in the generic path | A layer's dense projections split across cards inside the one placement path, any number of cards, mixtures included | In progress: dense layers on any number of cards, recorded in the decode graph, chosen from measured sums and products; mixtures not split yet |
| Expert parallelism | A mixture layer's experts split across cards where measured links make it pay | Done for experts that would otherwise sit on the host |
| Pipeline parallelism across machines | A chain of any length, the prompt-lookup verify crossing every cut | Done |
| Data parallelism | Copies of a model served side by side, requests spread over them | Done on one machine; across machines through the cluster's routing |
| Context parallelism | A long context's attention split across cards | Planned |
| Prefill and decode on different instances | A prompt read on one instance and decoded on another, its state handed over | Planned |
| CUDA on the published cudarc crate | No patched fork, graphs still recorded and replayed | Done |
| A Linux binary | One archive that starts on a machine without CUDA libraries and serves on the CPU, and a second that adds them for the cards | Done |
| CPU kernels chosen at run time | One binary serving AVX2 and AVX-512 machines, the level capped by configuration | Done |
| Qwen3.8 at full speed | The three Qwen3.8 files at or above the reference engine on one and two cards | In progress: on two cards near the reference, on one card behind |
| A coding agent on large models | verve completing a multi-step task on each Qwen3.8 file and on DeepSeek V4.1, tool calls and reasoning kept apart | In progress |
| A faster test suite | The checks a change must pass in a fraction of today's time | Planned |

## 0.4.0

| Feature | Acceptance | State |
|---|---|---|
| A backend for AMD cards | One model served on an AMD card through the same placement path, measured against the reference engine on that card | Planned |
| Forecasting and tabular models | Time series forecasts and tabular predictions served beside the language models, measured against naive baselines | Planned |
