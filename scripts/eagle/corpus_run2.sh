#!/bin/bash
# tools/eagle and its results/ live in the tree this was extracted from, not in this
# one. Name it rather than guess: running from the wrong root writes a valid-looking
# result against the wrong build.
cd "${EAGLE_ROOT:?set EAGLE_ROOT to the checkout holding tools/eagle}"
echo "=== real-corpus capture (--prompts, gen_len 256) ==="
target/release/eagle_capture "${OLLAMA_MODELS:-$HOME/.ollama/models}/blobs/sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f" results/eagle/caps_qwen3_corpus.safetensors 256 0 0 0.0 results/eagle/prompts.txt || exit 1
echo "=== train (default Huber) ==="
"${VLLM_PY:-$HOME/vllm/bin/python}" tools/eagle/train_head.py --data results/eagle/caps_qwen3_corpus.safetensors   --out results/eagle/head_qwen3_corpus.safetensors   --hidden 4096 --n-head 32 --n-kv-head 8 --head-dim 128 --intermediate 12288   --rope-base 1e6 --rms-eps 1e-6 --steps 8000 --batch 2048 --lr 5e-4 || exit 1
echo "=== measure real-corpus head ==="
target/release/eagle_decode_test "${OLLAMA_MODELS:-$HOME/.ollama/models}/blobs/sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f" results/eagle/head_qwen3_corpus.safetensors   4096 32 8 128 12288 0 0 64 4 "1,1849,374,264,6485,3575,13" 0 || true
