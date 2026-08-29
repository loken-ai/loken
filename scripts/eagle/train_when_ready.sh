#!/bin/bash
set -e
# tools/eagle and its results/ live in the tree this was extracted from, not in this
# one. Name it rather than guess: running from the wrong root writes a valid-looking
# result against the wrong build.
cd "${EAGLE_ROOT:?set EAGLE_ROOT to the checkout holding tools/eagle}"
GGUF=${OLLAMA_MODELS:-$HOME/.ollama/models}/blobs/sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f
PY="${VLLM_PY:-$HOME/vllm/bin/python}"; [ -x "$PY" ] || PY=python3
# wait for capture process (PID 736289) to exit
while kill -0 736289 2>/dev/null; do sleep 15; done
sleep 3
echo "capture done; training long run..."
"$PY" tools/eagle/train_head.py \
  --data results/eagle/caps_qwen3_big2.safetensors \
  --out results/eagle/head_qwen3_big2.safetensors \
  --hidden 4096 --n-head 32 --n-kv-head 8 --head-dim 128 --intermediate 12288 \
  --rope-base 1e6 --rms-eps 1e-6 --steps 12000 --batch 2048 --lr 5e-4
echo "training done; measuring acceptance..."
target/release/eagle_decode_test "$GGUF" results/eagle/head_qwen3_big2.safetensors \
  4096 32 8 128 12288 0 0 64 4 "1,1849,374,264,6485,3575,13" 0 || true
