#!/bin/bash
# tools/eagle and its results/ live in the tree this was extracted from, not in this
# one. Name it rather than guess: running from the wrong root writes a valid-looking
# result against the wrong build.
cd "${EAGLE_ROOT:?set EAGLE_ROOT to the checkout holding tools/eagle}"
echo "=== train XL (CPU-data, batch 4096, 15000 steps) ==="
"${VLLM_PY:-$HOME/vllm/bin/python}" tools/eagle/train_head.py --data results/eagle/caps_xl.safetensors   --out results/eagle/head_xl.safetensors   --hidden 4096 --n-head 32 --n-kv-head 8 --head-dim 128 --intermediate 12288   --rope-base 1e6 --rms-eps 1e-6 --steps 15000 --batch 4096 --lr 5e-4 || exit 1
echo "=== measure XL head k=2,3 (evalset, wall-clock) ==="
for kk in 2 3; do echo "-- k=$kk --"; target/release/eagle_decode_test "${OLLAMA_MODELS:-$HOME/.ollama/models}/blobs/sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f" results/eagle/head_xl.safetensors 4096 32 8 128 12288 0 0 48 $kk evalset 0 2>&1 | grep -E "acceptance:|WALL-CLOCK"; done
echo "=== XL DONE ==="
