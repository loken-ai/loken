#!/bin/bash
# The corpus this reads is NOT in the repository: it was assembled from working documents and
# is training input, not source. Point EAGLE_CORPUS at your own, or drop one in scripts/eagle/.
# tools/eagle and its results/ live in the tree this was extracted from, not in this
# one. Name it rather than guess: running from the wrong root writes a valid-looking
# result against the wrong build.
cd "${EAGLE_ROOT:?set EAGLE_ROOT to the checkout holding tools/eagle}"
echo "=== XL capture (7000 prompts, gen_len 32) ==="
target/release/eagle_capture "${OLLAMA_MODELS:-$HOME/.ollama/models}/blobs/sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f" results/eagle/caps_xl.safetensors 32 0 0 0.0 "${EAGLE_CORPUS:-scripts/eagle/corpus_xl_clean.txt}" || exit 1
echo "=== train (default Huber, 15000 steps, batch 8192) ==="
"${VLLM_PY:-$HOME/vllm/bin/python}" tools/eagle/train_head.py --data results/eagle/caps_xl.safetensors   --out results/eagle/head_xl.safetensors   --hidden 4096 --n-head 32 --n-kv-head 8 --head-dim 128 --intermediate 12288   --rope-base 1e6 --rms-eps 1e-6 --steps 15000 --batch 8192 --lr 5e-4 || exit 1
echo "=== measure XL head at k=2 and k=3 (evalset) ==="
for kk in 2 3; do
  echo "-- k=$kk --"
  target/release/eagle_decode_test "${OLLAMA_MODELS:-$HOME/.ollama/models}/blobs/sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f" results/eagle/head_xl.safetensors     4096 32 8 128 12288 0 0 48 $kk evalset 0 2>&1 | grep -E "acceptance:"
done
echo "=== XL DONE ==="
