#!/bin/bash
# tools/eagle and its results/ live in the tree this was extracted from, not in this
# one. Name it rather than guess: running from the wrong root writes a valid-looking
# result against the wrong build.
cd "${EAGLE_ROOT:?set EAGLE_ROOT to the checkout holding tools/eagle}"
# wait for the broad pipeline (capture/train/measure) to fully finish
while pgrep -f "broad_run|eagle_capture|train_head" >/dev/null; do sleep 20; done
sleep 5
echo "================ RIGOROUS evalset (6 prompts) per head ================"
for h in head_qwen3_gpu head_qwen3_big2 head_qwen3_corpus head_broad; do
  f=results/eagle/$h.safetensors
  [ -f "$f" ] || { echo "$h: MISSING"; continue; }
  echo "---- $h ----"
  target/release/eagle_decode_test "${OLLAMA_MODELS:-$HOME/.ollama/models}/blobs/sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f" "$f" 4096 32 8 128 12288 0 0 48 4 evalset 0 2>&1 | grep -E "acceptance:" || echo "  (error)"
done
echo "================ done ================"
