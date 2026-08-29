#!/usr/bin/env bash
#
# vllm-serve-hf.sh - serve a HuggingFace model with vLLM (OpenAI /v1 API) on
# vLLM's native/optimized path (FP16, or AWQ/GPTQ/FP8 quants - auto-detected
# from the repo's config). This is the way to benchmark vLLM "at its best",
# vs the Ollama-GGUF reuse path (which vLLM's loader rejects for K-quants).
#
# Usage:
#   scripts/vllm-serve-hf.sh Qwen/Qwen3-8B-AWQ
#   scripts/vllm-serve-hf.sh Qwen/Qwen3-8B-AWQ --name qwen3:8b   # match an ollama tag so one assay sweep hits both
#   scripts/vllm-serve-hf.sh Qwen/Qwen3-8B-FP8 --gpu 1 --tp 1 --max-len 8192
#   scripts/vllm-serve-hf.sh meta-llama/... -- --quantization fp8   # pass extra vllm args after --
#
# Notes:
#   * Weights auto-download to the HF cache on first serve (HF_HOME below).
#   * --name sets --served-model-name so `assay --models <that>` matches the
#     /v1/models id (use the ollama tag to compare the same logical model).
#   * vLLM auto-detects AWQ/GPTQ/FP8 from config.json; override with -- --quantization.
#   * For a model that needs >1 GPU (won't fit 16GB), use --tp 2 (and free GPU0).
set -euo pipefail

VENV="${VLLM_VENV:-$HOME/vllm}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"   # one HF store, so a model is fetched once
PORT=8000
GPU=1
TP=1
GMU=0.85
MAXLEN=8192
DTYPE=""          # let vLLM/quant pick (AWQ->fp16); override with --dtype
NAME=""
REPO=""
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)         PORT="$2"; shift 2 ;;
    --gpu)          GPU="$2"; shift 2 ;;
    --tp)           TP="$2"; shift 2 ;;
    --gpu-mem)      GMU="$2"; shift 2 ;;
    --max-len)      MAXLEN="$2"; shift 2 ;;
    --dtype)        DTYPE="$2"; shift 2 ;;
    --name)         NAME="$2"; shift 2 ;;
    --)             shift; EXTRA=("$@"); break ;;
    -h|--help)      sed -n '2,20p' "$0"; exit 0 ;;
    -*)             echo "unknown flag: $1" >&2; exit 2 ;;
    *)              REPO="$1"; shift ;;
  esac
done

[[ -n "$REPO" ]] || { echo "error: missing <hf-repo> (e.g. Qwen/Qwen3-8B-AWQ)" >&2; exit 2; }
[[ -x "$VENV/bin/vllm" ]] || { echo "error: vllm not found at $VENV/bin/vllm" >&2; exit 1; }
[[ -n "$NAME" ]] || NAME="$REPO"

echo "  HF repo    : $REPO"
echo "  Served as  : $NAME   (assay --models '$NAME')"
echo "  Serving on : GPU$GPU  tp=$TP  port $PORT  max-len $MAXLEN  gpu-mem $GMU"
echo "  HF cache   : $HF_HOME"
echo

# vLLM needs the venv `ninja` on PATH (runtime kernel JIT). Pin GPUs in PCI order.
export PATH="$VENV/bin:$PATH"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ "$TP" -gt 1 ]]; then
  unset CUDA_VISIBLE_DEVICES   # let vLLM use all visible GPUs for tensor-parallel
else
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

cmd=( "$VENV/bin/vllm" serve "$REPO"
      --served-model-name "$NAME"
      --port "$PORT"
      --tensor-parallel-size "$TP"
      --max-model-len "$MAXLEN"
      --gpu-memory-utilization "$GMU" )
[[ -n "$DTYPE" ]] && cmd+=( --dtype "$DTYPE" )
cmd+=( "${EXTRA[@]}" )

echo "+ ${cmd[*]}"
exec "${cmd[@]}"
