#!/usr/bin/env bash
#
# vllm-serve-ollama.sh - serve an Ollama GGUF blob with vLLM (OpenAI /v1 API)
# so the bench can hit the EXACT same weights Ollama runs (bit-for-bit fair).
#
# It resolves an Ollama tag -> its single-file GGUF blob via the local manifest
# store, then launches `vllm serve <blob>` with the environment vLLM needs on
# this box (venv `ninja` on PATH, single-GPU pinning, PCI bus order).
#
# Usage:
#   scripts/vllm-serve-ollama.sh qwen3:8b
#   scripts/vllm-serve-ollama.sh deepcoder:latest --gpu 1 --port 8000 --max-len 8192
#   scripts/vllm-serve-ollama.sh qwen3:8b --tokenizer Qwen/Qwen3-8B   # if GGUF-tokenizer conversion fails
#   scripts/vllm-serve-ollama.sh qwen3:8b -- --enforce-eager          # pass extra args to vllm after --
#
# Caveats (see project_vllm_install_bench memory):
#   * vLLM's GGUF path is experimental/under-optimized - this measures "who runs
#     this GGUF faster", not vLLM's production (FP8/AWQ) ceiling.
#   * Only dense archs load via GGUF (qwen3, llama, gemma, mistral, qwen2).
#     gpt-oss(MXFP4)/nemotron(mamba)/lfm2/qwen3.5(deltanet) will NOT load.
#   * GGUF tokenizer conversion is slow/unstable for large vocab - pass
#     --tokenizer <hf-repo> if startup hangs or errors on the tokenizer.
set -euo pipefail

VENV="${VLLM_VENV:-$HOME/vllm}"
OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
PORT=8000
GPU=1
GMU=0.85
MAXLEN=8192
TOKENIZER=""
# GGUF quant only supports float16/float32 (NOT bfloat16), and vLLM warns
# bf16 GGUF has precision issues on Blackwell - so default to float16.
DTYPE=float16
TAG=""
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)      PORT="$2"; shift 2 ;;
    --gpu)       GPU="$2"; shift 2 ;;
    --gpu-mem)   GMU="$2"; shift 2 ;;
    --max-len)   MAXLEN="$2"; shift 2 ;;
    --dtype)     DTYPE="$2"; shift 2 ;;
    --tokenizer) TOKENIZER="$2"; shift 2 ;;
    --)          shift; EXTRA=("$@"); break ;;
    -h|--help)   sed -n '2,30p' "$0"; exit 0 ;;
    -*)          echo "unknown flag: $1" >&2; exit 2 ;;
    *)           TAG="$1"; shift ;;
  esac
done

[[ -n "$TAG" ]] || { echo "error: missing <ollama-tag> (e.g. qwen3:8b)" >&2; exit 2; }
[[ -x "$VENV/bin/vllm" ]] || { echo "error: vllm not found at $VENV/bin/vllm" >&2; exit 1; }

# Resolve "<name>:<tag>" -> manifest file. Default tag is "latest". The Ollama
# manifest tree is <store>/manifests/registry.ollama.ai/<namespace>/<name>/<tag>
# (namespace defaults to "library" for official models).
name="${TAG%%:*}"
tag="${TAG#*:}"; [[ "$tag" == "$TAG" ]] && tag="latest"
manifest=""
for cand in \
  "$OLLAMA_MODELS/manifests/registry.ollama.ai/library/$name/$tag" \
  "$OLLAMA_MODELS/manifests/registry.ollama.ai/$name/$tag"; do
  [[ -f "$cand" ]] && { manifest="$cand"; break; }
done
if [[ -z "$manifest" ]]; then
  # last resort: search the tree for */<name>/<tag>
  manifest=$(find "$OLLAMA_MODELS/manifests" -type f -path "*/$name/$tag" 2>/dev/null | head -1)
fi
[[ -n "$manifest" && -f "$manifest" ]] || {
  echo "error: no manifest for '$TAG' under $OLLAMA_MODELS/manifests" >&2
  echo "       available:" >&2
  find "$OLLAMA_MODELS/manifests" -type f 2>/dev/null | sed "s|$OLLAMA_MODELS/manifests/registry.ollama.ai/||" >&2
  exit 1
}

# Extract the model-layer blob path + GGUF architecture from the manifest.
read -r BLOB ARCH < <(python3 - "$manifest" "$OLLAMA_MODELS" <<'PY'
import json, sys, os
manifest, store = sys.argv[1], sys.argv[2]
m = json.load(open(manifest))
dig = next((l['digest'] for l in m['layers']
            if l['mediaType'] == 'application/vnd.ollama.image.model'), None)
if not dig:
    sys.exit("no model layer in manifest")
blob = os.path.join(store, 'blobs', dig.replace('sha256:', 'sha256-'))
# Read GGUF arch from the header (general.architecture kv) without a full parse:
# scan the first 64KB for the key, value follows as a length-prefixed string.
arch = '?'
try:
    with open(blob, 'rb') as f:
        head = f.read(65536)
    key = b'general.architecture'
    i = head.find(key)
    if i != -1:
        # GGUF: key, then value-type(uint32=8 for string), then u64 len, then bytes
        import struct
        p = i + len(key)
        (vtype,) = struct.unpack_from('<I', head, p); p += 4
        (slen,) = struct.unpack_from('<Q', head, p); p += 8
        arch = head[p:p+slen].decode('utf-8', 'replace')
except Exception:
    pass
print(blob, arch)
PY
)
[[ -f "$BLOB" ]] || { echo "error: blob not found: $BLOB" >&2; exit 1; }

# Warn on archs that vLLM's GGUF path cannot load.
case "$ARCH" in
  qwen3|qwen2|llama|gemma|gemma2|gemma3|mistral|phi3|stablelm|starcoder2)
    : ;;  # known vLLM-GGUF-loadable dense archs
  *)
    echo "  ⚠️  GGUF arch '$ARCH' may not load on vLLM's experimental GGUF path" >&2
    echo "      (MoE/hybrid archs like gpt-oss/nemotron/lfm2/qwen3.5 are unsupported)." >&2 ;;
esac

sz=$(du -h "$BLOB" | cut -f1)
echo "  Ollama tag : $TAG  (arch=$ARCH, $sz)"
echo "  GGUF blob  : $BLOB"
echo "  Serving on : GPU$GPU  port $PORT  max-len $MAXLEN  gpu-mem $GMU"
[[ -n "$TOKENIZER" ]] && echo "  Tokenizer  : $TOKENIZER (HF)" \
                      || echo "  Tokenizer  : <from GGUF metadata> (pass --tokenizer <hf-repo> if this fails)"
echo

# vLLM needs the venv's `ninja` on PATH (runtime kernel JIT). Pin to one GPU in
# PCI bus order. --served-model-name set to the Ollama tag so the bench's
# --models value matches what /v1/models reports.
export PATH="$VENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cmd=( "$VENV/bin/vllm" serve "$BLOB"
      --served-model-name "$TAG"
      --port "$PORT"
      --max-model-len "$MAXLEN"
      --dtype "$DTYPE"
      --gpu-memory-utilization "$GMU" )
[[ -n "$TOKENIZER" ]] && cmd+=( --tokenizer "$TOKENIZER" )
cmd+=( "${EXTRA[@]}" )

echo "+ ${cmd[*]}"
exec "${cmd[@]}"
