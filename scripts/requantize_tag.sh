#!/bin/bash
# Produce a new ollama tag from an existing one by requantising chosen tensors.
#
# Why per-tensor and not a whole-model type: decode is bound by the bytes read per
# token, and those bytes are not evenly useful. In a dense transformer the feed-forward
# projections carry ~82% of the weights, so moving them one step down buys most of the
# saving, while attention and the embedding stay where they are and keep the quality
# that a uniform step would have spent everywhere.
#
# Why it matters that the whole thing lands in VRAM: with weights split across cards at
# batch 1 the cards run in sequence, so the time per token is the SUM of each card's
# bytes over its own bandwidth. Anything left on the host is read over a link an order of
# magnitude slower and dominates the sum. A variant that fits is worth more than a
# variant that is merely smaller.
#
# The blob is named by the sha256 of what was actually written. A manifest whose digest
# was invented instead points at nothing, no engine can load the tag, and the bench
# client answers the miss by pulling gigabytes from the network.
set -euo pipefail

QUANTIZE="${QUANTIZE:-/usr/local/lib/ollama/llama-quantize}"
STORE="${STORE:-${OLLAMA_MODELS:-$HOME/.ollama/models}}"
LIB="$STORE/manifests/registry.ollama.ai/library"

usage() {
    cat >&2 <<EOF
usage: $0 <source-tag> <new-tag> <base-type> [tensor=type ...]

  $0 deepseek-r1:70b deepseek-r1:70b-q2ffn q4_K \\
     ffn_gate=q2_K ffn_up=q2_K ffn_down=q3_K

Tensor names are the llama.cpp suffixes (ffn_gate, ffn_up, ffn_down, attn_q, attn_k,
attn_v, attn_output, token_embd, output). Anything not named keeps the base type.
EOF
    exit 1
}
[ $# -ge 3 ] || usage
SRC="$1"; DST="$2"; BASE="$3"; shift 3

tag_path() { case "$1" in *:*) echo "${1/:/\/}";; *) echo "$1/latest";; esac; }
SRC_MF="$LIB/$(tag_path "$SRC")"
DST_MF="$LIB/$(tag_path "$DST")"
[ -f "$SRC_MF" ] || { echo "unknown tag: $SRC" >&2; exit 1; }
[ -e "$DST_MF" ] && { echo "$DST already exists - pick another name" >&2; exit 1; }

SRC_DIGEST=$(python3 -c "
import json
d = json.load(open('$SRC_MF'))
print(next(l['digest'] for l in d['layers'] if 'image.model' in l['mediaType']))
")
SRC_BLOB="$STORE/blobs/${SRC_DIGEST/sha256:/sha256-}"
[ -f "$SRC_BLOB" ] || { echo "$SRC has a manifest but no weights: $SRC_BLOB" >&2; exit 1; }

# Requantising from an already-quantised source loses more than quantising from the
# original precision would. It is what the store holds, so it is what we measure - but
# the loss is real and the output has to be checked, not assumed.
TT=()
for spec in "$@"; do TT+=(--tensor-type "$spec"); done

SRC_GB=$(( $(stat -c%s "$SRC_BLOB") / 1000000000 ))
FREE_GB=$(df -B1000000000 --output=avail "$STORE" | tail -1 | tr -d ' ')
[ "$FREE_GB" -gt "$SRC_GB" ] || { echo "only ${FREE_GB} GB free, source is ${SRC_GB} GB" >&2; exit 1; }

WORK="$STORE/blobs/.requant-$$.gguf"
trap 'rm -f "$WORK"' EXIT
echo "▶ $SRC (${SRC_GB} GB) -> $DST   base=$BASE ${*:-}"
"$QUANTIZE" --allow-requantize "${TT[@]}" "$SRC_BLOB" "$WORK" "$BASE" "$(nproc)"

# Name the blob after what was written, never after what was intended.
DIGEST=$(sha256sum "$WORK" | cut -d' ' -f1)
OUT_BLOB="$STORE/blobs/sha256-$DIGEST"
SIZE=$(stat -c%s "$WORK")
mv "$WORK" "$OUT_BLOB"
trap - EXIT

mkdir -p "$(dirname "$DST_MF")"
python3 - "$SRC_MF" "$DST_MF" "sha256:$DIGEST" "$SIZE" <<'EOF'
import json, sys
src, dst, digest, size = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
m = json.load(open(src))
# Everything but the weights is carried over untouched: the chat template above all,
# since a model served with an approximation of its own format stops early and reads
# as a model defect.
for layer in m['layers']:
    if 'image.model' in layer['mediaType']:
        layer['digest'], layer['size'] = digest, size
json.dump(m, open(dst, 'w'))
EOF

echo "✓ $DST  $(( SIZE / 1000000000 )).$(( SIZE % 1000000000 / 100000000 )) GB  ->  $OUT_BLOB"
echo "  next: check the OUTPUT before the rate - a requantised model that answers"
echo "  fluently from a damaged distribution passes every check that is not a comparison."
