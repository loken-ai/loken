#!/usr/bin/env bash
# Does a change to the image path still produce the same image?
#
# The pipeline is deterministic at a fixed seed - two runs give byte-identical PNGs - so this
# can compare hashes rather than eyeball a render. That matters for a refactor: a transformer
# rearranged across devices, or a block type swapped for another, produces a picture that still
# looks like a teapot while being a different teapot, and nothing in a build or a test suite
# notices. The image is the specification.
#
#   scripts/image_parity.sh record [family]   # after a change you have verified BY LOOKING
#   scripts/image_parity.sh check  [family]   # before and after any change to the image path
#
# `family` is zimage (the default) or flux. They exercise different transformers, so a change
# to one says nothing about the other.
#
# It talks to a daemon you started; it does not start one, because the daemon's working
# directory and configuration decide which weights load and this must measure the one you run.
#
# THE HASH IS NOT A PROPERTY OF THE CODE ALONE. The placer reads free VRAM at load, so an
# ambient allocation decides whether the text encoder lands on a card or the host and whether
# the transformer runs whole or split - and those answer differently in the last bits. Measured
# across four runs of one binary: four placements, and not one hash. Run this on a machine with
# nothing else on the cards, or it reports a difference that is the machine's and not the code's.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

PORT="${PORT:-11435}"
HOST="${HOST:-127.0.0.1}"
FAMILY="${2:-zimage}"
BASELINE="scripts/image_parity.$FAMILY.sha256"

case "$FAMILY" in
  zimage) MODEL_DEFAULT=z-image-turbo ;;
  flux)   MODEL_DEFAULT=flux-schnell ;;
  *)      echo "unknown family '$FAMILY' - use zimage or flux" >&2; exit 2 ;;
esac
OUT="${OUT:-$(mktemp -d)/parity.png}"

# One prompt, one seed, one size. Not a matrix: this answers "did it change", and a matrix
# answers "is it good", which is a different question with a different instrument.
MODEL="${MODEL:-$MODEL_DEFAULT}"
PROMPT="a copper teapot on a wooden table, morning light"
SEED=424242
SIZE=512x512

render() {
  local body
  body=$(printf '{"model":"%s","prompt":"%s","size":"%s","n":1,"seed":%d,"response_format":"b64_json"}' \
         "$MODEL" "$PROMPT" "$SIZE" "$SEED")
  curl -sf --max-time 900 -X POST "http://$HOST:$PORT/v1/images/generations" \
       -H 'Content-Type: application/json' -d "$body" |
    python3 -c '
import base64, json, sys, pathlib
d = json.load(sys.stdin)
if not d.get("data"):
    sys.exit("the server returned no image: " + json.dumps(d)[:300])
pathlib.Path(sys.argv[1]).write_bytes(base64.b64decode(d["data"][0]["b64_json"]))
' "$1"
}

case "${1:-check}" in
  record)
    render "$OUT" || exit 1
    sha256sum < "$OUT" | cut -d' ' -f1 > "$BASELINE"
    echo "recorded $(cat "$BASELINE")"
    echo "the image is at $OUT - LOOK at it before trusting the hash, because a hash is happy"
    echo "to certify noise."
    ;;
  check)
    [ -f "$BASELINE" ] || { echo "no baseline: run '$0 record' first"; exit 2; }
    render "$OUT" || exit 1
    got=$(sha256sum < "$OUT" | cut -d' ' -f1)
    want=$(cat "$BASELINE")
    if [ "$got" = "$want" ]; then
      echo "identical - $got"
    else
      echo "DIFFERENT"
      echo "  expected $want"
      echo "  got      $got"
      echo "  the render is at $OUT; compare it with the one you recorded before deciding"
      echo "  whether this is a regression or an intended change."
      exit 1
    fi
    ;;
  *)
    echo "usage: $0 [record|check]" >&2
    exit 2
    ;;
esac
