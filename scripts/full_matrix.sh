#!/usr/bin/env bash
# Full matrix. Logs on a REAL filesystem: /tmp here is a tmpfs and every byte
# written to it is RAM the models also need.
# Same working directory as campaign.sh, and for the same reason - derived, not hardcoded.
cd "$(dirname "$(readlink -f "$0")")/../.."
M=$(tr '\n' ' ' < ~/bench-logs/models.txt)

# The thermal floor, measured ONCE and exported to every cell.
#
# Each cell would otherwise read its own floor at start-up, on cards still hot from the cell
# before, and the gate would open immediately on a reading that means nothing. Taken here,
# once, so every cell waits to come back to the same place.
#
# What makes a reading the floor is that nothing is using the cards - not that it is below
# some number. These cards idle at 49C in this room; a box that idled at 35 and one that idled
# at 55 would both be cold, and any absolute threshold would be wrong for one of them. So the
# check is for work, not for heat: no compute, and no resident memory beyond driver overhead.
if command -v nvidia-smi >/dev/null 2>&1; then
  busy=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -rn | head -1)
  held=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
  if [ "${busy:-0}" -gt 5 ] || [ "${held:-0}" -gt 600 ]; then
    echo "cards are still working (${busy}% busy, ${held} MiB held): stop what holds them, or the floor recorded here is a ceiling" >&2
    exit 2
  fi
  export GPU_IDLE_C
  GPU_IDLE_C=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -rn | head -1)
  echo "thermal floor for this matrix: ${GPU_IDLE_C}C (idle, nothing resident)"
fi

failed=0
for ctx in 4096 131072; do
  for pr in short medium long; do
    for st in 1 0; do
      echo "########## GPU ctx=$ctx prompts=$pr stream=$st"
      CTX=$ctx PROMPTS=$pr STREAM=$st ./campaign.sh $M || failed=$((failed + 1))
    done
  done
done
for ctx in 4096 131072; do
  for pr in short medium long; do
    echo "########## CPU ctx=$ctx prompts=$pr"
    CTX=$ctx MODE=cpu PROMPTS=$pr ./campaign.sh $M || failed=$((failed + 1))
  done
done
# A sweep that failed is not a sweep that measured nothing interesting: on 2026-08-31 all
# eighteen exited on a missing guard in under a minute and this line printed regardless, which
# reads as a finished campaign. It now states what happened.
if [ "$failed" -gt 0 ]; then
  echo "MATRICE INCOMPLETE: $failed sweep(s) failed - nothing here is a result" >&2
  exit 1
fi
echo "MATRICE COMPLETE"
