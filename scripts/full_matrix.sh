#!/usr/bin/env bash
# Full matrix. Logs on a REAL filesystem: /tmp here is a tmpfs and every byte
# written to it is RAM the models also need.
# Same working directory as campaign.sh, and for the same reason - derived, not hardcoded.
cd "$(dirname "$(readlink -f "$0")")/../.."
M=$(tr '\n' ' ' < ~/bench-logs/models.txt)
for ctx in 4096 131072; do
  for pr in short medium long; do
    for st in 1 0; do
      echo "########## GPU ctx=$ctx prompts=$pr stream=$st"
      CTX=$ctx PROMPTS=$pr STREAM=$st ./campaign.sh $M
    done
  done
done
for ctx in 4096 131072; do
  for pr in short medium long; do
    echo "########## CPU ctx=$ctx prompts=$pr"
    CTX=$ctx MODE=cpu PROMPTS=$pr ./campaign.sh $M
  done
done
echo "MATRICE COMPLETE"
