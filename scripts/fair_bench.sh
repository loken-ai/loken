#!/usr/bin/env bash
# Moved: the protocol now ships with the tool it drives, at assay/scripts/fair-run.sh, so a
# third party can replay a fair comparison from that repository alone. Keeping a second copy
# here is how the two drift - this is a shim, not an implementation.
#
# It supplies only where this tree keeps its binaries. Everything else is the published
# script's, including where ollama stores its blobs: set MODELS_DIR (or OLLAMA_MODELS, which
# ollama itself reads) in your environment if they are not in the default location. A path
# that is true on one machine does not belong in a file everyone gets.
exec env \
    ASSAY="${ASSAY:-../assay/target/release/assay}" \
    LOKEN_BIN="${LOKEN_BIN:-target/release/lokend}" \
    ${MODELS_DIR:+MODELS_DIR="$MODELS_DIR"} \
    ${OLLAMA_MODELS:+MODELS_DIR="$OLLAMA_MODELS"} \
    ../assay/scripts/fair-run.sh "$@"
