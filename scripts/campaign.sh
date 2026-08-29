#!/usr/bin/env bash
# The campaign, run through the recovered three-engine harness.
#
# fair_bench.sh owns the protocol - each engine measured alone with the others' processes
# stopped, ollama's silent CPU-fallback detected and retried, a thermal gate between runs.
# This only names the cells and keeps the report in step with them.
set -u
# The harness has to run from the tree holding the configuration it is measuring: launched from
# elsewhere it picks up another config, hence another VRAM budget and another placement, and
# reports that as a result rather than as an error. Derived from this script's own location, so
# neither renaming the checkout nor going through the campaign.sh symlink changes what runs.
cd "$(dirname "$(readlink -f "$0")")/.."
S="${CAMPAIGN_SCRATCH:-$(mktemp -d -t loken-campaign-XXXXXX)}"
CTX=${CTX:-4096}
# The sweep has four dimensions, not one: prompt length, context window, streaming and
# device. A table covering a single point of each says nothing about the rest.
PROMPTS=${PROMPTS:-short}
MODE=${MODE:-gpu}
# Both cards visible: the harness default, and the only setting that lets either engine
# place a model the way it would in production. Pinning one card measures a machine
# nobody runs.
PIN=${PIN:-}
./assert-binary-current.sh || exit 1
for m in "$@"; do
    tag=$(echo "$m" | tr ':/' '__')
    suffix=""; [ "$MODE" = cpu ] && suffix="_cpu"
    psuf=""; [ "$PROMPTS" != short ] && psuf="_$(echo "$PROMPTS" | tr ,  _)"
    mode="stream"; [ "${STREAM:-1}" = 1 ] || mode="non-stream"
    OUT="results/${tag}_${CTX}_${mode}${psuf}${suffix}.json"
    # A tag the store does not hold makes the bench client pull it from the network.
    # That is right for a workstation and wrong for a measurement: it fetched gigabytes
    # onto a full disk once already. A missing model is a cell to skip, not to download.
    case "$m" in *:*) mp="${m/:/\/}";; *) mp="$m/latest";; esac
    MF="${OLLAMA_MODELS:-$HOME/.ollama/models}/manifests/registry.ollama.ai/library/$mp"
    if [ ! -f "$MF" ]; then
        echo "ABSENT DU DEPOT, cellule ignoree: $m"; continue
    fi
    # A manifest is not the weights. A tag can name a model layer whose blob was never
    # written - the requantised 70B variants do exactly that - and the guard above then
    # passes a model no engine can load, which sends the bench client to the NETWORK for
    # it. Resolve the layer to a file on disk before naming the cell.
    BLOB=$(python3 -c "
import json,sys
d=json.load(open('$MF'))
print(next(l['digest'] for l in d['layers'] if 'image.model' in l['mediaType']).replace('sha256:','sha256-'))
" 2>/dev/null || true)
    if [ -z "$BLOB" ] || [ ! -f "${OLLAMA_MODELS:-$HOME/.ollama/models}/blobs/$BLOB" ]; then
        echo "MANIFESTE ORPHELIN (poids absents), cellule ignoree: $m"; continue
    fi
    # Already measured by THIS binary. The sweep is eighteen passes over the store and it
    # does get interrupted - by a defect worth stopping for, by a reboot, by a mistake of
    # mine - so resuming must not mean re-measuring what is already correct. The stamp is
    # the binary's mtime, the same identity assert-binary-current.sh uses, so a rebuild
    # invalidates every cell exactly as it should and nothing stale survives.
    BINT=$(stat -c %Y target/release/lokend)
    if [ -s "$OUT" ] && [ "$(python3 -c "
import json
try: print(json.load(open('$OUT')).get('engine_build'))
except Exception: print('')" 2>/dev/null)" = "$BINT" ]; then
        echo "DEJA MESURE par ce binaire, cellule conservee: $m"; continue
    fi
    echo "===== $m"
    # A cell that hangs must not take the sweep with it. The longest legitimate cell
    # measured so far is ~13 minutes (a 70B spilling to the host at 4096); the untested
    # regimes above - a 128k context, the same model on CPU - are slower and their upper
    # bound is unknown, so the cap is set well beyond any of them and only catches a cell
    # that is not progressing. It is announced when it fires: a silently truncated cell
    # reads as a model that produced nothing, which is a different result entirely.
    CELL_CAP=${CELL_CAP:-7200}
    # The third engine. fair_bench.sh has taken a VLLM_SERVE since it was written and no
    # driver ever set one, so the column the campaign's own goal names sat at 9 rows
    # against 204 - all from a single day, all on one cell shape.
    #
    # It cannot ride the GGUF: vLLM's loader rejects ollama's mixed-precision K-quants,
    # since a fused qkv wants uniform precision across q/k/v. So each tag needs a
    # hand-picked HF checkpoint of comparable bit width - AWQ 4-bit against Q4_K_M - and
    # only the tags that HAVE one get a vLLM cell. The rest are honestly empty:
    #   gemma4:*            no unambiguous HF peer for the GGUF arch 'gemma4'
    #   lfm2*, qwen3.5*     hybrid / DeltaNet arches vLLM cannot load
    #   nemotron-3-nano     an FP8 peer exists but is not in the cache and the disk is at 98%
    # CPU mode gets none: vLLM is GPU-only here.
    VLLM_SERVE=""
    if [ "$MODE" = gpu ]; then
        case "$m" in
            qwen3:8b)                VLLM_SERVE="hf:Qwen/Qwen3-8B-AWQ --name $m";;
            deepcoder:14b)           VLLM_SERVE="hf:Quickpanda/deepcoder-14b-preview-awq --name $m";;
            devstral:24b)            VLLM_SERVE="hf:cyankiwi/Devstral-Small-2507-AWQ-4bit --name $m";;
            gpt-oss:20b)             VLLM_SERVE="hf:openai/gpt-oss-20b --name $m";;
            qwen3-coder:30b)         VLLM_SERVE="hf:cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit --name $m --tp 2";;
        esac
    fi
    STREAM=${STREAM:-1} BENCH_MODE=$MODE GPU_PIN=$([ "$MODE" = gpu ] && echo "$PIN" || echo "") OUT="$OUT" \
        VLLM_SERVE="$VLLM_SERVE" \
        timeout --kill-after=60 "$CELL_CAP" \
        scripts/fair_bench.sh --models "$m" --prompts "$PROMPTS" --num-ctx "$CTX" \
        --max-tokens ${MAXTOK:-128} --iterations 3 2>&1 | tee "$S/run_${tag}_${CTX}${psuf}${suffix}.log" | tail -4
    # Captured before anything else runs: PIPESTATUS holds the last pipeline only, and
    # the test that reads it is itself a command.
    rc=${PIPESTATUS[0]}
    if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
        # The engines the interrupted harness left running are killed by the next cell,
        # which begins with its own kill_all and waits for VRAM to drain.
        echo "CELLULE INTERROMPUE apres ${CELL_CAP}s (pas de progression): $m ctx=$CTX $PROMPTS $MODE"
    fi
    [ -s "$OUT" ] || { echo "PAS DE RESULTAT pour $m"; continue; }
    python3 - "$OUT" "$(stat -c %Y target/release/lokend)" <<'STAMP'
import json,sys
p,b=sys.argv[1],int(sys.argv[2])
d=json.load(open(p)); d["engine_build"]=b
json.dump(d,open(p,"w"))
STAMP
    python3 scripts/bench_row.py docs/BENCHMARKS.md results >/dev/null
    # The plot is regenerated by the same cell that writes the row, so the two never
    # disagree about what has been measured. It opens from disk - no server, no network.
    python3 plot-bench.py >/dev/null
    # And the goal itself, after every cell. The per-cell numbers say what this model did;
    # this says whether the campaign is winning, which is the question being asked - and a
    # sweep that only scores itself at the end cannot tell a correction from a regression
    # while there is still time to act on it.
    python3 scripts/scoreboard.py results --brief || true
done
echo FIN
