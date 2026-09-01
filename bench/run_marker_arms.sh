#!/usr/bin/env bash
# Run the marker arms (bench/marker_arms.json) end to end, up to the point that
# needs a person: render with run_graph_arms.py, blind with blind_batch.py, then
# print the two steps that are the owner's -- scoring in score.html and joining
# the export with the sealed key through score_session.py.
#
# Usage:
#   bench/run_marker_arms.sh            # render + blind; about fifty minutes of card
#   RUNS=3 bench/run_marker_arms.sh     # seeds per arm (default 2, the owner's call)
#   DRY_RUN=1 bench/run_marker_arms.sh  # check the server, print the commands, exit
#
# Needs a running, UNARMED ComfyUI on :8188 (bench/restart_comfy.sh starts one)
# with an empty queue. The output share is resolved by bench/_paths.comfy_output()
# or H3_COMFY_OUTPUT; its path is never written here.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFY="${COMFY:-$REPO/../..}"
PY="$COMFY/.venv/bin/python"
PORT="${PORT:-8188}"
RUNS="${RUNS:-2}"
SEED="${SEED:-730451892}"
DATE="$(date +%F)"
SESSION="${SESSION:-marker_arms_$DATE}"
OUT="bench/results/${DATE}_marker_arms.jsonl"
cd "$REPO"
[ -x "$PY" ] || { echo "no venv python at $PY"; exit 1; }
run() { echo "+ $*"; [ "${DRY_RUN:-0}" = "1" ] || "$@"; }

# 1. The server: answering, not armed for somebody else's capture, queue empty.
if ! curl -sf --max-time 5 "http://127.0.0.1:$PORT/system_stats" >/dev/null; then
    echo "ComfyUI is not answering on :$PORT -- start it with bench/restart_comfy.sh"
    exit 3
fi
pid="$(ss -lptnH "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1 || true)"
if [ -n "$pid" ] && [ -r "/proc/$pid/environ" ]; then
    armed="$(tr '\0' '\n' < "/proc/$pid/environ" | grep -E '^H3_(CAPTURE|PDD_OBSERVE|QUANT_OBSERVE|SOL_OBSERVE)=' || true)"
    if [ -n "$armed" ]; then
        echo "REFUSING: the server (pid $pid) is ARMED for capture; these renders would land in that record:"
        echo "$armed"
        echo "Ask the session that armed it, or bench/restart_comfy.sh --force if the capture is yours."
        exit 5
    fi
fi
queue="$(curl -s --max-time 5 "http://127.0.0.1:$PORT/queue")"
case "$queue" in
    *'"queue_running": []'*) ;;
    *) echo "the queue is not empty; wait for it: $queue"; exit 4 ;;
esac

# 2. Render: the arms alternating, seed + run index per run, one discarded warmup first.
run "$PY" bench/run_graph_arms.py --manifest bench/marker_arms.json \
    --runs "$RUNS" --seed "$SEED" --warmup st_prose --out "$OUT"

# 3. Blind: neutral clips, a sealed key under internal/blind_keys/, score.html
#    beside the clips. Six contests: each split-line form and each cutoff form
#    against this repo's own form, matched by run index.
run "$PY" bench/blind_batch.py --jsonl "$OUT" --session "$SESSION" --shuffle-seed 41 \
    --brief-file bench/marker_arms_brief.md \
    --pairs st_prose,st_split_tag --pairs st_prose,st_tag_only --pairs st_prose,st_split_only \
    --pairs co_piped_tight,co_unpiped_spaced --pairs co_piped_tight,co_unpiped_tight \
    --pairs co_piped_tight,co_piped_spaced

cat <<MSG

Rendered and blinded. The rest is yours:
  1. Open score.html in the batch directory blind_batch printed above
     (Video/blind/$SESSION on the output share). Score every pair and single in
     the order shown, then Export scores -> scores_$SESSION.json.
  2. $PY bench/score_session.py --scores <path to scores_$SESSION.json> \\
         --key internal/blind_keys/$SESSION.json
     writes bench/results/<date>_${SESSION}_verdict.json, the per-arm verdict
     the sister engine is waiting for.
MSG
