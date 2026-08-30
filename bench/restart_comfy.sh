#!/usr/bin/env bash
# Restart ComfyUI and REFUSE to return until the new process is provably newer
# than the code it is meant to be running.
#
# ## Why this exists
#
# Three times on 2026-08-29 a measurement was taken against a ComfyUI that had
# not reloaded, and every one produced a plausible number rather than an error:
#
#   * a "merged kernel is bit-identical" claim, taken against a process that
#     started 82 seconds BEFORE the wheel was installed;
#   * a red-proof of a new guard that failed in the wrong place, because the
#     server predated the guard by an hour;
#   * a graph validation that passed against a schema missing a new node.
#
# Every one had the same cause: `pkill ... ; nohup ./start.sh &` in one
# compound command. The kill takes the shell down with it, the launch never
# runs, the OLD server keeps the port, and the next `curl /system_stats`
# answers cheerfully. `start.sh` then logs "Port 8188 is already in use" into
# a file nobody reads.
#
# **The failure mode is a stale success, so politeness is not a fix.** This
# script asserts the thing that was assumed.
#
# ## What it guarantees on exit 0
#
#   1. No process is left holding the port from before this run.
#   2. The server answers /system_stats.
#   3. Its start time is LATER than every path passed in --newer-than.
#
# Usage:
#   bench/restart_comfy.sh                       # restart, verify liveness
#   bench/restart_comfy.sh --newer-than vendor/sol_attn_minimax.py
#   bench/restart_comfy.sh --newer-than <(...)   # any number of --newer-than
#   bench/restart_comfy.sh --kernel              # shorthand: the installed
#                                                # comfy_kitchen dist-info
#
# Exit codes: 0 ok, 1 usage/setup, 2 port never freed, 3 never came up,
#             4 came up but is OLDER than something it must postdate.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFY="${COMFY:-$REPO/../..}"
PORT="${PORT:-8188}"
MODE="${MODE:-default}"
NEWER=()

while [ $# -gt 0 ]; do
    case "$1" in
        --newer-than) NEWER+=("$2"); shift 2 ;;
        --kernel)
            # Resolve the installed comfy_kitchen's dist-info rather than
            # taking a path: which build is installed is exactly the thing a
            # caller passing this flag does not want to have to know.
            ck="$(ls -d "$COMFY"/.venv/lib/python*/site-packages/comfy_kitchen-*.dist-info 2>/dev/null | head -1)"
            [ -n "$ck" ] || { echo "no comfy_kitchen dist-info under $COMFY/.venv"; exit 1; }
            NEWER+=("$ck"); shift ;;
        --mode) MODE="$2"; shift 2 ;;
        *) echo "unknown argument: $1"; exit 1 ;;
    esac
done

[ -x "$COMFY/start.sh" ] || { echo "no start.sh at $COMFY"; exit 1; }

port_pid() { ss -lptnH "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1; }

# --- stop ------------------------------------------------------------------
# By PORT OWNER, not by pattern. `pgrep -f main.py | head -1` picks the `uv
# run` wrapper on this box and leaves the real server holding the socket,
# which is its own documented way of reading a stale /object_info.
pid="$(port_pid)"
if [ -n "$pid" ]; then
    echo "== stopping pid $pid (owner of :$PORT)"
    kill "$pid" 2>/dev/null
    for _ in $(seq 1 60); do [ -z "$(port_pid)" ] && break; sleep 1; done
    if [ -n "$(port_pid)" ]; then
        echo "== still holding :$PORT after 60s, sending SIGKILL"
        kill -9 "$(port_pid)" 2>/dev/null
        for _ in $(seq 1 20); do [ -z "$(port_pid)" ] && break; sleep 1; done
    fi
fi
# Wrappers that outlive the server would otherwise re-take the port.
pkill -f "main.py --output" 2>/dev/null
sleep 1
if [ -n "$(port_pid)" ]; then
    echo "FAIL: :$PORT is still held by pid $(port_pid); refusing to start a second server"
    exit 2
fi

# --- start -----------------------------------------------------------------
# `setsid` and a closed stdin, so the server does not die with the shell that
# launched it -- and, critically, so the launch is a SEPARATE command from the
# kill above rather than the second half of a `;` chain that never runs.
LOG="${LOG:-/tmp/comfy_restart_$(date +%H%M%S).log}"
echo "== starting ($MODE), log $LOG"
( cd "$COMFY" && setsid nohup ./start.sh "$MODE" > "$LOG" 2>&1 < /dev/null & )

for _ in $(seq 1 120); do
    curl -s -m 3 "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1 && break
    sleep 2
done
if ! curl -s -m 3 "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1; then
    echo "FAIL: no server on :$PORT after 240s. Last log lines:"
    tail -5 "$LOG" 2>/dev/null
    exit 3
fi

pid="$(port_pid)"
started="$(ps -o lstart= -p "$pid" 2>/dev/null)"
started_epoch="$(date -d "$started" +%s 2>/dev/null || echo 0)"
echo "== up: pid $pid, started $started"

# --- the assertion this script exists for ----------------------------------
stale=0
for path in ${NEWER+"${NEWER[@]}"}; do
    if [ ! -e "$path" ]; then
        echo "FAIL: --newer-than $path does not exist"; stale=1; continue
    fi
    mtime="$(stat -c %Y "$path")"
    if [ "$started_epoch" -le "$mtime" ]; then
        echo "FAIL: the server started $(date -d @"$started_epoch" '+%H:%M:%S') but"
        echo "      $path changed $(date -d @"$mtime" '+%H:%M:%S') -- it is NOT running that code."
        stale=1
    else
        echo "  ok  postdates $path"
    fi
done
[ "$stale" -eq 0 ] || exit 4

echo "== ready"
