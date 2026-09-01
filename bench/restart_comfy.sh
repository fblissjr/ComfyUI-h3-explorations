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
#   bench/restart_comfy.sh --force               # restart even if the running
#                                                # server is ARMED for capture
#
# Exit codes: 0 ok, 1 usage/setup, 2 port never freed, 3 never came up,
#             4 came up but is OLDER than something it must postdate,
#             5 the running server is ARMED and --force was not given.
#
# ## Why exit 5 exists
#
# **The server process is the shared resource, not the GPU.** H3_CAPTURE and
# H3_PDD_OBSERVE live in the ENVIRONMENT of the process that starts the server,
# so a restart silently disarms whatever another session configured -- and
# nothing in /queue, nvidia-smi or /system_stats shows that a server is armed.
# Checking that the queue is empty and the card is free feels like diligence
# and answers the wrong question: an idle server can still be a configured one.
#
# On 2026-08-30 that cost a peer session three renders in half an hour, one of
# them a 345-frame job mid-flight. The worst of the three was a restart done to
# be TIDY -- clearing this session own stale capture spec before handing the
# server over, which cleared the peer spec in the same action.
#
# So the port owner environment is read from /proc/<pid>/environ and a restart
# that would disarm somebody REFUSES, naming what is armed. Before this, that
# operation succeeded and emitted nothing, which is this repo most-repeated
# failure shape.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFY="${COMFY:-$REPO/../..}"
PORT="${PORT:-8188}"
MODE="${MODE:-default}"
FORCE="${FORCE:-0}"
NEWER=()

# Environment keys that make a running server a CONFIGURED resource rather than
# a replaceable one. Add to this list; do not branch on it elsewhere.
#
# **H3_QUANT_OBSERVE added 2026-08-31, and it had been missing since the
# observer landed the same day.** `dit_observe.py` gates the Tier 1 quant
# observer on it (`os.environ.get("H3_QUANT_OBSERVE")`), so a server armed for
# that capture was NOT protected by the guard that exists to stop exactly this,
# and this script would have killed it without refusing. The commit that added
# the observer is `31b0d4a`, whose own message is "find that both observers were
# armed by default" -- the arming was thought about and the guard was not.
#
# The general form, since this list will grow again: **a new arming key is half
# the change.** Adding the env gate protects the capture from firing by
# accident; adding it here protects the capture from being disarmed by someone
# who cannot see it. Grep for the key in the module that reads it, not in the
# docs that describe it -- `docs/research/pdd/tier1_gate.md` item 2 asked for
# exactly this and it still had not been done.
ARMING_KEYS="H3_CAPTURE H3_PDD_OBSERVE H3_QUANT_OBSERVE H3_SOL_OBSERVE"

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
        --force) FORCE=1; shift ;;
        *) echo "unknown argument: $1"; exit 1 ;;
    esac
done

[ -x "$COMFY/start.sh" ] || { echo "no start.sh at $COMFY"; exit 1; }

port_pid() { ss -lptnH "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1; }

# --- refuse to disarm somebody else ---------------------------------------
# Read from the PORT OWNER's own environment, not from this shell's: the
# question is what the RUNNING server was started with, and this shell may
# never have had it.
armed_pid="$(port_pid)"
if [ -n "$armed_pid" ] && [ -r "/proc/$armed_pid/environ" ]; then
    armed=""
    for key in $ARMING_KEYS; do
        val="$(tr '\0' '\n' < "/proc/$armed_pid/environ" 2>/dev/null | grep "^$key=" | head -1)"
        if [ -n "$val" ]; then
            armed="$armed
  $val"
        fi
    done
    if [ -n "$armed" ]; then
        if [ "$FORCE" = "1" ]; then
            echo "== WARNING: pid $armed_pid is ARMED and --force was given; restarting anyway:$armed"
        else
            echo "REFUSING: the server on :$PORT (pid $armed_pid) is ARMED for capture:$armed"
            echo
            echo "Restarting would silently disarm it. That environment belongs to whichever"
            echo "session started the server, which may not be this one -- an empty queue and"
            echo "a free card do not tell you a server is unconfigured."
            echo
            echo "Ask the peer that armed it (ListAgents shows live sessions), or pass --force"
            echo "if you know the capture is finished or is yours."
            exit 5
        fi
    fi
fi

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
# `-A` (--ignore-ancestors, procps-ng 4.x) keeps the match off THIS script's
# own ancestors: `-f` matches any argv containing the pattern, and on
# 2026-09-01 that included the shell that invoked this script, whose command
# line quoted the literal in a trailing pgrep -- the caller died with a
# signal exit while the restart it started completed on its own.
pkill -A -f "main.py --output" 2>/dev/null
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
