#!/usr/bin/env bash
# Launcher for Spatial World Director (`director/server.py`).
# Local HTTP defaults to 127.0.0.1:8195; optional Tailscale HTTPS terminates on :8194.
# Never touches ComfyUI :8188 or app/ :8193.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COMFY_DIR="${H3_COMFY_DIR:-$(cd "$REPO_DIR/../.." && pwd)}"

HOST="${DIRECTOR_HOST:-127.0.0.1}"
PORT="${DIRECTOR_PORT:-8195}"
HTTPS_PORT="${DIRECTOR_HTTPS_PORT:-8194}"
COMFY_URL="${H3_COMFY_URL:-http://127.0.0.1:8188}"
STATE_DIR="${DIRECTOR_STATE_DIR:-$REPO_DIR/data/spatial_director_app}"

if [ -f "$REPO_DIR/h3.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$REPO_DIR/h3.env"
    set +a
fi

if [ -z "${H3_OUTPUT_DIR:-}" ] && [ -f "$COMFY_DIR/start.sh" ]; then
    parsed_out="$(sed -nE 's/.*--output-directory[= ]+("[^"]+"|'\''[^'\'']+'\''|[^ ]+).*/\1/p' "$COMFY_DIR/start.sh" | head -n 1 | tr -d "\"'")"
    if [ -n "$parsed_out" ]; then
        export H3_OUTPUT_DIR="$parsed_out"
    fi
fi

PYTHON_BIN="${H3_PYTHON:-$COMFY_DIR/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3 || true)"
fi

PID_FILE="$STATE_DIR/director.pid"
LOG_FILE="$STATE_DIR/director.log"

usage() {
    cat <<EOF
Usage: $(basename "$0") {start|stop|restart|status|logs} [--host H] [--port P] [--https-port P] [--comfy-url URL] [--state-dir DIR]
EOF
}

cmd="${1:-start}"
if [[ "$cmd" != -* ]]; then
    shift || true
else
    cmd="start"
fi

while [ "$#" -gt 0 ]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --https-port) HTTPS_PORT="$2"; shift 2 ;;
        --comfy-url|--comfy) COMFY_URL="$2"; shift 2 ;;
        --state-dir) STATE_DIR="$2"; PID_FILE="$STATE_DIR/director.pid"; LOG_FILE="$STATE_DIR/director.log"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

is_running() {
    [ -f "$PID_FILE" ] || return 1
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

tailscale_dns() {
    command -v tailscale >/dev/null 2>&1 || return 0
    tailscale status --json 2>/dev/null | "$PYTHON_BIN" -c '
import json, sys
try:
    d = json.load(sys.stdin)
    print((d.get("Self", {}).get("DNSName") or "").strip().rstrip("."))
except Exception:
    pass
' 2>/dev/null || true
}

do_start() {
    mkdir -p "$STATE_DIR"
    if is_running; then
        echo "Spatial World Director already running (pid $(cat "$PID_FILE")) at http://$HOST:$PORT/"
        return 0
    fi
    if [ ! -x "$PYTHON_BIN" ]; then
        echo "No Python interpreter found at $PYTHON_BIN" >&2
        exit 1
    fi

    cd "$REPO_DIR"
    nohup "$PYTHON_BIN" "$SCRIPT_DIR/server.py" \
        --host "$HOST" \
        --port "$PORT" \
        --comfy-url "$COMFY_URL" \
        --state-dir "$STATE_DIR" \
        >"$LOG_FILE" 2>&1 &
    local pid=$!
    echo "$pid" >"$PID_FILE"

    local i
    for i in $(seq 1 40); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "Spatial World Director exited unexpectedly. Log:" >&2
            tail -n 40 "$LOG_FILE" >&2 || true
            rm -f "$PID_FILE"
            exit 1
        fi
        if curl -fsS "http://$HOST:$PORT/api/schema" >/dev/null 2>&1; then
            break
        fi
        sleep 0.25
    done

    if ! curl -fsS "http://$HOST:$PORT/api/schema" >/dev/null 2>&1; then
        echo "Timed out waiting for http://$HOST:$PORT/api/schema" >&2
        exit 1
    fi

    echo "Spatial World Director listening on http://$HOST:$PORT/ (pid $pid, comfy=$COMFY_URL)"
    if command -v tailscale >/dev/null 2>&1; then
        local ts_dns
        ts_dns="$(tailscale_dns)"
        if [ -n "$ts_dns" ]; then
            if tailscale serve --bg --https="$HTTPS_PORT" "http://$HOST:$PORT" >/dev/null 2>&1; then
                echo "Tailscale HTTPS active at https://$ts_dns:$HTTPS_PORT/"
            fi
        fi
    fi
}

do_stop() {
    if command -v tailscale >/dev/null 2>&1; then
        tailscale serve --https="$HTTPS_PORT" off >/dev/null 2>&1 || true
    fi
    if is_running; then
        local pid
        pid="$(cat "$PID_FILE")"
        kill "$pid" 2>/dev/null || true
        for _ in $(seq 1 20); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.2
        done
        kill -9 "$pid" 2>/dev/null || true
        rm -f "$PID_FILE"
        echo "Stopped Spatial World Director (pid $pid)"
    else
        rm -f "$PID_FILE"
        echo "Spatial World Director is not running"
    fi
}

do_status() {
    if is_running; then
        echo "running pid=$(cat "$PID_FILE") url=http://$HOST:$PORT/ comfy=$COMFY_URL state=$STATE_DIR"
    else
        echo "stopped"
        return 1
    fi
}

do_logs() {
    mkdir -p "$STATE_DIR"
    touch "$LOG_FILE"
    tail -n 80 -f "$LOG_FILE"
}

case "$cmd" in
    start) do_start ;;
    stop) do_stop ;;
    restart) do_stop; do_start ;;
    status) do_status ;;
    logs) do_logs ;;
    *) echo "Unknown command: $cmd" >&2; usage >&2; exit 2 ;;
esac
