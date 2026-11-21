#!/usr/bin/env bash
# Launcher for Control Vector Studio (`bench/control_vectors.py ui`).
# Ports default to CV_STUDIO_PORT=8197 (loopback) and CV_STUDIO_HTTPS_PORT=8196
# (Tailscale HTTPS front), configurable via `h3.env` or `--port` / `--https-port`.
# Never touches ComfyUI :8188, H3-Image Studio :8192/:8193, or Spatial Director :8194/:8195.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMFY_DIR="${H3_COMFY_DIR:-$(cd "$REPO_DIR/../.." && pwd)}"

if [ -f "$REPO_DIR/h3.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$REPO_DIR/h3.env"
    set +a
fi

HOST="${CV_STUDIO_HOST:-127.0.0.1}"
PORT="${CV_STUDIO_PORT:-8197}"
HTTPS_PORT="${CV_STUDIO_HTTPS_PORT:-8196}"
MODE="${CV_STUDIO_MODE:-https}"
TOKEN_OVERRIDE="${CV_STUDIO_TOKEN:-}"
ROTATE_TOKEN=0
STATE_DIR="${CV_STUDIO_STATE_DIR:-$REPO_DIR/data/control_vector_studio}"

PYTHON_BIN="${H3_PYTHON:-$COMFY_DIR/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3 || true)"
fi

PID_FILE="$STATE_DIR/cv_studio.pid"
LOG_FILE="$STATE_DIR/cv_studio.log"
MODE_FILE="$STATE_DIR/cv_studio.mode"
TOKEN_FILE="$STATE_DIR/cv_studio.token"

usage() {
    cat <<EOF
Usage: $(basename "$0") {start|stop|restart|status|url|logs|help} [--host H] [--port P] [--https-port P] [--http] [--https] [--token TOK] [--rotate-token] [--state-dir DIR]
EOF
}

cmd="${1:-status}"
if [[ "$cmd" != -* ]]; then
    shift || true
else
    cmd="start"
fi

while [ "$#" -gt 0 ]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --port|-p) PORT="$2"; shift 2 ;;
        --https-port) HTTPS_PORT="$2"; shift 2 ;;
        --http) MODE="http"; shift ;;
        --https) MODE="https"; shift ;;
        --token) TOKEN_OVERRIDE="$2"; shift 2 ;;
        --rotate-token) ROTATE_TOKEN=1; shift ;;
        --state-dir)
            STATE_DIR="$2"
            PID_FILE="$STATE_DIR/cv_studio.pid"
            LOG_FILE="$STATE_DIR/cv_studio.log"
            MODE_FILE="$STATE_DIR/cv_studio.mode"
            TOKEN_FILE="$STATE_DIR/cv_studio.token"
            shift 2
            ;;
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

front_on() {
    local port="${1:-$HTTPS_PORT}"
    command -v tailscale >/dev/null 2>&1 || return 1
    tailscale serve status --json 2>/dev/null \
        | "$PYTHON_BIN" -c 'import json, sys; sys.exit(0 if sys.argv[1] in (json.load(sys.stdin).get("TCP") or {}) else 1)' "$port" 2>/dev/null
}

ensure_token() {
    mkdir -p "$STATE_DIR"
    if [ -n "$TOKEN_OVERRIDE" ]; then
        (umask 077 && printf '%s\n' "$TOKEN_OVERRIDE" >"$TOKEN_FILE")
    elif [ "$ROTATE_TOKEN" -eq 1 ] || [ ! -s "$TOKEN_FILE" ]; then
        local generated
        generated="$("$PYTHON_BIN" -c 'import secrets; print(secrets.token_urlsafe(24))')"
        (umask 077 && printf '%s\n' "$generated" >"$TOKEN_FILE")
    fi
    chmod 600 "$TOKEN_FILE" 2>/dev/null || true
    cat "$TOKEN_FILE"
}

read_token() {
    if [ -s "$TOKEN_FILE" ]; then
        cat "$TOKEN_FILE"
    fi
}

probe_ready() {
    local code
    code="$(curl -sS -o /dev/null -w '%{http_code}' "http://$HOST:$PORT/" 2>/dev/null || true)"
    [ "$code" = "403" ] || [ "$code" = "200" ]
}

do_start() {
    mkdir -p "$STATE_DIR"
    if [ ! -x "$PYTHON_BIN" ]; then
        echo "No Python interpreter found at $PYTHON_BIN" >&2
        exit 1
    fi
    if is_running; then
        local active_port="$PORT" active_token=""
        if [ -f "$MODE_FILE" ]; then
            read -r _ active_port _ <"$MODE_FILE" || true
        fi
        active_token="$(read_token)"
        echo "Control Vector Studio already running (pid $(cat "$PID_FILE")) at http://$HOST:${active_port:-$PORT}/?token=${active_token}"
        return 0
    fi

    local token
    token="$(ensure_token)"

    cd "$REPO_DIR"
    (
        umask 077
        if command -v setsid >/dev/null 2>&1; then
            setsid "$PYTHON_BIN" "$REPO_DIR/bench/control_vectors.py" ui \
                --host "$HOST" \
                --port "$PORT" \
                --no-browser \
                --token "$token" \
                </dev/null >"$LOG_FILE" 2>&1 &
        else
            nohup "$PYTHON_BIN" "$REPO_DIR/bench/control_vectors.py" ui \
                --host "$HOST" \
                --port "$PORT" \
                --no-browser \
                --token "$token" \
                </dev/null >"$LOG_FILE" 2>&1 &
        fi
        echo "$!" >"$PID_FILE"
    )
    local pid
    pid="$(cat "$PID_FILE")"
    printf '%s %s %s\n' "$MODE" "$PORT" "$HTTPS_PORT" >"$MODE_FILE"

    local i
    for i in $(seq 1 40); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "Control Vector Studio exited unexpectedly. Log:" >&2
            tail -n 40 "$LOG_FILE" >&2 || true
            rm -f "$PID_FILE"
            exit 1
        fi
        if probe_ready; then
            break
        fi
        sleep 0.25
    done

    if ! probe_ready; then
        echo "Timed out waiting for Control Vector Studio on http://$HOST:$PORT/" >&2
        tail -n 40 "$LOG_FILE" >&2 || true
        exit 1
    fi

    echo "Control Vector Studio listening on http://$HOST:$PORT/?token=$token (pid $pid)"
    if [ "$MODE" = "https" ] && command -v tailscale >/dev/null 2>&1; then
        local ts_dns
        ts_dns="$(tailscale_dns)"
        if [ -n "$ts_dns" ]; then
            if front_on "$HTTPS_PORT"; then
                echo "Tailscale HTTPS active at https://$ts_dns:$HTTPS_PORT/?token=$token"
            elif tailscale serve --bg --https="$HTTPS_PORT" "http://$HOST:$PORT" >/dev/null 2>&1 \
                || sudo -n tailscale serve --bg --https="$HTTPS_PORT" "http://$HOST:$PORT" >/dev/null 2>&1; then
                echo "Tailscale HTTPS active at https://$ts_dns:$HTTPS_PORT/?token=$token"
            elif [ -t 0 ] && [ -z "${ANTIGRAVITY_CLI:-}" ] && [ "${TERM:-dumb}" != "dumb" ]; then
                echo "Putting HTTPS in front on :$HTTPS_PORT with tailscale serve (sudo):"
                if sudo tailscale serve --bg --https="$HTTPS_PORT" "http://$HOST:$PORT" >/dev/null; then
                    echo "Tailscale HTTPS active at https://$ts_dns:$HTTPS_PORT/?token=$token"
                fi
            else
                echo "Note: To enable Tailscale HTTPS on :$HTTPS_PORT, run once in your terminal:" >&2
                echo "  sudo tailscale serve --bg --https=$HTTPS_PORT http://$HOST:$PORT" >&2
            fi
        fi
    fi
}

do_stop() {
    local active_mode="$MODE"
    local active_https_port="$HTTPS_PORT"
    if [ -f "$MODE_FILE" ]; then
        read -r active_mode _ active_https_port <"$MODE_FILE" || true
    fi
    if [ "${active_mode:-$MODE}" = "https" ] && command -v tailscale >/dev/null 2>&1; then
        tailscale serve --https="${active_https_port:-$HTTPS_PORT}" off >/dev/null 2>&1 \
            || sudo -n tailscale serve --https="${active_https_port:-$HTTPS_PORT}" off >/dev/null 2>&1 \
            || true
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
        rm -f "$PID_FILE" "$MODE_FILE"
        echo "Stopped Control Vector Studio (pid $pid)"
    else
        rm -f "$PID_FILE" "$MODE_FILE"
        echo "Control Vector Studio is not running"
    fi
}

do_status() {
    if is_running; then
        local active_mode="$MODE" active_port="$PORT" active_https="$HTTPS_PORT" token=""
        if [ -f "$MODE_FILE" ]; then
            read -r active_mode active_port active_https <"$MODE_FILE" || true
        fi
        token="$(read_token)"
        echo "Control Vector Studio: RUNNING (pid $(cat "$PID_FILE"), http://$HOST:${active_port:-$PORT}/?token=$token)"
        if [ "${active_mode:-$MODE}" = "https" ] && command -v tailscale >/dev/null 2>&1; then
            local ts_dns
            ts_dns="$(tailscale_dns)"
            if [ -n "$ts_dns" ]; then
                if front_on "${active_https:-$HTTPS_PORT}"; then
                    echo "  Tailscale HTTPS: https://$ts_dns:${active_https:-$HTTPS_PORT}/?token=$token"
                else
                    echo "  Tailscale HTTPS: not configured on :${active_https:-$HTTPS_PORT} (run: sudo tailscale serve --bg --https=${active_https:-$HTTPS_PORT} http://$HOST:${active_port:-$PORT})"
                fi
            fi
        fi
    else
        echo "Control Vector Studio: STOPPED"
        return 1
    fi
}

do_url() {
    if ! is_running; then
        echo "Control Vector Studio is not running. Start it with: comfy app control-vectors start" >&2
        return 1
    fi
    local active_mode="$MODE" active_port="$PORT" active_https="$HTTPS_PORT" token=""
    if [ -f "$MODE_FILE" ]; then
        read -r active_mode active_port active_https <"$MODE_FILE" || true
    fi
    token="$(read_token)"
    if [ "${active_mode:-$MODE}" = "https" ] && command -v tailscale >/dev/null 2>&1; then
        local ts_dns
        ts_dns="$(tailscale_dns)"
        if [ -n "$ts_dns" ]; then
            echo "https://$ts_dns:${active_https:-$HTTPS_PORT}/?token=$token"
            return 0
        fi
    fi
    echo "http://$HOST:${active_port:-$PORT}/?token=$token"
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
    url) do_url ;;
    logs) do_logs ;;
    help) usage ;;
    *) echo "Unknown command: $cmd" >&2; usage >&2; exit 2 ;;
esac
