"""Minimal stdlib HTTP client for a live ComfyUI (`:8188`) API server.

Shared across `bench/` drivers so scripts do not re-implement `/object_info`,
`/queue`, `/prompt`, `/history`, and error-body decoding by hand.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any

DEFAULT_COMFY_URL = "http://127.0.0.1:8188"


def _get_json(url: str, *, timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GET {url} failed with HTTP {exc.code}: {body}") from exc


def get_object_info(
    node_id: str | None = None,
    *,
    base_url: str = DEFAULT_COMFY_URL,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Return `/object_info` (all nodes) or `/object_info/{node_id}` JSON."""
    root = base_url.rstrip("/")
    url = f"{root}/object_info/{urllib.parse.quote(node_id, safe='')}" if node_id else f"{root}/object_info"
    return _get_json(url, timeout=timeout)


def get_queue(*, base_url: str = DEFAULT_COMFY_URL, timeout: float = 10.0) -> dict[str, Any]:
    """Return ComfyUI's live `/queue` JSON (`queue_running`, `queue_pending`)."""
    return _get_json(f"{base_url.rstrip('/')}/queue", timeout=timeout)


def has_input(
    node_id: str,
    input_name: str,
    *,
    base_url: str = DEFAULT_COMFY_URL,
    timeout: float = 10.0,
) -> bool:
    """True when `node_id` is registered on `base_url` and accepts `input_name`."""
    try:
        info = get_object_info(node_id, base_url=base_url, timeout=timeout)
    except RuntimeError:
        return False
    spec = info.get(node_id)
    if not isinstance(spec, dict):
        return False
    inp = spec.get("input") or {}
    return input_name in (inp.get("required") or {}) or input_name in (inp.get("optional") or {})


def submit_prompt(
    graph: dict[str, Any],
    *,
    client_id: str | None = None,
    base_url: str = DEFAULT_COMFY_URL,
    timeout: float = 30.0,
) -> str:
    """Queue an API graph on `/prompt`, surfacing `node_errors` clearly on failure."""
    url = f"{base_url.rstrip('/')}/prompt"
    body = json.dumps({"prompt": graph, "client_id": client_id or uuid.uuid4().hex}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_text = exc.read().decode("utf-8", errors="replace")
        try:
            err_json = json.loads(err_text)
        except Exception:
            err_json = None
        if isinstance(err_json, dict) and (err_json.get("error") or err_json.get("node_errors")):
            raise RuntimeError(
                f"ComfyUI /prompt rejected graph (HTTP {exc.code}): "
                f"error={err_json.get('error')} node_errors={err_json.get('node_errors')}"
            ) from exc
        raise RuntimeError(f"ComfyUI /prompt failed with HTTP {exc.code}: {err_text}") from exc
    if data.get("node_errors"):
        raise RuntimeError(f"ComfyUI /prompt reported node_errors: {data['node_errors']}")
    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI /prompt response missing prompt_id: {data}")
    return str(prompt_id)


def _format_execution_error(prompt_id: str, status: dict[str, Any]) -> str:
    for msg in status.get("messages") or ():
        if isinstance(msg, (list, tuple)) and len(msg) >= 2 and msg[0] == "execution_error":
            detail = msg[1] if isinstance(msg[1], dict) else {}
            return (
                f"prompt {prompt_id} failed on node {detail.get('node_id')} "
                f"({detail.get('node_type')}): {detail.get('exception_message')}"
            )
    return f"prompt {prompt_id} ended with status={status!r}"


def execution_duration_s(history_entry: dict[str, Any]) -> float | None:
    """Return GPU execution duration in seconds from `/history` timestamps, or `None` if absent."""
    msgs = (history_entry.get("status") or {}).get("messages") or ()
    ts_start: int | float | None = None
    ts_end: int | float | None = None
    for msg in msgs:
        if not isinstance(msg, (list, tuple)) or len(msg) < 2 or not isinstance(msg[1], dict):
            continue
        ts = msg[1].get("timestamp")
        if not isinstance(ts, (int, float)):
            continue
        if msg[0] == "execution_start":
            ts_start = ts
        elif msg[0] == "execution_success":
            ts_end = ts
    if ts_start is None or ts_end is None:
        return None
    return round(max(0.0, float(ts_end) - float(ts_start)) / 1000.0, 2)


def _queue_contains(items: Any, prompt_id: str) -> bool:
    return any(isinstance(it, (list, tuple)) and len(it) >= 2 and str(it[1]) == prompt_id for it in (items or ()))


def wait_for_prompt(
    prompt_id: str,
    *,
    base_url: str = DEFAULT_COMFY_URL,
    poll_interval: float = 1.5,
    timeout: float = 1800.0,
    queue_timeout: float | None = None,
) -> dict[str, Any]:
    """Poll `/history/{prompt_id}` until completion, starting `timeout` once execution begins."""
    root = base_url.rstrip("/")
    url = f"{root}/history/{urllib.parse.quote(prompt_id, safe='')}"
    submitted_at = time.monotonic()
    running_since: float | None = None
    while True:
        hist = _get_json(url, timeout=15.0)
        entry = hist.get(prompt_id)
        if isinstance(entry, dict):
            status = entry.get("status") or {}
            status_str = status.get("status_str")
            if status_str == "error":
                raise RuntimeError(_format_execution_error(prompt_id, status))
            if status_str == "success" or (status.get("completed") and entry.get("outputs")):
                return entry
        now = time.monotonic()
        q = get_queue(base_url=base_url, timeout=15.0)
        in_running = _queue_contains(q.get("queue_running"), prompt_id)
        in_pending = _queue_contains(q.get("queue_pending"), prompt_id)
        if running_since is None and (in_running or not in_pending):
            running_since = now
        if running_since is None:
            if queue_timeout is not None and (now - submitted_at) >= float(queue_timeout):
                raise TimeoutError(f"Timed out waiting {queue_timeout:.1f}s in queue for ComfyUI prompt {prompt_id}")
        elif (now - running_since) >= float(timeout):
            raise TimeoutError(f"Timed out waiting {timeout:.1f}s executing ComfyUI prompt {prompt_id}")
        time.sleep(float(poll_interval))


def collect_saved_files(history_entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten saved file entries from `/history/{prompt_id}['outputs']` across nodes."""
    items: list[dict[str, Any]] = []
    outputs = history_entry.get("outputs") or {}
    for node_id, node_out in outputs.items():
        if not isinstance(node_out, dict):
            continue
        for kind in ("gifs", "videos", "images", "audio"):
            for entry in node_out.get(kind) or ():
                if isinstance(entry, dict):
                    items.append({**entry, "node_id": str(node_id), "kind": kind})
    return items
