"""Shared `heylookitsanllm` (`POST /v1/messages`) HTTP/SSE client and capability-gating helpers.

Used by both `MiniMaxH3ImagePromptExpander` (`h3_image_nodes.py`) and
`MiniMaxH3AutoEditPrompt` (`auto_edit_prompt.py`):
- Resolves base URL without leaking host-specific LAN IPs into tracked graphs
  (`explicit > H3_HEYLOOK_URL env / gitignored h3.env > HEYLOOK_DEFAULT_BASE_URL`).
- Discovers model IDs at runtime from `GET /v1/models` (gated on `capabilities`,
  never `modalities`, preferring resident models from `GET /v1/system/metrics`).
- Downscales reference images client-side (`<= 2048 px` longest edge, Lanczos,
  EXIF-aware, PNG kept as PNG and photos encoded as quality-85 RGB JPEG).
- Capability-gates `thinking`, `thinking_budget`, `reasoning_effort`, and
  `engine.decoding.request_fields`.
- Retries HTTP 503 (`model_overloaded`) with `Retry-After` backoff, sends a fresh
  UUID `X-Request-ID` per attempt, cancels via `DELETE /v1/requests/{id}` on abort,
  separates `thinking_delta` from `text_delta`, and reports `generation_duration_ms`
  throughput defensively.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import io as _pyio
import json
import os
import sys
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, Callable

import httpx
import numpy as np
from PIL import Image, ImageOps

_HERE = Path(__file__).resolve().parent
if str(_HERE / "workflows") not in sys.path:
    sys.path.insert(0, str(_HERE / "workflows"))
from h3_config import HEYLOOK_DEFAULT_BASE_URL  # noqa: E402
import h3_env  # noqa: E402

HEYLOOK_IMAGE_MAX_EDGE = 2048
HEYLOOK_JPEG_QUALITY = 85
THINKING_MODES: tuple[str, ...] = ("model default", "on", "off")
SAMPLER_MODES: tuple[str, ...] = ("heylook / preset default", "custom")


class HeylookOverloadedError(RuntimeError):
    """HTTP 503 `model_overloaded`: queue signal on a single-user local server."""

    def __init__(self, message: str = "model_overloaded", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class HeylookStreamResult:
    text: str = ""
    thinking: str = ""
    model: str = ""
    request_id: str = ""
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: dict = field(default_factory=dict)
    performance: dict = field(default_factory=dict)


def _extract_inline_thinking(raw_text: str) -> str:
    """Extract leading `<think>...</think>` blocks when an engine emits tags inside `text`."""
    parts: list[str] = []
    s = raw_text.strip()
    while s.startswith("<think>"):
        end = s.find("</think>")
        if end < 0:
            break
        chunk = s[len("<think>"):end].strip()
        if chunk:
            parts.append(chunk)
        s = s[end + len("</think>"):].strip()
    if not parts and "</think>" in s and "<think>" not in s.split("</think>", 1)[0]:
        chunk = s.split("</think>", 1)[0].strip()
        if chunk:
            parts.append(chunk)
    return "\n\n".join(parts)


def normalize_heylook_base_url(base_url: str) -> str:
    raw = (base_url or "").strip().rstrip("/")
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(
            f"invalid heylook base_url {base_url!r}; expected http://<host>:<port> or https://<host>"
        )
    return raw


def resolve_heylook_base_url(base_url: str = "") -> str:
    """Resolve `base_url` with priority: explicit widget > `H3_HEYLOOK_URL` (env or `h3.env`) > default."""
    explicit = (base_url or "").strip()
    if explicit:
        return normalize_heylook_base_url(explicit)
    h3_env.load()
    env_url = (os.environ.get("H3_HEYLOOK_URL") or "").strip()
    if env_url:
        return normalize_heylook_base_url(env_url)
    return HEYLOOK_DEFAULT_BASE_URL


def prepare_heylook_image(path_or_bytes: bytes | str | Path) -> tuple[str, str]:
    """Downscale a file/bytes image for `heylook` `/v1/messages` (`(raw_base64, media_type)`).

    Matches `heylook`'s frontend contract: read `.format` BEFORE `exif_transpose`
    (which drops `.format` to `None`), apply EXIF orientation, bound the longest
    edge at 2048 px with Lanczos without enlarging, keep PNG as PNG, and encode
    everything else as quality-85 RGB JPEG.
    """
    src = _pyio.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    with Image.open(src) as opened:
        keep_png = (opened.format or "").upper() == "PNG"
        img = ImageOps.exif_transpose(opened)
    img.thumbnail((HEYLOOK_IMAGE_MAX_EDGE, HEYLOOK_IMAGE_MAX_EDGE), Image.Resampling.LANCZOS)
    buf = _pyio.BytesIO()
    if keep_png:
        img.save(buf, format="PNG", optimize=True)
        media_type = "image/png"
    else:
        img.convert("RGB").save(buf, format="JPEG", quality=HEYLOOK_JPEG_QUALITY)
        media_type = "image/jpeg"
    return base64.b64encode(buf.getvalue()).decode("ascii"), media_type


def prepare_heylook_tensor(img_tensor: Any, *, index: int = 1) -> tuple[str, str]:
    """Downscale a ComfyUI `[1, H, W, C]` float image tensor to `(raw_base64, media_type)`."""
    shape = tuple(int(d) for d in img_tensor.shape)
    if len(shape) != 4 or shape[0] != 1 or shape[3] not in (3, 4):
        raise ValueError(
            f"image_{index} must be a single still image with shape [1, H, W, 3|4], got {shape}"
        )
    arr = img_tensor[0].detach().cpu().numpy() if hasattr(img_tensor, "detach") else np.asarray(img_tensor[0])
    u8 = np.clip(np.rint(arr * 255.0), 0.0, 255.0).astype(np.uint8)
    has_alpha = u8.shape[2] == 4 and bool(np.any(u8[:, :, 3] < 255))
    pil_mode = "RGBA" if has_alpha else "RGB"
    img = Image.fromarray(u8 if has_alpha else u8[:, :, :3], mode=pil_mode)
    img.thumbnail((HEYLOOK_IMAGE_MAX_EDGE, HEYLOOK_IMAGE_MAX_EDGE), Image.Resampling.LANCZOS)
    buf = _pyio.BytesIO()
    if has_alpha:
        img.save(buf, format="PNG", optimize=True)
        media_type = "image/png"
    else:
        img.save(buf, format="JPEG", quality=HEYLOOK_JPEG_QUALITY)
        media_type = "image/jpeg"
    return base64.b64encode(buf.getvalue()).decode("ascii"), media_type


def _fact_value(node: Any, *keys: str) -> Any:
    cur = node
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur.get("value") if isinstance(cur, dict) else None


def _http_error(r: httpx.Response, base: str, path: str) -> Exception:
    if r.status_code == 503:
        try:
            retry = float(r.headers.get("Retry-After", ""))
        except ValueError:
            retry = None
        msg = "model_overloaded"
        try:
            err_msg = (r.json().get("error") or {}).get("message")
            if err_msg:
                msg = f"model_overloaded: {err_msg}"
        except Exception:
            pass
        return HeylookOverloadedError(msg, retry_after=retry)
    if r.status_code == 403:
        return PermissionError(
            f"{base}{path} answered HTTP 403 (heylook Host check / DNS-rebinding guard). "
            "Address the server by IP or 127.0.0.1, or add the hostname to `allowed_hosts` in heylook.toml."
        )
    if r.status_code == 401:
        return PermissionError(
            f"{base}{path} answered HTTP 401: heylook has no inference key, so a proxy in front of it rejected the request."
        )
    try:
        payload = r.json()
    except Exception:
        detail = r.text[:240]
    else:
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if detail is None and isinstance(payload, dict):
            detail = (payload.get("error") or {}).get("message")
        if detail is None:
            detail = str(payload)[:240]
    if r.status_code in (400, 422):
        return ValueError(f"heylook HTTP {r.status_code} on {path}: {detail}")
    return RuntimeError(f"heylook HTTP {r.status_code} on {path}: {detail}")


def discover_heylook_model(
    client: httpx.Client,
    base: str,
    requested_model: str = "",
    *,
    need: set[str] | frozenset[str] = frozenset({"chat"}),
) -> dict:
    """Resolve and capability-gate a `/v1/models` row at runtime (`capabilities`, never `modalities`)."""
    try:
        r = client.get(f"{base}/v1/models", headers={"Accept": "application/json"}, timeout=10.0)
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"cannot reach heylook at {base}/v1/models ({type(exc).__name__}: {exc}). "
            "Check that heylook is running and base_url points to its host:port."
        ) from exc
    if r.status_code >= 400:
        raise _http_error(r, base, "/v1/models")
    try:
        payload = r.json()
    except ValueError as exc:
        raise RuntimeError(f"{base}/v1/models returned non-JSON; is {base} a heylook server?") from exc
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError(f"{base}/v1/models returned no 'data' list; is {base} a heylook server?")
    usable = [row for row in rows if isinstance(row, dict) and row.get("id")]
    req_set = set(need)
    fit = [row for row in usable if req_set <= set(row.get("capabilities") or [])]

    wanted = (requested_model or "").strip()
    if wanted and wanted.lower() != "auto":
        chosen = next((row for row in usable if row["id"] == wanted), None)
        if chosen is None:
            fit_ids = [row["id"] for row in fit]
            raise ValueError(
                f"model {wanted!r} is not served by {base}/v1/models; "
                f"models serving {sorted(req_set)}: {fit_ids}"
            )
        caps = set(chosen.get("capabilities") or [])
        missing = sorted(req_set - caps)
        if missing:
            fit_ids = [row["id"] for row in fit]
            raise ValueError(
                f"model {wanted!r} lacks required capabilities {missing} (has {sorted(caps)}); "
                f"models serving {sorted(req_set)}: {fit_ids}"
            )
        return chosen

    if not fit:
        roster = [(row["id"], row.get("capabilities")) for row in usable]
        raise LookupError(f"no model at {base} serves capabilities {sorted(req_set)}; roster: {roster}")

    resident: set[str] = set()
    try:
        mr = client.get(f"{base}/v1/system/metrics", headers={"Accept": "application/json"}, timeout=5.0)
        if mr.status_code == 200:
            models_map = mr.json().get("models")
            if isinstance(models_map, dict):
                resident = set(models_map.keys())
    except Exception:
        pass
    return next((row for row in fit if row["id"] in resident), fit[0])


def expand_heylook_preset(
    client: httpx.Client,
    base: str,
    preset_id: str,
    model_row: dict,
) -> dict:
    """Expand a `/v1/presets` entry client-side with capability & depth filtering."""
    r = client.get(f"{base}/v1/presets", headers={"Accept": "application/json"}, timeout=10.0)
    if r.status_code >= 400:
        raise _http_error(r, base, "/v1/presets")
    presets = [p for p in (r.json().get("presets") or []) if isinstance(p, dict)]
    avail = [p.get("name") or p.get("id") for p in presets if p.get("name") or p.get("id")]

    wanted = (preset_id or "").strip()
    if not wanted:
        raise ValueError(
            f"'preset' is blank; choose a preset from {base}/v1/presets: {avail}"
        )
    preset = next((p for p in presets if p.get("id") == wanted), None)
    if preset is None:
        preset = next((p for p in presets if p.get("name") == wanted), None)
    if preset is None:
        lowered = wanted.lower()
        preset = next((p for p in presets if str(p.get("name") or "").lower() == lowered), None)
    if preset is None:
        raise ValueError(f"heylook preset {wanted!r} not found at {base}/v1/presets; available: {avail}")

    params = dict(preset.get("params") or {})
    body: dict = {}
    if preset.get("system_prompt"):
        body["system"] = preset["system_prompt"]

    caps = set(model_row.get("capabilities") or [])
    switch = params.pop("enable_thinking", None)
    budget = params.pop("thinking_budget_tokens", None)
    if budget is not None and "thinking_budget" in caps:
        body["thinking"] = {"budget_tokens": budget}
        if switch is not None:
            body["thinking"]["type"] = "enabled" if switch else "disabled"
    elif switch is not None and "thinking" in caps:
        body["thinking"] = bool(switch)

    effort = params.pop("reasoning_effort", None)
    depth = ((model_row.get("engine") or {}).get("thinking") or {}).get("depth") or {}
    if effort is not None and (
        depth.get("unknown") == "verbatim"
        or effort in (depth.get("values") or [])
        or effort in (depth.get("aliases") or {})
    ):
        body["reasoning_effort"] = effort

    body.update(params)
    return body


_EXTEND_TOKENS: tuple[tuple[str, str], ...] = (
    ("@extend:prepend", "prepend"),
    ("@extend:append", "append"),
    ("<!-- h3_extend:prepend -->", "prepend"),
    ("<!-- h3_extend:append -->", "append"),
)


def parse_extend_directive(system_prompt: str) -> tuple[str | None, str]:
    """Parse an optional first-line `@extend:prepend|append` overlay directive from a preset prompt.

    Returns `(mode, body)` where `mode` is `'prepend'`, `'append'`, or `None` when the preset
    is a normal standalone full-override `system_prompt`.
    """
    norm = (system_prompt or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = norm.split("\n")
    first_idx: int | None = None
    for idx, line in enumerate(lines):
        if line.strip():
            first_idx = idx
            break
    if first_idx is None:
        return None, ""
    head = lines[first_idx].strip().lower()
    for token, mode in _EXTEND_TOKENS:
        if head == token or (len(head) > len(token) and head.startswith(token) and head[len(token)].isspace()):
            inline_rest = lines[first_idx].strip()[len(token):].strip()
            tail = "\n".join(lines[first_idx + 1 :]).strip()
            parts = [p for p in (inline_rest, tail) if p]
            return mode, "\n\n".join(parts)
    return None, norm.strip()


def compose_preset_system_prompt(
    local_system_prompt: str,
    preset_system_prompt: str,
    *,
    preset_name: str = "",
) -> tuple[str, str, str | None]:
    """Compose a tracked local `.md` prompt with a preset under Direction A + `@extend` overlay rules.

    Returns `(effective_system_prompt, sys_source, extend_mode)`.
    """
    local_clean = (local_system_prompt or "").strip()
    mode, body = parse_extend_directive(preset_system_prompt)
    pname = (preset_name or "").strip()
    if mode == "prepend":
        parts = [p for p in (body, local_clean) if p]
        src = f"local+extend:{pname}" if pname else "local+extend"
        return "\n\n".join(parts), src, mode
    if mode == "append":
        parts = [p for p in (local_clean, body) if p]
        src = f"local+extend:{pname}" if pname else "local+extend"
        return "\n\n".join(parts), src, mode
    if body:
        src = f"preset:{pname}" if pname else "preset"
        return body, src, None
    return local_clean, "local", None


def build_heylook_request_body(
    *,
    model_row: dict,
    system_prompt: str,
    user_prompt: str,
    prepared_images: list[tuple[str, str]],
    thinking: str = "model default",
    request_seed: int | None = None,
    preset_fields: dict | None = None,
    sampler_mode: str = "heylook / preset default",
    sampler_overrides: dict | None = None,
    reasoning_effort_override: str = "",
    default_reasoning_effort: str = "",
    extra_messages: list[dict] | None = None,
) -> dict:
    """Assemble a capability-gated `POST /v1/messages` request body."""
    if thinking not in THINKING_MODES:
        raise ValueError(f"unknown thinking setting {thinking!r}; expected one of {list(THINKING_MODES)}")
    if sampler_mode not in SAMPLER_MODES:
        raise ValueError(f"unknown sampler_mode {sampler_mode!r}; expected one of {list(SAMPLER_MODES)}")

    caps = set(model_row.get("capabilities") or [])
    engine = model_row.get("engine") or {}
    allowed_fields = _fact_value(engine, "decoding", "request_fields")
    allowed_set = set(allowed_fields) if isinstance(allowed_fields, list) else None

    body: dict = dict(preset_fields or {})
    if system_prompt:
        body["system"] = system_prompt

    if sampler_mode == "custom" and sampler_overrides:
        body.update(sampler_overrides)

    thinking_meta = engine.get("thinking") or {}
    depth = thinking_meta.get("depth") or {}
    eff = (reasoning_effort_override or "").strip() if sampler_mode == "custom" else ""
    if eff:
        if (
            depth.get("unknown") == "verbatim"
            or eff in (depth.get("values") or [])
            or eff in (depth.get("aliases") or {})
        ):
            body["reasoning_effort"] = eff
        elif depth.get("values"):
            raise ValueError(
                f"model {model_row['id']!r} does not offer reasoning_effort={eff!r}; "
                f"valid depth values: {depth.get('values')}"
            )
    else:
        def_eff = (default_reasoning_effort or "").strip()
        if (
            def_eff
            and thinking != "off"
            and "reasoning_effort" not in body
            and (
                def_eff in (depth.get("values") or [])
                or def_eff in (depth.get("aliases") or {})
            )
        ):
            body["reasoning_effort"] = def_eff

    has_switch = "thinking" in caps or bool(thinking_meta.get("switch"))
    if thinking in ("on", "off"):
        want_on = thinking == "on"
        if has_switch:
            existing = body.get("thinking")
            if want_on and isinstance(existing, dict) and "budget_tokens" in existing:
                body["thinking"] = {**existing, "type": "enabled"}
            else:
                body["thinking"] = want_on
        elif not want_on:
            off_words = depth.get("off") or []
            if off_words:
                body["reasoning_effort"] = off_words[0]
            body.pop("thinking", None)
        else:
            body.pop("thinking", None)

    if request_seed is not None and (allowed_set is None or "seed" in allowed_set):
        body["seed"] = int(request_seed) & 0x7FFFFFFF

    if allowed_set is not None:
        sampler_keys = (
            "max_tokens", "temperature", "top_p", "top_k", "min_p",
            "repetition_penalty", "repetition_context_size", "presence_penalty", "seed",
        )
        for k in sampler_keys:
            if k in body and k not in allowed_set:
                del body[k]

    if prepared_images:
        content: list[dict] = []
        for idx, (b64_data, media_type) in enumerate(prepared_images, start=1):
            content.append({"type": "text", "text": f"<Picture {idx}>:"})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64_data,
                },
            })
        content.append({"type": "text", "text": user_prompt})
        messages: list[dict] = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "user", "content": user_prompt}]

    if extra_messages:
        messages.extend(extra_messages)

    body["model"] = model_row["id"]
    body["messages"] = messages
    body["stream"] = True
    return body


def _iter_sse(lines: Any, interrupt_check: Callable[[], None] | None = None):
    event: str | None = None
    for line in lines:
        if interrupt_check is not None:
            interrupt_check()
        if not line:
            event = None
            continue
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: ") and event:
            yield event, json.loads(line[6:])


def cancel_heylook_request(base: str, request_id: str, *, client: httpx.Client | None = None) -> bool:
    """`DELETE /v1/requests/{request_id}`. 404 means already finished (`False`)."""
    encoded = urllib.parse.quote(request_id, safe="")
    url = f"{base}/v1/requests/{encoded}"
    if client is not None:
        r = client.delete(url, timeout=5.0)
    else:
        with httpx.Client(trust_env=False, timeout=5.0) as c:
            r = c.delete(url)
    if r.status_code == 404:
        return False
    r.raise_for_status()
    return int(r.json().get("cancelled") or 0) > 0


def with_heylook_backoff(
    fn: Callable[[], HeylookStreamResult],
    *,
    attempts: int = 5,
    interrupt_check: Callable[[], None] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> HeylookStreamResult:
    """Retry HTTP 503 (`HeylookOverloadedError`) honouring `Retry-After` (floored at 1s, capped at 30s)."""
    for i in range(attempts):
        try:
            return fn()
        except HeylookOverloadedError as exc:
            if i == attempts - 1:
                raise
            wait_s = float(min(max(exc.retry_after or (2 ** i), 1.0), 30.0))
            remaining = wait_s
            while remaining > 0.0:
                if interrupt_check is not None:
                    interrupt_check()
                step = min(remaining, 0.25)
                sleep_fn(step)
                remaining -= step
    raise RuntimeError("unreachable")


def stream_heylook_message(
    base: str,
    body: dict,
    *,
    client: httpx.Client | None = None,
    interrupt_check: Callable[[], None] | None = None,
) -> HeylookStreamResult:
    """Stream one `POST /v1/messages` request with a fresh `X-Request-ID` and cancel-on-abort."""
    req_id = str(uuid.uuid4())
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "X-Request-ID": req_id,
    }
    payload = {**body, "stream": True}
    out = HeylookStreamResult(model=str(body.get("model") or ""), request_id=req_id)
    finished = False
    tracked_id = req_id

    def _run_with(c: httpx.Client) -> None:
        nonlocal finished, tracked_id
        try:
            with c.stream("POST", f"{base}/v1/messages", json=payload, headers=headers) as r:
                tracked_id = r.headers.get("X-Request-ID") or req_id
                out.request_id = tracked_id
                if r.status_code >= 400:
                    r.read()
                    raise _http_error(r, base, "/v1/messages")

                for event, data in _iter_sse(r.iter_lines(), interrupt_check=interrupt_check):
                    if event == "content_block_delta":
                        delta = data.get("delta") or {}
                        dtype = delta.get("type")
                        if dtype == "thinking_delta":
                            out.thinking += delta.get("thinking") or delta.get("text") or ""
                        elif dtype == "text_delta":
                            out.text += delta.get("text") or ""
                    elif event == "message_delta":
                        delta = data.get("delta") or {}
                        out.stop_reason = delta.get("stop_reason")
                        out.stop_sequence = delta.get("stop_sequence")
                        out.usage = data.get("usage") or {}
                    elif event == "message_stop":
                        out.performance = data.get("performance") or {}
                        finished = True
                        break
                    elif event == "error":
                        err = data.get("error") or {}
                        etype = err.get("type") or "error"
                        emsg = err.get("message") or str(data)
                        if etype == "invalid_request_error":
                            raise ValueError(f"heylook invalid_request_error: {emsg}")
                        raise RuntimeError(f"heylook {etype}: {emsg}")
        except BaseException:
            if not finished:
                try:
                    cancel_heylook_request(base, tracked_id, client=c)
                except Exception:
                    pass
            raise

    if client is not None:
        _run_with(client)
    else:
        with httpx.Client(trust_env=False, timeout=httpx.Timeout(None, connect=10.0)) as owned:
            _run_with(owned)

    if not finished:
        raise RuntimeError("heylook stream ended without message_stop")
    return out


def format_heylook_report(
    res: HeylookStreamResult,
    *,
    resolved_mode: str,
    shape: str | None = None,
    system_source: str | None = None,
    concept: str | None = None,
) -> str:
    """Format telemetry defensively (`generation_duration_ms` as throughput denominator)."""
    parts = [f"model={res.model}", f"mode={resolved_mode}"]
    if system_source:
        parts.append(f"sys={system_source}")
    if concept and str(concept).strip().lower() != "none":
        parts.append(f"concept={str(concept).strip()}")
    if shape:
        parts.append(f"shape={shape}")
    if res.stop_reason:
        parts.append(f"stop={res.stop_reason}")
    usage = res.usage if isinstance(res.usage, dict) else {}
    in_tok = usage.get("input_tokens")
    cached_tok = usage.get("cache_read_input_tokens")
    out_tok = usage.get("output_tokens")
    if in_tok is not None:
        parts.append(f"in={in_tok}" + (f"(+{cached_tok} cached)" if cached_tok is not None else ""))
    if out_tok is not None:
        parts.append(f"out={out_tok}")

    perf = res.performance if isinstance(res.performance, dict) else {}
    gen_tps = perf.get("generation_tps")
    gen_ms = perf.get("generation_duration_ms")
    if gen_tps is None and out_tok is not None and gen_ms is not None and gen_ms > 0:
        gen_tps = float(out_tok) / (float(gen_ms) / 1000.0)
    if gen_tps is not None:
        parts.append(f"tps={float(gen_tps):.1f}")
    if gen_ms is not None:
        parts.append(f"gen_ms={float(gen_ms):.0f}")
    return " ".join(parts)


def probe_heylook_server_info(base_url: str = "", *, fallback_env: bool = False) -> dict:
    """Probe live `heylook` `/v1/models`, `/v1/system/metrics`, and `/v1/presets` for UI autofill.

    When `fallback_env=True` and `base_url` is blank, resolves via `resolve_heylook_base_url("")`
    server-side while returning `"base_url": ""` so host-specific LAN IPs stay out of client state.
    """
    heylook_info: dict = {
        "base_url": "",
        "reachable": False,
        "error": None,
        "resident_models": [],
        "models": [],
        "presets": [],
    }
    raw_url = (base_url or "").strip()
    if not raw_url and not fallback_env:
        return heylook_info

    try:
        base = normalize_heylook_base_url(raw_url) if raw_url else resolve_heylook_base_url("")
        heylook_info["base_url"] = base if raw_url else ""
        with httpx.Client(trust_env=False, timeout=httpx.Timeout(3.0, connect=2.0)) as client:
            mr = client.get(f"{base}/v1/models", headers={"Accept": "application/json"})
            if mr.status_code >= 400:
                raise _http_error(mr, base, "/v1/models")
            rows = [
                r for r in (mr.json().get("data") or [])
                if isinstance(r, dict) and r.get("id") and "chat" in (r.get("capabilities") or [])
            ]
            heylook_info["models"] = [
                {
                    "id": r["id"],
                    "provider": r.get("provider") or "",
                    "capabilities": list(r.get("capabilities") or []),
                    "thinking_default": r.get("thinking_default"),
                    "sampler_defaults": dict(r.get("sampler_defaults") or {}),
                    "depth_values": list(
                        (((r.get("engine") or {}).get("thinking") or {}).get("depth") or {}).get("values")
                        or []
                    ),
                }
                for r in rows
            ]
            try:
                sr = client.get(f"{base}/v1/system/metrics", headers={"Accept": "application/json"})
                if sr.status_code == 200:
                    mmap = sr.json().get("models")
                    if isinstance(mmap, dict):
                        heylook_info["resident_models"] = list(mmap.keys())
            except Exception:
                pass
            try:
                pr = client.get(f"{base}/v1/presets", headers={"Accept": "application/json"})
                if pr.status_code == 200:
                    heylook_info["presets"] = [
                        {
                            "id": p.get("id") or "",
                            "name": p.get("name") or p.get("id") or "",
                            "system_prompt": p.get("system_prompt") or "",
                            "params": dict(p.get("params") or {}),
                        }
                        for p in (pr.json().get("presets") or [])
                        if isinstance(p, dict) and (p.get("name") or p.get("id"))
                    ]
            except Exception:
                pass
            heylook_info["reachable"] = True
    except Exception as exc:
        heylook_info["error"] = f"{type(exc).__name__}: {exc}"
    return heylook_info


def _comfy_interrupt_check() -> None:
    try:
        import comfy.model_management as mm
    except Exception:
        return
    mm.throw_exception_if_processing_interrupted()
