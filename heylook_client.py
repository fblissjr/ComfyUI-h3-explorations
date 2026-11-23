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
from datetime import datetime, timezone
import hashlib
import io as _pyio
import json
import logging
import os
import socket
import sys
import threading
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

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
DEFAULT_HEYLOOK_MODEL = ""
DEFAULT_STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)
THINKING_MODES: tuple[str, ...] = ("model default", "on", "off")
SAMPLER_MODES: tuple[str, ...] = ("heylook / preset default", "custom")

HEYLOOK_PROVENANCE_KEYS: tuple[str, ...] = (
    "author_kind",
    "requested_model",
    "resolved_model",
    "preset",
    "system_source",
    "system_prompt_sha256",
    "thinking",
    "reasoning_effort",
    "temperature",
    "seed",
    "repair_used",
    "input_tokens",
    "output_tokens",
    "created_utc",
)

logger = logging.getLogger(__name__)

# Canonical prompt-mode / pipeline keys mapped to `bench/sync_heylook_presets.py` catalog slugs.
CANONICAL_PRESET_SLUGS: dict[str, str] = {
    "t2va": "h3-video-t2va",
    "i2va": "h3-video-i2va",
    "fl2va": "h3-video-fl2va",
    "l2va": "h3-video-l2va",
    "ref2va": "h3-video-ref2va",
    "section_edits": "h3-auto-edit-sections",
    "edit_lines": "h3-auto-edit-lines",
    "t2i": "h3-image-t2i",
    "t2i + aspect_ratio": "h3-image-t2i-ar",
    "t2i_design": "h3-image-t2i-design",
    "t2i_design + aspect_ratio": "h3-image-t2i-design-ar",
    "edit": "h3-image-edit",
    "edit + aspect_ratio": "h3-image-edit-ar",
    "ph3_t2i": "h3-ph3-t2i",
    "ph3_edit": "h3-ph3-edit",
    "pika_edit_structured": "h3-general-image-edit",
    "general_image_edit": "h3-general-image-edit",
    "director_reticle_suggest": "h3-director-reticle-suggest",
    "director_scene_suggest": "h3-director-scene-suggest",
    "director_shot_continue": "h3-director-shot-continue",
}


class HeylookOverloadedError(RuntimeError):
    """HTTP 503 `model_overloaded`: queue signal on a single-user local server."""

    def __init__(self, message: str = "model_overloaded", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class HeylookCancelledError(RuntimeError):
    """Raised when an active or queued `heylook` request is cancelled."""


def _force_close_httpx_client(client: httpx.Client) -> None:
    """Shutdown live `httpcore` sockets before closing `client` so blocked `recv()` wakes on Linux."""
    transports = [getattr(client, "_transport", None), *(getattr(client, "_mounts", {}) or {}).values()]
    for tr in transports:
        if tr is None:
            continue
        pool = getattr(tr, "_pool", None)
        if pool is None:
            continue
        try:
            conns = list(getattr(pool, "_connections", ()) or ())
        except Exception:
            continue
        for conn in conns:
            inner = getattr(conn, "_connection", conn)
            stream = getattr(inner, "_network_stream", None)
            if stream is None:
                continue
            sock = None
            get_extra = getattr(stream, "get_extra_info", None)
            if callable(get_extra):
                try:
                    sock = get_extra("socket")
                except Exception:
                    sock = None
            if sock is None:
                sock = getattr(stream, "_sock", None)
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
    try:
        client.close()
    except Exception:
        pass


class HeylookCancelToken:
    """Thread-safe cancellation and upstream `DELETE /v1/requests/{id}` coordinator."""

    def __init__(
        self,
        cancel_exc: Callable[[], BaseException] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._cancel_exc = cancel_exc
        self._base_url: str = ""
        self._request_id: str = ""
        self._api_key: str | None = None
        self._upstream_confirmed: bool = False
        self._active_client: httpx.Client | None = None

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @property
    def active_request_id(self) -> str:
        with self._lock:
            return self._request_id

    def _make_exc(self, message: str = "heylook request cancelled") -> BaseException:
        if self._cancel_exc is not None:
            return self._cancel_exc()
        return HeylookCancelledError(message)

    def check(self) -> None:
        if self._cancel_event.is_set():
            raise self._make_exc()

    def attach_client(self, client: httpx.Client | None, *, if_unset: bool = False) -> bool:
        """Attach the active `httpx.Client` so `cancel()` can close stalled sockets after `DELETE`."""
        if client is None:
            return False
        cancelled_now = False
        attached = False
        with self._lock:
            if if_unset and self._active_client is not None:
                return False
            self._active_client = client
            attached = True
            cancelled_now = self._cancel_event.is_set()
        if cancelled_now:
            _force_close_httpx_client(client)
            raise self._make_exc()
        return attached

    def detach_client(self, client: httpx.Client | None = None) -> None:
        with self._lock:
            if client is None or self._active_client is client:
                self._active_client = None

    def begin_request(self, base: str, req_id: str, *, api_key: str | None = None) -> None:
        with self._lock:
            if self._cancel_event.is_set():
                raise self._make_exc()
            self._base_url = base
            self._request_id = str(req_id or "")
            self._api_key = api_key
            self._upstream_confirmed = False

    def update_request_id(self, req_id: str) -> None:
        rid = str(req_id or "").strip()
        if not rid:
            return
        with self._lock:
            if rid != self._request_id:
                self._request_id = rid
                self._upstream_confirmed = False

    def end_request(self) -> None:
        with self._lock:
            self._base_url = ""
            self._request_id = ""
            self._api_key = None
            self._upstream_confirmed = False

    def cancel(self) -> dict[str, Any]:
        """Signal local abort, issue `DELETE /v1/requests/{id}`, then close any stalled socket."""
        self._cancel_event.set()
        upstream_ok = self.cancel_if_needed()
        with self._lock:
            rid = self._request_id
            active_c = self._active_client
        if active_c is not None:
            _force_close_httpx_client(active_c)
        return {"cancelled": True, "request_id": rid, "upstream_cancelled": upstream_ok}

    def cancel_if_needed(self) -> bool:
        """Issue `DELETE /v1/requests/{id}` unless upstream already confirmed cancellation."""
        with self._lock:
            base = self._base_url
            rid = self._request_id
            api_key = self._api_key
            if not base or not rid:
                return False
            if self._upstream_confirmed:
                return True
        ok = False
        try:
            ok = bool(cancel_heylook_request(base, rid, client=None, api_key=api_key))
        except Exception:
            ok = False
        if ok:
            with self._lock:
                if self._request_id == rid:
                    self._upstream_confirmed = True
        return ok


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


def _origin_tuple(url: str) -> tuple[str, str | None, int]:
    p = urllib.parse.urlsplit(url)
    scheme = p.scheme.lower()
    host = (p.hostname or "").lower() or None
    port = p.port or (443 if scheme == "https" else 80)
    return (scheme, host, port)


def _read_h3_env_key_file() -> str:
    env_file = Path(os.environ.get("H3_ENV_FILE") or h3_env.ENV_FILE)
    try:
        parsed = h3_env.parse(env_file.read_text(encoding="utf-8"))
    except OSError:
        return ""
    return (parsed.get("H3_HEYLOOK_API_KEY_FILE") or "").strip()


def _resolve_heylook_api_key_details(explicit: str | None = None) -> tuple[str | None, str, str | None]:
    """Resolve access-gate key and non-secret source metadata without leaking secrets or mutating `H3_HEYLOOK_URL`."""
    if explicit and explicit.strip():
        return explicit.strip(), "explicit", None
    env_key = os.environ.get("HEYLOOK_API_KEY", "").strip()
    if env_key:
        return env_key, "HEYLOOK_API_KEY", None

    candidate: tuple[Path, str, bool] | None = None
    if "HEYLOOK_API_KEY_FILE" in os.environ:
        raw_path = os.environ["HEYLOOK_API_KEY_FILE"].strip()
        if raw_path:
            candidate = (Path(raw_path).expanduser(), "HEYLOOK_API_KEY_FILE", True)
    else:
        h3_file = os.environ.get("H3_HEYLOOK_API_KEY_FILE", "").strip() or _read_h3_env_key_file()
        if h3_file:
            candidate = (Path(h3_file).expanduser(), "H3_HEYLOOK_API_KEY_FILE", True)
        else:
            candidate = (Path.home() / ".heylook" / "api_key", "~/.heylook/api_key", False)

    if candidate is None:
        return None, "none", None

    p, src, is_explicit = candidate
    try:
        if not p.is_file():
            return None, "none", None
        st = p.stat()
    except OSError:
        return None, "none", None

    if is_explicit:
        if (st.st_mode & 0o027) != 0:
            return (
                None,
                "none",
                f"ignored {src}: must not be group-writable or readable by others (chmod 640 or 600)",
            )
    elif (st.st_mode & 0o077) != 0:
        return None, "none", f"ignored {src}: permissions must be mode 0600 (run chmod 600)"

    try:
        val = p.read_text(encoding="utf-8").strip()
    except OSError:
        return None, "none", None
    if not val:
        return None, "none", f"ignored {src}: file is empty"
    return val, src, None


def resolve_heylook_api_key(explicit: str | None = None) -> tuple[str | None, str]:
    """Resolve `heylook` access-gate key (`(key, source)`): explicit `0640`/`0600` (`0o027`) or auto-discovered `0600` (`0o077`)."""
    key, src, _warn = _resolve_heylook_api_key_details(explicit)
    return key, src


def heylook_auth_headers(api_key: str | None = None) -> dict[str, str]:
    """Return `{\"x-api-key\": key}` when a `heylook` credential is configured, else `{}`."""
    key, _src = resolve_heylook_api_key(api_key)
    return {"x-api-key": key} if key else {}


def _req_headers(
    extra: dict[str, str] | None = None,
    *,
    client: httpx.Client | None = None,
    api_key: str | None = None,
) -> dict[str, str]:
    hdrs = dict(extra or {})
    low_keys = {k.lower() for k in hdrs}
    if client is not None:
        low_keys |= {k.lower() for k in client.headers.keys()}
    if "x-api-key" not in low_keys and "authorization" not in low_keys:
        hdrs.update(heylook_auth_headers(api_key))
    return hdrs


def make_heylook_client(
    base: str,
    *,
    timeout: Any = None,
    api_key: str | None = None,
    event_hooks: dict[str, list[Any]] | None = None,
) -> httpx.Client:
    """Create a redirect-disabled, origin-scoped `httpx.Client` injecting `x-api-key` only for `base`'s origin."""
    expected_origin = _origin_tuple(normalize_heylook_base_url(base))
    key, _src = resolve_heylook_api_key(api_key)

    def _auth_origin_hook(request: httpx.Request) -> None:
        if _origin_tuple(str(request.url)) == expected_origin:
            if key and not ("x-api-key" in request.headers or "authorization" in request.headers):
                request.headers["x-api-key"] = key
        else:
            request.headers.pop("x-api-key", None)
            request.headers.pop("authorization", None)

    merged_hooks: dict[str, list[Any]] = {}
    for hook_name, fn_list in (event_hooks or {}).items():
        merged_hooks[hook_name] = list(fn_list)
    merged_hooks["request"] = [_auth_origin_hook, *merged_hooks.get("request", [])]

    return httpx.Client(
        trust_env=False,
        follow_redirects=False,
        timeout=timeout if timeout is not None else DEFAULT_STREAM_TIMEOUT,
        event_hooks=merged_hooks,
    )


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


def _http_error(
    r: httpx.Response,
    base: str,
    path: str,
    *,
    sent_auth: bool | None = None,
) -> Exception:
    try:
        payload = r.json()
    except Exception:
        payload = None
    err_obj = payload.get("error") if isinstance(payload, dict) and isinstance(payload.get("error"), dict) else {}
    err_code = err_obj.get("code")
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if detail is None:
        detail = err_obj.get("message")
    if detail is None and payload is not None:
        detail = str(payload)[:240]
    elif detail is None:
        detail = r.text[:240]

    if r.status_code == 503:
        if err_code == "auth_misconfigured":
            return PermissionError(
                f"{base}{path} answered HTTP 503 (auth_misconfigured): "
                f"{detail or 'access key file is misconfigured'}. "
                "Fix server key-file permissions or ensure the key file is non-empty."
            )
        try:
            retry = float(r.headers.get("Retry-After", ""))
        except ValueError:
            retry = None
        msg = "model_overloaded"
        if err_obj.get("message"):
            msg = f"model_overloaded: {err_obj['message']}"
        return HeylookOverloadedError(msg, retry_after=retry)
    if r.status_code == 403:
        if err_code == "csrf_rejected":
            return PermissionError(
                f"{base}{path} answered HTTP 403 (csrf_rejected): "
                f"state-changing request failed heylook's Origin/Sec-Fetch-Site check ({detail or 'csrf_rejected'})."
            )
        return PermissionError(
            f"{base}{path} answered HTTP 403 (heylook Host check / DNS-rebinding guard): "
            f"{detail or 'Host not allowed'}. "
            "Address the server by IP or 127.0.0.1, or add the hostname to allowed_hosts in heylook.toml."
        )
    if r.status_code == 401:
        if sent_auth is not None:
            had_key = sent_auth
        else:
            req = getattr(r, "request", None)
            had_key = bool(req and (req.headers.get("x-api-key") or req.headers.get("authorization")))
        if had_key:
            return PermissionError(
                f"{base}{path} answered HTTP 401 (auth_required): the configured heylook API key was rejected. "
                "Check HEYLOOK_API_KEY / key file and target URL."
            )
        _k, _src, key_warn = _resolve_heylook_api_key_details()
        warn_suffix = f" ({key_warn}.)" if key_warn else ""
        return PermissionError(
            f"{base}{path} answered HTTP 401 (auth_required): heylook's access gate requires authentication. "
            f"Set HEYLOOK_API_KEY, point HEYLOOK_API_KEY_FILE / H3_HEYLOOK_API_KEY_FILE at a 0640/0600 key file, or create a mode-0600 ~/.heylook/api_key.{warn_suffix}"
        )
    if r.status_code in (400, 422):
        return ValueError(f"heylook HTTP {r.status_code} on {path}: {detail}")
    return RuntimeError(f"heylook HTTP {r.status_code} on {path}: {detail}")


def resolve_heylook_model(requested_model: str = "") -> tuple[str, bool]:
    """Resolve desired `heylook` model ID: explicit caller > `H3_HEYLOOK_MODEL` env > Auto (`""`)."""
    explicit = (requested_model or "").strip()
    if explicit and explicit.lower() != "auto":
        return explicit, False
    h3_env.load()
    env_model = (os.environ.get("H3_HEYLOOK_MODEL") or "").strip()
    if env_model and env_model.lower() != "auto":
        return env_model, True
    return "", False


def resolve_heylook_preset_prefix(explicit: str = "") -> str:
    """Resolve remote preset prefix (`explicit > H3_HEYLOOK_PRESET_PREFIX > ""`)."""
    exp = (explicit or "").strip()
    if exp:
        return exp
    h3_env.load()
    return (os.environ.get("H3_HEYLOOK_PRESET_PREFIX") or "").strip()


def format_preset_target_name(slug: str, prefix: str = "") -> str:
    """Format a canonical preset slug with an optional remote preset prefix (e.g. `h3app_`)."""
    pfx = (prefix or "").strip()
    if not pfx:
        return slug
    core = slug
    for lead in ("h3-", "h3_"):
        if core.startswith(lead):
            core = core[len(lead):]
            break
    if "_" in pfx:
        core = core.replace("-", "_")
    return f"{pfx}{core}"


def discover_heylook_model(
    client: httpx.Client,
    base: str,
    requested_model: str = "",
    *,
    need: set[str] | frozenset[str] = frozenset({"chat"}),
    api_key: str | None = None,
    fallback_models: tuple[str, ...] | set[str] | frozenset[str] = (),
) -> dict:
    """Resolve and capability-gate a `/v1/models` row at runtime (`capabilities`, never `modalities`)."""
    req_hdrs = _req_headers({"Accept": "application/json"}, client=client, api_key=api_key)
    try:
        r = client.get(f"{base}/v1/models", headers=req_hdrs, timeout=10.0)
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

    wanted, from_env = resolve_heylook_model(requested_model)
    if wanted:
        chosen = next((row for row in usable if row["id"] == wanted), None)
        if chosen is None:
            if from_env or wanted in fallback_models:
                logger.warning(
                    "[h3] heylook model %r is not served by %s/v1/models; falling back to Auto (prefer resident)",
                    wanted,
                    base,
                )
            else:
                fit_ids = [row["id"] for row in fit]
                raise ValueError(
                    f"model {wanted!r} is not served by {base}/v1/models; "
                    f"models serving {sorted(req_set)}: {fit_ids}"
                )
        else:
            caps = set(chosen.get("capabilities") or [])
            missing = sorted(req_set - caps)
            if missing:
                if from_env or wanted in fallback_models:
                    logger.warning(
                        "[h3] heylook model %r lacks required capabilities %s; falling back to Auto (prefer resident)",
                        wanted,
                        missing,
                    )
                else:
                    fit_ids = [row["id"] for row in fit]
                    raise ValueError(
                        f"model {wanted!r} lacks required capabilities {missing} (has {sorted(caps)}); "
                        f"models serving {sorted(req_set)}: {fit_ids}"
                    )
            else:
                return chosen

    if not fit:
        roster = [(row["id"], row.get("capabilities")) for row in usable]
        raise LookupError(f"no model at {base} serves capabilities {sorted(req_set)}; roster: {roster}")

    resident: set[str] = set()
    try:
        mr = client.get(f"{base}/v1/system/metrics", headers=req_hdrs, timeout=5.0)
        if mr.status_code == 200:
            models_map = mr.json().get("models")
            if isinstance(models_map, dict):
                resident = set(models_map.keys())
    except Exception:
        pass
    return next((row for row in fit if row["id"] in resident), fit[0])


def _compile_preset_body_fields(preset: dict, model_row: dict) -> dict:
    """Compile capability- and depth-filtered request body defaults from a `/v1/presets` dict."""
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


def _match_preset_row(presets: list[dict], wanted: str) -> dict | None:
    cand = (wanted or "").strip()
    if not cand:
        return None
    preset = next((p for p in presets if p.get("id") == cand), None)
    if preset is None:
        preset = next((p for p in presets if p.get("name") == cand), None)
    if preset is None:
        lowered = cand.lower()
        preset = next((p for p in presets if str(p.get("name") or "").lower() == lowered), None)
    return preset


def expand_heylook_preset(
    client: httpx.Client,
    base: str,
    preset_id: str,
    model_row: dict,
    *,
    api_key: str | None = None,
) -> dict:
    """Expand a `/v1/presets` entry client-side with capability & depth filtering."""
    req_hdrs = _req_headers({"Accept": "application/json"}, client=client, api_key=api_key)
    r = client.get(f"{base}/v1/presets", headers=req_hdrs, timeout=10.0)
    if r.status_code >= 400:
        raise _http_error(r, base, "/v1/presets")
    presets = [p for p in (r.json().get("presets") or []) if isinstance(p, dict)]
    avail = [p.get("name") or p.get("id") for p in presets if p.get("name") or p.get("id")]

    wanted = (preset_id or "").strip()
    if not wanted:
        raise ValueError(
            f"'preset' is blank; choose a preset from {base}/v1/presets: {avail}"
        )
    preset = _match_preset_row(presets, wanted)
    if preset is None:
        raise ValueError(f"heylook preset {wanted!r} not found at {base}/v1/presets; available: {avail}")
    return _compile_preset_body_fields(preset, model_row)


def try_inherit_heylook_preset(
    client: httpx.Client,
    base: str,
    canonical_slug: str,
    model_row: dict,
    *,
    prefix: str = "",
    api_key: str | None = None,
) -> tuple[dict | None, str]:
    """Best-effort lookup of a synced canonical preset on `/v1/presets` for downstream inheritance.

    Returns `(expanded_preset_fields, resolved_preset_name)` when present on the server,
    or `(None, "")` when the preset has not been synced or `/v1/presets` is unavailable.
    Auth/host errors (`PermissionError`) still propagate cleanly.
    """
    slug = (canonical_slug or "").strip()
    if not slug:
        return None, ""
    eff_prefix = resolve_heylook_preset_prefix(prefix)
    target_name = format_preset_target_name(slug, eff_prefix)
    try:
        req_hdrs = _req_headers({"Accept": "application/json"}, client=client, api_key=api_key)
        r = client.get(f"{base}/v1/presets", headers=req_hdrs, timeout=10.0)
        if r.status_code >= 400:
            raise _http_error(r, base, "/v1/presets")
        presets = [p for p in (r.json().get("presets") or []) if isinstance(p, dict)]
        preset = _match_preset_row(presets, target_name)
        if preset is not None:
            resolved_name = str(preset.get("name") or target_name)
            return _compile_preset_body_fields(preset, model_row), resolved_name
        return None, ""
    except (ValueError, LookupError, RuntimeError, httpx.HTTPError):
        return None, ""


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
    thinking_budget: int | None = None,
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

    if thinking_budget is not None and thinking != "off" and "thinking_budget" in caps:
        existing_t = body.get("thinking")
        think_obj: dict[str, Any] = dict(existing_t) if isinstance(existing_t, dict) else {}
        think_obj["budget_tokens"] = int(thinking_budget)
        if thinking == "on" or existing_t is True:
            think_obj["type"] = "enabled"
        body["thinking"] = think_obj

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


def cancel_heylook_request(
    base: str,
    request_id: str,
    *,
    client: httpx.Client | None = None,
    api_key: str | None = None,
) -> bool:
    """`DELETE /v1/requests/{request_id}`. 404 means already finished (`False`)."""
    encoded = urllib.parse.quote(request_id, safe="")
    url = f"{base}/v1/requests/{encoded}"
    req_hdrs = _req_headers({}, client=client, api_key=api_key)
    if client is not None:
        r = client.delete(url, headers=req_hdrs, timeout=5.0)
    else:
        with make_heylook_client(base, timeout=5.0, api_key=api_key) as c:
            r = c.delete(url, headers=req_hdrs)
    if r.status_code == 404:
        return False
    r.raise_for_status()
    if r.status_code == 204 or not (r.content or b"").strip():
        return True
    try:
        payload = r.json()
    except ValueError:
        return True
    if isinstance(payload, dict) and "cancelled" in payload:
        return int(payload.get("cancelled") or 0) > 0
    return True


def with_heylook_backoff(
    fn: Callable[[], HeylookStreamResult],
    *,
    attempts: int = 5,
    interrupt_check: Callable[[], None] | None = None,
    cancel_token: HeylookCancelToken | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> HeylookStreamResult:
    """Retry HTTP 503 (`HeylookOverloadedError`) honouring `Retry-After` (floored at 1s, capped at 30s)."""

    def _check_all() -> None:
        if interrupt_check is not None:
            interrupt_check()
        if cancel_token is not None:
            cancel_token.check()

    for i in range(attempts):
        _check_all()
        try:
            return fn()
        except HeylookOverloadedError as exc:
            if i == attempts - 1:
                raise
            wait_s = float(min(max(exc.retry_after or (2 ** i), 1.0), 30.0))
            remaining = wait_s
            while remaining > 0.0:
                _check_all()
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
    cancel_token: HeylookCancelToken | None = None,
    api_key: str | None = None,
) -> HeylookStreamResult:
    """Stream one `POST /v1/messages` request with a fresh `X-Request-ID` and cancel-on-abort."""
    req_id = str(uuid.uuid4())
    headers = _req_headers(
        {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Request-ID": req_id,
        },
        client=client,
        api_key=api_key,
    )
    payload = {**body, "stream": True}
    out = HeylookStreamResult(model=str(body.get("model") or ""), request_id=req_id)
    finished = False
    server_rejected = False
    upstream_confirmed = False
    tracked_id = req_id

    def _check_abort() -> None:
        if interrupt_check is not None:
            interrupt_check()
        if cancel_token is not None:
            cancel_token.check()

    def _run_with(c: httpx.Client) -> None:
        nonlocal finished, server_rejected, upstream_confirmed, tracked_id
        attached_here = False
        _check_abort()
        if cancel_token is not None:
            attached_here = cancel_token.attach_client(c, if_unset=True)
            cancel_token.begin_request(base, req_id, api_key=api_key)
        done_ev = threading.Event()
        watcher: threading.Thread | None = None
        if cancel_token is None and interrupt_check is _comfy_interrupt_check:
            def _comfy_watch() -> None:
                nonlocal upstream_confirmed
                try:
                    import comfy.model_management as mm
                except Exception:
                    return
                while not done_ev.wait(0.25):
                    try:
                        if mm.processing_interrupted():
                            try:
                                if cancel_heylook_request(base, tracked_id, client=None, api_key=api_key):
                                    upstream_confirmed = True
                            except Exception:
                                pass
                            _force_close_httpx_client(c)
                            return
                    except Exception:
                        return

            watcher = threading.Thread(
                target=_comfy_watch,
                name="h3-heylook-comfy-cancel-watch",
                daemon=True,
            )
            watcher.start()

        eff_sse_check = _check_abort if (interrupt_check is not None or cancel_token is not None) else None
        try:
            with c.stream("POST", f"{base}/v1/messages", json=payload, headers=headers) as r:
                tracked_id = r.headers.get("X-Request-ID") or req_id
                out.request_id = tracked_id
                if cancel_token is not None:
                    cancel_token.update_request_id(tracked_id)
                if r.status_code >= 400:
                    server_rejected = True
                    r.read()
                    raise _http_error(r, base, "/v1/messages")
                if cancel_token is not None:
                    cancel_token.check()

                for event, data in _iter_sse(r.iter_lines(), interrupt_check=eff_sse_check):
                    if event == "message_start":
                        msg = data.get("message") if isinstance(data.get("message"), dict) else {}
                        srv_model = str(msg.get("model") or "").strip()
                        if srv_model:
                            out.model = srv_model
                        start_usage = msg.get("usage")
                        if isinstance(start_usage, dict):
                            out.usage = {**out.usage, **start_usage}
                    elif event == "content_block_delta":
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
                        delta_usage = data.get("usage")
                        if isinstance(delta_usage, dict):
                            out.usage = {**out.usage, **delta_usage}
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
            done_ev.set()
            if not finished and not server_rejected:
                if cancel_token is not None:
                    cancel_token.cancel_if_needed()
                elif not upstream_confirmed:
                    try:
                        cancel_heylook_request(
                            base,
                            tracked_id,
                            client=c if client is not None else None,
                            api_key=api_key,
                        )
                    except Exception:
                        pass
            if cancel_token is not None and cancel_token.is_cancelled:
                cancel_token.check()
            elif cancel_token is None and interrupt_check is _comfy_interrupt_check:
                interrupt_check()
            raise
        finally:
            done_ev.set()
            if watcher is not None:
                watcher.join(timeout=1.0)
            if cancel_token is not None:
                if attached_here:
                    cancel_token.detach_client(c)
                cancel_token.end_request()

    if client is not None:
        _run_with(client)
    else:
        with make_heylook_client(base, api_key=api_key) as owned:
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


def system_prompt_sha256(system_prompt: str) -> str:
    """Return canonical LF-normalized SHA-256 hex digest of `system_prompt` (`""` when empty)."""
    norm = (system_prompt or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not norm:
        return ""
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _normalize_thinking_provenance(
    thinking: Any = "model default",
    request_body: Mapping[str, Any] | None = None,
) -> str:
    if isinstance(request_body, Mapping) and "thinking" in request_body:
        raw_t = request_body.get("thinking")
        if isinstance(raw_t, bool):
            return "on" if raw_t else "off"
        if isinstance(raw_t, Mapping):
            ttype = str(raw_t.get("type") or "").strip().lower()
            budget = raw_t.get("budget_tokens")
            if ttype == "disabled":
                return "off"
            if budget is not None:
                return f"budget:{int(budget)}"
            if ttype == "enabled":
                return "on"
    if isinstance(thinking, bool):
        return "on" if thinking else "off"
    label = str(thinking if thinking is not None else "").strip()
    return label or "model default"


def build_heylook_provenance(
    *,
    requested_model: str = "",
    resolved_model: str = "",
    res: HeylookStreamResult | None = None,
    model_row: Mapping[str, Any] | None = None,
    preset: str = "",
    system_source: str = "local",
    system_prompt: str = "",
    request_body: Mapping[str, Any] | None = None,
    thinking: Any = "model default",
    reasoning_effort: str = "",
    temperature: float | None = None,
    seed: int | None = None,
    repair_used: bool = False,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    extra_usage: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = (),
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Build canonical 14-key `HeylookProvenance` dict for A/B model and prompt reproducibility."""
    eff_resolved = str(
        (res.model if res is not None and res.model else "")
        or resolved_model
        or ((model_row or {}).get("id") if isinstance(model_row, Mapping) else "")
        or ""
    ).strip()
    eff_req = str(requested_model or "").strip()
    eff_sys = str(
        system_prompt
        or ((request_body or {}).get("system") if isinstance(request_body, Mapping) else "")
        or ""
    )
    eff_effort = str(
        ((request_body or {}).get("reasoning_effort") if isinstance(request_body, Mapping) else "")
        or reasoning_effort
        or ""
    ).strip()

    eff_temp: float | None = None
    if temperature is not None:
        eff_temp = float(temperature)
    elif isinstance(request_body, Mapping) and request_body.get("temperature") is not None:
        eff_temp = float(request_body["temperature"])
    elif isinstance(model_row, Mapping):
        sdefs = model_row.get("sampler_defaults")
        if isinstance(sdefs, Mapping) and sdefs.get("temperature") is not None:
            eff_temp = float(sdefs["temperature"])

    eff_seed: int | None = None
    if seed is not None:
        eff_seed = int(seed)
    elif isinstance(request_body, Mapping) and request_body.get("seed") is not None:
        eff_seed = int(request_body["seed"])

    usage_rows: list[Mapping[str, Any]] = []
    if res is not None and isinstance(res.usage, Mapping):
        usage_rows.append(res.usage)
    for extra in extra_usage:
        if isinstance(extra, Mapping):
            usage_rows.append(extra)

    in_tok = (
        max(0, int(input_tokens))
        if input_tokens is not None
        else max(0, sum(int(u.get("input_tokens") or 0) for u in usage_rows))
    )
    out_tok = (
        max(0, int(output_tokens))
        if output_tokens is not None
        else max(0, sum(int(u.get("output_tokens") or 0) for u in usage_rows))
    )
    ts = (
        str(created_utc).strip()
        if created_utc
        else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    return {
        "author_kind": "heylook",
        "requested_model": eff_req,
        "resolved_model": eff_resolved,
        "preset": str(preset or "").strip(),
        "system_source": str(system_source or "local").strip(),
        "system_prompt_sha256": system_prompt_sha256(eff_sys),
        "thinking": _normalize_thinking_provenance(thinking, request_body),
        "reasoning_effort": eff_effort,
        "temperature": eff_temp,
        "seed": eff_seed,
        "repair_used": bool(repair_used),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "created_utc": ts,
    }


def format_heylook_provenance_toml(provenance: Mapping[str, Any]) -> str:
    """Serialize a `HeylookProvenance` mapping as a valid `[provenance]` TOML table (omitting `None`)."""
    lines = ["[provenance]"]
    for key in HEYLOOK_PROVENANCE_KEYS:
        if key not in provenance:
            continue
        val = provenance[key]
        if val is None:
            continue
        if isinstance(val, bool):
            lines.append(f"{key} = {'true' if val else 'false'}")
        elif isinstance(val, int):
            lines.append(f"{key} = {val}")
        elif isinstance(val, float):
            lines.append(f"{key} = {repr(val)}")
        else:
            lines.append(f"{key} = {json.dumps(str(val), ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def probe_heylook_server_info(
    base_url: str = "",
    *,
    fallback_env: bool = False,
    api_key: str | None = None,
) -> dict:
    """Probe live `heylook` `/v1/models`, `/v1/system/metrics`, and `/v1/presets` for UI autofill.

    When `fallback_env=True` and `base_url` is blank, resolves via `resolve_heylook_base_url("")`
    server-side while returning `"base_url": ""` so host-specific LAN IPs stay out of client state.
    """
    key, key_source = resolve_heylook_api_key(api_key)
    heylook_info: dict = {
        "base_url": "",
        "reachable": False,
        "error": None,
        "auth_configured": bool(key),
        "auth_source": key_source,
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
        hdrs = _req_headers({"Accept": "application/json"}, api_key=api_key)
        with make_heylook_client(base, timeout=httpx.Timeout(3.0, connect=2.0), api_key=api_key) as client:
            mr = client.get(f"{base}/v1/models", headers=hdrs)
            if mr.status_code >= 400:
                raise _http_error(mr, base, "/v1/models", sent_auth=bool(key))
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
                sr = client.get(f"{base}/v1/system/metrics", headers=hdrs)
                if sr.status_code == 200:
                    mmap = sr.json().get("models")
                    if isinstance(mmap, dict):
                        heylook_info["resident_models"] = list(mmap.keys())
            except Exception:
                pass
            try:
                pr = client.get(f"{base}/v1/presets", headers=hdrs)
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
