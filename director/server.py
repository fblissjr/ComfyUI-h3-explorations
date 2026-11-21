"""Standalone backend HTTP + WebSocket server for Interactive Spatial Director.

Proxies ComfyUI (`/object_info`, `/system_stats`, `/prompt`, `/upload/image`,
`/view`, `/history`, `/interrupt`, `/ws`, `/h3_spatial_director/forget`)
and persists local take metadata in `--state-dir` (`data/spatial_director_app/`).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import hashlib
import io
import json
import logging
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlsplit

from aiohttp import ClientSession, ClientTimeout, FormData, WSMsgType, web

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from workflows.h3_env import load as load_h3_env  # noqa: E402

load_h3_env()

from director import compiler, routes as director_routes  # noqa: E402
from director.media import (  # noqa: E402
    compute_ab_metrics_sync,
    decode_frame0_or_image,
    decode_video_frame_rgb,
    encode_mask_rgb_png,
    encode_rgba_cutout_png,
    extract_studio_cutout_rgba,
    extract_video_frame_png,
    prepare_vlm_reticle_images,
    segment_grabcut_mask,
    video_bytes_has_audio,
)
from director.store import (  # noqa: E402
    APP_MARK,
    SAFE_FILENAME_RE,
    SessionStore,
    build_recipe_document,
    extract_lineage_fields,
    first_image_from_node_output,
    first_image_from_outputs,
    first_video_from_node_output,
    first_video_from_outputs,
    normalize_imported_recipe,
    normalize_media_entry,
    parse_view_descriptor,
    safe_ref_view_url,
    safe_rel_name,
    sanitize_filename,
    take_prompt_preview,
    view_url,
)

# Backwards-compatible internal aliases
_encode_rgba_cutout_png = encode_rgba_cutout_png
_encode_mask_rgb_png = encode_mask_rgb_png
_segment_grabcut_mask = segment_grabcut_mask
_compute_ab_metrics_sync = compute_ab_metrics_sync
_decode_frame0_or_image = decode_frame0_or_image
_decode_video_frame_rgb = decode_video_frame_rgb
_extract_video_frame_png = extract_video_frame_png
_video_bytes_has_audio = video_bytes_has_audio
_build_recipe_document = build_recipe_document
_normalize_imported_recipe = normalize_imported_recipe
_extract_lineage_fields = extract_lineage_fields
_parse_view_descriptor = parse_view_descriptor
_sanitize_filename = sanitize_filename
_view_url = view_url
_take_prompt_preview = take_prompt_preview
_SAFE_FILENAME_RE = SAFE_FILENAME_RE
_first_image_from_node_output = first_image_from_node_output
_first_video_from_outputs = first_video_from_outputs

log = logging.getLogger("h3.director")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8195
DEFAULT_HTTPS_PORT = 8194
DEFAULT_STATE_DIR = _REPO_ROOT / "data" / "spatial_director_app"
STATIC_DIR = _THIS_DIR / "static"
UPLOADS_SUBFOLDER = "h3_spatial_director"
COOKIE_NAME = "h3_director_sid"
MAX_UPLOAD_BYTES = 256 * 1024 * 1024

DEFAULT_DOC_PATH = "docs/wiki/spatial_director_guide.md"
_DOC_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        ".agents",
        ".claude",
        "app",
        "data",
        "internal",
        "coderef",
        "vendor",
        "archive",
        "changelog.d",
        "__pycache__",
        "node_modules",
    }
)
_PINNED_STARTER_DOCS: tuple[str, ...] = (
    "docs/wiki/spatial_director_guide.md",
    "director/VISION.md",
    "director/ROADMAP.md",
    "docs/wiki/edit_guidance.md",
    "docs/wiki/steering.md",
    "director/AGENTS.md",
    "docs/wiki/steering/how_pole_a_erasure_works.md",
    "docs/wiki/steering/which_tool_and_what_to_call_it.md",
    "docs/wiki/steering/sound_cues_and_timed_masks.md",
    "docs/wiki/steering/studio_ui_guide.md",
)
_DOC_GROUP_ORDER: tuple[str, ...] = (
    "1. Spatial Director & Steering",
    "2. Overview & Wiki Routers",
    "3. Core Video, Clocks & Audio",
    "4. Single-Frame Image",
    "5. Steering & J-Lens",
    "6. Attention, Sol & Quant",
    "7. Prompting & System Prompts",
    "8. Masked V2V Pipeline",
    "9. Upstream Engines & Sweeps",
    "10. Operations & Checks",
    "11. Projects, Research & Bench",
)
_DOC_IMAGE_EXTS: frozenset[str] = frozenset({".svg", ".png", ".jpg", ".jpeg", ".webp"})

CSP_POLICY = (
    "default-src 'self'; "
    "img-src 'self' blob: data:; "
    "media-src 'self' blob: data:; "
    "style-src 'self'; "
    "script-src 'self'; "
    "connect-src 'self' ws: wss:; "
    "frame-ancestors 'none'; "
    "base-uri 'none'; "
    "form-action 'self'"
)

_SAFE_SID_RE = re.compile(r"^[A-Za-z0-9_-]{12,128}$")
_SAFE_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


def _get_resident_backbone(app: web.Application) -> str | None:
    runtime = app.get("runtime")
    return runtime.get("last_resident_backbone") if isinstance(runtime, dict) else app.get("last_resident_backbone")


def _set_resident_backbone(app: web.Application, value: str | None) -> None:
    runtime = app.get("runtime")
    if isinstance(runtime, dict):
        runtime["last_resident_backbone"] = value
    else:
        app["last_resident_backbone"] = value


def _strip_host_port(host_header: str) -> str:
    val = (host_header or "").strip().lower()
    if not val:
        return ""
    if val.startswith("["):
        end = val.find("]")
        return val[1:end] if end != -1 else val
    return val.rsplit(":", 1)[0] if ":" in val else val


def _discover_tailscale_host() -> str | None:
    try:
        proc = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout:
            return None
        doc = json.loads(proc.stdout)
        dns = ((doc.get("Self") or {}).get("DNSName") or "").strip().rstrip(".").lower()
        return dns or None
    except Exception:
        return None


def _default_allowed_hosts(listen_host: str, extra_hosts: list[str] | None = None) -> set[str]:
    allowed = {"127.0.0.1", "localhost", "::1"}
    lh = _strip_host_port(listen_host)
    if lh and lh not in ("0.0.0.0", "::"):
        allowed.add(lh)
    env_hosts = (os.environ.get("DIRECTOR_ALLOWED_HOSTS") or "").strip()
    for token in env_hosts.split(","):
        cleaned = _strip_host_port(token)
        if cleaned:
            allowed.add(cleaned)
    for token in extra_hosts or ():
        cleaned = _strip_host_port(token)
        if cleaned:
            allowed.add(cleaned)
    ts_dns = _discover_tailscale_host()
    if ts_dns:
        allowed.add(ts_dns)
    return allowed


def _is_https_request(request: web.Request) -> bool:
    if request.secure:
        return True
    proto = (request.headers.get("X-Forwarded-Proto") or "").strip().lower()
    return proto == "https"


def _apply_security_headers(
    response: web.StreamResponse,
    *,
    sid_to_set: str | None = None,
    secure_cookie: bool = False,
) -> web.StreamResponse:
    response.headers["Content-Security-Policy"] = CSP_POLICY
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers.setdefault("Cache-Control", "no-store")
    if sid_to_set:
        response.set_cookie(
            COOKIE_NAME,
            sid_to_set,
            httponly=True,
            samesite="Strict",
            secure=secure_cookie,
            path="/",
        )
    return response


def _resolve_output_dir() -> Path:
    env_out = (os.environ.get("H3_OUTPUT_DIR") or "").strip()
    if env_out:
        return Path(env_out)
    from workflows.h3_config import output_dir as cfg_output_dir

    return Path(cfg_output_dir())


def _backbone_for_image_workflow(workflow: str | None) -> str:
    wf = str(workflow or "").strip().lower()
    if wf == "pika_image":
        return "pika"
    if wf == "ph3_image":
        return "ph3_image"
    return "h3_image"


async def _comfy_get_json(app: web.Application, path: str, *, timeout: float = 3.0) -> Any | None:
    session: ClientSession = app["http_client"]
    base: str = app["comfy_url"]
    try:
        async with session.get(f"{base}{path}", timeout=ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                return None
            return await resp.json()
    except Exception:
        return None


async def _comfy_post_json(app: web.Application, path: str, payload: dict[str, Any], *, timeout: float = 15.0) -> tuple[int, Any]:
    session: ClientSession = app["http_client"]
    base: str = app["comfy_url"]
    async with session.post(
        f"{base}{path}",
        json=payload,
        timeout=ClientTimeout(total=timeout),
    ) as resp:
        text = await resp.text()
        try:
            data = json.loads(text) if text else {}
        except Exception:
            data = {"raw": text}
        return resp.status, data


async def _comfy_upload_bytes(
    app: web.Application,
    raw_bytes: bytes,
    filename: str,
    *,
    subfolder: str = UPLOADS_SUBFOLDER,
    content_type: str = "image/png",
) -> dict[str, Any]:
    session: ClientSession = app["http_client"]
    base: str = app["comfy_url"]
    form = FormData()
    form.add_field("image", raw_bytes, filename=filename, content_type=content_type)
    form.add_field("subfolder", subfolder)
    form.add_field("type", "input")
    form.add_field("overwrite", "true")
    async with session.post(f"{base}/upload/image", data=form, timeout=ClientTimeout(total=30.0)) as resp:
        body = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"ComfyUI /upload/image failed ({resp.status}): {body[:300]}")
        return json.loads(body)


async def _comfy_fetch_view(app: web.Application, desc: dict[str, str]) -> tuple[int, bytes, str]:
    session: ClientSession = app["http_client"]
    base: str = app["comfy_url"]
    qs = urlencode(desc)
    async with session.get(f"{base}/view?{qs}", timeout=ClientTimeout(total=60.0)) as resp:
        data = await resp.read()
        ctype = resp.headers.get("Content-Type") or mimetypes.guess_type(desc["filename"])[0] or "application/octet-stream"
        return resp.status, data, ctype


async def _fetch_image_with_temp_fallback(
    app: web.Application,
    spec: Any,
    *,
    default_type: str = "input",
) -> tuple[dict[str, str], int, bytes]:
    desc = _parse_view_descriptor(spec, default_type=default_type)
    status, raw_bytes, _ = await _comfy_fetch_view(app, desc)
    if (
        status == 404
        and desc["type"] == "input"
        and not desc["subfolder"]
        and isinstance(spec, str)
        and not spec.strip().startswith("/api/view")
    ):
        temp_desc = {**desc, "type": "temp"}
        t_status, t_bytes, _ = await _comfy_fetch_view(app, temp_desc)
        if t_status == 200 and t_bytes:
            return temp_desc, t_status, t_bytes
    return desc, status, raw_bytes


async def _materialize_input_image_ref(
    app: web.Application,
    spec: Any,
    *,
    sid: str,
    prefix: str = "anchor",
) -> tuple[str, bytes, dict[str, str]]:
    """Fetch ``spec`` and promote non-``input`` images (e.g. ``temp`` anchor stills) into ``input/h3_spatial_director/``."""
    desc, status, raw_bytes = await _fetch_image_with_temp_fallback(app, spec, default_type="input")
    if status != 200 or not raw_bytes:
        raise FileNotFoundError(f"source image unavailable ({status})")
    if desc["type"] == "input":
        ref = f"{desc['subfolder']}/{desc['filename']}" if desc["subfolder"] else desc["filename"]
        return ref, raw_bytes, desc
    store: SessionStore = app["store"]
    digest = hashlib.sha1(raw_bytes).hexdigest()[:8]
    stem = Path(desc["filename"]).stem
    ext = Path(desc["filename"]).suffix.lower() or ".png"
    in_name = _sanitize_filename(f"{prefix}_{stem}_{digest}{ext}", default_ext=".png")
    ctype = "image/png" if ext == ".png" else (mimetypes.guess_type(in_name)[0] or "application/octet-stream")
    up_res = await _comfy_upload_bytes(
        app,
        raw_bytes,
        in_name,
        subfolder=UPLOADS_SUBFOLDER,
        content_type=ctype,
    )
    up_sub = str(up_res.get("subfolder") or UPLOADS_SUBFOLDER)
    up_name = str(up_res.get("name") or in_name)
    in_ref = f"{up_sub}/{up_name}" if up_sub else up_name
    store.record_upload(in_ref, sid)
    return in_ref, raw_bytes, {"filename": up_name, "subfolder": up_sub, "type": "input"}


async def _probe_heylook_url(app: web.Application) -> str | None:
    raw_url = (os.environ.get("H3_HEYLOOK_URL") or "").strip().rstrip("/")
    if not raw_url:
        return None
    session: ClientSession = app["http_client"]
    try:
        async with session.get(f"{raw_url}/v1/models", timeout=ClientTimeout(total=0.8)) as resp:
            return raw_url if resp.status == 200 else None
    except Exception:
        return None


def _get_session_id(request: web.Request) -> tuple[str, bool]:
    existing = (request.cookies.get(COOKIE_NAME) or "").strip()
    if _SAFE_SID_RE.match(existing):
        return existing, False
    return secrets.token_urlsafe(24), True


async def handle_index(request: web.Request) -> web.StreamResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        return web.Response(
            text="Interactive Spatial Director static frontend is not installed yet.",
            status=503,
            content_type="text/plain",
        )
    return web.FileResponse(index_path)


def _resolve_safe_doc_path(
    raw_rel: str,
    *,
    allowed_exts: set[str] | frozenset[str],
    must_be_under_docs: bool = False,
) -> tuple[Path, str]:
    norm = str(raw_rel or "").replace("\\", "/").strip()
    if not norm or "\x00" in norm or norm.startswith("/") or re.match(r"^[A-Za-z]:", norm):
        raise ValueError("Invalid documentation path")
    parts = [p for p in norm.split("/") if p]
    if not parts or any(p in (".", "..") or p.startswith(".") or p in _DOC_SKIP_DIRS for p in parts):
        raise ValueError("Disallowed documentation path segment")
    canon_rel = "/".join(parts)
    if Path(canon_rel).suffix.lower() not in allowed_exts:
        raise ValueError(f"Unsupported documentation file extension for {canon_rel!r}")

    repo_root_res = _REPO_ROOT.resolve()
    cand = (_REPO_ROOT / canon_rel).resolve()
    if not cand.is_relative_to(repo_root_res):
        raise ValueError("Documentation path escapes repository root")
    resolved_rel = cand.relative_to(repo_root_res).as_posix()
    resolved_parts = Path(resolved_rel).parts
    if any(p.startswith(".") or p in _DOC_SKIP_DIRS for p in resolved_parts):
        raise ValueError("Documentation path resolves into disallowed folder")
    if must_be_under_docs:
        docs_root = (_REPO_ROOT / "docs").resolve()
        if not cand.is_relative_to(docs_root):
            raise ValueError("Documentation asset must reside under docs/")
    if not cand.is_file():
        raise FileNotFoundError(canon_rel)
    return cand, resolved_rel


def _extract_md_title(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for idx, line in enumerate(fh):
                if idx >= 40:
                    break
                s = line.strip()
                if s.startswith("# "):
                    title = s[2:].strip()
                    title = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", title)
                    title = title.replace("`", "").strip()
                    if title:
                        return title[:140]
    except Exception:
        pass
    return path.stem


def _categorize_doc_rel(rel: str) -> str:
    if rel in _PINNED_STARTER_DOCS:
        return "1. Spatial Director & Steering"
    top_routers = {"AGENTS.md", "VISION.md", "README.md", "INDEX.md", "docs/README.md", "docs/AGENTS.md"}
    if rel in top_routers or (rel.startswith("docs/wiki/") and "/" not in rel[len("docs/wiki/"):]):
        return "2. Overview & Wiki Routers"
    if rel.startswith("docs/wiki/core/"):
        return "3. Core Video, Clocks & Audio"
    if rel.startswith("docs/wiki/image/"):
        return "4. Single-Frame Image"
    if rel.startswith("docs/wiki/steering/"):
        return "5. Steering & J-Lens"
    if rel.startswith("docs/wiki/attention/"):
        return "6. Attention, Sol & Quant"
    if rel.startswith("docs/wiki/prompting/") or rel.startswith("docs/wiki/system_prompts/"):
        return "7. Prompting & System Prompts"
    if rel.startswith("docs/wiki/masked_v2v/"):
        return "8. Masked V2V Pipeline"
    if rel.startswith("docs/wiki/upstream/"):
        return "9. Upstream Engines & Sweeps"
    if rel.startswith("docs/wiki/ops/"):
        return "10. Operations & Checks"
    return "11. Projects, Research & Bench"


def _doc_sort_key(rel: str) -> tuple[Any, ...]:
    if rel in _PINNED_STARTER_DOCS:
        return (0, _PINNED_STARTER_DOCS.index(rel), rel)
    name = Path(rel).name
    lower_name = name.lower()
    if lower_name in ("agents.md", "index.md", "readme.md"):
        return (0, 0, rel)
    m_date = re.match(r"^(\d{4}-\d{2}-\d{2})", name)
    if m_date:
        # Reverse-chronological for dated notes.
        inv_date = "".join(chr(255 - ord(c)) for c in m_date.group(1))
        return (1, inv_date, rel)
    return (2, rel.lower(), rel)


def _collect_docs_catalog(query: str = "") -> dict[str, Any]:
    repo_root_res = _REPO_ROOT.resolve()
    candidates: list[Path] = []
    candidates.extend(_REPO_ROOT.glob("*.md"))
    for folder, pattern in (
        (_REPO_ROOT / "director", "*.md"),
        (_REPO_ROOT / "docs", "**/*.md"),
        (_REPO_ROOT / "bench", "*.md"),
        (_REPO_ROOT / "bench" / "results", "*.md"),
        (_REPO_ROOT / "workflows", "*.md"),
    ):
        if folder.is_dir():
            candidates.extend(folder.glob(pattern))

    seen: set[str] = set()
    entries: list[dict[str, Any]] = []
    for p in candidates:
        if not p.is_file():
            continue
        try:
            res_p = p.resolve()
            if not res_p.is_relative_to(repo_root_res):
                continue
            rel = res_p.relative_to(repo_root_res).as_posix()
        except Exception:
            continue
        parts = Path(rel).parts
        if any(part.startswith(".") or part in _DOC_SKIP_DIRS for part in parts):
            continue
        if rel in seen:
            continue
        seen.add(rel)
        entries.append(
            {
                "path": rel,
                "title": _extract_md_title(res_p),
                "group": _categorize_doc_rel(rel),
                "pinned": rel in _PINNED_STARTER_DOCS,
            }
        )

    by_group: dict[str, list[dict[str, Any]]] = {g: [] for g in _DOC_GROUP_ORDER}
    for ent in entries:
        by_group.setdefault(ent["group"], []).append(ent)

    groups: list[dict[str, Any]] = []
    ordered_items: list[dict[str, Any]] = []
    for gname in _DOC_GROUP_ORDER:
        items = sorted(by_group.get(gname, []), key=lambda it: _doc_sort_key(it["path"]))
        if items:
            groups.append({"group": gname, "items": items})
            ordered_items.extend(items)

    q_clean = str(query or "").strip()
    results: list[dict[str, Any]] = []
    if q_clean:
        terms = [t.lower() for t in q_clean.split() if t.strip()]
        scored: list[tuple[int, dict[str, Any]]] = []
        for ent in ordered_items:
            p = _REPO_ROOT / ent["path"]
            try:
                raw_text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            title_l = ent["title"].lower()
            path_l = ent["path"].lower()
            body_l = raw_text.lower()
            if not all((t in title_l or t in path_l or t in body_l) for t in terms):
                continue
            score = 0
            if ent["pinned"]:
                score -= 20
            for t in terms:
                if t in title_l:
                    score -= 10
                if t in path_l:
                    score -= 5
            snippets: list[str] = []
            for raw_line in raw_text.splitlines():
                stripped = raw_line.strip()
                if not stripped or set(stripped) <= set("|-:= "):
                    continue
                low_line = stripped.lower()
                if any(t in low_line for t in terms):
                    snippets.append(stripped[:160])
                    if len(snippets) >= 2:
                        break
            scored.append(
                (
                    score,
                    {
                        "path": ent["path"],
                        "title": ent["title"],
                        "group": ent["group"],
                        "snippets": snippets,
                    },
                )
            )
        scored.sort(key=lambda pair: (pair[0], _doc_sort_key(pair[1]["path"])))
        results = [item for _, item in scored[:60]]

    return {
        "total": len(ordered_items),
        "groups": groups,
        "items": ordered_items,
        "query": q_clean,
        "results": results,
    }


async def handle_docs_page(request: web.Request) -> web.StreamResponse:
    docs_path = STATIC_DIR / "docs.html"
    if not docs_path.is_file():
        return web.Response(
            text="Documentation hub page is not installed yet.",
            status=404,
            content_type="text/plain",
        )
    return web.FileResponse(docs_path)


async def handle_docs_index(request: web.Request) -> web.Response:
    q = str(request.rel_url.query.get("q") or "")
    catalog = _collect_docs_catalog(query=q)
    return web.json_response(
        {
            "ok": True,
            "default_doc": DEFAULT_DOC_PATH,
            **catalog,
        }
    )


async def handle_docs_file(request: web.Request) -> web.Response:
    raw_path = str(request.rel_url.query.get("path") or DEFAULT_DOC_PATH)
    try:
        target, rel = _resolve_safe_doc_path(raw_path, allowed_exts={".md"})
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except FileNotFoundError:
        return web.json_response({"ok": False, "error": "Document not found"}, status=404)

    raw_bytes = target.read_bytes()
    text = raw_bytes.decode("utf-8", errors="replace")
    st = target.stat()
    return web.json_response(
        {
            "ok": True,
            "path": rel,
            "title": _extract_md_title(target),
            "group": _categorize_doc_rel(rel),
            "markdown": text,
            "bytes": len(raw_bytes),
            "mtime": st.st_mtime,
        }
    )


async def handle_docs_asset(request: web.Request) -> web.StreamResponse:
    raw_path = str(request.rel_url.query.get("path") or "")
    try:
        target, _rel = _resolve_safe_doc_path(raw_path, allowed_exts=_DOC_IMAGE_EXTS, must_be_under_docs=True)
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except FileNotFoundError:
        return web.Response(status=404, text="not found")

    ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    if target.suffix.lower() == ".svg":
        ctype = "image/svg+xml"
    return web.Response(
        body=target.read_bytes(),
        status=200,
        content_type=ctype,
        headers={"Cache-Control": "no-store"},
    )


async def handle_static(request: web.Request) -> web.StreamResponse:
    tail = request.match_info.get("tail", "")
    if not tail or "\x00" in tail:
        raise web.HTTPNotFound()
    static_root = STATIC_DIR.resolve()
    if not static_root.is_dir():
        raise web.HTTPNotFound()
    candidate = (static_root / tail).resolve()
    if candidate == static_root or not candidate.is_relative_to(static_root) or not candidate.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(candidate)


async def handle_schema(request: web.Request) -> web.Response:
    app = request.app
    obj_info = await _fetch_object_info(app)
    heylook_url = await _probe_heylook_url(app)
    data = compiler.get_schema_data(obj_info, heylook_suggestions=bool(heylook_url))
    return web.json_response(data)


async def handle_system_stats(request: web.Request) -> web.Response:
    app = request.app
    stats = await _comfy_get_json(app, "/system_stats", timeout=2.5)
    reachable = isinstance(stats, dict)
    try:
        out_dir = _resolve_output_dir()
        out_label = f"{out_dir.name}/Video"
    except Exception:
        out_label = "Video"
    store: SessionStore = app["store"]
    try:
        state_label = str(store.state_dir.relative_to(_REPO_ROOT))
    except ValueError:
        state_label = store.state_dir.name
    return web.json_response(
        {
            "ok": True,
            "comfy_reachable": reachable,
            "comfy_url": app["comfy_url"],
            "system": (stats or {}).get("system") or {},
            "devices": (stats or {}).get("devices") or [],
            "output_dir": out_label,
            "state_dir": state_label,
            "resident_backbone": _get_resident_backbone(app),
        }
    )


async def handle_upload(request: web.Request) -> web.Response:
    sid: str = request["director_sid"]
    reader = await request.multipart()
    file_bytes: bytes | None = None
    filename = "upload.png"
    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name in ("file", "image"):
            filename = part.filename or filename
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = await part.read_chunk(65536)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    return web.json_response({"ok": False, "error": "upload_too_large"}, status=413)
                chunks.append(chunk)
            file_bytes = b"".join(chunks)
            break
    if not file_bytes:
        return web.json_response({"ok": False, "error": "missing_file"}, status=400)

    try:
        out_bytes, w, h, out_name, source_kind = _decode_frame0_or_image(file_bytes, filename)
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)

    ctype = mimetypes.guess_type(out_name)[0] or "image/png"
    try:
        res = await _comfy_upload_bytes(request.app, out_bytes, out_name, subfolder=UPLOADS_SUBFOLDER, content_type=ctype)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=502)

    saved_name = str(res.get("name") or out_name)
    saved_sub = str(res.get("subfolder") or UPLOADS_SUBFOLDER)
    saved_type = str(res.get("type") or "input")
    ref = f"{saved_sub}/{saved_name}" if saved_sub else saved_name
    store: SessionStore = request.app["store"]
    store.record_upload(ref, sid)

    resp_data: dict[str, Any] = {
        "ok": True,
        "filename": saved_name,
        "subfolder": saved_sub,
        "type": saved_type,
        "ref": ref,
        "width": w,
        "height": h,
        "view_url": _view_url(saved_name, saved_sub, saved_type),
        "source_kind": source_kind,
        "original_filename": Path(filename).name,
    }

    if source_kind == "video_frame0":
        vid_ext = Path(filename).suffix.lower() or ".mp4"
        vid_out_name = _sanitize_filename(Path(filename).stem, default_ext=vid_ext)
        vid_ctype = mimetypes.guess_type(vid_out_name)[0] or "video/mp4"
        try:
            _, _, total_frames = _decode_video_frame_rgb(file_bytes, filename, frame_index=-1)
        except Exception:
            total_frames = 39
        has_audio = _video_bytes_has_audio(file_bytes, filename)
        snapped_len = compiler.snap_to_on_both_clocks(max(39, int(total_frames)))
        try:
            v_res = await _comfy_upload_bytes(
                request.app,
                file_bytes,
                vid_out_name,
                subfolder=UPLOADS_SUBFOLDER,
                content_type=vid_ctype,
            )
            v_saved_name = str(v_res.get("name") or vid_out_name)
            v_saved_sub = str(v_res.get("subfolder") if "subfolder" in v_res else UPLOADS_SUBFOLDER).strip("/")
            v_saved_type = str(v_res.get("type") or "input")
            vid_ref = f"{v_saved_sub}/{v_saved_name}" if v_saved_sub else v_saved_name
            store.record_upload(vid_ref, sid)
            take_id = f"import_{secrets.token_hex(6)}"
            vid_desc = {"filename": v_saved_name, "subfolder": v_saved_sub, "type": v_saved_type}
            store.record_take(
                {
                    "take_id": take_id,
                    "kind": "imported",
                    "prompt_id": f"import_{secrets.token_hex(6)}",
                    "baseline_prompt_id": None,
                    "client_id": "",
                    "session_id": sid,
                    "status": "completed",
                    "mode": "imported_video",
                    "director_mode": "portal_path",
                    "length": snapped_len,
                    "source_frames": int(total_frames),
                    "has_audio": bool(has_audio),
                    "width": w,
                    "height": h,
                    "seed": 0,
                    "params": {"video_ref": vid_ref, "first_frame": ref, "length": snapped_len, "width": w, "height": h},
                    "clean_video": vid_desc,
                    "hud_video": None,
                    "baseline_video": None,
                    "created_at": time.time(),
                    "updated_at": time.time(),
                }
            )
            resp_data.update(
                {
                    "video_ref": vid_ref,
                    "video_filename": v_saved_name,
                    "video_subfolder": v_saved_sub,
                    "video_type": v_saved_type,
                    "video_view_url": _view_url(v_saved_name, v_saved_sub, v_saved_type),
                    "imported_take_id": take_id,
                    "total_frames": int(total_frames),
                    "snapped_length": snapped_len,
                    "has_audio": bool(has_audio),
                }
            )
        except Exception as exc:
            log.warning("Could not register uploaded MP4 as imported take (%s): %s", filename, exc)

    return web.json_response(resp_data)


async def _read_json_body(request: web.Request) -> dict[str, Any]:
    raw = await request.text()
    if not raw or not raw.strip():
        return {}
    doc = json.loads(raw)
    if not isinstance(doc, dict):
        raise ValueError("Request body must be a JSON object")
    return doc


_SCHEMA_OBJECT_INFO_CLASSES: tuple[str, ...] = (
    compiler.PREVIEW_NODE_CLASS,
    "MiniMaxH3SpatialDirector",
    "MiniMaxH3GuideLatent",
    "MiniMaxH3AudioRefineMask",
    "MiniMaxH3FrozenVideoCache",
    "SAM3_Detect",
    "MiniMaxH3SAM31Corrections",
    "CheckpointLoaderSimple",
)


async def _fetch_object_info(app: web.Application) -> dict[str, Any] | None:
    results = await asyncio.gather(
        *(_comfy_get_json(app, f"/object_info/{cls}", timeout=2.0) for cls in _SCHEMA_OBJECT_INFO_CLASSES),
        return_exceptions=True,
    )
    merged: dict[str, Any] = {}
    for item in results:
        if isinstance(item, dict):
            merged.update(item)
    if merged:
        return merged
    return await _comfy_get_json(app, "/object_info", timeout=3.0)


async def _submit_comfy_prompt(
    app: web.Application,
    graph: dict[str, Any],
    *,
    client_id: str,
    extra_meta: dict[str, Any] | None = None,
) -> str:
    extra = {APP_MARK: True, **(extra_meta or {})}
    payload = {"prompt": graph, "client_id": client_id, "extra_data": extra}
    status, data = await _comfy_post_json(app, "/prompt", payload, timeout=15.0)
    if status != 200 or not isinstance(data, dict) or not data.get("prompt_id"):
        raise RuntimeError(f"ComfyUI /prompt rejected graph ({status}): {data}")
    return str(data["prompt_id"])


async def handle_anchor(request: web.Request) -> web.Response:
    try:
        body = await _read_json_body(request)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)

    sid: str = request["director_sid"]
    client_id = str(body.get("client_id") or secrets.token_hex(8))
    obj_info = await _fetch_object_info(request.app)
    try:
        anchor_cfg = {
            **body,
            "workflow": str(body.get("workflow") or "h3_image"),
            "for_cutout": False,
        }
        compiled = compiler.compile_still(anchor_cfg, object_info=obj_info)
        prompt_id = await _submit_comfy_prompt(
            request.app,
            compiled["graph"],
            client_id=client_id,
            extra_meta={"kind": "anchor"},
        )
        _set_resident_backbone(request.app, _backbone_for_image_workflow(compiled["workflow"]))
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=502)

    take_id = f"anchor_{secrets.token_hex(6)}"
    request.app["store"].record_take(
        {
            "take_id": take_id,
            "kind": "anchor",
            "prompt_id": prompt_id,
            "baseline_prompt_id": None,
            "client_id": client_id,
            "session_id": sid,
            "status": "queued",
            "mode": compiled["workflow"],
            "seed": compiled["seed"],
            **_extract_lineage_fields(body),
            "params": body,
            "still_image": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
    )
    return web.json_response(
        {
            "ok": True,
            "take_id": take_id,
            "kind": "anchor",
            "prompt_id": prompt_id,
            "client_id": client_id,
            "workflow": compiled["workflow"],
            "seed": compiled["seed"],
            "preview_override_injected": compiled["preview_override_injected"],
            "graph": compiled["graph"],
        }
    )


async def handle_preview_in_mask(request: web.Request) -> web.Response:
    import numpy as np
    import torch
    from PIL import Image

    try:
        body = await _read_json_body(request)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)

    sid: str = request["director_sid"]
    source_spec = body.get("source_ref") or body.get("ref_image") or body.get("anchor_image")
    if not source_spec:
        return web.json_response({"ok": False, "error": "source_ref is required"}, status=400)
    target_prompt = str(body.get("target_prompt") or body.get("prompt") or "").strip()
    if not target_prompt:
        return web.json_response({"ok": False, "error": "target_prompt cannot be empty"}, status=400)

    try:
        source_ref_norm, src_bytes, src_desc = await _materialize_input_image_ref(
            request.app, source_spec, sid=sid, prefix="anchor"
        )
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except FileNotFoundError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=404)

    with Image.open(io.BytesIO(src_bytes)) as src_im:
        w, h = max(32, int(src_im.width)), max(32, int(src_im.height))

    region = body.get("region") if isinstance(body.get("region"), dict) else {}
    fx = float(region.get("x", body.get("x", 0.5)))
    fy = float(region.get("y", body.get("y", 0.5)))
    rx = max(0.03, min(0.85, float(region.get("rx", body.get("rx", 0.24)))))
    ry = max(0.03, min(0.85, float(region.get("ry", body.get("ry", 0.28)))))
    shape = str(region.get("shape", body.get("shape", "circle")) or "circle")
    weight = max(0.05, min(1.0, float(region.get("weight", body.get("weight", 1.0)))))
    halo_radius = max(0.0, min(0.5, float(region.get("halo_radius", body.get("halo_radius", 0.0)))))
    halo_weight = max(0.0, min(1.0, float(region.get("halo_weight", body.get("halo_weight", 0.30)))))
    feather = max(0.0, min(0.35, float(body.get("feather", 0.08))))
    invert_mask = bool(body.get("invert_mask", False))
    snap_to_include = bool(body.get("snap_to_include_mask", False))

    include_mask_spec = body.get("include_mask_ref")
    exclude_mask_spec = body.get("exclude_mask_ref")

    async def _fetch_mask_array(spec: Any) -> Any | None:
        if not spec:
            return None
        try:
            m_desc = _parse_view_descriptor(spec, default_type="input")
            ms, mb, _ = await _comfy_fetch_view(request.app, m_desc)
            if ms != 200 or not mb:
                return None
            with Image.open(io.BytesIO(mb)) as m_im:
                l_im = m_im.convert("L")
                if l_im.size != (w, h):
                    l_im = l_im.resize((w, h), Image.Resampling.BILINEAR)
                return np.asarray(l_im, dtype=np.float32) / 255.0
        except Exception:
            return None

    mask_f32: Any | None = None
    if snap_to_include and include_mask_spec:
        mask_f32 = await _fetch_mask_array(include_mask_spec)

    if mask_f32 is None:
        try:
            spatial_director = compiler.import_pack_module("spatial_director")
            ev5 = {
                "portal": [
                    {
                        "frame": 0,
                        "x": fx,
                        "y": fy,
                        "rx": rx,
                        "ry": ry,
                        "shape": shape if shape in ("circle", "ellipse", "vertical_split") else "circle",
                        "label": "PREVIEW",
                        "weight": weight,
                        "halo_radius": halo_radius,
                        "halo_weight": halo_weight,
                    },
                    {
                        "frame": 4,
                        "x": fx,
                        "y": fy,
                        "rx": rx,
                        "ry": ry,
                        "shape": shape if shape in ("circle", "ellipse", "vertical_split") else "circle",
                        "label": "PREVIEW",
                        "weight": weight,
                        "halo_radius": halo_radius,
                        "halo_weight": halo_weight,
                    },
                ]
            }
            out_t = spatial_director.MiniMaxH3SpatialDirector.execute(
                mode="portal_path",
                width=w,
                height=h,
                length=5,
                feather=feather,
                invert_mask=invert_mask,
                events_json=json.dumps(ev5),
            )[0]
            mask_f32 = out_t[0].detach().cpu().numpy().astype(np.float32)
        except Exception:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            short_side = float(min(w, h))
            cx, cy = fx * float(w), fy * float(h)
            if shape == "ellipse":
                d = np.sqrt(((xx - cx) / max(4.0, rx * w)) ** 2 + ((yy - cy) / max(4.0, ry * h)) ** 2) - 1.0
                mask_f32 = np.clip(0.5 - d / max(0.05, feather * 2.0), 0.0, 1.0)
            else:
                rad = max(4.0, rx * short_side)
                d = (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - rad) / max(2.0, feather * short_side)
                mask_f32 = np.clip(0.5 - d, 0.0, 1.0)
            if invert_mask:
                mask_f32 = 1.0 - mask_f32

    if include_mask_spec and not snap_to_include:
        inc_arr = await _fetch_mask_array(include_mask_spec)
        if inc_arr is not None:
            mask_f32 = np.maximum(mask_f32, inc_arr)

    excl_arr = await _fetch_mask_array(exclude_mask_spec)
    if excl_arr is not None:
        mask_f32 = np.clip(mask_f32 * (1.0 - excl_arr), 0.0, 1.0)

    if float(np.max(mask_f32)) <= 1e-3:
        return web.json_response({"ok": False, "error": "Active mask region is empty"}, status=400)

    store: SessionStore = request.app["store"]
    try:
        mask_png_bytes, _ = _encode_mask_rgb_png(mask_f32)
        mask_name = _sanitize_filename(
            f"preview_mask_{Path(src_desc['filename']).stem}_{secrets.token_hex(3)}.png",
            default_ext=".png",
        )
        up_res = await _comfy_upload_bytes(
            request.app,
            mask_png_bytes,
            mask_name,
            subfolder=UPLOADS_SUBFOLDER,
            content_type="image/png",
        )
        m_saved = str(up_res.get("name") or mask_name)
        m_sub = str(up_res.get("subfolder") or UPLOADS_SUBFOLDER)
        m_type = str(up_res.get("type") or "input")
        mask_ref = f"{m_sub}/{m_saved}" if m_sub else m_saved
        store.record_upload(mask_ref, sid)
    except Exception as exc:
        return web.json_response({"ok": False, "error": f"Failed to stage mask image: {exc}"}, status=502)

    client_id = str(body.get("client_id") or secrets.token_hex(8))
    obj_info = await _fetch_object_info(request.app)
    req_workflow = str(body.get("workflow") or body.get("preview_workflow") or "auto").strip()
    compile_cfg = {
        **body,
        "source_ref": source_ref_norm,
        "mask_ref": mask_ref,
        "target_prompt": target_prompt,
        "region_x": fx,
        "region_y": fy,
        "workflow": req_workflow,
        "resident_backbone": _get_resident_backbone(request.app),
    }
    try:
        compiled = compiler.compile_masked_still_edit(compile_cfg, object_info=obj_info)
        prompt_id = await _submit_comfy_prompt(
            request.app,
            compiled["graph"],
            client_id=client_id,
            extra_meta={"kind": "mask_preview"},
        )
        _set_resident_backbone(request.app, _backbone_for_image_workflow(compiled["workflow"]))
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=502)

    take_id = f"mprev_{secrets.token_hex(6)}"
    store.record_take(
        {
            "take_id": take_id,
            "kind": "mask_preview",
            "prompt_id": prompt_id,
            "baseline_prompt_id": None,
            "client_id": client_id,
            "session_id": sid,
            "status": "queued",
            "mode": compiled["workflow"],
            "seed": compiled["seed"],
            **_extract_lineage_fields(body),
            "params": compile_cfg,
            "still_image": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
    )
    return web.json_response(
        {
            "ok": True,
            "take_id": take_id,
            "kind": "mask_preview",
            "prompt_id": prompt_id,
            "client_id": client_id,
            "workflow": compiled["workflow"],
            "requested_workflow": req_workflow,
            "resident_backbone": _get_resident_backbone(request.app),
            "seed": compiled["seed"],
            "mask_ref": mask_ref,
            "mask_view_url": _view_url(m_saved, m_sub, m_type),
            "preview_override_injected": compiled["preview_override_injected"],
            "original_prompt": compiled["original_prompt"],
            "edited_prompt": compiled["edited_prompt"],
            "resolved_source_ref": source_ref_norm,
        }
    )


def _extract_rgba_cutout_bytes(raw_bytes: bytes, mask_b64: str | None = None) -> tuple[bytes, int, int, float]:
    cut_png, _mask_png, cw, ch, _bbox, coverage = extract_studio_cutout_rgba(raw_bytes, mask_b64)
    return cut_png, cw, ch, coverage


async def handle_stamp_cutout(request: web.Request) -> web.Response:
    try:
        body = await _read_json_body(request)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)

    sid: str = request["director_sid"]
    prompt_str = str(body.get("prompt") or "").strip()
    source_spec = body.get("source_ref") or body.get("source_image") or body.get("image_output")
    # Only extract RGBA directly when caller passed an image_output OR no prompt was specified;
    # when a text prompt is entered on stage, generate a dedicated T2I cutout (`for_cutout=True`)
    # even if an anchor image is loaded on stage.
    if source_spec and (body.get("image_output") or not prompt_str):
        try:
            desc, status, raw_bytes = await _fetch_image_with_temp_fallback(request.app, source_spec, default_type="input")
            if status != 200 or not raw_bytes:
                return web.json_response({"ok": False, "error": f"source image unavailable ({status})"}, status=404)
            png_bytes, mask_bytes, w, h, bbox_norm, coverage = await asyncio.to_thread(
                extract_studio_cutout_rgba,
                raw_bytes,
                body.get("mask"),
            )
            stem = Path(desc["filename"]).stem
            out_name = _sanitize_filename(f"stamp_{stem}.png", default_ext=".png")
            mask_name = _sanitize_filename(f"stamp_mask_{stem}.png", default_ext=".png")
            res, mask_res = await asyncio.gather(
                _comfy_upload_bytes(request.app, png_bytes, out_name, subfolder=UPLOADS_SUBFOLDER, content_type="image/png"),
                _comfy_upload_bytes(request.app, mask_bytes, mask_name, subfolder=UPLOADS_SUBFOLDER, content_type="image/png"),
            )
        except ValueError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=502)

        saved_name = str(res.get("name") or out_name)
        saved_sub = str(res.get("subfolder") or UPLOADS_SUBFOLDER)
        saved_type = str(res.get("type") or "input")
        ref = f"{saved_sub}/{saved_name}" if saved_sub else saved_name
        mask_saved = str(mask_res.get("name") or mask_name)
        mask_sub = str(mask_res.get("subfolder") or UPLOADS_SUBFOLDER)
        mask_ref = f"{mask_sub}/{mask_saved}" if mask_sub else mask_saved
        request.app["store"].record_upload(ref, sid)
        request.app["store"].record_upload(mask_ref, sid)
        return web.json_response(
            {
                "ok": True,
                "kind": "cutout",
                "cutout_ref": ref,
                "cutout_mask_ref": mask_ref,
                "bbox": bbox_norm,
                "filename": saved_name,
                "subfolder": saved_sub,
                "type": saved_type,
                "width": w,
                "height": h,
                "alpha_coverage": round(coverage, 4),
                "view_url": _view_url(saved_name, saved_sub, saved_type),
                "mask_view_url": _view_url(mask_saved, mask_sub, str(mask_res.get("type") or "input")),
            }
        )

    client_id = str(body.get("client_id") or secrets.token_hex(8))
    obj_info = await _fetch_object_info(request.app)
    try:
        stamp_cfg = {
            **body,
            "workflow": str(body.get("workflow") or "h3_image"),
            "for_cutout": True,
        }
        compiled = compiler.compile_still(stamp_cfg, object_info=obj_info)
        prompt_id = await _submit_comfy_prompt(
            request.app,
            compiled["graph"],
            client_id=client_id,
            extra_meta={"kind": "stamp"},
        )
        _set_resident_backbone(request.app, _backbone_for_image_workflow(compiled["workflow"]))
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=502)

    take_id = f"stamp_{secrets.token_hex(6)}"
    request.app["store"].record_take(
        {
            "take_id": take_id,
            "kind": "stamp",
            "prompt_id": prompt_id,
            "baseline_prompt_id": None,
            "client_id": client_id,
            "session_id": sid,
            "status": "queued",
            "mode": compiled["workflow"],
            "seed": compiled["seed"],
            "params": body,
            "still_image": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
    )
    return web.json_response(
        {
            "ok": True,
            "take_id": take_id,
            "kind": "stamp",
            "prompt_id": prompt_id,
            "client_id": client_id,
            "workflow": compiled["workflow"],
            "seed": compiled["seed"],
            "preview_override_injected": compiled["preview_override_injected"],
            "graph": compiled["graph"],
        }
    )


_VALID_SEGMENT_ROLES = {"cutout", "exclude", "include"}


async def _run_sam3_segment_live(
    app: web.Application,
    *,
    sid: str,
    input_ref: str,
    width: int,
    height: int,
    params: dict[str, Any],
    obj_info: dict[str, Any],
) -> tuple[Any, float, str] | None:
    import numpy as np
    from PIL import Image

    seg_params = dict(params)
    seg_params["image_ref"] = input_ref
    compiled = compiler.compile_sam3_segment(
        seg_params,
        image_width=width,
        image_height=height,
        object_info=obj_info,
    )
    seg_cid = f"seg_{secrets.token_hex(6)}"
    prompt_id = await _submit_comfy_prompt(
        app,
        compiled["graph"],
        client_id=seg_cid,
        extra_meta={"kind": "segment", "session_id": sid},
    )

    output_node = str(compiled.get("output_node_id") or "208")
    out_desc: dict[str, Any] | None = None
    try:
        deadline = time.monotonic() + 75.0
        while time.monotonic() < deadline:
            hist = await _comfy_get_json(app, f"/history/{prompt_id}", timeout=3.0)
            entry = hist.get(prompt_id) if isinstance(hist, dict) else None
            if isinstance(entry, dict):
                status_str = str((entry.get("status") or {}).get("status_str") or "")
                if status_str == "error":
                    return None
                outputs = entry.get("outputs") or {}
                node_out = outputs.get(output_node)
                if isinstance(node_out, dict):
                    out_desc = _first_image_from_node_output(node_out)
                    if out_desc is not None:
                        break
                if (entry.get("status") or {}).get("completed"):
                    break
            await asyncio.sleep(0.25)

        if out_desc is None:
            return None
        s, img_bytes, _ = await _comfy_fetch_view(app, {
            "filename": str(out_desc["filename"]),
            "subfolder": str(out_desc.get("subfolder") or ""),
            "type": str(out_desc.get("type") or "temp"),
        })
        if s != 200 or not img_bytes:
            return None
        with Image.open(io.BytesIO(img_bytes)) as m_img:
            l_img = m_img.convert("L")
            if l_img.size != (width, height):
                l_img = l_img.resize((width, height), Image.Resampling.BILINEAR)
            mask_f32 = np.asarray(l_img, dtype=np.float32) / 255.0
        coverage = float((mask_f32 > 0.35).mean())
        if coverage <= 0.001:
            return None
        return mask_f32, coverage, str(compiled.get("engine") or "sam3.1")
    finally:
        try:
            await _comfy_post_json(app, "/history", {"delete": [prompt_id]}, timeout=5.0)
        except Exception:
            pass


async def handle_segment(request: web.Request) -> web.Response:
    import numpy as np
    from PIL import Image

    try:
        body = await _read_json_body(request)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)

    sid: str = request["director_sid"]
    role = str(body.get("role") or "cutout").strip().lower()
    if role not in _VALID_SEGMENT_ROLES:
        return web.json_response(
            {"ok": False, "error": f"Unknown role {role!r}; expected one of {sorted(_VALID_SEGMENT_ROLES)}"},
            status=400,
        )

    source_spec = body.get("source_ref") or body.get("source_image") or body.get("image_output")
    if not source_spec:
        return web.json_response({"ok": False, "error": "source_ref is required"}, status=400)

    raw_pos = body.get("positive_points")
    if raw_pos is None and ("nx" in body or "ny" in body):
        raw_pos = [{"nx": float(body.get("nx", 0.5)), "ny": float(body.get("ny", 0.5))}]
    raw_neg = body.get("negative_points")
    prompt_text = str(body.get("prompt") or "").strip()
    if not raw_pos and not prompt_text:
        return web.json_response(
            {"ok": False, "error": "positive_points (or nx/ny) or prompt is required"},
            status=400,
        )

    try:
        input_ref, raw_bytes, desc = await _materialize_input_image_ref(
            request.app, source_spec, sid=sid, prefix="anchor"
        )
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except FileNotFoundError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=404)

    try:
        with Image.open(io.BytesIO(raw_bytes)) as src_img:
            rgb_img = src_img.convert("RGB")
            w, h = int(rgb_img.width), int(rgb_img.height)
            rgb_u8 = np.asarray(rgb_img, dtype=np.uint8)
        pos_px = compiler._normalize_sam3_points(raw_pos, width=w, height=h)
        neg_px = compiler._normalize_sam3_points(raw_neg, width=w, height=h)
    except Exception as exc:
        return web.json_response({"ok": False, "error": f"Invalid source image or points: {exc}"}, status=400)

    store: SessionStore = request.app["store"]
    obj_info = await _fetch_object_info(request.app)
    sam3_cap = compiler.detect_sam3_capability(obj_info)
    force_grabcut = bool(body.get("force_grabcut", False))

    sam_result: tuple[Any, float, str] | None = None
    if sam3_cap["available"] and not force_grabcut:
        try:
            sam_result = await _run_sam3_segment_live(
                request.app,
                sid=sid,
                input_ref=input_ref,
                width=w,
                height=h,
                params={**body, "positive_points": pos_px, "negative_points": neg_px},
                obj_info=obj_info or {},
            )
        except Exception as exc:
            log.warning("SAM3 segmentation failed, falling back to GrabCut: %s", exc)
            sam_result = None

    if sam_result is not None:
        mask_f32, coverage, engine = sam_result
    else:
        mask_f32, coverage = _segment_grabcut_mask(
            rgb_u8,
            positive_points=pos_px,
            negative_points=neg_px,
        )
        engine = "grabcut"

    try:
        mask_png_bytes, mask_u8 = _encode_mask_rgb_png(mask_f32)
        mask_name = _sanitize_filename(
            f"seg_{role}_{Path(desc['filename']).stem}_{secrets.token_hex(3)}.png",
            default_ext=".png",
        )
        m_res = await _comfy_upload_bytes(
            request.app, mask_png_bytes, mask_name, subfolder=UPLOADS_SUBFOLDER, content_type="image/png"
        )
        m_saved = str(m_res.get("name") or mask_name)
        m_sub = str(m_res.get("subfolder") or UPLOADS_SUBFOLDER)
        m_type = str(m_res.get("type") or "input")
        mask_ref = f"{m_sub}/{m_saved}" if m_sub else m_saved
        store.record_upload(mask_ref, sid)
        mask_url = _view_url(m_saved, m_sub, m_type)

        resp_payload: dict[str, Any] = {
            "ok": True,
            "role": role,
            "engine": engine,
            "resolved_source_ref": input_ref,
            "mask_ref": mask_ref,
            "mask_view_url": mask_url,
            "view_url": mask_url,
            "coverage": round(coverage, 4),
            "width": w,
            "height": h,
            "points_px": pos_px,
            "negative_points_px": neg_px,
        }
        if role == "cutout":
            cut_png_bytes, cut_mask_bytes, crop_w, crop_h, bbox_norm = _encode_rgba_cutout_png(rgb_u8, mask_u8)
            cut_name = _sanitize_filename(
                f"seg_cutout_{Path(desc['filename']).stem}_{secrets.token_hex(3)}.png",
                default_ext=".png",
            )
            c_res = await _comfy_upload_bytes(
                request.app, cut_png_bytes, cut_name, subfolder=UPLOADS_SUBFOLDER, content_type="image/png"
            )
            c_saved = str(c_res.get("name") or cut_name)
            c_sub = str(c_res.get("subfolder") or UPLOADS_SUBFOLDER)
            c_type = str(c_res.get("type") or "input")
            cutout_ref = f"{c_sub}/{c_saved}" if c_sub else c_saved
            store.record_upload(cutout_ref, sid)
            cut_url = _view_url(c_saved, c_sub, c_type)

            cmask_name = _sanitize_filename(
                f"seg_cmask_{Path(desc['filename']).stem}_{secrets.token_hex(3)}.png",
                default_ext=".png",
            )
            cm_res = await _comfy_upload_bytes(
                request.app, cut_mask_bytes, cmask_name, subfolder=UPLOADS_SUBFOLDER, content_type="image/png"
            )
            cm_saved = str(cm_res.get("name") or cmask_name)
            cm_sub = str(cm_res.get("subfolder") or UPLOADS_SUBFOLDER)
            cutout_mask_ref = f"{cm_sub}/{cm_saved}" if cm_sub else cm_saved
            store.record_upload(cutout_mask_ref, sid)

            resp_payload["cutout_ref"] = cutout_ref
            resp_payload["cutout_mask_ref"] = cutout_mask_ref
            resp_payload["cutout_view_url"] = cut_url
            resp_payload["cutout_width"] = crop_w
            resp_payload["cutout_height"] = crop_h
            resp_payload["bbox"] = bbox_norm
            resp_payload["view_url"] = cut_url
        return web.json_response(resp_payload)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=502)


async def handle_preflight(request: web.Request) -> web.Response:
    try:
        body = await _read_json_body(request)
        summary = compiler.compute_preflight_summary(body)
        return web.json_response(summary)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)


async def handle_compile(request: web.Request) -> web.Response:
    try:
        body = await _read_json_body(request)
        obj_info = body.get("object_info")
        if obj_info is None:
            obj_info = await _fetch_object_info(request.app)
        compiled = compiler.compile_spatial_take(body, object_info=obj_info)
        preflight = compiler.compute_preflight_summary(body)
        return web.json_response(
            {
                "ok": True,
                "mode": compiled["mode"],
                "director_mode": compiled["director_mode"],
                "seed": compiled["seed"],
                "length": compiled["length"],
                "width": compiled["width"],
                "height": compiled["height"],
                "queue_baseline": compiled["queue_baseline"],
                "preview_override_injected": compiled["preview_override_injected"],
                "original_prompt": compiled.get("original_prompt", ""),
                "edited_prompt": compiled.get("edited_prompt", ""),
                "region_summary": compiled.get("region_summary", ""),
                "nodes_summary": compiled["nodes_summary"],
                "preflight": preflight,
                "graph": compiled["graph"],
                "baseline_graph": compiled["baseline_graph"],
            }
        )
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=500)


async def handle_render(request: web.Request) -> web.Response:
    try:
        body = await _read_json_body(request)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)

    sid: str = request["director_sid"]
    client_id = str(body.get("client_id") or secrets.token_hex(8))
    # Auto-promote generated `type="temp"` anchor / last_frame still refs into ComfyUI input/
    for ref_key in ("source_ref", "last_frame"):
        raw_ref = body.get(ref_key)
        if not raw_ref:
            continue
        try:
            desc = _parse_view_descriptor(raw_ref, default_type="input")
        except ValueError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        needs_promote = (
            desc["type"] != "input"
            or (not desc["subfolder"] and desc["filename"].startswith("ComfyUI_temp_"))
        )
        if needs_promote:
            try:
                in_ref, _b, _d = await _materialize_input_image_ref(request.app, raw_ref, sid=sid, prefix=ref_key)
                body[ref_key] = in_ref
            except FileNotFoundError as exc:
                return web.json_response({"ok": False, "error": str(exc)}, status=404)

    obj_info = await _fetch_object_info(request.app)
    try:
        compiled = compiler.compile_spatial_take(body, object_info=obj_info)
        preflight = compiler.compute_preflight_summary(body)
        if not preflight["can_render"]:
            return web.json_response({"ok": False, "error": "; ".join(preflight["warnings"]), "preflight": preflight}, status=400)

        take_id = f"take_{secrets.token_hex(6)}"
        prompt_id = await _submit_comfy_prompt(
            request.app,
            compiled["graph"],
            client_id=client_id,
            extra_meta={"kind": "spatial", "role": "edited", "take_id": take_id},
        )
        baseline_prompt_id = None
        if compiled["baseline_graph"] is not None:
            baseline_prompt_id = await _submit_comfy_prompt(
                request.app,
                compiled["baseline_graph"],
                client_id=client_id,
                extra_meta={"kind": "spatial", "role": "baseline", "take_id": take_id},
            )
        _set_resident_backbone(request.app, "pika")
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=502)

    request.app["store"].record_take(
        {
            "take_id": take_id,
            "kind": "spatial",
            "prompt_id": prompt_id,
            "baseline_prompt_id": baseline_prompt_id,
            "client_id": client_id,
            "session_id": sid,
            "status": "queued",
            "mode": compiled["mode"],
            "director_mode": compiled["director_mode"],
            "length": compiled["length"],
            "width": compiled["width"],
            "height": compiled["height"],
            "seed": compiled["seed"],
            **_extract_lineage_fields(body),
            "params": body,
            "clean_video": None,
            "hud_video": None,
            "baseline_video": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
    )
    return web.json_response(
        {
            "ok": True,
            "take_id": take_id,
            "prompt_id": prompt_id,
            "baseline_prompt_id": baseline_prompt_id,
            "client_id": client_id,
            "mode": compiled["mode"],
            "length": compiled["length"],
            "width": compiled["width"],
            "height": compiled["height"],
            "seed": compiled["seed"],
            "queue_baseline": compiled["queue_baseline"],
            "preview_override_injected": compiled["preview_override_injected"],
            "preflight": preflight,
        }
    )


def _resolve_take_video_descriptor(
    store: SessionStore,
    body: dict[str, Any],
    *,
    default_type: str = "temp",
) -> tuple[dict[str, str], dict[str, Any] | None]:
    """Resolve ``(view_descriptor, source_take)`` from explicit ``video`` or ``take_id``."""
    take_id = str(body.get("take_id") or "").strip()
    take: dict[str, Any] | None = store.get_take(take_id) if take_id else None
    if take_id and take is None and not body.get("video"):
        raise LookupError(f"Unknown take_id {take_id!r}")

    if body.get("video"):
        return _parse_view_descriptor(body["video"], default_type=default_type), take

    if take is not None:
        role = str(body.get("role") or "").strip().lower()
        if role == "baseline":
            vid = take.get("baseline_video") or take.get("clean_video") or take.get("hud_video")
        elif role == "hud":
            vid = take.get("hud_video") or take.get("clean_video") or take.get("baseline_video")
        else:
            vid = take.get("clean_video") or take.get("hud_video") or take.get("baseline_video")
        if not vid:
            raise ValueError(f"Take {take_id!r} has no completed video yet")
        return _parse_view_descriptor(vid, default_type=default_type), take

    raise ValueError("video descriptor or take_id is required")


async def handle_extract_frame(request: web.Request) -> web.Response:
    """Extract any video frame (default ``-1`` tail) as lossless PNG into ComfyUI input."""
    try:
        body = await _read_json_body(request)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)

    sid: str = request["director_sid"]
    try:
        frame_index = int(body.get("frame_index", -1))
        src_desc, take = _resolve_take_video_descriptor(request.app["store"], body, default_type="temp")
    except LookupError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=404)
    except (ValueError, TypeError) as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)

    status, raw_bytes, _ctype = await _comfy_fetch_view(request.app, src_desc)
    if status != 200 or not raw_bytes:
        err_status = 404 if status == 404 else 502
        return web.json_response({"ok": False, "error": f"Unable to fetch source video ({status})"}, status=err_status)

    try:
        png_bytes, resolved_idx, total_frames, w, h = _extract_video_frame_png(
            raw_bytes,
            src_desc["filename"],
            frame_index=frame_index,
        )
        out_name = _sanitize_filename(f"{Path(src_desc['filename']).stem}_f{resolved_idx}.png", default_ext=".png")
        up = await _comfy_upload_bytes(
            request.app,
            png_bytes,
            out_name,
            subfolder=UPLOADS_SUBFOLDER,
            content_type="image/png",
        )
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=502)

    stored_name = str(up.get("name") or out_name)
    stored_sub = str(up.get("subfolder") if "subfolder" in up else UPLOADS_SUBFOLDER).strip("/")
    stored_type = str(up.get("type") or "input")
    ref = f"{stored_sub}/{stored_name}" if stored_sub else stored_name
    request.app["store"].record_upload(ref, sid)

    return web.json_response(
        {
            "ok": True,
            "ref": ref,
            "filename": stored_name,
            "subfolder": stored_sub,
            "type": stored_type,
            "frame_index": resolved_idx,
            "total_frames": total_frames,
            "width": w,
            "height": h,
            "view_url": _view_url(stored_name, stored_sub, stored_type),
            "source_take_id": take["take_id"] if take else None,
        }
    )


async def handle_redub(request: web.Request) -> web.Response:
    """Compile (and queue unless ``dry_run=True``) a 2nd-pass video-frozen audio re-dub."""
    try:
        body = await _read_json_body(request)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)

    sid: str = request["director_sid"]
    store: SessionStore = request.app["store"]
    take: dict[str, Any] | None = None
    src_has_audio: bool | None = None
    video_ref = str(body.get("video_ref") or "").strip()

    if not video_ref or body.get("video") or body.get("take_id"):
        try:
            src_desc, take = _resolve_take_video_descriptor(store, body, default_type="temp")
        except LookupError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=404)
        except ValueError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

        if take and "has_audio" in take:
            src_has_audio = bool(take["has_audio"])

        if src_desc["type"] == "input":
            video_ref = f"{src_desc['subfolder']}/{src_desc['filename']}" if src_desc["subfolder"] else src_desc["filename"]
        else:
            status, raw_bytes, _ctype = await _comfy_fetch_view(request.app, src_desc)
            if status != 200 or not raw_bytes:
                err_status = 404 if status == 404 else 502
                return web.json_response(
                    {"ok": False, "error": f"Unable to fetch source video ({status})"},
                    status=err_status,
                )
            src_has_audio = _video_bytes_has_audio(raw_bytes, src_desc["filename"])
            up_name = _sanitize_filename(src_desc["filename"], default_ext=".mp4")
            try:
                up = await _comfy_upload_bytes(
                    request.app,
                    raw_bytes,
                    up_name,
                    subfolder=UPLOADS_SUBFOLDER,
                    content_type="video/mp4",
                )
            except Exception as exc:
                return web.json_response({"ok": False, "error": str(exc)}, status=502)
            stored_name = str(up.get("name") or up_name)
            stored_sub = str(up.get("subfolder") if "subfolder" in up else UPLOADS_SUBFOLDER).strip("/")
            video_ref = f"{stored_sub}/{stored_name}" if stored_sub else stored_name
            store.record_upload(video_ref, sid)

    take_params = dict((take or {}).get("params") or {})
    compile_params: dict[str, Any] = {
        "video_ref": video_ref,
        "length": body.get("length") or (take or {}).get("length") or take_params.get("length") or 90,
        "width": body.get("width") or (take or {}).get("width") or take_params.get("width") or 1344,
        "height": body.get("height") or (take or {}).get("height") or take_params.get("height") or 768,
        "steps": body.get("steps", 6),
    }
    if body.get("redub_mode"):
        compile_params["redub_mode"] = str(body["redub_mode"])
    if "denoise" in body and body["denoise"] is not None:
        compile_params["denoise"] = body["denoise"]
    if "audio_mask" in body and body["audio_mask"] is not None:
        compile_params["audio_mask"] = float(body["audio_mask"])
    if "use_distill" in body and body["use_distill"] is not None:
        compile_params["use_distill"] = bool(body["use_distill"])
    if body.get("sigmas"):
        compile_params["sigmas"] = str(body["sigmas"])
    if src_has_audio is False:
        compile_params["redub_mode"] = "fresh"
        compile_params["denoise"] = 1.0
        compile_params.pop("use_distill", None)
    if "seed" in body:
        compile_params["seed"] = body["seed"]
    elif take and take.get("seed") is not None:
        compile_params["seed"] = take["seed"]

    audio_prompt = (
        body.get("audio_prompt")
        or body.get("target_prompt")
        or body.get("prompt")
        or take_params.get("target_prompt")
        or take_params.get("original_prompt")
    )
    if audio_prompt:
        compile_params["audio_prompt"] = audio_prompt

    if body.get("cues_text"):
        compile_params["cues_text"] = body["cues_text"]
    elif isinstance(body.get("sound_cues"), list) and body["sound_cues"]:
        compile_params["cues_text"] = "\n".join(str(x).strip() for x in body["sound_cues"] if str(x).strip())
    elif body.get("events_json") is not None:
        compile_params["events_json"] = body["events_json"]
    elif take_params.get("events_json") is not None:
        compile_params["events_json"] = take_params["events_json"]

    if body.get("source_length") is not None:
        compile_params["source_length"] = int(body["source_length"])
    elif "events_json" in compile_params and body.get("events_json") is None and take and take.get("length"):
        compile_params["source_length"] = int(take["length"])

    if isinstance(body.get("portal_cues"), list) and body["portal_cues"]:
        compile_params["portal_cues"] = body["portal_cues"]

    obj_info = await _fetch_object_info(request.app)
    try:
        compiled = compiler.compile_audio_redub(compile_params, object_info=obj_info)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)

    if bool(body.get("dry_run")):
        return web.json_response({"ok": True, "dry_run": True, **compiled})

    new_take_id = f"take_{secrets.token_hex(6)}"
    client_id = str(body.get("client_id") or secrets.token_hex(8))
    try:
        prompt_id = await _submit_comfy_prompt(
            request.app,
            compiled["graph"],
            client_id=client_id,
            extra_meta={"kind": "redub", "role": "edited", "take_id": new_take_id},
        )
        _set_resident_backbone(request.app, "pika")
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=502)

    store.record_take(
        {
            "take_id": new_take_id,
            "kind": "redub",
            "prompt_id": prompt_id,
            "baseline_prompt_id": None,
            "client_id": client_id,
            "session_id": sid,
            "status": "queued",
            "mode": compiled["mode"],
            "director_mode": compiled["mode"],
            "source_take_id": take["take_id"] if take else None,
            **_extract_lineage_fields(body, fallback_parent_id=take["take_id"] if take else None),
            "length": compiled["length"],
            "width": compiled["width"],
            "height": compiled["height"],
            "seed": compiled["seed"],
            "params": compile_params,
            "clean_video": None,
            "hud_video": None,
            "baseline_video": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
    )
    return web.json_response(
        {
            "ok": True,
            "take_id": new_take_id,
            "kind": "redub",
            "prompt_id": prompt_id,
            "client_id": client_id,
            "mode": compiled["mode"],
            "redub_mode": compiled["redub_mode"],
            "video_ref": compiled["video_ref"],
            "length": compiled["length"],
            "width": compiled["width"],
            "height": compiled["height"],
            "seed": compiled["seed"],
            "denoise": compiled["denoise"],
            "audio_mask": compiled["audio_mask"],
            "nodes_summary": compiled["nodes_summary"],
        }
    )


async def handle_escalate(request: web.Request) -> web.Response:
    try:
        body = await _read_json_body(request)
        res = compiler.escalate_plan(body)
        return web.json_response(res)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)


def _parse_context_suggest_reply(raw_content: str) -> tuple[str, str, list[dict[str, str]]]:
    """Parse plain Markdown (`scene_caption:`, `subject_label:`, `## suggestions`) or JSON fallback."""
    cleaned = raw_content.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    cleaned = re.sub(r"^```(?:markdown|md|json)?\s*|\s*```$", "", cleaned, flags=re.I | re.S).strip()

    if cleaned.startswith("{") or cleaned.startswith("["):
        try:
            parsed = json.loads(cleaned)
        except Exception:
            match_obj = re.search(r"\{.*\}", cleaned, re.S)
            match_arr = re.search(r"\[.*\]", cleaned, re.S)
            if match_obj and (not match_arr or match_obj.start() < match_arr.start()):
                parsed = json.loads(match_obj.group(0))
            elif match_arr:
                parsed = json.loads(match_arr.group(0))
            else:
                raise
        scene_caption = ""
        subject_label = ""
        raw_items: Any = parsed
        if isinstance(parsed, dict):
            scene_caption = str(parsed.get("scene_caption") or "").strip()[:280]
            subject_label = str(parsed.get("subject_label") or "").strip()[:120]
            raw_items = parsed.get("suggestions") or []
        if not isinstance(raw_items, list):
            raw_items = []
        suggestions = [
            {
                "title": str(s.get("title") or "").strip(),
                "target_prompt": str(s.get("target_prompt") or "").strip(),
                "preset": str(s.get("preset") or "sparkle_burst").strip() or "sparkle_burst",
            }
            for s in raw_items
            if isinstance(s, dict) and (s.get("target_prompt") or s.get("title"))
        ][:4]
        return scene_caption, subject_label, suggestions

    scene_caption = ""
    subject_label = ""
    suggestions: list[dict[str, str]] = []
    in_suggestions = False
    cur_item: dict[str, str] | None = None

    def _flush_item() -> None:
        nonlocal cur_item
        if cur_item and (cur_item.get("target_prompt") or cur_item.get("title")):
            cur_item.setdefault("preset", "sparkle_burst")
            suggestions.append(cur_item)
        cur_item = None

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^#{1,4}\s*suggestions\b", line, re.I):
            _flush_item()
            in_suggestions = True
            continue
        if re.match(r"^#{1,4}\s+", line):
            _flush_item()
            in_suggestions = False
            continue
        bullet_m = re.match(r"^(?:[-*•]+|\d+[.)])\s+(.*)$", line)
        body_line = bullet_m.group(1).strip() if bullet_m else line
        if not in_suggestions:
            kv = re.match(r"^(scene_caption|subject_label)\s*:\s*(.+)$", body_line, re.I)
            if kv:
                k = kv.group(1).lower()
                v = kv.group(2).strip().strip("`")
                if k == "scene_caption":
                    scene_caption = v[:280]
                elif k == "subject_label":
                    subject_label = v[:120]
                continue
        if "|" in body_line and ("title:" in body_line.lower() or "target_prompt:" in body_line.lower()):
            _flush_item()
            item: dict[str, str] = {}
            for seg in body_line.split("|"):
                if ":" not in seg:
                    continue
                sk, sv = seg.split(":", 1)
                sk_l = sk.strip().strip("`*_ ").lower()
                sv_s = sv.strip().strip("`")
                if sk_l in ("title", "target_prompt", "preset") and sv_s:
                    item[sk_l] = sv_s
            if item.get("target_prompt") or item.get("title"):
                item.setdefault("preset", "sparkle_burst")
                suggestions.append(item)
            continue
        kv_item = re.match(r"^(title|target_prompt|preset)\s*:\s*(.+)$", body_line, re.I)
        if kv_item:
            k = kv_item.group(1).lower()
            v = kv_item.group(2).strip().strip("`")
            if k == "title" or cur_item is None or k in cur_item:
                _flush_item()
                cur_item = {}
            cur_item[k] = v
            continue
        if in_suggestions and bullet_m:
            _flush_item()
            suggestions.append({"title": body_line[:40], "target_prompt": body_line, "preset": "sparkle_burst"})

    _flush_item()
    return scene_caption, subject_label, suggestions[:4]


async def handle_context_suggest(request: web.Request) -> web.Response:
    try:
        body = await _read_json_body(request)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)

    heylook_url = await _probe_heylook_url(request.app)
    if not heylook_url:
        return web.json_response({"ok": True, "enabled": False, "suggestions": []})

    session: ClientSession = request.app["http_client"]
    try:
        async with session.get(f"{heylook_url}/v1/models", timeout=ClientTimeout(total=1.5)) as resp:
            if resp.status != 200:
                return web.json_response({"ok": True, "enabled": False, "suggestions": []})
            models_doc = await resp.json()
        model_list = [m for m in (models_doc.get("data") or []) if isinstance(m, dict) and m.get("id")]
        source_spec = body.get("source_ref") or body.get("anchor_image") or body.get("ref_image")

        def _caps(m: dict[str, Any]) -> set[str]:
            raw_c = m.get("capabilities")
            if isinstance(raw_c, list):
                return {str(c).strip().lower() for c in raw_c}
            return set()

        chat_models = [m for m in model_list if not _caps(m) or "chat" in _caps(m)]
        chosen = None
        if source_spec:
            vision_models = [m for m in chat_models if "vision" in _caps(m)]
            if vision_models:
                chosen = vision_models[0]
        if chosen is None and chat_models:
            chosen = chat_models[0]
        model_id = str(chosen["id"]) if chosen else ""
        if not model_id:
            return web.json_response({"ok": True, "enabled": False, "suggestions": []})

        use_vision = bool(source_spec and ("vision" in _caps(chosen)))
        reg = body.get("region") if isinstance(body.get("region"), dict) else {}
        has_reticle = ("x" in body and "y" in body) or isinstance(body.get("region"), dict)
        x_n = max(0.0, min(1.0, float(reg.get("x", body.get("x", 0.5)))))
        y_n = max(0.0, min(1.0, float(reg.get("y", body.get("y", 0.5)))))
        rx_n = max(0.03, min(0.85, float(reg.get("rx", body.get("rx", 0.22)))))
        ry_n = max(0.03, min(0.85, float(reg.get("ry", body.get("ry", rx_n)))))

        content_blocks: list[dict[str, Any]] = []
        if use_vision:
            try:
                _s_desc, st_s, raw_img = await _fetch_image_with_temp_fallback(request.app, source_spec, default_type="input")
                if st_s == 200 and raw_img:
                    content_blocks.extend(
                        await asyncio.to_thread(
                            prepare_vlm_reticle_images,
                            raw_img,
                            has_reticle=has_reticle,
                            x_n=x_n,
                            y_n=y_n,
                            rx_n=rx_n,
                            ry_n=ry_n,
                        )
                    )
            except Exception:
                pass

        prompt_ctx = str(body.get("original_prompt") or body.get("scene_prompt") or "").strip()[:800]
        cur_target = str(body.get("prompt") or body.get("target_prompt") or "").strip()[:240]
        user_text_parts = []
        if prompt_ctx:
            user_text_parts.append(f"Scene description:\n{prompt_ctx}")
        if cur_target:
            user_text_parts.append(f"Current user draft: {cur_target}")
        if use_vision and has_reticle:
            user_text_parts.append(
                f"Image 1 marks the selected target reticle in blue at normalized ({x_n:.2f}, {y_n:.2f}); "
                "Image 2 is the zoomed crop of that target region. Reply in plain Markdown ONLY (no JSON, no code fences):\n"
                "scene_caption: <one concise visual sentence describing the whole scene>\n"
                "subject_label: <short noun phrase naming the subject inside the reticle>\n\n"
                "## suggestions\n"
                "- title: <2-4 word label> | target_prompt: <concise action inside reticle (under 12 words)> | preset: sparkle_burst"
            )
            system_msg = (
                "You ground interactive spatial video edits strictly in Image 1 (reticle) and Image 2 (zoomed crop). "
                "Name only visible subjects/objects inside the blue reticle and suggest up to 3 vivid localized edits "
                "using the requested plain Markdown headings and lines (never output JSON)."
            )
        else:
            user_text_parts.append(
                "Reply in plain Markdown ONLY (no JSON, no code fences) with up to 3 concise world-event / "
                "localized edit actions (under 12 words each) suited to this scene:\n"
                "## suggestions\n"
                "- title: <short 2-4 word label> | target_prompt: <concise action or localized edit> | preset: sparkle_burst"
            )
            system_msg = (
                "You suggest concise, vivid world-sandbox actions grounded strictly in the attached scene image or description. "
                "Do not invent unrelated subjects or animals not present in the scene. Output plain Markdown only (never JSON)."
            )
        content_blocks.append({"type": "text", "text": "\n\n".join(user_text_parts)})

        req_payload = {
            "model": model_id,
            "system": system_msg,
            "messages": [{"role": "user", "content": content_blocks}],
            "temperature": 0.6,
            "max_tokens": 512,
            "stream": False,
        }
        async with session.post(
            f"{heylook_url}/v1/messages",
            headers={"X-Request-ID": str(uuid.uuid4())},
            json=req_payload,
            timeout=ClientTimeout(total=12.0),
        ) as resp:
            if resp.status != 200:
                return web.json_response({"ok": True, "enabled": False, "suggestions": []})
            comp = await resp.json()

        raw_blocks = comp.get("content")
        if isinstance(raw_blocks, list):
            raw_content = "".join(
                str(b.get("text") or "")
                for b in raw_blocks
                if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            raw_content = str(comp.get("text") or "")

        scene_caption, subject_label, suggestions = _parse_context_suggest_reply(raw_content)
        if not suggestions and not scene_caption and not subject_label:
            raise ValueError("empty context_suggest reply")
        return web.json_response(
            {
                "ok": True,
                "enabled": True,
                "suggestions": suggestions,
                "scene_caption": scene_caption,
                "subject_label": subject_label,
                "reticle_grounded": bool(use_vision and has_reticle),
            }
        )
    except Exception:
        return web.json_response({"ok": True, "enabled": False, "suggestions": []})


async def _sync_store_history(app: web.Application) -> tuple[dict[str, Any], bool]:
    hist = await _comfy_get_json(app, "/history", timeout=3.0)
    if isinstance(hist, dict):
        app["store"].update_from_history(hist)
        return hist, True
    return {}, False


def _extract_active_queue_pids(queue_doc: Any) -> list[str]:
    if not isinstance(queue_doc, dict):
        return []
    seen: list[str] = []
    for bucket in ("queue_running", "queue_pending"):
        items = queue_doc.get(bucket)
        if not isinstance(items, list):
            continue
        for item in items:
            pid, _graph, _extra = director_routes._queue_item_parts(item)
            if pid and str(pid) not in seen:
                seen.append(str(pid))
    return seen


async def handle_takes(request: web.Request) -> web.Response:
    hist, hist_ok = await _sync_store_history(request.app)
    q_doc = await _comfy_get_json(request.app, "/queue", timeout=2.0)
    queue_ok = isinstance(q_doc, dict)
    active_pids = _extract_active_queue_pids(q_doc) if queue_ok else []
    active_set = set(active_pids)
    if queue_ok and hist_ok:
        request.app["store"].reconcile_orphaned_queued(active_set, set(str(k) for k in hist.keys()))
    takes = request.app["store"].list_takes()
    for t in takes:
        if "prompt_preview" not in t:
            t["prompt_preview"] = _take_prompt_preview(t)
        base_pid = str(t.get("baseline_prompt_id") or "")
        t["baseline_pending"] = bool(
            base_pid
            and t.get("status") == "completed"
            and not t.get("baseline_video")
            and not t.get("baseline_error")
            and (base_pid in active_set or (time.time() - float(t.get("updated_at") or t.get("created_at") or 0.0)) < 15.0)
        )
    return web.json_response(
        {
            "ok": True,
            "takes": takes,
            "active_prompt_ids": active_pids,
            "queue_ok": queue_ok,
        }
    )


async def handle_history(request: web.Request) -> web.Response:
    hist, _ = await _sync_store_history(request.app)
    takes = request.app["store"].list_takes()
    known_pids = {
        pid
        for t in takes
        for pid in (t.get("prompt_id"), t.get("baseline_prompt_id"))
        if pid
    }
    filtered = {pid: entry for pid, entry in hist.items() if pid in known_pids}
    return web.json_response({"ok": True, "history": filtered, "takes": takes})


async def handle_view(request: web.Request) -> web.StreamResponse:
    q = request.rel_url.query
    try:
        desc = _parse_view_descriptor(
            {
                "filename": q.get("filename", ""),
                "subfolder": q.get("subfolder", ""),
                "type": q.get("type", "temp"),
            },
            default_type="temp",
        )
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)

    session: ClientSession = request.app["http_client"]
    base: str = request.app["comfy_url"]
    qs = urlencode(desc)
    range_header = (request.headers.get("Range") or "").strip()
    upstream_headers = {"Range": range_header} if range_header else {}

    try:
        async with session.get(
            f"{base}/view?{qs}",
            headers=upstream_headers,
            timeout=ClientTimeout(total=60.0),
        ) as resp:
            if resp.status == 404:
                return web.Response(status=404, text="not found")
            if resp.status not in (200, 206):
                return web.Response(status=resp.status, text=await resp.text())
            data = await resp.read()
            ctype = resp.headers.get("Content-Type") or mimetypes.guess_type(desc["filename"])[0] or "application/octet-stream"
            upstream_status = resp.status
            upstream_cr = resp.headers.get("Content-Range")
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=502)

    if range_header and upstream_status == 200:
        m = _SAFE_RANGE_RE.match(range_header)
        if not m:
            return web.Response(status=416, headers={"Content-Range": f"bytes */{len(data)}"})
        total = len(data)
        start_s, end_s = m.group(1), m.group(2)
        if not start_s and not end_s:
            return web.Response(status=416, headers={"Content-Range": f"bytes */{total}"})
        if start_s:
            start = int(start_s)
            end = int(end_s) if end_s else total - 1
        else:
            suffix_len = int(end_s)
            start = max(0, total - suffix_len)
            end = total - 1
        if start >= total or end < start:
            return web.Response(status=416, headers={"Content-Range": f"bytes */{total}"})
        end = min(end, total - 1)
        chunk = data[start : end + 1]
        return web.Response(
            body=chunk,
            status=206,
            headers={
                "Content-Type": ctype,
                "Content-Range": f"bytes {start}-{end}/{total}",
                "Accept-Ranges": "bytes",
                "Cache-Control": "no-store",
            },
        )

    headers = {"Accept-Ranges": "bytes", "Cache-Control": "no-store"}
    if upstream_cr:
        headers["Content-Range"] = upstream_cr
    return web.Response(body=data, status=upstream_status, content_type=ctype, headers=headers)


async def handle_ab_metrics(request: web.Request) -> web.Response:
    try:
        body = await _read_json_body(request)
        edited_desc = _parse_view_descriptor(body.get("edited_video") or body.get("edited"), default_type="temp")
        base_desc = _parse_view_descriptor(body.get("baseline_video") or body.get("baseline"), default_type="temp")
    except Exception as exc:
        return web.json_response({"ok": False, "available": False, "error": str(exc)}, status=400)

    s1, b1, _ = await _comfy_fetch_view(request.app, edited_desc)
    s2, b2, _ = await _comfy_fetch_view(request.app, base_desc)
    if s1 != 200 or s2 != 200 or not b1 or not b2:
        return web.json_response(
            {"ok": False, "available": False, "error": "both edited_video and baseline_video are required"}
        )

    try:
        raw_events = body.get("events_json")
        events_str = json.dumps(raw_events) if isinstance(raw_events, dict) else str(raw_events or "")
        director_mode = str(body.get("director_mode") or "portal_path")
        feather = float(body.get("feather", 0.08))
        invert_mask = bool(body.get("invert_mask", False))
        include_sprite_envelope = bool(body.get("include_sprite_envelope", False))
        sprite_envelope_radius = float(body.get("sprite_envelope_radius", 0.18))
        payload = await asyncio.to_thread(
            _compute_ab_metrics_sync,
            b1,
            edited_desc["filename"],
            b2,
            base_desc["filename"],
            events_str,
            director_mode,
            feather,
            invert_mask=invert_mask,
            include_sprite_envelope=include_sprite_envelope,
            sprite_envelope_radius=sprite_envelope_radius,
        )
        return web.json_response(payload)
    except Exception as exc:
        return web.json_response({"ok": False, "available": False, "error": str(exc)}, status=400)


async def handle_export(request: web.Request) -> web.Response:
    try:
        body = await _read_json_body(request)
        src_desc = _parse_view_descriptor(body.get("video") or body, default_type="temp")
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)

    if src_desc["type"] != "temp":
        return web.json_response({"ok": False, "error": "export source must reside in temp"}, status=400)

    status, raw_bytes, _ = await _comfy_fetch_view(request.app, src_desc)
    if status != 200 or not raw_bytes:
        return web.json_response({"ok": False, "error": f"temp artifact not found ({status})"}, status=404)

    out_root = _resolve_output_dir()
    video_dir = out_root / "Video"
    video_dir.mkdir(parents=True, exist_ok=True)
    export_base = str(body.get("export_name") or src_desc["filename"])
    safe_name = _sanitize_filename(export_base, default_ext=".mp4")
    if not safe_name.lower().endswith(".mp4"):
        safe_name = f"{Path(safe_name).stem}.mp4"
    dest = (video_dir / safe_name).resolve()
    if not dest.is_relative_to(video_dir.resolve()):
        return web.json_response({"ok": False, "error": "invalid export path"}, status=400)

    tmp_file = video_dir / f".{safe_name}.{secrets.token_hex(4)}.tmp"
    tmp_file.write_bytes(raw_bytes)
    tmp_file.replace(dest)

    store: SessionStore = request.app["store"]
    take_id = str(body.get("take_id") or "").strip()
    take = store.get_take(take_id) if take_id else None
    if take is None:
        take = store.find_take_by_video_filename(src_desc["filename"])
    extra_recipe = body.get("recipe") if isinstance(body.get("recipe"), dict) else None
    recipe_doc = _build_recipe_document(
        take,
        video_filename=safe_name,
        variant=str(body.get("variant") or "clean"),
        extra_recipe=extra_recipe,
    )
    recipe_filename = None
    recipe_output_rel = None
    if recipe_doc is not None:
        sidecar_name = f"{Path(safe_name).stem}.director.json"
        sidecar_dest = (video_dir / sidecar_name).resolve()
        if sidecar_dest.is_relative_to(video_dir.resolve()):
            tmp_sidecar = video_dir / f".{sidecar_name}.{secrets.token_hex(4)}.tmp"
            tmp_sidecar.write_text(json.dumps(recipe_doc, indent=2), encoding="utf-8")
            tmp_sidecar.replace(sidecar_dest)
            recipe_filename = sidecar_name
            recipe_output_rel = f"Video/{sidecar_name}"

    return web.json_response(
        {
            "ok": True,
            "filename": safe_name,
            "subfolder": "Video",
            "type": "output",
            "output_rel": f"Video/{safe_name}",
            "view_url": _view_url(safe_name, "Video", "output"),
            "bytes": len(raw_bytes),
            "recipe_filename": recipe_filename,
            "recipe_output_rel": recipe_output_rel,
        }
    )


async def handle_import_recipe(request: web.Request) -> web.Response:
    """Validate and normalize an exported ``.director.json`` sidecar or recorded ``take_id`` for stage restore."""
    try:
        body = await _read_json_body(request)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)

    store: SessionStore = request.app["store"]
    raw_doc: Any = None
    take_id = str(body.get("take_id") or "").strip()
    if body.get("recipe_filename") is not None:
        raw_fn = str(body.get("recipe_filename") or "").strip()
        if (
            not raw_fn
            or "/" in raw_fn
            or "\\" in raw_fn
            or ".." in raw_fn
            or not _SAFE_FILENAME_RE.match(raw_fn)
            or not raw_fn.endswith(".director.json")
        ):
            return web.json_response({"ok": False, "error": "invalid recipe_filename"}, status=400)
        video_dir = (_resolve_output_dir() / "Video").resolve()
        target = (video_dir / raw_fn).resolve()
        if not target.is_relative_to(video_dir):
            return web.json_response({"ok": False, "error": "invalid recipe_filename"}, status=400)
        if not target.is_file():
            return web.json_response({"ok": False, "error": f"recipe sidecar not found: {raw_fn}"}, status=404)
        try:
            raw_doc = json.loads(target.read_text(encoding="utf-8"))
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"invalid recipe JSON: {exc}"}, status=400)
    elif isinstance(body.get("recipe"), dict):
        raw_doc = body["recipe"]
    elif take_id:
        raw_doc = store.get_take(take_id)
        if raw_doc is None:
            return web.json_response({"ok": False, "error": f"Unknown take_id {take_id!r}"}, status=404)
    else:
        return web.json_response({"ok": False, "error": "take_id, recipe_filename, or recipe object is required"}, status=400)

    try:
        normalized = _normalize_imported_recipe(raw_doc)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    return web.json_response({"ok": True, "recipe": normalized})


async def handle_interrupt(request: web.Request) -> web.Response:
    status, data = await _comfy_post_json(request.app, "/interrupt", {}, timeout=5.0)
    if status != 200:
        return web.json_response({"ok": False, "error": data}, status=502)
    return web.json_response({"ok": True})


async def handle_forget(request: web.Request) -> web.Response:
    try:
        body = await _read_json_body(request)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)

    store: SessionStore = request.app["store"]
    keep_uploads = set(str(x) for x in (body.get("keep_uploads") or ()))
    open_pids = [
        pid
        for t in store.list_takes()
        if t.get("status") == "queued" or t.get("pinned")
        for pid in (t.get("prompt_id"), t.get("baseline_prompt_id"))
        if pid
    ]
    payload = {
        "prompt_ids": body.get("prompt_ids"),
        "open_prompt_ids": open_pids,
        "keep_uploads": sorted(keep_uploads),
    }
    status, data = await _comfy_post_json(request.app, "/h3_spatial_director/forget", payload, timeout=15.0)
    if status != 200 or not isinstance(data, dict):
        return web.json_response({"ok": False, "error": data}, status=502)
    store.clear_forgotten(keep_uploads=keep_uploads)
    data.setdefault("protected_output_dir", "Video")
    return web.json_response(data)


def should_forward_ws_message(msg_obj: Any, *, client_id: str, store: SessionStore) -> bool:
    """Strictly scope WS JSON payloads per ``client_id``."""
    if not isinstance(msg_obj, dict):
        return True
    data = msg_obj.get("data")
    if isinstance(data, dict):
        msg_cid = data.get("client_id")
        if msg_cid is not None and str(msg_cid) != client_id:
            return False
        msg_pid = data.get("prompt_id")
        if msg_pid is not None:
            owner_cid = store.client_for_prompt(str(msg_pid))
            if owner_cid is not None and owner_cid != client_id:
                return False
    return True


async def push_ws_event(
    app: web.Application,
    payload: dict[str, Any],
    *,
    target_client_id: str | None = None,
) -> int:
    """Broadcast a JSON WS event only to matching ``clientId`` sockets."""
    store: SessionStore = app["store"]
    clients_map: dict[str, set[web.WebSocketResponse]] = app["ws_clients"]
    encoded = json.dumps(payload)
    sent = 0
    for cid, sockets in list(clients_map.items()):
        if target_client_id is not None and cid != target_client_id:
            continue
        if not should_forward_ws_message(payload, client_id=cid, store=store):
            continue
        for ws in list(sockets):
            try:
                await ws.send_str(encoded)
                sent += 1
            except Exception:
                sockets.discard(ws)
    return sent


async def handle_ws(request: web.Request) -> web.StreamResponse:
    client_id = (request.rel_url.query.get("clientId") or "").strip() or secrets.token_hex(8)
    ws_down = web.WebSocketResponse(heartbeat=25.0)
    await ws_down.prepare(request)

    app = request.app
    store: SessionStore = app["store"]
    clients_map: dict[str, set[web.WebSocketResponse]] = app["ws_clients"]
    clients_map.setdefault(client_id, set()).add(ws_down)

    comfy_ws_url = app["comfy_url"].replace("http://", "ws://", 1).replace("https://", "wss://", 1)
    target = f"{comfy_ws_url}/ws?clientId={quote(client_id)}"
    session: ClientSession = app["http_client"]

    try:
        async with session.ws_connect(target, heartbeat=25.0) as ws_up:
            async def up_to_down() -> None:
                async for msg in ws_up:
                    if msg.type == WSMsgType.TEXT:
                        try:
                            obj = json.loads(msg.data)
                        except Exception:
                            obj = None
                        if obj is not None:
                            if not should_forward_ws_message(obj, client_id=client_id, store=store):
                                continue
                            if obj.get("type") == "executed" and isinstance(obj.get("data"), dict):
                                d = obj["data"]
                                store.mark_executed(
                                    str(d.get("prompt_id") or ""),
                                    str(d.get("node") or ""),
                                    d.get("output") or {},
                                )
                        await ws_down.send_str(msg.data)
                    elif msg.type == WSMsgType.BINARY:
                        await ws_down.send_bytes(msg.data)
                    elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                        break

            async def down_to_up() -> None:
                async for msg in ws_down:
                    if msg.type == WSMsgType.TEXT:
                        await ws_up.send_str(msg.data)
                    elif msg.type == WSMsgType.BINARY:
                        await ws_up.send_bytes(msg.data)
                    elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                        break

            t1 = asyncio.create_task(up_to_down())
            t2 = asyncio.create_task(down_to_up())
            done, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done:
                task.exception()
    except Exception:
        # Keep downstream socket usable even if upstream ComfyUI restarts or isn't listening yet.
        async for msg in ws_down:
            if msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                break
    finally:
        bucket = clients_map.get(client_id)
        if bucket is not None:
            bucket.discard(ws_down)
            if not bucket:
                clients_map.pop(client_id, None)

    return ws_down


def make_security_middleware(allowed_hosts: set[str]):
    """Enforce Host/Origin allowlist, Sec-Fetch-Site guard, CSP headers, and HttpOnly SameSite=Strict cookie."""

    @web.middleware
    async def security_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
        host_hdr = _strip_host_port(request.headers.get("Host") or "")
        if host_hdr and host_hdr not in allowed_hosts:
            resp = web.json_response({"ok": False, "error": "forbidden_host"}, status=403)
            return _apply_security_headers(resp)

        origin = (request.headers.get("Origin") or "").strip()
        if origin and origin != "null":
            orig_host = (urlsplit(origin).hostname or "").strip().lower()
            if orig_host not in allowed_hosts:
                resp = web.json_response({"ok": False, "error": "forbidden_origin"}, status=403)
                return _apply_security_headers(resp)
        elif origin == "null":
            resp = web.json_response({"ok": False, "error": "forbidden_origin"}, status=403)
            return _apply_security_headers(resp)

        fetch_site = (request.headers.get("Sec-Fetch-Site") or "").strip().lower()
        if fetch_site == "cross-site":
            resp = web.json_response({"ok": False, "error": "forbidden_cross_site"}, status=403)
            return _apply_security_headers(resp)

        sid, is_new = _get_session_id(request)
        request["director_sid"] = sid
        secure_cookie = _is_https_request(request)

        try:
            resp = await handler(request)
        except web.HTTPException as exc:
            _apply_security_headers(
                exc,
                sid_to_set=sid if is_new else None,
                secure_cookie=secure_cookie,
            )
            raise

        # Note: WebSocketResponse prepares headers during `await ws.prepare(request)` before returning.
        if not isinstance(resp, web.WebSocketResponse):
            _apply_security_headers(
                resp,
                sid_to_set=sid if is_new else None,
                secure_cookie=secure_cookie,
            )
        return resp

    return security_middleware


def create_app(
    *,
    comfy_url: str = "http://127.0.0.1:8188",
    state_dir: Path | str = DEFAULT_STATE_DIR,
    listen_host: str = DEFAULT_HOST,
    extra_allowed_hosts: list[str] | None = None,
) -> web.Application:
    """Construct the configured aiohttp Application."""
    allowed_hosts = _default_allowed_hosts(listen_host, extra_allowed_hosts)
    app = web.Application(
        client_max_size=MAX_UPLOAD_BYTES,
        middlewares=[make_security_middleware(allowed_hosts)],
    )
    app["comfy_url"] = comfy_url.rstrip("/")
    app["allowed_hosts"] = allowed_hosts
    app["store"] = SessionStore(Path(state_dir))
    app["ws_clients"] = {}
    app["runtime"] = {"last_resident_backbone": None}

    async def _on_startup(a: web.Application) -> None:
        a["http_client"] = ClientSession()

    async def _on_cleanup(a: web.Application) -> None:
        cli = a.get("http_client")
        if cli is not None:
            await cli.close()

    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)

    app.router.add_get("/", handle_index)
    app.router.add_get("/docs", handle_docs_page)
    app.router.add_get("/api/docs", handle_docs_index)
    app.router.add_get("/api/docs/file", handle_docs_file)
    app.router.add_get("/api/docs/asset", handle_docs_asset)
    app.router.add_get("/static/{tail:.*}", handle_static)
    app.router.add_get("/api/schema", handle_schema)
    app.router.add_get("/api/system_stats", handle_system_stats)
    app.router.add_post("/api/upload", handle_upload)
    app.router.add_post("/api/stamp_cutout", handle_stamp_cutout)
    app.router.add_post("/api/segment", handle_segment)
    app.router.add_post("/api/anchor", handle_anchor)
    app.router.add_post("/api/preview_in_mask", handle_preview_in_mask)
    app.router.add_post("/api/preflight", handle_preflight)
    app.router.add_post("/api/compile", handle_compile)
    app.router.add_post("/api/render", handle_render)
    app.router.add_post("/api/extract_frame", handle_extract_frame)
    app.router.add_post("/api/redub", handle_redub)
    app.router.add_post("/api/escalate", handle_escalate)
    app.router.add_post("/api/context_suggest", handle_context_suggest)
    app.router.add_get("/api/takes", handle_takes)
    app.router.add_get("/api/history", handle_history)
    app.router.add_get("/api/view", handle_view)
    app.router.add_post("/api/ab_metrics", handle_ab_metrics)
    app.router.add_post("/api/export", handle_export)
    app.router.add_post("/api/import_recipe", handle_import_recipe)
    app.router.add_post("/api/interrupt", handle_interrupt)
    app.router.add_post("/api/forget", handle_forget)
    app.router.add_get("/api/ws", handle_ws)
    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Interactive Spatial Director standalone backend server")
    ap.add_argument("--host", default=os.environ.get("DIRECTOR_HOST", DEFAULT_HOST))
    ap.add_argument("--port", type=int, default=int(os.environ.get("DIRECTOR_PORT", DEFAULT_PORT)))
    ap.add_argument(
        "--comfy-url",
        "--comfy",
        dest="comfy_url",
        default=os.environ.get("H3_COMFY_URL", "http://127.0.0.1:8188"),
    )
    ap.add_argument(
        "--state-dir",
        default=os.environ.get("DIRECTOR_STATE_DIR", str(DEFAULT_STATE_DIR)),
    )
    ap.add_argument("--allow-host", action="append", default=[])
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = create_app(
        comfy_url=args.comfy_url,
        state_dir=Path(args.state_dir),
        listen_host=args.host,
        extra_allowed_hosts=list(args.allow_host),
    )
    web.run_app(app, host=args.host, port=int(args.port), print=None if args.quiet else log.info)


if __name__ == "__main__":
    main()
