"""ComfyUI cleanup route extension for Spatial World Director (`/h3_spatial_director/forget`).

Keeps top-level imports strictly stdlib so ``bench/check_imports.py`` passes offline.
ComfyUI modules (``aiohttp``, ``folder_paths``, ``server``) are imported lazily inside
runtime helpers.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Iterable

log = logging.getLogger(__name__)

APP_MARK = "h3_spatial_director"
UPLOADS = "h3_spatial_director"
_OUTPUT_KINDS = ("images", "audio", "gifs")


def _safe_under(root: Path, subfolder: str, filename: str) -> Path | None:
    """Resolve ``root / subfolder / filename`` only if it stays strictly inside ``root``."""
    if not filename or filename in (".", ".."):
        return None
    if "/" in filename or "\\" in filename or "\x00" in filename:
        return None
    sub = (subfolder or "").strip().strip("/\\")
    if sub:
        parts = [p for p in Path(sub).parts if p not in ("", ".")]
        if any(p == ".." for p in parts):
            return None
        candidate = root.joinpath(*parts, filename)
    else:
        candidate = root / filename
    try:
        root_res = root.resolve()
        cand_res = candidate.resolve()
    except OSError:
        return None
    if cand_res == root_res or not cand_res.is_relative_to(root_res):
        return None
    return cand_res


def upload_rel(name: str) -> str:
    """Return canonical ``h3_spatial_director/<file>`` if ``name`` points to a director upload."""
    if not isinstance(name, str) or not name or "\x00" in name:
        return ""
    norm = name.replace("\\", "/").strip("/")
    prefix = f"{UPLOADS}/"
    if not norm.startswith(prefix):
        return ""
    rest = norm[len(prefix):]
    if not rest or "/" in rest or rest in (".", ".."):
        return ""
    return f"{UPLOADS}/{rest}"


def _extract_upload_refs(obj: Any, found: set[str]) -> None:
    """Recursively collect any ``h3_spatial_director/<file>`` references inside ``obj``."""
    if isinstance(obj, str):
        rel = upload_rel(obj)
        if rel:
            found.add(rel)
    elif isinstance(obj, dict):
        for v in obj.values():
            _extract_upload_refs(v, found)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _extract_upload_refs(v, found)


def _queue_item_parts(item: Any) -> tuple[str | None, Any, dict[str, Any]]:
    """Extract ``(prompt_id, graph, extra_data)`` from a ComfyUI queue or history entry."""
    if isinstance(item, dict):
        pid = item.get("prompt_id")
        graph = item.get("prompt")
        extra = item.get("extra_data")
        if isinstance(graph, (list, tuple)):
            return _queue_item_parts(graph)
        return (
            str(pid) if pid is not None else None,
            graph,
            extra if isinstance(extra, dict) else {},
        )
    if isinstance(item, (list, tuple)):
        pid = str(item[1]) if len(item) > 1 and item[1] is not None else None
        graph = item[2] if len(item) > 2 else None
        extra = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
        return pid, graph, extra
    return None, None, {}


def _protected_output_roots(explicit_output_dir: Path | None = None) -> list[Path]:
    """Return resolved output directories that must NEVER be deleted by forget."""
    roots: list[Path] = []
    candidates: list[Path] = []
    if explicit_output_dir is not None:
        candidates.append(Path(explicit_output_dir))
    env_out = (os.environ.get("H3_OUTPUT_DIR") or "").strip()
    if env_out:
        candidates.append(Path(env_out))
    try:
        import folder_paths  # type: ignore

        comfy_out = folder_paths.get_output_directory()
        if comfy_out:
            candidates.append(Path(comfy_out))
    except Exception:
        pass
    try:
        from workflows.h3_config import output_dir as cfg_output_dir

        candidates.append(Path(cfg_output_dir()))
    except Exception:
        pass
    for cand in candidates:
        try:
            res = cand.resolve()
        except OSError:
            continue
        if res not in roots:
            roots.append(res)
            roots.append((res / "Video").resolve())
    return roots


def _is_under_any(path: Path, roots: Iterable[Path]) -> bool:
    try:
        res = path.resolve()
    except OSError:
        return True
    return any(res == r or res.is_relative_to(r) for r in roots)


def compute_forget_plan(
    *,
    history: dict[str, Any] | None = None,
    queue_running: Iterable[Any] = (),
    queue_pending: Iterable[Any] = (),
    prompt_ids: Iterable[str] | None = None,
    open_prompt_ids: Iterable[str] | None = None,
    protected_uploads: Iterable[str] | None = None,
    uploads_on_disk: Iterable[str] | None = None,
    temp_dir: Path | None = None,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Compute temp outputs and orphaned director uploads eligible for deletion.

    Protects:
    - All non-``temp`` artifacts (never touches ``output`` or ``$H3_OUTPUT_DIR/Video/``).
    - Uploads referenced by running, pending, or ``open_prompt_ids`` graphs,
      plus explicit ``protected_uploads``.
    """
    hist = history or {}
    requested_ids = {str(x) for x in prompt_ids} if prompt_ids is not None else None
    open_ids = {str(x) for x in (open_prompt_ids or ())}

    active_ids: set[str] = set()
    kept_uploads: set[str] = set()
    for raw_u in protected_uploads or ():
        rel = upload_rel(raw_u) or (upload_rel(f"{UPLOADS}/{raw_u}") if raw_u and "/" not in str(raw_u) else "")
        if rel:
            kept_uploads.add(rel)

    for entry in (*queue_running, *queue_pending):
        pid, graph, _ = _queue_item_parts(entry)
        if pid:
            active_ids.add(pid)
        _extract_upload_refs(graph, kept_uploads)

    protected_roots = _protected_output_roots(output_dir)
    temp_root = Path(temp_dir) if temp_dir is not None else None
    input_root = Path(input_dir) if input_dir is not None else None

    history_ids_to_delete: list[str] = []
    temp_files: list[dict[str, str]] = []
    seen_temp_keys: set[tuple[str, str]] = set()
    skipped_non_temp: list[dict[str, Any]] = []

    for pid_raw, item in hist.items():
        pid = str(pid_raw)
        if not isinstance(item, dict):
            continue
        _, graph, extra = _queue_item_parts(item.get("prompt"))
        if not extra and isinstance(item.get("extra_data"), dict):
            extra = item["extra_data"]
        is_ours = bool(extra.get(APP_MARK)) or (requested_ids is not None and pid in requested_ids)
        is_protected_session = pid in active_ids or pid in open_ids

        if not is_ours or is_protected_session or (requested_ids is not None and pid not in requested_ids):
            # Keep uploads referenced by active, open, non-director, or unselected history entries.
            _extract_upload_refs(graph, kept_uploads)
            continue

        history_ids_to_delete.append(pid)
        outputs = item.get("outputs") or {}
        if not isinstance(outputs, dict):
            continue
        for node_out in outputs.values():
            if not isinstance(node_out, dict):
                continue
            for kind in _OUTPUT_KINDS:
                entries = node_out.get(kind)
                if not isinstance(entries, list):
                    continue
                for f in entries:
                    if not isinstance(f, dict):
                        continue
                    ftype = str(f.get("type") or "")
                    fname = str(f.get("filename") or "")
                    sub = str(f.get("subfolder") or "")
                    if ftype != "temp":
                        skipped_non_temp.append({"filename": fname, "subfolder": sub, "type": ftype, "kind": kind})
                        continue
                    if not fname or "/" in fname or "\\" in fname or fname in (".", ".."):
                        continue
                    if temp_root is not None:
                        resolved = _safe_under(temp_root, sub, fname)
                        if resolved is None or _is_under_any(resolved, protected_roots):
                            skipped_non_temp.append({"filename": fname, "subfolder": sub, "type": ftype, "reason": "outside_temp_or_protected"})
                            continue
                    key = (sub.strip("/\\"), fname)
                    if key not in seen_temp_keys:
                        seen_temp_keys.add(key)
                        temp_files.append({"filename": fname, "subfolder": sub, "type": "temp", "kind": kind})

    if uploads_on_disk is not None:
        disk_uploads = sorted({u for raw in uploads_on_disk if (u := upload_rel(str(raw)))})
    elif input_root is not None:
        upload_dir = input_root / UPLOADS
        disk_uploads = []
        if upload_dir.is_dir():
            for p in sorted(upload_dir.iterdir()):
                if p.is_file():
                    rel = upload_rel(f"{UPLOADS}/{p.name}")
                    if rel:
                        disk_uploads.append(rel)
    else:
        disk_uploads = []

    uploads_to_delete = [u for u in disk_uploads if u not in kept_uploads]

    return {
        "history_ids_to_delete": sorted(history_ids_to_delete),
        "temp_files_to_delete": temp_files,
        "uploads_to_delete": uploads_to_delete,
        "protected_uploads": sorted(kept_uploads),
        "skipped_non_temp": skipped_non_temp,
    }


def delete_temp_files(
    plan: dict[str, Any],
    *,
    temp_dir: Path,
    input_dir: Path,
    output_dir: Path | None = None,
    delete_history_fn: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Execute a ``compute_forget_plan`` strictly within ``temp_dir`` and ``input_dir/h3_spatial_director``."""
    protected_roots = _protected_output_roots(output_dir)
    temp_root = Path(temp_dir)
    input_root = Path(input_dir)
    uploads_root = input_root / UPLOADS

    deleted_temp: list[str] = []
    deleted_uploads: list[str] = []
    deleted_history: list[str] = []

    for item in plan.get("temp_files_to_delete") or ():
        if not isinstance(item, dict) or item.get("type") != "temp":
            continue
        resolved = _safe_under(temp_root, str(item.get("subfolder") or ""), str(item.get("filename") or ""))
        if resolved is None or _is_under_any(resolved, protected_roots):
            continue
        if resolved.is_file():
            try:
                resolved.unlink()
                deleted_temp.append(str(resolved.relative_to(temp_root.resolve())))
            except OSError as exc:
                log.warning("[h3_director] failed deleting temp file %s: %s", resolved, exc)

    for rel in plan.get("uploads_to_delete") or ():
        norm = upload_rel(str(rel))
        if not norm:
            continue
        fname = norm.split("/", 1)[1]
        resolved = _safe_under(uploads_root, "", fname)
        if resolved is None or _is_under_any(resolved, protected_roots):
            continue
        if resolved.is_file():
            try:
                resolved.unlink()
                deleted_uploads.append(norm)
            except OSError as exc:
                log.warning("[h3_director] failed deleting upload %s: %s", resolved, exc)

    if delete_history_fn is not None:
        for pid in plan.get("history_ids_to_delete") or ():
            try:
                delete_history_fn(str(pid))
                deleted_history.append(str(pid))
            except Exception as exc:
                log.warning("[h3_director] failed deleting history %s: %s", pid, exc)

    return {
        "ok": True,
        "deleted_temp": deleted_temp,
        "deleted_uploads": deleted_uploads,
        "deleted_history": deleted_history,
        "protected_uploads": list(plan.get("protected_uploads") or ()),
        "skipped_non_temp": list(plan.get("skipped_non_temp") or ()),
    }


async def forget(request: Any) -> Any:
    """ComfyUI ``POST /h3_spatial_director/forget`` handler."""
    from aiohttp import web  # type: ignore
    import folder_paths  # type: ignore
    import server  # type: ignore

    body: dict[str, Any] = {}
    try:
        raw = await request.text()
        if raw and raw.strip():
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                body = parsed
    except Exception:
        return web.json_response({"ok": False, "error": "invalid_json"}, status=400)

    ps = server.PromptServer.instance
    pq = ps.prompt_queue
    history = pq.get_history()
    running, pending = pq.get_current_queue()

    temp_dir = Path(folder_paths.get_temp_directory())
    input_dir = Path(folder_paths.get_input_directory())
    output_dir = Path(folder_paths.get_output_directory())

    plan = compute_forget_plan(
        history=history,
        queue_running=running,
        queue_pending=pending,
        prompt_ids=body.get("prompt_ids"),
        open_prompt_ids=body.get("open_prompt_ids"),
        protected_uploads=body.get("keep_uploads") or body.get("protected_uploads"),
        temp_dir=temp_dir,
        input_dir=input_dir,
        output_dir=output_dir,
    )
    result = delete_temp_files(
        plan,
        temp_dir=temp_dir,
        input_dir=input_dir,
        output_dir=output_dir,
        delete_history_fn=pq.delete_history_item,
    )
    result["plan"] = plan
    return web.json_response(result)


def register() -> None:
    """Register ``POST /h3_spatial_director/forget`` on ComfyUI's ``PromptServer``."""
    import server  # type: ignore

    ps = getattr(server.PromptServer, "instance", None)
    if ps is None or getattr(ps, "app", None) is None:
        return
    route_path = "/h3_spatial_director/forget"
    for r in ps.app.router.routes():
        info = r.get_info() if hasattr(r, "get_info") else {}
        if info.get("path") == route_path and r.method == "POST":
            return
    ps.app.router.add_post(route_path, forget)
