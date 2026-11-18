"""Session storage, lineage extraction, and portable `.director.json` recipe helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

from director import compiler

log = logging.getLogger("h3.director")

APP_MARK = "h3_spatial_director"
SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,180}$")


def sanitize_filename(name: str, *, default_ext: str = ".png") -> str:
    raw = Path(name or "upload").name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(raw).stem).strip("._-") or "upload"
    ext = Path(raw).suffix.lower()
    if not ext or not re.fullmatch(r"\.[a-z0-9]{1,8}", ext):
        ext = default_ext
    digest = hashlib.sha256(f"{time.time_ns()}:{raw}".encode("utf-8")).hexdigest()[:8]
    return f"{stem[:64]}_{digest}{ext}"


def safe_rel_name(name: str) -> str:
    clean = Path(name or "").name
    if not clean or not SAFE_FILENAME_RE.match(clean):
        raise ValueError(f"Unsafe filename {name!r}")
    return clean


def view_url(filename: str, subfolder: str = "", media_type: str = "temp") -> str:
    params = {"filename": filename, "subfolder": subfolder, "type": media_type}
    return f"/api/view?{urlencode(params)}"


def parse_view_descriptor(spec: Any, *, default_type: str = "temp") -> dict[str, str]:
    if isinstance(spec, str):
        s = spec.strip()
        if s.startswith("/api/view?"):
            qs = urlsplit(s).query
            q = parse_qs(qs)
            return {
                "filename": safe_rel_name((q.get("filename") or [""])[0]),
                "subfolder": str((q.get("subfolder") or [""])[0]).strip("/\\"),
                "type": str((q.get("type") or [default_type])[0]),
            }
        if "/" in s:
            sub, fn = s.rsplit("/", 1)
            return {"filename": safe_rel_name(fn), "subfolder": sub.strip("/\\"), "type": default_type}
        return {"filename": safe_rel_name(s), "subfolder": "", "type": default_type}
    if isinstance(spec, dict):
        fn = safe_rel_name(str(spec.get("filename") or ""))
        sub = str(spec.get("subfolder") or "").strip("/\\")
        ftype = str(spec.get("type") or default_type).strip() or default_type
        if ".." in Path(sub).parts:
            raise ValueError(f"Invalid subfolder {sub!r}")
        if ftype not in ("temp", "input", "output"):
            raise ValueError(f"Invalid media type {ftype!r}")
        return {"filename": fn, "subfolder": sub, "type": ftype}
    raise ValueError("Media descriptor must be a filename string or {filename, subfolder, type} object")


def safe_ref_view_url(ref: str, *, default_type: str = "input") -> str:
    clean = str(ref or "").strip()
    if not clean:
        return ""
    try:
        desc = parse_view_descriptor(clean, default_type=default_type)
        return view_url(desc["filename"], desc["subfolder"], desc["type"])
    except Exception:
        return ""


def normalize_media_entry(entry: dict[str, Any]) -> dict[str, Any]:
    fn = str(entry.get("filename") or "")
    sub = str(entry.get("subfolder") or "")
    ftype = str(entry.get("type") or "temp")
    out: dict[str, Any] = {
        "filename": fn,
        "subfolder": sub,
        "type": ftype,
        "view_url": view_url(fn, sub, ftype) if fn else "",
    }
    if entry.get("format"):
        out["format"] = str(entry["format"])
    return out


def first_video_from_node_output(node_out: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("gifs", "videos"):
        items = node_out.get(key)
        if isinstance(items, list) and items and isinstance(items[0], dict) and items[0].get("filename"):
            return normalize_media_entry(items[0])
    return None


def first_image_from_node_output(node_out: dict[str, Any]) -> dict[str, Any] | None:
    items = node_out.get("images")
    if isinstance(items, list) and items and isinstance(items[0], dict) and items[0].get("filename"):
        return normalize_media_entry(items[0])
    return None


def first_video_from_outputs(outputs: dict[str, Any], *, preferred_nodes: tuple[str, ...]) -> dict[str, Any] | None:
    for nid in preferred_nodes:
        node_out = outputs.get(nid)
        if isinstance(node_out, dict):
            vid = first_video_from_node_output(node_out)
            if vid:
                return vid
    return None


def first_image_from_outputs(outputs: dict[str, Any], *, preferred_nodes: tuple[str, ...]) -> dict[str, Any] | None:
    for nid in preferred_nodes:
        node_out = outputs.get(nid)
        if isinstance(node_out, dict):
            img = first_image_from_node_output(node_out)
            if img:
                return img
    return None


def extract_lineage_fields(
    body: dict[str, Any],
    *,
    fallback_parent_id: str | None = None,
) -> dict[str, Any]:
    parent_tid = str(body.get("parent_take_id") or body.get("source_take_id") or fallback_parent_id or "").strip() or None
    raw_pf = body.get("parent_frame")
    parent_frame: int | None = None
    if raw_pf is not None and not isinstance(raw_pf, bool) and str(raw_pf).strip() != "":
        try:
            parent_frame = max(0, int(raw_pf))
        except (TypeError, ValueError):
            parent_frame = None
    return {"parent_take_id": parent_tid, "parent_frame": parent_frame}


def take_prompt_preview(take: dict[str, Any]) -> str:
    params = take.get("params") if isinstance(take.get("params"), dict) else {}
    raw = str(params.get("target_prompt") or params.get("prompt") or "").strip()
    if len(raw) > 96:
        return f"{raw[:95]}…"
    return raw


def build_recipe_document(
    take: dict[str, Any] | None,
    *,
    video_filename: str = "",
    variant: str = "clean",
    extra_recipe: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build portable ``.director.json`` recipe sidecar metadata (zero local filesystem paths)."""
    if take is None and not isinstance(extra_recipe, dict):
        return None
    raw_params: dict[str, Any] = {}
    if isinstance(take, dict) and isinstance(take.get("params"), dict):
        raw_params.update(copy.deepcopy(take["params"]))
    if isinstance(extra_recipe, dict):
        stage_part = extra_recipe.get("stage") if isinstance(extra_recipe.get("stage"), dict) else extra_recipe
        for k, v in stage_part.items():
            if k not in ("schema_version", "app", "exported_at", "video_filename", "variant"):
                raw_params[k] = copy.deepcopy(v)

    raw_events = raw_params.get("events_json")
    events_obj: dict[str, Any] | None = None
    if isinstance(raw_events, dict):
        events_obj = copy.deepcopy(raw_events)
    elif isinstance(raw_events, str) and raw_events.strip():
        try:
            parsed = json.loads(raw_events)
            if isinstance(parsed, dict):
                events_obj = parsed
        except Exception:
            events_obj = None

    stage_doc: dict[str, Any] = {
        "mode": str(raw_params.get("mode") or (take or {}).get("mode") or "i2v_portal"),
        "director_mode": str(raw_params.get("director_mode") or (take or {}).get("director_mode") or "portal_path"),
        "length": int(raw_params.get("length") or (take or {}).get("length") or 141),
        "width": int(raw_params.get("width") or (take or {}).get("width") or 1344),
        "height": int(raw_params.get("height") or (take or {}).get("height") or 768),
        "seed": int(raw_params.get("seed") if raw_params.get("seed") is not None else ((take or {}).get("seed") or 0)),
        "source_ref": str(
            raw_params.get("source_ref")
            or raw_params.get("first_frame")
            or raw_params.get("anchor_image")
            or raw_params.get("ref_image")
            or ""
        ).strip(),
        "last_frame": str(raw_params.get("last_frame") or "").strip(),
        "sprite_ref": str(raw_params.get("sprite_ref") or raw_params.get("sprite_cutout") or "").strip(),
        "sprite_mask_ref": str(raw_params.get("sprite_mask_ref") or "").strip(),
        "include_mask_ref": str(raw_params.get("include_mask_ref") or "").strip(),
        "exclude_mask_ref": str(raw_params.get("exclude_mask_ref") or "").strip(),
        "snap_to_include_mask": bool(raw_params.get("snap_to_include_mask", False)),
        "lock_first_frame": bool(raw_params.get("lock_first_frame", True)),
        "original_prompt": str(raw_params.get("original_prompt") or "").strip(),
        "target_prompt": str(raw_params.get("target_prompt") or raw_params.get("prompt") or "").strip(),
        "same_prompt": bool(raw_params.get("same_prompt", False)),
        "audio_refine": bool(raw_params.get("audio_refine", False)),
        "feather": float(raw_params.get("feather", 0.08)),
        "invert_mask": bool(raw_params.get("invert_mask", False)),
        "scale": float(raw_params.get("scale", 2.0)),
        "audio_scale": float(raw_params.get("audio_scale", 1.0)),
        "first_step": int(raw_params.get("first_step", 0)),
        "last_step": int(raw_params.get("last_step", 1000)),
        "streams": str(raw_params.get("streams") or "independent"),
        "control_vectors": copy.deepcopy(raw_params.get("control_vectors") or []),
        "couplings": copy.deepcopy(raw_params.get("couplings") or []),
        "portal_cues": copy.deepcopy(raw_params.get("portal_cues") or []),
        "events_json": events_obj if events_obj is not None else {},
    }
    ex = extra_recipe if isinstance(extra_recipe, dict) else {}
    res_vid = video_filename or str(ex.get("video_filename") or "")
    res_var = variant if video_filename else str(ex.get("variant") or variant or "clean")
    res_pf = (
        (take or {}).get("parent_frame")
        if (take and take.get("parent_frame") is not None)
        else (ex.get("parent_frame") if ex.get("parent_frame") is not None else raw_params.get("parent_frame"))
    )
    return {
        "schema_version": 1,
        "app": APP_MARK,
        "exported_at": float(ex.get("exported_at") or time.time()),
        "video_filename": Path(res_vid).name if res_vid else "",
        "variant": str(res_var or "clean"),
        "take_id": (take or {}).get("take_id") or ex.get("take_id"),
        "parent_take_id": (take or {}).get("parent_take_id") or ex.get("parent_take_id") or raw_params.get("parent_take_id"),
        "parent_frame": res_pf,
        "stage": stage_doc,
    }


def normalize_imported_recipe(raw_doc: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize an imported ``.director.json`` or ``Take`` record into stage state."""
    if not isinstance(raw_doc, dict):
        raise ValueError("Recipe must be a JSON object")

    if isinstance(raw_doc.get("stage"), dict):
        built = build_recipe_document(None, extra_recipe=raw_doc)
    elif isinstance(raw_doc.get("params"), dict) or raw_doc.get("take_id"):
        built = build_recipe_document(raw_doc)
    else:
        built = build_recipe_document(None, extra_recipe=raw_doc)
    if built is None:
        raise ValueError("Recipe payload is empty")

    stage = built["stage"]
    stage["length"] = compiler.snap_to_on_both_clocks(int(stage.get("length") or 141))
    ev = stage.get("events_json")
    if isinstance(ev, str):
        ev = json.loads(ev) if ev.strip() else {}
    if not isinstance(ev, dict):
        raise ValueError("Recipe events_json must decode to a JSON object")
    stage["events_json"] = ev

    ref_urls: dict[str, str] = {}
    for key in ("source_ref", "last_frame", "sprite_ref", "sprite_mask_ref", "include_mask_ref", "exclude_mask_ref"):
        val = str(stage.get(key) or "").strip()
        if val:
            ref_urls[key] = safe_ref_view_url(val, default_type="input")
    built["ref_view_urls"] = ref_urls
    return built


class SessionStore:
    """Thread-safe persistent store for Interactive Spatial Director takes, prompts, and uploads."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "takes.json"
        self._lock = threading.RLock()
        self._takes: dict[str, dict[str, Any]] = {}
        self._take_order: list[str] = []
        self._prompt_to_client: dict[str, str] = {}
        self._prompt_to_take: dict[str, tuple[str, str]] = {}
        self._open_sessions: dict[str, set[str]] = {}
        self._uploads: dict[str, str] = {}  # ref -> session_id
        self._load()

    def _load(self) -> None:
        if not self.state_file.is_file():
            return
        try:
            doc = json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("[h3_director] ignoring corrupt state file %s: %s", self.state_file, exc)
            return
        for t in doc.get("takes") or ():
            tid = str(t.get("take_id") or "")
            if tid:
                self._takes[tid] = t
                self._take_order.append(tid)
                cid = str(t.get("client_id") or "")
                for key, role in (("prompt_id", "edited"), ("baseline_prompt_id", "baseline")):
                    pid = str(t.get(key) or "")
                    if pid:
                        if cid:
                            self._prompt_to_client[pid] = cid
                        self._prompt_to_take[pid] = (tid, role)
        self._uploads.update({str(k): str(v) for k, v in (doc.get("uploads") or {}).items()})

    def _save(self) -> None:
        payload = {
            "version": 1,
            "takes": [self._takes[tid] for tid in self._take_order if tid in self._takes],
            "uploads": dict(self._uploads),
        }
        tmp = self.state_dir / f".takes.{secrets.token_hex(6)}.tmp"
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.state_file)

    def record_upload(self, ref: str, session_id: str) -> None:
        with self._lock:
            self._uploads[ref] = session_id
            self._save()

    def record_take(self, take: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            tid = str(take["take_id"])
            self._takes[tid] = copy.deepcopy(take)
            if tid in self._take_order:
                self._take_order.remove(tid)
            self._take_order.insert(0, tid)
            cid = str(take.get("client_id") or "")
            sid = str(take.get("session_id") or "")
            for key, role in (("prompt_id", "edited"), ("baseline_prompt_id", "baseline")):
                pid = str(take.get(key) or "")
                if pid:
                    if cid:
                        self._prompt_to_client[pid] = cid
                    self._prompt_to_take[pid] = (tid, role)
                    if sid:
                        self._open_sessions.setdefault(sid, set()).add(pid)
            self._save()
            return copy.deepcopy(self._takes[tid])

    def list_takes(self) -> list[dict[str, Any]]:
        with self._lock:
            return [copy.deepcopy(self._takes[tid]) for tid in self._take_order if tid in self._takes]

    def get_take(self, take_id: str) -> dict[str, Any] | None:
        with self._lock:
            t = self._takes.get(take_id)
            return copy.deepcopy(t) if t is not None else None

    def find_take_by_video_filename(self, filename: str) -> dict[str, Any] | None:
        target = Path(filename or "").name
        if not target:
            return None
        with self._lock:
            for tid in self._take_order:
                t = self._takes.get(tid)
                if not t:
                    continue
                for key in ("clean_video", "hud_video", "baseline_video"):
                    item = t.get(key)
                    if isinstance(item, dict) and Path(str(item.get("filename") or "")).name == target:
                        return copy.deepcopy(t)
        return None

    def client_for_prompt(self, prompt_id: str) -> str | None:
        with self._lock:
            return self._prompt_to_client.get(str(prompt_id))

    def update_from_history(self, history_map: dict[str, Any]) -> None:
        with self._lock:
            changed = False
            for pid, entry in (history_map or {}).items():
                info = self._prompt_to_take.get(str(pid))
                if not info or not isinstance(entry, dict):
                    continue
                tid, role = info
                take = self._takes.get(tid)
                if take is None:
                    continue
                st_doc = entry.get("status") or {}
                if isinstance(st_doc, dict) and str(st_doc.get("status_str") or "") == "error":
                    err_msg = "ComfyUI execution error"
                    for msg_item in st_doc.get("messages") or ():
                        if (
                            isinstance(msg_item, (list, tuple))
                            and len(msg_item) >= 2
                            and msg_item[0] == "execution_error"
                            and isinstance(msg_item[1], dict)
                        ):
                            err_msg = str(msg_item[1].get("exception_message") or err_msg)
                            break
                    if role == "baseline":
                        if take.get("baseline_error") != err_msg:
                            take["baseline_error"] = err_msg
                            take["updated_at"] = time.time()
                            changed = True
                    elif take.get("status") != "completed" and (take.get("status") != "error" or take.get("error") != err_msg):
                        take["status"] = "error"
                        take["error"] = err_msg
                        take["updated_at"] = time.time()
                        changed = True
                    continue
                outputs = entry.get("outputs") or {}
                if not isinstance(outputs, dict):
                    continue
                if role == "baseline":
                    vid = first_video_from_outputs(outputs, preferred_nodes=("162", "13"))
                    if vid and take.get("baseline_video") != vid:
                        take["baseline_video"] = vid
                        take["updated_at"] = time.time()
                        changed = True
                else:
                    hud = first_video_from_outputs(outputs, preferred_nodes=("13",))
                    clean = first_video_from_outputs(outputs, preferred_nodes=("162",))
                    still = first_image_from_outputs(outputs, preferred_nodes=("13",))
                    if hud and take.get("hud_video") != hud:
                        take["hud_video"] = hud
                        take["status"] = "completed"
                        take["updated_at"] = time.time()
                        changed = True
                    if clean and take.get("clean_video") != clean:
                        take["clean_video"] = clean
                        take["status"] = "completed"
                        take["updated_at"] = time.time()
                        changed = True
                    if still and take.get("still_image") != still:
                        take["still_image"] = still
                        take["status"] = "completed"
                        take["updated_at"] = time.time()
                        changed = True
            if changed:
                self._save()

    def mark_executed(self, prompt_id: str, node_id: str, output_dict: dict[str, Any]) -> None:
        if not prompt_id or not isinstance(output_dict, dict):
            return
        with self._lock:
            info = self._prompt_to_take.get(str(prompt_id))
            if not info:
                return
            tid, role = info
            take = self._takes.get(tid)
            if take is None:
                return
            vid = first_video_from_node_output(output_dict)
            img = first_image_from_node_output(output_dict)
            changed = False
            if role == "baseline" and vid:
                take["baseline_video"] = vid
                changed = True
            elif role == "edited":
                if str(node_id) == "162" and vid:
                    take["clean_video"] = vid
                    take["status"] = "completed"
                    changed = True
                elif str(node_id) == "13" and vid:
                    take["hud_video"] = vid
                    take["status"] = "completed"
                    changed = True
                elif str(node_id) == "13" and img:
                    take["still_image"] = img
                    take["status"] = "completed"
                    changed = True
            if changed:
                take["updated_at"] = time.time()
                self._save()

    def reconcile_orphaned_queued(self, active_pids: set[str], history_pids: set[str], *, grace_sec: float = 15.0) -> None:
        """Mark stale ``queued`` takes absent from ComfyUI ``/queue`` & ``/history`` as ``interrupted``."""
        now = time.time()
        with self._lock:
            changed = False
            for tid in self._take_order:
                t = self._takes.get(tid)
                if not t or t.get("status") != "queued":
                    continue
                pid = str(t.get("prompt_id") or "")
                age = now - float(t.get("created_at") or now)
                if pid and pid not in active_pids and pid not in history_pids and age >= grace_sec:
                    t["status"] = "interrupted"
                    if not t.get("error"):
                        t["error"] = "Job no longer in ComfyUI queue (interrupted or server restarted)."
                    t["updated_at"] = now
                    changed = True
            if changed:
                self._save()

    def clear_forgotten(self, *, keep_uploads: set[str] | None = None) -> list[str]:
        keep = set(keep_uploads or ())
        with self._lock:
            kept_takes: dict[str, dict[str, Any]] = {}
            kept_order: list[str] = []
            removed_pids: list[str] = []
            for tid in self._take_order:
                t = self._takes.get(tid)
                if not t:
                    continue
                if t.get("status") == "queued" or t.get("pinned"):
                    kept_takes[tid] = t
                    kept_order.append(tid)
                else:
                    for key in ("prompt_id", "baseline_prompt_id"):
                        pid = str(t.get(key) or "")
                        if pid:
                            removed_pids.append(pid)
                            self._prompt_to_client.pop(pid, None)
                            self._prompt_to_take.pop(pid, None)
            self._takes = kept_takes
            self._take_order = kept_order
            self._uploads = {u: s for u, s in self._uploads.items() if u in keep}
            self._save()
            return removed_pids
