#!/usr/bin/env python3
"""Sync tracked `docs/wiki/system_prompts/` presets to `heylook` without clobbering local overlays or params.

Usage:
    <comfy-venv-python> bench/sync_heylook_presets.py --list
    <comfy-venv-python> bench/sync_heylook_presets.py --dry-run [--diff] [--prefix h3app_]
    <comfy-venv-python> bench/sync_heylook_presets.py --apply [--prefix h3app_] [--only h3-video-ref2va,h3-image-edit] [--reset-params]

Preservation guarantees on every update (`PUT /v1/presets/{id}`):
1. Remote sampler `params` are preserved verbatim (sanitized to the `heylook` preset
   schema keys and stripped of `None`) unless `--reset-params` is explicitly passed.
2. Any local preamble / addendum wrapped in explicit sentinels:
       <!-- H3_LOCAL_PREAMBLE_START --> ... <!-- H3_LOCAL_PREAMBLE_END -->
       <!-- H3_LOCAL_ADDENDUM_START --> ... <!-- H3_LOCAL_ADDENDUM_END -->
   is extracted from the remote `system_prompt` and re-attached around the updated
   canonical repo prompt cleanly on every run (100% idempotent).
3. Optional gitignored overlay TOML (default `internal/heylook_local_overlays.toml`
   when present, or `--overlays <path>`) can configure `[global]` (`prefix`, `preamble`,
   `addendum`, `params`), per-preset `[presets."<name>"]` (`remote_name`, `preamble`,
   `addendum`, `params`), and `[overlay_preset]` (`name`, `directive`, `body`, `params`).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import difflib
from pathlib import Path
import re
import sys
import tomllib
from typing import Any
from urllib.parse import quote

import httpx

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import heylook_client as _hc  # noqa: E402

PREAMBLE_START = "<!-- H3_LOCAL_PREAMBLE_START -->"
PREAMBLE_END = "<!-- H3_LOCAL_PREAMBLE_END -->"
ADDENDUM_START = "<!-- H3_LOCAL_ADDENDUM_START -->"
ADDENDUM_END = "<!-- H3_LOCAL_ADDENDUM_END -->"

DEFAULT_OVERLAYS_PATH = REPO / "internal" / "heylook_local_overlays.toml"

# Exact `/v1/presets` validated `params` key set (`heylook-provider/references/routes.md`).
VALID_PRESET_PARAM_KEYS: frozenset[str] = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "max_tokens",
        "repetition_penalty",
        "repetition_context_size",
        "presence_penalty",
        "enable_thinking",
        "reasoning_effort",
        "seed",
        "thinking_budget_tokens",
    }
)


@dataclass(frozen=True)
class PresetSpec:
    name: str
    group: str
    rel_path: str
    description: str
    fenced_block_index: int | None = None


PRESET_CATALOG: tuple[PresetSpec, ...] = (
    # Standalone H3 Video prompts (pure plain-text section output)
    PresetSpec("h3-video-t2va", "video", "docs/wiki/system_prompts/h3_video/t2va_system_prompt.md", "T2VA 3-section video+audio prompt"),
    PresetSpec("h3-video-i2va", "video", "docs/wiki/system_prompts/h3_video/i2va_system_prompt.md", "I2VA first-frame video+audio prompt"),
    PresetSpec("h3-video-fl2va", "video", "docs/wiki/system_prompts/h3_video/fl2va_system_prompt.md", "FL2VA first+last frame video+audio prompt"),
    PresetSpec("h3-video-l2va", "video", "docs/wiki/system_prompts/h3_video/l2va_system_prompt.md", "L2VA last-frame video+audio prompt"),
    PresetSpec("h3-video-ref2va", "video", "docs/wiki/system_prompts/h3_video/ref2va_system_prompt.md", "Ref2VA 6-section multimodal video+audio prompt"),
    # Standalone MiniMax-H3-Image, P-H3-Image, + general ref-image prompts
    PresetSpec("h3-image-t2i", "image", "docs/wiki/system_prompts/h3_image/t2i_system_prompt.md", "MiniMax-H3-Image general T2I prompt"),
    PresetSpec("h3-image-t2i-ar", "image", "docs/wiki/system_prompts/h3_image/t2i_aspect_ratio_system_prompt.md", "MiniMax-H3-Image general T2I + aspect_ratio"),
    PresetSpec("h3-image-t2i-design", "image", "docs/wiki/system_prompts/h3_image/t2i_design_system_prompt.md", "MiniMax-H3-Image design/typography T2I"),
    PresetSpec("h3-image-t2i-design-ar", "image", "docs/wiki/system_prompts/h3_image/t2i_design_aspect_ratio_system_prompt.md", "MiniMax-H3-Image design T2I + aspect_ratio"),
    PresetSpec("h3-image-edit", "image", "docs/wiki/system_prompts/h3_image/edit_system_prompt.md", "MiniMax-H3-Image 1..16 ref edit prompt"),
    PresetSpec("h3-image-edit-ar", "image", "docs/wiki/system_prompts/h3_image/edit_aspect_ratio_system_prompt.md", "MiniMax-H3-Image 1..16 ref edit + aspect_ratio"),
    PresetSpec("h3-ph3-t2i", "image", "docs/wiki/system_prompts/h3_p_h3_image_system_prompt.md", "P-H3-Image T2I prompt (§1)", fenced_block_index=0),
    PresetSpec("h3-ph3-edit", "image", "docs/wiki/system_prompts/h3_p_h3_image_system_prompt.md", "P-H3-Image direct single-paragraph edit prompt (§2)", fenced_block_index=1),
    PresetSpec("h3-general-image-edit", "image", "docs/wiki/system_prompts/general_image_edit_system_prompt.md", "Pika ref2va / P-H3-Image 4-section visual edit prompt"),
    # Steering & masked guidance prompts
    PresetSpec("h3-control-vector", "steering", "docs/wiki/system_prompts/h3_control_vector_standalone_system_prompt.md", "Contrastive control-vector pair & edit-list prompt"),
    PresetSpec("h3-masked-edit-steering", "steering", "docs/wiki/system_prompts/h3_masked_edit_and_steering_system_prompt.md", "Masked EditGuider + Steer recipe architect prompt"),
    # 2x Surgical Auto-Edit rewrite prompts
    PresetSpec("h3-auto-edit-sections", "auto_edit_2x", "docs/wiki/system_prompts/h3_auto_edit/section_edits_system_prompt.md", "2x AutoEditPrompt section-rewrite prompt"),
    PresetSpec("h3-auto-edit-lines", "auto_edit_2x", "docs/wiki/system_prompts/h3_auto_edit/edit_lines_system_prompt.md", "2x AutoEditPrompt edit-lines prompt"),
    # Interactive Director spatial & multi-shot continuity prompts
    PresetSpec("h3-director-reticle-suggest", "director", "docs/wiki/system_prompts/h3_director/reticle_suggest_system_prompt.md", "Director reticle+ROI grounded action suggestion prompt"),
    PresetSpec("h3-director-scene-suggest", "director", "docs/wiki/system_prompts/h3_director/scene_suggest_system_prompt.md", "Director scene-grounded action suggestion prompt"),
    PresetSpec("h3-director-shot-continue", "director", "docs/wiki/system_prompts/h3_director/shot_continuation_system_prompt.md", "Director closed-loop tail-frame shot continuation prompt"),
)


@dataclass
class SyncAction:
    spec: PresetSpec
    target_name: str
    action: str  # "create" | "update" | "unchanged"
    remote_id: str | None
    old_system_prompt: str
    new_system_prompt: str
    old_params: dict[str, Any]
    new_params: dict[str, Any]
    has_local_blocks: bool = False
    diff_text: str = ""


_PREAMBLE_RE = re.compile(
    re.escape(PREAMBLE_START) + r"(.*?)" + re.escape(PREAMBLE_END),
    re.DOTALL,
)
_ADDENDUM_RE = re.compile(
    re.escape(ADDENDUM_START) + r"(.*?)" + re.escape(ADDENDUM_END),
    re.DOTALL,
)
_FENCED_TEXT_RE = re.compile(r"```text\s*\n(.*?)```", re.DOTALL)


def sanitize_preset_params(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Filter `params` strictly to non-null keys accepted by `/v1/presets`."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k in VALID_PRESET_PARAM_KEYS and v is not None:
            out[k] = v
    return out


def extract_local_blocks(remote_text: str) -> tuple[str, str]:
    """Extract explicit sentinel-wrapped `(preamble, addendum)` blocks from a remote `system_prompt`."""
    norm = (remote_text or "").replace("\r\n", "\n").replace("\r", "\n")
    m_pre = _PREAMBLE_RE.search(norm)
    preamble = m_pre.group(1).strip() if m_pre else ""
    m_add = _ADDENDUM_RE.search(norm)
    addendum = m_add.group(1).strip() if m_add else ""
    return preamble, addendum


def _merge_unique_blocks(remote_block: str, overlay_block: str) -> str:
    """Combine remote and overlay local blocks without duplicating identical text across syncs."""
    r = (remote_block or "").strip()
    o = (overlay_block or "").strip()
    if not o:
        return r
    if not r:
        return o
    if o == r or o in r:
        return r
    if r in o:
        return o
    return f"{r}\n\n{o}"


def compose_synced_system_prompt(
    canonical_text: str,
    *,
    preamble: str = "",
    addendum: str = "",
) -> str:
    """Deterministically compose canonical `.md` prompt with sentinel-fenced local blocks."""
    parts: list[str] = []
    pre = (preamble or "").strip()
    add = (addendum or "").strip()
    body = (canonical_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if pre:
        parts.append(f"{PREAMBLE_START}\n{pre}\n{PREAMBLE_END}")
    if body:
        parts.append(body)
    if add:
        parts.append(f"{ADDENDUM_START}\n{add}\n{ADDENDUM_END}")
    return "\n\n".join(parts)


def read_spec_text(spec: PresetSpec, repo: Path = REPO) -> str:
    """Read raw prompt text from `spec.rel_path`, extracting a ```text block when indexed."""
    raw = (repo / spec.rel_path).read_text(encoding="utf-8")
    if spec.fenced_block_index is None:
        return raw.strip()
    blocks = [m.group(1).strip() for m in _FENCED_TEXT_RE.finditer(raw)]
    if spec.fenced_block_index >= len(blocks):
        raise ValueError(
            f"{spec.rel_path} has {len(blocks)} ```text fenced block(s); expected index {spec.fenced_block_index}"
        )
    return blocks[spec.fenced_block_index]


def format_target_name(slug: str, prefix: str = "") -> str:
    """Format a catalog slug with an optional remote preset prefix (e.g. `h3app_`)."""
    return _hc.format_preset_target_name(slug, prefix)


def load_local_overlays(path: Path | None) -> dict[str, Any]:
    """Load optional gitignored overlay config (`[global]`, `[overlay_preset]`, `[presets.*]`)."""
    empty: dict[str, Any] = {"global": {}, "overlay_preset": {}, "presets": {}}
    if path is None or not path.is_file():
        return empty
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return empty
    g = raw.get("global") if isinstance(raw.get("global"), dict) else {}
    ov_p = raw.get("overlay_preset") if isinstance(raw.get("overlay_preset"), dict) else {}
    p = raw.get("presets") if isinstance(raw.get("presets"), dict) else {}
    return {"global": dict(g), "overlay_preset": dict(ov_p), "presets": dict(p)}


def filter_catalog(
    only: list[str] | None = None,
    *,
    repo: Path = REPO,
    overlay_name: str = "",
) -> tuple[list[PresetSpec], bool]:
    """Filter `PRESET_CATALOG` (and optional overlay preset) by name or group."""
    if not only:
        specs = list(PRESET_CATALOG)
        include_overlay = bool(overlay_name)
    else:
        tokens = {t.strip().lower() for item in only for t in item.split(",") if t.strip()}
        known = {s.name.lower() for s in PRESET_CATALOG} | {s.group.lower() for s in PRESET_CATALOG} | {"overlay"}
        if overlay_name:
            known.add(overlay_name.lower())
        unknown = sorted(tokens - known)
        if unknown:
            raise ValueError(
                f"unknown --only selector(s) {unknown}; valid names/groups: {sorted(known)}"
            )
        specs = [
            s for s in PRESET_CATALOG
            if s.name.lower() in tokens or s.group.lower() in tokens
        ]
        include_overlay = bool(overlay_name) and (
            "overlay" in tokens or overlay_name.lower() in tokens
        )
    for s in specs:
        fp = repo / s.rel_path
        if not fp.is_file():
            raise FileNotFoundError(f"canonical system prompt missing: {s.rel_path}")
        if s.fenced_block_index is not None:
            read_spec_text(s, repo=repo)
    return specs, include_overlay


def _find_remote_preset(remote_presets: list[dict[str, Any]], target_name: str) -> dict[str, Any] | None:
    """Match remote preset strictly by exact name (or case-insensitive exact name)."""
    for p in remote_presets:
        if p.get("name") == target_name:
            return p
    low = target_name.lower()
    for p in remote_presets:
        pname = str(p.get("name") or "").lower()
        if pname == low:
            return p
    return None


def _build_overlay_action(
    ov_cfg: dict[str, Any],
    g_ov: dict[str, Any],
    remote_presets: list[dict[str, Any]],
    *,
    reset_params: bool,
) -> SyncAction:
    ov_name = str(ov_cfg.get("name") or "").strip()
    directive = str(ov_cfg.get("directive") or "@extend:prepend").strip()
    dir_mode, dir_inline = _hc.parse_extend_directive(directive)
    if dir_mode is None:
        raise ValueError(
            f"[overlay_preset].directive must start with '@extend:prepend|append' or "
            f"'<!-- H3_EXTEND:prepend|append -->', got {directive!r}"
        )
    raw_body = ov_cfg.get("body") if "body" in ov_cfg else g_ov.get("preamble", "")
    _mode, clean_body = _hc.parse_extend_directive(str(raw_body or ""))
    new_sys = f"{directive}\n\n{clean_body}" if (clean_body and not dir_inline) else (
        f"{directive}\n{clean_body}".strip() if clean_body else directive
    )

    remote = _find_remote_preset(remote_presets, ov_name)
    old_sys = str((remote or {}).get("system_prompt") or "")
    old_params = sanitize_preset_params((remote or {}).get("params") or {})

    base_params: dict[str, Any] = {} if reset_params else dict(old_params)
    if isinstance(ov_cfg.get("params"), dict):
        base_params.update(sanitize_preset_params(ov_cfg["params"]))
    new_params = sanitize_preset_params(base_params)

    if remote is None:
        action = "create"
    elif old_sys.replace("\r\n", "\n").strip() == new_sys and old_params == new_params:
        action = "unchanged"
    else:
        action = "update"

    diff_lines = list(
        difflib.unified_diff(
            old_sys.splitlines(),
            new_sys.splitlines(),
            fromfile=f"remote/{ov_name}",
            tofile=f"overlay/{ov_name}",
            lineterm="",
        )
    )
    spec = PresetSpec(ov_name, "overlay", "<local_overlay>", f"Sticky {directive} overlay preset")
    return SyncAction(
        spec=spec,
        target_name=ov_name,
        action=action,
        remote_id=str(remote.get("id")) if remote and remote.get("id") else None,
        old_system_prompt=old_sys,
        new_system_prompt=new_sys,
        old_params=old_params,
        new_params=new_params,
        has_local_blocks=bool(clean_body),
        diff_text="\n".join(diff_lines),
    )


def plan_sync(
    specs: list[PresetSpec],
    remote_presets: list[dict[str, Any]],
    *,
    overlays: dict[str, Any] | None = None,
    reset_params: bool = False,
    prefix: str | None = None,
    include_overlay: bool = False,
    repo: Path = REPO,
) -> list[SyncAction]:
    """Compute idempotent create/update/unchanged sync actions without mutating the server."""
    ov = overlays or {"global": {}, "overlay_preset": {}, "presets": {}}
    g_ov = ov.get("global") or {}
    ov_preset = ov.get("overlay_preset") or {}
    p_map = ov.get("presets") or {}
    eff_prefix = str(prefix if prefix is not None else (g_ov.get("prefix") or ""))
    actions: list[SyncAction] = []

    for spec in specs:
        p_ov = p_map.get(spec.name) if isinstance(p_map.get(spec.name), dict) else {}
        default_name = format_target_name(spec.name, eff_prefix)
        target_name = str(p_ov.get("remote_name") or default_name).strip() or default_name
        canonical = read_spec_text(spec, repo=repo)
        remote = _find_remote_preset(remote_presets, target_name)

        old_sys = str((remote or {}).get("system_prompt") or "")
        old_params_raw = (remote or {}).get("params") or {}
        old_params = sanitize_preset_params(old_params_raw)

        remote_pre, remote_add = extract_local_blocks(old_sys)
        ov_pre = str(p_ov.get("preamble") if "preamble" in p_ov else g_ov.get("preamble") or "").strip()
        ov_add = str(p_ov.get("addendum") if "addendum" in p_ov else g_ov.get("addendum") or "").strip()

        eff_pre = _merge_unique_blocks(remote_pre, ov_pre)
        eff_add = _merge_unique_blocks(remote_add, ov_add)

        new_sys = compose_synced_system_prompt(canonical, preamble=eff_pre, addendum=eff_add)

        if reset_params:
            base_params: dict[str, Any] = {}
        else:
            base_params = dict(old_params)
        if isinstance(g_ov.get("params"), dict):
            base_params.update(sanitize_preset_params(g_ov["params"]))
        if isinstance(p_ov.get("params"), dict):
            base_params.update(sanitize_preset_params(p_ov["params"]))
        new_params = sanitize_preset_params(base_params)

        if remote is None:
            action = "create"
        elif old_sys.replace("\r\n", "\n").strip() == new_sys and old_params == new_params:
            action = "unchanged"
        else:
            action = "update"

        diff_lines = list(
            difflib.unified_diff(
                old_sys.splitlines(),
                new_sys.splitlines(),
                fromfile=f"remote/{target_name}",
                tofile=f"repo/{spec.rel_path}",
                lineterm="",
            )
        )
        actions.append(
            SyncAction(
                spec=spec,
                target_name=target_name,
                action=action,
                remote_id=str(remote.get("id")) if remote and remote.get("id") else None,
                old_system_prompt=old_sys,
                new_system_prompt=new_sys,
                old_params=old_params,
                new_params=new_params,
                has_local_blocks=bool(eff_pre or eff_add),
                diff_text="\n".join(diff_lines),
            )
        )

    if include_overlay and str(ov_preset.get("name") or "").strip():
        actions.append(_build_overlay_action(ov_preset, g_ov, remote_presets, reset_params=reset_params))
    return actions


def fetch_remote_presets(
    client: httpx.Client,
    base_url: str,
    *,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    hdrs = _hc._req_headers({"Accept": "application/json"}, client=client, api_key=api_key)
    r = client.get(f"{base_url}/v1/presets", headers=hdrs, timeout=10.0)
    if r.status_code >= 400:
        raise _hc._http_error(r, base_url, "/v1/presets", sent_auth=bool(hdrs.get("x-api-key")))
    payload = r.json()
    return [p for p in (payload.get("presets") or []) if isinstance(p, dict)]


def execute_sync(
    base_url: str,
    specs: list[PresetSpec],
    *,
    apply: bool = False,
    overlays: dict[str, Any] | None = None,
    reset_params: bool = False,
    prefix: str | None = None,
    include_overlay: bool = False,
    client: httpx.Client | None = None,
    api_key: str | None = None,
    repo: Path = REPO,
) -> list[SyncAction]:
    """Fetch `/v1/presets`, plan changes, and optionally apply `POST` / `PUT` updates."""
    base = _hc.normalize_heylook_base_url(base_url)

    def _run(c: httpx.Client) -> list[SyncAction]:
        remote_presets = fetch_remote_presets(c, base, api_key=api_key)
        actions = plan_sync(
            specs,
            remote_presets,
            overlays=overlays,
            reset_params=reset_params,
            prefix=prefix,
            include_overlay=include_overlay,
            repo=repo,
        )
        if not apply:
            return actions
        write_hdrs = _hc._req_headers({"Content-Type": "application/json"}, client=c, api_key=api_key)
        for applied_idx, act in enumerate(actions):
            body: dict[str, Any] = {
                "name": act.target_name,
                "system_prompt": act.new_system_prompt,
                "params": act.new_params,
            }
            if act.action == "create":
                r = c.post(f"{base}/v1/presets", json=body, headers=write_hdrs, timeout=10.0)
                if r.status_code >= 400:
                    err = _hc._http_error(r, base, "/v1/presets", sent_auth=bool(write_hdrs.get("x-api-key")))
                    raise RuntimeError(
                        f"failed creating preset {act.target_name!r} (after {applied_idx} prior action(s)): {err}"
                    ) from err
                created = r.json()
                if isinstance(created, dict) and created.get("id"):
                    act.remote_id = str(created["id"])
            elif act.action == "update":
                if not act.remote_id:
                    raise RuntimeError(f"cannot PUT preset {act.target_name!r}: missing remote id")
                quoted_id = quote(act.remote_id, safe="")
                r = c.put(f"{base}/v1/presets/{quoted_id}", json=body, headers=write_hdrs, timeout=10.0)
                if r.status_code >= 400:
                    err = _hc._http_error(
                        r, base, f"/v1/presets/{quoted_id}", sent_auth=bool(write_hdrs.get("x-api-key"))
                    )
                    raise RuntimeError(
                        f"failed updating preset {act.target_name!r} (id={act.remote_id!r}, after {applied_idx} prior action(s)): {err}"
                    ) from err
        return actions

    if client is not None:
        return _run(client)
    with _hc.make_heylook_client(base, timeout=10.0, api_key=api_key) as own:
        return _run(own)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="", help="heylook server URL (defaults to H3_HEYLOOK_URL / h3.env)")
    mode_grp = ap.add_mutually_exclusive_group()
    mode_grp.add_argument("--dry-run", action="store_true", help="plan only without writing (default)")
    mode_grp.add_argument("--apply", action="store_true", help="POST/PUT changed presets to heylook")
    ap.add_argument("--diff", action="store_true", help="print unified diffs for create/update actions")
    ap.add_argument("--only", action="append", default=[], help="filter by preset name, overlay, or group (video|image|steering|auto_edit_2x|director)")
    ap.add_argument("--prefix", default=None, help="override target preset name prefix (e.g. h3app_)")
    ap.add_argument("--reset-params", action="store_true", help="clear remote sampler params instead of preserving them")
    ap.add_argument("--overlays", type=Path, default=DEFAULT_OVERLAYS_PATH, help="optional local overlay TOML path")
    ap.add_argument("--list", action="store_true", help="list the tracked preset catalog offline and exit")
    args = ap.parse_args(argv)

    overlays = load_local_overlays(args.overlays)
    ov_name = str((overlays.get("overlay_preset") or {}).get("name") or "").strip()
    eff_prefix = str(args.prefix if args.prefix is not None else ((overlays.get("global") or {}).get("prefix") or ""))
    specs, include_overlay = filter_catalog(args.only or None, overlay_name=ov_name)
    if args.list:
        p_map = overlays.get("presets") or {}
        for s in specs:
            p_ov = p_map.get(s.name) if isinstance(p_map.get(s.name), dict) else {}
            default_name = format_target_name(s.name, eff_prefix)
            tname = str(p_ov.get("remote_name") or default_name).strip() or default_name
            print(f"{tname:<28} [{s.group:<12}] {s.rel_path} -- {s.description}")
        if include_overlay:
            print(f"{ov_name:<28} [{'overlay':<12}] <local_overlay> -- Sticky {(overlays.get('overlay_preset') or {}).get('directive', '@extend:prepend')} overlay")
        return 0

    base_url = _hc.resolve_heylook_base_url(args.base_url)
    actions = execute_sync(
        base_url,
        specs,
        apply=bool(args.apply),
        overlays=overlays,
        reset_params=bool(args.reset_params),
        prefix=args.prefix,
        include_overlay=include_overlay,
    )

    counts = {"create": 0, "update": 0, "unchanged": 0}
    mode_label = "APPLIED" if args.apply else "DRY-RUN"
    for act in actions:
        counts[act.action] = counts.get(act.action, 0) + 1
        flags = []
        if act.has_local_blocks:
            flags.append("local-blocks")
        if act.new_params:
            flags.append(f"params={len(act.new_params)}")
        flag_str = f" ({', '.join(flags)})" if flags else ""
        print(f"  {act.action:<9} {act.target_name:<28} <- {act.spec.rel_path}{flag_str}")
        if args.diff and act.diff_text:
            print(act.diff_text)

    print(
        f"\n[{mode_label}] create={counts['create']} update={counts['update']} "
        f"unchanged={counts['unchanged']} total={len(actions)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
