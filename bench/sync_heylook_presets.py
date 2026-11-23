#!/usr/bin/env python3
"""Sync tracked `docs/wiki/system_prompts/` presets to `heylook` without clobbering remote sampler params.

Usage:
    <comfy-venv-python> bench/sync_heylook_presets.py --list
    <comfy-venv-python> bench/sync_heylook_presets.py --dry-run [--diff] [--prefix h3app_]
    <comfy-venv-python> bench/sync_heylook_presets.py --apply [--prefix h3app_] [--only h3-video-ref2va,h3-image-edit] [--reset-params]

Sync guarantees on every update (`PUT /v1/presets/{id}`):
1. Each canonical prompt under `docs/wiki/system_prompts/` is written verbatim (stripping
   any legacy `H3_LOCAL_PREAMBLE` / `H3_LOCAL_ADDENDUM` HTML sentinels).
2. Remote sampler `params` are preserved verbatim (sanitized to `VALID_PRESET_PARAM_KEYS`
   with `None` removed) unless `--reset-params` is explicitly passed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import difflib
from pathlib import Path
import re
import sys
from typing import Any
from urllib.parse import quote

import httpx

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import heylook_client as _hc  # noqa: E402

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
    PresetSpec("h3-masked-edit-steering", "steering", "docs/wiki/system_prompts/h3_masked_region_edit_system_prompt.md", "Masked spatial/temporal region & beat-timed edit prompt"),
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
    diff_text: str = ""


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


def filter_catalog(
    only: list[str] | None = None,
    *,
    repo: Path = REPO,
) -> list[PresetSpec]:
    """Filter `PRESET_CATALOG` by preset name or group."""
    if not only:
        specs = list(PRESET_CATALOG)
    else:
        tokens = {t.strip().lower() for item in only for t in item.split(",") if t.strip()}
        known = {s.name.lower() for s in PRESET_CATALOG} | {s.group.lower() for s in PRESET_CATALOG}
        unknown = sorted(tokens - known)
        if unknown:
            raise ValueError(
                f"unknown --only selector(s) {unknown}; valid names/groups: {sorted(known)}"
            )
        specs = [
            s for s in PRESET_CATALOG
            if s.name.lower() in tokens or s.group.lower() in tokens
        ]
    for s in specs:
        fp = repo / s.rel_path
        if not fp.is_file():
            raise FileNotFoundError(f"canonical system prompt missing: {s.rel_path}")
        if s.fenced_block_index is not None:
            read_spec_text(s, repo=repo)
    return specs


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


def plan_sync(
    specs: list[PresetSpec],
    remote_presets: list[dict[str, Any]],
    *,
    reset_params: bool = False,
    prefix: str | None = None,
    repo: Path = REPO,
) -> list[SyncAction]:
    """Compute idempotent create/update/unchanged sync actions without mutating the server."""
    eff_prefix = str(prefix if prefix is not None else _hc.resolve_heylook_preset_prefix()).strip()
    actions: list[SyncAction] = []

    for spec in specs:
        target_name = format_target_name(spec.name, eff_prefix)
        canonical = read_spec_text(spec, repo=repo)
        remote = _find_remote_preset(remote_presets, target_name)

        old_sys = str((remote or {}).get("system_prompt") or "")
        old_params_raw = (remote or {}).get("params") or {}
        old_params = sanitize_preset_params(old_params_raw)
        new_sys = canonical.replace("\r\n", "\n").replace("\r", "\n").strip()
        new_params = {} if reset_params else dict(old_params)

        if remote is None:
            action = "create"
        elif old_sys.replace("\r\n", "\n").replace("\r", "\n").strip() == new_sys and old_params == new_params:
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
                diff_text="\n".join(diff_lines),
            )
        )

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
    reset_params: bool = False,
    prefix: str | None = None,
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
            reset_params=reset_params,
            prefix=prefix,
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
    ap.add_argument("--only", action="append", default=[], help="filter by preset name or group (video|image|steering|auto_edit_2x|director)")
    ap.add_argument("--prefix", default=None, help="override target preset name prefix (defaults to H3_HEYLOOK_PRESET_PREFIX or none; e.g. h3app_)")
    ap.add_argument("--reset-params", action="store_true", help="clear remote sampler params instead of preserving them")
    ap.add_argument("--list", action="store_true", help="list the tracked preset catalog offline and exit")
    args = ap.parse_args(argv)

    eff_prefix = str(args.prefix if args.prefix is not None else _hc.resolve_heylook_preset_prefix()).strip()
    specs = filter_catalog(args.only or None)
    if args.list:
        for s in specs:
            tname = format_target_name(s.name, eff_prefix)
            print(f"{tname:<28} [{s.group:<12}] {s.rel_path} -- {s.description}")
        return 0

    base_url = _hc.resolve_heylook_base_url(args.base_url)
    actions = execute_sync(
        base_url,
        specs,
        apply=bool(args.apply),
        reset_params=bool(args.reset_params),
        prefix=args.prefix,
    )

    counts = {"create": 0, "update": 0, "unchanged": 0}
    mode_label = "APPLIED" if args.apply else "DRY-RUN"
    for act in actions:
        counts[act.action] = counts.get(act.action, 0) + 1
        flags = [f"params={len(act.new_params)}"] if act.new_params else []
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
