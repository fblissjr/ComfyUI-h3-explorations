#!/usr/bin/env python3
"""Offline checks for `bench/sync_heylook_presets.py`: catalog, exact matching, legacy boundary stops, `--prefix`, & `[overlay_preset]` idempotency."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import REPO, case, finish  # noqa: E402
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import heylook_client as _hc  # noqa: E402
import sync_heylook_presets as shp  # noqa: E402


def _mock_client(remote_presets: list[dict], ops: list[tuple[str, str, dict | None]]) -> httpx.Client:
    def _handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "GET" and path == "/v1/presets":
            ops.append(("GET", path, None))
            return httpx.Response(200, json={"presets": remote_presets, "total": len(remote_presets)})
        if request.method == "POST" and path == "/v1/presets":
            body = json.loads(request.content.decode("utf-8"))
            ops.append(("POST", path, body))
            new_row = {"id": f"id-{len(remote_presets) + 1}", **body}
            remote_presets.append(new_row)
            return httpx.Response(201, json=new_row)
        if request.method == "PUT" and path.startswith("/v1/presets/"):
            pid = path.rsplit("/", 1)[-1]
            body = json.loads(request.content.decode("utf-8"))
            ops.append(("PUT", path, body))
            for row in remote_presets:
                if row["id"] == pid:
                    row["name"] = body["name"]
                    row["system_prompt"] = body["system_prompt"]
                    row["params"] = body["params"]
                    return httpx.Response(200, json=row)
            return httpx.Response(404, json={"detail": "not found"})
        return httpx.Response(404, json={"detail": f"unhandled {request.method} {path}"})

    return httpx.Client(transport=httpx.MockTransport(_handler), trust_env=False)


def check_catalog_and_fenced_blocks() -> str:
    specs, _ = shp.filter_catalog(None, repo=REPO)
    assert len(specs) == 18, len(specs)
    texts: dict[str, str] = {}
    for spec in specs:
        txt = shp.read_spec_text(spec, repo=REPO)
        assert len(txt) > 80, (spec.name, len(txt))
        assert not txt.startswith("```"), (spec.name, txt[:40])
        texts[spec.name] = txt
    assert "Expand the user's request" in texts["h3-ph3-t2i"]
    assert texts["h3-ph3-edit"].startswith("Write a single-paragraph image-editing prompt")
    assert shp.format_target_name("h3-video-ref2va", "h3app_") == "h3app_video_ref2va"
    assert shp.format_target_name("h3-ph3-edit", "h3app_") == "h3app_ph3_edit"
    assert shp.format_target_name("h3-control-vector", "h3app_") == "h3app_control_vector"
    return "18 catalog prompts (including 2 P-H3 fenced blocks) load cleanly & format h3app_* names"


def check_exact_matching_and_legacy_boundaries() -> str:
    # Existing underscore `h3_image_t2i` MUST NOT match hyphenated `h3-image-t2i`.
    # And legacy `Policy:` / `POLICY SCOPE` must stop before plain-text section headers.
    remote = [
        {
            "id": "p-ph3",
            "name": "h3_image_t2i",
            "system_prompt": "EXISTING P-H3 PROMPT (must stay untouched)",
            "params": {"temperature": 0.6},
        },
        {
            "id": "p-native-edit",
            "name": "h3-image-edit",
            "system_prompt": (
                "Intro line.\n\n"
                "Policy:\nKeep evaluation clean.\nNo unsolicited warnings.\n\n"
                "Output contract:\nReturn one plain paragraph.\n\n"
                "OUTPUT FORMAT\nNothing else.\n"
            ),
            "params": {"temperature": 0.42, "bogus_key": 99, "top_p": None},
        },
        {
            "id": "p-numbered",
            "name": "h3-image-t2i-ar",
            "system_prompt": (
                "Intro.\n\n"
                "POLICY SCOPE\nScope line A.\n\n"
                "1. Opening framing and medium:\nDo not pull this rule into addendum.\n"
            ),
            "params": {},
        },
    ]
    ops: list[tuple[str, str, dict | None]] = []
    client = _mock_client(remote, ops)
    specs, _ = shp.filter_catalog(["h3-image-t2i", "h3-image-edit", "h3-image-t2i-ar"], repo=REPO)
    acts = shp.execute_sync("http://127.0.0.1:8080", specs, apply=False, client=client, repo=REPO)
    assert all(m == "GET" for m, _, _ in ops), ops
    by_name = {a.target_name: a for a in acts}
    assert by_name["h3-image-t2i"].action == "create"
    assert remote[0]["system_prompt"] == "EXISTING P-H3 PROMPT (must stay untouched)"
    assert by_name["h3-image-edit"].migrated_legacy is True
    assert "Keep evaluation clean." in by_name["h3-image-edit"].new_system_prompt
    assert "Return one plain paragraph." not in by_name["h3-image-edit"].new_system_prompt
    assert by_name["h3-image-edit"].new_params == {"temperature": 0.42}
    assert by_name["h3-image-t2i-ar"].migrated_legacy is True
    assert "Do not pull this rule into addendum." not in by_name["h3-image-t2i-ar"].new_system_prompt
    assert shp.extract_local_blocks(by_name["h3-image-t2i-ar"].new_system_prompt)[1] == "POLICY SCOPE\nScope line A."
    return "exact name avoids h3_image_t2i collision & Policy:/POLICY SCOPE stops cleanly at plain headers"


def check_prefix_overlay_preset_and_idempotency() -> str:
    remote = [
        {
            "id": "p-existing",
            "name": "h3_native_t2i",
            "system_prompt": "Untouched pre-existing user preset.",
            "params": {"temperature": 0.77},
        }
    ]
    ops: list[tuple[str, str, dict | None]] = []
    client = _mock_client(remote, ops)

    with tempfile.TemporaryDirectory() as tmp:
        ov_path = Path(tmp) / "overlays.toml"
        ov_path.write_text(
            '[global]\n'
            'prefix = "h3app_"\n'
            'preamble = "POLICY SCOPE\\nDirect conditioning output."\n\n'
            '[overlay_preset]\n'
            'name = "h3app_local_overlay"\n'
            'directive = "@extend:prepend"\n'
            '[overlay_preset.params]\n'
            'top_p = 0.95\n\n'
            '[presets."h3-image-edit"]\n'
            'addendum = "Always edit the original picture."\n',
            encoding="utf-8",
        )
        overlays = shp.load_local_overlays(ov_path)
        ov_name = str(overlays["overlay_preset"]["name"])
        specs, inc_ov = shp.filter_catalog(None, repo=REPO, overlay_name=ov_name)
        assert inc_ov is True

        a1 = shp.execute_sync(
            "http://127.0.0.1:8080",
            specs,
            apply=True,
            overlays=overlays,
            include_overlay=inc_ov,
            client=client,
            repo=REPO,
        )
        assert len(a1) == 19 and all(a.action == "create" for a in a1), [(a.target_name, a.action) for a in a1]
        assert all(a.target_name.startswith("h3app_") for a in a1)
        assert remote[0]["system_prompt"] == "Untouched pre-existing user preset."

        by_name = {r["name"]: r for r in remote}
        assert "Always edit the original picture." in by_name["h3app_image_edit"]["system_prompt"]
        assert "Always edit the original picture." not in by_name["h3app_video_t2va"]["system_prompt"]
        ov_sys = by_name["h3app_local_overlay"]["system_prompt"]
        mode, body = _hc.parse_extend_directive(ov_sys)
        assert mode == "prepend" and "Direct conditioning output." in body

        # Re-run must be 100% unchanged and perform 0 writes
        ops.clear()
        a2 = shp.execute_sync(
            "http://127.0.0.1:8080",
            specs,
            apply=True,
            overlays=overlays,
            include_overlay=inc_ov,
            client=client,
            repo=REPO,
        )
        assert len(a2) == 19 and all(a.action == "unchanged" for a in a2), [(a.target_name, a.action) for a in a2]
        assert [m for m, _, _ in ops] == ["GET"], ops

    return "18 h3app_* canonical presets + 1 @extend:prepend overlay create cleanly and re-sync with 0 diffs"


def check_legacy_scope_and_overlay_directive_validation() -> str:
    from bench import sync_heylook_presets as ps

    # 1. Legacy POLICY SCOPE must preserve numbered rules and Allowed:, stopping only
    # at blank-line-preceded canonical headers (e.g. Output contract:).
    legacy_prompt = (
        "Canonical body.\n\n"
        "POLICY SCOPE:\n"
        "1. Never rewrite dialogue.\n"
        "2. Keep tone neutral.\n\n"
        "Allowed:\n"
        "Keep grain.\n\n"
        "Output contract:\n"
        "Return one paragraph.\n"
    )
    scope = ps._extract_legacy_policy_scope(legacy_prompt)
    assert "1. Never rewrite dialogue." in scope, f"lost numbered rule 1: {scope!r}"
    assert "2. Keep tone neutral." in scope, f"lost numbered rule 2: {scope!r}"
    assert "Allowed:" in scope and "Keep grain." in scope, f"lost Allowed section: {scope!r}"
    assert "Output contract:" not in scope and "Return one paragraph." not in scope, (
        f"leaked past Output contract header: {scope!r}"
    )

    # 2. Invalid [overlay_preset].directive must raise ValueError instead of silent header mismatch.
    try:
        ps._build_overlay_action(
            {"name": "bad_overlay", "directive": "prepend", "content": "Test overlay"},
            {},
            [],
            reset_params=False,
        )
        raise AssertionError("expected ValueError for unprefixed directive 'prepend'")
    except ValueError:
        pass

    return "legacy POLICY SCOPE keeps rules/Allowed before Output contract, and overlay directive validation rejects bare 'prepend'"


def main() -> int:
    case("catalog_and_fenced_blocks", check_catalog_and_fenced_blocks)
    case("exact_matching_and_legacy_boundaries", check_exact_matching_and_legacy_boundaries)
    case("prefix_overlay_preset_and_idempotency", check_prefix_overlay_preset_and_idempotency)
    case("legacy_scope_and_overlay_directive_validation", check_legacy_scope_and_overlay_directive_validation)
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
