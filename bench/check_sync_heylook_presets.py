#!/usr/bin/env python3
"""Offline checks for `bench/sync_heylook_presets.py`: sentinel preservation, legacy migration, params retention, & idempotency."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import REPO, case, finish  # noqa: E402
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


def check_catalog_files_exist() -> str:
    specs = shp.filter_catalog(None, repo=REPO)
    assert len(specs) >= 16, len(specs)
    for spec in specs:
        txt = (REPO / spec.rel_path).read_text(encoding="utf-8").strip()
        assert len(txt) > 80, (spec.name, len(txt))
    return f"{len(specs)} catalog system prompts present on disk"


def check_dry_run_migration_and_sentinels() -> str:
    remote = [
        {
            "id": "p-t2va",
            "name": "h3-video-t2va",
            "system_prompt": (
                f"{shp.PREAMBLE_START}\nUse dry British understatement.\n{shp.PREAMBLE_END}\n\n"
                "STALE PROMPT BODY\n\n"
                f"{shp.ADDENDUM_START}\nNever censor atmospheric smoke.\n{shp.ADDENDUM_END}"
            ),
            "params": {"temperature": 0.42, "enable_thinking": True, "bogus_key": 99, "top_p": None},
        },
        {
            "id": "p-ref2va",
            "name": "h3-video-ref2va",
            "system_prompt": (
                "OLD REF2VA BODY\n\n"
                "## POLICY SCOPE\nKeep adult studio realism intact.\n"
            ),
            "params": {"temperature": 0.55, "reasoning_effort": "high"},
        },
    ]
    ops: list[tuple[str, str, dict | None]] = []
    client = _mock_client(remote, ops)
    specs = shp.filter_catalog(["video"], repo=REPO)

    dry_actions = shp.execute_sync("http://127.0.0.1:8080", specs, apply=False, client=client, repo=REPO)
    assert all(m == "GET" for m, _, _ in ops), f"dry-run made write call: {ops}"
    by_name = {a.target_name: a for a in dry_actions}
    assert by_name["h3-video-t2va"].action == "update"
    assert by_name["h3-video-t2va"].new_params == {"temperature": 0.42, "enable_thinking": True}
    assert "Use dry British understatement." in by_name["h3-video-t2va"].new_system_prompt
    assert "Never censor atmospheric smoke." in by_name["h3-video-t2va"].new_system_prompt
    assert by_name["h3-video-ref2va"].migrated_legacy is True
    assert shp.ADDENDUM_START in by_name["h3-video-ref2va"].new_system_prompt
    assert "Keep adult studio realism intact." in by_name["h3-video-ref2va"].new_system_prompt
    assert by_name["h3-video-i2va"].action == "create"
    return "dry-run performs zero writes, migrates legacy POLICY SCOPE, sanitizes params, & preserves sentinels"


def check_apply_idempotency_overlays_and_reset() -> str:
    remote = [
        {
            "id": "p-img",
            "name": "Custom T2I Alias",
            "system_prompt": "Outdated caption prompt.",
            "params": {"temperature": 0.85, "seed": 123},
        }
    ]
    ops: list[tuple[str, str, dict | None]] = []
    client = _mock_client(remote, ops)
    specs = shp.filter_catalog(["h3-image-t2i"], repo=REPO)

    with tempfile.TemporaryDirectory() as tmp:
        ov_path = Path(tmp) / "overlays.toml"
        ov_path.write_text(
            '[global]\naddendum = "Global studio voice rule."\n\n'
            '[presets."h3-image-t2i"]\n'
            'remote_name = "Custom T2I Alias"\n'
            'preamble = "Local T2I preamble."\n'
            '[presets."h3-image-t2i".params]\n'
            "top_p = 0.92\n",
            encoding="utf-8",
        )
        overlays = shp.load_local_overlays(ov_path)
        a1 = shp.execute_sync("http://127.0.0.1:8080", specs, apply=True, overlays=overlays, client=client, repo=REPO)
        assert a1[0].action == "update" and a1[0].target_name == "Custom T2I Alias"
        assert a1[0].new_params == {"temperature": 0.85, "seed": 123, "top_p": 0.92}
        assert "Local T2I preamble." in remote[0]["system_prompt"]
        assert "Global studio voice rule." in remote[0]["system_prompt"]

        # Second apply with identical inputs must report unchanged and perform 0 PUT/POSTs.
        ops.clear()
        a2 = shp.execute_sync("http://127.0.0.1:8080", specs, apply=True, overlays=overlays, client=client, repo=REPO)
        assert a2[0].action == "unchanged", a2[0].diff_text
        assert [m for m, _, _ in ops] == ["GET"], ops

        # --reset-params drops prior remote params while keeping overlay params.
        a3 = shp.execute_sync(
            "http://127.0.0.1:8080",
            specs,
            apply=True,
            overlays=overlays,
            reset_params=True,
            client=client,
            repo=REPO,
        )
        assert a3[0].action == "update"
        assert remote[0]["params"] == {"top_p": 0.92}

    return "apply preserves & merges overlays idempotently, and --reset-params drops stale remote params"


def main() -> int:
    case("catalog_files_exist", check_catalog_files_exist)
    case("dry_run_migration_and_sentinels", check_dry_run_migration_and_sentinels)
    case("apply_idempotency_overlays_and_reset", check_apply_idempotency_overlays_and_reset)
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
