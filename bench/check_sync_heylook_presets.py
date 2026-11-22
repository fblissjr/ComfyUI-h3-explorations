#!/usr/bin/env python3
"""Offline checks for `bench/sync_heylook_presets.py`: catalog, exact matching, sentinel blocks, `--prefix`, & `[overlay_preset]` idempotency."""

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


def check_exact_matching_and_sentinel_blocks() -> str:
    # Existing underscore `h3_image_t2i` MUST NOT match hyphenated `h3-image-t2i`.
    # Only explicit `H3_LOCAL_PREAMBLE` / `H3_LOCAL_ADDENDUM` sentinels are preserved across syncs.
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
                f"{shp.PREAMBLE_START}\nFavor 35mm film grain.\n{shp.PREAMBLE_END}\n\n"
                "Outdated body text that must be replaced.\n\n"
                f"{shp.ADDENDUM_START}\nPreserve subject lighting.\n{shp.ADDENDUM_END}\n"
            ),
            "params": {"temperature": 0.42, "bogus_key": 99, "top_p": None},
        },
        {
            "id": "p-unfenced",
            "name": "h3-image-t2i-ar",
            "system_prompt": (
                "Outdated intro.\n\n"
                "1. Opening framing:\nOld text.\n"
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
    assert by_name["h3-image-edit"].has_local_blocks is True
    assert shp.extract_local_blocks(by_name["h3-image-edit"].new_system_prompt) == (
        "Favor 35mm film grain.",
        "Preserve subject lighting.",
    )
    assert "Outdated body text that must be replaced." not in by_name["h3-image-edit"].new_system_prompt
    assert by_name["h3-image-edit"].new_params == {"temperature": 0.42}
    assert by_name["h3-image-t2i-ar"].has_local_blocks is False
    assert shp.extract_local_blocks(by_name["h3-image-t2i-ar"].new_system_prompt) == ("", "")
    assert "Old text." not in by_name["h3-image-t2i-ar"].new_system_prompt
    return "exact name avoids h3_image_t2i collision, sanitizes params, & preserves only explicit sentinel blocks"


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
            'preamble = "Favor crisp 35mm natural framing."\n\n'
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
        assert mode == "prepend" and "Favor crisp 35mm natural framing." in body

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


def check_overlay_directive_validation() -> str:
    from bench import sync_heylook_presets as ps

    # 1. Sentinel extraction ignores un-fenced prose and parses both explicit sentinel blocks.
    assert ps.extract_local_blocks("Plain un-fenced text\nwith multiple lines.") == ("", "")
    fenced_prompt = (
        f"{ps.PREAMBLE_START}\nLead rule.\n{ps.PREAMBLE_END}\n\n"
        "Canonical body.\n\n"
        f"{ps.ADDENDUM_START}\nTail rule.\n{ps.ADDENDUM_END}\n"
    )
    assert ps.extract_local_blocks(fenced_prompt) == ("Lead rule.", "Tail rule.")

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

    return "sentinel extraction ignores un-fenced text & overlay directive validation rejects bare 'prepend'"


def check_access_gate_auth_forwarding() -> str:
    import json
    import os
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    from bench import sync_heylook_presets as ps

    secret = "hlk_sync_secret_987"
    seen: list[tuple[str, str, str | None]] = []

    class _AuthHandler(BaseHTTPRequestHandler):
        def log_message(self, _fmt, *_args):
            return

        def _json(self, code: int, obj: dict) -> None:
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _check_auth(self, method: str) -> bool:
            got = self.headers.get("x-api-key")
            seen.append((method, self.path, got))
            if got != secret:
                self._json(
                    401,
                    {"error": {"type": "authentication_error", "code": "auth_required", "message": "auth required"}},
                )
                return False
            return True

        def do_GET(self):
            if not self._check_auth("GET"):
                return
            self._json(
                200,
                {
                    "presets": [
                        {
                            "id": "p-upd",
                            "name": ps.format_target_name("h3-image-t2i", ""),
                            "system_prompt": "OUTDATED",
                            "params": {},
                        }
                    ]
                },
            )

        def do_POST(self):
            if not self._check_auth("POST"):
                return
            length = int(self.headers.get("Content-Length") or 0)
            _ = self.rfile.read(length)
            self._json(201, {"id": "p-new"})

        def do_PUT(self):
            if not self._check_auth("PUT"):
                return
            length = int(self.headers.get("Content-Length") or 0)
            _ = self.rfile.read(length)
            self._json(200, {"id": "p-upd"})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _AuthHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    prev_key = os.environ.get("HEYLOOK_API_KEY")
    prev_kfile = os.environ.get("HEYLOOK_API_KEY_FILE")
    prev_h3_kfile = os.environ.get("H3_HEYLOOK_API_KEY_FILE")
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        specs = ps.filter_catalog(["h3-image-t2i", "h3-image-t2i-design"])[0]
        assert len(specs) == 2

        # 0. Control: --api-key CLI flag is refused so secrets never appear on argv
        try:
            ps.main(["--api-key", secret, "--list"])
            raise AssertionError("expected SystemExit for removed --api-key flag")
        except SystemExit as exc:
            assert exc.code == 2, exc

        # 1. Missing key -> 401 PermissionError without issuing any writes
        os.environ.pop("HEYLOOK_API_KEY", None)
        os.environ.pop("H3_HEYLOOK_API_KEY_FILE", None)
        os.environ["HEYLOOK_API_KEY_FILE"] = ""
        try:
            ps.execute_sync(base, specs, apply=True)
            raise AssertionError("expected PermissionError on 401")
        except PermissionError as exc:
            assert "401" in str(exc) and secret not in str(exc), exc
        assert not any(m in ("POST", "PUT") for m, _p, _k in seen), seen

        # 2. Programmatic api_key arg still works when env/file are unset
        seen.clear()
        actions_arg = ps.execute_sync(base, specs, apply=True, api_key=secret)
        assert len(actions_arg) == 2 and all(k == secret for _m, _p, k in seen), (actions_arg, seen)

        # 3. HEYLOOK_API_KEY forwards x-api-key on GET, PUT, and POST
        seen.clear()
        os.environ["HEYLOOK_API_KEY"] = secret
        actions = ps.execute_sync(base, specs, apply=True)
        assert len(actions) == 2 and all(k == secret for _m, _p, k in seen), (actions, seen)
        assert any(m == "PUT" for m, _p, _k in seen) and any(m == "POST" for m, _p, _k in seen), seen
    finally:
        srv.shutdown()
        srv.server_close()
        if prev_key is None:
            os.environ.pop("HEYLOOK_API_KEY", None)
        else:
            os.environ["HEYLOOK_API_KEY"] = prev_key
        if prev_kfile is None:
            os.environ.pop("HEYLOOK_API_KEY_FILE", None)
        else:
            os.environ["HEYLOOK_API_KEY_FILE"] = prev_kfile
        if prev_h3_kfile is None:
            os.environ.pop("H3_HEYLOOK_API_KEY_FILE", None)
        else:
            os.environ["H3_HEYLOOK_API_KEY_FILE"] = prev_h3_kfile
    return "sync_heylook_presets removes --api-key CLI flag, keeps programmatic api_key, forwards x-api-key, and fails fast on 401"


def main() -> int:
    case("catalog_and_fenced_blocks", check_catalog_and_fenced_blocks)
    case("exact_matching_and_sentinel_blocks", check_exact_matching_and_sentinel_blocks)
    case("prefix_overlay_preset_and_idempotency", check_prefix_overlay_preset_and_idempotency)
    case("overlay_directive_validation", check_overlay_directive_validation)
    case("access_gate_auth_forwarding", check_access_gate_auth_forwarding)
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
