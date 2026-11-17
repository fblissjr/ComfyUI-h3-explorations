#!/usr/bin/env python3
"""Offline unit check for `bench/run_with_preview_gate.py`.

Exercises in-memory `ModelPreviewOverrideKJ` injection (`taeh3.safetensors` on multi-frame video,
`taeh3_image.safetensors` on still graphs, graceful skip when `/object_info` lacks it), WebSocket
`"kj_preview_override"` frame capture to `steps/` + `steps.jsonl`, early cancellation via
`--stop-after-step` + `POST /interrupt`, and full `/history` output collection against an in-process
`aiohttp` mock ComfyUI server (zero GPU, zero network, zero disk pollution outside a temp dir).
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from aiohttp import web
from aiohttp.test_utils import TestServer

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import run_with_preview_gate as gate  # noqa: E402
from _lib import case, finish  # noqa: E402

SAMPLE_PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nFAKE_OUTPUT").decode("ascii")
SAMPLE_JPG = base64.b64encode(b"\xff\xd8\xff\xe0FAKE_STEP_JPEG").decode("ascii")
SAMPLE_MP4 = base64.b64encode(b"\x00\x00\x00\x18ftypisomFAKE_MP4").decode("ascii")

MOCK_OBJECT_INFO = {
    gate.PREVIEW_OVERRIDE_NODE: {
        "input": {
            "required": {
                "model": ["MODEL"],
                "max_resolution": ["INT", {"default": 1024}],
                "jpeg_quality": ["INT", {"default": 80}],
                "suppress_default_preview": ["BOOLEAN", {"default": True}],
                "preview_frames": ["INT", {"default": 1}],
                "preview_fps": ["INT", {"default": 12}],
            },
            "optional": {
                "vae": ["VAE"],
                "tiny_vae": [["none", gate.TAEH3_VIDEO_FILE, gate.TAEH3_IMAGE_FILE], {}],
                "audio_vae": ["VAE"],
            },
        }
    }
}


def _sample_video_graph() -> dict[str, Any]:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "m.safetensors"}},
        "58": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["1", 0], "attention": "sage"}},
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": 1}},
        "9": {"class_type": "BasicGuider", "inputs": {"model": ["58", 0], "conditioning": ["5", 0]}},
        "10": {"class_type": "SamplerCustomAdvanced", "inputs": {"guider": ["9", 0]}},
        "13": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["11", 0], "filename_prefix": "Video/x"}},
    }


def _sample_still_graph() -> dict[str, Any]:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "img.safetensors"}},
        "4": {"class_type": "MiniMaxH3ImageSize", "inputs": {"megapixels": 4.0, "shape": "4:3"}},
        "5": {"class_type": "MiniMaxH3ImageConditioning", "inputs": {"prompt": "desk"}},
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
        "9": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["5", 0]}},
        "12": {"class_type": "MiniMaxH3ImageDownscale", "inputs": {"images": ["11", 0]}},
        "13": {"class_type": "PreviewImage", "inputs": {"images": ["12", 0]}},
    }


async def _run_mock_e2e(tmp: Path) -> None:
    ws_clients: dict[str, web.WebSocketResponse] = {}
    posted_graphs: list[dict[str, Any]] = []
    interrupts: list[dict[str, Any]] = []
    history: dict[str, Any] = {}
    seq = 0

    async def get_oi_node(request: web.Request) -> web.Response:
        node = request.match_info["node"]
        if node in MOCK_OBJECT_INFO:
            return web.json_response({node: MOCK_OBJECT_INFO[node]})
        return web.json_response({})

    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        cid = request.query.get("clientId", "")
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        ws_clients[cid] = ws
        async for _ in ws:
            pass
        ws_clients.pop(cid, None)
        return ws

    async def post_prompt(request: web.Request) -> web.Response:
        nonlocal seq
        seq += 1
        body = await request.json()
        cid = str(body["client_id"])
        pid = f"mock-p-{seq}"
        posted_graphs.append(body["prompt"])
        ws = ws_clients[cid]

        async def _drive() -> None:
            await asyncio.sleep(0.02)
            # Emit steps 1 and 2 via kj_preview_override; if interrupted at step 2, emit execution_interrupted
            await ws.send_json({
                "type": "kj_preview_override",
                "data": {"node_id": "165", "step": 1, "total": 6, "sigma": 0.973, "mime": "video/mp4", "image": SAMPLE_MP4, "w": 672, "h": 384},
            })
            await asyncio.sleep(0.02)
            await ws.send_json({
                "type": "kj_preview_override",
                "data": {"node_id": "165", "step": 2, "total": 6, "sigma": 0.923, "mime": "image/jpeg", "image": SAMPLE_JPG, "w": 672, "h": 384},
            })
            await asyncio.sleep(0.08)
            if any(i.get("prompt_id") == pid for i in interrupts):
                await ws.send_json({"type": "execution_interrupted", "data": {"prompt_id": pid}})
                return
            await ws.send_json({
                "type": "kj_preview_override",
                "data": {"node_id": "165", "step": 6, "total": 6, "sigma": 0.333, "mime": "image/jpeg", "image": SAMPLE_JPG, "w": 672, "h": 384},
            })
            history[pid] = {
                "status": {
                    "status_str": "success",
                    "completed": True,
                    "messages": [
                        ["execution_start", {"timestamp": 1000}],
                        ["execution_success", {"timestamp": 3500}],
                    ],
                },
                "outputs": {"13": {"images": [{"filename": "final.png", "subfolder": "", "type": "temp"}]}},
            }
            await ws.send_json({"type": "execution_success", "data": {"prompt_id": pid}})

        asyncio.create_task(_drive())
        return web.json_response({"prompt_id": pid})

    async def post_interrupt(request: web.Request) -> web.Response:
        interrupts.append(await request.json())
        return web.json_response({})

    async def get_history(request: web.Request) -> web.Response:
        pid = request.match_info["pid"]
        return web.json_response({pid: history[pid]} if pid in history else {})

    async def get_view(request: web.Request) -> web.Response:
        if request.query.get("filename") == "final.png":
            return web.Response(body=base64.b64decode(SAMPLE_PNG), content_type="image/png")
        return web.Response(status=404)

    app = web.Application()
    app.router.add_get("/object_info/{node}", get_oi_node)
    app.router.add_get("/ws", ws_handler)
    app.router.add_post("/prompt", post_prompt)
    app.router.add_post("/interrupt", post_interrupt)
    app.router.add_get("/history/{pid}", get_history)
    app.router.add_get("/view", get_view)

    async with TestServer(app) as srv:
        url = str(srv.make_url("")).rstrip("/")
        # Case A: early interrupt after step 2
        res_stop = await gate.run_preview_gate(
            _sample_video_graph(),
            out_dir=tmp / "early_stop",
            base_url=url,
            seed=777,
            stop_after_step=2,
        )
        assert res_stop["status"] == "interrupted", res_stop
        assert res_stop["injected"] is True, res_stop
        assert res_stop["interrupted_at_step"] == 2, res_stop
        assert res_stop["steps_count"] == 2, res_stop
        assert (tmp / "early_stop" / "steps" / "step_001_s0.9730.mp4").read_bytes() == base64.b64decode(SAMPLE_MP4)
        assert (tmp / "early_stop" / "steps" / "step_002_s0.9230.jpg").read_bytes() == base64.b64decode(SAMPLE_JPG)
        lines = (tmp / "early_stop" / "steps.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2 and json.loads(lines[1])["step"] == 2, lines
        assert posted_graphs[0]["6"]["inputs"]["noise_seed"] == 777
        assert posted_graphs[0]["165"]["class_type"] == gate.PREVIEW_OVERRIDE_NODE
        assert posted_graphs[0]["165"]["inputs"]["tiny_vae"] == gate.TAEH3_VIDEO_FILE

        # Case B: run to completion + collect final outputs
        res_full = await gate.run_preview_gate(
            _sample_still_graph(),
            out_dir=tmp / "full_run",
            base_url=url,
            seed=99,
        )
        assert res_full["status"] == "success", res_full
        assert res_full["injected"] is True, res_full
        assert res_full["steps_count"] == 3, res_full
        assert res_full["duration_s"] == 2.5, res_full
        assert posted_graphs[1]["165"]["inputs"]["tiny_vae"] == gate.TAEH3_IMAGE_FILE
        assert posted_graphs[1]["165"]["inputs"]["preview_frames"] == 1
        assert (tmp / "full_run" / "outputs" / "final.png").read_bytes() == base64.b64decode(SAMPLE_PNG)


def test_injection_logic() -> str:
    vid_off, inj_off = gate.inject_preview_override(_sample_video_graph(), {})
    assert not inj_off and "165" not in vid_off

    vid_on, inj_on = gate.inject_preview_override(_sample_video_graph(), MOCK_OBJECT_INFO, preview_frames=6)
    assert inj_on and vid_on["165"]["inputs"]["tiny_vae"] == gate.TAEH3_VIDEO_FILE
    assert vid_on["165"]["inputs"]["preview_frames"] == 6
    assert "vae" not in vid_on["165"]["inputs"] and "audio_vae" not in vid_on["165"]["inputs"]
    assert vid_on["9"]["inputs"]["model"] == ["165", 0]

    still_on, still_inj = gate.inject_preview_override(_sample_still_graph(), MOCK_OBJECT_INFO)
    assert still_inj and still_on["165"]["inputs"]["tiny_vae"] == gate.TAEH3_IMAGE_FILE
    assert still_on["165"]["inputs"]["preview_frames"] == 1
    assert still_on["9"]["inputs"]["model"] == ["165", 0]

    v3_info = {
        gate.PREVIEW_OVERRIDE_NODE: {
            "input": {
                "optional": {
                    "tiny_vae": ["COMBO", {"options": ["none", gate.TAEH3_VIDEO_FILE, gate.TAEH3_IMAGE_FILE]}],
                }
            }
        }
    }
    v3_vid, v3_inj = gate.inject_preview_override(_sample_video_graph(), v3_info)
    assert v3_inj and v3_vid["165"]["inputs"]["tiny_vae"] == gate.TAEH3_VIDEO_FILE
    v3_still, v3_sinj = gate.inject_preview_override(_sample_still_graph(), v3_info)
    assert v3_sinj and v3_still["165"]["inputs"]["tiny_vae"] == gate.TAEH3_IMAGE_FILE
    return "V1 + V3 COMBO video/still injection and absent-/object_info bypass verified"


def test_mock_ws_gate() -> str:
    with tempfile.TemporaryDirectory(prefix="h3_preview_gate_") as td:
        asyncio.run(_run_mock_e2e(Path(td)))
    return "steps.jsonl, step MP4/JPEG capture, --stop-after-step /interrupt, and completion verified"


def main() -> int:
    print("=== check_preview_gate.py (offline) ===")
    case("inject_preview_override contract", test_injection_logic)
    case("mock WS preview gate + early interrupt + full run", test_mock_ws_gate)
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
