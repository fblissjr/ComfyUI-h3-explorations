#!/usr/bin/env python3
"""Run an API workflow through live ComfyUI WebSocket events with a tiny-VAE preview gate.

Injects KJNodes `ModelPreviewOverrideKJ` (`taeh3.safetensors` for video, `taeh3_image.safetensors`
for single-frame stills, with `vae` and `audio_vae` unwired to save VRAM) in memory when offered by
the connected server, captures every step's `"kj_preview_override"` frame (`steps/*.jpg|webp|mp4`)
and `steps.jsonl`, optionally halts early at `--stop-after-step K` via `POST /interrupt`, and
otherwise collects final `/history` outputs into `--out-dir`.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import json
import sys
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

import aiohttp

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from _lib import comfy_client  # noqa: E402

PREVIEW_OVERRIDE_NODE = "ModelPreviewOverrideKJ"
GUIDER_NODES = ("BasicGuider", "MiniMaxH3EditGuider")
TAEH3_VIDEO_FILE = "taeh3.safetensors"
TAEH3_IMAGE_FILE = "taeh3_image.safetensors"


def _combo_choices(spec: dict[str, Any], input_name: str) -> list[str]:
    inp = spec.get("input") or {}
    for section in ("optional", "required"):
        entry = (inp.get(section) or {}).get(input_name)
        if not isinstance(entry, (list, tuple)) or not entry:
            continue
        if isinstance(entry[0], (list, tuple)):
            return [str(v) for v in entry[0]]
        if len(entry) > 1 and isinstance(entry[1], dict) and isinstance(entry[1].get("options"), list):
            return [str(v) for v in entry[1]["options"]]
    return []


def _is_still_graph(graph: dict[str, Any]) -> bool:
    """Return True when the API graph renders one frame (`SINGLE_FRAME`) rather than multi-frame video."""
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type")
        if ct == "VHS_VideoCombine":
            return False
        if ct in ("MiniMaxH3PH3ImageConditioning", "MiniMaxH3ImageConditioning", "MiniMaxH3ImageDownscale"):
            return True
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        ins = node.get("inputs") or {}
        if ins.get("single_frame") is True:
            return True
        if node.get("class_type") in ("MiniMaxH3Resolution", "MiniMaxH3Conditioning"):
            length = ins.get("length")
            if isinstance(length, int) and length == 1:
                return True
    return False


def _free_node_id(graph: dict[str, Any], preferred: str = "165") -> str:
    if preferred not in graph:
        return preferred
    idx = int(preferred) if preferred.isdigit() else 165
    while str(idx) in graph:
        idx += 1
    return str(idx)


def inject_preview_override(
    graph: dict[str, Any],
    object_info: dict[str, Any],
    *,
    preview_frames: int | None = None,
    max_resolution: int = 672,
    jpeg_quality: int = 80,
    preview_fps: int = 12,
) -> tuple[dict[str, Any], bool]:
    """Return `(patched_graph, injected_now)` wiring `ModelPreviewOverrideKJ` if offered live."""
    patched = copy.deepcopy(graph)
    is_still = _is_still_graph(patched)
    existing = [
        n for n in patched.values()
        if isinstance(n, dict) and n.get("class_type") == PREVIEW_OVERRIDE_NODE
    ]
    if existing:
        if preview_frames is not None:
            for node in existing:
                node.setdefault("inputs", {})["preview_frames"] = int(preview_frames)
        return patched, False

    kj_spec = object_info.get(PREVIEW_OVERRIDE_NODE)
    if not isinstance(kj_spec, dict) or not kj_spec:
        return patched, False
    choices = _combo_choices(kj_spec, "tiny_vae")
    want_tiny = TAEH3_IMAGE_FILE if is_still else TAEH3_VIDEO_FILE
    if want_tiny not in choices:
        return patched, False

    guiders = [
        (nid, node)
        for nid, node in patched.items()
        if isinstance(node, dict)
        and node.get("class_type") in GUIDER_NODES
        and isinstance((node.get("inputs") or {}).get("model"), list)
    ]
    if not guiders:
        return patched, False

    frames_val = int(preview_frames) if preview_frames is not None else (1 if is_still else 8)
    override_id = _free_node_id(patched, "165")
    model_src = guiders[0][1]["inputs"]["model"]
    patched[override_id] = {
        "class_type": PREVIEW_OVERRIDE_NODE,
        "inputs": {
            "model": model_src,
            "tiny_vae": want_tiny,
            "max_resolution": int(max_resolution),
            "jpeg_quality": int(jpeg_quality),
            "suppress_default_preview": True,
            "preview_frames": frames_val,
            "preview_fps": int(preview_fps),
        },
    }
    for _gid, gnode in guiders:
        if gnode["inputs"].get("model") == model_src:
            gnode["inputs"]["model"] = [override_id, 0]
    return patched, True


def override_graph_seed_and_prefix(
    graph: dict[str, Any],
    *,
    seed: int | None = None,
    filename_prefix: str | None = None,
) -> dict[str, Any]:
    """Return a copy of `graph` with optional `RandomNoise.noise_seed` and save prefix overrides."""
    out = copy.deepcopy(graph)
    if seed is not None:
        for node in out.values():
            if isinstance(node, dict) and node.get("class_type") == "RandomNoise":
                node.setdefault("inputs", {})["noise_seed"] = int(seed)
    if filename_prefix:
        for node in out.values():
            if not isinstance(node, dict):
                continue
            ins = node.get("inputs") or {}
            if node.get("class_type") in ("SaveImage", "VHS_VideoCombine") and "filename_prefix" in ins:
                ins["filename_prefix"] = str(filename_prefix)
    return out


def _mime_extension(mime: str) -> str:
    m = (mime or "").lower().strip()
    if m == "video/mp4":
        return "mp4"
    if m == "image/webp":
        return "webp"
    if m == "image/png":
        return "png"
    return "jpg"


def _ws_url(base_url: str, client_id: str) -> str:
    parsed = urllib.parse.urlsplit(base_url.rstrip("/"))
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{parsed.netloc}{parsed.path}/ws?clientId={urllib.parse.quote(client_id)}"


async def _fetch_kj_object_info(session: aiohttp.ClientSession, base_url: str) -> dict[str, Any]:
    root = base_url.rstrip("/")
    async with session.get(f"{root}/object_info/{PREVIEW_OVERRIDE_NODE}") as resp:
        if resp.status == 200:
            data = await resp.json()
            if isinstance(data, dict) and PREVIEW_OVERRIDE_NODE in data:
                return data
    async with session.get(f"{root}/object_info") as resp:
        if resp.status == 200:
            data = await resp.json()
            if isinstance(data, dict):
                return data
    return {}


async def _post_interrupt(session: aiohttp.ClientSession, base_url: str, prompt_id: str) -> None:
    root = base_url.rstrip("/")
    async with session.post(f"{root}/interrupt", json={"prompt_id": prompt_id}) as resp:
        if resp.status >= 400:
            await session.post(f"{root}/interrupt", json={})


async def _fetch_history_entry(
    session: aiohttp.ClientSession,
    base_url: str,
    prompt_id: str,
    *,
    attempts: int = 12,
    delay_s: float = 0.25,
) -> dict[str, Any] | None:
    url = f"{base_url.rstrip('/')}/history/{urllib.parse.quote(prompt_id, safe='')}"
    for i in range(attempts):
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.json()
                entry = data.get(prompt_id) if isinstance(data, dict) else None
                if isinstance(entry, dict) and entry.get("outputs") is not None:
                    return entry
        if i + 1 < attempts:
            await asyncio.sleep(delay_s)
    return None


async def _download_outputs(
    session: aiohttp.ClientSession,
    base_url: str,
    history_entry: dict[str, Any],
    outputs_dir: Path,
) -> list[dict[str, Any]]:
    items = comfy_client.collect_saved_files(history_entry)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    root = base_url.rstrip("/")
    downloaded: list[dict[str, Any]] = []
    for item in items:
        fname = item.get("filename")
        if not fname:
            continue
        query = urllib.parse.urlencode({
            "filename": fname,
            "subfolder": item.get("subfolder") or "",
            "type": item.get("type") or "output",
        })
        dest = outputs_dir / Path(str(fname)).name
        if dest.exists():
            dest = outputs_dir / f"{item.get('node_id', 'n')}_{dest.name}"
        async with session.get(f"{root}/view?{query}") as resp:
            if resp.status == 200:
                dest.write_bytes(await resp.read())
                downloaded.append({**item, "local_path": str(dest)})
            else:
                downloaded.append({**item, "local_path": None, "view_status": resp.status})
    return downloaded


async def run_preview_gate(
    graph: dict[str, Any],
    *,
    out_dir: Path,
    base_url: str = comfy_client.DEFAULT_COMFY_URL,
    seed: int | None = None,
    filename_prefix: str | None = None,
    stop_after_step: int | None = None,
    preview_frames: int | None = None,
    max_resolution: int = 672,
    timeout: float = 1800.0,
) -> dict[str, Any]:
    """Queue `graph`, capture WS `kj_preview_override` step frames, and collect outputs or interrupt."""
    out_dir = Path(out_dir)
    steps_dir = out_dir / "steps"
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_dir.mkdir(parents=True, exist_ok=True)
    steps_log = out_dir / "steps.jsonl"

    client_id = uuid.uuid4().hex
    root = base_url.rstrip("/")
    prepared = override_graph_seed_and_prefix(graph, seed=seed, filename_prefix=filename_prefix)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        oi = await _fetch_kj_object_info(session, root)
        prepared, injected = inject_preview_override(
            prepared, oi, preview_frames=preview_frames, max_resolution=max_resolution
        )
        steps_recorded: list[dict[str, Any]] = []
        status = "unknown"
        interrupted_at_step: int | None = None
        interrupt_sent = False
        prompt_id = ""

        async with session.ws_connect(_ws_url(root, client_id)) as ws:
            async with session.post(f"{root}/prompt", json={"prompt": prepared, "client_id": client_id}) as resp:
                resp_text = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"ComfyUI /prompt failed (HTTP {resp.status}): {resp_text}")
                resp_json = json.loads(resp_text)
                if resp_json.get("node_errors"):
                    raise RuntimeError(f"ComfyUI /prompt node_errors: {resp_json['node_errors']}")
                prompt_id = str(resp_json.get("prompt_id") or "")
                if not prompt_id:
                    raise RuntimeError(f"ComfyUI /prompt missing prompt_id: {resp_json}")

            loop = asyncio.get_running_loop()
            deadline = loop.time() + float(timeout)
            with steps_log.open("w", encoding="utf-8") as log_fp:
                while True:
                    rem = deadline - loop.time()
                    if rem <= 0:
                        raise TimeoutError(f"Timed out after {timeout:.1f}s waiting for {prompt_id}")
                    msg = await ws.receive(timeout=rem)
                    if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    try:
                        pkt = json.loads(msg.data)
                    except ValueError:
                        continue
                    mtype = pkt.get("type")
                    data = pkt.get("data") or {}

                    if mtype == "kj_preview_override" and data.get("image"):
                        step_idx = int(data["step"]) if data.get("step") is not None else (len(steps_recorded) + 1)
                        sigma = float(data["sigma"]) if data.get("sigma") is not None else None
                        mime = str(data.get("mime") or "image/jpeg")
                        ext = _mime_extension(mime)
                        sig_part = f"_s{sigma:.4f}" if sigma is not None else ""
                        fname = f"step_{step_idx:03d}{sig_part}.{ext}"
                        (steps_dir / fname).write_bytes(base64.b64decode(data["image"]))
                        rec = {
                            "step": step_idx,
                            "total": data.get("total"),
                            "sigma": sigma,
                            "delta": data.get("delta"),
                            "step_ms": data.get("step_ms"),
                            "avg_step_ms": data.get("avg_step_ms"),
                            "mime": mime,
                            "w": data.get("w"),
                            "h": data.get("h"),
                            "file": f"steps/{fname}",
                        }
                        steps_recorded.append(rec)
                        log_fp.write(json.dumps(rec) + "\n")
                        log_fp.flush()
                        if (
                            stop_after_step is not None
                            and not interrupt_sent
                            and step_idx >= int(stop_after_step)
                        ):
                            interrupt_sent = True
                            interrupted_at_step = step_idx
                            await _post_interrupt(session, root, prompt_id)

                    elif mtype == "execution_error" and data.get("prompt_id") == prompt_id:
                        raise RuntimeError(
                            f"prompt {prompt_id} failed on node {data.get('node_id')} "
                            f"({data.get('node_type')}): {data.get('exception_message')}"
                        )

                    elif mtype == "execution_interrupted" and data.get("prompt_id") == prompt_id:
                        status = "interrupted"
                        break

                    elif mtype == "execution_success" and data.get("prompt_id") == prompt_id:
                        status = "success"
                        break

                    elif mtype == "executing" and data.get("prompt_id") == prompt_id and data.get("node") is None:
                        status = "interrupted" if interrupt_sent else "success"
                        break

        outputs: list[dict[str, Any]] = []
        duration_s: float | None = None
        if status != "interrupted":
            entry = await _fetch_history_entry(session, root, prompt_id)
            if entry is not None:
                duration_s = comfy_client.execution_duration_s(entry)
                outputs = await _download_outputs(session, root, entry, out_dir / "outputs")

        summary = {
            "prompt_id": prompt_id,
            "status": status,
            "injected": injected,
            "steps_count": len(steps_recorded),
            "interrupted_at_step": interrupted_at_step,
            "duration_s": duration_s,
            "steps": steps_recorded,
            "outputs": outputs,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run a ComfyUI API graph with live tiny-VAE preview gating.")
    ap.add_argument("--graph", required=True, type=Path, help="Path to *_api.json workflow file")
    ap.add_argument("--out-dir", required=True, type=Path, help="Directory for steps/, steps.jsonl, outputs/, summary.json")
    ap.add_argument("--seed", type=int, default=None, help="Override RandomNoise.noise_seed")
    ap.add_argument("--filename-prefix", default=None, help="Override SaveImage/VHS_VideoCombine filename_prefix")
    ap.add_argument("--stop-after-step", type=int, default=None, help="Interrupt render once step K preview arrives")
    ap.add_argument("--preview-frames", type=int, default=None, help="Override ModelPreviewOverrideKJ.preview_frames")
    ap.add_argument("--max-resolution", type=int, default=672, help="Max preview dimension (default: 672)")
    ap.add_argument("--url", default=comfy_client.DEFAULT_COMFY_URL, help="ComfyUI server URL")
    ap.add_argument("--timeout", type=float, default=1800.0, help="Run timeout in seconds")
    args = ap.parse_args(argv)

    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    summary = asyncio.run(
        run_preview_gate(
            graph,
            out_dir=args.out_dir,
            base_url=args.url,
            seed=args.seed,
            filename_prefix=args.filename_prefix,
            stop_after_step=args.stop_after_step,
            preview_frames=args.preview_frames,
            max_resolution=args.max_resolution,
            timeout=args.timeout,
        )
    )
    print(json.dumps({
        "prompt_id": summary["prompt_id"],
        "status": summary["status"],
        "injected": summary["injected"],
        "steps_count": summary["steps_count"],
        "interrupted_at_step": summary["interrupted_at_step"],
        "duration_s": summary["duration_s"],
        "outputs": [o.get("local_path") or o.get("filename") for o in summary["outputs"]],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
