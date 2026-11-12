#!/usr/bin/env python3
"""MiniMax-H3-Image: the release-to-ComfyUI layout, the size rule and the shipped graphs' chain.

No model file and no card. What each case defends, i.e. what breaks if it is deleted:

- `qkv order is core's split` -- `bench/h3_image/release.py::comfy_qkv_order` turns the release's
  per-head `[q|k|v]` rows into the `[q_all|k_all|v_all]` rows core splits. Checked against core's
  own split on a random projection: q, k and v taken from the reordered weight the way
  `Attention.forward` takes them equal the per-head rows of the original. Control: the weight left
  in release order must NOT pass.
- `stacked heads read back whole` -- `stack_heads` writes head 0 whole and later heads as offsets;
  read through core's `_pdd_head` over one grid interval, each head comes back as it was. Control:
  whole heads stacked without the offset must NOT read back.
- `the writer's file opens and holds its tensors` -- `write_safetensors` against the safetensors
  library, bf16 included, with a planned dtype that disagrees refused.
- `sizes are the model card's` -- `h3_config.h3_image_size` at 16 megapixels gives the card's size
  table (transcribed below by hand from the release's README), for each of its ratios.
- `the node's shapes are the vendor's ratios` -- `H3_IMAGE_ASPECTS` is the `aspect_ratio` enum of
  the vendor's `request_params.json`, read from the vendored file.
- `a picture gives its own shape` and `no picture is refused` -- `resolve_size` on
  'same as the picture'.
- `expander replies parse plain prose, trailing aspect_ratio, thinking blocks and vendor JSON` --
  `parse_expander_reply` on the release's worked captions (`revised_prompt.{t2i,edit}.txt`), an
  arbitrary edit ratio composed with `resolve_size`, `'same as the picture'` / `'adaptive'`,
  `<think>...</think>` and code fences, and vendor `expanded_caption` JSON. Controls: unclosed
  `<think>`, bad ratio strings, broken JSON and empty captions are refused.
- `the shipped graphs` -- each `workflows/image/h3_image*_api.json` and its probes hands the
  sampler the grid in `H3_IMAGE_MANUAL_SIGMAS` through `ManualSigmas`, has no negative prompt or
  scheduler node, and an edit wires its picture into both the conditioning node and the size node
  with nothing between the loader and either.

`--release <MiniMax-H3-Image folder>` adds: the grid constant equals the release's
`pdd_config.json`. Without it that comparison is reported as skipped, not passed.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_h3_image.py [--release <folder>]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "workflows"))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

import h3_config  # noqa: E402
from h3_image import release as rel  # noqa: E402

# The model card's "Sizes" table, typed from the release's README.md (width, height).
CARD_SIZES = {
    "1:1": (4096, 4096), "4:3": (4736, 3552), "3:4": (3552, 4736), "3:2": (5024, 3360),
    "2:3": (3360, 5024), "16:9": (5472, 3072), "9:16": (3072, 5472), "21:9": (6272, 2688),
}

fails = 0


def report(ok: bool, name: str, detail: str = "") -> None:
    global fails
    fails += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail and not ok else ""))


def core_split(weight: torch.Tensor, x: torch.Tensor, heads: int, head_dim: int):
    """q, k, v as core takes them from a fused projection: three equal runs, each [heads, head_dim]."""
    q, k, v = (x @ weight.T).chunk(3, dim=-1)
    return tuple(t.view(heads, head_dim) for t in (q, k, v))


def qkv_order() -> None:
    heads, head_dim, hidden = 4, 6, 10
    g = torch.Generator().manual_seed(0)
    release_w = torch.randn(heads * 3 * head_dim, hidden, generator=g)
    x = torch.randn(hidden, generator=g)
    per_head = (x @ release_w.T).view(heads, 3, head_dim)          # the release's own layout
    want = tuple(per_head[:, i] for i in range(3))
    got = core_split(rel.comfy_qkv_order(release_w, heads, head_dim), x, heads, head_dim)
    report(all(torch.equal(a, b) for a, b in zip(got, want)), "qkv order is core's split")
    left = core_split(release_w, x, heads, head_dim)
    report(not all(torch.equal(a, b) for a, b in zip(left, want)),
           "control: a weight left in release order does not pass")


def read_head(bank: torch.Tensor, steps: int, k: int) -> torch.Tensor:
    """Head `k` as core's `_pdd_head` composes it over the single interval k -> k+1."""
    rows = bank.reshape(steps, -1, bank.shape[-1])
    return rows[0] if k == 0 else rows[0] + rows[k]


def stacked_heads() -> None:
    steps, out, hidden = 6, 5, 7
    heads = torch.randn(steps, out, hidden, generator=torch.Generator().manual_seed(1))
    bank = rel.stack_heads(heads)
    report(bank.shape == (steps * out, hidden) and
           all(torch.allclose(read_head(bank, steps, k), heads[k], atol=1e-6) for k in range(steps)),
           "stacked heads read back whole")
    plain = heads.reshape(steps * out, hidden)
    report(not all(torch.allclose(read_head(plain, steps, k), heads[k], atol=1e-6) for k in range(steps)),
           "control: whole heads stacked without the offset do not read back")


def writer() -> None:
    tensors = {"a.weight": torch.randn(3, 4).to(torch.bfloat16), "b.bias": torch.arange(5, dtype=torch.float32),
               "c": torch.tensor([[1, -2]], dtype=torch.int8)}
    entries = [(k, rel._ST_DTYPE[v.dtype], tuple(v.shape)) for k, v in tensors.items()]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.safetensors"
        size = rel.write_safetensors(path, entries, tensors.__getitem__, {"config": "{}"})
        with safe_open(str(path), "pt") as f:
            same = all(torch.equal(f.get_tensor(k), v) and f.get_tensor(k).dtype == v.dtype for k, v in tensors.items())
            meta = f.metadata() == {"config": "{}"}
        report(same and meta and size == path.stat().st_size, "the writer's file opens and holds its tensors")
        wrong = [("a.weight", "F32", (3, 4))]
        try:
            rel.write_safetensors(Path(tmp) / "y.safetensors", wrong, tensors.__getitem__, {})
            report(False, "control: a planned dtype that disagrees is refused")
        except ValueError:
            report(True, "control: a planned dtype that disagrees is refused")


def sizes() -> None:
    got = {r: h3_config.h3_image_size(16.0, *(int(v) for v in r.split(":"))) for r in CARD_SIZES}
    report(got == CARD_SIZES, "sizes are the model card's", f"{got}")
    params = json.loads((REPO / "vendor_guides" / "h3_image" / "prompt" / "t2i" / "request_params.json").read_text())
    enum = params["response_format"]["json_schema"]["schema"]["properties"]["aspect_ratio"]["enum"]
    report(list(h3_config.H3_IMAGE_ASPECTS) == enum, "the node's shapes are the vendor's ratios",
           f"{h3_config.H3_IMAGE_ASPECTS} against {enum}")
    sys.path.insert(0, str(REPO))
    try:
        from h3_image_nodes import (
            MiniMaxH3ImageDownscale,
            MiniMaxH3ImageSize,
            resolve_canvas_and_target_size,
            resolve_size,
        )
    except Exception as exc:  # comfy_api not importable outside ComfyUI's tree
        report(False, "the size node imports", repr(exc))
        return
    report(resolve_size(16.0, h3_config.H3_IMAGE_ASPECT_OF_PICTURE, (1500, 1000)) == CARD_SIZES["3:2"],
           "a picture gives its own shape")
    try:
        resolve_size(16.0, h3_config.H3_IMAGE_ASPECT_OF_PICTURE, None)
        report(False, "no picture is refused")
    except ValueError:
        report(True, "no picture is refused")
    override_size = h3_config.h3_image_size(16.0, 5, 3)
    override_out = tuple(MiniMaxH3ImageSize.execute(16.0, "1:1", picture=None, shape_from_expander="5:3"))
    fallback_out = tuple(MiniMaxH3ImageSize.execute(16.0, "4:3", picture=None, shape_from_expander=""))
    report(
        override_out == (*override_size, *override_size)
        and fallback_out == (*CARD_SIZES["4:3"], *CARD_SIZES["4:3"]),
        "shape_from_expander overrides shape when set and falls back when empty",
    )
    report(
        resolve_canvas_and_target_size(1.0, "1:1", None, enforce_4mp_floor=True) == (2048, 2048, 1024, 1024)
        and tuple(MiniMaxH3ImageSize.execute(1.0, "1:1")) == (2048, 2048, 1024, 1024)
        and tuple(MiniMaxH3ImageSize.execute(1.0, "1:1", enforce_4mp_floor=False)) == (1024, 1024, 1024, 1024),
        "sub-4MP requests lift DiT canvas to the vendor's 4.0-MP floor while keeping target_width/target_height",
    )
    gen = torch.Generator().manual_seed(42)
    img_in = torch.rand((1, 128, 128, 3), generator=gen, dtype=torch.float32)
    same_out = MiniMaxH3ImageDownscale.execute(img_in, 128, 128)[0]
    disabled_out = MiniMaxH3ImageDownscale.execute(img_in, 64, 64, enabled=False)[0]
    down_out = MiniMaxH3ImageDownscale.execute(img_in, 64, 64)[0]
    refused_zero = False
    try:
        MiniMaxH3ImageDownscale.execute(img_in, 0, 0)
    except ValueError:
        refused_zero = True
    from PIL import Image
    import numpy as np
    u8 = (img_in[0] * 255.0).clamp(0.0, 255.0).to(torch.uint8).numpy()
    pil_ref = torch.from_numpy(
        np.asarray(Image.fromarray(u8).resize((64, 64), resample=Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    ).unsqueeze(0)
    report(
        same_out is img_in and disabled_out is img_in and refused_zero and torch.equal(down_out, pil_ref),
        "MiniMaxH3ImageDownscale passes matching/disabled sizes untouched, refuses <=0, and matches 8-bit PIL Lanczos",
    )


def expander_replies() -> None:
    sys.path.insert(0, str(REPO))
    try:
        from h3_image_nodes import parse_expander_reply, resolve_size
    except Exception as exc:
        report(False, "parse_expander_reply imports", repr(exc))
        return
    examples = REPO / "vendor_guides" / "h3_image" / "examples"
    t2i = (examples / "revised_prompt.t2i.txt").read_text(encoding="utf-8").strip()
    edit = (examples / "revised_prompt.edit.txt").read_text(encoding="utf-8").strip()

    report(parse_expander_reply(t2i) == (t2i, None), "plain expander prose returns (prompt, None)")

    got_prompt, got_ratio = parse_expander_reply(f"<think>\nlock 4:3 first\n</think>\n\n{t2i}\n\naspect_ratio: 4:3\n")
    report(got_prompt == t2i and got_ratio == "4:3"
           and resolve_size(16.0, got_ratio, None) == CARD_SIZES["4:3"],
           "trailing aspect_ratio strips <think> and resolves to the card's size")

    arb_prompt, arb_ratio = parse_expander_reply(f"{edit}\n\naspect_ratio: 5:3")
    report(arb_prompt == edit and arb_ratio == "5:3"
           and resolve_size(16.0, arb_ratio, None) == h3_config.h3_image_size(16.0, 5, 3),
           "an arbitrary edit W:H ratio parses and resolves through resolve_size")

    same_a = parse_expander_reply(f"prefilled reasoning\n</think>\n{edit}\n\naspect_ratio: same as the picture")
    same_b = parse_expander_reply(f"```text\n{edit}\n\ntarget_aspect_ratio: adaptive\n```")
    report(same_a == (edit, h3_config.H3_IMAGE_ASPECT_OF_PICTURE)
           and same_b == (edit, h3_config.H3_IMAGE_ASPECT_OF_PICTURE),
           "edit 'same as the picture' and 'adaptive' normalize to H3_IMAGE_ASPECT_OF_PICTURE")

    vendor_t2i = json.dumps({"expanded_caption": t2i, "aspect_ratio": "4:3"})
    vendor_edit = "```json\n" + json.dumps({"expanded_caption": edit, "target_aspect_ratio": "1:1"}) + "\n```"
    report(parse_expander_reply(vendor_t2i) == (t2i, "4:3")
           and parse_expander_reply(vendor_edit) == (edit, "1:1"),
           "vendor expanded_caption JSON replies parse for both T2I and edit")

    bad_inputs = (
        ("unclosed <think>", f"<think>cut off mid-thought\n{t2i}"),
        ("zero in ratio", f"{t2i}\n\naspect_ratio: 0:3"),
        ("non-numeric ratio", f"{t2i}\n\naspect_ratio: landscape"),
        ("wrong JSON schema", '{"prompt": "wrong key"}'),
        ("truncated JSON", '{"expanded_caption": "unterminated'),
        ("ratio line with no caption", "aspect_ratio: 4:3"),
        ("empty post-think reply", "<think>done</think>\n\n"),
    )
    refused_all = True
    for label, raw in bad_inputs:
        try:
            parse_expander_reply(raw)
            refused_all = False
            report(False, f"control: malformed expander reply ({label}) is refused")
        except ValueError:
            pass
    if refused_all:
        report(True, f"control: all {len(bad_inputs)} malformed expander replies are refused")


def heylook_expander() -> None:
    import base64
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import io as _pyio
    import threading
    from PIL import Image
    import httpx

    sys.path.insert(0, str(REPO))
    try:
        from h3_image_nodes import (
            MiniMaxH3ImagePromptExpander,
            build_expander_catalog,
            prepare_heylook_image,
            prepare_heylook_tensor,
            stream_heylook_message,
        )
    except Exception as exc:
        report(False, "MiniMaxH3ImagePromptExpander imports", repr(exc))
        return

    examples = REPO / "vendor_guides" / "h3_image" / "examples"
    t2i_caption = (examples / "revised_prompt.t2i.txt").read_text(encoding="utf-8").strip()
    edit_caption = (examples / "revised_prompt.edit.txt").read_text(encoding="utf-8").strip()

    # 1. Client-side image resize: PNG stays PNG across exif_transpose, EXIF orientation
    #    rotates upright and caps longest edge at 2048, and ComfyUI tensors downscale cleanly.
    png_buf = _pyio.BytesIO()
    Image.new("RGB", (2600, 1300), (20, 40, 80)).save(png_buf, format="PNG")
    png_b64, png_mime = prepare_heylook_image(png_buf.getvalue())
    with Image.open(_pyio.BytesIO(base64.b64decode(png_b64))) as dec_png:
        png_ok = png_mime == "image/png" and dec_png.format == "PNG" and dec_png.size == (2048, 1024)

    exif = Image.Exif()
    exif[0x0112] = 6  # 90 deg CW: 3000x1500 stored -> 1500x3000 upright -> 1024x2048 capped
    jpg_buf = _pyio.BytesIO()
    Image.new("RGB", (3000, 1500), (180, 90, 40)).save(jpg_buf, format="JPEG", exif=exif.tobytes())
    jpg_b64, jpg_mime = prepare_heylook_image(jpg_buf.getvalue())
    with Image.open(_pyio.BytesIO(base64.b64decode(jpg_b64))) as dec_jpg:
        jpg_ok = jpg_mime == "image/jpeg" and dec_jpg.size == (1024, 2048)

    tensor_rgb = torch.zeros((1, 2000, 3000, 3), dtype=torch.float32)
    ten_b64, ten_mime = prepare_heylook_tensor(tensor_rgb, index=1)
    with Image.open(_pyio.BytesIO(base64.b64decode(ten_b64))) as dec_ten:
        ten_ok = ten_mime == "image/jpeg" and dec_ten.size == (2048, 1365)

    tensor_rgba = torch.ones((1, 64, 64, 4), dtype=torch.float32)
    tensor_rgba[:, :, :, 3] = 0.5
    _, alpha_mime = prepare_heylook_tensor(tensor_rgba, index=2)
    report(png_ok and jpg_ok and ten_ok and alpha_mime == "image/png",
           "client-side heylook image resize honours PNG, EXIF, 2048 cap and alpha")

    # 2. Live loopback heylook Messages SSE mock server: runtime discovery, 503 backoff,
    #    fresh X-Request-ID, block separation, error events, and DELETE cancellation.
    state: dict = {
        "seen_request_ids": [],
        "cancelled_ids": [],
        "bodies": [],
        "overload_next": True,
        "mode": "ok",
    }

    class _MockHeylook(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _json(self, code: int, obj: dict, headers: dict | None = None) -> None:
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if state["mode"] == "host403":
                self._json(403, {"detail": "Host not allowed"})
                return
            if self.path == "/v1/models":
                self._json(200, {
                    "object": "list",
                    "data": [
                        {
                            "id": "embed-only-local",
                            "modalities": ["text", "vision"],
                            "capabilities": ["embedding"],
                            "engine": {},
                        },
                        {
                            "id": "text-chat-cold",
                            "modalities": ["text"],
                            "capabilities": ["chat", "thinking"],
                            "engine": {
                                "thinking": {"switch": "enable_thinking", "depth": None, "budget": None},
                                "decoding": {"request_fields": {"value": ["seed", "temperature"]}},
                            },
                        },
                        {
                            "id": "text-chat-resident",
                            "modalities": ["text"],
                            "capabilities": ["chat", "thinking"],
                            "engine": {
                                "thinking": {"switch": "enable_thinking", "depth": None, "budget": None},
                                "decoding": {"request_fields": {"value": None}},
                            },
                        },
                        {
                            "id": "vlm-chat-local",
                            "modalities": ["text", "vision"],
                            "capabilities": ["chat", "vision", "thinking"],
                            "engine": {
                                "thinking": {"switch": "enable_thinking", "depth": None, "budget": None},
                                "decoding": {"request_fields": {"value": None}},
                            },
                        },
                    ],
                })
            elif self.path == "/v1/system/metrics":
                self._json(200, {"models": {"text-chat-resident": {"active": 0}}})
            elif self.path == "/v1/presets":
                self._json(200, {
                    "presets": [{
                        "id": "crisp",
                        "name": "crisp_preset",
                        "system_prompt": "PRESET SYSTEM PROMPT",
                        "params": {"temperature": 0.8, "enable_thinking": True, "reasoning_effort": "unoffered"},
                    }],
                })
            else:
                self._json(404, {"detail": "not found"})

        def do_DELETE(self):
            if self.path.startswith("/v1/requests/"):
                rid = self.path.rsplit("/", 1)[-1]
                state["cancelled_ids"].append(rid)
                self._json(200, {"cancelled": 1, "request_id": rid})
            else:
                self._json(404, {"detail": "not found"})

        def do_POST(self):
            if self.path != "/v1/messages":
                self._json(404, {"detail": "not found"})
                return
            rid = self.headers.get("X-Request-ID") or ""
            state["seen_request_ids"].append(rid)
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            state["bodies"].append(body)

            if state["overload_next"]:
                state["overload_next"] = False
                self._json(
                    503,
                    {"error": {"code": "model_overloaded", "message": "queue full"}},
                    headers={"Retry-After": "1", "X-Request-ID": rid},
                )
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("X-Request-ID", rid)
            self.end_headers()

            def emit(ev: str, payload: dict) -> bool:
                chunk = f"event: {ev}\ndata: {json.dumps(payload)}\n\n".encode("utf-8")
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    return True
                except (BrokenPipeError, ConnectionResetError):
                    return False

            if not emit("message_start", {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 0, "output_tokens": 0}}}):
                return
            if not emit("ping", {"type": "ping"}):
                return
            if state["mode"] == "inband_error":
                emit("error", {"type": "error", "error": {"type": "invalid_request_error", "message": "image rejected"}})
                return
            if state["mode"] == "truncated_stream":
                emit("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "partial"}})
                return

            is_vlm = body.get("model") == "vlm-chat-local"
            reply_text = (
                f"{edit_caption}\n\naspect_ratio: same as the picture"
                if is_vlm
                else f"{t2i_caption}\n\naspect_ratio: 4:3"
            )
            for ev, obj in (
                ("heylook_progress", {"type": "heylook_progress", "prefill": {"processed": 128, "total": 128}}),
                ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
                ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "plan composition"}}),
                ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                ("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}),
                ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": reply_text}}),
                ("content_block_stop", {"type": "content_block_stop", "index": 1}),
                ("message_delta", {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"input_tokens": 320, "cache_read_input_tokens": 256, "output_tokens": 180},
                }),
                ("message_stop", {
                    "type": "message_stop",
                    "performance": {"generation_duration_ms": 900.0, "request_duration_ms": 2500.0},
                }),
            ):
                if not emit(ev, obj):
                    return

    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockHeylook)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        # T2I: survives initial 503, uses fresh X-Request-ID per attempt, picks resident chat model,
        # separates thinking_delta from text_delta, expands preset safely, and computes throughput from generation_duration_ms (180 / 0.9s = 200.0 tps).
        t2i_out = tuple(MiniMaxH3ImagePromptExpander.execute(
            prompt="Overhead rustic farm table feast.",
            enabled=True,
            base_url=base,
            mode="auto",
            model="",
            thinking="on",
            seed=42,
            preset="crisp",
        ))
        ids = state["seen_request_ids"]
        last_body = state["bodies"][-1]
        report(
            t2i_out[0] == t2i_caption
            and t2i_out[1] == "4:3"
            and t2i_out[2] == "plan composition"
            and "model=text-chat-resident" in t2i_out[3]
            and "tps=200.0" in t2i_out[3]
            and len(ids) == 2 and ids[0] != ids[1]
            and last_body.get("system") != "PRESET SYSTEM PROMPT"
            and "reasoning_effort" not in last_body
            and "max_tokens" not in last_body,
            "heylook T2I picks resident chat model, retries 503 with fresh X-Request-ID and separates thinking",
        )

        # mode='heylook preset + aspect_ratio', editable custom_system_prompt overrides, custom sampler settings, and UI catalog route.
        MiniMaxH3ImagePromptExpander.execute(
            prompt="Overhead rustic farm table feast.",
            enabled=True,
            base_url=base,
            mode="heylook preset + aspect_ratio",
            custom_system_prompt="",
            preset="crisp_preset",
        )
        preset_ar_body = state["bodies"][-1]
        MiniMaxH3ImagePromptExpander.execute(
            prompt="Overhead rustic farm table feast.",
            enabled=True,
            base_url=base,
            mode="t2i + aspect_ratio",
            custom_system_prompt="USER EDITED SYSTEM PROMPT",
            sampler_mode="custom",
            temperature=1.15,
            top_p=0.9,
            top_k=40,
            min_p=0.05,
            max_tokens=2048,
            repetition_penalty=1.08,
            presence_penalty=0.1,
        )
        custom_sampler_body = state["bodies"][-1]
        catalog = build_expander_catalog(base)
        report(
            preset_ar_body.get("system", "").startswith("PRESET SYSTEM PROMPT")
            and "aspect_ratio:" in preset_ar_body.get("system", "")
            and custom_sampler_body.get("system") == "USER EDITED SYSTEM PROMPT"
            and custom_sampler_body.get("temperature") == 1.15
            and custom_sampler_body.get("top_p") == 0.9
            and custom_sampler_body.get("top_k") == 40
            and custom_sampler_body.get("max_tokens") == 2048
            and len(catalog["local_prompts"]) == 6
            and catalog["heylook"]["reachable"] is True
            and catalog["heylook"]["resident_models"] == ["text-chat-resident"]
            and [p["name"] for p in catalog["heylook"]["presets"]] == ["crisp_preset"],
            "heylook preset + aspect_ratio, editable system_prompt override, custom sampler fields and UI catalog hold",
        )

        # Edit with 2 reference pictures: requires 'vision' capability (selects vlm-chat-local)
        # and sends ordered <Picture 1> / <Picture 2> blocks on the user turn.
        img1 = torch.zeros((1, 128, 128, 3), dtype=torch.float32)
        img2 = torch.ones((1, 128, 128, 3), dtype=torch.float32)
        edit_out = tuple(MiniMaxH3ImagePromptExpander.execute(
            prompt="Take off the headphones and have them fist-bump.",
            enabled=True,
            base_url=base,
            mode="auto",
            model="",
            images={"image_2": img2, "image_1": img1},
        ))
        edit_body = state["bodies"][-1]
        ucontent = edit_body["messages"][0]["content"]
        labels = [b["text"] for b in ucontent if b["type"] == "text"]
        img_blocks = [b for b in ucontent if b["type"] == "image"]
        report(
            edit_body["model"] == "vlm-chat-local"
            and edit_out[0] == edit_caption
            and edit_out[1] == h3_config.H3_IMAGE_ASPECT_OF_PICTURE
            and labels == ["<Picture 1>:", "<Picture 2>:", "Take off the headphones and have them fist-bump."]
            and len(img_blocks) == 2 and all(b["source"]["type"] == "base64" for b in img_blocks),
            "heylook edit gates on vision capability and preserves <Picture 1..N> order",
        )

        # Disabled passthrough requires zero server traffic even on an unreachable port.
        before_calls = len(state["seen_request_ids"])
        off_out = tuple(MiniMaxH3ImagePromptExpander.execute(
            prompt=f"<think>offline</think>\n{t2i_caption}\n\naspect_ratio: 16:9",
            enabled=False,
            base_url="http://127.0.0.1:1",
        ))
        report(
            off_out[:3] == (t2i_caption, "16:9", "offline") and len(state["seen_request_ids"]) == before_calls,
            "disabled expander parses offline without touching the network",
        )

        # Controls: text-only model refused for edit, in-band error refused, truncated stream refused,
        # 403 Host check diagnosed, and mid-stream interrupt calls DELETE /v1/requests/{id}.
        controls_ok = True
        try:
            MiniMaxH3ImagePromptExpander.execute(
                prompt="Edit this.", enabled=True, base_url=base, model="text-chat-resident", images={"image_1": img1}
            )
            controls_ok = False
        except ValueError:
            pass

        try:
            MiniMaxH3ImagePromptExpander.execute(
                prompt="Test.", enabled=True, base_url=base, mode="heylook preset", preset="", custom_system_prompt=""
            )
            controls_ok = False
        except ValueError as exc:
            controls_ok &= "crisp_preset" in str(exc)

        state["mode"] = "inband_error"
        try:
            MiniMaxH3ImagePromptExpander.execute(prompt="Test.", enabled=True, base_url=base)
            controls_ok = False
        except ValueError as exc:
            controls_ok &= "image rejected" in str(exc)

        state["mode"] = "truncated_stream"
        try:
            MiniMaxH3ImagePromptExpander.execute(prompt="Test.", enabled=True, base_url=base)
            controls_ok = False
        except RuntimeError as exc:
            controls_ok &= "message_stop" in str(exc)

        state["mode"] = "host403"
        try:
            MiniMaxH3ImagePromptExpander.execute(prompt="Test.", enabled=True, base_url=base)
            controls_ok = False
        except PermissionError as exc:
            controls_ok &= "allowed_hosts" in str(exc)

        state["mode"] = "ok"
        state["cancelled_ids"].clear()
        with httpx.Client(trust_env=False) as c:
            seen_lines = 0

            def _abort_mid_stream() -> None:
                nonlocal seen_lines
                seen_lines += 1
                if seen_lines >= 3:
                    raise KeyboardInterrupt("user cancelled")

            try:
                stream_heylook_message(
                    base,
                    {"model": "text-chat-resident", "messages": [{"role": "user", "content": "hi"}]},
                    client=c,
                    interrupt_check=_abort_mid_stream,
                )
                controls_ok = False
            except KeyboardInterrupt:
                pass
        controls_ok &= len(state["cancelled_ids"]) == 1 and state["cancelled_ids"][0] == state["seen_request_ids"][-1]
        report(bool(controls_ok), "control: capability mismatch, in-band error, truncated stream, 403 and cancel all hold")
    finally:
        server.shutdown()
        server.server_close()


def graphs() -> None:
    folder = REPO / "workflows" / "image"
    paths = sorted(folder.glob("h3_image_*_api.json")) + sorted(folder.glob("h3_probe_h3_image_*_api.json"))
    if not paths:
        report(False, "the shipped graphs", "none found under workflows/image/")
        return
    for path in paths:
        g = json.loads(path.read_text())
        kinds = {n["class_type"] for n in g.values()}
        sig = [n for n in g.values() if n["class_type"] == "ManualSigmas"]
        ok = len(sig) == 1 and sig[0]["inputs"]["sigmas"] == h3_config.H3_IMAGE_MANUAL_SIGMAS
        ok &= not kinds & {"KSampler", "BasicScheduler", "CFGGuider", "ConditioningZeroOut", "CLIPTextEncode"}
        cond = next(n for n in g.values() if n["class_type"] == "MiniMaxH3ImageConditioning")["inputs"]
        size = next(n for n in g.values() if n["class_type"] == "MiniMaxH3ImageSize")["inputs"]
        ok &= isinstance(cond["width"], list) and isinstance(cond["height"], list)
        pictures = [v for k, v in cond.items() if k.startswith("images.")]
        if pictures:
            ok &= all(g[src[0]]["class_type"] == "LoadImage" for src in pictures)
            ok &= size.get("picture") == pictures[0] and size["shape"] == h3_config.H3_IMAGE_ASPECT_OF_PICTURE
        else:
            ok &= size["shape"] in h3_config.H3_IMAGE_ASPECTS
        expanders = [(nid, n["inputs"]) for nid, n in g.items() if n["class_type"] == "MiniMaxH3ImagePromptExpander"]
        if expanders:
            exp_id, exp_in = expanders[0]
            ok &= len(expanders) == 1
            ok &= cond["prompt"] == [exp_id, 0] and size.get("shape_from_expander") == [exp_id, 1]
            ok &= exp_in.get("base_url") == h3_config.HEYLOOK_DEFAULT_BASE_URL and exp_in.get("model") == ""
        else:
            ok &= isinstance(cond["prompt"], str) and bool(cond["prompt"].strip())
        vaedec = [(nid, n["inputs"]) for nid, n in g.items() if n["class_type"] == "VAEDecode"]
        down = [(nid, n["inputs"]) for nid, n in g.items() if n["class_type"] == "MiniMaxH3ImageDownscale"]
        save = [n["inputs"] for n in g.values() if n["class_type"] == "SaveImage"]
        size_ids = [nid for nid, n in g.items() if n["class_type"] == "MiniMaxH3ImageSize"]
        ok &= len(vaedec) == 1 and len(down) == 1 and len(save) == 1 and len(size_ids) == 1
        if vaedec and down and save and size_ids:
            vd_id, _ = vaedec[0]
            dn_id, dn_in = down[0]
            sz_id = size_ids[0]
            ok &= (
                dn_in.get("images") == [vd_id, 0]
                and dn_in.get("target_width") == [sz_id, 2]
                and dn_in.get("target_height") == [sz_id, 3]
                and save[0].get("images") == [dn_id, 0]
            )
        report(bool(ok), f"the shipped graph {path.name}")

    import build_workflows as bw
    g16 = bw.build_h3_image_api(
        "h3img_edit",
        prompt="Edit <Picture 1> through <Picture 16>.",
        ref_images=tuple(f"img_{i}.png" for i in range(16)),
        heylook=True,
    )
    c16 = next(n["inputs"] for n in g16.values() if n["class_type"] == "MiniMaxH3ImageConditioning")
    e16 = next(n["inputs"] for n in g16.values() if n["class_type"] == "MiniMaxH3ImagePromptExpander")
    loads16 = [nid for nid, n in g16.items() if n["class_type"] == "LoadImage"]
    ok16 = (
        len(loads16) == 16
        and [k for k in c16 if k.startswith("images.")] == [f"images.image_{i}" for i in range(16)]
        and [k for k in e16 if k.startswith("images.")] == [f"images.image_{i}" for i in range(1, 17)]
    )
    report(bool(ok16), "build_h3_image_api wires up to 16 reference pictures (0-based conditioning, 1-based expander)")

    g_edit_eg = bw.build_h3_image_api(
        "h3img_edit",
        prompt="Edit <Picture 1> through <Picture 16> under warm tungsten light.",
        ref_images=tuple(f"img_{i}.png" for i in range(16)),
        edit_guidance={"edits": "warm tungsten -> cool overcast"},
    )
    g_t2i_eg = bw.build_h3_image_api(
        "h3img_t2i",
        prompt="A brass compass resting on an oak workbench.",
        edit_guidance={"edits": "brass compass -> silver compass"},
    )
    ok_eg = (
        g_edit_eg.get("132", {}).get("class_type") == "MiniMaxH3EncodeImageReferences"
        and g_edit_eg["132"]["inputs"].get("canvas") == "explicit"
        and [k for k in g_edit_eg["132"]["inputs"] if k.startswith("images.")]
        == [f"images.image_{i}" for i in range(1, 17)]
        and g_edit_eg["5"]["class_type"] == "MiniMaxH3PromptOnImageReferences"
        and g_edit_eg["131"]["class_type"] == "MiniMaxH3PromptOnImageReferences"
        and g_edit_eg["5"]["inputs"] == {"encoded_references": ["132", 0], "prompt": ["133", 0]}
        and g_edit_eg["131"]["inputs"] == {"encoded_references": ["132", 0], "prompt": ["130", 0]}
        and g_edit_eg["10"]["inputs"]["latent_image"] == ["132", 1]
        and g_edit_eg["9"]["class_type"] == "MiniMaxH3EditGuider"
        and "132" not in g_t2i_eg
        and g_t2i_eg["5"]["class_type"] == "MiniMaxH3ImageConditioning"
        and g_t2i_eg["131"]["class_type"] == "MiniMaxH3ImageConditioning"
        and g_t2i_eg["5"]["inputs"]["prompt"] == ["133", 0]
        and g_t2i_eg["131"]["inputs"]["prompt"] == ["130", 0]
    )
    report(
        bool(ok_eg),
        "_wire_edit_guidance splits h3_image_edit into 1x MiniMaxH3EncodeImageReferences(canvas='explicit') + 2x MiniMaxH3PromptOnImageReferences",
    )


def release_grid(folder: Path | None) -> None:
    if folder is None:
        print("  skip  the grid constant against the release's pdd_config.json (no --release)")
        return
    meta = json.loads(rel.metadata(json.loads((folder / "pdd_config.json").read_text()))["config"])
    ours = [float(s) for s in h3_config.H3_IMAGE_MANUAL_SIGMAS.split(",")]
    report(ours == meta["transformer"]["pdd_sigmas"], "the grid constant is the release's pdd_config.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--release", type=Path)
    args = ap.parse_args()
    qkv_order()
    stacked_heads()
    writer()
    sizes()
    expander_replies()
    heylook_expander()
    graphs()
    release_grid(args.release)
    print("")
    print(f"  {fails} failure(s)" if fails else "  ok    MiniMax-H3-Image layout, sizes and graphs hold")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
