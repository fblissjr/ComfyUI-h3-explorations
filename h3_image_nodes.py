"""MiniMax-H3-Image: canvas size (`MiniMaxH3ImageSize`) and local `heylook` prompt expander (`MiniMaxH3ImagePromptExpander`).

1. `MiniMaxH3ImageSize`:
   Core's `MiniMax H3 Image Conditioning` takes a width and a height in pixels.
   Left at 0 it uses the first picture's own size, which for most pictures is
   below the size the model card says the model works well at; set by hand it
   means working out a size of the right area at the right shape with both sides
   a multiple of 32. This node does that arithmetic: it takes megapixels and a
   shape and gives the width and height to wire into the conditioning node.

   The rule is the vendor's (`size_for` in the release's `examples/context_ir.py`,
   held as `workflows/h3_config.py::h3_image_size`). The shape is one of the ten
   ratios the vendor's prompt expander may choose, the shape of a picture wired
   in, or an optional `shape_from_expander` string (`'4:3'`, `'5:3'`,
   `'same as the picture'`, ...) emitted by `MiniMaxH3ImagePromptExpander`.

2. `MiniMaxH3ImagePromptExpander`:
   Expands a short text-to-image request or `1..16`-picture edit instruction
   against a local or LAN `heylookitsanllm` server (`POST /v1/messages` SSE)
   using the standalone prompts in `docs/wiki/system_prompts/h3_image/`:
   - Resolves model ids at runtime from `GET /v1/models` (gated on `capabilities`
     `chat` and, when pictures are wired, `vision`, preferring a resident model
     from `GET /v1/system/metrics`; never a hardcoded model id).
   - Downscales wired reference pictures client-side (longest edge <= 2048 px,
     Lanczos, EXIF-aware for file/bytes callers, PNG kept for alpha/PNG inputs
     and quality-85 JPEG for photos) and sends them on the `user` turn in exact
     `<Picture 1>..<Picture N>` order.
   - Retries HTTP 503 (`model_overloaded`) with `Retry-After` backoff, sends a
     fresh UUID `X-Request-ID` per attempt, cancels in-flight runs via stream
     close + `DELETE /v1/requests/{id}` on ComfyUI interrupt, separates
     `thinking_delta` from `text_delta`, treats in-band `event: error` as a
     terminal failure, and parses `(prompt, shape)` through `parse_expander_reply`.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import io as _pyio
import json
import re
import sys
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Callable

import httpx
import numpy as np
from PIL import Image, ImageOps

_COMFY_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "comfy_api").is_dir()),
    None,
)
if _COMFY_ROOT is not None and str(_COMFY_ROOT) not in sys.path:
    sys.path.append(str(_COMFY_ROOT))

from comfy_api.latest import io

_HERE = Path(__file__).resolve().parent
if str(_HERE / "workflows") not in sys.path:
    sys.path.insert(0, str(_HERE / "workflows"))
from h3_config import (  # noqa: E402
    H3_IMAGE_ASPECT_OF_PICTURE,
    H3_IMAGE_ASPECTS,
    H3_IMAGE_MEGAPIXELS,
    H3_IMAGE_MEGAPIXELS_RANGE,
    HEYLOOK_DEFAULT_BASE_URL,
    h3_image_size,
)


def resolve_size(megapixels: float, shape: str, picture_size: tuple[int, int] | None) -> tuple[int, int]:
    """`(width, height)` for a shape choice; `picture_size` is `(width, height)` of the wired picture, if any."""
    if shape == H3_IMAGE_ASPECT_OF_PICTURE:
        if picture_size is None:
            raise ValueError(
                f"shape is '{H3_IMAGE_ASPECT_OF_PICTURE}' but no picture is wired into this node. "
                "Wire the picture being edited into 'picture', or choose a ratio.")
        x, y = picture_size
    else:
        x, y = (int(v) for v in shape.split(":"))
    return h3_image_size(megapixels, x, y)


_SAME_AS_PICTURE_ALIASES = frozenset({
    H3_IMAGE_ASPECT_OF_PICTURE.lower(),
    "same",
    "source",
    "adaptive",
})
_TRAILING_RATIO_RE = re.compile(
    r"(?:^|\n)\s*(?:target_)?aspect_ratio\s*:\s*([^\n]+?)\s*$",
    flags=re.IGNORECASE,
)


def _normalize_aspect_ratio(raw: str) -> str:
    val = raw.strip()
    if val.lower() in _SAME_AS_PICTURE_ALIASES:
        return H3_IMAGE_ASPECT_OF_PICTURE
    parts = val.split(":")
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        w, h = int(parts[0]), int(parts[1])
        if w > 0 and h > 0:
            return f"{w}:{h}"
    raise ValueError(
        f"invalid aspect_ratio {raw!r}; expected 'W:H' with positive integers or '{H3_IMAGE_ASPECT_OF_PICTURE}'"
    )


def parse_expander_reply(text: str) -> tuple[str, str | None]:
    """Split an LLM/VLM expander reply into `(clean_prompt, shape_or_none)`.

    Accepts:
    - Plain-text captions (returns `(prompt, None)`).
    - Plain-text captions followed by a trailing `aspect_ratio: <W:H | same as the picture>`
      (or `target_aspect_ratio: ...`) line (`docs/wiki/system_prompts/h3_image/`).
    - Vendor JSON objects (`{"expanded_caption": "...", "aspect_ratio" | "target_aspect_ratio": "..."}`).
    - Replies prefixed with `<think>...</think>` blocks (or prefilled `...</think>`) or wrapped
      in a single outer markdown code fence.

    Any non-`None` shape returned is either `H3_IMAGE_ASPECT_OF_PICTURE` or normalized `'W:H'`
    with positive integers, ready for `resolve_size`. Raises `ValueError` on unclosed `<think>`,
    malformed JSON, invalid ratio strings, or an empty caption.
    """
    if not isinstance(text, str):
        raise ValueError(f"expected str reply, got {type(text).__name__}")
    s = text.strip()
    while s.startswith("<think>"):
        end = s.find("</think>")
        if end < 0:
            raise ValueError("expander reply has an unclosed <think> block")
        s = s[end + len("</think>"):].strip()
    if "</think>" in s and "<think>" not in s.split("</think>", 1)[0]:
        s = s.split("</think>", 1)[1].strip()
    if "<think>" in s:
        raise ValueError("expander reply has an unclosed or mid-caption <think> tag")

    if s.startswith("```") and s.endswith("```"):
        first_nl = s.find("\n")
        if first_nl > 0:
            inner = s[first_nl + 1:-3].strip()
            if "```" not in inner:
                s = inner

    ratio: str | None = None
    if s.startswith("{"):
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as exc:
            raise ValueError(f"expander reply starts with '{{' but is not valid JSON: {exc}") from exc
        if not isinstance(obj, dict) or not isinstance(obj.get("expanded_caption"), str):
            raise ValueError("JSON expander reply must be an object with a string 'expanded_caption'")
        prompt = obj["expanded_caption"].strip()
        raw_ratio = obj.get("aspect_ratio") if "aspect_ratio" in obj else obj.get("target_aspect_ratio")
        if raw_ratio is not None:
            if not isinstance(raw_ratio, str):
                raise ValueError(f"JSON aspect_ratio must be a string, got {type(raw_ratio).__name__}")
            ratio = _normalize_aspect_ratio(raw_ratio)
    else:
        m = _TRAILING_RATIO_RE.search(s)
        if m is not None:
            ratio = _normalize_aspect_ratio(m.group(1))
            prompt = s[:m.start()].strip()
        else:
            prompt = s

    if not prompt:
        raise ValueError("expander reply has an empty caption")
    return prompt, ratio


class MiniMaxH3ImageSize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        low, high = H3_IMAGE_MEGAPIXELS_RANGE
        return io.Schema(
            node_id="MiniMaxH3ImageSize",
            display_name="MiniMax H3 Image Size",
            category="model/conditioning/minimax",
            description=(
                "Width and height for MiniMax H3 Image from megapixels and a shape. "
                "Wire them into MiniMax H3 Image Conditioning."
            ),
            inputs=[
                io.Float.Input(
                    "megapixels", default=H3_IMAGE_MEGAPIXELS, min=1.0, max=24.0, step=0.5,
                    tooltip=(
                        f"How large the picture is. The model is at its best near {H3_IMAGE_MEGAPIXELS:g} "
                        f"and works well from about {low:g} to {high:g}; below that the picture loses detail "
                        "and can fall apart. Smaller is faster and takes less memory.")),
                io.Combo.Input(
                    "shape", options=[*H3_IMAGE_ASPECTS, H3_IMAGE_ASPECT_OF_PICTURE], default="1:1",
                    tooltip=(
                        "Width to height. Choose a ratio, or 'same as the picture' to keep the shape of "
                        "the picture wired in below, which is what an edit of that picture usually wants.")),
                io.Image.Input(
                    "picture", optional=True,
                    tooltip=(
                        "Only its shape is read, and only when shape is 'same as the picture'. "
                        "The picture is not changed.")),
                io.String.Input(
                    "shape_from_expander", optional=True, force_input=True,
                    tooltip=(
                        "Optional shape ('4:3', '5:3', 'same as the picture', ...) from "
                        "MiniMax H3 Image Prompt Expander. Overrides 'shape' when non-empty; "
                        "falls back to 'shape' when empty or unwired.")),
            ],
            outputs=[
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(cls, megapixels, shape, picture=None, shape_from_expander=None) -> io.NodeOutput:
        picture_size = None if picture is None else (int(picture.shape[2]), int(picture.shape[1]))
        chosen_shape = (
            _normalize_aspect_ratio(shape_from_expander)
            if isinstance(shape_from_expander, str) and shape_from_expander.strip()
            else shape
        )
        width, height = resolve_size(megapixels, chosen_shape, picture_size)
        return io.NodeOutput(width, height)


MAX_H3_IMAGE_REFS = 16
HEYLOOK_IMAGE_MAX_EDGE = 2048
HEYLOOK_JPEG_QUALITY = 85
_SYSTEM_PROMPTS_DIR = _HERE / "docs" / "wiki" / "system_prompts" / "h3_image"

EXPANDER_PROMPT_FILES: dict[str, str] = {
    "t2i + aspect_ratio": "t2i_aspect_ratio_system_prompt.md",
    "t2i": "t2i_system_prompt.md",
    "t2i_design + aspect_ratio": "t2i_design_aspect_ratio_system_prompt.md",
    "t2i_design": "t2i_design_system_prompt.md",
    "edit + aspect_ratio": "edit_aspect_ratio_system_prompt.md",
    "edit": "edit_system_prompt.md",
}
EXPANDER_MODES: tuple[str, ...] = ("auto", *EXPANDER_PROMPT_FILES.keys(), "custom")
THINKING_MODES: tuple[str, ...] = ("model default", "on", "off")


class HeylookOverloadedError(RuntimeError):
    """HTTP 503 `model_overloaded`: queue signal on a single-user local server."""

    def __init__(self, message: str = "model_overloaded", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class HeylookStreamResult:
    text: str = ""
    thinking: str = ""
    model: str = ""
    request_id: str = ""
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: dict = field(default_factory=dict)
    performance: dict = field(default_factory=dict)


def _extract_inline_thinking(raw_text: str) -> str:
    """Extract leading `<think>...</think>` blocks when an engine emits tags inside `text`."""
    parts: list[str] = []
    s = raw_text.strip()
    while s.startswith("<think>"):
        end = s.find("</think>")
        if end < 0:
            break
        chunk = s[len("<think>"):end].strip()
        if chunk:
            parts.append(chunk)
        s = s[end + len("</think>"):].strip()
    if not parts and "</think>" in s and "<think>" not in s.split("</think>", 1)[0]:
        chunk = s.split("</think>", 1)[0].strip()
        if chunk:
            parts.append(chunk)
    return "\n\n".join(parts)


def normalize_heylook_base_url(base_url: str) -> str:
    raw = (base_url or "").strip().rstrip("/")
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(
            f"invalid heylook base_url {base_url!r}; expected http://<host>:<port> or https://<host>"
        )
    return raw


def resolve_expander_system_prompt(
    mode: str,
    *,
    has_images: bool,
    custom_system_prompt: str = "",
) -> tuple[str, str]:
    """Return `(resolved_mode, system_prompt_text)` for `MiniMaxH3ImagePromptExpander`."""
    if mode not in EXPANDER_MODES:
        raise ValueError(f"unknown expander mode {mode!r}; expected one of {list(EXPANDER_MODES)}")
    effective = ("edit + aspect_ratio" if has_images else "t2i + aspect_ratio") if mode == "auto" else mode
    if effective == "custom":
        sys_text = (custom_system_prompt or "").strip()
        if not sys_text:
            raise ValueError("mode is 'custom', but custom_system_prompt is empty")
        return effective, sys_text
    if effective.startswith("edit") and not has_images:
        raise ValueError(
            f"mode {effective!r} requires at least one reference picture wired into 'images' "
            "(or choose 'auto' / a 't2i*' mode for text-to-image)"
        )
    if effective.startswith("t2i") and has_images:
        raise ValueError(
            f"mode {effective!r} is text-to-image only, but reference pictures are wired into 'images'; "
            "choose 'auto', 'edit + aspect_ratio', 'edit', or 'custom'"
        )
    path = _SYSTEM_PROMPTS_DIR / EXPANDER_PROMPT_FILES[effective]
    return effective, path.read_text(encoding="utf-8").strip()


def prepare_heylook_image(path_or_bytes: bytes | str | Path) -> tuple[str, str]:
    """Downscale a file/bytes image for `heylook` `/v1/messages` (`(raw_base64, media_type)`).

    Matches `heylook`'s frontend contract: read `.format` BEFORE `exif_transpose`
    (which drops `.format` to `None`), apply EXIF orientation, bound the longest
    edge at 2048 px with Lanczos without enlarging, keep PNG as PNG, and encode
    everything else as quality-85 RGB JPEG.
    """
    src = _pyio.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    with Image.open(src) as opened:
        keep_png = (opened.format or "").upper() == "PNG"
        img = ImageOps.exif_transpose(opened)
    img.thumbnail((HEYLOOK_IMAGE_MAX_EDGE, HEYLOOK_IMAGE_MAX_EDGE), Image.Resampling.LANCZOS)
    buf = _pyio.BytesIO()
    if keep_png:
        img.save(buf, format="PNG", optimize=True)
        media_type = "image/png"
    else:
        img.convert("RGB").save(buf, format="JPEG", quality=HEYLOOK_JPEG_QUALITY)
        media_type = "image/jpeg"
    return base64.b64encode(buf.getvalue()).decode("ascii"), media_type


def prepare_heylook_tensor(img_tensor, *, index: int = 1) -> tuple[str, str]:
    """Downscale a ComfyUI `[1, H, W, C]` float image tensor to `(raw_base64, media_type)`."""
    shape = tuple(int(d) for d in img_tensor.shape)
    if len(shape) != 4 or shape[0] != 1 or shape[3] not in (3, 4):
        raise ValueError(
            f"image_{index} must be a single still image with shape [1, H, W, 3|4], got {shape}"
        )
    arr = img_tensor[0].detach().cpu().numpy() if hasattr(img_tensor, "detach") else np.asarray(img_tensor[0])
    u8 = np.clip(np.rint(arr * 255.0), 0.0, 255.0).astype(np.uint8)
    has_alpha = u8.shape[2] == 4 and bool(np.any(u8[:, :, 3] < 255))
    pil_mode = "RGBA" if has_alpha else "RGB"
    img = Image.fromarray(u8 if has_alpha else u8[:, :, :3], mode=pil_mode)
    img.thumbnail((HEYLOOK_IMAGE_MAX_EDGE, HEYLOOK_IMAGE_MAX_EDGE), Image.Resampling.LANCZOS)
    buf = _pyio.BytesIO()
    if has_alpha:
        img.save(buf, format="PNG", optimize=True)
        media_type = "image/png"
    else:
        img.save(buf, format="JPEG", quality=HEYLOOK_JPEG_QUALITY)
        media_type = "image/jpeg"
    return base64.b64encode(buf.getvalue()).decode("ascii"), media_type


def _fact_value(node, *keys):
    cur = node
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur.get("value") if isinstance(cur, dict) else None


def _http_error(r: httpx.Response, base: str, path: str) -> Exception:
    if r.status_code == 503:
        try:
            retry = float(r.headers.get("Retry-After", ""))
        except ValueError:
            retry = None
        msg = "model_overloaded"
        try:
            err_msg = (r.json().get("error") or {}).get("message")
            if err_msg:
                msg = f"model_overloaded: {err_msg}"
        except Exception:
            pass
        return HeylookOverloadedError(msg, retry_after=retry)
    if r.status_code == 403:
        return PermissionError(
            f"{base}{path} answered HTTP 403 (heylook Host check / DNS-rebinding guard). "
            "Address the server by IP or 127.0.0.1, or add the hostname to `allowed_hosts` in heylook.toml."
        )
    if r.status_code == 401:
        return PermissionError(
            f"{base}{path} answered HTTP 401: heylook has no inference key, so a proxy in front of it rejected the request."
        )
    try:
        payload = r.json()
    except Exception:
        detail = r.text[:240]
    else:
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if detail is None and isinstance(payload, dict):
            detail = (payload.get("error") or {}).get("message")
        if detail is None:
            detail = str(payload)[:240]
    if r.status_code in (400, 422):
        return ValueError(f"heylook HTTP {r.status_code} on {path}: {detail}")
    return RuntimeError(f"heylook HTTP {r.status_code} on {path}: {detail}")


def discover_heylook_model(
    client: httpx.Client,
    base: str,
    requested_model: str = "",
    *,
    need: set[str] | frozenset[str] = frozenset({"chat"}),
) -> dict:
    """Resolve and capability-gate a `/v1/models` row at runtime (`capabilities`, never `modalities`)."""
    try:
        r = client.get(f"{base}/v1/models", headers={"Accept": "application/json"}, timeout=10.0)
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"cannot reach heylook at {base}/v1/models ({type(exc).__name__}: {exc}). "
            "Check that heylook is running and base_url points to its host:port."
        ) from exc
    if r.status_code >= 400:
        raise _http_error(r, base, "/v1/models")
    try:
        payload = r.json()
    except ValueError as exc:
        raise RuntimeError(f"{base}/v1/models returned non-JSON; is {base} a heylook server?") from exc
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError(f"{base}/v1/models returned no 'data' list; is {base} a heylook server?")
    usable = [row for row in rows if isinstance(row, dict) and row.get("id")]
    req_set = set(need)
    fit = [row for row in usable if req_set <= set(row.get("capabilities") or [])]

    wanted = (requested_model or "").strip()
    if wanted and wanted.lower() != "auto":
        chosen = next((row for row in usable if row["id"] == wanted), None)
        if chosen is None:
            fit_ids = [row["id"] for row in fit]
            raise ValueError(
                f"model {wanted!r} is not served by {base}/v1/models; "
                f"models serving {sorted(req_set)}: {fit_ids}"
            )
        caps = set(chosen.get("capabilities") or [])
        missing = sorted(req_set - caps)
        if missing:
            fit_ids = [row["id"] for row in fit]
            raise ValueError(
                f"model {wanted!r} lacks required capabilities {missing} (has {sorted(caps)}); "
                f"models serving {sorted(req_set)}: {fit_ids}"
            )
        return chosen

    if not fit:
        roster = [(row["id"], row.get("capabilities")) for row in usable]
        raise LookupError(f"no model at {base} serves capabilities {sorted(req_set)}; roster: {roster}")

    resident: set[str] = set()
    try:
        mr = client.get(f"{base}/v1/system/metrics", headers={"Accept": "application/json"}, timeout=5.0)
        if mr.status_code == 200:
            models_map = mr.json().get("models")
            if isinstance(models_map, dict):
                resident = set(models_map.keys())
    except Exception:
        pass
    return next((row for row in fit if row["id"] in resident), fit[0])


def expand_heylook_preset(client: httpx.Client, base: str, preset_id: str, model_row: dict) -> dict:
    """Expand a `/v1/presets` entry client-side with capability & depth filtering."""
    r = client.get(f"{base}/v1/presets", headers={"Accept": "application/json"}, timeout=10.0)
    if r.status_code >= 400:
        raise _http_error(r, base, "/v1/presets")
    presets = r.json().get("presets") or []
    preset = next((p for p in presets if p.get("id") == preset_id), None)
    if preset is None:
        preset = next((p for p in presets if p.get("name") == preset_id), None)
    if preset is None:
        avail = [p.get("id") for p in presets if isinstance(p, dict) and p.get("id")]
        raise ValueError(f"heylook preset {preset_id!r} not found at {base}/v1/presets; available: {avail}")

    params = dict(preset.get("params") or {})
    body: dict = {}
    if preset.get("system_prompt"):
        body["system"] = preset["system_prompt"]

    caps = set(model_row.get("capabilities") or [])
    switch = params.pop("enable_thinking", None)
    budget = params.pop("thinking_budget_tokens", None)
    if budget is not None and "thinking_budget" in caps:
        body["thinking"] = {"budget_tokens": budget}
        if switch is not None:
            body["thinking"]["type"] = "enabled" if switch else "disabled"
    elif switch is not None and "thinking" in caps:
        body["thinking"] = bool(switch)

    effort = params.pop("reasoning_effort", None)
    depth = ((model_row.get("engine") or {}).get("thinking") or {}).get("depth") or {}
    if effort is not None and (
        depth.get("unknown") == "verbatim"
        or effort in (depth.get("values") or [])
        or effort in (depth.get("aliases") or {})
    ):
        body["reasoning_effort"] = effort

    body.update(params)
    return body


def build_heylook_request_body(
    *,
    model_row: dict,
    system_prompt: str,
    user_prompt: str,
    prepared_images: list[tuple[str, str]],
    thinking: str = "model default",
    request_seed: int | None = None,
    preset_fields: dict | None = None,
) -> dict:
    """Assemble a capability-gated `POST /v1/messages` request body."""
    if thinking not in THINKING_MODES:
        raise ValueError(f"unknown thinking setting {thinking!r}; expected one of {list(THINKING_MODES)}")

    caps = set(model_row.get("capabilities") or [])
    engine = model_row.get("engine") or {}
    allowed_fields = _fact_value(engine, "decoding", "request_fields")
    allowed_set = set(allowed_fields) if isinstance(allowed_fields, list) else None

    body: dict = dict(preset_fields or {})
    if system_prompt:
        body["system"] = system_prompt

    thinking_meta = engine.get("thinking") or {}
    has_switch = "thinking" in caps or bool(thinking_meta.get("switch"))
    if thinking in ("on", "off"):
        want_on = thinking == "on"
        if has_switch:
            existing = body.get("thinking")
            if want_on and isinstance(existing, dict) and "budget_tokens" in existing:
                body["thinking"] = {**existing, "type": "enabled"}
            else:
                body["thinking"] = want_on
        elif not want_on:
            off_words = (thinking_meta.get("depth") or {}).get("off") or []
            if off_words:
                body["reasoning_effort"] = off_words[0]
            body.pop("thinking", None)
        else:
            body.pop("thinking", None)

    if request_seed is not None and (allowed_set is None or "seed" in allowed_set):
        body["seed"] = int(request_seed) & 0x7FFFFFFF

    if allowed_set is not None:
        sampler_keys = (
            "max_tokens", "temperature", "top_p", "top_k", "min_p",
            "repetition_penalty", "repetition_context_size", "presence_penalty", "seed",
        )
        for k in sampler_keys:
            if k in body and k not in allowed_set:
                del body[k]

    if prepared_images:
        content: list[dict] = []
        for idx, (b64_data, media_type) in enumerate(prepared_images, start=1):
            content.append({"type": "text", "text": f"<Picture {idx}>:"})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64_data,
                },
            })
        content.append({"type": "text", "text": user_prompt})
        messages = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "user", "content": user_prompt}]

    body["model"] = model_row["id"]
    body["messages"] = messages
    body["stream"] = True
    return body


def _iter_sse(lines, interrupt_check: Callable[[], None] | None = None):
    event: str | None = None
    for line in lines:
        if interrupt_check is not None:
            interrupt_check()
        if not line:
            event = None
            continue
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: ") and event:
            yield event, json.loads(line[6:])


def cancel_heylook_request(base: str, request_id: str, *, client: httpx.Client | None = None) -> bool:
    """`DELETE /v1/requests/{request_id}`. 404 means already finished (`False`)."""
    encoded = urllib.parse.quote(request_id, safe="")
    url = f"{base}/v1/requests/{encoded}"
    if client is not None:
        r = client.delete(url, timeout=5.0)
    else:
        with httpx.Client(trust_env=False, timeout=5.0) as c:
            r = c.delete(url)
    if r.status_code == 404:
        return False
    r.raise_for_status()
    return int(r.json().get("cancelled") or 0) > 0


def with_heylook_backoff(
    fn: Callable[[], HeylookStreamResult],
    *,
    attempts: int = 5,
    interrupt_check: Callable[[], None] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> HeylookStreamResult:
    """Retry HTTP 503 (`HeylookOverloadedError`) honouring `Retry-After` (floored at 1s, capped at 30s)."""
    for i in range(attempts):
        try:
            return fn()
        except HeylookOverloadedError as exc:
            if i == attempts - 1:
                raise
            wait_s = float(min(max(exc.retry_after or (2 ** i), 1.0), 30.0))
            remaining = wait_s
            while remaining > 0.0:
                if interrupt_check is not None:
                    interrupt_check()
                step = min(remaining, 0.25)
                sleep_fn(step)
                remaining -= step
    raise RuntimeError("unreachable")


def stream_heylook_message(
    base: str,
    body: dict,
    *,
    client: httpx.Client | None = None,
    interrupt_check: Callable[[], None] | None = None,
) -> HeylookStreamResult:
    """Stream one `POST /v1/messages` request with a fresh `X-Request-ID` and cancel-on-abort."""
    req_id = str(uuid.uuid4())
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "X-Request-ID": req_id,
    }
    payload = {**body, "stream": True}
    out = HeylookStreamResult(model=str(body.get("model") or ""), request_id=req_id)
    finished = False
    tracked_id = req_id

    def _run_with(c: httpx.Client) -> None:
        nonlocal finished, tracked_id
        try:
            with c.stream("POST", f"{base}/v1/messages", json=payload, headers=headers) as r:
                tracked_id = r.headers.get("X-Request-ID") or req_id
                out.request_id = tracked_id
                if r.status_code >= 400:
                    r.read()
                    raise _http_error(r, base, "/v1/messages")

                for event, data in _iter_sse(r.iter_lines(), interrupt_check=interrupt_check):
                    if event == "content_block_delta":
                        delta = data.get("delta") or {}
                        dtype = delta.get("type")
                        if dtype == "thinking_delta":
                            out.thinking += delta.get("thinking") or delta.get("text") or ""
                        elif dtype == "text_delta":
                            out.text += delta.get("text") or ""
                    elif event == "message_delta":
                        delta = data.get("delta") or {}
                        out.stop_reason = delta.get("stop_reason")
                        out.stop_sequence = delta.get("stop_sequence")
                        out.usage = data.get("usage") or {}
                    elif event == "message_stop":
                        out.performance = data.get("performance") or {}
                        finished = True
                        break
                    elif event == "error":
                        err = data.get("error") or {}
                        etype = err.get("type") or "error"
                        emsg = err.get("message") or str(data)
                        if etype == "invalid_request_error":
                            raise ValueError(f"heylook invalid_request_error: {emsg}")
                        raise RuntimeError(f"heylook {etype}: {emsg}")
        except BaseException:
            if not finished:
                try:
                    cancel_heylook_request(base, tracked_id, client=c)
                except Exception:
                    pass
            raise

    if client is not None:
        _run_with(client)
    else:
        with httpx.Client(trust_env=False, timeout=httpx.Timeout(None, connect=10.0)) as owned:
            _run_with(owned)

    if not finished:
        raise RuntimeError("heylook stream ended without message_stop")
    return out


def format_heylook_report(
    res: HeylookStreamResult,
    *,
    resolved_mode: str,
    shape: str | None,
) -> str:
    """Format telemetry defensively (`generation_duration_ms` as throughput denominator)."""
    parts = [f"model={res.model}", f"mode={resolved_mode}"]
    if shape:
        parts.append(f"shape={shape}")
    if res.stop_reason:
        parts.append(f"stop={res.stop_reason}")
    usage = res.usage if isinstance(res.usage, dict) else {}
    in_tok = usage.get("input_tokens")
    cached_tok = usage.get("cache_read_input_tokens")
    out_tok = usage.get("output_tokens")
    if in_tok is not None:
        parts.append(f"in={in_tok}" + (f"(+{cached_tok} cached)" if cached_tok is not None else ""))
    if out_tok is not None:
        parts.append(f"out={out_tok}")

    perf = res.performance if isinstance(res.performance, dict) else {}
    gen_tps = perf.get("generation_tps")
    gen_ms = perf.get("generation_duration_ms")
    if gen_tps is None and out_tok is not None and gen_ms is not None and gen_ms > 0:
        gen_tps = float(out_tok) / (float(gen_ms) / 1000.0)
    if gen_tps is not None:
        parts.append(f"tps={float(gen_tps):.1f}")
    if gen_ms is not None:
        parts.append(f"gen_ms={float(gen_ms):.0f}")
    return " ".join(parts)


def _comfy_interrupt_check() -> None:
    try:
        import comfy.model_management as mm
    except Exception:
        return
    mm.throw_exception_if_processing_interrupted()


class MiniMaxH3ImagePromptExpander(io.ComfyNode):
    """Expand a short T2I request or `1..16`-picture edit instruction via a local/LAN `heylookitsanllm` server."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ImagePromptExpander",
            display_name="MiniMax H3 Image Prompt Expander (heylook)",
            category="model/conditioning/minimax",
            description=(
                "Expands a short request or 1..16-picture edit instruction into a finished "
                "MiniMax-H3-Image caption and canvas shape using a local or LAN heylookitsanllm "
                "server (/v1/messages). Resolves model ids at runtime from /v1/models and "
                "downscales wired reference pictures client-side before sending."
            ),
            inputs=[
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    tooltip=(
                        "Short scene request or edit instruction (referring to wired pictures in order, "
                        "or <Picture 1>..<Picture 16>). When 'enabled' is False, passed through offline "
                        "via parse_expander_reply without calling the server."
                    ),
                ),
                io.Boolean.Input(
                    "enabled",
                    default=True,
                    tooltip=(
                        "Call heylook to expand 'prompt'. Turn off to pass a finished caption "
                        "(or a pasted offline expander reply) straight through without network I/O."
                    ),
                ),
                io.String.Input(
                    "base_url",
                    default=HEYLOOK_DEFAULT_BASE_URL,
                    tooltip=(
                        "Base URL of your running heylookitsanllm server (e.g. http://127.0.0.1:8080 "
                        "locally, or http://<lan-or-tailscale-ip>:<port> on another machine; address "
                        "LAN hosts by IP so heylook's Host check passes)."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=list(EXPANDER_MODES),
                    default="auto",
                    tooltip=(
                        "Which standalone system prompt from docs/wiki/system_prompts/h3_image/ to use. "
                        "'auto' picks 't2i + aspect_ratio' when no pictures are wired and "
                        "'edit + aspect_ratio' when pictures are wired; 't2i_design*' is for posters, "
                        "typography, infographics, and UI boards."
                    ),
                ),
                io.String.Input(
                    "model",
                    default="",
                    tooltip=(
                        "Optional model id from GET /v1/models. Leave blank to auto-select a served "
                        "'chat' model (and 'vision' when pictures are wired), preferring one already "
                        "resident in memory."
                    ),
                ),
                io.Combo.Input(
                    "thinking",
                    options=list(THINKING_MODES),
                    default="model default",
                    tooltip=(
                        "Thinking switch on models that advertise 'thinking' (or depth 'off' words). "
                        "Reasoning blocks are separated onto the 'thinking' output and never reach "
                        "MiniMax H3 Image Conditioning."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip="Seed forwarded to heylook (and used by ComfyUI's cache/randomize control).",
                ),
                io.String.Input(
                    "preset",
                    default="",
                    advanced=True,
                    tooltip=(
                        "Optional preset id or name from GET /v1/presets on the heylook server "
                        "(for custom temperature, max_tokens, thinking budget, or reasoning_effort)."
                    ),
                ),
                io.String.Input(
                    "custom_system_prompt",
                    multiline=True,
                    default="",
                    advanced=True,
                    tooltip="System prompt used when mode is 'custom'.",
                ),
                io.Autogrow.Input(
                    "images",
                    optional=True,
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[f"image_{i}" for i in range(1, MAX_H3_IMAGE_REFS + 1)],
                        min=0,
                    ),
                    tooltip=(
                        "Optional reference pictures for editing (<Picture 1>..<Picture 16>). "
                        "Downscaled client-side to <= 2048 px on the longest edge for the vision LLM; "
                        "wire the full-size LoadImage outputs into MiniMax H3 Image Conditioning as usual."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="prompt",
                    tooltip="Expanded caption ready for MiniMax H3 Image Conditioning.",
                ),
                io.String.Output(
                    display_name="shape",
                    tooltip="Recommended aspect ratio ('4:3', 'same as the picture', ...), or empty if none. Wire into MiniMax H3 Image Size's shape_from_expander.",
                ),
                io.String.Output(
                    display_name="thinking",
                    tooltip="Reasoning trace emitted by the model, if any.",
                ),
                io.String.Output(
                    display_name="report",
                    tooltip="Resolved model id, mode, shape, token counts, and throughput.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        prompt: str = "",
        enabled: bool = True,
        base_url: str = HEYLOOK_DEFAULT_BASE_URL,
        mode: str = "auto",
        model: str = "",
        thinking: str = "model default",
        seed: int = 0,
        preset: str = "",
        custom_system_prompt: str = "",
        images: io.Autogrow.Type = None,
    ) -> io.NodeOutput:
        if not enabled:
            raw = (prompt or "").strip()
            if not raw:
                return io.NodeOutput("", "", "", "expander disabled (empty prompt)")
            inline_think = _extract_inline_thinking(raw)
            clean_prompt, ratio = parse_expander_reply(raw)
            return io.NodeOutput(clean_prompt, ratio or "", inline_think, "expander disabled (passthrough)")

        user_text = (prompt or "").strip()
        if not user_text:
            raise ValueError("MiniMaxH3ImagePromptExpander: 'prompt' is empty")

        names = sorted(images or {}, key=lambda name: int(name.rsplit("_", 1)[-1]))
        raw_images = [images[name] for name in names if images[name] is not None]
        if len(raw_images) > MAX_H3_IMAGE_REFS:
            raise ValueError(
                f"MiniMaxH3ImagePromptExpander takes at most {MAX_H3_IMAGE_REFS} images, got {len(raw_images)}"
            )

        has_images = len(raw_images) > 0
        resolved_mode, sys_prompt = resolve_expander_system_prompt(
            mode, has_images=has_images, custom_system_prompt=custom_system_prompt
        )
        prepared_images = [
            prepare_heylook_tensor(t, index=i) for i, t in enumerate(raw_images, start=1)
        ]
        base = normalize_heylook_base_url(base_url)
        need_caps = frozenset({"chat", "vision"} if has_images else {"chat"})

        with httpx.Client(trust_env=False, timeout=httpx.Timeout(None, connect=10.0)) as client:
            model_row = discover_heylook_model(client, base, model, need=need_caps)
            preset_name = (preset or "").strip()
            preset_fields = (
                expand_heylook_preset(client, base, preset_name, model_row)
                if preset_name
                else None
            )
            body = build_heylook_request_body(
                model_row=model_row,
                system_prompt=sys_prompt,
                user_prompt=user_text,
                prepared_images=prepared_images,
                thinking=thinking,
                request_seed=int(seed) & 0x7FFFFFFF,
                preset_fields=preset_fields,
            )
            res = with_heylook_backoff(
                lambda: stream_heylook_message(
                    base, body, client=client, interrupt_check=_comfy_interrupt_check
                ),
                interrupt_check=_comfy_interrupt_check,
            )

        think_trace = res.thinking or _extract_inline_thinking(res.text)
        try:
            clean_prompt, ratio = parse_expander_reply(res.text)
        except ValueError as exc:
            if res.stop_reason == "max_tokens":
                raise RuntimeError(
                    f"heylook stopped at max_tokens before completing a valid caption ({exc}). "
                    "Raise max_tokens in your heylook model/preset config or set thinking='off'."
                ) from exc
            raise

        report_str = format_heylook_report(res, resolved_mode=resolved_mode, shape=ratio)
        return io.NodeOutput(clean_prompt, ratio or "", think_trace, report_str)
