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
    H3_IMAGE_MIN_GENERATION_MEGAPIXELS,
    H3_IMAGE_SIZE_MULTIPLE,
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


def resolve_canvas_and_target_size(
    megapixels: float,
    shape: str,
    picture_size: tuple[int, int] | None,
    *,
    enforce_4mp_floor: bool = True,
) -> tuple[int, int, int, int]:
    """`(canvas_width, canvas_height, target_width, target_height)`.

    Matches `coderef/sglang` (`minimax_h3_image_generation_canvas`) and
    `coderef/vllm-omni` (`pipeline_minimax_h3_image.py`): when `megapixels < 4.0`
    and `enforce_4mp_floor` is enabled, the DiT generates on a 4.0 MP (`2048x2048`
    area) canvas of the same aspect ratio and `MiniMaxH3ImageDownscale` resizes
    the decoded image to `(target_width, target_height)` via 8-bit PIL Lanczos.
    At `megapixels >= 4.0`, canvas and target dimensions are identical.
    """
    target_w, target_h = resolve_size(megapixels, shape, picture_size)
    if enforce_4mp_floor and megapixels < H3_IMAGE_MIN_GENERATION_MEGAPIXELS:
        min_area = H3_IMAGE_MIN_GENERATION_MEGAPIXELS * 1024 * 1024
        scale = max(1.0, (min_area / (target_w * target_h)) ** 0.5)
        m = H3_IMAGE_SIZE_MULTIPLE
        canvas_w = max(m, int(round(target_w * scale / m)) * m)
        canvas_h = max(m, int(round(target_h * scale / m)) * m)
        return canvas_w, canvas_h, target_w, target_h
    return target_w, target_h, target_w, target_h


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
                "Canvas and target output size for MiniMax H3 Image from megapixels and a shape. "
                "Wire width and height into MiniMax H3 Image Conditioning, and target_width / "
                "target_height into MiniMax H3 Image Downscale for < 4 MP Lanczos downscale."
            ),
            inputs=[
                io.Float.Input(
                    "megapixels", default=H3_IMAGE_MEGAPIXELS, min=0.25, max=24.0, step=0.25,
                    tooltip=(
                        f"Requested output size in megapixels (1024x1024 units). The DiT is tuned for "
                        f"{low:g}..{high:g} MP (best near {H3_IMAGE_MEGAPIXELS:g}); below {low:g} MP, "
                        "enforce_4mp_floor generates at 4 MP and downscales cleanly via Lanczos.")),
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
                io.Boolean.Input(
                    "enforce_4mp_floor", default=True, optional=True,
                    tooltip=(
                        "Match SGLang and vLLM-Omni: when megapixels < 4.0, keep the DiT generation "
                        "canvas (width, height) at >= 4.0 MP so fine detail holds together, and emit "
                        "the requested dimensions on (target_width, target_height) for Lanczos downscale.")),
            ],
            outputs=[
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Int.Output(display_name="target_width"),
                io.Int.Output(display_name="target_height"),
            ],
        )

    @classmethod
    def execute(
        cls,
        megapixels,
        shape,
        picture=None,
        shape_from_expander=None,
        enforce_4mp_floor=True,
    ) -> io.NodeOutput:
        picture_size = None if picture is None else (int(picture.shape[2]), int(picture.shape[1]))
        chosen_shape = (
            _normalize_aspect_ratio(shape_from_expander)
            if isinstance(shape_from_expander, str) and shape_from_expander.strip()
            else shape
        )
        width, height, target_width, target_height = resolve_canvas_and_target_size(
            megapixels,
            chosen_shape,
            picture_size,
            enforce_4mp_floor=bool(enforce_4mp_floor),
        )
        return io.NodeOutput(width, height, target_width, target_height)


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
EXPANDER_MODES: tuple[str, ...] = (
    "auto",
    "heylook preset",
    "heylook preset + aspect_ratio",
    *EXPANDER_PROMPT_FILES.keys(),
    "custom",
)
THINKING_MODES: tuple[str, ...] = ("model default", "on", "off")
SAMPLER_MODES: tuple[str, ...] = ("heylook / preset default", "custom")

ASPECT_RATIO_FOOTER_T2I = (
    "Output contract:\n"
    "Output ONLY the single continuous English caption paragraph (with no internal line breaks), "
    "followed by one blank line and the exact final line `aspect_ratio: <ALLOWED_RATIO>` chosen from "
    "`1:1`, `5:4`, `4:3`, `3:2`, `16:9`, `21:9`, `4:5`, `3:4`, `2:3`, or `9:16`:\n\n"
    "<single continuous English paragraph>\n\n"
    "aspect_ratio: 4:3\n\n"
    "Do not output JSON, markdown fences, headings, or any other text before or after those lines."
)
ASPECT_RATIO_FOOTER_EDIT = (
    "Output contract:\n"
    "Output ONLY the single continuous English edit paragraph (with no internal line breaks), "
    "followed by one blank line and the exact final line `aspect_ratio: <same as the picture | W:H>` "
    "(use `same as the picture` whenever the edit keeps the base picture's canvas shape, or a reduced "
    "integer `W:H` such as `16:9`, `4:3`, `1:1`, or `9:16` when the edit crops or changes the canvas shape):\n\n"
    "<single continuous English paragraph>\n\n"
    "aspect_ratio: same as the picture\n\n"
    "Do not output JSON, markdown fences, headings, or any other text before or after those lines."
)


def _read_local_prompt(name: str) -> str:
    return (_SYSTEM_PROMPTS_DIR / EXPANDER_PROMPT_FILES[name]).read_text(encoding="utf-8").strip()


def append_aspect_ratio_footer(system_prompt: str, *, has_images: bool) -> str:
    """Append the `aspect_ratio:` output contract to a prose-only system prompt if not already present."""
    base_text = (system_prompt or "").strip()
    if "aspect_ratio:" in base_text.lower():
        return base_text
    footer = ASPECT_RATIO_FOOTER_EDIT if has_images else ASPECT_RATIO_FOOTER_T2I
    return f"{base_text}\n\n{footer}" if base_text else footer


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
    has_preset: bool = False,
) -> tuple[str, str]:
    """Return `(resolved_mode, system_prompt_text)` for `MiniMaxH3ImagePromptExpander`.

    Whenever `custom_system_prompt` is non-empty (either populated by the UI from a local mode or
    `heylook` preset, or edited directly by the user), that exact text wins as the system prompt.
    """
    if mode not in EXPANDER_MODES:
        raise ValueError(f"unknown expander mode {mode!r}; expected one of {list(EXPANDER_MODES)}")
    effective = ("edit + aspect_ratio" if has_images else "t2i + aspect_ratio") if mode == "auto" else mode
    if effective.startswith("edit") and not has_images:
        raise ValueError(
            f"mode {effective!r} requires at least one reference picture wired into 'images' "
            "(or choose 'auto', 'heylook preset*', or a 't2i*' mode for text-to-image)"
        )
    if effective.startswith("t2i") and has_images:
        raise ValueError(
            f"mode {effective!r} is text-to-image only, but reference pictures are wired into 'images'; "
            "choose 'auto', 'heylook preset*', 'edit + aspect_ratio', 'edit', or 'custom'"
        )

    override_text = (custom_system_prompt or "").strip()
    if override_text:
        if mode == "auto":
            if has_images and override_text == _read_local_prompt("t2i + aspect_ratio"):
                return effective, _read_local_prompt("edit + aspect_ratio")
            if not has_images and override_text == _read_local_prompt("edit + aspect_ratio"):
                return effective, _read_local_prompt("t2i + aspect_ratio")
        return effective, override_text

    if effective in ("heylook preset", "heylook preset + aspect_ratio"):
        return effective, ""
    if effective == "custom":
        if not has_preset:
            raise ValueError(
                "mode is 'custom', but custom_system_prompt is empty (and no heylook preset is set)"
            )
        return effective, ""
    return effective, _read_local_prompt(effective)


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


def expand_heylook_preset(
    client: httpx.Client,
    base: str,
    preset_id: str,
    model_row: dict,
) -> dict:
    """Expand a `/v1/presets` entry client-side with capability & depth filtering."""
    r = client.get(f"{base}/v1/presets", headers={"Accept": "application/json"}, timeout=10.0)
    if r.status_code >= 400:
        raise _http_error(r, base, "/v1/presets")
    presets = [p for p in (r.json().get("presets") or []) if isinstance(p, dict)]
    avail = [p.get("name") or p.get("id") for p in presets if p.get("name") or p.get("id")]

    wanted = (preset_id or "").strip()
    if not wanted:
        raise ValueError(
            f"'preset' is blank; choose a preset from {base}/v1/presets: {avail}"
        )
    preset = next((p for p in presets if p.get("id") == wanted), None)
    if preset is None:
        preset = next((p for p in presets if p.get("name") == wanted), None)
    if preset is None:
        lowered = wanted.lower()
        preset = next((p for p in presets if str(p.get("name") or "").lower() == lowered), None)
    if preset is None:
        raise ValueError(f"heylook preset {wanted!r} not found at {base}/v1/presets; available: {avail}")

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
    sampler_mode: str = "heylook / preset default",
    sampler_overrides: dict | None = None,
    reasoning_effort_override: str = "",
) -> dict:
    """Assemble a capability-gated `POST /v1/messages` request body."""
    if thinking not in THINKING_MODES:
        raise ValueError(f"unknown thinking setting {thinking!r}; expected one of {list(THINKING_MODES)}")
    if sampler_mode not in SAMPLER_MODES:
        raise ValueError(f"unknown sampler_mode {sampler_mode!r}; expected one of {list(SAMPLER_MODES)}")

    caps = set(model_row.get("capabilities") or [])
    engine = model_row.get("engine") or {}
    allowed_fields = _fact_value(engine, "decoding", "request_fields")
    allowed_set = set(allowed_fields) if isinstance(allowed_fields, list) else None

    body: dict = dict(preset_fields or {})
    if system_prompt:
        body["system"] = system_prompt

    if sampler_mode == "custom" and sampler_overrides:
        body.update(sampler_overrides)

    thinking_meta = engine.get("thinking") or {}
    depth = thinking_meta.get("depth") or {}
    eff = (reasoning_effort_override or "").strip() if sampler_mode == "custom" else ""
    if eff:
        if (
            depth.get("unknown") == "verbatim"
            or eff in (depth.get("values") or [])
            or eff in (depth.get("aliases") or {})
        ):
            body["reasoning_effort"] = eff
        elif depth.get("values"):
            raise ValueError(
                f"model {model_row['id']!r} does not offer reasoning_effort={eff!r}; "
                f"valid depth values: {depth.get('values')}"
            )

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
            off_words = depth.get("off") or []
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
                        "System prompt source: 'auto' picks repo 't2i + aspect_ratio' (no pictures) or "
                        "'edit + aspect_ratio' (with pictures); 'heylook preset' and "
                        "'heylook preset + aspect_ratio' pull the preset's system_prompt from "
                        "GET /v1/presets on the heylook server; the named t2i*/edit* modes use "
                        "docs/wiki/system_prompts/h3_image/*.md. Selecting a mode or preset in the UI "
                        "populates the editable system_prompt box below."
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
                    tooltip=(
                        "Optional preset name or id from GET /v1/presets on the heylook server. "
                        "Selecting a preset in the UI populates system_prompt and sampler fields below "
                        "for manual adjustment."
                    ),
                ),
                io.String.Input(
                    "custom_system_prompt",
                    multiline=True,
                    default="",
                    tooltip=(
                        "Editable system prompt sent as top-level 'system' on POST /v1/messages when "
                        "non-empty (auto-populated in the UI when you pick a mode or heylook preset). "
                        "Leave blank to resolve dynamically from 'mode' / 'preset'."
                    ),
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
                # Appended after 'images' to preserve existing positional widget/socket slots (docs/comfy_notes.md).
                io.Combo.Input(
                    "sampler_mode",
                    options=list(SAMPLER_MODES),
                    default="heylook / preset default",
                    tooltip=(
                        "'heylook / preset default' omits sampler overrides so heylook's publisher/model "
                        "cascade (or 'preset') decides. 'custom' sends the editable sampler fields below "
                        "(auto-populated in the UI from the selected model/preset)."
                    ),
                ),
                io.Float.Input(
                    "temperature",
                    default=0.7,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    tooltip="Sampling temperature sent when sampler_mode is 'custom' (0.0 .. 2.0).",
                ),
                io.Float.Input(
                    "top_p",
                    default=0.95,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Nucleus sampling cutoff sent when sampler_mode is 'custom' (0.0 .. 1.0).",
                ),
                io.Int.Input(
                    "top_k",
                    default=20,
                    min=0,
                    max=1000,
                    step=1,
                    tooltip="Top-k cutoff sent when sampler_mode is 'custom' (0 = disabled in heylook).",
                ),
                io.Float.Input(
                    "min_p",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Min-p cutoff sent when sampler_mode is 'custom' (0.0 .. 1.0).",
                ),
                io.Int.Input(
                    "max_tokens",
                    default=4096,
                    min=1,
                    max=131072,
                    step=64,
                    tooltip="Maximum output tokens (thinking + caption) sent when sampler_mode is 'custom'.",
                ),
                io.Float.Input(
                    "repetition_penalty",
                    default=1.0,
                    min=0.1,
                    max=2.0,
                    step=0.01,
                    tooltip="Repetition penalty sent when sampler_mode is 'custom' (1.0 = neutral).",
                ),
                io.Float.Input(
                    "presence_penalty",
                    default=0.0,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip="Presence penalty sent when sampler_mode is 'custom' (0.0 .. 2.0).",
                ),
                io.String.Input(
                    "reasoning_effort",
                    default="",
                    tooltip=(
                        "Optional model-specific reasoning depth word (from engine.thinking.depth.values) "
                        "sent when sampler_mode is 'custom'. Leave blank for template default."
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
        sampler_mode: str = "heylook / preset default",
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        max_tokens: int = 4096,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        reasoning_effort: str = "",
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

        preset_name = (preset or "").strip()
        has_images = len(raw_images) > 0
        uses_server_preset = mode in ("heylook preset", "heylook preset + aspect_ratio")
        resolved_mode, sys_prompt = resolve_expander_system_prompt(
            mode,
            has_images=has_images,
            custom_system_prompt=custom_system_prompt,
            has_preset=bool(preset_name) or uses_server_preset,
        )
        prepared_images = [
            prepare_heylook_tensor(t, index=i) for i, t in enumerate(raw_images, start=1)
        ]
        base = normalize_heylook_base_url(base_url)
        need_caps = frozenset({"chat", "vision"} if has_images else {"chat"})

        with httpx.Client(trust_env=False, timeout=httpx.Timeout(None, connect=10.0)) as client:
            model_row = discover_heylook_model(client, base, model, need=need_caps)
            preset_fields = (
                expand_heylook_preset(client, base, preset_name, model_row)
                if (preset_name or (uses_server_preset and not sys_prompt))
                else None
            )
            if not sys_prompt and preset_fields:
                preset_sys = (preset_fields.get("system") or "").strip()
                if uses_server_preset and not preset_sys:
                    raise ValueError(
                        f"mode is {mode!r}, but the resolved heylook preset has an empty system_prompt"
                    )
                if resolved_mode == "heylook preset + aspect_ratio" and preset_sys:
                    sys_prompt = append_aspect_ratio_footer(preset_sys, has_images=has_images)
                else:
                    sys_prompt = preset_sys

            sampler_overrides = (
                {
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "top_k": int(top_k),
                    "min_p": float(min_p),
                    "max_tokens": int(max_tokens),
                    "repetition_penalty": float(repetition_penalty),
                    "presence_penalty": float(presence_penalty),
                }
                if sampler_mode == "custom"
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
                sampler_mode=sampler_mode,
                sampler_overrides=sampler_overrides,
                reasoning_effort_override=reasoning_effort,
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


def build_expander_catalog(base_url: str = "") -> dict:
    """Return local system prompts plus best-effort live `heylook` models & presets for UI autofill."""
    local_prompts = {name: _read_local_prompt(name) for name in EXPANDER_PROMPT_FILES}
    heylook_info: dict = {
        "base_url": "",
        "reachable": False,
        "error": None,
        "resident_models": [],
        "models": [],
        "presets": [],
    }
    raw_url = (base_url or "").strip()
    if raw_url:
        try:
            base = normalize_heylook_base_url(raw_url)
            heylook_info["base_url"] = base
            with httpx.Client(trust_env=False, timeout=httpx.Timeout(3.0, connect=2.0)) as client:
                mr = client.get(f"{base}/v1/models", headers={"Accept": "application/json"})
                if mr.status_code >= 400:
                    raise _http_error(mr, base, "/v1/models")
                rows = [
                    r for r in (mr.json().get("data") or [])
                    if isinstance(r, dict) and r.get("id") and "chat" in (r.get("capabilities") or [])
                ]
                heylook_info["models"] = [
                    {
                        "id": r["id"],
                        "provider": r.get("provider") or "",
                        "capabilities": list(r.get("capabilities") or []),
                        "thinking_default": r.get("thinking_default"),
                        "sampler_defaults": dict(r.get("sampler_defaults") or {}),
                        "depth_values": list(
                            (((r.get("engine") or {}).get("thinking") or {}).get("depth") or {}).get("values")
                            or []
                        ),
                    }
                    for r in rows
                ]
                try:
                    sr = client.get(f"{base}/v1/system/metrics", headers={"Accept": "application/json"})
                    if sr.status_code == 200:
                        mmap = sr.json().get("models")
                        if isinstance(mmap, dict):
                            heylook_info["resident_models"] = list(mmap.keys())
                except Exception:
                    pass
                try:
                    pr = client.get(f"{base}/v1/presets", headers={"Accept": "application/json"})
                    if pr.status_code == 200:
                        heylook_info["presets"] = [
                            {
                                "id": p.get("id") or "",
                                "name": p.get("name") or p.get("id") or "",
                                "system_prompt": p.get("system_prompt") or "",
                                "params": dict(p.get("params") or {}),
                            }
                            for p in (pr.json().get("presets") or [])
                            if isinstance(p, dict) and (p.get("name") or p.get("id"))
                        ]
                except Exception:
                    pass
                heylook_info["reachable"] = True
        except Exception as exc:
            heylook_info["error"] = f"{type(exc).__name__}: {exc}"

    return {
        "local_prompts": local_prompts,
        "aspect_ratio_footers": {
            "t2i": ASPECT_RATIO_FOOTER_T2I,
            "edit": ASPECT_RATIO_FOOTER_EDIT,
        },
        "heylook": heylook_info,
    }


def register_expander_routes() -> None:
    """Register `GET /h3_image/expander_catalog` on ComfyUI's `PromptServer` when running inside ComfyUI."""
    try:
        import asyncio
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return
    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return

    async def expander_catalog(request: web.Request) -> web.Response:
        base_url = request.query.get("base_url", "")
        payload = await asyncio.to_thread(build_expander_catalog, base_url)
        return web.json_response(payload)

    instance.routes.get("/h3_image/expander_catalog")(expander_catalog)


def downscale_h3_image_output(images, target_width: int, target_height: int):
    """Resize decoded `[B, H, W, C]` images in `[0, 1]` to `(target_width, target_height)` via 8-bit PIL Lanczos.

    Matches `coderef/sglang` (`minimax_h3_resize_image_output`) and `coderef/vllm-omni`
    (`pipeline_minimax_h3_image.py`): when `(W, H) == (target_width, target_height)` or
    either target dimension is `<= 0`, returns `images` untouched (zero-copy passthrough
    at `>= 4.0` MP). Otherwise quantizes to `uint8` (`(x * 255).clamp(0, 255).to(uint8)`),
    resizes with `Image.Resampling.LANCZOS`, and returns `uint8 / 255.0`.
    """
    import torch

    tw, th = int(target_width or 0), int(target_height or 0)
    if images is None or tw <= 0 or th <= 0:
        return images
    cur_h, cur_w = int(images.shape[1]), int(images.shape[2])
    if (cur_w, cur_h) == (tw, th):
        return images
    channels = int(images.shape[3])
    if channels not in (3, 4):
        raise ValueError(f"expected [B, H, W, 3|4] image tensor, got {tuple(images.shape)}")
    mode = "RGBA" if channels == 4 else "RGB"
    out_frames = []
    for frame in images.detach().cpu():
        arr = (frame * 255.0).clamp(0.0, 255.0).to(torch.uint8).numpy()
        pil = Image.fromarray(arr, mode=mode).resize((tw, th), Image.Resampling.LANCZOS)
        out_frames.append(torch.from_numpy(np.array(pil, copy=True)))
    return torch.stack(out_frames, dim=0).to(dtype=images.dtype, device=images.device).div_(255.0)


class MiniMaxH3ImageDownscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ImageDownscale",
            display_name="MiniMax H3 Image Downscale",
            category="model/conditioning/minimax",
            description=(
                "Post-decode 8-bit PIL Lanczos resize for sub-4MP MiniMax-H3-Image requests "
                "(matching SGLang and vLLM-Omni). Zero-copy passthrough whenever target_width "
                "and target_height match the decoded canvas (all >= 4.0 MP renders) or are 0."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Int.Input(
                    "target_width", default=0, min=0, max=16384, step=1,
                    tooltip="Target output width from MiniMax H3 Image Size (0 or matching width passes through untouched).",
                ),
                io.Int.Input(
                    "target_height", default=0, min=0, max=16384, step=1,
                    tooltip="Target output height from MiniMax H3 Image Size (0 or matching height passes through untouched).",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
            ],
        )

    @classmethod
    def execute(cls, images, target_width=0, target_height=0) -> io.NodeOutput:
        return io.NodeOutput(downscale_h3_image_output(images, target_width, target_height))
