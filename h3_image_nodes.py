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

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import httpx
import numpy as np
from PIL import Image

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
    H3_IMAGE_DELIVER_MEGAPIXELS,
    H3_IMAGE_MEGAPIXELS,
    H3_IMAGE_MEGAPIXELS_RANGE,
    H3_IMAGE_MIN_GENERATION_MEGAPIXELS,
    H3_IMAGE_SIZE_MULTIPLE,
    h3_image_size,
)

try:
    from .heylook_client import (  # noqa: E402
        HEYLOOK_IMAGE_MAX_EDGE,
        HEYLOOK_JPEG_QUALITY,
        SAMPLER_MODES,
        THINKING_MODES,
        HeylookCancelToken,
        HeylookOverloadedError,
        HeylookStreamResult,
        _comfy_interrupt_check,
        _extract_inline_thinking,
        _fact_value,
        _http_error,
        _iter_sse,
        build_heylook_request_body,
        cancel_heylook_request,
        compose_preset_system_prompt,
        discover_heylook_model,
        expand_heylook_preset,
        format_heylook_report,
        make_heylook_client,
        normalize_heylook_base_url,
        parse_extend_directive,
        prepare_heylook_image,
        prepare_heylook_tensor,
        probe_heylook_server_info,
        resolve_heylook_base_url,
        stream_heylook_message,
        with_heylook_backoff,
    )
    from .auto_edit_prompt import (  # noqa: E402
        format_concept_brief_block,
        list_concept_options,
        load_concept_brief,
    )
except ImportError:  # run from the repo root by a check
    from heylook_client import (  # noqa: E402
        HEYLOOK_IMAGE_MAX_EDGE,
        HEYLOOK_JPEG_QUALITY,
        SAMPLER_MODES,
        THINKING_MODES,
        HeylookCancelToken,
        HeylookOverloadedError,
        HeylookStreamResult,
        _comfy_interrupt_check,
        _extract_inline_thinking,
        _fact_value,
        _http_error,
        _iter_sse,
        build_heylook_request_body,
        cancel_heylook_request,
        compose_preset_system_prompt,
        discover_heylook_model,
        expand_heylook_preset,
        format_heylook_report,
        make_heylook_client,
        normalize_heylook_base_url,
        parse_extend_directive,
        prepare_heylook_image,
        prepare_heylook_tensor,
        probe_heylook_server_info,
        resolve_heylook_base_url,
        stream_heylook_message,
        with_heylook_backoff,
    )
    from auto_edit_prompt import (  # noqa: E402
        format_concept_brief_block,
        list_concept_options,
        load_concept_brief,
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
    generate_megapixels: float | None = None,
) -> tuple[int, int, int, int]:
    """`(canvas_width, canvas_height, target_width, target_height)`.

    The target is `megapixels` at the shape. The canvas the DiT generates on is the largest of `megapixels`,
    `generate_megapixels` (when given) and, with `enforce_4mp_floor`, 4.0 MP, at the target's shape and on the
    32 px grid; `MiniMaxH3ImageDownscale` shrinks the decode from canvas to target. The canvas is never smaller
    than the target, so nothing is upscaled. `generate_megapixels=None` keeps the behaviour from before it existed.

    Matches `coderef/sglang` (`minimax_h3_image_generation_canvas`) and
    `coderef/vllm-omni` (`pipeline_minimax_h3_image.py`): when `megapixels < 4.0`
    and `enforce_4mp_floor` is enabled, the DiT generates on a 4.0 MP (`2048x2048`
    area) canvas of the same aspect ratio and `MiniMaxH3ImageDownscale` resizes
    the decoded image to `(target_width, target_height)` via 8-bit PIL Lanczos.
    Without `generate_megapixels`, at `megapixels >= 4.0` canvas and target dimensions are identical.
    """
    target_w, target_h = resolve_size(megapixels, shape, picture_size)
    sizes = [float(megapixels)]
    if enforce_4mp_floor:
        sizes.append(H3_IMAGE_MIN_GENERATION_MEGAPIXELS)
    if generate_megapixels is not None:
        sizes.append(float(generate_megapixels))
    canvas_mp = max(sizes)
    if canvas_mp > megapixels:
        min_area = canvas_mp * 1024 * 1024
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
                "Generation canvas and delivered size for MiniMax-H3-Image and P-H3-Image, from a shape and "
                "megapixels. The DiT generates at generate_megapixels (MiniMax-H3-Image is best near 16, "
                "P-H3-Image near 8) and you get back megapixels: wire width and height into the conditioning "
                "node, and target_width / target_height into MiniMax H3 Image Downscale, which shrinks the "
                "decode with Lanczos. The canvas is never below megapixels."
            ),
            inputs=[
                io.Float.Input(
                    "megapixels", default=H3_IMAGE_DELIVER_MEGAPIXELS, min=0.25, max=24.0, step=0.25,
                    tooltip=(
                        "Size of the picture you get back, in megapixels (1024x1024 units). The picture is "
                        "generated at generate_megapixels and shrunk to this with Lanczos; set it as high as "
                        "generate_megapixels to keep the full size.")),
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
                io.Float.Input(
                    "generate_megapixels", default=H3_IMAGE_MEGAPIXELS, min=0.25, max=24.0, step=0.25,
                    optional=True,
                    tooltip=(
                        f"Size the model generates at, in megapixels. MiniMax-H3-Image works from {low:g} to "
                        f"{high:g} and is best near {H3_IMAGE_MEGAPIXELS:g}; P-H3-Image works best at 4 to 8. "
                        "Generating big and shrinking to megapixels keeps detail and fewer mistakes in small "
                        "text. Never below megapixels.")),
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
        generate_megapixels=H3_IMAGE_MEGAPIXELS,
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
            generate_megapixels=generate_megapixels,
        )
        return io.NodeOutput(width, height, target_width, target_height)


MAX_H3_IMAGE_REFS = 16
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


def expand_image_prompt(
    *,
    prompt: str = "",
    enabled: bool = True,
    base_url: str = "",
    mode: str = "auto",
    model: str = "",
    thinking: str = "model default",
    seed: int = 0,
    preset: str = "",
    custom_system_prompt: str = "",
    raw_images: Sequence[Any] | None = None,
    prepared_images: list[tuple[str, str]] | None = None,
    sampler_mode: str = "heylook / preset default",
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 20,
    min_p: float = 0.0,
    max_tokens: int = 4096,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    reasoning_effort: str = "",
    client: httpx.Client | None = None,
    interrupt_check: Callable[[], None] | None = None,
    cancel_token: HeylookCancelToken | None = None,
    prefer_preset_system: bool = True,
    custom_sampler_overrides: dict[str, Any] | None = None,
    fallback_system_prompt: str = "",
    resolved_mode_override: str = "",
    allow_plain_fallback: bool = False,
    concept: str = "none",
) -> dict[str, Any]:
    """Expand a short T2I or 1..16-image edit request via local/LAN `heylook` and `parse_expander_reply`."""
    if not enabled:
        raw = (prompt or "").strip()
        if not raw:
            return {
                "prompt": "",
                "shape": "",
                "thinking": "",
                "report": "expander disabled (empty prompt)",
                "resolved_mode": "",
                "model": "",
            }
        inline_think = _extract_inline_thinking(raw)
        clean_prompt, ratio = parse_expander_reply(raw)
        return {
            "prompt": clean_prompt,
            "shape": ratio or "",
            "thinking": inline_think,
            "report": "expander disabled (passthrough)",
            "resolved_mode": "passthrough",
            "model": "",
        }

    user_text = (prompt or "").strip()
    concept_name = (concept or "none").strip() or "none"
    concept_brief = None if concept_name.lower() == "none" else load_concept_brief(concept_name)
    if not user_text and concept_brief is None:
        raise ValueError("MiniMaxH3ImagePromptExpander: enter a 'prompt' or choose a 'concept'")

    if prepared_images is not None:
        if len(prepared_images) > MAX_H3_IMAGE_REFS:
            raise ValueError(
                f"MiniMaxH3ImagePromptExpander takes at most {MAX_H3_IMAGE_REFS} images, got {len(prepared_images)}"
            )
        prepared = list(prepared_images)
    else:
        img_list = [img for img in (raw_images or ()) if img is not None]
        if len(img_list) > MAX_H3_IMAGE_REFS:
            raise ValueError(
                f"MiniMaxH3ImagePromptExpander takes at most {MAX_H3_IMAGE_REFS} images, got {len(img_list)}"
            )
        prepared = [prepare_heylook_tensor(t, index=i) for i, t in enumerate(img_list, start=1)]

    preset_name = (preset or "").strip()
    has_images = len(prepared) > 0
    if concept_brief is not None:
        concept_block = format_concept_brief_block(concept_brief, mode="image")
        if user_text:
            eff_user_text = f"{concept_block}\n\nUser image request:\n{user_text}"
        elif has_images:
            eff_user_text = (
                f"{concept_block}\n\nUser image request:\n"
                "Apply this concept cleanly to the attached <Picture N> reference image(s)."
            )
        else:
            eff_user_text = (
                f"{concept_block}\n\nUser image request:\n"
                "Create a vivid single-frame visual scene embodying this concept."
            )
    else:
        eff_user_text = user_text
    uses_server_preset = mode in ("heylook preset", "heylook preset + aspect_ratio")
    custom_clean = (custom_system_prompt or "").strip()
    fallback_clean = (fallback_system_prompt or "").strip()
    if fallback_clean and not custom_clean and not uses_server_preset:
        resolved_mode = (resolved_mode_override or mode or "auto").strip()
        sys_prompt = fallback_clean
    else:
        resolved_mode, sys_prompt = resolve_expander_system_prompt(
            mode,
            has_images=has_images,
            custom_system_prompt=custom_system_prompt,
            has_preset=bool(preset_name) or uses_server_preset,
        )
    base = resolve_heylook_base_url(base_url)
    need_caps = frozenset({"chat", "vision"} if has_images else {"chat"})
    int_cb = interrupt_check if interrupt_check is not None else _comfy_interrupt_check

    def _run_with_client(http_client: httpx.Client) -> tuple[HeylookStreamResult, dict, str, str, bool]:
        model_row = discover_heylook_model(http_client, base, model, need=need_caps)
        preset_fields = (
            expand_heylook_preset(http_client, base, preset_name, model_row)
            if (preset_name or (uses_server_preset and not sys_prompt))
            else None
        )
        int_cb()
        if cancel_token is not None:
            cancel_token.check()
        eff_sys_prompt = sys_prompt
        eff_resolved_mode = resolved_mode
        sys_source = "custom" if custom_clean else "local"
        preset_won = False
        active_preset_fields = preset_fields
        preset_sys = str((preset_fields or {}).get("system") or "").strip()
        extend_mode, _extend_body = parse_extend_directive(preset_sys) if preset_sys else (None, "")
        custom_extend_mode, _custom_extend_body = parse_extend_directive(custom_clean) if custom_clean else (None, "")
        if custom_clean and custom_extend_mode is not None:
            local_mode = (
                mode
                if mode in ("t2i", "edit", "t2i + aspect_ratio", "edit + aspect_ratio")
                else "auto"
            )
            _, local_base = resolve_expander_system_prompt(
                local_mode,
                has_images=has_images,
                custom_system_prompt="",
                has_preset=False,
            )
            if not local_base:
                local_base = fallback_clean
            eff_sys_prompt, sys_source, _ = compose_preset_system_prompt(
                local_base,
                custom_clean,
                preset_name=preset_name or "custom",
            )
            if resolved_mode == "heylook preset + aspect_ratio" and "aspect_ratio:" not in eff_sys_prompt:
                eff_sys_prompt = append_aspect_ratio_footer(eff_sys_prompt, has_images=has_images)
            if preset_fields:
                active_preset_fields = {k: v for k, v in preset_fields.items() if k != "system"}
        elif not custom_clean and preset_fields and preset_sys and extend_mode is not None:
            local_base = eff_sys_prompt
            if not local_base:
                _, local_base = resolve_expander_system_prompt(
                    "auto",
                    has_images=has_images,
                    custom_system_prompt="",
                    has_preset=False,
                )
            eff_sys_prompt, sys_source, _ = compose_preset_system_prompt(
                local_base,
                preset_sys,
                preset_name=preset_name,
            )
            if resolved_mode == "heylook preset + aspect_ratio" and "aspect_ratio:" not in eff_sys_prompt:
                eff_sys_prompt = append_aspect_ratio_footer(eff_sys_prompt, has_images=has_images)
            active_preset_fields = {k: v for k, v in preset_fields.items() if k != "system"}
        elif not eff_sys_prompt and preset_fields:
            if uses_server_preset and not preset_sys:
                raise ValueError(
                    f"mode is {mode!r}, but the resolved heylook preset has an empty system_prompt"
                )
            if resolved_mode == "heylook preset + aspect_ratio" and preset_sys:
                eff_sys_prompt = append_aspect_ratio_footer(preset_sys, has_images=has_images)
            else:
                eff_sys_prompt = preset_sys
            if preset_sys:
                sys_source = f"preset:{preset_name}" if preset_name else "preset"
                preset_won = True
        elif prefer_preset_system and not custom_clean and preset_fields:
            if preset_sys:
                eff_sys_prompt = ""
                sys_source = f"preset:{preset_name}" if preset_name else "preset"
                preset_won = True
                if mode == "auto":
                    eff_resolved_mode = f"preset:{preset_name}" if preset_name else "preset"

        sampler_overrides = (
            dict(custom_sampler_overrides)
            if custom_sampler_overrides is not None
            else (
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
        )
        body = build_heylook_request_body(
            model_row=model_row,
            system_prompt=eff_sys_prompt,
            user_prompt=eff_user_text,
            prepared_images=prepared,
            thinking=thinking,
            request_seed=int(seed) & 0x7FFFFFFF,
            preset_fields=active_preset_fields,
            sampler_mode=sampler_mode,
            sampler_overrides=sampler_overrides,
            reasoning_effort_override=reasoning_effort,
            default_reasoning_effort=reasoning_effort,
        )
        res = with_heylook_backoff(
            lambda: stream_heylook_message(
                base, body, client=http_client, interrupt_check=int_cb, cancel_token=cancel_token
            ),
            interrupt_check=int_cb,
            cancel_token=cancel_token,
        )
        return res, model_row, eff_resolved_mode, sys_source, preset_won

    try:
        if client is not None:
            res, model_row, eff_resolved_mode, sys_source, preset_won = _run_with_client(client)
        else:
            with make_heylook_client(base) as own_client:
                attached_own = cancel_token.attach_client(own_client) if cancel_token is not None else False
                try:
                    res, model_row, eff_resolved_mode, sys_source, preset_won = _run_with_client(own_client)
                finally:
                    if cancel_token is not None and attached_own:
                        cancel_token.detach_client(own_client)
    except BaseException:
        if cancel_token is not None and cancel_token.is_cancelled:
            cancel_token.check()
        raise

    think_trace = res.thinking or _extract_inline_thinking(res.text)
    ratio = ""
    if allow_plain_fallback or preset_won:
        try:
            clean_prompt, ratio = parse_expander_reply(res.text)
        except ValueError as exc:
            raw_plain = (res.text or "").strip()
            while raw_plain.startswith("<think>"):
                end_tk = raw_plain.find("</think>")
                if end_tk < 0:
                    raw_plain = ""
                    break
                raw_plain = raw_plain[end_tk + len("</think>") :].strip()
            if "</think>" in raw_plain and "<think>" not in raw_plain.split("</think>", 1)[0]:
                raw_plain = raw_plain.split("</think>", 1)[1].strip()
            if "<think>" in raw_plain:
                raw_plain = ""
            if raw_plain.startswith("```") and raw_plain.endswith("```"):
                first_nl = raw_plain.find("\n")
                if first_nl > 0:
                    inner = raw_plain[first_nl + 1 : -3].strip()
                    if "```" not in inner:
                        raw_plain = inner
            clean_prompt = raw_plain
            if not clean_prompt:
                if res.stop_reason == "max_tokens":
                    raise RuntimeError(
                        f"heylook stopped at max_tokens before producing a prompt ({exc}). "
                        "Raise max_tokens in your heylook model/preset config or set thinking='off'."
                    ) from exc
                raise ValueError(f"heylook returned no usable prompt ({exc})") from exc
    else:
        try:
            clean_prompt, ratio = parse_expander_reply(res.text)
        except ValueError as exc:
            if res.stop_reason == "max_tokens":
                raise RuntimeError(
                    f"heylook stopped at max_tokens before completing a valid caption ({exc}). "
                    "Raise max_tokens in your heylook model/preset config or set thinking='off'."
                ) from exc
            raise

    report_str = format_heylook_report(
        res,
        resolved_mode=eff_resolved_mode,
        shape=ratio,
        system_source=sys_source,
        concept=concept_brief.stem if concept_brief is not None else None,
    )
    return {
        "prompt": clean_prompt,
        "shape": ratio or "",
        "thinking": think_trace,
        "report": report_str,
        "resolved_mode": eff_resolved_mode,
        "model": res.model or model_row.get("id", ""),
    }


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
                    default="",
                    tooltip=(
                        "Optional heylookitsanllm server URL override. Leave blank to read H3_HEYLOOK_URL "
                        "from your environment or gitignored h3.env, falling back to http://127.0.0.1:8080 "
                        "(or http://<lan-or-tailscale-ip>:<port> on another machine)."
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
                # Appended after 'images' to preserve existing positional widget/socket slots (docs/wiki/ops/comfy_notes.md).
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
                # Appended 2026-11-21: optional concept preset dropdown (positional widget compatibility).
                io.Combo.Input(
                    "concept",
                    options=list_concept_options(),
                    default="none",
                    optional=True,
                    tooltip=(
                        "Optional preset card (.toml under data/edit_concepts/, data/control_vectors/, "
                        "or prompt_bank/control_vectors/) woven into the still-image request. Leave 'none' "
                        "to expand purely from 'prompt'."
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
        base_url: str = "",
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
        concept: str = "none",
    ) -> io.NodeOutput:
        names = sorted(images or {}, key=lambda name: int(name.rsplit("_", 1)[-1]))
        raw_images = [images[name] for name in names if images[name] is not None]
        res = expand_image_prompt(
            prompt=prompt,
            enabled=enabled,
            base_url=base_url,
            mode=mode,
            model=model,
            thinking=thinking,
            seed=seed,
            preset=preset,
            custom_system_prompt=custom_system_prompt,
            raw_images=raw_images,
            sampler_mode=sampler_mode,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            reasoning_effort=reasoning_effort,
            concept=concept,
        )
        return io.NodeOutput(res["prompt"], res["shape"], res["thinking"], res["report"])


def build_expander_catalog(base_url: str = "") -> dict:
    """Return local system prompts plus best-effort live `heylook` models & presets for UI autofill."""
    local_prompts = {name: _read_local_prompt(name) for name in EXPANDER_PROMPT_FILES}
    heylook_info = probe_heylook_server_info(base_url, fallback_env=True)
    return {
        "local_prompts": local_prompts,
        "concepts": list_concept_options(),
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


def delivered_size(
    decoded_w: int,
    decoded_h: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int]:
    """Resolve the delivered `(width, height)` for a decoded image without cropping or upscaling.

    - If `(target_width, target_height)` matches `(decoded_w, decoded_h)` in aspect within the 32-px grid's
      half-step tolerance (`±16 px` on scaled axes: ``abs(dw * th - dh * tw) * 2 <= 32 * (tw + th)``),
      use `(target_width, target_height)`.
    - Otherwise, keep the decoded image's own shape at the target's pixel count on the 32-px grid via
      `h3_image_size` (so `P-H3-Image` `canvas="auto"` downscales without stretching).
    - In both branches, never enlarge either side: if `out_w > dw` or `out_h > dh`, return `(dw, dh)`.
    """
    dw, dh = int(decoded_w), int(decoded_h)
    tw, th = int(target_width), int(target_height)
    if min(dw, dh) < 1:
        raise ValueError(f"decoded_w and decoded_h must be >= 1; got ({dw}, {dh})")
    if min(tw, th) < 1:
        raise ValueError(f"target_width and target_height must be >= 1; got ({tw}, {th})")
    # Exact integer equivalent of ∃ scale > 0 with |dw - scale*tw| <= M/2 and |dh - scale*th| <= M/2.
    if abs(dw * th - dh * tw) * 2 <= H3_IMAGE_SIZE_MULTIPLE * (tw + th):
        out_w, out_h = tw, th
    else:
        out_w, out_h = h3_image_size((tw * th) / (1024 * 1024), dw, dh)
    if out_w > dw or out_h > dh:
        return dw, dh
    return out_w, out_h


def downscale_h3_image_output(images, target_width: int, target_height: int, enabled: bool = True):
    """Resize decoded `[B, H, W, C]` images in `[0, 1]` toward `(target_width, target_height)` via 8-bit PIL Lanczos.

    Resolves the output size with `delivered_size` so aspect ratios differing beyond `±16 px` keep the decoded
    shape at the target pixel count and neither side is ever enlarged. When `enabled` is False or the resolved
    output size equals `(W, H)`, returns `images` untouched (zero-copy passthrough). Otherwise quantizes to `uint8`
    (`(x * 255).clamp(0, 255).to(uint8)`), resizes with `Image.Resampling.LANCZOS`, and returns `uint8 / 255.0`.
    """
    import torch

    if not enabled or images is None:
        return images
    cur_h, cur_w = int(images.shape[1]), int(images.shape[2])
    out_w, out_h = delivered_size(cur_w, cur_h, target_width, target_height)
    if (cur_w, cur_h) == (out_w, out_h):
        return images
    channels = int(images.shape[3])
    if channels not in (3, 4):
        raise ValueError(f"expected [B, H, W, 3|4] image tensor, got {tuple(images.shape)}")
    mode = "RGBA" if channels == 4 else "RGB"
    out_frames = []
    for frame in images.detach().cpu():
        arr = (frame * 255.0).clamp(0.0, 255.0).to(torch.uint8).numpy()
        pil = Image.fromarray(arr, mode=mode).resize((out_w, out_h), Image.Resampling.LANCZOS)
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
                "Post-decode 8-bit PIL Lanczos downscale toward MiniMax H3 Image Size's target_width / "
                "target_height. Preserves the decoded image's aspect ratio (at the target pixel count) when "
                "shapes differ beyond ±16 px (such as P-H3-Image canvas='auto'), and never enlarges either side "
                "(zero-copy passthrough when no shrink is needed or enabled is False)."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Int.Input(
                    "target_width", default=2048, min=1, max=16384, step=1,
                    tooltip=(
                        "Target delivered width from MiniMax H3 Image Size; used directly when the decoded "
                        "aspect matches within ±16 px, else supplies target pixel budget. Never enlarges."
                    ),
                ),
                io.Int.Input(
                    "target_height", default=2048, min=1, max=16384, step=1,
                    tooltip=(
                        "Target delivered height from MiniMax H3 Image Size; used directly when the decoded "
                        "aspect matches within ±16 px, else supplies target pixel budget. Never enlarges."
                    ),
                ),
                io.Boolean.Input(
                    "enabled", default=True, optional=True,
                    tooltip="When False, passes decoded images through untouched without resizing.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
            ],
        )

    @classmethod
    def execute(cls, images, target_width=2048, target_height=2048, enabled=True) -> io.NodeOutput:
        return io.NodeOutput(downscale_h3_image_output(images, target_width, target_height, enabled=bool(enabled)))
