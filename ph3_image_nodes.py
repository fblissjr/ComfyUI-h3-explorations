"""MiniMax H3 p-h3-image prompt upsampler node (`MiniMaxH3PH3ImagePromptUpsampler`).

Wraps `PrunaAI/P-H3-Image`'s fine-tuned `Qwen3.5-9B` prompt upsampler
(`p-h3-image_prompt_upsampler_int8_convrot.safetensors`, loaded by core's
`CLIPLoader` with `type="qwen_image"`) for both text-to-image and multi-image
editing (1 to 8 images), with a bypass toggle (`enabled=False` by default) and
lazy `clip` evaluation so a workflow can ship with pre-authored prompts without
loading the 9.1 GB upsampler checkpoint unless the user turns `enabled` on.

Why a dedicated node rather than the stock `ChatTemplate` + `TextGenerateLTX2Prompt`
or `TextEncodeQwenImageEditPlus` chain in `PrunaAI_P-H3-Image-Comfy/workflows/*.json`:

1. `ChatTemplate` is not a core ComfyUI node, and core's `TextGenerateLTX2Prompt`
   (`comfy_extras/nodes_textgen.py`) wraps its input in Gemma 3/4 `<start_of_turn>`
   tags with LTX-2's video system prompt.
2. Core's `TextEncodeQwenImageEditPlus` (`comfy_extras/nodes_qwen.py`) accepts at
   most 3 images, downscales them to 384x384 (`147,456` px vs upstream's `1,000,000`
   px), injects Qwen-Image-Edit's system prompt, and outputs `CONDITIONING` rather
   than calling `clip.generate`.
3. `PrunaAI/P-H3-Image` (`diffusers` `modular_pipelines/p_h3_image/prompt_upsampler.py`
   and `prompt_upsampler_processor/chat_template.jinja`) formats a user-only turn
   with no system prompt and opens `<think>\\n` on the assistant turn:
   - T2I (`max_new_tokens=1024`):
     `<|im_start|>user\\nAspect ratio: {1:1|4:3|3:4|3:2|2:3|16:9|9:16}\\nRequest: {request}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n`
   - Edit (`max_new_tokens=1536`, 1..8 images resized at their own aspect ratio to
     ~1 MP at multiples of 32, Lanczos, no white padding):
     `<|im_start|>user\\n<Picture 1>: <|vision_start|><|image_pad|><|vision_end|>[<Picture 2>: ... ]Edit <Picture 1>. {request}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n`
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

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
    PH3_IMAGE_EDIT_MAX_NEW_TOKENS,
    PH3_IMAGE_T2I_MAX_NEW_TOKENS,
)

# Inherited from diffusers `src/diffusers/modular_pipelines/p_h3_image/prompt_upsampler.py`
# and `before_encoder.py` (PrunaAI/P-H3-Image, commit c55ec279).
ASPECT_RATIOS: dict[str, float] = {
    "1:1": 1.0,
    "4:3": 4.0 / 3.0,
    "3:4": 3.0 / 4.0,
    "3:2": 3.0 / 2.0,
    "2:3": 2.0 / 3.0,
    "16:9": 16.0 / 9.0,
    "9:16": 9.0 / 16.0,
}
UPSAMPLER_IMAGE_PIXELS = 1_000_000
CANVAS_MULTIPLE = 32
MAX_EDIT_IMAGES = 8
T2I_MAX_NEW_TOKENS = PH3_IMAGE_T2I_MAX_NEW_TOKENS
EDIT_MAX_NEW_TOKENS = PH3_IMAGE_EDIT_MAX_NEW_TOKENS
# Inherited from Qwen35Tokenizer.eos_token_id (`<|im_end|>`, comfy/text_encoders/qwen35.py:1299).
QWEN35_IM_END_ID = 248046
# Matches `PH3_IMAGE_EDIT_VERBATIM` in `comfy_extras/nodes_minimax_h3.py:28`.
_EDIT_VERBATIM_RE = re.compile(r"^\s*(Edit <Picture \d+>|subject_definitions:)")

EDIT_SECTIONS: tuple[str, ...] = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)
_SECTION_RE = re.compile(r"(?m)^(" + "|".join(EDIT_SECTIONS) + r"):[ \t]*$")
_RETENTION_RE = re.compile(
    r"^<(Picture|Subject) (\d+)>(?: \(([^)]*)\))?:\s*"
    r"(fully_preserved|partially_preserved|attribute_transfer|weak_reference)"
    r"\s*[-–—]\s*(.*)$"
)
_VISION_SLOT = "<|vision_start|><|image_pad|><|vision_end|>"


def aspect_ratio_label(w: int, h: int) -> str:
    """Closest aspect-ratio label in `ASPECT_RATIOS` to `w / h` (in log space)."""
    if w < 1 or h < 1:
        raise ValueError(f"width and height must be positive, got {w}x{h}")
    ratio = w / h
    return min(ASPECT_RATIOS, key=lambda label: abs(math.log(ratio / ASPECT_RATIOS[label])))


def size_at_area(
    w: int,
    h: int,
    pixels: int = UPSAMPLER_IMAGE_PIXELS,
    multiple: int = CANVAS_MULTIPLE,
) -> tuple[int, int]:
    """`(w, h)` resized to ~`pixels` at its own aspect ratio, sides multiples of `multiple`.

    Matches `diffusers` `p_h3_image.before_encoder._size_at_area`: an image whose
    sides are already multiples of `multiple` and whose area is within 20% of
    `pixels` keeps its size untouched.
    """
    if w < 1 or h < 1:
        raise ValueError(f"image dimensions must be positive, got {w}x{h}")
    if (
        w % multiple == 0
        and h % multiple == 0
        and 0.8 * pixels <= w * h <= 1.2 * pixels
    ):
        return w, h
    ratio = w / h
    return (
        max(multiple, round(math.sqrt(pixels * ratio) / multiple) * multiple),
        max(multiple, round(math.sqrt(pixels / ratio) / multiple) * multiple),
    )


def is_edit_prompt(prompt: str) -> bool:
    """Whether `prompt` conforms to one of the two legal P-H3-Image edit forms.

    Matches `diffusers` `p_h3_image.prompt_upsampler.is_edit_prompt`:
    1. A six-section `ref_en` brief (`EDIT_SECTIONS` in exact order) with
       `[keyframe completion]` appearing once and every non-empty line of
       `retention_analysis` matching `<Picture|Subject i>: <tag> - <clause>`.
    2. A single-paragraph instruct prompt starting with `"Edit <Picture 1>."`
       and containing `"Keep"`.
    """
    text = prompt.strip()
    names = _SECTION_RE.findall(text)
    if names:
        parts = _SECTION_RE.split(text)
        body = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
        retention = [
            line.strip()
            for line in body.get("retention_analysis", "").splitlines()
            if line.strip()
        ]
        return (
            tuple(names) == EDIT_SECTIONS
            and text.count("[keyframe completion]") == 1
            and bool(retention)
            and all(_RETENTION_RE.match(line) for line in retention)
        )
    return text.startswith("Edit <Picture 1>.") and "Keep" in text and "\n" not in text


def build_upsampler_chat(
    request: str,
    num_images: int = 0,
    aspect_ratio: str = "1:1",
) -> str:
    """Render the exact ChatML prompt `prompt_upsampler_processor/chat_template.jinja` produces.

    Starts with `<|im_start|>user\\n` so `Qwen35ImageTokenizer.tokenize_with_weights`
    (`comfy/text_encoders/qwen35.py:1250`) sets `skip_template = True` and replaces
    each `<|image_pad|>` token (248056) with the corresponding tensor in `images`.
    """
    cleaned = request.strip()
    if not cleaned:
        raise ValueError("MiniMaxH3PH3ImagePromptUpsampler: `prompt` must not be empty.")
    if num_images < 0 or num_images > MAX_EDIT_IMAGES:
        raise ValueError(
            f"MiniMaxH3PH3ImagePromptUpsampler supports 0 to {MAX_EDIT_IMAGES} images, got {num_images}."
        )
    if num_images > 0:
        prefix = "".join(f"<Picture {i}>: {_VISION_SLOT}" for i in range(1, num_images + 1))
        if _EDIT_VERBATIM_RE.match(cleaned):
            user_body = f"{prefix}{cleaned}"
        else:
            user_body = f"{prefix}Edit <Picture 1>. {cleaned}"
    else:
        if aspect_ratio not in ASPECT_RATIOS:
            raise ValueError(
                f"Unknown aspect_ratio {aspect_ratio!r}; expected one of {tuple(ASPECT_RATIOS)}."
            )
        user_body = f"Aspect ratio: {aspect_ratio}\nRequest: {cleaned}"
    return f"<|im_start|>user\n{user_body}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def _resize_upsampler_image(image):
    """Resize a `[B, H, W, C]` ComfyUI image tensor (first frame) to ~1 MP at multiples of 32 (Lanczos, RGB)."""
    import comfy.utils

    frame = image[:1, ..., :3]
    img_h, img_w = int(frame.shape[1]), int(frame.shape[2])
    target_w, target_h = size_at_area(img_w, img_h, UPSAMPLER_IMAGE_PIXELS, CANVAS_MULTIPLE)
    if (img_w, img_h) == (target_w, target_h):
        return frame
    samples = frame.movedim(-1, 1)
    resized = comfy.utils.common_upscale(samples, target_w, target_h, "lanczos", "disabled")
    return resized.movedim(1, -1)


class MiniMaxH3PH3ImagePromptUpsampler(io.ComfyNode):
    """Expand a T2I request or 1..8-image edit instruction with P-H3-Image's Qwen3.5-9B upsampler, or pass `prompt` through when `enabled` is False."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PH3ImagePromptUpsampler",
            display_name="MiniMax H3 p-h3-image Prompt Upsampler",
            category="model/conditioning/minimax",
            description=(
                "Expands a short text-to-image request or 1..8-image edit instruction into "
                "the full prompt PrunaAI/P-H3-Image expects, using its fine-tuned Qwen3.5-9B "
                "prompt upsampler (`p-h3-image_prompt_upsampler_int8_convrot.safetensors`). "
                "When `enabled` is False (the default), passes `prompt` straight through and "
                "never loads `clip`."
            ),
            inputs=[
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    tooltip=(
                        "The prompt or edit instruction. Passed straight through when "
                        "`enabled` is False; expanded by the Qwen3.5-9B prompt upsampler "
                        "when `enabled` is True."
                    ),
                ),
                io.Boolean.Input(
                    "enabled",
                    default=False,
                    tooltip=(
                        "Off by default so pre-authored prompts run without loading the "
                        "9.1 GB Qwen3.5-9B upsampler. Turn on to rewrite a short request "
                        "or edit instruction."
                    ),
                ),
                io.Int.Input(
                    "width",
                    default=2912,
                    min=CANVAS_MULTIPLE,
                    max=16384,
                    step=CANVAS_MULTIPLE,
                    tooltip=(
                        "Target canvas width in pixels, used to select the closest aspect "
                        "ratio bucket (1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16) for T2I "
                        "upsampling."
                    ),
                ),
                io.Int.Input(
                    "height",
                    default=2912,
                    min=CANVAS_MULTIPLE,
                    max=16384,
                    step=CANVAS_MULTIPLE,
                    tooltip=(
                        "Target canvas height in pixels, used to select the closest aspect "
                        "ratio bucket (1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16) for T2I "
                        "upsampling."
                    ),
                ),
                io.Int.Input(
                    "max_new_tokens",
                    default=T2I_MAX_NEW_TOKENS,
                    min=64,
                    max=4096,
                    step=64,
                    tooltip=(
                        "Maximum tokens generated by the upsampler (thinking + prompt). "
                        "Upstream uses 1024 for T2I and 1536 for image editing."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip="Seed passed to clip.generate.",
                ),
                io.Clip.Input(
                    "clip",
                    optional=True,
                    lazy=True,
                    tooltip=(
                        "The `p-h3-image_prompt_upsampler_int8_convrot.safetensors` CLIP "
                        "(loaded via CLIPLoader with type='qwen_image'). Lazy: not loaded "
                        "when `enabled` is False."
                    ),
                ),
                io.Autogrow.Input(
                    "images",
                    optional=True,
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[f"image_{i}" for i in range(1, MAX_EDIT_IMAGES + 1)],
                        min=0,
                    ),
                    tooltip=(
                        "Optional edit images (`image_1` is the image to edit, `image_2`..`image_8` "
                        "are references). Each is resized at its own aspect ratio to ~1 MP "
                        "(multiples of 32, Lanczos) with no white padding."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.String.Output(display_name="thinking"),
            ],
        )

    @classmethod
    def check_lazy_status(cls, enabled=False, **kwargs):
        if not enabled:
            return []
        return ["clip"] if "clip" in kwargs and kwargs["clip"] is None else []

    @classmethod
    def execute(
        cls,
        prompt: str,
        enabled: bool = False,
        width: int = 2912,
        height: int = 2912,
        max_new_tokens: int = T2I_MAX_NEW_TOKENS,
        seed: int = 0,
        clip=None,
        images: io.Autogrow.Type = None,
    ) -> io.NodeOutput:
        if not enabled:
            return io.NodeOutput(prompt, "")
        if clip is None:
            raise ValueError(
                "MiniMaxH3PH3ImagePromptUpsampler: `enabled` is True, but no `clip` "
                "(`p-h3-image_prompt_upsampler_int8_convrot.safetensors`, CLIPLoader "
                "type='qwen_image') is connected."
            )
        tok_cls = type(getattr(clip, "tokenizer", None)).__name__
        if "Minimax" in tok_cls or "MiniMax" in tok_cls or "PH3Image" in tok_cls:
            raise ValueError(
                f"MiniMaxH3PH3ImagePromptUpsampler: connected `clip` has tokenizer {tok_cls!r} "
                "(the MiniMax H3 text encoder). Connect `p-h3-image_prompt_upsampler_int8_convrot.safetensors` "
                "loaded via CLIPLoader with `type='qwen_image'` (node 24) instead."
            )

        names = sorted(images or {}, key=lambda name: int(name.rsplit("_", 1)[-1]))
        raw_images = [images[name] for name in names if images[name] is not None]
        if len(raw_images) > MAX_EDIT_IMAGES:
            raise ValueError(
                f"MiniMaxH3PH3ImagePromptUpsampler takes at most {MAX_EDIT_IMAGES} images, "
                f"got {len(raw_images)}."
            )
        vision_images = [_resize_upsampler_image(img) for img in raw_images]

        ar_label = "1:1" if vision_images else aspect_ratio_label(width, height)
        chat_text = build_upsampler_chat(
            prompt, num_images=len(vision_images), aspect_ratio=ar_label
        )
        budget = int(max_new_tokens)

        tokens = clip.tokenize(chat_text, images=vision_images, skip_template=True, min_length=1)
        generated_ids = clip.generate(
            tokens,
            do_sample=False,
            max_length=budget,
            temperature=1.0,
            top_k=50,
            top_p=1.0,
            min_p=0.0,
            repetition_penalty=1.0,
            presence_penalty=0.0,
            seed=seed,
            mtp=True,
        )
        if hasattr(generated_ids, "tolist"):
            flat_ids = generated_ids.tolist()
        else:
            flat_ids = list(generated_ids)
        if flat_ids and isinstance(flat_ids[0], (list, tuple)):
            flat_ids = list(flat_ids[0])
        ended = QWEN35_IM_END_ID in flat_ids
        if ended:
            flat_ids = flat_ids[: flat_ids.index(QWEN35_IM_END_ID)]
        decoded = clip.decode(flat_ids)
        for stop_tok in ("<|im_end|>", "<|endoftext|>"):
            if stop_tok in decoded:
                decoded = decoded.split(stop_tok, 1)[0]

        if not ended:
            raise RuntimeError(
                f"MiniMaxH3PH3ImagePromptUpsampler: generation hit max_new_tokens={budget} "
                f"without emitting `<|im_end|>` ({QWEN35_IM_END_ID}). Raise `max_new_tokens` "
                f"or set `enabled=False`:\n{decoded}"
            )
        if decoded.count("</think>") != 1:
            raise RuntimeError(
                f"MiniMaxH3PH3ImagePromptUpsampler: expected exactly 1 `</think>` tag within "
                f"max_new_tokens={budget} (got {decoded.count('</think>')}). Raise `max_new_tokens` "
                f"or set `enabled=False`:\n{decoded}"
            )
        thinking, _, rewritten = decoded.partition("</think>")
        thinking = thinking.replace("<think>", "", 1).strip()
        rewritten = rewritten.strip()
        if len(rewritten.split()) < 5:
            raise RuntimeError(
                f"MiniMaxH3PH3ImagePromptUpsampler: rewritten prompt is too short (< 5 words):\n{decoded}"
            )
        if vision_images and not is_edit_prompt(rewritten):
            raise RuntimeError(
                "MiniMaxH3PH3ImagePromptUpsampler: rewritten edit prompt does not match "
                "either legal P-H3-Image edit form (6-section `[keyframe completion]` brief "
                f"or single-paragraph `Edit <Picture 1>. ... Keep ...`):\n{rewritten}"
            )
        return io.NodeOutput(rewritten, thinking)
