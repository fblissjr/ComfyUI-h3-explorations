"""MiniMax H3 p-h3-image conditioning and prompt upsampler nodes
(`MiniMaxH3PH3ImageConditioning`, `MiniMaxH3PH3ImagePromptUpsampler`).

1. `MiniMaxH3PH3ImageConditioning`:
   Implements `PrunaAI/P-H3-Image`'s exact `diffusers` conditioning and resize
   contract (`coderef/diffusers/src/diffusers/modular_pipelines/p_h3_image/{before_encoder,encoders,before_denoise}.py`)
   and `MiniMax-H3`'s float16 conditioning-latent rounding (`coderef/MiniMax-H3/Ref2VA/video_vae/klvae.py:910`),
   closing four gaps in ComfyUI core's `PH3ImageConditioning` (`comfy_extras/nodes_minimax_h3.py:668-731`):
   - **Area-range clamping (`size_in_area_range`) vs. downscale-only (`_size_within_area`)**:
     Core's `_size_within_area` scales down only (`min(1.0, ...)`), leaving a small
     `image_1` (e.g. `1024x1024`) below the 4 MP (`4 * 1024**2`) canvas floor P-H3-Image
     was distilled for. `MiniMaxH3PH3ImageConditioning` clamps secondary references
     (`image_2..image_8`) to `[1024**2, 8 * 1024**2]` (`1..8 MP`), clamps `image_1`
     (and the `canvas='auto'` output canvas) to `[4 * 1024**2, 8 * 1024**2]`
     (`4..8 MP`), and resizes Qwen3-VL `vision_images` via `size_at_area` (`~1 MP`,
     both up and down, keeping native size when already a multiple of 32 within 20%).
   - **Decoupled explicit canvas (`canvas='explicit'`) vs. coupled reference area**:
     In core's `PH3ImageConditioning`, passing non-zero `width, height` sets
     `area = width * height` for VAE reference scaling as well as the output canvas.
     In `diffusers` (`before_encoder.py:80, 164-166`) and `MiniMaxH3PH3ImageConditioning`,
     `canvas='explicit'` changes the output canvas only; reference images stay in
     `[1024**2, 8 * 1024**2]`.
   - **Float16 conditioning-latent rounding (`encoders.py:64-65`, `klvae.py:910`)**:
     Rounds the unnormalized seeded posterior sample (`seed=42`) to `torch.float16`
     (~11-bit mantissa) before normalizing `(latents - latents_mean) / latents_std`.
   - **Strict 8-image cap (`MAX_EDIT_IMAGES = 8`, `image_1..image_8`)**:
     Matches `before_encoder.py:46, 158` (core's `PH3ImageConditioning` exposed 9 slots).

2. `MiniMaxH3PH3ImagePromptUpsampler`:
   Wraps `PrunaAI/P-H3-Image`'s fine-tuned `Qwen3.5-9B` prompt upsampler
   (`p-h3-image_prompt_upsampler_int8_convrot.safetensors`, loaded by core's
   `CLIPLoader` with `type="qwen_image"`) for both text-to-image and multi-image
   editing (1 to 8 images), with a bypass toggle (`enabled=False` by default) and
   lazy `clip` evaluation so a workflow can ship with pre-authored prompts without
   loading the 9.1 GB upsampler checkpoint unless the user turns `enabled` on.
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
# Inherited from `diffusers` `p_h3_image.before_encoder.PH3ImagePrepareInputsStep`
# (`coderef/diffusers/src/diffusers/modular_pipelines/p_h3_image/before_encoder.py:97-100, 118`).
DEFAULT_T2I_SIZE = 2912
REFERENCE_MIN_PIXELS = 1024 ** 2
REFERENCE_MAX_PIXELS = 8 * 1024 ** 2
CANVAS_MIN_PIXELS = 4 * 1024 ** 2
CANVAS_MODES: tuple[str, ...] = ("auto", "explicit")
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


def round_canvas(
    w: int,
    h: int,
    multiple: int = CANVAS_MULTIPLE,
) -> tuple[int, int]:
    """Round `(w, h)` to multiples of `multiple` (`diffusers` `before_encoder._round_canvas`)."""
    if w < 1 or h < 1:
        raise ValueError(f"canvas dimensions must be positive, got {w}x{h}")
    return (
        max(multiple, round(w / multiple) * multiple),
        max(multiple, round(h / multiple) * multiple),
    )


def size_in_area_range(
    w: int,
    h: int,
    min_pixels: int = REFERENCE_MIN_PIXELS,
    max_pixels: int = REFERENCE_MAX_PIXELS,
    multiple: int = CANVAS_MULTIPLE,
) -> tuple[int, int]:
    """`(w, h)` clamped into `[min_pixels, max_pixels]`, sides multiples of `multiple`.

    Matches `diffusers` `p_h3_image.before_encoder._size_in_area_range`
    (`coderef/diffusers/src/diffusers/modular_pipelines/p_h3_image/before_encoder.py:60-68`).
    Unlike ComfyUI core's `_size_within_area` (`comfy_extras/nodes_minimax_h3.py:668-671`),
    which only scales down (`min(1.0, ...)`), this also scales small images up to
    `min_pixels` (`4 MP` for `image_1` under `canvas='auto'`, `1 MP` for secondary
    references) and leaves in-range images whose sides are already multiples of 32
    untouched.
    """
    if w < 1 or h < 1:
        raise ValueError(f"image dimensions must be positive, got {w}x{h}")
    area = w * h
    scale = math.sqrt(min(max(area, min_pixels), max_pixels) / area)
    return (
        max(multiple, round(w * scale / multiple) * multiple),
        max(multiple, round(h * scale / multiple) * multiple),
    )


def resolve_ph3_image_geometry(
    image_sizes: list[tuple[int, int]],
    width: int = DEFAULT_T2I_SIZE,
    height: int = DEFAULT_T2I_SIZE,
    canvas: str = "auto",
) -> tuple[tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]]:
    """Compute `(canvas_size, reference_sizes, vision_sizes)` matching `diffusers` `before_encoder.py`.

    - Without images (T2I): `canvas_size` is `round_canvas(width, height)`, and both
      reference/vision lists are empty.
    - With edit images (`1..8`):
      - Every image is clamped to `[REFERENCE_MIN_PIXELS, REFERENCE_MAX_PIXELS]` (`1..8 MP`).
      - When `canvas == 'auto'` (default), `image_1` and `canvas_size` are clamped to
        `[CANVAS_MIN_PIXELS, REFERENCE_MAX_PIXELS]` (`4..8 MP`) at `image_1`'s aspect ratio
        (`before_encoder.py:168-170`).
      - When `canvas == 'explicit'`, `canvas_size` is `round_canvas(width, height)` while
        `reference_sizes` remain in `[1..8 MP]` (`before_encoder.py:164-166`), decoupling
        the output canvas from the VAE reference resolution.
      - `vision_sizes` are resized via `size_at_area(w, h, UPSAMPLER_IMAGE_PIXELS, 32)` (`~1 MP`)
        for Qwen3-VL (`before_encoder.py:173-175`).
    """
    if canvas not in CANVAS_MODES:
        raise ValueError(
            f"MiniMaxH3PH3ImageConditioning: unknown canvas mode {canvas!r}; "
            f"expected one of {CANVAS_MODES}."
        )
    if len(image_sizes) > MAX_EDIT_IMAGES:
        raise ValueError(
            f"MiniMaxH3PH3ImageConditioning supports at most {MAX_EDIT_IMAGES} edit images, "
            f"got {len(image_sizes)}."
        )
    if not image_sizes:
        return round_canvas(width, height, CANVAS_MULTIPLE), [], []

    reference_sizes = [
        size_in_area_range(w, h, REFERENCE_MIN_PIXELS, REFERENCE_MAX_PIXELS, CANVAS_MULTIPLE)
        for w, h in image_sizes
    ]
    if canvas == "explicit":
        canvas_size = round_canvas(width, height, CANVAS_MULTIPLE)
    else:
        reference_sizes[0] = size_in_area_range(
            image_sizes[0][0],
            image_sizes[0][1],
            CANVAS_MIN_PIXELS,
            REFERENCE_MAX_PIXELS,
            CANVAS_MULTIPLE,
        )
        canvas_size = reference_sizes[0]
    vision_sizes = [
        size_at_area(w, h, UPSAMPLER_IMAGE_PIXELS, CANVAS_MULTIPLE)
        for w, h in image_sizes
    ]
    return canvas_size, reference_sizes, vision_sizes


def format_edit_prompt(prompt: str) -> str:
    """Ensure an edit prompt starts with `'Edit <Picture i>'` or `'subject_definitions:'`.

    Matches `diffusers` `p_h3_image.encoders` (`coderef/diffusers/src/diffusers/modular_pipelines/p_h3_image/encoders.py:251-252`)
    and `comfy_extras/nodes_minimax_h3.py:40-42, 724`.
    """
    cleaned = prompt.strip()
    if not cleaned:
        raise ValueError("MiniMaxH3PH3ImageConditioning: `prompt` must not be empty.")
    if _EDIT_VERBATIM_RE.match(cleaned):
        return cleaned
    return f"Edit <Picture 1>. {cleaned}"


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


def _resize_image_to(image, target_size: tuple[int, int]):
    """Resize a `[B, H, W, C]` ComfyUI image tensor (first frame, RGB) to `target_size = (w, h)` (Lanczos)."""
    import comfy.utils

    frame = image[:1, ..., :3]
    img_h, img_w = int(frame.shape[1]), int(frame.shape[2])
    target_w, target_h = target_size
    if (img_w, img_h) == (target_w, target_h):
        return frame
    samples = frame.movedim(-1, 1)
    resized = comfy.utils.common_upscale(samples, target_w, target_h, "lanczos", "disabled")
    return resized.movedim(1, -1)


def _resize_upsampler_image(image):
    """Resize a `[B, H, W, C]` ComfyUI image tensor (first frame) to ~1 MP at multiples of 32 (Lanczos, RGB)."""
    frame = image[:1, ..., :3]
    img_h, img_w = int(frame.shape[1]), int(frame.shape[2])
    target_size = size_at_area(img_w, img_h, UPSAMPLER_IMAGE_PIXELS, CANVAS_MULTIPLE)
    return _resize_image_to(frame, target_size)


def _encode_ph3_image_condition(vae, image_tensor):
    """Encode an edit reference image with `vae.encode` and apply upstream float16 rounding.

    Both `diffusers` (`coderef/diffusers/src/diffusers/modular_pipelines/p_h3_image/encoders.py:64-65`)
    and `MiniMax-H3` (`coderef/MiniMax-H3/Ref2VA/video_vae/klvae.py:910`) round the
    unnormalized seeded posterior sample (`seed=42`) to `torch.float16` before
    normalizing `(latents - latents_mean) / latents_std`. ComfyUI core's
    `PH3ImageVAE.encode` (`comfy/ldm/minimax/vae.py:1022-1029`) omits `.to(torch.float16)`.
    Never mutate `fsm.latents_mean` or `fsm.latents_std` around `vae.encode`:
    `ModelPatcherDynamic.load` (`comfy/model_patcher.py:2114-2120`) snapshots
    `named_buffers()` into `backup_buffers` on first load and `restore_loaded_backups`
    (`comfy/model_patcher.py:1939-1940, 1960`) restores them on subsequent
    `VAEDecode` loads.
    """
    import torch

    z = vae.encode(image_tensor)
    fsm = getattr(vae, "first_stage_model", None)
    mean = getattr(fsm, "latents_mean", None) if fsm is not None else None
    std = getattr(fsm, "latents_std", None) if fsm is not None else None
    if isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor):
        mean_t = mean.view(1, -1, 1, 1, 1).to(device=z.device, dtype=torch.float32)
        std_t = std.view(1, -1, 1, 1, 1).to(device=z.device, dtype=torch.float32)
        unnorm = z.float() * std_t + mean_t
        rounded = unnorm.to(torch.float16).float()
        return ((rounded - mean_t) / std_t).to(dtype=z.dtype)
    return z.to(torch.float16).to(dtype=z.dtype)


def _av_latent(width: int, height: int, video_frames: int = 1, audio_frames: int = 2) -> dict:
    """Create the empty nested `(video, audio)` latent dict expected by `MiniMaxH3` for `P-H3-Image`.

    Matches `comfy_extras/nodes_minimax_h3.py:92-97`:
    - `video`: `[1, 24, video_frames, height // 16, width // 16]` (24 channels, 16x spatial compression)
    - `audio`: `[1, 32, 2, audio_frames]` (`comfy/model_base.py:2280` reads `shape[-1]` as `audio_t`)
    """
    import comfy.model_management
    import comfy.nested_tensor
    import torch

    device = comfy.model_management.intermediate_device()
    video_latent = torch.zeros(
        [1, 24, video_frames, height // 16, width // 16],
        device=device,
    )
    audio_latent = torch.zeros(
        [1, 32, 2, audio_frames],
        device=device,
    )
    return {"samples": comfy.nested_tensor.NestedTensor((video_latent, audio_latent))}


class MiniMaxH3PH3ImageConditioning(io.ComfyNode):
    """Condition `PrunaAI/P-H3-Image` for single-frame T2I or 1..8-image editing with `diffusers` parity."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PH3ImageConditioning",
            display_name="MiniMax H3 p-h3-image Conditioning",
            category="conditioning/video_models",
            description=(
                "Prepares conditioning and the empty nested (video, audio) latent for "
                "PrunaAI/P-H3-Image text-to-image (no images) and image editing (1 to 8 "
                "reference images), matching HuggingFace diffusers' `p_h3_image` pipeline "
                "and MiniMax-H3's float16 conditioning-latent rounding."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input(
                    "vae",
                    optional=True,
                    tooltip="Required when edit images (`image_1`..`image_8`) are connected.",
                ),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    tooltip=(
                        "Text-to-image prompt, 6-section `[keyframe completion]` edit brief, "
                        "or short edit instruction (`Edit <Picture 1>. ` is prepended "
                        "automatically when edit images are connected and the prompt does "
                        "not already start with `Edit <Picture i>` or `subject_definitions:`)."
                    ),
                ),
                io.Int.Input(
                    "width",
                    default=DEFAULT_T2I_SIZE,
                    min=CANVAS_MULTIPLE,
                    max=16384,
                    step=CANVAS_MULTIPLE,
                    tooltip=(
                        "Output canvas width in pixels (rounded to a multiple of 32). "
                        "Used for T2I, and for editing when `canvas` is `'explicit'`. "
                        "Ignored when edit images are connected and `canvas` is `'auto'`."
                    ),
                ),
                io.Int.Input(
                    "height",
                    default=DEFAULT_T2I_SIZE,
                    min=CANVAS_MULTIPLE,
                    max=16384,
                    step=CANVAS_MULTIPLE,
                    tooltip=(
                        "Output canvas height in pixels (rounded to a multiple of 32). "
                        "Used for T2I, and for editing when `canvas` is `'explicit'`. "
                        "Ignored when edit images are connected and `canvas` is `'auto'`."
                    ),
                ),
                io.Combo.Input(
                    "canvas",
                    options=list(CANVAS_MODES),
                    default="auto",
                    tooltip=(
                        "'auto' (default): in edit mode, derives the output canvas from "
                        "`image_1` clamped to `[4 MP, 8 MP]` (`4*1024^2..8*1024^2`), "
                        "matching diffusers when width/height are omitted; in T2I mode, "
                        "uses `width x height`. 'explicit': uses `width x height` for the "
                        "output canvas while keeping edit reference images clamped to "
                        "`[1 MP, 8 MP]` (decoupled from the output canvas)."
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
                        "Optional edit images (`image_1` is the primary image to edit; "
                        "`image_2`..`image_8` are additional references). Leave empty for "
                        "text-to-image."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        prompt: str,
        width: int = DEFAULT_T2I_SIZE,
        height: int = DEFAULT_T2I_SIZE,
        canvas: str = "auto",
        vae=None,
        images: io.Autogrow.Type = None,
    ) -> io.NodeOutput:
        import node_helpers

        try:
            from .h3_spans import attach_span_masks
        except ImportError:
            from h3_spans import attach_span_masks

        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise ValueError("MiniMaxH3PH3ImageConditioning: `prompt` must not be empty.")

        names = sorted(images or {}, key=lambda name: int(name.rsplit("_", 1)[-1]))
        raw_images = [images[name] for name in names if images[name] is not None]
        if len(raw_images) > MAX_EDIT_IMAGES:
            raise ValueError(
                f"MiniMaxH3PH3ImageConditioning supports at most {MAX_EDIT_IMAGES} edit images, "
                f"got {len(raw_images)}."
            )

        image_sizes = [(int(img.shape[2]), int(img.shape[1])) for img in raw_images]
        canvas_size, ref_sizes, vision_sizes = resolve_ph3_image_geometry(
            image_sizes,
            width=width,
            height=height,
            canvas=canvas,
        )
        canvas_w, canvas_h = canvas_size

        if raw_images:
            if vae is None:
                raise ValueError(
                    "MiniMaxH3PH3ImageConditioning: `vae` is required when edit images are connected."
                )
            vision_images = [
                _resize_image_to(img, vis_size)
                for img, vis_size in zip(raw_images, vision_sizes)
            ]
            refs = []
            for img, ref_size in zip(raw_images, ref_sizes):
                z = _encode_ph3_image_condition(vae, _resize_image_to(img, ref_size))
                refs.append(
                    {
                        "kind": "image",
                        "latent_h": int(z.shape[-2]),
                        "latent_w": int(z.shape[-1]),
                        "latent": z,
                    }
                )
            formatted_prompt = format_edit_prompt(cleaned_prompt)
            tokens = clip.tokenize(formatted_prompt, images=vision_images)
            cond = clip.encode_from_tokens_scheduled(tokens)
            cond = node_helpers.conditioning_set_values(
                cond,
                {"minimax_refs": refs},
            )
        else:
            tokens = clip.tokenize(cleaned_prompt)
            cond = clip.encode_from_tokens_scheduled(tokens)

        attach_span_masks(cond, clip, tokens)
        return io.NodeOutput(cond, _av_latent(canvas_w, canvas_h, 1, 2))


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
