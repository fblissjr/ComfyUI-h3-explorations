"""Dedicated 1x Video Prompt Expander node (`MiniMaxH3PromptExpander`) powered by `heylook` + `h3_prompt_contract`.

Exact video sibling of `MiniMaxH3ImagePromptExpander` (`h3_image_nodes.py`):
- Takes a short plain-English idea (and/or optional `concept` preset), wired media
  (`references`, `first_frame`, `last_frame`), and clip `length`.
- Infers the exact H3 prompt mode (`ref2va`, `t2va`, `i2va`, `fl2va`, `l2va`, `single_frame`)
  from wired sockets when `mode="auto"`.
- Expands via local/LAN `heylookitsanllm` (`/v1/messages`) with deterministic
  `vendor_guides/base_en.md` Part One + `S.SS` stamping and `preflight_graph`-equivalent
  mechanical validation (with 1-shot repair).
- Wires straight into standard 1x `MiniMaxH3Conditioning` / `MiniMaxH3ReferenceConditioning`
  + `BasicGuider` workflows (and `MiniMaxH3ReleaseReferences.video_end_percent` via slot 1),
  requiring zero `EditGuider`, zero duplicate conditioning, and zero empty `original` box.
"""

from __future__ import annotations

from typing import Any

from comfy_api.latest import io

try:
    from . import h3_prompt_contract as _pc
    from .auto_edit_prompt import (
        REWRITE_MODES,
        SAMPLER_MODES,
        THINKING_MODES,
        list_concept_options,
        run_auto_edit_rewrite,
    )
    from .reference_conditioning import H3References
except ImportError:  # pragma: no cover - direct repo-root import
    import h3_prompt_contract as _pc  # type: ignore[no-redef]
    from auto_edit_prompt import (  # type: ignore[no-redef]
        REWRITE_MODES,
        SAMPLER_MODES,
        THINKING_MODES,
        list_concept_options,
        run_auto_edit_rewrite,
    )
    H3References = io.Custom("MINIMAX_H3_REFERENCES")


def resolve_expander_mode(
    mode: str,
    *,
    references: Any = None,
    first_frame: Any = None,
    last_frame: Any = None,
    length: int = 141,
) -> str:
    """Resolve `mode='auto'` cleanly from wired media sockets and clip `length`."""
    raw = (mode or "auto").strip().lower()
    if raw not in _pc.PROMPT_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {_pc.PROMPT_MODES}")
    if raw != "auto":
        return raw
    if references is not None:
        return "ref2va"
    if first_frame is not None and last_frame is not None:
        return "fl2va"
    if first_frame is not None:
        return "i2va"
    if last_frame is not None:
        return "l2va"
    if int(length) == 1:
        return "single_frame"
    return "t2va"


def expand_video_prompt(
    *,
    prompt: str = "",
    concept: str = "none",
    mode: str = "auto",
    length: int = 141,
    enabled: bool = True,
    base_url: str = "",
    model: str = "",
    thinking: str = "on",
    reasoning_effort: str = "medium",
    seed: int = 0,
    references: Any = None,
    first_frame: Any = None,
    last_frame: Any = None,
    preset: str = "",
    sampler_mode: str = "heylook / preset default",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    custom_system_prompt: str = "",
    use_cache: bool = True,
) -> dict[str, Any]:
    """Expand a short video idea + wired media into a guide-verified H3 prompt."""
    prompt_text = (prompt or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    concept_name = (concept or "none").strip() or "none"
    eff_length = int(length)
    eff_mode = resolve_expander_mode(
        mode,
        references=references,
        first_frame=first_frame,
        last_frame=last_frame,
        length=eff_length,
    )

    kf_images: dict[str, Any] = {}
    if references is None:
        kf_list = [t for t in (first_frame, last_frame) if t is not None]
        for idx, tensor in enumerate(kf_list, start=1):
            kf_images[f"image_{idx}"] = tensor

    manifest = _pc.describe_references(references, n_extra_images=len(kf_images))
    video_end_percent = 0.33 if manifest.has_synthetic_video else 1.0
    dur_s = f"{(1 if eff_length == 1 else _pc._rules.snap_length(max(5, eff_length))) / 24.0:.2f}"

    if not enabled:
        if not prompt_text:
            raise ValueError(
                "MiniMaxH3PromptExpander: 'prompt' is empty while enabled=False"
            )
        out_prompt = prompt_text
        if eff_mode in _pc.KEYFRAME_BASE_MODES and "integrated_multimodal_description:" in out_prompt:
            out_prompt = _pc.stamp_part_one(out_prompt, mode=eff_mode, length=eff_length)
        return {
            "prompt": out_prompt,
            "video_end_percent": video_end_percent,
            "thinking": "",
            "report": (
                f"expander disabled (passthrough) | mode={eff_mode} len={eff_length} "
                f"({dur_s}s) rel_vid={video_end_percent:g}"
            ),
            "resolved_mode": eff_mode,
            "cache_hit": False,
        }

    if not prompt_text and concept_name.lower() == "none":
        raise ValueError(
            "MiniMaxH3PromptExpander: enter a short 'prompt' idea or choose a 'concept' preset"
        )

    res = run_auto_edit_rewrite(
        original="",
        concept=concept_name,
        edit_intent=prompt_text,
        take_mode="fast_1x",
        timing="whole_clip",
        enabled=True,
        base_url=base_url,
        model=model,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
        seed=seed,
        images=kf_images or None,
        preset=preset,
        sampler_mode=sampler_mode,
        temperature=temperature,
        max_tokens=max_tokens,
        custom_system_prompt=custom_system_prompt,
        use_cache=use_cache,
        references=references,
        length=eff_length,
        prompt_mode=eff_mode,
    )
    out_prompt = res["edited_prompt"]
    report = (
        f"mode={eff_mode} len={eff_length} ({dur_s}s) "
        f"rel_vid={video_end_percent:g} | {res['report']}"
    )
    return {
        "prompt": out_prompt,
        "video_end_percent": video_end_percent,
        "thinking": res["thinking"],
        "report": report,
        "resolved_mode": eff_mode,
        "cache_hit": bool(res.get("cache_hit")),
    }


class MiniMaxH3PromptExpander(io.ComfyNode):
    """Expand a short idea + wired keyframes/references into a guide-verified H3 video prompt via `heylook`."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptExpander",
            display_name="MiniMax H3 Video Prompt Expander (heylook)",
            category="conditioning/minimax",
            description=(
                "1x video Easy Button (sibling of MiniMaxH3ImagePromptExpander): expands a short "
                "scene idea and optional concept preset + wired references or first/last keyframes "
                "into a guide-verified H3 prompt (T2VA, I2VA, FL2VA, L2VA, multimodal ref2va, or "
                "single_frame) via local/LAN heylookitsanllm (/v1/messages). Wires straight into "
                "standard MiniMaxH3Conditioning / MiniMaxH3ReferenceConditioning + BasicGuider."
            ),
            inputs=[
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    tooltip=(
                        "Short plain-English scene idea (or a finished H3 prompt when enabled=False). "
                        "Can be used alone or combined with 'concept' below."
                    ),
                ),
                io.Combo.Input(
                    "concept",
                    options=list_concept_options(),
                    default="none",
                    tooltip=(
                        "Optional preset card (.toml under data/edit_concepts/, data/control_vectors/, "
                        "or prompt_bank/control_vectors/). Leave 'none' to author purely from 'prompt'."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=list(_pc.PROMPT_MODES),
                    default="auto",
                    tooltip=(
                        "H3 prompt contract mode. 'auto' infers ref2va (references wired), fl2va "
                        "(first_frame + last_frame), i2va (first_frame), l2va (last_frame), "
                        "single_frame (length=1), or t2va (text only)."
                    ),
                ),
                io.Int.Input(
                    "length",
                    default=141,
                    min=1,
                    max=362,
                    tooltip=(
                        "Clip frame count (wire from MiniMaxH3Resolution length output). Used to "
                        "stamp exact S.SS duration in FL2VA/L2VA Part One lines (1 = single_frame)."
                    ),
                ),
                io.Boolean.Input(
                    "enabled",
                    default=True,
                    tooltip=(
                        "Call heylook to expand 'prompt'. Turn off to pass a hand-written H3 prompt "
                        "straight through with zero network I/O."
                    ),
                ),
                io.String.Input(
                    "base_url",
                    default="",
                    tooltip=(
                        "Optional heylook server URL override. Leave blank to read H3_HEYLOOK_URL "
                        "from environment / gitignored h3.env, falling back to http://127.0.0.1:8080."
                    ),
                ),
                io.String.Input(
                    "model",
                    default="",
                    tooltip=(
                        "Optional model id from GET /v1/models. Leave blank to auto-select a served "
                        "'chat' model (plus 'vision' when media is wired), preferring resident models."
                    ),
                ),
                io.Combo.Input(
                    "thinking",
                    options=list(THINKING_MODES),
                    default="on",
                    tooltip=(
                        "Thinking switch on models advertising 'thinking'. Reasoning blocks are "
                        "routed to the 'thinking' output and never reach H3 conditioning."
                    ),
                ),
                io.String.Input(
                    "reasoning_effort",
                    default="medium",
                    tooltip=(
                        "Reasoning depth vocabulary word (e.g. 'medium'). Capability-gated and "
                        "omitted on models without depth support (e.g. Gemma-4)."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip="Seed forwarded to heylook (when supported) and ComfyUI node caching.",
                ),
                H3References.Input(
                    "references",
                    optional=True,
                    tooltip=(
                        "Optional ordered H3 references chain (from MiniMaxH3AppendRefImage / "
                        "AppendRefVideo / AppendSpriteTrajectory / AppendRefAudio) for multimodal "
                        "ref2va prompt grounding and video_end_percent auto-release."
                    ),
                ),
                io.Image.Input(
                    "first_frame",
                    optional=True,
                    tooltip="Optional start keyframe image (triggers I2VA, or FL2VA with last_frame).",
                ),
                io.Image.Input(
                    "last_frame",
                    optional=True,
                    tooltip="Optional end keyframe image (triggers L2VA, or FL2VA with first_frame).",
                ),
                io.String.Input(
                    "preset",
                    default="",
                    optional=True,
                    tooltip="Optional preset name or id from GET /v1/presets on heylook.",
                ),
                io.Combo.Input(
                    "sampler_mode",
                    options=list(SAMPLER_MODES),
                    default="heylook / preset default",
                    optional=True,
                    tooltip="'heylook / preset default' uses server cascade; 'custom' sends temperature and max_tokens.",
                ),
                io.Float.Input(
                    "temperature",
                    default=0.7,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    optional=True,
                    tooltip="Sampling temperature sent when sampler_mode is 'custom'.",
                ),
                io.Int.Input(
                    "max_tokens",
                    default=4096,
                    min=64,
                    max=32768,
                    step=64,
                    optional=True,
                    tooltip="Maximum output tokens sent when sampler_mode is 'custom'.",
                ),
                io.String.Input(
                    "custom_system_prompt",
                    multiline=True,
                    default="",
                    optional=True,
                    tooltip="Optional editable override for the expander system prompt.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.Float.Output(display_name="video_end_percent"),
                io.String.Output(display_name="thinking"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        prompt: str = "",
        concept: str = "none",
        mode: str = "auto",
        length: int = 141,
        enabled: bool = True,
        base_url: str = "",
        model: str = "",
        thinking: str = "on",
        reasoning_effort: str = "medium",
        seed: int = 0,
        references: Any = None,
        first_frame: Any = None,
        last_frame: Any = None,
        preset: str = "",
        sampler_mode: str = "heylook / preset default",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        custom_system_prompt: str = "",
    ) -> io.NodeOutput:
        res = expand_video_prompt(
            prompt=prompt,
            concept=concept,
            mode=mode,
            length=length,
            enabled=enabled,
            base_url=base_url,
            model=model,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            seed=seed,
            references=references,
            first_frame=first_frame,
            last_frame=last_frame,
            preset=preset,
            sampler_mode=sampler_mode,
            temperature=temperature,
            max_tokens=max_tokens,
            custom_system_prompt=custom_system_prompt,
        )
        return io.NodeOutput(
            res["prompt"],
            float(res["video_end_percent"]),
            res["thinking"],
            res["report"],
        )
