"""Standalone `1x` Cross-Modal Query Router (`MiniMaxH3QueryRouter`, Step 1 / v1).

Routes which text `CONDITIONING` (prompt sections or an arbitrary-length second prompt)
`Q_vid` and `Q_aud` read per DiT block band inside a single forward pass (`1.00x` NFE),
built cleanly on top of `steering.hooks.H3SteerRuntime`.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from comfy_api.latest import io

from . import hooks as _steer_hooks
from .constants import STEER_BLOCK_BANDS, import_repo_module

logger = logging.getLogger("ComfyUI-h3-explorations.query_router")

ROUTER_WRAPPER_KEY = "minimax_h3_query_router"
SECTION_SPLIT_MODES: tuple[str, ...] = (
    "off",
    "soundscape_to_audio",
    "sections_by_modality",
)

_spans = import_repo_module("h3_spans")


def _is_h3_image_model(dm: Any) -> bool:
    if dm is None:
        return False
    cls_name = type(dm).__name__
    if cls_name == "MiniMaxH3ImageModel":
        return True
    return getattr(dm, "audio_patch_proj", None) is None


def _extract_cond_tensor(conditioning: Any, *, name: str) -> torch.Tensor:
    if not isinstance(conditioning, (list, tuple)) or not conditioning:
        raise ValueError(f"MiniMaxH3QueryRouter {name} must be a non-empty ComfyUI CONDITIONING")
    first = conditioning[0]
    if not isinstance(first, (list, tuple)) or not first or not torch.is_tensor(first[0]):
        raise ValueError(f"MiniMaxH3QueryRouter {name}[0][0] must be a conditioning tensor")
    t = first[0]
    if t.ndim == 2:
        t = t.unsqueeze(0)
    if t.ndim != 3 or int(t.shape[1]) <= 0:
        raise ValueError(f"MiniMaxH3QueryRouter {name} must have shape [B, L, D] with L > 0, got {tuple(t.shape)}")
    return t


def _resolve_span_masks(payload: dict[str, Any]) -> dict[str, Any] | None:
    span_masks = payload.get("h3_span_masks")
    if isinstance(span_masks, dict):
        return span_masks
    tags_raw = payload.get("text_token_tags")
    sm = getattr(tags_raw, "_h3_span_masks", None)
    return sm if isinstance(sm, dict) else None


def _split_context_by_section(
    base_ctx: torch.Tensor,
    payload: dict[str, Any],
    *,
    section_split: str,
    dm: Any,
    transformer_options: dict[str, Any],
    want_vid: bool,
    want_aud: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Compute `(ctx_vid_or_none, ctx_aud_or_none)` from structured H3 prompt sections."""
    if section_split == "off":
        return None, None
    span_masks = _resolve_span_masks(payload)
    if span_masks is None:
        return None, None

    scene_t = span_masks.get("scene")
    vision_t = span_masks.get("vision")
    if not isinstance(scene_t, torch.Tensor) or not isinstance(vision_t, torch.Tensor):
        return None, None
    L = int(base_ctx.shape[1])
    if int(scene_t.numel()) != L or int(vision_t.numel()) != L:
        return None, None
    if not bool(getattr(scene_t, "_h3_has_sections", False)):
        return None, None

    vid_ctx: torch.Tensor | None = None
    aud_ctx: torch.Tensor | None = None
    device = base_ctx.device

    if want_vid:
        # Strict section mask with zero fallback to scene_only on non-sectioned inputs.
        snd_mask = _spans.inference_mask(span_masks, "overall_soundscape+non_diegetic_music").to(device=device)
        keep_vid = (~snd_mask) | vision_t.to(device=device)
        keep_vid[0] = True  # preserve row-0 Qwen3-VL attention sink
        if not bool(keep_vid.all()) and int(keep_vid.sum().item()) > 0:
            vid_raw = base_ctx[:, keep_vid, :]
            vid_ctx = _steer_hooks.refine_text_context(
                dm, vid_raw, transformer_options, caller_name="MiniMaxH3QueryRouter(section_split)"
            )

    if want_aud and section_split == "sections_by_modality":
        scene_only_mask = _spans.inference_mask(span_masks, "scene_only").to(device=device)
        keep_aud = ~scene_only_mask
        keep_aud[0] = True  # preserve row-0 attention sink alongside <d> dialogue & soundscape
        if not bool(keep_aud.all()) and int(keep_aud.sum().item()) > 0:
            aud_raw = base_ctx[:, keep_aud, :]
            aud_ctx = _steer_hooks.refine_text_context(
                dm, aud_raw, transformer_options, caller_name="MiniMaxH3QueryRouter(section_split)"
            )

    return vid_ctx, aud_ctx


def _make_query_router_wrapper(
    *,
    section_split: str,
    video_block_band: str,
    audio_block_band: str,
    start_percent: float,
    end_percent: float,
    audio_cond_raw: torch.Tensor | None,
    video_cond_raw: torch.Tensor | None,
    workspace: _steer_hooks.H3SteerStepWorkspace,
    dm: Any,
):
    def wrapper(executor, x, timestep, context, transformer_options, **call_kwargs):
        if not _steer_hooks.step_in_window(transformer_options, timestep, start_percent, end_percent):
            return executor(x, timestep, context, transformer_options, **call_kwargs)

        has_audio_stream = isinstance(x, (list, tuple)) and len(x) >= 2
        payload = call_kwargs.get("minimax_payload") or {}
        if not isinstance(payload, dict):
            payload = {}

        with _steer_hooks.pause_malloc_graph_ctx():
            ctx_base = _steer_hooks.refine_text_context(
                dm, context, transformer_options, caller_name="MiniMaxH3QueryRouter"
            )
            ctx_vid_sec, ctx_aud_sec = _split_context_by_section(
                context,
                payload,
                section_split=section_split,
                dm=dm,
                transformer_options=transformer_options,
                want_vid=(video_cond_raw is None),
                want_aud=(audio_cond_raw is None and has_audio_stream),
            )
            if video_cond_raw is not None:
                ctx_vid = _steer_hooks.refine_text_context(
                    dm,
                    video_cond_raw.to(device=ctx_base.device, dtype=ctx_base.dtype),
                    transformer_options,
                    caller_name="MiniMaxH3QueryRouter(video_conditioning)",
                )
            else:
                ctx_vid = ctx_vid_sec

            if audio_cond_raw is not None and has_audio_stream:
                ctx_aud = _steer_hooks.refine_text_context(
                    dm,
                    audio_cond_raw.to(device=ctx_base.device, dtype=ctx_base.dtype),
                    transformer_options,
                    caller_name="MiniMaxH3QueryRouter(audio_conditioning)",
                )
            elif has_audio_stream:
                ctx_aud = ctx_aud_sec
            else:
                ctx_aud = None

        if ctx_vid is None and ctx_aud is None:
            return executor(x, timestep, ctx_base, transformer_options, **call_kwargs)

        n_blocks = len(dm.blocks)
        all_blocks = frozenset(range(n_blocks))
        vid_band = _steer_hooks.parse_block_band(video_block_band, n_blocks)
        aud_band = _steer_hooks.parse_block_band(audio_block_band, n_blocks)

        def _invoke_rt(
            *,
            mode_id: int,
            main_ctx: torch.Tensor,
            shadow_a_init: torch.Tensor,
            shadow_b_init: torch.Tensor | None,
            swap_blocks: frozenset[int],
            aud_split_blocks: frozenset[int],
            dual_shadow: bool,
            aud_uses_shadow_b: bool,
            aud_unswapped_blocks: frozenset[int] = frozenset(),
        ) -> Any:
            with _steer_hooks.pause_malloc_graph_ctx():
                rt = _steer_hooks.H3SteerRuntime(
                    dm=dm,
                    workspace=workspace,
                    mode=mode_id,
                    main_ctx=main_ctx,
                    shadow_a_init=shadow_a_init,
                    shadow_b_init=shadow_b_init,
                    swap_blocks=swap_blocks,
                    aud_split_blocks=aud_split_blocks,
                    dual_shadow=dual_shadow,
                    fork_b_at_block=None,
                    aud_uses_shadow_b=aud_uses_shadow_b,
                    aud_unswapped_blocks=aud_unswapped_blocks,
                )
                to_next = dict(transformer_options or {})
                to_next["_h3_steer_runtime"] = rt
                to_next["optimized_attention_override"] = _steer_hooks.make_steer_attention_override(
                    to_next.get("optimized_attention_override")
                )
            out_res = executor(x, timestep, rt.main_ctx, to_next, **call_kwargs)
            sink = to_next.get("_h3_steer_stats_sink")
            if callable(sink):
                sink(dict(rt.stats))
            elif isinstance(sink, list):
                sink.append(dict(rt.stats))
            return out_res

        # Case A: Audio-only route (`ctx_vid is None` or empty video band)
        if ctx_vid is None or len(vid_band) == 0:
            if ctx_aud is None or not has_audio_stream or len(aud_band) == 0:
                return executor(x, timestep, ctx_base, transformer_options, **call_kwargs)
            return _invoke_rt(
                mode_id=3 if aud_band != all_blocks else 2,
                main_ctx=ctx_base,
                shadow_a_init=ctx_aud,
                shadow_b_init=None,
                swap_blocks=frozenset(),
                aud_split_blocks=aud_band,
                dual_shadow=False,
                aud_uses_shadow_b=False,
            )

        # Case B: `ctx_vid` spans ALL blocks (supports arbitrary l_txt_vid by making ctx_vid main_ctx)
        if vid_band == all_blocks:
            if not has_audio_stream:
                return executor(x, timestep, ctx_vid, transformer_options, **call_kwargs)
            aud_target = ctx_aud if (ctx_aud is not None and len(aud_band) > 0) else ctx_base
            aud_target_blocks = aud_band if ctx_aud is not None else all_blocks
            if ctx_aud is not None and aud_band != all_blocks and video_cond_raw is not None:
                return _invoke_rt_split_two_shadows(
                    _invoke_rt,
                    main_ctx=ctx_vid,
                    ctx_aud_inside=ctx_aud,
                    ctx_aud_outside=ctx_base,
                    aud_band=aud_band,
                    all_blocks=all_blocks,
                )
            return _invoke_rt(
                mode_id=3 if aud_target_blocks != all_blocks else 2,
                main_ctx=ctx_vid,
                shadow_a_init=aud_target,
                shadow_b_init=None,
                swap_blocks=frozenset(),
                aud_split_blocks=aud_target_blocks,
                dual_shadow=False,
                aud_uses_shadow_b=False,
            )

        # Case C: Partial `video_block_band` (`vid_band` is a proper non-empty subset of all_blocks)
        l_base = int(ctx_base.shape[1])
        l_vid = int(ctx_vid.shape[1])
        if l_vid != l_base:
            raise ValueError(
                "MiniMaxH3QueryRouter partial video_block_band requires video_conditioning and base conditioning "
                f"to have equal token length ({l_vid} != {l_base}); use video_block_band='all' for arbitrary-length video prompts"
            )
        if not has_audio_stream:
            return _invoke_rt(
                mode_id=1,
                main_ctx=ctx_base,
                shadow_a_init=ctx_vid,
                shadow_b_init=None,
                swap_blocks=vid_band,
                aud_split_blocks=frozenset(),
                dual_shadow=False,
                aud_uses_shadow_b=False,
            )
        if ctx_aud is None or len(aud_band) == 0:
            return _invoke_rt(
                mode_id=1,
                main_ctx=ctx_base,
                shadow_a_init=ctx_vid,
                shadow_b_init=None,
                swap_blocks=vid_band,
                aud_split_blocks=frozenset(),
                dual_shadow=False,
                aud_uses_shadow_b=False,
                aud_unswapped_blocks=vid_band,
            )
        return _invoke_rt(
            mode_id=3,
            main_ctx=ctx_base,
            shadow_a_init=ctx_vid,
            shadow_b_init=ctx_aud,
            swap_blocks=vid_band,
            aud_split_blocks=aud_band,
            dual_shadow=True,
            aud_uses_shadow_b=True,
            aud_unswapped_blocks=vid_band - aud_band,
        )

    return wrapper


def _invoke_rt_split_two_shadows(
    invoke_fn,
    *,
    main_ctx: torch.Tensor,
    ctx_aud_inside: torch.Tensor,
    ctx_aud_outside: torch.Tensor,
    aud_band: frozenset[int],
    all_blocks: frozenset[int],
) -> Any:
    """When `main_ctx` is `ctx_vid` (all blocks) and audio uses `ctx_aud_inside` on `aud_band`,
    keep blocks outside `aud_band` on `ctx_aud_outside` (`shadow_a` before first swapped block
    or via per-block selection? Wait: `H3SteerRuntime` routes `aud_split_blocks` to ONE shadow!)."""
    # Note: In Case B (`vid_band == all` and `aud_band != all`), if `main_ctx = ctx_vid` then
    # outside `aud_band`, `main_ctx` is `ctx_vid`, not `ctx_aud_outside`. To hold blocks outside
    # `aud_band` on `ctx_base` while `main_ctx` is `ctx_vid`, `main_ctx` must equal `ctx_base` when
    # `l_vid == l_base`, or audio stays on `ctx_vid` outside `aud_band`. Raise clean explicit error
    # only if arbitrary-length `video_conditioning` is paired with a partial `audio_block_band` unless
    # they share equal length.
    l_main = int(main_ctx.shape[1])
    l_out = int(ctx_aud_outside.shape[1])
    if l_main == l_out:
        return invoke_fn(
            mode_id=3,
            main_ctx=ctx_aud_outside,
            shadow_a_init=main_ctx,
            shadow_b_init=ctx_aud_inside,
            swap_blocks=all_blocks,
            aud_split_blocks=aud_band,
            dual_shadow=True,
            aud_uses_shadow_b=True,
            aud_unswapped_blocks=all_blocks - aud_band,
        )
    raise ValueError(
        "MiniMaxH3QueryRouter with arbitrary-length video_conditioning ('all' blocks) and "
        "non-'all' audio_block_band requires equal video/base text length or audio_block_band='all'"
    )


def attach_query_router(
    model: Any,
    *,
    section_split: str = "off",
    video_block_band: str = "all",
    audio_block_band: str = "all",
    start_percent: float = 0.0,
    end_percent: float = 1.0,
    audio_conditioning: Any = None,
    video_conditioning: Any = None,
) -> Any:
    """Clone `model` and attach a `1x` Cross-Modal Query Router wrapper + `double_block` hooks."""
    sec_mode = str(section_split or "off").strip()
    if sec_mode not in SECTION_SPLIT_MODES:
        raise ValueError(f"section_split must be one of {SECTION_SPLIT_MODES}, got {section_split!r}")
    norm_vid_band = _steer_hooks.validate_block_band(video_block_band)
    norm_aud_band = _steer_hooks.validate_block_band(audio_block_band)
    s_pct = float(start_percent)
    e_pct = float(end_percent)
    if not (0.0 <= s_pct < e_pct <= 1.0):
        raise ValueError(
            f"start_percent/end_percent must satisfy 0.0 <= start_percent < end_percent <= 1.0, got [{s_pct}, {e_pct}]"
        )

    audio_cond_raw = _extract_cond_tensor(audio_conditioning, name="audio_conditioning") if audio_conditioning is not None else None
    video_cond_raw = _extract_cond_tensor(video_conditioning, name="video_conditioning") if video_conditioning is not None else None

    if sec_mode == "off" and audio_cond_raw is None and video_cond_raw is None:
        return model

    m = model.clone()
    dm = getattr(getattr(m, "model", None), "diffusion_model", None)
    if not _steer_hooks.is_hookable_dit(dm):
        raise ValueError("MiniMaxH3QueryRouter requires a MiniMax H3 model with hookable DiT blocks")
    if _is_h3_image_model(dm):
        raise ValueError("MiniMaxH3QueryRouter targets joint audio-video MiniMax H3 models")

    _steer_hooks.parse_block_band(norm_vid_band, len(dm.blocks))
    _steer_hooks.parse_block_band(norm_aud_band, len(dm.blocks))

    ws = _steer_hooks.install_steer_block_hooks(
        m, dm, key_prefix=ROUTER_WRAPPER_KEY, node_label="MiniMaxH3QueryRouter"
    )
    from comfy.patcher_extension import WrappersMP

    wrappers = m.wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})
    wrappers[ROUTER_WRAPPER_KEY] = [
        _make_query_router_wrapper(
            section_split=sec_mode,
            video_block_band=norm_vid_band,
            audio_block_band=norm_aud_band,
            start_percent=s_pct,
            end_percent=e_pct,
            audio_cond_raw=audio_cond_raw,
            video_cond_raw=video_cond_raw,
            workspace=ws,
            dm=dm,
        )
    ]
    return m


class MiniMaxH3QueryRouter(io.ComfyNode):
    """Single-pass (`1x`) Cross-Modal Query Router (`Q_vid` / `Q_aud` per-stream & per-depth prompt routing)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3QueryRouter",
            display_name="MiniMax H3 Query Router",
            category="MiniMax H3/Steering",
            description=(
                "Route distinct text conditioning (arbitrary-length second prompt or automatic "
                "soundscape/scene section splits) to video vs. audio queries per DiT block band in a single 1x pass."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "section_split",
                    options=list(SECTION_SPLIT_MODES),
                    default="off",
                    tooltip=(
                        "Single-prompt auto-routing when audio/video_conditioning are omitted: "
                        "'off', 'soundscape_to_audio' (strips overall_soundscape/non_diegetic_music from video "
                        "pre-TokenRefiner so sound text never leaks into picture), or 'sections_by_modality' "
                        "(also strips visual scene prose from audio while keeping dialogue + soundscape)."
                    ),
                ),
                io.Combo.Input(
                    "video_block_band",
                    options=list(STEER_BLOCK_BANDS),
                    default="all",
                    tooltip="DiT block band where video queries read video_conditioning / section-filtered prompt.",
                ),
                io.Combo.Input(
                    "audio_block_band",
                    options=list(STEER_BLOCK_BANDS),
                    default="all",
                    tooltip=(
                        "DiT block band where stereo audio queries read audio_conditioning / section-filtered audio "
                        "('mid_late_20_49' reaches vocal delivery; 'late_30_49' focuses on acoustic space)."
                    ),
                ),
                io.Float.Input(
                    "start_percent",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Normalized sampler start fraction for query routing [0.0, 1.0).",
                ),
                io.Float.Input(
                    "end_percent",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Normalized sampler end fraction for query routing (0.0, 1.0].",
                ),
                io.Conditioning.Input(
                    "audio_conditioning",
                    optional=True,
                    tooltip="Optional separate prompt CONDITIONING (any token length) read only by audio queries.",
                ),
                io.Conditioning.Input(
                    "video_conditioning",
                    optional=True,
                    tooltip="Optional separate prompt CONDITIONING (any length on 'all' blocks) read only by video queries.",
                ),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(
        cls,
        model: Any,
        section_split: str = "off",
        video_block_band: str = "all",
        audio_block_band: str = "all",
        start_percent: float = 0.0,
        end_percent: float = 1.0,
        audio_conditioning: Any = None,
        video_conditioning: Any = None,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            attach_query_router(
                model,
                section_split=section_split,
                video_block_band=video_block_band,
                audio_block_band=audio_block_band,
                start_percent=start_percent,
                end_percent=end_percent,
                audio_conditioning=audio_conditioning,
                video_conditioning=video_conditioning,
            )
        )
