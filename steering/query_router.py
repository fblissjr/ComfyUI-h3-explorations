"""Standalone `1x` Cross-Modal Query Router (`MiniMaxH3QueryRouter`, Step 1 / v1).

Routes which text `CONDITIONING` (prompt sections or an arbitrary-length second prompt)
`Q_vid` and `Q_aud` read per DiT block band inside a single forward pass (`1.00x` NFE),
built cleanly on top of `steering.hooks.H3SteerRuntime`.
"""

from __future__ import annotations

from collections import OrderedDict
import logging
from typing import Any

import torch
from comfy_api.latest import io

from . import hooks as _steer_hooks
from .constants import ENCODER_DIM, STEER_BLOCK_BANDS, import_repo_module

logger = logging.getLogger("ComfyUI-h3-explorations.query_router")

ROUTER_WRAPPER_KEY = "minimax_h3_query_router"
SECTION_SPLIT_MODES: tuple[str, ...] = (
    "off",
    "soundscape_to_audio",
    "sections_by_modality",
)
AUDIO_CHANNEL_MODES: tuple[str, ...] = _steer_hooks.AUDIO_CHANNEL_MODES
REF_AUDIO_STREAM_MODES: tuple[str, ...] = ("both", "audio_only")
_FULL_GAIN_EPS = 1e-6
_REF_CACHE_MAX = 4

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


def _extract_cond_tags(conditioning: Any) -> torch.Tensor | None:
    if not isinstance(conditioning, (list, tuple)) or not conditioning:
        return None
    first = conditioning[0]
    if not isinstance(first, (list, tuple)) or len(first) < 2 or not isinstance(first[1], dict):
        return None
    tags_raw = first[1].get("minimax_token_tags")
    if not isinstance(tags_raw, torch.Tensor):
        return None
    return tags_raw.detach().to(device="cpu", dtype=torch.long).flatten()


def _resolve_span_masks(payload: dict[str, Any], call_kwargs: dict[str, Any] | None = None) -> dict[str, Any] | None:
    kw = call_kwargs or {}
    span_masks = kw.get("h3_span_masks")
    if isinstance(span_masks, dict):
        return span_masks
    span_masks = payload.get("h3_span_masks")
    if isinstance(span_masks, dict):
        return span_masks
    tags_raw = payload.get("text_token_tags")
    sm = getattr(tags_raw, "_h3_span_masks", None)
    return sm if isinstance(sm, dict) else None


def _resolve_pre_refiner_ctx(
    context: torch.Tensor,
    payload: dict[str, Any],
    call_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor | None:
    if context.ndim == 3 and int(context.shape[-1]) == ENCODER_DIM:
        return context
    kw = call_kwargs or {}
    tags_raw = payload.get("text_token_tags")
    expected_len = int(context.shape[1])
    for cand in (
        kw.get("h3_pre_refiner_ctx"),
        payload.get("h3_pre_refiner_ctx"),
        getattr(tags_raw, "_h3_pre_refiner_ctx", None),
    ):
        if isinstance(cand, dict):
            cand = cand.get("ctx")
        if not isinstance(cand, torch.Tensor):
            continue
        if cand.ndim == 2:
            cand = cand.unsqueeze(0)
        if cand.ndim == 3 and int(cand.shape[-1]) == ENCODER_DIM and int(cand.shape[1]) == expected_len:
            return cand
    return None


def _sync_call_kwargs_for_main(
    call_kwargs: dict[str, Any],
    *,
    main_ctx: torch.Tensor,
    ctx_base: torch.Tensor,
    video_cond_raw: torch.Tensor | None,
    video_cond_tags: torch.Tensor | None,
    keep_vid_mask: torch.Tensor | None,
) -> dict[str, Any]:
    if main_ctx is ctx_base:
        return call_kwargs
    next_kwargs = dict(call_kwargs)
    payload = call_kwargs.get("minimax_payload")
    next_payload = dict(payload) if isinstance(payload, dict) else {}
    l_main = int(main_ctx.shape[1])
    l_base = int(ctx_base.shape[1])

    if video_cond_raw is not None:
        if video_cond_tags is not None and int(video_cond_tags.numel()) == l_main:
            next_payload["text_token_tags"] = video_cond_tags.flatten()
        else:
            existing_tags = next_payload.get("text_token_tags")
            if isinstance(existing_tags, torch.Tensor) and int(existing_tags.numel()) != l_main:
                next_payload.pop("text_token_tags", None)
    elif keep_vid_mask is not None:
        existing_tags = next_payload.get("text_token_tags")
        if isinstance(existing_tags, torch.Tensor) and int(existing_tags.numel()) == int(keep_vid_mask.numel()):
            next_payload["text_token_tags"] = existing_tags.flatten()[keep_vid_mask.to(device=existing_tags.device)]

    if l_main != l_base:
        next_payload.pop("layout", None)
    next_kwargs["minimax_payload"] = next_payload
    return next_kwargs


def _split_context_by_section(
    base_ctx: torch.Tensor,
    pre_refiner_ctx: torch.Tensor | None,
    payload: dict[str, Any],
    call_kwargs: dict[str, Any],
    *,
    section_split: str,
    dm: Any,
    transformer_options: dict[str, Any],
    want_vid: bool,
    want_aud: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.BoolTensor | None, torch.BoolTensor | None, bool]:
    """Compute `(ctx_vid_or_none, ctx_aud_or_none, keep_vid_mask, keep_aud_mask, used_pre_refiner)`."""
    if section_split == "off":
        return None, None, None, None, False
    span_masks = _resolve_span_masks(payload, call_kwargs)
    if span_masks is None:
        return None, None, None, None, False

    scene_t = span_masks.get("scene")
    vision_t = span_masks.get("vision")
    if not isinstance(scene_t, torch.Tensor) or not isinstance(vision_t, torch.Tensor):
        return None, None, None, None, False
    L = int(base_ctx.shape[1])
    if int(scene_t.numel()) != L or int(vision_t.numel()) != L:
        return None, None, None, None, False
    if not bool(getattr(scene_t, "_h3_has_sections", False)):
        return None, None, None, None, False

    used_pre = pre_refiner_ctx is not None
    src_ctx = (
        pre_refiner_ctx.to(device=base_ctx.device, dtype=base_ctx.dtype)
        if pre_refiner_ctx is not None
        else base_ctx
    )

    vid_ctx: torch.Tensor | None = None
    aud_ctx: torch.Tensor | None = None
    keep_vid_mask: torch.BoolTensor | None = None
    keep_aud_mask: torch.BoolTensor | None = None
    device = base_ctx.device

    if want_vid:
        # Strict section mask with zero fallback to scene_only on non-sectioned inputs.
        snd_mask = _spans.inference_mask(span_masks, "overall_soundscape+non_diegetic_music").to(device=device)
        keep_vid = (~snd_mask) | vision_t.to(device=device)
        keep_vid[0] = True  # preserve row-0 Qwen3-VL attention sink
        if not bool(keep_vid.all()) and int(keep_vid.sum().item()) > 0:
            vid_raw = src_ctx[:, keep_vid, :]
            vid_ctx = _steer_hooks.refine_text_context(
                dm, vid_raw, transformer_options, caller_name="MiniMaxH3QueryRouter(section_split)"
            )
            keep_vid_mask = keep_vid

    if want_aud and section_split == "sections_by_modality":
        scene_only_mask = _spans.inference_mask(span_masks, "scene_only").to(device=device)
        keep_aud = ~scene_only_mask
        keep_aud[0] = True  # preserve row-0 attention sink alongside <d> dialogue & soundscape
        if not bool(keep_aud.all()) and int(keep_aud.sum().item()) > 0:
            aud_raw = src_ctx[:, keep_aud, :]
            aud_ctx = _steer_hooks.refine_text_context(
                dm, aud_raw, transformer_options, caller_name="MiniMaxH3QueryRouter(section_split)"
            )
            keep_aud_mask = keep_aud

    return vid_ctx, aud_ctx, keep_vid_mask, keep_aud_mask, used_pre


def _make_query_router_wrapper(
    *,
    section_split: str,
    video_block_band: str,
    audio_block_band: str,
    start_percent: float,
    end_percent: float,
    audio_cond_raw: torch.Tensor | None,
    audio_cond_tags: torch.Tensor | None,
    video_cond_raw: torch.Tensor | None,
    video_cond_tags: torch.Tensor | None,
    workspace: _steer_hooks.H3SteerStepWorkspace,
    dm: Any,
    ref_cache: dict[str, Any],
    aud_to_vid_gain: float = 1.0,
    audio_channel: str = "both",
    audio_time_start: float = 0.0,
    audio_time_end: float = 1.0,
    ref_img_gain: float = 1.0,
    ref_img_block_band: str = "mid_late_20_49",
    ref_audio_stream: str = "both",
):
    logged_modes: set[str] = set()
    _ = (ref_img_gain, ref_img_block_band, ref_audio_stream)  # disarmed; kept for schema parity

    def wrapper(executor, x, timestep, context, transformer_options, **call_kwargs):
        if not _steer_hooks.step_in_window(transformer_options, timestep, start_percent, end_percent):
            return executor(x, timestep, context, transformer_options, **call_kwargs)

        has_audio_stream = isinstance(x, (list, tuple)) and len(x) >= 2
        payload = call_kwargs.get("minimax_payload") or {}
        if not isinstance(payload, dict):
            payload = {}

        pre_refiner_ctx = _resolve_pre_refiner_ctx(context, payload, call_kwargs) if section_split != "off" else None
        cache_entries = ref_cache.setdefault("entries", OrderedDict())
        cache_key = (
            id(context),
            id(pre_refiner_ctx) if pre_refiner_ctx is not None else 0,
            str(context.device),
            str(context.dtype),
            bool(has_audio_stream),
        )
        cached = cache_entries.get(cache_key)
        if (
            cached is not None
            and cached["context"] is context
            and cached["pre_ctx"] is pre_refiner_ctx
        ):
            cache_entries.move_to_end(cache_key)
            ctx_base = cached["ctx_base"]
            ctx_vid = cached["ctx_vid"]
            ctx_aud = cached["ctx_aud"]
            keep_vid_mask = cached["keep_vid_mask"]
            keep_aud_mask = cached["keep_aud_mask"]
        else:
            with _steer_hooks.pause_malloc_graph_ctx():
                ctx_base = _steer_hooks.refine_text_context(
                    dm, context, transformer_options, caller_name="MiniMaxH3QueryRouter"
                )
                ctx_vid_sec, ctx_aud_sec, keep_vid_mask, keep_aud_mask, used_pre = _split_context_by_section(
                    context,
                    pre_refiner_ctx,
                    payload,
                    call_kwargs,
                    section_split=section_split,
                    dm=dm,
                    transformer_options=transformer_options,
                    want_vid=(video_cond_raw is None),
                    want_aud=(audio_cond_raw is None and has_audio_stream),
                )
                if section_split != "off" and (ctx_vid_sec is not None or ctx_aud_sec is not None):
                    mode_tag = f"{section_split}:{'pre_refiner' if used_pre else 'post_refiner_fallback'}"
                    if mode_tag not in logged_modes:
                        logged_modes.add(mode_tag)
                        logger.info("[h3] MiniMaxH3QueryRouter section_split=%s", mode_tag)
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

            cache_entries[cache_key] = {
                "context": context,
                "pre_ctx": pre_refiner_ctx,
                "ctx_base": ctx_base,
                "ctx_vid": ctx_vid,
                "ctx_aud": ctx_aud,
                "keep_vid_mask": keep_vid_mask,
                "keep_aud_mask": keep_aud_mask,
            }
            cache_entries.move_to_end(cache_key)
            while len(cache_entries) > _REF_CACHE_MAX:
                cache_entries.popitem(last=False)

        if (
            ctx_vid is None
            and ctx_aud is None
            and float(aud_to_vid_gain) >= (1.0 - _FULL_GAIN_EPS)
        ):
            return executor(x, timestep, ctx_base, transformer_options, **call_kwargs)

        base_tags_raw = payload.get("text_token_tags")
        base_tags = (
            base_tags_raw.detach().to(device="cpu", dtype=torch.long).flatten()
            if isinstance(base_tags_raw, torch.Tensor)
            else None
        )
        vid_tags: torch.Tensor | None = None
        if ctx_vid is not None:
            if video_cond_raw is not None:
                if video_cond_tags is not None and int(video_cond_tags.numel()) == int(ctx_vid.shape[1]):
                    vid_tags = video_cond_tags
            elif keep_vid_mask is not None and base_tags is not None and int(base_tags.numel()) == int(keep_vid_mask.numel()):
                vid_tags = base_tags[keep_vid_mask.to(device=base_tags.device)]
            elif base_tags is not None and int(base_tags.numel()) == int(ctx_vid.shape[1]):
                vid_tags = base_tags

        aud_tags: torch.Tensor | None = None
        if ctx_aud is not None:
            if audio_cond_raw is not None:
                if audio_cond_tags is not None and int(audio_cond_tags.numel()) == int(ctx_aud.shape[1]):
                    aud_tags = audio_cond_tags
            elif keep_aud_mask is not None and base_tags is not None and int(base_tags.numel()) == int(keep_aud_mask.numel()):
                aud_tags = base_tags[keep_aud_mask.to(device=base_tags.device)]
            elif base_tags is not None and int(base_tags.numel()) == int(ctx_aud.shape[1]):
                aud_tags = base_tags

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
            aud_outside_uses_shadow_a: bool = False,
            shadow_a_tags: torch.Tensor | None = None,
            shadow_b_tags: torch.Tensor | None = None,
            invoke_call_kwargs: dict[str, Any] = call_kwargs,
        ) -> Any:
            if isinstance(transformer_options, dict) and transformer_options.get("_h3_steer_runtime") is not None:
                raise ValueError(
                    "MiniMaxH3QueryRouter cannot stack with another active 1x H3SteerRuntime "
                    "(e.g. hook-mode MiniMaxH3Steer with streams!='both' or block_band!='all'); "
                    "use MiniMaxH3Steer with streams='both' & block_band='all', or split_backend='legacy_2x', "
                    "or disable one of the two 1x attention hooks."
                )
            if isinstance(transformer_options, dict) and transformer_options.get("_h3_spatial_qr_rt") is not None:
                raise ValueError(
                    "MiniMaxH3QueryRouter cannot share the same sampler step with an active "
                    "MiniMaxH3SpatialQueryRouter; separate their start/end_percent windows or use "
                    "MiniMaxH3SpatialQueryRouter's audio_zone ('base', 'region', 'binaural_split')."
                )
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
                    aud_to_vid_gain=float(aud_to_vid_gain),
                    audio_channel=audio_channel,
                    audio_time_start=float(audio_time_start),
                    audio_time_end=float(audio_time_end),
                    shadow_a_tags=shadow_a_tags,
                    shadow_b_tags=shadow_b_tags,
                    aud_outside_uses_shadow_a=aud_outside_uses_shadow_a,
                )
                to_next = dict(transformer_options or {})
                to_next["_h3_steer_runtime"] = rt
                to_next["optimized_attention_override"] = _steer_hooks.make_steer_attention_override(
                    to_next.get("optimized_attention_override")
                )
            out_res = executor(x, timestep, rt.main_ctx, to_next, **invoke_call_kwargs)
            sink = to_next.get("_h3_steer_stats_sink")
            if callable(sink):
                sink(dict(rt.stats))
            elif isinstance(sink, list):
                sink.append(dict(rt.stats))
            return out_res

        # Case A: Audio-only text route or standalone coupling gating
        if ctx_vid is None or len(vid_band) == 0:
            if ctx_aud is None or not has_audio_stream or len(aud_band) == 0:
                if has_audio_stream and len(aud_band) > 0 and float(aud_to_vid_gain) < (1.0 - _FULL_GAIN_EPS):
                    return _invoke_rt(
                        mode_id=3 if aud_band != all_blocks else 2,
                        main_ctx=ctx_base,
                        shadow_a_init=ctx_base,
                        shadow_b_init=None,
                        swap_blocks=frozenset(),
                        aud_split_blocks=aud_band,
                        dual_shadow=False,
                        aud_uses_shadow_b=False,
                        shadow_a_tags=base_tags,
                    )
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
                shadow_a_tags=aud_tags,
            )

        # Case B: `ctx_vid` spans ALL blocks (supports arbitrary l_txt_vid by making ctx_vid main_ctx)
        if vid_band == all_blocks:
            main_call_kwargs = _sync_call_kwargs_for_main(
                call_kwargs,
                main_ctx=ctx_vid,
                ctx_base=ctx_base,
                video_cond_raw=video_cond_raw,
                video_cond_tags=video_cond_tags,
                keep_vid_mask=keep_vid_mask,
            )
            if not has_audio_stream:
                return executor(x, timestep, ctx_vid, transformer_options, **main_call_kwargs)
            if ctx_aud is not None and len(aud_band) > 0 and aud_band != all_blocks:
                return _invoke_rt_split_two_shadows(
                    _invoke_rt,
                    main_ctx=ctx_vid,
                    ctx_aud_inside=ctx_aud,
                    ctx_aud_outside=ctx_base,
                    aud_band=aud_band,
                    all_blocks=all_blocks,
                    main_call_kwargs=main_call_kwargs,
                    outside_call_kwargs=call_kwargs,
                    vid_tags=vid_tags,
                    aud_tags=aud_tags,
                    base_tags=base_tags,
                )
            if ctx_aud is None:
                has_aud_gate = (
                    float(aud_to_vid_gain) < (1.0 - _FULL_GAIN_EPS)
                    or str(audio_channel or "both").strip() != "both"
                    or float(audio_time_start) > 1e-6
                    or float(audio_time_end) < 1.0 - 1e-6
                )
                if has_aud_gate and aud_band != all_blocks:
                    return _invoke_rt(
                        mode_id=3 if len(aud_band) > 0 else 2,
                        main_ctx=ctx_vid,
                        shadow_a_init=ctx_base,
                        shadow_b_init=None,
                        swap_blocks=frozenset(),
                        aud_split_blocks=aud_band,
                        dual_shadow=False,
                        aud_uses_shadow_b=False,
                        aud_unswapped_blocks=all_blocks - aud_band,
                        aud_outside_uses_shadow_a=True,
                        shadow_a_tags=base_tags,
                        invoke_call_kwargs=main_call_kwargs,
                    )
                return _invoke_rt(
                    mode_id=2,
                    main_ctx=ctx_vid,
                    shadow_a_init=ctx_base,
                    shadow_b_init=None,
                    swap_blocks=frozenset(),
                    aud_split_blocks=all_blocks,
                    dual_shadow=False,
                    aud_uses_shadow_b=False,
                    shadow_a_tags=base_tags,
                    invoke_call_kwargs=main_call_kwargs,
                )
            if len(aud_band) == 0:
                return _invoke_rt(
                    mode_id=2,
                    main_ctx=ctx_vid,
                    shadow_a_init=ctx_base,
                    shadow_b_init=None,
                    swap_blocks=frozenset(),
                    aud_split_blocks=frozenset(),
                    dual_shadow=False,
                    aud_uses_shadow_b=False,
                    aud_unswapped_blocks=all_blocks,
                    aud_outside_uses_shadow_a=True,
                    shadow_a_tags=base_tags,
                    invoke_call_kwargs=main_call_kwargs,
                )
            return _invoke_rt(
                mode_id=2,
                main_ctx=ctx_vid,
                shadow_a_init=ctx_aud,
                shadow_b_init=None,
                swap_blocks=frozenset(),
                aud_split_blocks=all_blocks,
                dual_shadow=False,
                aud_uses_shadow_b=False,
                shadow_a_tags=aud_tags,
                invoke_call_kwargs=main_call_kwargs,
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
                shadow_a_tags=vid_tags,
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
                shadow_a_tags=vid_tags,
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
            shadow_a_tags=vid_tags,
            shadow_b_tags=aud_tags,
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
    main_call_kwargs: dict[str, Any],
    outside_call_kwargs: dict[str, Any],
    vid_tags: torch.Tensor | None = None,
    aud_tags: torch.Tensor | None = None,
    base_tags: torch.Tensor | None = None,
) -> Any:
    """Route `Q_vid` to `main_ctx` across all blocks, `Q_aud` to `ctx_aud_inside` on `aud_band`,
    and `Q_aud` to `ctx_aud_outside` outside `aud_band` (works for both equal and unequal `l_vid`)."""
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
            shadow_a_tags=vid_tags,
            shadow_b_tags=aud_tags,
            invoke_call_kwargs=outside_call_kwargs,
        )
    return invoke_fn(
        mode_id=3,
        main_ctx=main_ctx,
        shadow_a_init=ctx_aud_outside,
        shadow_b_init=ctx_aud_inside,
        swap_blocks=frozenset(),
        aud_split_blocks=aud_band,
        dual_shadow=True,
        aud_uses_shadow_b=True,
        aud_unswapped_blocks=all_blocks - aud_band,
        aud_outside_uses_shadow_a=True,
        shadow_a_tags=base_tags,
        shadow_b_tags=aud_tags,
        invoke_call_kwargs=main_call_kwargs,
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
    aud_to_vid_gain: float = 1.0,
    audio_channel: str = "both",
    audio_time_start: float = 0.0,
    audio_time_end: float = 1.0,
    ref_img_gain: float = 1.0,
    ref_img_block_band: str = "mid_late_20_49",
    ref_audio_stream: str = "both",
) -> Any:
    """Clone `model` and attach a `1x` Cross-Modal Query Router wrapper + `double_block` hooks."""
    sec_mode = str(section_split or "off").strip()
    if sec_mode not in SECTION_SPLIT_MODES:
        raise ValueError(f"section_split must be one of {SECTION_SPLIT_MODES}, got {section_split!r}")
    norm_vid_band = _steer_hooks.validate_block_band(video_block_band)
    norm_aud_band = _steer_hooks.validate_block_band(audio_block_band)
    norm_ref_img_band = _steer_hooks.validate_block_band(ref_img_block_band)
    s_pct = float(start_percent)
    e_pct = float(end_percent)
    if not (0.0 <= s_pct < e_pct <= 1.0):
        raise ValueError(
            f"start_percent/end_percent must satisfy 0.0 <= start_percent < end_percent <= 1.0, got [{s_pct}, {e_pct}]"
        )
    a2v_gain = float(aud_to_vid_gain)
    if not (0.0 <= a2v_gain <= 1.0):
        raise ValueError(f"aud_to_vid_gain must lie in [0.0, 1.0], got {aud_to_vid_gain!r}")
    ch_mode = str(audio_channel or "both").strip()
    if ch_mode not in AUDIO_CHANNEL_MODES:
        raise ValueError(f"audio_channel must be one of {AUDIO_CHANNEL_MODES}, got {audio_channel!r}")
    t0_aud = float(audio_time_start)
    t1_aud = float(audio_time_end)
    if not (0.0 <= t0_aud < t1_aud <= 1.0):
        raise ValueError(f"audio_time_start/end must satisfy 0.0 <= start < end <= 1.0, got [{t0_aud}, {t1_aud}]")
    r_img_gain = float(ref_img_gain)
    if not (0.0 <= r_img_gain <= 1.0):
        raise ValueError(f"ref_img_gain must lie in [0.0, 1.0], got {ref_img_gain!r}")
    r_aud_mode = str(ref_audio_stream or "both").strip()
    if r_aud_mode not in REF_AUDIO_STREAM_MODES:
        raise ValueError(f"ref_audio_stream must be one of {REF_AUDIO_STREAM_MODES}, got {ref_audio_stream!r}")

    if r_img_gain < (1.0 - _FULL_GAIN_EPS):
        logger.warning(
            "[h3] MiniMaxH3QueryRouter ref_img_gain=%.3f is deprecated/disarmed (safe no-op; keeps ref_img intact)",
            r_img_gain,
        )
    if r_aud_mode == "audio_only":
        logger.warning(
            "[h3] MiniMaxH3QueryRouter ref_audio_stream='audio_only' is deprecated/disarmed (safe no-op; keeps ref_audio intact)",
        )

    audio_cond_raw = _extract_cond_tensor(audio_conditioning, name="audio_conditioning") if audio_conditioning is not None else None
    audio_cond_tags = _extract_cond_tags(audio_conditioning) if audio_conditioning is not None else None
    video_cond_raw = _extract_cond_tensor(video_conditioning, name="video_conditioning") if video_conditioning is not None else None
    video_cond_tags = _extract_cond_tags(video_conditioning) if video_conditioning is not None else None

    if (
        sec_mode == "off"
        and audio_cond_raw is None
        and video_cond_raw is None
        and a2v_gain >= (1.0 - 1e-6)
    ):
        return model

    m = model.clone()
    dm = getattr(getattr(m, "model", None), "diffusion_model", None)
    if not _steer_hooks.is_hookable_dit(dm):
        raise ValueError("MiniMaxH3QueryRouter requires a MiniMax H3 model with hookable DiT blocks")
    if _is_h3_image_model(dm):
        raise ValueError("MiniMaxH3QueryRouter targets joint audio-video MiniMax H3 models")

    _steer_hooks.parse_block_band(norm_vid_band, len(dm.blocks))
    _steer_hooks.parse_block_band(norm_aud_band, len(dm.blocks))
    _steer_hooks.parse_block_band(norm_ref_img_band, len(dm.blocks))

    ws = _steer_hooks.install_steer_block_hooks(
        m, dm, key_prefix=ROUTER_WRAPPER_KEY, node_label="MiniMaxH3QueryRouter"
    )
    from comfy.patcher_extension import CallbacksMP, WrappersMP

    ref_cache: dict[str, Any] = {}
    m.add_callback_with_key(
        CallbacksMP.ON_CLEANUP,
        f"{ROUTER_WRAPPER_KEY}_refcache",
        lambda *args, _rc=ref_cache, **kwargs: _rc.clear(),
    )
    wrappers = m.wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})
    wrappers[ROUTER_WRAPPER_KEY] = [
        _make_query_router_wrapper(
            section_split=sec_mode,
            video_block_band=norm_vid_band,
            audio_block_band=norm_aud_band,
            start_percent=s_pct,
            end_percent=e_pct,
            audio_cond_raw=audio_cond_raw,
            audio_cond_tags=audio_cond_tags,
            video_cond_raw=video_cond_raw,
            video_cond_tags=video_cond_tags,
            workspace=ws,
            dm=dm,
            ref_cache=ref_cache,
            aud_to_vid_gain=a2v_gain,
            audio_channel=ch_mode,
            audio_time_start=t0_aud,
            audio_time_end=t1_aud,
            ref_img_gain=r_img_gain,
            ref_img_block_band=norm_ref_img_band,
            ref_audio_stream=r_aud_mode,
        )
    ]
    return m


class MiniMaxH3QueryRouter(io.ComfyNode):
    """Single-pass (`1x`) Cross-Modal Query Router (`Q_vid` / `Q_aud` text, coupling, stereo/time, & ref routing)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3QueryRouter",
            display_name="MiniMax H3 Query Router",
            category="MiniMax H3/Steering",
            description=(
                "Route distinct text conditioning, Q_aud->K_vid coupling gain, stereo/time slices, and multimodal "
                "reference attenuation across DiT block bands in a single 1x pass."
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
                io.Float.Input(
                    "aud_to_vid_gain",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    optional=True,
                    tooltip=(
                        "Cross-modal coupling weight [0.0, 1.0] for audio queries attending to video keys inside "
                        "audio_block_band (`log(gain)` bias; 1.0 = unmasked, 0.0 = blind to video and blocks the "
                        "1-hop K_vid->text->Q_aud relay). On a single prompt with section_split='off', visual words "
                        "still inside the prompt text remain readable by audio unless section_split='sections_by_modality' "
                        "or clean audio_conditioning is wired."
                    ),
                ),
                io.Combo.Input(
                    "audio_channel",
                    options=list(AUDIO_CHANNEL_MODES),
                    default="both",
                    optional=True,
                    tooltip=(
                        "Stereo target channel(s) routed to audio_conditioning: 'both', 'left', or 'right'. "
                        "Use audio_block_band='late_30_49' (or 'mid_late_20_49') for phase-synced binaural routing."
                    ),
                ),
                io.Float.Input(
                    "audio_time_start",
                    default=0.0,
                    min=0.0,
                    max=0.99,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "Normalized audio timeline start [0.0, 1.0) for 1x temporal Q_aud routing. "
                        "Use audio_block_band='late_30_49' (or 'mid_late_20_49') to avoid mid-clip script restarts."
                    ),
                ),
                io.Float.Input(
                    "audio_time_end",
                    default=1.0,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "Normalized audio timeline end (0.0, 1.0] for 1x temporal Q_aud routing. "
                        "Use audio_block_band='late_30_49' (or 'mid_late_20_49') to avoid mid-clip script restarts."
                    ),
                ),
                io.Float.Input(
                    "ref_img_gain",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    optional=True,
                    tooltip="(Deprecated no-op; retained for workflow compatibility; in-place V[ref_img] scaling disarmed.)",
                ),
                io.Combo.Input(
                    "ref_img_block_band",
                    options=list(STEER_BLOCK_BANDS),
                    default="mid_late_20_49",
                    optional=True,
                    tooltip="(Deprecated no-op; retained for workflow compatibility.)",
                ),
                io.Combo.Input(
                    "ref_audio_stream",
                    options=list(REF_AUDIO_STREAM_MODES),
                    default="both",
                    optional=True,
                    tooltip="(Deprecated no-op; retained for workflow compatibility; in-place V[ref_audio] muting disarmed.)",
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
        aud_to_vid_gain: float = 1.0,
        audio_channel: str = "both",
        audio_time_start: float = 0.0,
        audio_time_end: float = 1.0,
        ref_img_gain: float = 1.0,
        ref_img_block_band: str = "mid_late_20_49",
        ref_audio_stream: str = "both",
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
                aud_to_vid_gain=aud_to_vid_gain,
                audio_channel=audio_channel,
                audio_time_start=audio_time_start,
                audio_time_end=audio_time_end,
                ref_img_gain=ref_img_gain,
                ref_img_block_band=ref_img_block_band,
                ref_audio_stream=ref_audio_stream,
            )
        )
