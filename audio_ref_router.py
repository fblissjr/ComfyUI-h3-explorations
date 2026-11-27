"""Single-pass (`1x` NFE) audio-reference block-depth router (`MiniMaxH3AudioRefRouter`).

Separates melody/timing conditioning from vocal/instrument timbre conditioning on
MiniMax-H3 audio-video generation without patching ComfyUI core and without mutating
key/value buffers in place:

1. **Per-block `Q_aud` side-SDPA routing:** In early DiT blocks (`[0, split_block)`),
   target audio queries (`audio`) read the melody scaffold (`cond_audio` or `<Audio 2>`
   `ref_audio[1]`) at `melody_gain_early` and the voice/style reference (`<Audio 1>`
   `ref_audio[0]`) at `timbre_gain_early`. In late vocal-timbre blocks (`[split_block, 50)`),
   `Q_aud` applies `melody_gain_late` (default `0.0` -> `-1e4` firewall) to the melody
   scaffold and `timbre_gain_late` (default `2.0` -> `+log(2)`) to `<Audio 1>`.
   Because target audio rows are only ~1-3% of the packed sequence, exact side-SDPA on
   `Q_aud` runs at ~1.00x wall-clock cost while `MiniMaxH3Sol`, Kitchen, or Sage handle
   the rest of the sequence at full speed.
2. **Independent `melody_clean_level` softening:** Blunts sharp original-singer formants
   on ONLY the melody scaffold latents (`cond_audio` or `ref_audio[1]`) using deterministic
   seeded Gaussian noise clamped at `max(melody_clean_level, audio_stream_time)`, leaving
   `<Audio 1>` (`ref_audio[0]`) 100% clean (avoiding core's shared `audio_cond_noise_aug`).
3. **Selective `melody_end_percent` release:** Drops ONLY the melody scaffold segment
   (`cond_audio` or `ref_audio[1]`) from `PackedLayout` and `cond_audio_latents` once the
   sampler step fraction reaches `melody_end_percent`, preserving `<Audio 1>` and exact
   3D RoPE positions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import functools
import logging
import math
import weakref
from typing import Any

import torch
import torch.nn.functional as F
from comfy_api.latest import io

try:
    from . import h3_layout as _h3layout
except ImportError:
    import h3_layout as _h3layout

try:
    from steering import hooks as _steer_hooks
    from steering import nodes as _steer_nodes
except ImportError:
    from .steering import hooks as _steer_hooks
    from .steering import nodes as _steer_nodes

logger = logging.getLogger("ComfyUI-h3-explorations.audio_ref_router")

AUDIO_REF_ROUTER_WRAPPER_KEY = "minimax_h3_audio_ref_router"
MELODY_SOURCES: tuple[str, ...] = ("cond_audio", "ref_audio_2")
_GAIN_IDENTITY_EPS = 1e-6
_ZERO_GAIN_BIAS = -1e4


def _is_av_minimax_h3(dm: Any) -> bool:
    if dm is None:
        return False
    if getattr(dm, "audio_patch_proj", None) is None:
        return False
    blocks = getattr(dm, "blocks", None)
    return blocks is not None and len(blocks) > 0


def _validate_float(name: str, val: Any, *, lo: float, hi: float) -> float:
    f = float(val)
    if not math.isfinite(f) or f < lo or f > hi:
        raise ValueError(f"MiniMaxH3AudioRefRouter {name} must be in [{lo}, {hi}], got {val!r}")
    return f


def _validate_int(name: str, val: Any, *, lo: int, hi: int) -> int:
    i = int(val)
    if i < lo or i > hi:
        raise ValueError(f"MiniMaxH3AudioRefRouter {name} must be in [{lo}, {hi}], got {val!r}")
    return i


def _keyframe_audio_count(keyframes: Any, orig_layout: Any = None) -> int:
    n = sum(
        1
        for kf in (keyframes or ())
        if isinstance(kf, Mapping) and kf.get("audio_latent") is not None
    )
    if n == 0 and orig_layout is not None and hasattr(orig_layout, "segments"):
        n = sum(1 for _a, _b, k in orig_layout.segments if k == "cond_audio")
    return n


def _has_melody_segment(payload: Mapping[str, Any], melody_source: str) -> bool:
    layout = payload.get("layout")
    if melody_source == "cond_audio":
        if _keyframe_audio_count(payload.get("keyframes"), layout) > 0:
            return True
        return False
    # melody_source == "ref_audio_2"
    n_ref_aud = 0
    for r in payload.get("refs") or ():
        if isinstance(r, Mapping) and (
            r.get("audio_latent") is not None or r.get("kind") == "audio"
        ):
            n_ref_aud += 1
    if n_ref_aud >= 2:
        return True
    if layout is not None and hasattr(layout, "segments"):
        return sum(1 for _a, _b, k in layout.segments if k == "ref_audio") >= 2
    return False


def _drop_melody_from_layout(
    orig_layout: Any,
    *,
    melody_source: str,
) -> tuple[Any, int]:
    """Drop ONLY `cond_audio` or 2nd `ref_audio` (`index 1`) from `PackedLayout`, preserving 3D RoPE."""
    if not (
        hasattr(orig_layout, "segments")
        and hasattr(orig_layout, "position_ids")
        and hasattr(orig_layout, "signature")
    ):
        return orig_layout, 0

    dropped = 0
    ref_aud_idx = 0
    drop_indices: set[int] = set()
    for seg_i, (a, b, kind) in enumerate(orig_layout.segments):
        n = int(b) - int(a)
        if melody_source == "cond_audio" and kind == "cond_audio":
            drop_indices.add(seg_i)
            dropped += n
        elif kind == "ref_audio":
            if melody_source == "ref_audio_2" and ref_aud_idx == 1:
                drop_indices.add(seg_i)
                dropped += n
            ref_aud_idx += 1

    if dropped == 0:
        return orig_layout, 0

    kept_pos: list[torch.Tensor] = []
    new_segments: list[tuple[int, int, str]] = []
    new_img_pos: list[torch.Tensor] = []
    new_img_update: list[torch.Tensor] = []
    new_audio_pos: list[torch.Tensor] = []
    new_audio_update: list[torch.Tensor] = []
    off = 0
    for seg_i, (a, b, kind) in enumerate(orig_layout.segments):
        if seg_i in drop_indices:
            continue
        n = int(b) - int(a)
        kept_pos.append(orig_layout.position_ids[int(a) : int(b)])
        new_segments.append((off, off + n, str(kind)))
        if kind in ("cond", "ref_img"):
            new_img_pos.append(torch.arange(off, off + n))
            new_img_update.append(torch.zeros(n, dtype=torch.bool))
        elif kind == "video":
            new_img_pos.append(torch.arange(off, off + n))
            new_img_update.append(torch.ones(n, dtype=torch.bool))
        elif kind in ("cond_audio", "ref_audio"):
            new_audio_pos.append(torch.arange(off, off + n))
            new_audio_update.append(torch.zeros(n, dtype=torch.bool))
        elif kind == "audio":
            new_audio_pos.append(torch.arange(off, off + n))
            new_audio_update.append(torch.ones(n, dtype=torch.bool))
        off += n

    released = copy.copy(orig_layout)
    released.seq_len = off
    released.position_ids = torch.cat(kept_pos, dim=0)
    released.img_pos = torch.cat(new_img_pos) if new_img_pos else torch.empty(0, dtype=torch.long)
    released.img_update = (
        torch.cat(new_img_update) if new_img_update else torch.empty(0, dtype=torch.bool)
    )
    released.audio_pos = (
        torch.cat(new_audio_pos) if new_audio_pos else torch.empty(0, dtype=torch.long)
    )
    released.audio_update = (
        torch.cat(new_audio_update) if new_audio_update else torch.empty(0, dtype=torch.bool)
    )
    released.signature = orig_layout.signature
    released.segments = new_segments
    return released, dropped


def _strip_melody_payload_entries(
    payload: Mapping[str, Any],
    orig_layout: Any,
    *,
    melody_source: str,
) -> dict[str, Any]:
    """Return a shallow copy of `payload` with ONLY the melody scaffold entries stripped."""
    new_payload = dict(payload)
    n_kf_aud = _keyframe_audio_count(payload.get("keyframes"), orig_layout)
    orig_aud_lats = list(payload.get("cond_audio_latents") or ())

    if melody_source == "cond_audio":
        kept_kf: list[dict[str, Any]] = []
        for kf in payload.get("keyframes") or ():
            if not isinstance(kf, Mapping):
                continue
            if kf.get("latent") is not None:
                entry = dict(kf)
                entry.pop("audio_latent", None)
                kept_kf.append(entry)
        if kept_kf:
            new_payload["keyframes"] = kept_kf
        else:
            new_payload.pop("keyframes", None)
        new_payload["cond_audio_latents"] = orig_aud_lats[n_kf_aud:] if n_kf_aud > 0 else orig_aud_lats
        return new_payload

    # melody_source == "ref_audio_2": drop only 2nd audio-carrying ref (index 1)
    kept_refs: list[dict[str, Any]] = []
    ref_aud_idx = 0
    dropped_ref = False
    for blk in payload.get("refs") or ():
        if not isinstance(blk, Mapping):
            continue
        kind = blk.get("kind")
        has_ref_aud = blk.get("audio_latent") is not None or (
            kind in ("audio", "video_audio") and int(blk.get("ref_audio_t", 0) or 0) > 0
        )
        if has_ref_aud:
            if ref_aud_idx == 1:
                dropped_ref = True
                if kind == "video_audio" and blk.get("latent") is not None:
                    vid_only = {
                        k: v for k, v in blk.items() if k not in ("audio_latent", "ref_audio_t")
                    }
                    vid_only["kind"] = "video"
                    vid_only["ref_audio_t"] = 0
                    kept_refs.append(vid_only)
                ref_aud_idx += 1
                continue
            ref_aud_idx += 1
        kept_refs.append(dict(blk))

    if kept_refs:
        new_payload["refs"] = kept_refs
    else:
        new_payload.pop("refs", None)

    drop_lat_idx = n_kf_aud + 1
    if dropped_ref and 0 <= drop_lat_idx < len(orig_aud_lats):
        new_payload["cond_audio_latents"] = [
            lat for i, lat in enumerate(orig_aud_lats) if i != drop_lat_idx
        ]
    return new_payload


def _soften_melody_latents(
    payload: Mapping[str, Any],
    orig_layout: Any,
    *,
    melody_source: str,
    eff_clean: float,
) -> dict[str, Any]:
    """Mix deterministic Gaussian noise into ONLY melody scaffold latents, keeping `<Audio 1>` clean."""
    orig_aud_lats = list(payload.get("cond_audio_latents") or ())
    if not orig_aud_lats or eff_clean >= 1.0 - _GAIN_IDENTITY_EPS:
        return dict(payload)

    n_kf_aud = _keyframe_audio_count(payload.get("keyframes"), orig_layout)
    if melody_source == "cond_audio":
        target_indices = set(range(min(n_kf_aud, len(orig_aud_lats))))
    else:
        idx = n_kf_aud + 1
        target_indices = {idx} if 0 <= idx < len(orig_aud_lats) else set()

    if not target_indices:
        return dict(payload)

    base_seed = int(payload.get("seed", 0)) + 17
    new_lats: list[Any] = []
    for i, z in enumerate(orig_aud_lats):
        if i in target_indices and torch.is_tensor(z):
            gen = torch.Generator("cpu").manual_seed(base_seed + i)
            noise = torch.randn(z.shape, generator=gen, dtype=torch.float32).to(device=z.device)
            z_soft = (float(eff_clean) * z.to(torch.float32) + (1.0 - float(eff_clean)) * noise).to(
                dtype=z.dtype
            )
            new_lats.append(z_soft)
        else:
            new_lats.append(z)

    new_payload = dict(payload)
    new_payload["cond_audio_latents"] = new_lats
    return new_payload


def _gain_to_bias(gain: float) -> float:
    g = float(gain)
    if g <= _GAIN_IDENTITY_EPS:
        return _ZERO_GAIN_BIAS
    return float(math.log(g))


def _gather_seq(t: torch.Tensor, spans: Sequence[tuple[int, int]], seq_axis: int) -> torch.Tensor:
    if len(spans) == 1:
        a, b = spans[0]
        return t.narrow(seq_axis, a, b - a)
    return torch.cat([t.narrow(seq_axis, a, b - a) for a, b in spans], dim=seq_axis)


def _build_q_aud_bias(
    seq_len: int,
    mel_spans: Sequence[tuple[int, int]],
    mel_gain: float,
    tim_spans: Sequence[tuple[int, int]],
    tim_gain: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    apply_mel = bool(mel_spans) and abs(float(mel_gain) - 1.0) > _GAIN_IDENTITY_EPS
    apply_tim = bool(tim_spans) and abs(float(tim_gain) - 1.0) > _GAIN_IDENTITY_EPS
    if not apply_mel and not apply_tim:
        return None
    bias = torch.zeros((1, 1, 1, int(seq_len)), device=device, dtype=dtype)
    if apply_mel:
        b_mel = _gain_to_bias(mel_gain)
        for a, b in mel_spans:
            bias[:, :, :, int(a) : int(b)] = b_mel
    if apply_tim:
        b_tim = _gain_to_bias(tim_gain)
        for a, b in tim_spans:
            bias[:, :, :, int(a) : int(b)] = b_tim
    return bias


class H3AudioRefRouterRuntime:
    """Per-step runtime computing biased `Q_aud` side-SDPA without mutating `K` or `V`."""

    def __init__(
        self,
        *,
        melody_source: str,
        split_block: int,
        melody_gain_early: float,
        melody_gain_late: float,
        timbre_gain_early: float,
        timbre_gain_late: float,
    ) -> None:
        self.melody_source = str(melody_source)
        self.split_block = int(split_block)
        self.melody_gain_early = float(melody_gain_early)
        self.melody_gain_late = float(melody_gain_late)
        self.timbre_gain_early = float(timbre_gain_early)
        self.timbre_gain_late = float(timbre_gain_late)
        self.active = True
        self._pending_aud_bhnd: torch.Tensor | None = None
        self._pending_au_spans: tuple[tuple[int, int], ...] = ()
        self._pending_heads: int = 0
        self._pending_head_dim: int = 0
        self.stats: dict[str, int] = {
            "blocks_routed": 0,
            "blocks_early": 0,
            "blocks_late": 0,
            "blocks_identity_skipped": 0,
        }

    def gains_for_block(self, block_idx: int) -> tuple[float, float, bool]:
        is_early = int(block_idx) < self.split_block
        if is_early:
            return self.melody_gain_early, self.timbre_gain_early, True
        return self.melody_gain_late, self.timbre_gain_late, False

    def compute_pre_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        options: Any,
        *,
        seq_axis: int = 2,
    ) -> bool:
        """Compute side-SDPA for `Q_aud` only; returns True if `splice_post_attention` should run."""
        self.reset_pending()
        if not self.active or not isinstance(options, dict):
            return False
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            return False
        seq_len = int(q.shape[seq_axis])
        if int(k.shape[seq_axis]) != seq_len or int(v.shape[seq_axis]) != seq_len:
            return False

        blk = _h3layout.block_index(options, seq_len)
        if blk is None:
            return False
        segs = _h3layout.segments(options, seq_len)
        if not segs:
            return False

        mel_gain, tim_gain, is_early = self.gains_for_block(blk)
        if (
            abs(mel_gain - 1.0) <= _GAIN_IDENTITY_EPS
            and abs(tim_gain - 1.0) <= _GAIN_IDENTITY_EPS
        ):
            self.stats["blocks_identity_skipped"] += 1
            return False

        au_spans = tuple((a, b) for a, b, kind in segs if kind == "audio" and b > a)
        if not au_spans:
            return False

        ref_aud_spans = [(a, b) for a, b, kind in segs if kind == "ref_audio" and b > a]
        if self.melody_source == "cond_audio":
            mel_spans = tuple((a, b) for a, b, kind in segs if kind == "cond_audio" and b > a)
        else:
            mel_spans = (ref_aud_spans[1],) if len(ref_aud_spans) >= 2 else ()
        tim_spans = (ref_aud_spans[0],) if len(ref_aud_spans) >= 1 else ()

        bias = _build_q_aud_bias(
            seq_len,
            mel_spans,
            mel_gain,
            tim_spans,
            tim_gain,
            device=q.device,
            dtype=q.dtype,
        )
        if bias is None:
            self.stats["blocks_identity_skipped"] += 1
            return False

        if seq_axis == 1:
            # NHD [B, S, H, D] -> BHND [B, H, S, D]
            q_aud = _gather_seq(q, au_spans, 1).transpose(1, 2).contiguous()
            k_bh = k.transpose(1, 2)
            v_bh = v.transpose(1, 2)
            heads = int(q.shape[2])
            head_dim = int(q.shape[3])
        else:
            # BHND [B, H, S, D]
            q_aud = _gather_seq(q, au_spans, 2).contiguous()
            k_bh = k
            v_bh = v
            heads = int(q.shape[1])
            head_dim = int(q.shape[3])

        aud_out = F.scaled_dot_product_attention(
            q_aud,
            k_bh,
            v_bh,
            attn_mask=bias,
            dropout_p=0.0,
            is_causal=False,
        )
        self._pending_aud_bhnd = aud_out
        self._pending_au_spans = au_spans
        self._pending_heads = heads
        self._pending_head_dim = head_dim
        self.stats["blocks_routed"] += 1
        if is_early:
            self.stats["blocks_early"] += 1
        else:
            self.stats["blocks_late"] += 1
        return True

    def splice_post_attention(
        self,
        out_main: torch.Tensor,
        *,
        nhd: bool = False,
        skip_output_reshape: bool = False,
    ) -> None:
        aud_bhnd = self._pending_aud_bhnd
        spans = self._pending_au_spans
        heads = self._pending_heads
        head_dim = self._pending_head_dim
        self.reset_pending()
        if aud_bhnd is None or not spans:
            return
        total_aud = sum(b - a for a, b in spans)
        off = 0
        if nhd:
            # out_main: [B, S, H, D]
            src = aud_bhnd.transpose(1, 2).to(dtype=out_main.dtype)
            for a, b in spans:
                n_seg = b - a
                out_main[:, a:b, :, :].copy_(src[:, off : off + n_seg, :, :])
                off += n_seg
        elif skip_output_reshape:
            # out_main: [B, H, S, D]
            src = aud_bhnd.to(dtype=out_main.dtype)
            for a, b in spans:
                n_seg = b - a
                out_main[:, :, a:b, :].copy_(src[:, :, off : off + n_seg, :])
                off += n_seg
        else:
            # out_main: [B, S, H * D]
            inner = heads * head_dim
            src_2d = aud_bhnd.transpose(1, 2).reshape(int(aud_bhnd.shape[0]), total_aud, inner).to(
                dtype=out_main.dtype
            )
            for a, b in spans:
                n_seg = b - a
                out_main[:, a:b, :].copy_(src_2d[:, off : off + n_seg, :])
                off += n_seg

    def reset_pending(self) -> None:
        self._pending_aud_bhnd = None
        self._pending_au_spans = ()
        self._pending_heads = 0
        self._pending_head_dim = 0


def make_audio_ref_router_override(previous: Any = None):
    """Build call-local `optimized_attention_override` delegating cleanly to `previous`."""

    def override(
        func,
        q,
        k,
        v,
        heads,
        mask=None,
        attn_precision=None,
        skip_reshape=False,
        skip_output_reshape=False,
        **kwargs,
    ):
        options = kwargs.get("transformer_options")

        def _pass_through(q_u, k_u, v_u):
            if not torch.is_tensor(q_u):
                if previous is not None and getattr(previous, "container_function", None) is not None:
                    return previous.container_function(
                        q_u,
                        k_u,
                        v_u,
                        heads,
                        mask=mask,
                        attn_precision=attn_precision,
                        skip_reshape=skip_reshape,
                        skip_output_reshape=skip_output_reshape,
                        **kwargs,
                    )
                q_u, k_u, v_u = q_u.take(), k_u.take(), v_u.take()
            target = func if previous is None else functools.partial(previous, func)
            return target(
                q_u,
                k_u,
                v_u,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )

        rt = options.get("_h3_audio_ref_router_rt") if isinstance(options, dict) else None
        if rt is None or not getattr(rt, "active", False) or not skip_reshape or mask is not None:
            return _pass_through(q, k, v)

        q_p = q if torch.is_tensor(q) else q.peek()
        k_p = k if torch.is_tensor(k) else k.peek()
        v_p = v if torch.is_tensor(v) else v.peek()
        if q_p.ndim != 4 or k_p.ndim != 4 or v_p.ndim != 4:
            del q_p, k_p, v_p
            return _pass_through(q, k, v)

        ran_pre = False
        try:
            with _steer_hooks.pause_malloc_graph_ctx(), torch.no_grad():
                ran_pre = rt.compute_pre_attention(q_p, k_p, v_p, options, seq_axis=2)
            del q_p, k_p, v_p
            out_main = _pass_through(q, k, v)
            if ran_pre:
                with _steer_hooks.pause_malloc_graph_ctx(), torch.no_grad():
                    rt.splice_post_attention(
                        out_main, nhd=False, skip_output_reshape=bool(skip_output_reshape)
                    )
            return out_main
        except Exception:
            rt.reset_pending()
            raise

    if getattr(previous, "container_function", None) is not None:
        override.container_function = functools.partial(override, None)
    override.h3_audio_ref_router_override = True
    override.h3_previous = previous
    return override


def h3_audio_ref_router_seam_pre(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    transformer_options: Any,
    *,
    seq_axis: int = 1,
) -> bool:
    """Seam called from `attention.py::make_minimax_attn_forward` on standalone Sage/LSE paths."""
    rt = (
        transformer_options.get("_h3_audio_ref_router_rt")
        if isinstance(transformer_options, dict)
        else None
    )
    if rt is None or not getattr(rt, "active", False):
        return False
    with _steer_hooks.pause_malloc_graph_ctx(), torch.no_grad():
        return bool(rt.compute_pre_attention(q, k, v, transformer_options, seq_axis=seq_axis))


def h3_audio_ref_router_seam_post(
    out_main: torch.Tensor,
    transformer_options: Any,
    *,
    nhd: bool = True,
) -> None:
    rt = (
        transformer_options.get("_h3_audio_ref_router_rt")
        if isinstance(transformer_options, dict)
        else None
    )
    if rt is None or not getattr(rt, "active", False):
        return
    with _steer_hooks.pause_malloc_graph_ctx(), torch.no_grad():
        rt.splice_post_attention(out_main, nhd=nhd, skip_output_reshape=False)


def h3_audio_ref_router_seam_reset(transformer_options: Any) -> None:
    rt = (
        transformer_options.get("_h3_audio_ref_router_rt")
        if isinstance(transformer_options, dict)
        else None
    )
    if rt is not None:
        rt.reset_pending()


def _make_audio_ref_router_wrapper(
    *,
    melody_source: str,
    split_block: int,
    melody_gain_early: float,
    melody_gain_late: float,
    timbre_gain_early: float,
    timbre_gain_late: float,
    melody_end_percent: float,
    melody_clean_level: float,
    start_percent: float,
    end_percent: float,
    default_shift_video: float = 12.0,
    default_shift_audio: float = 3.0,
):
    logged_notes: set[str] = set()
    released_layout_cache: weakref.WeakKeyDictionary[Any, dict[str, tuple[Any, int]]] = (
        weakref.WeakKeyDictionary()
    )
    any_non_identity_gain = any(
        abs(float(g) - 1.0) > _GAIN_IDENTITY_EPS
        for g in (melody_gain_early, melody_gain_late, timbre_gain_early, timbre_gain_late)
    )

    def wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
        is_av_x = isinstance(x, (list, tuple)) and len(x) >= 2
        if not is_av_x:
            return executor(x, timestep, context, transformer_options, **kwargs)

        call_to = dict(transformer_options) if isinstance(transformer_options, Mapping) else {}
        payload = kwargs.get("minimax_payload") or {}
        if not isinstance(payload, Mapping):
            payload = {}
        call_kwargs = dict(kwargs)

        has_melody = _has_melody_segment(payload, melody_source)
        melody_active = (
            _steer_hooks.step_in_window(call_to, timestep, 0.0, melody_end_percent)
            if (has_melody and melody_end_percent < 1.0)
            else True
        )
        drop_melody = bool(has_melody and not melody_active)
        soften_melody = bool(
            has_melody and not drop_melody and melody_clean_level < 1.0 - _GAIN_IDENTITY_EPS
        )

        if drop_melody or soften_melody:
            orig_layout = payload.get("layout")
            if (
                torch.is_tensor(x[0])
                and x[0].ndim == 5
                and torch.is_tensor(x[1])
                and x[1].ndim >= 1
            ):
                target_sig = (
                    int(context.shape[1]),
                    int(x[0].shape[2]),
                    (int(x[0].shape[3]) + 1) // 2 * 2,
                    (int(x[0].shape[4]) + 1) // 2 * 2,
                    int(x[1].shape[-1]),
                )
                if orig_layout is None or getattr(orig_layout, "signature", None) != target_sig:
                    from comfy.ldm.minimax.model import PackedLayout

                    orig_layout = PackedLayout(
                        *target_sig,
                        keyframes=payload.get("keyframes"),
                        refs=payload.get("refs"),
                    )

            if drop_melody:
                per_layout = None
                if orig_layout is not None:
                    try:
                        per_layout = released_layout_cache.get(orig_layout)
                    except TypeError:
                        per_layout = None
                cached = per_layout.get(melody_source) if per_layout is not None else None
                if cached is not None:
                    cached_layout, dropped_rows = cached
                    released_layout = orig_layout if cached_layout is None else cached_layout
                else:
                    released_layout, dropped_rows = _drop_melody_from_layout(
                        orig_layout, melody_source=melody_source
                    )
                    if orig_layout is not None:
                        try:
                            if per_layout is None:
                                per_layout = {}
                                released_layout_cache[orig_layout] = per_layout
                            per_layout[melody_source] = (
                                None if released_layout is orig_layout else released_layout,
                                dropped_rows,
                            )
                        except TypeError:
                            pass
                new_payload = _strip_melody_payload_entries(
                    payload, orig_layout, melody_source=melody_source
                )
                if orig_layout is not None:
                    new_payload["layout"] = released_layout
                call_kwargs["minimax_payload"] = new_payload
                if "melody_dropped" not in logged_notes:
                    logged_notes.add("melody_dropped")
                    logger.info(
                        "[h3] MiniMaxH3AudioRefRouter melody_end_percent=%.2f (%s): released %d melody rows "
                        "(seq_len %s -> %s, <Audio 1> & 3D RoPE preserved)",
                        melody_end_percent,
                        melody_source,
                        dropped_rows,
                        getattr(orig_layout, "seq_len", None),
                        getattr(released_layout, "seq_len", None),
                    )
            elif soften_melody:
                aud_t = _steer_nodes._audio_stream_time(
                    timestep,
                    call_to,
                    default_shift_video=default_shift_video,
                    default_shift_audio=default_shift_audio,
                )
                eff_clean = (
                    float(melody_clean_level)
                    if aud_t is None
                    else max(float(melody_clean_level), float(aud_t))
                )
                if eff_clean < 1.0 - _GAIN_IDENTITY_EPS:
                    new_payload = _soften_melody_latents(
                        payload,
                        orig_layout,
                        melody_source=melody_source,
                        eff_clean=eff_clean,
                    )
                    call_kwargs["minimax_payload"] = new_payload
                    if "melody_softened" not in logged_notes:
                        logged_notes.add("melody_softened")
                        logger.info(
                            "[h3] MiniMaxH3AudioRefRouter melody_clean_level=%.2f (%s): softened melody scaffold "
                            "at eff_clean=%.3f while keeping <Audio 1> ref_audio pristine",
                            melody_clean_level,
                            melody_source,
                            eff_clean,
                        )

        route_active = any_non_identity_gain and _steer_hooks.step_in_window(
            call_to, timestep, start_percent, end_percent
        )
        if not route_active:
            return executor(x, timestep, context, call_to, **call_kwargs)

        rt = H3AudioRefRouterRuntime(
            melody_source=melody_source,
            split_block=split_block,
            melody_gain_early=melody_gain_early,
            melody_gain_late=melody_gain_late,
            timbre_gain_early=timbre_gain_early,
            timbre_gain_late=timbre_gain_late,
        )
        prev_override = call_to.get("optimized_attention_override")
        call_to["_h3_audio_ref_router_rt"] = rt
        if not getattr(prev_override, "h3_audio_ref_router_override", False):
            call_to["optimized_attention_override"] = make_audio_ref_router_override(
                previous=prev_override
            )

        try:
            out = executor(x, timestep, context, call_to, **call_kwargs)
            if "routed_once" not in logged_notes and rt.stats["blocks_routed"] > 0:
                logged_notes.add("routed_once")
                logger.info(
                    "[h3] MiniMaxH3AudioRefRouter (%s, split_block=%d): routed %d blocks "
                    "(early=%d [mel=%.2f, tim=%.2f], late=%d [mel=%.2f, tim=%.2f])",
                    melody_source,
                    split_block,
                    rt.stats["blocks_routed"],
                    rt.stats["blocks_early"],
                    melody_gain_early,
                    timbre_gain_early,
                    rt.stats["blocks_late"],
                    melody_gain_late,
                    timbre_gain_late,
                )
            return out
        finally:
            rt.active = False
            rt.reset_pending()
            call_to.pop("_h3_audio_ref_router_rt", None)

    return wrapper


class MiniMaxH3AudioRefRouter(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AudioRefRouter",
            display_name="MiniMax H3 Audio Ref Router",
            category="model/conditioning/minimax",
            description=(
                "Single-pass (1x NFE) audio-reference block-depth router for music covers and "
                "vocal/style adaptation. Routes target audio queries (Q_aud) to read the melody "
                "scaffold (cond_audio or <Audio 2>) in early DiT blocks [0, split_block) and "
                "suppresses it in late vocal-timbre blocks [split_block, 50) while boosting "
                "<Audio 1> voice/instrument timbre. Also supports independent melody-scaffold "
                "cleanliness and step-fraction release without affecting <Audio 1>."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "melody_source",
                    options=list(MELODY_SOURCES),
                    default="cond_audio",
                    tooltip=(
                        "Which audio conditioning segment supplies the Song X melody scaffold: "
                        "'cond_audio' (timeline-aligned audio from MiniMaxH3GuideLatent) or "
                        "'ref_audio_2' (<Audio 2>, second ref_audio segment)."
                    ),
                ),
                io.Int.Input(
                    "split_block",
                    default=24,
                    min=0,
                    max=50,
                    step=1,
                    tooltip=(
                        "DiT block boundary separating early rhythm/melody blocks [0, split_block) "
                        "from late vocal-timbre/instrument blocks [split_block, 50). 0 applies late "
                        "gains to all blocks; 50 applies early gains to all blocks."
                    ),
                ),
                io.Float.Input(
                    "melody_gain_early",
                    default=1.0,
                    min=0.0,
                    max=4.0,
                    step=0.05,
                    tooltip=(
                        "Linear attention gain on Q_aud -> K_melody in early blocks [0, split_block). "
                        "1.0 is identity, 0.0 installs a -1e4 firewall, >1.0 boosts by +log(gain)."
                    ),
                ),
                io.Float.Input(
                    "melody_gain_late",
                    default=0.0,
                    min=0.0,
                    max=4.0,
                    step=0.05,
                    tooltip=(
                        "Linear attention gain on Q_aud -> K_melody in late blocks [split_block, 50). "
                        "Default 0.0 blocks original-singer vocal timbre from leaking into late blocks."
                    ),
                ),
                io.Float.Input(
                    "timbre_gain_early",
                    default=1.0,
                    min=0.0,
                    max=4.0,
                    step=0.05,
                    tooltip=(
                        "Linear attention gain on Q_aud -> K_ref_audio[0] (<Audio 1>, Target B) in "
                        "early blocks [0, split_block). Lowering to 0.5-0.8 reduces Target B's "
                        "original rhythm pulling against Song X's melody."
                    ),
                ),
                io.Float.Input(
                    "timbre_gain_late",
                    default=2.0,
                    min=0.0,
                    max=4.0,
                    step=0.05,
                    tooltip=(
                        "Linear attention gain on Q_aud -> K_ref_audio[0] (<Audio 1>, Target B) in "
                        "late blocks [split_block, 50). Default 2.0 (+log(2)) reinforces Target B's "
                        "vocal timbre and acoustic instrumentation."
                    ),
                ),
                io.Float.Input(
                    "melody_end_percent",
                    default=0.50,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "Step fraction [0.0, melody_end_percent) where the melody scaffold remains "
                        "packed in the sequence (1.0 keeps it on all steps; 0.50 on 6-step Pika "
                        "releases it after step 2 while preserving <Audio 1> and exact 3D RoPE)."
                    ),
                ),
                io.Float.Input(
                    "melody_clean_level",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "Independent cleanliness share applied ONLY to the melody scaffold latents "
                        "(clamped at max(melody_clean_level, audio_stream_time)), leaving <Audio 1> "
                        "pristine. Lowering to 0.65-0.80 softens sharp original-singer formants."
                    ),
                ),
                io.Float.Input(
                    "start_percent",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip="Step-fraction start of Q_aud block-depth attention routing.",
                ),
                io.Float.Input(
                    "end_percent",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip="Step-fraction end of Q_aud block-depth attention routing.",
                ),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def validate_inputs(
        cls,
        melody_source="cond_audio",
        split_block=24,
        melody_gain_early=1.0,
        melody_gain_late=0.0,
        timbre_gain_early=1.0,
        timbre_gain_late=2.0,
        melody_end_percent=0.50,
        melody_clean_level=1.0,
        start_percent=0.0,
        end_percent=1.0,
        **kwargs,
    ) -> bool | str:
        try:
            if melody_source not in MELODY_SOURCES:
                return f"melody_source must be one of {MELODY_SOURCES}, got {melody_source!r}"
            _validate_int("split_block", split_block, lo=0, hi=50)
            _validate_float("melody_gain_early", melody_gain_early, lo=0.0, hi=4.0)
            _validate_float("melody_gain_late", melody_gain_late, lo=0.0, hi=4.0)
            _validate_float("timbre_gain_early", timbre_gain_early, lo=0.0, hi=4.0)
            _validate_float("timbre_gain_late", timbre_gain_late, lo=0.0, hi=4.0)
            _validate_float("melody_end_percent", melody_end_percent, lo=0.0, hi=1.0)
            _validate_float("melody_clean_level", melody_clean_level, lo=0.0, hi=1.0)
            sp = _validate_float("start_percent", start_percent, lo=0.0, hi=1.0)
            ep = _validate_float("end_percent", end_percent, lo=0.0, hi=1.0)
            if ep < sp:
                return f"end_percent ({ep}) must be >= start_percent ({sp})"
        except Exception as exc:
            return str(exc)
        return True

    @classmethod
    def execute(
        cls,
        model,
        melody_source="cond_audio",
        split_block=24,
        melody_gain_early=1.0,
        melody_gain_late=0.0,
        timbre_gain_early=1.0,
        timbre_gain_late=2.0,
        melody_end_percent=0.50,
        melody_clean_level=1.0,
        start_percent=0.0,
        end_percent=1.0,
    ) -> io.NodeOutput:
        dm = model.get_model_object("diffusion_model")
        if not _is_av_minimax_h3(dm):
            raise ValueError(
                "MiniMaxH3AudioRefRouter requires a MiniMax H3 audio-video diffusion model"
            )
        if melody_source not in MELODY_SOURCES:
            raise ValueError(f"melody_source must be one of {MELODY_SOURCES}, got {melody_source!r}")
        sb = _validate_int("split_block", split_block, lo=0, hi=50)
        mge = _validate_float("melody_gain_early", melody_gain_early, lo=0.0, hi=4.0)
        mgl = _validate_float("melody_gain_late", melody_gain_late, lo=0.0, hi=4.0)
        tge = _validate_float("timbre_gain_early", timbre_gain_early, lo=0.0, hi=4.0)
        tgl = _validate_float("timbre_gain_late", timbre_gain_late, lo=0.0, hi=4.0)
        mep = _validate_float("melody_end_percent", melody_end_percent, lo=0.0, hi=1.0)
        mcl = _validate_float("melody_clean_level", melody_clean_level, lo=0.0, hi=1.0)
        sp = _validate_float("start_percent", start_percent, lo=0.0, hi=1.0)
        ep = _validate_float("end_percent", end_percent, lo=0.0, hi=1.0)
        if ep < sp:
            raise ValueError(f"end_percent ({ep}) must be >= start_percent ({sp})")

        m = model.clone()
        import comfy.patcher_extension

        default_sv = float(getattr(dm, "sigma_shift_video", 12.0))
        default_sa = float(getattr(dm, "sigma_shift_audio", 3.0))
        wrapper = _make_audio_ref_router_wrapper(
            melody_source=str(melody_source),
            split_block=sb,
            melody_gain_early=mge,
            melody_gain_late=mgl,
            timbre_gain_early=tge,
            timbre_gain_late=tgl,
            melody_end_percent=mep,
            melody_clean_level=mcl,
            start_percent=sp,
            end_percent=ep,
            default_shift_video=default_sv,
            default_shift_audio=default_sa,
        )
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            AUDIO_REF_ROUTER_WRAPPER_KEY,
            wrapper,
        )
        return io.NodeOutput(m)
