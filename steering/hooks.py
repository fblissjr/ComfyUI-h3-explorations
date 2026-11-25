"""Compiler-safe attention-hook runtime (`auto_1x_kv` stream split, `block_band`, and seam helpers)."""

from __future__ import annotations

import contextlib
import functools
import logging
import math
import re
from collections import OrderedDict
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

from .constants import POST_REFINER_DIM, STEER_BLOCK_BANDS, STEER_SPLIT_BACKENDS

logger = logging.getLogger("ComfyUI-h3-explorations.steer_hooks")

_BAND_RANGE_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")
_WORKSPACE_CACHE_MAX = 2
AUDIO_CHANNEL_MODES: tuple[str, ...] = ("both", "left", "right")


def pause_malloc_graph_ctx():
    """Return `comfy.model_prefetch.pause_malloc_graph()` when available, else no-op."""
    try:
        import comfy.model_prefetch as _mp

        pause_fn = getattr(_mp, "pause_malloc_graph", None)
        if callable(pause_fn):
            return pause_fn()
    except Exception:
        pass
    return contextlib.nullcontext()


def step_in_window(
    transformer_options: dict[str, Any] | None,
    timestep: Any,
    start_percent: float,
    end_percent: float,
) -> bool:
    """True when the current sampler step lies in `[start_percent, end_percent)`."""
    if start_percent >= end_percent:
        return False
    if start_percent <= 0.0 and end_percent >= 1.0:
        return True
    t_opts = transformer_options or {}
    sched = t_opts.get("sample_sigmas")
    cur = t_opts.get("sigmas")
    if cur is None and timestep is not None:
        cur = timestep
    if sched is not None and cur is not None:
        sched_t = (
            sched.detach().flatten().to(dtype=torch.float64, device="cpu")
            if isinstance(sched, torch.Tensor)
            else torch.tensor(list(sched), dtype=torch.float64)
        )
        if int(sched_t.numel()) >= 2:
            cur_val = float(
                cur.detach().flatten()[0].item()
                if isinstance(cur, torch.Tensor)
                else cur[0]
                if isinstance(cur, (list, tuple))
                else cur
            )
            if cur is timestep and cur_val > 1.0 and float(sched_t.max().item()) <= 1.0001:
                cur_val /= 1000.0
            n_steps = int(sched_t.numel()) - 1
            step_idx = int((sched_t[:n_steps] - cur_val).abs().argmin().item())
            frac = step_idx / n_steps
            return bool(start_percent <= frac < end_percent)
    if cur is not None:
        cur_val = float(
            cur.detach().flatten()[0].item()
            if isinstance(cur, torch.Tensor)
            else cur[0]
            if isinstance(cur, (list, tuple))
            else cur
        )
        if cur is timestep and 1.0 < cur_val <= 1000.0:
            cur_val /= 1000.0
        if 0.0 <= cur_val <= 1.0:
            frac = 1.0 - cur_val
            return bool(start_percent <= frac < end_percent)
    return True


def refine_text_context(
    dm: Any,
    ctx: torch.Tensor,
    transformer_options: dict[str, Any] | None = None,
    *,
    caller_name: str = "H3 query routing",
) -> torch.Tensor:
    """Project and pass pre-refiner `[B, L, 5120]` text states through `TokenRefiner` idempotently."""
    if ctx.ndim == 3 and ctx.shape[-1] == POST_REFINER_DIM:
        return ctx
    if dm is None or not (hasattr(dm, "token_refiner") and hasattr(dm, "condition_proj")):
        raise ValueError(
            f"{caller_name} received context with hidden_size={ctx.shape[-1]} "
            f"(expected {POST_REFINER_DIM}) and could not access diffusion_model.token_refiner"
        )
    # Match core `preprocess_text_embeds`: TokenRefiner runs without DiT attention overrides.
    with pause_malloc_graph_ctx(), torch.no_grad():
        return torch.stack(
            [
                dm.token_refiner(
                    dm.condition_proj(ctx[b_idx]),
                    transformer_options={},
                )
                for b_idx in range(int(ctx.shape[0]))
            ],
            dim=0,
        )


def validate_block_band(band_spec: Any) -> str:
    """Validate `block_band` preset or `'A-B'` range string without requiring model depth."""
    raw = str(band_spec if band_spec is not None else "all").strip() or "all"
    if raw in (*STEER_BLOCK_BANDS, "none"):
        return raw
    m = _BAND_RANGE_RE.match(raw)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo < 0 or hi < lo:
            raise ValueError(f"block_band range 'A-B' requires 0 <= A <= B, got {raw!r}")
        return f"{lo}-{hi}"
    raise ValueError(
        f"block_band must be one of {STEER_BLOCK_BANDS} (or 'none' / 'A-B' range), got {band_spec!r}"
    )


def parse_block_band(band_spec: Any, n_blocks: int = 50) -> frozenset[int]:
    """Resolve `block_band` preset or `'A-B'` inclusive range into block indices in `[0, n_blocks)`."""
    n_b = max(0, int(n_blocks))
    norm = validate_block_band(band_spec)
    if norm == "all":
        return frozenset(range(n_b))
    if norm == "none":
        return frozenset()
    if norm == "mid_late_20_49":
        return frozenset(range(min(20, n_b), n_b))
    if norm == "late_30_49":
        return frozenset(range(min(30, n_b), n_b))
    if norm == "early_0_19":
        return frozenset(range(0, min(20, n_b)))
    m = _BAND_RANGE_RE.match(norm)
    assert m is not None
    lo, hi = int(m.group(1)), int(m.group(2))
    if hi >= n_b and n_b > 0:
        raise ValueError(
            f"block_band range {norm!r} exceeds diffusion model depth (n_blocks={n_b})"
        )
    return frozenset(range(lo, min(hi + 1, n_b)))


def is_hookable_dit(dm: Any) -> bool:
    """True when `dm` exposes ComfyUI's H3 `DiTBlock` sequence (`blocks[i].adaln_proj`)."""
    blocks = getattr(dm, "blocks", None)
    if blocks is None:
        return False
    try:
        n = len(blocks)
    except Exception:
        return False
    if n <= 0:
        return False
    b0 = blocks[0]
    return hasattr(b0, "adaln_proj") and hasattr(b0, "attn") and hasattr(b0, "mlp")


def _slice_rope(rope_freqs: Any, tx_a: int, tx_b: int) -> Any:
    if rope_freqs is None:
        return None
    if rope_freqs.ndim >= 2 and rope_freqs.shape[1] >= tx_b:
        return rope_freqs[:, tx_a:tx_b]
    return rope_freqs[tx_a:tx_b]


def _slice_mod_segments(
    mod_segments: Sequence[tuple[int, int, Any]], tx_a: int, tx_b: int
) -> list[tuple[int, int, Any]]:
    out: list[tuple[int, int, Any]] = []
    for a, b, row in mod_segments:
        if b <= tx_a or a >= tx_b:
            continue
        sa = max(int(a), tx_a)
        sb = min(int(b), tx_b)
        if torch.is_tensor(row) and row.ndim >= 1:
            r_slice = row[sa - int(a) : sb - int(a)]
            out.append((sa - tx_a, sb - tx_a, r_slice))
        else:
            out.append((sa - tx_a, sb - tx_a, row))
    return out


def _adapt_text_mod_segments(
    mod_txt_main: Sequence[tuple[int, int, Any]],
    l_txt_main: int,
    l_txt_shadow: int,
    *,
    shadow_tags: torch.Tensor | None = None,
) -> list[tuple[int, int, Any]]:
    if not mod_txt_main:
        return []
    int_row = next((int(seg[2]) for seg in mod_txt_main if not torch.is_tensor(seg[2])), None)
    if (
        isinstance(shadow_tags, torch.Tensor)
        and int(shadow_tags.numel()) == int(l_txt_shadow)
        and int(l_txt_shadow) > 0
        and int_row is not None
    ):
        row_base = (int_row // 3) * 3
        tags_list = shadow_tags.detach().view(-1).tolist()
        runs: list[tuple[int, int, Any]] = []
        run_start = 0
        for i in range(1, int(l_txt_shadow) + 1):
            if i == int(l_txt_shadow) or int(tags_list[i]) != int(tags_list[run_start]):
                runs.append((run_start, i, row_base + int(tags_list[run_start])))
                run_start = i
        return runs
    if l_txt_shadow == l_txt_main:
        return list(mod_txt_main)
    # Untagged unequal-length text shadow must default to the text AdaLN row (tag 1),
    # never `mod_txt_main[0][2]` which is often a vision pad row (tag 0) on fl2va/ref2va.
    row = next(
        (seg[2] for seg in mod_txt_main if not torch.is_tensor(seg[2]) and (int(seg[2]) % 3) == 1),
        mod_txt_main[-1][2],
    )
    if torch.is_tensor(row) and row.ndim >= 1:
        if row.shape[0] >= l_txt_shadow:
            row_s = row[:l_txt_shadow]
        else:
            row_s = row[0:1].expand(l_txt_shadow, *row.shape[1:])
        return [(0, int(l_txt_shadow), row_s)]
    return [(0, int(l_txt_shadow), row)]


def _build_shadow_text_rope(
    dm: Any,
    l_txt_main: int,
    l_txt_shadow: int,
    device: torch.device,
    dtype: torch.dtype,
    fallback_rope: Any,
) -> Any:
    if l_txt_shadow == l_txt_main or fallback_rope is None:
        return fallback_rope
    rope_fn = getattr(dm, "rope_freqs", None)
    if callable(rope_fn):
        pos = torch.zeros(int(l_txt_shadow), 3, dtype=torch.float64)
        pos[:, 0] = torch.arange(int(l_txt_shadow), dtype=torch.float64) + float(
            int(l_txt_main) - int(l_txt_shadow)
        )
        angles = rope_fn(pos, device)
        half = angles.shape[-1] // 2
        ang = angles[:, :half]
        c, s = torch.cos(ang), torch.sin(ang)
        return torch.stack([c, -s, s, c], dim=-1).reshape(1, angles.shape[0], 1, half, 2, 2).to(dtype)
    if torch.is_tensor(fallback_rope) and fallback_rope.ndim >= 2 and fallback_rope.shape[1] >= l_txt_shadow:
        return fallback_rope[:, -int(l_txt_shadow) :]
    return None


def _slice_audio_spans(
    raw_au_spans: Sequence[tuple[int, int]],
    *,
    audio_channel: str = "both",
    audio_time_start: float = 0.0,
    audio_time_end: float = 1.0,
) -> list[tuple[int, int]]:
    """Slice channel-major `(Left[0..T-1], Right[0..T-1])` `audio` spans by channel & time window."""
    ch = str(audio_channel or "both").strip()
    if ch not in AUDIO_CHANNEL_MODES:
        raise ValueError(f"audio_channel must be one of {AUDIO_CHANNEL_MODES}, got {audio_channel!r}")
    t0 = float(audio_time_start)
    t1 = float(audio_time_end)
    if not (0.0 <= t0 < t1 <= 1.0):
        raise ValueError(f"audio_time_start/end must satisfy 0.0 <= start < end <= 1.0, got [{t0}, {t1}]")

    if ch == "both" and t0 <= 1e-6 and t1 >= 1.0 - 1e-6:
        return [(int(sa), int(sb)) for sa, sb in raw_au_spans]

    sliced: list[tuple[int, int]] = []
    for sa, sb in raw_au_spans:
        sa_i, sb_i = int(sa), int(sb)
        span_len = sb_i - sa_i
        if span_len < 2 or (span_len % 2) != 0:
            sliced.append((sa_i, sb_i))
            continue
        t_len = span_len // 2
        i0 = max(0, min(t_len - 1, int(round(t0 * t_len))))
        i1 = max(i0 + 1, min(t_len, int(round(t1 * t_len))))
        if ch == "left":
            sliced.append((sa_i + i0, sa_i + i1))
        elif ch == "right":
            sliced.append((sa_i + t_len + i0, sa_i + t_len + i1))
        elif i0 == 0 and i1 == t_len:
            sliced.append((sa_i, sb_i))
        else:
            sliced.append((sa_i + i0, sa_i + i1))
            sliced.append((sa_i + t_len + i0, sa_i + t_len + i1))
    return sliced


def _complement_spans(
    raw_spans: Sequence[tuple[int, int]],
    target_spans: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return sorted intervals inside `raw_spans` omitted from `target_spans`."""
    if not raw_spans:
        return []
    targets = sorted((int(a), int(b)) for a, b in target_spans if int(b) > int(a))
    comp: list[tuple[int, int]] = []
    for ra, rb in raw_spans:
        cur = int(ra)
        end = int(rb)
        if end <= cur:
            continue
        for ta, tb in targets:
            if tb <= cur or ta >= end:
                continue
            if ta > cur:
                comp.append((cur, min(ta, end)))
            cur = max(cur, min(tb, end))
            if cur >= end:
                break
        if cur < end:
            comp.append((cur, end))
    return comp


def _build_aud_attn_bias(
    k_len: int,
    vid_spans: Sequence[tuple[int, int]],
    blocked_aud_spans: Sequence[tuple[int, int]],
    txt_delta: int,
    tx_b: int,
    gain: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Build additive `[1, 1, 1, k_len]` logit bias for `Q_aud -> K_vid` and blocked intra-audio spans."""
    g = float(gain)
    has_vid = (g < 1.0 - 1e-6) and bool(vid_spans)
    has_aud = bool(blocked_aud_spans)
    if not has_vid and not has_aud:
        return None
    mask = torch.zeros((1, 1, 1, int(k_len)), device=device, dtype=dtype)
    if has_vid:
        bias_val = -1e4 if g <= 1e-6 else float(math.log(g))
        for va, vb in vid_spans:
            sa = int(va) + (txt_delta if int(va) >= tx_b else 0)
            sb = int(vb) + (txt_delta if int(vb) >= tx_b else 0)
            sa_c = max(0, min(int(k_len), sa))
            sb_c = max(sa_c, min(int(k_len), sb))
            if sb_c > sa_c:
                mask[:, :, :, sa_c:sb_c] = bias_val
    if has_aud:
        for aa, ab in blocked_aud_spans:
            sa = int(aa) + (txt_delta if int(aa) >= tx_b else 0)
            sb = int(ab) + (txt_delta if int(ab) >= tx_b else 0)
            sa_c = max(0, min(int(k_len), sa))
            sb_c = max(sa_c, min(int(k_len), sb))
            if sb_c > sa_c:
                mask[:, :, :, sa_c:sb_c] = -1e4
    return mask


def _apply_audio_gain_bhnd(k_bh: torch.Tensor, v_bh: torch.Tensor, options: dict[str, Any]) -> bool:
    if not isinstance(options, dict):
        return False
    if options.get("_h3_audio_gain_applied") or not options.get("minimax_h3_audio_gain"):
        return False
    from .constants import import_repo_module
    _h3_attn = import_repo_module("attention")

    _h3_attn.apply_audio_gain(k_bh, v_bh, options, 2)
    options["_h3_audio_gain_applied"] = True
    return True


def _apply_rms_rope_txt(
    block: Any,
    q_s: torch.Tensor,
    k_s: torch.Tensor,
    rope_s: Any,
    *,
    heads: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    l_txt = int(q_s.shape[0])
    if rope_s is not None:
        import comfy.model_management as comfy_mm
        import comfy.quant_ops

        q_r = q_s.view(1, l_txt, heads, head_dim)
        k_r = k_s.view(1, l_txt, heads, head_dim)
        qw = comfy_mm.cast_to(block.attn.q_norm.weight, device=device)
        kw = comfy_mm.cast_to(block.attn.k_norm.weight, device=device)
        rot = int(rope_s.shape[-3]) * 2
        comfy.quant_ops.ck.rms_rope_split_half_(
            q_r, k_r, rope_s, qw, kw, epsilon=block.attn.q_norm.eps, rot_dim=rot
        )
        return q_r[0], k_r[0]
    q_n = block.attn.q_norm(q_s.view(l_txt, heads, head_dim))
    k_n = block.attn.k_norm(k_s.view(l_txt, heads, head_dim))
    return q_n, k_n


class _WorkspaceBuffers:
    __slots__ = (
        "h_shadow_a",
        "h_shadow_b",
        "q_a",
        "k_a",
        "v_a",
        "q_b",
        "k_b",
        "v_b",
        "k_txt_saved",
        "v_txt_saved",
        "aud_out_bhnd",
        "aud_comp_out_bhnd",
    )

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        l_txt_main: int,
        l_txt_a: int,
        l_txt_b: int,
        heads: int,
        head_dim: int,
        hidden_dim: int,
        dual_shadow: bool,
    ) -> None:
        self.h_shadow_a = torch.empty((l_txt_a, hidden_dim), device=device, dtype=dtype)
        self.q_a = torch.empty((1, heads, l_txt_a, head_dim), device=device, dtype=dtype)
        self.k_a = torch.empty((1, heads, l_txt_a, head_dim), device=device, dtype=dtype)
        self.v_a = torch.empty((1, heads, l_txt_a, head_dim), device=device, dtype=dtype)
        self.k_txt_saved = torch.empty((1, heads, l_txt_main, head_dim), device=device, dtype=dtype)
        self.v_txt_saved = torch.empty((1, heads, l_txt_main, head_dim), device=device, dtype=dtype)
        if dual_shadow:
            self.h_shadow_b = torch.empty((l_txt_b, hidden_dim), device=device, dtype=dtype)
            self.q_b = torch.empty((1, heads, l_txt_b, head_dim), device=device, dtype=dtype)
            self.k_b = torch.empty((1, heads, l_txt_b, head_dim), device=device, dtype=dtype)
            self.v_b = torch.empty((1, heads, l_txt_b, head_dim), device=device, dtype=dtype)
        else:
            self.h_shadow_b = None
            self.q_b = None
            self.k_b = None
            self.v_b = None
        self.aud_out_bhnd: torch.Tensor | None = None
        self.aud_comp_out_bhnd: torch.Tensor | None = None

    def ensure_aud_out(
        self, total_aud: int, heads: int, head_dim: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        cur = self.aud_out_bhnd
        if (
            cur is not None
            and cur.device == device
            and cur.dtype == dtype
            and cur.shape == (1, heads, total_aud, head_dim)
        ):
            return cur
        self.aud_out_bhnd = torch.empty((1, heads, total_aud, head_dim), device=device, dtype=dtype)
        return self.aud_out_bhnd

    def ensure_aud_comp_out(
        self, total_comp: int, heads: int, head_dim: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        cur = self.aud_comp_out_bhnd
        if (
            cur is not None
            and cur.device == device
            and cur.dtype == dtype
            and cur.shape == (1, heads, total_comp, head_dim)
        ):
            return cur
        self.aud_comp_out_bhnd = torch.empty((1, heads, total_comp, head_dim), device=device, dtype=dtype)
        return self.aud_comp_out_bhnd


class H3SteerStepWorkspace:
    """Per-clone reusable GPU workspace allocated strictly under `pause_malloc_graph()`."""

    def __init__(self, max_entries: int = _WORKSPACE_CACHE_MAX) -> None:
        self._max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[tuple[Any, ...], _WorkspaceBuffers] = OrderedDict()

    def acquire(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        l_txt: int | None = None,
        l_txt_main: int | None = None,
        l_txt_a: int | None = None,
        l_txt_b: int | None = None,
        heads: int,
        head_dim: int,
        hidden_dim: int,
        dual_shadow: bool,
    ) -> _WorkspaceBuffers:
        l_main = int(l_txt_main if l_txt_main is not None else (l_txt if l_txt is not None else 0))
        if l_main <= 0:
            raise ValueError("l_txt / l_txt_main must be > 0")
        la = int(l_txt_a if l_txt_a is not None else l_main)
        lb = int(l_txt_b if l_txt_b is not None else l_main)
        key = (
            str(device),
            str(dtype),
            l_main,
            la,
            lb if dual_shadow else 0,
            int(heads),
            int(head_dim),
            int(hidden_dim),
            bool(dual_shadow),
        )
        existing = self._entries.get(key)
        if existing is not None:
            self._entries.move_to_end(key)
            return existing
        bufs = _WorkspaceBuffers(
            device=device,
            dtype=dtype,
            l_txt_main=l_main,
            l_txt_a=la,
            l_txt_b=lb,
            heads=heads,
            head_dim=head_dim,
            hidden_dim=hidden_dim,
            dual_shadow=dual_shadow,
        )
        self._entries[key] = bufs
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
        return bufs

    def clear(self) -> None:
        self._entries.clear()


class H3SteerRuntime:
    """Call-local Mode 1-3 attention routing state attached to `transformer_options`."""

    def __init__(
        self,
        *,
        dm: Any,
        workspace: H3SteerStepWorkspace,
        mode: int,
        main_ctx: torch.Tensor,
        shadow_a_init: torch.Tensor,
        shadow_b_init: torch.Tensor | None,
        swap_blocks: frozenset[int],
        aud_split_blocks: frozenset[int],
        dual_shadow: bool,
        fork_b_at_block: int | None,
        aud_uses_shadow_b: bool,
        aud_unswapped_blocks: frozenset[int] = frozenset(),
        save_aud_k_on_block_exit: bool = True,
        aud_to_vid_gain: float = 1.0,
        audio_channel: str = "both",
        audio_time_start: float = 0.0,
        audio_time_end: float = 1.0,
        ref_img_gain: float = 1.0,
        ref_img_blocks: frozenset[int] = frozenset(),
        ref_audio_only: bool = False,
        shadow_a_tags: torch.Tensor | None = None,
        shadow_b_tags: torch.Tensor | None = None,
        aud_outside_uses_shadow_a: bool = False,
    ) -> None:
        self.dm = dm
        self.workspace = workspace
        self.n_blocks = len(dm.blocks)
        self.mode = int(mode)
        self.main_ctx = main_ctx
        self.shadow_a_init = shadow_a_init
        self.shadow_b_init = shadow_b_init
        self.l_txt_a = int(shadow_a_init.shape[-2])
        self.l_txt_b = int(shadow_b_init.shape[-2]) if shadow_b_init is not None else None
        self.swap_blocks = swap_blocks
        self.aud_split_blocks = aud_split_blocks
        self.aud_unswapped_blocks = aud_unswapped_blocks
        self.dual_shadow = bool(dual_shadow)
        self.fork_b_at_block = fork_b_at_block
        self.aud_uses_shadow_b = bool(aud_uses_shadow_b)
        self.save_aud_k_on_block_exit = bool(save_aud_k_on_block_exit)
        self.aud_to_vid_gain = float(aud_to_vid_gain)
        self.audio_channel = str(audio_channel or "both").strip()
        self.audio_time_start = float(audio_time_start)
        self.audio_time_end = float(audio_time_end)
        self.ref_img_gain = float(ref_img_gain)
        self.ref_img_blocks = ref_img_blocks
        self.ref_audio_only = bool(ref_audio_only)
        self.shadow_a_tags = shadow_a_tags
        self.shadow_b_tags = shadow_b_tags
        self.aud_outside_uses_shadow_a = bool(aud_outside_uses_shadow_a)
        self._max_aud_outside_block = (
            max(aud_unswapped_blocks)
            if (self.aud_outside_uses_shadow_a and aud_unswapped_blocks)
            else -1
        )
        self.active = True
        self._bufs: _WorkspaceBuffers | None = None
        self._cur: dict[str, Any] | None = None
        self._rope_a: Any = None
        self._rope_b: Any = None
        self.stats: dict[str, Any] = {
            "mode": self.mode,
            "kv_swaps": 0,
            "aud_splices": 0,
            "shadow_blocks": 0,
            "ref_img_damps": 0,
            "ref_audio_isolations": 0,
            "blocks_touched": [],
        }

    def prepare_block(self, block_index: int, args: dict[str, Any]) -> None:
        self._cur = None
        if not self.active:
            return
        layout = args.get("layout")
        if layout is None or not hasattr(layout, "segments"):
            return
        text_seg = next((s for s in layout.segments if s[2] == "text"), None)
        if text_seg is None:
            return
        tx_a, tx_b, _ = text_seg
        l_txt = int(tx_b - tx_a)
        if l_txt <= 0:
            return
        l_txt_a = self.l_txt_a
        l_txt_b = l_txt if self.fork_b_at_block is not None else (self.l_txt_b if self.l_txt_b is not None else l_txt)
        raw_au_spans = [(int(sa), int(sb)) for (sa, sb, kind) in layout.segments if kind == "audio" and sb > sa]
        in_aud = bool(raw_au_spans) and (block_index in self.aud_split_blocks)
        if in_aud:
            au_spans = _slice_audio_spans(
                raw_au_spans,
                audio_channel=self.audio_channel,
                audio_time_start=self.audio_time_start,
                audio_time_end=self.audio_time_end,
            )
            au_comp_spans = _complement_spans(raw_au_spans, au_spans)
        else:
            au_spans = raw_au_spans
            au_comp_spans = []
        total_aud = sum(b - a for a, b in au_spans)
        total_aud_comp = sum(b - a for a, b in au_comp_spans)
        vid_spans = [(int(sa), int(sb)) for (sa, sb, kind) in layout.segments if kind == "video" and sb > sa]
        ref_img_spans = [(int(sa), int(sb)) for (sa, sb, kind) in layout.segments if kind == "ref_img" and sb > sa]
        ref_audio_spans = [(int(sa), int(sb)) for (sa, sb, kind) in layout.segments if kind == "ref_audio" and sb > sa]

        x_full = args["img"]
        device = x_full.device
        dtype = x_full.dtype
        hidden_dim = int(x_full.shape[-1])
        block = self.dm.blocks[block_index]
        heads = int(block.attn.heads)
        head_dim = int(block.attn.head_dim)

        in_swap = block_index in self.swap_blocks
        if in_swap and l_txt_a != l_txt:
            raise ValueError(
                f"in-place block_band swap at block {block_index} requires equal shadow/main text length "
                f"({l_txt_a} != {l_txt})"
            )

        if self._bufs is None or block_index == 0:
            self._bufs = self.workspace.acquire(
                device=device,
                dtype=dtype,
                l_txt_main=l_txt,
                l_txt_a=l_txt_a,
                l_txt_b=l_txt_b,
                heads=heads,
                head_dim=head_dim,
                hidden_dim=hidden_dim,
                dual_shadow=self.dual_shadow,
            )
        bufs = self._bufs
        if block_index == 0:
            src_a = self.shadow_a_init[0] if self.shadow_a_init.ndim == 3 else self.shadow_a_init
            bufs.h_shadow_a.copy_(src_a.to(device=device, dtype=dtype))
            if self.dual_shadow and self.shadow_b_init is not None and bufs.h_shadow_b is not None:
                src_b = self.shadow_b_init[0] if self.shadow_b_init.ndim == 3 else self.shadow_b_init
                bufs.h_shadow_b.copy_(src_b.to(device=device, dtype=dtype))
            rope_txt_init = _slice_rope(args.get("rope_freqs"), tx_a, tx_b)
            self._rope_a = _build_shadow_text_rope(self.dm, l_txt, l_txt_a, device, dtype, rope_txt_init)
            self._rope_b = (
                _build_shadow_text_rope(self.dm, l_txt, l_txt_b, device, dtype, rope_txt_init)
                if self.dual_shadow
                else None
            )
        if self.dual_shadow and self.fork_b_at_block == block_index and bufs.h_shadow_b is not None:
            bufs.h_shadow_b.copy_(x_full[tx_a:tx_b])

        in_aud = bool(au_spans) and (block_index in self.aud_split_blocks)
        in_ref_img_damp = False
        in_ref_aud_iso = False
        in_aud_unswapped = bool(au_spans) and (not in_aud) and (block_index in self.aud_unswapped_blocks)
        is_last = block_index >= (self.n_blocks - 1)

        need_a_kv = (
            in_swap
            or (in_aud and not self.aud_uses_shadow_b)
            or (self.aud_outside_uses_shadow_a and in_aud_unswapped)
            or (
                not is_last
                and (
                    bool(self.swap_blocks)
                    or (bool(self.aud_split_blocks) and not self.aud_uses_shadow_b)
                    or (self.aud_outside_uses_shadow_a and block_index < self._max_aud_outside_block)
                )
            )
        )
        need_b_kv = bool(
            self.dual_shadow
            and bufs.h_shadow_b is not None
            and ((self.fork_b_at_block is None) or (block_index >= self.fork_b_at_block))
            and (in_aud or not is_last)
        )
        if not need_a_kv and not need_b_kv and not in_aud_unswapped and not in_ref_img_damp and not in_ref_aud_iso:
            return

        if not need_a_kv and not need_b_kv:
            self._cur = {
                "block_index": block_index,
                "block": block,
                "heads": heads,
                "head_dim": head_dim,
                "tx_a": tx_a,
                "tx_b": tx_b,
                "l_txt": l_txt,
                "l_txt_a": l_txt_a,
                "l_txt_b": l_txt_b,
                "au_spans": au_spans,
                "total_aud": total_aud,
                "au_comp_spans": au_comp_spans,
                "total_aud_comp": total_aud_comp,
                "vid_spans": vid_spans,
                "ref_img_spans": ref_img_spans,
                "ref_audio_spans": ref_audio_spans,
                "in_ref_img_damp": in_ref_img_damp,
                "in_ref_aud_iso": in_ref_aud_iso,
                "mod_txt_a": (),
                "mod_txt_b": (),
                "gate_msa": None,
                "shift_mlp": None,
                "scale_mlp": None,
                "gate_mlp": None,
                "need_a_kv": False,
                "need_b_kv": False,
                "need_finish_a": False,
                "need_finish_b": False,
                "in_aud_unswapped": in_aud_unswapped,
                "shadow_a_attn_2d": None,
                "shadow_b_attn_2d": None,
                "done_pre": False,
                "has_aud_splice": False,
                "has_aud_comp_splice": False,
            }
            return

        from comfy.ldm.minimax.model import _mod_scale_shift

        mod_txt = _slice_mod_segments(args["mod_segments"], tx_a, tx_b)
        mod_txt_a = _adapt_text_mod_segments(mod_txt, l_txt, l_txt_a, shadow_tags=self.shadow_a_tags)
        mod_txt_b = (
            _adapt_text_mod_segments(mod_txt, l_txt, l_txt_b, shadow_tags=self.shadow_b_tags)
            if self.dual_shadow
            else mod_txt
        )
        rope_txt = _slice_rope(args.get("rope_freqs"), tx_a, tx_b)
        rope_a = rope_txt if l_txt_a == l_txt else self._rope_a
        rope_b = rope_txt if l_txt_b == l_txt else self._rope_b
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(args["t_emb"])

        if need_a_kv:
            h_in_a = _mod_scale_shift(block.norm1(bufs.h_shadow_a), shift_msa, scale_msa, mod_txt_a)
            q_a, k_a, v_a = block.attn.qkv_proj(h_in_a).split(heads * head_dim, dim=-1)
            v_a = v_a.view(l_txt_a, heads, head_dim)
            q_a, k_a = _apply_rms_rope_txt(block, q_a, k_a, rope_a, heads=heads, head_dim=head_dim, device=device)
            bufs.q_a.copy_(q_a.transpose(0, 1).unsqueeze(0))
            bufs.k_a.copy_(k_a.transpose(0, 1).unsqueeze(0))
            bufs.v_a.copy_(v_a.transpose(0, 1).unsqueeze(0))
            del h_in_a, q_a, k_a, v_a

        if need_b_kv and bufs.h_shadow_b is not None:
            assert bufs.q_b is not None and bufs.k_b is not None and bufs.v_b is not None
            h_in_b = _mod_scale_shift(block.norm1(bufs.h_shadow_b), shift_msa, scale_msa, mod_txt_b)
            q_b, k_b, v_b = block.attn.qkv_proj(h_in_b).split(heads * head_dim, dim=-1)
            v_b = v_b.view(l_txt_b, heads, head_dim)
            q_b, k_b = _apply_rms_rope_txt(block, q_b, k_b, rope_b, heads=heads, head_dim=head_dim, device=device)
            bufs.q_b.copy_(q_b.transpose(0, 1).unsqueeze(0))
            bufs.k_b.copy_(k_b.transpose(0, 1).unsqueeze(0))
            bufs.v_b.copy_(v_b.transpose(0, 1).unsqueeze(0))
            del h_in_b, q_b, k_b, v_b

        self._cur = {
            "block_index": block_index,
            "block": block,
            "heads": heads,
            "head_dim": head_dim,
            "tx_a": tx_a,
            "tx_b": tx_b,
            "l_txt": l_txt,
            "l_txt_a": l_txt_a,
            "l_txt_b": l_txt_b,
            "au_spans": au_spans,
            "total_aud": total_aud,
            "au_comp_spans": au_comp_spans,
            "total_aud_comp": total_aud_comp,
            "vid_spans": vid_spans,
            "ref_img_spans": ref_img_spans,
            "ref_audio_spans": ref_audio_spans,
            "in_ref_img_damp": in_ref_img_damp,
            "in_ref_aud_iso": in_ref_aud_iso,
            "mod_txt_a": mod_txt_a,
            "mod_txt_b": mod_txt_b,
            "gate_msa": gate_msa,
            "shift_mlp": shift_mlp,
            "scale_mlp": scale_mlp,
            "gate_mlp": gate_mlp,
            "need_a_kv": need_a_kv,
            "need_b_kv": need_b_kv,
            "need_finish_a": (not is_last) and need_a_kv,
            "need_finish_b": (not is_last) and need_b_kv,
            "in_aud_unswapped": in_aud_unswapped,
            "shadow_a_attn_2d": None,
            "shadow_b_attn_2d": None,
            "done_pre": False,
            "has_aud_splice": False,
            "has_aud_comp_splice": False,
        }

    def pre_attention(
        self,
        q_p: torch.Tensor,
        k_p: torch.Tensor,
        v_p: torch.Tensor,
        options: dict[str, Any] | None,
        *,
        seq_axis: int,
    ) -> bool:
        cur = self._cur
        bufs = self._bufs
        if not self.active or cur is None or bufs is None or cur["done_pre"]:
            return False
        b_idx = options.get("block_index") if isinstance(options, dict) else None
        if b_idx is not None and int(b_idx) != int(cur["block_index"]):
            return False
        cur["done_pre"] = True

        if seq_axis == 1:
            q_bh = q_p.transpose(1, 2)
            k_bh = k_p.transpose(1, 2)
            v_bh = v_p.transpose(1, 2)
        else:
            q_bh, k_bh, v_bh = q_p, k_p, v_p

        tx_a = cur["tx_a"]
        tx_b = cur["tx_b"]
        l_txt = cur["l_txt"]
        l_txt_a = cur["l_txt_a"]
        l_txt_b = cur["l_txt_b"]
        heads = cur["heads"]
        head_dim = cur["head_dim"]
        inner = heads * head_dim
        au_spans = cur["au_spans"]
        total_aud = cur["total_aud"]
        au_comp_spans = cur["au_comp_spans"]
        total_aud_comp = cur["total_aud_comp"]
        vid_spans = cur["vid_spans"]
        ref_img_spans = cur["ref_img_spans"]
        ref_audio_spans = cur["ref_audio_spans"]
        in_ref_img_damp = bool(cur.get("in_ref_img_damp", False))
        in_ref_aud_iso = bool(cur.get("in_ref_aud_iso", False))
        block_index = cur["block_index"]
        in_swap = block_index in self.swap_blocks
        in_aud = bool(au_spans) and (block_index in self.aud_split_blocks)
        in_aud_unswapped = bool(cur.get("in_aud_unswapped", False)) and bool(au_spans)
        k_dtype = k_bh.dtype
        v_dtype = v_bh.dtype
        k_len_main = int(k_bh.shape[2])

        def _extract_q_spans(spans: Sequence[tuple[int, int]]) -> torch.Tensor:
            return (
                q_bh[:, :, spans[0][0] : spans[0][1], :].to(dtype=k_dtype)
                if len(spans) == 1
                else torch.cat([q_bh[:, :, a:b, :].to(dtype=k_dtype) for a, b in spans], dim=2)
            )

        # Isolate non-target audio rows (`au_comp_spans`) from target `K_aud` while K_txt_main & V[ref_*] are intact.
        if (in_aud or in_aud_unswapped) and au_comp_spans and total_aud_comp > 0:
            comp_bias = _build_aud_attn_bias(
                k_len_main, (), au_spans, 0, tx_b, 1.0, k_bh.device, k_dtype
            )
            attn_comp = F.scaled_dot_product_attention(
                _extract_q_spans(au_comp_spans), k_bh, v_bh, attn_mask=comp_bias, dropout_p=0.0, is_causal=False
            )
            comp_buf = bufs.ensure_aud_comp_out(total_aud_comp, heads, head_dim, k_bh.device, attn_comp.dtype)
            comp_buf.copy_(attn_comp)
            cur["has_aud_comp_splice"] = True
            del comp_bias, attn_comp

        if in_aud_unswapped and not self.aud_outside_uses_shadow_a:
            q_aud_u = _extract_q_spans(au_spans)
            gain_u = 1.0 if self.aud_split_blocks else self.aud_to_vid_gain
            aud_bias_u = _build_aud_attn_bias(
                k_len_main, vid_spans, au_comp_spans, 0, tx_b, gain_u, k_bh.device, k_dtype
            )
            attn_aud_u = F.scaled_dot_product_attention(
                q_aud_u, k_bh, v_bh, attn_mask=aud_bias_u, dropout_p=0.0, is_causal=False
            )
            aud_buf = bufs.ensure_aud_out(total_aud, heads, head_dim, k_bh.device, attn_aud_u.dtype)
            aud_buf.copy_(attn_aud_u)
            cur["has_aud_splice"] = True
            del q_aud_u, attn_aud_u, aud_bias_u

        k_slice = k_bh[:, :, tx_a:tx_b, :]
        v_slice = v_bh[:, :, tx_a:tx_b, :]
        did_save = False

        def _save_main() -> None:
            nonlocal did_save
            if not did_save:
                bufs.k_txt_saved.copy_(k_slice)
                bufs.v_txt_saved.copy_(v_slice)
                did_save = True

        def _restore_main() -> None:
            k_slice.copy_(bufs.k_txt_saved.to(dtype=k_dtype))
            v_slice.copy_(bufs.v_txt_saved.to(dtype=v_dtype))

        need_q_a = (
            cur["need_finish_a"]
            or (in_aud and not self.aud_uses_shadow_b)
            or (in_aud_unswapped and self.aud_outside_uses_shadow_a)
        )
        need_q_b = cur["need_finish_b"] or (in_aud and self.aud_uses_shadow_b)

        if self.dual_shadow and cur["need_b_kv"] and need_q_b:
            assert bufs.q_b is not None and bufs.k_b is not None and bufs.v_b is not None
            take_aud_b = in_aud and self.aud_uses_shadow_b
            txt_delta_b = l_txt_b - l_txt
            aud_bias_b = (
                _build_aud_attn_bias(
                    k_len_main + txt_delta_b,
                    vid_spans,
                    au_comp_spans,
                    txt_delta_b,
                    tx_b,
                    self.aud_to_vid_gain,
                    k_bh.device,
                    k_dtype,
                )
                if take_aud_b
                else None
            )
            if l_txt_b == l_txt:
                if not in_swap:
                    _save_main()
                k_slice.copy_(bufs.k_b.to(dtype=k_dtype))
                v_slice.copy_(bufs.v_b.to(dtype=v_dtype))
                k_src_b, v_src_b, own_kv_b = k_bh, v_bh, False
            else:
                k_src_b = torch.cat(
                    [k_bh[:, :, :tx_a, :], bufs.k_b.to(dtype=k_dtype), k_bh[:, :, tx_b:, :]],
                    dim=2,
                )
                v_src_b = torch.cat(
                    [v_bh[:, :, :tx_a, :], bufs.v_b.to(dtype=v_dtype), v_bh[:, :, tx_b:, :]],
                    dim=2,
                )
                own_kv_b = True

            if aud_bias_b is None:
                q_parts_b: list[torch.Tensor] = []
                if cur["need_finish_b"]:
                    q_parts_b.append(bufs.q_b.to(dtype=k_dtype))
                if take_aud_b:
                    q_parts_b.append(_extract_q_spans(au_spans))
                if q_parts_b:
                    q_comb_b = q_parts_b[0] if len(q_parts_b) == 1 else torch.cat(q_parts_b, dim=2)
                    attn_b = F.scaled_dot_product_attention(q_comb_b, k_src_b, v_src_b, dropout_p=0.0, is_causal=False)
                    off_b = 0
                    if cur["need_finish_b"]:
                        cur["shadow_b_attn_2d"] = (
                            attn_b[:, :, :l_txt_b, :][0]
                            .transpose(0, 1)
                            .reshape(l_txt_b, inner)
                            .to(dtype=bufs.h_shadow_a.dtype)
                        )
                        off_b = l_txt_b
                    if take_aud_b:
                        aud_buf = bufs.ensure_aud_out(total_aud, heads, head_dim, k_bh.device, attn_b.dtype)
                        aud_buf.copy_(attn_b[:, :, off_b : off_b + total_aud, :])
                        cur["has_aud_splice"] = True
                    del q_comb_b, attn_b
            else:
                if cur["need_finish_b"]:
                    txt_vid_bias_b = (
                        _build_aud_attn_bias(
                            k_len_main + txt_delta_b,
                            vid_spans,
                            (),
                            txt_delta_b,
                            tx_b,
                            self.aud_to_vid_gain,
                            k_bh.device,
                            k_dtype,
                        )
                        if take_aud_b and (self.aud_to_vid_gain < 1.0 - 1e-6) and bool(vid_spans)
                        else None
                    )
                    attn_tb = F.scaled_dot_product_attention(
                        bufs.q_b.to(dtype=k_dtype),
                        k_src_b,
                        v_src_b,
                        attn_mask=txt_vid_bias_b,
                        dropout_p=0.0,
                        is_causal=False,
                    )
                    cur["shadow_b_attn_2d"] = (
                        attn_tb[0].transpose(0, 1).reshape(l_txt_b, inner).to(dtype=bufs.h_shadow_a.dtype)
                    )
                    del attn_tb, txt_vid_bias_b
                if take_aud_b:
                    q_aud_b = _extract_q_spans(au_spans)
                    attn_ab = F.scaled_dot_product_attention(
                        q_aud_b, k_src_b, v_src_b, attn_mask=aud_bias_b, dropout_p=0.0, is_causal=False
                    )
                    aud_buf = bufs.ensure_aud_out(total_aud, heads, head_dim, k_bh.device, attn_ab.dtype)
                    aud_buf.copy_(attn_ab)
                    cur["has_aud_splice"] = True
                    del q_aud_b, attn_ab
                del aud_bias_b
            if own_kv_b:
                del k_src_b, v_src_b
            if l_txt_b == l_txt and not in_swap:
                _restore_main()

        run_shadow_a_sdpa = False
        if cur["need_a_kv"] and need_q_a:
            take_aud_a = (in_aud and not self.aud_uses_shadow_b) or (
                in_aud_unswapped and self.aud_outside_uses_shadow_a
            )
            aud_gain_a = self.aud_to_vid_gain if in_aud else 1.0
            txt_delta_a = l_txt_a - l_txt
            aud_bias_a = (
                _build_aud_attn_bias(
                    k_len_main + txt_delta_a,
                    vid_spans,
                    au_comp_spans,
                    txt_delta_a,
                    tx_b,
                    aud_gain_a,
                    k_bh.device,
                    k_dtype,
                )
                if take_aud_a
                else None
            )
            if cur["need_finish_a"] or take_aud_a:
                run_shadow_a_sdpa = True
                if l_txt_a == l_txt:
                    if not in_swap:
                        _save_main()
                    k_slice.copy_(bufs.k_a.to(dtype=k_dtype))
                    v_slice.copy_(bufs.v_a.to(dtype=v_dtype))
                    k_src_a, v_src_a, own_kv_a = k_bh, v_bh, False
                else:
                    k_src_a = torch.cat(
                        [k_bh[:, :, :tx_a, :], bufs.k_a.to(dtype=k_dtype), k_bh[:, :, tx_b:, :]],
                        dim=2,
                    )
                    v_src_a = torch.cat(
                        [v_bh[:, :, :tx_a, :], bufs.v_a.to(dtype=v_dtype), v_bh[:, :, tx_b:, :]],
                        dim=2,
                    )
                    own_kv_a = True

                if aud_bias_a is None:
                    q_parts_a: list[torch.Tensor] = []
                    if cur["need_finish_a"]:
                        q_parts_a.append(bufs.q_a.to(dtype=k_dtype))
                    if take_aud_a:
                        q_parts_a.append(_extract_q_spans(au_spans))
                    q_comb_a = q_parts_a[0] if len(q_parts_a) == 1 else torch.cat(q_parts_a, dim=2)
                    attn_a = F.scaled_dot_product_attention(q_comb_a, k_src_a, v_src_a, dropout_p=0.0, is_causal=False)
                    off_a = 0
                    if cur["need_finish_a"]:
                        cur["shadow_a_attn_2d"] = (
                            attn_a[:, :, :l_txt_a, :][0]
                            .transpose(0, 1)
                            .reshape(l_txt_a, inner)
                            .to(dtype=bufs.h_shadow_a.dtype)
                        )
                        off_a = l_txt_a
                    if take_aud_a:
                        aud_buf = bufs.ensure_aud_out(total_aud, heads, head_dim, k_bh.device, attn_a.dtype)
                        aud_buf.copy_(attn_a[:, :, off_a : off_a + total_aud, :])
                        cur["has_aud_splice"] = True
                    del q_comb_a, attn_a
                else:
                    if cur["need_finish_a"]:
                        txt_vid_bias_a = (
                            _build_aud_attn_bias(
                                k_len_main + txt_delta_a,
                                vid_spans,
                                (),
                                txt_delta_a,
                                tx_b,
                                self.aud_to_vid_gain,
                                k_bh.device,
                                k_dtype,
                            )
                            if (
                                take_aud_a
                                and in_aud
                                and (not in_swap)
                                and (self.aud_to_vid_gain < 1.0 - 1e-6)
                                and bool(vid_spans)
                            )
                            else None
                        )
                        attn_ta = F.scaled_dot_product_attention(
                            bufs.q_a.to(dtype=k_dtype),
                            k_src_a,
                            v_src_a,
                            attn_mask=txt_vid_bias_a,
                            dropout_p=0.0,
                            is_causal=False,
                        )
                        cur["shadow_a_attn_2d"] = (
                            attn_ta[0].transpose(0, 1).reshape(l_txt_a, inner).to(dtype=bufs.h_shadow_a.dtype)
                        )
                        del attn_ta, txt_vid_bias_a
                    if take_aud_a:
                        q_aud_a = _extract_q_spans(au_spans)
                        attn_aa = F.scaled_dot_product_attention(
                            q_aud_a, k_src_a, v_src_a, attn_mask=aud_bias_a, dropout_p=0.0, is_causal=False
                        )
                        aud_buf = bufs.ensure_aud_out(total_aud, heads, head_dim, k_bh.device, attn_aa.dtype)
                        aud_buf.copy_(attn_aa)
                        cur["has_aud_splice"] = True
                        del q_aud_a, attn_aa
                    del aud_bias_a
                if own_kv_a:
                    del k_src_a, v_src_a
                if l_txt_a == l_txt and not in_swap:
                    _restore_main()
        elif in_swap and cur["need_a_kv"]:
            k_slice.copy_(bufs.k_a.to(dtype=k_dtype))
            v_slice.copy_(bufs.v_a.to(dtype=v_dtype))

        # Multimodal ref_img / ref_audio in-place V mutations disarmed (safe no-op).

        if in_swap:
            self.stats["kv_swaps"] += 1
        if run_shadow_a_sdpa or (self.dual_shadow and cur["need_b_kv"]) or in_aud_unswapped:
            self.stats["shadow_blocks"] += 1
        self.stats["blocks_touched"].append(block_index)
        return True

    def reset_pre_block(self) -> None:
        cur = self._cur
        if cur is not None:
            cur["done_pre"] = False
            cur["has_aud_splice"] = False
            cur["has_aud_comp_splice"] = False

    def post_attention(
        self,
        out_main: torch.Tensor,
        *,
        nhd: bool = False,
        skip_output_reshape: bool = False,
    ) -> None:
        cur = self._cur
        bufs = self._bufs
        if not self.active or cur is None or bufs is None:
            return

        def _splice_spans(src_bhnd: torch.Tensor | None, spans: Sequence[tuple[int, int]], total_len: int) -> bool:
            if src_bhnd is None or not spans or total_len <= 0:
                return False
            off = 0
            if nhd:
                src = src_bhnd.transpose(1, 2).to(dtype=out_main.dtype)
                for a, b in spans:
                    n_seg = b - a
                    out_main[:, a:b, :, :].copy_(src[:, off : off + n_seg, :, :])
                    off += n_seg
            elif skip_output_reshape:
                src = src_bhnd.to(dtype=out_main.dtype)
                for a, b in spans:
                    n_seg = b - a
                    out_main[:, :, a:b, :].copy_(src[:, :, off : off + n_seg, :])
                    off += n_seg
            else:
                inner = cur["heads"] * cur["head_dim"]
                src_2d = src_bhnd[0].transpose(0, 1).reshape(1, total_len, inner).to(dtype=out_main.dtype)
                for a, b in spans:
                    n_seg = b - a
                    out_main[:, a:b, :].copy_(src_2d[:, off : off + n_seg, :])
                    off += n_seg
            return True

        if cur.get("has_aud_splice"):
            if _splice_spans(bufs.aud_out_bhnd, cur["au_spans"], cur["total_aud"]):
                self.stats["aud_splices"] += 1
            cur["has_aud_splice"] = False
        if cur.get("has_aud_comp_splice"):
            _splice_spans(bufs.aud_comp_out_bhnd, cur["au_comp_spans"], cur["total_aud_comp"])
            cur["has_aud_comp_splice"] = False

    def finish_block(self, block_index: int, out: Any) -> None:
        cur = self._cur
        bufs = self._bufs
        self._cur = None
        if not self.active or cur is None or bufs is None:
            return
        if int(cur["block_index"]) != int(block_index) or block_index >= (self.n_blocks - 1):
            return
        from comfy.ldm.minimax.model import _mod_gate, _mod_scale_shift

        block = cur["block"]
        gate_msa = cur["gate_msa"]
        shift_mlp = cur["shift_mlp"]
        scale_mlp = cur["scale_mlp"]
        gate_mlp = cur["gate_mlp"]

        if cur["need_finish_a"] and cur["shadow_a_attn_2d"] is not None:
            mod_txt_a = cur["mod_txt_a"]
            attn_out = block.attn.out_proj(cur["shadow_a_attn_2d"])
            cur["shadow_a_attn_2d"] = None
            _mod_gate(bufs.h_shadow_a, gate_msa, attn_out, mod_txt_a)
            del attn_out
            h_mlp = _mod_scale_shift(block.norm2(bufs.h_shadow_a), shift_mlp, scale_mlp, mod_txt_a)
            _mod_gate(bufs.h_shadow_a, gate_mlp, block.mlp(h_mlp), mod_txt_a)
            del h_mlp

        if (
            cur["need_finish_b"]
            and cur["shadow_b_attn_2d"] is not None
            and bufs.h_shadow_b is not None
        ):
            mod_txt_b = cur["mod_txt_b"]
            attn_out_b = block.attn.out_proj(cur["shadow_b_attn_2d"])
            cur["shadow_b_attn_2d"] = None
            _mod_gate(bufs.h_shadow_b, gate_msa, attn_out_b, mod_txt_b)
            del attn_out_b
            h_mlp_b = _mod_scale_shift(block.norm2(bufs.h_shadow_b), shift_mlp, scale_mlp, mod_txt_b)
            _mod_gate(bufs.h_shadow_b, gate_mlp, block.mlp(h_mlp_b), mod_txt_b)
            del h_mlp_b


class H3SteerBlockPatch:
    """Stateless per-clone `patches_replace['dit'][('double_block', i)]` adapter."""

    def __init__(self, block_index: int, previous: Any = None) -> None:
        self.block_index = int(block_index)
        self.previous = previous

    def __call__(self, args: dict[str, Any], extra_args: dict[str, Any]) -> dict[str, Any]:
        to = args.get("transformer_options")
        rt = to.get("_h3_steer_runtime") if isinstance(to, dict) else None
        if rt is not None and getattr(rt, "active", False):
            with pause_malloc_graph_ctx(), torch.no_grad():
                rt.prepare_block(self.block_index, args)
        orig_blk = extra_args["original_block"]
        out = orig_blk(args) if self.previous is None else self.previous(args, extra_args)
        if rt is not None and getattr(rt, "active", False):
            with pause_malloc_graph_ctx(), torch.no_grad():
                rt.finish_block(self.block_index, out)
        return out

    def to(self, device_or_dtype: Any):
        if hasattr(self.previous, "to"):
            self.previous = self.previous.to(device_or_dtype)
        return self

    def cleanup(self) -> None:
        if hasattr(self.previous, "cleanup"):
            self.previous.cleanup()

    def models(self) -> list[Any]:
        if hasattr(self.previous, "models"):
            return list(self.previous.models())
        return []


def has_steer_block_patch(patch: Any) -> bool:
    """Walk `.previous` chain to detect an already-installed `H3SteerBlockPatch`."""
    cur = patch
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, H3SteerBlockPatch):
            return True
        seen.add(id(cur))
        cur = getattr(cur, "previous", None)
    return False


def make_steer_attention_override(previous: Any = None) -> Callable[..., Any]:
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

        rt = options.get("_h3_steer_runtime") if isinstance(options, dict) else None
        if rt is None or not getattr(rt, "active", False) or not skip_reshape or mask is not None:
            return _pass_through(q, k, v)

        q_p = q if torch.is_tensor(q) else q.peek()
        k_p = k if torch.is_tensor(k) else k.peek()
        v_p = v if torch.is_tensor(v) else v.peek()
        if q_p.ndim != 4 or k_p.ndim != 4 or v_p.ndim != 4:
            return _pass_through(q, k, v)

        did_set_gain = False
        ran_pre = False
        try:
            with pause_malloc_graph_ctx(), torch.no_grad():
                did_set_gain = _apply_audio_gain_bhnd(k_p, v_p, options)
                ran_pre = rt.pre_attention(q_p, k_p, v_p, options, seq_axis=2)
            del q_p, k_p, v_p
            out_main = _pass_through(q, k, v)
            if ran_pre:
                with pause_malloc_graph_ctx(), torch.no_grad():
                    rt.post_attention(out_main, nhd=False, skip_output_reshape=bool(skip_output_reshape))
            return out_main
        finally:
            if did_set_gain and isinstance(options, dict):
                options.pop("_h3_audio_gain_applied", None)

    if getattr(previous, "container_function", None) is not None:
        override.container_function = functools.partial(override, None)
    override.h3_steer_override = True
    override.h3_previous = previous
    return override


def h3_steer_attn_seam_pre(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    transformer_options: Any,
    *,
    seq_axis: int = 1,
) -> bool:
    """Seam called from `attention.py::make_minimax_attn_forward` on standalone Sage/LSE paths."""
    rt = transformer_options.get("_h3_steer_runtime") if isinstance(transformer_options, dict) else None
    if rt is None or not getattr(rt, "active", False):
        return False
    with pause_malloc_graph_ctx(), torch.no_grad():
        return bool(rt.pre_attention(q, k, v, transformer_options, seq_axis=seq_axis))


def h3_steer_attn_seam_reset_block(transformer_options: Any) -> None:
    """Reset per-block seam latch if Sage falls back to `_stock_forward`."""
    rt = transformer_options.get("_h3_steer_runtime") if isinstance(transformer_options, dict) else None
    if rt is not None and getattr(rt, "active", False):
        rt.reset_pre_block()


def h3_steer_attn_seam_post(
    out_main: torch.Tensor,
    transformer_options: Any,
    *,
    nhd: bool = True,
) -> None:
    """Splice `aud_out_bhnd` before `out_proj` on standalone Sage/LSE paths."""
    rt = transformer_options.get("_h3_steer_runtime") if isinstance(transformer_options, dict) else None
    if rt is None or not getattr(rt, "active", False):
        return
    with pause_malloc_graph_ctx(), torch.no_grad():
        rt.post_attention(out_main, nhd=nhd, skip_output_reshape=False)


def install_steer_block_hooks(
    model_clone: Any,
    dm: Any,
    *,
    key_prefix: str,
    node_label: str = "MiniMaxH3Steer",
) -> H3SteerStepWorkspace:
    """Validate H3 compatibility, attach reusable `H3SteerBlockPatch` hooks, and return workspace."""
    from comfy.patcher_extension import CallbacksMP, WrappersMP

    t_opts = getattr(model_clone, "model_options", {}).setdefault("transformer_options", {})
    dm_wrappers = getattr(model_clone, "wrappers", {}).get(WrappersMP.DIFFUSION_MODEL, {})
    dit_replaces = t_opts.get("patches_replace", {}).get("dit", {})
    if (
        hasattr(model_clone, "_h3_frozen_video_cache")
        or any("frozen_video_cache" in str(wk) for wk in dm_wrappers)
    ):
        raise ValueError(
            f"{node_label} block_band / 1x stream split cannot be combined with MiniMaxH3FrozenVideoCache "
            "(cached steps skip double_block hooks)"
        )
    for b_patch in dit_replaces.values():
        b_mod = getattr(b_patch, "__module__", "") or ""
        b_qname = getattr(b_patch, "__qualname__", "") or ""
        if "nodes_sparse_attention" in b_mod or "make_h3_block_patch" in b_qname:
            raise ValueError(
                f"{node_label} block_band / 1x stream split requires dense optimized_attention or "
                "MiniMaxH3Sol, not core chunked BlockSparseAttention"
            )
    ws = H3SteerStepWorkspace()
    model_clone.add_callback_with_key(
        CallbacksMP.ON_CLEANUP,
        f"{key_prefix}_workspace",
        lambda *args, _ws=ws, **kwargs: _ws.clear(),
    )
    for b_idx in range(len(dm.blocks)):
        existing = dit_replaces.get(("double_block", b_idx))
        if not has_steer_block_patch(existing):
            model_clone.set_model_patch_replace(
                H3SteerBlockPatch(b_idx, previous=existing),
                "dit",
                "double_block",
                b_idx,
            )
    return ws
