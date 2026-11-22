"""Compiler-safe attention-hook runtime and reusable GPU workspace for `MiniMaxH3Steer`.

Implements the unified 4-mode routing architecture specified in
`docs/wiki/research/control_vectors/2026-11-22_unified_steer_attention_hook_spec.md`:

- Mode 0: pure post-refiner prelude (`band == all`, `streams == both`, zero hooks).
- Mode 1: inverted block-band shadow K/V routing (e.g. `mid_late_20_49`, `late_30_49`,
  `early_0_19`, or custom `A-B` inclusive ranges).
- Mode 2: 1x single-pass `Q_aud` split (`video_only` / `audio_only` with `auto_1x_kv`).
- Mode 3: combined block-band + 1x stream split (clean block-forked dual shadow).

All shadow staging buffers live in per-clone `H3SteerStepWorkspace` recycled in-place
under `comfy.model_prefetch.pause_malloc_graph()`, leaving `original_block(...)` inside
active `malloc_graph` compilation (`MALLOC_GRAPH_BREAKS == 0`, `MALLOC_GRAPH_ROGUES == 0`),
with zero full-sequence `K`/`V` clones (`<= 128 MiB` peak VRAM delta) and unified dual-path
coverage across `MiniMaxH3Sol`, standalone `MiniMaxH3SageAttention`, `comfy kitchen`,
Stock PyTorch SDPA, and `MiniMaxH3ExactBlocks`.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import re
from collections import OrderedDict
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

logger = logging.getLogger("ComfyUI-h3-explorations.steer_hooks")

STEER_BLOCK_BANDS: tuple[str, ...] = (
    "all",
    "mid_late_20_49",
    "late_30_49",
    "early_0_19",
)
STEER_SPLIT_BACKENDS: tuple[str, ...] = (
    "auto_1x_kv",
    "legacy_2x_pass",
)

_BAND_RANGE_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")
_WORKSPACE_CACHE_MAX = 2


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


def _slice_mod_segments(mod_segments: Sequence[tuple[int, int, Any]], tx_a: int, tx_b: int) -> list[tuple[int, int, Any]]:
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


def _apply_audio_gain_bhnd(k_bh: torch.Tensor, v_bh: torch.Tensor, options: dict[str, Any]) -> bool:
    """Apply `MiniMaxH3AudioAttentionGain` once on `BHND` if armed and not yet applied."""
    if not isinstance(options, dict):
        return False
    if options.get("_h3_audio_gain_applied") or not options.get("minimax_h3_audio_gain"):
        return False
    try:
        from . import attention as _h3_attn
    except ImportError:
        import attention as _h3_attn

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
    """Run stock H3 post-QKV RMSNorm + split-half RoPE on `[L_txt, H*D_h]` text rows."""
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
    )

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        l_txt: int,
        heads: int,
        head_dim: int,
        hidden_dim: int,
        dual_shadow: bool,
    ) -> None:
        self.h_shadow_a = torch.empty((l_txt, hidden_dim), device=device, dtype=dtype)
        self.q_a = torch.empty((1, heads, l_txt, head_dim), device=device, dtype=dtype)
        self.k_a = torch.empty((1, heads, l_txt, head_dim), device=device, dtype=dtype)
        self.v_a = torch.empty((1, heads, l_txt, head_dim), device=device, dtype=dtype)
        self.k_txt_saved = torch.empty((1, heads, l_txt, head_dim), device=device, dtype=dtype)
        self.v_txt_saved = torch.empty((1, heads, l_txt, head_dim), device=device, dtype=dtype)
        if dual_shadow:
            self.h_shadow_b = torch.empty((l_txt, hidden_dim), device=device, dtype=dtype)
            self.q_b = torch.empty((1, heads, l_txt, head_dim), device=device, dtype=dtype)
            self.k_b = torch.empty((1, heads, l_txt, head_dim), device=device, dtype=dtype)
            self.v_b = torch.empty((1, heads, l_txt, head_dim), device=device, dtype=dtype)
        else:
            self.h_shadow_b = None
            self.q_b = None
            self.k_b = None
            self.v_b = None
        self.aud_out_bhnd: torch.Tensor | None = None

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
        l_txt: int,
        heads: int,
        head_dim: int,
        hidden_dim: int,
        dual_shadow: bool,
    ) -> _WorkspaceBuffers:
        key = (str(device), str(dtype), int(l_txt), int(heads), int(head_dim), int(hidden_dim), bool(dual_shadow))
        existing = self._entries.get(key)
        if existing is not None:
            self._entries.move_to_end(key)
            return existing
        bufs = _WorkspaceBuffers(
            device=device,
            dtype=dtype,
            l_txt=l_txt,
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
    """Call-local Mode 1–3 attention routing state attached to `transformer_options`."""

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
        save_aud_k_on_block_exit: bool = True,
    ) -> None:
        self.dm = dm
        self.workspace = workspace
        self.n_blocks = len(dm.blocks)
        self.mode = int(mode)
        self.main_ctx = main_ctx
        self.shadow_a_init = shadow_a_init
        self.shadow_b_init = shadow_b_init
        self.swap_blocks = swap_blocks
        self.aud_split_blocks = aud_split_blocks
        self.dual_shadow = bool(dual_shadow)
        self.fork_b_at_block = fork_b_at_block
        self.aud_uses_shadow_b = bool(aud_uses_shadow_b)
        self.save_aud_k_on_block_exit = bool(save_aud_k_on_block_exit)
        self.active = True
        self._bufs: _WorkspaceBuffers | None = None
        self._cur: dict[str, Any] | None = None
        self.stats: dict[str, Any] = {
            "mode": self.mode,
            "kv_swaps": 0,
            "aud_splices": 0,
            "shadow_blocks": 0,
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
        au_spans = [(int(sa), int(sb)) for (sa, sb, kind) in layout.segments if kind == "audio" and sb > sa]
        total_aud = sum(b - a for a, b in au_spans)

        x_full = args["img"]
        device = x_full.device
        dtype = x_full.dtype
        hidden_dim = int(x_full.shape[-1])
        block = self.dm.blocks[block_index]
        heads = int(block.attn.heads)
        head_dim = int(block.attn.head_dim)

        if self._bufs is None or block_index == 0:
            self._bufs = self.workspace.acquire(
                device=device,
                dtype=dtype,
                l_txt=l_txt,
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
        if self.dual_shadow and self.fork_b_at_block == block_index and bufs.h_shadow_b is not None:
            bufs.h_shadow_b.copy_(x_full[tx_a:tx_b])

        in_swap = block_index in self.swap_blocks
        in_aud = bool(au_spans) and (block_index in self.aud_split_blocks)
        is_last = block_index >= (self.n_blocks - 1)

        # Does shadow_a need QKV projection at this block?
        need_a_kv = (
            in_swap
            or (in_aud and not self.aud_uses_shadow_b)
            or (
                not is_last
                and (
                    bool(self.swap_blocks)
                    or (bool(self.aud_split_blocks) and not self.aud_uses_shadow_b)
                )
            )
        )
        need_b_kv = bool(
            self.dual_shadow
            and bufs.h_shadow_b is not None
            and ((self.fork_b_at_block is None) or (block_index >= self.fork_b_at_block))
            and (in_aud or not is_last)
        )
        if not need_a_kv and not need_b_kv:
            return

        from comfy.ldm.minimax.model import _mod_scale_shift

        mod_txt = _slice_mod_segments(args["mod_segments"], tx_a, tx_b)
        rope_txt = _slice_rope(args.get("rope_freqs"), tx_a, tx_b)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(args["t_emb"])

        if need_a_kv:
            h_in_a = _mod_scale_shift(block.norm1(bufs.h_shadow_a), shift_msa, scale_msa, mod_txt)
            q_a, k_a, v_a = block.attn.qkv_proj(h_in_a).split(heads * head_dim, dim=-1)
            v_a = v_a.view(l_txt, heads, head_dim)
            q_a, k_a = _apply_rms_rope_txt(block, q_a, k_a, rope_txt, heads=heads, head_dim=head_dim, device=device)
            bufs.q_a.copy_(q_a.transpose(0, 1).unsqueeze(0))
            bufs.k_a.copy_(k_a.transpose(0, 1).unsqueeze(0))
            bufs.v_a.copy_(v_a.transpose(0, 1).unsqueeze(0))
            del h_in_a, q_a, k_a, v_a

        if need_b_kv and bufs.h_shadow_b is not None:
            assert bufs.q_b is not None and bufs.k_b is not None and bufs.v_b is not None
            h_in_b = _mod_scale_shift(block.norm1(bufs.h_shadow_b), shift_msa, scale_msa, mod_txt)
            q_b, k_b, v_b = block.attn.qkv_proj(h_in_b).split(heads * head_dim, dim=-1)
            v_b = v_b.view(l_txt, heads, head_dim)
            q_b, k_b = _apply_rms_rope_txt(block, q_b, k_b, rope_txt, heads=heads, head_dim=head_dim, device=device)
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
            "au_spans": au_spans,
            "total_aud": total_aud,
            "mod_txt": mod_txt,
            "gate_msa": gate_msa,
            "shift_mlp": shift_mlp,
            "scale_mlp": scale_mlp,
            "gate_mlp": gate_mlp,
            "need_a_kv": need_a_kv,
            "need_b_kv": need_b_kv,
            "need_finish_a": (not is_last) and need_a_kv,
            "need_finish_b": (not is_last) and need_b_kv,
            "shadow_a_attn_2d": None,
            "shadow_b_attn_2d": None,
            "done_pre": False,
            "has_aud_splice": False,
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

        # Zero-copy views into BHND (`seq_axis == 2`) regardless of incoming layout.
        if seq_axis == 1:
            q_bh = q_p.transpose(1, 2)
            k_bh = k_p.transpose(1, 2)
            v_bh = v_p.transpose(1, 2)
        else:
            q_bh, k_bh, v_bh = q_p, k_p, v_p

        tx_a = cur["tx_a"]
        tx_b = cur["tx_b"]
        l_txt = cur["l_txt"]
        heads = cur["heads"]
        head_dim = cur["head_dim"]
        inner = heads * head_dim
        au_spans = cur["au_spans"]
        total_aud = cur["total_aud"]
        block_index = cur["block_index"]
        in_swap = block_index in self.swap_blocks
        in_aud = bool(au_spans) and (block_index in self.aud_split_blocks)
        k_dtype = k_bh.dtype
        v_dtype = v_bh.dtype

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

        need_q_a = cur["need_finish_a"] or (in_aud and not self.aud_uses_shadow_b)
        need_q_b = cur["need_finish_b"] or (in_aud and self.aud_uses_shadow_b)

        # Phase 1: If shadow_b (clean base) runs this block (Mode 3 dual-shadow), attend q_b / q_aud against k_b
        if self.dual_shadow and cur["need_b_kv"] and need_q_b:
            assert bufs.q_b is not None and bufs.k_b is not None and bufs.v_b is not None
            if not in_swap:
                _save_main()
            k_slice.copy_(bufs.k_b.to(dtype=k_dtype))
            v_slice.copy_(bufs.v_b.to(dtype=v_dtype))
            q_parts_b: list[torch.Tensor] = []
            if cur["need_finish_b"]:
                q_parts_b.append(bufs.q_b.to(dtype=k_dtype))
            if in_aud and self.aud_uses_shadow_b:
                for a, b in au_spans:
                    q_parts_b.append(q_bh[:, :, a:b, :].to(dtype=k_dtype))
            if q_parts_b:
                q_comb_b = q_parts_b[0] if len(q_parts_b) == 1 else torch.cat(q_parts_b, dim=2)
                attn_b = F.scaled_dot_product_attention(q_comb_b, k_bh, v_bh, dropout_p=0.0, is_causal=False)
                off_b = 0
                if cur["need_finish_b"]:
                    cur["shadow_b_attn_2d"] = (
                        attn_b[:, :, :l_txt, :][0].transpose(0, 1).reshape(l_txt, inner).to(dtype=bufs.h_shadow_a.dtype)
                    )
                    off_b = l_txt
                if in_aud and self.aud_uses_shadow_b:
                    aud_buf = bufs.ensure_aud_out(total_aud, heads, head_dim, k_bh.device, attn_b.dtype)
                    aud_buf.copy_(attn_b[:, :, off_b : off_b + total_aud, :])
                    cur["has_aud_splice"] = True
                del q_comb_b, attn_b
            if not in_swap:
                _restore_main()

        # Phase 2: Shadow A (steered in Mode 1 / 3 / audio_only, unsteered base in video_only Mode 2)
        run_shadow_a_sdpa = False
        if cur["need_a_kv"] and need_q_a:
            # In Mode 1 suffix blocks (in_swap and not in_aud): if main keeps k_a/v_a, we must update main slice BEFORE SDPA
            if not in_swap:
                _save_main()
            k_slice.copy_(bufs.k_a.to(dtype=k_dtype))
            v_slice.copy_(bufs.v_a.to(dtype=v_dtype))
            q_parts_a: list[torch.Tensor] = []
            if cur["need_finish_a"]:
                q_parts_a.append(bufs.q_a.to(dtype=k_dtype))
            if in_aud and not self.aud_uses_shadow_b:
                for a, b in au_spans:
                    q_parts_a.append(q_bh[:, :, a:b, :].to(dtype=k_dtype))
            if q_parts_a:
                run_shadow_a_sdpa = True
                q_comb_a = q_parts_a[0] if len(q_parts_a) == 1 else torch.cat(q_parts_a, dim=2)
                attn_a = F.scaled_dot_product_attention(q_comb_a, k_bh, v_bh, dropout_p=0.0, is_causal=False)
                off_a = 0
                if cur["need_finish_a"]:
                    cur["shadow_a_attn_2d"] = (
                        attn_a[:, :, :l_txt, :][0].transpose(0, 1).reshape(l_txt, inner).to(dtype=bufs.h_shadow_a.dtype)
                    )
                    off_a = l_txt
                if in_aud and not self.aud_uses_shadow_b:
                    aud_buf = bufs.ensure_aud_out(total_aud, heads, head_dim, k_bh.device, attn_a.dtype)
                    aud_buf.copy_(attn_a[:, :, off_a : off_a + total_aud, :])
                    cur["has_aud_splice"] = True
                del q_comb_a, attn_a
            if not in_swap:
                _restore_main()
        elif in_swap and cur["need_a_kv"]:
            # Final swap block (e.g. block 49 in Mode 1/3) where shadow_a is done updating MLP: just write k_a/v_a for main
            k_slice.copy_(bufs.k_a.to(dtype=k_dtype))
            v_slice.copy_(bufs.v_a.to(dtype=v_dtype))

        if in_swap:
            self.stats["kv_swaps"] += 1
        if run_shadow_a_sdpa or (self.dual_shadow and cur["need_b_kv"]):
            self.stats["shadow_blocks"] += 1
        self.stats["blocks_touched"].append(block_index)
        return True

    def reset_pre_block(self) -> None:
        cur = self._cur
        if cur is not None:
            cur["done_pre"] = False
            cur["has_aud_splice"] = False

    def post_attention(
        self,
        out_main: torch.Tensor,
        *,
        nhd: bool = False,
        skip_output_reshape: bool = False,
    ) -> None:
        cur = self._cur
        bufs = self._bufs
        if not self.active or cur is None or bufs is None or not cur["has_aud_splice"]:
            return
        aud_bhnd = bufs.aud_out_bhnd
        if aud_bhnd is None:
            return
        au_spans = cur["au_spans"]
        if not au_spans:
            return

        off = 0
        if nhd:
            # Direct Sage/LSE NHD layout: `[1, S, H, D_h]`
            aud_src = aud_bhnd.transpose(1, 2).to(dtype=out_main.dtype)
            for a, b in au_spans:
                n_seg = b - a
                out_main[:, a:b, :, :].copy_(aud_src[:, off : off + n_seg, :, :])
                off += n_seg
        elif skip_output_reshape:
            # Raw `BHND` layout: `[1, H, S, D_h]`
            aud_src = aud_bhnd.to(dtype=out_main.dtype)
            for a, b in au_spans:
                n_seg = b - a
                out_main[:, :, a:b, :].copy_(aud_src[:, :, off : off + n_seg, :])
                off += n_seg
        else:
            # Standard reshaped output `[1, S, H*D_h]` consumed by `self.out_proj(out.squeeze(0))`
            total_aud = cur["total_aud"]
            inner = cur["heads"] * cur["head_dim"]
            aud_2d = aud_bhnd[0].transpose(0, 1).reshape(1, total_aud, inner).to(dtype=out_main.dtype)
            for a, b in au_spans:
                n_seg = b - a
                out_main[:, a:b, :].copy_(aud_2d[:, off : off + n_seg, :])
                off += n_seg

        cur["has_aud_splice"] = False
        self.stats["aud_splices"] += 1

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
        mod_txt = cur["mod_txt"]
        gate_msa = cur["gate_msa"]
        shift_mlp = cur["shift_mlp"]
        scale_mlp = cur["scale_mlp"]
        gate_mlp = cur["gate_mlp"]

        if cur["need_finish_a"] and cur["shadow_a_attn_2d"] is not None:
            attn_out = block.attn.out_proj(cur["shadow_a_attn_2d"])
            cur["shadow_a_attn_2d"] = None
            _mod_gate(bufs.h_shadow_a, gate_msa, attn_out, mod_txt)
            del attn_out
            h_mlp = _mod_scale_shift(block.norm2(bufs.h_shadow_a), shift_mlp, scale_mlp, mod_txt)
            _mod_gate(bufs.h_shadow_a, gate_mlp, block.mlp(h_mlp), mod_txt)
            del h_mlp

        if (
            cur["need_finish_b"]
            and cur["shadow_b_attn_2d"] is not None
            and bufs.h_shadow_b is not None
        ):
            attn_out_b = block.attn.out_proj(cur["shadow_b_attn_2d"])
            cur["shadow_b_attn_2d"] = None
            _mod_gate(bufs.h_shadow_b, gate_msa, attn_out_b, mod_txt)
            del attn_out_b
            h_mlp_b = _mod_scale_shift(block.norm2(bufs.h_shadow_b), shift_mlp, scale_mlp, mod_txt)
            _mod_gate(bufs.h_shadow_b, gate_mlp, block.mlp(h_mlp_b), mod_txt)
            del h_mlp_b


class H3SteerBlockPatch:
    """Stateless per-clone `patches_replace['dit'][('double_block', i)]` adapter.

    Keeps `extra_args['original_block'](args)` inside active `malloc_graph` compilation
    while isolating Mode 1–3 shadow state transitions under `pause_malloc_graph()`.
    """

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
