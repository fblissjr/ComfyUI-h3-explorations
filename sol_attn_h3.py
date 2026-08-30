"""Sol-Attn for MiniMax-H3, on comfy-kitchen's merged CUDA kernel.

## This is a FORK, and vendor/README.md's rules are why it exists here

`vendor/sol_attn_minimax.py` is upstream's node, and its whole value is that
it is upstream's: when it and our expectations disagree, that disagreement is
a finding rather than a merge artifact. That property was spent on 2026-08-29,
when the merged kernel's API change was absorbed by editing the vendored file
in place -- v3.1 and v3.2 in its lineage table are ours, not drops. So this
file takes vendor/README.md's third option, the one it says not to reach for
casually: fork it, rename it so the fork is obvious, and record the divergence.

Forked from vendored v3.2. Every local change is listed below; nothing else
differs, which is what keeps a future upstream drop a diff rather than a
rewrite.

  - **`centroid_tail` and `reuse_qkv_memory` are gone**, along with the
    signature probing that existed to pass them where they were accepted.
    Comfy-Org/comfy-kitchen#117 removed both from every entry. Keeping them as
    inert widgets was the right call for a file whose node id sits in 145
    saved graphs; it is the wrong call for a new node id, which pays no such
    debt.
  - **The kernel API is asserted once, at patch time**, rather than probed per
    call. A missing kwarg used to surface as a TypeError inside `override`,
    which catches everything and falls through to dense -- so a build without
    the argument rendered successfully, slower and numerically different, and
    said nothing. `_require_kernel` fails the node before sampling instead.
  - **`pooled_tail` is exposed.** That is the kernel's `tail`, renamed here
    because `centroid_tail` occupied the short name for two weeks and meant
    something else entirely. See the section below.
  - **`morton_curve` offers `hilbert` directly.** `sol_curves.install()`
    monkey-patched `morton_perm` on the live vendored module to add it, which
    needed the module resolved by identity and a zero-patched-modules guard.
    Owning the file removes the problem rather than guarding it.

## What the kernel takes, and what this node does with each

Read from `comfy_kitchen.sol_attn`'s signature and from `constraints.py`'s
`sol_attn_common_call_rule`, not from a version string: both the pre-merge and
post-merge builds call themselves `0.2.31`.

  tau, topk_ratio   selection, exposed as the `selection` combo.
  scale             taken from the caller's kwargs.
  sink_blocks       H3's conditioning sink, derived from the packed layout.
  sink_q            see `_sink_blocks`.
  tail              exposed as `pooled_tail`.
  key_bias          NOT exposed. Reachable, and legal only where the biased
                    keys are sink-covered, since the pooled term cannot see a
                    per-token bias -- which on H3 means the conditioning rows
                    and nothing else. It would be a prompt-adherence knob the
                    model was never trained against, so it is documented here
                    rather than offered. `_ineligible` also declines masked
                    attention outright, which is the only form core would hand
                    us one in.
  block_len         NOT exposed, and inert for this path. It marks live rows
                    in a caller-PADDED block. H3's packed sequence is
                    contiguous and pads nothing, so the kernel derives the
                    ragged final block from T on its own. The one caller that
                    needs it is VSA cube tiling -- see the VSA node.
  coarse_gate       NOT exposed here, and it cannot be. It is a learned
                    projection of the BLOCK INPUT, and this node installs an
                    `optimized_attention_override`, which receives Q/K/V
                    already built. Reaching it means replacing the block
                    forward, which is a different node.

`sol_attn_chunked`, the second entry, is also out of reach from here for the
same structural reason: it consumes fused qkv projection chunks and applies
rope and RMSNorm itself, so there is nothing left for an attention override to
be handed.

## `pooled_tail`, which is the one new knob with teeth

Sol-Attn's contribution is not the routing. Unselected blocks contribute one
pooled term each, so the whole sequence stays in the softmax denominator, and
the paper's ablation shows that advantage WIDENING as sparsity rises.
`pooled_tail=False` deletes it: softmax over the routed blocks only.

That is not a speed knob to reach for on this model. It is there because it is
what SLA is -- with `top-k (SLA)` selection it reproduces the routing the
lightx2v Turbo-SLA LoRA was distilled under, on the CUDA kernel and through
`optimized_attention`, which reaches all 52 `Attention` modules rather than
the 50 that `MiniMaxH3SLARouter`'s object patch could see. Covering the two
token-refiner calls also needs `min_tokens` dropped below their ~311 rows;
at the shipped threshold they stay dense.

Requires comfy_kitchen with `sol_attn` (bf16, head_dim 128, sm_80+);
everything else falls back to the existing attention backend.
"""

import logging
import re
import sys
from functools import partial

import torch

from comfy_api.latest import io

from .block_spec import parse_blocks

try:
    import comfy_kitchen as _ck
    _CK_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency
    _ck = None
    _CK_IMPORT_ERROR = exc

HEAD_DIM = 128
BLOCK_SIZE = 64

_stats = {"sparse": 0, "dense_fallback": 0, "outside_range": 0,
          "dense_block": 0, "errors": 0}
_seen = set()
_BLOCK_INDEX_HOOKED = set()


def parse_tau_profile(spec, count):
    """Parse "0-30=2.0; 39-42=0.9" into {block: tau}.

    Entries are separated by ';' or newlines, so a multiline text node works as
    well as a single line, and '#' starts a comment. Blocks not listed keep the
    node's base tau; the block side takes dense_blocks syntax, so "0-2,47=1.8"
    is valid. Later entries win where they overlap.
    """
    profile = {}
    for entry in re.split(r"[;\n]", str(spec)):
        entry = entry.split("#", 1)[0].strip()
        if not entry:
            continue
        blocks, sep, value = entry.partition("=")
        if not sep:
            raise ValueError(f"tau_profile entry {entry!r} needs '=', e.g. '39-42=0.9'")
        try:
            level = float(value)
        except ValueError:
            raise ValueError(f"tau_profile entry {entry!r} has a non-numeric tau")
        for block in parse_blocks(blocks, count):
            profile[block] = level
    return profile


def _install_block_index(model):
    """Publish the running block index into transformer_options."""
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return False
    if id(model) in _BLOCK_INDEX_HOOKED:
        return True

    def make_hook(index):
        def hook(_module, _args, kwargs):
            options = kwargs.get("transformer_options")
            if isinstance(options, dict):
                options["sol_block"] = index
            return None
        return hook

    for index, block in enumerate(blocks):
        block.register_forward_pre_hook(make_hook(index), with_kwargs=True)
    _BLOCK_INDEX_HOOKED.add(id(model))
    return True


def sol_attn_stats():
    """Dispatch counters since process start (or last reset)."""
    return dict(_stats)


def reset_sol_attn_stats():
    for key in _stats:
        _stats[key] = 0
    _seen.clear()


def _log_once(key, message):
    if key not in _seen:
        _seen.add(key)
        logging.info(f"[h3-sol] {message}")


def _log_kernel_failure(exc):
    # Full traceback on the first distinct failure, short line on repeats.
    key = ("kernel_failure", type(exc).__name__, str(exc))
    first = key not in _seen
    _seen.add(key)
    logging.error(f"[h3-sol] kernel failed ({exc}); falling back", exc_info=first)


# ---------------------------------------------------------------------------
# Morton (Z-order) reordering of H3's video span. Inlined from the reference
# pack's _morton.py / _morton_h3.py, minus the Wan path.
# ---------------------------------------------------------------------------

_PERM_CACHE = {}

# Curve names owned by `sol_curves` rather than by this file. Imported lazily
# inside `morton_perm` so this module stays importable without torch-heavy
# siblings resolved, and named here so the combo and the dispatch cannot drift.
_CURVES_OURS = ("hilbert",)


def morton_perm(grid, device, curve="3d"):
    """Token permutation and its inverse, for one video grid.

    `hilbert` is dispatched to `sol_curves`. In the vendored node that curve
    arrived by rebinding this global on the live module at execute time, which
    needed the module resolved by identity because a running ComfyUI can hold
    two module objects for one file. Owning the file makes it an import.

    curve="3d"       interleave t/h/w equally.
    curve="2d_frame" Z-order within each frame, frames left in original order.
                     Use this when the temporal axis is not uniformly spaced --
                     MiniMax-H3's FRAME_PER_TOKEN is (1, 4, 4, 4, 4), so
                     index-adjacent frames are 1 or 4 real frames apart and a 3D
                     curve groups temporally distant tokens together.
    """
    if curve in _CURVES_OURS:
        from .sol_curves import hilbert_perm
        return hilbert_perm(grid, device)
    key = (tuple(int(x) for x in grid), curve)
    hit = _PERM_CACHE.get(key)
    if hit is None:
        frames, height, width = key[0]
        linear = torch.arange(frames * height * width, dtype=torch.int64)
        area = height * width
        z = linear // area
        rem = linear - z * area
        y = rem // width
        x = rem - y * width

        def part1by2(value):
            value = value & 0x1FFFFF
            value = (value | (value << 32)) & 0x1F00000000FFFF
            value = (value | (value << 16)) & 0x1F0000FF0000FF
            value = (value | (value << 8)) & 0x100F00F00F00F00F
            value = (value | (value << 4)) & 0x10C30C30C30C30C3
            value = (value | (value << 2)) & 0x1249249249249249
            return value

        if curve == "2d_frame":
            # frame index stays the most significant key, so frames never mix
            code = (z << 42) | part1by2(x) | (part1by2(y) << 1)
        else:
            code = part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2)
        perm = linear[torch.argsort(code)]
        hit = (perm, torch.argsort(perm))
        _PERM_CACHE[key] = hit
    return hit[0].to(device), hit[1].to(device)



def _h3_log_once(message):
    _log_once(("h3", message), f"H3 Morton: {message}")


_INSTALLED = set()
_PATCHED_LAYOUTS = set()
BLOCK_SIZE = 64  # kernel block size; the permutation is aligned to this grid
_DEVICE_CACHE = {}
# id(position_ids) -> (layout, span). The layout is kept alive deliberately so
# the id cannot be recycled underneath us; there is one entry per distinct shape.
_SPANS = {}


def _perm_for(grid, curve, device, start):
    """Morton permutation for a video span starting at absolute row ``start``.

    The kernels block from absolute position 0, so a span that does not start on
    a block boundary splits every Z-order cell across two blocks, joining
    opposite ends of the volume. Rotating by the misalignment realigns the
    cells; the ragged group it displaces lands in the block shared with the
    conditioning rows, which the exact-KV sink already keeps exact.
    """
    pad = (-int(start)) % BLOCK_SIZE
    key = (tuple(grid), curve, str(device), pad)
    hit = _DEVICE_CACHE.get(key)
    if hit is None:
        perm, inverse = morton_perm(grid, device, curve)
        if pad:
            perm = torch.roll(perm, pad)
            inverse = torch.argsort(perm)
        hit = (perm, inverse)
        _DEVICE_CACHE[key] = hit
    return hit


def _video_span(layout, latent_t, latent_h, latent_w):
    """(start, stop, grid) for the target video segment, or None.

    The grid is stored rather than a permutation so the curve can be changed
    between runs without rebuilding layouts.
    """
    segments = getattr(layout, "segments", None)
    if not segments:
        return None
    span = next(((a, b) for a, b, kind in segments if kind == "video"), None)
    if span is None:
        return None
    start, stop = span
    grid = (int(latent_t), int(latent_h) // 2, int(latent_w) // 2)
    if grid[0] * grid[1] * grid[2] != stop - start:
        logging.info(f"[h3-sol] H3 Morton skipped: video segment {stop - start} rows "
                     f"does not match grid {grid}")
        return None
    return start, stop, grid


def _patch_packed_layout(module):
    """Register the video span of every PackedLayout built, without mutating it."""
    layout_cls = getattr(module, "PackedLayout", None)
    if layout_cls is None:
        raise RuntimeError(f"{module.__name__} has no PackedLayout")
    if id(layout_cls) in _PATCHED_LAYOUTS:
        return
    original_init = layout_cls.__init__

    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t, *args, **kwargs):
        original_init(self, text_len, latent_t, latent_h, latent_w, audio_t, *args, **kwargs)
        try:
            span = _video_span(self, latent_t, latent_h, latent_w)
        except Exception as exc:                      # never break model construction
            logging.info(f"[h3-sol] H3 Morton span resolution failed: {exc}")
            span = None
        segs = getattr(self, "segments", []) or []
        bounds = next(((a, b) for a, b, kind in segs if kind == "video"), None)
        # Target audio is the segment immediately before video; sink_q only
        # needs THOSE query rows dense, not the (possibly huge) reference rows.
        audio = next(((a, b) for a, b, kind in segs if kind == "audio"), None)
        if torch.is_tensor(getattr(self, "position_ids", None)) and bounds is not None:
            _SPANS[id(self.position_ids)] = (self, bounds, audio, span)

    layout_cls.__init__ = __init__
    _PATCHED_LAYOUTS.add(id(layout_cls))


def install_h3_morton(model):
    """Idempotently install the reorder. Inert without transformer_options['sol_morton']."""
    if id(model) in _INSTALLED:
        return
    for attr in ("rope_freqs", "_forward", "blocks"):
        if not hasattr(model, attr):
            raise RuntimeError(f"MiniMax-H3 Morton needs .{attr} on the diffusion model")

    _patch_packed_layout(sys.modules[type(model).__module__])

    original_forward = model._forward
    original_rope_freqs = model.rope_freqs

    def _forward(x, timestep, context, transformer_options={}, **kwargs):
        previous = getattr(model, "_sol_morton_active", False)
        model._sol_morton_active = bool(transformer_options.get("sol_morton"))
        model._sol_morton_curve = transformer_options.get("sol_morton_curve", "3d")
        model._sol_transformer_options = transformer_options
        try:
            return original_forward(x, timestep, context,
                                    transformer_options=transformer_options, **kwargs)
        finally:
            model._sol_morton_active = previous
            model._sol_morton_span = None
            model._sol_morton_state = None
            model._sol_transformer_options = None
            transformer_options.pop("sol_h3_video_span", None)
            transformer_options.pop("sol_h3_audio_span", None)

    def rope_freqs(position_ids, device):
        """Publish the layout only. Permuting happens in the block hook, which is
        the single place that sees the tokens and the rope table together."""
        model._sol_morton_span = None
        model._sol_morton_state = None
        entry = _SPANS.get(id(position_ids))
        if entry is None:
            _h3_log_once("no layout registered; Morton and the conditioning sink are inactive")
            return original_rope_freqs(position_ids, device)

        _layout, bounds, audio, span = entry
        # Publish the video-segment boundary so the attention override can keep
        # the conditioning rows (text / audio / reference) exact.
        options = getattr(model, "_sol_transformer_options", None)
        if options is not None:
            options["sol_h3_video_span"] = bounds
            options["sol_h3_audio_span"] = audio
        if getattr(model, "_sol_morton_active", False):
            if span is None:
                _h3_log_once("video grid does not match the segment; Morton inactive")
            else:
                model._sol_morton_span = span
        return original_rope_freqs(position_ids, device)

    def pre_hook(module, args):
        # Blocks are called as block(h, t_emb, mod_segments, rope_freqs, ...).
        if len(args) < 4:
            return None

        if module is not first:
            state = getattr(model, "_sol_morton_state", None)
            if state is None:
                return None
            # Hidden states are already reordered; reuse the table built once.
            return args[:3] + (state[1],) + tuple(args[4:])

        model._sol_morton_state = None
        span = getattr(model, "_sol_morton_span", None)
        if span is None:
            return None
        start, stop, grid = span
        h, rope = args[0], args[3]

        # One decision, both tensors in hand. Splitting this across rope_freqs
        # and the hook is what corrupts output when the two guards disagree.
        if not torch.is_tensor(h) or h.ndim != 2 or h.shape[0] < stop:
            _h3_log_once(f"hidden states do not cover the video span {(start, stop)}; inactive")
            return None
        if not torch.is_tensor(rope) or rope.ndim < 2 or rope.shape[1] != h.shape[0]:
            _h3_log_once(f"rope table {tuple(rope.shape) if torch.is_tensor(rope) else type(rope)} "
                      f"does not match {h.shape[0]} tokens; inactive")
            return None

        curve = getattr(model, "_sol_morton_curve", "3d")
        _h3_log_once(f"ACTIVE: video span [{start}, {stop}), grid {grid}, "
                     f"curve {curve}")
        perm, inverse = _perm_for(grid, curve, h.device, start)
        h = h.clone()
        h[start:stop] = h[start:stop][perm]

        # rope_rotation_table is elementwise per row, so permuting the table is
        # identical to permuting position_ids -- and keeps the decision here.
        full = torch.arange(rope.shape[1], device=rope.device)
        full[start:stop] = perm + start
        rope = rope.index_select(1, full)

        model._sol_morton_state = (inverse, rope)
        return (h,) + args[1:3] + (rope,) + tuple(args[4:])

    def post_hook(_module, _args, output):
        state = getattr(model, "_sol_morton_state", None)
        span = getattr(model, "_sol_morton_span", None)
        model._sol_morton_state = None
        if state is None or span is None:
            return None
        start, stop, _grid = span
        inverse, _rope = state
        if not torch.is_tensor(output) or output.shape[0] < stop:
            logging.error("[h3-sol] H3 Morton: block stack changed the token count "
                          f"({tuple(output.shape)}); cannot restore order")
            return None
        output = output.clone()
        output[start:stop] = output[start:stop][inverse]
        return output

    first = model.blocks[0]
    model._forward = _forward
    model.rope_freqs = rope_freqs
    for block in model.blocks:
        block.register_forward_pre_hook(pre_hook)
    model.blocks[-1].register_forward_hook(post_hook)
    _INSTALLED.add(id(model))



def _ineligible(q, k, mask, dim_head, min_tokens):
    """Why this call can't use Sol-Attn, or None if it can. q/k are BTHD."""
    if _ck is None or not hasattr(_ck, "sol_attn"):
        return "comfy_kitchen sol_attn unavailable"
    if q.device.type != "cuda":
        return "not cuda"
    if q.dtype != torch.bfloat16:
        return f"dtype {q.dtype} (kernel is bf16-only)"
    if dim_head != HEAD_DIM:
        return f"head_dim {dim_head} != 128"
    if mask is not None:
        return "masked attention"
    if q.shape[1] != k.shape[1]:
        return "cross-attention (kept dense)"
    if q.shape != k.shape:
        # GQA or any other q/k mismatch would silently index wrong.
        return f"q/k shape mismatch {tuple(q.shape)} vs {tuple(k.shape)}"
    if q.shape[1] < min_tokens:
        return f"seq {q.shape[1]} < {min_tokens}"
    return None


# What this node passes on every eligible call. Asserted ONCE, at patch time,
# against the entry that will actually be called.
#
# **Not probed per call, and that is the whole point.** A kwarg the build does
# not take raises a TypeError inside `_run`, and `override` catches every
# exception and falls through to `dense()`. So an API mismatch used to render
# successfully -- slower, numerically different, silent. The vendored node
# solved that by adapting to whatever the signature accepted; this one refuses
# to start instead, because there is exactly one supported kernel now and
# quietly running a different computation is the failure mode to avoid.
_REQUIRED_KERNEL_KWARGS = ("tau", "scale", "sink_blocks", "sink_q",
                           "topk_ratio", "tail")


def _require_kernel():
    """Raise unless `comfy_kitchen.sol_attn` accepts everything we pass.

    Called from `_apply_patch`, which propagates out of `execute` and fails the
    node before sampling. Never call it from the dispatch path: an exception
    raised there is swallowed by `override` and becomes the silent dense
    render this exists to prevent.
    """
    if _ck is None:
        raise RuntimeError(f"comfy_kitchen is not importable: {_CK_IMPORT_ERROR}")
    fn = getattr(_ck, "sol_attn", None)
    if fn is None:
        raise RuntimeError(
            "the installed comfy_kitchen has no sol_attn. The stock PyPI wheel "
            "declares the same version as a build carrying it, so read the "
            "local version segment (comfy_kitchen-<version>.dist-info), not "
            "the version. bench/check_sol_kernel.py reports which is installed.")
    import inspect
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return                      # not introspectable; the call itself decides
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return
    missing = [k for k in _REQUIRED_KERNEL_KWARGS if k not in params]
    if missing:
        raise RuntimeError(
            f"comfy_kitchen.sol_attn does not accept {missing}. This node "
            f"targets the merged kernel (Comfy-Org/comfy-kitchen#117); a "
            f"pre-merge build takes centroid_tail and max_blocks instead of "
            f"tail. Both builds report version 0.2.31, so upgrade by the local "
            f"version segment rather than the version.")


def _run(q, k, v, heads, skip_reshape, skip_output_reshape, scale,
         tau, min_tokens, verbose, sink_blocks=(0, 0), sink_q=(0, 0),
         topk_ratio=0.0, tail=True):
    """Returns the attention output, or None if this call should stay dense."""
    if skip_reshape:
        b, _, _, dim_head = q.shape          # BHND
        qs, ks, vs = (t.transpose(1, 2) for t in (q, k, v))
    else:
        b, _, dim_head = q.shape             # B, N, heads*dim_head
        dim_head //= heads
        qs, ks, vs = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

    reason = _ineligible(qs, ks, None, dim_head, min_tokens)
    if reason is not None:
        _stats["dense_fallback"] += 1
        if verbose:
            _log_once((tuple(qs.shape), reason), f"dense {tuple(qs.shape)}: {reason}")
        return None

    # One call, no signature adaptation. `_require_kernel` has already
    # asserted at patch time that this build takes every one of these, which
    # is the only place such a check can be made: raised from here it would be
    # caught by `override` and turned into a silent dense render.
    #
    # `key_bias`, `block_len` and `coarse_gate` are left at their defaults --
    # None, None and None. The module docstring says why each is unreachable
    # or inert on this path rather than merely unused.
    out = _ck.sol_attn(qs, ks, vs, tau=tau, scale=scale,
                       sink_blocks=list(sink_blocks), sink_q=list(sink_q),
                       topk_ratio=topk_ratio, tail=tail)      # BTHD
    _stats["sparse"] += 1
    if verbose:
        sel = (f"topk={topk_ratio:.3f}" if topk_ratio else f"tau={tau}")
        _log_once((tuple(qs.shape), "sparse", tail),
                  f"sparse {tuple(qs.shape)} {sel} cuda-int8"
                  + ("" if tail else " NO POOLED TAIL (SLA/VSA fine stage)"))

    if skip_output_reshape:
        return out.transpose(1, 2)           # BHND
    return out.reshape(b, -1, heads * dim_head)


def _sink_blocks(transformer_options, tokens, mode):
    """(exact-KV blocks, dense-query blocks) for MiniMax-H3's conditioning rows.

    H3 packs [text][cond][ref][audio][video] into one sequence; sparsifying the
    conditioning rows costs sync and prompt adherence. exact_kv measures ~3%,
    exact_kv_and_rows ~17%, so exact-KV is the default and rows are opt-in.
    """
    if mode == "off":
        return (0, 0), (0, 0)
    span = (transformer_options or {}).get("sol_h3_video_span")
    if span is None:
        return (0, 0), (0, 0)
    video_start, video_stop = span
    if tokens < video_stop or video_start <= 0:
        return (0, 0), (0, 0)
    blocks = (0, (video_start + BLOCK_SIZE - 1) // BLOCK_SIZE)
    if mode != "exact_kv_and_rows":
        return blocks, (0, 0)
    # Dense-query protection exists for the TARGET AUDIO rows; reference rows
    # only need the exact-KV side. Fall back to the whole conditioning range
    # when the layout did not publish an audio span.
    audio = (transformer_options or {}).get("sol_h3_audio_span")
    if audio is None:
        return blocks, blocks
    audio_start, _audio_stop = audio
    return blocks, (audio_start // BLOCK_SIZE, blocks[1])


def make_override(tau=1.0, min_tokens=4096,
                  sigma_start=None, sigma_end=None, verbose=False,
                  sink_conditioning="exact_kv", dense_blocks=frozenset(),
                  tau_profile=None, previous=None, topk_ratio=0.0, tail=True):
    """Build an optimized_attention_override callable.

    ``previous`` chains any override already installed on the model: every path
    that declines hands off to it first, falling through to ``func`` only if
    there is none.
    """

    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):

        def dense():
            target = func if previous is None else partial(previous, func)
            return target(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                          skip_reshape=skip_reshape,
                          skip_output_reshape=skip_output_reshape, **kwargs)

        if mask is not None:
            _stats["dense_fallback"] += 1
            return dense()

        # Depth gates: a block can be kept dense outright or given its own tau.
        block = None
        if dense_blocks or tau_profile:
            block = kwargs.get("transformer_options", {}).get("sol_block")
        if block in dense_blocks:
            _stats["dense_block"] += 1
            return dense()
        block_tau = tau_profile.get(block, tau) if tau_profile else tau

        # Sampling-percentage gate, so the paper's dense warm-up steps work.
        if sigma_start is not None or sigma_end is not None:
            sigmas = kwargs.get("transformer_options", {}).get("sigmas")
            if sigmas is not None:
                sigma = float(sigmas[0])
                if (sigma_start is not None and sigma > sigma_start) or \
                   (sigma_end is not None and sigma < sigma_end):
                    _stats["outside_range"] += 1
                    return dense()

        tokens = q.shape[2] if skip_reshape else q.shape[1]
        sink, sink_q = _sink_blocks(kwargs.get("transformer_options"), tokens,
                                    sink_conditioning)
        if verbose and sink != (0, 0):
            _log_once((tokens, sink, sink_q),
                      f"conditioning sink: KV blocks {sink} exact, dense query blocks {sink_q}")

        try:
            out = _run(q, k, v, heads, skip_reshape, skip_output_reshape,
                       kwargs.get("scale", None), block_tau, min_tokens, verbose,
                       sink, sink_q, topk_ratio, tail)
        except Exception as exc:
            _stats["errors"] += 1
            _log_kernel_failure(exc)
            return dense()
        return dense() if out is None else out

    return override


def _compose_module_patch(module, patched_forward):
    """Gate an object-patched attention forward (e.g. KJNodes' mem-efficient
    Sage): calls Sol-Attn would take run the stock forward and reach the
    override; the rest keeps the patch. Gate params come from
    transformer_options["sol_compose"]; when absent the patch runs as-is.
    """
    stock = type(module).forward

    def forward(*args, **kwargs):
        options = kwargs.get("transformer_options")
        if not isinstance(options, dict):
            options = next((a for a in args if isinstance(a, dict) and "sol_compose" in a), {})
        gate = options.get("sol_compose")
        x = args[0] if args else None
        # KJNodes' low-VRAM block patch hands x over in a single-item list.
        tensor = x[0] if isinstance(x, list) and len(x) == 1 and torch.is_tensor(x[0]) else x
        take = gate is not None and torch.is_tensor(tensor) and tensor.device.type == "cuda" \
            and tensor.dtype == torch.bfloat16 and tensor.ndim in (2, 3)
        if take:
            # H3 packs tokens first (s, dim); Wan/LTX2 are batch-first.
            tokens = tensor.shape[0] if tensor.ndim == 2 else tensor.shape[1]
            take = tokens >= gate["min_tokens"]
        if take:
            sigmas = options.get("sigmas")
            if sigmas is not None:
                sigma = float(sigmas[0])
                take = not (sigma > gate["sigma_start"] or sigma < gate["sigma_end"])
        if take:
            delegate = options.get("sol_take_forward")
            if delegate is not None:
                # a cooperating patch's forward that reaches optimized_attention while
                # keeping its own low-VRAM behavior; preferred over the stock forward
                return delegate(module, *args, **kwargs)
            if tensor is not x:
                x.clear()  # the stock forward wants the tensor; consume the hand-off list
                args = (tensor,) + args[1:]
            return stock(module, *args, **kwargs)
        return patched_forward(*args, **kwargs)

    forward._sol_composed = True
    return forward


_COMPOSE_HOOKED = set()


def _install_compose_hooks(model, attn_attr):
    """Compose at sampling time, once all object patches are applied: a node
    downstream of ours overwrites the same object-patch key, so execute-time
    composition alone loses. The pre-hooks re-wrap any foreign attn forward
    before each block runs; inert unless sol_compose is published.
    """
    if id(model) in _COMPOSE_HOOKED:
        return

    def pre_hook(block, args):
        attn = getattr(block, attn_attr, None)
        if attn is None:
            return None
        fwd = attn.__dict__.get("forward")
        if fwd is None or getattr(fwd, "_sol_composed", False):
            return None
        if getattr(fwd, "_uses_optimized_attention", False):
            return None  # patch routes through optimized_attention; the override composes directly
        if getattr(fwd, "__func__", None) is type(attn).forward:
            return None  # unpatch leaves the stock forward as an instance attr
        attn.forward = _compose_module_patch(attn, fwd)
        _log_once(("composed", attn_attr),
                  f"composing with a patched {attn_attr}.forward; Sol-Attn takes "
                  "eligible self-attention calls, the patch keeps the rest")
        return None

    for block in model.blocks:
        block.register_forward_pre_hook(pre_hook)
    _COMPOSE_HOOKED.add(id(model))


def _apply_patch(model, *, tau, start_percent, end_percent, min_tokens,
                 sink_conditioning, morton, morton_curve, dense_blocks,
                 verbose, tau_profile, topk_ratio=0.0, tail=True):
    # Before anything else: fail here if the installed kernel cannot take what
    # this node passes. Patch time is the only place that can be said -- see
    # `_require_kernel`.
    _require_kernel()
    diffusion_model = model.get_model_object("diffusion_model")
    is_h3 = hasattr(diffusion_model, "rope_freqs") and hasattr(diffusion_model, "_forward")

    # H3 publishes its segment layout from the same hooks Morton uses, so the
    # conditioning sink needs them installed even when reordering is off.
    reorder = False
    if is_h3 and (morton or sink_conditioning != "off"):
        install_h3_morton(diffusion_model)
        reorder = morton
    elif morton:
        logging.warning(
            f"[h3-sol] Morton skipped: {type(diffusion_model).__name__} is not "
            "MiniMax-H3. Sol-Attn itself still applies.")

    blocks = getattr(diffusion_model, "blocks", None)
    count = len(blocks) if blocks is not None else 0
    dense = parse_blocks(dense_blocks, count)
    profile = parse_tau_profile(tau_profile or "", count)
    if (dense or profile) and not _install_block_index(diffusion_model):
        logging.warning(
            f"[h3-sol] dense_blocks/tau_profile ignored: "
            f"{type(diffusion_model).__name__} has no .blocks list to index")
        dense, profile = frozenset(), {}
    if dense:
        logging.info(f"[h3-sol] keeping blocks {sorted(dense)} dense of {count}")

    model_sampling = model.get_model_object("model_sampling")
    sigma_start = float(model_sampling.percent_to_sigma(start_percent))
    sigma_end = float(model_sampling.percent_to_sigma(end_percent))

    m = model.clone()
    previous = m.model_options["transformer_options"].get("optimized_attention_override")
    if previous is not None:
        logging.info("[h3-sol] chaining onto an existing attention override")

    # Forward-level patches bypass optimized_attention entirely; gate each one.
    composed = []
    for key, patched in list(m.object_patches.items()):
        if not key.endswith(".forward"):
            continue
        owner = key.rsplit(".", 2)[-2].lower()
        if "attn" not in owner or "cross" in owner or owner == "attn2":
            continue  # Sol-Attn never takes cross-attention; leave it patched
        if getattr(patched, "_uses_optimized_attention", False):
            continue
        module = m.get_model_object(key[: -len(".forward")])
        m.add_object_patch(key, _compose_module_patch(module, patched))
        composed.append(key)
    if composed:
        logging.info(f"[h3-sol] composed with {len(composed)} patched attention forward(s)")
    if is_h3:
        _install_compose_hooks(diffusion_model, "attn")

    m.model_options["transformer_options"]["sol_compose"] = {
        "sigma_start": sigma_start, "sigma_end": sigma_end,
        "min_tokens": min_tokens}
    m.model_options["transformer_options"]["optimized_attention_override"] = \
        make_override(tau=tau, min_tokens=min_tokens,
                      sigma_start=sigma_start, sigma_end=sigma_end,
                      verbose=verbose, sink_conditioning=sink_conditioning,
                      dense_blocks=dense, tau_profile=profile, previous=previous,
                      topk_ratio=topk_ratio, tail=tail)
    if reorder:
        m.model_options["transformer_options"]["sol_morton"] = True
        m.model_options["transformer_options"]["sol_morton_curve"] = morton_curve
    reset_sol_attn_stats()
    return io.NodeOutput(m)


class MiniMaxH3SolAttn(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SolAttn",
            display_name="MiniMax H3 Sol-Attn",
            is_experimental=True,
            category="model/attention/minimax",
            description=(
                "Training-free block-sparse attention (Sol-Attn, arXiv "
                "2607.24027) for MiniMax-H3, on comfy_kitchen's merged CUDA "
                "kernel. bf16 and head_dim 128 only; ineligible calls fall "
                "back to whatever attention node sits upstream, which on "
                "these graphs is sage rather than dense torch. The win grows "
                "with sequence length, so leave min_tokens high.\n\n"
                "Forked from the vendored upstream node. Two of its widgets "
                "(centroid_tail, reuse_qkv_memory) are gone because "
                "comfy-kitchen#117 removed them from the kernel, and "
                "pooled_tail is new. See the module docstring for which "
                "kernel arguments this node deliberately does not expose."
            ),
            inputs=[
                io.Model.Input("model"),
                io.DynamicCombo.Input("selection", options=[
                    io.DynamicCombo.Option("adaptive tau", [
                        io.Float.Input("tau", default=1.0, min=0.0, max=4.0,
                                       step=0.05,
                                       tooltip="Threshold beta, the paper's own "
                                               "parameter. Higher is sparser: "
                                               "1.0 ~ 16% of blocks kept exact, "
                                               "1.5 ~ 7%, 2.0 ~ 2.7%. The paper "
                                               "never sweeps it."),
                        io.String.Input("tau_profile", optional=True,
                                        force_input=True,
                                        tooltip="Per-block tau overriding the base "
                                                "value. 'blocks=tau' entries "
                                                "separated by ';' or newlines."),
                    ]),
                    io.DynamicCombo.Option("top-k (SLA)", [
                        io.Float.Input("keep_percent", default=10.0, min=0.5,
                                       max=95.0, step=0.5,
                                       tooltip="Percent of key blocks each query "
                                               "block keeps exactly (sinks and the "
                                               "diagonal ride on top). With the "
                                               "lightx2v SLA turbo LoRA: 15 is the "
                                               "value it was distilled against, 10 "
                                               "is community-validated and faster. "
                                               "Without the LoRA, higher is closer "
                                               "to dense."),
                    ]),
                ], tooltip="How exact key blocks are chosen per query block. "
                           "'adaptive tau': threshold at tau sigmas of the score "
                           "distribution, so density varies per head and block. "
                           "'top-k (SLA)': a fixed keep_percent everywhere, the "
                           "selection the lightx2v SLA LoRAs were distilled "
                           "against. Pair top-k with pooled_tail OFF to reproduce "
                           "SLA exactly."),
                io.Float.Input("start_percent", default=0.2, min=0.0, max=1.0, step=0.01,
                               tooltip="Run dense before this point. The paper uses "
                                       "0.2. Never measured here at any value, and "
                                       "it costs a flat quarter of Sol's opportunity "
                                       "at every step count."),
                io.Float.Input("end_percent", default=0.9, min=0.0, max=1.0, step=0.01,
                               tooltip="Run dense after this point. This is a SIGMA "
                                       "band, not a step fraction, so which steps "
                                       "land inside it depends on the step count. "
                                       "The shipped graphs lower it per step count "
                                       "so the LAST step stays dense; a graph whose "
                                       "steps you edit by hand will not."),
                io.Int.Input("min_tokens", default=12288, min=0, max=1 << 20, step=512,
                             tooltip="Sequences shorter than this stay dense. "
                                     "H3's two token-refiner attention calls run on "
                                     "the text span alone (~311 rows), so any value "
                                     "above that keeps them off Sol; the 50 DiT "
                                     "calls are far above either. Drop it to 0 only "
                                     "if you deliberately want the refiner blocks "
                                     "routed too."),
                io.Combo.Input("sink_conditioning",
                               options=["exact_kv", "exact_kv_and_rows", "off"],
                               default="exact_kv_and_rows",
                               tooltip="exact_kv: every query sees the packed "
                                       "text/audio/reference rows exactly (~3% cost). "
                                       "exact_kv_and_rows: additionally runs the TARGET "
                                       "AUDIO query rows dense, which is what keeps "
                                       "generated audio intact; reference rows stay "
                                       "sparse, so the cost no longer scales with "
                                       "reference size."),
                io.Boolean.Input("pooled_tail", default=True,
                                 tooltip="The kernel's `tail`. ON, every unselected "
                                         "block still contributes one pooled term, so "
                                         "the whole sequence stays in the softmax "
                                         "denominator. That correction IS Sol-Attn's "
                                         "contribution, and the paper's ablation shows "
                                         "its advantage growing as sparsity rises.\n\n"
                                         "OFF, unselected blocks are dropped outright: "
                                         "softmax over the routed blocks only. Upstream "
                                         "calls this the SLA / VSA fine stage. Combined "
                                         "with 'top-k (SLA)' it reproduces the routing "
                                         "the Turbo-SLA LoRA was distilled under. "
                                         "Turning it off WITHOUT such a LoRA removes "
                                         "the method's own correction and is not a "
                                         "speed setting to reach for."),
                io.Boolean.Input("morton", default=False,
                                 tooltip="Reorder video tokens so each 64-token block is "
                                         "a compact 3D neighbourhood. Exactly neutral "
                                         "for dense attention."),
                io.Combo.Input("morton_curve", options=["3d", "2d_frame", "hilbert"],
                               default="3d",
                               tooltip="3d interleaves t/h/w equally and scores best on "
                                       "captured activations at the shipped canvas, "
                                       "though it is also the most canvas-variable of "
                                       "the three. 2d_frame orders within each frame and "
                                       "leaves frame order alone. hilbert is this repo's, "
                                       "2D within each frame, and has the best block "
                                       "geometry of the three; geometry does not rank "
                                       "orderings. See docs/morton.md."),
                io.Boolean.Input("verbose", default=False),
                io.String.Input("dense_blocks", default="",
                                tooltip="Transformer blocks kept off Sol, e.g. '0-2,32'. "
                                        "Negative indices count from the end. NOTE these "
                                        "run on the fallback backend, which on these "
                                        "graphs is sage, not exact attention -- use "
                                        "MiniMaxH3ExactBlocks for that."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, selection, start_percent, end_percent, min_tokens,
                sink_conditioning, pooled_tail, morton, morton_curve, verbose,
                dense_blocks) -> io.NodeOutput:
        topk = selection["selection"] == "top-k (SLA)"
        return _apply_patch(
            model, tau=selection.get("tau", 1.0),
            start_percent=start_percent, end_percent=end_percent,
            min_tokens=min_tokens, sink_conditioning=sink_conditioning,
            morton=morton, morton_curve=morton_curve, dense_blocks=dense_blocks,
            verbose=verbose, tau_profile=selection.get("tau_profile"),
            topk_ratio=selection["keep_percent"] / 100.0 if topk else 0.0,
            tail=pooled_tail)
