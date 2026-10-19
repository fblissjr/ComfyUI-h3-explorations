"""Sol-Attn for MiniMax-H3, on comfy-kitchen's CUDA kernel: `MiniMaxH3Sol`.

**Redesigned 2026-09-27.** `docs/research/2026-09-27_sol_node_redesign.md` is
the plan and the reasons, and `bench/results/2026-09-27_sol_node_compound_audit.md`
the evidence. It replaced `MiniMaxH3SolAttn` (a fork of the vendored upstream
node, 2026-08-30), which is deleted together with the code only it reached:
Morton token reordering, `tau_profile`, top-k (SLA) selection with
`pooled_tail`, the `sol_block` hooks and the `PackedLayout` patch. Git has
them; `vendor/sol_attn_minimax.py` stays as the pristine upstream reference.

## What the kernel takes, and what this node does with each

Read from `comfy_kitchen.sol_attn`'s signature (asserted once, at patch time,
by `_require_kernel`), not from a version string.

  tau               the node's `tau`.
  scale             taken from the caller's kwargs.
  sink_blocks,      H3's conditioning sink, derived from the layout core
  sink_q            publishes (`h3_layout`, `_sink_blocks`).
  tail              always on: the pooled term for unselected blocks IS the
                    method (the SLA lane that turned it off closed).
  topk_ratio        always 0: tau is the only selection.
  token_aug         the node's `token_routing`, per block.
  qk_balance,       the node's `quantizer` (both are fork options; the node
  rotate            refuses at patch time on a build without them).
  blk_cnt           passed only when `H3_SOL_OBSERVE` or `H3_SOL_SWEEP` is armed.
  key_bias          NOT exposed: legal only where the biased keys are
                    sink-covered, which on H3 means the conditioning rows,
                    and the model was never trained against such a bias.
  block_len,        NOT exposed here: they belong to VSA's cube tiling,
  coarse_gate       which FastH3 gets from core's `BlockSparseAttention`.

`sol_attn_chunked`, the second entry, is out of reach from an attention
override: it consumes chunks of the fused `qkv_proj` output and applies rope
and RMSNorm itself, and by the time an override is called both have run. Its
memory saving is `MiniMaxH3SolChunked`'s subject (`sol_chunked_h3.py`).

Requires comfy_kitchen with `sol_attn` (bf16 or fp16, head_dim 128, sm_80+).
Calls this node declines (dense_blocks, outside the sigma window, short calls,
masks) run on the attention override under it, which the node names in its
log and settings record.
"""

import logging
import re
from functools import partial

import torch

from comfy_api.latest import io

from .block_spec import parse_blocks
from . import h3_layout as _h3layout
from . import sol_observe
from . import h3_capture as _capture
from . import sol_block_probe as _probe
from . import sol_call_timer as _timer
from . import sol_tau_sweep as _sweep
from . import sparse_table as _table

try:
    import comfy_kitchen as _ck
    _CK_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency
    _ck = None
    _CK_IMPORT_ERROR = exc

HEAD_DIM = 128
BLOCK_SIZE = 64
# The `sink_conditioning` modes, in the order the node's combo lists them.
# `_sink_blocks` refuses anything else, so a patched setting that misspells
# one fails at the first attention call instead of silently running exact_kv.
SINK_CONDITIONING_MODES = ("exact_kv", "exact_kv_and_rows", "exact_kv_and_all_rows", "off")

_stats = {"sparse": 0, "dense_fallback": 0, "outside_range": 0,
          "dense_block": 0, "errors": 0}
_seen = set()


def _parse_block_profile(spec, count, name, cast, example):
    """Parse "0-30=X; 39-42=Y" into {block: value}, for one value kind.

    Entries are separated by ';' or newlines, so a multiline text node works as
    well as a single line, and '#' starts a comment. Blocks not listed keep the
    node's base value; the block side takes dense_blocks syntax, so "0-2,47=X"
    is valid. Later entries win where they overlap.

    `cast` both converts and validates, so a value the kernel would refuse is
    refused HERE. That placement is the whole point: every caller of this runs
    at patch time, where a refusal fails the node before sampling rather than
    stopping a render part-way.
    """
    profile = {}
    for entry in re.split(r"[;\n]", str(spec)):
        entry = entry.split("#", 1)[0].strip()
        if not entry:
            continue
        blocks, sep, value = entry.partition("=")
        if not sep:
            raise ValueError(f"{name} entry {entry!r} needs '=', e.g. {example!r}")
        level = cast(value.strip(), entry)
        for block in parse_blocks(blocks, count):
            profile[block] = level
    return profile


# The kernel's own admissible set, from `comfy_kitchen.sol_attn`'s docstring:
# zero, or a multiple of 64 up to 256. Refused here rather than at the call,
# for the reason `_parse_block_profile` gives.
TOKEN_AUG_BUDGETS = (0, 64, 128, 192, 256)


def parse_token_aug_profile(spec, count):
    """Parse "0,24,32=64" into {block: token_aug budget}.

    Per block and not global BECAUSE the grade says so: on the 2026-09-04
    capture, token routing lowered Sol's error against exact attention on four
    of five captured blocks at every step and RAISED it on block 49 at every
    step, so the one configuration measured to be wrong is the one a global
    switch would express. `docs/research/2026-09-04_sol_token_aug_grade.md`
    owns the numbers.

    The budget was measured inert across 64/128/256 in both accuracy and
    isolated kernel time, so 64 is the value to use if any: it buys what the
    larger ones buy for the smallest buffer. Turning it on at all is not free.
    """
    def _budget(value, entry):
        try:
            budget = int(value)
        except ValueError:
            raise ValueError(f"token_aug_blocks entry {entry!r} has a non-integer budget")
        if budget not in TOKEN_AUG_BUDGETS:
            raise ValueError(
                f"token_aug_blocks entry {entry!r}: budget {budget} is not one of "
                f"{list(TOKEN_AUG_BUDGETS)}; the kernel takes zero or a multiple "
                f"of 64 up to 256")
        return budget
    return _parse_block_profile(spec, count, "token_aug_blocks", _budget, "0,24,32=64")


#: The node's `dense_blocks` default (2026-09-25, the owner): the three blocks
#: whose K-norm is lopsided on the released checkpoint, where Sol's routed INT8
#: error is largest. On the block-49 capture, kitchen's dense INT8 kernel sits
#: well below Sol's routed error on the same heads
#: (`data/results/2026-09-15_ck_int8_attention_block49.json`), at a small cost
#: measured once (`bench/results/2026-09-15_block49_community_chain.md`).
#: Option A candidate: shields the top 3 worst middle blocks (>24% error) plus
#: the terminal block 49 (which feeds final_layer.video_out). Adopted 2026-10-02.
SOL_DENSE_OPTION_A = "39,41,42,49"

#: Option B candidate: shields the entire 23%–26% middle error plateau plus
#: the terminal block 49.
SOL_DENSE_OPTION_B = "39,40,41,42,49"

#: Option C / Full Ridge Shield: shields blocks 38 through 42 plus
#: the terminal block 49. Discovered in Test 9A to eliminate the Block 38 spike
#: under tau=1.3, reducing peak network error to 17.73% and ref_img error to 19.37%.
SOL_DENSE_OPTION_C = "38,39,40,41,42,49"
SOL_DENSE_RIDGE_SHIELD = SOL_DENSE_OPTION_C

#: Historical 2026-09-25 tail default (from unrotated INT8 K-norm outliers):
SOL_DENSE_HISTORICAL_TAIL = "45,48,49"

#: Active default for Sol dense_blocks: SOL_DENSE_OPTION_C ("38,39,40,41,42,49").
#: Adopted 2026-10-02 on local probe error, which the re-read showed was
#: mostly blocks leaving the measured set
#: (`bench/results/2026-10-02_sol_campaign_reanalysis.md`); KEPT the same
#: night by owner decision because the output-level panel found the
#: dense_blocks choice does not move the owner's verdict, and Option C is the
#: list `SOL_SINK_DEFAULT` was judged with
#: (`bench/results/2026-10-02_sol_dense_blocks_panel.md`). Measured, then
#: decided. `workflows/h3_config.py::SOL_DENSE_TAIL` carries the same value,
#: and `bench/check_attention_defaults.py` holds the two together.
SOL_DENSE_TAIL = SOL_DENSE_OPTION_C
#: The node's `sink_conditioning` default. `exact_kv_and_all_rows` since
#: 2026-10-02 (owner decision, measured): on the output-level panel it was the
#: one Sol setting the owner rated fine on the dialogue scene, where every
#: `exact_kv_and_rows` arm came out louder than dense; it lost on no scene
#: (`bench/results/2026-10-02_sol_dense_blocks_panel.md`). Its cost: it
#: also runs the reference rows exact, which on a video reference is most of
#: Sol's saving. `exact_kv_and_rows` before. Mirrored by
#: `workflows/h3_config.py::SOL_SINK_DEFAULT`.
SOL_SINK_DEFAULT = "exact_kv_and_all_rows"
#: Token routing's budget per query block when a preset turns it on;
#: `parse_token_aug_profile` says why 64.
TOKEN_ROUTING_BUDGET = 64
# The four captured blocks where the grade improved; block 49 is the fifth
# and the one it hurt (docs/research/2026-09-04_sol_token_aug_grade.md).
TOKEN_ROUTING_MEASURED_BLOCKS = (0, 24, 32, 40)
TOKEN_ROUTING_TAIL = 5


def sol_attn_stats():
    """Dispatch counters since process start (or last reset)."""
    return dict(_stats)


def reset_sol_attn_stats():
    for key in _stats:
        _stats[key] = 0
    _seen.clear()


def _log_once(key, message, level=logging.INFO):
    """Say something once per process. `level` because most of what this
    reports is diagnostic, but a few things are a silent change to what the
    model computes and INFO is where those go to be ignored."""
    if key not in _seen:
        _seen.add(key)
        logging.log(level, f"[h3-sol] {message}")


def _ineligible(q, k, mask, dim_head, min_tokens, dtypes=(torch.bfloat16,)):
    """Why this call can't use Sol-Attn, or None if it can. q/k are BTHD.

    `dtypes`: what the caller accepts. The direct kernel entry takes bf16 and
    fp16 (`comfy_kitchen.sol_attn`'s docstring); the old node offered bf16
    only, and says so in its messages."""
    if _ck is None or not hasattr(_ck, "sol_attn"):
        return "comfy_kitchen sol_attn unavailable"
    if q.device.type != "cuda":
        return "not cuda"
    if q.dtype not in dtypes:
        return f"dtype {q.dtype} (this node takes {', '.join(str(d) for d in dtypes)})"
    if dim_head != HEAD_DIM:
        return f"head_dim {dim_head} != 128"
    if mask is not None:
        # **Dead from the override path, deliberately kept.** `_run` is called
        # with `None` here because `override` has already returned dense on a
        # mask, with its own message. This branch is the second line of
        # defence for a DIRECT caller -- a bench script driving `_run` -- and
        # is the reason `_ineligible` reads as a complete eligibility test
        # rather than one with a hole in it.
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

    Called from `_apply_sol`, which propagates out of `execute` and fails the
    node before sampling rather than part-way through a render.
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
    # Only an ARMED observer needs the count out-parameter. Unarmed, the node
    # never passes it, so an older wheel keeps rendering; armed against such
    # a wheel it must fail here, not record nothing and look complete.
    if (sol_observe.enabled() or _sweep.enabled()) and "blk_cnt" not in params:
        raise RuntimeError(
            "H3_SOL_OBSERVE or H3_SOL_SWEEP is set, but the installed "
            "comfy_kitchen.sol_attn has no blk_cnt argument, so the route cannot "
            "be observed. Rebuild the kernel with vendor/rebuild_kernel.sh, or "
            "start the server without them.")


def _bthd(q, k, v, heads, skip_reshape):
    """(qs, ks, vs, b, dim_head): the BTHD views the kernel wants, from either
    layout `optimized_attention` hands over. Views only, no copy."""
    if skip_reshape:
        b, _, _, dim_head = q.shape          # BHND
        qs, ks, vs = (t.transpose(1, 2) for t in (q, k, v))
    else:
        b, _, dim_head = q.shape             # B, N, heads*dim_head
        dim_head //= heads
        qs, ks, vs = (t.view(b, -1, heads, dim_head) for t in (q, k, v))
    return qs, ks, vs, b, dim_head


def _run(q, k, v, heads, skip_reshape, skip_output_reshape, scale,
         tau, min_tokens, verbose, sink_blocks=(0, 0), sink_q=(0, 0),
         topk_ratio=0.0, tail=True, blk_cnt=None, token_aug=0, qk_balance=False, rotate=False,
         dtypes=(torch.bfloat16,), tau_map=None):
    """Returns the attention output, or None if this call should stay dense.

    `tau_map`, when given, is the kernel's per-head, per-query-block tau
    (`_tau_map`). Forwarded ONLY when not None, like `blk_cnt`: a call with no
    table and a row setting one `sink_q` range can express passes no such
    keyword, so it is the call every earlier measurement was taken on.

    `blk_cnt`, when given, is an int32 (B, H, ceil(T/64)) buffer the kernel
    fills with its routed-block counts from this same call. It is forwarded
    ONLY when not None: the unarmed path passes no such keyword, so a wheel
    without the argument keeps working and the default call is unchanged.
    """
    qs, ks, vs, b, dim_head = _bthd(q, k, v, heads, skip_reshape)

    reason = _ineligible(qs, ks, None, dim_head, min_tokens, dtypes)
    if reason is not None:
        _stats["dense_fallback"] += 1
        if verbose:
            _log_once((tuple(qs.shape), reason), f"dense {tuple(qs.shape)}: {reason}")
        return None

    # One call, no signature adaptation. `_require_kernel` has already
    # asserted at patch time that this build takes every one of these, so a
    # build that does not fails the node before sampling rather than here.
    #
    # `key_bias`, `block_len` and `coarse_gate` are left at their defaults --
    # None, None and None. The module docstring says why each is unreachable
    # or inert on this path rather than merely unused.
    #
    # `token_aug` is forwarded ONLY when non-zero, for the same reason
    # `blk_cnt` is forwarded only when not None: zero is the kernel's own
    # default and omitting it keeps the call byte-identical to what every
    # measurement in this repo was taken on.
    extra = {} if blk_cnt is None else {"blk_cnt": blk_cnt}
    if token_aug:
        extra["token_aug"] = int(token_aug)
    if qk_balance:                      # same rule: False is the kernel's default
        extra["qk_balance"] = True
    if rotate:
        extra["rotate"] = True
    if tau_map is not None:
        extra["tau_map"] = tau_map
    out = _ck.sol_attn(qs, ks, vs, tau=tau, scale=scale,
                       sink_blocks=list(sink_blocks), sink_q=list(sink_q),
                       topk_ratio=topk_ratio, tail=tail, **extra)      # BTHD
    _stats["sparse"] += 1
    if verbose:
        sel = (f"topk={topk_ratio:.3f}" if topk_ratio else f"tau={tau}")
        _log_once((tuple(qs.shape), "sparse", tail, int(token_aug), bool(qk_balance)),
                  f"sparse {tuple(qs.shape)} {sel} cuda-int8"
                  + (f" token_aug={int(token_aug)}" if token_aug else "")
                  + (" qk_balance" if qk_balance else "")
                  + (" rotate" if rotate else "")
                  + ("" if tail else " NO POOLED TAIL (SLA/VSA fine stage)"))

    if skip_output_reshape:
        return out.transpose(1, 2)           # BHND
    return out.reshape(b, -1, heads * dim_head)


def _sink_blocks(transformer_options, tokens, mode):
    """(exact-KV blocks, dense-query blocks) for MiniMax-H3's conditioning rows.

    H3 packs [text][cond][ref][audio][video] into one sequence; sparsifying the
    conditioning rows costs sync and prompt adherence. The exact-KV side covers
    every conditioning row in every mode but `off`. The dense-query side is
    ONE contiguous block range, because that is what the kernel's `sink_q`
    takes, so the modes differ only in where that range starts:

      exact_kv               no dense query rows
      exact_kv_and_rows      the TARGET AUDIO rows to the end of conditioning,
                             which on the packed layout is audio alone;
                             references before it stay sparse. The shipped
                             default (`h3_config.SOL_CUDA_DEFAULTS`)
      exact_kv_and_all_rows  every conditioning query row, references included.
                             "Text and audio dense, references sparse" is not
                             expressible in one range when references sit
                             between them, and this is the range that covers
                             both. On t2v there are no reference rows and the
                             two ranges differ by the text rows alone; on a
                             ref2v graph with a video reference the difference
                             is the reference's rows, which can be most of the
                             conditioning. Priced in docs/attention/sol_attn.md, the
                             recomputed sink-share table

    `exact_kv_and_rows` falls back to the all-rows range when the layout did
    not publish an audio span, which is a guard against an upstream layout
    change, not dead code (docs/attention/sol_attn.md, the sink section). Pure: no
    tensor, no kernel, graded on CPU by bench/check_sol_node_equivalence.py.
    """
    return sink_ranges(_h3layout.span(transformer_options, "video", tokens),
                       _h3layout.span(transformer_options, "audio", tokens), tokens, mode)


def sink_ranges(video, audio, tokens, mode):
    """The pure half of `_sink_blocks`: the two block ranges from the target
    video and target audio spans (each `(start, stop)` or None). Callers that
    hold spans rather than transformer_options (the capture graders) call this
    directly."""
    if mode not in SINK_CONDITIONING_MODES:
        raise ValueError(f"sink_conditioning {mode!r} is not one of {SINK_CONDITIONING_MODES}")
    if mode == "off" or video is None:
        return (0, 0), (0, 0)
    video_start, video_stop = video
    if tokens < video_stop or video_start <= 0:
        return (0, 0), (0, 0)
    blocks = (0, (video_start + BLOCK_SIZE - 1) // BLOCK_SIZE)
    if mode == "exact_kv":
        return blocks, (0, 0)
    if mode == "exact_kv_and_all_rows":
        return blocks, blocks
    # exact_kv_and_rows: dense-query protection exists for the TARGET AUDIO
    # rows; reference rows only need the exact-KV side. Fall back to the whole
    # conditioning range when the layout has no audio segment.
    if audio is None:
        return blocks, blocks
    audio_start, _audio_stop = audio
    return blocks, (audio_start // BLOCK_SIZE, blocks[1])


#: The `rows` input's two choices. "as sink_conditioning" leaves the dense
#: query rows to `sink_conditioning`, exactly as before the input existed.
ROWS_FOLLOW = "as sink_conditioning"
ROWS_PER_SEGMENT = "per segment"
ROW_EXACT, ROW_ROUTED = "exact", "routed"
#: The three classes of conditioning query rows a graph can set apart. Text is
#: core's `text` segment, audio is the TARGET audio, and reference is every
#: other segment ahead of the target video (cond, cond_audio, ref_img,
#: ref_audio), so the three together are always the whole conditioning prefix.
SEGMENT_CLASSES = ("text", "reference", "audio")
#: The tau that makes a (head, query block) dense: the kernel routes a key
#: block whose pooled score is at least tau spreads above the row mean, so a
#: large negative tau routes them all. **Inherited** from the fork's own test
#: (`tests/test_sol_attn.py::test_tau_map_very_negative_is_a_dense_query_block`),
#: and on a capture cell it reproduces `sink_q` over the same range bit for
#: bit (`bench/results/2026-10-03_kitchen_tau_map.md`).
DENSE_TAU = -1.0e30


def segment_class(kind):
    """The row class of one of core's segment kinds, or None for the target video."""
    if kind == "video":
        return None
    if kind in ("text", "audio"):
        return kind
    return "reference"


def exact_query_blocks(segments, tokens, exact):
    """Sorted indices of the 64-token query blocks to run exact, from core's
    segment table and a `{class: bool}` choice. A block that any exact segment
    overlaps is exact, which is `sink_ranges`' own rounding (floor of the
    start, ceiling of the stop). Same guards as `sink_ranges`: nothing without
    a target-video span, or on a call shorter than the layout."""
    video = next(((a, b) for a, b, kind in segments or () if kind == "video"), None)
    if video is None or tokens < video[1] or video[0] <= 0:
        return ()
    blocks = set()
    for a, b, kind in segments:
        if a >= video[0] or not exact.get(segment_class(kind), False):
            continue
        blocks.update(range(a // BLOCK_SIZE, (min(b, video[0]) + BLOCK_SIZE - 1) // BLOCK_SIZE))
    return tuple(sorted(blocks))


def row_plan(segments, tokens, mode, row_segments):
    """(exact-KV blocks, `sink_q` range, dense columns) for one call.

    `row_segments` None is `ROWS_FOLLOW`: `sink_ranges` decides, as before.
    Otherwise the exact query blocks come from the three segment classes; when
    they form one run it goes to the kernel as `sink_q`, and only a broken run
    (text and audio exact with references routed between them) needs the dense
    columns of a tau map. The exact-KV side is `sink_conditioning`'s either way.
    Pure: graded on CPU by `bench/check_sol_node_equivalence.py`."""
    video = next(((a, b) for a, b, kind in segments or () if kind == "video"), None)
    audio = next(((a, b) for a, b, kind in segments or () if kind == "audio"), None)
    sink, sink_q = sink_ranges(video, audio, tokens, mode)
    if row_segments is None:
        return sink, sink_q, ()
    blocks = exact_query_blocks(segments, tokens, row_segments)
    if not blocks:
        return sink, (0, 0), ()
    if blocks[-1] - blocks[0] + 1 == len(blocks):
        return sink, (blocks[0], blocks[-1] + 1), ()
    return sink, (0, 0), blocks


_MAP_CACHE = {}
#: Entries before the cache is dropped and rebuilt. **Reasoned**: one map per
#: (block, row setting, sequence length) per device, a few hundred KB each; a
#: two-stage graph over 50 blocks stays well under it.
_MAP_CACHE_MAX = 256


def _tau_map(head_taus, tau, heads, query_blocks, dense_cols, device):
    """The kernel's `tau_map`: float32 (heads, query blocks). Each row is the
    head's tau from the table (`head_taus`), or the node's `tau` with no table
    row for this block; the `dense_cols` query blocks get `DENSE_TAU`."""
    key = (head_taus, float(tau), int(heads), int(query_blocks), dense_cols, str(device))
    tmap = _MAP_CACHE.get(key)
    if tmap is None:
        if len(_MAP_CACHE) >= _MAP_CACHE_MAX:
            _MAP_CACHE.clear()
        if head_taus is None:
            tmap = torch.full((heads, query_blocks), float(tau), dtype=torch.float32, device=device)
        else:
            if len(head_taus) != heads:
                raise RuntimeError(f"[h3-sol] the tau table has {len(head_taus)} heads per block "
                                   f"and this call has {heads}")
            tmap = torch.tensor(head_taus, dtype=torch.float32, device=device) \
                .view(heads, 1).expand(heads, query_blocks).contiguous()
        if dense_cols:
            tmap[:, list(dense_cols)] = DENSE_TAU
        _MAP_CACHE[key] = tmap
    return tmap


#: The SLA lane closed on 2026-09-27, so the node runs Sol's own selection
#: (tau) with its pooled tail on, always. Kept as names because the recorders
#: (`sol_observe`, `sol_block_probe`) take them per call.
_TOPK_RATIO = 0.0
_TAIL = True
#: What the direct kernel entry takes (`comfy_kitchen.sol_attn`'s docstring).
_SOL_ACCEPTED_DTYPES = (torch.bfloat16, torch.float16)


def _tensor(t):
    """q, k or v as a tensor, whether it arrived bare or in one of core's
    single-owner `AttentionTensorContainer`s. A read: ownership stays put."""
    return t if torch.is_tensor(t) else t.peek()


def make_override(tau=1.0, min_tokens=12288,
                  sigma_start=None, sigma_end=None, verbose=False,
                  sink_conditioning="exact_kv", dense_blocks=frozenset(),
                  token_aug_profile=None, previous=None, qk_balance=False,
                  rotate=False, settings=None, tau_table=None, table_blocks=None,
                  row_segments=None):
    """Build an optimized_attention_override callable.

    ``tau_table`` (None, or ``{"name", "sha256"}`` of a table),
    ``table_blocks`` (its ``{block: taus per head}``) and ``row_segments``
    (None, or ``{class: exact?}`` over `SEGMENT_CLASSES`) are the per-head and
    per-segment policy (`sparse_table.py`, `row_plan`). With all three None
    the kernel call is the one this node made before they existed. The stamp
    records ``tau_table`` and ``row_segments``; the table's values are
    identified by the hash, not copied (`provenance.py::SOL_CLOSURE_KEYS`).

    The block index and segment bounds come from what core publishes
    (`h3_layout`); a kernel error raises rather than rendering dense
    (`docs/research/2026-09-27_sol_node_redesign.md`).

    ``previous`` chains any override already installed on the model: every path
    that declines hands off to it first, falling through to ``func`` only if
    there is none.

    **Core's container protocol (2026-10-03).** When ``previous`` carries a
    ``container_function`` (core's `set_model_optimized_attention` copies one
    from a backend that has it), the override returned here carries one too,
    and `wrap_attn` then hands q, k and v over in their single-owner containers
    instead of taking them first. A call this node declines passes the
    containers on untouched, so the fallback frees the bf16 q, k and v once it
    has quantized them; before, this frame and `wrap_attn`'s held them for the
    whole dense kernel. A call Sol takes takes the tensors here, as `wrap_attn`
    did. With no ``previous``, or one without a ``container_function`` (the
    sage override), nothing changes: there is no container entry to hand to.
    `data/results/2026-10-03_sol_container_protocol.json` has the memory and
    the bit-equality of the two kitchen entries this swaps between.

    ``settings`` is the node configuration as a plain dict, recorded by
    `sol_observe` once per distinct configuration and referenced from every
    call row. Unused unless the observer is armed.
    """
    settings = dict(settings or {})
    topk_ratio, tail, dtypes = _TOPK_RATIO, _TAIL, _SOL_ACCEPTED_DTYPES

    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        # **The capture seam** (`h3_capture.seam_begin`/`seam_end`, 2026-09-19):
        # this override receives every DiT attention call on BOTH chains, since
        # Sol chains onto whichever dense node came first, so it is the one place
        # a capture works on the default kitchen chain too. The host copy is
        # taken before the call and the file written after it, tagged with the
        # route that ran. Unarmed this is one attribute read.
        ticket = None
        if _capture.enabled:
            options = kwargs.get("transformer_options")
            shape = _tensor(q).shape
            ticket = _capture.seam_begin(
                _h3layout.block_index(options, shape[2] if skip_reshape else shape[1]),
                _tensor(q), _tensor(k), _tensor(v), heads, skip_reshape,
                transformer_options=options)
        out = _decide_and_run(func, q, k, v, heads, mask=mask, attn_precision=attn_precision,
                              skip_reshape=skip_reshape,
                              skip_output_reshape=skip_output_reshape, **kwargs)
        if ticket is not None:
            options = kwargs.get("transformer_options")
            _capture.seam_end(ticket, options.get("h3_attn_route", "unknown")
                              if isinstance(options, dict) else "unknown")
        return out

    def _decide_and_run(func, q, k, v, heads, mask=None, attn_precision=None,
                        skip_reshape=False, skip_output_reshape=False, **kwargs):

        # Read once per call, never cached across calls: the server reads the
        # environment at import, and a test may arm and disarm the module.
        observing = sol_observe.enabled()
        options = kwargs.get("transformer_options")
        # `held`: q, k and v are core's containers (the container entry at the
        # foot of `make_override`). This frame then keeps no tensor of its own
        # across a dense call, or the fallback's `del` would free nothing.
        held = not torch.is_tensor(q)
        box = type(q)
        shape, device = _tensor(q).shape, _tensor(q).device
        tokens = shape[2] if skip_reshape else shape[1]
        batch = shape[0]
        # The block label is read whenever a consumer exists. Before
        # 2026-09-01 only the depth gates read it, so an armed recorder on a
        # canonical graph (no dense_blocks, no profile) would have had no
        # block identity at all.
        # Core publishes the index for every block; `h3_layout` trusts it only
        # when this call spans the whole packed sequence (not the refiner).
        block = _h3layout.block_index(options, tokens)
        block_tau = tau
        # Absent from the profile means zero, which is the kernel's default and
        # the shipped state: token routing is opt-in per block, never global.
        block_aug = token_aug_profile.get(block, 0) if token_aug_profile else 0
        sink, sink_q, dense_cols = row_plan(_h3layout.segments(options, tokens), tokens,
                                            sink_conditioning, row_segments)
        head_taus = table_blocks.get(block) if table_blocks else None
        counts = None

        # **The route this call actually took, published for the capture.**
        # A caller cannot know it in advance: Sol's composition gate upstream
        # decides only on `min_tokens` and the sigma window, while
        # `dense_blocks`, eligibility and kernel failure are decided HERE. So a
        # capture tagged before this function runs claims an attribution it
        # cannot support -- which is what the first verification run produced,
        # tagging a block in `dense_blocks` as `sol` when it had run on sage.
        #
        # The same seam feeds the recorder, for the same reason. `record` may
        # raise `SolObserveError`; nothing here catches it, and the only
        # `except` below wraps the kernel call alone, so an observer failure
        # aborts the armed render rather than becoming a dense fallback.
        def route(name, reason=None):
            if isinstance(options, dict):
                options["h3_attn_route"] = name
            if observing:
                sol_observe.record(
                    route=name, reason=reason, counts=counts if name == "sol" else None,
                    options=options, settings=settings, block=block,
                    block_tau=block_tau, tokens=tokens, batch=batch,
                    heads=heads, sink=sink, sink_q=sink_q, tail=tail,
                    topk_ratio=topk_ratio, min_tokens=min_tokens)
            # The Sol-versus-fallback probe (`sol_block_probe.py`), armed by
            # H3_SOL_PROBE: a call that did not route through Sol is recorded
            # as skipped, with its reason, so its record can be checked for
            # completeness against the schedule. Unarmed this is one bool.
            if name != "sol" and _probe.enabled():
                _probe.skip(route=name, reason=reason, options=options, settings=settings,
                            block=block, block_tau=block_tau, tokens=tokens,
                            batch=batch, heads=heads, sink=sink, sink_q=sink_q,
                            tail=tail, topk_ratio=topk_ratio, min_tokens=min_tokens)

        def _timed(route_name):
            # H3_SOL_TIME only; see sol_call_timer.py. No sync is added.
            sig = (options or {}).get("sigmas")
            return _timer.span(route_name, block=block,
                               sigma=float(sig[0]) if sig is not None else None,
                               tokens=tokens, batch=batch, heads=heads)

        def dense():
            if held:
                target = previous.container_function
            else:
                target = func if previous is None else partial(previous, func)
            if _timer.enabled():
                with _timed("dense"):
                    return target(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                                  skip_reshape=skip_reshape,
                                  skip_output_reshape=skip_output_reshape, **kwargs)
            return target(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                          skip_reshape=skip_reshape,
                          skip_output_reshape=skip_output_reshape, **kwargs)

        if mask is not None:
            # **Loud, and not gated on `verbose`.** Declining here is a real
            # change to what the model computes -- the call runs on the
            # fallback backend instead of Sol -- and a completed render cannot
            # be told apart from one where Sol ran. Deliberate, which a reader
            # looking at the output cannot see; hence the warning.
            #
            # Once per process, at WARNING, because it is unreachable on every
            # shipped graph today: this is an ATTENTION mask, and core's H3
            # blocks pass none (`comfy/ldm/minimax/model.py`, `mask=None`). A
            # latent `noise_mask` does not arrive here. It becomes a per-token
            # timestep, so the frozen-audio and masked-source graphs run on
            # Sol. Raised by a peer session 2026-08-30; the last three
            # sentences corrected 2026-10-04, when it still said no node here
            # writes a `noise_mask`.
            _stats["dense_fallback"] += 1
            route("masked", "attention mask present")
            _log_once(("masked",),
                      "this call carries an attention mask, which the Sol "
                      "kernel cannot express, so it is running on the fallback "
                      "backend instead. The render will succeed and will not "
                      "be a Sol render. Every later masked call is silent.",
                      level=logging.WARNING)
            return dense()

        # The tau sweep (`sol_tau_sweep.py`), armed by H3_SOL_SWEEP. Ahead of
        # the depth and sigma gates on purpose: every block and step is
        # measured, and the model gets the fallback's output. The tensors are
        # read, not taken, so under the container entry the fallback still
        # consumes its containers and this frame keeps q, k and v alive for
        # the Sol calls after it. Unarmed this is one bool.
        if _sweep.enabled():
            tq, tk, tv = _tensor(q), _tensor(k), _tensor(v)
            qs, ks, vs, b, dim_head = _bthd(tq, tk, tv, heads, skip_reshape)
            if _ineligible(qs, ks, None, dim_head, min_tokens, dtypes) is None:
                scale = kwargs.get("scale", None)

                def sol_at(sweep_tau, sweep_counts):
                    # The node's own call with tau varied and no exact query
                    # rows; no table, so a table on the node is not swept.
                    return _run(tq, tk, tv, heads, skip_reshape, skip_output_reshape,
                                scale, sweep_tau, min_tokens, False, sink, (0, 0),
                                topk_ratio, tail, blk_cnt=sweep_counts, token_aug=block_aug,
                                qk_balance=qk_balance, rotate=rotate, dtypes=dtypes)

                def stock():
                    o = torch.nn.functional.scaled_dot_product_attention(
                        qs.transpose(1, 2), ks.transpose(1, 2), vs.transpose(1, 2), scale=scale)
                    return o if skip_output_reshape else o.transpose(1, 2).reshape(b, -1, heads * dim_head)

                route("sweep")
                return _sweep.run(dense_fn=dense, sol_fn=sol_at, stock_fn=stock, heads=heads,
                                  skip_output_reshape=skip_output_reshape, options=options,
                                  settings=settings, block=block, tokens=tokens, batch=batch,
                                  sink=sink, device=device)
            del tq, tk, tv, qs, ks, vs

        # Depth gate: a block in dense_blocks runs on the fallback.
        if block in dense_blocks:
            _stats["dense_block"] += 1
            route("dense_block", f"block {block} in dense_blocks")
            return dense()

        # Sampling-percentage gate, so the paper's dense warm-up steps work.
        if sigma_start is not None or sigma_end is not None:
            sigmas = kwargs.get("transformer_options", {}).get("sigmas")
            if sigmas is not None:
                sigma = float(sigmas[0])
                if (sigma_start is not None and sigma > sigma_start) or \
                   (sigma_end is not None and sigma < sigma_end):
                    _stats["outside_range"] += 1
                    route("outside_range",
                          f"sigma {sigma:.4g} outside [{sigma_end}, {sigma_start}]")
                    return dense()

        if verbose and sink != (0, 0):
            _log_once((tokens, sink, sink_q),
                      f"conditioning sink: KV blocks {sink} exact, dense query blocks {sink_q}")

        if observing:
            # Allocated OUTSIDE the try, so a kernel failure leaves it unfilled
            # and unrecorded rather than recorded as zeros.
            counts = torch.empty((batch, heads, (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE),
                                 dtype=torch.int32, device=device)
        tmap = None
        if head_taus is not None or dense_cols:
            tmap = _tau_map(head_taus, block_tau, heads,
                            (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE, dense_cols, device)
            if verbose:
                _log_once((tokens, tau_table["name"] if tau_table else None,
                           block if head_taus is not None else None, dense_cols),
                          f"tau map on block {block}: "
                          + (f"table {tau_table['name']}" if head_taus is not None else f"tau {block_tau}")
                          + (f", {len(dense_cols)} query block(s) dense by segment" if dense_cols else ""))
        if held:
            # Sol's kernel takes tensors, so ownership moves to this frame for
            # the call, which is where `wrap_attn` put it before.
            q, k, v = q.take(), k.take(), v.take()
        try:
            if _timer.enabled():
                with _timed("sol"):
                    out = _run(q, k, v, heads, skip_reshape, skip_output_reshape,
                               kwargs.get("scale", None), block_tau, min_tokens, verbose,
                               sink, sink_q, topk_ratio, tail, blk_cnt=counts,
                               token_aug=block_aug, qk_balance=qk_balance, rotate=rotate, dtypes=dtypes,
                               tau_map=tmap)
            else:
                out = _run(q, k, v, heads, skip_reshape, skip_output_reshape,
                           kwargs.get("scale", None), block_tau, min_tokens, verbose,
                           sink, sink_q, topk_ratio, tail, blk_cnt=counts,
                           token_aug=block_aug, qk_balance=qk_balance, rotate=rotate, dtypes=dtypes,
                           tau_map=tmap)
        except torch.OutOfMemoryError:
            # Unwrapped, so core's OOM handling (`execution.py`'s is_oom and
            # its hint) still recognises it.
            _stats["errors"] += 1
            route("kernel_error", "OutOfMemoryError")
            raise
        except Exception as exc:
            _stats["errors"] += 1
            route("kernel_error", f"{type(exc).__name__}: {exc}"[:200])
            # A failed kernel call run dense is a render that succeeds and is
            # not a Sol render; this node refuses that, as the repo does
            # everywhere else. (MiniMaxH3SolAttn ran it dense and logged.)
            raise RuntimeError(
                f"[h3-sol] the Sol kernel failed on block {block} ({type(exc).__name__}: "
                f"{exc}). Not falling back to dense attention: the render would "
                f"succeed and not be a Sol render. Remove the node to render "
                f"without Sol.") from exc
        if out is None:
            reason = None
            if observing:
                qs, ks, _vs, _b, dim_head = _bthd(q, k, v, heads, skip_reshape)
                reason = _ineligible(qs, ks, None, dim_head, min_tokens, dtypes)
                del qs, ks, _vs
            route("ineligible", reason)
            if held:
                q, k, v = box(q), box(k), box(v)
            return dense()
        route("sol")
        if _probe.enabled():
            # Runs the chained fallback (`dense`, the shipped sage override on
            # the canonical graphs) on the SAME q/k/v, records Sol against it,
            # and returns Sol's output under trajectory=sol or the fallback's
            # under trajectory=sage. Placed after `route("sol")` so the route
            # recorder's row and this one describe the same call. Under the
            # container entry the tensors go back in a container each, which
            # the fallback consumes on its one run.
            if held:
                q, k, v = box(q), box(k), box(v)
            return _probe.compare(out, dense, skip_output_reshape=skip_output_reshape,
                                  options=options, settings=settings, block=block,
                                  block_tau=block_tau, tokens=tokens, batch=batch,
                                  heads=heads, sink=sink, sink_q=sink_q, tail=tail,
                                  topk_ratio=topk_ratio, min_tokens=min_tokens, counts=counts)
        return out

    # Core's container protocol: `wrap_attn` calls this in place of taking the
    # tensors and calling `override`, and passes no `func`, which the held
    # path never reads (`dense` goes to `previous.container_function`). Set
    # only when there is such an entry to hand the containers to; `wrap_attn`
    # tests for the attribute, not its value.
    if getattr(previous, "container_function", None) is not None:
        override.container_function = partial(override, None)
    return override


def _record_composed(module, gate, options, tensor, reason):
    """A call Sol's composition gate declined ran on the composed foreign
    forward (Sage on the canonical graphs) and never reached the override --
    so the override's recorder never saw it. Until 2026-09-01 that left every
    outside-window DiT call of a canonical render absent from the record
    (Codex's review). Same builder as the override, `path="composed_patch"`,
    no counts: nothing here reroutes the call's numerics."""
    settings = gate.get("settings") if isinstance(gate, dict) else None
    if not settings:
        return
    if torch.is_tensor(tensor) and tensor.ndim in (2, 3):
        tokens = tensor.shape[0] if tensor.ndim == 2 else tensor.shape[1]
        batch = 1 if tensor.ndim == 2 else tensor.shape[0]
    else:
        tokens, batch = 0, 0
    block = _h3layout.block_index(options, tokens)
    block_tau = settings.get("tau", 0.0)
    sink, sink_q = _sink_blocks(options, tokens, settings.get("sink_conditioning", "off"))
    if sol_observe.enabled():
        sol_observe.record(
            route="composed_patch", reason=reason, counts=None, options=options, settings=settings,
            block=block, block_tau=block_tau, tokens=tokens, batch=batch,
            heads=getattr(module, "heads", None) or 0, sink=sink, sink_q=sink_q,
            tail=_TAIL, topk_ratio=_TOPK_RATIO,
            min_tokens=int(settings.get("min_tokens", 0)), path="composed_patch")
    if _probe.enabled():
        _probe.skip(
            route="composed_patch", reason=reason, options=options, settings=settings,
            block=block, block_tau=block_tau, tokens=tokens, batch=batch,
            heads=getattr(module, "heads", None) or 0, sink=sink, sink_q=sink_q,
            tail=_TAIL, topk_ratio=_TOPK_RATIO,
            min_tokens=int(settings.get("min_tokens", 0)))


def _compose_module_patch(module, patched_forward):
    """Gate an object-patched attention forward (e.g. KJNodes' mem-efficient
    Sage): calls Sol-Attn would take run the stock forward and reach the
    override; the rest keeps the patch. Gate params come from
    transformer_options["sol_compose"]; when absent the patch runs as-is.

    A declined call is recorded by `_record_composed` when the observer is
    armed, with the gate's own verdict as the route, because it will not
    reach the override's recorder.
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
        declined = None                      # the gate's verdict, once it says no
        take = gate is not None
        if take and not (torch.is_tensor(tensor) and tensor.device.type == "cuda"
                         and tensor.dtype in _SOL_ACCEPTED_DTYPES and tensor.ndim in (2, 3)):
            take = False
            declined = "ineligible: input is not a cuda bf16 2D/3D tensor"
        if take:
            # H3 packs tokens first (s, dim); Wan/LTX2 are batch-first.
            tokens = tensor.shape[0] if tensor.ndim == 2 else tensor.shape[1]
            if tokens < gate["min_tokens"]:
                take = False
                declined = f"ineligible: seq {tokens} < {gate['min_tokens']}"
        if take:
            sigmas = options.get("sigmas")
            if sigmas is not None:
                sigma = float(sigmas[0])
                if sigma > gate["sigma_start"] or sigma < gate["sigma_end"]:
                    take = False
                    declined = (f"outside_range: sigma {sigma:.4g} outside "
                                f"[{gate['sigma_end']}, {gate['sigma_start']}]")
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
        if declined is not None and (sol_observe.enabled() or _probe.enabled()):
            _record_composed(module, gate, options, tensor, declined)
        return patched_forward(*args, **kwargs)

    forward._sol_composed = True
    return forward




def _install_compose_hooks(model, attn_attr):
    """Compose at sampling time, once all object patches are applied: a node
    downstream of ours overwrites the same object-patch key, so execute-time
    composition alone loses. The pre-hooks re-wrap any foreign attn forward
    before each block runs; inert unless sol_compose is published.
    """
    # A marker on the model, not its id() in a module-level set: an id is
    # recycled once a model is freed, and a reloaded model on a recycled id
    # would be taken for hooked.
    if getattr(model, "_sol_h3_compose_hooked", False):
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
    model._sol_h3_compose_hooked = True


# ---------------------------------------------------------------------------
# MiniMaxH3Sol: the redesigned node (2026-09-27)
# ---------------------------------------------------------------------------
# docs/research/2026-09-27_sol_node_redesign.md is the plan and the reasons;
# bench/results/2026-09-27_sol_node_compound_audit.md is the evidence. What
# differs from MiniMaxH3SolAttn, which it replaced and which was deleted with
# the code only it reached:
#
#   - inputs: tau (the only selection left: the SLA lane closed), one
#     `quantizer` combo in place of the qk_balance and rotate booleans, and
#     token routing as a DynamicCombo whose `custom` option carries its list.
#     No Morton, no tau_profile, no top-k, no pooled_tail (the tail is on).
#   - block index and segment bounds from what core publishes (`h3_layout`);
#     this node patches nothing in core and installs no hooks of its own
#     (the composition hooks for an object-patched sage forward excepted).
#   - it re-installs its override on top at every step, as core's sparse node
#     does, so an attention node placed after it becomes its fallback rather
#     than silently replacing it.
#   - a kernel error raises instead of rendering dense.
#   - bf16 and fp16 are taken; the direct kernel entry accepts both.
#   - the dense fallback under it is named in the log and in the settings
#     record, so a render says which kernel ran its dense calls.

#: quantizer choice -> (qk_balance, rotate). Both are fork options of the
#: kitchen kernel, graded against each other on the shipped PDD8 graphs'
#: captures (`bench/results/2026-09-27_sol_redesign_test2.md`). The balance
#: gate opens on some heads of every block
#: (`data/results/2026-09-27_qk_balance_gate_on_capture.json`).
SOL_QUANTIZERS = {
    "plain": (False, False),
    "balanced": (True, False),
    "rotated": (False, True),
    "balanced+rotated": (True, True),
}
#: Measured (owner decision, 2026-09-27): "rotated" has the lowest Sol
#: quantization error of the four on all 28 PDD8 Sol-block cells and costs
#: less than "balanced" (`bench/results/2026-09-27_sol_redesign_test2.md`).
#: "balanced" was the default before, inherited from qk_balance=True.
SOL_QUANTIZER_DEFAULT = "rotated"

SOL_ROUTING_OFF = "off"
SOL_ROUTING_MEASURED = "measured blocks (0, 24, 32, 40)"
SOL_ROUTING_EARLY_MIDDLE = "early and middle (all but the last five)"
SOL_ROUTING_ALL = "all blocks (needs a balanced quantizer)"
SOL_ROUTING_CUSTOM = "custom"
SOL_ROUTING_CHOICES = (SOL_ROUTING_OFF, SOL_ROUTING_MEASURED, SOL_ROUTING_EARLY_MIDDLE,
                       SOL_ROUTING_ALL, SOL_ROUTING_CUSTOM)


def sol_routing_blocks(choice, blocks_spec, count, *, qk_balance):
    """{block: budget} for MiniMaxH3Sol's token-routing choice.

    `all blocks` needs the balance factor on: on the block-49 capture token
    routing raised the error plain and rotated, and lowered it only with the
    balance on (`data/results/2026-09-15_sol_token_aug_x_options_b49_s15.json`,
    `fixed_wheel` rows). Those balanced rows were measured on a kernel whose
    token stage scored an unbalanced centroid against balanced keys (fixed on
    h3-frontier, 2026-09-27); the re-grade on the fixed kernel is test 3 of the
    redesign and may move this rule."""
    if choice not in SOL_ROUTING_CHOICES:
        raise ValueError(f"token_routing {choice!r} is not one of {list(SOL_ROUTING_CHOICES)}")
    if choice == SOL_ROUTING_OFF:
        return {}
    if choice == SOL_ROUTING_CUSTOM:
        if not str(blocks_spec or "").strip():
            raise ValueError("token_routing is 'custom' and its blocks list is empty. Type the "
                             "blocks (e.g. '0,24,32,40=64'), or choose 'off'.")
        return parse_token_aug_profile(blocks_spec, count)
    if choice == SOL_ROUTING_MEASURED:
        missing = [b for b in TOKEN_ROUTING_MEASURED_BLOCKS if b >= count]
        if missing:
            raise ValueError(f"token_routing {choice!r} names MiniMax H3 blocks; this model "
                             f"has {count} and lacks {missing}")
        blocks = TOKEN_ROUTING_MEASURED_BLOCKS
    elif choice == SOL_ROUTING_EARLY_MIDDLE:
        blocks = range(max(count - TOKEN_ROUTING_TAIL, 0))
    else:
        if not qk_balance:
            raise ValueError(
                f"token_routing {choice!r} needs a quantizer with the balance on "
                f"('balanced' or 'balanced+rotated'): on the last block token routing "
                f"lowered the error only with it. Or use {SOL_ROUTING_EARLY_MIDDLE!r}.")
        blocks = range(count)
    return {int(b): TOKEN_ROUTING_BUDGET for b in blocks}


def _describe_override(override):
    """A name for the attention override a Sol node falls back to."""
    if override is None:
        return "stock attention (no override below Sol)"
    if getattr(override, "h3_kernel", None) == "sage":
        cells = dict(zip(getattr(getattr(override, "__code__", None), "co_freevars", ()) or (),
                         getattr(override, "__closure__", None) or ()))
        try:
            kwargs = cells["kernel_kwargs"].cell_contents
        except (KeyError, ValueError):
            kwargs = None
        return f"sage, MiniMaxH3SageAttention ({kwargs})" if kwargs else "sage, MiniMaxH3SageAttention"
    cells = dict(zip(getattr(getattr(override, "__code__", None), "co_freevars", ()) or (),
                     getattr(override, "__closure__", None) or ()))
    try:
        fn = cells["optimized_attention"].cell_contents
        return f"{getattr(fn, '__name__', fn)} (set_model_optimized_attention: core's ModelAttentionBackend)"
    except (KeyError, ValueError):
        pass
    return f"{getattr(override, '__module__', '?')}.{getattr(override, '__qualname__', repr(override))}"


def _chain_contains(override, ours, depth=16):
    """Whether one of `ours` sits anywhere under `override`, following the
    `previous` each override closes over (ours, core's sparse node and core's
    set_model_optimized_attention all name it that or hold none)."""
    for _ in range(depth):
        if override is None:
            return False
        if override in ours:
            return True
        cells = dict(zip(getattr(getattr(override, "__code__", None), "co_freevars", ()) or (),
                         getattr(override, "__closure__", None) or ()))
        try:
            override = cells["previous"].cell_contents
        except (KeyError, ValueError):
            override = getattr(override, "h3_previous", None)
    return False


def _apply_sol(model, *, tau, quantizer, dense_blocks, sink_conditioning,
               token_routing, routing_blocks, start_percent, end_percent,
               min_tokens, verbose, tau_table=_table.NONE, row_segments=None):
    if quantizer not in SOL_QUANTIZERS:
        raise ValueError(f"quantizer {quantizer!r} is not one of {list(SOL_QUANTIZERS)}")
    qk_balance, rotate = SOL_QUANTIZERS[quantizer]
    _require_kernel()
    import inspect
    try:
        params = inspect.signature(_ck.sol_attn).parameters
    except (TypeError, ValueError):     # not introspectable: the call decides, as in _require_kernel
        params = None
    for flag, name in ((qk_balance, "qk_balance"), (rotate, "rotate")):
        if flag and params is not None and name not in params:
            raise RuntimeError(
                f"quantizer {quantizer!r} needs {name}, and the installed comfy_kitchen.sol_attn "
                f"has no {name} argument. It is carried on the owner's fork (h3-frontier); "
                f"rebuild with vendor/rebuild_kernel.sh, or choose 'plain'.")

    diffusion_model = model.get_model_object("diffusion_model")
    base = getattr(model, "model", None)
    get_dtype = getattr(base, "get_dtype_inference", None)
    if get_dtype is not None and get_dtype() not in _SOL_ACCEPTED_DTYPES:
        raise RuntimeError(
            f"the model computes in {get_dtype()}, and Sol's kernel takes bf16 or fp16, so "
            f"every call would run dense. Load the DiT in bf16 or fp16, or remove this node.")

    blocks = getattr(diffusion_model, "blocks", None)
    count = len(blocks) if blocks is not None else 0
    dense = parse_blocks(dense_blocks, count)
    aug = sol_routing_blocks(token_routing, routing_blocks, count, qk_balance=qk_balance)
    if aug and params is not None and "token_aug" not in params:
        raise RuntimeError("token routing is on, and the installed comfy_kitchen.sol_attn has no "
                           "token_aug argument (Comfy-Org/comfy-kitchen #156, 0.2.33).")

    # The per-head table and the per-segment rows both reach the kernel as a
    # tau map. Refused here, before sampling, on a build without it.
    table = None
    if tau_table != _table.NONE:
        table = _table.load(tau_table)
        heads = getattr(getattr(blocks[0], "attn", None), "heads", None) if count else None
        if heads is None:
            raise RuntimeError("a tau table needs the model's head count, and this model's "
                               "blocks do not publish one (blocks[0].attn.heads)")
        _table.require_fits(table, int(heads), count)
    if row_segments is not None:
        unknown = sorted(set(row_segments) - set(SEGMENT_CLASSES))
        if unknown or set(row_segments) != set(SEGMENT_CLASSES):
            raise ValueError(f"row_segments must set exactly {list(SEGMENT_CLASSES)}, got "
                             f"{sorted(row_segments)}")
    if (table is not None or row_segments is not None) and params is not None \
            and "tau_map" not in params:
        raise RuntimeError(
            "a tau table or per-segment rows need tau_map, and the installed "
            "comfy_kitchen.sol_attn has no tau_map argument. It is carried on the owner's "
            "fork (h3-frontier); rebuild with vendor/rebuild_kernel.sh, or choose "
            f"tau_table '{_table.NONE}' and rows '{ROWS_FOLLOW}'.")

    model_sampling = model.get_model_object("model_sampling")
    sigma_start = float(model_sampling.percent_to_sigma(start_percent))
    sigma_end = float(model_sampling.percent_to_sigma(end_percent))

    m = model.clone()
    settings = {
        "node": "MiniMaxH3Sol", "tau": float(tau), "quantizer": quantizer,
        "qk_balance": bool(qk_balance), "rotate": bool(rotate), "tail": True, "topk_ratio": 0.0,
        "min_tokens": int(min_tokens), "sink_conditioning": sink_conditioning,
        "start_percent": float(start_percent), "end_percent": float(end_percent),
        "sigma_start": sigma_start, "sigma_end": sigma_end,
        "dense_blocks": sorted(int(b) for b in dense),
        "token_routing": token_routing,
        "token_aug_blocks": {str(k): int(v) for k, v in sorted(aug.items())},
        "n_blocks": count,
        "tau_table": tau_table,
        "tau_table_sha256": None if table is None else table["sha256"],
        "tau_table_provenance": None if table is None else table["provenance"],
        "rows": ROWS_FOLLOW if row_segments is None else
                {c: (ROW_EXACT if row_segments[c] else ROW_ROUTED) for c in SEGMENT_CLASSES},
    }
    installed = set()

    def install(transformer_options):
        """Put this node's override on top of whatever is on the hook, and name
        what it falls back to. Idempotent once it is on top; run at patch time
        and again each step (ON_PREPARE_STATE)."""
        current = transformer_options.get("optimized_attention_override")
        if current in installed or _chain_contains(current, installed):
            # On top, or already under another override that re-installs
            # itself each step (core's sparse node does): wrapping it again
            # would grow the chain by two every step.
            return
        fallback = _describe_override(current)
        override = make_override(
            tau=tau, min_tokens=min_tokens, sigma_start=sigma_start, sigma_end=sigma_end,
            verbose=verbose, sink_conditioning=sink_conditioning, dense_blocks=dense,
            token_aug_profile=aug, previous=current, qk_balance=qk_balance, rotate=rotate,
            settings=dict(settings, dense_fallback=fallback),
            tau_table=None if table is None else {"name": table["name"], "sha256": table["sha256"]},
            table_blocks=None if table is None else table["blocks"],
            row_segments=row_segments)
        installed.add(override)
        transformer_options["optimized_attention_override"] = override
        transformer_options["sol_compose"] = {
            "sigma_start": sigma_start, "sigma_end": sigma_end, "min_tokens": min_tokens,
            "settings": dict(settings, dense_fallback=fallback)}
        _log_once(("sol_fallback", fallback), f"MiniMaxH3Sol: dense calls run on {fallback}")

    options = m.model_options["transformer_options"]
    install(options)
    import comfy.patcher_extension
    m.add_callback_with_key(
        comfy.patcher_extension.CallbacksMP.ON_PREPARE_STATE, "h3_sol",
        lambda model_patcher, timestep, model_options: install(model_options["transformer_options"]))

    # An object-patched attention forward (the sage node) bypasses
    # optimized_attention; gate it so the calls Sol takes reach the override.
    composed = []
    for key, patched in list(m.object_patches.items()):
        if not key.endswith(".forward"):
            continue
        owner = key.rsplit(".", 2)[-2].lower()
        if "attn" not in owner or "cross" in owner or owner == "attn2":
            continue
        if getattr(patched, "_uses_optimized_attention", False):
            continue
        module = m.get_model_object(key[: -len(".forward")])
        m.add_object_patch(key, _compose_module_patch(module, patched))
        composed.append(key)
    if hasattr(diffusion_model, "blocks"):
        _install_compose_hooks(diffusion_model, "attn")
    if _sweep.enabled():
        # The sweep is its own record on the fallback's trajectory. A route
        # or probe record written beside it would describe calls that never
        # ran as the node's, and a composed forward keeps the calls outside
        # the sigma window away from the override, so the sweep would miss
        # them without saying so.
        if sol_observe.enabled() or _probe.enabled():
            raise RuntimeError("H3_SOL_SWEEP is armed together with H3_SOL_OBSERVE or H3_SOL_PROBE. "
                               "Arm the sweep alone: it returns the fallback's output on every call.")
        if composed:
            raise RuntimeError("H3_SOL_SWEEP is armed and this model has an object-patched attention "
                               f"forward ({composed[0]}), which keeps some calls from the sparse "
                               "node. Sweep a graph whose dense node is ModelAttentionBackend.")
        logging.warning(f"[h3-sol] tau sweep ARMED ({_sweep.spec()['spec']}): this render gets the "
                        f"fallback's attention on every call and is NOT a Sol render")

    logging.info(
        f"[h3-sol] MiniMaxH3Sol on: sigma window [{sigma_end:.4g}, {sigma_start:.4g}] "
        f"(start_percent {start_percent}, end_percent {end_percent}), tau {tau}, "
        f"quantizer {quantizer}, token routing on {len(aug)} block(s), "
        f"dense blocks {sorted(dense)}, sink {sink_conditioning}"
        + ("" if table is None else f", tau table {table['name']} ({len(table['blocks'])} block(s))")
        + ("" if row_segments is None else f", rows {settings['rows']}")
        + (f", composed with {len(composed)} patched forward(s)" if composed else ""))
    if sol_observe.enabled():
        logging.info(f"[h3-sol] route observation ARMED ({sol_observe.spec()['spec']}); "
                     f"timings from this render are not quotable")
    reset_sol_attn_stats()
    return io.NodeOutput(m)


class MiniMaxH3Sol(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Sol",
            display_name="MiniMax H3 Sparse Attention",
            search_aliases=["sol", "sol-attn", "sparse attention", "block sparse"],
            is_experimental=True,
            category="model/attention/minimax",
            description=(
                "Training-free block-sparse attention (Sol-Attn, arXiv 2607.24027) "
                "for MiniMax-H3, on comfy_kitchen's CUDA kernel. Put it after the "
                "attention node whose kernel should run the dense calls (the "
                "blocks in dense_blocks, the steps outside the start/end window, "
                "short calls); it names that kernel in the log. It keeps itself on "
                "top of the attention hook at every step, so a node placed after it "
                "becomes its fallback rather than replacing it."),
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("tau", default=1.0, min=0.0, max=4.0, step=0.05,
                               tooltip="How sparse. A key block is computed exactly when "
                                       "its pooled score sits at least tau standard "
                                       "deviations above the row mean; the rest become one "
                                       "pooled term each. Higher is sparser and faster. "
                                       "1.0 is what sglang and NVLabs' H3 profiles use."),
                io.Combo.Input("quantizer", options=list(SOL_QUANTIZERS),
                               default=SOL_QUANTIZER_DEFAULT,
                               tooltip="How Sol's INT8 quantizers treat q and k. 'balanced' "
                                       "rescales q up and k down per channel on heads whose "
                                       "K energy sits in a few loud channels; 'rotated' "
                                       "multiplies every row by a fixed Hadamard matrix (the "
                                       "one kitchen's dense int8 attention uses). Both leave "
                                       "every attention score unchanged in exact arithmetic; "
                                       "they change INT8 rounding. Needs the owner's kitchen "
                                       "build for anything but 'plain'."),
                io.String.Input("dense_blocks", default=SOL_DENSE_TAIL,
                                tooltip="Blocks kept off Sol, e.g. '0-2,32'; negative indices "
                                        "count from the end. They run on the dense fallback. "
                                        f"Default '{SOL_DENSE_TAIL}' (Option C / Full Ridge Shield): shields the entire middle "
                                        "routing-error plateau (38, 39, 40, 41, 42) and terminal block (49). "
                                        f"Option B is '{SOL_DENSE_OPTION_B}'; Option A is '{SOL_DENSE_OPTION_A}'."),
                io.Combo.Input("sink_conditioning", options=list(SINK_CONDITIONING_MODES),
                               default=SOL_SINK_DEFAULT,
                               tooltip="How the packed conditioning rows (text, references, "
                                       "target audio) are protected. exact_kv: every query "
                                       "attends them exactly. exact_kv_and_rows: also runs the "
                                       "target-audio query rows dense. exact_kv_and_all_rows "
                                       "(default): every conditioning query row dense, references "
                                       "included; on a video reference this costs most of Sol's "
                                       "saving. off: none."),
                io.DynamicCombo.Input("token_routing", options=[
                    io.DynamicCombo.Option(SOL_ROUTING_OFF, []),
                    io.DynamicCombo.Option(SOL_ROUTING_MEASURED, []),
                    io.DynamicCombo.Option(SOL_ROUTING_EARLY_MIDDLE, []),
                    io.DynamicCombo.Option(SOL_ROUTING_ALL, []),
                    io.DynamicCombo.Option(SOL_ROUTING_CUSTOM, [
                        io.String.Input("blocks", default="0,24,32,40=64",
                                        tooltip="'layers=budget', e.g. '0,24,32=64'. "
                                                f"Budget one of {list(TOKEN_AUG_BUDGETS)}."),
                    ]),
                ], tooltip=f"Per query block, up to {TOKEN_ROUTING_BUDGET} of the best tokens "
                           "outside the routed blocks are attended exactly, and the rest "
                           "of those blocks is scored token by token instead of pooled. "
                           "Costs time. On captures it lowered the error on four blocks and "
                           "raised it on the last one unless the balance was on."),
                io.Float.Input("start_percent", default=0.2, min=0.0, max=1.0, step=0.01,
                               tooltip="Dense before this point of the schedule."),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Dense after this point. A sigma band, not a step "
                                       "fraction."),
                io.Int.Input("min_tokens", default=12288, min=0, max=1 << 20, step=512,
                             advanced=True,
                             tooltip="Shorter calls stay dense: on H3 that is the two "
                                     "text-only token-refiner calls."),
                io.Boolean.Input("verbose", default=True, advanced=True,
                                 tooltip="Log once per call shape whether it ran on Sol or "
                                         "dense, and why."),
                io.Combo.Input("tau_table", options=[_table.NONE] + _table.list_tables(),
                               default=_table.NONE, optional=True,
                               tooltip="A calibrated table of tau per block and head, in place "
                                       "of the one tau above on the blocks it lists (a block it "
                                       "does not list keeps tau). Files in sparse_tables/. "
                                       "Needs the owner's kitchen build."),
                io.DynamicCombo.Input("rows", options=[
                    io.DynamicCombo.Option(ROWS_FOLLOW, []),
                    io.DynamicCombo.Option(ROWS_PER_SEGMENT, [
                        io.Combo.Input("text_rows", options=[ROW_EXACT, ROW_ROUTED],
                                       default=ROW_EXACT,
                                       tooltip="The prompt's own rows, vision tokens included."),
                        io.Combo.Input("reference_rows", options=[ROW_EXACT, ROW_ROUTED],
                                       default=ROW_EXACT,
                                       tooltip="Keyframe and reference rows (image, video, audio)."),
                        io.Combo.Input("audio_rows", options=[ROW_EXACT, ROW_ROUTED],
                                       default=ROW_EXACT,
                                       tooltip="The target audio rows."),
                    ]),
                ], optional=True,
                    tooltip="Which conditioning query rows run exact. 'as sink_conditioning' "
                            "leaves it to that input, as before. 'per segment' sets text, "
                            "reference and audio rows apart; all three exact is "
                            "exact_kv_and_all_rows, audio alone is exact_kv_and_rows. Which "
                            "keys every row attends exactly stays with sink_conditioning. "
                            "Text and audio exact with references routed needs the owner's "
                            "kitchen build."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def fingerprint_inputs(cls, tau_table=_table.NONE, **kwargs):
        # A table rewritten under the same name changes no input, and the node
        # cache would hand back the model patched with the old values (found
        # in review, 2026-10-03). The file's bytes are the input.
        return _table.file_fingerprint(tau_table)

    @classmethod
    def execute(cls, model, tau, quantizer, dense_blocks, sink_conditioning, token_routing,
                start_percent, end_percent, min_tokens=12288, verbose=True,
                tau_table=_table.NONE, rows=None) -> io.NodeOutput:
        row_segments = None
        if rows is not None and rows["rows"] == ROWS_PER_SEGMENT:
            row_segments = {c: rows[f"{c}_rows"] == ROW_EXACT for c in SEGMENT_CLASSES}
        return _apply_sol(
            model, tau=tau, quantizer=quantizer, dense_blocks=dense_blocks,
            sink_conditioning=sink_conditioning,
            token_routing=token_routing["token_routing"],
            routing_blocks=token_routing.get("blocks", ""),
            start_percent=start_percent, end_percent=end_percent,
            min_tokens=min_tokens, verbose=verbose,
            tau_table=tau_table, row_segments=row_segments)
