"""SCAFFOLDING -- NOT IMPLEMENTED. Per-block Sol-Attn error probe, CUDA path.

**Nothing here runs yet.** Every function body is a stub that raises. This file
exists so the plan's Track A2 has a shape to argue with before any of it is
written, and so the node id and input order are decided once, deliberately,
rather than during implementation. See
`internal/plan_2026-08-16_sol_fp16_and_triton_retirement.md`, Track A2.

## Why this exists

The Triton pack carried `SolAttnBlockProbe`, which computed every attention
call both sparse and dense and logged per-block relative error worst-first. It
is the instrument for choosing a `dense_blocks` list, and it was the last live
reason that pack existed here.

**The pack was deleted on 2026-08-16 before this was written**, so the port
target is upstream at a pinned commit rather than a local path:
`https://github.com/kijai/ComfyUI-SolAttn_triton` at `842c4ea`. Fetch it into a
scratch dir to read; do not reinstall it into `custom_nodes/`.

`docs/SOLATTN.md` names the thing it is for: at `tau` above roughly 1.5 a small
persistent object can dissolve partway through a clip. The stated fix is to
force the most approximation-sensitive blocks dense, and `SOL_ARTIFACT_INSURANCE`
in `workflows/h3_config.py:280` is a guessed starting set that is deliberately
not wired, pending a probe run nobody has done.

## What is being ported, and what is not

**Ported unchanged**, from `kijai/ComfyUI-SolAttn_triton@842c4ea:__init__.py:323-342`:
the wrapper calls the installed override for the sparse result, calls `func`
for the dense reference, records the pair, and **returns the dense result**.
That last detail is load-bearing and is easy to lose: returning the sparse
result would let each block's error compound into every later block's number.

**Not ported:** anything Triton. `make_probe_override` contains none -- it wraps
whatever override is installed, so it works against the CUDA node in principle.

## Three additions, each from something measured since the original was written

1. **Routed density per block.** Nothing in this repo measures it.
   `sol_attn_stats()` (`vendor/sol_attn_minimax.py:102-104`) counts dispatches,
   not blocks. Density is what decides whether Sol's exact branch is a thin
   slice or most of the work, which is the open question behind the 16-bit PV
   decision (plan Track B, gate B0a).
2. **Report the tail, not only the mean.** `docs/morton.md` found the p10
   mattered where the mean did not: `2d_frame`'s p10 centroid fidelity sits
   below raster at blocks 0 and 49 while its mean sits above.
3. **Per-segment error.** H3 packs text, audio, references and video into one
   sequence. The audio rows are thin -- `docs/SOLATTN.md` calls them the shape a
   block-sparse router drops first -- and a whole-tensor error number cannot see
   them. Split by the `PackedLayout` spans the compose hooks already publish.

## THE RISK THAT MUST BE RUN, NOT REASONED

**Unverified: whether a probe wrapping `optimized_attention_override` sees the
CUDA node's DiT calls at all.**

Our sage node object-patches `diffusion_model.blocks.{i}.attn.forward`, which
deletes the `optimized_attention` call site for all 50 DiT blocks. Sol's
`_compose_module_patch` gates that patch and calls `stock()` inside the sigma
window, which should reach the override. **Should.** That is an inference from
source, and `docs/SOLATTN.md`'s Ordering section documents two separate nodes
that look like they compose and do not. The same shape has cost this repo
several confident wrong claims.

So the first thing this node needs is not a feature, it is a control: **assert
the number of distinct blocks seen equals the number the compose hook reports
patching.** An empty or short list must be a loud failure, never a quiet
"no errors found" -- a probe that measures nothing looks exactly like a kernel
with no error, and it is most convincing when it is emptiest.

## Node id and input order are permanent

CLAUDE.md's one rule that matters. `node_id` is baked into every saved graph's
`type` field and inputs are matched positionally. **Append only.** No shipped
graph should ever wire this -- it is a diagnostic that roughly doubles render
cost, since every call is computed twice.

## Specification as of 2026-09-03 (Codex's, adopted by the owner)

Two modes, explicit record fields `trajectory` and `returned_backend`:
`trajectory=sol` computes Sol and Sage on identical q/k/v and RETURNS SOL
(the production population; upstream Sol error in later inputs is
intentional); `trajectory=sage` computes both and RETURNS SAGE (the
fallback trajectory, which is what a dense block runs; never call it
"dense exact"). The two renders' cells are never aggregated. The retained
2026-09-03 Base16 capture is the validation fixture: the instrument's
summaries must reproduce its cells within a stated tolerance before a new
render is trusted.

Per measured cell, record: identity (capture id, prompt id, render index,
schedule occurrence and index, sigma, schedule length, block, actual
route, compare status and reason); shape and layout (B/H/T/D and the
authoritative `PackedLayout.segments`, published by a neutral helper that
Sol or `h3_capture.py` can arm -- not derived from geometry);
configuration (tau or top-k, tail, resolved sink and sink-query ranges,
ordering, window, min_tokens, dense_blocks, requested Sage mode, the Sage
kernel actually dispatched, both kernel build ids); cost (blk_cnt, kernel
density, pair-weighted density, with their different denominators kept);
Sol-versus-the-actual-shipped-Sage error on identical q/k/v (whole-call
relative L2 and cosine; absolute-difference RMS and reference RMS; per-head
numerator, denominator, relative L2 and cosine; per-segment aggregates;
per-head and per-segment row-distribution summaries: count, mean, p50,
p90, p99, max). A zero denominator yields null with numerator and
denominator retained.

Behaviour: compare only calls that routed through Sol, recording skip
reasons elsewhere; confirm the counterfactual is the configured Sage auto
kernel on this box, not generic dense attention; write no q/k/v; join to
`sol_observe` by capture id, prompt, schedule occurrence and block; treat
the render's timing as void; unarmed, add no allocation, kernel call, copy
or synchronisation beyond the branch. Memory: do not clone production
q/k/v blindly; if the Sage counterfactual is head-chunked, first prove on
the fixture that it matches the 56-head call within a stated tolerance.

Controls before trusting a record: armed and unarmed outputs bitwise equal
to canonical Sol; unarmed produces no records and no extra Sage call;
Sage dispatch telemetry observed, a stock-attention substitution fails;
identical and deliberately perturbed fixtures validate every metric;
metrics reproduce the retained capture's cells; completeness -- every
expected active schedule occurrence x blocks 0-49 exactly once; segments
contiguous over [0, T), wrong or shuffled boundaries fail; duplicate or
missing blocks, mixed prompt ids, NaN/Inf and wrong schedule populations
fail; PDD completeness derived from the PDD node's actual SIGMAS, never a
nominal step count.

The scaffold's original note -- return the DENSE result so a block's error
cannot compound into later blocks -- is the `trajectory=sage` mode above,
kept as a named mode rather than the only one.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Set when the implementation lands. Until then every entry point raises, so a
# graph that wires this node fails loudly rather than rendering a silent no-op.
_IMPLEMENTED = False

_NOT_IMPLEMENTED = (
    "sol_block_probe is scaffolding and has no implementation yet. See "
    "internal/plan_2026-08-16_sol_fp16_and_triton_retirement.md, Track A2."
)


def make_probe_override(inner):
    """Wrap an installed override so every call is computed both ways.

    Port target: `kijai/ComfyUI-SolAttn_triton@842c4ea:__init__.py:323-342`.

    Contract, and the parts that must not drift:
      - call ``inner(func, q, k, v, heads, **common)`` for the sparse result
      - call ``func(q, k, v, heads, **common)`` for the dense reference
      - record the pair against the block index in ``transformer_options``
      - **return the dense result**, so error cannot compound across blocks
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


def _record(block, sparse, reference, segments=None):
    """Accumulate one call's error.

    TODO(scaffolding): decide the metric before writing this. The original logs
    a relative error. `docs/SOLATTN.md` warns that a cosine and an rtol from
    different harnesses cannot be compared and that they fail in opposite
    directions, so whatever is chosen has to be stated in the output itself,
    not left to the reader.

    ``segments`` carries the PackedLayout spans so text / audio / reference /
    video can be reported separately (addition 3 in the module docstring).
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


def _record_density(block, routed_blocks, total_blocks):
    """Accumulate routed density for one call (addition 1).

    TODO(scaffolding): the kernel knows this and does not currently report it.
    Two candidate sources, and neither is verified:
      - `blk_cnt`, which `sol_attn_route.cu` already writes per (b, h, query
        block). Reaching it means the kernel exposing it, i.e. an upstream
        change on the fork.
      - recomputing the routing decision host-side from the same threshold,
        which is a second implementation and therefore also a cross-check.

    The second is more work and worth more: an independent implementation is
    the thing this repo keeps finding it needed.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


def summarize():
    """Emit the worst-first per-block table at sampling end.

    Must include, and must refuse to print without:
      - blocks seen against blocks expected (the control -- see the docstring)
      - the metric's name and how it was computed
      - mean AND p10/p90 per block (addition 2)
      - routed density per block (addition 1)
      - per-segment breakdown (addition 3)
      - tau, and the sigma window, since a profile taken at a gentler setting
        is measured where the failure does not occur
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


def install(model):
    """Clone the model, wrap its override, register the summary callback.

    Port target: `SolAttnBlockProbe.execute`,
    `kijai/ComfyUI-SolAttn_triton@842c4ea:__init__.py:627-646`. `_install_block_index`
    already exists at `vendor/sol_attn_minimax.py:80-100` and should be reused
    rather than reimplemented.

    Must raise if no override is installed. A probe with nothing to measure is
    the empty-check failure mode, and it renders successfully.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


# TODO(scaffolding): the ComfyNode subclass goes here once the above works.
#
#   node_id      "MiniMaxH3SolBlockProbe"   <- permanent once a graph saves it
#   category     "model/debug/minimax"       (matches MiniMaxH3ProvenanceStamp)
#   inputs       io.Model.Input("model")     <- APPEND ONLY after this ships
#   outputs      io.Model.Output()
#
# Register in `nodes.py`'s `H3ExplorationsExtension.get_node_list()` by
# APPENDING to the list, never inserting.
#
# Decide before shipping: node or bench script? A probe must run inside
# sampling to see the calls, which forces a node. But a node is a permanent
# schema commitment for a diagnostic nobody should wire. One option worth
# weighing is an env-gated install like `h3_capture.py`'s `H3_CAPTURE`, which
# needs no node, no schema and no graph change -- and which is already the
# pattern this repo uses for exactly this problem.
