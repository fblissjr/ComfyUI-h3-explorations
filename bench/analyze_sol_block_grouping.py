#!/usr/bin/env python3
"""Sol-Attn groups tokens 64 at a time. Which side of that grouping costs the accuracy: the queries, the keys, the rule, or nothing that a reused decision could not pay for? Float reference on captures, CPU only.

    python bench/analyze_sol_block_grouping.py --check-only
    python bench/analyze_sol_block_grouping.py CAP.pt [...] --out bench/results/<date>_sol_block_grouping.json
    python bench/analyze_sol_block_grouping.py --steps CAP.pt CAP.pt [...] --out <the same json>
    python bench/analyze_sol_block_grouping.py CAP.pt --figures <dir> --out <the same json>
    python bench/analyze_sol_block_grouping.py --summarize <json> > <date>_sol_block_grouping.md

## The question

`sweep_sol_block_size_on_capture.py` asked whether 64 tokens per block is a
ceiling and answered "16 is lower on nearly every cell". It could not say WHY,
and "make the block smaller" is a kernel rewrite. A block is one grouping
decision with two sides and a rule:

  - the KEY side. 64 keys collapse to one mean key and one summed value, so a
    key block holding one token this query needs and 63 it does not is a bad
    summary however the routing behaves.
  - the QUERY side. 64 queries share ONE routing decision, so a block whose
    queries want different keys is served badly whatever the rule does.
  - the RULE. The decision is the query-block centroid against the centred
    key-block centroid, thresholded at tau sigma. A better rule at the SAME
    granularity would be free of any kernel-layout change.

and a fourth axis that decides what a finer decision could cost: if the routed
set barely moves between denoising steps, an expensive decision can be taken
once and reused.

## What it computes

Per capture, per token ordering (raster, the node's `3d` Morton) and per block
size (64, 16), at tau 1.0 with the shipped sink ranges:

  A  within-block spread. Per video block and head, the mean squared distance
     of the block's keys from the block's mean key over the mean squared norm
     of those keys (0 = identical tokens, about 1 = unrelated). Queries too.
     This is how unlike the tokens that share a block are; the pooled term
     replaces them all with the mean, so it is the key side's raw material.
  B  missed mass. For a fixed stratified sample of video query tokens, the
     query's EXACT softmax row over all keys, then (i) the share of its mass
     sitting in key blocks Sol did not route exactly, (ii) among those pooled
     blocks, the share of their mass carried by the top eighth of their tokens
     (mass-weighted; 8 of 64, 2 of 16), and (iii) the smallest number of
     individual key tokens covering 90 percent of the row, against the number
     of tokens Sol attended exactly.
  C  query-side disagreement. For sampled query blocks, each individual
     query's own ideal key-block set (its top-k blocks by exact mass, k = what
     the block actually routed) against the block's actual routed set, and
     against its neighbours' ideal sets. Reported twice: over all blocks, and
     over DISCRETIONARY blocks only, with the forced ones (diagonal +-1 and
     the sinks) removed from both sets and from k, because those agree by
     construction and inflate the overlap on both sides.
  D  a Quest-style bound as a candidate routing rule. Per key block, the
     per-channel min and max of its keys; the upper bound on any token score
     in that block from a query centroid is the sum over channels of
     max(q_c min_c, q_c max_c). Rank by that instead of by the centroid score,
     take the same NUMBER of blocks Sol took, and report B(i) again.
  E  route stability across steps. The Jaccard overlap of the routed key-block
     sets, per query block and head, between two captured steps of one DiT
     block. All blocks and discretionary only, as in C.

## How to read it

**Cells are at equal tau, NOT at equal cost.** tau is pinned at 1.0, so the
four (ordering, size) cells route at four different densities, and a lower
missed mass at block 16 is partly just "block 16 routed more". Every B and D
row carries its `density` and `routed_tokens` for that reason. In this data the
confound turns out to be small -- all four cells route within a few percent of
one another on every capture -- but that is an observation about these
captures, not a property of tau, and it is why the density is printed rather
than argued about. `sweep_sol_block_size_on_capture.py` is the instrument that
holds cost fixed by construction.

**Missed mass is NOT output error.** It is the share of a query's exact mass
that Sol routed through the POOLED branch, and Sol still supplies an
approximation of that mass; how good that approximation is, this does not
measure. `--cross-check` reads the output rel_l2 curves from
`sweep_sol_block_size_on_capture.py` on the same captures and counts the cells
where the two metrics rank the two orderings the same way. Read the
disagreements it names before quoting a missed-mass ratio as an error ratio.

**Every Jaccard has a chance floor and it differs per cell.** Two random size-k
subsets of N overlap at about k/(2N-k), which at these densities is around an
eighth. `jaccard_chance` sits beside every measured Jaccard; a number near it
means no agreement at all.

A and B(ii) say whether the KEY side is the problem: high spread with mass
concentrated in a few tokens of a pooled block means the pooled mean is
throwing away something a finer key block would keep. C says whether the QUERY
side is: low discretionary overlap between a block's own queries means one
decision per 64 queries cannot be right for all of them. D says whether the
RULE is: if the Quest bound recovers most of the missed mass at the same block
count, granularity was never the binding constraint. E says what a finer
decision could be amortised over.

## Limits

Captured blocks and steps of two clips only; a head PREFIX of 8 of 56, not a
sample; fp32 throughout, so there is no INT8 term and this is a ceiling on what
a grouping change could buy rather than a forecast of a kernel. B, C and D are
properties of the exact attention distribution on captured q/k/v at one step of
one DiT block: they bound what finer granularity COULD buy and predict neither
a kernel nor a clip. Captures must have been taken with reordering OFF; the
payload cannot say so, and that is on the caller. Routing COST, which grows
with the square of the block count, is in no number here.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import math
import re
import sys
import time
from datetime import date
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import _sol_attn_reference as oracle  # noqa: E402
import analyze_sol_error as ase  # noqa: E402
from _live_sol import live_sol  # noqa: E402

BLOCK_SIZES = (64, 16)
ORDERINGS = ("raster", "3d")
NODE_BLOCK = 64                 # the unit `_sink_blocks` and `_perm_for` speak in
TAU = 1.0
TOP_FRACTION = 8                # B(ii): the top 1/8 of a block's tokens
COVER = 0.9                     # B(iii): the mass a token set must cover

#: The reading of the tables, kept here rather than typed into the record so
#: `--summarize` rebuilds the whole document from the data file and no number
#: is ever transcribed into prose. Everything before "What follows" is read off
#: the tables above it; everything after is inference and is labelled as one.
VERDICT = [
    "**There are two regimes and they want different things.** On DiT block 0 the",
    "attention is diffuse: the median query needs more than twenty thousand",
    "individual key tokens to cover 90 percent of its mass, which is about what Sol",
    "routes anyway, and a pooled block's mass is spread across its tokens rather",
    "than concentrated. On blocks 24, 32, 40 and 49 the median query needs a few",
    "hundred tokens while Sol routes tens of thousands, and a pooled block carries",
    "almost all of its mass in the top eighth of its tokens. Every statement below",
    "is about the second regime; nothing measured here improves the first.",
    "",
    "**The three levers do not rest on the same evidence.** The block-size lever is",
    "corroborated by output error on these same captures (`2026-09-18_sol_block_size.md`,",
    "where block 16 is lower on nearly every cell). The query-side and routing-rule",
    "levers below rest on missed mass alone, which is not error. Treat them as",
    "well-motivated directions to measure next, not as established gains.",
    "",
    "**The block size is the smallest of the three levers.** Going from 64 to 16 at",
    "the same tau lowers missed mass to roughly three quarters of the block-64",
    "figure. Taking the SAME number of blocks but ranking them by the true largest",
    "token score in each block rather than by the centroid score lowers it further",
    "on the mid blocks. Taking the same number of blocks but letting each query",
    "choose its own set -- the per-query oracle, which no per-block rule can beat --",
    "lowers it to a small fraction of Sol's on those blocks. That ordering is stable",
    "across both clips.",
    "",
    "**The query side is where the headroom is, and C says so from the other",
    "direction.** A block's own queries agree with the block's discretionary routed",
    "set at a Jaccard well above the chance floor and far from 1, and they agree",
    "with each other no better. The block with the lowest agreement is the block",
    "where the per-query oracle gains the most. One routing decision per 64 queries",
    "is not right for all 64 of them.",
    "",
    "**The Quest bound is the wrong rule here, and that is not an implementation",
    "defect.** The bound dominates the true block maximum on every sampled pair, as",
    "it must, and still ranks worse than the shipped centroid rule on most cells.",
    "The per-channel min/max relaxation is too loose at head dimension 128 over 64",
    "keys: it rewards a block for being wide rather than for being relevant. The",
    "`blockmax` arm is the same objective computed exactly and it does beat the",
    "centroid rule, so the objective is fine and the relaxation is not.",
    "",
    "**Routing decisions survive between steps.** Adjacent captured steps keep most",
    "of the routed set, steps six to eight apart keep well over half, and the widest",
    "pair measured -- eleven steps -- still keeps about half, against a chance floor",
    "near an eighth. The `3d` order neither helps nor hurts this.",
    "",
    "**What follows is inference, not measurement.** The per-query oracle is a",
    "bound, not a rule: it is what a decision taken per query could reach. The cheap",
    "approximation to it -- splitting a query block into two or four sub-blocks that",
    "route separately against the same key blocks -- is NOT measured here and should",
    "be the next thing measured, because it costs routing work linear in the split",
    "and no change to the key-side layout. `blockmax` is not a rule either:",
    "computing a true per-block maximum per query block is a full score pass. What",
    "it argues is that a per-block key SUMMARY approximating a maximum rather than a",
    "mean is worth designing, and that E is what would pay for a summary too",
    "expensive to rebuild every step. Finer blocks remain the option that needs a",
    "kernel rewrite and buys the least of the three, and their routing cost, which",
    "grows with the square of the block count, is in none of these numbers.",
]


VERDICT_ARMS = [
    "",
    "## The arms, the frozen pattern and the frame residue",
    "",
    "**Output error and missed mass disagree, and the disagreement is systematic.**",
    "The per-query oracle has the least missed mass in every cell by construction,",
    "and on the late blocks it does NOT have the least error. It optimises the mass",
    "routed exactly, not the output, so it is free to drop blocks the pooled tail",
    "happens to approximate well and keep ones it approximates badly. Read the F",
    "tables in that order: `rel_l2` decides, missed mass explains.",
    "",
    "**Both levers together are the best cheap arm, and the kernel's price undoes",
    "most of it.** `split4_lse4` -- a 16-row query sub-block routing with a key",
    "score taken as the log-sum-exp over four sub-means -- has the lowest error of",
    "every implementable arm on every cell measured, at Sol's cost per sub-block.",
    "But a 64-row CTA walks the UNION of its four sub-blocks' sets, and the",
    "inflation table shows that union is materially larger than the per-sub-block",
    "cost. Tuned so the UNION costs what Sol costs, a pure query split loses to Sol",
    "on most cells. So a query split is an accuracy lever at a fixed union, not a",
    "free win, and the union is the number any kernel proposal has to quote.",
    "",
    "**Changing only the key-side RANKING is the cheapest thing on the table.**",
    "`max4` and `lse4` touch neither the exact stage nor the tile layout: they swap",
    "one mean for a max or a log-sum-exp over four sub-means in the route kernel.",
    "They are a small, consistent gain on nearly every cell, and they are the only",
    "arm here with no layout consequence at all.",
    "",
    "**The token stage that already exists reproduces this pack's own grade of it,",
    "from an independent emulation.** At equal cost `token_aug` is a large win on",
    "DiT block 0 and a loss on block 49, which is the shape",
    "`docs/research/2026-09-04_sol_token_aug_grade.md` recorded on the kernel. The",
    "emulation adds the mechanism: most of the win is not the admitted tokens but",
    "the per-token tail the kernel builds from the candidates that miss the cut,",
    "which replaces one block-pooled term per unrouted block with one term per",
    "token at the group centroid's own score. That is why it helps exactly where",
    "the attention is diffuse.",
    "",
    "**HISA's cut is a loss here, not a saving.** Restricting the token pass to the",
    "top unrouted blocks by the route's own block score is worse than the full scan",
    "on every cell, and much worse on block 0. The candidate mass this model needs",
    "is not concentrated in the blocks the centroid ranks highest.",
    "",
    "**What limits the token stage is the shared centroid, not the budget.** Ranking",
    "the SAME candidate tokens by the query's own exact mass instead of by the",
    "centroid shared across 128 query rows collapses the tokens needed to recover",
    "half of Sol's missed mass by one to two orders of magnitude on the mid blocks.",
    "The kernel's budget ceiling is not what is binding.",
    "",
    "**LoSA's premise holds and gives Sol nothing.** A frozen key-block set keeps a",
    "high share of the later steps' mass, close to the paper's own curve, and its",
    "cost is the problem: reaching the mass target needs far more block coverage",
    "than Sol routes, so it is not a speed candidate at its own operating point. At",
    "MATCHED cardinality, Sol rebuilt at each step matches or beats the frozen set",
    "at the far steps on every DiT block measured. The research file names that",
    "outcome in advance as the clean negative, and it is what the sample shows. The",
    "frozen set's fifth percentile decays faster than its mean, so the heads that",
    "drift are not the average ones.",
    "",
    "**The one-pixel-frame latents are already routed, so there is no prior to",
    "add.** The residue-0 frames do draw more mass than their share of keys on every",
    "cell but block 0, reproducing the mass record. Sol's MISSED mass on them is at",
    "or below their share of the true mass on every cell: Sol misses them LESS than",
    "their mass warrants, and the classes it misses disproportionately are residues",
    "1 to 3. The residue control separates them cleanly, so this is an informative",
    "negative rather than an absent effect, and a routing prior favouring residue 0",
    "would be pushing on a door Sol already has open.",
    "",
    "**What follows is inference.** The one arm worth a kernel conversation is the",
    "key-side ranking, because it is confined to the route kernel and costs a fixed",
    "multiple of a scorer that is already a tiny fraction of the call. A query split",
    "needs its union priced before it is worth designing, and this record prices it",
    "for `split4` only. `token_aug` is already in the kernel and already has a",
    "per-block profile in the shipped config; what this adds is a reason for the",
    "shape of that profile and a warning that HISA's cut would remove the benefit.",
    "And every one of these is a capture proxy: the research file's own caution is",
    "that an estimator which won on two offline proxies lost in pixels on every",
    "prompt sglang tried, so the most any row here can do is nominate one candidate",
    "for a blind render panel.",
]


#: Every function a recorded number actually passes through. The fingerprint
#: below hashes THESE and nothing else, so editing a control or the report
#: generator does not invalidate a measurement it cannot have changed, while
#: touching any arithmetic on the measurement path does.
MEASUREMENT_PATH = (
    "block_lengths", "forced_mask", "route_mask", "key_block_bounds", "quest_rank",
    "topk_mask_by_count", "_sol_from_mask", "within_block_spread", "ordering_index",
    "video_blocks", "stratified_video_queries", "route_scores", "key_submean_scores",
    "forced_for", "_cost_of", "bisect_tau", "union_rows", "token_aug_group",
    "sol_style_output", "run_arms", "measure_query_disagreement", "run_step_pairs",
    "run_frames", "block_masses", "prefix_sets", "diagonal_floor", "run_losa",
)


def measurement_fingerprint() -> str:
    """A short hash of the measurement path's source, for pinning a record to its code.

    A stored number and the controls beside it have to come from one revision,
    or a record can claim a control passed for code that no longer exists. This
    hashes the source of `MEASUREMENT_PATH` only: a control fix or a wording
    change in the report is not a reason to distrust a number, and a change to
    the arithmetic is.
    """
    import ast
    import hashlib
    src = Path(__file__).read_text()
    tree = ast.parse(src)
    want = set(MEASUREMENT_PATH)
    chunks = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want:
            chunks.append(ast.get_source_segment(src, node) or "")
            want.discard(node.name)
    if want:
        raise RuntimeError(f"measurement_fingerprint: {sorted(want)} not found at module level")
    return hashlib.sha256("\n".join(chunks).encode()).hexdigest()[:16]



# ----------------------------------------------------------------- instrument

@contextlib.contextmanager
def block_size(size):
    """The vendored oracle at another block size; it reads a module global.

    `sweep_sol_block_size_on_capture.block_size` also patches comfy_kitchen's
    eager module, which drags in a CUDA package this script does not need and a
    CPU-only box may not have. Same idea, smaller blast radius.
    """
    mods = (oracle, ase)
    old = [m.BLOCK for m in mods]
    for m in mods:
        m.BLOCK = size
    try:
        yield
    finally:
        for m, o in zip(mods, old):
            m.BLOCK = o


def block_lengths(t: int, size: int) -> torch.Tensor:
    """Live rows per block: `size` everywhere but a ragged last block."""
    n = (t + size - 1) // size
    lengths = torch.full((n,), float(size))
    if n * size - t:
        lengths[-1] = float(t - (n - 1) * size)
    return lengths


def forced_mask(n: int, sink_kv, sink_q) -> torch.Tensor:
    """[N, N] bool: the pairs the algorithm keeps exact whatever the score says."""
    idx = torch.arange(n)
    forced = (idx.view(1, -1) - idx.view(-1, 1)).abs() <= 1
    forced |= ((idx >= sink_kv[0]) & (idx < sink_kv[1])).view(1, n)
    forced |= ((idx >= sink_q[0]) & (idx < sink_q[1])).view(n, 1)
    return forced


def route_mask(q, k, tau, sink_kv, sink_q, size):
    """([H, N, N] bool routed, [H, N, D] query-block centroids) from fp32 q, k of shape [1, H, T, D].

    The query-block mean of the block scores is linear in the queries, so it is
    the centroid against the centred key centroids: one N x N product per head.
    Same arithmetic as `sweep_sol_block_size_on_capture.routed_counts`, which
    returns only the counts; this keeps the mask.
    """
    _b, h, t, d = q.shape
    n = (t + size - 1) // size
    log2s = (d ** -0.5) * ase._LOG2E
    lengths = block_lengths(t, size)
    forced = forced_mask(n, sink_kv, sink_q)
    pad = n * size - t
    mask = torch.empty((h, n, n), dtype=torch.bool)
    centroids = torch.empty((h, n, d), dtype=torch.float32)
    for head in range(h):
        fq, fk = q[0, head], k[0, head]
        if pad:
            fq = F.pad(fq, (0, 0, 0, pad))
            fk = F.pad(fk, (0, 0, 0, pad))
        centroid = fq.view(n, size, d).sum(1) / lengths.view(n, 1)
        kc = fk.view(n, size, d).sum(1) / lengths.view(n, 1)
        kcc = kc - kc.mean(0, keepdim=True)
        kc_var = kcc.pow(2).mean(0)
        colmean = (centroid @ kcc.T) * log2s
        thr = tau * torch.sqrt((centroid.pow(2) * kc_var).sum(-1) * log2s * log2s + 1e-6)
        mask[head] = (colmean > thr.view(n, 1)) | forced
        centroids[head] = centroid
    return mask, centroids


def key_block_bounds(k, size):
    """Per-channel (min, max) of each key block, [H, N, D] each, live rows only."""
    _b, h, t, d = k.shape
    n = (t + size - 1) // size
    full = (n - 1) * size
    kmin = torch.empty((h, n, d), dtype=torch.float32)
    kmax = torch.empty((h, n, d), dtype=torch.float32)
    for head in range(h):
        fk = k[0, head]
        body = fk[:full].view(n - 1, size, d)
        kmin[head] = torch.cat([body.amin(1), fk[full:].amin(0, keepdim=True)])
        kmax[head] = torch.cat([body.amax(1), fk[full:].amax(0, keepdim=True)])
    return kmin, kmax


def cross_check_rel_l2(doc, prior_paths):
    """Does missed mass rank the two orderings the way OUTPUT ERROR does? Per cell, per block size.

    Missed mass is the share of a query's exact mass routed through the pooled
    branch. It is not error: Sol still supplies an approximation of that mass,
    and the fidelity of the pooled term is a separate quantity nothing here
    measures. So a cell can push less mass into the pooled branch and still
    come out further from dense attention.

    `sweep_sol_block_size_on_capture.py` measured output rel_l2 against fp32
    dense attention on the SAME captures, the same head prefix and the same
    tau, at matched routed density. This reads those curves at the density
    plain order routes at tau 1.0 and counts the cells where the two metrics
    put the two orderings in the same order. Disagreements are named, not
    averaged away: they are the cells where this record must not be read as a
    statement about error.
    """
    import orjson
    import sweep_sol_block_size_on_capture as sweep
    curves = {}
    for path in prior_paths:
        data = orjson.loads(Path(path).read_bytes())
        capset = data.get("conditions", {}).get("capture_set", Path(path).stem)
        for cell, rows in data["cells"].items():
            curves[f"{capset}/{cell}"] = rows
    out = {"prior_records": [Path(p).name for p in prior_paths], "by_block_size": {}}
    for size in BLOCK_SIZES:
        same, total, disagree = 0, 0, []
        for cid, cell in doc["cells"].items():
            rows = curves.get(cid)
            if rows is None or f"3d_b{size}" not in rows:
                continue
            base = next((x for x in rows["raster_b64"] if x["tau"] == TAU), None)
            if base is None:
                continue
            er = sweep.error_at(rows[f"raster_b{size}"], base["density"])
            e3 = sweep.error_at(rows[f"3d_b{size}"], base["density"])
            if er is None or e3 is None:
                continue
            mr = cell["B"][f"raster_b{size}"]["missed_mass"]["mean"]
            m3 = cell["B"][f"3d_b{size}"]["missed_mass"]["mean"]
            total += 1
            if (e3 < er) == (m3 < mr):
                same += 1
            else:
                disagree.append({"cell": cid, "rel_l2_3d_over_raster": round(e3 / er, 3),
                                 "missed_3d_over_raster": round(m3 / mr, 3)})
        out["by_block_size"][str(size)] = {"cells": total, "same_ordering": same,
                                           "disagreements": disagree}
    return out


def quest_rank(centroid, kmin, kmax):
    """[M, N] Quest upper bound of any token score in each key block, from query centroids [M, D].

    sum_c max(q_c min_c, q_c max_c) = q+ . max + q- . min, so it is two matmuls.
    Centring the keys shifts every bound in a row by one per-query constant, so
    it does not change the ranking; these are the RAW captured keys.
    """
    return centroid.clamp(min=0) @ kmax.T + centroid.clamp(max=0) @ kmin.T


def topk_mask_by_count(score, counts):
    """[.., N] bool: EXACTLY the `counts` highest entries per row, `counts` an int tensor shaped like score[..., 0].

    The count varies per row, so this takes the largest count once and trims
    each row to its own. The obvious alternative -- sort, read the count-th
    value, keep everything at least that large -- is wrong here rather than
    merely approximate: these scores are softmax masses, most blocks carry a
    mass that underflows fp32 to exactly zero, and a threshold of zero keeps
    EVERY block. `controls()` demonstrates that failure on a fixture.
    """
    kmax = int(counts.max())
    idx = torch.topk(score, kmax, dim=-1).indices
    keep = torch.arange(kmax).expand(idx.shape) < counts.unsqueeze(-1)
    out = torch.zeros_like(score, dtype=torch.bool)
    out.scatter_(-1, idx, keep)
    return out


def _sol_from_mask(q, k, v, mask, size):
    """Sol-Attn's output built from a SUPPLIED route mask. Dense, for the control only.

    Everything but the routing decision is the vendored algorithm: pooled mean
    keys, summed values, centred keys, the query block's centroid tail, one
    softmax over both branches. If the mask disagrees with the oracle's own
    decision anywhere, the outputs disagree.
    """
    _b, h, t, d = q.shape
    n = mask.shape[-1]
    log2s = (d ** -0.5) * ase._LOG2E
    lengths = block_lengths(t, size)
    pad = n * size - t
    fq, fk, fv = q[0], k[0], v[0]
    pk = F.pad(fk, (0, 0, 0, pad)) if pad else fk
    pv = F.pad(fv, (0, 0, 0, pad)) if pad else fv
    kc = pk.view(h, n, size, d).sum(2) / lengths.view(1, n, 1)
    vc = pv.view(h, n, size, d).sum(2)
    k_mean = kc.mean(1, keepdim=True)
    kcc = kc - k_mean
    kh = fk - k_mean
    qblk = torch.arange(t) // size

    s_tok = (fq @ kh.transpose(-1, -2)) * log2s                      # [H, T, T]
    s_blk = (fq @ kcc.transpose(-1, -2)) * log2s                     # [H, T, N]
    colmean = torch.zeros(h, n, n)
    colmean.scatter_add_(1, qblk.view(1, t, 1).expand(h, t, n), s_blk)
    colmean = colmean / lengths.view(1, n, 1)

    ex_tok = mask.gather(1, qblk.view(1, t, 1).expand(h, t, n))      # [H, T, N]
    keep = ex_tok.repeat_interleave(size, dim=-1)[..., :t]
    neg = torch.finfo(torch.float32).min
    s_tok = s_tok.masked_fill(~keep, neg)
    s_tail = colmean.gather(1, qblk.view(1, t, 1).expand(h, t, n)).masked_fill(ex_tok, neg)

    logits = torch.cat([s_tok, s_tail], dim=-1)
    p = torch.exp2(logits - logits.amax(dim=-1, keepdim=True))
    p = p.masked_fill(logits <= neg, 0.0)
    num = p[..., :t] @ fv + p[..., t:] @ vc
    den = p[..., :t].sum(-1) + (p[..., t:] * lengths.view(1, 1, n)).sum(-1)
    return num / den.clamp_min(1e-30).unsqueeze(-1)


def controls():
    """Everything downstream rests on `route_mask`. This is what would turn it red.

    1. Its row sums must equal `sweep_sol_block_size_on_capture.routed_counts`,
       which is reviewed code that reports the same counts a different way.
       Weak on its own -- same formula -- but it pins the sink and ragged-block
       handling.
    2. Sol rebuilt from the mask must match the VENDORED oracle at a ragged and
       an aligned length, at both block sizes. This is the real control: a
       mask wrong for one (query block, key block) pair moves the output.
    3. A mutation: one routed pair flipped off must turn 2 red. A check that
       cannot fail is not a check.
    4. `topk_mask_by_count` must keep exactly its count on a row of softmax
       masses where most entries have underflowed to zero, which is the shape
       every C and D ranking actually has. The threshold spelling this
       replaced kept the whole row there, so the number is reported rather
       than asserted: it is why the helper is written the way it is.
    """
    import sweep_sol_block_size_on_capture as sweep
    out = {}
    # 4, first, because it is cheap and it is a property of the helper alone
    torch.manual_seed(3)
    logits = torch.randn(4, 500) * 30.0
    masses = torch.softmax(logits, dim=-1)
    # a count that reaches past the last non-zero entry, which is exactly what
    # a query block routing 350 of 1631 blocks asks for on a concentrated row
    want = ((masses > 0).sum(-1) + 10).clamp(max=masses.shape[-1])
    got = topk_mask_by_count(masses, want)
    thr = torch.sort(masses, dim=-1, descending=True).values.gather(-1, (want - 1).unsqueeze(-1))
    out["topk_rows_with_underflowed_zeros"] = int((masses == 0).any(-1).sum())
    out["topk_kept_exactly_the_count"] = bool(torch.equal(got.sum(-1), want))
    out["topk_threshold_spelling_would_have_kept"] = float((masses >= thr).sum(-1).float().mean())
    worst_count, worst_out, mutated = 0, 0.0, 1.0
    for size in BLOCK_SIZES:
        for t in (514, 1024):                      # ragged, then block-aligned
            torch.manual_seed(7)
            h, d = 2, 64
            q, k, v = (torch.randn(1, h, t, d) for _ in range(3))
            scale = NODE_BLOCK // size
            sink_kv = (0, 2 * scale)
            sink_q = (1 * scale, 3 * scale)
            mask, _ = route_mask(q, k, TAU, sink_kv, sink_q, size)
            theirs = sweep.routed_counts(q, k, TAU, sink_kv, sink_q, size, torch.device("cpu"))
            worst_count = max(worst_count, int((mask.sum(-1) - theirs).abs().max()))
            with block_size(size):
                bthd = lambda x: x.permute(0, 2, 1, 3).contiguous()   # noqa: E731
                ref = oracle.sol_attn(bthd(q), bthd(k), bthd(v), tau=TAU,
                                      sink_blocks=list(sink_kv), sink_q=list(sink_q))
            ref = ref.permute(0, 2, 1, 3)[0].float()
            mine = _sol_from_mask(q, k, v, mask, size)
            worst_out = max(worst_out, float((mine - ref).norm() / ref.norm()))
            if t == 514 and size == 64:
                bad = mask.clone()
                free = (~forced_mask(mask.shape[-1], sink_kv, sink_q)) & bad[0]
                where = free.nonzero()[0]
                bad[0, where[0], where[1]] = False
                broke = _sol_from_mask(q, k, v, bad, size)
                mutated = float((broke - ref).norm() / ref.norm())
    out["routed_counts_vs_sweep_worst_block_difference"] = worst_count
    out["mask_rebuilt_sol_vs_vendored_oracle_worst_rel_l2"] = worst_out
    out["mutation_one_routed_pair_dropped_rel_l2"] = mutated

    # 5. measurement F's own arithmetic, ISOLATED. `sol_style_output` builds
    #    Sol's output for a set of query ROWS under an arbitrary route, and that
    #    is what gives every arm in F an output error. It is graded here against
    #    `_sol_from_mask` -- proven bit-identical to the vendored oracle by
    #    control 2 -- GIVEN THE SAME MASK, so the routing decision is not part
    #    of the comparison and the two differ only in arithmetic.
    #
    #    It used to be graded end to end against `eager_sol_reference`, which
    #    makes its OWN routing decision from a different float32 reduction order
    #    (one matmul here, a scatter-add of per-token scores there). That
    #    control was environment-dependent: green on this box, red on another
    #    at 4.59e-02, and the cause is one discretionary pair at t=1024 sitting
    #    about 1e-6 from the threshold -- flipping that single pair reproduces
    #    the red value to four figures. A control whose oracle re-decides the
    #    thing under test cannot tell a tie from a defect, so the end-to-end
    #    number and the tightest threshold margin are REPORTED below and no
    #    longer gate. `probe_oracle_gate_scaling.py` measured the same effect
    #    on the shipped gate; this is that finding arriving a second time.
    worst_f, mask_same, worst_e2e, tightest = 0.0, True, 0.0, float("inf")
    mutated_f = 0.0
    for t in (514, 1024):
        torch.manual_seed(11)
        h, d = 2, 64
        q, k, v = (torch.randn(1, h, t, d) for _ in range(3))
        sink_kv, sink_q = (0, 2), (1, 3)
        n = (t + NODE_BLOCK - 1) // NODE_BLOCK
        colmean, base, _kcc, k_mean, lengths = route_scores(q, k, NODE_BLOCK, NODE_BLOCK)
        forced = forced_for(n, n, NODE_BLOCK, NODE_BLOCK, sink_kv, sink_q)
        mask = (colmean > TAU * base.unsqueeze(-1)) | forced
        mask_same &= bool(torch.equal(mask, route_mask(q, k, TAU, sink_kv, sink_q, NODE_BLOCK)[0]))
        free = ~forced.unsqueeze(0).expand_as(colmean)
        tightest = min(tightest, float((colmean - TAU * base.unsqueeze(-1)).abs()[free].min()))
        blk = torch.arange(t) // NODE_BLOCK
        scale = d ** -0.5
        s_raw = torch.einsum("hcd,hsd->hcs", q[0], k[0]) * scale
        s_log2 = (s_raw - torch.einsum("hcd,hd->hc", q[0], k_mean[:, 0, :]).unsqueeze(-1) * scale) * ase._LOG2E
        vc = torch.stack([F.pad(v[0, hh], (0, 0, 0, n * NODE_BLOCK - t)).view(n, NODE_BLOCK, d).sum(1)
                          for hh in range(h)])
        routed = mask[:, blk, :]
        et = routed.gather(2, blk.view(1, 1, -1).expand(h, t, t))
        mine = sol_style_output(s_log2, v[0], vc, lengths, et, ~routed, colmean[:, blk, :])
        same_mask_ref = _sol_from_mask(q, k, v, mask, NODE_BLOCK)
        worst_f = max(worst_f, float((mine - same_mask_ref).norm() / same_mask_ref.norm()))
        ref = ase.eager_sol_reference(q, k, v, tau=TAU, sink_blocks=list(sink_kv),
                                      sink_q=list(sink_q))[0]
        worst_e2e = max(worst_e2e, float((mine - ref).norm() / ref.norm()))
        if t == 514:
            # the mutation: the pooled tail carries real weight, so scaling its
            # logit must move the output. Without this the control could pass by
            # ignoring the branch that distinguishes Sol from dense attention.
            bad = sol_style_output(s_log2, v[0], vc, lengths, et, ~routed,
                                   colmean[:, blk, :] * 1.05)
            mutated_f = float((bad - same_mask_ref).norm() / same_mask_ref.norm())
    out["route_scores_threshold_equals_route_mask"] = mask_same
    out["sol_style_output_vs_same_mask_reference_worst_rel_l2"] = worst_f
    out["sol_style_output_tail_scaled_mutation_rel_l2"] = mutated_f
    out["reported_not_gating"] = {
        "sol_style_output_vs_chunked_reference_worst_rel_l2": worst_e2e,
        "tightest_discretionary_threshold_margin": tightest,
        "why_not_gating": "the chunked reference re-decides the routing in a different float32 "
                          "reduction order, so a pair within fp32 noise of the threshold flips "
                          "with the BLAS path and this number jumps by a whole block's worth. "
                          "It is a tie, not a divergence; control 5 removes the tie by fixing "
                          "the mask on both sides.",
    }

    out["passed"] = bool(worst_count == 0 and worst_out < 1e-5
                         and mask_same and worst_f < 1e-5
                         and mutated > 100 * max(worst_out, 1e-9)
                         and mutated_f > 100 * max(worst_f, 1e-9)
                         and out["topk_kept_exactly_the_count"])
    return out


# ------------------------------------------------------------------ geometry

def ordering_index(node, name, grid, start, tokens):
    """`index[position] = token` for one ordering; only the video rows move."""
    if name == "raster":
        index = torch.arange(tokens)
    else:
        perm, _inv = node._perm_for(grid, name, "cpu", start)
        index = torch.cat([torch.arange(start), start + perm.cpu()])
    assert torch.equal(torch.sort(index).values, torch.arange(tokens)), f"{name}: not a bijection"
    return index


def video_blocks(start, tokens, size):
    """Block ids lying wholly inside the video span AND wholly live.

    The video span starts mid-block (1545 is 24 blocks and 9 rows), so one
    block mixes conditioning with video and the last block is ragged. Both are
    dropped, so raster and `3d` describe the same set of blocks.
    """
    n = (tokens + size - 1) // size
    first = (start + size - 1) // size
    last = n if n * size == tokens else n - 1
    return torch.arange(first, last)


def stratified_video_queries(grid, start, per_frame, seed):
    """`per_frame` raster token indices from each latent frame, fixed seed.

    The SAME physical tokens for every ordering and block size, so a cell
    difference is the grouping and not the sample.
    """
    frames, height, width = grid
    gen = torch.Generator().manual_seed(seed)
    area = height * width
    picks = []
    for t in range(frames):
        sel = torch.randperm(area, generator=gen)[:per_frame]
        picks.append(start + t * area + sel.sort().values)
    return torch.cat(picks)


def jaccard_chance(k1, k2, n):
    """Expected Jaccard of two independent uniform subsets of sizes k1, k2 from n."""
    inter = k1 * k2 / max(n, 1)
    union = k1 + k2 - inter
    return float(inter / union) if union > 0 else 0.0


def dist(x):
    """The shape of a distribution, not a single number."""
    f = x.flatten().float()
    qs = torch.quantile(f, torch.tensor([0.1, 0.5, 0.9]))
    return {"mean": round(float(f.mean()), 6), "p10": round(float(qs[0]), 6),
            "median": round(float(qs[1]), 6), "p90": round(float(qs[2]), 6)}


# ------------------------------------------------------------ A: block spread

def within_block_spread(x, size, blocks):
    """[H, len(blocks)] mean squared distance from the block mean over mean squared norm.

    mean||x - xbar||^2 = mean||x||^2 - ||xbar||^2, so it is one pass and the
    ratio is in [0, 1]: 0 identical, about 1 mutually unrelated.
    """
    _b, h, t, d = x.shape
    n = (t + size - 1) // size
    lengths = block_lengths(t, size)
    pad = n * size - t
    fx = F.pad(x[0], (0, 0, 0, pad)) if pad else x[0]
    blk = fx.view(h, n, size, d)
    bar = blk.sum(2) / lengths.view(1, n, 1)
    msn = blk.pow(2).sum(-1).sum(-1) / lengths.view(1, n)
    return (1.0 - bar.pow(2).sum(-1) / msn.clamp_min(1e-30))[:, blocks]


# --------------------------------------------------------- the per-capture run

def cell_key(ordering, size):
    return f"{ordering}_b{size}"


def run_capture(path, args, node, recipe, verbose=True):
    """Everything A to D for one capture file. Returns one cell dict."""
    t0 = time.time()
    q, k, v = ase.load_capture(path)
    q, k, v = (x[:, :args.heads].to(torch.float32).contiguous() for x in (q, k, v))
    tokens = q.shape[2]
    grid = tuple(int(x) for x in args.grid.split(","))
    start = args.video_start
    a0, a1 = (int(x) for x in args.audio_span.split(","))
    if start + grid[0] * grid[1] * grid[2] != tokens:
        raise SystemExit(f"{Path(path).name}: layout says video starts at {start} with grid {grid}, "
                         f"capture has {tokens} rows")
    d = q.shape[-1]
    scale = d ** -0.5
    layout = {"sol_h3_video_span": (start, tokens), "sol_h3_audio_span": (a0, a1)}
    sink_kv64, sink_q64 = node._sink_blocks(layout, tokens, recipe["sink_conditioning"])

    sample = stratified_video_queries(grid, start, args.per_frame, args.seed)
    heads = q.shape[1]
    # the layout is per capture SET, not global: the two sets pack a different
    # number of text rows, so one --video-start cannot describe both and a
    # record that carried only the last one would mislabel the earlier cells
    cell = {"tokens": int(tokens), "heads": heads, "sample_queries": int(sample.numel()),
            "video_start": start, "audio_span": [a0, a1], "grid_thw": list(grid)}

    # ---- the query's exact attention row is the same physical object for every
    # ordering and size, so it is computed once and every cell reads it.
    prepared = {}
    for ordering in ORDERINGS:
        index = ordering_index(node, ordering, grid, start, tokens)
        pos = torch.argsort(index)
        for size in BLOCK_SIZES:
            sc = NODE_BLOCK // size
            sink_kv = (sink_kv64[0] * sc, sink_kv64[1] * sc)
            sink_q = (sink_q64[0] * sc, sink_q64[1] * sc)
            n = (tokens + size - 1) // size
            qs = q[:, :, index]
            ks = k[:, :, index]
            mask, centroid = route_mask(qs, ks, TAU, sink_kv, sink_q, size)
            vb = video_blocks(start, tokens, size)
            lengths = block_lengths(tokens, size)
            prepared[cell_key(ordering, size)] = dict(
                index=index, pos=pos, keyblk=pos // size, size=size, n=n, mask=mask, centroid=centroid,
                sink_kv=sink_kv, sink_q=sink_q, video=vb, lengths=lengths,
                forced=forced_mask(n, sink_kv, sink_q),
                spread_key=within_block_spread(ks, size, vb),
                spread_query=within_block_spread(qs, size, vb),
                kbounds=key_block_bounds(ks, size),
            )
            del qs, ks

    # ---------------------------------------------------------------- A
    cell["A_spread"] = {}
    cell["density"] = {}
    for name, p in prepared.items():
        cell["A_spread"][name] = {
            "key": dist(p["spread_key"]), "query": dist(p["spread_query"]),
            "video_blocks": int(p["video"].numel()),
        }
        cell["density"][name] = round(float(p["mask"].float().mean()), 5)
        if verbose:
            print(f"  A {name:9s} key spread mean {cell['A_spread'][name]['key']['mean']:.4f} "
                  f"query spread mean {cell['A_spread'][name]['query']['mean']:.4f} "
                  f"density {cell['density'][name]:.4f}", flush=True)

    # ----------------------------- D: candidate route masks at the same COUNT
    # Built before the B pass, for the query blocks the sample touches only.
    # Three arms beside Sol's own centroid-against-centroid decision:
    #   quest     the per-channel min/max upper bound, a rule a kernel could run
    #   blockmax  the TRUE largest token score in the block, which that bound
    #             relaxes. It separates "the bound is loose" from "ranking a
    #             block by its best token is the wrong objective here".
    #   oracle    (in the B pass) the query's own exact mass per block: the
    #             floor any rule at this granularity could reach for that query
    for name, p in prepared.items():
        qblk = p["pos"][sample] // p["size"]
        sel = torch.unique(qblk)
        row_of = torch.full((p["n"],), -1, dtype=torch.long)
        row_of[sel] = torch.arange(sel.numel())
        kmin, kmax = p["kbounds"]
        nheads, n, size = p["mask"].shape[0], p["n"], p["size"]
        counts = p["mask"][:, sel, :].sum(-1)                        # [H, M]
        quest = torch.empty((nheads, sel.numel(), n), dtype=torch.bool)
        bmax = torch.empty_like(quest)
        big = torch.finfo(torch.float32).max
        k_ord = k[0][:, p["index"], :]
        pad = n * size - k_ord.shape[1]
        valid = 0.0
        for head in range(nheads):
            bound = quest_rank(p["centroid"][head, sel], kmin[head], kmax[head]) * scale
            quest[head] = topk_mask_by_count(bound.masked_fill(p["forced"][sel], big), counts[head])
            rows = []
            for m0 in range(0, sel.numel(), 256):
                s = (p["centroid"][head, sel[m0:m0 + 256]] @ k_ord[head].T) * scale
                if pad:
                    s = F.pad(s, (0, pad), value=-big)
                rows.append(s.view(s.shape[0], n, size).amax(-1))
                del s
            true_max = torch.cat(rows)
            # the bound must dominate what it bounds, or `quest` is not Quest
            valid += float((bound >= true_max - 1e-3).float().mean()) / nheads
            bmax[head] = topk_mask_by_count(true_max.masked_fill(p["forced"][sel], big), counts[head])
            del rows, true_max, bound
        del k_ord
        p["quest_bound_dominates_true_max"] = round(valid, 6)
        p["quest"], p["blockmax"] = quest, bmax
        p["quest_row_of"] = row_of
        p["quest_overshoot"] = float((quest.sum(-1) - counts).float().mean())
        del p["kbounds"]

    # ---------------------------------------------------------------- B and D
    acc = {name: dict(missed=[], missed_quest=[], missed_blockmax=[], missed_oracle=[],
                      pooled_mass=0.0, pooled_top=0.0,
                      routed_tokens=[], routed_blocks=[]) for name in prepared}
    n90_all = []
    top_m = {name: max(1, prepared[name]["size"] // TOP_FRACTION) for name in prepared}
    for s0 in range(0, sample.numel(), args.chunk):
        qi = sample[s0:s0 + args.chunk]
        qc = q[0][:, qi, :]                                          # [H, C, D]
        scores = torch.einsum("hcd,hsd->hcs", qc, k[0]) * scale
        p_exact = torch.softmax(scores, dim=-1)
        del scores
        ps = torch.sort(p_exact, dim=-1, descending=True).values
        cs = torch.cumsum(ps, dim=-1)
        n90_all.append((cs < COVER).sum(-1) + 1)
        del ps, cs
        for name, p in prepared.items():
            size, n = p["size"], p["n"]
            qblk = p["pos"][qi] // size
            routed = p["mask"][:, qblk, :]                           # [H, C, N]
            mass = torch.zeros(p_exact.shape[0], qi.numel(), n)
            mass.index_add_(2, p["keyblk"], p_exact)
            missed = (mass * ~routed).sum(-1)
            acc[name]["missed"].append(missed)
            acc[name]["routed_tokens"].append((routed.float() * p["lengths"]).sum(-1))
            acc[name]["routed_blocks"].append(routed.sum(-1))
            row = p["quest_row_of"][qblk]
            qrouted = p["quest"][:, row, :]
            acc[name]["missed_quest"].append((mass * ~qrouted).sum(-1))
            bmrouted = p["blockmax"][:, row, :]
            acc[name]["missed_blockmax"].append((mass * ~bmrouted).sum(-1))
            del qrouted, bmrouted
            # the per-query oracle: the best block set for THIS query at this
            # count. Mass is in [0, 1], so 2.0 forces a block to the top.
            omask = topk_mask_by_count(mass.masked_fill(p["forced"][qblk].unsqueeze(0), 2.0),
                                       routed.sum(-1))
            acc[name]["missed_oracle"].append((mass * ~omask).sum(-1))
            del omask
            # concentration inside the POOLED blocks
            p_ord = p_exact if name.startswith("raster") else p_exact[..., p["index"]]
            pad = n * size - p_ord.shape[-1]
            if pad:
                p_ord = F.pad(p_ord, (0, pad))
            blkp = p_ord.view(p_exact.shape[0], qi.numel(), n, size)
            topm = torch.topk(blkp, top_m[name], dim=-1).values.sum(-1)
            pooled = ~routed
            acc[name]["pooled_mass"] += float((mass * pooled).sum())
            acc[name]["pooled_top"] += float((topm * pooled).sum())
            del mass, routed, blkp, topm, p_ord
        del p_exact, qc
        if verbose:
            print(f"  B {s0 + qi.numel()}/{sample.numel()} queries", end="\r", flush=True)
    n90 = torch.cat(n90_all, dim=1)
    cell["B"] = {}
    cell["D"] = {}
    for name, p in prepared.items():
        missed = torch.cat(acc[name]["missed"], dim=1)
        rt = torch.cat(acc[name]["routed_tokens"], dim=1)
        rb = torch.cat(acc[name]["routed_blocks"], dim=1)
        cell["B"][name] = {
            "density": cell["density"][name],
            "missed_mass": dist(missed),
            "pooled_block_top_fraction_share": round(acc[name]["pooled_top"] / max(acc[name]["pooled_mass"], 1e-30), 6),
            "top_tokens_per_block": top_m[name],
            "routed_tokens": dist(rt),
            "routed_blocks": dist(rb),
            "tokens_for_90pc_mass": dist(n90.float()),
            "routed_tokens_over_tokens_for_90pc": dist(rt / n90.float()),
        }
        mq = torch.cat(acc[name]["missed_quest"], dim=1)
        mb = torch.cat(acc[name]["missed_blockmax"], dim=1)
        mo = torch.cat(acc[name]["missed_oracle"], dim=1)
        base = missed.mean().clamp_min(1e-30)
        cell["D"][name] = {
            "density": cell["density"][name],
            "missed_mass_sol": dist(missed),
            "missed_mass_quest": dist(mq),
            "missed_mass_blockmax": dist(mb),
            "missed_mass_oracle": dist(mo),
            "quest_over_sol_mean": round(float(mq.mean() / base), 4),
            "blockmax_over_sol_mean": round(float(mb.mean() / base), 4),
            "oracle_over_sol_mean": round(float(mo.mean() / base), 4),
            "blocks_kept_over_sol": round(p["quest_overshoot"], 4),
            "quest_bound_dominates_true_max": p["quest_bound_dominates_true_max"],
        }
        if verbose:
            d = cell["D"][name]
            print(f"  B {name:9s} missed mean {cell['B'][name]['missed_mass']['mean']:.4f} "
                  f"p90 {cell['B'][name]['missed_mass']['p90']:.4f}  "
                  f"pooled top-{top_m[name]} share {cell['B'][name]['pooled_block_top_fraction_share']:.3f}", flush=True)
            print(f"  D {name:9s} missed: sol {d['missed_mass_sol']['mean']:.4f}  "
                  f"quest {d['missed_mass_quest']['mean']:.4f}  "
                  f"blockmax {d['missed_mass_blockmax']['mean']:.4f}  "
                  f"per-query oracle {d['missed_mass_oracle']['mean']:.4f}", flush=True)
    for p in prepared.values():
        del p["quest"], p["blockmax"]

    # ---------------------------------------------------------------- C
    cell["C"] = {}
    for name, p in prepared.items():
        cell["C"][name] = measure_query_disagreement(
            q, k, p, scale, args.c_blocks, args.c_queries, args.seed, args.chunk)
        if verbose:
            c = cell["C"][name]
            print(f"  C {name:9s} query-vs-block Jaccard {c['query_vs_block']['all']['mean']:.3f} "
                  f"(chance {c['jaccard_chance_all']:.3f}), discretionary "
                  f"{c['query_vs_block']['discretionary']['mean']:.3f} "
                  f"(chance {c['jaccard_chance_discretionary']:.3f})", flush=True)

    cell["seconds"] = round(time.time() - t0, 1)
    return cell


def measure_query_disagreement(q, k, p, scale, n_blocks, n_queries, seed, chunk):
    """C: do the queries of one block want the same key blocks as each other, and as the block?"""
    heads = p["mask"].shape[0]
    size, n = p["size"], p["n"]
    gen = torch.Generator().manual_seed(seed + 1)
    vb = p["video"]
    pick = vb[torch.randperm(vb.numel(), generator=gen)[:n_blocks].sort().values]
    jac_all, jac_disc, pair_all, pair_disc = [], [], [], []
    k_all, k_disc, n_disc_all = [], [], []
    per_group = max(1, chunk // max(1, min(size, n_queries)))
    for g0 in range(0, pick.numel(), per_group):
        group = pick[g0:g0 + per_group]
        rows, owners = [], []
        for b in group.tolist():
            offs = torch.randperm(size, generator=gen)[:min(size, n_queries)].sort().values
            rows.append(p["index"][b * size + offs])
            owners.append(torch.full((offs.numel(),), b, dtype=torch.long))
        rows = torch.cat(rows)
        owners = torch.cat(owners)
        qc = q[0][:, rows, :]
        scores = torch.einsum("hcd,hsd->hcs", qc, k[0]) * scale
        p_exact = torch.softmax(scores, dim=-1)
        del scores, qc
        mass = torch.zeros(heads, rows.numel(), n)
        mass.index_add_(2, p["keyblk"], p_exact)
        del p_exact
        for b in group.tolist():
            sel = (owners == b).nonzero().squeeze(-1)
            m = mass[:, sel, :]                                      # [H, nq, N]
            routed = p["mask"][:, b, :]                              # [H, N]
            forced = p["forced"][b]                                  # [N]
            disc = ~forced
            counts = routed.sum(-1)                                  # [H]
            ideal = topk_mask_by_count(m, counts.view(heads, 1).expand(heads, sel.numel()))
            inter = (ideal & routed.unsqueeze(1)).sum(-1).float()
            jac_all.append(inter / (2 * counts.view(heads, 1).float() - inter).clamp_min(1e-9))
            pair_all.append(_pairwise_jaccard(ideal))
            k_all.append(counts.float().mean())
            # discretionary only: the forced pairs agree by construction
            md = m.masked_fill(forced.view(1, 1, n), -1.0)
            cd = (routed & disc).sum(-1)
            ideald = topk_mask_by_count(md, cd.view(heads, 1).expand(heads, sel.numel())) & disc
            interd = (ideald & (routed & disc).unsqueeze(1)).sum(-1).float()
            jac_disc.append(interd / (2 * cd.view(heads, 1).float() - interd).clamp_min(1e-9))
            pair_disc.append(_pairwise_jaccard(ideald))
            k_disc.append(cd.float().mean())
            n_disc_all.append(float(disc.sum()))
        del mass
    kk = float(torch.stack(k_all).mean())
    kd = float(torch.stack(k_disc).mean())
    nd = sum(n_disc_all) / len(n_disc_all)
    return {
        "blocks": int(pick.numel()), "queries_per_block": int(min(size, n_queries)),
        "query_vs_block": {"all": dist(torch.cat(jac_all, dim=1)),
                           "discretionary": dist(torch.cat(jac_disc, dim=1))},
        "query_vs_query": {"all": dist(torch.cat(pair_all, dim=1)),
                           "discretionary": dist(torch.cat(pair_disc, dim=1))},
        "jaccard_chance_all": round(jaccard_chance(kk, kk, n), 4),
        "jaccard_chance_discretionary": round(jaccard_chance(kd, kd, nd), 4),
        "mean_routed_blocks": round(kk, 1),
        "mean_discretionary_routed_blocks": round(kd, 1),
    }


def _pairwise_jaccard(ideal):
    """[H, pairs] Jaccard between every pair of queries' ideal sets in one block."""
    f = ideal.float()
    inter = f @ f.transpose(1, 2)                                    # [H, nq, nq]
    sizes = f.sum(-1)
    union = sizes.unsqueeze(2) + sizes.unsqueeze(1) - inter
    jac = inter / union.clamp_min(1e-9)
    nq = ideal.shape[1]
    iu = torch.triu_indices(nq, nq, offset=1)
    return jac[:, iu[0], iu[1]]


# ------------------------------------------------------------ E: step to step

def run_step_pairs(paths, args, node, recipe, verbose=True):
    """E: how much of the routed set survives from one denoising step to another."""
    groups = {}
    for path in paths:
        m = re.search(r"_b(\d+)_s(\d+)", Path(path).name)
        if not m:
            raise SystemExit(f"{Path(path).name}: cannot read block and step from the name")
        groups.setdefault((Path(path).resolve().parent.name, int(m.group(1))), []).append(
            (int(m.group(2)), path))
    rows = []
    grid = tuple(int(x) for x in args.grid.split(","))
    for (capset, blk), members in sorted(groups.items()):
        members.sort()
        masks = {}
        for step, path in members:
            q, k, _v = ase.load_capture(path)
            q, k = (x[:, :args.heads].to(torch.float32).contiguous() for x in (q, k))
            tokens = q.shape[2]
            start = args.video_start
            a0, a1 = (int(x) for x in args.audio_span.split(","))
            if start + grid[0] * grid[1] * grid[2] != tokens:
                raise SystemExit(f"{Path(path).name}: grid {grid} does not fit {tokens} rows from {start}")
            layout = {"sol_h3_video_span": (start, tokens), "sol_h3_audio_span": (a0, a1)}
            sink_kv64, sink_q64 = node._sink_blocks(layout, tokens, recipe["sink_conditioning"])
            for ordering in ORDERINGS:
                index = ordering_index(node, ordering, grid, start, tokens)
                for size in BLOCK_SIZES:
                    sc = NODE_BLOCK // size
                    mask, _ = route_mask(q[:, :, index], k[:, :, index], TAU,
                                         (sink_kv64[0] * sc, sink_kv64[1] * sc),
                                         (sink_q64[0] * sc, sink_q64[1] * sc), size)
                    masks[(step, cell_key(ordering, size))] = (
                        mask, video_blocks(start, tokens, size),
                        forced_mask(mask.shape[-1], (sink_kv64[0] * sc, sink_kv64[1] * sc),
                                    (sink_q64[0] * sc, sink_q64[1] * sc)))
            del q, k
            if verbose:
                print(f"  E {capset} block {blk} step {step}: masks built", flush=True)
        steps = sorted({s for s, _ in members})
        for s1, s2 in itertools.combinations(steps, 2):
            for name in (cell_key(o, s) for o in ORDERINGS for s in BLOCK_SIZES):
                m1, vb, forced = masks[(s1, name)]
                m2, _, _ = masks[(s2, name)]
                a = m1[:, vb, :]
                b = m2[:, vb, :]
                inter = (a & b).sum(-1).float()
                union = (a | b).sum(-1).float()
                ka, kb = a.sum(-1).float(), b.sum(-1).float()
                disc = ~forced[vb]                                    # [V, N]
                ad, bd = a & disc.unsqueeze(0), b & disc.unsqueeze(0)
                interd = (ad & bd).sum(-1).float()
                uniond = (ad | bd).sum(-1).float()
                n = m1.shape[-1]
                rows.append({
                    "capture_set": capset, "dit_block": blk, "steps": [s1, s2], "cell": name,
                    "jaccard_all": dist(inter / union.clamp_min(1)),
                    "jaccard_discretionary": dist(interd / uniond.clamp_min(1)),
                    "intersection_over_first": round(float((inter / ka.clamp_min(1)).mean()), 4),
                    "density_first": round(float(ka.mean() / n), 5),
                    "density_second": round(float(kb.mean() / n), 5),
                    "jaccard_chance_all": round(jaccard_chance(float(ka.mean()), float(kb.mean()), n), 4),
                    "jaccard_chance_discretionary": round(
                        jaccard_chance(float(ad.sum(-1).float().mean()), float(bd.sum(-1).float().mean()),
                                       float(disc.sum(-1).float().mean())), 4),
                })
                if verbose:
                    r = rows[-1]
                    print(f"  E {capset} b{blk} s{s1}->s{s2} {name:9s} "
                          f"J {r['jaccard_all']['mean']:.3f} (chance {r['jaccard_chance_all']:.3f}) "
                          f"disc {r['jaccard_discretionary']['mean']:.3f} "
                          f"(chance {r['jaccard_chance_discretionary']:.3f})", flush=True)
        del masks
    return rows


# ------------------------------------------------ F: arms at equal executed cost
#
# COST here is one number and it is the one the kernel pays in MMA work: the
# count of (query token, key token) pairs attended EXACTLY, per head, with the
# forced diagonal and the sinks included. Every arm is tuned until its mean
# cost over the sampled queries equals Sol's at tau 1.0 on that cell, so a row
# that wins has won at Sol's price rather than by spending more.
#
# Arms that keep 64-token key blocks and 64-row query blocks (`max4`, `lse4`,
# the per-query oracle) match cost by taking the same NUMBER of blocks Sol
# took, because equal count is equal cost when the blocks are the same size;
# the code still measures the cost rather than assuming it. Arms that change
# the query granularity (`split2`, `split4`) or add a token stage
# (`token_aug`) are bisected on their own threshold.

SUB_ROWS = {"split2": 32, "split4": 16}
KEY_SUBS = 4                    # sub-means per 64-token key block, SubBlock's n_k
TOKEN_BUDGETS = (64, 128, 256)  # kitchen's allowed `token_aug` budgets, NTOK_MAX = 256
TOKEN_WINDOW_LOG2 = 8.0         # kitchen's histogram floor: ref - 8 log2 units


def route_scores(q, k, size, qsub):
    """Sol's routing score and its tau=1 threshold, at a query granularity of `qsub` rows.

    Returns (colmean [H, NQ, NK], base [H, NQ], kcc [H, NK, D], k_mean [H, 1, D],
    lengths [NK]), where a pair is routed exactly when `colmean > tau * base`
    or forced. The threshold is linear in tau, so one call serves a whole
    bisection.

    `qsub` is the QUERY granularity only: the key side stays at `size`, which
    is what every arm here varies independently. At `qsub == size` this is
    `route_mask`'s arithmetic with the mask left unformed.
    """
    _b, h, t, d = q.shape
    nk = (t + size - 1) // size
    nq = (t + qsub - 1) // qsub
    log2s = (d ** -0.5) * ase._LOG2E
    lengths = block_lengths(t, size)
    qlen = block_lengths(t, qsub)
    colmean = torch.empty((h, nq, nk), dtype=torch.float32)
    base = torch.empty((h, nq), dtype=torch.float32)
    kcc = torch.empty((h, nk, d), dtype=torch.float32)
    k_mean = torch.empty((h, 1, d), dtype=torch.float32)
    padk, padq = nk * size - t, nq * qsub - t
    for head in range(h):
        fq, fk = q[0, head], k[0, head]
        pq = F.pad(fq, (0, 0, 0, padq)) if padq else fq
        pk = F.pad(fk, (0, 0, 0, padk)) if padk else fk
        centroid = pq.view(nq, qsub, d).sum(1) / qlen.view(nq, 1)
        kc = pk.view(nk, size, d).sum(1) / lengths.view(nk, 1)
        km = kc.mean(0, keepdim=True)
        kc_c = kc - km
        colmean[head] = (centroid @ kc_c.T) * log2s
        base[head] = torch.sqrt((centroid.pow(2) * kc_c.pow(2).mean(0)).sum(-1) * log2s * log2s + 1e-6)
        kcc[head], k_mean[head] = kc_c, km
    return colmean, base, kcc, k_mean, lengths


def key_submean_scores(q, k, size, qsub, nsub, reduce, k_mean):
    """[H, NQ, NK] score of each key block from `nsub` sub-means instead of one mean.

    `max` is MiniMax-M3's indexer objective and `lse` is sglang's SubBlock
    form. Both are route-stage only: the exact stage, the pooled tail and the
    tile layout are untouched, so an arm that wins here is a change to one
    kernel's scoring and nothing else.

    The sub-means are centred by the SAME per-channel mean the block score
    uses, so the two rankings differ only in the reduction.
    """
    _b, h, t, d = q.shape
    nk = (t + size - 1) // size
    nq = (t + qsub - 1) // qsub
    step = size // nsub
    log2s = (d ** -0.5) * ase._LOG2E
    lengths = block_lengths(t, size)
    qlen = block_lengths(t, qsub)
    out = torch.empty((h, nq, nk), dtype=torch.float32)
    padk, padq = nk * size - t, nq * qsub - t
    for head in range(h):
        fq, fk = q[0, head], k[0, head]
        pq = F.pad(fq, (0, 0, 0, padq)) if padq else fq
        # the ragged tail is padded with the block mean, so a short final block
        # keeps the mean it would have had rather than being pulled toward zero
        pk = fk
        if padk:
            tail = fk[(nk - 1) * size:].mean(0, keepdim=True).expand(padk, d)
            pk = torch.cat([fk, tail])
        centroid = pq.view(nq, qsub, d).sum(1) / qlen.view(nq, 1)
        sub = pk.view(nk, nsub, step, d).mean(2) - k_mean[head].view(1, 1, d)   # [NK, nsub, D]
        scores = torch.einsum("qd,nsd->qns", centroid, sub) * log2s             # [NQ, NK, nsub]
        out[head] = scores.amax(-1) if reduce == "max" else torch.logsumexp(scores, dim=-1)
        del scores, sub
    _ = lengths
    return out


def forced_for(nq, nk, size, qsub, sink_kv, sink_q):
    """[NQ, NK] forced pairs when the query granularity is `qsub` rows.

    The diagonal is a property of the 64-token KEY grid, so a 16-row query
    sub-block forces the same three key blocks its parent block does. Sinks
    are unchanged.
    """
    per = max(1, size // qsub)
    parent = torch.arange(nq) // per
    idx = torch.arange(nk)
    forced = (parent.view(-1, 1) - idx.view(1, -1)).abs() <= 1
    forced |= ((idx >= sink_kv[0]) & (idx < sink_kv[1])).view(1, nk)
    forced |= ((parent >= sink_q[0]) & (parent < sink_q[1])).view(nq, 1)
    return forced


def _cost_of(routed, lengths):
    """Mean exact (query token, key token) pairs per query row, over [.., NK] routed rows."""
    return float((routed.float() * lengths).sum(-1).mean())


def bisect_tau(score, base, forced, rows, lengths, target, lo=0.02, hi=60.0, steps=40):
    """The tau whose mean exact cost over `rows` matches `target`. Cost falls as tau rises."""
    def cost(tau):
        routed = (score > tau * base.unsqueeze(-1)) | forced
        return _cost_of(routed[:, rows, :], lengths)
    if cost(hi) > target:
        return hi, cost(hi)
    if cost(lo) < target:
        return lo, cost(lo)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if cost(mid) > target:
            lo = mid
        else:
            hi = mid
    tau = 0.5 * (lo + hi)
    return tau, cost(tau)


def union_rows(routed, size, qsub, nq64):
    """[H, NQ64, NK]: the union of a 64-row block's sub-block sets, which is what a CTA walks."""
    per = size // qsub
    h, nq, nk = routed.shape
    pad = nq64 * per - nq
    if pad:
        routed = torch.cat([routed, torch.zeros(h, pad, nk, dtype=torch.bool)], dim=1)
    return routed.view(h, nq64, per, nk).any(2)


def token_aug_group(g, qs, ks, colmean, k_mean, mask_block, budget, size, tok_group, pool, n):
    """Kitchen's `token_aug` selection for ONE query group, emulated from the kernel's geometry.

    Geometry read from the kernel rather than assumed:
      - one centroid per `TOK_GROUP` neighbouring query blocks, the mean of
        their rows (`sol_attn_token.cu::sol_token_group_kernel`; `TOK_GROUP` is
        2 in `sol_layout.cuh`, so 128 query rows);
      - the candidates are the tokens of the blocks NONE of the group's members
        routed, the AND of their unrouted bitmaps (same kernel);
      - one token list and one budget per GROUP, `n_tok` of 64, 128 or 256
        (`NTOK_MAX`), and the group index is what `tok_cnt` and `tok_idx` are
        keyed by;
      - a token more than 8 log2 units below the group's best unrouted pooled
        score is outside the histogram window and can never be admitted;
      - the admitted tokens become exact for EVERY row of the group, and the
        candidates that miss the cut do not vanish: they become a per-token
        softmax state at the centroid's own scores, which REPLACES the route's
        block-pooled tail over the candidate blocks.

    Two deliberate departures, both in the record: the kernel admits whole
    histogram bins, so it takes at most the budget where this takes exactly
    it; and the kernel scores at INT8 where this is fp32. Both make this the
    optimistic end of what the kernel would select.

    `pool` > 0 applies HISA's cut: the token pass sees only the top-`pool`
    unrouted blocks by the route's own block score.
    """
    _b, heads, s_len, d = qs.shape
    log2s = (d ** -0.5) * ase._LOG2E
    qb0, qb1 = g * tok_group, min(colmean.shape[1], g * tok_group + tok_group)
    r0, r1 = qb0 * size, min(s_len, qb1 * size)
    cen = qs[0][:, r0:r1, :].mean(1)                                       # [H, D]
    cand = ~mask_block[:, qb0:qb1, :].any(1)                               # [H, N] unrouted by every member
    grp_score = colmean[:, qb0:qb1, :].amax(1)                             # the group's pooled scores
    if pool:
        keepn = torch.full((heads,), min(pool, n))
        cand = cand & topk_mask_by_count(grp_score.masked_fill(~cand, -1e30), keepn)
    ref = torch.where(cand, grp_score, torch.full_like(grp_score, -1e30)).amax(-1)
    s_tok = torch.einsum("hd,hsd->hs", cen, ks[0] - k_mean) * log2s        # [H, S]
    keyblk = torch.arange(s_len) // size
    ok = cand.gather(1, keyblk.view(1, -1).expand(heads, -1))
    ok &= s_tok >= (ref.unsqueeze(-1) - TOKEN_WINDOW_LOG2)
    take = max(1, min(budget, int(ok.sum(-1).max())))
    idx = torch.topk(s_tok.masked_fill(~ok, -1e30), take, dim=-1).indices   # [H, take]
    keep = torch.gather(ok, 1, idx)
    return idx, keep, cand, s_tok


def run_arms(path, args, node, recipe, verbose=True):
    """F for one capture: every arm tuned to Sol's executed cost, then missed mass AND output error."""
    t0 = time.time()
    q, k, v = ase.load_capture(path)
    q, k, v = (x[:, :args.heads].to(torch.float32).contiguous() for x in (q, k, v))
    tokens = q.shape[2]
    grid = tuple(int(x) for x in args.grid.split(","))
    start, size = args.video_start, NODE_BLOCK
    a0, a1 = (int(x) for x in args.audio_span.split(","))
    if start + grid[0] * grid[1] * grid[2] != tokens:
        raise SystemExit(f"{Path(path).name}: grid {grid} does not fit {tokens} rows from {start}")
    d = q.shape[-1]
    scale = d ** -0.5
    log2s = scale * ase._LOG2E
    layout = {"sol_h3_video_span": (start, tokens), "sol_h3_audio_span": (a0, a1)}
    sink_kv, sink_q = node._sink_blocks(layout, tokens, recipe["sink_conditioning"])
    sample = stratified_video_queries(grid, start, args.per_frame, args.seed)
    heads, n = q.shape[1], (tokens + size - 1) // size
    cell = {"tokens": int(tokens), "heads": heads, "sample_queries": int(sample.numel()),
            "token_group_query_blocks": args.tok_group, "token_budget": args.token_budget,
            "hisa_pool_blocks": args.hisa_pool, "key_submeans": KEY_SUBS}
    for ordering in ORDERINGS:
        index = ordering_index(node, ordering, grid, start, tokens)
        pos = torch.argsort(index)
        qs, ks, vs = (x[:, :, index] for x in (q, k, v))
        # from here on every index is a POSITION in this ordering, never a raster
        # token id: q, k and v are permuted, so a key column is a position and a
        # key block is `position // size`.
        keyblk = torch.arange(tokens) // size
        qpos = pos[sample]
        colmean, base, kcc, k_mean, lengths = route_scores(qs, ks, size, size)
        forced = forced_for(n, n, size, size, sink_kv, sink_q)
        sol_mask = (colmean > TAU * base.unsqueeze(-1)) | forced
        qblk = qpos // size
        target = _cost_of(sol_mask[:, qblk, :], lengths)
        counts = sol_mask[:, qblk, :].sum(-1)
        vc = torch.stack([F.pad(vs[0, hh], (0, 0, 0, n * size - tokens)).view(n, size, d).sum(1)
                          for hh in range(heads)])                               # [H, N, D] summed values
        if verbose:
            print(f"  F {ordering:6s} Sol cost {target:.0f} exact key tokens per query row", flush=True)

        arms = {}
        arms["sol"] = {"routed": sol_mask[:, qblk, :], "tail_logit": colmean[:, qblk, :],
                       "setting": {"tau": TAU}}
        for name, reduce in (("max4", "max"), ("lse4", "lse")):
            sc = key_submean_scores(qs, ks, size, size, KEY_SUBS, reduce, k_mean)
            big = torch.finfo(torch.float32).max
            routed = topk_mask_by_count(sc[:, qblk, :].masked_fill(forced[qblk].unsqueeze(0), big), counts)
            arms[name] = {"routed": routed, "tail_logit": colmean[:, qblk, :],
                          "setting": {"rule": f"{reduce} over {KEY_SUBS} key sub-means, Sol's own block count"}}
            del sc
        for name, sub in SUB_ROWS.items():
            cm_s, base_s, _kcc, _km, _len = route_scores(qs, ks, size, sub)
            f_s = forced_for(cm_s.shape[1], n, size, sub, sink_kv, sink_q)
            rows_s = qpos // sub
            tau_i, cost_i = bisect_tau(cm_s, base_s, f_s, rows_s, lengths, target)
            routed_i = ((cm_s > tau_i * base_s.unsqueeze(-1)) | f_s)
            uni = union_rows(routed_i, size, sub, n)
            infl = _cost_of(uni[:, qblk, :], lengths) / max(cost_i, 1e-9)
            # and again, tuned so the UNION -- what a CTA actually walks -- is Sol's cost
            def union_cost(tau, cm_s=cm_s, base_s=base_s, f_s=f_s, sub=sub):
                r = (cm_s > tau * base_s.unsqueeze(-1)) | f_s
                return _cost_of(union_rows(r, size, sub, n)[:, qblk, :], lengths)
            lo, hi = 0.02, 60.0
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if union_cost(mid) > target:
                    lo = mid
                else:
                    hi = mid
            tau_u = 0.5 * (lo + hi)
            routed_u = ((cm_s > tau_u * base_s.unsqueeze(-1)) | f_s)
            arms[name] = {"routed": routed_i[:, rows_s, :], "tail_logit": cm_s[:, rows_s, :],
                          "setting": {"tau_ideal_cost": round(tau_i, 4), "sub_rows": sub,
                                      "union_inflation": round(float(infl), 4),
                                      "walk_cost_over_sol": round(float(infl), 4),
                                      "tau_union_cost": round(tau_u, 4)}}
            arms[name + "_union"] = {"routed": routed_u[:, rows_s, :], "tail_logit": cm_s[:, rows_s, :],
                                     "setting": {"tau": round(tau_u, 4), "sub_rows": sub,
                                                 "walk_cost_over_sol": 1.0,
                                                 "note": "tuned so the CTA's UNION walk costs what Sol costs; "
                                                         "its own cost row is the MMA work, which is lower"}}
            if name == "split4":
                sc = key_submean_scores(qs, ks, size, sub, KEY_SUBS, "lse", k_mean)
                cnt_s = routed_i[:, rows_s, :].sum(-1)
                big = torch.finfo(torch.float32).max
                r2 = topk_mask_by_count(sc[:, rows_s, :].masked_fill(f_s[rows_s].unsqueeze(0), big), cnt_s)
                arms["split4_lse4"] = {"routed": r2, "tail_logit": cm_s[:, rows_s, :],
                                       "setting": {"sub_rows": sub, "rule": "split4's count, lse4's ranking"}}
                del sc
            del cm_s, base_s, f_s, routed_i, routed_u, uni
        cell.setdefault("arms_setting", {})[ordering] = {a: arms[a]["setting"] for a in arms}

        # token_aug: the block stage is loosened until block cost plus the token
        # budget lands on Sol's, so the tokens are paid for out of the blocks
        tok = {}
        for name, pool in (("token_aug", 0), ("token_aug_hisa", args.hisa_pool)):
            def tcost(tau, pool=pool):
                mask_b = (colmean > tau * base.unsqueeze(-1)) | forced
                blk = _cost_of(mask_b[:, qblk, :], lengths)
                return blk + args.token_budget, mask_b
            lo, hi = TAU, 60.0
            if tcost(lo)[0] > target:
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    if tcost(mid)[0] > target:
                        lo = mid
                    else:
                        hi = mid
            tau_b = 0.5 * (lo + hi)
            mask_b = (colmean > tau_b * base.unsqueeze(-1)) | forced
            tok[name] = {"tau_block": round(tau_b, 4), "mask": mask_b, "pool": pool,
                         "setting": {"tau_block": round(tau_b, 4), "budget": args.token_budget,
                                     "tok_group_query_blocks": args.tok_group,
                                     "hisa_pool_blocks": pool or None}}
            cell["arms_setting"][ordering][name] = tok[name]["setting"]

        # ---- one chunked pass: exact row, exact output, then every arm
        acc = {a: {"missed": [], "cost": [], "err2": 0.0} for a in
               list(arms) + list(tok) + ["oracle"]}
        ref2 = 0.0
        # Missed mass is a masked sum and costs nothing; the emulated OUTPUT is
        # two S-wide matmuls per arm, so it runs on every `err_stride`-th
        # sampled query. The subset is strided, not a prefix, so it still
        # covers every latent frame.
        recov = {"n50": [], "n90": [], "reached50": 0, "reached90": 0, "rows": 0, "admissible": [],
                 "o50": [], "o90": [], "oreached50": 0, "oreached90": 0}
        for s0 in range(0, sample.numel(), args.chunk):
            sl = slice(s0, s0 + args.chunk)
            qi, qb = qpos[sl], qblk[sl]
            qc = qs[0][:, qi, :]
            s_raw = torch.einsum("hcd,hsd->hcs", qc, ks[0]) * scale
            p_exact = torch.softmax(s_raw, dim=-1)
            out_exact = p_exact @ vs[0]
            ref2 += float(out_exact[:, torch.arange(s0, s0 + qi.numel()) % args.err_stride == 0, :]
                          .pow(2).sum())
            # centred log2 logits: the constant shift cancels in the softmax but
            # must match the pooled logits, which are built from centred keys
            s_log2 = (s_raw - torch.einsum("hcd,hd->hc", qc, k_mean[:, 0, :]).unsqueeze(-1) * scale) * ase._LOG2E
            del s_raw, qc

            esub = torch.arange(s0, s0 + qi.numel()) % args.err_stride == 0
            def score_arm(name, exact_tok, pooled_blk, tail_logit, extra=None):
                missed = (p_exact * ~exact_tok).sum(-1)
                acc[name]["missed"].append(missed)
                acc[name]["cost"].append(exact_tok.float().sum(-1))
                if not bool(esub.any()):
                    return
                ex = (extra[0][:, esub, :], extra[1][:, esub, :]) if extra else (None, None)
                out = sol_style_output(s_log2[:, esub, :], vs[0], vc, lengths, exact_tok[:, esub, :],
                                       pooled_blk[:, esub, :], tail_logit[:, esub, :], *ex)
                acc[name]["err2"] += float((out - out_exact[:, esub, :]).pow(2).sum())
                del out

            missed_sol = None
            for name, arm in arms.items():
                routed = arm["routed"][:, sl, :]
                et = routed.gather(2, keyblk.view(1, 1, -1).expand(heads, qi.numel(), tokens))
                score_arm(name, et, ~routed, arm["tail_logit"][:, sl, :])
                if name == "sol":
                    missed_sol = acc["sol"]["missed"][-1]
                del et, routed
            for name, ta in tok.items():
                routed = ta["mask"][:, qb, :]
                et = routed.gather(2, keyblk.view(1, 1, -1).expand(heads, qi.numel(), tokens))
                cand = torch.zeros(heads, qi.numel(), n, dtype=torch.bool)
                extra_l = torch.full((heads, qi.numel(), tokens), -1e30)
                extra_m = torch.zeros(heads, qi.numel(), tokens, dtype=torch.bool)
                for j, g in enumerate((qb // args.tok_group).tolist()):
                    hit = ta.setdefault("cache", {}).get(g)
                    if hit is None:
                        hit = token_aug_group(g, qs, ks, colmean, k_mean, ta["mask"],
                                              args.token_budget, size, args.tok_group, ta["pool"], n)
                        ta["cache"] = {g: hit}     # one chunk's groups are contiguous
                    idx, keep, cnd, s_tok = hit
                    cand[:, j, :] = cnd
                    extra_l[:, j, :] = s_tok
                    extra_m[:, j, :] = cnd.gather(1, keyblk.view(1, -1).expand(heads, -1))
                    et[:, j, :] |= torch.zeros(heads, tokens, dtype=torch.bool).scatter_(1, idx, keep)
                extra_m &= ~et                      # admitted tokens left the tail
                score_arm(name, et, (~routed) & ~cand, colmean[:, qb, :], (extra_l, extra_m))
                del et, routed, cand, extra_l, extra_m
            mass = torch.zeros(heads, qi.numel(), n)
            mass.index_add_(2, keyblk, p_exact)
            om = topk_mask_by_count(mass.masked_fill(forced[qb].unsqueeze(0), 2.0), counts[:, sl])
            et = om.gather(2, keyblk.view(1, 1, -1).expand(heads, qi.numel(), tokens))
            score_arm("oracle", et, ~om, colmean[:, qb, :])
            del mass, om, et

            # How many individual key tokens buy back Sol's missed mass, at Sol's
            # OWN block routing. This is the added-cost question, not the
            # equal-cost one: it prices the token stage rather than comparing it.
            for row in esub.nonzero().squeeze(-1).tolist():
                gg = int(qb[row]) // args.tok_group
                idx, keep, cnd, _st = token_aug_group(gg, qs, ks, colmean, k_mean, sol_mask,
                                                      args.recover_cap, size, args.tok_group, 0, n)
                tgt = missed_sol[:, row].unsqueeze(-1)
                recov["admissible"].append(keep.float().sum(-1))

                def curve(order_mask, order_vals, store50, store90, reach):
                    got = torch.gather(p_exact[:, row, :], 1, order_vals) * order_mask
                    cum = got.cumsum(-1)
                    for frac, key, tag in ((0.5, store50, "50"), (0.9, store90, "90")):
                        hit = cum >= frac * tgt
                        any_hit = hit.any(-1)
                        first = torch.where(any_hit, hit.float().argmax(-1) + 1,
                                            torch.full_like(any_hit, args.recover_cap, dtype=torch.long))
                        recov[key].append(first.float())
                        recov[reach + tag] += int(any_hit.sum())

                curve(keep, idx, "n50", "n90", "reached")
                # the same candidate tokens ranked by the QUERY's own exact mass:
                # the ceiling a per-query scorer could reach over the same scan,
                # which separates the budget from the 128-row centroid
                cand_tok = cnd.gather(1, keyblk.view(1, -1).expand(heads, -1))
                oidx = torch.topk(p_exact[:, row, :].masked_fill(~cand_tok, -1.0),
                                  idx.shape[1], dim=-1).indices
                curve(torch.gather(cand_tok, 1, oidx), oidx, "o50", "o90", "oreached")
                recov["rows"] += int(keep.shape[0])
                del idx, keep, cnd, cand_tok, oidx
            del p_exact, out_exact, s_log2
            if verbose:
                print(f"  F {ordering} {min(s0 + args.chunk, sample.numel())}/{sample.numel()}",
                      end="\r", flush=True)

        rows = {}
        for name in acc:
            miss = torch.cat(acc[name]["missed"], dim=1)
            cost = torch.cat(acc[name]["cost"], dim=1)
            rows[name] = {"missed_mass": dist(miss), "cost_mean": round(float(cost.mean()), 1),
                          "cost_over_sol": round(float(cost.mean()) / target, 4),
                          "rel_l2_vs_exact": round(math.sqrt(acc[name]["err2"] / max(ref2, 1e-30)), 6)}
        if recov["n50"]:
            n50 = torch.cat(recov["n50"]); n90 = torch.cat(recov["n90"])
            o50 = torch.cat(recov["o50"]); o90 = torch.cat(recov["o90"])
            rows["_token_recovery"] = {
                "what": "individual key tokens from the blocks Sol left unrouted, needed to recover "
                        "half and ninety percent of Sol's missed mass at Sol's own tau. `centroid` "
                        "ranks them the way kitchen's token stage does, by the score of the centroid "
                        "shared by TOK_GROUP query blocks; `per_query` ranks the SAME candidates by "
                        "the query's own exact mass, which is the ceiling a finer scorer could reach "
                        "over the same scan. The gap between them is the price of the shared centroid, "
                        "not of the budget.",
                "cap": args.recover_cap,
                "centroid": {"tokens_for_half": dist(n50), "tokens_for_ninety": dist(n90),
                             "share_reaching_half": round(recov["reached50"] / max(recov["rows"], 1), 4),
                             "share_reaching_ninety": round(recov["reached90"] / max(recov["rows"], 1), 4)},
                "per_query": {"tokens_for_half": dist(o50), "tokens_for_ninety": dist(o90),
                              "share_reaching_half": round(recov["oreached50"] / max(recov["rows"], 1), 4),
                              "share_reaching_ninety": round(recov["oreached90"] / max(recov["rows"], 1), 4)},
                "admissible_tokens_in_window": dist(torch.cat(recov["admissible"])),
                "token_budget_the_kernel_offers": list(TOKEN_BUDGETS)}
        cell.setdefault("arms", {})[ordering] = rows
        if verbose:
            for name, r in rows.items():
                if name.startswith("_"):
                    print(f"  F {ordering:6s} token recovery, centroid rank: half at "
                          f"{r['centroid']['tokens_for_half']['median']:.0f}, ninety at "
                          f"{r['centroid']['tokens_for_ninety']['median']:.0f} "
                          f"(share reaching ninety {r['centroid']['share_reaching_ninety']:.2f}); "
                          f"per-query rank: half at {r['per_query']['tokens_for_half']['median']:.0f}, "
                          f"ninety at {r['per_query']['tokens_for_ninety']['median']:.0f} "
                          f"(share {r['per_query']['share_reaching_ninety']:.2f}); admissible "
                          f"{r['admissible_tokens_in_window']['median']:.0f}", flush=True)
                    continue
                print(f"  F {ordering:6s} {name:16s} cost {r['cost_over_sol']:.3f}x  "
                      f"missed {r['missed_mass']['mean']:.4f}  rel_l2 {r['rel_l2_vs_exact']:.4f}", flush=True)
        del arms, tok, acc, qs, ks, vs, colmean, base, kcc, vc, sol_mask
    cell["seconds"] = round(time.time() - t0, 1)
    return cell


def sol_style_output(p_scores, v_head, vc, lengths, exact_tok, pooled_blk, pooled_logit,
                     extra_tok_logit=None, extra_tok_mask=None):
    """Sol's own arithmetic for a set of query rows under ANY route, so an arm has an output error.

    `p_scores` are the CENTRED token logits in log2 units, [H, C, S]; `exact_tok`
    marks the keys attended exactly; `pooled_blk` marks the key blocks that
    contribute a block-pooled term at `pooled_logit`; `extra_tok_*` carry
    `token_aug`'s per-token tail, which replaces the block term over the
    candidate blocks. One softmax over every branch, weights by block length
    for the pooled terms, exactly as `_sol_attn_reference.sol_attn` does.
    """
    neg = -1e30
    s_exact = p_scores.masked_fill(~exact_tok, neg)
    s_pool = pooled_logit.masked_fill(~pooled_blk, neg)
    parts = [s_exact, s_pool]
    if extra_tok_logit is not None:
        parts.append(extra_tok_logit.masked_fill(~extra_tok_mask, neg))
    m = torch.stack([p.amax(-1) for p in parts], dim=-1).amax(-1, keepdim=True)
    e_exact = torch.exp2(s_exact - m).masked_fill(s_exact <= neg, 0.0)
    e_pool = torch.exp2(s_pool - m).masked_fill(s_pool <= neg, 0.0)
    num = e_exact @ v_head + e_pool @ vc
    den = e_exact.sum(-1) + (e_pool * lengths).sum(-1)
    if extra_tok_logit is not None:
        s_ex = parts[2]
        e_ex = torch.exp2(s_ex - m).masked_fill(s_ex <= neg, 0.0)
        num = num + e_ex @ v_head
        den = den + e_ex.sum(-1)
        del e_ex
    return num / den.clamp_min(1e-30).unsqueeze(-1)


# --------------------------- H: does Sol under-route the one-pixel-frame latents?

def run_frames(path, args, node, recipe, verbose=True):
    """H: where Sol's missed mass sits by latent-frame residue, with the residue control.

    `bench/map_attention_mass_on_capture.py` found that the latent frames whose
    index is 0 mod 5 -- the ones whose RoPE time span is ONE pixel frame where
    the rest span four (`comfy/ldm/minimax/model.py`, `FRAME_PER_TOKEN`) -- draw
    more exact attention mass than their share of keys on every captured DiT
    block but block 0. That says where the mass is; it does not say whether
    Sol's routing keeps up with it.

    This asks the second question. For the shipped `sol` arm at tau 1.0 it
    splits the sampled queries' exact mass, and Sol's MISSED mass, by the key's
    residue class, and reports each class's share of missed mass against its
    share of keys and against its share of true mass. Sol under-routes a class
    when its share of the missed mass exceeds its share of the true mass.

    **The residue classes 1 to 4 are the control**, exactly as the mass record
    uses them: an effect of the one-pixel-frame latents lifts residue 0 alone,
    while a binning or scene artifact lifts any of them. The frame index is
    `(key - video_start) // frame_tokens`, the same arithmetic that record uses,
    so the two agree by construction rather than by coincidence.

    Plain token order only, which is the order the captures were taken in.
    """
    t0 = time.time()
    q, k, _v = ase.load_capture(path)
    q, k = (x[:, :args.heads].to(torch.float32).contiguous() for x in (q, k))
    tokens = q.shape[2]
    grid = tuple(int(x) for x in args.grid.split(","))
    start, size = args.video_start, NODE_BLOCK
    a0, a1 = (int(x) for x in args.audio_span.split(","))
    area = grid[1] * grid[2]
    if start + grid[0] * area != tokens:
        raise SystemExit(f"{Path(path).name}: grid {grid} does not fit {tokens} rows from {start}")
    scale = q.shape[-1] ** -0.5
    layout = {"sol_h3_video_span": (start, tokens), "sol_h3_audio_span": (a0, a1)}
    sink_kv, sink_q = node._sink_blocks(layout, tokens, recipe["sink_conditioning"])
    mask, _cen = route_mask(q, k, TAU, sink_kv, sink_q, size)
    heads, n = q.shape[1], (tokens + size - 1) // size
    keyblk = torch.arange(tokens) // size
    # class 0..4 are the video residues, class 5 is every conditioning key
    cls = torch.full((tokens,), 5, dtype=torch.long)
    frame = (torch.arange(tokens) - start) // area
    cls[start:] = frame[start:] % 5
    sample = stratified_video_queries(grid, start, args.per_frame, args.seed)
    missed = torch.zeros(6, dtype=torch.float64)
    total = torch.zeros(6, dtype=torch.float64)
    rows = 0
    for s0 in range(0, sample.numel(), args.chunk):
        qi = sample[s0:s0 + args.chunk]
        qb = qi // size
        p = torch.softmax(torch.einsum("hcd,hsd->hcs", q[0][:, qi, :], k[0]) * scale, dim=-1)
        routed = mask[:, qb, :]
        et = routed.gather(2, keyblk.view(1, 1, -1).expand(heads, qi.numel(), tokens))
        miss = p * ~et
        total += torch.zeros(heads, qi.numel(), 6).index_add_(2, cls, p).sum((0, 1)).double()
        missed += torch.zeros(heads, qi.numel(), 6).index_add_(2, cls, miss).sum((0, 1)).double()
        rows += heads * qi.numel()
        del p, routed, et, miss
    keys = torch.zeros(6).index_add_(0, cls, torch.ones(tokens))
    share_keys = (keys / tokens).tolist()
    share_mass = (total / total.sum()).tolist()
    share_missed = (missed / missed.sum().clamp_min(1e-30)).tolist()
    labels = [f"residue_{j}" for j in range(5)] + ["conditioning"]
    out = {"ordering": "raster", "tau": TAU, "sampled_queries": int(sample.numel()),
           "heads": heads, "video_frames": grid[0], "frame_tokens": area,
           "missed_mass_share_of_total": round(float(missed.sum() / total.sum()), 6),
           "classes": {}}
    for j, name in enumerate(labels):
        out["classes"][name] = {
            "share_of_keys": round(share_keys[j], 5),
            "share_of_true_mass": round(share_mass[j], 5),
            "share_of_missed_mass": round(share_missed[j], 5),
            "true_mass_over_keys": round(share_mass[j] / max(share_keys[j], 1e-12), 4),
            "missed_over_true_mass": round(share_missed[j] / max(share_mass[j], 1e-12), 4),
            "missed_over_keys": round(share_missed[j] / max(share_keys[j], 1e-12), 4)}
    out["seconds"] = round(time.time() - t0, 1)
    if verbose:
        for name, c in out["classes"].items():
            print(f"  H {name:13s} keys {c['share_of_keys']:.4f} mass {c['share_of_true_mass']:.4f} "
                  f"({c['true_mass_over_keys']:.3f}x keys)  missed {c['share_of_missed_mass']:.4f} "
                  f"({c['missed_over_true_mass']:.3f}x its own mass)", flush=True)
    return out


# ------------------------------------------- G: LoSA's frozen-pattern recall

def block_masses(q, k, blocks, size, device, chunk):
    """[H, len(blocks), NK] exact softmax mass per (query block, key block), fp32.

    `M(i, j)` is the mean over the query block's rows of that row's exact
    attention mass landing in key block j, so every row of `M` sums to 1. The
    score matrix is never materialised beyond one chunk of query rows.
    """
    _b, h, t, d = q.shape
    nk = (t + size - 1) // size
    scale = d ** -0.5
    keyblk = (torch.arange(t, device=device) // size)
    out = torch.zeros(h, blocks.numel(), nk, device=device)
    kh = k[0].to(device)
    for bi, b in enumerate(blocks.tolist()):
        r0, r1 = b * size, min(t, (b + 1) * size)
        acc = torch.zeros(h, nk, device=device)
        for c0 in range(r0, r1, chunk):
            qc = q[0][:, c0:min(c0 + chunk, r1), :].to(device)
            p = torch.softmax(torch.einsum("hcd,hsd->hcs", qc, kh) * scale, dim=-1)
            acc += torch.zeros(h, p.shape[1], nk, device=device).index_add_(2, keyblk, p).sum(1)
            del qc, p
        out[:, bi, :] = acc / float(r1 - r0)
    del kh
    return out


def prefix_sets(m, theta):
    """[H, B, NK] bool: per row, the shortest set of key blocks whose mass reaches `theta`."""
    order = torch.argsort(m, dim=-1, descending=True)
    cum = torch.gather(m, -1, order).cumsum(-1)
    rank = (cum < theta).sum(-1, keepdim=True)
    keep = torch.arange(m.shape[-1]).view(1, 1, -1) <= rank
    return torch.zeros_like(m, dtype=torch.bool).scatter_(-1, order, keep)


def diagonal_floor(counts, blocks, nk, sink_kv):
    """[H, B, NK] bool: the structure-matched floor, sinks then a widening diagonal band.

    A uniform random set is not the control: attention's diagonal clears it
    trivially, so the floor has to carry the structure a routed set gets free.
    """
    idx = torch.arange(nk)
    pri = torch.where(((idx >= sink_kv[0]) & (idx < sink_kv[1])).view(1, -1),
                      torch.zeros(1, nk), 1.0 + (idx.view(1, -1) - blocks.view(-1, 1)).abs().float())
    order = torch.argsort(pri, dim=-1)
    h = counts.shape[0]
    rank = torch.arange(nk).view(1, 1, -1).expand(h, blocks.numel(), nk)
    keep = rank < counts.unsqueeze(-1)
    return torch.zeros(h, blocks.numel(), nk, dtype=torch.bool).scatter_(
        -1, order.unsqueeze(0).expand(h, -1, -1), keep)


def run_losa(paths, args, node, recipe, verbose=True):
    """G: LoSA's premise, asked of H3 captures. Build a key-block set once, measure it later.

    LoSA (arXiv 2608.12032) builds a per-(head, query block) set of key blocks
    at an early denoising step, freezes it, and reports that its recall of the
    exact attention mass holds for the rest of the schedule (its Fig. 2(b): a
    mean near 99%, 98.7% at the final step). This asks the same of H3's
    captures, with the comparators the premise needs to be worth anything:

      - `sol_at_t`     Sol's own routed set REBUILT at step t. If Sol matches
                       or beats the frozen set, freezing buys nothing here
                       whatever its recall is. This arm carries the decision.
      - `oracle_at_t`  the top-|Omega| blocks by the step's OWN mass. Oracle
                       minus frozen is the drift cost, separated from the rule.
      - `floor`        sinks plus a widening diagonal band at the same
                       cardinality: the structure a routed set gets for free.

    Two frozen sets: `theta`, the shortest prefix reaching the mass target, and
    `matched`, the top-k by mass with k equal to Sol's own routed count at the
    build step, so one arm answers the recall question and the other answers
    it at Sol's price.

    **Exact block masses need a full softmax row per query row**, so every
    query block of every head is a GPU job. `--query-blocks` samples them with
    a fixed seed and `--device` moves the work, which makes the shape of the
    answer readable on CPU while the full run waits for the card. A record
    built from a sample must say so.
    """
    size = NODE_BLOCK
    device = torch.device(args.device)
    grid = tuple(int(x) for x in args.grid.split(","))
    groups = {}
    for path in paths:
        m = re.search(r"_b(\d+)_s(\d+)", Path(path).name)
        if not m:
            raise SystemExit(f"{Path(path).name}: cannot read block and step from the name")
        groups.setdefault((Path(path).resolve().parent.name, int(m.group(1))), []).append(
            (int(m.group(2)), path))
    rows = []
    for (capset, blk), members in sorted(groups.items()):
        members.sort()
        masses, routed, meta = {}, {}, {}
        for step, path in members:
            q, k, _v = ase.load_capture(path)
            q, k = (x[:, :args.heads].to(torch.float32).contiguous() for x in (q, k))
            tokens = q.shape[2]
            start = args.video_start
            a0, a1 = (int(x) for x in args.audio_span.split(","))
            layout = {"sol_h3_video_span": (start, tokens), "sol_h3_audio_span": (a0, a1)}
            sink_kv, sink_q = node._sink_blocks(layout, tokens, recipe["sink_conditioning"])
            n = (tokens + size - 1) // size
            index = ordering_index(node, args.losa_ordering, grid, start, tokens)
            qs, ks = q[:, :, index], k[:, :, index]
            vb = video_blocks(start, tokens, size)
            gen = torch.Generator().manual_seed(args.seed + 2)
            pick = vb[torch.randperm(vb.numel(), generator=gen)[:args.query_blocks].sort().values]
            masses[step] = block_masses(qs, ks, pick, size, device, args.chunk).cpu()
            routed[step] = route_mask(qs, ks, TAU, sink_kv, sink_q, size)[0][:, pick, :]
            meta = {"pick": pick, "n": n, "sink_kv": sink_kv}
            del q, k, qs, ks
            if verbose:
                print(f"  G {capset} block {blk} step {step}: masses for {pick.numel()} "
                      f"query blocks", flush=True)
        steps = sorted(masses)
        t0 = steps[0]
        pick, n, sink_kv = meta["pick"], meta["n"], meta["sink_kv"]
        frozen = {"theta": prefix_sets(masses[t0], args.theta),
                  "matched": topk_mask_by_count(masses[t0], routed[t0].sum(-1))}
        for kind, omega in frozen.items():
            card = omega.sum(-1)
            floor = diagonal_floor(card, pick, n, sink_kv)
            for t in steps[1:]:
                mt = masses[t]
                arms = {"frozen": (mt * omega).sum(-1),
                        "sol_at_t": (mt * routed[t]).sum(-1),
                        "oracle_at_t": (mt * topk_mask_by_count(mt, card)).sum(-1),
                        "floor": (mt * floor).sum(-1)}
                rows.append({
                    "capture_set": capset, "dit_block": blk, "build_step": t0, "step": t,
                    "frozen_kind": kind, "theta": args.theta if kind == "theta" else None,
                    "ordering": args.losa_ordering, "sampled": True,
                    "query_blocks_sampled": int(pick.numel()), "heads": int(card.shape[0]),
                    "cardinality_mean": round(float(card.float().mean()), 1),
                    "coverage_mean": round(float(card.float().mean()) / n, 5),
                    "sol_cardinality_mean": round(float(routed[t].sum(-1).float().mean()), 1),
                    "recall": {name: {"mean": round(float(v.mean()), 5),
                                      "p5": round(float(torch.quantile(v.flatten().float(), 0.05)), 5),
                                      "min": round(float(v.min()), 5)}
                               for name, v in arms.items()},
                    "drift_cost_mean": round(float((arms["oracle_at_t"] - arms["frozen"]).mean()), 5),
                    "frozen_minus_sol_mean": round(float((arms["frozen"] - arms["sol_at_t"]).mean()), 5),
                })
                if verbose:
                    r = rows[-1]
                    print(f"  G {capset[:12]} b{blk} s{t0}->s{t} {kind:8s} |O| "
                          f"{r['cardinality_mean']:.0f} frozen {r['recall']['frozen']['mean']:.4f} "
                          f"(p5 {r['recall']['frozen']['p5']:.4f}) sol "
                          f"{r['recall']['sol_at_t']['mean']:.4f} oracle "
                          f"{r['recall']['oracle_at_t']['mean']:.4f} floor "
                          f"{r['recall']['floor']['mean']:.4f}", flush=True)
        del masses, routed
    return rows


# --------------------------------------------------------------------- images

def _palette(ids):
    """A repeating categorical colour, hue-hopped by the golden ratio so consecutive ids never look alike."""
    import colorsys
    import numpy as np
    hues = (ids.astype(float) * 0.6180339887) % 1.0
    flat = np.array([colorsys.hsv_to_rgb(h, 0.62, 0.95) for h in hues.ravel()])
    return flat.reshape(ids.shape + (3,))


def write_figures(path, outdir, args, node, recipe, frames=(10, 40, 80), size=64):
    """The token groupings, painted on the latent grid. Returns the file names written."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    grid = tuple(int(x) for x in args.grid.split(","))
    gt, gh, gw = grid
    start = args.video_start
    a0, a1 = (int(x) for x in args.audio_span.split(","))
    q, k, v = ase.load_capture(path)
    q, k, v = (x[:, :args.heads].to(torch.float32).contiguous() for x in (q, k, v))
    tokens = q.shape[2]
    scale = q.shape[-1] ** -0.5
    layout = {"sol_h3_video_span": (start, tokens), "sol_h3_audio_span": (a0, a1)}
    sink_kv, sink_q = node._sink_blocks(layout, tokens, recipe["sink_conditioning"])

    sample = stratified_video_queries(grid, start, args.per_frame, args.seed)
    written = []
    panels = {}
    for ordering in ORDERINGS:
        index = ordering_index(node, ordering, grid, start, tokens)
        pos = torch.argsort(index)
        mask, _ = route_mask(q[:, :, index], k[:, :, index], TAU, sink_kv, sink_q, size)
        vb = video_blocks(start, tokens, size)
        spread = torch.full((mask.shape[-1],), float("nan"))
        spread[vb] = within_block_spread(k[:, :, index], size, vb).mean(0)
        routed = torch.full((mask.shape[-1],), float("nan"))
        routed[vb] = mask[:, vb, :].sum(-1).float().mean(0)
        missed = torch.zeros(sample.numel())
        for s0 in range(0, sample.numel(), args.chunk):
            qi = sample[s0:s0 + args.chunk]
            p_exact = torch.softmax(torch.einsum("hcd,hsd->hcs", q[0][:, qi, :], k[0]) * scale, dim=-1)
            m = torch.zeros(p_exact.shape[0], qi.numel(), mask.shape[-1])
            m.index_add_(2, pos // size, p_exact)
            r = mask[:, pos[qi] // size, :]
            missed[s0:s0 + qi.numel()] = (m * ~r).sum(-1).mean(0)
            del p_exact, m, r
        panels[ordering] = dict(pos=pos, spread=spread, routed=routed, missed=missed)

    area = gh * gw
    tok_of = lambda t, y, x: start + t * area + y * gw + x            # noqa: E731

    def paint(ordering, frame, kind):
        pos = panels[ordering]["pos"]
        ids = np.empty((gh, gw), dtype=np.int64)
        for y in range(gh):
            for x in range(gw):
                ids[y, x] = int(pos[tok_of(frame, y, x)]) // size
        if kind == "blocks":
            return ids, _palette(ids)
        src = panels[ordering][kind].numpy()
        return ids, src[ids]

    def grid_axes(ax, title):
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("latent w (42)", fontsize=7)
        ax.set_ylabel("latent h (24)", fontsize=7)

    for frame in frames:
        # 1. block identity, both orderings, each alone and side by side
        for ordering in ORDERINGS:
            _ids, rgb = paint(ordering, frame, "blocks")
            fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
            ax.imshow(rgb, interpolation="nearest", aspect="auto")
            grid_axes(ax, f"block membership, {ordering} order, latent frame {frame}, block {size}")
            fig.tight_layout(); name = f"blocks_{ordering}_frame{frame:02d}.png"
            fig.savefig(outdir / name); plt.close(fig); written.append(name)
        fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=100)
        for ax, ordering in zip(axes, ORDERINGS):
            _ids, rgb = paint(ordering, frame, "blocks")
            ax.imshow(rgb, interpolation="nearest", aspect="auto")
            grid_axes(ax, f"{ordering}: one colour per 64-token block, latent frame {frame}")
        fig.tight_layout(); name = f"blocks_compare_frame{frame:02d}.png"
        fig.savefig(outdir / name); plt.close(fig); written.append(name)

        # 2. within-block key spread, one colour scale for both orderings
        lo = float(np.nanmin([panels[o]["spread"].numpy() for o in ORDERINGS]))
        hi = float(np.nanmax([panels[o]["spread"].numpy() for o in ORDERINGS]))
        for ordering in ORDERINGS:
            _ids, val = paint(ordering, frame, "spread")
            fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
            im = ax.imshow(val, interpolation="nearest", aspect="auto", cmap="magma", vmin=lo, vmax=hi)
            fig.colorbar(im, ax=ax, shrink=0.8, label="key spread (0 alike, 1 unrelated)")
            grid_axes(ax, f"within-block key spread, {ordering}, latent frame {frame}")
            fig.tight_layout(); name = f"spread_{ordering}_frame{frame:02d}.png"
            fig.savefig(outdir / name); plt.close(fig); written.append(name)

        # 3. how many key blocks each query block routed
        rlo = float(np.nanmin([panels[o]["routed"].numpy() for o in ORDERINGS]))
        rhi = float(np.nanmax([panels[o]["routed"].numpy() for o in ORDERINGS]))
        for ordering in ORDERINGS:
            _ids, val = paint(ordering, frame, "routed")
            fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
            im = ax.imshow(val, interpolation="nearest", aspect="auto", cmap="viridis", vmin=rlo, vmax=rhi)
            fig.colorbar(im, ax=ax, shrink=0.8, label="key blocks routed exactly")
            grid_axes(ax, f"routed key blocks per query block, {ordering}, latent frame {frame}")
            fig.tight_layout(); name = f"routed_{ordering}_frame{frame:02d}.png"
            fig.savefig(outdir / name); plt.close(fig); written.append(name)

        # 4. missed mass at the sampled queries of this frame
        lo4 = float(min(panels[o]["missed"].min() for o in ORDERINGS))
        hi4 = float(max(panels[o]["missed"].max() for o in ORDERINGS))
        in_frame = ((sample - start) // area == frame).nonzero().squeeze(-1)
        if in_frame.numel():
            rem = (sample[in_frame] - start) % area
            ys, xs = (rem // gw).numpy(), (rem % gw).numpy()
            for ordering in ORDERINGS:
                fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
                ax.set_facecolor("#f2f2f2")
                sc = ax.scatter(xs, ys, c=panels[ordering]["missed"][in_frame].numpy(),
                                cmap="inferno", vmin=lo4, vmax=hi4, s=110, edgecolors="k", linewidths=0.3)
                fig.colorbar(sc, ax=ax, shrink=0.8, label="share of exact mass in pooled blocks")
                ax.set_xlim(-0.5, gw - 0.5); ax.set_ylim(gh - 0.5, -0.5)
                grid_axes(ax, f"missed mass at sampled queries, {ordering}, latent frame {frame}")
                fig.tight_layout(); name = f"missed_{ordering}_frame{frame:02d}.png"
                fig.savefig(outdir / name); plt.close(fig); written.append(name)

    # the brick: which latent frames one `3d` block spans
    centre = tok_of(40, gh // 2, gw // 2)
    bid = int(panels["3d"]["pos"][centre]) // size
    span = range(38, 44)
    fig, axes = plt.subplots(1, len(span), figsize=(15, 3), dpi=140)
    for ax, frame in zip(axes, span):
        img = np.zeros((gh, gw))
        for y in range(gh):
            for x in range(gw):
                img[y, x] = 1.0 if int(panels["3d"]["pos"][tok_of(frame, y, x)]) // size == bid else 0.0
        ax.imshow(img, interpolation="nearest", aspect="auto", cmap="Greys", vmin=0, vmax=1)
        ax.set_title(f"frame {frame}: {int(img.sum())} tokens", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"the single 3d block holding the centre token of frame 40 (block {bid}), across frames "
                 f"{span.start} to {span.stop - 1}", fontsize=10)
    fig.tight_layout(); name = "bricks_3d_centre.png"
    fig.savefig(outdir / name); plt.close(fig); written.append(name)
    return written


# -------------------------------------------------------------------- reports

def summarize(path, live=None, fingerprint=None, unpinned=()):
    """The dated record, rebuilt from the data file so no number is ever re-typed.

    `live` is the control block RE-RUN by the caller on the current code, and it
    is what the record prints: the block stored in the JSON describes whatever
    revision wrote it, which is exactly the thing that went wrong once.
    """
    import orjson
    data = orjson.loads(Path(path).read_bytes())
    cond = data["conditions"]
    names = [cell_key(o, s) for o in ORDERINGS for s in BLOCK_SIZES]
    # the capture-set directory names are longer than the rest of a row put
    # together, so the tables carry a tag and the legend carries the name
    tags = {s: chr(ord("A") + i) for i, s in enumerate(cond["capture_sets"])}
    short = lambda cid: cid.split("/")[-1] if "/" not in cid else (  # noqa: E731
        f"{tags.get(cid.split('/')[0], '?')} {cid.split('/')[-1]}")
    out = []
    w = out.append
    w(f"# Where Sol-Attn's block grouping sends mass to the pooled branch, {data['measured']}")
    w("")
    w(f"Model: {data['model']}.")
    w("")
    w("Script: `bench/analyze_sol_block_grouping.py`, whose docstring is the method.")
    w(f"Data: `{Path(path).name}`. CPU only, fp32, tau {cond['tau']}, head prefix of")
    w(f"{cond['heads']} of 56, sink conditioning `{cond['sink_conditioning']}`, block")
    w(f"sizes {cond['block_sizes']}, orderings {cond['orderings']}. Rows are tagged by")
    w("capture set: " + ", ".join(f"**{tags[c]}** = `{c}`" for c in cond["capture_sets"]) + ".")
    if data.get("figures"):
        w("")
        w("The same groupings painted on the latent grid, for the capture"
          + ("s " if len(data["figures"]) > 1 else " ")
          + ", ".join(f"`{f}`" for f in sorted(data["figures"]))
          + ", are under `internal/2026-09-19_block_grouping/` with their own README.")
        w("That directory is gitignored, so the pictures are local to the box that ran")
        w("this; `--figures DIR` rebuilds them from the capture.")
    w("")
    w("## Instrument")
    w("")
    if live is not None:
        w("The controls below were RE-RUN against this file when this record was")
        w("generated, not read out of the data file, and the record refuses to write")
        w("if they fail or if a stored measurement carries a different measurement-path")
        w(f"fingerprint. Current fingerprint: `{fingerprint}`.")
        w("")
        w("| control | value |")
        w("|---|---|")
        for key, val in live.items():
            if key == "reported_not_gating":
                continue
            w(f"| {key} | {val} |")
        w("")
        ng = live.get("reported_not_gating", {})
        if ng:
            w("Reported, deliberately NOT gating:")
            w("")
            for key, val in ng.items():
                if key == "why_not_gating":
                    continue
                w(f"- `{key}`: {val}")
            w("")
            w(ng.get("why_not_gating", ""))
            w("")
        if unpinned:
            w("Measured before the fingerprint existed, so pinned to code only by the")
            w("date of this record: " + ", ".join(f"`{u}`" for u in sorted(unpinned)) + ".")
            w("")
    w("## What this is and is not")
    w("")
    w("The Sol ALGORITHM in fp32 and the exact attention it approximates, on captured")
    w("post-RoPE q/k/v. No INT8 term, no kernel, no clip.")
    w("")
    w("**Missed mass is not output error.** It is the share of a query's exact")
    w("softmax mass that Sol routed through the POOLED branch rather than attending")
    w("exactly. Sol still supplies an approximation of that mass, and the fidelity of")
    w("the pooled term is a quantity nothing here measures, so a configuration can")
    w("push less mass into the pooled branch and still land further from dense")
    w("attention. Output error on these same captures lives in")
    w("`2026-09-18_sol_block_size.md`.")
    cc = (data.get("instrument") or {}).get("missed_mass_vs_rel_l2_ordering")
    if cc:
        for size, row in cc["by_block_size"].items():
            w("")
            w(f"At block {size}, missed mass and that record's output rel_l2 rank the two")
            w(f"orderings the same way on {row['same_ordering']} of {row['cells']} cells, "
              "compared at the density plain")
            w("order routes at this tau.")
            for bad in row["disagreements"]:
                w(f"They disagree on `{bad['cell']}`: rel_l2 moves by "
                  f"{bad['rel_l2_3d_over_raster']}x under `3d` while missed mass moves by "
                  f"{bad['missed_3d_over_raster']}x.")
        w("")
        w("Those cells are where this record must not be read as a statement about")
        w("error. Regenerate the comparison with `--cross-check`.")
    w("")
    w("**Cells are at equal tau, not at equal cost.** The four (ordering, block size)")
    w("cells route at four different densities, printed in every table. In this data")
    w("they all land within a few percent of one another, so the confound is small,")
    w("but that is an observation about these captures rather than a property of tau.")
    w("`sweep_sol_block_size_on_capture.py` is the instrument that holds cost fixed by")
    w("construction. The rest of the instrument controls are in the JSON under")
    w("`instrument`.")
    w("")
    w("## A. How unlike are the tokens that share a block?")
    w("")
    w("Mean squared distance of a block's keys from its mean key over their mean")
    w("squared norm: 0 identical, about 1 unrelated. Video blocks only (the block")
    w("straddling the conditioning rows and the ragged last block are excluded).")
    w("The pooled term replaces every key in a block by that mean, so this is the")
    w("raw material of the key side.")
    w("")
    w("| cell | " + " | ".join(f"{n} key mean / p90" for n in names) + " |")
    w("|---|" + "---|" * len(names))
    for cid, cell in data["cells"].items():
        cols = [f"{cell['A_spread'][n]['key']['mean']:.3f} / {cell['A_spread'][n]['key']['p90']:.3f}"
                for n in names]
        w(f"| {short(cid)} | " + " | ".join(cols) + " |")
    w("")
    w("| cell | " + " | ".join(f"{n} query mean / p90" for n in names) + " |")
    w("|---|" + "---|" * len(names))
    for cid, cell in data["cells"].items():
        cols = [f"{cell['A_spread'][n]['query']['mean']:.3f} / {cell['A_spread'][n]['query']['p90']:.3f}"
                for n in names]
        w(f"| {short(cid)} | " + " | ".join(cols) + " |")
    w("")
    w("## B. Does pooling hide mass the query wanted?")
    w("")
    w("Sampled video query tokens, stratified over latent frames, the same physical")
    w("tokens in every cell. `missed` is the share of the query's EXACT softmax mass")
    w("sitting in key blocks its query block did not route. `top` is the share of a")
    w("pooled block's mass carried by the top eighth of its tokens, mass-weighted.")
    w("`density` is the share of key blocks routed, and moves with the cell.")
    w("")
    w("| cell | " + " | ".join(f"{n} density / missed mean / missed p90 / top" for n in names) + " |")
    w("|---|" + "---|" * len(names))
    for cid, cell in data["cells"].items():
        cols = []
        for n in names:
            b = cell["B"][n]
            cols.append(f"{b['density']:.3f} / {b['missed_mass']['mean']:.4f} / "
                        f"{b['missed_mass']['p90']:.4f} / {b['pooled_block_top_fraction_share']:.3f}")
        w(f"| {short(cid)} | " + " | ".join(cols) + " |")
    w("")
    w("Tokens needed to cover 90 percent of a query's exact mass, against the tokens")
    w("Sol actually attended exactly. The first is a property of the query and does")
    w("not move with the cell; the second is the cell's cost.")
    w("")
    w("| cell | tokens for 90% (median / p90) | " + " | ".join(f"{n} routed tokens" for n in names) + " |")
    w("|---|---|" + "---|" * len(names))
    for cid, cell in data["cells"].items():
        b0 = cell["B"][names[0]]
        cols = [f"{cell['B'][n]['routed_tokens']['mean']:.0f}" for n in names]
        w(f"| {short(cid)} | {b0['tokens_for_90pc_mass']['median']:.0f} / "
          f"{b0['tokens_for_90pc_mass']['p90']:.0f} | " + " | ".join(cols) + " |")
    w("")
    w("## C. Do the queries of one block want the same keys?")
    w("")
    w("Each individual query's own ideal key-block set (top-k by its exact mass, k =")
    w("what its block routed) against the block's actual routed set, and against its")
    w("neighbours'. Jaccard has a chance floor at these densities, printed beside")
    w("every number. `disc` removes the forced blocks (diagonal and sinks) from both")
    w("sets and from k; those agree by construction, so `disc` is the number that")
    w("answers the question.")
    w("")
    w("| cell | " + " | ".join(f"{n} q-vs-block disc (chance) / q-vs-q disc" for n in names) + " |")
    w("|---|" + "---|" * len(names))
    for cid, cell in data["cells"].items():
        cols = []
        for n in names:
            c = cell["C"][n]
            cols.append(f"{c['query_vs_block']['discretionary']['mean']:.3f} "
                        f"({c['jaccard_chance_discretionary']:.3f}) / "
                        f"{c['query_vs_query']['discretionary']['mean']:.3f}")
        w(f"| {short(cid)} | " + " | ".join(cols) + " |")
    w("")
    w("| cell | " + " | ".join(f"{n} q-vs-block all (chance)" for n in names) + " |")
    w("|---|" + "---|" * len(names))
    for cid, cell in data["cells"].items():
        cols = []
        for n in names:
            c = cell["C"][n]
            cols.append(f"{c['query_vs_block']['all']['mean']:.3f} ({c['jaccard_chance_all']:.3f})")
        w(f"| {short(cid)} | " + " | ".join(cols) + " |")
    w("")
    w("## D. Would a better rule at the same granularity do it?")
    w("")
    w("Mean missed mass under four ranking rules, each keeping the same NUMBER of key")
    w("blocks Sol kept, on the same sampled queries. `sol` is the shipped centroid")
    w("against centred centroid. `quest` is the per-channel min/max upper bound, a")
    w("rule a kernel could run. `blockmax` ranks by the TRUE largest token score in")
    w("the block, which that bound relaxes, so it separates a loose bound from a")
    w("wrong objective. `oracle` ranks by the query's own exact mass per block: the")
    w("floor any rule at this granularity could reach for that query.")
    w("")
    w("| cell | " + " | ".join(f"{n} sol / quest / blockmax / oracle" for n in names) + " |")
    w("|---|" + "---|" * len(names))
    for cid, cell in data["cells"].items():
        cols = []
        for n in names:
            dd = cell["D"][n]
            cols.append(f"{dd['missed_mass_sol']['mean']:.4f} / "
                        f"{dd['missed_mass_quest']['mean']:.4f} / "
                        f"{dd['missed_mass_blockmax']['mean']:.4f} / "
                        f"{dd['missed_mass_oracle']['mean']:.4f}")
        w(f"| {short(cid)} | " + " | ".join(cols) + " |")
    w("")
    if data.get("step_pairs"):
        w("## E. Does the routed set survive from one step to the next?")
        w("")
        w("Jaccard of the routed key-block sets per video query block and head, between")
        w("two captured steps of one DiT block. `disc` excludes the forced blocks, which")
        w("are identical by construction. Chance floors in brackets.")
        w("")
        w("| capture set | DiT block | steps | cell | density | J all (chance) | J disc (chance) |")
        w("|---|---|---|---|---|---|---|")
        for r in data["step_pairs"]:
            w(f"| {tags.get(r['capture_set'], '?')} | {r['dit_block']} | {r['steps'][0]}-{r['steps'][1]} | {r['cell']} "
              f"| {r['density_first']:.3f}/{r['density_second']:.3f} "
              f"| {r['jaccard_all']['mean']:.3f} ({r['jaccard_chance_all']:.3f}) "
              f"| {r['jaccard_discretionary']['mean']:.3f} ({r['jaccard_chance_discretionary']:.3f}) |")
        w("")
    arm_cells = {cid: c for cid, c in data["cells"].items() if "arms" in c}
    if arm_cells:
        shown = ["sol", "max4", "lse4", "split2", "split4", "split4_union",
                 "split4_lse4", "token_aug", "token_aug_hisa", "oracle"]
        w("## F. Arms at equal executed cost")
        w("")
        w("COST is the count of (query token, key token) pairs attended EXACTLY per")
        w("head, forced diagonal and sinks included, and every arm is tuned until its")
        w("mean cost over the sampled queries equals Sol's at tau 1.0 on that cell. So")
        w("a row that wins has won at Sol's price. `oracle` is the per-query ceiling:")
        w("the best block set for that query at that count, which no per-block rule")
        w("can beat.")
        w("")
        w("The arms. `max4` and `lse4` keep Sol's 64-row query block and 64-token key")
        w("block and only change the RANKING, scoring a key block by the max or the")
        w("log-sum-exp over four 16-key sub-means instead of by its single mean.")
        w("`split2` and `split4` keep the key side and split the QUERY block into 32-")
        w("or 16-row sub-blocks that route separately with Sol's own rule.")
        w("`split4_union` is `split4` tuned so that the UNION of the four sub-blocks'")
        w("sets -- what a CTA actually walks -- costs what Sol costs, which is the")
        w("kernel's real price. `token_aug` is kitchen's own token stage, emulated")
        w("from the kernel's geometry (one centroid per 2 query blocks, candidates")
        w("the blocks neither member routed, budget 64 per group, and the leftover")
        w("candidates becoming a per-token tail), with the block stage loosened until")
        w("block cost plus budget lands on Sol's. `token_aug_hisa` adds HISA's cut.")
        w("")
        w("**`rel_l2` is the output error** of Sol's own arithmetic under each route,")
        w("against exact attention on the same query rows. It is the number that")
        w("decides, and it is not missed mass: the two disagree on sign in this table.")
        w("")
        for o in ORDERINGS:
            w(f"### Output error under each route, {o} order")
            w("")
            w("| cell | " + " | ".join(shown) + " |")
            w("|---|" + "---|" * len(shown))
            for cid, c in arm_cells.items():
                r = c["arms"][o]
                w(f"| {short(cid)} | " + " | ".join(f"{r[a]['rel_l2_vs_exact']:.4f}" for a in shown) + " |")
            w("")
        w("### Missed mass under each route, plain order")
        w("")
        w("The same arms by the share of exact mass they route through the pooled")
        w("branch. Compare it with the table above: on several cells the arm with the")
        w("least missed mass is not the arm with the least error.")
        w("")
        w("| cell | " + " | ".join(shown) + " |")
        w("|---|" + "---|" * len(shown))
        for cid, c in arm_cells.items():
            r = c["arms"]["raster"]
            w(f"| {short(cid)} | " + " | ".join(f"{r[a]['missed_mass']['mean']:.4f}" for a in shown) + " |")
        w("")
        w("### Union inflation: what a query split really costs")
        w("")
        w("A 64-row CTA walks the UNION of its sub-blocks' routed sets, so a query")
        w("split that matches Sol's cost per sub-block makes the CTA load more tiles.")
        w("`inflation` is that union over the per-sub-block cost at the tuned tau.")
        w("")
        w("| cell | split2 inflation (plain / 3d) | split4 inflation (plain / 3d) |")
        w("|---|---|---|")
        for cid, c in arm_cells.items():
            s2 = [c["arms_setting"][o]["split2"]["union_inflation"] for o in ORDERINGS]
            s4 = [c["arms_setting"][o]["split4"]["union_inflation"] for o in ORDERINGS]
            w(f"| {short(cid)} | {s2[0]:.3f} / {s2[1]:.3f} | {s4[0]:.3f} / {s4[1]:.3f} |")
        w("")
        w("### What the token stage can buy, and what limits it")
        w("")
        w("At Sol's OWN routing, the individual key tokens needed to recover half of")
        w("Sol's missed mass, ranked two ways: by the score of the centroid kitchen's")
        w("token stage shares across 128 query rows, and by the query's own exact mass")
        w("over the SAME candidates. The gap between them is the price of the shared")
        w("centroid rather than of the budget, which the kernel caps at 256.")
        w("")
        w("| cell | centroid rank, tokens for half | per-query rank, tokens for half | share reaching ninety (centroid / per-query) |")
        w("|---|---|---|---|")
        for cid, c in arm_cells.items():
            rc = c["arms"]["raster"]["_token_recovery"]
            w(f"| {short(cid)} | {rc['centroid']['tokens_for_half']['median']:.0f} | "
              f"{rc['per_query']['tokens_for_half']['median']:.0f} | "
              f"{rc['centroid']['share_reaching_ninety']:.2f} / "
              f"{rc['per_query']['share_reaching_ninety']:.2f} |")
        w("")
        w(f"Capped at {arm_cells[list(arm_cells)[0]]['arms']['raster']['_token_recovery']['cap']} tokens; "
          "a row at the cap did not get there.")
        w("")

    if data.get("losa"):
        w("## G. LoSA's frozen pattern, on a sample")
        w("")
        w("A per-(head, query block) set of key blocks built at the EARLIEST captured")
        w("step and frozen, then measured at each later step of the same DiT block.")
        w("`theta` keeps the shortest prefix reaching the mass target; `matched` keeps")
        w("as many blocks as Sol itself routed at the build step. `sol_at_t` is Sol's")
        w("own set REBUILT at step t and it is the arm that decides: if Sol matches or")
        w("beats the frozen set at equal cardinality, freezing buys nothing here.")
        w("`oracle_at_t` is the best set of that size at that step, so oracle minus")
        w("frozen is the drift cost. `floor` is sinks plus a widening diagonal band at")
        w("the same size, the structure a routed set gets for free.")
        w("")
        w("**These rows are a SAMPLE**: exact block masses need a full softmax row per")
        w("query row, so a fixed-seed sample of query blocks per cell was measured on")
        w("CPU. The count is in every row, and the full run belongs on the card.")
        w("")
        w("| set | block | steps | kind | coverage | frozen (p5) | sol at t | oracle | floor |")
        w("|---|---|---|---|---|---|---|---|---|")
        for r in data["losa"]:
            rec = r["recall"]
            w(f"| {tags.get(r['capture_set'], '?')} | {r['dit_block']} | "
              f"{r['build_step']}-{r['step']} | {r['frozen_kind']} | {r['coverage_mean']:.3f} | "
              f"{rec['frozen']['mean']:.4f} ({rec['frozen']['p5']:.4f}) | "
              f"{rec['sol_at_t']['mean']:.4f} | {rec['oracle_at_t']['mean']:.4f} | "
              f"{rec['floor']['mean']:.4f} |")
        w("")
        n_q = data["losa"][0]["query_blocks_sampled"]
        w(f"Sampled query blocks per cell: {n_q}. Heads: {data['losa'][0]['heads']}. "
          f"Plain token order.")
        w("")

    res_cells = {cid: c for cid, c in data["cells"].items() if "frame_residue" in c}
    if res_cells:
        w("## H. Does Sol under-route the one-pixel-frame latents?")
        w("")
        w("`bench/map_attention_mass_on_capture.py` found that latent frames whose")
        w("index is 0 mod 5 -- the ones whose RoPE time span is one pixel frame where")
        w("the rest span four -- draw more exact attention mass than their share of")
        w("keys. That says where the mass is. This asks whether Sol's routing keeps up")
        w("with it: for the shipped `sol` arm, each residue class's share of the")
        w("MISSED mass against its share of the true mass. Above 1 means Sol misses")
        w("that class more than its mass warrants. **Residues 1 to 4 are the control**;")
        w("an effect of the one-pixel-frame latents lifts residue 0 alone.")
        w("")
        w("| cell | mass/keys, residue 0 | missed/mass by residue 0 | 1 | 2 | 3 | 4 |")
        w("|---|---|---|---|---|---|---|")
        for cid, c in res_cells.items():
            cl = c["frame_residue"]["classes"]
            w(f"| {short(cid)} | {cl['residue_0']['true_mass_over_keys']:.3f} | "
              + " | ".join(f"{cl['residue_' + str(j)]['missed_over_true_mass']:.3f}" for j in range(5)) + " |")
        w("")
        w("Plain token order, tau 1.0, the same sampled video queries as F.")
        w("`conditioning` is carried in the JSON as a sixth class and is always exact.")
        w("")

    w("## What this says")
    w("")
    for line in data.get("verdict") or VERDICT:
        w(line)
    if any("arms" in c for c in data["cells"].values()) or data.get("losa"):
        for line in VERDICT_ARMS:
            w(line)
    w("")
    w("## Limits")
    w("")
    w("Two clips, the captured DiT blocks and steps only. A head PREFIX of the first")
    w("8 of 56, not a sample. Equal tau, not equal cost. fp32 with no INT8 term, so")
    w("every number is a ceiling on what a grouping change could buy and not a")
    w("forecast of a kernel. Routing COST, which grows with the square of the block")
    w("count, is in nothing here, so block 16's numbers do not carry their own price.")
    w("**Missed mass is not output error**, and on the cells named above the two rank")
    w("the orderings oppositely; the pooled term's own fidelity is measured nowhere")
    w("here. These are properties of one attention call on captured inputs, not of a")
    w("rendered clip, and this pack has a standing example of error against exact")
    w("attention and a watched clip disagreeing.")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captures", nargs="*", help="captures for A to D")
    ap.add_argument("--steps", nargs="*", default=[], help="captures for E; paired by DiT block within a set")
    ap.add_argument("--heads", type=int, default=8, help="head PREFIX")
    ap.add_argument("--grid", default="102,24,42", help="token T,H,W of the video segment")
    ap.add_argument("--audio-span", default="395,1545")
    ap.add_argument("--video-start", type=int, default=1545)
    ap.add_argument("--per-frame", type=int, default=20, help="B: sampled query tokens per latent frame")
    ap.add_argument("--c-blocks", type=int, default=200, help="C: sampled query blocks")
    ap.add_argument("--c-queries", type=int, default=16, help="C: sampled queries within each block")
    ap.add_argument("--chunk", type=int, default=64, help="query rows held in memory at once")
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--check-only", action="store_true", help="run the instrument controls and stop")
    ap.add_argument("--figures", metavar="DIR", help="also paint the groupings of the FIRST capture into DIR")
    ap.add_argument("--arms", nargs="*", default=None, metavar="CAP.pt",
                    help="measurement F: the equal-executed-cost routing arms, on these captures")
    ap.add_argument("--tok-group", type=int, default=2,
                    help="query blocks per token-stage centroid; kitchen's TOK_GROUP")
    ap.add_argument("--token-budget", type=int, default=64,
                    help="token-stage budget per group; kitchen takes 64, 128 or 256")
    ap.add_argument("--frames", nargs="*", default=[], metavar="CAP.pt",
                    help="measurement H: Sol's missed mass by latent-frame residue, with the residue control")
    ap.add_argument("--losa", nargs="*", default=[], metavar="CAP.pt",
                    help="measurement G: LoSA's frozen-pattern recall, over steps of one DiT block")
    ap.add_argument("--losa-ordering", default="raster", choices=list(ORDERINGS))
    ap.add_argument("--query-blocks", type=int, default=100,
                    help="measurement G: query blocks sampled per cell, fixed seed")
    ap.add_argument("--theta", type=float, default=0.99, help="measurement G: LoSA's mass target")
    ap.add_argument("--device", default="cpu", help="measurement G: torch device for the mass passes")
    ap.add_argument("--err-stride", type=int, default=4,
                    help="measurement F: emulate the OUTPUT on every Nth sampled query "
                         "(missed mass still uses all of them)")
    ap.add_argument("--recover-cap", type=int, default=4096,
                    help="measurement F: largest token count the recovery curve looks at")
    ap.add_argument("--hisa-pool", type=int, default=64,
                    help="HISA cut: unrouted blocks the token pass may look inside")
    ap.add_argument("--cross-check", nargs="+", metavar="PRIOR.json", default=[],
                    help="sweep_sol_block_size_on_capture records on the same captures; missed mass is "
                         "ranked against their output rel_l2 and the agreement is stored under `instrument`")
    ap.add_argument("--summarize", metavar="RESULT.json", help="print the dated record for a finished run")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.summarize:
        # A record must never be able to claim a green control for code that is
        # not the code its numbers came from. So the controls are RE-RUN here on
        # the current source, and every group's stored fingerprint is checked
        # against the current measurement path. Red controls, or a stored
        # fingerprint that no longer matches, refuse the write outright; a group
        # measured before fingerprinting existed is reported as unpinned rather
        # than trusted silently.
        import orjson as _o
        live = controls()
        fp = measurement_fingerprint()
        if not live["passed"]:
            sys.stderr.write("REFUSING TO WRITE THE RECORD: the instrument controls do not pass on "
                             "this code.\n" + _o.dumps(live, option=_o.OPT_INDENT_2).decode() + "\n")
            return 1
        doc = _o.loads(Path(args.summarize).read_bytes())
        stale, unpinned = [], []
        groups = [(f"{cid} F", c.get("F_measurement_fingerprint"))
                  for cid, c in doc.get("cells", {}).items() if "arms" in c]
        groups += [(f"{cid} H", (c.get("frame_residue") or {}).get("measurement_fingerprint"))
                   for cid, c in doc.get("cells", {}).items() if "frame_residue" in c]
        groups += [(f"losa row {i}", r.get("measurement_fingerprint"))
                   for i, r in enumerate(doc.get("losa", []))]
        for name, got in groups:
            if got is None:
                unpinned.append(name)
            elif got != fp:
                stale.append(name)
        if stale:
            sys.stderr.write("REFUSING TO WRITE THE RECORD: these groups were measured by a "
                             "different measurement path than the one in this file, so their "
                             "numbers and these controls are not one revision. Re-run them.\n  "
                             + "\n  ".join(stale) + "\n")
            return 1
        sys.stdout.write(summarize(args.summarize, live=live, fingerprint=fp, unpinned=unpinned))
        return 0

    import orjson
    print("instrument controls:")
    ctl = controls()
    ctl["measurement_fingerprint"] = measurement_fingerprint()
    for key, val in ctl.items():
        print(f"  {key}: {val}")
    if not ctl["passed"]:
        print("FAIL: the route mask does not reproduce the vendored oracle, or the mutation did not move it")
        return 1
    if args.check_only:
        return 0
    if not args.out:
        ap.error("--out is required unless --check-only or --summarize")

    node = live_sol()
    sys.path.insert(0, str(REPO / "workflows"))
    import h3_config
    recipe = h3_config.SOL_RECOMMENDED_CUDA

    out = Path(args.out)
    doc = orjson.loads(out.read_bytes()) if out.exists() else {}
    cells = doc.get("cells", {})
    capture_sets = set(doc.get("conditions", {}).get("capture_sets", []))
    layouts = doc.get("conditions", {}).get("layouts", {})

    def flush():
        doc.update({
            "measured": date.today().isoformat(),
            "produced_by": "bench/analyze_sol_block_grouping.py",
            "what": "where Sol-Attn's 64-token block grouping loses accuracy: within-block spread, "
                    "missed mass under pooling, query-side disagreement inside a block, a Quest-style "
                    "routing rule at the same granularity, and route stability across denoising steps",
            "model": "MiniMax H3, int8 convrot checkpoint, base 16-step text to video at the trained canvas",
            "conditions": {"device": "cpu", "dtype": "float32", "heads": args.heads, "tau": TAU,
                           "layouts": layouts,
                           "sink_conditioning": recipe["sink_conditioning"],
                           "pooled_tail": recipe["pooled_tail"],
                           "block_sizes": list(BLOCK_SIZES), "orderings": list(ORDERINGS),
                           "queries_per_latent_frame": args.per_frame,
                           "c_blocks": args.c_blocks, "c_queries_per_block": args.c_queries,
                           "top_fraction_of_block": TOP_FRACTION, "mass_covered": COVER,
                           "seed": args.seed, "capture_sets": sorted(capture_sets)},
            "instrument": dict(doc.get("instrument", {}), **ctl),
            "how_to_read": "cells are at equal tau, NOT equal cost: every B and D row carries its density. "
                           "Jaccard has a chance floor that differs per cell and sits beside each value. "
                           "Discretionary numbers exclude the forced blocks, which agree by construction.",
            "cells": cells,
        })
        out.write_bytes(orjson.dumps(doc, option=orjson.OPT_INDENT_2))

    for cap in args.captures:
        capset = Path(cap).resolve().parent.name
        m = re.search(r"_b(\d+)_s(\d+)", Path(cap).name)
        cid = f"{capset}/b{m.group(1)}_s{m.group(2)}" if m else f"{capset}/{Path(cap).stem}"
        if cid in cells:
            print(f"{cid}: already in {out.name}, skipping")
            continue
        print(f"{cid}:", flush=True)
        cells[cid] = run_capture(cap, args, node, recipe)
        capture_sets.add(capset)
        layouts[capset] = {"video_start": args.video_start, "grid_thw": cells[cid]["grid_thw"],
                           "audio_span": cells[cid]["audio_span"], "tokens": cells[cid]["tokens"]}
        print(f"  took {cells[cid]['seconds']} s", flush=True)
        flush()

    if args.steps:
        doc["step_pairs"] = doc.get("step_pairs", []) + run_step_pairs(args.steps, args, node, recipe)
        for cap in args.steps:
            capset = Path(cap).resolve().parent.name
            capture_sets.add(capset)
            layouts.setdefault(capset, {"video_start": args.video_start,
                                        "grid_thw": [int(x) for x in args.grid.split(",")],
                                        "audio_span": [int(x) for x in args.audio_span.split(",")]})
        flush()

    if not cells and not args.steps:
        flush()

    for cap in (args.arms or []):
        capset = Path(cap).resolve().parent.name
        m = re.search(r"_b(\d+)_s(\d+)", Path(cap).name)
        cid = f"{capset}/b{m.group(1)}_s{m.group(2)}" if m else f"{capset}/{Path(cap).stem}"
        if cid in cells and "arms" in cells[cid]:
            print(f"{cid}: arms already in {out.name}, skipping")
            continue
        print(f"{cid} arms:", flush=True)
        got = run_arms(cap, args, node, recipe)
        # A to E is the committed baseline in the same cell, so every scalar F
        # writes is namespaced: `sample_queries` differs between the two
        # measurements and an unprefixed merge silently relabels A to E's
        # conditions with F's. That happened once and was repaired by hand.
        cells.setdefault(cid, {}).update({key if key in ("arms", "arms_setting") else f"F_{key}": val
                                          for key, val in got.items()})
        cells[cid]["F_measurement_fingerprint"] = ctl["measurement_fingerprint"]
        capture_sets.add(capset)
        layouts.setdefault(capset, {"video_start": args.video_start,
                                    "grid_thw": [int(x) for x in args.grid.split(",")],
                                    "audio_span": [int(x) for x in args.audio_span.split(",")]})
        print(f"  took {got['seconds']} s", flush=True)
        flush()

    for cap in (args.frames or []):
        capset = Path(cap).resolve().parent.name
        m = re.search(r"_b(\d+)_s(\d+)", Path(cap).name)
        cid = f"{capset}/b{m.group(1)}_s{m.group(2)}" if m else f"{capset}/{Path(cap).stem}"
        if "frame_residue" in cells.get(cid, {}):
            print(f"{cid}: residue split already in {out.name}, skipping")
            continue
        print(f"{cid} frame residue:", flush=True)
        cells.setdefault(cid, {})["frame_residue"] = run_frames(cap, args, node, recipe)
        cells[cid]["frame_residue"]["measurement_fingerprint"] = ctl["measurement_fingerprint"]
        capture_sets.add(capset)
        flush()

    if args.losa:
        fresh = run_losa(args.losa, args, node, recipe)
        for row in fresh:
            row["measurement_fingerprint"] = ctl["measurement_fingerprint"]
        doc["losa"] = doc.get("losa", []) + fresh
        for cap in args.losa:
            capture_sets.add(Path(cap).resolve().parent.name)
        flush()

    if args.cross_check:
        ctl["missed_mass_vs_rel_l2_ordering"] = cross_check_rel_l2(
            {"cells": cells}, args.cross_check)
        flush()
        import orjson as _o
        print(_o.dumps(ctl["missed_mass_vs_rel_l2_ordering"], option=_o.OPT_INDENT_2).decode())

    if args.figures:
        if not args.captures:
            ap.error("--figures needs a capture")
        written = write_figures(args.captures[0], args.figures, args, node, recipe)
        doc.setdefault("figures", {})[Path(args.captures[0]).name] = sorted(written)
        flush()
        print(f"wrote {len(written)} images into {args.figures}")

    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
