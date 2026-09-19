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
    out["passed"] = bool(worst_count == 0 and worst_out < 1e-5
                         and mutated > 100 * max(worst_out, 1e-9)
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

def summarize(path):
    """The dated record, rebuilt from the data file so no number is ever re-typed."""
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
    w("## What this says")
    w("")
    for line in data.get("verdict") or VERDICT:
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
    ap.add_argument("--cross-check", nargs="+", metavar="PRIOR.json", default=[],
                    help="sweep_sol_block_size_on_capture records on the same captures; missed mass is "
                         "ranked against their output rel_l2 and the agreement is stored under `instrument`")
    ap.add_argument("--summarize", metavar="RESULT.json", help="print the dated record for a finished run")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.summarize:
        sys.stdout.write(summarize(args.summarize))
        return 0

    import orjson
    print("instrument controls:")
    ctl = controls()
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
