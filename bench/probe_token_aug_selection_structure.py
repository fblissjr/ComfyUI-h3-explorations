#!/usr/bin/env python3
"""Where does token_aug's run-to-run variation sit, and does it have the shape
of a per-centroid selection set?

`bench/probe_token_aug_determinism.py` established that the variation is not
accumulation order: on block 49 the deltas are many times the output's own mean
magnitude on a small fraction of rows. It could not say what varies, and its
record says the admitted set "is not observable from outside the kernel".

That sentence is half wrong, and this file is the correction. The kernel's own
source says exactly how the set is chosen (comfy-kitchen, branch
`sol-blk-cnt-0.2.33`, `comfy_kitchen/backends/cuda/sage_attention/
sol_attn_token.cu`, the file's header comment and pass 2):

  * A centroid is shared by TOK_GROUP neighbouring query blocks, so the unit of
    selection is TOK_GROUP * BLOCK contiguous token rows, per head.
  * Pass 1 histograms every candidate token's score. Pass 2 admits WHOLE BINS
    from the top until the next bin would overflow `n_tok`, and thresholds at
    that bin's lower edge. So the boundary is a bin edge, not a rank cut --
    which is why open experiment 29's arm 1, "the gap between the last admitted
    token's score and the first rejected one", names a quantity the kernel does
    not compute.
  * A token that clears the threshold reserves a slot with
    `atomicAdd(&tok_cnt[...])` and is only listed `if (slot < n_tok)`.

The last point is the mechanism this probe is built to look for. The kernel's
docstring claims the selected set "does not depend on scheduling", and that
holds only while the number of tokens clearing the threshold is at most
`n_tok`. If ever it is more, the losers are decided by the order the atomics
land in, which is a scheduling-dependent set -- the exact symptom on block 49.

## What this measures, and what it cannot

It measures the SHAPE of the moving rows, per head, against the shape the
mechanism predicts:

  a SELECTION flip     moving rows cluster inside a few TOK_GROUP*BLOCK-aligned
                       windows, because one centroid's admitted set changed and
                       every query row sharing that centroid sees it.

  anything per-token   moving rows scatter across windows, roughly one window
                       per moving row, because nothing ties them together.

The discriminator is rows-per-touched-window, whose null value is about 1.0.
The null is not asserted from theory: every arm permutes its own moving rows
within each head and recomputes the same statistic, so the run carries a
scattered control measured by the same code on the same movement.

Clustering alone would not separate a shared centroid from two neighbouring
query blocks, so `halves_stats` adds the alignment test and its shifted-grid
control; its docstring says what each grid means.

What it does NOT test is whether the unstable windows are the same set in every
launch. Each launch is compared against the first, so an unstable window can
agree with the first by chance and go uncounted; identity is the wrong test and
the record reports the pairwise overlap and the union instead.

**It cannot observe `tok_cnt`.** Whether a window's admitted count actually
exceeds `n_tok` is one out-parameter away -- the same shape as the `blk_cnt`
parameter this fork already added -- and until that exists this probe can
support or refute the mechanism's fingerprint but never confirm it. Clustering
at the centroid's granularity is consistent with a changed admitted set and
with any other per-centroid effect; a scattered result is what would refute it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Inherited, not measured: comfy_kitchen/backends/cuda/sage_attention/
# sol_layout.cuh on the branch the installed wheel was built from. BLOCK is the
# query block; TOK_GROUP is how many of them share one token-routing centroid.
# Their product is the number of contiguous token rows one selection covers.
BLOCK = 64
TOK_GROUP = 2
GROUP_ROWS = BLOCK * TOK_GROUP


def kernel_build():
    """The installed wheel's local version segment, read rather than named.

    Every comfy-kitchen build calls itself the same upstream version, so the
    local segment after the `+` is the only thing that says which one ran. This
    record's whole subject is one kernel's behaviour, so it has to carry it.
    """
    import importlib.metadata
    try:
        return importlib.metadata.version("comfy-kitchen")
    except importlib.metadata.PackageNotFoundError:
        return "<no dist metadata>"


def load_cell(path):
    """q, k, v as the kernel takes them: capture writes [B, H, S, D]."""
    import torch
    d = torch.load(path, map_location="cuda", mmap=False)
    q, k, v = (d[n].permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
               for n in ("q", "k", "v"))
    return q, k, v, {"block": d.get("block"), "step": d.get("step"),
                     "sigma": d.get("sigma"), "seq_len": q.shape[1],
                     "heads": q.shape[2]}


def moved_mask(out, first, head_chunk):
    """Per (row, head) did anything move, reduced over head_dim only.

    Chunked over heads because the tensors are ~1.5 GiB in bf16 at H3's real
    sequence length; the difference is taken in float32 so a small delta cannot
    round away into the bf16 grid it is being measured on.
    """
    import torch
    T, H = out.shape[1], out.shape[2]
    mask = torch.zeros((T, H), dtype=torch.bool)
    peak = 0.0
    for h0 in range(0, H, head_chunk):
        h1 = min(H, h0 + head_chunk)
        d = (out[0, :, h0:h1, :].float() - first[0, :, h0:h1, :].float()).abs().amax(dim=-1)
        mask[:, h0:h1] = (d > 0).cpu()
        peak = max(peak, float(d.max()))
        del d
    return mask, peak


def window_stats(mask, rows_per_window):
    """Rows per touched window, per head, and the cells they sit in.

    The statistic that discriminates: a per-centroid selection change moves many
    rows inside one window, so rows-per-touched-window is well above 1. Anything
    that moves rows independently gives about 1, because each moving row tends
    to land in a window of its own.
    """
    T, H = mask.shape
    n_windows = (T + rows_per_window - 1) // rows_per_window
    cells, rows_moved = {}, 0
    for h in range(H):
        idx = mask[:, h].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        rows_moved += int(idx.numel())
        for w in (idx // rows_per_window).tolist():
            cells[(h, w)] = cells.get((h, w), 0) + 1
    heads = {h for h, _ in cells}
    return {
        "rows_moved": rows_moved,
        "heads_with_movement": len(heads),
        "windows_touched": len(cells),
        "windows_per_head_total": n_windows,
        "rows_per_touched_window": (rows_moved / len(cells)) if cells else 0.0,
        "max_rows_in_one_window": max(cells.values()) if cells else 0,
        "window_rows": rows_per_window,
    }, cells


def halves_stats(mask, offset):
    """Do both BLOCK-sized halves of a centroid's window move together?

    The alignment test, and the reason it is not redundant with clustering.
    Clustering alone cannot separate "one centroid's set changed" from "two
    neighbouring query blocks each changed", because TOK_GROUP=2 puts those two
    query blocks in the same window either way. But a centroid's admitted set is
    SHARED by its two query blocks, so a change moves both halves; anything
    per-query-block moves one.

    Run on the aligned grid and again on a grid shifted by BLOCK: if the unit
    really is the aligned window, shifting it straddles two real windows and the
    both-halves fraction should fall. If both grids score the same, the
    alignment claim is not supported and only the clustering survives.
    """
    both = one = 0
    for h in range(mask.shape[1]):
        idx = mask[:, h].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        seen = {}
        for r in idx.tolist():
            w, off = divmod(r - offset, GROUP_ROWS)
            if r - offset < 0:
                continue
            seen.setdefault(w, set()).add(off // BLOCK)
        for halves in seen.values():
            if len(halves) == 2:
                both += 1
            else:
                one += 1
    total = both + one
    return {"windows": total, "both_halves": both,
            "frac_both_halves": (both / total) if total else 0.0}


def shuffled_stats(mask, rows_per_window, seed):
    """The same statistic on the same movement, scattered within each head.

    The deliberate violation: if this does not come back near 1.0, the
    statistic is not measuring clustering and no clustered reading from it
    means anything.
    """
    import torch
    g = torch.Generator().manual_seed(seed)
    shuffled = torch.zeros_like(mask)
    T = mask.shape[0]
    for h in range(mask.shape[1]):
        n = int(mask[:, h].sum())
        if n:
            shuffled[torch.randperm(T, generator=g)[:n], h] = True
    return window_stats(shuffled, rows_per_window)[0]


def _mean_jaccard(sets):
    """Overlap between the launches' unstable-window sets, pairwise."""
    if len(sets) < 2:
        return None
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            vals.append(len(sets[i] & sets[j]) / len(union) if union else 1.0)
    return sum(vals) / len(vals)


def run_arm(q, k, v, repeats, token_aug, tau, head_chunk, shuffle_seed,
            baseline=None):
    """One arm's stability, and optionally whether the lever engaged at all.

    `baseline` is the plain arm's output. Comparing this arm's first launch
    against it separates the two readings a deterministic result would
    otherwise conflate: token_aug ran and its selection was stable, or
    token_aug admitted nothing here and this is the plain path under another
    name. Those support opposite conclusions about a block, so a stability
    claim without this is ambiguous.
    """
    import comfy_kitchen as ck
    import torch
    first = None
    scale = 0.0
    union = None
    per_launch = []
    peak = 0.0
    for _ in range(repeats):
        kw = {"tau": tau}
        if token_aug:
            kw["token_aug"] = token_aug
        out = ck.sol_attn(q, k, v, **kw)
        if first is None:
            first = out.clone()
            scale = float(out.abs().float().mean())
        else:
            mask, p = moved_mask(out, first, head_chunk)
            peak = max(peak, p)
            per_launch.append(window_stats(mask, GROUP_ROWS)[1])
            union = mask if union is None else (union | mask)
        del out
        torch.cuda.empty_cache()

    engaged = None
    if baseline is not None and first is not None:
        bmask, bpeak = moved_mask(first, baseline, head_chunk)
        rows = int(bmask.any(dim=1).sum())
        engaged = {
            "rows_differing_from_plain": rows,
            "max_abs_delta_from_plain": bpeak,
            "lever_engaged": rows > 0,
            "what": "this arm's output against the plain arm's. If the lever "
                    "did not engage, a deterministic result below says nothing "
                    "about token_aug's stability -- it is the plain path.",
        }
        del bmask

    if union is None or not bool(union.any()):
        r = {
            "repeats": repeats, "token_aug": token_aug,
            "mean_abs_output": scale, "max_abs_delta": peak,
            "rows_moved": 0, "shape": "deterministic",
            "reading": "no element moved over any launch, so there is no "
                       "structure to have a shape",
        }
        if engaged is not None:
            r["engagement"] = engaged
            if not engaged["lever_engaged"]:
                r["shape"] = "lever inert"
                r["reading"] = ("output is bit-identical to the plain arm, so "
                                "token_aug admitted nothing here. This is not "
                                "evidence of a stable selection.")
        return r, first

    stats, _ = window_stats(union, GROUP_ROWS)
    null = shuffled_stats(union, GROUP_ROWS, shuffle_seed)
    aligned = halves_stats(union, 0)
    offset = halves_stats(union, BLOCK)
    sets = [frozenset(c) for c in per_launch]
    ratio = stats["rows_per_touched_window"]
    # Reasoned, not measured: a factor of four over the run's own scattered
    # control, and over 1.0 in absolute terms. Chosen to be far from both the
    # null and the observed values rather than tuned to either; the record
    # carries both numbers, so a reader can apply a different line.
    clustered = ratio >= 4.0 and ratio >= 4.0 * null["rows_per_touched_window"]
    result = {
        "repeats": repeats, "token_aug": token_aug,
        "mean_abs_output": scale, "max_abs_delta": peak,
        "delta_over_mean_output": peak / max(scale, 1e-9),
        **stats,
        "scattered_control": {
            "rows_per_touched_window": null["rows_per_touched_window"],
            "windows_touched": null["windows_touched"],
            "what": "the same moving rows permuted within each head. Its value "
                    "is what this statistic reports when nothing ties the rows "
                    "together, measured rather than assumed.",
        },
        "both_halves_move": {
            "aligned_grid": aligned,
            "grid_shifted_by_one_query_block": offset,
            "what": "a centroid's admitted set is shared by its TOK_GROUP "
                    "query blocks, so a changed set moves both halves of the "
                    "window. The shifted grid is the control: it straddles two "
                    "real windows, so if the aligned fraction is not the "
                    "higher of the two, the movement is not aligned to the "
                    "centroid's window and only the clustering stands.",
        },
        "windows_touched_per_launch": [len(s) for s in sets],
        "mean_pairwise_jaccard_between_launches": _mean_jaccard(sets),
        "why_not_a_set_identity_test": (
            "each launch is compared against the first, so a window whose "
            "selection is unstable can still agree with the first launch by "
            "chance and go uncounted. The union over launches is the pool of "
            "unstable windows; a single launch touches a subset of it. "
            "Identity across launches is therefore the wrong test and the "
            "overlap is reported instead."),
        "shape": "clustered at the centroid's granularity" if clustered else "scattered",
        "reading": (
            f"movement sits in {stats['windows_touched']} windows of "
            f"{GROUP_ROWS} rows at {ratio:.1f} rows per window, against "
            f"{null['rows_per_touched_window']:.1f} for the same rows "
            f"scattered. That is the granularity of one token-routing "
            f"centroid (TOK_GROUP={TOK_GROUP} query blocks of {BLOCK}), which "
            f"is what a changed admitted SET looks like from outside."
            if clustered else
            f"movement is spread at {ratio:.1f} rows per window against "
            f"{null['rows_per_touched_window']:.1f} scattered, so it does not "
            f"have the granularity of a per-centroid selection change."),
    }
    if engaged is not None:
        result["engagement"] = engaged
    return result, first


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True, help="subject qkv_*.pt")
    ap.add_argument("--also-cell", action="append", default=[],
                    help="another cell, run at the subject's budget as a control")
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--token-aug", type=int, action="append", default=[],
                    help="budget; repeatable. Default 64 only")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--head-chunk", type=int, default=4)
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()
    budgets = args.token_aug or [64]

    arms = {}
    q, k, v, meta = load_cell(args.cell)
    print(f"subject cell: block {meta['block']} step {meta['step']} "
          f"seq {meta['seq_len']} heads {meta['heads']}")
    print(f"window = TOK_GROUP {TOK_GROUP} x BLOCK {BLOCK} = {GROUP_ROWS} rows\n")

    # The plain arm first: if this moves, nothing below belongs to token_aug.
    plain_first = None
    for name, aug in [("plain_token_aug_0", 0)] + [(f"token_aug_{b}", b) for b in budgets]:
        r, out_first = run_arm(q, k, v, args.repeats, aug, args.tau, args.head_chunk,
                               args.shuffle_seed, baseline=plain_first)
        if aug == 0:
            plain_first = out_first
        arms[name] = r
        print(f"  {name:20s} rows {r['rows_moved']:6d}  "
              f"windows {r.get('windows_touched', 0):5d}  "
              f"rows/window {r.get('rows_per_touched_window', 0):6.1f}  "
              f"(scattered {r.get('scattered_control', {}).get('rows_per_touched_window', 0):.1f})  "
              f"max in one {r.get('max_rows_in_one_window', 0):4d}  "
              f"both halves {r.get('both_halves_move', {}).get('aligned_grid', {}).get('frac_both_halves', 0):.2f}"
              f" vs shifted {r.get('both_halves_move', {}).get('grid_shifted_by_one_query_block', {}).get('frac_both_halves', 0):.2f}  "
              f"-> {r['shape']}")
    del q, k, v, plain_first

    import torch
    torch.cuda.empty_cache()
    # Every control cell runs the plain arm too, so a deterministic result can
    # be told apart from a lever that never engaged there.
    for path in args.also_cell:
        cq, ckk, cv, cmeta = load_cell(path)
        _, cplain = run_arm(cq, ckk, cv, 1, 0, args.tau, args.head_chunk,
                            args.shuffle_seed)
        r, _ = run_arm(cq, ckk, cv, args.repeats, budgets[0], args.tau,
                       args.head_chunk, args.shuffle_seed, baseline=cplain)
        r["block"], r["step"] = cmeta["block"], cmeta["step"]
        arms[f"block_{cmeta['block']}_step_{cmeta['step']}_token_aug_{budgets[0]}"] = r
        eng = r.get("engagement", {})
        print(f"  block {cmeta['block']:>2} step {cmeta['step']:<3} "
              f"rows {r['rows_moved']:6d}  windows {r.get('windows_touched', 0):5d}  "
              f"engaged {eng.get('lever_engaged')} "
              f"({eng.get('rows_differing_from_plain', 0)} rows vs plain)  -> {r['shape']}")
        del cq, ckk, cv, cplain
        torch.cuda.empty_cache()

    record = {
        "what": "the shape of token_aug's run-to-run variation: whether the "
                "moving rows cluster at the granularity of one token-routing "
                "centroid, which is what a changed admitted set looks like "
                "from outside the kernel",
        "produced_by": "bench/probe_token_aug_selection_structure.py",
        "supersedes_framing_in": "data/results/2026-09-08_token_aug_determinism_shape.json",
        "subject_cell": Path(args.cell).name, "subject": meta,
        "kernel_build": kernel_build(),
        "tau": args.tau,
        "selection_unit": {
            "block": BLOCK, "tok_group": TOK_GROUP, "rows": GROUP_ROWS,
            "provenance": "inherited from comfy_kitchen sol_layout.cuh on the "
                          "branch the installed wheel was built from",
        },
        "arms": arms,
        "cannot_settle": [
            "whether a window's admitted count actually exceeds n_tok. tok_cnt "
            "is not exposed; an out-parameter in the shape of blk_cnt would "
            "settle it and does not exist.",
            "that a clustered result proves a changed admitted set. It is "
            "consistent with any per-centroid effect; a scattered result is "
            "what would have refuted the mechanism.",
            "what the SHIPPED call does. These calls pass no sink_blocks or "
            "sink_q, where the node does. Sinks change which key blocks are "
            "candidates, and the candidate pool is what an overflow is an "
            "overflow of, so a per-window result here need not carry over to a "
            "render's own selection.",
            "the size of the unstable pool. Each launch is compared against "
            "the first, so windows_touched is a lower bound that grows with "
            "repeats; the pairwise overlap beside it says how far from "
            "converged the union is. Nothing in the reading depends on the "
            "count, only on its shape.",
        ],
    }
    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
