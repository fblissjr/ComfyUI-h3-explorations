#!/usr/bin/env python3
"""Set two Sol probe records side by side, block by block.

Input: two summaries written by `bench/check_sol_probe.py --record --json`
(the schema is `bench/results/2026-09-03_sol_probe_base16_standoff.json`;
read its `how_to_read` and `header.metric` first). Each carries, per block,
the mean rel_l2 of Sol's output against the chained fallback over the
compared cells, the per-segment means, the worst per-head cosine, and a
per-step trend keyed by schedule index. This tool pairs the two records'
blocks by block id and reports:

  - per block: rel_l2 mean in A, in B, and B/A; the rank of the block in
    each record's own ordering; the worst-head cosine in each; the text,
    video and audio segment means in each;
  - the Spearman rank correlation of the two block rankings, computed from
    average ranks with numpy (no scipy);
  - the per-segment means side by side, averaged over the paired blocks;
  - the per-step trend side by side, averaged over the paired blocks at
    each schedule index, with each record's sigma at that index;
  - whether the two headers name the same builds, since a different kernel
    build on one side confounds every row above.

    python bench/compare_sol_probe_records.py A.json B.json [--json OUT.json]
    python bench/compare_sol_probe_records.py --controls

The record it writes carries basenames only and refuses any absolute path
(the scrub in `bench/score_session.py`). Its `reading` field says what the
inputs' own `how_to_read` says: this ranks and compares two disagreements
between two approximations of exact attention, and does not say which side
is closer to exact.

Controls (`--controls`), run before first use on 2026-09-04 against the
standoff record, all green:

  - the record against itself: every ratio exactly one, every per-step and
    per-segment ratio exactly one, Spearman exactly one, no unpaired block;
  - the record against a copy with the block LIST shuffled, ids kept: the
    same outcome to the bit, because pairing is by block id and not by
    position;
  - the record against a copy with the block LABELS reversed (the block
    ranked first carries the stats of the block ranked last, and so on):
    Spearman exactly minus one, and the ratios are not all one;
  - the record against a copy with the block labels randomly permuted:
    Spearman below one and the per-block ratios not all one, while the
    all-block mean ratio stays one because the multiset of values did not
    change;
  - a block dropped from one side is listed as unpaired rather than
    matched to anything;
  - the scrub refuses an absolute path.

Exit 0 when the comparison ran (or every control held), 1 when a control
failed, 2 when the records cannot be paired at all.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np

SEGMENTS = ("text", "video", "audio")


def _scrub(node, where="record"):
    """Refuse rather than ship somebody's storage layout into bench/results."""
    if isinstance(node, dict):
        for k, v in node.items():
            _scrub(v, f"{where}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _scrub(v, f"{where}[{i}]")
    elif isinstance(node, str):
        if node.startswith("/") or node.startswith("~") or "/home/" in node or "/Users/" in node:
            sys.exit(f"refuse: {where} carries an absolute path: {node!r}")


def load(path: Path) -> dict:
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rec.get("blocks"), list) or not rec["blocks"]:
        sys.exit(f"{Path(path).name}: no `blocks` list; expected a check_sol_probe --json summary")
    return rec


def _builds(header: dict) -> dict:
    """The build identity: versions and git heads, never an install path,
    so two installs of one kernel at two locations count as the same build."""
    return {k: v for k, v in (header.get("builds") or {}).items() if not k.endswith("_path")}


def _identity(rec: dict, name: str) -> dict:
    """What the record was measured on, without the storage layout the
    header's `spec` carries."""
    h = rec.get("header") or {}
    rendered = [(r.get("rendered") or {}) for r in rec.get("renders") or []]
    return {
        "file": name,
        "trajectory": h.get("trajectory"),
        "capture": h.get("capture"),
        "when": h.get("when"),
        "builds": _builds(h),
        "rendered": [{k: r.get(k) for k in ("prompt_id", "prompt_sha256", "length", "canvas", "seed")}
                     for r in rendered],
        "blocks": len(rec["blocks"]),
        "cells": rec.get("cells"),
        "violations": len(rec.get("violations") or []),
    }


def _ratio(b, a):
    if a is None or b is None:
        return None
    if a == 0:
        return None
    return b / a


def average_ranks(values: np.ndarray, descending: bool = True) -> np.ndarray:
    """1-based ranks with ties given their average rank; rank 1 is the
    largest value when descending."""
    v = -values if descending else values
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=np.float64)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson correlation of the average ranks; exact 1.0 for identical
    orderings and exact -1.0 for a reversed one when there are no ties."""
    if len(x) < 2:
        return None
    rx, ry = average_ranks(x), average_ranks(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if d == 0:
        return None
    return float((rx * ry).sum() / d)


def compare(a: dict, b: dict, name_a: str, name_b: str) -> dict:
    ba = {blk["block"]: blk for blk in a["blocks"]}
    bb = {blk["block"]: blk for blk in b["blocks"]}
    paired = sorted(set(ba) & set(bb))
    if not paired:
        sys.exit(f"no block id is present in both records ({name_a}, {name_b})")
    only_a = sorted(set(ba) - set(bb)); only_b = sorted(set(bb) - set(ba))

    means_a = np.array([ba[k]["rel_l2"]["mean"] for k in paired], dtype=np.float64)
    means_b = np.array([bb[k]["rel_l2"]["mean"] for k in paired], dtype=np.float64)
    rank_a = average_ranks(means_a); rank_b = average_ranks(means_b)

    blocks = []
    for i, k in enumerate(paired):
        xa, xb = ba[k], bb[k]
        sa, sb = xa.get("segments_rel_l2_mean") or {}, xb.get("segments_rel_l2_mean") or {}
        blocks.append({
            "block": k,
            "cells": {"a": xa.get("cells"), "b": xb.get("cells")},
            "rel_l2_mean": {"a": float(means_a[i]), "b": float(means_b[i]), "ratio_b_over_a": _ratio(means_b[i], means_a[i])},
            "rank": {"a": float(rank_a[i]), "b": float(rank_b[i])},
            "worst_head_cos": {"a": xa.get("worst_head_cos"), "b": xb.get("worst_head_cos")},
            "segments_rel_l2_mean": {
                s: {"a": sa.get(s), "b": sb.get(s), "ratio_b_over_a": _ratio(sb.get(s), sa.get(s))}
                for s in SEGMENTS
            },
        })
    blocks.sort(key=lambda d: d["rank"]["a"])

    # per-segment means over the paired blocks, each side
    segments = {}
    for s in SEGMENTS:
        va = [ba[k]["segments_rel_l2_mean"].get(s) for k in paired]
        vb = [bb[k]["segments_rel_l2_mean"].get(s) for k in paired]
        va = [v for v in va if v is not None]; vb = [v for v in vb if v is not None]
        ma = float(np.mean(va)) if va else None; mb = float(np.mean(vb)) if vb else None
        segments[s] = {"a": ma, "b": mb, "ratio_b_over_a": _ratio(mb, ma), "blocks": {"a": len(va), "b": len(vb)}}

    # per-step trend: mean over paired blocks at each schedule index, each side
    def by_step(blocks_map):
        out = {}
        for k in paired:
            for st in blocks_map[k].get("per_step") or []:
                idx = st.get("schedule_index")
                if idx is None or st.get("rel_l2") is None:
                    continue
                out.setdefault(idx, {"rel_l2": [], "cos": [], "sigma": []})
                out[idx]["rel_l2"].append(st["rel_l2"])
                if st.get("cos") is not None:
                    out[idx]["cos"].append(st["cos"])
                if st.get("sigma") is not None:
                    out[idx]["sigma"].append(st["sigma"])
        return out
    sa, sb = by_step(ba), by_step(bb)
    steps = []
    for idx in sorted(set(sa) | set(sb)):
        ea, eb = sa.get(idx), sb.get(idx)
        ma = float(np.mean(ea["rel_l2"])) if ea else None
        mb = float(np.mean(eb["rel_l2"])) if eb else None
        sig_a = float(np.mean(ea["sigma"])) if ea and ea["sigma"] else None
        sig_b = float(np.mean(eb["sigma"])) if eb and eb["sigma"] else None
        steps.append({
            "schedule_index": idx,
            "sigma": {"a": sig_a, "b": sig_b,
                      "same": (None if sig_a is None or sig_b is None else bool(abs(sig_a - sig_b) <= 1e-6))},
            "rel_l2_mean": {"a": ma, "b": mb, "ratio_b_over_a": _ratio(mb, ma)},
            "cos_mean": {"a": float(np.mean(ea["cos"])) if ea and ea["cos"] else None,
                         "b": float(np.mean(eb["cos"])) if eb and eb["cos"] else None},
            "blocks": {"a": len(ea["rel_l2"]) if ea else 0, "b": len(eb["rel_l2"]) if eb else 0},
        })

    ha, hb = _builds(a.get("header") or {}), _builds(b.get("header") or {})
    build_diff = {k: {"a": ha.get(k), "b": hb.get(k)} for k in sorted(set(ha) | set(hb)) if ha.get(k) != hb.get(k)}
    all_a, all_b = float(means_a.mean()), float(means_b.mean())
    return {
        "produced_by": "bench/compare_sol_probe_records.py",
        "inputs": {"a": _identity(a, name_a), "b": _identity(b, name_b)},
        "builds_identical": not build_diff,
        "builds_differ": build_diff,
        "paired_blocks": len(paired),
        "unpaired_blocks": {"only_a": only_a, "only_b": only_b},
        "all_blocks_rel_l2_mean": {"a": all_a, "b": all_b, "ratio_b_over_a": _ratio(all_b, all_a)},
        "spearman_block_ranking": spearman(means_a, means_b),
        "blocks_ranked_by_a": blocks,
        "segments_rel_l2_mean": segments,
        "per_step": steps,
        "reading": (
            "rel_l2 in each input is Sol's output against the chained fallback (the shipped "
            "sage override) on identical q/k/v, per block and step, on the trajectory each "
            "header names. Both are approximations of exact attention, so a large value says "
            "the two disagree, not which is wrong. This record ranks and compares two such "
            "disagreements: ratio_b_over_a above one says B's Sol-versus-fallback disagreement "
            "is larger than A's on that row, and spearman_block_ranking says how far the two "
            "records order the blocks the same way. It does not say which side is closer to "
            "exact, and it does not say which record's render looked better. One scene and "
            "one seed per input; a different build on one side (builds_differ) confounds every row."
        ),
    }


def _f(v, w=8, p=4):
    return f"{'':>{w}}" if v is None else f"{v:>{w}.{p}f}"


def table(rec: dict) -> str:
    ia, ib = rec["inputs"]["a"], rec["inputs"]["b"]
    out = [f"A: {ia['file']}  ({ia['capture']}, trajectory {ia['trajectory']}, "
           f"{', '.join(str(r.get('prompt_id')) + ' seed ' + str(r.get('seed')) for r in ia['rendered']) or 'no render row'})",
           f"B: {ib['file']}  ({ib['capture']}, trajectory {ib['trajectory']}, "
           f"{', '.join(str(r.get('prompt_id')) + ' seed ' + str(r.get('seed')) for r in ib['rendered']) or 'no render row'})",
           f"builds identical: {rec['builds_identical']}" + (f"  differ: {rec['builds_differ']}" if rec["builds_differ"] else ""),
           f"paired blocks {rec['paired_blocks']}; only in A {rec['unpaired_blocks']['only_a']}; only in B {rec['unpaired_blocks']['only_b']}",
           "",
           "  block  rank A  rank B   rel_l2 A  rel_l2 B    B/A   worst-head cos A  B      text A / B          video A / B         audio A / B"]
    for d in rec["blocks_ranked_by_a"]:
        m = d["rel_l2_mean"]; w = d["worst_head_cos"]; s = d["segments_rel_l2_mean"]
        segs = "   ".join(f"{_f(s[k]['a'], 7)} / {_f(s[k]['b'], 7)}" for k in SEGMENTS)
        out.append(f"  {d['block']:5d}  {d['rank']['a']:6.1f}  {d['rank']['b']:6.1f}   {_f(m['a'])}  {_f(m['b'])}  {_f(m['ratio_b_over_a'], 6, 3)}   "
                   f"{_f(w['a'], 8)} {_f(w['b'], 8)}   {segs}")
    ab = rec["all_blocks_rel_l2_mean"]
    out += ["",
            f"  all paired blocks: rel_l2 mean A {_f(ab['a'], 0)}  B {_f(ab['b'], 0)}  B/A {_f(ab['ratio_b_over_a'], 0, 3)}",
            f"  Spearman of the block ranking: {_f(rec['spearman_block_ranking'], 0)}",
            "",
            "  segment    rel_l2 mean A   rel_l2 mean B     B/A   (blocks A / B)"]
    for k in SEGMENTS:
        s = rec["segments_rel_l2_mean"][k]
        out.append(f"  {k:8s}   {_f(s['a'], 13)}   {_f(s['b'], 13)}   {_f(s['ratio_b_over_a'], 5, 3)}   ({s['blocks']['a']} / {s['blocks']['b']})")
    out += ["", "  step   sigma A   sigma B   rel_l2 A  rel_l2 B    B/A    cos A    cos B   (blocks A / B)"]
    for st in rec["per_step"]:
        m = st["rel_l2_mean"]; c = st["cos_mean"]; sg = st["sigma"]
        flag = "" if sg["same"] in (True, None) else "  sigma differs"
        out.append(f"  {st['schedule_index']:4d}   {_f(sg['a'], 7)}   {_f(sg['b'], 7)}   {_f(m['a'])}  {_f(m['b'])}  {_f(m['ratio_b_over_a'], 6, 3)}  "
                   f"{_f(c['a'], 7)}  {_f(c['b'], 7)}   ({st['blocks']['a']} / {st['blocks']['b']}){flag}")
    out += ["", "  " + rec["reading"]]
    return "\n".join(out)


# ---------------------------------------------------------------- controls

def _all_ratios(rec: dict) -> list[float | None]:
    out = [d["rel_l2_mean"]["ratio_b_over_a"] for d in rec["blocks_ranked_by_a"]]
    out += [d["segments_rel_l2_mean"][s]["ratio_b_over_a"] for d in rec["blocks_ranked_by_a"] for s in SEGMENTS]
    out += [rec["segments_rel_l2_mean"][s]["ratio_b_over_a"] for s in SEGMENTS]
    out += [st["rel_l2_mean"]["ratio_b_over_a"] for st in rec["per_step"]]
    out.append(rec["all_blocks_rel_l2_mean"]["ratio_b_over_a"])
    return out


def _relabel(rec: dict, mapping: dict) -> dict:
    """A copy whose block `mapping[k]` carries the stats block `k` had."""
    c = copy.deepcopy(rec)
    for blk in c["blocks"]:
        blk["block"] = mapping[blk["block"]]
    return c


def controls(path: Path) -> int:
    rec = load(path)
    name = path.name
    bad = 0

    def case(label, ok, detail=""):
        nonlocal bad
        print(f"  {'ok  ' if ok else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
        bad += 0 if ok else 1

    # 1. self against self
    same = compare(rec, rec, name, name)
    ratios = _all_ratios(same)
    case("self: every ratio exactly one", all(r == 1.0 for r in ratios), f"{len(ratios)} ratios")
    case("self: Spearman exactly one", same["spearman_block_ranking"] == 1.0, f"{same['spearman_block_ranking']}")
    case("self: nothing unpaired", not same["unpaired_blocks"]["only_a"] and not same["unpaired_blocks"]["only_b"]
         and same["paired_blocks"] == len(rec["blocks"]))
    case("self: builds identical", same["builds_identical"] is True)

    # 2. the block list shuffled, ids kept: pairing is by id, so the outcome is identical
    shuffled = copy.deepcopy(rec)
    random.Random(20260904).shuffle(shuffled["blocks"])
    assert [b["block"] for b in shuffled["blocks"]] != [b["block"] for b in rec["blocks"]]
    vs_shuffled = compare(rec, shuffled, name, "shuffled.json")
    strip = lambda r: {k: v for k, v in r.items() if k != "inputs"}  # noqa: E731
    case("list shuffled, ids kept: outcome identical to self", strip(vs_shuffled) == strip(same))

    # 3. labels reversed along the ranking: Spearman exactly -1
    ranked = sorted(rec["blocks"], key=lambda d: d["rel_l2"]["mean"], reverse=True)
    ids = [d["block"] for d in ranked]
    reversed_map = dict(zip(ids, ids[::-1]))
    vs_rev = compare(rec, _relabel(rec, reversed_map), name, "reversed.json")
    case("labels reversed: Spearman exactly minus one", vs_rev["spearman_block_ranking"] == -1.0, f"{vs_rev['spearman_block_ranking']}")
    case("labels reversed: per-block ratios not all one",
         any(d["rel_l2_mean"]["ratio_b_over_a"] != 1.0 for d in vs_rev["blocks_ranked_by_a"]))
    case("labels reversed: rank B is the mirror of rank A",
         all(d["rank"]["a"] + d["rank"]["b"] == len(ids) + 1 for d in vs_rev["blocks_ranked_by_a"]))

    # 4. labels randomly permuted: Spearman below one, per-block ratios not all one, all-block mean ratio one
    perm = ids[:]
    random.Random(7).shuffle(perm)
    assert perm != ids
    vs_perm = compare(rec, _relabel(rec, dict(zip(ids, perm))), name, "permuted.json")
    case("labels permuted: Spearman below one", vs_perm["spearman_block_ranking"] < 1.0, f"{vs_perm['spearman_block_ranking']:.4f}")
    case("labels permuted: per-block ratios not all one",
         any(d["rel_l2_mean"]["ratio_b_over_a"] != 1.0 for d in vs_perm["blocks_ranked_by_a"]))
    case("labels permuted: all-block mean ratio one (same multiset)",
         abs(vs_perm["all_blocks_rel_l2_mean"]["ratio_b_over_a"] - 1.0) < 1e-12)

    # 5. a block dropped from one side is reported unpaired, not silently matched
    dropped = copy.deepcopy(rec); gone = dropped["blocks"].pop()["block"]
    vs_drop = compare(rec, dropped, name, "dropped.json")
    case("block dropped from B: listed as only_a", vs_drop["unpaired_blocks"]["only_a"] == [gone]
         and vs_drop["paired_blocks"] == len(rec["blocks"]) - 1)

    # 6. the writer refuses an absolute path
    try:
        _scrub({"x": [{"y": "/somewhere/absolute"}]}); refused = False
    except SystemExit:
        refused = True
    case("scrub refuses an absolute path", refused)

    print("\n" + ("every control held" if not bad else f"{bad} control(s) FAILED"))
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("a", nargs="?", type=Path, help="probe summary A (check_sol_probe --json)")
    ap.add_argument("b", nargs="?", type=Path, help="probe summary B")
    ap.add_argument("--json", type=Path, default=None, help="also write the comparison as a tracked record")
    ap.add_argument("--controls", action="store_true",
                    help="run the controls against A (default: the standoff record) and exit")
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    if args.controls:
        return controls(args.a or here / "results" / "2026-09-03_sol_probe_base16_standoff.json")
    if not (args.a and args.b):
        ap.print_help(); return 2
    rec = compare(load(args.a), load(args.b), args.a.name, args.b.name)
    print(table(rec))
    if args.json:
        _scrub(rec)
        args.json.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        print(f"\nrecord written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
