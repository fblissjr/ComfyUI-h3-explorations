#!/usr/bin/env python3
"""Speed beside what the owner noticed, per scene, per arm. Never pass/fail.

"It's not black or white, it's a whole bunch of grey in the middle." This
table puts each arm's sampler time next to every verdict and every note the
owner gave it, scene by scene, and stops there.

Inputs: one or more verdict records written by `bench/score_session.py`
(`pairs.by_pair[]` carries the per-pair verdict, the two arms and the
owner's free text), and one or more outputs records carrying per-arm timings
(`arms[].label`, `sampler_s`, `total_s`, `decode_s`, `seed`, `graph`, an
optional `note`; `bench/build_outputs_record.py` writes that shape). Arm
labels are `<scene>_<rung>`, split on the last underscore.

    python bench/frontier_table.py --verdict V.json [--verdict ...] \\
        --outputs O.json [--outputs ...] [--floor sage] [--dense-rung dense] \\
        [--json OUT.json]
    python bench/frontier_table.py --controls

Per scene, one row per arm:

  - sampler time, total time, and the arm's timing note verbatim;
  - speed relative to the scene's FLOOR arm (`--floor`, default `sage`,
    the owner's decision of 2026-09-04 that sage is always on), as floor
    sampler seconds over the arm's; and relative to the scene's dense arm
    when one exists, the reference the owner thinks in;
  - the verdicts the arm received across its contests, resolved from each
    pair's `preferred_arm` and `verdict` into preferred, not preferred,
    same and can't tell;
  - every pair the arm appeared in, prefixed with the opposing arm, with
    the pair's verdict and the owner's free text verbatim; a pair scored
    without text says so rather than vanishing.

An arm with a verdict but no timing, or a timing but no verdict, is listed
with the missing side marked, never dropped. A `--floor` naming a rung no
arm of a scene carries refuses, naming the scene. The record carries
basenames only and refuses an absolute path.

The record's `reading`: speed is one clip per arm from the named outputs
record, in whatever cache state that record says; the verdicts and notes are
one judge at one seed, read blind through score.html and joined through the
sealed key; a count here is a count of pairs, not a distribution.

Controls (`--controls`), run before first use on 2026-09-04 against the
2026-09-03 ladder (its verdict record and its outputs record), all green:
the floor row of every scene has speed one against itself; the standoff pdd8
row carries the owner's "totally different scenes" note against dense; the
sage row of every scene but subway shows only same or can't tell against
dense (one of those pairs was scored with no text, which is why the
verdict is read from the pair and not from a note); a floor absent from a scene refuses naming the scene; an arm dropped
from the outputs stays in the table with its timing marked missing, and an
arm dropped from the verdict stays with its verdicts marked missing; the
same label in two outputs records refuses.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

VERDICT_BUCKETS = ("preferred", "not_preferred", "same", "cant_tell")
# inherited: the ladder's labels and the baseline rule in CLAUDE.md call the
# stock-attention arm `dense`; `--dense-rung` overrides it.
DENSE_RUNG = "dense"
# decided: the owner, 2026-09-04, sage is always on and every speed is read
# against it (docs/roadmap.md, forward plan 2026-09-04).
FLOOR_RUNG = "sage"


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


def split_label(label: str) -> tuple[str, str]:
    """`<scene>_<rung>` on the last underscore; a label without one is its
    own scene with an empty rung, which the table shows as such."""
    if "_" not in label:
        return label, ""
    scene, rung = label.rsplit("_", 1)
    return scene, rung


def _bucket(pair: dict, arm: str) -> str | None:
    """Which of the four buckets this pair is for `arm`, from the joined
    row's `preferred_arm` and `verdict`; None when the pair was not scored."""
    if not pair.get("scored"):
        return None
    pref = pair.get("preferred_arm")
    if pref == arm:
        return "preferred"
    if pref is not None:
        return "not_preferred"
    v = str(pair.get("verdict") or "").strip().lower()
    if v == "same":
        return "same"
    if v.startswith("can"):
        return "cant_tell"
    return None


def load_verdicts(paths: list[Path]) -> dict:
    """Per arm: bucket counts, the contests it sat in, and every pair note."""
    arms: dict[str, dict] = {}
    sources = []
    for p in paths:
        rec = json.loads(p.read_text(encoding="utf-8"))
        by_pair = ((rec.get("pairs") or {}).get("by_pair")) or []
        if not by_pair:
            sys.exit(f"{p.name}: no pairs.by_pair; expected a score_session verdict record")
        sources.append({"file": p.name, "session": rec.get("session"), "measured": rec.get("measured"),
                        "scored_at": rec.get("scored_at"), "pairs": len(by_pair)})
        for pair in by_pair:
            c1, c2 = pair.get("clip_1") or {}, pair.get("clip_2") or {}
            a, b = c1.get("label"), c2.get("label")
            if not a or not b:
                continue
            text = (pair.get("text") or {}).get("differs")
            for arm, other, seed in ((a, b, c1.get("seed")), (b, a, c2.get("seed"))):
                entry = arms.setdefault(arm, {
                    "counts": dict.fromkeys(VERDICT_BUCKETS, 0), "unscored": 0,
                    "contests": set(), "seeds": set(), "pairs": [], "sources": set()})
                entry["contests"].add(pair.get("contest") or f"{a} vs {b}")
                entry["sources"].add(p.name)
                if seed is not None:
                    entry["seeds"].add(seed)
                bucket = _bucket(pair, arm)
                if bucket is None:
                    entry["unscored"] += 1
                else:
                    entry["counts"][bucket] += 1
                entry["pairs"].append({
                    "against": other, "pair": pair.get("pair"), "verdict": pair.get("verdict"),
                    "bucket": bucket, "this_arm_was": "clip 1" if arm == a else "clip 2",
                    "text": text or None, "source": p.name})
    for e in arms.values():
        e["contests"] = sorted(e["contests"]); e["seeds"] = sorted(e["seeds"]); e["sources"] = sorted(e["sources"])
    return {"arms": arms, "sources": sources}


def load_outputs(paths: list[Path]) -> dict:
    """Per arm: the timing fields and the note, from one record each."""
    arms: dict[str, dict] = {}
    sources = []
    for p in paths:
        rec = json.loads(p.read_text(encoding="utf-8"))
        rows = rec.get("arms")
        if not isinstance(rows, list) or not rows:
            sys.exit(f"{p.name}: no arms list; expected an outputs record")
        sources.append({"file": p.name, "what": rec.get("what"), "arms": len(rows),
                        "timing_note": (rec.get("timing") or {}).get("note")})
        for r in rows:
            label = r.get("label")
            if not label:
                continue
            if label in arms:
                sys.exit(f"refuse: arm {label!r} appears in both {arms[label]['source']} and {p.name}; "
                         f"one timing per arm, pick the record")
            arms[label] = {
                "sampler_s": r.get("sampler_s"), "total_s": r.get("total_s"), "decode_s": r.get("decode_s"),
                "seed": r.get("seed"), "graph": r.get("graph"), "note": r.get("note") or None,
                "source": p.name,
            }
    return {"arms": arms, "sources": sources}


def _ratio(num, den):
    if num is None or den is None or den == 0:
        return None
    return num / den


def build(verdicts: dict, outputs: dict, floor: str, dense_rung: str) -> dict:
    v_arms, o_arms = verdicts["arms"], outputs["arms"]
    labels = sorted(set(v_arms) | set(o_arms))
    scenes: dict[str, list[str]] = {}
    for lab in labels:
        scenes.setdefault(split_label(lab)[0], []).append(lab)

    out_scenes = []
    for scene in sorted(scenes):
        arms = scenes[scene]
        floor_arm = f"{scene}_{floor}"
        if floor_arm not in arms:
            sys.exit(f"refuse: scene {scene!r} has no {floor!r} arm to read speed against "
                     f"(its arms: {', '.join(arms)}); pick --floor from a rung every scene carries")
        dense_arm = f"{scene}_{dense_rung}"
        if dense_arm not in arms:
            dense_arm = None
        floor_s = (o_arms.get(floor_arm) or {}).get("sampler_s")
        dense_s = (o_arms.get(dense_arm) or {}).get("sampler_s") if dense_arm else None

        rows = []
        for lab in arms:
            rung = split_label(lab)[1]
            t = o_arms.get(lab)
            v = v_arms.get(lab)
            missing = []
            if t is None:
                missing.append("timing: no arm with this label in any outputs record")
            elif t.get("sampler_s") is None:
                missing.append("timing: sampler_s is null" + (f" ({t['note']})" if t.get("note") else ""))
            if v is None:
                missing.append("verdicts: no pair with this label in any verdict record")
            sampler = t.get("sampler_s") if t else None
            rows.append({
                "arm": lab, "rung": rung,
                "timing": t,
                "speed_vs_floor": _ratio(floor_s, sampler),
                "speed_vs_dense": _ratio(dense_s, sampler) if dense_arm else None,
                "verdicts": ({"counts": v["counts"], "unscored_pairs": v["unscored"], "contests": v["contests"],
                              "seeds": v["seeds"]} if v else None),
                "pairs": v["pairs"] if v else [],
                "missing": missing,
            })
        rows.sort(key=lambda r: ((r["timing"] or {}).get("sampler_s") is None,
                                 (r["timing"] or {}).get("sampler_s") or 0.0))
        out_scenes.append({
            "scene": scene, "floor_arm": floor_arm, "dense_arm": dense_arm,
            "floor_sampler_s": floor_s, "dense_sampler_s": dense_s,
            "floor_speed_missing": floor_s is None,
            "rows": rows,
        })

    return {
        "produced_by": "bench/frontier_table.py",
        "inputs": {"verdicts": verdicts["sources"], "outputs": outputs["sources"]},
        "floor_rung": floor, "dense_rung": dense_rung,
        "speed": "sampler seconds of the scene's floor arm over the arm's own; above one is faster than the floor. "
                 "speed_vs_dense is the same against the scene's dense arm when one exists",
        "verdict_buckets": {"preferred": "the pair's preferred_arm is this arm",
                            "not_preferred": "the pair's preferred_arm is the other arm",
                            "same": "the judge answered same", "cant_tell": "the judge answered can't tell"},
        "scenes": out_scenes,
        "reading": (
            "Speed is one clip per arm from the named outputs record, in whatever cache state that "
            "record's timing note says it was rendered in; a null sampler time is shown as missing, "
            "never estimated. The verdicts and notes are one judge at one seed, read blind through "
            "score.html and joined through the sealed key by bench/score_session.py; a count here is a "
            "count of pairs the arm sat in, not a preference over a distribution, and a note is the "
            "owner's sentence about one stack against one opposing clip. Nothing here passes or fails."
        ),
    }


def _f(v, w=7, p=1, suffix=""):
    return f"{'':>{w}}" if v is None else f"{v:>{w}.{p}f}{suffix}"


def table(rec: dict) -> str:
    out = [f"floor rung {rec['floor_rung']!r}, dense rung {rec['dense_rung']!r}; "
           f"verdicts from {', '.join(s['file'] for s in rec['inputs']['verdicts'])}; "
           f"timings from {', '.join(s['file'] for s in rec['inputs']['outputs'])}"]
    for sc in rec["scenes"]:
        out += ["", f"== {sc['scene']}   floor {sc['floor_arm']}"
                    + (f", dense {sc['dense_arm']}" if sc["dense_arm"] else ", no dense arm")
                    + ("   FLOOR TIMING MISSING" if sc["floor_speed_missing"] else ""),
                "  rung        sampler s   total s   x floor   x dense   pref  not  same  can't   notes/missing"]
        for r in sc["rows"]:
            t = r["timing"] or {}
            v = (r["verdicts"] or {}).get("counts")
            counts = ("  ".join(f"{v[k]:>4d}" for k in VERDICT_BUCKETS) if v else
                      "   -     -     -     -")
            out.append(f"  {r['rung'] or '(none)':<10}  {_f(t.get('sampler_s'), 9)}  {_f(t.get('total_s'), 8)}  "
                       f"{_f(r['speed_vs_floor'], 7, 2, 'x')}  {_f(r['speed_vs_dense'], 7, 2, 'x')}   {counts}")
            for m in r["missing"]:
                out.append(f"      MISSING {m}")
            if t.get("note"):
                out.append(f"      timing note: {t['note']}")
            for n in r["pairs"]:
                out.append(f"      vs {split_label(n['against'])[1] or n['against']} ({n['pair']}, "
                           f"this arm was {n['this_arm_was']}, verdict {n['verdict']}): "
                           f"{n['text'] if n['text'] else '(no note written)'}")
    out += ["", "  " + rec["reading"]]
    return "\n".join(out)


# ------------------------------------------------------------------ controls

def controls() -> int:
    verdict = RESULTS / "2026-09-04_ladder_2026-09-03_verdict.json"
    outputs = RESULTS / "2026-09-03_ladder_outputs.json"
    bad = 0

    def case(label, ok, detail=""):
        nonlocal bad
        print(f"  {'ok  ' if ok else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
        bad += 0 if ok else 1

    V, O = load_verdicts([verdict]), load_outputs([outputs])
    rec = build(V, O, FLOOR_RUNG, DENSE_RUNG)
    _scrub(rec)
    by_scene = {s["scene"]: s for s in rec["scenes"]}
    row = lambda scene, rung: next(r for r in by_scene[scene]["rows"] if r["rung"] == rung)  # noqa: E731

    case("every scene's floor row has speed one against itself",
         all(row(s, FLOOR_RUNG)["speed_vs_floor"] == 1.0 for s in by_scene), f"{len(by_scene)} scenes")
    pdd = row("standoff", "pdd8")
    notes_vs_dense = [n for n in pdd["pairs"] if split_label(n["against"])[1] == DENSE_RUNG]
    case("standoff pdd8 carries the owner's 'totally different scenes' note against dense",
         any("totally different scenes" in (n["text"] or "") for n in notes_vs_dense),
         f"{len(notes_vs_dense)} note(s) against dense")
    sage_ok = {}
    for s in by_scene:
        if s == "subway":
            continue
        vs = [n["verdict"] for n in row(s, FLOOR_RUNG)["pairs"] if split_label(n["against"])[1] == DENSE_RUNG]
        sage_ok[s] = bool(vs) and all(str(x).lower() in ("same", "can't tell") for x in vs)
    case("sage against dense reads same or can't tell on every scene but subway",
         all(sage_ok.values()), ", ".join(f"{s}: {'ok' if ok else 'NOT'}" for s, ok in sage_ok.items()))
    try:
        build(V, O, "nonesuch", DENSE_RUNG); msg = None
    except SystemExit as e:
        msg = str(e)
    case("a floor absent from a scene refuses naming the scene",
         msg is not None and "scene" in msg and any(s in msg for s in by_scene), "" if msg else "did not refuse")
    O2 = copy.deepcopy(O); del O2["arms"]["diner_sol"]
    r2 = build(V, O2, FLOOR_RUNG, DENSE_RUNG)
    d = next(r for r in next(s for s in r2["scenes"] if s["scene"] == "diner")["rows"] if r["rung"] == "sol")
    case("an arm dropped from the outputs stays, timing marked missing, notes intact",
         d["timing"] is None and any(m.startswith("timing") for m in d["missing"]) and d["pairs"])
    V2 = copy.deepcopy(V); del V2["arms"]["opera_solfp16"]
    r3 = build(V2, O, FLOOR_RUNG, DENSE_RUNG)
    d = next(r for r in next(s for s in r3["scenes"] if s["scene"] == "opera")["rows"] if r["rung"] == "solfp16")
    case("an arm dropped from the verdict stays, verdicts marked missing, timing intact",
         d["verdicts"] is None and any(m.startswith("verdicts") for m in d["missing"]) and d["timing"])
    sd = row("standoff", DENSE_RUNG)
    case("a null sampler time is marked missing with the record's note, not estimated",
         sd["speed_vs_floor"] is None and any("sampler_s is null" in m for m in sd["missing"]))
    try:
        load_outputs([outputs, outputs]); dup = None
    except SystemExit as e:
        dup = str(e)
    case("the same label in two outputs records refuses", dup is not None and "both" in dup)
    try:
        _scrub({"x": "/somewhere"}); refused = False
    except SystemExit:
        refused = True
    case("scrub refuses an absolute path", refused)
    print("\n" + ("every control held" if not bad else f"{bad} control(s) FAILED"))
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--verdict", action="append", type=Path, default=[], help="a score_session verdict record; repeatable")
    ap.add_argument("--outputs", action="append", type=Path, default=[], help="an outputs record with per-arm timings; repeatable")
    ap.add_argument("--floor", default=FLOOR_RUNG, help=f"the rung every speed is relative to (default {FLOOR_RUNG})")
    ap.add_argument("--dense-rung", default=DENSE_RUNG, help=f"the rung shown as the second reference (default {DENSE_RUNG})")
    ap.add_argument("--json", type=Path, default=None, help="also write the table as a tracked record")
    ap.add_argument("--controls", action="store_true", help="run the controls against the 2026-09-03 ladder and exit")
    args = ap.parse_args()
    if args.controls:
        return controls()
    if not args.verdict or not args.outputs:
        ap.print_help(); return 2
    rec = build(load_verdicts(args.verdict), load_outputs(args.outputs), args.floor, args.dense_rung)
    print(table(rec))
    if args.json:
        _scrub(rec)
        args.json.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        print(f"\nrecord written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
