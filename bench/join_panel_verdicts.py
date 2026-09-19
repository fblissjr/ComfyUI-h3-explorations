#!/usr/bin/env python3
"""Join a scored blind PANEL with its sealed key, and apply the panel's stop rules.

    python bench/join_panel_verdicts.py --scores scores_<session>.json \\
        --reference-label fully_dense --anchor-defect-label fully_dense_with_defect \\
        [--annotations <file>] [--partial]
    python bench/join_panel_verdicts.py --self-test

A panel is built by `bench/blind_panel.py`: contests of two named clips across
scenes, each with a kind (`real`, `two_seed_decoy`, `identical_decoy`,
`low_anchor`), scored in the same `score.html` as a blind-batch session. This
is its join. Like `bench/score_session.py`, it is the only place the key is
opened, and only after the scores exist.

## Reuse, not a fork

The per-pair join is `bench/score_session.py`, run unchanged. A panel key
names each slot by its label alone, where a batch key carries a label, row and
seed, so the panel key is restated in the batch shape in a temporary directory
and `score_session.py` joins it. Its record is the base of this one: the same
`pairs.by_pair[*].verdict` that `bench/tally_judge_verdicts.py` reads. What
this adds is the `panel` block: contests by scene and kind, and the rules.

## The rules, in order (`docs/research/2026-09-19_evaluation_one_judge.md`, section 0.3)

1. Validity. The identical pair drew a winner, or the low anchor was not
   caught (the judge did not prefer the unaltered side): the session is VOID
   and nothing else is read. The two-seed decoy drew a winner: the session
   stands, but only defect-named preferences count, which is how real contests
   are counted anyway; the record says the decoy fired.
2. "Cannot tell": no real contest where the reference arm (the fully dense
   render) won with a defect named in the loser. The record states the 95%
   upper bound this puts on the per-scene detection rate for the number of
   real contests actually scored.
3. "Can tell": at least two real contests, on different scenes, where the
   reference arm won with a defect of the SAME class named in the loser.
4. Anything else: undecided at this size; add new scenes, not more seeds.

"Defect-named" is a reading of the judge's free text, which code cannot make.
Every decisive real contest gets `defect_named` (true / false / null) and
`defect_class`, filled by the reader after unblinding in an annotations file.
While any decisive real contest has `defect_named` null, no conclusion is
printed and the record says `incomplete`. On that first run the tool writes an
annotations template holding each such contest's free text.

It can fail: `--self-test` joins fixture sessions through the same code and
exits 1 unless an identical pair with a winner comes out void, a missed anchor
comes out void, a missing annotation withholds the conclusion, and the three
outcomes land where the rules put them. No GPU.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
KEY_DIR = REPO / "internal" / "blind_keys"
RESULTS = REPO / "bench" / "results"
SCORE_SESSION = HERE / "score_session.py"
KINDS = ("real", "two_seed_decoy", "identical_decoy", "low_anchor")
SLOTS = ("Clip 1", "Clip 2")
CONFIDENCE = 0.95      # normative: the level the evaluation file's bounds are stated at


def detection_bound(n: int) -> float | None:
    """Exact one-sided upper bound on a per-scene detection rate after zero detections in n scenes."""
    return None if n <= 0 else 1 - (1 - CONFIDENCE) ** (1 / n)


def batch_shape(keydoc: dict) -> dict:
    """The panel key restated as a blind-batch key: each pair slot becomes {"label", "row", "seed"}."""
    key = {}
    for name, ent in keydoc["key"].items():
        if "clip_1" in ent and "clip_2" in ent:
            ent = dict(ent)
            for slot in ("clip_1", "clip_2"):
                if isinstance(ent[slot], str):
                    ent[slot] = {"label": ent[slot], "row": None, "seed": None}
        key[name] = ent
    return {**keydoc, "key": key}


def run_score_session(scores_path: Path, keydoc: dict, partial: bool) -> dict:
    session = keydoc["session"]
    with tempfile.TemporaryDirectory() as tmp:
        key_path = Path(tmp) / f"{session}.json"      # the name score_session records
        key_path.write_text(json.dumps(batch_shape(keydoc)))
        out = Path(tmp) / "verdict.json"
        cmd = [sys.executable, str(SCORE_SESSION), "--scores", str(scores_path), "--key", str(key_path),
               "--out", str(out)] + (["--partial"] if partial else [])
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode != 0:
            sys.stderr.write(run.stdout + run.stderr)
            raise SystemExit(f"refuse: bench/score_session.py could not join these scores (exit {run.returncode})")
        return json.loads(out.read_text())


def winner(verdict: str | None) -> str | None:
    for slot in SLOTS:
        if verdict and verdict.startswith(slot):
            return slot
    return None


def panel_block(record: dict, keydoc: dict, annotations: dict, reference: str, anchor_defect: str | None) -> dict:
    key = keydoc["key"]
    rows = {r["pair"]: r for r in (record.get("pairs") or {}).get("by_pair", [])}
    notes = (annotations or {}).get("contests", {})
    contests, problems = [], []
    for pair, ent in sorted(key.items()):
        if "clip_1" not in ent:
            continue
        kind = ent.get("kind")
        if kind not in KINDS:
            raise SystemExit(f"refuse: {pair} has kind {kind!r}; this key is not a panel key")
        row = rows.get(pair, {})
        verdict = row.get("verdict")
        won_slot = winner(verdict)
        labels = {"Clip 1": ent["clip_1"], "Clip 2": ent["clip_2"]}
        c = {"contest": ent["contest"], "kind": kind, "pair": pair, "scored": bool(row.get("scored")),
             "verdict": verdict, "preferred_arm": labels[won_slot] if won_slot else None,
             "text": row.get("text", {}), "note": ent.get("note", ""),
             "graph_differences_a_to_b": ent.get("graph_differences_a_to_b", {})}
        if kind == "real":
            n = notes.get(ent["contest"], {})
            c["defect_named"] = n.get("defect_named")
            c["defect_class"] = n.get("defect_class")
            c["decisive"] = won_slot is not None
        if kind == "low_anchor":
            sides = set(labels.values())
            defect = ent.get("b_label") or anchor_defect
            if defect is None or defect not in sides:
                problems.append(f"low anchor {ent['contest']}: say which label is the altered side "
                                f"(--anchor-defect-label; the slots are {sorted(sides)})")
            c["defect_label"] = defect
            c["caught"] = (won_slot is not None and c["preferred_arm"] != defect) if c["scored"] else None
        contests.append(c)

    real = [c for c in contests if c["kind"] == "real"]
    identical = [c for c in contests if c["kind"] == "identical_decoy"]
    anchors = [c for c in contests if c["kind"] == "low_anchor"]
    decoys = [c for c in contests if c["kind"] == "two_seed_decoy"]
    labels_seen = {c["preferred_arm"] for c in real if c["preferred_arm"]} | \
                  {lab for c in real for lab in (key[c["pair"]]["clip_1"], key[c["pair"]]["clip_2"])}
    if real and reference not in labels_seen:
        problems.append(f"--reference-label {reference!r} is not an arm of any real contest ({sorted(labels_seen)})")

    validity = {
        "identical_pair_drew_a_winner": any(c["preferred_arm"] for c in identical),
        "low_anchor_missed": any(c.get("caught") is False for c in anchors),
        "two_seed_decoy_fired": any(c["preferred_arm"] for c in decoys),
        "controls_present": {"identical_decoy": len(identical), "low_anchor": len(anchors),
                             "two_seed_decoy": len(decoys)},
    }
    scored_real = [c for c in real if c["scored"]]
    unannotated = [c["contest"] for c in scored_real if c["decisive"] and c["defect_named"] is None]
    unscored = [c["contest"] for c in contests if not c["scored"]]
    reference_wins = [c for c in scored_real if c["preferred_arm"] == reference and c["defect_named"] is True]
    classes: dict[str, set] = {}
    for c in reference_wins:
        if c["defect_class"]:
            classes.setdefault(c["defect_class"], set()).add(c["contest"])
    n = len(scored_real)
    cant = sum(1 for c in scored_real if c["verdict"] and "can't tell" in c["verdict"])

    if problems:
        conclusion, why = "refused", problems
    elif validity["identical_pair_drew_a_winner"] or validity["low_anchor_missed"]:
        conclusion = "void"
        why = [w for w, hit in (("the identical pair drew a winner", validity["identical_pair_drew_a_winner"]),
                                ("the low anchor was not caught", validity["low_anchor_missed"])) if hit]
    elif unscored or unannotated:
        conclusion = "incomplete"
        why = ([f"unscored: {unscored}"] if unscored else []) + \
              ([f"decisive real contests with defect_named still null: {unannotated}"] if unannotated else [])
    elif any(len(v) >= 2 for v in classes.values()):
        conclusion = "can_tell"
        why = [f"{reference} won with {cls!r} named on {sorted(v)}" for cls, v in classes.items() if len(v) >= 2]
    elif not reference_wins:
        conclusion = "cannot_tell"
        why = [f"no real contest where {reference} won with a defect named, in {n} scored"]
    else:
        conclusion = "undecided"
        why = [f"{len(reference_wins)} defect-named win(s) for {reference}, not two of one class; "
               f"add {n} new scenes and apply the rules again"]
    return {
        "reference_label": reference,
        "contests": contests,
        "validity": validity,
        "n_real_scored": n,
        "reference_wins_defect_named": [c["contest"] for c in reference_wins],
        "defect_classes": {k: sorted(v) for k, v in classes.items()},
        "cant_tell_real": cant,
        "scenes_failed": n > 0 and cant * 2 >= n,
        "bound_if_no_detection": {"confidence": CONFIDENCE, "scenes": n,
                                  "per_scene_detection_rate_below": detection_bound(n),
                                  "if_doubled_scenes": 2 * n,
                                  "at_double_below": detection_bound(2 * n)},
        "conclusion": conclusion,
        "why": why,
        "rules": "docs/research/2026-09-19_evaluation_one_judge.md, section 0.3",
    }


def annotation_template(panel: dict) -> dict:
    return {"contests": {c["contest"]: {"defect_named": None, "defect_class": None,
                                        "judge_text": c["text"], "verdict": c["verdict"],
                                        "preferred_arm": c["preferred_arm"]}
                         for c in panel["contests"] if c["kind"] == "real" and c.get("decisive")}}


def join(scores_path: Path, keydoc: dict, annotations: dict, reference: str, anchor_defect: str | None,
         partial: bool) -> dict:
    record = run_score_session(scores_path, keydoc, partial)
    record["joined_by"] = record.pop("produced_by", None)
    record["produced_by"] = "bench/join_panel_verdicts.py over bench/score_session.py"
    record["panel"] = panel_block(record, keydoc, annotations, reference, anchor_defect)
    return record


def report(record: dict) -> None:
    p = record["panel"]
    print(f"session {record['session']}: {p['conclusion'].upper()}")
    for w in p["why"]:
        print(f"  {w}")
    v = p["validity"]
    print(f"  validity: identical pair winner {v['identical_pair_drew_a_winner']}, anchor missed "
          f"{v['low_anchor_missed']}, two-seed decoy fired {v['two_seed_decoy_fired']}")
    b = p["bound_if_no_detection"]
    if b["per_scene_detection_rate_below"] is not None:
        print(f"  zero detections in {b['scenes']} scenes would bound the per-scene detection rate below "
              f"{b['per_scene_detection_rate_below']:.3f} at {b['confidence']:.0%} "
              f"({b['at_double_below']:.3f} at {b['if_doubled_scenes']})")
    for c in p["contests"]:
        extra = f" defect_named={c.get('defect_named')} class={c.get('defect_class')}" if c["kind"] == "real" else ""
        print(f"  {c['kind']:16s} {c['contest']:28s} {str(c['verdict']):14s} -> {c['preferred_arm']}{extra}")


# ------------------------------------------------------------------ self-test

def _fixture(tmp: Path, verdicts: dict) -> tuple[Path, dict]:
    """A panel of six real contests and three controls, scored with the given verdicts by contest."""
    names = [f"scene{i}" for i in range(1, 7)]
    key, pairs = {}, {}
    spec = [(n, "real", "default", "fully_dense") for n in names] + [
        ("decoy", "two_seed_decoy", "dense_seed1", "dense_seed2"),
        ("identical", "identical_decoy", "fully_dense", "fully_dense"),
        ("anchor", "low_anchor", "fully_dense", "fully_dense_with_defect")]
    for i, (contest, kind, a, b) in enumerate(spec, 1):
        pair = f"pair_{i:02d}.mp4"
        key[pair] = {"contest": contest, "kind": kind, "note": "", "clip_1": a, "clip_2": b,
                     "graph_differences_a_to_b": {}}
        v = verdicts.get(contest, "same")
        pairs[pair] = {"verdict": v, "differs": f"fixture text for {contest}"}
    keydoc = {"session": "selftest", "shuffle_seed": 1, "key": key}
    scores = {"session": "selftest", "rubric": [{"id": "note", "type": "text"}],
              "pair_rubric": [{"id": "differs", "type": "text"},
                              {"id": "verdict", "type": "choice",
                               "options": ["Clip 1 better", "Clip 2 better", "same", "can't tell"]}],
              "clips": {}, "pairs": pairs}
    path = tmp / "scores_selftest.json"
    path.write_text(json.dumps(scores))
    return path, keydoc


def self_test() -> int:
    ok_anchor = {"anchor": "Clip 1 better"}               # slot 1 is the unaltered side in the fixture
    wins = {"scene1": "Clip 2 better", "scene2": "Clip 2 better"}   # slot 2 is fully_dense
    all_notes = {f"scene{i}": {"defect_named": False, "defect_class": None} for i in range(1, 7)}
    cases = [
        ("identical pair with a winner is void", {**ok_anchor, "identical": "Clip 2 better"}, None, "void"),
        ("missed anchor is void", {"anchor": "same"}, None, "void"),
        ("anchor preferring the altered side is void", {"anchor": "Clip 2 better"}, None, "void"),
        ("decisive contest without annotation withholds the conclusion", {**ok_anchor, **wins}, None, "incomplete"),
        ("no defect-named win for dense is cannot_tell", {**ok_anchor, **wins}, {"contests": all_notes},
         "cannot_tell"),
        ("two same-class wins for dense is can_tell", {**ok_anchor, **wins},
         {"contests": {**all_notes, "scene1": {"defect_named": True, "defect_class": "morph"},
                       "scene2": {"defect_named": True, "defect_class": "morph"}}}, "can_tell"),
        ("two different-class wins is undecided", {**ok_anchor, **wins},
         {"contests": {**all_notes, "scene1": {"defect_named": True, "defect_class": "morph"},
                       "scene2": {"defect_named": True, "defect_class": "flicker"}}}, "undecided"),
        ("a firing two-seed decoy does not void", {**ok_anchor, "decoy": "Clip 1 better"}, {"contests": all_notes},
         "cannot_tell"),
    ]
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for label, verdicts, notes, want in cases:
            scores, keydoc = _fixture(Path(tmp), verdicts)
            got = join(scores, keydoc, notes or {}, "fully_dense", "fully_dense_with_defect", False)["panel"]
            hit = got["conclusion"] == want
            bad += not hit
            print(f"  {'ok  ' if hit else 'FAIL'}  {label}: {got['conclusion']}"
                  f"{'' if hit else f' (expected {want})'}")
        bound = detection_bound(6)
        print(f"  info  zero detections in 6 scenes bound the rate below {bound:.3f}")
    print("self-test " + ("passed" if not bad else f"FAILED ({bad})"))
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--scores", type=Path)
    ap.add_argument("--key", type=Path, help="default internal/blind_keys/<session>.json")
    ap.add_argument("--annotations", type=Path,
                    help="reader's defect_named / defect_class per real contest; default "
                         "internal/blind_keys/<session>_annotations.json, written as a template when absent")
    ap.add_argument("--reference-label", default="fully_dense",
                    help="the arm whose defect-named win means the other arm is visibly worse")
    ap.add_argument("--anchor-defect-label", default=None,
                    help="the label of the low anchor's altered side, when the key does not record it")
    ap.add_argument("--partial", action="store_true", help="passed to score_session.py")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.scores:
        ap.error("--scores is required (or --self-test)")

    session = json.loads(args.scores.read_text()).get("session")
    key_path = args.key or KEY_DIR / f"{session}.json"
    if not key_path.is_file():
        raise SystemExit(f"refuse: no key at {key_path.name} for session {session!r}")
    keydoc = json.loads(key_path.read_text())
    ann_path = args.annotations or KEY_DIR / f"{session}_annotations.json"
    annotations = json.loads(ann_path.read_text()) if ann_path.is_file() else {}

    record = join(args.scores, keydoc, annotations, args.reference_label, args.anchor_defect_label, args.partial)
    out = args.out or RESULTS / f"{_dt.date.today().isoformat()}_{session}_verdict.json"
    out.write_text(json.dumps(record, indent=1))
    report(record)
    if record["panel"]["conclusion"] == "incomplete" and not ann_path.is_file():
        ann_path.write_text(json.dumps(annotation_template(record["panel"]), indent=1))
        print(f"wrote the annotations template {ann_path.name}: fill defect_named and defect_class, then re-run")
    print(f"wrote {out.relative_to(REPO) if out.is_relative_to(REPO) else out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
