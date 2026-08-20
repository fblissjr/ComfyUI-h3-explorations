#!/usr/bin/env python3
"""Join an exported blind scoring file with the sealed key into a verdict.

An instrument, not a check. This is **the only place the key is opened**, and
it opens it only after the scores exist: pass it a `scores_<session>.json`
exported from the batch's `score.html` and the key
`internal/blind_keys/<session>.json`, and it writes
`bench/results/<date>_<session>_verdict.json`.

## What it produces

**Pairs, which are the session's point.** Grouped by contest -- the unordered
pair of arms the stack put together -- each contest gets a verdict tally
resolved through the key, so "Clip 1 better" becomes the arm that actually sat
in slot 1, with `same` and `can't tell` counted apart. Per side, the tag counts
land on the arm that side was. Every free-text answer is kept with the pair it
came from, which arm sat in each slot, and both seeds.

A contest tally is a preference over distributions, not a per-pair verdict: a
stack presents two different samples (CLAUDE.md, the different-sample rule).
The record says so in its own `reading` field.

**Singles, per arm:** n, the notes, the tag counts, the flag count with the
clips carrying it, and -- only when the session's rubric had a `scale`
question -- the values, mean and median. Each clip's row, seed and graph are
in `by_clip`, so any number can be walked back to its render.

## Refusals

- the key's session name differs from the scores file's
- the scores file names an item that is not in the key
- the scores cover fewer items than the key, or leave a required question
  blank, unless `--partial` is passed; the missing items are named
- any absolute path would land in the record

A question is required when its type is `scale` or `choice`, or when the
rubric marks it `"required": true`. The pair's free-text field is not required
by default, so a pair the judge had nothing to say about does not block the
join.

The item universe comes from the key, which holds one entry per clip and per
pair, so the joiner runs on a box that never had the output share mounted.
`--manifest` cross-checks the key against the batch's MANIFEST when the share
is to hand.

    python bench/score_session.py --scores scores_session1.json
    python bench/score_session.py --scores s.json --key internal/blind_keys/s.json --partial
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import statistics
import sys
from pathlib import Path

_SUMMARY = (__doc__ or "").split("\n")[0]

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
KEY_DIR = REPO / "internal" / "blind_keys"
RESULTS = REPO / "bench" / "results"

SLOTS = ["Clip 1", "Clip 2"]
TIE = "same"
UNSURE = "can't tell"


def _required(q: dict) -> bool:
    return q.get("required") is True or q.get("type") in ("scale", "choice")


def _blank(v) -> bool:
    return v in (None, "", [], {}) or (isinstance(v, dict) and not v)


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


def summarise(values: list) -> dict:
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    out = {"values": values, "n": len(nums)}
    if nums:
        out["mean"] = round(statistics.fmean(nums), 3)
        out["median"] = statistics.median(nums)
    return out


def _slot_from_verdict(answer: str) -> str | None:
    """Which slot a verdict picks, or None for a tie or a shrug."""
    for slot in SLOTS:
        if answer.startswith(slot):
            return slot
    return None


def _tally_bucket(answer: str) -> str:
    return UNSURE if "can't tell" in answer or "cannot tell" in answer else TIE


def main() -> int:
    ap = argparse.ArgumentParser(description=_SUMMARY)
    ap.add_argument("--scores", required=True, help="scores_<session>.json exported from score.html")
    ap.add_argument("--key", default=None,
                    help="default internal/blind_keys/<session>.json from the scores file's session")
    ap.add_argument("--manifest", default=None,
                    help="the batch's MANIFEST.json, cross-checked against the key when given")
    ap.add_argument("--out", default=None,
                    help="default bench/results/<date>_<session>_verdict.json")
    ap.add_argument("--partial", action="store_true",
                    help="allow a scores file that does not cover every item")
    args = ap.parse_args()

    scores_path = Path(args.scores)
    scores = json.loads(scores_path.read_text())
    session = scores.get("session")
    if not session:
        sys.exit(f"refuse: {scores_path} has no session name")

    key_path = Path(args.key) if args.key else KEY_DIR / f"{session}.json"
    if not key_path.is_file():
        sys.exit(f"refuse: key not found at {key_path}")
    keydoc = json.loads(key_path.read_text())
    if keydoc.get("session") != session:
        sys.exit(f"refuse: key session {keydoc.get('session')!r} is not the scores' "
                 f"session {session!r}; these are different batches")
    key = keydoc.get("key") or {}
    if not key:
        sys.exit(f"refuse: {key_path} carries no key entries")

    clip_key = {n: e for n, e in key.items() if "label" in e}
    pair_key = {n: e for n, e in key.items() if "clip_1" in e and "clip_2" in e}
    stray = set(key) - set(clip_key) - set(pair_key)
    if stray:
        sys.exit(f"refuse: key entries that are neither a clip nor a pair: {sorted(stray)}")

    if args.manifest:
        man = json.loads(Path(args.manifest).read_text())
        if man.get("session") != session:
            sys.exit(f"refuse: manifest session {man.get('session')!r} is not {session!r}")
        m_clips = {c["clip"] for c in man.get("clips", [])}
        m_pairs = {p["pair"] for p in man.get("pairs", [])}
        if m_clips != set(clip_key) or m_pairs != set(pair_key):
            sys.exit("refuse: manifest and key disagree on the batch's items; "
                     f"only in manifest {sorted((m_clips | m_pairs) - set(key))}, "
                     f"only in key {sorted(set(key) - m_clips - m_pairs)}")

    rubric = scores.get("rubric") or []
    pair_rubric = scores.get("pair_rubric") or []
    if clip_key and not rubric:
        sys.exit("refuse: the scores file carries no rubric, so its clip answers cannot be read")
    if pair_key and not pair_rubric:
        sys.exit("refuse: the scores file carries no pair rubric, so its pair answers cannot be read")

    got_clips = scores.get("clips") or {}
    got_pairs = scores.get("pairs") or {}
    unknown = sorted((set(got_clips) - set(clip_key)) | (set(got_pairs) - set(pair_key)))
    if unknown:
        sys.exit(f"refuse: the scores name items this key does not hold: {unknown}; "
                 "the scores and the key are from different batches")

    def gaps(got: dict, universe: dict, qs: list) -> list[str]:
        out = []
        for name in sorted(universe):
            a = got.get(name)
            if a is None:
                out.append(f"{name}: not scored")
                continue
            blank = [q["id"] for q in qs if _required(q) and _blank(a.get(q["id"]))]
            if blank:
                out.append(f"{name}: blank {', '.join(blank)}")
        return out

    missing = gaps(got_pairs, pair_key, pair_rubric) + gaps(got_clips, clip_key, rubric)
    if missing and not args.partial:
        sys.exit("refuse: the scores do not cover the batch:\n  " + "\n  ".join(missing) +
                 "\npass --partial to join anyway")

    # ---- pairs, grouped by contest ---------------------------------------
    contests: dict[str, dict] = {}
    by_pair = []
    for name in sorted(pair_key):
        ent = pair_key[name]
        slots = {"Clip 1": ent["clip_1"], "Clip 2": ent["clip_2"]}
        # Not `arms`: the singles block below declares that name as a dict, and
        # one annotated name per scope wins for the whole scope.
        pair_arms = sorted({s["label"] for s in slots.values()})
        cname = " vs ".join(pair_arms)
        c = contests.setdefault(cname, {
            "arms": pair_arms, "n_pairs": 0, "n_scored": 0,
            "reading": "a preference over distributions, not a per-pair verdict: "
                       "each stack presents two different samples",
            "tally": {pair_arms[0]: 0, pair_arms[-1]: 0, TIE: 0, UNSURE: 0},
            "tags_by_arm": {}, "notes": [],
        })
        c["n_pairs"] += 1
        a = got_pairs.get(name) or {}
        row = {"pair": name, "contest": cname, "scored": bool(a),
               "clip_1": {k: slots["Clip 1"].get(k) for k in ("label", "row", "seed")},
               "clip_2": {k: slots["Clip 2"].get(k) for k in ("label", "row", "seed")}}
        if a:
            c["n_scored"] += 1
        for q in pair_rubric:
            v = a.get(q["id"])
            if _blank(v):
                continue
            if q["type"] == "choice":
                row["verdict"] = v
                slot = _slot_from_verdict(str(v))
                if slot:
                    won = slots[slot]["label"]
                    row["preferred_arm"] = won
                    c["tally"][won] = c["tally"].get(won, 0) + 1
                else:
                    bucket = _tally_bucket(str(v))
                    row["preferred_arm"] = None
                    c["tally"][bucket] = c["tally"].get(bucket, 0) + 1
            elif q["type"] == "tags":
                per_side = v if q.get("sides") else {"Clip 1": v, "Clip 2": []}
                got_tags = {}
                for slot, tags in (per_side or {}).items():
                    if slot not in slots or not tags:
                        continue
                    arm = slots[slot]["label"]
                    bucket = c["tags_by_arm"].setdefault(arm, {})
                    for t in tags:
                        bucket[t] = bucket.get(t, 0) + 1
                    got_tags[arm] = tags
                if got_tags:
                    row.setdefault("tags", {}).update(got_tags)
            elif q["type"] == "text":
                c["notes"].append({
                    "pair": name, "question": q["id"], "text": v,
                    "clip_1_arm": slots["Clip 1"]["label"], "clip_1_seed": slots["Clip 1"].get("seed"),
                    "clip_2_arm": slots["Clip 2"]["label"], "clip_2_seed": slots["Clip 2"].get("seed")})
                row.setdefault("text", {})[q["id"]] = v
            elif q["type"] == "scale":
                row.setdefault("scales", {})[q["id"]] = v
            elif q["type"] == "flag":
                row.setdefault("flags", []).append(q["id"])
        by_pair.append(row)
    for c in contests.values():
        # Both arms stay in the tally even at zero -- "v11 2, v10 0" is the
        # reading; dropping the loser makes a sweep look like a single vote.
        c["tally"] = {k: v for k, v in c["tally"].items()
                      if v or k in c["arms"]}

    # ---- singles, per arm -------------------------------------------------
    arms: dict[str, dict] = {}
    by_clip = []
    for name in sorted(clip_key):
        ent = clip_key[name]
        a = got_clips.get(name)
        prov = {"clip": name, "row": ent.get("row"), "seed": ent.get("seed"),
                "graph": ent.get("graph"), "arm": ent["label"], "scored": a is not None}
        if a:
            prov["answers"] = a
        by_clip.append(prov)
        if a is None:
            continue
        arm = arms.setdefault(ent["label"],
                              {"n": 0, "tags": {}, "flags": {}, "notes": [], "scales": {}})
        arm["n"] += 1
        for q in rubric:
            v = a.get(q["id"])
            if q["type"] == "scale":
                arm["scales"].setdefault(q["id"], []).append(v)
            elif q["type"] == "tags" and v:
                tags = v if isinstance(v, list) else [t for side in v.values() for t in side]
                for t in tags:
                    arm["tags"][t] = arm["tags"].get(t, 0) + 1
            elif q["type"] == "flag" and v:
                arm["flags"].setdefault(q["id"], []).append(
                    {"clip": name, "row": ent.get("row"), "seed": ent.get("seed")})
            elif q["type"] == "text" and v:
                arm["notes"].append({"clip": name, "row": ent.get("row"),
                                     "seed": ent.get("seed"), "question": q["id"], "text": v})
    for arm in arms.values():
        arm["scales"] = {qid: summarise(vals) for qid, vals in arm["scales"].items() if vals}
        arm["flags"] = {qid: {"count": len(v), "clips": v} for qid, v in arm["flags"].items()}
        if not arm["scales"]:
            del arm["scales"]

    record = {
        "measured": _dt.date.today().isoformat(),
        "session": session,
        "produced_by": f"bench/score_session.py joining {scores_path.name} with "
                       f"internal/blind_keys/{key_path.name}",
        "scored_at": scores.get("scored_at"),
        "jsonl": keydoc.get("jsonl"),
        "shuffle_seed": keydoc.get("shuffle_seed"),
        "partial": bool(missing),
        "not_covered": missing,
        "rubric": rubric,
        "pair_rubric": pair_rubric,
        "pairs": {"n_total": len(pair_key),
                  "n_scored": sum(1 for r in by_pair if r["scored"]),
                  "contests": contests,
                  "by_pair": by_pair} if pair_key else None,
        "clips": {"n_total": len(clip_key),
                  "n_scored": sum(1 for c in by_clip if c["scored"]),
                  "arms": arms,
                  "by_clip": by_clip} if clip_key else None,
    }
    _scrub(record)

    out = Path(args.out) if args.out else RESULTS / f"{record['measured']}_{session}_verdict.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    _print_summary(record, rubric)
    print()
    print(f"wrote {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")
    return 0


def _print_summary(record: dict, rubric: list[dict]) -> None:
    pairs = record.get("pairs")
    if pairs:
        print(f"pairs: {pairs['n_scored']} of {pairs['n_total']} scored")
        for cname in sorted(pairs["contests"]):
            c = pairs["contests"][cname]
            bits = ", ".join(f"{k} {v}" for k, v in c["tally"].items())
            print(f"  {cname}: {bits or 'no verdict given'}"
                  f"  ({c['n_scored']} of {c['n_pairs']} scored)")
            for arm in c["arms"]:
                tags = c["tags_by_arm"].get(arm)
                if tags:
                    print(f"    tags {arm}: " + ", ".join(f"{k} {v}" for k, v in sorted(tags.items())))
            n_notes = len(c["notes"])
            if n_notes:
                print(f"    {n_notes} free-text note{'' if n_notes == 1 else 's'} in the record")
        print("  a preference over distributions; each stack is two different samples")

    clips = record.get("clips")
    if not clips or not clips["arms"]:
        print("\nno scored clips")
        return
    arms = clips["arms"]
    scale_ids = [q["id"] for q in rubric if q["type"] == "scale"]
    flag_ids = [q["id"] for q in rubric if q["type"] == "flag"]
    tag_ids = sorted({t for a in arms.values() for t in a["tags"]})
    w = max(len(a) for a in arms)
    head = f"\n{'arm'.ljust(w)}  {'n':>3}"
    head += "".join(f"  {i[:9]:>9}" for i in scale_ids)
    tagw = {t: max(len(t), 3) for t in tag_ids}
    head += "".join(f"  {t:>{tagw[t]}}" for t in tag_ids)
    head += "".join(f"  {i[:6]:>6}" for i in flag_ids)
    print(head)
    print("-" * (len(head) - 1))
    for label in sorted(arms):
        a = arms[label]
        line = f"{label.ljust(w)}  {a['n']:>3}"
        for qid in scale_ids:
            s = (a.get("scales") or {}).get(qid) or {}
            line += f"  {s['mean']:>9.2f}" if "mean" in s else f"  {'-':>9}"
        for t in tag_ids:
            line += f"  {a['tags'].get(t, 0):>{tagw[t]}}"
        for qid in flag_ids:
            line += f"  {a['flags'].get(qid, {}).get('count', 0):>6}"
        print(line)
    print("counts and means only; the record holds every value, the notes, and each clip's seed")


if __name__ == "__main__":
    sys.exit(main())
