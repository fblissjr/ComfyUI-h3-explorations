#!/usr/bin/env python3
"""How often does the judge call a pair "same" or "can't tell", and does the slot order pull?

    python bench/tally_judge_verdicts.py [--out bench/results/<date>_judge_tie_rate.json]

Reads every `bench/results/*verdict*.json` that carries scored pair contests in
the blind-batch join's shape (`pairs.by_pair[*].verdict`, one of the pair
rubric's four answers) and counts the answers per file and in total. It also
counts wins by slot (clip 1 against clip 2, which `bench/blind_batch.py` draws
at random per pair) and gives the exact two-sided binomial probability of a
split at least that uneven if slot did not matter.

Why: the small-panel arithmetic in
`docs/research/2026-09-19_evaluation_one_judge.md` (section 0.4 and 3.3) needs
the judge's tie rate, which decides how often a five-scene rule passes a
judge who is guessing. These are contests between different arms, not decoys:
a tie here mixes the judge's habit with arms that really are alike, so the
rate is what the rule meets in practice, not the judge's false-positive rate.
That needs decoys: a panel record (`bench/join_panel_verdicts.py`) marks each
pair's kind, and its decoy verdicts are counted apart under `decoys`, where a
picked winner is a false positive. Only real contests enter the tie rate.

Panels recorded only as prose (the 2026-09-17 and 2026-09-18 records) are not
read. No GPU.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from math import comb
from pathlib import Path

import orjson

REPO = Path(__file__).resolve().parent.parent
ANSWERS = ("Clip 1 better", "Clip 2 better", "same", "can't tell")


def two_sided_binomial(k: int, n: int) -> float:
    """Exact two-sided p for k of n at p = 0.5 (sum of outcomes no more likely than k)."""
    if n == 0:
        return 1.0
    probs = [comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in probs if p <= probs[k] * (1 + 1e-12)))


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    per_file, total = [], Counter()
    decoys: dict[str, Counter] = {"two_seed_decoy": Counter(), "identical_decoy": Counter()}
    for path in sorted((REPO / "bench" / "results").glob("*verdict*.json")):
        data = orjson.loads(path.read_bytes())
        block = data.get("pairs") if isinstance(data, dict) else None
        pairs = block.get("by_pair") if isinstance(block, dict) else None
        if not isinstance(pairs, list):
            continue
        # A panel record (`bench/join_panel_verdicts.py`) marks each pair's kind. Only
        # real contests count toward the tie rate; decoy verdicts are tallied apart,
        # because a winner on a decoy is the false positive this tally cannot see otherwise.
        kind_of = {c["pair"]: c["kind"] for c in (data.get("panel") or {}).get("contests", [])}
        for p in pairs:
            k = kind_of.get(p.get("pair"), "real")
            if k in ("two_seed_decoy", "identical_decoy") and p.get("scored"):
                decoys[k][p.get("verdict")] += 1
        pairs = [p for p in pairs if kind_of.get(p.get("pair"), "real") == "real"]
        counts = Counter(p.get("verdict") for p in pairs if p.get("scored"))
        unknown = set(counts) - set(ANSWERS)
        if unknown:
            raise SystemExit(f"refuse: {path.name} has answers outside the rubric: {sorted(map(str, unknown))}")
        same_seed = sum(1 for p in pairs if p.get("scored")
                        and (p.get("clip_1") or {}).get("seed") == (p.get("clip_2") or {}).get("seed"))
        n = sum(counts.values())
        per_file.append({"file": str(path.relative_to(REPO)), "scored": n, "same_seed_pairs": same_seed,
                         "partial": data.get("partial"), **{a: counts.get(a, 0) for a in ANSWERS}})
        total += counts

    n = sum(total.values())
    if n == 0:
        raise SystemExit("refuse: no scored pair contests found; nothing to tally")
    decisive = total["Clip 1 better"] + total["Clip 2 better"]
    record = {
        "tool": "bench/tally_judge_verdicts.py",
        "files": per_file,
        "total": {a: total.get(a, 0) for a in ANSWERS},
        "scored": n,
        "tie_rate": (total["same"] + total["can't tell"]) / n,
        "same_rate": total["same"] / n,
        "cant_tell_rate": total["can't tell"] / n,
        "decisive": decisive,
        "slot_1_share_of_decisive": total["Clip 1 better"] / decisive if decisive else None,
        "slot_split_two_sided_p": two_sided_binomial(total["Clip 1 better"], decisive),
        "decoys": {k: {"scored": sum(c.values()),
                       "winner_picked": c["Clip 1 better"] + c["Clip 2 better"],
                       **{a: c.get(a, 0) for a in ANSWERS}} for k, c in decoys.items()},
    }
    text = orjson.dumps(record, option=orjson.OPT_INDENT_2)
    sys.stdout.write(text.decode() + "\n")
    if args.out:
        args.out.write_bytes(text + b"\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
