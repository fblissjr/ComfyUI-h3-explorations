#!/usr/bin/env python3
"""Select the deterministic text-only T2VA regression holdout.

`active_plan.md`: text-only rows are excluded from the vision-traced `oneshot`
calibration run and remain a held-out regression population, so that a v2
candidate can be shown not to have lost text-only behaviour while gaining
multimodal behaviour. This picks that population; the bundle is then built with
`build_native_h3_calibration_batch.py --population text-only`.

**Disjointness here is not the vision holdout's disjointness, and pretending it
is would be the emptiest kind of green.** These rows carry no media at all, so
"no shared media file" and "no shared visual family" are true by construction
and assert nothing -- a check whose input already satisfies it cannot fail. The
axis that carries information is the prompt:

- exact prompt hash, and the pool builder's normalised prompt hash, unique
  within the population and absent from calibration and from the vision
  holdout;
- a near-duplicate prompt review by 8-gram Jaccard, both across the split and
  within the selected set, reported as a ranked list rather than gated on a
  threshold, for the reason the pool-wide media review records: a threshold
  tight enough to miss nothing fires on this source's shared house format.

Every H3-IR target_ir shares a section skeleton, so a raw Jaccard floor is
high for unrelated rows. The report therefore states the population's own
background level beside the top pairs, so a reader can tell a real overlap from
the format everything shares.

Stratified on the axes that distinguish text-only rows from each other:
dialogue markers present or absent, contract duration, and prompt length.

Deterministic under `--seed`. CPU only, no CUDA, no model, no server.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))

EXCLUDED = BENCH / "results" / "archive" / "v2_encoder" / "2026-08-24_h3_calibration_pool_excluded.jsonl"
DURATION = re.compile(r"duration_seconds: (\d+)")


def shingles(text: str, k: int = 8) -> set:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(words[i:i + k]) for i in range(max(0, len(words) - k + 1))}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def contract_duration(row: dict) -> str:
    users = [m for m in row.get("messages", []) if m["role"] == "user"]
    if not users:
        return "?"
    found = DURATION.search(users[0]["content"])
    return found.group(1) if found else "?"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excluded", type=Path, default=EXCLUDED)
    parser.add_argument("--rows", type=int, default=13,
                        help="sized to match the vision holdout")
    parser.add_argument("--bundle", action="append", default=[], type=Path,
                        help="calibration or vision-holdout bundle to compare "
                             "prompts against; repeatable")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--out", type=Path,
                        default=BENCH / "results" / "archive" / "v2_encoder" / "2026-08-25_t2va_holdout_rows.json")
    args = parser.parse_args()

    from build_h3_calibration_pool import pinned_snapshot

    root, revision = pinned_snapshot()
    population = [json.loads(line)
                  for line in args.excluded.read_text().splitlines()]
    stray = [r["id"] for r in population if r["primary_role"] != "no-vision"]
    if stray:
        print(f"the exclusion file carries vision-bearing rows: {stray[:5]}")
        return 2

    source = {}
    for line in (root / "data" / "train.jsonl").read_text().splitlines():
        record = json.loads(line)
        source[record["id"]] = record

    # every row the other bundles present, by prompt
    other_ids: dict[str, list[str]] = {}
    for path in args.bundle:
        manifest = path / "presentation.json"
        if not manifest.is_file():
            print(f"{path} has no presentation.json")
            return 2
        other_ids[path.name] = [r["row_id"]
                                for r in json.loads(manifest.read_text())["rows"]]
    if not other_ids:
        print("no bundles given; a disjointness claim with nothing to be "
              "disjoint from is not a claim")
        return 2
    other_rows = sorted({r for rows in other_ids.values() for r in rows})

    enriched = []
    for row in population:
        raw = source.get(row["id"])
        if raw is None:
            continue
        ir = raw.get("target_ir") or ""
        enriched.append({
            "row": row,
            "ir": ir,
            "bytes": len(ir.encode("utf-8")),
            "markers": bool(row["overlays"]["markers"]),
            "duration": contract_duration(raw),
        })

    lengths = sorted(e["bytes"] for e in enriched)
    third, two_thirds = lengths[len(lengths) // 3], lengths[2 * len(lengths) // 3]

    def band(entry) -> str:
        if entry["bytes"] <= third:
            return "short"
        if entry["bytes"] <= two_thirds:
            return "medium"
        return "long"

    rng = random.Random(args.seed)
    buckets: dict[tuple, list] = {}
    for entry in enriched:
        buckets.setdefault((entry["markers"], band(entry)), []).append(entry)
    for key in buckets:
        buckets[key].sort(key=lambda e: e["row"]["id"])
        rng.shuffle(buckets[key])

    # Round-robin the marker x length cells so the selection keeps the
    # population's own shape rather than a corner of it, then break ties toward
    # duration variety.
    order = sorted(buckets, key=lambda k: (not k[0], k[1]))
    chosen, seen_durations = [], Counter()
    while len(chosen) < args.rows:
        progressed = False
        for key in order:
            if len(chosen) >= args.rows:
                break
            bucket = buckets[key]
            if not bucket:
                continue
            bucket.sort(key=lambda e: seen_durations[e["duration"]])
            pick = bucket.pop(0)
            seen_durations[pick["duration"]] += 1
            chosen.append(pick)
            progressed = True
        if not progressed:
            break

    chosen_ids = {e["row"]["id"] for e in chosen}
    chosen_shingles = {e["row"]["id"]: shingles(e["ir"]) for e in chosen}
    other_shingles = {r: shingles((source.get(r) or {}).get("target_ir") or "")
                      for r in other_rows}

    across = []
    for row_id, mine in chosen_shingles.items():
        for other, theirs in other_shingles.items():
            score = jaccard(mine, theirs)
            if score:
                across.append({"holdout": row_id, "other": other,
                               "jaccard": round(score, 5)})
    across.sort(key=lambda e: -e["jaccard"])

    within = []
    keys = sorted(chosen_shingles)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            score = jaccard(chosen_shingles[keys[i]], chosen_shingles[keys[j]])
            if score:
                within.append({"a": keys[i], "b": keys[j],
                               "jaccard": round(score, 5)})
    within.sort(key=lambda e: -e["jaccard"])

    # the background level this source's shared house format produces
    rest = [e for e in enriched if e["row"]["id"] not in chosen_ids]
    sample = rng.sample(rest, min(40, len(rest)))
    background = [jaccard(shingles(a["ir"]), shingles(b["ir"]))
                  for i, a in enumerate(sample) for b in sample[i + 1:]]
    background_summary = {
        "pairs": len(background),
        "median": round(statistics.median(background), 5) if background else None,
        "max": round(max(background), 5) if background else None,
    }

    exact = {e["row"]["prompt_sha256"] for e in chosen}
    normalised = {e["row"]["normalized_prompt_sha256"] for e in chosen}
    other_exact = {(source.get(r) or {}).get("target_ir") for r in other_rows}
    other_hashes = {__import__("hashlib").sha256((t or "").encode()).hexdigest()
                    for t in other_exact}

    document = {
        "purpose": ("the deterministic text-only T2VA regression holdout named "
                    "by active_plan.md; built as a bundle with "
                    "build_native_h3_calibration_batch.py --population text-only "
                    "so the layer-50 comparator grades text rows the same way "
                    "as the vision holdout"),
        "producer": Path(__file__).name,
        "seed": args.seed,
        "dataset": {"repo_id": "StellarVoyager/H3-IR", "revision": revision},
        "population": {"file": args.excluded.name, "rows": len(population),
                       "role": "no-vision"},
        "disjointness": {
            "note": ("these rows carry no media, so media and visual-family "
                     "disjointness are true by construction and assert nothing. "
                     "The informative axis is the prompt."),
            "compared_against": other_ids,
            "exact_prompt_hashes_unique_within_selection":
                len(exact) == len(chosen),
            "normalised_prompt_hashes_unique_within_selection":
                len(normalised) == len(chosen),
            "exact_prompt_shared_with_other_bundles":
                sorted(exact & other_hashes),
            "row_ids_shared_with_other_bundles":
                sorted(chosen_ids & set(other_rows)),
            "near_duplicate_prompt_background": background_summary,
            "top_prompt_overlap_across_the_split": across[:10],
            "top_prompt_overlap_within_the_selection": within[:10],
        },
        "achieved": {
            "rows": len(chosen),
            "with_dialogue_markers": sum(1 for e in chosen if e["markers"]),
            "by_length_band": dict(Counter(band(e) for e in chosen)),
            "by_duration_seconds": dict(Counter(e["duration"] for e in chosen)),
        },
        "rows": [
            {
                "row_id": e["row"]["id"],
                "duration_seconds": e["duration"],
                "prompt_bytes": e["bytes"],
                "length_band": band(e),
                "dialogue_markers": e["row"]["overlays"]["markers"],
                "audio_label": e["row"]["overlays"]["audio_label"],
                "prompt_sha256": e["row"]["prompt_sha256"],
                "normalized_prompt_sha256": e["row"]["normalized_prompt_sha256"],
            }
            for e in chosen
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")

    print(f"{len(chosen)} text-only row(s) at {revision[:12]}")
    for entry in document["rows"]:
        markers = ",".join(entry["dialogue_markers"]) or "-"
        print(f"  {entry['row_id']}  {entry['duration_seconds']:>2}s  "
              f"{entry['prompt_bytes']:5d}B  {entry['length_band']:6s}  {markers}")
    achieved = document["achieved"]
    print(f"\nmarkers {achieved['with_dialogue_markers']}/{len(chosen)}  "
          f"lengths {achieved['by_length_band']}  "
          f"durations {achieved['by_duration_seconds']}")
    disjoint = document["disjointness"]
    print(f"prompt background (unrelated pairs in this population): "
          f"median {disjoint['near_duplicate_prompt_background']['median']}, "
          f"max {disjoint['near_duplicate_prompt_background']['max']}")
    if across:
        top = across[0]
        print(f"top overlap across the split: {top['jaccard']} "
              f"({top['holdout']} | {top['other']})")
    problems = (disjoint["exact_prompt_shared_with_other_bundles"]
                or disjoint["row_ids_shared_with_other_bundles"]
                or not disjoint["exact_prompt_hashes_unique_within_selection"])
    print(f"prompt-disjoint from the other bundles: "
          f"{'YES' if not problems else 'NO'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
