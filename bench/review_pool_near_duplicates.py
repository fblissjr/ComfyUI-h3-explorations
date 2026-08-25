#!/usr/bin/env python3
"""Pool-wide near-duplicate review, and the component map it implies.

`calibration_data_pool.md` reduces the accepted pool to indivisible components
by connecting rows that share an *exact* media SHA-256. `active_plan.md`
requires a near-duplicate review before any split is frozen, and on the first
split built without one a holdout row turned out to be the same photo series as
three calibration rows: same studio, same model, frames matched shot for shot,
byte-different files. Exact hashing cannot see that. This script is the review.

What it does, in order:

1. Perceptually hash every distinct image the accepted pool declares, from the
   pinned snapshot.
2. Bucket by that hash so the candidate search is not quadratic over the whole
   pool, then rank every within-window pair by a background-masked correlation.
3. Report the candidate volume and its tiering *before* anybody rules on pairs,
   because a window that produces more candidates than a person will look at
   must say so rather than let the unexamined tail read as clean.
4. Emit the corrected component map implied by the pairs an adjudication file
   rules `duplicate` or `uncertain` -- exact components unioned with those
   edges -- so a splitter can assign by whole visual family. The exact map
   travels beside it, and so does the list of rows that changed family.

The threshold is priced, not asserted: the same run reports how often images
drawn from *distinct* exact components land inside the window anyway. On this
dataset that rate is not negligible, because a large share of it is three-view
turnaround sheets on a white field and a difference hash matches that template.
That is why step 4 consumes adjudicated verdicts rather than raw distances.
Neither metric is sufficient alone, which the 2026-08-25 review measured both
ways: pairs at Hamming 0 that are different subjects sharing the template, and
a pair at correlation 0.43 that is the same figure rendered twice.

CPU only, no CUDA, no model, no server. Hashing the pool takes a few minutes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))

from review_v2_calibration_bundle import (  # noqa: E402
    foreground_correlation,
    hamming,
    perceptual_hashes,
    reduce_image,
)

POOL = BENCH / "results" / "2026-08-24_h3_calibration_pool.jsonl"


def bucket_keys(bits: int, bands: int = 8) -> list[tuple[int, int]]:
    """Banded LSH keys: two hashes within `bands`-1 bits share at least one.

    A 64-bit hash split into `bands` blocks means a pair differing by fewer
    bits than there are blocks must agree on some block exactly. That makes the
    candidate search linear in practice without changing which pairs it can
    find inside that bound.
    """
    width = 64 // bands
    mask = (1 << width) - 1
    return [(i, (bits >> (i * width)) & mask) for i in range(bands)]


def load_pool(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def hash_pool_images(pool: list[dict], root: Path) -> tuple[dict, list[str]]:
    seen: dict[str, int] = {}
    failed = []
    for row in pool:
        for rel in row.get("images") or []:
            if rel in seen:
                continue
            try:
                seen[rel] = perceptual_hashes(root / rel, "image")[0]
            except Exception as exc:
                failed.append(f"{rel}: {exc}")
    return seen, failed


def candidate_pairs(hashes: dict, threshold: int, bands: int) -> list[tuple]:
    buckets: dict[tuple, list[str]] = defaultdict(list)
    for rel, bits in hashes.items():
        for key in bucket_keys(bits, bands):
            buckets[key].append(rel)
    pairs = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for a, b in itertools.combinations(sorted(members), 2):
            distance = hamming(hashes[a], hashes[b])
            if distance <= threshold:
                pairs.add((a, b, distance))
    return sorted(pairs, key=lambda p: p[2])


def price_threshold(pool: list[dict], hashes: dict, threshold: int,
                    sample: int, seed: int) -> dict:
    """How often unrelated images land inside the window anyway."""
    by_component: dict[str, list[str]] = defaultdict(list)
    for row in pool:
        for rel in row.get("images") or []:
            if rel in hashes:
                by_component[row["media_component"]].append(rel)
    rng = random.Random(seed)
    components = sorted(c for c, v in by_component.items() if v)
    picked = [hashes[rng.choice(by_component[c])]
              for c in rng.sample(components, min(sample, len(components)))]
    pairs = inside = 0
    for i in range(len(picked)):
        for j in range(i + 1, len(picked)):
            pairs += 1
            if hamming(picked[i], picked[j]) <= threshold:
                inside += 1
    return {"components_sampled": len(picked), "pairs": pairs, "inside": inside,
            "rate": round(inside / pairs, 6) if pairs else None}


class Union:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def corrected_components(pool: list[dict], duplicate_pairs: list[tuple]) -> dict:
    """Exact components unioned with adjudicated perceptual edges."""
    owner: dict[str, list[str]] = defaultdict(list)
    for row in pool:
        for rel in row.get("images") or []:
            owner[rel].append(row["id"])
    union = Union()
    for row in pool:
        union.union(row["id"], row["media_component"])
    merged = 0
    for rel_a, rel_b, _ in duplicate_pairs:
        rows_a, rows_b = owner.get(rel_a, []), owner.get(rel_b, [])
        if not rows_a or not rows_b:
            continue
        before = union.find(rows_a[0])
        union.union(rows_a[0], rows_b[0])
        if union.find(rows_a[0]) != before or before != union.find(rows_b[0]):
            merged += 1
    groups: dict[str, list[str]] = defaultdict(list)
    for row in pool:
        groups[union.find(row["id"])].append(row["id"])
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    return {"components": len(groups), "edges_applied": merged,
            "largest": sizes[0] if sizes else 0,
            "multi_row": sum(1 for s in sizes if s > 1),
            "rows_in_multi_row": sum(s for s in sizes if s > 1),
            "map": {min(v): sorted(v) for v in groups.values()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=POOL)
    parser.add_argument("--threshold", type=int, default=6,
                        help="Hamming window for candidate generation")
    parser.add_argument("--bands", type=int, default=8,
                        help="LSH bands; must exceed the threshold to be exact")
    parser.add_argument("--correlation-threshold", type=float, default=0.6)
    parser.add_argument("--adjudication", type=Path,
                        help="verdicts for candidate pairs; without it the run "
                             "reports the candidate volume and rules on nothing")
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--component-map-out", type=Path)
    args = parser.parse_args()

    if args.bands <= args.threshold:
        print(f"bands ({args.bands}) must exceed threshold ({args.threshold}) "
              f"or the banded search can miss pairs inside the window")
        return 2

    from build_h3_calibration_pool import pinned_snapshot

    root, revision = pinned_snapshot()
    pool = load_pool(args.pool)
    hashes, failed = hash_pool_images(pool, root)
    print(f"hashed {len(hashes)} distinct images from {len(pool)} pooled rows "
          f"at {revision[:12]}; {len(failed)} unhashable")

    pairs = candidate_pairs(hashes, args.threshold, args.bands)
    print(f"{len(pairs)} candidate pairs inside Hamming {args.threshold}")

    reductions: dict[str, object] = {}
    scored = []
    for rel_a, rel_b, distance in pairs:
        for rel in (rel_a, rel_b):
            if rel not in reductions:
                reductions[rel] = reduce_image(root / rel)
        scored.append((round(foreground_correlation(reductions[rel_a],
                                                    reductions[rel_b]), 4),
                       distance, rel_a, rel_b))
    scored.sort(key=lambda s: (-s[0], s[1]))
    strong = [s for s in scored if s[0] >= args.correlation_threshold]
    print(f"{len(strong)} of them also clear foreground correlation "
          f"{args.correlation_threshold}")

    pricing = price_threshold(pool, hashes, args.threshold, args.sample, 7)
    print(f"threshold pricing: {pricing['inside']} of {pricing['pairs']} pairs "
          f"drawn from distinct exact components land inside the window "
          f"({pricing['rate']})")

    owner: dict[str, list[str]] = defaultdict(list)
    for row in pool:
        for rel in row.get("images") or []:
            owner[rel].append(row["id"])
    by_component = {row["id"]: row["media_component"] for row in pool}

    def crosses_components(rel_a: str, rel_b: str) -> bool:
        comps_a = {by_component[r] for r in owner.get(rel_a, [])}
        comps_b = {by_component[r] for r in owner.get(rel_b, [])}
        return bool(comps_a - comps_b) and bool(comps_b - comps_a)

    crossing = [s for s in scored if crosses_components(s[2], s[3])]
    crossing_strong = [s for s in strong if crosses_components(s[2], s[3])]
    print(f"{len(crossing)} candidates cross an exact-component boundary "
          f"({len(crossing_strong)} of those also strong) -- these are the ones "
          f"that would change the component map")

    report = {
        "revision": revision,
        "pool_rows": len(pool),
        "distinct_images": len(hashes),
        "unhashable": failed,
        "hamming_threshold": args.threshold,
        "correlation_threshold": args.correlation_threshold,
        "threshold_pricing": pricing,
        "candidates": len(scored),
        "candidates_strong": len(strong),
        "candidates_crossing_components": len(crossing),
        "candidates_crossing_components_strong": len(crossing_strong),
        "correlation_histogram": dict(sorted(Counter(
            round(min(max(s[0], -1.0), 1.0), 1) for s in scored).items())),
        "crossing_pairs": [
            {"foreground_correlation": c, "hamming": d, "a": a, "b": b,
             "rows_a": owner.get(a, []), "rows_b": owner.get(b, []),
             "components_a": sorted({by_component[r] for r in owner.get(a, [])}),
             "components_b": sorted({by_component[r] for r in owner.get(b, [])})}
            for c, d, a, b in crossing],
    }

    adjudicated = []
    if args.adjudication and args.adjudication.is_file():
        loaded = json.loads(args.adjudication.read_text())
        ruled = {(e["a"], e["b"]): e["verdict"] for e in loaded["pairs"]}
        report["adjudication_scope"] = loaded.get("scope")
        unruled = [s for s in crossing if (s[2], s[3]) not in ruled]
        report["unruled_crossing_pairs"] = len(unruled)
        # `uncertain` becomes an edge. At a split boundary the cost of joining two
        # families that were separable is a slightly coarser split; the cost of
        # separating one family is a holdout that measures nothing. They are not
        # symmetric, so the tie goes to joining -- and the verdict stays visible
        # as `uncertain` rather than being laundered into `duplicate`.
        adjudicated = [(a, b, d) for c, d, a, b in crossing
                       if ruled.get((a, b)) in ("duplicate", "uncertain")]
        report["adjudicated_duplicate_pairs"] = sum(
            1 for c, d, a, b in crossing if ruled.get((a, b)) == "duplicate")
        report["adjudicated_uncertain_pairs"] = sum(
            1 for c, d, a, b in crossing if ruled.get((a, b)) == "uncertain")
        if unruled:
            print(f"{len(unruled)} crossing candidates carry no verdict; the "
                  f"component map below applies only the ruled ones")
    else:
        report["unruled_crossing_pairs"] = len(crossing)
        report["adjudicated_duplicate_pairs"] = 0
        print("no adjudication supplied: reporting candidate volume only, and "
              "the component map is unchanged from the exact one")

    corrected = corrected_components(pool, adjudicated)
    component_map = corrected.pop("map")
    report["corrected_components"] = corrected
    print(f"component map: {corrected['components']} components, "
          f"{corrected['multi_row']} holding more than one row, largest "
          f"{corrected['largest']}, from {corrected['edges_applied']} applied edges")

    if args.out:
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.component_map_out:
        exact = {row["id"]: row["media_component"] for row in pool}
        corrected_by_row = {row_id: component
                            for component, members in component_map.items()
                            for row_id in members}
        moved = sorted(r for r in exact if exact[r] != corrected_by_row[r])
        args.component_map_out.write_text(json.dumps(
            {"revision": revision,
             "basis": "exact media SHA-256 components unioned with adjudicated "
                      "perceptual duplicate and uncertain edges",
             "producer": Path(__file__).name,
             "adjudication": args.adjudication.name if args.adjudication else None,
             "adjudicated_edges": corrected["edges_applied"],
             "unexamined_weak_candidates": report["unruled_crossing_pairs"],
             "caveat": (
                 (f"{report['unruled_crossing_pairs']} candidate pairs crossing "
                  f"a component boundary carry no verdict, so rows may still "
                  f"share a visual family through one of them")
                 if report["unruled_crossing_pairs"] else
                 (f"every candidate crossing a component boundary is ruled. "
                  f"What remains unexamined is everything the Hamming "
                  f"{args.threshold} window never proposed: two images of one "
                  f"subject far enough apart in hash to fall outside it are not "
                  f"in this map, and nothing here would show that")),
             "rows_that_changed_component": moved,
             "exact_component_by_row": exact,
             "corrected_component_by_row": corrected_by_row,
             "corrected_components": component_map}, indent=2, sort_keys=True) + "\n")
        print(f"{len(moved)} rows changed component; map written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
