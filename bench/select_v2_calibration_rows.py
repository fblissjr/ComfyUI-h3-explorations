#!/usr/bin/env python3
"""Pick the v2 calibration rows and a component-disjoint holdout from the pool.

The sprint version of Gate 3's split. It keeps the one property that protects
the artifact -- calibration and holdout share no exact-media component, so the
numerical check is not a memorisation test -- and drops the rest (rejection
manifests with reasons, near-duplicate review).

Selection is under two budgets, because the host cache is 46 to 69 KB per
token at peak (`bench/results/2026-08-25_gate2a_*`) and AWQ's grid search
re-runs each layer's parent over every cached batch about 21 times, so both
memory and time scale with total tokens, while the row envelope scales with
the longest row:

- `--max-row-tokens`: an estimated per-row ceiling; and
- `--total-tokens`: an estimated total for the calibration set.

Estimates, not measurements: visual tokens follow the accepted v2 geometry
(`bench/build_native_h3_calibration_batch.py`) from the pool's recorded source
dimensions -- reference stills at `min(1, 2048 / short_edge)` rounded to 32,
keyframes on the adapted canvas, reference video as one 1,008-token block per
second of the contract's duration -- and text tokens are the released
tokenizer's count of the user message. The builder then produces the exact
lengths, and the run record supersedes these numbers.

Role shares follow the pool's partition with floors for the rare roles, so the
small families are present rather than proportionally absent. Deterministic:
candidates are ordered by a seeded shuffle of their ids.

    python bench/select_v2_calibration_rows.py --rows 100 --holdout 12 \\
        --out bench/results/<date>_v2_calibration_selection.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import random
import re
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
POOL = BENCH / "results" / "2026-08-24_h3_calibration_pool.jsonl"

from h3_producer_provenance import producer_provenance  # noqa: E402

REF_SHORT_EDGE = 2048
MULTIPLE = 32
PATCH = 16
MERGE = 2
CANVAS_AREA = 1344 * 768
VIDEO_TOKENS_PER_SECOND = 1008  # one two-frame block per second at 2 fps sampling
LABEL_OVERHEAD = 8  # "<Picture i>: " plus vision start/end, per media item

# Floors for the rare roles at any population size; the remainder is filled
# proportionally to the pool's partition.
ROLE_FLOORS = {
    "video-reference": 6,
    "keyframe-plus-reference": 6,
    "single-image": 8,
    "keyframe-only": 8,
}
ROLES = ("multi-image-2-3", "multi-image-4-9", "keyframe-only", "single-image",
         "keyframe-plus-reference", "video-reference")


def _dataset_root() -> tuple[Path, str]:
    spec = importlib.util.spec_from_file_location(
        "_h3_pool_builder", BENCH / "build_h3_calibration_pool.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load build_h3_calibration_pool.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pinned_snapshot()


def _round32(value: float) -> int:
    return max(MULTIPLE, int(round(value / MULTIPLE)) * MULTIPLE)


def reference_still_tokens(width: int, height: int) -> int:
    scale = min(1.0, REF_SHORT_EDGE / min(width, height))
    w, h = _round32(width * scale), _round32(height * scale)
    return (w // (PATCH * MERGE)) * (h // (PATCH * MERGE))


def keyframe_tokens(width: int, height: int) -> int:
    # `adapt_canvas`: keep the aspect, fit the canvas area, round to 32.
    scale = math.sqrt(CANVAS_AREA / (width * height))
    w, h = _round32(width * scale), _round32(height * scale)
    return (w // (PATCH * MERGE)) * (h // (PATCH * MERGE))


def estimate_row(pool_row: dict, raw_row: dict, tokenizer) -> dict:
    # The encoder is presented the H3 prompt, which is the row's `target_ir`;
    # the user message is the request and only supplies the contract's duration.
    user = next(m for m in raw_row["messages"] if m["role"] == "user")["content"]
    match = re.search(r"duration_seconds:\s*([0-9.]+)", user)
    duration = float(match.group(1)) if match else 5.0
    prompt = raw_row.get("target_ir") or ""
    text_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    visual = 0
    items = 0
    roles = pool_row.get("picture_roles") or {}
    for index, (width, height) in enumerate(pool_row.get("image_dimensions") or [], start=1):
        items += 1
        if roles.get(str(index)) == "keyframe":
            visual += keyframe_tokens(width, height)
        else:
            visual += reference_still_tokens(width, height)
    for _ in pool_row.get("videos") or []:
        items += 1
        visual += int(math.ceil(duration)) * VIDEO_TOKENS_PER_SECOND
    total = text_tokens + visual + items * LABEL_OVERHEAD
    return {"text_tokens_est": text_tokens, "visual_tokens_est": visual,
            "tokens_est": total, "duration_seconds": duration, "media_items": items}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--holdout", type=int, default=12)
    parser.add_argument("--max-row-tokens", type=int, default=16000)
    parser.add_argument("--total-tokens", type=int, default=400000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--source-dir", required=True,
                        help="released text encoder directory, for the tokenizer")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.source_dir)
    root, revision = _dataset_root()
    pool = [json.loads(line) for line in POOL.read_text().splitlines()]
    raw = {}
    for line in (root / "data" / "train.jsonl").read_text().splitlines():
        row = json.loads(line)
        raw[row["id"]] = row

    estimates = {}
    for row in pool:
        estimates[row["id"]] = estimate_row(row, raw[row["id"]], tokenizer)

    rng = random.Random(args.seed)
    order = sorted(pool, key=lambda r: hashlib.sha256(f"{args.seed}:{r['id']}".encode()).hexdigest())
    rng.shuffle(order)  # seeded; the hash sort above already fixes the order

    # Quotas: floors first, then the remainder proportional to the pool.
    pool_counts = collections.Counter(r["primary_role"] for r in pool)
    quotas = {role: ROLE_FLOORS.get(role, 0) for role in ROLES}
    remainder = args.rows - sum(quotas.values())
    if remainder < 0:
        raise SystemExit("--rows is below the sum of the role floors")
    proportional_roles = [r for r in ROLES if r not in ROLE_FLOORS]
    weight = sum(pool_counts[r] for r in proportional_roles)
    for role in proportional_roles:
        quotas[role] += int(round(remainder * pool_counts[role] / weight))
    while sum(quotas.values()) > args.rows:
        quotas[max(proportional_roles, key=lambda r: quotas[r])] -= 1
    while sum(quotas.values()) < args.rows:
        quotas[min(proportional_roles, key=lambda r: quotas[r])] += 1

    calibration, used_components, total = [], set(), 0
    skipped = collections.Counter()
    # Fill every role in proportion to its quota, one row at a time, taking the
    # role whose share of its quota is furthest behind. The budget then
    # truncates all families proportionally instead of starving whichever one
    # happened to come last.
    queues = {role: [r for r in order if r["primary_role"] == role] for role in ROLES}
    taken = {role: 0 for role in ROLES}
    while True:
        open_roles = [r for r in ROLES if taken[r] < quotas[r] and queues[r]]
        if not open_roles:
            break
        role = min(open_roles, key=lambda r: taken[r] / quotas[r])
        row = queues[role].pop(0)
        est = estimates[row["id"]]["tokens_est"]
        if est > args.max_row_tokens:
            skipped["over_row_cap"] += 1
            continue
        if total + est > args.total_tokens:
            skipped["over_total_budget"] += 1
            # This family cannot take another row within the budget; close it
            # so the loop terminates once every family is closed.
            queues[role] = [r for r in queues[role]
                            if total + estimates[r["id"]]["tokens_est"] <= args.total_tokens]
            continue
        calibration.append(row)
        used_components.add(row["media_component"])
        total += est
        taken[role] += 1

    holdout = []
    per_role_holdout = max(1, args.holdout // len(ROLES))
    for role in ROLES:
        taken = 0
        for row in order:
            if taken >= per_role_holdout or len(holdout) >= args.holdout:
                break
            if row["primary_role"] != role or row["media_component"] in used_components:
                continue
            if any(h["id"] == row["id"] for h in holdout):
                continue
            if estimates[row["id"]]["tokens_est"] > args.max_row_tokens:
                continue
            holdout.append(row)
            used_components.add(row["media_component"])
            taken += 1

    calibration_components = {r["media_component"] for r in calibration}
    holdout_components = {r["media_component"] for r in holdout}
    if calibration_components & holdout_components:
        raise SystemExit("calibration and holdout share a media component")

    def describe(rows):
        return [{"id": r["id"], "primary_role": r["primary_role"],
                 "media_component": r["media_component"],
                 "image_dimensions": r.get("image_dimensions"),
                 "videos": len(r.get("videos") or []),
                 **estimates[r["id"]]} for r in rows]

    achieved = collections.Counter(r["primary_role"] for r in calibration)
    report = {
        "purpose": "sprint selection of the v2 calibration rows and a "
                   "component-disjoint holdout; estimates, superseded by the "
                   "builder's exact lengths",
        "producer": producer_provenance(__file__),
        "pool": {"file": POOL.name, "rows": len(pool), "dataset_revision": revision},
        "budgets": {"rows": args.rows, "holdout": args.holdout,
                    "max_row_tokens_est": args.max_row_tokens,
                    "total_tokens_est": args.total_tokens, "seed": args.seed},
        "quotas": quotas,
        "achieved": {"rows": len(calibration), "by_role": dict(achieved),
                     "tokens_est_total": total,
                     "longest_row_tokens_est": max(
                         (estimates[r["id"]]["tokens_est"] for r in calibration), default=0),
                     "skipped": dict(skipped),
                     "components": len(calibration_components)},
        "component_disjoint": True,
        "calibration": describe(calibration),
        "holdout": describe(holdout),
        "estimate_rules": {
            "reference_still": "min(1, 2048/short_edge), round to 32, tokens = (w/32)*(h/32)",
            "keyframe": "adapt to the 1344x768 canvas area at source aspect, round to 32",
            "video": f"{VIDEO_TOKENS_PER_SECOND} tokens per second of contract duration",
            "text": "released tokenizer count of the row's target_ir, the H3 prompt, no special tokens",
            "labels": f"{LABEL_OVERHEAD} tokens per media item",
        },
    }
    out = Path(args.out).expanduser().resolve()
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"calibration {len(calibration)} rows, est {total:,} tokens, longest est "
          f"{report['achieved']['longest_row_tokens_est']:,}, by role {dict(achieved)}, "
          f"skipped {dict(skipped)}")
    print(f"holdout {len(holdout)} rows, roles "
          f"{dict(collections.Counter(r['primary_role'] for r in holdout))}")
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
