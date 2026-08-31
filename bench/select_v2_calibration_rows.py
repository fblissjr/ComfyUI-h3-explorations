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

The holdout reserves small-source components first (`--holdout-small-source`,
the plan's locked "at least two"), preferring rows whose small image is a
reference still: under the `upscale_2048` policy only reference stills are
upscaled, so a small keyframe never exercises the policy the holdout grades.

Rebuilding a holdout after a near-duplicate review, with the calibration rows
already consumed by a run: `--rows 0` selects no calibration rows,
`--keep-holdout` carries the previous holdout forward, and `--exclude-row`,
`--exclude-component`, `--exclude-prompt-term` remove what the review named.
The first such rebuild (2026-08-25) dropped one holdout row whose frames were a
shot-for-shot match of three calibration rows from the same product catalogue
series; the series spans dozens of pool rows across many exact-media
components, which is why a prompt-term exclusion exists.

    python bench/select_v2_calibration_rows.py --rows 100 --holdout 12 \\
        --out bench/results/<date>_v2_calibration_selection.json

Four later additions, each off unless its flag is given. With none of them
present the role-quota path above runs unchanged; `bench/check_calibration_selector.py`
holds that to the pre-change revision of this file rather than to a claim.

**Token-balanced strata (`--stratum`).** Role quotas count rows, and AWQ's
statistics are per *token*, so a role quota says nothing about how the token
mass is distributed. On the 2026-08-25 run it went 29 rows and 214k tokens with
roughly nine tenths of the tokens visual, which is the same thing as saying the
H3 schema positions -- `<d>` dialogue blocks, `<Picture i>: ` labels,
`<Video k>: `, timestamps -- were about a tenth of what the scales were fitted
on. A stratum here is `primary role | marker presence`, marker presence being
whether the row's `target_ir` carries any marker the release appends past the
stock vocabulary (`vendor_config.h3_markers()`, derived from the vendored
declaration, never retyped). Each occupied stratum is given a share of the
*token* budget rather than a row count: `--stratum-token-share NAME=SHARE`
names one, the rest split what is left equally, and `--stratum-floor-share` is
the minimum any occupied stratum may be given -- below it the design refuses
rather than emitting a stratum that can hold no row. Every occupied stratum is
guaranteed its first row even if that row overshoots its target; after that a
row is admitted only if it fits inside the stratum's remaining target.

A **language** overlay is deliberately not a third axis: the pool row schema
(`bench/results/archive/v2_encoder/2026-08-24_h3_calibration_pool.jsonl`) records `channel`,
`primary_role`, `picture_roles` and an `overlays` block of
`wide_or_tall`/`small_source`/`markers`/`audio_label`/`video_audio_track`, and
no language field. A language tag *is* observable inside the `<d>[...]` blocks
of `target_ir` and is far from balanced there, so the overlay is available to
whoever wants to derive it -- but deriving it here would be this file inventing
a pool field, and the report says the axis was skipped rather than reporting an
axis that was never applied.

**`--max-vision-tokens-per-row`** caps the estimated *visual* tokens one row
may contribute. A row over the cap is skipped, never truncated: truncating
would change the presentation the builder emits, and a calibration row that is
not what the deployed path presents is worse than an absent one. The cap
reaches calibration only. The holdout grades the still policy and a vision cap
there would silently pick which geometries get graded.

**`--component-map FILE`** assigns rows by *corrected visual family* instead of
exact media component (`bench/review_pool_near_duplicates.py`'s output). Both
are recorded per row. Under the map three things change: every exclusion widens
to the whole family rather than the exact component, no two calibration rows
may share a family, and the split assertion is by family. The map must carry a
`caveat`; a map that does not say what it cannot see is refused, because the
2026-08-25 map's own caveat is that same-brief-different-render relatedness
among the images is unexamined, and a split graded against a map without that
sentence would read as stronger than it is.

**`--text-only-share`** admits the pool's text-only T2VA rows (the exclusion
file `build_native_h3_calibration_batch.py --population text-only` builds from)
up to a share of the token budget, into their own `text_only` list in the
record. They are NOT merged into `calibration`, and the reason is measured, not
stylistic: `docs/research/qwen3-vl-special-tokens-post-training/canonical/2026-08-24_calibration_input_seam.md`
Q1a/Q1b show one `oneshot` call traces the model once and that trace fixes the
modality envelope for the whole run -- trace from a vision row and a text-only
row raises, trace from a text-only row and every later vision row silently
loses its media. So a selection carrying both is two bundles and two runs, or a
second traced graph, which is a real change to `llm-compressor`'s sequential
pipeline. The record names the rows and states that constraint so the choice is
made rather than inherited.
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
REPO = BENCH.parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(REPO))
POOL = BENCH / "results" / "archive" / "v2_encoder" / "2026-08-24_h3_calibration_pool.jsonl"
TEXT_ONLY_POOL = BENCH / "results" / "archive" / "v2_encoder" / "2026-08-24_h3_calibration_pool_excluded.jsonl"

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

MARKERS_TAG = "markers"
NO_MARKERS_TAG = "no-markers"

# A stratum that lands within this fraction of its token target is on target;
# a lumpy population cannot hit a target exactly, and calling every stratum
# "short" because the last row did not fit trains the reader to skip the line.
SHORT_OF_TARGET = 0.9


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


STILL_POLICIES = ("max_no_upscale", "upscale_2048")


def reference_still_tokens(width: int, height: int, policy: str = "max_no_upscale") -> int:
    """Merged tokens of a reference still under the builder's two policies.

    `upscale_2048` is the vendor serving convention sglang implements: every
    reference still goes to a 2048 short edge, upscaling included, nearest 32,
    no area cap. `max_no_upscale` is the same ceiling without the upscale.
    """
    ratio = REF_SHORT_EDGE / min(width, height)
    scale = ratio if policy == "upscale_2048" else min(1.0, ratio)
    w, h = _round32(width * scale), _round32(height * scale)
    return (w // (PATCH * MERGE)) * (h // (PATCH * MERGE))


def keyframe_tokens(width: int, height: int) -> int:
    # `adapt_canvas`: keep the aspect, fit the canvas area, round to 32.
    scale = math.sqrt(CANVAS_AREA / (width * height))
    w, h = _round32(width * scale), _round32(height * scale)
    return (w // (PATCH * MERGE)) * (h // (PATCH * MERGE))


def estimate_row(pool_row: dict, raw_row: dict, tokenizer,
                 still_policy: str = "max_no_upscale") -> dict:
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
            visual += reference_still_tokens(width, height, still_policy)
    for _ in pool_row.get("videos") or []:
        items += 1
        visual += int(math.ceil(duration)) * VIDEO_TOKENS_PER_SECOND
    total = text_tokens + visual + items * LABEL_OVERHEAD
    return {"text_tokens_est": text_tokens, "visual_tokens_est": visual,
            "tokens_est": total, "duration_seconds": duration, "media_items": items}


# --------------------------------------------------------------------------
# strata


def stratum_of(pool_row: dict, target_ir: str, markers: list[str]) -> str:
    """`primary role | marker presence`, read off the row's own H3 prompt.

    Marker presence is tested against the prompt text rather than the pool's
    `overlays.markers`, which is the pool builder's own count of the same
    thing: two readings of one property, and a stratum that silently followed
    the builder's list could not disagree with it.
    """
    tag = MARKERS_TAG if any(m in target_ir for m in markers) else NO_MARKERS_TAG
    return f"{pool_row['primary_role']}|{tag}"


def stratum_targets(occupied: list[str], explicit: dict[str, float],
                    floor_share: float, budget: int) -> dict[str, dict]:
    """Token target per occupied stratum, from explicit shares plus an equal split.

    Refuses rather than rescaling when the floor cannot be met: a stratum whose
    share is below the floor is one the budget cannot represent, and quietly
    shrinking it produces a stratum of zero rows that the achieved table then
    reports as a share of nothing.
    """
    unknown = sorted(set(explicit) - set(occupied))
    if unknown:
        raise SystemExit(
            f"--stratum-token-share names strata that are not occupied by any "
            f"eligible row: {unknown}. Occupied: {sorted(occupied)}")
    for name, share in sorted(explicit.items()):
        if share < floor_share:
            raise SystemExit(
                f"--stratum-token-share {name}={share} is below the floor "
                f"{floor_share}; raise the share or lower --stratum-floor-share")
    spoken = sum(explicit.values())
    if spoken > 1.0 + 1e-9:
        raise SystemExit(f"--stratum-token-share values sum to {spoken}, above 1.0")
    free = [s for s in occupied if s not in explicit]
    shares = dict(explicit)
    if free:
        each = (1.0 - spoken) / len(free)
        if each < floor_share:
            raise SystemExit(
                f"{len(free)} unnamed strata would take {each:.4f} of the budget "
                f"each, below the floor {floor_share}. Either name fewer strata "
                f"explicitly, lower --stratum-floor-share, or raise --total-tokens")
        for name in free:
            shares[name] = each
    return {name: {"share": shares[name], "tokens": int(budget * shares[name])}
            for name in sorted(shares)}


# --------------------------------------------------------------------------
# corrected visual families


def load_component_map(path: Path) -> tuple[dict[str, str], dict]:
    """The corrected family assignment, and its own statement of what it misses.

    A map with no `caveat` is refused. `review_pool_near_duplicates.py` writes
    one because its window cannot reach same-brief-different-render pairs among
    the images; a split graded against a map that has dropped that sentence
    would read as a stronger disjointness claim than the map can support.
    """
    loaded = json.loads(Path(path).expanduser().read_text())
    caveat = (loaded.get("caveat") or "").strip()
    if not caveat:
        raise SystemExit(
            f"{Path(path).name} carries no `caveat`. A component map that does "
            f"not say what it cannot see must not be used to claim a family-"
            f"disjoint split; regenerate it with review_pool_near_duplicates.py")
    family = loaded.get("corrected_component_by_row")
    if not isinstance(family, dict) or not family:
        raise SystemExit(f"{Path(path).name} carries no corrected_component_by_row")
    meta = {"file": Path(path).name, "caveat": caveat,
            "basis": loaded.get("basis"),
            "adjudicated_edges": loaded.get("adjudicated_edges"),
            "unexamined_weak_candidates": loaded.get("unexamined_weak_candidates"),
            "rows_mapped": len(family),
            "families": len(set(family.values()))}
    return family, meta


def family_disjointness_problems(calibration: list[dict], holdout: list[dict],
                                 unit_of) -> list[str]:
    """Every family holding rows on both sides of the split, named.

    Factored out so a control can hand it a deliberately mutated assignment;
    the selector cannot produce one by construction, which is exactly why the
    assertion has to be reachable from outside the selector.
    """
    cal = collections.defaultdict(list)
    for row in calibration:
        cal[unit_of(row)].append(row["id"])
    hold = collections.defaultdict(list)
    for row in holdout:
        hold[unit_of(row)].append(row["id"])
    return [f"calibration and holdout share unit {unit}: "
            f"calibration {cal[unit]} against holdout {hold[unit]}"
            for unit in sorted(set(cal) & set(hold))]


# --------------------------------------------------------------------------


def _composition(rows: list[dict], estimates: dict) -> dict:
    text = sum(estimates[r["id"]]["text_tokens_est"] for r in rows)
    visual = sum(estimates[r["id"]]["visual_tokens_est"] for r in rows)
    total = sum(estimates[r["id"]]["tokens_est"] for r in rows)
    return {"rows": len(rows), "tokens_est": total,
            "text_tokens_est": text, "visual_tokens_est": visual,
            "label_tokens_est": total - text - visual,
            "text_share": round(text / total, 4) if total else 0.0,
            "visual_share": round(visual / total, 4) if total else 0.0}


def _fill_strata(queues: dict[str, list], targets: dict[str, dict], estimates: dict,
                 admit, take, budget: int, row_ceiling: int | None) -> tuple[list, int, dict, bool]:
    """Greedy token fill: always extend the stratum furthest behind its target.

    `admit(row)` is a PURE predicate returning None to accept or a reason string
    to drop the row from its queue permanently; `take(row)` is called only on a
    row that is actually selected. The two are separate because a row that would
    overshoot its stratum's remaining target is left in the queue for a later
    pass -- if `admit` had claimed the row's family on the way past, that row
    would come back as its own duplicate.
    """
    taken: list = []
    achieved = {name: 0 for name in targets}
    exhausted: set = set()
    total = 0
    skipped: collections.Counter = collections.Counter()
    ceiling_bound = False
    while True:
        if row_ceiling is not None and len(taken) >= row_ceiling:
            ceiling_bound = True
            break
        open_strata = [s for s in targets if queues.get(s) and achieved[s] < targets[s]["tokens"]]
        if not open_strata:
            break
        name = min(open_strata, key=lambda s: achieved[s] / max(1, targets[s]["tokens"]))
        queue = queues[name]
        chosen = None
        index = 0
        while index < len(queue):
            row = queue[index]
            reason = admit(row)
            if reason is not None:
                skipped[reason] += 1
                queue.pop(index)
                continue
            est = estimates[row["id"]]["tokens_est"]
            if total + est > budget:
                skipped["over_total_budget"] += 1
                queue.pop(index)
                continue
            # Every occupied stratum is guaranteed its first row; after that a
            # row has to fit inside what the stratum has left.
            if achieved[name] and achieved[name] + est > targets[name]["tokens"]:
                index += 1
                continue
            chosen = queue.pop(index)
            break
        if chosen is None:
            # Nothing left in this stratum fits: it closes short of its target.
            exhausted.add(name)
            queues[name] = []
            continue
        take(chosen)
        taken.append(chosen)
        achieved[name] += estimates[chosen["id"]]["tokens_est"]
        total += estimates[chosen["id"]]["tokens_est"]
    return (taken, total,
            {"achieved": achieved, "skipped": dict(skipped),
             "exhausted": sorted(exhausted)},
            ceiling_bound)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--holdout", type=int, default=12)
    parser.add_argument("--max-row-tokens", type=int, default=16000)
    parser.add_argument("--total-tokens", type=int, default=400000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--exclude-selection", action="append", default=[],
                        help="a previous selection file; every media component it "
                             "used, in calibration or holdout, is excluded here so two "
                             "selections can be merged into one disjoint bundle")
    parser.add_argument("--still-policy", choices=STILL_POLICIES, default="max_no_upscale",
                        help="the builder's reference-still policy the estimate follows")
    parser.add_argument("--keep-holdout", default=None,
                        help="a previous selection file whose holdout rows are kept, "
                             "minus any excluded below; the rest of the holdout is "
                             "filled around them")
    parser.add_argument("--exclude-row", action="append", default=[],
                        help="pool row id to exclude, with its whole media component")
    parser.add_argument("--exclude-component", action="append", default=[],
                        help="media component id to exclude")
    parser.add_argument("--exclude-prompt-term", action="append", default=[],
                        help="case-insensitive term; every row whose target_ir contains "
                             "it is excluded with its whole media component")
    parser.add_argument("--holdout-small-source", type=int, default=2,
                        help="small-source components reserved for the holdout before "
                             "the per-role fill")
    parser.add_argument("--stratum", action="store_true",
                        help="token-balanced strata (primary role x marker presence) "
                             "instead of the role row-quota path")
    parser.add_argument("--stratum-token-share", action="append", default=[],
                        metavar="NAME=SHARE",
                        help="target share of the token budget for one stratum, e.g. "
                             "'video-reference|markers=0.15'; implies --stratum. "
                             "Unnamed occupied strata split what is left equally")
    parser.add_argument("--stratum-floor-share", type=float, default=0.02,
                        help="minimum share of the token budget any occupied stratum "
                             "may be given; the design refuses below it")
    parser.add_argument("--max-vision-tokens-per-row", type=int, default=0,
                        help="cap on a calibration row's estimated VISUAL tokens; a row "
                             "over it is skipped, never truncated. 0 disables")
    parser.add_argument("--component-map", default=None,
                        help="corrected visual-family map from "
                             "review_pool_near_duplicates.py; assignment, exclusion "
                             "widening and the split assertion all move to families")
    parser.add_argument("--text-only-share", type=float, default=0.0,
                        help="share of the token budget reserved for text-only T2VA "
                             "rows, reported in their own list; they cannot share a "
                             "traced run with vision rows (see the module docstring)")
    parser.add_argument("--pool", default=None,
                        help="override the accepted pool file; for controls")
    parser.add_argument("--text-only-pool", default=None,
                        help="override the text-only exclusion file; for controls")
    parser.add_argument("--dataset-root", default=None,
                        help="override the pinned H3-IR snapshot root; for controls")
    parser.add_argument("--source-dir", required=True,
                        help="released text encoder directory, for the tokenizer")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    explicit_shares: dict[str, float] = {}
    for spec in args.stratum_token_share:
        name, _, value = spec.rpartition("=")
        if not name or not value:
            raise SystemExit(f"--stratum-token-share expects NAME=SHARE, got {spec!r}")
        explicit_shares[name] = float(value)
    stratum_mode = args.stratum or bool(explicit_shares)
    if args.text_only_share < 0 or args.text_only_share >= 1:
        raise SystemExit("--text-only-share must be in [0, 1)")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.source_dir)
    if args.dataset_root:
        root = Path(args.dataset_root).expanduser().resolve()
        revision = f"override: {args.dataset_root}"
    else:
        root, revision = _dataset_root()
    pool_path = Path(args.pool).expanduser() if args.pool else POOL
    pool = [json.loads(line) for line in pool_path.read_text().splitlines()]
    raw = {}
    for line in (root / "data" / "train.jsonl").read_text().splitlines():
        row = json.loads(line)
        raw[row["id"]] = row

    text_only_pool: list[dict] = []
    if args.text_only_share:
        text_only_path = (Path(args.text_only_pool).expanduser()
                          if args.text_only_pool else TEXT_ONLY_POOL)
        text_only_pool = [json.loads(line)
                          for line in text_only_path.read_text().splitlines()]
        stray = sorted(r["id"] for r in text_only_pool if r.get("primary_role") != "no-vision")
        if stray:
            raise SystemExit(
                f"{text_only_path.name} carries vision-bearing rows {stray[:5]}; "
                f"--text-only-share admits the no-vision population only")

    estimates = {}
    for row in pool + text_only_pool:
        estimates[row["id"]] = estimate_row(row, raw[row["id"]], tokenizer, args.still_policy)

    markers: list[str] = []
    if stratum_mode:
        import vendor_config

        markers = vendor_config.h3_markers()

    family_map: dict[str, str] = {}
    map_meta: dict = {}
    if args.component_map:
        family_map, map_meta = load_component_map(Path(args.component_map))

    def unit_of(row: dict) -> str:
        if family_map:
            return family_map.get(row["id"]) or row.get("media_component") or row["id"]
        return row.get("media_component") or row["id"]

    rng = random.Random(args.seed)
    order = sorted(pool, key=lambda r: hashlib.sha256(f"{args.seed}:{r['id']}".encode()).hexdigest())
    rng.shuffle(order)  # seeded; the hash sort above already fixes the order

    # Quotas: floors first, then the remainder proportional to the pool.
    pool_counts = collections.Counter(r["primary_role"] for r in pool)
    quotas = {role: ROLE_FLOORS.get(role, 0) for role in ROLES}
    remainder = args.rows - sum(quotas.values())
    if args.rows == 0 or stratum_mode:
        quotas = {role: 0 for role in ROLES}
        remainder = 0
    elif remainder < 0:
        raise SystemExit("--rows is below the sum of the role floors")
    proportional_roles = [r for r in ROLES if r not in ROLE_FLOORS]
    weight = sum(pool_counts[r] for r in proportional_roles)
    if not stratum_mode and args.rows:
        for role in proportional_roles:
            quotas[role] += int(round(remainder * pool_counts[role] / weight))
        while sum(quotas.values()) > args.rows:
            quotas[max(proportional_roles, key=lambda r: quotas[r])] -= 1
        while sum(quotas.values()) < args.rows:
            quotas[min(proportional_roles, key=lambda r: quotas[r])] += 1

    calibration, used_components, total = [], set(), 0
    skipped = collections.Counter()
    for previous in args.exclude_selection:
        prior = json.loads(Path(previous).expanduser().read_text())
        for row in (prior.get("calibration") or []) + (prior.get("holdout") or []) \
                + (prior.get("text_only") or []):
            used_components.add(row["media_component"])
            if family_map and row["id"] in family_map:
                used_components.add(family_map[row["id"]])
    # Explicit exclusions, by row, by component, and by prompt term, each
    # widened to the whole exact-media component -- or, under --component-map,
    # to the whole corrected visual family.
    by_id = {r["id"]: r for r in pool}
    excluded_components: dict[str, str] = {}
    for row_id in args.exclude_row:
        if row_id not in by_id:
            raise SystemExit(f"--exclude-row {row_id} is not in the pool")
        excluded_components[unit_of(by_id[row_id])] = f"row {row_id}"
    for component in args.exclude_component:
        excluded_components[component] = f"component {component}"
    for term in args.exclude_prompt_term:
        for row in pool:
            if term.lower() in (raw[row["id"]].get("target_ir") or "").lower():
                excluded_components.setdefault(unit_of(row), f"prompt term {term!r}")
    used_components |= set(excluded_components)
    excluded_rows = sum(1 for r in pool if unit_of(r) in excluded_components)

    kept_holdout = []
    if args.keep_holdout:
        previous = json.loads(Path(args.keep_holdout).expanduser().read_text())
        for row in previous["holdout"]:
            if unit_of(by_id[row["id"]]) in excluded_components:
                continue
            kept_holdout.append(by_id[row["id"]])
    order = [r for r in order if unit_of(r) not in used_components]

    vision_budget = int(args.total_tokens * (1.0 - args.text_only_share))
    strata_report: dict = {}
    if stratum_mode:
        eligible = [r for r in order
                    if estimates[r["id"]]["tokens_est"] <= args.max_row_tokens
                    and not (args.max_vision_tokens_per_row
                             and estimates[r["id"]]["visual_tokens_est"]
                             > args.max_vision_tokens_per_row)
                    and not (family_map and r["id"] not in family_map
                             and (r.get("images") or r.get("videos")))]
        stratum = {r["id"]: stratum_of(r, raw[r["id"]].get("target_ir") or "", markers)
                   for r in order}
        occupied = sorted({stratum[r["id"]] for r in eligible})
        if not occupied:
            raise SystemExit("no eligible row survives the caps; nothing to stratify")
        targets = stratum_targets(occupied, explicit_shares,
                                  args.stratum_floor_share, vision_budget)
        queues = {name: [r for r in order if stratum[r["id"]] == name]
                  for name in targets}

        def admit(row: dict) -> str | None:
            est = estimates[row["id"]]
            if est["tokens_est"] > args.max_row_tokens:
                return "over_row_cap"
            if (args.max_vision_tokens_per_row
                    and est["visual_tokens_est"] > args.max_vision_tokens_per_row):
                return "over_vision_cap"
            if family_map and row["id"] not in family_map and (
                    row.get("images") or row.get("videos")):
                return "carries_media_but_unmapped"
            if family_map and unit_of(row) in used_components:
                return "unit_already_used"
            return None

        def take(row: dict) -> None:
            used_components.add(unit_of(row))

        calibration, total, fill, ceiling_bound = _fill_strata(
            queues, targets, estimates, admit, take, vision_budget,
            args.rows if args.rows else None)
        skipped.update(fill["skipped"])
        cheapest = {}
        for name in targets:
            costs = [estimates[r["id"]]["tokens_est"] for r in eligible
                     if stratum[r["id"]] == name]
            cheapest[name] = min(costs) if costs else None
        strata_report = {
            "axes": "primary_role x marker presence in target_ir",
            "markers": markers,
            "markers_observed_in_pool": {
                marker: sum(1 for r in pool
                            if marker in (raw[r["id"]].get("target_ir") or ""))
                for marker in markers},
            "markers_source": "vendor_config.h3_markers(): declared "
                              "additional_special_tokens with no added_tokens_decoder "
                              "entry, i.e. the ones the release appends past the stock "
                              "vocabulary",
            "axis_reach": (
                "the marker axis separates only the markers this pool actually "
                "carries; see markers_observed_in_pool. A marker at zero rows is "
                "one no selection from this pool can put weight on, whatever the "
                "stratum is called"),
            "language_overlay": (
                "skipped: the pool row schema records channel, primary_role, "
                "picture_roles and overlays "
                "(wide_or_tall/small_source/markers/audio_label/video_audio_track) "
                "and no language field. A language tag is observable inside the "
                "<d>[...] blocks of target_ir; deriving it here would be this "
                "selector inventing a pool field"),
            "floor_share": args.stratum_floor_share,
            "explicit_shares": explicit_shares,
            "vision_token_budget": vision_budget,
            "unspent_tokens": vision_budget - total,
            "materially_short_threshold": SHORT_OF_TARGET,
            "strata_materially_short": sorted(
                name for name in targets
                if sum(estimates[r["id"]]["tokens_est"] for r in calibration
                       if stratum[r["id"]] == name)
                < SHORT_OF_TARGET * targets[name]["tokens"]),
            "strata_whose_queue_ran_out": fill["exhausted"],
            "rows_ceiling_bound": ceiling_bound,
            "targets": targets,
            "target_below_cheapest_eligible_row": sorted(
                name for name, cost in cheapest.items()
                if cost is not None and targets[name]["tokens"] < cost),
            "achieved": {
                name: {
                    **_composition([r for r in calibration if stratum[r["id"]] == name],
                                   estimates),
                    "target_tokens": targets[name]["tokens"],
                    "target_share": round(targets[name]["share"], 4),
                    "achieved_share_of_budget": round(
                        sum(estimates[r["id"]]["tokens_est"] for r in calibration
                            if stratum[r["id"]] == name) / max(1, vision_budget), 4),
                    "cheapest_eligible_row_tokens_est": cheapest[name],
                }
                for name in sorted(targets)
            },
        }
    else:
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
            if (args.max_vision_tokens_per_row
                    and estimates[row["id"]]["visual_tokens_est"]
                    > args.max_vision_tokens_per_row):
                skipped["over_vision_cap"] += 1
                continue
            if total + est > vision_budget:
                skipped["over_total_budget"] += 1
                # This family cannot take another row within the budget; close it
                # so the loop terminates once every family is closed.
                queues[role] = [r for r in queues[role]
                                if total + estimates[r["id"]]["tokens_est"] <= vision_budget]
                continue
            calibration.append(row)
            used_components.add(unit_of(row))
            total += est
            taken[role] += 1

    # Text-only rows: their own budget, their own list, never merged into the
    # vision calibration (see the module docstring on the traced modality
    # envelope).
    text_only: list[dict] = []
    text_only_report: dict = {}
    if args.text_only_share:
        budget = args.total_tokens - vision_budget
        t_order = sorted(text_only_pool,
                         key=lambda r: hashlib.sha256(
                             f"{args.seed}:{r['id']}".encode()).hexdigest())
        t_stratum = {r["id"]: stratum_of(r, raw[r["id"]].get("target_ir") or "",
                                         markers or ["<d>"])
                     for r in t_order}
        t_eligible = [r for r in t_order
                      if estimates[r["id"]]["tokens_est"] <= args.max_row_tokens]
        t_occupied = sorted({t_stratum[r["id"]] for r in t_eligible})
        t_targets = {name: {"share": 1.0 / len(t_occupied),
                            "tokens": int(budget / len(t_occupied))}
                     for name in t_occupied} if t_occupied else {}
        t_queues = {name: [r for r in t_order if t_stratum[r["id"]] == name]
                    for name in t_targets}

        def t_admit(row: dict) -> str | None:
            if estimates[row["id"]]["tokens_est"] > args.max_row_tokens:
                return "over_row_cap"
            return None

        text_only, t_total, t_fill, _ = _fill_strata(
            t_queues, t_targets, estimates, t_admit, lambda row: None, budget, None)
        text_only_report = {
            "share_of_total_tokens": args.text_only_share,
            "token_budget": budget,
            "source": (Path(args.text_only_pool).name if args.text_only_pool
                       else TEXT_ONLY_POOL.name),
            "constraint": (
                "MEASURED, 2026-08-24_calibration_input_seam.md Q1a/Q1b: one "
                "`oneshot` call traces the model once and that trace fixes the "
                "modality envelope for the whole run. A text-only row after a "
                "vision trace raises; a vision row after a text-only trace loses "
                "its media silently. These rows are therefore a SEPARATE bundle "
                "and a separate run -- `build_native_h3_calibration_batch.py "
                "--population text-only` -- or they need a second traced graph, "
                "which is a change to llm-compressor's sequential pipeline. They "
                "are not merged into `calibration` here so the choice is made "
                "rather than inherited"),
            "targets": t_targets,
            "achieved": {**_composition(text_only, estimates),
                         "skipped": t_fill["skipped"]},
        }

    holdout = []
    holdout_provenance: dict[str, str] = {}

    def take_holdout(row, why: str) -> None:
        holdout.append(row)
        holdout_provenance[row["id"]] = why
        used_components.add(unit_of(row))

    for row in kept_holdout:
        take_holdout(row, "kept from previous holdout")

    # Small-source reserve first, one row per component, reference stills
    # ahead of keyframes (see the module docstring).
    def small_reference(row) -> bool:
        roles = row.get("picture_roles") or {}
        return any(roles.get(str(i)) != "keyframe" and w * h < 500_000
                   for i, (w, h) in enumerate(row.get("image_dimensions") or [], start=1))

    small_components = {unit_of(r) for r in holdout
                        if (r.get("overlays") or {}).get("small_source")}
    for row in sorted(order, key=lambda r: not small_reference(r)):
        if len(small_components) >= args.holdout_small_source or len(holdout) >= args.holdout:
            break
        if not (row.get("overlays") or {}).get("small_source"):
            continue
        if unit_of(row) in used_components:
            continue
        if estimates[row["id"]]["tokens_est"] > args.max_row_tokens:
            continue
        take_holdout(row, "small-source reserve")
        small_components.add(unit_of(row))
    if len(small_components) < args.holdout_small_source:
        raise SystemExit(f"holdout reserves {len(small_components)} small-source "
                         f"components; --holdout-small-source asks {args.holdout_small_source}")

    per_role_holdout = max(1, args.holdout // len(ROLES))
    for role in ROLES:
        taken = sum(1 for h in holdout if h["primary_role"] == role)
        for row in order:
            if taken >= per_role_holdout or len(holdout) >= args.holdout:
                break
            if row["primary_role"] != role or unit_of(row) in used_components:
                continue
            if any(h["id"] == row["id"] for h in holdout):
                continue
            if estimates[row["id"]]["tokens_est"] > args.max_row_tokens:
                continue
            take_holdout(row, "per-role fill")
            taken += 1

    calibration_components = {unit_of(r) for r in calibration}
    holdout_components = {unit_of(r) for r in holdout}
    if calibration_components & holdout_components:
        raise SystemExit("calibration and holdout share a media component")
    shared_units = family_disjointness_problems(calibration, holdout, unit_of)
    if shared_units:
        raise SystemExit("; ".join(shared_units))

    def describe(rows, provenance=None):
        out = []
        for r in rows:
            entry = {"id": r["id"], "primary_role": r["primary_role"],
                     "media_component": r.get("media_component") or r["id"],
                     "image_dimensions": r.get("image_dimensions"),
                     "videos": len(r.get("videos") or []),
                     "small_source": bool((r.get("overlays") or {}).get("small_source"))}
            if provenance:
                entry["selected_by"] = provenance[r["id"]]
            if stratum_mode or args.text_only_share:
                ir = raw[r["id"]].get("target_ir") or ""
                entry["stratum"] = stratum_of(r, ir, markers or ["<d>"])
                entry["markers_present"] = sorted(
                    m for m in (markers or ["<d>"]) if m in ir)
            if family_map:
                entry["corrected_family"] = family_map.get(r["id"])
            entry.update(estimates[r["id"]])
            out.append(entry)
        return out

    achieved = collections.Counter(r["primary_role"] for r in calibration)
    report = {
        "purpose": "sprint selection of the v2 calibration rows and a "
                   "component-disjoint holdout; estimates, superseded by the "
                   "builder's exact lengths",
        "producer": producer_provenance(__file__),
        "pool": {"file": pool_path.name, "rows": len(pool),
                 "dataset_revision": revision},
        "still_policy": args.still_policy,
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
        "exclusions": {"components": excluded_components, "rows_excluded": excluded_rows,
                       "previous_selections": args.exclude_selection,
                       "kept_holdout_from": args.keep_holdout},
        "holdout_small_source_components": sorted(small_components),
        "calibration": describe(calibration),
        "holdout": describe(holdout, holdout_provenance),
        "estimate_rules": {
            "reference_still": ("2048/short_edge, upscaling included" if args.still_policy == "upscale_2048"
                                else "min(1, 2048/short_edge)") + ", round to 32, tokens = (w/32)*(h/32)",
            "keyframe": "adapt to the 1344x768 canvas area at source aspect, round to 32",
            "video": f"{VIDEO_TOKENS_PER_SECOND} tokens per second of contract duration",
            "text": "released tokenizer count of the row's target_ir, the H3 prompt, no special tokens",
            "labels": f"{LABEL_OVERHEAD} tokens per media item",
        },
    }
    report["token_composition"] = _composition(calibration, estimates)
    if stratum_mode:
        report["strata"] = strata_report
        report["selection_mode"] = "token-balanced strata"
    else:
        report["selection_mode"] = "role row quotas"
    if args.max_vision_tokens_per_row:
        report["max_vision_tokens_per_row"] = {
            "cap": args.max_vision_tokens_per_row,
            "applies_to": "calibration only; a vision cap on the holdout would "
                          "choose which geometries the holdout grades",
            "rows_skipped": skipped.get("over_vision_cap", 0),
            "note": "rows over the cap are skipped, never truncated: a truncated "
                    "presentation is not what the deployed path emits",
        }
    if family_map:
        report["component_map"] = {
            **map_meta,
            "assignment": "corrected visual family, not exact media component",
            "widens": ["--exclude-row", "--exclude-component",
                       "--exclude-prompt-term", "--exclude-selection"],
            "calibration_families": len(calibration_components),
            "holdout_families": len(holdout_components),
            "one_calibration_row_per_family": True,
        }
    if args.text_only_share:
        report["text_only"] = describe(text_only)
        report["text_only_selection"] = text_only_report
    out = Path(args.out).expanduser().resolve()
    out.write_text(json.dumps(report, indent=2) + "\n")
    comp = report["token_composition"]
    print(f"calibration {len(calibration)} rows, est {total:,} tokens, longest est "
          f"{report['achieved']['longest_row_tokens_est']:,}, by role {dict(achieved)}, "
          f"skipped {dict(skipped)}")
    print(f"token composition: text {comp['text_tokens_est']:,} "
          f"({comp['text_share']:.1%}), visual {comp['visual_tokens_est']:,} "
          f"({comp['visual_share']:.1%}), labels {comp['label_tokens_est']:,}")
    if stratum_mode:
        worst = sorted(strata_report["achieved"].items(),
                       key=lambda kv: abs(kv[1]["achieved_share_of_budget"]
                                          - kv[1]["target_share"]))[-1]
        unnamed = sorted(v["share"] for name, v in strata_report["targets"].items()
                         if name not in explicit_shares)
        print(f"strata {len(strata_report['targets'])} occupied, "
              + (f"{len(explicit_shares)} named, unnamed share "
                 f"{unnamed[0]:.3f} each; " if unnamed else "all named; ")
              + f"furthest from target {worst[0]} at "
                f"{worst[1]['achieved_share_of_budget']:.3f} "
                f"against {worst[1]['target_share']:.3f}")
        short = strata_report["strata_materially_short"]
        if short:
            print(f"UNSPENT {strata_report['unspent_tokens']:,} of "
                  f"{vision_budget:,} tokens. Below {SHORT_OF_TARGET:.0%} of target: "
                  f"{', '.join(short)}. The pool has no further eligible row for "
                  f"them under these caps -- raise --max-vision-tokens-per-row, "
                  f"give them a smaller --stratum-token-share, or lower "
                  f"--total-tokens")
    if args.text_only_share:
        t = text_only_report["achieved"]
        print(f"text-only {t['rows']} rows, est {t['tokens_est']:,} tokens, "
              f"reported separately: they cannot share a traced run with vision rows")
    print(f"holdout {len(holdout)} rows, roles "
          f"{dict(collections.Counter(r['primary_role'] for r in holdout))}, "
          f"small-source components {len(small_components)}, "
          f"excluded {len(excluded_components)} components / {excluded_rows} rows")
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
