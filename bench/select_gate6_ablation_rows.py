#!/usr/bin/env python3
"""Select the render population for the Gate 6 reference-upscale ablation.

`active_plan.md` names the first Gate 6 experiment on v2: (A) no-upscale on
both paths, (B) the Qwen view at a 2048 short edge with the VAE keeping stage
one, (C) full parity at 2048. Matched seeds, judged blind as a distribution,
with the per-arm row and VRAM cost recorded beside the verdict.

**The arms differ if and only if a still's SHORT edge is below 2048**, which is
computed here from the shipped sizing code rather than assumed. `max`/no-upscale
clamps its scale with `min(1.0, 2048/short_edge)`, so a source already at or
past a 2048 short edge is untouched by every arm; `qwen_view_size` and
`allow_upscale=True` both drive the short edge *to* 2048, up or down. A row
whose stills all sit at or above that boundary produces bit-identical
conditioning in all three arms and cannot inform the question.

That is why the population is stratified by **upscale factor**, `2048 /
shortest short edge in the row`, and not by source size:

    extreme    factor >= 4     the 2048 view is mostly manufactured pixels
    moderate   2 <= factor < 4
    mild       factor < 2      the common serving case, a 2048x1152 still
    control    factor == 1     no arm differs; a null control, see below

**The factor is computed over a row's REFERENCE stills only.** A keyframe takes
the resolved target canvas and no arm touches it, so counting keyframe
dimensions puts a row in a band its reference stills are not in -- the first run
of this selector did exactly that and returned a "mild" row whose three arms
were bit-identical. For the same reason a keyframe-ONLY row is excluded
outright: the still policy reaches none of its pictures, so it cannot
discriminate between arms at any factor.

**The control row is deliberate and is not a wasted render.** Its three arms
receive bit-identical conditioning, so any arm-labelled difference a judge
reports on it is a labelling, seeding or pipeline error rather than a geometry
effect. It costs one row to make the rest of the session falsifiable. It is
marked `null_control` in the output so nobody scores it as evidence about
geometry.

Selection constraints, all enforced and reported:

- every row's visual family, under the corrected component map, is disjoint
  from the calibration bundle and from the holdout bundles;
- the rows are in distinct families from each other, so the population is not
  internally redundant;
- single-reference and multi-reference rows both appear, and at least one
  keyframe-plus-reference row, because keyframes take canvas geometry and are
  therefore untouched by the still policy -- a row mixing both is the only one
  that exercises two geometries at once.

Deterministic: rows are chosen by a seeded shuffle within each stratum. CPU
only, no CUDA, no model, no server. Run it with the ComfyUI venv python
(`docs/comfy_notes.md`): it imports the shipped reference geometry.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))

POOL = BENCH / "results" / "2026-08-24_h3_calibration_pool.jsonl"
MAP = BENCH / "results" / "2026-08-25_pool_component_map_corrected.json"

STRATA = ("extreme", "moderate", "mild", "control")

# stratum -> the roles to fill it with, in order. Chosen so the population
# covers single and multi reference in the bands where the arms differ most,
# and spends its two keyframe-plus-reference rows where a mixed-geometry row
# is most informative.
#
# **There is no `extreme` single-image slot because the source has none.** No
# single-reference row in the pool carries a still short enough to reach a 4x
# upscale, so that slot was unfillable rather than unfilled; the third extreme
# slot goes to a second multi-image row instead, because `extreme` is the band
# the question is most about and two rows is thin. An unfilled slot is still
# reported rather than quietly dropped -- see `unfilled_slots`.
PLAN = (
    ("extreme", "multi-image-2-3"),
    ("extreme", "multi-image-4-9"),
    ("extreme", "multi-image-2-3"),
    ("moderate", "single-image"),
    ("moderate", "multi-image-2-3"),
    ("moderate", "multi-image-4-9"),
    ("moderate", "keyframe-plus-reference"),
    ("mild", "single-image"),
    ("mild", "multi-image-2-3"),
    ("mild", "multi-image-4-9"),
    ("mild", "keyframe-plus-reference"),
    ("control", None),
)


def _arms():
    """The three arms, from the shipped sizing code.

    Imported through `build_native_h3_calibration_batch._repo_module` because
    this repo's modules use relative imports and `comfy_extras` does a bare
    `import nodes` that this repo's own `nodes.py` would shadow. That module
    already owns both traps; re-solving them here would be a second copy.
    """
    import build_native_h3_calibration_batch as builder

    geometry = builder._repo_module("reference_geometry")
    conditioning = builder._repo_module("reference_conditioning")

    def arms(width: int, height: int) -> dict:
        no_upscale = geometry.fit_reference_image(
            width, height, size_policy="max", short_edge=2048, allow_upscale=False)
        upscaled = geometry.fit_reference_image(
            width, height, size_policy="max", short_edge=2048, allow_upscale=True)
        qwen_only = conditioning.qwen_view_size(width, height, 2048)
        return {
            "A_no_upscale": {"vae": list(no_upscale), "qwen": list(no_upscale)},
            "B_qwen_2048": {"vae": list(no_upscale), "qwen": list(qwen_only)},
            "C_full_parity": {"vae": list(upscaled), "qwen": list(upscaled)},
        }

    return arms


def visual_tokens(size) -> int:
    """Merged Qwen tokens for one still at a 32-pixel patch grid."""
    return (size[0] // 32) * (size[1] // 32)


def stratum_of(short_edges: list[int]) -> str | None:
    if not short_edges:
        return None
    smallest = min(short_edges)
    if smallest >= 2048:
        return "control"
    factor = 2048 / smallest
    if factor >= 4:
        return "extreme"
    if factor >= 2:
        return "moderate"
    return "mild"


def keyframe_ordinals(row: dict) -> set[int]:
    return {int(n) for n, role in (row.get("picture_roles") or {}).items()
            if role == "keyframe"}


def reference_short_edges(row: dict) -> list[int]:
    """Short edges of the row's reference stills. Keyframes are excluded.

    A keyframe is placed on the resolved target canvas by
    `MiniMaxH3ImageToVideo`, so the still policy never reaches it and its
    dimensions say nothing about which band the row belongs in.
    """
    keyframes = keyframe_ordinals(row)
    return [min(dim) for index, dim in enumerate(row["image_dimensions"], start=1)
            if index not in keyframes]


def describe(row: dict, stratum: str) -> str:
    edges = reference_short_edges(row)
    smallest = min(edges)
    factor = 2048 / smallest
    role = row["primary_role"]
    if stratum == "control":
        return (f"null control: every reference still already sits at or past "
                f"a 2048 short edge (smallest is {smallest}), so all three arms build "
                f"bit-identical conditioning. Any arm-labelled difference a "
                f"judge reports here is a labelling or seeding error, not "
                f"geometry.")
    where = {
        "extreme": (f"the 2048 view is mostly manufactured pixels: the "
                    f"shortest edge is {smallest}, a {factor:.1f}x linear "
                    f"upscale"),
        "moderate": (f"a {factor:.1f}x linear upscale from a {smallest}-pixel "
                     f"short edge, between the two extremes"),
        "mild": (f"the common serving case: a {smallest}-pixel short edge, a "
                 f"{factor:.1f}x upscale, which is what most shipped graphs "
                 f"actually feed"),
    }[stratum]
    extra = ""
    if role == "keyframe-plus-reference":
        extra = (" Mixed geometry: its keyframe takes the target canvas and is "
                 "untouched by the still policy, so this row shows the arms "
                 "moving one reference and not the other in a single request.")
    elif role == "single-image":
        extra = " Single reference: the identity question with nothing to average against."
    return f"{where}.{extra}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=POOL)
    parser.add_argument("--component-map", type=Path, default=MAP)
    parser.add_argument("--bundle", action="append", default=[], type=Path,
                        help="calibration bundle whose families are excluded")
    parser.add_argument("--holdout", action="append", default=[], type=Path,
                        help="holdout bundle whose families are excluded")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--out", type=Path,
                        default=BENCH / "results" / "2026-08-25_gate6_upscale_ablation_rows.json")
    args = parser.parse_args()

    from build_h3_calibration_pool import pinned_snapshot

    root, revision = pinned_snapshot()
    arms = _arms()
    # One pass over the split, keyed by id. The first version searched the whole
    # file per row with a substring match, which is quadratic and would happily
    # match an id appearing inside some other field.
    source = {}
    for line in (root / "data" / "train.jsonl").read_text().splitlines():
        record = json.loads(line)
        source[record["id"]] = record
    pool = {json.loads(line)["id"]: json.loads(line)
            for line in args.pool.read_text().splitlines()}
    family = json.loads(args.component_map.read_text())["corrected_component_by_row"]

    excluded_rows: dict[str, list[str]] = {}
    for kind, paths in (("calibration", args.bundle), ("holdout", args.holdout)):
        for path in paths:
            manifest = path / "presentation.json"
            if not manifest.is_file():
                print(f"{path} has no presentation.json")
                return 2
            excluded_rows.setdefault(kind, []).extend(
                r["row_id"] for r in json.loads(manifest.read_text())["rows"])
    if not excluded_rows:
        print("no bundles given; a disjointness claim with nothing to be "
              "disjoint from is not a claim")
        return 2

    blocked = {family[r] for rows in excluded_rows.values() for r in rows
               if r in family}
    unmapped = [r for rows in excluded_rows.values() for r in rows
                if r not in family]

    candidates: dict[str, dict[str, list[dict]]] = {}
    for row in pool.values():
        if row["video_count"]:
            continue          # the ablation is about still geometry
        if family.get(row["id"]) in blocked:
            continue
        dims = row["image_dimensions"]
        if not dims or len(dims) != row["image_count"]:
            continue
        edges = reference_short_edges(row)
        if not edges:
            continue          # keyframe-only: the still policy reaches nothing
        stratum = stratum_of(edges)
        if stratum is None:
            continue
        candidates.setdefault(stratum, {}).setdefault(
            row["primary_role"], []).append(row)

    rng = random.Random(args.seed)
    for stratum in candidates:
        for role in candidates[stratum]:
            candidates[stratum][role].sort(key=lambda r: r["id"])
            rng.shuffle(candidates[stratum][role])

    chosen: list[dict] = []
    used_families: set[str] = set()
    unfilled: list[str] = []
    for stratum, role in PLAN:
        buckets = candidates.get(stratum, {})
        pool_for_slot = (buckets.get(role, []) if role
                         else [r for rows in buckets.values() for r in rows])
        pick = next((r for r in pool_for_slot
                     if family.get(r["id"]) not in used_families), None)
        if pick is None:
            unfilled.append(f"{stratum}/{role or 'any'}")
            continue
        used_families.add(family[pick["id"]])
        chosen.append({"row": pick, "stratum": stratum})

    entries = []
    for item in chosen:
        row, stratum = item["row"], item["stratum"]
        keyframes = keyframe_ordinals(row)
        media = []
        per_arm_tokens = {"A_no_upscale": 0, "B_qwen_2048": 0, "C_full_parity": 0}
        for index, (rel, dim) in enumerate(
                zip(row["images"], row["image_dimensions"]), start=1):
            width, height = dim
            is_keyframe = index in keyframes
            record = {
                "label": f"<Picture {index}>",
                "role": "keyframe" if is_keyframe else "reference-still",
                "path": rel,
                "sha256": row["media_sha256"].get(rel),
                "source_size": [width, height],
            }
            if is_keyframe:
                record["note"] = ("keyframe: takes the resolved target canvas, "
                                  "so no arm moves it")
            else:
                sizes = arms(width, height)
                record["arms"] = sizes
                record["arms_identical"] = (
                    sizes["A_no_upscale"] == sizes["B_qwen_2048"] ==
                    sizes["C_full_parity"])
                for arm, view in sizes.items():
                    per_arm_tokens[arm] += visual_tokens(view["qwen"])
            media.append(record)
        entries.append({
            "row_id": row["id"],
            "stratum": stratum,
            "null_control": stratum == "control",
            "primary_role": row["primary_role"],
            "family": family[row["id"]],
            "image_count": row["image_count"],
            "prompt_sha256": row["prompt_sha256"],
            "prompt_bytes": len(
                ((source.get(row["id"]) or {}).get("target_ir") or "")
                .encode("utf-8")),
            "overlays": row["overlays"],
            "why": describe(row, stratum),
            "media": media,
            "qwen_visual_tokens_by_arm": per_arm_tokens,
        })

    document = {
        "purpose": ("render population for the Gate 6 reference-upscale "
                    "ablation: arms A no-upscale, B Qwen-only 2048, C full "
                    "parity, matched seeds, judged blind as a distribution"),
        "producer": Path(__file__).name,
        "seed": args.seed,
        "dataset": {"repo_id": "StellarVoyager/H3-IR", "revision": revision},
        "component_map": args.component_map.name,
        "arms": {
            "A_no_upscale": "size_policy=max, allow_upscale=False, one view for both towers",
            "B_qwen_2048": "size_policy=max, allow_upscale=False for the VAE; "
                           "qwen_short_edge=2048 gives the encoder its own view",
            "C_full_parity": "size_policy=max, allow_upscale=True, one 2048 view for both towers",
        },
        "where_the_arms_differ": (
            "if and only if a still's SHORT edge is below 2048. Computed from "
            "the shipped sizing code, not assumed: no-upscale clamps with "
            "min(1.0, 2048/short_edge), so a source at or past a 2048 short "
            "edge is untouched by every arm."),
        "disjointness": {
            "basis": "corrected visual family, not media file",
            "calibration_rows_excluded": len(excluded_rows.get("calibration", [])),
            "holdout_rows_excluded": len(excluded_rows.get("holdout", [])),
            "families_blocked": len(blocked),
            "bundle_rows_absent_from_the_map": unmapped,
            "selected_families": sorted(used_families),
            "shared_with_calibration_or_holdout": sorted(
                {e["family"] for e in entries} & blocked),
            "selected_families_are_distinct": len(used_families) == len(entries),
        },
        "unfilled_slots": unfilled,
        "rows": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")

    print(f"{len(entries)} row(s) at {revision[:12]}")
    for entry in entries:
        tokens = entry["qwen_visual_tokens_by_arm"]
        ratio = (tokens["C_full_parity"] / tokens["A_no_upscale"]
                 if tokens["A_no_upscale"] else 1.0)
        flag = "  NULL CONTROL" if entry["null_control"] else ""
        print(f"  {entry['stratum']:9s} {entry['primary_role']:24s} "
              f"{entry['row_id']}  {entry['image_count']} still(s)  "
              f"A={tokens['A_no_upscale']:6d} C={tokens['C_full_parity']:6d} "
              f"tok (x{ratio:.2f}){flag}")
    if unfilled:
        print(f"  unfilled slots: {unfilled}")
    shared = document["disjointness"]["shared_with_calibration_or_holdout"]
    print(f"\nfamily-disjoint from calibration and holdout: "
          f"{'YES' if not shared else f'NO -- shares {shared}'}")
    if unmapped:
        print(f"  {len(unmapped)} bundle row(s) are absent from the component "
              f"map, so their families could not be blocked: {unmapped}")
    return 1 if shared or unmapped else 0


if __name__ == "__main__":
    raise SystemExit(main())
