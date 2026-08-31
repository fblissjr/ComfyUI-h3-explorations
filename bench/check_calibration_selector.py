#!/usr/bin/env python3
"""Hold `bench/select_v2_calibration_rows.py`'s new selection paths to a standard.

The selector gained a token-balanced stratum design, a per-row vision cap, a
corrected-family assignment and a text-only share. All four are off unless
their flag is given, and the thing most likely to go wrong is not any of them
-- it is that adding them quietly moved the role-quota path that produced the
population a five-hour run is already fitted on.

Six arms, and only the first two are about the old path:

1. **Determinism.** Two runs under identical flags must write byte-identical
   files. The selection is a seeded hash order, so anything that leaks
   iteration order of a set or a dict into the result shows here.
2. **Pre-change revision.** The same default flags are run against the LAST
   COMMITTED REVISION OF THE SELECTOR BEFORE the stratum work
   (`BASELINE_REV`), extracted from git into a scratch tree, and the two
   selections must name the same rows in the same order. This is the control
   the repo prefers over asserting against numbers the check computed itself:
   the comparison is against code that predates the change, not against an
   expectation written next to it. Today's selection files live in the point
   session's `internal/`, so they cannot be the baseline; the code that wrote
   them can. If the revision is unreachable -- a shallow clone -- the arm says
   `not evaluated` and does not print green.
3. **Stratum token shares.** On a synthetic pool of uniform-cost rows, where
   hitting a token share exactly is achievable, every occupied stratum's
   achieved share must land within `SHARE_TOLERANCE` of its target. The
   synthetic pool exists because the real one cannot: `video-reference` holds
   twenty rows and a twelfth of the budget is more than they cost, so a real
   run is legitimately short and would make the tolerance meaningless.
4. **Vision cap.** A row whose estimated visual tokens exceed
   `--max-vision-tokens-per-row` must be absent from the selection, and the
   skip must be counted. The arm asserts its own precondition first: the same
   row IS selected with the cap absent. Without that, a row missing for budget
   reasons would read as a row the cap excluded.
5. **Family disjointness.** Under `--component-map` the split must place no
   family on both sides and no family twice in calibration. Because the
   selector cannot produce a violation by construction, the red control
   mutates the ASSIGNMENT instead and requires
   `family_disjointness_problems` to name the family -- and requires the
   unmutated assignment to be clean, on a split that actually has rows on both
   sides.
6. **Refusals.** A component map with its `caveat` removed, a stratum share
   naming a stratum no eligible row occupies, and shares that leave the
   unnamed strata below the floor must each be refused by exit code and by a
   message naming the cause. A map that does not say what it cannot see is the
   one that matters: the 2026-08-25 map's caveat is that same-brief-different-
   render relatedness among the images is unexamined.

CPU only, no CUDA, no server, no model weights -- but it needs the released
tokenizer directory for the token estimate, the same one the selector needs.
Run it with the ComfyUI venv python (`docs/comfy_notes.md`).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent
sys.path.insert(0, str(BENCH))

SELECTOR = BENCH / "select_v2_calibration_rows.py"
POOL = BENCH / "results" / "archive" / "v2_encoder" / "2026-08-24_h3_calibration_pool.jsonl"
COMPONENT_MAP = BENCH / "results" / "archive" / "v2_encoder" / "2026-08-25_pool_component_map_corrected.json"
DEFAULT_SOURCE = REPO / "coderef" / "llm-compressor" / "models" / "qwen3-vl-32b-bf16"

# The last commit that touched the selector before the stratum work. Pinned
# rather than HEAD: once this change is committed, HEAD becomes the changed
# file and the comparison would be the new code against itself.
BASELINE_REV = "7bc07ede47cddc55e1350f838d54920d3f4a6867"

# The default invocation both sides of arm 2 run. Small enough to be quick,
# large enough that every role floor binds.
DEFAULT_FLAGS = ["--rows", "35", "--holdout", "12"]

SHARE_TOLERANCE = 0.02  # absolute, against a target share of 1/strata


def source_dir() -> Path | None:
    named = os.environ.get("H3_BF16_ENCODER_DIR")
    for candidate in ([Path(named)] if named else []) + [DEFAULT_SOURCE]:
        if candidate.is_dir() and (candidate / "tokenizer_config.json").is_file():
            return candidate
    return None


def run_selector(script: Path, flags: list[str], out: Path, source: Path,
                 cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *flags,
         "--source-dir", str(source), "--out", str(out)],
        capture_output=True, text=True, cwd=str(cwd or REPO),
    )


# --------------------------------------------------------------------------
# synthetic pool: uniform-cost rows, so a token share is exactly achievable


SYNTH_ROLES = ("multi-image-2-3", "keyframe-only", "single-image", "video-reference")
SYNTH_PER_STRATUM = 60
SYNTH_SMALL = (512, 512)     # 256 visual tokens under either still policy
SYNTH_LARGE = (2048, 2048)   # 4,096 visual tokens: over the vision cap, under the row cap


def write_synthetic(root: Path) -> tuple[Path, Path, list[str]]:
    """A pool whose strata are equal-cost, plus one oversized row per stratum."""
    pool_rows, raw_rows, oversized = [], [], []
    for role in SYNTH_ROLES:
        for markers in (True, False):
            for index in range(SYNTH_PER_STRATUM + 1):
                big = index == SYNTH_PER_STRATUM
                row_id = f"synth-{role}-{int(markers)}-{index:03d}"
                width, height = SYNTH_LARGE if big else SYNTH_SMALL
                if big:
                    oversized.append(row_id)
                pool_rows.append({
                    "id": row_id,
                    "primary_role": role,
                    "picture_roles": {"1": "reference"},
                    "image_dimensions": [[width, height]],
                    "images": [f"media/{row_id}.jpg"],
                    "videos": [],
                    "media_component": row_id,
                    "overlays": {"small_source": False, "markers": {}},
                })
                body = "a medium shot establishes the room and holds. " * 4
                target = (f"<d>[English] line {index}.</d> {body}"
                          if markers else body)
                raw_rows.append({
                    "id": row_id,
                    "target_ir": target,
                    "messages": [{"role": "user",
                                  "content": "duration_seconds: 5.0"}],
                })
    pool = root / "pool.jsonl"
    pool.write_text("".join(json.dumps(r) + "\n" for r in pool_rows))
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "train.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in raw_rows))
    return pool, root, oversized


# --------------------------------------------------------------------------
# arms


def arm_determinism(source: Path, tmp: Path) -> tuple[dict, list[str]]:
    first, second = tmp / "det_a.json", tmp / "det_b.json"
    for out in (first, second):
        done = run_selector(SELECTOR, DEFAULT_FLAGS, out, source)
        if done.returncode:
            return {"status": "selector failed"}, [
                f"determinism arm: selector exited {done.returncode}: "
                f"{done.stderr.strip()[-400:]}"]
    same = first.read_bytes() == second.read_bytes()
    return ({"identical": same, "bytes": first.stat().st_size},
            [] if same else ["two runs under identical flags differ"])


def arm_baseline_revision(source: Path, tmp: Path) -> tuple[dict, list[str]]:
    probe = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e",
                            f"{BASELINE_REV}:bench/select_v2_calibration_rows.py"],
                           capture_output=True, text=True)
    if probe.returncode:
        return ({"status": "not evaluated",
                 "revision": BASELINE_REV,
                 "why": "the pre-change revision of the selector is not in this "
                        "checkout (a shallow clone reaches no history), so the "
                        "role-quota path was compared against nothing"}, [])
    show = subprocess.run(["git", "-C", str(REPO), "show",
                           f"{BASELINE_REV}:bench/select_v2_calibration_rows.py"],
                          capture_output=True, text=True, check=True)
    # The baseline resolves its pool and its sibling imports from its own
    # directory, so it runs from a scratch tree that mirrors bench/ by symlink
    # rather than from bench/ itself. Nothing is written into the repo.
    scratch = tmp / "baseline_bench"
    scratch.mkdir()
    for sibling in BENCH.glob("*.py"):
        (scratch / sibling.name).symlink_to(sibling)
    # `results` is rebuilt entry by entry rather than symlinked wholesale, so
    # the archived records can be re-exposed at the FLAT names the baseline
    # revision knew. The pool moved to `results/archive/v2_encoder/` on
    # 2026-08-31 when the v2 lane was archived; the pre-change selector predates
    # that and hardcodes the flat path, which is correct for what it is -- a
    # frozen copy of how the code looked. Rewriting history to match today's
    # layout would defeat the control. Nothing is written into the repo.
    results = scratch / "results"
    results.mkdir()
    for entry in (BENCH / "results").iterdir():
        (results / entry.name).symlink_to(entry)
    for archived in (BENCH / "results" / "archive" / "v2_encoder").iterdir():
        flat = results / archived.name
        if not flat.exists():
            flat.symlink_to(archived)
    baseline = scratch / "baseline_selector.py"
    baseline.write_text(show.stdout)
    out_old, out_new = tmp / "base_old.json", tmp / "base_new.json"
    old = run_selector(baseline, DEFAULT_FLAGS, out_old, source)
    new = run_selector(SELECTOR, DEFAULT_FLAGS, out_new, source)
    problems = []
    for label, done in (("pre-change", old), ("current", new)):
        if done.returncode:
            problems.append(f"{label} selector exited {done.returncode}: "
                            f"{done.stderr.strip()[-400:]}")
    if problems:
        return {"status": "selector failed"}, problems
    a, b = json.loads(out_old.read_text()), json.loads(out_new.read_text())
    record = {"revision": BASELINE_REV, "flags": DEFAULT_FLAGS}
    for side in ("calibration", "holdout"):
        ids_a = [r["id"] for r in a[side]]
        ids_b = [r["id"] for r in b[side]]
        record[f"{side}_rows"] = len(ids_a)
        if ids_a != ids_b:
            problems.append(
                f"the role-quota path moved: {side} was {ids_a} under "
                f"{BASELINE_REV[:8]}, is {ids_b} now")
    for key in ("quotas", "achieved"):
        if a[key] != b[key]:
            problems.append(f"the role-quota path's {key} moved: {a[key]} -> {b[key]}")
    return record, problems


def arm_stratum_shares(source: Path, tmp: Path) -> tuple[dict, list[str]]:
    root = tmp / "synth_shares"
    root.mkdir()
    pool, dataset, _ = write_synthetic(root)
    out = tmp / "synth_shares.json"
    done = run_selector(SELECTOR, [
        "--stratum", "--rows", "1000", "--holdout", "0",
        "--holdout-small-source", "0", "--total-tokens", "80000",
        "--max-vision-tokens-per-row", "1000",
        "--pool", str(pool), "--dataset-root", str(dataset)], out, source)
    if done.returncode:
        return {"status": "selector failed"}, [
            f"stratum arm: selector exited {done.returncode}: "
            f"{done.stderr.strip()[-400:]}"]
    report = json.loads(out.read_text())
    strata = report["strata"]["achieved"]
    problems = []
    if len(strata) != len(SYNTH_ROLES) * 2:
        problems.append(f"expected {len(SYNTH_ROLES) * 2} occupied strata, "
                        f"got {sorted(strata)}")
    worst = 0.0
    for name, got in sorted(strata.items()):
        delta = abs(got["achieved_share_of_budget"] - got["target_share"])
        worst = max(worst, delta)
        if delta > SHARE_TOLERANCE:
            problems.append(
                f"{name}: achieved share {got['achieved_share_of_budget']:.4f} "
                f"against target {got['target_share']:.4f}, off by {delta:.4f} "
                f"(tolerance {SHARE_TOLERANCE})")
    return ({"strata": len(strata), "tolerance": SHARE_TOLERANCE,
             "worst_absolute_share_error": round(worst, 4),
             "achieved": {k: v["achieved_share_of_budget"] for k, v in strata.items()}},
            problems)


def arm_vision_cap(source: Path, tmp: Path) -> tuple[dict, list[str]]:
    root = tmp / "synth_cap"
    root.mkdir()
    pool, dataset, oversized = write_synthetic(root)
    common = ["--stratum", "--rows", "1000", "--holdout", "0",
              "--holdout-small-source", "0", "--total-tokens", "400000",
              "--pool", str(pool), "--dataset-root", str(dataset)]
    uncapped, capped = tmp / "cap_off.json", tmp / "cap_on.json"
    for flags, out in ((common, uncapped),
                       (common + ["--max-vision-tokens-per-row", "1000"], capped)):
        done = run_selector(SELECTOR, flags, out, source)
        if done.returncode:
            return {"status": "selector failed"}, [
                f"vision-cap arm: selector exited {done.returncode}: "
                f"{done.stderr.strip()[-400:]}"]
    off = {r["id"] for r in json.loads(uncapped.read_text())["calibration"]}
    on_report = json.loads(capped.read_text())
    on = {r["id"] for r in on_report["calibration"]}
    problems = []
    # Precondition: the cap must be what removes them. A row absent from BOTH
    # runs proves nothing, and this arm has no subject unless the uncapped run
    # actually took one.
    reachable = sorted(set(oversized) & off)
    if not reachable:
        problems.append(
            "vision-cap arm reached no subject: the uncapped run selected none "
            "of the oversized rows, so their absence under the cap is not "
            "evidence the cap did anything")
    leaked = sorted(set(oversized) & on)
    if leaked:
        problems.append(f"rows over --max-vision-tokens-per-row were selected: {leaked}")
    counted = on_report.get("max_vision_tokens_per_row", {}).get("rows_skipped", 0)
    if counted < len(reachable):
        problems.append(
            f"the cap skipped {counted} rows but {len(reachable)} oversized rows "
            f"were selectable without it; the skip is not being counted")
    return ({"oversized_rows": len(oversized),
             "selected_without_cap": len(reachable),
             "selected_with_cap": len(leaked),
             "rows_skipped_recorded": counted}, problems)


def arm_family_disjointness(source: Path, tmp: Path) -> tuple[dict, list[str]]:
    from select_v2_calibration_rows import family_disjointness_problems

    if not COMPONENT_MAP.is_file():
        return ({"status": "not evaluated",
                 "why": f"{COMPONENT_MAP.name} is absent, so family assignment "
                        f"was not exercised"}, [])
    out = tmp / "family.json"
    done = run_selector(SELECTOR, [
        "--stratum", "--rows", "60", "--holdout", "12",
        "--total-tokens", "200000",
        "--component-map", str(COMPONENT_MAP)], out, source)
    if done.returncode:
        return {"status": "selector failed"}, [
            f"family arm: selector exited {done.returncode}: "
            f"{done.stderr.strip()[-400:]}"]
    report = json.loads(out.read_text())
    cal, hold = report["calibration"], report["holdout"]
    problems = []
    if not cal or not hold:
        problems.append("family arm reached no subject: the split has an empty "
                        "side, so disjointness is true by construction")
    family = {r["id"]: r["corrected_family"] for r in cal + hold}
    if any(v is None for v in family.values()):
        problems.append("a selected row carries no corrected_family")
    clean = family_disjointness_problems(cal, hold, lambda r: family[r["id"]])
    if clean:
        problems.append(f"the emitted split is not family-disjoint: {clean}")
    if len({family[r['id']] for r in cal}) != len(cal):
        problems.append("two calibration rows share one corrected family")
    # Red control: move one holdout row into a calibration row's family and
    # require the predicate to name it. The selector cannot emit this split, so
    # the assertion has to be reachable from outside the selector.
    mutated = dict(family)
    mutated[hold[0]["id"]] = family[cal[0]["id"]]
    caught = family_disjointness_problems(cal, hold, lambda r: mutated[r["id"]])
    if not caught:
        problems.append(
            "RED CONTROL FAILED: a holdout row moved into a calibration row's "
            "family and family_disjointness_problems reported nothing")
    # Second control: an exclusion under the map must widen to the whole
    # family, not the exact component. Pick a family with more than one row.
    pool = [json.loads(line) for line in POOL.read_text().splitlines()]
    loaded = json.loads(COMPONENT_MAP.read_text())["corrected_component_by_row"]
    sizes: dict[str, list[str]] = {}
    for row in pool:
        sizes.setdefault(loaded[row["id"]], []).append(row["id"])
    multi = sorted((f for f, rows in sizes.items() if len(rows) > 2),
                   key=lambda f: (-len(sizes[f]), f))
    widened = {}
    if not multi:
        problems.append("exclusion-widening control reached no subject: the map "
                        "has no family with more than two rows")
    else:
        target_family = multi[0]
        victim = sorted(sizes[target_family])[0]
        out_ex = tmp / "family_excluded.json"
        done = run_selector(SELECTOR, [
            "--stratum", "--rows", "60", "--holdout", "12",
            "--total-tokens", "200000",
            "--component-map", str(COMPONENT_MAP),
            "--exclude-row", victim], out_ex, source)
        if done.returncode:
            problems.append(f"exclusion-widening control: selector exited "
                            f"{done.returncode}: {done.stderr.strip()[-300:]}")
        else:
            excluded = json.loads(out_ex.read_text())
            hit = [r["id"] for r in excluded["calibration"] + excluded["holdout"]
                   if loaded.get(r["id"]) == target_family]
            baseline_hit = [r["id"] for r in cal + hold
                            if loaded.get(r["id"]) == target_family]
            if not baseline_hit:
                problems.append(
                    "exclusion-widening control reached no subject: no row of "
                    f"family {target_family} was selected before the exclusion, "
                    f"so its absence after proves nothing")
            if hit:
                problems.append(
                    f"--exclude-row {victim} under a component map left "
                    f"{len(hit)} row(s) of its family {target_family} in the "
                    f"selection: {hit}")
            widened = {"family": target_family,
                       "rows_in_family": len(sizes[target_family]),
                       "selected_before": len(baseline_hit),
                       "selected_after": len(hit)}
    return ({"calibration_rows": len(cal), "holdout_rows": len(hold),
             "calibration_families": len({family[r["id"]] for r in cal}),
             "mutation_caught": bool(caught),
             "exclusion_widening": widened,
             "map_caveat_present": bool(
                 json.loads(COMPONENT_MAP.read_text()).get("caveat"))}, problems)


def arm_refusals(source: Path, tmp: Path) -> tuple[dict, list[str]]:
    problems, record = [], {}
    if COMPONENT_MAP.is_file():
        stripped = json.loads(COMPONENT_MAP.read_text())
        stripped.pop("caveat", None)
        no_caveat = tmp / "map_no_caveat.json"
        no_caveat.write_text(json.dumps(stripped))
        done = run_selector(SELECTOR, [
            "--stratum", "--rows", "10", "--holdout", "0",
            "--holdout-small-source", "0",
            "--component-map", str(no_caveat)], tmp / "refuse_map.json", source)
        record["caveatless_map_exit"] = done.returncode
        if done.returncode == 0:
            problems.append("a component map with no `caveat` was accepted")
        elif "caveat" not in (done.stderr + done.stdout):
            problems.append("the caveatless-map refusal does not name the caveat")
    else:
        record["caveatless_map_exit"] = "not evaluated: map absent"
    unknown = run_selector(SELECTOR, [
        "--stratum-token-share", "no-such-role|markers=0.5",
        "--rows", "10", "--holdout", "0", "--holdout-small-source", "0"],
        tmp / "refuse_share.json", source)
    record["unknown_stratum_exit"] = unknown.returncode
    if unknown.returncode == 0:
        problems.append("a --stratum-token-share naming an unoccupied stratum "
                        "was accepted")
    floor = run_selector(SELECTOR, [
        "--stratum-token-share", "keyframe-only|markers=0.99",
        "--rows", "10", "--holdout", "0", "--holdout-small-source", "0"],
        tmp / "refuse_floor.json", source)
    record["floor_breach_exit"] = floor.returncode
    if floor.returncode == 0:
        problems.append("shares leaving the unnamed strata below the floor "
                        "were accepted")
    return record, problems


ARMS = (
    ("determinism", arm_determinism),
    ("pre_change_revision", arm_baseline_revision),
    ("stratum_token_shares", arm_stratum_shares),
    ("vision_cap", arm_vision_cap),
    ("family_disjointness", arm_family_disjointness),
    ("refusals", arm_refusals),
)


def main() -> int:
    source = source_dir()
    if source is None:
        print(f"FAIL no tokenizer directory: set H3_BF16_ENCODER_DIR or place the "
              f"released encoder at {DEFAULT_SOURCE.relative_to(REPO)}")
        return 1
    failures: list[str] = []
    unevaluated: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for name, arm in ARMS:
            record, problems = arm(source, tmp)
            failures.extend(problems)
            status = record.get("status") if isinstance(record, dict) else None
            if status == "not evaluated":
                unevaluated.append(name)
                print(f"-- {name}: NOT EVALUATED -- {record.get('why')}")
            elif problems:
                print(f"FAIL {name}")
                for problem in problems:
                    print(f"     {problem}")
            else:
                print(f"ok   {name}: {json.dumps(record, sort_keys=True)[:300]}")
    if unevaluated:
        print(f"\n{len(unevaluated)} arm(s) could not be evaluated: "
              f"{', '.join(unevaluated)}. A green line above does not cover them.")
    if failures:
        print(f"\nFAIL {len(failures)} problem(s)")
        return 1
    print("\nok all arms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
