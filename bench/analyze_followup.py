#!/usr/bin/env python3
"""Run the measurement reads on the 2026-09-26 follow-up batch as its arms land.

Reads `data/results/2026-09-26_followup.jsonl` (`bench/followup_arms.json`'s
rows), finds each arm's clip and saved video latent on the output share by its
label (`run_graph_arms.py` appends `_<label>` to every filename prefix), and
runs the tool each prediction names
(`bench/results/2026-09-26_distill_run_predictions.md`):

- `looks`: `measure_clip_tone.py` and `measure_clip_temporal.py` on every
  `look_*` arm (O1's dark blocking, F7's pull toward a typical look).
- `ladder`: `latent_path_distance.py` per scene, with PDD8 (exact) as the
  reference and PDD4, PDD6 and merged PDD8 against it (P1, P5).
- `vsa`: `measure_clip_temporal.py`, `measure_block_period.py` and
  `measure_clip_tone.py` on FastH3 with VSA on and off (P7).
- `spec`: the specificity ladder for the owner's O2 (`t2va_spec_typical`,
  `_specific`, `_unusual`; `bench/followup_*` from `a8432ca8`). Per rung and
  model (FlashGen, PDD8, FastH3), `measure_clip_temporal.py`,
  `measure_clip_resolution.py` and `measure_clip_tone.py`, so the owner's
  read of quality per rung sits beside the numbers. The seed-collapse test it
  replaced was cancelled unrendered (owner: seed is the wrong axis).
- `reverse`: the reverse step switch on `subway_chase`
  (`bench/followup_reverse_switch_arms.json`). Both handoffs and FlashGen
  alone are read against PDD8 (exact) as the reference.
- `swap`: FastH3's conditioning swap (`bench/fasth3_swap_arms.json`, its own
  rows file). Per scene, FastH3, base weights with FastH3's conditioning,
  FastH3 with the base's, and the base on FastH3's harness:
  `measure_clip_tone.py` and `measure_clip_temporal.py` on the clips, and
  `latent_path_distance.py` with FastH3 as the reference.
- `gates`: #35 (`bench/fasth3_gates_arms.json`, its own rows file). Per scene,
  FastH3 as the reference against fl2va plus FastH3's gates, FastH3 without
  them, today's FastH3 rerun and the swap batch's plain fl2va: tone,
  temporal and divergence, as `swap`.
- `dial`: #36 (`bench/fasth3_gate_dial_arms.json`). Per scene, FastH3 at gate
  scale 1 (the loader's bit-exact `overlay_all`) against scales 0.75, 0.5 and
  0: tone, temporal, resolution and divergence, as `gates`.
- `transplant`: FlashGen on block ranges (`bench/flashgen_transplant_arms.json`,
  fastdude's, its own rows file). Per scene, full FlashGen from the rerun and
  the 0-49, 0-33 and 34-49 arms: tone and temporal on the clips, and
  latent distance with full FlashGen as the reference.
- `late_switch`: open_experiments #34 (`bench/followup_late_switch_arms.json`,
  2026-09-27, its own rows file). Per scene, PDD8 from the rerun, the reverse
  switch with full FlashGen (`rev_h080`) and with FlashGen on blocks 34-49
  (`rev_late_h080`): tone, temporal and resolution on the clips, and latent
  distance with PDD8 as the reference. Its files are dated 2026-09-27.
- `grid`: the PDD8 finisher grid (`bench/pdd8_finisher_grid_arms.json`,
  2026-09-27). Per scene, PDD8 against its finishes from 0.8: full FlashGen,
  FlashGen on blocks 34-49 and the base. Arms from earlier rows files are found
  on the share by label.
- `fg46`: #46 in the same batch. Per scene, full FlashGen against FlashGen
  alone on blocks 34-49 (`flashgen_late`).

**Latent distance is divergence, not effect size.** Any change to the model
diverges the trajectory from step 0, so a final-latent distance says how far
two samples ended apart, not how large a change's effect is: FlashGen with its
adaln removed (about 0.1% of the modulation) lands 0.57-0.71 from full
FlashGen (fastdude, 2026-09-26). Read the tone and temporal columns for what
an arm does; the distance files are named `_divergence` for this reason.

Scenes whose prompts ask for frame-to-frame brightness change are left out of
temporal reads, per the manifest's `analysis_notes`. An arm that has not
landed yet is skipped and listed. Each group writes
`data/results/2026-09-26_followup_<group>_<tool>.json`.

    python bench/analyze_followup.py [--group looks|ladder|vsa ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from _paths import comfy_output  # noqa: E402

ROWS = REPO / "data" / "results" / "2026-09-26_followup.jsonl"
OUT = REPO / "data" / "results"
PY = sys.executable
#: from the manifest's analysis_notes: prompts that ask for brightness change
#: frame to frame, which the temporal measures would read as boiling or flicker.
TEMPORAL_EXCLUDE = ("kpop_dance_studio", "silent_film", "subway_chase_short")


def rows(path: Path = ROWS) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if not r.get("warmup") and not r.get("error"):
                out[r["label"]] = r
    return out


def newest(pattern: str, root: Path):
    hits = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def clip(label: str, out_root: Path):
    return newest(f"*_{label}_0*[0-9].mp4", out_root / "Video")


def latent(label: str, out_root: Path):
    return newest(f"*_video_{label}_0*_.latent", out_root / "latents")


def run(tool: str, args: list, json_out: Path):
    cmd = [PY, str(HERE / tool), *map(str, args), "--json", str(json_out)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout[-4000:])
    if res.returncode != 0:
        print(res.stderr[-2000:], file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", action="append",
                    choices=("looks", "ladder", "vsa", "spec", "reverse", "swap", "transplant", "late_switch", "grid", "fg46", "gates", "dial"))
    args = ap.parse_args()
    groups = args.group or ["looks", "ladder", "vsa", "spec", "reverse", "swap", "transplant", "late_switch", "grid", "fg46", "gates", "dial"]
    out_root = comfy_output()
    if not ROWS.exists():
        print(f"no rows yet: {ROWS.relative_to(REPO)} does not exist")
        return 0
    have = rows()
    missing = []

    if "looks" in groups:
        looks = sorted(l for l in have if l.startswith("look_"))
        clips = [c for c in (clip(l, out_root) for l in looks) if c]
        missing += [l for l in looks if not clip(l, out_root)]
        if clips:
            print("== looks: tone")
            run("measure_clip_tone.py", clips, OUT / "2026-09-26_followup_looks_tone.json")
            print("== looks: temporal")
            run("measure_clip_temporal.py", [*clips, "--stride", "4"],
                OUT / "2026-09-26_followup_looks_temporal.json")

    if "ladder" in groups:
        scenes = sorted({l.split("__")[0] for l in have if l.endswith(("__pdd4", "__pdd6", "__pdd8_merge"))})
        for scene in scenes:
            ref = latent(f"{scene}__pdd8", out_root)
            others = [latent(f"{scene}__{a}", out_root) for a in ("pdd4", "pdd6", "pdd8_merge")]
            others = [o for o in others if o]
            if not ref or not others:
                missing.append(f"{scene} ladder latents")
                continue
            print(f"== ladder: {scene}")
            run("latent_path_distance.py", [ref, *others], OUT / f"2026-09-26_followup_ladder_{scene}.json")

    if "vsa" in groups:
        scenes = sorted({l.split("__")[0] for l in have if l.endswith("__fasth3_novsa")})
        pairs = []
        for scene in scenes:
            on, off = clip(f"{scene}__fasth3", out_root), clip(f"{scene}__fasth3_novsa", out_root)
            if on and off and scene not in TEMPORAL_EXCLUDE:
                pairs += [on, off]
            else:
                missing.append(f"{scene} vsa pair")
        if pairs:
            print("== vsa: temporal")
            run("measure_clip_temporal.py", [*pairs, "--stride", "4"], OUT / "2026-09-26_followup_vsa_temporal.json")
            print("== vsa: block period")
            run("measure_block_period.py", pairs, OUT / "2026-09-26_followup_vsa_block.json")
            print("== vsa: tone")
            run("measure_clip_tone.py", pairs, OUT / "2026-09-26_followup_vsa_tone.json")

    if "spec" in groups:
        labs = [f"spec_{r}__{m}" for r in ("typical", "specific", "unusual")
                for m in ("flashgen", "pdd8", "fasth3")]
        clips = [clip(l, out_root) for l in labs]
        missing += [l for l, c in zip(labs, clips) if not c]
        clips = [c for c in clips if c]
        if clips:
            print("== spec: temporal")
            run("measure_clip_temporal.py", [*clips, "--stride", "4"], OUT / "2026-09-26_followup_spec_temporal.json")
            print("== spec: resolution")
            run("measure_clip_resolution.py", clips, OUT / "2026-09-26_followup_spec_resolution.json")
            print("== spec: tone")
            run("measure_clip_tone.py", clips, OUT / "2026-09-26_followup_spec_tone.json")

    if "reverse" in groups:
        ref = latent("subway_chase__pdd8", out_root)
        others = [latent(l, out_root) for l in
                  ("subway_chase__rev_h063", "subway_chase__rev_h080", "subway_chase__flashgen")]
        if ref and all(others[:2]):
            print("== reverse: against PDD8 (exact) on subway_chase")
            run("latent_path_distance.py", [ref, *[o for o in others if o]],
                OUT / "2026-09-26_followup_reverse_switch.json")
        else:
            missing.append("reverse switch latents")

    for group, rows_file, ref_arm, arms, day in (
            ("swap", "2026-09-26_fasth3_swap.jsonl", "fasth3",
             ("swap_fasth3adaln", "swap_baseadaln", "fl2va_contract"), "2026-09-26"),
            ("transplant", "2026-09-26_flashgen_transplant.jsonl", "flashgen",
             ("flashgen_blk0_49", "flashgen_blk0_33", "flashgen_blk34_49"), "2026-09-26"),
            ("late_switch", "2026-09-27_late_switch.jsonl", "pdd8",
             ("rev_h080", "rev_late_h080"), "2026-09-27"),
            ("grid", "2026-09-27_finisher_grid.jsonl", "pdd8",
             ("rev_h080", "rev_late_h080", "rev_base_h080"), "2026-09-27"),
            ("fg46", "2026-09-27_finisher_grid.jsonl", "flashgen",
             ("flashgen_late",), "2026-09-27"),
            # #35: fl2va plus FastH3's gates, FastH3 minus them, today's rerun
            # of FastH3, and the swap batch's plain fl2va on the same harness.
            ("gates", "2026-09-29_fasth3_gates.jsonl", "fasth3",
             ("fl2va_gates", "fasth3_nogates", "fasth3_rerun", "fl2va_contract"), "2026-09-29"),
            # #36: the gate dial, FastH3 through the overlay loader (alpha 1 is
            # bit-exact FastH3) with the gate row scales times 0.75, 0.5 and 0.
            ("dial", "2026-09-29_fasth3_gate_dial.jsonl", "overlay_all",
             ("g075", "g050", "g000"), "2026-09-29")):
        if group not in groups:
            continue
        landed = rows(OUT / rows_file)
        scenes = sorted({l.split("__")[0] for l in landed})
        for scene in scenes:
            labs = [f"{scene}__{a}" for a in (ref_arm, *arms)]
            clips = [clip(l, out_root) for l in labs]
            lats = [latent(l, out_root) for l in labs]
            missing += [l for l, c in zip(labs, clips) if not c]
            got = [c for c in clips if c]
            if len(got) < 2:
                continue
            print(f"== {group}: {scene} tone")
            run("measure_clip_tone.py", got, OUT / f"{day}_{group}_{scene}_tone.json")
            if scene not in TEMPORAL_EXCLUDE:
                print(f"== {group}: {scene} temporal")
                run("measure_clip_temporal.py", [*got, "--stride", "4"],
                    OUT / f"{day}_{group}_{scene}_temporal.json")
            if group in ("late_switch", "grid", "fg46", "gates", "dial"):
                print(f"== {group}: {scene} resolution")
                run("measure_clip_resolution.py", got, OUT / f"{day}_{group}_{scene}_resolution.json")
            if lats[0] and any(lats[1:]):
                print(f"== {group}: {scene} divergence from {ref_arm} (not an effect size)")
                run("latent_path_distance.py", [lats[0], *[x for x in lats[1:] if x]],
                    OUT / f"{day}_{group}_{scene}_divergence.json")

    if missing:
        print("not landed or not found:", ", ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
