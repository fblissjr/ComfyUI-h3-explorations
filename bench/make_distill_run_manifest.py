#!/usr/bin/env python3
"""Write the manifest for the 2026-09-26 fresh distill run: many bank scenes, four models, one seed.

The owner, 2026-09-26: FlashGen's prompt adherence "needs badly to be tested. I'd
use a diverse set of scenes from prompt bank. Varying lengths and goals based on
the json metadata in prompt bank". Seed-matched clips are different scenes per
distill (`docs/distills/which_distill.md`), so the comparison is across many scenes,
judged against each scene's beat checklist (`bench/adherence_checklists.json`),
not across seeds of one scene.

Arms are grouped by model, so the card loads each model state once. Every arm
patches the scene's prompt by bank id and its `MiniMaxH3Resolution.length` to
the scene's declared frames, so no arm renders a prompt off its length.

    python bench/make_distill_run_manifest.py [--graphs graphs.json] --out bench/distill_run_arms.json

`--graphs` maps each role to a graph path: the saved-latent twins, once built.
Without it, the shipped graphs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BANK = REPO / "prompt_bank" / "bank.json"

#: The scenes, chosen by bank metadata for spread in length, shot count,
#: speaker count and what each prompt tests (see the manifest's `why`).
SCENES = ["t2va_noodle_bar", "t2va_post_office", "t2va_rooftop_pov", "t2va_box_office",
          "t2va_radio_drama", "t2va_desert_crew", "t2va_slapstick_moving_piano",
          "t2va_kpop_dance_studio", "t2va_courtroom_verdict", "t2va_samurai_bamboo_duel",
          "t2va_silent_film"]
#: Model roles, in run order.
MODELS = ["base", "pdd8", "flashgen", "fasth3"]
#: The saved-latent twins (the VAE session, 0.153.0): every arm's latent is
#: kept, so routing experiments can reuse these renders with no card time.
DEFAULT_GRAPHS = {
    "base": "workflows/distill_experiments/h3_text_to_video_savelat_api.json",
    "pdd8": "workflows/distill_experiments/h3_text_to_video_pdd_savelat_api.json",
    "flashgen": "workflows/distill_experiments/h3_text_to_video_flashgen_savelat_api.json",
    "fasth3": "workflows/distill_experiments/h3_probe_t2v_fasth3_8step_contract_savelat_api.json",
    "flashgen_dense": "workflows/distill_experiments/h3_probe_t2v_flashgen_r64_4step_branch_dense_savelat_api.json",
    "pdd8_refine": "workflows/distill_experiments/h3_probe_t2v_pdd8_audio_refine_savelat_api.json",
    "flashgen_refine": "workflows/distill_experiments/h3_probe_t2v_flashgen_4step_audio_refine_savelat_api.json",
    "route3": "workflows/distill_experiments/h3_probe_t2v_step_switch_flashgen_pdd8_savelat_api.json",
    "flashgen_s08": "workflows/distill_experiments/h3_text_to_video_flashgen_savelat_api.json",
    "flashgen_s12": "workflows/distill_experiments/h3_text_to_video_flashgen_savelat_api.json",
    "flashgen_noadaln": "workflows/distill_experiments/h3_text_to_video_flashgen_savelat_api.json",
}
#: A short bank scene NOT in the run, for one warmup per model state (the VAE
#: session: the first render of each state pays pinning, staging and LoRA
#: patching). On a run scene at the held seed it would make that scene's real
#: arm a cache hit.
WARMUP_SCENE = "t2va_swimming_lesson"
#: Extras, each on a subset. **Reasoned**:
#: - FlashGen dense on the motion scenes, where Sol's sparsity is most likely
#:   to cost adherence;
#: - refine on the two dialogue-and-music scenes, the pairs route 5 needs;
#: - route 3 where motion and detail both matter.
EXTRAS = {
    "flashgen_dense": ["t2va_slapstick_moving_piano", "t2va_kpop_dance_studio",
                       "t2va_samurai_bamboo_duel", "t2va_rooftop_pov"],
    "pdd8_refine": ["t2va_courtroom_verdict", "t2va_radio_drama"],
    "flashgen_refine": ["t2va_courtroom_verdict", "t2va_radio_drama"],
    "route3": ["t2va_slapstick_moving_piano", "t2va_kpop_dance_studio",
               "t2va_courtroom_verdict", "t2va_samurai_bamboo_duel"],
    "flashgen_s08": ["t2va_courtroom_verdict", "t2va_kpop_dance_studio", "t2va_samurai_bamboo_duel"],
    "flashgen_s12": ["t2va_courtroom_verdict", "t2va_kpop_dance_studio", "t2va_samurai_bamboo_duel"],
    "flashgen_noadaln": ["t2va_courtroom_verdict", "t2va_kpop_dance_studio", "t2va_samurai_bamboo_duel"],
}
#: FlashGen variants through `MiniMaxH3LoRABranch`'s granular inputs (0.152.0):
#: does its grade and adherence move with strength, and is the timestep
#: modulation (adaln) what carries the grade? **Reasoned** choices: +-20%
#: strength brackets the publisher's 1.0 without leaving the regime; "no adaln"
#: isolates the one module kind that shapes the per-step dynamics.
VARIANT_PATCHES = {
    "flashgen_s08": ["MiniMaxH3LoRABranch.strength=0.8"],
    "flashgen_s12": ["MiniMaxH3LoRABranch.strength=1.2"],
    "flashgen_noadaln": ['MiniMaxH3LoRABranch.modules="no adaln"'],
}
SEED = 730451892


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    graphs = dict(DEFAULT_GRAPHS)
    if args.graphs:
        graphs.update(json.loads(args.graphs.read_text()))
    frames = {p["id"]: p["frames"] for p in json.loads(BANK.read_text())["prompts"]}
    missing = [s for s in SCENES if s not in frames]
    if missing:
        raise SystemExit(f"not in the bank: {missing}")

    arms, patches, skipped = {}, [], []
    def add(role, scene):
        if graphs.get(role) is None:
            skipped.append(f"{role}:{scene}")
            return
        label = f"{scene.removeprefix('t2va_')}__{role}"
        arms[label] = graphs[role]
        patches.append(f"{label}:MiniMaxH3Conditioning.prompt=@bank:{scene}")
        patches.append(f"{label}:MiniMaxH3Resolution.length={frames[scene]}")
        for extra_patch in VARIANT_PATCHES.get(role, []):
            patches.append(f"{label}:{extra_patch}")
    # The frozen-row probe first, on the refine twin's own default prompt, so it
    # leaves no cache hit for any run arm (the VAE session's advice).
    arms["frozen_row_probe"] = graphs["pdd8_refine"]
    for role in MODELS:
        warm = f"warmup_{role}"
        arms[warm] = graphs[role]
        patches.append(f"{warm}:MiniMaxH3Conditioning.prompt=@bank:{WARMUP_SCENE}")
        patches.append(f"{warm}:MiniMaxH3Resolution.length={frames[WARMUP_SCENE]}")
        for scene in SCENES:
            add(role, scene)
        if role in ("base", "flashgen"):
            # FlashGen at its trained length, against the base, on the five-second subway twin
            add(role, "t2va_subway_chase_short")
        for extra, scenes in EXTRAS.items():
            if extra.startswith(role + "_"):
                for scene in scenes:
                    add(extra, scene)
    arms["warmup_route3"] = graphs["route3"]
    patches.append(f"warmup_route3:MiniMaxH3Conditioning.prompt=@bank:{WARMUP_SCENE}")
    patches.append(f"warmup_route3:MiniMaxH3Resolution.length={frames[WARMUP_SCENE]}")
    for scene in EXTRAS["route3"]:
        add("route3", scene)

    manifest = {
        "what": ("The fresh distill run, 2026-09-26: base, PDD8, FlashGen (rank 64 at the "
                 "call) and FastH3 (contract) on eleven bank scenes at one seed, each at its "
                 "declared length, plus FlashGen dense, refine pairs, the step-switch arm, and "
                 "the five-second subway twin. Built by bench/make_distill_run_manifest.py."),
        "why": ("The owner, 2026-09-26: test FlashGen's prompt adherence across diverse bank "
                "scenes, and what each distill is good and bad at (docs/distills/which_distill.md), fresh, "
                "with everything measured. Seed-matched clips are different scenes per "
                "distill, so the unit is the scene, judged blind against its beat checklist."),
        "scenes": {s: frames[s] for s in SCENES + ["t2va_subway_chase_short"]},
        "arms": arms,
        "patches": patches,
        "seeds": {"first": SEED, "note": "held across every arm"},
        "skipped_until_built": skipped,
        "order": ("frozen_row_probe first; then per model state a warmup_<role> on "
                  f"{WARMUP_SCENE} and its arms; drop warmup_* rows at analysis. The refine "
                  "arms reuse their model's pass 1 from the node cache, so their rows time "
                  "the refine pass alone."),
        "after": ("bench/record_render_substrate.py before the server stops (/history is "
                  "volatile); then measure_clip_tone, measure_clip_delta --json, "
                  "measure_clip_resolution, measure_clip_loudness, "
                  "measure_dialogue_transcription, telemetry_report.py --json"),
        "run": (f"run_graph_arms.py --manifest {args.out} --runs 1 --seed {SEED} --hold-seed "
                f"--no-alternate --out data/results/2026-09-26_distill_run.jsonl"),
        "predictions": "none registered",
    }
    args.out.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(arms)} arms; skipped until built: {len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
