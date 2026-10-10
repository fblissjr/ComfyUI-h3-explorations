#!/usr/bin/env python3
"""Write the manifest for the 2026-09-26 follow-up batch: the distill run's
remaining arms on the 0.154.0 code, and the tests its predictions need.

The distill run (`make_distill_run_manifest.py`) was stopped after its base
arms: every later arm ran PDD on the merged path, and the owner's 0.154.0
change applies every LoRA on int8 at the call ("if you need to stop the renders
because they dont make sense anymore, please do so, and then restart comfy from
a place where you can actually test all the code you guys wrote"). Its base
arms (`er_sde`, no LoRA) are code-independent and are reused as they are.

The batch is ordered by priority, so a partial night still answers the main
questions. Each group's `why` names the predictions it tests
(`bench/results/2026-09-26_distill_run_predictions.md`). Every arm saves its
latent (a `_savelat` or `_x0` twin), so path distances need no decode.

    python bench/make_followup_manifest.py --out bench/followup_arms.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BANK = REPO / "prompt_bank" / "bank.json"
W = "workflows/distill_experiments/"

#: Graphs by role; each saves its latent.
G = {
    "base": W + "h3_text_to_video_savelat_api.json",
    "base_euler16": W + "h3_probe_t2v_base_euler16_savelat_api.json",
    "base_euler32": W + "h3_probe_t2v_base_euler32_savelat_api.json",
    "base_euler32_x0": W + "h3_probe_t2v_base_euler32_x0_api.json",
    "pdd8": W + "h3_text_to_video_pdd_savelat_api.json",
    "pdd8_x0": W + "h3_text_to_video_pdd_x0_api.json",
    "pdd8_merge": W + "h3_text_to_video_pdd_savelat_api.json",
    "pdd4": W + "h3_text_to_video_pdd_4step_savelat_api.json",
    "pdd6": W + "h3_text_to_video_pdd_manual_sigmas_savelat_api.json",
    "pdd16": W + "h3_probe_t2v_pdd16_savelat_api.json",
    "pdd32": W + "h3_probe_t2v_pdd32_savelat_api.json",
    "flashgen": W + "h3_text_to_video_flashgen_savelat_api.json",
    "flashgen_dense": W + "h3_probe_t2v_flashgen_r64_4step_branch_dense_savelat_api.json",
    "flashgen_s08": W + "h3_text_to_video_flashgen_savelat_api.json",
    "flashgen_s12": W + "h3_text_to_video_flashgen_savelat_api.json",
    "flashgen_noadaln": W + "h3_text_to_video_flashgen_savelat_api.json",
    "fasth3": W + "h3_probe_t2v_fasth3_8step_contract_savelat_api.json",
    "fasth3_novsa": W + "h3_probe_t2v_fasth3_8step_contract_novsa_savelat_api.json",
    "route3": W + "h3_probe_t2v_step_switch_flashgen_pdd8_savelat_api.json",
    "pdd8_refine": W + "h3_probe_t2v_pdd8_audio_refine_savelat_api.json",
    "flashgen_refine": W + "h3_probe_t2v_flashgen_4step_audio_refine_savelat_api.json",
}
#: Per-role patches beyond prompt and length.
EXTRA = {
    "pdd8_merge": ['MiniMaxH3PDDLoRA.backbone_apply="merge"'],
    "flashgen_s08": ["MiniMaxH3LoRABranch.strength=0.8"],
    "flashgen_s12": ["MiniMaxH3LoRABranch.strength=1.2"],
    "flashgen_noadaln": ['MiniMaxH3LoRABranch.modules="no adaln"'],
}

#: The distill run's scenes (its base arms rendered 2026-09-26).
RUN_SCENES = ["t2va_noodle_bar", "t2va_post_office", "t2va_rooftop_pov", "t2va_box_office",
              "t2va_radio_drama", "t2va_desert_crew", "t2va_slapstick_moving_piano",
              "t2va_kpop_dance_studio", "t2va_courtroom_verdict", "t2va_samurai_bamboo_duel",
              "t2va_silent_film"]
#: The motion scenes the PDD ladder runs on, set by the 2026-09-26
#: prompt-fitness audit: each fixes its head count, so a clone is countable.
#: kpop is out (its dancers were written against a mirror wall). subway_chase
#: is in: "only two people in the whole station", and the owner saw PDD8 clone
#: a man there at 1 s at this seed (`2026-09-26_subway_v2_s1.md`).
MOTION = ["t2va_subway_chase", "t2va_slapstick_moving_piano", "t2va_samurai_bamboo_duel"]
#: One still scene, for the merged-vs-exact and base-Euler pairs.
STILL = "t2va_radio_drama"
#: The scene that carries the per-step x0 capture: the one where PDD8's clone
#: was seen, so the capture can show the step it enters at.
X0_SCENE = "t2va_subway_chase"
#: The look family, cut by the owner, 2026-09-26, to "a black and white / rich
#: blacks / shadows / lighting in different distills. i dont need the base
#: versions": the low-key anchor and noir first, then neon and anime.
LOOKS_CORE = ["t2va_look_anchor", "t2va_look_noir"]
LOOKS_REST = ["t2va_look_neon", "t2va_look_anime"]
LOOK_MODELS = ["pdd8", "flashgen", "fasth3"]
WARMUP_SCENE = "t2va_swimming_lesson"
SEED = 730451892


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    frames = {p["id"]: p["frames"] for p in json.loads(BANK.read_text())["prompts"]}
    for s in RUN_SCENES + MOTION + LOOKS_CORE + LOOKS_REST + [WARMUP_SCENE, "t2va_subway_chase_short"]:
        if s not in frames:
            raise SystemExit(f"not in the bank: {s}")

    arms, patches, groups = {}, [], []
    warmed = set()

    def add(role, scene, label=None, graph=None):
        label = label or f"{scene.removeprefix('t2va_')}__{role}"
        if label in arms:
            raise SystemExit(f"duplicate arm {label}")
        arms[label] = graph or G[role]
        patches.append(f"{label}:MiniMaxH3Conditioning.prompt=@bank:{scene}")
        patches.append(f"{label}:MiniMaxH3Resolution.length={frames[scene]}")
        for p in EXTRA.get(role, []):
            patches.append(f"{label}:{p}")
        return label

    def warm(role):
        # One throwaway render per model state, on a scene no arm uses at the
        # held seed, so the next real arm is neither a cold load nor a cache hit.
        if role in warmed:
            return
        warmed.add(role)
        add(role, WARMUP_SCENE, label=f"warmup_{role}")

    def group(name, why, fn):
        start = len(arms)
        fn()
        groups.append({"group": name, "why": why, "arms": list(arms)[start:]})

    def g_core():
        for role in ("pdd8", "flashgen", "fasth3"):
            warm(role)
            for s in RUN_SCENES:
                add(role, s)
            if role == "flashgen":
                add(role, "t2va_subway_chase_short")

    def g_ladder():
        # The owner's 4-hour cap, 2026-09-26: no base renders, and pdd16 and
        # pdd32 cut (4 against 6 against 8 decides the tail question).
        # PDD8 exact on the clone scene carries the x0 capture; merged beside it
        # asks whether the exact branch removes the clone the owner saw.
        add("pdd8_x0", X0_SCENE, label=f"{X0_SCENE.removeprefix('t2va_')}__pdd8")
        warm("pdd8_merge")
        for s in [X0_SCENE, STILL]:
            add("pdd8_merge", s)
        for role in ("pdd4", "pdd6"):
            for s in MOTION:
                add(role, s)
        warm("fasth3_novsa")
        # Not kpop, whose strobing LED bars confound the temporal measures.
        for s in ["t2va_slapstick_moving_piano", "t2va_samurai_bamboo_duel"]:
            add("fasth3_novsa", s)

    def g_looks(looks):
        def fn():
            for role in LOOK_MODELS:
                warm(role)
                for s in looks:
                    add(role, s)
        return fn

    def g_extras():
        for s in ["t2va_kpop_dance_studio", "t2va_samurai_bamboo_duel"]:
            add("flashgen_dense", s)
        for role in ("flashgen_s08", "flashgen_s12", "flashgen_noadaln"):
            for s in ["t2va_courtroom_verdict", "t2va_kpop_dance_studio"]:
                add(role, s)

    # The "smoke" group (a lightx2v turbo file through the branch) left with the
    # turbo graphs on 2026-09-26; its row in 2026-09-26_followup.jsonl and
    # bench/followup_arms.json stay as the record of what ran.
    group("looks_core", "the owner's priority: black and white, rich blacks, shadows and "
          "lighting across the distills (F7, O1)", g_looks(LOOKS_CORE))
    group("core", "the distill run's remaining arms on the 0.154.0 code: FlashGen "
          "adherence (P1, P8), PDD8 clones (P4), grade among the distills (P5, P6). "
          "The run's base arms are reused", g_core)
    group("ladder", "merged vs exact PDD8 (F4, vaedude P5), PDD at 4 and 6 steps against "
          "the core's 8 (F3, vaedude P1), FastH3 with VSA off (F6, vaedude P7). The x0 "
          "capture (F5, vaedude P3) rides on the core's PDD8 arm on X0_SCENE", g_ladder)
    group("extras", "FlashGen dense (P2), strength and no-adaln (P3)", g_extras)
    group("looks_rest", "neon and anime, if time remains", g_looks(LOOKS_REST))

    manifest = {
        "what": ("The 2026-09-26 follow-up batch, on the 0.154.0 code (every LoRA on int8 at "
                 "the call). Built by bench/make_followup_manifest.py."),
        "why": ("The distill run stopped after its base arms, because every later arm ran "
                "PDD merged; this finishes it on the exact branch and adds the tests the "
                "registered predictions need."),
        "predictions": "bench/results/2026-09-26_distill_run_predictions.md",
        "analysis_notes": ("exclude kpop (strobing LED bars), silent_film (exposure "
                           "flicker) and subway_chase_short (flickering tubes) from "
                           "measure_clip_temporal, or mask the flicker: each prompt asks "
                           "for frame-to-frame brightness change"),
        "deferred": ("every base render (base-Euler 16 and 32, the base on the looks), "
                     "pdd16 and pdd32, refine, step-switch: the owner's 4-hour cap, "
                     "2026-09-26. F1, F2 and vaedude P2 and P4 wait for them"),
        "reuses": ("the distill run's base arms, data/results/2026-09-26_distill_run.jsonl, "
                   "rows noodle_bar__base through subway_chase_short__base"),
        "groups": groups,
        "arms": arms,
        "patches": patches,
        "seeds": {"first": SEED, "note": "held across every arm"},
        "run": (f"run_graph_arms.py --manifest {args.out} --runs 1 --seed {SEED} --hold-seed "
                f"--no-alternate --out data/results/2026-09-26_followup.jsonl"),
    }
    args.out.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(arms)} arms in {len(groups)} groups: "
          + ", ".join(f"{g['group']} {len(g['arms'])}" for g in groups))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
