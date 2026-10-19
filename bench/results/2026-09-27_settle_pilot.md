# #43 pilot: does a frame settle later where the picture changes more? (2026-09-27)

`docs/state/open_experiments.md` #43, the premise under the owner's motion-adaptive
schedule. CPU only, on two renders that already existed: `subway_chase` at seed
730451892 and 345 frames, on PDD8 (8 steps) and on the base on Euler at 32
steps (PDD's teacher path, sharing PDD8's starting noise).

**Method.** `bench/analyze_settle_vs_delta.py` joins two per-render records:
- the x0 records `2026-09-26_x0_steps_subway_pdd8.json` and
  `2026-09-27_x0_steps_subway_base_euler32.json` (`bench/x0_step_frames.py`):
  per latent frame, each step's `to_final`;
- the delta record `2026-09-27_settle_pilot_delta.json`
  (`bench/measure_clip_delta.py --json` on each render's own clip): the
  per-frame change and the cut times.

Per latent frame it computes:
- its mean delta;
- the fractional step at which `to_final` first crosses under a threshold τ,
  as a fraction of the run;
- a threshold-free "late mass": mean `to_final` over the run, over its first
  value.

The rank correlation uses average ranks: PDD8's 8 steps tie most frames, and
the first version of the script, without tie handling, reported a strong
negative correlation that was mostly an artifact of frame order. Cut frames are
reported apart. Full per-frame output is in `2026-09-27_settle_pilot_pdd8.json`
and `2026-09-27_settle_pilot_base_euler32.json`.

| render | τ 0.5 | τ 0.3 | τ 0.2 | τ 0.1 | late mass |
|---|---|---|---|---|---|
| base, Euler 32 | +0.556 | +0.602 | +0.503 | +0.545 | +0.378 |
| PDD8 | +0.507 | -0.041 | -0.142 | -0.063 | -0.267 |

(Spearman's rank correlation between a frame's delta and how late it settles,
over the non-cut frames.)

## Reading

- **On the base, frames that change more settle later, at every threshold.**
  The teacher keeps refining its high-change frames into the late steps.
- **On PDD8 the relation is gone after the first threshold.** Moving frames
  are resolved on about the same schedule as still ones, or slightly earlier.
- **Inference, not measured:** PDD8's fused tail does not give high-change
  frames the late refinement its teacher gives them. That fits the owner's
  read that PDD8 "sucks at motion" and does well at low delta
  (`docs/state/next_steps.md`), and it points the same way as the
  motion-adaptive idea. A finer tail on moving scenes would restore late
  steps where the teacher uses them.
- **#43's decision rule is not met as written.** It expected high-delta frames
  to settle later on PDD8 and not on the base. The data shows the reverse
  pattern: the late settling is the teacher's, and PDD8 lacks it. That
  supports building a motion-aware partition only if the missing late
  refinement is what costs PDD8 on motion. This pilot cannot show that.
- **Cuts (the "pixel wipe"):** too few to read. Subway has two cuts, so two or
  three cut frames per render. On the base they settle with or slightly after
  the rest, not earlier.

## Limits

- One scene, one seed, two renders.
- Delta is change, not motion.
- Each render's delta comes from its own clip, and PDD8 and the base are
  different takes of the scene.
- A correlation is not a cause.

## What would decide it

The scene pass #43 names: high-delta scenes (slapstick_moving_piano,
samurai_bamboo_duel) and low-delta scenes (courtroom_verdict, radio_drama) on
both x0 twins. The base's positive relation should hold on the high-delta
scenes. If PDD8 lacks it there too, the next test is causal: PDD at a finer
tail (PDD16, `h3_probe_t2v_pdd16_savelat`, or a denser `envelope_partition`)
on one high-delta scene. If moving frames then regain late settling and
motion detail rises, the adaptive partition has its target.
