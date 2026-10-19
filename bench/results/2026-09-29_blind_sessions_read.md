# Five blind sessions, read (2026-09-29)

The owner scored five blind batches (`docs/method/eval_comparison.md` section 3),
seed 730451892, one clip per arm, so each is a look and not a distribution.
Verdict records, from `bench/score_session.py` after the scores were in:

- `2026-09-29_finish_fasth3_covered_market_verdict.json` (t2v);
- `2026-09-29_finish_fasth3_r2v_market_verdict.json` (ref2va, one reference);
- `2026-09-29_gates_look_anchor_verdict.json` (#35's arms, look_anchor);
- `2026-09-29_overlay_refiner_look_anchor_verdict.json` (refiner off);
- `2026-09-29_fasth3_gate_dial_look_anchor_verdict.json` (the #36 gate dial,
  scored later the same day; its rows are
  `2026-09-29_blind_rows_fasth3_gate_dial_look_anchor.jsonl`).

The blinded rows the keys point at are `2026-09-29_blind_rows_<session>.jsonl`
(the render rows filtered of the failed first control and relabelled to the arm
name; the ref2va batch used its render rows directly). The gates batch has one
unscored single, joined with `--partial`.

## The finisher: PDD8 on the base to sigma 0.8, then a finish

Arms: FlashGen (the shipped finish), FastH3 at 10/3 (`s10`, handoff 0.769231)
and FastH3 at 12/3 (`s12`, handoff 0.8). All three arms of a scene share one
pass-1 latent.

1. **`s12` is indistinguishable from the FlashGen finish, on t2v and on
   ref2va.** ref2va pair, the owner: "same exact thing"; single: "looks great
   and sounds great, cant tell diff between this and clip 1". The t2v pair was
   scored `same`.
2. **`s10` is grainy, artifacty and compressed-looking on both**, audio fine,
   otherwise the same scene. t2v: "clip 1 is very grainy / artifacty /
   compressed looking. otherwise its the same though". ref2va: "grainy /
   artifacty / compressed looking but otherwise looks good".
3. **FastH3 handles the reference on ref2va at `s12`.** The transfer was
   untrained (FastH3 has never seen a reference token) and the owner saw no
   loss against the FlashGen finish, on one scene and one clip.
4. **So, on this evidence (one scene each, two tasks), the finish should run at
   PDD8's 12/3, not FastH3's own 10/3.** Why `s10` is grainy is not tested here. The two arms
   differ in shift, handoff sigma (0.769231 against 0.8) and the tail's rungs,
   and this session changed all three together. The audio was not the
   difference: it was fine on `s10`, as the audio sigma reasoning
   (`h3_config.STEP_SWITCH_FASTH3`) had it exact there. The grain is in the
   video.

## #35's arms, look_anchor

- FastH3 against fl2va plus FastH3's gates: both good, "just different scenes
  slightly", FastH3 "very slightly better with colors and shape proportions".
- FastH3 against FastH3 without gates: "audio is very quiet in #1 [no gates].
  fine in clip 2. differnet scenes - clip 1 is darker. so maybe slight edge to
  clip 2 [FastH3]".

By eye this agrees with the measures (`2026-09-29_fasth3_gates.md`): the gates
arm sits next to FastH3, and the no-gates arm is worse, darker, and here has
quiet audio, which the measures did not cover. One clip each; the audio
difference is one hearing and untested.

## Refiner off

Scored `same` ("same thing?"). On paper the change is near null
(`2026-09-29_refiner_requant.md`); by eye it is null on this scene.

## The gate dial, look_anchor (#36)

FastH3 through the overlay loader at gate scale 1, 0.75 and 0.5
(`2026-09-29_fasth3_gate_dial.md` has the measures).

- **0.75 against full: can't tell.** "slightly different scenes but both look
  good". No sign that a lower setting reads as less over-polished.
- **0.5 against full: full slightly ahead.** "spoken delivery of dialogue is
  slightly better in clip 1 [full] but both are also different scenes slightly
  so i cant say for certain". The 0.5 single: "slightly more muted colors and
  spoken delivery of dialogue is a little off but not bad".

By eye this agrees with the measures, which said the dial takes colour and
motion down with the detail: nothing gained at 0.75, and at 0.5 a little lost
(muted colour, dialogue delivery). One scene, one clip per arm, so it does not
show the dial is worse everywhere, only that this scene gave no reason to use
it.
