# Sapiens2 part segmentation in the masked lane: the class map every load stands on

**Sapiens2's part model gives a class to every pixel of the tracked person, and the lane builds four things on it: the
region of a part load (hair, face and neck, upper clothing, hands), the class map each capture grades a part against,
the mouth drawn on a head reference, and the keep-out of every person who is not a subject.** It runs in every preview
through `MiniMaxH3Sapiens2Loader` and `MiniMaxH3SubjectParts` (`../../sapiens2_parts.py`, since 2026-10-05), on the
folder `h3_config.SAPIENS2["segmentation"]`, with the class names and colours from
`../../vendor/meta_sapiens2/seg_classes29.json` (`sapiens2_parts.CLASS_NAMES`). Since 2026-10-19, on a face or head
load it decides face against hair only inside SAM 3.1's head track ("The head tube", below). Its classes are not the
things a person would choose: face and hair are safe, the rest need a second look. The overview of every perception
model is [`perception_models.md`](perception_models.md).

## How it is wired

- **The node.** `MiniMaxH3SubjectParts` shows the model the tracked subject alone (2026-10-06, `131cc6d6`), and hands
  out the part mask, a report, a `held` mask of what the subject holds and the class map (`classes`). The parts it can
  tick are `sapiens2_parts.PARTS`; its own ticks are `sapiens2_parts.PARTS_ON` (`h3_config.SUBJECT_PARTS`). How the
  Masked Source takes the mask (`replace` is `the wired parts`) is [`pieces.md`](pieces.md).
- **Shipped graphs.** `h3_video_to_video_masked_parts_song_pdd8_api.json` (2026-10-06, `9ae35f54`) and the head and
  upper body on the ref2va motion graph (`h3_config.MASKED_UPPER_PARTS`, 2026-10-07, `c93b5301`); the table is in
  [`overview.md`](overview.md).
- **In a job.** `../../workflows/masked_job_graph.py` puts the loader and the part node in every preview
  (`h3_config.MASKED_JOB_PARTS`: the node's own fill of a doubted frame off, so the capture can grade its own fill).
  A load names its parts with `bench/shot_to_load.py job --parts` and `--parts-on` (2026-10-13 and 2026-10-14).
- **What reads the class map.**
  - the capture's grade, which keeps a part on a frame only where it lies on the part's own classes
    (`bench/capture_masked_run.py::grade_holds`; [`state_signals.md`](state_signals.md), "Whether the face shows");
  - the owner map, which settles a pixel two tracks share by class (`bench/capture_masked_run.py::owner_map`; depth
    settles what it leaves contested: [`depth.md`](depth.md));
  - the strangers keep-out, every person the model finds on the whole frame less the subjects' grown tracks
    (`bench/strangers.py`; the grow is `h3_config.MASKED_STRANGERS`, with its provenance beside it);
  - the head reference's mouth, drawn where the pose model's lip lines lie on the class map's mouth
    (`bench/head_reference.py::MOUTH_CHECK`; records
    [`2026-10-15_class_mouth_gate.md`](../../bench/results/2026-10-15_class_mouth_gate.md) and
    [`2026-10-15_mouth_runs.md`](../../bench/results/2026-10-15_mouth_runs.md));
  - the refusal of the largest-piece rule on a part holding a face or hair (`bench/shot_to_load.py`;
    `bench/capture_masked_run.py::refuse_pieces_on_a_head`, 2026-10-16).
- **Matting.** The loader has a slot for the matting model (`sapiens2_parts.MATTE_ARCH`); `h3_config.SAPIENS2["matting"]`
  is `"none"` and that lane is closed ([`perception_trials.md`](perception_trials.md), "Sapiens2 matting").

## The head tube

**On a face or head load the region is SAM 3.1's head, and Sapiens2 only says which of it is face and which is hair.**
`bench/head_tube_parts.py::head_tube_for_capture` runs in the capture's files step whenever the subject's spec carries a
head track (the chain's default since 2026-10-19, `b7927da1`) or `--head-tube` is given, and `head_tube=0` turns it off
(`bench/capture_masked_run.py`, `use_head_tube`). It does three things:

1. bounds the head by the head track, itself clipped to the person track (`8b50cecf`; the tracker side is
   [`sam31.md`](sam31.md) and `subject_track.py::TRACK_HEAD`);
2. runs or reads the part model's class probabilities on one steady box for the shot and smooths them over three frames
   (`bench/head_tube_parts.py::smooth_seg_in_tube`);
3. labels each frame `facing`, `side_with_eye` or `back_of_head` from the pose model's face points
   (`bench/head_tube_parts.py::cross_check_face_points`, 2026-10-17, `a15463e9`); `back_of_head` is a proposal listed
   for a look, never an emptied frame by itself.

On a head-and-hair load every non-face pixel inside the tube is hair, so a turn the part model reads as clothing or
background cannot punch a hole in the hair (`252667b4`). Why it was built: on the kitchen 759-761 the part model
labelled the lead's face as clothing or background, and the head tube built on 2026-10-17 never ran SAM 3.1 in the
capture, so those labels alone decided the face; with the head track her face part was whole on 759-761, 772 and
784-785 ([`docs/state/decisions.md`](../state/decisions.md), 2026-10-19). Tested on one shot;
the first loads on other kinds of shots compare masks with and without it before rendering. What lies off the head is
dropped by the same head track (`bench/capture_masked_run.py::off_head`, `HEAD_TRACK_OFF_HEAD_MARGIN`).

## The part classes are not things a person would choose

Seen on 2026-10-11 by one session, on stills of two shots of one clip, in the class maps of one preview (the per-shot
table is in the untracked [`data/fun/poc/segments.json`](../../data/fun/poc/segments.json)): one class, apparel, held a
hat, a collar, gloves and a belt together; a whole dress was one class; bare forearms were labelled hands; on a shot of
a dancer many of the classes that appeared did so on a handful of frames only; and a share of the hair lay outside the
tracked person on every shot. The face and the hair were the two parts that looked right. So a choice of what to
replace offered from these classes is safe for a face and for hair, and needs a second look before it offers clothing,
a hat or hands.

## Limits

- **On a turn the face class fails.** The part model calls a turned face clothing or background (above), and a turned
  beard hair; a wrist in front of a chin hides the beard behind it, so a check that counts face labels alone is blind
  there (1306-1309 and 1408-1411, 2026-10-18:
  [`2026-10-18_pointmap_finding.md`](../../internal/gemini/2026-10-18_pointmap_finding.md)).
- **How small a face it can read is not measured.** Below some size the lip, teeth and tongue classes stop being found;
  a probe (one face at a ladder of sizes) is trial 5 in [`perception_trials.md`](perception_trials.md).
- **Enlarging the picture first changes nothing it reads**, on one run (2026-10-13, `48fac55f`), so the preview has no
  zoomed stage.
- **Hair past the track.** A share of the hair lies outside the tracked person; the strangers keep-out counts a
  subject's own hair past her track as hers (2026-10-14, `4e1366c9`).

## Trials and records

| date | what | record |
|---|---|---|
| 2026-10-05 | first run: the class order is upstream's; hair on every frame, where SAM 3's phrase missed four of 48 | [`2026-10-05_sapiens2_first_frame.md`](../../bench/results/2026-10-05_sapiens2_first_frame.md) |
| 2026-10-06 | the head and upper body as the region, on the ref2va motion graph and on the fast chain | [`2026-10-06_masked_v2v_body_window_arms.md`](../../bench/results/2026-10-06_masked_v2v_body_window_arms.md) |
| 2026-10-10 | a face part kept or emptied where the head is turned away | [`2026-10-10_face_part_kept_or_emptied_on_a_turn.md`](../../bench/results/2026-10-10_face_part_kept_or_emptied_on_a_turn.md) |
| 2026-10-15 | the reference's mouth from the class map, a frame at a time and over time | [`2026-10-15_class_mouth_gate.md`](../../bench/results/2026-10-15_class_mouth_gate.md), [`2026-10-15_mouth_runs.md`](../../bench/results/2026-10-15_mouth_runs.md) |
| 2026-10-18 | the chin squares on the kitchen 751-785 kept as the source's, dropped by the part model | [`2026-10-18_kitchen_two_pass_validation.md`](../../bench/results/2026-10-18_kitchen_two_pass_validation.md) |
