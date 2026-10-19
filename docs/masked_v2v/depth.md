# Depth in the masked lane: MoGe-2 says who is in front where two masks overlap

**Depth answers one question in this lane: of two masks that touch or cross on a frame, which is nearer the camera.**
It is read by Meta's HyDen MoGe-2 metric point model, through the loader vendored in `../../vendor/meta_moge2/`
(`vendor/meta_moge2/hyden_moge2.py::load_hyden_moge2`, since 2026-10-17), by `bench/measure_moge2_depth.py`, only on a load's overlap
frames, and only when the job turns it on: `bench/shot_to_load.py job --depth` (off by default, since 2026-10-18). The
capture then settles shared pixels by it. On the two crossings it was measured on it settled nearly every contested
pixel. It cannot separate faces side by side at the same distance, or a hand on a face. How to run it is
[`running_a_job.md`](running_a_job.md), "Depth: who is in front"; the overview of every perception model is
[`perception_models.md`](perception_models.md).

## How it is wired

- **The measure.** `bench/measure_moge2_depth.py` runs `HyDenMoGe` (`../../vendor/meta_moge2/hyden_moge2.py`) on raw
  `[0, 1]` frames fitted to the canvas and writes `depth.npz` (metres, a valid mask, per-frame intrinsics), a
  `record.json` and a sheet to look at (`4725ff5c`). Its check is `bench/check_measure_moge2_depth.py`.
- **In a job.** `bench/shot_to_load.py::overlap_frames_of` finds the frames where two subject tracks, or a keep-out and
  the replaced subject's track, overlap; `bench/shot_to_load.py::make_depth` measures only those into
  `<capture>/depth.npz`, and none when there are none (`3980ed5c`). `bench/load_chain.py` then passes `--depth` and every
  preview subject's class map to the capture.
- **In the capture.** `bench/capture_masked_run.py::read_depth_npz` refuses a file of another canvas or missing a frame;
  `bench/capture_masked_run.py::owner_map` takes each subject's median depth on its eroded interior and gives a shared
  pixel to the one strictly in front, by `DEPTH_ORDER_MIN_M` and `DEPTH_ORDER_REL`, with `DEPTH_EDGE_TOL_M` for an arm
  reaching forward and `DEPTH_CORE_ERODE_PX`, `DEPTH_CORE_LEAST_PX` for the interior (`1c213f41`; each constant has its
  provenance beside it). A back subject's part is clipped where a closer one owns the pixel.
- **At assembly.** `bench/assemble_delivery.py::restore_weight` routes `restore=<other>:whole` through the owners on a
  frame depth settled, so a person walking behind is not punched out of the one in front.
- **Why vendored.** The loader follows `coderef/metadepth` closely, which is under Meta's own licence, so it lives under
  `vendor/` with that licence and a README naming its source ([`docs/state/decisions.md`](../state/decisions.md),
  2026-10-17).

## What it measured

From [`2026-10-18_moge2_depth_on_crossings.md`](../../bench/results/2026-10-18_moge2_depth_on_crossings.md), on CPU
while the card held the server; no render:

- on 16 frames of four spans, the people and hands in the visible front-to-back order;
- the hall crossing (1409-1417): contested shared pixels from 99.33% to 0.0016%, and 53,938 px of the back subject's
  part clipped off the front subject;
- the kitchen crossing (755-780): contested shared pixels from 91.02% to 0%;
- cost: paid only on overlap frames (9 on the hall crossing, 18 over all of 604-1009).

What a render showed: on `lead_fp_0751` the mom's keep-out with depth carved whole cells out of the lead's region,
square blocks round her head (2026-10-18; not kept).

## Depth and geometry

The depth and geometry models in the store, and what each is to the lane:

| model | in the store as | state |
|---|---|---|
| HyDen MoGe-2 metric point | `facebook_hyden-mogev2-metric-point` | **wired** behind `--depth`, as above |
| HyDen Depth Anything V2 relative depth | `facebook_hyden-da2-relative-depth` | a candidate, loaded by nothing: a second reading on the same frames, worth a run only where two readings disagreeing would be looked at by eye ([`perception_trials.md`](perception_trials.md)) |
| Depth Anything V2 | the ComfyUI checkout's `models/depthanything/` | on disk, loaded by no code of this pack; the base of the row above |
| Sapiens2 pointmap 1b and 5b | `facebook_sapiens2-pointmap-1b`, `-5b` | **tried and dropped** on 2026-10-18: the 1b gives a flat sheet, the 5b a soft edge at 18 times MoGe-2's time ([`perception_trials.md`](perception_trials.md), "Sapiens2 pointmap") |
| HyDen MoGe-2 surface normal, the Sapiens2 normal models | not on disk | no use in this lane anybody has named |

Depth may be shown to the model only as a layer drawn into the load's one reference video, never as a number or
through a control adapter ([`docs/state/decisions.md`](../state/decisions.md), 2026-10-11). No depth layer has been
drawn; depth today plans the masks and nothing else.

## The camera for the body mesh

Meta's own SAM 3D Body demo estimates the field of view with MoGe-2 before it poses a body; this pack's body node runs on
the fallback ([`sam3d_body.md`](sam3d_body.md), the differences table). MoGe-2's single-frame field of view held within
1.5 to 3.2 degrees on medium shots and swung by 47 to 75 degrees on a tight close-up and where a torso filled the frame
(the record, section 5), so it cannot be handed to the mesh frame by frame on close-ups. Within one frame the swing does
not touch the order, since the shift it recovers is one value for the whole frame. Whether a shot-level field of view
puts the mesh closer on the tracked person is trial 3 in [`perception_trials.md`](perception_trials.md).

## Limits

- Faces side by side at about the same distance, and a hand on a face: one depth map cannot see a beard behind a wrist.
- Only on frames where masks overlap; a load with none runs no pass.
- Not measured on the card: GPU wall time and memory.
