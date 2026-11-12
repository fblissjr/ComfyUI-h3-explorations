# A drawn on/off signal as a full-length guide video: the first render (2026-11-12)

Date: 2026-11-12

A throwaway, run to decide one thing: on the shipped text-to-video graph, does a drawn signal given as a
full-length guide video time an event in the picture? It does not. The picture comes back as the signal
itself. The sound, which the guide did not hold, lands on the signal's windows.

The question came from two clips shared with the owner, in which a hop and a ball's bounce fall on the pulses
of a drawn signal. How those clips were made is not established here. The owner's account is that their author
used a guide node.

## 1. What was rendered

- **Graph.** `workflows/h3_text_to_video_pika_control_vector_api.json` with its control-vector node removed.
  The two guide arms add `MiniMaxH3GuideLatent` (`guide_latent.py`) between the conditioning node and the
  preflight node, with the guide video anchored at frame 0 and nothing else changed.
- **Prompt.** `prompt_bank/t2va_offpath_tortoise.txt` with two clauses changed, read in full before queueing:
  the tortoise also springs off the road three times, and the soundscape adds a thud on each landing. The
  full text is in the run's record.
- **Length and seed.** The bank's frame count for that prompt and the graph's seed, as in the day's other
  tortoise renders.
- **Arms.**
  - `plain`: the prompt, no guide.
  - `bars_a`: the same, with a guide video that is black except for one white bar, lit on three windows. The
    bar's height is the signal: full at the start of a window, falling to nothing by its end.
  - `bars_b`: the same, with the three windows moved later so that they do not overlap `bars_a`'s.
- **Server.** ComfyUI as it was running; every arm returned `success`.
- **Repo commit:** `f56d55e8`.

The script, the arm graphs and the record (`make_arms.py`, `arm_*.json`, `exp1_record.json`) are in the
untracked `data/guide_signal_2026-11-12/`. The guide videos are under `$H3_INPUT_DIR/guide_signal/`. The clips
are under `$H3_OUTPUT_DIR/Video/guide_signal_2026-11-12/exp1/`.

## 2. What came back

Measured by `measure_exp1.py` in the run folder, which wrote `exp1_measure.json`. Nothing here was listened to
or watched in motion.

- **The picture of both guide arms is the guide.** A black frame with the white bar, on exactly the frames
  the guide's bar is lit, and no street, man or tortoise at any point.
- **The sound of both guide arms follows the windows.** The level inside the windows is many times the level
  outside them, in both arms. In `bars_a` the level peaks a couple of frames after each window opens. In
  `bars_b` the peaks sit on that arm's later windows, not on `bars_a`'s.
- **`plain` does not show a hop.** In every third frame of the clip the tortoise is on the road. The hop
  clause alone did not produce one on this seed.
- **The guide arms cost more.** The preflight line in the server log shows the guide adding about as many
  rows again as the clip has, and each guide arm took several times as long as `plain` (`exp1_record.json`).

## 3. What to listen for and look at

- **What the sound in the windows is.** A level cannot say whether it is a thud, a tone or noise. The two
  `-audio` clips have it.
- **Whether `plain` hops between the frames sampled.** Playback settles it.

## 4. Readings, not findings

- **A full-length picture guide is copied.** That is what a guide is for on this model (a first or last
  frame the output must show), and at full length it leaves the picture nothing to generate. This route is
  closed on the stock graph. It says nothing about a guide LoRA trained to read a guide as control, about a
  short guide, or about a guide dropped partway through sampling.
- **The stream that was left free timed itself to the stream that was held.** The sound was not guided, and
  it put its events on the picture's bar, including after the windows were moved. That is the same coupling
  the shared clips show, in the other direction, and it is the reason to expect a sound-only guide to time
  the picture.

## 5. What this cannot see

- **More than one prompt, seed and bar design.**
- **Whether a held sound moves the picture.** Only the reverse was rendered.
- **The reference route.** A signal video given as `<Video 1>`, with the prompt saying what it means, sits
  ahead of the output on the time axis and claims no part of the picture. It was not tried.
