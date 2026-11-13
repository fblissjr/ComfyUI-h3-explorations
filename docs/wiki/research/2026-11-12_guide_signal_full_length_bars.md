# A drawn on/off signal as a full-length guide video: the first renders (2026-11-12)

Date: 2026-11-12

A throwaway, run to decide one thing: on the shipped text-to-video graph, does a drawn signal given as a
full-length guide video time an event in the picture? It does not. The picture comes back as the signal
itself, whether the guide is held for the whole of sampling or dropped after the early steps. The sound, which
the guide did not hold, lands on the signal's windows.

The question came from two clips shared with the owner, in which a hop and a ball's bounce fall on the pulses
of a drawn signal. How those clips were made is not established here. The owner's account is that their author
used a guide node. The author publishes one, read the same day: it anchors several short guides, each a
picture, a clip or a sound, and gives each a fraction of sampling after which it is dropped.

## 1. What was rendered

- **Graph.** `workflows/h3_text_to_video_pika_control_vector_api.json` with its control-vector node removed.
  The guide arms add `MiniMaxH3GuideLatent` (`guide_latent.py`) between the conditioning node and the
  preflight node, with the guide video anchored at frame 0 and nothing else changed.
- **Prompt.** `prompt_bank/t2va_offpath_tortoise.txt` with two clauses changed, read in full before queueing:
  the tortoise also springs off the road three times, and the soundscape adds a thud on each landing. The
  full text is in the run's record. The change has a fault, in section 4.
- **Length and seed.** The bank's frame count for that prompt and the graph's seed, as in the day's other
  tortoise renders.
- **Arms.**
  - `plain`: the prompt, no guide.
  - `bars_a`: the same, with a guide video that is black except for one white bar, lit on three windows. The
    bar's height is the signal: full at the start of a window, falling to nothing by its end.
  - `bars_b`: the same, with the three windows moved later so that they do not overlap `bars_a`'s.
  - `bars_a_rel`, `bars_b_rel`: the two bars arms again with `MiniMaxH3ReleaseReferences`
    (`control_vector_nodes.py`) on the model, which drops the guide's rows after the early steps. Its setting
    is in `exp1_release_record.json`.
- **Server.** ComfyUI as it was running; every arm returned `success`.
- **Repo commit:** `f56d55e8` for the first three arms, `ac9ff2f8` for the released two.

The scripts, the arm graphs and the records (`make_arms.py`, `make_release_arms.py`, `arm_*.json`,
`exp1_record.json`, `exp1_release_record.json`) are in the untracked `data/guide_signal_2026-11-12/`. The
guide videos are under `$H3_INPUT_DIR/guide_signal/`. The clips are under
`$H3_OUTPUT_DIR/Video/guide_signal_2026-11-12/exp1/`.

## 2. What came back

Measured by `measure_exp1.py` in the run folder, which wrote `exp1_measure.json`. Nothing here was listened to
or watched in motion.

- **The picture of every guide arm is the guide.** A black frame with the white bar on the frames the guide's
  bar is lit, and no street, man or tortoise at any point. In the released arms the bar's edges are a little
  soft and the picture is otherwise the same: the early steps had already settled it.
- **The sound of every guide arm follows the windows.** The level inside the windows is many times the level
  outside them. In `bars_a` the level peaks a couple of frames after each window opens. In `bars_b` the
  peaks sit on that arm's later windows, not on `bars_a`'s. The released arms show the same.
- **The sound in the windows is a tone, not a thud.** The energy added inside the windows is in the middle of
  the band, with its strongest frequency near the one a crossing signal's beep would have, and the low band
  where a landing would sit hardly moves. The prompt names a crossing signal that beeps. Another session found
  this first, from one clip; the script here repeats it on all four.
- **`plain` does not show a hop.** In every third frame of the clip the tortoise is on the road, and another
  session that looked at every frame found the same.
- **The guide arms cost more.** The preflight line in the server log shows the guide adding about as many
  rows again as the clip has. The held arms took several times as long as `plain`, and the released arms
  about half as long as the held ones (the two record files).

## 3. What to listen for and look at

- **Whether the tone is the crossing beep.** A spectrum cannot say what a sound is. The `-audio` clips have
  it.
- **Whether the released arms differ from the held ones in anything but the bar's edge.** Stills of one were
  looked at; the other was only measured.

## 4. Readings, not findings

- **A full-length picture guide is copied, held or released.** That is what a guide is for on this model (a
  first or last frame the output must show), and at full length it leaves the picture nothing to generate.
  Dropping it after the early steps does not help, because those steps decide the picture. This route is
  closed on the stock graph. It says nothing about a guide LoRA trained to read a guide as control, or about
  a short guide.
- **The stream that was left free timed itself to the stream that was held.** The sound was not guided, and
  it put its events on the picture's bar, including after the windows were moved. That is the same coupling
  the shared clips show, in the other direction, and it is the reason to expect a sound-only guide to time
  the picture.
- **The free stream chose a sound the prompt already had.** It did not make the thud the prompt asks for on
  a landing; it made the beep. A guide sound meant to time a landing should sound like one.
- **`plain` cannot be read as "the prompt alone will not hop".** The changed prompt kept "plods beside his
  feet" next to the new hop clause, so it asks for two gaits at once. That is the fault
  [`control_vectors/2026-11-12_new_vectors_three_ways.md`](control_vectors/2026-11-12_new_vectors_three_ways.md)
  section 10 names for the morning's batch, made again here. A control for a hop needs the plodding replaced.

## 5. What this cannot see

- **More than one prompt, seed and bar design.**
- **Whether a held sound moves the picture.** Only the reverse was rendered.
- **The reference route.** A signal video given as `<Video 1>`, with the prompt saying what it means, sits
  ahead of the output on the time axis and claims no part of the picture. It was not tried.

## 6. For the next runs

Facts that came out of the day's reviews of this render, by this session and others, each checked against
the code here.

- **A guide never reaches the text encoder.** It is attached after the prompt is encoded and has no label in
  the prompt, so it can time an event the model already ties to that picture or sound, and cannot carry a
  meaning the prompt assigns. A signal that has to mean something (which of two people, which direction)
  belongs in a reference video.
- **The text encoder sees a reference video at two frames a second.** A pulse shorter than the gap between
  those frames can fall wholly between them, as this render's later windows would have. The video model's
  own copy has every frame. The line is in core's reference node, marked "Qwen sees the video at 2 fps".
- **The pack can already drop picture guides partway.** `MiniMaxH3ReleaseReferences` does it for reference
  pictures and picture guides together. It does not cover a sound guide.
- **A release written as a timestep range can land exactly on a step.** The shipped Pika graphs' sigma list
  (`ManualSigmas` in each graph) holds the shifted values of round fractions, and core's range test is
  strict on both sides, so at a round fraction the copy with the guide and the copy without both run on one
  step. Print core's `percent_to_sigma` against the graph's list before choosing a fraction.
- **A sound-only guide is not snapped to a clip length**, and sits on a whole audio position when its frame
  is a multiple of three. A picture guide is one frame or a clip on the model's frame grid, and the shared
  clips' author keeps picture guides on multiples of seventeen.
