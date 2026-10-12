# An example masked job: a face and hair replaced, with eyelids from marks on a body-model video

last updated: 2026-10-12

**An example of one load that rendered, not a recipe and not a text to
copy.** It is the shape of the load the owner picked on playback on
2026-10-11: one person's face and hair replaced on one shot of one video,
one seed. On that shot the marks brought her eyelids down to the
original's level. On the next shot it was tried on, a singing close-up of
another person, it did nothing for her eyes, and why is not known. Read
the two records before leaning on any of it:
[`2026-10-11_kitchen_shot_lid_marks_on_the_body_model.md`](../../results/2026-10-11_kitchen_shot_lid_marks_on_the_body_model.md)
and
[`2026-10-11_lead_closeup_lid_marks_did_not_carry.md`](../../results/2026-10-11_lead_closeup_lid_marks_did_not_carry.md).
The practice as patterns with their conditions is in
[`docs/wiki/masked_v2v.md`](../../../docs/wiki/masked_v2v.md), "The motion
video as a body model, with marks drawn on its face".

## What is here

| file | what |
|---|---|
| `job.example.json` | the job file of that load, with every path of that day's media replaced by a placeholder |
| `text.example.txt` | the whole text that load was given, byte for byte. An example prompt: written from the frames of that one window and true of that window only |
| `README.md` | this page |

Nothing here is read by a tool, a check or a graph. A job is built from a
copy under the untracked `data/`, with real paths and a text of its own.

## What was and was not exercised

The job file is the picked load's own, with one line added at its top
(`what`, a description the repository's index reads; a real job file has
no such key and the builder does not look for it) and ten values changed: the job's
and the load's names, the note, the source, the output folder, the still,
the capture's folder, the text's label, and the two files the load is
handed. The builder's own reader accepts its job block
(`masked_job_graph.check_job`). **Nothing was built, planned or rendered
from this file**: the placeholders are not files. No check holds it, so
if the builder's keys change it can go stale; the builder's docstring
(`bench/masked_job.py`) is the reference for the keys, not this page.

## The job file, key by key

A job is one video and its people; a load is one person on one stretch.
How a job is run, with the capture, its gate and the two looks before a
render, is in `docs/wiki/masked_v2v.md`, "Running a job".

| key | in the example | what it says, and what has to exist first |
|---|---|---|
| `job.source`, `first`, `frames` | a video, source frames 14 to 154 | The stretch is one shot. The video needs its analysis folder under `data/` (`bench/clip_analysis.py init`) |
| `job.canvas` | 1024 by 768 | The size everything is rendered at. The loader fits the source to it by cutting to the canvas's shape and then scaling (`bench/_lib/frames.py::FIT` is the same fit for the tools) |
| `job.subjects` | one person, found by the phrase `person`, the largest on source frame 60, with a reference still | The preview tracks the subject from that frame. `still` is a file in the server's input folder; the load shows it whole to both models |
| `job.capture` | a folder under `data/` | The capture of this job's masks, made by `bench/capture_masked_run.py files` and checked (`verify`, `preflight --gate`) before anything is planned |
| `job.texts` | one text, its file, the frames it was written from, and who wrote it | A text is tied to its frames and its writer. The builder refuses a load its text does not cover |
| `loads[].subject`, `others` | the second subject; `keepout` kept out of the margin | `others` are capture subjects whose masks the region's margin may not cross. Here it was her hands, arms and what she holds, as a mask of its own |
| `carried`, `parts_mask`, `parts_first` | the part, from a mask file handed in by name | What is replaced is a part of the tracked person, here her face and hair as one piece. A mask made after the capture's own is handed in by path, with the source frame its frame 0 is on |
| `margin`, `feather`, `edge` | 64 px, 8 px, latent cells | The region is the part grown by the margin and rounded out to the model's 16 px cells; the feather softens the laid edge |
| `motion`, `motion_file`, `motion_first`, `motion_edge` | a wired video, its file, source frame 14, a short edge of 768 | The motion video: one frame a source frame, shown to the video model on every frame and to the text encoder at one picture a second. Here it is the body-model video with the lid marks, made as described below |
| `window`, `context` | 141, 90 | One window holds the load whole |
| `seed` | one seed | Every figure of that day is one seed a load |

## The text, section by section

`text.example.txt` is the picked render's text. For each part: why it is
there, what it speaks of, and what had to be made before it was true.

**Read this first.** A text is the caption of what is generated inside the
region, beside a picture and a sound that are kept as they are. It names
each reference and says how it is used; a reference the text does not
speak of is one the model cannot know to use. It says nothing that is not
known from the frames or the track. And this one is not a model of a
caption: it still holds sentences about what "stays as it is" and what is
"the scene's own", kept the same across that day's loads so that one
thing changed at a time. A render with them deleted came out as the same
piece. Do not carry them to a new text.

1. **`subject_definitions`, the first line: who `<Subject 1>` is.**
   *Why:* the model is told which reference is the person and what of her
   the still shows. Its last clause, "who follows the movements and the
   facial expressions of the figure in `<Video 1>`", is one of the four
   places that tie her to the motion video.
   *Context:* `<Picture 1>` is the reference still, shown whole to both
   models. The words for her look are read off that still.
   *Required first:* the still, in the server's input folder.

2. **`subject_definitions`, the second line: what `<Video 1>` is.**
   *Why:* this is the line the day's renders turned on. It says what the
   video is, as the model sees it (a plain grey body model on a black
   ground, with shapes on its face), that the shapes are part of its
   facial expression, that she follows its movements and expressions, and
   that its look and the shapes are not drawn. It gives the shapes no
   colour and names no feature of hers.
   *Context:* `<Video 1>` is the load's motion video and nothing of the
   source's picture is in it.
   *Required first, in this order:*
   - the body-model video of the tracked person on these frames, drawn at
     the canvas by the job's preview (`MiniMaxH3SubjectBoxes`,
     `MiniMaxH3BodyPose`, `MiniMaxH3BodyMeshVideo`);
   - the lash-line points of the ORIGINAL on every frame, from the source
     fitted to the canvas, in the tracked person's box:
     `bench/pose_along_shot.py run`, then `read` to see what the points
     can carry (a face only a few pixels tall has lids too close to draw
     from);
   - the marks drawn on the body-model video from those points:
     `bench/draw_lid_marks.py`, each eye's shape grown about its own
     centre by one factor for the shot (`h3_config.LID_MARKS`, or
     `--grow`; in this load the factor was three), with frames a person
     names left bare, and its sheet looked at before the video is used;
   - `motion_first`: the source frame the video's frame 0 is on.
   If the reference is drawn another way, this line says what that one
   is.

3. **`summary`.**
   *Why:* one sentence of what the finished shot shows and which of it is
   the still's (her face and her hair) and which is the scene's.
   *Context:* the scene's side is everything outside the region, which is
   the original's own pixels.
   *Required first:* the part and the keep-out decided, so the sentence
   says only what the region can draw. That day a test's first text
   asked for hair that its face-only region could not draw, and was cut
   before it rendered because no outcome of it could have been read.

4. **`retention_analysis`, the still's line.**
   *Why:* says what of the still is kept (her face and hair) and what is
   not used (its clothing and background).
   *Context:* the vendor guide's retention markers
   (`vendor_guides/ref_en.md`).
   *Required first:* nothing new.

5. **`retention_analysis`, the video's line.**
   *Why:* the same statement for the motion video: its movements and
   facial expressions are transferred; its grey look, the shapes and the
   black ground are not. The third of the four places.
   *Context:* a line for `<Video 1>` as a source of movement is this
   pack's form; the vendor guide keeps `<Video N>` for a relationship to
   a whole video (`docs/prompting.md`, section 9.4). A render with this
   line reworded to say what is retained came out the same.
   *Required first:* the marked video, as under 2.

6. **`detailed_description`: the style line and `[Shot 1]`.**
   *Why:* the caption of the shot, from this window's frames: where she
   is, what she wears, what the light is, and one sentence, "She follows
   the movements and the facial expressions of the figure in
   `<Video 1>`", the fourth place. It does not say what her head, her
   face or her eyes do: a sentence that did would leave a render unable
   to say whether the text or the reference did it, and it would have to
   be written again for every shot.
   *Context:* the kept picture and the kept sound give a sentence its
   timing; a sentence about something neither marks has nothing to time
   it.
   *Required first:* the frames of this window looked at by whoever
   writes it. Another window is another text.

7. **`overall_soundscape` and `non_diegetic_music`.**
   *Why:* the format's last two fields. The sound of the load is the
   source's own track, frozen; these lines were left as the earlier text
   had them and were never tested by themselves.
   *Context:* the track is cut to the load by the builder.
   *Required first:* nothing made; the track is the source's.

## What changed between this text and the ones before it

The four places in 1, 2, 5 and 6 are the whole difference from the load
this one is a change of, whose text took only her head from the video.
The fragment alone, and what to hold to when it is set into another
load's text, is in `docs/wiki/masked_v2v.md`, "The text: the fragment".
