# Handover: read this before anything else

**Rewritten at the end of each working day by the session that led it; the
day before is in git.** This one was written on the evening of 2026-10-11 by
`mrfire`, which held the masked lane's build and the card from midday, after
`mrhop` handed both over. It is one party's account. Its writer's memory was
cut and summarised once that evening, and it knows the morning only from
`mrhop`'s handoff and from [`decisions.md`](decisions.md): where this page
and a record, a commit or a transcript disagree, they are right. A session
that took no part in the day is writing its postmortem and will check this
page against the transcripts.

## Before anything: one shot is one shot

The day's good result was read too far the same evening, by the session
that wrote this page first of all. Three things not to assume:

- **The lid-marks form is not a practice yet.** Marks of the original's eye
  openings, drawn on the body-model video, brought a redrawn face's lids to
  the original's level on ONE shot, and the owner's eye on playback agreed
  with the measure there. On the second shot it was tried on, the same
  evening, her eyes did not follow at all. The written-up practice
  ([`masked_v2v.md`](masked_v2v.md), "The motion video as a body model,
  with marks drawn on its face") says so at its top; read the two records
  below before using it, and do not build it into a graph.
- **A level is not following.** Even on the shot where it worked, no render
  opened her eyes where the original's open. Nothing rendered so far shows
  an eye followed in time.
- **The pose model's figures are trusted in a narrow band.** The eye
  opening is right at its two ends and not in the middle; the mouth figure
  failed its trial; the score of the lid points does not say lids were
  seen. A band read off an original did not carry to a render. All of it:
  `../../bench/results/2026-10-11_sapiens2_pose_first_frames.md`, the
  evening's additions.

The rule of 2026-10-10 still stands in front of all of it: for the load in
front of you, line up its text, its window's frames, its mask and its
audio yourself before you build on any of it (`AGENTS.md`, "When a render is
wrong").

## The day in five sentences

The owner ruled in the morning that the masked lane is built segments
first, on one video, with every decision of theirs answered on a local
page. The render of 2026-10-10 the owner had called close was made again
through today's tools and kept as the reference. Its fault was named and
then measured: her head follows the original's and her eyes do not. A
motion video with the original's eye openings drawn on the body model
fixed that on that shot under a short plain text, and the owner picked it
on playback. Carried to a second shot it did nothing for her eyes, and the
day closed there with one render proposed.

## State when the day closed

- **Commits.** Read `git status -sb`; the owner pushes from their own
  terminal through the day and no session pushed. The day's commits are
  docs and records, plus one tool with its check and three constants
  (`12e460a1`). **No node and no generated workflow changed today**, on
  purpose: the form was not to be built into a graph on one shot.
- **The ComfyUI server is stopped** (by this page's writer, 22:28Z, at the
  owner's ask; the port's owner was killed with an empty queue). Start it
  with the recipe in [`../comfy_notes.md`](../comfy_notes.md). It had run
  all day on the code of `9de6210d`.
- **The owner's page is up** on port 8765 (`data/fun/poc/`, untracked,
  never hosted; `data/fun/poc/README.md` ends with a list of where
  everything is). The owner answers there; every answer is one raw line in
  `data/fun/poc/answers.jsonl`. Read that file whole before saying what the
  owner said: a typed word counts over a button beside it, and a line with
  `taken_down_by` was typed in a terminal and saved by a session.
- **A full check sweep** was started by the session that has the page a
  minute before the server was stopped; read its result from that session
  before calling the tree green, and run one on a started server before
  any node or schema change.
- **Built, looked at, and NOT rendered**, both on the close-up of source
  frames 1575 to 1729: the same load with a shorter wording the owner
  asked to try (`data/2026-10-11_fun_set_inputs/set_close_1575_e.json`,
  first look recorded, no plan); and marks drawn for the third subject on
  frames 1466 to 1535 (no job yet). Both were held when the first
  close-up render came back a null: neither would say why.

## What the day found, and where each is written

| what | where |
|---|---|
| the owner's rulings, in substance, with dated corrections | [`decisions.md`](decisions.md), 2026-10-11 |
| the close render of the 10th made again and kept | `../../bench/results/2026-10-11_close_render_made_again.md` |
| the body-model video carries her head; a face-only region keeps her own hair; nothing carried her eyes | `../../bench/results/2026-10-11_kitchen_shot_head_and_eyes_two_tests.md` |
| the edit-talk in that load's text did nothing that shows; lid marks carry her lids as a level; a plain text is enough; the owner's pass over all eleven renders | `../../bench/results/2026-10-11_kitchen_shot_lid_marks_on_the_body_model.md` |
| the form did not repeat on the lead's singing close-up; what differs between the two loads | `../../bench/results/2026-10-11_lead_closeup_lid_marks_did_not_carry.md` |
| the pose model: what its points can and cannot measure | `../../bench/results/2026-10-11_sapiens2_pose_first_frames.md` |
| which models are on disk, what loads each, and the seven trials | [`meta_perception_models.md`](meta_perception_models.md) |
| on the lead's small street faces the lids are about a pixel apart on the canvas, so no mark can be drawn from them at whole-frame size | the kitchen lid-marks record, "Which other shots can take it" |

## How a render was made and read today, which is worth repeating

Every step is a tracked tool; every bench tool here wants
`H3_OUTPUT_DIR` and `H3_INPUT_DIR` set to the server's folders.

1. A job file under a NEW name beside its text
   (`bench/masked_job.py`'s docstring; a run never reuses a name). The
   job's note says the one thing changed and what the render decides.
2. A capture of its own (`bench/capture_masked_run.py files ... --loads
   <job>`), then `verify` and `preflight --gate`. A flag overridden is a
   person's statement made after looking at the frames it names.
3. The render graph built and diffed, input by input, against the graph
   it is a change of. A strip of every frame with its text
   (`capture_masked_run.py video --load`).
4. A FIRST look by a session that did not make it, which records it
   itself (`looked`). It reads the text whole against the frames.
5. The plan (`build plan`, `queue ... plan --wait`), the capture made
   again on the node's plan (`--run`, never with `--loads`), and the
   planned region held against a known render's saved region. A SECOND
   look. Then the render.
6. Read from files first: the saved region, the pixels and the saved
   latents against the render it is a change of.
7. The pose model on the original and on the render, in the tracked
   box, one table a clip. A render made twice at one seed is the only
   floor there is; without one, say two figures are equal or not and
   call nothing a difference.
8. **The files and no conclusions** to each reader, who writes its
   reading before it reads anyone's. Every correction of substance
   today came from this step.
9. The render stacked with its masks, the reference video and frame
   numbers (`capture_masked_run.py video --render`), on the page, for the
   owner's eye. Never a bare render.

## Where things are

- **On the share**, under the server's output folder: this video's things
  go under `Video/fun/` by the owner's ruling (`review_clips/`, the stacked
  clips of earlier renders; `layered_view/`; `prompts/`;
  `pose_first_frames_2026-10-11/`). Today's renders are under
  `Video/mrfire/<job>/<load>/`, each with a `_windows` folder holding its
  saved region and latents, and today's stacked clips under
  `Video/mrfire/review_2026-10-11/`. The `_plan` folders beside the
  renders are temporary.
- **Untracked data of the day**, under `data/`:
  `2026-10-11_fun_gate_inputs/` (the kitchen shot's job files, texts and
  the two marked reference videos), `2026-10-11_fun_set_inputs/` (the
  close-up's, and the newer of the two scripts that draw marks),
  one capture a job (`2026-10-11_gate_<x>_0014/`,
  `2026-10-11_set_close_1575*/`) and the same names with `_job` (built
  graphs and queue records), `2026-10-11_sapiens2_pose_consecutive/` (the
  pose model's points and one table a clip, with the three one-off
  scripts), `2026-10-11_depth_who_is_in_front/` (trial 2, not finished).
- **The page's review list**: `data/fun/review/items.json`, clips in
  `data/fun/review/media/`.
- **Tracked tools the work ran on**: `bench/masked_job.py`,
  `bench/capture_masked_run.py`, `bench/contact_sheets.py`, and from
  today `bench/draw_lid_marks.py` with `bench/check_draw_lid_marks.py`
  and `bench/_lib/face_points.py` (one definition of each figure read off
  the pose model's points). **Still one-offs**, named so nobody builds on
  them unknowing: the pose pass along a shot
  (`data/2026-10-11_sapiens2_pose_consecutive/run.py`) and the two
  readers beside it (`read_tables.py`, `read_from_files.py`); a session
  is writing them as tracked tools with a check each.
- **Asks and answers**: the H3 Open List (the owner's page with Claude),
  rows g00 to g10 and t58 to t82; the cards on the local page are cut
  from those rows by the session that has the page.
- **Private notes** of each session of the day: `internal/claude/`,
  files dated 2026-10-11.

## Tomorrow, in the writer's suggested order, and why

The owner decides the order.

1. **The owner sees the close-up render** (it is on the page, stacked,
   after the kitchen clips). Three sessions read it as a null for her
   eyes; nobody has judged it on playback.
2. **A look that costs no render**: what of the close-up's reference the
   model is actually shown. A shut eye's mark there is a sliver, and the
   new marks tool reports marks too thin to leave any white. Read the
   reference at the size the video model is given it and at the rate the
   text encoder is given it (the builder's own printed line for the load
   says both).
3. **One render, on the owner's word**: the close-up load with the marks
   at three times, the kitchen's factor, and nothing else changed. Why
   this one: it is the only difference that can be changed alone without
   new code. If it carries, the size rule goes by the face and not by the
   canvas. If it does not, the next single change is the region (a face
   and hair load, or the kitchen shot's face-only load with marks), and
   after that the reference zoomed in with the load, which is node code:
   a zoomed load takes no motion video today (`video_mask.py`, the note
   at `RENDER_ZOOM_REPLACED`).
4. **Then, and only if step 3 carries**, the two held loads: the shorter
   wording, and the third subject.
5. **The two remaining tools** (the pose pass along a shot, a render read
   against its original), each with a check; then the one-offs are
   retired by name.
6. **Nothing of this form goes into the builder or a generated workflow**
   until it has held on two shots. When it does: the builder's motion
   choice, the generated graph, a full sweep, a restart, by one session.
7. **The record of a video on the share** (the owner's "yes, but" of this
   day: whatever they save on the page is structured into the record by a
   session's tool, not by the page). Not started. With it: the cast
   tool's answers "not sure" and "not shown", and the owner's who-is-who
   answers for frames 0 to 603, which are raw lines in the page's file
   and in no cast file.
8. **The model trials** (`meta_perception_models.md`, "The trials"; the
   owner's word was to run all seven in order). Trial 1 is judged. Trial
   2's one-off is written and has not run to its end: the two HyDen
   checkpoints do not load as the reference code's example loads them
   (the script beside the data says how far it got).
9. **Owed from the owner's cards and not started**: the still cut to head
   and hair against the whole still; the singing sentence on four of the
   lead's loads; the contact sheets moved to the share; the owner's three
   notes on the tracker's numbered pictures.

## What the writer is not certain about

- **Why the close-up failed.** Five things differ from the kitchen load
  and none is separated. The record lists them. The writer leans to the
  marks' size and would not be surprised to be wrong: on the stretch
  where the marks are large and plain, the renders without any marks are
  already near the original, so that stretch says nothing either way.
- **Whether a face-only region takes an expression from a motion video at
  all.** The one face-only load of the kitchen shot had no marks.
- **Whether the plain text is really as good as the longer one.** Three
  renders agreed, on one seed and nearly one text: one result seen three
  times. Her head sat a little further from the original's place in all
  three; what would win that back has not been rendered.
- **Whether the measure's agreement with the owner's eye holds anywhere
  else.** Eleven renders of one shot, sorted alike by both. Their "not
  good" was not always about her eyes.
- **The wording that is written up.** The owner chose it on a render they
  liked; it rests on that one shot. Each load still keeps its own copy of
  the reference's lines in its text file, and the close-up's text also
  holds the 10th's standing sentences and one sentence about a handset
  that is true of part of its window, left so its text would differ from
  the praised render's by four additions only. That is a debt.
- **Anything about the morning** beyond what `decisions.md` and
  `mrhop`'s handoff (`internal/claude/2026-10-11_mrhop_handoff_to_mrfire.md`)
  say.

## What to carry forward

- **Read per-frame columns before a headline.** The first reading of the
  marks said her eyes moved toward the original's; split by where the
  original's eyes are open, they had only come down. A blind reader found
  it. The split existed in the table from the first minute.
- **A reader given files and no conclusions** is worth more than a second
  opinion on a conclusion. Give it the paths, and wait.
- **Write what a render is expected to show before it samples**, and what
  would count as the reference being obeyed against the picture. It made
  the close-up's null readable in minutes.
- **A rule that worked on one face is a starting value.** A size rule by
  the marks' width failed on a profile within the hour; its replacement,
  by the canvas, may be what failed the close-up.
- **A click is not a go when words follow it.** The owner's answers came
  through three terminals and the page; each was read first hand in the
  page's file before anything was queued on it.
- **One committer a change, and nothing uncommitted in a config file**:
  the builder refuses to queue beside uncommitted node code, and it
  counts `workflows/h3_config.py`.
