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

*Checked the same evening against the transcripts by the day's postmortem,
[`2026-10-11_span_fun-lid-marks-one-shot.md`](../research/postmortems/2026-10-11_span_fun-lid-marks-one-shot.md),
written by a session that took no part in the day. Its section 8 has the
rows. This page is right in outline and is left as written below; these are
the places it is wrong or short:*

- *"Every step is a tracked tool": step 7, the pose model on the original
  and on the render, ran on one-off scripts all day, as "Where things are"
  says further down.*
- *"A flag overridden is a person's statement": true of the overrides. After
  the afternoon's first capture, no capture that led to a render has an
  outcome entered for its flags beyond its maker's overrides; flags went
  round as a count.*
- *"Each was read first hand ... before anything was queued on it": true of
  what was queued. Three things were built on the first part of a run of
  relayed messages and withdrawn by its end; none was rendered.*
- *The morning is thinner here than in the rows. Not on this page: the
  morning's whole-person render of the opening shot, which failed and whose
  cause was never separated; that the floor `PART_LEAST_PX` emptied half of
  one load's part mask while its gate read clear; the second reader's
  warning that the overnight preview's masks should not be planned from; and
  the owner's points on the joined clip, which no render of the day
  addressed.*
- *Three things the rows put ahead of where "Tomorrow" has them, or that it
  leaves out: a zoomed load that keeps its motion video (the owner has twice
  called the zoomed render the better one and has not been asked); the
  vendor guide's own form of the reference text on the large marks (named as
  the next single change three times and not run); and a second seed of the
  picked render.*
- *One thing this page does not say and the rows do: the result was first
  told to the owner from a whole-shot average by two sessions in two
  terminals, while the blind reader's split was already in a peer's inbox,
  and the owner's word to keep moving through the scenes came before the
  correction.*

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

## After the page was first committed: a fault in the tools' own reading of a source, and three tools

Added the same night by this page's writer, when the last session of the
day had finished.

- **The bench tools fitted a source to the canvas in a different order
  from the loader, about half a row off on this video.**
  `bench/_lib/frames.py::FIT` scaled to cover and then cut; the graphs'
  video loader cuts to the canvas's shape and then scales. Found by the
  session writing the pose tool, which held the tool to giving a one-off's
  points exactly. Corrected in `100f9e09`, whose message says how it was
  found and what it means for every capture made before it; read that
  message before trusting an older capture's reading of a render against
  its source. `bench/check_loader_fit.py` now holds the fit to the
  loader's own line. **No workflow and no node carried it**: the job
  graphs load through the loader itself, and the Masked Source's own
  resize cuts first too. **No figure in the records of 2026-10-11
  moves**: they were made by one-offs that cut first.
- **What that correction does to a capture, read both ways on one real
  capture by this page's writer** (the kitchen load, made again on the
  corrected fit beside the same capture made earlier that day; the
  figures are in the message of the commit that placed this paragraph):
  the gate's flags are the same, every one. The two per-frame columns
  about the source's own picture move a hair. **The reading that moves is
  `changed`**: its floor falls, so every "times the floor" it prints rises
  with nothing redrawn differently, and a multiple of the floor quoted
  before that night would read higher if made again. One threshold is a
  multiple of that floor (`capture_masked_run.source_at_the_ends`: a
  render within three floors of its source is not read). Not read both
  ways, reasoned only: `look` and the un-tinting of a review.
- **The older stacked clips on the owner's page** (`Video/fun/review_clips/`)
  and its plain stills were built with a scale-first fit: their top row
  sits that half row from the render. Known, not remade; nobody measures
  from them. `data/fun/poc/README.md` says what to change when one is
  next built.
- **Three tools are tracked, each with a check**, written that evening by
  one session from the day's one-offs: `bench/pose_along_shot.py` (the
  pose model on every frame of a shot; `run` needs the card, `read` says
  what a shot's points can carry before anything is drawn from them),
  `bench/draw_lid_marks.py` (the marks on a body-model video, with a
  `verify` for a second reader) and
  `bench/read_render_against_original.py` (a render's face beside its
  original's, split by where the original's eyes are open, with a floor
  or a plain "no floor given"). One definition of each figure:
  `bench/_lib/face_points.py`; their starting values:
  `h3_config.POSE_READING` and `LID_MARKS`. The one-offs under
  `data/2026-10-11_sapiens2_pose_consecutive/` are superseded and stay as
  what the day's tables were made by. One thing still open: the list of
  the pose model's point names is an untracked file
  (`data/2026-10-11_sapiens2_pose_consecutive/names308.json`, read out of
  Meta's keypoint config), so a fresh checkout cannot run the pose tool
  until it has a tracked home. *Decided at the close and not built when
  this was written: Meta's Sapiens2 names (the pose points and the part
  model's classes and colours) go into a folder of their own with Meta's
  licence file beside them, and the part node reads its names and
  colours from there, which is a node change with a restart and a smoke
  test ([`decisions.md`](decisions.md), 2026-10-11, the last entry).* [`masked_v2v.md`](masked_v2v.md), "How it
  was made", names both tools and says nothing joins them up: no node, no
  shipped graph, and the builder is handed a marked video and cannot make
  one.
- **What the owner said the tools are for, at the close** (in substance;
  the Open List's row t83 has it):
  - A tool gives data and its context and leaves the reading to whoever
    runs it. It does not print a verdict.
  - A standing check is worth having only where it tells the runner
    something before time is spent. For the fit that means: a capture
    says up front whether the source's shape is the canvas's and how much
    the loader cuts, not a comparison run after every render.
  - **A figure is a tape reading.** Near the original means nothing by
    itself; sometimes a higher or a lower figure is what is wanted; and
    the scales are not known yet. What exists is single points, not
    scales: one shot where the owner's verdicts fall on two sides of one
    eye figure, a line drawn after their answers; a band read off one
    blurred close-up, whose middle is where the second shot's render sat;
    a floor from one render made twice. Nothing says where a head's
    offset becomes visible, what difference in pixels a viewer sees, or
    anything for a mouth, a likeness or a movement. A scale would come
    from pairing the owner's verdicts on playback with the figures over
    more shots, one measure at a time.
  - **The capture tool's purpose is the whole path.** For each thing a
    node uses on the way from source to target (the source's frames, a
    mask, a reference video, a still, the text, the sound), a reader can
    see how it came in, every change made to it, by which node with which
    values, at which step, from which node to which, for every window
    that runs. The fit fault is the plain case of what that would have
    shown. And the capture as a workflow is wanted for the same reason.
    Nothing is built for this yet and it is not sized: it is the
    direction for the capture tool, and its first step is to list what
    the saved graph, the plan, the saved region and the tables already
    hold of it.

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
  docs and records, plus three bench tools with their checks, three
  constants and the correction of the bench tools' fit (the section
  above). **No node and no generated workflow changed today**, on
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
- **A full check sweep** (`bench/run_checks.py`, the card masked) was run
  that evening by the session that has the page and read by it: no check
  red. It began a minute before the server was stopped, and the tree
  moved under it; the one commit of code that landed meanwhile (the tool
  that reads a render against its original, with its check) had its
  check and two others run again alone afterwards, green. **What it does
  not say:** nothing on the card was checked, and no check ran against a
  live server: two could not reach the stopped one and the checks that
  need the card were skipped as the tool skips them. So run a sweep on a
  started server before any node or schema change. Its counts and the
  list of what was not graded are in the message of the commit that
  placed this paragraph.
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
  that evening `bench/pose_along_shot.py`, `bench/draw_lid_marks.py` and
  `bench/read_render_against_original.py`, each with its check, over
  `bench/_lib/face_points.py`. *When this page was first committed two
  of the three were still one-offs; the section near the top says what
  landed.*
- **Asks and answers**: the H3 Open List (the owner's page with Claude),
  rows g00 to g10 and t58 to t82; the cards on the local page are cut
  from those rows by the session that has the page.
- **Private notes** of each session of the day: `internal/claude/`,
  files dated 2026-10-11.

## Tomorrow, in the writer's suggested order, and why

The owner decides the order.

1. **The owner sees the close-up render** (it is on the page, stacked,
   after the kitchen clips). Three sessions read it as a null for her
   eyes, each before seeing another's, and a fourth worked the eye figure
   again from the same tables; nobody has judged it on playback.
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
5. **Use the three tracked tools for the next render's reading**, and
   make its points with `bench/pose_along_shot.py`; the one-offs are not
   run again. *First written as "the two remaining tools": both landed
   the same night.*
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
