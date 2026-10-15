# Masked video to video: how it works, what it cannot do, where to go next

last updated: 2026-10-13 (a load rendered zoomed in takes a motion video it names, in "A small subject, rendered zoomed in" and "Limits known today"; what the two said before is in [`decisions.md`](decisions.md) under that date); before that 2026-10-12 (reorganized: an index at the top, the sections in the order a reader meets the lane, and every clip, file and function the page names opened and held to its sentence; what moved, what was cut and what was corrected is in [`decisions.md`](decisions.md) under that date, the table of what was opened is [`2026-10-12_masked_v2v_page_held_to_its_clips.md`](../../bench/results/2026-10-12_masked_v2v_page_held_to_its_clips.md), and the page's earlier edits are listed in "How this page grew" at the end)

Written by hand. This is the lane's map for a reader who has not followed
it: the pieces in the order a render meets them, the limits each one has
shown, and the directions worth trying. It carries no measurement. A value
is named by the constant that holds it, a result by the dated record that
holds it, and where this page and the code disagree the code is right.

The dated account, with the owner's words, is
[`../h3_audio_freeze.md`](../h3_audio_freeze.md) section 4. What was
decided is in [`decisions.md`](decisions.md) under 2026-10-04. Each session's
working notes are in [`../research/masking/`](../research/masking/).

*2026-10-12: "the masking board" below names a board that was retired that
day ([`decisions.md`](decisions.md), 2026-10-12: the Open List is the one
place a decision is asked and answered). A card's name is kept where a
finding was filed under it. Whether each card was carried to the Open List
was not checked when this note was written.*

## Index

One line a section, in the page's order, with the question it answers.

| | section | the question it answers |
|---|---|---|
| the lane | "What it is" | what the lane does, which graphs ship, and what the mechanism is in core |
| | "The frame every masked text is written in" | whose side a text is written from, and how little a first text says |
| the rules | "The rule: data before a render, and the same data after" | what has to exist before a render and what is read first after one |
| | "Reading one window whole: its text, its frames, its mask, its audio" | what to put side by side before building on a load or diagnosing a render |
| | "Rules of the lane" | the lane's habits in one list, each with where its reasoning is |
| a job | "Running a job: one tool, on the video's analysis folder" | how a job is built and queued, stage by stage, and what the builder refuses or shows |
| | "Two neighbouring shots in one load: a tool, not a default" | when a load may hold a cut, what it is made from, what it costs, and what is not known |
| | "Seeing what the tracker and the masks did" | which file or command holds what a run was given, what it wrote, and the file that gets watched |
| | "Who is who: a session reads first, the owner answers, and both are kept" | how a tracker's pick becomes the owner's answer in the cast file, and where what each of them thought is kept |
| the pieces | "The pieces, in the order a render meets them" | what each of the four nodes does, with its inputs and their dated notes |
| | "Nothing is kept between runs by a key" | what was once reused by a key, what became of each, and what "version" means in three places |
| | "A latent step is several frames, and its grid is fixed for a whole load" | why a frame cannot be treated alone, what a cut inside a step does, and why a window follows its context |
| what renders showed | "What is known from renders, by form" | which forms have rendered, on what, and where each result is written |
| | "The motion video as a body model, with marks drawn on its face" | what the marked body-model reference is, how it is made, its text, and what it carried on one shot and not on a second |
| | "A small subject, rendered zoomed in" | what `render_region` does, what it buys, and what its one render showed |
| limits and directions | "Known limits" | what the lane cannot do, each with where the evidence is |
| | "Directions" | what is worth trying and what is known about each |
| where to look | "Where to look" | which file answers a question this page does not |
| | "How this page grew" | what was added on which day, and how the page was reordered |

## What it is

Take a video with its own audio, keep everything outside one person, and
replace that person from a reference still. The shipped graph is
`workflows/h3_video_to_video_masked_song_pdd8_api.json`: a video loader, the
SAM 3 checkpoint, three pack nodes (who, what happens to them, the prompt),
and the PDD8 song chain with Sol-Attn.

A second shipped graph, `workflows/h3_video_to_video_masked_song_ref2va_motion_api.json`
(2026-10-05), is for a shot where the replaced person must move as the
original moved: the same lane on the ref2va base at
`h3_config.MASKED_MOTION_STEPS`, with the Masked Source's `motion_reference`
on (`MASKED_MOTION_SOURCE`), so the subject's own frames on grey reach the
text encoder as `<Video 1>` and the prompt ties the subject's motion to it.
Why ref2va and why that count is measured, not reasoned:
`../../bench/results/2026-10-05_masked_v2v_motion_arms.md` (fl2va ignores the
reference at any step count, distilled or not; ref2va loses it at eight). It
costs the step count over the PDD8 graph, which stays the default for a
shot that needs no movement from the source.
Two more graphs since 2026-10-06, neither rendered when that was written,
and a third since 2026-10-07:

- `workflows/h3_video_to_video_masked_review_api.json` is the look before a
  render. It is the default graph with the song node on `preview`, so
  nothing samples and no model loads, and with what the render graphs leave
  unwired: the tracker's numbered tiles and shot table are saved, the Masked
  Source's preview shows the region, and the prompt node shows the text. A
  wrong shot is corrected here, in the Subject Track's `corrections`, and the
  mask it tracks is kept for the render graphs.
- `workflows/h3_video_to_video_masked_parts_song_pdd8_api.json` replaces a
  part of the subject that Sapiens2 finds: `MiniMaxH3Sapiens2Loader` and
  `MiniMaxH3SubjectParts` feed the Masked Source's `parts`, and `replace` is
  `the wired parts`. At the part node's own ticks that is hair, face and
  neck, on the original's body and clothes, so the body's movement is the
  source's. Ticking another part changes the region, and the prompt node's
  `picture_gives` then has to say what the still provides. With upper
  clothing and hands ticked as well, that is `the head and upper body`
  (2026-10-06): the still's head and what it wears above the waist, on the
  original's legs. It is the choice for a still that shows the person from
  the chest up, which cannot dress legs; where it came from and what has
  and has not rendered is in `masked_prompt_text.py`'s docstring. The
  fast parts graph does not tick those parts; the graph below does. (Said
  until 2026-10-07 that no shipped graph ticked them.) What the part node was seen to find,
  on which clips:
  `../../bench/results/2026-10-05_sapiens2_first_frame.md`.
- `workflows/h3_video_to_video_masked_upper_song_ref2va_motion_api.json`
  (2026-10-07) is the ref2va motion graph with the region cut to the head
  and upper body: the part node at `h3_config.MASKED_UPPER_PARTS` (hair,
  face and neck, upper clothing, hands), the legs the source's own, and the
  prompt node told so. It is the one recipe the owner called solid on
  playback, on one window of one clip at two lengths and one seed, and
  better at a smaller margin on a clip where the subject is small:
  `../../bench/results/2026-10-06_masked_v2v_body_window_arms.md`. That arm
  rendered a typed text; this graph renders the node's wording for the same
  region. That wording rendered the same day on the same window and seed,
  beside the typed arm, and the two are stacked on the output share
  (`Video/compare_stacked/body_345_p7_typed_beside_node_over_mask.mp4`:
  the typed arm on the left, the node's on the right, each over its own
  mask, titles burned in and no frame numbers).
  The owner watched the pair on 2026-10-07: both good, and not told apart.
  (Said for a few hours that day that the wording had not been watched,
  which was this page not knowing that arm existed.) Start here for a still that shows the
  person from the chest up. The same region on the fast chain was called
  broken in that record, so no fast graph carries it.

All five are in `workflows/daily/` too, as `h3_mask_pdd8_api.json`,
`h3_mask_ref2va_motion_api.json`, `h3_mask_review_api.json`,
`h3_mask_parts_pdd8_api.json` and `h3_mask_upper_ref2va_motion_api.json`
(`h3_config.DAILY_GRAPHS`).

One more generated graph wires a Masked Source, a probe and not a shipped
graph: `h3_probe_v2v_masked_song_ref2va_motion_cache_api.json`, the ref2va
motion graph with the frozen video cache (`MiniMaxH3FrozenVideoCache`) on
the song node's model, so the rows a masked window keeps are computed once
a window. Its first run saved no time and the owner retired it on
2026-10-06; the cause was one call running on the wrong attention kernel,
the corrected pair is
`../../bench/results/2026-10-07_frozen_cache_masked_window_fixed.md`, and
the owner brought the masked use back into the tree on 2026-10-07. It has
run on one window. No shipped or daily graph carries it.

It is not a trained task. The release trains t2va, fl2va and ref2va; a
spatial mask on a base checkpoint is an inference-time method. The mechanism
is core's: a latent noise mask on H3 is a per-token timestep
(`comfy/ldm/minimax/model.py::mask_row_values`), and the clean latent is put
back into the kept tokens every step. *2026-10-09: what the model is shown
in a kept token is the clean latent mixed with noise at the level core uses
for a reference, labelled at that level
(`comfy/model_base.py::MiniMaxH3.scale_latent_inpaint`); "clean" on this page
means that. How vllm-omni's mask editing and the third-party masking packs
compare with this lane, and what they do that it has not tried:
[`references.md`](references.md), "Masks and edited video upstream, beside
the masked lane".*

## The frame every masked text is written in

*Moved here on 2026-10-12, word for word: the first paragraph from the head
of "The text: the fragment", where it was written earlier that day, and "Say
the least first" from under the Masked Prompt node. Both are about every text of
the lane and not about one node or one form of reference.*

**The frame every masked text is written in** (the owner, 2026-10-10, in
the postmortem of that day's span, and again 2026-10-12 to the lead): the
model is not editing a video it can see. It generates new pixels in the
masked region, and nothing is replaced from its side. What steers it is
what its training makes of the frozen frames and the sound round the hole,
the prompt, and the references. So the text gives it enough to infer what
happens there: each reference asset described, and what of it is preserved,
partially preserved or transferred (`retention_analysis`); what is
generated (a face, or a face and hair); and the movements and expressions
from the body-model video with its shapes. A sentence that describes the
scene as already lit, framed and cut, or a thing as "the scene's own", is
addressed to an editor, not to this model, and was ruled out on the 10th.

**Say the least first, and add a sentence only when a render asks for it**
(the owner, 2026-10-10: "let the model guide and infer until you need to get
more specific", "just like progressive disclosure"). It is the rule "The
text says nothing it cannot know" (under the Masked Prompt node, below)
carried to a text written by hand for one window. The frozen picture round
the hole and the frozen audio already tell the model most of what a text
would: the setting, the light, the framing, the pose where the region is
small, and whether a voice is on the track. So a first text discloses in this
order and stops:

1. **Who is drawn and from which reference, and this part is never
   minimal**: the subject's definition, what `<Picture 1>` provides, and the
   vendor guide's marker for it (`vendor_guides/ref_en.md`, section 4.1):
   `fully_preserved` for a whole person; `partially_preserved` when only a
   part of the still is taken, with what is kept and what is not used said
   in the line; `attribute_transfer` for a motion video.
2. **Nothing about the frozen picture.** *Corrected 2026-10-15 (the owner):
   until then this step asked for "where the subject is, and what on them is
   the scene's own (the hair or clothing the plate keeps), in a sentence".*
   The masked region is a hole of noise and H3 generates new video in it;
   where a person stands, what she wears and the room round her are frozen
   context it already has, and naming them does not tell it which face goes
   in which hole. That is a sentence to an editor.
3. **Nothing about action, expression or voice.**

Then render, look, and add one sentence for what was missing: only what is
plainly visible in the source at the frames the sentence is about, or
measured. A voice is measured (a separation of the track says where one
is); an open mouth in a still is not a voice. An added sentence is checked
as a default is, against the source, and it is the first thing deleted when
a render misbehaves.

What the first day of this showed, on one clip, one seed each, read from
stills and not judged on playback, so a direction and not a record: a
whole-person swap with hands inside the hole did not draw a two-handed
action until one sentence named it; a sentence placing a gesture "near the
end" of a window was drawn through most of the window; and a sentence
saying the subject performs the voice was written for shots where the track
has no voice. Two of the three are a sentence added before a render asked
for it.

Where the rest about a text is:

- what the Masked Prompt node writes, and "The text says nothing it cannot
  know": piece 3 of "The pieces, in the order a render meets them".
- the lines for a body model with marks on its face, and the condition they
  hold under: "The text: the fragment".
- reading a load's text whole against its frames: "Reading one window
  whole" and "Rules of the lane".
- the reference format and the vendor's guides: [`../prompting.md`](../prompting.md).

## The rule: data before a render, and the same data after

Set by the owner on 2026-10-10 for every session on this lane: "take a data
driven approach to all of this". It is a rule, not advice, because the lane's
earlier jobs spent their card time finding out what a few minutes of looking
at masks would have said, and explained failures by a session's reading that
the next render refuted.

1. **Before a render: capture, flags, a look, then the render.** A
   no-sampling run saves what every tracker and part node made, for every
   person and object in the load (`bench/capture_masked_run.py`, "Seeing what
   the tracker and the masks did" below). Its `preflight` lists what is at
   risk, each flag with its reason and its frames: a person taken in a shot
   they are not in, a part mask off its person, one person's region over
   another, things inside a region that are not its subject, a text that
   disagrees with a measured fact. The flagged frames are looked at. The
   render then loads the masks that were checked and does not track again.
   Nothing renders on a guess about what the tracker took.
2. **After a render: the capture is read first.** When a render is wrong,
   the first step is a lookup, not a probe: on the frames in question, was
   the mask on the right person, what else was inside the region, what was
   the model shown as movement and as text, was there a voice. A new probe
   comes only after the tables have been read and found silent.
3. **A "why" cites a row, a frame range or a measured figure.** A cause
   stated without one is labelled a reading, and gets a control before
   anything is built on it.
4. **Every flag gets its outcome, and every surprise becomes a flag or a
   gap.** What happened on the flagged frames is written beside the flag, so
   a threshold's provenance can move from reasoned to measured. A fault no
   flag predicted is a new rule in the preflight, or an entry in
   `data/CAPTURE_GAPS.md` saying what was not captured, too coarse, filtered
   wrongly or resting on a one-person assumption.
5. **Captured data lives under `data/`**, which is not tracked. A tool
   worth running twice lives in the tracked tree with a check, never in a
   scratchpad: the 2026-10-07 captures were made by scripts that were not
   kept.

What the upfront data can and cannot say: it says where and why a render is
at risk. It does not say what the model will draw inside the region; that
still takes a render, and the flags say which shot to render first and which
frames to look at.

## Reading one window whole: its text, its frames, its mask, its audio

How the lane got to work, written down so nobody has to find it again
(owner, 2026-10-10). It is a habit and not a gate: nothing checks that it
was done, and no more refusals are wanted for it. On that day a wrong
render was put down to a signal four times, renders were spent, and the
cause was in the text each time; the other sessions had never seen the
texts, only one session's description of them. The account is
[`2026-10-10_span_fun-three-people.md`](../research/postmortems/2026-10-10_span_fun-three-people.md),
section 1.1.

**When.** Before building on a load somebody else made, or you made on
another day. And first, before any test render, when a render is wrong in a
way that is not obvious.

**What.** Take ONE window of ONE load and put four things side by side for
the same frames. Assume one of them is out of step until you have looked.

| what | where it is today |
|---|---|
| **The text**, whole, as the model gets it | `bench/masked_job.py build` prints each load's text in full with the frames it was written from. The look's strip draws it under the frames: `bench/capture_masked_run.py video <capture> --source CLIP --job JOB.json --load LOAD`. The texts are short. Read the text itself; hand another session the file, never a sentence about it; and when a cause is not obvious, show the owner the whole text |
| **The window's frames**: which source frames, which are held, which share a latent step | the raw frames themselves are already laid out by frame number in [`contact_sheets/`](../../contact_sheets/AGENTS.md), one folder a video (`bench/contact_sheets.py find <video> <frame>`): open a sheet before sampling the video again. Which of them a window gets: the plan the song node's preview writes, `<name>_plan.json`, and each window's `<name>_window_N_planned_region.npz` ("Seeing what the tracker and the masks did", below); "A latent step is several frames", below, for why a frame can be drawn with no mask of its own. For a finished render, `bench/recomposite_window.py --read-files` reads the window's own files with nothing decoded |
| **The mask** on those frames, and whose it is | the capture's stacked, frame-numbered video (`bench/capture_masked_run.py video`), its per-frame tables, and the region each window was really given, `<name>_window_N_region.npz` |
| **The audio** the load was given, and whether a voice is on those frames | `build` prints an `audio:` line for each load (a sentence, not yet data: the audio in a load's record is not landed). The capture's voice table, and `bench/capture_masked_run.py shots --loads`, which says how many of a load's frames have a voice and where the source's mouth is open with none |

**What it is for.** Every fault of 2026-10-10 that cost a render was two of
these four disagreeing: a text written from one shot on a load of another;
a frame drawn with no mask of its own; one sample of audio deciding a
frame; a text saying the scene is quiet over a song. Each was gated on its
own and passed.

**Not built.** One command that shows the four together for a window (the
masking board, `build-why-command-text-mask-four-timelines-first`), and the
statement of who is in each window with what it rests on
([`handover.md`](handover.md), "After the close"). Until they are, the four
places above are the route. *2026-10-12: `handover.md` no longer has a
section of that name. Who is in each shot is now `data/<video>/shots.json`,
answered on the page `bench/who_page.py` makes (`AGENTS.md`, "Where state
lives"); the one command is still not built.*

## Rules of the lane

Moved here from `AGENTS.md` on 2026-10-12, when that file was cut to what a
session cannot work around; these are the lane's own habits, each earned on
2026-10-10 or 2026-10-11, and the two sections above hold the reasoning.

- **A masked render is run on data, before and after.** A no-sampling
  capture and its flags before it samples (`bench/capture_masked_run.py`);
  the same capture read first when it goes wrong; a "why" cites a row, a
  frame range or a measured figure. Captured data lives under the untracked
  `data/<video>/`.
- **Preprocessing data is looked at the moment it is made**, by the tool's
  `verify` and by a reader who did not make it (a subagent counts): the mask
  on the right subject on every frame, the part, the motion video, each
  file's frame offset. A passing check says no rule failed, never that the
  data is right; a capture records who made it and who looked.
- **A scene's text is data, served by whatever plans the shots**, never
  stored verbatim in code or in a workflow. It is written from the frames of
  one window and is true of that window only; it is a plain caption of the
  finished shot (what is seen and heard), never a statement of what is kept,
  and it says nothing about sound, action or expression not known from the
  frames or the track. Know which text a load gets, read it whole against
  that load's frames before it is used, and check after the render that it
  did what was expected.
- **When a render is wrong, read four things together before anything is
  suspected or queued**: its text, its mask, and how the four timelines line
  up (the input video's frames, the mask's frames, the sampler's frames and
  the audio). The owner's count is that it is one of those about half the
  time; on 2026-10-10 three renders went to a kernel setting and an audio
  file before the text was read, and one sentence fixed it. The text is read
  whole by whoever reasons about the load, never as a summary; when the
  cause is not obvious, the owner is shown the whole text. "Reading one
  window whole", above, is where each of the four is.
- **A graph posted by hand is a temporary plan**: whoever posts one reads the
  full text first and writes beside the job which built graph it came from,
  the one thing changed, by whom, and what the render is meant to decide.
  (Since the builder landed, nobody has; the job's `note` holds the same.)
- **The reference for a face, head or hair replacement is the body model cut
  to the head with the face points drawn on it, and the text says exactly
  what that video is** (the owner, 2026-10-12, about 20:12Z, after the two
  cropped-reference tests on 1039-1060; made with `bench/head_reference.py`,
  the three steps as "Running a job" has them). On the lead's three hall
  shots that evening the owner judged this form better than the lid marks
  on playback (the eyes on 1222, the mouth on 1156, the gaze on 1093;
  `data/fun/verdicts/verdicts.jsonl`, 21:25-21:28Z), where the readings from
  frames had called the eyes a draw: what shows in motion is the owner's to
  see. A face too small or too turned for the lines (the lead on 604-948, 65
  px, 379 of 406 frames under the drawing line) takes the plain body model,
  and the note says why. The standard is provisional on holding up across
  the rest of the video (the owner's own words, 20:18Z).
- **A rule that depends on the picture takes its value from the input video,
  up front** (`subject_track.auto_cuts`, `auto_match`); a constant measured on
  another clip is a marked starting value, and the tool says when this
  clip's own figures disagree with it.
- **Nothing an earlier run made is reused by a key** ("Nothing is kept
  between runs by a key", below): a mask is preprocessing data for the job
  that reads it, rebuilt when preprocessing runs again.
- **The canvas follows the source's shape, and every mask is made at it.**
  A job is made on `h3_config.MASKED_JOB_CANVAS`, the 4:3 row of
  [`../h3_resolutions.md`](../h3_resolutions.md), because the video is 4:3
  (`ffprobe` on its working copy) and the whole frame then fits with nothing
  cropped or padded. The 3:2 row of that page (1152x768) is as much a trained
  size and is the wrong shape for this source. Every preview, track and mask is
  made at the job's canvas (`capture_masked_run.py verify`,
  `inputs_at_the_canvas`), so a change of canvas is every preview made again,
  and pieces kept before it are on another canvas than pieces made after. A
  load rendered zoomed in samples the same canvas whatever its box ("A small
  subject, rendered zoomed in", below): it buys size for the subject, not
  time. Asked and settled with the owner on 2026-10-13
  ([`decisions.md`](decisions.md)).
- **A window's length is a choice, and one shot stays in one window where a
  valid length holds it.** H3 takes every frame count on its own grid: what
  `h3_rules.snap_length` returns (the grid of `h3_rules.FRAME_FACTOR` and
  `h3_rules.FRAME_REMAINDER`), from core's floor up to
  `h3_rules.MAX_LENGTH`. The lengths the planner and the builder choose from
  (`loop_plan.CHAIN_LENGTHS`, read by `bench/masked_job.py`) are a default
  somebody set, not a limit of the model, and nothing about the model stops
  a load from sampling fewer frames than the shortest of them or more than
  the longest. Fewer: a short single-window load needs no frames past its
  own, and a window longer than its load is sampling that is thrown away
  (the plan's own line says how many frames the last one is held for).
  The only reason to sample more frames than a load's own is an overlap
  buffer before its start or past its end, for a join to what comes before
  or follows; a shot bounded by a cut on both sides needs none. The few
  frames a load holds past its end at its fitted length are the grid's
  rounding, not a buffer.
  More: every frame added raises what attention costs, and with enough
  references a long window can run out of memory, but it is the right
  choice when it lets a longer static shot sit in one window with no cut
  between windows. Keep one shot or cut in a single window wherever a valid
  length holds it: it is one text, written from that window's frames, and
  no join inside the shot to settle. A shot is a cut: "shot" is H3's own
  word for one (the owner, 2026-10-13), and it is the word a text uses when
  it has to describe more than one (`docs/prompting.md`, the `[Shot N]`
  form). One cut is not divided between two windows: a shot in the kitchen
  with both subjects in it is one window, however it is lit or however
  long, up to what the model takes. A cut to another place is another shot,
  and as the lane is built each shot is its own load with a text written
  for that one shot. (Two short shots could share one window where the
  tracking across the cut is easy, and the text would then have to
  describe both, as `[Shot 1]` and `[Shot 2]`; no load is made that way
  today. *Later the same day: one can be made that way on purpose, by a
  job that names its two shots; "Two neighbouring shots in one load: a
  tool, not a default", below. It stays a tool: no builder or planner
  makes one unasked.*) A shot is divided between windows only when it has to be: it is
  longer than `h3_rules.MAX_LENGTH`; memory or time will not hold the
  whole window; an edge case; a fix or polish pass on part of it. As built, a
  load longer than the planner's longest length runs in chained windows
  even where `h3_rules.MAX_LENGTH` would hold it whole
  ([`next_steps.md`](next_steps.md)). Not every count on the grid is exact
  on the audio clock as well (`workflows/h3_config.py`, the note on
  `LONG_LENGTH`), and it does not have to be for a window that stands
  alone: core rounds the audio latent's count (`temporal_shape` in core's
  H3 node) and the lane cuts exactly that many audio steps
  (`audio_freeze.slice_window`); the audio clock is a rule of the seam
  between two windows (`audio_freeze.window_geometry`). The owner's word
  of 2026-10-13 ([`decisions.md`](decisions.md)), and built that day: a
  load one window holds is given the shortest length of the grid that
  holds it (`masked_job_graph.window_for`), the planner takes such a
  length for a track it holds whole as one window
  (`loop_plan.stands_alone`), and a load may name a `window` of its own.
  A load too long for one window runs in the chained lengths, as before.
  At a short window Sol-Attn may decline a call, under its own least
  tokens (`sol_attn_h3._ineligible`); the load is then sampled by the
  model's plain attention, which is fine (the owner, 2026-10-13: Sol's
  sparsity can only take quality away). The server's log says which, once
  for each shape of call (`dense (...): seq N < ...`, printed while the
  node's `verbose` is on); it is read on each short load and written in
  its item as data.
- **What is on a frame of an input video is already in `contact_sheets/`**
  (`bench/contact_sheets.py find <video> <frame>`): open a sheet before
  sampling a video again.

## Running a job: one tool, on the video's analysis folder

A masked job is built and queued by `bench/masked_job.py`, from a job file,
and by nothing else: until 2026-10-10 every session wrote its own script for
this, each patching a graph lifted out of an earlier render. The tool's
docstring is the reference; `bench/check_masked_job.py` holds it. What a
session starting a job needs to know:

- **It starts from the input video's analysis folder.** `data/<stem>/`, named
  by the video's file name, made by `bench/clip_analysis.py init`
  (`clip_folder.py` has the rules of that folder). The builder refuses a job
  whose video has none, or whose file is no longer the one the folder was
  made from. Added 2026-10-10: the folder holds the video's identity and the
  copy jobs render from, and nothing measured yet; the trackers, the capture
  and the settings of a job are still made per job, as below.
- **The stages, in order, each a `build` and a `queue`:** `preview` (who is
  where: one tracker a subject, each told who the earlier ones hold, parts,
  class maps, pose, mesh; nothing sampled), then `collect`, which fails by
  name when a file is missing or a subject was found in no shot and prints
  the capture's `files` command; the capture and its gate
  (`bench/capture_masked_run.py`); `plan` a load (the song node says what
  each window will be given), which the capture reads; `render` a load,
  which is refused unless the capture passes `verify` again, its gate is
  clear, and the gate read the node's own plan for that load.
- **Nothing an earlier run made reaches a load by a key** (AGENTS.md, the
  operative rule of that name; added 2026-10-10). A load's masks are files
  of the capture the job file names, read by path; its mesh is the file its
  own job's preview record lists, or the one the load names (`motion_file`);
  the graph it builds reuses nothing an earlier run made (the kept-mask
  store and stored-window reuse were removed on 2026-10-10, with their
  inputs). The analysis folder is found from a file's name and then held to
  that file's size and head hash: an identity, not a store.
- **A load made again is a new job name on the same capture.** The tool
  never reuses a name, and a capture can serve any number of jobs: the run
  names inside it are the loads' and are new each time. Added 2026-10-10,
  after two face loads were pulled and re-made that way.
- **A load is one subject on one shot**, written in the capture's words, with
  its text given to it. It is no shorter than the song node's shortest extent
  (`bench/masked_job.py::shortest_load` reads it from the node): a shorter
  stretch is left as the source, with why, or joined to a neighbour. What a load carries by default, and why each value,
  is the `MASKED_JOB_*` block of `workflows/h3_config.py`; a value measured
  on one clip is a starting value, a job file may set any of them, and a
  load's build record says which were set by the job and what the capture
  measured on this clip.
- **A load's text is data of the job, tied to the frames it was written
  from** (the owner, 2026-10-10; AGENTS.md, the operative rules on a scene's
  text). In the job file a text is a file beside it, the source frames it
  was written from and who wrote it. A text serves the loads inside those
  frames and is of one shot, so one text for loads of different shots is
  refused, by name, and so are the same words under two labels. `build`
  prints each load's text in full with where it was written from. A load
  that holds a cut cannot be built for now: one load a shot, each with its
  own text. That last part is an edge accepted, not a rule of the model
  (`bench/masked_job.py::ONE_SHOT`). *2026-10-13: until that day this
  bullet ended there. A load may now hold ONE cut when its job block names
  `two_shots`, and only then; the section "Two neighbouring shots in one
  load: a tool, not a default", below, has it.*
- **A load of two subjects in one pass** (2026-10-12, the owner's "yes,
  build it next" on Open List row z09; a load's `subjects`, a list of two).
  Each entry is one subject's own: its label, its `region` in the plan's
  words (`face`, `face and hair`, `whole`), its part where it takes one,
  its still, its reference video with the frame it begins on and its short
  edge, and the keep-outs it needs. The order of the list is the order the
  model is given them: the first subject's still is `<Picture 1>` and its
  reference `<Video 1>`, the second's `<Picture 2>` and `<Video 2>`
  (`workflows/masked_job_graph.py::one_subject_view`). The builder
  makes the tracked mask the union of the two tracks and the region the
  union of the two parts, joins both subjects' keep-outs with neither
  subject ever among them, chains the two stills and wires the two
  references to the Masked Source's two inputs. Every file is a path the
  load gives; the build record says which picture and which video went to
  which subject, and the shown lines say which of the six labels the text
  names. **What its text must name:** `<Subject 1>` and `<Subject 2>`, one
  definition each, the first saying it is the person in `<Picture 1>` and
  the second the person in `<Picture 2>` (until 2026-10-15 this also asked
  for where each of them is in the picture: that is the frozen picture, see
  step 2 of "The frame every masked text is written in"); and, for each subject that has a reference, `<Video 1>` or
  `<Video 2>`, with the sentence that says what that video is and the line
  that says what is taken from it, in the vendor guide's form
  ([`../prompting.md`](../prompting.md), section 9). The numbers follow the
  list's order and nothing else. The builder writes none of it and refuses
  none of it: it prints yes or NO for each label. What a reader has to
  know before the first one renders:
  - **The text alone tells the model which still is whose.** A reference
    still is tokens and rows the model attends to from everywhere; nothing
    in H3 or in this pack ties one to a place in the picture, with one
    region or with two. What there is: the text's two subject definitions,
    each paired with its own still and its own movement video, and each
    movement video made from its own subject's body. Whether H3 puts
    each reference in the right hole is not known until a render shows it:
    a swap or a blend of the two faces is what the first render of such a
    load is read for.
  - **One margin for both**, the load's `margin`: the node grows its
    region by one number. A subject whose own loads ran at a smaller margin
    carries the wider ring here (the lead's face at the second subject's
    margin, on the first use). A second part and margin in the node would
    end that; it is node code and is not built.
  - **The capture reads it as a load of both** (since 2026-10-12; until
    then it was the first subject's, the rule about a region over another
    subject fired on the second and was answered by a person). No subject
    of the load is another subject to it, and a rule that is a fact about a
    subject is read once for each. A few rules still read the first subject
    only, named in [`next_steps.md`](next_steps.md): they read the one mask
    the node is given and cannot say whose a missing part is. Put first the
    subject whose region is larger: a hair band and those rules are
    the first subject's. A capture must be built with the tool as it is now
    for this to apply. *Since 2026-10-12 the
    capture knows this much of the pair:* a plan it works out itself, before
    the node's plan is read, is worked out from both subjects' parts as one
    mask (until then it was the first subject's alone, and a look before
    the plan showed nothing of what is replaced on the second); the run
    records both subjects; and the look's strip (`video --load`) takes such
    a load, says under the picture which `<Subject N>`, `<Picture N>` and
    `<Video N>` is whose in the list's order, shows the two reference
    videos side by side each under its label (`--motion` twice), and fills
    each subject's part in that subject's colour. That binding is in the
    text alone, so the look is where a crossed pair is caught.
  - A hair band is the first subject's. A two-subject load is not rendered
    zoomed in.
  No render had been made of one when this was written.
- **A hair band under a head-and-hair part** (2026-10-12; a load's
  `hair_band`, a share of the part's height, with no default: a load that
  does not ask has none). A still whose hair falls past the original's jaw
  is drawn into the margin under the part, and there every cell was the
  clothing keep-out's to give back; that mask's pieces change frame to
  frame, so the region's lower edge moved and the hair was drawn where it
  reached and cut where it did not (the long kitchen shot, source frames
  693 to 703, on playback). With the key the builder writes a copy of that
  keep-out with a band cut out under the part and wires the copy
  (`bench/masked_job.py::hair_band`): the part's box, each side the median
  over neighbouring frames so the edge holds still, a little wider, as deep
  as the share asked. It adds nothing to the region; the margin still
  bounds what is regenerated, and the other keep-outs still give their
  cells back inside the band. The constants and their provenance are
  `h3_config.MASKED_JOB_HAIR_BAND`; the copy is a file of the build, named
  in the build record with the band's figures, and the capture's own mask
  is not touched. **How it was taken in:** not tested on that shot by
  itself (the owner, 2026-10-12: not worth testing that scene alone). It
  goes into her face-and-hair loads from here and is read on the first of
  them that renders, by a session that did not build it. No render had been
  made with a band when this was written, and this page says nothing of
  what one does. *2026-10-12: `bench/read_hair_band.py` prints a band as
  figures a frame, for the band load and a pair without it, from the plans
  alone or with the two renders; it says what the pair differs in besides
  the band and gives no verdict.*
- **Somebody looks before a plan and before a render** (the owner,
  2026-10-10: a passing check never says the masks are on the right person
  or the text is of these frames), **and the look is shown, never refused**
  (the owner, 2026-10-12, Open List row c22). `build plan` and `queue` of a
  render record the look the capture's `looks.json` holds for the load,
  later than the capture was last written, that showed this load's frames
  with this text (`bench/masked_job.py::looked_at`), or that none is
  recorded, with why and the strip's path (`look_or_shown`), printed with
  the build's lines and before the queue; until 2026-10-12 they refused the
  load instead. The strip with the whole text beside the frames, and the
  record of the look, are `bench/capture_masked_run.py` `video` and
  `looked`. The capture is written again when it reads the plans, so a job
  is looked at twice.
- **Not built, added 2026-10-10:** the one-window patch (until it is,
  `bench/patch_render_window.py build` and the session script the masking
  board's `note-do-not-build-on-these-2026-10-10` names are the only route,
  and neither is checked); the analysis folder's own passes, its
  `profile.json` and the rules that set a load's values from the clip
  (the masking board, `build-per-video-analysis-folder`); a subject's
  `seeds` in a job file (the tracker has the input, section 1 below); a
  cheap variant of a built load for a test that changes one thing, and a
  load over several shots with a text that describes each shot (both on
  the same card; *2026-10-13: the second is built for two neighbouring
  shots, in the section below*). *2026-10-12: the first of those two is half built.
  `bench/sweep_loads.py` writes copies of job files with one named change
  (a subject's still, a text, a load's motion video, or one setting of
  every load) and new names, proves each copy differs from its source in
  that and the names only, records the sweep and prints the builder's
  commands in order; `bench/check_sweep_loads.py` holds it. It builds and
  queues nothing, so a copy still goes through plan, the capture, the look
  and the gate like any load: it is not cheaper to run, only harder to get
  wrong.*

## Two neighbouring shots in one load: a tool, not a default

Built 2026-10-13 at the owner's word ([`decisions.md`](decisions.md)). One
load a shot is still what every tool makes: a join at a cut costs nothing,
a cut inside a latent step is a known fault ("A latent step is several
frames", below), and a long window costs more than two short ones. This is
the other thing, reached for on a person's word for one pair of shots.

**What it could buy**, in the owner's words: one face for a subject across
a cut, where two shots of them sit side by side; and a tough shot helped by
an easier one beside it. Neither is shown yet: no load had rendered this
way when this section was written.

**Why it may sit well with the model: the owner's account** (2026-10-13,
to the lead; their reasoning, not something measured here). H3 does well
with shots because it expects several of them inside its longest length,
and each new shot breaks up the drift a long single take decays into. That
is their reason a window fitted to a short shot is no worse than a long
one, and a reason a window of two shots may suit the model. What was
measured that day bears on the first half only: one short shot rendered at
a long window and again at its fitted one with nothing else changed
(`lead_fp_1920c` and `lead_fp_1920d` on the review page, one pair, one
seed, measured by the lead's session script and watched by the owner, who
found the fitted one at least as good). Nothing measured says anything yet
of two shots in one window.

**The opt-in.** A job block's `two_shots`: the source frame the second shot
begins on, the two shots, why they share a window and who said so
(`workflows/masked_job_graph.py::two_shots_of`). Nothing sets it for
anybody. Without it the builder refuses a load that holds a cut, as before.
With it:

- the load is held to one subject, one window that holds both shots whole,
  the named cut inside it, not zoomed and not a continuation
  (`masked_job_graph.check_load`);
- its text may hold that one cut and no other, as the capture's own shot
  tables have them, and has to carry a `[Shot 1]` and a `[Shot 2]` and no
  third (`bench/masked_job.py::held_as_two_shots`). The headers are held by
  presence. Whether each shot's sentences are true of its frames is a
  reader's to see on the strip, as for any text;
- the build's record carries the key and its printed lines say TWO SHOTS IN
  ONE WINDOW, with whose word it is.

The builder holds the key to itself and to the capture's shot tables. It
does not read the shots record: `bench/two_shot_load.py` does, when it
writes the job (`shots_of`). A job file written by hand can therefore name
a cut the record does not have; the key is a person's word.

**What it is made from: the two shots' own ordinary loads**
(`bench/two_shot_load.py job --a A.json --b B.json`). Nothing is tracked or
posed again. Each preview's track, part and class-map videos are laid end
to end, lossless, into files of a new name, and every frame of every joined
file is held equal to its source's by its decoded hash before anything is
built on it. The shot tables and pose tables are joined the same way, the
second's frame numbers moved on, and the joined table holds the cut where
the two meet. The two head references are laid end to end too, each side
with its own box and scale, so the reference's framing jumps at the cut as
the source does. The joined masks go into a folder shaped as a preview's
job folder and said in both its files to be joined, so
`bench/load_chain.py` reads it by name like any preview and every step of
the chain runs unchanged. The reason for making it this way and not with a
tracker that crosses the cut: a pair made both ways then differs in the
window and in no mask. One step of the chain is not for it, BY PRACTICE
AND NOT BY CODE: `no-look` records that no look is owed for an ordinary
load, on its maker's statement that the load is not the first of its kind
(`--not-first`), and a two-shot load is looked at. Nothing in
`bench/load_chain.py` refuses `no-look` on a job that names `two_shots`; a
reviewer found that on 2026-10-13 and the refusal is not built.

**What the two loads must be.** The same subject, source, canvas, still,
settings and reference form; on two neighbouring shots of the shots
record; and meeting at the cut, the first ending on the frame before it and
the second beginning on it, so that the one load replaces exactly the
frames the two replace. The tool's docstring lists what it refuses and
what is not built.

**The cut and the latent steps.** A step's grid counts from the load's
first frame, so a cut falls inside a step or on its edge by arithmetic
alone. The tool prints it before anything is built on the job, with the
planner's own function (`loop_plan.split_steps`) on the frames the
subject's part is on: the step that holds the cut, which of the capture's
two flags that will be (`region_carried_across_a_cut` blocks the gate,
`region_shared_across_a_cut` does not), and the first frames, from the
load's own and one cycle on, that put the cut on a step's edge. Moving the
first load's first frame onto one of those is the lever; the frames before
it are left as the source and said so in `left_out`.

**The node finds the cut itself.** The builder wires no shot table to the
Masked Source, so the node reads a load's cuts from the picture
(`video_mask.source_cuts`) and gates the composite by them (`cut_gate`,
`unlaid_frames`). `bench/two_shot_load.py plan-cuts JOB.json`, run after
the chain's `plan`, prints the cuts in the node's own plan beside the
job's, and exits 1 when they are not the same list. A load whose node
missed its cut is not rendered as it stands.

**What it costs.** One window as long as both shots together, where the
ordinary way samples each shot at its own fitted length; attention grows
faster than the frames, so a step of the long window costs more than a
step of each short one added together. The seconds a step are the
server's own, read from its log (`GET /internal/logs/raw`), never a queue's
wall time: `land` and `reference` free the card, and what is queued next
starts cold. The figures of the first pair made both ways go in its record
under `bench/results/` when it has rendered, and this paragraph will point
at it: seconds a step for each of the three loads, the sequence length the
log prints for each, and whether the card held the window.

**Not known.** What the model draws in a step that holds frames of two
shots. What it makes of a reference whose framing jumps at the cut.
Whether the face is more one person across the cut than in two loads from
the same still, which is what the first pair is made to show. Whether a
real preview's tables, joined, read through the capture without fault: the
check holds the join on made tables, and no real load has run.

## Seeing what the tracker and the masks did

Written 2026-10-10, after a session had to be told this tooling existed. It
is in four places and none of them is one command. Read this before writing
a probe.

**What a render will be given, before it renders, and what it was given,
after.** The song node writes both itself, into `<prefix>_windows/`
(2026-10-10). A render leaves `<name>_window_N_region.npz` beside each
window's latent: the fitted mask, the token region per latent step, the
margin, the window's first frame, its trim and the composite's settings and
cuts (`video_mask.save_window_region`, read with `load_window_region`). A
PREVIEW of the same graph (`preview` on, a source wired) samples nothing and
writes `<name>_window_N_planned_region.npz`, the same file made by the same
calls, and `<name>_plan.json`: each window's frames, what it writes, the
latent steps a cut splits and the frames the composite will leave as the
source (`audio_freeze_song.write_plan`). Neither is read by a node. A plan
is replaced by the next preview and left alone by a render, so the two sit
side by side. What a plan cannot hold is what `only what changed` will
keep; that needs the decode, and `bench/recomposite_window.py` lays a saved
window again from its stored latent and its region.

**One command for the masks of a run, any number of people**:
`bench/capture_masked_run.py files` reads the mask videos a preview saved and
a render's own review, and writes `data/<date>_<name>/` with a `subject` on
every row and file, the rows across subjects (`frames.csv`: each pair's
overlap, a run's region on another subject, the margin cells given back) and
a README of what it does not hold; `preflight` lists what is at risk before a
render; `video` draws the stacked, frame-numbered picture from those same
files. Its docstring is the schema; `bench/check_capture_masked_run.py` is its
check; what is still missing is in `data/CAPTURE_GAPS.md`. What the models
behind the lane's signals do and do not give (a body mesh has no mouth; no
model here scores lip sync) is in
[`meta_perception_models.md`](meta_perception_models.md).

**One command for the file that gets watched**: `bench/assemble_delivery.py`
takes a table of renders by source frame range and writes one file at the
source's own rate with the source's own audio packets, the frames no render
covers taken from the original, and passes over the same frames merged by
what each changed. It proves by decode that no frame was dropped, doubled or
moved and that the audio is the source's, and it raises its own flags from
the per-frame table of what each render changed. Its docstring is the
account of the table's rows, of each proof and of each flag, and
`bench/check_assemble_delivery.py` is its check. What each proof shows and
does not, the files built with the tool and every change to it by date:
[`bench/results/2026-10-10_delivery_files_and_their_records.md`](../../bench/results/2026-10-10_delivery_files_and_their_records.md).
*Cut on 2026-10-12 from a paragraph that restated the docstring flag by
flag.* What a session needs from it here:

- Cut a render's rows on the source's cuts.
- Given capture folders (`--capture`) it settles shared pixels by whose
  they are, and where two tracked masks claim the same pixel (an arm
  reaching across somebody) the capture's `owners.npz` says whose it is. A
  patch has no run in a capture, so its row says whose it is
  (`subject=<label>`).
- A row can give a class of a subject back to the source
  (`restore=<subject>.<Class>`, read from the capture's class map): a thing
  inside a region that should not have been redrawn, an earring inside a
  face, is put back here and not by asking the sampler to keep it, which
  changes what the sampler draws beside it. With no class
  (`restore=<subject>`) a row gives back the whole of another subject, and
  `restore=<subject>:whole` takes nothing out.
- A shot left as the source on purpose is written in the table, as a
  `source first-last subject=<label> why` line.
- With `--size source` the file is at the source's own size: every frame is
  the source's picture, never scaled, and only what a render changed is
  scaled up and put back over it.
- `--compare A.check.json B.check.json` prints two records side by side:
  the same stretch with and without a restore is the before and after a "by
  class" question asks for.
- Judge a render's colour on this file: it is BT.709 and says so, as
  renders are since the song node's writer converts and tags, and a render
  written before that plays off in a player that guesses the matrix from
  the size.
- A file listed in its folder's `LOCKED.md` is never built over.
- How a region SITS in its plate (its detail, grain, tone and cast against
  the plate just outside it and against what stood there before) is
  `bench/region_against_plate.py`, which needs no mask either; run it
  before tuning a look by eye.

**What every masked render already wrote beside itself** (the song node,
`audio_freeze_song.py`):

- `<prefix>_NNNNN_shots.json` and `.md`: the Subject Track's shot table
  (`shot_table.py`, whose docstring is the schema). Per shot: its frames, the
  people found on the frame shown with a number each, who was taken and why,
  the frames with and without the subject, each loss and what was looked at
  after it. A person's number is theirs on that shot's shown frame only; it
  is not an id that holds across shots.
- `<prefix>_NNNNN_with_mask.mp4`: the render over the source with the
  regenerated region coloured, drawn from the token mask the sampler was
  given ("The song node" below).
- The song node's report, in `/history` and on the graph's text preview: the
  window plan, what share of each window regenerated, the Masked Source's
  warnings about a part mask it doubts, the seconds by stage.

**The look before a render**, nothing sampled: the review graph ("What it
is"). To keep the masks themselves, wire a mask-to-image node and a video
save node onto the Subject Track's `mask`, the Subject Parts' mask and the
Masked Source's `mask`: no shipped graph does, and a session's own preview
graph is the usual way. `MiniMaxH3SaveShotTable` saves the table and the
numbered tiles.

**The inside of the tracker** (each tracker call, where it stopped, every
look after a loss with its candidates): `bench/subject_regain_looks.py`,
whose docstring has the commands (`run`, `render`, `replay`). A separate
process on the card, outside the server. It takes one of the pick rules and
cannot yet follow a person named by a frame or a correction.

**A finished render against its source**:
`bench/masked_render_against_source.py` (`align`, `timing`, `flicker`,
`landing`; run `align` first), `bench/measure_subject_motion.py`,
`bench/measure_subject_yaw.py`.

**The captures of 2026-10-07** are under `data/` (untracked), each folder
with a README that gives its layout: per-frame tables of the track, the
part and the region, and every node's report, for several windows. The
scripts that made them were a session's own and were not kept, so the
folders are a format to copy, not a tool to run.

**What does not exist**: a trace the workflow writes itself
([`next_steps.md`](next_steps.md), "Asked for and not done"), and any id for
a person that holds across shots or across nodes. **One node follows one
person** (`subject_track.py`'s docstring; the tracker is seeded with one
mask). Two people are two Subject Track nodes in one graph, which share
nothing, each with its own table; the second person's mask goes to the
Masked Source's `others` when the two stand close. *Until 2026-10-12 this
paragraph ended "None of the captures or tools above carries a field for
which person a file is about", written before the capture tool had one:
`bench/capture_masked_run.py files` puts a `subject` on every row and file
(its docstring), and a delivery table's row can say whose it is. The node's
own files, the shot table and the mask review, still carry none.*

## Who is who: a session reads first, the owner answers, and both are kept

A tracker follows whoever it was started on. Started on the largest person
of a frame it is guessing, and the first batch of guesses kept against the
owner's answers (`data/<video>/picks_against_answers.jsonl`, the table
beside it) shows how often the guess was somebody else. So nothing is built
on a preview until a person has said who its tracker holds, and what each
reader thought is kept, so the next batch can be read against this one.
The steps, each with the tool that holds it:

1. **A session that did not build the preview reads its tiles** and writes
   one line a preview: the picked person's number, what they look like,
   whether that is the right person, why, and `thinks_it_is`, the number of
   the person the reader thinks IS the subject (`bench/who_page.py`'s
   docstring has the fields; the line goes in the batch's notes file).
2. **The who page shows that reading as a guess, closed, under the tile**
   (`bench/who_page.py --notes`), and the owner clicks. Their typed words
   are read with their clicks, whole, and reported beside what was stored;
   a click is never read without them.
3. **The cast run stores the owner's answers and writes the picks batch in
   the same run** (`bench/page_answers_to_cast.py --picks`; refused without
   it). The owner's answer is the reference. A shot their own words doubt
   is marked unsure by a person, with the lines of those words, and is
   counted apart (`--owner-unsure`).
   A shot the record is `unsure` of for a subject (a cut the owner has
   not answered on) is planned by nothing and is asked about: its preview
   is made to ask (`bench/shot_to_load.py preview --ask SUBJECT:why`), the
   owner is asked once, on its numbered picture, and no load is built on
   it until they have answered.
4. **Each batch is read against the last**: counts, never a rate from one
   batch, with what the table cannot see
   (`bench/picks_against_answers.py`'s docstring).
5. **Before the owner is asked anything, what they already answered is
   read**, and only the missing fact is asked, with why the earlier answer
   does not cover it. The cast file is the record
   (`data/<video>/cast/cast.jsonl`, each line with its frame, its box and
   the answers lines it came from), and the who page prints the answer
   that already stands for a subject above that subject's buttons. An
   example of the rule doing its work, 2026-10-13: the owner asked whether
   the small figure at the back of the hall shot is the second subject;
   the cast file held their own answer of 2026-10-11 saying so on two
   frames of that shot, and the capture's tracker was seeded from it. A
   session answers that from the record and asks nothing.
6. **A shot the tracker cut into several is shown cut by cut**, each cut
   as long as the shortest load on its own frame, with every person's box
   in a color of their own (`bench/who_page.py --every-cut-of`): one frame
   of a shot of many cuts answers for that cut and no other.
   The answers then go to the tracker a cut each: a preview of that shot
   made again takes the tracker's cuts from the table its earlier preview
   saved (`bench/shot_to_load.py preview --seed-cuts-of`), and each cut
   that holds a box gets its own seed. Without the cuts a shot gets one
   seed, the box latest in the cast file, and every other cut is matched
   against how the subject looks on that one frame.

**What waits on the owner's own word.** "The owner is asked only the
leftovers" is not the practice. Every box in the cast file is the owner's
answer, so every subject of every tile still takes their click. The day
they say a session's reading may stand as a seed, the page asks only where
a session could not tell, where the subject is not outlined, or where the
owner's own words doubt the shot, and the cast file takes a session's box
marked as a session's and replaced by the owner's click. Until then a
session's reading seeds nothing.

## The pieces, in the order a render meets them

### 1. `MiniMaxH3SubjectTrack` (`subject_track.py`): which pixels are the person

What to type as a SAM phrase, and why a description does not pick one person
out: [`sam3_prompting.md`](sam3_prompting.md).

One mask per frame, empty where the person is not on screen. The module
docstring is the authority and lists the steps; `follow` is the function.

- **Cuts.** `cut_scores` is one minus the correlation of consecutive frames'
  gradient maps at `CUT_SIZE`: a cut moves the edges, a lighting change does
  not. The threshold is the user's value, or `auto_cuts`: the middle of the
  widest gap in the clip's own scores above `CUT_FLOOR`.
- **People.** Each shot is looked at `PROBE_OFFSET` frames in. Core's
  detector is asked for `subject_phrase` with a count (`counted`), because
  core returns one detection per phrase without one.
- **The pick.** `pick` is a rule (largest, most central, best match for the
  phrase); the shipped default is `most central` since 2026-10-07
  (`h3_config.SUBJECT_TRACK`, `decisions.md`), and it was `largest` for
  every render before that date. On a named frame it is applied there. Left automatic,
  `main_subject` takes the person the rule favours for most of the clip's
  frames. **How each rule measures** (`subject_track.choose`): `largest` is
  the mask with the most pixels; `most central` is the mask whose centre of
  mass is nearest the frame's centre, in the frame's own proportions since
  2026-10-07 (before that each axis was scaled alike, which measures a wide
  frame as if it were square). **No rule is right everywhere**: on one clip
  each of the two was right on one window and wrong on another, and on a
  window that opens with the subject small the automatic pick named
  somebody else under both
  (`../../bench/results/2026-10-07_subject_track_calls_on_masks.md`;
  `../../bench/results/2026-10-07_masked_switch_keep_prompt_verdicts.md`,
  section 5). The pick is a first guess for each shot; the preview tile
  shows who was taken and a correction settles it.
- **The match.** A person in another shot is compared with the subject in
  two places, and the lower counts: the trunk's features under the top third
  of their mask (`top_third`, `signature`), and under the head SAM 3 finds
  for `head_phrase` inside their mask (`head_of`). When the pick frame shows
  other people, their average is subtracted first (`relative`). A person
  with no head is no match. The line is the user's value or `auto_match`.
- **A clip with one person.** With nobody else on the pick frame and the
  match automatic, a shot under the line is probed to its end and its best
  frame showing one person with a head is taken, whatever it scores.
- **Tracking.** Each shot is tracked forward and backward from its seed with
  core's `SAM3_VideoTrack` on its `initial_mask` path, no text prompt.
- **Corrections.** `corrections` overrides one shot at a time, after the
  automatic pass and on the frame that shot's tile shows: a person by the
  number on their outline, or nobody (`parse_corrections`, `_correct`). The
  other shots and the frames the tiles show do not move. A corrected shot
  is followed from the person named and, since 2026-10-07, looked for again
  like any other shot when the tracker lets go of them (the next item);
  before that it was tracked once and left empty from the frame of the
  loss. A correction says who the subject is, not that the tracker will
  hold them.
- **A subject let go inside a shot.** A tracker call is seeded once, so a
  subject the tracker let go was lost for the rest of the shot.
  A run of frames the track leaves empty is probed every `PROBE_STRIDE`
  frames from the side that is tracked, and the track is seeded again on
  the first frame that shows the subject (`regain`). Who that is, is
  decided against a gallery of the shot's own tracked frames
  (`gallery_frames`, `gallery_scores`, `clear_best`), never by being the
  only detection; `REGAIN_SAME` and `REGAIN_MARGIN` in `subject_track.py`
  carry their provenance. Frames that stay empty are in the report and in
  the shot table's `frames_without_subject`. A shot corrected by hand is
  searched the same way, with the corrected track's own frames as the
  gallery (since 2026-10-07, the item above).
- **The same person as an earlier run.** The shot table carries the picked
  shot's gallery. `subject_from` on the Subject Track takes an earlier
  run's `shot_table` output, or the path of the `..._shots.json` it saved,
  and the pick on this load is then made against that gallery and not by
  `pick`. If nobody on any frame looked at is that person, nothing is
  picked and the report says so. For a long clip rendered in pieces
  (`bench/join_stretches.py`). Its limit: the score falls when the person
  is much smaller or larger than the gallery shows them
  (`bench/results/2026-10-06_subject_track_regain_and_handover.md`).
- **What it shows.** One labelled tile per shot and a text report: who was
  taken, who was the best candidate where nobody was, every score with the
  line marked, every phrase and threshold used.
- **One detection a person** (added 2026-10-10). The detector can return
  one person several times, as masks nested in each other; they used to be
  numbered as several people with their labels drawn on top of each other.
  They are joined before anyone is numbered, picked or compared
  (`subject_tracks.py::one_each`), unless the one that would be kept may be
  a part, and the report, the table and the tile's header count both. A
  correction by number is read off a tile made after this change.
- **A seed** (added 2026-10-10). `seeds` says who the subject is in a
  shot by a frame of the load and a box, one line a shot
  (`frame 410: 543, 253, 157, 186`, or `frame 410: none`), so a tool can
  write what until now only a person reading this tracker's tile could
  (`subject_track.py::parse_seeds`). The detection on that frame the box
  names is taken and tracked; a box nothing agrees with leaves the shot
  empty and says so, with the nearest detection's box
  (`subject_track.py::SEED_AGREES`, a starting value). A seed on the
  picked shot is the pick. The frame is the load's own: whoever writes a
  seed subtracts the load's first source frame.
- **Who is who after a cut** (added 2026-10-10, a lane in progress). Each
  tracker decides a shot alone, against one reference and a line. On a
  two-subject load both left a shot empty that held both subjects, and the
  trackers' own galleries show why a line cannot do it there and a
  comparison between the named subjects can:
  [`2026-10-10_who_is_who_across_shots.md`](../../bench/results/2026-10-10_who_is_who_across_shots.md).
  The shot table now keeps what a refused shot was judged on
  (`shot_table.py::LOOKS_ARE`, `SIGNATURES_ARE`), and the report says when
  the automatic line was put in a gap between two shots
  (`subject_track.py::moved_by_a_gap`). After a preview,
  `bench/who_is_who_across_shots.py corrections` reads the trackers' tables
  of one load and writes one file for the gate, a row per shot: `answered`
  when every tracker's call on it is a person's word (a pick, a typed
  correction, a seed) or nothing with a head was detected, `ask` otherwise,
  with the question in the tile's own numbers. It compares nobody with
  anybody and decides nothing. *2026-10-12: tables of two previews (a
  load of two subjects whose tracks come from different previews) are cut
  to the load's frames with `--frames`, and refused by name when a table
  does not cover them or the cuts inside them differ; a person's number
  is then each preview's own detection of a tile's frame, which the
  file says.* *2026-10-10: the sum over all the subjects
  at once, which this tool carried for a few hours and which this bullet
  described, was removed the day it was built (`docs/wiki/decisions.md`, 2026-10-10, "What the owner cut from the tools around the nodes"): it was the
  tracker's own likeness arranged again, not a second vote, and it handed a
  stranger to a subject who was away.*

Why not core's tracker with a text prompt: across a cut it starts new
objects, it has an object cap that detection stops at for good, and a
confident detection overwrites the tracked mask. The module docstring cites
where in core.

### 2. `MiniMaxH3MaskedSource` (`video_mask.py`): what happens to those pixels

- **The region.** The mask is grown by `grow_pixels` and turned into a token
  mask (`grow`, `token_mask`). A token is regenerated or kept whole, in
  space and in time: the video VAE packs frames in runs (`run_lengths`), so
  a token covers several frames.
- **`edge`** (optional, 2026-10-09; a trial, off by default). `whole tokens` is the
  rule above. `latent cells` hands the sampler the mask per latent cell, half a
  token's side: core labels a token by the most regenerated of its cells and
  puts the source back cell by cell, so a kept cell inside a regenerated token
  is the source's in the result (`video_mask.EDGES` has the provenance). It is
  aimed at the rounding that keeps the region near twice a small subject's own
  area with no margin at all. `bench/check_video_mask.py` item 15 holds that
  it loses no subject pixel and moves no label. *Until 2026-10-12 this said
  nothing had rendered with it, which was true when it was written on
  2026-10-09. The kitchen shot's picked render carries it (the graph saved
  beside its job, named under "How it was made"), and the built graphs that do are listed by
  `grep -l '"edge": "latent cells"' data/*_job/*.render.json`. No render with it has been read beside the same render without it, so
  what it changes in a picture is still not known.*
- **`keep`** (optional, 2026-10-07). A second mask, from any node, of what
  must stay the original even inside the region: something the subject
  holds, a person standing close, anything passing in front. The part node's
  `held` output (`sapiens2_parts.py`) is a mask of what the subject holds,
  made without a name: inside their mask, not body or clothing to the part
  model, near the lips or a hand. `keep` is taken out
  of the token mask after the grow (`window`), so the margin cannot run back
  over it, and in whole tokens: a token that holds any kept pixel on any of
  its frames is kept, so a little of what surrounds a small object stays
  too. The model is given those tokens clean, as it is the rest of the
  plate, and draws around them; the composite shows the source there.
  Refused together with `paint_out` or a softened start, which change the
  pixels under the subject. Unwired, nothing changes. The owner's first use:
  a prop the original holds is otherwise under the noise and comes back as
  whatever the model guesses. Rendered once, 2026-10-07, with a mask from a
  phrase that SAM held for the window's first seconds only; not judged.
  **Dated note, 2026-10-10: do not `keep` something that lies inside or
  against the part being replaced.** A `keep` of the lead's jewellery class
  inside a face-only region held the jewellery and, because a kept token is
  whole, also held the skin beside it; on the pixels the sampler was free to
  redraw, the face then moved back toward the original's over the window
  (one clip, one lead, one window, one seed; the masking board,
  `find-mrpop-keep-inside-a-face-brings-the-original-back`, names the
  measure). It is the head-and-hair result again: real pixels of the
  original next to the hole. Put such a thing back at assembly instead,
  where the sampler never sees it.
- **`replace`.** `whole subject`, or `head and hair`, which keeps the body's
  pixels and finds the part with SAM 3 from `part_phrases`.
- **`motion_video`** (optional, 2026-10-10) with `motion_reference` on
  `a video I wire` (`video_mask.MOTION_WIRED`): a video that runs beside the
  source frame for frame and is shown to the model as the movement, in place
  of anything cut from the source: a body mesh of the original, a pose, a map
  of where a mouth opens. It says how the subject moves without showing the
  original and without the text saying it. Each window is shown its own
  frames of it (`wired_motion`), which a reference video appended to the
  chain is not: that one is cut from frame zero for every window. With
  `motion_vae` on, the video model has its own copy at every frame; off, the
  text encoder sees two frames a second. The prompt has to say what
  `<Video 1>` is and what is taken from it. A body mesh for it is made from
  the track with `MiniMaxH3SubjectBoxes` (`subject_boxes.py`): one box a
  frame round the tracked mask, wired to the body model's box input, so the
  mesh is of the tracked person and the model's crop is of them and not of
  the whole frame. Rendered as an appended video on
  one window before this input existed, with a body mesh from core's SAM 3D
  Body nodes and the video model's copy: on that one shot and seed the
  action was drawn with no action words, and things inside the region came
  back as other things until one sentence named them (the masking board,
  `route-motion-signal-not-words`). What that body model gives and does not
  (no mouth; one whole-frame crop when given no box) is in
  [`meta_perception_models.md`](meta_perception_models.md).
  *2026-10-11: a body model with the source's eye openings drawn on its
  face, and the text that goes with it, is the section "The motion video as
  a body model, with marks drawn on its face" below.*
  *2026-10-12: a second wired video. `motion_video_2`, with its own
  `motion_short_edge_2`, is shown to the model as the next video reference
  after the first (`<Video 2>` when the chain holds no video of its own),
  for a load whose text gives each video something of its own to carry, as
  the vendor guide writes several references
  ([`../prompting.md`](../prompting.md), section 9). `motion_vae` is one
  switch for both, and a second with no first is refused. In a job it is
  the load's `motion_file_2`, with `motion_first_2` and `motion_edge_2`
  (`../../workflows/masked_job_graph.py`); the builder shows whether the
  load's text names `<Video 2>`, and the prompt node says above its text
  that it writes nothing about one. Built at the owner's word (Open List
  row 16) and not rendered with when this was written; it reaches the card
  at the server's next restart.*
- **A setting it will refuse is refused when the graph is queued**
  (2026-10-10): a feather wider than the margin, or a softened start with
  `paint_out` (`video_mask.settings_refusal`, called by the node's
  `validate_inputs` and again by `execute` for a value that arrives through
  a link). Until then these failed only when the node ran, behind whatever
  was ahead of it in the queue. A refusal that needs a tensor (`keep`,
  `others`) is still made at run time.
- **What is encoded.** The source frames themselves; with `paint_out`, a copy
  with the subject filled in from its surroundings (`fill_subject`).
- **The composite.** After the decode the source's pixels come back outside
  the region. `only what changed` (`changed_alpha`) keeps the render where
  it differs from the source or where the old subject stood, and restores
  the source in the rest of the margin, so the margin can be generous.
- **No kept mask.** "Nothing is kept between runs by a key", after the
  four pieces.

### 3. `MiniMaxH3MaskedPrompt` (`masked_prompt.py`): the prompt

The reference format is six sections of prose, and in this lane almost none
of it depends on the clip, because the plate holds the setting, the framing
and the cuts. The node writes the text from who the still shows (`subject`,
a few of the user's own words: `woman`, `man wearing a red cap`), whether
they are the voice on the track (`voice`), and what the still provides
(`picture_gives`). It works the rest out: the pronouns from `subject`, and
from the Masked Source wired into its `source` what is replaced and whether
a motion reference is on, in which case it writes the `<Video 1>` lines. So
the text and the Masked Source cannot disagree. `add_to_shot` adds
sentences to the shot as written. On every run the node shows one line per
input saying what it did with it, then the text.

**The text says nothing it cannot know** (the owner, 2026-10-07). The node
writes without seeing the clip, and H3 acts on words: a clause written to
cover a case ("turning when they turn", a description of how a mouth
moves) is read as an instruction. The upper-body role is in the vendor
guide's form at under half its earlier length: who the subject is in the
user's words, said in the definition, the retention line and the shot; the
scene as finished, not as an edit; movement as one relationship to
`<Video 1>`; the voice in one sentence; one shot paragraph, because the node
does not know which window of a clip a render takes. The head and
whole-person roles share the movement sentence and are otherwise as they
were (`decisions.md`, 2026-10-07; `masked_prompt_text.py`).

*2026-10-10: the rule was applied to all three roles, and "as they were"
no longer holds. An audit of the node's sentences
([`../../bench/results/2026-10-10_prompt_text_hardcoded_in_code.md`](../../bench/results/2026-10-10_prompt_text_hardcoded_in_code.md))
found two kinds the rule forbids and the owner ordered them fixed: what
the node cannot know of a clip (the look, how long the subject is there,
what else is in the scene, a shadow, the original's hair, what breath,
eyes, head and hands do), and talk of an edit (what stays, what is kept,
what is "the scene's own"). The whole-person and head roles now open
their shot with the upper-body role's sentence that the subject is in the
scene, and the soundscape is one general line shared with the reference
composer. The upper-body role, the one wording a person has accepted,
changed only in that sound line where a motion reference is wired: it
keeps four sentences of the kinds cut
from the others (the scene in its summary, the style opening, the light
and the camera) until a render with and without them is read
(`masked_prompt_text.py`, `HELD`). Neither cut role has rendered as it now
stands. [`decisions.md`](decisions.md), that date.*

*2026-10-11: the node's lines for a motion video take body movement from
"the person in `<Video 1>`" and say nothing of a facial expression. A load
whose motion video is a body model with marks on its face was given a text
of another form, written by a session and handed in as a file: the section
"The motion video as a body model, with marks drawn on its face", which
holds the lines and one whole text as an example. By the owner's ruling of
that day it stays a documented practice for now and is not put into the
node.*

**Say the least first, and add a sentence only when a render asks for it**
(the owner, 2026-10-10) is the rule above carried to a text written by hand
for one window. It is in "The frame every masked text is written in", near
the top of this page, since 2026-10-12.

- **Set `subject` to match the still.** The shipped graphs say "person"
  because their still is a placeholder. On the one pair rendered, the motion
  graph turned the subject with either word and started the turn later with
  "person" than with "man":
  `../../bench/results/2026-10-06_masked_v2v_person_text.md`. One window and
  one seed, read by the lane's metrics, not watched.
- **More words in `subject` are strong in both directions.** Where a text
  and a reference disagree the text wins ([`../prompting.md`](../prompting.md),
  "Silence is not neutral"), so words that match the still should hold a
  look the render drifts from, and words that do not match override the
  still. Say only what is in the still. Nothing beyond the three single
  words has rendered in this lane.
- **Where the sentences live.** Constants in `masked_prompt_text.py`, whose
  docstring says which have rendered and which have not. Changing one
  changes every graph that wires the node; `bench/check_masked_prompt.py
  --write` then rewrites the bank's copies.
- **Writing a prompt by hand** is still possible: type it into the song
  node's `prompt` and leave this node out.
- **`the wired parts`** can be any part of the subject, so the node asks
  for `picture_gives` to be set and writes nothing until it is.
- **A `preview` run of the song node tracks the subject** when this node's
  `source` is wired. `masked_prompt.py` says why. *Until 2026-10-12 this
  said "the first time; the mask is kept and the render that follows tracks
  nothing". No mask has been kept since 2026-10-10: whether a render tracks
  again is core's own node cache's to say (that file's docstring), and a
  job's render loads the mask videos its preview wrote.*
- **The graders read what the encoder reads**: `workflows/prompts.py::carriers`
  resolves the node's text from the graph.

### 4. The song node (`audio_freeze_song.py`): the render

Each window starts from the source's frames over its span
(`video_mask.window`), regenerates the masked tokens with the track's audio
frozen, and composites. A window whose mask is empty is written from the
source without sampling. One sampler per window is what carries the mask.

Beside the render it writes a **mask review**, `<prefix>_NNNNN_with_mask.mp4`
(`save_mask_review` on the song node, on by default): the render on top and,
below it, the source with what was regenerated coloured in, the same frames
under the same track. The lower row is drawn from the token mask the sampler
was given, so the grow and the snap to tokens are in the picture; its legend
names the mask for what the Masked Source's `replace` makes it (the tracked
subject, the head and hair, or the parts taken), since on a parts graph the
mask and the tracked subject are not the same thing. `video_mask.overlay_pieces`
and `window_layers` own the picture; `bench/check_video_mask.py` item 12 holds it.

**What a reference costs** is in the song node's report since 2026-10-12
(the owner asked, and nothing in a render's report said). For each window
it encodes, a line a reference: the tokens the text encoder is given for
it, which sit ahead of the prompt, and the rows of the video model's own
copy, or that it has none; then one line for all of them, beside the
prompt's tokens and the window's own rows
(`audio_freeze_song.reference_costs`). It is counted from sizes by
`reference_report.py`, the arithmetic the reference report node shows
before a run: nothing is encoded for it, and a count that fails is one
line and never a stopped render. `bench/check_video_mask.py` item 17 holds
it. **A plan prints the same lines** (later on 2026-10-12, after the first
plan on the new node printed none: the count sat in the render's
conditioning loop, which a preview never reaches). A preview now asks for
the reference chain as well as the source (`PREVIEW_ASKS`: still no model,
a chain being pictures from loaders) and counts each window's references
through the one function a render builds them with
(`audio_freeze_song.window_references`). A plan's count is made with no
text encoder loaded, so the prompt is not counted and the encoder's bounds
on a picture's size are core's defaults where a render reads the loaded
encoder's own; its last line says so, and the render's lines are the ones
to keep. A window whose references cannot be counted says so in one line
and the plan stands.

## Nothing is kept between runs by a key

*A section of its own since 2026-10-12; until then these two paragraphs
stood under the Masked Source, below a bullet "No kept mask" that said the
same in short: until 2026-10-10 `mask_store.py` kept the finished mask on
disk under a key (`reuse_mask`), and on a hit the tracker did not run; it
is removed, a mask being temporary data made by the preprocessing step for
the job that reads it.*

**Nothing is kept between runs by a key (2026-10-10).** The owner's rule:
nothing an earlier run made is reused by a key; a result is made again, or
handed in explicitly as a named file or a wired input (`AGENTS.md`,
"Operative rules"; [`decisions.md`](decisions.md), 2026-10-10). What this
lane had, and what became of each:

| what | where it was | what it was keyed on | now |
|---|---|---|---|
| the finished mask | `output/masks/` (`mask_store.py`, `reuse_mask`) | the queued graph, a sample of the frames, the files' stats, a hand-bumped number per node | REMOVED with its input, module and check. The key could not see a change to the code that makes a mask; a stale mask hid a fix on 2026-10-07 and the store was off in code from that day |
| a rendered window, reused | beside the render (`loop_resume.py`, `reuse_windows`) | the queued graph upstream of the song node, the track, the window's text, length, start and seed, the window before it | REMOVED: the keys and the reuse. The window FILES stay: a window's video and its sampled latent are written on every run, and read again only when a person names them (`continue_from`, `bench/recomposite_window.py`) |
| a window's source encode and prompt encode | the server's memory (`window_keep.py`) | the identity of the live objects they were made from | REMOVED with its module and check. It was switched by `reuse_windows` on the node and not by the code switch, so it was live in any graph that set that input |
| the shot table | `<prefix>_NNNNN_shots.json`, and the Subject Track's `shot_table` output | `TABLE_VERSION`, the file's format | unchanged: a file a person wires into `subject_from`, which is the explicit form the rule allows |

A mask made once and used many times is a mask VIDEO a job's graph loads
from a file its capture wrote: named, visible, and replaced when the
capture is made again.

**Three things called a version, and none is the other.** The pack's
version is the newest heading in `CHANGELOG.md`, assigned by
`bench/build_changelog.py` from fragments (semver; `pyproject.toml`'s
`version` is what ComfyUI's registry reads and is kept equal to it by the
same build). `MASK_VERSION` is not a version of a node: it is one integer
per node class that goes into the kept mask's key, bumped only when that
node's code would make a different mask from the same inputs, so the
numbers on different nodes (`subject_track.py`, `sapiens2_parts.py`) are
unrelated and never compared. *Until 2026-10-12 this list named
`video_mask.py` too, which has had no such number since 2026-10-10, as the
next sentence says.* The kept mask it keyed
was removed on 2026-10-10 and the Masked Source's own number went with it;
the trackers keep theirs because the job builder reads the Subject Track's
for its corrections rule. `TABLE_VERSION` (`shot_table.py`) is
the shot table's file format, checked on read. A node's identity is its
`node_id` in `bench/node_id_manifest.json`, which has no number at all.

## A latent step is several frames, and its grid is fixed for a whole load

The video model does not work in frames. Core's `FRAME_PER_TOKEN`
(`comfy/ldm/minimax/model.py`) says how many pixel frames each latent step
covers, and it cycles: `video_mask.run_lengths` is that cycle laid along a
window. Everything this lane decides per step it decides for every frame
of the step: the region (`token_mask` takes the maximum over a step's
frames), what `keep` and `others` hand back, the weight the composite lays
the render with. A frame cannot be regenerated on its own, and neither can
a frame be left alone while its neighbours in the step are redrawn.

**The grid does not move from window to window.** A window's length is on
`loop_plan.CHAIN_LENGTHS` and its context on the three values
`check_window_settings` allows, so what a window adds is a multiple of
`loop_plan.GRID`, which is a whole number of cycles. Every window of a load
therefore cuts its steps at the same places counted from the LOAD's first
frame: a step starts where the frame's number from the load's start,
modulo the cycle's length, is one of the cycle's own offsets. Two things
follow, and both are arithmetic that needs no render:

- **Which cuts of the source split a step is known before anything is
  queued**, from the cut list and the load's first frame alone. A cut that
  falls inside a step puts one to three frames of the other shot under the
  same region. On the first clip this was looked for, the frames the
  assembler found repainted at four cuts were exactly the frames this
  arithmetic names, and the three cuts it puts on a step's edge had none
  (`data/CAPTURE_GAPS.md`, U5; `bench/capture_masked_run.py`'s
  `region_carried_across_a_cut` is the flag).
- **The load's first frame is a lever, for a load with few cuts.** Moving
  it by fewer frames than the cycle is long moves every step's edge with
  it, so the cut that matters most can be put on an edge.
  `loop_plan.split_steps` lists the steps a load splits
  (`bench/check_audio_freeze.py`, the step grid case); a function that
  ranked every start one cycle back was removed on 2026-10-10, moot under
  one load per shot. All of a load's cuts cannot be cleared in general: a
  cut is on an edge for as many starts as the cycle has steps, out of as
  many as it has frames, so the chance that one start clears several cuts
  falls with each cut. On the first clip's whole-subject piece no start
  cleared every cut the subject is beside.

**The cycle is the video VAE's, and the decode is not local to a step.** A
reading of core's code (`comfy/ldm/minimax/vae.py`: `MiniMaxH3VideoVAE`'s
`encode_temporal`, `decode_temporal` and `blend`, `ViT3DDecoder.forward`),
made 2026-10-10, with nothing run:

- The VAE cuts a video into clips of `clip_length` frames and that length
  is one cycle of `FRAME_PER_TOKEN`: the encoder's `time_down` gives the
  frames a token holds, and `frame_pre_padding` is why a clip's first
  token holds one. So a load's first frame fixes the VAE's clip boundaries
  as well as the step edges. They are one grid.
- **Encode: each clip by itself.** The clips are encoded in a loop, one at
  a time, a short last clip padded with copies of its last frame. Inside a
  clip the convolutions are causal and the norms are per frame. A latent
  step is therefore shaped by its own frames and by earlier frames of its
  own clip, by no later frame, and by nothing outside its clip.
- **Decode: a window of steps at once.** The decoder is a transformer over
  a clip's tokens and the next clip's first `token_overlap` of them, every
  token attending to every other inside a spatial tile of `tile_size`
  pixels. Every decoded frame of a clip is shaped by every step of its
  clip and the head of the next. The first `frame_overlap` frames of each
  clip after the first are a linear cross-fade from the previous window's
  decode of those same frames, and the clip's first frame is taken from
  the previous window whole.

So no frame beside a cut is decoded from its own step alone, on either
side, whether or not the cut splits a step. That is how any clip with a
cut decodes with no mask at all, and is not a fault by itself. What is
special to a split step is the latent: one token holds frames of two
shots, and inside the region the sampler has to make that token up.
Whether the frames on the subject's side of such a token are worse than
their neighbours is not measured.

What the lane does about a split step today: the composite lays nothing on
the frames of a step that lie across a cut from every frame the subject is
on (`video_mask.cut_gate`; the song node's report names them). The sampler
still regenerates those cells, since it cannot do otherwise, and the case
where the subject is on both sides of the cut in two different places is
open. A seam between windows is a different thing from a cut: the frames a
window shares with the one before are frozen latent steps copied whole
(`audio_freeze_song.py`), which is why the context is a length that ends
on a step's edge.

**Everything else that is held over a step.** An audit of the lane's code
on 2026-10-10 for every place that holds a thing over a latent step's frames
or reads per frame what is per step. Read from the code; the one measured
row says so.

| place | what it holds over a step | what that does at a cut inside the step | named by |
|---|---|---|---|
| `video_mask.token_mask` (the region) | the maximum of the subject's cells over the step's frames | the region lies on the other shot's frames and the sampler regenerates them | the preflight's `region_carried_across_a_cut` (`loop_plan.split_steps`); the song node's report |
| `pixel_alpha`, the composite's weight | the step's region on each of its frames | the render is laid on the other shot's frames | fixed: `cut_gate` in `lay_window` |
| `changed_alpha`'s hold | the maximum over the step of "changed, or the old subject stood here" | the far frames differ from the source over the whole region, so the subject's own frames keep the whole region and less of the margin is restored (measured, [`2026-10-10_saved_windows_laid_again.md`](../../bench/results/2026-10-10_saved_windows_laid_again.md); *until 2026-10-12 this row named that record's `.json`, which left the tree when the record took its own tables and is under the untracked `data/2026-10-10_recomposite_decodes/`*) | nothing; `data/CAPTURE_GAPS.md` U6 |
| `keep` in `window` | a token with a kept pixel on any frame of the step is kept on all of them | with the subject on both sides, one shot's keep leaves the source showing inside the other shot's subject | the preflight's flag for a subject on both sides of a cut in one step |
| `others` in `window` | the same hold, taken out unless the subject's own mask has a pixel in the token on a frame of the step | the same | the same flag |
| `cut_gate` | which side of each cut the subject is on, per step | with the subject on both sides it gates nothing: each side has the other's region | the same flag; open (a region per side) |
| `start_zero_tokens` (a late start) | the body's hole and the top share kept, as maxima over the step | the other shot's frames start from the same emptied cells; the kept top of one shot is kept on the other | nothing; only with `start_from` other than noise |
| the mask review's region layer (`token_region`) | the region the sampler was given, per step | under `whole region` a gated frame still shows the region tinted though nothing was laid on it; under `only what changed` the outline is the gated weight | a reading caveat, not a fault |
| the capture's planned region | per frame, where the sampler's is per step | a plan's "inside the region" figures are a floor, most of all at a cut and across a fast move | `data/CAPTURE_GAPS.md` 30 |
| the frozen context | whole latent steps copied from the window before | after a cut the context is another shot's picture | one load per shot |
| the video VAE | clips of one cycle, encoded apart and causally, decoded a window of steps at a time | both shots share a decode window; ordinary, and not the lane's to change | the paragraph above |

Not held over a step, checked: the motion reference and a wired motion
video are cut per frame in step with the source, a short one holding its
last frame; the margin under `the subject's size` is each frame's own, from
a running median that keeps a step at a cut; the zoom's boxes are per shot;
the trim and what a window writes fall on a step's edge.

**What "frozen" is, and why a window follows its context.** Read from
core's code on 2026-10-10, nothing run. A row at mask 0, a context row or a
plate row alike, is conditioning inside the picture: on every step the
sampler feeds it the clean latent (`comfy/model_base.py`,
`MiniMaxH3.scale_latent_inpaint`), the model labels it at the conditioning
timestep (`comfy/ldm/minimax/model.py`, `_forward`: `t_pin_v`), and its
velocity is zeroed. It is never noised. A reference still and a reference
video's own copy carry the same label by another path, but their level can
be set (`reference_noise.py`), and a masked row's cannot: both places read
core's constant and not the payload that node writes. So a continuation
window is shown the frames it inherits exactly as clean and as trusted as
its references, at the canvas's full size, next to the frames it draws,
and a short window holds more rows of them than of a motion video at a
small short edge. Renders of 2026-10-08 and 2026-10-10 followed the
context's pose where the motion video's had moved on
(`docs/research/postmortems/`, the masking board's card on it).

The levers that loosen it without a patch to core: fewer context frames
(`context_frames`), none (a load of its own, `continue_from` unwired), a
longer window, a larger motion video, and `context_noise` on the song node
and the window node (`audio_freeze.CONTEXT_NOISE`): core runs a mask value
between 0 and 1 as that row's own strength, so the context's rows can be
shown part noised and redrawn by that share. With a source wired the
plate's rows stay at 0 (the song node takes the minimum with the region),
so only the region's share of the context loosens. Its default is the mask
every render so far was made with; any other value is untested.
[`window_context.md`](window_context.md) is the ledger for this question:
the mechanism, every lever with its kind and status, and each result as a
dated line.

## What is known from renders, by form

A router and not a result: each row says in a line what this page and its
records already say, and where. One shot or one seed is said as that. A
form with no row has not rendered, or its render was not read.

| the form | what it was seen on | what it carried | what it did not show | where it is written |
|---|---|---|---|---|
| a whole subject on the fast chain, the generic prompt | the first clips, 2026-10-04 | the person replaced from the still, in the hole the mask gives | movement: the new subject faces the camera when the original turns, and a late start carries the turn with the original's hair and clothes | "Known limits", "Replacing the person"; `../../bench/results/2026-10-04_masked_v2v_first_run.md`, `../../bench/results/2026-10-04_masked_v2v_band.md` |
| the subject's own frames on grey as `<Video 1>` (`motion_reference`), on ref2va | the arms of 2026-10-05 | the subject's movement, at `h3_config.MASKED_MOTION_STEPS` | fl2va ignores the reference at any step count; ref2va loses it at eight | "What it is"; `../../bench/results/2026-10-05_masked_v2v_motion_arms.md` |
| the head and upper body on the ref2va motion graph | one window of one clip at two lengths and one seed; a typed text beside the node's wording | the owner called it solid on playback, and the two wordings both good and not told apart | the same region on the fast chain was called broken | "What it is"; `../../bench/results/2026-10-06_masked_v2v_body_window_arms.md`; the stacked pair on the share |
| `keep` | once on 2026-10-07; inside a face-only region on 2026-10-10 (one clip, one window, one seed) | on 2026-10-10 the jewellery that was kept was held | the 2026-10-07 render was not judged; inside a face the kept tokens held the skin beside the jewellery too, and the face moved back toward the original's | piece 2, `keep` |
| a body mesh as the motion video | one window, one shot and seed, as an appended video before the input existed | the action, with no action words | things inside the region came back as other things until one sentence named them | piece 2, `motion_video` |
| a body model with lid marks as the motion video | the second subject's kitchen shot, her face and hair, one seed a render; then the lead's sung close-up, her face only | on the kitchen shot her head following the source's, and her lids at the source's level with the marks enlarged | following in time; a second seed; and on the close-up it did not repeat | "The motion video as a body model, with marks drawn on its face" |
| a small subject rendered zoomed in | one street shot, one load, one seed | it ran clean and the face is replaced where it shows, in stills | not judged on playback; whether it reads as the reference at that zoom is open | "A small subject, rendered zoomed in" |
| a continuation window after its context | renders of 2026-10-08 and 2026-10-10 | the context's pose | the motion video's pose where it had moved on | "A latent step is several frames", the paragraph on what "frozen" is; [`window_context.md`](window_context.md) |

## The motion video as a body model, with marks drawn on its face

**How to read this section: patterns with their conditions, not a recipe**
(the owner's ruling at the close of 2026-10-11, in substance). Nothing
below is a procedure that gives a result. It is a few patterns, each of
which did a named job under the conditions it was tried in, with our best
account of why. Do not assume one holds on a shot unlike the one it was
seen on. Do not start from nothing either: begin from the pattern nearest
your case, say which of its conditions your shot shares and which it does
not, and treat the first render as a test of the pattern, read against its
source before anything is built on it.

| the pattern | what it was for | the conditions it was seen under | why we think it worked | where it stands |
|---|---|---|---|---|
| a body-model video of the subject as the load's motion video | her head following the source's | one shot; one person, seated, her face AND her hair redrawn | the model takes movement from a video reference, and with her hair redrawn nothing else in the picture fixes where her head is | held on that shot. On a load that redraws a face only, her head sat as near the source's without the video, so it is not shown there |
| the source's eye openings drawn LARGE on that body model's face | her eyelids at the source's level | the same shot; the source's eyes lowered for most of it; each mark several of the model's cells wide | marks that fit inside one cell did little and the same marks enlarged did, under the same words: a mark has to be large enough for the model to see | held on that shot as a level, never as following in time. Did not repeat on a second shot, which differed in several ways at once: another person, a face only, marks grown less against her face, eyes fully shut for a long stretch |
| a text that says what the reference video is and that she follows it, and nothing of what her face does | the model using the reference, with a render still able to show whether the reference or the words did it | the same shot, with the enlarged marks | the words license the reference and the picture carries the content: a longer text that related her eyes to the figure's did no better | held on that shot, on one seed and nearly one text; the owner picked it on playback |
| the pose model's eye figure on the source and on the render, taken apart by where the source's eyes are open and where they are lowered | a first check of a render's eyes before a person looks | the renders of one shot, with one render made twice as the floor | on that shot it fell on the same sides as the owner's verdicts on playback | trusted at its two ends only; its mouth figure is not trusted; a line drawn after the verdicts were seen |

**When a pattern fails on a new shot, that is a row for this table, not a
reason to delete one:** add the shot and the condition that differed, and
keep the pattern with the narrower condition it now has. The notes that
follow are the history of exactly that. Every figure and caveat behind a
row is in the records this section points at.

**Provisional, written 2026-10-11. It worked on one shot and may not on
others.** The shot is the second subject's opening kitchen shot, her face
and hair redrawn, one seed a render. Two more shots were rendering when
this was written, one of them the first test of following in time. When
they are read this section gets a dated note, and if the practice does not
hold it is changed here and not left standing. It is ours, from one shot,
and not the vendor's: mark it so wherever it is repeated. The two records
hold every figure and every caveat:
[`2026-10-11_kitchen_shot_head_and_eyes_two_tests.md`](../../bench/results/2026-10-11_kitchen_shot_head_and_eyes_two_tests.md)
and
[`2026-10-11_kitchen_shot_lid_marks_on_the_body_model.md`](../../bench/results/2026-10-11_kitchen_shot_lid_marks_on_the_body_model.md).
The owner's rulings behind it are in [`decisions.md`](decisions.md) under
2026-10-11.

**2026-10-11, later the same evening: on a second shot it did not
repeat.** The lead's sung close-up, her face only, one seed, given a
body-model video of her with lid marks and the wording below. Her eyes
read as they do in the render of those frames made on 2026-10-10 with no
reference video at all: open through the long stretch where the source's
are shut, and less than wide where the source's are wide. So on that shot
the marks gave neither a level nor timing that can be told from having no
marks. Her mouth sings as it did before, and her head is as that earlier
render's. Four sessions read it and agree, one of them blind to the
others: the session that took the looks, the one that holds the build, a
blind reader, and the one writing this page, which worked the eye figure
again from the tables. What each kind of reading supports is kept apart.
From the tables: the render with marks and the render without sit at the
same number on both stretches. From stills: that her eyes are open at
that number, which lies in the middle the eye figure cannot read
([`meta_perception_models.md`](meta_perception_models.md), "The trials",
the note under the table). No render of this load was made twice, so no
floor exists for it and "the same" is all that is said, never a
difference. One shot, one seed. The record, with the table and where the
files are:
[`2026-10-11_lead_closeup_lid_marks_did_not_carry.md`](../../bench/results/2026-10-11_lead_closeup_lid_marks_did_not_carry.md).

**Why is not known and is not said here.** Several things differ from the
kitchen load together and none is separated: the person, a face only
against a face and hair, the mask and its margin, how much the marks were
grown, how a shut eye comes out when it is drawn, and a sentence in the
text that she sings. The record lists them, and the one render it puts
to the owner next: the same load with the marks grown as much as the
kitchen's and nothing else changed. Read everything below as what one
shot showed and a second did not.

### What the reference video was

One video, the load's motion video (`motion_video` under the Masked Source
above), the size of the canvas, one frame for each frame of the load:

- **A plain grey body model on a black ground**, posed on every frame as
  the person in the source is posed on that frame, so it moves as she
  moves. Nothing of the source's picture is in it: no face of hers, no
  clothes, no room. Its own face is a neutral modelled face with sockets,
  a nose and a mouth, and it never changes: the body model sets the jaw
  and every expression to zero before it is posed
  ([`meta_perception_models.md`](meta_perception_models.md), "SAM 3D Body",
  "The face"). On the frames looked at its mouth is shut.
- **On that face, over each eye, a white shape with a thin black edge**
  filling the space between her upper and her lower lash line on that
  frame: an almond where her eye is open, a sliver where it is lowered or
  shut. Lids only; nothing is drawn for the mouth, the brows or the
  pupils. Each shape is drawn larger than her eye, grown about its own
  centre by one factor for the whole shot. In the render the owner picked
  the factor was three (the drawing script's `--grow`), which makes each
  shape several cells of the model's grid wide and has the two reach
  across the bridge of the nose.

The file is untracked and is not shown here:
`data/2026-10-11_fun_gate_inputs/second_mesh_with_lids_0014_x3.mp4`.

### How it was made: the nodes and the code

1. **The body model.** A job's preview draws it from the track:
   `MiniMaxH3SubjectBoxes` (`subject_boxes.py`) gives one box a frame round
   the tracked person, and `MiniMaxH3BodyModelLoader`, `MiniMaxH3BodyPose`
   and `MiniMaxH3BodyMeshVideo` (`body_pose.py`) pose the model in it and
   draw it. Their settings are `h3_config.MASKED_JOB_BODY_MODEL`,
   `MASKED_JOB_BOXES`, `MASKED_JOB_BODY_POSE` and `MASKED_JOB_MESH`. The
   kitchen shot's was the one drawn on 2026-10-10 with those nodes at those
   values, handed in as a named file.
2. **The lash-line points.** The Sapiens2 pose model on every frame of the
   source, fitted to the canvas as the loader fits it, in the tracked
   person's box. No node runs that model. `bench/pose_along_shot.py` does:
   its `run` needs the card and writes each frame's points and scores, the
   points' names and the source frame of the first row, with a record
   beside them that says how the frames were fitted; its `read` needs no
   card and says what a shot's points can carry before anything is drawn
   from them (the face's height and the gap between the lash lines on the
   canvas, beside `h3_config.POSE_READING["lid_gap_px_to_draw"]`), with no
   verdict. The model folder is `h3_config.POSE_MODEL` in the owner's
   store. The checkpoint does not name its points. The tool takes Meta's
   list of names from `vendor/meta_sapiens2/pose_keypoints308.json`, kept
   with Meta's licence file beside it, for a model folder whose own left
   and right pairs are the list's. That says the folder is the model the
   list is for; the list's order is held to Meta's own file by
   `bench/check_meta_sapiens2.py` where the reference checkout is present
   (the tool's docstring, "The names"). *Tracked since the night of
   2026-10-11; until the names landed that night this step said the tool
   could not run from a fresh checkout. The day's renders used
   the one-off it replaces,
   `data/2026-10-11_sapiens2_pose_consecutive/run.py`.*
3. **The marks.** `bench/draw_lid_marks.py` draws the shapes on the
   body-model video from those points. It writes the marked video, a
   record beside it and, given the source, a sheet of a few frames laid
   over the original to look at before the video is used. It reads its own
   video back, and its `verify` is for a second reader, with that reader's
   own account of which frame each file begins on. Each shape is grown by
   `h3_config.LID_MARKS["face_px"]` over the shot's median face height, or
   by a factor given by hand. That constant is a starting value measured on
   this one shot and says so; another clip's own faces set it for that clip
   (`../../AGENTS.md`, "A rule that depends on the picture"). The tool
   leaves no frame bare by itself: frames where a head is turned away are
   named by a person. It says when a shot's lash lines sit closer on the
   canvas than `h3_config.POSE_READING["lid_gap_px_to_draw"]`, and still
   draws. *Tracked since the evening of 2026-10-11. The kitchen shot's
   video was drawn before that, by the one-off the tool was written from:
   `data/2026-10-11_fun_gate_inputs/draw_lids_on_mesh.py`.*
   **A second drawing, since 2026-10-12 and not yet rendered with:**
   `bench/draw_face_points.py` draws the pose model's face points on the
   body-model video as white lines and dots at the face's true size (each
   eye's lash ring, eyelid and crease lines and iris, the brows, the nose,
   the lips), with the same record, read-back, sheet and `verify`, and
   `bench/check_draw_face_points.py` holds it. A part whose points a person
   saw are off her face is left out by name. What it cannot say is whether
   the lines can be told apart at a face's size on the canvas: its sheet
   is where that is seen. On the first shot drawn they could not, and
   `bench/crop_reference_to_head.py` (the same day, held by
   `bench/check_crop_reference_to_head.py`, not yet rendered with either)
   cuts a reference video to one box round the shot's head and enlarges it
   to the canvas, and writes the points carried into the new picture, so
   that the lids or the face are drawn on the cut body model at the new
   size. A reference cut this way no longer lies over the source pixel for
   pixel, and the line of text that says what `<Video 1>` is has to say
   so.
   **One command for that form** (2026-10-12, after the owner made it
   the form of every head or head-and-hair replacement:
   [`decisions.md`](decisions.md)): `bench/head_reference.py` cuts the
   plain body model, cuts the source by the same box, draws the face on
   the cut body model with no dots and the mouth left out, and writes the
   reference, its sheet and one record that names every input by path and
   the commits it ran on; `bench/check_head_reference.py` holds the
   joining. Its width, no dots and the left-out mouth are the first
   references' own, from one look, and each can be set aside.
4. **The load.** A job file's load names the marked video: `motion: mesh`,
   `motion_file`, `motion_first` (`../../workflows/masked_job_graph.py`),
   and its `text` names a text file. It goes through `bench/masked_job.py`
   like any load, with the capture, its gate and the looks ("Running a
   job" above). Nothing was posted by hand.
5. **The render's graph**, as the builder wrote it: `MiniMaxH3MaskedSource`
   with `motion_reference` on `a video I wire`, the video model's own copy
   of the video on (`motion_vae`), the video at the canvas's short edge and
   `start_from` on `noise`; `MiniMaxH3AudioFreezeSong`; `MiniMaxH3Sol`;
   `MiniMaxH3AppendRefImage` for the still; and video loaders for the
   source, the masks and the reference video. The graph is saved beside
   the job: `data/2026-10-11_gate_p4_0014_job/second_hf_0014_p4.render.json`.

**Steps 2 and 3 are tracked tools, each with a check; nothing joins them
up.** No node runs the pose model or draws marks, no shipped graph has
this form, and the builder cannot make a marked video; it can only be
handed one. *Until the night of 2026-10-11 this paragraph said the pose
pass was a one-off script under the untracked `data/`.*

### What it carried on that shot

- **The body model carries her head.** With it her head follows the
  source's; without it, and without the sentences that go with it, her head
  holds nearly still (the two-tests record).
- **Nothing carried her lids until the marks did.** In every render
  without them her eyes read wider open than the source's. With the
  enlarged marks her lids come down to about the source's level where the
  source's are lowered.
- **It is a level, and following in time is not shown.** Where the source's
  eyes are open no load opened hers as far as the source's, the render
  without marks included, and the shot has no blink. *Later the same
  evening: the shot that tests timing was rendered and her eyes did not
  follow; the note at the top.*
- **On playback the owner picked the load with the enlarged marks and the
  text below**, and called the other loads with enlarged marks as good.
  They also put one load with marks at true size level with those, the one
  whose text was laid out the vendor guide's way, and not the true-size
  load in this pack's form: the same pair the measure separates ("Why we
  think it worked" below). Their saved lines are in the local page's
  untracked `data/fun/poc/answers.jsonl`.

### The text: the fragment

**SINCE 2026-10-14 THE ONE HOME OF A LOAD'S FIXED LINES IS `bench/load_text.py`** (the owner's word that day: the
text says exactly what the reference video shows and the least else; every window's text consistent with the others).
`bench/shot_to_load.py text --place-and-action "..."` makes the text from the reference's record (what was drawn: the
parts, the eye's form, the mouth), the plan's words for the subject and the video, and the voice table, around the one
sentence a person writes; a text written whole by a person is the exception (`--by-hand "why"`). Its examples are the
kept loads it reproduces: lead_fp_1730 and lead_fp_1755 (10/13), second_hfp_0484c (10/14), in
`bench/check_load_text.py`. What this section quotes below is the form of 10/11 ("shapes drawn on its face"), kept as
what the lane wrote then; it describes no load the chain makes now.

*The frame this text is written in stood at the head of this section for
part of 2026-10-12 and is a section of its own since the page was
reorganized that day: "The frame every masked text is written in", near
the top of this page. Read it first.*

A reference the text does not speak of is one the model cannot know to
use, and a text that says what her face does leaves a render unable to say
whether the text or the reference did it (the rulings). So the text says
what the video is and that she follows it, in four places, and nothing
else about it. Copied from the picked render's text
(`data/2026-10-11_fun_gate_inputs/second_hf_0014_p4.txt`):

The end of the subject's definition, after who the subject is:

```text
, who follows the movements and the facial expressions of the figure in <Video 1>.
```

The line that says what the video is:

```text
<Video 1> is a plain grey body model on a black ground, with shapes drawn on its face that are part of its facial expression; <Subject 1> follows its movements and its facial expressions, and its grey look and the shapes themselves are not drawn.
```

Its retention line:

```text
<Video 1> (motion source): attribute_transfer - the movements and the facial expressions of the figure are transferred to <Subject 1>; the grey body model's look, the shapes themselves and its black ground are not.
```

One sentence in the shot:

```text
She follows the movements and the facial expressions of the figure in <Video 1>.
```

What to hold to when it is used on another load:

- **The second line describes that video.** Grey, a black ground, shapes on
  its face: if the reference is drawn another way the line says what that
  one is, as the model sees it. The rulings have layers being added to the
  video one at a time, and this line changes with each.
- **Nothing names a feature, a turn of the head or what she does**, and the
  shapes are given no colour. A sentence about her eyes is ruled out, a
  clause that only relates them to the reference's included.
- **The pronoun and the subject's number are the load's.**
- **A shorter wording, "movements and expressions", was under test** on
  the next shot when this was written. *It had not rendered when that
  shot's first render was read.*
- **A line for `<Video 1>` as a source of movement is this pack's form**,
  measured on 2026-10-05
  (`../../bench/results/2026-10-05_masked_v2v_motion_arms.md`). The vendor
  guide keeps `<Video N>` for a relationship to a whole video
  ([`../prompting.md`](../prompting.md), section 9.4). The same content
  laid out the guide's way was rendered once, with marks at true size; the
  lid-marks record has what it did.
- **A pattern, not a paste** (the owner, 2026-10-12, about 17:41Z; the
  words are in [`decisions.md`](decisions.md)): this is how one shot was
  solved, in the vendor guide's form for a reference. When the reference
  video is the marked grey mesh, the lines that say what it is and that she
  follows its movements and expressions apply; a plain mesh gets a plainer
  line. The scene is kept simple: the frozen frames and the sound carry it,
  and a text does not inventory what is behind her. And whoever builds or
  diagnoses a load reads its text in full, with its inputs seen: never a
  summary, never secondhand.
- **The condition the pattern holds under, in the owner's words**
  (2026-10-12, about 18:00Z, in the lead's terminal; logged in
  [`decisions.md`](decisions.md) that day): "don't take that fragment of
  text as a 'paste this onto EVERY prompt' verbatim. If you use a body mesh
  and you use a sapiens2 overlay data, and youre just doing face
  replacements. it makes sense. If you're doing something else, it doesn't
  anymore, because the reference video may have changed, or maybe its not
  about the face. In our video, I think its always about the face and
  head... but just an fyi. Don't make it a rule that ends up getting
  inserted into things that causes future debugging hell." So the four
  places fit a load whose reference video is the marked mesh and whose
  region is a face, or a face and head. On any other load they are not
  assumed, and no tool inserts them (the next bullet).
- **It is not what the Masked Prompt node writes.** The node's lines for a
  motion video (`masked_prompt_text.VIDEO_DEFINITION`, `VIDEO_RETENTION`,
  `MOVES`) call the video a person and take only body movement from it. By
  the owner's ruling of 2026-10-11 this form stays a documented practice
  for now and is not put into code: a session writes a load's text from
  its frames, sets the four places into it, and the job's note says where
  they came from.

### One whole text, as an example

**An example, not a text to copy.** The whole text of the render the
owner picked is kept so the fragment can be seen in its place. Every
sentence about the scene in it was written from that one shot's frames and
is true of those frames only; a text for another load is written from that
load's frames (`../../AGENTS.md`, "A scene's text is data"). And it is not
a model of a caption: it still holds sentences about what "stays as it is"
and what is "the scene's own", the kind the owner ruled out on 2026-10-10.
They were kept the same across the day's loads on purpose, so that one
thing changed at a time, and a render of this shot with them deleted came
out as the same piece (the lid-marks record, the load called `text`). Do
not carry them to a new text.

The whole text is kept in one place, beside an example of that load's job
file: [`bench/examples/masked_job_lid_marks/text.example.txt`](../../bench/examples/masked_job_lid_marks/text.example.txt).
The page beside it
([`README.md`](../../bench/examples/masked_job_lid_marks/README.md)) goes
through it section by section: why each part is there, what it speaks of,
and what had to be made before it was true.

### Why we think it worked, and what is not shown

What the renders support, and no further. Each is one shot and one seed.

- **The body model, shown as a video, is what her head follows.** Taken
  away, her head held nearly still. *That is the kitchen load, where her
  hair was redrawn with her face. It is not shown for a load that redraws
  a face only: on the sung close-up the earlier render with no motion
  video sits as near the source's head as the one with it (the note at
  the top), and the two-tests record has the same of that shot.*
- **Larger marks did more, under the same words.** At true size each lay
  inside one cell of the model's grid and her lids moved a little (against
  the render with no marks, whose text also lacked a clause relating her
  eyes to the figure's, so that small step is not the marks' alone).
  Enlarged, under the same words, they brought her lids to the source's
  level. That step is the picture's and no sentence's. Small marks were
  not unseen, though: the same true-size marks under the other form of
  words (the vendor guide's layout) brought her lids most of the way. Part
  of what size buys is in the drawing itself: the dark edge and the
  encoder take a strip along a mark's outline, so a thin mark is mostly
  edge and little white (`../../bench/draw_lid_marks.py`, the comment at
  `EDGE_PX`; held by `../../bench/check_draw_lid_marks.py`).
- **The words say that the video is followed; the picture says how.** A
  text that only says what the video is and that she follows it did the
  same for her lids as a longer one that related her eyes to the figure's.
  For her head it did a little less: the limit below. Saying what is
  retained of the video, or what the shapes are for, changed nothing by
  the measures of her lids, of where her head sits and of her lip gap, on
  this shot at this seed.
- **Her look comes from the still and not from the figure, as far as was
  looked.** On the stills the readers looked at no render shows a grey face
  or the shapes, and no neutral white was found in a box round her eyes on
  any frame; the rest of her face was not measured, and greyness was not
  measured at all. One reader found a light streak on each upper lid, over
  a run of frames in the middle of the shot, in the load with enlarged
  marks, where the render without marks has a small bright point inside
  each eye and no streak (its figures:
  `data/2026-10-11_mrtie_gate_o_read/`). Nobody has judged it on playback.
  Every text of the day said the figure's look is not used or not drawn,
  so no render shows what happens without those words.

Not shown: that she follows in time (nothing blinks in that shot, and
where the source's eyes are open hers opened less far than the source's in
every render); a second seed; any other shot. The three plain wordings are
one seed and nearly one text, so their agreeing is one result seen three
times. And the form of the words moved her face once, between two loads
with the same true-size marks, for a reason nobody has: under this pack's
form her lips were held parted through the whole shot and her lids moved
a little, and under the guide's form her mouth was as in the render
without marks and her lids came most of the way.

### Limits known today

- **A small face has no lids to draw.** On the lead's street shots the gap
  between her lash lines at whole-frame size is too small to draw marks
  from or to read a render by (the lid-marks record, "Which other shots can
  take it"). The pose model's own score for the lid points does not say
  this: it stays high on such a face and on a head turned away, so the
  score cannot be what decides where marks are drawn.
- **A load rendered zoomed in takes a motion video only as a file the load
  names, and the node does not cut that video to the box** (since
  2026-10-13: `video_mask.zoomed_source`, "A WIRED MOTION VIDEO IS HANDED
  ON AS IT WAS WIRED"; the builder's rule is at
  `workflows/masked_job_graph.py::RENDER_REGIONS`). So the reference of
  this section in its cropped form (the body model cut to the head, which
  has a framing of its own) can go with the zoom of the section below; a
  whole-frame body model cannot, and the builder refuses the preview's
  own. Until that day this line read "takes no motion video ... the
  section below and this one cannot be combined today". No render had
  been made of the pair when this was written, and the preview stage
  (tracker, part model, pose, body model) still has no zoom: a face the
  pose model cannot read at whole-frame size gets no drawn points from
  the zoom alone.
- **Her head sits a little further from the source's under the plain
  text**, while its turn is the same in every render. The word "head" left
  the text along with several other changes, so the cause is not separated.
- **The figure's mouth never moves.** On a shot where the
  subject sings, a text of this form says both that she sings and that she
  follows the figure's facial expressions. Not rendered when this was
  written. *Rendered later the same evening, on the lead's sung close-up:
  her mouth sings as it did in the earlier render that had the singing
  sentence and no reference video, by the reader's eye; the figure's still
  mouth did not shut hers. One shot, one seed.*

### Reading a render of this form

The pose model on every frame of the source and of the render, as the
records do: the eye opening taken apart by where the source's eyes are open
and where they are lowered, and where the head sits in pixels beside any
correlation, since a correlation hid the head's offset on this shot. A
level is stated with a floor beside it, the same render made twice; that
floor bounds two makes of one render and says nothing of a second seed. The
eye figure is trusted at its two ends and not in between, and the mouth
figure is not trusted
([`meta_perception_models.md`](meta_perception_models.md), "The trials",
the note under the table).

*Since the evening of 2026-10-11 one tracked tool does this reading:
`../../bench/read_render_against_original.py`, from the points files of the
source and of each render. The stretches where the source's eyes are open
and shut are best given by a person who looked, by name; without two makes
of one render handed in as a floor it prints every difference and calls
none a difference; it writes the lip gap as a column and says nothing of
a mouth, which is read by eye. Its figures are defined once, in
`../../bench/_lib/face_points.py`, and the lines it reads them against are
`h3_config.POSE_READING`, each a starting value. It replaces the three
sessions' own scripts the records of that day were read with.*

### What comes next for the video

By the rulings: segmentation and depth are to be shown to the model as
further layers of this one video, never as more reference videos, added
one at a time and each read as a single render against the kept one: the
pose model's own drawing of the face points, then the part colours, then
depth shading.

## A small subject, rendered zoomed in

Built 2026-10-10; when this was written no render had been made with it.

**Why.** A subject small in the frame is a few of the model's tokens. On
one clip the owner did not read a small face as replaced, and then did,
once its load stopped fading at the end (the open tail, now the only tail): so how much of that
was the face's size is not settled, and smaller subjects than that one are
in the clip. `video_mask.RENDER_WHOLE`'s comment holds the case.

**What it does.** `render_region` on the Masked Source (`RENDER_REGIONS`:
the whole frame, or zoomed in on what is replaced). Under the zoomed choice
the song node crops the load to ONE box round what is replaced
(`video_mask.zoom_box`, never smaller than `ZOOM_LARGEST` allows), scales it
to the canvas (`zoomed_source`), and
runs everything as it does on any clip: the margins, the token region, the
motion reference, the encode, the sampler, the decode and `lay_window` all
see a clip that happens to be a crop. `lay_back` then puts the region into
the whole frame, so a window's video and the joined file are the whole
frame as ever. Where the weight is 0 the pixel is the loaded frame's own.

**What it buys, and what it does not.** Tokens for the subject while it is
drawn. Not delivered pixels: the window's video is still the canvas. For a
delivery built at the source's size the laid picture in the zoomed space
is written beside each window and joined (`<stem>_zoomed.mp4`), with
`<stem>_zoomed.json` saying the box (`audio_freeze_song.zoomed_record`).

**One box for the whole load.** A continuation's kept frames are the
window before's latent, so two windows with two boxes would hand the model
a context of another framing; and a box held across a cut frames nobody on
one side. So the box is fixed over every frame loaded, and a load with a
cut inside it is refused: load one shot at a time. A subject who crosses
the frame makes the box most of the frame; under `ZOOM_SMALLEST` the load
is rendered on the whole frame and the report says so. A zoomed run
continues nothing and is not continued (`continue_from` is refused both
ways).

**A wired motion video goes in as it was wired** (2026-10-13; refused from
2026-10-10 until then). The zoom does nothing to it and it does nothing to
the zoom: the box is from the replaced part alone, and a zoomed window is
shown the same reference a whole-frame window of the load would be, to the
bit (`video_mask.zoomed_source`, `wired_motion`). What that means depends
on the video. One with a framing of its own, the body model cut to one box
round the head (`bench/head_reference.py`), is in neither the frame's
coordinates nor the box's on any load, and is the same reference here:
this is what the pairing is for. One that runs beside the whole frame (a
whole-frame body model) still shows the whole frame, its figure at the
frame's place and size beside a plate that is the box enlarged; nothing
cuts it to the box, the report says so, and the job builder refuses the
preview's own body model on a zoomed load and takes only a `motion_file`
the load names. The zoom does not enlarge or sharpen the reference.

**The box is in the pixels the node was handed.** The job graphs' loader
fits the clip to the canvas before the Masked Source
(`workflows/masked_job_graph.py::loader`), so in a job-built graph the
box is of canvas-size frames and the whole zoom is enlargement of those:
the plate the model sees is those frames enlarged by that factor, not a
sharper picture. A loader that hands the
clip's own frames gives the box more pixels. The report's line says which,
and the files carry the size as `source_size`. Only the region is laid
back, so the plate's softness is in no delivered pixel; whether it changes
what the model draws is the first render's question.

**What changes meaning.** `grow_pixels` and `feather_pixels` are pixels of
the zoomed picture, so on the whole frame they are divided by the zoom
(the report gives both). A window's region file and a preview's plan are
of the zoomed space and carry the box under `zoom`
(`video_mask.load_window_region`); a reader that lays them or counts
their cells maps through the box or refuses. The mask review of a zoomed
run is of the zoomed picture, and its legend says so.

**Checks.** `bench/check_video_mask.py` item 24 and the zoomed case of its
song loop (both hold the wired video); `bench/check_recomposite_window.py`'s
zoomed window; `bench/check_masked_job.py` for the builder's rule and the
build record's line.

**Dated lines.**

- 2026-10-10, the first plan of a zoomed load (the lead's preview of one
  street shot, a face part, through the job builder; the plan is under
  that job's output folder): with `largest_zoom` at two and at three the
  box came to the same size, set by "the box with its room" and not by
  the cap, well under either setting. The subject crosses the frame
  during the shot, so the union of her part's boxes over the load, with
  `ZOOM_ROOM`, is what limits the zoom: on a subject who travels, the
  setting is a ceiling the load does not reach, and two arms at two
  settings are one arm. The report's line says which of the two set the
  box; read it before queueing a second arm.
- 2026-10-10, the first zoomed render, that shot (one load, one seed):
  it ran clean and wrote its zoomed picture and the box beside it. Read
  from its own files, the plate is the loaded source to the same figure
  as the accepted whole-frame render of the load, outside the box and
  inside it, and the zoomed picture scaled into the box agrees with the
  piece inside the region
  ([`2026-10-10_saved_windows_laid_again`](../../bench/results/2026-10-10_saved_windows_laid_again.md),
  `first_zoomed_render_read_from_its_files`). As seen by the lead in
  stills, not judged on playback: the face is replaced where it shows and
  no edge of the box shows. Whether it reads as the reference at this
  zoom is the owner's to say and is open.
- 2026-10-10, the end of the day: CUT DOWN to what its one render used,
  until the owner rules on that render ([`decisions.md`](decisions.md),
  "What the owner cut from the masked lane's nodes"). Removed: the choice
  "zoomed in on the tracked subject" (a bigger box, less zoom); the input
  `largest_zoom` (a constant now, `video_mask.ZOOM_LARGEST`: on its first
  use the box set the zoom and not the setting); the rules by which a
  zoomed run took the box of a stored window it continued (no render had
  done it; a zoomed run now continues nothing); the cut of a wired motion
  video to the box (refused instead). The lines above that speak of two
  zoomed choices or of a setting are of their hour.
- 2026-10-10, NOT BUILT, the lever that would reach the setting on a
  subject who travels: a box that FOLLOWS the part through the load (a
  smoothed track, cut at sub-pixel positions) in place of one fixed box.
  What it changes: the subject stands still in the zoomed picture and the
  surroundings slide; the files carry a box a frame, so the laying back,
  the delivery tool's contract (`fixed`) and the capture's mapping each
  take a box a frame; a continuation takes the stored window's boxes for
  its kept frames and carries the track on from them. What is not known
  until a render: what the model draws on a plate that slides, and
  whether the drawn part holds still against the head when the track is a
  pixel off. The cheaper things to try first are a smaller `ZOOM_ROOM`
  and the box round the part where it was round the tracked subject.
- 2026-10-13: A ZOOMED RUN TAKES A WIRED MOTION VIDEO, uncut (the owner's
  yes on the Open List's row p04, 2026-10-12, to "the zoomed load that
  keeps its motion video"; built the next day). What came back is not
  what was cut on the 10th: that was a cut of the wired video to the box
  (one render, no difference), and it is still gone. What this is for:
  the frozen form (the body model cut to the head with the face points)
  on faces the whole frame gives too few tokens, which the clip holds on
  the lead's street walk, the second subject's 1010-1038 and the phone
  insets. NOT RENDERED when written. Not built with it: a preview stage
  that runs zoomed, so the part model's and the pose model's readings of
  a small subject are what they were (`next_steps.md` has the case of
  hands read as face), and the capture's reading of a zoomed load.
- 2026-10-10: the softness the report names (the node was handed frames
  already fitted to the canvas, so the whole zoom is enlargement) goes
  away only when the loader hands the clip's own frames, which is the
  third step of the analysis-folder loader (the masking board,
  `build-renders-read-the-analysis-folder`).

## Known limits

Each line names where the evidence is. "Seen" means on a render or a tile.

**Finding the person**

- **It is not an identity model.** SAM 3's trunk says what a thing is, not
  who. People who look alike score alike: on the car clip one shot of the
  lead scores the same as another woman.
  `../../bench/results/2026-10-04_subject_track_three_clips.md`.
- **The rule was chosen on the three clips it passes.** No clip it has not
  seen has been tried. Same record.
- **A miss leaves the original in the shot; a wrong take replaces somebody
  else.** One shot is corrected by hand with the Subject Track's
  `corrections`: `shot 3: person 2` takes that person, `shot 3: none` leaves
  the shot alone, with the shot's number and the person's number read off
  the preview. `subject_track.py`, "A correction";
  `../../bench/check_subject_track.py`, item 7. Checked on stand-ins for
  SAM 3, and run on one real clip, where a missed shot was taken by a typed
  correction: `../../bench/results/2026-10-06_subject_track_defaults.md`,
  "The correction". (This passage said until 2026-10-05 that there was no
  way to correct one shot short of naming a frame or a value for the whole
  clip, and until 2026-10-07 that a correction had not yet run on a clip.)
- **Seen from behind the subject is not found.** A shot in which they never
  face the camera is left alone. `subject_track.py`, "How far the matching
  can be trusted".
- **A change of framing lowers the score of the same person.** The report
  says when a shot is framed much closer or wider than the pick. A crop of
  the head run through the trunk again holds across the one-person clip and
  fails on the car clip; it is not in the node. Same record.
- **In a one-person clip a cutaway to a different lone person is taken.**
  Naming a value for `match` turns that rule off.
- **Cuts.** A dissolve or a jump cut inside a cutaway can score under the
  threshold. The report prints the highest steps with the threshold marked.
- **Cost.** Two detector passes on every frame looked at, and a tracker pass
  per shot, on every run that tracks: nothing keeps a mask between runs
  (2026-10-10). A job pays it once, in its preview, and its renders load
  the mask videos that preview wrote.

**Replacing the person**

- **Movement is not carried.** On the generic prompt the new subject faces
  the camera when the original turns. Starting the sampler part-way into its
  schedule carries the turn and the original's hair and clothes with it;
  doing so from a softened copy of the original turns him and still changes
  his clothes. Neither is a fix.
  `../../bench/results/2026-10-04_masked_v2v_band.md`, "The turn without a
  prompt", and `../../bench/results/2026-10-04_masked_v2v_turn_soft_arms.md`.
- **The hole decides size and place.** The reference still carries no height
  or build. A tight margin keeps the new subject in the original's
  footprint; a wide one lets a differently shaped subject fit and lets him
  stand somewhere else. `../h3_audio_freeze.md` section 4.
- **`head and hair` gives an oversized head on a long-haired original.**
  Kept as an option, not a recommendation. Same section.
- **`paint_out` was judged worse** and stays off. Same section.
- **The mask is the body, not what the body does to the room.** The
  original's shadow and reflections stay.
- **Two-sampler graphs do not carry the mask as wired.**
  `MiniMaxH3RestorePlate` (`plate_restore.py`) exists for it and is in no
  shipped graph.
- **No control adapter.** The owner ruled it out for this lane
  (`decisions.md`, 2026-10-04).
- **The first clip's flicker was set aside**, its dark lines being seams in
  the source's own backdrop. One part of it has no explanation yet.
  `../../bench/results/2026-10-04_masked_v2v_first_run.md` and
  `docs/research/masking/2026-10-04_mrhf.md` (no longer in the tree since 2026-10-11), section 4.

**Using it**

- **Inputs marked advanced are not folded away** in the owner's frontend, so
  both nodes show every input.
- **Core's SAM 3 checkpoint does not load in a plain process outside the
  server**: the text projection's shape is refused (seen by two sessions).
  mrblue's probes leave that one key out of the load in their own process,
  the projection not being used for the conditioning; the scripts are under
  `internal/claude/`, which is not tracked, so a new probe has to repeat it.

## Directions

[`next_steps.md`](next_steps.md) holds the list that is acted on. This is the
wider set, with what each would buy and what is known about it.

**Telling people apart**

- **A model trained for identity**, used where SAM's features are used now:
  on the head, at the seed frames only. It is the real answer to people who
  look alike and to a change of framing. None is wired; a licence check
  comes first.
- **Correcting one shot by hand** is built, without a frontend widget: two
  numbers typed off the preview ("Finding the person" above). What is left
  of it is a person the detector never found, who has no number: core's
  detector takes points and boxes, and nothing here passes them. (Said
  until 2026-10-05 that this needs a frontend widget.)
- **A fourth clip**, chosen before the rule is looked at again.
- **Upstream SAM 3's caller-mask request** (`coderef/sam3`) would replace the
  per-shot seeding with the tracker's own. Not served by the multiplex
  checkpoint and not in core's port.

**The turn, and movement in general**

- What the owner has left to choose: a clause narrowed to facing, which is
  prompt text; or accepting different clothes on the new subject.
- Untried on the late start: a start between the schedule's first two knots,
  more seeds, another clip. The record lists them.
- **The scene's clothes on the new subject**: a crop of the original's torso
  as a second reference. Untried.

**What the body does to the room**

- **Removing the original's shadow and reflections.** Nothing here does it.
  VOID was tested for it on 2026-10-05 and removed:
  `../../bench/results/2026-10-05_void_plate_turn.md`.
- **A soft edge instead of a hard one**: a matting model on the subject.
  The nodes for it exist (`sapiens2_parts.py`, `MiniMaxH3SubjectParts`); the
  parts graph wires the part mask and not the matte, which is in no graph.
  (Said until 2026-10-06 that the nodes were in no shipped graph.) What was read of the model:
  `docs/research/masking/2026-10-04_mrhf.md` (no longer in the tree since 2026-10-11), "Matting".

**The render**

- The mask through two samplers, with the plate restored between them.

## Where to look

| question | where |
|---|---|
| what a node input does | the node's `define_schema` tooltip |
| why a default is what it is | the comment beside the constant, and `../../workflows/h3_config.py` (`SUBJECT_TRACK`, `MASKED_SOURCE`) |
| what would go red | `bench/check_subject_track.py`, `bench/check_video_mask.py`, `bench/check_masked_prompt.py`, `bench/check_plate_restore.py`; [`../checks.md`](../checks.md) |
| what the owner said of a render | `../../bench/results/2026-10-04_masked_v2v_first_run.md`, `../../bench/results/2026-10-04_masked_v2v_band.md` |
| how the prompts are written | `../../masked_prompt_text.py` (the node's sentences), `../../prompt_bank/` (`ref2va_masked_*`), [`../prompting.md`](../prompting.md) |
| what was measured on which clip | `../../bench/results/2026-10-04_subject_track_three_clips.md` |
| what the tracker and the masks did on a run | "Seeing what the tracker and the masks did" above |

## How this page grew

Reorganized on 2026-10-12, in one commit. The order until then, as the
page had grown: "What it is"; "Running a job"; "Reading one window whole";
"The rule: data before a render, and the same data after"; "Rules of the
lane"; "The pieces, in the order a render meets them" (with "Nothing is
kept between runs by a key" and "Say the least first" inside it); "A latent
step is several frames"; "The motion video as a body model, with marks
drawn on its face"; "A small subject, rendered zoomed in"; "Known limits";
"Directions"; "Seeing what the tracker and the masks did"; "Where to look".
No heading another file cites changed its words. What was cut, moved and
corrected is in [`decisions.md`](decisions.md) under 2026-10-12, and the
table of the clips and files that were opened is
[`2026-10-12_masked_v2v_page_held_to_its_clips.md`](../../bench/results/2026-10-12_masked_v2v_page_held_to_its_clips.md).

2026-10-13: "Who is who: a session reads first, the owner answers, and
both are kept" added after "Seeing what the tracker and the masks did", at
the lead's order, from the day's first batch of picks kept against answers.

2026-10-13, the evening: "Two neighbouring shots in one load: a tool, not
a default" added after "Running a job", when the tool was built at the
owner's word; three dated notes where the page had said such a load could
not be built.

The header's list of edits as it stood before that day:

2026-10-11 (at the close, how to read the section on the body model with marks: patterns with their conditions, not a recipe; later the same evening, a dated note that the practice did not repeat on a second shot, and the marks tool named where a one-off was; the section "The motion video as a body model, with marks drawn on its face": what the reference video was, how it was made, the lines of text for it and one whole text as an example; with a dated note under `motion_video` and under the prompt node); 2026-10-10 (a section on latent steps, the grid a load fixes, and cuts; a dated note under `keep`; `MiniMaxH3SubjectBoxes` named under `motion_video`; the rule "data before a render, and the same data after"; "Seeing what the tracker and the masks did" with the capture tool; "Say the least first" under the prompt; a setting the Masked Source refuses is refused at queue time; `motion_video`); 2026-10-09 (the `edge` input; a dated note on what "clean" means in a kept token, and a pointer to the upstream cross-check); 2026-10-06 (a loss inside a shot is searched and `subject_from`; the prompt node; the review and parts graphs); 2026-10-05 (the ref2va motion graph; the Sapiens2 nodes named); 2026-10-04 (first written, after the Subject Track's third clip)
