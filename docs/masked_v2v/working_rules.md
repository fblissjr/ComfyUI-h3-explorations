# Working rules of the lane: data first, one window read whole

The lane's settled rules are [`data/AGENTS.md`](../../data/AGENTS.md); each video's state is its own
`data/<video>/AGENTS.md`. This page is the lane's working habits and the reasoning behind them, each earned on a day
when a render was spent finding out what a few minutes of looking would have said. Where a habit is a settled rule,
it links there and does not restate it.

## The rule: data before a render, and the same data after

Set by the owner on 2026-10-10 for every session on this lane. It is a rule, not advice, because the lane's earlier
jobs explained failures by a session's reading that the next render refuted.

1. **Before a render: capture, flags, a look, then the render.** A no-sampling run saves what every tracker and part
   node made, for every person and object in the load (`bench/capture_masked_run.py`; the files are in
   [`running_a_job.md`](running_a_job.md), "Seeing what the tracker and the masks did"). Its `preflight` lists what is
   at risk, each flag with its reason and its frames: a person taken in a shot they are not in, a part mask off its
   person, one person's region over another, things inside a region that are not its subject, a text that disagrees
   with a measured fact. The flagged frames are looked at. The render then loads the masks that were checked and does
   not track again. Nothing renders on a guess about what the tracker took.
2. **After a render: the capture is read first.** When a render is wrong, the first step is a lookup, not a probe: on
   the frames in question, was the mask on the right person, what else was inside the region, what was the model
   shown as movement and as text, was there a voice. A new probe comes only after the tables have been read and found
   silent.
3. **A "why" cites a row, a frame range or a measured figure.** A cause stated without one is labelled a reading, and
   gets a control before anything is built on it.
4. **Every flag gets its outcome, and every surprise becomes a flag or a gap.** What happened on the flagged frames is
   written beside the flag (`bench/capture_masked_run.py outcome`), so a threshold's provenance can move from reasoned
   to measured. A fault no flag predicted is a new rule in the preflight, or an entry in `data/CAPTURE_GAPS.md`. The
   faults found after renders, set against what said so before, are [`known_before_a_render.md`](known_before_a_render.md).
5. **Captured data lives under `data/`**, which is not tracked. A tool worth running twice lives in the tracked tree
   with a check, never in a scratchpad.

What the upfront data can and cannot say: it says where and why a render is at risk. It does not say what the model
will draw inside the region; that still takes a render, and the flags say which shot to render first and which frames
to look at.

## Reading one window whole: its text, its frames, its mask, its audio

A habit and not a gate: nothing checks that it was done. On 2026-10-10 a wrong render was put down to a signal four
times, renders were spent, and the cause was in the text each time; the other sessions had never seen the texts, only
one session's description of them
([`2026-10-10_span_fun-three-people.md`](../research/postmortems/2026-10-10_span_fun-three-people.md), section 1.1).

**When.** Before building on a load somebody else made, or you made on another day. And first, before any test
render, when a render is wrong in a way that is not obvious.

**What.** Take ONE window of ONE load and put four things side by side for the same frames. Assume one of them is out
of step until you have looked.

| what | where it is |
|---|---|
| **The text**, whole, as the model gets it | `bench/what_h3_got.py` on the queued graph; the look's strip draws it under the frames (`bench/capture_masked_run.py video <capture> --source CLIP --job JOB.json --load LOAD`). Read the text itself; hand another session the file, never a sentence about it; when a cause is not obvious, show the owner the whole text |
| **The window's frames**: which source frames, which are held, which share a latent step | the raw frames by number in [`contact_sheets/`](../../contact_sheets/AGENTS.md) (`bench/contact_sheets.py find <video> <frame>`): open a sheet before sampling the video again. Which of them a window gets: the song node's `<name>_plan.json` and each window's `<name>_window_N_planned_region.npz`; why a frame can be drawn with no mask of its own: [`pieces.md`](pieces.md), "A latent step is several frames, and its grid is fixed for a whole load". For a finished render, `bench/recomposite_window.py --read-files` reads the window's own files with nothing decoded |
| **The mask** on those frames, and whose it is | the capture's stacked, frame-numbered video (`bench/capture_masked_run.py video`), its per-frame tables, and the region each window was really given, `<name>_window_N_region.npz` |
| **The audio** the load was given, and whether a voice is on those frames | the capture's voice table, and `bench/capture_masked_run.py shots --loads`, which says how many of a load's frames have a voice and where the source's mouth is open with none |

**What it is for.** Every fault of 2026-10-10 that cost a render was two of these four disagreeing: a text written
from one shot on a load of another; a frame drawn with no mask of its own; one sample of audio deciding a frame; a
text saying the scene is quiet over a song. Each was gated on its own and passed.

**Not built.** One command that shows the four together for a window. Who is in each shot is
`data/<video>/shots.json`, answered on the page `bench/who_page.py` makes ([`who_is_who.md`](who_is_who.md)).

## Rules of the lane

- **A masked render is run on data, before and after**: the section above.
- **Preprocessing data is looked at the moment it is made**, by the tool's `verify` and by a reader who did not make
  it (a subagent counts): the mask on the right subject on every frame, the part, the motion video, each file's frame
  offset. A passing check says no rule failed, never that the data is right; a capture records who made it and who
  looked (`bench/capture_masked_run.py looked`).
- **A scene's text is data, served by whatever plans the shots**, never stored verbatim in code or in a workflow. It
  is written from the frames of one window and is true of that window only. What it may say is
  [`data/AGENTS.md`](../../data/AGENTS.md), "Settled for the masked lane"; how it is made is [`text.md`](text.md).
  Know which text a load gets, read it whole against that load's frames before it is used, and check after the
  render that it did what was expected.
- **When a render is wrong, read four things together before anything is suspected or queued**: its text, its mask,
  and how the four timelines line up (the input video's frames, the mask's frames, the sampler's frames and the
  audio). By the owner's count it is one of those about half the time; on 2026-10-10 three renders went to a kernel
  setting and an audio file before the text was read, and one sentence fixed it. "Reading one window whole", above,
  is where each of the four is.
- **A graph posted by hand is a temporary plan**: whoever posts one reads the full text first and writes beside the
  job which built graph it came from, the one thing changed, by whom, and what the render is meant to decide (the
  job's `note` holds the same for a built load).
- **The reference for a face, head or hair replacement** is settled in [`data/AGENTS.md`](../../data/AGENTS.md) (the
  owner, 2026-10-12, after the two cropped-reference tests on 1039-1060; made with `bench/head_reference.py`). On the
  lead's three hall shots that evening the owner judged it better than the lid marks on playback (the eyes on 1222, the
  mouth on 1156, the gaze on 1093; `data/fun/verdicts/verdicts.jsonl`, 21:25-21:28Z), where the readings from frames
  had called the eyes a draw: what shows in motion is the owner's to see. A face too small or too turned for the drawn
  lines (the lead on 604-948, 65 px, 379 of 406 frames under the drawing line) takes the plain body model, and the note
  says why. What the form
  carried on the shots it was judged on is [`renders_by_form.md`](renders_by_form.md).
- **A rule that depends on the picture takes its value from the input video, up front** (root
  [`AGENTS.md`](../../AGENTS.md); `subject_track.auto_cuts`, `auto_match`): a constant measured on another clip is a
  marked starting value, and the tool says when this clip's own figures disagree with it.
- **Nothing an earlier run made is reused by a key** ([`pieces.md`](pieces.md), "Nothing is kept between runs by a
  key"): a mask is preprocessing data for the job that reads it, rebuilt when preprocessing runs again.
- **The canvas follows the source's shape, and every mask is made at it.** A job is made on
  `h3_config.MASKED_JOB_CANVAS`, the 4:3 row of [`docs/h3/geometry.md`](../h3/geometry.md), because the video is
  4:3 (`ffprobe` on its working copy) and the whole frame then fits with nothing cropped or padded. Every preview,
  track and mask is made at the job's canvas (`capture_masked_run.py verify`, `inputs_at_the_canvas`), so a change of
  canvas is every preview made again. A load rendered zoomed in samples the same canvas whatever its box: it buys size
  for the subject, not time. Settled with the owner on 2026-10-13 ([`docs/state/decisions.md`](../state/decisions.md)).
- **One shot is ONE load a subject, and one window wherever `h3_rules.MAX_LENGTH` holds it.** A shot is a cut ("shot"
  is H3's own word for one, and the word a text uses for more than one: `[Shot N]` in
  [`docs/prompting/prompting.md`](../prompting/prompting.md)). A shot is never split into separate fresh loads
  ([`docs/state/decisions.md`](../state/decisions.md), 2026-10-15, "One shot is never split"): frames inside it where
  the subject's face is hidden or her part is empty stay inside the same load. A shot with two subjects is one load
  for each, each from the shot's first frame to its last. A load one window holds is given the shortest length of the
  grid that holds it (`masked_job_graph.window_for`, `loop_plan.stands_alone`), and a load may name a `window` of its
  own; a longer load runs in chained windows, each on `h3_rules.chain_lengths()`, counted from the load's first frame
  (shorter windows inside one load are allowed on that grid, the owner, 2026-10-15). Two neighbouring shots in one
  load is a tool reached for on a person's word, never a default ([`running_a_job.md`](running_a_job.md)).
- **What a window's length costs and allows.** H3 takes every frame count on its grid (`h3_rules.snap_length`, from
  `h3_rules.FRAME_FACTOR` and `h3_rules.FRAME_REMAINDER`), from core's floor up to `h3_rules.MAX_LENGTH`. The chain's
  lengths (`loop_plan.CHAIN_LENGTHS`, which is `h3_rules.chain_lengths()`) are the lengths a window that meets another
  at a seam may take, not a limit on a window that stands alone. The only reason to sample more frames than a load's
  own is an overlap buffer for a join; a shot bounded by a cut on both sides needs none, and the few frames a load
  holds past its end at its fitted length are the grid's rounding. Every frame added raises what attention costs, and
  with enough references a long window can run out of memory. Not every count on the grid is exact on the audio
  clock (`h3_rules.on_both_clocks`), and it does not have to be for a window that stands alone: the audio clock is a
  rule of the seam between two windows (`audio_freeze.window_geometry`); core rounds the audio latent's count and the
  lane cuts exactly that many audio steps (`audio_freeze.slice_window`).
- **At a short window Sol-Attn may decline a call** under its own least tokens (`sol_attn_h3._ineligible`); the load is
  then sampled by the model's plain attention, which is fine (the owner, 2026-10-13: Sol's sparsity can only take
  quality away). The server's log says which once for each shape of call (`dense (...): seq N < ...`, with the node's
  `verbose` on); it is read on each short load and written in its item.
- **What is on a frame of an input video is already in `contact_sheets/`** (`bench/contact_sheets.py find <video>
  <frame>`): open a sheet before sampling a video again.
