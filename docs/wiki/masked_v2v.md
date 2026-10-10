# Masked video to video: how it works, what it cannot do, where to go next

last updated: 2026-10-10 (a section on latent steps, the grid a load fixes, and cuts; a dated note under `keep`; `MiniMaxH3SubjectBoxes` named under `motion_video`; the rule "data before a render, and the same data after"; "Seeing what the tracker and the masks did" with the capture tool; "Say the least first" under the prompt; a setting the Masked Source refuses is refused at queue time; `motion_video`); 2026-10-09 (the `edge` input; a dated note on what "clean" means in a kept token, and a pointer to the upstream cross-check); 2026-10-06 (a loss inside a shot is searched and `subject_from`; the prompt node; the review and parts graphs); 2026-10-05 (the ref2va motion graph; the Sapiens2 nodes named); 2026-10-04 (first written, after the Subject Track's third clip)

Written by hand. This is the lane's map for a reader who has not followed
it: the pieces in the order a render meets them, the limits each one has
shown, and the directions worth trying. It carries no measurement. A value
is named by the constant that holds it, a result by the dated record that
holds it, and where this page and the code disagree the code is right.

The dated account, with the owner's words, is
[`../h3_audio_freeze.md`](../h3_audio_freeze.md) section 4. What was
decided is in [`decisions.md`](decisions.md) under 2026-10-04. Each session's
working notes are in [`../research/masking/`](../research/masking/README.md).

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
Two more graphs since 2026-10-06, neither rendered yet:

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
  (`Video/compare_stacked/body_345_p7_typed_beside_node_over_mask.mp4`).
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
  area with no margin at all. Nothing has rendered with it;
  `bench/check_video_mask.py` item 15 holds that it loses no subject pixel and
  moves no label.
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
- **The kept mask.** `mask_store.py` keeps the finished mask on disk, keyed
  on the upstream graph, a fingerprint of the frames, the input files'
  stats and every upstream node's `MASK_VERSION` (`mask_key`). On a hit the
  Masked Source never asks
  for its `mask` input, so the tracker does not run. `MASK_KEY_SKIP` names
  the settings that do not change the mask. **Disabled in code since
  2026-10-07** (`video_mask.MASK_REUSE_ENABLED` is False; `reuse_mask` on
  the node defaults to off and cannot turn it on): every run tracks afresh.
  See "What is kept between runs, and what can go stale" below.

**What is kept between runs, and what can go stale.** Three things are
written by one run and read by a later one, and a fourth lives in memory:

| what | where | its key or version | reused today |
|---|---|---|---|
| the finished mask | `output/masks/` (`mask_store.py`) | the queued graph, the frames, the files' stats, each upstream node's `MASK_VERSION` | no: `video_mask.MASK_REUSE_ENABLED` |
| a rendered window | beside the render, `<prefix>_windows/` (`loop_resume.py`) | the queued graph upstream of the song node, the track, the window's text, length, start and seed, the window before it | no: `audio_freeze_song.WINDOW_REUSE_ENABLED` |
| the shot table | `<prefix>_NNNNN_shots.json`, and the Subject Track's `shot_table` output | `TABLE_VERSION`, the file's format | read only when wired into `subject_from` |
| a window's source encode and prompt encode | the server's memory (`window_keep.py`) | the identity of the live objects they were made from | yes, under `reuse_windows`; a restart clears it |

**None of the first three keys holds anything derived from this pack's
code.** A result made before a change to how a subject is followed, or to
anything else the key does not name, is handed back after it when the
settings are the same; the only guard is a number somebody has to remember
to bump. That happened on 2026-10-07 (`decisions.md`), and the owner had
both disk stores switched off in code that evening. The two inputs stay on
their nodes so saved graphs still load, and their tooltips say they have no
effect on what is stored. To turn them back on: a fingerprint of the pack's
code in each key (and in the shot table), and a check in the sweep that
every key that reaches disk carries it. Not built.
`bench/check_mask_store.py` holds the shipped state of both switches and
still exercises the mask store with its switch on.

**Three things called a version, and none is the other.** The pack's
version is the newest heading in `CHANGELOG.md`, assigned by
`bench/build_changelog.py` from fragments (semver; `pyproject.toml`'s
`version` is what ComfyUI's registry reads and is kept equal to it by the
same build). `MASK_VERSION` is not a version of a node: it is one integer
per node class that goes into the kept mask's key, bumped only when that
node's code would make a different mask from the same inputs, so the
numbers on different nodes (`subject_track.py`, `sapiens2_parts.py`,
`video_mask.py`) are unrelated and never compared; it leaves with the kept
mask (card `build-remove-kept-mask`). `TABLE_VERSION` (`shot_table.py`) is
the shot table's file format, checked on read. A node's identity is its
`node_id` in `bench/node_id_manifest.json`, which has no number at all.

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

**Say the least first, and add a sentence only when a render asks for it**
(the owner, 2026-10-10: "let the model guide and infer until you need to get
more specific", "just like progressive disclosure"). It is the rule above
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
2. **Where the subject is, and what on them is the scene's own** (the hair
   or clothing the plate keeps), in a sentence.
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
  `source` is wired, the first time; the mask is kept and the render that
  follows tracks nothing. `masked_prompt.py` says why.
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
  `loop_plan.split_steps` lists the steps a load splits and
  `loop_plan.first_frame_choices` orders every start one cycle back by how
  many frames it lays across a cut (`bench/check_audio_freeze.py`, the
  step grid case). All of a load's cuts cannot be cleared in general: a
  cut is on an edge for as many starts as the cycle has steps, out of as
  many as it has frames, so the chance that one start clears several cuts
  falls with each cut. On the first clip's whole-subject piece no start
  cleared every cut the subject is beside; the function's output on that
  check's cuts and shots is the record.

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
| `changed_alpha`'s hold | the maximum over the step of "changed, or the old subject stood here" | the far frames differ from the source over the whole region, so the subject's own frames keep the whole region and less of the margin is restored (measured, `bench/results/2026-10-10_saved_windows_laid_again.json`) | nothing; `data/CAPTURE_GAPS.md` U6 |
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
  per shot. The kept mask removed it from every later run; with the kept
  mask disabled in code (2026-10-07) every run pays it.

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
  `../research/masking/2026-10-04_mrhf.md`, section 4.

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
  `../research/masking/2026-10-04_mrhf.md`, "Matting".

**The render**

- The mask through two samplers, with the plate restored between them.
- The kept mask across a server restart, which no run has exercised (and
  none will while it is disabled in code).

## Seeing what the tracker and the masks did

Written 2026-10-10, after a session had to be told this tooling existed. It
is in four places and none of them is one command. Read this before writing
a probe.

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
the per-frame table of what each render changed: a pass that redrew
something on frames its subject has no mask on, far from its subject, with
an area that steps at some frame (a cut, a change of framing, or another
person), a frame or two just across a cut of the source and no further (a
pass whose region ran over the cut; found from the source's own
frame-to-frame change, with no capture), or the same pixels as another
pass. Cut a render's rows on the source's cuts. Given capture
folders (`--capture`) it settles shared pixels by whose they are: of the
rows that changed a pixel, those whose subject holds it can take it and the
latest of them does, so a render and a patch of it are settled by the
table's order between themselves and neither loses its own subject to
another person's pass; a row whose subject is not known is never ruled out
by a mask. A patch has no run in a capture, so its row says whose it is
(`subject=<label>`). It writes its table and flags into the capture
folders. Two of its proofs exist because a file passed without them: every
row must show on the pixels it was meant to (`rows_shown`), and no small
square of a decoded frame may sit far from the frame the table makes, which
is what fails a file built from another table or by an older rule. Where two tracked masks claim the same pixel (an arm reaching
across somebody) a mask does not say whose it is: the capture's
`owners.npz` does, from the class maps, and the assembler reads it for the
shared pixels and for `restore=<subject>`, leaving what it marks contested
to the later row. Its record also says what the file did to every subject a
capture knows (`subjects`: tracked pixels off the source, in all and by
class, inside the track and outside it, beside the floor), and
`--compare A.check.json B.check.json` prints two records side by side: the
same stretch with and without a restore is the before and after a "by
class" question asks for. How a region SITS in its plate (its detail, grain,
tone and cast against the plate just outside it and against what stood
there before) is `bench/region_against_plate.py`, which needs no mask
either; run it before tuning a look by eye. With `--size source` the file is at the source's own size: every
frame is the source's picture, never scaled, and only what a render changed
is scaled up and put back over it, so nothing outside the regenerated region
is resampled and the rows the loader's crop dropped are kept. A row can
give a class of a subject back to the source (`restore=<subject>.<Class>`,
read from the capture's class map): a thing inside a region that should not
have been redrawn, an earring inside a face, is put back here and not by
asking the sampler to keep it, which changes what the sampler draws beside
it. With no class (`restore=<subject>`) a row gives back the whole of
another subject: wherever that subject's tracked mask is and the piece's own
subject's is not, so a pass on one person never shows what it changed of
another; `restore=<subject>:whole` takes nothing out, for a pass that has no
business inside the other's mask whoever is in front. Judge a
render's colour on this file: it is BT.709 and says so,
as renders are since the song node's writer converts and tags, and a render
written before that plays off in a player that guesses the matrix from the
size. Its docstring is the account; `bench/check_assemble_delivery.py` is
its check.

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
  given ("The song node" above).
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
Masked Source's `others` when the two stand close. None of the captures or
tools above carries a field for which person a file is about.

## Where to look

| question | where |
|---|---|
| what a node input does | the node's `define_schema` tooltip |
| why a default is what it is | the comment beside the constant, and `../../workflows/h3_config.py` (`SUBJECT_TRACK`, `MASKED_SOURCE`) |
| what would go red | `bench/check_subject_track.py`, `bench/check_video_mask.py`, `bench/check_mask_store.py`, `bench/check_masked_prompt.py`, `bench/check_plate_restore.py`; [`../checks.md`](../checks.md) |
| what the owner said of a render | `../../bench/results/2026-10-04_masked_v2v_first_run.md`, `../../bench/results/2026-10-04_masked_v2v_band.md` |
| how the prompts are written | `../../masked_prompt_text.py` (the node's sentences), `../../prompt_bank/` (`ref2va_masked_*`), [`../prompting.md`](../prompting.md) |
| what was measured on which clip | `../../bench/results/2026-10-04_subject_track_three_clips.md` |
| what the tracker and the masks did on a run | "Seeing what the tracker and the masks did" above |
