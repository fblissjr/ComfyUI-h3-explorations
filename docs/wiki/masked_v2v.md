# Masked video to video: how it works, what it cannot do, where to go next

last updated: 2026-10-10 (the rule "data before a render, and the same data after"; "Seeing what the tracker and the masks did" with the capture tool; "Say the least first" under the prompt; a setting the Masked Source refuses is refused at queue time); 2026-10-09 (the `edge` input; a dated note on what "clean" means in a kept token, and a pointer to the upstream cross-check); 2026-10-06 (a loss inside a shot is searched and `subject_from`; the prompt node; the review and parts graphs); 2026-10-05 (the ref2va motion graph; the Sapiens2 nodes named); 2026-10-04 (first written, after the Subject Track's third clip)

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
  not searched.
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
- **`replace`.** `whole subject`, or `head and hair`, which keeps the body's
  pixels and finds the part with SAM 3 from `part_phrases`.
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
something on frames its subject has no mask on, far from its subject, far
more than it usually does, or the same pixels as another pass. Given capture
folders (`--capture`) it settles shared pixels by whose mask they lie in
before falling back on the table's order, and writes its table and flags
into them. Judge a render's colour on this file: it is BT.709 and says so,
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
