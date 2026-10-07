# Masked video to video: how it works, what it cannot do, where to go next

last updated: 2026-10-06 (a loss inside a shot is searched and `subject_from`; the prompt node; the review and parts graphs); 2026-10-05 (the ref2va motion graph; the Sapiens2 nodes named); 2026-10-04 (first written, after the Subject Track's third clip)

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
back into the kept tokens every step.

## The pieces, in the order a render meets them

### 1. `MiniMaxH3SubjectTrack` (`subject_track.py`): which pixels are the person

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
  phrase). On a named frame it is applied there. Left automatic,
  `main_subject` takes the person the rule favours for most of the clip's
  frames.
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
  other shots and the frames the tiles show do not move.
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
  the settings that do not change the mask.

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
  per shot. The kept mask removes it from every later run.

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
- The kept mask across a server restart, which no run has exercised.

## Where to look

| question | where |
|---|---|
| what a node input does | the node's `define_schema` tooltip |
| why a default is what it is | the comment beside the constant, and `../../workflows/h3_config.py` (`SUBJECT_TRACK`, `MASKED_SOURCE`) |
| what would go red | `bench/check_subject_track.py`, `bench/check_video_mask.py`, `bench/check_mask_store.py`, `bench/check_masked_prompt.py`, `bench/check_plate_restore.py`; [`../checks.md`](../checks.md) |
| what the owner said of a render | `../../bench/results/2026-10-04_masked_v2v_first_run.md`, `../../bench/results/2026-10-04_masked_v2v_band.md` |
| how the prompts are written | `../../masked_prompt_text.py` (the node's sentences), `../../prompt_bank/` (`ref2va_masked_*`), [`../prompting.md`](../prompting.md) |
| what was measured on which clip | `../../bench/results/2026-10-04_subject_track_three_clips.md` |
