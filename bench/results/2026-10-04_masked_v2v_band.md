# Masked video to video: a subject among other people

last updated: 2026-10-04 (later arms added the same day)

**Status: rendered, not judged.** Read from a one-frame-per-second sheet and
three single frames; lip sync, edges at full size and the audio are the
owner's to judge on playback.

## What ran

`bench/masked_v2v_band_arms.json`: 1:35 to 2:07 of a six-person band video
with coloured lighting changes and cutaways, the lead singer replaced from
the same reference still as `2026-10-04_masked_v2v_first_run.md`, on the
PDD8 song chain with the audio frozen hard. Prompt
`ref2va_masked_lead_swap`: the clip-agnostic prompt with the claim that the
subject is alone removed. Rows: `2026-10-04_masked_v2v_band_arms.jsonl`.
The clip is under `Video/mrpink/` on the output share.

## Isolating one person among several

Tracker probes on the span, each read from an overlay sheet or a per-object
table (presence, size and horizontal position per shot):

| phrase | what it marked |
|---|---|
| "person", object cap raised | everyone, and the cap was spent before the main shot: the lead is untracked from the neon take on |
| "man with long hair" | the whole front row and the men in the cutaways |
| "lead singer" | only the centre singer in the group shots, and the most prominent person in each cutaway |

"lead singer" broken into its objects gives the lead as three of them, one
per shot he is in, with the whole neon take a single object present in every
frame. Selecting those three by index is what the arm does. Another member
is re-identified across a cut as one object, so identity across cuts is not
simply lost; it is unreliable.

**The indices are bookkeeping for one clip.** They hold for this loader
start, frame cap and phrase, and a different clip needs the probe again.

## What the frames show

- The lead is replaced in the warm group shot and through the neon take, at
  the original's place in the frame, and the others are untouched where he
  does not cover them.
- **The light follows.** Under the green and purple washes his skin and
  shirt take the room's colour.
- In the push-in he stands with hands on hips as the two beside him do.
- The cutaways and the opening two-shot are untouched: nothing was masked
  there.
- **The shot from behind is wrong.** The group turns its back; he faces the
  camera. The prompt names no action, and nothing told him to turn.
- He wears the still's T-shirt and cap among five green sweatshirts. The
  prompt asked for that; it is a choice, not a failure.

## Later the same day: the turn, the clothes, and what `replace` can do

All on the cleaned still (`f_img_outside_nomark.png`; the first still carried
a watermark sparkle that the model painted on the shirt as a bright dot, and
the owner withdrew it, so `band_generic_32s` cannot be re-run as written).
Read from sheets and single frames; the owner's words are quoted.

| arm | what changed | what the frames show |
|---|---|---|
| a third-window prompt block (`ref2va_masked_lead_turns_away`), windows 1 and 2 reused; run from `internal/`, not in the manifest | the prompt names the turn | he turns away with the group, a little after the original does. It works and it is specific prompting, which the owner does not want as the answer |
| head and hair, outline only (interrupted after one window) | `replace` = head and hair, the region the head-and-hair outline | sweatshirt, its print and the neighbours intact, the hair gone from the chest; the head about half as wide again as the original's, on the collar with no neck. Owner: "a giant bobblehead. doesnt work" |
| `band_head_w1` | the region extended down to where the hair ends, so neck and shoulders are drawn too | the same oversized head. The part detection also cost several minutes before sampling (`per_node_s` in the rows: the Masked Source node) |
| `band_follows_32s` | whole subject; the prompt ties his facing, movement and clothes to the people beside him (`ref2va_masked_lead_follows`) | he turns with the group at the right moment, and also turns his back twice when nobody else does; a plain grey-green T-shirt for most of the clip and a green long-sleeved top only from behind. No dot on the shirt |

**What this says.** The model frames what it paints to the hole: a
whole-body hole gives a person in proportion, a head-and-chest hole gives a
head-and-shoulders portrait that fills it. Nothing in the per-token-timestep
method anchors scale, so head-only replacement on a long-haired original is
not reachable by shaping the region. And a prompt cannot reliably carry the
original's movement: naming the shot works, a generic clause over-applies.
Not tried: the source as a motion reference, which the model is trained for
and which pays the reference's rows on every step.

## The margin, the pose and the member nobody can see

`band_changed_w1` (the composite that keeps only what changed, the mask grown
twice as far, the cleaned still) against `band_generic_32s` (whole region,
the default margin, the old still), stacked with the source as
`compare_band_w1_stacked.mp4` on the output share. The owner's reading, from
their annotated frames:

- Both renders keep the visible band members "mostly the same", with changes
  too small to notice unless the clips are stacked.
- At the wider margin the replaced lead stands differently, and that uncovers
  the member behind him on the left, whom the source hides behind the
  original's height and long hair. The model invents him. At the default
  margin a sliver of an invented face shows in the same place.
- "The invented person on the left in the back keeps the same identity
  throughout."

**Why the pose differs** (inferred, endorsed by the owner, and confirmed below):
the only thing telling the model where the original stood is the shape of the hole, so a wider hole lets him stand somewhere else. Two things differ between those renders, the margin and the
still, and either changes the sample; `band_changed32_w1` puts only the
margin back and says which. The composite cannot be the cause: it runs after
sampling. The still is a close selfie and carries no height or build, so the
subject's size and place come from the hole and the scene.

**Confirmed by `band_changed32_w1`** (the same window, seed, still and
composite, the margin alone put back to the default; stacked with the wider
one as `compare_band_w1_margin.mp4`): at the default margin he stands in the
original's footprint and the member behind him stays hidden. The owner, on
that pair: the wider margin "preserved identity better". So the same control
trades likeness against place: a tight hole makes the new subject take the
original's outline, a wide one lets him keep his own proportions and stand
where they put him. One seed each.

**What it means for the method.** Replacing a subject with a smaller outline
uncovers space the source never shows, whatever the margin. The margin sets
how much: tight keeps the new subject in the original's footprint, wide gives
a differently shaped subject room and exposes more. Nothing is built for it.

## The shipped graph, version one

`band_subject_w1`, then the same arm at thirty-two seconds (the label kept so
the first window was reused): the shipped graph after it moved to
`MiniMaxH3SubjectTrack`. One pick, the largest `person` on a frame of the
warm group shot; no object indices. Stacked against the source as
`compare_band_32s_v1.mp4`, and its first window against the typed-index
render as `compare_band_w1_subject_node.mp4`.

- The tracker's report on this clip: every cut found, the three shots the
  lead is in taken, the rest absent, in under a minute for the clip
  (`per_node_s`, node 105). At its first default threshold it also took one
  cutaway; the default was then set from this run's numbers
  (`subject_track.py::MATCH_THRESHOLD` says which).
- (seen) The lead replaced in every shot he is in, the neighbours intact.
- The owner: "looks pretty good", and "one problem: he doesnt turn around at
  the end". That is the movement limit recorded above, on the generic prompt.
- The kept mask was written on the first run. The second run did not track
  again, but inside one server session core's own cache explains that; a
  restart between runs is the test of the kept mask and has not been done.

## Chains compared on one window

Same window, seed, still, margin and composite; stacked as
`compare_band_w1_chains.mp4` and `compare_band_w1_sol.mp4`. Times are in the
rows.

| arm | chain | the owner |
|---|---|---|
| `band_changed_w1` | PDD8 baked, Sol-Attn | "worked best" |
| `band_baseline_w1` | undistilled, base step count, stock attention, `er_sde` | worse likeness; "er_sde inserts noise. thats why"; "not worth the long render time" |
| `band_changed_dense_w1` | PDD8 baked, Sol's window closed so every step is the dense int8 fallback | "no sol changed the clothes" |

## The tracker with nothing set

`band_auto_w1`: `band_subject_w1` with no tracker setting at all, after the
Subject Track took its pick frame, its match and its cut threshold
automatically (a peer's rework, validated by a second peer outside the
server on this clip and the first one). This is the run through a server
and the shipped graph.

- Mask-only probes first, both clips, every setting automatic. Band: every
  cut found, the lead taken in the three shots he is in, the six other shots
  absent. Solo: every cut found, the singer taken in all seven shots. The
  thresholds, the pick frames and the per-shot scores the node reported are
  the ones both peers recorded from their own runs. Seconds: `per_node_s`, node
  105.
- The window: the song node's report gives the same share of regenerated
  tokens as `band_subject_w1`, and the Masked Source the same share of the
  frame touched over the clip, to the digit each prints.
- (seen, one frame) The lead replaced, the neighbours intact. The sample is
  not the earlier one: the tracker now starts from a different frame, so the
  mask differs by a few pixels and he stands a little further right, with
  the invented member behind him showing. Stacked as
  `compare_band_w1_auto.mp4`. Not judged by the owner.

## The turn without a prompt: a late start

The owner, on version one: "one problem: he doesnt turn around at the end",
and "i really dont wanna prompt turns around". A peer session's suggestion:
start the sampler a little way into its schedule, so some of the source
survives under the mask as a pose hint. One window from 1:52, which holds the
shot where the group turns away; the generic group prompt, the same seed, the
shipped graph and two copies of it with core's `SplitSigmas` dropping the
first knots of the distill's schedule. Rows `turn_control`, `turn_late1`,
`turn_late2`; clips `masked_v2v_turn_turn_*` on the share. Read from frame
tiles across the turn and one frame early in the window.

| arm | knots dropped | the turn | who he is |
|---|---|---|---|
| `turn_control` | none | faces the camera while the others turn to the wall | the reference: cap, short hair, T-shirt |
| `turn_late1` | one | turns with the group, at the original's moment | a hybrid: the reference's cap and roughly its face, the original's long hair, a hooded sweatshirt that is neither's |
| `turn_late2` | two | turns with the group, at the original's moment | the original: long hair and the green sweatshirt. The owner: "he turns ... and has the hair and clothes from the original guy lol. but he does turn" |

- **A late start carries the pose and the original with it.** On this
  schedule there is no knot that follows the movement and keeps the
  reference: one knot already brings the hair back and loses the clothes.
  One window, one seed. A third arm at three knots was cancelled after the
  second.
- What it does establish: the pose can be carried with no prompt text, so
  the question is how to let the coarse shape through without the look. The
  peer's follow-up, untried: degrade the source inside the region (a hard
  blur, no colour) before the late start, so facing survives and hair and
  clothes do not.
- The late arms were also quicker, since they sample fewer steps
  (`per_node_s`). The second and third arms did not run the tracker or the
  Masked Source again; inside one server session core's cache explains that.
- A peer looked at the control shot and withdrew a second idea, keeping the
  original's body below a row: the frame ends at his thighs for the whole
  shot, so nothing that would be kept carries facing.

## Not established

- **How to carry the original's movement with no prompt text and keep the
  reference's look.** Open; the section above is where it stands.
- Whether the kept mask is read back after a server restart. Every reuse seen
  today was inside one session, where core's cache would do the same.

- Whether the regenerated tokens beside him alter the neighbours they
  overlap. A peer drew up a design for protecting them.
- Whether the faint remnant of the original subject a peer measured in the
  first clip's flicker shot is here too.
