# Known limits and directions

What the lane cannot do today, each with where the evidence is, and what is worth trying. "Seen" means on a render or
a tile. The list that is acted on is [`docs/state/next_steps.md`](../state/next_steps.md); what renders showed by form
is [`renders_by_form.md`](renders_by_form.md).

## Known limits

### Finding the person

The tracker and its SAM 3.1 behaviour are [`sam31.md`](sam31.md); recognizing a named person is
[`who_is_who.md`](who_is_who.md).

| limit | what it means | evidence |
|---|---|---|
| it is not an identity model | SAM's trunk says what a thing is, not who; people who look alike score alike (on the car clip one shot of the lead scores the same as another woman) | `../../bench/results/2026-10-04_subject_track_three_clips.md` |
| the rule was chosen on the three clips it passes | no unseen clip has been tried | the same record |
| a miss leaves the original in the shot; a wrong take replaces somebody else | a shot is corrected by a typed correction (`shot 3: person 2`, `shot 3: none`) or a seed; run on one real clip, where a missed shot was taken by a typed correction | `subject_track.py`, "A correction"; `../../bench/check_subject_track.py`, item 7; `../../bench/results/2026-10-06_subject_track_defaults.md`, "The correction" |
| seen from behind is not found | a shot in which the subject never faces the camera is left alone | `subject_track.py`, "How far the matching can be trusted" |
| a change of framing lowers the score of the same person | the report says when a shot is framed much closer or wider than the pick; a head crop through the trunk again holds on the one-person clip and fails on the car clip, and is not in the node | the three-clips record |
| in a one-person clip a cutaway to a different lone person is taken | naming a value for `match` turns that rule off | the same |
| a dissolve or a jump cut inside a cutaway can score under the cut threshold | the report prints the highest steps with the threshold marked | the same |
| cost | two detector passes on every frame looked at and a tracker pass per shot, on every run that tracks; a job pays it once, in its preview | `subject_track.py` |

### Replacing the person

| limit | what it means | evidence |
|---|---|---|
| movement is not carried by the generic prompt | the new subject faces the camera when the original turns; a late start carries the turn and the original's hair and clothes; from a softened copy it turns and still changes clothes. Neither is a fix; the motion video is ([`pieces.md`](pieces.md)) | `../../bench/results/2026-10-04_masked_v2v_band.md`, "The turn without a prompt"; `../../bench/results/2026-10-04_masked_v2v_turn_soft_arms.md` |
| the hole decides size and place | the still carries no height or build; a tight margin keeps the new subject in the original's footprint, a wide one lets a differently shaped subject fit and stand elsewhere | [`docs/h3/audio_freeze.md`](../h3/audio_freeze.md), section 4 |
| `head and hair` gives an oversized head on a long-haired original | kept as an option, not a recommendation | the same section |
| `paint_out` was judged worse | stays off | the same section |
| the mask is the body, not what the body does to the room | the original's shadow and reflections stay | |
| two-sampler graphs do not carry the mask as wired | `MiniMaxH3RestorePlate` (`plate_restore.py`) exists for it and is in no shipped graph | |
| no control adapter | the owner ruled it out for this lane | [`docs/state/decisions.md`](../state/decisions.md), 2026-10-04 |
| the first clip's flicker | set aside, its dark lines being seams in the source's own backdrop; one part has no explanation | `../../bench/results/2026-10-04_masked_v2v_first_run.md` |
| a small face does not read as the reference's person | every outcome reading says a region differs from the source; none says it looks like the reference | [`state_signals.md`](state_signals.md), "Whether a render looks like the reference" |

### Using it

- **Inputs marked advanced are not folded away** in the owner's frontend, so the nodes show every input.
- **Core's SAM 3 checkpoint does not load in a plain process outside the server**: the text projection's shape is
  refused (seen by two sessions). mrblue's probes leave that one key out of the load in their own process; the scripts
  are under `internal/claude/`, which is not tracked, so a new probe has to repeat it.

## Directions

### Telling people apart

- **A model trained for identity**, used where SAM's features are used now, on the head, at the seed frames only. It
  is the real answer to people who look alike and to a change of framing. None is wired; a licence check and the
  owner's call come first ([`who_is_who.md`](who_is_who.md)).
- **A person the detector never found** has no number to type. Core's detector takes points and boxes; nothing here
  passes them ([`sam31.md`](sam31.md), "What is not known").
- **A fourth clip**, chosen before the rule is looked at again.
- **Upstream SAM 3's caller-mask request** (`coderef/sam3`) would replace the per-shot seeding with the tracker's own.
  Not served by the multiplex checkpoint and not in core's port.

### Movement

- Untried on the late start: a start between the schedule's first two knots, more seeds, another clip (the turn
  record lists them).
- **The scene's clothes on the new subject**: a crop of the original's torso as a second reference. Untried.

### What the body does to the room

- **Removing the original's shadow and reflections.** Nothing here does it. VOID was tested for it on 2026-10-05 and
  removed: `../../bench/results/2026-10-05_void_plate_turn.md`.
- **A soft edge**: a matting model on the subject. The nodes exist (`sapiens2_parts.py`, `MiniMaxH3SubjectParts`);
  the matte is in no graph ([`perception_models.md`](perception_models.md), the matting row). The model was read
  once, on 2026-10-04, and that reading is not in the tree.

### The render

- The mask through two samplers, with the plate restored between them.
