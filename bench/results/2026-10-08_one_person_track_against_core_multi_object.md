# This pack's one-person track beside core's text-prompted multi-object track, on one load (2026-10-08)

lane: masked
verdict: on one load of one clip, this pack's Subject Track held a mask of the subject on every frame, with one let-go it recovered from on the same frame; core's text-prompted multi-object track, at its own object cap, held the subject and the person in front under their own ids through the stretch where the two overlap, then used its last free slots while the crowd moved and held nobody from a little past the middle of the load to its end; one run of each, not the same task, and not a ranking of trackers

**Read this first.** One clip, one load, one run of each tracker. The two
were not asked the same thing: core's node was given the phrase and tracks
everyone it detects; this pack's node was told which person, by a rule on a
frame a person named, and follows only that one. Nothing below says which is
the better tracker in general. It says what each did here, and why the
masked lane's subject is followed by this pack's node and not by core's.

**What was asked.** The owner, 2026-10-08: show every object core's tracker
holds, each with its id, with our subject marked, and say whether his id
survives the whole load and whether the person standing in front of him over
frames 275 to 315 has an id of their own (that mask is what the Masked
Source's `keep` or `others` input would be fed).

**How.** Both ran on the masked lane's server, on the same frames from the
loader node the masked graphs use (the `load` block of the json).

- *Core*: `SAM3_VideoTrack` with the phrase through `CLIPTextEncode`,
  detection on every frame, and `max_objects` at the value that selects the
  tracker's own cap (`comfy/ldm/sam3/tracker.py::INTERNAL_MAX_OBJECTS`). Its
  `SAM3_TrackPreview` was written, and `SAM3_TrackToMask` wrote one lossless
  grey video per object id for the ids the json names as written.
- *Ours*: `MiniMaxH3SubjectTrack` at the inputs in the json's `ours.inputs`.
  Its figures are the node's own report, read from `/history`.
- *The comparison*: per frame, the written object that covers most of this
  pack's mask of the subject is "the id on him" (the rule and its threshold
  are in the json). Core's preview was also compared with the source, frame
  by frame, for how much of the picture carries any fill; that measure
  covers every id, including the ones not written as masks.

Scratch scripts, not kept; the json is their whole output.

**What each did.**

- *Ours held him throughout.* The node reports one shot, the subject on
  screen on every frame of the load, and one frame on which the tracker let
  go and the node's search took him back on that same frame, with the
  likeness and the next person's in `ours.node_report`.
- *Core held him, under one id, for the first part of the load and then not
  at all.* The id on him never changed while it lasted; it covered
  nearly all of this pack's mask on the typical frame and most of it on
  the worst (`core.id_on_our_subject`). From the frame
  in `core.preview.last_frame_with_a_fill` on, the preview carries no fill
  anywhere: nobody is tracked to the end of the load.
- *Why it stops, read in core's code and consistent with the figures, not
  proven by them.* An id is a slot; slots are never freed or reused, and
  detection runs only while a slot is free
  (`track_video_with_detection`, `_match_and_add_detections`). The per-id
  spans show nearly every early id ending within a few frames of one
  another, where the crowd begins to move; the preview on one frame just
  after shows ids up to the cap. With the slots used, nothing is detected
  again, so nobody is picked up afterwards. A lower `max_objects` would
  reach that point sooner; the node has no higher one.
- *The person in front has a steady id of their own over 275 to 315*
  (`core.the_person_in_front_275_to_315`): present on every frame of the
  range, beside the subject's id, which is also present on every one.

**So what, for the masked lane.**

- The subject is followed by this pack's node. On this load core's pass
  would have left the second half of the render with no mask.
- Core's pass is useful for the other people over the part of a load it
  covers: an id per neighbour, where this pack's node gives one mask. It
  cannot supply a mask of the others to the end of this load.
- A mask of everyone on each frame does not need identities. Core's
  per-frame detector (`SAM3_Detect`) has no slots to run out of; whether it
  finds people who are bent over is not known. Not run when this was
  written.

**What this does not show.** The json's `not_shown` list is the full
statement. In short: nothing was repeated; one threshold and one phrase;
core at its own cap with detection on every frame and no other setting
tried; the two times in the json cover different work in unknown cache
states and are not a speed comparison; no outline was graded for being on
the right pixels; ids above the written range are known only from the
preview.

**Files.**

- [`2026-10-08_one_person_track_against_core_multi_object.json`](2026-10-08_one_person_track_against_core_multi_object.json):
  the load, the inputs of both nodes, both prompt ids, each written id's
  spans of frames, objects holding a mask as runs of frames, the id on the
  subject as runs, the preview's filled share at intervals and around the
  frames where it ends, the node's own report for ours, the two times with
  what each covers, the host, and `not_shown`.
- The review video, the per-object masks and core's preview are on no
  tracked path: they show the owner's clip.
