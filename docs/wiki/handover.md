# Handover: read this before anything else

Rewritten 2026-10-22 at the close by mrtim (the Claude lead; Gemini away as a lead, its reviewer reading by the owner's hand),
with mrcat then mrtiger on the card. Where this page and a record disagree, the record is right: `git log`,
[`decisions.md`](decisions.md) 2026-10-22 (its first entry holds the owner's rulings of the day), the owner's rules at the top of
[`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), the frame math in [`../masked_v2v/frames_rules.md`](../masked_v2v/frames_rules.md),
and `data/fun/` (laid.jsonl, plan.json, owner_answers.jsonl, verdicts, shots.json, cast). Running notes:
`internal/claude/2026-10-22_mrtim/render_plan.md`; the card's: `internal/claude/2026-10-22_mrtiger/handoff.md`.

## State when closed

**The owner's call at the close: start tomorrow fresh, end to end.** No render ran after second_hfp_5533c. Every shot still to
render has (or is getting, mrtiger's last step) a fresh preview on the day's final code; tomorrow's chain starts from those, job
onward, and nothing older is reused.

**What changed today, in what a load gets** (each landed after a blind read; the commits say the rest):
- The text: the owner's short form ("the woman with blonde hair whose appearance comes from <Picture 1>"), no shortening of her
  hair, no negatives, the eyes and (only where an open mouth is drawn) the mouth following the figure (`bench/load_text.py`).
- The drawing: a dark iris where the eye looks down, opened to `EYE_OPEN`'s least; the lead's outlined open mouth (plan
  `reference.open_mouth`); `<Video 1>` cropped on the grey head (plan `reference.crop` "cranium" for all three subjects,
  `bench/crop_reference_to_head.py`; a frame where she is hidden does not size the box).
- The window: a short shot's padding goes after it, held copies of its last frame where the next shot begins
  (`frames_rules.md`); the zoom to a head size exists but is off (`zoom_up_to`, default 2).
- The gate: a reference with no figure for a run of frames stops a load; her hair is kept inside the hole, and a lost head is
  asked as plan `head_marks` (hidden, hers, not hers).

**Kept today:** 0415n, 4602d, 5011e, 1306g (dad), 4602c (dad), 1156g (lead), 0278m. **On the page, answered:** everything open.

## What not to assume

- **The gaze is parked, not fixed.** On 0278m the eyes stay down to 287 and open as her head lifts: the dark iris fills a propped
  slit, which reads open on a raised head. The owner's idea for one later test: an iris whose position in a white eye says where
  she looks. Shut eyes are drawn open too (1156g, 1163-1188): name a shut stretch `--eyes-shut-on ... --eyes-shut-form closed`.
- **The stacked clip's `<Video 1>` "head" inset is the strip's own zoom**, not what H3 got; it cut 4479f's profile where the
  reference holds her whole head.
- **1093-1095 and 1099 have a wrong body-model fit** (upside down, eyes on her hands) and are not marked hidden: the 1093 pair
  draws no face there; if it misbehaves there, those frames are the reason.
- **The cut on the share** is the rebuild started 21:04Z (laid 355df89e); 0278m was laid after it, so the next rebuild takes it.
- **A code fault for tomorrow:** 4647e's reference crashed in `bench/draw_face_points.py` `find_mesh_chin_y` (IndexError, an index past
  the 768-row frame; `data/2026-10-22_mrtiger_jobs/chain/second_hfp_4647e.head_reference.out`), likely the cranium crop with her face
  points off the frame. Fix it before 4647 renders.
- **Strangers files:** the first job on a preview writes it; previews whose first job ran after b7cb94b4 carry the whole-frame
  part map the hair check needs. 5114 and 4912 were re-previewed at the close for that; 1093 and 1419 get theirs on the new cut.
- **A load's letter can read older than it is** (second_hfp_0014b after 0014f): the next-letter scan misses older days.

## Tomorrow, in order

1. Start ComfyUI and the review server by the recipe (`docs/comfy_notes.md`); read mrtiger's handoff for the fresh previews and
   their letters.
2. Render from the fresh previews, job onward, the owner watching each on the page: the mom's 0014 (14-167, the owner saw mask
   bleeding round her on 0014f), 1093 (the pair: default zoom and `--zoom-up-to 4`, the test that decides the zoom; the owner's
   hidden marks are in the plan), 4647 (no face drawn where the dancers hide her), 5533 (5533c is a candidate rendered on older
   data), 5114, 4912 (expect head-mark questions), 1419 (the third pass on the newest cut); the dad's 4726, 5011, 5461, 5675; the
   lead's backfill (2050 with `--no-face-on` on its transition runs, 516 with the owner's "zoom the best you can", 3719 from the
   new box) and the tails.
3. One rebuild per batch of keeps; nothing laid while it runs.
4. After the video, the tool notes in the running notes' last lines (the review inset's framing, the next-letter scan, the cut's
   laid hash, the pair zoom key, the record wording Gemini flagged); then the parked gaze test.

## Where things are

- Mailboxes: `internal/claude/2026-10-22_mrtim_to_mrcat.md` (T-entries to T41), `..._mrcat_to_mrtim.md` (S-entries to S65; mrcat
  then mrtiger). The Gemini reviewer's reading: `internal/gemini/2026-10-22_eyes_mouth_reading.md` (§1-8).
- Proofs: `Video/fun/looks/2026-10-22_mouth_forms/` (eyes, mouth, crops), `Video/fun/looks/2026-10-22_4479_nape/` (the dark nape).
- Worktrees left (landed ones removed, branches kept): claude-seed-every-box (held), claude-change-check and claude-round-hole
  (parked), mrgem-cut-map (Gemini's).
- ComfyUI: stopped by mrtiger at the close (its handoff gives the time); the review server: see the card's handoff.
