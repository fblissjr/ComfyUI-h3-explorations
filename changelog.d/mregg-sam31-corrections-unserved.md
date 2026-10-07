bump: patch

### Added

- **A SAM 3.1 Corrections node, written and checked, not yet served.**
  `sam31_corrections.py` takes the model and text encoder any checkpoint
  loader returned and gives back clones that carry two object patches:
  the image mapped to the range Meta's code hands the model, at the
  trunk's first layer, and the text encoder's activation set to the one
  Meta built it with (the two departures recorded in
  `bench/results/2026-10-07_sam3_core_against_meta.md`). It loads nothing
  and manages no memory; ComfyUI applies the patches on each load and
  removes them on unload. It leaves alone what is already right: an
  encoder that already runs exact GELU, and frames that arrive already
  mapped. It is NOT in the pack's node list: its last acceptance, a graph
  on the server's queue with an H3 render between two reads, has not run.
- **Two checks for it.** `bench/check_sam31_corrections.py` on stand-in
  modules through core's real patcher (in the sweep), and
  `bench/check_sam31_corrections_on_card.py` on the real model with the
  patcher class the server uses, a stock pair and its corrected clone
  sharing one model, read in turn through core's detect and track nodes
  and after an unload (needs the card). `docs/checks.md` has both rows.
- **`bench/subject_track_under_nudge.py`**: today's Subject Track run on
  the same frames as fed and with every value moved one level of 255,
  comparing each shot's decision, each likeness against its line and each
  regain between the two.
- **`subject_tracks.py`, the first model-free piece of the tracker to
  come**, with `bench/check_subject_tracks.py`: a shape test that does not
  count a mask slid onto the frame's border, or a fragment, as a subject;
  a watch that doubts a track no detection agrees with; and corrections
  addressed by a place in the clip (a frame or a time) so one text serves
  every load of a long clip. Nothing uses it yet.
