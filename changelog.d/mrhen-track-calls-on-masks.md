bump: patch

### Added

- **The Subject Track's tracker calls judged on their own masks**
  (`bench/results/2026-10-07_subject_track_calls_on_masks.md`, five data
  files beside it). `subject_tracks.unbroken` and `gallery_span` had been
  accepted on recorded step values; here every call the node makes to the
  tracker goes through them at frame size as it comes back. The jump of the
  largest person's track is cut on masks as it was on recorded steps, and
  the most central person's calls on three windows are never cut. A second
  jump on the same stretch has no frame of warning before it. On a fifth
  window, where the mask shrinks to a fraction of its area, the track is
  one continuous mask that a fresh detect reproduces at every look, so a
  refresh from the detector would change nothing there, and the pick rules
  name different detections on its pick frame.
- **Who stands in the subject's way, named without a model**
  (`subject_tracks.in_the_way`): of the detections on a frame that are not
  the subject, the one whose mask covers most of the subject's box. The box
  and not the mask, because the detector's masks of two people hardly
  overlap even where one stands in front of the other. It names a first
  guess for a shot, for a person to be followed in a tracker call of their
  own and kept out of the regenerated region; nothing follows or keeps
  anybody yet, and the share that counts is reasoned, not measured.
  Imported by no node.

### Changed

- `bench/check_subject_tracks.py` names the jump with no warning as a known
  cost of `gallery_span` (one good frame left out, no bad one let in), and
  `gallery_span`'s note says what its one-frame trim was reasoned from and
  what it is not shown to cover.
