bump: minor

### Added

- **`MiniMaxH3SubjectParts` has a `held` output**: one mask per frame of what
  the subject holds, made with no name asked of any model. The tracker's mask
  of a person takes in what they hold and the part model has no class for it,
  so inside the subject's own mask a pixel the part model calls background is
  something attached to them that is not them; it is taken where it lies near
  the lips or a hand, and a frame with too few such pixels is left empty. Two
  optional inputs set the reach and the smallest size (`held_near`,
  `held_smallest`). Meant for the Masked Source's `keep`, so a cigarette, a
  cone or a microphone stays the original's while the person is replaced, and
  nothing small is tracked on its own. The owner's ask of 2026-10-07; the
  anchor on the hands is from a sketch the owner brought from Gemini, and the
  mouth is added because their first example starts there. Checked on a
  painted frame; not yet run on a clip. No generated graph wires it, and the
  node's other outputs are unchanged.
- **`check_subject_parts.py`, the held case**: two held things taken exactly;
  the same thing far from lips and hands as the control the reach must
  decide; the same thing outside the subject's mask never taken; no anchor,
  nothing held; a speck dropped by the smallest size alone. Seen red with
  every body class as an anchor and with the reach ignored.
