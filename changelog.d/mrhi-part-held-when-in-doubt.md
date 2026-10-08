bump: minor

### Changed

- **The Subject Parts node holds a frame whose part it does not trust, as
  it holds a frame with no part** (`sapiens2_parts.subject_parts`,
  `hold_missing`, on by default). "Found" is one pixel, so a frame where
  the labels were a sliver of the subject, or lay mostly off the subject's
  mask, counted as found and came back as that sliver. The Masked Source
  then regenerated the sliver and its margin, on whoever stood there, and
  left the subject's own head and chest as the original's: the board's
  `build-small-subject-part-and-margin`, and the part report's own
  "likely someone else's" line. The frames `part_coverage.summarise`
  already doubts (nothing on the subject, mostly outside, far under the
  clip's median) are now treated as not found, and the hold that existed
  gives each the nearest trusted frame's part, moved from that frame's
  subject box to this one's and cut to this frame's subject. Nothing new is
  tuned: the rule and its two thresholds are `part_coverage.py`'s. What a
  hand or the lips held on such a frame is dropped, since it was read off
  somebody else's labels. A clip with no trusted frame is left as found.
  A frame with no part at all whose nearest found frame was such a sliver
  now takes a trusted frame's part too, since the sliver is no longer a
  frame to hold from. The report names the frames apart from the missed ones, and
  `MASK_VERSION` on the node is 3. With `hold_missing` off every frame
  comes back as it was found, as before.
- `bench/check_subject_parts.py` item 5 gains the case, with the node as it
  was as its control (the hold off: the sliver comes back and
  `part_coverage` doubts it); item 11's sliver case now runs with the hold
  off, which is where a sliver is still returned.

### Not done

- Not run through the node on a clip: the frames it would hold on the clip
  it was written for were read off a no-sampling preview's masks, and the
  hold was applied to those masks outside the node.
