bump: patch

### Added

- `bench/capture_masked_run.py` says whose a segment is: `subjects/<label>/segments_whose.csv` has, per frame and segment, the pixels inside the subject's own tracked mask, inside another subject's, and in neither. The part node cuts its class map to the subject's mask widened by its margin, so within that margin of the outline a class says what a thing is and not whose; on one measured frame 81% of a "hand" lay in the margin and 57% of it inside the other subject's mask. A preflight rule, `segment_mostly_outside_its_own_track`, names a class a plan relies on (a part's own classes, or a segment inside a region) when under half of it is inside its own track.
