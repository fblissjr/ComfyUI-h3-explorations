bump: minor

### Added

- `MiniMaxH3SubjectBoxes` (`subject_boxes.py`): a tracked subject's mask as one box a frame, in the form core's SAM 3D Body prediction takes (a list for each frame, a box as `x`, `y`, `width`, `height`), widened by `margin` and held inside the frame, with no box on a frame the mask is empty on. Nothing on a stock server turns a mask into those boxes, and without one that prediction takes the whole frame as one person's crop: undefined with two people in the frame, and a body drawn on frames the subject is not in. One node per person. `bench/check_video_mask.py` item 18 holds the boxes, the empty frame, the margin at the frame's edge and that core's own reader reads the list as one box a frame; `bench/node_id_manifest.json` records the node. No existing node, input, default or graph changes.
