bump: minor

### Added

- `MiniMaxH3SubjectParts` gains a sixth output, `classes`: one mask per frame whose grey level is the part model's class on each pixel of the subject (hair, face and neck, each hand and arm, the clothing, lips, teeth and tongue: every class in `sapiens2_parts.CLASS_NAMES`, whatever is ticked), 0 off the subject. The node already computed that map and discarded it after merging the ticked classes into `parts`. It is for a capture that gives each part of a person its own id and colour (`bench/capture_masked_run.py`), saved lossless. `sapiens2_parts.class_mask` writes the index as a level and `class_indices` reads it back from a saver that rounds or one that truncates. The five outputs before it are unmoved and no input or default changes; `bench/node_id_manifest.json` records the appended output and `bench/check_subject_parts.py` gains the case that holds it.
