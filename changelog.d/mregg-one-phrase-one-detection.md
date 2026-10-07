bump: patch

### Fixed

- **The Subject Track sends one phrase asking for one detection to SAM 3
  bare.** `subject_track.py::counted` wrote `person:1` when `max_people` was
  1 (the input's minimum), and for a phrase typed with its own `:1`. Core's
  tokenizer takes a short cut for one prompt asking for one detection and
  encodes the text as typed
  (`comfy/text_encoders/sam3_clip.py::SAM3TokenizerWrapper.tokenize_with_weights`),
  so the text encoder was given the `:1` as words. A list of phrases is
  encoded part by part and keeps its counts. No shipped graph sets
  `max_people` to 1, so no render so far was affected; the default of 16 is
  unchanged. `bench/check_subject_track.py` drives core's own tokenizer: the
  tokens of what `counted` writes for a lone phrase are the bare phrase's,
  with a control that core still encodes `person:1` as typed. Found by a
  reading of core's tokenizer for the owner's question of whether the pack
  uses core's SAM 3 correctly; core's issue 15811 is the same fault seen
  from its own node.
