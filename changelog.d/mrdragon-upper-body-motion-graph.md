bump: minor

### Added

- **`h3_video_to_video_masked_upper_song_ref2va_motion`, with the daily copy
  `h3_mask_upper_ref2va_motion`: the head and upper body replaced on the
  ref2va motion graph.** The part node at `h3_config.MASKED_UPPER_PARTS`
  (hair, face and neck, upper clothing, hands) is the region, the legs stay
  the source's, the subject's own frames are the motion reference, and the
  prompt node is told the still gives the head and upper body
  (`MASKED_UPPER_SOURCE`, `MASKED_UPPER_PROMPT`). It is the one recipe the
  owner called solid on playback on 2026-10-06
  (`bench/results/2026-10-06_masked_v2v_body_window_arms.md`), which until
  now existed only as patches in a run script; the shipped parts graph is on
  the fast chain with hair and face only, the combination that record calls
  broken for this region. The generated graph equals that arm's stored graph
  on every setting but the prompt: the arm rendered a typed text, this graph
  renders the node's wording for the same region
  (`prompt_bank/ref2va_masked_person_upper_motion.txt`), which has not been
  watched. Not rendered as generated.
- The generator's `masked_parts` takes the part node's widget values as well
  as `True`; `bench/check_widget_deviations.py` declares the two ticks and
  `docs/prompt_audit.md` has the new text's row. No node code changes and no
  other graph moves.
