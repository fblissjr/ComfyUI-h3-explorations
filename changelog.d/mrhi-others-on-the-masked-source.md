bump: minor

### Added

- **The Masked Source takes a mask of the people round the subject, and the
  margin does not grow over them** (`others`, a new last input, optional;
  `video_mask.window`). A token that holds any of their pixels and none of
  the subject's own mask stays the source's. A token that holds any pixel
  of the subject's own mask still regenerates, whoever else is in it, so
  nothing of what is replaced is given up. "Own" is the mask the node
  settled on: where only a part of the subject is replaced, a mask of
  everybody that includes the subject takes the margin off the rest of
  their body, so the subject is left out of it (found in review, before any
  render). Unwired, nothing changes. Why: with two people inside
  the region the model drew the new subject on the other one
  (`bench/results/2026-10-07_masked_switch_keep_prompt_verdicts.md`,
  section 5). Why not `keep`: `keep` wins over the subject as well, which
  is right for a thing the subject holds and wrong for a neighbour, since a
  small subject shares most of their tokens with somebody and those would
  go back to the source, the original subject with them (the masking board,
  finding `mhi-05`). `keep` is unchanged and still wins where both are
  wired.
- The node takes a plain mask batch, one per source frame at the frames'
  own size, and refuses another length or size by name; nothing is
  stretched or cut to fit. It refuses `others` with `paint_out` or a
  softened start, as it does `keep`. The preview strip shows the people
  kept out in blue. `bench/node_id_manifest.json` records the appended
  input.
- `bench/check_video_mask.py` item 14 holds it; its control is the same
  mask on `keep`, which must cost the subject tokens.

### Not done

- Not rendered, and no graph wires it. Who goes on the input is not built
  into any node: the first use is a mask made outside the graph.
