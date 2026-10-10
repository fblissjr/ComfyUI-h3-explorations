bump: minor

### Changed

- `MiniMaxH3MaskedSource` refuses at queue time what it used to refuse only when it ran. `video_mask.settings_refusal` holds the two refusals that need nothing but the node's own widgets (a feather wider than the margin; a softened start with `paint_out`); the node's `validate_inputs` calls it when the graph is submitted, and `execute` calls it again for a value that arrives through a link. Core stops testing an input a validation function names, so the margin's and the feather's ranges and the `start_from` choice are tested there, against `GROW_PIXELS_MAX` and `FEATHER_PIXELS_MAX`, which the schema now reads too. `bench/check_video_mask.py` item 16 holds it, and was seen red with the validation returning True and with the refusal removed. No input, default or graph changes.

### Added

- `docs/wiki/masked_v2v.md` gains "The rule: data before a render, and the same data after", set by the owner on 2026-10-10 for every session on the lane, with one line in `AGENTS.md` and an entry in `docs/wiki/decisions.md`: a no-sampling capture and its flags before a render, the capture read first after a bad one, a cause cited to a row or a measured figure, every flag given its outcome, captured data under the untracked `data/`. The page's "Seeing what the tracker and the masks did" now names `bench/capture_masked_run.py` (added in the next entry) and the wiki page on what the models behind the lane's signals give.
