bump: patch

### Added

- `docs/wiki/state_signals.md`, routed from the wiki index: a ledger of
  what code can say about a subject in the masked lane and what it cannot
  yet. One entry per signal (which way the head faces, whether the face
  shows, head lift and a hand at the face, who is in front of whom, and how
  a signal reaches the model): the tool or node that measures it, the
  record, what is unverified or unbuilt, and which session holds it. It is
  annotated with a dated line when a status changes and never rewritten;
  a claim that came from a session's message and has no record says so.
  It exists because this status had been told to the owner in a
  conversation and was written nowhere a later session would read.

### Fixed

- `bench/check_distill_grid.py` raised instead of reporting when the vendor
  README under `coderef/` is absent: its two vendor cases skip, as designed,
  but the first of them was also the only place that told ComfyUI the
  process has no card, so the later import of core's model config raised
  core's own error on a masked card, which is not an `ImportError` and was
  not caught. Every import of core in the file now goes through one
  function that sets the flag first. With the README absent the check
  again ends with its two skips named and exit 2.
