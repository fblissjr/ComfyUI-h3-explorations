bump: patch

### Added

- `docs/wiki/known_before_a_render.md`, routed from the wiki index: every
  fault of a masked job found after a render on 2026-10-10, each against
  the data that existed before the render (the trackers' shot tables, the
  plan's arithmetic, the class maps, a pose pass and the voice table on the
  source), whether we had it, and the preflight rule that raises it now or
  the words "not built". A row is not finished until a rule raises it
  before sampling.
