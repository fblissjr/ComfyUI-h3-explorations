# Text-only T2VA regression holdout, 2026-08-25

**Status:** Dated build record. Descriptive counts describe this build.
**What:** the deterministic text-only regression population `active_plan.md`
names — pool rows excluded from the vision-traced `oneshot` calibration because
a sequential trace admits one modality envelope. **Not calibration input.**
**Selector:** [`bench/select_t2va_holdout_rows.py`](../select_t2va_holdout_rows.py)
→ [`2026-08-25_t2va_holdout_rows.json`](2026-08-25_t2va_holdout_rows.json).
**Builder:** `bench/build_native_h3_calibration_batch.py --population text-only
--device cpu`, so the layer-50 comparator grades these rows through the same
path as the vision holdout.
**Bundle:** built out of tree; `presentation.json` hashes to
`899d3f13aa16f36c28a2b1469011c4dfd6fc0e94a3896fc1cefd16a257342e54`. Thirteen
rows, sized to match the vision holdout. No row carries a vision position, a
visual block, `pixel_values`, `image_grid_thw`, or a media file. That absence
*is* the population; it is not a dropped input.

## Disjointness, and why the vision holdout's version of it would be empty here

These rows carry no media, so "no shared media file" and "no shared visual
family" are true by construction. Reporting them green would be the emptiest
kind of check — the input already satisfies the outcome, so it cannot fail. The
axis that carries information is the prompt.

- Exact and normalised prompt hashes are unique within the selection and absent
  from the calibration bundle and the vision holdout. No row id is shared.
- Near-duplicate prompts were reviewed by 8-gram Jaccard, across the split and
  within the selection, **reported against the population's own background
  level** rather than gated on a threshold. Every H3-IR `target_ir` shares a
  section skeleton, so a bare floor is high for unrelated rows; the number that
  means something is how a pair compares to unrelated pairs from the same
  source.

**Result:** the strongest cross-split prompt overlap in this build sits an order
of magnitude *below* the highest overlap between two unrelated rows of the same
population. There is no prompt-level contamination to argue about.

## Composition

Stratified on what actually distinguishes text-only rows from one another —
dialogue markers, contract duration, prompt length — rather than on the role
partition, which is a single value here.

- Dialogue markers present in seven of the thirteen rows, deliberately richer
  than the population's own share, because the marker families are the reason
  a text-only regression arm is worth keeping.
- Prompt length spread across short, medium and long bands.
- Contract durations spread from four to fifteen seconds, eleven distinct
  values across thirteen rows.
- One row carries Chinese dialogue inside its `<d>` block. That is coverage,
  not a defect: a multilingual row is exactly what a marker regression arm
  should contain, and the curation note that surfaced it is the check working.

## What the builder change was, and its guard

`--population text-only` reads the pool's exclusion complement instead of the
pool. **The two populations cannot mix**, and that is asserted rather than
assumed: text-only mode refuses a bundle containing any vision-bearing row, and
`--family` is refused outright because it selects a vision primary role. A
text-only row inside a vision bundle would be the silent modality drop the plan
forbids, so the mode is explicit at the command line and recorded in the
bundle's provenance.

## Two false reds this population exposed in the review harness

Running [`review_v2_calibration_bundle.py`](../review_v2_calibration_bundle.py)
against the first build turned every row red twice, on a bundle that was
correct:

- `bundle_files` required a media file per row. A text-only row has none and is
  not missing one.
- `family_disjointness` required every row to appear in the corrected component
  map. That map is built over the vision-bearing pool, so a text-only row is
  absent from it by construction.

Both are `CLAUDE.md`'s rule that a thing gaining an "absent" state gives every
assertion about it a third case, and that **correctly absent is not broken**.
Both now key on whether the row carries a media tensor, so a *vision* row that
lost its media file or fell out of the map is still red — asserted by two new
mutations, `missing-media-file` and `unmapped-media-row`, both caught.

The family arm now also declares itself vacuous on a bundle whose rows carry no
media, rather than reporting a green that asserts nothing.

**A third defect surfaced in the controls themselves.** Running the violation
arm against this bundle crashed: three mutations selected their subject with a
bare `next(...)`, which raises `StopIteration` rather than the `LookupError`
the arm reports as "could not be applied". Every mutation now names its
precondition and refuses by name. On this bundle ten of the fourteen refuse,
which is the honest outcome — a text-only population cannot exercise a vision
control, and the arm says so instead of scoring it.
