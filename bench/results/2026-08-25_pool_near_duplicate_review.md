# Pool-wide near-duplicate review, 2026-08-25

**Status:** Dated review record and a partial result. Descriptive counts are
observations of one scan, not standing claims.
**Subject:** the accepted H3-IR candidate pool in
[`2026-08-24_h3_calibration_pool.jsonl`](2026-08-24_h3_calibration_pool.jsonl),
whose components `calibration_data_pool.md` derives from *exact* media SHA-256.
**Harness:** [`bench/review_pool_near_duplicates.py`](../review_pool_near_duplicates.py),
CPU only.
**Adjudication:** [`2026-08-25_pool_near_duplicate_adjudication.json`](2026-08-25_pool_near_duplicate_adjudication.json).
**Output the splitter reads:** [`2026-08-25_pool_component_map_corrected.json`](2026-08-25_pool_component_map_corrected.json),
which carries the exact map, the corrected map, and the rows that moved between
them.

`active_plan.md` requires this review before any split is frozen, and nobody had
run it. The first split built without it put a holdout row in the same photo
series as three calibration rows
([`2026-08-25_v2_calibration_set_review.md`](2026-08-25_v2_calibration_set_review.md)).

## Result

**Partial, and deliberately so.** The exact-media map is wrong in the direction
that matters — it separates rows that share a visual family — and the corrected
map repairs the part that has been looked at. The rest of the candidate list is
**unexamined, not clean**, and the map file says so in its own `caveat` field.

Every distinct pooled image was hashed; none failed. Inside the Hamming window,
the large majority of candidate pairs cross an exact-component boundary, so they
are exactly the pairs that would change the map. Applying only the adjudicated
edges merges components, moves rows into a different family, and grows the
largest component substantially past its exact-map size.

## Why this cannot be a threshold

**Neither metric discriminates alone, and that was measured both ways.**

- A difference hash at distance 0 routinely pairs *different* subjects, because
  a large share of H3-IR is three-view character and garment turnaround sheets
  on a white field and the hash matches that template.
- A background-masked correlation of 0.43 — deep in what a correlation gate
  would discard — turned out to be the same bodybuilder figure rendered twice
  on the same sheet at the same pixel dimensions. The correlation is low only
  because the poses differ slightly, and the two files carry ten pooled rows
  between them across two exact components.

So the window is a candidate generator and nothing more. The run prices it: the
report records how often images drawn from distinct exact components land inside
it anyway. Verdicts come from looking at rendered pairs, and the record keeps
what was looked at separate from what was not.

## What was examined

- **Every crossing candidate clearing the correlation threshold.** Ruled
  individually from rendered side-by-side pairs.
- **Every crossing candidate at Hamming 1 or less that the correlation missed.**
  Added after the 0.43 pair above proved the correlation gate leaks; this is the
  band that pair came from.
- **A stratified sample of the remaining weak band**, drawn across correlation
  bands with a recorded seed.

The strong band splits roughly evenly between real duplicates and the
turnaround template. The low-Hamming weak band is mostly template, with a small
number of genuine same-subject pairs — which is the finding that matters, since
a correlation gate would have discarded all of them. The stratified sample of
the remaining weak band found only template pairs, which is evidence about that
band and not a clearance of it.

## The third verdict

Adjudication allows `duplicate`, `distinct`, and `uncertain`. `uncertain` exists
because several men's tailoring turnarounds in the same colourway could not be
told apart at review scale, and forcing them into a binary would have recorded a
confidence nobody had.

`uncertain` becomes a component edge. At a split boundary the two errors are not
symmetric: joining two families that were separable gives a slightly coarser
split, while separating one family gives a holdout that measures nothing. The
tie goes to joining — and the verdict stays visible as `uncertain` in the record
rather than being laundered into `duplicate`.

## What is still open

The unruled crossing candidates are the remainder of the weak band. Before
Gate 3 freezes a split, either they get adjudicated, or the split is drawn to
avoid the families they could join and that choice is recorded. Two ways to make
the remainder cheaper, neither built:

- Compare the *panels* of a turnaround sheet rather than the whole frame. The
  template is the confound; segmenting it removes the confound rather than
  thresholding around it.
- Cluster first and adjudicate clusters. Many of the ruled pairs belong to a few
  families — one catalogue series, one world-map asset, one figure sheet — so
  ruling on a family is a fraction of the work of ruling on its pairs.
