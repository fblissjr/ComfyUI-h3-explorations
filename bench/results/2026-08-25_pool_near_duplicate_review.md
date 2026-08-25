# Pool-wide near-duplicate review, 2026-08-25

**Status:** Dated review record. Complete for the candidate set the window
proposed; see "What is still open" for what that window cannot reach.
Descriptive counts are observations of one scan, not standing claims.
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

**Complete for the candidate set, updated 2026-08-25 after the cluster pass.**
The exact-media map is wrong in the direction that matters — it separates rows
that share a visual family — and the corrected map repairs it. Every candidate
crossing an exact-component boundary now carries a verdict.

What remains unexamined is what the Hamming window never proposed: two images of
one subject far enough apart in hash to fall outside it are not in this map, and
nothing here would show that. The map's `caveat` field says exactly this, and it
changes text when the unruled count is non-zero rather than carrying a fixed
sentence that could go stale.

**Updated 2026-08-25, later: the first pass hashed images only.** The pool's
video files were outside the candidate window entirely, and the sentence above
said "every distinct pooled image", which was accurate and read as broader than
it was. Videos are now hashed as sampled frame sets, compared against each other
*and against every still*, and — because there are few enough — inspected
exhaustively as a contact sheet rather than only through the window. What that
found is below.

Every distinct pooled image and video was hashed; none failed. Inside the Hamming window,
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

Pair by pair, from rendered side-by-side comparisons:

- **Every crossing candidate clearing the correlation threshold.**
- **Every crossing candidate at Hamming 1 or less that the correlation missed.**
  Added after the 0.43 pair above proved the correlation gate leaks.

The strong band splits roughly evenly between real duplicates and the
turnaround template. The low-Hamming weak band is mostly template, with a few
genuine same-subject pairs — the finding that matters, since a correlation gate
would have discarded all of them.

Then, cluster-first, the whole remainder:

**Ruling a cluster is not ruling its pairs, and that distinction is the method.**
A cluster is a connected component of the candidate graph, so a shared template
chains unrelated images into one blob — here, 83 images and 285 pairs in a
single component. Giving that component one verdict would be wrong either way.
So each cluster was inspected as a montage and *duplicate subgroups were named
inside it*; a pair is duplicate only when both its images sit in a named
subgroup.

That is where cluster-first pays. Looking at 83 thumbnails at once found two
real subgroups — one repeated figure sheet and one repeated portrait — that 285
separate pairwise looks would have found at many times the cost, and the
grouping was obvious from the montage in a way it is not from any single pair.
The remaining twenty-odd clusters are small and were ruled the same way.

The great majority of the cluster-reviewed remainder is template. That is a
result, not a shortcut: it is the same confound the strong band showed,
confirmed at scale.

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

## What including video found

Two things, neither reachable by an image-only scan:

- **A reference still is a frame of a reference video.** Same promotional card,
  same actors, same composition; one static, one animated. Byte-different, in
  two different exact components, at a hash distance well inside the window
  once videos were hashed at all. The still's row is in the running calibration
  bundle and the video's row is in no holdout, so no split in flight is
  violated — but a later split drawing that video's family would have been
  contaminated with nothing to say so.
- **Two videos are the same shot list rendered twice.** The same four beats in
  the same order — a crystal rose with a golden butterfly on a cliff, the
  butterfly lifting off, a flower meadow, a castle-on-river reveal — but
  different renders: different castle, different mountains, a pink rose against
  a white one. Not the same pixels; the same brief.

**The second one is a limit on this whole method, not a finding about two
files.** It sits at more than twice the Hamming threshold and was never a
candidate; widening the window that far would swamp the list, since unrelated
pairs already reach a fifth of a percent at Hamming 12. It was found only
because nineteen files is small enough to look at exhaustively. **The equivalent
among the images is unexamined and the population is too large to eyeball.**

It is ruled `uncertain` rather than `duplicate`: a holdout drawn from one of
those clips would still measure different activations, so calling them the same
media would overstate it — but two rows generated from one brief are not
independent samples of the source either. `uncertain` is an edge, on the
standing asymmetry.

A pair found this way has to be able to reach the map, so an adjudication entry
may now carry `outside_window` and is applied on the strength of the ruling
alone. The window generates candidates; it does not bound what is true.

## A verdict that was corrected, and why it is visible

One pair in the *calibration* split adjudication was first ruled distinct
because its two frames share nothing, and later re-ruled duplicate when this
review adopted a series-level reading: a holdout drawn from one brand's
catalogue is not independent of calibration drawn from the same catalogue,
whatever the individual frames show. The earlier verdict is recorded on the
entry rather than overwritten, because a rule change that quietly rewrites past
judgements hides that the standard moved.

## What is still open

- **The window, not the ruling.** Pairs beyond the Hamming threshold were never
  proposed, so a same-subject pair that hashes far apart is invisible to this
  whole exercise. Widening the window raises the candidate count sharply; the
  measured chance rate is what would tell you whether it is worth it.
- **Panel comparison, still unbuilt.** Comparing the panels of a turnaround
  sheet rather than the whole frame would remove the template confound instead
  of ruling around it. Cluster-first made that unnecessary at this scale; a
  larger pool would change that.
