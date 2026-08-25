# v2 calibration set review, 2026-08-25

**Status:** Dated review record. Descriptive counts below are observations of
one bundle at one moment, not standing claims.
**Subject:** the mixed-geometry v2 calibration bundle (55 rows, `still_policy`
recorded per row) and its two holdout bundles (the same 12 rows built under
`upscale_2048` and under `max_no_upscale`), all produced out of tree by
`bench/build_native_h3_calibration_batch.py` from rows chosen by
`bench/select_v2_calibration_rows.py`.
**Harness:** [`bench/review_v2_calibration_bundle.py`](../review_v2_calibration_bundle.py),
CPU only, no CUDA, no server.
**Adjudication:** [`2026-08-25_v2_split_near_duplicate_adjudication.json`](2026-08-25_v2_split_near_duplicate_adjudication.json).

## Verdict

**Proceed on the calibration side; the split cannot be frozen.** The two
findings are both holdout-side. The calibration rows themselves passed every
structural arm.

The running job was confirmed to read the calibration bundle and nothing else:
its command line named that bundle and a 55-row prefix, with neither holdout
directory on it. So the holdout defects did not contaminate the run, and
aborting it would not have repaired them. (That run was afterwards killed by the
host OOM killer for an unrelated reason, recorded by the quantization lane.)

## What the harness recomputes rather than reads

The bundle's own `presentation.json` records the geometry, labels, token counts
and digests the builder produced. Grading those against themselves would be the
defect class the rejected preflight shipped, so this harness reimplements
`adapt_canvas`, the still-policy scale, the 17n+5 frame grid and the 2 fps walk
from the release constants, and cross-checks its copies against the installed
`nodes_minimax_h3` before using them. Media digests are recomputed from the
pinned H3-IR snapshot, prompts from the snapshot's `train.jsonl`.

## Findings

### 1. The holdout drew from a calibration photo series

**MEASURED.** Holdout row `train-00508-59139e4e6c98631b` is the same JOSINY
footwear catalogue series as calibration rows
`train-00795-cff93b01d353955c`, `train-00559-661153c0cc861266` and
`train-00315-0e6e0c0d1c58c9bc`: one studio, one model, one shot list, brand mark
in the same corner, frames matched shot for shot, only the product changing.
Eight media pairs cross the split. Every pair was rendered side by side and
looked at; `train-00315`'s own `target_ir` names the brand.

An independent signal agrees: on 8-gram Jaccard over `target_ir`, the top
cross-split prompt pair in the whole bundle is `train-00795 | train-00508`.

This trips `active_plan.md`'s stop condition on calibration and holdout
overlapping by near-duplicate media. The exact-media component map could not
have caught it, because the files are byte-different — which is the gap the
plan's near-duplicate review exists to close, and it had not been run.

The repair is holdout-side and does not touch calibration.

### 2. The holdout reserved no small-source component

**MEASURED.** `active_plan.md` locks "reserve at least two small-source
components for holdout". Neither holdout bundle carries one. Of the pool's
small-source rows, one is in calibration and none is in holdout.

This requirement had no assertion anywhere until this harness; it is now in
`docs/checks.md`'s uncontrolled-requirements table with a named check.

## What passed

All executed, all recomputed:

- **Media identity.** Every declared media file re-hashed from the pinned
  snapshot. Row-declared, bundle-recorded and pool digests agree everywhere.
  Every bundled tensor file is present, hash-correct, and named by a row; no
  file in the directory is unaccounted for.
- **Prompt provenance.** Every row presents its own `target_ir`, byte length
  included. None presents the user request. No chat framing.
- **Presentation order.** `labels_in_order` is the request order each row's own
  `available_media_labels` declares, with a 1..n counter per type; label kinds
  and media paths agree with the source row's per-kind arrays.
- **Per-row geometry.** Every reference still recomputes to its recorded
  upstream size under its row's declared policy, and the row policy agrees with
  the provenance map. Every keyframe sits on `adapt_canvas` with the crop its
  declared temporal role takes. The rows are split 19 `upscale_2048` and 36
  `max_no_upscale`, as the plan's mixed population requires.
- **Reference video.** Release role policy: 768-short-edge canvas under the
  area budget, sampling at 2 fps, timestamps presented as the mean of each
  two-frame pair with an odd tail filled by repeating the last frame.
- **Token accounting.** Sequence lengths, the text/vision split, per-block
  merged tokens, grids, packed patch rows and vision spans all agree with each
  other and with the batch tensors.

Two observations that read like defects and are not:

- Two of the three reference clips are shorter than their contract duration.
  `_prepare_reference_video` walks *down* to the 17n+5 grid when the source is
  short, so a shorter prepared length is the deployed behaviour, not truncation
  damage. It is why one row's exact length lands several thousand tokens under
  the selector's estimate.
- Video timestamps render as `0.2` where the pair mean is `0.25`. That is the
  release's `%.1f` label formatting, not drift.

## Curation notes, for the owner rather than the gate

- One row presents a 360x359 source interpolated to 2048x2048, a 32x area gain.
  That is the owner's `upscale_2048` decision working as decided; it is also the
  most extreme manufactured-pixel row in the set and the one worth showing if
  anyone asks what the decision bought.
- One row's `target_ir` never mentions two of the pictures it presents.
- Three of the 55 calibration rows are the JOSINY series above, and seven more
  near-duplicate media pairs sit inside calibration. Redundant mass, not a
  defect.

## Method: why the near-duplicate arm is adjudicated rather than thresholded

A plain perceptual-hash threshold is not usable on this dataset. A large share
of H3-IR is three-view character and garment turnaround sheets on a white
field, and a difference hash matches that *template*. The first pass reported
six cross-split hits at Hamming 6 or less; four were unrelated subjects sharing
the layout — a Korean armour set against a bodybuilder, a blue blazer against a
silver dress. A gate that fires on correct data trains the reader to ignore it.

So the arm now ranks candidates by a background-masked foreground correlation,
which is the part able to disagree with the hash, and grades them against a
recorded adjudication file. A candidate nobody has ruled on is reported as
unreviewed rather than passed; one ruled a duplicate is reported as a defect;
one ruled distinct passes with its reason kept in the record. The threshold is
also priced: images drawn from distinct exact-media components land inside the
candidate window at a rate the run reports, so a reader can see what the window
costs.

## Red controls

`--violation-arm` mutates a scratch copy of `presentation.json` whose tensor
files symlink the real ones, so every arm runs against real media and only the
named field is wrong. Eleven mutations, each graded on **the arm it targets
gaining a problem the unmutated baseline did not have**: a wrong declared media
digest, a wrong bundle-file digest, a prompt that is not `target_ir`, reordered
labels, a label pointing at another item's file, a still recorded at a size its
policy does not produce, a row policy disagreeing with its items, a last-frame
keyframe carrying the first-frame crop, shifted scaffold timestamps, a sequence
length disagreeing with the batch tensors, and a calibration row pointing at a
holdout file. All eleven caught, 2026-08-25.

**Two versions of this control were wrong first, and both are worth keeping
visible.** The first compared exit codes — and the live bundle already carried
two reds, so every mutation "passed" without the review noticing any of them.
That is precisely the failure this file exists to catch, reproduced inside the
file; the fix is the per-arm baseline diff. The second was a mutation that
silently did nothing: the raw scaffold separates words with U+0120, so a regex
written against the decoded form never matched, and the timestamp arm looked
green because the mutation never applied. Only the baseline diff surfaced it.
