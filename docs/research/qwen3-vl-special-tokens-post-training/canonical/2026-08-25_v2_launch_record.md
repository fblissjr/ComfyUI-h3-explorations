# The v2 calibration launch: the host budget that was missing, and what ran

**Status:** Authoritative record of the first v2 calibration launch
**Recorded:** 2026-08-25
**Authority note:** decisions marked **ACTING-POINT DECISION** were made by the
technical point under the arrangement in
[`2026-08-25_gate2_arrangement.md`](2026-08-25_gate2_arrangement.md). The
owner approved the overnight calibration run and, after the first launch
died, the relaunch plan (largest population that fits, declared margin,
cgroup cap) the same day. Nothing here deploys anything.

## The first launch died, and the reason was measurable beforehand

**MEASURED.** The 55-row mixed-geometry population (595k tokens) launched at
12:52 was killed by the kernel at 13:06 with the pilot alone holding 122 GB of
anonymous host memory (`journalctl -k`, pid 647279). Nothing else on the box
was large.

The cost had been visible in the Gate 2B arms and was not read as a budget:
peak RSS 65.97 GiB at 1,100 tokens and 68.54 GiB at 8,703 with the weights
resident at 63.9 (`bench/results/archive/v2_encoder/2026-08-25_gate2b_row1_expanded_kv.json`,
`..._rows3_expanded_kv.json`). A 2-decoder-layer probe on the first 8 bundle
rows pinned it at scale: **110,331 tokens peaked at 50.8 GiB with only the
2-layer model resident, about 428 KB of host per population token**, of which
the intermediates cache is 78 KB
(`bench/results/2026-08-25_gate2b_host_budget_prefix8_2layers.json`). The
rest is AWQ's parent-argument cache: every batch's FP32 inputs for all three
parents of a layer held at once, `down_proj`'s 25,600-wide input dominating.
That cost scales with total population tokens, not with the longest row, and
`n_grid` does not change it.

This is the population/cache budget dimension the sprint dropped from Gate 2B
by owner decision. It was the wrong dimension to drop, and the 32 GiB host
reserve written into the first launch was an extrapolation from arms seventy
times smaller. Recorded so the next reader prices a population before
launching it: **host anonymous memory = resident weights + ~430 KB x tokens**,
against a 125 GiB box with ~119 GiB available when ComfyUI is down.

## The arrangement that fits: weights on the bridge's disk tier

**MEASURED.** With `--host-reserve-gib` large enough that the bridge's CPU
budget (`MemAvailable - reserve`) cannot hold the weights, the remainder goes
to its disk tier: a meta tensor in the offload cache, data read through
`safetensors` from a symlink into the source shard, and on the first update
the symlink is replaced by a staged file in the offload directory
(`compressed_tensors.offload.cache.DiskCache`). The same 8-row probe under
that tier peaked at 46.2 GiB (the resident probe's 50.8 minus what moved to
disk), staged 46 files at exit, and **emitted a packed candidate**
(`pack-quantized`, 117 ignores, same byte count as the CPU-tier smoke)
(`..._prefix8_2layers_disk_tier.json`). The reserve must leave the CPU tier
room for the smallest placement: at 117 GiB the bridge refused to put the
whole model on disk; 114 worked.

**The control that only knew one tier.** The pilot's modifier-entered control
hashes layer 0's `q_proj` before and after the modifier. Under the CPU tier
the tensor is hashed in place. Under the disk tier it is a meta tensor, the
control returned `None` for both readings, and `None == None` reported
"weight unchanged" on a run whose staged files proved the opposite. Fixed the
same afternoon: the control now reads a disk-tier weight through the cache's
index and records the tier, the file and whether it is staged; a meta tensor
it cannot index raises rather than returning `None`. The instance is the
standing rule about assumptions that have met one implementation, and it
would have refused to emit tonight's candidate.

**ACTING-POINT DECISION.** The v2 calibration runs with the weights on the
disk tier (`--host-reserve-gib 114`), under a user-scope cgroup cap
(`systemd-run --scope -p MemoryMax=115G`) so that an overrun kills the pilot
and nothing else on the shared host. Disk reads are one layer per onload and
are not on the critical path; the AWQ state is what the host is for.

## Host safety net, and the real fix, decided with the owner

**OWNER-DECISION, 2026-08-25.** Two things follow from the budget above.

1. **Insurance: a 128 GB swap file on the root NVMe** (`/swap2.img`, priority
   below the existing 8 GB one, persisted in `fstab`). It changes what an
   overrun costs, from a kernel kill to a slow run, and nothing else: the
   cgroup cap on the pilot remains the first line, and no population is sized
   against swap. It is not a way to run a larger population. The modifier's
   parent-argument cache is re-streamed about 63 times per layer (21 grid
   passes x 3 mappings), so any part of it that lives in swap is pulled
   through NVMe on every pass; at the 55-row size that is a day, not a night.
   **Status: executed by the owner at 16:00 on 2026-08-25; `swapon --show`
   lists `/swap2.img`, 128G, and `fstab` carries it at priority -2 (this
   session shows it at -1 beside the original; the priorities equalise until
   the next boot, which changes nothing about what it is for).**
2. **The real fix for a larger population is in the modifier, not the host.**
   AWQ holds every batch's FP32 inputs for all three parents of a layer at
   once so the scale search can re-run the parents. A larger calibration
   population needs that cache off anonymous memory (disk-backed, or BF16
   where the search tolerates it) or fewer passes over it. That is a code
   change to `llm-compressor`'s `AWQModifier`, named here as **deferred
   work**, to be taken up only if Gate 5 says more calibration mass is
   wanted; nothing about tonight's candidate depends on it.
3. **OWNER-DECISION, 16:35 the same day: layer-boundary checkpoint and
   resume for the pilot**, so a mid-run failure costs one layer instead of
   the run. Today's run has no such thing: the staged files of completed
   layers are keyed by tensor object id and nothing can re-enter the
   sequential pipeline at layer N. Assigned to mr_sparkles as a separate
   module, integrated into the pilot only after this run lands, and accepted
   only when an interrupted-then-resumed probe run is bit-identical to an
   uninterrupted one. **Parked at 17:00 the same day by owner decision:**
   the mechanism was proven at fixture scale on CPU (commit 1c427bd) and the
   pilot integration and real-weight proof were judged not worth their cost
   for a five-hour run; a long run is still all or nothing until a longer
   one is planned.

## The split, rebuilt

mr_data's independent review of the calibration set
(`bench/review_v2_calibration_bundle.py`,
`bench/results/2026-08-25_v2_calibration_set_review.md`) passed every arm on
the calibration rows and found two holdout defects:

- holdout row train-00508 was a shot-for-shot match of three calibration rows
  from one footwear catalogue series, byte-different files the exact-media
  component map could not see. The series spans 40 pool rows across about
  twenty exact components, all multi-image-4-9; holdout row train-00238 is in
  it too (adjudicated distinct, dropped anyway as a same-series row);
- the holdout reserved no small-source component against the locked
  "at least two".

`bench/select_v2_calibration_rows.py` gained `--keep-holdout`,
`--exclude-row`, `--exclude-component`, `--exclude-prompt-term` and
`--holdout-small-source` (reference stills preferred: only reference stills
are upscaled under `upscale_2048`, so a small keyframe never exercises the
policy the holdout grades). The rebuilt holdout is 13 rows, both geometries
from the same rows, two small-source reference-still components, and one
multi-image-4-9 row: after the series exclusion no second row in that family
fits under the 24k row cap at 2048, every free one carrying six or more
stills. That is a limit of the family at that geometry, recorded rather than
worked around.

**The corrected family map caught what the pairwise arm cannot.** mr_data's
pool-wide review (`bench/results/2026-08-25_pool_component_map_corrected.json`,
partial and saying so) merges exact components along adjudicated duplicate
edges. The trimmed calibration bundle was pairwise green against the holdout
and still shared one corrected family with it: holdout row train-00808 sits in
the pool's largest family, which two calibration rows had joined through
edges entirely inside the pool. Both calibration rows were dropped; the
harness now grades family disjointness as its own arm.

## What launched

**Launched 13:37:42 on 2026-08-25**, log and launcher in the point session's
scratchpad, report `bench/results/2026-08-25_v2_calibration_run.json`,
candidate directory
`coderef/llm-compressor/models/qwen3-vl-32b-h3-w4a16-awq-v2-nativecal` (new;
the deployed artifact, its symlink and the source are untouched).

- Population: 29 rows, 214,187 tokens, longest row 15,424; 9 rows under
  `upscale_2048` and 20 under `max_no_upscale`; every role present, roles by
  policy in the bundle's `presentation.json`. It is the 35-row B selection
  (~300k tokens, the better set on role coverage) cut from the top of its
  token distribution to fit the measured budget, then minus the two
  family-crossing rows. The catalogue series is excluded from calibration as
  well as holdout.
- Recipe: the committed v2 recipe (`bench/h3_awq_recipe.py`), duo scaling,
  `n_grid` 20, observer on CPU; boundary asserted before the first forward.
- Arrangement: `comfy_exact_bf16_store`, expanded-KV attention, all 64
  layers, weights 59.6 GiB on the disk tier and 2.6 GiB on CPU at load.
- Projected host: ~88 GiB of modifier state plus the CPU tier, about 25 GiB
  under the cap; a watchdog reports the high-water mark as it crosses
  thresholds.

## Gate 5 acceptance, pre-registered before any v2 number exists

**PLAN, written 2026-08-25 at 15:50 while the run was at layer ~28.** The bar
is set here so the result cannot be graded against a threshold chosen after
seeing it.

Instrument: `bench/compare_transformers_comfy_layer50.py`'s ComfyUI arm with
`--w4-path` and `--all-rows`, replaying the holdout bundle's recorded patches
through the deployed stack, layer-50 state per row; reference the BF16
ComfyUI arm; candidates the v1 and v2 single-file artifacts; all six captures
(3 arms x 2 geometries) in one session, because the adapter hash is part of
capture provenance. Aggregated by `bench/summarize_h3_holdout_captures.py`.
Presentation is the bundle's by construction, which satisfies the plan's
"policy forced into both arms" for items 1, 2 and 4 of Gate 5; item 3 (each
artifact's own declared deployed path) is the capture instrument's fixture
comparison and is optional for the sprint.

The bar, per geometry (`upscale_2048` and `max_no_upscale`), on the 13
holdout rows, relative L2 against BF16 at layer 50:

1. v2's median over rows is below v1's, on all rows and on vision rows;
2. v2 is below v1 row by row on at least 10 of 13 rows;
3. no v2 row's vision relative L2 exceeds v1's worst row of that geometry;
4. text rows: v2's median within 2x of v1's (W4 text is already near-exact,
   so this only guards a regression). The same criterion, and only that one,
   applies to the text-only T2VA holdout (13 rows, no media, built by mr_data
   at 16:10 the same day and graded through the same comparator), which is
   the plan's held-out regression population for the rows calibration never
   traced.

All four on both geometries: **accept**, and the candidate goes into the
graphs for Gate 6. Criterion 1 failing on either geometry: the calibration
hurt; **reject**, and the population or recipe is the suspect, not the
instrument. Criteria 2 to 4 failing with 1 passing: accept with the failing
rows named in the record, and those rows go into the blind pairs.

The question behind the bar, from the layer-50 baseline
([`2026-08-24_layer50_processor_policy_benchmark.md`](2026-08-24_layer50_processor_policy_benchmark.md)):
v1 sits at relative L2 0.229 against BF16 at its own narrow image budget and
0.527 at release bounds, vision cosine 0.966 against 0.832. v2 was calibrated
on both views. Whether it holds at 2048 where v1 collapses is the number that
decides the Qwen-only-upscale ablation's premise, and it is reported
separately from accept/reject.

Not answered by Gate 5: anything perceptual (Gate 6, blind pairs, the
owner's judgement), the marker question (the corpus arms on v2), and a guard
refusal, which is an instrument fault to fix and rerun, never a verdict.

## Gate 5 result, 2026-08-25 17:36: reject against the bar

**MEASURED.** `bench/results/2026-08-25_v2_holdout_layer50.json`, the
comparator's ComfyUI arm on the 13-row holdout in both geometries and the
13-row text-only bundle, BF16 the reference, v1 and v2 the candidates, all
captures in one session (the 21.7k-token row needed `--reserve-vram-gib 16`;
the rest ran at 10). No refusals. Relative L2 at layer 50 against BF16:

| population | v1 median / max | v2 median / max | v2 better, rows |
|---|---:|---:|---:|
| `upscale_2048`, all rows | 0.312 / 0.776 | 0.333 / 0.634 | 8 of 13 |
| `upscale_2048`, vision positions | 0.330 / 0.894 | 0.351 / 0.729 | 8 of 13 |
| `max_no_upscale`, all rows | 0.359 / 0.919 | 0.367 / 0.843 | 6 of 13 |
| `max_no_upscale`, vision positions | 0.391 / 1.321 | 0.403 / 0.930 | 6 of 13 |
| text positions, either geometry | 0.064 / 0.086 | 0.063 / 0.083 | 12 and 11 of 13 |
| text-only T2VA bundle | 0.067 / 0.741 | 0.061 / 0.892 | -- |

> **Annotation, 2026-08-26, owner-approved: the verdict stands and the metric
> is the thing it exposed.** Re-deriving the aggregates from the record above
> shows the **means** at `upscale_2048` are a wash -- v1 0.3088, v2 0.3073, v2
> marginally ahead -- while the medians move against v2, and v2's max is 0.634
> against v1's 0.776. At `max_no_upscale` the means favour v1 (0.3991 vs
> 0.4067). So v2 did not degrade uniformly: it cut the tail and lifted the
> middle, and the bar was written on medians and row-win counts, both of which
> reward "improve most rows slightly" and punish exactly that shape. **This
> does not reopen the verdict** -- criterion 1 was pre-registered, it is what
> makes the result legible rather than post-hoc rescuable, and criterion 3
> passed precisely because of the tail behaviour. It is recorded because the
> metric is already named below as this result's third suspect, and this is
> what that suspicion looks like in the numbers. Anyone re-registering a bar
> for a W4 candidate should decide **before the run** whether tail or median is
> the thing being bought.

Against the pre-registered bar: criterion 1 (v2's median below v1's) fails on
both geometries; criterion 2 (10 of 13 rows) fails on both; criterion 3 (no
v2 row worse than v1's worst) passes on both; criterion 4 (text within 2x)
passes, v2 marginally better on text everywhere. **Reject: v2 does not
replace v1.**

What the rows say, **INFERENCE** from the per-row values: v2 improves the
rows v1 was worst on (four rows by more than 0.1 at no-upscale, two at 2048)
and regresses rows v1 was good on by as much (the video-only row 0.158 to
0.357 in both geometries; two no-upscale stills from 0.13 to 0.37). The same
error, redistributed, not reduced. Both artifacts sit at median cosine
0.93 to 0.95 against BF16 on vision rows, so the 4-bit grid is the dominant
term and the calibration moved where it lands. The suspects, in the order
the record supports: the population (29 rows; the observer control had
already shown 14 of 192 mappings flipping their ratio on a one-row set), then
the recipe (`o_proj` unsmoothed under GQA, group 128), then the metric (raw
layer-49 state, where a few high-norm tokens can dominate relative L2;
tokenwise cosine minima are 0.5 to 0.7 for both).

**Two confounds found after the verdict, SOURCE, 17:50.** (1) v1's snapshot
recipe (`config/qwen3vl_32b_minimax_h3_w4a16_awq/recipe.yaml`) records
`duo_scaling: false`; v2 ran with `duo_scaling: true` (the run record's
recipe). v2 therefore differs from v1 in the calibration data *and* in the
scale-search rule, so this comparison cannot attribute its result to the
native-H3 data. Nobody checked recipe parity against v1 before the launch.
(2) The candidate's own `recipe.yaml`, and the snapshot copied from it, is
empty (`default_stage: {}`): the emit path saved after the session had
closed, so the artifact does not carry its recipe. The run record
(`bench/results/2026-08-25_v2_calibration_run.json`) does, and is the
provenance until the emit path is fixed. Both go to the next point.

**Overfit test, MEASURED 18:25** (`bench/results/archive/v2_encoder/2026-08-25_v2_calibration_rows_layer50.json`,
same comparator, v2's own 29 calibration rows): v2's median relative L2 is
11% below v1's there (0.336 vs 0.378; better on 18 of 29 rows, text 24 of
29) against 7% above it on the unseen holdout. Reading, stated before the
number under the rule "at least 15% better on its own rows is the overfit
signature, within 5% means the data barely moved the scales": in between,
leaning overfit. The scales followed the calibration rows' statistics. The
finding that matters more is the floor: both artifacts sit at 0.33 to 0.38
on every population and the data moves that by about a tenth either way, so
at this recipe the ceiling of any data improvement is small. The recipe owns
the result; the data decides the sign of a ten-percent term.

What v2 is still for: it declares the release image bounds and accepts the
2048 view that v1 clamps, at fidelity comparable to v1 on identical replayed
inputs, so the Gate 6 ablation can run on it as the encoder that takes the
view. That is not an acceptance.

**The comparison nobody had run.** The 2026-08-23 record compared W4A16,
INT8 ConvRot and NVFP4 encoders by render time only; its open item was this
layer-50 comparison. Today's instrument grades a W4 artifact; the two
ComfyUI-native encoders need a `--clip-path` through core's own loader. That
is the next measurement: four encoders, one holdout, one reference, and the
answer to which is closest to the release.

## Four encoders on one holdout, 18:45: INT8 ConvRot is the encoder

**MEASURED** (`bench/results/2026-08-25_four_encoders_holdout_layer50.json`;
the two ComfyUI-native files captured through core's own `load_clip` via the
comparator's `--clip-path`, same rows, same replayed inputs, same BF16
reference, no refusals). Median relative L2 at layer 50 against BF16:

| encoder | `upscale_2048` | `max_no_upscale` | text-only |
|---|---:|---:|---:|
| `int8_convrot` (ComfyUI-native) | 0.021 (cos 0.9998) | 0.027 | 0.006 |
| `nvfp4_awq` (ComfyUI-native) | 0.147 | 0.374 | 0.039 |
| W4A16 AWQ v1 | 0.312 | 0.359 | 0.067 |
| W4A16 AWQ v2 | 0.333 | 0.367 | 0.061 |

The rule stated before the table: INT8 well inside half of v1's band on
vision rows makes INT8 the encoder for Gate 6 and the marker arms, with W4
continuing only as the small-host variant. It is fifteen times inside.

**OWNER-DECISION, 2026-08-25 evening:** move to INT8 for the encoder where it
fits. **ACTING-POINT DECISION:** `qwen3vl_32b_minimax_h3_int8_convrot` is the
encoder of record for the Gate 6 ablation and the marker-corpus arms; the
W4A16 lane continues as the small-host variant only, with one GPTQ run on
the same 29 rows as its method check and no further calibration-data work
unless that run changes the floor by more than half. What remains to
measure for INT8 is cost, not fidelity: encode time per prompt and host
residency on the real graphs (the occupancy instrument), since the 26 GB
file streams through dynamic offload on a 24 GB card. The
`duo_scaling false` attribution rerun is parked as superseded.

This closes the open item of the 2026-08-23 record: the encoders were
compared by render time then and by layer-50 fidelity now.

## What this record does not establish

- Anything perceptual. The numerical result is above; whether the
  redistribution is visible in a render is Gate 6's question, and a rendered
  clip cannot A/B a numerical change.
- Whether 214k tokens is enough calibration. It is what fits; the AWQ default
  is a fraction of it, and the role coverage is what mr_data preferred over
  the 18-row alternative. A larger population needs the parent-argument cache
  off the host, which is a code change to the modifier, not a budget.
- A same-subject pair the Hamming-6 window never proposed. The pool-wide
  near-duplicate review was completed later the same day (commit 329cad3:
  every crossing candidate ruled, ten of the last 376 duplicates, 381
  corrected families) and the launched split re-grades green under the
  corrected map; what remains unexamined is whatever the window itself
  cannot see, which the map's derived caveat now states.
