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
resident at 63.9 (`bench/results/2026-08-25_gate2b_row1_expanded_kv.json`,
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

## What this record does not establish

- Anything about the candidate's numerical or perceptual quality. Gate 5's
  acceptance against BF16 on the rebuilt holdout is unchanged and is the next
  thing that runs on the candidate, all three arms in one session because the
  adapter hash is part of capture provenance.
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
