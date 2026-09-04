# Sol-Attn token routing (`token_aug`) graded on the Base16 capture

last updated: 2026-09-04

**This file owns what kijai's token-routing change to Sol-Attn does on this
repo's own captured activations, at the call level, and what that does and
does not license.** The records are
[`2026-09-04_sol_exact_base16_capture_1128df6_token_aug.json`](../../bench/results/2026-09-04_sol_exact_base16_capture_1128df6_token_aug.json)
(twenty-five Base16 cells, every arm against one fp32 dense reference) and
[`2026-09-04_sol_exact_random_1128df6_token_aug_timing.json`](../../bench/results/2026-09-04_sol_exact_random_1128df6_token_aug_timing.json)
(the isolated kernel call at the capture's rows and heads), both produced by
[`bench/measure_sol_exact_variants.py`](../../bench/measure_sol_exact_variants.py)
with its `--token-aug` arms added the same day. The upstream change is
[Comfy-Org/comfy-kitchen PR 156](https://github.com/Comfy-Org/comfy-kitchen/pull/156);
[`../sol_upstream.md`](../sol_upstream.md) holds the survey it came out of.
Nothing here was rendered, and a rendered clip could not have A/B'd it
(`CLAUDE.md`).

## Conclusions

- **Token routing lowers Sol's error against exact attention on four of the
  five captured blocks at every captured step, and raises it on block 49 at
  every step.** The aggregate hides the second half: read the per-file rows.
  Per head, block 49's loss is a few heads losing a lot (the same head at
  every step) while most heads gain a little; under the norm-weighted
  metric those few heads dominate.
- **The budget knob is nearly inert.** Sixty-four, one hundred twenty-eight
  and two hundred fifty-six tokens land within a hair of each other in
  accuracy and in time, and on random inputs they are identical to every
  digit. The kernel admits whole histogram bins until the budget would
  overflow (`sol_attn_token.cu`, header comment on that branch), so the
  budget selects a bin edge rather than a count.
- **The cost is one isolated-call number, not a render time**: the timing
  record carries it beside the plain call at the same shape, on the same
  wheel, in the same process. It is the only place that figure lives.
- **So `token_aug` is a per-block policy candidate, not a global switch.**
  Roadmap step 7 (forward plan 2026-09-04) said "if it holds on our capture
  it moves the frontier under every policy above"; it does not hold on block
  49, so it moves into step 4, the block policy, as one more thing the probe
  can rank per block. It goes in through the vendoring rule when and if it
  goes in; nothing is installed.
- **The control held exactly.** The scratch wheel's plain arms reproduce the
  2026-09-03 installed-wheel record
  ([`2026-09-03_sol_exact_base16_capture_d25f2e8.json`](../../bench/results/2026-09-03_sol_exact_base16_capture_d25f2e8.json))
  to every printed digit on every aggregate, so the two builds agree
  wherever the knob is off and the delta below is the knob's alone.

## 1. What was graded, and on what

| | |
|---|---|
| kernel | kijai's `sol_token_aug_main` at `1128df6`, the PR 156 head on 2026-09-04, fetched into the workspace clone behind `coderef/comfy-kitchen` as `kijai/sol_token_aug_main`; built from a detached worktree of that sha with the version line tagged `+sol.1128df6`; wheel kept under that clone's `dist/`; unzipped into a scratch directory and selected through `PYTHONPATH`. The venv's wheel (`0.2.31+sol.d25f2e8`, `.venv/comfy_kitchen_build.json`) was not touched. The record's `kernel_source` field says the same |
| capture | the 2026-09-03 Base16 dense-trajectory control, five blocks by five steps, manifest `bench/results/2026-09-03_capture_manifest_base16.json`, retained outside the repo |
| reference | fp32 chunked softmax attention per head over every row, the grader's own |
| arms | the plain `topk_0.10` and `tau_1.0_no_sinks` arms of the 2026-09-03 record, each repeated with `token_aug` at 64, 128 and 256; sage arms skipped (`--no-sage`), already on file for these cells in the 2026-09-03 record |
| footing | identical to the 2026-09-03 record: same files, same reference, same metrics, same script; the only difference is the wheel and the knob |
| what `tau_1.0_no_sinks` is | an unsunk diagnostic, not the shipped call: the capture carries no segment table, so no sink ranges are passed (the 2026-09-03 note, addendum) |

## 2. Direction

Row-mean relative L2 of the `token_aug=64` arm as a ratio of the plain arm
on the same cell, `tau_1.0_no_sinks`, from the record's `per_file` rows.
Below one is better. Step order 4, 8, 12, 14, 15.

| block | ratio per step | worst-head cosine, plain to aug (step 15) |
|---|---|---|
| 0 | 0.68, 0.62, 0.60, 0.60, 0.60 | 0.9238 to 0.9891 |
| 24 | 0.79, 0.76, 0.73, 0.71, 0.71 | 0.9699 to 0.9696 |
| 32 | 0.70, 0.68, 0.69, 0.67, 0.68 | 0.9559 to 0.9556 |
| 40 | 0.58, 0.58, 0.61, 0.60, 0.61 | 0.9415 to 0.9376 |
| 49 | 1.11, 1.18, 1.25, 1.30, 1.32 | 0.9588 to 0.9033 |

The `topk_0.10` arm moves the same way on every block (ratios in the
record's `topk_0.10_token_aug_64` rows), with block 49 again the one that
worsens. Aggregate over all twenty-five cells, both arms, all three budgets:
the record's `aggregate` block; the direction there is "better", which is
four blocks outvoting one.

**Block 49 per head** (`per_head` lists in the record): at step 15, eight
heads of fifty-six lose and fourteen gain under `token_aug=64`; the largest
loss is on head 11, the same head at every step, and it is several times
the largest gain. The whole-tensor relative L2 on block 49 rises by more
than the row mean does, so the losing heads carry heavy rows. Block 40, the
largest gainer, is the mirror: its plain worst head is the worst in the
whole capture and token routing lifts it most. Neither the mechanism behind
head 11 nor whether it is visible is established here.

## 3. Cost

`kernel_ms` against `kernel_ms_token_aug_{64,128,256}` in the timing record:
an isolated warm call at the capture's shape (one batch, the capture's rows,
fifty-six heads), median of ten, plain versus each budget, in one process on
one wheel. Comparable to nothing outside that record: not to a render, not
to the 2026-09-01 timing rows at a different shape, and not to the PR's own
"about a third more attention time", which was measured in kijai's setting.

## 4. What this changes, and where

- **Roadmap step 7** now reads with this result beside it: a per-block
  candidate for step 4, not a frontier move. The online probe
  (`sol_block_probe.py`) is the instrument that would rank it per block on
  the shipped call with sink ranges, which this offline record cannot.
- **Nothing is installed.** The shipped kernel is unchanged and
  `bench/check_sol_kernel.py` reports which one runs. If token routing is
  adopted for a block set, the build goes through `vendor/rebuild_kernel.sh`
  from a clean worktree at a chosen sha, which writes the venv's build
  record, and the node gains a knob; both are node code and a restart before
  anything counts.
- **The `blk_cnt` out-parameter does not cherry-pick onto that branch**
  (conflicts in four files, tested 2026-09-04), so a wheel carrying both the
  probe's count and token routing is a rebase.
- **The grader gained three flags** (`--token-aug`, `--limit`,
  `--kernel-source`); its first run on a new arm was one file, read in
  full, before the record run.

## Not established

- that the block 49 loss, or the other blocks' gains, produce a perceptible
  render difference; a blind pair per candidate is the standard
  (`../eval_comparison.md`, section 3), one seed anchors and does not decide;
- what token routing does on the shipped call with sink ranges: every arm
  here is unsunk, and errors can cancel, so the shipped call's delta is not
  bounded by these;
- the mechanism behind head 11 on block 49, or whether it is the same head
  the 2026-09-03 records single out under other arms;
- whether the direction holds on Base20, PDD8, FL2VA or Ref2VA, each its own
  population;
- whether kijai's named symptom (a brightness pulse on a five-latent-frame
  period, his PR body) exists in our renders at all; nothing here measured
  a render, and the ladder verdict names brightness on a PDD8 pair, not a
  Sol pair;
- any render-time cost; the timing arm is an isolated call.
