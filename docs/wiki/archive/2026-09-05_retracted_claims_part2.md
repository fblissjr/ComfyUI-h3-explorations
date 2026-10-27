# Retracted Claims Ledger — Part 2 (Rows 21–40, August–October 2026)

> **Archived from [`docs/evidence.md`](../../evidence.md) on 2026-10-27.**
> Full verbatim postmortem notes for rows 21–40 of `## Do not rely on these`. For the active summary table and settled rules, see [`docs/evidence.md`](../../evidence.md).

| claim | why not | what restores it |
|---|---|---|
| **reference-load table: 35.1% / 57.9%** | Wrong on three axes — v1 formula (v2 stops running reference queries dense), 362 frames, 1344x768. The shipped reference arms were 345 at 1024x768, and are 362 since 2026-08-16. **Half redone 2026-08-16**: `docs/SOLATTN.md` now carries a v1-vs-v2 table, with `A` measured from `temporal_shape` and the v1 column reproduced from the doc's own formula as a control. 57.9% becomes **35.6%**, and the 23-point swing becomes **0.5 points**. Still derived — each row's `S` is recovered from the published `exact_kv` percentage rather than counted. **This retraction had existed since 2026-08-16 and three claims in `SOLATTN.md`'s own prose went on asserting the withdrawn numbers regardless** — a retraction in a ledger does not retract its consumers. **Attempted and abandoned the same day, with the reason:** `bench/count_packed_rows.py` makes video and target audio exact and `docs/h3_references.md` gives measured reference costs, but recomputing `S/T` from those reproduces the published `exact_kv` column on only 2 of 5 rows (1.4% vs 1.5%, 29.5% vs 29.6%) and misses badly on the video-reference rows (49.4% vs 35.1% at 345f). **The published `S` values are not reconstructible from anything committed** — the gap is the text estimate at heavy reference load, which needs the encoder. So this row's label cannot become "measured" by arithmetic; it would take a render at each configuration | **nothing, and it no longer matters — see the next paragraph.** The conclusion drawn from this table does not depend on `S` |
| **"with Sol on, sage gets nothing"** | Retracted 2026-08-14. Reasoned from `min_tokens` and forgot the sigma window. Sage runs 5 of 16 steps | — corrected in place |
| **`min_tokens` 4096 is "very likely wrong"** | Retracted, and the retraction's reasoning was itself corrected 2026-08-14: it is one `optimized_attention` call in source but **52 modules** through it. Conclusion survives — the 50 DiT blocks run at the full packed length, above both thresholds, and the 2 refiner blocks run at the text span, below both — so 4096 and 12288 still select the same thing | — corrected in place |
| **"fp16 lands on the steps where precision matters most"** | Inference, not measurement. `start_percent` has never been measured at any length on either backend | measure `start_percent` (Run 2, not started) |
| **any bench progress read from its own stdout mid-run** | The warmup `print` lacks `flush=True`, and `tee` makes stdout block-buffered, so finished lines sit in the buffer. Read ComfyUI's progress lines instead | add `flush=True` |
| **Morton is "worth 1.16x alone", at "94% GPU utilisation"** | Retracted 2026-08-16. Triton, 362 frames, stacked on int8 — correct for what it measured, and not a description of the CUDA backend, where the permutation is free. It had been the standing argument for `morton=False` in `h3_config.py` and in `docs/SOLATTN.md`'s Configuration findings, and the two disagreed for several hours | a CUDA Morton arm at fixed tau, if anyone wants a cost number at all |
| **the CUDA replacement, "0.8 s of 861, or 1.0009x"** | **Do not quote this either**, and it is the more instructive of the two. It is one arm of a two-arm control presented as the isolated number: the dense pair moved +0.8 s and the sparse pair moved −1.2 s, i.e. opposite signs, both at or under the bench's measured run-to-run spread on one run per arm. Morton-on cannot be faster, so the pair is measuring noise. The correct claim is unquantified — **free** | more runs per arm, if the cost is ever worth pinning. Nothing currently depends on it |
| **any A/B of a numerical knob that rests on one rendered clip per arm**, including the 2026-08-13 comparison that chose `fp16 (most accurate)` | **Not controlled evidence about the knob.** The sampling trajectory diverges completely from any perturbation on any sampler, so the changed arm is a *different sample*, not a degraded version of the same one. Sound as a preference between two outputs; never a result about the kernel. Ranked against it, the call-level measurement from `bench/grade_sage_on_capture.py` is the stronger claim, which is the opposite of how the two were weighted at the time | many seeds per arm judged blind in aggregate, or a call-level comparison on captured activations |
| **"the DiT is sensitive to the seven H3 marker rows"**, and the 0.135 relative-L2 figure in [`data/results/2026-08-27_marker_epsilon.json`](../../../data/results/2026-08-27_marker_epsilon.json) read as evidence about markers | **Withdrawn 2026-08-27, the same day, by the control it never had.** Replacing the seven marker embedding rows with the table mean moves the DiT's denoised prediction 0.135 video / 0.148 audio at step 8, at fixed noise, latents and sigma. That number has no null distribution, and a peer session caught it before it was quoted anywhere. The control ([`2026-08-27_marker_epsilon_control.json`](../../../data/results/2026-08-27_marker_epsilon_control.json)) applies the SAME substitution to non-marker rows, matched on what the marker arm actually perturbs -- **16 of 408 prefix positions, not seven rows**, because only two of the seven occur in the prompt and each occurs eight times. At step 8: markers 0.135/0.148, `arb_profile` (2 ids, 16 positions) 0.116/0.212, `arb_draw_1` (11 ids) 0.241/0.430, `arb_draw_2` (8 ids) 0.289/0.418. **The markers sit at the bottom of the control range, not above it.** Two things make this a clean negative rather than a weak one. The controls were perturbed about **6% less hard per row** (0.987 against 1.050 mean relative L2 on the INT8 table -- relative L2 divides by the row's own norm and the untrained marker rows have small norms), so the bias ran the markers' way and they still lost. And the effect tracks the number of **distinct ids** rather than positions: the two-id arms cluster, the eight- and eleven-id arms are two to three times higher at the same sixteen positions, because perturbing eleven words damages more of a prompt than perturbing one word eight times. The markers are two ids, which is why they are at the floor. **The measurement stands; only the reading is withdrawn** -- the DiT plainly responds to those rows, and so it does to any comparable sixteen | **nothing at the prediction level restores it, and the natural next arm was cancelled by this row**: `release_id` vs `mean_init_rows` judged blind was going to cost about seven hours of card time and an hour of the owner's judging, motivated by the rows being load-bearing. It would take evidence on an axis this does not measure -- outsized attention mass on marker positions from `bench/measure_dit_prefix_attention.py`, whose `prefix_marker_token` class has never fired because the only capture carries no markers, or a perceptual separation nobody has looked for |
| **the 2026-08-15 ordering arms read as a matched pair** | **They were never paired.** The arms carry a different seed per clip, so the comparison was two unrelated samples before the divergence argument above even applies. Check the seed before trusting any pair at all | re-render the arms on matched seeds |
| **the fl2va and ref2va time curves differ by "7.8% relative"**, as stated in `bench/convert_pdd_lora.py`'s docstring and `docs/h3_pdd.md` | **UNSUPPORTED, marked 2026-09-03.** The figure dates from a 2026-08-26 comparison that no dated record under `bench/results/` carries, so it has no conditions and no origin a reader can run. The direction stands (`h3_pdd.silu_temb_grid` is partition-specific, which is why the converter derives it from `--base`); the size is a copy with no home. `docs/pdd_artifacts.md` points here instead of repeating it | a record from a script that compares the two partitions' `silu(time_embedder(t))` grids over the 1025-row time grid, or delete the figure from the two places that state it |
| **block 49 "carries about four times any other block's error with every block routed on both builds", and "no `dense_blocks` list touches it"**, as first stated in `CHANGELOG.md` 0.99.27 for the Base16 capture records | **Withdrawn 2026-09-03, the same day, on Codex's review.** True only on the block-max build and only under the norm-weighted whole-tensor relative L2; per row block 49 is barely above block 40, and on the running-max build block 40 is the worse one (`bench/results/2026-09-03_sol_exact_base16_capture_{d25f2e8,24908e1}.json`, `exact_all_routed.rel_l2` against `rel_l2_row_mean`). And a `dense_blocks` entry sends the block to sage, bypassing Sol's exact stage entirely, so it does touch it. What stands: a large exact-stage hotspot on the block-max build under the whole-tensor metric, in a few high-norm rows | an all-50-block Sol-versus-sage record on the production trajectory, then a candidate set tested as a set |
| **the Base16 records' `tau_1.0` arm is "the shipped tau"**, and (briefly) "an upper bound on the shipped call's error" | **Withdrawn 2026-09-03.** The node passes sink ranges from the segment table and a capture taken with Sol absent has no table, so the arm is the shipped tau without the shipped call; renamed `tau_1.0_no_sinks`. The "upper bound" that replaced the first label lasted an hour: forced-pair counts are monotone in the sink ranges, relative L2 and cosine are not, because errors can cancel. It is an unsunk diagnostic | a capture whose records carry the segment table (a neutral layout-publication helper armed by Sol or `h3_capture.py`), then the arm re-run with the node's own sink ranges |
| **any reading of the 2026-09-04 ladder verdict as settled: "PDD8 is visibly worse", "sage and Sol are indistinguishable from dense", or "fp16 sage helps"** ([`data/results/2026-09-04_ladder_2026-09-03_verdict.json`](../../../data/results/2026-09-04_ladder_2026-09-03_verdict.json)) | **Not withdrawn; bounded from the day it was written.** One seed per arm, so each contest is one stack judged once. Within a scene every contest shares one clip per rung, so a rung that lost twice in a scene lost on one clip and one defect seen from two stacks. The singles were not scored, so nothing audible was judged and the manifest's audio questions stand open; and stacked pairs were silent by the stacker's design in every session before `3637f6d` (2026-09-05), so no pair verdict in this or any earlier session judged anything audible, whatever the singles carried. **And the PDD8 rung is confounded**, as the owner said on reading the result: every rung above dense carries sage, the three Sol rungs carry Sol, and pdd8 carries the PDD-specific Sol `end_percent` besides; no arm renders PDD8 without Sol or without sage, so its losses attribute to nothing narrower than "the shipped PDD8 graph". What the record does give: the owner's note on every pair resolved through the key, and a direction per scene. The one result consistent across all five scenes is that the shipped PDD8 graph lost to the dense baseline on a named defect each time, and the owner named it as PDD unprompted on two of those pairs while blind. The audio half is the owner's report in conversation, kept in the record's `owner_report_after_join` with that provenance (good across the board, a bit quieter on the PDD clips), and [`data/results/2026-09-04_ladder_audio_loudness.json`](../../../data/results/2026-09-04_ladder_audio_loudness.json) is the per-clip level beside it, one clip per arm, descriptive | PDD8 under stock attention and under sage alone on the same scenes and seed, so the shipped pdd8 clips become one rung of a three-rung PDD ladder with its own baseline (`workflows/build_workflows.py`, the `dense_attn` extra); then a second seed on the same scenes and rungs (the 2026-09-03 session postmortem, forward item 4), and the singles scored through the page for the audio half |
| **"PDD8 is worse than sage at sixteen steps" or "Sol on two of eight steps is free" read as settled from the PDD ladder verdict** ([`data/results/2026-09-05_pdd_ladder_2026-09-04_verdict.json`](../../../data/results/2026-09-05_pdd_ladder_2026-09-04_verdict.json)) | **Bounded, 2026-09-05.** One seed, five scenes, pairs only; the singles were not scored, so audio rests on the owner's pair notes ("louder" for the floor on every compared scene) and on `data/results/2026-09-04_pdd_ladder_audio_loudness.json`, which is descriptive. What the record does support directionally: the floor won all five against PDD8 under sage alone and four with one can't-tell against the narrow window, so the PDD8 loss attributes to PDD itself (schedule or merge, undecided until `pdd_bake_2026-09-05` is scored) and not to Sol; Sol on two of eight steps read same as sage alone on all five; the shipped four-step window lost on two (standoff, subway). Two of the named defects (the diner "menu across the wall", the subway artifacts) appeared on every PDD arm of their scene and on neither floor clip. |
| **"the bake fixed PDD8" or "the bake changed nothing"**, read from the bake pair verdict ([`data/results/2026-09-05_pdd_bake_2026-09-05_verdict.json`](../../../data/results/2026-09-05_pdd_bake_2026-09-05_verdict.json)) | **Bounded, 2026-09-05.** One seed, five scenes, pairs only, one verdict corrected by the owner after the join, exported value kept beside it in the file. Directionally: the baked arm lost to the sage floor on four scenes and read same on one, on the properties the merged arm lost on the day before (level, lighting, framing, the subway sign), so the PDD8 defects attribute to the eight-step schedule and not to the requantised merge at this seed. Merged against baked read same on two, merged won two and baked won one on composition, so the merge noise's visible effect is not larger than one sample's variation. The bake's stored-weight case (`2026-08-31_bake_realisation.json`) is untouched by this; what it did not deliver is a visible quality change. |
| **"PDD's audio loss is ComfyUI's audio change of variable", and any reading of `bench/measure_pdd_audio_carry.py` or of the mean-sigma arms in [`data/results/2026-08-28_audio_carry_ablation.json`](../../../data/results/2026-08-28_audio_carry_ablation.json) as measuring or fixing a carry error** | **Refuted analytically, 2026-09-25.** Given the same velocity, core's single-schedule carry at the block start lands exactly where the vendor's two-schedule audio Euler step lands, at every partition tried; moving the coefficient off the block start is what produces an error ([`bench/results/2026-09-25_upstream_pdd_comparison.md`](../../../bench/results/2026-09-25_upstream_pdd_comparison.md) section 3). `measure_pdd_audio_carry.py` compares against a block mean of the transform, not against the vendor's integrator. The ablation's energy gains are real observations of a departure from the exact path, not of a repair. Scope: Euler at the scheduled sigmas, with or without a uniform audio mask | nothing restores the mechanism. The observations stand as they are recorded, and the audio loss at coarse partitions is PDD's own. Not compared: the sampler's inpaint blend, which has no vendor counterpart |
| **"Sol without sage is worse than the sage floor" as settled**, read from the just-Sol verdict ([`data/results/2026-09-05_sol_nosage_2026-09-04_verdict.json`](../../../data/results/2026-09-05_sol_nosage_2026-09-04_verdict.json)) | **Bounded, 2026-09-05.** One seed, five scenes, pairs only, singles unscored. The floor won four with one can't-tell and Sol as shipped won three with two same, but the notes name composition on all but one pair (framing, the orchestra behind the singer, the diner menu, the suspect behind the chaser), which the owner then ruled count as prompt adherence (2026-09-05 evening; the record was first read as sample divergence and says so); the one rendering-quality loss is stairwell (lighting, skin fidelity, level). Direction against just-Sol on adherence at one seed; the shipped chain stays the leader on the 2026-09-03 ladder verdict and on this. |
| **"the encoder pathway carries identity on its own" or "reference rows are just a label"**, read from the 2026-09-04 reference pathway verdict ([`data/results/2026-09-04_ref_pathway_2026-09-03_verdict.json`](../../../data/results/2026-09-04_ref_pathway_2026-09-03_verdict.json)) | **Not withdrawn; bounded from the day it was written.** One seed, pairs only, no single heard. The owner could not tell the both-pathway arm from the encoder-only arm on either conditioner and named no identity item as differing; that is a "can't tell" at one seed, which bounds the effect as not large on this scene and says nothing about its sign. The one thing named, static background people on every encoder-only arm, is a signature at one seed and not a mechanism. The two encoder-only arms read as the same and are different renders ([`data/results/2026-09-04_ref_pathway_encoder_arms_pixels.json`](../../../data/results/2026-09-04_ref_pathway_encoder_arms_pixels.json)) | a second seed on the same five arms (the manifest's `run` line), the singles scored, and the static-people signature either recurring or not |
| **Sol and the shipped sage fallback were compared "on one footing"** via [`2026-09-03_sage_on_base16_capture.json`](../../../data/results/2026-09-03_sage_on_base16_capture.json) | **Withdrawn 2026-09-03.** That record scored 512 sampled rows against float64; the Sol records scored every row against fp32. Same statistic name, different sample and reference. The Sol records were regenerated with `sage_<mode>` arms on the same q/k/v, every row, the same reference, and the sage arms agree across both kernel builds; the direction they give is in `docs/research/2026-09-03_sol_exact_pquant_and_base_capture.md`'s addendum. The separate sage record stays as what it was | nothing further: the matched arms are the comparison to cite |

---

## Environment, because it is not in git

The venv and ComfyUI were both changed on 2026-08-14 and neither is
version-controlled here. Updated 2026-08-22, where noted:

- `comfy-kitchen` replaced: `0.2.31` → `0.2.31+sol.c04ef20`, **built from
  source for sm_89 only**. It will not work on another architecture, and it
  declares a version the stock wheel also declares.
  **Rebuilt 2026-08-22 to `0.2.31+sol.23d1a66`**, the same branch five commits
  on, for the rewritten routing kernel and `topk_ratio`. Everything above
  still holds; only the tag moved.
  **Rebuilt again 2026-08-29 to `0.2.31+sol.dae00a1`** -- and this one is NOT
  just a tag move. Sol-Attn merged upstream (comfy-kitchen#117) with a
  reshaped API: `centroid_tail`, `reuse_qkv_memory` and `max_blocks` gone,
  `tail`, `block_len` and `coarse_gate` added. Measured against the branch
  build on a matched arm with two controls
  (`data/results/2026-08-29_sol_kernel_merge_equivalence.json`): the
  non-Sol baseline is bit-identical across the swap and the Sol arm differs by
  **rel L2 7.67e-05** on video, nothing on audio. So every Sol figure above
  survives, and **a figure quoted to more than three significant figures
  across the two builds does not.**
- `nanobind` 2.14.0 added.
- `custom_nodes/ComfyUI-SolAttn-cuda/` created; its `sol_attn_minimax.py` is a
  **symlink into this repo's `vendor/`**. Editing through the installed path
  writes into the tracked file.
- `coderef/comfy-kitchen-kijai/` cloned with submodules; at `c04ef20` from
  2026-08-14 and at `23d1a66` since 2026-08-22. Its `pyproject.toml` is
  **edited during a build only** — `vendor/rebuild_kernel.sh` applies the
  version-tag patch, substitutes the checkout's own short sha, and reverts on
  every exit path, so the tree is left as found. It was carried as a standing
  working-tree modification until 2026-08-14, which blocked `git pull` outright
  rather than conflicting loudly as intended.

**The model build is software substrate too, and it was missing from this list.**
`workflows/h3_config.py` points at `minimax_h3_fl2va_pruned_int8_convrot` and
`minimax_h3_ref2va_pruned_int8_convrot`, so every number in this repo was measured
on weights that are **pruned, convrot-rotated and int8** — three separate
properties, not one. `fp8_scaled` and `w4a8_mixed` builds of the same models sit
on disk beside them.

**What "pruned" is, and what it costs -- measured 2026-08-20 against the
unpruned `int8_convrot` files** ([`data/results/2026-08-20_adaln_pruning_residual.json`](../../../data/results/2026-08-20_adaln_pruning_residual.json),
`bench/analyze_adaln_pruning.py`). The unpruned DiT carries a timestep MLP and
a full-width AdaLN projection per block (`[96768, 2688]`; the projections are
41% of the DiT's parameters). The pruned file replaces them with the rank-8
SVD of the mean-centred `silu(e(t))` curve: `adaln_t_table` is `U @ Sigma`
(its column norms equal the singular values to four digits, and the fit residual
equals the SVD residual) and the curve's mean is folded into each block's bias.
Every other tensor is byte-identical between the pruned and unpruned file of
the same checkpoint, so the pruned build is the unpruned build with the AdaLN
swapped and nothing else. At the modulation output the swap costs 0.1-0.2%
of the whole output per block and 0.06-0.35% at the timesteps a render
evaluates; on the final layer, which the unpruned file stores in bf16, the
residual is ~0.02%, so most of the per-block figure is int8 error on the
*unpruned* side, and the rank-8 truncation itself is ~0.02%. **fl2va and ref2va
lose the same**: the per-block residual ratio ref2va/fl2va runs 0.97-1.01. The
hypothesis that ref2va compresses worse under its own factorisation, the one
ref2va-specific mechanism this repo had found with a concrete test, is refuted.
**And the sensitivity to it, measured 2026-08-21**
([`data/results/2026-08-21_pruning_sensitivity.json`](../../../data/results/2026-08-21_pruning_sensitivity.json),
`bench/grade_pruning_sensitivity.py`, `docs/open_experiments.md` #22). A
perturbation of that size still moves the network output by **5.6-9.4%**
relative L2 on the first-step velocity, at a fixed input. That is not
invisible, and it is roughly two orders of magnitude larger than the
modulation residual above, so **do not read "the pruning costs 0.1-0.2% of the
modulation" as "the pruning costs 0.1-0.2% of the output"** -- the error
compounds through the stack, peaking around block 36. What keeps it in
proportion is the scale beside it: on the same input the `fp8_scaled` build
differs from `int8_convrot` by 12-21%, so the pruning is smaller than a
quantisation choice this repo already ships.

**fl2va and ref2va are equally sensitive**, which closes the ref2va question
from the other side: ref2va moves 0.86 of what fl2va moves on the reference
input and 0.81 on the plain one, consistently in the opposite direction to the
hypothesis. The determinism floor was exactly zero -- the repeat arm reproduced
its twin bit for bit across a checkpoint reload -- so none of this is noise.
**The entry's pre-registered prediction (under 1%) was wrong**, and is recorded
as wrong in the record's caveats rather than dropped.

Side fact from the same record: the time embedder itself differs between the
two checkpoints (10.6% relative, cosine 0.994), so the fine-tune moved `e(t)`,
not only the projections. Still unmeasured: the int8 error of the linears
against the bf16 release, which needs the bf16 files.

This is not only bookkeeping. `convrot` applies a rotation (`docs/roadmap.md`,
established while grading the ref LoRA), so any claim reasoning about q/k geometry
— centroid similarity, block locality, what a token ordering buys — is reasoning
about a rotated, pruned, quantized space. Whether that reaches Sol's routing or
the orderings is unmeasured and filed as open experiment 19.

**Where a run does and does not record it.** A conforming capture manifest already
says which build ran: `models` requires `unet`, `clip` and `video_vae`, and those
filenames are self-describing. But `weight_quantization` and `vae_quantization`
were in no `required` list and `bench/check_capture_manifest.py` inspected neither -- **fixed 2026-08-17**, both now required and asserted,
so a manifest can omit both and still pass. **A bench run emits no manifest at
all**, so a timing carries no record of the weights underneath it — the same shape
as the power-limit gap below, and enforced by nothing.

Record the node hash and the `comfy-kitchen` tag with any measurement. The
branch rebases and the build declares a stock version number, so nothing else
can tell two builds apart.

**The host is the other half of this, and `docs/hardware.md` owns it.** That
list is software; the machine underneath has its own ungoverned state, and the
one that moves timings is the GPU board power limit. It is set outside the repo,
appears in no workflow, log or capture manifest, survives reboots once a systemd
unit exists for it, and changes s/it without changing a single output byte.
`bench/hwinfo.py` prints it and says so when it is not stock — run it alongside
the node hash and paste both. Nothing enforces this.

The power limit on this box **was changed away from stock on 2026-08-17**, after
every timing on `docs/bench_plan.md` and in `docs/SOLATTN.md` had been recorded.
Those measurements are not retracted — they were correct at stock — but
reproducing one now means checking `hwinfo.py` first and resetting the limit if
the comparison is meant to be like-for-like.
