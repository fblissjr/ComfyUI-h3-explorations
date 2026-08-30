# Where in the DiT does PDD matter, and what else can be turned

**Opened 2026-08-30.** Two questions that were being conflated with a third.
The third — per-block *quantisation* sensitivity — is answered and closed;
these two are not, and this file says what would answer them.

Owner's framing, and it is the right one: *"it's not just about quant error per
block. It's where in the DiT does running undistilled, or at a lower distill or
a higher distill, make the most difference."*

---

## 1. What is settled, so nobody re-derives it

**Per-block quantisation sensitivity is not a variable.** int8-vs-bf16
stored-weight error spans **1.086x** across the 50 blocks
(`bench/results/2026-08-30_pdd_block_magnitude.json`). There is no sensitive
block. What spread exists is by module KIND — `attn.out_proj` runs 1.18x
`qkv_proj` — not by depth.

**Merging PDD into an int8 weight inflates that error, and the inflation
tracks the LoRA rather than the base.** 0.00942 -> 0.01058 at strength 1.0,
correlating 0.78 with the module's own `||BA||/||W||`
(`bench/results/2026-08-30_pdd_quant_interaction.json`). So a strength
schedule keyed to quantisation would turn PDD down exactly where PDD is doing
the most work. **That plan is dead**; `docs/h3_pdd.md` carries the reasoning
and the two fixes that replaced it.

Neither of those answers the depth question. They are about the *quantiser*.

---

## 2. Four depth profiles, and they disagree

This is the actual state of knowledge about depth in this DiT. Every column is
measured; no column is about PDD strength.

| depth | perturbation → output latent | local error AT block | PDD's own update | fl2va↔ref2va distance |
|---|---|---|---|---|
| 0-4 | **highest**, 0.0306 at b0 | lowest, 0.097 | low, 0.0043 | **0.216 — most shared** |
| 5-14 | ~0.024 | rising | **plateau, 0.006-0.008** | 0.319 |
| 15-29 | ~0.024 | rising | plateau | 0.319 |
| 30-39 | bump at b32, 0.0272 | 0.172 | trough, 0.003 | 0.324 |
| 40-47 | falling, 0.0177 | **highest, 0.223** | trough | 0.324 |
| 48-49 | **lowest**, 0.0110 at b48 | 0.191 | **spike, b49 0.0173** | 0.325 |

Sources, in column order: `2026-08-29_block_propagation.json`,
`2026-08-29_dense_block_ranking.json`, `2026-08-30_pdd_block_magnitude.json`,
`2026-08-29_h3_partition_distance.json`.

**Read across, and the disagreement is the finding.**

- **Columns 1 and 2 run OPPOSITE.** Local error grows with depth; how much of
  it reaches the output shrinks with depth. `workflows/h3_config.py` records
  this as the reason `dense_blocks` is `0-2,32` and not the tail — propagation
  decides the knob, local error does not.
- **Column 3 is anti-aligned with column 1.** PDD's update is smallest exactly
  where perturbations propagate hardest, and largest at block 49 where they
  propagate least. Whatever the distillation was solving for, it was not
  "spend effort where it travels furthest".
- **Column 4 says the first five blocks are the shared ones.** fl2va and
  ref2va are 0.216 apart through block 4 and a flat ~0.32 everywhere after. The
  front of this model is the part the two partitions agree on, and it is also
  the part that reaches the output hardest.
- **Block 49 is its own case in every column** — lowest propagation, high local
  error, and by far the largest PDD update (`blocks.49.mlp.fc2` at 0.044, ten
  times a typical module). It sits directly on the output head PDD replicates.

---

## 3. The question nobody has measured

**Where does varying PDD STRENGTH change the output most?** Not the quantiser,
not Sol's kernel error — the distillation itself, per block, including above
1.0.

The machinery exists and has never been pointed at it.
`bench/probe_block_propagation.py` runs one perturbation at exactly one block,
saves the output latent, and scores rel L2 against a baseline; its arms live in
`output/latents/h3_block_propagation/`. It produced column 1 above, for Sol.

### The design trap, and it is why this is not a one-line repoint

**That probe is controlled by an accident of Sol's shape that a strength change
does not share.** Sol takes a sigma window, and the probe sets it to contain
only the FINAL step — so steps 0-2 are bit-identical across arms by
construction, and the saved latent differs only by what happened at one block
on one forward. That is what buys it exemption from CLAUDE.md's different-sample
rule.

**PDD's strength is a weight patch applied at load. It affects every step.** So
arms would diverge from step 0, the trajectories would separate completely, and
the output latents would be different samples rather than a perturbation and
its baseline. The probe would return numbers, and the numbers would mean
nothing. **Do not run it that way.**

Two designs survive the trap:

1. **Gate the delta to one step.** Only possible for an UN-MERGED block, because
   only those apply their delta at the call rather than in the weight — so
   `unmerged_blocks` is what makes a controlled strength probe reachable at all.
   Needs a sigma window on the un-merged forward: the node already installs a
   capture patch on `diffusion_model.forward` that runs before the blocks and
   receives the timestep, so the current sigma is available to record.
2. **One forward, no sampler.** Fix an input at production shape, run a single
   forward per arm, read the output. Controlled by construction and cheaper.
   This is the method `grade_sage_on_capture.py` uses and CLAUDE.md names as the
   only controlled comparison available for a numerical knob — but **no harness
   here runs a full DiT forward with modified weights.** The capture graders
   replay attention only. This one has to be built.

### The arms the owner asked for

Strengths **0.0, 0.5, 1.0, 1.5** per block — off, half, shipped, and
over-distilled. The last has never been tested at any block, and nothing says
the vendor's 1.0 is a maximum. `strength=1.0` at every block is the shared
baseline, so a block costs three arms, not four. At the eleven blocks column 1
used, that is 33 arms and one baseline.

Canvas: **1344x768 at 345 frames**, owner's choice, 14.375 s and legal
(`h3_rules.snap_length(345) == 345`, `latent_t` 87). Close to the 362-frame
maximum, so this is a production-length answer rather than a cheap-canvas one.

---

## 4. What else can be turned, besides strength

Read off `pdd_lora.py`'s schema and execute path on 2026-08-30. Three of these
are exposed, three are structurally present and unreachable, and the split
matters.

### Exposed today

| knob | what it reaches |
|---|---|
| `strength` | backbone AND adaln together — see the gap below |
| `head_strength` | the fused output heads alone, sentinel -1.0 follows `strength` |
| `patch_heads` | heads entirely off: the backbone-and-adaln control arm |
| `steps` | which partition of the 32-point grid the render walks |
| `nfe` | forces a uniform block count, ignoring the schedule — the deliberate off-schedule arm |
| `unmerged_blocks` | placement: which blocks apply their delta at the call rather than in the weight |

### Present in the design, not reachable from a graph

- **The adaln update cannot be scaled apart from the backbone.** Both are
  folded into one dict and applied by a single `add_patches(loaded, strength)`.
  `head_strength` exists because the three surfaces are worth ablating apart;
  that argument was made for the heads and never finished for the modulation.
  **This is the cheapest missing axis** — the keys are already separable at the
  point they are assembled.
- **Per-block strength.** `ModelPatcher.add_patches` records a strength per
  call, so splitting the patch dict by block and calling it once per group is
  about ten lines. Measured to be misaimed for quantisation (§1); **unmeasured
  for quality**, which is the open question in §3.
- **Per-kind selection.** `unmerged_blocks` moves a whole block's four
  projections. The by-kind quantisation spread is real — `attn.out_proj` at
  1.18x `qkv_proj` — and nothing can act on it.
- **The partition SHAPE.** `envelope_partition` front-loads wide blocks by a
  rule, chosen because the tail governs quality. It is one legal tiling among
  hundreds; `2026-08-28_handoff.md` records a DP over 528 spans finding
  `[28,2,1,1]` minimises the worst per-block fusion loss. Not selectable.
- **Per-INTERVAL head weighting.** The bank is 32 replicated heads and
  `fusion_plan` averages a block's heads by step size. Weighting them otherwise,
  or scaling `head_strength` per grid interval rather than globally, is
  untouched — and CLAUDE.md's tail findings say the tail is where quality
  lives, so this is the axis the partition work points at.

### Two more axes, both reported by the mask-lane session on 2026-08-30

**The SAMPLER, and it is not free the way it looks.** `docs/h3_pdd.md` records
that a second-order or stochastic sampler is *structurally invisible* to the
node's boundary guard: heun evaluates its corrector at `sigmas[i+1]`, which IS
the next block boundary, so the guard measures ~0 distance and calls it
healthy while every corrector selects the NEXT block's fused head. Half the
model evaluations use the wrong head and nothing says so. So sampler is an
axis with a known trap, not an open dial — and any depth probe must stay on
euler for the same reason `probe_block_propagation.py` does.

  *Related, and this paragraph is a correction of itself:*
  `MiniMaxH3EulerAncestralEta0SchedulerAdapter` carries eta=0 and is therefore
  plain euler rather than `euler_ancestral`. **This file claimed on 2026-08-30
  that the class is "vllm-omni's, not sglang's". That was wrong and is
  withdrawn.** The class exists in BOTH trees — sglang's at
  `coderef/sglang/python/sglang/multimodal_gen/runtime/models/schedulers/scheduling_minimax_h3_euler_ancestral.py:137`
  and vllm-omni's at
  `coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/scheduling_minimax_h3_euler_ancestral.py:105`.
  They are near-identical and not identical, so one vendors the other and the
  direction is not established.

  **The cause was a search that did not follow symlinks**, not a naming
  collision: `coderef/sglang` is a symlink, and `find`/`grep -r` skip those by
  default, so the search that "established" the attribution could only ever
  have returned vllm-omni. CLAUDE.md now carries the general rule.

  **What is true about sglang's sampler is narrower than either version said.**
  Its H3 pipeline does not instantiate the adapter at all —
  `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines/minimax_h3_pipeline.py:45-48`
  says the
  scheduler is intentionally absent, `model_index` carries `scheduler=null`,
  and stages accept `scheduler=None`. The denoise loop runs its own in-place
  step. So sglang does run deterministic euler, but in the loop rather than
  through that class, and the class is where the math is *documented* rather
  than where it *executes*. Reported by the mask-lane session, verified here by
  reading both files. `docs/research/sglang_h3_pipeline.md` owns sglang and
  already attributes the class correctly.

**MASK GRANULARITY, and it is gated by ComfyUI rather than by the model.**
Reported by the mask lane, measured by them, not re-derived here:
`MiniMaxH3._token_grid_masks` does `ceil(mask * 256) / 256` and
`_denoise_mask_values` drops the mask entirely once `amin >= 1 - 1e-3`. So the
finest sub-1 strength the model can be given is 255/256; anything above is
promoted to 1.0 and discarded. Every value on that grid is bf16-exact, so the
`_apply_model` cast costs nothing at stock precision.

Three reasons it is not a bench axis today, and the third is the one that
decides it:

  1. **Nothing here produces a graded mask.** Established rather than
     inferred: no graded mask node in the conditioning or keyframe lanes, and
     no shipped graph wires a mask-typed node at all.
  2. **Reaching off the 1/256 grid needs a process-wide patch** of five
     ComfyUI methods that does not retire until restart, and it changes an
     unrelated graph's feather quantisation afterwards. Unusable as an axis
     without a process boundary between arms — which collides directly with
     this repo's new cache-state rule.
  3. **There is an open upstream correctness bug on that path**
     (Comfy-Org/ComfyUI#15981, #15978, fix proposed in #15988): the model
     conditions each row at `1 - r*sigma` while `CONST.calculate_denoised`
     converts with the global sigma. The error scales as
     `m*(1-r)*sigma*(x0-noise)` — zero at `m=1` and `m=0`, largest in the
     middle. **So the error varies with the same knob you would be varying**,
     and the axis is not measurable until the fix lands. That is the
     disqualifier; the other two are merely obstacles.

  Their writeup is `internal/20260830_motion_context_multiref_analysis.md`.
  Cite the measurements from it, not the third-party pack.

### Fixed, and not a free parameter

**The shift.** PDD's block boundaries ARE the shifted schedule at 12/3, bit for
bit. A PDD arm moves the step count and nothing else; changing the shift breaks
the correspondence the whole mechanism rests on.

---

## 5. Not established

Everything in §1 is stored-weight distance. Everything in §2 column 1 is one
seed, one prompt, one canvas, one step, on the BASE model with no PDD in the
path. §3 has not been run at all. No number here is perceptual, and the
different-sample rule means none of them can be turned into one by rendering a
pair and looking.
