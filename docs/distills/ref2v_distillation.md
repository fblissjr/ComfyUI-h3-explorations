# Why ref2v resists step distillation

**Ref2v is a different problem wearing the same interface as fl2v, and an fl2v distill does not transfer to it.**
Three facts separate them:

1. **Conditioning geometry.** A first-frame keyframe sits at exactly the target's own first-frame rotary coordinate, on
   the same spatial grid; a reference sits on its own grid at a rotary time the target never visits, and shifts the
   target's origin by however much reference material precedes it. fl2v gets a free zero-offset alignment prior; ref2v
   gets none (Fact A).
2. **Different weights.** `transformer_ref` is a separate partition about 4.2% (relative Frobenius) from fl2va across
   the attention and MLP projections; the whole 8-step turbo LoRA is 0.036%. The target moved about 120 times further
   than the distillation reaches (Fact B).
3. **No ref2v training path in the trainer that made the fl2v turbo LoRAs** (Fact C). This is the reason it had not
   been done; 1 and 2 are why it is hard.

**Dedicated ref2v distills now exist**, which retires Fact C as a universal and leaves 1 and 2 standing as the reason
an fl2v distill cannot transfer:

- lightx2v's ref2v-trained turbo, on disk as `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` (header read
  2026-08-18: PEFT rank 128, alpha 8.0, training_scale 0.0625, base `minimax_h3_fl2va_bf16`, attn+mlp of all 50 blocks
  plus the token refiner). Never tested here; the lightx2v turbo lane closed on 2026-09-26
  (`docs/state/roadmap.md`, "Closed lanes").
- PDD's Ref2VA file, distilled on `transformer_ref` itself, so Fact B does not bite it ([`pdd.md`](pdd.md)).

Written 2026-08-13 against ComfyUI `12666983` (v0.32.0) and the diffusers reference at `6b3740be8`. Some citations
point into `coderef/` (gitignored symlinks to diffusers, LightX2V, Minimax-H3-Turbo and DiffSynth-Studio); every claim
marked measured was reproduced against files that ship with ComfyUI or the model download.

## Fact A: the rotary clock

**Verified** in `comfy/ldm/minimax/model.py`, `PackedLayout.__init__`. The cursor starts at `text_len`. The keyframe
branch never touches it: each keyframe takes a time from the target's own clock and the target's spatial grid,
reused verbatim (`g[:, 1:] = frame`):

```python
if pixel_index == 0:
    cond_t = float(text_len)
elif frame_count is not None and pixel_index == frame_count - 1:
    cond_t = float(text_len) + sum(_video_t_spans(latent_t)) - FRAME_RESCALE
```

The reference branch resets and advances it, and each reference computes its own spatial grid from its own latent
dimensions; the target's segments then take whatever `cursor` holds:

```python
cursor = float(text_len)
...
cursor += 1.0                                        # per image
cursor += float(rt)                                  # per standalone audio
cursor += max(float(rt), sum(_video_t_spans(vt)))    # per video
```

The diffusers pipeline does the same (`modular_pipelines/minimax_h3/before_denoise.py`), and its `references.py` says
the reference order "advances the shared audio/video rotary clock, so a different order is a different request."

**Measured**, driving the real `PackedLayout`:

```
fl2v:  keyframe t = 220.0     target's first latent frame t = 220.0
       keyframe spatial == target frame spatial: True   (torch.equal)

ref2v: reference t = 220.0    target t0 = 221.0
       reference h-range [0.000, 31.000]   target h-range [3.905, 27.087]
       reference w-range [0.000, 31.000]   target w-range [-5.166, 36.158]
```

The target's rotary origin moves 0 in every fl2v case and 1 to 1206 units across realistic ref2v requests, against a
target span of about 207. Both grids are area-normalised into the same nominal 32-unit box, so a reference of the same
aspect lands on the same ranges, but nothing forces references to match the target's aspect. **Inferred:** RoPE encodes
relative position, so ref2v must recover correspondence from content alone, and a few-step student must reproduce that
while collapsing 16 or more evaluations into 4, across a geometry that varies per request.

## Fact B: a different checkpoint, and far away

**Verified:** diffusers' `MiniMaxH3Ref2VALoopDenoiser` sets `transformer_name="transformer_ref"`; LightX2V selects
`"transformer_ref" if task == "ref2av" else "transformer"` and builds a separate pipeline; ComfyUI ships two files,
`minimax_h3_fl2va_pruned_*` and `minimax_h3_ref2va_pruned_*`, whose weights differ everywhere, down to the timestep
embedder (10.6% relative on `e(t)`, measured 2026-08-20 on the unpruned files,
`data/results/2026-08-20_adaln_pruning_residual.json`); the 8-step turbo LoRA's metadata names its base as
`Comfy-Org/MiniMax-H3 minimax_h3_fl2va_bf16.safetensors`.

**Measured**, both `fp8_scaled` checkpoints dequantised by `weight_scale`, mean relative Frobenius delta:

| quantity | mean rel. delta |
|---|---|
| fl2va to ref2va | **0.042** |
| turbo 8-step v1.0 LoRA | **0.00036** |
| turbo 4-step v1.0 768p LoRA | 0.0017 |

- **The tensor key sets are identical** (1082 shared, none unique), so an fl2v LoRA loads onto ref2va with zero
  unmatched keys and nothing errors. The same silent-success class `bench/check_distill_settings.py` was written for,
  and the reason the PDD node fingerprints the partition.
- **The checkpoint delta is a fairly uniform 3 to 5% across all 50 blocks**; RMS norm weights move about 0.5%. (A
  second exception, `final_layer.adaln_proj.linear.weight` at 1.92, "essentially rewritten", was withdrawn
  2026-08-20: on curve-form checkpoints that number compared coefficient matrices on `adaln_t_table` bases whose
  columns 4-7 differ in sign between the parents. At the modulation output the final layer differs about 3% overall and
  about 12% in its time-varying part, the largest adaln item and the same order as the linears;
  `data/results/2026-08-20_dit_internals.json`.)
- **The official turbo LoRAs leave the modulation path alone.** They touch 208 modules, all `attn.qkv_proj`,
  `attn.out_proj`, `mlp.fc1` and `mlp.fc2` in the 50 blocks plus the 2 token-refiner blocks; never `final_layer`,
  per-block `adaln_proj`, the norms, the patch projections or `condition_proj`. That is a point in favour of an fl2v
  turbo degrading on ref2va rather than breaking.

### A third-party LoRA adapts the path the official one avoids

Read from the safetensors headers, 2026-08-13:

| LoRA | modules | what it touches | rank |
|---|---|---|---|
| `minimax_h3_fl2v_turbo_8step_v1.0` | 208 | `qkv_proj`, `out_proj`, `fc1`, `fc2` only | 128 / 384 |
| `minimax_h3_turbo_v4_step600_ema` | **259** | the same 208, **plus 51 `adaln_proj.linear`** | 64, **adaln at 16** |
| `minimax_h3_turbo_4step_ema_ckpt850` | 259 | same shape as v4 | 64 / 16 |

The 51 are the 50 per-block `adaln_proj.linear` and `final_layer.adaln_proj.linear`, at a deliberately lower rank.
On our pruned base that update cannot land as a weight patch; `ComfyUI-MiniMax-H3-Turbo`'s `MiniMaxH3TurboLoRA`
re-injected it every forward from a `silu(t_emb)` grid it ships, and the stock loader applies the backbone, skips the
rest with a log line, and renders wrong. No graph here can run it: the owner removed the pack on 2026-09-23, with its
graphs and the generator's `turbo_pack` path (`docs/state/decisions.md`, 2026-09-23).

So the thesis is narrow: the official turbo LoRAs are fl2v distillations that do not adapt the conditioning-modulation
path, so on ref2va they run a schedule the model never saw with fl2v's modulation. A LoRA that adapts that path is not
covered by any measurement here. Community reports match the split (lightx2v failing to blend references on ref2va
while the v4 family holds adherence), and the header difference is a mechanism that would produce it. Two v4 ref2va
probes and `h3_probe_ref2v_turbo` (an fl2v distill on ref2va) were graphed, never judged, and retired with their lanes.

Reproduce with the checkpoints in `models/diffusion_models/` and the LoRA in `models/loras/`: dequantise each
`*.weight` by its `*.weight_scale` and compare Frobenius norms; the LoRA delta is `(alpha/rank) * B @ A`, metadata
`training_rank: 128, training_alpha: 8.0, training_scale: 0.0625`.

## Fact C: the turbo trainer has no ref2v path

The DMD trainer that produced the lightx2v turbo LoRAs rejects non-text conditioning (its token-tag guard raises), rolls
out from pure noise on a prompt-only dataset, fixes the latent shape from config with no bucketing, and never sets the
packed-sequence builder's reference-row field nonzero; `ref2v` appears nowhere in its tree. Adding it means a
multimodal tokenizer path, reference row packing, a variable-length rollout and a second checkpoint to distill. Their
roadmap lists "Develop distillation based on Ref2V" as future work. DiffSynth's H3 supervised fine-tuning path does
support references, so ref2v fine-tuning existed before ref2v step distillation.

## Hypotheses tested

| hypothesis | verdict |
|---|---|
| conditioning-distribution width | **confirmed, larger than expected**: sequence-length spread 1.11x for fl2v at a fixed canvas, 4.21x for ref2v; condition rows 0 to 2,016 against 1,024 to 105,312, a 103x range |
| rotary-clock coupling | **confirmed**, Fact A |
| reference rows re-injected every step and never denoised | **refuted as a differentiator**: the mechanism is byte-identical for both; what differs is the frozen fraction, about 5% of the sequence in fl2v against up to 69% in ref2v |
| guidance or CFG dependence | **refuted**: the checkpoint is guidance-distilled with no unconditional branch for any task |
| training-data scarcity | **refuted as stated**: DMD here is data-free; what is missing is a realistic conditioning distribution and the code path |
| identity is a high-frequency, low-tolerance target | **confirmed, sharpened**: at 4 steps and shift 12 the final Euler step covers 80% of the trajectory, with no evaluation below sigma 0.5 |

## Running an fl2v turbo on ref2v anyway

The lightx2v turbo lane is closed (2026-09-26), so this is guidance for reopening it. Ranked by expected payoff:

1. **Use the 8-step v1.0, not the 4-step** (the one recommendation backed by measurement). It perturbs the weights 5x
   less (0.00036 against 0.0017), so it degrades toward "the base at 8 steps". Its shift is 12/3, the ref2v default,
   so nothing else moves; the 4-step 768p needs 6/3 and changes schedule and LoRA at once. Its schedule leaves a 0.632
   final jump instead of 0.800. Against it: the 768p was trained at 1344x768, the ref2v node's default canvas, where the
   8-step saw 544p mixed aspect (`coderef/Minimax-H3-Turbo/README.md`, the model table), so at 1:1 or 9:16 the 768p file
   is the one off-distribution. Worth one arm after the 8-step baseline.
2. **Lower the strength.** A public evaluation found 0.75 necessary even in distribution for t2v with v0.1 (4x jitter at
   1.0, 1.7x at 0.75). The delta scales linearly (`strength * (alpha/rank) * B @ A`); sweep 1.0 / 0.85 / 0.7 / 0.5 at a
   fixed seed and expect a monotone trade. **Use 0.01 as the control, not 0.0**: 0.0 short-circuits and skips the
   dequantise/add/requantise round trip (recorded in `workflows/h3_config.py`).
3. **Split the schedule base-last**: the student for the high-noise majority, the base for the last one or two steps,
   because the student's deficit is high-frequency detail, resolved at low sigma, which is what ref2v exists for. Both
   orderings hand a model an input whose sigma label does not match its noise content, unmeasured for H3. In ComfyUI:
   `SplitSigmas`, two `SamplerCustomAdvanced` passes, the LoRA model first and the plain model second, both on the same
   shift.
4. **Do not move the shift.** Shift 6 at 8 steps would cut the final jump to 0.462, but takes the LoRA off the schedule
   it was distilled at (the failure `bench/check_distill_settings.py` catches) and confounds any strength change.

**What failure to look for**, most likely first:

1. **Identity drift, not collapse**: the right category of person in roughly the right clothes, with the specific face,
   hairline, freckles, logo text or fabric weave gone. Compare a still against the reference at 100%.
2. **Reference influence weakening later in the request.** The last reference sits nearest the target positionally; a
   reference behaving differently at slot 1 and slot 9 is the rotary-coupling failure. Re-run with the order reversed.
   fl2v cannot produce this, so it confirms the mechanism.
3. **Temporal crawl on locked-off shots**: public measurements found 8x and 19x normalised jitter for v0.1 at strength
   1.0 on near-static shots.
4. **Audio degrading first.** The turbo releases were all distilled at audio shift 3, so audio is the axis least likely
   to break; if audio degrades and video does not, suspect the graph.
5. **Not expected: garbage or NaN.** The delta is 0.036% of the weight norm and avoids `final_layer`, `adaln_proj`, the
   norms and the patch projections; a breakdown points at wiring.

## What could not be determined

- Whether lightx2v tried ref2v distillation and failed, or has not started; no public postmortem.
- **What the rotary displacement costs.** That it happens, and its size, are measured; its effect on quality is not.
  The reference-reorder control, on the base first, would settle it.
- Whether the weight distance predicts LoRA transfer failure: Frobenius norm is a crude proxy, and the 4% is an average
  over modules that may not be equally load-bearing.
- Whether any published work distills a reference-conditioned video model; none found before PDD's Ref2VA release.

## See also

- `bench/check_distill_settings.py`: each distill LoRA at the shift and steps it was distilled at; it deliberately does
  not police the task type.
- [`../method/checks.md`](../method/checks.md), the check index; [`../state/open_experiments.md`](../state/open_experiments.md),
  what else is unmeasured.
