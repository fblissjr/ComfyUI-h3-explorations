# Why ref2v resists step distillation

> **Overtaken in part, 2026-08-18.** The premise sentence below — no ref2v
> distill exists — is no longer true: lightx2v released a dedicated
> ref2v-trained turbo, and it is on disk here as
> `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` (header read
> 2026-08-18: PEFT rank 128, alpha 8.0, training_scale 0.0625, base
> `minimax_h3_fl2va_bf16`, attn+mlp of all 50 blocks plus the token refiner).
> That empirically retires reason 3 (the trainer had no ref2v path — it does
> now, or was extended). Reasons 1 and 2 stand as written: they explain why an
> *fl2v* turbo cannot transfer to ref2v, which the dedicated checkpoint does
> not contradict. Untested here in any way; owner decision 2026-08-18 is that
> turbo/distilled arms are out of scope for now.

lightx2v has shipped three FL2VA turbo LoRAs for MiniMax H3 and none for
ref2v. Their roadmap lists "Develop distillation based on Ref2V" as future
work. This document is the answer to *why*, worked out from the code rather
than guessed, plus what to expect if you run ref2v with an fl2v distill LoRA
anyway.

Written 2026-08-13 against ComfyUI `12666983` (v0.32.0) and the diffusers
reference at `6b3740be8`.

> Some citations point into `coderef/`, which holds gitignored symlinks to
> reference implementations (diffusers, LightX2V, Minimax-H3-Turbo,
> DiffSynth-Studio). If you cloned this repo you do not have those. Every
> claim marked MEASURED was reproduced against files that ship with ComfyUI
> or with the model download.

---

## Short answer

Ref2v is not a harder version of the same problem. It is a different problem
wearing the same interface. Three things separate it from fl2v, and only the
first is the one people expect.

1. **fl2v conditioning is positionally identical to the target; ref2v
   conditioning is not.** A first-frame keyframe occupies the exact same
   three-dimensional rotary coordinate as the target's own first latent frame,
   on the same spatial grid, bit for bit. A reference occupies its own grid at
   its own scale, at a rotary time the target never visits, and the target's
   own origin then shifts by however much reference material preceded it. So
   fl2v hands the network a free zero-offset alignment prior and ref2v hands
   it none.
2. **ref2v is a different set of weights.** MiniMax ships `transformer_ref` as
   a separate partition. The fl2va-to-ref2va delta is about **4.2 percent**
   relative Frobenius across the attention and MLP projections; the entire
   8-step turbo LoRA is **0.036 percent**. The distillation target moved
   roughly 120 times further than the distillation itself reaches.
3. **The training stack has no ref2v path at all.** The DMD trainer that
   produced these LoRAs is data-free, runs at one fixed latent shape, and has
   an explicit guard rejecting anything but text conditioning. Its
   packed-sequence builder cannot express a reference row.

The third is the reason it has not been done. The first two are the reason it
would be hard even with the code.

---

## Fact A: the rotary clock

**Verified** in `comfy/ldm/minimax/model.py`, `PackedLayout.__init__`.

The cursor is initialised once:

```python
cursor = text_len
row = text_len
```

**The keyframe branch never touches `cursor`.** Each keyframe takes a temporal
coordinate drawn from the target's own clock:

```python
if pixel_index == 0:
    cond_t = float(text_len)
elif frame_count is not None and pixel_index == frame_count - 1:
    cond_t = float(text_len) + sum(_video_t_spans(latent_t)) - FRAME_RESCALE
```

and the target's spatial grid, computed once from the target canvas and reused
verbatim (`g[:, 1:] = frame`).

**The reference branch resets and then advances it**, per block:

```python
cursor = float(text_len)
...
cursor += 1.0                                        # per image
cursor += float(rt)                                  # per standalone audio
cursor += max(float(rt), sum(_video_t_spans(vt)))    # per video
```

and each reference computes its own spatial grid from its own latent
dimensions. The target's audio and video segments then take whatever `cursor`
holds by the time the references are done.

The authoritative diffusers pipeline does the same arithmetic in
`modular_pipelines/minimax_h3/before_denoise.py`, and `references.py` states
the consequence in its own words: the reference order "advances the shared
audio/video rotary clock, so a different order is a different request."

**Measured**, driving the real `PackedLayout`:

```
fl2v:  keyframe t = 220.0     target's first latent frame t = 220.0
       keyframe spatial == target frame spatial: True   (torch.equal)

ref2v: reference t = 220.0    target t0 = 221.0
       reference h-range [0.000, 31.000]   target h-range [3.905, 27.087]
       reference w-range [0.000, 31.000]   target w-range [-5.166, 36.158]
```

Displacement of the target's rotary origin: **0 in every fl2v case, 1 to 1206
units across realistic ref2v requests**, against a target whose entire span is
about 207.

Both grids are area-normalised into the same nominal 32-unit box, so a
reference of the *same* aspect ratio lands on the same ranges. Nothing forces
references to match the target's aspect, so in ref2v the reference-to-target
relative position is a free variable of the request. In fl2v it is pinned at
zero.

**Inferred from that:** RoPE encodes relative position, so fl2v gets a strong
free alignment prior — the conditioning row for pixel frame 0 sits at relative
offset zero from the target row it should reproduce. Ref2v has to recover
correspondence from content alone through full self-attention. A few-step
student must reproduce that mechanism while also collapsing 16 or more
evaluations into 4, across a conditioning geometry that varies per request.

---

## Fact B: it is a different checkpoint, and it is far away

**Verified.** ref2v runs a separate weight partition, not a mode of the fl2v
model:

- diffusers `MiniMaxH3Ref2VALoopDenoiser` sets
  `transformer_name="transformer_ref"`.
- LightX2V selects
  `"transformer_ref" if task == "ref2av" else "transformer"` and builds a
  separate pipeline for it.
- ComfyUI ships two distinct files, `minimax_h3_fl2va_pruned_*` and
  `minimax_h3_ref2va_pruned_*`.
- The 8-step turbo LoRA's own metadata names its base:
  `base_model: Comfy-Org/MiniMax-H3 minimax_h3_fl2va_bf16.safetensors`.

**Measured**, both `fp8_scaled` checkpoints, dequantised by `weight_scale`,
relative Frobenius norm of the difference:

| quantity | mean rel. delta |
|---|---|
| fl2va to ref2va | **0.042** |
| turbo 8-step v1.0 LoRA | **0.00036** |
| turbo 4-step v1.0 768p LoRA | 0.0017 |

Two things follow, and both matter more than the headline ratio:

**The tensor key sets are identical** — 0 keys on either side only, 1082
shared. So an fl2v turbo LoRA loads onto the ref2va checkpoint with zero
unmatched keys and **nothing errors**. This is the same silent-success failure
class `bench/check_distill_settings.py` was written for.

**The LoRA does not touch the modulation path at all.** The delta is a
fairly uniform 3 to 5 percent across all 50 blocks, with one exception: RMS
norm weights move about 0.5 percent. (This paragraph used to name a second
exception, `final_layer.adaln_proj.linear.weight` at **1.92**, "essentially
rewritten". Withdrawn 2026-08-20: these are curve-form checkpoints, and that
number compared coefficient matrices sitting on `adaln_t_table` bases whose
columns 4-7 differ in sign between the parents. At the modulation output the
final layer differs ~3% overall and ~12% in its time-varying part -- the
largest adaln item, and the same order as the linears.
`bench/results/2026-08-20_dit_internals.json`.) The turbo LoRAs touch 208
modules, all of them `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1` and `mlp.fc2`
in the 50 blocks plus the 2 token-refiner blocks. They do not touch
`final_layer`, per-block `adaln_proj`, the norms, the patch projections or
`condition_proj`. **That is a point in favour of the experiment working at
all**, and it is why total breakdown would be surprising.

### A third-party LoRA adapts exactly the path this one avoids

Measured 2026-08-13 by reading the safetensors headers, no GPU:

| LoRA | modules | what it touches | rank |
|---|---|---|---|
| `minimax_h3_fl2v_turbo_8step_v1.0` | 208 | `qkv_proj`, `out_proj`, `fc1`, `fc2` only | 128 / 384 |
| `minimax_h3_turbo_v4_step600_ema` | **259** | the same 208, **plus 51 `adaln_proj.linear`** | 64, **adaln at 16** |
| `minimax_h3_turbo_4step_ema_ckpt850` | 259 | same shape as v4 | 64 / 16 |

The 51 extra modules are the 50 per-block `adaln_proj.linear` **and
`final_layer.adaln_proj.linear`** -- the modulation path, which the official
LoRAs leave alone. (The sentence here used to say these are "where the two
checkpoints differ most"; withdrawn 2026-08-20 with the 1.92 figure above.
They differ there by a few percent, like everywhere else.)

A separate low rank for that path (16, against 64 for attention and MLP) is a
deliberate design decision, not an artifact of how the LoRA was extracted.

**This narrows the thesis of this document.** The claim was never that
distilling ref2va is impossible -- it is that the *official* turbo LoRAs are
fl2v distillations that do not adapt the conditioning-modulation path, so
using one on ref2va asks the model to run a schedule it never saw while its
modulation is still the one fl2v needs. A LoRA that does adapt that path is
not covered by any measurement here. General prompting research reports
exactly this split in practice -- lightx2v failing to blend references on
ref2va while the v4 family holds prompt adherence -- and the header
difference above is a mechanism that would produce that result.

**Untested here.** Nothing in this repo has yet rendered ref2va with v4, and
the weight analysis says only that v4 *touches* the right modules, not that
it touches them *well*. `h3_probe_ref2v_turbo` currently runs ref2va with an
fl2v distill, which is the arm the community reports as the failing one; its
missing twin is the same graph with v4.

Reproduce with the checkpoints in `models/diffusion_models/` and the LoRA in
`models/loras/`: dequantise each `*.weight` by its `*.weight_scale`, then
compare Frobenius norms. The LoRA delta is `(alpha/rank) * B @ A`, and its
metadata records `training_rank: 128, training_alpha: 8.0,
training_scale: 0.0625`.

---

## Fact C: there is no ref2v training path

The DMD trainer that produced these LoRAs rejects non-text conditioning
outright — its token-tag guard raises on anything else. The rollout is pure
noise, the dataset is prompt-only with no pixel values, the latent shape is
fixed from config with no bucketing, and the packed-sequence builder's
reference-row field is never set nonzero by any caller. Searching the training
tree for `ref2v` returns nothing.

Adding ref2v means a multimodal tokenizer path, reference row packing, a
variable-length rollout, and a second checkpoint to distill. That is a
project, not a flag.

Worth noting the asymmetry: DiffSynth's H3 **supervised fine-tuning** path
*does* support references. So the industry has ref2v fine-tuning and lacks
ref2v step-distillation.

---

## Hypotheses that did not survive

Recorded because they are the obvious guesses, and three of them are wrong.

| hypothesis | verdict |
|---|---|
| Conditioning-distribution width | **Confirmed, and larger than expected.** fl2v sequence-length spread is 1.11x at fixed canvas; ref2v is 4.21x. Condition rows run 0 to 2,016 in fl2v against 1,024 to 105,312 in ref2v, a 103x range. |
| Rotary-clock coupling | **Confirmed**, quantified above. |
| Reference rows re-injected every step and never denoised | **Refuted as a differentiator.** The mechanism is byte-identical for both, with the same pinning. What survives is the frozen *fraction*: about 5 percent of the sequence in fl2v against up to 69 percent in ref2v. |
| Guidance or CFG dependence | **Refuted.** No guidance mechanism exists for any task; the checkpoint is guidance-distilled and there is no unconditional branch. |
| Training-data scarcity | **Refuted as stated.** DMD here is genuinely data-free, so "reference pairs are hard to source" does not bite. The sharper version is that what is missing is a realistic *conditioning* distribution, plus the absent code path. |
| Identity is a high-frequency, low-tolerance target | **Confirmed**, and sharpened: at 4 steps and shift 12 the final Euler step covers 80 percent of the trajectory in one jump, with zero evaluations below sigma 0.5. |

---

## If you run it anyway

You should. How it fails is informative, and the setup is one node change.
Recommendations ranked by expected payoff.

### 1. Use the 8-step v1.0, not the 4-step

The only recommendation backed by a direct measurement rather than an
argument.

- It perturbs the weights **5x less** (0.00036 against 0.0017). When the base
  weights underneath are already 4 percent wrong, the smaller perturbation
  degrades toward "base model at 8 steps" rather than toward "a distilled
  trajectory aimed at the wrong model".
- Its shift is 12/3, the same as the ref2v default, so **nothing else in the
  graph moves**. The 4-step 768p needs 6/3, which changes the schedule and the
  LoRA at once and makes the result uninterpretable.
- Its schedule leaves a 0.632 final jump instead of 0.800 — 21 percent less of
  the trajectory extrapolated, which matters most for exactly the
  high-frequency identity content ref2v exists for.
- The vendor lists it as usable at 8 or 4 steps, so it gives you a 4-step arm
  without swapping checkpoints.

Against it: the 4-step 768p was trained at 1344x768, which is the ref2v node's
default canvas, where the 8-step was trained at 544p mixed aspect. A real
argument the other way, and worth one arm — after the 8-step baseline exists.

### 2. Lower the LoRA strength

Cheap, continuous, and already validated in distribution.

A public H3 turbo evaluation found strength 0.75 necessary even for
*in-distribution* t2v with v0.1 — 4x jitter at 1.0 dropping to 1.7x at 0.75.
If 0.75 was needed on the model the LoRA was actually trained for, going out of
distribution is not the moment to run 1.0.

ComfyUI applies the delta as `strength * (alpha/rank) * B @ A`, so strength
scales the perturbation linearly with no threshold effect. Sweep 1.0 / 0.85 /
0.7 / 0.5 at fixed seed and expect a monotone trade: identity fidelity and
texture improve as strength drops, step-efficiency degrades, and somewhere
below about 0.5 you are paying 8 steps for a model that needs 16.

**Use 0.01 as the control, not 0.0.** Strength 0.0 short-circuits and skips the
dequantise/add/requantise round trip entirely, so it is not a like-for-like
baseline. This is recorded at `workflows/h3_config.py`.

### 3. The two-stage split, but base-LAST

Medium payoff, real implementation cost, and **the intuitive ordering is the
wrong one here**.

The natural instinct, and the one that works on other models, is base on the
high-noise steps and the distilled student on the finish. For ref2v that is
backwards. The student's measured deficit is high-frequency detail, and
high-frequency detail is resolved at low sigma. Ref2v's whole purpose is
high-frequency identity. So the intuitive ordering puts the student's known
weakness exactly where ref2v's demand is highest.

Base-last — the distilled student for the high-noise majority, the un-LoRA'd
base for the last one or two steps — spends the base model's cost where it
buys most and keeps the speedup where the student is strong.

Two honest caveats. **Both orderings have a handoff mismatch**: a distilled
student's state after step 1 is not on the base model's trajectory, so the base
receives an input whose sigma label does not match its actual noise content,
and the reverse direction has the same problem mirrored. This has not been
measured for H3. And the student *is* trained to be robust to its own
intermediate states — the DMD trainer samples the rollout end point uniformly
and applies its loss there — but that says nothing about robustness to another
model's intermediate states.

In ComfyUI: `SplitSigmas` on the `BasicScheduler` output, two
`SamplerCustomAdvanced` passes, LoRA-loaded model on the first and plain model
on the second. **Both passes must carry the same `MiniMaxH3SigmaShift`**, or
the two halves sample different curves.

### 4. Move the shift — last, and probably not at all

There is a principled argument for lowering it: shift 6 at 8 steps puts one
evaluation at sigma 0.462 and cuts the final jump from 0.632 to 0.462, which is
real for detail. The argument against is stronger.

The 8-step v1.0 was distilled at shift 12. Changing it takes the LoRA off its
own schedule, which is the precise silent failure
`bench/check_distill_settings.py` exists to catch. It also confounds: move
shift and strength together and you cannot attribute the result.

If you want a shift-6 arm, the honest way to get it is the 4-step v1.0 768p
LoRA, which was *distilled* at 6/3 and at 1344x768, matching the ref2v canvas.
That is one coherent configuration rather than a mismatched one.

---

## What failure to look for

In order of likelihood.

1. **Identity drift, not collapse.** The subject stays recognisably the right
   category of person or object in roughly the right clothes. What goes is the
   specific face: bone structure, hairline, freckles, logo text, fabric weave.
   Compare a still against the reference at 100 percent, on the face and on any
   printed text.
2. **Reference influence weakening as it moves later in the request.** The
   target's rotary origin is pushed by everything before it, so the last
   reference sits positionally nearest the target. If the same reference
   behaves differently at slot 1 and slot 9, that is the rotary-coupling
   failure, and it is **diagnostic** — re-run with the reference order reversed
   and compare. fl2v cannot produce this signature, so seeing it confirms the
   mechanism rather than just the symptom.
3. **Temporal crawl on locked-off shots.** Public measurements found 8x and 19x
   normalised jitter for v0.1 at strength 1.0 on shots where the baseline
   barely moves. A static-camera scene is the most sensitive detector of
   over-strength.
4. **Soundtrack degradation before video degradation.** The audio stream runs
   its own shift of 3.0, and all three turbo releases were distilled at audio
   shift 3, so this is the axis *least* likely to break — which makes it a
   useful control. If audio degrades and video does not, suspect the graph
   rather than the hypothesis.
5. **Not expected: garbage or NaN.** The LoRA delta is 0.036 percent of the
   weight norm and avoids `final_layer`, `adaln_proj`, the norms and the patch
   projections, which is where the checkpoints differ most. Total breakdown
   would point at a wiring error, not at distribution shift.

---

## What could not be determined

- **Whether lightx2v tried ref2v distillation and failed, or simply has not
  started.** The roadmap line says only "Develop distillation based on Ref2V".
  The absence of ref2v code in the training tree is consistent with either. No
  public postmortem found.
- **How much the rotary displacement actually costs.** That it happens, and its
  magnitude, are measured. Its effect on output quality is not, and guessing a
  number would be worse than saying so. The experiment that would settle it is
  the reference-reorder control above, run on the base model first to establish
  that reordering matters at all, then on the distilled one.
- **Whether the weight distance predicts LoRA transfer failure.** Frobenius
  norm is a crude proxy. A small low-rank delta aligned with functionally
  important directions can matter far more than its norm suggests, and the 4
  percent figure is an average over modules that may not be equally load
  bearing.
- **Whether any published work distills a reference-conditioned video model.**
  None found.

---

## See also

- `bench/check_distill_settings.py` — enforces that each turbo LoRA is loaded
  at the shift and steps it was distilled at, and deliberately does *not*
  police which task type it is loaded into, so the experiment above is not
  blocked.
- `docs/checks.md` — the check index.
- `docs/open_experiments.md` — what else is unmeasured, and why.
