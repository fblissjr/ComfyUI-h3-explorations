# Why PDD costs audio more than video, and what to try tonight

Written 2026-08-28, prompted by the owner's question: audio is always the thing
that is off in these 4-step and 8-step distilled arms, and is that a latent
ComfyUI bug?

**No bug was found.** What was found is a structural interaction between how
ComfyUI samples H3's audio and what a PDD head returns. It is audio-only by
construction, it should worsen as blocks widen, and the existing data already
scores that prediction.

[`../../h3_pdd.md`](../../h3_pdd.md) owns the PDD contract and carries the
short form. This file is the working detail and the experiment list.

---

## 1. The mechanism

### Confirmed from source

**The vendor steps audio on its own scheduler.**
`coderef/alibaba-pai_MiniMax-H3-Acc-LoRAs/predict_t2v.py:36` passes
`pipeline.scheduler.shift` AND `pipeline.audio_scheduler.shift` — two
schedulers, audio integrated in its own time at shift 3.

**ComfyUI does not.** `comfy/model_sampling.py:328` carries the audio latent
scaled onto the video schedule so the pack is "an ordinary single-schedule flow
latent", and `comfy/ldm/minimax/model.py:530-551` undoes and redoes that carry
around every forward:

    carry  = sigma_a / sigma_v                        # before _forward
    x      = [x[0], audio_src * carry]                # network sees clean audio
    ...
    out[1] = (1 - s) * (audio_src * carry)            # after _forward
             + (1 + (s - 1) * sigma_a) * out[1]       # s = shift_v/shift_a = 4

That last expression is a **change of variable on an instantaneous derivative**,
evaluated at this step's `sigma_a`.

**PDD's head sits inside that.** `MiniMaxH3PDDLoRA` patches
`final_layer.{video,audio}_out.forward`, which is inside `_forward`, so the
fused head runs in the clean-audio domain and the transform is applied to its
output. **That placement is correct** — this is not the bug.

### The inference

A fused head does not return an instantaneous velocity. It returns the block's
**mean velocity** over `[t_n, t_{n+L}]` — that is the whole idea of PDD. An
instantaneous change of variable, evaluated at the block's starting sigma,
applied to a block-averaged quantity, is exact only as the block narrows.

**Video has no such transform**, because video is the reference stream the carry
is defined against. So the error is audio-only and should grow with block width.

**This is reasoning from source, not a measurement.** It is labelled that way
everywhere it appears.

---

## 2. The evidence, such as it is

`bench/results/2026-08-28_pdd_partition_fidelity_362.json`, 362 frames,
1152x768, one seed:

| arm | widest block | video rel L2 | audio rel L2 | audio/video |
|---|---|---|---|---|
| u8 | 4 | 0.542 | 0.858 | 1.58 |
| mix6 | 8 | 0.536 | 0.899 | 1.68 |
| u4 | 8 | 0.537 | 0.923 | 1.72 |
| opt4 | 28 | 0.522 | 0.992 | 1.90 |

**Video is flat across a 7x range of block width and slightly improves. Audio
rises monotonically, and so does the ratio.**

**What carries the argument is the ordering, not the magnitudes.** Audio rel L2
is raw-waveform and phase-sensitive, which is a poor absolute metric — but phase
noise does not produce a monotone trend in block width with video flat.

**What would break it:** any arm where audio error does NOT track the widest
block, or where video starts tracking it too.

---

## 3. Tonight, ranked by evidence per unit of cost

### 3.1 The block-width simulation — free, no GPU, do this first

The mechanism predicts a specific error, and it can be computed directly rather
than rendered. Take the transform

    f(v, sigma_a) = (1 - s) * x_a + (1 + (s - 1) * sigma_a) * v

and compare, over a block `[t_n, t_{n+L}]`:

  * `f` applied to the block-MEAN velocity at the block's starting sigma — what
    the PDD path actually does; against
  * the mean over the block of `f` applied to the INSTANTANEOUS velocity at each
    sigma — what an exact treatment would give.

**RUN 2026-08-28: `bench/measure_pdd_audio_carry.py`. Consistent, and it
corrected its own premise on the way.**

| L | blocks | max abs(applied - exact) | rel to spread of f |
|---|---|---|---|
| 1 | 32 | 0.1425 | 0.538 |
| 2 | 16 | 0.2831 | 0.566 |
| 4 | 8 | 0.5435 | 0.604 |
| 8 | 4 | 0.9693 | 0.646 |
| 16 | 2 | 1.5408 | 0.685 |
| 28 | 2 | 1.3018 | 0.620 |

Monotone across the uniform widths, **10.8x from L=1 to L=16**.

**The premise it corrected.** This section originally predicted the gap would be
ZERO at L=1, on the reasoning that the block is one interval so the mean is the
instantaneous value. The script asserted that, got 0.1425, and refused to
report. The premise was wrong: L=1 is one grid INTERVAL, not a point, and
`sigma_a` varies across it — so the transform at the interval's start already
differs from the transform averaged over it. **L=1 is a floor, carrying whatever
the sampler's own discretisation costs, not a zero.** The refutable claim is
growth above that floor, and that is what the script now tests.

The `[28,4]` row sits at 1.30, BELOW L=16, and is not a failure: a non-uniform
partition places its widest block somewhere specific on the curve, so its gap
depends on where the block sits as well as how wide it is. Monotonicity is
checked across uniform widths only, and the script says so.

**What it does not claim.** It measures the GEOMETRY of the transform — how much
`f` varies across a block, which is exactly what a block-mean discards — using a
unit velocity field. It is **not** a predicted render error, and a genuinely
constant velocity field would give zero regardless. That assumption is in the
docstring rather than buried.

### 3.2 tail6 / tail5 — already queued, just needs grading

Both are in the queue now. Their final block is width **4**, the same as `u8`,
where `u4`'s is 8.

**Prediction: their audio lands near u8's 0.858 rather than u4's 0.923, while
their video stays flat with everything else.** That is the mechanism's
prediction on arms that were queued for an unrelated reason, which makes it a
cleaner test than one designed to confirm it.

Grade with `bench/grade_pdd_partitions.py` — both are already in its `MANUAL`
and `ARMS` tables.

### 3.3 The identity-transform diagnostic — one render

Set `shift_audio` equal to `shift_video` on `MiniMaxH3SigmaShift`. Then
`s = shift_v/shift_a = 1`, and the transform above collapses to
`out[1] = out[1]` — the identity. If the audio penalty largely disappears, the
mechanism is confirmed directly.

**This is a diagnostic, not a shipping option.** It changes the audio schedule
the model is asked to sample, so the output is not a better render, it is a
different question. Expect the audio to be wrong in a NEW way; what matters is
whether the block-width TREND flattens.

Note the shift guard: `MiniMaxH3PDDLoRA` refuses a render whose shift is not the
file's, so this arm will need that guard satisfied or explicitly bypassed —
check before queueing, it is `8e82c51`.

### 3.4 Audio-only refinement on the base model — the practical fix, bigger build

`coderef/ComfyUI-H3-AudioRefine` already does this for turbo: freeze video with
a per-stream `denoise_mask`, run audio-only steps, and take the MODEL from
BEFORE the LoRA so those steps run undistilled. Its README:7 is the same
complaint from the other direction — "4-step video is acceptable but 4-step
audio is not".

If the mechanism here is right, this fix works for a reason its authors did not
state: the refinement steps are undistilled, so they return instantaneous
velocities, and the change of variable is exact for them.

Its three nodes are registered on this box. The composition question — its
`set_model_patch_replace("dit", "double_block", i)` plus a
`WrappersMP.DIFFUSION_MODEL` wrapper against our object patches — is enumerated
in [`2026-08-28_handoff.md`](2026-08-28_handoff.md) and has not been run.

---

## 4. What NOT to conclude

- **Not "ComfyUI is broken".** Carrying audio on one schedule is a deliberate
  design that makes the pack a single-schedule latent, and it is presumably fine
  for undistilled sampling, where every step does return an instantaneous
  velocity. The interaction is with PDD specifically.
- **Not "audio is 2x worse".** That number is raw-waveform rel L2 against a
  distilled trajectory. It is phase-sensitive, it is one seed, and it is not
  perceptual. The owner's ears and `AudioRefine`'s README are the perceptual
  evidence; this metric is the trend evidence.
- **Not that this explains the 4-step video problem.** That is separate and
  already answered: four evaluations has exactly ONE legal partition, `[8,8,8,8]`,
  so its 80% final Euler step is forced rather than chosen.
