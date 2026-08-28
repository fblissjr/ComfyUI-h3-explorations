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

### The transform is AFFINE, which sharpens the claim and then weakens it

Raised by a peer session 2026-08-28 and confirmed. The transform is affine in
`out[1]`:

    out[1] = A + B * out[1]        A = (1-s) * audio_src * carry
                                   B = 1 + (s-1) * sigma_a

**Affine maps commute with averaging.** So a fused head returning a block MEAN
is harmless in principle — that was not the right framing, and the first version
of this section had it. The error is entirely that the code **freezes A and B at
the block's starting sigma** while they vary across the block.

That is pure schedule arithmetic and needs no model, so it can be computed.
Relative variation of B across a block, `(Bmax-Bmin)/mean(B)`, at shift_v 12:

| audio_shift | s | w=4 | w=8 | w=28 |
|---|---|---|---|---|
| 12 | 1.00 | 0 | 0 | 0 |
| 6 | 2.00 | 0.384 | 0.522 | 0.316 |
| **3 (ships)** | **4.00** | **0.663** | **0.980** | **0.778** |
| 1 | 12.00 | 0.918 | 1.518 | 1.991 |

**It is not monotone in width**, because a 28-wide block sits mostly in the flat
early region. Where a block sits matters as much as how wide it is.

### The right quantity is INTEGRATED drift, and it orders all four arms

Max-of-relative-variation was the wrong statistic and mis-ranked `opt4`. The
mechanism implies something else: freezing `B` costs error at EVERY step in the
block, proportional to how far `B` has drifted from its frozen value. So the
quantity is the **integrated absolute deviation** — over each block, the sum of
`|B(sigma) - B(block start)|` across the grid points it spans.

| arm | partition | integrated drift | observed audio/video |
|---|---|---|---|
| u8 | [4]x8 | 7.19 | 1.58 |
| mix6 | [4,4,4,4,8,8] | 10.88 | 1.68 |
| u4 | [8]x4 | 12.22 | 1.72 |
| opt4 | [28,2,1,1] | 23.19 | 1.90 |

    by integrated:  u8 < mix6 < u4 < opt4
    by observed:    u8 < mix6 < u4 < opt4      MATCH, all four

**`opt4` no longer needs the off-distribution story.** It is worst because a
28-wide block accumulates drift at 28 grid points, which a max cannot see and an
integral does. The post-hoc note below is kept because this does not rule the
start-at-30 explanation out, but it is no longer needed to save the mechanism.

**The caveat is not small, and it belongs next to the result.** Four data points,
and several metrics were tried before this one — a metric that fits four points
was cheap to find. What makes integrated drift worth more than a fit is that it
is **derived from the mechanism** (the error accumulates per step by
construction) rather than selected for agreement. **Call it a consistency check,
not a confirmation.**

### Two arithmetic corrections made along the way

**The earlier disagreement is NOT settled, and should not be written up as
normalisation.** Two sessions got 0.474 (by `B[block start]`) and 0.980 (by
`B[mean]`) for the same w=8 quantity. A third computation of the mean
normalisation gives 0.498, BELOW the by-start figure rather than above it, so
"three normalisations spanning 3x" does not account for it. Since integrated
drift reproduces exactly across both sessions, the `B` grids agree and the gap
is specific to that one metric's computation. **It no longer bears on any
conclusion — max-of-relative was withdrawn — but it is an open discrepancy, not
a resolved one.**

**`mix6` and `tail6` are different arms and were briefly conflated.** `mix6` is
`[4,4,4,4,8,8]`, wide blocks at the END, integrated 10.88. `tail6` is
`[8,8,4,4,4,4]`, wide blocks at the FRONT, integrated 8.54. The claim that
`mix6` and `u4` have identical variation came from reading `mix6` as `tail6`;
withdrawn.

### First perceptual result: tail6 fixes what u4 broke, at matched canvas

**Owner's read, 2026-08-28, unprompted:** `..._00007` (tail6) "looks and sounds
about as good as I'd expect for a 4 step"; `..._00004` (uniform 4) is "jaggedy
lines and scratchy audio".

> **THE FILENAMES LIE, and this one will cost somebody an hour.** Both files are
> called `text_to_video_pdd_4step_0000N` because they were produced by mutating
> the 4-step graph with `ManualSigmas`, and the muxer names output after the
> graph. **`00007` ran SIX evaluations, not four.** Only `00004` is actually a
> 4-evaluation render. Read the partition out of the embedded workflow
> (`SamplerCustomAdvanced.sigmas` -> `ManualSigmas`), never off the filename.

So the owner's phrasing understates it: six evaluations delivering what he would
accept from four, against a four-evaluation arm that is unusable. And "scratchy"
is the useful word — a high-frequency audio artifact, which is the shape a
drifting coefficient would produce rather than a gross timing or content error.

**It is a matched pair.** Diffing the workflows embedded in both PNGs, the only
differences are the sigma source (node SIGMAS -> ManualSigmas), the now-inert
`steps` value, and the ManualSigmas node itself. Same canvas 1344x768, same seed,
same prompt, same LoRA, same Sol settings.

| | 00004 | 00007 |
|---|---|---|
| partition | [8,8,8,8] | [8,8,4,4,4,4] |
| evaluations | 4 | 6 |
| final Euler step | **80.0%** | **63.2%** |
| integrated drift | 12.22 | **8.54** |

**Both predictions point the same way and both are satisfied**: the video
argument wanted a narrower final block, the audio mechanism wanted lower
integrated drift, and tail6 was built for the FIRST reason before the second
existed.

**What it is worth.** One seed, and two arms differing in a knob are different
samples — `CLAUDE.md`'s rule applies. What raises it above that: the complaint
was specific, it predated tail6, tail6 was designed for an unrelated reason, and
the improvement was volunteered rather than asked for. That is the
resolved-complaint shape, which is the strongest form a single render takes and
is still not a measurement.

**It does NOT score the pre-registered test below**, which needs 1152x768 to
compare against `mix6`. This is the perceptual half; that one is the numeric
half, and they are different claims.

### The practical conclusion: 4 evaluations is a bad operating point, and cannot be fixed

The owner's read of the pair was not marginal — `00004` "looks really bad",
`00007` looks and sounds decent. Same canvas, same seed, same everything but the
partition.

**Four evaluations has no better configuration.** Enumerated: the only partition
of the 32-point grid into 4 blocks with starts on multiples of `L_min` and widths
within `L_max` is `[8,8,8,8]`. So its 80% final Euler step and its 12.22
integrated drift are forced, not chosen. **There is nothing to tune.**

**Five and six evaluations are a different operating point, not a small
improvement:**

| arm | evals | final step | integrated drift |
|---|---|---|---|
| uniform 4 (the shipped `_4step` arm) | 4 | **80.0%** | 12.22 |
| tail5 `[8,8,8,4,4]` | 5 | 63.2% | 9.82 |
| tail6 `[8,8,4,4,4,4]` | 6 | 63.2% | 8.54 |
| uniform 8 (the vendor's count) | 8 | 63.2% | 7.19 |

**tail5 buys the 8-step's final step at five evaluations.** The knee is at 5-6
with a tail-weighted partition, not at 4 uniform — and the shipped `_4step`
graph sits on the wrong side of it.

**What shipping that would need.** `resolve_emit_steps` refuses counts that do
not divide 32, so a 5- or 6-evaluation arm needs `ManualSigmas` today. The
follow-on named in [`2026-08-28_handoff.md`](2026-08-28_handoff.md) §2.2 —
accept an explicit non-uniform knot list rather than only divisors — was
deliberately deferred there as "do not build this first". **It now has a reason
behind it**: not a convenience, but the only way to ship the operating point the
arithmetic and one perceptual pair both point at.

**Still one seed**, and the recommendation rests as much on the forced-partition
arithmetic as on the render. Before changing a shipped graph, score the
pre-registered test below and get a second seed.

### PRE-REGISTERED: tail6 against mix6, and why tail5 was not a test

**`tail5` versus `u4` does not discriminate, and was withdrawn.** tail5 has MORE
evaluations (5 vs 4) AND lower drift (9.82 vs 12.22), so integrated drift and
plain evaluation count both predict tail5 wins. Two outcomes, one outcome.

**`tail6` against `mix6` is the test**, and it is sharper than anything else
proposed here:

| | tail6 | mix6 |
|---|---|---|
| partition | [8,8,4,4,4,4] | [4,4,4,4,8,8] |
| evaluations | 6 | 6 |
| width multiset | {8,8,4,4,4,4} | {8,8,4,4,4,4} |
| widest block | 8 | 8 |
| **wide blocks sit** | **at the FRONT** | **at the END** |
| **integrated drift** | **8.54** | **10.88** |

**Evaluation count, widest block and width distribution all predict these two
are IDENTICAL.** Placement is the only difference, and integrated drift is the
only account that reads placement. `mix6` is already scored at 1.68, so this
costs ONE arm.

**Prediction: `tail6` lands clearly below 1.68, near `u8`'s 1.58. At or above
`mix6` kills integrated drift** — report it killed rather than refitted, because
placement is the only thing that differs and placement is the only thing the
quantity reads.

> **THE ARMS RENDERED SO FAR CANNOT BE USED FOR THIS.** `tail6` was queued
> through a scratch runner that posted the shipped graph, which is **1344x768**.
> Every scored arm above is **1152x768**, the `fast` tier that
> `grade_pdd_partitions.py` sets in its own `CANVAS` constant. Scoring the
> existing tail6 render against 1.68 reads a canvas change as a placement
> effect. Re-render through the grader:
> `bench/grade_pdd_partitions.py ref32 u8 mix6 tail6`, one batch, one canvas,
> ref32 alongside so everything is graded against the same reference.

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
