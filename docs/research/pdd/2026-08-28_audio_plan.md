# Audio under PDD: the plan, for the session that picks this up

Written 2026-08-28 for a fresh session. Self-contained on purpose — you should
not need the conversation this came out of.

[`audio_under_pdd.md`](audio_under_pdd.md) is the finding and the reasoning.
**This file is the execution plan**: what is done, what is queued, what to run,
and what each outcome would mean. Read that one first; it is short.

---

## The claim you are testing

ComfyUI carries H3's audio latent on the **video** schedule and converts
velocities with a change of variable on an **instantaneous** derivative
(`comfy/ldm/minimax/model.py:530-551`). A PDD fused head returns the block's
**mean** velocity. An instantaneous transform applied to a block-averaged
quantity is exact only as the block narrows. Video has no such transform.

**So: an audio-only error that grows with block width.** The vendor does not
have this problem because it steps audio on its own scheduler
(`coderef/alibaba-pai_MiniMax-H3-Acc-LoRAs/predict_t2v.py:36`).

This is **reasoning from source plus a trend**, not a demonstrated cause. Your
job is to move it or kill it.

---

## State when this was written

**Done.**

- `bench/measure_pdd_audio_carry.py` — the transform's geometry against block
  width. Monotone across uniform widths, 0.1425 at L=1 to 1.5408 at L=16, about
  10.8x. Consistent. It measures the transform, NOT a render error.
- `bench/results/2026-08-28_pdd_partition_fidelity_362.json` — four arms at 362
  frames. Video flat across a 7x range of block width, audio monotone, ratio
  1.58 to 1.90.

**Queued and unfinished.** `tail6` and `tail5` on the market t2v arm, posted by
a scratch runner rather than a committed script. If they are gone, requeue: both
are in `bench/grade_pdd_partitions.py`'s `MANUAL` and `ARMS`.

**Parked.** `internal/parked/2026-08-28_head_blocks_dynamiccombo.patch`, with
its own README. Unrelated to audio; land it in one commit after a restart.

---

## The experiments, in order

### A. Grade tail6 / tail5 — free if they rendered

They exist for an unrelated reason (a VIDEO argument about the final Euler
step), which makes them a **cleaner test than one designed to confirm this**.
Their final block is width 4, the same as `u8`; `u4`'s is 8.

    <comfy-venv>/bin/python bench/grade_pdd_partitions.py ref32 u8 u4 tail5 tail6

(Use ComfyUI's own virtualenv, not a repo-local one: a bare `uv run`
resolves an environment without `packaging` and these scripts import
torch and comfy.)

**Predict before you look.** Audio for tail5/tail6 near `u8`'s 0.858 rather than
`u4`'s 0.923, with video flat alongside everything else near 0.52-0.54.

- Audio tracks the FINAL block width, not the evaluation count -> supports.
- Audio tracks the evaluation count instead (tail6 at 6 evals landing near u4)
  -> the block-width story is wrong, and it is probably just "fewer steps,
  worse audio".
- Video moves too -> the audio-only half is wrong; look for a shared cause.

### B. The identity-transform diagnostic — one render, the direct test

Set `shift_audio` equal to `shift_video` on `MiniMaxH3SigmaShift`. Then
`s = shift_v/shift_a = 1` and the transform collapses to `out[1] = out[1]`.

**Read this before queueing it:**

- `MiniMaxH3PDDLoRA` has a run-time shift guard (`8e82c51`) that refuses a
  render whose shift is not the file's. This arm WILL trip it. Satisfy it
  deliberately or bypass it deliberately; do not delete it.
- **This is a diagnostic, not a shipping option.** It changes the schedule the
  model is asked to sample. The audio will be wrong in a NEW way. Do not judge
  it by ear and do not compare it to a normal render.
- What matters is only whether the **block-width trend flattens** — run it at
  two widths (4 and 8 evaluations) and compare the audio gap between them, not
  the absolute numbers.

### C. Audio-only refinement — the practical fix, not a test

`coderef/ComfyUI-H3-AudioRefine` freezes video with a per-stream `denoise_mask`,
runs audio-only steps, and takes the MODEL from **before** the LoRA so those
steps are undistilled. Its three nodes are registered on this box.

If the mechanism is right, this works for a reason its authors did not state:
undistilled steps return **instantaneous** velocities, so the change of variable
is exact for them.

Composition is unverified — its `set_model_patch_replace("dit", "double_block", i)`
plus a `WrappersMP.DIFFUSION_MODEL` wrapper against our object patches on
`diffusion_model.forward` and `final_layer.*`. Different surfaces, which raises
the prior and is not evidence. Enumerated in
[`2026-08-28_handoff.md`](2026-08-28_handoff.md) §2.3.

---

## Traps, all of which cost time today

- **Do not compare renders that differ in more than one thing.** Four things
  changed between the last good 4-step render and the bad one; three were inert
  and the one that mattered took a per-row check to find.
- **A rendered clip cannot A/B a numerical change.** `CLAUDE.md` is right and it
  bit twice. Arms that differ in a knob are different SAMPLES. Use
  `grade_pdd_partitions.py` against the 32-step reference, which is controlled
  by construction, or judge blind in aggregate.
- **Audio rel L2 is raw-waveform and phase-sensitive.** It is a trend
  instrument, not a quality measure. Never quote a magnitude from it as
  perceptual.
- **Read the producer before comparing two records.** A file named
  `..._control.json` carries `tau: 1.3` in its metadata and an EMPTY `rows`
  list, because its `controls` are the dense and sparse LIMITS. Reading the
  field name gives a comparison that does not exist.
- **Check which stage a term belongs to.** `quant_l2` in
  `analyze_sol_error.py` is Sol's CUDA kernel INT8, not checkpoint or encoder
  weight quantisation — the weights are already applied when q, k, v exist.
  A claim was written and withdrawn on exactly this.
- **The card is shared.** Other sessions queue renders. Check
  `/queue` before restarting ComfyUI, and restart by PORT OWNER
  (`ss -lptn 'sport = :8188'`), because `pkill -f` patterns miss the real
  process and leave a stale server serving an old schema.
- **The tree is shared.** Stage by explicit path and commit promptly. A node
  schema change and its generator emission cannot be staged separately; someone
  else's `git add` will land half of it.

---

## What would make this worth writing up

**A confirmed mechanism**, meaning B flattens the trend, or A shows audio
tracking final block width in an arm that was queued for another reason.

**Or a clean refutation**, which is just as good and cheaper to reach: audio
tracking evaluation count rather than block width, or video moving with audio.
The mechanism is currently an inference with one supporting trend and one
supporting simulation, and neither can distinguish it from "fewer steps, worse
audio" on its own.

**What it would change if confirmed:** prefer partitions with narrow FINAL
blocks — which is the same conclusion `tail5`/`tail6` reached from the video
side for a completely different reason — and treat audio-only refinement as a
correction with a known mechanism rather than a heuristic.
