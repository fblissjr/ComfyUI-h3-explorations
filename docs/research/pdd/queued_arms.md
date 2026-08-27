# Queued arms, 2026-08-27

Everything here is blocked on one thing only: the card was busy with another
session's `bench/measure_marker_epsilon.py` run when these came up. None is
blocked on design, tooling or a decision.

**This is a session queue, not a roadmap.** [`docs/roadmap.md`](../../roadmap.md)
owns what we are trying to find out next and
[`docs/open_experiments.md`](../../open_experiments.md) owns what is
deliberately not measured and why. Nothing below is deliberately unmeasured --
it is measurable today, on tooling that exists, as soon as the GPU is free.
Delete an entry when it runs; if one is still here in a week it belongs in one
of those two documents instead.

Each entry names what it decides, because an arm that cannot change a decision
is not worth the card time.

---

## 1. `reuse_qkv_memory`, identity

**Decides:** whether we can turn it on. Nothing else.

One render, because the control already exists.
`Video/scratch/ctl_current_00001.mp4` is the 4-step dialogue arm at
`reuse_qkv_memory=False`; render the identical payload with it `True` and
compare **decoded frames**, not containers:

```
ffmpeg -v error -i A.mp4 -f rawvideo -pix_fmt rgb24 - | md5sum
```

Identical means the flag cannot change output and turning it on is free.
Different means `h3_config`'s "verified numerically identical to the normal
entry" does not hold end to end, and the flag becomes a numerical knob needing
`docs/eval_comparison.md` section 3 rather than a free win.

**Do not try to measure what it saves.** `bench/bench_e2e_h3.py` already spent
2026-08-14 on that and records why it failed: the sampled peak resolved the
resident-weight plateau rather than the attention transient the flag targets,
and every arm came back identical to the megabyte. A flag that moves only a
brief transient needs an instrument neither peak in that bench provides. The
VRAM benefit stays upstream-reported until someone builds one.

**One thing to watch:** sage takes ownership of the float q/k/v list, and this
flag writes the attention output into that same buffer. If those interact, an
end-to-end identity test is what surfaces it -- which is another reason to run
this before trusting the flag rather than after.

## 2. `start_percent`, speed only

**Decides:** whether the quality question is worth asking at all.

`start_percent` has never been measured at any value, ever -- `docs/SOLATTN.md`
says so in its knob table and again in its open-experiments table. It forces the
top of the trajectory dense at a flat 25% of evaluations at every step count: 4
of 16, 2 of 8, 1 of 4. At 0.0 the 4-step arm goes from 2 sparse steps to 3.

Matched pair at 0.2 against 0.0, timing only, on a 4-step arm. If the saving is
small the quality question never needs asking, which is the whole point of
running the cheap half first.

If it is large, the quality half is a numerical knob and needs the blind
multi-seed process. Both directions are arguable and neither has evidence: the
first step's input is pure noise, which argues it is the most redundant place to
route sparsely, and it also sets global composition, which argues a routing
error there propagates into everything after.

## 3. The dialogue scene at 8 NFE, and head-free at 4

**Decides:** whether 4 evaluations is simply too few for this scene, or whether
the head machinery is what is costing quality.

The open question from this session. The derived-`nfe` rewrite is proven to have
changed nothing -- pre-change and post-change renders of this arm are
bit-identical on frames and audio -- so whatever is or is not wrong with the
4-step dialogue clip was there before and is not a regression.

Two arms, each one widget from the shipped graph:

| arm | change | reads as |
|---|---|---|
| 8 NFE | `BasicScheduler.steps` 4 -> 8 | is the step count responsible |
| head-free at 4 | `patch_heads` off | is the head machinery responsible |

`thirty_two_intervals.html` is why 8 is the interesting comparison: the final
evaluation is 63% of the sigma path at 8 NFE against 80% at 4, and Sol covers 5
of 8 steps against 2 of 4. Both accelerations are doing more work at 4.

Remember what a pair can and cannot say. These answer "did each arm meet the
brief", not "which is better" -- two arms differing in a numerical knob are
different samples, not degraded versions of one.

## 4. The PDD blind session

**Decides:** the standing open item -- whether PDD is actually good.

Everything built so far establishes correctness, and nobody has judged quality.
`docs/h3_pdd.md` records the shape: PDD is pitched against the **turbo distill**,
not against base, and `h3_image_ref_plus_text_to_video_turbo_4step.json` is the
matched control with the LoRA node as the only difference. That comparison is
the owner's to run through `docs/eval_comparison.md` section 3.

Stated confound, unchanged: the turbo was distilled at 544p mixed aspect and the
paired arm renders 1344x768, so it is outside its training canvas. PDD's own
training canvas is not stated in its metadata, so moving to 544p swaps a known
confound for an unknown one.

---

## Not queued, and why

**`min_tokens`.** Now 12288 and inert -- every DiT call is 31k-128k tokens and
every token-refiner call is ~311 rows, so no value we might pick selects
differently. The only question with an answer in it is where the Sol-against-sage
crossover actually sits on this box, and that needs a short-sequence arm this
repo does not render. Real, but it buys nothing.

**Anything needing a merged Comfy-Org/ComfyUI#15908.** The trial ran on a branch
and is reverted; core is stock. Re-applying is one `gh pr diff` away when there
is a reason.
