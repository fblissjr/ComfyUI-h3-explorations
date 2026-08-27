# Queued arms, 2026-08-27

A session queue, not a roadmap. [`docs/roadmap.md`](../../roadmap.md) owns what
to work on and [`docs/open_experiments.md`](../../open_experiments.md) owns what
is deliberately unmeasured. Nothing here is either: it is measurable today, on
tooling that exists. Delete an entry when it runs; one still here in a week
belongs in one of those two instead.

## Status: nothing is staged, for the second time today

Twenty-nine arms were built, queued, and stopped mid-flight when the owner moved
every graph back to the v1 conditioner (`72e97c3`). One arm was interrupted
inside `SamplerCustomAdvanced` and wrote nothing; the other twenty-eight never
started. No output survives and none should be looked for.

**Rebuild from the shipped graphs rather than patching payloads.** Three things
have moved under them since they were written, and each one silently invalidates
a payload that names the old value:

1. `MiniMaxH3AppendRefImage.size_policy` became a DynamicCombo (`e6e527e`), so
   `short_edge` and `allow_upscale` are spelled `size_policy.short_edge` and
   `size_policy.allow_upscale` in API form.
2. `qwen_short_edge` defaults to 512 rather than 0.
3. `MODELS["clip"]` is the **v1** encoder again (`72e97c3`), which changes what
   a reference costs the prompt and therefore what several arms below decide.

The graphs carry all three. Regenerating picks them up for free.

---

## What changed under everything, and what the encoder switch did to it

Reference tokens land in the **text segment, ahead of the prompt**, so they do
not merely cost sequence length -- they compete with the prompt for it. That is
structural and did not move.

What moved is whether anything can act on it. Priced on the shipped reference
graph at 1344x768 x 362, two references:

| | DiT ref rows | qwen tokens | prompt's share of its segment |
|---|---|---|---|
| v2, upscale, `qwen_short_edge` 0 | 9,408 | 9,408 | **9.5%** |
| v2, upscale, `qwen_short_edge` 512 | 9,408 | 592 | **63%** |
| v2, no upscale | 2,368 | 2,368 | 30% |
| v1 -- what every shipped graph runs now | 9,408 | ~580 | ~63% |

**Under v1 the knob cannot move, and that is measured rather than assumed.**
v1's still bounds are a 1.5x window (200704..301056), narrow enough that
`smart_resize` lands every non-square reference on the identical view whatever
it was prepared at: 512, 1024 and 2048 all arrive as 264 merged tokens at 16:9,
266 at 4:3, and only square moves at all
([`bench/results/2026-08-27_qwen_view_under_snapshot.json`](../../../bench/results/2026-08-27_qwen_view_under_snapshot.json)).
Under v2's bounds the same knob spans 448 to 7,296.

**Observed once, under v2:** a two-speaker scene at the old default rendered
with the dialogue attributed to the wrong subject. The subject-to-speaker
binding lives in the prompt tokens whose share had collapsed. One arm, one seed;
the mechanism is arithmetic, the conclusion is not measured, and the render that
would have tested it was in the queue that stopped.

`h3_config.REF_QWEN_SHORT_EDGE` stays at 512 and its note says why: inert on
every shipped graph today, live again the moment anyone moves the encoder back.

**The two knobs do different jobs.** `allow_upscale` decides what the DiT sees.
`qwen_short_edge` decides what the prompt competes with. Conflating them is what
made an earlier framing here wrong, and under v1 they are separate for a second
reason -- one of them does nothing.

---

## The arms, and what each decides

Grouped by what a group is testing, since a single arm rarely decides anything
on its own.

### Group A -- does the prompt come back? NOT ANSWERABLE ON A SHIPPED GRAPH

**Withdrawn as a render arm.** It was `qwen_short_edge` 0 against 512, judged on
whether the dialogue attribution holds. On v1 both arms produce the same encoder
view, so the pair is one arm rendered twice.

It also had a confound worth recording, because it would have survived the
encoder switch: v2-at-0 differs from v1 in **both** the weights and the bounds,
so a win for 512 was consistent with "the proportion was the problem" and with
"v2's weights are worse and shrinking the view happens to help". The four-encoder
holdout says v2's weights are a wash against v1's on every geometry, which makes
the second reading unlikely but does not exclude it.

**What replaces it needs no render.** Hold the weights fixed and vary only the
snapshot -- v2 weights under v1's bounds against v2 weights under its own, via
`h3_awq_encoder.install_source_processors(image_bounds=...)`, compared at layer
50 on the same rows. Two outcomes: the snapshot is the whole encoder-side
difference and the ratio is the mechanism, or the bounds are not what changed the
state and 512 was treating a symptom. Owned by the encoder lane.

### Group B -- what does upscaling buy?

**Decides whether `allow_upscale` should stay on.** The Gate 6 question, which
has had graphs, a 12-family population, matched seeds and a preflight since
2026-08-25 and has never run.

| arm | `allow_upscale` | DiT rows |
|---|---|---|
| `ref_ups` | on -- vendor-matching | 9,408 |
| `ref_noups` | off | 2,368 |

**The encoder switch made this cleaner, not worse.** It needed both arms held at
one Qwen view so the DiT half moved alone; under v2 that meant setting
`qwen_short_edge` on both and trusting it. Under v1 the clamp does it by
construction -- both arms reach the encoder as the same picture whatever the
stage-one size, so the only surviving difference is DiT reference rows. This is
now the isolation the group was designed for rather than an approximation of it.

3 seeds. A quality judgement, not a categorical failure.

### Group C -- is 4 NFE too few, or is it the head machinery?

**Decides where PDD's quality cost sits.** Two arms, each one widget from the
shipped graph.

| arm | change |
|---|---|
| `pdd8` | 8 evaluations |
| `pdd4_headfree` | `patch_heads` off |

`thirty_two_intervals.html` is why 8 is the comparison: the final evaluation is
63% of the sigma path at 8 NFE against 80% at 4, and Sol covers 5 of 8 steps
against 2 of 4. Both accelerations work harder at 4.

### Group D -- the two knob questions

**Decidable from one render each**, unlike everything above.

| arm | decides |
|---|---|
| `t2v4_reuse_on` against control | whether `reuse_qkv_memory` is free -- bit-identical decoded frames or it is not |
| `t2v4_start0` against control | what `start_percent` 0.2 costs in seconds, at 3 seeds for variance |

`reuse_qkv_memory`: **do not try to measure what it saves.**
`bench/bench_e2e_h3.py` spent 2026-08-14 on that and records why it failed -- the
sampled peak resolved the resident-weight plateau rather than the attention
transient the flag targets. Identity only. Watch for one interaction: sage takes
ownership of the float q/k/v list and this flag writes the output into that same
buffer.

`start_percent` has never been measured at any value, ever, and costs a flat 25%
of evaluations at every step count. Run the cheap half first; if the saving is
small the quality question never needs asking.

### Group E -- the standing item

**The PDD blind session.** Everything built so far establishes correctness and
nobody has judged quality. PDD is pitched against the **turbo distill**, not
against base, and `h3_image_ref_plus_text_to_video_turbo_4step.json` is the
matched control with the LoRA node as the only difference. `docs/eval_comparison.md`
section 3, and the owner's to run.

Stated confound: the turbo was distilled at 544p mixed aspect and the paired arm
renders 1344x768. PDD's own training canvas is not in its metadata, so moving to
544p swaps a known confound for an unknown one.

---

## Reading the outputs

Everything lands in `output/Video/refcmp/` as `{arm}_s{seed}`, in three files:

| file | what it is |
|---|---|
| `{name}.mp4` | video only |
| `{name}-audio.mp4` | the same video with the AAC track muxed -- **this is the one to watch** |
| `{name}.png` | poster frame |

Compare **decoded frames**, never the container: `save_metadata` embeds the
workflow, so two byte-different mp4s can hold identical video.

```
ffmpeg -v error -i A.mp4 -f rawvideo -pix_fmt rgb24 - | md5sum
```

Identical containers do imply identical frames; differing containers imply
nothing.

---

## Not queued, and why

**`min_tokens`.** 12288 since today and inert -- every DiT call is 31k-128k
tokens and every token-refiner call is ~311 rows, so no value selects
differently. The only live question is where the Sol-against-sage crossover
sits, and that needs a short-sequence arm this repo does not render.

**Anything needing Comfy-Org/ComfyUI#15908.** The trial ran on a branch and is
reverted; core is stock. Re-applying is one `gh pr diff` away.

**Base-against-PDD as a quality comparison.** Base runs 16 steps on `er_sde`,
PDD runs 4 on `euler`. Several things move at once, so it answers "did each arm
meet its brief" and never "which is better".
