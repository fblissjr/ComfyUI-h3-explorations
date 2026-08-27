# Queued arms, 2026-08-27

A session queue, not a roadmap. [`docs/roadmap.md`](../../roadmap.md) owns what
to work on and [`docs/open_experiments.md`](../../open_experiments.md) owns what
is deliberately unmeasured. Nothing here is either: it is measurable today, on
tooling that exists. Delete an entry when it runs; one still here in a week
belongs in one of those two instead.

## Status: nothing is staged

**Thirty payloads were built and are now void.** Two things invalidated them
after they were written, and both are worth knowing before rebuilding:

1. `MiniMaxH3AppendRefImage.size_policy` became a DynamicCombo (`e6e527e`), so
   `short_edge` and `allow_upscale` are spelled `size_policy.short_edge` and
   `size_policy.allow_upscale` in API form. The flat names the payloads used no
   longer exist on the node.
2. `qwen_short_edge` now defaults to 512 rather than 0 (below), which changes
   what every reference arm should carry.

Rebuild them from the shipped graphs rather than patching: the graphs were
rebuilt onto the new schema in `3bbf8e1` and now carry both changes, so
regenerating picks them up for free.

Two arms did complete before the stop, and **neither should be judged**:
`base16_ref2va` at one seed, which is the render that exposed the finding below
and is degraded for that reason, and one other.

---

## What changed under everything: the prompt lost its own segment

This is the session's most consequential finding and it reframes the arms below.

Reference tokens land in the **text segment, ahead of the prompt**, so they do
not merely cost sequence length -- they compete with the prompt for it. Under
the v1 encoder every reference clamped to ~290 merged tokens whatever it was
prepared at, so this could not bite. The v2 encoder shipped 2026-08-27 declaring
the release's own bounds, and references now arrive intact.

Priced on the shipped reference graph at 1344x768 x 362, two references:

| | DiT ref rows | qwen tokens | packed | prompt's share of its segment |
|---|---|---|---|---|
| v2, upscale, `qwen_short_edge` 0 -- the old default | 9,408 | 9,408 | 128,971 | **9.5%** |
| v2, upscale, `qwen_short_edge` 512 -- shipped now | 9,408 | 592 | 120,155 | **63%** |
| v2, no upscale | 2,368 | 2,368 | 114,891 | 30% |
| v1 clamp -- what every earlier render got | 9,408 | ~580 | ~120,000 | 63% |

**Observed once:** a two-speaker scene at the old default rendered with the
dialogue attributed to the wrong subject. The subject-to-speaker binding lives
in the prompt tokens whose share had collapsed. One arm, one seed; the mechanism
is arithmetic, the conclusion is not yet measured.

`h3_config.REF_QWEN_SHORT_EDGE` is 512 as of today and says in as many words
that it is a prior rather than a measurement -- 63% is v1's ratio, and v1's
ratio was an accident of a snapshot's pixel bounds rather than a number anyone
chose.

**The two knobs do different jobs, and conflating them is what made an earlier
framing here wrong.** `allow_upscale` decides what the DiT sees.
`qwen_short_edge` decides what the prompt competes with. Before v2 they moved
together and reading them as one knob cost nothing.

---

## The arms, and what each decides

Grouped by what a group is testing, since a single arm rarely decides anything
on its own.

### Group A -- does the prompt come back?

**Decides whether `REF_QWEN_SHORT_EDGE = 512` stays.** The one arm that moves a
default currently resting on reasoning.

| arm | change | reads as |
|---|---|---|
| `ref_qwen0` | `qwen_short_edge` 0 | reproduces the degraded render |
| `ref_qwen512` | `qwen_short_edge` 512 | the shipped default |

Same seed, same prompt, same references, `allow_upscale` on for both. Judged on
whether the dialogue attribution holds, which is a legible pass/fail rather than
a taste judgement -- unusually, this pair CAN be read from one seed each, because
the failure is categorical. `encoderman` has offered to own this arm; the
encoder contract is that lane's.

### Group B -- what does upscaling buy, now that it is separable?

**Decides whether `allow_upscale` should stay on.** The Gate 6 question, which
has had graphs, a 12-family population, matched seeds and a preflight since
2026-08-25 and has never run -- arm A was stopped before it wrote.

| arm | `allow_upscale` | DiT rows |
|---|---|---|
| `ref_ups` | on -- vendor-matching | 9,408 |
| `ref_noups` | off | 2,368 |

Both at `qwen_short_edge` 512, which is what makes this a clean test of the DiT
half alone. Before today it could not be isolated: turning upscale off also
rescued the prompt, so the two effects were confounded.

3 seeds. This one is a quality judgement, not a categorical failure.

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
