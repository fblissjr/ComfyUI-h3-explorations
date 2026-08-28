# Queued arms, 2026-08-27

A session queue, not a roadmap. [`docs/roadmap.md`](../../roadmap.md) owns what
to work on and [`docs/open_experiments.md`](../../open_experiments.md) owns what
is deliberately unmeasured. Nothing here is either: it is measurable today, on
tooling that exists. Delete an entry when it runs; one still here in a week
belongs in one of those two instead.

## Status: read the state table below, not this heading

Arm counts go stale within the hour, so this section does not carry one --
**Current state** below is generated from the payloads and the server's history
and is the only place to read status.

**Most arms render the ref dialogue graph** -- two image references, eight `<d>`
lines, the stairwell exchange. That is a change from what several groups were
originally written against, and it has one reason: the market t2v prompt those
arms used was disqualified as a sample by the owner on 2026-08-27, so anything
built on it answers nothing. The exceptions are the `F_` prompt-conformance
arms and the `S_` Sol sweep, which are deliberately on the market scene because
prompt form and Sol settings are what they vary.

The batch is `Video/batch/{arm}_s{seed}`. Sol's `end_percent` is derived from
each arm's step count by the builder rather than carried over, so the 8-step arm
runs 0.87 and the 4-step arms 0.74.

## The queue that was stopped, for the second time today

Twenty-nine arms were built, queued, and stopped mid-flight when the owner moved
every graph back to the v1 conditioner (`72e97c3`). One arm was interrupted
inside `SamplerCustomAdvanced` and wrote nothing; the other twenty-eight never
started. No output survives and none should be looked for.

One arm below is an exception: `h3_image_ref_plus_text_to_video_dialogue_pdd_4step`
was built into the generator afterwards (`3ef32aa`) and is ready to render as
shipped.

**Rebuild the rest from the shipped graphs rather than patching payloads.** Three things
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

## Current state, derived from the payloads on 2026-08-27 evening

**This table is generated from the queued payloads and the server's own
history, not from recall.** Every arm renders the ref dialogue graph except the
`F_` and `S_` families, which are the market t2v scene and carry no references.
The narrative sections below say WHY each group exists; this says what is
actually on the card and with what settings.

Seeds are 730451892/3/4 throughout. `qwen512` is
`h3_config.REF_QWEN_SHORT_EDGE`, inert under the v1 encoder and kept because it
re-arms under v2. Sol is written `start-end tau<t> dense<blocks> min<tokens>`.

### A' -- does attribution hold on v1

ref dialogue, 2 refs, 8 `<d>` lines. **PASSED**

| arm | status | steps | len | heads | refs | Sol |
|---|---|---|---|---|---|---|
| `h3_r2v_dialogue_pdd_4step_00001` | **PASSED** (owner) | 4 | 362 | on | 2 max up=True qwen=512 | 0.2-0.74 tau1.0 dense0-1 min12288 |

### B -- reference sizing at 362 frames

the full sizing axis. `match` had NEVER been rendered before tonight -- 84 of 84 shipped graphs use `max`

| arm | status | steps | len | heads | refs | Sol |
|---|---|---|---|---|---|---|
| `B_match_s730451892` | queued | 4 | 362 | on | 2 match qwen=512 | 0.2-0.74 tau1.0 dense0-1 min12288 |
| `B_match_s730451893` | queued | 4 | 362 | on | 2 match qwen=512 | 0.2-0.74 tau1.0 dense0-1 min12288 |
| `B_match_s730451894` | queued | 4 | 362 | on | 2 match qwen=512 | 0.2-0.74 tau1.0 dense0-1 min12288 |
| `B_noupscale_s730451892` | done | 4 | 362 | on | 2 max up=False qwen=512 | 0.2-0.74 tau1.0 dense0-1 min12288 |
| `B_noupscale_s730451893` | done | 4 | 362 | on | 2 max up=False qwen=512 | 0.2-0.74 tau1.0 dense0-1 min12288 |
| `B_noupscale_s730451894` | done | 4 | 362 | on | 2 max up=False qwen=512 | 0.2-0.74 tau1.0 dense0-1 min12288 |
| `B_upscale_s730451893` | done | 4 | 362 | on | 2 max up=True qwen=512 | 0.2-0.74 tau1.0 dense0-1 min12288 |
| `B_upscale_s730451894` | done | 4 | 362 | on | 2 max up=True qwen=512 | 0.2-0.74 tau1.0 dense0-1 min12288 |

### L -- the same sizing axis at 294 frames (12.25s)

~99,691 packed against 362's ~120,077, which is below both of tonight's OOM points. Second length for the sizing question

| arm | status | steps | len | heads | refs | Sol |
|---|---|---|---|---|---|---|
| `L_12s_match_s730451892` | queued | 4 | 294 | on | 2 match qwen=512 | 0.2-0.74 tau1.0 dense0-1 min12288 |
| `L_12s_noupscale_s730451892` | queued | 4 | 294 | on | 2 max up=False qwen=512 | 0.2-0.74 tau1.0 dense0-1 min12288 |
| `L_12s_upscale_s730451892` | queued | 4 | 294 | on | 2 max up=True qwen=512 | 0.2-0.74 tau1.0 dense0-1 min12288 |

### C -- heads, and step count with Sol on

`patch_heads`, and 8 steps with Sol's window moving with it

| arm | status | steps | len | heads | refs | Sol |
|---|---|---|---|---|---|---|
| `C_pdd4_headfree_s730451892` | done | 4 | 362 | off | 2 max up=True qwen=512 | 0.2-0.74 tau1.0 dense0-1 min12288 |
| `C_pdd4_headfree_s730451893` | done | 4 | 362 | off | 2 max up=True qwen=512 | 0.2-0.74 tau1.0 dense0-1 min12288 |
| `C_pdd4_headfree_s730451894` | done | 4 | 362 | off | 2 max up=True qwen=512 | 0.2-0.74 tau1.0 dense0-1 min12288 |
| `C_pdd8_s730451892` | **OOM** | 8 | 362 | on | 2 max up=True qwen=512 | 0.2-0.87 tau1.0 dense0-1 min12288 |
| `C_pdd8_s730451893` | done | 8 | 362 | on | 2 max up=True qwen=512 | 0.2-0.87 tau1.0 dense0-1 min12288 |
| `C_pdd8_s730451894` | not run -- cancelled while Sol was wrongly suspected of the OOM | 8 | 362 | on | 2 max up=True qwen=512 | 0.2-0.87 tau1.0 dense0-1 min12288 |

### C2 -- step count, clean

4 vs 8 with Sol REMOVED from both

| arm | status | steps | len | heads | refs | Sol |
|---|---|---|---|---|---|---|
| `C2_pdd4_nosol_s730451892` | done | 4 | 362 | on | 2 max up=True qwen=512 | absent |
| `C2_pdd4_nosol_s730451893` | done | 4 | 362 | on | 2 max up=True qwen=512 | absent |
| `C2_pdd4_nosol_s730451894` | done | 4 | 362 | on | 2 max up=True qwen=512 | absent |
| `C2_pdd8_nosol_s730451892` | RUNNING | 8 | 362 | on | 2 max up=True qwen=512 | absent |
| `C2_pdd8_nosol_s730451893` | queued | 8 | 362 | on | 2 max up=True qwen=512 | absent |
| `C2_pdd8_nosol_s730451894` | queued | 8 | 362 | on | 2 max up=True qwen=512 | absent |

### D -- the two knobs

`reuse_qkv_memory` (identity, SETTLED bit-identical) and `start_percent` (timing)

| arm | status | steps | len | heads | refs | Sol |
|---|---|---|---|---|---|---|
| `D_reuse_on_s730451892` | done | 4 | 362 | on | 2 max up=True qwen=512 | 0.2-0.74 tau1.0 dense0-1 min12288 REUSE |
| `D_start0_s730451892` | **OOM** | 4 | 362 | on | 2 max up=True qwen=512 | 0.0-0.74 tau1.0 dense0-1 min12288 |
| `D_start0_s730451893` | done | 4 | 362 | on | 2 max up=True qwen=512 | 0.0-0.74 tau1.0 dense0-1 min12288 |
| `D_start0_s730451894` | done | 4 | 362 | on | 2 max up=True qwen=512 | 0.0-0.74 tau1.0 dense0-1 min12288 |

### F -- prompt conformance

market t2v scene, guide-conformant rewrite against the original

| arm | status | steps | len | heads | refs | Sol |
|---|---|---|---|---|---|---|
| `F_market_v1_s730451893` | done | 4 | 362 | on | none (t2v) | 0.2-0.74 tau1.0 dense0-1 min12288 |
| `F_market_v1_s730451894` | done | 4 | 362 | on | none (t2v) | 0.2-0.74 tau1.0 dense0-1 min12288 |
| `F_market_v2_s730451892` | done | 4 | 362 | on | none (t2v) | 0.2-0.74 tau1.0 dense0-1 min12288 |
| `F_market_v2_s730451893` | done | 4 | 362 | on | none (t2v) | 0.2-0.74 tau1.0 dense0-1 min12288 |
| `F_market_v2_s730451894` | done | 4 | 362 | on | none (t2v) | 0.2-0.74 tau1.0 dense0-1 min12288 |

### S -- Sol settings on the rewritten market scene

one seed each. `S_mktv2_yesterday` is the exact 2026-08-26 19:48 config, read off an embedded workflow

| arm | status | steps | len | heads | refs | Sol |
|---|---|---|---|---|---|---|
| `S_mktv2_dense_none` | queued | 4 | 362 | on | none (t2v) | 0.2-0.74 tau1.0 densenone min12288 |
| `S_mktv2_end090` | queued | 4 | 362 | on | none (t2v) | 0.2-0.9 tau1.0 dense0-1 min12288 |
| `S_mktv2_min4096` | queued | 4 | 362 | on | none (t2v) | 0.2-0.74 tau1.0 dense0-1 min4096 |
| `S_mktv2_sol_off` | queued | 4 | 362 | on | none (t2v) | absent |
| `S_mktv2_start0` | queued | 4 | 362 | on | none (t2v) | 0.0-0.74 tau1.0 dense0-1 min12288 |
| `S_mktv2_tau13` | queued | 4 | 362 | on | none (t2v) | 0.2-0.74 tau1.3 dense0-1 min12288 |
| `S_mktv2_yest_no_dense` | queued | 4 | 362 | on | none (t2v) | 0.2-0.9 tau1.0 dense0-1 min4096 |
| `S_mktv2_yesterday` | queued | 4 | 362 | on | none (t2v) | 0.2-0.9 tau1.0 densenone min4096 |

### G -- does the shipped 8-step reference graph run

shipped defaults, unmodified

| arm | status | steps | len | heads | refs | Sol |
|---|---|---|---|---|---|---|
| `G_shipped_pdd8_ref_asis` | not run -- held: needs its own cold server | 8 | 362 | on | 2 max up=True qwen=512 | 0.2-0.87 tau1.0 dense0-1 min12288 |
| `G_shipped_pdd8_ref_reuse` | not run -- held: same, and it is the paired half | 8 | 362 | on | 2 max up=True qwen=512 | 0.2-0.87 tau1.0 dense0-1 min12288 REUSE |

Re-derive it with:

    python bench/record_render_substrate.py

which also prints each render's position in its server session and how many of
its nodes were cache hits -- **the two numbers that decide whether a duration or
a memory outcome from this batch means anything.** See the cache section below.

---

## The arms, and what each decides

Grouped by what a group is testing, since a single arm rarely decides anything
on its own.

### Group A -- does the prompt come back? NOT ANSWERABLE ON A SHIPPED GRAPH

**Withdrawn as a render arm.** It was `qwen_short_edge` 0 against 512, judged on
whether the dialogue attribution holds. On v1 both arms produce the same encoder
view, so the pair is one arm rendered twice. The *question* survives the
withdrawal and is Group A' below; only the comparison collapsed.

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

### Group A' -- does attribution hold on v1 at all?

> **RAN 2026-08-27, and it PASSES.** Owner's verdict on
> `h3_r2v_dialogue_pdd_4step_00001-audio.mp4`: "looks great". 3.9 min at 4
> evaluations. Attribution holds on v1 with references, dialogue and a distill
> together. **Confirmation, not cause** -- a pass is equally consistent with the
> bounds story, with the seed, and with the original failure being intermittent.
> The causal question is still the layer-50 bounds pair, which needs no render.

**Decides whether today's revert bought what it was done for.** The withdrawal
above took a comparison out; this is not one, and it should not have gone with
it. The misattribution was seen once, under v2. The entire case for moving every
graph back to v1 is that v1's bounds restore the prompt's share of its segment.
Nobody has checked that attribution actually holds under v1, so the largest
decision of the day rests on arithmetic plus one negative observation.

| arm | what it is |
|---|---|
| `h3_image_ref_plus_text_to_video_dialogue_pdd_4step` | references, dialogue and a distill, on v1 |

Built 2026-08-27 (`3ef32aa`). Every **pair** of {references, dialogue, distill}
already shipped and the triple did not, which is why nothing could reproduce the
failure -- it needed all three. The dialogue reference graph with the ref2va PDD
config applied: 4 steps, euler, Sol `end_percent` derived to 0.74 rather than
inherited from the 16-step parent. Seed matches the base graph, so the two are
readable against each other if anyone wants that later.

**One render, one seed, and no blind session** -- rare here, and worth saying
why it is legitimate rather than a shortcut. The prompt binds `<Subject 1>` to
`<Picture 2>` and `<Subject 2>` to `<Picture 1>` **by number**, over eight `<d>`
lines in a concrete stairwell exchange. Whether the woman speaks the woman's
lines is a fact about the render, not a preference between two clips.
CLAUDE.md's different-sample rule governs comparisons; nothing is being compared
here, so it does not bite.

**What it cannot do, and the entry has to say so.** A pass does not establish
that v1's bounds are why. Attribution could hold for reasons unrelated to them --
a different seed, PDD's own behaviour, or a failure that was always
intermittent. It is confirmation, not cause; the causal question is the layer-50
bounds pair in Group A, which needs no render. **A failure is the more
informative outcome**: it would say the ratio story is wrong and the revert did
not buy what it was meant to.

Priced at ~120,077 packed tokens by `bench/preflight_graph.py`, which could not
have told you that until `d7dd575` -- it read the pre-DynamicCombo widget names
and reported 113,037.

### What died on 2026-08-27, and why nothing replaced it

A 4-against-8 finding was assembled during the evening and is **not recorded as
a finding**, because the owner disqualified its only failing sample: "maybe the
prompt just sucked. anyway you can not use that one."

So the state is: **two good renders at 4 evaluations on one prompt family, one
disqualified render, and no failure case.** Not "4 fails on motion", not "4
fails on complex scenes" -- no established failure at all.

Four explanations were fitted to that one render and all four were refuted
within the hour, each by a measurement that took under a minute:

| explanation | what killed it |
|---|---|
| static single-setup vs multi-shot | all three prompts are 3-shot with 2 hard cuts |
| camera and subject motion | motion cannot degrade the AUDIO stream, and the audio was bad too |
| the prompt violates the guide's tags or order | it does not; it is base-en's 3 fields, same structure as the GOOD t2v prompt |
| the prompt is underspecified for its ambition | 387 tokens against the good t2v prompt's 408. The 985 it was compared against is the r2v prompt -- a different prompt family |

The two good renders are also **not independent**: they share their
`overall_soundscape` verbatim and differ only in t2v against r2v. Two prompts,
not three scenes.

**Recording any of the four would be worse than recording nothing**, because
each is a plausible mechanism with a refutation attached, and the mechanism is
the half people carry forward. Anyone who wants to know where 4 evaluations
fails writes a scene for it and renders it.

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

**RUNNING 2026-08-27, and narrower than the group above describes.** Three
`B_noupscale` seeds against three at `allow_upscale` on (A' plus two more), all
on the **ref dialogue graph** rather than the Gate 6 refview population. That is
one scene, not twelve families. It can say whether upscaling is visible on this
scene; it cannot settle Gate 6, which still has never run and still wants its
population.

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

**RUNNING 2026-08-27 on the ref dialogue graph, not the t2v one.** The t2v
prompt this group was written against is the disqualified sample above, so an
arm built on it would answer nothing. Same two widgets, sound prompt, seed
matched to A'. `C_pdd8` picked up Sol `end_percent` 0.87 from the builder's
`SOL_END_PERCENT_BY_STEPS` rather than inheriting 4-step's 0.74 -- the failure
that shipped three broken arms by hand earlier that day is structurally
unavailable through the generator.

**Read `C_pdd8` only if the difference is GROSS.** Step count is a numerical
change, so CLAUDE.md's different-sample rule applies at full force: the 8-step
clip is a DIFFERENT SAMPLE, not a better-resolved version of the 4-step one, and
a matched seed does not rescue that (measured 2026-08-18 -- two arms differing
only in sage `mode` diverged at frame 0 under a deterministic sampler). If
neither clip shows an obvious defect, the honest read is "no visible difference
on one pair", never "8 is no better". Ranking them needs a distribution nobody
has budgeted.

### Group D -- the two knob questions

**Decidable from one render each**, unlike everything above.

| arm | decides |
|---|---|
| `D_reuse_on` against control | whether `reuse_qkv_memory` is free -- bit-identical decoded frames or it is not |
| `D_start0` against control | what `start_percent` 0.2 costs in seconds, at 3 seeds for variance |

> **`D_start0`'s timing is INVALIDATED as run, 2026-08-27.** Its control is A',
> which ran first after a restart and paid the full model load; the `D_start0`
> arms run mid-batch on cache hits. That comparison measures cache position, not
> `start_percent`, and three seeds a side does not help because they all sit on
> the same side of the same systematic difference. It needs re-running under
> `--cache-none`. See the section above. `D_reuse_on` is unaffected -- it is
> decided on decoded frames.

**RUNNING 2026-08-27 on the ref dialogue graph**, not the t2v one these were
named for, for the same reason as Group C. Two consequences worth stating.
`reuse_qkv_memory` is an identity check and a heavier sequence is a strictly
better test of it, so ~120k beats ~109k here. `start_percent` is a timing
measurement, so the absolute seconds are not comparable to anything measured on
t2v -- only the within-pair delta is the answer, and it runs 3 seeds a side.

`reuse_qkv_memory`: **do not try to measure what it saves.**
`bench/bench_e2e_h3.py` spent 2026-08-14 on that and records why it failed -- the
sampled peak resolved the resident-weight plateau rather than the attention
transient the flag targets. Identity only. Watch for one interaction: sage takes
ownership of the float q/k/v list and this flag writes the output into that same
buffer.

`start_percent` has never been measured at any value, ever, and costs a flat 25%
of evaluations at every step count. Run the cheap half first; if the saving is
small the quality question never needs asking.

### A render is not a pure function of its graph

Recorded 2026-08-27 after it invalidated one arm, one instrument and two
hypotheses in the same hour. **ComfyUI's execution cache persists across
prompts within a server session**, keyed by input signature
(`main.py`'s default `CacheType.RAM_PRESSURE`; the server runs with no
`--cache-*` flag). Two graphs differing in one widget share every cached node
upstream of it.

What that did here. `C_pdd8` at two seeds differs only in `RandomNoise`, so the
second submission hit cache on the encoder loader, the conditioning, the UNet
loader and every patch node -- it re-executed the sampler and below. The first
seed ran first after a restart and paid the whole load; it **OOM'd**. The second
paid almost none of it and **rendered**. That was read here as the same
configuration giving opposite outcomes, and it is nothing of the kind.

**So any arm whose outcome is TIME or MEMORY has to state what ran before it.**
Output is unaffected -- a cache hit returns tensors already computed from
identical inputs -- so perceptual comparisons are safe and this rule does not
touch them.

| protocol | when |
|---|---|
| cold restart before every arm | the question is "does this fit as people run it". Reproduces shipped conditions, cache included |
| `--cache-none` | the question is a comparison BETWEEN arms. Makes them comparable to each other; not the shipped configuration, and it changes the memory profile itself |
| neither | any comparison decided on pixels |

Three instruments died to this before it was written down: a "Sol at 8 steps
OOMs" reading, a fits-or-does-not-fit oracle for `reuse_qkv_memory` that was
sound except for assuming determinism, and a same-seed repeat that would have
been a near-total cache hit reporting success without executing the thing that
failed. None of them was wrong about the model. All three treated the runtime
between runs as inert.

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
