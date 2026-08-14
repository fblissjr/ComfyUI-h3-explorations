# Sol-Attn bench plan, from 2026-08-14

What to measure next, in what order, what each run decides, and what it
predicts. Predictions are written **before** the run so a result can contradict
them; a plan that cannot be wrong is a to-do list.

Companion to `docs/open_experiments.md`, which is the opposite list — things
deliberately not measured, with the blocker for each.

## The state this starts from

Everything about the CUDA node so far is a kernel-level or arithmetic result.
**There is no end-to-end render measurement of it at all.** The e2e numbers in
`docs/SOLATTN.md` are Triton, and the ones at length 124 are below the token
floor where anything is visible.

So the first run is not a knob sweep. It is the baseline everything else is a
delta against.

## Ground rules for every run here

- **≥60,000 video tokens or the run is uninformative.** 362 frames at 1344x768
  is 107,856, near the model's ~100k ceiling. `bench_e2e_h3.py` warns below the
  floor. A null result under it reads as "this knob does nothing" when the
  truth is "this run could not have shown anything".
- **16 steps**, the shipped `SAMPLING` value, not the bench's 20 default.
- **`--runs 2` plus the discarded warmup.** Arms alternate, so drift is shared.
- **One backend per invocation.** Mixing them compares two kernels and calls it
  a knob; the bench refuses arms whose knobs the active backend lacks.
- **Restart ComfyUI after any node-code change**, then confirm the reload by
  reading the changed value back out of `/object_info` before trusting a run.
- Free the GPU (`POST /free` with `unload_models`) before any CUDA check, or it
  OOMs and looks like a regression.
- Record the kernel build. `check_sol_kernel.py` prints it; paste it in.

A render at 362 frames / 16 steps is roughly 10 minutes, so budget
`(arms x 2 + 1) x 10` minutes.

---

## Scoreboard: predictions against outcomes

Kept because a plan that is never scored is a to-do list. Run 1, 2026-08-14:

> **The length objection to this table is WITHDRAWN, 2026-08-14.** It said
> `--length 362` was "not a legal length". The reference *pipeline* does
> refuse 362 — `max_duration` is a hard-coded 15.0 s and 362 is 15.083 s —
> but **362 is the longest length H3 was trained on**, and the reference's
> ceiling lands one grid step short of it. Calling it illegal was an inference
> from a validator presented as a fact about the checkpoint.
>
> What still stands: these ran against an **fp8** sage baseline the graphs do
> not ship, and the `reuse_qkv_memory` column could not see what it measured.
> Those are the reasons to redo Run 1. The length is not one of them, and the
> redo does not need to move to 345 to be valid.

| question | predicted | measured (at 362) | verdict |
|---|---|---|---|
| Sol vs sage, sampler | 1.35–1.55x | **1.611x** | wrong, low |
| `centroid_tail` on vs off | 5–10% | **2.5%** | wrong, high |
| `reuse_qkv_memory` VRAM | ~1 GB | — | **uninformative**, broken instrument |

Two of three wrong, in opposite directions, which is what pre-registering is
for. The third was not a negative result: the VRAM column was reporting
torch-active bytes and could not have shown a saving.

**A caveat was attached here on 2026-08-14 and then had to be withdrawn the
same day.** It said the length was illegal; it was not. Worth keeping as the
record of both moves: the first was right that an unqualified number in a
verdict column is what gets quoted, and the second is that a caveat asserted
past its evidence costs more than the bare number did. `h3_rules.py`
transcribed the reference's validator correctly and this file turned that into
a claim about the model.

**What the run cost that the plan did not predict:** two bench defects, both
found by running rather than reading. `bench_e2e_h3.py` had been benching
`mode="auto"` sage since 2026-08-13, and its peak-VRAM column was not
measuring device VRAM. Both are now checked (`check_bench_matches_shipped.py`)
or fixed. Budget for this: the first real run of any harness after a gap is
partly a test of the harness.

## Pre-registered: the v2 `sink_q` line on `h3_probe_sol_on_refs`

Written **before** the re-run, 2026-08-14, because a prediction recorded after
the fact is not one. The first attempt died during model staging, one step
before the line printed, but its preflight survived in the log — so the
prediction is arithmetic over measured row counts, not a guess.

From that run's preflight, sequence 120,608: video 102,816, text 8,450,
references 8,192, audio 1,150. So `video_start = 17,792`, the conditioning
region is `ceil(17792/64)` = 278 blocks, and v2 starts the dense query range at
`text_len // 64` = `8450 // 64` = 132.

```
[sol_attn] conditioning sink: KV blocks (0, 278) exact, dense query blocks (132, 278)
```

**Pass/fail, not interpretation.** A narrowing of 132 of 278 blocks, against 4
of 23 on the t2v twin. Both failure modes read `(0, 278)` — v2 not engaging,
and the `audio is None` path falling back to v1 silently — so a start of 132 is
unambiguous and a start of 0 is unambiguous the other way. Anything else means
the block arithmetic is wrong.

This is the pair the two probe graphs exist for. Run the t2v twin in the same
session and both lines can be read against each other.

### The discriminating number does not depend on clip length

Added before the re-run, and it makes this cheap. **The query start is
`text_len // 64` and text does not scale with frame count.** Only video and
audio rows do. So the 132 is invariant: it is the same at 39 frames as at 345,
and a short run tests the number that discriminates without the load that
killed the box on the first attempt.

At length 39, holding the prompt and both references **unchanged** — which is
mandatory, since substituting the prompt is what made every earlier preflight
figure describe the harness instead of a graph:

```
text        8,450   unchanged, does not scale with length
references  8,192   unchanged
audio         130   ~80 rows/s at 1.625 s
video      12,096   12 latent frames x 1008
sequence   28,868   still far above min_tokens 4096, so Sol engages
```

`video_start` = 16,772 → 263 blocks. Predicted:

```
[sol_attn] conditioning sink: KV blocks (0, 263) exact, dense query blocks (132, 263)
```

**The start is the assertion; the end is the weaker half.** 132 is arithmetic
over a measured text length. 263 carries an extrapolated audio row count, so a
small miss there is a bad audio estimate, not a failed mechanism. A start of
132 confirms v2; a start of 0 is v1 or the silent `audio is None` fallback;
anything else means the block arithmetic is wrong.

If this passes, the 345-frame run becomes a confirmation rather than the only
way to learn the answer.

### RESULT 2026-08-14: v2 confirmed, and the prediction was wrong

Ran at length 39, prompt and both references untouched, `verbose=True`:

```
[h3] preflight ... 39 frames ... sequence length 28,868
[sol_attn] conditioning sink: KV blocks (0, 263) exact, dense query blocks (260, 263)
[sol_attn] sparse (1, 28868, 56, 128) tau=1.3 cuda-int8
```

| | predicted | logged |
|---|---|---|
| sequence | 28,868 | **28,868** |
| KV blocks | `(0, 263)` | **`(0, 263)`** |
| query start | 132 | **260** |

**v2 works, and works harder than predicted.** The whole conditioning region
stays exact as keys and values, and only the last **3 blocks of 263** run dense
on the query side. Those 3 are the target audio. Reference queries are sparse,
which is the entire point of the change, and a v1 fallback would have printed
`(0, 263)`.

**The query start was wrong because I carried a t2v identity into a reference
graph.** v2 uses `audio_start // 64`. On t2v, `audio_start == text_len` because
the layout is `[text][audio][video]` with nothing in between — I had written
that sentence myself, correctly, days-of-argument earlier in this session. With
references the layout is `[text][refs][audio][video]`, so
`audio_start = 8450 + 8192 = 16,642` and `16642 // 64 = 260`. Not 132.

That is the same failure this whole plan documents: **a relation derived under
one configuration, applied to another.** It is the fourth instance today and
the first where the author caught it, and only because the number was written
down in advance where it could go red. Predicting `text_len // 64` and seeing
260 is a caught error; predicting nothing and seeing 260 is a shrug.

The pre-registration also localised the error precisely. Sequence and KV blocks
matched to the digit, so the segment arithmetic was right and exactly one term
was wrong — no hunting.

**Consequence for the 345-frame run:** it is now a confirmation, not the
experiment. Same references and prompt, so `audio_start` is unchanged and the
query start should still be **260**; only the KV end moves with video and audio
rows. The mechanism question is answered.

**Not a speed measurement.** Reference rows are pinned exact and cannot be
sparsified, so this arm should be slower per token than the t2v twin while
still verifying the mechanism.

---

## Run 1 — the foundation, plus two knobs that ride along free

```bash
python bench/bench_e2e_h3.py --length 362 --steps 16 --runs 2 \
  --arms "sage,shipped,shipped[centroid_tail=0],shipped[reuse_qkv_memory=1]"
```

Four arms, ~90 minutes, one shared `sage` control. Three questions at once.

**Q1: is Sol worth shipping on at all, here, on the CUDA kernel?**
`sage` vs `shipped`. This is the number the project does not have.
*Prediction:* 1.35–1.55x on the sampler. The Triton path measured 1.39x at
tau 1.2 with int8 at 362 frames; the CUDA kernel routes in INT8 unconditionally
and upstream reports it 1.4x over Triton e2e, but that 1.4x was at his settings
on his box, and our tau is 1.3 rather than 1.2. If it lands under 1.2x, suspect
a silent dense fallback before believing the number — check the log for
`cuda-int8`.

**Q2: how much of the CUDA advantage is `centroid_tail`?**
`shipped` vs `shipped[centroid_tail=0]`. **This one has a deadline** — upstream
is weighing making the toggle unconditional, and if that lands the question
becomes unanswerable.
*Prediction:* 5–10% on the sampler, per upstream's own e2e figure. If it comes
out near 1.4x, then the earlier claim this repo retracted (that `centroid_tail`
*is* the CUDA-over-Triton gap) was right after all and the retraction was
wrong. Either result is worth having.

**Q3: what does `reuse_qkv_memory` buy in headroom?**
`shipped` vs `shipped[reuse_qkv_memory=1]`, read from the peak VRAM column.
Verified numerically identical, so this cannot change output — it is pure
headroom, and headroom is what gates longer clips. The heaviest shipped config
peaks at 21,186 MiB of 24,564, leaving ~3.4 GB.
*Prediction:* ~1 GB saved at this length (upstream says ~1.2 GB at 80k tokens,
and this is 108k), and sampler time within noise. If it saves nothing, the
buffer is not being reused and the flag is inert here.

**What Run 1 decides:** whether Sol stays opt-in or becomes the default; whether
`centroid_tail` needs defending upstream; and whether `reuse_qkv_memory` should
be turned on in `SOL_RECOMMENDED_CUDA`.

---

## Run 1b — tau 1.0 against 1.3, with a positive control

Pre-registered 2026-08-14, before running. Owner's call to redo the tau sweep
after kijai said "should do a tau sweep at some point"; owner's constraint that
it must not be t2v-only.

### Why the old sweep does not answer this

A tau sweep was already run on **Triton** Sol and found drift above ~1.5. Three
things make that non-transferable, and the first is a live risk rather than a
technicality.

**`centroid_tail` moves the cliff and probably moves it down.** Triton pools
the tail **per row**. The CUDA node defaults `centroid_tail=True` — one pooled
tail per 64-token query block, 64x coarser. Coarser pooling is exactly the
mechanism that would relocate a tau cliff, in the direction of damage at *lower*
tau. If the CUDA cliff is under 1.5, the shipped 1.3 is already past the edge.
The old finding was measured on the other tail mode and cannot rule that out.

**The band that decides anything was never sampled.** The old sweep established
"≥1.5 is bad". Kijai says quality *peaks* at 1.0 and degrades monotonically
above it. Both can hold — gradual from 1.0, cliff at 1.5 — and 1.0 against 1.3
is the only comparison that moves `SOL_RECOMMENDED_CUDA`.

**Regime.** The old arms ran at 124 frames, below the token floor, where
long-range temporal dependence — the thing tau damages — is weakest. (An
earlier draft of this said 362 was "not a legal length" as a second reason.
Withdrawn: 362 is trained, and only the reference pipeline declines it.)

### What tau actually touches — checked, not assumed

`_sink_blocks` returns `blocks = (0, ceil(video_start/64))` as the exact-KV set
in **every** mode except `off`. So text, keyframe, reference and audio rows are
exact keys for every query **at every tau**. Tau does not gate the
reference→video pathway at all; it sparsifies **video↔video** attention.

This matters for how the run is justified. References are not more
tau-sensitive. What a reference buys is an **oracle**: `1-man.png` is a face,
and "does this still look like the reference" is a judgement a person can make,
where "does this look worse" is the judgement that once rated one plain-sage
render "dramatically more interesting" than two others differing only by seed.

### AMENDED before running — the range goes UP, not down

The draft below picked 1.0 / 1.15 / 1.3 and said "drop everything above, we
know ≥1.5 is bad". That premise came from the old Triton sweep and it did not
survive contact with upstream. Kijai, 2026-08-14, running **tau 2.0**:

> `8/8 [01:38<00:00, 12.26s/it]` @tau 2.0, which is very sparse — it looked
> good. i think we can push it up

Owner's read: "not good but acceptable."

That is in apparent tension with his own "1.0 is where quality peaks" and it
resolves the same way both statements are true: degradation from 1.0 is
**monotonic but gentle enough that 2.0 is still usable**. If that holds on our
shapes, sitting at 1.3 leaves a large amount of speed on the table for a
quality difference nobody has looked at — and 2.0 is dramatically sparser.

So the sweep brackets the curve instead of sampling one end. Narrowing to
1.0–1.3 would have measured the half where nothing is at stake.

### Arms

Five renders, one seed, 345 frames, CUDA, `centroid_tail` at its shipped
default. Three tau values spanning the range, on the workload with an oracle;
two on the workload written to surface artifacts.

| arm | graph | canvas | tau | role |
|---|---|---|---|---|
| refs-1.0 | `h3_probe_sol_on_refs` | 1024x768 | 1.0 | kijai's quality peak |
| refs-1.3 | `h3_probe_sol_on_refs` | 1024x768 | 1.3 | shipped baseline |
| refs-2.0 | `h3_probe_sol_on_refs` | 1024x768 | 2.0 | the one he says is usable |
| t2v-1.3 | `h3_probe_sol_on` | 1344x768 | 1.3 | control, shipped |
| t2v-2.0 | `h3_probe_sol_on` | 1344x768 | 2.0 | control, far end |

**The t2v pair is a control, not a second experiment, and it is the part that
makes a null interpretable.** The refs prompt is "the camera trucks right with
small amplitude at slow speed" — a single continuous shot with mild motion,
which is the *least* favourable content for surfacing a block-sparse artifact.
This repo has already recorded that weakness once: the earlier quality nulls
came from a slow-camera, diffuse-fog prompt "where a block-sparse artifact is
least likely to surface". The t2v prompt was written deliberately with a whip
pan, brick and railings, rain texture and percussive audio so a router artifact
has somewhere to show.

So: **refs null + t2v null** means tau 1.0-vs-1.3 is invisible at 345 frames.
**refs null + t2v difference** means our reference test content cannot detect
this and the null is about the prompt, not about tau. Without the control those
two are indistinguishable, which is the "a run under the floor produces a null
that reads as a finding" trap in a different costume.

Refs run at **1024x768** because the 1344x768 refs graph killed the server
during DiT staging earlier today, with both references upscaled to 2048.

### Predictions

- **Speed:** 1.0 slower than 1.3 on both pairs, single-digit percent. Tau
  changes routing density, and the sink is small on t2v.
- **t2v quality:** a visible difference at 345 frames, or the control has
  failed and no null from this run means anything.
- **refs quality:** genuinely uncertain, which is why it is being run. If 1.0
  is the quality peak, face identity should degrade monotonically 1.0 → 1.3 →
  2.0 across the clip.
- **The decision it changes:** if 1.0 and 2.0 are hard to tell apart on the
  refs pair, tau is not the lever this repo has treated it as, and the shipped
  1.3 should move UP for the speed. If 2.0 is clearly worse and 1.0 clearly
  better, 1.3 is a defensible middle and stays.
- **What would falsify kijai's "push it up":** 2.0 visibly damaged on either
  workload at 345 frames. His datapoint was 8 steps at an unrecorded canvas
  and length; ours is 16 steps at 345, where long-range temporal dependence is
  strongest and a router artifact has the most room to accumulate.

### The gate, stated honestly

There is no instrument (`docs/open_experiments.md` #14). This is a person
watching each clip to the end, tracking the face against `1-man.png` on the
refs pair and one small persistent object on the t2v pair. Stills cannot judge
it — the failure is a small object dissolving over ~4 frames, and a grid of
sampled shot-times will miss it. **Renders produced unattended; the judgement is
not made until the owner watches them.** Recording a speed delta and filing
this as answered is the exact failure this plan keeps documenting.

---

## Run 1c — upstream's two questions, sharing Run 1b's control

Added 2026-08-14, mid-run. Kijai, signing off:

> gotta sleep so good luck testing, I'm most interested in: is morton worth
> anything / is centroid_tail ok as default

Both are **quality** questions and this repo has only speed numbers for either.
`morton=False` was set on a Triton speed result (1.16x alone, a net loss
stacked on int8); its quality effect is unmeasured, which kijai has said
himself. `centroid_tail=True` is the CUDA node's default and we measured 2.5%
e2e — a cost, not a verdict on whether it is *ok*.

**`centroid_tail` is the one with a clock on it.** Upstream is weighing making
it unconditional. If that lands, `centroid_tail=False` disappears and the A/B
separating the toggle from the kernel becomes unrunnable. That is the only
experiment in this plan that can expire.

### AMENDED mid-run — no t2v, no turbo LoRAs

Owner, 2026-08-14, while the t2v arms were running:

> stop testing t2v and dont test distill turbo loras - just base with sage/sol
> and use ref images/videos/audio and input images

So the design below is rewritten and the t2v arms are cancelled, including the
one already rendering. The argument for t2v — that its prompt was written to
surface router artifacts — was a good argument for a workload the owner does
not run. **A knob validated on content nobody renders is not validated**, which
is the same shape as every other finding on this page: measured under one
configuration, applied to another.

`t2v-1.3` had completed (10:18 at 345 frames, 1344x768) and is kept only as a
timing datapoint. It is no longer the control.

**The control is now `h3_probe_sol_on_refs` at shipped settings** — tau 1.3,
`centroid_tail=True`, `morton=False`, 1024x768, `allow_upscale=False`. Every
arm differs from it by one field. Base model plus sage and Sol throughout; no
turbo or distill LoRA anywhere in this run.

| arm | differs by | answers |
|---|---|---|
| `refs-1.3` | — | the control |
| `refs-1.3-centroid_off` | `centroid_tail=False` | kijai: is the default ok |
| `refs-1.3-morton_on` | `morton=True` | kijai: is morton worth anything |
| `refs-1.0` | `tau=1.0` | his stated quality peak |
| `refs-2.0` | `tau=2.0` | the setting he says is usable |

The cost of moving off t2v is real and worth stating: the reference prompt is a
single continuous shot with the camera trucking right at small amplitude, which
is the *least* favourable content for surfacing a block-sparse artifact. A null
on any of these arms is therefore weak evidence, not proof of no effect. What
replaces the t2v control as a sanity check is the face oracle — `1-man.png` is
a reference the subject can be compared against directly.

**Still uncovered, and the owner asked for it:** video references, audio
references, and input images. No Sol-enabled graph exists for any of those —
the shipped reference graphs omit the Sol node from their API form entirely, so
it cannot be patched in at submit time. That needs a `sol_on=True` entry in
`GRAPHS`, which is a graph addition rather than a bench flag.

### The efficient part: the control already exists

`t2v-1.3` from Run 1b is shipped settings at tau 1.3 — `centroid_tail=True`,
`morton=False`. It is rendering now. So each question costs **one** render
against it, not a pair:

| arm | differs from `t2v-1.3` by | answers |
|---|---|---|
| `t2v-1.3-centroid_off` | `centroid_tail=False` | is the default ok |
| `t2v-1.3-morton_on` | `morton=True` | is morton worth anything |

Same seed, same prompt, same canvas, same length, same tau. One variable each.

Run on t2v deliberately, and this is the one place t2v is the *right* choice:
its prompt was written with a whip pan, brick and railings, rain texture and
percussive audio so a router artifact has somewhere to show. Neither of these
knobs is about reference handling, so the reference oracle buys nothing here
and the artifact-sensitive content buys everything.

### Predictions

- **`centroid_tail=False`**: slower by roughly the 2.5% already measured, and
  *slightly* more accurate — it is the finer per-row tail. If the quality
  difference is invisible at 345 frames, "ok as default" is answered yes and
  upstream can make it unconditional without cost.
- **`morton=True`**: slower (1.16x on Triton, and the CUDA cost is unmeasured).
  Quality genuinely unknown — this is the one arm here with no prior at all.
  Reordering video tokens into compact 3D neighbourhoods changes *which* blocks
  the router keeps, which is the most plausible mechanism for a quality change
  in the whole config.
- **What would surprise me**: morton visibly helping. If it does, the shipped
  `False` is trading an unmeasured gain for a measured speed win, and that is
  the wrong trade for a repo whose default is accuracy over speed.

### Ordering

These jump ahead of Run 1b's remaining reference arms. The tau question has
upstream's own datapoint at 2.0 and no deadline; `centroid_tail` has a
deadline and an upstream request. Run 1b's refs arms follow.

---

## RESULT — morton drops a reference feature. Owner-observed, control pending.

2026-08-14. Five reference arms at seed 730451892, 345 frames, 1024x768,
`allow_upscale=False`, base checkpoints, sage + Sol, no LoRA.

### Speed, which is the half that answers nothing

| arm | sampler+decode | vs control |
|---|---|---|
| `refs-1.0` | 462.0 s | +2.4% |
| `refs-1.3` (control) | 451.0 s | — |
| `refs-1.3-centroid_off` | 453.4 s | +0.5% |
| `refs-1.3-morton_on` | 446.8 s | **-0.9%** |
| `refs-2.0` | 420.4 s | -6.8% |

All five: sequence 83,840 tokens, `conditioning sink: KV blocks (0, 86) exact,
dense query blocks (68, 86)`. Identical across arms, as it should be — the sink
does not depend on tau.

**These knobs are far smaller here than on t2v.** `centroid_tail` costs 0.5%
against the 2.5% measured on t2v; the whole tau range 1.0→2.0 spans 9.9%.
Consistent with the exact-work arithmetic: reference rows are pinned exact at
any tau, so there is less for Sol to sparsify. On the workload actually
rendered, tau is barely a speed lever.

### The finding, and it came from a person looking

The owner, comparing first frames: every arm except morton shows the
reference's snow gullies on the high slopes; morton shows green forested
hillside with no peak at all.

Verified against the reference. `2-mountain_landscape.png` is a snow-capped
peak with white gullies running down dark rock — the most distinctive thing in
it. The prompt says `<Subject 2>` is the environment "whose architecture,
palette, and lighting are carried into the target video", `fully_preserved`.
Checked both tau extremes independently: 1.0 and 2.0 both retain the gullies.

So **four arms retain the feature and morton alone loses it**, across a routing
change (tau 1.0 → 2.0) large enough that it should have disturbed it if
anything would.

**Morton did engage** — verified, not inferred:
`[sol_attn] H3 Morton: ACTIVE: video span [5504, 83840), grid (102, 24, 32),
curve 2d_frame`, and 5504 is exactly the 86-block sink boundary.

### Why this is not yet a result

Every arm diverged compositionally — different building angles, camera
positions, one gained a pond. That is expected at 16 steps of a flow ODE and is
this repo's own recorded trap: comparing finished renders measures chaos, not
quality. **n=1 for morton.** A 4-vs-1 pattern is more than divergence should
produce, since divergence scatters randomly rather than removing one specific
referenced feature from exactly the arm that reorders tokens — but that is an
argument, not a control.

**The control is queued:** the same pair at seed 424242, control first. If
morton loses the peak again and the control keeps it again, it is morton. If
the control also loses it, the first observation was the trajectory moving.

### Provisional answer to kijai's question

"Is morton worth anything" — **provisionally no, and possibly negative.** It
costs 0.9% *in its favour* on speed, i.e. it is free, and it appears to cost
reference fidelity. If the seed control holds, `morton=False` is right for the
reason the config never had: it was set on a Triton speed measurement, and the
quality axis kijai flagged as untested is the one that condemns it.

`centroid_tail` is untouched by this: `centroid_off` retained the feature, so
nothing here argues against the default. Its quality verdict still needs
watching, not stills.

---

## Run 2 — `start_percent`, the knob with no justification

```bash
python bench/bench_e2e_h3.py --length 362 --steps 16 --runs 2 \
  --arms "shipped,shipped+start0.0,shipped+start0.1,shipped+start0.3"
```

Four arms, ~90 minutes. 0.4 is dropped: it costs three of sixteen steps of
sparsity, and nothing suggests the quality gain is worth that when 0.3 costs one.

`start_percent=0.2` is the only knob in the shipped config with no measured
rationale — it is the paper's number, carried through. Upstream reports a later
start affects motion least, which would make it the cheapest quality lever.

The band is **not** a step fraction. Computed for `simple` at `shift_video=12.0`:

| start | sparse steps of 16 |
|---|---|
| 0.0 | 15 (94%) |
| 0.1 | 13 (81%) |
| **0.2 shipped** | **11 (69%)** |
| 0.3 | 10 (62%) |
| 0.4 | 8 (50%) |

*Prediction:* sampler time falls roughly with the sparse-step count, so 0.0
should be ~15% faster than 0.2 and 0.3 ~5% slower. The interesting result is
quality, not time: the moving-content artifact should appear at the low end
first, and if motion really is the least-affected axis, 0.3 should buy it back
for less time than lowering tau does.

**This run cannot be judged from stills.** The failure mode is a small
persistent object dissolving partway through a clip, over about four frames. It
needs watching to the end, tracking one small object. The bench's long prompt
was written with a whip pan, brick and railings, rain texture and percussive
audio precisely so a router artifact has somewhere to show.

---

## Run 3 — `min_tokens`: CANCELLED, there is nothing to measure

Superseded 2026-08-14. This was planned as "does sparsifying the small
conditioning calls matter", on the belief that a render makes attention calls
at several sizes -- the smoke log shows 2048 and 4608 alongside 12,264.

**Those small calls are our own instrumentation.** `SageChainAssert` fires two
synthetic probes, 2048 and 4608 tokens, precisely to check the sparse gate
takes one and declines the other. They are the entire population.

H3's DiT has **exactly one** attention site, `comfy/ldm/minimax/model.py`,
and S there is the full packed length. Frame counts satisfy `n % 17 == 5`, so
at 1344x768 the shortest clip past 5 frames is 22 frames -> S = 7,194, already
above 4096; only a 5-frame render (S ~ 2,096) falls below. (Found by the sage
fork's claude, confirmed against the installed tree here.)

So at any real length `min_tokens` at 4096 and at 12288 select the same thing:
everything. The knob cannot change what happens, and an arm would measure
noise and report it as "within prediction" -- the worst kind of green.

Two things follow, both more useful than the cancelled run:

- **`min_tokens=4096` in `SOL_RECOMMENDED_CUDA` is harmless, not wrong.** The
  earlier note calling it "very likely wrong" was reasoning from a call
  distribution that does not exist. It only bites below 22 frames, which is
  below the token floor anyway.
- **Sage still runs 5 of 16 steps in a Sol arm.** RETRACTED 2026-08-14: this
  said sage takes "zero DiT calls" with Sol on, so a sage-config change moved
  only the sage-only arm. That was right about the `min_tokens` gate and
  ignored the **sigma window**. At `start=0.2/end=0.9`, steps 0-3 and 15 fall
  outside Sol's window; the compose gate declines them and they run sage's
  forward patch at fp16. So `mode="auto"` -> `fp16` slows BOTH arms, and the
  corrected ratio improves less than the one-sided reasoning predicted.
  Corollary worth testing: a Sol render's peak VRAM may be set by those 5
  dense steps rather than the 11 sparse ones, which would explain
  `reuse_qkv_memory` measuring nothing -- it shrinks an allocation on the
  steps that are not setting the peak.

## PRIORITY INVERSION, 2026-08-14: stop measuring t2v

Measured today, and it reverses the ordering this plan was built on.

The first shipped-graph segment breakdown (345f, 1344x768) is
`104,277 = video 102,816 + audio 1,150 + text 311`. Conditioning is **1.4% of
the sequence**. So on t2v, `exact_kv` against `exact_kv_and_rows` moves 2.8% of
the attention work to 2.5%. There is nothing there to find, and the sink work
aimed at t2v was aimed at the wrong workload.

The larger reversal: **t2v is where Sol-Attn helps MOST, not least.** Reference
rows are pinned exact, so they raise the token count without adding anything
Sol can sparsify. Arithmetic over the measured row counts, at 10% assumed
routed density:

| configuration | forced exact | attn ceiling | e2e ceiling |
|---|---|---|---|
| t2v | 2.8% | 8.00x | 1.84x |
| 3 image refs at `match` | 8.2% | 5.76x | 1.76x |
| 1 video ref, 124f | 32.2% | 2.57x | 1.47x |
| 1 video ref, 345f | **59.2%** | **1.58x** | **1.24x** |

Run 1 measured 1.611x e2e on t2v against a 1.84x ceiling -- 87% of what the
arithmetic allows, which is a sanity check on both.

This repo (and this plan) has been saying reference-heavy work is "the workload
with the most reason to want Sol". **That is backwards.** It is the workload
where Sol has the least room.

It is also the real argument for v2, and not the one in its tooltip: v2 does
nothing for t2v (1.84x -> 1.85x) and takes the 345-frame video-reference case
from 59.2% forced exact to 36.6%, e2e ceiling 1.24x -> 1.45x.

**Decision: drop t2v sink work entirely.** Keep t2v Sol arms, because that is
where the win lives. Move `sink_conditioning` to the reference bench, where it
is now the difference between Sol being worth running and not.

Caveat: arithmetic over measured row counts with an assumed 10% routed density.
The density is the soft input and it moves every ceiling above.

## Run 4 — `sink_conditioning` at reference load

**Blocked on a build.** `bench_e2e_h3.py` is t2v-only — no reference wiring at
all. This needs `LoadImage` → `MiniMaxH3ReferenceFit` → the r2v path, as a
`--refs` axis, so a paired same-seed A/B is still possible.

It is the most expensive to set up. It was also called "the highest-value
unmeasured question", and both that ranking and the number behind it are
withdrawn — see below.

> **The 23-point swing is retracted. Do not quote it, here or anywhere.**
> Arithmetic over `docs/h3_references.md`'s row counts put one video reference
> at **57.9%** of attention forced exact under `exact_kv_and_rows` against
> **35.1%** under `exact_kv`. Wrong on three axes for anything we ship:
> a **v1 formula** — kijai's v2 stopped running reference *queries* dense, so
> the query-side term no longer scales with reference size at all; a **362-frame
> target**, which is not a legal length; and **1344x768**, where every shipped
> reference arm is 1024x768.
>
> The "345-frame video reference" phrasing is its own trap and is why this
> survived a caveat sweep: 345 is the **reference** length and reads as the
> shipped config, which hides that the **target** was 362. A retracted number
> wearing a legal-looking number next to it is harder to spot than a bare one.

**And the ranking inverted.** Reference rows are pinned exact, so they raise the
token count without adding anything Sol can sparsify — arithmetic over the
preflight's measured rows puts a video-reference arm's attention ceiling near
1.58x against t2v's ~8x. Reference-heavy is where Sol has the **least** room,
not the most, which is the opposite of what this plan assumed when it ranked
this run first.

The question is still worth answering — `sink_conditioning` is a real knob and
nobody has measured it at reference load on either node version. What is gone
is the number that made it look urgent. Re-derive under v2 before scheduling
this, and expect a smaller answer.

*Prediction:* at one image reference at `match` the two settings are within
noise; with a video reference `exact_kv` is worth 15–25% of sampler time, and
the cost shows up in generated audio, which is the reason
`exact_kv_and_rows` is on.

---

## Run 5 — re-baseline the frontier above the floor

Everything in `docs/SOLATTN.md`'s frontier table is at length 124 = 37,296
tokens. Until this runs, most of that page cannot be quoted for the CUDA path.

Lowest priority not because it does not matter, but because Runs 1–3 produce
most of it as a side effect.

---

## Deliberately not planned

- **CUDA vs Triton e2e, ours.** Confirmatory only: the migration already
  happened, on an accuracy argument that does not depend on the ratio. One arm
  of `shipped_triton` under `--sol-backend triton` gets it whenever someone
  wants it.
- **`routed_cap_percent`.** Upstream reports ~30 as lossless with 3x headroom.
  It trades quality for memory, and `reuse_qkv_memory` addresses memory without
  touching quality, so measure that first and come back only if headroom is
  still the binding constraint.
- **`tau` re-sweep.** Already measured at 362 frames on Triton, and the artifact
  onset near 1.5 is the binding constraint rather than the timing.
