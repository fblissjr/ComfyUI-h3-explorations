# The two scene arms: what to look for, and what these renders cannot tell you

`h3_ref2v_scene_subway` and `h3_ref2v_scene_kitchen`, queued 2026-08-28. They
are the first shipped graphs carrying any marker but `<d>`, and the first
ref2va prompts inside the guide's word budget.

**This is a viewing guide, not a result.** It says what the arms were built to
answer and how to read them. Nothing here is a claim about output — that is
what the render is for. If the intent below is not what you wanted from them,
the arms are wrong, not the guide.

---

## 1. What is actually being rendered

Both arms are identical in everything but scene and reference:

| | |
|---|---|
| canvas | 1344x768, 362 frames, 24 fps — 15.08 s, a trained canvas |
| sampler | `er_sde`, `simple`, 16 steps, denoise 1.0 |
| weights | `ref2va_pruned_int8_convrot` DiT, `int8_convrot` encoder, video VAE fp16, audio VAE fp32 |
| attention | sage + Sol, both on, chain asserted |
| reference | `size_policy=max`, `dit_short_edge=2048`, no upscale, `qwen_short_edge=512` |
| output | `Video/h3_r2v_scene_subway`, `Video/h3_r2v_scene_kitchen` |

Subway's reference is a woman performer; kitchen's is a man's face. One
`character` picture each, no environment reference.

**They are not arms of one experiment.** Different scenes, different
references, different speaker counts. Nothing is learned by comparing them to
each other.

---

## 2. What one render each CAN settle

These are presence/absence questions. A single clip answers them because the
behaviour either occurs or does not — no distribution needed.

**Ranked by what is most unknown.**

### 2.1 Does `<|lyrics_start|>` produce singing rather than speech?

The sharpest question here. Both scenes contain sung runs and spoken runs in
the same clip, by design, so the comparison is internal and does not depend on
a second render.

- **subway** Shot 1: "Nobody waits on the northbound line." /
  "Everybody's leaving on time." — sung, wrapped in the lyrics pair.
  Shot 3: "Hold the door and hold your line." — sung.
  Shot 2 and Shot 4 are spoken, unwrapped.
- **kitchen** Shot 2: "Keep it moving, keep it hot." — sung, under the breath,
  against a radio. Every other line is spoken.

Listen for whether the wrapped lines carry pitch and meter and the unwrapped
ones do not. **If everything is spoken, that is a real finding** and it is
the first evidence we have either way — the marker is undocumented and has
never been rendered here.

### 2.2 Does `<|cutoff|>` truncate the last line?

Both scenes end mid-sentence on purpose:

- subway: "Get the next one and meet me at the" `<|cutoff|>`
- kitchen: "And tell them the special is" `<|cutoff|>`

Look for the line running out at the final frame rather than being completed,
rushed to fit, or dropped entirely. Completed or rushed both mean the marker
did not do the job; dropped is a third case and is not the same as truncated.

### 2.3 Does the on-screen text appear at all?

- subway: a platform sign reading **NORTHBOUND - PLATFORM 2**
- kitchen: a ticket reading **TABLE 12 - 2 COVERS - FIRE**

Both are written twice: once as base §4.5's double-quoted string with
typography and placement named, and once as a `<|caption_start|>` pair. **That
confound is deliberate** — see §4.

Read the letters. Legible and correct, legible and wrong, or letter-shaped
noise are three different outcomes; the third is a documented failure mode for
under-specified strings and would say the typography description was not
enough rather than that text does not render.

### 2.4 Does speaker attribution hold?

subway carries **three** speakers (S1 busker, S2 woman in the raincoat, S3 man)
across four shots; kitchen carries two. Misattribution in a multi-speaker
reference scene is the failure that `qwen_short_edge=512` exists to prevent,
and this is the widest test of it we have shipped.

Watch for lines coming out of the wrong mouth, voices swapping between shots,
or the reference identity speaking a line assigned to someone else.

### 2.5 Does the reference identity survive four shots and three cuts?

The reference person appears in Shots 1 and 3 in both scenes and is absent in
between. Look for the face drifting across the gap.

### 2.6 Do the four shots and their cut times actually appear?

subway cuts at 00:03.500, 00:07.000, 00:11.000; kitchen at 00:03.500,
00:07.500, 00:11.500. Four distinct shots, cutting near those marks. A
four-shot prompt rendering as one continuous take is a finding about long
multi-shot descriptions, which nothing here has tested before.

---

## 3. What these renders CANNOT tell you

**Do not read any of the following out of them.**

- **Whether the longer description helped.** These run 392 and 425 words
  against the usual 42-68, but there is no short-form arm on the same scene and
  seed. "Longer is better" is not testable here and was never the claim; the
  claim is only that a prompt in the guide's range is now shippable.
- **Whether any marker contributed anything the prose did not.** Every marker
  here sits beside prose describing the same thing. Isolating a marker needs a
  matched-seed on/off pair on one scene, which is designed and not built.
- **Anything from comparing subway to kitchen.** Different scenes and
  references.
- **Anything about a knob.** `docs/eval_comparison.md` and CLAUDE.md's
  different-sample rule apply unchanged: a rendered clip cannot A/B a numerical
  change, because the trajectory diverges completely from any perturbation.
  These arms are "does the brief come out", not "is this setting better".

A bad clip here is **one draw**. It licenses "this did not work on this seed",
never "this does not work".

---

## 4. The caption question, stated honestly

`<|caption_start|>` appears in neither guide, in no vendor script, and in no
worked example. Its meaning is OPEN. An earlier house reading of it as signage
was withdrawn, and `docs/prompting.md` now places it as a sibling of `<d>` and
says to use base §4.5's quoted string as the primary route and the marker as
**an addition, never a replacement**.

**These scenes originally broke that rule** — they carried the signage as a
bare marker with no quoted string. Fixed in `46a7cc6` before queueing, because
rendering them as they stood would have been ambiguous in the least useful
direction: no sign appearing would not have separated "the marker does
nothing" from "there was no guide-legal route to a sign in the prompt".

So what the render can now say is that **on-screen text renders in our stack**
— which is currently supported only by one colleague's clip, with the prompt
not captured. What it cannot say is which of the two routes produced it.

One thing worth knowing while reading any marker verdict:
`bench/preflight_graph.py`'s caption rules are taken, by its own comment, from
these two scenes. It grading them clean is circular and is not evidence they
are right. The rule they actually broke lives in the manual, which no check
reads.

---

## 4b. What they turned out to be for, which is not what they were built for

Added 2026-08-28 after the renders landed, because it changes their standing.

A peer session established that **artifact severity under PDD tracks motion**
(+0.676 within-clip, coarse partition against fine, frame by frame), and that
the dialogue scenes which look fine at 4 steps contain **no high-motion frames
at all** — so they are structurally incapable of showing a motion-dependent
defect, however carefully or blindly they are scored. `docs/eval_comparison.md`
carries the process rule that follows: a perceptual claim states the motion
regime it was judged in.

They also identified a coverage gap: no high-motion **ref2va** scene existed,
every wide moving scene being t2v. **These two arms close it, and by a margin
nobody predicted.** Measured with `bench/measure_clip_motion.py` on the renders
themselves:

| clip | median | p90 | % frames busy |
|---|---|---|---|
| dialogue, base 16-step | 0.0036 | 0.0082 | 0.6% |
| dialogue, PDD 4-step | 0.0068 | 0.0171 | 5.3% |
| market t2v (`h3_t2v_00015`) | 0.0106 | 0.0219 | 13.0% |
| **`scene_kitchen`** | **0.0283** | 0.0876 | **73.4%** |
| **`scene_subway`** | **0.0295** | 0.0527 | **77.6%** |

About 2.8x the market scene, and the busy-frame share is a different regime
rather than a higher number on the same one. So the corpus's strongest
motion instrument is a pair of arms built for marker coverage and word budget,
and neither purpose predicted it — the four-shot structure and the crowd,
platform and service-line content did.

**What that does NOT yet give you.** Both arms rendered at 16 steps base.
Nothing here is a PDD arm, so they cannot currently discriminate a
motion-dependent distillation artifact — they establish that the *substrate*
exists, not that anything has been measured on it. PDD 4-step and 8-step
variants of these two are the cheap next step, and until they exist "looks
fine at 4 steps" remains a statement about low-motion scenes.

**One confound worth carrying, found in a prompt rather than a render.** The
stairwell dialogue text asks for "the camera shakes slightly with the
operator's breathing". Both dialogue arms share it, so their comparison stays
controlled, but that scene cannot cleanly answer whether distillation changes
camera movement, because the movement is prompted. The market prompt asks for
no shake and is the clean substrate for that question. These two scene arms
prompt camera moves throughout and are **not** clean for it either.

---

## 5. Recording the verdict

Free text, per the house pattern: what you saw, what differed from the brief,
one coarse verdict. Not a scale.

The useful shape for these is per-question rather than per-clip, because §2's
items are independent — singing can work while the caption fails. A verdict
of "bad" on the whole clip loses which of the six it was.

If any of §2.1-2.3 comes back negative, that is a **marker** finding and
belongs in `docs/prompting.md`'s OPEN table with the render named, since those
rows currently say "nothing" under what checks them.
