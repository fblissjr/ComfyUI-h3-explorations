# Prompt length, paired render: pre-registration

**Written 2026-08-16, after both renders completed and BEFORE either clip was
watched.** That ordering is the whole point of the file. Once you have seen the
output, "which one is better" acquires an answer and the reasons arrive to fit
it. `docs/bench_plan.md` exists for the same reason and this follows its shape.

Nothing here is a result. The verdict goes in a section at the bottom, added
after the owner judges.

---

## The arms

| | arm A | arm B |
|---|---|---|
| `detailed_description` | 46 words | 391 words |
| every other section | **byte-identical** | **byte-identical** |
| prompt tokens | 242 | 646 |
| packed sequence | ~124,582 | ~124,986 |
| render | 28 min, peak 17,840 MiB | 28 min, peak 17,904 MiB |
| output | `Video/h3_r2v_00008-audio.mp4` | `Video/h3_r2v_00009-audio.mp4` |

Built by loading `workflows/h3_image_ref_plus_text_to_video_api.json` once and
substituting the `detailed_description` body only; verified programmatically
that exactly one section differs and the section set is unchanged. Same seed
(730451892), same references, same canvas, same sampler, same 362 frames,
same fl2va + ref-LoRA path.

Both arms are **single shot and silent**. Dialogue and shot structure were
deliberately held back — see "What this cannot answer".

## Why length alone, and why that is not a clean variable anyway

`internal/ref2va_prompt_patterns.md:1105-1126` pre-registered this arm before
today and named the confound: **prompt length moves the rotary origin.**
`comfy/ldm/minimax/model.py:307-318` packs `[text | keyframes | refs | target]`,
gives text positions `arange(text_len)`, then uses `cursor = text_len` as the
temporal origin for every reference block and both target segments. Arm B adds
~404 text tokens, so every reference and target row sits ~404 positions later
than in arm A.

**Length and RoPE origin cannot be separated by this experiment.** A difference
is attributable to "a longer description" as a package, not to the prose.

## Predictions

Committed before viewing. Confidence is 1-5.

| # | prediction | conf |
|---|---|---|
| 1 | **Entrance timing is more deliberate in B.** A says nothing about when he enters; B says "early in the shot". The paired LoRA comparison showed ~5 s vs ~1.5 s entrances on a prompt with no timing at all, so this region is currently free variation | 3 |
| 2 | **B's environment matches `<Picture 2>` more closely.** A says only "a medium shot establishes `<Subject 2>`"; B names the turquoise lake, the mirror reflection, the wildflower foreground, the gold-lit upper faces and the conifer stands | 4 |
| 3 | **B integrates the subject's lighting better.** B explicitly asks for him relit by low warm light from frame left, sitting inside `<Subject 2>`'s lighting; A is silent. Failure mode to look for is the studio photo's shadow carried onto his cheek as if it were a facial feature | 3 |
| 4 | **Camera behaviour is indistinguishable.** Both request `trucks right with small amplitude at slow speed`, identical strings. This is the **internal control**: if the camera differs, that is single-seed noise and it calibrates how much of predictions 1-3 could be noise too | 4 |
| 5 | **Neither clip contains architecture.** Both prompts define `<Subject 2>` as carrying "architecture" and `2-mountain_landscape.png` has none — no buildings, no structures. Genuinely uncertain, which is why it is worth pinning | 2 |

### Prediction 5 is a second experiment riding along

The "architecture" wording is a template defect, not a design. It arrived by
accident and it happens to test the owner's own n=1 observation that
`subject_definitions` is binding authority and `retention_analysis` cannot
correct it — the case where a brunette reference described as blonde rendered
blonde despite `fully_preserved`.

All three outcomes are informative:

- **buildings in both** — a second independent instance of definitions binding,
  on an *absent-attribute* rather than a *wrong-attribute*, which is a stronger
  claim than the blonde case
- **buildings in neither** — the rule is narrower than the blonde case implied,
  and probably binds on identity attributes rather than on inventing absent ones
- **buildings in one** — a difference in prompt adherence between the two, which
  is binary and unmistakable where predictions 1-3 are matters of degree

## What would count as a difference

Forced choice per attribute, plus confidence 1-5, judged with the reference
images open:

1. environment fidelity to `<Picture 2>`
2. subject identity fidelity to `<Picture 1>` (face structure, hair, clothing)
3. lighting integration — is he lit by the scene, or carrying the studio
4. entrance timing and whether he settles at centre as described
5. camera path — expected tie, this is the control
6. architecture present / absent

**A model ignoring the prompt can win on prettiness.** The question is
adherence, not appeal, and those come apart.

## What this cannot answer

- **Dialogue and multi-shot structure.** Held back deliberately. Every shipped
  reference prompt is single-shot and 19 of 20 are silent, and adding three
  variables at once would make a difference unattributable. That is arm 2.
- **Anything general about prompt length.** n=1, one seed, one scene, one prompt
  pair. Direction only.
- **Whether B's prose is any good.** The 391 words are one Claude's attempt at
  the guide's shape. A null result could mean length does not matter, or that
  this particular expansion was poor. That ambiguity is a real limit and cannot
  be resolved by looking harder at these two clips.
- **In-distribution behaviour.** No ref2va example anywhere — MiniMax's
  executable scripts, the official guide, DiffSynth, the Custom-GPT pack —
  exceeds 8 seconds. These run at 15.083 s, roughly double the longest duration
  any authority demonstrates.

## Already settled, independent of the verdict

**A guide-length description costs +404 tokens, +64 MiB and no measurable
time** — 0.3% of the sequence, 28 minutes either way. So "long prompts are
expensive" is not a reason to avoid them, whatever the quality answer turns out
to be.

---

## Verdict

Judged 2026-08-16 by the owner, with frames sampled at 1 / 5 / 9 / 14 s.
**Four of five predictions confirmed. The fifth — the control — was refuted,
and its refutation is the most useful thing here.**

| # | prediction | outcome |
|---|---|---|
| 1 | entrance more deliberate in B | **confirmed.** B has him in frame at 1 s. A is empty at 1 s and he arrives around midway |
| 2 | B's environment matches `<Picture 2>` more closely | **confirmed, strongly.** B reproduces the lake, the mirror reflection, the wildflower foreground and the gold-lit faces. A has a mountain silhouette and none of the rest |
| 3 | B integrates the subject's lighting better | **confirmed.** B lights him warm from frame left, inside the scene. A ends up indoors and backlit — a lighting failure caused by finding 5 rather than by the reference |
| 4 | camera indistinguishable (the control) | **refuted, and the control was invalid.** See below |
| 5 | neither clip contains architecture | **refuted: A has it, B does not** |

### The headline: a detailed description overrode a defective definition

`subject_definitions` is **byte-identical** in both arms and both say
`<Subject 2>` carries "architecture". `2-mountain_landscape.png` has none — no
buildings, no structures.

- **A** opens *inside* a timber veranda with carved brackets, a chalet at frame
  left, and keeps him under that roof for the rest of the clip.
- **B** has no structure anywhere in any sampled frame.

The only difference is that A's `detailed_description` says nothing about the
environment beyond "a medium shot establishes `<Subject 2>`", while B spends
~150 words naming the lake, the reflection, the meadow and the conifers.

**So the defective definition drove the render only where the description was
silent.** That extends the owner's blonde/brunette observation rather than
repeating it: that one established `subject_definitions` beats
`retention_analysis`; this establishes `detailed_description` beats
`subject_definitions`, when it is specific enough. Both are n=1 and both are on
this model in this pipeline, which is more than any guide offers.

The practical form: **a wrong word in `subject_definitions` is load-bearing
exactly to the extent that nothing downstream contradicts it.**

### Why the control was not a control, which invalidates the noise bound

Both prompts carry the identical string `The camera trucks right with small
amplitude at slow speed`. **B executes a camera move; A is essentially static
between 9 s and 14 s.**

The control assumed identical text is an identical instruction. It is not. In A
that sentence is 13 of 46 words, asserted and unsupported. In B it closes 391
words and is *elaborated* — "for the whole take, so the far ridge shifts
slightly against him and the reflection slides across the water while his
position in frame stays near centre" — so the move has consequences the rest of
the description can be checked against.

**You cannot hold one sentence constant inside a prompt whose length you are
varying**, because a conditioning model reads the sentence in context, not as a
string. The control was invalid by construction.

Consequence, stated plainly: **this experiment has no working noise bound.**
Predictions 1, 2, 3 and 5 all point the same way and the architecture result is
binary rather than a matter of degree, which is why the direction is credible.
But nothing here separates "long prompts help" from "this seed happened to
differ", and the instrument that was supposed to do that measured something
else instead.

### What this licenses, and what it does not

**Licensed:** describe the environment explicitly in `detailed_description`,
even when a reference supplies it, because a silent description leaves a
defective or generic definition in charge. And fix `subject_definitions` — the
"architecture" wording is a template defect that has shipped on every image-
reference arm.

**Not licensed:** any general claim that 350-500 words beats 46. n=1, one seed,
one scene, one prompt pair, no working control, at 15.083 s — roughly double
the longest duration any prompting authority demonstrates. The next arm should
vary length on a scene with **no defective definition to override**, which
would separate "long prompts help" from "long prompts happened to correct a bug
in this prompt".

### Still settled, independent of all of the above

+404 tokens, +64 MiB, no measurable time difference. Cost is not a reason to
avoid guide-length prompts.
