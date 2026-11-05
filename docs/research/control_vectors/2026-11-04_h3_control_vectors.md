# Control vectors for H3

A dated research record, written 2026-11-04 at the owner's ask, for the owner. It is a
design and an experiment plan, written like a paper for a reader doing this out of curiosity: the claim first, the
evidence behind it last, with current build and experiment status tracked in the Status section below.

**Terms.** A *control vector* (the papers also say *steering vector*) is a direction in a model's activations, added
at run time with a strength. Ours are made the contrastive way: the mean difference between prompts that differ in
one quality (contrastive activation addition, CAA). The *injection point* is where the vector is added. ComfyUI-Dials
and ComfyUI-MAINodes call their H3 version "dials"; that word is used here only for theirs.

## Abstract

H3 has no negative prompt and no guidance branch, reads one long structured prompt, and refines that prompt's text once
before every block of the DiT reads it. We think a control vector added to the refined text rows can shift one quality
of a render (a film stock, how much grain, whether lettering appears, how a room sounds) continuously and with less
collateral change than rewriting the prompt, which adds tokens, moves every media row's time origin and can tip a
quality from ignored to dominant. The first build is deliberately thin: a mean-difference vector, added after the
token refiner, masked off the vision rows, measured under the prompt frame the render uses. One gate decides whether
the project continues: on two shots and two seeds, does a control vector give something a well-written prompt edit
does not? A removal (negative strength on "lettering") is the likeliest win, since nothing else in H3 can subtract; a
film look is the hard case, since prompts already do looks well. Everything else (sigma schedules, holds, earlier
encoder layers, speaker scoping, discovery of unnamed directions) waits on that answer.

## Status

Updated 2026-11-05 at `main` `42e3d3bb`. The plan below (sections 2-6) is unchanged; this section says how far it has got.

**Built and on `main`**

| piece | commits | how it was checked |
|---|---|---|
| this paper | `6bb1eaad`, `cca2eacb` | two skeptical reads of the earlier draft, one Gemini read |
| Phase 1: `control_vectors.py`, `h3_spans.py`, the vector file format, `bench/check_control_vectors.py` | `d5541e44`, `fd55dc9b`, follow-ups in `4ecaa7df` | Gemini's read and two Claude reads (before and after landing), break lists |
| Phase 0: pair files in `prompt_bank/control_vectors/` (16mm 1974, lettering, noir, reverb, and bare twins), four edit baselines (`prompt_bank/*_cv_*.txt`), plus six joint audio-video pair sets (`felt_muppet`, `cursed_cctv`, `chaos_panic`, `infomercial_fail`, `fever_melt`, `vocal_strain`) | `17c908f5`, `2026-11-05` | Gemini's prompt audit and read; `control_vectors.load_pairs` |
| Phase 2: `MiniMaxH3MakeControlVector`, `MiniMaxH3ApplyControlVectors`, `bench/control_vectors.py` (`make`, `inspect`, `grade`), `build_control_vector_arm` | `4ecaa7df`, `0ac7ce04` | Gemini's read; a Claude cross-read of `c215b327` whose findings were fixed in `e0ce534f`; landed before a read of `e0ce534f` at the owner's call |
| Built early from Phase 4: holds, `project_out` at make time, `inspect` and the meter | in the Phase 1-2 commits | C4, C7 |
| The easy batch: overwrite-safe vector names (`default_vector_filename`, `save_vector(..., overwrite=False)`), an apply-time `mode` (`add`, `project_out`, `amplify`), a step window (`start_percent`, `end_percent`, by fraction of schedule steps), timing prints in `make` and `grade`, `MiniMaxH3LayerNormMatch`, unshipped gate and probe graph builders (`build_control_vector_gate_arms`, `build_control_vector_probe_arm`) | `dcbec82d` | CPU checks (`bench/check_control_vectors.py`, `bench/check_schema_defaults.py`, `bench/check_imports.py`, `bench/check_doc_inventory.py`) and 0 graph differences; the pre-landing read waived by the owner |
| Pika H3 Flex `token_refiner` LoRA branch support (`lora_branch.py` `h3_lora_branches` metadata, `_attach_refiner_lora_branches`, `__lora_<8hex>` filename disambiguation, `make --lora`, `grade --pika`) and self-contained authoring system prompt ([`2026-11-05_control_vector_system_prompt.md`](2026-11-05_control_vector_system_prompt.md)) | `42e3d3bb` | independent blind read (`b9b3f726`) + `bench/check_control_vectors.py` + `bench/break_checks.py` |

**Run on the GPU (`2026-11-05`, [`bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md))**

- **Base vs. Pika `token_refiner` vectors** (`data/results/2026-11-05_cv_base_vs_pika_comparison.json`): cosine similarity between base and Pika `svd995` post-refiner vectors is `0.996611` on `film_16mm_1974` (`fl2va`), `0.996678` on `lettering` (`fl2va`), and `0.995730` on `lettering` (`ref2va`); `svd995` vs. full-rank `pika_comfy` is `0.999991`.
- **Row-0 norm census (E2.1) & Step-2 velocity grade (`grade --pika`, `data/results/2026-11-05_cv_grade_film_16mm_1974_pika.json`)**: Qwen3-VL encoder row 0 is `3.59x` the median text row norm (`435.80` vs. `121.48`), which `token_refiner` normalizes to `1.12x` (`104.37` vs. `93.31`). On `t2va_offpath_tortoise`, `ctx_matched_strength = 1.0352`, Step-2 (`sigma = 0.9231`) `matched_strength_video = 1.3543`, and `s = +2.0` reaches `cos_edit_video = +0.4889` and `cos_edit_audio = +0.6785` against the 83-token prompt edit while keeping $L = 276$ fixed.
- **24 Pika 6-step renders at `1344x768x141` (`t2va` gate, 6 fun vectors, and `ref2va` masking/windows)**:
  - `t2va` gate (`output/Video/cv_pika_gate/`, clips `01`–`10`): `film_16mm_1974` at `s = +2.71` (`apply_holds=False`) morphed the modern 2020s London bus into a 1970s half-cab AEC Routemaster while keeping the man and tortoise fixed; `lettering` at `s = -1.0` removed street signage at fixed $L = 276$ without the architecture re-roll caused by the 51-token negative prompt edit (`07_lettering_edit`, $L = 327$).
  - Joint audio-video steering (`output/Video/cv_pika_fun/`, clips `11`–`16`): combining 8 visual pairs (`integrated_multimodal_description:`) + 2 acoustic pairs (`overall_soundscape:`) achieved `split_half_cosine = 0.92–0.93` and steered both streams simultaneously (e.g., `felt_muppet` turned the subject into a green-felt Muppet with a Muppet voice yell despite `"He produces no vocal sound"` in the base prompt).
  - `ref2va` (`output/Video/cv_pika_ref2va/`, clips `17`–`24` on `ref2va_woman_blonde_street.txt`): 3D Video VAE `ref_latents` keep `<Subject 1>`'s face and outfit locked to `woman_blonde.jpg` while `film_16mm_1974`, `noir`, and `felt_muppet` transform the surrounding street, cars, lighting, and vocal delivery. Applying `film_16mm_1974` on a late step window (`start_percent = 0.33, end_percent = 1.0`, clip `23`) locked 100% of the baseline street geometry and parked cars (`MAD = 12.31` vs. `37.18` full-window) while applying only the 16mm optical color/grain grade.

**Open, in order**

1. The pre-gate measurements (section 3.2) at real geometry: row norms (E2.1), the nulls (E2.2), the mask (E2.3), the
   prefix (E2.4), the frame (E2.5), the pooling (E2.6), int8 against bf16 (E2.7), the meter (E2.8). A dense forward at
   full geometry is about 120 s (`bench/measure_marker_epsilon.py`), so each is minutes of GPU time.
2. The gate (section 3.3): matched strengths from `grade`, then the gate graphs shipped through `GRAPHS`, renders on
   both shots and seeds, and the owner's reading.
3. Only if the gate earns it: the section 3.4 experiments, whose code is now built; then the section 3.5 reach work,
   which still needs code: per-speaker and `overall_soundscape` spans (new parsing in `h3_spans.py`, whose `speaker`
   category is only the `(S1)` tag tokens) routed through an object patch on `extra_conds`, an encoder-point apply,
   the LLM-layer hook, and morph.
4. Section 6, each only if a named vector fails in a way it would fix.
5. A post-landing read of the Phase 2 and easy-batch code, if wanted (`internal/claude/2026-11-04_dials/POST_LANDING_READ_BRIEF.md`).
6. Appendix A: the contradicted docs are still uncorrected (checked 2026-11-04: `docs/h3_conditioning_end_to_end.md:47`
   unchanged; `docs/wiki/references.md` has a dated note but still pins the old sglang hash).

## 1. Hypothesis and theory

### 1.1 What a control vector is

Encode prompts A and B that differ in one quality, average the difference over many such pairs, and the subject and
wording that vary between pairs cancel, leaving the shared direction. Add a multiple of it to a new prompt's
conditioning and the render moves along that quality. No training and no images; a vector file is kilobytes. It can
turn up how much of something the model already knows. It cannot teach anything new: not a person, not a style the
model lacks.

Vocabulary used below: a *hold* is a direction projected out of a vector so it does not change that quality; the
*meter* is the projection of a prompt's conditioning onto a vector, how far along it the prompt already sits; a *step
schedule* is strength as a function of sigma.

### 1.2 Hypotheses

- **The main claim.** On H3's long structured prompts, a control vector shifts one quality with less collateral
  change (subject, composition, action, sound) than editing the words, and at strengths words cannot give: a fraction,
  beyond the strongest wording, or reversed.
- **The mask claim.** In fl2va and ref2va the text sequence also holds the reference pictures and video blocks. Leaving
  those rows out of the push keeps the references as they were without weakening the effect.
- **The frame claim.** A vector measured from pairs set in the kind of prompt frame the render uses (and, for ref2va,
  under the render's reference prefix) is cleaner than one measured from short bare phrases.
- **The ablation claim.** Our version (masked, framed, measured under the prefix) beats a faithful MAINodes-style
  dial (unmasked, short pairs, plain mean) on the same pairs.

### 1.3 Why it should work, and why H3 is different from an LLM

The method is the LLM steering lineage (activation addition, CAA, persona vectors, representation engineering's
control vectors). ComfyUI-MAINodes has already shown it works on H3 at the post-refiner point (section 8). What makes
H3 its own case, all read in code (section 7):

- **No guidance branch.** H3 is guidance-distilled: one positive pass per step, no negative prompt. A control vector at
  negative strength is the only way to subtract.
- **The text is refined once and read by every block.** Core runs `condition_proj` and the 2-block token refiner once
  per sampling, then the text rows sit at the front of one joint sequence that all 50 blocks attend over. A push there
  reaches video and audio alike.
- **A prompt edit is not free.** Adding words adds tokens: the text length L sets every media row's time origin, and
  new tokens compete in the encoder's causal attention and the DiT's joint attention. A control vector changes no row
  count.
- **The text sequence is mixed.** In fl2va and ref2va the vision blocks sit inside the text span (tag 0) and get the
  video adaLN branch in the DiT; labels, timestamps, speaker tags and `<d>` dialogue are text (tag 1). A uniform push
  moves all of it, which is the case for the mask.
- **The encoder is causal, the refiner is not.** Qwen3-VL computes each row from the rows before it, so a reference
  prefix shapes every prompt row after it. The token refiner then attends both ways over the whole span, so after it
  every row, the vision rows included, carries part of any difference.

### 1.4 Why it might not

- A 32B encoder reads "shot on 16mm in 1974" very well; a prompt edit may simply win.
- MAINodes' own lessons say some qualities do not come from this point at all (camera shake, zero gravity) and that
  holds leak.
- Generic wording can swamp a vector; MAINodes measured that at the point before the refiner, and debias exists for it.
- A vector made from short pairs is measured in a different attention regime from a long scene prompt (the refiner
  has no RoPE and attends over the whole span), so it may not transfer.

### 1.5 What we expect it to do, and not do

| expect | do not expect |
|---|---|
| removals with no negative-prompt alternative: lettering, extras, a score nobody asked for | new identities or styles the model does not know |
| continuous amounts of a look: grain, sheen, depth of field, haze | likeness control (not an identity tool) |
| looks and sound qualities with the scene's wording left alone | delivery and voice control in the first build (needs dialogue spans) |
| a meter: where a prompt already sits along a vector, with no render | per-frame control over clip time (MAINodes' two-pass blend; not planned) |

## 2. Approach

### 2.1 The first build

```
pairs file (prompt_bank/control_vectors/)
   -> MiniMaxH3MakeControlVector   in the server: encode each pair in its frame (and, for ref2va,
                                   on top of the cached reference pass), condition_proj + refiner,
                                   pool scene-text rows, mean of B - A  -> vector file
   -> MiniMaxH3ApplyControlVectors DIFFUSION_MODEL wrapper: context += strength * v on tag-1 rows,
                                   tags read from the payload; a list of vectors summed in float64
   -> render
```

Left out of the first build on purpose: debias, holds, sigma schedules, per-block strength, the encoder and LLM-layer
points, dialogue and speaker spans, picture-derived vectors, audio protect. Each comes back only through an experiment
in section 3.

Choices, each from a correction to the earlier draft (the reviews are listed in section 8.3):

- **After the refiner, not the encoder.** The encoder point is the same push as MAINodes' pre-refiner bias, where they
  measured generic wording swamping the vector. Portability across model variants buys little: the local checkpoints
  fall into three text-path families (section 7.3).
- **A DIFFUSION_MODEL wrapper, not a hook on `preprocess_text_embeds`.** That function receives only the text tensor,
  never the tags, so it cannot mask. The wrapper has the refined context and the payload's tags together.
- **Add, do not rotate.** Every DiT block normalizes its input first, so at block 0 a rotation and an add of the
  matching angle give the same input, and rotation's top end erases the prompt.
- **Pool scene-text rows after the refiner.** The refiner attends both ways, so rows before the first differing word
  carry the difference too; "pool from the first differing token" is exact only at the encoder.
- **Strength has one unit per injection point, stated in the vector file.** After the refiner, every row has just
  come out of the refiner's final RMSNorm, so row norms are already equalized and say little; there, strength 1 is
  one mean pair difference (the vector as stored), and the gate's strengths are multiples of the edit-matched
  strength, which `grade` finds. At the encoder and LLM-layer points the rows are not normalized, so strength is a
  fraction of the median tag-1 row norm there, with row 0 left out until E2.1 shows whether it is an outlier. Either
  way it is a unit, not a fix for overcooking.
- **The vector file names its recipe**: the base checkpoint, LoRA files and strengths, the encoder file, the mode and
  the prefix. A model mismatch is refused; a prefix mismatch is reported.

### 2.2 Baselines and controls

- **The prompt edit**, written as well as the house writes. The bank's look family (`t2va_look_anchor` and its siblings,
  each holding the content and changing one look axis) is the model.
- **A faithful MAINodes-style dial** on the same pairs: unmasked, short bare pairs, plain mean over all rows.
- **Nulls**: a random partition of the pairs into two disjoint halves (`n - floor(n/2)` and `floor(n/2)`), taking half the difference between the two halves' weighted means (each normalized by its own `sum(|w|)` so a shared direction cancels identically for even or odd `n` and equal or unequal weights) and rescaling to the vector's norm (2026-11-04: used to say "A and B swapped in a random half of the pairs and rescaled to the vector's norm", which leaves `1/n` of the signal for odd `n` and ignores weights); a random direction of the
  same norm; a semantically empty edit (the prompt with " ." appended); strength 0. Shuffling pairs is not a null: a
  mean of pair differences is mean(B) minus mean(A) whatever the pairing.

### 2.3 The gate

| result | then |
|---|---|
| a control vector row is preferred on both shots and seeds, for either vector | continue with that vector as the reference |
| only the removal vector wins | narrow to removals and negative strengths |
| the edit wins and the vector moved the prediction the way the edit did | the post-refiner point adds nothing words do not; stop building, keep the meter, try one reach probe before closing |
| the edit wins and the vector moved it some other way | the pairs teach a nuisance; rewrite them once and gate once more |
| the vector never moves the prediction beyond the nulls | the point does not reach these qualities; go to the reach experiments or stop |

### 2.4 Why this order

The project should be cheap to stop. The gate needs only the math, `make`, the apply node with the tag mask, the
nulls and one generated sheet arm, so those are built first and properly (tracked, checked, on a worktree branch).
The inspection tools, the meter, the vector library and everything in section 6 come after the gate, or not at all.

## 3. Experiments

### 3.1 Common setup

- **Two readings, kept apart.** `grade` compares the DiT's prediction with and without a vector at fixed noise, latent
  and sigma (the method of `bench/measure_encoder_quant_dit.py`), on video rows and audio rows separately: how far and
  which way it moved. The owner's eye on a stacked ladder decides whether the change is the quality and worth having.
  A grade figure never says good or broken.
- **Hard checks, green or red**: strength 0 reproduces the plain prediction exactly; masked rows are bitwise unchanged;
  L is unchanged; a vector made on another model is refused in the tool's own words.
- **Shots**: two per experiment, two seeds each. For the gate, `t2va_offpath_tortoise` (a zebra-crossing street whose
  prompt asks for no text, so any lettering is spontaneous) and `ref2va_look_anchor` (a reference-led shot, where the
  mask and the long structured prompt matter). Canvas from `workflows/h3_config.py::CANVAS_TIERS` at a probe size
  `AGENTS.md` allows; length from `h3_config.LENGTH`.
- **The base arm only**: stock attention plus Sol as generated for renders (`workflows/build_workflows.py::build_control_vector_arm`), and stock dense attention by default for single-step `grade` activation probes (`--sol` opt-in, because Sol's dynamic sparse mask re-routes keys when conditioning changes; `bench/measure_marker_epsilon.py:33-46`; 2026-11-04: used to say "stock attention plus Sol as generated" without distinguishing render arms from `grade`'s activation probe), no LoRA, the base step count. Every result names its
  DiT file and the ComfyUI core revision.
- **Pairs never contain the test shots**, so a vector cannot win by memorizing them.
- **Every prompt is read in full** from the queued graph (`bench/what_h3_got.py`) before it renders; graphs come from
  `workflows/build_workflows.py`.

### 3.2 Before the gate: does the instrument read true?

All activation work; one throwaway render run.

| | question | setup | decides |
|---|---|---|---|
| E2.1 | are a few rows or channels outliers? | the first `make` prints per-row norms (row 0 against the median) and the share of the vector in its largest channels | whether row 0 is left out, whether a channel-scaled cosine is needed |
| E2.2 | does a vector move the prediction more than the nulls? | grade the vector and every null at three probe steps from the arm's own schedule | whether anything is worth rendering |
| E2.3 | does the mask change the prediction? | the same vector masked and unmasked on `ref2va_look_anchor` and `i2va_look_anchor` captures | whether the mask earns its code |
| E2.4 | does the prefix matter? | the same pairs made under no prefix and on top of the ref2va reference pass; cosine of the two vectors and of their prediction changes | whether one library vector serves every mode, or vectors are also made per shot |
| E2.5 | does the frame matter? | the same quality made from bare phrases and from pairs set in full scene frames | how pair files are written |
| E2.6 | which pooling after the refiner? | scene-text rows against all tag-1 rows against rows from the first difference on | the pooling rule |
| E2.7 | do vectors carry between encoder files? | made on the int8_convrot and the bf16 encoder | whether the encoder file belongs in the recipe check |
| E2.8 | does the meter read sensibly? | noir and faded-16mm vectors from held-out pairs; meter the anchor and its eight look siblings | whether the meter is a useful prompt tool, gate or no gate |
| E2.9 | does the sheet arm work? | one shot, one seed, three strengths | nothing beyond that it runs; not a result |

A pair-count rule replaces a guessed constant: add pairs until two disjoint halves of them agree (split-half
cosine) and stop rising.

### 3.3 The gate

Two vectors chosen to bracket the question:

1. **16mm 1974 against modern digital**: the hard case. Prompts already render looks, so a win here generalizes.
2. **On-screen lettering, at negative strength**: the likely win. Pairs are a plain surface against the same surface
   covered in printed lettering, over many subjects. The edit arm writes the negation in words, as the house already
   does (the `t2va_look_neon` brief asks for no lettering).

Optional third: **dry studio against cathedral reverb**, only the soundscape differing, judged by ear.

Arms per shot and seed (base shared):

| arm | what | why |
|---|---|---|
| base | strength 0 | the reference row |
| edit | the B words written in, or the negation written in | the baseline |
| vector, matched | the strength whose prediction change on video rows equals the edit's | equal push: the comparison is direction and collateral, not size |
| vector, ladder | 0.25, 0.5 and 2 times matched | continuous dosage, the thing words cannot do |
| vector, reversed | minus matched (for the removal vector, minus double) | a direction the edit cannot express |
| MAINodes-style | the faithful dial at matched strength | whether our changes matter |

Before any render, `grade` prints each arm's movement on video and audio rows against the nulls, and the cosine
between the vector's prediction change and the edit's. The edit also changes L, which moves every media row's
position relative to the unchanged prompt rows (relative RoPE leaves target-to-target and reference-to-target offsets
alone). A length-matched padding control (the base prompt with as many neutral tokens added as the edit adds) sizes
that part, so `grade` can print how much of the edit's movement is its words and how much its length. A
vector whose movement sits inside the null spread is not rendered. The matched strength comes from `grade`, so it has
the edit's own movement as its provenance.

On `ref2va_look_anchor` the ladder doubles as a collateral test: does the referenced person stay the same person as
the look moves, masked and MAINodes-style side by side.

Renders: 4 base, then 28 per vector (edit plus six vector arms, at two shots by two seeds), before pruning by grade.
One page on the local review page, rows stacked with frame numbers, each clip headed by its load name. Questions in
free text, asked once: which row reads most as the quality (or most free of lettering); which kept the shot as it
was; whether the half, double and reversed rows are something you would want on a slider.

### 3.4 After the gate: control (only on a vector that passed)

| | hypothesis | arms | decides |
|---|---|---|---|
| E4.1 | a look vector applied only late keeps the base composition; a layout vector (empty against packed street) applied late does little | early-only and late-only step schedules at the gate's matched peak | whether step schedules (a port of MAINodes') earn a place on base H3 |
| E4.2 | a late vector does something a late prompt switch does not | stock `ConditioningSetTimestepRange`: base prompt early, edit late, beside E4.1's late row; first a hard check that it runs with H3's payload | whether step schedules are worth building at all |
| E4.3 | earlier encoder layers are coherent at all | stock `CLIPSetLastLayer` at layers 45 and 40, raw; norm-matched needs a small node | whether the LLM-layer point (section 3.5) is worth building |
| E4.4 | `project_out` removes only what the prompt carries; a negative add works either way | the lettering vector on `t2va_noodle_bar` (asks for a sign) and the tortoise shot (asks for none) | which removal the apply node offers, and where |
| E4.5 | a hold takes one property out without leaking | cozy with a warm-tint hold, meter before and after | whether holds stay |
| E4.6 | stacked vectors add, or share one limit | three vectors together, grade against the sum of each alone | whether a list of vectors is calibrated as a stack |

E4.1 runs with and without the frozen cache: text rows stay live on its cached steps, so a vector or step schedule in the
wrapper reaches every block; only the frozen rows' attention inputs come from the first step.

### 3.5 Reach: what the post-refiner point cannot do

Run on a quality that failed the gate or that MAINodes lists as unreachable.

| | hypothesis | decides |
|---|---|---|
| E5.1 | the encoder-point vector, after debias, moves the prediction the same way as the post-refiner one | whether a second injection point is worth keeping |
| E5.2 | a vector added at a middle encoder layer reaches "static shot against strongly shaking camera", which the post-refiner point does not | whether the LLM-layer point and morph are built further |
| E5.3 | morphing two equal-length prompts at a layer is smoother than a vector ladder on the same quality | whether morph is a tool |
| E5.4 | an energy vector on S2's rows only stays on S2 | whether speaker scoping survives joint attention; dialogue and speaker spans reach the wrapper through an object patch on `extra_conds`, which receives every conditioning key |

### 3.6 Long shots, one try each

- Minus "talking" for silent subjects, where words leave mouths moving.
- Minus "cinematic", the house opener, for a plainer realism.
- Push only the keyframe rows in i2va: the bidirectional refiner may carry a look from the picture into the text.
- Slow motion as a strength.
- One-pair vectors from the look family (anchor against noir).
- Reverse playback.

## 4. What to build: examples

Grouped by what they exercise. "Gate" marks qualities a prompt edit is likely to do poorly at. Every pair keeps every
word but the quality, describes the image rather than the action, and gives the A side its own neutral opposite.

| group | vectors |
|---|---|
| look, era, material | 16mm 1974 (gate's hard case); grain amount (gate); felt puppets; tilt-shift miniature; wet sheen without rain (gate); minus "cinematic" |
| light and weather | hard against soft key light (gate); haze density (`amplify` on a prompt that has haze); colour temperature with no time of day; fog distance; rain removal by `project_out` where the prompt mentions rain |
| camera and lens | depth of field (gate); wide against telephoto (reach); steady against shaking (reach); framing tightness and camera height (layout, for sigma tests) |
| performance and energy | motion amount; crowd density; a dog's mood; delivery energy on one speaker (reach) |
| sound, room, music | room reverb; ambience level; score mood; score presence at negative strength; heard through a wall |
| removal by negative strength | lettering (gate); strangers; mouths moving; interface overlays |
| arithmetic | cozy minus orange (holds); noir plus rain plus 1974 (stacking); 30% noir with 70% 16mm (weights); a vector made on beaches applied to interiors (transfer) |
| masked video-to-video lane | match the plate's texture; match the plate's light; strangers out; silent mouth. Each behind the lane's own tool or plan data, never a shared default |

What a prompt edit cannot easily say: a fraction of a look; the negative of something the prompt never names; the same
change with L held fixed; a change only in the late steps; one consistent push across a chain of windows whose
prompts differ.

## 5. Build plan

Every phase is a branch in its own worktree under `../../h3_worktrees/`, cut from current `main`, landed after green
`bench/check_*.py`, verbatim `bench/break_checks.py` output and a blind read by a session that did not build it.

| phase | delivers | must hold before the next |
|---|---|---|
| 0 | the core revision recorded; the gate's pair files, edit prompts and nulls written | the edits are as well written as the house's |
| 1 | `control_vectors.py` (pure torch: means, weights, the null constructions, holds, `project_out`, mean pairwise cosine, split-half, `meter_score`); `h3_spans.py` (tags, labels, timestamps, `<d>` and speaker tags by their special-token ids); the vector file format; `bench/check_control_vectors.py` | tested with no GPU |
| 2 | `MiniMaxH3MakeControlVector`, `MiniMaxH3ApplyControlVectors`, `bench/control_vectors.py make`, `inspect` and `grade`, `build_control_vector_arm`; section 3.2 | the hard checks are green; E2.2 shows movement beyond the nulls |
| 3 | the gate, section 3.3 | the owner's reading of the ladder |
| 4 | apply-time `mode` (`add`, `project_out`, `amplify`), step schedules (`start_percent`, `end_percent`), `MiniMaxH3LayerNormMatch`, `build_control_vector_gate_arms`, `build_control_vector_probe_arm` (`ConditioningSetTimestepRange`, `CLIPSetLastLayer`) | each piece graded before it is kept |
| 5 | the encoder and LLM-layer points, morph, dialogue and speaker spans | each reach experiment's answer |
| 6 | section 6, each only if a named vector fails in a way it would fix | |

## 6. Later, and not doing

**Later**, each with what would start it:

- **Picture-derived vectors**, MAINodes' clever design: pool the text rows after a picture, which read it, rather than
  the vision rows. When a look can be shown but not named.
- **Per-block strength**, as a hook that refuses the frozen cache, whose cached steps skip block hooks.
- **Audio protect** (audio queries reading the text without the vector): custom attention inside Sol and core attention.
  Only if the dialogue mask leaves voices drifting.
- **Discovery**: an atlas of the encoder output's top principal directions over the house prompts, rendered and
  labelled. Ten directions with random controls first, to see whether they are qualities at all.
- **A self-report labeller**: steer stock Qwen3-VL-32B-Instruct (H3's encoder, which needs its layers 50 to 63 and LM
  head) and ask it what it is thinking about, as vlens did. Needs the same weights and a steering endpoint on the
  owner's Mac server (heylookitsanllm), sketched as: prompt, vector, layer, strength in row-norm units, positions,
  max tokens. A hint, not ground truth.
- **An automatic scorer** for ladders, such as EmbeddingGemma 2 (one space for text, video and audio, local weights):
  untested on style, light and camera; not for delivery zero-shot.
- **A trained slider** (Concept Sliders style, on the model's own predictions) for a quality no vector reaches;
  contrastive weight steering (Fierro et al. 2025) reports trained directions generalize further.
- **A LoRA's fingerprint**: project the shift a LoRA makes to the refined text rows onto the vector library.
- **Subspaces, LEACE erasure, covariance matching**: only if a named vector fails in a way they would fix.
- **Carrying a vector to another model family** (ridge maps at tokens ending on the same character, as in
  neural-pairing).

**Not doing**: rotation as the default; neuron-level sweeps (vlens found one neuron did nothing where the mean
direction worked); grids of vector, layer, block and sigma; per-layer strength profiles for every vector; a 200-direction
atlas; prompt-side `<think>` steering (H3 has no template).

## 7. Facts the design rests on

Read in code or weights on 2026-11-04, ComfyUI core at `5f415e31a`.

### 7.1 The encoder

- Stock Qwen3-VL-32B-Instruct, byte-identical (`docs/evidence.md`, `data/results/2026-08-25_released_encoder_is_stock.json`);
  64 decoder layers, of which H3 reads the output of HF `layers.49`. Core builds only those 50, with no final norm or
  LM head (`comfy/text_encoders/llama.py::Qwen3VL_32BConfig`); both local encoder files hold only them.
- No chat template, `<think>`, BOS, EOS or padding (`comfy/text_encoders/minimax.py:3-4`;
  `comfy/text_encoders/qwen3vl.py:150-151`). `<d>`, `</d>` and the other markers are special tokens
  (`comfy/text_encoders/minimax.py:181`), also listed in the release's FL2VA and Ref2VA tokenizer configs.
- Sequence order: t2va is the prompt; fl2va is `<Picture i>: ` plus a vision block per keyframe, then the prompt;
  ref2va is each condition in request order (pictures, `<Audio j>: ` labels with no tensor, `<Video k>: ` with a
  `<T.T seconds>` stamp and vision block per frame pair), then the prompt.
- Tags: 0 over each vision block including its start and end tokens, 1 everywhere else
  (`comfy/text_encoders/minimax.py:79-86`).
- Causal; after a vision block, text resumes at a compressed M-RoPE position (`comfy/text_encoders/qwen_vl.py:92-115`).
  DeepStack adds vision-tower features to the vision rows after LLM layers 0 to 2 (`comfy/text_encoders/llama.py:1092`).
- An LLM layer writes its output into its input tensor (`comfy/text_encoders/llama.py:842`): a capture clones.
- Earlier layers: `SDClipModel.set_clip_options` takes a negative index or a list (`comfy/sd1_clip.py:152-164`). Stock
  `CLIPSetLastLayer` gives the raw residual at layers 27 to 50 through the shipped conditioning nodes; the list form
  returns a 4-D stack H3 cannot use; the pack's encode-once path refuses a selected layer (`reference_encode.py:183`).

### 7.2 From conditioning to the DiT

- `MiniMaxH3.extra_conds` refines the text once per sampling (`comfy/model_base.py:2223-2230`) and puts the tags in
  `minimax_payload` (`comfy/model_base.py:2240, 2268`); `_forward` reads them (`comfy/ldm/minimax/model.py:722`).
  `preprocess_text_embeds` itself receives only the text tensor.
- Every conditioning key reaches `extra_conds` as a keyword argument (`comfy/samplers.py:937-951`); H3's `extra_conds`
  forwards only the keys it knows, so a span map of our own needs an object patch on `extra_conds`, which core applies
  by setting the attribute on the model (`comfy/model_patcher.py:1138-1139`). The patch is in place when `extra_conds`
  runs: `outer_sample` loads the model through `prepare_sampling` (`comfy/samplers.py:1242`,
  `comfy/sampler_helpers.py:201`) before `inner_sample` processes the conds (`comfy/samplers.py:1226`). Found by the
  Gemini read.
- The refiner: 2 blocks, no RoPE, no adaLN, no mask, ending in an RMSNorm (`comfy/ldm/minimax/model.py:250-276`).
- The DiT: text first in one joint sequence with no attention mask; text rows take the video timestep and are updated
  by all 50 blocks; the final layer reads only the target rows; L sets every media row's time origin
  (`comfy/ldm/minimax/model.py:353-473, 649-792`). Each block normalizes its input before attention.
- No guidance branch: one positive pass per step.
- Conds with different payloads never batch together, so timestep-ranged conds keep their own rows and tags; the
  pack's preflight already handles scheduled conds of different text lengths (`preflight.py:170-187`).
- The frozen cache keeps text rows always live (`frozen_video_cache.py:102`), and its cached steps run block parts
  directly, so block hooks do not fire there (`frozen_video_cache.py:513`); its guard does not report this.

### 7.3 Weights across variants

- `context_embedder` (`condition_proj`) and every `token_refiner` tensor differ between the release's `transformer`
  and `transformer_ref`. Every local fl2va checkpoint matches `transformer`, every ref2va checkpoint matches
  `transformer_ref`, and FastH3's `condition_proj` differs from both: three text-path families.
- `condition_proj` is bf16 in every local checkpoint.
- Pika-H3-Flex's Comfy-layout LoRAs carry keys on `token_refiner.blocks.0` and `.1` but none on `condition_proj`.
  `lora_branch.py` applies LoRAs as separate branches, never merged, so a recipe hash, not a weight hash, identifies
  the model.

### 7.4 Where comfy-kitchen enters

It computes the space, nothing more: core's int8 and convrot layouts and their GEMM (`comfy/quant_ops.py`) and the
encoder's RoPE (`comfy/text_encoders/llama.py:653`). Sol, in the owner's fork, runs the DiT's attention, through which
the text rows reach video and audio; `grade` records which attention ran. The refiner runs on core's default attention.

### 7.5 sglang and the release

Both stacks agree on everything in 7.1 and 7.2 that matters here. They differ in encoder precision (bf16 in sglang,
fp32 in ComfyUI) and in image and reference-video resizing, so a vector made in ComfyUI is valid in ComfyUI. The
release ships no encoder or DiT Python; sglang is its serving path.

## 8. Prior art and sources

### 8.1 ComfyUI-Dials and ComfyUI-MAINodes (matlowai)

The first H3 implementation, read in code (`coderef/ComfyUI-MAINodes` at `1880dfc`, `coderef/ComfyUI-Dials` at
`7dd0b7e`). GPL-3.0-or-later and AGPL-3.0, so we port ideas and copy no code.

- **What they do**: encode each pair with core's encode call, project and refine, take a plain mean over every row
  (`coderef/ComfyUI-MAINodes/dials/core.py:422-433`); debias by an uncentred SVD over fixed nuisance templates; holds
  projected out; a DIFFUSION_MODEL wrapper adds strength times the vector to every refined text row
  (`coderef/ComfyUI-MAINodes/dials/core.py:92-102`); a rack summed in float64; step schedules
  (`coderef/ComfyUI-MAINodes/dials/envelopes.py`); per-block strengths as a block patch; clip-time curves by a second
  model pass; picture dials pooling the text rows after the picture; a probe-bank fit with a placebo bank.
- **What we keep**: the mechanism, holds, weights, the float64 sum, step schedules, picture dials, the probe fit.
- **What we change**: the mask (theirs pushes vision rows, labels and dialogue); pairs measured in frames and under
  the prefix (theirs are short t2va phrases); the quality statistic (theirs is the mean cosine of each pair to a mean
  that includes it, which flatters small sets, about 1/sqrt(n) for unrelated pairs, `coderef/ComfyUI-MAINodes/dials/core.py:464`;
  we use mean pairwise cosine); the nulls; the gate.
- **Their lessons** (`coderef/ComfyUI-MAINodes/dials/LESSONS.md`), their claims: pairs are learned literally; equal
  slider numbers are not equal pushes; holds are linear and leak; the first half of the blocks carries the effect; on
  Pika only the first steps matter; some qualities do not come from this point.

### 8.2 Other work, as their claims

- **The owner's Krea 2 work** (krea2-explorations): concept directions with `add`, `amplify`, `subtract` and
  `project_out`; mean pairwise consistency; layer sweeps over Krea 2's 12 fused layers. Krea 2's DiT reads those
  layers through a trained fusion, so it is weak precedent for H3's single layer.
- **The owner's Z-Image work** (llm-dit-experiments): hidden-layer choice and weighted layer blends
  (`src/llm_dit/encoders/z_image.py`); outlier channels at Qwen3-4B's penultimate layer
  (`experiments/research/outlier_dimension_396.md`), variance outliers rather than an offset.
- **vlens** (SauersML, MIT, `coderef/vlens` at `972b14c`): Qwen3-VL-8B states drawn by Qwen-Image-2.1-Turbo. Concept
  steering at layer 18 worked on a blank prompt at one typical state size; one neuron did not. Middle layers draw
  hidden thoughts; for ordinary prompts the trained last layer is the ceiling. Prompt morphing at a layer.
- **The Neuron Persona Library** (neuronpersonas.com) and a related tweet: single MLP neurons of Qwen3 pushed along
  their write direction at a multiple of the median residual norm; a backoff ladder on incoherence.
- **Persona vectors** (Anthropic, 2025; from memory): trait description to contrastive prompts to a mean difference;
  used to steer and to monitor by projection, the source of the meter idea.
- **Steering Language Models with Weight Arithmetic** (Fierro et al. 2025, read in the summary in wassname's
  constitutional_grafting repo): trained contrastive weight directions generalize further than activation steering; a
  bias-only variant sits between. constitutional_grafting's own rule that stacked adapters share one budget, and that
  a teacher model's defaults drift into generated data.
- **EmbeddingGemma 2** (arXiv 2610.37364): a spread-out regularizer against narrow cones; one space for text, video and
  audio.
- **neural-pairing** (SauersML): two LLMs linked through hidden states by a trained cross-attention; write size relative
  to the reader's residual norm; ridge maps at matched tokens.
- **The mean-difference estimator** is widely reported as a strong steering baseline, and plain prompting often beats
  steering; uncited here, from memory, which is one reason the gate exists.

### 8.3 This record's own trail

The earlier draft, two skeptical review passes by a session that did not write it, an experiment proposal, traces of
ComfyUI core's and sglang's H3 paths, a source study of MAINodes and Dials, and a Gemini read are kept locally under
internal/claude/2026-11-04_dials/ (untracked). The corrections they made are applied in place above.

## Appendix A. Docs the code contradicted on 2026-11-04

Not yet corrected; each fix and its `docs/wiki/decisions.md` entry belongs to whoever holds that file.

- `docs/h3_conditioning_end_to_end.md:47`: a video's soundtrack carries an `<Audio j>: ` label in Qwen.
- Same file, `:233`: core splices the vision rows between the start and end tokens; it does not replace image-pad
  positions.
- Same file, `:389-391`: cond rows are at `max(t, 0.999)` and `max(t_a, 1.0)`, not fixed timesteps.
- `docs/wiki/references.md:68`: pins sglang at `803b4fb31c`; the checkout is at `16dd97df61`.
- Drifted line citations in `docs/research/h3_dit_implementations.md` (cites `comfy/model_base.py` at 2208, now 2223),
  `bench/measure_dit_prefix_attention.py:28` (cites `comfy/ldm/minimax/model.py` at 624, now 722) and
  `docs/research/sglang_h3_pipeline.md` (cites the sglang H3 `stages/denoising.py` at 651-654, now 711-713).

## Appendix B. Open questions

- Whether row 0 is an outlier (E2.1).
- How `start_percent` lands on the shipped schedules, and the core keyframe node writing entry 0's keyframes to every
  entry, for E4.2.
- What the four unmatched tensors in the release's native `FL2VA/` and `Ref2VA/` text paths are.
