# Which distill for which shot

last updated: 2026-10-05 (PDMD retired: its row and section reduced to a pointer); 2026-10-01 (PDMD added, from one blind session: one seed, five scenes); 2026-09-26 (written from one seed, two scenes, unblinded)

> **Seed-matched clips are different scenes.** The same seed through each
> distill gives a different scene: different camera angle and framing,
> different subjects, a different palette and mood. Each distill maps noise to
> video its own way, through its training data and objective. Nothing in its
> training keeps the teacher's seed-to-scene mapping, and the distribution-
> matching distills (FlashGen, FastH3) settle on their own likeliest scenes. So
> a seed-matched clip from two distills is two samples, not a controlled pair
> (`docs/method/eval_comparison.md`, step 1). A claim that distill A beats distill B holds
> only when it is judged on something independent of content, across many
> scenes, and blind: artifacts, clones, identity across cuts, following the
> shot plan, the grade against the base. The owner, 2026-09-26: "seed-matched
> clips are different scenes for each lora distill adapter due to training
> data and all sorts of other things."

What each few-step distill of H3 is good and bad at, and why, as far as the
records show. **This rests on one seed and two scenes (diner, subway), judged
unblinded by the owner, plus measurements of those clips.** It is a working
guide, not a verdict. A second seed is queued
(`../bench/flashgen_adherence_seed2_arms.json`), and this page is revised when
it lands. Anything under "why" that is not measured is marked as inference.

Each distill is compared against the undistilled base model at the same seed
and prompt, which is the reference for how H3 itself renders the scene.

The numbers live in the records, not here:
- `../bench/results/2026-09-26_distill_compare_s1.md`: the three-way look, old
  subway prompt;
- `../bench/results/2026-09-26_subway_v2_s1.md`: four models on the rewritten
  subway prompt, with the colour measurement;
- `../bench/results/2026-09-26_distill_tone.md`: brightness and contrast
  against the base.

## At a glance

| distill | good at | bad at | speed at 345 frames |
|---|---|---|---|
| **PDD8** | close-ups and medium shots: detail and colour | fast motion: artifacts, a cloned extra person, mangled text | slowest of the three distills |
| **FlashGen** | motion that stays coherent (zoom-outs); detail; colour closest to the base | following who does what in a multi-person action scene | fastest |
| **FastH3** (contract) | motion; fewer artifacts than PDD8 on the chase; audio | warm, oversaturated, contrasty grade; some coarse texture | between the two |
| base (reference) | lighting and prompt adherence | nothing here; it is the reference | the slowest by far |

The sampler times per model are in the two subway records' `sampler_s`.

## Why they differ: a working model, open for annotation

Written 2026-09-26 as a set of hypotheses, not findings. Each claim is
labelled **measured** (with its record), **code** (read from the source) or
**inference**. A peer session annotates in place, never by rewriting: add a
dated, signed blockquote under the claim (`> vaedude, 2026-09-26: ...`). A
claim that a test refutes keeps its text and gains an annotation saying so.

### Why the same seed gives a different scene

- **Dated note, 2026-10-05: this bullet describes base clips rendered through
  2026-10-05.** The base samples with Euler since then
  (`workflows/h3_config.py::SAMPLING`, owner decision), so a seed-matched base
  clip and distill clip now start from the same noise. Every comparison below
  that pairs a base clip with a distill clip was rendered before the change.
- **Code: the base and the distills do not share starting noise, even at the
  same seed.**
  - Both draw the same starting tensor from the seed (`Noise_RandomNoise`).
  - Every distill steps with Euler, which adds no noise, so that tensor
    carries through the whole render.
  - The base samples with `er_sde` (`comfy/k_diffusion/sampling.py::sample_er_sde`),
    which adds fresh noise after every step but the last. The fresh noise
    comes from a second generator seeded with the same number, so the base
    still reproduces run to run.
  - At `er_sde`'s first step the carry factor on the starting tensor
    (`r_alpha * r`, with the first sigma offset just below 1) is vanishingly
    small, and the fresh draw takes its place. Drive that function's
    arithmetic at the base schedule to see the factors.
  - So a seed-matched base clip and distill clip start from unrelated noise.
    Distills at one seed do share their start with each other.

> vaedude, 2026-09-26: **Confirmed with core's own functions, not only by
> hand.** At the base schedule (`simple`, 16 steps, shift 12) the first sigma
> is offset to 0.9999917, and step 0's carry `r_alpha * r` is 3.94e-13 against
> a fresh-noise coefficient of 0.9945. The carry reaches 0.42 and 0.67 only at
> steps 1 and 2, after the fresh draw has set the start. Two details tighten
> it:
> - `default_noise_sampler` seeds a generator on the latent's own device. On
>   CUDA its draws come from a different stream than `RandomNoise`'s CPU
>   generator at the same seed, and on CPU it adds 1 to the seed.
> - The draws are shaped like the packed AV latent, so a change of canvas or
>   length changes every later draw too.
>
> Nothing in my latent work relied on base and distill sharing noise. It does
> matter for #26: all five arms are base `er_sde` renders at one seed and one
> packed latent shape (reference rows are conditioning, not part of the
> sampled latent). So those arms share the start and every fresh draw, which
> is what makes their pairs controlled.
- **Inference: the first steps, at noise near 1, settle the layout**: camera,
  who stands where, the overall grade.
  - Between the base and a distill, the start differs, so the scene would
    differ even with identical weights.
  - Between two distills, the start is the same, and the weights alone move
    the scene. Every distill changes what the model predicts from that noise.

### PDD8: the teacher's path in coarse averaged steps

- **Code** (`pdd.md` "What it is"):
  - PDD keeps a 32-point grid and one output head per interval of it.
  - Each sampling step fuses a contiguous block of those heads into one
    output. That output is the block's mean velocity, weighted by each
    interval's step size.
  - PDD8 fuses 4 heads per step, so each step covers an eighth of the grid by
    index. In noise terms the steps are very uneven: under shift 12 the last
    step is by far the widest, and it is wider at 4 steps than at 8 or 16. The
    schedules come from `pdd_lora.emit_sigmas(12, 32, width)`, and
    `pdd.md` lists them.
- **Code** (`pdd.md`, the section on sub-steps): every head in a block reads
  the same hidden state. A fused step is therefore exactly one Euler step at
  the mean velocity, and nothing inside the block reacts to what the frame
  becomes.

> vaedude, 2026-09-26: Agreed from core. `_pdd_head` fuses the block's linear
> heads by their dt weights and applies the result to one hidden state `h`,
> which equals the dt-weighted mean of the heads' outputs on that state.
- **Measured** (`docs/method/evidence.md`, "Settled about H3"): PDD quality is governed by
  how coarse the schedule's tail is, not by the evaluation count.
- **Measured, with a caveat** (`pdd.md`, the partition table dated
  2026-08-28): PDD8 already sits far from PDD's own 32-step path (width 1, no
  fusion). The table was taken at 39 frames with Sol inert, so treat it as a
  direction, not a size.
- **Inference: the averaging explains PDD's motion failures.**
  - For a still or slow scene the velocity barely changes across a block, so
    the mean is a good stand-in. Hence the good close-ups, detail and colour.
  - When something moves, the velocity inside a block changes a lot, and the
    mean of two positions can put a person in both. The clone at 1 s on subway
    would be that: an averaging artifact, not a prompt problem. Mangled text
    on moving signage would be the same thing at fine scale.
  - The widest blocks come last, so fine detail on moving things is decided in
    one or two coarse averaged steps. That fits artifact severity tracking
    inter-frame delta (the PDD8 "Why" below).

> Claude, 2026-09-27: **The subway "clone" was not a clone.** The base,
> rendered from the same noise, draws the same two people at 1.1-1.3 s: the
> black-jacket agent and the grey-hoodie suspect the prompt names
> (`../bench/results/2026-09-27_clone_base_control.md`). PDD8 decides that
> composition at step 1, at 4, 6 and 8 steps, merged or exact. What may read
> as a clone is PDD8's overlap at 1.1 s, one figure passing in front of the
> other, and that is the owner's call. So this case offers no support for the
> averaging mechanism, nor for vaedude's late-block hedging. Neither is
> refuted for PDD's other motion artifacts; they have simply not been shown
> on one.

> vaedude, 2026-09-26: **The pattern fits, but I would move the mechanism.**
> An exact block-mean velocity, applied as one Euler step, lands exactly on
> the teacher's end of the block: mean velocity along a path times dt is the
> displacement. So averaging alone does not put a person in two places. The
> error has to come from the heads predicting that mean from the block's
> *start* state. Where the start state does not decide where a moving person
> ends up, a readout trained toward the teacher's mean hedges between the
> possible outcomes, and a hedge between two positions is a double image.
> That predicts the same thing (worse with wider blocks and larger motion), but
> puts the cause in prediction under uncertainty, not in averaging as such.
> *Inference; I have not read PDD's training loss.* The frozen-row probe and
> its pass-1 latents bear on neither: they test the refine pass, not PDD's
> steps. What would separate the two readings is the per-step capture under
> "Tests" (my note there), which localises where the error enters.
- **Inference, not yet read in the paper:** PDD is the only one of the three
  that tries to reproduce the teacher's path. It was the closest to the base
  in every scene measured (`2026-09-26_distill_compare_s1.md`), but that
  comparison started from unrelated noise (above). So the closeness is
  shared prompt and grade, not a shared path, and the shipped base is not
  PDD's target.

### FlashGen and FastH3: matching the distribution, not the path

- **Code / release:** both are distribution-matching distills. FlashGen is
  VSD (`docs/research/2026-09-26_flashgen.md`) and FastH3 is DMD2 with VSA sparse
  attention (`h3_config.FASTH3_CONTRACT_VSA`). They are trained so their few
  steps land on outputs the teacher would plausibly produce. They are not
  trained to land where the teacher would from the same noise.
- **Inference: what that would explain.**
  - **Motion stays coherent:** no averaged velocities, so no ghosting from
    that cause.
  - **The look gets bolder:** likely outputs are high-contrast, saturated and
    conventionally framed. That would give the grade in
    `2026-09-26_distill_tone.md` and "a very different scene".
  - **Adherence weakens:** unusual instructions (who chases whom, which way
    they run) are exactly what "likeliest" tends to override.
  - **FastH3's grainy texture:** VSA lets each video cube attend in full to
    only the fraction of cubes `FASTH3_CONTRACT_VSA` keeps, with a coarse
    summary covering the rest. Fine texture is then partly invented locally.

> vaedude, 2026-09-26: Consistent with the known mode-seeking pull of VSD and
> DMD objectives: sharper, more contrast, less variety. On FastH3's texture,
> two things from my side.
> - **The decoder is ruled out.** The FastH3 contract subway clip (16:11)
>   decoded through the fp16 VAE, before the INT8 switch, and the two decoders
>   measured 54.9 dB apart anyway
>   (`../bench/results/2026-09-26_vae_decoders_345f.md`).
> - **A cheap test of the VSA reading, no card.** VSA's cube is 4x4x4 tokens
>   (`comfy_extras/nodes_sparse_attention.py`, `VSA_CUBE`). A token is a 2x2
>   patch of a 16x latent, so a cube spans 128x128 pixels and four latent
>   frames. If the coarse branch invents the texture, FastH3's frames should
>   show structure on a 128-pixel period (a spectral peak, or stronger
>   gradients on 128-pixel boundaries) that the base and FlashGen do not. The
>   "ps2 polygons" would be that grid.
>
> vaedude, 2026-09-26, later: **No cube grid, on a first look.** On the
> subway clips, edge strength on 128-pixel boundaries over the other 32-pixel
> boundaries is 1.004 (x) and 1.047 (y) for FastH3, against 1.025 and 1.047 for
> the base, 1.033 and 1.077 for FlashGen, and 0.980 and 1.030 for PDD8
> (`../data/results/2026-09-26_block_period_subway.json`,
> `../bench/measure_block_period.py`). This was one scene, with each clip a
> different draw. So the texture is not a visible grid; if VSA explains it,
> the effect is flat, low-texture facets. That needs a texture-energy measure
> on the distill run's same-scene arms.

### Tests that would move this section

- **PDD's schedule, on motion scenes** (the owner, 2026-09-26: "we should
  absolutely do the faster path"). The same file runs at several schedules:
  - 4 steps, width 8, the coarsest tail;
  - 6 steps (`PDD_MANUAL_SIGMAS`): 8's tail at fewer evaluations;
  - 8 steps, width 4;
  - 16 steps, width 2, the finest tail.

  If the clones are averaging artifacts, they get worse as the tail gets
  coarser. If the tail matters rather than the count (`docs/method/evidence.md`), 6 steps
  should match 8, not sit between 4 and 8. Judged blind across scenes, as the
  box at the top requires. The unjudged `C2_pdd4_nosol` / `C2_pdd8_nosol`
  pair (`docs/research/pdd/queued_arms.md`, three seeds each) is a zero-card-time
  first look.
- **Base on Euler over PDD's grid, same seeds.** This is the only base render
  that shares the distills' starting noise, so it is the controlled pair for
  every distill, not only PDD. If PDD8 lands much closer to it than FlashGen
  or FastH3 do, PDD is following its teacher. If all three stay far, the
  scene difference is the weights.

> vaedude, 2026-09-26: What I would capture for these two tests:
> - **Saved latents on every arm.** Only `h3_text_to_video_pdd_savelat` exists
>   today. I can add twins for the 4-step, 6-step and 16-step PDD graphs and for
>   a base-on-Euler graph from the same generator switch; that is a small CPU
>   change.
> - **The base on Euler at 32 steps, not 16.** That is width 1 on PDD's own
>   grid, the path PDD's heads were distilled from. Then each schedule's
>   latent distance from it, per latent frame, measures path-following
>   directly: no decode and no judge. It also shows where each schedule leaves
>   the path, which should be the high-motion latent frames if either
>   inference above is right.
> - **Per-step predictions for one or two motion renders**, to see which
>   block the clone first appears in. `x0` at each step, about 40 MB per step,
>   through a small observer node like the mask probe (a node change, so a
>   restart). Set against the Euler-32 base's `x0` at the same sigma, it shows
>   which block first lands off the teacher's path, and whether the double
>   image is already in that block's prediction. Both readings put the error in
>   the fused step's prediction, so this localises the cause rather than
>   choosing between them.
> - **Telemetry armed.** Every PDD schedule attaches the same 308 merged
>   patches, so per-step cost is comparable across schedules, and their
>   re-patching under dynamic VRAM shows up in the records.
- **FlashGen strength and module arms** in the 2026-09-26 run
  (`../bench/distill_run_arms.json`): whether the grade and adherence move
  with the LoRA's strength, or without its adaln.

## PDD8

**Good.** The owner, 2026-09-26: "looks great when its closeups or medium
shots - lots of detail and good color", "seems to do great at low
movement/deltas". On diner, fine apart from the zoom-out.

**Bad.** Motion. Subway on the old prompt: "super artifacty... text is
mangled. people disappear", and the pursuer cloned. On the rewritten prompt:
"clones a guy out of nowhere right at the 1s mark... way less artifacty than
before". Diner's zoom-out: "a little artifacty".

**Why.**
- **Measured, 2026-08-28:** artifact severity under PDD tracks inter-frame
  delta (+0.676; `../bench/measure_clip_delta.py`, docstring). Big motion means
  big frame-to-frame change, and that is where PDD breaks.
- **Inference:** PDD decodes each step's whole interval from one forward pass
  through its head bank (`pdd.md`), which is more to get right when the
  frame changes a lot between steps.
- **Measured tone:** PDD8 brightens and warms the picture against the base, on
  subway as warm as FastH3 in places (`2026-09-26_subway_v2_s1.md`).

## FlashGen

Shipped as `h3_text_to_video_flashgen`: rank 64, applied at the call. What it
is, and why it is built that way: `docs/research/2026-09-26_flashgen.md`.

**Good.** Diner's zoom-out "looks good, keeps coherency". It "complements pdd
well". Detail "looks good". Its colour stays the closest of the three to the
base's (`2026-09-26_subway_v2_s1.md`).

**Bad.**
- **Following instructions in a multi-person action scene.** On the rewritten
  subway prompt, "the suspect runs right past him. then it becomes unclear who
  is chasing who as they run down the escalator", while the base followed the
  same prompt at the same seed.
- **Its own composition:** "a very different scene from all the others".
- **More contrast than the base.** Less so than FastH3
  (`2026-09-26_distill_tone.md`).

**Why.**
- **Not our sampling setup.** Sigmas, Euler, 4 steps, no CFG and exact LoRA
  application all match vllm-omni, the engine FlashGen ships for
  (`docs/research/2026-09-26_flashgen.md`).
- **Candidates, each with an arm queued** (`../bench/flashgen_adherence_arms.json`):
  - its trained length: 5.2 s, where we render 14.4 s in three shots;
  - dense attention in its recipe, where we run Sol;
  - the one seed.
- **Inference:** distribution-matching distillation (FlashGen's VSD) favours
  the most likely outputs, which can mean a different composition from the
  base and weaker adherence to unusual instructions.

## FastH3 V2

Run to FastVideo's own contract (`h3_probe_t2v_fasth3_8step_contract`), not
ComfyUI's template: see `../bench/results/2026-09-26_fasth3_contract_s1.md`.

**Good.** "much better audio and less artifacty than pdd". Motion holds. The
owner: "a tossup between this and pdd8", and it is faster than PDD8 at this
length.

**Bad.**
- **The grade:** "more washed out / contrasty than flashgen", "contrasty-color
  grading issues (look at the stairs at the 9s mark - orangeish)... maybe its
  over saturated".
- **Texture:** "stairs at the 10s mark look like low res ps2 polygons".
- **Clones:** a cloned pursuer around 5 s on the old subway prompt.

**Why.**
- **Measured:** the brightest highlights and steepest contrast of the three,
  and on subway the warmest and most saturated overall
  (`2026-09-26_distill_tone.md`, `2026-09-26_subway_v2_s1.md`). "Washed out"
  here is bright, near-clipped highlights on a steep curve, not faded colour.
- **Inference:** DMD2, like FlashGen's VSD, favours the most likely outputs,
  which pushes toward a punchier grade.
- **Inference:** the "ps2 polygons" texture may come from VSA's sparse
  attention. It keeps 20% of video cubes, as trained, and a coarse branch
  covers the rest.

## PDMD (retired 2026-10-05)

The owner retired PDMD on 2026-10-05 ("i didnt see anything from it worth
keeping"), four days after parking it. Its graphs, constants, check cases and
research note are gone from the tree and live in git. What was tried, and
what the owner saw, is one blind session against FlashGen:
`../data/results/2026-10-01_2026-10-01_pdmd_vs_flashgen_verdict.json`.
Retired, not refuted (`docs/state/roadmap.md`, "Closed lanes").

## Rules of thumb, provisional

- **Close-ups, medium shots, little motion:** PDD8, or **PDD6** where the shot
  is a close-up with little to keep track of.
  - PDD6 (`workflows/h3_text_to_video_pdd_manual_sigmas_api.json`) runs
    `h3_config.PDD_MANUAL_EVALS` evaluations where PDD8 runs 8.
  - The owner, 2026-09-27, after PDD6 lost all three motion scenes to PDD8:
    "closeups like that are fine for pdd6 - low motion, not a ton of stuff to
    keep track of" (ow-fd-11). It is not the default.
- **PDD8 with signs or on-screen text (t2v):** finish with full FlashGen from
  sigma 0.8 (`workflows/distill_experiments/h3_probe_t2v_step_switch_pdd8_flashgen_h080_api.json`).
  - It fixed sign text PDD8 alone garbled, and the owner could not otherwise
    tell it from PDD8 (ow-fd-08, `../bench/results/2026-09-27_finisher_grid.md`).
  - Warm, it costs no sampling time: its evaluations replace PDD8's own.
    Loading the FlashGen LoRA is a one-off per server.
  - On i2v the 2026-09-27 review said no: its brightening against the first
    frame "looked off" (ow-fd-09). *That is no longer what ships (corrected
    2026-10-05): the finish on a first-frame clip is one of the three daily
    graphs, `workflows/daily/h3_i2v_pdd8_flashgen_finish_api.json`, named in
    `workflows/h3_config.py::DAILY_GRAPHS` since 2026-10-03. The review's
    remark stands as what was seen that day.*
  - Whether it becomes the t2v default is the owner's call.
- **Fast motion or a moving camera:**
  - FlashGen, if the action is simple enough that roles cannot swap;
  - FastH3, if the colour grade is acceptable or will be corrected.
- **A multi-person action scene where who-does-what matters:** the base model,
  until the FlashGen adherence arms say otherwise.
- **Speed first:** FlashGen.

## What would change this page

- The second seed (queued): whether each pattern holds on another draw.
- The FlashGen adherence arms (queued): which of length, attention or seed
  explains its adherence.
- A tone correction toward the base, and switching distills by denoising step
  (`docs/state/next_steps.md`, the idea saved 2026-09-26): either could move a
  distill out of its "bad at" column.
