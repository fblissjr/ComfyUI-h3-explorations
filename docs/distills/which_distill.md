# Which distill for which shot

What each few-step distill of H3 is good and bad at, and why, as far as the records show. **It rests on one seed and
two scenes (diner, subway), judged unblinded by the owner, plus measurements of those clips.** It is a working
guide, not a verdict. A second seed is queued (`bench/flashgen_adherence_seed2_arms.json`) and this page is revised
when it lands. Each distill is compared against the undistilled base at the same seed and prompt.

> **Seed-matched clips are different scenes.** The same seed through each distill gives a different camera, framing,
> subjects, palette and mood: each distill maps noise to video its own way, and the distribution-matching distills
> (FlashGen, FastH3) settle on their own likeliest scenes. So a seed-matched pair is two samples, not a controlled
> pair (`docs/method/eval_comparison.md`, step 1). A claim that distill A beats distill B holds only when judged blind,
> across many scenes, on something independent of content: artifacts, clones, identity across cuts, following the
> shot plan, the grade against the base (the owner, 2026-09-26).

The records: `bench/results/2026-09-26_distill_compare_s1.md` (the three-way look, old subway prompt);
`bench/results/2026-09-26_subway_v2_s1.md` (four models on the rewritten subway prompt, with the colour measurement
and each model's `sampler_s`); `bench/results/2026-09-26_distill_tone.md` (brightness and contrast against the base).

## At a glance

| distill | good at | bad at | speed at 345 frames |
|---|---|---|---|
| **PDD8** | close-ups and medium shots: detail and colour | fast motion: artifacts, mangled moving text | slowest of the three distills |
| **FlashGen** | motion that stays coherent (zoom-outs); detail; colour closest to the base | who does what in a multi-person action scene | fastest |
| **FastH3** (contract) | motion; fewer artifacts than PDD8 on the chase; audio | warm, oversaturated, contrasty grade; some coarse texture | between the two |
| base (reference) | lighting and prompt adherence | nothing here; it is the reference | slowest by far |

## Rules of thumb, provisional

| the shot | use | why, and the record |
|---|---|---|
| close-up or medium shot, little motion | PDD8; **PDD6** (`workflows/h3_text_to_video_pdd_manual_sigmas_api.json`, `h3_config.PDD_MANUAL_EVALS` evaluations) for a close-up with little to track | PDD6 lost all three motion scenes to PDD8; the owner accepts it for low-motion close-ups (ow-fd-11, 2026-09-27). Not the default |
| PDD8 with signs or on-screen text | finish with full FlashGen from sigma 0.8 (`workflows/distill_experiments/h3_probe_t2v_step_switch_pdd8_flashgen_h080_api.json`) | fixed sign text PDD8 garbled, otherwise indistinguishable from PDD8 (ow-fd-08, `bench/results/2026-09-27_finisher_grid.md`). Warm, it costs no sampling time; the FlashGen load is a one-off per server. Whether it becomes the t2v default is the owner's call |
| a first-frame clip | the PDD8-to-FlashGen finish ships as a daily graph, `workflows/daily/h3_i2v_pdd8_flashgen_finish_api.json` (`h3_config.DAILY_GRAPHS`) | the 2026-09-27 review saw its brightening against the first frame as off (ow-fd-09); that is what was seen that day |
| fast motion or a moving camera | FlashGen if the action is simple enough that roles cannot swap; FastH3 if its grade is acceptable or will be corrected | the motion columns above |
| a multi-person action where who does what matters | the base model | until the FlashGen adherence arms say otherwise |
| speed first | FlashGen | |

## PDD8

- **Good:** close-ups and medium shots, with lots of detail and good colour, and low movement; diner was fine apart
  from its zoom-out.
- **Bad:** motion. Subway on the old prompt was heavily artifacted, with mangled text and people disappearing; on the
  rewritten prompt, much less artifacted, with an apparent extra person at 1 s; the diner zoom-out a little artifacted.
- **The subway "clone" is not a clone** (2026-09-27). The base, from the same noise, draws the same two people at
  1.1-1.3 s, the agent and the suspect the prompt names (`bench/results/2026-09-27_clone_base_control.md`); PDD8
  decides that composition at step 1, at 4, 6 and 8 steps. What may read as a clone is one figure passing in front of
  the other, which is the owner's call.
- **Measured:** artifact severity under PDD tracks inter-frame delta (+0.676; `bench/measure_clip_delta.py`,
  docstring). PDD8 brightens and warms the picture against the base, on subway as warm as FastH3 in places
  (`2026-09-26_subway_v2_s1.md`).

## FlashGen

Ships as `h3_text_to_video_flashgen`: rank 64, applied at the call. What it is and why it is built that way:
`docs/research/2026-09-26_flashgen.md`.

- **Good:** the diner zoom-out keeps its coherence; it complements PDD; detail is good; its colour stays the closest
  of the three to the base (`2026-09-26_subway_v2_s1.md`).
- **Bad:** on the rewritten subway prompt the suspect runs past the agent and then it is unclear who chases whom down
  the escalator, where the base followed the same prompt at the same seed; its own composition, a very different
  scene from the others; more contrast than the base, less than FastH3 (`2026-09-26_distill_tone.md`).
- **Not our sampling setup:** sigmas, Euler, 4 steps, no CFG and exact LoRA application all match vllm-omni, the engine
  FlashGen ships for. Candidates for the adherence loss, each with an arm queued (`bench/flashgen_adherence_arms.json`):
  its trained length (5.2 s, where we render 14.4 s in three shots), dense attention in its recipe where we run Sol,
  and the one seed.

## FastH3 V2

Run to FastVideo's own contract (`h3_probe_t2v_fasth3_8step_contract`), not ComfyUI's template:
`bench/results/2026-09-26_fasth3_contract_s1.md`.

- **Good:** much better audio and fewer artifacts than PDD; motion holds; a tossup with PDD8 for the owner, and faster
  than PDD8 at this length.
- **Bad:** the grade (washed out and contrasty, an orange cast on the stairs at 9 s, perhaps oversaturated); texture
  (the stairs at 10 s look like low-resolution polygons); a cloned pursuer around 5 s on the old subway prompt.
- **Measured:** the brightest highlights and steepest contrast of the three, and on subway the warmest and most
  saturated (`2026-09-26_distill_tone.md`, `2026-09-26_subway_v2_s1.md`). "Washed out" here is bright, near-clipped
  highlights on a steep curve, not faded colour. The decoder is ruled out: the contract subway clip decoded through
  the fp16 VAE and the two decoders measured 54.9 dB apart (`bench/results/2026-09-26_vae_decoders_345f.md`). No
  128-pixel cube grid shows: edge strength on 128-pixel over other 32-pixel boundaries is 1.004 (x) and 1.047 (y) for
  FastH3, against 1.025 / 1.047 for the base, 1.033 / 1.077 for FlashGen and 0.980 / 1.030 for PDD8, one scene
  (`data/results/2026-09-26_block_period_subway.json`, `bench/measure_block_period.py`; vaedude, 2026-09-26).

## PDMD (retired 2026-10-05)

The owner retired PDMD on 2026-10-05, four days after parking it, seeing nothing worth keeping. Its graphs, constants,
check cases and research note are gone from the tree and live in git. What was tried, one blind session against
FlashGen: `data/results/2026-10-01_2026-10-01_pdmd_vs_flashgen_verdict.json`. Retired, not refuted
(`docs/state/roadmap.md`, "Closed lanes").

## Why they differ: a working model

Each claim is labelled **measured** (with its record), **code** (read from source) or **inference**. A peer annotates
with a dated, signed note under the claim; a refuted claim keeps its text and gains a note saying so.

### Why the same seed gives a different scene

- **Code: since 2026-10-05 the base samples with Euler** (`workflows/h3_config.py::SAMPLING`, owner decision), so a
  seed-matched base clip and distill clip now start from the same noise. **Every base-against-distill comparison on
  this page was rendered before that, on `er_sde`, from unrelated noise.** `er_sde` adds fresh noise after every step
  but the last from a second generator seeded with the same number; at its first step the carry on the starting
  tensor is vanishingly small and the fresh draw takes its place. Confirmed with core's own functions (vaedude,
  2026-09-26): at the base schedule (`simple`, 16 steps, shift 12) the first sigma is 0.9999917 and step 0's carry
  `r_alpha * r` is 3.94e-13 against a fresh-noise coefficient of 0.9945. `default_noise_sampler` draws on the latent's
  own device (a different stream from `RandomNoise`'s CPU generator on CUDA; seed + 1 on CPU), shaped like the packed
  AV latent, so a change of canvas or length changes every later draw. Distills, all on Euler, share their start with
  each other. Base `er_sde` arms at one seed and one packed shape (open experiment #26) share every draw, which is what
  made those pairs controlled.
- **Inference:** the first steps, at noise near 1, settle the layout (camera, who stands where, the grade). Between two
  distills the start is the same, so the weights alone move the scene.

### PDD8: the teacher's path in coarse averaged steps

- **Code** ([`pdd.md`](pdd.md), "What it is"): each step fuses a contiguous block of the 32 heads into one `dt`-weighted
  mean velocity applied as one Euler step, and every head in a block reads the same hidden state (`_pdd_head`), so
  nothing inside a block reacts to what the frame becomes. PDD8 fuses 4 heads per step; under shift 12 the last step is
  by far the widest, and wider at 4 steps than at 8 or 16 (`pdd_lora.emit_sigmas(12, 32, width)`).
- **Measured** (`docs/method/evidence.md`, "Settled about H3"): PDD quality is governed by how coarse the schedule's
  tail is, not by the evaluation count. PDD8 sits far from PDD's own 32-step path, taken at 39 frames with Sol inert,
  so a direction and not a size ([`pdd.md`](pdd.md), "What is measured").
- **Inference, and where the error enters is open.** One reading: when something moves, the velocity inside a block
  changes a lot and the mean misplaces it, and the widest blocks come last, deciding fine moving detail in one or two
  coarse steps. vaedude's reading (2026-09-26): an exact block-mean velocity applied as one Euler step lands on the
  teacher's block end, so averaging alone cannot double a person; the error is the heads predicting that mean from the
  block's start state, hedging between outcomes the start does not decide, and a hedge between two positions is a
  double image. Both predict worse with wider blocks and larger motion. The subway case supports neither (above); the
  per-step capture below would localise it.
- **Inference:** PDD is the only one of the three that tries to reproduce the teacher's path. It was closest to the base
  in every scene measured (`2026-09-26_distill_compare_s1.md`), but from unrelated noise, so the closeness is shared
  prompt and grade, not a shared path.

### FlashGen and FastH3: matching the distribution, not the path

- **Code / release:** both are distribution-matching distills: FlashGen is VSD (`docs/research/2026-09-26_flashgen.md`),
  FastH3 is DMD2 with VSA sparse attention (`h3_config.FASTH3_CONTRACT_VSA`). Their few steps land on outputs the
  teacher would plausibly produce, not where it would land from the same noise.
- **Inference:** no averaged velocities, so no ghosting from that cause; likely outputs are high-contrast, saturated and
  conventionally framed, which fits the grade and the different scene; unusual instructions (who chases whom) are what
  "likeliest" overrides; consistent with the known mode-seeking pull of VSD and DMD (vaedude). FastH3's texture may come
  from VSA, which lets each video cube attend in full to only the fraction of cubes `FASTH3_CONTRACT_VSA` keeps (a
  4x4x4-token cube spans 128x128 pixels and four latent frames), with a coarse summary for the rest; no visible grid
  showed (above), so if VSA explains it the effect is flat, low-texture facets, which needs a texture-energy measure on
  the distill run's same-scene arms.

### Tests that would move this section

- **PDD's schedule on motion scenes** (owner, 2026-09-26): the same file at 4 steps (width 8), 6 (`PDD_MANUAL_SIGMAS`,
  8's tail at fewer evaluations), 8 (width 4) and 16 (width 2), judged blind across scenes. If the tail matters rather
  than the count, 6 should match 8. The unjudged `C2_pdd4_nosol` / `C2_pdd8_nosol` pair
  (`docs/research/pdd/queued_arms.md`, three seeds each) is a zero-card-time first look.
- **The base on Euler over PDD's grid, same seeds**, the controlled pair for every distill. vaedude's capture plan: saved
  latents on every arm (only `h3_text_to_video_pdd_savelat` exists; twins for the 4-, 6- and 16-step graphs and a
  base-on-Euler graph are a small CPU change); the base on Euler at 32 steps (`h3_probe_t2v_base_euler32`, width 1 on
  PDD's grid), so each schedule's latent distance from it per latent frame measures path-following with no decode and no
  judge; per-step `x0` for one or two motion renders (about 40 MB a step, through a small observer node, so a restart)
  against the Euler-32 base at the same sigma, to see which block first leaves the path; telemetry armed (every PDD
  schedule attaches the same 308 merged patches).
- **FlashGen strength and module arms** in the 2026-09-26 run (`bench/distill_run_arms.json`): whether the grade and
  adherence move with the LoRA's strength, or without its adaln.

## What would change this page

- The second seed (queued): whether each pattern holds on another draw.
- The FlashGen adherence arms (queued): which of length, attention or seed explains its adherence.
- A tone correction toward the base, and switching distills by denoising step (`docs/state/next_steps.md`, the idea
  saved 2026-09-26; `docs/research/2026-09-26_distill_routing.md`): either could move a distill out of its "bad at"
  column.
