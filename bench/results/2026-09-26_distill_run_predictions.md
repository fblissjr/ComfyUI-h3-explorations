# Predictions for the 2026-09-26 distill run and its follow-up

Registered before any distill arm of the run was looked at. When this was
committed the run was on its base arms (`bench/distill_run_arms.json`, rows in
`2026-09-26_distill_run.jsonl`). A prediction is kept as written. When results
land, each gets a dated verdict line under it: held, failed, or could not tell.
The reasoning behind each is `docs/distills/which_distill.md`, "Why they differ: a
working model".

Two facts constrain what the run can say:
- **The base and the distills do not share starting noise at one seed**
  (`er_sde` replaces it at step 1; `docs/distills/which_distill.md`). A tone or layout
  difference between the base and a distill in this run is two unrelated
  samples, not a controlled pair.
- **The three distills do share starting noise** (all Euler, no injected
  noise), so comparisons among them are the fairest this run offers. They
  still differ in weights, so each is its own scene.

**Method note, 2026-09-27, before the verdicts are written.** Final-latent
distance at one seed measures trajectory divergence, not effect size.
FlashGen with no adaln, whose modulation change is about 0.1%
(`2026-09-26_flashgen_weights.md`), lands 0.57-0.71 from full FlashGen, and
dense attention 0.51 (`2026-09-26_followup.jsonl` latents).

So a verdict that rests on a magnitude uses the tone, chroma and temporal
measures and the owner's eye, not latent distance: that covers P3, F4,
vaedude's P1 and P5, and FT1. An event survives the divergence and still
reads from latents, for example the x0 two-figure read.

## The owner

In the owner's words, 2026-09-26:
- FastH3 "perceptively looked like it was lower resolution and then some parts
  of pdd did", and "my eyes see it as like... low bitrate streaming video".
- "its always in shows like the witcher on netflix where dark/night scenes
  looked so blocky and splotchy" -- "of what you could look for. thats my
  theory."
- On the encode: "i dont think its the video this was just an example."

**O1, as a testable statement:** the distills render dark and night regions
blocky and splotchy, the way a starved streaming encode does, and the base
does not. It shows most in the look family's low-key anchor and night
variants. By the owner's reading, FastH3 is worst, and parts of PDD show it.
Fails if the base's shadows are as blocky as the distills' on the same look
variants.

*Verdict, 2026-09-27: not supported on the one clip tested* (see F8). PDD8's
own decoded darks are not blocky. The blocking the owner describes appears
when the save node encodes to 8-bit h264. The owner's side note, "i dont
think its the video this was just an example", is contradicted on this clip.

**O2, the owner, 2026-09-26:** "its probably way overfit on too little data.
thats why prompt adherence sucks and going OOD is easy". FlashGen's card says it
was trained data-free (VSD against the teacher's own samples) at 1344x768 and
5.2 s only, so "too little data" reads here as too narrow a training
distribution: its prompt set, one length, one canvas. As testable statements:
- **Seed collapse:** across three seeds of one prompt, FlashGen's clips vary
  less than PDD8's (latent distance between seeds, and the layout by eye).
- **Length:** FlashGen follows the subway roles better at its trained 5 s
  (`subway_chase_short`) than at 14.4 s.
- Fails if FlashGen's seeds vary as much as PDD8's.

*2026-09-26, the owner, before any result: seed spread is a poor test ("the
initial noise could still have a heavy influence on what it turns into"). The
signature is prompt specificity: "prompting something that fits within its
guided path will look great on flashgen. going outside that path and weird
shit happens". The seed arms were cancelled unrendered. The test is the
specificity ladder instead (`t2va_spec_typical`, `_specific`, `_unusual`, at
the trained 5 s): FlashGen's quality holds on the typical rung and falls on
the unusual one, more than PDD8's or FastH3's does.*

## Claude (this session)

### Tonight's run

- **P1. FlashGen follows fewer checklist beats than the base.**
  - The gap is largest on beats about who does what to whom and which way
    things move: slapstick, kpop, samurai, courtroom.
  - It is smallest on scenes with one subject or little motion: radio_drama,
    post_office, noodle_bar.
  - Why: a distribution-matching distill favours likely outputs, and role and
    direction instructions are where "likely" and "asked for" part.
  - Fails if FlashGen matches the base on the multi-person beats across most
    of those scenes.
- **P2. FlashGen dense does not fix its adherence.**
  - Dense and Sol follow about the same number of beats on the four motion
    scenes.
  - Why: the adherence loss is in the weights, not in the attention sparsity.
  - Fails if dense follows clearly more beats on three or more of the four.
- **P3. FlashGen's grade moves with its adaln update and its strength, and
  its adherence does not.**
  - Without adaln: less contrast and brightness than full FlashGen, and less
    coherent detail. The few-step dynamics live partly in the timestep
    modulation.
  - At strength 0.8: grade slightly toward the base, detail softer.
  - At 1.2: more contrast and saturation, possibly artifacts.
  - Beats followed stay about the same across the three.
  - Fails if removing adaln leaves the grade unchanged.

  *Verdict, 2026-09-27: failed on adaln, half held on strength.* Tone from
  `2026-09-27_flashgen_variants_tone.json`, on courtroom and kpop.
  - **Adaln (failed):** without adaln the grade matches full FlashGen within
    noise (contrast, haze, saturation, detail). FlashGen's adaln change is
    about 0.1% of modulation.
  - **Strength (half held):** on courtroom 1.2 raises contrast (0.280
    against 0.258) and saturation and lowers haze, and 0.8 goes the other
    way. Kpop barely moves.
  - The haze does not grow with strength: more FlashGen means less haze.
  - Dense attention matches Sol on tone (kpop, samurai). Adherence (P2) is
    for the owner's checklist read.
- **P4. PDD8 clones or doubles people on the motion scenes, not on the still
  ones.**
  - Motion scenes: slapstick, kpop, samurai, rooftop_pov.
  - Its artifact severity ranks with each scene's inter-frame delta
    (`measure_clip_delta.py`).
  - Its detail and colour are best on radio_drama, post_office, courtroom.
- **P5. Among the distills, FastH3 is the most saturated and warmest on most
  scenes, FlashGen next, and PDD8 the least.**
  - Measured by `measure_clip_tone.py`.
  - Not claimed against the base (see above).

  *Verdict, 2026-09-27: half held.* From the VAE session's
  `2026-09-26_distill_signatures.md`, 13 matched scenes from the follow-up
  rerun.
  - **Held:** FastH3 is the most saturated (highest on 11/13; chroma highest
    on 13/13) and the warmest (11/13).
  - **Failed:** FlashGen, not PDD8, is the least saturated (lowest on 11/13)
    and the least warm (11/13). The medians run FastH3 0.387 > PDD8 0.364 >
    FlashGen 0.305 on saturation, and FastH3 0.102 > PDD8 0.092 > FlashGen
    0.075 on warmth.
  - PDD8 has the lowest contrast (12/13) and the dimmest highlights (11/13),
    which may be what reads as "least saturated" by eye. That is a question
    for the owner, not a finding.
  - *The owner's answer, 2026-09-27, relayed by the VAE session:* asked
    whether PDD8 reads less colourful, or flatter and dimmer, the owner said
    "The latter usually - but usually naturally so". So the owner's "pdd
    least" was contrast and level, not colour, and the owner does not read
    that grade as a defect.
- **P6. FastH3 shows more temporal texture instability than FlashGen on the
  same scene.**

  *Verdict, 2026-09-27: not supported.* Boil on the four looks, FastH3
  against FlashGen: anchor 0.656 against 0.666, noir 0.633 against 0.494,
  neon 0.536 against 0.593, anime 0.467 against 0.476. That is mixed and
  mostly level, and the VAE session's subway first look had FastH3 boiling
  least. The owner's "low bitrate" read is not a temporal instability these
  measures see. This is the owner's "low bitrate streaming video". It is
  graded once the motion-compensated measure exists. The per-frame
  resolution measures do not separate them.
- **P7. Step-switch (FlashGen layout, then PDD detail) clones less than PDD8
  on its four scenes, and keeps more detail than FlashGen.** A visible
  handoff artifact is possible and would count against the route.
- **P8. On the five-second subway twin, FlashGen follows the roles better
  than on the 14-second prompt, and still worse than the base.**

### The follow-up batch (after the run and a restart)

- **F1. The base on Euler at 16 steps is a different scene from the shipped
  base at the same seed.**
  - Its tone is punchier than the `er_sde` base.
  - If its tone sits where the distills do, most of "every distill shifts
    the grade" is the base's sampler, not distillation.
  - Fails if base-Euler's tone matches base-`er_sde`'s.
- **F2. PDD8 lands closer to the base on Euler at 32 steps (the teacher's
  grid) than FlashGen or FastH3 do.** Measured per latent frame, with no
  decode.
- **F3. PDD clones follow the coarseness of the last step: 4 steps worst,
  6 and 8 alike, 16 rarest.**
  - Six has 8's last step at fewer evaluations.
  - Fails if 6 sits halfway between 4 and 8, which would mean the count, not
    the tail.

  *Verdict, 2026-09-27: failed on clones, and the step-count question
  answered another way.* From the VAE session's `2026-09-27_ladder.md`, on
  subway, slapstick and samurai.
  - The subway clone appears at 4, 6 and 8 steps, merged or exact, so clones
    do not follow the tail (and F5 put this one at step 1).
  - On quality: fine detail runs PDD4 < PDD6 < PDD8.
    - PDD6 is within 5-10% of PDD8, with level motion, at three quarters of
      the evaluations.
    - PDD4 is 22-32% below, with the least motion and more boil.
  - PDD6 is the candidate for the owner's faster PDD path.
  - Contrast barely moves with step count, so PDD8's flat grade is not a
    step-count effect.
- **F4. PDD8 exact vs merged moves the output less than FlashGen's switch
  did.**

  *Verdict, 2026-09-27: not measurable as written.* Final-latent distance is
  divergence (the method note). Merged against exact on subway is 0.58, a
  different take with the same character: both double the figure at
  latents 8-9 (the VAE session's merged preview, `2026-09-27_ladder.md`). PDD's delta is a larger fraction of an int8 step, so the merge lost
  less of it.
- **F5. In the per-step capture, the double image first appears at PDD8's
  last two steps, not in the early ones.**

  *Verdict, 2026-09-26: failed.* The VAE session previewed the x0 of
  `subway_chase__pdd8` (exact branch) through core's latent-to-RGB factors
  (`bench/x0_step_frames.py`, `2026-09-26_x0_steps_subway_pdd8.json`).
  - At step 0 there is one blob, already smearing wider at latent 9.
  - At step 1 there are two distinct figures at latents 8 and 9, and steps
    2-7 keep and sharpen both.
  - Late-step change per latent frame is flat, with nothing concentrated
    where the clone is.

  The clone is a composition decision at the highest noise, not late-block
  averaging, which fails the VAE session's P3 too. So a late handoff (the
  reverse step-switch, at 0.8 or 0.632) should not remove it. A different
  first block or early model might: route 3, FlashGen first. Whether the base
  makes the same choice from this seed and prompt is the open control.
  Registered before the reverse-switch renders land.

  *Annotation, 2026-09-27: the "two figures" are the prompt's two people.*
  The VAE session's base control (`2026-09-27_clone_base_control.md`) found,
  at full resolution, a black-jacket agent and a grey-hoodie suspect in both
  the base and PDD8, as the prompt names them. So step 1 decides the
  two-person composition, and that is not a clone. The verdict "failed"
  stands (nothing appears late). "The clone is decided at step 1" was the
  wrong framing. What remains is PDD8's overlap at about 1.1 s, where one
  figure passes in front of the other: the owner's call by eye.
- **F6. FastH3 with VSA off shows less temporal texture instability and the
  same grade.** Sparsity would explain the texture, and the weights the
  grade.

  *Note, 2026-09-26, the VAE session, before any result: the VSA-off arm
  departs from FastH3's contract twice.*
  - *It runs dense attention where FastH3 was trained sparse.*
  - *Core's dense forward ignores `to_gate_compress`
    (`comfy/ldm/minimax/model.py`, "unused by the dense forward"), which
    FastVideo trains always on. So the learned coarse branch goes too.*

  *A texture or grade change in that arm cannot be charged to sparsity
  alone.*

  *Verdict, 2026-09-27: half held, with the confound above.* From
  `analyze_followup.py --group vsa`, on slapstick and samurai at 730451892.
  - **Instability held on one scene:** boil fell from 0.60 to 0.43 on
    slapstick and was flat on samurai (0.47 against 0.45).
  - **"Same grade" failed:** samurai went much hazier with VSA off (haze
    0.14 to 0.32), brighter in the midtones and less saturated. Slapstick
    lost a little contrast and saturation.
  - **Detail roughly halved with VSA off** (0.102 to 0.075 on slapstick,
    0.099 to 0.049 on samurai), and so did slapstick's motion detail. So
    FastH3's high fine detail, the owner's "over-polish", comes with its
    trained attention path: sparsity plus the learned coarse branch, which
    this arm cannot separate.
  - No cube-period grid either way.
- **F7. The look family (`t2va_look_*`).**
  - The base follows each requested look.
  - The distills do not shift tone by a constant offset. They pull the
    unusual looks toward typical ones: more saturation into the muted
    variant, deeper blacks into the faded-16mm variant's lifted blacks, and
    colour leaking into black-and-white noir.
  - The pull is largest for FastH3, then FlashGen, then PDD8.
  - Fails, and points to a fixed tone bias instead (adaln, a learned curve),
    if each distill's shift from the base is the same whatever the look.

- **F8. Part of the "low bitrate" look is our own encode, not the model.**
  - The owner, 2026-09-26: "its always in shows like the witcher on netflix
    where dark/night scenes looked so blocky and splotchy".
  - Every graph saves through `VHS_VideoCombine` as 8-bit 4:2:0 h264 at
    CRF 19 (the generator's save node). x264's default adaptive quantisation
    starves dark flat regions of bits.
  - Prediction: decoding a `_savelat` latent losslessly shows less blocking
    and banding in the shadows than the mp4 of the same render. The gap is
    largest on grainy output (FastH3), where grain takes the bits.
  - Fails if the lossless decode is as blocky in the shadows as the mp4:
    then the blocking is the model's.

  *Verdict, 2026-09-27: held on one clip.* The VAE session's
  `2026-09-27_o1_lossless.md`, on look_anchor PDD8, 22 frames.
  - Decoded losslessly from the latent, the dark regions show no blocking
    excess (at or below the bright regions at 8 and 16 px).
  - The mp4 of the same frames adds blocking in the darks only.
  - A FlashGen look clip, the one that lifts blacks, would confirm it.

## vaedude

Sent by that session on 2026-09-26, before it had viewed any of the run's
clips, and placed here verbatim at its request.

> P1. PDD clones come from prediction under uncertainty in coarse late blocks. So by latent L2 distance from base-Euler-32 at the same seed, per latent frame: pdd4 > pdd6 ≈ pdd8 > pdd16 > pdd32, and the distance concentrates in high-delta latent frames. pdd6 sits nearer pdd8 than midway to pdd4 (the tail matters, not the count). Visibly, clones get worse as the tail coarsens, and pdd32 has none on the same scene.
>
> P2. The fusion pair: PDD32 (width 1, no fusion) sits closer to base-Euler-32 than PDD8 does on every scene. The gap is largest on motion scenes and near zero on still close-ups.
>
> P3. The x0 capture shows the double image first in a late PDD8 block (step 6 or 7 of 8), not in the early layout blocks, and base-Euler-32's x0 at the same sigma shows one person.
>
> P4. The sampler confound: base on Euler (16) is more contrasty and saturated than base on er_sde at the same seed, so part of tonight's "distills are contrastier than base" is the base's SDE softening, not the distill. By measure_clip_tone, base-Euler lands between base-er_sde and the distills.
>
> P5. Merged against exact PDD8, at tonight's seeds: latent distance well above run-to-run noise on every scene (the merge noises the backbone delta, cos 0.39), and exact sits closer to base-Euler-32 than merged. By eye, a small difference, not a new scene.
>
> P6. FlashGen adherence: at 124 frames with the 5-second prompt, role-swaps are rarest. Dense attention and strength 0.8/1.2 don't fix role adherence; "no adaln" changes the grade more than adherence.
>
> P7. FastH3's "low bitrate" is temporal. With VSA on, high-frequency energy on static surfaces varies more frame to frame after flow warping ("boiling"), and moving regions lose more HF detail than the base's do. With VSA off, both move toward the base. Still no 128-px grid either way.
>
> P8. Timing, under telemetry: at equal step counts PDD merged costs more per step than PDD exact-branch under dynamic VRAM, because 308 merged patches are re-applied to every weight streamed back in each step. Low confidence: the branch pays two small matmuls per module plus an A/B copy per call.

Note, 2026-09-26: tonight's PDD8 arms were stopped before rendering (the run
halted after its base arms), so P5 is tested against merged PDD8 arms rendered
in the follow-up batch at the same seeds.
