# Sol dense_blocks panel on the ref2va finish (2026-10-02)

**Status: stopped early by the owner, and decided.** Two of four scenes at
one of two seeds were rendered, scored and judged, plus the dancer scene's
dense arm, before the owner stopped the panel on 2026-10-02: "so i guess we
found what doesnt work. and it doesnt buy enough time to be worth it. so lets
stop all the renders and just decide on a good solid general default".
The decision is at the end. The open questions stay open; nothing renders
them now.

**Why.** `bench/results/2026-10-02_sol_campaign_reanalysis.md` found that
Options A, B and C were adopted on local probe error alone: unfixed seeds,
routing on, nothing scored at the output. This panel asks the output-level
question on the shipped graph.

**Design.** `bench/sol_dense_blocks_panel_arms.json` (its `what`). Shipped
`workflows/distill_experiments/h3_probe_r2v_step_switch_pdd8_flashgen_h080_savelat_api.json`,
PDD8 then FlashGen finisher, 1344x768, 345 frames, Sol at the graph's own
settings (tau 1.0, routing off, `start_percent` 0.0 on the PDD node and 0.2
on the finisher). Four one-picture bank scenes (`ref2va_market_stallholder`,
`ref2va_stairwell_dialogue_backstage`, `ref2va_studio_dancer_close_refs`,
`ref2va_scene_subway`) at seeds 730451892 and 730451893. Six arms, patched on
both Sol nodes: `dense` (`dense_blocks=0-49`, every block on the kitchen INT8
fallback, the reference), `none` (empty), `tail` (45,48,49), `optB`
(39-42,49), `optC` (38-42,49, shipped), `optCall` (Option C with
`sink_conditioning=exact_kv_and_all_rows`).

**Scoring.** `bench/score_sol_dense_blocks_panel.py` against the rows in
`data/results/2026-10-02_sol_dense_blocks_panel.jsonl`: relative L2 of each
arm's saved latent from the `dense` arm at the same scene and seed. Timing is
`sampler_s` from the same rows, with no probe armed, so it is valid; every
cell was rendered on a warm server.

**Owner judgement.** On 2026-10-02 the owner clicked through the first
eleven clips (seed 730451892), with each clip's setting hidden until it was
rated, marking each "looks fine" or "something's off" with optional tags.
In the owner's words, "Motion" means the action goes wrong, "like super fast
transition or tossing coins in the air or walking backward with two crates",
and "Artifacts" was usually ticked alongside it. The review page was a
claude.ai artifact; it and its uploaded clips were deleted the same evening
at the owner's request, and the verdicts below are the only copy kept. Two later verdicts, on backstage `optCall` and `optC`, came from the owner watching the clip on the share by its filename, so the setting was known; the table says so.

## Seed 730451892, two scenes

| scene | arm | video vs dense | audio vs dense | sampler s | owner |
|---|---|--:|--:|--:|---|
| market | dense | 0 | 0 | 479 | fine |
| market | optCall | 0.873 | 0.668 | 337 | off: artifacts, motion |
| market | optC | 0.915 | 0.559 | 298 | off: identity, motion, faces, artifacts |
| market | optB | 0.925 | 0.874 | 293 | off: artifacts, motion |
| market | tail | 0.937 | 0.715 | 285 | off: motion, artifacts |
| market | none | 0.939 | 0.712 | 272 | off: motion |
| backstage | dense | 0 | 0 | 480 | fine |
| backstage | optC | 0.599 | 0.771 | 294 | off: "loud and looks kinda not as good just like option b" (watched by filename) |
| backstage | optB | 0.614 | 0.662 | 289 | off: audio |
| backstage | optCall | 0.642 | 0.576 | 333 | fine, "audio is good too" (watched by filename, setting known) |
| backstage | tail | 0.652 | 0.707 | 281 | fine |
| backstage | none | 0.657 | 0.669 | 268 | off: audio |

Mean distance between pairs of non-dense arms at the same cell: market
0.740, backstage 0.621 (the scorer's `sol_pairwise`).

## What this shows so far

1. **Sol changes what happens, not how the image looks.** On market every
   Sol arm got a scripted action wrong and the dense render did not. The
   owner rated two Sol clips before seeing the dense one, so this is not only
   a comparison against the reference.
2. **The scene matters far more than `dense_blocks`.** The action-heavy
   scene's Sol arms all sit much further from dense than the dialogue
   scene's, and the action-heavy scene is where the owner saw problems.
   Within a scene the six arms sit close together, closer to each other than
   to dense.
3. **`dense_blocks` does not predict the verdict.** Option C drew the most
   tags on market; the tail was the one Sol arm judged fine on backstage. The
   campaign tuned a knob that does not move the outcome here.
4. **The trade is Sol against dense.** Sol's sampler time is well below
   dense's on both scenes; on the action scene that saving buys a visible
   change to the action.

5. **On the dialogue scene Sol makes the audio louder, unless conditioning
   query rows are exact.** The owner's audio flags on backstage were "just
   being loud and jarring. maybe the others were too quiet". Measured with
   `ffmpeg -af ebur128` on each clip's `-audio.mp4`:

   | backstage arm | integrated loudness (LUFS) | loudness range (LU) |
   |---|--:|--:|
   | dense | -17.9 | 3.7 |
   | optCall | -16.8 | 4.2 |
   | optC | -14.1 | 6.0 |
   | none | -13.8 | 6.1 |
   | tail | -13.5 | 5.6 |
   | optB | -13.4 | 5.9 |

   Every arm that leaves text and reference query rows sparse
   (`exact_kv_and_rows`) is louder than dense with a wider range; the arm that
   makes them exact (`exact_kv_and_all_rows`) sits near dense. On market the
   six arms are within a narrow band of each other (same measurement), so the
   effect shows on the dialogue scene and not the action scene. The owner
   rated `tail` fine before any comparison and flagged `none` and `optB`
   later, which fits loudness being judged relative to what came before.
   Mechanism, not measured: text tokens are updated by sparse attention in
   every block under `exact_kv_and_rows`, which shifts the dialogue and voice
   conditioning the audio is generated from.

One seed, two scenes, one judging pass: these are the shape, not the size.
Market's dense render being the one clean clip could be a lucky seed; the
second seed decides that.

## Open questions

- **Does distance from dense predict the scenes Sol hurts?** Prediction
  written before the rest renders: dancer (fast motion, cuts on the beat)
  and subway (four shots, two speakers) sit at least as far from dense as
  market, and the owner sees action problems on them. If it holds, the
  distance is a way to screen scenes without watching them.
- **Does running the first steps dense fix the action?** The action and
  layout are set at high noise, in the PDD stage's first evaluations, and
  the shipped PDD node runs Sol from step 0. The 2026-10-01 owner panel
  (`2026-10-01_start_percent_panel.md`) could not tell 0.0 from 0.2 on t2v
  pairs, but judged them without a dense reference beside them. Test:
  market and dancer at PDD-node `start_percent` 0.0, 0.2 and 0.5, plus dense,
  one fixed seed.
- **Should Sol run only in the finisher?** Two of eight evaluations, after
  the action is set: part of the speedup, with the action fully dense. Test:
  PDD node at `dense_blocks=0-49`, finisher as shipped.
- **Is the loudness shift a dialogue-scene effect of sparse text rows?**
  Finding 5 rests on one scene and one seed. The second seed, and subway (two
  speakers), say whether every Sol arm with `exact_kv_and_rows` runs louder
  than dense on dialogue and `exact_kv_and_all_rows` does not. If it holds,
  `all_rows` is the one Sol setting with a measured audible benefit, against
  its extra sampler time (`optCall` against `optC` in the table).
- **Is the "Motion" failure long-range attention?** A hypothesis, not
  measured: keeping an action coherent across a clip needs attention between
  distant frames, which block-sparse selection is most likely to drop. A
  probe capture of the market scene, comparing routed density on
  video-to-video blocks far apart in time, would test it.

## Decision (owner, 2026-10-02)

The owner asked for "a good solid general default" from what this panel
showed. Set in 0.185.0:

- **`sink_conditioning` becomes `exact_kv_and_all_rows`**
  (`sol_attn_h3.py::SOL_SINK_DEFAULT`, mirrored in
  `workflows/h3_config.py`). It is the one Sol setting the owner rated fine
  on the dialogue scene, where every `exact_kv_and_rows` arm ran louder than
  dense, and it lost on no scene: the condition the roadmap's 2026-09-10 plan
  set for this flip. Its price is extra sampler time against plain Option C
  (the table above).
- **`dense_blocks` stays Option C.** The choice did not move the verdict,
  and C is what `all_rows` was judged with; the cheaper lists were never
  rendered with `all_rows`.
- **Sol stays on.** On the action scene no Sol setting held up, so a render
  where the action matters most can still run with Sol off (every block
  dense); that is a per-render choice, not the default.
- **Applied to every graph, video references included.** The roadmap's plan
  kept ref2va on `exact_kv_and_rows` because `all_rows` also runs a
  reference video's rows exact, which costs most of Sol's saving on those
  graphs. The owner asked for one general default, so it is one; the video
  reference graphs are where that costs time, unmeasured here.
