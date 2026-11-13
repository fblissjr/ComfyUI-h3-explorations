# The Krea 3 technical report, read for H3

2026-11-04. A dated record, not guidance. The owner asked whether anything in Krea's technical report
(`https://www.krea.ai/blog/krea-3-technical-report`, published 2026-11-03) applies here. A subagent read the whole
page and mapped it against this repo; a session checked the quotes against the page and the repo citations against
the files. Labels: **[R]** the report, **[F]** a file here, **[O]** reasoning. The same report read for the MOSS
repo is that repo's `research/krea3-report-for-moss-2026-11-04.md`.

## What the report is

Krea 3 is a closed image-editing model (up to 8 references) built for speed and cost. The authors call the
architecture "deliberately standard"; the contributions are training, data and post-training [R]:
- **References** are VAE-encoded and appended as clean tokens with the timestep modulation at t = 0. A VLM text
  encoder sees them downsampled to about 384² pixels.
- **Resolution:** reference and target resolutions are sampled independently, with native 2K training.
- **Shift:** a stronger timestep shift for image-to-image than for text-to-image.
- **Captions:** structured reasoning captions.
- **RL:** with pairwise and blind-VLM judges.
- **Inference work:** caching, quantization and kernels, with details deferred to a later report.

Almost all of it needs training at a scale this repo does not have. What can carry over is inference settings and
measurement.

## What could apply here, ranked by the owner's priorities

1. **Smaller references inside the DiT, for reference-to-video.**
   - [R]: "pairing a high-resolution output with lower-resolution references shortens the sequence" with little
     quality loss, except where fine detail such as text has to be copied.
   - [F]: the pack's default is the vendor's 2048 short edge with upscaling; a reference can reach about 7.5 MP
     (`docs/h3_references.md`, around line 244). On the day's Pika motion-transfer render the two 2048 stills were
     14,592 of 135,388 packed tokens (`data/results/2026-11-04_pika_step2_showcase_repro.jsonl`). No measurement of
     quality against size exists. `bench/refview2_arms.json` covers the text encoder's view and upscaling, not this.
   - [O]: Krea trained for mismatched sizes; whether H3 tolerates smaller references is unknown.
   - Cheapest test:
     1. Price the tokens with the conditioner's preview.
     2. Round-trip the stills through the video VAE at 2048, 1024 and 768, with no diffusion.
     3. Then a matched-seed pair, 2048 against 1024, on two reference-to-video shots.
2. **A stronger sigma shift for reference-to-video only.**
   - [R]: "most of the global structure … is decided at high noise levels". A more aggressive shift for
     image-to-image "significantly improved" structure in their ablations, applied in training and inference.
   - [F]: every H3 task runs `h3_config.SIGMA_SHIFT` (12 video, 3 audio).
   - [O]: changing it at inference only is a guess, and it must not touch Pika H3 Flex's own knots.
   - Test: base H3, shift 12 against one higher marked value, two shots, read blind.
3. **Rephrase rather than reseed a shot that keeps failing.** Low priority.
   - [R]: on 97 hard inputs, different paraphrases "solve almost twice as many cases as only varying the seed".
     Krea attributes part of the seed sameness to classifier-free guidance.
   - [F]: H3 graphs run without CFG.
   - Test: one stuck shot, four seeds against four paraphrases, each prompt read in full first.

## Already here, or closed

- **Clean reference rows:** condition timestep 0.999 (`docs/h3_references.md`, line 551).
- **A smaller text-encoder view of each reference:** `qwen_view=separate` at 512, the owner's decision of
  2026-10-03 (`docs/evidence.md`). The day's Pika step 2 graphs used `shared` on purpose, to match Pika's pipeline.
  Step 3 should use one setting for both arms.
- **Reference key/value caching:** needs a model trained for one-way attention. No training-free cache for a
  bidirectional video DiT was found (`docs/research/2026-09-19_in_context_reference_and_sinks.md`).
- **Structured prompts:** the vendor reference format, with keep-and-change analysis (`vendor_guides/ref_en.md`).
- **Quantization:** int8 convrot. Activation-rounding experiment #23 is closed (`docs/roadmap.md`, "Closed lanes").
- **Step caching:** surveyed and parked, with DPCache as the one option (`docs/research/2026-09-25_step_caching_survey.md`).
- **Few-step distills:** closed except Pika H3 Flex (`docs/roadmap.md`).

## Does not apply

- **Pretraining and finetuning:** packing, 2K training, the 30:70 text-to-image mix, data mining, synthetic pairs,
  captioning pipelines.
- **RL and preference tuning:** only their judges could carry over, as measurements.
- **"Optimized CFG":** H3 runs without guidance.
- **Context parallelism:** one GPU.
- **Kernels and caching:** no details are given.

## Not verified

Whether H3's reference-to-video training used references at a different resolution from the target, or a
different shift. The release does not say, so both ranked items 1 and 2 are untested extrapolations.
