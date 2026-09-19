# Evaluation for one judge and one GPU: accelerated video+audio

> **Provenance, 2026-09-19.** Written by a research subagent of the `libguy`
> session for the lead's wider sweep, read-only, no GPU; section 0 was written
> first, at the lead's request, for the dense-against-default panel. The
> commissioning session re-checked its repo pointers and found them correct:
> the self-pair refusal (`bench/blind_batch.py:271-272`), the seed-base note
> (`bench/blind_batch.py:61-63`), render determinism and the 6 kHz low-pass
> rows in `bench/results/2026-09-19_audio_sage_vs_kitchen.md`, and Sol-Engine's
> ranking policy (`coderef/Sana/evals/tiers.toml:12-15,46-51`). It also
> rechecked two of section 0's binomial figures (the 0.451 bound for zero of
> five, and 8 decisive pairs for a 0.9 win rate). Everything in section 0 is a
> recommendation. Paper figures are the papers' own, several read through a
> summarising fetch tool; recheck against the PDF before reusing one.

Provenance. Research subagent, 2026-09-19, read-only, no GPU, no git. Repo
context read in full: `docs/eval_comparison.md`, `.claude/skills/h3-ab-session/SKILL.md`,
`internal/2026-09-19_question_review.md` (all sections, A, B, C.11 and F
closely), `bench/results/2026-09-19_audio_sage_vs_kitchen.md`, the tools
`bench/blind_batch.py` (argument handling), `bench/rubrics/default.json`,
`bench/quality_metrics.py` and `bench/diff_clip_graphs.py` (docstrings), and
`coderef/Sana/evals/` (tiers, rubric) plus `coderef/Sana/tools/vision/lpips_judge.py`.
Papers read in full text through a fetch-and-extract tool (HTML version; numbers
requested verbatim with their table or figure, but not read line by line by
me): AVBench 2605.24652, VideoScore2 2509.22799, VBench 2311.17982 (ar5iv),
VBench-2.0 2503.21755, THEval 2511.04520, the reference-free speech metric
audit 2609.13150, the AV-sync metric audit 2608.25157, Artifact-Bench
2605.18984, HEART 2605.14513, PASA 2604.12219, Veda 2605.30325, Sol-Attn
2607.24027, the MOS-JND study 2602.17010, early failure detection 2603.14320,
and VoxSim 2407.18505 (PDF text, section 4.1). Code read: `joonson/syncnet_python`
`run_pipeline.py`, the ColorVideoVDP README. Abstract pages only: everything
else in the source list, each marked there. Every arXiv id below was opened on
its abstract page. Numbers quoted from papers are THEIR measurements on THEIR
setups; the only numbers of mine are labelled binomial arithmetic (sections 0,
3 and 4), with the formulas in section 3.3.

## 0. For the next GPU job: fully dense against the shipped default, blind [I]

Written first at the lead's request. Everything in this section is a
recommendation [I]; the numbers are labelled binomial arithmetic (section
3.3 has the formulas) or quoted with their source.

### 0.1 Design

| item | what | cost | what it answers |
|---|---|---|---|
| contests | shipped default against fully dense, same seed per scene, **five scenes chosen before rendering**, each at its declared `frames`, drawn for different hard moments (entering figure, hands with objects, on-screen text, fast motion, dialogue) | 10 renders | the question itself |
| two-seed decoy | fully dense at seed s against fully dense at seed s' on one of the five scenes; needs a second arm label with its own seed base (`bench/blind_batch.py:61-63`) | 1 render | **the C.1 denominator**: how often this judge picks a winner between two takes of one arm. Unconditional in any batch meant to move a default |
| identical-pair decoy | one dense clip stacked against itself | 0 GPU; needs a flag, since `bench/blind_batch.py:271-272` refuses a self-pair | an attention check: does the judge report differences that are not there. Does **not** substitute for the two-seed decoy |
| low anchor | one dense clip against the same clip with a known, mild, aligned post-process defect: a short frame-hold stutter in the picture and the 6 kHz low-pass the audio record already characterised (`bench/results/2026-09-19_audio_sage_vs_kitchen.md:293`) | 0 GPU | the hit rate: could the judge see and hear a known defect that session |

Eight stacks in one shuffled session (five contests, two decoys, one anchor),
audio heard on every pair (the page's "Hear" row), predictions written in the
brief but defects not named to the judge (`docs/eval_comparison.md`, the
priming caution). Run `bench/diff_clip_graphs.py --expect` on every contest
before stacking; each contest must differ only in the attention chain and Sol.

### 0.2 Scoring ties and "can't tell"

The page's four verdicts stay as they are. The join reads each pair as one of:

- **defect-named preference**: a winner plus free text naming a defect inside
  the losing clip (morph, ghost, melted text, flicker, desync, muffled). This is
  the only outcome that counts as "the judge can tell".
- **staging preference**: a winner whose text names only a difference between
  takes (wardrobe, which character speaks, framing). Recorded, not counted as
  detection (review section A).
- **same**: evidence of no difference. Counts toward "cannot tell".
- **can't tell**: no judgement formed. Counts toward "cannot tell" for the
  stop rule, but reported apart: three or more of five says the scenes, not
  the arms, failed (`docs/eval_comparison.md`, verdict item 3).

For any directional statistic, drop same and can't tell (sign-test
convention) and report how many were dropped.

### 0.3 Stop rules for this panel

1. **Validity first.** If the identical pair drew a winner, or the anchor went
   unnoticed, record the session and do not read it. If the two-seed decoy drew
   a confident winner, discount every staging preference in the session and
   read only defect-named ones.
2. **"The judge cannot tell"**: no defect-named preference for dense on any of the
   five scenes. Per review F1, stop tuning Sol for quality and decide on
   seconds. State the size honestly: zero defect-named losses in five scenes
   bounds the per-scene detection rate below 0.451 at 95%, in ten scenes below
   0.259.
3. **"The judge can tell"**: two defect-named preferences for dense naming the same
   class of defect on different scenes. Sol or INT8 costs visible quality; the
   next job is the `start_percent` ladder, not more of this panel.
4. **One defect-named preference, or two of different classes**: undecided at
   five. Add five new pre-chosen scenes (not more seeds of the same five) and
   apply 2 or 3 again at ten. At ten without a decision: "not separable at this
   n", decide on cost.

### 0.4 Worked small-n numbers

- **False-positive rate from decoys** (exact one-sided 95% upper bound). With
  zero confident verdicts on k decoys: k = 1 gives 0.950, 3 gives 0.632, 5
  gives 0.451, 10 gives 0.259, 20 gives 0.139. With one confident verdict in
  5, 10, 20: 0.657, 0.394, 0.216. One decoy can only catch a bad session; the
  rate itself is estimated by pooling decoys across sessions, which is fair
  because the judge is the same person.
- **Pairs needed to separate a win rate from a coin** (ties dropped, one-sided
  exact sign test, alpha 0.05, power 0.8): 0.9 needs 8 decisive pairs with 7
  wins; 0.75 (1 JOD) needs 23 with 16; 0.6 needs 158 with 90. Nothing under 5
  decisive pairs can reach alpha 0.05 (0.5^5 = 0.031).
- **The review's own rule** (win or tie every scene, win at least two, five
  scenes) passes a coin-flipping judge with probability 0.031 if the judge never
  ties, 0.119 at a 40% tie rate, 0.120 at 60%; it passes a genuinely better
  arm (win 0.6, tie 0.3, loss 0.1) with probability 0.564. The one-loss veto is
  most of that: (1 - loss rate)^5.
- **Sequential**: Wald's SPRT for 0.5 against 0.75 (alpha 0.05, beta 0.2)
  stops for "better" after seven straight decisive wins and for "no better"
  after three straight losses; one loss cancels 1.71 wins.
- **Beta(1,1) posterior** that the win rate exceeds 0.5: 2 of 2 is 0.875, 4 of
  5 is 0.891, 5 of 5 is 0.984, 8 of 10 is 0.967.
- **Measuring a metric's agreement with the owner later** is the same
  binomial: agreement is the sign of the metric difference against the owner's
  verdict on decisive pairs. To show 0.9 agreement against chance takes about
  8 decisive pairs, 0.75 about 23, and 0.6 is out of reach. The on-disk stock
  is what `ls bench/results/*verdict*.json` lists: only some carry pair
  contests (`pairs.contests.*.tally`), and none has metrics logged beside it.

### 0.5 Which automatic metrics tracked humans for accelerated or approximated video

Short answer: **none was validated against people on acceleration artifacts
specifically.** The closest agreement statistics, each with its unit:

| metric or judge | statistic | unit and setup | source |
|---|---|---|---|
| PSNR/SSIM/LPIPS against dense | none reported | Sol-Attn reports them against dense attention with no human study | 2607.24027, Tables 1-5 |
| aligned pairwise VLM gate plus LPIPS max (Sol-Engine's shipped policy) | none reported; absolute thresholds disabled | the upstream H3 optimiser's ranking rule | `coderef/Sana/evals/tiers.toml:12-15,46-51` |
| best MLLM on pairwise video realism | 53.1% vs humans 86.4% (Gemini 3.1 Pro 48.6%) | per pair, two generated videos of similar content | Artifact-Bench 2605.18984, Table 2 |
| sparse against full attention, human side by side | ties: 63% on motion quality; VQ preference 50% vs 49% | per video, 304 prompts, trained sparse attention | Veda 2605.30325, Figure 7 (not correlated with its VBench table) |
| VBench / VBench-2.0 dimensions | Spearman over per-model win ratios; 81.70% to 99.46% (2.0) | **4 models** | 2311.17982; 2503.21755, Table IV |
| AVBench evaluators | Pearson 0.80 to 0.98 system-level; 85.4% per-pair average, 98.1% speech content | 4 model families; per pair across commercial models | 2605.24652, Figure 6, section 9.1 |
| VideoScore2 | PLCC 60.37 average; accuracy 44.35 | per video, heterogeneous generators | 2509.22799, Table 5 |
| LSE-C / LSE-D | Spearman -0.164 (p 0.53) / -0.269 (p 0.30) | 17 talking-head methods | THEval 2511.04520, Table 2 |
| UTMOS / DNSMOS / SCOREQ | 0.892 / 0.678 / 0.909 pairwise on artifact-rich speech; 0.510 / 0.516 / 0.502 on defect-free (human ceiling 0.764) | per pair | 2609.13150 |
| full-reference video metrics on diffusion VSR | LPIPS, DISTS best, none "sufficient" to replace subjective tests | aligned outputs, 6 methods | 2605.25940 (abstract) |

What to log beside this panel's verdicts, so agreement can be measured later:
section 5.2. The single item most worth adding for this panel is the
alignment index (per-frame PSNR/SSIM against dense, audio envelope
correlation), because it says whether each contest is a same-take comparison,
in which case LPIPS max and ColorVideoVDP JOD are readable, or two takes, in
which case no reference metric is.

## The answer, on one screen

- **No paper I could find validates any automatic metric against human
  judgement on acceleration artifacts specifically.** Acceleration papers,
  Sol-Attn's own included, report PSNR/SSIM/LPIPS against the dense output plus
  VBench, and either run no human study or run one side by side and never
  correlate it with the metrics. [V, section 1.1]
- **The human-alignment numbers that exist are system-level over a handful of
  generators**: VBench and VBench-2.0 correlate per-model win ratios over four
  models; AVBench over four model families; THEval over seventeen methods. They
  say a metric ranks very different generators; they say nothing about two
  near-identical renders of one model. [V, section 1.2]
- **In the regime that matches ours (two clean clips close in quality),
  2026 audits put automatic judges near chance**: MOS predictors at about 0.51
  pairwise accuracy on defect-free commercial TTS against a human ceiling of
  0.764 (2609.13150); the best MLLM at 53.1% on pairwise video realism against
  humans at 86.4% (Artifact-Bench, Table 2); AV-sync metrics disagreeing with
  each other at Krippendorff alpha 0.066 (2608.25157); LSE-C/LSE-D
  negatively, and not significantly, related to preference (THEval Table 2). [V]
- **So a metric cannot stand in for the owner here.** What is worth doing is
  logging a small, cheap set beside every verdict, so agreement can be measured
  on the owner's own verdicts, and treating reference-based distance as valid only for
  pairs that are recognisably the same take (review B.4). The one reference
  metric scaled in human units is ColorVideoVDP (1 JOD = 75% pairwise
  preference), calibrated on compression and display distortions, not
  generative ones. [V for the metric, I for the use]
- **At five scenes and one pair each, the review's own decision rule passes a
  coin-flipping judge between 3% and 12% of the time depending on how often the judge
  calls ties, and passes a genuinely better arm only a third to a half of the
  time for the effects tabulated.** Only a decoy tells which regime a batch is
  in. [I, labelled arithmetic, section 3.3]
- **Two decoys, doing different jobs.** A clip stacked against itself costs
  nothing and checks attention; `bench/blind_batch.py:271-272` refuses it today
  ("needs two distinct arm labels"). Only a same-arm, two-seed decoy measures
  what the review's C.1 asks, how often this judge picks a winner between two
  takes of one arm, and because renders are bit-deterministic
  (`bench/results/2026-09-19_audio_sage_vs_kitchen.md:105`) that pair differs by
  the seed and nothing else. [V for the refusal and the determinism, I for the
  use]

## What this means for H3, ranked

Ranked by how directly each could change a default within a week. Each says
what to measure first, zero-GPU first.

1. **Put controls into the next blind batch before it decides `start_percent`
   or the F1 dense-against-default question** (section 0 has the design). A
   two-seed decoy (one render), unconditional in any batch meant to move a
   default, because it is the only control that measures the C.1 denominator;
   an identical-pair decoy (free), which is an attention check and cannot
   substitute for it; a post-processed low anchor (free). Then read the
   verdicts with the operating characteristic in section 3.3 rather than as a
   tally. [I]
   *Measure first:* nothing new. Count "same" and "can't tell" in the existing
   verdict records (`bench/results/*_verdict.json`, contest `tally` fields) to
   see what tie rate the judge actually runs at, which is the input the
   section 3.3 arithmetic needs. The one on-file near-null reading is session 1
   of 2026-08-20 (`docs/roadmap.md:683-691`; LoRA recipes, not a decoy).
2. **An alignment index on every pair, then reference metrics only on the
   aligned ones.** Per-frame PSNR/SSIM trajectory against the dense render and
   the audio record's 20 ms envelope correlation decide whether a pair is inside
   B.4's regime; only then log LPIPS max, DISTS and ColorVideoVDP JOD. This is
   what could make distance-to-dense a legitimate screen for a `start_percent`
   sweep (whose small moves are the likeliest to stay aligned). [I]
   *Measure first:* the alignment index on the clips already stacked and scored
   (`Video/reorder_panel/`, `Video/sage_chain_panel/`), CPU. How many real pairs
   stay aligned is unknown and decides whether item 2 is worth building.
3. **Audio: add the one speech check with high reported agreement, and stop
   expecting any metric to answer "more natural".** Whisper transcription error
   against the scripted dialogue line had the highest human agreement of any
   dimension in AVBench (98.1% instance-level, section 9.1). It detects garbled
   or dropped words, not prosody. MOS predictors, LSE and sync scores are
   alarms for gross defects only. [V for the numbers, I for the use]
   *Measure first:* transcription error on the two clean chain pairs and the
   aligned token-order pairs named in the audio record. CPU is enough.
4. **C.11: two candidate functionals of attention error from 2026 work, both
   computable on existing captures, neither validated against humans.** HEART
   measures error at the model output (the denoising velocity), weighted toward
   low temporal frequency, but its weights were fitted to distance-to-dense,
   not to visible defects; PASA blames flicker on the same blocks being
   approximated step after step. [V for what they claim, I for the transfer]
   *Measure first:* on captures already on disk, the temporal spectrum of Sol's
   attention error per block (reshape by latent frame index; report the low
   and the high temporal shares, since HEART and an artifact taxonomy point
   opposite ways), and route persistence across steps (the B.6 "E" measure).
   With the scored scenes on file this can only falsify: a random score
   perfectly separates two clean scenes from three morphed ones one time in
   ten (section 4).
5. **Do not build**: a VLM pairwise judge as a decider (Artifact-Bench; this
   is what Sol-Engine ships, `coderef/Sana/evals/tiers.toml:46-51`), a MOS
   predictor or LSE as a decider, VBench scores (system-level validation only),
   FVD or FVMD (distribution metrics, meaningless at one clip per arm). [V
   evidence, I conclusion]

## 1. Automatic metrics against humans for accelerated video

### 1.1 What acceleration work measures, and what it does not

- **Sol-Attn (2607.24027)**: PSNR, SSIM and LPIPS against dense attention in
  Tables 1-5, VBench for text-to-video (Table 1); "PSNR, SSIM, and LPIPS
  quantify fidelity to dense attention outputs". No human study, no audio, and
  MiniMax H3 is not among its models. [V, full text via extractor]
- **Sol-Engine, the upstream that ships Sol-Attn on H3**, ranks configs by
  "aligned pairwise Gemini severity/status, aligned LPIPS, and then higher speed
  as a tie-breaker"; "LPIPS is telemetry and ranking signal, not an absolute
  delivery threshold for lossy generative dimensions"
  (`coderef/Sana/evals/tiers.toml:12-15`), with both absolute thresholds set to
  `disabled_for_lossy_generative_dimensions` (`:46-51`). The judge rubric
  (`coderef/Sana/evals/rubrics/gemini_visual_artifact_gate.md:38-53`) is a
  twelve-category artifact taxonomy (ghosting and smearing, melting and
  morphing, temporal flicker and popping, patch-boundary discontinuity,
  degraded text, faces and hands, and so on). LPIPS is logged per frame with
  mean, median and max (`coderef/Sana/tools/vision/lpips_judge.py:33-46`). The
  H3 model entry names an eval profile
  (`coderef/Sana/models/minimax_h3/model.toml:10`) that is not in the clone,
  nothing under `coderef/Sana/evals/` mentions audio, and the public H3
  deployment page names LPIPS as its only quality measure
  (https://nvlabs.github.io/Sana/Sol-Engine/H3/). [V] No agreement statistic
  between that Gemini gate and people is given anywhere I found. [V negative]
- **Veda (2605.30325)** is the clearest human study of sparse attention
  against full attention I found: side by side on Waver-bench (304 prompts),
  four dimensions, reported as preference and tie rates ("preference rates for
  VQ and PF are nearly identical (e.g., 50% vs. 49% for VQ), while a 63% tie
  rate in MQ" at 90% sparsity, Figure 7). VBench is reported separately (Table
  1) and never correlated with the human result. Veda needs distillation
  training, so only its evaluation design transfers to frozen H3. [V]
- **PASA (2604.12219)** measures its flicker claim with VBench temporal
  flickering (Table 1) and runs no human study. [V] VBench temporal
  flickering is, as the extractor summarised VBench's dimension section, a mean
  absolute difference between consecutive frames; if so it is close to this
  repo's `delta`, which `docs/eval_comparison.md` records failing twice as an
  across-clip account. [I; the definition not read line by line]

### 1.2 Where human-alignment numbers exist, read the unit

| source | what is correlated | unit | number (their setup) |
|---|---|---|---|
| VBench 2311.17982 | VBench win ratio vs human win ratio, per dimension | per model, **4 models** (LaVie, ModelScope, VideoCrafter, CogVideo) | Spearman per dimension; [?] exact range not extracted |
| VBench-2.0 2503.21755 | same | per model, **4 models** (Sora-480p, HunyuanVideo, CogVideoX-1.5, Kling 1.6) | 81.70% to 99.46% across 18 dimensions (Table IV, Figure 11) |
| AVBench 2605.24652 | metric win ratio vs 2AFC human win ratio, 4 expert annotators | per model family, **4** (Kling 2.6, Sora 2, Veo 3 Fast, Wan 2.6) | Pearson: SyncNet 0.8194, NISQA 0.8012, speech content 0.9779 (Figure 6) |
| AVBench, section 9.1 | automatic choice vs human choice on a pair | **per pair** | 85.4% mean over seven dimensions, 98.1% speech content (Figure 13); per-dimension values only in the figure |
| VideoScore2 2509.22799 | predicted vs human 1-5 score | per video, heterogeneous generators | accuracy 44.35 average, PLCC 60.37 average (Table 5); out-of-domain average 50.37 (Table 6) |
| THEval 2511.04520 | metric vs human "more realistic" win rate | per method, **17** methods, 3,519 ratings | LSE-C rho -0.164 (p 0.530), LSE-D -0.269 (p 0.297), THEval final 0.870 (Table 2) |

[V for each row as extracted.] The per-pair row is the only one that speaks to
a judgement like the owner's, and its pairs are drawn across different
commercial generators, whose gaps are larger than any gap between two H3
arms. [I]

### 1.3 The regime that matches ours: two clips close in quality

- **Artifact-Bench (2605.18984)**, pairwise video realism: two generated videos
  with similar content, pick the one with fewer or milder artifacts. Best
  model 53.1% (VideoVeritas 8B), Gemini 3.1 Pro 48.6%, humans 86.4%;
  fine-grained artifact identification under 10% for every model against 80.3%
  for humans (Table 2). "Temporal-spatial artifacts are inherently more
  challenging because they cannot be identified from isolated frames alone."
  Qwen3-VL 32B-Instruct, the family H3's text encoder comes from, scores 39.5%
  overall. [V] This is the direct evidence against a VLM pairwise judge as a
  decider on H3 arms. [I]
- **Reference-free speech metrics on clean pairs** (2609.13150), section 2.

### 1.4 Distance to the exact render, and the one human-scaled version

- **Reference metrics on generative outputs.** On diffusion video
  super-resolution, "CNN-based full-reference models, such as LPIPS, DISTS, and
  CVQA-FR show significantly higher correlation coefficients than both
  conventional full- as well as the tested no-reference models", yet "none of
  the tested video quality models reach sufficient accuracy so as to replace
  complementary subjective testing", with VMAF overrating over-sharp results
  (2605.25940, abstract only). [V abstract] Super-resolution keeps the
  reference's structure, so this is the aligned regime: the best case for a
  reference metric, and still not sufficient. [I]
- **ColorVideoVDP (2401.11485, SIGGRAPH 2024)**: full-reference, models
  spatial and temporal contrast sensitivity, reports quality in JOD. From the
  README: "If method A produces a video with the quality score of 8 JOD and
  method B gives the quality score of 9 JOD, it means that 75% of the
  population will choose method B over A." Needs a display model (size,
  resolution, peak luminance, viewing distance); `pip install cvvdp`, CUDA
  recommended, CPU supported but slow; `cvvdp-ml` variants since 0.5.0. Its
  calibration data are streaming distortions (compression, rescaling,
  transmission) plus the XR-DAVID display distortions. [V, abstract and
  README] It is the only metric found whose output is already in the units a
  "can the owner tell" question needs, and it is valid only where the two
  renders are the same take. [I] No 2026 work applying it, or any JOD-scaled
  metric, to accelerated or quantized video diffusion turned up. [V negative;
  searched "ColorVideoVDP / VDP JOD video diffusion acceleration quantization"]
- **The MOS-JND study (2602.17010)**, compression on HD-VJND, 20
  participants, DCR: only 10.71% of sources showed a significant difference
  between the reference and the first-JND video; "a DCR study with few
  participants may fail to detect meaningful differences between JND videos
  and the reference, as their quality is often too close." [V] For one judge,
  a rating design cannot see a threshold-sized difference; a forced choice or
  same-different design at least can. [I]
- **The one upstream H3 fidelity calibration** is sglang's cosine-versus-dense
  sweep with a stated run-to-run floor and a visible re-framing at the low end,
  already in `docs/research/2026-09-17_rotation_and_lowbit_attention_survey.md`
  section D.6. It ties a distance to "visibly re-frames the shot", not to a
  human threshold. [V, via that survey]

### 1.5 Disqualified at one clip per arm

FVD and FVMD (2407.16124, "measuring similarity between these features via the
Fréchet distance") compare distributions and need many samples per arm. [V
abstract] VBench scores are per-video but validated only at system level
(1.2). [V]

### 1.6 The gap

Searched without success: human-metric agreement specifically on acceleration
artifacts (flicker, morphing, identity drift, duplicated figures) for sparse
attention, caching, quantization or distillation; a 2026 calibration of a
distance to the exact render against a human threshold; a tie-rate or
indistinguishability study for quantized or cached video diffusion. Queries
included combinations of "human study", "correlation", "PSNR LPIPS against full
model", "just noticeable difference", "JOD", "tie rate", "side-by-side",
"training-free acceleration" with 2026. [V negative] The honest summary is that
this repo's owner-verdict corpus, with metrics logged beside it, would be one
of the few datasets of its kind. [I]

## 2. Audio and audio-visual metrics for generated video with speech

| metric | what it measures | agreement evidence (unit) | on this box | use on H3 |
|---|---|---|---|---|
| SyncNet LSE-C / LSE-D (Wav2Lip 2008.10010) | lip-audio embedding distance and offset confidence | THEval: -0.164 / -0.269 vs realism preference, not significant (17 methods); AVBench: 0.8194 system-level (4 families); Wav2Lip "achieves top algorithmic scores by optimizing the metric directly, yet it was among the least selected methods" (THEval) | small CNN plus S3FD [I] | gross desync alarm only |
| Synchformer DeSync | audio-video offset in seconds | tracks a controlled shift at Kendall tau 0.84 [0.78, 0.90]; against the PEAVS perceptual proxy tau 0.07 (2608.25157) | [I] small | offset alarm against the broadcast threshold below |
| ImageBind / JavisScore | audio-visual semantic relevance | tau 0.20 against the PEAVS proxy (2608.25157) | [I] | no |
| PEAVS 2404.07336 (2024) | learned 5-point AV-sync score | Pearson 0.79 set-level, 0.54 clip-level (abstract) | [?] weights not checked | candidate, weak per clip |
| UTMOS 2204.02152, DNSMOS 2010.15258, SCOREQ | speech MOS prediction | pairwise accuracy on artifact-rich BVCC: UTMOS 0.892, DNSMOS 0.678, SCOREQ 0.909 (human ceiling 0.887); on defect-free TTS-HP: 0.510, 0.516, 0.502 (ceiling 0.764); "a content-blind clip-duration cue matches the best metric" (2609.13150) | [I] small; DNSMOS ships ONNX | alarm for audible defects; not for "more natural" |
| Whisper-large-v3 transcription vs text | speech content accuracy | AVBench: 98.1% per-pair (section 9.1), 0.9779 system-level (Figure 6) | [I] fits a 4090; CPU slower | yes, against the scripted line |
| speaker-embedding cosine (ECAPA-TDNN, WavLM-ECAPA) | voice identity | off the shelf, LCC 0.768 and 0.733 with human similarity ratings (VoxSim 2407.18505, section 4.1, Table 3) | [I] small | voice drift between arms, per turn |
| Audiobox Aesthetics 2502.05139 | four axes for speech, music, sound | abstract only; [?] numbers not read | [I] | candidate for non-speech audio |

**Gap, stated once:** the "on this box" column is unmeasured. No runtime was
measured and no model size was checked for this draft, so the brief's "runs on
one consumer GPU in minutes" is not answered here beyond the inference that
every model listed is far smaller than the DiT. Timing each on one existing
345-frame clip, CPU and card, is the cheapest thing to settle before 5.2 is
built. [?]

Notes, each [V] unless marked:

- **What the SyncNet pipeline drops.** `syncnet_python/run_pipeline.py`
  resamples video to 25 fps and audio to 16 kHz mono, splits shots with
  `scenedetect`'s `ContentDetector`, detects faces with S3FD at confidence 0.9
  and scale 0.25, and keeps tracks of at least 100 frames with faces of at
  least 100 pixels (defaults `--min_track 100`, `--min_face_size 100`). H3
  renders at 24 fps with several cuts and wide multi-speaker shots, so [I]
  short shots and small faces produce no score at all, and a score that exists
  describes only the close-ups. That selection has to be recorded beside any
  LSE number.
- **The 2026 AV-sync audit** (2608.25157, ECCV 2026 workshop) finds the four
  metrics it tests "measure different problems": Synchformer tracks offsets
  and misses content disruption, the embedding metrics track perception better
  and miss offsets, and fusing them did not help. It recommends reporting a
  "Reliability Card" per metric family with confidence intervals. LSE-C/D were
  not in its set.
- **A human threshold for offset exists and is normative**: ITU-R BT.1359
  detectability of +45 ms (audio leading) to -125 ms (audio lagging), as
  quoted by 2212.01686. [V secondary; the recommendation text itself not
  opened; the acceptability range of about +90 to -185 ms appears only in
  secondary sources, [?]] An offset estimate inside the detectability window
  cannot explain a sync complaint. [I]
- **MOS predictors on clean pairs.** 2609.13150 also finds that optimizing a
  single metric as a reward "improves the metric while independent human
  judgment and listening tests decline", and that a calibrated ensemble holds
  up better. The regime split (agree on defects, near chance on clean pairs) is
  the same split the audio record hit from the other side: nothing clears the
  floor between two clean chains. [V for the paper, I for the parallel]
- **Talking video.** THEval's own mouth visual quality and head-motion metrics
  track its realism preference (0.765 and 0.763, final score 0.870, Table 2,
  17 methods) where LSE does not. [V] They target single talking heads, not
  multi-speaker scenes. [I]
- **"More natural" is prosody and delivery.** No metric above measures it,
  which the audio record already concluded for its own descriptors
  (`bench/results/2026-09-19_audio_sage_vs_kitchen.md`, "Power"). [V]
- **What the pack already has**: the per-second band grid, loudness, HNR and
  hum descriptors with a built floor (the audio record, "Tools and settings"),
  and a mouth-motion-versus-loudness coupling instrument with a circular-shift
  null (`bench/measure_audio_video_coupling.py`, docstring). [V]

## 3. Small-panel statistics for one judge

### 3.1 Controls, borrowed from listening-test practice

- **MUSHRA (ITU-R BS.1534-3)** hides a copy of the reference among the test
  items and adds anchors, "typically a 7 kHz and a 3.5 kHz low-pass version of
  the reference", then disqualifies "all listeners who rate the hidden
  reference repeat below 90 MUSHRA points for more than 15% of all test
  items". [V, via the Wikipedia summary of the recommendation] The two
  controls answer different questions: the hidden reference measures false
  alarms, the anchor measures whether the listener could hear a known defect
  that session. [I]
- **Mapped to H3** [I]:
  - *Identical-pair decoy* = hidden reference. Free: one file stacked twice.
    An attention check, and the nearest null for **aligned** real pairs:
    "does the judge report differences that are not there".
  - *Two-seed decoy* = same arm, two seeds, the review's proposal, and the
    only control that answers C.1: "does the judge pick a winner between two
    takes of one arm". Since renders are deterministic, the pair differs by the
    seed alone. One extra render, unconditional in any batch meant to move a
    default; the identical pair does not replace it. Note that
    `bench/blind_batch.py` matches pairs by run index and seeds them
    identically "unless the seed bases were set apart on purpose"
    (`bench/blind_batch.py:61-63`), so it needs a second arm label with its
    own seed base.
  - *Low anchor* = the dense clip with a known, aligned degradation applied in
    post. The audio record's instrument check already built two (a 6 kHz
    low-pass and a -3 dB gain, `bench/results/2026-09-19_audio_sage_vs_kitchen.md:292-293`).
    A video anchor could be a short frame-hold stutter or a per-frame
    luminance jitter over a second or two. Missing the anchor makes that
    batch's "same" verdicts non-evidence.
- **Same, can't tell, and signal detection.** The rubric's split between
  *same* and *can't tell* (`docs/eval_comparison.md`, section 3) is already a
  same-different design. The decoys estimate the false-alarm rate, the anchor
  the hit rate; with a handful of each, d' is not estimable precisely, and the
  controls act as vetoes, not estimates. [I]
- **Oddity (triangle) test**, for the question "can the judge tell at all" (review
  F1): three clips, two from one arm and one from the other, **all at
  different seeds** so composition gives no cue, pick the odd one. Chance is
  1/3, it needs no preference direction, and staging cannot drive it. Cost: three
  renders per trial. The standard is ISO 4120:2021 (tables of minimum correct
  answers; [?] the table itself not opened). [I for the design]

### 3.2 Scaling models, ties and order

- Bradley-Terry with ties: Rao and Kupper (1967) and Davidson (1970, JASA 65,
  317-328, doi 10.1080/01621459.1970.10481082). [V citation from the
  publisher listing; paper not opened] Thurstone case V scaling and JOD units
  with confidence intervals: Perez-Ortiz and Mantiuk, pwcmp (1712.03686). [V
  abstract] Order effects with strength-dependent ties: Glickman (2025,
  doi 10.1177/1471082X251400474). [?] listing only, the page refused access.
- [I] With one judge and two arms per contest, a scaled model adds nothing to
  the sign count. It earns its place for a ladder of three or more arms against
  one reference (a `start_percent` sweep at 0, 0.1, 0.2, 0.3), because it pools
  the transitive information across contests, and Davidson's model keeps the
  ties instead of discarding them. Slot order is already randomised per pair
  (`docs/eval_comparison.md`, section 3), which is the precondition for
  estimating a position bias later.

### 3.3 Labelled arithmetic: what five pairs can and cannot say

All binomial, independent scenes, no data. W, T, L are the per-scene
probabilities that a verdict goes to the candidate, is a tie (same or can't
tell), or goes against it.

**The review's rule** (section A: win or tie on every scene, win on at least
two, five scenes) passes with probability
`sum_{k=2..5} C(5,k) W^k T^(5-k)`.

| judge | W / T / L | P(rule passes) |
|---|---|---|
| null, never ties | 0.5 / 0 / 0.5 | 0.031 |
| null | 0.4 / 0.2 / 0.4 | 0.074 |
| null | 0.3 / 0.4 / 0.3 | 0.119 |
| null | 0.2 / 0.6 / 0.2 | 0.120 |
| null | 0.1 / 0.8 / 0.1 | 0.058 |
| real effect | 0.6 / 0.3 / 0.1 | 0.564 |
| real effect | 0.4 / 0.5 / 0.1 | 0.434 |
| real effect | 0.5 / 0.3 / 0.2 | 0.305 |

The one-loss veto dominates the power: `(1-L)^5` alone is 0.774 at L = 0.05,
0.590 at 0.1, 0.328 at 0.2. [I, arithmetic] Reading: the rule is a fair guard
against a coin-flipping judge only if the judge's null tie rate is known,
which is exactly what the decoys measure; and a candidate that is truly better
but loses one scene in ten on staging fails the rule four times in ten, which
is why the review's defect-versus-staging distinction must be binding. [I]

**How many decisive pairs a sign test needs** (ties dropped, one-sided exact,
alpha 0.05, power 0.8): 8 pairs with 7 wins to detect a 0.9 win rate; 23 with
16 wins for 0.75, which is 1 JOD; 158 with 90 wins for 0.6. No result below 5
decisive pairs can reach alpha 0.05 at all (0.5^5 = 0.031). [I, arithmetic]
One judge at a 0.75 win rate is therefore a multi-session question, and a
0.6 effect is out of reach.

**Sequential version (Wald SPRT, Annals of Math. Stat. 16(2), 1945,
doi 10.1214/aoms/1177731118)**, H0 win rate 0.5 against H1 0.75, alpha 0.05,
beta 0.2: stop for H1 when the likelihood ratio reaches 16, for H0 when it
falls to 0.211. Each win multiplies it by 1.5, each loss by 0.5, so one loss
cancels 1.71 wins; seven straight wins stop for H1, three straight losses stop
for H0. [I, arithmetic; V for the method] Anytime-valid variants (test
martingales, e-values) keep the error guarantee under any stopping rule,
including "stop when the owner is tired" (Ramdas, Grunwald, Vovk, Shafer,
2210.01948). [V abstract]

**Bayesian reading** (Beta(1,1) prior on the win rate, ties dropped),
P(win rate > 0.5): 2 of 2 wins 0.875; 3 of 3 0.938; 4 of 5 0.891; 5 of 5
0.984; 8 of 10 0.967. [I, arithmetic: `1 - I_0.5(k+1, n-k+1)`] Two wins out of
two, the shape of many verdicts on file, is still one chance in eight of the
arm being no better.

**What decoys buy** (exact one-sided 95% upper bound on the false-alarm rate):
after 0 false alarms in 1, 3, 5, 10, 20 decoys the bound is 0.950, 0.632,
0.451, 0.259, 0.139; with one false alarm in 5, 10, 20 it is 0.657, 0.394,
0.216. [I, arithmetic: `1 - 0.05^(1/k)` for zero] One decoy per batch
identifies a bad judge-scene combination (a confident verdict on an identical
pair), not a good one; the rate is only estimable by pooling decoys across
batches, which is fair because the judge is the same person.

**Oddity test, chance 1/3**: minimum correct for alpha 0.05 is 4 of 5, 5 of 6,
6 of 8, 7 of 10. [I, arithmetic]

### 3.4 Pooling across scenes

[I] The scene is the unit. Two seeds of one scene are two readings of that
scene, not two independent readings of the arm; a scene-by-arm interaction
cannot be separated from noise at one pair per scene. Report the per-scene
sign pattern and a sign test over scenes; when a scene has several seeds, take
its majority first. A mixed model with a scene effect needs more pairs than
this judge will score.

### 3.5 Using the metrics once they are logged

Prediction-powered inference (2301.09633) gives valid confidence intervals for
a quantity estimated from a few human labels plus many model predictions,
narrower as the predictor improves; "AutoEval done right" (2403.07008) applies
it to model evaluation and reports up to 50% more effective human sample size
with GPT-4 as the predictor. [V abstracts] [I] For this repo it is the right
end state, not the first step: it assumes a predictor that agrees with the
owner, and none has been shown to. First measure agreement per metric: on
decisive pairs, does the sign of the metric difference match the owner's
verdict. That is the binomial of section 3.3, so the horizon is known in
advance [I, arithmetic]: about 8 decisive pairs to show 0.9 agreement against
chance, about 23 for 0.75, and a 0.6 agreement is out of reach for one judge.
A defect-class breakdown (5.1 item 6) multiplies those counts per class.

## 4. Screening: internal signals that predict the final perceptual difference

What 2026 work offers toward C.11 ("which functional of attention error
predicts a visible artifact"). None is validated against a human verdict.

- **HEART (2605.14513, renamed from HASTE; training-free, per head).** Claims
  "a threshold that appears acceptable under attention-output error may still
  produce a disproportionate error in the denoising velocity", because
  "downstream layers can attenuate or amplify local perturbations". It
  calibrates per head, in isolation with the other heads dense, at four
  timesteps, on a velocity error weighted by 3D-FFT band with weights
  (LL, LH, HL, HH) = (1.0, 0.5, 0.01, 0.01), the first letter temporal and
  the second spatial frequency (Eq. 7, section 4.3), so errors that persist
  across frames count a hundred times more than frame-to-frame changes. The
  weights come from Table 1: equal-relative-magnitude perturbations injected
  into the velocity of Wan2.1-1.3B, one band at a time, scored against the
  dense output (LL: PSNR 28.19, LPIPS 0.1024; HH: PSNR 30.35, LPIPS 0.0791;
  "Lower-frequency perturbations are more harmful"). Further evidence: Figure 3
  (attention-output error, velocity error and sparsity per head), and a Table 4
  ablation where the weighted objective beats plain velocity MSE by a small
  margin (PSNR 21.61 against 21.54, LPIPS 0.2337 against 0.2369 at 1.44x). No
  comparison against an attention-output objective in that table, no human
  study. [V]
  **Caution** [I]: "harm" in Table 1 is distance to the dense render, the
  measure the review's B.4 says rewards staying on the same take. A
  low-frequency perturbation moves composition and so moves every reference
  metric; that it is more *visible as a defect* is not shown. The weighting
  also puts flicker-like error (high temporal frequency) at 0.01, the opposite
  of what an artifact taxonomy would do. For C.11 it is a candidate to test, not
  an answer.
- **PASA (2604.12219, training-free).** Claims deterministic block routing
  produces "extremely skewed allocation" (the same high-attention blocks always
  exact, the same mid and low blocks always approximated), "a fundamental
  driver of temporal flickering"; evidence is Figure 4 (a stochastic bias
  redistributing selections) and VBench temporal flickering; it allocates
  density per step from the L1 distance between consecutive velocity
  predictions (Figure 3, three phases). No human study. [V] Sol routes
  deterministically from block means (the setting), so the claim applies to it
  in form. [I]
- **SVOO (2603.18636, ICML 2026)**: per-layer attention sparsity is "an
  intrinsic layer-wise property, with only minor variation across different
  inputs". [V abstract] If that holds for H3, block-level policy profiled on
  one scene transfers to others. [I]
- **Step-level signals.** TeaCache (2411.19108): timestep-modulated input
  differences "correlate strongly" with output differences and gate caching.
  SenCache (2602.24208): output sensitivity to latent and timestep
  perturbations "is a key predictor of caching error". RACER, "Disagree to
  Accelerate" (2608.01740): disagreement between two feature forecasts as a
  free runtime signal. All predict an output change at a step, not a
  perceptual difference; none reports a correlation with a human judgement.
  [V abstracts]
- **Early previews.** Early failure detection (2603.14320) decodes
  intermediate latents and scores text-video alignment with ViCLIP; failures
  are called on average at step 11 of 50 (CogVideoX) and 10 (Wan), against a
  threshold loosely checked by people. It detects prompt-level failure, not
  fine artifacts. [V]

**What this suggests for C.11** [I]. The two 2026 candidates are (a) where
the error lands in temporal frequency, per HEART, and (b) whether the same
blocks are approximated step after step, per PASA. Both are computable on the
existing captures: (a) by reshaping Sol's attention-output error field by
latent frame index (which needs `token_order` recorded, review section E) and
taking its temporal spectrum per block, reporting the low and the high share
separately, because HEART's weighting and a flicker-and-morph taxonomy predict
opposite signs and only the owner's verdicts can say which is right on H3; (b)
as route overlap across captured steps, the B.6 "E" measure, reported against
a shuffled-route null. HEART's velocity-level error needs the rest of the network
and is a GPU measurement; the review's structured-injection probe (C.2) is its
analogue here.

**How much the scored scenes can say** [I, arithmetic]. With two scenes
called clean and three called morphed, a score unrelated to the truth ranks
them perfectly one time in `C(5,2) = 10`. A perfect separation on the scenes
on file is therefore weak support and a failure to separate is informative.
Pre-register the functional, the regions and the direction before computing,
as the review asks for B.2.

## 5. Protocol recommendation for this repo [I]

Everything here is a proposal. It keeps the owner's process
(`docs/eval_comparison.md` section 3; free text first, pairs not singles,
tallies not means, no 1-5 scale) and adds only controls and logging that cost
the owner little or nothing.

### 5.1 Every blind batch

1. **Contests against the fully dense render**, not against today's default,
   wherever the question is "can the judge tell": the review's reframed question
   (section A) and the MUSHRA shape (reference, test, hidden reference,
   anchor).
2. **One two-seed decoy, in every batch meant to move a default.** The
   reference arm at a second seed base against itself at the first, one
   render. It is the C.1 denominator and the only control that is.
3. **One identical-pair decoy.** A clip of the reference arm stacked against
   itself, slot labels drawn as usual, recorded in the sealed key as a decoy.
   Zero GPU; an attention check, not a substitute for item 2. Needs a flag in
   `bench/blind_batch.py` that allows a self-pair (today refused at
   `:271-272`) and a field in `bench/score_session.py`'s output that tallies
   decoys apart from contests.
4. **One low anchor**, the reference with a fixed, recorded post-process
   degradation, aligned by construction. Audio: the 6 kHz low-pass already
   characterised. Video: a short frame-hold stutter. Choose its strength once,
   so hit rates are comparable across batches.
5. **Run `bench/diff_clip_graphs.py --expect` on every contest before
   stacking** (review F10; the tool exists, `bench/blind_batch.py` does not
   call it).
6. **Optional defect tags per half from a fixed list**, replacing the
   meaningless "same" and "can't tell" per-half tags the page reuses today
   (`bench/rubrics/default.json:13`; `docs/eval_comparison.md`, "What the
   judge records", item 4, which records it as a rubric defect). A starting list: the Sol-Engine categories
   ghosting, morphing, flicker, block seams, degraded text, faces or hands,
   blur, colour shift, plus audio ones (muffled, level, garbled words, desync,
   voice change). Free text stays primary. The tags exist so that agreement
   can later be computed per defect class.

### 5.2 Logged beside every verdict, no owner time

Written into the verdict record (or a sidecar keyed by pair), each with its
own floor built the way the audio record built one: identical pair (zero by
determinism), codec round trip, and a reference knob.

| group | metric | valid when |
|---|---|---|
| provenance | graph diff against the declared knob; decoded-stream sha per half | always |
| alignment | per-frame PSNR and SSIM trajectory against dense; audio 20 ms envelope correlation with lag search (the audio record's) | always; this decides the rest |
| video, reference | LPIPS per frame (max and 95th percentile, not mean), DISTS, ColorVideoVDP JOD with one fixed display model, `bench/quality_metrics.py` seams and jitter | aligned pairs only |
| video, descriptive | per-shot delta and detail (`bench/measure_clip_delta.py`) | always, as covariates, never as a verdict |
| audio | the audio record's band grid and loudness set; Whisper transcription error against the scripted line; speaker-embedding cosine between arms per turn; one MOS predictor on dialogue spans; Synchformer offset against the BT.1359 window; the coupling ratio | always; LSE only with its dropped-shot count |
| cost | sampler seconds from the render row | always |

Not logged as a decider: a VLM pairwise judge (Artifact-Bench). If one is run
at all, run it as one more column to be falsified against the owner.

Where it runs [I]: every model above is small next to the DiT, but the card
is shared with the ComfyUI server, and CLAUDE.md treats the server process as
the resource. Run the logging on CPU, or on the card only when no render is
queued and no armed server is up; no runtime was measured for this draft.

### 5.3 Stop rules

- **A batch counts** only if the identical decoy was called *same* or *can't
  tell* and the anchor was noticed. Otherwise record it and do not count it.
- **"Indistinguishable from dense"** (ship the cheaper setting): five scenes
  chosen before the arm existed, no defect-named loss, controls clean. State
  it with its bound: zero losses in five scenes bounds the per-scene loss rate
  below 0.451 at 95%, in ten scenes below 0.259 (3.3). That is the honest size
  of an "indistinguishable" verdict at this n.
- **"Worse"**: two defect-named losses of the same defect class on different
  scenes. One loss can be staging; the same defect twice rarely is.
- **Anything in between** after ten scenes: stop rendering, record "not
  separable at this n", and decide on cost (tier 1 of the review's table).
- **A metric may nominate** candidates only after its agreement with the
  owner's defect-named losses is measured with an interval on logged pairs,
  and it clears chance. Until then it is a column.

## Sources

Repo and local clones (read):
- `docs/eval_comparison.md`; `.claude/skills/h3-ab-session/SKILL.md`;
  `internal/2026-09-19_question_review.md`; `bench/results/2026-09-19_audio_sage_vs_kitchen.md`;
  `bench/blind_batch.py:61-63,271-272`; `bench/rubrics/default.json:13`;
  `bench/quality_metrics.py` (docstring); `bench/diff_clip_graphs.py`
  (docstring); `bench/measure_audio_video_coupling.py` (docstring);
  `docs/roadmap.md:683-691`; `docs/research/2026-09-17_rotation_and_lowbit_attention_survey.md` D.6.
- `coderef/Sana/evals/tiers.toml:12-15,46-51`;
  `coderef/Sana/evals/rubrics/gemini_visual_artifact_gate.md:38-53,80-88`;
  `coderef/Sana/tools/vision/lpips_judge.py:33-46`;
  `coderef/Sana/models/minimax_h3/model.toml:10`.

Papers, full text via extractor:
- AVBench, https://arxiv.org/abs/2605.24652
- VideoScore2, https://arxiv.org/abs/2509.22799 (2025; the evaluator 2026 benchmarks build on)
- VBench, https://arxiv.org/abs/2311.17982 (2023; the benchmark 2026 work reports)
- VBench-2.0, https://arxiv.org/abs/2503.21755 (2025)
- THEval, https://arxiv.org/abs/2511.04520 (v4 2026)
- The Limits of Reference-Free Speech Quality Metrics..., https://arxiv.org/abs/2609.13150
- What Do Audio-Visual Synchronization Metrics Actually Measure?, https://arxiv.org/abs/2608.25157
- Artifact-Bench, https://arxiv.org/abs/2605.18984
- HEART (was HASTE), https://arxiv.org/abs/2605.14513
- Ride the Wave (PASA), https://arxiv.org/abs/2604.12219
- Veda, https://arxiv.org/abs/2605.30325
- Sol-Attn, https://arxiv.org/abs/2607.24027
- Is there a relationship between MOS and JND?, https://arxiv.org/abs/2602.17010
- Early Failure Detection and Intervention in Video Diffusion Models, https://arxiv.org/abs/2603.14320
- VoxSim, https://arxiv.org/abs/2407.18505 (2024; section 4.1 only)

Abstract pages only:
- VABench, https://arxiv.org/abs/2512.09299 (no human alignment claimed in the abstract)
- T2AV-Compass, https://arxiv.org/abs/2512.21094
- PEAVS, https://arxiv.org/abs/2404.07336 (2024)
- ColorVideoVDP, https://arxiv.org/abs/2401.11485 (2024) and README, https://github.com/gfxdisp/ColorVideoVDP/blob/main/README.md
- How Accurate are Video Quality Models for Diffusion-Based VSR?, https://arxiv.org/abs/2605.25940
- Perez-Ortiz and Mantiuk, https://arxiv.org/abs/1712.03686 (2017)
- Prediction-powered inference, https://arxiv.org/abs/2301.09633 (2023)
- AutoEval done right, https://arxiv.org/abs/2403.07008 (2024, revised 2026)
- Safe anytime-valid inference, https://arxiv.org/abs/2210.01948 (2022)
- TTSDS2, https://arxiv.org/abs/2506.19441 (not used for a number)
- Audiobox Aesthetics, https://arxiv.org/abs/2502.05139 (2025)
- DNSMOS, https://arxiv.org/abs/2010.15258; UTMOS, https://arxiv.org/abs/2204.02152
- Wav2Lip (LSE metrics are defined in the paper body, not opened), https://arxiv.org/abs/2008.10010
- FVMD, https://arxiv.org/abs/2407.16124 (2024)
- SVOO, https://arxiv.org/abs/2603.18636
- SenCache, https://arxiv.org/abs/2602.24208
- Disagree to Accelerate (RACER), https://arxiv.org/abs/2608.01740
- TeaCache, https://arxiv.org/abs/2411.19108 (2024)
- LPIPS, https://arxiv.org/abs/1801.03924 (2018; not used for a number)
- VMBench, https://arxiv.org/abs/2503.10076 (2025; correlation unit not established, not used)
- ArtifactLens, https://arxiv.org/abs/2602.09475 (images only; not used)

Code:
- syncnet_python pipeline, https://raw.githubusercontent.com/joonson/syncnet_python/master/run_pipeline.py

Standards and classical statistics:
- ITU-R BS.1534-3 (MUSHRA), via https://en.wikipedia.org/wiki/MUSHRA
- ITU-R BT.1359 thresholds, via https://arxiv.org/abs/2212.01686 (secondary)
- ISO 4120:2021 triangle test, https://www.iso.org/standard/76666.html (tables not opened)
- Davidson 1970, https://doi.org/10.1080/01621459.1970.10481082 (listing only)
- Wald 1945, https://doi.org/10.1214/aoms/1177731118
- Glickman 2025, https://doi.org/10.1177/1471082X251400474 ([?] listing only)

Could not open or verify: the ISO 4120 tables; the BT.1359 text; Glickman
2025; VBench's exact per-dimension correlation range; AVBench per-dimension
instance accuracies other than speech content (shown only in its Figure 13).
