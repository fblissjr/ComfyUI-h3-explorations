# Which scored pairs are the same take: picture and sound alignment, 2026-09-19

Tool: `bench/measure_pair_alignment.py` (CPU, ffmpeg; no GPU). Numbers:
`bench/results/2026-09-19_pair_alignment.json`, one row per pair with the
graph fields that differ, per-frame picture summaries and the sound envelope
correlation. List it with:

    python3 -c "import orjson; [print(r['label'], r['verdict'], r['sound_verdict'], round(r['picture']['ssim_luma']['mean'],3)) for r in orjson.loads(open('bench/results/2026-09-19_pair_alignment.json','rb').read())['pairs']]"

## Why

A reference metric against another render (PSNR, SSIM, LPIPS, ColorVideoVDP,
an audio band grid) measures what a knob did only when the two clips are the
same take. When the knob moved the sample into another performance, the same
number measures the difference between performances
(`docs/research/2026-09-19_evaluation_one_judge.md`, item 2). The lead's
dense-against-default panel needs to know, per contest, which regime it is in.

## What was compared

Every pair the owner has scored or listened to where both clips are on disk,
all at matched frame counts, B against A:

- controls: a clip against itself; the 2026-09-15 diner default against its
  2026-09-18 re-render, which the rebuild record found bit-identical
  (`bench/results/2026-09-18_kitchen_0.2.35_rebuild.md`); two two-seed pairs
  of one arm on the 107-frame noodle bar, plain and `3d`;
- the reorder panel's plain-against-`3d` pairs as that record defines them
  (`bench/results/2026-09-18_sol_reorder_panel.md`), plus `market_v2` and
  `cafe_kids_v2`; the short-clip panel; the two on-length pairs;
- the chain pairs of `bench/results/2026-09-19_audio_sage_vs_kitchen.md`
  (meerkat and diner clean, and the kitchen-scene pair it showed to be sage
  against sage);
- fully dense against the other arms, where a dense render exists (crowd
  churn, meerkat).

## What it found, in direction

- **The tool can fail, and the controls land where they must.** The self pair
  and the three-day re-render read `identical` in picture and sound; both
  two-seed pairs read `two_takes` and `diverged`, far below every non-control
  pair. During development a pairs file expecting the wrong verdict for the
  self pair made the run exit 1.
- **Most knob pairs on full-length scenes are not the same take.** Token
  order, the dense chain and fully dense against the default all keep a
  scene's staging and cast, but on most scenes they move figures, poses or
  framing enough that luma SSIM stays well under the line. The pairs that
  stay aligned in picture are the meerkat scene (every arm, dense included),
  post office, night porter and swimming lesson. On crowd churn, fully dense
  against either default arm is two takes.
- **For the dense-against-default panel:** a reference picture metric
  against the dense clip is readable on a scene only if this tool says
  `same_take` for that contest. On this evidence, expect that on a minority
  of scenes, and choose which metric to log per contest accordingly.
- **Picture and sound come apart, in both directions.** Several pairs keep
  their sound performance while the picture moves (hardware aisle, market,
  diner, crowd churn, the noodle bar, the diner chain pair); several keep the
  picture while the sound diverges (post office at both lengths, night
  porter, the meerkat default arms against dense). So the two verdicts are
  reported apart, and an audio metric is readable where the sound verdict is
  `aligned`, whatever the picture says.
- **Independent agreement with the audio record.** The envelope correlations
  for the two clean chain pairs match the values the 2026-09-19 audio record
  computed with its own code.

## The threshold, and how it moved

`SAME_TAKE_MIN_LUMA_SSIM` is reasoned, not measured against a judge. Its first
value sat between a re-sampled pair (`cafe_kids`, whose sound had diverged
into a different performance) and the audio-aligned pairs. A visual check of
side-by-side frames on five pairs (hardware aisle, market, `market_v2` at two
times, `cafe_kids`, `cafe_kids_v2`) showed that below the current line the
pairs share staging and cast but not positions: the same hands with the bag
in another pose, the same stall worker shifted across the frame, the same
children in swapped seats, and on `market_v2` a different shot altogether by
mid-clip. A reference metric on such a pair measures the pose. The line moved up to where the aligned pairs start. The sound threshold
is inherited from the audio record. Neither has been checked against the
owner's sense of "the same take"; the tool reports the per-frame fraction
above the line (`aligned_frame_fraction`) and the minimum frame, so a reader
can see a pair that aligns in one shot and not the next.

Not settled here: whether any of this predicts a verdict. It says which
pairs a distance can be read on, not what the distance means.
