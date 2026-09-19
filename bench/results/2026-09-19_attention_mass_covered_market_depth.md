# Where exact attention mass goes, all 56 heads, covered-market depth capture, 2026-09-19

Tool: `bench/map_attention_mass_on_capture.py`, CPU, read-only on the capture.
Numbers: `bench/results/2026-09-19_attention_mass_covered_market_depth.json`
(per cell, per head). The per-cell tables this record reads come from

    python bench/map_attention_mass_on_capture.py --summarize bench/results/2026-09-19_attention_mass_covered_market_depth.json

## What was measured

Capture `2026-09-19_covered_market_sage_chain_depth` (inventory:
`bench/results/2026-09-19_capture_inventory_covered_market_depth.json`): the
covered-market scene, text to video, blocks 0, 8, 16, 24, 32, 40 and 44 to 49
at steps 8 and 15, every head. Its layout is text, then target audio, then
102 latent frames of video, no references. For a seeded sample of query rows
(text, audio, and an equal number per latent frame of video), the exact fp32
softmax over every key, reduced per head to: mass per key segment; mass on
the keys Sol always attends exactly (the conditioning prefix, rounded up to a
64-block, and the query's own 64-block with its neighbours); mass per latent
frame; and the heaviest keys. Sampled rows, not all rows; one scene; this is
where mass goes, not how much of it Sol loses.

## What it found, in direction

- **Most of a video query's mass is outside Sol's forced ranges, and heads
  differ most at depth.** On most heads, most of the mass lands on keys Sol
  must route or pool: neither the prefix nor the query's own neighbourhood.
  In the deep blocks (44 to 49) the spread across heads is widest: some heads
  put almost all their mass on the prefix and the local band, others almost
  none. That is question C.4's premise holding on this capture: a block is not
  one kind of head. Whether the kernel handles the two kinds differently well
  is not measured here.
- **Video queries attend text more with depth and audio hardly at all.** The
  share of video-query mass on text keys grows from nearly nothing in the
  early blocks to its largest in blocks 44 to 47. The share on audio keys stays
  small at every depth.
- **Audio queries look at video in a few blocks only.** Audio queries keep
  most of their mass on audio keys, except at block 8, where most of it goes
  to video, and blocks 32 and 40, where a substantial share does. In the deep
  blocks they barely look at video. So cross-modal attention from sound to
  picture is concentrated in a few blocks.
- **The heaviest single keys sit in the forced prefix in the deep blocks, and
  in the video span in the middle.** At blocks 44 to 49 nearly all of each
  head's heaviest keys are text or audio rows, which Sol already attends
  exactly. At blocks 8, 32 and 40, half or more of them are video rows.
- **Every fifth latent frame draws more mass: the period-5 hypothesis holds on
  average.** Latent frames whose index is a multiple of 5 draw more incoming
  mass from video queries than the average frame, in every cell but block 0,
  and residue 0 tops a head's frame curve more often than the one-in-five a
  flat curve gives in every cell but block 0 (by a wide margin in blocks 16,
  32 and 40, narrowly in block 48). The control is what makes this a finding:
  residues 1 and 3 sit below average in almost every cell, so it is not
  binning. Residue 2 is the one other class lifted, in blocks 32 and 40 and at
  step 15 in block 24. Among the heaviest video keys, frames at multiples of 5
  are over-represented in blocks 8 and 16 and at step 15 in blocks 32 and 40,
  but not at step 8 in those two.
- **The first latent frame is not a general sink here.** Only a minority of
  heads give frame 0 markedly more than the average frame's mass.

## Reading, labelled as inference

The latent frames at multiples of 5 are the ones whose RoPE time span is one
pixel frame against four for the rest (`comfy/ldm/minimax/model.py`,
`FRAME_PER_TOKEN`), which matches a video VAE that starts every 17-frame chunk
with a single-frame latent (1 plus 4 times 4). If so, those latents are the
chunk-initial frames, and the model treats them as anchors. That is the
mechanism this suggests; nothing here tests it.

What it could mean for Sol: those frames are whole, contiguous runs of
blocks, so a router could be told to favour them. Whether Sol's block-mean
router already routes them, or pools mass it should not, is the next
question, and it is answerable on the same capture: the share of Sol's
missed mass that falls on residue-0 frames, against their share of keys. It
belongs in the block-grouping tool, not in a kernel change, until that says
it matters.

## Not measured here

How much mass Sol actually misses on these heads (this is the exact
distribution only); any capture with reference rows (none exists); any other
scene; the distilled checkpoint.
