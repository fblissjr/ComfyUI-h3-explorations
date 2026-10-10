# The MiniMax-M3 to H3 conditioning bridge: what, why, how, and why it probably will not work

Written 2026-09-15 on the H3 side, from the code, the captured files and the
dated records named below. The sister document is the heylook checkout's
`docs/architecture/m3_activation_capture_plan.md` (the owner's inference-server
repo), which owns the capture side: how M3 is loaded and read, how the Qwen
targets are produced on the Mac, and the status log of every capture. This
document owns the consumer side: what H3 actually eats, what the adapter has
to produce, what has been measured, and the case against the whole idea. Where
the two disagree, section 11 says so.

Prose here carries pointers; the numbers live in `bench/results/` records
named inline. When a sentence here and a record disagree, the record is right.

## 1. The question, in two claims

Can a small learned adapter turn MiniMax-M3's decoder representations into the
conditioning the released MiniMax-H3 video model expects, well enough that H3
accepts it (**compatibility**), and, separately, does conditioning that way do
anything ordinary Qwen conditioning cannot (**value**)?

Both encoders stay frozen. Only the adapter trains, on captured files, on
consumer hardware. The two claims are kept apart because the first is a
prerequisite for the second and proves nothing about it: an adapter trained
to reproduce Qwen's states is rewarded for being Qwen, and any information M3
has that Qwen lacks is, under that loss, noise.

## 2. Why anyone would try

- **H3's conditioning is a frozen LLM's hidden state**, not a learned text
  embedding. The DiT was trained against Qwen3-VL-32B's layer-50 residual
  (section 3). A hidden state is a coordinate system another model could in
  principle be mapped into.
- **M3 is multimodal and larger.** If a later multimodal model reads a
  reference image and a prompt together better than the pinned Qwen does, a
  bridge could carry that into H3 without retraining the DiT.
- **One asymmetry was observed on the first smoke captures**: swapping the
  image moved M3's later text rows far more than it moved Qwen's, at their
  respective feature points and quantisations, on one pair (the heylook
  checkout's `internal/research/` notes dated 2026-09-13 and 2026-09-14;
  gitignored there). That is the one measured hint that M3 carries image
  information into text where Qwen barely does. It is one pair.
- **A hypothesis about a hosted service** using a different encoder motivated
  the idea and remains unverified. Nothing here depends on it and nothing here
  tests it.

## 3. What H3 consumes

Every item is read from the installed ComfyUI checkout, not from a model card.

- **Feature point.** The text encoder is Qwen3-VL-32B truncated to fifty
  layers, no final norm, no head: `Qwen3VL_32BConfig` in the checkout's
  `comfy/text_encoders/llama.py` hardcodes `num_hidden_layers`, `final_norm`
  and `lm_head`, and the shipped int8 file carries exactly fifty language
  layers and no top-level norm weight (`h3_config.MODELS["clip"]`). The
  conditioning is the raw residual after block fifty, width 5120.
- **Presentation.** No chat template. T2VA is the prompt bytes alone; a first
  frame adds `<Picture 1>: `, a vision-start marker, one pad per merged patch
  and a vision-end marker before the prompt (`comfy/text_encoders/minimax.py`,
  module docstring). The seven H3 markers (`<d>`, `</d>` and five others) are
  single ids fixed by the release tokenizer; `docs/evidence.md` "The marker
  ids are the release's own" and
  `data/results/2026-08-27_marker_tokenization_alignment.json` are the
  record. A capture made with the stock Qwen tokenizer splits `<d>` into three
  tokens and, because the encoder is causal, every row after the first marker
  is then a different computation from the one the DiT sees.
- **Token tags.** Each row carries a modality tag (visual or text) that the
  DiT uses to pick adaLN modulation rows; the two vision markers are tagged
  visual, so the visual row count is the pad count plus two per image.
- **What the DiT does with the rows.** `preprocess_text_embeds` in
  `comfy/ldm/minimax/model.py` applies `condition_proj` (a linear map from
  5120 to the DiT width) and a two-block token refiner with its own attention
  over the text rows, then packs the result ahead of the video and audio
  tokens. Both modules are bf16 in the shipped int8 DiT. This is the
  consumer's eye: the only view of the conditioning the DiT ever has.
- **Prefix length is a coordinate origin.** The packed layout sets the
  temporal origin of the video and audio grids at the end of the text span
  (`PackedLayout`), so any bridge that changes the row count moves the whole
  target timeline. The adapter must emit exactly the native row layout.
- **Pixels.** A first frame is stretched to the canvas and the same tensor
  goes to the encoder and the VAE (`comfy_extras/nodes_minimax_h3.py`,
  `MiniMaxH3ImageToVideo.execute`). The encoder's own image processor then
  patchifies at its defaults; the merged grid for the contract canvas is
  recorded per example in the image bank's expected-ids files (gitignored,
  `internal/claude/m3-h3-image-bank/inputs/`), not typed here.

## 4. What M3 provides

Owned by the sister document; summarised here so the mapping problem is
visible from one page.

- **Feature point.** The pre-norm residual after fifty-nine of sixty blocks
  (`l_out-58`), width 6144, captured on the Mac by a standalone tool on
  llama.cpp's evaluation callback, which is the last tensor that still
  carries every row including the image rows the stock embeddings endpoint
  drops. Repeat and micro-batch captures are bit-identical on that build.
- **Presentation.** Raw bytes, no chat template: for text the prompt alone;
  for an image, the media marker, a newline, then the prompt. The capture
  wraps the image chunk in an image-start row before and an image-end row
  after it.
- **Visual grid.** Not readable from the runtime, which stores a flat token
  count for this projector; the capture tool derives it from the projector's
  own resize rule and refuses a capture whose derived grid does not multiply
  out to the chunk's rows. On the 1344 by 768 canvas the grid is 31 by 18,
  row-major with x fastest, on a resized 868 by 504 image.
- **Quantisation.** The artifact is an aggressive integer quant with an fp16
  projector; Qwen on the Mac is 8-bit. Both are fixed across train and test.
  Neither is the released bf16 weight, and the tolerance in section 6 is
  what says how much that matters.

## 5. Alignment: two of three parts are not open

The handoffs named alignment as the open design problem. For the text rows
and the visual rows it is not; the open part is smaller than it looked.

- **Text rows align by byte offset.** Both tokenizers are files, both are on
  this box, and both reproduce the captured ids byte for byte from the prompt
  bytes. Each Qwen row's byte span is covered by one M3 row for most rows, by
  part of one for the rest, and by a few rows for a handful. The rule used is
  "read the last M3 row overlapping the span", which is causal-consistent on
  both sides. This exists at inference without either decoder.
- **Visual rows align by normalised grid coordinate.** Both grids sit on the
  same prepared pixels and share the aspect ratio, so a Qwen pad at a given
  fraction of height and width reads the M3 cell at that fraction. Whether
  that correspondence carries content is the image lane's first question
  (section 7).
- **What is open**: rows with no counterpart (Qwen's picture label and two
  markers against M3's image-start and image-end rows), whether one cell is
  the right receptive field for a pad or several are, and whether any of
  this survives a different M3 feature point.

A resampler that learns free cross-attention from target queries to all
source rows is therefore the second experiment, not the first: it has to beat
the deterministic rule, and on the text lane it did not (section 7).

## 6. The metric, and what "accepted" means

Compatibility as stated, "H3 accepts it", has no tolerance attached, and
without one every number is unreadable. Two decisions fix that.

- **Score at the consumer's eye.** Error is measured after `condition_proj`
  and the token refiner, on the whole sequence, not in the raw 5120 space.
  The raw space is dominated by an attention-sink first row two orders of
  magnitude above the median and a handful of massive-activation channels
  (per-scene norms in the records below); a raw MSE spends its capacity on
  what the refiner normalises away.
- **The tolerance is what two legitimate Qwen runtimes already disagree by.**
  The shipped int8 encoder on this box and the Mac's 8-bit MLX capture were
  run on the same eleven prompts and compared on each scene's common id
  prefix, with the refiner fed the prefix on both arms:
  `data/results/2026-09-14_m3_bridge_runtime_tolerance_int8_vs_mlx8.json`.
  A bridge inside that distance is "as good as another Qwen"; outside it, it
  is a worse Qwen by a stated factor. The record also shows the runtimes'
  first-row norms differ by a small constant factor on every scene, which
  looks like a leading-token runtime difference and is unexplained.

## 7. What has been measured

Arms and controls are defined in the adapter's README under this checkout's
gitignored `internal/claude/m3-h3-adapter/`; the records are tracked.

**Text lane** (eleven T2VA scenes from `prompt_bank/`, eight train and three
held out by scene family, fixed before capture):

- `data/results/2026-09-14_m3_bridge_text_lane_mlx_targets_seed0.json`
  and `_seed1.json`: against the Mac's stock-tokenised targets.
- `data/results/2026-09-14_m3_bridge_text_lane_comfy_targets_seed0.json`:
  the same arms against this box's encoder states with the release
  tokenizer, which is the DiT's real input.

Quoted from those records (held-out scenes, content rows, relative L2 after
the refiner, lower is better; the two target sides agree to the second
decimal):

| arm | held-out |
|---|---:|
| global mean | 0.91 |
| token-id mean | 0.73 |
| Qwen input-embedding ridge (no M3) | 0.67 |
| aligned M3 ridge | 0.59 |
| aligned M3 + embedding ridge | 0.56 |
| aligned MLP | 0.55 |
| M3 ridge, source shuffled | 0.91 |
| M3 ridge, another scene's source | 0.87 |
| resampler as contracted (hashed queries, free cross-attention) | 0.74 |
| resampler, query only | 0.73 |
| runtime tolerance, two Qwen runtimes (section 6) | 0.02 to 0.03 |

Three readings, each of which the Mac side checked against its captures:

1. **M3 carries transferable text signal.** Given the byte alignment, a
   plain ridge map beats every no-M3 arm on held-out content rows, and both
   source controls fall to the floor.
2. **The contracted resampler did not use M3.** It scores the same as its
   own query-only ablation; with hashed queries and eight scenes it learned
   the prompt template. This is the stop condition written into the design
   note that proposed it.
3. **Nothing is within tolerance.** The best arm is more than an order of
   magnitude outside the distance two Qwen runtimes agree to. By the only
   standard that means "accepted", no text-lane arm is compatible.

**Image lane** (twelve I2VA scenes from the owner's stills, fifteen examples
with three flipped-frame variants as the image-dependence control, captured
and validated on both sides): the fit is written and unrun. `docs/wiki/next_steps.md`
carries the pointer and the three readings it must produce: visual pad rows
against a per-position mean and a shuffled source; post-image text against
the text lane's number; and the flipped pairs, where reading the unflipped
frame's M3 cell mirrored across x should score like the matched frame if the
map is spatial, and like the wrong image if it reads the rows as a bag.

## 8. Why it probably will not work

In order of how much each one costs the idea.

1. **The gap to tolerance is not a tuning gap.** The best cheap map is more
   than an order of magnitude outside what two Qwen runtimes disagree by
   (the two records in section 7), on the easiest lane, with the alignment
   handed to it. Closing that with a few thousand
   rows of supervision would require the two residual spaces to be related
   by something simpler than they are.
2. **A loss that rewards being Qwen cannot reward being better than Qwen.**
   Compatibility trains the adapter toward the teacher; value would need a
   downstream objective through the DiT, which is a different and far more
   expensive project. A compatible bridge with nothing extra is a slower
   Qwen.
3. **The value claim has a cheaper baseline it has not beaten.** M3 writes
   the prompt, ordinary Qwen encodes it. No adapter, no capture, no
   injection. Until a scene family exists where that path fails and a bridge
   succeeds, the bridge has no job.
4. **The DiT reads a subspace the loss does not know.** Error after the
   refiner is the right metric, but the refiner is two blocks and the DiT
   is fifty; directions the DiT is sensitive to may be small at the refiner
   output and large in the render. A feature-space win can still render
   wrong, and a rendered clip cannot A/B a numerical change (`CLAUDE.md`).
5. **The data budget is a handful of scenes sharing one house style.** Every
   bank prompt opens with the same field names, shot markers and dialogue
   tags; template rows are scored separately for that reason, and the neural
   arms memorise the train set at a few thousand rows. More scenes cost a
   Mac capture window each and do not change point 1.
6. **Both encoders are quantised and neither is the released weight.** The
   tolerance record bounds how much that matters for Qwen; nothing bounds it
   for M3, whose quant is far more aggressive.
7. **The image rows are the only place value could come from, and they are
   the hardest part.** Different projectors, different grids, different
   receptive fields, no shared training. If the visual rows map no better
   than the position-mean and shuffled controls, the lane that carried the
   motivation is closed.

## 9. What would change that verdict

- **The image lane's visual rows map**, the flipped-pair check shows the map
  reads the picture, and the error is within a small factor of the text
  lane's. That would say the projector output is usable and move the
  question to point 2 above.
- **A scene family where M3-written text through Qwen fails** and the
  bridged conditioning does not, on matched seeds, judged the way
  `docs/eval_comparison.md` requires. That is the only shape a value claim
  can take.
- **A different M3 feature point** (final-norm rows, or an earlier layer)
  closing a large part of the tolerance gap on the text lane. One Mac
  capture pass with the existing runner tests it; nothing measured says it
  would.

## 10. Stop rules

Held to since 2026-09-14:

- The text lane is done. It did not fail, and it does not need another fit.
- The image lane stops if the visual pad rows score no better than the
  per-position mean and the shuffled source after the refiner.
- The project stops, or pivots to the text-mediated baseline, if no value
  claim can be stated as a behaviour on a named scene family that
  M3-written text through ordinary Qwen cannot produce.
- No capture window is spent on a larger bank until one of the readings in
  section 9 exists.

## 11. Where the sister document and this one disagree

Checked against the heylook plan document at its commits of 2026-09-14.
None of these is a contradiction of fact; each is a place where one document
carries something the other should.

- **The tolerance is measured.** The sister document's night status names
  it as the H3 side's next step; the record is now
  `data/results/2026-09-14_m3_bridge_runtime_tolerance_int8_vs_mlx8.json`
  and the meaning of "accepted" in section 6 follows from it.
- **The value baseline is absent there.** Its experiment statement ends at
  "whether the bridge improves H3"; this document's point 3 in section 8
  names the cheaper path that has to lose first.
- **"The bridge, elsewhere, on whatever machine Codex chooses"** is stale:
  the bridge is fit on the H3 side, on files, and Codex left the project on
  2026-09-14.
- **The feature table there still names the final-norm output** as the plan's
  M3 feature, while every capture used the pre-norm residual; the document
  says so further down, and the capture profile pins it. A reader who stops
  at the table gets the wrong feature.
- **The marker finding is described as a third declaration site** there and
  as "no file assigns them an id; the loader assigns them by sequential
  append" here (`docs/evidence.md`). Both are true of the release files; the
  operative fact both agree on is that the release tokenizer directory yields
  single ids and the stock one does not.

## 12. Where things live

- **This checkout**: the consumer-side code and records. Tracked: this
  document, the four `bench/results/2026-09-14_m3_bridge_*` records, the
  pointer in `docs/wiki/next_steps.md`. Gitignored, under `internal/claude/`:
  the take of 2026-09-14 with its results section and next-session list, the
  adapter code and its per-run JSON, the image-bank builder and its inputs;
  under `internal/codex/`: the pilot contract, the text-bank manifest and the
  T2VA pair validator.
- **The heylook checkout**: the capture instruments, the captured pairs for
  both banks with their receipts, and the plan document. Its `internal/`
  tree is gitignored too; the owner syncs the two.
- **Division of labour**: the Mac runs the large models, one at a time, and
  captures; this side owns the H3 judgement, the prompts, the pixels, the
  metric and every fit. Sessions on the two machines cannot message each
  other; the owner relays.
