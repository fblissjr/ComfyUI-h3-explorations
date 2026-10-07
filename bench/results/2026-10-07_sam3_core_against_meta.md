# ComfyUI's SAM 3.1 against Meta's code, stage by stage on equal inputs (2026-10-07)

lane: masked
verdict: with the same weights, ComfyUI's port tokenizes as Meta's code does and its trunk and detector agree with Meta's on an equal input to the size of float32 against bf16; it departs in two places before the model, the range of the image its nodes hand over and the activation its text encoder runs; the trackers agree where every run agrees (one person alone is lost within a few frames in both, the first sixteen hold in both) and no difference beyond sixteen people is established, because the same arm ends with every person or a dozen fewer on frames two brightness levels apart

**What was asked.** The owner, 2026-10-07: whether ComfyUI's native SAM 3.1
is correct, "down to the tokenizer", traced end to end. ComfyUI's port
(`comfy/ldm/sam3`, `comfy_extras/nodes_sam3.py`,
`comfy/text_encoders/sam3_clip.py`) is a re-implementation, so "correct" here
means "computes what Meta's released code computes from the same weights".

**How.** [`bench/sam3_parity_ladder.py`](../sam3_parity_ladder.py) runs
ComfyUI's port beside the copy of Meta's inference code under
[`meta_sam3/`](../../meta_sam3/README.md), one stage at a time, each stage
given an equal input so a difference found lower down does not leak upward.
Both sides load the one weights file `workflows/h3_config.py::SEGMENTER`
names. ComfyUI computes in float32 with TF32 off for the text, trunk and
detector rungs; Meta's code under the bf16 autocast it enters itself, so a
difference of that size is the floor of this comparison and not a finding.
The tracker rung compares outcomes, not tensors, with ComfyUI at its default
precision. Outside the server, the card otherwise idle. Every number is in
[`2026-10-07_sam3_core_against_meta.json`](2026-10-07_sam3_core_against_meta.json)
and every table below is printed from it by the tool's `render`; where a
sentence and the file disagree the file is right.

**What it leaves out, on purpose.** Nothing here says what a clip shows. A
clip is its file name and its technical facts. One phrase asked of a frame
was chosen to describe something in it; its text is in no file.

**Compute precision is not part of this record.** It was measured
separately, by an independent session, with floors and a yardstick:
[`2026-10-07_sam3_precision_arms.md`](2026-10-07_sam3_precision_arms.md).
That record also found the two departures below without this record's
notes, and a third this ladder does not run: ComfyUI's detect node
thresholds the class score alone and removes no overlapping detections,
where Meta's code thresholds the class score multiplied by the presence
score and removes them.

## The rungs

### Tokens

31 prompt parts from 29 typed phrases; 0 differ.

### The two weights files

`sam3.1_multiplex_fp16.safetensors` against `sam3.1_multiplex_fp32.safetensors` cast to float16: 1590 tensors equal, 0 differ, 0 differ in shape; only in theirs 0, only in ours ['detector.backbone.language_backbone.encoder.text_projection'].

### Text encoder

Core's 24 text MLPs ship quick_gelu: True. Relative L2 of core's features against Meta's, real tokens only.

| phrase | tokens | encoder, as shipped | encoder, exact GELU | after the resizer, as shipped | after the resizer, exact GELU |
|---|---|---|---|---|---|
| `person` | 3 | 0.04818 | 0.0026 | 0.03706 | 0.00337 |
| `head` | 3 | 0.06481 | 0.00278 | 0.04233 | 0.00332 |
| `a person` | 4 | 0.06024 | 0.00247 | 0.04677 | 0.00351 |
| `person wearing a hat` | 6 | 0.14886 | 0.00906 | 0.11074 | 0.00762 |
| `the tallest person` | 5 | 0.14784 | 0.00734 | 0.11928 | 0.00663 |
| `man in a red shirt` | 7 | 0.10987 | 0.00469 | 0.07871 | 0.00454 |
| `hair` | 3 | 0.08571 | 0.00363 | 0.06019 | 0.00402 |

One detect on `thrill_2160.mkv` at 56.5 s, the frame in the trained range.

| phrase | detections, as shipped | exact GELU | presence, as shipped | exact GELU | lowest best IoU between the two sets |
|---|---|---|---|---|---|
| person | 2 | 2 | 0.9744 | 0.9718 | 1.0 |
| head | 1 | 1 | 0.9855 | 0.9839 | 1.0 |
| a describing phrase (text not recorded) | 3 | 3 | 0.061 | 0.0319 | 0.994 |

### Image range and trunk

`thrill_2160.mkv` at 56.5 s, 1024x768. Core's node hands its trunk [0.0, 1.0]; Meta's loader hands its trunk [-1.0, 1.0]. Patch embedding: weights differ by 0.0, bias in core False, in Meta False.

| trunk features, last level | relative L2 | cosine |
|---|---|---|
| equal tensor core vs meta | 0.04907 | 0.99922 |
| core as its node feeds vs meta | 0.65114 | 0.78773 |
| core resize in metas range vs meta | 0.21544 | 0.97722 |
| meta given the unmapped image vs meta | 0.65054 | 0.788 |
| core as fed vs meta given the same unmapped image | 0.03654 | 0.99975 |

### Detector, on Meta's image tensor

`thrill_2160.mkv` at 56.5 s, phrase `person`, 200 queries.

| core's text | presence logit core / Meta | joint score logits: relative L2, cosine | kept over 0.5 core / Meta | top ten by Meta: Meta | the same queries: core | their mask IoU |
|---|---|---|---|---|---|---|
| exact GELU | 3.4538 / 3.4375 | 0.09377, 0.99562 | 2 / 2 | [0.957, 0.554, 0.492, 0.094, 0.067, 0.056, 0.055, 0.055, 0.054, 0.051] | [0.959, 0.561, 0.456, 0.092, 0.07, 0.059, 0.053, 0.079, 0.038, 0.046] | [0.999, 1.0, 0.997, 0.833, 0.992, 0.995, 0.972, 0.686, 0.365, 0.998] |
| as shipped | 3.5553 / 3.4375 | 0.09509, 0.9955 | 2 / 2 | [0.957, 0.554, 0.492, 0.094, 0.067, 0.056, 0.055, 0.055, 0.054, 0.051] | [0.962, 0.565, 0.459, 0.093, 0.069, 0.059, 0.055, 0.08, 0.039, 0.046] | [0.999, 1.0, 0.997, 0.84, 0.994, 0.995, 0.972, 0.697, 0.396, 1.0] |

### Tracker, from the same masks: seeds from one detect by core on the first frame, as its node gets it

`vma.mp4` from 121.93 s, 72 frames at 24.0 a second, 1344x760; 64 detections on the seed frame. Counts of seeded people with a non-empty mask. With several people meta's tracker-only path applies neither of its two between-people rules; core applies both.

| arm | seeded | with a mask at frames 0, 18, 36, 54, 71 | the last frame each is on | the lone person's object score, first frames |
|---|---|---|---|---|
| core, frames as its node gets them | alone | [1, 0, 0, 0, 0] | [3] | [10.0, 4.45, 2.53, 0.31, -0.39, -1.77, -2.47, -2.5] |
| core, frames as its node gets them | with one neighbour | [2, 1, 0, 0, 0] | [16, 18] |  |
| core, frames as its node gets them | 17 | [17, 16, 16, 16, 16] | [5, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71] |  |
| core, frames as its node gets them | 32 | [32, 32, 31, 19, 16] | [32, 38, 38, 38, 39, 40, 40, 41, 43, 50, 51, 51, 56, 60, 62, 64, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71] |  |
| core, frames in the trained range | alone | [1, 0, 0, 0, 0] | [3] | [10.0, 4.52, 2.0, 0.02, -0.53, -1.79, -2.57, -2.6] |
| core, frames in the trained range | with one neighbour | [2, 2, 2, 2, 2] | [71, 71] |  |
| core, frames in the trained range | 17 | [17, 16, 16, 16, 16] | [6, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71] |  |
| core, frames in the trained range | 32 | [32, 32, 32, 22, 20] | [38, 50, 50, 51, 52, 52, 52, 54, 55, 56, 61, 61, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71] |  |
| Meta's tracker | alone | [1, 0, 0, 0, 0] | [2] | [10.0, 4.19, 1.48, -0.49, -1.9, -2.5, -3.02, -2.88] |
| Meta's tracker | with one neighbour | [2, 2, 2, 2, 2] | [71, 71] |  |
| Meta's tracker | 17 | [17, 16, 16, 16, 16] | [6, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71] |  |
| Meta's tracker | 32 | [32, 32, 32, 31, 31] | [66, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71] |  |

### Tracker, from the same masks: seeds from one detect by core on the first frame, in the trained range

`vma.mp4` from 121.93 s, 72 frames at 24.0 a second, 1344x760; 63 detections on the seed frame. Counts of seeded people with a non-empty mask. With several people meta's tracker-only path applies neither of its two between-people rules; core applies both.

| arm | seeded | with a mask at frames 0, 18, 36, 54, 71 | the last frame each is on | the lone person's object score, first frames |
|---|---|---|---|---|
| core, frames as its node gets them | alone | [1, 0, 0, 0, 0] | [2] | [10.0, 2.97, 1.27, -0.08, -1.63, -2.67, -2.8, -2.82] |
| core, frames as its node gets them | with one neighbour | [2, 0, 0, 0, 0] | [8, 9] |  |
| core, frames as its node gets them | 17 | [17, 16, 16, 16, 16] | [12, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71] |  |
| core, frames as its node gets them | 32 | [32, 32, 22, 16, 16] | [30, 31, 32, 32, 32, 32, 32, 32, 34, 36, 36, 37, 37, 38, 39, 40, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71] |  |
| core, frames in the trained range | alone | [1, 0, 0, 0, 0] | [2] | [10.0, 2.66, 0.74, -0.57, -2.05, -3.02, -3.16, -3.48] |
| core, frames in the trained range | with one neighbour | [2, 0, 0, 0, 0] | [8, 8] |  |
| core, frames in the trained range | 17 | [17, 16, 16, 16, 16] | [17, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71] |  |
| core, frames in the trained range | 32 | [32, 32, 31, 32, 18] | [56, 56, 56, 57, 57, 59, 60, 60, 60, 60, 61, 62, 63, 64, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71] |  |
| Meta's tracker | alone | [1, 0, 0, 0, 0] | [2] | [10.0, 2.64, 0.82, -0.33, -1.76, -2.64, -2.67, -2.97] |
| Meta's tracker | with one neighbour | [2, 0, 0, 0, 0] | [8, 8] |  |
| Meta's tracker | 17 | [17, 16, 16, 16, 16] | [8, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71] |  |
| Meta's tracker | 32 | [32, 32, 32, 32, 32] | [71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71] |  |

### The same core arms on frames decoded another way

The same window, seeds and core arms from a scratch probe run earlier the same day, not by this tool; its frames were decoded by calling ffmpeg directly (fps, crop, scale) where the tool uses the loader node. Seeds: one detect by core on the first frame as its node gets it. The two sets of frames are the same frames in time and differ by 1.74 to 1.8 levels of 255 on average.

| | core, frames as its node gets them | core, frames in the trained range |
|---|---|---|
| 32 seeded with a mask at the last frame | 16 | 32 |
| 17 seeded last frame the seventeenth is on | 14 | 71 |
| alone frames on | 5 | 3 |

## What this establishes

Each line says how it is known.

- **Tokens: the same** (measured, CPU). Every prompt part of the list in
  the tool's `TOKEN_PHRASES`, with ComfyUI's count syntax parsed by its own
  tokenizer, gives the ids Meta's tokenizer gives at the model's context
  length.
- **The ComfyUI-org weights file is ours cast to float16** (measured, every
  tensor). It carries no folded normalisation, so the range departure below
  is the same with either file.
- **Text encoder: departs, and the activation is the whole of it**
  (measured). As shipped the features differ from Meta's by the "as
  shipped" columns; with the activation switched in memory to exact GELU
  they agree to the floor. For the one-word phrases the lane ships the
  detections on one frame are the same either way; for a describing phrase
  the presence score halves.
- **Image range: departs** (measured at the trunk's input, and read:
  `comfy_extras/nodes_sam3.py` resizes and does nothing else, Meta's
  `io_utils.py` maps to -1..1). The patch embedding has no bias on either
  side and equal weights, so nothing downstream can absorb it. Meta's own
  trunk given ComfyUI's unmapped image is as far from itself as ComfyUI's
  is, and ComfyUI's trunk given the unmapped image matches Meta's given the
  same: the port's arithmetic is right and its input is not.
- **Trunk: the same on an equal tensor** (measured), to the floor.
- **Detector: the same on what it is confident about** (measured, one
  frame, one phrase). The presence logit agrees to the second decimal, the
  detections both sides keep score within a hundredth and their masks
  coincide; among low-scoring queries neither side keeps, scores and masks
  wander, as float32 against bf16 would make them.
- **Tracker, where every run agrees** (measured, one dense window, two seed
  sets, and the earlier run on other frames): the largest person followed
  alone is lost within a few frames in ComfyUI's port and in Meta's tracker
  alike, with the same falling object score; the first sixteen of a seeded
  set are all still there at the end in every arm; and given the same seeds,
  ComfyUI with the range mapped and Meta's tracker end the one-person and
  the two-person arm on the same frame or one apart. A seventeenth person,
  alone in a second group, is lost early in every arm: on the same frame in
  the two on one seed set, nine frames apart on the other.

## What it does not establish

- **Any difference between the trackers beyond sixteen people.** The last
  table is the reason: the same arm, same seeds, same window, ends with
  every person on frames decoded one way and a dozen fewer on the same
  frames decoded another, the two a couple of brightness levels apart.
  Meta's tracker kept nearly all of 32 in both its runs and ComfyUI's about
  half in four, which is a hint and no more: one run per arm, no repeat
  under a small change of input, and Meta's tracker-only path applies
  neither of the two rules its full pipeline applies between people while
  ComfyUI applies both. The session that wrote this record claimed more
  than this twice during the day, on the owner's board, and withdrew it
  both times.
- **That a neighbour rescues a person lost when followed alone.** It did
  with one seed set and not with the other, alike in all three arms.
- **What mapping the range is worth for tracking.** The fault is certain; a
  benefit shows in some arms here and is inside the noise in others.
- **Which input the model does better on.** No picture was judged.
- **The tracker stage by stage**, the detector beyond one frame and one
  phrase, Meta's full session logic (its association, confirmation and
  refresh rules), or any clip but the two named.
- **Nothing here is changed in this pack.** No node maps the range or
  switches the activation today.

## Reproducing it

From ComfyUI's environment, outside the server, the card free, with `T` for
`bench/sam3_parity_ladder.py` and `J` for a json path of your own:

```
<python> T tokens --json J
<python> T files --theirs <the ComfyUI-org file> --ours <ours> --json J
<python> T text --clip thrill_2160.mkv --second 56.5 --width 1024 --json J --describing "<a phrase>"
<python> T trunk --clip thrill_2160.mkv --second 56.5 --width 1024 --json J
<python> T detector --clip thrill_2160.mkv --second 56.5 --width 1024 --json J
<python> T tracker --clip vma.mp4 --second 121.93 --seconds 3 --width 1344 --rate 24 --seeds-from node --json J
<python> T tracker --clip vma.mp4 --second 121.93 --seconds 3 --width 1344 --rate 24 --seeds-from trained --json J
<python> T render --json J
```

## Files

- `2026-10-07_sam3_core_against_meta.json`: one key per rung as the tool
  wrote it, each with its environment (the ComfyUI commit, torch, the TF32
  flags); `earlier_run_other_decode` holds the few counts of the earlier
  scratch run the last table names, with the measured difference between
  the two sets of frames.
