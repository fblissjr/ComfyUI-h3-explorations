# ComfyUI's SAM 3.1 against Meta's code, stage by stage on equal inputs (2026-10-07)

lane: masked
verdict: with the same weights, ComfyUI's port tokenizes as Meta's code does, its trunk agrees with Meta's on an equal input to within a measured bf16 floor, and its detector agrees on what it is confident about; it departs in two places before the model, the range of the image its nodes hand over and the activation its text encoder runs; from the same starting masks on one dense window, read against a one-level change of the input, Meta's tracker keeps more plausible masks than ComfyUI's at sixteen subjects and holds a second group of sixteen where ComfyUI's does not, and a subject followed alone is lost within a few frames by both

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
detector rungs (the tool's self-test against float64 is printed with them);
Meta's code under the bf16 autocast it enters itself. The trunk and
detector rungs carry their own FLOOR rows: ComfyUI against itself at bf16,
and ComfyUI against itself with the frame moved one level of 255. The
tracker rungs compare outcomes, not tensors, with ComfyUI at its default
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
notes, and measured a third that was first read in the code
([`docs/research/masking/2026-10-07_mryolk_stage_table.md`](../../docs/research/masking/2026-10-07_mryolk_stage_table.md),
rows 15 and 17): ComfyUI's detect node thresholds the class score alone and
removes no overlapping detections, where Meta's code thresholds the class
score multiplied by the presence score and removes them. The detector rung
here now counts both rules over the same queries.

**What changed in this record after its first commit, the same day.** The
first version said the bf16 floor had not been measured and called the
resize a departure "about four times" the equal-tensor difference; the
floors are measured now and the resize sits UNDER the one-level floor. It
said no difference between the trackers beyond sixteen was established,
because single runs moved by a dozen people with a two-level change of
input; the there-and-back rung has a nudged twin of every arm, and the
difference is read against it. The first run's tables are kept below as
they were.

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
| FLOOR core float32 vs core under bf16 same tensor | 0.06635 | 0.99822 |
| FLOOR core float32 same frame moved one level | 0.43833 | 0.90383 |
| equal tensor core vs meta | 0.04907 | 0.99922 |
| core as its node feeds vs meta | 0.65114 | 0.78773 |
| core resize in metas range vs meta | 0.21544 | 0.97722 |
| meta given the unmapped image vs meta | 0.65054 | 0.788 |
| core as fed vs meta given the same unmapped image | 0.03654 | 0.99975 |

TF32 flags {'matmul': False, 'cudnn': False}; self-test relative error against float64: convolution 7.3e-07, matmul 4.2e-07.

### Image range and trunk, at the file's own size

`lotsofpeopledance_0414_0720.mkv` at 16.0 s, 1920x1080. Core's node hands its trunk [0.0, 0.934]; Meta's loader hands its trunk [-1.0, 0.882]. Patch embedding: weights differ by 0.0, bias in core False, in Meta False.

| trunk features, last level | relative L2 | cosine |
|---|---|---|
| FLOOR core float32 vs core under bf16 same tensor | 0.03804 | 0.99971 |
| FLOOR core float32 same frame moved one level | 0.33347 | 0.9448 |
| equal tensor core vs meta | 0.03503 | 0.99982 |
| core as its node feeds vs meta | 0.65828 | 0.78806 |
| core resize in metas range vs meta | 0.22659 | 0.97482 |
| meta given the unmapped image vs meta | 0.65868 | 0.78758 |
| core as fed vs meta given the same unmapped image | 0.0397 | 0.99962 |

TF32 flags {'matmul': False, 'cudnn': False}; self-test relative error against float64: convolution 7.3e-07, matmul 4.2e-07.

### Detector, on Meta's image tensor: the first run

`thrill_2160.mkv` at 56.5 s, phrase `person`, 200 queries. "Over 0.5" counts queries over 0.5 on the joint score, before any overlap removal: it is not what either side returns.

| core's text | presence logit core / Meta | joint score logits: relative L2, cosine | queries over 0.5 core / Meta | top ten by Meta: Meta | the same queries: core | their mask IoU |
|---|---|---|---|---|---|---|
| exact GELU | 3.4538 / 3.4375 | 0.09377, 0.99562 | 2 / 2 | [0.957, 0.554, 0.492, 0.094, 0.067, 0.056, 0.055, 0.055, 0.054, 0.051] | [0.959, 0.561, 0.456, 0.092, 0.07, 0.059, 0.053, 0.079, 0.038, 0.046] | [0.999, 1.0, 0.997, 0.833, 0.992, 0.995, 0.972, 0.686, 0.365, 0.998] |
| as shipped | 3.5553 / 3.4375 | 0.09509, 0.9955 | 2 / 2 | [0.957, 0.554, 0.492, 0.094, 0.067, 0.056, 0.055, 0.055, 0.054, 0.051] | [0.962, 0.565, 0.459, 0.093, 0.069, 0.059, 0.055, 0.08, 0.039, 0.046] | [0.999, 1.0, 0.997, 0.84, 0.994, 0.995, 0.972, 0.697, 0.396, 1.0] |

### Detector, on Meta's image tensor

`lotsofpeopledance_0414_0720.mkv` at 16.0 s, phrase `person`, 200 queries. Rules: the node's is class score over 0.5; Meta's is class times presence over 0.4, then overlap removal at 0.1 on the smaller mask.

| core's text | presence logit core / Meta | joint score logits: relative L2, cosine | queries over 0.5 on the joint score, core / Meta | top ten by Meta: Meta | the same queries: core | their mask IoU |
|---|---|---|---|---|---|---|
| exact GELU | 2.8536 / 2.7969 | 0.36242, 0.93378 | 26 / 25 | [0.914, 0.91, 0.91, 0.906, 0.906, 0.895, 0.895, 0.895, 0.89, 0.887] | [0.914, 0.915, 0.913, 0.887, 0.909, 0.894, 0.86, 0.885, 0.879, 0.069] | [0.998, 0.999, 0.998, 0.997, 0.999, 0.999, 0.997, 0.994, 0.991, 0.001] |
| as shipped | 3.192 / 2.7969 | 0.36474, 0.93255 | 27 / 25 | [0.914, 0.91, 0.91, 0.906, 0.906, 0.895, 0.895, 0.895, 0.89, 0.887] | [0.93, 0.931, 0.929, 0.904, 0.925, 0.911, 0.877, 0.903, 0.897, 0.072] | [0.998, 0.999, 0.998, 0.997, 0.998, 0.998, 0.997, 0.993, 0.99, 0.002] |

The floor for that row: core against itself with the frame moved one level of 255: presence logit 2.8536 and 2.9333; joint score logits relative L2 0.27039, cosine 0.96384.

What each rule keeps of the same queries (core's text with exact GELU):

| frame, s | presence logit core / Meta | core's outputs, the node's rule | core's outputs, Meta's rule | Meta's outputs, Meta's rule |
|---|---|---|---|---|
| 16.0 | 2.8536 / 2.7969 | 27 | 26 | 26 |
| 18.0 | 2.8994 / 2.8906 | 28 | 28 | 30 |
| 138.0 | 2.6011 / 2.6406 | 48 | 49 | 49 |
| 140.0 | 1.7975 / 1.7891 | 45 | 45 | 48 |
| 142.0 | 0.188 / 0.1221 | 30 | 1 | 0 |
| 147.13 | 1.139 / 1.1328 | 41 | 34 | 40 |
| 150.0 | 1.1084 / 1.2188 | 52 | 44 | 49 |

TF32 flags {'matmul': False, 'cudnn': False}; self-test relative error against float64: convolution 7.3e-07, matmul 4.2e-07.

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

### There and back: seeds from one detect by core on the first frame, corrected

`vma.mp4` from 121.93 s, 72 frames at 24.0 a second played forward and then in reverse (143 frames, the last being the first), 1344x760; 64 detections on the seed frame. One run per arm. A subject is PLAUSIBLE when its mask passes the shape test ({'grid': 252, 'least_cells': 32, 'border_cells': 6, 'border_share_most': 0.5, 'box_fill_least': 0.1}); shape, not identity. With several subjects meta's tracker-only path applies neither of its two between-subject rules; core applies both.

Seeded: alone.

| arm | with a mask, end of forward | plausible, end of forward | plausible, back at the start | back on its own seed | on another's seed | plausible subject-frames | against itself nudged: at a half or more, median | against Meta's tracker: at a half or more, median |
|---|---|---|---|---|---|---|---|---|
| core, as its node gets the frames | 0 | 0 | 1 | 1 | 0 | 4 of 143 | 1 of 1, 0.996 | 1 of 1, 0.973 |
| core, as its node gets the frames, nudged | 0 | 0 | 1 | 1 | 0 | 4 of 143 |  | 1 of 1, 0.973 |
| core, mapped before the node | 0 | 0 | 1 | 1 | 0 | 4 of 143 | 1 of 1, 0.996 | 1 of 1, 0.978 |
| core, mapped before the node, nudged | 0 | 0 | 1 | 1 | 0 | 4 of 143 |  | 1 of 1, 0.978 |
| core, corrected at the first layer | 0 | 0 | 1 | 1 | 0 | 4 of 143 | 1 of 1, 0.995 | 1 of 1, 0.978 |
| core, corrected at the first layer, nudged | 0 | 0 | 1 | 1 | 0 | 4 of 143 |  | 1 of 1, 0.978 |
| Meta's tracker | 0 | 0 | 1 | 1 | 0 | 4 of 143 | 1 of 1, 0.994 |  |
| Meta's tracker, nudged | 0 | 0 | 1 | 1 | 0 | 4 of 143 |  | 1 of 1, 0.994 |

Seeded: with one neighbour.

| arm | with a mask, end of forward | plausible, end of forward | plausible, back at the start | back on its own seed | on another's seed | plausible subject-frames | against itself nudged: at a half or more, median | against Meta's tracker: at a half or more, median |
|---|---|---|---|---|---|---|---|---|
| core, as its node gets the frames | 0 | 0 | 2 | 2 | 0 | 20 of 286 | 2 of 2, 0.978 | 2 of 2, 0.939 |
| core, as its node gets the frames, nudged | 0 | 0 | 2 | 2 | 0 | 20 of 286 |  | 2 of 2, 0.928 |
| core, mapped before the node | 0 | 0 | 2 | 2 | 0 | 20 of 286 | 2 of 2, 0.973 | 2 of 2, 0.946 |
| core, mapped before the node, nudged | 0 | 0 | 2 | 2 | 0 | 20 of 286 |  | 2 of 2, 0.951 |
| core, corrected at the first layer | 0 | 0 | 2 | 2 | 0 | 21 of 286 | 2 of 2, 0.93 | 2 of 2, 0.905 |
| core, corrected at the first layer, nudged | 0 | 0 | 2 | 2 | 0 | 20 of 286 |  | 2 of 2, 0.952 |
| Meta's tracker | 0 | 0 | 2 | 2 | 0 | 20 of 286 | 2 of 2, 0.929 |  |
| Meta's tracker, nudged | 0 | 0 | 2 | 2 | 0 | 21 of 286 |  | 2 of 2, 0.929 |

Seeded: 16.

| arm | with a mask, end of forward | plausible, end of forward | plausible, back at the start | back on its own seed | on another's seed | plausible subject-frames | against itself nudged: at a half or more, median | against Meta's tracker: at a half or more, median |
|---|---|---|---|---|---|---|---|---|
| core, as its node gets the frames | 16 | 13 | 15 | 16 | 0 | 1960 of 2288 | 16 of 16, 0.965 | 11 of 16, 0.881 |
| core, as its node gets the frames, nudged | 16 | 12 | 15 | 16 | 0 | 1971 of 2288 |  | 11 of 16, 0.878 |
| core, mapped before the node | 15 | 13 | 16 | 16 | 0 | 1956 of 2288 | 15 of 16, 0.969 | 11 of 16, 0.928 |
| core, mapped before the node, nudged | 15 | 12 | 15 | 16 | 0 | 1929 of 2288 |  | 13 of 16, 0.926 |
| core, corrected at the first layer | 15 | 12 | 16 | 16 | 0 | 1954 of 2288 | 15 of 16, 0.972 | 13 of 16, 0.93 |
| core, corrected at the first layer, nudged | 15 | 12 | 15 | 16 | 0 | 1925 of 2288 |  | 13 of 16, 0.925 |
| Meta's tracker | 16 | 16 | 16 | 16 | 0 | 2245 of 2288 | 16 of 16, 0.98 |  |
| Meta's tracker, nudged | 16 | 16 | 16 | 16 | 0 | 2248 of 2288 |  | 16 of 16, 0.98 |

Seeded: 32.

| arm | with a mask, end of forward | plausible, end of forward | plausible, back at the start | back on its own seed | on another's seed | plausible subject-frames | against itself nudged: at a half or more, median | against Meta's tracker: at a half or more, median |
|---|---|---|---|---|---|---|---|---|
| core, as its node gets the frames | 17 | 14 | 31 | 32 | 0 | 2408 of 4576 | 28 of 32, 0.831 | 11 of 32, 0.256 |
| core, as its node gets the frames, nudged | 16 | 12 | 31 | 32 | 0 | 2344 of 4576 |  | 11 of 32, 0.213 |
| core, mapped before the node | 17 | 13 | 32 | 32 | 0 | 2715 of 4576 | 28 of 32, 0.923 | 12 of 32, 0.386 |
| core, mapped before the node, nudged | 16 | 12 | 31 | 32 | 0 | 2647 of 4576 |  | 13 of 32, 0.389 |
| core, corrected at the first layer | 16 | 12 | 32 | 32 | 0 | 2684 of 4576 | 28 of 32, 0.931 | 13 of 32, 0.394 |
| core, corrected at the first layer, nudged | 15 | 12 | 31 | 32 | 0 | 2627 of 4576 |  | 13 of 32, 0.378 |
| Meta's tracker | 32 | 26 | 31 | 32 | 0 | 4218 of 4576 | 28 of 32, 0.93 |  |
| Meta's tracker, nudged | 32 | 26 | 32 | 32 | 0 | 4086 of 4576 |  | 28 of 32, 0.93 |

### The same core arms on frames decoded another way

The same window, seeds and core arms from a scratch probe run earlier the same day, not by this tool; its frames were decoded by calling ffmpeg directly (fps, crop, scale) where the tool uses the loader node. Seeds: one detect by core on the first frame as its node gets it. The two sets of frames are the same frames in time and differ by 1.74 to 1.8 levels of 255 on average.

| | core, frames as its node gets them | core, frames in the trained range |
|---|---|---|
| 32 seeded with a mask at the last frame | 16 | 32 |
| 17 seeded last frame the seventeenth is on | 14 | 71 |
| alone frames on | 5 | 3 |

## What this establishes

Each line says how it is known.

- **Tokens: the same** (measured, CPU), on every prompt part of the tool's
  `TOKEN_PHRASES`, with ComfyUI's count syntax parsed by its own tokenizer.
- **The ComfyUI-org weights file is ours cast to float16** (measured, every
  tensor). It carries no folded normalisation.
- **Text encoder: departs, and the activation is the whole of it**
  (measured). With the activation switched in memory to exact GELU the
  features agree to bf16 rounding. For the one-word phrases the lane ships
  the detections on one frame are the same either way; for a describing
  phrase the presence score halves, at a presence Meta's own detection rule
  would reject, so for such a phrase the activation matters only together
  with that rule.
- **Image range: departs** (measured at the trunk's input, and read:
  `comfy_extras/nodes_sam3.py` resizes and does nothing else; Meta's
  list-of-images route in `io_utils.py`, which is the route these rungs
  drive, and its image processor map to -1..1; the value mapping of Meta's
  other loading routes was not compared). The patch embedding has no bias
  on either side and equal weights. Meta's own trunk given ComfyUI's
  unmapped image is as far from itself as ComfyUI's is, and ComfyUI's trunk
  given the unmapped image matches Meta's given the same: the port's
  arithmetic is right and its input is not.
- **How large that is, on two measures.** On the trunk's features the range
  fault is one and a half to two times what a one-level change of the frame
  does (the row "core as its node feeds vs meta" against the one-level
  FLOOR row, on two clips). On tracked masks, there and back from the same
  seeds: nothing at sixteen subjects beyond the floor, and about a tenth
  more plausible subject-frames at thirty-two, four times the floor. Real,
  and not an order above the noise of this footage.
- **Trunk: the same on an equal tensor, to bf16 rounding** (measured): the
  equal-tensor row sits at or under the FLOOR row for ComfyUI against
  itself at bf16, at two frame sizes.
- **The resize: a departure of the size of a one-level change** (measured,
  two frames, one at the file's own 1080p). ComfyUI's bilinear resize
  without smoothing against Meta's Pillow resize moves the trunk's features
  by less than the one-level FLOOR row does. Mapping the range does not
  correct it.
- **Detector: the same on what it is confident about** (measured, two
  clips). On the crowd clip nine of Meta's ten highest-scoring detections
  have the same score within two hundredths and the same mask in ComfyUI;
  the tenth is carried by another query there. Over all queries the joint
  scores differ by a little more than a one-level change of the frame
  does. **The two detection rules** keep the same number of detections
  within three on four of seven frames and within eight on two; on the
  seventh, where the presence score is low, ComfyUI's node keeps thirty
  and Meta's rule one or none (measured; the precision record's third
  departure, reproduced here on another clip).
- **Trackers, there and back, with a floor** (measured, one dense window,
  seeds from a corrected detect, each arm also run nudged): a subject
  followed alone is lost within a few frames by ComfyUI and by Meta's
  tracker, and so are two together; at sixteen, Meta's tracker keeps a
  plausible mask on nearly every subject-frame and ComfyUI's on about six
  in seven, a difference ten times the floor; at thirty-two, Meta's ends
  the forward pass with all of them and ComfyUI's with about half, in both
  versions of the input. Correcting the range at the first layer and
  mapping the frames before the node are the same within the floor.
  "Plausible" is a shape test (not a strip along the frame's border, not a
  fragment), not a check of identity.

## What it does not establish

- **Why ComfyUI's tracker keeps less.** Meta's tracker here is its
  tracker-only path, which applies neither of the two rules its full
  pipeline applies between subjects; ComfyUI applies both, with its own
  values. Those rules as ComfyUI sets them are the first suspect; nothing
  here separates them from the port.
- **That a subject who ends on their own seed was followed.** In the
  there-and-back rung every arm ends on its own seeds, including arms that
  lost the subject after four frames: the last frame is the first again and
  the tracker's kept memory of it finds the subject. That column is read
  with plausible subject-frames, never alone.
- **Which input the model does better on.** No picture was judged.
- **The tracker stage by stage**, Meta's full session logic (its
  association, confirmation and refresh rules), more than one window for
  the tracker rungs, or any clip but the three named.
- **That mapping the range and switching the activation make ComfyUI's
  pipeline Meta's.** They correct those two departures. The resize, the
  detection rule, the detect node's refinement passes, the hole-fill value,
  the pointer token, the rules between subjects and the order of threshold
  and resize on output stay ComfyUI's
  ([`docs/research/masking/2026-10-07_mryolk.md`](../../docs/research/masking/2026-10-07_mryolk.md)
  walks through each).

## What uses it

[`sam31_corrections.py`](../../sam31_corrections.py) attaches the two
corrections to a loaded model as ComfyUI patches;
[`2026-10-07_sam31_corrections_on_a_queue.md`](2026-10-07_sam31_corrections_on_a_queue.md)
is its acceptance.

## Reproducing it

From ComfyUI's environment, outside the server, the card free, with `T` for
`bench/sam3_parity_ladder.py` and `J` for a json path of your own:

```
<python> T tokens --json J
<python> T files --theirs <the ComfyUI-org file> --ours <ours> --json J
<python> T text --clip thrill_2160.mkv --second 56.5 --width 1024 --json J --describing "<a phrase>"
<python> T trunk --clip thrill_2160.mkv --second 56.5 --width 1024 --json J
<python> T trunk --clip lotsofpeopledance_0414_0720.mkv --second 16 --width 0 --json J
<python> T detector --clip lotsofpeopledance_0414_0720.mkv --second 16 --width 0 --more-seconds 18 138 140 142 147.13 150 --json J
<python> T there-and-back --clip vma.mp4 --second 121.93 --seconds 3 --width 1344 --rate 24 --seeds-from corrected --json J
<python> T tracker --clip vma.mp4 --second 121.93 --seconds 3 --width 1344 --rate 24 --seeds-from node --json J
<python> T render --json J
```

## Files

- `2026-10-07_sam3_core_against_meta.json`: one key per rung as the tool
  wrote it, each with its environment. `detector` (no clip in its key) and
  the two `tracker, seeds ...` keys are the first run's, kept in the shape
  they were recorded in; `earlier_run_other_decode` holds the few counts of
  a scratch run the last table names.
