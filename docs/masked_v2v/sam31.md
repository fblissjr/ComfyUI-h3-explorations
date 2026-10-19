# SAM 3.1 in this pack: what to ask it, and how our tracking differs from ComfyUI core

**SAM 3.1 says where a person or thing is, never who.** It finds every instance of a short noun phrase, and the
lane's tracker follows one person a shot from a seed. Every place in the lane that finds, tracks or masks a subject
with SAM rests on this page; H3's own prompt is a different subject ([`docs/prompting/prompting.md`](../prompting/prompting.md)).

## In short

| question | answer | detail |
|---|---|---|
| what to type | the bare noun: `person`, `head`, `hand`, a thing's plain singular noun | "The rules", below |
| how to pick one person among several | never with a longer phrase: find all with `person`, then choose by the pick rule, a typed correction or a seed box | "Choosing one among several", below |
| which tracker runs | ours, `MiniMaxH3SubjectTrack` (`subject_track.py`): shot by shot, seeded, one object, bidirectional, with regain; core's text-prompted `SAM3_VideoTrack` is not run across a clip | [`sam31_core.md`](sam31_core.md), "Layer 2" |
| what is corrected at run time | three departures of core from Meta (vision input range, text activation, per-object tracker memory), all on in every job preview (`h3_config.MASKED_JOB_SAM_CORRECTIONS`) | [`sam31_core.md`](sam31_core.md), "Layer 1" |
| what is left as core computes it | seven more differences (`sam31_corrections.py::NOT_CORRECTED`) | [`sam31_core.md`](sam31_core.md) |
| who a tracked person is, across cuts and sessions | not SAM's to say: the owner's answer, kept in the cast file | [`who_is_who.md`](who_is_who.md) |
| the dated studies | ComfyUI against Meta stage by stage, the minimal core fixes of 2026-10-18 | [`docs/research/masking/`](../research/masking/); "Where the evidence is", below |

Where this page and a record disagree, the record is right; where it and the code disagree, the code is.

## What SAM 3.1 takes

A short noun phrase, or an example box, or a click. Meta's own description
is "a short text phrase or exemplars"
(`coderef/sam3/README.md`), and the text side did not change from SAM 3 to
SAM 3.1: 3.1 changed the tracker (`coderef/sam3/RELEASE_SAM3p1.md`). The
detector answers "where is every one of these", not "which one do you mean"
and not "what is this".

In this pack every phrase SAM is given is a visible, editable input, by the
owner's rule (2026-10-04). The shipped ones are in
`workflows/h3_config.py::SUBJECT_TRACK` (`subject_phrase`, `head_phrase`).

## The rules

| to find | ask for | why, and where it is counted |
|---|---|---|
| a person | the bare word, `person` | It is the commonest form in Meta's benchmarks and the one the benchmark almost never expects to be refused. [`2026-10-07_sam3_benchmark_phrases.md`](../../bench/results/2026-10-07_sam3_benchmark_phrases.md), "People" |
| a part of a person | the part's noun alone (`head`, `hand`, `hair`) | The bare part noun far outnumbers "a person's hand" in every dataset. Same record, "People". For parts inside one tracked person this pack uses the part model, not a phrase: `sapiens2_parts.py` |
| a thing | its plain singular noun, at most an article and one colour or modifier | That is the shape of nearly every phrase the benchmarks ask. Same record, "What a phrase looks like" |
| a small thing | the same, and do not depend on it | Things that small are in the benchmark by size, but a given small noun can be nearly untested (the record's "Size" names one). A phrase found the lane's test prop for the first seconds of a window and then not at all, by this pack's tracker and by ComfyUI's own per-frame text tracker alike (2026-10-07, not yet in a record) |
| what a person is holding | no phrase: the part node's `held` output | Inside the subject's mask, not body or clothing to the part model, near the lips or a hand. Nothing small is tracked on its own. `sapiens2_parts.py`, "What the subject holds" |
| one particular person among several | `person`, then choose | See the next section |

## Choosing one among several: not with a longer phrase

**Do not describe which one** ("the man in the red shirt", "the leftmost
woman", "the object in his hand").

- Meta's benchmarks barely contain such phrases, so the model is untested on
  them, which is not the same as failing: the benchmark record, "What a
  phrase looks like" and its fourth line to act on. Meta's own examples are
  one or two plain words throughout:
  [`docs/research/masking/2026-10-07_sam31_phrases.md`](../research/masking/2026-10-07_sam31_phrases.md).
- Under Meta's own scoring (the next section) the descriptive phrases an
  independent session tried returned nothing at all
  ([`2026-10-07_sam3_precision_arms.md`](../../bench/results/2026-10-07_sam3_precision_arms.md),
  the third departure).
- Meta's way to choose is to find all with the plain word and then select:
  a box or a click on top of the phrase, an object id in video, or a
  vision-language model that looks and picks
  (`coderef/sam3/examples/sam3.1_video_predictor_example.ipynb`,
  `sam3_agent.ipynb`; the phrase guide's "What Meta's own examples do").
- This pack's way today: the Subject Track's pick rule, its match across a
  cut, and typed corrections (`shot N: person K`) or seed boxes (`shot N: frame F x0,y0,x1,y1`): [`pieces.md`](pieces.md), piece 1, and [`who_is_who.md`](who_is_who.md).
  ComfyUI's detect node accepts boxes and points; nothing here passes them
  yet (read, not run).

## What ComfyUI does differently from Meta with a phrase

Each matters more the longer or rarer the phrase.

- **The count is ComfyUI's syntax, not the model's.** `person:16` asks the
  node for up to sixteen detections (`comfy/text_encoders/sam3_clip.py:48-84`,
  `SAM3TokenizerWrapper.tokenize_with_weights`); for counts above one, the
  `:N` is stripped before the text reaches the model. For `:1` (`person:1`),
  core's fast path (`comfy/text_encoders/sam3_clip.py:51-54`) forgets to strip
  `:1` and tokenizes the literal characters `:` and `1` into the prompt;
  `subject_track.py::counted` (`subject_track.py:395-415`) works around that
  by sending bare `name` when the count is one. The Subject Track's
  `max_people` sets the count. On a frame with more people than the count,
  the subject may not be among those returned, which decides whether a pick
  or a regain can see them at all:
  [`2026-10-07_subject_alone_or_in_a_group.md`](../../bench/results/2026-10-07_subject_alone_or_in_a_group.md).
- **The presence score is ignored.** Meta multiplies every detection's score
  by a score for "is this thing in the picture at all" (`presence` from
  `comfy/ldm/sam3/detector.py:213-214`) and removes overlapping detections;
  ComfyUI's `SAM3_Detect` (`comfy_extras/nodes_sam3.py:217-226`) and
  `track_video_with_detection` (`comfy/ldm/sam3/tracker.py:1697-1701`)
  threshold `scores.sigmoid()` alone.
  On a frame where the thing is barely present ComfyUI returns detections
  Meta's rule drops. **Read, not run (2026-10-07): Meta's released image
  code appears to apply the presence score twice**, once inside the model
  where the class score is written (`coderef/sam3/sam3/model/sam3_image.py`,
  under `supervise_joint_box_scores`, which both builders set) and again in
  the processor before its threshold
  (`coderef/sam3/sam3/model/sam3_image_processor.py`). That is Meta's
  single-image path. Its video pipeline thresholds the model's joint score
  as written, with no second factor
  (`coderef/sam3/sam3/model/sam3_video_base.py`, where detections are read),
  and that is the path the records below compare with, so their "Meta's
  rule", one multiplication, is Meta's video rule. No record here ran the
  image processor. A proposal to follow Meta has to say which path it
  means. The
  precision record's third departure;
  [`2026-10-07_sam3_core_against_meta.md`](../../bench/results/2026-10-07_sam3_core_against_meta.md)
  has both rules over the same detections.
- **The text encoder runs another activation than Meta built it with.** Core
  sets `"hidden_act": "quick_gelu"` in `comfy/text_encoders/sam3_clip.py:6`
  where Meta's `VETextEncoder` uses `nn.GELU()`
  (`coderef/sam3/sam3/model/text_encoder_ve.py:130`). Nil
  to small for `person` and `head` and for a plain noun with one modifier
  (the shape the rules above ask for; a session preparing the upstream fix,
  2026-10-07, on public images, reported and not in a tracked record),
  larger for longer phrases. The corrections node
  (`sam31_corrections.py::MiniMaxH3SAM31Corrections`, `sam31_corrections.py:837-873`)
  sets it right at run time for whatever loader feeds it, corrects the
  image's value range, and filters tracker memory by per-object score:
  [`2026-10-07_sam31_corrections_on_a_queue.md`](../../bench/results/2026-10-07_sam31_corrections_on_a_queue.md),
  [`2026-10-10_sam31_corrections_on_one_stretch.md`](../../bench/results/2026-10-10_sam31_corrections_on_one_stretch.md).
  Every masked job preview graph wires it with all three corrections on
  (`workflows/masked_job_graph.py:280-287`,
  `workflows/h3_config.py::MASKED_JOB_SAM_CORRECTIONS` at
  `workflows/h3_config.py:643-644`), while the single-pass shipped templates in
  `workflows/build_workflows.py:2259-2276` wire `CheckpointLoaderSimple`
  directly. Even
  corrected, a descriptive phrase is still scored by ComfyUI's rule and not
  Meta's.
- **On the CPU, and wherever ComfyUI does not use PyTorch's attention, the
  detector does not hide the padding of a phrase.** The detector hands its
  text mask to ComfyUI's attention as a true/false mask
  (`comfy/ldm/sam3/detector.py:54-57`, `SplitMHA.forward`, called from
  `EncoderLayer.forward` at `comfy/ldm/sam3/detector.py:101`,
  `DecoderLayer.forward` at `comfy/ldm/sam3/detector.py:139`, and
  `SegmentationHead.forward` at `comfy/ldm/sam3/detector.py:352`); PyTorch's
  attention reads that as "attend where true", while the sub-quadratic and
  split attention functions add it to the scores, which hides nothing
  (`comfy/ldm/modules/sub_quadratic_attention.py`). ComfyUI picks those two
  when PyTorch attention is not enabled (`comfy/ldm/modules/attention.py`,
  where `optimized_attention` is chosen), which is the default in a `--cpu`
  process. Found and localised by a session reviewing the upstream fix,
  2026-10-07, on public images: with `--use-pytorch-cross-attention` a CPU
  process gives the card's detections. Not reported upstream and not yet in
  a tracked record. It is not only long phrases: the same session reports
  a plain modifier-and-noun phrase gaining detections of other things on
  that path. The server log line "Using pytorch attention" is the
  observable for which path a process is on. **A detection score from a CPU
  process run without that flag is evidence about the faulty path only**;
  text features are not affected.

## How many, and for how long

- Sixteen people for one phrase is inside what Meta's benchmarks cover;
  thirty-two is at their edge, in images and in video. The benchmark record,
  its fifth line and "People". Sixteen is also the size of one group in SAM
  3.1's tracker (`RELEASE_SAM3p1.md`, Object Multiplex).
- Benchmark clips are short. A phrase held over minutes, across cuts, is
  this pack's own machinery and not something the model was tested on.
- **A subject who shrinks a great deal through one long shot is outside what
  the video benchmark covers.** A wide spread between a track's largest and
  smallest annotated frame is common there, but most of it is an object
  entering, leaving or being cut by the frame's edge; a sustained shrink of
  that order with no gap, over a long track, is all but absent for people
  and for vehicles, and growing is commoner than shrinking (counted from
  the annotation files by a helper of the lead session, 2026-10-07; the
  script and its aggregates are not yet tracked). Untested is not failing:
  on the one such window measured here the shipped track held as one mask
  ([`2026-10-07_subject_track_calls_on_masks.md`](../../bench/results/2026-10-07_subject_track_calls_on_masks.md)).
  The annotation files keep slivers of a few pixels and do not say why a
  frame is unannotated.
- A small object in the video benchmark leaves and returns often, and is
  usually gone only briefly in third-person footage and for much longer in
  egocentric footage. A regain should expect short dropouts as normal:
  [`2026-10-07_sam3_benchmark_targets.md`](../../bench/results/2026-10-07_sam3_benchmark_targets.md),
  which also has how many instances a pair holds for the words this lane
  asks (`person`, `head`, `hand`, `hair`).

## What is not known

- Whether a box or a click through ComfyUI's detect node (`SAM3_Detect`'s
  `points` / `boxes` path in `comfy/ldm/sam3/detector.py:225-312`) selects
  one person more cleanly than IoU-matching `person` detections against a
  seed box in `subject_track.py::parse_seeds`. Read in the code, not run.
- Whether patching `SAM31Tracker._forward_propagation` (`comfy/ldm/sam3/tracker.py:1391`)
  to select `tokens_obj[obj_range, best_idx]` instead of `tokens_obj[:, 0]`
  for `obj_ptr_proj` measurably changes tracking on our shots.
- Whether per-bucket/per-object memory selection (`bench/sam_memory_selection.py:67-169`)
  helps multi-object (`N_obj > 1`) calls of `SAM31Tracker` without hurting
  companion objects in the same multiplex bucket.

## Where the evidence is

| record or note | what it holds |
|---|---|
| [`2026-10-07_sam3_benchmark_phrases.md`](../../bench/results/2026-10-07_sam3_benchmark_phrases.md) | Meta's four datasets counted from their annotation files: phrase shapes, how often each is asked where nothing is annotated, sizes, people, video. `bench/sam3_dataset_phrases.py` reproduces it. The datasets are gated and are not in this repository |
| [`docs/research/masking/2026-10-07_sam31_phrases.md`](../research/masking/2026-10-07_sam31_phrases.md) | what Meta's README, notebooks, tests and eval code do with a phrase, by file and line |
| [`2026-10-07_sam3_precision_arms.md`](../../bench/results/2026-10-07_sam3_precision_arms.md) | precision, and three departures of ComfyUI from Meta measured blind |
| [`2026-10-07_sam3_core_against_meta.md`](../../bench/results/2026-10-07_sam3_core_against_meta.md) | ComfyUI against Meta's code stage by stage on equal inputs (`[0, 1]` vs `[-1, 1]`, `quick_gelu` vs `gelu`, `presence` gating, 32-object seeded tracking) |
| [`2026-10-07_sam31_corrections_on_a_queue.md`](../../bench/results/2026-10-07_sam31_corrections_on_a_queue.md) | `MiniMaxH3SAM31Corrections` (`correct_image_range` and `correct_text_activation`) accepted on a server's queue |
| [`2026-10-10_sam31_corrections_on_one_stretch.md`](../../bench/results/2026-10-10_sam31_corrections_on_one_stretch.md) | `MiniMaxH3SAM31Corrections` (`image_range` and `text_activation` alone and together) on the 447-frame kitchen stretch of `fun` (`shots 1–6`): why `text_activation` alone changes 0 detections on `person`/`head` and how `image_range` shifts trunk cosine similarity |
| [`2026-10-08_one_person_track_against_core_multi_object.md`](../../bench/results/2026-10-08_one_person_track_against_core_multi_object.md) | `MiniMaxH3SubjectTrack` (seeded per shot, `max_objects=1`, with `one_each`, `take_back`, `take_back_by_place`, `drop_specks`, and `others`) against ComfyUI core's text-prompted `SAM3_VideoTrack` (`max_objects=8` and `64`) on `fun` and `lotsofpeople` |
| [`2026-10-06_bordered_source_cuts.md`](../../bench/results/2026-10-06_bordered_source_cuts.md), [`2026-10-06_subject_track_defaults.md`](../../bench/results/2026-10-06_subject_track_defaults.md), [`2026-10-07_subject_track_under_nudge.md`](../../bench/results/2026-10-07_subject_track_under_nudge.md), [`2026-10-07_subject_alone_or_in_a_group.md`](../../bench/results/2026-10-07_subject_alone_or_in_a_group.md) | border stripping for cut detection (`flat_borders`), requiring a detected `head` on the pick frame (`head_of`), how `pick`, `match` and `regain` hold under a 1-level nudge, and what following a subject alone or in a group does |
| [`docs/research/masking/2026-10-07_sam31_in_comfyui.md`](../research/masking/2026-10-07_sam31_in_comfyui.md) | SAM 3.1 in ComfyUI end to end |
| [`docs/research/masking/2026-10-07_sam31_core_vs_meta_stages.md`](../research/masking/2026-10-07_sam31_core_vs_meta_stages.md) | the same, one row per stage with the file and lines on ComfyUI's side and on Meta's: the place to look up where a stage lives, the rules between followed objects among them (its tracker rows) |
| [`2026-10-18_sam31_minimal_core_fixes.md`](../research/masking/2026-10-18_sam31_minimal_core_fixes.md) | the 6 minimal surgical fixes (~25 lines across 4 files) that would bring ComfyUI core's existing SAM 3 / 3.1 detector, tokenizer, and single-shot tracker to numerical and functional parity with Meta's reference |
| a reading of 2026-10-06, no longer in the tree | Meta's video pipeline for one phrase, frame by frame, and what ComfyUI's port does instead at each step and at a cut. A reading; nothing was run for it |

