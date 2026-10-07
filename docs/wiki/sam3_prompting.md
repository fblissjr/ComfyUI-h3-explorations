# Asking SAM 3.1 for something: what a valid phrase is, and what to do when a phrase is not enough

last updated: 2026-10-07 (first written, from the day's reading of Meta's code and benchmark annotations; later the same day, what each ComfyUI difference does to a plain noun with one modifier)

Written by hand. This page is the operative form: one rule a line, and a
pointer to the record that holds the count behind it. It carries no numbers
of its own. Where it and a record disagree, the record is right; where it
and the code disagree, the code is. H3's own prompt is a different subject
and lives in [`docs/prompting.md`](../prompting.md).

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
  [`2026-10-07_mryolk_phrases.md`](../research/masking/2026-10-07_mryolk_phrases.md).
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
  cut, and typed corrections (`shot N: person K`): `docs/wiki/masked_v2v.md`.
  ComfyUI's detect node accepts boxes and points; nothing here passes them
  yet (read, not run).

## What ComfyUI does differently from Meta with a phrase

Each matters more the longer or rarer the phrase.

- **The count is ComfyUI's syntax, not the model's.** `person:16` asks the
  node for up to sixteen detections; the `:N` is stripped before the text
  reaches the model. The Subject Track's `max_people` sets it. On a frame
  with more people than the count, the subject may not be among those
  returned, which decides whether a pick or a regain can see them at all:
  [`2026-10-07_subject_alone_or_in_a_group.md`](../../bench/results/2026-10-07_subject_alone_or_in_a_group.md).
- **The presence score is ignored.** Meta multiplies every detection's score
  by a score for "is this thing in the picture at all" and removes
  overlapping detections; ComfyUI's node thresholds each detection alone.
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
- **The text encoder runs another activation than Meta built it with.** Nil
  to small for `person` and `head` and for a plain noun with one modifier
  (the shape the rules above ask for; a session preparing the upstream fix,
  2026-10-07, on public images, reported and not in a tracked record),
  larger for longer phrases. The corrections node
  (`sam31_corrections.py`, `MiniMaxH3SAM31Corrections`) sets it right at run
  time for whatever loader feeds it, and corrects the image's value range as
  well: [`2026-10-07_sam31_corrections_on_a_queue.md`](../../bench/results/2026-10-07_sam31_corrections_on_a_queue.md).
  No shipped graph wires it yet. Even corrected, a descriptive phrase is
  still scored by ComfyUI's rule and not Meta's.

- **On the CPU, and wherever ComfyUI does not use PyTorch's attention, the
  detector does not hide the padding of a phrase.** The detector hands its
  text mask to ComfyUI's attention as a true/false mask
  (`comfy/ldm/sam3/detector.py`, the cross-attention's `mask`); PyTorch's
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

- Whether a box or a click through ComfyUI's detect node selects one person
  reliably here. Read in the code, not run.

## Where the evidence is

| record or note | what it holds |
|---|---|
| [`2026-10-07_sam3_benchmark_phrases.md`](../../bench/results/2026-10-07_sam3_benchmark_phrases.md) | Meta's four datasets counted from their annotation files: phrase shapes, how often each is asked where nothing is annotated, sizes, people, video. `bench/sam3_dataset_phrases.py` reproduces it. The datasets are gated and are not in this repository |
| [`2026-10-07_mryolk_phrases.md`](../research/masking/2026-10-07_mryolk_phrases.md) | what Meta's README, notebooks, tests and eval code do with a phrase, by file and line |
| [`2026-10-07_sam3_precision_arms.md`](../../bench/results/2026-10-07_sam3_precision_arms.md) | precision, and three departures of ComfyUI from Meta measured blind |
| [`2026-10-07_sam3_core_against_meta.md`](../../bench/results/2026-10-07_sam3_core_against_meta.md) | ComfyUI against Meta's code stage by stage on equal inputs |
| [`2026-10-07_sam31_corrections_on_a_queue.md`](../../bench/results/2026-10-07_sam31_corrections_on_a_queue.md) | the corrections node accepted on a server's queue |
| [`2026-10-07_subject_track_under_nudge.md`](../../bench/results/2026-10-07_subject_track_under_nudge.md), [`2026-10-07_subject_alone_or_in_a_group.md`](../../bench/results/2026-10-07_subject_alone_or_in_a_group.md) | how the pick, the match and the regain hold under a change too small to see, and what following a subject alone or in a group does |
| [`docs/research/masking/2026-10-07_mryolk.md`](../research/masking/2026-10-07_mryolk.md) | SAM 3.1 in ComfyUI end to end |
| [`2026-10-07_mryolk_stage_table.md`](../research/masking/2026-10-07_mryolk_stage_table.md) | the same, one row per stage with the file and lines on ComfyUI's side and on Meta's: the place to look up where a stage lives, the rules between followed objects among them (its tracker rows) |
| [`2026-10-06_mrsun.md`](../research/masking/2026-10-06_mrsun.md) | Meta's video pipeline for one phrase, frame by frame, and what ComfyUI's port does instead at each step and at a cut. A reading; nothing was run for it |
