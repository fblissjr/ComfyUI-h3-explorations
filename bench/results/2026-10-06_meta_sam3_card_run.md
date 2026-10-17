# Meta's SAM 3.1 video predictor from the copy in this pack: its first run on the card (2026-10-06)

lane: masked
verdict: the copy runs end to end on this machine with nothing installed, and one session per shot gives one id per shot that follows the lane's stored subject mask; a long shot's memory is not bounded, descriptive phrases return nothing on the second clip because the presence score rejects them, and the session across a cut was not measured

**What this is.** The record of the first time the code under `meta_sam3/`
ran a session: four processes in one slot on the card and a fifth, a
short probe, later the same day, with ComfyUI's models
freed and nothing else running. It was written by the session that ran it
(mrold), from the logs its run script wrote. The numbers are in
`2026-10-06_meta_sam3_card_run.json`; every table below is printed from that
file by a script, and where a sentence and the file could disagree the file
is right. Each finding says whether it was measured here, read from code,
inferred, or not established.

**What it leaves out, on purpose.** Nothing here says what either clip
shows. A clip is its file name and its technical facts. Three of the phrases
asked of the second clip were chosen off what it shows, so they are "phrase
A", "phrase B" and "phrase C" here and their text is not in this repo. A
count of objects is what the detector returned for a phrase, and is given
as that.

**What it is not.** It is not a comparison of Meta's pipeline with ComfyUI
core's port on matched inputs, and nothing here ranks the two. The one place
the two meet is the stored subject mask, which core's port made earlier the
same day and which is used as a reference for "is this the same object",
not as ground truth.

## What was asked

Whether Meta's own inference code, copied and edited as
`meta_sam3/README.md` describes, runs a session
on this machine's Python and numpy; what a session costs in seconds and in
memory; what it returns across the cuts of a multi-shot clip; and whether
the model leaves the card and comes back. During the day the lane's lead
added a second clip: a frame with many detections, phrases that pick one
object out of them, and a stretch where the lane's tracker loses its subject
inside a shot.

## Setup

One process at a time, outside ComfyUI, in ComfyUI's own environment (the
json's `environment`). The tree is the copy as committed. The weights are
the repacked safetensors the pack already uses, given to `load_state_dict`.
Each session ran inside a bfloat16 autocast the run script entered and left,
with both tf32 flags set by the script; the copy sets neither
(`meta_sam3/README.md`, "What the copy leaves to its caller").

The tables' `tree` column has two values. `proto` is the copy as committed
under `meta_sam3/`. `first` is an earlier scratch tree that is not in the
repo, the one the committed tree was made from: the same upstream commit,
with absolute imports, no header, the builder untrimmed and the tokenizer's
environment line still in it. It ran only where a stage failed on `proto` and fell back, so a row
marked `first` is not a second shipped thing. Where the same frames ran on
both, the masks are equal (the json's `trees`).

A session is Meta's request API as released: `start_session` with a list of
PIL images, `add_prompt` with a phrase on the session's first frame,
`propagate_in_video` in both directions, `close_session`. A one-frame
request is the same three calls on a session of one frame.

**One setting was not Meta's.** The builder has the detector take a batch of
frames at once (`batched_grounding_batch_size` in
`meta_sam3/sam3/model_builder.py`). At the built value the first process
ran out of memory, so every other process set a lower value on the built
model (the json's `settings.detector_frame_batch`). Measured: the built
value and the one lower value. Not measured: any other.

The two clips, as the json's `clips` has them: `thrill_2160.mkv` from 56.0 s
as frames another session extracted (the command is not recorded; its cuts
agree with the lane's shot table for the same load), and `vma.mp4` from
36.0 s as frames made with the video loader's own ffmpeg arguments. For the
second clip the frame index was checked against frames of the same load
that a graph had saved (an encoded copy, so the comparison is loose):

| saved frame | closest of the extraction | mean absolute difference there | next closest, times further |
|---|---|---|---|
| 0 | 0 | 1.274 | 3.8 |
| 100 | 100 | 1.298 | 3.1 |
| 240 | 240 | 1.861 | 12.2 |
| 359 | 359 | 1.403 | 4.5 |

## The four processes

| process | detector frame batch | what | ended |
|---|---|---|---|
| 1 | 16 | the 501-frame session ran out of memory while propagating; the script then kept the failed session's memory, so every later stage failed too | stopped by hand |
| 2 | 4 | the stages in `sessions` under run `main` | stopped by hand before its last stage |
| 3 | 4 | four one-frame requests at an output threshold of 0.05, one long session at the higher cap, the round trip again | exit 0 |
| 4 | 4 | one 81-frame session at each cap | exit 0 |

## What ran, and what it cost

**It runs** (measured). The build loads every weight; the two scripted box
functions run on the card under this Python, and the third
`torch.jit.script` site, in `SAM2Transforms`, is never built on the 3.1
path; `start_session` works with the declared edit; the outputs are numpy
arrays per frame, in ascending order, each frame once. Autocast is off
after import, after build and outside every session, and
`TOKENIZERS_PARALLELISM` is unset at exit of a process started with it
unset (the json's `build` and `process_state`).

**One session per shot on the first clip** (measured). The opening shot,
then each of the eleven other shots of the load:

| session | frames | ids | frames with no object | seconds in all | of it propagating | peak GiB | over the model |
|---|---|---|---|---|---|---|---|
| opening | 0 to 500 (501) | 18 | 0 | 77.7 | 64.88 | 20.38 | 16.73 |
| shot02 | 501 to 543 (43) | 5 | 0 | 6.25 | 4.88 | 8.85 | 5.2 |
| shot03 | 544 to 610 (67) | 15 | 8 | 9.58 | 7.67 | 9.45 | 5.8 |
| shot04 | 611 to 625 (15) | 4 | 0 | 2.4 | 1.65 | 8.32 | 4.67 |
| shot05 | 626 to 704 (79) | 15 | 0 | 11.52 | 9.35 | 10.07 | 6.42 |
| shot06 | 705 to 722 (18) | 8 | 0 | 2.88 | 2.04 | 8.44 | 4.79 |
| shot07 | 723 to 756 (34) | 15 | 0 | 5.27 | 4.06 | 9.01 | 5.36 |
| shot08 | 757 to 779 (23) | 9 | 0 | 3.53 | 2.61 | 8.5 | 4.85 |
| shot09 | 780 to 841 (62) | 16 | 6 | 9.07 | 7.22 | 9.59 | 5.94 |
| shot10 | 842 to 939 (98) | 5 | 30 | 12.99 | 10.39 | 9.33 | 5.68 |
| shot11 | 940 to 1037 (98) | 5 | 0 | 13.37 | 10.97 | 9.72 | 6.07 |
| shot12 | 1038 to 1064 (27) | 5 | 18 | 3.69 | 2.68 | 8.33 | 4.68 |

|  | frames | seconds |
|---|---|---|
| one session per shot, twelve sessions | 1065 | 158.2 in all, 128.4 propagating |
| today's tracker, another session's figure through the server | 1065 | 188 |

The second row is not this run's: it is another session's figure for a
different kind of run, and is here only to say the two are of the same
order.

A shot shorter than the delay Meta's session keeps before it confirms an
object (read from code: the hotstart arguments in `model_builder.py`) still
returns every frame with its objects (measured: `shot04`).

**The score is the object's, not the frame's** (measured). Over a session
each id has one distinct score, a few have two (`distinct_scores_per_id` on
each session in the json). So a per-frame confidence read from this output
is the object's score, repeated.

## Memory

**A session's peak grows with its length, and the longest shot here nearly
fills the card** (measured: the table above). Frames on the card are part of
it. The opening shot again with `offload_video_to_cpu`:

|  | frames on the card | frames on the host |
|---|---|---|
| tree | proto | first |
| allocated before, GiB | 3.646 | 7.231 |
| allocated after start_session, GiB | 6.492 | 7.232 |
| peak allocated, GiB | 20.38 | 21.12 |
| peak over what was loaded, GiB | 16.73 | 13.89 |
| start_session, s | 6.78 | 5.7 |
| propagating, s | 64.88 | 65.7 |
| in all, s | 77.7 | 77.02 |

Keeping the frames on the host saves what the frames took and nothing else,
costs no time, and gives equal masks on every frame (measured, with the
caution that the second column ran on the earlier scratch tree with a
second model on the card: see "What went wrong in the run").

**What grows is per object** (inferred). A session that finds no object
stays at the floor at a length where sessions with objects have grown
(the second clip's table below against the per-shot table). Which tensors
are kept was not read. Two flags of the tracker that might bound it,
`trim_past_non_cond_mem_for_eval` and `offload_output_to_cpu_for_eval`, were
found by search and not run. The 3.1 model's `init_state` takes no
`offload_state_to_cpu` (read from code; it is why the declared edit to
`start_session` exists).

**No clean limit was measured.** The one session longer than the opening
shot ran out of memory, but with a second model on the card (failures
table). What a wrapper can rely on from this run: the detector's frame
batch has to be set for the card, and a shot much longer than the opening
one cannot be assumed to fit.

## The sessions against the stored subject mask

The stored mask is on in some of the shots and off in the others. Where it
is on, one id of Meta's session is the best match on all or nearly all of
those frames: the table's fourth column against its third, where the
difference is frames with no object in shot 3 and frames where another id
matched better in shot 11 (measured, by the definition in the json's
`against_stored_mask`):

| shot | frames | stored mask on | best id is one id on | mean IoU | median | frames with no object | first frame with an object |
|---|---|---|---|---|---|---|---|
| 1 | 0 to 500 | 501 | 501 | 0.9778 | 0.979 | 0 | 0 |
| 2 | 501 to 543 | 43 | 43 | 0.9732 | 0.9733 | 0 | 501 |
| 3 | 544 to 610 | 67 | 59 | 0.8596 | 0.9766 | 8 | 552 |
| 4 | 611 to 625 | 0 |  |  |  | 0 | 611 |
| 5 | 626 to 704 | 79 | 79 | 0.963 | 0.9638 | 0 | 626 |
| 6 | 705 to 722 | 0 |  |  |  | 0 | 705 |
| 7 | 723 to 756 | 33 | 33 | 0.9516 | 0.965 | 0 | 723 |
| 8 | 757 to 779 | 0 |  |  |  | 0 | 757 |
| 9 | 780 to 841 | 0 |  |  |  | 6 | 786 |
| 10 | 842 to 939 | 0 |  |  |  | 30 | 872 |
| 11 | 940 to 1037 | 47 | 45 | 0.938 | 0.9893 | 0 | 940 |
| 12 | 1038 to 1064 | 0 |  |  |  | 18 | 1056 |

Shot 3's mean is pulled down by its first frames, where the session returns
no object. Three other shots also begin with frames that have none. Why is
not established: a detection under the detector's threshold on those
frames, or the session's start-up delay, would both look like this.

The value a node would hand downstream for these twelve sessions, every
object, by three ways of storing it (measured on the run's own masks):

| records | cropped and bit-packed, MiB | full-frame bits, MiB | as float32, GiB |
|---|---|---|---|
| 8244 | 88.32 | 772.88 | 24.15 |

## One-frame requests

| run | clip | frame | phrase | cap | output threshold | objects | lowest and highest score | seconds in all |
|---|---|---|---|---|---|---|---|---|
| main | thrill_2160.mkv | 12 | person | 16 | as built | 2 | 0.5817 to 0.957 | 0.4 |
| main | thrill_2160.mkv | 12 | head | 16 | as built | 1 | 0.9179 to 0.9179 | 0.39 |
| main | thrill_2160.mkv | 505 | person | 16 | as built | 6 | 0.5277 to 0.8904 | 0.4 |
| main | vma.mp4 | 0 | person | 16 | as built | 21 | 0.5586 to 0.9375 | 0.47 |
| main | vma.mp4 | 0 | phrase A | 16 | as built | 0 |  | 0.37 |
| main | vma.mp4 | 0 | phrase B | 16 | as built | 0 |  | 0.39 |
| main | vma.mp4 | 192 | person | 16 | as built | 33 | 0.7388 to 0.9375 | 0.52 |
| main | vma.mp4 | 192 | phrase A | 16 | as built | 0 |  | 0.39 |
| main | vma.mp4 | 192 | phrase B | 16 | as built | 0 |  | 0.38 |
| main | vma.mp4 | 0 | person | 64 | as built | 21 | 0.5586 to 0.9375 | 0.47 |
| main | vma.mp4 | 192 | person | 64 | as built | 33 | 0.7388 to 0.9375 | 0.54 |
| follow-up 1 | vma.mp4 | 0 | phrase A | 16 | 0.05 | 0 |  | 0.55 |
| follow-up 1 | vma.mp4 | 192 | phrase A | 16 | 0.05 | 0 |  | 0.37 |
| follow-up 1 | vma.mp4 | 0 | phrase B | 16 | 0.05 | 0 |  | 0.37 |
| follow-up 1 | vma.mp4 | 192 | phrase B | 16 | 0.05 | 0 |  | 0.37 |

A one-frame request costs a fraction of a second (measured). Read from
code: Meta's cap on objects is not applied to a one-frame session
(`sam3_video_base.py`, the `is_image_only` test beside `max_num_objects`),
which is why the count for `person` on the second clip is the same at both
caps and above the lower one.

**Phrases A and B return nothing on the second clip** (measured), on the
two frames tried and, for phrase A, on every frame of a session over a
later stretch (next table). On ComfyUI core's port the same two phrases
each returned one detection on this load (another session's observation,
not re-run here).

**Why: presence** (measured, with the gate read from code). A fifth
process later the same day read the presence score with a forward hook on
the decoder's presence head, on one-frame requests:

| clip | frame | phrase | presence | objects | lowest and highest score |
|---|---|---|---|---|---|
| thrill_2160.mkv | 12 | person | 0.9693 | 2 | 0.5817 to 0.957 |
| thrill_2160.mkv | 12 | head | 0.9831 | 1 | 0.9179 to 0.9179 |
| vma.mp4 | 0 | person | 0.9679 | 21 | 0.9179 to 0.9375 |
| vma.mp4 | 0 | phrase A | 0.0025 | 0 |  |
| vma.mp4 | 0 | phrase B | 0.0716 | 0 |  |
| vma.mp4 | 0 | phrase C | 0.0252 | 0 |  |
| vma.mp4 | 4 | person | 0.9724 | 21 | 0.914 to 0.9416 |
| vma.mp4 | 4 | phrase A | 0.0094 | 0 |  |
| vma.mp4 | 4 | phrase B | 0.0839 | 0 |  |
| vma.mp4 | 192 | person | 0.9756 | 33 | 0.9259 to 0.9375 |
| vma.mp4 | 192 | phrase A | 0.0071 | 0 |  |
| vma.mp4 | 192 | phrase B | 0.1097 | 0 |  |
| vma.mp4 | 192 | phrase C | 0.0061 | 0 |  |
| vma.mp4 | 640 | person | 0.7154 | 2 | 0.5039 to 0.5311 |
| vma.mp4 | 640 | phrase A | 0.0362 | 0 |  |
| vma.mp4 | 640 | phrase B | 0.0601 | 0 |  |
| vma.mp4 | 680 | person | 0.7994 | 10 | 0.5277 to 0.6332 |
| vma.mp4 | 680 | phrase A | 0.0021 | 0 |  |
| vma.mp4 | 680 | phrase B | 0.0107 | 0 |  |
| vma.mp4 | 696 | person | 0.7372 | 3 | 0.5039 to 0.5461 |
| vma.mp4 | 696 | phrase A | 0.0034 | 0 |  |
| vma.mp4 | 696 | phrase B | 0.0302 | 0 |  |
| vma.mp4 | 720 | person | 0.7241 | 5 | 0.5 to 0.6246 |
| vma.mp4 | 720 | phrase A | 0.0029 | 0 |  |
| vma.mp4 | 720 | phrase B | 0.0256 | 0 |  |

The score column holds only the highest eight scores of a request, so its
lower end is not the lowest score where there are more objects than that.

For phrases A, B and C the presence score is near zero on every frame
tried, where it is high for `person` on the same frames. Meta's detector
multiplies each query's class score by presence (`sam3_image.py`, the joint
score under `supervise_joint_box_scores`) and drops what is at or under its
threshold (`sam3_multiplex_base.py`, `pos_pred_mask`), so a phrase with
presence that low returns nothing whatever its class scores are. ComfyUI
core's port is read as dropping presence
(`docs/research/masking/2026-10-06_mrsun.md`), which is the difference
between the two on a single frame with the same weights. Not measured: the
class scores themselves.

One consequence, stated without ranking either port: the phrases that picked
the subject for the day's renders of this clip did so on a port that drops
the presence score. Phrase C, the object alone, was tried as a second
concept to join to a person and is rejected the same way on this clip.

**Two rows of the earlier table test nothing**, and so does half of the
probe. The rows at an output threshold of 0.05 are after the detector's own
gate, and are kept only so nobody runs them again. The probe's second arm
set the two detection thresholds and the output threshold to zero on the
built model to list every query's score; it returned exactly what the built
model returns on every row, so some gate remained that was not read, and it
gives no score under the built ones.

**The same probe says why a session can track nothing on frames where a
one-frame request finds objects** (scores measured, the threshold read from
code, the link inferred). On the later frames of the table `person` returns
objects at scores under the builder's `new_det_thresh`, the score a
detection needs before a session starts a new track from it. A one-frame
request has no such step.

## The second clip's sessions

| run | frames | phrase | cap | result | ids | frames with no object | first frame with an object | propagating, s | peak over what was loaded, GiB |
|---|---|---|---|---|---|---|---|---|---|
| main | 600 to 760 (161) | phrase A | 16 | ran | 0 | 161 |  | 15.05 | 5.42 |
| main | 0 to 48 (49) | person | 64 | ran | 21 | 0 | 0 | 7.27 | 6.41 |
| main | 0 to 48 (49) | person | 16 | ran | 16 | 0 | 0 | 6.15 | 6.03 |
| follow-up 1 | 600 to 760 (161) | person | 64 | OutOfMemoryError |  |  |  |  |  |
| follow-up 2 | 640 to 720 (81) | person | 64 | ran | 2 | 73 | 713 | 7.97 | 4.99 |
| follow-up 2 | 640 to 720 (81) | person | 16 | ran | 2 | 73 | 713 | 7.72 | 4.98 |

- The cap is the builder's `max_num_objects` (measured on the two short
  sessions: the session at the lower cap holds exactly that many ids and the
  one at the higher cap holds more, at a small cost in seconds and memory).
- `person` over frames 640 to 720 returns no object until late in the
  stretch, at either cap (measured). The session began at frame 640, so
  this does not say what a session begun earlier would hold there. The
  probe above gives the likely reason: detections on those frames score
  under the threshold for a new track.
- What a subject's id does across the frames where the lane's tracker loses
  it was the question, and it is not answered: the phrase that names that
  subject returns nothing, and the `person` session over the whole stretch
  at the higher cap ran out of memory.

## Off the card and back

| state | allocated GiB |
|---|---|
| loaded | 3.654 |
| after `.to("cpu")` | 0.384 |
| after the position-encoding caches are dropped | 0.013 |
| after the loose tensors that were on the card are moved | 0.008 |

The model holds tensors outside its parameters and buffers, which `.to()`
does not move:

| attribute | held on | tensors | MiB |
|---|---|---|---|
| buffer_cpu_batched | cpu | 4 | 192.0 |
| compilable_cord_cache | cuda:0 | 1 | 0.0 |
| freqs_cis | cuda:0 | 8 | 5.1 |
| freqs_cis_imag | cuda:0 | 8 | 2.5 |
| freqs_cis_real | cuda:0 | 8 | 2.5 |

Measured: with the position-encoding caches dropped and the card-side loose
tensors moved, the process holds almost nothing on the card; the model is
back in a fraction of a second; and a short session before and after the
round trip gives equal masks, the caches refilled by the session itself.

**Moving every loose tensor to the card breaks the model** (measured, by
accident). `buffer_cpu_batched` is kept on the host by Meta's code; moved to
the card, the next session raises in `_postprocess_output_batched` (the
json's `round_trip`). The equal-masks result is from the second attempt,
which put back only what had been on the card.

## The box prompt

On a fresh session with no phrase, two points labelled as a box's corners
under an object id return no object, and propagation raises (the json's
`box_prompt`), on both trees (measured). The reading that such a request
prompts one id by its box does not hold for a fresh session. Not run: the
same request after a phrase in the same session.

## What went wrong in the run

- **The first process ran at Meta's detector batch** and ran out of memory
  on the opening shot. The run script then held the failed session's memory
  past its own cleanup, so every later stage of that process failed for
  memory too. It was stopped and its rows are not in the tables.
- **The round-trip helper broke the model in the main run** (above). From
  that stage on, each session failed first on the committed tree for that
  reason and ran on the earlier scratch tree, with both models on the card.
  The `tree` column says which. The session across the cuts then ran out of
  memory, which is why it is not measured.
- **A by-product, not a design:** where the same frames ran on both trees
  the masks are equal (the json's `trees`). The trim of the builder changed
  no output there.
- **The main run was stopped with the wrong process id** before its last
  stage, and the process held the card some seconds longer than the slot's
  owner had been told.
- **An output threshold was reported as a test of the detector** before the
  gate in front of it was read. Withdrawn above.

| run | session | tree | frames | error | allocated before, GiB |
|---|---|---|---|---|---|
| main | unload_after__failed_on_proto | proto | 40 | TypeError | 3.476 |
| main | opening_offload__failed_on_proto | proto | 501 | TypeError | 7.231 |
| main | across__failed_on_proto | proto | 611 | TypeError | 7.231 |
| main | across | first | 611 | OutOfMemoryError | 7.231 |
| main | box__failed_on_proto | proto | 501 | RuntimeError | 7.231 |
| main | box__failed_on_first | first | 501 | RuntimeError | 7.231 |
| main | box | first | 501 | RuntimeError | 7.231 |
| follow-up 1 | vma_600_760_person_cap64 | proto | 161 | OutOfMemoryError | 7.301 |

## Not run

The json's `not_run` is the list. The two that bear on a design: what
bounds a long shot's memory, and what ids do across a cut inside one
session.
