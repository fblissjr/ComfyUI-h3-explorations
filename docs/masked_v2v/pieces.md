# The pieces, in the order a render meets them

Four nodes make a masked render: who the person is, what happens to their pixels, the text, and the render. Each
node's module docstring is the authority and its `define_schema` tooltips say what an input does; this page says what
each piece decides and links to the evidence. The lane as a whole is [`overview.md`](overview.md); the data flow of a
job is [`pipeline.md`](pipeline.md).

After the four pieces: why nothing is kept between runs, how the latent grid ties frames together, and the zoomed
render.

## The pieces, in the order a render meets them

### 1. `MiniMaxH3SubjectTrack` (`subject_track.py`): which pixels are the person

One mask per frame, empty where the person is not on screen. It is built on SAM 3.1, and everything about SAM itself
(what to type as a phrase, why a description does not pick one person out, the runtime corrections, how our tracker
differs from ComfyUI core's text-prompted tracker, cut finding, the pick, the match across a cut, regain, seeds and
corrections) is [`sam31.md`](sam31.md) and its code map [`sam31_core.md`](sam31_core.md). What the lane relies on:

- **Shots first.** The tracker finds the source's cuts and decides each shot on its own (`subject_track.py::auto_cuts`).
- **The pick is a first guess.** The shipped rule is `most central` (`h3_config.SUBJECT_TRACK`); no rule is right
  everywhere (`../../bench/results/2026-10-07_subject_track_calls_on_masks.md`). The preview tile shows who was taken,
  and a person's answer settles it: a typed correction (`shot 3: person 2`, `shot 3: none`) or a seed box from the
  cast file (`subject_track.py::parse_seeds`; the frame is the load's own, so whoever writes a seed subtracts the
  load's first source frame).
- **The same person in another shot is a likeness, not an identity.** The match is SAM's trunk features, which say
  what a thing is, not who; how a named person is recognized again is [`who_is_who.md`](who_is_who.md).
- **One node follows one person.** Two people are two Subject Track nodes; a later tracker is told who the earlier
  ones hold (`others`). The second person's mask goes to the Masked Source's `others` when the two stand close.
- **What it shows**: one numbered tile per shot and a shot table (`shot_table.py`, whose docstring is the schema). A
  person's number is theirs on that shot's shown frame only.

### 2. `MiniMaxH3MaskedSource` (`video_mask.py`): what happens to those pixels

- **The region.** The mask is grown by `grow_pixels` and turned into a token mask (`video_mask.token_mask`). A token
  is regenerated or kept whole, in space and in time: the video VAE packs frames in runs (`video_mask.run_lengths`),
  so a token covers several frames (the latent grid, below).
- **`replace`.** `whole subject`; `head and hair`, which keeps the body's pixels and finds the part with SAM 3 from
  `part_phrases`; or `the wired parts`, a part of the subject the Sapiens2 part node finds (`MiniMaxH3SubjectParts`,
  `sapiens2_parts.py`). The part classes are not things a person would choose; which are safe to offer is in
  [`perception_models.md`](perception_models.md), "The part classes are not things a person would choose".
- **`edge`** (a trial, off by default). `whole tokens` is the rule above; `latent cells` hands the sampler the mask
  per latent cell, half a token's side, so a kept cell inside a regenerated token is the source's in the result
  (`video_mask.EDGES` has the provenance; `bench/check_video_mask.py` item 15 holds it). It aims at the rounding that
  makes the region near twice a small subject's own area. The built graphs that use it are listed by
  `grep -l '"edge": "latent cells"' data/*_job/*.render.json`. No render with it has been read beside the same render
  without it, so what it changes in a picture is not known.
- **`keep`** (optional). A second mask of what must stay the original inside the region: something the subject holds,
  a person standing close, anything passing in front. It is taken out of the token mask after the grow, in whole
  tokens, so a little of what surrounds a small object stays too, and the model draws around it. Refused with
  `paint_out` or a softened start. **Do not `keep` something that lies inside or against the part being replaced**: a
  kept token is whole, so it holds the skin beside the thing too, and the face then moves back toward the original's
  (one clip, one window, one seed, 2026-10-10). Give such a thing back at assembly instead (`restore=` in
  `bench/assemble_delivery.py`), where the sampler never sees it. The lane's rule on keep-outs on a subject's own body
  is in [`data/AGENTS.md`](../../data/AGENTS.md), "Settled for the masked lane".
- **`motion_video`** with `motion_reference` on `a video I wire` (`video_mask.MOTION_WIRED`): a video that runs beside
  the source frame for frame and is shown to the model as the movement: a body mesh of the original, a pose, a
  drawing of the face points. Each window is shown its own frames of it (`video_mask.wired_motion`), which a reference
  video appended to the chain is not. With `motion_vae` on, the video model has its own copy at every frame; off, the
  text encoder sees two frames a second. The text has to say what `<Video 1>` is and what is taken from it
  ([`text.md`](text.md)). A body mesh for it is made from the track with `MiniMaxH3SubjectBoxes`
  (`subject_boxes.py`), one box a frame round the tracked mask, so the mesh is of the tracked person. The reference
  form the lane uses for a face, head or hair replacement is in [`data/AGENTS.md`](../../data/AGENTS.md); how it is
  made and what it carried is [`renders_by_form.md`](renders_by_form.md). What the body model gives and does not (no
  mouth) is [`perception_models.md`](perception_models.md).
- **`motion_video_2`**, with its own `motion_short_edge_2`: a second wired video, shown as the next video reference
  after the first, for a text that gives each video something of its own to carry
  ([`docs/prompting/prompting.md`](../prompting/prompting.md), section 9). `motion_vae` is one switch for both, and a
  second with no first is refused. In a job it is the load's `motion_file_2` with `motion_first_2` and
  `motion_edge_2` (`../../workflows/masked_job_graph.py`). Not rendered with when this was written.
- **`render_region`**: the whole frame, or zoomed in on what is replaced ("A small subject, rendered zoomed in",
  below).
- **A setting it will refuse is refused when the graph is queued**: a feather wider than the margin, or a softened
  start with `paint_out` (`video_mask.settings_refusal`). A refusal that needs a tensor (`keep`, `others`) is made at
  run time.
- **What is encoded.** The source frames themselves; with `paint_out`, a copy with the subject filled in from its
  surroundings (`video_mask.fill_subject`). `paint_out` was judged worse and stays off.
- **The composite.** After the decode the source's pixels come back outside the region. `whole region` keeps all of
  H3's picture inside the region; `only what changed` (`video_mask.changed_alpha`) keeps the render where it differs
  from the source or where the old subject stood. The job chain's choice is `h3_config.MASKED_LOAD_COMPOSITE`.
- **The held tail.** A window past its load's last frame holds that frame with its mask, so the region stays open
  there (`video_mask.held_mask`); why is [`window_context.md`](window_context.md), "The held tail of a short load".

### 3. `MiniMaxH3MaskedPrompt` (`masked_prompt.py`): the prompt

The node writes the text from who the still shows (`subject`, a few of the user's own words), whether they are the
voice on the track (`voice`), and what the still provides (`picture_gives`). It works the rest out from the Masked
Source wired into its `source` (what is replaced, whether a motion reference is on), so the text and the Masked Source
cannot disagree. On every run it shows one line per input saying what it did with it, then the text.

- **The text says nothing it cannot know.** The node writes without seeing the clip, and H3 acts on words: a clause
  written to cover a case is read as an instruction. So it writes nothing of the clip's look, of how long the subject
  is there, of what else is in the scene, or of an edit (what stays, what is "the scene's own"). The audit behind it is
  [`../../bench/results/2026-10-10_prompt_text_hardcoded_in_code.md`](../../bench/results/2026-10-10_prompt_text_hardcoded_in_code.md).
  The upper-body role keeps four sentences of those kinds until a render with and without them is read
  (`masked_prompt_text.py`, `HELD`).
- **The job chain does not use this node's wording for a load.** A load's text is made by `bench/load_text.py` from
  the reference's record and the plan; the frame every text is written in is [`text.md`](text.md). The node's lines
  for a motion video (`masked_prompt_text.VIDEO_DEFINITION`, `VIDEO_RETENTION`, `MOVES`) take only body movement
  from it.
- **Set `subject` to match the still**, and say only what is in it: where a text and a reference disagree the text
  wins ([`docs/prompting/prompting.md`](../prompting/prompting.md), "Silence is not neutral"). Nothing beyond single
  words has rendered in this lane (`../../bench/results/2026-10-06_masked_v2v_person_text.md`).
- **`the wired parts`** can be any part, so the node asks for `picture_gives` and writes nothing until it is set.
- **Where the sentences live**: constants in `masked_prompt_text.py`, whose docstring says which have rendered.
  Changing one changes every graph that wires the node; `bench/check_masked_prompt.py --write` rewrites the bank's
  copies. The graders read what the encoder reads: `workflows/prompts.py::carriers`.
- **Writing a prompt by hand** is possible: type it into the song node's `prompt` and leave this node out.

### 4. The song node (`audio_freeze_song.py`): the render

Each window starts from the source's frames over its span (`video_mask.window`), regenerates the masked tokens with
the track's audio frozen, and composites. A window whose mask is empty is written from the source without sampling.
One sampler per window is what carries the mask; a two-sampler graph does not carry it as wired
(`MiniMaxH3RestorePlate`, `plate_restore.py`, exists for it and is in no shipped graph).

- **The mask review**, `<prefix>_NNNNN_with_mask.mp4` (`save_mask_review`, on by default): the render on top and the
  source below with what was regenerated coloured in, drawn from the token mask the sampler was given, so the grow
  and the snap to tokens are in the picture (`video_mask.overlay_pieces`; `bench/check_video_mask.py` item 12).
- **What a render will be given, and what it was given.** A preview (`preview` on) samples nothing and writes
  `<name>_plan.json` and `<name>_window_N_planned_region.npz` (`audio_freeze_song.write_plan`); a render writes
  `<name>_window_N_region.npz` beside each window's latent (`video_mask.save_window_region`). Where each file is read
  is [`running_a_job.md`](running_a_job.md), "Seeing what the tracker and the masks did".
- **What a reference costs**: for each window, a line a reference with the tokens the text encoder is given for it
  and the rows of the video model's own copy, then one line for all of them beside the prompt and the window's own
  rows (`audio_freeze_song.reference_costs`, counted from sizes by `reference_report.py`; `bench/check_video_mask.py`
  item 17). A plan prints the same lines, counted with no text encoder loaded, so the prompt is not counted there and
  the render's lines are the ones to keep.
- **Windows.** A load one window holds is given the shortest length of the grid that holds it; a longer one runs in
  chained windows, each continuing from the frames of the one before. What ties a window to the one before it is
  [`window_context.md`](window_context.md); how a window's length is chosen is
  [`working_rules.md`](working_rules.md), "Rules of the lane".

## Nothing is kept between runs by a key

The repo's rule (root [`AGENTS.md`](../../AGENTS.md), "Nothing an earlier run made is reused by a key"): a result is
made again, or handed in explicitly as a named file or a wired input. What this lane once kept, and what became of
each:

| what | where it was | now |
|---|---|---|
| the finished mask | `output/masks/` (`mask_store.py`, `reuse_mask`) | removed with its input, module and check: its key could not see a change to the code that makes a mask, and a stale mask hid a fix on 2026-10-07 |
| a rendered window, reused | `loop_resume.py`, `reuse_windows` | the keys and the reuse removed. A window's video and its sampled latent are still written on every run, and read again only when a person names them (`continue_from`, `bench/recomposite_window.py`) |
| a window's source and prompt encodes | the server's memory (`window_keep.py`) | removed with its module and check |
| the shot table | `<prefix>_NNNNN_shots.json` and the Subject Track's `shot_table` output | unchanged: a file a person wires into `subject_from`, the explicit form the rule allows |

A mask made once and used many times is a mask video a job's graph loads from a file its capture wrote: named,
visible, and replaced when the capture is made again. Whether a render tracks again is core's own node cache's to
say; a job's render loads the mask videos its preview wrote.

**Three things called a version, and none is the other.** The pack's version is the newest heading in
`CHANGELOG.md`, assigned by `bench/build_changelog.py` (`pyproject.toml`'s `version` is kept equal to it).
`MASK_VERSION` is one integer per tracker node class (`subject_track.py`, `sapiens2_parts.py`), bumped only when that
node would make a different mask from the same inputs; the numbers on different nodes are unrelated, and the job
builder reads the Subject Track's for its corrections rule. `TABLE_VERSION` (`shot_table.py`) is the shot table's
file format, checked on read. A node's identity is its `node_id` in `bench/node_id_manifest.json`.

## A latent step is several frames, and its grid is fixed for a whole load

The video model does not work in frames. Core's `FRAME_PER_TOKEN` (`comfy/ldm/minimax/model.py`) says how many pixel
frames each latent step covers, and it cycles: `video_mask.run_lengths` is that cycle laid along a window. Everything
this lane decides per step it decides for every frame of the step: the region (`token_mask` takes the maximum over a
step's frames), what `keep` and `others` hand back, the weight the composite lays the render with. A frame cannot be
regenerated on its own, and a frame cannot be left alone while its neighbours in the step are redrawn.

**The grid does not move from window to window.** A window's length is on `loop_plan.CHAIN_LENGTHS` and its context
on the values `loop_plan.check_window_settings` allows, so what a window adds is a multiple of `loop_plan.GRID`, a
whole number of cycles. Every window of a load cuts its steps at the same places counted from the LOAD's first
frame. Two things follow, both arithmetic that needs no render:

- **Which cuts of the source split a step is known before anything is queued**, from the cut list and the load's
  first frame alone. A cut inside a step puts one to three frames of the other shot under the same region. On the
  first clip this was looked for, the frames the assembler found repainted at four cuts were exactly the frames this
  arithmetic names, and the three cuts it puts on a step's edge had none (`data/CAPTURE_GAPS.md`, U5; the capture's
  flag is `region_carried_across_a_cut`).
- **The load's first frame is a lever, for a load with few cuts.** Moving it by fewer frames than the cycle is long
  moves every step's edge with it, so the cut that matters most can be put on an edge (`loop_plan.split_steps`
  lists the steps a load splits; `bench/check_audio_freeze.py`, the step grid case). All of a load's cuts cannot be
  cleared in general: the chance that one start clears several cuts falls with each cut.

**The cycle is the video VAE's, and the decode is not local to a step** (read from core's
`comfy/ldm/minimax/vae.py`, 2026-10-10, nothing run):

- The VAE cuts a video into clips of one cycle of `FRAME_PER_TOKEN`, so a load's first frame fixes the VAE's clip
  boundaries as well as the step edges. They are one grid.
- **Encode: each clip by itself**, causally: a latent step is shaped by its own frames and earlier frames of its own
  clip, by no later frame, and by nothing outside its clip.
- **Decode: a window of steps at once.** The decoder attends over a clip's tokens and the head of the next clip,
  inside a spatial tile, and cross-fades each clip's first frames from the previous window's decode.

So no frame beside a cut is decoded from its own step alone, on either side; that is how any clip with a cut decodes
and is not a fault by itself. What is special to a split step is the latent: one token holds frames of two shots,
and inside the region the sampler has to make that token up. Whether the frames on the subject's side of such a
token are worse than their neighbours is not measured.

**What the lane does about a split step**: the composite lays nothing on the frames of a step that lie across a cut
from every frame the subject is on (`video_mask.cut_gate`; the song node's report names them). The sampler still
regenerates those cells, and the case where the subject is on both sides of the cut in two different places is open.
A seam between windows is a different thing from a cut: the frames a window shares with the one before are frozen
latent steps copied whole, which is why the context is a length that ends on a step's edge.

**Laying a piece by its matte at the source's size: one weight, one pass.** The node saves its own blend weight (the
matte: 255 where H3 regenerated, the feather between, 0 outside), and `bench/assemble_delivery.py` lays the piece by
it. At the canvas's size that is exact. At the source's size the law is one weight scaled once: per piece, a map at
the canvas that is 1 where the piece owns the pixel and the matte's value on its band, scaled up once; then
`new = beneath + weight * (H3's picture - beneath)`, where beneath is the source or an earlier piece, which is what
makes overlaps right. Laid as two passes (the full region, then its soft band, each scaled on its own) the two
scaled weights do not add back to one where they meet, and up to a quarter of the change falls back to what lies
beneath: a ring of the original inside every piece's feather (up to 34 of 130 levels at this video's crop scale; the
second read of 2026-10-16, `internal/claude/2026-10-16_mrtony/review2_matte_d18.md`, probes `t/seam.py`,
`t/seam13.py`; the reader's one-pass form put the ring at 0). Use H3's own picture there, unmixed from the saved
piece, not "the sharp source plus the piece's change" (`under + (up - was)`, how the matte is laid at the source's
size as of 6f3eafd6), which at full weight adds the original's high-pass between the canvas and the source, the old
person's fine edges on the new face (the D23 Gemini's critique, 2026-10-16; not yet measured). The case that guards it
measures the feather's inner edge. The lane's rule that every piece is laid by
its own matte is in [`data/AGENTS.md`](../../data/AGENTS.md) (`bench/assemble_delivery.py::lay_plane`). Not in
latent space: the VAE works at the canvas, and a decoded frame would carry the untouched picture through the VAE's
round trip.

**Everything else that is held over a step** (an audit of the lane's code, 2026-10-10; read from the code, the one
measured row says so):

| place | what it holds over a step | what that does at a cut inside the step | named by |
|---|---|---|---|
| `video_mask.token_mask` (the region) | the maximum of the subject's cells over the step's frames | the region lies on the other shot's frames and the sampler regenerates them | the preflight's `region_carried_across_a_cut` (`loop_plan.split_steps`); the song node's report |
| `pixel_alpha`, the composite's weight | the step's region on each of its frames | the render is laid on the other shot's frames | fixed: `cut_gate` in `lay_window` |
| `changed_alpha`'s hold | the maximum over the step of "changed, or the old subject stood here" | the subject's own frames keep the whole region and less of the margin is restored (measured, [`2026-10-10_saved_windows_laid_again.md`](../../bench/results/2026-10-10_saved_windows_laid_again.md); its tables' data are under the untracked `data/2026-10-10_recomposite_decodes/`) | nothing; `data/CAPTURE_GAPS.md` U6 |
| `keep` in `window` | a token with a kept pixel on any frame of the step is kept on all of them | with the subject on both sides, one shot's keep leaves the source showing inside the other shot's subject | the preflight's flag for a subject on both sides of a cut in one step |
| `others` in `window` | the same hold, taken out unless the subject's own mask has a pixel in the token | the same | the same flag |
| `cut_gate` | which side of each cut the subject is on, per step | with the subject on both sides it gates nothing | the same flag; open (a region per side) |
| `start_zero_tokens` (a late start) | the body's hole and the top share kept, as maxima over the step | the other shot's frames start from the same emptied cells | nothing; only with `start_from` other than noise |
| the mask review's region layer (`token_region`) | the region the sampler was given, per step | under `whole region` a gated frame still shows the region tinted though nothing was laid on it | a reading caveat, not a fault |
| the capture's planned region | per frame, where the sampler's is per step | a plan's "inside the region" figures are a floor, most of all at a cut and across a fast move | `data/CAPTURE_GAPS.md` 30 |
| the frozen context | whole latent steps copied from the window before | after a cut the context is another shot's picture | one load per shot |
| the video VAE | clips of one cycle, encoded apart and causally, decoded a window of steps at a time | both shots share a decode window; ordinary | the paragraph above |

Not held over a step, checked: the motion reference and a wired motion video are cut per frame in step with the
source, a short one holding its last frame; the margin under `the subject's size` is each frame's own; the zoom's
box is per load; the trim and what a window writes fall on a step's edge.

**Why a window follows its context.** A row at mask 0, a context row or a plate row alike, is conditioning inside
the picture, as clean and as trusted as a reference. The mechanism, every lever that loosens it (`context_noise`
among them) and the dated results are [`window_context.md`](window_context.md).

## A small subject, rendered zoomed in

**Why.** A subject small in the frame is a few of the model's tokens. `video_mask.RENDER_WHOLE`'s comment holds the
case. The job chain renders zoomed in where the node's box allows, by default (`bench/shot_to_load.py`, the defaults
a load carries; `--whole-frame "why"` or the plan's `zoom: false` for the whole frame).

**What it does.** Under `render_region` zoomed in (`video_mask.RENDER_REGIONS`) the song node crops the load to ONE
box round what is replaced (`video_mask.zoom_box`, never smaller than `video_mask.ZOOM_LARGEST` allows), scales it to
the canvas (`video_mask.zoomed_source`), and runs everything as on any clip: margins, token region, motion
reference, encode, sampler, decode and `lay_window` all see a clip that happens to be a crop. `lay_back` then puts
the region into the whole frame, so a window's video and the joined file are the whole frame as ever.

**What it buys, and what it does not.** Tokens for the subject while it is drawn, not delivered pixels and not time:
the window is still sampled at the canvas. For a delivery at the source's size the laid picture in the zoomed space
is written beside each window and joined (`<stem>_zoomed.mp4`, with `<stem>_zoomed.json` saying the box;
`audio_freeze_song.zoomed_record`).

**One box for the whole load.** A continuation's kept frames are the window before's latent, and a box held across a
cut frames nobody on one side, so the box is fixed over every frame loaded, a load with a cut inside it is refused,
and a zoomed run continues nothing and is not continued. A subject who travels across the frame makes the box most
of the frame: then the union of her part's boxes with `ZOOM_ROOM` sets the zoom, not the cap, and the report's line
says which of the two set the box (read it before queueing a second arm). Under `ZOOM_SMALLEST` the load is rendered
on the whole frame and the report says so. A two-subject load is not rendered zoomed in.

**A wired motion video goes in as it was wired.** The zoom does nothing to it: the box is from the replaced part
alone, and a zoomed window is shown the same reference a whole-frame window would be (`video_mask.zoomed_source`,
`wired_motion`). A reference with a framing of its own, the body model cut to one box round the head
(`bench/head_reference.py`), is the same reference here, which is what the pairing is for. A whole-frame body model
would show the whole frame beside a plate that is the box enlarged, so the job builder refuses the preview's own body
model on a zoomed load and takes only a `motion_file` the load names (`workflows/masked_job_graph.py::RENDER_REGIONS`).

**The box is in the pixels the node was handed.** The job graphs' loader fits the clip to the canvas before the
Masked Source (`workflows/masked_job_graph.py::loader`), so the zoom is enlargement of canvas-size frames, not a
sharper picture; the report's line says which, and the files carry `source_size`. Only the region is laid back, so
the plate's softness is in no delivered pixel. It goes away only when the loader hands the clip's own frames.

**What changes meaning.** `grow_pixels` and `feather_pixels` are pixels of the zoomed picture, so on the whole frame
they are divided by the zoom (the report gives both). A window's region file and a preview's plan are of the zoomed
space and carry the box under `zoom` (`video_mask.load_window_region`); a reader that lays them or counts their cells
maps through the box or refuses. The mask review of a zoomed run is of the zoomed picture, and its legend says so.

**Checks.** `bench/check_video_mask.py` item 24 and the zoomed case of its song loop; `bench/check_recomposite_window.py`'s
zoomed window; `bench/check_masked_job.py` for the builder's rule and the build record's line.

**What has been seen.** The first zoomed render (one street shot, one load, one seed, 2026-10-10) ran clean: read
from its own files, the plate is the loaded source to the same figure as the whole-frame render of the load, inside
the box and outside it ([`2026-10-10_saved_windows_laid_again`](../../bench/results/2026-10-10_saved_windows_laid_again.md),
`first_zoomed_render_read_from_its_files`); in stills the face is replaced where it shows and no edge of the box
shows. What renders since have shown at this zoom is in each video's verdicts, not here.

**Not built.** A box that follows the part through the load (the subject still in the zoomed picture, the
surroundings sliding; every file and the delivery tool would take a box a frame). The cheaper things first: a smaller
`ZOOM_ROOM`. And a preview stage that runs zoomed, so the part model's and the pose model's readings of a small
subject are still made at whole-frame size.
