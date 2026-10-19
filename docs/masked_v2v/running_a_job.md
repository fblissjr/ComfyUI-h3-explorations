# Running a job, and the tools around it

How a masked load is made, from a shot to a render, with the tools that hold each step, and where to look at what a
run was given and what it wrote. The data flow stage by stage is [`pipeline.md`](pipeline.md); the commands for a day
(one load, the review server, the rebuild, the close) are [`docs/working/runbook.md`](../working/runbook.md), section
(b); the working rules are [`working_rules.md`](working_rules.md).

**The chain.** `bench/shot_to_load.py` makes a load's job file from the video's records (`preview`, `job`,
`reference`, `text`), and `bench/load_chain.py` takes that job file through every later step (`first`, `plan`,
`on-plan`, `table`, `no-look`, `render`, `land`, `page`). Their docstrings are the reference: what each step reads,
refuses and prints. Each job step prints THE DEFAULTS THIS LOAD CARRIES (`bench/shot_to_load.py::defaults_carried`)
and keeps them in the job file. Under both, `bench/masked_job.py` builds and queues the graphs and
`bench/capture_masked_run.py` makes the capture and its gate.

## Running a job: one tool, on the video's analysis folder

A masked job's graphs are built and queued by `bench/masked_job.py`, from a job file, and by nothing else; the chain
above drives it. The tool's docstring is the reference; `bench/check_masked_job.py` holds it. What a session needs to
know:

- **It starts from the input video's analysis folder.** `data/<stem>/`, named by the video's file name, made by
  `bench/clip_analysis.py init` (`clip_folder.py` has the rules of that folder). The builder refuses a job whose video
  has none, or whose file is no longer the one the folder was made from (it is held to the file's size and head hash:
  an identity, not a store).
- **The stages, each a `build` and a `queue`:** `preview` (who is where: one tracker a subject, each told who the
  earlier ones hold; parts, class maps, pose, mesh; nothing sampled), then `collect`, which fails by name when a file
  is missing or a subject was found in no shot; the capture and its gate (`bench/capture_masked_run.py`); `plan` a load
  (the song node says what each window will be given), which the capture reads; `render` a load, refused unless the
  capture passes `verify` again, its gate is clear, and the gate read the node's own plan for that load.
- **Nothing an earlier run made reaches a load by a key.** A load's masks are files of the capture the job file names,
  read by path; its mesh is the file its own job's preview record lists, or the one the load names (`motion_file`).
- **A load made again is a new job name on the same capture.** The tool never reuses a name, and a capture can serve
  any number of jobs.
- **A load is one subject on one shot**, written in the capture's words, with its text given to it. It is no shorter
  than the song node's shortest extent (`bench/masked_job.py::shortest_load`): a shorter stretch is left as the
  source, with why, or joined to a neighbour. What a load carries by default, and why each value, is the
  `MASKED_JOB_*` block of `workflows/h3_config.py`; a value measured on one clip is a starting value, a job file may
  set any of them, and a load's build record says which were set by the job and what the capture measured.
- **A load's text is data of the job, tied to the frames it was written from.** In the job file a text is a file
  beside it, the source frames it was written from and who wrote it. One text for loads of different shots is
  refused, by name, and so are the same words under two labels. A load holds no cut (`bench/masked_job.py::ONE_SHOT`,
  an edge accepted, not a rule of the model) unless its job block names `two_shots` (below). How a text is made is
  [`text.md`](text.md).
- **Somebody looks before a plan and before a render, and the look is shown, never refused** (the owner, 2026-10-12,
  Open List row c22). `build plan` and `queue` of a render record the look the capture's `looks.json` holds for the
  load, later than the capture was last written, that showed this load's frames with this text
  (`bench/masked_job.py::looked_at`), or that none is recorded, with why and the strip's path. The strip and the
  record of the look are `bench/capture_masked_run.py` `video` and `looked`. `bench/load_chain.py no-look` records
  that no look is owed for an ordinary load on its maker's statement that it is not the first of its kind.
- **A hair band under a head-and-hair part** (a load's `hair_band`, a share of the part's height, with no default): a
  still whose hair falls past the original's jaw is drawn into the margin under the part, where a clothing keep-out
  gives cells back frame by frame, so the region's lower edge moved. With the key the builder writes a copy of that
  keep-out with a band cut out under the part and wires the copy (`bench/masked_job.py::hair_band`; constants and
  provenance `h3_config.MASKED_JOB_HAIR_BAND`). `bench/read_hair_band.py` prints a band as figures a frame, for the
  band load and a pair without it, and gives no verdict. Default keep-outs on a subject's own body are retired
  ([`data/AGENTS.md`](../../data/AGENTS.md)).
- **A load of two subjects in one pass is a closed lane** ([`docs/state/roadmap.md`](../state/roadmap.md), "Closed
  lanes", 2026-10-15): the first render, `second_lead_0852`, blended the two faces. The tooling stays in the tree,
  unused (`bench/shot_to_load.py pair`; a load's `subjects`, a list of two, read through
  `workflows/masked_job_graph.py::one_subject_view`, the first subject's still `<Picture 1>` and reference
  `<Video 1>`). What it showed of the model: the text alone tells the model which still is whose; nothing in H3 or in
  this pack binds a reference to a hole (`comfy/text_encoders/minimax.py`, the loop over `minimax_ref_items`;
  `comfy/ldm/minimax/model.py`, `PackedLayout`). The next test the owner named, should the lane reopen: one hole a
  pass, the second subject's load, then the lead's load given that render as its video.
- **A cheap variant of a built load**: `bench/sweep_loads.py` writes copies of job files with one named change (a
  still, a text, a load's motion video, or one setting of every load) and new names, proves each copy differs from
  its source in that and the names only, and prints the builder's commands in order (`bench/check_sweep_loads.py`). A
  copy still goes through plan, the capture, the look and the gate like any load.
- **Not built:** a one-window patch (`bench/patch_render_window.py build` is the only route, and is not checked against
  the chain); the analysis folder's own passes, its `profile.json` and the rules that set a load's values from the
  clip.

## A shot, made whole: the recipe the owner kept on 2026-10-15

The hall shot 1419-1465 was made this way and the owner kept it on playback, then asked for it to be applied wherever
it is a reusable pattern. It replaced six short pieces with joins and stretches of the original between them. The
steps, each with its tool:

1. **One load a subject over the whole shot** ([`docs/state/decisions.md`](../state/decisions.md), 2026-10-15, one
   shot is never split). Up to `h3_rules.MAX_LENGTH` frames it is one window; longer, chained windows inside the one
   load, each on `h3_rules.chain_lengths()`, the seam put on a quiet frame with `shot_to_load job --seam`. The job
   step's context is `h3_config.MASKED_JOB_CONTEXT`; `context_noise` on a load of several windows is a person's word
   (`--context-noise X --context-noise-why WHY`; [`window_context.md`](window_context.md), "Results, dated").
2. **Read every frame of it for each subject first**, on a sheet: where the face shows, where it is hidden, turned away
   or out of the picture. Hidden and turned-away frames stay inside the load, with no face drawn there
   (`--no-face-on A-B=WHY`): a face drawn on the back of a head is the 5461 fault. A frame where the face shows and
   the part model labels nothing takes `--hold-part A-B=WHY`. The reference's printed cues point at frames to look
   at; they decide nothing.
3. **Voice from the plan, checked by watching and listening** ([`text.md`](text.md), "Say the least first").
4. **The text as the generator writes it today**, read whole from the queued graph (`bench/what_h3_got.py`) before it
   renders.
5. **Another of our subjects inside the margin is given back at assembly**, never drawn by this load (`restore=` on
   the table), and gets a load of her own if the owner wants her swapped.
6. **A joined clip of the shot with only the new pieces laid**, beside the cut as it stands, for the owner on
   playback; laid in the table (`--prefer`) only when they keep it.

**A frame the part model misses is missed for everything read from it** (2026-10-15, 2823c's 3098-3104): the face
part, the classes given back at assembly (Eyeglass, hands, the earrings' Apparel@head) and the class-map mouth the
reference draws all come from the same class map, so a frame that `--hold-part` fills for the face has the same hole
in each of them. Until the tool carries them itself, read a held span's restores and mouth on the plan look, not only
its part.

What it does not cover yet: continuing a piece the owner kept (a stored window off the chained grid cannot be
continued, 2823b), and two subjects who touch.

## Two neighbouring shots in one load: a tool, not a default

Built 2026-10-13 at the owner's word ([`docs/state/decisions.md`](../state/decisions.md)). One load a shot is still
what every tool makes: a join at a cut costs nothing, a cut inside a latent step is a known fault
([`pieces.md`](pieces.md), "A latent step is several frames, and its grid is fixed for a whole load"), and a long
window costs more than two short ones. This is reached for on a person's word for one pair of shots.

**What it could buy**: one face for a subject across a cut where two shots of them sit side by side, and a tough
shot helped by an easier one beside it. Neither is shown: no load had rendered this way when this was written. The
owner's account of why it may suit the model is that H3 expects several shots inside its longest length; what was
measured bears only on its first half (one short shot rendered at a long window and at its fitted one, `lead_fp_1920c`
and `lead_fp_1920d`, one pair, one seed: the fitted one at least as good).

**The opt-in.** A job block's `two_shots`: the source frame the second shot begins on, the two shots, why they share a
window and who said so (`workflows/masked_job_graph.py::two_shots_of`). With it:

- the load is held to one subject, one window that holds both shots whole, the named cut inside it, not zoomed and
  not a continuation (`masked_job_graph.check_load`);
- its text may hold that one cut and no other, and has to carry a `[Shot 1]` and a `[Shot 2]` and no third
  (`bench/masked_job.py::held_as_two_shots`; the headers are held by presence, and whether each shot's sentences are
  true of its frames is a reader's to see on the strip);
- the build's printed lines say TWO SHOTS IN ONE WINDOW, with whose word it is.

The builder does not read the shots record; `bench/two_shot_load.py` does, when it writes the job (`shots_of`), so a
hand-written job file can name a cut the record does not have: the key is a person's word.

**What it is made from: the two shots' own ordinary loads** (`bench/two_shot_load.py job --a A.json --b B.json`).
Nothing is tracked or posed again. Each preview's track, part and class-map videos are laid end to end, lossless, and
every frame of every joined file is held equal to its source's by its decoded hash. The shot and pose tables are
joined the same way, and the two head references too, each with its own box and scale, so the reference's framing
jumps at the cut as the source does. The joined masks go into a folder shaped as a preview's job folder, so
`bench/load_chain.py` runs unchanged. A two-shot load is looked at: `no-look` is not for it, by practice and not by
code (nothing in `bench/load_chain.py` refuses it on a job that names `two_shots`).

**What the two loads must be.** The same subject, source, canvas, still, settings and reference form; two neighbouring
shots of the shots record; meeting at the cut. The tool's docstring lists what it refuses and what is not built.

**The cut and the latent steps.** The tool prints, with the planner's own function (`loop_plan.split_steps`), the step
that holds the cut, which of the capture's two flags that will be (`region_carried_across_a_cut` blocks the gate,
`region_shared_across_a_cut` does not), and the first frames that put the cut on a step's edge. Moving the first
load's first frame onto one of those is the lever; the frames before it are left as the source, in `left_out`.

**The node finds the cut itself** (`video_mask.source_cuts`, gated by `cut_gate`, `unlaid_frames`).
`bench/two_shot_load.py plan-cuts JOB.json`, after the chain's `plan`, prints the cuts in the node's own plan beside the
job's and exits 1 when they differ. A load whose node missed its cut is not rendered as it stands.

**What it costs.** One window as long as both shots together; attention grows faster than the frames. The seconds a
step are the server's own, read from its log (`GET /internal/logs/raw`), never a queue's wall time.

**Not known.** What the model draws in a step that holds frames of two shots; what it makes of a reference whose
framing jumps at the cut; whether the face is more one person across the cut than in two loads from the same still;
whether a real preview's joined tables read through the capture without fault.

## Depth: who is in front

Off by default (`bench/shot_to_load.py::defaults_carried`). A load turns it on with `bench/shot_to_load.py job --depth
[WHY]`, which finds the frames where two subject tracks or a keep-out overlap, runs `bench/measure_moge2_depth.py` on
just those frames into `<capture>/depth.npz`, and makes `bench/load_chain.py` pass `--depth` (with every preview
subject's `classes=`) to `bench/capture_masked_run.py files`.

- **What it fixes**: settles shared pixels `owner_map` leaves `OWNER_CONTESTED` by foreground depth
  (`capture_masked_run.DEPTH_WIN_M`), clips a back person's region so it is not punched into a front person, and gates
  `restore=<other>:whole` in `bench/assemble_delivery.py::restore_weight`.
- **What it cannot do**: faces side by side at the same distance (within `DEPTH_WIN_M`); hand-on-face contact; a steady
  single-frame camera FOV reading on close-ups.
- **Cost and measurements**: paid only on the load's overlap frames; figures, camera FOV ranges and commands are in
  [`2026-10-18_moge2_depth_on_crossings.md`](../../bench/results/2026-10-18_moge2_depth_on_crossings.md). The depth
  models in the store are listed in [`perception_models.md`](perception_models.md), "Depth and geometry".

## Seeing what the tracker and the masks did

Read this before writing a probe.

**What a render will be given, before it renders, and what it was given, after.** The song node writes both into
`<prefix>_windows/`. A render leaves `<name>_window_N_region.npz` beside each window's latent: the fitted mask, the
token region per latent step, the margin, the window's first frame, its trim and the composite's settings and cuts
(`video_mask.save_window_region`, read with `load_window_region`). A preview samples nothing and writes
`<name>_window_N_planned_region.npz`, made by the same calls, and `<name>_plan.json`: each window's frames, what it
writes, the latent steps a cut splits and the frames the composite will leave as the source
(`audio_freeze_song.write_plan`). A plan is replaced by the next preview and left alone by a render, so the two sit
side by side. What a plan cannot hold is what `only what changed` will keep; `bench/recomposite_window.py` lays a
saved window again from its stored latent and its region.

**One command for the masks of a run, any number of people**: `bench/capture_masked_run.py files` reads the mask
videos a preview saved and a render's own review, and writes `data/<date>_<name>/` with a `subject` on every row and
file, the rows across subjects (`frames.csv`: each pair's overlap, a run's region on another subject, the margin cells
given back) and a README of what it does not hold; `preflight` lists what is at risk before a render; `video` draws the
stacked, frame-numbered picture from those files. Its docstring is the schema; `bench/check_capture_masked_run.py` is
its check; what is still missing is in `data/CAPTURE_GAPS.md`. What the models behind the lane's signals give and do
not is [`perception_models.md`](perception_models.md) and [`state_signals.md`](state_signals.md).

**One command for the file that gets watched**: `bench/assemble_delivery.py` takes a table of renders by source frame
range and writes one file at the source's own rate with the source's own audio packets, the frames no render covers
taken from the original. It proves by decode that no frame was dropped, doubled or moved and that the audio is the
source's, and raises its own flags from the per-frame table of what each render changed. Its docstring is the account
of the table's rows, each proof and each flag; `bench/check_assemble_delivery.py` is its check; the files built with
it and every change to it by date are
[`2026-10-10_delivery_files_and_their_records.md`](../../bench/results/2026-10-10_delivery_files_and_their_records.md).
In practice the table is made by `bench/rebuild_delivery.py` ([`data/AGENTS.md`](../../data/AGENTS.md), "One ledger,
made new on every landing"). What a session needs from it here:

- Cut a render's rows on the source's cuts.
- Given capture folders (`--capture`) it settles shared pixels by whose they are; where two tracked masks claim the
  same pixel the capture's `owners.npz` says whose it is. A patch has no run in a capture, so its row says whose it is
  (`subject=<label>`).
- A row can give a class of a subject back to the source (`restore=<subject>.<Class>`, read from the capture's class
  map): a thing inside a region that should not have been redrawn is put back here and not by asking the sampler to
  keep it, which changes what the sampler draws beside it. `restore=<subject>` gives back the whole of another subject,
  and `restore=<subject>:whole` takes nothing out.
- A shot left as the source on purpose is written in the table, as a `source first-last subject=<label> why` line.
- With `--size source` the file is at the source's own size: only what a render changed is scaled up and put back.
- `--compare A.check.json B.check.json` prints two records side by side.
- Judge a render's colour on this file: it is BT.709 and says so.
- A file listed in its folder's `LOCKED.md` is never built over.
- How a region sits in its plate (detail, grain, tone and cast against the plate just outside it) is
  `bench/region_against_plate.py`; run it before tuning a look by eye.

**What every masked render already wrote beside itself** (the song node, `audio_freeze_song.py`):

- `<prefix>_NNNNN_shots.json` and `.md`: the Subject Track's shot table (`shot_table.py`, whose docstring is the
  schema). Per shot: its frames, the people found on the frame shown with a number each, who was taken and why, the
  frames with and without the subject, each loss and what was looked at after it.
- `<prefix>_NNNNN_with_mask.mp4`: the mask review ([`pieces.md`](pieces.md), piece 4).
- The song node's report, in `/history` and on the graph's text preview: the window plan, what share of each window
  regenerated, the Masked Source's warnings about a part mask it doubts, the reference costs, the seconds by stage.
- A sidecar beside every video the lane makes (`bench/sidecar.py`): command, commit, inputs with sha256, each window's
  text and graph.

**The look before a render**, nothing sampled: the review graph ([`overview.md`](overview.md), "What it is"), or a
job's preview and its strip. `MiniMaxH3SaveShotTable` saves the table and the numbered tiles.

**The inside of the tracker** (each tracker call, where it stopped, every look after a loss with its candidates):
`bench/subject_regain_looks.py`, whose docstring has the commands (`run`, `render`, `replay`); a separate process on
the card, outside the server. It takes one of the pick rules and cannot yet follow a person named by a frame or a
correction.

**A finished render against its source**: `bench/masked_render_against_source.py` (`align`, `timing`, `flicker`,
`landing`; run `align` first), `bench/measure_subject_motion.py`, `bench/measure_subject_yaw.py`,
`bench/read_render_against_original.py` (the eye figure, [`renders_by_form.md`](renders_by_form.md)).

**The captures of 2026-10-07** are under `data/` (untracked), each folder with a README giving its layout; the
scripts that made them were not kept, so the folders are a format to copy, not a tool to run.

**What does not exist**: a trace the workflow writes itself ([`docs/state/next_steps.md`](../state/next_steps.md),
"Asked for and not done"), and any id for a person that holds across shots or across nodes. The capture's files and a
delivery table's rows say whose they are; the node's own files (the shot table, the mask review) carry no subject.
