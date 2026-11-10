# Simplifying the masked lane after `fun`: the plan

last updated: 2026-11-10

A proposal from one session, written at the owner's ask once `fun` was done. **Nothing here is decided.** Each step
starts on the owner's go, and other sessions' answers to the same rethink are folded in before step 4 starts. The
pipeline it refers to is written out, with every step linked to its code, on
[`../wiki/masked_v2v.md`](../wiki/masked_v2v.md), "The pipeline, end to end".

What it rests on:
- the perception code, read on 2026-11-10;
- an evidence inventory of the lane's records (decisions, research, postmortems, `bench/results/`);
- a survey of perception models released from late 2025 on, with papers' claims kept apart from independent evidence;
- the 2026-11-10 depth test of MoGe-3 against HyDen MoGe-2 (untracked, under `internal/scratch/images/`).

## The order, and why

1. **Move the lane's records into one SQLite file.** It changes no picture, and every later step writes into it.
2. **Turn the forgotten opt-ins on by default in the code.** Small, with no open question.
3. **Test the `gem/perception-lhf` branch against kept pieces.** Most of the structural simplification is already
   built there.
4. **One stage that smooths over time, then one model swap at a time.** This needs step 3's structure to sit on.

## 1. The lane's records in SQLite

**The problem.**
- **Scattered state:** the lane's state is thousands of JSON and JSONL files. Under `data/fun` alone,
  `find data/fun -name '*.json' | wc -l` and `find data/fun -name '*.jsonl' | wc -l` count them.
- **Relationships rebuilt by every tool:** what made what is rebuilt by each tool that needs it. 23 tools under
  `bench/` read `laid.jsonl` (`grep -rln laid.jsonl --include=*.py bench`), and each joins it to the ledger, the
  answers and the captures its own way.
- **History by copies:** history is kept by copying files aside (`ls data/fun/laid.jsonl.before_*`).
- **Staleness from timestamps:** whether an input is stale is worked out from file times against git commits
  (`bench/inspect_mask_steps.py`, check `[0]`), because no record says which code made which file from which input.

**Why SQLite and not DuckDB.**
- **Writers:** the lane has many small writers at once. Several sessions, and the review server taking the owner's
  clicks (`data/<video>/poc/serve.py`, `do_POST`), each append or replace a few rows at a time.
  - SQLite in WAL mode serves many readers and one writer at a time, with transactions, and its writers queue.
  - DuckDB lets only one process open a database for writing. The review server holding it would lock every session
    out.
- **Kind of work:** DuckDB is built for analytic scans over large tables, which this lane rarely needs. When it does
  (a scan over every capture's figures), DuckDB can read the SQLite file through its SQLite extension, or a Parquet
  export.
- **Dependencies:** SQLite is in Python's standard library; DuckDB is not installed in the ComfyUI venv.

**Where it lives.** `data/<video>/lane.sqlite`, on the local disk where `data/` already is. Not on the media share:
SQLite's locking is not safe on a network filesystem, and the share is CIFS.

**What goes in, and what stays a file.**
- **In the database:** every record with relationships.
  - The video and its shots; who is on each shot, and the owner's answers about it (today `shots.json`,
    `cast/cast.jsonl`, `owner_answers.jsonl`).
  - The plan per subject (`plan.json`).
  - Previews, jobs, loads, captures and renders, each with its inputs, its outputs, and the code stamp that made it.
  - Laid pieces (`laid.jsonl`).
  - Review items (`review/items.json`).
- **Files stay files:** arrays and media (`.npz`, `.mkv`, `.mp4`, sheets). The database holds a row for each: its
  path, its `sha256`, the code stamp that wrote it, and the row it was made from.
- **The key relationship:**
  - **The chain:** a derived file points at its inputs and its producer.
  - **What it replaces:** "is this prep stale" becomes a query, in place of mtime comparisons and the rule, kept by
    hand, that a changed dependency invalidates everything made from it.
  - **What it adds:** "what was made from this preview" becomes a query too.

**History.** Rows are never edited in place where history matters (laid pieces, answers, the plan). A new row
supersedes the old one, with who and why, so "what stands" is a view and "what stood on 2026-10-21" is a query. That
replaces the `.before_*` copies and the read-only answer snapshots.

**How to get there without breaking `fun`.**
1. **Schema and importer:** write the schema and an importer for `fun`'s existing files.
2. **Prove it on `fun`:** what the database says must match what the current tools say, for the delivery table, the
   laid pieces and the owner's keeps.
3. **Move the tools one at a time to read the database,** starting with the tools that join records:
   `bench/laid.py`, `bench/delivery_table.py`, `bench/rebuild_delivery.py`, `bench/review_pages.py`,
   `bench/said_on_shot.py`. While a tool moves, it keeps writing the old file too, so nothing that still reads the file
   breaks.
4. **The next video starts on the database alone.**

**A readable dump.** `plan.json` and `shots.json` are read by people as well as tools. A dump to text (per video, per
table) keeps that, and makes a change reviewable as a diff.

**What it retires:**
- the `.before_*` copies and the read-only answer snapshots;
- the mtime-against-commit staleness checks;
- most of the per-tool joining code;
- `review/items.json` as one large file rewritten whole (`wc -c data/fun/review/items.json`).

## 2. Forgotten opt-ins become defaults

Depth was opt-in, and was forgotten for days after 2026-10-20
([`2026-10-25_pass_contamination_and_mask_step_forensics.md`](2026-10-25_pass_contamination_and_mask_step_forensics.md)).
The head track was opt-in before it.
- **Where depth stands now:** on for `fun` only because `data/fun/plan.json` sets it per subject. The code's default
  is still off (`bench/shot_to_load.py::make_depth`).
- **The owner's rule:** of 2026-10-14, lane tools are chain defaults
  ([`../wiki/decisions/2026-10-14.md`](../wiki/decisions/2026-10-14.md)).

**The change.** Depth and the head track are on in the code, and off only with a person's words. Sweep the job and
preview options for any other tool that should be a default. `bench/shot_to_load.py` is a load-step tool, so the
change is built in a worktree and landed when nothing is rendering.

## 3. Test `gem/perception-lhf` against kept pieces

**What the branch holds** (tip `16b0805c`, unmerged;
[`2026-10-27_perception_lhf_branch.md`](2026-10-27_perception_lhf_branch.md)):
- per-stage files written once;
- one Sapiens class table in place of six copies;
- a depth gate on hair;
- head-corridor crops.

It has never been tested on real footage.

**The problem it addresses.** Parts, pose, strangers, the head tube and the depth cores all start from the person
track. The track is grown, held, carved and cut again at each stage. The duplication is counted in
[`code_map_contracts.md`](code_map_contracts.md), Part 4.

**The test.** Run the branch on two or three shots the owner kept: a crossing, a face-only shot, a hair shot. Compare
its masks with the masks those kept pieces were made from, and render only where they differ. The owner's keeps are
the ground truth now that `fun` is done.
- **If it holds,** merge it.
- **If it does not,** the test names the stage that breaks, before anything new is written.

## 4. One stage that smooths over time, then one swap at a time

**The root cause.** SAM 3.1 tracks in video; Sapiens2 parts, Sapiens2 pose, SAM 3D Body and depth each see one frame
alone. Downstream, each tool grew its own hold or filter for the flicker:
- holds of missed or doubtful frames;
- dip bridging (`bench/head_tube_parts.py::bridge_short_tube_dips`);
- a 3-frame filter on class probabilities;
- the floor in `grade_holds`;
- the strangers' `hold_frames`;
- the hair chain the records call bolted on.

Several of those fixes caused the next fault.

**The change.** One stage takes every per-frame output and decides over time once, in place of a hold in each tool.
Each swap below is then tried alone, on the same kept shots, cheapest and best-evidenced first:
1. **Depth: MoGe-3 ViT-G through ComfyUI core's loader** (`comfy/ldm/moge`). On the 2026-11-10 test:
   - it called opposite to HyDen on one frame, where HyDen was wrong;
   - its field of view held far steadier within a shot;
   - it calls "too close" more often.

   It runs inside Comfy's model memory, under MIT.
2. **Face and hair as their own SAM 3.1 objects,** seeded from the head track, against Sapiens2's face/hair split.
   Per-frame SAM phrase prompts were tried early and dropped as slow and gappy
   (`bench/results/2026-10-04_masked_v2v_band.md`); tracking from a mask, as the person and head tracks do, has not
   been tried.
   - **If it holds,** Sapiens2 is needed only for the drawn face's points.
   - **With it,** the strangers keep-out can come from SAM 3.1's people alone.
3. **The body mesh smoothed over time:** core's SAM 3D Body smoothing node (`comfy_extras/nodes_sam3d_body.py`), or
   SAM-Body4D (MIT, which wraps the same two models), against the per-frame mesh.

## What held up, and stays

- **SAM 3.1,** tracked from a mask and never a text prompt, with its three runtime corrections and memory selection
  (`sam31_corrections.py`).
- **Who is who as a person's seeds,** never a likeness score (the owner, 2026-10-10).
- **Depth for who owns shared pixels.**
- **The body-model video** for head motion, with the face drawn from pose points.

## What not to do

- Rewrite `bench/capture_masked_run.py` from scratch beside the branch that already splits it.
- Swap several models in one change.
- Judge a step on stills: the owner's eye on playback accepts a piece.
- Reopen video matting: closed by the owner on 2026-10-18
  ([`../wiki/decisions/2026-10-18_to_2026-10-20.md`](../wiki/decisions/2026-10-18_to_2026-10-20.md)).
