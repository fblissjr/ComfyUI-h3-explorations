# Simplifying the masked lane after `fun`: the plan

last updated: 2026-11-10

**Why** (the owner, 2026-11-10): `fun` goes into a database as data, deterministic, so that `fun2` (the same footage,
only the lead replaced) runs only against the lead and composites everyone else from `fun`'s record unchanged.
Beneath that sits the repo's own purpose (`VISION.md`): the owner's experiments with H3, shared, whose output is claims a
reader can re-derive.

Written by one session at the owner's ask once `fun` was done. Steps 1 and 2 have the owner's agreement (2026-11-10).
Each later step starts on the owner's go, and other sessions' answers to the same rethink are folded in before step 4
starts. **Step 1 is the session `mrsql`'s.** Its drafts are in `internal/scratch/lane_db/`: `schema.sql`, `mapping.tsv`,
the dry-run scanner and the coverage check, and `dry_run_fun_summary.md`. The owner's 31 `ask` rulings are owed before
the importer runs. The
pipeline it refers to is written out, with every step linked to its code, on
[`../wiki/masked_v2v.md`](../masked_v2v.md), "The pipeline, end to end".

What it rests on:
- the perception code, read on 2026-11-10;
- an evidence inventory of the lane's records (decisions, research, postmortems, `bench/results/`);
- a survey of perception models released from late 2025 on, with papers' claims kept apart from independent evidence;
- the 2026-11-10 depth test of MoGe-3 against HyDen MoGe-2 (untracked, under `internal/scratch/images/`).

## Next, in order (2026-11-10, after steps 1 and 2)

Written by `mrsql` at the owner's ask. It says where each step stands and what comes next. The detail of what was done
is in "Where things stand", below, and in [`../wiki/decisions.md`](../decisions.md) under 2026-11-10.

| step | what | where it stands |
|---|---|---|
| 1 | the lane's records in SQLite, first tier | **done.** `data/lane.sqlite` holds `fun` and `fun2` ([`2026-11-10_lane_record_first_tier.md`](2026-11-10_lane_record_first_tier.md)) |
| 2 | forgotten opt-ins on by default | **done.** Depth is the code's default; the other switches were already defaults or stay opt-in |
| 3 | `fun2`'s set-up, then its lead loads | **set-up done, loads not started.** Its laid list is re-seeded, and the owner kept five lead renders of 2026-11-01 |
| 4 | test `gem/perception-lhf` on `fun2`'s lead-alone shots | not started |
| 5 | one stage that smooths over time, then one swap at a time | not started |
| 6 | the record's later tiers | alongside; footage facts across two plans are done, and the second tier is next (below) |

**What comes next, in this order:**

0. **First, the tools find a project's folders themselves, and the links go** (the owner's yes, 2026-11-10, at the
   day's close). The owner wants the top of `data/` clean; the dated entries there are the links to `data/fun/`.
   - **The change:** a tool that follows a recorded `data/<day>_<load>` reads it at `data/<video>/<day>_<load>` where
     that is where the folder is, and a new load's folders are made in a project's folder, not at the top of `data/`.
     One resolver, called by every tool that follows a recorded path; load-step tools are built in a worktree.
   - **The proof, before a link is deleted:** with the links taken away in a scratch view, the delivery table ties
     every row of `fun`'s final table to its capture's run, and a rebuild runs end to end.
   - **Then:** the links are deleted (`find data -maxdepth 1 -type l -lname 'fun/2026-*'`).
   - **Which folder a new `fun2` load goes in** is still the owner's to say ("Waiting on the owner").
1. **A `fun2` cut from what already exists.** Proposed by `mrsql`; it waits on the owner's go.
   - **What it is:** one `bench/rebuild_delivery.py --clip` for `fun2`, from the kept `second` and `third` pieces and
     the owner's five kept lead renders. No new render.
   - **What it tests:** the whole premise of `fun2`, that only the lead changes and the rest is composited as it
     stands, which has never been run end to end. It is also the first full rebuild since the links went in at the
     capture folders' old names; so far only the delivery table has been tried.
   - **What the owner gets:** the five lead pieces composited for the first time. On 1419-1465 the new lead lies on
     top of `third_fp_1419c`, so the frames where the two regions meet are the place to look.
2. **Two rules settled before a new lead load is made.** Both are the owner's calls ("Waiting on the owner").
   - **The context rule of 2026-10-25,** which the code does not enforce. The built answer goes into the job step
     the way step 2's did: in a worktree, landed when nothing is rendering.
   - **Where a new load's folders go.** Today's tools make them at the top of `data/`.
3. **Step 3's loads:** the lead-alone shots first, each load's masks compared with the masks `fun`'s kept lead piece
   on the same frames was made from. The shots that already have a kept lead piece owe no load
   (`data/fun_lead2/AGENTS.md` lists them).
4. **Steps 4 and 5,** as written below, each on the owner's go.

**Alongside, the record (step 6), in `mrsql`'s order:**

1. **Done (2026-11-10): footage facts said in a later project's plan.** Each plan's word is kept, the latest plan's
   stands (`head_mark_standing` and the other `*_standing` views), and a difference between two plans is named in the
   importer's note. A head mark written into `fun2`'s plan now supersedes `fun`'s.
2. **The second tier:** previews, loads' settings, captures and renders as rows, with what each step read and how
   each link is known. This is where the record holds each capture at the place it is now.
3. **The review items, and the verdicts file's lines from a terminal.**
4. **The tools onto the database, one at a time,** starting with the ones that join records.

**Waiting on the owner:**

- **A go for the `fun2` cut** (item 1 above).
- **The context rule: extend, or refuse.** `mrsql`'s recommendation is to extend: the job step runs a load cut inside
  a longer shot to its window's length, masked end to end, and the extra frames are chopped at the lay, as the ruling
  of 2026-10-25 already says. It would apply to new loads only; the pieces `fun` kept stay. Whether the subject is
  visible in the padding frames of `fun`'s loads has not been looked at.
- **Where `fun2`'s new loads go.** Two readings stand, and the owner has not said which:
  - **`data/fun/`:** the owner's words on 2026-11-10 were that everything should be in `fun`, said of the capture
    folders, and `fun2`'s loads of 2026-11-01 are already there.
  - **`data/fun_lead2/`:** each project's loads in its own folder, as "Where things stand" reads it.
- **Which of 2026-11-01's other lead renders stand,** if any. Five are kept and `lead_fp_2823c` is not liked; the
  rest of that day's folder under `Video/fun/gemlead2/` is listed, not laid.

**Not assigned:** steps 3 to 5 have no session's name on them. The session that wrote this plan has ended; `mrsql`
holds the record (steps 1 and 6) and landed step 2.

## Where things stand (2026-11-10)

- **Step 1's first tier is built** by `mrsql` (`b561376c`): `bench/lane_record.py`, with its check
  `bench/check_lane_db.py` and its record [`2026-11-10_lane_record_first_tier.md`](2026-11-10_lane_record_first_tier.md).
  - **The record:** the owner ruled the open fields (`b7bf86d7`). `data/lane.sqlite` is built from both projects,
    with `fun2` a project on `fun`'s video (`a5bd98e2`).
  - **`fun2`'s list:** re-seeded from the record on the owner's go (`25219d48`).
- **The moved captures are answered with links** (the owner's choice, 2026-11-10; `mrsql`, `5c5da3cb`).
  - **The move:** `fun`'s capture and job folders moved under `data/fun/` on 2026-11-09, while 614 job files and the
    final cut's 88 table rows still name them at `data/<day>_<load>`.
  - **The fix:** each old name is now a link to `fun/<day>_<load>`, 1622 in all
    (`find data -maxdepth 1 -type l -lname 'fun/2026-*'`), none dangling. Everything stays in `data/fun/`, and no
    record or code changed.
  - **What `mrsql` checked:** `bench/delivery_table.py` ties every row of the final table to its run again.
  - **Not yet checked:** `bench/rebuild_delivery.py` end to end, and the load tools.
- **Still open, the owner's call: where `fun2`'s new loads go.** `bench/load_chain.py` makes a new load's folders at
  `data/<day>_<name>_job`, at the top of `data/`. The owner wants everything kept in its project folder, so `fun2`'s
  loads would belong under `data/fun_lead2/`. That is a change to a load-step tool, built in a worktree.
- **Corrected:** `bench/check_doc_links.py`'s broken pointers are not from the move. The links brought them only from
  4276 to 4269. Of the rest, 4162 are its no-bare-filenames rule firing inside generated records under `data/fun/`
  (ledgers, tables), and 102 are malformed links in those same files. This bullet used to say "most likely from the
  same move".
- **Step 2 is landed** (`mrsql`, on the owner's go; `8d61067f`, docs `c64f47b5`).
  - **Depth:** the code's default in `bench/shot_to_load.py`. Turning it off takes a person's words (`--no-depth WHY`,
    or the plan's `depth: {"on": false, "why": ...}`).
  - **Already defaults:** the head track (`h3_config.SUBJECT_TRACK["track_head"]`) and memory selection.
  - **Stays opt-in:** the head tube, since no word of the owner's makes it a default.
- **Corrected: the strangers keep-out is set by the data, not the plan.** `bench/shot_to_load.py::strangers_by_the_data`
  sets it for every subject from whether anyone else is in the shot. The plan's `keep_outs` decides only where a
  preview saved no shot table. This bullet used to say the lead's strangers keep-out was the plan's word, and a
  candidate for the step 2 defaults; it already is a default, by the data.
- **Open, and the owner's call: the 2026-10-25 context rule is not enforced in code.** The rule says the original
  subject is never shown unmasked in a window, and that a load cut inside a longer shot runs masked to its window's
  length.
  - **What the code does:** a load shorter than its window is padded by `workflows/masked_job_graph.py::window_pad`.
    Padding frames still inside the shot stay real (`held_context`), and under the default `context_shows: as filmed`
    nothing is regenerated on them. No check refuses that. `dropped_or_unmasked_face_in_context` covers only frames
    dropped inside the load.
  - **How often it happened in `fun`:** 19 loads were padded with real same-shot frames, from 4 to 45 each.
    - **How it was counted:** `window_pad` and `held_context` run over every load of the job files under `data/fun/`.
    - **Laid in `fun`'s final cut** (all kept by the owner on playback): `lead_fp_0516f`, `0559`, `0579`, `3098b`,
      `4227e`, `4310c`, `4774c`, `6125c`.
    - **Not checked:** whether the subject is visible in those padding frames.
  - **Why it matters for `fun2`:** its lead loads on those shots meet the same case.
  - **The choice:** a step 2 default where the job step extends such a load to its window, masked end to end, or
    refuses it.

- **`fun` is closed.** It ended on 2026-10-31 at about 17:30 to 18:00 Central (the owner).
  - **Its final cut:** `output/Video/fun/day_2026-10-31/fun_0000_6394_joined_2026-10-31T2231Z_from_laid.mp4`.
  - **Its final state:** the cut's record names the laid list `70ead45b...`, which is `data/fun/laid.jsonl` as it
    stands.
  - **Its copies:** the source, `input/fun.mp4`, is back. The 24 fps working copy was rebuilt by its recorded recipe,
    byte-identical to its record (`data/fun/copies_rebuilt_2026-11-10/README.md`).
- **`fun2` is next**, in `data/fun_lead2/`.
  - **The change:** the same footage, with only the lead replaced: a new reference, `img_lead_2.jpg`, face only as
    before.
  - **What stays:** `second` and `third` composite exactly as in `fun`'s final laid list.
  - **Its old start:** everything dated 2026-11-01 Central or later (from 05:00Z in the records' stamps) is `fun2`'s,
    including work run under the old conditions inside `fun`'s folders.

## The order, and why

1. **Done: the lane's records in SQLite, first tier.** Import what a person said and decided, plus `fun`'s closing state.
   The test: the database reproduces the delivery table of `fun`'s final cut, which names laid list `70ead45b...`. It
   changes no picture.
2. **Done: turn the forgotten opt-ins on by default in the code.** Small, with no open question, and in before
   `fun2`'s first load.
3. **Set up `fun2` and start its lead loads,** lead-alone shots first (below). Every lead load's masks are compared with
   the masks `fun`'s kept lead piece on the same frames was made from: same footage and same face-only region, so they
   should match. That comparison is the baseline steps 4 and 5 are measured against.
4. **Test the `gem/perception-lhf` branch** on `fun2`'s lead-alone shots against that baseline, before it makes any
   `fun2` piece. Most of the structural simplification is already built there.
5. **One stage that smooths over time, then one model swap at a time,** tested the same way. The face/hair split
   matters here: the lead is face only, so Sapiens2's face class is what draws the lead's region.
6. **The record's later tiers run alongside:** every file as an artifact (hashed in the background), the recorded
   links, then the inferred links, labelled. The tools move onto the database one at a time.

**`fun2` does not wait for every tool to move.** It runs on today's tools, which work. The importer re-runs as `fun2`
writes, until each tool reads and writes the database itself. Waiting for the whole move would hold `fun2` for weeks
for no picture gained.

## `fun2`: what the lead's loads are

From `fun`'s final laid list, the 47 shots with a laid lead piece fall into three cases:

| case | shots | how `fun2`'s lead is made |
|---|---|---|
| the lead alone on the shot | 39 | a fresh load on the working copy, as in `fun` |
| the lead laid last | 2: 604-1009, 4647-4725 | on the cut that holds `second` and `third`, as in `fun` |
| the lead laid before another subject | 6: 1010-1038, 1093-1133, 1222-1305, 1306-1418, 1419-1465, 1466-1729 | the lead becomes the **last** pass, made on `fun`'s cut that holds the kept `second` and `third`, with them kept out by their tracks |

- **The rule this departs from:** the third case departs from the standing remake rule, that remaking an earlier pass
  remakes every pass after it ([`AGENTS.md`](AGENTS.md), "The owner's rules for passes, prep and judging"). The
  owner's word for `fun2` (2026-11-10: only the lead changes, and `second` and `third` composite as before) overrides
  it there.
- **Where it is written:** that goes in `data/fun_lead2/AGENTS.md` before the first load, so no session applies the
  old rule.
- **Where to look:** frames where the new lead and a kept piece change the same pixels. The assembler flags them
  (`two_pieces_change_the_same_pixels`).

**Before the first `fun2` load** (all done on 2026-11-10):
- **The laid list:** `data/fun_lead2/laid.jsonl` was re-seeded from the lane's database on the owner's go (`mrsql`,
  `25219d48`). It holds `fun`'s final `second` and `third` pieces as they stand, and `fun2`'s own lead lines.
- **The project page:** `data/fun_lead2/AGENTS.md` and its one-line `CLAUDE.md` are written.
- **The 2026-11-01 work:**
  - **Kept:** five lead renders, kept by the owner (`data/fun_lead2/owner_answers.jsonl` lines 1 to 5):
    `lead_fp_0294f`, `lead_fp_1156h`, `lead_fp_1419e`, `lead_fp_3909d`, `lead_fp_4053e`. Those five shots (294-414,
    1156-1188, 1419-1465, 3909-3966, 4053-4126) owe no fresh load. On 4053 the piece covers 4053-4091, as `fun`'s did.
  - **Not liked:** `lead_fp_2823c` (line 6).
  - **To watch on 1419-1465:** the kept `lead_fp_1419e` was rendered on the `2026-10-31T2110Z` cut. That cut held
    `fun`'s kept `second` but not its kept `third_fp_1419c`, laid later. Look where the lead meets the third in
    `fun2`'s cut.

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

**Where it lives.** One file for the lane, `data/lane.sqlite`, on the local disk where `data/` already is. It is never
in git (`.gitignore`). Not on the media share: SQLite's locking is not safe on a network filesystem, and the share is
CIFS. One file rather than one a video, because projects share footage (below).

**Projects share footage: `fun2` is `fun`'s video with another lead.** The record has a project level above the video.
- **Facts about the footage hang off the video and carry to every project on it:**
  - the shots and their cuts;
  - who is on each shot, and the who-is-who seeds;
  - where a head is hidden, and whose hair is whose;
  - things detectors take for people;
  - who sings where;
  - the video's copies.
- **Facts about a replacement hang off the project:**
  - a subject's still, text and recipe;
  - the loads and renders;
  - what is laid;
  - the owner's verdicts on renders.
- **An answer is classed by what it is about:** the footage (a video fact) or a render (a project fact).

That is the owner's rule of 2026-10-21, that prep is not reused and only the owner's words carry, written into the
record. A new project on the same footage starts from every footage answer already given, and from nothing derived.

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

### The data model, first draft (agreed in shape with the owner, 2026-11-10)

```
project ── video          (fun and fun2 are two projects on one video)

the video's footage facts, shared by its projects:
video ─┬─ copy (source, working copies)
       ├─ shot ── appearance, seed, head marks, voice spans, not-people

each project's own records:
project ─┬─ subject_recipe, subject_text (the lead's still and words differ per project)
         ├─ preview ── load ── step ── artifact
         ├─ laid ── cut
         └─ review_item ── answer about a render

the first draft's detail, unchanged otherwise:
video ─┬─ copy (working copies, cuts)
       ├─ subject
       ├─ shot ── appearance (who is on it, per a person's word)
       ├─ answer (everything the owner said, on any page)
       │    ├── seed (who-is-who boxes)
       │    └── subject_recipe / head_mark (the plan, per subject, with history)
       ├─ preview ── load ── step ──┬── artifact ── location
       │                            └── step_input (what a step read)
       │                     ├── flag, look
       └─ laid (a kept piece on a shot, superseded, never edited)
            └── cut ── cut_piece
                 └── review_item ── answer
```

**Pointers to the data: content for identity, a named root for location.**

```sql
artifact(id, sha256 NOT NULL, bytes, media, kind, subject_id,
         first_source_frame, frames,                 -- frame k of the file is source frame first+k
         made_by_step_id NOT NULL, written_at)
location(artifact_id, root, relpath, size, mtime, verified_at, UNIQUE(root, relpath))
root(name PRIMARY KEY)                                -- 'share_output' | 'share_input' | 'data' | ...
step_input(step_id, artifact_id, member NULL)         -- member: one key of an .npz, when that is all it read
```

- **Roots are names**, resolved at run time from what the tools already use (`H3_OUTPUT_DIR`, `H3_INPUT_DIR`, the
  repo's `data/`). No absolute path is stored.
- **The `sha256` is the identity.** One artifact can have several locations, and a moved file is a new `location`, not
  a new artifact.
- **Write once, enforced.** Registering a different `sha256` at an existing `(root, relpath)` is refused. That is the
  fault the records found, where a capture written several times in place left its `.npz` and `.mkv` disagreeing
  ([`code_map_contracts.md`](code_map_contracts.md), Section 2c.2).
- **Verification is cheap.** It re-hashes only files whose size or mtime changed since `verified_at`, and reports a
  missing file rather than dropping it.
- **The frame offset is stated once,** on the artifact. Today each tool works out a file's frame 0 itself.

**No JSON stuffed into fields.**
1. **Typed columns and child tables.** What anything filters, joins or checks on is a typed column. A list is a child
   table: an answer's frames are `answer_frame(answer_id, frame)` rows, and a laid piece's frames are range rows.
2. **The plan gets typed tables, one per concept**, for example:

   ```sql
   subject_recipe(subject_id, parts, margin_px INTEGER, zoom_up_to REAL,
                  crop TEXT CHECK (crop IN ('head','face','cranium')), depth_on INTEGER,
                  valid_from, valid_to, said_by, answer_id)
   ```

   A new knob is a schema migration. That cost is wanted: every new knob has to say what it covers and what retires
   it, instead of slipping into a blob and being forgotten.
3. **SQLite enforces it:**
   - `STRICT` tables;
   - `PRAGMA foreign_keys=ON` on every connection, since SQLite leaves foreign keys off by default;
   - `CHECK` constraints for enumerations;
   - one numbered migration per schema change (`PRAGMA user_version`).
4. **JSON in exactly two places, read by nothing:**
   - `import_source(file, line, raw)`, so every row traces back to the line it came from;
   - a step's `argv`, the command as it ran.

   A check fails on any `json_extract` in a tool, and on any third JSON column.
5. **The importer refuses unknown keys.** A mapping file, readable by a person, sends every key of today's files to
   a column, a child table, or "kept raw only". An unmapped key stops the import. `owner_answers.jsonl` alone carries
   about 40 optional keys today. The mapping is therefore also the list of every field the lane has grown, with a
   decision on each, and a field no tool reads is named rather than carried.
6. **Per-frame figures stay in their files**, behind an artifact row, until the analysis layer below reads them.

### Later: an analysis layer in DuckDB, built from the record

The record is normalized because it takes small concurrent writes, history by supersession, and a provenance graph.
Analysis wants a star instead, and the lane already makes the facts: the per-frame tables and figures of every capture
(`find data/fun -name '*.csv' | wc -l`).

**The layer:** a DuckDB file, built from the SQLite record plus those files, never written by a tool, and rebuildable at
any time.
- **Facts**, one row each:
  - a load and source frame;
  - a render;
  - an owner verdict.
- **Dimensions:**
  - video, shot, subject;
  - load and recipe, as type-2 rows with valid-from and valid-to;
  - code version, perception model, date, who.

It is built once there is a question worth it. The first candidate: how many renders a shot took before a keep, by
recipe, which says which recipe defaults to change.

**How to get there without breaking `fun`.**
1. **Schema and importer:** write the schema and an importer for `fun`'s existing files.
2. **Prove it on `fun`:** what the database says must match what the current tools say, for the delivery table, the
   laid pieces and the owner's keeps.
3. **Move the tools one at a time to read the database,** starting with the tools that join records:
   `bench/laid.py`, `bench/delivery_table.py`, `bench/rebuild_delivery.py`, `bench/review_pages.py`,
   `bench/said_on_shot.py`. While a tool moves, it keeps writing the old file too, so nothing that still reads the file
   breaks.
4. **`fun2` runs on today's tools while they move.** The importer re-runs as it writes, and a project after `fun2`
   starts on the database alone. `fun2` reads `fun`'s footage facts from the record, and `fun`'s kept lead pieces are
   the comparison for every perception change: same frames, same lead region, only the reference differs.

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
  ([`../wiki/decisions/2026-10-14.md`](../decisions/2026-10-14.md)).

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
  ([`../wiki/decisions/2026-10-18_to_2026-10-20.md`](../decisions/2026-10-18_to_2026-10-20.md)).
