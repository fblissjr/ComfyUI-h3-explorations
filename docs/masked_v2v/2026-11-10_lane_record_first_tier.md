# The lane's record in SQLite: the first tier, as built

last updated: 2026-11-10

A dated record of step 1 of [`2026-11-10_simplification_plan.md`](2026-11-10_simplification_plan.md): what was
built, what it was held to, and what the build found. The plan says why; this says what exists.

## Where things stand

- **Built:** the schema's first tier, the importer, the exporter and the proof, in
  [`bench/lane_db/`](../../bench/lane_db/), run as [`bench/lane_record.py`](../../bench/lane_record.py). Its check is
  [`bench/check_lane_db.py`](../../bench/check_lane_db.py).
- **Proved on a scratch build of `fun`:** `bench/lane_record.py prove --project fun --work SCRATCH` prints the three
  comparisons below and exits 0.
- **The mapping's open fields are ruled** (the owner, 2026-11-10; [`../wiki/decisions.md`](../wiki/decisions.md)):
  - **Typed**, because today's tools read them: the plan's `keep_outs` with `keep_outs_by`, `load_prefixes`, a laid
    piece's `owner_looks`, and `carved_whole_by`.
  - **`recipe` is the one source** of the crop and mouth forms; `plan.reference` is kept raw only.
  - **Raw only:** the rare fields of the answer lines, `text.not_used`, one laid line's recipe folder, and the
    manifest's `masks`.
- **`data/lane.sqlite` is built** from every project of `data/projects.json`, by `bench/lane_record.py build --db
  data/lane.sqlite`. It is rebuilt whole when a record file changes; nothing updates it in place yet.
- **`fun2` is in it as a project** on `fun`'s video ("`fun2` as a project", below).
- **`fun2`'s starting list is one command:** `bench/lane_record.py export --project fun2 --db data/lane.sqlite --out
  FOLDER --subjects second,third` writes the pieces `fun2` keeps from `fun`, as they stand, and no lead piece. Nothing
  has been written into `data/fun_lead2/` by it; that is the owner's go.

## What the first tier holds

Three migrations in [`bench/lane_db/migrations/`](../../bench/lane_db/migrations/): the first tier's tables; the
change that lets one file hold more than one project and video (a seed, an answer and a laid line are each a line of a
named file, and `project_keeps`); and each plan's footage facts kept as that plan's word, with the latest standing.
Each table's comment says whether it is a footage fact or a project's.

- **The project level:** `project` sits above `video`. Which projects there are, which video each works on, which
  dated records are whose and which cut closed a project are read from a new record, `data/projects.json`, written
  from the owner's decisions of 2026-11-10 ([`../wiki/decisions.md`](../wiki/decisions.md)).
- **Footage facts, shared by every project on the video:** the copies, the shots and the shots they came from, who is
  on each shot with the answers a later one replaced, the who-is-who seeds, head marks, voice spans, facial hair,
  not-people, aliases, and the owner's answers.
- **A project's facts:** the plan per subject (recipe, text, keep-outs, restores, overrides, clothing words), the
  laid list with every state a piece has been in, the removed pieces, and the final cut with its delivery table's
  rows.
- **Files:** a record file is a `location` (a named root and a relative path). What a record says a file's `sha256`
  was is kept on the row that says it, as an `artifact`. A location's own `artifact_id` is set only by a hashing
  pass, and a trigger refuses other bytes at a place that has its bytes.
- **Every imported record** is in `import_source`, verbatim, tied to the row made from it.

**The second tier is the next migration:** previews, loads' settings, steps, what each step read, and `how_known`
(recorded, inferred, unknown) on each link. The first tier makes no inferred link, so that column has nothing to hold
yet. The review items, the verdicts file's lines from a terminal, and the job and capture files are not imported.

## `fun2` as a project

A build imports every project of `data/projects.json`, a video at a time. The first project on a video brings the
footage facts. A later one brings only its own.

- **What `fun2` keeps is the owner's word, recorded once:** `keeps` in `data/projects.json` names `second` and `third`
  from `fun`. The view `project_laid` is then what a project lays: its own standing lines, and the standing pieces it
  keeps from the earlier project. No row is copied.
- **Its plan is its own:** the lead's still, text words and restore differ from `fun`'s, and the lead's strangers
  keep-out is there. The footage facts in its plan file equal `fun`'s today.
- **Its shots file must be the footage's.** It is byte for byte `fun`'s; the importer refuses one that is not.
- **Its laid file was re-seeded from the record on the owner's go (2026-11-10).** Before that it was a snapshot that
  still held `fun`'s lead pieces, which `fun2` does not keep; every `second` and `third` line of it was already a
  piece `fun2` keeps, exactly as it stands in `fun`. The file as it was is kept beside it as
  `laid.jsonl.before_reseed_2026-11-10T2034Z`. `bench/lane_record.py compare --project fun2 --db data/lane.sqlite`
  now reads equal on all three of its files, and `bench/laid.py verify` finds every render's bytes as laid.
- **Its own lead lines are the owner's keeps.** Five lead renders of 2026-11-01 were on the list with `owner: kept`
  and no cited line of the owner's. The owner kept them on 2026-11-10; their lines are in
  `data/fun_lead2/owner_answers.jsonl`, and each laid line's history shows the reset and the keep
  (`bench/laid.py set-owner`). None has been composited into a cut yet. The owner did not like `lead_fp_2823c` of
  the same day, which is not on the list; those words are taken down too, with no choice recorded.

**Footage facts a plan file carries, and which word stands.** Head marks, voice spans, facial hair, not-people,
aliases and the film's words are facts about the footage, and today each project's plan file says them.
- **Each plan's word is kept** under the project whose plan it is, so each plan file comes back from the record as it
  is.
- **The latest plan's word stands:** the `*_standing` views (`head_mark_standing` and the rest) give the word of the
  last project on the video that has a plan. A head mark written into `fun2`'s plan while it is worked supersedes
  `fun`'s, and `fun`'s rows stay as what stood before.
- **A difference is named:** where a plan says a footage fact differently from the plan before it, the importer's note
  lists the fields. `fun`'s and `fun2`'s plans agree on all of them today.
- **The cost:** the same fact said by two plans is two rows. That is what the two files say; the tools that write
  footage facts still write them into a project's plan.

## What it was held to

`prove` builds twice and prints figures, not a verdict on any picture:

1. **Two builds of the same files hold the same rows.** A build is a function of the files; nothing is updated in
   place.
2. **Each record file equals what the database gives back**, as data, with the fields the mapping keeps raw taken
   out: `shots.json`, `plan.json`, `cast/cast.jsonl`, `laid.jsonl`, `laid.removed.jsonl`.
3. **The delivery table made from the database's files equals the final cut's table on record**, section for section
   and row for row, by the unchanged `bench/delivery_table.py`. Only when and by what command the table was made is
   left out.

**What the proof cannot see:** the picture, and anything the first tier does not hold. Two things the table tool
reads are second-tier and are read from their files: the ledger on record, and each load's job file.

## What the build found

- **The capture folders moved, and the records still name the old place.** The folders `data/<day>_<load>` were moved
  under `data/fun/` on 2026-11-09. The ledger and the delivery table on record name each capture at the old place, so
  today's `bench/delivery_table.py` on today's files ties no row to its capture's run. `prove` reads the ledger with
  each capture at its new place and says so in its output. Only the delivery table was tried; any other tool that
  follows a recorded capture path was not.
  - **Answered with links, on the owner's word (2026-11-10):** the folders stay in `data/fun/`, and a link at each
    old name `data/<day>_<load>` points at its folder there. On the untouched records, today's
    `bench/delivery_table.py` then ties every row to its capture's run and gives the final cut's table again, with no
    path rewritten. `find data -maxdepth 1 -type l -lname 'fun/2026-*'` lists the links; deleting them undoes it.
  - **Still open:** `bench/load_chain.py` builds a new load's job folder at `data/<day>_<name>_job`, so a new load
    lands at the top of `data/` and not in a project's folder.
- **The lead's `keep_outs` is read by the load tool.** `data/fun/plan.json` holds `keep_outs: ["strangers"]` for the
  lead, and `bench/shot_to_load.py` reads it (`keep_outs_of`, and the strangers hold). The handoff's draft had it
  kept raw only; it is a child table here. An earlier form of this line said a plan without it would make the lead's
  loads with no strangers keep-out, which overstated it: the keep-out follows who else is in the shot for every
  subject (`strangers_by_the_data`), and the plan's key decides only where a preview saved no shot table.
  `load_prefixes` is the same case: `bench/shot_coverage.py` and
  `bench/render_ledger.py` read it.
- **`plan.reference` adds nothing `recipe` lacks on `fun`.** The load tool reads `recipe` first and `reference` as
  the fallback. The importer keeps `reference` raw and refuses a plan whose `reference` says a form its `recipe` does
  not.
- **Some answer lines hold a `frames` that is not a span** (a text and a joined number, as an early page wrote it).
  They are kept raw only; the importer's note names the lines.
- **Many answers are not classed.** The importer classes an answer as about the footage, a render or the plan by
  rules a reader can check (`importer.Import.about`, and `answers_on_loads` for an answer whose row is a load's
  name), and leaves the rest `not classed` rather than guess. Its note prints the count of each, a file at a time.
- **The laid list's history is consistent.** Every `history` entry starts from the state the one before it left, and
  each line's `owner`, `by` and `owner_word` are what its last entry left. The importer refuses a list where that
  does not hold.
- **The plan's count of the lead's shots holds:** `laid_standing` gives the same number of shots with a lead piece,
  and of shots where the lead shares frames with another subject, as the plan's table of `fun2`'s loads.

## What is next

1. The second tier, which brings each load's capture as a place. That is where the moved capture folders are
   answered in the record; for today's tools they are answered by the links at the old names.
2. The review items and the verdicts file's lines from a terminal.
3. The tools onto the database, one at a time. A tool that writes a footage fact should then write it once, for the
   video, and the two-rows cost above goes away.
