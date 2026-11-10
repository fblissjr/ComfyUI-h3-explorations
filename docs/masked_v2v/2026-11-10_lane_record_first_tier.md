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
- **Not built yet:** `data/lane.sqlite` itself. `build` refuses the lane's own file while a line of
  [`bench/lane_db/mapping.tsv`](../../bench/lane_db/mapping.tsv) is `ask` (`bench/lane_record.py asks` lists them).
  The owner has not ruled on them.
- **`fun2`'s starting list is one command:** `bench/lane_record.py export --project fun --db DB --out FOLDER
  --subjects second,third` writes `fun`'s standing laid pieces of those subjects and no lead piece. Nothing has been
  written into `data/fun_lead2/` by it.

## What the first tier holds

One migration, [`bench/lane_db/migrations/0001_first_tier.sql`](../../bench/lane_db/migrations/0001_first_tier.sql).
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
- **The lead's `keep_outs` is live.** `data/fun/plan.json` holds `keep_outs: ["strangers"]` for the lead, and
  `bench/shot_to_load.py` reads it (`keep_outs_of`, and the strangers hold). The handoff's draft had it kept raw
  only; a plan given back without it would make the lead's loads with no strangers keep-out. It is a child table
  here, pending the owner's ruling. `load_prefixes` is the same case: `bench/shot_coverage.py` and
  `bench/render_ledger.py` read it.
- **`plan.reference` adds nothing `recipe` lacks on `fun`.** The load tool reads `recipe` first and `reference` as
  the fallback. The importer keeps `reference` raw and refuses a plan whose `reference` says a form its `recipe` does
  not.
- **Some answer lines hold a `frames` that is not a span** (a text and a joined number, as an early page wrote it).
  They are kept raw only; the importer's note names the lines.
- **Many answers are not classed.** The importer classes an answer as about the footage, a render or the plan by
  rules a reader can check (`importer.Import.about`), and leaves the rest `not classed` rather than guess. Its note
  prints the count of each.
- **The laid list's history is consistent.** Every `history` entry starts from the state the one before it left, and
  each line's `owner`, `by` and `owner_word` are what its last entry left. The importer refuses a list where that
  does not hold.
- **The plan's count of the lead's shots holds:** `laid_standing` gives the same number of shots with a lead piece,
  and of shots where the lead shares frames with another subject, as the plan's table of `fun2`'s loads.

## What is next

1. The owner's rulings on the `ask` lines, then `build --db data/lane.sqlite`.
2. `fun2` as a project: its own plan imported, its footage facts checked against `fun`'s rather than imported twice.
3. The second tier.
