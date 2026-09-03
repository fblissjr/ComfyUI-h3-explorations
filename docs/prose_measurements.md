# Prose carries pointers, records carry numbers

The rule, why it exists here, how to write under it, and the plan for the
prose that predates it. `bench/list_prose_measurements.py` is the instrument;
`bench/results/2026-09-03_prose_measurements_baseline_v2.json` is where the
backlog stood when this was adopted. The `_baseline.json` beside it was taken
the same day under an earlier pattern set that missed scientific notation and
bare decimals; every record carries `pattern_version`, and counts across
versions are not comparable. No count is restated below, on purpose.

## The rule

**A measurement in prose is a copy of a fact whose home is somewhere the
machine can re-derive, and this repo has no invalidation for copies.** So prose
does not carry the number. It carries what was compared, the direction of the
result, and a pointer to the thing that owns the value: a script, a results
file, a constant, a command. The reader runs the pointer; the sentence never
goes stale because it never held the value.

`a 1.2x improvement on xyz` fails on every axis at once. Nobody can tell what
xyz was when it was measured, where the multiplier came from, what it was measured
against, on what hardware, at what canvas, in what cache state, or under which
commit. A pointer to the record answers all of those, because a record that
does not answer them is not a record yet (see the tiers below).

This extends the decorative-number rule in `CLAUDE.md`'s Guiding Principles.
That rule deletes numbers a reader does not act on. This one governs the
numbers a reader DOES act on: they move out of the sentence and into a home.

## What counts

Three kinds of number, and what each gets:

| kind | example of the class | what the prose carries |
|---|---|---|
| **derivable now** | a file size, a row count, the shipped default, a schedule statistic | the command or the constant's path (`workflows/h3_config.py::REF_QWEN_SHORT_EDGE`), never the value |
| **measured once** | a speedup, a relative error, a wall time, a VRAM peak | the direction in words plus a pointer to the dated record that carries the conditions |
| **normative** | a limit you are setting, an exit code, a threshold you chose | allowed, cited by the one constant that holds it, with the word for how it was arrived at: measured, inherited, or reasoned |

Not measurements, and not chased: **identifiers**. A canvas (`1344x768`), a
date, a version, a block index, a token id, a frame count that names a grid
point. A number that names a thing rather than measures one is a name.

**Quotations of withdrawn claims are not claims.** `docs/SOLATTN.md`'s
do-not-rely table and `docs/evidence.md`'s retracted list quote the numbers
they withdraw, and a reader who remembers the old sentence needs to see it
there. Leave those, in quotes, with the retraction date. They are the residual
the inventory will always show, and closure is measured against them.

## Where numbers live

Dated records. A file whose job is to carry values together with when and
under what conditions they were taken:

- `bench/results/` -- the primary home. A record carries the script that
  wrote it, the commit, the hardware (`bench/hwinfo.py`), the inputs, and the
  cache state where a render is involved (`CLAUDE.md`'s caching rule).
  Since `57b3200` a render record also carries `rendered`: the prompt-bank
  id, the prompt's hash, length, canvas and seed. That is the block a pointer
  from prose relies on, so a record without it is the copy-with-a-different-
  extension case below.
- `CHANGELOG.md`, session logs, postmortems -- past tense, per version or per
  date.
- Files the inventory skips as records: `RECORD_PATTERNS` in
  `bench/list_prose_measurements.py`, each with its reason on the line.

A generated region in a doc is also a home, because the generator owns the
values and `--check` invalidates the copy: `docs/pdd_artifacts.md`'s inventory
block and `docs/prompt_catalogue.md` are the pattern. **A table of derivable
values in a hand-written doc is the most expensive kind of copy this repo has**
-- dozens of numbers, one edit each, no check -- and the fix is to generate the
table rather than to edit the rows.

## How to write under it

Pointers must resolve. `bench/check_doc_links.py` already verifies `path:line`
and `path::symbol` citations and relative links, so a pointer written in one of
those forms is checked; a pointer to nothing is worse than a number.

Shapes that replace the number:

- *"X is N GiB"* becomes *"X's footprint is whatever `ls -l` says of the file
  `h3_config.MODELS["clip"]` names"*. The reader gets the command, and the
  sentence is true under every future artifact.
- *"A is Nx faster than B"* becomes *"A beats B on this box; the margin, the
  canvas, the frame count and the cache state are in
  `bench/results/<date>_<run>.json`"*. Direction stays, magnitude moves.
- *"The default is N"* becomes *"the default is what `define_schema` on
  `MiniMaxH3AppendRefImage` declares"*, or the constant by `path::symbol`.
- A normative value stays: *"`SCHEMA_VERSIONS` accepts these and nothing
  else"*, cited by name so there is one copy.

The check to run before committing prose that describes work is `claim-audit`.
It reads the added lines as untrusted claims and re-derives each by executing a
command; a measurement with no command behind it is what it reports.

## The wiki

`docs/wiki/index.md` is derived by `bench/build_wiki_index.py` from
`CLAUDE.md`'s routing tables and a walk of the link graph. Its generator's
docstring makes the same argument this file makes: a wiki that retyped the
blurbs would be a second copy with no invalidation. So the wiki is already the
shape the rule wants -- a router that owns no values -- and it needs no
migration of its own.

What follows from that:

- **The index inherits whatever `CLAUDE.md` carries.** It is skipped as a
  generated file, correctly, but its blurbs come from a governed one. A
  measurement in a `CLAUDE.md` row appears in the index verbatim; fixing the
  row fixes both, which is why `CLAUDE.md` is step one below.
- **The written pages beside it** (`references.md`, `stages.md`,
  `prompting.md`) are governed like any other doc and sit in the inventory.
  A frame count that names a grid point is an identifier and will show as a
  hit; leave it.
- **Records have no route yet.** The migration will create many links from
  docs into `bench/results/*.json`. `bench/check_doc_links.py` verifies they
  resolve, but the index walks markdown only, so a record is reachable through
  whichever doc cites it and nothing lists them. When enough links exist to
  make it worth a page, the fix is the same move as the index: a page
  generated from the links themselves, listing which record each doc cites,
  never written by hand. Do not build it before the links exist.

## The plan for existing prose

The inventory is the worklist. `bench/list_prose_measurements.py` prints
per-file totals; `--file <path>` prints every hit with its line. It catches
unit-bearing numbers only (multipliers, percentages, sizes, times, rates,
counts with a measured noun) and skips code spans and the record set. Bare
counts are a reader's job, not a regex's -- `claim-audit` measured that class
and says why.

### Order

By read rate, not by hit count. A stale number in a file every session opens
costs more than fifty in a deep dive.

1. `CLAUDE.md` -- small, read first, and it carries measurements of its own.
2. The "read these before you start" set the `CLAUDE.md` table names:
   `docs/roadmap.md`, `docs/evidence.md`, `docs/checks.md`,
   `docs/comfyui_vendor_gaps.md`, `docs/custom_node_gaps.md`.
3. The authorities: `docs/SOLATTN.md`, `docs/h3_references.md`,
   `docs/h3_pdd.md`, `docs/prompting.md`.
4. Table-heavy derivable docs, by generation rather than by editing:
   `docs/h3_resolutions.md`, `docs/h3_input_impacts.md`,
   `docs/h3_geometry_and_nodes.md`. The relative-cost column is a function of
   the token count and belongs to a script.
5. `docs/research/` -- deep dives, one per session when already in the file
   for another reason. Do not sweep them.

Never edit a file another agent has uncommitted changes in without asking;
`git status` first. This checkout is shared.

### Per number: find the origin, then pick the tier

For each hit, three cheap probes, in order:

    grep -rn '<value>' bench/results/ docs/ internal/log/ 2>/dev/null
    git log -S'<value>' --oneline -- <the doc>        # the commit that wrote it
    git show <that commit> --stat                     # what else landed with it

The commit that introduced the sentence usually landed beside the record or
the script that produced the value, and the session log for that date names
the run. Then:

**Tier 1, derivable now.** The value is a function of the current tree.
Replace the number with the command or constant. Run it; if the current output
differs from what the prose said, write the withdrawal in the same sentence
("this used to say N") rather than silently rewording, per `CLAUDE.md`'s
correction rule.

**Tier 2, measured with a record.** The value traces to a `bench/results/`
file or a dated section. Replace the number with direction plus pointer. Check
the record carries its conditions; a bare number in a JSON file is a copy with
a different extension. If the conditions are recoverable from the commit or
the log, add them to the record. If not, it is tier 3.

**Tier 3, measured with no findable record.** Do NOT invent a pointer. A
pointer to a plausible script that did not produce the value is worse than the
bare number, because it lends the number a provenance it never had. Two moves:

- Apply the decorative test. If the reader's next action does not depend on
  the magnitude, delete the number and keep the direction in words.
- If it is load-bearing, keep it in place and mark it: *"measured as roughly
  N on <date from git blame>; no record survives; do not rely on it"*, and add
  a row to `docs/evidence.md`'s do-not-rely list. The repo already has that
  quarantine, so use it.

**Re-measuring is not the fix for tier 3.** It makes the sentence true today
and useless tomorrow (Guiding Principles: generating a decorative number is not
a lesser fix than deleting it). Re-measure only when the magnitude changes a
decision, and then the result goes into a new dated record and the prose gets
the pointer.

**Tier 4, quotation of a withdrawn claim.** Leave it. It is the residual.

### Closure

Re-run the inventory with `--json` into a new dated file under
`bench/results/` after each pass. The two records carry the counts; no doc
restates them. When the governed set's residual is tier 4 only, a `--check`
mode with an allowlist of quoted lines becomes a gate that can be green on a
correct tree, and only then. Before that point a gate is red while the state
is correct, which `CLAUDE.md` says trains a reader to ignore red.

### After closure

- A generated records page under `docs/wiki/`, built from the doc-to-record
  links the migration creates (see The wiki above).
- `--check` on the inventory, with the quoted-withdrawal allowlist.

### Handoffs

Numbers a session found while in one file and left for whoever is next in
the file that carries them, per the deep-dives rule above.

- `docs/h3_pdd.md` and the docstring of `bench/convert_pdd_lora.py` both
  carry the partition time-curve difference that
  `docs/pdd_artifacts.md` marked unsupported on 2026-09-03 (the row is in
  `docs/evidence.md`'s do-not-rely table). Point both at that row.
- `docs/h3_pdd.md` carries the bake-residual range that the
  `bench/results/2026-08-26_pdd_conversion_*.json` records own.

### Do not

- Do not add a regex gate now. The inventory is a report; the class it cannot
  see (bare counts) is `claim-audit`'s.
- Do not touch the record set. Its numbers are the point.
- Do not rewrite a transcribed paper or a vendor document. Their numbers are
  theirs.
- Do not round, "tidy" or refresh a number in passing. Either it moves to a
  home or it is marked unsupported.
