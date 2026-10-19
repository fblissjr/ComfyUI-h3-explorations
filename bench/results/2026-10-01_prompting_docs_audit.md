# Prompting docs and code: one source of truth (2026-10-01)

What was changed is CHANGELOG 0.179.1 and `docs/state/decisions.md` (2026-10-01).
This record is what the audit found and did NOT change, with who has to decide.
Method: three read-only passes (code and comments, documents, the specificity
records), each claim re-checked against the file before it was acted on. Where
a finding was not re-checked it says so.

## Done later the same day (the owner: "Remove dead code")

Removed, with the rebuilt graphs byte-identical to the checked-in ones: `scene_prompt()`
and the image-lane block (`IMAGE_FORMATS`, `_IMAGE_SCENES`, `_MARKER_PROSE`) from
`workflows/build_workflows.py`; the retired `MiniMaxH3ReferenceFit` id from
`_REF_IMAGE_NODES` and its four call sites; `bench/convert_t2va_to_ref2va.py`; the
unreachable ref2va branch in `bench/check_prompt_docs_sync.py::page_examples_grade`
and the no-op branch in `bench/build_prompt_bank.py::shape_problems`. The first two
bullets of the next section describe what was there and are kept as the record.

Left, on purpose: the CLOSED RECORD experiment scripts (dated records came from them,
and one is imported by a live check), and `bench/prompts/hf_fl2va_robecouch.txt` and
`hf_reference_sushi_cat_gothic_hall.txt`, which nothing reads but are data from the
closed AWQ lane and not code.

## Done after the owner asked for the code gaps (0.181.0 and 0.181.1)

The song node's prompts are graded, catalogued and named through one registry
(`h3_config.PROMPT_INPUTS`) and `workflows/prompts.py::carriers`, with a completeness case
and a pin of the template grammar. The stamped-header FAIL now catches every spelling and a
malformed header, the retention-line test reads compound ids, and
`bench/check_prompt_rule_controls.py` holds the red controls and pins the closed-set copies
to the guide. The shot, speaker and clock patterns have one definition in
`bench/preflight_graph.py`, and `loop_plan.CUT_TIME` is pinned to it. The guide file names
and line numbers in code are the real ones, and the prompting-related comments that cited a
rule `CLAUDE.md` no longer holds point at `docs/method/rules_history.md`. The portable copies agree
with the manual and twelve of their rule sentences are pinned. What those items described is
kept below as the record.

Still open from the code gaps: the repo-wide `CLAUDE.md` citations in bench scripts that are
not about prompts (`grep -rn "CLAUDE.md" bench --include=*.py`; some still hold), and the published claude.ai copy, which
is still the 2026-09-01 snapshot.

## Needs the owner's decision (it deletes code or changes what ships)

- **Dead generator code.** `scene_prompt()` in `workflows/build_workflows.py`
  has no caller and holds a third copy of the Part One strings. The image-lane
  block (`IMAGE_FORMATS`, `_IMAGE_SCENES`, `_MARKER_PROSE` and their comments)
  is referenced nowhere. `_REF_IMAGE_NODES` still carries the retired
  `MiniMaxH3ReferenceFit` id in every slot and every caller discards it. The
  retire-code-with-the-knob rule says delete; nothing here was deleted.
- **`bench/convert_t2va_to_ref2va.py`.** Named by no doc or script. It parses the
  `(played by [Actor])` form, which is in no bank file, defaults `--src` and
  `--dst` to `prompt_bank`, and raises on the first bank file it reads; on
  suitable input it would write into the bank with no manifest entry and a
  trailing newline. Delete it, or make it refuse a `--dst` inside the bank.
- **The camera warnings** (`tracks right`, `whip pan`, `dolly`) stay by the
  owner's ruling of 2026-09-01, now in `docs/prompting/prompting.md` section 4.
- **The published portable page.** The claude.ai copy named in
  `docs/prompting/portable/snapshots.json` is the 2026-09-01 snapshot: it states the
  stamped shot header as the rule and its good example carries one. Nothing
  compares it to the file. Republishing is outward-facing, so it was not done.
- **The portable copies' prose.** Not edited here. In the HTML: the timestamp rule's
  chip says House (the manual says OWNER); the "produces no vocal sound" line is the
  un-narrowed form; "cut timestamps picked by vibe" is a remnant of the old rule;
  the date line. In the system prompt: the N/A habit warning is tagged `[guide]`
  (it is HOUSE), "eight turns across three shots" is a count that disagrees with
  the bank and with the manual, and a count of graded outputs sits in prose. The
  sync check pins the system prompt only on the three Part One strings and the HTML
  on Part One, the camera tokens, the examples and three quotations, so a prose rule
  in either can invert and stay green (shown by an in-memory mutation).
- **Mouth-closing, `docs/prompting/prompting.md` section 5.6.** Reconciled to the positional
  pattern of section 14.3 and the copies, where it said "every line". The section 10
  examples cue at shot ends too. If the owner prefers "every line", change 5.6 and
  the copies together.

## Real gaps in code (a code change, not prose)

- **`MiniMaxH3AudioFreezeSong` is missing from three of the lists of prompt-carrying
  nodes.** It carries the prompt of every song graph. It is in the lists of
  `bench/check_prompt_guide_conformance.py` and `bench/check_ref_prompt_labels.py`,
  and not in those of `workflows/prompts.py`, `bench/build_prompt_catalogue.py` or
  `bench/check_camera_vocabulary.py` (re-checked by grep). The consequences below are
  the audit pass's runs and were not re-run here: `preflight_graph.py` answers
  "nothing to grade" on a song graph, `prompts.describe` returns no bank id, the
  catalogue and the bank's `ships` column miss the graph (`t2va_song_flicker_lists`
  shows none), the camera check never reads it, and `run_graph_arms.py`'s
  off-length refusal cannot see it.
- **The stamped-header FAIL has holes** (`bench/preflight_graph.py`): a lowercase
  `at 00:05.200,`, a time written elsewhere in the header, and a malformed
  `[Shot 2, 00:05.200]` header all pass with no finding, and no check proves the FAIL
  can fire. `docs/prompting/prompting.md` section 11 records the holes.
- **The retention-line `(Sx)` test** uses `\(S\d+\)`, so a compound `(S1,S2)` in
  `retention_analysis` passes while a single id fails.
- **Unpinned copies of closed sets**, equal today and checked by nothing: the
  sections, retention markers and task types (in `preflight_graph.py` and
  `build_prompt_bank.py`), amplitude and speed (`build_prompt_bank.py`), the marker
  strings, the shot-header and speaker-id regexes in the catalogue and the corpus
  diff, and the `At MM:SS` pattern in `loop_plan.py`.
- Smaller: `bench/check_prompt_docs_sync.py::page_examples_grade` has an unreachable
  ref2va branch; `bench/build_prompt_bank.py::shape_problems` has a no-op branch;
  `bench/prompts/` holds two files nothing reads; the base-guide line numbers are
  off by one or two at four sites; about a dozen comments attribute a rule to
  `CLAUDE.md` that the current file does not contain; the vendor guide file names
  `base-en.txt` and `ref-en.txt` print in user-facing FAIL messages though ours are
  `base_en.md` and `ref_en.md` (the line numbers still match).

## Marked CLOSED RECORD, and not quite right

- `bench/compile_marker_corpus.py` is imported by the live `check_marker_corpus.py`.
- `bench/run_marker_arms.sh` runs a manifest that says "Do not run as is" and carries
  no marking itself.
- `bench/gen_phaseb_grid.py` can no longer build its ref2va arms (`_ref_prompt` exits
  at the bank door).
- `bench/score_shot_ablation.py` scores a closed ablation and is unmarked.

## Fixes that are not confirmed

- **The market coins.** The text names the stallholder, a fixed-text pair was
  rendered, and no scored read is recorded. The cafe fixes are the only ones with a
  before-and-after render.
- **The radio drama silence.** The two renders are one configuration rendered twice,
  bit-identical, so neither shows the fix.
- **The specificity classes have no grader.** `bench/adherence_checklists.json` has
  no reader in code and no score anywhere; the scenes that taught the classes have no
  checklist. `docs/prompting/prompting.md` section 16.4 says what exists. A mechanical grader
  for most of these classes is not possible on the text; a render-side one would need
  a person counter, which nothing here has.

## Left alone in the manual

- History notes sit inside rules (section 13 "Withdrawn 2026-09-01", 14.4, 14.5, the
  "Added 2026-09-01" lines in 5.8 and 5.9). `CLAUDE.md` says history goes in
  `docs/state/decisions.md`. Moving them is a larger edit than this pass.
- Headings cited by title ("Four layers", "Three traps", the section 8 field names)
  are unnumbered, so a pointer to them cannot be stable. Numbering them changes
  titles that code docstrings quote.
- A rule stated in several places inside the manual (timestamps, the silent phrase,
  the speech budget, `N/A`): section 11 restates what the prose sections own.

## Still red, and not from this pass

*2026-10-01, later: `check_doc_links.py` is clean again (the two `docs/h3/audio_freeze.md` citations
were corrected by another session), and "fourteen" below was a count that moves with whether a GPU
is visible to the sweep. Read the non-zero exits of `for f in bench/check_*.py` instead of this list;
what follows is what it said when written.*


- `check_doc_links.py`: two citations in `docs/h3/audio_freeze.md` to a sister pack
  path that is not on this machine. Red at `HEAD` too.
- Fourteen other `bench/check_*.py` exit non-zero from the environment: no `comfy`
  import under bare `python`, no HF cache, no server, or gitignored `internal/` files
  that `check_no_owner_paths.py` scans.

## What was verified

- `check_prompt_docs_sync`, `check_camera_vocabulary`,
  `check_prompt_guide_conformance`, `check_ref_prompt_labels`,
  `check_speaker_id_control`, `check_graph_discovery`, `check_doc_inventory`,
  `check_skill_routes`, `build_prompt_bank --check`, `build_prompt_catalogue --check`:
  green after the edits.
- The generator rebuilt to a scratch directory is byte-identical to the checked-in
  graphs, and every code edit in this pass is a comment or a docstring (AST compared
  for the generator, `run_graph_arms.py` and `prompts.py`'s code).
