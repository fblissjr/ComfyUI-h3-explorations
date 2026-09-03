# Prompting H3: where the rules come from, and what checks them

last updated: 2026-09-03

A map, not a rulebook. It says where each prompting rule comes from, where
prompt text lives, and which check catches a mistake. When it disagrees with a
file it points at, that file is right. Agents: `.claude/skills/h3-prompt`
routes by task to the same owners and the command that verifies the result.

## Sources, ranked

Only the first binds.

| rank | source | standing |
|---|---|---|
| 1 | the vendor's guides, `vendor_guides/base_en.md` and `ref_en.md` | the only authority; break one and the prompt is off the training distribution |
| 2 | the vendor's API payloads, `coderef/MiniMax-H3/scripts/readme/*.sh` | evidence of practice, not rules |
| 3 | the vendor's prompt-writing skill | a router that states no rule; its guide files are byte-identical to ours |
| 4 | [`docs/prompting.md`](../prompting.md) | our reading, every rule labelled by how much it binds |

`internal/PROMPTING.md` was deleted on 2026-09-01. Sources 1 to 3 are gitignored
and do not ship.

## Who owns which question

| question | go to |
|---|---|
| How do I write a prompt? | [`docs/prompting.md`](../prompting.md), the single source of truth for the rules |
| Where does prompt text live? Where do I start from? | `prompt_bank/`, one file per prompt. Every prompt a graph renders is there, plus the house-authored ones. [`docs/prompt_bank.md`](../prompt_bank.md) is its generated table: `ships` says which graphs render an entry, `adapt` which can take another frame count without a rewrite, `tests` what a scene is a test of |
| What does the repo render? | [`docs/prompt_catalogue.md`](../prompt_catalogue.md), generated from the graphs, with a `bank id` column |
| Is it any good? | [`docs/prompt_audit.md`](../prompt_audit.md), hand-written, one verdict per scene |
| Reference labels and image sizing | [`docs/h3_references.md`](../h3_references.md) |
| Legal canvases and lengths | [`docs/h3_resolutions.md`](../h3_resolutions.md), [`docs/h3_geometry_and_nodes.md`](../h3_geometry_and_nodes.md) |
| Comparing two prompts fairly | [`docs/eval_comparison.md`](../eval_comparison.md) |
| Handing the rules to someone outside the repo | [`docs/portable/h3_prompt_standard.html`](../portable/h3_prompt_standard.html), published at [claude.ai](https://claude.ai/code/artifact/daa73be5-0bb7-4d02-8456-5dc107acea54); a writer-model version at [`docs/portable/h3_system_prompt.md`](../portable/h3_system_prompt.md). Both are copies, so their quotations are checked, not trusted. Dated snapshots beside them are frozen and pinned in `docs/portable/snapshots.json` |

## What keeps each link true

| link | check |
|---|---|
| guides → the graders' templates and camera vocabulary | parsed from the guide files at load; a missing guide is a loud error |
| `prompt_bank/` → `build_workflows.py` | the generator loads every shipped prompt by id via `workflows/prompts.py` and refuses a missing entry; the two keyframe defaults are re-timed at the graph's length and checked at the declared one |
| `prompt_bank/` → `prompt_bank.md` | `build_prompt_bank.py --check` regrades every entry; a `recorded_findings` entry is reported, not gated |
| generator → `workflows/*.json` | generated; nothing is true of a graph until rebuilt |
| graphs → `prompt_catalogue.md` | `build_prompt_catalogue.py --check` |
| catalogue → `prompt_audit.md` | `check_prompt_docs_sync.py` fails on a scene with no verdict, and checks every quotation in the portable files against its source |
| shipped prompts → vendor practice | `bench/diff_prompt_corpus.py`, a report |

Nothing checks the manual's prose itself. Before trusting a generated table:

    python bench/build_prompt_catalogue.py --check
    python bench/build_prompt_bank.py --check

Experiments name prompts by bank id: `bench/run_graph_arms.py` takes
`@bank:<id>`, and every render row and capture manifest records the id, text
hash, length, canvas and seed (`workflows/prompts.py::describe`).

## Grade a draft before rendering

    python bench/grade_prompt_text.py --mode fl2va --length 345 draft.txt

Same grader the graphs go through, no extra rules, nonzero exit on a FAIL.
A prompt is conformant at one duration, so always pass `--length`. For Ref2VA,
pass `--like` with a graph that wires the references the prompt declares. For a
prompt already in a graph, `bench/preflight_graph.py <graph.json>`.

## Known traps

| trap | where |
|---|---|
| The model renders what is described, not what is named; expand every name into a visual phrase and reuse it | `prompting.md` §15.3 |
| Nothing injects the duration; writers overrun the clip | `prompting.md` §15.2, §3.3 |
| The speech budget's shape is unsettled; do not cite it as measured | `prompting.md` §15.4 |
| On-screen text must be typed literally | `prompting.md` §6 |
| `(S4,S5)` means unison, not "and also" | `prompting.md` §15.3 |
| No negative prompt; guidance is distilled in | `prompting.md` §15.1 |
| Camera vocabulary is checked by `bench/check_camera_vocabulary.py`; amplitude and speed fail, an off-table motion warns | `prompting.md` §11, §13 |
| Mood words in the music line break a guide sentence nothing checks; found across a dozen bank entries on 2026-09-03 | `prompt_audit.md` |

**A rendered clip cannot A/B a prompt change.** Two arms that differ in any way
give different samples, not a better and a worse version of one. A perceptual
claim needs many seeds per arm, judged blind, in aggregate.

## Open, closed, withdrawn

Owned by `prompting.md` §14.3 and re-derived by `bench/diff_prompt_corpus.py`.
Shot line breaks: closed, ours were conformed on 2026-09-01 and earlier clips
of those scenes are not matched-seed comparable. Mouth closing: withdrawn, we
already followed the positional rule and the vendor corpus does not confirm the
pattern. Turns per shot: open in base format, within vendor practice in
reference format. Writing `N/A` for music and a one-line soundscape: house
habits, legal, never decided.

The distinction behind all of it: a sentence the guide states is a rule; an
example it shows is not. Two rules were invented here by reading examples as
rules, and both were retracted.
