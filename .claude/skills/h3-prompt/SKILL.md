---
name: h3-prompt
description: Route any work on an H3 prompt in this repo -- "write a prompt", "edit this prompt", "improve this scene", "adapt it", "convert this t2v prompt to ref2va", "why did this render badly", "is this prompt correct", "what should the speaker tags be" -- to the file that owns each answer, and to the command that verifies the result. Points at them; restates nothing that could drift.
reviewed: edc3ab03
---

# Working on an H3 prompt

Nothing here is a rule. Each line names the file that is. **If a line here
disagrees with the file it names, the file is right and this skill is stale.**
Name sections, not rules: a rule restated here is a second place to edit.

## The one authority

`docs/prompting/prompting.md` is the single source of truth for every mode, and it is
self-contained. Its opening table says where everything else lives. Its section
"Four layers, and every rule says which one it is" defines how binding each
rule is; carry the layer whenever you quote a rule.

## By what you are doing

**Writing a new one.** Fix the frame count first: a prompt is only correct at a
duration (`docs/prompting/prompting.md` sections 3.3 and 10). Then section 2 for the
structure of the mode, section 3.1 for shot headers, section 4 for camera,
section 5 for speakers and dialogue (5.10 fits dialogue to the shot), section 6
for on-screen text, section 8 for the audio fields and section 9 when it has
references. Section 16 is what to pin and what to leave to the model, with a
reading pass to run before rendering. Section 10 has graded worked examples to
copy the shape of.

**Editing or improving a shipped one.** The text lives in `prompt_bank/`;
`docs/prompting/prompt_bank.md` says which graphs render each entry and how the generator
loads one by id. A composed ref2va entry also has source fragments in
`workflows/build_workflows.py` (`REF_SCENE_SHOTS`, `_composed_from_bank`), which
the build checks against the bank file. Read the entry's verdict in
`docs/prompting/prompt_audit.md`, which also says whether a matched pair carries it. Edit
the bank file, never a `workflows/*.json`, then rebuild the bank doc and the
graphs.

**Converting between modes.** Section 2 for the target mode's structure,
section 12 for where the guides are silent or disagree, section 14.3 for where
our shipped prompts diverged and what happened to each. Do not pattern-match a
fix across the base and reference formats; section 12 owns the boundary.

**Deciding whether a prompt is good.** `python bench/grade_prompt_text.py --mode <mode> <prompt.txt>`
grades loose text; `python bench/preflight_graph.py` grades a prompt already in a graph
and prices the sequence; `python bench/diff_prompt_corpus.py` reports where our
prompts diverge from vendor practice. Each docstring is its contract. None of
them can see whether a prompt pinned what the scene turns on; section 16.4 says
what can.

**Handing the rules to someone outside this repo.**
`docs/prompting/portable/h3_prompt_standard.html`. A copy published elsewhere is a dated
snapshot of whatever it was published from.

## Before you believe your own edit

- Rebuild, then run `bench/build_prompt_bank.py --check`,
  `bench/check_prompt_docs_sync.py`, `bench/check_prompt_guide_conformance.py`,
  `bench/check_prompt_rule_controls.py` and `bench/check_ref_prompt_labels.py`.
- A node that carries a prompt belongs in `h3_config.PROMPT_INPUTS`, the one
  list graders read; `bench/check_prompt_guide_conformance.py` fails on one that
  is not there.
- A rendered clip cannot A/B a prompt change: `docs/method/eval_comparison.md`
  owns what a perceptual claim needs.
- Cite the observable, never a sentence that describes it (`CLAUDE.md`).

`bench/check_skill_routes.py` reports when a file named here has changed
since `reviewed` above; re-read this skill against it and bump the commit.
