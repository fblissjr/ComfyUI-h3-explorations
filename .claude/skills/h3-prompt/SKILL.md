---
name: h3-prompt
description: Route any work on an H3 prompt in this repo -- "write a prompt", "edit this prompt", "improve this scene", "adapt it", "convert this t2v prompt to ref2va", "why did this render badly", "is this prompt correct", "what should the speaker tags be" -- to the file that owns each answer, and to the command that verifies the result. Points at them; restates nothing that could drift.
---

# Working on an H3 prompt

Nothing here is a rule. Each line names the file that is. **If a line here
disagrees with the file it names, the file is right and this skill is stale.**

## The one authority

`docs/prompting.md` is the single source of truth for every mode. It is
self-contained -- you do not need `internal/` to write a prompt. Every rule in
it carries a layer: **GUIDE-STATED** binds (breaking it is off-distribution),
**GUIDE-SHOWN** is worth following and must not be enforced, **OWNER** and
**HOUSE** are ours and may be wrong, **OPEN** is unsettled. Collapsing those is
how two rules were invented here and retracted, so carry the layer whenever you
quote a rule.

`docs/wiki/prompting.md` is the router if you want the map first.

## By what you are doing

**Writing a new one.** Pick the mode first (§2) -- it decides the field
structure and the exact Part One line, and the three keyframe templates are not
interchangeable. §10 has graded worked examples per mode; copy the nearest one's
shape. Fix the frame count before writing: `S.SS` and every cut timestamp
resolve against the snapped length, so a prompt is only correct at a duration.

**Editing or improving a shipped one.** Find it in `docs/prompt_catalogue.md`
(generated from the graphs -- run `bench/build_prompt_catalogue.py --check`
first, it has gone stale) and read its verdict in `docs/prompt_audit.md`.
**Then edit `workflows/build_workflows.py`, never a `workflows/*.json`**, and
rebuild -- nothing is true of a graph until it is regenerated. If the prompt is
carried by a matched pair or an experiment arm, the audit says so: changing one
arm and not its twin destroys the experiment.

**Converting between modes.** The dangerous one. Read §2 for the target mode's
structure and §14.3 for where our prompts have diverged. Two traps:
- **Base and reference formats take OPPOSITE corrections.** A fix that is right
  for t2va can be wrong for ref2va and the reverse; never pattern-match across
  that boundary. §12 lists where the guides are silent or disagree.
- **The Part One line changes with the mode and so do the brackets.** Getting
  it wrong is silent -- the render succeeds and the alignment is simply wrong.

**Deciding whether a prompt is good.** `bench/grade_prompt_text.py --mode <mode>
--length <frames> file.txt` grades loose text against the guide's mechanical
rules; `bench/preflight_graph.py <graph.json>` does the same for a prompt
already in a graph, and prices the packed sequence. Both report and never
refuse. `bench/diff_prompt_corpus.py` reports where our prompts diverge from
vendor practice on things no guide states.

**Handing the rules to someone outside this repo.**
`docs/portable/h3_prompt_standard.html`, published and self-contained.

## Before you believe your own edit

- Rebuild the graphs and re-run `bench/check_prompt_docs_sync.py`, which grades
  the manual's own quotations and re-runs every §10 example through the grader.
- A rendered clip **cannot** A/B a prompt change: two arms draw different
  samples and diverge at frame 0. `docs/eval_comparison.md` owns what a
  perceptual claim actually needs, and it is a distribution, not a pair.
- Prose in this repo has been wrong and wrong-about-wrong. Cite the observable
  -- the guide, the graph, the generator -- not a sentence that describes it.
