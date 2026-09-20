---
name: h3-prompt
description: Route any work on an H3 prompt in this repo -- "write a prompt", "edit this prompt", "improve this scene", "adapt it", "convert this t2v prompt to ref2va", "why did this render badly", "is this prompt correct", "what should the speaker tags be" -- to the file that owns each answer, and to the command that verifies the result. Points at them; restates nothing that could drift.
reviewed: 0b8c3b1e
---

# Working on an H3 prompt

Nothing here is a rule. Each line names the file that is. **If a line here
disagrees with the file it names, the file is right and this skill is stale.**

## The one authority

`docs/prompting.md` is the single source of truth for every mode, and it is
self-contained. Its section "Four layers, and every rule says which one it
is" defines how binding each rule is; carry the layer whenever you quote a
rule. `docs/wiki/prompting.md` is the router if you want the map first.

## By what you are doing

**Writing a new one.** Section 2 of `docs/prompting.md` decides the field
structure per mode; section 10 has graded worked examples to copy the shape
of. Fix the frame count before writing: a prompt is only correct at a
duration, for the reason section 2 gives. Fit scene beats and shot counts to the narrative. Shot headers carry NO
timestamps (the owner's house rule, 2026-09-18: `[Shot 2] The shot cuts to ...`);
a time is used only to split action inside one shot. Fit dialogue to the shot's speaking time
so actors neither rush nor sit in dead air (section 5.10). In ref2va the speaker
id is reused at every vocal event and a soundtrack cue takes `<Audio N>` instead
(section 5.1, stated by ref 5.4; base_en states neither, section 12.13). Place
every speaker in the frame (section 5.2); give every action an agent and keep
every count consistent with the shot (section 15.3 items 6 and 7, house). For character likeness in T2VA, introduce
subjects as `[Name] (played by [Actor] in [Show])` once in Shot 1, keeping vocal
timbre in narrative prose outside `<d>`.

**Editing or improving a shipped one.** The text lives in `prompt_bank/`
and nowhere else; `docs/prompt_bank.md` says which graphs render each
entry and how the generator loads one by id. Read its verdict in
`docs/prompt_audit.md`, which also says whether the prompt is carried by a
matched pair. Edit the bank file, never a `workflows/*.json`, then rebuild
the bank doc and the graphs.

**Converting between modes.** Section 2 for the target mode's structure,
section 12 for where the guides are silent or disagree, section 14.3 for
where our shipped prompts diverged and what happened to each. Do not
pattern-match a fix across the base and reference formats; section 12 owns
the boundary.

**Deciding whether a prompt is good.** `python bench/grade_prompt_text.py --mode <mode> <prompt.txt>`
grades loose text; `python bench/preflight_graph.py` grades a prompt already in a graph
and prices the sequence; `python bench/diff_prompt_corpus.py` reports where our
prompts diverge from vendor practice. Each docstring is its contract.

**Handing the rules to someone outside this repo.**
`docs/portable/h3_prompt_standard.html`.

## Before you believe your own edit

- Rebuild and run `bench/check_prompt_docs_sync.py`.
- A rendered clip cannot A/B a prompt change: `docs/eval_comparison.md`
  owns what a perceptual claim needs.
- Cite the observable, never a sentence that describes it (`CLAUDE.md`).

`bench/check_skill_routes.py` reports when a file named here has changed
since `reviewed` above; re-read this skill against it and bump the commit.
