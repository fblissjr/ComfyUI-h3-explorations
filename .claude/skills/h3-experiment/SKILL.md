---
name: h3-experiment
description: Route the design of a new render experiment or probe in this repo -- "test whether", "set up an arm", "new probe graph", "write a prompt for", "sweep", "try this LoRA / sampler / canvas / steps", "what should the prompt say" -- to the authorities for what counts as an answer, how an arm becomes a graph, how a prompt is written, and what to run before the card is touched. Points at them; restates nothing that could drift.
reviewed: b3ff4a4
---

# Designing an experiment in this repo

Nothing here is the process. Each line names the file that is, in the order a
design goes through them.

1. **What would count as an answer.** `docs/roadmap.md` ("The right rail:
   what good means", then "The regime question"). `docs/open_experiments.md`
   says what is deliberately unmeasured and why. `docs/evidence.md` says
   which numbers must not be built on.

2. **Relative to what, in which regime, at which canvas and seeds.** The
   baseline is defined in words in `VISION.md` and named as a graph in
   `CLAUDE.md`; every claim is relative to it and says so. Ask the owner
   which canvases and aspect ratios before any bench, and default to ones
   cheaper than 16:9 (a standing instruction). `docs/roadmap.md`'s regime
   section says what transfers between the base model and the distilled
   students. A perceptual claim needs a distribution of seeds, never a pair
   (`CLAUDE.md`).

3. **How an arm becomes a graph.** `workflows/build_workflows.py` generates
   every graph from `workflows/h3_config.py`; a `workflows/*.json` is never
   hand-edited, and nothing is true of a graph until it is rebuilt. A probe
   row is written by copying an existing one: `workflows/build_workflows.py::_probe_note`
   is the template and its docstring the contract. Sol-Attn is on in every
   video graph except the stems in
   `bench/check_attention_defaults.py::SOL_EXEMPT_STEMS`. A setting no
   vendor row attests runs as a `bench/run_graph_arms.py --set` patch, not
   as a shipped row; that script's docstring says when.

4. **How a prompt is written.** `docs/prompting.md` owns it end to end, and
   the `h3-prompt` sibling skill routes into it by task. Shipped prompt text
   lives in `prompt_bank/`, loaded by id; `docs/prompt_bank.md` says how.
   `bench/check_ref_prompt_labels.py` enforces that a reference prompt's
   labels match what the graph wires.

5. **Before the card.** `bench/preflight_graph.py <graph>` on any new or
   hand-built graph. `bench/check_distill_settings.py` on any LoRA row.
   Regenerate and run the fast checks `docs/checks.md` indexes. A new bench
   tool gets one throwaway invocation read end to end before any batch, and
   an instrument gets a deliberate-violation test (`CLAUDE.md`).

6. **Running and judging.** The `h3-ab-session` sibling skill and
   `docs/eval_comparison.md` section 3 own everything from the first render
   to the verdict record.

If a line here disagrees with the file it names, the file is right and this
skill is stale. `bench/check_skill_routes.py` reports when a named file has
changed since `reviewed` above; re-read this skill against it and bump the
commit.
