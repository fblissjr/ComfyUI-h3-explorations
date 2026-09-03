---
name: h3-ab-session
description: Route any rendered comparison in this repo -- "compare these", "A/B", "which LoRA / checkpoint / sampler / knob is better", "blind session", "score the clips", "judge the pair" -- to the one documented process (render with matched seeds, blind, score before unblinding, record the aggregate). Points at the authority; restates nothing that could drift.
reviewed: 2f19c5c
---

# A rendered comparison in this repo

Read `docs/eval_comparison.md`, section 3, and follow it. That file owns the
process; this skill only makes sure it is reached. Every claim is relative to
the baseline `VISION.md` defines and `CLAUDE.md` names as a graph.

The four pieces, in the order the process runs them; each script's docstring
is its contract:

1. `bench/run_graph_arms.py` renders the arms.
2. `bench/blind_batch.py` blinds them and seals the key under
   `internal/blind_keys/`.
3. The owner scores in the app `bench/blind_score_app.py` generates.
4. `bench/score_session.py` opens the key and writes the verdict record
   under `bench/results/`.

Three rules the process does not relax, each owned elsewhere:

- A rendered clip cannot A/B a numerical knob: `CLAUDE.md`, operative rules.
- The key stays sealed until the scores exist, and the judge sees no arm
  label anywhere: `docs/eval_comparison.md` section 3.
- The output share's path is typed in the shell, never written into the
  repo: `bench/_paths.py` and the `H3_COMFY_OUTPUT` variable it reads.

If a line here disagrees with the file it names, the file is right and this
skill is stale. `bench/check_skill_routes.py` reports when a named file has
changed since `reviewed` above; re-read this skill against it and bump the
commit.
