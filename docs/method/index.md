# Method: how the repo decides something is true

| page | what it is for |
|---|---|
| [`evidence.md`](evidence.md) | **start here before stating a number**: what must not be relied on, what holds, the retraction-consumer ledger, and the environment that is not in git |
| [`checks.md`](checks.md) | the standard every check is held to, how to run them, and the index of every check with what it defends. Start here before changing behavior or adding a check |
| [`eval_comparison.md`](eval_comparison.md) | the A/B process (matched seeds, blinded, scored before unblinding, recorded as a distribution), which scene can show a difference, and the stacking tool |
| [`sustainability.md`](sustainability.md) | the direction argument and its do-not-do list. Read before proposing process |
| [`prose_measurements.md`](prose_measurements.md) | why a number in prose is a copy and where it goes instead. Read before writing a speedup, size or time into a doc |
| [`config_drift.md`](config_drift.md) | why `workflows/h3_config.py` drifts and what would stop it. Read before adding prose to a config file |
| [`drift_frontier.md`](drift_frontier.md) | the doc-drift and control-calibration work opened 2026-08-17, one row per decision, and what is still open |
| [`check_postmortems.md`](check_postmortems.md) | what individual check defects taught; history, not operative |
| [`rules_history.md`](rules_history.md) | `CLAUDE.md` as it stood before the 2026-09-03 cut, frozen: the dated instances behind each rule |

## The paths a judged result takes

- **Anything judged by a person**: `bench/run_graph_arms.py` -> `bench/blind_batch.py` -> `score.html` ->
  `bench/score_session.py`. A comparison that skipped a step is two samples, not a result.
- **A panel across scenes**: `bench/blind_panel.py` -> `score.html` -> `bench/join_panel_verdicts.py`, with the stop
  rules of [`research/2026-09-19_evaluation_one_judge.md`](../research/2026-09-19_evaluation_one_judge.md).
- **Before building any pair**: `bench/diff_clip_graphs.py` says what two clips actually differ in, from their
  embedded graphs.
- **A numerical change** is graded on captured activations (`bench/grade_sage_on_capture.py`), never on a rendered clip.
- **After an experiment**: `bench/clip_recipe_coverage.py` lists rendered recipes that have no graph in the repo.

Every tool's own description is in [`bench/INDEX.md`](../../bench/INDEX.md).
