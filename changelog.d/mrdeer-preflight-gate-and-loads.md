bump: minor

### Added

- **The preflight is a gate a builder can read.** `flags.json` from
  `bench/capture_masked_run.py preflight` carries `verdict` (`clear` or
  `blocked`), `blocking` and `overridden`, and every flag a `key` (its rule,
  who it is about and its frames). `preflight --gate` exits `GATE_BLOCKED`
  while a flag at the top level stands with no override; without `--gate`
  the exit is 0 as before. `outcome <capture> <flag> overridden --by WHO
  --note WHY` records a decision to render past a flag (both required) and
  recomputes the verdict; the override holds for the flag as it stood, under
  any number, and a flag that names other frames on the next run blocks
  again. What a render did against a flag is still `yes` or `no` and is not
  an override.
- **A plan can be one load, and a pass a list of loads.** `--plan` takes
  `at=FIRST,frames=N` (the plan exists on those frames only and its latent
  steps are counted from its own first frame, ending on its last) with
  `text=`, `audio=` and `still=` carried into the manifest under `load`;
  `--loads FILE.json` gives a whole pass as a builder writes it, defaults
  once and one entry a load. One rule for a load,
  `load_starts_on_a_small_subject`: the subject's mask on the load's first
  frame is under `SMALL_START` of the largest it gets in that load.

`bench/check_capture_masked_run.py` pins both: the verdict and the override
through a real `outcome` call, a pass file, a load's own frames, and a cut
that a load from its shot's first frame does not have inside a step.
