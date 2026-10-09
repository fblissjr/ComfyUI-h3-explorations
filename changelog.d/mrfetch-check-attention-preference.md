bump: minor

### Added

- **`bench/check_model_contents.py` fails when a file a graph loads sets its
  own attention.** ComfyUI can now take the attention a module runs from the
  model file or from a LoRA (a key ending in `.config`), and uses it whenever
  no node has set an attention override. A graph that wires any attention
  node is not steered by it; the stamped dense baseline has none, so a file
  carrying the key would change what that render runs with nothing in the
  graph to show it. The check now opens every `.safetensors` name a constant
  in `workflows/h3_config.py` or a generated graph carries, bench graphs
  included, and is red on one that has the key unless
  `bench/results/model_contents_baseline.json` names it under
  `attention_preference_approved` with a reason (the owner's choice,
  2026-10-09). No file named today carries one. On every run it writes a
  file with such a key and is red if its own scan misses it. `--report`
  lists each file read, and `--update-baseline` carries the approved list.
  `docs/checks.md` has the row and the escape it closes.
