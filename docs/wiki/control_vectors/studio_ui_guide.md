# The browser UI

last updated: 2026-11-09

[Back to the start page](../control_vectors.md)

The browser UI for control vectors is not maintained at present (owner, 2026-11-09: left for another day).
It was not opened or checked during the 2026-11-09 work, and the changes made that day to the build code,
the pair loader and the prompts were not carried into it. Use the command line
([`preset_catalog_and_cli_reference.md`](preset_catalog_and_cli_reference.md)) or the nodes
([`comfyui_nodes_and_controls.md`](comfyui_nodes_and_controls.md)).

It starts with:

```bash
<comfy-venv-python> bench/control_vectors.py ui
```

Its code is the package `bench/control_vector/` (`ui_server.py`, `ui_state.py`, and the pages under
`web/`). The earlier version of this page mapped each screen to a command; that table described the UI as
designed and was not checked against it, so it is not repeated here.
