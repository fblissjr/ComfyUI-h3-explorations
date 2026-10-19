# docs/h3/: the model and its ComfyUI path

Loaded when a session opens a file here; `docs/AGENTS.md` and the root `AGENTS.md` still govern.

- **`index.md` lists every page here.** Read it first; a new page gets a row.
- **Canvas, frame grid and length values are code** (`h3_rules.py`, `resolution.py`, `workflows/h3_config.py`);
  a page says what they mean and points at them.
- **Two models live here** (the DiT and the text encoder), and "attention" and "capture" name something at each: say
  which one a claim is about.
