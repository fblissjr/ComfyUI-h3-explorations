# docs/prompting/: writing H3 prompts

Loaded when a session opens a file here; `docs/AGENTS.md` and the root `AGENTS.md` still govern.

- **`index.md` lists every page here.** Read it first; a new page gets a row.
- **`prompting.md` is the only home of a prompting rule.** The `h3-prompt` skill routes here.
- **Generated, never edited by hand**: `prompt_bank.md` (`bench/build_prompt_bank.py`) and `prompt_catalogue.md`
  (`bench/build_prompt_catalogue.py`). A scene's text is data in `prompt_bank/`, never in code or a graph.
- **`portable/` is read by `bench/check_prompt_docs_sync.py`**: the pinned rule sentences must stay in the manual and
  both copies, and a dated snapshot there is never edited.
