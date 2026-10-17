# prompt_bank/: the house prompts

Loaded when a session opens a file here; the root `AGENTS.md` still governs.

- **House-authored H3 prompts, one a file, with `bank.json` their manifest.** `docs/prompt_bank.md` is generated
  from them by `bench/build_prompt_bank.py`; never hand-edit it.
- **The authority on how to write an H3 prompt is MiniMax's own guides** in `vendor_guides/`; `docs/prompting.md` is
  our reading of them.
- **A prompt is written for its frame count**; rendering it at another length makes the model improvise past the
  script. Short is allowed (`docs/h3_resolutions.md`, "The length axis").
- **A scene's text is data**, kept here or beside its frames, never in code or a graph.
