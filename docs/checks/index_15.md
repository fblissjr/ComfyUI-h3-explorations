# Check Index Detail — Part 15 (`masked_code_map.py` to `smoke_h3.py`)

> **Extracted from [`docs/checks.md`](../checks.md) on 2026-10-27.**
> Full verbatim `defends`, `needs`, and `claims block` rows for `masked_code_map.py` through `smoke_h3.py`. For the complete compact index and check standard, see [`docs/checks.md`](../checks.md).

| check | defends | needs | claims block |
|---|---|---|---|
| `masked_code_map.py` | **a generator and AST drift gate; `--check` is the gate.** Parses the 44 git-tracked masked-lane Python modules with `ast`, generates `docs/masked_v2v/CODE_MAP_INDEX.md` (`--write`) with live `StartLine-EndLine` coordinates, 8-char AST body digests, extracted `cannot_see` and key constants (`2e.1`), the bidirectional module dependency graph (`2e.2`, `--deps`), function call graph (`--calls`), artifact/dict-key writer and reader "both-ends" queries (`--data`), and pipeline step subprocess chains (`2e.3`, `--chain`), reports which narrative stage/contract blocks in `docs/masked_v2v/code_map_stages.md` and `docs/masked_v2v/code_map_contracts.md` had their underlying Python functions change (`--drift`), and in `--check` mode verifies that `CODE_MAP_INDEX.md` matches the live AST, every `<path>.py::<symbol>` citation resolves, every code-map `.md` file stays under `MAX_FILE_BYTES` (`42,000` bytes, fitting inside a single `read_file` call) and `MAX_LINE_CHARS` (`400` chars, preventing single-line table `edit_file` stalls), and zero machine-specific `/home/` or `/Users/` paths appear (2026-10-27) | the standard library (`ast`); no CUDA, no model weights, no server | no |
| `check_masked_code_map.py` | that `bench/masked_code_map.py` extracts live AST symbols, tuple-unpacked constants (`SHORTEST_CHAINED`, `MAX_LENGTH`, `HELD_EDGE_EMPTY_RUN`, `HELD_EDGE_LEAST`), `CANNOT_SEE_CONSTANTS`, module-level dependency graphs (`compute_dependency_graph`), function-level call graphs (`compute_call_graph`), artifact/dict-key writer and reader chains (`scan_ast_data_hits`), and CLI step chains (`compute_step_chain`) across all 44 masked-lane files, renders `CODE_MAP_INDEX.md` with no lines exceeding `MAX_LINE_CHARS`, and fails `--check` on each of the five negative controls (a corrupted AST digest, a missing symbol citation, an oversized file exceeding `MAX_FILE_BYTES`, an oversized line exceeding `MAX_LINE_CHARS`, and a leaked `/home/` or `/Users/` path) while passing both in the full repository and inside `bench/break_checks.py` copies where `docs/` is omitted (2026-10-27; seen red via `bench/break_checks.py`) | `bench/_lib`, the standard library (`ast`); no CUDA, no model weights, no server | no |
| `preflight_graph.py` | **not a gate -- a report, run by hand before you queue.** Grades a prompt against the guide's mechanical rules (ordinals derived from sockets, every defined label cited in `detailed_description`, one retention line per label, markers never crossing the visual/audio sets, no `(Sx)` in `retention_analysis`, `<d>` placement and language tag, NO timestamp on any shot header (owner rule 2026-09-18; every spelling, and a malformed header, since 2026-10-01), mid-shot times inside the clip) and prices the packed sequence statically. Takes paths, never globs, so hand-built graphs are gradeable -- which is why it exists: the two prompt checks only ever saw `workflows/*_api.json`. **Reports, never refuses**, because a tool that blocks you in your own repo gets disabled. Names what it cannot count: a video reference is a floor, not a budget. Imports `wired_labels`, `_audio_sections_optional` and `_STRUCTURE_PROBES` rather than restating them -- the first run without the last two reported FAIL on 7 of 8 image graphs for sections that structurally cannot apply. **Graded one guide and one node name until 2026-08-21**, and so answered `nothing to grade` over every base-format graph -- a line indistinguishable from a pass. It now selects ref-en's six sections or base-en's three from the graph's own sockets, reads the shot and dialogue rules against whichever guide's main field applies (`coderef/MiniMax-H3/skills/h3-prompt-writing/references/ref-en.txt:229` names the pair), applies the 350-500 word budget only on the ref side where `coderef/MiniMax-H3/skills/h3-prompt-writing/references/ref-en.txt:242` puts it, and grades base-en's Part One in both directions: a keyframe task must carry the alignment instruction, and T2VA must not. **Gained the reference-audio grader 2026-08-22**, and it is the named control for two "musts" that had none: that every ref-audio socket reaches the node through a `TrimAudioDuration`, and that the baked trim still agrees with `length` -- which it does not the moment a bench patches the length alone, as `run_graph_arms --set` routinely does. It also ffprobes the media each socket resolves to and warns on a mono track (gap 7, which raises rather than degrading) -- reporting it, never fixing it, and reporting "unreadable" as its own state so a missing ffprobe cannot read as a pass **Since 2026-10-01 it reads every prompt carrier from `workflows/prompts.py::carriers`**, so a song graph is graded as the texts its Prompt Lists expand to (a placeholder with no list, or a list nothing uses, is a FAIL, as the node refuses both) and no longer answers "nothing to grade"; pricing is not modelled for the song node and the output says so. | reference images on disk; no CUDA, no model, no server | no |
| `smoke_h3.py` | the H3 chain composes and runs, after any node-pack update. **The only thing here that actually POSTs a prompt**, and on 2026-08-13 it was the only reason anyone discovered every API graph was unsubmittable -- `validate_api` was asserting the wrong shape, so no static check could have found it | live ComfyUI, GPU, model | no |


**2026-08-23 amendments to the long rows above.**
`check_reference_order.py` now admits only the four served VHS video-loader
classes, even when an unrelated node happens to expose values at slots 0 and
3. Soundtrack provenance follows both branches of `JoinAudioChannels`;
`foreign` is diagnostic rather than refused because ownership is the video's
typed record. `preflight_graph.py` now reads typed append chains as well as
legacy sockets, and base Part One is exact: mode, final shot, and snapped
duration are resolved against templates parsed from the release guide. Its red
harness now carries the plausible wrong-mode, wrong-duration, and wrong-shot
mutations that presence-only grading missed. These clauses supersede the older
soundtrack-refusal and Part-One-presence wording retained inside the historical
rows.

The shipped-reference migration supersedes the older explicit-trim wording in
the `preflight_graph.py` row. All generated reference workflows now use the
typed compiler, whose duration cap and mono normalization are derived at run
time; preflight still audits explicit trims on native socket graphs. The Qwen
bounds audit follows typed image append chains and refuses partial resolution.

Release video sizing is local policy, not native status. The runtime check
pins the real processor's raw 31-versus-padded-32 boundary and the separate
VAE/Qwen views; preflight marks `comfy` or `release` as active and calls the
latter a local typed policy. Native core remains the `comfy` row.

"claims block" means the file carries a `Claims, i.e. what breaks if a case is
deleted:` header enumerating what each case defends. The table, rather than a
count copied into prose, records which files carry one.


