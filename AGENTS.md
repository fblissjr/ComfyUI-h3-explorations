# ComfyUI-h3-explorations

MiniMax H3 research hub for ComfyUI: attention kernels, keyframe and
provenance nodes, benchmarks, workflows, and a masked video-to-video lane.
`README.md` is what ships; `VISION.md` is what the repo holds itself to.
This file is the operative form: short, and every rule here is one a
session cannot work around without noticing. A rule that is easily worked
around is friction, not a rule, and goes.

**The repo is broader than the current project.** This is the owner's whole MiniMax H3 research repo (every module
and node in `INDEX.md`, every use in `docs/wiki/h3_uses.md`). Masked video-to-video, and the projects in it such as
`fun`, is one use: its rules and state are `data/AGENTS.md`. So a lane rule is never read as a rule of H3 or the pack;
common code (reference conditioning, the shared nodes) serves every use, the lane included, so a finding about it
applies everywhere; and a shared node or default changes only when it is right for every use, masking-only behavior
going behind the lane's own tool, input or plan data.

## Do not trust prose, including this file's

When prose and code disagree, the code is right. Every sentence that names
a default, a shipped artifact or what "every graph" does is a cache with no
invalidation. Cite the observable: `h3_config.MODELS["clip"]` over "the
shipped encoder is"; a node's `define_schema` over "the default is"; a walk
of `graph_paths` over "every graph". When prose has lost, correct it and
log what it used to claim in `docs/wiki/decisions.md`. The same rule holds
one layer down in the masked lane: a capture's `.npz` and `.json` files are
the chain's cache, and the node's own inputs and writes are the artifact
(`docs/masked_v2v/verifying_raw_frames_and_keeps.md`).

## Settled about H3

Each was measured or decided and has been re-derived as news at least
once; `docs/evidence.md`, "Settled about H3", has the rest.

- Sol-Attn is on by default in every shipped video workflow; the exempt set
  is `bench/check_attention_defaults.py::SOL_EXEMPT_STEMS`.
- The baseline is `workflows/bench/h3_text_to_video_dense_stamped_api.json`:
  stock attention, no sage, Sol or LoRA, the base step count.
- 1344x768 is a trained canvas; probes run at 1152x768 or 1344x768. The
  longest window is 362 frames, `h3_rules.MAX_LENGTH`, the owner's word of
  2026-10-15, and there is no other cap. Longest is not best: attention grows
  with the square of the sequence, so a window is as long as its shot needs
  and no longer (`docs/h3_resolutions.md`, "The length axis").
- Two models live here and "attention" and "capture" each name something at
  the DiT and at the encoder: check what a node takes before carrying a
  claim across.

## Operative rules

- **A tinkering repo.** When sglang and ComfyUI's own node or comfy-kitchen
  agree on a default and ours differs, adopt theirs in `workflows/h3_config.py`
  and the generator, rebuild, and say in the commit what it replaced. One
  upstream alone, or a knob no upstream has, is a judgement call. Closed lanes
  stay closed unless the owner reopens them (`docs/roadmap.md`, "Closed lanes").
- **Never hand-edit `workflows/*.json`.** `workflows/build_workflows.py`
  generates them, and nothing is true of a graph until it is rebuilt.
  Every walker goes through `workflows/h3_config.py::graph_paths`.
- **Never run `uv` in this repo** (`uv run`, `uv sync`, `uv add`): with the ComfyUI venv active it has emptied
  that venv and dirtied `uv.lock`. Run every script with the ComfyUI venv's python directly (`docs/comfy_notes.md`).
- **Nothing in this pack patches ComfyUI core**; `bench/check_vsa_core_patch.py`
  is the observable.
- **The server process is the resource, not the GPU.** An idle server can
  be an armed one: read `/proc/<pid>/environ` for `H3_*` keys and ask before
  killing it. Start and stop it by the recipe in `docs/comfy_notes.md`.
  ComfyUI caches: a wall time or VRAM figure says which cache state it was
  measured in; `/history` is the observable for which arm ran.
- **The shared tree.** `git status` before editing, and commit promptly with
  `git commit -F <msg> -- <paths>`. Never `git add`, `commit -a`, a bare
  `git commit` or `--amend`: each has swept a peer's work into another
  commit. The one exception is `git add -- <file>` for a new file you
  created. A file holding a peer's uncommitted hunk is theirs to commit.
  Subagents never run git in the shared tree (a worker commits only on its
  own branch in its own worktree). The owner pushes from their own terminal, so
  every commit message is written as already public. While the card runs,
  any file its chain executes (the node pack's code and every `bench/` tool
  a load step calls) is edited in a worktree outside `custom_nodes` and
  landed by the Gemini lead between steps, once the Claude lead clears it: an uncommitted edit in the shared
  tree is run by the next load.
- **The changelog is written once a day, from the commits**
  (`changelog.d/README.md`). A commit carries no changelog entry.
- **Nothing an earlier run made is reused by a key.** A result is made
  again, or handed in explicitly as a named file the job says it reads.
- **A record under `bench/results/` is its `.md`; data goes under the
  untracked `data/<video>/`; video files go under the share's
  `output/Video/<video>/`, never under `data/`.**
- **A constant carries its provenance beside it**: measured, inherited, or
  reasoned. A rule that depends on the picture takes its value from the
  input video, up front; a constant from another clip is a marked starting
  value.
- **Green or red only for a hard failure**: a node a graph names that does
  not exist, a tracked tool with no index row, a link that does not resolve.
  Anything of degree is data printed with what it was measured on and what
  it cannot see, and the reading is left to whoever runs it. No fuzzy red.
- **A rendered clip cannot A/B a numerical change**; grade on captured
  activations (`bench/grade_sage_on_capture.py`).
- **No render that does not decide something.** One clip, one seed, one
  shot is not a result; a repeat and a different shot are two tests.
- **Checks**: `docs/checks.md` is the standard and the index; add one only
  when something escapes the existing ones, and every tool worth keeping is
  tracked with one, never parked in `internal/` or a scratchpad.

## Where state lives

One home per kind of fact. Not two.

| the fact | its one home |
|---|---|
| decisions waiting on the owner, their answers, and their passing thoughts | the local files (`docs/wiki/decisions.md`, `docs/wiki/handover.md`, `docs/masked_v2v/`, and `data/<video>/`); a hosted page such as the H3 Open List may carry decisions to the owner and is never read back as the record |
| everything about a video in the masked lane (what stands, what it needs, who is in each shot, the owner's answers and verdicts) | `data/AGENTS.md`, "Where the lane's state lives" |
| the day boundary | `docs/wiki/handover.md` (about 80 lines, four headings: state when closed; what not to assume; tomorrow, in order; where things are) and the top block of `docs/wiki/next_steps.md` |
| what was decided or reversed, and when | `docs/wiki/decisions.md` |
| everything else | `docs/wiki/index.md`, the router |

## How the owner and the sessions talk

- **The owner's working preferences** (their clock, batching, never re-asking, where links go, how a clip is
  shown, how a question names its page, writing) are `docs/wiki/working_with_the_owner.md`, read by every session
  of either model before it speaks to the owner.
- **One inbox.** The owner's thoughts go in the Open List's box as they come,
  with a time. Nothing there is a go. The lead reads it at its next natural
  stop and writes one of three words under each: done (with where), queued
  (as a row, with where in the order), or parked (with why).
- **The terminal is for a go and for an answer to a question a session
  asked.** A session that receives a thought in the terminal writes it to
  the inbox and carries on. A go is a row on the Open List or a line in the
  lead's terminal; a session that is not the lead takes the lead's relay as
  the go and writes the relay's time in the job's note.
- **Two leads, one each side** (the owner, 2026-10-17). The Gemini lead holds the card, git on `main`, the ledger,
  the owner's pages and the plan file, and leads the other Gemini sessions; the Claude lead holds the Open List and
  the inbox, clears each landing, and writes the changelog, the handover and `decisions.md`. They talk only to each
  other; the mailbox, the landing steps and what each holds are `internal/gemini/PROTOCOL.md`. When one model is not
  available on a day, there is one designated lead, which holds both sides' roles; where it is unclear what to do or
  how to go forward, that lead asks the owner instead of guessing (the owner, 2026-10-18). A Gemini lead cannot write
  the Open List, so on such a day its questions go to the owner in its terminal.
- **A lead starting a day reads, in this order, before its first word to anyone** (the owner, 2026-10-18):
  `docs/wiki/handover.md`; `docs/wiki/working_with_the_owner.md`; `internal/gemini/PROTOCOL.md` (its "Starting a
  day as the Gemini lead" has the rest of the order); `docs/comfy_notes.md`, before the first command; then
  `data/AGENTS.md` and the project's own before lane work. Then `date -u`, `git status`, HEAD and the ports, and its
  first entry says what it read and what it starts on.
- **A build goes on its own branch** in a worktree outside `custom_nodes` (`h3_worktrees/<builder>-<item>`), cut from
  current main, never pushed, and lands by pathspec from the merge base (`git diff main...<branch>`) after its own
  blind read and the Claude lead's clearance. A refusal test passes only on the tool's own words with no traceback.
- **A change to data, an input or an output names both ends.** A build that adds or changes a file, a key, a field or
  a format lists every tool that writes it and every tool that reads it (by grep, with file:line), and says what each
  does with the change: updated, unaffected and why, or refusing on purpose. Its READY carries that list; the Claude lead
  clears nothing without it. Today's misses were each one step away: a matte key the assembler read, a re-laid graph
  the ledger trusted, a page the review server served (2026-10-17).
- **A render's prompt is read in FULL before it renders**, from the graph
  as queued (`bench/what_h3_got.py`), never from a summary, a preview or a
  paraphrase: by the session that queues it, and by a second reader for
  any shot the owner names. The owner's word, 2026-10-15.
- **Handoff over compaction**: a session near its limit writes its notes,
  the rows it owes and the job's note, and the next session starts from
  files. Sessions hand off staggered, never both at once.

## The lanes

- **Folders with their own `AGENTS.md`**: `bench/`, `workflows/`, `docs/`, `docs/masked_v2v/`, `prompt_bank/`, `archive/`, `vendor/`,
  `vendor_config/`, `vendor_guides/`, `standalone/`, `contact_sheets/` and `data/`. Read the folder's before working in
  it. Claude Code loads it when it reads a file there; the Antigravity CLI evaluates folder rules only during prompt
  expansion (walking up from where the session starts, or a file @-mentioned in a prompt), not when a tool reads a file
  mid-turn (its rules page, CLI tab; seen 2026-10-17).
- Masked video to video: read `data/AGENTS.md`, `docs/masked_v2v/AGENTS.md`, and `docs/masked_v2v/PROTOCOL.md` (the lane's rules and execution protocol, tracked) and the project's own
  `data/<video>/AGENTS.md` (its state, local and untracked like the rest of `data/`) before lane work.
- Prompts: `docs/prompting.md`; a scene's text is data served beside its
  frames, never stored in code or a graph.
- Reference implementations: `coderef/` (gitignored) is read, never
  imported; `docs/wiki/references.md`.
- `workflows/h3_config.py` holds every shared constant, one copy each;
  `vendor_config/` holds the release's own config files verbatim;
  `internal/` is gitignored and never shipped.
