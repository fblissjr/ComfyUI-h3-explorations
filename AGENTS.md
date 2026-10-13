# ComfyUI-h3-explorations

MiniMax H3 research hub for ComfyUI: attention kernels, keyframe and
provenance nodes, benchmarks, workflows, and a masked video-to-video lane.
`README.md` is what ships; `VISION.md` is what the repo holds itself to.
This file is the operative form: short, and every rule here is one a
session cannot work around without noticing. A rule that is easily worked
around is friction, not a rule, and goes.

## Do not trust prose, including this file's

When prose and code disagree, the code is right. Every sentence that names
a default, a shipped artifact or what "every graph" does is a cache with no
invalidation. Cite the observable: `h3_config.MODELS["clip"]` over "the
shipped encoder is"; a node's `define_schema` over "the default is"; a walk
of `graph_paths` over "every graph". When prose has lost, correct it and
log what it used to claim in `docs/wiki/decisions.md`.

## Settled about H3

Each was measured or decided and has been re-derived as news at least
once; `docs/evidence.md`, "Settled about H3", has the rest.

- Sol-Attn is on by default in every shipped video workflow; the exempt set
  is `bench/check_attention_defaults.py::SOL_EXEMPT_STEMS`.
- The baseline is `workflows/bench/h3_text_to_video_dense_stamped_api.json`:
  stock attention, no sage, Sol or LoRA, the base step count.
- 1344x768 is a trained canvas; probes run at 1152x768 or 1344x768 with 345
  frames (`docs/h3_resolutions.md`).
- Two models live here and "attention" and "capture" each name something at
  the DiT and at the encoder: check what a node takes before carrying a
  claim across.
- A face, head or hair replacement takes its reference as the body model cut
  to one box round the head with the pose model's face points drawn on it
  (eye outlines with the iris, brows, nose; `bench/head_reference.py`), and a
  text that says exactly what that video is. The owner's standard of
  2026-10-12, judged better by their eye on three of the lead's shots that
  evening (`docs/wiki/decisions.md`, 2026-10-12; `data/fun/verdicts/`).
  The lid marks at full-frame size are the form it replaced.

## Operative rules

- **A tinkering repo.** When sglang and ComfyUI's own node or comfy-kitchen
  agree on a default and ours differs, adopt theirs in `workflows/h3_config.py`
  and the generator, rebuild, and say in the commit what it replaced. One
  upstream alone, or a knob no upstream has, is a judgement call. Closed lanes
  stay closed unless the owner reopens them (`docs/roadmap.md`, "Closed lanes").
- **Never hand-edit `workflows/*.json`.** `workflows/build_workflows.py`
  generates them, and nothing is true of a graph until it is rebuilt.
  Every walker goes through `workflows/h3_config.py::graph_paths`.
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
  Subagents never run git. The owner pushes from their own terminal, so
  every commit message is written as already public.
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
| decisions waiting on the owner, their answers, and their passing thoughts | the H3 Open List (the owner's page with Claude): one card a decision; the "Tell Claude anything" box is the owner's inbox |
| what a video still needs | `bench/shot_coverage.py` on `data/<video>/shots.json`, `plan.json` and the delivery table; regenerated, never typed |
| who is in each shot, and the owner's answers on the pictures | `data/<video>/shots.json` and `cast/cast.jsonl`; `bench/who_page.py` is the page the owner answers on |
| the owner's verdicts on renders | `data/<video>/verdicts/verdicts.jsonl`, printed under each render by the coverage table |
| the day boundary | `docs/wiki/handover.md` (about 80 lines, four headings: state when closed; what not to assume; tomorrow, in order; where things are) and the top block of `docs/wiki/next_steps.md` |
| what was decided or reversed, and when | `docs/wiki/decisions.md` |
| everything else | `docs/wiki/index.md`, the router |

## How the owner and the sessions talk

- **One inbox.** The owner's thoughts go in the Open List's box as they come,
  with a time. Nothing there is a go. The lead reads it at its next natural
  stop and writes one of three words under each: done (with where), queued
  (as a row, with where in the order), or parked (with why).
- **The terminal is for a go and for an answer to a question a session
  asked.** A session that receives a thought in the terminal writes it to
  the inbox and carries on. A go is a row on the Open List or a line in the
  lead's terminal; a session that is not the lead takes the lead's relay as
  the go and writes the relay's time in the job's note.
- **Two sessions during the day**: one holds the card and builds, one reads
  cold and builds tooling. A third only as a blind reader on a finished
  result, briefed with file paths and no conclusions; a subagent counts.
- **The owner, when available, takes the cheap looks first** (who is in a
  shot, what is replaced), before anything is built on them: in batches
  where possible, one by one only when a decision is urgent.
- **A render is handed over stacked** (the masked input with its segments
  and ids on top, the render below, frame numbers burned in), never bare;
  a thing seen on playback is reported with the file and the frame.
- **Handoff over compaction**: a session near its limit writes its notes,
  the rows it owes and the job's note, and the next session starts from
  files. Sessions hand off staggered, never both at once.

## The lanes

- Masked video to video: `docs/wiki/masked_v2v.md` ("The rule: data before
  a render, and the same data after"; "Reading one window whole"; "Rules
  of the lane").
- Prompts: `docs/prompting.md`; a scene's text is data served beside its
  frames, never stored in code or a graph.
- Reference implementations: `coderef/` (gitignored) is read, never
  imported; `docs/wiki/references.md`.
- `workflows/h3_config.py` holds every shared constant, one copy each;
  `vendor_config/` holds the release's own config files verbatim;
  `internal/` is gitignored and never shipped.
