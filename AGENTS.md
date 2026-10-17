# ComfyUI-h3-explorations

MiniMax H3 research hub for ComfyUI: attention kernels, keyframe and
provenance nodes, benchmarks, workflows, and a masked video-to-video lane.
`README.md` is what ships; `VISION.md` is what the repo holds itself to.
This file is the operative form: short, and every rule here is one a
session cannot work around without noticing. A rule that is easily worked
around is friction, not a rule, and goes.

**The repo is broader than the current project.** This pack is for all things H3: many custom nodes (reference
conditioning, attention, keyframes, prompts, capture and more; every module and the nodes it defines are in
`INDEX.md`, every use in `docs/wiki/h3_uses.md`), the masked lane among them. **The project in hand** is masked
video generation: replacing people in an existing music video, `fun`, a region at a
time; its process end to end is `docs/wiki/masked_v2v_pipeline.md`, its lane `docs/wiki/masked_v2v.md`. A change made
for the project does not narrow the pack: a node it touches keeps serving its other uses.

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
- 1344x768 is a trained canvas; probes run at 1152x768 or 1344x768. The
  longest window is 362 frames, `h3_rules.MAX_LENGTH`, the owner's word of
  2026-10-15, and there is no other cap. Longest is not best: attention grows
  with the square of the sequence, so a window is as long as its shot needs
  and no longer (`docs/h3_resolutions.md`, "The length axis").
- To H3 a masked load is not an edit. The region is a hole of noise and H3
  generates new video in it; everything outside is frozen context it already
  has, and the frozen video and the audio round the hole steer what it makes
  (it was trained on video and audio jointly: told she sings, it opens her
  mouth with the song, with no word of when). A text says who is generated
  in each hole and from which reference, never the frozen picture (where a
  person stands, what she wears, the room) and never as an edit (`docs/wiki/masked_v2v.md`, "The frame every masked
  text is written in"). The owner's word of 2026-10-10, made core 2026-10-15.
- A text says a person sings only where they sing in that shot or window, and
  otherwise says nothing about singing or talking: no "silent", no "does not
  sing", no "talks" (the owner, 2026-10-15). Who sings is plan data,
  `subjects.<s>.voice`; only "sings" writes a line (`bench/load_text.py`).
- Two models live here and "attention" and "capture" each name something at
  the DiT and at the encoder: check what a node takes before carrying a
  claim across.
- A face, head or hair replacement takes its reference as the body model cut
  to one box round the head with the pose model's face points drawn on it
  (`bench/head_reference.py`), and a text that says exactly what that video
  is. What every load carries beyond that is the chain's defaults, never a
  session's pick: each job step prints them as "THE DEFAULTS THIS LOAD
  CARRIES" and keeps them in the job file
  (`bench/shot_to_load.py::defaults_carried`; the drawing's form is
  `REFERENCE_DEFAULT`). Turning one off takes a written reason. The owner's
  standard of 2026-10-12, made the chain's defaults on 2026-10-14 so the
  tools are not forgotten (`docs/wiki/decisions.md`, both days). A better
  form does not remake a piece the owner kept.

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
  Subagents never run git in the shared tree (a worker commits only on its
  own branch in its own worktree). The owner pushes from their own terminal, so
  every commit message is written as already public. While the card runs,
  any file its chain executes (the node pack's code and every `bench/` tool
  a load step calls) is edited in a worktree outside `custom_nodes` and
  landed by the Claude lead, between steps, on the Gemini lead's word: an uncommitted edit in the shared
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
- **One ledger, made new on every landing.** What stands in a video, where
  each piece came from and the gaps between pieces are the render ledger and
  the delivery table (`bench/render_ledger.py`, then `bench/delivery_table.py`,
  from the build records). After every landing or stopped load the Gemini
  lead makes both new, then makes the owner's two local pages from those same two files
  (`bench/review_pages.py`): `PROVENANCE.html` and `TO_REVIEW.html` in the
  video's folder on the share (`Video/<video>/`, one copy each, never a day's
  folder). A session, and the outside reviewer, reads the files; the owner
  reads the pages; nothing is typed or written twice. A piece the table does
  not lay is not in the video.
- **Checks**: `docs/checks.md` is the standard and the index; add one only
  when something escapes the existing ones, and every tool worth keeping is
  tracked with one, never parked in `internal/` or a scratchpad.

## Where state lives

One home per kind of fact. Not two.

| the fact | its one home |
|---|---|
| decisions waiting on the owner, their answers, and their passing thoughts | the H3 Open List (the owner's page with Claude): one card a decision; the "Tell Claude anything" box is the owner's inbox |
| what stands in a video, where each piece came from, and the gaps | the newest ledger and delivery table under `data/<date>_<video>_delivery_tables/`, generated, kept by the Gemini lead; the owner's `Video/<video>/PROVENANCE.html` and `TO_REVIEW.html` on the share are made from them |
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
- **Two leads, one each side** (the owner, 2026-10-17; until then two Claude
  sessions and a Gemini reviewer). "The lead" elsewhere in this file means
  whichever of the two holds the thing named.
  - **The Gemini lead** (a named Gemini session) co-leads the day and leads
    every other Gemini session: it holds the card (ComfyUI, captures, plan
    strips, renders), builds or has its workers build, keeps the ledger, the
    delivery table and the owner's pages current, and edits the plan file.
  - **The Claude lead** (one Claude session) is oversight and the method's
    expert: it rules on method when the leads disagree, holds the Open List
    and the inbox (Gemini cannot write there), is the only one that commits
    to `main`, and writes the changelog, the handover and `decisions.md`.
  - **They talk only to each other**, by `internal/gemini/PROTOCOL.md`: two
    append-only files a day, one writer each
    (`<date>_claude_to_gemini.md`, `<date>_gemini_to_claude.md`); every ask
    carries a deadline and a default that is never "wait"; only an Open List
    row blocks anything. Words the owner gives the Gemini lead come to the
    Claude lead as `OWNER SAID` and go on the Open List. Gemini workers talk
    to the Gemini lead only, by its own files.
- **A build goes on its own branch** in a git worktree outside
  `custom_nodes` (`h3_worktrees/<builder>-<item>`), cut from current main and
  committed there, never in the shared tree and never pushed. It comes to the
  Claude lead `READY` with its checks' output, `bench/break_checks.py`'s
  output and a blind read by a session that did not build it; the Claude
  lead lands it from the merge base (`git diff main...<branch>`), by
  pathspec, so a stale copy cannot delete newer lines (the owner,
  2026-10-16), after re-running one of its checks itself. A refusal test
  passes only on the tool's own refusal words with no Python traceback.
- **Gemini watches video and hears audio natively**, where a Claude session
  sees stills and measures, so the playback questions go to it: motion,
  flicker, hair coming and going, a mouth with the singing, eyes closing.
  Hand it the stacked clips with sound and the frames to watch, and say in
  the request that it uses its native video and audio understanding first,
  with code only to augment it, never in its place. Its watching is a
  reader's, never the owner's acceptance.
- **The owner, when available, takes the cheap looks first** (who is in a
  shot, what is replaced), before anything is built on them: in batches
  where possible, one by one only when a decision is urgent.
- **A render's prompt is read in FULL before it renders**, from the graph
  as queued (`bench/what_h3_got.py`), never from a summary, a preview or a
  paraphrase: by the session that queues it, and by a second reader for
  any shot the owner names. The owner's word, 2026-10-15.
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
