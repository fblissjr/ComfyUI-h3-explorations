# data/: the masked video-to-video lane and the projects in it

Loaded when a session opens a file here; the root `AGENTS.md` still governs. Masked video-to-video is one use of the
pack, not the pack (root `AGENTS.md`, "The repo is broader than the current project"): a rule here is the lane's.

## Start here

| you need | it is |
|---|---|
| where a project stands today | `docs/wiki/handover.md`, then the project's own `data/<video>/AGENTS.md` |
| what is in the cut and what gaps remain | `data/<video>/laid.jsonl`; the newest `data/<date>_<video>_delivery_tables/frame_coverage_*.md`; `bench/shot_coverage.py` |
| the day-to-day commands (ComfyUI, one load, the review server, the rebuild, the close) | `docs/wiki/runbook.md` |
| the pipeline end to end, and the lane's rules | `docs/wiki/masked_v2v_pipeline.md`, `docs/wiki/masked_v2v.md` |
| after any landing or stopped load | `bench/rebuild_delivery.py --video <video>` |

## The projects in motion

| video | its state |
|---|---|
| `fun` | `data/fun/AGENTS.md` |

A new video gets its own `data/<video>/AGENTS.md` (and the one-line `CLAUDE.md`), so nothing one project settled is
read as true of the next. A project's file is local, untracked like the rest of `data/` (the owner backs `data/` up
daily); this file, the lane's rules for any video, is the one tracked file here, so a builder's worktree has it.

## How the lane works, end to end

- The process: `docs/wiki/masked_v2v_pipeline.md`. The lane's rules: `docs/wiki/masked_v2v.md` ("The rule: data
  before a render, and the same data after"; "Reading one window whole"; "Rules of the lane"). Running a load, the
  review server, the ledger and the end of day: `docs/wiki/runbook.md`.

## Settled for the masked lane

- To H3 a masked load is not an edit. The region is a hole of noise and H3 generates new video in it; everything
  outside is frozen context it already has, and the frozen video and the audio round the hole steer what it makes (it
  was trained on video and audio jointly: told she sings, it opens her mouth with the song, with no word of when). A
  text says who is generated in each hole and from which reference, never the frozen picture (where a person stands,
  what she wears, the room) and never as an edit (`docs/wiki/masked_v2v.md`, "The frame every masked text is written
  in"). The owner's word of 2026-10-10, made core 2026-10-15.
- A text says a person sings only where they sing in that shot or window, and otherwise says nothing about singing or
  talking: no "silent", no "does not sing", no "talks" (the owner, 2026-10-15). Who sings is plan data,
  `subjects.<s>.voice`; only "sings" writes a line (`bench/load_text.py`).
- A face, head or hair replacement takes its reference as the body model cut to one box round the head with the pose
  model's face points drawn on it (`bench/head_reference.py`), and a text that says exactly what that video is. What
  every load carries beyond that is the chain's defaults, never a session's pick: each job step prints them as "THE
  DEFAULTS THIS LOAD CARRIES" and keeps them in the job file (`bench/shot_to_load.py::defaults_carried`; the drawing's
  form is `REFERENCE_DEFAULT`). Turning one off takes a written reason. A better form does not remake a piece the
  owner kept.
- No default keep-outs on a subject's own body: a thing is kept as the original only where a look names it ("she
  holds an egg on 1010-1017"); the second subject's `held`, `hands`, `clothes` and hair band were retired on
  2026-10-17 (`decisions.md`). Keep-outs for other people stay: another subject's track and strangers.
- Every laid piece is laid by its own matte under one compositing law (`bench/assemble_delivery.py::lay_plane`,
  2026-10-17). A matte is reconstructed from the render's own inputs, never today's defaults
  (`bench/recomposite_piece.py --save-own-matte`).

## One ledger, made new on every landing

What stands in a video, where each piece came from and the gaps are the render ledger, the delivery table and the gap
check, made by one command, `bench/rebuild_delivery.py`, after every landing or stopped load; it also makes the
owner's two pages, `Video/<video>/TO_REVIEW.html` and `PROVENANCE.html` on the share (one copy each), and takes the
close-of-day snapshot of the answers file. A session reads the files; the owner reads the pages; nothing is typed
twice. A piece the table does not lay is not in the video. The Gemini lead keeps them while it holds the card.

## Where the lane's state lives

| the fact | its one home |
|---|---|
| what stands in a video, where each piece came from, the gaps | the newest ledger, table and gap check under `data/<date>_<video>_delivery_tables/` |
| what is laid in the cut | `data/<video>/laid.jsonl` (`bench/laid.py`) |
| the plan: subjects, parts, keep-outs, clothing and voice by span | `data/<video>/plan.json`, edited only with a dated copy beside it and a `_by` line saying whose word |
| who is in each shot | `data/<video>/shots.json` and `cast/cast.jsonl`, written by their tools |
| every answer the owner gave on the local pages, raw | `data/<video>/owner_answers.jsonl`: append-only, cited by line, never edited |
| the owner's verdicts on renders | `data/<video>/verdicts/verdicts.jsonl` |
| a reader's playback reading of a load or clip | `data/<day>_<load>/playback_reading_<reader>.md`, written by the playback reader session and read by `bench/review_pages.py` (via `reader_file` / `reader_files` in `data/<video>/review/start_here.json`) |
| videos | the share's `Video/<video>/`, never under `data/` |

## Handing a render over

- A render is handed over stacked (`bench/stack_delivery.py`: source, what changed, the cut, per person what H3 saw),
  frame numbers burned in, never bare; a thing seen on playback is reported with the file and the frame. A clip shown
  to be judged shows every subject of the shot replaced, as the cut would.
- A clip on the review page is whole, or has its gaps counted right above the player from the gap check on its own
  frames (per person, per kind, frames and spans); a clip with gaps and no such line is not shown
  (`docs/wiki/working_with_the_owner.md`, the owner of 2026-10-17).
- Playback questions go to Gemini, which watches video and hears audio natively: say in the request that it watches
  and listens first, code only to augment. Its watching is a reader's, never the owner's acceptance.
- Each playback reader writes its reading to `data/<day>_<load>/playback_reading_<reader>.md`, starting at the top
  (before the first `---` or `##` body heading; or per clip under `### <clip>`) with a four-field card:
  - `Sees`: `frames: what` (frame numbers and what is seen on playback)
  - `Cannot see`: what the clip or view cannot show
  - `Verdict`: `keep`, `remake`, or `unsure` (the reader's)
  - `Why`: one line
  `bench/review_pages.py` reads that `.md` file from `reader_file` (or `reader_files`) in `data/<video>/review/start_here.json`
  (never typed inline into `start_here.json`), renders the card folded closed (`<details class="why reader-card">`) with
  a one-line `<summary>` that never shows the verdict so the owner watches first, and where the owner has answered that
  clip (`data/<video>/owner_answers.jsonl` or `verdicts.jsonl`), shows the owner's verdict and `agree` or `differ` beside
  the two verdicts.

