# data/: the masked video-to-video lane and the projects in it

Loaded when a session opens a file here; the root `AGENTS.md` still governs. Masked video-to-video is one use of the
pack, not the pack (root `AGENTS.md`, "The repo is broader than the current project"): a rule here is the lane's.

## The project in motion

- **`fun`**: replacing people in an existing music video (`fun.mp4`, frames 0-6394 of the working copy), a region at
  a time. Where it stands: `docs/wiki/handover.md` first; then the newest ledger, delivery table and gap check under
  `data/<date>_fun_delivery_tables/`; the owner's decisions on the H3 Open List. What it still needs:
  `bench/shot_coverage.py` on `data/fun/shots.json`, `plan.json` and the newest table.
- **Who is in each shot of `fun` is answered by the owner for the whole video.** Read it, never ask it again unless a
  reading leaves you uncertain (`docs/wiki/working_with_the_owner.md`): `data/fun/shots.json` (the record, written
  only by its tools), `data/fun/cast/cast.jsonl`, the raw `data/fun/owner_answers.jsonl`, the generated view
  `data/fun/who_v2.json` (`bench/who_view.py`), and the page `/who/` on the review server.

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
| videos | the share's `Video/<video>/`, never under `data/` |

## Handing a render over

- A render is handed over stacked (`bench/stack_delivery.py`: source, what changed, the cut, per person what H3 saw),
  frame numbers burned in, never bare; a thing seen on playback is reported with the file and the frame. A clip shown
  to be judged shows every subject of the shot replaced, as the cut would.
- Playback questions go to Gemini, which watches video and hears audio natively: say in the request that it watches
  and listens first, code only to augment. Its watching is a reader's, never the owner's acceptance.
