# Handover: read this before anything else

Rewritten 2026-10-29 at 18:20Z (13:20 Central) by the Claude advisory session (`mrfam`), from the files, at a mid-day
handoff: the Gemini lead session (`gemso`) compacted at least six times, its last messages looped, and the owner is
replacing it. Where this page and a record disagree, the record is right: `git log`, [`decisions.md`](decisions.md)
(2026-10-29 at the top), the lane rules in [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md) and
[`../masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md), the queue in
[`../masked_v2v/fun_render_backlog.md`](../masked_v2v/fun_render_backlog.md) ("Today's queue"), and `data/fun/`.

## State when handed off

- **Kept and laid today, 2026-10-29** (`data/fun/owner_answers.jsonl:1675-1691`, `data/fun/laid.jsonl` 77 rows,
  `sha256 d3f8f7df`): `lead_fp_0438b`, `lead_fp_1134b`, `lead_fp_4272c` (15:22Z, replacing `0438e`, `1134`, `4272b`),
  `lead_fp_4053d` (16:10Z, replacing `4053c`, the first reference under `dc07ccee`), `lead_fp_3240f` (16:38Z, new
  row, the first shot that had never been rendered), `lead_fp_4774c` (about 18:00Z, new row, 4774 to 4826),
  `lead_fp_3476c` (kept about 18:25Z, "looks amazing, you got it at all the transitions"; not yet laid when this was
  written, so the ledger reads 77 rows until `laid.py` adds it). Seven keeps, zero wasted renders. Rejected:
  `second_hfp_4912o` (15:22Z, dark hair).
- **Reopened by the owner and taken off the ledger** (`laid.removed.jsonl` rows 15 and 16, 18:06Z):
  `second_hfp_0014g` (14 to 167, "janky, artifacts around the mask edges") and `second_hfp_0278m` (278 to 293, her
  eyes open where the source's are not). Their remakes are the lead's next renders (below). Until they are kept, a
  rebuilt cut shows the original mom on those frames; the owner knows and has not ruled whether a wrong piece should
  stay laid until replaced (`decisions.md`, 2026-10-29, the two reopening entries).
- **Nothing rendered is waiting on a verdict.** `lead_fp_3476c`'s reading is at
  `data/2026-10-29_lead_fp_3476c/looks/lead_fp_3476c.reading.md`; its lay is step 2 below.
- **Built, not rendered**: `lead_fp_5049b` (5049 to 5104, 56 frames, on `shot_c_5044f`): job, reference and text done;
  the outgoing lead's last note says the reference is re-run with `--no-draw-face-on 5097-5104` (already in the job
  file, `data/2026-10-29_gemlead_jobs/lead_fp_5049b.json`) before `first`. Head-off holds on 5058 to 5059, 5066 to
  5078, 5094 to 5096 from looks at the zoomed doorway sheets; 5044 to 5048 and 5105 to 5113 recorded `not seen`.
- **Fresh previews on the card today**: `shot_c_5044f`, `4774f`, `3476f`, `4227f`, `3240f`, `3595f`, `6105g` (all
  queued; strangers run on 5044f, 4227f, 3595f; not yet on 6105g). Preview graphs for the two remakes, `shot_c_0014f`
  and `shot_c_0278g`, are built on CPU and not yet queued.
- **The cut**: the clip on the review page is `fun_0000_6394_joined_2026-10-29T1450Z_from_laid`, assembled 15:12Z
  from the eight keeps of 10-28 and nothing kept today. The 18:14Z run under `data/2026-10-29_fun_delivery_tables/`
  wrote a ledger, table and reading from the current 77 rows but no clip and no rebuild record. The close rebuild
  with `--clip` is owed.
- **Code on `main` today**: none. Docs only: the backlog's queue table and 4912 row (`e3df1261`..`de902712`), the
  reader rule in the router, protocol and skill (`093759dd`), the owner's invalidation rule (`a48d97fc`), the
  reopenings and keeps (`bb6e2f8d`, `729efc7a`), the 10-28 changelog and bookkeeping (`5afa1ddb`).
- **The 4912 test**: branch `gem/4912-hair` at `21753055` in `../../h3_worktrees/gem-4912-hair`, one commit: an
  optional `encode_frames` input on the masked source node, graph and job-builder wiring, check cases. The test job
  `second_hfp_4912p` and its lightness-lifted conditioning clip are built under `data/2026-10-29_second_hfp_4912p*`
  and `internal/gemini/2026-10-29_4912/` (root cause, strips, script). Masks, depth, reference (remade under current
  code, byte-identical) and seed are `4912o`'s. Awaiting: the blind read (the Claude session), then the landing by
  pathspec from the merge base on an empty queue with a server restart, then the `gem4912` session runs the one
  render through every gate. The owner's rule: if it comes back dark, document and move on.
- **Servers**: `:8188` and `:8765` up; the ComfyUI queue was empty at 18:18Z.

## What not to assume

- The outgoing lead did not write this page; nothing here comes from its memory. Its compaction count and the
  reconstruction of its day from the `agy` transcripts are the fresh session's subagent's job, as a postmortem record
  under `internal/gemini/`, and a cross-check on this page, not a replacement for it.
- `dc07ccee`'s reference box is validated by two keeps on shots it had never crossed (`4053d`, `3240f`); the three
  renders made under the earlier rule were kept as they were. No remake is owed on that account.
- 4912 is active, not parked, and the owner never parked it; the owner's rule for it is one render, then document and
  move on. 6105 is unblocked by the lead's verified detections and two `not seen` spans, not by any word of the owner's.
- A session's measured reading is not a hold, and a tool's floor is not a look: 4774's cut moved from 4823 to 4826 on a
  look at the frames; 5044's drops were checked frame by frame against the visible-face floor and then looked at.
- Twice in two days the SAM head track locked onto her hair and the head tube suppressed her face (`4647b` yesterday,
  `4774c` today, 4806 to 4812); both times the remedy was `--no-head-track`. A face-only subject can afford that; a
  hair-and-face one cannot. Two build candidates, not for today: seed the head track on a frame with face keypoints;
  the head tube does not suppress a face when the part model's face pixels are many times the head track's.
- Depth bleeds across an occlusion edge onto the re-emerging subject's first sliver, so the keep-in-front margin cannot
  separate them (4912 frame 4926, `internal/gemini/2026-10-29_4912/strip_4926_depth_who_is_who.jpg`). A general
  effect and a build candidate; untouched for the 4912 test on purpose.
- Every time written on a record is read from the clock or a file; this morning's estimates in the Claude file were
  wrong by hours and are corrected there.

## Next, in order

1. Read this page, `decisions.md` (2026-10-29), the backlog's "Today's queue" table and row 1.9, and
   `docs/comfy_notes.md` before the first command.
2. Lay `lead_fp_3476c` with `laid.py` against its keep line in `owner_answers.jsonl` (about 18:25Z), if the outgoing lead has not.
3. `lead_fp_5049b`: reference as the job says, `first`, `plan`, `on-plan`, `table`, the look by a reader who did not
   build it, the full prompt with `what_h3_got.py`, `render`, `land`, the reading, `page`. Never `land && page`.
4. The two remakes: queue `shot_c_0278g` and `shot_c_0014f` on the card; `second_hfp_0278n` with
   `--eyes-shut-on 278-293 --eyes-shut-form closed`; `second_hfp_0014h` on the recipe with today's chain. Readers
   compare the eyes and the mask edges against the old pieces as well as the source, since those are the faults named.
5. The 4912 landing when the Claude blind read is green: on an empty queue, by pathspec from the merge base, restart
   the server by the recipe in `docs/comfy_notes.md`, then the `gem4912` session renders `second_hfp_4912p`.
6. Then `lead_fp_4227c` and `lead_fp_3595b` (previews and strangers fresh; flags in backlog §5.1), then 6105 as three
   stretches once strangers has run on `shot_c_6105g`.
7. Close: `rebuild_delivery.py --video fun --clip`, hash read, the page from it; `decisions.md` and this page from the
   files; `changelog.d/day-2026-10-29.md`; the server stopped by the recipe after reading `/proc/<pid>/environ`.

## Where things are

- Today's queue and freshness per span: [`../masked_v2v/fun_render_backlog.md`](../masked_v2v/fun_render_backlog.md), §3.
- Today's rulings: [`decisions.md`](decisions.md), 2026-10-29 (reopenings, the invalidation rule, the reader rule).
- Readings of every render today: `data/2026-10-29_<load>/looks/<load>.reading.md`.
- The Claude day file, with the postmortem of 10-28, the plan, the check table and every correction:
  `internal/claude/2026-10-29_mrfam/postmortem_and_plan.md`; the validation of the lead's postmortem beside it.
- The owner's annotated frame on 4912: `data/fun/owner_sheets/2026-10-29_4912_frame_4921_neighbor_hair_owner.webp`.
- Ledger and removals: `data/fun/laid.jsonl` (77 rows) and `data/fun/laid.removed.jsonl` (16 rows).
