# Handover: read this before anything else

Rewritten 2026-10-23 at the close by mrtx, the one lead (no Gemini lead; Gemini sessions reviewed the day at the owner's
hand). Where this page and a record disagree, the record is right: `git log`, [`decisions.md`](decisions.md) 2026-10-23,
the lane rules in [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), and `data/fun/`. Read next, and check before acting
on it: `internal/claude/2026-10-23_mrtx/tomorrow_plan.md`, the next day's rough plan, by cause, with the reasons. The
bugs it numbers are `bugs_for_tomorrow.md` beside it; the running notes are `notes.md` there.

## State when closed

- **Landed today**, each after a blind read: the beard below the jaw under a plan word (0764f585), the second subject's
  hair wording (84d01c5d), a load's own `--margin` (823e4934), short head-tube dips bridged (de4ed3cd), the closed mouth
  drawn on the lips (62e895b8, untested in a render), `--hold-part` (e350885f), strangers only on another detected
  person (14838f44), the tracker (513783e6: every owner box seeds, drawn boxes approximate, `--head-per-stretch`), a kept
  accessory placed on the render's own face (9efd2f2b), and the semicolon refusal in `--head-per-stretch` (38f1f47b).
- **ComfyUI** was restarted once, at 22:16Z, for the tracker; stopped at the close, unarmed, with an empty queue. **The review server**: stopped at the close.
- **The cut**: the 2026-10-24T0045Z rebuild (`data/2026-10-24_fun_delivery_tables/rebuild_fun_2026-10-24T0045Z.json`;
  stamped by UTC, made the evening of 2026-10-23). Its table and cut name the laid list as it stands, its cut check
  passes, and its gap check has no failures. Laid today: `data/fun/laid.jsonl` lines 43-60, each citing the owner's answer; second_hfp_5114i
  is `unconfirmed`, a stand-in until its closed-mouth remake.
- **Every review item is answered.** Remakes owed: the lead's 3967 (her face lost on 3969-3970) and the third subject's
  4479, 5011, 5114, 5461 and 5675. 3098b is kept: its render is fine; its drawing is a replay test, not a remake.
- **The cast file**: the second subject's drawn box on 4697 is withdrawn (`data/fun/cast/cast.jsonl:193`); it took the
  third subject.
- **Not landed**: `gem-chin-and-lips-guard` (Gemini's, for the next day's blind read and replay) and
  `claude-hair-own-track` (parked), with the older `claude-round-hole`, `claude-change-check`, `claude-seed-every-box`
  and `mrgem-cut-map`. The worktrees of the day's landed branches are removed and their branches kept; `git worktree
  list` shows what is left, none of it holding uncommitted work.
  Some of the day's scratch from `/tmp` is kept under `internal/claude/2026-10-23_mrtx/kept_from_tmp/`: the owner's
  sheet boxes and the sheet-to-frame fits for 4647 and 4912. The third subject's blue boxes still to enter are 4647's
  only (4719, 4721, 4722; bug 20); 4912's boxes for the second are in.

## What not to assume

- **The third subject's loads take the default margin.** A larger `--margin` let H3 copy the drawing's white outlines
  onto his face (decisions.md, 2026-10-23). 4479d staying clean at 64 does not make it safe. Nor is the margin proven
  the whole cause: the lines came where the drawing was least enlarged (4479d 5.33x clean; 5011e 2.27x, 5675f 1.52x
  not; bug 27), and Gemini's mechanism is unchecked (bug 29).
- **5114i's remake waits on the chin guard** (`gem-chin-and-lips-guard`, 52c38f23, not landed): no closed mouth goes on
  before it. Its blind read and a replay against kept drawings, lit necks included, are a step of their own; until
  then 5114i stays `unconfirmed`.
- **A drawing that ran has not been checked.** Its log prints figures (the face score, the mouth gate, the body-model
  clock) and none of them decides anything. On 3098b, 4479d and 3967b the marks sat off the grey face (bugs 27, 29, 31).
- **The drawn eyes are what keep the gaze alive.** Without them every subject had dead eyes copying their reference
  photo; the lead's hallway (1306t), 1156g and 0278m show what they solved. Eye marks are left off only where the eyes
  cannot be seen in the film anyway, and those keeps are the mark check's controls.
- **`--no-face-on` is not a way to hide bad marks** on a face-only load (the lead's, the third's): it says no face
  shows, and the capture empties the part on those frames, leaving the original face as filmed (`data/AGENTS.md`).
- **The grey head's own points exist**, in every preview's `<preview>_<subject>_pose_table.json` (`body_pose.py`). The
  plain cut's points file is a copy of the film's, and the mouth gate reads only the lit grey figure.
- **A load's recorded `MASK_VERSION` is read from disk** and proves nothing about the live server. The proof is the shot
  table's `ran_code`, the class's own text at `/object_info`, and the port owner's start time.
- **Old strangers files are refused** (the form changed with 14838f44): every remake starts from a fresh preview.
- **Keep means done**: a piece with a fault a remake will fix is marked wrong (`data/AGENTS.md`).
- **Land only with the card held.** Two steps ran on uncommitted landing code today; it changed nothing by luck (bug 28).
- **The reviewer's readings are checked before use.** Two of its claims did not hold tonight (decisions.md).

## Tomorrow, in order

The plan's section 3 has the reasons (the landed tools first, new code only where they fail). This order revises it
after the close's reviews (decisions.md, 2026-10-24); where the two differ, this one stands:

1. Its section 0: the cut is current, the server unarmed, the tree clean, the owner's answers read to the end. Then
   the name check (bug 1, bench only), before any new job: `free_name`'s `taken` also counts every day's
   `data/*_jobs/<name>.json`, a moved-aside `<name>.json.<suffix>` included (4053b's evening clash was one).
2. The third subject's loads back on the default margin.
3. The previews that need no new code, on the card while steps 4-5 are built (landed only with the card held): 3967
   (bug 26), 4912; 4227, 3476, 4774, 5044; 4310, 5935c, 6105, 3595; 3240 (lead_fp_3240e's fault, bug 17, is the
   strangers form 14838f44 replaced). 4647 after its missing boxes (bugs 20, 23).
4. A pose-table printout per drawing: the frames where the marks miss the grey face. It decides nothing; a reader
   names the bad ones. Then a small drawing-only word that leaves those frames' eye, brow and nose marks out and keeps
   the part (`bench/draw_face_points.py --bare` already draws nothing on named frames: wire it through, rather than
   build anew). Not `--no-face-on`, which empties a face-only part.
5. 3967: `--hold-part 3967-3970@3971` and the drawing-only word on 3969-3970. The film's points on 3969 also size the
   reference's one crop box: pass those frames to `bench/head_reference.py::fill_from_neighbours` too, or compare the
   box with them left out.
6. The beard: first, on CPU, the classes under his jaw on 5675, 5011, 5114 and 5461 from the preview's class video
   (a capture's `classes__*.npz` is the head tube's clipped map), and 5114c's and 5461d's drawing enlargement. Then
   one render of 5675, the region grown downward only at the default margin, against 5675c; 5011, 5114 and 5461 by
   its result.
7. 4479: the drawing-only word on the frames named, then a zoom test on its own.
8. 1419: a reader's sheet, hidden marks where she is truly hidden. Not `--hold-part`: it holds a part unmoved, and she
   walks across 1426-1459. Expect the boxes not to seed 1431-1436 (bug 22): plan the `_head_seed` fix
   (`subject_track.py:1237`) and the day's one restart, with the card held, if the sheet shows her visible there.
9. second_hfp_5114i's closed-mouth remake, once the chin guard is landed (above).

Waiting on the owner: 0516c's frame 597 and 3759b's 3856-3857 (the share's `Video/fun/looks/2026-10-23_head_marks/`).

## Where things are

- `internal/claude/2026-10-23_mrtx/`: `notes.md`, `bugs_for_tomorrow.md`, `tomorrow_plan.md`, `restart_steps.md`,
  `gemini_traces_sorted.md`, the commit messages and the server and rebuild logs.
- The reviews: `internal/gemini/2026-10-23_*.md` (the briefs, `design_review.md`, `day_review.md`).
- The owner's boxes drawn on sheets: `data/fun/owner_sheets/`. Today's looks: the share's `Video/fun/looks/2026-10-23_*`.
