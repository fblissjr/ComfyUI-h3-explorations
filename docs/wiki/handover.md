# Handover: read this before anything else

Written 2026-10-21 by mrtent (the Claude lead after mrtusk, from about 16:18Z), with mrgordon on the card and Gemini
(mrgemx, with its reader subagent mrgemx-reader) advising. Where this page and a record disagree, the record is right:
- `data/fun/laid.jsonl`, `data/fun/owner_answers.jsonl`, `data/fun/verdicts/verdicts.jsonl`, `data/fun/shots.json`,
  `data/fun/cast/cast.jsonl`;
- [`decisions.md`](decisions.md), 2026-10-21 at the top;
- `git log`.
Running notes: `internal/claude/2026-10-21_mrtent/notes.md`. The plan for fixing what misleads:
`internal/claude/2026-10-21_mrtent/fix_plan.md`. The mailboxes: `internal/claude/2026-10-21_mrtent_to_mrgordon.md`
(T1-T25) and `..._mrgordon_to_mrtent.md` (S1-S17e); `internal/gemini/2026-10-21_*`.

## State at this writing (about 17:55Z)

- **The owner reviewed** every new second-subject piece on the review page; their verdicts are in owner_answers
  :1360-1399. decisions.md 2026-10-21 lists them.
- **The sync (step 1)** removed the ten pieces called wrong, set four kept, and took the verdicts. Its `--clip` rebuild
  was running at this writing (mrgordon, from 17:15Z). When it ends, in order (T17, T23, T24):
  1. land `claude-noface-hair` 27a9c806 (CLEARED; `internal/claude/2026-10-21_ready_noface-hair.md`);
  2. take out 0278c, set 4726c kept, and record "nothing to make" for the third on 5792, with no rebuild;
  3. start ComfyUI; render the 5768e hair test on its existing preview, BEFORE the eye fix lands (one variable);
  4. queue the lead's 23 never-run previews plus a new 1189 preview.
- **Built, in blind read:** `claude-eyes-closed` 0b8ab179: lowered eyes drawn closed, both eyes per frame
  (`internal/claude/2026-10-21_ready_eyes-closed.md`; review: `..._review_eyes-closed.md` when it lands). It lands
  only after the 5768e test renders.
- **mrgordon is building** W3d-a (`h3_worktrees/mrgordon-cut-map-a`): build_cut_map made reliable. It is parked
  while the sync runs.
- **Servers:** the review server :8765 (pid 17874, mrtusk's, left for the owner). ComfyUI starts after the sync.

## What not to assume

- **A kept subject stays.** It is never remade to make the other subject; the other subject is made on the newest
  cut that holds it, with the whole chain made again on that cut. Pass 1's output is pass 2's source.
- **`--no-face-on` emptied her hair** until claude-noface-hair lands. After it lands, check that the frames in the
  record (part_grades.json `did`) show her hair kept on hair-and-face loads.
- **The eye fix has a flicker risk** on loads where one eye alone dips past the line for a frame or two (4726c,
  4479b, 5533b, 5768e). The builder's note has the counts. None of the three eye tests has it. Decide after the tests.
- **A reader's read, Claude's or Gemini's, is never the owner's.** The owner's eye on playback accepts a piece.
- **The tiny-face pieces (1093c, 1419c, 4647d)** may need more zoom than the 2x cap. The zoom is held: low priority
  by the owner's word (Open List r21_08).
- **1039d's 1039-1040** stay drawn open after the eye fix (the dropped second trigger). Watch them in its test.

## Next up, in order

1. Steps 1-4 of the state above (mrgordon).
2. **Reader jobs** go from mrgordon to Gemini's reader directly (`internal/gemini/2026-10-21_mrgordon_to_reader.md`;
   brief `internal/gemini/2026-10-21_reader_brief.md`): a look at each preview, at each load before it renders, and
   a card on each render. Only problems come to the lead.
3. **The eye fix:** after its blind read and the lead's clearance, land it, then render the eye tests 0415g, 1039d
   and 0484f on their existing previews.
4. **Remakes, each from a fresh preview** after the tests are read:
   - the mom: 5533 and 4647 on the new cut (a kept third is there); 4479, 5114, 5011 and 0278 on the raw source;
     4912 after its tracking is solved;
   - the mouth (r18_05): 0014f, the lead's 2625 and 1097c.
5. **The lead's first renders** (23 shots plus 1189) after the eye fix, in batches the owner reviews.
6. **The third's second passes** on shots whose mom is kept (4602, 5461, 5675 now), each on the newest cut.
7. **The docs pass (W6) and data docs**, after both builds land: §8 as the trace order, §1's pass order, the
   `--no-face-on` lines, the 0415g claims, the runbook's commands, data/fun/AGENTS.md as pointers, and the runbook as
   the one copy (`fix_plan.md`).
8. **Builds queued:** W7 (reuse made visible), the stop-and-show gate with fresh-eyes clearance (r21_04/05), the
   every-box tracker seeding (r21_06), the rest of W3d, pick_frames reading the eyes.

## Where things are

- Today's traces: `internal/claude/2026-10-21_{dark_hair_trace,g10_verify,1039d_eyes_trace,4726c_specks_trace}.md`.
- Today's audits: `internal/claude/2026-10-21_audit_data_docs.md`, `..._audit_masked_v2v_runbook.md`.
- Build notes and reviews: `internal/claude/2026-10-21_{ready,review}_*.md`.
- Worktrees to remove at the close, once each is checked landed and clean: claude-control-audit (scratch),
  mrsim-laid-owner, mrsim-laid-frames, mrsim-audio-pcm, claude-pick-frames, and today's builds once landed.
  mrgem-cut-map is Gemini's: ask it.
