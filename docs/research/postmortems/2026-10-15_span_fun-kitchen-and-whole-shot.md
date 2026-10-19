# Postmortem, 2026-10-15: the kitchen came apart, and the whole shot came back

Written by the lead of the day (mrtall) at the close, from `docs/state/decisions.md` (2026-10-15), the three seats'
close-out notes (`internal/claude/2026-10-15_{mrarc,mrlol,mrvid}_closeout.md`), the outside reviewer's files
(`internal/gemini/2026-10-15_*`) and the lead's own session. What a postmortem here is for: who got what wrong and why,
ideas deferred, where the sessions converged. No blame. Figures live in the files named; this page points at them.

## 1. What the day produced

- The owner kept, on playback: the lead's 2823b, 0630d, 0805d, 2625 and 751c; the second subject's 5461c, 5114d and
  1010j; the dad's 1452; and the hall shot 1419-1465 made whole (lead_fp_1419d, third_fp_1419d). Their verdicts:
  `data/fun/poc/answers.jsonl`; what stands: `data/2026-10-15_fun_delivery_tables/`, the newest ledger and table. The
  dad's kept 1452 was then replaced in the table by the kept whole-shot hall cut. At the close the kitchen is still the
  split cut (0604d and 0949d meeting at 948/949) until the second subject's whole-shot load lands.
- The owner's rules, each in docs/state/decisions.md (2026-10-15), and those every session must not work around also in AGENTS.md
  ("Settled about H3", "How the owner and the sessions talk"): one load a subject over the
  whole shot; chained windows only inside one load, on the seam grid; the window ceiling is `h3_rules.MAX_LENGTH` and
  no cap of 345 anywhere; a render's prompt is read in full before it renders; to H3 a masked load is not an edit, so a
  text never describes the frozen picture; "sings" only where a person sings, nothing else about voice; a reference
  still's definition is specific and the description says less; two subjects in one masked pass is closed; two
  Claude sessions.
- The recipe that made the hall shot, kept and written down: `docs/masked_v2v/overview.md`, "A shot, made whole".
- Tools landed (each with a cold reader but one; commits in `git log --since=2026-10-15`): `--no-face-on`,
  `--hold-part`, `--seam A,B`, `--no-mouth-on`, `--no-strangers`, `--eyes-shut-on`, the voice key and the hair change
  in `load_text`, no place-and-action sentence, the mouth said only where drawn, pairs through the chain (now unused),
  the 362 ceiling and `bench/check_length_ceiling.py`, `bench/stack_rows.py`, `what_h3_got` holding rebuilt masks to
  their frames, the smoke reading Sol's earlier line.
- Coverage at the close: `data/2026-10-15_fun_delivery_tables/coverage_fun_2026-10-15T2145Z.txt` (worked through to
  3166 with the gaps named there; little after it).

## 2. What went wrong, who, and why

1. **The kitchen (604-1009) broke, by process.** On 2026-10-14 a lead narrowed the owner's one-shot rule of 10/13 to
   "a shot may be split where no face shows"; the shot was then made of separate loads, and the second subject jumped
   to "a different version of herself" at 948/949 where two fresh renders met (the reviewer's G12 found the flicker in
   her raw renders, not the blending). The lead of the 15th first suspected the lead's pieces; measurement refuted it.
   Why: a session's reading of an owner's rule replaced the rule. Now: the owner's words stand, a reversal is logged,
   and "one shot is never split" is in docs/state/decisions.md.
2. **The lead wrote texts as an editor.** The first two-subject draft told the women apart by "on the right", "on the
   left" and the apron, all frozen picture; the owner: "to h3 this is not an edit". mrlol's read of the tracks also
   found "left and right" false on a long stretch (docs/state/decisions.md, the "not an edit" entry). Why: describing the scene instead of the holes. Now a line of AGENTS.md.
3. **The lead carried a reviewer's claim to the owner as a decision.** The lead's hair "seam" came from the reviewer and
   a measurement; the owner had never seen it ("i dont remember lead singers hair ever being an issue at all"), and the
   A/B was made on a cut whose switches to the original swamped any hair. Why: escalating secondhand findings without
   the owner's eye first, and testing on broken material. Dropped.
4. **The outside reviewer quoted rules that do not exist** (from a session summary; its own correction note names
   them, `internal/gemini/2026-10-15_kitchen_two_subject_text.md`) and gave eye spans the card seat's own sheet refuted
   (docs/state/decisions.md, the evening answers). Both caught by checking the files and the frames. Now: verbatim quotes with line numbers,
   grepped first; a reviewer's frame list is a pointer, never data a load takes.
5. **The two-subject pass blended the faces.** The owner asked for the test; the code read beforehand said nothing ties
   a reference to a hole (`docs/masked_v2v/overview.md`, the two-subject section). The pair tooling went to a lane now closed. The
   cost was knowingly paid for a direct
   answer. A blind reader called it "no swap"; the card seat's stills beside the render caught the cross.
6. **Continuing 2823b was offered and then found impossible** (its stored window, off the chained grid, cannot be
   continued; the cold reader of `--continue-from`). Why: the options went to the owner before the stored window's
   geometry was read. The owner then chose a one-window remake (docs/state/decisions.md, the evening answers; Open List row 6).
7. **Process slips, each caught:** a fix landed before its cold read (416183e6); a commit swept a peer's uncommitted
   hunk (e72eddf9, put back by 06db6954); a chain script ran text and first after a FAILED reference (5935b); the lead
   wrote several times without reading the clock; `check_length_ceiling` was red on main at the close (its fix is in
   mrgo's landing); the owner's "we regressed" on 751b (answers.jsonl), a mouth mend that dropped correct mouths and a
   reference run on uncommitted shared-tree edits (docs/state/decisions.md, 2026-10-15); the review page churned for the owner (links to the Open List, cards
   in a folded section, then items) before it matched how they review.

## 3. What went well

- Cold readers before landing found real faults every time (mrlol's close-out) (the pair's forcing rule, `--hold-part`'s specks, the
  seam floor, `what_h3_got` trusting its own record, the continuation's geometry).
- Reading each reference's record and sheet before `first` stopped bad loads before they cost a render, and each stop
  became a tool (`--no-mouth-on`, `--no-face-on`, the off-track fix).
- The reviewer's native watching and listening confirmed the lead does not sing in the hall (G17) and the pair test's
  crossed faces (G18, after the card seat's stills beside the render had caught it), and its C4 found the clean context
  on the kitchen's inner window, which the lead checked in the code and `docs/masked_v2v/window_context.md` (2026-10-10 rows).
- The hall shot: once `--no-face-on` stopped faces being drawn on the backs of heads, the recipe made it and the owner
  kept it.

## 4. Deferred, with where each lives

- One hole a pass (a load given another's render as its video): next_steps, item 7; the assembler must judge such a
  piece against its base video.
- Context noise as a default for inner windows: after the kitchen render, if its 898 seam holds on playback.
- `--no-face-on` also emptying the part on face-only loads; `load_text` refusing a voice change inside one load; the
  Masked Prompt node's voice and frozen-picture wording: mrgo's queue.
- The written label on a reference: parked unlanded (worktree mrlol-label).
- The wiki's remaining staleness: `internal/claude/2026-10-15_wiki_audit.md`.
- Declined by the owner: an identity-drift reading ("just have me watch it"); the lead's hair restore (both in
  docs/state/decisions.md, the evening answers).
- Unlanded at the close: `--continue-from` (its cold read blocks the 2823b use and lists further findings, mrlol's
  close-out), the hold form (no cold read, mrvid's close-out); 5935b stopped at a failed reference; 4310b pending.
- The owner's asks still to make: the far-table mother, the dad on 1397-1418, her face over 2200-2247, 1093-1133 hair
  alone, 5675 (docs/state/decisions.md).

## 5. Where the sessions converged

- On the owner's eye as the only acceptance: readers point at frames, never rule a piece in or out.
- On reading the code before believing prose, a reviewer's quote, or a summary (the two-subject binding, context noise,
  the reviewer's citations, the eye spans).
- On making a durable form of every ruling the same hour: AGENTS.md, docs/state/decisions.md and the lane doc, with the old claim
  kept beside the new.
