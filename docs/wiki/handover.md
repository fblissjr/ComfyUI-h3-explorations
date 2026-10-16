# Handover: read this before anything else

**Rewritten at the end of each working day by the session that led it; the
day before is in git.** This one was written at the close of 2026-10-15 (the
evening, the owner's time) by `mrtall`, the lead, with `mrgo` on the card and
its tooling from the evening (`mrarc`, `mrlol` and `mrvid` held the day and
closed with notes). Where this page and a record disagree, the record is right
(`decisions.md`, 2026-10-15; `data/fun/poc/answers.jsonl`; the newest ledger and
table; the Open List).

> **IF EVERYTHING LOOKS WORSE TOMORROW, LOOK HERE FIRST.** On the owner's go of 2026-10-15 late ("No we're doing it
> tonight"), a new session, mrnight, built a change to every new head reference: a much tighter cut (the face filling
> the picture), the body model rendered at the cut's size, and the mesh's own eye sockets painted out with the iris in
> its leading corner (spec: `internal/claude/2026-10-15_gaze_spec.md`). It was to be tested on two renders before
> anything else used it: the lead's glance (1097-1133) and her phone call in the hall (1575-1729, the kept 1575b, where
> the eyes already worked). Whether it landed, and on what reading, is in `decisions.md` and mrnight's note. If the
> new references make things worse, the old cut is the default it was built beside: say so to the owner and use it.
> **STOPPED FOR THE NIGHT, NOT LANDED** (the owner, about 23:50Z: stop it "unless you think its worth finishing"; the lead:
> finish to a clean point in the worktree, land nothing). It is on branch `mrnight-gaze` (484d6302) in its worktree; where it stands
> is `internal/claude/2026-10-15_mrnight_closeout.md`. Tomorrow: a cold read, land at the card's clear, then the two
> tests. THE PIECES, SEPARATELY (take any alone; why each, harm, the lead's pick of one render): `internal/claude/2026-10-15_gaze_pieces.md`.
> Before it is wired, one question for its design (the lead, 2026-10-15 late, measured on 0604g's face points,
> `second_0604_1009_points.npz`): her face grows about 3x across 604-1009 (about 33 px at 604 to 64 by 943, 100-133 in the
> bow), so one box SIZE a load either clips the bow's head or leaves the start small. One box size a WINDOW (0604g's seam
> at 898 falls where she comes close) is the candidate; untested whether a scale step at a seam shows. When it lands, the
> new cut is OPT-IN and OFF by default (`--face-fill` / `--follow` on the reference step; every job prints a
> "reference cut" row under THE DEFAULTS THIS LOAD CARRIES saying which is on). Making it the default is a decision: only
> after both test renders read well on the owner's playback, written in `decisions.md`, with the defaults row, the check
> that holds the default (`check_shot_to_load`), this handover and `masked_v2v.md`'s recipe changed in the same commit.

## State when closed

- **Two Claude sessions from now on** (the owner): a lead, and one that holds the card and builds its tools
  (`AGENTS.md`, "How the owner and the sessions talk"). The evening's card seat's brief:
  `internal/claude/2026-10-15_brief_mrgo.md`.
- **What stands:** the newest ledger and table under `data/2026-10-15_fun_delivery_tables/` (2145Z at writing, newer if
  the card landed after), and the coverage reading beside it. The owner's pages: `Video/fun/PROVENANCE.html` and
  `TO_REVIEW.html` on the share, served at `http://100.87.141.65:8765/fun_share/`; the interactive review page is
  `http://100.87.141.65:8765/`.
- **Kept today on playback:** the lead's 2823b, 0630d, 0805d, 2625, 751c; the second subject's 5461c, 5114d, 1010j;
  the hall shot 1419-1465 made whole (lead_fp_1419d, third_fp_1419d), which replaced the short pieces there.
- **On the card tonight (mrgo), in order:** the second subject's whole kitchen load, `second_hfp_0604f` (604-1009,
  seam 898, context 90, noise 0.8 per load, eyes shut drawn on 938-953 and 968-1009 only, `--track-not lead:1003-1009`;
  job `data/2026-10-15_fun_kitchen_inputs/second_hfp_0604f.json`); then `--mouth-on` landed and `lead_fp_2823c`
  (2823-3109, one window; job `data/2026-10-15_fun_past2049_inputs/lead_fp_2823c.json`; its tail 3098-3105 held with
  the mouth drawn, 3106-3109 dropped). mrgo writes `internal/claude/2026-10-15_mrgo_closeout.md` at its end; its
  scripts and sheets are `internal/claude/2026-10-15_mrgo_scripts/`. Whatever did not land, its note says why.
- **Servers:** ComfyUI was restarted by the lead at 21:52:56Z (pid 3933887; mrgo's brief names the old 2104361, which
  is gone) and belongs to the lead's session; the review server is the lead's (pid 22533), started from the repo root
  with `python3 data/fun/poc/serve.py` (port 8765). Both are stopped at the close unless the card is still sampling, in which case
  the card seat stops ComfyUI when it finishes, and says so in its notes.
- **The owner reviewed nothing after about 22:15Z.** Nothing rendered tonight is judged.

## What not to assume

- **The day's close-out notes predate the evening's rulings; decisions.md wins.** The kitchen load runs at context 90
  with noise 0.8 (not 141, not waiting on `--context`); 2823-3109 is remade in one window (not continued from 2823b);
  the written label is parked (decisions.md's later entry over its earlier "still built and run").
- **Continuing a piece is allowed only from one the owner KEPT** (their choice for 2823b, then impossible on its grid);
  it is never a way to split a shot into fresh loads. The dad's 1397-1418 continues 1310c only if 1310c's stored window
  is on the chained grid; otherwise the owner is asked.

- **One load a subject over the whole shot.** Never fresh loads split inside a shot; chained windows inside one load
  only, on `h3_rules.chain_lengths()`. A load up to `h3_rules.MAX_LENGTH` (362) is one window.
- **The recipe is "A shot, made whole"** (`masked_v2v.md`). Pieces the owner kept are not remade for a better form;
  their explicit calls are (the kitchen, 2823-3109).
- **A text never describes the frozen picture, and says "sings" only where the person sings.** The generator
  (`bench/load_text.py`) writes today's form; a two-subject text is hand-written and the lane is closed.
- **Two subjects in one masked pass is closed** (`../roadmap.md`, "Closed lanes"); multi-person H3 generation is not.
- **A stored window off the chained grid cannot be continued** (2823b's is 277 frames). Check a parent's window before
  offering a continuation.
- **`--no-face-on` blanks the reference only; it does not empty the part** (mrgo's queue).
- **A reviewer's frame list or citation is a pointer, never data.** Today the reviewer's eye spans and four of its
  quoted rules were wrong; read the frames and grep the quote.
- **The owner's eye on playback is the only acceptance**, and a problem the owner has not seen is not theirs to decide
  (the lead's hair, dropped).

## Tomorrow, in order

1. **THE OWNER REVIEWS FIRST THING; they asked for it at the close ("make sure its set for a new session for me to look
   at tomorrow").** Before anything else: check what landed overnight (the card seat's notes, the newest ledger), then
   put each at the top of the review page (`http://100.87.141.65:8765/` reads `data/fun/review/items.json` in its
   order; the landing's page step posts the item; Start here on `TO_REVIEW.html` is
   `internal/claude/2026-10-15_mrtall_scripts/start_here.json`, then `pages.sh STAMP PREVIOUS_TABLE.json`) and give the
   owner the direct links, in this order: (a) the second subject's whole kitchen shot 604-1009, with what to watch (her
   identity across the join at 898; her turn and bow, 945-1009; her eyes facing the camera on 950-970, where the
   reference's contrast eye is drawn at its largest: the owner asked whether a solid black iris would look weird); (b) lead_fp_2823c beside the kept 2823b as one clip
   (`bench/stack_rows.py`, rows: the masked input, 2823c, 2823b), keep one; its note lists what differs from 2823b.
   Show the kitchen as a joined clip of 604-1009 with the new piece laid alone for her (a candidate, as the hall was:
   add a `604-1009:second=<the new load>` line to `prefers.txt` for the clip only, build with `joined_clip.sh`, then put
   `prefers.txt` back). `prefers.txt` still lays the old `second_hfp_0604d` there until the owner keeps the new one;
   only then does its line change. If either did not land, say why in one line. The Open List's "Right now" lines say the same to the owner.
2. The kitchen's lead: `--no-face-on` emptying the part on face-only loads first; then one hole a pass (a load given the
   second subject's render as its video, the assembler judging it against that base); then the lead over 613-974, one
   window.
3. The rest of `next_steps.md`, "The queue of 2026-10-15": the hall's far-table mother, 1093-1133 hair alone, 4310b,
   5935-6104 (why its reference failed first), the 2050 shot to about 2247, `--continue-from` after a fresh cold read
   (the dad over 1397-1418 only if 1310c's window is on the grid), then the gaps and the shots after 3166.
4. A clean-up job on the wiki: `internal/claude/2026-10-15_wiki_audit.md`.
4b. The gate's two rules that read a load's part BEFORE the person's word (`part_empty` read before word=/drop=;
   `region_on_a_frame_with_no_mask` not reading named-empty frames) are REPLACED by one rule (the owner, 2026-10-15
   late: "lets kill this rule is there some other information we can give it thats useful in its place"): one table a
   frame, read AFTER every word is applied (does the load generate her region; does the reference draw her face; who
   decided, measurement or a person's word with its reason), and one rule: the two agree on every frame. Region yes and
   face no is a hole with nothing to follow (tonight's `--no-face-on` fault); region no and face yes is a flip to the
   original (the 761 kind). Agreement is no flag, whoever decided it. HOW IT IS BUILT (the owner): the two old rules are
   REMOVED and this one takes their place; no third rule beside them, and no new rules added in its comments. The
   rule's comment carries why it exists: the owner's words above; the two old rules crying wolf on 2823d (3099-3101
   held, 3106-3107 dropped on purpose), so overrides were becoming routine; and the two faults it is there to catch,
   the hole with nothing to follow (the hall lead, 2026-10-15) and the flip to the original (761, 2026-10-15). The
   owner and the lead decided the replacement together; changing it again is the same kind of decision.
   BUILT, NOT LANDED (mrnight, 2026-10-16 00:23Z): branch `mrnight-fixes`, e72e85af + 6ffd21ac, with --no-face-on
   emptying her face (item 4 above, done the same way). On copies of 10 real captures it caught 0751b's 761 and the hall
   lead's turned cheek, and stayed quiet on 2823d's deliberate frames; it does not land until the face region stops
   counting a NECK seen from behind (third_fp_1419d 1431-1447; lead 1420, 1422). State, the proposed chin cut (untested)
   and the next commands: `internal/claude/2026-10-15_mrnight_fixes_note.md`, "STATE FOR TOMORROW"; the comparison:
   the lead's scratch `gate_compare/README.md` (copied to `internal/claude/2026-10-15_gate_compare/`).
4c. A REFERENCE IS NEVER MOVED OR RENAMED ONCE A RENDER HAS READ IT (the owner, 2026-10-16: "this needs to be fixed";
   the cache check, `internal/claude/2026-10-15_cache_check/README.md`). `head_reference` already refuses to write over
   files; the hole is the hand set-aside (`<span>_SUPERSEDED_<why>`) that frees the span's name for a remake, so a job
   record then points at a file it never read: 0751b, 1097b, 1156, 5114d. The fix: a remake takes the NEXT free name
   (`<subject>_<first>_<last>_b`, ...) unless `--existing`; the job keeps the reference's sha256 (it does since 631a9206)
   and verify fails where the file at that path no longer has it. Where the four went: 5114d's recorded sha256
   (f71c9fc07d25a020) matches `second_5114_5129_SUPERSEDED_5114d_open_on_closed/` byte for byte; 0751b, 1097b and 1156
   predate the digests, and their set-aside copies (`..._SUPERSEDED_751b_flicker_761`, `..._SUPERSEDED_before_defaults`,
   `..._SUPERSEDED_unheld`) are dated minutes before each render, likely but unproven. Of the four only 5114d is laid
   (table t_fun_0000_6394_2026-10-15T2218Z). Any reading that paired 1097b's render with the reference now at the span's
   path (the gaze readings of 10/14) used the wrong file and is re-read against the set-aside copy.
4e. ONE RULE IN PLACE OF THE HAND-WRITTEN `--hold-part` SPANS (the lead's proposal from the flip survey of 2026-10-16,
   `internal/claude/2026-10-15_flip_survey/README.md`, 394 captures, 306 candidate flips): every flip on record is the
   part model's output being empty, collapsed or under the floor, and then our floor or grade emptying her part, while
   SAM 3.1 and the face points still had her (SAM lost her on 13 of 306, the body model on 10, none in the cut). The
   rule: where SAM's track and the face points both have her, the floor never empties the frame; her part is held from
   the neighbours, as `--hold-part` does by hand. Its test set is the survey's 126 part-model frames (`flip_table.csv`);
   built in a worktree, a case seen red, a second reader, not landed while a load runs. With it: the ledger counts
   frames the floor empties as left as the original (`render_ledger.left_by_the_load` reads only `drop=`; it misses 15
   frames of the kept hall cut, 1419d and third 1419d). Not seen by the survey: a face regenerated to look like the
   original under a present mask (a per-frame face likeness would be needed).
4d. The chain records ComfyUI's list of cached nodes for every run (the owner's yes, 2026-10-16), beside the run's record.
5. Tooling queued for the card seat, in order after the kitchen's lead (`--mouth-on` and `track_from` are tonight's,
   above, and landed or not by mrgo's note): the ledger
   counting every frame a piece leaves as the original, not only `drop=` frames; the assembler protecting an earlier
   row by what it changed (for one hole a pass); `--hold-part` carrying the restore classes (Eyeglass, hands, Apparel@head) across its span and printing what
   else the class map leaves empty there (`masked_v2v.md`, "A frame the part model misses"); `load_text` writing "sings" span-wise inside one load instead of refusing
   a voice change; whether the Masked Prompt node's voice and frozen-picture wording still renders anywhere.
6. The outside reviewer's C5 (provenance, `internal/gemini/2026-10-15_provenance_c5.md`): two faults fixed at the close
   (today's verdicts taken into `data/fun/verdicts/verdicts.jsonl`; the lead's rebuild now reads the newest table);
   the ledger's "left as the original" is item 5; the kitchen's old pieces stand as placeholders until her whole-shot
   load replaces them.

## Where things are

- Today's rulings: `decisions.md`, 2026-10-15. The postmortem:
  `../research/postmortems/2026-10-15_span_fun-kitchen-and-whole-shot.md`.
- The close-out notes of the day's seats: `internal/claude/2026-10-15_{mrarc,mrlol,mrvid}_closeout.md`; in-flight
  patches copied out of temp folders: `internal/claude/2026-10-15_handoff/`.
- Worktrees, in the worktrees folder beside ComfyUI's `custom_nodes` (its full path is in the close-out notes): `mrlol-no-face-on` (`--continue-from`, 71ee3c8c, cold read
  owed), `mrlol-label` (parked), `mrvid-hold` (the hold form, 97512bda, cold read owed), `mrgo-mouth-on` (landed
  tonight or not: mrgo's note says); `mrgo-context-noise` is landed (3e5ea8f1).
- The kitchen's facts: `internal/claude/2026-10-15_mrarc_scripts/kitchen_plan_facts.md`; the reviewer's C4:
  `internal/gemini/2026-10-15_kitchen_approach_c4_mrgem.md`.
- The Open List: `https://claude.ai/artifact/7Jcf7G7jeB9v9kdqRHs1Fm` (today's rows r15_01 to r15_08, all answered or
  closed).
- The lead's scripts (rebuild, pages, joined clips, prefers): `internal/claude/2026-10-15_mrtall_scripts/`.
