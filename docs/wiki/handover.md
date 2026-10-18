# Handover: read this before anything else

**Rewritten at the end of each working day by the Claude lead; the day before is in git.** Written for 2026-10-17 by
`mrtabby` (the Claude lead) at 00:35Z on 2026-10-18, and completed at the close (01:45Z) by `mrgemsol` (the Gemini lead), who does not return on 2026-10-18: tomorrow's
Gemini lead is a new session, starting from this page and the mailbox.
Where this page and a record disagree, the record is right (`decisions.md`, newest first; `data/fun/laid.jsonl`; the
newest table and gap check; `git log`). The mailbox is `internal/gemini/` (`PROTOCOL.md`;
`2026-10-17_claude_to_gemini.md` C1-C146, `2026-10-17_gemini_to_claude.md` G1-G62). Updated by the Claude lead at 01:55Z
on 2026-10-18 for two builds started after the close (below).

## FIRST TOMORROW: the gaps, why they are there, and filling them in song order

The owner wants every gap filled, in song order, as soon as tomorrow starts, so nobody has to remember them again. The
one list is generated, never typed: `bench/frame_coverage.py` (made by `bench/rebuild_delivery.py --video fun`; the
close's is `data/2026-10-18_fun_delivery_tables/frame_coverage_fun_0000_6394_2026-10-18T0141Z.md`), then
`bench/shot_coverage.py` for what each shot needs. Two kinds, and why each exists:

1. **Shots with a piece where the original still shows while she is tracked** (the check's first table; 12 shots,
   438 frames). In song order: 294-414, 438-483, 604-1009, 1010-1038, 1093-1133, 1134-1155, 1419-1465, 1466-1729,
   4647-4725, 5114-5129, 5461-5497, 5533-5564. WHY, three causes the check separates: (a) the pieces were cut short
   round frames where her face mask (Sapiens, frame by frame, no memory) collapsed, so stretches between kept pieces were
   never rendered (the kitchen; the hall's 1306s stopping at 1396); (b) frames emptied INSIDE a kept piece by the holds,
   floors and drops (68 frames; the check's section 3 gives each cause: 294-414, 438-483, 1134-1155, 1419-1465);
   (c) no piece made or kept for one subject of a shot (4647-4725, 5114-5129, 5461-5497, 5533-5564, 1093-1133).
2. **Shots nobody has captured** (the check's second table; 49 shots, 3471 subject-frames, 2050-2622 the largest at 573
   frames; nothing is laid after 5564). WHY: no capture was ever made of them; the work went shot by shot in song order
   and stopped, and today's planned captures did not run (the Claude lead's miss). Each needs a capture before a load.

LANDED after the close (`54eb32a3`, gem-memsel; C129, C131): the gap check's ladder and the counts above each player.
Per frame and person it names the furthest step: no capture; captured, not rendered (an older capture holds her
track); rendered, not kept (the clips and windows); kept and laid. On the 0141Z table its steps add to the old totals
exactly (438 while tracked = 191 + 179 + 68; 3,471 not known = 2,703 + 542 + 163 + 63), and the kitchen's gaps for the
lead read "rendered, not kept" with the 0613 renders named. REBUILT at 04:05Z (the owner's word,
`data/2026-10-18_fun_delivery_tables/rebuild_fun_2026-10-18T0405Z.json`; the same totals, the pages with their gap
counts): tonight's candidates can go on Start here with their gaps counted (C131). NOT TESTED: the pages, which nobody
has opened yet. The memory selection in the node was checked through it by gem-memsel (W7): 710 on and 715 off as its
W4, and with the chain's default (the range corrected, memory on) her face is back at 709, the owner's look.

And one more before the kitchen load (C136, the owner's yes): the head tube's turned-away label only proposes (it calls
709, 710, 712 and 735-737 turned away where the owner sees the side of her face: a profile lumped with the back of her
head, C138), the holds never fill a frame a person says
has no face, and a person's drop is never carried silently: the job step
refuses to carry it onto frames where the capture sees her face and lists them for a look. The face lines follow the same
rule (C140): carried only as a person's word, refused where the capture sees her face. 0613h's own words, from the
owner's look tonight: drop 687-708, 747-750 and 974 only, the face lines off on those frames only and ON on 709-746
(the side of her face, one eye); a Gemini reader checks the reference's drawn lines on 705-715 and 730-752 first. gem-matte's handoff (end of
`internal/gemini/2026-10-17_worker_gem-matte.md`) has the three-state label's design and a one-shot card test for the
first free slot: Sapiens re-run on the steady shot box over 680-760, against the preview's saved classes on 709-712 and
735-737.

Tonight renders none of the gaps (the owner's word at the close, C126): the kitchen and hall wait for the head tube's
gate. The two that are next, from the same check and tonight's `lead_fp_0613g`:

| shot | who | the original in the cut | the window to load | note |
|---|---|---|---|---|
| kitchen 604-1009 | the lead | 606-629, 681-750, 779-804, 834-851, 966-974 | 613-974, one window of 362 (`lead_fp_0613h`, head tube on) | 0613g (candidate) covers it today but leaves 687-710, 733-738, 747-750 and 974 as the original; her face shows on 709-712 and 733-746 (owner's looks, C132-C135) |
| kitchen 604-1009 | the lead | 604-612 | not planned: inside no window | 9 frames; decide whether 0613h starts at 604 (604-965 is 362 frames and ends where the kept 0852b ends) |
| kitchen 604-1009 | the lead | 975-1009 | none | her face is out of the picture 975-1009 (`next_steps.md`, the 10-15 queue); check on the frames before loading |
| hall 1306-1418 | the lead | 1397-1418 | 1397-1418, 22 frames, context from the kept 1306s | 1306s's own render ends at 1396 |
| hall 1306-1418 | the dad | 1397-1418 | 1397-1418, 22 frames, context from the kept 1310c | the 10-13 look that left it out was overruled 10-16 |

**To run one gap end to end** (`docs/wiki/runbook.md`, "(b) One load, from preview to land", is the full recipe):
1. Only once the head tube is landed and its gate passed: `bench/shot_to_load.py preview`, then `job` with the head
   tube's switch on (its name is in gem-matte's READY) and read THE DEFAULTS THIS LOAD CARRIES; `reference`; `text`.
2. `bench/load_chain.py first`, `plan`, `on-plan`; a second session's look at the plan strip (`capture_masked_run.py
   looked`); the face masks on the window's hard frames read against the gate's numbers.
3. The prompt read in FULL from the queued graph: `bench/what_h3_got.py --stage plan`.
4. `bench/load_chain.py render`, then `land` (read back, checked against the plan, pose pass, record, stacked clip).
5. `bench/stack_delivery.py compare` against the current cut; `bench/rebuild_delivery.py --video fun`; Start here.
6. Laid only on the owner's keep: a line in `data/fun/laid.jsonl` (`bench/laid.py`), then `rebuild_delivery` again.

## State when closed

- **Two leads** since the morning (`AGENTS.md`): mrgemsol (Gemini) holds the card, git on main, the ledger and the
  owner's pages; mrtabby (Claude) holds the Open List, the records and every clearance. Workers: gem-matte (perception
  builds) and gem-memsel (started 00:29Z; SAM memory selection, a measurement only).
- **Landed today** (`git log 483d5b38..HEAD`): one compositing law (`assemble_delivery.lay_plane`) and mattes for all 43
  laid pieces; `bench/recomposite_piece.py`; `bench/rebuild_delivery.py`, one command for the ledger, table, gap check
  and pages; the who view; the floor holds and off-head pruning (D22b, `9f139d0c`); the mouth fixes behind
  `--closed-lip-line`, off by default (`c796c025`); the hole label wording for `original_tracking_not_known` by cause
  (`b1be216b`); the `person_track`-bounded SAM 3.1 head tube behind `--head-tube`, off by default (`c4b395bf`); the
  vendored HyDen MoGe-2 depth/intrinsics loader under `vendor/meta_moge2/` and `bench/measure_moge2_depth.py`
  (`4725ff5c`); the runtime SAM 3 / 3.1 memory-selection tool `bench/sam_memory_selection.py` (`f390a25a`); the rule that
  a change names everything that writes and reads it (`AGENTS.md`); the folder `AGENTS.md` files; `meta_sam3` removed;
  `data/` untracked but for its two instruction files.
- **The cut** is unchanged since 1747Z: 43 laid pieces (`data/fun/laid.jsonl` verified 43/43 at `0141Z`). All four of
  tonight's candidate renders completed, landed, were compared against the `1747Z` cut with `stack_delivery.py compare`,
  and were read natively on playback (`G56`, `G58`, `G59`, `G60`), none laid: `second_hfp_0604h` (the mom's kitchen
  `604-1009`), `lead_fp_0613g` (the lead's `613-974` in one window), `second_hfp_0278d` (`278-293`), and `lead_fp_0438d`
  (`438-483`, control). Per C131, candidate clips whose ranges contain a gap for a person who shows wait for tomorrow's
  player gap-count banner before going on Start here (listed with their exact gap spans in `G62`).
- **After the close** (the owner's word: mrgemsol does not return tomorrow, so the Claude lead landed what was cleared):
  - LANDED `13d9639f`: `bench/sam_memory_selection.py`'s three `fun` scenes read each capture's `subjects/<subj>/` masks
    (bit-packed), from captures NAMED explicitly, never a blocked or older one (C146, C147: kitchen
    `data/2026-10-17_lead_fp_0613g/`; the hall `data/2026-10-13_lead_fp_1306s/` and `data/2026-10-13_third_fp_1310c/`;
    1419 `data/2026-10-15_lead_fp_1419d/`; `data/2026-10-12_lead_fp_1306_span_1306_1418_blocked` refused), each masks
    file's sha256 recorded. RUN after the close (gem-memsel, `internal/gemini/2026-10-17_worker_gem-memsel.md` W4;
    `data/2026-10-17_sam_memory_selection/summary__*.json`): on the person tracks (the kitchen body, the hall, 1419)
    memory selection changes nothing or one frame. On the lead's FACE tracked alone over 680-760 it does: SAM 3.1 off
    is back at 715, on at 710 (the owner's look: the side of her face from 709); SAM 3 base off is empty 691-751, on
    is back at 702, which the owner's look calls hair until 709, so "back" there is a mask, not yet checked against the
    picture. The owner's word: SAM 3.1's memory selection goes ON in the chain
    (decisions.md, 2026-10-17). LANDED `35706971`: `MiniMaxH3SAM31Corrections` input `memory_selection` (optional; on
    by default, absent means on), for one-object tracks on SAM 3.1 only (several objects and SAM 3 base keep ComfyUI's
    memory); a load turns it off with `--no-memory-selection` and a reason, and THE DEFAULTS name it. TESTED: the node's
    22 cases, the on-card check on ModelPatcherDynamic, the server queue test with an H3 render (on a scratch copy at
    port 8191, stopped after), the face over 680-760 of 0613g through the node (710 on, 715 off; 709 and 714 with the
    range corrected), three blind reads (two Claude, one Gemini: `internal/gemini/2026-10-18_review_memsel_node.md`).
    NOT TESTED: several objects or a mid-video detection on the real model (CPU stand-in only), a real load through
    the chain. SAM 3 base is PARKED, not owed (the owner, 2026-10-18): the chain runs 3.1, the node leaves base alone,
    and on the face test base was worse off (61 frames lost) and early on (back at 702, on her hair); its converted
    checkpoint stays on disk for a comparison only. Captures made from now carry it; kept pieces are not remade for it.
  - LANDED `a15463e9`: the three-state face label (facing; side with an eye; back of the head, which only proposes) and
    drops and face lines as a person's word: a carried `drop` or `--no-face-on` is refused on frames where the capture
    sees her face; the holds never fill a back-of-head frame unless a person's word holds it (C136-C140,
    `internal/gemini/2026-10-18_brief_build_faces.md`). On 0613g's 680-760 its labels match the owner's looks exactly
    (back of the head 687-708 and 747-750; side with an eye 709-712 and 733-746), and the refusal flags exactly 709, 710
    and 733-738 of 0613g's old drop. TESTED: its checks (9, 52, 31 cases) on main; a blind read
    (`internal/gemini/2026-10-18_review_gemmatte_faces.md`). NOT TESTED: any shot but 0613g's kitchen, whose frames its
    lines were set on; a real load through the chain.
  The four branches of the evening (`gemini-hole-label`, `gemmatte-masks`, `gemmatte-moge`, `gem-memsel-tool`) are
  landed and removed.
- **The close (C126, C143)**: `bench/rebuild_delivery.py --video fun` ran at `2026-10-18T0141Z`
  (`data/2026-10-18_fun_delivery_tables/rebuild_fun_2026-10-18T0141Z.json` and
  `frame_coverage_fun_0000_6394_2026-10-18T0141Z.md`; identical gap counts to `1747Z`, with the `b1be216b` hole label
  applied). `bench/sam_memory_selection.py` converted `sam3_fp32.safetensors` on CPU and ran all 4 arms on `lotsofpeople`
  (`data/2026-10-17_sam_memory_selection/summary__lotsofpeople.json`: in the 16-dancer `sam31` `group` call, `mem_on`
  holds the `most_central` dancer across frames `95-116` and `130-140` at `7,006-11,914 px` where `mem_off` drops to
  `0 px`; the three `fun` scenes need the `subjects/<subj>/` subpath in `build_default_scenes()` first thing tomorrow).
  ComfyUI (port 8188) and the review server (`data/fun/poc/serve.py`, port 8765) are both stopped (`ss -ltnp` verified
  free). Start the review server again (`docs/wiki/runbook.md`) before pointing the owner at a page.
- **Open List**: r17_00 is FIRST TOMORROW (the gaps, why, and the next loads); r17_06 (the mom's kitchen re-lays) and
  r17_07 (the perception order) wait on the owner; r17_08 is tonight's state; r17_09 records the licence answer. The
  inbox is clear.

## What not to assume

- **"Tracking not known" on the stacked view usually means no piece was made there**, not that a track was lost (1402,
  the owner's screenshot): the lead's 1306s ends at 1396, so 1397-1418 is the original for her and the dad. Until the
  label lands, read the gap check's cause column.
- **The kitchen's holes are a face-mask failure, not a tracker failure.** The person tracker held the lead on 399 of 406
  frames of 604-1009; the face mask (Sapiens, frame by frame) collapses on 680-712, 733-751, 779-793, 800-803 and
  834-845, and the old kitchen loads were cut round those frames. 0613g leaves 687-710, 733-738, 747-750 and 974 as the original
  because its job CARRIES `drop=` on them, a 10-16 reading ("only her hair shows") carried since 0613b; tonight's read
  says her face shows on 708-712 and 736-738. The owner's eye on 700-760 decides before 0613h (C130).
- **The head tube is not a fill, and in the capture it does not re-run Sapiens**: it smooths the preview's saved class
  map (gem-matte, C137), so the steady-box gain is not there yet. Also not a fill: Its 3-frame average bridges 1-2 frames; its frame labels change nothing; its
  turned-away lines were measured on 684-707 of the same shot (`internal/gemini/2026-10-17_head_tube_how_it_works.md`).
- **Nothing gem-matte built is a default.** Each switch is off until its tests pass and the owner says yes.
- **SAM**: core has no memory selection under any name; Meta has it on for SAM 3 base and off for 3.1 with no stated
  reason; its score is a mean over a whole group, so off is plausible for 3.1's 16-object use, and our chain tracks one
  person a call. "Patch it" means a run-time patch like sage, Sol and `sam31_corrections.py`: respect dynamic memory,
  no odd clones. SAM 3 base is converted (`models/checkpoints/sam3_fp32.safetensors`, exact readback parity) and tested
  beside 3.1 by `bench/sam_memory_selection.py` (`f390a25a`).
- **MoGe-2's camera is unreliable on tight close-ups** (field of view spread 46 degrees on 3060-3109, 1.5 on 755-780).
- **The 48 untouched shots were not captured today**; that was the Claude lead's miss (r16n_03).
- **Times**: write them from `date -u`; three of the Claude lead's estimates tonight ran ahead and were corrected.

## Tomorrow, in order

1. Start ComfyUI and the review server (`data/fun/poc/serve.py`, `docs/wiki/runbook.md`). Read the night's mailbox
   (`G62`, "End of day", and `C118-C144`). First builds before filling gaps:
   - the 4-step gap ladder (`C129`) + player gap-count banner (`C131`), so candidate clips with gaps can go on Start here
     with their per-person, per-kind counts above the player;
   - the `C136`/`C137`/`C140` drop + `--no-face-on` + three-state face-visibility build before `lead_fp_0613h` (`drop` and
     `--no-face-on` on `687-708, 747-750, 974` only, face lines ON on `709-746`, plus the one-shot steady-box Sapiens
     card test on `680-760` from `gem-matte`'s handoff).
2. The owner's looks, Start here (once the player gap-count banner lands, C131): 0613g's compare clip at 680-760 first,
   then 0604h, 0278d and the 0438d control. Then r17_06 and r17_07.
3. Finish `gem-memsel`'s remaining 3 `fun` scene measurements (`kitchen_604_1009`, `hall_1306_1418`, `regain_1419_1465`
   after pointing `build_default_scenes()` in `bench/sam_memory_selection.py` at `subjects/<subj>/masks__*.npz`;
   `lotsofpeople` already ran and held `most_central` across `95-116` and `130-140` in the 16-dancer `sam31` `group` call).
   If memory selection holds her better on `fun` too, wire it as a switch in `sam31_corrections.py`'s form. Then the
   tracker review's item 10 changes, `frames_on_others` first (`internal/gemini/2026-10-17_review_subject_track_regain.md`).
4. Fill the gaps in song order (`data/2026-10-18_fun_delivery_tables/frame_coverage_fun_0000_6394_2026-10-18T0141Z.md`),
   and capture the 48 untouched shots in song order in the card's free time.
5. Run the end-to-end tests (`internal/gemini/2026-10-17_test_scenes_agreed.md`, `2026-10-17_e2e_tests_gem-matte.md`)
   through the whole chain before any perception switch goes on.
6. DONE after the close: the 2026-10-17 changelog entry (`a0ae7025`, 0.290.0), from the Gemini lead's draft and the
   evening's commits.
   Parked, no default: the matting composite (`internal/gemini/2026-10-17_brief_build_matting_relay.md`, on hold).

## Where things are

- **Each session's end-of-day notes**, read before starting:
  - the Claude lead (mrtabby): this page; working notes in `internal/claude/2026-10-17_mrtabby/` (`mentor_notes.md`,
    `test_scenes_claude.md`, `who_record_v2.md`);
  - the Gemini lead (mrgemsol): its closing entry `G62`, the last G entry of `internal/gemini/2026-10-17_gemini_to_claude.md`,
    headed "End of day";
  - gem-matte: "End-of-Day Handoff" at the end of `internal/gemini/2026-10-17_worker_gem-matte.md`;
  - gem-memsel: "End-of-Day Handoff" at the end of `internal/gemini/2026-10-17_worker_gem-memsel.md` (and
    `data/2026-10-17_sam_memory_selection/summary__lotsofpeople.json`);
  - the readers' reviews: `internal/gemini/2026-10-17_review_*.md`.
- **Made tonight, to read with the candidates**: the Gemini lead's playback readings,
  `data/2026-10-17_<load>/playback_reading_mrgemsol.md` for `second_hfp_0604h`, `lead_fp_0613g`, `second_hfp_0278d` and
  `lead_fp_0438d` (a reader's watch, not the owner's keep); the clips under `Video/fun/review_clips/`
  (`<load>_<first>_<last>_stacked.mp4` and `_compare.mp4`, and `lead_fp_0613g_0680_0760_compare.mp4`, the clip the
  owner's looks on 708-738 were made on); the memory measurement `data/2026-10-17_sam_memory_selection/` and its review
  `internal/gemini/2026-10-17_review_gem_memsel.md`; the head tube's write-up
  `internal/gemini/2026-10-17_head_tube_how_it_works.md` (corrected by C137: it smooths the preview's saved class map).
- Decisions: `docs/wiki/decisions.md`, 2026-10-17. The owner's preferences: `docs/wiki/working_with_the_owner.md`.
  Day-to-day commands: `docs/wiki/runbook.md`. The lane: `data/AGENTS.md`, then `data/fun/AGENTS.md` (local).
- The owner's raw answers: `data/fun/owner_answers.jsonl`. Who is who: `data/fun/shots.json` and `cast/cast.jsonl`.
- Pages: `Video/fun/TO_REVIEW.html` and `PROVENANCE.html` on the share, from `bench/rebuild_delivery.py` (`0141Z`); the
  review server `data/fun/poc/serve.py` and ComfyUI are stopped at the close (`docs/wiki/runbook.md` to start).
- The Claude lead's notes: `internal/claude/2026-10-17_mrtabby/` (mentor notes, test scenes, who record).
- Briefs and reviews of the night: `internal/gemini/2026-10-17_brief_*.md` and `2026-10-17_review_*.md`.
