# Handover: read this before anything else

**Rewritten at the end of each working day by the session that led it; the
day before is in git.** This one was written on the night of 2026-10-12 by
`mrfancy`, the lead from 21:46Z (after `mrforest`), with `mrsnow` on the
card from 21:57Z and `mrham` on tooling until it handed off at 22:20Z. Every
time here is read from the clock or a file; where this page and a record
disagree, the record is right (`decisions.md`, the day's section; the
readings under `data/2026-10-12_<load>/reading_*.md`; the Open List).

## State when closed (2026-10-12, about 23:10Z by the clock; the exact time is in the commit)

- **The owner's standing goes**: finish the video with the tools we have
  (20:13Z); her face and hair on every shot of hers (row z10); keep pushing,
  the rough edges later (22:00Z); the whole of `fun.mp4` to its end is the
  aim (22:01Z). The owner left at 22:13Z; nothing waits on them. Two rows
  wait for their return and block nothing: 18 (14-154: `headref` or `p4` in
  the end) and 19 (their typed words on the 1920-6394 shots: nine lines
  unclear).
- **The form** (`AGENTS.md`, "Settled"): the body model cut to the head
  with the face points drawn, and a text that says what that video is.
  Four of her shots and five of the lead's are rendered in it today.
  What the four of hers say, each read by the lead and a blind reader
  (the readings' part 3): the drawn outline follows her eyes IN TIME at
  the start of a window or in a short one (14-80; 278-289; the opening,
  gaze and lowering on 415-437, 0.52 of the original's fall where the
  marks carried 0.08) and opens them at the camera late in a long window
  (81-154; 290-293); the head's place shifts with the crop on every shot
  but 14-154 (14 to 22 px); the mouth is nearer the original's than the
  marks' renders. The owner's eye: 14-154 "looks good" (21:58Z).
- **The hair band** (`hair_band` 0.6 on her face-and-hair loads): its
  first three renders (278, 415, 484) drew nothing visible in the band;
  her hair never reached it (a bob on 278, to the shoulders on 415 in
  both forms). Carried, untested as a fix for the 693-703 fault.
  `bench/read_hair_band.py` gives its figures (mrham, 65feaa34, 2101a1d1).
- **Pieces that stand tonight** and where two passes meet:
  `internal/claude/2026-10-12_delivery_dry_run.md` (mrham, 22:22Z; read
  its five things before the assembly).
- **The night's joined clip**, the owner's ask: `Video/fun/fun_0000_1919_joined_2026-10-12.mp4`
  on the share (1920 frames at the source's rate and size, the source's
  sound; its `.txt` beside it says which piece is on each stretch; its
  `.check.json` passes; the table `data/2026-10-12_fun_delivery_tables/`;
  the coverage table `data/fun/coverage_2026-10-12_joined.txt`). Nobody has
  watched it on playback. In it: her four shots and 604-1060 in the one
  form, the lead's hall shots, 1466-1534 and the close-up, the 10th's
  pieces elsewhere (the third subject as whole-person renders), the held
  shots as the original.
- **The servers**: BOTH STOPPED by mrsnow at the owner's word (ComfyUI 8188
  killed by pid 23:24:18Z, the port free 23:24:24Z; the page 8765 killed
  23:24:49Z, free 23:24:52Z). The morning's card holder starts ComfyUI by
  the recipe (`docs/comfy_notes.md`) and the page with
  `python3 data/fun/poc/serve.py` from the repo's root, detached
  (`data/fun/poc/README.md`), BEFORE the owner's looks; the page's items
  need nothing rebuilt.
- **The tree**: main at the commit that carries this page; nothing pushed; `data/`, `internal/` and the share hold the night's records, untracked. The changelog fragment for the day:
  `changelog.d/day-2026-10-12.md`, built and committed with this page by the README's steps.

## What not to assume

- That the drawn form holds the lids down through a long window: on two
  shots it opened her eyes at the camera after about 80 frames (14-154)
  and after 12 (278, a 16-frame shot with a 141-frame window). The lid
  marks held the lids down on 14-154 and nowhere else. No form has done
  both on one shot; the lid marks on the crop (`1039_g`'s form) is the
  untested one.
- That the band does anything yet: it has only opened cells the model
  filled with the original's apron. Its test needs a render where her
  hair is drawn long enough to reach the keep-out.
- That the two-subject load renders: it is built (703b3816) and the
  capture reads it (b6e75c28), never rendered; its first render is the
  hall shot 1222-1305 with the third subject, after his inputs exist.
- That the 1920-6394 previews can seed on the owner's words alone: 19 of
  the 29 seed rows carry a frame the owner never saw (row 19, the caution
  at the foot of `data/fun/who_1920_words_read.md`).
- That `uv run` is safe here: it rewrote `uv.lock` tonight (restored); run
  every tool with the ComfyUI venv's python (`docs/comfy_notes.md`).

## Tomorrow, in order

1. The owner's looks on the local page (8765): the four cutaways and
   kitchen opening of hers, the lead's hall shots and 1466-1535; rows 18
   and 19 on the Open List.
2. The lead's 1061-1092 (the plain body model cut, ending at 1079) and 1306-1402 (the capture on the load's own frames, ending at 1396), the rulings in their captures' `looks.json`; then 168-185 and 516-603 after their part sizes are counted against the capture's floor.
3. The third subject's three shots (1222-1305 as the two-subject load,
   1306-1410, 1466-1729, face only): his track, body model and points
   first, then the loads in the form.
4. The delivery of 0-1919 from the dry run: a new table
   (`bench/assemble_delivery.py`), the `restore=` rows read again where
   two passes meet (the dry run's section E), the whole clip assembled,
   the coverage table run on it, the owner's look.
5. The previews on 1920-6394 (one tracker a subject on the copy
   `fun_1920_6395_24p.mov`), seeded by a builder rule on a sampled frame
   the record marks the subject on; the nine unclear shots last; then the
   loads shot by shot and the delivery of the whole clip.
6. The postmortem from `internal/claude/2026-10-12_postmortem_evidence.md`
   (candidates: the clock slips; the tree collision on `h3_config.py`; the
   1093 region fault; the sweep tool's first run; frames against playback
   on the hall shots; 703b3816's sixteen seconds; `uv.lock`).

## Where things are

- The sessions' notes and handoffs: `internal/claude/2026-10-12_mrfancy_notes.md`,
  `..._mrsnow_handoff.md` and notes, `..._mrham_handoff.md`,
  `..._mrforest_handoff.md`, `..._mrsound_handoff.md`, `..._mrprompt_handoff.md`.
- The card's order: `internal/claude/2026-10-12_card_order_rest_of_video.md`.
- Readings: `data/2026-10-12_<capture>/reading_mrfancy.md` with
  `reading_mrfancy_read.*` and `reading_blind/reading.md` for
  `gate_p4_0014_headref`, `second_hf_0278_b_hr_band`, `second_hf_0415_b_hr_band`,
  `second_hf_0484_b_hr_band`, `lead_fp_1466`; the lead's hall shots and close-up under mrforest's name.
- The owner's verdicts: `data/fun/verdicts/verdicts.jsonl`; the page items:
  `data/fun/review/items.json`; the page: `data/fun/poc/`.
- Renders: `Video/fun/mrdoe/<load>/` (her cutaways), `Video/mrfire/gate_p4_0014_headref/`,
  `Video/fun/mrhorn/` (the lead's hall shots), `Video/fun/mrsnow/lead_fp_1466/`; stacked clips
  `Video/fun/review_clips/`; strips `Video/fun/looks/`.
- The inputs made tonight: `Video/fun/mrsnow/` (references, previews, the
  lead's 1466 preview and head reference).
