# Decisions and reversals

Written by hand. One dated line per decision the owner made, or per claim in
the prose that was corrected: what changed, what it used to say, where it
lives now, and the commit. Newest first. `CLAUDE.md` and `VISION.md` carry no
history notes; this page is where those go. Another document may also keep a
dated note in place, where a reader would otherwise trust the stale text.

Older history lives elsewhere and is not copied here:

- [`CHANGELOG.md`](../../CHANGELOG.md): every change, by version.
- [`docs/rules_history.md`](../rules_history.md): `CLAUDE.md` as it stood
  before the 2026-09-03 cut, frozen.
- [`docs/roadmap.md`](../roadmap.md): "Closed lanes", and the dated "Owner
  decisions" and forward-plan sections.
- `bench/results/`: the verdict records, each with its conditions.

## 2026-10-17

- **Tonight: the hole label and the head tube, both** (the owner, about 23:55Z, "we should do both of them before the
  end of the night if u and mrgemsol agree"). (1) The gap check, stacked view and pages say "the original: no piece made
  for these frames" where they said "tracking not known" (the kind key kept), since a hole read as a lost track at 1402.
  (2) gem-matte's head tube (gemmatte-masks), bounded by the person tracker's mask and wired into the capture behind its
  own switch, off by default, run on the kitchen 613-974 and the hall 1306-1418. The gate is the masks before and after
  on the same frames, no render: only where the face holds through the turns are the holes rendered (1397-1418 for the
  lead and the dad, the kitchen's), as candidates. This replaces "we don't use any of this tonight" for this build only.

- **A build that changes data, an input or an output names every writer and reader of it** (the owner, about 23:15Z:
  "make sure if we're changing data or inputs or outputs whatever feeds that or receives that or both know"). Written
  into AGENTS.md beside the build rule and into the protocol's pre-flight: the READY lists each, with file:line and
  what each does with the change. The day's three side effects one step away are the reason.

- **The night's proposals of 2026-10-16, answered** (the owner, about 14:07Z, in the Claude lead's terminal; the
  Open List's rows r16n_01-08, r16_01 and r16_02). Recorded here late, at about 22:05Z: an independent design read this
  evening missed the keep-out change because only the Open List and `plan.json` held it.
  - **No default keep-outs for the second subject** (r16n_06, r16_02): `held`, `hands` and `clothes`, and with them her
    hair band, are retired from her loads (`data/fun/plan.json`, `subjects.second.keep_outs_removed_by`,
    `hair_band_removed_by`). They cut her neck and hair at 287, 422 and 483-489 (`held` reading her white apron as
    background near her lips and hands; `hands` taking her bare upper arms under her hair). A thing is kept as the
    original only where a look names it. The lead's `strangers` keep-out stays. Each shot of hers names her clothing
    from a look at its own frames.
  - **One compositing law** (r16n_01): built and landed as cecd8db1, after the re-composite tool saved a matte for
    every laid piece.
  - **Re-lay, not re-render, where the render is fine** (r16n_07, r16_01): her three unconfirmed kitchen pieces re-laid
    whole as candidates; the owner's other kept pieces are not re-laid; 278, 415 and 484 are rendered again under the
    keep-out change, since a re-lay cannot undo what the keep-outs cut.
  - **Captures of the untouched shots in song order** (r16n_03); **all four process rules** (r16n_04); **D19c landed**
    (r16n_05); **SAM 3.1 tested on the kitchen first** (r16n_02: it tracks the whole head, so it suits head-and-hair
    loads, and a face-only load still needs the parts split inside it); **1010j kept, 1010c dropped, a hair fill on
    1032-1038** (r16n_08).

- **Corrected: `coderef/metadepth` is checked out** (found by an independent Gemini reader, about 21:50Z).
  `docs/wiki/meta_perception_models.md` said Meta's metadepth code was not under `coderef/` and unread; it is there, and
  both HyDen checkpoints load with torch and torchvision alone. No node loads them yet.

- **Standing go: scenes that are bad or have gaps render without waiting for the owner's look at a plan strip**
  (the owner, about 21:05Z: "its better to just go and render scenes that are bad or have gaps"). Each still has its
  prompt read in full from the graph as queued, and THE DEFAULTS THIS LOAD CARRIES, by the session that queues it
  (AGENTS.md); each lands as a candidate, not laid, with its stacked view and a reader's playback notes; the owner
  judges them later, in batches. A kept piece is still not remade for a better form.

- **The Antigravity CLI evaluates folder rule files only during prompt expansion, not when a tool reads a file**
  (2026-10-17, about 20:40Z, settled by an independent Gemini reader). Its rules page's general section says rules load
  "whenever Antigravity reads or edits a file"; its CLI tab says "The Antigravity CLI evaluates workspace,
  directory-scoped, global, and plugin rules during prompt expansion". The reader read a file under `bench/` mid-turn
  and its rules still held only the root AGENTS.md. Claude Code does load a folder's `CLAUDE.md` on reading a file there
  (seen today). So the root AGENTS.md names every folder with its own file, to be read before working there. The
  Gemini lead's 15:16Z report took the general section's word.

- **A project's own state goes in `data/<video>/AGENTS.md`; `data/AGENTS.md` holds only the lane's rules, for any
  video** (an independent Gemini reader's point, about 20:10Z: a new video would otherwise inherit `fun`'s settled
  facts, such as who's-who being answered). `data/AGENTS.md` opens with a start-here table and a table of the
  projects in motion; `data/fun/AGENTS.md` is `fun`'s. The root's lane paragraph is shorter, the two-leads and build
  rules point at `internal/gemini/PROTOCOL.md` for the mechanics, and "never run uv in this repo" is now a root rule.

- **The masked lane and the project `fun` get their own instruction file, `data/AGENTS.md`; the root AGENTS.md
  keeps only what holds for the whole pack** (the owner, about 20:00Z). Moved, unchanged in substance: the lane's
  settled rules, the one-ledger rule (now naming `bench/rebuild_delivery.py`), the video rows of "Where state lives",
  handing a render over stacked, and Gemini's playback questions. Removed as superseded: "the owner takes the cheap
  looks first" (who is in each shot is answered; batching is `working_with_the_owner.md`). `data/` stays untracked
  except `data/AGENTS.md` and its one-line `CLAUDE.md` (`.gitignore`: `/data/*` with two exceptions).

- **No cap on parallel builds by subagents** (the owner, about 16:30Z, on the Claude lead's note that three were open):
  r16n_04's fourth rule, "at most two builds open", came from 10-16, when four builders queued behind one reader. A lead
  orchestrating its own subagents in parallel is not that case. What stays: each branch has its own blind read before
  it is sent, and the Claude lead clears in the order sent (`internal/gemini/PROTOCOL.md`, rule 4).

- **The same folder files for docs/, prompt_bank/, archive/, vendor/, vendor_config/, vendor_guides/ and
  standalone/** (the owner, about 15:30Z, "yes, write the nested instruction"). Each says only what holds in that
  folder (verbatim vendor files, retired code, generated catalogues, separate packages) and points at its source.
  `meta_sam3/` gets none: its README and its copy check (`bench/check_meta_sam3_copy.py`) already govern it.

- **A folder whose files need their own rules gets an `AGENTS.md`, and a one-line `CLAUDE.md` beside it that imports
  it (`@AGENTS.md`), as the root does** (the owner, about 15:25Z: no two copies). Neither harness reads an `INDEX.md`
  on its own: Antigravity loads `AGENTS.md`, `GEMINI.md` and `.agents/rules/` when it opens a file in a folder (the
  Gemini lead, from its docs); Claude Code loads `CLAUDE.md` the same way. First: `bench/` and `workflows/` (each
  points at its `INDEX.md` and holds only that folder's rules), and the import beside `contact_sheets/AGENTS.md`,
  which Claude sessions had not been loading.

- **AGENTS.md says the repo is the owner's whole H3 research repo and masked video-to-video one use of it** (the
  owner, about 15:15Z: so sessions neither read too much into lane rules, nor too little into common code, nor
  overwrite what is right for another use). The scope paragraph names the three ways to get it wrong; the lane's three
  settled rules (a masked load is not an edit, the singing line, the head reference) moved from "Settled about H3" to
  a section of their own, "Settled for the masked lane", unchanged.

- **The owner's working preferences are a tracked page both models read: `docs/wiki/working_with_the_owner.md`**
  (the owner, about 15:20Z, yes to the Claude lead's and the Gemini lead's same proposal). Until then they lived only
  in the Claude sessions' private memory, which Gemini sessions never see; the Gemini lead's audit of that memory
  (`internal/gemini/2026-10-17_memory_review.md`) found most of it stale against today's roles. Written without the
  owner's words; AGENTS.md points at it. The memory keeps the why behind each and points here.

- **The owner's page answers are a primary record: `data/fun/owner_answers.jsonl`, out of `poc/`** (the owner, about
  14:20Z: "like its a super important artifact now no?"; they answered who is in each shot for the whole video, 0-6394,
  and were about to be asked again). Moved whole (sha256 unchanged); `poc/answers.jsonl` is now a link to it so the
  running server and every citation by line still resolve; `serve.py` writes the new path; a read-only snapshot is
  taken at each close into `data/fun/owner_answers_snapshots/`. AGENTS.md "Where state lives" has its row. Before, it
  lived only in the review page's proof-of-concept folder, untracked, with no copy.

- **The Gemini lead runs git on `main`; the Claude lead runs none** (the owner, about 14:30Z: so the Gemini side learns
  the motions). The Claude lead still clears every landing (re-runs one check, reads the diff from the merge base) and
  hands its own files to the Gemini lead to commit by pathspec. Replaces this morning's "the Claude lead is the only one
  that commits to `main`". AGENTS.md and `internal/gemini/PROTOCOL.md`, "Git today".

- **No guidance names a frame count under 362 as a cap or a default; short windows are encouraged** (the owner,
  about 14:20Z: "Remove all references to that 345"; "always make sure there's guidance that the frames can be low too.
  More frames = quadratic attention"). Removed from AGENTS.md, `docs/evidence.md`, `docs/comfy_notes.md`,
  `docs/h3_resolutions.md`, `docs/h3_references.md`, `docs/roadmap.md`, `docs/h3_input_impacts.md`,
  `docs/h3_audio_freeze.md` and one stale line of `next_steps.md` (it said the default length was one grid step under
  362; `h3_config.LONG_LENGTH` is `MAX_LENGTH`). Kept: the measured records, the changelog, this file's history and
  the tables that list every grid length, since each says what was run or what exists. The chained window's default
  stays the longest length on both clocks, derived by `h3_rules.chain_lengths`, never typed (the owner's 3a).

- **A length whole on both the video and the audio clock is a best practice, not a rule** (the owner, about 14:15Z, on
  a community report relayed that morning). A one-window load takes any 17n+5 length up to `MAX_LENGTH`; only a seam
  between chained windows is held to both clocks, as before (`h3_rules.chain_lengths`). Written with the why in
  `docs/h3_resolutions.md`, "Whole on the audio clock".

- **Two leads, one each side: a Gemini lead co-leads and leads the Gemini sessions; the Claude lead is oversight and
  the method's expert** (the owner, in the Claude lead's terminal, about 13:45Z; the Claude usage cap is near). It
  replaces "two Claude sessions" (AGENTS.md, 2026-10-15) and the Gemini reviewer's request/outbox mailbox (2026-10-15).
  The Gemini lead (mrgemsol today) holds the card, the ledger, the delivery table, the owner's pages and the plan file,
  and briefs its own workers; the Claude lead (mrtabby today) holds the Open List and the inbox, `main`, the changelog,
  the handover and this file. They talk only to each other by `internal/gemini/PROTOCOL.md` (two one-writer files a day;
  every ask with a deadline and a default that is never "wait"). Before, AGENTS.md said the lead kept the ledger and
  pages and a Claude session held the card.

## 2026-10-16

- **A Gemini build goes on its own git branch, not a diff file in `internal/gemini/`** (the owner, at 23:32Z: "lets
  have them stop being forced to only use internal/gemini"). Two diffs tonight (Gemini's D19 and mrtaco's matte D18) were
  cut from a copy older than main and would have deleted lines later commits added (decisions.md 1f143f59, handover
  lines). Now: a build is a branch in a worktree under `h3_worktrees/gemini-<item>`, cut from current main and committed
  there; the lead lands it after a blind read from the merge base (`git diff main...<branch>`) by pathspec. Never the
  shared tree, never a push. `internal/gemini/` stays the mailbox (requests, outbox, reviews). What it replaces:
  AGENTS.md's "reviews, never builds, through files in `internal/gemini/`".

- **The second subject's clothing words are per shot; on 1010-1038 she wears a dark sleeveless top, not the apron**
  (mrtony, the lead, at 23:00Z, correcting its own words of 20:59Z). The plan's one `clothing` line ("a white apron
  with a bib ... over a sleeveless floral blouse") cited source frames 1011 and 1020; on 1015 (cropped from the source)
  she wears a dark sleeveless top at the table, and no apron shows on 1010-1038. `second_hfp_1010c` was rendered with
  the apron line and draws apron straps on 1011-1036 and a full bib on 1037-1038 (the blind render read,
  `internal/claude/2026-10-16_mrtony/review_render_1010c.md`). The owner's ruling stands (her clothing is redrawn
  with her and named); only the session's words were wrong. What it replaces: one `clothing` line for every shot.
  1010c is a candidate beside the kept 1010j and is not laid; it goes to the owner with this reading.

- **The dad's 1397-1418 is made, not left as the original; the look that left it out is overruled** (the owner, at 22:35Z,
  showing 1397 and 1398 in the whole-video view: "see the big beard? that's the original's, for both people"). The 10-13 look
  (third_fp_1310c's left_out, kept by the lead's ruling this morning) said his raised arm covers his face from 1398; on 1398 his
  face is visible. Both he and the lead get a whole-shot load over 1306/1310-1418, planned only, so neither turns back to the
  original at 1397. What it replaces: the entry below, "The dad's 1306-1309 and 1397-1418 stay the original", for 1397-1418.

- **A Gemini builder builds `shot_to_load.py job --carry-words-from`** (the owner at 2026-10-16T22:18Z, offering the idle reader
  "to take stuff off" mrtaco's plate; this job only): a remade job carries every person's word of the old job by the tool,
  tagged, and refuses a word it cannot carry, after two hand copies today lost one (the kitchen's floor 480; 21 spans for
  17). Brief D20 in `internal/gemini/2026-10-16_requests.md`; a Claude reader reads the diff and the lead commits.

- **A Gemini builder builds the pages and shot_coverage reading the gap check** (the owner, "Yes", in the lead's terminal at
  2026-10-16T22:17Z; this job only, as for the frame table): `review_pages.py` and `shot_coverage.py` read the gap check of the same
  table and refuse without it, so no page calls a frame done that the gap check shows as the original. The brief is D19 in
  `internal/gemini/2026-10-16_requests.md`; a Claude reader reads the diff and the lead commits.

- **Before the next renders: the node's exact matte is saved and laid; the floor holds rather than empties; the
  per-frame matte and re-compositing kept pieces are tried first** (the lead, mrtony, at 21:29Z, the owner having
  asked "should we re-design anything before we go load it again" on a Gemini reader's redesign, and given no
  preference among the four). The diagnosis taken: three stages each decide which pixels to keep, two of them from
  colour differences. In order: (1) the node saves the per-pixel matte it composited with beside each window's region
  file, and the assembler lays exactly that, never a threshold; landed before second_hfp_1010c, lead_fp_1010f and the
  kitchen render. (2) A tracked person's part is never emptied by the size floor: it is held from the nearest kept
  frame unless a look says no face there (`--no-face-on`, drop); landed before any new plan uses it. (3) A matte that
  follows the person frame by frame (the old and new person's masks, inside the regenerated tokens) in place of the
  token staircase is TRIED on two existing renders by re-compositing, no sampling; the owner judges the rim on
  playback. (4) Re-compositing a kept piece rendered "only what changed" as "whole region" from its saved latent
  (`bench/recomposite_window.py`) is TRIED on second_hfp_0604d as a before-and-after; no kept piece is re-laid without
  the owner's word on that piece. Deferred: keep-outs composited after decoding instead of pinned before sampling (it
  changes what H3 is given, so it needs new renders to test).

- **The keep-out carve is done on the node's own grid, and the region the owner looks at is the one the render
  uses** (the lead, mrtony, ruling route C at about 20:20Z on mrtaco's options; landed 77ea523a and ac6b7a10, recorded
  20:47Z). The canvas carve of gate 3 left squares inside her outline the node still gave back, because the node
  pools a keep-out over each latent step and cuts on its own grid (the zoomed picture's for a zoomed load). Now, at the
  on-plan rebuild of every planned run, each keep-out loses every pixel under a node unit that is more than half hers on
  any frame of its step; the region is remade with the node's own `video_mask.window` on the carved files, the masks are
  hashed after that rebuild, and the land step refuses a render whose own region differs from that prediction unit by
  unit. The node's own count of units given back inside her own track, canvas carve alone then grid carve
  (`internal/claude/2026-10-16_mrtaco/probe_grid.out`): second_hfp_0014f (zoomed 1.333) hands 168 to 0, strangers 4 to 0,
  held and clothing 0; second_hfp_0604g (not zoomed) hands 100 to 0, the rest 0. Its extra cost in keep-out pixels:
  0014f hands 30680, clothing 5562, strangers 1514; 0604g hands 91134, clothing 1758, strangers 701. The control
  (`route_c_control.out`): the prediction made from each load's own files equals its render's region unit for unit on
  lead_fp_0438c and second_hfp_0014f. Earlier approximations (0/503/232 and 34) are replaced by these. A load planned
  before the carve is refused with "make the load again under a new name" (second_hfp_1010k).

- **The mom's clothing next to her new face and hair is redrawn with her; the stay keep-out is retired** (the owner,
  in the lead's terminal at 19:25Z, choosing "Redraw it" when mrtony asked whether H3 should redraw her apron strap, bib
  and blouse or keep them as the original). The plan's `look.stay` (mrtwo's ruling of 2026-10-14, arm K, on mrpre's
  apron figures) kept her own clothing as one keep-out drawn from her own track; the gate 3 carve would have emptied it
  (a blind read, `internal/claude/2026-10-16_mrtony/review_gate3.md`: on 0014f the carve takes 1416632 px of it, all but
  15213 inside her track), while her job file still announced it. Removed from `data/fun/plan.json` (the plan before:
  `data/fun/plan.before_no_stay_2026-10-16T192527Z.json`; the old block under `look.stay_removed_by`). Her clothing is
  named in the load's text, as accessories are. Risk the owner accepted: the strap or collar may shift near her neck.
  Takes effect on her next load; nothing made changes.

- **The assembler lays a "whole region" render's region whole, every piece, at both sizes; a render made with "only
  what changed" keeps the node's own choice** (the lead, mrtony, at 18:59Z, ruling gate 4's scope on mrtaco's
  probe, `internal/claude/2026-10-16_mrtaco/probe_gate4.py` and its `.out` files). The open item of the simplification
  entry below said "where pieces share frames"; the probe found the cause is wider: at `--size source`, which every
  rebuild uses, `assemble_delivery.laid_over` weights every piece by its own per-frame `changed()` test, the first piece
  and a piece alone on its frames included (the holes trace's item 5). On the two "whole region" renders it laid back as
  the source 45.1% of 0014f's regenerated region (worst frame 117) and 54.6% of the kept street lead_fp_0438c's (worst
  443), pixels a median 2.4 levels off the source but the sharp original's texture inside the upscaled new face. Now: a
  piece whose render's composite is "whole region" (read from its render graph) is laid by its run's `region` (the
  node's regenerated cells: what H3 replaced, margin included), at both sizes; where pieces share frames the overlap is
  split by the subjects' masks as before. A piece rendered with "only what changed" keeps `changed()`: its render is
  already the source where the node kept it, and laying its region would swap the original for the render's softer
  copy of it. A piece with no region record keeps `changed()` and says so. Every kitchen piece of today's cut was
  rendered with "only what changed", so the kitchen does not change; the street and any later "whole region" piece do.

- **No keep-out gives back a cell inside the person's own track: the capture takes her track out of each keep-out
  before it writes the files the node reads, rather than refusing the load** (the lead, mrtony, ruling the form of the third gate the simplification entry below
  names, on mrtaco's probe). Built first as a refusal, gate 3 would have stopped nearly every load of the mom's: her
  plan's keep-outs `held`, `hands` and `clothes` are her own things, so they lie inside her own track by construction
  (on the saved regions: 0604e/f/g, `held` on 317 frames up to 57 cells, `hands` on 279 up to 87; 0014f, `hands` on
  100 frames up to 55; `internal/claude/2026-10-16_mrtaco/probe_gate3.out`, empty when this was first written and filled
  by mrtaco's rerun at shares 0.5 and 0.75 the same evening; at 0.75 a few of these frames drop and the lead's 1-cell
  `strangers` hits on 3240b-d go to none); the lead's `strangers` keep-out touched her
  track by 1 to 5 cells on a few frames of 2823b, 3759, 5935 and 3240b-d. Now: the capture's files step, after every mask is
  read and before it writes the `track__` files a load's `others` are built from (`masked_job.capture_files`, about
  `bench/masked_job.py:607-640`), takes out of each keep-out that is not one of our people the cells that are more
  than half the union of the run subjects' own tracks (whole cells: on 0014f that takes 34989 of 144813 `hands` px and
  4930 of 15213 clothing px lying just outside her track, `internal/claude/2026-10-16_mrtony/review_gate3.md`); it records per frame the pixels and cells each keep-out lost
  (data); a cell given back inside her own track after that is a hard failure, since it can only be a fault. What it
  cannot see: a zoomed load's tokens are on the zoomed grid, the carve on the canvas's. (The lead's wording was
  corrected twice the same hour: mrtaco objected that a carve in the capture alone would not reach the node, then
  found the node is handed the capture's own copy of each keep-out, so the capture is where the carve reaches both.)
  Another person's track over hers is unchanged (that is who stands in front, not a keep-out).
  So her own hands, held things and clothes inside her region are regenerated with the rest of her; outside her track
  the keep-outs still act. The plan's keep-outs are not edited. What it replaces: keep-outs taking cells inside a
  person's own track, a source of the holes in `internal/gemini/2026-10-16_holes_trace.md`.

- **A video's sidecar is its full file name plus `.json`: `<name>.mp4.json`** (the owner left the name to the lead,
  mrtony, at about 17:57Z: "consistent and not collide with other names and be easily understood that it goes with
  it"). `bench/sidecar.py` as built named it `<stem>.json`, which about 15 tools already write as their own record
  beside the videos they make (`head_reference.py`, `draw_face_points.py`, `shot_to_load.py` and others), so the two
  would collide. No file under `data/` or the share's `Video/fun/` was named `*.mp4.json` when this was decided; the
  assembler's own check record is already `<name>.mp4.check.json`. Every writer of a sidecar uses this name, and only
  it.

- **No largest-piece rule on any load whose part holds a face or hair, the lead's included** (the lead, mrtony, at
  about 17:50Z, ruling the scope of the gate the entry below names, on mrtaco's question). The entry below removed the
  rule from the mom's face-and-hair loads; the lead's plan still carried `pieces: largest` on her `face_and_neck`
  loads (mrtwo, 2026-10-14, against enclosed holes on 3262-3265 under the old composite). It put the original back
  inside her region the same way: on the kitchen candidate `lead_fp_0613c` it dropped pieces on 180 of 362 frames, up
  to 9333 px on 772 and 6312 on 850; on the street `lead_fp_0438c`, 5 of 46 frames, up to 114 px (each load's
  `files_on_plan` output under `data/2026-10-16_<load>_job/chain/`). Removed from `data/fun/plan.json` (the plan
  before: `data/fun/plan.before_lead_pieces_2026-10-16T175022Z.json`); the gate in `bench/shot_to_load.py` refuses
  `pieces` on a load whose part holds hair or `face_and_neck` on any frame. Both candidates stay on the owner's page
  as rendered; their look decides them. The holes the rule was set against are open again: closing a hole the region
  encloses, not dropping a piece, is the way that does not give the original back.

- **The simplification wins: H3's picture is kept whole inside a person's region, and nothing puts the original back
  inside it** (the owner: "The simplification wins. Make sure this never happens again"), after a code trace
  (`internal/gemini/2026-10-16_holes_trace.md`) found four live sources of the original showing through the mom's new
  face on second_hfp_0014d: the composite "only what changed" (19% of the regenerated pixels restored, the render's own
  record), keep-outs carving cells out of her own track (a strangers track at her ear, a held track at her neckline),
  the capture's largest-piece rule (smaller pieces of her face and hair dropped on 146 of 154 frames), and the hands
  give-back at assembly. Now: every load's composite is `h3_config.MASKED_LOAD_COMPOSITE` ("whole region"); the mom's
  plan has no `pieces` and no `restore` (the plan before: `data/fun/plan.before_simplification_2026-10-16T172912Z.json`);
  the lead's Eyeglass and hands give-backs on 3060-3109 stay only for the kept 2823b, marked `exception`, and go when
  that shot is made again. To make it hold: gates that refuse the old ways (a face-and-hair load with the largest-piece
  rule, a give-back of a person's own parts not marked as an exception, a keep-out that is not one of our people taking
  cells inside the person's own track), built next. STILL OPEN (mrgan's read): where two pieces share frames (the kitchen, 604-948), the assembler lays each later piece only where its own per-frame test says it changed the picture (`assemble_delivery.changed`, over `CHANGE` levels), so a whole-region render's barely-changed pixels can still come back as the source there; the fix is the assembler laying a piece's carried region (its run's `region.npz`) in place of its changed pixels, queued with the gates. What it replaces: the 2026-10-04 choice of "only what changed" for
  the band clip (one clip, one seed), the largest-piece rule of 2026-10-13/14, and the hands give-back of 2026-10-14.

- **No accessory give-backs at assembly; an accessory that matters is named in the load's text** (the owner, "I
  agree", after finding the mom's original earring and necklace leaking through her new face in the one view, frames
  47, 55 and 125 of 14-167). The `Apparel@head` restore (mrtall's extension of 2026-10-15, below, of the owner's
  keep-accessories rule of 2026-10-14) gave back any part-model Apparel touching the head, grown 6 px, on every row of
  the lead and the mom; Apparel is also collar and blouse, and the label moves frame to frame, so it pasted moving
  patches of the original round both heads (the assembler's own check flagged it on nearly every piece of hers:
  d014, 332 of 0604d's 345 frames). Removed from `data/fun/plan.json` for both (the plan before:
  `data/fun/plan.before_no_apparel_2026-10-16T164222Z.json`); the cut is rebuilt from the same renders. Kept for now:
  the lead's `Eyeglass` give-backs on 3063-3109 and 3909-4052 (removing the first would take her glasses off the kept
  2823b where its render lost them); they go when those shots are made again with the glasses in the text. Giving
  OTHER people back inside a region stays (it is not an accessory). Also to test (one render): the composite `whole
  region` in place of `only what changed`, which restores the source inside the region wherever the render did not
  change the picture enough (`workflows/masked_job_graph.py` takes a per-load `composite`). What it replaces: the
  2026-10-15 earrings entry and the accessory half of 2026-10-14's rule.

- **The dad's 1306-1309 and 1397-1418 stay the original; the frame check is to say why** (the lead, mrtoon, on mrgan's
  correction of its own sheet look). The kept `third_fp_1310c` left them out on a blind reader's look at every frame
  (its job's `left_out`: his face behind a blurred hand and arm; from about 1403 he crosses the lead, which "A shot,
  made whole" does not cover). A whole-shot candidate (ruled at first, then withdrawn before any job) would add little
  and remake a kept piece. `bench/frame_coverage.py` lists such frames as the original while tracked because it reads
  no look; it is to read a laid piece's `left_out` and the shots file's looks and print them as "left out by a look",
  with who and why. Also recorded: 1533-1631 nothing to make (his back to the camera; a session's look, mrgan's sheets,
  `data/fun/shots.json`).

- **`face_kitchen` is not laid; the lead's own whole-shot kitchen load fills her gaps** (the owner, in the lead's
  terminal: "agree do it"). The per-frame gap check (`bench/frame_coverage.py`, mrgan) found that none of the 242
  frames where a tracked person shows as the original inside a shot with a piece of hers is covered by any render the
  rule lets stand; the only ones covering them are 10-10 renders with no build record (the owner's rule of 10-13: a
  piece whose making cannot be said is rendered again). The owner had kept `face_kitchen` (604-1050) on 10-11; it
  stays unlaid. The lead's load over the kitchen is made next, by the recipe of "A shot, made whole", on the source
  (one hole a pass is not built; every kept kitchen piece of hers was made this way). What it replaces: the
  handover's item 11, which waited on the gate branch and one hole a pass.

- **Trust before renders: one list of what is laid, then one joined clip of the whole video from it** (the owner, in
  the lead's terminal at the day's start: "I want trust that what Im reviewing is being recorded and what im watching
  is the same thing you are" ... "And then just produce a joined clip end to end of what we have as far as we have it,
  from the provenance ledger. And I review that."). No render until it is done. In order: `data/fun/laid.jsonl` the
  only thing the table reads for what is laid (span, subject, load, the file's sha256, who chose it and why), seeded
  from the table of 2026-10-16T0043Z and every line not backed by the owner's keep marked unconfirmed; the review
  page records the sha256 of the clip each answer was given on; a per-frame gap check (where a tracked person shows as
  the original, where two renders of one person meet) feeding the ledger; a check of every laid piece's reference as
  rendered against the file now; then the end-to-end joined clip, and the owner's review of it is the baseline, on one
  review page (the owner: "one review page not 10 different versions"). Two sessions: the lead builds 1, 2, 4 and 5; a
  second session builds the gap check (its brief: `internal/claude/2026-10-16_brief_frame_coverage.md`). What
  it replaces: the handover's "Tomorrow, in order" items 5-12 (the gate, gaze, restore carry, hold form and the rest
  wait), and the four places the cut was decided (the owner's keeps, `prefers.txt`, two lines typed into
  `rebuild.sh`, the newest table by file time).

## 2026-10-15

- **What lands, and what is undone** (the owner, 2026-10-16 ~00:30Z: "if we are breaking shit undo breaking shit" ...
  "Yes unless its obvious to you another claude session and gemini as a no brainer"). A change that breaks something
  is reverted (a revert commit, never a rewrite). A change lands only when it is an obvious win to the lead, another
  Claude session and the outside reviewer. A check's colour is not the verdict (the owner: "if something is red it
  doesnt mean good or bad without context ... or green for that matter"): a red is read for what it checks and why it
  went red (a real break, a stale case, the environment) before anything is reverted, and a green says only that its
  cases held, not that the picture is right.

- **A text never tells H3 what is NOT in the movement video** (the owner, 2026-10-15 late, on a clause the lead built
  for lead_fp_2823d, "on some frames nothing is drawn on its face": "actually dont say it at all ... its inferred ...
  you cant say some face may not be there cuz next its something else and its like... yo dont look at this thing that
  isnt there"). The text says what is drawn; frames where nothing is drawn are left for H3 to see. The clause was never
  landed (branch deleted; it had passed its case and a cold read). What it replaces: the lead's fix of the same evening.

- **The outside reviewer builds one job: a reference box a window** (the owner, 2026-10-15 late, in the lead's
  terminal: "tell gemini to build this as you have it in mind and you make sure its good after"). An exception to
  AGENTS.md's "reviews, never builds", for this job only: request C15 in `internal/gemini/2026-10-15_requests.md`, in
  its own worktree off the unlanded gaze branch, no git, the reference only (the input's zoom stays one box a load,
  `video_mask.py:1795-1796`), off by default, landed only after the lead's review, a cold read and the owner's
  playback of a test render. What it replaces: nothing yet; the per-load reference box stays the default.
  CANCELLED the same night (about 00:05Z): the owner, "not sure i like it at all yet"; the lead then found the one case
  where her eyes carried (2823b) had a 439 px face on the canvas itself, so a reference-only box rests on an untested
  premise. The reviewer built nothing; it answered C14 only.
- **The gaze work: the touch-nothing test first, every other piece set aside** (the owner, 2026-10-16: "this i agree
  with"). One render, 1097-1133 with only a tight reference made outside the chain, plus regression renders on kept
  pieces where her eyes already work (1575b, 0805d, 0630d, 2625); nothing lands unless the glance carries and nothing
  kept gets worse on the owner's playback (`internal/claude/2026-10-15_gaze_pieces.md`).
- **The gate's two rules that read a part before the word are replaced by one** (the owner, 2026-10-15 late: "lets kill
  this rule is there some other information we can give it thats useful in its place"): region and reference agree on
  every frame after every word is applied; no third rule, none in comments; its comment says why. Built, not landed
  (branch `mrnight-fixes`); `handover.md`, "Tomorrow, in order", has its state.
- **A reference is never renamed or moved once a render has read it** (the owner, 2026-10-16: "this needs to be fixed");
  a remake takes the next free name, and verify fails where a recorded sha256 no longer matches. 0751b, 1097b, 1156 and
  5114d point at files they never read (their originals sit in `*_SUPERSEDED_*` folders; 5114d's proven by sha256).
- **Every run records the nodes the server served from its cache** (the owner, 2026-10-16: "that needs to be solved
  tonight"). Landed after 0604g (1ff12a36).
- **A tracking count on every load** (the owner, 2026-10-16), and **one hold rule from the flip survey in place of the
  hand-written --hold-part spans** (the lead's proposal): the tracking count is queued; the hold rule is DROPPED as
  proposed, on the dry run and the reviewer's C16 (the face points read "face" on backs of heads and hands, and do not
  exist when the capture runs); main's `SHORT_FILL` on "nothing labelled" stays, perhaps raised to 4 (`handover.md`, 4).

*Written by the lead of the day (mrtall) as the rulings are made; mrarc holds the card, mrlol tooling.*

- **The owner's evening answers, on the review page and in the lead's terminal (21:16Z to 22:15Z), recorded here
  because they were relayed only to the Open List and the sessions until the postmortem's blind reader found them
  missing:**
  - **2823-3109 is REMADE as one load in one window, beside 2823b, and the owner keeps one** (terminal, about 21:55Z:
    "Remake 2823-3109"). REVERSES the 18:52Z line above ("so it is not remade; 3098-3109 continues its window"): the
    owner first chose "continue 2823b over 3098-3109" (answers.jsonl, 21:17:40Z, "Continue just make sure you have
    enough of her face visible (pre-overlap?) to be able to track her"), then the cold reader of `--continue-from` found
    2823b's stored window (277 frames, off the chained grid) cannot be continued. If her face shows on 3110-3166, the
    load is the whole shot to 3166. Open List row 6.
  - **The dad keeps 1310c; 1397-1418 to be made** (review page 21:20-21:22Z: "the singing is perfect here", "he still
    turns into the source dad ... all the way to end"; 1306-1309 "nitpicking if this is too much work"). As a
    continuation of 1310c only if its stored window is on the chained grid; otherwise back to the owner. Row 3.
  - **The far-table mother in the hall gets a load of her own** (21:22Z: "swap her with a load of her own"; "Could we
    make an attempt to at least try and put blonde hair?"). Row 8. And **1093-1133: her, tiny in the background, hair
    alone** (terminal, about 22:00Z: "worst case if its easier to just make it blonde hair that works too").
  - **The spinning ball: her face swapped from 2200 to about 2247** (21:24Z: "swap her face from 2200 to around frame
    2247 or so"), the rest of the 2050 shot by the recipe. Row 1.
  - **Who is in a shot:** no second subject in 1156-1188 ("1156-1188 = the lead singer") or 1466-1729 (the owner's
    who-page answers, read by the lead from a screenshot because the page was opened from the share and did not save);
    `data/fun/shots.json`, backups beside the lead's scripts.
  - **The second subject's kitchen load:** context noise 0.8 per load, not a default (the lead's ruling on mrgo's
    question; the default waits for this render's seam on the owner's playback); her shut eyes drawn only where the card
    seat's sheet shows them shut (938-953, 968-1009), never on downcast frames.
  - **Dropped:** the lead's hair restore ("i dont remember lead singers hair ever being an issue at all"); an
    identity-drift reading ("you dont need to do that, just have me watch it"); the written label on references (parked,
    unlanded, as the one-pass lane closed).
- **A zoomed load's reference and its input zoom are unlinked, but that is NOT why the gaze test failed; CORRECTED the
  same evening** (the owner, 2026-10-15 late: "it sounds like a bug / oversight to me", "Yes so fix that", "If youre
  zooming in on her eyes in the sideway gaze shot and you cant get results maybe this is why"). The lead first called
  the unlinking the cause (the node hands a zoomed load its motion video "as it is, not cut to the box",
  `workflows/masked_job_graph.py` above RENDER_REGIONS; `video_mask.zoomed_source`), then read the figures: on 10/14's
  zoomed test the input zoom was 1.80x (this file, 2026-10-14, "Gaze is a known limit") and the reference's own head
  cut 1.71x (`references/lead_1097_1133/lead_head_reference_1097_1133.json`, scale), the same scale within a few
  percent, and the glance still did not carry. Both were loose because each is one box round her whole movement in a
  dancing shot. So linking the two is not built; the lever is a much tighter crop of both, so the eye spans several
  of the reference's 32x32-px tokens (`next_steps.md`, the queue of 2026-10-15, item 8). The lead's error: a cause
  stated before its figures were read.

- **The wiki corrected against today's rulings** (the owner, 2026-10-15 evening: "Maybe worth you reading through our
  llm wiki docs to make sure everything is solid and still accurate and flag things that look off"). Three cold readers
  read the wiki and AGENTS.md against the code; the lines that contradicted today's rulings were corrected in
  masked_v2v.md, h3_uses.md, next_steps.md, state_signals.md and index.md, each with a dated italic note of what it used to
  say. Multi-person H3 GENERATION is open; only two subjects in one masked pass is closed (the owner: "h3_uses seems to
  be a general guide for more than just masked generation - you can do multiple people in h3 generation"). What is left
  (stale "not built" lines, retired-board pointers, dead paths, two docstrings, code against the voice ruling) is one
  list for a clean-up job: internal/claude/2026-10-15_wiki_audit.md.
- **Two Claude sessions, not three** (the owner, 2026-10-15 about 21:35Z, in the lead's terminal: "I'd like to just keep
  it limited to 2 claude sessions - you and a new fresh one"). mrarc, mrlol and mrvid closed with notes
  (`internal/claude/2026-10-15_*_closeout.md`); mrgo holds the card and the tooling from the evening, briefed by
  `internal/claude/2026-10-15_brief_mrgo.md`. What it replaced: AGENTS.md "Three seats from the start of each day".
  ComfyUI was restarted by the lead (pid 3933887) so the server belongs to a live session.
- **Two or more subjects in one masked pass is closed** (the owner, 2026-10-15, about 21:35Z: "yeah lets kill multi
  person for now (again) then", then "or rather multi in one pass"). A row in `docs/roadmap.md`, "Closed lanes". One
  hole a pass (a load given another load's render as its video) stays open and is built after `--continue-from`; the
  labelled-reference arm is parked in its worktree (mrlol-label), unlanded.
- **Two holes in one pass blended the two faces; the next test fills one hole a pass** (the owner, 2026-10-15, about
  21:25Z, on the review page's `second_lead_0852`: "the middle one blended faces"). The test the owner asked for
  ("try it on a short window - like 100ish frames"): the kitchen, 852-958, both subjects in one load through
  `shot_to_load pair` and `load_chain`, the text paired label by label (draft 8), read whole from the queued graph by
  the lead and by the outside reviewer. One render, one seed, one shot, so a direction and not a result. Next, on the
  same frames: the second subject's load, then the lead's load given that render as its video, so each pass has one
  hole and one pair of references. The labelled-reference arm (the owner's hint) is still built and run, as cheap.
- **A text says "sings" where a person sings, and nothing else about voice** (the owner, 2026-10-15, about 21:07Z:
  "keep it simple - just say singing if theyre singing in a shot/window. if theyre not, dont mention anything about
  talking or singing. the end. make that a rule."; earlier the same day, of the kitchen's second subject: "just dont
  mention anything about the mom talking or singing"). What it replaced: a "talks, not heard" line ("<Subject 1> (S1)
  talks, her lips moving with words that are not heard; ... the singing is not hers") and a draft "<Subject 1> neither
  sings nor speaks". The plan's voice key stays as data; only "sings" writes a line. A line in `AGENTS.md`, "Settled
  about H3".
- **Shorter windows inside one load are allowed, on the seam grid** (the owner, 2026-10-15, about 20:15Z: "you can do
  shorter windows too just make sure the frame multiples we documented either yesterday or day before are right to
  avoid seams"). Read with the morning's rule (one load per subject over the whole shot, e95b41ee): a load may run as
  several chained windows, each on `h3_rules.chain_lengths()` (17n+5 and a multiple of 3), so every window cuts its
  latent steps on the same grid counted from the load's first frame (`masked_v2v.md`, "A latent step is several
  frames, and its grid is fixed for a whole load") and each later window continues from the one before. What stays
  ruled out is a shot split into separate fresh loads (the kitchen's 948/949 jump). `shot_to_load job --seam` refused
  a plan of three windows or more; that refusal goes.
- **The longest window is 362 frames, `h3_rules.MAX_LENGTH`; no cap of 345 anywhere** (the owner, 2026-10-15: "362 is
  the max I have no idea why 345 was set - that was ages ago. remove that reference to 345 please. i dont want it
  coming back"; AGENTS a071659e). What it used to claim: `h3_config.LONG_LENGTH` was 345, and
  `docs/h3_geometry_and_nodes.md` said "The shipped graphs render at h3_config.LONG_LENGTH, which is 345, not 362".
  Built by mrvid (branch mrvid-length): every long graph takes 362, the stamped baseline among them, so a figure
  measured on the baseline before the change is one grid step shorter. A window chained on another stays on the seam
  grid (17n+5 and a multiple of 3), whose longest under 362 is 345, derived and never typed (`LONG_WINDOW`);
  `bench/check_length_ceiling.py` goes red on any 345 not named in its list of measured figures.
- **A reference still's definition is specific; the generation's description says less** (the owner, 2026-10-15,
  about 19:50Z, on the kitchen's two-subject text: "specificity in the reference image definition is important. in the
  actual generation and how its used... sometimes less is more"). The second subject's definition now names her face
  from the still, "with blue-grey eyes, fair skin and blonde hair", where the owner had said earlier in the day that
  "blonde hair" was all it needed; the detailed_description keeps only her short hair. Her hair's retention line is
  "the woman's face and the blonde colour of her hair from <Picture 1> are retained; her hair's length is changed to
  short, ending at her shoulders" (the owner's question of about 19:20Z; the outside reviewer's G13 found no better
  wording): partially_preserved in `vendor_guides/ref_en.md` 4.1 means some defined characteristics are changed, so a
  changed hair is said to be changed, never "retained".
- **To H3 a masked load is not an edit, and a text never describes the frozen picture** (the owner, 2026-10-15,
  about 19:40Z, on the lead's draft of the kitchen's two-subject text: "it cant see a mom with a white apron", "thats
  all frozen", "to h3 this is not an edit - you have to internalize this and make sure its core to this project", "we
  have created two empty holes of noise and h3 is generating new video in those holes", and "the video around it and
  the audio guide it - it was jointly trained on video/audio, thats why the music / singing guides the singer to sing
  and open her mouth by stating shes singing without having to say when"). The draft had told the two
  women apart by "on the right" and "on the left"; mrlol's read of 0604d's tracks found the lead on the right of the
  seated woman on 606-761, so the words were false for 156 frames as well. Reversed in `masked_v2v.md`: step 2 of
  "The frame every masked text is written in" asked for "where the subject is, and what on them is the scene's own",
  and the two-subject section asked for "where each of them is in the picture"; both now say the opposite. Made a
  line of `AGENTS.md`, "Settled about H3". Open, and read on the first render: whether H3 puts each pair of
  references in the right hole.
- **The lead's sunglasses and hands on 2823-3097 are given back, and the run is not cut short** (mrtall, after
  15:08Z, on mrarc's question). On 3060-3097 she raises the white sunglasses over her face with both hands. The
  run stays whole, since cutting it at 3059 would switch her face back to the original for the close-up's last 38
  frames. The plan takes two restore spans: Eyeglass on 3063-3097, once the glasses are on her face (Eyeglass also
  fires on her eye makeup), by the owner's keep-accessories rule (the same glasses as the span on 3909-4052, mrtwo
  2026-10-14T17:53:09Z); and Left_Hand+Right_Hand on 3060-3097, by mrtwo's standing rule of 2026-10-14 21:18Z
  (her own hands in front of her face are given back). Where it lives: `data/fun/plan.json`.
- **second_hfp_5461c takes a hands-and-arms restore span on 5461-5497**, by the same standing rule (its gate's
  f001: her own hands and arms up to 3577 px inside the region on all 37 frames).
- **second_hfp_1010e's held override covers the egg in her fist** (mrtall, after 15:10Z, on mrarc's overlays
  `data/2026-10-14_fun_second_inputs/second_hfp_1010e.held_1015_1018_1023.jpg` and `..._held_1015_close.png`).
  The held mask left 115, 69 and 6 px on 1015, 1018 and 1023: her fingertips, her fist and forearm, and on 1015
  the rim of a white egg she holds at her chin. This is mrtwo's rule of 2026-10-14 after 21:18Z for held things
  she touches near her face, widened from 1010d's "fingers and hand" to name the egg.
- **One shot is never split: each subject gets ONE load over the whole shot** (the owner, in the lead's terminal
  after 19:25:31Z: "Stop breaking one shot into multiple windows"; and at 19:24:43Z on the kitchen: "This whole scene
  really needs to be rethought, I feel like we're just smashing things together and dont know provenance"). This
  REVERSES the lead's ruling of 2026-10-14, "A shot may be split into runs where the split falls on frames with no
  face of the subject", which narrowed the owner's rule of 2026-10-13 (b2cccac9, "do not break one shot into two
  windows"). That narrowing is what left the kitchen 604-1009 as four short pieces of the lead with the original
  between them, and two independent renders of the second subject meeting at 948/949. From now: a subject's load runs
  from the first to the last frame of the shot she is in; frames inside it where her face is hidden or her part is
  empty stay inside the same load (`--hold-part`, `--least` with a reason, or the part simply empty there), never a
  reason to split; a shot longer than one window is one load if the node can chain it (asked of the tooling seat).
  Continuing one load from another is not the way to join a shot (the continuation build is paused). Pieces already
  standing that split a shot are remade as whole-shot loads, each put to the owner first; the kitchen first, as one
  plan on one page before anything renders.
- **The second subject's hair is short, shoulder-length blonde, by the text** (the owner, in the lead's terminal
  between 19:14Z and 19:17Z: "for the mom: just put in the prompt as a partially preserved element that her hair is
  short shoulder length blonde hair. For the ref image <Picture x> part of the prompt i think all you need to put in
  is that she has blonde hair"). `data/fun/plan.json` subjects.second.text: `looks` "blonde hair" (was "long, loose
  blonde hair parted in the middle, fair skin and blue-grey eyes"), `hair` "her short, shoulder-length blonde hair"
  (was "her long, loose blonde hair"); backup `internal/claude/2026-10-15_mrtall_scripts/plan.before_mrtall_second_hair_*.json`.
  Her still is not changed. Pieces of hers already kept were made long-haired; her loads from now on carry the new
  words.
- **The owner kept lead_fp_2823b, so it is not remade; 3098-3109 continues its window; the eye test is cancelled**
  (the owner on the review page, 18:52Z: "eyes and mouth and movements and all look fantastic"). What it replaces: the
  lead's ruling after 18:50Z to remake 2823 once as 2823c over 2823-3109, and the eye A/B ruled after 18:36Z. The
  readers' frame-by-frame finding that her eyes do not close where hers do stands as a row; the owner's eye on
  playback accepts the piece. The snap at 3097/3098 is mended by a short load over 3098-3109 continuing 2823b's
  stored window, once the job step takes `--continue-from` and `--hold-part`.
- **Who sings is the plan's data, per shot and subject** (the owner, 19:05Z, on second_hfp_5675b: "Subject 2
  shouldnt be singing ... She's mouthing like she's talking but its not audible ... frame 1306 scene is correct ...
  We should have a way to handle these via data and the plan to say whether or not to include singing in the
  prompt"). A plan key, for example `voice`: "sings", "talks, not heard" or "silent", read by `bench/load_text.py` in
  place of the voice table's guess where the plan names one. Seeded: 5675-5715 the second subject "talks, not heard";
  1306 the third subject "sings". 5675 is then made again.
- **The second subject's still: fun_mom.jpg and fun_mom_edit.jpg are the same edited image, on purpose** (the owner,
  about 19:15Z in the lead's terminal). The owner removed a purse strap from the original photo and saved the result
  under both names, in case a session reached for the older name later. Both files are byte-identical (mrvid,
  `internal/claude/2026-10-15_mrvid_shorter_hair.md`); no copy of the photo as first given is kept on this machine.
  Jobs name `fun_mom_edit.jpg`.
- **Drawing no mouth where the class map shows no teeth stays an option, off by default** (mrtall, after 18:50Z, on
  second_hfp_5114e, the test ruled after 17:01Z; `data/2026-10-15_second_hfp_5114e/look_render/reading.md`). With
  `--class-mouth-needs-hole` its reference drew a mouth only on 5116, and the render's mouth was open on every frame,
  as on 5114d with the mouth drawn: on 5119-5123 and 5127-5129 hers is closed or pursed. So on her 44 px face, nothing
  drawn does not let the render follow her; 630d's 658-674 (open in step with hers, nothing drawn) is the one
  counter-case and is not a default's evidence. Open: whether the still's own mouth sets the render's when the
  reference gives none, and the text promising a drawn mouth where the reference leaves it out.
- **2823 is made again once, over 2823-3109 in one window; 3110-3166 holds no lead** (mrtall, after 18:50Z, on the
  owner's note on the 2823-3166 joined clip, "good up until 3097; 3098 is when she turns back into the original", and
  mrarc's look, overlay `data/2026-10-15_fun_past2049_inputs/lead_3098_3131_part_and_track.jpg`). On 3098-3107 she is
  in full view as the camera pans and she slides out at the left; the part model labels nothing on 3098-3099 and
  3101-3104. From about 3110 the lead has left, and the woman with short dark hair and cat-eye glasses on 3122-3166 is
  not the lead, though the preview's lead track follows her; the queue's run 3131-3164 is dropped. The remake (2823c)
  waits for the eye test, the lip lines' on-the-head check and a person's word to hold her part across 3101-3104, so
  one render carries them all; until then the seam at 3097/3098 stands.
- **Earrings are given back at assembly where the part model's Apparel touches her head** (mrtall, after 18:21Z, on
  mrlol's measure `internal/claude/2026-10-15_mrlol_scripts/earring_apparel.txt`). The part model has no earring
  class; her pearl earring on lead_fp_0805d is labelled Apparel and the render repaints it (805d's render reader: gone
  on 814-825). Apparel is also her collar and blouse, so the restore takes only Apparel pieces touching her Hair or
  Face_Neck (grown 6 px, a starting value), for the lead and the second subject on all their rows, by the owner's
  keep-accessories rule of 2026-10-14. It applies at assembly, so pieces already kept get their earrings back with no
  re-render. The load's text does not name them (the one-form rule).
- **The smoke's "node order" line had been unable to go green since 2026-09-27; it now reads the graph** (mrvid,
  c8501106, after mrarc's 18:12Z restart smoke). Its log needle ("chaining onto an existing attention override") was
  printed by the Sol node until e0b8cc55 deleted it, so every smoke since read MISSING there. The order is now read
  from the graph the smoke submits. At the default 39 frames "sparse ran" says "not owed at this length" where Sol's
  own decline line shows. `bench/check_smoke_h3.py` holds every needle to the code literal that prints it, so a stale
  needle goes red in a check, not on the card.
- **The hair coming and going: two node fixes go, one does not** (mrtall, after 17:52Z, on the outside reviewer's
  frame-by-frame watch `internal/gemini/2026-10-15_watch_hair_item4.md` and mrlol's offline measures
  `internal/claude/2026-10-15_mrlol_scripts/hair_item4_record.md`). Built in `video_mask.py`: (a) a keep-out piece
  under 256 px (one cell; a starting value) that is on at most 2 frames of a four-frame step does not hole the step,
  and a one-cell hole whose four neighbours are in the region and which is not her own part is closed; this mends
  0014d's flashes at 57-60 and 159-166. (b) Region islands touching none of her own part are dropped (all 43 on the
  two windows lie on her apron, chest or the bowl). Not built: (c) filling pockets in the composite's changed mask,
  which finds none on the flashes. 0604d's bottom-left hair stopping at her blouse and apron strap is the owner's
  ruled form (Open List row 10 of 2026-10-12, "clothes left alone, hair to the shoulder line"), not a fault.
- **The strangers keep-out on a close-up of her alone is turned off per load, by a look; ruling (A) of 2026-10-14
  stands** (mrtall, after 17:12Z, on mrlol's reading of `bench/strangers.py` and its measures
  `internal/claude/2026-10-15_mrlol_scripts/strangers_on_her.txt`). On 2823b and 1097c the strangers mask is 99% the
  lead's own hair, hands, bracelets, glasses, neck and dress just beyond her narrow track, and each load needed an
  override. Dropping every piece that touches her (the lead's first ruling, after 17:05Z) would reverse ruling (A),
  "another person's arm across her stays a stranger", so it is withdrawn. Instead `shot_to_load job --no-strangers WHY`
  turns the keep-out off for a load whose every frame a person has looked at and found nobody else in, with the
  look named in the job's defaults. A geometric rule (a piece whose core classes join hers pixel by pixel is hers)
  is a row, needing a part-model rerun to measure.
- **Mouth v2: steadier over time, and the three turned mouths on 805 kept; a closed mouth drawn as an outline stays**
  (mrtall, after 16:50Z, on mrlol's measures against the outside reviewer's watched table (G6),
  `internal/claude/2026-10-15_mrlol_scripts/lip_gap.*`, `mouth_flicker.*`). Built: a one- or two-frame drop-out
  between drawn frames is filled where on_body is at least 0.45; where the lip lines' share is at least 0.60 and near
  at most 0.20, on_body's floor is 0.47 (805's 819, 822 and 823 back); and the lines, once drawn, are left only
  under a share of 0.6 (2823 from 46 form switches to 8, none of one frame). All starting values. Not built: dropping
  one-frame islands (it loses 805's open 824), and "no teeth or tongue, so draw no outline": the class map shows
  neither, nor a lip gap, on 9 of 14 open frames, so it cannot tell a closed mouth from an open one on these faces.
- **A hair-alone load says hair alone and draws no face; 5461 is the test of a hair-alone load with no reference**
  (mrtall, after 16:20Z, on mrlol's measures `internal/claude/2026-10-15_mrlol_scripts/hair_alone_faces.txt`). On
  5461 a quarter of a face or less shows on most frames (Face_Neck over face height squared, median 0.36 against
  484's 1.53), yet the pose model places a whole frontal face, and the reference draws it as large as a plain face's.
  The body model holds her head raised (nose pitch median 11.9 degrees against 484's 9.9) while she shows only her
  crown. Ruled: (a) the generated text of a whole-load hair-alone load names her hair alone, and drops the plan's
  `looks` (its eye words invite a face); (b) no face lines on hair-alone frames (`--no-face-on`, passed like
  `--no-mouth-on`); (c) one test, not yet a default: 5461 made with no reference video (motion none), so the
  source's own picture carries her bow. Its render decides the default. A `--parts-on` hair span inside a face load
  keeps the window's reference, with no face lines on those frames.
- **5461-5497 stays the original today: the hair-only load stopped at its plan, the second try** (mrtall, after
  16:10Z, on mrarc's `data/2026-10-14_fun_batch_b_inputs/second_hp_5461b.STOPPED_AT_PLAN.md`). Its plan look reads
  a face drawn onto her crown on 5466-5497. Two causes are named in the tools: a hair-only load's generated text
  still says "her face is the face of <Picture 1>", and its reference still draws the face's lines (and holds the
  head raised from 5481 while she stays bowed). With both mended, it may be tried once more as a new load, on the
  lead's word: the stop rule is against blind retries, not against a retry whose fault was named and fixed.
- **The second subject on 5461-5497 is made hair only** (mrtall, after 15:53Z, on mrarc's blind plan look of
  second_hfp_5461d, `data/2026-10-15_second_hfp_5461d/look_blind/reading.md`). She is bowed with both hands over her
  face on all 37 frames; no face shows. The owner's answer on row z24 (2026-10-13) covers it: where she bows her
  head and no face shows, her hair stays the still's. So the load is hair only (second_hp_5461b: the job step names a hair-only load "hp", and its next free letter is b), not left as the original
  (mrarc's recommendation (b)). If its plan look reads a face pulled onto her crown, it stays the original with
  that reason.
- **The lead's straw hat on 2625-2711 is kept through the load's text, not a restore span; 4310-4394 runs at the
  480 floor** (mrtall, after 15:47Z, on mrarc's read `internal/claude/2026-10-15_mrarc_scripts/four_runs_read_2026-10-15.md`).
  The part model has no hat class, so by the owner's keep-accessories rule of 2026-10-14 the hat is named in the
  place-and-action sentence, and the brim is a named look on the plan strip and for the reader. 4310-4394: her
  face is small in a wide shot (px median 3873 against the capture's 4096), as on 604-851's far stretches; the
  zoom default stands. The Eyeglass `bench/runs_left.py` reports on 5935-6020 and 4310-4394 is strangers'
  sunglasses; she gets no Eyeglass span there.
- **Every file the card's chain executes is edited in a worktree while the card runs** (mrtall, after 15:33Z, on
  mrarc's finding). lead_fp_2823's reference ran on uncommitted edits to `bench/head_reference.py` in the shared
  tree (its record: ran_on changed_since true), because the lead's tooling brief named five chain files for the
  worktree rule and told the tooling seat that `head_reference.py` "lands in main with its check". The rule is
  every file the chain runs, not a list. That reference is set aside.
- **The mouth gate covers both forms, and lead_fp_2823 waits on it and on the sunglasses** (mrtall, after
  15:33Z). 2823's reference, drawn in the lip-lines form, has no mouth on 61 of 275 frames by the lines' share,
  wide-open frames among them (2862-2867, 2912-2922, 2957-2970). A frame under the share now falls back to the
  class-map mouth where the geometric gate passes, else no mouth. The reference also failed its off-track check on
  3075-3087: her white sunglasses lie outside her track, so the face points' middle lands off it. The check is to
  count the subject's restore classes (Eyeglass) with her track. The run is not cut at 3073 (the switch ruled out
  above).
- **The outside reviewer talks to the lead through files, and gets the playback questions** (the owner, about
  15:20Z-15:30Z in the lead's terminal). From today the owner relays nothing between the Gemini reviewer and the
  sessions: requests go in `internal/gemini/<date>_requests.md` from the lead, and answers come back in the
  reviewer's own files plus a line in `<date>_outbox.md`. The reviewer wrote its own way to wait for work. The
  owner: it was trained on video and audio and sees motion and hears sound natively, so use that. The lead gives
  it the watching (motion, flicker, a mouth with the singing, eyes) on the stacked clips with sound. What it used
  to be: on 10/14 the owner pointed it at each piece of work and relayed its notes. Where it lives: `AGENTS.md`,
  "How the owner and the sessions talk" (a264a5da).
- **The owner's two pages move to the video's folder and stay there** (the owner, about 15:24Z in the lead's
  terminal). `PROVENANCE.html` and `TO_REVIEW.html` are written to `Video/fun/` on the share, one copy each,
  made new by the lead after every landing or stopped load from the newest ledger and table
  (`bench/review_pages.py`; the lead's session script is `internal/claude/2026-10-15_mrtall_scripts/pages.sh`).
  What it used to say: `AGENTS.md`, "One ledger, made new on every landing", put them "in the day's folder on
  the share", and on 10/13 and 10/14 they were written to `Video/fun/day_2026-10-13/`. That folder's two
  pages are now forwarding stubs. The lead keeps the ledger: it is the source of truth that sessions and the
  outside reviewer read for what stands, where each piece came from, and the gaps. `--day-folder` still points
  at `day_2026-10-13`, where `bench/day_pieces.py`'s piece folders are read from.
- **The mouth mend of 2026-10-14 (eed4acb9) drops correct mouths; the class-map mouth gets its own per-frame
  gate** (mrtall, after 15:19Z, on mrarc's finding; `internal/claude/2026-10-15_mrarc_scripts/refcheck/`).
  Under eed4acb9 a frame gets no mouth when the pose model's lip lines lie on the class map's mouth under
  `MOUTH_CHECK["on_share"]` (0.8). On small or turned faces the lines miss even where her mouth is open, so the
  references remade under it drop the mouth on most frames: lead_0630_0680 on 40 of 51 (share median 0.33),
  lead_0805_0833 on 19 of 29, second_1010_1031 on 22 of 22, second_5114_5129 on 16 of 16. This includes 632-647,
  where 630b's reader called the class-map mouth right. What it replaces: mrtwo's ruling of 2026-10-14 that one
  constant decides both forms. The class-map form is now kept or dropped a frame at a time by where its lips
  sit on her face (the outside reviewer's parked geometric gate, section 5A of its 10/14 consolidated file, as
  the starting shape). It is tested on the 10/14 readers' frames before landing. 630c, 805c, 1010e and 5114c do
  not render; each goes again as the next letter on a reference made under the gate.
- **751b's flicker on 761: a short fill is accepted where the part model labelled nothing** (mrtall's go, after
  15:14Z, on mrlol's reading of `data/2026-10-14_lead_fp_0751b/subjects/lead/part_grades.json` and
  `doubted_frames.json`). The rule: a frame the grade empties, inside a "nothing labelled" run of one or two
  frames, with a part on both neighbours inside the load. Across the 10/14 and 10/15 captures it would change
  six frames. Corrected: the handover of 10/14 (item 3 (d)) said the hold's first loop emptied 761. It kept the
  142 px saved speck, and the floor loop emptied it. 761 is empty in either order, so the order is still not the
  cause. Built in a worktree, landing on the lead's word.
- **The 2050-2622 shot is built only up to where the group picture starts to warp, about 2205**; the warped
  picture, the spinning ball, its shards and the bubbles wait on the owner (Open List row 1 of 10/15,
  `Video/fun/day_2026-10-15/sheet_2200_2622_every12.jpg`). What it replaces: the 10/14 queue's run 2050-2248,
  which reaches into the warp.

## 2026-10-14

*Written by the lead of the day (mrtwo) with the commits; the owner's words are the Open List's inbox notes
`owner_terminal_2026-10-14T*`.*

- **One ledger, made new on every landing, is a standing rule** (the owner, about 15:20Z in the lead's
  terminal: built into the project's rules for later sessions). The lead owns it. After every landing or
  stopped load, the render ledger and the delivery table are made new from the build records. The owner's
  `PROVENANCE.html` and `TO_REVIEW.html` are made from those same two files, never typed and never written
  twice. What it replaces: on 2026-10-13 the pages were a lead's session scripts with no owner named for
  keeping them current; one input (the stopped loads) was a typed list and the review file's decisions were
  typed into the script. Where it lives now: `AGENTS.md`, "Operative rules" and "Where state lives". The
  pages becoming tracked tools that read only generated files is the tooling seat's job of the day.
- **The evening's standing rules, so the card does not wait on the lead** (the lead, 2026-10-14, about 21:14-21:18Z).
  A remade load's look rests on an earlier blind reading when `bench/look_carry.py` finds every frame's track, part
  and region equal or a subset; otherwise a reader is owed. A gate that names a restore span of the subject's own hands
  and arms on the load's frames gets that span in the plan and the override, without asking; spans stay on the load's
  frames, never the whole video. It replaced the lead measuring and ruling each by hand.
- **A frame whose lips do not lie on the mouth draws no mouth** (the lead, 2026-10-14, eed4acb9). Three references
  drew the mouth off the face (in profile; on the second subject's chest on hair-alone frames), all from the class-map
  fallback. Kept at the existing `MOUTH_CHECK["on_share"]` of 0.8 over a new 0.5 fitted to three loads. A kept piece
  made in the older form (0294d) stands.
- **Keep-out flags read their own size** (the lead, 2026-10-14). Strangers' specks under a floor are data (90d6a633),
  for the strangers mask only: a small face of ours still counts. Held pixels that are thin slivers or the subject's
  own clothing below the pose model's chin point are data (cadde74a); a chin scored under 0.3 counts as none. The
  part floor is counted at the size the load renders at (8d1c86d2). A stay keep-out empty on every frame is dropped
  and said (69cfe18e).
- **Gaze is a known limit; no more card time on the eye's form for it** (the lead, 2026-10-14, on the measure the
  lead set in advance). The pose model's iris points do follow a sideways look (lead_fp_1097b, 1119-1133: +0.25 to +0.31
  eye widths, scores 0.96-1.0) and the reference draws it, but no render followed: median iris x +0.253 in the original
  against -0.022 (the ring), -0.030 (filled), -0.030 (contrast) on the whole frame and +0.004 zoomed 1.80x
  (internal/claude/2026-10-14_mrdrill_scripts/iris_offset_render_*_1097_1133.csv). The contrast eye stays the default
  (no worse by this measure, and it paints out the sculpted eyeball). Open: a larger zoom or a reference of the eyes
  alone, not tried. In the same run the drawn mouth DID carry (lead_fp_1156c: the mouth opens with hers on 1170 and
  1182, where lead_fp_1156's stayed shut).
- **AGENTS.md's reference-form line points at the chain's defaults** (the owner, about 18:13Z: "make these our
  defaults and stop forgetting these tools we built"; built in e92085d5). What it used to say: the reference is
  drawn with "eye outlines with the iris, brows, nose", a list of what was drawn on 2026-10-12. What it says now:
  every load carries the chain's defaults (the reference held to the track, the eye, the mouth by its check,
  zoom, the strangers keep-out, the part's largest piece, the hair band, restore spans, the text), printed by each
  job step and kept in the job file, and a default is turned off only with a written reason.
- **A better form does not remake a kept piece** (the owner, about 18:12Z: they have loved many of the clips;
  the nitpicking is for tooling that handles future windows without being rebuilt). The defaults of the day (the
  contrast eye, the drawn mouth, zoom, the generated reference sentence) apply to windows still to make. Of the
  2026-10-12 remakes, only those with a known fault go on (14-154 and 415-437 rated wrong, 604-948's hair, 278-293's
  wide eyes) plus 1010-1031, never made; 1039-1060 stays the owner's keep `second_hf_1039_g`. This narrows the
  morning's "every piece of 2026-10-12 is made again" below.
- **One prompt form on every load, and the tools on by default** (the owner, about 18:05Z, on `lead_fp_1156`:
  her mouth does not open and the piece is weaker than it could be). Every window's text is the same reference
  prompt we have been using; no per-load words for the face (not "grin", not "mouth wide open"): the reference
  video carries the face. What carries it better is the tools, and each becomes a default of the chain from the
  plan rather than a session's choice per load: zoomed wherever the node allows, the mouth drawn on the reference
  where its lip lines lie on the mouth (else left out, with the figure printed), the iris between the lids. What
  it replaces: each of these was opt-in, so `lead_fp_1156` rendered on the whole frame with no mouth drawn.
- **Keep accessories, for the whole of `fun`** (the owner, about 17:45Z, on the lead's white sunglasses on
  3909-4052). A face replacement keeps what the person wears on the face and head: sunglasses, earrings, a hat.
  What the part model names (its `Eyeglass` class, which `sapiens2_parts.PARTS["face_and_neck"]` puts inside the
  face region) is given back to the original at assembly by a plan key per subject, written onto that subject's
  rows by the generated table (the tooling seat's job of the day). What it does not name, the load's text names
  so the model draws it.
- **A shot may be split into runs where the split falls on frames with no face of the subject** (the lead,
  2026-10-14, on the tooling seat's design read `internal/claude/2026-10-14_mrdrill_box_read.md`, untracked,
  and the card's `lead_fp_1466b`/`lead_fp_1575b`). The rule of 2026-10-13 that one shot is not divided between
  windows (b2cccac9) is about not cutting a shot where the face carries across the cut; a run of frames with no
  face of hers (the back of her head, a spin) carries nothing across, so the runs on either side are made as
  loads of their own and the frames between stay as the original. This is what lets a shot the node refuses
  to zoom in one box (her path wider than the frame, as `lead_fp_0168`) be made as zoomed runs with no new
  code. A box that moves inside one load is not built: its centre would jump inside a latent step, whose four
  frames share one region in the node. Each split is said in the job's note with the frames and why.
- **Every piece of 2026-10-12 is made again, each as its whole shot, through the present chain** (the
  owner, about 14:43Z: the 10/12 pieces are not trusted). Until the owner judges a remake, the lead lays it
  over the owner's 10/12 keep on the same frames with `delivery_table.py --prefer`, citing that word.
- **The team is three seats from the start of each day**: a lead that reads, rules and holds the Open List,
  a card session and a tooling session (the owner, about 14:20Z; written into `AGENTS.md`, "How the owner
  and the sessions talk", on their yes of about 17:05Z). What it replaces: "Two sessions during the day",
  one on the card and one on tooling, with the lead's work folded into whichever held the card.

- **A load's text is generated from its records, around one sentence a person writes** (the owner, about 18:05Z: every
  window's text consistent with the others, saying exactly what the reference shows and the least else; the lead's
  rulings of about 18:10Z: the closer view "of the head and shoulders", nothing typed by sight; no expression sentences).
  `bench/load_text.py`, called by `shot_to_load.py text --place-and-action`, makes the fixed lines from the reference's
  record, the plan's `text` words for the subject and the video, and the voice table. What it replaces: each load's text
  copied by hand from the last, whose `<Subject 2>` sentence could name what an earlier reference drew; and
  `masked_v2v.md`'s "The text: the fragment", which quoted the 10/11 form ("shapes drawn on its face") as the form to
  copy. That section now points at the generator. (Written by the tooling seat, mrdrill, at the lead's order.)

## 2026-10-13

*The entries from here to the next note were written by the afternoon's lead
session (mrpin) from the morning lead's files (`internal/` notes and handoff,
untracked), the Open List's inbox notes `owner_terminal_2026-10-13T*`, and
the commits. The owner's words were taken down by the morning's lead (mrfit);
the inbox note or commit named is the nearer record.*

- **A piece is laid only from its own build record, and a clip from before
  the 12th is an example, never a piece** (the owner, 2026-10-13, in the
  lead's terminal: the inbox notes of 13:36Z and 13:39Z; with them, that
  "not possible" is not an answer where an earlier day did the thing, and
  that the whole of the source video is the aim, every shot a piece once
  through the builder). A render with no build record, one whose queue
  record names no file, or one whose file disagrees with its record is not
  laid and is rendered again. What it replaces: the joined clip of the 12th
  laid the 10th's pieces wherever the 12th had none, and held stretches as
  the source on sessions' readings. Where it lives now: step 3 of the rule
  in `bench/delivery_table.py`, and the render ledger's `MAKING` column
  (`bench/render_ledger.py`). What it costs on playback is the Open List's
  row 22: the lead's face is the source again wherever only a 10th piece
  existed.
- **The delivery table is generated from records, never typed** (the
  morning lead's brief to the tooling session, at the owner's ask of 13:16Z
  for a clip that is "the manifest's pieces and nothing scraped from
  folders"; `bench/delivery_table.py`, 33860bc5 and after). Which render
  stands on a shot is one stated rule: the lead's `--prefer` with a printed
  reason; the owner's keep; else the newest render in the frozen form whose
  file agrees with its record. The afternoon lead's rulings on it, each in
  the tool: a render the owner's last choice says is not the one does not
  stand and cannot be preferred (e09baaec); the page's own positive button
  is a keep (dc454c17); the plain cut stands as the form's exception with
  its reason handed in by `--prefer`, since no record holds why a load took
  it. What it replaces: a table typed at the end of a day from memory of
  which render was the good one.
- **One working copy of the whole video, one clock** (the morning lead,
  2026-10-13, after the owner's 13:39Z note; the copy and its record are
  beside each other under the server's input folder, `fun_0000_6394_24p`).
  `data/fun/manifest.json` names it since 15:18Z and `data/fun/shots.json`
  is the two halves merged (its `span` and `written_by` say how; the first
  half is kept as `shots_0000_1919.json`). What it replaces: two part
  copies, the second on an offset clock.
- **A load's masks sat on the wrong frames when the load began after its
  preview, and a look read the symptom as the picture's** (found by the
  tooling session, mrolaf, 14:34Z, measured against the preview's own track
  video). The session scripts that ran every load of the 12th and 13th
  passed the load's own first frame as `at=` on each `--mask`; `at=` is the
  source frame the mask FILE begins on (`bench/capture_masked_run.py`'s
  docstring). It is right only when a load begins where its preview does.
  Two builds of the third subject's middle shot had every mask early by the
  gap between the load's first frame and the preview's; a blind reader then
  saw the part sitting on his shoulder and arm on the load's first frames,
  and the load was cut to start later, which made the offset larger.
  Nothing was rendered on either (their folders carry `SUPERSEDED.txt`).
  What it used to claim: the morning's notes and the job file of the second
  build said the first frames were cut "after the blind reader's second
  look"; that look was of shifted masks and the cut is void. Where it lives
  now: `verify` fails a sighting whose `at` is not its preview's job's
  first frame and names by which it could not check (75d20ec4); the load
  chain reads `at` from the preview's own job, so it is not typed
  (`bench/load_chain.py`). The load as rendered is `third_fp_1310c`, its
  note saying which look its frames rest on.
- **The strip's movement panel never cropped the reference** (the tooling
  session, 14:13Z, measured on a strip of the day; 1d946fd1). What it used
  to claim: the morning's handoff and its draft for this page said the
  panel "crops the 4:3 reference to a near-square" and had misled two
  readers. The tile was always the whole frame, fitted and padded; a
  reference on a black ground has no edge to see on a black panel, and a
  head crop cuts its figure at the reference's own edge. The tile's outline
  is now drawn and its heading says it is the whole frame.
- **One load from its preview to the page is one tracked tool, and the job
  file is the record a load is made again from** (the afternoon lead's
  rulings to the tooling session, 14:32Z and 14:51Z; `bench/load_chain.py`,
  c4c71d32; `bench/shot_to_load.py`, c7cc24e4). What a load depends on is a
  key in its job file or a named file it points at (`job.preview`, the mask
  options, the form of its reference and why); a fact of the video lives
  with the video (the voice tables in its manifest; each subject's still
  and part ticks in `data/<video>/plan.json`, so a reference picture
  changes in one line); a fact of the machine is an environment key
  (`H3_MODEL_STORE`). The drawn form is the default and the plain cut is
  taken only on a person's word with its reason; the tool prints the face
  and lash-line figures and decides nothing from them. A head reference is
  made on the load's own frames and belongs to a shot, a subject and a
  span, never to a load's name. One preview a shot, every subject present.
  The go for a render is a file beside the job file, which is not edited
  after its plan is built. What it replaces: four session scripts copied
  from session to session with the name changed, thirteen positional
  arguments each; three different things having decided drawn points or
  the plain cut; a reference made over a whole shot before a look cut the
  load.
- **The third subject's face-only loads, and what goes with a face** (the
  morning lead built them on the owner's rulings of the 11th and 12th; the
  blind readings are beside each capture, `reading_blind.md`). His region
  is the part model's face and neck, so his beard, moustache and the fringe
  over his forehead are replaced with the still's face. What it used to
  claim: each render's page item said his beard was the original's;
  corrected on the page from the readings. Whether that stands is the
  owner's (the Open List's row 23).
- **The afternoon's lead is an Opus session working from files** (the
  owner, 2026-10-13, about 14:05Z in the morning lead's terminal, to keep
  that session's context; the handoff is in `internal/`). The tooling
  session reports to the afternoon's lead from then.

- **The `h3-ab-session` skill is removed** (the owner, 2026-10-13, about
  17:00Z, in the afternoon lead's terminal: deleted and staged by them, to
  go in the next commit). It restated nothing: it routed a rendered
  comparison to `docs/eval_comparison.md`, which was and stays the
  authority for the process, and to `bench/blind_batch.py::KEY_DIR`. What
  it used to claim elsewhere: `docs/open_experiments.md` named the skill
  beside section 3 of that page as the way a blind session is run (the
  mention is dropped); the generated `INDEX.md` listed it (regenerated).
  The dated records under `bench/results/` and `docs/research/` that name
  the skill are history and are left as written.

*The entries below were written by the tooling session (mrolaf) with its
commits.*

- **No zoomed preview stage is built: one run said enlarging the picture
  changes nothing the part model reads** (the lead's go for the run,
  2026-10-13; the tooling session ran it). `next_steps.md` had said since
  2026-10-12 that the preview stage "has no zoom", that the part model
  "at whole-frame size calls her hands face from about 1025" on
  1010-1038, and that a preview "run on an enlarged picture of the
  subject, with every table mapped back to the frame's coordinates, is a
  tool to write". The premise did not hold against the code: the part
  model, the body model and the pose pass each already crop to the
  subject and resize to their own input. And it did not hold against a
  run: the same preview on those 29 frames cut to a box round the second
  subject and enlarged twice reads the face class over the same area
  (`data/2026-10-13_second_cz_1010/`, the reading and its two tables).
  By a session's look at five frames, her face does not show from about
  1025: the part there is her hands. `next_steps.md` is corrected. What
  the run does not cover is said there.

- **A load rendered zoomed in takes a wired motion video, as it is wired**
  (the owner's yes on the Open List's row p04, 2026-10-12 17:05Z, "the
  zoomed load that keeps its motion video: build it next"; built
  2026-10-13 by the tooling session at the lead's relay). This reverses
  one line of the cut of 2026-10-10 below ("a zoomed run with a wired
  motion video is refused") and none of the rest: the cut of a wired
  video to the box and the choice `a video I wire, zoomed in` stay
  removed. `video_mask.zoomed_source` hands the video on untouched and
  its docstring says what the zoom does and does not do to it; the job
  builder takes a zoomed load's motion video only as a `motion_file` the
  load names and still refuses the preview's own whole-frame body model
  there (`workflows/masked_job_graph.py`). Prose that said otherwise and
  is corrected in the same change: `video_mask.py` at
  `RENDER_ZOOM_REPLACED` ("a zoomed run with a wired motion video [is]
  removed") and the node's tooltip ("a load with a cut in it is refused,
  and so is a run that continues another or wires a motion video");
  `masked_v2v.md`, "Limits known today" ("A load rendered zoomed in takes
  no motion video ... the section below and this one cannot be combined
  today") and "A small subject, rendered zoomed in" ("a zoomed run with a
  wired motion video is refused"); the builder's note at `RENDER_REGIONS`
  ("takes no motion video (the node refuses one)"); `docs/checks.md`,
  the row of `check_video_mask.py` ("a wired motion video refused"). Not
  rendered when written.
- **The floor that empties a small part is a starting value, and a
  capture says where a clip's own sizes fall against it** (the owner's
  answer on the Open List's row p07, 2026-10-12: set the floor from this
  clip, or mark it a starting value and have the tool say when a clip's
  own sizes disagree; built 2026-10-13 by the tooling session at the
  lead's relay). `bench/capture_masked_run.py::PART_LEAST_PX` keeps its
  value and loses its standing: its comment had said "Reasoned: under
  that the model has nothing to draw the part in", and on 2026-10-12 it
  emptied a small face on about half the frames of a load whose
  whole-frame render the owner had accepted, with nothing printed. Now
  `files` prints, for every subject with a part, the part's size on the
  capture's own frames, the floor and who set it, the frames under it and
  the frames it emptied (`part_sizes`, also in the manifest and in
  `part_grades.json`); `least=PX` on a `--mask` sets the floor for that
  subject and `least=0` empties nothing by size; `sizes CAPTURE...`
  prints the same for captures that exist. The second form of the
  owner's answer was taken and not the first: a floor worked out from the
  clip needs a rule for what a fragment is, and the one tried on paper (a
  share of the part's median size) would empty whole faces on a shot
  where the subject walks toward the camera. No new red: the figures are
  printed and the reading is left to whoever runs the capture. A capture
  with no `least=` empties what it emptied before (run by hand on the
  street walk's saved masks: the same frames as the capture of the 12th;
  with `least=1024` and `least=0`, fewer and none by size). What the
  sizes cannot see is printed with them: whether a small part is a sliver
  or a whole small face, and the size it is drawn at on a zoomed load.
- **An analysis folder keeps every copy its video has had, and a job
  renders from the present one** (the lead, mrpin, 2026-10-13 about 16:04Z,
  on the tooling session's audit; the owner's ask of that morning behind
  it, that a load of yesterday builds again tomorrow with one thing
  changed). What it replaces: the manifest held ONE copy's name, so when
  the manifest was pointed at a copy of the whole video that afternoon,
  every job that named the first copy found no folder and could not be
  built, though the file had not moved. Now `copies_before` in the
  manifest holds each earlier copy with its identity, the source frame
  its frame 0 is, what replaced it and why; a folder is found by any of
  those names and a file is held to the entry of its own name
  (`clip_folder.py`); `bench/clip_analysis.py init --copy` keeps the copy
  it replaces and `copy-before` enters one dropped earlier;
  `bench/shot_to_load.py` writes the PRESENT copy into a job whatever copy
  its preview was made on, and refuses a copy on another frame clock.
  PROSE THAT LOST TO CODE, corrected the same day: `clip_folder.py`'s
  docstring said the bench tools "and the node that reads it" import it.
  No node does and none did when this was written: a job's graph is built
  by `bench/masked_job.py`. The module is at the repo's root, so it is
  landed as node code is; the server reads nothing from it.
- **A working copy's sound is stretched as its picture is re-stamped, and
  is read after it is made** (the lead, mrpin, 2026-10-13 about 15:46Z).
  The copy of the whole video made that morning had its picture held to
  the source's frame by frame and its sound not read at all; the sound ran
  at the source's speed and fell behind the picture by a thousandth of
  the elapsed time. It is replaced by a new file under a new name that
  holds its picture stream untouched; the old file stays, with a dated
  line in its record. `bench/copy_sound.py` reads a copy's sound against
  its source's; the record is
  `bench/results/2026-10-13_working_copy_sound_against_its_picture.md`.
  A voice table is made from the SOURCE's sound at the source's frame
  rate, never from a copy's.

*The entry below was written by the lead of the later part of the day
(mrtan), from the owner's words in its terminal.*

- **The masked lane's canvas stays as it is, and why** (the owner,
  2026-10-13, in the lead's terminal, early in their afternoon: they asked
  whether the insets would sample faster on a small canvas, then whether
  1152x768 would be the more native size, and agreed it stays once the
  three facts below were read out). The job's canvas
  (`workflows/h3_config.py::MASKED_JOB_CANVAS`) is the 4:3 row of
  `docs/h3_resolutions.md`; 1152x768 is that page's 3:2 row, as much a
  trained size and the wrong shape for a 4:3 source (`ffprobe` on the
  working copy gives the source's size). Every preview, track and mask is
  made at the job's canvas and `verify` holds them to it
  (`inputs_at_the_canvas`), so another canvas is every preview made again
  and kept pieces on two canvases. And a load rendered zoomed in scales its
  box to the same canvas (`video_mask.zoomed_source`), so an inset costs a
  whole render's sampling and buys size for the face, not time. What it
  used to claim: the constant's comment says only that the value was
  inherited from the job files since 2026-10-10 and names the table's row;
  nothing said why that row, and the question was re-derived in the
  terminal. Where it lives now: `masked_v2v.md`, "Rules of the lane". Not
  queued: one render of a zoomed inset on a 3:2 box and canvas, which is
  where the full frame's shape would not matter.

*The entry below was written by the lead that followed (mrtower), from the
owner's words in its terminal.*

- **A window's length is a choice: any frame count valid for H3, and one
  shot in one window where it fits** (the owner, 2026-10-13, in the lead's
  terminal, in their afternoon, after asking why the rest of the video
  would cost so much card time). Their words, in substance: there is no
  reason to render the same number of frames every time; the length is
  adjustable for a reason and only has to be a count H3 takes; the
  planner's list begins where it does because someone set it so and for no
  other reason; `h3_rules.MAX_LENGTH` is the most, and very short counts
  are possible though they would not use the shortest; more frames raise
  what attention costs and may run out of memory with enough references,
  and are still right when they let a longer static shot sit in one window
  with no cut; and one shot or cut is better kept in a single window, which
  also makes its text easier to write. What prompted it: every load of the
  day sampled the shortest length of `loop_plan.CHAIN_LENGTHS` whatever its
  own length (each plan's line says how many frames its last one was held
  for), so most of what a short piece sampled was thrown away. What the
  repo already held: `h3_rules.snap_length` and the grid it rounds to;
  nothing in the lane's page said the builder's lengths were a default and
  not a limit. What it used to claim: a session's list of what going fast
  did not allow carried "no shorter sampling" as if ruled; that was one
  word of an earlier lead's about fewer sampling steps, the owner never
  ruled on it, and a shorter window had not been raised. Where it lives
  now: `masked_v2v.md`, "Rules of the lane". When this was written the
  builder's change was in hand and what the lane's sound does with a count
  that is not exact on the audio clock had not been read. Both were done
  later the same day: the sound's count is rounded by core and cut exactly
  by the lane, and nothing refuses it for a window that stands alone
  (`audio_freeze.slice_window`; the audio clock is the seam's rule,
  `audio_freeze.window_geometry`); and the builder gives a load one window
  holds the shortest length of the grid that holds it
  (`masked_job_graph.window_for`), with the planner, the window node and
  the song node's least window changed to take it (`loop_plan.stands_alone`).
  What the pages used to claim of this: [`window_context.md`](window_context.md)
  carried two entries of 2026-10-10, that a shorter window was "not built"
  and then that it was "not scoped as a patch" because the planner's
  shortest was the shortest length on both clocks inside the vendor's and
  core's stated ranges. Those entries stand as dated; the reason they give
  for where the list begins is one a session wrote, and the owner's word
  is that it began there because somebody set it so.
- **A shot is a cut, and a window is split only when it has to be** (the
  owner, 2026-10-13, in the lead's terminal, later the same afternoon,
  after agreeing that the record's boundary at frame 1920 of `fun.mp4` is
  not a cut). Their words, in substance: when they say shot they mean a
  cut, and "shot" is H3's language for one; one shot is not broken into
  two windows (their example: the shot in the kitchen with both subjects
  at the start of the video); it is divided only where that is absolutely
  necessary: an edge case, a fix or polish pass, resources that will not
  hold the full window, or a shot longer than `h3_rules.MAX_LENGTH`. A cut
  to another place is another shot: two short ones can share a window
  where the tracking is easy, and the text then has to account for both.
  As the lane is built, each shot is its own load with its own text and no
  load holds two. *Corrected the same hour: this entry and the lane's rule
  first said that a text with several shots "stays in one window", which
  read their sentence as a rule about texts. They asked what that meant
  and said what they mostly mean: do not break one shot into two windows.*
  With it, of the frames past a load's
  own: the only reason to sample more is an overlap buffer before its
  start or past its end; and of Sol-Attn at a short window: if it declines
  a call that is fine, since its sparsity can only take quality away. What
  prompted it: a trial of the joined clip over the whole video failed its
  own verdict at two boundaries, and a look at the frames showed the shots
  record had split one continuous shot (1898-1944) at the frame where the
  short working copy ends, while the cut it was right about (2799/2800)
  was one the assembler's own reading of the picture had missed. What the
  repo held: `docs/prompting.md` has the `[Shot N]` form; nothing said a
  text's shots and a load's windows are two different things. Where it
  lives now: `masked_v2v.md`, "Rules of the lane". Not built when this was
  written: the record's two halves made one shot, the assembler taking
  its cuts from the record, and one window for a load between the
  planner's longest length and the model's
  ([`next_steps.md`](next_steps.md)).
- **Two neighbouring shots in one load: built as a tool, not a default**
  (the owner, 2026-10-13, in the lead's terminal later the same afternoon,
  after agreeing that one load a shot stays the default: build it, as one
  more tool, and run it on whatever pair the lead thinks best). What they
  want it for, in their words earlier that hour: one face for a subject
  across a cut, and a tough shot helped by an easier one beside it. Built
  that evening by the session the owner started for it, on the lead's plan
  and word: a job block's `two_shots` key
  (`workflows/masked_job_graph.py::two_shots_of`), written only by
  `bench/two_shot_load.py` from the two shots' ordinary loads, and the one
  thing by which `bench/masked_job.py` lets a text hold a cut
  (`held_as_two_shots`). What the repo held, and what it used to claim:
  `masked_v2v.md`, "Running a job", said "A load that holds a cut cannot
  be built for now", and its "Not built" list named "a load over several
  shots with a text that describes each shot"; `bench/masked_job.py` said
  such a load "is not built yet" (`ONE_SHOT`). Each now carries a dated
  note. Unchanged: one load a shot is what every tool makes unless a job
  names the key. Where it lives now: `masked_v2v.md`, "Two neighbouring
  shots in one load: a tool, not a default". No load had rendered that way
  when this was written.

- **An ordinary load renders on the gate alone; a look is owed where a forcing
  rule stands or the load is the first of its kind** (the owner's go-fast
  word of 2026-10-13, relayed by the day's leads and set down in their
  handoffs; built into the chain the same day at the lead's order). Until
  then `bench/load_chain.py render` refused every load whose capture held no
  look by somebody who did not make it, and every load of the day had been
  one with a forced look, so nobody met the refusal; once ordinary loads
  came, a blind reader was being spent on each to get past it. The chain now
  has a step of its own, `no-look`: where the gate is clear and none of the
  forcing rules stands (`load_chain.FORCING_FLAGS`, `FORCING_FIGURES`,
  another subject's clothing or hands inside the region, the head
  reference's read-backs), the load's MAKER records that no look is owed,
  with their own words that the load is not the first of its kind; `render`
  takes that record where there is no look, holds it again to the capture's
  flags as they stand, and the go file says NO LOOK, RENDERED ON THE GATE
  ALONE. The record is not a look and is never counted as one; with a
  forcing rule standing it is refused, the flag named, and a look by
  somebody else is owed as before. What the list names and no flag says
  today is printed in every such record (`FORCING_NOT_READ`). What it used
  to claim: `AGENTS.md` said "One look per load, by a reader who did not
  build it, who writes what it expects the render to show before it
  samples". The owner took that sentence out the same evening (their word
  in the lead's terminal: the answer is to remove it), and the lead made
  the edit; the sentence beside it, that the owner takes the cheap looks
  first, stays, with their addition of the same hour: in batches where
  possible, one by one only when a decision is urgent. Where a look is owed, and that it is by somebody who did
  not make the capture, is now held by the chain (`render` and `no-look`
  in `bench/load_chain.py`) and said in this entry, not in `AGENTS.md`.

- **Where the owner has kept more than one render of the same frames, which
  stands** (the owner, 2026-10-13, in the lead's terminal, after keeping a
  piece of a whole shot, `lead_fp_1898` on 1898-1943, over frames where
  they had earlier kept a piece of its second half, `lead_fp_1920c`). Their
  rule, in order: their comments on the pieces and which keep is the more
  recent; where that does not settle it, the simpler and more elegant
  piece, the one easier to make again, to join in and to use as the
  pattern for other shots; unless measurement shows a clear difference in
  quality between them, which then decides. What the delivery table held
  until then: a keep stands (`bench/delivery_table.py`, its docstring), with
  nothing said for two keeps on the same frames. How it is applied until
  the tool says more: the later keep stands, both keeps and the owner's
  words on each are printed, and the lead's `--prefer` carries the other
  two grounds with the reason written. On that shot all three point the
  same way: the whole-shot piece is the later keep and is one piece for
  one shot. Their preference, said the same evening with the three pieces
  of that shot measured side by side (the whole-shot piece a little softer
  inside the region than the half; its step across the frame where the
  half begins about the original's own, the half's two to three times
  that): a whole shot wins over creating a seam.

- **On the lead's zoomed loads the capture reads her part with
  `pieces=largest` from the start** (the lead, mrtower, 2026-10-13, about
  20:22Z, after the plan looks of the first four inset loads). What
  prompted it: every one of those looks said the same two things, that her
  face is one piece and is never divided, and that every piece away from
  it is not her face; and the first loads through the chain, which set no
  `pieces`, had put a margin on another subject's arm where specks of the
  part lay (`lead_fp_1424b`, its forced look under
  `data/2026-10-13_lead_fp_1424b/`). So a remake after a look was being
  spent on a thing the look always found. What it covers: the lead's loads
  rendered zoomed in (the phone and oval insets, the small-face shots);
  every other forcing flag still forces a look. What it does not cover:
  her plain single-window loads, which keep the flag as it is for now, and
  the other two subjects. Where it lives: the
  plan's `pieces` and `pieces_by` for the subject
  (`bench/shot_to_load.py::PIECES`), the `pieces_by` naming this ruling;
  the tool still sets it for nobody. The guard, a practice of the maker
  and not built into the chain: the capture's `files` step prints what the
  setting takes away a frame at a time, and where that is half the kept
  piece or more on any frame the maker stops and a look is owed. Half is a
  starting value from four loads of one clip, reasoned and not measured
  against a render.
- **The shots record was wrong at five places, and what a shot's frames
  are is now mended by one tool** (the lead on a blind reader's look at
  the frames, and the owner's clicks where who is in a half had to be
  answered, 2026-10-13, from about 21:20Z). One boundary of the record was
  not a cut (the frame where the short working copy of `fun.mp4` ends, in
  the middle of an inset shot); three cuts were missing (a wipe that
  uncovers the lead, a cut from a crowd to a kitchen, a cut inside the
  last shot); one sat two frames before the plain cut. What prompted it: a
  trial of the whole-video joined clip failed its own verdict at two
  boundaries, and the look that followed
  (`data/fun/cuts_look_2026-10-13/reading.md`, the frames either side of
  each boundary) found the record, not the clip, at fault at one of them.
  What it used to claim: the record's shots were taken as cuts by every
  tool that reads it, and the tracker's cuts inside a long shot as shot
  changes; most of the tracker's cuts in the last shot were a head going
  down or an arm across the lens. Mended with `bench/shots_record.py`
  (`merge`, `split`, `move`, `answer`; a copy of the record before each),
  never by hand, each mend carrying its reason in the shot's note; where a
  split left a half nobody had answered on, the owner's answer was asked
  on the who-is-who page and read in from their own lines. Not done when
  this was written: the previews of three of the mended shots seeded from
  those answers; and nobody has looked at the frames between the reader's
  sheets.
- **Why a short shot in its own window is no loss: the owner's account,
  theirs and not measured here** (the owner, 2026-10-13, in the lead's
  terminal, the same evening). H3 does well with shots because it expects
  several inside its longest length, and each new shot breaks up the drift
  a long take decays into. It is written out, with what was and was not
  measured that day, in [`masked_v2v.md`](masked_v2v.md), "Two
  neighbouring shots in one load: a tool, not a default". With it, their
  words on where a window begins and ends: a shorter window makes more
  sense unless the context before it is needed, and the sound is not to be
  cut in the middle of a sung line or a beat of the vocal. No rule came of
  that at their word; it is data to have in front of whoever sets a
  window's edges ([`window_edges_data.md`](window_edges_data.md)).
- **What the owner is shown to judge is the shot as the final cut; and
  where review links may be written** (the owner, 2026-10-13, in the
  lead's terminal that evening, after clicking through single loads' clips
  and watching the day's joined clip). Their words, in substance: a clip
  that shows half the subjects of a shot swapped (the kitchen with only
  one of the two replaced) is really hard to judge; a clip put to them has
  both, like the final cut; for every piece they want the day it was
  rendered, what to look at and what to compare it with; and the links to
  review go in no hosted page and no tracked file, only one local file
  they can find. What prompted it: they kept a load of the lead on a hall
  shot and asked what to do with a clip in which the third subject was
  still the original; and they could not judge the joined clip without
  knowing which pieces were the day's and which the day before's. What
  stands unchanged: a single load's stacked clip still goes on the local
  page as the record of that load (`AGENTS.md`, "A render is handed over
  stacked"). What changes: the list handed to the owner points at a clip
  of the SHOT with every standing piece laid, each piece marked with its
  render day and their last choice; a shot with one subject in it is
  already its final cut. Built that evening as session scripts, not yet
  tools: six per-shot clips, a local review file and a local provenance
  page in the day's folder on the share; each as a tracked tool is a row
  in [`next_steps.md`](next_steps.md).
- **The third subject's look is no beard; and one line of the owner's was
  taken out of the record at their word** (the owner, 2026-10-13, the same
  evening). Their answer on the Open List's row for his beard was as
  rendered, his beard gone. That evening one load of him drew a moustache
  and beard, and they clicked keep on it before watching the joined clip;
  having watched it they said the beardless look stands whatever they had
  clicked, and asked that the data not say they kept the beard. What was
  done, with a copy of each file kept first: in the review page's raw
  answers file that one line was replaced in place by a line saying a line
  stood there and was removed at the owner's word (so every later line
  keeps its number); in `verdicts.jsonl` the copied keep was removed and
  their words stand in its place. The page's file is otherwise never
  rewritten, and this is the one exception on record. Observed on the way,
  one load and two changes at once, so not a rule: the load that drew the
  beard sampled a window that also held frames of him left as the
  original, bearded; the load made again holds only its own frames, zoomed
  in, and drew none.
- **Where nothing is made is printed by the table, not listed by hand**
  (the owner, 2026-10-13, the same evening: a tool that finds the frames
  missing between pieces, run by itself by the common operations, and
  loud; and the ledger should hold what a piece leaves as the original).
  What prompted it: three places on the joined clip where they saw a
  subject go back to the original, each a stretch nothing was made on;
  and the lead's own hand-written list of gaps, which had missed several
  stretches the records already held. Built the same evening:
  `bench/piece_gaps.py` with its check, printed at the end of every
  `bench/delivery_table.py` run; the render ledger carries the frames each
  render leaves as the original, read from its capture. What it used to
  claim: the handover of that day carried a typed list of known gaps;
  it now points at the tool.

## 2026-10-12

*Written by the afternoon's lead session (mrfern) from the operative files
(`AGENTS.md` as rewritten this morning, the Open List's rows, the commits) and
from the morning lead's handoff; nothing here was heard first hand by the
writer, and the row or commit named is the nearer record.*

- **The whole fun video is in scope, not the first 1919 frames alone** (the owner, 2026-10-12, about 20:27Z, in the lead's terminal, asked whether the rest of the source should be worked: "having the whole thing in scope would be good cuz I'd like to just see if we can apply everything we have all the way through at some point"). Until now every shot table, plan, render and delivery was of the working copy `fun_0000_1919_24p.mov` (frames 0-1919 of the 6395-frame source `fun.mp4`; `data/fun/shots.json` says so), and nothing past 1919 had a cut, a track or a plan. From here: a second working copy of frames 1920-6395 made the same way, its cuts by scene detection as a starting table for the owner's who-is-in-each-shot answers, then the trackers' previews on it, then the loads in the settled form (the crop with the face points), after the first 1919 frames are done. Logged by mrforest, 2026-10-12T20:27:27Z.
- **The reference for a head (or head and hair) replacement is always cropped to the head, with the face points drawn on it** (the owner, 2026-10-12, about 20:12Z, in the lead's terminal, after the two cropped-reference tests on 1039-1060 were read: the crop alone (second_hf_1039_g, the lid marks on the head crop) opened her eyes as the face points on the crop (second_hf_1039_f) did, 0.57 against 0.61 of the original's fall, where the same marks at full-frame size gave 0.29: "We should probably still crop when doing just head (or head+hair) replacements anyway - always - and also add the eye dots to cover cases where they look left and right or around"). So the form from here: the body model cut to one box round the head (`bench/crop_reference_to_head.py`; the whole frame when the face already fills it, as on the lead's close-up), then the pose model's face points drawn on the cut video (`bench/draw_face_points.py`: the eye outlines with the iris, the brows, the nose; the mouth left out), and the text saying exactly what that video is. Before this the marked body model at full-frame size was the form (the owner's word of 17:35Z), and it never opened her eyes on seven renders. The owner's own words on the page at 20:18:36Z, under second_hf_1039_g after clicking 'keep this one': "All methods used here should be a standard going forward if it holds up for everything else. **ONLY** for face/head/hair/etc though. I think." So the form is the standard for face, head and hair replacements only, and it is provisional on holding up across the rest of the video. Logged by mrforest, 2026-10-12T20:12:17Z; the page words added 2026-10-12T20:19:10Z.
- **A clip's title on the local page starts with the load's name** (the owner, 2026-10-12, about 20:07Z, in the lead's terminal, after looking for 1039_f on the page and not finding it: "going forward can we label that as the title - what u refer to it as"). The name the sessions use for a render in the terminal and on the Open List (the load's name, the page item's `id`, e.g. `second_hf_1039_f`) is the first thing in the item's heading and in the page's nav, with the plain description after it. Before this the heading was the description alone and the name was nowhere the owner could read it. The page renders the id first (its rebuild of this evening); new items carry `title` as "<load name>: <description>". Logged by mrforest, 2026-10-12T20:08:48Z.
- **The body model against the source is figures, not a failure** (the
  lead's go, 2026-10-12, on the card holder's report of the one-command
  reference tool's first two real runs). `bench/draw_face_points.py` and
  `bench/draw_lid_marks.py` set how much the body-model video changes from
  frame to frame against how much the source does, at no slip and a frame
  each way, in the box where the body model ever is. They now print and
  record the three figures, the difference to the best neighbour, the box
  and the frames, and what the measure cannot see (anything else that
  moves in that box: a second person, the camera); `passed` is the
  read-back alone, that the file holds what was drawn on the frames it was
  told. A wrong clock is seen on the sheet and the strip. What it used to
  say: a neighbour ahead by `LAG_MARGIN` was FAILED with exit 1 and
  `passed: false` in the record; on the lead in the long kitchen shot, with
  a second person moving in the box, the neighbour was ahead by about a
  hundredth while the body model's outline lay best on the lead's own mask
  at no slip (the card holder's control).
- **An answer on a plain frame is a `present` entry of the shots record,
  written by the tool, and never a line of the cast file** (the lead's
  ruling, 2026-10-12, in its message answering the tooling session's dry
  run of 21:08:40Z on the owner's answers on frames 1920 to 6394;
  `bench/page_answers_to_cast.py --shots`). A cast line says which outlined number is which person and
  seeds a tracker; `none of the outlined people` on a frame with no
  outlines says nothing true about the cast, and `cast_file.py seeds
  --none-outlined empty` would empty a shot the owner said the subject is
  in. A subject is `yes` in a shot when any answered frame has them on it,
  `no` when every answered frame says not; `how` names the frames. The
  owner's words on a shot are printed and copied nowhere; they count over a
  button, so a person reads them against the entries before a preview is
  planned on them. What it used to be: this day's entry further down
  ("reach the cast file through one tool") had a plain frame's answer
  listed for `shots.json` by hand, and the tool required `--preview`.
- **A plan prints what each reference costs, as a render does** (found by
  the card holder at 20:28:53Z on the first plan built on the node that
  prints the cost lines: it printed none; the lead's order of the same
  evening made it a job). What it used to do: count in the render's
  conditioning loop only, which a preview returns before. A preview of
  the song node now asks for the reference chain as well as the source,
  which loads no model, and counts through the function a render builds
  its references with. A plan's count has no text encoder behind it and
  says so. Node code, landed with one restart.
  [`masked_v2v.md`](masked_v2v.md), piece 4.
- **A two-subject load is built: two stills and two references in one
  pass, the model told which is whose by the text alone** (the owner's
  click on Open List row z09, "yes, build it next", and their word of
  20:33Z to integrate what exists before the bigger batches, both as the
  lead relayed them; the three rulings on its shape are the lead's, given
  to the session that built it before it wrote any of it). A load's
  `subjects` is a list of two; the builder joins their tracks, parts and
  keep-outs, chains the two stills and wires the two references, so the
  first subject is `<Picture 1>` and `<Video 1>` and the second
  `<Picture 2>` and `<Video 2>`. Ruled: text-only binding IS the
  two-subject load by the owner's meaning, and a swap or blend of the two
  faces is what the first render is read for; ONE margin for the load,
  the larger need's, with the other subject's wider ring written down as
  the load's known cost and a second margin in the node named as what
  would end it; the capture reads the load as its first subject's, the
  subject with the larger region, and its flag about a region over
  another subject is answered there by hand. What it used to be: one
  subject a load, two passes over a shot with both people in it, and the
  second pass's keep-out of the first where the two overlap. Builder code
  only: no node code, no restart. No render had been made of one.
  [`masked_v2v.md`](masked_v2v.md), "Running a job".
- **A hair band is built into the job builder, and it is read by a session
  that did not build it** (the owner, 2026-10-12, as the lead relayed it to
  the session that built it: at about 18:52Z "the issue that caused the
  error - yes, we should build that"; at 20:34Z that the long kitchen shot
  is not worth testing alone; at 20:39Z that the builder does not test it
  or read its first render; at 20:41Z that the session which reads that
  render makes any change it calls for). A load's `hair_band` cuts a band
  out of the clothing keep-out under a head-and-hair part, its edge held
  still across frames (`bench/masked_job.py::hair_band`,
  `h3_config.MASKED_JOB_HAIR_BAND`, a case in `bench/check_masked_job.py`).
  What it used to be: every cell under the part was the keep-out's to give
  back, frame by frame. Builder code only, no node code and no restart. It
  is untested on a render as of its commit, and nothing here or in
  [`masked_v2v.md`](masked_v2v.md), "Running a job", says what it does to a
  picture.
- **More than one reference video into the Masked Source, and what a
  reference costs printed by the song node** (the owner's click on Open
  List row 16 at 19:35:27Z, "build it next, ahead of the zoomed load", and
  their question of what a reference costs; both as the lead relayed them
  to the session that built it, which heard neither first hand). The node
  takes a second wired video, `motion_video_2` with `motion_short_edge_2`,
  on the end of its inputs, and the song node shows it as the next video
  reference after the first, so a text can give each video something of
  its own to carry; the job key is `motion_file_2` with `motion_first_2`
  and `motion_edge_2`. The song node's report prints, for each window it
  encodes, what each reference costs. What it used to be: one wired video
  a load, and the cost of a reference printed nowhere in a render's
  report. Built in a worktree and brought in as one commit, since an open
  hunk in a node file holds the builder's queue; no render had been made
  with it, and it reaches the card at the server's next restart, which is
  the card holder's. [`masked_v2v.md`](masked_v2v.md), piece 2,
  `motion_video`, and piece 4.
- **`masked_v2v.md` was reorganized, and held to the clips, the files and
  the code it names** (the owner, 2026-10-12, in the lead's terminal,
  offering a session for it: "Its a mess right now"; and on top of the
  lead's brief: "Have them validate against actual clips where theyre
  referenced". Session mrprompt, from 17:56Z by the clock; this entry is
  that session's own). An index is at the page's top, and the order it had
  is in its last section, "How this page grew". No heading another file
  cites changed its words. Moved, word for word: the frame every masked
  text is written in, from the head of "The text: the fragment" to a
  section of its own, with "Say the least first" from under the prompt
  node; "Nothing is kept between runs by a key" and "Three things called a
  version", from under the Masked Source to a section after the pieces;
  "Seeing what the tracker and the masks did", to follow "Running a job".
  Cut: the delivery tool's paragraph, down to what a session needs and a
  pointer to `bench/assemble_delivery.py`'s docstring, which says the rest;
  the bullet "No kept mask", folded into the section that said it at
  length. Added from what the page already said: the table "What is known
  from renders, by form", and the owner's condition on the fragment beside
  "A pattern, not a paste". What the page used to claim and no longer does,
  each with the old words kept beside the new:
  - of `edge`, "Nothing has rendered with it", true when written on
    2026-10-09. The kitchen shot's picked render carries `latent cells`
    (`data/2026-10-11_gate_p4_0014_job/second_hf_0014_p4.render.json`);
    no render with it has been read beside one without.
  - a preview of the song node tracks the subject "the first time; the mask
    is kept and the render that follows tracks nothing". No mask is kept
    since 2026-10-10 (`masked_prompt.py`, its docstring).
  - the `MASK_VERSION` numbers were listed on `video_mask.py` too, which
    has had none since 2026-10-10.
  - "None of the captures or tools above carries a field for which person a
    file is about". `bench/capture_masked_run.py files` puts a `subject` on
    every row and file.
  - the composite's hold was "measured,
    `bench/results/2026-10-10_saved_windows_laid_again.json`". That file
    left the tree at a3aeab08; the page names the record's `.md`.
  - "Two more graphs since 2026-10-06, neither rendered yet", which now
    reads as of the day it was written.
  - the statement of who is in each window was pointed at `handover.md`,
    "After the close", a section that page no longer has; said beside the
    sentence, with where the fact lives now.
  The table of every clip and file opened, with the frames looked at:
  [`2026-10-12_masked_v2v_page_held_to_its_clips.md`](../../bench/results/2026-10-12_masked_v2v_page_held_to_its_clips.md).
- **Every comment the owner typed on the local page is a record to use,
  and where a typed comment and a clicked button disagree the words are the
  source of truth** (the owner, in the lead's terminal, 2026-10-12 between
  17:10Z and 17:14Z by the clock, having found the page's footer and asked
  that the 397 saved lines be used and reachable). The 11th's rule "words
  count over buttons" made general. What was done on it: a ledger tool
  (`bench/page_answers_ledger.py`) lays every line of theirs beside the
  record that cites it and lists the rest whole
  (`data/fun/verdicts/answers_ledger.md`); their typed words on the who
  page's rows went into `data/fun/shots.json` as shot notes (20 lines, the
  coverage tool prints them); their words on the joined clip's spots of the
  11th went into the verdicts record (34 lines); the 11th's part choices
  and kitchen pick are cited on the plan's subjects. Still uncited by a repo
  record after that: clicks on who-is-who tiles that a later click of theirs
  superseded (the cast keeps the latest), and their words on the Open List's
  own rows, which the list holds; an export of the list into `data/` is the
  next step so the ledger can count them.
- **Six answers on the Open List, read late by the lead** (the owner's
  clicks, the store's times): the look before a render is SHOWN and never
  refused (row c22, 15:51:48Z; the builder's refusal to plan or queue a
  load with no look recorded goes, a change queued in `bench/masked_job.py`);
  frames 0 to 1919 are finished first, the rest prepared alongside (z03,
  15:50:07Z); today's order A to E stands (z04, 15:50:31Z); the Sol sink
  default stays `exact_kv_and_all_rows` for every graph and the masked
  lane keeps `exact_kv_and_rows` (z06, 15:50:41Z; nothing changes in code);
  the four reds of the census are cut (z08, 15:52:12Z; queued); the
  two-subject load is built next (z09, 16:19:02Z; queued).
- **The look is shown and never refused, in code** (row c22; done by the
  lead on 2026-10-12, after 17:30Z by the clock; the owner's word on the
  patch, in the terminal: apply it, and remove the test if it is broken and
  causing friction). `bench/masked_job.py` used to refuse `build plan` and
  `queue` of a render for a load with no look recorded on its capture
  (`looked_at`, since 2026-10-10); now `look_or_shown` puts the matching
  look, or that none is recorded with why and the strip's path, into the
  build record and the printout, and nothing refuses on it. The check's
  cases on the refusal became cases on the record. The lane's rule that
  somebody looks stands as a rule of practice, not of the tool.
- **The fragment is a pattern, not a paste; a prompt is read in full** (the
  owner, 2026-10-12, in the lead's terminal at about 17:41Z by the clock,
  correcting the lead's relay of their 17:33Z word "make sure the prompt
  aligns"). Their words: "I didn't say every prompt has to use that
  fragment. I said thats how we solved one specific one. Its not a copy
  paste, its a pattern that worked. And we need to follow the vendor guide
  prompting for refs. If our ref video is a gray mesh dude with a shape on
  him, part of that fragment works, because its following those things. So
  for the mom with the overlay of the mesh and the sapiens2 data for
  capturing eye/expressions and the mesh for turns, you can keep it simple
  and refer to that part of the prompt. You dont need to get super specific
  about the scene itself. You let the ref video and frozen video/audio guide
  it by the prompt." And: "every prompt needs to be read in full, not
  summarized or heard secondhand. If youre trying to build or diagnose a
  scene - you better know what inputs went into it, what they look like,
  and what the prompt says EXACTLY." What the lead had relayed at 17:35Z
  (the four places verbatim on every load) is withdrawn; the two loads
  built on it (second_hf_1039_b, second_hf_0859d) stand, since their
  reference is the marked mesh and those lines fit it. The marks stay on
  every load of her. [`masked_v2v.md`](masked_v2v.md), "The text: the
  fragment", carries the ruling beside the fragment. The owner, minutes
  later, restating the frame every masked text is written in: "we're
  generating faces (and hair i guess) from the perspective of a FROZEN
  video in h3. when youre making a prompt, h3 has no idea whats around
  that little hole/mask of noise we're creating. All it knows is we're
  generating something from a ref image and from a ref video. We're not
  video editing. We're using references." And what the marked video IS:
  "the sam3d body mesh + the sapiens2 parts" (`masked_v2v.md`, "How it was
  made: the nodes and the code"). And at about 18:00Z, the condition on
  the pattern, so that nobody makes it a rule: "don't take that fragment
  of text as a 'paste this onto EVERY prompt' verbatim. If you use a body
  mesh and you use a sapiens2 overlay data, and youre just doing face
  replacements. it makes sense. If you're doing something else, it doesn't
  anymore, because the reference video may have changed, or maybe its not
  about the face. In our video, I think its always about the face and
  head... but just an fyi. Don't make it a rule that ends up getting
  inserted into things that causes future debugging hell." No tool inserts
  those sentences (the ruling of 2026-10-11 that the form stays a
  documented practice and is not put into code stands; a grep of the
  code for its words finds none); a session writes each load's text from
  its frames and its own reference.
- **No video clip under `data/`, the tools included** (the owner,
  2026-10-12, in the lead's terminal at about 17:39Z: "All videos we agreed
  go in output folder Video/fun somewhere"; AGENTS.md has the rule).
  `bench/capture_masked_run.py video --load` wrote a load's strip beside
  its capture under `looks/` since 2026-10-10 (42 strips by 17:40Z); it now
  writes to the share's `Video/<video>/looks/<capture>/` (`strip_home`), the
  record beside the capture naming the full path, and the 42 were moved
  with their records rewritten. The local page's own viewing copies under
  `data/fun/poc/media` and `data/fun/review/media` (252 files) are the next
  move, after the long shot's reading. The tracked masks (`.mkv` under a
  capture's `subjects/`) are the tracker's data and stay unless the owner
  says they count.
- **What each subject carries is unchanged and holds for every shot**: the
  lead, the face only; the second subject, the face and hair only; the third
  subject, the face only (the owner, in the lead's terminal, 2026-10-12
  between 16:44Z and 16:48Z by the clock, restating the defaults of
  2026-10-11 after the kitchen scene's whole-person question). No shot
  departs from these by region; `data/fun/plan.json` carries the words on
  each subject.
- **Her face and hair for everything; no whole-person replacement of the
  second subject anywhere; the earlier renders are not reused, the shots
  are rendered fresh with the tools, and the faults seen are fixed first**
  (the owner, in the lead's terminal, 2026-10-12 between 16:43Z and 16:46Z
  by the clock, on being shown that the 10th's whole-person piece of the
  kitchen scene could stand). This supersedes the line below it from a few
  minutes earlier: the 10th's piece is not used on 604-1009 either. The
  first fault to fix is the mottled patch at her shoulder (the margin round
  her hair redrawing her blouse at every latent step): the next loads keep
  her upper clothing out of the region (load B of the cutaways; the same
  keep-out on 604-1009 and her tail shots); what that does to her long hair
  at the shoulder line is read on the first render. This is the go for
  those renders, written in each job's note with these words and times.
- **Her whole-person render stands on the kitchen scene 604-1009** (the
  owner, in the lead's terminal, 2026-10-12 at about 16:40Z by the clock:
  "the glitching was not in that segment"). What it narrows: the 2026-10-11
  ruling that the second subject's whole-person renders are not used, which
  rests on the segments where she glitched: the page's rows 2a and 2b,
  1015-1085 and 1042-1049, the two shots after this scene (the owner's
  correction between 16:41Z and 16:43Z by the clock; an earlier line here said 14-167, which was
  the egg, not a glitch). The owner's reading of the cause, a hypothesis:
  a whole person forced into a small mask glitched there, and face and
  hair should not; the face-and-hair render of 1010-1060 is its test. On 604-1009 the 10th's piece
  (`mom_kitchen_mesh`, in that day's table a with a patch on 752-780) is a
  usable form beside today's face-and-hair render; the owner compares the
  two on the same frames before the plan's region for the stretch is
  settled. `data/fun/plan.json`, the 604-1060 departure, carries the words.
- **A masked load's text is written from the model's side: it generates
  the hole, and everything frozen round it steers it** (the owner, in the
  lead's terminal, 2026-10-12 between 16:16Z and 16:20Z by the clock; first
  said 2026-10-10 at 14:09Z, where the postmortem of that span holds it and
  the day's texts went the other way). The
  model is not editing a video it can see; it is generating new pixels in
  the masked region, steered by what its training makes of the frozen
  frames and the sound round the hole, by the prompt, and by the references.
  So the text gives it enough to infer what happens there: each reference
  asset described and what of it is preserved, partially preserved or
  transferred (the guide's `retention_analysis`), what is generated (here a
  face, or a face and hair), and the movements and expressions from the
  body-model video with its shapes. Now written at the head of the masked
  lane page's text section ([`masked_v2v.md`](masked_v2v.md), "The text:
  the fragment"; moved later the same day, when the page was reorganized,
  to its own section "The frame every masked text is written in"), which
  had the fragment's rules and not the frame; the same words
  are the reason a two-subject load (row z09) is written as two pictures
  each described, one union region, one motion video with both figures.
- **Windows align with cuts, and a new window never starts within about a
  second of a cut** (the owner, in the lead's terminal, 2026-10-12 between
  15:42Z and 15:50Z by the clock, answering Open List row z07: "if you need to make shorter windows,
  thats fine. Just try to align them with cuts. Dont start a new window a
  second right before a 5 second cut is about to make a new cut a few frames
  later"). A standing rule for the builder's window choice: shorter windows
  are allowed; a window boundary sits on a cut where one is near; a window
  that would begin within about 24 frames of a cut is not begun there. Loads
  are one shot each today (row c24), so inside a shot it binds the choice
  near the shot's end and any long load that spans cuts. The builder
  (`bench/masked_job.py`) does not read the cuts (`data/<video>/shots.json`)
  when it chooses windows yet; that change is on
  [`next_steps.md`](next_steps.md). The same answer gave the go on the
  continuation load `second_hf_0859c` (949-1009 from the stored first
  window of `second_hf_0604`, which failed after its first window at
  15:13Z), until then the lead's go on the owner's 14:57Z sentence.
- **`AGENTS.md` is the operative rulebook, about 110 lines, and every rule in
  it is one a session cannot work around without noticing** (the owner's
  word; commit 7bac71e9). What it used to be: the longer file with the
  routes, which now live only in [`index.md`](index.md). The owner's
  housekeeping rulings of the morning are in it: two sessions during the day
  (one holds the card and builds, one reads cold and builds tooling; a third
  only as a blind reader of a finished result, a subagent counting as one);
  handoff over compaction, staggered, never both sessions at once; one
  inbox (the Open List's box) answered done / queued / parked; the terminal
  only for a go or an answer to a session's question; a rule easily worked
  around is friction, not a rule, and goes.
- **Green and red are for a hard failure, everywhere, not only in what is
  built from here on.** The 2026-10-11 line below left the scope open; the
  owner closed it this morning (to the morning lead, in the terminal): a
  check or a gate level that goes red on a matter of degree is to print its
  figure with its context instead. The code cuts are owed and none is made
  yet; `AGENTS.md`, "Green or red only for a hard failure", is the rule. The
  look before a render (Open List row c22: refuse, show, or drop) is still
  the owner's to answer and is not changed by this.
- **No video under `data/`**: a capture's mask copies and every video file
  go under the share's `output/Video/<video>/`; the data (tables, points,
  records) stays local under `data/<video>/` (the owner, this morning; the
  rule is in `AGENTS.md`, "A record under `bench/results/`"). What it used
  to say: the 2026-10-11 handover left the mask copies' home open.
- **The review page's code goes into the tree as `bench/review_page/` when
  it is built** (Open List row t72, decided this morning); until then the
  page lives untracked under `data/fun/poc/`. The old boards are retired;
  the Open List (the owner's page with Claude) is the one place a decision
  is asked and answered, and the local page on 8765 is where the owner
  looks at pictures and clips.
- **Plan state has three homes and no more**: what a video still needs is
  `bench/shot_coverage.py` run on `data/<video>/shots.json`, `plan.json`
  and the delivery table (regenerated, never typed;
  `../../bench/results/2026-10-12_fun_shot_coverage.md` is the first
  record); the day boundary is [`handover.md`](handover.md) and the top of
  [`next_steps.md`](next_steps.md); decisions are this page. What it used
  to be: Open List row z01 was a typed reading of where the clip stood, by
  stretch; it now points at the tool and the record (row rewritten
  2026-10-12).
- **The owner's clicks on the who page reach the cast file through one
  tool** (`bench/page_answers_to_cast.py`; commits b20de77c to 2f035c5b):
  the latest button per numbered person stands, a `not sure` stores nothing,
  typed words are printed beside the tile and never parsed, and an answer on
  a plain source frame (no tracker outlined anybody) is listed for
  `data/<video>/shots.json` by hand, not stored. A click is about the frame
  shown, never the whole shot. The who page shows a long shot's first- and
  last-quarter frames plain as well, so a late entrance has a frame to be
  said on (f5d5c417); a renders page on the share lays every render in frame
  order with the owner's verdict lines under it (4ee6d450).
- **The sessions' memory folder was cut from 55 notes to 16** at the owner's
  word, after a fresh reader's friction audit: it holds the owner's
  preferences about how to work, facts about things outside this repo and
  how sessions coordinate, and never project state.
- **Corrected claim: every time the morning lead typed as Z from about
  14:00Z on this day is about five hours ahead** (its Open List write-backs
  on rows z01, z05, z06 and g01, its reply in the inbox, the note times in
  `data/fun/plan.json`): the local offset was added to a time that was
  already UTC, the slip the 2026-10-11 postmortem's section 1.7 names,
  made again by the session that wrote it up. The stores' own timestamps
  count; the machine-written times in the job records
  (`data/2026-10-12_second_hf_0604_job/*.queued.json`) are right.
- **The owner's eye on playback stands beside the frame readings and can
  overturn them** (the owner, 2026-10-12, 21:25Z to 21:28Z by the clock, on
  the three hall pairs `lead_g_1222_fp_cv`, `lead_g_1156_fp_cv` and
  `lead_g_1093_fp_cv` against the lid-marks renders of the same shots: the
  readers from frames and the pose figures had called the eyes a draw; the
  owner on playback saw the gain in motion on all three, eyes, mouth and
  gaze, and kept them: "keep those they're good to go"). From here a render
  is told with both: what the frames and figures say, and what the owner saw
  in motion; neither argues the other (`data/fun/verdicts/verdicts.jsonl`).
- **604-948 stays as `second_hf_0604_b`; nothing is rendered there** (the
  owner's click on Open List row 17 at 21:43:20Z and their words "leave it
  be"). The two-subject pass could not be rendered on that shot (the lead's
  face under the capture's floor on 151 of 345 frames; `data/2026-10-12_second_lead_0604`),
  and the two statements it rested on (row 13: made once in the form that
  wins; 20:34Z: not worth testing alone) pulled apart; the owner chose the
  piece that exists. The lead's piece there stays the 10th's `face_kitchen`.
  The two-subject load's first render moves to the hall shot 1222-1305.
- **The owner's go for the night: keep pushing through the video, the rough
  edges later** (the owner, 2026-10-12, 22:00Z by the clock, in the lead's
  terminal, leaving for the night: "Just keep pushing as far as you can get
  in the video and we'll see where the rough edges land later"; at 22:19Z:
  "have mrham wrap up when convenient and you and mrsnow finish up"; at
  22:21Z: mrham hands off to one of the two). So from 22:00Z no render waits
  on a row; form choices are the lead's on the record, reversible by the
  owner's look; rows written for the owner stay open (18 and 19).
- **Her kitchen opening 14-154 stands in the one form, `second_hf_0014_p4_headref`,
  on the owner's eye** (the owner on playback, 21:58:32Z: "looks good btw",
  before the readings were told; `verdicts.jsonl` line 169). What the frames
  say, both readers blind to each other (`data/2026-10-12_gate_p4_0014_headref/reading_mrfancy.md`,
  `reading_blind/`): outside the eyes it is p4's picture; the eyes follow the
  original in time on 14-80 (p4 dropped them twelve frames early) and open at
  the camera on 81-154 while hers stay down (p4 kept them down). So on this
  shot the drawn outline carried WHEN the eyes move and the marks carried HOW
  FAR the lid is down; one seed each. Row 18 asks the owner which stands in
  the end; p4 is kept beside it.
- **The hair band's first render drew nothing in the band, and the band is
  carried into her other face-and-hair loads** (`second_hf_0278_b_hr_band`,
  the lead's ruling, 2026-10-12, 22:12Z by the clock, on the owner's word of
  20:37Z). The band opened 7-46 cells a frame under her jaw that the pair
  kept out; the model drew the original's apron there (about 1 grey level
  from the pair; no patch, seam or flicker beyond the floor, by two readers
  and `bench/read_hair_band.py`'s first run). Her hair was drawn as a bob
  that never reaches the band, shorter than the pair's, so the band is
  untested as a fix for the 693-703 fault, which needs a render where the
  hair is drawn long. The pair without the band (`second_hf_0278_b_hr`) was
  planned and not rendered: a like-for-like would show the same nothing
  (`data/2026-10-12_second_hf_0278_b_hr_job/NOT_RENDERED.txt`). The eyes on
  that render: lowered on 278-289 where the marks' render had none lowered,
  open at the camera on 290-293 in both.
- **The owner's typed words on the 1920-6394 shots were parsed and confirmed
  before any preview is planned there** (the owner, 2026-10-12, 22:15Z by
  the clock: "we need to have this parsed and confirmed by someone before we
  get further along"). A cold reader's parse (`data/fun/who_1920_words_read.md`)
  and mrham's confirmation as the second reader (its section at the file's
  foot, 22:17Z): no shot disagrees with the buttons; 23 shots carry a place
  usable to seed a tracker; nine lines are unclear and are Open List row 19.
  The caution for the seeds: 19 of the 29 seed rows carry the shot's first
  frame, which the owner was never shown; only 10 rows on 7 shots carry a
  frame the owner named.
- **Her three cutaways and the lead's 1466-1534 are rendered in the one form;
  the hair band's three renders drew nothing in the band** (the lead's
  readings with a blind reader each, 2026-10-12, 22:12Z to 23:05Z by the
  clock: `data/2026-10-12_second_hf_0415_b_hr_band/`, `..._0484_b_hr_band/`,
  `data/2026-10-12_lead_fp_1466/`). 415-437: the drawn form carried the
  lowered lids at the start and the opening's timing (0.52 of the original's
  fall where the marks' render carried 0.08) and neither form carried her
  widening or her lowering at the end; 484-515: the lids open at the camera
  on every frame where hers are down at the egg (the marks' render down for
  five frames, then the same); 1466-1534: the mouth sings on the sung phrases
  and the eyes narrow where hers do (0.55 of the fall), the 10th's render of
  the same frames having a still mouth. The band opened 8-23 cells a frame
  on 484's bib and the model drew the apron; the one strand that reached a
  strap tonight was drawn by the no-band pair of 484 and held still. Each is
  her or the lead's piece on its shot on the standing goes; which of her
  pieces stands on each cutaway in the end is the owner's (the eyes are the
  rough edge, named on the page).
- **The night ends at a stopping point: no load is built after 23:03Z; the
  last card job is a joined clip of 0-1919** (the owner, 2026-10-12, 23:03Z
  by the clock, in the lead's terminal: end the night when the Open List and
  everything local is set for tomorrow; "an end of day joined clip across our
  best shots of the day in a linear frames sequence in one video file in
  output folder Video/fun"). mrsnow, back after a usage-limit stop at
  22:48Z, assembles it from mrham's dry run; the lead's 1061-1092 (ruled:
  the plain body model, ending at 1079) and 1306-1402 (ruled: the capture on
  the load's own frames, ending at 1396) are tomorrow's first loads, their
  rulings in their captures' `looks.json`; the servers are stopped after the
  join. The first two-subject render waits for a day the owner can look.

## 2026-10-11

*Written by the session that took the build over (mrfire) from what the
session the owner spoke to (mrhop) took down the same day; where this and
that session's record disagree, its record is the nearer one.*

- **Which perception models are on disk, and which code loads each, is now
  one table** (the owner's ask; session mrice):
  [`meta_perception_models.md`](meta_perception_models.md), "What is on
  disk, what loads it, and what to try next". What it used to say: that page
  said in three places that nothing here runs the Sapiens2 keypoint model. A
  one-off ran it on eight frames this day
  (`../../bench/results/2026-10-11_sapiens2_pose_first_frames.md`); each of
  the three sentences keeps a dated note in place. Read from the code and
  not from a run: the matting model in the store is left off on purpose
  (the comment beside `h3_config.SAPIENS2`), because the Masked Source reads
  the hard part mask and not a matte, so it is not the next model to wire.
  New on the page: the depth and geometry models that could say which of two
  crossing masks is in front, on disk or not, and why MoGe had come up
  before (the body model's camera, not depth). Nothing was run for it.
- **Segments first, designed as one thing with the planning data and with
  objects** (owner, after a whole-person render of the opening shot redrew
  about half the frame, where the render of 2026-10-10 that replaced only
  the face and hair of the same frames was the close one). What it
  replaces: a job file that says, before any picture exists, whether a
  subject is taken whole or as one fixed set of part ticks
  (`carried: track|held`, `h3_config.SUBJECT_PARTS`), and three fixed cases
  of text in the Masked Prompt node. Now, as a direction and with nothing
  built: every subject of the input video is cut into its most granular
  parts on every frame, every reference still is cut the same way, and
  what is replaced, kept out or left on a shot is a choice a person makes
  from a view of those parts, so that whole body, face only, or hair and
  clothes is data handed in and not a code path. Objects are in the same
  design and kept apart from people, with what the preprocessing can tell
  of how they relate (held by, in front of, passes, touches). The planning
  data is designed with it and not after: per video, in source frames, who
  is in each shot, what is replaced in each shot, and the text of each.
  **Before any of it changes code**, the close render of 2026-10-10 is
  made again through the tools as they stand, unchanged; what that run
  shows the tools cannot say is the first input to the design.
- **This clip's defaults for what is replaced** (owner): the lead, the
  face only; the second subject, face and hair; the third subject, the
  face only, keeping his own hair. A default covers a subject's shots
  until a shot is changed.
- **A clip handed to the owner to judge is stacked with its masks and
  carries frame numbers, on one local page, with its prompt shown**
  (owner, on reviewing a joined file by the second it played at). The
  source on top with each subject's mask, part and margin in its own
  colour and its label, the render under it, the source frame number
  burned into both. The tool is `bench/capture_masked_run.py video`. A bare
  render is not handed over, and nothing of the owner's media goes to a
  hosted page.
- **A working database may be a preview build, and is short-lived**
  (owner): what a tool needs to query is read in place from the tables
  the tools already write, a database earns a file only where something
  needs it, and any of it can be moved later. What a person submits on a
  page stays one appended line a submit in a plain file.
- **One local page** (owner, later the same day): the page a second
  session had made for looking at spots, numbered pictures and segments
  (`data/fun/poc/`) is the one page; the review clips with their notes and
  prompts are shown on it from `data/fun/review/items.json`, and the page
  first made for them is stopped once they have moved. What the owner
  saves on it is one appended line a save.
- **A video's things live on the share, under the output folder, in one
  folder a video** (owner): `Video/fun/` for this clip, holding its data,
  its videos and its frames, so the owner can open them. What it
  replaces: the one folder a video under the repo's untracked `data/`
  (`clip_folder.py`, `data/fun/`), and through it the sentence in
  `AGENTS.md` that puts data under `data/`. Nothing has moved yet. Two
  things to square when it does: the share is a network mount, so plain
  and appended files only and no database file with more than one writer
  there; and the contact sheets' own rule of 2026-10-10 gives them one
  place, `contact_sheets/`, so a frames folder on the share is either
  that place moved or a second place, which is the owner's to say.
  `clip_folder.py` is the one function that says where a video's folder
  is, so the move is made there.
- **A text is written by a session, to the vendor's guide, and says what
  is true of the job** (owner): `vendor_guides/ref_en.md` and
  `docs/prompting.md`; the job is a generation from reference stills that
  follows a motion reference video, and is not an edit of a video. What
  SAM is asked follows `docs/wiki/sam3_prompting.md`. *A reading, not the
  owner's ruling:* the guide asks for each shot described in full, and the
  rule of 2026-10-10 says the least and nothing that is not seen or
  heard; the session that took this down reads the two together as
  everything that is seen on the frames, as fully as the guide asks,
  nothing unseen or unheard, and no sentence about what stays or is kept.
  The text the Masked Prompt node wrote for a whole person after
  2026-10-10's cuts names no place and no framing, and its first render
  drew a face filling the frame.
- **A mask that is changed is shown as changed; the plan is stated once a
  video, and a shot that departs from it says why** (owner), with a
  session's reason or the owner's.
- **The second look is against the data and the text read whole**
  (owner), by someone who knows who and what is in the shot and whether
  the right things are replaced; never against a paraphrase. A subject
  known to be in a shot with no mask there is a problem.
- **The first target is the clip's first 600 frames**, every shot through
  the new record and each clip handed over stacked (owner). Before it:
  the close render of 2026-10-10 made again, then one design page for
  the owner's yes.
- **Of the perception models on disk that no code loads, the Sapiens2
  pose model is wired first** (owner), because it serves head movement
  and expression, and it is tried on a handful of frames where the answer
  is plain by eye before anything is built on it. The depth and audio
  models wait until the design needs them. Which models are loaded is
  read from the code and a server log, never from this line.
- **The close render of 2026-10-10 was made again through the tools as
  they stand** (the session that took the build, the same day; the
  owner's look is still to come). The whole load went through
  `bench/masked_job.py` with nothing posted by hand: that day's mask files,
  mesh video, text and seed handed in as named files, and its
  hands-arms-and-held-things mask taken by the capture as a subject of its
  own, which a load's `others` can name. Read from files: the region the
  node planned and the region the render saved are that day's on every
  latent step and frame; how close the window's saved latent and the
  pictures are to that day's is in the record,
  [`2026-10-11_close_render_made_again.md`](../../bench/results/2026-10-11_close_render_made_again.md).
  Two things it showed
  the tools cannot do: no kept code writes the two special masks that
  render used (a cleaned face-and-hair mask and the keep-out), so they
  cannot be made again; and the capture takes two runs under one name
  without a word and prints every flag twice. The inputs, the captures
  and the figures are under the untracked `data/2026-10-11_fun_gate_inputs/`
  and `data/2026-10-11_gate_hf_0014*`. *Corrected within the hour: "they
  cannot be made again" is wrong. No kept code writes them, and both are
  recovered exactly from what that day's preview saved; the rule for
  each is in the record. What stands is that the rule lives in no tool.*
- **The owner watched the remade render beside the one of 2026-10-10,
  stacked on the local page, and kept it** (their saves on the page, read
  whole by the session that made the run): it looks good, it looks like
  the older one, they cannot say which is better, and of the two the one
  made by today's tools is the one to go with. So the condition set
  before any segments-first change is met, and that load's reference is
  today's render. The button chosen of three was "not sure"; the reading
  of it is in the record.
- **Later the same day, the owner, as the session they spoke to took it
  down:** objects are built together with the rest and not after it. The
  folder on the share, `Video/fun/`, is for new work only and work is
  built there from now; the dated folders stay where they are. (That
  folder already held unrelated renders of September; how the two are
  kept apart is with the owner.) The second subject is not replaced
  whole: her whole-person renders glitch, and for her it is face and hair
  only, which the owner, having watched the face-and-hair render of
  2026-10-10 with the mesh on the opening shot, called good and to be
  kept. Where she is small behind the lead (frames 1010 to 1038) the
  owner expects face and hair to do better and that it will need the
  pose, depth or body model as well; nothing has been tried there. The
  two delivery files that had been marked not to be used were removed
  with their check records on the owner's word.
  *Two cautions added the same evening, from a session that read the
  owner's own rows: "the second subject is not replaced whole" is the
  wording of the session that took it down, and stronger than what the
  owner said, which was about the clips then on the review page; on
  2026-10-10 they had praised two whole-person renders of her. And her
  default is NOT settled: on the page the owner later chose her face
  only and wrote that on each of her shots, where this page's line above
  and the render they kept say face and hair. It is put to the owner;
  nothing is built on either reading until they answer.*
  *Answered within the hour (owner, taken down by the session that has
  the page): the plan is the second subject's face AND hair, and the lead
  and the third subject face only; the face-only answers on the page
  were a slip. And where a button and the words typed beside it
  disagree, the words count. That settles the button on the remade
  render as well: its words are the verdict.*
- **Three verdicts of the owner on playback, on renders of 2026-10-10**
  (taken down by the same session). Where the lead sings, the text says
  so: on the close-up of frames 1575 to 1729 the render whose text has
  the one sentence that she sings shows her singing with the replaced
  face and was called good, and the render of the same frames without it
  has the right face and a mouth that does not move; the lead's piece for
  frames 1222 to 1305 looks right and its mouth does not move either. So
  the sentence is wanted on a load that replaces the subject who is
  heard, and is no longer only a thing to test. It is one load and one
  seed with the sentence; what it does on other loads, and where the
  track has no voice, is not known. **The zoomed render stays**: on the
  street shot of frames 438 to 483 the owner saw a gain in the render
  made zoomed in on what is replaced over the whole-frame one, so
  `render_region: part` and the delivery tool's lay-back are kept, where
  the standing question of 2026-10-10 had them going if no gain was
  seen. It is the method on the table for the small faces. One clip, one
  seed; the capture and its gate cannot read a zoomed load yet.
  *Corrected the same day, on a second session's read of the three
  graphs, checked by diffing the texts: the two clips the owner compared
  differ by TWO sentences, not one. The text without the singing
  sentence also holds a sentence about a telephone handset that "stays
  as it is", which the singing text lacks. The pair that differs by the
  one added sentence is the singing render against a third render of
  those frames, made with that handset sentence removed and nothing
  added; all three had the same audio, region and seed. And the added
  sentence is three things at once: that she sings, the handset held to
  her ear, and her mouth opening wide on the sung phrases. A test that
  carries "the singing sentence" to another load says which of the three
  it carries.*

- **A text is written from the side of what is generated, and it names
  what each reference gives** (owner, 2026-10-11 evening, said to the
  session that has the page while the two tests on the kitchen shot were
  read; taken down in substance). What is generated is what lies inside
  the mask, beside a picture and a sound that are kept as they are, with
  its reference stills and videos; the text is the caption of that. A
  reference the text does not speak of is one the model cannot know to
  use, so the text changes whenever a reference does. And movement and
  expression are to come from ONE motion video: the body mesh with the
  pose model's points drawn on its face, with a text that says both are
  taken from it. Until now the text of a load with a mesh took only the
  head from it, and the mesh's face is blank, which fits what the two
  tests found
  ([`2026-10-11_kitchen_shot_head_and_eyes_two_tests.md`](../../bench/results/2026-10-11_kitchen_shot_head_and_eyes_two_tests.md)):
  the head follows and the eyes do not. A direction; the overlay is not
  built, and whether a sentence about her eyes is tried before it is the
  owner's to say.
  *Corrected the same evening: the mesh's face is not blank. It is a
  neutral modelled face with sockets, a nose and a mouth, which never
  changes; seen on the sheet drawn to check the overlay.*
- **What the owner ruled while the overlay was built and rendered**
  (2026-10-11 evening, to the session that has the page and to the one
  that has the build; taken down in substance, each on the kitchen load
  of the record above):
  - **No softened start** on these loads. Asked for and dropped within
    minutes; a builder key written for it was reverted before any commit.
  - **No sentence about her eyes, not even as a render to compare
    against.** It does not scale: a sentence is written for one shot by
    whoever read its frames, where a reference is made by a pass over the
    clip and named once. And a text that says what her expression is
    leaves a render unable to say whether the text or the reference did
    it. This also covers a clause that only relates her eyes to the
    reference's, which the first three overlay texts carried; a second
    reader caught it before they were read.
  - **The overlay is the one thing built for her eyes**: the body-mesh
    video with the pose model's lid points drawn on its face, as the one
    motion video of the load.
  - **The form of the text for it.** Say what the reference video is, as
    the model sees it, then that the subject follows its movements and its
    facial expressions. Nothing more: no head turns, no named feature, no
    statement of what she does. Where the reference carries marks, the
    line that describes it says what they are for in the reference (part
    of its facial expression) and that its look and the marks themselves
    are not drawn; the marks are given no colour.
  - **The reference video's retention line** may be what was missing:
    the vendor guide (`vendor_guides/ref_en.md`, its retention markers)
    has a line say what of a reference is preserved, and the texts so far
    did not say it of the video. One render tries it; nothing is settled.
  - **Segmentation and depth are to be shown to the model, as layers in
    that one reference video**, not as further reference videos and not
    only as planning data. They are added one at a time, each a single
    render against the kept one: the lid marks; the pose model's own
    drawing of the face points; the part colours; depth shading. The
    line describing the video changes with each.
  - **What times a sentence**: the kept sound and the kept picture round
    the mask. A sentence states a fact and the model places it where the
    track and the picture agree with it, which is how the one sentence
    that the lead sings worked. So a sentence about something neither the
    sound nor the surrounding picture marks has nothing to time it. A
    reading from one load each.
  - **After this load, the same form goes to the other shots** of the
    first target, one shot of the lead first; relayed by the session that
    has the page and not yet said to the build. One thing found while
    preparing it and put to the owner: a load rendered zoomed in on what
    is replaced takes no motion video (`video_mask.py`, the note at
    `RENDER_ZOOM_REPLACED`), and the lead's render the owner called the
    best of its stretch was a zoomed one.
  What the overlay renders showed is not in a record yet. The owner sees
  light dots in her eyes on playback of the one with enlarged marks; no
  session has placed them on stills or by a count, and the owner said not
  to dig for them. Whether a wording takes them away is what the renders
  with the form above are for.
  *Corrected within the hour: the dot is not in her eyes and not from the
  overlay. The owner pointed at it on a still: a small warm point on the
  side of her nose, which a second reader then found on every frame of
  every render of this shot, the kept one and the 10th's included. The
  owner closed it. So no render of the form above is read for it, and the
  render that tries the retention line is read for her eyes and head like
  the others.*
- **What the owner ruled at the close of 2026-10-11**, on the local page
  and in three sessions' terminals; each line was read in the page's own
  file of answers before anything was queued on it. Taken down in
  substance.
  - **The kitchen shot's pick.** The owner judged all eleven renders of
    it on playback and picked the one with the wording they had chosen;
    they could not tell it from the two other plain-text renders, and
    liked the enlarged-marks render before it as much. Their eye and the
    pose model's table sorted the eleven the same way
    ([`2026-10-11_kitchen_shot_lid_marks_on_the_body_model.md`](../../bench/results/2026-10-11_kitchen_shot_lid_marks_on_the_body_model.md),
    "Added later the same evening").
  - **Other examples before more is read into one clip.** Two shots, the
    lead's singing close-up first and the third subject's face after, in
    that wording word for word; and a shorter wording ("the movements and
    the expressions") to be tried as well, if the form works. This
    answers the line above that had it second hand.
  - **Where the reference's wording lives: a documented practice, not
    code.** In the wiki, with the fragment alone, one whole text as an
    example that is plainly not for copying, what the reference video
    consisted of and how it was made, and why it is thought to have
    worked; said to be from one case; and changed when it turns out not
    to be true. It is
    [`masked_v2v.md`](masked_v2v.md), "The motion video as a body model,
    with marks drawn on its face". Each load keeps its own copy of the
    lines in its text file.
  - **The day's tools are tracked, and the day's way of working is
    written down where the next session starts.** The marks on a
    body-model video, the pose pass along a shot and a render read
    against its original become tracked tools with a check each, written
    by one session. The builder's motion choice and a generated workflow
    for the form wait until it has held on two shots: decided by the
    sessions and not objected to. [`handover.md`](handover.md) says where
    every folder, tool and one-off is.
  - **The ComfyUI server is stopped when the day's card work is done.**
    Stopped that evening by the session that held it.
  **And the form did not repeat.** On the close-up her mouth sings and
  her eyes read as they do in the render of 2026-10-10 that was given no
  motion video
  ([`2026-10-11_lead_closeup_lid_marks_did_not_carry.md`](../../bench/results/2026-10-11_lead_closeup_lid_marks_did_not_carry.md)).
  The shorter wording and the third subject's shot were built and held,
  since neither would say why. One render is put to the owner for the
  morning: the same close-up with the marks at three times. So the
  wording above is the owner's pick for a shot where it worked, and is
  not yet a rule for another.
- **The day's postmortem, written from the six sessions' transcripts by a
  session that took no part in the day** (the owner's ask; session mrmuse):
  [`2026-10-11_span_fun-lid-marks-one-shot.md`](../research/postmortems/2026-10-11_span_fun-lid-marks-one-shot.md).
  Nothing is decided by it. It ends on nine small things to pick up, in an
  order that is the owner's, and it marks each of the previous postmortem's
  nine as done, half done or not touched (dated annotations in that page).
  What it says the day showed: the way of working held, and a figure over a
  whole shot was still told as a result before the second reader's split was
  read; the form went to a second shot that changed several things at once;
  and the ideas that slipped were single arms that make a result readable
  (a second seed, one change on the same load, the vendor guide's form on
  the large marks, a zoomed load that keeps its motion video), each deferred
  as "next" more than once. What this page used to claim: the entry above
  says the hold on the builder and the generated workflow was "decided by
  the sessions and not objected to"; the rows show the owner was not asked,
  and asked again at the close that the code and tooling be left in a good
  spot. [`handover.md`](handover.md) carries a dated note of where it is
  wrong or short.
- **What the owner said the bench tools are for, after the day had
  closed** (2026-10-11, late; three messages typed to the session that
  has the page, read in the page's own file of answers; in substance):
  - **A tool gives data and its context and leaves the reading to
    whoever runs it.** It does not print a verdict. The three tools
    written that evening print figures with what they were measured on
    and the starting value each is held against, and say no more.
  - **A standing check is worth having only where it tells the runner
    something before time is spent.** Asked of a comparison between a
    fitted source and a render's untouched pixels after every render:
    no. What meets it: a capture saying up front whether the source's
    shape is the canvas's and how much the loader cuts. Not built.
  - **A figure is a measurement and not a judgement**, as a tape says a
    length and not whether it is the right one. Being near the original
    means nothing in isolation, a higher or a lower figure is sometimes
    what is wanted, and the scales are not known yet: what the day has is
    single points on one or two shots ([`handover.md`](handover.md) lists
    them), not a scale for any measure.
  - **Green and red are for a hard failure**, a thing plainly broken
    that stops the work (a graph that names a node that does not exist,
    a missing file, an import that fails). Anything of degree or of
    judgement is given as data beside its context and left to the
    runner; a bare score with nothing beside it helps little. Said when
    told that a green check does not mean the thing was exercised. **Its
    scope is open and was asked once**: whether it covers the softer of
    the existing checks and the levels of the capture's gate, or only
    what is built from here on. No existing check or gate level is
    changed or retired on this sentence alone.
  - **The capture tool's purpose is the whole path of the data.** For
    each thing a node uses on the way from source to target, a reader
    is to be able to see how it came in, every change made to it, what
    made the change and why, at which step of the workflow, from which
    node to which, with which values set on the node, for every window
    that runs. The capture as a workflow is wanted for the same reason.
    Nothing is built or sized for it; it is the direction the capture
    tool is held to from here, and its first step is a list of what the
    saved graph, the plan, the saved region and the tables already hold
    of it.
  - **Meta's Sapiens2 names are kept in a folder of their own, with
    Meta's licence file beside them.** The pose model's list of point
    names (its checkpoint names none, so the tools need the list) and
    the part model's class names and colours go there, each list as a
    file saying where upstream keeps it. The part node reads its names
    and colours from there; that is a node change, so it lands with a
    restart and a smoke test (the same preview before and after, its
    class map equal array for array). The lists are held two ways: to
    upstream's files where that checkout is present, and the point names
    to the checkpoint's own left and right pairs. Decided by the owner
    on being shown the options. Built the same night at
    `vendor/meta_sapiens2/` (`58e49dd1`, then the node in `5054d39a`),
    with the restart and the smoke test.
  - **Meta's folders live under `vendor/`.** The Sapiens2 names go to
    `vendor/meta_sapiens2/` from the start; `meta_sam3/` moves under
    `vendor/` later as a change of its own, since node code imports from
    it. The pack carries a party's licence as that party's own file and
    writes no wording of its own about it.
  - **Video files go under the output folder's `Video/` on the share,
    not under the repository's `data/`.** The marked reference videos
    and the mask videos made on 2026-10-11 under `data/` stay where
    built graphs name them until a job is next built, and move then
    under a new job name. Open: whether this reaches the capture tool's
    own mask copies and strips, which it writes into a capture's folder
    by design and which the built graphs read.
  - **A layered view of the preprocessing is built as shared library
    code**, and what it adds goes into an existing smaller tool or node
    where it fits, with no code duplicated across them. Asked after four
    sessions had differed on whether it should be a mode of the capture
    tool or a tool of its own, and had each ended at the same condition:
    one reader of a source and one set of drawing code, in one place.
    What follows from it: the capture tool's drawing helpers move into
    `bench/_lib` first, as a change of their own with a check and a
    sweep on a started server; a view reads a capture folder and not
    raw mask paths; it is the look recorded before a plan or says it is
    extra; one palette and one legend. The owner then said of the one
    layered clip that exists that the view is wanted; that is not a
    verdict on each of its panels. It is sized together with the capture
    tool's purpose above, and nothing of it is built.
  What prompted it, the same night: the bench tools' own fit of a source
  to the canvas was found to differ from the loader's by about half a
  row on this video, and nothing recorded how the source came in. The
  correction, what it means for older captures and what was read both
  ways on a real capture: [`handover.md`](handover.md), "After the page
  was first committed".

## 2026-10-10

- **What is on a frame of an input video is made once and kept, in
  `contact_sheets/`; and the file of who is who holds only what a person
  said** (owner, late that night, on what eight sessions answered when asked
  for a better way to write down who is in a scene). What it replaces: every
  session sampled the input video again into its own scratch folder, at its
  own spacing, and read six or eight stills; the form stopped earlier that
  night had a person or a tool write a statement for every subject on every
  window of a load's plan. Now: `bench/contact_sheets.py` writes a video's
  raw frames as sheets into ONE folder a video, named by the video's file
  name, each sheet named by its first and last frame; the folder is ignored
  by git except its `AGENTS.md`, which says to open a sheet before sampling
  again and to make something else only for a closer look, a mask, or what
  a frame number and the raw frame cannot answer. The sheet of a few chosen
  frames a window, carded that night for the capture, is not built: these
  answer it, and a session still looks at frames when in doubt. On who is
  who the owner took two points from the answers: the question is "who are
  the numbered people on this picture" and never "is this subject in", and
  the file that keeps the answers holds only what a person said, the
  tracker's opinion staying in its own table. `bench/cast_file.py` is
  already both; what changes is that no tool drafts a tracker's take into
  it. Not decided: whether a person's answer says how sure it is, and in
  what form; and nothing reads the cast file yet. The board card is
  `idea-who-is-in-a-scene-written-once-per-video`.
  - 2026-10-10, an hour later, on the owner's three objections to the
    first form of the sheets (a tool that writes nothing must say why; a
    session could make a set somewhere else; a sheet could be wrong because
    a frame rate was off, unknown or assumed): the tool takes no other
    folder; every refusal and the "nothing written" line say why there is
    one set and what to do; and a frame number is proved, not assumed. It
    reads each frame's own timestamp, reads the written sheets back against
    frames read by timestamp at the cuts, removes a set it cannot show
    right, and has `verify` for a set already there
    (`contact_sheets/AGENTS.md`, "Can the numbers be trusted").
  - 2026-10-10, the same hour: FOUR THINGS THE OWNER MARKED TO ACTIVELY
    AVOID, from what the sessions answered. (1) A new place for a fact that
    is already written: the answer is to take the other places away, so
    who is where is typed into the cast file and nowhere else, and the
    tools' own switches for it go. (2) A record with no way to say "I
    looked and I am still not sure": the answer itself needs that value.
    (3) The three costs of answering a picture: depending on the cuts being
    right (an answer is of a frame, never of a shot, so a missed cut costs a
    question and not a wrong answer); one picture being one frame (ask at
    more frames where the tracker's table shows the people on screen
    changing, and show which frame an answer is of beside each window); and
    a wrong answer inherited by every later job (who, when and which frame
    shown wherever it is used, taken back in one line with the old one
    kept, the tracker's disagreement shown first, the outcome written after
    a render). (4) A rule for every fault: first ask which view that exists
    would have shown it and why nobody opened it; a new rule comes in only
    with one going out. Of these only the frame-and-box form and the
    withdrawal exist (`bench/cast_file.py`); the rest is not built, and (4)
    is a habit with no mechanism behind it.
  - 2026-10-10, the same night: WHOSE LOOK COUNTS (owner, agreeing with a
    point two sessions raised and nobody contested). The owner's eyes on
    playback are the real "seen". A session's look is at stills, and is
    kept apart from it: an answer says which of the two it rests on, and a
    session's look is never recorded as, or shown as, the owner's. Not
    built: the cast file records who answered and not on what kind of look.
  - 2026-10-10, the same night: HOW SURE, AND BY WHOM (owner, settling the
    point left open above). An answer says how sure it is in BOTH a word
    and a number, kept short, and says who gave it: the session's name, or
    that it was the owner. The reason given is that an answer can then be
    audited. What the sessions had all attached to a confidence still
    stands and was not overruled: only the one who answers writes it, it is
    never shown without what it rests on, and no tool compares it to a
    line, blends it with a tracker's figure or refuses on it. Not decided:
    whether a tool may list the least sure first. Not built.
  - 2026-10-10, the same night: HOW IT IS PUT IN PLACE. Asked by the owner
    how, when and who, with the condition that it is not another check, all
    seven sessions said the morning and not that night, and agreed an order
    whose first step is no code: answer the pictures of one real stretch by
    hand with the cast file as it is, then give the cast file what it lacks
    and one reader every tool calls, show the answers where a window is
    already shown, let an answer start the tracker, and take each old place
    away in the same commit as the tool that starts reading the cast file.
    The order, who takes which piece and what would make it fail are on the
    masking board, `plan-who-is-in-a-scene-put-in-place`. Nothing of it is
    built.

- **The text is read whole, and never at second hand; and the day's
  handover is corrected by the day's postmortem** (owner, to the session that wrote
  the day's postmortem, on its finding that every other session had known
  the texts only from the lead session's descriptions of them). What it
  replaces: nothing said how a text travels between sessions; a text went
  out as a clause in a message, and a rule written that morning had kept
  one of the sentences the owner ruled out that night. Now `AGENTS.md`,
  "When a render is wrong", says the text is read whole by whoever reasons
  about the load, and that the owner is shown the whole text when a cause
  is not obvious. The postmortem is
  [`2026-10-10_span_fun-three-people.md`](../research/postmortems/2026-10-10_span_fun-three-people.md);
  what [`handover.md`](handover.md) used to claim and where it was wrong is
  that postmortem's section 8, and the page carries a dated note at its
  head with the same list.

- **A scene's text is data served by what plans the shots, never verbatim
  in code or a workflow; and it is a caption, not an instruction** (owner,
  to the lead, the evening a face render's mouth stayed shut through a
  whole window). What it replaces: the pack's masked text ends every prompt
  with two constants (`masked_prompt_text.py`, `SOUNDSCAPE` and `MUSIC`)
  that say the scene is quiet and has no music, whatever the track holds;
  the job files of that day mapped one text to several shots, so a shot in
  which a hat goes on was drawn from a text with no hat; and a sentence the
  lead added to one load's text to protect a telephone ("the handset ...
  stay as they are") was the one difference between a piece whose mouth
  followed the song and a piece whose mouth stayed shut. The owner's
  reading: the model has no idea anything is being replaced; it generates
  in places while it hears the track, so edit-talk is read as a
  description of the picture, and the vendor's captions were written for
  audio the model generates or for labelled audio references, not for
  frozen audio. The rule is in `AGENTS.md`, "Operative rules". Not built:
  a text kept with the frames it was written from, the builder refusing a
  load whose frames its text does not cover, and the sound lines taken
  from per-video data. A second session's audit of every prompt sentence
  held in code is a dated record under `bench/results/`.

- **The masked text's hardcoded assertions cut** (owner, on that audit,
  `bench/results/2026-10-10_prompt_text_hardcoded_in_code.md`: fix the
  prompts and have a second session audit the fix). What it replaces, all
  in `masked_prompt_text.py`, the text the Masked Prompt node writes for
  every graph that wires it: sentences that said, of a clip the node never
  sees, that the picture is photorealistic live-action, that the subject is
  there for the whole take at one size, what else is in the scene, that
  the plate is steady and sharp, that the subject casts a shadow, that the
  original has hair on the shoulders and the clothing a print, and what the
  subject's breath, eyes, head, shoulders and hands do; and sentences that
  spoke of an edit (what takes whose place, what stays, what is kept, what
  is "the scene's own", a camera that "does not change"). Now the
  whole-person and head roles open their shot with one sentence, that the
  subject is in the scene, and hold only what the reference provides and
  how the mouth follows the chosen voice. The soundscape was a line saying the scene is quiet with a brush
  of clothing; it is now the general line the reference composer already
  wrote, held once and read by both (approved by the lead; read and
  accepted by the owner the same night, as the lead relays it; reasoned,
  not measured: the field was not tested on its own).
  - *The music line did not change, and the entry above misreads it.*
    `non_diegetic_music: N/A` does not say the track has no music. That
    field is music only the audience hears, the guide counts a song a
    person on screen performs as part of the description, and `N/A` is its
    form for no such music (`vendor_guides/base_en.md`, 4.7). So it holds
    for a subject who performs the track, which is every shipped masked
    graph, and the audit's row calling it an unconditional fault was too
    strong: found by the fix's auditor. What is not known: with `voice`
    silent over a score nobody on screen makes, `N/A` may be false, and
    the node cannot tell.
  - *The head-and-upper-body role is held back* (the lead, the same
    night). It is the one wording a person has watched and accepted, and
    an accepted result is not changed without a render; the order to fix
    the prompts and that rule meet here. So only its sound line changed
    where a motion reference is wired, which is every shipped graph of
    that role (with none wired, one sentence nobody had rendered, that the
    upper body and the legs "move as one body", was cut at the auditor's
    word), and it keeps four sentences of the kinds cut elsewhere, named in the
    file as `HELD`: the scene "already lit, framed and cut" in its
    summary, the style opening, the light sentence and the camera
    sentence. They go after one render of the upper graph with and
    without them, on a window already accepted, is read.
  - *What it costs.* Each cut text is under the reference guide's word
    budget for a description, as the upper-body role already was; nothing
    was written to fill the count. The whole-person and motion entries of
    the bank no longer hold the wording the measured arms ran. Neither cut
    role has rendered as it now stands.
  - *Not done.* The bank's hand-typed masked texts from before the node
    (`ref2va_masked_subject_swap`, `_head_swap`, the three `_lead_` texts,
    `_stage_singer`) still carry sentences of both kinds; no shipped graph
    reads them through the node. The mouth sentence still says the lips
    rest whenever the voice pauses, which goes further than the choice of
    voice; it was left alone because the one measured reading of a mouth
    that night was a sentence about singing opening it. The description has no style opening,
    which the guide asks for (`vendor_guides/ref_en.md`, 5.2) and the node
    cannot write truthfully. The sound lines from a video's own track are
    the analysis folder's work.

- **Nothing an earlier run made is reused by a key** (owner, to the lead,
  on the list of stores a session drew up that evening). Until this line
  the pack had a kept-mask store (`reuse_mask`), stored-window reuse
  (`reuse_windows`) and an in-memory keep of source latents and
  conditionings, each switched off in code or by default since 2026-10-04
  to 2026-10-07 after a stale stored mask hid a fix, and each still in the
  nodes' schemas; and a reference store on the references lane's nodes.
  Now a result is made again or handed in explicitly as a named file or
  wired input; a mask is temporary preprocessing data for the job that
  reads it. The stores, their inputs, code, checks and prose are removed in
  a node change of their own; window files a person names (`keep_windows`,
  `continue_from`) are not stores and stay. The rule is in `AGENTS.md`,
  "Operative rules".

- **What the owner cut from the tools around the nodes, the same evening**
  (owner, to the lead: no complexity for nothing; a simpler thing that
  accepts an edge is fine; "You make the calls but yes"). Each session
  marked its own additions of the day keep, simplify or remove; the lists
  below are those, as approved. NOTHING OF IT IS IN THE TREE when this line
  is written except where a commit is named; a dated line is added under
  each list as it lands. The node cuts are the next entry; the owner later
  said they are made the same evening, after the renders, in whole pieces.
  The owner's word on the prose: when a thing is cut, every doc reference
  to it is cut with it, and where a reader would look for it a dated line
  says what was removed and why, pointing here.
  - THE TRACKER AND THE CORRECTIONS TOOL (`subject_track.py`,
    `bench/who_is_who_across_shots.py`). Its owner's own sentence: the
    tracker's likeness, the tool's sum and the tool's evidence total are one
    measurement arranged three ways, not three votes. KEEP: the join of
    duplicate detections; one row a shot with what each tracker called; the
    question in the tile's own numbers; a typed, picked or seeded answer is
    a person's word; "nobody detected" needs no question. SIMPLIFY: `others`
    MARKS the people another tracker holds and no longer removes them from
    a frame's list (with that go the per-look left-out counts and "complete
    looks only"); the only-person rule is off whenever `others` is wired.
    REMOVE: the sum and the evidence total as deciders, their levels, the
    rule that a row agrees when three readings agree, `--suggest-lead`,
    `--unsure take`, `track_score` in the shot table, the signature vectors
    in `looks` (the boxes stay), the `marked` mesh style and its `legend`.
    `seeds` stays only while the seeded-memory test or the cast file
    (`bench/cast_file.py`) writes it.
    2026-10-10, later, from that tool's owner reading this list against
    its own: KEEP also the first line when a tracker picked nobody, and
    "only headless detections" counting as nobody. SIMPLIFY also the shot
    table's `duplicates_joined` block to one count. REMOVE also the tool's
    measurement modes (`galleries`, `looks`, `--crops`), which leave the
    tool or stay only as the record's reproduction, and with the sum its
    two reading rules (people seen seldom left out; names carried by place
    from look to look). The sum and the evidence total are removed WHOLE,
    not kept as columns: a column nothing reads is what was cut. The gate
    reads `suggests`, `lead` and `looks_used` from the corrections file, so
    that cut lands together with the capture tool dropping those fields.
    2026-10-10, the same night, WHAT OF THIS LIST IS IN THE TREE, by its
    owner. LANDED, in one stretch with the gate's two commits between them:
    `others` marks held people and removes nobody, with the left-out counts,
    the per-look only-person bookkeeping and "complete looks only" gone and
    the only-person rule off whenever `others` is wired (`e9de8cc3`; the
    shot table is version 2 and `MASK_VERSION` is 14, so a correction typed
    before it is refused until a preview is run again); the corrections
    tool cut to its reading half, the sum, the evidence total, the levels,
    the three-way agreement, `--suggest-lead`, `--unsure take`, the
    measurement modes and the two reading rules removed whole, with
    `hand_out`, `same_people` and `sure` out of `subject_tracks.py`
    (`ace202c4`; the gate's side is `317d9c65` and `da9b8446`). Every doc
    reference went in those commits. NOT YET CUT, still approved: `track_score`
    in the shot table, the signature vectors in `looks`, the
    `duplicates_joined` block down to one count, and the `marked` mesh style
    with its `legend`; the last is node code and takes a cold read, a sweep
    and a restart of its own. `seeds` stays: `bench/cast_file.py` writes its
    text, though no job carries it yet.
  - THE CAPTURE AND ITS GATE (`bench/capture_masked_run.py`): its own
    entry, by its session, is further down this day's list ("What the owner
    cut from the capture tool and its gate"); the large simplification is ONE
    STATEMENT in the job file, `job.who`, read by `verify`, the gate, the
    builder and the delivery, with the rule that coverage blocks. (That
    entry reached the tree inside commit `23b01495`, the lead's, which
    swept in the session's uncommitted hunk by a pathspec commit of this
    file; the words are that session's.)
  - THE JOB BUILDER AND THE ANALYSIS FOLDER (`bench/masked_job.py`,
    `workflows/masked_job_graph.py`, `clip_folder.py`). KEEP: the stages
    and `collect`; the gate before a render; the audio cut; what each model
    is shown; `window`, `context` and `motion_file`; the folder's identity
    and its fingerprint. REMOVE: the load keys no job used
    (`continue_from` and its noise value, the zoomed mesh, `motion_edge`,
    the held tail as a choice, `parts_mask`, `parts_first`, `by`,
    `motion_first`), each refused by name if a job file still carries it;
    the function that works out which contexts the planner takes; the
    placeholder capture line; the build record's copy of what the clip
    measured; in the folder module the mask table, the hand layer, the
    closed list and the second-stem rule. With the node cuts the builder
    stops setting the inputs they remove.
  - THE DELIVERY TOOL (`bench/assemble_delivery.py`). KEEP: the table and
    the lay; the proofs read from the file; every subject laid on every
    shot or a line saying why not; the lock; `restore=` for another subject
    and for the row's own class. SIMPLIFY: where two rows changed a pixel,
    the later row wins and `restore=` says the exceptions (one form, not
    two); the shots proof down to three readings; cuts from the job's
    analysis only; one reader of a piece's fade; the flags down to the
    three ever acted on. REMOVE: a class restore naming another subject,
    `--soften`, the canvas-size laying path, the per-frame csv, and
    `--alias` when this clip's last file is built. Already out:
    `restore=<subject>:contested`, added and removed the same evening
    (`92dac2c1`).
    - *2026-10-10, the tool's session, read against its own list
      (`bench/results/2026-10-10_delivery_files_and_their_records.md`,
      "What of the tool is kept, simplified and removed", which holds all
      eighteen lines): no line above is wrong; four things it leaves out.
      KEEP also `rows_shown`, the proof that each row's picture shows where
      it was meant to. SIMPLIFY also the reading of several captures that
      cover one frame, to one analysis folder a clip, which waits for that
      folder. The canvas-size path goes together with its only user, the
      older patch route `bench/patch_render_window.py build`. And the
      zoomed lay-in (`0e7f69fb`) is neither kept nor cut yet: it stays
      until the first delivery with a zoomed row is read, and goes if the
      gain cannot be seen or measured there. The capture session has said
      nothing of its reads the per-frame csv. The order: the whole-clip
      file is built with the tool as it is; then the current joined file
      is rebuilt with "the later row wins" and compared, and that
      comparison is the record for that change; then the cuts.*
  - THE LOOK, the one thing ADDED by this: one picture strip a load, one
    line saying who looked, and the builder refusing a load nobody looked
    at; no fingerprint. Not built.

- **What the owner cut from the masked lane's nodes, and how far the cut
  has got** (owner, relayed by the lead the same evening, on a
  keep-simplify-remove list the latent-step session drew up). NOTHING OF IT
  IS IN THE TREE when this line is written: it lands as one change, on an
  idle card, with one sweep and a rebuild of `workflows/`, cold-read, and a
  half-landed removal was ruled worse than none. The list, so the session
  that lands it starts here and not from a conversation:
  - REMOVE, with code, checks and prose (never rendered, or no difference
    on the one render): `motion_blur_share`, `motion_blur_pixels`,
    `motion_grey` on the Masked Source and everything only they use
    (`blur_frames`, `grey_frames`, `blur_sigmas`, the look notes); the
    motion choice `a video I wire, zoomed in` (`wired_ground`); `motion_box`
    as an input, replaced by a rule with no input (a part pass frames the
    part, a whole pass the person).
  - REMOVE `loop_plan.first_frame_choices` (moot under one load per shot)
    and the switch that reproduced renders from before the no-mask rule
    (`lay_frames_with_no_mask`, the version switch in
    `video_mask.load_window_region`, the re-lay tool's `no_mask=`). The
    capture's readers of both went first (bc9f58bf).
  - `held_tail`: the open tail is the only behaviour and the input goes, as
    a change of its own. The gate's plate-tail rule goes after it is served.
  - THE ZOOMED RENDER, simplified until the owner rules on its one render:
    `render_region` keeps two choices (the whole frame, zoomed in on what is
    replaced); `largest_zoom` becomes a constant; a zoomed run continues
    nothing and is not continued (a whole-frame run on a zoomed window is
    still refused, by `video_mask.stored_zoom`); a zoomed run with a wired
    motion video is refused; the zoomed pictures are not kept between runs.
    The zoomed file and its json STAY, because the delivery tool's lay-in
    reads them (0e7f69fb: run, fixed, canvas, source_size, frames, box,
    zoom, region_cut_at_box_sides). The report says "enlarged", not
    "softer".
  - THE STORES, under the rule above: `reuse_mask` with `mask_store.py`,
    `bench/check_mask_store.py` and the Masked Source's kept-mask branch
    (its lazy inputs stay lazy by `replace` alone); `reuse_windows` with
    the keys in `loop_resume.py` (the graph signature, the track hash, the
    window and stored keys, the reuse of a stored window) and the song
    node's `first`, `reused` and restacked-review paths, a window's latent
    then being saved on every run; `window_keep.py` whole, with its check
    (it was switched by `reuse_windows` and NOT by the code switch, so it
    was live in any graph that set the input); the reference store in
    `reference_encode.py` (`_Store`, `STORE`) with `keep_references` on the
    reference conditioning node and `bench/measure_keep_references.py`,
    while the two nodes that encode references once and WIRE the result
    stay. In shipped graphs `reuse_mask` is false in twelve and
    `reuse_windows` false in sixteen: the generator stops writing both and
    the rebuild drops them. `MASK_VERSION` on the trackers stays (the job
    builder reads it); only its sentence about a kept mask's key goes.
  - ADD, the one thing: a text input `analysis` on the song node,
    `<file name>|<fingerprint>`, checked against the video's analysis
    folder (`clip_folder.folder_of_file`, `read_manifest`), refusing a
    missing or changed folder, and written into each window's region file
    and the plan. No loader node, no settings record.
  - KNOWN AND ACCEPTED: a graph saved from the canvas addresses widgets by
    position and will not survive inputs removed from the middle (owner:
    not a concern); a render from before 2026-10-10 laid again differs on
    its frames with no mask.
  - DONE IN A SCRATCH COPY ONLY, as a patch that applied to c77a244a with
    its checks green there: `first_frame_choices` and the old-render
    switch. Everything else is not started. The session's scratch copy is
    not a place anything rests on; this list is.
  - 2026-10-10, later the same evening: LANDED, by the owner's word that the
    cut goes in once the renders finish, as five pieces of the nodes, each
    its own commit, each read cold as a patch by the assembler's session
    first, with the job builder's matching change straight after each:
    `first_frame_choices` and the old-render switch (ae12af0a; no builder
    change); the four stores (efc99317, builder a3a23384); the held tail
    (3e78f3de, builder eb735b3a); the motion inputs (0daa147c, builder
    25122b4e); the zoom cut down (5e0d7468, builder 368fbaa8).
    `bench/node_id_manifest.json` lost nine inputs across them
    (`reuse_windows`, `reuse_mask`, `keep_references`, `held_tail`,
    `motion_box`, `motion_blur_share`, `motion_blur_pixels`, `motion_grey`,
    `largest_zoom`). NOT IN when this line is written: the ADD (the
    `analysis` input), which is built and read cold and waits a day by the
    lead's ruling; the rebuild of `workflows/`, the full sweep and the
    restart, which are the lead's and follow these commits, so until they
    are done the shipped graphs still name the two reuse inputs and the
    server still runs the nodes as they were; the gate's plate-tail rule,
    which goes once the held tail's change is served. One thing was done
    that the list above does not say: the json beside a zoomed file is
    version 2 (the key `largest_zoom` left it).
  - 2026-10-10, after the lead's sweep: the stores' removal had left five
    stale sentences, against the rule that every reference goes in the
    removal's own commit. Two were pointers to removed symbols, found by
    `bench/check_doc_links.py` in the full sweep (`docs/h3_references.md`,
    a dated research note); it had not been among the single checks run
    at the landing, and it is the one to run with any removal. Three named
    no symbol and no check sees them: the help text of the encode node and
    of the song node, which still promised a reuse, and two docstrings.
    Corrected in 7c902d53 and e550c443, text only. How the three were
    found: a grep of every touched node for the words a keep is described
    in (reuse, kept, stays in memory, a hit), read line by line.

- **No more data files under `bench/results/`** (owner, to the lead: the
  folder had been taking files that serve the same purpose as `data/`).
  Until this line a dated record was a `.md` with a `.json` of the same
  name beside it, both tracked; `git ls-files bench/results` and `du` say
  what that had come to. Now a record is its `.md` with its tables in it,
  the data it was made from lives under the untracked `data/` with the
  record pointing at it, and only a small file a tracked check or tool
  reads stays tracked. The rule is in `AGENTS.md`, "Operative rules". The
  day's own `.json` files move in each session's tidy-up; the older ones
  are not moved by this line, since checks read some of them.

- **Preprocessing data is verified the moment it is made, by code and by
  eyes, and a second session reads it** (owner, to the lead, while the first
  renders through the tracked builder were on the card). Until this line the
  order was a capture, `verify`, the gate, then a render; `verify` and the
  gate say that no rule failed, and that day's previews had held wrong
  takes that no rule named (counted in the untracked
  `data/2026-10-10_fun_rest_who_is_where_truth.json`). Now the
  step straight after a preview is captured is the tool's check AND a look
  at what was made by someone who did not make it, recorded, with what eyes
  found that no code checks written down as a gap. The rule is in
  `AGENTS.md`, "Operative rules"; the first such reading is a dated record
  under `bench/results/`, written by the session that reads it; the tool
  that makes the look a refusing step is not built.

- **Who is who starts from two votes that must agree, recorded by a
  neutral party or the lead; no classifier for each video; a cut finder
  need not be perfect** (owner, to the session that wrote the plan). The
  two votes are the likeness as built and SAM's tracker seeded from a
  confirmed look, which is the owner's own idea; where they differ a
  person is asked. What each vote said and the answer given is written
  down by the job's lead session or one with no part in the tool, not by
  the session whose tool cast a vote. A small classifier trained on each
  video, which the plan had as a last resort, is removed and not to be
  proposed again unasked: the owner's reason is that the more complex a
  thing is the more of it fails. And a missed cut is to be made cheap,
  not impossible. In
  [`who_is_who_evidence.md`](who_is_who_evidence.md), first screen and
  sections 2, 6 and 8; what the page said of the classifier is in
  50989af7.
- **A rule that depends on the picture takes its value from the input
  video, up front** (owner: "we need to determine some rules by video input
  upfront not hardcoded rules for all", said as a standing rule they
  believed was already written; it was not). Now in `AGENTS.md`, operative
  rules. What it replaces in practice: gate and tracker lines held as
  constants measured on one clip (a likeness line, a face's size in tokens,
  a turn in degrees). Those stay as starting values marked unmeasured or
  one-clip; a tool that can measure the clip in its preview sets the line
  from the clip and says when the two disagree. Said while test clips for
  the who-is-who design were being chosen: two of them are sharp modern
  video and the job's own clip is soft film.
- **The changelog is written once a day, from the commits** (owner). Until
  now every commit wrote a fragment under `changelog.d/`, built
  `CHANGELOG.md` and bumped `pyproject.toml`, and `AGENTS.md` told a default
  adopted from upstream to "say in the CHANGELOG what it replaced". A commit
  now carries no entry and no version bump, and says such things in its
  message; one session summarises the day's `git log` into one fragment at
  the end of the day and builds once. Detail goes in a postmortem. Why: with
  several sessions in one tree the two files every build writes were in
  everybody's commit; on this day three commits carried a peer's entry and
  each session's commit waited on the build before it. `AGENTS.md`
  (operative rules), `changelog.d/README.md`, `bench/build_changelog.py`'s
  docstring. Commit subjects no longer begin with a version.
- **A passage of prose that lost to the code, found by the
  lotsofpeopledance postmortem**
  (`docs/research/postmortems/2026-10-04_feature_lotsofpeopledance-end-to-end.md`).
  `masked_v2v.md` used to claim that "a shot corrected by hand is not
  searched". 0408a1f1 changed that on 2026-10-07, and the item above it on
  the same page already said so. The postmortem's other stale passage, the
  `MASKED_SOURCE` comment, is the entry below. No code or default changed.
- **Two statements corrected from the notes of the 2026-10-08 sessions, whose
  lessons had stayed in notes.** `docs/h3_references.md` said only "Set
  `force_rate=24` on the loader", which is right for a reference's time
  labels and drops a frame a second from a 25 fps source; it now carries a
  dated note that a source the result must stay in step with is loaded as an
  every-frame 24 fps copy with `force_rate` 0. `workflows/h3_config.py`'s
  comment on `MASKED_SOURCE` said the margin taken from the subject's size
  "has not rendered"; it has, and has not been judged as a pair. (mrpop,
  after the owner asked that what past sessions learned be in the tracked
  tree and not in session folders.)

- **A hand-written masked text says the least first and gains a sentence
  only when a render asks for it** (the owner: "let the model guide and
  infer until you need to get more specific", "just like progressive
  disclosure"; and of lip sync, to see what the model infers from the frozen
  audio first). What is never minimal is the reference: the subject's
  definition, what the still provides and the vendor guide's retention
  marker. `masked_v2v.md`, "Say the least first", states the order and what
  the first day showed. It extends 2026-10-07's "the text says nothing it
  cannot know" from the prompt node's generated text to a text written for
  one window after looking at it; no node text, graph or default changed.
- **A masked render is run on data, before and after: a rule, at the
  owner's word** ("take a data driven approach to all of this ... it needs
  to be a rule in the masking core docs and wiki"). Capture and flags before
  a render, the same capture read first after a bad one, a cause cited to a
  row or a figure, every flag given its outcome. `masked_v2v.md`, "The rule:
  data before a render, and the same data after", and one line in
  `AGENTS.md`. The same day the owner said node changes and sweeps are
  welcome where they remove a class of failure, and that a new job is not
  built by patching the last job's graph.
- **Where the masked lane's tracker and mask data can be seen is written
  down**, after a session had to be told the tooling existed:
  `masked_v2v.md`, "Seeing what the tracker and the masks did". It also
  records what a read of the tree found missing: no tool captures a run end
  to end (the 2026-10-07 captures under `data/` were made by scripts that
  were not kept), and nothing carries a label for which person a mask is
  on.
- **What the owner cut from the capture tool and its gate, and how far the
  cut has got** (owner, relayed by the lead the same evening, on a
  keep-simplify-remove list the capture's session drew up; approved as
  written). Done when this line is written: the size floor in
  the grade (0416ac3f), the older-file switch out of the reader (bc9f58bf),
  five data files out of `bench/results` (0d32b52d). EVERYTHING BELOW IS
  STILL IN THE TREE; each removal is one commit with its check cases shrunk
  to match, on an idle card. The order is the lead's. Why it was cut: most
  of the flags on the evening's main capture came from two warning rules
  that the tool's own author had left out of every list sent to the lead
  (the counts are in that capture's `flags.json`; untracked), and two face
  loads with a part in fragments were in that pile.
  - FIRST, ONE STATEMENT of who is in a shot, in the job file: `job.who`, a
    list of `{subject, frames [first, last], is "not in" | "left out", why,
    by}` in source frames; a load is the "in" and is never written; a range
    may cross a cut. Read by `verify` and the gate through `--job`, by the
    builder (coverage blocks at `build plan`: every subject and frame is
    under a load or a statement) and by the delivery. It replaces five
    forms a person used in one evening: `--not-in`, `--no-mask`, the
    builder's `left_out`, an override, and for a whole shot a typed
    `none`.
  - THEN WHAT IT RETIRES in `bench/capture_masked_run.py`: `--not-in`,
    `--no-mask` and `no_mask_statements.json`; the rules
    `absent_with_people_on_screen`, `taken_where_not_expected`,
    `taken_under_its_own_line`, `taken_near_the_line`,
    `subject_never_found`, `load_has_a_shot_without_the_subject`,
    `who_is_who_unanswered`, `who_is_who_not_run`. In their place ONE rule:
    the tracker's table disagrees with the statement (taken inside "not
    in"; absent or no mask under a load; neither a load nor a statement
    where the tracker has a mask or people). Kept through it: another
    subject's region over a left-out subject always blocks and says the
    left-out subject may be mostly in no file.
  - SIMPLIFY: `verify` to its input checks and "a track is empty where the
    statement says the subject is" (the comparison of two region readers
    goes, with the reader of a render's tinted review); the owner map to
    "one claimant owns, two is both" (the class-map settling,
    `contested_by_class_too` and `part_pixels_owned_by` go); the size
    warnings to a size printed per load by `shots --loads`, with no line;
    the part warnings to one line a subject; `rules_input.json` to the
    fields `shots` prints.
  - REMOVE: `segment_inside_region`,
    `segment_mostly_outside_its_own_track`,
    `region_mostly_not_the_subject`; `--plan` and `--loads` with
    `held_tail_not_checked` and `kept_frames_not_checked`; `look`; `mouth`,
    `mouth_open_with_no_voice` and the two text-against-voice rules; the
    pose table as a gate input (`bench/measure_subject_yaw.py` stays);
    `two_tracks_on_one_person`; `region_carried_across_a_cut` and
    `region_shared_across_a_cut`; a second sighting of one subject; `hold=`;
    `diagnose`. `original_shown_in_the_held_tail` goes after the node's
    `held_tail` removal is served.
  - KEEP: the per-frame tables; a run's region from the node's saved files
    and plan; `changed` with its ends; `track_empty_in_a_taken_shot`,
    `region_on_a_frame_with_no_mask`, `region_over_another_subject`,
    `kept_pixels_inside_the_part`, `plan_writes_past_the_load`; the verdict
    and the override; `shots`; the graded part; `video`; `looked`; the
    refusals (a zoomed run, a load with no start frame, a finished capture
    rebuilt).
  - NOT BUILT AND WANTED: the builder's refusal on `looks.json`; the gate
    reading a zoomed run. Both have cards on the masking board.
  - 2026-10-10, later the same evening (the capture's session): two
    changes to the list above, each from something that happened after it
    was written. ADDED TO KEEP: `load_lays_nothing_on_frames` (9b3c49d1),
    a load whose own mask is empty on frames where its subject is tracked;
    it is the state a frame-by-frame reading by another session found
    missing, and the grade's size floor (0416ac3f) feeds it. MOVED FROM
    REMOVE TO SIMPLIFY: the mouth against the voice. A face load was
    rendered again with the isolated vocal as its audio to see whether the
    vocal moves the mouth; the voice table had no voiced frame on that
    load, so the test could say nothing, and the warning
    `mouth_open_with_no_voice` had named the stretch beforehand in a list
    nobody read. What stays of it is one line a load in `shots --loads`
    (voiced frames of the load; frames where the source's mouth is open
    with no voice), not a flag; the `mouth` reading and the two text rules
    still go.
  - 2026-10-10, late (the capture's session): what has landed, and one
    more change. LANDED with the tracker's change, in one stretch: the
    absence question counts only the people no other tracker holds
    (317d9c65), and the who-is-who reader for the tool cut to its reading
    half, with `suggests`, `lead`, `looks_used`, `correction` and
    `people_seen` gone from the flag, `rules_input.json` and `shots`
    (da9b8446). MOVED FROM REMOVE TO KEEP, renamed: `diagnose`. The owner's
    rule the same night (`AGENTS.md`, "Operative rules": a wrong render is
    read as its text, its mask and four timelines before anything is
    suspected) wants one command that prints those three first and the
    measurements after; `diagnose` is the measurements and becomes its last
    part. Not built: the masking board, `build-why-command-text-mask-four-timelines-first`,
    which says what the builder's record lacks for it (a load's audio as
    data).

## 2026-10-09

- **In whole-frame video to video the still is left out when the source
  already shows the person and the video model has its own copy of the
  source.** Recorded as a rule at the owner's word, in that narrowed form:
  it is a choice a clip makes, not a default, and no graph or default
  changed. `docs/h3_references.md`, "Edit a source video", states it.
  `bench/results/2026-10-09_whole_frame_text_and_copy_size.md` holds the
  renders it rests on (stills from two renders of one clip against one) and,
  kept as a tracked file at the owner's decision the same day, every
  whole-frame render of that session: the pairs in which a window's text
  decided whether it kept its source's framing, the sizes the video model's
  copy held at, and what was ruled out for a held frame and a timing lead
  that are still unexplained.
- **An upstream read across the sister checkouts, core and kitchen; no
  default moved, and three line citations became symbols.**
  `docs/research/sglang_h3_pipeline.md` cited sglang's canonical request,
  its dropped `target` keys and its unknown condition key by line range
  (lines 278 to 378 of the request validation file for the first), all
  from its 2026-08-25 read. sglang's `17b2f35ae7` shortened that file and
  `bench/check_doc_links.py` went red on the first range; the other two
  had drifted without tripping it. All three
  name symbols now, with a dated note that the page's other line citations
  are that read's. `docs/wiki/masked_v2v.md` said the model "is given those
  tokens clean"; it keeps the word and a dated note says what core shows
  there (`comfy/model_base.py::MiniMaxH3.scale_latent_inpaint`). The read
  itself, with what core's new attention preference means for a graph on
  stock attention and how upstream's mask editing compares with the masked
  lane: [`references.md`](references.md), "What moved by 2026-10-09".
  Nothing was decided by the owner in it.

## 2026-10-08

- **sglang's serving code does have one mask path, in the stage a ComfyUI
  sampler drives; three statements about it are corrected** (a scoped
  re-read at sglang `214347891a`, asked for by the owner: does sglang do
  anything with frozen audio or with video inputs).
  `docs/h3_audio_freeze.md` section 8 used to say "No inpainting, retake,
  mask, extension, continuation, prefix, chunking, looping or streaming path
  exists anywhere in the H3 serving code". That holds for sglang's own
  pipeline and no longer for the whole tree: the stage that runs one DiT
  step for a ComfyUI graph takes core's video and audio denoise masks and
  labels masked target rows as core does. The sentence now says "in sglang's
  own H3 pipeline" and keeps a dated note. `docs/research/sglang_comparison.md`,
  "Eighth read", described that stage and did not mention its masks; the
  "Ninth read" there has it, with the rest of the read. Two line citations
  in `docs/h3_references.md` had moved with their files and are by name now:
  the soundtrack rule was cited at lines 193-202 of sglang's task
  profiles and the frame sampling at lines 706-718 of its reference
  encoding, against lines 170-179 of `comfy/text_encoders/minimax.py`.
  Nothing else that was checked again in section 8 had moved (its dated
  note lists what was and was not);
  the answer to the owner's question is no, and where it is written is
  `docs/research/sglang_comparison.md`, "Ninth read".

## 2026-10-07

- **Nothing kept on disk is reused until its key changes when the code
  does** (the owner, the same evening, on both stores): kept masks
  (`video_mask.MASK_REUSE_ENABLED`) and stored rendered windows
  (`audio_freeze_song.WINDOW_REUSE_ENABLED`) are False in the code, and the
  nodes' `reuse_mask` and `reuse_windows` inputs default to off and cannot
  turn either on. Both keys are built from inputs and hand-bumped numbers,
  and neither can tell a result made by earlier code. What a session keeps
  in memory is unaffected. To turn them back on: a fingerprint of the pack's
  code in both keys, and a check in the sweep that every key that reaches
  disk carries it. Not built. The entry on `reuse_mask` below was written an
  hour earlier, when the default alone had moved.
- **The kept mask on disk is off by default until the masked lane is ready
  for production** (the owner, the same evening they had said it could
  stay). `MiniMaxH3MaskedSource.reuse_mask` defaults to off in the node,
  `h3_config.MASKED_SOURCE` and every generated masked graph; the store and
  the input remain. A kept mask is only as fresh as the `MASK_VERSION` of
  the nodes that made it, and the entry above is what a forgotten bump
  costs. **Checked in the day's server logs**: of the day's renders, eleven
  loaded a kept mask from disk; one of them, the first confirmation render
  of the corrected-shot fix, loaded a mask the current code would not have
  made. The other ten are judged safe by reading which commits changed
  mask-making code between the mask's writing and its reuse, not by
  recomputing them. It costs the tracker and the part model on every run.
- **The fix for a corrected shot did not run on its first render, and
  `most central` is measured in the frame's own proportions.** The
  confirmation render of the corrected-shot search (the entry below) loaded
  the kept mask of the render before the fix: `MiniMaxH3SubjectTrack`'s
  `MASK_VERSION` had not been bumped, so the same inputs found the same key
  and nothing was tracked (the server's log says "mask kept from an earlier
  run"). It is 10 now. With the same bump, `subject_track.choose` measures
  `most central` with both axes in units of the frame's width; each axis
  ran alike before, which treats a wide frame as square. **That is a
  correction of the measure and not a fix of a known miss**: on the two
  frames of one clip where the subject's detection is known, both measures
  rank him alike (second of twelve on one, first of sixteen on the other),
  and measuring from the head and shoulders ranks him worse on the second.
  Tried at the owner's word; the figures are in the changelog entry.
- **A shot corrected by hand is looked for again when its track lets go**
  (the owner: "lets fix that now"). A corrected shot was tracked once from
  the corrected seed and left empty from the frame the tracker let go; a
  window that starts with the subject small, where the automatic pick names
  somebody else and a correction is the only way to follow him, rendered the
  original person from that frame to its end. `subject_track.follow` now
  runs its search on corrected shots as on any other, with the corrected
  track's own frames as the gallery. The search is still by likeness with a
  margin; the search by place built the same day is not wired in.
- **Twelve steps again on the masked ref2va motion graphs: the sixteen of
  earlier this evening is reversed by the owner on the pair it was waiting
  for.** Twelve and sixteen on the pull-out window, one seed, everything
  else the same: "exactly the same" to the owner, a ghosting artifact near
  five seconds included. `h3_config.MASKED_MOTION_STEPS` is 12 with that
  verdict as its provenance. The entry below is kept as written.
- **The masked ref2va motion graphs run sixteen steps, the base count; they
  ran twelve** (the owner: twelve "was set too early in this effort", and it
  is "not time to optimize yet"). `h3_config.MASKED_MOTION_STEPS` now reads
  `SAMPLING["steps"]`, and the six generated graphs that carry it are
  rebuilt. Twelve was the lowest count the 2026-10-05 ladder saw carry the
  source's movement, taken for speed before any masked render had been
  judged against sixteen; the base graphs had already rejected twelve for
  not following the prompt (the note beside `SAMPLING`). **Decided, not
  measured**: a twelve-against-sixteen pair on two masked windows was
  rendering when this was set. Sampler (Euler), scheduler (`simple`) and
  the shifts are unchanged and match the base recipe and the release's own
  config. Every masked ref2va render before this ran twelve steps.
- **The masked graphs load the source at the canvas in one resize** (the
  owner, on reading the loader beside sglang's: "do it right", then "do it
  now"). The generator gave the loader the canvas width alone, so a 16:9
  file was scaled by the loader, then cropped and scaled again by the fit,
  bilinear; it now gives the height as well, and the loader crops and scales
  in the one ffmpeg pass that sets the frame rate
  (`workflows/build_workflows.py`, `freeze_song_source`). Measured on a few
  frames of one 16:9 clip, emulating both paths with ffmpeg: the two-step
  path carries about a fifth less fine detail than one pass (mean absolute
  Laplacian; the figures are in the changelog entry). sglang decodes a
  reference video in one pass with Lanczos; the loader's scale is ffmpeg's
  default, and Lanczos is left as a refinement the loader does not expose.
  A 4:3 file loaded at a 4:3 canvas was never affected. **Every masked
  render of a 16:9 clip before this had the softer source**, on both sides
  of every comparison. The frames the tracker and the part model see change
  size with it, so every kept mask is recomputed; what that does to SAM's
  masks is measured after this entry, not before.
- **The Masked Prompt node's upper-body text is rewritten in the vendor
  guide's form, at under half the length, and says nothing it cannot know**
  (the owner, on two seeds of one window:
  `bench/results/2026-10-07_masked_switch_keep_prompt_verdicts.md`, section
  4). The finished scene is described and not an edit to it; the user's
  words for the subject are said in the definition, the retention line and
  the shot; the voice is one sentence. **No sentence says what the subject
  will do**: the node writes without seeing the clip, a word such as
  "turning" is read as an instruction, and on one seed the subject faced
  away at a cut. For the same reason the movement sentence every role shares
  is now the relationship alone (`masked_prompt_text.MOVES`). It was: a
  replacement described against "that person", lip and body choreography
  from the whole-person role, and "turning when they turn and by as much,
  facing where they face, gesturing when they gesture". One shot paragraph
  stays: the node does not know which window of a clip a render takes. The
  head and whole-person roles keep their text until each has a judged pair.
  What the node writes now has not itself been rendered; it differs from
  the judged text in having one shot paragraph and no list of movements.
- **The Subject Track picks the most central person by default, not the
  largest** (the owner, on a measured difference). On the three windows
  tested, two stretches of the lane's crowd clip and one of a second clip,
  the two rules name different detections; `largest` takes a figure low in
  the frame on the crowd clip and one whose box touches two edges of the
  frame on the second, and `most central` the person meant. With
  `most central`, followed alone, the subject keeps a mask on every frame
  of the stretch that had been called hard, each frame's mask overlapping
  the one before: continuity, which no label confirms. **The evidence is
  not yet in a tracked record**; the tracker lane's record of the central
  figure is owed. *Corrected the same day on a fresh session's reading:
  this entry, the 0.220.0 changelog entry and the comment beside the
  default said "a figure at the frame's edge" on "the lane's crowd clip"
  for all three windows and cited
  `bench/results/2026-10-07_subject_track_under_nudge.md`, every arm of
  which was picked with `largest`.* Changed
  together: the node's default (`subject_track.py`),
  `h3_config.SUBJECT_TRACK` and every generated masked graph. It was
  `largest` since the node was written; an arm recorded before this date
  was picked with `largest`, and reproducing one needs `pick` set to that.
- **Reversed by the owner: the frozen-row cache's masked use is back in the
  tree.** Retired on 2026-10-06 for saving no time; the next morning's
  kernel record found the cached block's attention had run on torch's
  kernel, the same pair with that corrected saved what the design predicted
  (`bench/results/2026-10-07_frozen_cache_masked_window_fixed.md`), and the
  owner, having watched the pair, restored the module, its check, the
  generator's `masked_cache` and the probe graph as they stood before the
  retirement. One change on top: the cached block no longer removes the
  attention override. `docs/wiki/masked_v2v.md`, `docs/wiki/next_steps.md`
  and `archive/frozen_cache_masked/README.md` said "retired" and now say
  this. **Also corrected:** the fixed-pair record said its change walked the
  override chain to the backend under Sol-Attn. On Sol's real override it
  walked nowhere and left the override in place, so what was measured is Sol
  declining the call to the backend below it; the record carries a dated
  note and the tree's code says what it does.
- **Corrected: the upper-body recipe had rendered on the prompt node's
  text.** Written into three places with the new graph (0.215.0): that the
  node's wording for that region "has not been watched" and had rendered
  only "on another clip" (`docs/wiki/masked_v2v.md`,
  `prompt_bank/bank.json`, `docs/prompt_audit.md`), taken from the arms
  record, whose rows predate the arm. It rendered on 2026-10-06 on the same
  window and seed, beside the typed arm; the stacked pair is on the output
  share and the record carries a dated note. The owner then watched the
  pair: both good, not told apart, which confirms the node's wording for
  the graph. Found when the owner asked on the board for samples.

- **Corrected: a typed correction has run on a real clip.**
  `docs/wiki/masked_v2v.md`, "Known limits", said "not yet run on a clip",
  and the masking board's card said the same and held a demonstration for
  the owner's word. It ran on 2026-10-06 on the car window
  (`bench/results/2026-10-06_subject_track_defaults.md`, "The correction").
  Found when the owner asked on the board what became of the card. What is
  still not built is a click or a box for a person the detector never
  found.

- **Corrected: the frozen-row cache's first masked run did not run its
  attention on the kitchen kernel** (`bench/results/2026-10-07_frozen_cache_rectangle_kernel.md`).
  What the prose used to say: the 2026-10-06 record, "The call goes to the
  kitchen backend's int8 attention, the kernel the dense stock steps use",
  and from it the archive's README and `next_steps.md`, that the miss is
  the kernel's cost on a rectangle. The kernel timed alone says the
  rectangle costs the square's rate; the cached block's call falls to
  torch's kernel because it drops the override that carries the backend.
  No decision is changed by this entry: the owner retired the masked use on
  that record, and reopening it is theirs. The same call is in the live
  module, so the audio-refine pass's cached steps run torch's kernel too;
  it is left as it is, since changing it changes that pass's numbers.

## 2026-10-06

- **The frozen-row cache's masked use is retired; the node stays for the
  audio-refine pass** (owner, 2026-10-06, relayed by the lane's lead
  session, on the first masked run:
  `bench/results/2026-10-06_frozen_cache_masked_window.md`). On a Sol-Attn
  graph the cache saves no sampling time at that window's live share and
  departs from the stock step on every cached step; on the refine pass it
  is measured to pay (`bench/results/2026-09-25_frozen_cache_s1.md`). The
  owner's terms: cite why, and keep the masked code where it is plainly
  used by nothing, which is `archive/`. **Not done at that commit**, done the
  same evening (the retirement commit after 25103f0e): the masked gate,
  `masked_cache` in the generator and the probe graph left the tree for
  `archive/frozen_cache_masked/`. What was tried is one
  window, one seed, no halo and no refresh, and the record claims no more
  than that. Three pieces of prose are corrected with it. What they used to
  say: `docs/wiki/next_steps.md`
  listed the run as owed and its saving as modelled;
  `frozen_video_cache.LIVE_SHARE_LIMIT`'s comment said the limit "sits under
  break-even" and that "the first masked run's step times replace the
  model"; the generator's note on the probe graph said no graph carries the
  cache "until a masked window has been timed and looked at with it".

- **Prose corrected: the frozen video cache had run on the card, and is in
  one masked graph.** The comment on `h3_config.FROZEN_VIDEO_CACHE` ended
  "Not yet run on the card"; the refine pass ran through it on the card on
  2026-09-25 and again on 2026-10-06
  (`bench/results/2026-09-25_frozen_cache_s1.md`,
  `bench/results/2026-10-06_frozen_cache_stage_split.md`), and the comment
  now points at the later record. `docs/wiki/next_steps.md` said of the
  cache "It is in no masked graph"; the generator's `masked_cache` argument
  now writes one probe graph with it
  (`h3_probe_v2v_masked_song_ref2va_motion_cache_api.json`; retired the
  same day to `archive/frozen_cache_masked/`), and no shipped or daily graph
  carries it. No masked window has run through it.

- **Prose corrected: the frozen-row cache takes a partly masked video.**
  `docs/wiki/next_steps.md` listed "a frozen-row cache for a partly masked
  video (`frozen_video_cache.py::_gate` takes a wholly frozen one only)" as
  owed; the gate now takes any call that freezes some row and regenerates
  another, and the entry says what is owed instead (an arm, a matched pair,
  the owner's eye). `docs/checks.md` said of `check_frozen_video_cache.py`
  "text rows live must beat text cached (`LIVE_KINDS` monkeypatched)"; that
  ordering held by the seed on the check's model and is no longer asserted,
  and the constant is `ALWAYS_LIVE_KINDS`. The row says what the check holds
  now.

- **The masked graphs' prompt is written by a node, and the fl2va PDD8 graph
  ships the generic text** (owner, 2026-10-06: generalise the lane beyond one
  clip, and a prompt that covers different scenarios without the long
  reference-format text being retyped). `MiniMaxH3MaskedPrompt`
  (`masked_prompt.py`) writes it from who the still shows, two choices and
  the Masked Source wired into it; the sentences are constants in
  `masked_prompt_text.py`. What it replaced: both shipped masked graphs held
  a typed text in the song node. The PDD8 graph's was
  `ref2va_masked_subject_swap`, which says one performer, alone, standing
  through a fourteen-second static shot; it now ships
  `ref2va_masked_person_swap`, the text mrhf wrote on 2026-10-04 to claim
  none of that. The motion graph's was `ref2va_masked_subject_motion`, which
  says "the man"; it ships the same text for "the person"
  (`ref2va_masked_person_motion`), a placeholder still being anybody. Against
  what: the generic text for "the man" is what every arm of
  `bench/results/2026-10-05_masked_v2v_motion_arms.md` ran, with and without
  the motion lines; the "person" form was rendered once, one window and one
  seed, without them (mrhf's note beside the texts under `internal/`, which
  is not tracked). The person text with the motion lines has not been
  rendered, and neither have the node's head-and-hair and silent texts.
  *Later the same day:* it rendered once, beside the same graph with "a
  man". The man text reproduced the 2026-10-05 arm exactly, so the node,
  the shipped graph and the day's code change nothing in the output on
  that window and seed; the person text turned him and started the turn
  later. The default stays "a person" and the lane's page says to set
  `subject`: `bench/results/2026-10-06_masked_v2v_person_text.md`.

## 2026-10-05

- **Sol starts at 0.2 on every graph; the 2026-10-01 call that set 0.0 on
  PDD graphs is reversed** (owner, 2026-10-05: "0.0 makes no sense to have
  on. zero dense doesnt make sense"; "no 0.0 default for sol anywhere").
  `h3_config.SOL_PDD_OVERRIDES` is empty again and the `sol_only` candidate's
  0.0 override is gone; every PDD graph rebuilt. Against what: the 2026-10-01
  panel showed only that a text-to-video finish could not be told apart at
  0.0 and 0.2; the ref2va PDD8 arms of 2026-10-05 ran Sol from step zero and
  lost the subject's video reference, and the Sol-off and Sol-at-0.2 arms of
  the same evening are in `bench/results/2026-10-05_masked_v2v_motion_arms.md`.
- **VOID is out, and the code built for it is removed** (owner, 2026-10-05,
  on the board: "Park VOID entirely, lets remove the code we built from it -
  it was built for cogvideox and its ghosting and not worth keeping or
  preserving. Note we tested it and thats it. No current state docs or
  workflows with it please."). Removed: `MiniMaxH3VoidConditioning`, the
  checkpoint converter and their two checks, all last present at commit
  df3bd21b. This reverses the line below from earlier the same day, which
  parked the route and said the node and the converter stay. Kept: the
  record of the test, `bench/results/2026-10-05_void_plate_turn.md`.

- **The masked lane gets a second shipped graph on ref2va for shots that
  need the original's movement** (owner, 2026-10-05, on the masking board:
  "For this you just need my approval? If so yes").
  `workflows/h3_video_to_video_masked_song_ref2va_motion_api.json`: the
  Masked Source's motion reference on, the ref2va base at
  `h3_config.MASKED_MOTION_STEPS`. Against what: eleven arms on the band
  clip's turn (`bench/results/2026-10-05_masked_v2v_motion_arms.md`): fl2va
  ignores an encoder-only video reference of the subject at any step count,
  distilled or not; ref2va carries it at 12 and 16 and loses it at 8 with or
  without its PDD bake. The fl2va PDD8 graph stays the default. The owner
  asked where 12 comes from: the lowest count on a three-point ladder (8,
  12, 16) seen to carry the reference, one seed at 8 and 12.
  Corrected the same evening: the graph's first prompt (the shipped masked
  text with the reference lines grafted on) did not turn him on its own
  render, which mrteal's yaw metric caught; the shipped prompt is now the
  exact text the arms measured, and the threshold ladder gained 10 (no turn).
- **A check may run a coderef file as a one-off numeric reference; nothing
  shipped depends on coderef** (owner, 2026-10-05, on the masking board:
  "yes you can test it just dont write code that relies on it here - just
  one offs"). Raised by mrteal's `bench/check_sam3d_body_vith.py`, which
  runs Meta's backbone file in a child process to compare against. The
  clause is in `AGENTS.md`, "Reference implementations".
- **A clip that starts on a still gets the FlashGen finish: the docs now say
  what the code ships** (owner, 2026-10-05, on the masking board: "Please
  resolve this so it reflects current state code"). `h3_config.DAILY_GRAPHS`
  has carried `h3_i2v_pdd8_flashgen_finish` since 2026-10-03, among "the
  three graphs the owner renders with" (0.185.24). Three passages still said
  the opposite, from the 2026-09-27 finisher review ("looked off",
  ow-fd-09): `docs/h3_distills.md` under "Rules of thumb", two entries
  further down this page, and one line of `next_steps.md` ("i2v stays PDD8
  alone"). Each now carries a dated note; none is deleted, since the review
  did say it.

- **The VOID clean-plate route is parked** (owner, 2026-10-05, on every arm
  rendered that day: "Not a single clip showed any improvement."; their
  reason for stopping, as mryellow relayed it and not in their words: VOID
  is trained on CogVideoX, so it may be a waste of time). Parked, not ruled
  out: a plate under the hole is still wanted, and this tool on this base
  through core's port is set aside. No pass 2, no run of upstream's own
  code, no blended windows. What was found on the way stays:
  `MiniMaxH3VoidConditioning` and `bench/convert_void_checkpoint.py` with
  their checks, and in `bench/results/2026-10-05_void_plate_turn.md` the four
  places core's port differs from upstream's inference, one of which
  (the pixel range the mask is encoded at) reaches core's own template and
  is the owner's to report or not.

- **The generated indexes list tracked files only, and the sweep holds them
  current** (owner, 2026-10-05: "do not index anything that isnt git
  tracked"; 0.190.18). `bench/build_index.py` reads `git ls-files` and
  nothing else. It used to list what was on disk under `bench/`, so an
  untracked script appeared in `bench/INDEX.md`. The changelog's 0.189.8
  entry said nothing gated the generator's `--check`;
  `bench/check_doc_inventory.py` now runs it.
- **A masking note said the VOID passes were only in the upstream repo**
  (`docs/research/masking/2026-10-04_mrhf.md`, "VOID for the plate with
  nobody in it": the Comfy-Org repackage "holds only" the T5 encoder, the VAE
  and RAFT). Wrong, and against the same note's section 5: both passes are
  in the repackage too, and only those load in core, upstream's having
  diffusers key names. mrorange found it reading core's template and the
  file's header; the note carries a dated correction and
  `bench/results/2026-10-05_void_checkpoint_conversion.md` the evidence.

- **The 2026-09-13 blind reference-view session is closed unscored** (owner,
  2026-10-05: "Im not sure how much value it added anyway", after the sealed
  key under `internal/blind_keys/` was found gone from disk). Its question,
  the `qwen_view` default, was settled by eye on 2026-10-03. Nothing else
  depends on the key; a new session seals its own.
- **No history rewrite** (owner, 2026-10-05: "Dont worry about the history
  rewrite, can move on from that"): 563328d9 stays in history and every
  later hash stands.

- **The HF hybrid checkpoint `unet_hybrid_b30` is retired** (owner,
  2026-10-05: "Remove that hybrid model, its long been deprecated", after
  mrteal found its symlink dangling and the download gone). The key leaves
  `h3_config.MODELS`, the loader map and the model-contents baseline; the
  four dangling links (b15, b20, b25, b30) leave `models/diffusion_models/`.
  Our own `unet_hybrid_adaln_all` stays, with its file. No graph used either.

- **`check_no_owner_paths.py` scans the files git would take, not the
  gitignored trees** (mrteal, on mryellow's brief for the owner; 0.189.5).
  It used to walk everything under the repo, `internal/` and `data/`
  included, and was red on raw captures and session notes there with no
  tracked file among the hits. `docs/checks.md` index row has the reasons
  and the two ways to look wider.
- **The `h3-ab-session` skill no longer routes to `internal/blind_keys/`**
  (0.189.5). The directory did not move: `bench/blind_batch.py::KEY_DIR`
  still names it, and the skill now points at that constant. It is absent
  from this box, so the sealed keys of earlier blind sessions are not on
  disk; nothing in the repo records who removed them or when.
- **`check_model_contents.py` used to print "in the baseline but no longer
  named by h3_config" for `unet_hybrid_b30`**, which `h3_config.MODELS`
  names (0.189.5). The file's symlink dangled. It was left open as the
  owner's call and the baseline was not regenerated; the owner retired the
  key the same day (the entry above), and the baseline dropped its
  fingerprint in 0.190.1.
- **`docs/checks.md` "Running them" said exactly two checks need
  `PYTHONPATH`** (`check_clone_v_wiring.py`, `check_correctness.py`; 0.189.4).
  A sweep without the variable found `check_reference_encode.py` needed it
  too, and now none does: each calls `bench/_lib::bootstrap`. Its Gaps item 3
  used to list the split as open. The index rows for the first two used to
  read "CUDA, `PYTHONPATH`"; both returned 0 with "skipping" when masked and
  now exit 2. The `check_reload_invariance.py` row used to say only that it
  renders; it now says when it exits 2 and that a sweep never posts.
- **The at-the-call LoRA node reads Kohya-style keys** (owner, 2026-10-05,
  on the sglang read: "may as well"; 0.189.0). `lora_branch.py::native_keys`
  renames them through core's table. It replaced a refusal that pointed such
  a file at `LoraLoaderModelOnly`. The module docstring's "Formats" paragraph
  used to say every key must sit under `diffusion_model.`.
- **PDMD is retired and removed** (owner, 2026-10-05: "pdmd - lets retire it.
  i didnt see anything from it worth keeping"; 0.188.0). Removed: the four
  graphs, the `PDMD_*` constants, the contract case in
  `check_distill_settings.py` (which now fails a graph loading a PDMD LoRA),
  `bench/measure_pdmd_lora_conversion.py`, `bench/pdmd_vs_flashgen_arms.json`
  and `docs/research/pdmd/`. Retired, not refuted: one blind look. The
  records stay in `bench/results/` (`2026-10-01_*pdmd*`), the code in git, and
  the lane in `docs/roadmap.md` "Closed lanes", whose row used to say the
  graphs "stay buildable on kijai's files" and names what would reopen it no
  longer. `docs/h3_distills.md` used to carry a PDMD row and section; it keeps
  a pointer to the verdict record.
- **The base samples with Euler; `er_sde` is gone from every graph** (owner,
  2026-10-05: "lets change our default sampler for all base / non-distill
  workflows - or rather, any workflow that uses er_sde - to now use euler. no
  er_sde anywhere"; 0.187.0). It replaced `er_sde` / `simple`, the default
  since 2026-08-15. The prompt was vllm-omni keeping Euler its default when it
  added `res_multistep`. Not measured here: the base at its step count on
  Euler. `workflows/h3_config.py::SAMPLING` has the provenance. Prose
  corrected with it: `docs/h3_distills.md` said the base and the distills do
  not share starting noise at one seed, `docs/custom_node_gaps.md` item 2
  listed the sampler as an open gap against the reference engines, and
  `docs/research/h3_dit_implementations.md` said the shipped graphs split
  between `er_sde` and `euler`. Each keeps its text under a dated note.

## 2026-10-04

- **The Subject Track sets itself, and was validated by two other sessions
  before it landed** (owner, 2026-10-04: "OK - if mrblue can validate this
  works, im good", "And same with mrpink when hes done and ready"). mrblue's
  run found the first draft failing on a one-person clip; the rework landed
  as 0.186.45. That evening, on the owner's word ("If you know how to solve
  those, do it. Everyone else is offline"), a third clip broke it again and
  the node now compares a person in two places and asks a lone person for a
  head (0.186.47). It replaced: a pick frame and a match value the user had
  to set, then a score floor that kept a microphone out by a narrow margin.
  [`masked_v2v.md`](masked_v2v.md) has how it works and its limits.
- **Inputs marked advanced stay visible, and that is accepted** (owner,
  2026-10-04, on finding they do not fold away in their frontend: "its
  visible without clicking anything", then "yep thats fine leave it"). The
  marking stays in the schemas and is not relied on to shorten a node.
- **A late start from a softened copy of the original is not adopted for the
  turn.** Eight arms: he turns, and his clothes change, and on one seed his
  hair (`bench/results/2026-10-04_masked_v2v_turn_soft_arms.md`). The code
  the arms ran on is saved beside that record and is not in the tree. What
  is left is the owner's to choose between: a clause about facing, or
  different clothes.

- **Masked video to video, version one, is the two-node design on the
  distilled chain** (owner, 2026-10-04: "Yes I did say the two node design",
  "Lets make that the shipped workflow", "if good, we can lock in and say
  success for v1"). Subject Track picks one person once and follows them
  across cuts; Masked Source says what happens to them, at the margin and
  composite the owner chose; the PDD8 song chain with Sol-Attn renders it.
  Decided against, with the owner's words where there are any: a ControlNet
  ("I dont want a controlnet though"); the baseline chain for this work
  ("i dont think its worth the long render time"); two samplers, parked for
  another day; more work on the first clip ("not worth the effort", its dark
  lines being seams in the source's own backdrop). Open at the time of
  writing: the replaced subject does not follow the original's movement
  unless the prompt names it (owner: "he doesnt turn around at the end").

- **Masked video to video is built on the song node, not on a daily graph or
  a ControlNet** (owner, 2026-10-04: "Not sure i wanna use a controlnet model
  unless we absolutely have to"; "we should look to our frozen audio workflows
  and prompts for reference because the lip sync worked great in them"). The
  owner asked for the distilled daily graphs as the base; the PDD8 song chain
  was used because a mask does not survive two samplers as wired
  (`docs/h3_audio_freeze.md` section 4, with the fix a peer found). SAM 3.1 runs from Meta's original
  checkpoint repacked here (`bench/convert_sam3_checkpoint.py`), at the
  owner's word: "I wanted original weights so we have control over
  conversion".
- **`docs/h3_references.md` said "There is no mask"** of editing a source
  video. Still true of the ref2va edit; a dated note there now points at the
  masked path.
- **A node's tooltip says what the input does, when to change it and what it
  costs; why a default is what it is goes in a comment and here** (owner,
  2026-10-04: "make the tooltip and code comments and everything else
  actually simple to understand. Chosen by the owner means nothing to someone
  using it"; and of the widget names, "I like the widget names as-is because
  its part of the pipeline"). Applied to the reference nodes (0.186.1); the
  input ids stay.
- **The masked source keeps its mask across runs, and nothing is saved or
  loaded by hand** (owner, 2026-10-04, asked whether pre-encoding could speed
  up testing, and shown that the tracker and the part detection cost more
  than a window of sampling after a restart: "save the mask, not the
  latent"). `MiniMaxH3MaskedSource.reuse_mask`, on by default, `mask_store.py`.
  They are kept in the output folder's `masks/` (owner, the same day: "output
  folder in a masks folder like latents is"), not in ComfyUI's user folder,
  where the first commit put them.
- **Masked renders are one pass for now; two samplers are for another day**
  (owner, 2026-10-04: "its more complexity than we need ... just focus on one
  pass"). `MiniMaxH3RestorePlate` stays registered and unused; the latent
  output and the daily-chain graph it would need are not being built.
- **An omitted `qwen_view` or `size_policy` stays a loud failure, with no
  None branch** (owner, 2026-10-04, on the two options put to them). Comments
  in `reference_conditioning.py` and a docstring in
  `bench/check_reference_runtime.py` used to claim core substitutes the
  schema's first option for an API prompt that omits a DynamicCombo. Core
  substitutes nothing and the node fails at execute. The same day the owner
  restated the default: the text encoder's copy is `separate` at a 512 short
  edge "for now", not `shared`.

## 2026-10-03

- **The graphs the owner renders with get their own folder, and the
  workflows are to be organized by purpose** (owner, 2026-10-03: "if we're
  using something regularly ... why keep that in distill experiments?" and
  "can we organize our workflows while we're here? not just throw it all in
  one giant folder"). Landed: `workflows/daily/` with the ref2va, t2v and i2v
  PDD8-then-FlashGen finish graphs (`h3_config.DAILY_GRAPHS`). Chosen and
  not yet landed: shipped picks by task, experiments by topic, the
  instrumented twins and bench graphs apart; prepared on the local branch
  `workflows-by-purpose`. This narrows the 2026-09-27 rule that session-made
  distill graphs go to `distill_experiments`: a graph in daily use is not an
  experiment.
- **Last-bits differences in the conditioning are accepted** (owner,
  2026-10-03: "i dont care if it differs by such a small amount"; on the
  board: "dont care if theyre not bit identical"). That is the difference
  between encoding references and prompt in one pass and keeping the
  references across prompt edits (`data/results/2026-10-03_reference_split.json`).
  It is a switch, `keep_references`, and on in no generated graph: the
  generator change that turned it on was withdrawn before commit while a 512
  encoder view was about to become the default, where the one pass is already
  a few seconds (CHANGELOG 0.185.19).
- **A still can be an encoder-only reference by its own switch** (owner,
  2026-10-03: "maybe we should at least have a setting in our ref node(s) to
  use vae or not", after one clip rendered that way: "look the same to me,
  cant tell the difference"). `use_vae` on `MiniMaxH3AppendRefImage`, on by
  default; no default moved (`bench/results/2026-10-03_encoder_only_reference.md`).
- **A still's encoder view is `separate` at 512 by default again** (owner,
  2026-10-03: "yeah lets do 512, unless theres a reason not to based on
  everything else we did", then "i agree make it 512 default"; said in peer
  sessions and relayed). The append node's `qwen_view` default, the constant
  (`h3_rules.REF_QWEN_SHORT_EDGE`) and the generated reference graphs move
  together. **It reverses the `qwen_view` part of the 2026-09-13 vendor-parity
  decision below**; `size_policy=max`, the 2048 short edge and `allow_upscale`
  stay at parity, and the vendor still gives one copy to both readers.
  **Evidence, and it is mixed.** Three clips of the PDD8 then FlashGen finish
  graph at the judged seed, not blind: market and backstage at the 512 view
  with the shipped sparse attention ("both looked pretty good to me"), and
  backstage at the 512 view with dense attention, rendered to separate the
  view from sparse attention ("dialogue is good here and everything is
  fine"). The sampler ran about a minute shorter on the sparse pair
  (`bench/results/2026-10-03_sol_output_check.md`, section 5).
  *The level moves with the view:* backstage against the dense clip at the
  shared view is +1.1 LU for the shipped render, +3.8 LU at 512 with sparse
  attention and +4.1 LU at 512 with dense
  (`data/results/2026-10-03_backstage_qview512_clip.json`), the range the
  owner called loud on 2026-10-02; the owner heard the dense 512 clip and
  passed it. On the market scene the 512 clip's level does not move against
  the shipped clip (`2026-10-03_encoder_only_reference.md`, loudness).
  *A swapped line:* on the backstage clip at 512 with sparse attention a line
  was spoken by the wrong character; at 512 with dense attention it was not.
  The generated graphs run sparse attention, so the pairing that becomes the
  default is the one whose only dialogue clip had the swapped line. One clip,
  not attributed.
  **Not looked at:** more than one reference, a small face in a wide still,
  text or a logo in the still, a base (non-distilled) graph. No render was
  made on the rebuilt graphs. This change was built, held when the level
  result came back, and released when the owner passed the separating clip.
  The remedy for a still that needs the detail, or a scene that comes out
  loud, is `qwen_view=shared` on its append node. The instruments stay on
  `shared`: the `_savelat` and `_x0` twins and the bench graphs (byte-for-byte
  comparisons), and the `h3_probe_refview2_*` scenes (their manifest's
  `parity` arm). A workflow saved before this keeps the view it was saved
  with. What the prose used to claim: `docs/evidence.md` "Reference sizing",
  `docs/h3_references.md`, `docs/custom_node_gaps.md` and
  `docs/h3_conditioning_end_to_end.md` called `shared` the default; each has a
  dated note now.

- **The output check goes ahead of the per-head calibration** (owner,
  2026-10-03: "and output check first"). Every sparse lever in flight is
  lossy, nobody is judging clips, and local error did not predict the
  2026-10-02 verdicts, so a calibrated table had nothing that could accept
  it. The sweep instrument and calibrator are built and wait
  (`bench/results/2026-10-03_tau_sweep.md`); the check's record is
  `bench/results/2026-10-03_sol_output_check.md`.

- **The sparse attention default stays as it is** (owner, 2026-10-03: "leave
  the shipped default as is", and no more renders on the question). On the
  market scene at three seeds the owner passed dense attention at all three,
  sparse in the first stage with a dense finisher at all three, and the
  shipped setting (sparse in both stages) at two; the third has the double
  crate, and it is the seed the 2026-10-02 panel judged. A dense finisher
  costs part of the saving. The owner kept the default; a dense finisher is
  a per-render remedy, not a shipped setting.
  `bench/results/2026-10-03_sol_output_check.md`.

- **No automatic judge for action defects** (owner, 2026-10-03, asked
  whether to build one: "no"). The output check found that latent distance
  from the dense render, cut count and loudness do not predict the owner's
  verdict on the market scene, where the defect is an action done wrong. The
  owner's glance stays the acceptance test for a lossy change; latent
  distance keeps one use, certifying that a render is nearly the dense
  sample. `bench/results/2026-10-03_sol_output_check.md`.

- **Corrected: `docs/SOLATTN.md`'s options table carried two retired
  defaults.** The `dense_blocks` row said `45,48,49` and the
  `sink_conditioning` row said `exact_kv_and_rows`, and that no graph ships
  `exact_kv_and_all_rows`. Both rows now point at
  `sol_attn_h3.py::SOL_DENSE_TAIL` and `SOL_SINK_DEFAULT`. Found in refdude's
  review of the ref2va path, 2026-10-03.

- **Sparse settings go per head and per segment** (owner, 2026-10-03, on the
  lever board: per-head table, "love it. lets do it"; separating text, video,
  audio and reference rows, "a granularity dive ive been wanting"). The same
  day the owner said no more judging sessions for now and asked for
  lower-level work on the kernels and on how the nodes compose with core. The
  kernel half is in the installed kitchen build (`tau_map`,
  `bench/results/2026-10-03_kitchen_tau_map.md`, 0.185.7); the node and the
  calibrated table are not built.

- **Corrected: `docs/sol_upstream.md` named the installed kitchen build by version.** Its 2026-10-02
  section said `vendor/rebuild_kernel.sh --check` reports the build current against core's pin
  "which core moved to `0.2.36`", and pointed at the 2026-10-01 merge record for its contents. The
  build was rebuilt on 2026-10-03 against core's next pin. The section keeps its text with a dated
  note that points at the pin in ComfyUI's `requirements.txt`, the build record beside the venv and
  `bench/check_sol_kernel.py`. The 2026-10-01 entry below and the dated records that name that
  day's build are history and stand.
- **Corrected: `docs/hardware.md` said host RAM was "ample" and stopped there.** Every module is
  configured below the speed it reports (found by a dotfiles session, verified with `udevadm`).
  `bench/hwinfo.py` now prints each slot and flags it; the doc says what it can and cannot move.
  Its "Last updated" header read 2026-08-17 with a last commit of 2026-08-25.
- **Corrected: `bench/profile_sol_stages.py` wrote a fixed `model` string** ("captures from a base
  16-step t2v render") into every record, and `data/results/2026-10-03_sol_stages_ref2va.json`
  carried it for a ref2va capture. The tool now reads the source render from the capture set's
  `manifest.json`; that record's field is rewritten to what the tool writes now.

- **The Sol node is shown as "MiniMax H3 Sparse Attention"** (owner, 2026-10-03:
  "MiniMax H3 Sparse Attention works"; not core's exact name, which is "Model
  Sparse Attention"). The owner wants the node to be the home for sparsity and
  attention efficiency in general, not Sol's method alone. Display name only:
  `node_id` stays `MiniMaxH3Sol` (`docs/comfy_notes.md`, the `node_id` rule), so
  no graph changes. It used to read "MiniMax H3 Sol-Attn" (0.185.1).

## 2026-10-02

- **Sol's `sink_conditioning` default becomes `exact_kv_and_all_rows`; `dense_blocks`
  stays Option C** (owner decision, 0.185.0). The owner stopped the output-level
  panel after one judged seed and asked for "a good solid general default".
  `all_rows` was the one Sol setting rated fine on the dialogue scene and lost on
  no scene; the dense_blocks choice did not move the verdict. Applied to every
  graph, video references included, against the roadmap's 2026-09-10 plan that
  kept ref2va on `exact_kv_and_rows`. Evidence and cost:
  `bench/results/2026-10-02_sol_dense_blocks_panel.md`. Retired the
  `h3_candidate_t2v_sol_allrows` graph, now the same as its base. The
  `SOL_DENSE_TAIL` comments used to say Test 9A "proved" Option C; they now say it
  was kept by decision.

- **The dense_blocks campaign's report was corrected against its captures**
  (0.184.13, `bench/results/2026-10-02_sol_campaign_reanalysis.md`). It used to
  say Test 9C ran `exact_kv_and_all_rows` (it repeated Test 8's settings), that
  the Test 9 arms held the seed fixed (every run has its own seed), that Test 7
  shows multi-shot robustness (it shares Test 6's settings with a different
  prompt and seed), and that Options A to C and `start_percent=0.2` lowered
  error (on the cells every run measured, they sit within prompt-and-seed
  noise). The Option A, B and C adoptions below rest on local probe error alone.
  `SOL_DENSE_TAIL` is unchanged pending the owner.

- **Sol's `dense_blocks` adopts Option C / Full Ridge Shield (`"38,39,40,41,42,49"`)** (owner decision, 2026-10-02).
  Replaces Option B (`"39,40,41,42,49"`) as active default `SOL_DENSE_TAIL`. Empirical findings
  from Test 8 ($\tau=1.3$) revealed that Block 38 alone breached the 20% error ceiling (22.09% peak call error).
  Test 9A proved that shielding Block 38 alongside Blocks 39–42 and 49 drops whole-network peak call error to
  a record-low 17.73% (Block 43 is now the worst remaining block), while ref_img max error drops below
  20% (19.37%) and routed density remains ultra-sparse at 15.18% (~20.3% relative compute reduction over
  $\tau=1.1$). 166 workflow graphs rebuilt and validated against the live server. (0.184.12).

- **Sol's `dense_blocks` adopts Option B (`"39,40,41,42,49"`)** (owner decision, 2026-10-02).
  Replaces Option A (`"39,41,42,49"`) as active default `SOL_DENSE_TAIL`. Empirical findings
  from Test 5 and Test 6 proved that shielding Block 40 eliminates the last middle routing spike
  (where Block 40 had 19.17% error under Option A), bringing whole-network peak call error down
  from 23.26% to 19.74% (the first time peak error broke below 20%). 166 workflow graphs rebuilt
  and validated against the live server. (0.184.11).

- **Prose corrected by the 2026-10-02 upstream read** (0.184.9). Each keeps a
  dated note in place:
  - `references.md`, "What moved by 2026-09-25", said sglang keeps the AdaLN
    affine in fp32 and core rounds at more points. sglang's bf16 block kernel
    rounds at the same three points as core's `_mod_scale_shift`, and has since
    2026-08-18, so the claim was wrong when written.
  - `research/sglang_h3_pipeline.md` section 10 and the seventh read in
    `research/sglang_comparison.md` said sglang's ComfyUI app runs H3 in
    server mode only. Since `f1e62e3a2e` it also runs the H3 DiT per step under
    a ComfyUI graph.
  - `custom_node_gaps.md` item 3 said diffusers decodes the H3 VAE in fp32
    under autocast. Since `51a454be9` its decoder runs in the pipeline dtype.
  - `sol_upstream.md`'s 2026-09-25 INT8 VAE paragraph and the matching
    sentence in `references.md` still described the file as removed; the
    owner reopened it on 2026-09-26.
  - `research/2026-09-25_step_caching_survey.md` listed Spectrum as literature
    only; sglang ships it for H3, opt-in, since `ae47bcd4da`.
  - Core line citations in `sol_upstream.md` and `h3_capture.py` re-read
    after `2d6b7328` moved H3's embed span.

- **Test 2 confirms middle error peak invariant across two-stage sampler and exposes token routing limits**
  (empirical finding, 2026-10-02). Running `h3_text_to_video_pdd8_flashgen_finish_api.json` (6 steps PDD8 +
  2 steps FlashGen finisher; 119,102 tokens with 1 reference image at 2752x1536 yielding 7,360 VAE reference
  rows; 400 DiT calls captured across two sequential `MiniMaxH3Sol` nodes without overwriting) verified that
  the middle error plateau is invariant across samplers: Block 39 at 25.18% avg relative $L_2$ (vs 25.22% in T1),
  Block 42 at 24.59% (vs 25.67% in T1), Block 40 at 23.93% (vs 23.43% in T1), and quiet tail blocks 46 (7.37%)
  and 47 (6.53%). Furthermore, `token_routing="measured"` succeeded on localized layers (Block 0 dropped to 5.37%,
  Block 32 dropped to 12.99%) but failed catastrophically on diffuse middle layers (Block 40 Head 47 exploded to
  111.06% error, cos 0.6655), proving diffuse middle layers cannot be rescued by token routing and require
  full dense execution (`dense_blocks`). Preflight geometry confirms Triton int32 crossing safety at 99,864 tokens
  via 64-bit index arithmetic and 64-bit CUDA quantizer offsets (`ELEMENT_OFFSET_BITS=64`). Full analysis in
  [`docs/research/sparse/sol_dense_blocks_reanalysis.md`](../research/sparse/sol_dense_blocks_reanalysis.md).

- **Sol's `dense_blocks` adopts Option A (`"39,41,42,49"`), with Option B (`"39,40,41,42,49"`)
  named** (owner, 2026-10-02: "change the sol dense detail in whatever builds the workflows and the
  h3 config and sol attn h3 to be the Option A candidate list, but also include the Option B candidate
  list as a named variable, so switching is possible"). Replaces the 2026-09-25 tail default
  `SOL_DENSE_TAIL = "45,48,49"`. `SOL_DENSE_OPTION_A`, `SOL_DENSE_OPTION_B` and
  `SOL_DENSE_HISTORICAL_TAIL` are defined in `sol_attn_h3.py` and `workflows/h3_config.py`.
  `SOL_DENSE_TAIL` defaults to Option A. 166 graphs rebuilt and validated against the live server.
  (0.184.8).

- **Re-evaluating Sol's default `dense_blocks="45,48,49"` under `quantizer="rotated"`** (instrumentation
  finding and experimental trial). The 2026-09-25 default `dense_blocks="45,48,49"` was established
  under unrotated INT8 attention, where channel outlier spikes in $K$-norm destroyed quantization
  accuracy on blocks 45, 48, and 49 (`docs/h3_block49_quant_error.md`). With `quantizer="rotated"`
  (Hadamard rotation) active since 2026-09-27, activation energy is evenly dispersed across all 128
  channels: Test 1's all-block instrumentation run (Ref2VA, 119k tokens, 8 PDD steps, unmasked) proved
  that Block 48 drops to 7.14% avg relative $L_2$ error (lower than almost all middle blocks), and
  blocks 46–47 are 6–7%. However, routing sparsity truncation error reveals a severe error spike in the
  middle semantic integration layers: Block 42 at 25.67% avg relative $L_2$ error (max 27.01%),
  Block 39 at 25.22%, Block 41 at 24.35%, and Block 40 at 23.43% (worst heads losing up to 80%
  magnitude with cosine similarities down to 0.77). Retaining Block 48 in `dense_blocks` is wasteful
  while leaving the 25% middle cluster unprotected. Block 49 remains mandatory as it directly feeds
  `final_layer.video_out`. Top candidate sets under a 5-test matrix: Option A (`"39,41,42,49"`, 4 blocks)
  and Option B (`"39,40,41,42,49"`, 5 blocks). Full analysis and test plan in
  [`docs/research/sparse/sol_dense_blocks_reanalysis.md`](../research/sparse/sol_dense_blocks_reanalysis.md)
  (0.184.7).

## 2026-10-01

- **PDMD is parked, on kijai's resized files** (owner, 2026-10-01: "just keep kijais and note it.
  nothing else to do with it"). The blind first look preferred FlashGen, and kijai's resize was
  "same" as full rank on every scene (`docs/h3_distills.md`, "PDMD"). `h3_config.PDMD_LORA` and
  `PDMD_2STEP_LORA` name kijai's files. The full-rank conversions, their converter and the four
  kijai-arm probes are retired; the owner deleted the downloaded weights. This reverses 0.182.6's
  choice of full rank as the default (below). `docs/roadmap.md`, "Closed lanes", says what reopens it
  (0.184.6).

- **FlashGen and PDMD keep `start_percent` 0.2** (owner, 2026-10-01: "change it back", scoped to
  the FlashGen/PDMD extension). 0.184.1 had extended the PDD result to them without a measurement;
  0.184.3 reverts it. PDD stays at 0.0.

- **Sol's `start_percent` is 0.0 on FlashGen and PDMD graphs too** (owner, 2026-10-01: "flashgen
  should change i think"; PDMD: "may as well"). This extends the PDD decision below to the distill LoRAs
  applied at the call, through `h3_config.SOL_DISTILL_LORA_OVERRIDES`, kept apart from
  `SOL_PDD_OVERRIDES`. **Not measured on them:** the panel's FlashGen pass starts at sigma 0.8, below
  the window, so no FlashGen or PDMD render from noise has been judged at 0.0. 32 graphs rebuilt,
  `start_percent` the only change. The first PDMD-against-FlashGen renders
  (`data/results/2026-10-01_pdmd_vs_flashgen.jsonl`) ran before it, at 0.2 on every arm (0.184.1).

- **The at-call LoRA branch stays as it is** (owner, 2026-10-01: neither fix). Fusing swiglu into
  fc2's projection or folding the add into kitchen's int8 matmul would recover part of the branch's
  cost, at the price of exact parity with the published `H3ExactLoRA` and, for the kitchen route, a
  carried kernel (`bench/results/2026-10-01_lora_branch_profile.md`).

- **Sol's `start_percent` is 0.0 on PDD graphs, 0.2 elsewhere** (owner, 2026-10-01, option A of two:
  the distill graphs only, not every Sol graph). Five blind pairs on the t2v finish showed no
  difference and 0.0 is much faster; the base graphs were not tested, so they keep 0.2. Upstreams
  disagree (core 0.2, sglang a fixed dense step count), so the adopt-upstream rule did not decide it.
  `docs/SOLATTN.md` and `h3_config.py` used to say the knob had never been measured (0.184.0).

- **Kitchen int8 stays Sol's dense fallback** (the owner reopened sage fp8++ rotated for it on
  2026-10-01). Measured the same day: kitchen is more accurate on every dense cell, the steps before
  Sol's window included, and sage saves about one percent of the sampler. `next_steps.md` used to
  list the lane as parked pending exactly this (0.183.5).

- **PDMD ships as `h3_text_to_video_pdmd`, unrendered** (owner, 2026-10-01: "Go for it"). It is PDMD
  4-step on its trainer's contract: the full-rank file at the call, Euler on `simple` at 12/3, the
  repo's Sol default. The 2-step file and kijai's two resizes are `distill_experiments/` probes, and
  `check_distill_settings.py` grades all four. The first render is the owner's call (`30b82bb6`,
  0.183.1).

- **PDMD runs our full-rank conversion; kijai's rank-reduced files are the arm** (owner,
  2026-10-01: "Sounds like we both agree"). Kijai's resize keeps a fixed share of each delta and less
  where q/k/v hit his rank cap (`bench/results/2026-10-01_pdmd_{4,2}step_lora_conversion.json`).
  Applied at the call, rank buys little speed (`bench/results/2026-10-01_lora_branch_profile.md`).
  So the default carries the published delta exactly, as `FLASHGEN_R64_LORA` does for FlashGen
  (`bench/convert_pdmd_lora.py`, `12cd81ad`, 0.182.6).

- **h3-mutant-distill gains a second node, `H3KeyframeCanvas`** (owner, 2026-10-01: "I agree with
  your auto resize approaches"), so its i2v examples size the canvas from the first frame instead of
  stretching it. The pack's README no longer reads as one node only; `H3ExactLoRA` still loads every
  adapter. Keyframes are not sized like references: the 2048 short edge is ref2va's alone (0.183.0).

- **comfy-kitchen's build branch moved to upstream main `3f7210f`** (owner: merge what upstream has).
  `h3-frontier` took seven upstream commits by merge (`aade8d5`); the build is
  `0.2.36+sol.aade8d5.up.3f7210f`. Outputs are bit-identical on every layer and render checked, and
  the int8 GEMM's qkv and fc1 are faster. Record: `bench/results/2026-10-01_kitchen_merge_aade8d5.md`.
  The branch was pushed to `origin` and `nas` the same day, at the owner's word, as fast-forwards to
  `aade8d5`. The kitchen's own tests must be run from a copy outside the clone, and its `test_bindings_*` tests are HIP-only checks that poison a CUDA run
  (same record).

- **Audio refine is never a stage of a mutant example** (owner, 2026-10-01: "usually not
  worth the extra cost"). It belongs in a workflow of its own, named `*_audio_refine`.
  None of the nine examples had one; `bench/build_mutant_examples.py` now refuses an
  example carrying `AUDIO_REFINE_CLASSES` under any other stem (0.181.4).

- **`h3_config.py`'s `end_percent` note said "sage still takes the steps before
  `start_percent`"**; since 2026-09-15 the default graphs run those steps on
  `DENSE_BACKEND_NODE` (kitchen int8, as the server log's `[h3-sol]` line says), with
  sage only on the `FLOOR_STEMS` arms. Corrected in place (0.181.2).

- **The i2v PDD8 pick is promoted to `workflows/`, and two recipes the mutant repos did
  not ship become examples there** (owner, 2026-10-01, "ok to both" to the two calls in
  `bench/results/2026-10-01_r2v_i2v_graph_coverage.md`). `h3_first_frame_to_video_pdd`
  left `distill_experiments/` for the root (0.179.2). `h3_i2v_flashgen` and
  `h3_r2v_pdd8_flashgen_finish` join h3-mutant-distill's example workflows (0.180.0), each
  held to its pack graph by `bench/check_mutant_parity.py`. The second call named no
  rows, so this reads it as the unshipped rows with positive evidence. Left out: the
  FastH3 finishers (parked 2026-09-29, and the FastH3 mutant is on hold) and the i2v
  FlashGen finish (rejected 2026-09-27). Say so and they are one recipe each. Nothing was
  pushed or uploaded: the GitHub README and the HF card here carry the new rows, and
  publishing them is the owner's action.

- **The wiki's prompting page is folded into `docs/prompting.md` and removed**
  (owner: resolve any redundancy so there is one source of truth). It was a
  router that also restated rules, and the restatements had drifted. What it
  used to claim that was wrong:
  - the shot-header timestamp rule "is not yet measured in either direction"
    (measured once on 2026-09-18; the 2026-09-26 correction below missed this
    page);
  - turns per shot "open in base format" (§14.3 closed it on 2026-09-01);
  - "sources 1 to 3 are gitignored" (the guides in `vendor_guides/` are
    tracked);
  - the bank's `adapt` column as which prompts "can take another frame count
    without a rewrite" (the bank preamble says it is not a permission);
  - that the portable files' every quotation is checked (the check pins the
    Part One strings and the camera table in the system prompt, and more in the
    HTML, never their prose rules).
  `docs/wiki/index.md` had also described it as ranking five sources with at
  least five worked examples per mode; it ranked four and carried none. Its two
  routing tables moved into the opening of `docs/prompting.md` and into its §11.
- **`docs/prompting.md` corrected where it disagreed with itself or the code.**
  - §5.6 said every line takes a mouth-closing cue and every on-screen
    non-speaker "produces no vocal sound"; §14.3 and §15.3 item 5 had narrowed
    both. The cue is positional and the phrase is for a character who never
    vocalises.
  - §9.10 repeated the withdrawn "42-68 words" and said the two scene arms were
    never rendered. Both gone; the preflight command stands in for the count.
  - §15.2 said nothing is stripped. Core's tokenizer strips nothing; our
    conditioning nodes strip both ends (0.160.0).
  - The owner's ruling of 2026-09-01, that cinematography wording outside base
    §4.3's table is acceptable, lived only in the portable copies and in a code
    comment that cited "section 8". It is now §4, and the camera warnings are
    informational.
  - The no-header-timestamps rule is labelled OWNER, as the layer table defines
    it; §3.1 and §11 said HOUSE.
  - §13's `workflows/*.json` glob skipped `workflows/distill_experiments/`; the
    command now walks `h3_config.graph_paths`.
  - "All five sources" in the header and the index: four are live.
- **Specificity is written down (`docs/prompting.md` §16)** from the owner's
  statement on 2026-10-01: pin what matters where the scene is ambiguous, and
  because only so much can be controlled, pick the few that matter and let the
  model carry the rest. The records behind it were scattered across the
  changelog, `bench/results/` and two research notes; §16 indexes them. It
  records that nothing measures the second half, and that the bank's
  "specificity ladder" is a different experiment.
- **The `h3-prompt` skill carried one rule found nowhere else**, to introduce a
  likeness as `[Name] (played by [Actor] in [Show])`. It came from another
  session's external writer craft (2026-09-11). It is now an OPEN line in
  `docs/prompting.md` §9.11, and the skill names sections and restates no rule.

- **Prompt-carrying nodes have one registry, and prose that said otherwise is
  corrected** (the owner asked for the song-node gap to be redesigned; 0.181.0).
  `workflows/prompts.py` said the catalogue "keys on the same set" of nodes
  (it did not), and `bench/preflight_graph.py` called its list "every node that
  carries a prompt" (true until the song node, which it never knew). Both lists
  are now `h3_config.PROMPT_INPUTS`. The manual's section 3.1 said a malformed shot
  header makes the shot list empty and "the grader is removed rather than reddened",
  and its section 11 said such a header "takes the shot rules inert"; preflight now
  FAILs it (0.181.1), and the same pass made the stamped-header test catch
  every spelling and the retention-line speaker-id test read compound ids.

- **The portable copies' prose is synced to the manual** (0.181.1). What they used
  to say: the HTML labelled the timestamp rule House (the manual says OWNER), gave
  the un-narrowed "every on-screen character not given an explicit 'produces no
  vocal sound'", said "cut timestamps picked by vibe overrun the clip", and carried
  a 2026-09-01 date line. The writer prompt tagged the `N/A` habit warning
  `[guide]` (it is house), gave a turn count that disagreed with the bank, carried an
  undated count of outputs graded clean, and told the writer never to write an
  absence while the bank's own cast-count fixes state one. Both now carry the
  specificity rules, and twelve rule sentences are pinned across the manual and the
  copies (`RULE_PINS` in `bench/check_prompt_docs_sync.py`). The published claude.ai
  copy is still the 2026-09-01 snapshot.

- **`docs/checks.md` said the vendor guide lives in gitignored `internal/`**, in the
  conformance row's needs cell and the camera row's allowlist note. The guides are
  `vendor_guides/`, tracked and hash-pinned, and the camera check parses the one it
  needs. Code comments carried the same claim, and cited `base-en.txt` and
  `ref-en.txt` (not our files) with line numbers off by one or two; all corrected.

- **Prompting-related comments cited `CLAUDE.md` for rules it no longer holds**:
  "cite one before building", the one-implementation trap, the second-reader finding,
  "a default is not a decision", "when something gains an off state", and the
  numbers-in-prose rule. They now point at `docs/rules_history.md`, `docs/checks.md`,
  `docs/evidence.md` and `docs/prose_measurements.md`. Bench scripts that are not
  about prompts still cite it: `grep -rn "CLAUDE.md" bench --include=*.py`.

- **Two docstrings said a bare `workflows/*_api.json` glob misses nothing; it has
  missed `distill_experiments/` since 2026-09-27.** `bench/preflight_graph.py`
  said its single glob "currently misses nothing" because `GRAPH_DIRS` was
  `("",)`, and `bench/check_graph_discovery.py` said its rule held "while
  `GRAPH_DIRS` is `("",)`". `GRAPH_DIRS` gained `distill_experiments` on
  2026-09-27, so the first command skipped every distill graph, the ref2va and i2v
  ones included. Both docstrings are corrected and `preflight_graph.py`'s usage
  line carries both globs. Found while checking that those graphs are kept
  (`bench/results/2026-10-01_r2v_i2v_graph_coverage.md`: they are, and
  `bench/clip_recipe_coverage.py` now asks the question).

## 2026-09-29

- **Two ref2va findings re-checked by a second pass; details corrected**
  (board `ref2va-verify`, mrblue). Both of h3lora's findings mostly hold, and
  their records are left as written; this line is where a reader who trusted a
  detail is pointed. The records are
  `bench/results/2026-09-29_ref2va_partition_delta_verify.md` and
  `2026-09-29_ref2va_block49_verify.md`. What changed:
  - The block 49 record's attention-mass figures were read as made with the
    script's docstring command; they were made at `--queries 256` (the default
    is 384), and the median effective-key figure moves with sample size. Only
    `heads_eff_under_20` supports "peakiness alike".
  - The block 49 record's "the text rows' gates are exactly zero, so nothing
    trained them" was stated for the whole ref2va text span. That span includes
    vision rows tagged as video, whose block 49 modulation is not zero. The
    dropped-row claim is unaffected.
  - The partition-delta record's block-0 audio modulation "standout" is partly a
    small-denominator ratio, and the audio tail's rise starts near block 30, not
    block 45.
  - `2026-09-29_partition_delta_map.jsonl` predates its script: `cos` is above 1
    in about half its rows. Regenerate it before reading `cos`.
  - Stale lines in `next_steps.md` were corrected in the same pass: #36 "waits
    on the owner's go" (it was rendered and read, and its blind batch awaits the
    owner), and PDD8-finished-by-FastH3 "the scoring is what remains" (it was
    scored and parked, two lines up).
  - What the checks now support: ref2va's block 49 is the hardest INT8 cell
    mainly because the grade counts rows the model discards
    (`2026-09-27_sol_redesign_test2.md` did not say why), and "not a time warp"
    holds under a control that can fail (`bench/control_time_warp.py`).
  - Later the same day, on the real kernels (`2026-09-29_grade_video_audio_real_kernels.md`,
    all 56 heads): over the video and audio rows ref2va's block 49 is close to t2v's,
    so "the hardest cell" was mostly the grade. A first run at the grader's default of
    8 heads read the opposite and was withdrawn; the default is a trap for block 49.

- **FastH3 finisher and FastH3 mutant: parked** (owner). No more FastH3-finisher
  tests, and the FastH3 mutant (overlay plus loader, and shipped loader graphs)
  is on hold, because FastH3 still requires its full weights. This records
  what was tried, not a verdict on the method: PDD8 then FastH3 at 12/3 was
  indistinguishable from the FlashGen finish by eye on one t2v and one ref2va
  scene, and at 10/3 it was grainy
  (`bench/results/2026-09-29_blind_sessions_read.md`). Whether FastH3's finish
  adds anything on audio or motion was not measured. Built and kept: the
  finisher graphs, the overlay and its loader. Board: `fasth3-finisher` and
  `overlay-followups`, both parked.
- **FastH3's look travels with its gates, not its backbone change** (#35,
  measured, not judged by eye). `2026-09-27_fasth3_swap.md` put the look in
  "its weights and gates" and could not say which; the gates arm answers:
  fl2va plus FastH3's gate tensors behaves like FastH3, and FastH3 without
  them behaves like fl2va. Two predictions filed on 2026-09-27 (G1, G2) were
  falsified. It also closes the general adapter (`fasth3-adapter` on the board):
  the backbone change is not low-rank in bf16 and does not carry the look.
  Record: `bench/results/2026-09-29_fasth3_gates.md`; rank:
  `bench/results/2026-09-29_fasth3_bf16_rank.md`. One seed, three scenes, and
  the no-gates arm also drops VSA's coarse branch, so gate values against the
  branch's presence is open.
- **Block 49's INT8 problem is specific to unrotated attention** (owner,
  2026-09-29). It is sage's plain fp8++ and stock Sol's per-row K scale that
  meet block 49's four loud K channels badly; comfy-kitchen's `int8_attention`
  already rotates q and k and never had the problem, and the sage fork's
  rotated mode (`qk_rotate`, the sage node's `auto` since 0.129.0) was added
  to address it. `docs/h3_block49_quant_error.md` says so; the router row in
  `index.md` now does too, and the passages of `docs/SOLATTN.md` that read
  block 49 as a general last-block problem carry dated notes: the output head
  "reached directly" (propagation was measured 2026-08-29 and moves least for
  45, 48 and 49), the dense fallback described as sage, and the K-channel fold
  as "the move at block 49". Nothing was removed.
- **TaoMate-H3 is history only** (owner, again 2026-09-29; deprecated
  2026-09-27). `CLAUDE.md`, `docs/h3_quant_policy.md` and
  `docs/h3_block49_quant_error.md` said or implied it was a live reference for
  the tail's precision; each now says otherwise.

- **Eight nodes retired, `llmcompressor` dropped, 33 old bench scripts deleted**
  (owner, 2026-09-29; CHANGELOG 0.173.0 to 0.176.0). Each of
  `MiniMaxH3KeyframeCanvas`, `MiniMaxH3ReferenceFit`, `MiniMaxH3ReferenceVideoFit`,
  `MiniMaxH3MarkerArm`, `SageChainAssert`, `MiniMaxH3ReferenceReport`,
  `MiniMaxH3VSAAttention` and `MiniMaxH3QuantObserve` was in no shipped graph.
  Open experiment #23 is closed, not refuted; the AWQ and GPTQ recipe files that
  needed `llmcompressor` went with the dependency ("llmcompressor is for awq
  stuff - remove that dependency"); the bench scripts went on the owner's "most
  benches that havent been touched since august can probably go", applied as: last
  commit before 2026-09-01, no live reference, and not cited by a September
  record. What prose said before: `docs/custom_node_gaps.md` called
  `MiniMaxH3ReferenceVideoFit` "NOT deprecated ... still live", `SageChainAssert`
  "registered so saved graphs still load", and `MiniMaxH3VSAAttention` a live
  alternative to Sol; `docs/wiki/next_steps.md` told a reader to wire
  `MiniMaxH3ReferenceReport` before a reference render. Each now points at the
  replacement (`MiniMaxH3Conditioning`, `MiniMaxH3AppendRefImage`, core's
  `BlockSparseAttention` in `vsa` mode, the conditioner's preview). Saved graphs
  outside this repo that wire a retired node no longer load. `docs/roadmap.md`
  "Closed lanes" carries #23 and the VSA node.

## 2026-09-28

- **The launcher dropped `--fast fp16_accumulation`, and `comfy.env` dropped
  `NVIDIA_TF32_OVERRIDE=1`** (owner: "Yes make these changes"). Both DiTs
  compute bf16/int8 only, so the flag reached only the fp16 text encoders,
  and its `PRIORITIZE_FP16` made core PR 16508 an fp16 switch for H3. The
  override forced TF32 onto fp32 matmuls, the audio VAE's among them.
  `docs/sol_upstream.md`, `next_steps.md` and #33 used to say this launcher
  runs fp16 accumulation; each carries a dated note. The launcher and its
  reasons live in the dotfiles repo (`comfy/start.sh`, default mode).
  `bench/compare_vae_decoders.py` defaults to no `--fast` to match.

## 2026-09-27

- **PDD6 is a low-motion and close-up option, not the default** (owner:
  "closeups like that are fine for pdd6 - low motion, not a ton of stuff to
  keep track of"; chose "Low-motion option"). PDD8 won all three motion
  scenes: a sword touched on samurai, a missing hand on slapstick, morphing on
  subway (finding ow-fd-11). The graph stays;
  `../h3_distills.md`, "Rules of thumb", says when to use it.
- **The PDD8 finish: full FlashGen from sigma 0.8 on t2v, none on i2v** (the
  owner's review, ow-fd-08 and ow-fd-09).
  - On t2v the finishes were near indistinguishable by eye. Either FlashGen
    finish fixed sign text that PDD8 alone and the base finish garbled.
  - On i2v the finish's brightening "looked off", so PDD8 alone stays.
    *(Not the state of the code since 2026-10-03; see 2026-10-05.)*
  - Warm, it costs no sampling time: its evaluations replace PDD8's.
  - Whether it replaces the default t2v PDD graph is the owner's call.
  - `../../bench/results/2026-09-27_finisher_grid.md`, "The owner's review".
- **Closed, no change: late-only FlashGen as PDD8's finisher (#34).**
  Late-only matched full FlashGen as a finisher on seven scenes, and the full
  finish added no haze for it to remove
  (`../../bench/results/2026-09-27_late_switch.md`). The base finish (#37) has
  its own entry below.
- **Closed: late-only FlashGen alone (#46); no stripped LoRA is built**
  (owner's review, ow-fd-10).
  - Late-only was more natural on the single-figure beach ladder, and more so
    on the unusual rung, where full FlashGen walked in slow motion.
  - It lost coherence where people and objects must hold: a third person on
    subway_chase_short, and an unrecognisable piano on FT1's slapstick.
  - O2 (FlashGen overfit) stays open, not refuted. FT1's "half the haze" does
    not generalise at 124 frames (annotated in its record).
- **Answered for selection: #45, the kitchen's VSA against FastVideo's.**
  - At blocks 0 and 24 the kitchen matches a FastVideo-exact reference to
    within about 1%, so the selection deviations do not explain FastH3's
    over-polish.
  - At block 49 the gap is int8 on the text rows, not selection.
  - Building the reference node, to see whether that is visible, is the
    owner's call.
  - `../../bench/results/2026-09-27_vsa_selection_grade.md`.

- **The save format stays H.264 8-bit crf 19** (owner, #38). H.265 10-bit
  at crf 22 measured about half the dark blocking at half the size
  (`bench/results/2026-09-27_encode_format_ab.md`), and the owner saw no
  difference in the pair: "Dont think I can see a difference". The two
  encodes differ by under one level in the darks. So O1's `dark_block8`
  excess of about +0.05 is below the owner's eye (vd-m12), and O1's
  "splotchy blacks" are not the codec's 8 px blocking; that question is open
  on a clip the owner names.
- **PDD8 then full FlashGen from sigma 0.8 becomes a top-level t2v graph**
  (owner: "then yeah switch it"; 0.167.0), as
  `workflows/h3_text_to_video_pdd8_flashgen_finish_api.json`. The finisher
  grid review found it better than PDD8 alone on three t2v scenes and never
  worse. Full and late-only finishes were indistinguishable, so the full
  one, as trained. Fresh runs sample in the same time as PDD8: its 6 PDD8
  evaluations plus 2 FlashGen ones are 8, like PDD8's own. It is an additional
  graph: `h3_text_to_video_pdd` stays the default until the owner says
  otherwise. i2v stays PDD8 alone, because the finish brightens the frame at
  once. *(Not the state of the code since 2026-10-03; see 2026-10-05.)*
- **The base-model finish (#37) closes with no change** (owner's review,
  vd-v01). It was never preferred, and it morphed a face on courtroom. It is
  not inert: it matched the FlashGen finishes on noodle_bar and partly fixed
  spec_unusual's sign text. PDD8's coarse tail is part of its weakness, and
  FlashGen fixes it better. The measures called it gentle and missed the
  legibility gain. `step_switch_to="base"` stays in the generator for the
  bench.

- **Sol's default quantizer is `rotated`** (owner, 0.166.0). It was
  `balanced`, inherited from `qk_balance=True` (2026-09-15). The owner chose
  it from test 2's grade on the shipped PDD8 graphs: lowest quantization
  error on every Sol-block cell, cheaper than `balanced`, small effect on
  total error. The owner said to switch at the next ComfyUI restart, with the
  reviews done. `bench/results/2026-09-27_sol_redesign_test2.md`.

- **Our conditioning nodes strip the prompt at both ends** (owner: "strip
  leading and trailing whitespace in our conditioning nodes and \n at the end
  (not in the middle - \n has value in the middle)"; "make sure no prompts in
  prompt_bank have the trailing \n"). `h3_rules.normalize_prompt`, in
  `a5ab229f` (0.160.0). Core's tokenizer keeps edge whitespace, so a trailing
  newline was one more token and a different sample at the same seed; the
  generator already stripped bank prompts. The bank check now fails on edge
  whitespace. Core's own H3 nodes still do not strip.
- **Closed: the encoder precision study.** `ENCODER_INT8` stays the default.
  At the int8 DiT, int8's effect sits under a floor that a dose-response
  confirmed (`data/results/2026-09-27_encoder_quant_floor.json`). Floor-free,
  at the bf16 conditioning path, its image-token error is no worse than random
  and its text-token error is somewhat worse
  (`data/results/2026-09-27_encoder_quant_refiner.json`).
  `docs/open_experiments.md` #47 closes unbuilt.

- **`qk_balance` is not inert behind the dense tail** (Sol redesign test 1,
  0.163.1). The audit (§4 item 1), the redesign doc, `docs/SOLATTN.md` and
  `docs/h3_block49_quant_error.md` had said or implied that the balance gate
  opens only on blocks 45, 48 and 49, so the shipped dense tail left it
  nothing to do. Renders in three modes and a gate measurement on captures
  refute that. Each of those docs now keeps a dated note.
  `bench/results/2026-09-27_sol_redesign_test1.md`.

- **TaoMate-H3 is deprecated and removed** (owner: "taomate is deprecated
  you can remove everything from it here. we're not gonna pursue it
  anymore."; 0.161.0). Removed: the `MiniMaxH3TaoMateStreamSampler` node and its runtime
  (`taomate_stream_sampler.py`, `taomate_streaming.py`), the two probe graphs
  (`h3_probe_taomate_3step`, `h3_probe_taomate_3step_audio_freeze`), the
  `TAOMATE_*` constants in `workflows/h3_config.py`, the check
  `bench/check_taomate_streaming.py` and the TaoMate cases in
  `check_distill_settings.py` and `check_distill_grid.py`, the converter
  `bench/convert_taomate_lora.py`, the harness
  `bench/verify_taomate_stream.py`, the three arm manifests, and
  `docs/h3_taomate.md`. Deprecated by the owner, not pursued; not a verdict on
  the method. The records stay in `bench/results/` (`2026-09-15_taomate_*`),
  the code in git, and the lane is in `docs/roadmap.md` "Closed lanes". The
  upstream checkouts stay as references (`references.md`, "The streaming
  references: TaoMate").
- **The owner's working encoder is the bf16 pruned file; a better int8 of our
  own is not worth building** (owner: "since the encoder runs once, I agree -
  it may not be worth the effort. Especially if we can run a pruned bf16. I
  care mostly about its vision encoder not losing precision"). Both encoder
  files had been deleted by accident. The int8 file was restored from the
  owner's Comfy-Org download, whose sha256 matches the one the Hub recorded.
  The bf16 pruned file was rebuilt from the release by
  `bench/convert_h3_bf16_encoder.py`, which reads the sharded `text_encoder/`
  directly since `c743d100`. The weight-side headroom record already put the
  shipped int8 at its format's floor
  (`data/results/2026-08-29_int8_convrot_headroom.json`), so the encoder
  quantisation lane stays closed.
- **Corrected: what the 2026-09-20 entry below says `ENCODER_INT8` was
  measured against.** It was only ever graded against the two W4A16
  artifacts. It has now been measured against the bf16 pruned file on
  shipped graphs and on the violin-maker scene in three modes
  (`data/results/2026-09-27_encoder_int8_vs_bf16_conditioning.json`).
  - The vision tower and embedding table are bf16 in both files and add no
    error.
  - Text tokens stay close.
  - Image tokens carry a heavy tail that the int8 decoder adds, and one
    violin ref2va patch is inflated well past its bf16 norm.
  - bf16 costs seconds per encode.
- **Running: does the DiT respond to that tail** (owner: "The experiment
  would be good"). `bench/measure_encoder_quant_dit.py` (`3acade9e`) runs
  the base DiT at fixed noise, latent and sigma with a null, the treatment,
  an image-tokens-only arm, a norm-matched random control and a one-edit
  scale row. Predictions were filed before any comparison printed. The
  decision it gates is in [`next_steps.md`](next_steps.md).

- **The Sol redesign ships as `MiniMaxH3Sol`, and `MiniMaxH3SolAttn` is
  deleted once the generated graphs move** (owner: "1 delete", "3 your pick on
  name"). The kitchen build branch `h3-frontier` is pushed to the owner's
  GitHub fork and `nas`, never to Comfy-Org ("Never upstream to comfy").
- **Closed: the SLA lane and Morton** (owner: "SLA is closed unless you see
  some reason not to"; "Morton seems dead too unless you see remnants of it").
  Checked first:
  - kitchen's top-k has been unchanged since 2026-09-04, and it lacks SLA's
    learned linear branch;
  - no H3 implementation in core, kitchen or `coderef/` reorders tokens.
  The new Sol node drops top-k, `keep_percent`, `pooled_tail`, `morton`,
  `morton_curve` and the code only they reach. Both are in `docs/roadmap.md`
  "Closed lanes". The redesign's tests use **test renders, not blind panels**
  (owner: "I dont need blind renders just test renders").
- **Approved: redesign `MiniMaxH3SolAttn` as a new node and freeze the old one**
  (owner: "I'm good with this approach"). Retiring an input retires the code
  only it reaches, with its checks and prose ("since its all git tracked
  anyway"). Morton becomes one `reorder` input, with one test as a speed
  lever before removal. The plan, reasons, tests and bug list:
  `docs/research/2026-09-27_sol_node_redesign.md`.
- **Reversed: the kitchen build tracks upstream main, not ComfyUI's pinned
  tag** (owner: "I want to be able to stay on the frontier here... So long as
  we know what exists where and why"). It used to be that untagged main was
  "not built by policy" (`vendor/rebuild_kernel.sh`, since 2026-09-11). That
  kept #207/#208 (kitchen int8 attention; #208 adds an Ada-only cached-Q path
  at head_dim 128 with no mask, which is how H3's dense backend calls it) and
  #192's `fp16_conv3d` depth gate out for a release cycle (the gate turned
  out not to reach the tiled H3 encode:
  `bench/results/2026-09-27_vae_encoder_fp16acc.md`). Now we build
  `h3-frontier`, upstream main plus our Sol commits, moved forward by merge.
  The installed version names both halves (`+sol.<ours>.up.<base>`), and the
  build record lists the base, its distance past the pin and the carried
  commits. `h3-build` (8176242) is the retired tag-based line. A rebase of it
  was refused as a destructive rewrite of a shared clone, which is also why
  the branch moves by merge. 0.158.0.
- **Corrected: `vsa_attention.py` "cannot run", its defect "unreachable".**
  Stock core has built `to_gate_compress` since e308cc73 (#16072), so the old
  refusal passes and the `_publish_layout` leak is reachable. The node is
  parked by an explicit refusal (0.157.1). The remaining stale prose (the
  module docstring, `docs/research/vsa/vsa_node.md`, and
  `check_vsa_core_patch.py`'s "applied from the draft PR" message) is listed
  for correction.
- **Corrected: the VSA prose, following the entry above.**
  `vsa_attention.py`'s docstring, description and refusal said the node cannot
  run, core has no `gate_compress`, PR #15958 is an unmerged draft, the defect
  is unreachable, "nothing has been rendered", the T8 pack is "the only other
  implementation", and keep 10 is "VSA's published 0.90 sparsity". They now
  say it is parked, core builds the gate since e308cc73, it rendered on
  2026-08-30 on a draft-patched core, and FastH3 V2 trains at keep 20
  (`h3_config.FASTH3_CONTRACT_VSA`). `docs/research/vsa/vsa_node.md` said core
  was stock without the gate and the blocker stood until #15958 merged; it now
  says the same as the module. `bench/check_vsa_core_patch.py` said present
  support was "applied from PR #15958 ... DRAFT"; it now says stock core has
  it and that the check cannot tell stock from a local edit.
- **Corrected: `docs/SOLATTN.md` and `docs/wiki/next_steps.md` on Sol against
  core.** The upstream-policy table had the `dense_blocks` cells swapped
  (core's column held our `SOL_DENSE_TAIL`, ours said empty); the option table
  said the node and configs "ship empty as of 2026-09-02" and called
  `qk_balance` off and a declared deviation of the policy graph; the kernel
  table called `sol_attn_chunked` structurally unreachable. Now: core's
  default is empty and ours is the tail; empty was 2026-09-02 to 2026-09-25;
  `qk_balance` is off in the node and shipped on; `MiniMaxH3SolChunked`
  reaches the chunked kernel. Both pages said core's and our Sol outputs
  "differ by no more than the all-routed floor" and that ours as shipped "sits
  closer to exact" than core's defaults. The 2026-09-10 record
  (`data/results/2026-09-10_sol_impl_capture_grade.json`, the `vs_ours`
  fields) has them above our floor in most cells and in every carried cell,
  and ours at that day's policy losing on block 0 and the per-row mean while
  winning the whole-tensor mean through block 49; that is what they say now.
- **Corrected: Sol code prose.** `sol_chunked_h3.py` and
  `bench/check_sol_chunked.py`'s docstring said the producer's routing
  threshold is stale; it is the K centring mean and the V scale (with
  kitchen's margin) that come from the previous step, and `kcvar` is current.
  `sol_attn_h3.py` said every local change since the fork was listed and
  named sage as the fallback; it now says only the fork-time changes are listed,
  and names `ModelAttentionBackend` (comfy kitchen attention) on the default
  graphs. `workflows/h3_config.py`'s `min_tokens` comment named sage as the
  fallback on every graph, and `SOL_CUDA_DEFAULTS` was called what the node
  "gives you untouched" while it pins `qk_balance` True; comments only, no
  value changed. `docs/sol_upstream.md` cited core's `model.py` a line early
  (:168-171, :623, :754; now :169-172, :624, :755) and now notes the V
  scale's clip margin.
## 2026-09-26

- **Distill research graphs moved to `workflows/distill_experiments/`**
  (owner, 2026-09-27; 0.157.0). 49 graphs, routed by
  `build_workflows._is_distill_experiment`; the everyday distill graphs stay
  at the root. The TaoMate graphs failed the live validation because their
  LoRA files are no longer on disk, so that build was written without
  validation; every other graph validated.
- **FastH3's pruned file does not share fl2va's curve basis** (measured,
  `bench/results/2026-09-26_fasth3_weights.md`). Its `adaln_t_table` and
  every `adaln_proj` differ from fl2va's by several times their own norm, so
  the conditioning moves between checkpoints only as a whole set
  (`bench/build_adaln_swap.py`). `h3_config.MODELS["unet_fasth3_v2"]`'s
  comment said it was "pruned int8 convrot on the fl2va curve basis".
- **The lightx2v turbo and Turbo-SLA LoRAs are retired, and turbo LoRAs are
  a closed lane** (owner: "lightx and turbo stuff irrelevant"; on SLA, "i
  thought we stopped that ages ago"). The distill lanes are FastH3, PDD and,
  undecided, FlashGen. Fourteen graphs went (0.156.0): `h3_text_to_video_turbo`,
  `h3_probe_turbo_768p_owner`, `_sla`, `_sla_dense`, `h3_probe_turbo_home_canvas`,
  `h3_probe_t2v_turbo_lx12_sage`, `h3_probe_split_base_first` and `_last`,
  `h3_probe_ref2v_turbo`, `h3_image_ref_plus_text_to_video_turbo_4step`, and
  the four `h3_probe_ref_turbo768p_*`. What prose claimed and no longer does:
  `h3_config.py`'s shift note carried the five-row lightx2v shift table and
  said the 768p students were "the trap" at 6/3; `docs/SOLATTN.md` counted two
  shipped graphs under the token floor; `docs/open_experiments.md` #20 said the
  two shipped SLA arms addressed its open half; `docs/h3_references.md` named
  `h3_probe_ref2v_turbo` as the one ref2va exception. The SLA probes had been
  generated since 2026-08-20 and never rendered. `check_distill_settings.py`
  now fails any graph loading a turbo LoRA, and its vendor-row grading left
  with the rows. The 2026-09-05 turbo rung closes unscored; its records and
  blind sessions stay. The model files are still on disk; deleting them is
  the owner's call.
- **Bug, fixed in 0.154.8: the exact branch stacked on another model's
  branch.** When one server process rendered Turbo, then PDD, then FlashGen,
  each wrapped the previous model's still-applied forward. The owner caught it
  by eye ("flashgen looks awful blocky"; "it's the code"). My first explanation,
  a regression between 13:05 and 16:07 read off a 16-pixel grid metric, was
  wrong: the owner named a clip it rated blocky as fine. The contaminated
  renders are listed in `bench/results/2026-09-26_followup_contamination.md`.
- **Owner: every LoRA on an int8 checkpoint goes through our exact branch**
  (0.154.0). This covers PDD's backbone and refiner (`MiniMaxH3PDDLoRA`
  `backbone_apply`), and the generator's `lora_branch` now defaults on.
  Before this, the branch was FlashGen's alone, PDD merged everything, and
  `unmerged_blocks` was an off-by-default knob that could not reach
  `mlp.fc2`. The measurement behind it:
  `bench/results/2026-09-26_int8_lora_requant.json`. Merges survive only on
  the two control probes. The policy lives in `docs/h3_quant_policy.md`.
- **Corrected: the audio-refine pass does keep frozen video exact.** A
  one-execution probe measured it
  (`bench/results/2026-09-26_frozen_row_probe.md`). The 2026-09-25 record
  said its refine arms reused pass 1 from cache, and the 46 dB gap was read
  against that; in fact pass 1 re-rendered and differed. Dated notes are in
  that record and in `docs/research/2026-09-26_distill_routing.md`. The open
  question is PDD8's run-to-run reproducibility. FlashGen reproduces bit for
  bit across servers.
- **Draft decodes for scouting are declined** (owner: "i dont think its worth
  it no matter what. thats probably why its always used as a preview node
  only"). `docs/open_experiments.md` #31 is closed as declined, and the draft
  graph `h3_text_to_video_flashgen_draft` is retired (0.153.1). Latent saving,
  the `draft_decode` switch and `bench/decode_draft_keepers.py` stay. The
  owner also set aside previews for now ("dont worry about adding preview
  stuff").
- **Every flagged prompt fixed, the bench prompt included** (owner: "Yes fix
  all"). `t2va_frontier_standoff` is the bench and baseline prompt, so the
  dense baseline graph's text changed (0.151.2). A baseline render before
  0.151.2 is not a same-prompt comparison with one after. The refview2 ablation
  prompts changed too, while their blind session is pending; the clips already
  rendered keep the old text.
- **The prompt bank is fixed for the subway-chase defect class** (owner: "we
  need to fix all our prompts"). 38 of 139 prompts got minimal edits, graded
  unchanged. The composed ref2va prompts, the refview2 ablation scenes (a blind
  session is pending) and anything whose fix changes what it tests are listed
  for the owner in `next_steps.md`, unedited. 0.151.1.
- **Corrected: the no-header-timestamps rule has been measured once.**
  `prompting.md` section 3.1 and the portable standard said it was "not yet
  measured in either direction". `bench/results/2026-09-18_timestamps_diner.md`
  had measured one same-seed pair on the rule's own day, and the owner
  preferred the timestamped take slightly. The owner keeps the house rule
  knowingly (asked 2026-09-26, answered "1"). A blind multi-scene test is the
  open alternative.
- **`t2va_subway_chase` rewritten** (owner, after the three-way distill look:
  "make sure its super clear and specific about what is going on in this
  scene").
  - **The scene now states its cast:** exactly two people, and the suspect
    never speaks.
  - **One physical path**, turnstiles then stairs down then the platform, where
    shot 2 had the agent on stairs and shot 3 looked up an escalator nobody was
    placed on. The owner saw an escalator chase in the wrong direction and a
    cloned pursuer.
  - **Every action has an agent** (`prompting.md` section 15.3, items 6 and 7).
  - It grades 0 FAIL and 0 WARN, as before. Every subway render before this
    entry used the old text, and each row records the prompt hash.
- **The shipped video VAE is the INT8 ConvRot build** (owner: "yes switch"),
  `h3_config.MODELS["video_vae"]`, 0.151.0. It reverses the 2026-08-21
  removal (`bc25d89`), and the closed-lanes row in `docs/roadmap.md` is gone.
  `VIDEO_VAE_FP16` keeps the fp16 name for the comparison tools.
  `MiniMaxH3VAEPrecision` now refuses to cast a quantized half. Prose that
  named the fp16 file as shipped carries dated notes in `next_steps.md` and
  `capture_manifest_schema.md`.
- **A distill runs on its trainer's contract, not a downstream template**
  (owner, approving the postmortem's proposal). The rule and its instance are
  in `references.md`, "A distill's reference is its trainer's contract", and
  CLAUDE.md points there.
- **The owner could not tell the INT8 ConvRot video VAE from fp16** on one
  345-frame latent, decoded both ways and watched sighted
  (`bench/results/2026-09-26_vae_decoders_345f.md`). The lane was reopened
  for measurement the same day. The shipped VAE is still fp16 until the owner
  decides to switch.
- **Corrected: where the server log lives.** `docs/comfy_notes.md` said
  `user/comfyui_<port>.log` always holds the current session. The server
  started 2026-09-26 by `start.sh` wrote only to its stdout pipe, and that
  file stopped at 2026-09-16. The paragraph now names `/internal/logs/raw`,
  core's in-memory buffer, with a dated note in place. The same claim was also
  in the restart recipe ("redirecting the launcher to `/dev/null` loses
  nothing"), now corrected with its own note.
- **Pipeline telemetry exists** (owner: "we should know whats happening
  throughout our pipeline in a structured schema and be able to record it end
  to end"). `H3_TELEMETRY` is `docs/pipeline_telemetry.md`. Two things said in
  session that day were wrong, and are corrected there rather than repeated:
  "Staged" is reserved address space on the card, not data held in RAM; and
  `--mmap-torch-files` does not affect `.safetensors`, which dynamic VRAM
  memory-maps anyway.
- **Corrected: FastH3 V2 was built to ComfyUI's template, not FastVideo's
  contract.**
  - `docs/wiki/references.md` said the FastH3 file was "Not on disk here, and
    no graph of ours loads it". A dated note there now says what does.
  - `docs/research/vsa/fastvideo_vsa_checkpoint.md` quoted Preview v1's 0.9
    sparsity with nothing saying V2 is 0.8.
  - The `h3_config.FASTH3_CORE_VSA` comment said the template "wins here".
    It now names the contract arms, `h3_config.FASTH3_CONTRACT_*` (0.148.0),
    and the record that compares them.
- **FlashGen ships as `h3_text_to_video_flashgen`**, with rank 64 applied at
  the call (owner: "lets take our best stab at making a good workflow ship",
  while unable to look). Chosen by source rather than by eye:
  - the int8 merge keeps little of FlashGen's delta
    (`bench/results/2026-09-26_int8_lora_requant.json`);
  - vllm-omni's native route applies the LoRA at run time.

  The render pair is unjudged (`bench/results/2026-09-26_flashgen_lora_path_s1.md`).
  If the owner prefers the merged take, the loader swaps back. 0.146.0.
- **The shipped refine graphs carry the frozen-video cache** (owner, on the
  unblinded pair: "Cant hear a difference whatsoever"). A practical default in
  a tinkering repo: it roughly halves the refine sampler
  (`bench/results/2026-09-25_frozen_cache_s1.md`). The PDD8 and FlashGen
  refine probes wire it. `h3_probe_t2v_flashgen_4step_audio_refine_uncached`
  replaces the `_cached` probe as its control. 0.144.0.

## 2026-09-25

- **Sol's default `dense_blocks` is `45,48,49`** (owner: "why wouldnt we set
  dense blocks to 45,48,49 by default?", then "do both"). A practical default
  in a tinkering repo, overriding `h3_config`'s earlier condition that the
  shared default stay empty until all 50 blocks are measured and a set
  validated; that condition is kept with a dated note. The kitchen-chain
  render of the tail is unscored. `h3_probe_t2v_no_dense_tail` is the control.
  0.142.0. *(2026-10-02, re-evaluated with real data under `quantizer="rotated"`:
  On an unmasked 50-block Ref2VA capture at sequence length 119,102, Hadamard
  rotation resolved the unrotated K-norm outlier problem, dropping block 48
  to 7.14% avg relative $L_2$ error [max 9.41%] and block 45 to 9.55%. In contrast,
  routing sparsity truncation error heavily concentrates in the middle semantic
  integration cluster: block 42 at 25.67% [max 27.01%], block 39 at 25.22%,
  block 41 at 24.35%, and block 40 at 23.43% [worst heads losing 57%–80% magnitude
  with min cosine down to 0.77]. Keeping block 48 dense while leaving 39–42 sparse
  wastes dense compute. Block 49 remains mandatory as it feeds `final_layer.video_out`
  [13.73% error]. Top candidates under active trial: Option A `"39,41,42,49"` (4 blocks)
  and Option B `"39,40,41,42,49"` (5 blocks). See 2026-10-02 entry and
  [`../research/sparse/sol_dense_blocks_reanalysis.md`](../research/sparse/sol_dense_blocks_reanalysis.md)).*
- **`token_routing` reworked into one dropdown** (owner: "that token routing
  field UX is confusing"). `off` replaces "text field" as the default, and the
  list is read only under `custom`. The old default meant off by an empty list,
  which is the no-sentinel rule's shape. 0.142.0.
- **Reference-contract case 5c retired** (owner). It asserted that core's
  forced-CLIP-schedule branch drops `minimax_token_tags`: `comfy/sd.py` read
  only `o[:2]` there, so a graph wiring CLIP hooks would tag every row as
  text. Core `6bfaacc6` (#16400, 2026-09-20) merges `o[2]` on that branch too,
  the case went red, and its docstring said to retire rather than repair it.
  `comfyui_vendor_gaps.md` gap 2b, `custom_node_gaps.md` ("Core holds four")
  and `checks.md` described the gap as live and are corrected.
- **Continuation note corrected: LongMedia's temporal offset is partial.** It
  leaves a keyframe's audio and a reference video's frames unshifted, which
  desyncs both. The note had the first second-hand and missed the second
  (`../research/2026-09-25_temporal_offset_and_adaln_rounding.md`).
- **The FastH3 V2 symlink was repointed** from a relative target, which did not
  resolve inside `models/diffusion_models/`, to the file on the Storage share.
  The owner had placed it; the file is complete.
- **The stale Kijai PDD symlinks were removed** (owner). The four
  `MiniMax-H3-*-Acc-8Step*_comfy.safetensors` links in the ComfyUI checkout's
  `models/loras/h3/` pointed at the pre-2026-08-27 upload, which merged core
  decodes to a doubled head. No graph named them. The files are still in the
  Storage share.
- **Upstream reports are drafted, not posted** (owner: save them internally).
  The sglang PDD builder's fc1 swap and bf16 heads were confirmed by running
  sglang's own tools; the drafts are in the gitignored
  `internal/upstream_reports/`.
- **PDD reopened for research** (owner: "dig into the upstream stuff's PDD
  (not only comfy) to see if we can do better than comfy native AND our
  current"). `docs/roadmap.md` "Closed lanes" had PDD quality work parked
  since 2026-09-05. The reopen covers reading and CPU work; a render in this
  lane still waits on the owner's go. Also reopened for research: continuation
  by guide rows, the tile-seam question and the upstream digs (the upstream
  survey's buckets 3 and 5), framed as evidence for the owner's open decisions.
- **The PDD audio change-of-variable mechanism is refuted, and prose resting on it
  is corrected** (`../research/pdd/2026-09-25_upstream_pdd_comparison.md`; numbers in
  `bench/results/2026-09-25_upstream_pdd_comparison.md`, section 3; commit
  `aa9f2420`). Each has a dated note in place.
  - `audio_under_pdd.md` section 1 and "vary the TRANSFORM" said the block-start
    carry is an audio-only error growing with width. Core's carry equals the
    vendor's two-schedule audio Euler step exactly, so the loss is PDD's own.
  - `h3_pdd.md` "Why audio suffers more than video" derived the same mechanism.
  - `evidence.md`'s PDD bullet ended "vary the transform at fixed partition";
    withdrawn, and a do-not-rely row added.
  - `audio_carry_probe.py`, `bench/measure_pdd_audio_carry.py` and
    `bench/run_audio_carry_arms.py` described their premise as a correction.
    Their measured effects stand as departures from the exact path.
  - `pdd_math.py::fuse_heads` and `pdd_lora.py` said the vendor fuses in bf16;
    diffusers keeps the heads fp32.
  - `pdd_implementations.md` section 4.1 called the stale copies the vendor's;
    they are Kijai's conversions, re-uploaded upstream 2026-08-27T21:25Z. The
    "reported, not verified" notes in `2026-08-28_handoff.md` and
    `comfyui_h3_t2va_trace.md` are now verified.
- **Prose corrected by the 2026-09-25 upstream survey**
  ([`references.md`](references.md), "What moved by 2026-09-25";
  [`../sol_upstream.md`](../sol_upstream.md), "comfy-kitchen and core,
  2026-09-25"; [`../research/sglang_comparison.md`](../research/sglang_comparison.md),
  "Seventh read"). Each has a dated note in place. Commit `d226ec0f`, with
  its checks recorded in
  `bench/results/2026-09-25_upstream_survey_checks.md` in the follow-up.
  - "No engine implements PDD" (`references.md`, `pdd_implementations.md`
    section 1, the `index.md` row): sglang has implemented it since
    `973fb44471` (2026-09-23).
  - `h3_pdd.md` "Core is learning this" read Comfy-Org/ComfyUI#15908 as open.
    It merged 2026-08-29. The question it said would go live on merge, whether
    our node keeps the head half, has been live since then and is undecided.
  - `references.md`'s vllm-omni #7693 bullet named a function #7913 deleted,
    and called the rounding SM90-only; vllm-omni now keeps fp32 on every arch.
  - `h3_audio_freeze.md` section 3, "No shipped graph writes a `noise_mask`":
    stale since the freeze nodes shipped.
  - The rotation survey's E.5, "LightX2V refuses any feature cache on H3" and
    ships Sol and caching "never combined": `8652c6f1` ships DPCache with Sol.
  - `sglang_h3_pipeline.md`, FastH3 "Registered under `registry.py`": its
    config now registers from its own pipeline-config file.
  - `sol_upstream.md`, SubBlock's `sage_fp8` as SM90 only: now SM90 and SM120.

## 2026-09-23

- **Video files carry no metadata; the graph lives in the first-frame PNG**
  (owner's policy, "metadata saved in png files - that's it"; the VHS fork
  follows it). `loop_output.py` stopped embedding a `comment` tag and
  `creation_time`; 0.138.0. Prose that said the mp4 carries the graph was
  corrected in `loop_output.py`, `audio_freeze_song.py` (docstring and the
  `save_metadata_png` tooltip), `bench/diff_clip_graphs.py`,
  `bench/verify_vsa_render.py` and `docs/eval_comparison.md`; dated notes were
  added in place in `docs/h3_pdd.md` and `docs/research/vsa/vsa_node.md`,
  whose two-mechanism findings still describe every older clip.
- **Reference-video graphs load through VHS's ffmpeg loader** (owner's
  request, relayed by the dotfiles session). `h3_config.REF_VIDEO_LOADER`;
  0.137.0. The loaders keep different frames on 25 and 30 fps sources, so
  renders from before and after are not substrate-comparable; the record is
  `data/results/2026-09-23_vhs_loader_comparison.json`. Prose that named
  `VHS_LoadVideo` as the reason for the choice ("the one that exposes
  `force_rate`": both do) was corrected in `build_workflows.py`,
  `docs/h3_references.md` and `docs/comfyui_vendor_gaps.md`.
- **The larryvrh turbo pack is gone** (owner removed it from the install and
  said to remove the graphs that use it). Three probe graphs, the generator's
  `turbo_pack` path and the `TURBO_PACK_*` constants; 0.137.0. The finding
  that lived beside those constants in `h3_config.py`: the v4 LoRA cannot be
  loaded by `LoraLoaderModelOnly` on our checkpoint, for two independent
  reasons. Its keys are bare where `comfy/lora.py` expects a
  `diffusion_model.` prefix, so nothing loads. And its 51 `adaln_proj` modules
  were trained full-width against a curve-form base whose adaln input is
  8 wide, so no loader can add them as a weight patch. That is why the pack
  shipped its own `silu(t_emb)` grid. The full comment is in git at
  `workflows/h3_config.py` as of 0.136.2. `docs/h3_ref2v_distillation.md`
  now says the ref2va-with-v4 probes were retired with the pack before any
  render of them was judged, so that question is unanswered, not closed.

## 2026-09-20

- **The sage fork's masked CUDA path is wrong, and this pack's refusal to use
  it is now a correctness guard rather than a scoping choice** (found while
  building a sage override for ComfyUI's Qwen-Image 2.1, which is the first
  target either repo has that passes a mask). `sageattn()` routes masked sm89
  calls to the fp8++ kernel's `MaskMode::kGeneral`, added in the fork's v0.5.5;
  that path applies `attn_mask` to the last 128 key columns only and silently
  ignores everything before it. It has never been correct -- not a regression.

  *(Updated later the same day: the kernel was fixed -- `apply_general_mask`
  sat at `apply_causal_mask`'s call sites, which is a diagonal-bound
  optimisation that does not generalise, so the main loop masked nothing. The
  unmasked and causal paths are bit-identical across the fix, so nothing below
  about H3 changes. This pack still declines masked calls, now because Triton
  skips fully masked K blocks and the CUDA kernel does not, making it faster
  and more accurate on masks rather than the only correct one.)*

  *(Corrected 2026-09-22: the reason in the note above named the wrong
  mechanism. The fork's own backlog entry "Route masked sm89 calls to Triton
  rather than fp8++" puts Triton's accuracy edge on PV quantized to fp16
  rather than fp8. It also finds Triton winning on a dense mask, where
  skipping blocks buys little, so the speed gap is not isolated. The
  conclusion, that Triton is the masked path to use and this pack keeps
  declining, is unchanged; `attention.py` now states the reason without
  claiming a mechanism. Also, "last 128 key columns" was a fixed-shape
  reading: the fork's `docs/cuda_mask_kernel_scoping.md` finds the window
  tile-aligned, the last two K blocks, so its width varies with kv_len.)*

  **H3 is unaffected, by construction rather than by luck.** Its single
  attention call site passes `mask=None` as a literal, not a variable
  (`comfy/ldm/minimax/model.py`, `Attention.forward`), so no code path reaches
  the defect. `denoise_mask` and `audio_denoise_mask` are per-row velocity and
  sigma scaling applied to the output; they never touch attention. **Multiple
  references lengthen the packed sequence and do not introduce a mask** --
  checked specifically, because it is the obvious place a mask would appear.
  Nothing rendered by this pack needs re-checking.

  What changed here: the comment in `attention.py` above the mask decline used
  to say "Sage has no mask support on this path", which is now false in detail
  and right in effect, and read as an invitation to relax the guard once sage
  gained masks. It now says why the guard stays. No behaviour change; the
  override already declined masked calls.

  The one correct masked entry point is `sageattn_qk_int8_pv_fp16_triton`. Note
  the naming trap: fp16 *triton* is correct, fp16 *cuda* silently drops masks
  whole, and the mode names do nothing to warn about it. If this pack ever
  grows a masked path, that is the only one to use.

  Lives in the sage fork: `docs/cuda_mask_kernel_scoping.md` (owner),
  `CHANGELOG.md` "Known kernel bugs" (record),
  `tests/repros/repro_fp8_mask_window.py` (the gate, exits non-zero while
  present), `tests/bench/masked_kernel_survey/` (every entry point against a
  causal, a prefix and a suffix mask).


- **The AWQ encoder comparison is about two badly executed artifacts, not about
  the method, and it must not be read as evidence against quantising our own
  encoder** (owner, 2026-09-20, in their words: "we poorly quantized that - it
  should NOT be proof of US quantizing being a bad thing... We control the
  calibration if quantize, and group size, and everything else! We just did it
  super poorly when we did it with AWQ, and then priorities shifted"). Every
  record of the four-encoder holdout stands as measured; what changes is the
  inference drawn from it. The lane closed because a poor first attempt
  coincided with attention moving elsewhere, not because the approach was shown
  to be unpromising, and nothing in `bench/results/` ever established the
  latter. **Calibration data is the untapped area**, and the part our own
  programme did worst; the recipe is where the effort went and not where it was
  lost. Annotated in place at `workflows/h3_config.py::ENCODER_INT8`,
  `docs/evidence.md`, `docs/roadmap.md` "Closed lanes", `docs/rules_history.md`,
  `bench/results/archive/v2_encoder/README.md`,
  `docs/research/archive/awq_quantization_suite/README.md` and its report, and inside
  `data/results/2026-08-25_four_encoders_holdout_layer50.json` itself.
- **All four arms of that holdout hold the vision tower at BF16**, which the
  record did not say and a reader could not infer from it. Read from the file
  headers: `int8_convrot` and `nvfp4_awq` carry `.comfy_quant` markers on
  language-decoder linears only, and both W4A16 arms carry none at all. So the
  holdout compares decoder treatment and load path, and is **silent on tower
  precision**. A session reading it on 2026-09-20 concluded the opposite and
  had to retract. **The owner's position is that the vision tower needs to be
  BF16, for several reasons** (owner, 2026-09-20); that is a standing position,
  not a measurement, and no record here measures it.
- **The owner primarily runs the BF16 Qwen3-VL encoder**, although
  `h3_config.MODELS["clip"]` ships `ENCODER_INT8` on every graph (owner,
  2026-09-20). The shipped default is not the owner's working configuration, so
  a default here is not an endorsement: `ENCODER_INT8` won a holdout against
  two artifacts the owner calls badly executed.
- **`bench/check_model_contents.py` asserts what is inside every model
  `h3_config` names**, against a committed fingerprint baseline.
  `check_model_files.py` resolves names through `/object_info` and never opens
  a file, so a reship or a rebuild under an unchanged name was invisible to
  every check. `--report` prints the census, including matched-shape controls
  and their absence. The two text encoders are the only files here carrying no
  control, which is why neither can settle a role-versus-shape question about
  the tower.

- **The prompt rules of 0.128.0 are written down, and four of the six turn out
  to be the vendor guides' own** (the owner: "the rules should be what's in
  `vendor_guides/`"). `4bd7b429` fixed six wording classes across the bank and
  touched no rule-stating file, so the rules existed only as edited prompt text.
  Read back against the guides: **reusing `(Sx)` at every vocal event is STATED**
  by ref §5.4, as is the exemption that a verbal cue inside a reused soundtrack
  takes `<Audio N>` and no new id; **a subject's position in the frame is
  STATED** by base §4.1 and ref §5.3; a voice descriptor contradicting the
  identity is a reading of base §4.4's "stable identity" list. **base_en states
  neither reuse nor its absence**, so the same text can be conformant as a base
  prompt and not as a ref2va one (`docs/prompting.md` §12.13). Agentless actions
  and contradicted counts are in neither guide and stay HOUSE, tagged as such in
  §15.3 items 6 and 7.
- **`bench/preflight_graph.py::speaker_id_rules` is the check**, FAIL on ref2va and a
  note on the base modes, because that is where the guides put the rule. Shown
  red on the bank as it stood at `4bd7b429^` and green on the same entries now;
  the whole bank passes `build_prompt_bank.py --check` with it in.
- **Not decided, the owner's call.** 0.128.0 applied ref §5.4's reuse rule to
  base-mode prompts too and five base entries were not swept, so the bank does
  it in some base prompts and not others. The guide does not require it there.
  The note says which entries; sweeping them or leaving base alone is a
  decision, not a fix.
- **`docs/prompt_audit.md` is stale against the current bank text** and now says
  so in place. Its verdicts predate `705063a3` and `4bd7b429`; a green
  `check_prompt_docs_sync.py` does not speak to it, because that check tests
  that a verdict exists and not that it is current.

## 2026-09-19

- **`MiniMaxH3EncoderLoader` takes core's `device` input** (owner: whatever
  device the user puts the encoder on is respected). Same options, same
  `model_options` and optional as core's `CLIPLoader` has it
  (`h3_encoder_loader.DEVICES`); before, the node had no placement input and
  its rebuild factory would have dropped one. The node's description also
  told users to use "the AWQ loader instead" for a W4A16 file; that loader
  was deleted on 2026-09-13, and the sentence is gone.
- **Upstream survey: three pieces of prose had lost.** `docs/sol_upstream.md`
  described kitchen PR 171 (chunked `key_bias`) as an open draft with a rebase
  hazard for `blk_cnt`, core PR 16239 as an open PR that would rewrite core's
  Sol node, and kitchen PR 176 (W4A8 GEMV) as assessed and not carried. 171
  and 16239 were closed unmerged by their author on 2026-09-16; 176 merged
  into `v0.2.35`, which this install has carried since the 0.123.1 rebuild.
  Dated notes now stand beside each, and the new section "comfy-kitchen and
  core, 2026-09-19" holds the state.
- **`docs/open_experiments.md` #28 (the kitchen VAE kernels) said both PRs
  were open and the check waited on a tag.** Both halves merged (kitchen 167
  in `v0.2.34`, core 16187 on 2026-09-15) and have been in this install
  since its 2026-09-15 core pull; the item's 2026-09-19 status says
  so and notes that its before-and-after capture check has no record. No
  decision taken: reopening it is the owner's call.
- **`docs/research/sglang_comparison.md`'s Sol defaults table said our dense
  attention outside the Sol window is Sage.** It has been kitchen's dense
  attention since 2026-09-15 (`workflows/h3_config.py::DEFAULT_DENSE_CHAIN`);
  a dated note under the table says so. The same survey found sglang's new
  RTX 5090 recipe sets memory placement only, so the adopt-upstream rule has
  nothing new to act on (that page's "Sixth read").
- **`docs/research/2026-09-17_rotation_and_lowbit_attention_survey.md` D.9
  credited DFSAttn with "layer-wise sparsity profiling + bidirectional
  co-clustering".** That mechanism is SVOO's (arXiv 2603.18636, whose v1
  title D.9 also listed separately); DFSAttn is arXiv 2605.23445 (Hilbert
  reordering, hierarchical block scoring, mask caching). Found independently
  by two research subagents (`docs/research/2026-09-19_token_selection_scorers.md`
  section 1, `docs/research/2026-09-19_video_sparse_attention_2026.md` "ID
  check"); corrected in place by the survey's owner in `15db381`.

## 2026-09-18

- **No timestamps on shot headers (the owner).** "The only time you should use
  timestamps in prompts is if you break stuff up mid-shot." Removed from the
  whole prompt bank, the shipped graphs, the prompting guide's examples, the
  portable system prompt and the prompt skill; `bench/preflight_graph.py`
  fails a stamped header. It departs from the vendor guides' stated format, on
  the owner's judgement that fixed cut times are "probably a big cause of
  issues"; one same-seed render with and without them is the first evidence
  either way (`Video/timestamps_test/`).
- **`morton` stays off by default, and the panel did NOT decide it** *(corrected 2026-09-19)*. Five valid on-length
  pairs scored blind, one seed each: plain order better on two, the `3d` reorder on one, two ties; the sixth pair
  (the kitchen scene) differed in chain and `qk_balance` as well as token order and is struck
  (`bench/results/2026-09-18_sol_reorder_panel.md`). A panel that split is not a panel that agrees; the reorder
  stays off because nothing argues for turning it on, and the short-clip panel, still unscored, is now the
  evidence that matters, since the one visible win for the reorder was on a short clip.
- **The noodle bar is retired at full length (the owner).** Its prompt is
  written for 107 frames and was rendered at 345, out of distribution; post
  office (141) likewise. A bank prompt is rendered at the length its bank
  entry declares, and `bench/run_graph_arms.py` now refuses anything else
  without a flag. The 2026-09-17 entry below stands as what was seen and is
  withdrawn as evidence about normal use
  (`bench/results/2026-09-18_off_length_prompts.md`).

## 2026-09-17

- **The owner scored every noodle bar stack: only Sol's `3d` token reorder is
  clean.** Every plain-order arm morphs around the four second mark (both
  dense chains, Sol `rotate`, sage balanced and rotated, all three
  token-routing presets, and the default rendered on 2026-09-15); the reorder
  at tau 1.0 looks best and at tau 1.3 second
  (`bench/results/2026-09-17_sol_options_noodlebar_batch.md`).
  - **What it decides.** Making the reorder work with ComfyUI's memory
    compiler is the next piece of work; the node refuses `morton` while the
    compiler is active (`CHANGELOG.md` 0.122.1).
  - **What it does not decide.** Any default. The scene is a stress scene: a
    discriminating one, and one scene at one seed. A default moves on a panel
    of scenes ([`../eval_comparison.md`](../eval_comparison.md), "A stress
    scene is not a typical scene").
  - **Corrected the same evening.** Several 2026-09-17 stacks had the
    2026-09-15 default on top as their reference; the day's runner scripts had
    passed the prompt with a trailing newline, so those rows are different
    samples. Recorded in both batch records; token routing showed no visible
    benefit and stays off.

- **`SageChainAssert` taken out of every generated workflow and the
  generator** (owner, 2026-09-17; `cfeeaa1`). The node stays registered, so
  saved graphs still load.
  - **Why.** On the default chain it could only confirm that no sage kernel
    ran. Its generation-time flags refused an intended editor swap to our
    sage node on `h3_candidate_t2v_pdd8_baked`.
  - **Lost with it.** The call-time probe on the sage arms, the "nothing
    patched" guard on the baselines, and the "[h3] sol window" log line.
    `bench/check_attention_defaults.py` still grades the wiring.
  - **Corrected.** Three docs still showed our sage node as the default dense
    node after the 2026-09-15 flip, which that change's prose sweep missed:
    - `docs/h3_geometry_and_nodes.md`'s chain block;
    - `docs/wiki/stages.md`'s attention row;
    - `docs/SOLATTN.md`'s Ordering diagram, which also showed the assert.

## 2026-09-15

- **Two bench manifests and one probe graph retired** (the owner deferred the
  call to the sage-fork session, which chose it; `531fbab`). Both would re-run as
  something other than their records: `bench/ladder_arms.json` (rendered
  2026-09-03, verdict `data/results/2026-09-04_ladder_2026-09-03_verdict.json`)
  refuses on its sage-mode patch now that `h3_text_to_video` has no sage node,
  and `bench/sol_core_ab_arms.json` with `h3_probe_t2v_sol_core` (rendered
  2026-09-10, no verdict record) would set ours on kitchen against core's node
  on sage. The four manifests that pointed at the ladder for what each scene
  tests carry the descriptions inline. **Corrected:** `docs/wiki/next_steps.md`
  said the core-versus-ours A/B was "BUILT not rendered"; its 2026-09-10 records
  say it rendered.
- **Sage out of the default attention chain; Sol's `qk_balance` on** (owner,
  2026-09-15; `28d8ee5`). **Reverses the 2026-09-04 floor decision** ("sage
  always on is the floor", `docs/roadmap.md` forward plan 2026-09-04, quoted
  in `bench/sol_core_ab_arms.json` and `bench/sol_nosage_arms.json`, which stay
  as records of what those runs held). Every video graph that names no mode
  now carries core's `ModelAttentionBackend` at `h3_config.DENSE_BACKEND_NODE`
  under Sol, and `h3_config.SOL_RECOMMENDED_CUDA` carries `qk_balance=True`.
  Taken on capture grades and one render per arm, not a blind verdict: the
  community-chain arms were unscored when it landed
  (`bench/results/2026-09-15_block49_community_chain.md`,
  `data/results/2026-09-15_ck_int8_attention_block49.json`). What stays on
  sage or stock attention, and why, is
  `bench/check_attention_defaults.py::FLOOR_STEMS`. `h3_probe_t2v_balanced`
  and `h3_probe_t2v_ck_balanced` retired. Prose that said the fallback under
  Sol is sage on every graph (`docs/SOLATTN.md` knob table and two dense-block
  sections, `docs/wiki/next_steps.md` "Now") carries a dated note in place.
  The policy page's table rows for this landed early, in `76b2378`.
- **`docs/research/2026-09-14_block49_quant_error.md` moved to
  `docs/h3_block49_quant_error.md` and revised** (owner, 2026-09-15): it is
  a reference page now, not a dated research note. Its "So what" was
  rewritten with the blast radius (a design property of INT8 attention
  meeting this model, affecting every H3 user on any quantized-attention
  kernel, not a bug in any kernel and not confined to these forks) and a
  numbered status: diagnosed and closed; two levers built, measured, both
  off; open items are the blind visibility check, the factor in Sol's
  quantizer, and sending the finding upstream. Every link was rewritten;
  `bench/build_wiki_index.py` reaches the new path.
- **Kitchen build moved to `0.2.34+sol.2aff3c5`.** ComfyUI's pin went to
  `comfy-kitchen==0.2.34`; `h3-build` was rebased onto the tag by the recipe
  in `vendor/rebuild_kernel.sh`, old tip archived as `archive/h3-build-0.2.33`,
  six `blk_cnt` commits reapplied clean. Brings the H3 VAE kernels (#167/#175),
  persistent RoPE allocations (#162), `compress-mode=size` (#165). Kijai's
  branches carried nothing unmerged on the CUDA side; open PRs assessed and
  not carried: `docs/sol_upstream.md`, 2026-09-15. `docs/SOLATTN.md` "Install
  the CUDA kernel" used to say main had moved past 0.2.33 with no tag.
- **`docs/h3_block49_quant_error.md` gained sections 7
  (the sage-side `qk_balance` kernel fix) and "So what".** No default changed.

## 2026-09-14

- **The generator runs masked, as documented, rather than needing the card**
  (owner, choosing the code fix over a doc line saying it needs the GPU). Prose
  that lost: `docs/checks.md` ("While a render is on the card") said the
  generator takes ComfyUI's CPU path when no device is visible, and the
  generator's docstring said nothing touches the GPU; a masked build raised in
  `resolution.py`'s core import before the CPU switch ran, at `1345791` and
  after. Fixed in the generator (`_core_cpu_when_no_card`) and in
  `bench/check_generator_constants.py`; `docs/checks.md` is true again and the
  docstring now says a visible card gets a CUDA context (CHANGELOG 0.108.1).
- **The frozen-audio loop simplified before extending** (owner, after the
  footguns were listed: "simplifying is the only way to avoid that"; the
  worry was a new place starting where a window ends, not where the song
  changes section). `prompt_mode`, random window lengths, `frames:` lines, the
  implicit wildcard fallback and Fill Prompt Lists' `count` went; the list
  node's `seed` became `shuffle`; the song node gained `timeline` and
  `preview`, planned by `loop_plan.py` (CHANGELOG 0.108.0). Prose that lost,
  and what it claimed: the shot-workflow bullet below said the song node's
  prompt blocks and `frames:` lines do what those workflows did (the timeline
  and `--- label` blocks do now); `prompt_lists.py` said a placeholder with no
  list node reads `name.txt` or `name.json` from the wildcards folder;
  `docs/comfy_notes.md` and `bench/check_node_ids.py` said an input may only
  ever be appended (removal is the owner's call now that shipped graphs are
  API format); `docs/h3_geometry_and_nodes.md` gave `MiniMaxH3ImageToVideo`
  as the t2v/i2v node without saying the pack wires `MiniMaxH3Conditioning`.
- **Shipped workflows are API format only** (owner: "If I dont need a layout,
  node titles, bypassed nodes (we dont even use those), or notes in workflows,
  why dont we just simplify all of them to API json workflows?"). The UI twins,
  `build_ui`, `UIGraph`, `validate_ui`, `cross_check`, the in-graph notes and
  `bench/check_workflow_schema.py` went; `validate_api` gained the value-type
  and DynamicCombo-member checks `validate_ui` had, and member order. The editor
  loads an API file by input name (the frontend's `loadApiJson`, read from the
  source of comfyui_frontend_package 1.52.7, not exercised). Prose that lost,
  and what it claimed: `UIGraph`'s docstring said the generator made no
  widget-to-input conversions (its Resolution and PDD step links were exactly
  that); `bench/smoke_h3.py` said Sol-Attn ships off; `bench/check_graph_discovery.py`
  said it had no exemptions (it has one); `docs/custom_node_gaps.md` item 3
  described a UI/API node mismatch that no longer exists. The deleted notes
  also carried stale claims (the Sol window at 0.2/0.9, a keyframe canvas node
  the graphs do not wire, sage-only twins of the Sol probes); they went with
  the notes. Facts that lived only in the notes moved to their docs, and the
  move corrected doc prose that had lost: `docs/SOLATTN.md` printed `[sol_attn]`
  log lines (the prefix is `[h3-sol]`); `docs/h3_geometry_and_nodes.md` told
  readers to wire `MiniMaxH3KeyframeCanvas` on every i2v and fl2v graph (the
  conditioner owns the canvas), showed a node order through the deleted
  `SolAttnPatch`, and said `MiniMaxH3SigmaShift` sits in every shipped graph;
  `docs/h3_references.md` said the concise swap twin stays shipped (retired
  2026-08-22); `docs/h3_ref2v_distillation.md` said no v4 ref2va graph existed;
  `docs/h3_image_editing.md` pointed at a note in a graph that now lives only
  under `archive/`. Plan: `internal/2026-09-14_api_only_workflows_plan.md`.
- **Two duplicate Sol-on probes retired** (owner: "retire the
  duplicative/redundant workflows"). `h3_probe_sol_on` and `h3_probe_sol_on_i2v`
  matched `h3_text_to_video` and `h3_first_frame_to_video` in everything but
  the output name once Sol went on by default. What prose claimed: the
  generator read `h3_probe_sol_on` "against h3_text_to_video.json, which is
  now sage-only", and `check_bench_matches_shipped.py` said the plain t2v graph
  omits Sol. This session's own note first counted `h3_probe_sol_on_all_refs`
  among the duplicates; its reference video and audio disprove that, and it
  stays.
- **Prompt lists serve every loop, not the song node** (owner: "I should be
  able to use this for any type of looping workflow we build here"; agreed
  the approach the same day). A node that loops inside itself fills through
  `prompt_lists.fill_windows`, takes `lists` and fingerprints the wildcard
  files it may read, and `bench/check_prompt_lists.py` fails one that does
  not. A graph that loops by chaining nodes, or one clip per queue, uses
  `MiniMaxH3FillPromptLists` by index; value N of a list depends only on the
  list and N, so both routes agree. The fallback stays: a placeholder with no
  list node reads a wildcard file of its name, shuffled at seed 0 (the
  option the session recommended over always raising, or a seed widget for
  it, and the owner agreed).
- **No live preview node in generated graphs** (owner: it wastes GPU).
  `ModelPreviewOverrideKJ` with taeh3 left every UI graph and the generator's
  `preview` path. What the prose used to claim: the generator's node note and
  graph comment called it "arguably the largest optimization here" and worth
  more than any kernel knob; `docs/h3_geometry_and_nodes.md` said the same.
  The owner also answered the API-only plan's open questions
  (`internal/2026-09-14_api_only_workflows_plan.md`, top): loop stage 2 goes
  before the conversion.
- **The frozen-audio loop plan, confirmed** (owner, in answers the same day).
  Encoding comes before sampling: the track and each distinct prompt once.
  Reference stills go with every window's prompt, with a workflow for it.
  The song node keeps its window files in a per-graph working folder
  (`<prefix>_windows/`, kept by default) and gains resume from the first
  changed window, with its seed held fixed between queues so a re-queue can
  reuse windows. Prompt lists fill `__name__` placeholders -- not `{name}`,
  which the frontend's dynamic prompts rewrite, and not `[name]` or `<name>`,
  which H3 prompts already use -- one list per node with its own seed held
  fixed, in order or reshuffled on each pass with no repeat before the list
  is used up, advancing only when a window uses the placeholder. Wildcard
  files live in `wildcards/` under ComfyUI's input directory (the
  `--input-directory` override included) and are selectable on the node.
  `docs/wiki/next_steps.md` carries the stages.
- **The shot-per-window workflows retired, unrendered** (owner: never used).
  `workflows/h3_text_to_video_audio_freeze_shots.json` and
  `..._shots_repeat.json` with their API twins, the generator's
  `freeze_shots` knob, and `MiniMaxH3JoinWindows` went. The song node's
  prompt blocks and `frames:` lines do what they did. The two-window seam
  graph (`workflows/h3_text_to_video_audio_freeze_2windows_api.json`) stays.
- **A loop feeds the encoder no per-window image** (owner, after discussion).
  A frame from the previous window would carry drift forward and force an
  encode between windows; the anchor for identity is a fixed reference still.
  `docs/h3_audio_freeze.md` step 5 said a first-frame keyframe was "needed
  before the loop, because the loop anchors each window on a frame"; the loop
  shipped anchored on the previous window's frozen latent tail. Corrected in
  place.
- **References work on the fl2va checkpoint** (owner: done often). Three
  places said or implied otherwise and are corrected:
  `MiniMaxH3ReferenceConditioning`'s description ("Use an H3 reference
  checkpoint"), `docs/h3_geometry_and_nodes.md`'s node table (fl2va "for
  t2v/i2v, ref2va for reference-to-video"), and `docs/h3_audio_freeze.md`
  step 6 ("the fix is the ref2va checkpoint with a per-window reference
  image").
- **The song node's seed has a control widget.** The generator's comment said
  "no control widget: the node's seed input declares none"; the frontend draws
  one by name for any INT called `seed` (`useIntWidget.ts`), so the shipped
  song UI graphs loaded shifted. The node declares it now
  (`audio_freeze_song.py`, the `seed` input).
- **The song node's docstring and graph note said its first run had not
  happened.** `data/results/2026-09-12_audio_freeze_song_smoke.jsonl` records
  one. The docstring points there now.

## 2026-09-13

- **The AWQ lane's code is deleted** (owner). The lane closed on 2026-08-27
  (`docs/roadmap.md` "Closed lanes"); today `h3_awq_encoder.py` and its
  `MiniMaxH3AWQEncoderLoader`, the `config/` W4 snapshot directories,
  `docs/h3_awq_encoder.md` and the bench tools that served them
  (`build_h3_awq_standalone.py`, `check_h3_awq_encoder.py`,
  `convert_h3_awq_candidate.py`, `capture_h3_encoder_states.py`,
  `measure_qwen_view_under_snapshot.py`, `measure_still_policy_token_cost.py`,
  `build_native_h3_calibration_batch.py`) went with it, as did
  `h3_config.ENCODER_V1` and `ENCODER_V2`. The records under `bench/results/`
  and `docs/research/` stay as history. `h3_config.MODELS["clip"]` is
  `ENCODER_INT8`, and every generated graph loads it through
  `MiniMaxH3EncoderLoader` (`workflows/build_workflows.py`, the loader
  comment). What the prose used to claim: `docs/wiki/stages.md` named the
  AWQ loader as the alternate encoder load; `docs/custom_node_gaps.md` §5.1
  said every graph wires core's `CLIPLoader` and the adapter was live code
  read by preflight and the config; `docs/h3_geometry_and_nodes.md` called it
  "the loader used by every generated graph".
- **The `encoder` option is gone from `MiniMaxH3ReferenceConditioning`**
  (owner). `image_policy` and `video_policy` each offer `comfy` (default) and
  `release` (`reference_geometry.IMAGE_POLICIES`, the conditioner's
  `define_schema`). On the shipped core-loaded encoder `encoder` always
  resolved to `comfy`, since the CLIP carried no contract, and for stills
  `release` and `comfy` produce the same geometry at every legal short edge on
  that encoder (`data/results/2026-08-29_qwen_view_under_snapshot.json`).
  `video_policy`'s default moved from `encoder` (which ran as `comfy`) to
  `comfy`; every graph was rebuilt. What the prose used to claim:
  `docs/comfyui_vendor_gaps.md` listed `video_policy=encoder` as the "shipped
  hybrid encoder policy" and `docs/h3_references.md` said "the encoder-aware
  hybrid is now the generated default"; both had been marked dormant on
  2026-08-29 and are now marked removed.
- **`MiniMaxH3AppendRefImage` defaults are vendor parity** (owner):
  `size_policy=max`, `dit_short_edge=2048`, `allow_upscale=True`,
  `qwen_view=shared`, which is what sglang, diffusers and DiffSynth do
  (`docs/research/sglang_h3_pipeline.md` "Reference stills"): one prepared
  still feeds both the video VAE and Qwen3-VL. Read the values from the node's
  `define_schema`. Before today `allow_upscale` was off (since 2026-08-28) and
  `qwen_view` was `separate` at 512 (since 2026-08-27, on one observation,
  CHANGELOG 0.82.0). `h3_rules.REF_QWEN_SHORT_EDGE` is now only the value
  pre-filled when a user picks `separate`; `REF_VIDEO_BUDGET` still sets
  `ref_upscale=False` on the video-bearing reference arms, as an arm setting
  for memory. What the prose used to claim: `docs/evidence.md` "Reference
  sizing" called the 512 view the shipped default; `docs/custom_node_gaps.md`
  §4.1 said "three independent implementations agree, and we differ";
  `docs/h3_conditioning_end_to_end.md` said most append nodes set
  `qwen_short_edge=512`.
- **The reference-view ablation is rebuilt** (owner). The three-arm Gate 6
  family (`h3_probe_refview_{a_source,b_qwen2048,c_parity}`,
  `bench/gate6_refview_arms.json`) was priced on 2026-08-25, never rendered,
  and is deleted. The new one is five ref2va scene graphs
  (`h3_config.REFVIEW2_SCENES`, `workflows/h3_probe_refview2_*.json`) built
  at the node defaults, with six arms per scene as widget patches in
  `bench/refview2_arms.json`. Unrendered; nothing is claimed.
- **`docs/h3_input_impacts.md` pointed at `preflight.py:28`** for the int32
  crossing; the line moved, and `preflight.py::_INT32_FUSED` is the pointer now.
  The same section gains the second ceiling, the CUDA v-side `uint32` wrap in
  the sage fork's `csrc/fused/fused.cu`, from the fork's own CHANGELOG.

## 2026-09-14

- **`dense_blocks="0-1"` decided against.** The roadmap's open call since
  2026-08-16. Two instruments that share no code agree the first blocks carry
  the least error on both terms; `dense_blocks` stays `""`. Reasoning in
  `docs/roadmap.md` under the original item and `docs/SOLATTN.md`, "The
  defaults, re-read against the sage-side error records, 2026-09-14".
- **`MiniMaxH3ChannelBalance` added, off by default, an experiment.** Folds a
  per-channel q/k rebalancing into the norm weights of the blocks whose
  K-norm is lopsided (45, 48, 49 on the shipped checkpoint). No workflow
  wires it; no default changed. Why the blocks are lopsided and what the fold
  buys: `docs/h3_block49_quant_error.md`.
- **Corrected: "K offset predicts `smooth_k`'s benefit."** The sage fork's
  real-activation spike printed that inference; graded across ten cells the
  benefit tracks block depth and error magnitude, not the offset, and the
  line is removed there. `smooth_k` stays off.

## 2026-09-12

- **PDD reopened for the audio-freeze lane only** (owner: "worth testing if
  PDD works with this since it's faster iteration"). The lane was parked on
  2026-09-05 for quality work; this is iteration speed, and the documented
  PDD weakness is its audio, which a frozen track takes out of PDD's hands.
  `workflows/h3_candidate_t2v_pdd8_baked_audio_freeze.json` is the graph;
  the parked quality work stays parked.
- **Audio-freeze lane opened** (owner): a known track frozen into the target
  audio rows with a per-stream mask, on the fl2va base first, the LTX pack's
  loop geometry on H3's grid after; reference audio parked until the loop
  runs because it regenerates the track. `docs/h3_audio_freeze.md` owns it.
  Before this, the mask path was known here only as something no shipped
  graph used (2026-08-30) and the looping packs were explicitly unread
  (`custom_node_gaps.md` section 8).

## 2026-09-11

- **The frontier table in `next_steps.md` counted a dense last step.** It said
  the leader runs five sage and eleven Sol of sixteen; since `07b903c` it runs
  four and twelve (reasoned, not rendered). A note above the table says so.
- **`docs/comfy_notes.md` said Sage runs `fp16 (most accurate)`, not `auto`.**
  The config has shipped `auto` since `497b421` (2026-08-18), which scoped the
  fp16 verdict to renders without Sol; the paragraph now says so.
- **The `shown red` column is gone from `docs/checks.md`** (owner). It
  recorded which checks had been shown failing under the retired
  red-before-green rule; git history has the cells.
- **`docs/prompting.md` 5.10 rewritten around the owner's point** (owner): too
  many words in a shot make the speaker rush. It carried a second word-budget
  formula beside §3.4's and blamed the rushed delivery on the Audio VAE; it now
  budgets from speaking time, points at §3.4, and claims no mechanism.
- **`CLAUDE.md` cut to what every session acts on** (owner). Seven "Settled
  about H3" facts, the numeric-input and capture-broadly rules, and the
  `coderef/` search advice moved to `docs/evidence.md`, `VISION.md` and
  [`references.md`](references.md); nothing was withdrawn. "A rendered clip
  cannot A/B a numerical change" stayed, because code and docs cite it as
  `CLAUDE.md`'s.
- **`bench/red/` removed** (owner). The red harnesses and their shared spine
  served only the retired red-before-green rule, and nothing imported or ran
  them; git history has them.
- **The `h3-experiment` skill removed** (owner). Its two steps nothing else
  held are in `docs/comfy_notes.md` "Adding a probe or an arm".
- **Documented commands run the ComfyUI venv's python** (owner). They said
  `uv run --active --no-sync python`; the venv's own interpreter is exactly
  what the server runs and involves no uv project step.
- **Run bench scripts with `python`, not `uv run`** (owner). `docs/eval_comparison.md`
  and the `h3-prompt` skill said `uv run python bench/...`; plain `uv run` in
  this repo builds a repo-local venv and, with `VIRTUAL_ENV` set, can recreate
  the ComfyUI one, as `docs/comfy_notes.md` records.
- **`CLAUDE.md` routes to the wiki instead of carrying its tables** (owner;
  CHANGELOG 0.99.72). The "What is where" tables moved into
  [`index.md`](index.md), which is now written by hand.
  `bench/build_wiki_index.py` used to generate that page from `CLAUDE.md`; it
  now only reports documents no link reaches.
- **History notes leave `CLAUDE.md`** (owner; CHANGELOG 0.99.72). Its rule
  "when you find prose that lost, correct it and say what it used to claim"
  now logs the old claim on this page.
- **"A perceptual claim needs a distribution of seeds judged blind, never a
  pair" is withdrawn** (owner; CHANGELOG 0.99.72) from `VISION.md`,
  `CLAUDE.md`, the `h3-experiment` skill, `docs/open_experiments.md` and
  [`docs/prompting.md`](../prompting.md), under the tinkering-repo rule.
  `docs/eval_comparison.md` still describes the blind process for when one
  is wanted.
- **Where the Sol-Attn kernel comes from.** `CLAUDE.md` said it was
  "installed from comfy-kitchen main". It is built from the owner's fork by
  `vendor/rebuild_kernel.sh`: the tag ComfyUI pins plus the `blk_cnt`
  commits, enforced since `d378479`.
- **The tinkering-repo rule** (owner). `CLAUDE.md` opens with it: not
  research-grade, rigour proportional to the claim, and a default that
  sglang and ComfyUI's own node or comfy-kitchen agree on is adopted without
  waiting on our evals.
- **No check has to be shown red before it is trusted** (owner, `48e1219`).
  `docs/checks.md` "The standard" says what it used to require.
- **Closed lanes moved into the repo** (`48e1219`). They were recorded only
  in agent memory; `docs/roadmap.md` "Closed lanes" is their home.
- **The probe canvas** (`48e1219`). The `h3-experiment` skill said to bench
  canvases cheaper than 16:9 by default; probes run at 1152x768 or 1344x768
  with 345 frames, the owner's rule of 2026-08-30.
- **Sol runs through the last step** (owner, `07b903c`). `end_percent` is 1.0
  on every graph, adopting sglang's and core's default. It used to stop short
  so the last step ran dense; the retired values are in the comments beside
  `workflows/h3_config.py::SOL_END_PERCENT_BY_STEPS` and `SOL_PDD_OVERRIDES`.
- **A region for her hair, built** (the owner, 2026-10-12, about 18:52Z by
  the clock, in the lead's terminal, after seeing her hair disappear and
  fill in at frames 693 and 703 of the long kitchen shot in form B: "the
  issue that caused the error - yes, we should build that"). Form B keeps
  her clothing out of the region and gives the margin's cells back at the
  shoulder line, so the still's long hair is cut where the region ends and
  the edge moves with the clothes mask. The change queued: a region that
  keeps out only the strap's cells and leaves a still-edged band for the
  hair below the jaw (`next_steps.md`, the 2026-10-12 gaps).
- **A text says exactly what the reference video is, then references it**
  (the owner, 2026-10-12, between about 18:50Z and 18:56Z by the clock,
  in the lead's terminal, after five renders of the second subject with
  the lid overlay never widened her eyes where the original's open): "say
  the shapes are what they are. tell the prompt exactly what the reference
  video is"; "h3 needs to know what the ref video is and what its looking
  at. then after that we can reference it properly." So the line that
  describes `<Video 1>` says what is in it as the model sees it (a grey
  body model of a person on a black ground, posed as the person moves, its
  eyes drawn as white shapes edged in black that open and close as the
  person's do), and the subject's clause then references it (the movements
  of the head and the facial expressions of the figure); nothing says what
  her own face does. The first load under it: `second_hf_1039_d`, one
  change from `second_hf_1039_c`, its text. The texts before it called the
  marks "shapes drawn on its face that are part of its facial expression"
  and never said they were eyes (Open List row 14).
