# Handover: read this before anything else

Rewritten 2026-10-21 at the close by mrtak (the Claude lead after mrturtle), with mract on the card and Gemini (mrgemx)
building. Where this page and a record disagree, the record is right: `git log`, [`decisions.md`](decisions.md) 2026-10-21,
the owner's rules at the top of [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), and `data/fun/` (laid.jsonl,
owner_answers.jsonl, verdicts, shots.json, cast). Running notes: `internal/claude/2026-10-21_mrtak/notes.md`.

## State when closed

**What the evening taught us (start here):**
- **A figure against the source misleads for a replaced part** (the owner). Dark hair "close to the source" read as fine on
  4912e. The fix is to correct that reading, not add a tool: brief `internal/claude/2026-10-21_mrtak/replaced_part_measure_brief.md`
  (reviewed by Gemini, G28). Until it lands, no hair figure against the source says anything about the second subject.
- **4912: her hair was dark on every frame** (the owner), so the crown frames at the end do not explain it. Her crown on
  4944-4946 is in no mask (the builder's read, confirmed by Gemini's G29; G23 was wrong there). The owner: test before
  building anything for it.
- **Negatives never help** (the owner): a "not used" sentence puts those words in front of the model. The release's guide
  writes none in its examples. "Only" is fine (the owner).
- **The masked render path rides the old song node.** `audio_freeze_song.py` is the live render node of every masked load;
  the owner uses no audio-freeze graph. Tonight's window build threaded the padding through six places. **Read Gemini's
  report first: [How the masked render path works, and what to simplify](../../internal/gemini/2026-10-21_render_path_report.md)**
  (`internal/` is local, not in git; C61 corrects one claim: chained shots are live, 2050-2622 and 604-1009 run several
  windows).
- **3719: the owner's box on 3729 sits on a man behind the singer**, most likely a slip of the box (the owner asked; 20 of
  21 lead previews track her, reader A1), so not a scale fault. A new box on 3719-3758 replaces it.
- **Gemini's specifics need checking** before they are relayed (about a third off in detail today, never the core).

**Landed this evening** (`git log e40e817a..main`): the lowered-eye span word; the head-tube re-seed only on her own face
points; the text fix ("face and blonde hair", no "not used" clause) with the plan words `kept` = "blonde" and `looks` =
"blonde hair, blue eyes, and fair skin" (backup `data/fun/plan.json.before_t71_*`); `preview --for`; no negatives in the
Masked Prompt node and `bench/examples/` removed; the window build (both clocks, real context frames, zoom across a cut).
ComfyUI restarted once for it (T76; the pid and start time are in the card's S-entry).

**Tonight's tests** (S52, S53): **0415 rendered both ways**, second_hfp_0415i (A: the old text with both "not used"
clauses, `--say-not-used`) and second_hfp_0415j (B: the default), seed 730451892, the lowered-eye word on 415-437; both
paged at the top of the review page, not laid. **4912 is held**: second_hfp_4912b stopped at `first`, its reference failed
on 4946 (the dancer's face in front of her) and the gate blocked on her part empty where she is hidden (4921-4926,
4941-4946). The review server is stopped: start it to watch (`data/fun/poc/serve.py`).

**Held off main, on purpose** (each branch also on the `nas` remote):
- `claude-seed-every-box`: every owner box seeds its tracker. Node code (988bc644, 9ce71f58, f87a7a44); needs a blind read
  and lands only together with a ComfyUI restart. READY `internal/claude/2026-10-21_ready_seed-every-box.md`.
- `claude-change-check` (df2bf603): parked by the owner; replaced by the measurement brief above.

## What not to assume

- **`--say-not-used` is test-only.** It exists for tonight's A renders; retire it next.
- **The cut on the share (1714Z) is behind laid.jsonl.** No `--clip` rebuild until a batch of keeps.
- **Held copies remain for chained shots** (over 345 frames). One-window loads are padded with real frames (Gemini's C58).
- **The capture's `RETENTION_KEPT`** reads a kept retention line as no voice sentence; a hand-written one that names a voice
  is not flagged (the text-fix review).
- **A preview made with `--for b` on a cut that holds b's own piece is taken silently** (the `--for` review). Use the raw
  source for a lone replacement.
- **Nothing the 1093 preview showed is fixed tonight**: the tracker refused her box on the 1714Z cut.
- **A load whose reference failed still reaches `first`** (4912b; the gate blocked it). With no face points the head tube
  refuses every re-seed, "no face points were given": that is the failed reference, not missing wiring (checked: both
  callers pass `points_path`). The chain could stop at the failed reference instead.

## Tomorrow, in order

1. The owner watches the 0415 A/B clips. B is no worse: retire `--say-not-used`. Then 4912b: the owner's word on 4946
   (the dancer's face) and on her hidden frames (left as the original), then its A/B. If she comes out dark again, compare
   its inputs with 5768f's, before any build. Tools to fix: back-to-back A/B jobs took one reference folder (S52); the
   review item has no fields for a load's text form, seed and span words (S53).
2. One sweep of negatives everywhere (the owner): `workflows/build_workflows.py` `h3_ref_video_motion` text and its bank copy,
   `docs/h3_references.md` "say the scene is not reused", bank prompts "not present in the target video"; put "only" back in
   `masked_prompt_text.VIDEO_RETENTION`. The silent-voice and camera lines stay until a render shows something.
3. The owner redraws the lead's box on 3719-3758 (a review-page item).
4. The measurement fix (the brief), then the render-path simplification (Gemini's report): one home for window, padding
   and trim; trim before the region is saved; plain node inputs over the hidden graph read; the window computed once;
   retire the audio-freeze graphs (the owner uses none).
5. The seeding node code (blind read, land with a restart).
6. Then renders on fresh prep: the mom's remakes (4479, 5114, 5011, 0278 raw; 5533, 4647 on the 1714Z cut), the dad's second
   passes (4602, 4726, 5461, 5675), the lead's gap-page shots (owner_answers.jsonl:1402-1465).
7. Queued: W7 (staleness from a chain record), `laid.py set-owner --recite`, a leaner `<Subject 2>` text, whole-video prep,
   1039's head pose, the bigger zoom (parked).

## Where things are

- Mailboxes: `internal/claude/2026-10-21_mrtent_to_mrgordon.md` (T-entries, to T76+), `..._mrgordon_to_mrtent.md` (S-entries);
  `internal/gemini/2026-10-21_*` (C-entries to C61, G-entries to G31).
- READYs and blind reads: `internal/claude/2026-10-21_ready_*.md`, `..._review_*.md`, `internal/gemini/2026-10-21_review_*.md`.
- The reader's R1: `internal/gemini/2026-10-21_reader_r1_previews.md`. Preview sheets: `Video/fun/looks/2026-10-21_r1_lead_previews/`.
- Worktrees left (S54 removed the 16 landed or dead ones, branches kept): claude-seed-every-box (the next day's merge),
  claude-change-check (parked), mrgem-cut-map (Gemini's; mostly landed, the rest reverted on purpose 10-20: ask it).
- ComfyUI and the review server are both stopped (S54; 23:00Z).
- A worktree check that imports ComfyUI needs core's `nodes` first and `--cpu` with argument parsing on (the builders' and
  the lead's runners in their scratchpads show how).
