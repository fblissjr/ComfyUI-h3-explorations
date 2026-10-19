# Prompt audit: what follows the guides, and what to do about it

One hand-written verdict per scene that [`prompt_catalogue.md`](prompt_catalogue.md) lists, keyed to its scene names:
**keep**, **revise**, **rewrite** or **discard**. `bench/check_prompt_docs_sync.py` fails on a catalogue scene with no
verdict here; it does not check that a verdict is current.

- **Every verdict is a source read.** None rests on a render unless its row says so.
- **Which authority a defect breaks decides what it means** ([`prompting.md`](prompting.md) §14.1 and "Four layers"):
  breaking a vendor guide (`vendor_guides/`) is off-distribution; breaking an OWNER or HOUSE rule may mean the rule is
  wrong; an OPEN item is not a defect. Two rules were once invented here by reading guide examples as statements, so
  where the guide is ambiguous this page says ambiguous.
- **The verdicts predate two bank-wide edits and have not been re-derived against the current text:** `705063a3`
  removed every shot-header timestamp, and `4bd7b429` (0.128.0) made one-clause fixes across `prompt_bank/` (unplaced
  speakers, agentless actions, contradicted counts, voice descriptors that contradicted the character, silence markers
  on characters who speak later, missing speaker ids). The scenes it touched are its `git show --stat`. Re-reading the
  table is owed.
- **Older findings cite `internal/PROMPTING.md`** (its sections 7F, 7H and the like), deleted on 2026-09-01 ([`prompting.md`](prompting.md)
  §14.5). The STATED RULE / NOT A RULE notes above `LONG_T2V_PROMPT` in the generator are a second copy of
  `prompting.md` sections 4, 5.2 and 8, which own them.

## Verdicts

### The core scenes

Ordered by how many graphs carry the scene.

| scene | mode | verdict | why |
|---|---|---|---|
| `LONG_T2V_PROMPT` (market) | t2v | ~~rewrite~~ **rewritten `d5be353`** | carried four official-guide defects (below) and is the t2v default; fixed in the generator and every carrying graph rebuilt. **Clips from before 2026-08-28 are not matched-seed comparable with clips after it** |
| `ref2va_role_character_environment` | ref2va | **revise** | far under ref §5.2's word budget; one shot |
| `ref2va_role_character_garment_environment` | ref2va | revise | same budget gap |
| `ref2va_image_video_audio_music` | ref2va | keep | a cache probe; scene content is not what it measures |
| `fl2va_interior_converge` | fl2va | keep | conforms |
| `I2V_PROMPT` | i2va | keep | conforms; carries base-en's I2VA Part One template and ships in `h3_first_frame_to_video*` |
| `ref2va_video_character_swap` | ref2va | revise | budget |
| `BENCH_T2V_PROMPT` (bank `t2va_frontier_standoff`) | t2v | keep | the t2v bench pair's scene since 2026-09-03, chosen off the market prompt for its failure surfaces (a figure at distance, a painted sign, dialogue, a silent bystander); grades clean in the bank |
| `DIALOGUE_REF2V_PROMPT` (stairwell) | ref2va | ~~rewrite~~ **done `d5be353`, `f5b3651`** | its headers (`[Shot 1, 00:00.000-00:06.000]`) contradicted both guides and silently emptied preflight's shot list; headers fixed, soundscape closed |
| `ref2va_video_sound_structure` | ref2va | keep | a policy probe |
| `DIALOGUE_T2V_PROMPT` (stairwell t2v) | t2v | ~~revise~~ **done `d5be353`, `f5b3651`** | out-of-table motion phrase replaced; soundscape closed |
| `R2V_PROMPT` (bank `ref2va_image_ref_default`) | ref2va | ~~rewrite~~ **rewritten `d5be353`** | described ten seconds on a 15.083 s graph, with a camera clause outside §4.3 in every dimension; now a fourteen-second single take |
| `MARKET_REF2V_PROMPT` | ref2va | **keep** | three shots, two speakers, inside ref §5.2's word band |
| `l2va_interior_converge` | l2va | **keep** | Part One resolves fully; conforms. Rendered once, judged bad, undiagnosed (below) |

### Experiment pairs

| scene | mode | verdict | why |
|---|---|---|---|
| `T2V_AISLE_SHORT` / `T2V_AISLE_LONG` | t2va | **keep** | a matched description-length pair: identical dialogue, cuts, camera moves and subjects, elaborated only. Do not edit one alone |
| `T2V_SORTLINE_SHORT` / `T2V_SORTLINE_LONG` | t2va | **keep** | the second length pair, on an independent scene, for the same reason |
| `T2V_RAIL_LONG` / `T2V_CHURN_LONG` | t2va | **keep** | a pre-registered predictability-versus-delta pair, both about 500 words, one shot, no dialogue: delta predicts rail is worst, predictability predicts rail is clean |
| `ref2va_scene_kitchen` | ref2va | **keep** | four shots, inside the word band, the first shipped prompt to carry the lyrics, caption and cutoff markers; [`scene_arm_renders.md`](scene_arm_renders.md) is its viewing guide |
| `ref2va_scene_subway` | ref2va | **keep** | as kitchen; not an arm of the same experiment (different scene, reference and speaker count) |

### The composed ref2va scenes

All pass `check_prompt_guide_conformance.py` and carry no preflight FAIL. **The budget `revise` rows are one defect**
(see "Open findings"), listed per scene only so every catalogue name resolves.

| scene | mode | verdict | why |
|---|---|---|---|
| `ref2va_image_audio_voice` | ref2va | revise | budget only: `detailed_description` in the low forties against 350-500 |
| `ref2va_image_audio_music` | ref2va | revise | budget only |
| `ref2va_video_continuation` | ref2va | revise | budget only |
| `ref2va_video_motion_transfer` | ref2va | revise | budget only |
| `ref2va_video_structure_only` | ref2va | revise | budget only |
| `ref2va_image_video_structure` | ref2va | revise | budget only |
| `ref2va_video_garment_edit` | ref2va | **keep** | inside the word band |
| `ref2va_video_image_edit` | ref2va | **keep** | inside the word band, and the densest label set that ships: five items, each on its own line |

### The reference-view ablation (2026-09-13)

The scenes of `bench/refview2_arms.json` (`h3_config.REFVIEW2_SCENES`), each graded clean at 345 frames
([`prompt_bank.md`](prompt_bank.md) has the grade command). Verdicts on the text; the ablation's verdicts will be the
owner's, blind over seeds, in `bench/results/`.

| scene | mode | verdict | why |
|---|---|---|---|
| `ref2va_diner_breakup_refs` | ref2va | **keep** | both speakers and the room from three pictures; inside the band |
| `ref2va_night_porter_refs` | ref2va | **keep** | one picture, one sung line, two shots; inside the band; the non-human-face scene |
| `ref2va_stairwell_dialogue_backstage` | ref2va | **keep** | two speakers from one picture (a backstage pair), deliberately; inside the band |
| `ref2va_stairwell_dialogue_circus` | ref2va | **keep** | the same dialogue as stylised 3D animation from two animated-character stills; inside the band |
| `ref2va_studio_dancer_close_refs` | ref2va | **keep** | one picture, no vocal sound, two shots cut on the beat; inside the band |

### The audio-freeze lane

| scene | mode | verdict | why |
|---|---|---|---|
| `t2va_studio_dancer_close` | t2va | **keep** | the owner rated both single-window renders (`h3_t2v_audio_freeze_dancer_close_frozen_s2_00001`, `..._loose_s2_00001`, 2026-09-12) very good; the loose one read slightly better, which is the freeze lane's data point (`docs/h3/audio_freeze.md`) |
| `t2va_song_flicker_lists` | t2va | **revise** | `non_diegetic_music` carries a mood word, "melancholic" (base §4.7; `prompting.md` §8). The rest conforms: no header times, the singer identified at first appearance with `(S1)`, no lyrics by design, camera phrases inside the closed sets, and every text its two lists expand to grades clean. Editing it changes a shipped graph, so it is the owner's call |

### The masked lane: the Masked Prompt node's texts

Each is what `MiniMaxH3MaskedPrompt` writes for one combination of its choices (`masked_prompt_text.py`;
`bench/check_masked_prompt.py` holds copy and node together). None names a setting, shot, cut or camera, on purpose:
the source video's own frames hold those outside the mask, and one text serves every window of any clip. Whether that
costs anything is open (`bench/masked_v2v_arms.json`). **The node's sentences were cut on 2026-10-10** (every sentence
asserting what it cannot know of a clip, and every one speaking of an edit; `docs/state/decisions.md`, that date;
`bench/results/2026-10-10_prompt_text_hardcoded_in_code.md`). The upper-body entry changed only in its sound line;
the others now run under the reference guide's word budget, recorded on each in the bank, and none of those has
rendered as it now stands.

| scene | mode | verdict | why |
|---|---|---|---|
| `ref2va_masked_person_swap` | ref2va | **keep** | the shipped prompt of `h3_video_to_video_masked_song_pdd8`, the node at its defaults. Rendered once in its pre-cut form, one window and one seed on the band clip, not judged by the owner (`bench/results/2026-10-04_masked_v2v_first_run.md`) |
| `ref2va_masked_person_motion` | ref2va | **keep** | the shipped prompt of `h3_video_to_video_masked_song_ref2va_motion`, with the Masked Source's motion reference on: `<Video 1>` named as the source of the subject's motion and timing and nothing else (an `attribute_transfer` line), emitted by the song node at run time (`reference_order.motion_records` tells the static checks). Measured on the band clip's turn (`bench/results/2026-10-05_masked_v2v_motion_arms.md`); rendered 2026-10-06 beside the man text on the same graph, window and seed, it carries the turn and starts it later (`bench/results/2026-10-06_masked_v2v_person_text.md`), not watched |
| `ref2va_masked_person_head` | ref2va | **keep** (text only) | the node when the Masked Source replaces head and hair. Not rendered |
| `ref2va_masked_person_upper_motion` | ref2va | **keep** | the shipped prompt of `h3_video_to_video_masked_upper_song_ref2va_motion` (2026-10-07): head and upper body from the still, motion reference on. Its node wording rendered beside the typed-text arm the owner called solid, and once on another clip; the owner watched both on 2026-10-07 and could not tell them apart (`bench/results/2026-10-06_masked_v2v_body_window_arms.md`, the note of 2026-10-07) |

## Open findings

**The ref2va budget gap** (ref §5.2): the range is 350-500 words for generation tasks, "normally", not a hard bound;
video-editing descriptions are exempt; dialogue-dense content may prioritise the spoken timeline; base_en states no
budget. By the declared task prefix, the pure `reference generation` prompts (42-198 words when classified) are under;
the `video editing` ones are exempt; `video continuation` is not named by the rule. Two editing prompts also declare
`reference generation`, and the guide does not resolve that combination, though `[video editing + reference
generation + audio reuse]` is ref §3's own illustration of the prefix (ref_en:127). Treat it as unresolved. Safe either
way: never pad toward the range, let information load set the length, and note ref §5.2's "A single shot does not
automatically justify a shorter description", since every under-budget prompt is one shot. `_ref_prompt` takes
`scene=`, so multi-shot ref2va text (`REF_SCENE_SHOTS`) is reachable from the generator. `prompting.md` §13 has the
command that lists the current under-budget arms.

**Mood words in `non_diegetic_music`, corpus-wide** (t2va, **revise**): base §4.7 forbids abstract mood words for the
score and nothing checks it; a 2026-09-03 review found them in at least a dozen bank entries. Constant across any
ladder's rungs, so not a confound; a corpus pass, not a hold.

**What `LONG_T2V_PROMPT` carried, and what its rewrite showed.** Four defects against base §4.3 and §4.4: `tracks left`
(conflates `Truck Left` with `Tracking Shot`), `at medium amplitude and moderate speed` (neither in the table),
`whip pan` (absent, and re-describing a cut its header had made), and S2 introduced in Shot 1 but described only in
Shot 2. The rewrite changed only those, same seed, canvas, checkpoint, sampler, no LoRA, 16 steps, and the owner, who
had flagged this one scene as looking and sounding wrong before anyone read the guide against it, judged the result
fantastic (`h3_t2v_00015`, 2026-08-28). That is a resolved complaint, not a measurement. It was the t2v default across
seventeen graph files, so earlier comparisons on the t2v default scored arms on an off-distribution scene; paired
orderings stand, absolute impressions do not.

**The speaker sentence in both dialogue scenes' `overall_soundscape`** (closed `f5b3651`). The sentence describing the
two voices was out of scope by what the field is (base §4.6 lists "non-verbal human sounds"; ref §6 agrees), without
settling whether describing voices repeats them. It was removed from both constants, and the fifth within-shot line
transition gained its own per-line cue ("He answers at once"), so both prompts carry the pacing on every transition and
stay matched. The arm is judged by ear and has not been re-rendered since its baseline moved twice on 2026-08-28. **Not
the same case:** the voice-timbre reference in `ref2va_image_audio_voice`. Ref §6 gives a speaking voice no home in the
two audio sections, and the guide's form lives in `subject_definitions` with its marker in `retention_analysis`;
OPEN, not a defect, because the guide is silent rather than contradictory.

**Markers are in graphs and unjudged.** `h3_ref2v_scene_subway` and `h3_ref2v_scene_kitchen` carry every marker but
`<d>`; they rendered and [`scene_arm_renders.md`](scene_arm_renders.md) reads them, but no marker verdict is recorded.
The `T2V_SCENES` copies (`subway`, `kitchen`, `clinic`) carry markers in t2v form and reach no graph (they are injected
through `--print-scene` / `--set`, by design), so marker coverage in the t2v layout is zero. The repo's three caption
uses encode a signage reading `internal/PROMPTING.md` later withdrew.

**The L2VA render was bad and is undiagnosed.** `h3_l2v_00001` (2026-08-28) was judged bad. Do not repeat it expecting
a different answer. Not ruled out, in order: the prompt body is unvalidated beyond conformance; it rendered at
768x768, not a trained canvas; the mode may be weak (one endpoint, an unconstrained opening); the keyframe is a
landscape photograph, something to converge on rather than a subject to arrive at. Established: the mode runs end to
end and its alignment sentence resolves.

**House-rule findings, not guide defects:** the market Shot 1 runs 11 words against a computed speech budget of 8.75;
the stairwell arms run four turns in one shot, deliberately; "produces no vocal sound" appeared in no prompt at the
time of the audit; and two disagreements for the owner, `internal/PROMPTING.md` §7H's one-unwrapped-line rule against
the owner's later v6 layout, and the guides' `<scenetrans>` / `<cutoff>` spellings, which match nothing in the
release's declared tokens.

The divergences between shipped prompts and vendor practice (shot line breaks, mouth closing, turns per shot) and the
one-line-per-item check are [`prompting.md`](prompting.md) §14.3's.

## What to render next

Ranked by what would be learned.

1. **One marker scene plus its §4.5 double-quoted control.** Neither is interpretable alone; the caption marker is
   unmeasured.
2. **`<scenetrans>` across a cut, and a truncated line, each rendered both ways**: the two markers this repo and the
   sister engine spell differently (this repo writes the continuity in prose and `<|cutoff|>` tight against `</d>`; the
   sister engine requires `<scenetrans>` in both halves and writes `<cutoff>` after a space). No gate on either side
   sees the difference (`grade_prompt_text.py` passed `</d> <cutoff>` on 2026-09-01). The base texts are
   `prompt_bank/t2va_desert_crew.txt` (a line across a cut) and `prompt_bank/t2va_cable_car.txt` (a truncated line);
   each variant is one substitution at render time, confirmed with `cmp`. Built and deferred: the seven arms, the
   owner's calls and the commands are `bench/marker_arms.json` and
   [`docs/state/open_experiments.md`](../state/open_experiments.md) #24. Read as meets-the-brief per arm, never as A/B
   (`docs/method/eval_comparison.md`).
3. **A motion sweep** over the §4.3 types no shipped graph exercises (`Zoom`, `Tilt`, `Pedestal`, `Arc`, `POV`, `Roll`,
   `Shake`). A graded prompt for every row exists in `prompt_bank/`, and [`prompt_bank.md`](prompt_bank.md) derives which
   carries which.
4. **A no-dialogue scene with a real §4.7 score.** The bank carries several, unrendered: `fl2va_drawbridge`,
   `i2va_cargo_bay`, `fl2va_origami_crane`, `fl2va_paper_train`.

**The coverage map** is [`prompt_bank.md`](prompt_bank.md)'s **Not exercised** lines: which value of every closed set
the guides name (frame counts on the grid, §4.3's motion rows and modifiers, §4.2's cut phrasings, §4.1's styles, ref
§3's task types, ref §4's markers) no graded prompt exercises. The sister engine's plan uses that map as the idea set
for its conformance harness, which is why it lives in the tracked tree.
