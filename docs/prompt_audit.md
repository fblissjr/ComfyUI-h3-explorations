# Prompt audit: what follows the guides, and what to do about it

last updated: 2026-09-01

The judgement half of [`prompt_catalogue.md`](prompt_catalogue.md), which is
generated and states no opinion. This one is written by hand, keyed to the scene
names that file emits, and carries a verdict per scene: **keep**, **revise**,
**rewrite**, or **discard**.

## Three authorities, never merged

A defect means nothing until you know which authority it breaks, and these are
not interchangeable:

| | what it is | what a violation means |
|---|---|---|
| the official guides | the vendor's own text, `vendor_guides/` (base and ref) | the prompt is **off-distribution** from what the model was trained on |
| `internal/PROMPTING.md` | house rules, derived from the guides plus experience on this box | the prompt deviates from **what we decided**, which may mean the rule is wrong |
| the STATED RULE / NOT A RULE notes above `LONG_T2V_PROMPT` | the existing adjudication of which guide sentences are rules at all | a NOT A RULE finding is not a defect |

Both `internal/` paths are gitignored and do not ship, which is why they are
named in backticks rather than linked. **Two rules have already been invented
here by reading guide *examples* rather than guide *statements*, and retracted**
— so where the guide is ambiguous this file says ambiguous rather than picking a
reading.

**Everything below is a source read.** No claim here rests on a render.

---

## Verdicts

> **The three `rewrite` verdicts were ACTED ON in `d5be353` (2026-08-28), after
> this table was written.** `LONG_T2V_PROMPT`, `DIALOGUE_REF2V_PROMPT` and
> `R2V_PROMPT` are fixed in the generator and every carrying graph was rebuilt
> from it, so the `rewrite` rows below record **what was wrong**, not what still
> is — and the prose beneath them saying the defects are "all still present" is
> superseded by that commit. `check_prompt_guide_conformance.py` is green and
> preflight grades the rewritten graphs at zero FAIL and zero WARN. The `revise`
> rows are open, but read the budget one through its own **Corrected
> 2026-08-28** note below rather than through this table's "systematic" wording.
> The consequence that outlives all of it: because the canonical t2v arm now
> renders different text, **clips from before 2026-08-28 are not matched-seed
> comparable with clips after it.**

Ordered by blast radius — how many graphs carry the scene.

| scene | mode | verdict | why |
|---|---|---|---|
| `LONG_T2V_PROMPT` (market) | t2v | **rewrite** | four official-guide defects, and it is the t2v default |
| `derived:h3_image_ref_plus_text_to_video` | ref2va | **revise** | far under ref §5.2's word budget; one shot |
| `derived:h3_probe_capture_ref3` | ref2va | revise | same budget gap |
| `derived:h3_probe_cache_easy` | ref2va | keep | a cache probe; scene content is not what it measures |
| `derived:h3_first_last_frame_to_video` | fl2va | keep | conforms |
| `I2V_PROMPT` | i2va | keep | conforms. **Was listed as fl2va until 2026-09-01; that was wrong** -- it carries base-en's I2VA Part One template and ships in `h3_first_frame_to_video*`, and the fl2va scene is `derived:h3_first_last_frame_to_video` |
| `derived:h3_probe_ref2v_split_turbo_pack` | ref2va | revise | budget |
| `DIALOGUE_REF2V_PROMPT` (stairwell) | ref2va | ~~rewrite~~ **done `d5be353`, `f5b3651`** | shot-header format the guides both contradict, and it silently defeated preflight. Headers fixed; soundscape closed |
| `derived:h3_probe_release_video_policy` | ref2va | keep | a policy probe |
| `DIALOGUE_T2V_PROMPT` (stairwell t2v) | t2v | ~~revise~~ **done `d5be353`, `f5b3651`** | out-of-table motion phrase replaced; soundscape closed |
| `R2V_PROMPT` (ten-second) | ref2va | **rewrite** | describes 10 s on a 15.083 s graph |
| r2v-swap family | ref2va | revise | defines a `<Picture N>` ref §2.2 says not to define |
| the remaining `derived:` ref2va scenes | ref2va | revise | the budget gap is systematic |

### Verdicts added 2026-09-01

**The table above did not cover every scene the catalogue lists.** The gap was
not neglect: `prompt_catalogue.md` was itself stale until 2026-09-01, so the
missing scenes had never appeared in the document this file is keyed to. Every
one is graded below. To check that this file still covers the catalogue, compare
its scene headings against the verdicts here rather than trusting a count. **All of them pass
`check_prompt_guide_conformance.py` and none carries a FAIL from
`preflight_graph.py`** — the verdicts are about judgement, not mechanics.

| scene | mode | verdict | why |
|---|---|---|---|
| `T2V_AISLE_SHORT` / `T2V_AISLE_LONG` | t2va | **keep** | a matched description-length pair: identical dialogue, cuts, camera moves and subjects, elaborated only. The content IS the manipulation, so a content edit to one and not the other destroys the experiment. Do not "improve" either alone |
| `T2V_SORTLINE_SHORT` / `T2V_SORTLINE_LONG` | t2va | **keep** | the second length pair, on an independent scene, for the same reason. Two scenes carrying one manipulation is what separates a result from an anecdote |
| `T2V_RAIL_LONG` / `T2V_CHURN_LONG` | t2va | **keep** | a pre-registered predictability-versus-delta pair, both ~500 words and one shot, no dialogue. Delta predicts rail is worst; predictability predicts rail is clean. They cannot both be right, which is the point |
| `MARKET_REF2V_PROMPT` | ref2va | **keep** | three shots, two speakers, and inside ref §5.2's word band — one of the few that is. Nothing to revise |
| `derived:h3_ref2v_scene_kitchen` | ref2va | **keep** | four shots, inside the word band, and the first shipped prompt to carry the lyrics, caption and cutoff markers. `docs/scene_arm_renders.md` is its viewing guide |
| `derived:h3_ref2v_scene_subway` | ref2va | **keep** | as kitchen. **Not an arm of the same experiment** — different scene, reference and speaker count, so nothing is learned by comparing the two to each other |
| `derived:h3_last_frame_to_video` | l2va | **keep** | the L2VA prompt whose absence §13 of `prompting.md` reported until 2026-09-01. Part One resolves fully; conforms |
| `derived:h3_ref_audio_voice` | ref2va | revise | budget only: `detailed_description` sits in the low forties against 350-500 |
| `derived:h3_ref_image_audio` | ref2va | revise | budget only |
| `derived:h3_ref_video_continue` | ref2va | revise | budget only |
| `derived:h3_ref_video_motion` | ref2va | revise | budget only |
| `derived:h3_ref_video_only` | ref2va | revise | budget only |
| `derived:h3_ref_video_to_video` | ref2va | revise | budget only |
| `derived:h3_ref_video_edit` | ref2va | **keep** | inside the word band |
| `derived:h3_ref_video_image_edit` | ref2va | **keep** | inside the word band, and the densest label set that ships — five items, each correctly on its own line |

**The budget `revise` rows are one defect, not six**, and they are the same
finding the catch-all row above already carried. They are listed individually
only so the catalogue's scene names all resolve to something here.

### Misalignments: one closed, one withdrawn, one open

`prompting.md` §14.3 is the long form and owns all three.

1. **Shot line breaks — CLOSED 2026-09-01.** Vendor practice, and a third-party
   corpus, run multi-shot prompts inline in one paragraph with no
   counterexample. `LONG_T2V_PROMPT` and the four aisle/sortline arms were
   collapsed in the generator and every carrying graph rebuilt; word counts are
   unchanged, only newlines. **Clips of those scenes from before that date are
   not matched-seed comparable with clips after it.** The length-experiment
   pairs changed identically, so that experiment is intact.
2. **Mouth closing — WITHDRAWN.** This file previously recorded our dialogue
   prompts as diverging on it. They do not. The rule is positional — cue when
   the shot continues, never when the line ends the shot — and our corpus
   satisfies it. The earlier finding came from a per-line ratio, which is the
   wrong statistic for a positional rule. **Provenance caveat:** the pattern is
   a third-party corpus observation, NOT vendor practice — the vendor's
   shot-final cell holds one line and its mid-shot cell runs the other way. Do
   not cite it as vendor.
3. **Turns per shot — CLOSED 2026-09-01 by the owner.**
   `DIALOGUE_T2V_PROMPT` stacks four turns in `[Shot 1]` and three in
   `[Shot 2]` against a base-format vendor practice of one. **The owner's
   verdict on the render is that the scene clearly works**, so this is a
   sanctioned capability rather than a divergence to fix, and the scene keeps
   its **keep** verdict. One render settles it because "does the model deliver
   stacked turns at all" is presence/absence; it does NOT establish that
   stacking beats cutting, which would need matched seeds and a distribution.
   `prompting.md` §14.3 carries the reasoning and the vendor-evidence
   correction that went with it.

**Checked and found clean:** ref-en *states* one line per item in
`subject_definitions` (ref-en:37) and `retention_analysis` (ref-en:157). Every
shipped ref2va prompt satisfies both. A first pass flagged twelve by counting
labels per line; that test is wrong, because ref-en:37 explicitly allows a
source-only `<Picture N>` to be cited inside another item's line.

---

## The four that matter

### `LONG_T2V_PROMPT` — rewritten, and the render is the owner's verdict

**Outcome, 2026-08-28.** The rewritten scene rendered (`h3_t2v_00015`) and the
owner's judgement was "fantastic", against a pre-existing complaint that this
same scene looked and sounded wrong while everything else looked fine.

**What that is, and what it is not.** It is one render at one seed, so it is not
a measured claim and must never be cited as one — CLAUDE.md's different-sample
rule applies and a distribution was not gathered. But it is the strongest shape
this class of evidence takes, and the shape is worth naming because the parts
are what make it worth anything:

- the complaint was **specific and predated the diagnosis** — the owner
  identified this scene, not a general dissatisfaction, before anybody read the
  guide against it;
- the defects were then found **independently by reading the vendor guide**,
  not by looking for something to blame;
- the fix was **targeted at those defects only** — three camera phrases into
  base 4.3's closed sets, one speaker identity moved to first appearance;
- and **nothing else moved**: same seed, same canvas, same checkpoint, same
  sampler, no LoRA, 16 steps.

So the honest statement is: a scene the owner had flagged as bad, carrying four
documented guide violations, stopped being bad when only those violations were
repaired. That is a resolved complaint, not a measurement, and the difference
matters if anyone later wants to claim camera vocabulary is worth N percent of
anything.

**The practical consequence is bigger than one scene.** This prompt was the t2v
default across seventeen graph files. If the rewrite is why the render improved,
every earlier comparison on the t2v default was scoring arms against each other
on a degraded scene. Paired arms shared it so their orderings stand, but the
absolute impressions from those runs were formed on something the vendor's own
guide says is off-distribution.

### `LONG_T2V_PROMPT` — rewrite

The t2v default, and the widest-reaching prompt in the repo. Four defects
against the **official** guide, all recorded on 2026-08-27 and all still
present:

- `tracks left` conflates `Truck Left` with the separate `Tracking Shot` row of
  base §4.3's table — two different entries.
- `at medium amplitude and moderate speed` — neither value appears in that table.
- `whip pan` is absent from the table entirely, and it re-describes a cut the
  `[Shot 3] At 00:09.000` header has already made.
- S2 is introduced in Shot 1 as "a young porter (S2)" and not described until
  Shot 2. Base §4.4 requires identity **where the speaker first appears**.

`bench/preflight_graph.py` grades it green; motion vocabulary is the
"enforced by nothing" row in [`checks.md`](checks.md).

**Why this is the priority.** It is not one bad scene — it is the default across
the canvas probes, head-chunks, sol-on, square-canvas, split-base and
turbo-owner arms. Every comparison that used the t2v default rendered a scene
carrying three untrained camera phrases. Both arms shared it, so paired results
are not invalidated; but they were exercising an off-distribution scene rather
than a representative one, which is a weaker claim than those runs implied.

**The rewrite is mechanical**: three phrases swapped for §4.3 vocabulary, the
porter's identity moved into Shot 1. The scene itself is fine.

### `DIALOGUE_REF2V_PROMPT` — rewrite, discharged `d5be353`

Shot headers read `[Shot 1, 00:00.000-00:06.000]`. Ref §5.1 and base §4.2 both
state `[Shot 1]` takes no timestamp and later shots use `[Shot N] At MM:SS.mmm`.

**The second half is worse than the first.** `bench/preflight_graph.py`'s shot
regex requires a literal `]` after the digits, so on this prompt `shots` comes
back empty and **three shot rules go inert** — silently. A malformed header does
not fail the grader, it removes the grader. That is the "correctly absent and
not covered look identical" trap, in a live prompt.

### `R2V_PROMPT` — rewrite

States "throughout the ten-second sequence" and lays spans out to `10.0s`, on a
graph that renders 362 frames — **15.083 s**. Five seconds are described by
nothing. Its camera clause (`executes a controlled, slow-speed lateral truck…
on a smooth dolly axis, tracking parallel to`) is outside §4.3's vocabulary in
every dimension at once.

### The ref2va budget gap — revise, and narrower than first stated

**Corrected 2026-08-28.** This section said the gap was systematic across every
ref2va prompt. Ref §5.2 scopes it more tightly than that, and the scoping
changes which prompts are actually in breach:

- the range is **350-500 words for GENERATION tasks**, and it is "normally",
  not a hard bound;
- **video-editing descriptions are exempt** — the guide says they "scale with
  the complexity of the source video and do not have to follow the
  generation-task range";
- **dialogue-dense content** may prioritise fitting the spoken timeline over
  reaching a count;
- and **base_en states no word budget at all**, so this is a ref2va rule only.

Classified by the task prefix each prompt declares: the ten pure
`reference generation` prompts (42-198 words) are genuinely under. The three
`video editing` ones are exempt. `video continuation` is not named by the rule.
Two of the editing prompts also declare `reference generation`, and the guide
does not resolve that combination — an ambiguity, not a defect, and a sharper
one than it first looks. **`[video editing + reference generation + audio reuse]`
is ref §3's own worked illustration of the prefix format** (ref_en:127), so a
job that is simultaneously an edit and a generation is the shape the guide chose
to demonstrate its syntax with, not a corner we backed into. Ref §5.2, fifteen
sections later, gives the generation range and the editing exemption and never
mentions the combination.

Treat it as **unresolved rather than decided in either direction**: picking one
would be house judgement wearing a guide citation. The half both readings agree
on is safe to act on — never pad toward the range to satisfy it, and let
information load set the length.

One line of §5.2 forecloses the obvious excuse: **"A single shot does not
automatically justify a shorter description."** Our under-budget prompts are all
one shot.

**The fix may not be prose.** `REF_SCENE_SHOTS` exists and is unreachable —
`_ref_prompt(scene=)` defaults to `None` and no call site passes it. Wiring one
argument buys multi-shot ref2va and most of the budget at once. Verify that
before writing 300 words by hand.

---

### Both dialogue scenes named their speakers in `overall_soundscape` — CLOSED `f5b3651`

Raised 2026-08-28 by a peer session auditing the catalogue, held open the same
day, and closed the same day. **The sentence no longer exists in any prompt**;
this entry is kept because the reasoning is what will otherwise be
re-litigated, and because two things asserted here turned out not to be facts.

**What was there.** Both `DIALOGUE_T2V_PROMPT` and `DIALOGUE_REF2V_PROMPT`
carried the identical sentence "Two speaking voices, a measured adult female
voice and a lower adult male voice, trading short clipped lines with almost no
gap between them" in `overall_soundscape`, while already declaring those
speakers with `<d>` and stable ids in the main field.

**The grounds are the field's positive scope, not the no-repeat clause.**
base §4.6 does say dialogue "should not be repeated here", and a describing
sentence arguably slips past that since it repeats no words. The operative
reading is the same sentence's enumeration: the field summarizes "ambient
sound, physical action sounds, and **non-verbal** human sounds", listing
breathing, laughter and panting. Verbal human sound is the category that
enumeration excludes by naming its complement. ref §6 reaches the same place
independently. So it was out of scope by what the field IS, without having to
settle whether describing voices counts as repeating them.

**Two guards argued against touching it, and both were weaker than they
looked.**

*The comment calling the sentence load-bearing.* `build_workflows.py` named it
one of three devices making the arm judgeable and closed "Do not 'improve' it
without rendering the result beside this one" — and its empirical half,
"remove those and the same lines come out spaced and unjudgeable", **was never
observed**. No arm has been rendered without them; `bench/results/` has
nothing either way. Corrected in the comment rather than quietly dropped.

*The "reproduced verbatim" claim.* Also false by the time it was read.
`d5be353` had already changed the camera phrase in both constants on base §4.3
grounds with no paired render. The 2026-08-08 clips are the output of a prompt
neither constant now matches, so the paired-render instruction had one
disclosed exception before this change and has two after it.

**The asymmetry, and why the fix dissolves it rather than trading it off.**
This is the part worth keeping. Both prompts carried "answers immediately"
twice and "says at once" twice, covering four of the **five** within-shot line
transitions — Shot 1's fourth line was a bare "He says". `DIALOGUE_REF2V_PROMPT`
also carried the cue a third time in `summary`; `DIALOGUE_T2V_PROMPT` has no
`summary` field. So dropping the sentence from both cost them different things,
and "treat both constants the same way" (which the matched pair requires) pulled
against the content (which was not symmetric).

The move that resolves it is not moving the global sentence into
`integrated_multimodal_description` — that was proposed here and is **not** what
shipped, because the guide only ever shows the per-line form in that field.
`f5b3651` instead closed the fifth transition with a fifth per-line cue ("He
answers at once"), in the form the guide demonstrates, in both constants. Both
prompts now carry the pacing on every within-shot transition, the ref2va twin
still has its `summary` line, the pair stays matched, and neither loses
anything. **Nothing was traded.**

**Not yet done, and not a documentation matter:** the arm is judged by ear and
its baseline moved twice on 2026-08-28. It has not been re-rendered against the
2026-08-08 clips, which are in any case no longer the same prompt.

**A related case that is NOT this and must not be swept in**: the voice-timbre
reference in `derived:h3_ref_audio_voice`. Ref §6 places a reference
relationship "in the section that matches the audible layer" and names two —
ambience/SFX, and audience-only score. A speaking voice is neither, so the guide
gives it no home there; the form the guide does give lives in
`subject_definitions` with its marker in `retention_analysis`. That reads as the
wrong layer, but the guide is **silent** rather than contradictory, and this
repo has retracted two rules read out of guide silence. OPEN, not a defect.

## Marker coverage: reachable since `930d296`, still unrendered

**Until 2026-08-28 no shipped graph carried any marker but `<d>`.** The other
markers lived only in scene text that `_ref_prompt(scene=...)` could render and
that **no call site ever asked for**, so `<|caption_start|>`,
`<|caption_end|>`, `<|lyrics_start|>`, `<|lyrics_end|>` and `<|cutoff|>`
appeared in the generator and in zero graphs. `h3_ref2v_scene_subway` and
`h3_ref2v_scene_kitchen` wire two of those scenes and close that.

Both were queued 2026-08-28.
[`docs/scene_arm_renders.md`](scene_arm_renders.md) is the viewing guide: what
one render each can settle, what it cannot, and why the caption route is
deliberately confounded.

**What is still open, and it is the larger half.** The arms exist; nothing has
been rendered through them, so every marker but `<d>` remains *unexercised
against the model* even though it is now *present in a graph*. Wiring changed
what the corpus contains, not what has been observed.

The `T2V_SCENES` copies of the same five scenes remain unwired. `subway`,
`kitchen` and `clinic` each carry markers in t2v form and reach no graph;
they are injected through `--print-scene`/`--set` rather than shipped, which
is a deliberate design and not an oversight — but it does mean marker coverage
in the t2v layout is still zero.

This matters more than a coverage gap normally would, because this repo has
spent real effort on marker questions — the seven ids, the untrained rows, the
marker arms — while **rendering nothing that uses them**. Note also that
PROMPTING.md §7F's caption-as-subtitle reading postdates all three of our
caption uses, which encode a signage reading it withdrew.

---

## House-rule findings, reported separately

Not official-guide defects. Listed so the distinction survives.

- **§4.2 speech budget**: the market Shot 1 runs 11 words against a computed
  budget of 8.75.
- **§4.3 turn-taking**: the stairwell arms run four turns in one shot. This is
  **deliberate and documented** in the generator; it is an experiment, not a slip.
- **"produces no vocal sound"**: no prompt states it. PROMPTING.md asks for it.
- **Two disagreements, reported and not resolved** — these are for the owner:
  PROMPTING.md §7H's one-unwrapped-line rule against the owner's later v6
  layout; and the guides' `<scenetrans>` / `<cutoff>` spellings, which **match
  nothing in the release's declared token list**. The second is the more
  interesting: a guide describing a mechanism whose token does not exist.

---

## What to design next

Ranked by what would be learned, not by count.

**Items 1 and 2 are DONE and are kept as done rows rather than deleted**, so the
ranking still reads as the argument it was and nobody re-proposes them. L2VA
shipped in `b753fe1` (`h3_last_frame_to_video`), the market rewrite landed in
`d5be353`, and the latent placeholder bug item 1 predicted — `scene_prompt`'s
L2VA branch emitting `[Shot N]` and `S.SS` literally — was real and was fixed in
`d964088`. **Neither has rendered yet**, so what they bought is conformance, not
a result.

1. ~~**An L2VA prompt.**~~ Shipped `b753fe1`; the latent bug it predicted was
   real. **Rendered 2026-08-28 (`h3_l2v_00001`) and the owner's verdict was
   bad. DEFERRED, not diagnosed** — nobody has looked at why, and the candidate
   causes are not separated, so do not repeat the render expecting a different
   answer. What is NOT yet ruled out, in the order I would rule it out:
   the prompt body is mine and unvalidated by anything but conformance; it
   rendered at 768x768, which is square and not the trained canvas this repo
   insists on for anything load-bearing; the mode itself may simply be weak,
   since the model gets one endpoint and an unconstrained opening; and the
   keyframe is a landscape photograph, which is a thing to converge *on* rather
   than a subject to arrive at. The one thing that IS established is mechanical:
   the mode runs end to end and its alignment sentence resolves.
2. ~~**A conformant market rewrite.**~~ Done, `d5be353`.
3. **One marker scene**, plus **its §4.5 double-quoted control**. Neither is
   interpretable alone — and PROMPTING.md says outright the caption marker is
   unmeasured.
4. **A `<scenetrans>` line across a cut, rendered both ways, and a truncated
   line rendered both ways** — the two markers this repo and the sister engine
   spell differently. `<scenetrans>` matches no declared token: this repo
   writes the continuity in prose only, the sister engine requires the tag in
   both halves. `<|cutoff|>` is the declared token: this repo writes it tight
   against `</d>`, the sister engine writes the guide's `<cutoff>` after a
   space. Neither side has rendered either form, and no gate on either side
   can see the difference — `grade_prompt_text.py` passed `</d> <cutoff>`
   with no warning on 2026-09-01. Four arms, matched seeds, read as
   meets-the-brief per arm and never as A/B (`docs/eval_comparison.md`).
   Both repos proposed this render independently on 2026-09-01. Whatever
   happens is informative. The two base texts are in the bank:
   `prompt_bank/t2va_desert_crew.txt` carries the line across a cut in prose
   and `prompt_bank/t2va_cable_car.txt` the truncated line; each variant is one
   substitution made at render time and confirmed with `cmp` before queueing.
   **Built and deferred 2026-09-01:** the seven arms, the owner's calls and
   the exact commands are in `bench/marker_arms.json` and
   [`open_experiments.md`](open_experiments.md) #24. Nothing has rendered.
5. **Wire `REF_SCENE_SHOTS` through `scene=`** — one argument, and the ref2va
   budget gap mostly closes.
6. **A motion sweep** over the §4.3 types no SHIPPED graph exercises: `Zoom`,
   `Tilt`, `Pedestal`, `Arc`, `POV`, `Roll`, `Shake`. A graded prompt for every
   row of the table now exists in `prompt_bank/` and
   [`prompt_bank.md`](prompt_bank.md) derives which carries which; the render
   is still owed.
7. **A no-dialogue scene with a real §4.7 score.** Every current scene is
   `non_diegetic_music: N/A`. The bank carries
   several, unrendered: `fl2va_drawbridge`, `i2va_cargo_bay`,
   `fl2va_origami_crane`, `fl2va_paper_train`.

**The coverage map.** [`prompt_bank.md`](prompt_bank.md) derives, from the
graded prompts in `prompt_bank/`, which value of every closed set the guides
name is exercised and which is not: the frame counts on the grid, base §4.3's
motion rows and modifiers, §4.2's cut phrasings, §4.1's styles, ref §3's task
types and ref §4's markers. Read its **Not exercised** lines rather than a list
here, which would rot; on 2026-09-01 the only one was `keyframe completion`,
which no shipped ref2va graph can carry. The sister engine's plan names that
map as the shared idea set for its local-model conformance harness, which is
why it lives in the tracked tree rather than under `internal/`.

---

## The cheap gate this audit argues for

`docs/checks.md` records motion vocabulary as decidable and unbuilt, splitting
it correctly: a **denylist** of terms absent from §4.3's table is cheap; proving
every phrase in-vocabulary is hard. Until today that row had no escaped
instance. It has one now — the market prompt rendered badly, was read by hand,
and every gate passed it. **That clears this repo's bar for building the
denylist half**, and nothing here argues for the hard half.
