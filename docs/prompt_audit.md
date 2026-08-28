# Prompt audit: what follows the guides, and what to do about it

last updated: 2026-08-28

The judgement half of [`prompt_catalogue.md`](prompt_catalogue.md), which is
generated and states no opinion. This one is written by hand, keyed to the scene
names that file emits, and carries a verdict per scene: **keep**, **revise**,
**rewrite**, or **discard**.

## Three authorities, never merged

A defect means nothing until you know which authority it breaks, and these are
not interchangeable:

| | what it is | what a violation means |
|---|---|---|
| the official guides | the vendor's own text, `internal/official_prompt_guides/` (base and ref) | the prompt is **off-distribution** from what the model was trained on |
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
| `I2V_PROMPT` | fl2va | keep | conforms |
| `derived:h3_probe_ref2v_split_turbo_pack` | ref2va | revise | budget |
| `DIALOGUE_REF2V_PROMPT` (stairwell) | ref2va | **rewrite** | shot-header format the guides both contradict, and it silently defeats preflight |
| `derived:h3_probe_release_video_policy` | ref2va | keep | a policy probe |
| `DIALOGUE_T2V_PROMPT` (stairwell t2v) | t2v | revise | one out-of-table motion phrase |
| `R2V_PROMPT` (ten-second) | ref2va | **rewrite** | describes 10 s on a 15.083 s graph |
| r2v-swap family | ref2va | revise | defines a `<Picture N>` ref §2.2 says not to define |
| the remaining `derived:` ref2va scenes | ref2va | revise | the budget gap is systematic |

---

## The four that matter

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

### `DIALOGUE_REF2V_PROMPT` — rewrite

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
does not resolve that combination — an ambiguity, not a defect.

One line of §5.2 forecloses the obvious excuse: **"A single shot does not
automatically justify a shorter description."** Our under-budget prompts are all
one shot.

**The fix may not be prose.** `REF_SCENE_SHOTS` exists and is unreachable —
`_ref_prompt(scene=)` defaults to `None` and no call site passes it. Wiring one
argument buys multi-shot ref2va and most of the budget at once. Verify that
before writing 300 words by hand.

---

## Marker coverage: the gap nothing measures

**No shipped graph carries any marker but `<d>`.** `<|lyrics_start|>`,
`<|caption_start|>` and `<|cutoff|>` live only in `T2V_SCENES`, reachable
through `--print-scene` and through no graph.

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

1. **An L2VA prompt.** The only base mode with **zero** coverage. And
   `scene_prompt()`'s L2VA branch returns `[Shot N]` and `S.SS` unsubstituted
   where the fl2v path resolves both — so the gap is a latent bug, not just a
   missing scene.
2. **A conformant market rewrite**, which the generator already names as a
   planned arm.
3. **One marker scene**, plus **its §4.5 double-quoted control**. Neither is
   interpretable alone — and PROMPTING.md says outright the caption marker is
   unmeasured.
4. **A `<scenetrans>` line across a cut** — the one guide mechanism whose token
   does not appear in the release's list. Whatever happens is informative.
5. **Wire `REF_SCENE_SHOTS` through `scene=`** — one argument, and the ref2va
   budget gap mostly closes.
6. **A motion sweep** over the §4.3 types nothing exercises: `Zoom`, `Tilt`,
   `Pedestal`, `Arc`, `POV`, `Roll`, `Shake`.
7. **A no-dialogue scene with a real §4.7 score.** Every current scene is
   `non_diegetic_music: N/A`.

---

## The cheap gate this audit argues for

`docs/checks.md` records motion vocabulary as decidable and unbuilt, splitting
it correctly: a **denylist** of terms absent from §4.3's table is cheap; proving
every phrase in-vocabulary is hard. Until today that row had no escaped
instance. It has one now — the market prompt rendered badly, was read by hand,
and every gate passed it. **That clears this repo's bar for building the
denylist half**, and nothing here argues for the hard half.
