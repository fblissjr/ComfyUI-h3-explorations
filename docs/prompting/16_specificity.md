> **Part of [`docs/prompting.md`](../prompting.md) (Specificity: What to Pin and What to Leave to the Model (§16)).**
> For the core prompt structure (`§1`–`§5.7`) and master router, see [`docs/prompting.md`](../prompting.md).

## 16. Specificity: what to pin, and what to leave to the model

Added 2026-10-01. **Layer: OWNER for the principle, HOUSE for everything derived
from it; neither guide states any of it** apart from §5.2's placement, which
base §4.1 does. The owner's statement of it: specificity is something to be
graded and evaluated and documented as important, so "specify what matters and
whats important if its ambiguous", but "you can only control so many things in a
prompt / scene - pick the most important ones and let the model guide the rest".

**How much this rests on.** This repo is not research-grade (`CLAUDE.md`, "A
tinkering repo"). Every case below is one clip at one seed, watched by the
owner, and one has a before-and-after render. A rendered clip cannot A/B a
prompt change (`eval_comparison.md`), so read them as observed failures and not
as measurements. The first half of the principle has a long record. **The second
half, what to leave, has none: no record measures that over-specifying hurts**,
and §16.3 says what it does and does not stand on.

### 16.1 Silence is a decision the model makes

The prompt is conditioning, not instruction (§15.1, §15.2). What it leaves open
the model fills with the likeliest thing, and different samples fill it
differently.

| the open slot | what rendered | record |
|---|---|---|
| who drops the coins into the tin (market) | coins from the middle of a crate, from nowhere, or from a hand, depending on the arm | `bench/results/2026-09-15_block49_community_chain.md`, `bench/results/2026-09-18_sol_reorder_panel.md`. The fixed text names the stallholder; no scored render of the fixed text is recorded |
| how many hamburgers (cafe) | two on each plate where the prompt said "a plate of hamburgers" | `sol_reorder_panel.md`; the same panel reported no burger problem on the fixed text, the only before-and-after pair in this section |
| where a speaker sits (cafe) | the speaker staged out of frame and the line given to a character declared silent | `sol_reorder_panel.md`, fixed and confirmed the same way |
| how many people are in the station (subway) | a pursuer cloned, a third person from nowhere, an escalator the text never placed | `bench/results/2026-09-26_distill_compare_s1.md`, `2026-09-26_subway_v2_s1.md`, `2026-09-27_clone_base_control.md` |
| who is silent (radio drama) | every actor's lips moving at the start, because nothing said the listeners were silent | `bench/results/2026-09-27_render_dataset/findings.jsonl` (`ow-fd-04`); the fix is not confirmed |
| what a reference's definition claims, where the description is silent | the defective definition drove the render | [`prompt_length_experiment.md`](../prompt_length_experiment.md) |
| what the script does once it has ended | undeclared people and motion | `eval_comparison.md` ("A stress scene is not a typical scene"), `bench/results/2026-09-18_off_length_prompts.md` |

**Ambiguity costs the evaluation as well as the clip.** An unnamed agent carried
a kitchen-versus-sage kernel question for two days on a hand that did not
reproduce (`eval_comparison.md`), and the owner counts a composition difference
as a prompt-adherence loss (`eval_comparison.md`, 2026-09-05). So pin whatever an
A/B has to hold constant.

### 16.2 What to pin

Each class has one owner. The first six are written up elsewhere; the last four
are written here.

| class | owner | grader |
|---|---|---|
| who causes each event or sound | §15.3 item 6 | nothing |
| how many of each thing | §15.3 item 7 | nothing |
| where each speaker is, identity at first appearance, a voice that fits | §5.2 | nothing |
| who is silent | §5.6, §15.3 item 5 | nothing |
| how many words fit the shot | §5.10 | nothing |
| the literal on-screen string | §6 | nothing |
| the cast count | below | nothing |
| who is in each shot | below | nothing |
| one geography | below | nothing |
| what the scene is a test of | below | nothing |

**The cast count.** Say how many people are in the scene, and where a test
counts people, say that nobody else appears: "only two people in the whole
station: a fleeing suspect and the agent chasing him" (`t2va_subway_chase`),
"No one else appears at the counter or on the street behind."
(`t2va_noodle_bar`). Prefer the positive count to a bare absence. The portable
writer prompts say a model cannot render an absence ("no logo", "nobody else is
there") and to write the visible evidence instead; the bank's own cast fixes
state the exclusion beside the count; nothing here separates the two, and §15.6
records negative constraints as contested. [HOUSE, observed]

**Who is in each shot.** Say who is in frame in each shot and how anyone leaves.
A character present in one shot and absent from the next with no exit rendered
as an empty corridor. [HOUSE, observed on the old subway text, read from the
text in other prompts]

**One geography.** The places, paths and directions the text sets up agree
across the shots, and one path is stated once. A text that contradicts itself,
an "empty" cart that sprays oranges, a gate that hangs open beside its cut
padlock, an "extreme close medium shot", hands the model a coin toss. Prefer
screen-space words ("enters from the left edge of frame") to body-relative or
vague ones. [HOUSE, observed; the screen-space advice is from an external writer
prompt]

**What the scene is a test of.** A prompt written to test something has to make
that thing countable or visible: a mirror wall doubles every dancer on purpose,
so a clone could not be told apart, and the wall is matte black now. The `tests`
field of [`prompt_bank.md`](../prompt_bank.md) says what each scene is a test of.
[OWNER, 2026-09-26: "certain about how specific and well written that prompt is
for testing what u wanna test"]

### 16.3 What to leave to the model, and how to choose

**A pin is necessary and not sufficient.** On the fully pinned subway text the
base model followed it, and the distills still cloned a person, swapped roles or
added an escalator the text never names (`2026-09-26_subway_v2_s1.md`,
[`h3_distills.md`](../h3_distills.md)). How much a pin binds depends on the model,
so the prompt is not the only lever and cannot carry every detail.

**Choosing the pins.** Rank a candidate by what breaks the scene if the model
guesses it wrong: who is on screen and how many; who causes each event the scene
turns on; what a test or an A/B has to hold constant; what the viewer will
count, read or hear literally, such as a number, a string or a spoken line.
Pin those. [HOUSE reasoning from §16.1, not measured]

**Leaving the rest.** Texture, minor props, the exact framing between cuts and
the timing of cuts are left to the model. Cut times are the owner's deliberate
case (§3.1): the one pair measured cut at nearly the same points with and
without them. [OWNER]

**Two things are not left.** Who is on screen, and how long the script runs. A
script that ends before the clip does is filled with undeclared people and
motion (§15.2), so the beats are fitted to the declared frames (§3.3, §5.10).

**What is not known.** Whether over-specifying hurts. The pair built to ask,
the aisle and sortline prompts at two lengths, was never judged, and the owner's
one remark in that vein concerns shots per second on one distill, not words
(`docs/research/pdd/2026-08-28_scene_complexity.md`). "One primary change per
beat" (§15.6) is third-party and unmeasured. Do not read this section as "less
is more" or as "more is better".

**A name collision.** The bank's "specificity ladder" (`t2va_spec_typical`,
`_specific`, `_unusual`) tests how a distill behaves as detail and unusualness
rise (`bench/results/2026-09-27_spec_ladder.md`). It is not evidence for or
against anything here.

### 16.4 How specificity is graded today

- **Mechanically, nothing.** The graders in §11 decide form. The bank-wide fix
  for these classes left every edited prompt grading as it had before (CHANGELOG
  0.151.1), which is the demonstration that the graders cannot see them.
- **By the owner's eye**, against `bench/adherence_checklists.json`: per-scene
  beats (cast, geography, count, role) drafted from each prompt's own text and
  marked with `bench/rubrics/distill_run.json`. No script reads the checklist
  and no score is recorded. Because it is drafted from the prompt it can grade
  only what the prompt pinned, and the scenes that taught these classes (market,
  cafe_kids, diner) have none.
- **By a fitness read**, as in CHANGELOG 0.154.6: is the prompt specific enough
  for what it tests. One-off, with no standing record.
- **`prompt_audit.md`** judges guide conformance from the source text. It is not
  a specificity grade.
- **On the render**, only the cut count (`bench/measure_clip_tone.py`) and the
  Whisper dialogue error (`bench/measure_dialogue_transcription.py`). Nothing
  counts people or checks a direction.

**A reading pass before rendering** [HOUSE; a reading aid, not a gate]:

1. How many people are in the scene, and does anyone else appear?
2. Who is in each shot, and how does each one leave?
3. For every action, who does it? For every sound, what makes it?
4. For every number, is it the number the shot shows?
5. Is every speaker placed, identified at first appearance, and voiced to fit?
6. Is every character who does not speak stated silent?
7. Do the places, paths and directions agree across the shots?
8. What is this scene a test of, and is that countable or visible?
9. What did I leave open on purpose, and am I content for the model to decide it?

The last question is the half of the principle that is easy to forget.
