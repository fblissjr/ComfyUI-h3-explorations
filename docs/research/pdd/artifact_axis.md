# What actually makes a distilled clip fall apart

One evening, 2026-08-28, three sessions, four accounts registered and **three
of them dead by the end of it**. The one still standing was the owner's, stated
before anyone measured anything.

Written the night the arms were queued and before any of them rendered.
**Section 6 is pre-registered and section 7 is what to do with the results.**
If you are reading this after they landed, score section 6 before forming a
view — the whole point of writing it down was that we kept fooling ourselves.

[`audio_under_pdd.md`](audio_under_pdd.md) is the audio side of the same night
and is separate. [`../../eval_comparison.md`](../../eval_comparison.md) owns the
process rule this produced.

---

## 0. WHAT YOU ARE LOOKING AT — read this first

All clips land in the output `Video/` folder. VHS writes two files per render:
`NAME_00001.mp4` is video only and `NAME_00001-audio.mp4` is the same video
with the audio muxed in. **Watch the `-audio` one.**

### Already rendered, waiting on your eye

| file starts with | what it tests | what to look for |
|---|---|---|
| `shotablation_shots23_` | **the third ablation point.** Two shots, the crate lift plus the wide walk-away | Simply: is it good like the other two? Three-for-three means two setups always hold and the split is clean |
| `h3_r2v_scene_subway_` | markers — sung vs spoken in one clip, a truncated last line, on-screen sign | Do the wrapped lines actually **sing**? Does the last line run out mid-sentence? The platform sign already reads correctly |
| `h3_r2v_scene_kitchen_` | same, second scene | Same three. **Caveat: two of its four prompted cuts did not happen**, so shots 3 and 4 are not where the prompt put them |

### Queued tonight, six of mine, in queue order

| file starts with | tests | if it looks GOOD | if it looks BAD |
|---|---|---|---|
| `h3_t2v_rail_long_` | **the big one.** One 15 s take, rail dolly across a boxy house, nothing else moves. Long and densely detailed | **Length, detail and delta are all cleared at once** — only your predictability reading survives | Delta was right and three of our accounts were wrong for nothing |
| `h3_t2v_churn_long_` | same length, same one-shot, but handheld with flashing LEDs and a crowd | Predictability is not the axis either — back to the drawing board | Confirms the pair: same delta, opposite structure, opposite result |
| `h3_t2v_aisle_short_` | hardware aisle, normal-length prompt | — | — |
| `h3_t2v_aisle_long_` | **same scene, more detail asked for.** Same shots, cuts, dialogue, camera | Long being fine supports "length is not the problem" | Long being worse means demand matters after all |
| `h3_t2v_sortline_short_` | recycling sort line, normal length | — | — |
| `h3_t2v_sortline_long_` | same, elaborated | Two scenes agreeing is a result; one is an anecdote | — |

**Watch the rail one first.** It is the only clip that can settle three
questions at once, and ghosting on a flat concrete wall with square windows is
impossible to miss or to argue about.

**One warning on the four aisle/sortline clips.** They are all three setups in
fifteen seconds, which is the regime you identified as broken. If all four look
bad, that is probably the floor rather than an answer about prompt length —
the PDD session's 241-frame arms are the ones that tell us which. So do not
read "both look bad" as "length does not matter".

### Not mine, running in the same queue

The PDD session has duration-control arms at 241 frames, and the audio session
has a carry probe. Their naming is theirs; ask them. Both were told what is in
this file.

---

## 1. The observation

Distilled arms ghost and smear — a lifted crate bends, faces melt mid-clip,
fruit on a stall loses its edges — and the dialogue scenes at 4 steps look
fine. The question was why, and whether "looks fine at 4 steps" was ever a
statement about distillation rather than about the scenes it was said on.

## 2. Account 1 — DELTA. Dead.

**Claim.** Severity tracks frame-to-frame change, measured at +0.676
within-clip, coarse partition against fine, cuts masked.

**How it died.** Twice, and the second time is the interesting one.

Across shots it was already backwards: in the market clip the owner names the
shot he finds worst, and it carries **0.23x the delta** of the shot he calls
nearly clean. The camera is locked there, so frame-differencing reads a quiet
shot where the eye reads a busy one.

Then the shot ablation. `shots12` has the **highest** delta of the three arms —
above the bad clip — and is fine. An account that predicts severity from delta
has to call it the worst.

**The +0.676 is not retracted.** It is within-clip and frame-level, and it can
hold while delta fails as an across-clip account. What is retired is using
delta to choose or judge a scene.

**The completed set, measured.** `shots23` landed after the accounts above
died, and is unjudged:

| arm | delta | detail | verdict |
|---|---|---|---|
| 3-shot original | 0.0149 | 0.0635 | **bad** |
| `shots12` | 0.0209 | 0.0580 | great |
| `shots13` | 0.0122 | 0.0640 | great |
| `shots23` | 0.0165 | 0.0622 | **unjudged** |

Delta spans the two-shot arms in both directions around the bad clip, which is
the shape of a variable that is not doing the work.

**A label came off it too.** "Motion" was a gloss; the quantity is
mean |frame[n] − frame[n−1]|, which a cut, a light switching on and a whip pan
all score high on with nothing moving. `bench/measure_clip_delta.py`.

## 3. Account 2 — DETAIL DEMANDED. Dead on the arm that tested it.

**Claim.** What matters is the fine structure the prompt asks to be resolved —
a wide market shot gives each orange one or two latent cells where a closeup
gives a face hundreds. Demand is a cause and is readable from the prompt;
detail measured on the render is a result, and is ambiguous, because ghosted
fruit reads as low detail.

**How it died.** The demand ordering read off the market prompt makes `shots12`
the lowest-demand arm, so predicted best. It was judged worse than `shots13`.
Then both two-shot arms came back good, so the ordering explains nothing at
all.

**Not fully buried.** The `aisle` and `sortline` pairs in section 6 vary demand
at fixed shot count, which no shot swap can do. If long is not worse than short
in either scene, demand is finished rather than one arm down.

## 4. Account 3 — THE CRATE LIFT. Dead within the hour it was proposed.

**Claim.** The failure is attached to one action, not to any aggregate — the
owner had named the lifted crate as the break point repeatedly, across renders,
before any of this analysis. `shots12` contains it and `shots13` does not.

**How it died.** `shots12` contains the crate lift and **the box did not
ghost** — the owner said so unprompted. The same action, from the same prompt
text, ghosts in the three-shot clip and holds in the two-shot one. So the
action is not sufficient, and something about its surroundings is doing the
work.

That is also the cleanest within-scene control the night produced, and it
arrived by accident.

## 5. What survives — the owner's, in two parts

**Part one, stated before any measurement:** *"too much shit in one scene? so
you make the stuff gigantic and it loses its shit."* Three setups in fifteen
seconds breaks; two does not, whichever two.

**Part two, stated after both two-shot arms came back clean, and it is the
sharper half:** *"i bet even long prompts are fine. so long as you dont
introduce like 5 unique shots and tons of shit changes everywhere like lights
flashing different led colors every second of every shot from different places
and people moving around or a handheld camera fight scene where the model cant
easily predict what comes next like it could say... a tracking shot of a boxy
house going from left to right across. steady and not changing position at all,
just moving left to right like its on a rail."*

**The axis is how hard the next frame is to predict, not how much it changes.**
That single distinction is what all three dead accounts missed, and it is why
the rail example is worth a render on its own: a rail move translating the
whole frame is near-maximal delta and near-zero uncertainty. Everything is
where it was, shifted.

## 6. Queued, and what each one can kill — PRE-REGISTERED

Six arms, all t2v, 1344x768, 362 frames, PDD 4-step, euler, sage and Sol on,
matching the market PDD configuration. Queued 2026-08-28, none rendered at
time of writing.

**The predictability pair. One shot each, no cuts, both long** — so shot count,
duration and length are all held and only predictability varies.

| arm | what it is | delta says | owner's reading says |
|---|---|---|---|
| `rail_long` | rail dolly across a boxy house, **nothing else moves** | worst | **clean** |
| `churn_long` | handheld night market, independently flashing LEDs, crowd, steam | — | breaks |

**`rail_long` is the load-bearing arm of the night.** It is long, densely
detailed and the highest-delta thing in this repo. **Clean → length, detail
demand and delta are exonerated in one render and only predictability
survives.** Ghosting on a rigid concrete facade is unmissable, so it needs no
scoring finesse.

**The demand pairs. Two scenes, fixed shot count**, holding subjects, actions,
camera moves, dialogue and cut times exactly, varying only elaboration.

| arms | demand says | length-of-conditioning says |
|---|---|---|
| `aisle_short` / `aisle_long` | long worse, on the elaborated surfaces | no consistent direction |
| `sortline_short` / `sortline_long` | same, in both scenes | — |

**A confound to know before reading a null from these**, raised by the PDD
session: all four are three setups at about five seconds each, which is the
regime part one identifies as broken. If the real axis is per-shot duration
rather than total content, both short and long come back bad and that reads as
"no length effect" when it is a floor. The 241-frame arms from the PDD lane
land first and say which regime these are in. **Do not interpret a null here
until those have landed.**

## 7. The two mistakes that killed our accounts, in general form

Worth more than any of the findings, because both were made twice in one
evening by different people.

**Scoring a subset of a set that is still rendering.** Delta looked *confirmed*
by exactly the arm that refutes it, because two arms of three had landed. The
crate-lift account died the same way, on a comparison proposed while a third
arm was queued. The fix is not more caution about which account to believe; it
is not scoring until the set is complete.

**Ablating by removal moves everything at once.** Dropping a shot removes its
delta, its demanded content, its specific action, and lengthens every surviving
shot. Four variables, one manipulation, and the design cannot separate them by
construction. Every account in sections 2 to 4 was fitted to a comparison that
could not distinguish them.

**And the record of who was right.** Three registered accounts died tonight,
all three from the sessions, none from the owner. His was stated first, in
plain language, from watching the clips, and has survived every arm so far.
That is not a compliment; it is a note about where the hypotheses should come
from next time.
