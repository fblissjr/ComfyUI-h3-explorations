# The eval flywheel: how a month of looking became rules, and which rules survived

last updated: 2026-11-10

A dated record of the masked lane's project `fun`, from 2026-10-04 to 2026-10-31: how the owner's looks at renders were
turned into tools that looked first, and what that taught. Written at the owner's ask once `fun` closed. Subjects are
named by their labels (`lead`, `second`, `third`).

**What it rests on:**
- a timeline built from the dated decision files (`docs/wiki/decisions.md`, `docs/wiki/decisions/`), the postmortems
  (`docs/research/postmortems/`), the results records, the checks index (`docs/wiki/ops/checks.md`), and the owner's answers
  (`data/fun/owner_answers.jsonl`, untracked);
- a subagent built the timeline; about 30 of its citations were spot-checked against the files and every one held.

Every claim points at the line that says it. Figures point at how they were counted.

## The short version

- **At the start the owner looked at everything.** Every render was judged by eye, for every kind of fault, and the
  first postmortem of the project says it plainly: "Nothing that caught a real fault was a check"
  (`docs/wiki/research/postmortems/2026-10-10_span_fun-three-people.md:73`). The catches came from a second session
  reading the real files, or from the owner.
- **Each catch was turned into something that looked first:** a rule, a flag, a check, a tool or a page. New checks
  peaked between 2026-10-10 and 2026-10-14, and the owner's answers per day fell from hundreds to tens.
- **Many of those rules were then killed.** The ones that died were mostly thresholds or keep-outs made for one clip and
  spread to others. The ones that lasted caught structural faults with a hard signature: frames out of place, a wrong
  frame rate, a stale or reused input, a laid list that disagrees with the cut.
- **The flywheel slowed the owner's looks; it never replaced them.** Whether a face, a mouth or hair looks right stayed
  the owner's call to the end. Those catches became recipes and plan words, not checks.

## How the load moved

Counted from `data/fun/owner_answers.jsonl` and `data/fun/verdicts/verdicts.jsonl`, by the UTC date of each line's `at`
(`at[:10]`), owner lines only (`who == "owner"`).
- **The "who" column** counts the who-is-who page's clicks (`row == "who"`).
- **Verdicts:** these are deduplicated on (`at`, `row`, `chosen`, `words`), because through 2026-10-16 each answer
  was written twice.
- **What is not counted:** the answers file starts 2026-10-11, when the local page was built. Looks before that went
  to a hosted board.

| UTC day | owner answers | of them who-is-who clicks | verdicts |
|---|---|---|---|
| 10-11 | 269 | 36 | 123 |
| 10-12 | 746 | 700 | 46 |
| 10-13 | 138 | 93 | 45 |
| 10-14 | 52 | 0 | 53 |
| 10-15 | 68 | 0 | 68 |
| 10-16 | 41 | 0 | 62 |
| 10-19 to 10-20 | 19 | 0 | 23 |
| 10-21 | 115 | 0 | 57 |
| 10-22 | 53 | 3 | 52 |
| 10-23 | 70 | 0 | 78 |
| 10-24 to 10-27 | 67 | 0 | 59 |
| 10-28 to 10-31 | 96 | 0 | 63 |

**New masked-lane checks, by day.** Each `bench/check_*.py` is dated by the commit that added it
(`git log --diff-filter=A --format=%ad --date=short`), filtered to the masked lane by what its row in `docs/wiki/ops/checks.md`
defends, then curated by hand:

| days | checks added |
|---|---|
| 10-04 to 10-07 | 14 |
| 10-10 | 13 |
| 10-11 | 5 |
| 10-12 | 11 |
| 10-13 | 13 |
| 10-14 | 9 |
| 10-15 to 10-17 | 14 |
| 10-18 to 10-27 | 1 to 3 on some days, none on others |
| 10-28 to 10-31 | none |
| **total** | **92** |

**How to read the two tables together:**
- **The who-is-who page moved one kind of question onto one page.** On 2026-10-12 the owner answered 700 who-is-who
  clicks in a day. Those answers became the tracker's seeds for the rest of the project, and the question was never
  asked again at that scale.
- **The other answers settled at tens a day once the checks were in.** The 2026-10-21 rise is the day the owner set
  the rules for passes, prep and judging (`docs/wiki/masked_v2v/AGENTS.md`, "The owner's rules for passes, prep and
  judging").
- **What the tables cannot show:** answers per render. The capture and job folders mix previews, jobs and captures, so
  there is no clean count of renders a day to divide by.

## Four stages

### 1. Everything by eye (2026-10-04 to 2026-10-10)

The first catches came from stacked videos and stills the owner looked at.
- **Original pixels:** `keep` pixels brought the original man back inside the regenerated region.
- **Stale masks:** a confirmation render loaded a stale kept mask (`docs/wiki/decisions/2026-10-05_to_2026-10-09.md:85-91`).
- **The text:** the owner, before any session, named the prompt text as why a mouth stayed shut
  (`docs/wiki/research/postmortems/2026-10-10_span_fun-three-people.md:95`).

Each became a rule: no original pixels of the subject inside the region; nothing an earlier run made is reused by a key
(every mask store removed, `docs/wiki/decisions/2026-10-10_part2.md:40-55`); a text is data and a caption, not an
instruction.

The first rule that refused work also came here. "The look" refused to build a load nobody had looked at
(`docs/wiki/decisions/2026-10-10_part2.md:5-7`), after an independent reader found four wrong loads among twelve
queued.
- **Why it went:** two days later it was killed as a refusal. It spent a reader on every ordinary load to get past it,
  and the owner ruled that a test that is broken and causing friction goes
  (`docs/wiki/decisions/2026-10-12.md:168-187`).
- **What replaced it:** the look stayed, shown and not refused.

### 2. Rules pile up, then the simplification (2026-10-10 to 2026-10-16)

This is where the flywheel spun fastest: about 60 new checks in a week. It is also where most of the rules that later
died were born.
- **The good:**
  - **The who-is-who page** (`bench/who_page.py`): one page for a question the owner had answered through five forms.
  - **Stacked clips** with masks and frame numbers burned in: one owner still had ended a 23-minute search by three
    sessions.
  - **Masks on the right frames:** `verify` now fails masks laid on the wrong frames (`docs/wiki/decisions/2026-10-13.md:75-94`).
  - **The delivery table** is generated from build records, after the owner saw three stretches of the joined clip
    return to the original.
- **The bad: one-clip thresholds and keep-outs.**
  - **`PART_LEAST_PX`,** reasoned on another clip, silently emptied a small face on about half the frames of a load the
    owner had accepted the day before (`docs/wiki/research/postmortems/2026-10-11_span_fun-lid-marks-one-shot.md:152`).
  - **The mouth gate** (`MOUTH_CHECK` 0.8) dropped correct mouths on 40 of 51 frames
    (`docs/wiki/decisions/2026-10-15.md:320-330`).
  - **The `held` keep-out,** made for one egg in one shot, became a default on every load of that subject. Her apron
    read as "held".
  - **The largest-piece rule** spread from one subject's loads to another's, dropping pieces on 180 of 362 frames of
    `lead_fp_0613c` (`docs/wiki/decisions/2026-10-16_to_2026-10-17.md:305-315`).

On 2026-10-16 the owner and the lead ruled "the simplification wins": every load composites its whole region. The
largest-piece rule, the "only what changed" composite, and the `held`, `hands`, `clothes`, `hair_band` and `stay`
keep-outs all went in one day (`docs/wiki/decisions/2026-10-16_to_2026-10-17.md:74-79`, `:317-329`). A later record
measured the fault on every earlier version of the kitchen shot, and none after
(`docs/wiki/research/postmortems/2026-10-18_mom_kitchen_flicker.md:23-28`).

### 3. Structure holds, appearance keeps moving (2026-10-17 to 2026-10-27)

The checks that lasted from here on caught structure:
- **24 fps end to end,** after hard square edges traced to a 23.976 fps cut
  (`docs/wiki/research/postmortems/2026-10-18_24fps_joined_cut.md:89-95`);
- **one laid list** (`laid.jsonl`) with a per-frame gap check, and gaps counted above every clip shown to the owner;
- **a blocking flag on a visible face left unmasked** in a window (`dropped_frame_has_visible_face`);
- **frames picked where something changes** (`bench/pick_frames.py`), after evenly spaced sheets missed the frames
  where a subject leaves the picture.

The appearance rules kept reversing:
- **The accessory give-back** (keeping a subject's sunglasses as filmed):
  - added 2026-10-14;
  - killed 2026-10-16, when earrings and a necklace leaked as moving patches;
  - reinstated 2026-10-23, placed by the rendered face;
  - pinned to named loads 2026-10-27, after double sunglasses on `lead_fp_3967c`.
- **The context-frame rule** changed three times in five days
  (`docs/wiki/decisions.md`, 2026-10-25, the three-path rule).

Two failures of the flywheel itself show up here:
- **A script looped overrides.** It auto-overrode every blocking flag, which is how dark hair reached several renders.
  The rule that followed: an override is a person's act after a look, never a loop in a script
  (`docs/wiki/decisions/2026-10-21.md:60-74`).
- **A self-written reason beat the recipe.** The third subject's beard margin of 64 to 96 painted the drawing's
  outlines onto his face. It was reverted to 16, then went through again on `5114d`, because the recipe gate accepted
  any reason a session wrote for itself. The recipe now lives in `plan.json`, and the job step refuses a differing
  value (`docs/wiki/decisions.md`, 2026-10-26).

### 4. Readers before the owner (2026-10-28 to 2026-10-31)

On 2026-10-28 the owner asked why faults were not checked before a render reached the review page. The answer was a
rule: every landed render is read, before it is paged, by a reader who did not make it. The reading names what it
sees, with frame numbers and figures, and gives no verdict and no hold (`docs/wiki/decisions.md`, 2026-10-28 and
2026-10-29).

The same rule had been the other way round before, "no session looks at a render before the owner does"
(`docs/wiki/decisions/2026-10-22.md:10-12`). Followed to the letter, it had sent a render with visible dark hair to the
page unread.

By the end the owner still rejected paged renders, so the flywheel had slowed the looks, not ended them. No new checks
were added after 2026-10-27; the owner's answers held at tens a day.

## What lasted and what died

**Lasted:** structural and provenance checks with a hard signature.
- **Frames in their place:** `at` alignment, 24 fps.
- **Inputs:** made fresh, never reused by a key; reference immutability.
- **What is laid:** one laid list, with `sha256` and gap counts.
- **Flags tied to a pixel measure,** such as `dropped_frame_has_visible_face` and `keep_out_inside_own_track`.
- **Pages that put one kind of question in one place:** who-is-who, the review queue, stacked clips with frame
  numbers, frames picked by change.

**Died:** thresholds and defaults made for one clip, and fixes that spread.
- **Thresholds:** `PART_LEAST_PX` as a hard floor, `MOUTH_CHECK` 0.8, `BARELY_CHANGED` 0.3, the cranium score floor.
- **Margins:** the beard margin.
- **Keep-outs and rules made for one load:** `held`, the largest-piece rule, the strangers detections for one load
  (`426638a6`, which carved another subject's hair on another shot until `ba86fbe9`).
- **Refusals people had to work around:** the look as a refusal; a body-model lag check that failed a load with
  exit 1.

The gate's own code says it: of its thresholds, almost none takes its value from the clip in hand yet
(`bench/capture_masked_run.py:229-233`, which labels each one THIS CLIP, CARRIED or REASONED).

## What it taught

1. **Log every answer, append-only, citing what was shown.** Everything above could be counted only because the owner's
   clicks went into one file that nothing edits. The flywheel needs that record to know what it caught.
2. **Turn a catch into data first, and a rule later.** The first answer to a fault was usually a new rule, flag or
   refusal, and the first postmortem lists that reflex among the day's wrong beliefs
   (`docs/wiki/research/postmortems/2026-10-10_span_fun-three-people.md:190`). The figures that lasted were printed data
   with no verdict. "A figure is a measurement and not a judgement" (`docs/wiki/decisions/2026-10-11.md:352`).
3. **Structure becomes a check; appearance becomes a recipe and the owner's eye.** Frames, rates, hashes and lists have
   a hard signature, and their checks held. Mouths, hair, blur and likeness did not; they became plan words and
   recipes, judged on playback.
4. **A threshold says where it came from and what retires it.** THIS CLIP, CARRIED or REASONED is already written
   beside each gate constant. What is missing is the other half: what would show the value wrong, and what retires it.
   The rules that died were the ones nobody asked that of.
5. **A gate is passed only by a person's word.** Both failures of the flywheel were bypasses: a script that looped
   overrides, and a reason a session wrote for itself. "A rule easily worked around is friction, not a rule, and goes"
   (`docs/wiki/decisions/2026-10-12.md:319-320`). The other half: a rule worth keeping cannot be worked around except
   by the owner.
6. **"One rule in, one rule out" needs a mechanism.** It was written down on 2026-10-10 as "a habit with no mechanism
   behind it" (`docs/wiki/decisions/2026-10-10_part1.md:54-58`), and the rule count grew until a single day removed six.
7. **Readers who did not make the thing.** The best early catches came from a second session reading real files without
   the author's conclusion. By the end that was a rule.

## What it suggests next

These are suggestions, not decisions.
- **A rule ledger in the lane's record.** `data/lane.sqlite` could hold each rule, flag or check as a row: the answer
  line it was born from, what it covers, where its value came from, what would retire it, and the line that killed it.
  That gives the "one in, one out" habit its mechanism. It also makes "which rules are one-clip" a query, and it turns
  the leftovers question (what does this knob cover, does that still hold, what retires it) into a column.
- **`fun2` as the test of what lasted.** Same footage, a new lead. Structural checks should carry unchanged. Any rule
  that has to change for `fun2` is a rule that was made for `fun`.

## Not confirmed

- **Who typed two answer lines:** lines 1698 and 1702 of `data/fun/owner_answers.jsonl` are marked as the owner's but
  read as session prose.
- **How the checks-per-day count was made:** curated by judgement, and dated by when each file was added, not when its
  row was written.
- **Coverage before 2026-10-11:** looks before then went to a hosted board and are not in the table.
- **What was not opened:** no render, still or commit diff was opened for this record.
