---
mode: span
scope: tau-and-morton
date: 2026-08-14
range: 6e85e48..HEAD (2026-08-14 15:00–17:45), renders 15:55–17:39
summary: The tau/morton sweep produced one apparent finding — morton dropping a reference feature — which was judged from first frames, sent upstream, and failed to replicate at a second seed within the hour; the sweep's real result is that on reference workloads these knobs are nearly free and tau is barely a lever.
artifacts:
  - docs/bench_plan.md
  - docs/evidence.md
  - docs/open_experiments.md
  - docs/SOLATTN.md
  - h3_rules.py
  - resolution.py
  - bench/bench_e2e_h3.py
  - bench/check_retraction_consumers.py
  - workflows/h3_config.py
  - workflows/build_workflows.py
  - workflows/h3_probe_sol_on_refs.json
  - workflows/h3_probe_sol_on_all_refs.json
  - workflows/h3_probe_sol_on_i2v.json
  - vendor/UPSTREAM.md
  - internal/postmortems/2026-08-14_span_cuda-sol-migration.md
  - 6e85e48
  - a02a519
  - 874158e
  - bd392c2
  - 3b86b21
supersedes:
---

# The tau sweep, and a finding that did not survive the hour

Afternoon and evening of 2026-08-14, continuing from
`2026-08-14_span_cuda-sol-migration.md`. Single session, owner intermittently
away, ~2 hours of card time across eleven renders.

## 1. What went well

**Pre-registration caught an error the same way it is supposed to.** The v2
`sink_q` prediction was written before the run: KV `(0, 263)`, query start 132.
Sequence and KV matched to the digit; the query start was **260**. Because
sequence and KV were right, exactly one term was wrong and there was no
hunting — the prediction localised the error. The cause was carrying a t2v
identity (`audio_start == text_len`) into a reference graph, where the layout is
`[text][refs][audio][video]`.

**Running the length-invariant form made a dead question cheap.** The first
attempt at the reference sink verification killed the server during DiT
staging. Rather than repeat it, the query start was shown to be `text_len // 64`
and therefore independent of clip length, so a 39-frame run tested the
discriminating number at a fraction of the VRAM. Recorded in `docs/bench_plan.md`
*before* running.

**A retraction check now exists and caught its author twice.** It fired on its
own landing commit — the `docs/checks.md` entry documenting it introduced a
tracked phrase into an unlisted file. Then, after being fixed to normalise
whitespace and Python string-concatenation seams, it found a live consumer in
`CLAUDE.md` that the naive matcher had missed all along (`874158e`).

**Coverage that did not exist was built.** Sol had no graph for reference
video, reference audio, or input images — the shipped reference graphs omit the
node from their API form entirely, so it could not be patched in at submit time.
`bd392c2` added `h3_probe_sol_on_all_refs` and `h3_probe_sol_on_i2v`.

## 2. What did not go well

**The morton finding was judged from first frames and does not replicate.**
This is the important one.

At seed 730451892, the morton arm's first frame lacked the reference's
snow-gullied peak while four other arms' first frames had it. That 4-vs-1
pattern was recorded as a probable morton effect (`3b86b21`), hedged as n=1,
and **sent upstream to kijai**. Within the hour, the seed-424242 control ran and
the owner observed that the morton clip **has the mountains and streams**. A
later frame of the seed-1 morton clip also carries the man's face, the alpine
lake and its reflection.

So the observation was framing and trajectory, not morton.

`docs/SOLATTN.md` states in plain text that a grid of stills at sampled
shot-times cannot catch this class of thing and that the failure mode is
temporal. The comparison used **one frame per clip**. *Structural form: the
repo's own documented method failure was performed by the person who had cited
it in three commit messages that day.*

**The control was designed so it could fail to answer.** The pre-registered
test was "does morton lose the peak again". Nothing required the peak to be
*in frame*. At seed 424242 the first frames of both arms are close meadow shots
with no distant view at all, so the still-based version of the test was
unanswerable — and only the owner watching the video resolved it.

**A digit was transposed and propagated.** The fused-qkv int32 crossing is
99,864 (`2**31 // 21504`). It was written as 99,846 in messages, inherited by a
subagent from that prompt, and reached `vendor/UPSTREAM.md`, the file staged to
go to kijai. Caught by recomputing rather than re-reading (`a02a519`). The two
values are equally plausible on sight and support the same conclusion.

**A subagent repeated two errors across three invocations** — the transposed
digit, and asserting the tau run "was never launched" while it was rendering.
Both came from stale visibility asserted as fact. Its output stopped being
relayed.

**`pkill -f` matched the shell running it**, exit 144, killing the intended
target and also the heredoc that was supposed to write the replacement script.

## 3. Deviations from the plan

| Planned | Shipped | Verdict |
|---|---|---|
| tau 1.0 / 1.15 / 1.3, dropping ≥1.5 | 1.0 / 1.3 / 2.0 | Amended before running, on kijai's tau-2.0 datapoint |
| t2v as the artifact-sensitive control | Cancelled mid-run | Correct: owner does not render t2v, and a knob validated on unrendered content is not validated |
| centroid/morton on t2v, sharing a control | Moved to the reference graph | Same reason |
| 362 treated as illegal | Withdrawn | The reference pipeline's validator is not a training boundary |
| morton finding | Recorded, sent upstream, then failed to replicate | Reported too early |
| Sol coverage for video/audio/input-image refs | Two graphs added, both rendered | As planned |

## 4. Escapes (tests)

**Nothing checks that a quality judgement used an appropriate instrument.** The
morton error passed every mechanical gate in the repo — all 17 checks green,
graphs validated, arms correctly isolated, one variable per run, same seed. The
defect was in *how the output was looked at*, which nothing covers. This is
`docs/open_experiments.md` #14 restated as an escape rather than a gap.

**The 362 warning added at 14:02 was a check going red while the state was
correct**, shipped hours after that standard was quoted. Rewritten, not removed.

**`check_retraction_consumers.py` was green while passing over a live
consumer**, because prose wraps at 79 columns and its matcher was a raw
substring test. Two seams, both now closed.

## 5. Forward items — where we left off

**Everything queued has finished. Card is free, server up, tree clean, all 17
checks pass.**

1. **Correct the message to kijai.** It says morton "clearly left out a detail
   from the painting" and "doesnt seem to help but it does at least here kinda
   hurt". That does not replicate. He is asleep; the correction should land
   before he acts on it. Done when sent.

2. **Watch the eleven clips.** `/mnt/hub/ai/img/output/Video/tau_run/`. Nothing
   in this postmortem is a quality verdict, because none was earned — every
   judgement so far came from stills, which is the failure above. Arms:
   `refs-1.0/1.3/2.0`, `refs-1.3-centroid_off`, `refs-1.3-morton_on`,
   `seed2-refs-1.3`, `seed2-refs-1.3-morton_on`, `all_refs-1.3`, `i2v-1.3`,
   plus `t2v-1.3` from before the directive.

3. **`all_refs-1.3` and `i2v-1.3` have never been looked at.** Both completed
   (17:19 and 17:24). `all_refs` is the heavy case at 157,727 tokens —
   references 34.3%, audio refs 8.8%. It is the first Sol render at full
   reference load and nobody has seen it.

4. **Kijai's two questions are still open on the axis he asked about.** Speed:
   morton is free here (−0.9%), `centroid_tail` costs 0.5%. Quality: unanswered.
   `centroid_off` retained the mountain detail at seed 1, which is the only
   quality datapoint either question has, and it rests on the same discredited
   first-frame method.

5. **The tau result is the sweep's real finding and it survived.** On a
   reference workload the whole 1.0→2.0 range spans 9.9% of render time
   (462.0 s → 420.4 s), against `centroid_tail`'s 2.5% on t2v. Reference rows
   are pinned exact at any tau, so there is less to sparsify. **Tau is barely a
   speed lever on the workload actually rendered.**

6. **Fix the still-based judgement, or stop making quality claims.** #14 is now
   load-bearing rather than aspirational: it blocks both of kijai's questions,
   the tau quality question, and it is what let a wrong result reach upstream.
   The cheapest honest interim rule is that no quality claim leaves this repo
   from stills.

7. **Carried from the morning postmortem, unmoved:** items 2, 5, 6, 7, 8 of
   2026-08-13, plus the real-activation capture (`h3_capture.py` fixed but never
   run) and the four `vendor/UPSTREAM.md` items, of which the token-counter
   stride is drafted and unsent.

8. **Environment, unchanged and not in git:** `comfy-kitchen
   0.2.31+sol.c04ef20` built for sm_89 only; Sol node vendored at
   `d856ba83557d18fb` (v2); ComfyUI writes to `/mnt/hub/ai/img/output`, not
   `~/ComfyUI/output`.
