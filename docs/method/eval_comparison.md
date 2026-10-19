# Comparing renders: the A/B process and the stacking tool

**A comparison meant to be quoted is rendered at matched seeds, blinded, scored
before unblinding, and recorded as a preference over distributions**: section 3,
"The standard A/B process". Before that, choose a scene that can show the
difference ("What a scene can show"). Sections 1 and 2 are the stacking tool,
[`bench/stack_eval_clips.py`](../../bench/stack_eval_clips.py): side-by-side or
top-to-bottom comparison videos with synchronized playback and metadata overlays.

---


## What a scene can show

**A perceptual claim states the scene it was judged on, because a scene can be
incapable of showing the defect being judged.** Blinding controls who knows which
arm; it does nothing about whether the content can express the difference. Three
findings decide which scene to judge on:

| finding | what to do with it | status |
|---|---|---|
| within a clip, artifact severity in a distilled arm tracks frame-to-frame change (correlation **+0.676**, cuts masked; the high-change quartile carries 1.18x the low one) | expect busier frames inside a shot to be worse | measured 2026-08-28, unretracted |
| across clips and shots, frame-to-frame change does **not** predict severity; it failed twice | never choose or judge a scene by it | refuted twice, below |
| what breaks a distilled arm is the content the prompt **demands**: how much fine structure, and how many setups in one clip (three setups in fifteen seconds broke, two did not) | judge a distillation on a scene that asks for the thing that breaks; read demand from the prompt, before rendering | one observer, three clips, one seed: a candidate |

### Frame-to-frame change, and why it is not motion

`bench/measure_clip_delta.py` computes mean |frame[n] - frame[n-1]|: change,
whatever causes it. A cut, a light switching on and a camera whip all score high
with nothing moving. Call it **delta**, not motion. On shipped renders the scenes
differ enormously in it:

| scene | median delta | frames above 0.02 |
|---|---|---|
| market (t2v, wide, crowd) | 0.0142 | **47.6%** |
| dialogue, PDD 4-step | 0.0082 | 6.4% |
| dialogue, base 16-step | 0.0047 | **0.0%** |

**Across shots it runs opposite to severity.** On `text_to_video_pdd_4step_00007`
(cuts located by delta peak), the owner identifies shot 3 as where ghosting and
melting appear and shot 2 as only slightly ghosted:

| shot | mean delta | spatial detail | change concentration |
|---|---|---|---|
| 1 | 0.0253 | 0.0632 | 32.4% |
| 2 — crate lift | 0.0252 | 0.0507 | 38.2% |
| **3 — crowd and fruit** | **0.0058** | **0.0666** | 64.6% |

The worst shot has 0.23x the delta and 1.32x the spatial detail of the one called
nearly clean: the camera is locked and the stalls static, so differencing reads a
quiet shot where the eye reads a busy one.

### Demand, not result

**The quantity is the fine structure the prompt asks to be resolved**, readable
from the text before anything renders (the owner's correction of the first
statement, which measured detail on the output). Detail measured on the output is
the wrong quantity twice: it is downstream of the cause, and ghosted fruit reads as
low detail, so it cannot tell a plain scene from a destroyed one. A wide market
shot asks for each orange in one or two latent cells, where a stairwell closeup
gives a face hundreds, and fewer steps means less refinement at exactly that scale.

**The shot-count ablation, scored 2026-08-28** (the first pass read only one arm
and was corrected within the hour; both two-shot arms are good):

| arm | prompted shots | delta | detail (output) | verdict |
|---|---|---|---|---|
| `00007` original | 3 | 0.0149 | 0.0635 | **bad** — ghosting, melting |
| `shots13` | 2 | 0.0122 | 0.0640 | great |
| `shots12` | 2 | **0.0209** | 0.0580 | great, the box did not ghost |

- **Delta is refuted** without scoring finesse: `shots12` has the highest delta and
  is fine.
- **Output detail gives no story**: `shots12` lowest and good, `shots13` highest and
  good, the bad clip between them.
- **Prompted shot count is the only clean split.** `shots12` rendered an unprompted
  extra cut (peaks at frames 182 and 219, where only 180 was asked for) and was still
  fine, so what breaks the original is not the number of segments on screen; total
  content demanded survives that.

**The pre-registered demand pairs are the experiment.**
`h3_text_to_video_{aisle,sortline}_{short,long}` vary how much one scene asks at
fixed shot count, holding subjects, actions, camera moves, dialogue and cut times,
so they cannot move delta and content together.

| arms | demand predicts | the account it beats |
|---|---|---|
| `h3_text_to_video_{aisle,sortline}_{short,long}` | long worse than short in **both** scenes, damage on the elaborated surfaces | length-of-conditioning: no consistent direction |
| market shot-count ablation | `shots12` best (lowest total demand), `shots13` worst | delta: `shots12` worst, being the all-high-delta arm |

Arms, outcomes and the duration control that confounded one are in
[`2026-08-28_scene_complexity.md`](../research/pdd/2026-08-28_scene_complexity.md).

**The dialogue scenes are audio instruments, not quality probes**: eight lines,
seven speaker changes, two registers, and almost no delta. The distilled dialogue
arm shows more frame-to-frame change than the undistilled one on the same prompt,
which is either motion the base did not generate or the artifact registering as
motion ([`audio_under_pdd.md`](../research/pdd/audio_under_pdd.md)).

## A stress scene is not a typical scene, and neither carries a default alone

**A scene one candidate passes and the others fail earns a panel, not a flip of
the default.** One scene at one seed says a candidate can matter, not that it
matters in general.

**The panel.** Four or five bank prompts chosen for different hard moments (a
person entering frame, hands with objects, on-screen text, fast motion, a static
shot), each at one seed, plain against the candidate, scored on one question at a
time (for the reorder: does anything morph, yes or no). A default moves when the
panel agrees, and the record says which scenes and what was asked.

**Render a bank prompt only at the length its entry declares.** The runner refuses
otherwise. The noodle bar, which carried this section's first example, is a
107-frame prompt that every arm rendered at 345: past its script the model
improvises, and the morph started where the script ended. It was an
out-of-distribution render, not a stress scene, and it is retired at that length
(`bench/results/2026-09-18_off_length_prompts.md`; the batch is
`bench/results/2026-09-17_sol_options_noodlebar_batch.md`). In it, every arm with
plain token order morphed and the two with Sol's `3d` token reorder did not: that
shape, one candidate passing where the rest fail, is what a stress scene is for.

**What makes a scene fragile, on any attention stack** (`prompt_bank/t2va_noodle_bar.txt`):

- it scripts a few seconds of action for a longer clip, and nothing injects the
  duration into the model ([`prompting.md`](../prompting/prompting.md) section 15.2,
  and 3.3 for the frame grid), so the model fills the rest with undeclared people
  and motion;
- its hardest moment is structural: a slow zoom-out keeps opening the bottom of the
  frame while a person arrives from off frame, which a diffusion model fades in
  rather than walks in.

A tighter prompt says where the entering person starts, that the rest of the street
is empty, and either fills the timeline with beats or shortens the clip.

This pack has made the one-scene mistake before: the market scene at seed 730451892
carried the kitchen-versus-sage question for two days on a missing hand that did not
reproduce on five other scenes, and whose prompt sentence named no one doing the
action (`bench/results/2026-09-15_block49_repro_batch.md`).

**Two instrument lessons from the same evening.**

- **One character in a prompt is a different sample.** A trailing newline on a bank
  prompt made two renders of "the same scene and seed" as far apart as two
  unrelated arms. Rows of a stack share the prompt byte for byte; the `prompt` in
  each clip's first-frame PNG is how to check (`bench/diff_clip_graphs.py` reads it;
  clips from before 2026-09-23 also carry it in the mp4's own tags). Renders on this
  stack repeat bit for bit, so a PSNR of infinity between two clips is an identity
  test, and anything less is a different input.
- **An option that improves a capture metric does not thereby fix what the eye
  sees.** Rotation and balancing lower the INT8 error; none touched this artifact,
  which comes from which tokens share a block, and the reorder is the only lever
  that changes that.


## 1. Automatic Layout Optimization

The tool automatically detects canvas aspect ratio ($W/H$) to pick the optimal stacking layout:

| Canvas Geometry | Aspect Ratio ($W/H$) | Default Layout | Resolution Example | Display Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Widescreen / Landscape** | $\ge 1.2$ (16:9, 4:3, 3:2, 21:9) | **Vertical (Top / Bottom)** | $1344\times768 \to 1344\times1536$ | Side-by-side on wide clips creates unwieldy $2688\text{px}+$ ultrawide videos. Vertical stacking fits standard 1440p/4K displays cleanly. |
| **Portrait / Vertical** | $\le 0.9$ (9:16, 3:4) | **Horizontal (Side-by-Side)** | $768\times1344 \to 1536\times1344$ | Side-by-side combines two tall portrait videos into a balanced landscape 16:9/4:3 viewing frame. |
| **Square** | $0.9 < W/H < 1.2$ (1:1) | **Horizontal (Side-by-Side)** | $768\times768 \to 1536\times768$ | Side-by-side fits standard 16:9 desktop monitors. |

---

## 2. Common Usage Commands

### Standard Comparison (Auto-Layout):
```bash
python bench/stack_eval_clips.py clip1.mp4 clip2.mp4 -o comparison.mp4
```

### Labeled Comparison (e.g. Arm A vs. Arm B):
```bash
python bench/stack_eval_clips.py \
    clip1.mp4 clip2.mp4 \
    --label1 "Sol (tau 1.0)" \
    --label2 "sage dense" \
    -o comparison_labeled.mp4
```

### Force Layout Override:
```bash
# Force side-by-side regardless of aspect:
python bench/stack_eval_clips.py clip1.mp4 clip2.mp4 --layout horizontal

# Force top-to-bottom regardless of aspect:
python bench/stack_eval_clips.py clip1.mp4 clip2.mp4 --layout vertical
```

### Blind Evaluation Workflow:
Randomizes the assignment of Clip 1 and Clip 2, stamps anonymous overlays, and writes a sealed keyfile:
```bash
python bench/stack_eval_clips.py \
    arm_a.mp4 arm_b.mp4 \
    --blind \
    --keyfile internal/blind_key_test.json \
    -o eval_blind_comparison.mp4
```

---

## 3. The standard A/B process (since 2026-08-20)

`stack_eval_clips.py` is the presentation layer. The process around it, for
any comparison that is meant to be quoted:

1. **Render with `bench/run_graph_arms.py`**, arms alternating, `--runs N
   --seed S` so every arm sees the same seed per run index, `--warmup` on the
   first arm. Every row records its graph sha, patches, seed, `prompt_id` and
   substrate (including the power limit). One render per arm is two samples,
   not a comparison (root `AGENTS.md`, "No render that does not decide something"), so N is the number
   of seeds the claim needs, and for a perceptual claim that is many.
   **A manifest names a graph, not a render.** The graph is whatever the
   generator last built, so re-running an old manifest renders today's
   defaults; the row's graph sha is what actually ran. *Since 2026-09-15
   (`28d8ee5`)* every manifest that points at a default graph renders the
   kitchen-backend chain, and re-running one written before then is a
   different experiment from its record, not a repeat of it.
2. **Blind with `bench/blind_batch.py`**: neutral `clip_NN.mp4` copies under
   `Video/blind/<session>/`, a MANIFEST with row indices only, and a sealed
   key in `internal/blind_keys/<session>.json` (gitignored). For a two-arm
   session add `--pairs A,B`: the i-th clip of each arm, matched by run index,
   stacked by this tool's layout rule as `pair_NN.mp4` with "Clip 1" / "Clip 2"
   in a per-pair random order, which the key also records. The stack file
   carries no audio, but the scoring page plays the audio of the half the
   judge selects from that half's own single, in sync, and exports which
   halves were heard per pair (`pair_audio_heard`; since 2026-09-05, before
   which every pair was judged deaf). The singles carry the audio, copied
   from the muxed `-audio.mp4` the combine node wrote, and a single that
   would be silent refuses the batch unless `--silent-ok` says the graph
   saves no audio. Rows flagged
   `suspect_cache_hit` or `error`, or whose clip cannot be found, refuse the
   whole batch. The output root is `--output-root`, else `H3_COMFY_OUTPUT`,
   else the `--output-directory` the live server was launched with, read
   from its command line by `bench/_paths.py::comfy_output`; with none of
   the three it refuses rather than resolving against the local directory.
3. **Score before unblinding**, into a sheet written in advance (rubric first,
   then rows), one pass in the shuffled order. Only then open the key and
   write the per-arm aggregates to `bench/results/<date>_<session>_verdict.json`.
   A preference is a preference over distributions, stated that way.

For a single pair outside a session -- two clips that already exist -- the
`--blind --keyfile` form in section 2 is still right, with `-o` set to a
neutral name, since the default output name carries both input stems.

### What a matched seed does not match

Two riders on the different-sample rule, both established 2026-08-28. Neither
weakens step 1; both change what "matched" is allowed to mean.

**A matched seed does not match the audio stream when the arms differ in canvas
or length.** Noise for the video and audio streams is drawn from one seeded
generator consumed in order (`comfy/sample.py`), so the audio noise depends on
the video latent's element count. Two arms at one seed that differ in
resolution or frame count therefore differ in their audio noise as well as in
the knob under test. *Read from source, not measured.* Consequence for this
process: an arm pair that varies canvas or length is not a controlled
comparison of anything audible, and a session mixing canvases cannot pool its
audio judgements.

**The first run after a state change is not the arm's settled behaviour.** A
render has a warm-up transient; the first run after a configuration change
differs from what that configuration settles on, and it hits both arms of a
pair equally -- so a single matched pair reads exactly like a regression. This
is what `--warmup` on the first arm in step 1 is for, and it is the reason the
flag is not optional for a quoted comparison. *Reported by a peer session from
eight settled runs on the card, four per arm; not verified here.*

A corollary worth stating because it has caught people: **compare decoded
pixels, never file bytes.** The containers embed metadata -- the mp4 case was
already recorded here, and `SaveImage` embeds the prompt JSON the same way, so
two byte-different files can be pixel-identical.


### Scoring: the page and the joiner

Step 3's "sheet written in advance" is now a page written into the batch. The
rule it encoded is unchanged -- questions fixed before the first clip plays,
one pass in the shuffled order, key opened only afterwards -- but a markdown
table loses a row the moment the judge scrolls, so `bench/blind_batch.py`
writes `score.html` beside the clips instead.

`bench/blind_score_app.py` generates that page. It is self-contained -- inline
CSS and JS, no external request, relative video sources -- so it works opened
as a local file from the share. It reads the batch's MANIFEST, the rubric file
and the brief file, and **nothing else**: not the sealed key, and not the
JSONL, whose rows carry the arm label in a field.

- **Pairs are the primary view** and the page opens on them. A pair asks what
  differs between the two halves and which way it goes, as free text, plus
  quick tags clicked per half and one coarse verdict. There is no numeric
  scale on a pair: a stack shows two different samples, and the thing a judge
  can report about them is the difference, not a rating of each.
- Singles are secondary. They carry the audio, which the stacks do not, and
  anything wrong with one clip on its own.
- The rubric is a JSON file -- `bench/rubrics/default.json`, or
  `bench/rubrics/scales.json` for the 1-5 form. `--brief-file` puts the
  session's brief at the top of the page, collapsed.
- Answers are held in the browser per session, so a reload loses nothing.
  "Export scores" writes `scores_<session>.json` and prints the same JSON into
  a textarea, because a page opened as a local file cannot always start a
  download.

### What the judge records, and what to ignore (written with the owner, 2026-09-05)

The owner asked, across three sessions, what actually matters on the page.
This is the answer, and the brief of every future session should point here
rather than restate it.

**The two halves of a pair are two samples, not one clip rendered two
ways.** The trajectory diverges at frame zero under any change to the
sampler's inputs (root `AGENTS.md`, "A rendered clip cannot A/B a numerical
change"), so two arms at one seed differ in composition, blocking, faces and
line delivery before either arm's quality is in play. **Composition
differences count, framed as prompt adherence** (owner's decision,
2026-09-05 evening, after scoring three sessions): a half whose scene makes
less sense for the prompt (the orchestra behind the singer, the suspect
behind the chaser, a menu across a wall) has adhered less, and that is a
loss for its arm, recorded in the text so the join can separate adherence
from rendering quality. Until that evening this section said to ignore
composition as sample divergence; the three 2026-09-05 verdict records were
first read that way and their notes say so. Rendering quality is judged
beside adherence: texture and skin, text legibility, motion coherence,
flicker, brightness, audio clarity and level, sync. The question to answer
is "which of these two would I rather have received for this prompt, and
why", and the text says which kind of loss it was. One pair per scene is one sample; the read is across
the session's scenes, and a rung that loses on four scenes for the same
defect is a finding where a rung that loses one pair on composition is
noise. The text field is how the join tells the two apart.

**Pairs, in order of importance.**

1. **Hear.** Under each stack a row reads "Hear: Clip 1 / Clip 2 / neither"
   and starts on neither; the stack itself is silent. Click a half and its
   own single's audio plays in sync. Hear both before a verdict. Which
   halves were heard is exported and the join carries it per pair.
2. **What differs, and which way.** The field the verdict record is built
   from. One or two plain sentences naming the defect and the half it is
   on: "Clip 2's sign text melts by the end, Clip 1 holds it; skin on 2
   looks waxy; otherwise the same." Not a description of the scene, which
   both halves share by construction.
3. **Verdict**, one of four. *Same* means the judge looked and would not
   care which half they got; a pixel difference is assumed and is not what
   the word means. *Can't tell* means no judgement could be formed: the
   halves differ but neither is better, or the pair gave nothing to hold on
   to. The tally counts them apart because they say opposite things: a run
   of *same* says the arm is free, a run of *can't tell* says the test was
   uninformative and points at the scene or the arm design.
4. **Per-half quick tags** are optional shorthand. *good*, *not good* and
   *off* mean something on a half. *same* and *can't tell* on a half mean
   nothing: the pair tab reuses the singles' option list, which is a rubric
   defect recorded in `docs/state/next_steps.md`; the join ignores those two
   tags on a half, and a judge should put them in the verdict only.

**Singles.** Anything wrong with one clip on its own, and **always the
audio**, because the stacks cannot carry it: quieter, muffled, desynced,
missing. Also whatever would be noticed without a comparison: melted text, a
collapsing face, a hard cut, a frozen figure, flicker. *Reject outright* is
for garbage only.

**What to watch for, per lane.** A session's brief names its own list; the
standing ones are the defects the owner named blind on the shipped PDD8 rung
on 2026-09-03 (brightness, compressed skin texture, melted on-screen text,
shaky framing) for any PDD session, and for any Sol session the things a
routing policy can break: motion smearing, background figures going static,
texture in the dark, on-screen text.

**Then export**, save into the session's folder as `scores_<session>.json`
(paste the textarea into a file if the download is refused), and say it is
there; `bench/score_session.py` does the rest.

### The lean pairs-only session (2026-09-29, owner)

The page above asks for a free-text difference, per-half tags and a verdict on
every pair, then asks again about each single. The owner found that slow and
unclear, and the two per-half `same` and `can't tell` tags meaningless on their
own (they are, as item 4 above says). For a comparison whose question is "does
this arm change what I would rather watch", use the lean form:

- `bench/blind_score_app.py --pairs-only --pair-rubric bench/rubrics/pair_preference.json`
  scores the pairs only. The singles stay in the batch as each half's audio and
  are not shown to the judge; the export carries `pairs_only`, and
  `bench/score_session.py` then does not ask for them.
- One required question per pair: **which would you rather have received for
  this prompt**, `Clip 1 better`, `Clip 2 better` or `same` (you would not care
  which you got). It is the standard forced-choice preference with a tie option,
  over picture and sound together. `can't tell` is gone: it separated "looked and
  did not care" from "no judgement possible" and the judge could not tell them
  apart either. A run of `same` is the arm being free.
- Optional: an audio preference (`didn't listen` is an answer), one line on why,
  and a flag for something broken in either clip. No per-half tags, no scales.
- The joiner tallies only the `verdict` question; `audio` is kept per pair and
  per contest (`other_choices`), `broken` per pair.
- Build it with one `--pairs` per contest, then regenerate the page with the flags
  above (`blind_batch.py` writes the default page). When the arms have one row per
  scene, the labels are per scene and so are the contests: read the aggregate over
  scenes by arm from `by_pair`. The blinding, the sealed key and the
  score-before-unblinding rule are unchanged, and so is the different-sample rule:
  a preference over distributions, one pair per scene is one sample.
- Not built: an identical-clip control pair (the same clip on both sides, which
  should come back `same`), the usual way to see how often a judge picks a side
  by position alone. It needs a duplicate of an arm's output on the share.

### Predictions before the verdict: the PDD bake pair as the worked example

A session's brief should say, before anyone scores, what the arm changes,
what should therefore move, what should not, and how the result reads either
way. The judge can then be wrong in a named way. The bake pair
(`pdd_bake_2026-09-05`, manifest `bench/pdd_bake_arms.json`) is the first
session written this way, with the bake lane's predictions folded in on
2026-09-05:

- **What the arm changes, once.** Every int8 backbone linear in all fifty
  blocks is quantised once from the release weights plus the PDD delta,
  instead of dequantised, patched and requantised with stochastic rounding
  on every cast. Nothing else moved: adaln patch, head bank, refiner patch,
  schedule, seed, prompt, sage alone. So the merged half equals the baked
  half plus a small random weight perturbation on every backbone linear,
  about the size of the checkpoint's own quantisation error
  (`docs/research/merge_requantisation.md`;
  `data/results/2026-08-31_merge_realisation_pdd.json`), spread uniformly.
- **Should move if the hypothesis holds**: the defects that look like added
  quantisation noise. Compressed or waxy skin, melted or ringing on-screen
  text, a brightness or contrast shift, lost fine detail in hair, fabric and
  grain, banding in smooth gradients. Improved on the baked half and the
  hypothesis holds; identical and the merge noise is below what one sample
  shows.
- **Should NOT move**: anything the eight-step partition owns, which the
  bake does not touch. Motion smear on fast action, flicker at block
  boundaries, shaky framing, temporal coherence ([`evidence.md`](evidence.md), "PDD quality
  is governed by the sigma schedule's coarseness"). If those clear on the
  baked half, suspect a sample difference, not the bake.
- **Audio is in the pair.** The audio tokens run through the same backbone
  linears, so merge noise perturbs the audio stream exactly as the video.
  Listen for high-frequency dullness or hiss, smeared transients, sibilance;
  speech content and sync should not change. Level is a measurement, not a
  judgement: `bench/measure_clip_loudness.py` on the outputs record, beside
  the verdict. If the ladder's "a bit quieter" on PDD8 was merge noise, the
  baked clip's level moves toward the sage floor's; if not, the level
  belongs to the heads or the schedule (`docs/research/pdd/audio_under_pdd.md`).
- **How "same" reads.** Merged-versus-baked judged *same* at one seed is a
  success on hygiene regardless (exact weights, no stochastic dependence,
  one artifact, no per-cast requantise), a small success on cost, and a
  null on the hypothesis only in the weak sense: the merge noise's visible
  effect is not larger than one sample's variation. The stronger reading is
  the triangle: baked against the sage floor at the same seed, beside
  merged against sage, which the 2026-09-03 ladder already judged (merged
  lost every scene). Baked reading same-as-sage where merged read worse is
  the positive result even when the direct pair reads *can't tell*; baked
  also losing to sage on the same defects says the defects are the
  schedule's, and the next lever is the partition, not the weights.
- **Priming, a caution for the next brief.** Do not name the expected
  defects to the judge before the pair verdict; put them in the singles'
  questions. A judge primed to look for melted text finds it in one half.
  The three briefs built on 2026-09-04 and 2026-09-05 do name the four
  PDD8 defects up front; their verdict records should say so.

**`--pairs` repeats.** A session with more than two arms is judged as one
reference arm against each of the others at matched seeds, one `--pairs` per
contest. `pair_NN` numbering runs continuously across contests so the judge
cannot read the contest off a filename, and the same two arms twice is refused
in either order.

`bench/score_session.py` is the only place the key is opened, and it opens it
only once the scores exist. It joins the export with
`internal/blind_keys/<session>.json` into
`bench/results/<date>_<session>_verdict.json`: per contest, the verdict tally
resolved through the key, so "Clip 1 better" becomes whichever arm actually sat
in slot 1, with `same` and `can't tell` counted apart; per arm, the tag counts,
the notes and the flags; per clip, its row, seed and graph. It refuses a scores
file that does not cover the batch, naming what is missing, unless `--partial`;
refuses a key whose session name is not the scores' session; and refuses to
write an absolute path into the record.

**That file is the only place grades live, and the mistake is worth naming
because someone made it on 2026-09-08.** A peer session looking for the
sage-versus-dense result opened `bench/results/<date>_<session>_arms_judged.jsonl`
instead, which is the RENDER record: arms, timings and the judge's free-text
notes, with no scores in it. Its clips tab reads `n_scored 0`, and they came
within one step of reporting that the comparison had been staged and never
graded, when it existed and answered the question being asked. Grades are in
the verdict json; the judged jsonl is what was rendered and what the judge
said, not what the judge decided. An empty result from the wrong artifact
looks exactly like a real absence.

A contest tally is a preference over distributions, not a per-pair verdict, and
the record carries that reading in its own field. Blinding controls who knows
which arm; it does not make two samples comparable.

```bash
H3_COMFY_OUTPUT=<share> python bench/blind_batch.py \
    --jsonl bench/results/<date>_<session>.jsonl \
    --session <session> --shuffle-seed <n> \
    --pairs ref,other --pairs ref,another \
    --brief-file <brief.txt>

# open <share>/Video/blind/<session>/score.html, score every item, Export scores

python bench/score_session.py --scores scores_<session>.json
```
