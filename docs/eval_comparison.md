# Paired Video Evaluation & Stacking Guide

Tool: [`bench/stack_eval_clips.py`](../bench/stack_eval_clips.py)

Utility for building side-by-side or top-to-bottom comparison videos with synchronized playback and metadata overlays for blind and qualitative evaluations.

---


## Motion regime is part of a perceptual claim, not context for it

Added 2026-08-28, from an owner observation that turned out to be measurable and
to bear on judgements already made.

**Artifact severity in distilled arms tracks MOTION.** Comparing a coarse
partition against a fine one frame by frame with shot cuts masked, the effect
correlates with per-frame motion at **+0.676**, and the high-motion quartile
carries 1.18x the low-motion one.

**And this repo's scenes differ enormously in motion.** Measured on shipped
renders:

| scene | median motion | frames above 0.02 |
|---|---|---|
| market (t2v, wide, crowd) | 0.0142 | **47.6%** |
| dialogue, PDD 4-step | 0.0082 | 6.4% |
| dialogue, base 16-step | 0.0047 | **0.0%** |

**So a scene can be incapable of showing the defect being judged.** A dialogue
scene with zero high-motion frames cannot discriminate a motion-dependent
artifact, however carefully it is scored, however blind the process, and however
many seeds it gets. Blinding controls who knows which arm; it does nothing about
whether the content can express the difference.

**The rule.** A perceptual claim about a distilled arm states the motion regime
it was judged in. "Looks fine at 4 steps" on the dialogue corpus is a claim about
that corpus. If the question is whether a distillation holds up, the scene has to
contain the thing that breaks.

### Correction, same day: DELTA is the wrong axis for choosing a scene

The measurement above is real and unretracted. **Its label is wrong twice, and
the second one changes what you should do with it.**

**First, "motion" was a gloss.** What is computed is mean |frame[n] -
frame[n-1]| — frame-to-frame CHANGE, whatever causes it. No flow, no tracking.
A cut, a light switching on and a camera whip all score high with nothing
moving in the scene's own terms. Read every "motion" above as **delta**.
`bench/measure_clip_delta.py` is the instrument.

**Second, and worse: across shots, severity runs OPPOSITE to delta.** The owner
identifies shot 3 of the market scene — wide, crowd, crates of small coloured
fruit — as where ghosting and melting appear, and shot 2, a crate lift, as
"just ghosts a bit". Measured per shot on `text_to_video_pdd_4step_00007`, cuts
located by delta peak rather than assumed:

| shot | mean delta | spatial detail | change concentration |
|---|---|---|---|
| 1 | 0.0253 | 0.0632 | 32.4% |
| 2 — crate lift | 0.0252 | 0.0507 | 38.2% |
| **3 — crowd and fruit** | **0.0058** | **0.0666** | 64.6% |

The worst shot has **0.23x the delta and 1.32x the spatial detail** of the one
called nearly clean. The camera is locked and the stalls are static, so
frame-differencing reads a quiet shot where the eye reads a busy one.

**Both numbers survive because they measure different things.** The +0.676 is
WITHIN a clip, frame to frame, cuts masked — busier frames may well be worse
inside a shot. The table above is ACROSS shots. What does not survive is using
delta to choose a scene, which is what the rule above was for.

**The hypothesis, in the owner's correction of it.** First stated here as
"detail at the latent resolution limit" measured on the render. He corrected
the quantity the same evening: *"what i meant by ghosted fruit was.... theres
too much detail in whats being asked of the scene / frames being generated."*

**Demand, not result.** What matters is the fine structure the PROMPT asks to
be resolved, which is readable from the text before anything renders. Detail
measured on the output is the wrong quantity twice over: it is downstream of
the cause, and it is ambiguous between a plain scene and a destroyed one, since
ghosted fruit reads as low detail. A wide market shot demands each orange be
resolved in one or two latent cells where a stairwell closeup gives a face
hundreds, and fewer steps means less refinement at exactly that scale.

**Pre-registered predictions, written before the arms rendered.**

| arms | demand predicts | the account it beats |
|---|---|---|
| `h3_text_to_video_{aisle,sortline}_{short,long}` | long worse than short in **both** scenes, damage on the elaborated surfaces | length-of-conditioning: no consistent direction |
| market shot-count ablation | `shots12` best (lowest total demand), `shots13` worst | delta: `shots12` worst, being the all-high-delta arm |

The two rows oppose each other on `shots12`, which is what makes it worth
scoring first. And the demand pairs hold subjects, actions, camera moves,
dialogue and cut times exactly, so they manipulate demand more cleanly than any
shot swap can — a shot swap necessarily changes content.

**One reading that was tested and failed**, recorded so it is not re-proposed:
that shot 3 is bad because many things move independently. Its change is more
CONCENTRATED than shot 2's — 64.6% of it in the busiest 5% of pixels against
38.2% — because the camera is locked. Colour versus greyscale also makes no
difference here (0.0058 against 0.0061), so a greyscale metric is not the gap.

**The revised rule.** A perceptual claim states the regime it was judged in,
and delta alone does not describe the regime. Report delta AND spatial detail;
where they disagree, as they do across the market shots, say which one the
scene was chosen for.

**This does not retire the dialogue scenes.** They remain the sharpest AUDIO
instrument available -- eight lines, seven speaker changes, two registers -- and
that is a different axis from the one above. The error is calling such a scene a
quality probe rather than an audio probe.

`docs/research/pdd/audio_under_pdd.md` has the measurements and the one thing
they cannot settle: the distilled dialogue arm shows MORE frame-to-frame change
than the undistilled one on the same prompt, which is either motion the base did
not generate or the artifact registering as motion.

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
uv run python bench/stack_eval_clips.py clip1.mp4 clip2.mp4 --layout horizontal

# Force top-to-bottom regardless of aspect:
uv run python bench/stack_eval_clips.py clip1.mp4 clip2.mp4 --layout vertical
```

### Blind Evaluation Workflow:
Randomizes the assignment of Clip 1 and Clip 2, stamps anonymous overlays, and writes a sealed keyfile:
```bash
uv run python bench/stack_eval_clips.py \
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
   not a comparison -- CLAUDE.md's different-sample rule -- so N is the number
   of seeds the claim needs, and for a perceptual claim that is many.
2. **Blind with `bench/blind_batch.py`**: neutral `clip_NN.mp4` copies under
   `Video/blind/<session>/`, a MANIFEST with row indices only, and a sealed
   key in `internal/blind_keys/<session>.json` (gitignored). For a two-arm
   session add `--pairs A,B`: the i-th clip of each arm, matched by run index,
   stacked by this tool's layout rule as `pair_NN.mp4` with "Clip 1" / "Clip 2"
   in a per-pair random order, which the key also records. Stacks carry no
   audio; the singles do. Rows flagged `suspect_cache_hit` or `error`, or
   whose clip cannot be found, refuse the whole batch.
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

A contest tally is a preference over distributions, not a per-pair verdict, and
the record carries that reading in its own field. Blinding controls who knows
which arm; it does not make two samples comparable.

```bash
H3_COMFY_OUTPUT=<share> uv run python bench/blind_batch.py \
    --jsonl bench/results/<date>_<session>.jsonl \
    --session <session> --shuffle-seed <n> \
    --pairs ref,other --pairs ref,another \
    --brief-file <brief.txt>

# open <share>/Video/blind/<session>/score.html, score every item, Export scores

uv run python bench/score_session.py --scores scores_<session>.json
```
