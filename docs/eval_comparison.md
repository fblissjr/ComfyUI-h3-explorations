# Paired Video Evaluation & Stacking Guide

Tool: [`bench/stack_eval_clips.py`](../bench/stack_eval_clips.py)

Utility for building side-by-side or top-to-bottom comparison videos with synchronized playback and metadata overlays for blind and qualitative evaluations.

---

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
