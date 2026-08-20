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
    --label1 "LoRA (fl2va + ref_lora)" \
    --label2 "Checkpoint (ref2va base)" \
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
