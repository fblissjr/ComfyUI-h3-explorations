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
