# Path 3A: Image Understanding — 2D Bounding Boxes, Polygon Segmentation & SAM 3.1 Integration

Companion reference to [`../SKILL.md`](../SKILL.md).

## 0. Native-First Gate (`agy` Check Before Using Path 3A)

- **Check local `<preview>_people.npz` and `seg29.mkv` first (`$0` API spend)**: Before calling `gemini-api` for bounding boxes or masks, check whether local SAM 3.1 detections (`<preview>_people.npz`) or Sapiens-2B body/face/hair parts (`seg29.mkv`) already cover the subject on that frame using `$PY` and `agy`'s built-in `read_file` tool on the frame/strip.
- **When Path 3A IS permitted**:
  1. `agy` cannot produce schema-calibrated `0..1000` `box_2d` / polygon `mask` JSON natively from chat text, and
  2. You need off-card `0..1000` `box_2d` / `mask` grounding while the GPU (`:8188`) is busy—for example, to disambiguate which of multiple SAM 3.1 person detections on a frame matches a cast reference still (`<Picture 1>`), or to localize hair/crown pixels missed by SAM 3.1 / Sapiens-2B (`HAIR_IN_HOLE_CANNOT_SEE`).
- **Owner media rule ([`docs/wiki/working_with_the_owner.md`](../../../../docs/wiki/working_with_the_owner.md))**: Always ask the owner and name the file before uploading project media via `gemini-api`.

---

## 1. Where Gemini Image Grounding Fits in `docs/masked_v2v/`

Our local perception models (`SAM 3.1` in `subject_track.py` / `bench/strangers.py` and `Sapiens-2B` in `sapiens2_parts.py`) run on the GPU (`:8188`). Using Gemini's native object detection (`box_2d`) and polygon segmentation (`mask`) via `gemini-api` provides a **zero-VRAM, off-card perception layer** while ComfyUI holds the card:

1. **Group 3 `cast.jsonl` Seed Verification (`bench/cast_file.py::drawn`)**:
   - Pass a shot keyframe alongside the subject's reference still (`<Picture 1>`) in a multi-image prompt to locate the target subject after a cut or heavy occlusion.
   - Convert Gemini's normalized `[ymin, xmin, ymax, xmax]` (`0..1000`) box to pixel coordinates on `1024x768` and snap it to the verified SAM 3.1 detection in `<preview>_people.npz` (`IoU >= 0.50`, required by `bench/cast_file.py:356`).
2. **Resolving `HAIR_IN_HOLE_CANNOT_SEE` & `OTHER_CANNOT_SEE` (`bench/capture_masked_run.py:5169-5179`)**:
   - When SAM 3.1 loses a head behind a hat brim or raised arm (`her_head_lost_in_view`) or Sapiens-2B labels an emerging crown as `Background` (e.g., `4912` frames `4922–4925`, `4944–4945`), query Gemini on the raw frame to verify whether the subject's head is `hidden`, `hers`, or `not hers` before recording `plan.json` `head_marks`, or extract a polygon `mask` of the emerging hair/crown.
3. **Auditing `restore=` Accessory Presence (Regime 1 vs. Regime 2)**:
   - Detect whether H3 already rendered an accessory (`Eyeglass`, `Apparel@head`, microphone, guitar strap) inside a landed render frame (Regime 1, where `restore=` must NOT be applied) vs. omitted it (Regime 2, where `restore=` is needed).

---

## 2. Coordinate Convention Warning (`[ymin, xmin, ymax, xmax]` in `0..1000`)

- **Gemini `box_2d`**: Always `[ymin, xmin, ymax, xmax]` (note **`y` first, `x` second**!) scaled to `[0, 1000]`.
- **Gemini `mask`**: Polygon list of `[[x, y], ...]` vertices scaled to `[0, 1000]`.
- **Repo `cast.jsonl` (`bench/cast_file.py`)**: Expects `--box FRAME:LEFT,TOP,WIDTH,HEIGHT` in `picture_size` (`1024x768`), i.e., `[xmin_px, ymin_px, width_px, height_px]`.
- **Descaling formula for canvas `(W, H)` (e.g., `W = 1024, H = 768`)**:
  $$x_1 = \frac{\text{xmin}}{1000} \cdot W, \quad y_1 = \frac{\text{ymin}}{1000} \cdot H, \quad x_2 = \frac{\text{xmax}}{1000} \cdot W, \quad y_2 = \frac{\text{ymax}}{1000} \cdot H$$
  $$\text{LEFT} = x_1, \quad \text{TOP} = y_1, \quad \text{WIDTH} = x_2 - x_1, \quad \text{HEIGHT} = y_2 - y_1$$

---

## 3. One-Command Grounding + SAM 3.1 `<preview>_people.npz` Snap (`scripts/gemini_media.py boxes`)

Use [`../scripts/gemini_media.py`](../scripts/gemini_media.py) to run detection/segmentation, descale `[ymin, xmin, ymax, xmax]` to both `[x1, y1, x2, y2]` and `cast.jsonl` `[left, top, width, height]` without `x`/`y` transposition risk, and optionally match against `<preview>_people.npz`:

```bash
# Detect a subject on frame 4686 using her reference still, and snap to SAM 3.1 detections in _people.npz:
"$PY" .gemini/skills/video-understanding/scripts/gemini_media.py boxes "$FRAME_PNG" \
  --ref "$REF_STILL_PNG" \
  --prompt "Locate the woman from the first reference image in the second scene frame (her head and blonde hair)." \
  --people "$PREVIEW_PEOPLE_NPZ" \
  --frame 4686 \
  --with-masks
```

The JSON output includes:
- `box_2d_1000`: raw `[ymin, xmin, ymax, xmax]` (`0..1000`)
- `box_xyxy_px`: `[x1, y1, x2, y2]` in image pixels
- `box_ltwh_px`: `[left, top, width, height]` formatted for `bench/cast_file.py box --box FRAME:LEFT,TOP,WIDTH,HEIGHT`
- `sam_match`: when `--people` and `--frame` are supplied, reports `best_iou`, `passes_0_50_gate`, and the exact snapped SAM 3.1 detection box `sam_box_ltwh` in `cast.jsonl` coordinates.

---

## 4. Direct `gemini-api` Structured JSON Call (`box_2d` + `mask`)

When calling `gemini-api agent run` directly for segmentation masks, set `"thinking_level": "minimal"` in `"generation_config"` (per the Gemini segmentation guide) and `"resolution": "high"` on the image input:

```bash
FILE_ID=$(gemini-api files upload "$FRAME_PNG" --wait)

gemini-api agent run -m "${GEMINI_VIDEO_MODEL:-gemini-3.8-flash}" --stream=false -o json --body "{
  \"input\": [
    {
      \"type\": \"image\",
      \"uri\": \"https://generativelanguage.googleapis.com/v1beta/${FILE_ID}\",
      \"mime_type\": \"image/png\",
      \"resolution\": \"high\"
    },
    {
      \"type\": \"text\",
      \"text\": \"Detect the subjects and their hair/head regions. Return box_2d as [ymin, xmin, ymax, xmax] normalized to 0-1000, mask as a polygon of [x, y] coordinates normalized to 0-1000, and a descriptive label.\"
    }
  ],
  \"response_format\": {
    \"type\": \"text\",
    \"mime_type\": \"application/json\",
    \"schema\": {
      \"type\": \"object\",
      \"properties\": {
        \"boxes\": {
          \"type\": \"array\",
          \"items\": {
            \"type\": \"object\",
            \"properties\": {
              \"box_2d\": {\"type\": \"array\", \"items\": {\"type\": \"integer\"}},
              \"mask\": {\"type\": \"array\", \"items\": {\"type\": \"array\", \"items\": {\"type\": \"integer\"}}},
              \"label\": {\"type\": \"string\"}
            },
            \"required\": [\"box_2d\", \"mask\", \"label\"]
          }
        }
      },
      \"required\": [\"boxes\"]
    }
  },
  \"generation_config\": {
    \"thinking_level\": \"minimal\"
  }
}"

gemini-api files delete "$FILE_ID"
```
