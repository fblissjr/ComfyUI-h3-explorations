# Path 3B: Agentic Vision (`code_execution` Zoom-and-Inspect) & Per-Item `media_resolution`

Companion reference to [`../SKILL.md`](../SKILL.md).

## 0. Native-First Gate (`agy` Check Before Using Path 3B)

- **Inside the Antigravity (`agy`) harness, `agy` can already crop, zoom, and compute pixel math locally at `$0` API spend**:
  1. Call `read_file` directly on the `.png` / `.jpg` image (`contact_sheets/<video>/`, `bench/inspect_mask_steps.py --strips`, or extracted frames).
  2. If a tile is small or you need RGB/luma shifts (`B-R`, luma delta inside `parts_held`), run a local `$PY` script (`PIL`, `cv2`, `numpy`) to crop the exact tile to `/tmp/crop.png` or compute the numbers locally, then call `read_file` on `/tmp/crop.png`.
- **When Path 3B (`gemini_media.py zoom`) IS permitted**:
  - Only when you specifically need an external model in `gemini-api` to run autonomous remote sandboxed `code_execution` (`tools: [{"type": "code_execution"}]` + `"resolution": "ultra_high"`) or when referencing the Gemini 3 / Gemini 4 per-item `media_resolution` token table below for `gemini-api` calls.
- **Owner media rule ([`docs/wiki/working_with_the_owner.md`](../../../../docs/wiki/working_with_the_owner.md))**: Always ask the owner and name the file before uploading project media via `gemini-api`.

---

## 1. Per-Content-Item `media_resolution` Token Table (Gemini 3 & Gemini 4)

Gemini 3 and Gemini 4 models allow setting `"resolution"` on **each individual media item** inside the `input` array. You can mix resolutions in a single request (for example, `"resolution": "low"` on a clean reference portrait and `"resolution": "ultra_high"` on a dense multi-row diagnostic strip):

| `resolution` Value | Image Tokens (Gemini 3 / 4) | Video Tokens / Frame (Gemini 3 / 4) | Audio Tokens / Sec | Recommended Use in This Repo |
| :--- | :--- | :--- | :--- | :--- |
| `"unspecified"` (Default) | `1120` | `70` | `25` | General single-still questions |
| `"low"` | `280` | `70` | `25` | Reference stills (`<Picture i>`) passed for coarse identity context |
| `"medium"` | `560` | `70` | `25` | Routine frame pair comparisons |
| `"high"` | `1120` | `280` | `25` | **Mandatory for `24 fps` static video QA**, burned-in frame numbers, and `box_2d` grounding |
| `"ultra_high"` | `2240` | N/A (image item only) | N/A | **Dense contact sheets** (`contact_sheets/<video>/`), `bench/inspect_mask_steps.py --strips` JPEGs, and tiny occluded head/hair slivers |

---

## 2. How Agentic Vision (`tools: [{"type": "code_execution"}]`) Works on Images

While `"processing": "agentic"` on **video** is banned on multi-panel strips because server-side video cropping cannot see the tile grid layout, **Code Execution with Images** (`tools: [{"type": "code_execution"}]` on an **image** input) works differently and is purpose-built for dense visual artifacts:
- The model receives the full image at `"resolution": "ultra_high"` (`2240 tokens`) and writes explicit Python code (`PIL`, `numpy`, `matplotlib`) in its sandbox to:
  1. **Zoom and inspect**: Crop specific tile coordinates or frame numbers (`code_execution_call`) and re-examine the cropped high-resolution patch in the next step (`code_execution_result`).
  2. **Visual math & pixel measurement**: Compute RGB/luma channel shifts (e.g., comparing replaced-part `B-R` and luma inside `parts_held` between source and rendered frames) or count grid cells.
  3. **Annotate**: Draw bounding boxes or arrows on an output image to prove where an artifact sits.

---

## 3. High-Value Repo Use Cases for `code_execution` + `ultra_high` (When Not Cropping Locally)

1. **Inspecting `bench/inspect_mask_steps.py --strips /tmp/<load>_strips.jpg`**:
   - `--strips` writes a multi-row diagnostic image (`RED` = `carried_px`, `BLUE` = regenerated latent step, `GREEN` = `strangers`, `YELLOW` = recoverable `clean_whole` reference).
   - Passing the strip image with `"resolution": "ultra_high"` and `tools: [{"type": "code_execution"}]` lets the model crop into individual flagged frames (such as `4698` stranger self-swallow or `4924` occluded gap) and read the exact contour overlap.
2. **Inspecting `contact_sheets/<video>/` grids**:
   - Let the model zoom into specific tiles on a contact sheet to read small burned-in frame numbers and verify whether a subject or accessory is present.
3. **Quick Inline Pair Comparison (`<20 MB`)**:
   - When `code_execution` is not needed and two images are under `~14 MB` total, run `gemini-api analyze -i source_frame.png -i rendered_frame.png "Compare hair boundary and sunglasses"` directly.

---

## 4. Running Agentic Image Zoom (`scripts/gemini_media.py zoom` or Direct CLI)

### Using `scripts/gemini_media.py zoom`

```bash
"$PY" .gemini/skills/video-understanding/scripts/gemini_media.py zoom "/tmp/${LOAD}_strips.jpg" \
  --prompt "Crop into each flagged frame tile, read the frame number, and report where the RED carried mask shrinks or touches the GREEN strangers contour."
```

### Direct `gemini-api` Command

```bash
FILE_ID=$(gemini-api files upload "/tmp/${LOAD}_strips.jpg" --wait)

gemini-api agent run -m "${GEMINI_VIDEO_MODEL:-gemini-3.8-flash}" --stream=false -o json --body "{
  \"input\": [
    {
      \"type\": \"image\",
      \"uri\": \"https://generativelanguage.googleapis.com/v1beta/${FILE_ID}\",
      \"mime_type\": \"image/jpeg\",
      \"resolution\": \"ultra_high\"
    },
    {
      \"type\": \"text\",
      \"text\": \"Use Python code to crop and zoom into the individual frame tiles in this diagnostic strip. Report the exact frame numbers and describe the RED, BLUE, GREEN, and YELLOW mask contours on each tile.\"
    }
  ],
  \"tools\": [
    {\"type\": \"code_execution\"}
  ]
}"

gemini-api files delete "$FILE_ID"
```
