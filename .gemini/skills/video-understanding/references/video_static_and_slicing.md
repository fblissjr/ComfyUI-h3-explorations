# Path 2A: Static Video Processing, Frame-Span Slicing, and File Input Methods

Companion reference to [`../SKILL.md`](../SKILL.md).

## 0. Native-First Gate (`agy` Check Before Using Path 2A)

- **Never use Path 2A on any `<100 MB` local `.mp4` clip (including `.1fps.<first>-<last>.mp4` strips, `<Video 1>` reference `.mp4`s, raw node renders `<load>_00001.mp4`, and `24 fps` review clips)**: Call `agy`'s built-in `read_file` (`view_file`) tool directly on the `.1fps.mp4` and/or `24 fps` `.mp4` at `$0` API spend. If `1:1` every-frame inspection is needed on a `<100 MB` local `.mp4` that does not yet have a `.1fps.mp4` companion, encode `<stem>.1fps.<first>-<last>.mp4` locally in `<1s` on CPU via [`bench/_lib/frames.py::OneASecond`](../../../../bench/_lib/frames.py) and call `read_file` on it. [`../scripts/gemini_media.py`](../scripts/gemini_media.py) mechanically refuses local `<100 MB` videos when a `.1fps.*.mp4` sibling exists on disk.
- **When Path 2A IS permitted**:
  1. Slicing a `[first..last]` shot span (`start_offset` / `end_offset` at `fps: 24`) out of a `>100 MB` master cut (`*_joined_*_from_laid.mp4`) that exceeds `read_file`'s `100 MB` cap, or
  2. Inspecting an already-uploaded remote `files/<id>` at `fps: 24` when local `.1fps.mp4` generation via `OneASecond` is not available.
- **Owner media rule ([`docs/wiki/working_with_the_owner.md`](../../../../docs/wiki/working_with_the_owner.md))**: Always ask the owner and name the file before uploading project media via `gemini-api`.

---

## 1. How Static Video Processing Works (API Default)

By default, all Gemini models (including Gemini 4 Argon and Gemini 3.8 Flash) process video in **static** mode (`"processing": "static"` or `{"type": "static", ...}`):
- **Default sampling**: Extracts `1 frame/second` (`1 fps`) and places all extracted frames into context in a single pass, alongside single-channel audio (`32 tokens/s`).
- **Why use Static over Agentic on short clips (`<5 min`)**: Static processing avoids the extra thinking tokens (`total_thought_tokens`) and tool-call round-trips (`total_tool_use_tokens`) incurred by timeline navigation when most or all of a short clip is relevant.
- **Per-item `resolution` on video (Gemini 3 and Gemini 4)**:
  - `"low"` / `"medium"` / `"unspecified"` (default): `~70 visual tokens/frame` (`~100 tokens/s` with audio and metadata).
  - `"high"`: `~280 visual tokens/frame` (`~300 tokens/s` with audio and metadata). **Always set `"resolution": "high"` when reading burned-in frame numbers or multi-panel diagnostic strips.**

---

## 2. File Input Methods (`gemini-api`)

| Method | Size Limit | Retention / Cleanup | Best Use in Repo |
| :--- | :--- | :--- | :--- |
| **Inline (`gemini-api analyze -i <path>`)** | `<20 MB` total request (`~14 MB` raw file before base64) | Stateless (no remote object created) | Quick single-still, pair-still, or tiny clip questions where custom `VideoContent.processing` is not needed |
| **Files API (`gemini-api files upload <path> --wait`)** | Up to `2 GB` per file (`20 GB` project quota) | `48 hours` TTL; delete immediately after use with `gemini-api files delete "$FILE_ID"` | Any `24 fps` static inspection, `start_offset`/`end_offset` shot slicing, or `>14 MB` video |
| **GCS Registration (`gemini-api files register gs://...`)** | GCS object limit | Registered reference (no copy) | Cloud-hosted assets (rarely used in local ComfyUI workflows) |

> **CLI quirk (`v0.4.4`)**: Never pass `-i files/<id>` to `gemini-api generate` or `gemini-api analyze` when you need custom `"processing"` (`fps`, `start_offset`, `end_offset`, or `agentic`); the CLI serializes `-i files/<id>` as `{"type": "document", "uri": "..."}`. Always pass `--body` to `gemini-api agent run` with an explicit `"type": "video"` block and place the `"type": "text"` prompt **after** the video part.

---

## 3. Whole-Clip `24 fps` Static Inspection

Use when inspecting a short raw `24 fps` clip that does not have a `.1fps.<first>-<last>.mp4` companion:

```bash
FILE_ID=$(gemini-api files upload "$CLIP_PATH" --wait)

gemini-api agent run -m "${GEMINI_VIDEO_MODEL:-gemini-3.8-flash}" --stream=false -o json --body "{
  \"input\": [
    {
      \"type\": \"video\",
      \"uri\": \"https://generativelanguage.googleapis.com/v1beta/${FILE_ID}\",
      \"mime_type\": \"video/mp4\",
      \"resolution\": \"high\",
      \"processing\": {\"type\": \"static\", \"fps\": 24}
    },
    {
      \"type\": \"text\",
      \"text\": \"Inspect every frame at 24 fps and report exact frame numbers.\"
    }
  ]
}"

gemini-api files delete "$FILE_ID"
```

---

## 4. Server-Side Shot-Window Slicing (`start_offset` / `end_offset` at `24.0 fps`)

Every working copy and delivery cut in `docs/masked_v2v/` is locked to exact `24.0 fps` ([`h3_rules.py::FPS`](../../../../h3_rules.py), `24`). Instead of running `ffmpeg` and uploading a separate file for each shot when inspecting a large working copy or master cut (`*_joined_*_from_laid.mp4`), upload the video **once** and slice any `[first..last]` frame span on the server using `start_offset` and `end_offset` (in seconds):

$$\text{start\_offset} = \frac{\text{first}}{24.0}, \qquad \text{end\_offset} = \frac{\text{last} + 1}{24.0}$$

### Using `scripts/gemini_media.py slice` (Recommended)

```bash
# Upload the master cut once for the session:
FILE_ID=$(gemini-api files upload "$MASTER_MP4" --wait)

# Inspect frames 4912..4946 at 24 fps high resolution (computes start_offset/end_offset automatically):
"$PY" .gemini/skills/video-understanding/scripts/gemini_media.py slice "$FILE_ID" \
  --first 4912 --last 4946 --fps 24 \
  --prompt "Inspect the second subject's hair and face across frames 4912 to 4946."

# Clean up when finished with all shot slices:
gemini-api files delete "$FILE_ID"
```

### Direct `gemini-api` Command

```bash
START_S=$(awk "BEGIN {printf \"%.3f\", 4912 / 24.0}")
END_S=$(awk "BEGIN {printf \"%.3f\", (4946 + 1) / 24.0}")

gemini-api agent run -m "${GEMINI_VIDEO_MODEL:-gemini-3.8-flash}" --stream=false -o json --body "{
  \"input\": [
    {
      \"type\": \"video\",
      \"uri\": \"https://generativelanguage.googleapis.com/v1beta/${FILE_ID}\",
      \"mime_type\": \"video/mp4\",
      \"resolution\": \"high\",
      \"processing\": {
        \"type\": \"static\",
        \"start_offset\": ${START_S},
        \"end_offset\": ${END_S},
        \"fps\": 24
      }
    },
    {
      \"type\": \"text\",
      \"text\": \"Inspect this shot slice at 24 fps and report any transition or mask seam artifacts.\"
    }
  ]
}"
```
