# Path 2B: Opt-In Agentic Video Understanding & Multi-Video Mixing

Companion reference to [`../SKILL.md`](../SKILL.md).

## 0. Native-First Gate (`agy` Check Before Using Path 2B)

- **Never use Path 2B when `*_STACKED.1fps.*.mp4` or `<stem>.1fps.<first>-<last>.mp4` companions exist**: Call `agy`'s built-in `read_file` tool on the `<80 MB` `.1fps.mp4` parts at `$0` API spend.
- **When Path 2B IS permitted**:
  1. Navigating a `>100 MB` or `>5–10 min` **single-canvas** master cut (`*_joined_*_from_laid.mp4`) that `read_file` cannot load (`100 MB` ceiling), or
  2. Mixing `"processing": "agentic"` on a long single-canvas master cut with `"processing": {"type": "static", "fps": 24}` on a candidate shot render in a single cross-video continuity query (`--compare-shot`).
- **Owner media rule ([`docs/wiki/working_with_the_owner.md`](../../../../docs/wiki/working_with_the_owner.md))**: Always ask the owner and name the file before uploading project media via `gemini-api`.

---

## 1. Opt-In Requirement & Supported Models

Agentic video understanding is **never enabled by default**. Even on Gemini 4 Argon and Gemini 3.8 Flash, omitting `"processing"` defaults to `1 fps` static frame extraction.
- **How to enable**: Set `"processing": "agentic"` on the `"type": "video"` input object.
- **Supported models**: `gemini-3.8-flash` (default), `gemini-4-argon`, `gemini-3.6-flash`, `gemini-3.5-flash-lite`, and newer models.
- **How it works**: The model inspects the audio transcript first and dynamically loads visual frames or audio segments on demand at adaptive frame rates and resolutions (`processing_call` and `processing_result` steps interleaved with `thought` steps in `interaction.steps`), reducing video input token usage by up to `88%` on long-form content.

---

## 2. When to Enable vs. When to Keep Off

### Enable `"processing": "agentic"` when:
- Inspecting a **long-form single-canvas video** (`>5–10 minutes` or `>100 MB`, such as a full-length `~414 MB` `*_joined_*_from_laid.mp4` master cut) when its `*_STACKED.1fps.*.mp4` parts are not the target.
- Running **needle-in-a-haystack** or targeted timeline retrieval (e.g., *"Find every shot transition where the third subject appears with a microphone"*).
- **Comparing a long master cut against a short reference clip** in a single request by mixing `"processing": "agentic"` and `"processing": "static"` across two video inputs (Section 4 below).

### Keep `"processing": "agentic"` OFF when:
- Inspecting **short clips (`<5 min`) where every frame matters**: Static mode (`Path 1` `.1fps.mp4` or `Path 2A` `fps: 24`) is faster and avoids navigation thinking tokens (`total_thought_tokens`) and tool-call round-trips (`total_tool_use_tokens`).
- Inspecting **multi-panel / stacked diagnostic strips (`looks/`, `review_clips/`, `*_STACKED.mp4`)**:
  - **Why it is banned on stacked strips**: Server-side dynamic spatial cropping across multi-row diagnostic tiles misreads stacked rows. In live tests on a `35-frame` stacked look strip, `"processing": "agentic"` caused `gemini-3.8-flash` to hallucinate a nonexistent hard scene cut at frame `04937` and `gemini-4-argon` to hallucinate a 3D facial mesh collapsing into a vertical ribbon. [`../scripts/gemini_media.py`](../scripts/gemini_media.py) mechanically refuses `*_STACKED*` and `.1fps.` filenames on `agentic-video`.

---

## 3. Single-Canvas Agentic Video Inspection (`scripts/gemini_media.py agentic-video` or Direct CLI)

### Using `scripts/gemini_media.py agentic-video` (Recommended)

```bash
# Single-canvas agentic timeline search with automatic remote file cleanup and processing_call trace summary:
"$PY" .gemini/skills/video-understanding/scripts/gemini_media.py agentic-video "$LARGE_MASTER_MP4" \
  --prompt "Locate shot transitions and summarize the timeline across this master cut. Use MM:SS timestamps."
```

### Direct `gemini-api` Command

For long videos where multiple timeline navigation passes take longer, either keep streaming enabled (`--stream=true`, the CLI default) or run asynchronously with `--background` and poll via `gemini-api agent status <interaction_id>` to avoid connection or token-validity timeouts.

```bash
FILE_ID=$(gemini-api files upload "$LARGE_MASTER_MP4" --wait)

# Synchronous JSON run (add --jq '.steps[] | select(.type=="processing_call" or .type=="model_output")' to inspect navigation trace):
gemini-api agent run -m "${GEMINI_VIDEO_MODEL:-gemini-3.8-flash}" --stream=false -o json --body "{
  \"input\": [
    {
      \"type\": \"video\",
      \"uri\": \"https://generativelanguage.googleapis.com/v1beta/${FILE_ID}\",
      \"mime_type\": \"video/mp4\",
      \"processing\": \"agentic\"
    },
    {
      \"type\": \"text\",
      \"text\": \"Locate shot transitions and summarize the timeline across this master cut. Use MM:SS timestamps.\"
    }
  ]
}"

gemini-api files delete "$FILE_ID"
```

### Multi-Turn Follow-Ups on an Agentic Video Session
- **Stateful (recommended in CLI)**: Capture the `.id` from the first interaction and pass `--previous-interaction-id "$INTERACTION_ID"` on follow-up `gemini-api agent run` calls so the server retains the navigated video context.
- **Stateless**: If re-sending `step_list` manually, you must include all `processing_call` and `processing_result` steps from the prior response, or the navigated visual context is lost.

---

## 4. Mixing `"agentic"` and `"static"` Modes Across Multiple Videos (`--compare-shot`)

Because `"processing"` is configured per video item in `input`, you can combine agentic navigation on a long master cut with frame-accurate `24 fps` static inspection on a specific shot render in one call:

### Using `scripts/gemini_media.py agentic-video --compare-shot`

```bash
"$PY" .gemini/skills/video-understanding/scripts/gemini_media.py agentic-video "$MASTER_CUT_MP4" \
  --compare-shot "$CANDIDATE_SHOT_MP4" \
  --prompt "Compare the candidate shot render (second video) against adjacent shots of the same subject in the master cut (first video) for hair color and lighting continuity."
```

### Direct `gemini-api` Command

```bash
MASTER_ID=$(gemini-api files upload "$MASTER_CUT_MP4" --wait)
SHOT_ID=$(gemini-api files upload "$CANDIDATE_SHOT_MP4" --wait)

gemini-api agent run -m "${GEMINI_VIDEO_MODEL:-gemini-3.8-flash}" --stream=false -o json --body "{
  \"input\": [
    {
      \"type\": \"video\",
      \"uri\": \"https://generativelanguage.googleapis.com/v1beta/${MASTER_ID}\",
      \"mime_type\": \"video/mp4\",
      \"processing\": \"agentic\"
    },
    {
      \"type\": \"video\",
      \"uri\": \"https://generativelanguage.googleapis.com/v1beta/${SHOT_ID}\",
      \"mime_type\": \"video/mp4\",
      \"resolution\": \"high\",
      \"processing\": {\"type\": \"static\", \"fps\": 24}
    },
    {
      \"type\": \"text\",
      \"text\": \"Compare the candidate shot render (second video) against adjacent shots of the same subject in the master cut (first video) for hair color and lighting continuity.\"
    }
  ]
}"

gemini-api files delete "$MASTER_ID"
gemini-api files delete "$SHOT_ID"
```
