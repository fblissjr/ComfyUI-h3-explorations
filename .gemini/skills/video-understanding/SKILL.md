---
name: video-understanding
description: Route video, image, and audio inspection in this repo with a strict Native-First Gate inside the Antigravity (agy) harness — always using built-in read_file (.1fps.<first>-<last>.mp4, <100 MB 24 fps .mp4, .png/.jpg) and local $PY (PIL/cv2/numpy, <preview>_people.npz, bench/voice_spans.py) at $0 API spend first, and calling gemini-api ONLY for capabilities agy cannot perform natively.
reviewed: 3c8214a9
---

# Video, Image, and Audio Understanding Router

This skill routes media QA and perception across `read_file`, local `$PY` tools, and the `gemini-api` CLI. To keep context lean, **read only the specific reference file in `references/` when you hit a task `agy` cannot perform natively**.

## Rule 0 — Native-First Gate (`agy` Native Tools Before Any `gemini-api` Call)

Inside the Antigravity (`agy`) harness, **never invoke `gemini-api` or `scripts/gemini_media.py` for a task `agy` can perform natively (`$0` API key spend, zero external uploads)**:

1. **Every-frame video QA (`<100 MB`) -> Native `read_file` on `.1fps.<first>-<last>.mp4`**:
   - Every look strip (`looks/`), review clip (`review_clips/`), and stacked delivery (`*_STACKED.1fps.<first>-<last>.mp4`) already has (or can be given via [`bench/_lib/frames.py::OneASecond`](../../../bench/_lib/frames.py)) a `.1fps.` companion where each frame is held `1.0s`. Calling `read_file` on `.1fps.mp4` gives deterministic `1:1` frame alignment at `$0` API spend.
   - `scripts/gemini_media.py` mechanically refuses uploading any local `<100 MB` `.mp4` that has a `<stem>.1fps.*.mp4` sibling on disk.
2. **Audio, vocal sync, and real-time motion (`<100 MB`) -> Native `read_file` on `24 fps` `.mp4` + local [`bench/voice_spans.py`](../../../bench/voice_spans.py)**:
   - Call `read_file` directly on the `<100 MB` `24 fps` `.mp4` clip and run [`bench/voice_spans.py`](../../../bench/voice_spans.py) locally before considering any `gemini-api` transcription call.
3. **Images, contact sheets, and `--strips` JPEGs -> Native `read_file` + local `$PY` (`PIL` / `cv2` / `numpy`)**:
   - Call `read_file` directly on `.png` / `.jpg` stills. If a tile on a contact sheet or `bench/inspect_mask_steps.py --strips` JPEG is small, or if you need RGB/luma pixel math, crop or measure locally with `$PY` (`PIL`, `cv2`, `numpy`) and call `read_file` on the local crop.
4. **Subject boxes and masks -> Local `<preview>_people.npz` and `seg29.mkv` first**:
   - Inspect existing SAM 3.1 detections in `<preview>_people.npz` and Sapiens-2B parts in `seg29.mkv` locally with `$PY` before making any remote `box_2d` / `mask` grounding call.
5. **When `gemini-api` escalation IS permitted (capabilities `agy` cannot do natively)**:
   - **File size `>100 MB`**: `read_file` rejects files over `100 MB` (e.g., a `~414 MB` unsplit `*_joined_*_from_laid.mp4` master cut).
   - **Server-side `24 fps` static inspection or `start_offset`/`end_offset` slicing on a video without `.1fps.mp4`**: Native `read_file` is locked to `1 fps` sampling on raw `24 fps` video (`23` of every `24` frames are skipped unless `.1fps.mp4` exists).
   - **Opt-in `"processing": "agentic"` timeline navigation**: Long single-canvas videos (`>5–10 min` or `>100 MB`) where transcript-first dynamic frame loading or mixed `agentic` + `static` multi-video comparison is needed.
   - **Schema-calibrated `0..1000` `box_2d` / polygon `mask` grounding**: Off-card localization when the GPU (`:8188`) is busy and local SAM 3.1 / Sapiens-2B missed the subject or hair (`HAIR_IN_HOLE_CANNOT_SEE`).
   - **Owner media rule ([`docs/wiki/working_with_the_owner.md`](../../../docs/wiki/working_with_the_owner.md))**: Never upload the owner's media (renders, clips, stills, audio) to `gemini-api` without asking and naming what leaves the machine.

---

## Progressive Disclosure Routing Matrix

| Path / Capability | Can `agy` Do This Natively? | Mandatory Native Path First (`$0` API Spend) | When to Escalate to `gemini-api` + Reference Guide | Helper Subcommand (`.gemini/skills/video-understanding/scripts/gemini_media.py`) |
| :--- | :--- | :--- | :--- | :--- |
| **Path 1: Native `read_file` + Local `$PY`** | **YES** | `read_file` on `.1fps.<first>-<last>.mp4` + `24 fps` `.mp4` + `.png`/`.jpg`; local `$PY` (`PIL`/`cv2`, `_people.npz`, `voice_spans.py`). | **Never** when Path 1 satisfies the task. | Native `read_file` tool + local `$PY` |
| **Path 2A: `gemini-api` Static `24 fps` & Shot Slicing** | **NO** on `>100 MB` files or raw `24 fps` video without `.1fps.mp4` (`read_file` samples at `1 fps`) | Use `.1fps.mp4` via `read_file` whenever present (or build with `OneASecond`). | Raw `24 fps` video without `.1fps.mp4`, or slicing `[first..last]` spans from an uploaded `>100 MB` master cut. See [`references/video_static_and_slicing.md`](references/video_static_and_slicing.md). | `gemini_media.py slice <files/id> --first F --last L --prompt "..."` |
| **Path 2B: `gemini-api` Opt-In Agentic Video (`"processing": "agentic"`)** | **NO** (`read_file` has no `>100 MB` support or dynamic agentic pass) | Use `*_STACKED.1fps.*.mp4` parts via `read_file` when inspecting laid shots. | Long single-canvas videos (`>5–10 min` or `>100 MB`) or mixed `agentic` (master cut) + `static` (candidate shot) calls. **BANNED on stacked strips.** See [`references/video_agentic.md`](references/video_agentic.md). | `gemini_media.py agentic-video <master.mp4> [--compare-shot <shot.mp4>] --prompt "..."` |
| **Path 3A: `gemini-api` `0..1000` `box_2d` & Polygon `mask`** | **NO** (chat text is not schema-calibrated to `0..1000` `box_2d`/`mask` heads) | Check `<preview>_people.npz` and `seg29.mkv` locally with `$PY` first. | Off-card `0..1000` `box_2d` / `mask` grounding when local SAM 3.1 / Sapiens-2B missed a head/hair or needs Group 3 `cast.jsonl` seed disambiguation. See [`references/image_boxes_and_masks.md`](references/image_boxes_and_masks.md). | `gemini_media.py boxes <frame.png> [--ref <pic.png>] [--people <npz> --frame F] [--with-masks] --prompt "..."` |
| **Path 3B: `gemini-api` Agentic Image Zoom (`code_execution` + `ultra_high`)** | **YES** via local `$PY` crop + `read_file` | Crop tiles or compute pixel math locally with `$PY` (`PIL`/`cv2`/`numpy`) and call `read_file` on the crop. | Only when remote sandboxed `code_execution` + `"resolution": "ultra_high"` is explicitly needed (or for the Gemini 3/4 `media_resolution` token table). See [`references/image_agentic_zoom.md`](references/image_agentic_zoom.md). | `gemini_media.py zoom <strips.jpg> --prompt "..."` |
| **Path 4: `gemini-api` Speaker & Lyric Transcription** | **YES** on `<100 MB` clips (`read_file` + `bench/voice_spans.py`) | Run [`bench/voice_spans.py`](../../../bench/voice_spans.py) (`BS-RoFormer` + `silero_vad`) and `read_file` on the `<100 MB` `24 fps` `.mp4`. | Only on `>100 MB` videos or when multi-speaker visual+audio attribution is needed to fill `voice_spans.py`'s `cannot_see` gap. See [`references/audio_and_transcription.md`](references/audio_and_transcription.md). | `gemini_media.py transcribe-spans <clip.mp4> --shot-first F [--visual-speakers]` |

---

## 1. Path 1 (Mandatory Default): `read_file` on `.1fps.<first>-<last>.mp4` + `24 fps` `.mp4`

Use this path for all look-strip (`$H3_OUTPUT_DIR/Video/<video>/looks/...`), review-clip (`$H3_OUTPUT_DIR/Video/<video>/review_clips/...`), and stacked delivery (`*_STACKED.1fps.<first>-<last>.mp4`) inspection:

1. **Automatic `.1fps.` generation ([`bench/_lib/frames.py::OneASecond`](../../../bench/_lib/frames.py))**:
   - [`bench/capture_masked_run.py`](../../../bench/capture_masked_run.py) (invoked with `--one-a-second` by [`bench/load_chain.py`](../../../bench/load_chain.py)) and [`bench/stack_delivery.py`](../../../bench/stack_delivery.py) (invoked with `--one-a-second` by [`bench/delivery_table.py`](../../../bench/delivery_table.py)) automatically encode `<stem>.1fps.<first:05d>-<last:05d>.mp4` alongside every `24 fps` clip.
   - Each source frame is held for `1.0s` (`-r 1`, no audio, `-crf 18`) in parts capped under `bench/_lib/frames.py::ONE_A_SECOND_BYTES` (`80_000_000` bytes, under `read_file`'s `100 MB` ceiling).
2. **Every-frame visual QA**:
   - Call `read_file` on `<stem>.1fps.<first>-<last>.mp4`. Because `read_file` samples video at `1 fps` midpoints (`t ≈ N + 0.5s`), timestamp `00:00` maps `1:1` to source frame `first`, `00:01` to `first + 1`, through `00:(N-1)` to `last` with zero boundary drift.
3. **Audio and real-time cadence QA**:
   - Call `read_file` on the companion `24 fps` `.mp4` (`<stem>.mp4` or `<stem>_stacked.mp4`) to inspect audio sync, vocal delivery, and real-time motion cadence.

---

## 2. Global `gemini-api` Rules (When Escalation Is Permitted)

1. **Flexible model selection (`-m "${GEMINI_VIDEO_MODEL:-gemini-3.8-flash}"`)**:
   - Omit `"model"` from `--body` JSON payloads and pass `-m "${GEMINI_VIDEO_MODEL:-gemini-3.8-flash}"` on the CLI so callers can switch models (`gemini-3.8-flash` default, `gemini-4-argon`, `gemini-flash-latest`, `gemini-3.6-flash`, `gemini-3.5-flash-lite`, or future models in `gemini-api models`) without editing JSON templates.
2. **Built-in credential store (`gemini-api configure` / `gemini-api auth login`)**:
   - Credentials live in `gemini-api`'s OS keychain or user config (`--no-keyring`), checked via `gemini-api whoami`, or in `GEMINI_API_KEY`. Never reference key file paths, run `cat` on secret files, or pass raw keys inline. If `gemini-api whoami` reports `--api-key [unset]`, stop and ask the user to run `gemini-api configure` or `gemini-api auth login`.
3. **Delete remote `Files API` uploads immediately after use**:
   - Local files under `~14 MB` (`<20 MB` base64) can be sent inline with zero cleanup. Uploaded `files/<id>` objects persist for `48h` against the `20 GB` project quota unless deleted with `gemini-api files delete "$FILE_ID"`.
