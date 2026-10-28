# Path 4: Audio & Active-Speaker Transcription (`gemini_media.py transcribe-spans` & `gemini-api transcribe`)

Companion reference to [`../SKILL.md`](../SKILL.md).

## 0. Native-First Gate (`agy` Check Before Using Path 4)

- **Always use local [`bench/voice_spans.py`](../../../../bench/voice_spans.py) + `agy`'s built-in `read_file` tool on `<100 MB` clips first (`$0` API spend)**:
  1. Run [`bench/voice_spans.py`](../../../../bench/voice_spans.py) (`BS-RoFormer-SW` vocal stem isolation + `silero_vad` at `16 kHz`) to get exact `[start_frame, end_frame]` voiced spans on the clip's audio track.
  2. Call `read_file` on the `<100 MB` `24 fps` `.mp4` (and `.1fps.<first>-<last>.mp4`) to hear the audio and visually see which on-screen subject is singing or speaking.
- **When Path 4 (`gemini_media.py transcribe-spans` or `gemini-api transcribe`) IS permitted**:
  1. The video exceeds `read_file`'s `100 MB` ceiling, or
  2. You specifically need structured multi-speaker JSON (`speaker`, `start_frame`, `end_frame`, `text`) with `24 fps` visual speaker attribution (`--visual-speakers`) across a multi-person shot to fill [`bench/voice_spans.py`](../../../../bench/voice_spans.py)'s `cannot_see` gap (`L27–30`: *"which on-screen subject is singing or speaking, or whether a subject's mouth is visibly open while silent"*).
- **Owner media rule ([`docs/wiki/working_with_the_owner.md`](../../../../docs/wiki/working_with_the_owner.md))**: Always ask the owner and name the file before uploading project media via `gemini-api`.

---

## 1. Where Audio & Speaker Transcription Fits in `docs/masked_v2v/`

In [`docs/masked_v2v/AGENTS.md`](../../../../docs/masked_v2v/AGENTS.md) §7 and [`docs/masked_v2v/reference_and_prompt_rules.md`](../../../../docs/masked_v2v/reference_and_prompt_rules.md):
- The unified `<Subject 2>` prompt ([`bench/load_text.py`](../../../../bench/load_text.py)) includes `MOUTH_CLAUSE` only when `mouth_drawn_open` is true, and uses `TRANSFERRED_DRAWN_NO_MOUTH` / `FOLLOWS_DRAWN_NO_MOUTH` when `mouth_drawn` is false on every frame.
- [`bench/voice_spans.py`](../../../../bench/voice_spans.py) and `open_frames` track when a vocal stem is active, and its docstring (`L27–30`) explicitly notes what it **cannot see**: which on-screen subject is singing when multiple people are in the shot.
- Using `scripts/gemini_media.py transcribe-spans` converts `MM:SS(.mmm)` transcript segments directly into `24.0 fps` `[start_frame, end_frame]` spans (`shot_first + round(t_seconds * 24.0)`).

---

## 2. Frame-Aligned Speaker Spans (`scripts/gemini_media.py transcribe-spans`)

```bash
# Audio-only or multimodal (--visual-speakers) speaker span extraction mapped to 24.0 fps shot frames:
"$PY" .gemini/skills/video-understanding/scripts/gemini_media.py transcribe-spans "$CLIP_MP4" \
  --shot-first 4912 --shot-last 4946 \
  --visual-speakers \
  --prompt "Identify which on-screen subject (lead singer vs. second subject) is singing or speaking and transcribe the lyrics."
```

- Without `--visual-speakers`, `transcribe-spans` runs audio-focused timestamp transcription.
- With `--visual-speakers`, `transcribe-spans` passes `"type": "video"`, `"resolution": "high"`, and `"processing": {"type": "static", "fps": 24}` (plus `start_offset`/`end_offset` if slicing a master cut) so the model sees lip movement at `24 fps` while transcribing audio.
- **Timestamp-to-frame formula (`24.0 fps`)**:
  $$\text{frame} = \text{shot\_first} + \text{round}(t_{\text{seconds}} \times 24.0)$$

---

## 3. Direct CLI Transcription (`gemini-api transcribe`)

For standalone audio/video files where frame-number conversion is not needed, `gemini-api transcribe` outputs `json`, `srt`, `vtt`, or `text` directly:

```bash
# JSON transcript with speaker labels and timestamps:
gemini-api transcribe -i "$CLIP_MP4" \
  -m "${GEMINI_VIDEO_MODEL:-gemini-3.8-flash}" \
  --format json \
  --out /tmp/shot_transcript.json

# Or SRT captions:
gemini-api transcribe -i "$CLIP_MP4" \
  -m "${GEMINI_VIDEO_MODEL:-gemini-3.8-flash}" \
  --format srt \
  --out /tmp/shot_transcript.srt
```
