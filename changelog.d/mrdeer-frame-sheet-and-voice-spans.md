bump: minor

### Added

- `bench/_lib/frames.py`: one reader of frames and masks by frame number for the tools that set a render beside its source (`probe`, `stream`, `read_mask`, `write_mask_video`, and `FIT`, the loader's fit). `bench/capture_masked_run.py` now imports it in place of its own copy.
- `bench/frame_sheet.py`: a sheet of numbered frames from several videos on one clock, with a fixed crop or one that follows a mask, and masks outlined. Every "as seen in stills" line is written from such a sheet and every session that read a render had written its own tool for it.
- `bench/voice_spans.py`: where a voice is on a clip's track by video frame, on this machine only (`separate` with torchaudio's Hybrid Demucs from weights already in the torch hub cache, never fetched; `spans`; `cut`, a window's stretch of the vocals stem to load in place of the clip's track). It gives the same spans and the same cut as the session scripts the day's lip-sync and voice findings rested on, and prints its own control.
- `bench/check_frame_sheet.py` and `bench/check_voice_spans.py`, with their rows in `docs/checks.md`.
