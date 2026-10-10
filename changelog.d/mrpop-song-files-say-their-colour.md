bump: patch

### Fixed

- The song node's video files are BT.709 and say so. `audio_freeze_song.py` piped rgb frames to ffmpeg and named no matrix, so every render and mask review held BT.601 values with no colour tag, and a player that takes BT.709 for a picture of that size showed the whole frame off in colour beside its own source, kept pixels included. The writer now converts with `loop_output.TO_BT709`, labels primaries and transfer with `SAY_BT709`, and tags the file with `BT709_TAGS`, as the VHS writer does; in the review's stacked form the drawn half is converted before the stack. A file read back by the video loader gives the same rgb as before. `bench/check_audio_freeze.py` item 7 holds all four tags and that a file's values match its tag, and fails on the old writer. Files written before and after this change must not be joined by stream copy. Found and written by session mrcorn. `INDEX.md` is rebuilt for a wiki page added in 0.235.3.
