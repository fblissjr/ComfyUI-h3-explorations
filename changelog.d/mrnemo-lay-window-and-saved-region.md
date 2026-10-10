bump: minor

### Added

- A masked render saves what each window's composite was run with: `<prefix>_windows/<name>_window_N_region.npz`, beside the window's latent, holds the window's fitted mask (bits), its token region, its margin, where it starts in the source, its trim and the source's composite settings and cuts (`video_mask.save_window_region`, `load_window_region`, `loop_resume.region_path`). With the stored latent and the source's own frames that is everything the composite takes, so a finished render can be laid again off the server, exactly. A render whose tracker ran inside its own graph used to leave its mask nowhere (`data/CAPTURE_GAPS.md`, gap 22). The file is written whole or not at all, a stale one goes with the stale latent, `keep_windows` off removes it, and a failure to write it is a line of the report and never the render's. Its size and write time on a real window's mask are in the commit message.

### Changed

- The song node's composite is one function, `video_mask.lay_window` (the weight under either `composite`, the cut gate, the blend, the report's lines), and the node calls nothing else for it. The arithmetic is unchanged. It exists so that a bench tool can lay a saved window with the code the render ran, not a copy of it.

### Fixed

- The song node's report opened with a frame number where its count of reused windows belongs on any masked run that composited a window: 0.254.0's gate assigned the window's first frame to `first`, the count the report's first line and the joined mask review read. The review's own use only mattered when a window's review file was missing. The lift takes the assignment out.

### Checks

- `bench/check_video_mask.py`, item 20: `lay_window` is the pieces' own answer under both `composite` choices, a frame across a cut is the source bit for bit and no other frame moves, the lines are the report's; the region file reads back as written with one margin and with each frame's own, and a window laid from the file is the window laid from the tensors; the song node lays through `lay_window` alone, saves the region, removes a stale one and assigns `first` once. Seen red with the gate taken out of `lay_window` and with the region read back a step late.
