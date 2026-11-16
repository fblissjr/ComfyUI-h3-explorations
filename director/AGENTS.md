# Interactive Spatial Director (`director/`)

Standalone desktop web application backend, ComfyUI route extension, runtime workflow compiler, and launcher for interactive `[F, H, W]` spatial editing, D-Pad cutouts, and 3-frame audio cues on MiniMax H3 / Pika H3 Flex.

## Scope & Ownership

- `__init__.py` exposes [`_register_director_routes`](../__init__.py) to register `POST /h3_spatial_director/forget` on ComfyUI's server without patching ComfyUI core.
- `director/routes.py` implements `compute_forget_plan`, `delete_temp_files`, `forget`, and `register` with lazy runtime imports (`aiohttp`, `folder_paths`, `server`), cleaning temp `"images"`, `"audio"`, and `"gifs"` plus orphaned `h3_spatial_director/` uploads while never touching `$H3_OUTPUT_DIR/Video/` (`workflows/h3_config.py::output_dir`).
- `director/compiler.py` compiles still anchors (`compile_still` from `workflows/image/h3_pika_image_t2i_api.json`, `workflows/image/h3_p_h3_image_t2i_api.json`, `workflows/image/h3_image_t2i_api.json`), SAM 3.1 segmentation (`compile_sam3_segment` with `MiniMaxH3SAM31Corrections` -> `SAM3_Detect`), spatial video takes (`compile_spatial_take`: `r2v_portal`, `i2v_portal`, `t2v_portal`, `world_targets`, including `combined` director mode, `first_frame` + `last_frame`, optional 2nd-pass `MiniMaxH3FrozenVideoCache` + `MiniMaxH3AudioRefineMask`), and standalone video-frozen audio re-dub (`compile_audio_redub`: `polish` `0.5x` base `BasicScheduler` matching `workflows/h3_config.py::AUDIO_REFINE` or `fresh` `1.0x` Pika `ManualSigmas`, with nested `LTXVConcatAVLatent.fit_audio`). It enforces `h3_rules.py::on_both_clocks`, computes offline `99,864`-token preflight (`compute_preflight_summary`), scales plans (`escalate_plan`), and dynamically injects `ModelPreviewOverrideKJ` (`"165"`) before guider `"9"` (`preview_fps=7` with `taeh3.safetensors` for video, `preview_fps=1` with `taeh3_image.safetensors` for stills) only when `/object_info` exposes it.
- `director/server.py` serves the Light-Mode CSP web UI (`/` and `/static/`, including modular `director/static/js/stage.js`, `director/static/js/sequencer.js`, `director/static/js/ab_compare.js`, `director/static/js/foley_synth.js`, and `director/static/js/app.js` with master `<video>` audio clock playback and multi-point/text SAM 3.1 refinement), validates `Host`/`Origin`/`Sec-Fetch-Site` and `HttpOnly; SameSite=Strict` session cookies, stores session state under `data/spatial_director_app/` (`--state-dir`), exposes all `/api/*` endpoints (including HTTP `Range`-capable `/api/view`, per-`clientId` `/api/ws`, `POST /api/segment`, `POST /api/extract_frame`, `POST /api/redub`, and MP4 video import as stage takes in `POST /api/upload`), and exports approved temp MP4s into `$H3_OUTPUT_DIR/Video/`.
- An optional local launcher (when symlinked at `director/studio.sh`) manages start/stop/status/url/logs via `h3.env` / CLI port configuration without touching ComfyUI `:8188`.

## Rules

1. Never use bare filenames in prose/citations; keep all paths repo-relative (no absolute user-home, mount paths, or owner names).
2. Draft takes write only to `temp/` (`"save_output": false` on `"13"` HUD and `"162"` clean `VHS_VideoCombine`, plus `PreviewImage` on anchor stills). Only explicit `POST /api/export` promotes an MP4 into `$H3_OUTPUT_DIR/Video/`.
3. Keep `director/routes.py` and `director/compiler.py` top-level imports strictly stdlib so `bench/check_imports.py` passes offline.
4. Keep spatial take lengths on `h3_rules.py::on_both_clocks` (`39, 90, 141, 192, 243, 294, 345`) and sound cues snapped to 3-frame (`125 ms`) boundaries.
5. `POST /api/context_suggest` returns `{ok: true, enabled: false, suggestions: []}` unless `H3_HEYLOOK_URL` is configured and reachable.
