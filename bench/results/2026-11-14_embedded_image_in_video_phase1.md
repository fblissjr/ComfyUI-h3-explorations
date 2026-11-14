# Embedded Image Generation & Editing in H3 Video Workflows — Phase 1 Live Bench, 2026-11-14

**Question.** On a reloaded ComfyUI server (`aa943b17`+) running the same zero-swap `P-H3-Image LoRA` (`8 -> 4 MP` `16:9`) + 6-step Pika `fl2va` (`1344x768`, `length = 141` / `5.875s` at `24 fps`) resident setup from [the Phase 0B bench](2026-11-14_embedded_image_in_video_phase0b.md):

1. Does [`MiniMaxH3ReleaseReferences`](../../control_vector_nodes.py) (`guide_end_percent`) smoothly release an edited mid-clip keyframe anchored via [`MiniMaxH3GuideLatent`](../../guide_latent.py) at `frame_idx = 68` (`4 * 17`, `2.83s`) early enough to preserve trajectory freedom while transferring the rainy night + lit-lamp look?
2. How much does aligning the prompt (`MIDCLIP_GUIDE_PROMPT`, describing the mid-clip transition and evening rain in `overall_soundscape`) affect post-guide visual persistence and audio compared with neutral `I2VA_PROMPT`?
3. Does [`MiniMaxH3EditGuider`](../../edit_guidance.py) (`scale = 1.65`, `last_step = 2`, `streams = "independent"`, `audio_scale = 1.0`) auto-release cleanly from 2-pass contrastive guidance onto 1-pass `[original]` via [`_keyframes_extend_prefix`](../../edit_guidance.py#L115-L129) (`_release_to_original_after`) on live `fl2va` sampling?

Runners and local metric artifacts live in gitignored [`data/embedded_image_video_2026-11-14/run_phase1_guide_release_ab.py`](../../data/embedded_image_video_2026-11-14/run_phase1_guide_release_ab.py) and [`data/embedded_image_video_2026-11-14/analyze_phase1.py`](../../data/embedded_image_video_2026-11-14/analyze_phase1.py).

---

## 6-Step `_step_in_window` Quantization Finding (`0.34` -> `0.33`)

Inspecting [`control_vector_nodes.py::_step_in_window`](../../control_vector_nodes.py#L1152-L1188) alongside `workflows/h3_config.py::PIKA_MANUAL_SIGMAS` (`1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0`, `n_steps = 6`) showed that `step_idx = 2` evaluates at `frac = 2 / 6 = 0.333333...`. Any threshold above `1/3` (such as `0.34`) activates steps `[0, 1, 2]` (`3/6`), making it **identical** to `guide_end_percent = 0.50` (`[0, 1, 2]`). Lowering the early-release arm to **`guide_end_percent = 0.33`** cleanly cuts off at `[0, 1]` (`2/6` steps), releasing mid-clip guide rows at step `2` (`sigma_v = 0.923077`, unshifted midpoint `b = 0.50`). Combined with the matched Phase 0B `i2va` control (`0/6`), this gives a clean `6/6 -> 3/6 -> 2/6 -> 0/6` step dose curve on identical `seed = 4301` and `I2VA_PROMPT`.

---

## Live `:8188` Telemetry & Row-Release Verification

- **Preflight & Zero-Swap Runtime:** All 5 Phase 1 graphs passed `bench/preflight_graph.py::grade` (`0 FAIL` per arm) and completed on `http://127.0.0.1:8188` under one shared `client_id` (`Swap = 0 B`, host `free -b` after analysis: `used = 50,686,078,976 B`, `available = 84,122,001,408 B`).
- **Exact `guide_end_percent` Row Release (`PackedLayout`):** Live `/internal/logs/raw` telemetry confirmed mid-clip guide dropping with endpoints intact on Arms 2–4:
  - Arm 2 (`guide_end_percent=0.50`): `released 1008 mid-clip guide cond rows while preserving endpoints & refs (PackedLayout seq_len 46063 -> 45055, 3D RoPE positions preserved)` (`72.07 s` vs full-pin `164.13 s` initial warm/compile pass).
  - Arm 3 (`guide_end_percent=0.33`): `released 1008 mid-clip guide cond rows while preserving endpoints & refs (PackedLayout seq_len 46063 -> 45055, 3D RoPE positions preserved)` (`68.06 s`).
  - Arm 4 (`guide_end_percent=0.50`, `MIDCLIP_GUIDE_PROMPT`): `released 1008 mid-clip guide cond rows while preserving endpoints & refs (PackedLayout seq_len 46104 -> 45096, 3D RoPE positions preserved)` (`82.08 s`, includes 32B Qwen3-VL text encode).
- **`EditGuider` Prefix-Extension Auto-Release:** Arm 5 logged `edit guidance: streams=independent (video=1.65, audio=1) across steps 0..2; text length 1241 (original) and 1241 (edited) tokens; keyframes 1 (original) and 2 (edited)` and finished in **`112.08 s`** (`+40–44 s` over single-pass Arms 2/3 for `3` extra contrastive DiT passes on steps `0..2`, with steps `3..5` falling back to 1-pass `[original]`). *(Note: `sol.n_evals = 6` in provenance sidecars records scheduler steps `sigmas[:-1].numel()` per [`provenance.py`](../../provenance.py#L283-L286), not guider forward-pass multiples.)*

---

## Quantitative Comparison (`141` Frames, `1344x768`, `frame_idx = 68`, `seed = 4301`)

Reference targets are the cached Phase 0B `P-H3-Image` `8 -> 4 MP` stills (`t2i_first_frame_00001_.png`: luma `95.98`, `B - R = -38.07`; `edit_second_frame_00001_.png`: luma `53.45`, `B - R = +2.18`; `B vs A` MAE `51.32`).

| Arm (`seed = 4301`) | Wall (s) | MP4 Size (B) | `f0 -> A` MAE | `f68 -> B` MAE (PSNR) | `f140 -> B` MAE (PSNR) | Closest to `B` (`idx`: MAE) | `67->68 / 68->69` Adj MAE (`[60..76]` / Clip Mean) | `B - R` (`f0..67` / `f68` / `f69..140`) | Audio RMS / Peak (Active) |
|---|---:|---:|---:|---:|---:|---|---|---:|---|
| *Phase 0B Ref:* `i2va` Control (`0/6`) | `82.06` | `3,337,552` | `17.31` | `53.31` (`10.46 dB`) | `64.99` (`9.16 dB`) | `f52`: `52.16` | `4.20 / 4.61` (`n/a` / `2.66`) | `-38.14 / -36.92 / -36.38` | `-38.12 / -0.40 dBFS` (`0.0763`) |
| **Arm 1:** `guide_end = 1.00` (`6/6`) | `164.13` | `3,182,183` | `17.19` | **`5.03`** (**`29.11 dB`**) | `34.09` (`13.07 dB`) | `f68`: `5.03` | `1.39 / 1.23` (`1.26` / `2.11`) | `-19.22 / +0.91 / -4.71` | `-40.65 / -1.29 dBFS` (`0.0795`) |
| **Arm 2:** `guide_end = 0.50` (`3/6`) | `72.07` | `3,376,510` | `17.27` | `10.54` (`22.40 dB`) | `35.10` (`12.97 dB`) | `f68`: `10.54` | `1.60 / 1.56` (`1.37` / `2.17`) | `-19.44 / +0.13 / -4.82` | `-40.09 / -0.48 dBFS` (`0.0864`) |
| **Arm 3:** `guide_end = 0.33` (`2/6`) | `68.06` | `3,292,638` | `17.31` | `16.68` (`18.90 dB`) | `37.10` (`12.69 dB`) | `f75`: `16.41` | `1.56 / 1.50` (`1.50` / `2.27`) | `-18.48 / +3.23 / -2.25` | `-40.98 / -4.94 dBFS` (`0.0776`) |
| **Arm 4:** `0.50` + `MIDCLIP_GUIDE_PROMPT` | `82.08` | `3,323,767` | `17.29` | `10.55` (`22.36 dB`) | **`33.88`** (**`13.19 dB`**) | `f71`: `10.50` | `1.29 / 1.31` (`1.20` / `2.09`) | `-19.49 / +0.46 / -3.82` | `-37.95 / -0.00 dBFS` (**`0.3712`**) |
| **Arm 5:** `EditGuider` (`s=1.65`, `last=2`) | `112.08` | `3,501,598` | `17.26` | `10.22` (`22.81 dB`) | `35.31` (`12.86 dB`) | `f72`: `10.15` | `1.30 / 1.34` (`1.29` / `2.31`) | **`-15.30 / +1.65 / -2.97`** | `-40.67 / -0.69 dBFS` (`0.0729`) |

*Saved videos under `$H3_OUTPUT_DIR/Video/embedded_image_video_2026-11-14/{phase1_guide_end_100,phase1_guide_end_050,phase1_guide_end_033,phase1_guide_end_050_aligned_prompt,phase1_edit_guider_scale165_last2}_00001-audio.mp4`; provenance stamps under `$H3_OUTPUT_DIR/provenance/stamp_20261114T212410+0000_5505c623.json` through `..._f25b3fae.json`; `.1fps.00000-00140.mp4` review strips alongside them under `$H3_OUTPUT_DIR/Video/embedded_image_video_2026-11-14/`.*

---

## Native Media & Audio Observations (`read_file` on `.1fps` and `24 fps` `-audio.mp4`)

1. **Smooth dose-response with zero boundary pop (`67->68->69`):** Unlike single-step artifacts, dropping mid-clip `cond` rows after step `1` (`0.33`) or step `2` (`0.50`) creates no visual tear or snap at `frame 68` (`boundary_adj_mae` `1.29–1.60` vs clip mean `2.09–2.31`). Visually, `guide_end = 1.00` locks the clockmaker's forceps/hand tightly onto `image_b`'s raised placement right through `f68` before swinging down afterward, whereas `0.50` and `0.33` allow visibly freer vertical motion of the balance wheel and earlier relaxation of the hand downward while preserving the blue storm window and lit banker's lamp.
2. **Dark storm ambience persists past release (`f68 -> f102` and `f140`):** Even with `guide_end_percent = 0.33` (`2/6` steps) and `I2VA_PROMPT`, `f85` and `f102` stay cool indigo night (`B - R` `+0.25` to `+2.98`). Between `f119` and `f140`, `mean_r` rises modestly (`64..69`, `B - R` `-10.59` to `-13.57`) mainly because the lit amber oil lamp casts stronger warm tungsten spill across the clockmaker's reaching right arm and foreground desk, while the left paned window stays cool dusk slate-blue rather than reverting to golden afternoon sunlight (`f0 luma 93.4` vs `f140 luma 57.5–61.6`).
3. **Why prompt alignment matters (`Arm 4` vs `Arm 2`):** Identical latent conditioning (`guide_end_percent = 0.50`) yields virtually identical `f68 -> B` visual match (`10.54` vs `10.55`), yet `MIDCLIP_GUIDE_PROMPT` keeps `f140` slightly cooler/closer to `image_b` (`33.88` vs `35.10`) **and** drives a `4.3x` jump in audio activity (`active_frac = 0.3712` vs `0.0864`), natively fading in steady evening rain against the windowpanes at `00:01–00:03` followed by sharp bench tool clicks at `00:03–00:05`. A visual-only `GuideLatent` cannot write the `overall_soundscape` stream.
4. **`EditGuider` (`Arm 5`) pulls transition onset earlier with clean audio isolation:** Contrastive scaling (`scale = 1.65`, `last_step = 2`) shifts the dusk/rain transition noticeably earlier (`f34 luma = 70.95, B - R = -11.58` vs `77.79, -19.49` in Arm 2; `f51` already at night `B - R = +1.00`) while giving the coolest clip-wide `B - R` (`-8.88`) and clean unclipped tool-click audio (`streams="independent", audio_scale=1.0`, `active_frac = 0.0729`), at a cost of slight extra floating bright embers/dust motes and `+40 s` over 1-pass `guide_end = 0.50`.

---

## Practical Selection Rule (Phase 0B + Phase 1 Combined)

| Need | Best Pattern Measured | Notes |
|---|---|---|
| Full convergence onto an edited ending (`t = length - 1`) + rain soundscape | **Phase 0B Arm 1 (`fl2va` bookends + `FL2VA_PROMPT`)** | Reaches `MAE = 5.11` (`28.98 dB`) at `f140` in 1 pass |
| Mid-clip visual beat at `17m` without late trajectory pin | **`MiniMaxH3GuideLatent(17m)` + `MiniMaxH3ReleaseReferences(guide_end_percent = 0.33..0.50)`** | Align prompt lighting & `overall_soundscape` if mood changes (`68–82 s` on 1 pass) |
| Exaggerated mid-clip look onset without re-pinning late motion | **`MiniMaxH3EditGuider(scale ~ 1.65, last_step = 2, streams = "independent")`** | Auto-releases to 1-pass `[original]` via `_keyframes_extend_prefix` (`112 s`) |
| Continuous lighting transfer without touching `first_frame` | *Do not use single-pair `MakeMediaControlVector(pooling="scene_text")`* | Measured `raw_norm = 5.999 < 14`; supply a latent target or multi-pair style dataset |
