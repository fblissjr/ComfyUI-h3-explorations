# Pika H3 Flex `ref2va`: `size_policy="tiered"` vs. `max(allow_upscale=False)` on a `1024x1024` portrait (2026-11-22)

lane: distill
verdict: tiered keeps clean identity and slightly crisper floral-blazer and gallery detail at +1.89 s/step (+22.2% across steady-state steps 2-6) and +250 MiB peak VRAM under qwen_view="separate"; commit 99034acc is safe to keep

**Question.** Commit `99034acc` switched the 11 shipped Pika `ref2va` graphs from `size_policy="max", allow_upscale=False`
to `size_policy="tiered"` (matching Pika's release and sglang `#51935`), with no Pika render run yet to price the extra
reference rows or inspect the picture. On a `1024x1024` still under our shipped `qwen_view="separate"` (`512`), what
does `tiered` (`2048x2048` DiT rows) cost in sampler time and VRAM, and does identity hold cleanly?

**Answer.**
- **Steady-state sampler cost:** `8.53 s/step -> 10.42 s/step` across warm steps 2–6 (`+1.89 s/step`, `+22.2%`, `~+10 s`
  per 6-step `141f` render), even though total packed length grows only `+6.8%` (`44,861 -> 47,933`), because shipped
  graphs pass `rows.reference_rows="exact"` on `MiniMaxH3Sol`. Total wall clock tied (`86.38 s` vs. `86.35 s`) only
  because arm A ran cold after `/free` (`MiniMaxH3EncodeReferences` `14.88 s` vs. `5.66 s` warm).
- **Peak VRAM:** `23,872 MiB -> 24,122 MiB` (`+250 MiB`), cleanly inside the 24 GB RTX 4090 at `141` frames.
- **Picture & sound (inspected directly on MP4s, contact sheet, and `f070` 3-up crop):** Both arms preserve `<Picture 1>`'s
  facial features, pale silver-grey hairline streak, white tee, and pink-and-white floral blazer, walk smoothly down
  the sunlit travertine gallery, and deliver clean speech (`"We actually locked the east wing an hour ago."`) with
  synced lips and footstep foley (zero mid-clip garble). `tiered` renders slightly sharper blazer weave/lapel edges and
  background detail (`Laplacian 71.42` vs. `61.44`; `DINOv3 ref_patch_cos_mean 0.393` vs. `0.383`).
- **Scope note:** Tested on 1 scene (`141f`) at the owner's direction; `362f` (`107,856` video rows) and multi-still
  Pika loads were not rendered here.

---

## The runs

**Graph.** Base `workflows/h3_image_ref_plus_text_to_video_pika_api.json` (`minimax_h3_ref2va_pruned_int8_convrot.safetensors`,
`h3/pika_h3_flex_ref2va_svd995_bf16.safetensors` on `MiniMaxH3LoRABranch`, `MiniMaxH3SigmaShift(12.0, 3.0)`,
`ModelAttentionBackend("comfy kitchen attention")`, `MiniMaxH3Sol(dense_blocks="49", rows.reference_rows="exact",
rows.video_reference_rows="routed")`, `lcm`, 6-step `ManualSigmas`, `MiniMaxH3EncodeReferences` +
`MiniMaxH3PromptOnReferences`, plus `PreviewAny` on `MiniMaxH3Preflight` output `3`), run sequentially on an idle
`:8188` after `POST /free`.

- **Reference still:** `t2i_29_edit_reference_portrait_seed7.png` (`1024x1024` portrait in `$H3_INPUT_DIR`).
- **Canvas & seed:** `1344x768`, `length = 141` (`39 + 51 * 2`, `5.875 s` at `24 fps`, `42` video latent frames),
  `noise_seed = 730451892`.
- **Encoder view (shared across both arms):** `qwen_view = "separate"`, `qwen_view.qwen_short_edge = 512` (`512x512`,
  `258` vision tokens in `MiniMaxH3Preflight`).
- **Only node 50 (`MiniMaxH3AppendRefImage`) changed:**
  - `max_no_upscale`: `size_policy = "max"`, `size_policy.dit_short_edge = 2048`, `size_policy.allow_upscale = False`
    -> `1024x1024` (`1,024` DiT ref rows).
  - `tiered`: `size_policy = "tiered"` -> `2048x2048` (`4,096` DiT ref rows, `+3,072`).

---

## Packed tokens, timings, and VRAM

Recorded from live `MiniMaxH3Preflight` UI output, `:8188` WebSocket `executing`/`progress` events, and `0.25 s`
`nvidia-smi` polling (`data/pika_tiered_ab_2026-11-22/summary.json`):

| metric | `max_no_upscale` (`1024x1024`) | `tiered` (`2048x2048`) | delta |
|---|---:|---:|---:|
| DiT ref rows | `1,024` (`2.3%`) | `4,096` (`8.5%`) | `+3,072` (`4.0x`) |
| `qwen_vision` / `text` / `video` / `audio` | `258` / `1,031` / `42,336` / `470` | `258` / `1,031` / `42,336` / `470` | `0` |
| **Total packed sequence** | **`44,861`** | **`47,933`** | **`+3,072` (`+6.8%`)** |
| `MiniMaxH3EncodeReferences` | `14.879 s` (cold after `/free`) | `5.657 s` (warm) | `-9.222 s` |
| `MiniMaxH3PromptOnReferences` | `2.589 s` | `2.576 s` | `-0.013 s` |
| `SamplerCustomAdvanced` total | `57.994 s` | `67.953 s` | `+9.959 s` (`+17.2%`) |
| Step 1 (incl. GPU model prep) | `14.930 s` | `15.345 s` | `+0.415 s` |
| **Steady-state steps 2–6** | **`42.635 s` (`8.527 s/step`)** | **`52.100 s` (`10.420 s/step`)** | **`+9.465 s` (`+1.893 s/step`, `+22.2%`)** |
| `VAEDecode` + `VAEDecodeAudio` + mux | `10.078 s` | `10.056 s` | `-0.022 s` |
| Total wall clock | `86.378 s` | `86.345 s` | `-0.033 s` |
| **Peak GPU VRAM** | **`23,872 MiB`** | **`24,122 MiB`** | **`+250 MiB`** |

---

## Visual, DINOv3, and audio readouts

| metric | `max_no_upscale` | `tiered` | notes |
|---|---:|---:|---|
| Laplacian sharpness (mean over every 10th frame) | `61.44` | `71.42` | `+16.2%` on `tiered` |
| DINOv3 `ref_cos` (`f0 / mean / max`, 13 frames) | `0.319 / 0.287 / 0.331` | `0.294 / 0.240 / 0.318` | `bench/probe_dinov3_copy_freeze.py` |
| DINOv3 `ref_patch_cos` (`mean / max`) | `0.383 / 0.397` | `0.393 / 0.423` | `triage_notes: []` on both |
| DINOv3 `temporal_self_cos` (`mean / min`) | `0.9642 / 0.9314` | `0.9701 / 0.9490` | smooth motion on both |
| Audio (`32 kHz` stereo, `5.856 s`, `rms_dbfs` / `peak`) | `-25.45 dBFS` / `0.6264` | `-23.88 dBFS` / `0.7572` | clean dialogue & footsteps |
| Inter-arm RGB distance | `MAE = 21.30`, `PSNR = 17.18 dB` | — | mostly column/pedestal shift |

---

## Files

- Output media under `$H3_OUTPUT_DIR/Video/pika_tiered_ab_2026-11-22/`:
  - `portrait_1024_max_no_upscale_00001-audio.mp4`
  - `portrait_1024_tiered_00001-audio.mp4`
  - `portrait_1024_ab_contact_sheet.png` (frames `0, 35, 70, 105, 140`, top `max_no_upscale`, bottom `tiered`)
  - `portrait_1024_f070_center_3up.png` (`<Picture 1>`, `f070 max`, `f070 tiered`)
- Scrubbed run log and preflight/DINOv3 readouts: `data/pika_tiered_ab_2026-11-22/summary.json`.

## What this cannot see

- One scene, one `1024x1024` still, one `141`-frame length, one seed (`730451892`).
- Smaller stills (`384x288`, where `tiered` enlarges ~50x to `2720x2048` = `5,440` DiT rows), `362`-frame Pika clips,
  or multi-reference loads (where `+1.89 s/step` per enlarged still and `+250 MiB` headroom matter more).
