# Render times after the GPU link went from PCIe 4.0 x8 to x16 (2026-11-18)

**Question (from the dotfiles session, with the owner's approval).** The card was reseated on 2026-11-18 and its link
went from x8 to x16. Every earlier load and stage-swap time in this pack was taken at x8. What does x16 change in a
real H3 render?

**Answer.** On the three graphs of `2026-11-17_h3_encoder_streaming.md`, each warm render finished 1.2 to 2.0 s
sooner. All of it came before sampling.
- **Before sampling (the encoder pass and the DiT's return to the card)**, the link carried 20 to 22 GB/s at its
  peaks (`nvidia-smi dmon`).
- **Sampling did not change.** In steady sampling the card sat at 100% SM and 350 to 360 W, the 360 W cap, with
  almost no PCIe traffic. At 141 frames the 19.5 GiB DiT stays on the card. Sampling here is bound by the power
  cap, not the link.
- **Longer clips and ref2va were not measured.** Their larger activations could push part of the DiT off the card,
  where the link would matter every step.

## Conditions

- **Card:** RTX 4090, PCIe 4.0 x16. `nvidia-smi`: width 16, max 16. It idles at gen 1 and trains up under load.
- **Power cap:** 360 W since 2026-11-04 (`nvidia-smi` `power.limit`). This record and the x8 record it compares
  against share it.
- **Memory:** 4000 MT/s. That is the dotfiles session's figure, not checked here; a BIOS change is pending, so
  numbers bound by host memory may move again.
- **CUDA toolkit pin 13.5** (`<comfy>/start.sh`) since 2026-11-18. torch 2.15.0+cu134.
- **ComfyUI 0.45.0** (core `0263e4880`), comfy-kitchen `0.2.41+sol.b63d8c7.up.7d6a22a`, comfy-aimdo 0.6.0.
- **Server:** `comfyui.service`, `start.sh default`, started 11:44 Central with no `--fast`. Dynamic VRAM on.
  Started after a reboot, so the page cache was cold.
- **The x8 record ran with `--fast fp16_accumulation`.** The service journal shows that flag on every launch from
  2026-11-17 11:25 to the end of that day. Today's server had it off. The flag cannot reach the DiT, which computes
  in bf16 or int8 (`supported_models.py`'s `MiniMaxH3` lists bf16 and fp32 only). It does reach the text encoder,
  which core runs in fp16. So the per-render totals below carry both changes, while the step times carry only the
  link. `2026-11-18_vae_fp16_accumulation.md` measures the flag alone.

## Method

Same graphs and scheme as `2026-11-17_h3_encoder_streaming.md` section 1, driven by a throwaway harness (session
scratchpad, not kept):
- **Graphs:** `image/h3_image_t2i_api.json`, `image/h3_image_edit_api.json`, and `h3_text_to_video_pika_api.json`
  at length 141.
- **Runs:** one warm-up and two measured runs each, with a sentence appended to the prompt every run so the encoder
  ran each time.
- **Times:** the server's "Prompt executed" line, and tqdm's sampler line from `GET /internal/logs/raw`.
- **Link traffic:** `nvidia-smi dmon -s puct -d 1` for the whole session.

The x8 figures are the same lines in `journalctl -u comfyui.service` for the 11-17 record's runs: pid 291927, 13:48
to 14:01 Central, `keep_in_ram` off.

Data: `data/results/2026-11-18_pcie_x16_render_times.json`, holding the runs, their stage and progress lines, and
the `dmon` log.

Each warm-up paid a cold page cache today and is not compared. The Pika warm-up came from an earlier harness that
was cut off; the server finished it (72.37 s), and the next submission of that same graph was a cache hit (0.00 s).

## 1. Per render

"Prompt executed", in seconds, warm runs:

| graph | x8, 2026-11-17 | x16, 2026-11-18 |
|---|---|---|
| `image/h3_image_t2i_api.json` | 27.30, 27.12 | 25.66, 25.90 |
| `image/h3_image_edit_api.json` | 30.44, 30.47 | 28.73, 28.46 |
| `h3_text_to_video_pika_api.json`, 141 frames | 67.35, 67.50 | 65.77, 66.17 |

The 11-17 record lists its third Pika run as stopped. The journal shows it completing (67.50 s), so it is used here.

## 2. Sampling steps

tqdm's whole-pass rate, today:
- text to image: 2.65 to 2.67 s/it;
- edit: 3.24 to 3.25 s/it;
- Pika at 141 frames: 7.73 to 7.77 s/it.

No x8 tqdm line survives: the journal does not keep the bar. The closest x8 figure is
`2026-11-17_pika_sampler_and_t2i_vision_probes.md`:
- about 2.67 s/step (image) and 7.85 to 7.93 s/step (Pika);
- read from WebSocket intervals on steps 2 onward, with a tiny-VAE preview decoded every step. That is a
  different method with extra work per step, so it is context, not a comparison.

## 3. Where the link is busy

`dmon` samples PCIe throughput over a short window once a second, so its figures are rates at an instant, not
byte counts. Pika's second measured run (start 11:54:39):

| phase | time | receive (`rxpci`) | SM, power |
|---|---|---|---|
| encoder pass | 11:54:41 to :44 | 8.3, 21.8, 14.3 GB/s | 89 to 100%, 140 to 200 W |
| sampling, step 1 | :45 to :52 | 0, then a 12.6 and 7.3 GB/s burst at :51 and :52 (the DiT's return) | 100%, 279 to 361 W |
| sampling, steps 2 to 6 | :53 to 11:55:35 | 0 to 0.6 GB/s, one 2.3 GB/s second | 100%, 342 to 360 W |
| decode | :36 to :42 | under 0.02 GB/s; 3.6 GB/s out to the host at :40 | 100% |

The two image graphs show the same pattern: peaks of 21 to 22 GB/s before sampling, and steady sampling at the
cap.

## 4. What this cannot see

- **The flag confound** in section 1, above.
- **The split of the 1.2 to 2.0 s** between the encoder pass and the DiT's return. The server logs no stage lines
  on a warm run under dynamic VRAM, and `dmon`'s one-second samples cannot divide it.
  `2026-11-18_text_encoder_int8_vs_bf16_speed.md` times the encoder node itself.
- **One machine, one seed per graph, two measured runs.**
- **Clips longer than 141 frames, ref2va, and masked_v2v windows were not run.** Those are where the DiT could
  stream every step.
