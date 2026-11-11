# Sol `start_percent` and `dense_blocks` panel on 6-step Pika, Pass 1 (2026-11-11)

**lane:** sol
**verdict:** Pass 1 rendered and blinded (`pika_sol_pass1_2026-11-11`); awaiting owner playback verdict before Pass 2 or any default change.

## Why

The shipped `MiniMaxH3Sol` defaults (`start_percent=0.2`, `dense_blocks="38,39,40,41,42,49"`, `sink_conditioning="exact_kv_and_all_rows"`, `quantizer="rotated"`, `tau=1.0`, `token_routing="off"`) were set during the PDD8 + FlashGen campaign ([2026-10-02_sol_dense_blocks_panel.md](2026-10-02_sol_dense_blocks_panel.md)). Two parts of that history do not carry over to the 6-step Pika H3 Flex workflow (`workflows/h3_text_to_video_pika_api.json`):

1. **Schedule mapping at 6 steps (`shift_video=12.0`).** Pika's 6-step `ManualSigmas` grid evaluates the DiT at $\sigma \in \{1.0,\; 0.972973,\; 0.923077,\; 0.8,\; 0.571429,\; 0.333333\}$. Through `ModelSamplingAV(shift=12.0)`, `start_percent=0.2` maps to $\sigma_{\text{start}} = 0.979592$, so **only Step 0 ($\sigma = 1.0$) runs dense** and Steps 1–5 ($\sigma \le 0.972973$) all run sparse while coarse subject motion and shot composition are still forming.
2. **Distill-specific `dense_blocks`.** Blocks `38..42` were added from single-call probe spikes on PDD8 + FlashGen, and the October blind panel found `dense_blocks` did not prevent broken scripted action on `market`. Block `49` is the output projection block and stays dense on every non-dense arm tested here.

## Design (`bench/pika_sol_panel_arms.json`)

- **Base workflow:** `workflows/h3_text_to_video_pika_api.json` (`PIKA_FL2VA_SVD995_LORA` at strength `1.0` via `MiniMaxH3LoRABranch`, `lcm` sampler, `PIKA_MANUAL_SIGMAS`, `shift_video=12.0`, `shift_audio=6.5`).
- **Scene and length:** `prompt_bank/t2va_covered_market.txt` (`1344x768`, **345 frames** / `14.375 s`, patched from the graph's `362`-frame ceiling via `MiniMaxH3Resolution.length=345` so the render ends with the bank prompt's scripted `10–14.4 s` third shot and satisfies `bench/run_graph_arms.py`'s bank-length check; stresses physical crate/coin action and two-speaker `S1`/`S2` dialogue in one clip).
- **Seed & warmup:** Seed `730451892` (matching October's first seed), preceded by `--warmup pika_market__shipped` at seed `730451891` so Qwen3-VL-32B text encoding and model load stay out of measured rows without hitting ComfyUI's node-output cache (`suspect_cache_hit=false` on every row).
- **Substrate:** NVIDIA GeForce RTX 4090, `comfy-kitchen 0.2.41+sol.b63d8c7.up.7d6a22a` (`h3-frontier`, restarted before the run), PyTorch `2.15.0+cu134`, no `H3_SOL_PROBE` or `H3_SOL_OBSERVE` armed.
- **What `dense` means:** `pika_market__dense` sets `MiniMaxH3Sol.dense_blocks="0-49"`, routing all `300/300` DiT block-calls (`6 steps × 50 blocks`) to `ModelAttentionBackend` (`comfy-kitchen` INT8 dense attention with built-in `convrot128`), **not** BF16/FP32 exact SDPA. Token-refiner calls (`2` blocks on text tokens) run once outside the DiT override in `extra_conds` on core's default SDPA across all arms.
- **Fixed across all 5 arms:** `sink_conditioning="exact_kv_and_all_rows"`, `quantizer="rotated"`, `tau=1.0`, `token_routing="off"`, `min_tokens=12288`.

| arm | `start_percent` | $\sigma_{\text{start}}$ | full-dense steps | `dense_blocks` (sparse steps) | dense / sparse calls (`300` total) |
|---|---:|---:|---|---|---:|
| `pika_market__dense` | `0.20` | `0.979592` | `0, 1, 2, 3, 4, 5` (via `0-49`) | `0-49` | `300 / 0` |
| `pika_market__shipped` | `0.20` | `0.979592` | `0` | `38,39,40,41,42,49` | `80 / 220` |
| `pika_market__s0_b49` | `0.20` | `0.979592` | `0` | `49` | `55 / 245` |
| `pika_market__s01_b49` | `0.28` | `0.968610` | `0, 1` | `49` | `104 / 196` |
| `pika_market__s012_b49` | `0.55` | `0.907563` | `0, 1, 2` | `49` | `153 / 147` |

*(Note: `start_percent=0.50` maps to $\sigma_{\text{start}} = 0.92307692$, `8e-8` below the rounded Step 2 sigma `0.923077`; `0.55` places the cutoff unambiguously below Step 2 and above Step 3's `0.8`.)*

## Pass 1 timings and audio loudness (`seed=730451892`)

Rows recorded in `data/results/2026-11-11_pika_sol_panel_pass1.jsonl`; derived metrics in `data/results/2026-11-11_pika_sol_panel_pass1_metrics.json`. Loudness measured on each muxed `-audio.mp4` via `bench/measure_clip_loudness.py::ebur128`.

| arm | dense calls | `sampler_s` | $\Delta$ vs `dense` | $\Delta$ vs `shipped` | `decode_s` | `total_s` | integrated (`LUFS`) | `LRA` (`LU`) | true peak (`dBFS`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pika_market__dense` | `300` | `302.3` | `0.0` (`1.00x`) | `+109.5` | `18.8` | `325.2` | `-23.5` | `24.4` | `-10.7` |
| `pika_market__s012_b49` | `153` | `228.8` | `-73.5` (`1.32x`) | `+36.0` | `18.8` | `251.7` | `-23.4` | `24.6` | `-10.8` |
| `pika_market__s01_b49` | `104` | `204.5` | `-97.8` (`1.48x`) | `+11.7` | `18.8` | `227.5` | `-23.4` | `24.9` | `-11.2` |
| `pika_market__shipped` | `80` | `192.8` | `-109.5` (`1.57x`) | `0.0` | `18.8` | `215.8` | `-22.8` | `23.3` | `-9.1` |
| `pika_market__s0_b49` | `55` | `180.3` | `-122.0` (`1.68x`) | `-12.6` | `18.8` | `203.1` | `-23.0` | `21.8` | `-11.3` |

## Descriptive output distances vs `dense` (decoded clip, same seed)

Descriptive only — distance from `dense` measures how far an approximation moved the sample at seed `730451892`, not whether the sample looks better or worse. `coarse8` average-pools `336x192` RGB frames over `8x8` spatial blocks (`42x24` grid, matching `bench/score_output_distance.py`'s coarse-layout pooling scale) overall and across the prompt's three scripted shot spans (`0–4 s`, `4–10 s`, `10–14.375 s`); inter-frame delta and shot cuts are from `bench/measure_clip_delta.py`.

| arm | pixel rel L2 | `coarse8` rel L2 | `0–4 s` | `4–10 s` | `10–14.4 s` | audio wave rel L2 | delta median / p90 | cuts (`s`) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `pika_market__dense` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0132 / 0.0434` | `4.083, 8.375` |
| `pika_market__s012_b49` | `0.0807` | `0.0420` | `0.0360` | `0.0439` | `0.0445` | `0.5490` | `0.0135 / 0.0434` | `4.083, 8.375` |
| `pika_market__s01_b49` | `0.1944` | `0.1169` | `0.1012` | `0.1269` | `0.1147` | `0.7458` | `0.0134 / 0.0450` | `4.083, 8.375` |
| `pika_market__shipped` | `0.3595` | `0.2611` | `0.2346` | `0.2775` | `0.2595` | `1.0570` | `0.0134 / 0.0481` | `4.083, 8.375` |
| `pika_market__s0_b49` | `0.3590` | `0.2624` | `0.2214` | `0.2872` | `0.2587` | `0.9761` | `0.0130 / 0.0457` | `4.083, 8.375` |

*(Direct distance between `pika_market__shipped` and `pika_market__s0_b49`: `0.2805` pixel rel L2, `0.1804` `coarse8` rel L2.)*

## Observations before blind playback

1. **Exact linear cost per dense block-call.** Across all five arms (`55` to `300` dense calls), `sampler_s` obeys $\text{sampler\_s} \approx 153.0 + 0.498 \cdot N_{\text{dense}}$ to within `0.3 s` (`0.498 s` saved per block-call moved from `int8_attention` to `sol_attn` at `1344x768`, `345` frames); `decode_s` is `18.8 s` on all six runs.
2. **Blocks `38..42` do not pull Pika closer to `dense`.** `shipped` (`38..42,49`, `192.8 s`) and `s0_b49` (`49` only, `180.3 s`) sit at the exact same distance from `dense` (`0.3595` vs `0.3590` pixel rel L2; `0.2611` vs `0.2624` `coarse8` rel L2). Keeping `38..42` dense on Steps 1–5 costs `12.6 s` without moving distance to `dense`.
3. **Spending the dense budget on early whole steps moves the trajectory toward `dense`.**
   - `s01_b49` (Steps 0–1 dense, `49` thereafter) costs `+11.7 s` over `shipped` (`204.5 s` vs `192.8 s`) and cuts `coarse8` distance to `dense` more than in half (`0.2611` $\to$ `0.1169`).
   - `s012_b49` (Steps 0–2 dense, `49` thereafter) costs `+36.0 s` over `shipped` (`228.8 s`, still `73.5 s` faster than `dense`) and brings `coarse8` distance down to `0.0420` (`0.0807` full-pixel rel L2).
4. **Audio loudness holds steady under `exact_kv_and_all_rows`.** Integrated loudness spans `-23.5` to `-22.8 LUFS` across all five arms (`0.7 LUFS` total spread), and `s01_b49` / `s012_b49` sit within `0.1 LUFS` and `0.5 LU` of `dense`. Every arm places its two shot cuts at `4.083 s` and `8.375 s`.

## Blinded batch & next step

Blinded singles (`clip_01.mp4`..`clip_05.mp4`, muxed audio) and four `dense`-vs-arm stacked pairs (`pair_01.mp4`..`pair_04.mp4`, with synchronized per-half audio in `score.html`) were built by `bench/blind_batch.py` under `<output>/Video/blind/pika_sol_pass1_2026-11-11/` with the key sealed at `internal/blind_keys/pika_sol_pass1_2026-11-11.json`.

Defaults in `sol_attn_h3.py` and `workflows/h3_config.py` remain unchanged pending owner playback of Pass 1 and, if Pass 1 separates the arms perceptually, Pass 2 (`ref2va` Pika + second seed).
