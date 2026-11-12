# Sol `start_percent` and `dense_blocks` panel on 6-step Pika, Pass 1 + Pass 2 (2026-11-11)

**lane:** sol
**verdict:** Across 3 cells (`15` measured clips, `12` owner-scored stacked pairs vs `dense`: `t2va_covered_market` at seeds `730451892` and `730451894`, plus `ref2va_scene_kitchen` with `2048px` `<Picture 1>` at seed `730451892`), Sol on 6-step Pika costs no playback-visible quality at `start_percent=0.2, dense_blocks="49"` (`s0_b49`, `55/300` dense block-calls). Dropping blocks `38..42` from `dense_blocks` saves `10.1–12.6 s` of sampler time over `shipped` (`109.2–122.0 s` vs all-dense INT8) with identical distance to `dense` and zero losses on blind playback (tagged `"good"` against `dense`'s `"off"` in both properly blind Pass 2 cells); adding dense early steps (`start_percent=0.28` or `0.55`) pulls the trajectory numerically toward `dense`'s specific sample draw (including when `dense`'s own draw has a staging quirk) without improving sample quality.

## Why

The shipped `MiniMaxH3Sol` defaults (`start_percent=0.2`, `dense_blocks="38,39,40,41,42,49"`, `sink_conditioning="exact_kv_and_all_rows"`, `quantizer="rotated"`, `tau=1.0`, `token_routing="off"`) were set during the PDD8 + FlashGen campaign ([2026-10-02_sol_dense_blocks_panel.md](2026-10-02_sol_dense_blocks_panel.md)). Two parts of that history do not carry over automatically to the 6-step Pika H3 Flex workflows (`workflows/h3_text_to_video_pika_api.json`, `workflows/h3_image_ref_plus_text_to_video_pika_api.json`):

1. **Schedule mapping at 6 steps (`shift_video=12.0`).** On `simple` schedules (`shift` cancels), `start_percent=0.2` runs `4/16` steps dense on Base16 and `3/12` steps dense on masked V2V. Pika's 6-step `ManualSigmas` grid (`PIKA_CONTRACT_KNOTS = (1000, 750, 500, 250, 100, 40, 0)`) evaluates the DiT at $\sigma \in \{1.0,\; 0.972973,\; 0.923077,\; 0.8,\; 0.571429,\; 0.333333\}$. Through `ModelSamplingAV(shift=12.0)`, `start_percent=0.2` maps to $\sigma_{\text{start}} = 0.979592$, so **only Step 0 ($\sigma = 1.0$) runs dense** and Steps 1–5 ($\sigma \le 0.972973$) run sparse.
2. **Distill-specific `dense_blocks`.** Blocks `38..42` were added from single-call probe spikes on PDD8 + FlashGen, and the October blind panel found `dense_blocks` did not move the owner's verdict. Block `49` is the terminal DiT block feeding `final_layer.video_out` and stays dense on every non-dense arm tested here.

## Design (`bench/pika_sol_panel_arms.json`, `bench/pika_sol_panel_pass2_t2v_arms.json`, `bench/pika_sol_panel_r2v_arms.json`)

- **Three cells (`5` measured arms each + discarded per-checkpoint warmups):**
  1. **Pass 1 (`t2va`, seed `730451892`):** `workflows/h3_text_to_video_pika_api.json` (`PIKA_FL2VA_SVD995_LORA`), `prompt_bank/t2va_covered_market.txt` (`1344x768`, `345` frames / `14.375 s`, patched from `362`), warmup at `730451891`.
  2. **Pass 2 Cell 1 (`t2va`, second seed `730451894`):** Same `t2va_covered_market` graph and 5 arms at fresh seed `730451894` (warmups at `730451893`).
  3. **Pass 2 Cell 2 (`ref2va`, seed `730451892`):** `workflows/h3_image_ref_plus_text_to_video_pika_api.json` (`PIKA_REF2VA_SVD995_LORA`, `shift_video=12.0`, `shift_audio=3.0`), `prompt_bank/ref2va_scene_kitchen.txt` (`1344x768`, `345` frames, 4 shots, `"TABLE 12 - 2 COVERS - FIRE"` ticket text, `S1`/`S2` dialogue + sung lyrics + `<|cutoff|>`), `<Picture 1>` = `old_man.jpg` (`2048px` DiT short edge), warmup at `730451891`.
- **Substrate:** NVIDIA GeForce RTX 4090, `comfy-kitchen 0.2.41+sol.b63d8c7.up.7d6a22a` (`h3-frontier`), PyTorch `2.15.0+cu134`, no `H3_SOL_PROBE` or `H3_SOL_OBSERVE` armed (`suspect_cache_hit=false` and `error=null` on all `19` rendered rows).
- **What `dense` means:** `MiniMaxH3Sol.dense_blocks="0-49"` routes all `300/300` DiT block-calls (`6 steps × 50 blocks`) to `ModelAttentionBackend` (`comfy-kitchen` INT8 dense attention with built-in `convrot128`), **not** BF16/FP32 exact SDPA. Token-refiner calls (`2` blocks on text tokens) run once outside the DiT override in `extra_conds` on core's default SDPA across all arms.
- **Fixed across all arms:** `sink_conditioning="exact_kv_and_all_rows"`, `quantizer="rotated"`, `tau=1.0`, `token_routing="off"`, `min_tokens=12288`.

| arm suffix | `start_percent` | $\sigma_{\text{start}}$ | full-dense steps | `dense_blocks` (sparse steps) | dense / sparse calls (`300` total) |
|---|---:|---:|---|---|---:|
| `dense` | `0.20` | `0.979592` | `0, 1, 2, 3, 4, 5` (via `0-49`) | `0-49` | `300 / 0` |
| `shipped` | `0.20` | `0.979592` | `0` | `38,39,40,41,42,49` | `80 / 220` |
| `s0_b49` | `0.20` | `0.979592` | `0` | `49` | `55 / 245` |
| `s01_b49` | `0.28` | `0.968610` | `0, 1` | `49` | `104 / 196` |
| `s012_b49` | `0.55` | `0.907563` | `0, 1, 2` | `49` | `153 / 147` |

*(Note: `start_percent=0.50` maps to $\sigma_{\text{start}} = 0.92307692$, `8e-8` below the rounded Step 2 sigma `0.923077`; `0.55` places the cutoff unambiguously below Step 2 and above Step 3's `0.8`.)*

## Timings and audio loudness across all three cells

Raw rows: `data/results/2026-11-11_pika_sol_panel_pass1.jsonl`, `data/results/2026-11-11_pika_sol_panel_pass2_t2v.jsonl`, `data/results/2026-11-11_pika_sol_panel_pass2_r2v.jsonl`. Derived metrics: `data/results/2026-11-11_pika_sol_panel_pass1_metrics.json`, `data/results/2026-11-11_pika_sol_panel_pass2_metrics.json`.

### Cell A — Pass 1 `t2va_covered_market` (`seed=730451892`)

| arm | dense calls | `sampler_s` | $\Delta$ vs `dense` | $\Delta$ vs `shipped` | `decode_s` | `total_s` | integrated (`LUFS`) | `LRA` (`LU`) | true peak (`dBFS`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pika_market__dense` | `300` | `302.3` | `0.0` (`1.00x`) | `+109.5` | `18.8` | `325.2` | `-23.5` | `24.4` | `-10.7` |
| `pika_market__s012_b49` | `153` | `228.8` | `-73.5` (`1.32x`) | `+36.0` | `18.8` | `251.7` | `-23.4` | `24.6` | `-10.8` |
| `pika_market__s01_b49` | `104` | `204.5` | `-97.8` (`1.48x`) | `+11.7` | `18.8` | `227.5` | `-23.4` | `24.9` | `-11.2` |
| `pika_market__shipped` | `80` | `192.8` | `-109.5` (`1.57x`) | `0.0` | `18.8` | `215.8` | `-22.8` | `23.3` | `-9.1` |
| `pika_market__s0_b49` | `55` | `180.3` | `-122.0` (`1.68x`) | `-12.6` | `18.8` | `203.1` | `-23.0` | `21.8` | `-11.3` |

### Cell B — Pass 2 `t2va_covered_market` (`seed=730451894`)

| arm | dense calls | `sampler_s` | $\Delta$ vs `dense` | $\Delta$ vs `shipped` | `decode_s` | `total_s` | integrated (`LUFS`) | `LRA` (`LU`) | true peak (`dBFS`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pika_market_s2__dense` | `300` | `302.3` | `0.0` (`1.00x`) | `+109.4` | `18.9` | `325.1` | `-18.9` | `23.9` | `-7.9` |
| `pika_market_s2__s012_b49` | `153` | `229.1` | `-73.2` (`1.32x`) | `+36.3` | `18.9` | `251.9` | `-18.7` | `24.8` | `-6.2` |
| `pika_market_s2__s01_b49` | `104` | `204.8` | `-97.5` (`1.48x`) | `+12.0` | `18.9` | `227.9` | `-19.1` | `26.2` | `-6.8` |
| `pika_market_s2__shipped` | `80` | `192.8` | `-109.4` (`1.57x`) | `0.0` | `18.8` | `215.8` | `-20.3` | `24.1` | `-7.0` |
| `pika_market_s2__s0_b49` | `55` | `182.7` | `-119.5` (`1.65x`) | `-10.1` | `19.0` | `229.5` | `-20.7` | `25.3` | `-9.1` |

*(Note: `pika_market_s2__s0_b49` ran in a follow-up `run_graph_arms.py` invocation where `total_s` included text re-encoding after the `ref2va` batch; `sampler_s = 182.7 s` is the isolated `SamplerCustomAdvanced` execution time.)*

### Cell C — Pass 2 `ref2va_scene_kitchen` + `old_man.jpg` at `2048px` (`seed=730451892`)

| arm | dense calls | `sampler_s` | $\Delta$ vs `dense` | $\Delta$ vs `shipped` | `decode_s` | `total_s` | integrated (`LUFS`) | `LRA` (`LU`) | true peak (`dBFS`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pika_kitchen_r2v__dense` | `300` | `358.9` | `0.0` (`1.00x`) | `+98.0` | `18.8` | `382.3` | `-10.7` | `4.2` | `-0.8` |
| `pika_kitchen_r2v__s012_b49` | `153` | `293.4` | `-65.5` (`1.22x`) | `+32.5` | `18.8` | `316.2` | `-10.8` | `4.4` | `-0.8` |
| `pika_kitchen_r2v__s01_b49` | `104` | `271.4` | `-87.5` (`1.32x`) | `+10.5` | `18.8` | `294.4` | `-10.0` | `4.2` | `-0.7` |
| `pika_kitchen_r2v__shipped` | `80` | `260.9` | `-98.0` (`1.38x`) | `0.0` | `18.8` | `284.0` | `-9.9` | `4.0` | `-0.1` |
| `pika_kitchen_r2v__s0_b49` | `55` | `249.7` | `-109.2` (`1.44x`) | `-11.2` | `18.8` | `273.1` | `-10.0` | `4.0` | `-0.4` |

### Linear timing fits across regimes

- **`t2va` (`1344x768`, `345f`, no ref tokens):** $\text{sampler\_s} \approx 153.0 + 0.498 \cdot N_{\text{dense}}$ on Pass 1 (`0.3 s` max residual) and $154.2 + 0.492 \cdot N_{\text{dense}}$ on Pass 2 (`1.4 s` max residual).
- **`ref2va` (`1344x768`, `345f`, `2048px` `<Picture 1>` under `exact_kv_and_all_rows`):** $\text{sampler\_s} \approx 225.2 + 0.446 \cdot N_{\text{dense}}$ (`0.14 s` max residual across all 5 arms). Under `exact_kv_and_all_rows`, the `2048px` reference tokens add exact KV sink blocks and exact query sink rows inside `sol_attn` on every sparse call, shifting the sparse baseline up by `+72.2 s` and narrowing the per-call dense-vs-sparse delta from `0.498 s` to `0.446 s`.

## Descriptive output distances vs `dense` (decoded clip, same seed)

Descriptive only — distance from `dense` measures how far an approximation moved the sample at a fixed seed, not sample quality. As the between-seed reference scale, two `dense` draws on `t2va_covered_market` (`730451892` vs `730451894`) sit at **`0.7441` pixel rel L2** and **`0.6606` `coarse8` rel L2**.

| cell | arm | pixel rel L2 | `coarse8` rel L2 | `0–4 s` | `4–10 s` | `10–14.4 s` | audio wave rel L2 | delta median / p90 | cuts (`s`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `t2va` (`730451892`) | `dense` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0132 / 0.0434` | `4.083, 8.375` |
| `t2va` (`730451892`) | `s012_b49` | `0.0807` | `0.0420` | `0.0360` | `0.0439` | `0.0445` | `0.5490` | `0.0135 / 0.0434` | `4.083, 8.375` |
| `t2va` (`730451892`) | `s01_b49` | `0.1944` | `0.1169` | `0.1012` | `0.1269` | `0.1147` | `0.7458` | `0.0134 / 0.0450` | `4.083, 8.375` |
| `t2va` (`730451892`) | `shipped` | `0.3595` | `0.2611` | `0.2346` | `0.2775` | `0.2595` | `1.0570` | `0.0134 / 0.0481` | `4.083, 8.375` |
| `t2va` (`730451892`) | `s0_b49` | `0.3590` | `0.2624` | `0.2214` | `0.2872` | `0.2587` | `0.9761` | `0.0130 / 0.0457` | `4.083, 8.375` |
| `t2va` (`730451894`) | `dense` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0124 / 0.0426` | `3.875, 6.583` |
| `t2va` (`730451894`) | `s012_b49` | `0.1299` | `0.0453` | `0.0235` | `0.0568` | `0.0389` | `0.4272` | `0.0122 / 0.0428` | `3.875, 6.583` |
| `t2va` (`730451894`) | `s01_b49` | `0.2679` | `0.1293` | `0.0724` | `0.1597` | `0.1128` | `0.6258` | `0.0123 / 0.0419` | `3.875, 6.583` |
| `t2va` (`730451894`) | `shipped` | `0.4737` | `0.3245` | `0.2654` | `0.3712` | `0.2951` | `0.8588` | `0.0119 / 0.0411` | `3.875, 6.583` |
| `t2va` (`730451894`) | `s0_b49` | `0.4746` | `0.3242` | `0.2665` | `0.3746` | `0.2881` | `0.8753` | `0.0118 / 0.0415` | `3.875, 6.583` |
| `ref2va` (`730451892`) | `dense` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0149 / 0.0548` | `11.208` |
| `ref2va` (`730451892`) | `s012_b49` | `0.0936` | `0.0521` | `0.0420` | `0.0623` | `0.0439` | `0.5761` | `0.0151 / 0.0558` | `11.208` |
| `ref2va` (`730451892`) | `s01_b49` | `0.2070` | `0.1270` | `0.1046` | `0.1489` | `0.1088` | `0.7330` | `0.0147 / 0.0556` | `11.208` |
| `ref2va` (`730451892`) | `shipped` | `0.3845` | `0.2734` | `0.1725` | `0.3284` | `0.2547` | `0.9163` | `0.0154 / 0.0539` | `6.917, 11.208` |
| `ref2va` (`730451892`) | `s0_b49` | `0.4083` | `0.3009` | `0.2286` | `0.3401` | `0.2966` | `0.9410` | `0.0148 / 0.0518` | `6.917, 11.167` |

Across all three cells, `shipped` (`38..42,49`) and `s0_b49` (`49` only) sit at virtually the same distance from `dense` (`0.2611` vs `0.2624` on Cell A; `0.3245` vs `0.3242` on Cell B; `0.2734` vs `0.3009` on Cell C), while `s01_b49` and `s012_b49` step monotonically toward `dense` (`~0.12` and `~0.04–0.05` `coarse8` rel L2).

## Playback verdicts (`bench/score_session.py`)

Joined verdict records:
- Pass 1 (`pika_sol_pass1_2026-11-11`): `data/results/2026-11-11_pika_sol_pass1_verdict.json` (unblinded before owner scoring because the agent's frame-strip report named the clip mapping in chat).
- Pass 2 `t2va` (`pika_sol_pass2_t2v_2026-11-11`): `data/results/2026-11-11_pika_sol_pass2_t2v_verdict.json` (**100% blind**; keys held unopened until both Pass 2 score files were written).
- Pass 2 `ref2va` (`pika_sol_pass2_r2v_2026-11-11`): `data/results/2026-11-11_pika_sol_pass2_r2v_verdict.json` (**100% blind**).

| session | contest (`dense` vs arm) | pair verdict | `dense` side tag | Sol-arm side tag | owner note (resolved to arm) |
|---|---|---|---|---|---|
| Pass 1 `t2va` (`730451892`, unblinded) | `dense` vs `shipped` | `can't tell` | `can't tell` (single: `good`) | `can't tell` (single: `good`) | — |
| Pass 1 `t2va` (`730451892`, unblinded) | `dense` vs `s0_b49` | `can't tell` | `can't tell` (single: `good`) | `can't tell` (single: `good`) | — |
| Pass 1 `t2va` (`730451892`, unblinded) | `dense` vs `s01_b49` | `can't tell` | `can't tell` (single: `good`) | `can't tell` (single: `good`) | — |
| Pass 1 `t2va` (`730451892`, unblinded) | `dense` vs `s012_b49` | `can't tell` | `can't tell` (single: `good`) | `can't tell` (single: `good`) | — |
| Pass 2 `t2va` (`730451894`, **blind**) | `dense` vs `shipped` | `can't tell` | `good` | `good` | *"she has gray hair in clip 2 [`dense`]. doesnt put a coin anywhere in clip 1 [`shipped`]"* |
| Pass 2 `t2va` (`730451894`, **blind**) | `dense` vs `s0_b49` | `can't tell` | **`off`** | **`good`** | *"gray hair in clip 1 [`dense`]. puts a coin in the box in clip 1 [`dense`] at 5-6s mark in clip 1."* |
| Pass 2 `t2va` (`730451894`, **blind**) | `dense` vs `s01_b49` | `same` | `off` | `off` | *"she has gray hair in both clips and doesnt put a coin anywhere in both clips i assume theyre the same"* |
| Pass 2 `t2va` (`730451894`, **blind**) | `dense` vs `s012_b49` | `same` | `off` | `off` | *"she has gray hair in both clips and doesnt put a coin anywhere in both clips i assume theyre the same"* |
| Pass 2 `ref2va` (`730451892`, **blind**) | `dense` vs `shipped` | `can't tell` | `same` | `same` | — |
| Pass 2 `ref2va` (`730451892`, **blind**) | `dense` vs `s0_b49` | *(unselected)* | **`off`** | **`good`** | *"person walking behind in clip 1 [`s0_b49`]. woman at 5s mark has a blue hair cover in clip 1 [`s0_b49`]."* |
| Pass 2 `ref2va` (`730451892`, **blind**) | `dense` vs `s01_b49` | `can't tell` | `same` | `same` | — |
| Pass 2 `ref2va` (`730451892`, **blind**) | `dense` vs `s012_b49` | `can't tell` | `same` | `same` | — |

## What Pass 1 + Pass 2 establish

1. **`s0_b49` (`start_percent=0.2, dense_blocks="49"`) never loses to `dense`, and is the fastest arm tested.** On Pass 1 (`730451892`), frame-by-frame crop inspection had flagged Shot 1 porter staging and Shot 2$\to$3 coin-tin continuity differences on the 1-dense-step arms (`shipped`, `s0_b49`), none of which registered on the owner's 24-fps playback (`5/5` singles `"good"`, `4/4` pairs `"can't tell"`). On Pass 2's second seed (`730451894`, scored blind), the direction flipped: `dense` itself drew a trajectory that omitted the porter in Shot 1 and dropped coins into his own orange crate at `5–6 s` (which `s01_b49` and `s012_b49` faithfully reproduced, earning `"same"` + `"off"` on both halves), whereas `s0_b49` and `shipped` brought the porter in behind her shoulder in Shot 1 and avoided the coin-in-crate quirk (`s0_b49` tagged `"good"` vs `dense` `"off"`).
2. **`ref2va` identity, typography, and dialogue loudness hold across all 5 arms.** On `ref2va_scene_kitchen` with `old_man.jpg` (`2048px` DiT short edge, `exact_kv_and_all_rows`), all 5 arms retained the peaked cap badge, white side hair, red `封` cheek talisman, and uniform across all cuts, rendered `"TABLE 12 - 2 COVERS - FIRE"` legibly on the ticket rail, and kept integrated loudness within `0.9 LUFS` (`-9.9` to `-10.8 LUFS`). In the blind pair against `dense`, `s01_b49`, `s012_b49`, and `shipped` scored `"can't tell"` / `"same"`, and `s0_b49` (`249.7 s` vs `358.9 s` `dense`) received `"good"` vs `dense`'s `"off"`.
3. **Blocks `38..42` buy nothing on Pika.** Keeping `38..42` dense on Steps 1–5 (`shipped`) costs `+10.1–12.6 s` per `345`-frame render over `s0_b49` (`49` only) without reducing distance to `dense` or improving blind verdicts on either `t2va` or `ref2va`.
4. **Owner decision (2026-11-11): `SOL_DENSE_TAIL = "49"` everywhere, all other Sol defaults unchanged.** `start_percent=0.2` already gives `4/16` dense steps on Base16 and `3/12` dense steps on 12-step masked V2V (`simple` scheduler), and `1/6` dense step (`s0_b49`) proved sufficient on 6-step Pika across both seeds and `ref2va`; `quantizer="rotated"`, `sink_conditioning="exact_kv_and_all_rows"`, `tau=1.0`, and `token_routing="off"` stay as-is. `SOL_DENSE_TAIL` is updated from `"38,39,40,41,42,49"` to `"49"` across `sol_attn_h3.py`, `workflows/h3_config.py`, and every shipped Sol workflow so the node schema and all lanes agree.
