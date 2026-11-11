# J-Lens: CPU X-Ray Diagnostics, Side-Effect Scrubbing, and Live Word Lab

Last updated: 2026-11-11.

[`j_lens.py`](../../../j_lens.py) (backed by `models/j_lens/j_lens_h3.safetensors`, derivation in [`../../research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md`](../../research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md)) translates directly between H3's `5376`-d post-refiner space (`fl2va` and `ref2va`) and `24,912` single-token English words on **CPU in `< 15 ms`** (`0 GB` VRAM).

---

## 1. How J-Lens works under the hood

H3 taps `Qwen3-VL-32B` at `layers.49` (`5120`-d) rather than the final language-model head at `layers.63`, and then passes those tokens through a DiT-specific `condition_proj` + 2-block bidirectional `TokenRefiner` (`5376`-d). You cannot multiply a `5376`-d vector by Qwen's `lm_head` directly.

J-Lens bridges both gaps analytically:

1. **Upper-Qwen Jacobian ($\bar{J}_{49\to 63} \in \mathbb{R}^{5120 \times 5120}$)**:
   - Calibrated over `layers.50..63` + final RMSNorm, excluding token `0` (`||e_0|| = 15,309` vs. `292.3` median) so attention-sink geometry does not distort the linear map.
2. **Per-family broadcast `TokenRefiner` Jacobian ($\bar{J}_{\text{ref}}^{(f)} \in \mathbb{R}^{5376 \times 5120}$ for `fl2va` and `ref2va`)**:
   - Because control vectors are **broadcast across a span of tokens** rather than added to a single token in isolation, self-attention inside the 2-block `TokenRefiner` transmits the shared shift through its value/output path ($\beta W_O W_V J_{\text{RMSNorm}}$). Modeling that broadcast attention path raises forward Jacobian cosine from `0.51` to **`0.836`**.
3. **Covariance-weighted Wiener inverse ($J_{\text{Wiener}}^{(f)} \in \mathbb{R}^{5120 \times 5376}$)**:
   - Maps any `5376`-d post-refiner vector back to `layers.49` (`5120`-d) using the empirical token covariance prior rather than isotropic Tikhonov damping (`0.878` downstream cosine vs. `0.782` isotropic).
4. **7 syntactic cloze baselines & `27,407`-token partition union**:
   - Evaluates log-probability shifts ($\Delta \log p$) across 7 cloze contexts (visual medium, lighting/film stock, body proportions, room motion, signage, surface texture, and `overall_soundscape`), normalizing over a `27,407`-token `logsumexp` union that matches full `151,936`-vocab `log_softmax` to `0.0098` nats while running in `~14.5 ms` on CPU.

---

## 2. The three high-leverage jobs J-Lens wins at

### A. Pre-render X-ray audit (`inspect` and automatic build readouts)
Every `build`, `from-prompt`, `make-media`, `words`, and `sculpt` command automatically runs `j_lens.decode_vector` and prints the top `+` and `-` English words (and writes them into `<vector>.md`):
```bash
CUDA_VISIBLE_DEVICES= /path/to/comfy-venv/bin/python bench/control_vectors.py inspect \
  film_16mm_1974__ref2va__diff_span__lora_865ee189.safetensors \
  --prompt prompt_bank/16_ref2va_diner.txt
```
- **Why you should read this before rendering**:
  - When a vector has a sharp physical target (`film_16mm_1974`, `claymation_stopmotion`, `blizzard`), concrete nouns and textures lead the `+` list.
  - When `wardrobe_1970s` averaged 6 incompatible 1970s outfits on 2026-11-11, its J-Lens readout printed `vintage (+1.82), psychedelic (+1.38), nostalgic (+1.34), disco (+1.20)` with **no clothing words at all**—predicting in `< 15 ms` on CPU that the render would tint the diner's mood without changing `<Subject 1>`'s sweater.
  - Passing `--prompt <scene.txt>` adds a **Token-Wise Prompt Receptor Scan** (`analyze_prompt_coupling`), ranking which exact tokens in your prompt align or clash with the vector.

### B. Surgical side-effect scrubbing (`hold_words` / `--hold-words`)
Projects single-token nuisance words out of **any** `.safetensors` vector or live `words` vector in `< 1 ms` on CPU (`hold_alpha` in `[0.0, 2.0]`, default `1.0`):
- **Verified render (`$H3_OUTPUT_DIR/Video/cv_e2e_verify/10_t2va_jlens_words_00001-audio.mp4`)**:
  - Goal: turn a sunny outdoor citrus market (`prompt_bank/04_t2va_market.txt`) into a freezing snowstorm **without** putting a winter parka over the vendor's canvas apron.
  - Settings on `MiniMaxH3ApplyControlVectors`:
    - `vector_name = "none"`
    - `words = "blizzard, snowdrift, freezing, snow, frost"`
    - `against_words = "warm, sunny, indoor, summer"`
    - `hold_words = "coat, parka"`
    - `hold_alpha = 0.5`
    - `strength = 1.65`, `mode = "add_norm_matched"`, `mask = "scene_only"`
  - Result: snowdrifts cover the crates, awning, and cobblestones and snowflakes blow across the stall, while the vendor remains in his dark-green canvas apron and rolled sleeves instead of growing a parka.
- **Checking for hitchhiking words**: J-Lens's own `readiness.off_target_words` flags unexpected high-gain words that ride along with a word list so you can add them straight into `hold_words`.

### C. Zero-file `< 2 ms` live Word Lab (`words` on node or CLI)
Type comma-separated single-token English words straight into `MiniMaxH3ApplyControlVectors.words` (with `vector_name = "none"`), or save a reusable `.safetensors` pair on CPU via:
```bash
CUDA_VISIBLE_DEVICES= /path/to/comfy-venv/bin/python bench/control_vectors.py words \
  "blizzard, snowdrift, freezing, snow, frost" \
  --against-words "warm, sunny, indoor, summer" \
  --hold-words "coat, parka" --hold-alpha 0.5 \
  --name winter_squall_no_parka
```
- **Built-in `against_words` presets**:
  - `@neutral` (or empty): contrasts against J-Lens's neutral cloze baseline.
  - `@prompt_subject` (or `@prompt`): inspects the active prompt at step 0 and automatically contrasts against its existing medium and subject words.
  - `@replace_human`, `@replace_live_action`, `@replace_digital`: built-in negative anchors for non-human/animation/analog style shifts.

---

## 3. Measured limits of J-Lens synthesis vs. 10-pair GPU vectors

Across our starter bank, CPU-synthesized J-Lens vectors (`synthesize_from_words`) achieve **`cos = +0.30` to `+0.52`** against 10-pair GPU `.safetensors` vectors (`cyberpunk_neon_rain +0.52`, `crystallize_ice +0.50`, `film_16mm_1974 +0.50`, `balloon_head_inflate +0.46`, `claymation_stopmotion +0.37`, `felt_muppet +0.35`, `bioluminescent_glow +0.30`).

Two structural limits explain the gap:

1. **Single-token BPE vocabulary ceiling (`24,912` words)**:
   - J-Lens's lexicon intentionally contains only **complete single-token** English words in Qwen's BPE tokenizer.
   - Multi-token domain words—including `halation`, `plasticine`, `muppet`, `bioluminescent`, `luminescent`, `iridescent`, `rime`, `tuxedo`, `pyrotechnics`, `inferno`, and `scorching`—are **dropped as OOV** (`pos_words_unmatched`), with a warning logged.
   - If you use J-Lens `words`, pick punchy single-token English words (`clay, puppet, sculpting, felt, blizzard, frost, neon, grain, blast, flame, smoke`). When your concept hinges on multi-token phrases or relational prose, build a 6–10 pair GPU vector (`build` / `from-prompt`) or use [`MiniMaxH3EditGuider`](../edit_guidance.md).
2. **Unigram bag-of-words vs. 50-layer contextual physics**:
   - J-Lens adds unigram directions linearly; a 10-pair GPU build runs full sentences through 50 causal Qwen layers + 2 bidirectional `TokenRefiner` blocks, capturing coupled motion + `overall_soundscape:` dynamics that single words alone only approximate.

---

## 4. Verifying or rebuilding the J-Lens bundle

Check that `models/j_lens/j_lens_h3.safetensors` is healthy on CPU (`< 1s`, `0` VRAM):
```bash
CUDA_VISIBLE_DEVICES= /path/to/comfy-venv/bin/python bench/control_vectors.py build-j-lens --verify-only
```
> **Caution before running `build-j-lens --overwrite`**: A full GPU calibration (`build-j-lens` without `--verify-only`) requires an existing `--base-bundle` holding upper-Qwen (`layers.50..63`) tensors plus both `fl2va` and `ref2va` DiTs on GPU. Back up `models/j_lens/j_lens_h3.safetensors` before passing `--overwrite`.
