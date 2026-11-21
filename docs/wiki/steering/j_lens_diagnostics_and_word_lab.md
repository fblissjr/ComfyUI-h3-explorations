# J-Lens: CPU X-Ray Diagnostics, Side-Effect Scrubbing, and Live Word Lab

Last updated: 2026-11-21 (Subspace X-Ray `decode_erasure_payload`, per-token `||E(h_i)||_2` scan, `Model2Vec` contextual lexicon, and `< 2 ms` `against`-SVD word erasure).

> **What to call it.** "J-Lens" is this repo's name for it. In anything written for other people, call it "a Jacobian-lens word readout for H3 control vectors, adapted from Anthropic's Jacobian lens (J-lens)": "the word readout" when it reads or scrubs, "a word vector" when it builds. Why, and what can be claimed for it, is in [`which_tool_and_what_to_call_it.md`](which_tool_and_what_to_call_it.md). A plain-words walk through the mechanism is in [`how_the_word_readout_works.md`](how_the_word_readout_works.md).

[`j_lens.py`](../../../j_lens.py) (backed by `models/j_lens/j_lens_h3.safetensors` plus the optional `Model2Vec`-style contextual multi-token lexicon `models/j_lens/model2vec_h3_lexicon.safetensors`, derivation in [`../../research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md`](../research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md)) translates directly between H3's `5376`-d post-refiner space (`fl2va`, `fl2va_base`, `ref2va`, and `h3_image`) and `24,912` single-token English words + `180` curated multi-token/2-word film, material, optical, and acoustic terms on **CPU in `< 15 ms`** (`0 GB` VRAM).

---

## 1. How J-Lens works under the hood

H3 taps `Qwen3-VL-32B` at `layers.49` (`5120`-d) rather than the final language-model head at `layers.63`, and then passes those tokens through a DiT-specific `condition_proj` + 2-block bidirectional `TokenRefiner` (`5376`-d). You cannot multiply a `5376`-d vector by Qwen's `lm_head` directly.

J-Lens bridges both gaps analytically:

1. **Upper-Qwen Jacobian ($\bar{J}_{49\to 63} \in \mathbb{R}^{5120 \times 5120}$)**:
   - Calibrated over `layers.50..63` + final RMSNorm, excluding token `0` (`||e_0|| = 15,309` vs. `292.3` median) so attention-sink geometry does not distort the linear map.
2. **Per-family broadcast `TokenRefiner` Jacobian ($\bar{J}_{\text{ref}}^{(f)} \in \mathbb{R}^{5376 \times 5120}$ for `fl2va`, `fl2va_base`, `ref2va`, and `h3_image`)**:
   - Because control vectors are **broadcast across a span of tokens** rather than added to a single token in isolation, self-attention inside the 2-block `TokenRefiner` transmits the shared shift through its value/output path ($\beta W_O W_V J_{\text{RMSNorm}}$). Modeling that broadcast attention path raises forward Jacobian cosine from `0.51` to **`0.836`**.
3. **Covariance-weighted Wiener inverse ($J_{\text{Wiener}}^{(f)} \in \mathbb{R}^{5120 \times 5376}$)**:
   - Maps any `5376`-d post-refiner vector back to `layers.49` (`5120`-d) using the empirical token covariance prior rather than isotropic Tikhonov damping (`0.878` downstream cosine vs. `0.782` isotropic).
4. **7 syntactic cloze baselines & `27,407`-token partition union**:
   - Evaluates log-probability shifts ($\Delta \log p$) across 7 cloze contexts (visual medium, lighting/film stock, body proportions, room motion, signage, surface texture, and `overall_soundscape`), normalizing over a `27,407`-token `logsumexp` union that matches full `151,936`-vocab `log_softmax` to `0.0098` nats while running in `~14.5 ms` on CPU.

---

## 2. The three high-leverage jobs J-Lens wins at

### A. Pre-render Subspace X-ray audit (`inspect`, `decode_erasure_payload`, and `WARP` token-erasure scan)
Every `build`, `from-prompt`, `make-media`, `words`, and `sculpt` command automatically runs `j_lens.decode_vector` + `j_lens.decode_erasure_payload` (plus `j_lens.check_b_side_alignment` on pair builds, warning immediately if `b_side` target words drop out of `(+)` or Pole-A baseline words dominate `(-)`) and prints both the 1D and Subspace Erasure readouts:
```bash
CUDA_VISIBLE_DEVICES= /path/to/comfy-venv/bin/python bench/control_vectors.py inspect \
  noir --prompt prompt_bank/t2va_sitcom_kitchen.txt
```
- **What the Subspace X-Ray reveals in `< 15 ms` on CPU**:
  - **Adds (`erasure.mu_d +` & `rescued vs 1D`):** Shows what asymmetric row-softmax matching (`tau = 0.15`) adds vs. naive 1D averaging. Across our 38-concept census (`76` files; starter bank now also includes `de_llm_slop`), **`28/38` concepts rescue key target words** into `erasure.mu_d` that washed out of `v_1d`—for example, `noir` rescues **`black (+1.52), footage, archival`** (`v_1d` lost `"black"` completely!), `film_16mm_1974` rescues **`grain, tape`**, and `claymation_stopmotion` rescues **`puppet`**.
  - **Erases (`erasure.vr[0..3]`):** Decodes each orthonormal Pole-A singular axis weighted by `erasure.diag` (`vr[1] (d=0.81): warm (+2.02), natural (+1.91), lighting (+1.68), warmth (+1.63)` on `noir`).
  - **Token-Wise Prompt Receptor & `WARP` Erasure Scan (`--prompt` / `--meter`):** Ranks both alignment (`aligns=[...]`) and per-token Pole-A erasure energy (`erases=[warm (0.36), ...]` on CPU via `analyze_prompt_coupling`, or exact post-refiner `||E(h_i)||_2` via `--meter <prompt.txt>`), showing why conflicting lighting words get erased while `"porter"` and `"crates"` stay near zero.
  - When `wardrobe_1970s` averaged 6 incompatible 1970s outfits on 2026-11-11, its word readout printed `vintage (+1.82), psychedelic (+1.38), nostalgic (+1.34), disco (+1.20)` with **no clothing words at all**—predicting before any GPU render that it would tint mood without changing `<Subject 1>`'s sweater.

### B. Surgical side-effect scrubbing (`hold_words` / `--hold-words`)
Projects single-token nuisance words out of **any** `.safetensors` vector or live `words` vector in `< 1 ms` on CPU (`hold_alpha` in `[0.0, 2.0]`, default `1.0`):
- **Verified render (`$H3_OUTPUT_DIR/Video/cv_e2e_verify/10_t2va_jlens_words_00001-audio.mp4`)**:
  - Goal: turn a sunny outdoor citrus market (`prompt_bank/04_t2va_market.txt`) into a freezing snowstorm **without** putting a winter parka over the vendor's canvas apron.
  - Settings (`MiniMaxH3WordVector` -> `MiniMaxH3Steer`; originally verified via inline `words` on legacy `ApplyControlVectors`):
    - `MiniMaxH3WordVector`: `words = "blizzard, snowdrift, freezing, snow, frost"`, `against_words = "warm, sunny, indoor, summer"`, `hold_words = "coat, parka"`, `hold_alpha = 0.5`
    - `MiniMaxH3Steer` (`custom_vector` wired from `MiniMaxH3WordVector`): `vector = "none"`, `strength = 1.65`, `mode = "add_norm_matched"`, `target = "scene_only"`
  - Result: snowdrifts cover the crates, awning, and cobblestones and snowflakes blow across the stall, while the vendor remains in his dark-green canvas apron and rolled sleeves instead of growing a parka.
- **Checking for hitchhiking words**: J-Lens's own `readiness.off_target_words` flags unexpected high-gain words that ride along with a word list so you can add them straight into `hold_words`.

### C. Zero-file `< 2 ms` live Word Lab + `against`-Subspace Erasure (`MiniMaxH3WordVector` or CLI `words`)
Wire `MiniMaxH3WordVector` (`words`, `against_words`, `hold_words`) into `MiniMaxH3Steer.custom_vector` (with `vector = "none"` and default `mode = "add_norm_matched_erasure"`), or save reusable `.safetensors` files on CPU via:
```bash
CUDA_VISIBLE_DEVICES= /path/to/comfy-venv/bin/python bench/control_vectors.py words \
  "noir, chiaroscuro, monochrome, silver gelatin, 35mm" \
  --against-words "@replace_digital, sunlit, warm, daylight, tungsten, color" \
  --hold-words "rain, coat" --hold-alpha 0.5 \
  --name quick_noir_erasure
```
- **Zero-GPU `against`-Subspace Erasure (`erasure.{mu_d,vr,diag}` in `< 2 ms`):** Instead of collapsing `against_words` down to a single 1D mean `mu_neg`, `synthesize_from_words` projects each individual `against` word/preset item into `5376`-d post-refiner space, scrubs `hold_words`, and extracts an orthonormal rank-`r` (`r <= 8`) erasure basis `erasure.vr` + ridge weights `erasure.diag`. When wired into `MiniMaxH3Steer` (`mode = "add_norm_matched_erasure"`), `MiniMaxH3WordVector` erases conflicting prompt tokens on the fly with zero `.toml` authoring!
- **Built-in `against_words` presets (usable standalone or mixed inside comma lists like `"@replace_digital, sunlit, warm"`)**:
  - `@neutral` (or empty): contrasts against the neutral cloze baseline (1D only).
  - `@prompt_subject` (or `@prompt`): inspects the active prompt at step 0 and automatically builds a dynamic per-prompt `against` erasure subspace over its medium and subject words.
  - `@replace_human`, `@replace_live_action`, `@replace_digital`: built-in multi-word negative anchors that automatically populate `erasure.vr` for non-human, animation, and analog film shifts.

---

## 3. `Model2Vec` Multi-Token Lexicon & Remaining Limits vs. 10-Pair GPU Vectors

Across single-token unigrams alone, CPU word vectors (`synthesize_from_words`) achieved `cos = +0.30` to `+0.52` against 10-pair GPU `.safetensors` vectors. On `2026-11-21`, we closed the biggest vocabulary gap while keeping a clear boundary for full GPU builds:

1. **`Model2Vec`-style contextual multi-token & phrase lexicon + BPE subword fallbacks**:
   - Previously, multi-token domain terms (`origami`, `muppet`, `plasticine`, `halation`, `pyrotechnics`, `kodachrome`, `anamorphic`, `cyanotype`, `monochrome`, `reverb`, `16mm`, `35mm`, `silver gelatin`, `direct flash`, `paper origami`) dropped as `unmatched` because Qwen's BPE splits them into subwords.
   - [`bench/control_vector/build_model2vec_lexicon.py`](../../../bench/control_vector/build_model2vec_lexicon.py) distills `180` curated visual, lighting, material, audio, and baseline terms across contextual cloze templates through `Qwen3-VL-32B` + all 4 H3 `TokenRefiner` recipes (`fl2va`, `ref2va`, `fl2va_base`, `h3_image`) with Minish Lab top-2 common-mode PC removal into `$H3_MODEL_STORE/j_lens/model2vec_h3_lexicon.safetensors`. When that optional file is absent, `j_lens.py` automatically falls back to curated semantic subword decompositions (`_SHORT_WORD_FALLBACKS` + phrase splitting) so core film/material terms never silently drop to `unmatched`.
2. **Static word bag vs. 50-layer sentence physics**:
   - Even with `Model2Vec` + `against`-SVD erasure (which powers both [`workflows/h3_text_to_video_pika_word_vector_erasure_api.json`](../../../workflows/h3_text_to_video_pika_word_vector_erasure_api.json) and H3 Pocket Console's `< 3 ms` dual `max(cos_v1, cos_mu)` CPU Vibe Search via `POST /api/lab/vibe_search` in [`app/cv_lab.py`](../../../app/cv_lab.py)), an 8–12 pair GPU `.toml` build still encodes full multi-clause sentences through 50 causal Qwen layers + 2 bidirectional `TokenRefiner` blocks, capturing richer coupled motion + `overall_soundscape:` dynamics and broader Pole-A subspace coverage (`N = 8..12` pairs). Use `MiniMaxH3WordVector` to sketch and iterate in `< 2 ms`, then graduate keeper concepts to an 8–10 pair `.toml` vector (`build` / `from-prompt`), or use [`MiniMaxH3EditGuider`](../edit_guidance.md) for named noun/wardrobe swaps.

---

## 4. Verifying or rebuilding the J-Lens bundle

Check that `models/j_lens/j_lens_h3.safetensors` is healthy on CPU (`< 1s`, `0` VRAM):
```bash
CUDA_VISIBLE_DEVICES= /path/to/comfy-venv/bin/python bench/control_vectors.py build-j-lens --verify-only
```
> **Caution before running `build-j-lens --overwrite`**: A full calibration (`build-j-lens` without `--verify-only`) requires an existing `--base-bundle` holding upper-Qwen (`layers.50..63`) tensors plus the configured `fl2va`, `ref2va`, and `H3_IMAGE_UNET` (`h3_image`) DiT checkpoints (`fl2va_base` reuses the un-LoRA'd `fl2va` DiT). Back up `models/j_lens/j_lens_h3.safetensors` before passing `--overwrite`.
