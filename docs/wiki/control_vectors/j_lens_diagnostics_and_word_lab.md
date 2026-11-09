# 4. J-Lens Diagnostics, Bundle Builder & Word Lab

[← Back to Control Vectors Overview](../control_vectors.md)

[`j_lens.py`](../../../j_lens.py) and [`bench/control_vector/j_lens_builder.py`](../../../bench/control_vector/j_lens_builder.py) provide a **`< 2 ms` CPU diagnostic lens, orthogonal word scrubber, and fast unigram synthesizer** backed by `models/j_lens/j_lens_h3.safetensors`.

---

## 1. What Is Inside `models/j_lens/j_lens_h3.safetensors` (`h3_j_lens_v1`)?

The J-Lens bundle (`~643 MB` on disk, loaded on CPU via `j_lens.load_j_lens_bundle()`) contains 11 tensors (`REQUIRED_BUNDLE_TENSORS` in [`bench/control_vector/j_lens_builder.py`](../../../bench/control_vector/j_lens_builder.py)) and a `24,912`-word vocabulary index (`vocab_words_json`):

| Tensor Key | Shape & Dtype | Mathematical Definition & Provenance |
|---|---|---|
| **`M_union`** | `(27407, 5120) float16` | Pre-multiplied lexical unembedding rows $W_{\text{head}}[\text{union\_idx}] \bar{J}_{49 \to 63}$ for `27,407` single-token English words (`24,912` valid words indexed by `valid_pos_in_union`). |
| **`cloze_base_union`** | `(7, 27407) float32` | Unsteered baseline next-token log-probabilities across the 7 syntactic H3 cloze prefixes (`CLOZE_PREFIXES` in [`j_lens.py`](../../../j_lens.py)). |
| **`J_49_to_63`** | `(5120, 5120) float32` | Expected Jacobian transporting perturbations from `Qwen3-VL layers.49` (`5120`-d) through `layers.50..63` and final `RMSNorm`. |
| **`Sigma_49_p030`** | `(5120, 5120) float16` | Spectral-filtered fractional covariance $\Sigma_{49,\text{filt}}^{+0.30}$ of non-sink (`tok1+`) activations at `layers.49`, with the top-1 common-mode eigenvalue $\lambda_{\max}$ zeroed out and remaining eigenvalues raised to $p = 0.30$ ([`compute_sigma_49_and_p030`](../../../bench/control_vector/j_lens_builder.py)). |
| **`J_ref_fl2va`**<br>**`J_ref_ref2va`** | `(5376, 5120) float32` | Analytical expected Jacobians $\bar{J}_{\text{ref}}^{(f)} = \mathbb{E}[\partial \mathbf{r}_{\text{PR}} / \partial \mathbf{h}_{49}]$ across `condition_proj` and the 2-block `TokenRefiner` (including the broadcast attention value path $\beta W_O W_V J_{\text{RMSNorm}_1}$ and SwiGLU FFN $J_{\text{FFN}} J_{\text{RMSNorm}_2}$) in [`compute_refiner_jacobian_and_wiener`](../../../bench/control_vector/j_lens_builder.py). |
| **`J_wiener_fl2va`**<br>**`J_wiener_ref2va`** | `(5120, 5376) float32` | Closed-form empirical-covariance Wiener inverses $\Sigma_{49} (\bar{J}_{\text{ref}}^{(f)})^\top \left(\bar{J}_{\text{ref}}^{(f)} \Sigma_{49} (\bar{J}_{\text{ref}}^{(f)})^\top + \lambda^2 I_{5376}\right)^{-1}$ ($\gamma = 0.20$ shrinkage, $\lambda_{\text{rel}} = 0.03$), mapping any `5376`-d `vector.post_refiner` back into `5120`-d encoder space for lexical decoding. |
| **`union_idx`, `valid_idx`, `valid_pos_in_union`** | `int32` index arrays | Tokenizer ID and row lookup tables for the `24,912` valid English vocabulary words (`vocab_words_json` in header metadata). |

---

## 2. How to Verify or Rebuild `j_lens_h3.safetensors` (`build-j-lens`)

[`bench/control_vector/j_lens_builder.py`](../../../bench/control_vector/j_lens_builder.py) provides verification and rebuilding under `bench/control_vectors.py build-j-lens`:

1. **Verify Schema, Shapes, Norms & Round-Trip Synthesis (`< 50 ms`, CPU)**:
   ```bash
   "$PY" bench/control_vectors.py build-j-lens --verify-only
   ```
2. **Rebuild `Sigma_49_p030`, `J_ref_{fl2va,ref2va}`, and `J_wiener_{fl2va,ref2va}` from Local ComfyUI Checkpoints**:
   Recomputes the empirical `layers.49` covariance $\Sigma_{49,\text{filt}}^{0.30}$ across `--num-prompts` calibration prompts from `prompt_bank/` and the analytical `TokenRefiner` Jacobians + Wiener inverses for both `fl2va` and `ref2va` (preserving upper-Qwen `J_49_to_63`, `M_union`, and `cloze_base_union` from `--base-bundle` or the existing bundle):
   ```bash
   "$PY" bench/control_vectors.py build-j-lens --num-prompts 24
   ```

---

## 3. Reading J-Lens Diagnostics (`inspect` & `.md` Sidecars)

Every time you build, sculpt, or inspect a control vector:
```bash
"$PY" bench/control_vectors.py inspect models/control_vectors/film_16mm_1974__fl2va__diff_span__lora_2a8c2b52.safetensors --prompt my_scene.txt
```
`j_lens.py` prints two complementary lexical readouts and four readiness metrics:

1. **`j_lens (+)` and `j_lens (-)` (Log-Odds Shift through `J_wiener` + `J_49_to_63`)**:
   Projects `vector.post_refiner` (`5376`-d) back into `5120`-d via `J_wiener_{family}`, transports it through `M_union` ($W_{\text{head}} \bar{J}_{49 \to 63}$), and scores the net log-odds gain across all `24,912` English words.
   - `j_lens (+)` shows the top words pushed **up** by `+1.0 * v`.
   - `j_lens (-)` shows the top words pushed **up** by `-1.0 * v` (what the vector suppresses or moves away from).
2. **`cov_49 (+)` and `cov_49 (-)` (Spectral-Filtered Layer-49 Cosine)**:
   Applies $\Sigma_{49,\text{filt}}^{+0.30}$ (`Sigma_49_p030`) to `vector.encoder` and ranks vocabulary words by layer-49 cosine alignment.
3. **Readiness Metrics (`assess_steering_readiness`)**:
   - **`top1_gain` / `top3_mean_gain`**: Projected log-odds shift of the strongest pushed vocabulary words (`>= 0.95` for contextual pair vectors; `>= 2.5` for `from-words` unigram vectors).
   - **`refiner_transmission` ($\lVert J_{\text{ref}} \hat{\mathbf{v}}_{\text{enc}}\rVert_2$)**: How strongly the 2-block `TokenRefiner` transmits the encoder direction (`>= 0.50` indicates strong survival through the refiner).
   - **`echo_ratio`**: Fraction of the top-8 decoded words that semantically match your target concept (`[0.0, 1.0]`).
   - **`off_target_words`**: Unintended companion words appearing in the top-8 readout—pass any unwanted words to `--hold-words "word1, word2"` to scrub them out on CPU in `< 1 ms`.
