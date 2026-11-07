# 2026-11-06: Holistic Resolution Proposal for H3 Control Vectors, `J-Lens`, and Active Worktrees

**Date**: 2026-11-06 (All 5 holistic invariants landed on `main` at `93dbe8d3`, `0d2ad8a9`, and `4ea73adb`)  
**Companion docs**: [`2026-11-06_fresh_eye_audit_and_holistic_resolution.md`](2026-11-06_fresh_eye_audit_and_holistic_resolution.md), [`2026-11-06_holistic_stack_review_and_invariants.md`](2026-11-06_holistic_stack_review_and_invariants.md)  
**Related docs**: [`../../wiki/control_vectors.md`](../../wiki/control_vectors.md), [`2026-11-06_j_lens_and_refiner_jacobian.md`](2026-11-06_j_lens_and_refiner_jacobian.md), [`2026-11-06_mainodes_comparison_and_pika_roadmap.md`](2026-11-06_mainodes_comparison_and_pika_roadmap.md), [`2026-11-06_detangle_comparison_and_borrowable_concepts.md`](2026-11-06_detangle_comparison_and_borrowable_concepts.md).

---

## 1. Goal & Architectural Philosophy

In the last 36 hours (`42e3d3bb`..`0d2ad8a9`, plus the active worktrees `cvman-holistic-workbench` at `86592f0c` and `dials-holistic-checks`), the H3 control vector stack grew from offline `.toml` pair builders into a multi-tier steering system spanning:
1. `< 2 ms` CPU `J-Lens` word synthesis, recipe composition, and `14.5 ms` `5376`-d verbalization ([`j_lens.py`](../../../j_lens.py)),
2. Zero-`.toml` canonical 10-pair compilation with `{props}`/`{surface}` binding, coupled soundscapes, and `target_prompt` `Pair 0` anchoring ([`h3_pair_compiler.py`](../../../h3_pair_compiler.py)),
3. 32-px M-RoPE media-contrast vectors ([`control_vector_nodes.py`](../../../control_vector_nodes.py#L1898-L2145)), and
4. Runtime cross-vector holds, section masks, per-stream AV conditioning splits, and `ref2va` 3D VAE reference gating ([`control_vectors.py`](../../../control_vectors.py), [`control_vector_nodes.py`](../../../control_vector_nodes.py), [`h3_spans.py`](../../../h3_spans.py)).

Following the fresh-eye audit ([`2026-11-06_fresh_eye_audit_and_holistic_resolution.md`](2026-11-06_fresh_eye_audit_and_holistic_resolution.md)) and cross-verification against disk by `mrprompt`, `jlens`, `cvman`, and `dials`, all five sessions have converged on the exact mathematical and engineering resolution below.

---

## 2. Invariants 1 & 2: One Hold-Projection & Norm Contract + Twin-Space Symmetry

### 2.1 The Problem
Hold projection $\mathbf{v}_\perp(\alpha) = (I - \alpha U^\top U)\mathbf{v}$ had **four conflicting L2-norm and $\alpha$-bound behaviors** across the repo:

| Path | Code Site | Current $\alpha$ Bounds | Current Post-Hold Norm Behavior |
|---|---|---|---|
| **1. Build-time `.toml` `[[hold]]`** | [`control_vector_nodes.py:867`](../../../control_vector_nodes.py#L867), [`bench/control_vectors.py:461, 2103`](../../../bench/control_vectors.py#L2103) | `1.0` (`project_out` caps to `[0.0, 1.0]` at [`control_vectors.py:1169`](../../../control_vectors.py#L1169)) | **Preserves projection norm drop** ($\lVert\mathbf{v}_\perp\rVert_2 \le \lVert\mathbf{v}\rVert_2$; `head_swell` drops `85.01 -> 77.65`) |
| **2. Runtime chained `mode="hold"`** | [`control_vectors.py:2110-2130, 2335`](../../../control_vectors.py#L2110-L2130) | Unclamped at [`control_vector_nodes.py:1309, 1322`](../../../control_vector_nodes.py#L1309-L1322) and unvalidated in `_apply_holds_to_shift64` | **Preserves projection norm drop** on `total_shift64` |
| **3. `j_lens.py` / `inspect --hold-words`** | [`j_lens.py:308, 686-689, 761, 1961`](../../../j_lens.py#L686-L689) | `[0.0, 2.0]` | **Preserves projection norm drop** (`norm_after_holds < norm_before_holds`) on `decode_safetensors`, `decode_recipe`, and `save_composed_recipe_vector` |
| **4. Runtime `hold_words` on `vector_name` & `cvman` `_apply_post_refiner_holds`** | [`control_vector_nodes.py:1633-1641`](../../../control_vector_nodes.py#L1633-L1641), `cvman` `bench/control_vectors.py:259` | `[0.0, 2.0]` | **Silently rescales back to `orig_nrm`** (`vec_held * (orig_nrm / res_nrm)`) |

Furthermore, during `make` and `build`, [`control_vector_nodes.py:911-929`](../../../control_vector_nodes.py#L911-L929) and [`bench/control_vectors.py:481-495, 2120-2138`](../../../bench/control_vectors.py#L2120-L2138) omit `unit_mean=use_unit_mean` and never project `[[hold]]` out of `vector.encoder` (`5120`-d), so [`j_lens.py::decode_safetensors`](../../../j_lens.py#L703-L725) compares an *un-held* `vector.encoder` against a *held* `vector.post_refiner`.

### 2.2 The Unified Contract
1. **Standardize $\alpha \in [0.0, 2.0]$ everywhere**:
   - In [`control_vectors.py::project_out`](../../../control_vectors.py#L1169):
     ```python
     if not math.isfinite(a_val) or a_val < 0.0 or a_val > 2.0:
         raise ValueError(f"project_out alpha must be in [0.0, 2.0], got {alpha!r}")
     ```
   - In [`control_vectors.py::_apply_holds_to_shift64`](../../../control_vectors.py#L2110-L2130), validate `0.0 <= h_alpha <= 2.0` on every hold tuple.
   - In [`bench/check_control_vectors.py:1840-1844`](../../../bench/check_control_vectors.py#L1840-L1844), test `alpha=1.5` as valid over-projection and `alpha=2.5` as the out-of-range refusal.
2. **Never silently rescale a pre-built vector back to `orig_nrm` after hold projection**:
   - In [`control_vector_nodes.py:1626-1641`](../../../control_vector_nodes.py#L1626-L1641) (and `_apply_post_refiner_holds` at `bench/control_vectors.py:259` in `cvman-holistic-workbench`):
     ```python
     vec_held = _jl_mod.apply_soft_holds(
         vec_pr,
         [h_synth["vector.post_refiner"]],
         alpha=float(hold_alpha),
     )
     res_nrm = float(torch.linalg.vector_norm(vec_held.to(dtype=torch.float64)).item())
     if res_nrm <= 1e-6 * max(1.0, orig_nrm):
         raise ValueError(
             f"hold_words {hold_words_str!r} completely cancelled control vector {vec_path.name!r}"
         )
     vec_pr = vec_held
     ```
   - **Why**: Under `mode="add"`, removing a correlated nuisance subspace *should* remove that subspace's energy ($\lVert\mathbf{v}_\perp\rVert_2 < \lVert\mathbf{v}\rVert_2$), matching `inspect --hold-words` and `mode="hold"`. When the user wants constant post-hold shift magnitude (`67.5 * strength`), `mode="add_norm_matched"` ([`control_vectors.py:2216-2219, 2347-2356`](../../../control_vectors.py#L2216-L2219)) **already** rescales `vec_pr` (or the post-hold shift) to `NORM_MATCHED_TARGET = 67.5`.
3. **Twin-Space Symmetry: Apply `unit_mean` and `[[hold]]` projection to `vector.encoder` (`5120`-d)**:
   - In [`control_vector_nodes.py:820-935`](../../../control_vector_nodes.py#L820-L935) and [`bench/control_vectors.py:450-500, 2065-2139`](../../../bench/control_vectors.py#L2065-L2139):
     - Pass `unit_mean=use_unit_mean` to `cv.mean_difference(enc_a_mat, enc_b_mat, w_vec, unit_mean=use_unit_mean)`.
     - When `holds` are pooled and `not no_holds`, pool `enc_hold_a` and `enc_hold_b` with the same `mask_a_enc` / `mask_b_enc` (row 0 excluded), build `enc_basis = cv.hold_basis(...)`, and compute `enc_final_vec = cv.project_out(enc_raw_vec, enc_basis)` before unit-normalizing `vector.encoder`.

---

## 3. Invariant 3: One Compiler Owner (`h3_pair_compiler.py`) & Collision-Free Multi-Span `differing_token_mask` (`h3_spans.py`)

### 3.1 Single Compiler Owner
- Remove `_inject_target_prompt_into_pairset` in `cvman-holistic-workbench` (`bench/control_vectors.py:173`) and route all short-spec expansion, `{props}`/`{surface}` binding, coupled `description + overall_soundscape` pairs, `hold:` groups, and `target_prompt` `Pair 0` (`weight = 2.5`) injection through [`h3_pair_compiler.py::compile_words_to_pairset`](../../../h3_pair_compiler.py#L948-L1050) (`93dbe8d3`).

### 3.2 Multi-Span `_DIFF_MASK_CACHE` Single-Span Eviction & `other_ids` Forwarding
- At [`h3_spans.py:708-713`](../../../h3_spans.py#L708-L713), `differing_token_bounds(a, b)` currently writes to `_DIFF_MASK_CACHE` only when `not bool(mask_a[first_diff:end_a].all())` (multi-span). If a subsequent pair reuses the same `a` with the same outer bounds `(first_diff, end_a)` but is single-span, the stale multi-span entry is not evicted.
- **Fix**:
  1. In [`h3_spans.py:708-713`](../../../h3_spans.py#L708-L713), evict `(tuple(a), first_diff, end_a)` and `(tuple(b), first_diff, end_b)` when single-span:
     ```python
     if not bool(mask_a[first_diff:end_a].all()) or not bool(mask_b[first_diff:end_b].all()):
         while len(_DIFF_MASK_CACHE) >= 512:
             _DIFF_MASK_CACHE.pop(next(iter(_DIFF_MASK_CACHE)))
         _DIFF_MASK_CACHE[(tuple(a), first_diff, end_a)] = mask_a
         _DIFF_MASK_CACHE[(tuple(b), first_diff, end_b)] = mask_b
     else:
         _DIFF_MASK_CACHE.pop((tuple(a), first_diff, end_a), None)
         _DIFF_MASK_CACHE.pop((tuple(b), first_diff, end_b), None)
     ```
  2. Pass `other_ids=ids_b` (for `mask_a`) and `other_ids=ids_a` (for `mask_b`) at the `pooling_mask` call sites in `control_vector_nodes.py` and `bench/control_vectors.py` (which [`h3_spans.py:723, 747-750`](../../../h3_spans.py#L723-L750) already supports).

---

## 4. Invariant 4: Basis-Aware `J-Lens` Readout, Intra-Word Cloze Admissibility Weighting, & Stemmer Accuracy (`j_lens.py`)

### 4.1 Morphological Stemmer Fix (`"operatic" -> "opera"`) in [`j_lens.py::_resolve_word_indices`](../../../j_lens.py#L1650-L1666)
- Previously at [`j_lens.py:1650-1665`](../../../j_lens.py#L1650-L1665), stripping `"ic"` from `"operatic"` produced `stem = "operat"`, which missed `"opera"` in `word_to_idxs` and fell into `vw.startswith("operat")`, matching `["operation", "operator", "operating", "operative"]`.
- **Fix (in [`j_lens.py:1653-1661`](../../../j_lens.py#L1653-L1661))**: Before falling back to `vw.startswith(stem)` prefix expansion, check exact truncated and `-e`-restored root candidates in `word_to_idxs`:
  ```python
  if stem in word_to_idxs:
      stem_ids.extend(word_to_idxs[stem])
      break
  if len(stem) >= 5 and stem[:-1] in word_to_idxs:
      stem_ids.extend(word_to_idxs[stem[:-1]])
      break
  if (stem + "e") in word_to_idxs:
      stem_ids.extend(word_to_idxs[stem + "e"])
      break
  ```
  This maps `"operatic" -> "opera"`, `"dramatic" -> "drama"`, `"chromatic" -> "chroma"`, and `"cinematic" -> "cinema"` directly to their exact vocabulary entries before prefix fallback.

### 4.2 Intra-Word Cloze Admissibility Weighting (Preserving `1/N` Inter-Word Normalization) in [`j_lens.py:211-221, 1697-1725`](../../../j_lens.py#L1697-L1725)
In `<comfy>/models/j_lens/j_lens_h3.safetensors`, `cloze_base_union` has shape `[7, 27407]`, and `valid_pos_in_union` (`[24912]`) maps the `24,912` rows of `M_centered` (`word_to_idxs`) into `cloze_base_union`.
- **Disk measurement & implementation (`mrprompt` + `dials`)**:
  - In [`load_j_lens_bundle`](../../../j_lens.py#L211-L221):
    ```python
    cloze_base = f.get_tensor("cloze_base_union").to(dtype=torch.float32)
    piece_weights = torch.sigmoid(
        (torch.log_softmax(cloze_base, dim=-1)[:, valid_pos].max(dim=0).values + 11.5) / 2.0
    )
    ```
  - *Within* a single word's matched BPE variants `w_ids` (`len(w_ids) > 1` in [`_mean_word_direction`](../../../j_lens.py#L1712-L1718), e.g. `"pixel"` matching space-prefixed `" pixel"` at $\max_p \ell_p = -7.76$, `" Pixel"` at $-11.21$, and mid-word suffix `"pixel"` at $-15.75$), weighting the BPE variants within `w_ids` by `piece_weights[w_ids]` cleanly down-weights mid-word continuation pieces (`0.866` vs. `0.536` vs. `0.107`) before unit-normalizing that word's direction:
    ```python
    rows = m_centered[w_ids]
    if piece_weights is not None and len(w_ids) > 1:
        pw = piece_weights[w_ids].unsqueeze(-1)
        w_vec = (rows * pw).sum(dim=0) / pw.sum().clamp_min(1e-8)
    else:
        w_vec = rows.mean(dim=0)
    word_vecs.append(F.normalize(w_vec, dim=0))
    ```
  - *Across* distinct words (`word_vecs`), every word remains unit L2-normalized (`F.normalize(w_vec, dim=0)`) so high-prior modifiers like `"grain"` ($\max_p \ell_p = -5.03$, $\sigma = 0.962$) never outweigh specific nouns like `"blizzard"` ($\max_p \ell_p = -16.45$, $\sigma = 0.078$).

---

## 5. Invariant 5: Per-Stream Conditioning + Splicing, Clamped Stream Holds, & Runtime Node Hygiene

### 5.1 Why Per-Stream Conditioning + Splicing (`ctx_video`, `ctx_audio`) Beats Velocity-Delta Blending
In [`control_vector_nodes.py:1289-1400`](../../../control_vector_nodes.py#L1289-L1400), `_make_control_vector_wrapper` builds `ctx_video = _build_steered_context(context, "video_scale")` and `ctx_audio = _build_steered_context(context, "audio_scale")`, executes two DiT passes (`out_vid`, `out_aud`), and splices `[out_vid[0], out_aud[1]]`.
- **Multi-vector AV diversity in $O(1)$ passes**: A user can chain:
  - Node 1: `cursed_cctv` (`video_scale=1.0, audio_scale=0.0` — visual style only)
  - Node 2: `operatic_bellow` (`video_scale=0.0, audio_scale=1.0` — audio soundscape only)
  - Node 3: `vocal_strain` (`mode="hold", video_scale=0.0, audio_scale=1.0` — audio hold only)
  and evaluate both streams on the DiT's true nonlinear conditioning manifold in **at most 2 DiT passes total**, whereas a 2-pass `(out_base, out_steered)` velocity blend collapses all specs into a single `ctx_steered`.
- **Clamping `mode="hold"` stream scales**:
  - Keep `_spec_stream_scale` active across all 5 `APPLY_MODES` (`"add"`, `"add_norm_matched"`, `"project_out"`, `"amplify"`, `"hold"`).
  - At [`control_vector_nodes.py:1309, 1322`](../../../control_vector_nodes.py#L1309-L1322), when `_effective_entry_mode(sp) == "hold"`, clamp the effective stream-scaled hold weight to `[0.0, 2.0]`:
    ```python
    em0 = _effective_entry_mode(sp)
    s0 = sp.strength * _spec_stream_scale(sp, stream_attr)
    if em0 == "hold":
        s0 = max(0.0, min(2.0, s0))
    ```
    (and identically for `sj` at `L1322`). This allows `video_scale=0.0, audio_scale=1.0` on a `mode="hold"` node to scrub a nuisance direction from audio only (`alpha_aud = 1.0, alpha_vid = 0.0`) while guaranteeing a negative stream scale can never invert a hold into an addition.
- **Documentation alignment**: Update [`docs/wiki/control_vectors.md:113`](../../wiki/control_vectors.md#L113) so the prose states the per-stream conditioning + `[out_vid[0], out_aud[1]]` splice formula accurately.

### 5.2 Cross-Family `fingerprint_inputs` ([`control_vector_nodes.py:2614-2628`](../../../control_vector_nodes.py#L2614-L2628))
- Stat all family-resolved candidates `(None, "fl2va", "ref2va")` in `fingerprint_inputs` (unwrapping when a single file resolves so existing 3-tuple `(path, mtime_ns, size)` callers stay compatible, and returning a tuple of 3-tuples when both `__fl2va__` and `__ref2va__` siblings exist) so rebuilding either family sibling on disk invalidates ComfyUI's execution cache:
  ```python
  resolved_stats: list[tuple[str, int, int]] = []
  for pref_fam in (None, "fl2va", "ref2va"):
      try:
          full = resolve_vector_file_path(raw, preferred_family=pref_fam)
          st = os.stat(full)
          entry = (str(full), st.st_mtime_ns, st.st_size)
          if entry not in resolved_stats:
              resolved_stats.append(entry)
      except Exception:
          continue
  vec_fp = (
      resolved_stats[0]
      if len(resolved_stats) == 1
      else (tuple(resolved_stats) if resolved_stats else raw)
  )
  ```

### 5.3 Explicit Warnings for `vector_name` + `words` and Missing `_h3_span_masks` on `@prompt_subject`
1. **Both `vector_name != "none"` and `words` set on one node ([`control_vector_nodes.py:1600-1730`](../../../control_vector_nodes.py#L1600-L1730))**:
   - Log a clear `logger.warning` (or override `vector_name` when `words` is non-empty) so a user typing `words` into a workflow node whose `vector_name` dropdown defaults to `film_16mm_1974...` is warned immediately.
2. **`@prompt_subject` without `_h3_span_masks` ([`control_vector_nodes.py:1240-1242`](../../../control_vector_nodes.py#L1240-L1242))**:
   - Log a one-time `logger.warning` when `sp.metadata.get("deferred_against_prompt")` is `True` and `eff_prompt` is empty (when a user manually wires a core conditioning node instead of `MiniMaxH3Conditioning` / `MiniMaxH3ReferenceConditioning`).

### 5.4 `j_lens.*` Provenance Scope for Base-vs-Pika Recipe Compatibility ([`control_vectors.py:2036-2047`](../../../control_vectors.py#L2036-L2047))
- In [`control_vectors.py::check_vector_compatibility`](../../../control_vectors.py#L2036-L2047), when `vec_loras != cur_loras` and `not allow_recipe_mismatch`:
  - Check `prov_made_by = str(metadata.get("provenance", {}).get("made_by", ""))`.
  - If `prov_made_by.startswith("j_lens.")` and `vec_family == cur_family`, emit an advisory warning instead of raising `ValueError`.
  - This lets CPU-synthesized/composed `J-Lens` vectors load seamlessly on both base 30-step UNets (`loras=[]`) and Pika 6-step UNets while preserving all `C5` `.safetensors` LoRA refusal assertions in [`bench/check_control_vectors.py:1570-1658`](../../../bench/check_control_vectors.py#L1570-L1658).

---

## 6. Landed Consolidation Sequence on `main`

1. **`mrprompt-holistic-compiler` (`93dbe8d3`)** — **Landed on `main`**:
   - Multi-span `differing_token_mask`, coupled `description + overall_soundscape` pairs, `{props}`/`{surface}`, `hold:`, and `target_prompt` in `h3_pair_compiler.py`.
2. **`jlens-holistic-readout` (`0d2ad8a9`)** — **Landed on `main`**:
   - `basis`/`basis_note` in `assess_steering_readiness`, `token_heatmap` in `analyze_prompt_coupling`, `decode_recipe`, `save_composed_recipe_vector`, and `"->"` / `"|||"` split in `synthesize_from_words`.
3. **`cvman-holistic-workbench` + `dials-holistic-checks` (`4ea73adb`)** — **Landed on `main`**:
   - Enforced Invariants 1–5 across `control_vector_nodes.py`, `control_vectors.py`, `h3_spans.py`, `h3_pair_compiler.py`, `j_lens.py`, `bench/control_vectors.py`, `bench/control_vector_ui.html`, and `docs/wiki/control_vectors.md`, with `C1`–`C7` green in `bench/check_control_vectors.py` and all worktrees removed.
