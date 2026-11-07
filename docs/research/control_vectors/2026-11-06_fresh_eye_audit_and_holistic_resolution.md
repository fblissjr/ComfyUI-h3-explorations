# 2026-11-06: Fresh-Eye Audit of H3 Control Vectors (`2026-11-05`–`2026-11-06`) and Holistic Resolution Blueprint

**Date**: 2026-11-06 (All 5 holistic invariants landed on `main` at `93dbe8d3`, `0d2ad8a9`, and `4ea73adb`)  
**Companion docs**: [`2026-11-06_holistic_resolution_proposal.md`](2026-11-06_holistic_resolution_proposal.md), [`2026-11-06_holistic_stack_review_and_invariants.md`](2026-11-06_holistic_stack_review_and_invariants.md)  
**Scope**: All control-vector work committed on `2026-11-05` (`42e3d3bb`..`eeae6225`) and `2026-11-06` (`7a791660`..`4ea73adb`).  
**Related docs**: [`../../wiki/control_vectors.md`](../../wiki/control_vectors.md), [`2026-11-04_h3_control_vectors.md`](2026-11-04_h3_control_vectors.md), [`2026-11-06_j_lens_and_refiner_jacobian.md`](2026-11-06_j_lens_and_refiner_jacobian.md), [`2026-11-06_mainodes_comparison_and_pika_roadmap.md`](2026-11-06_mainodes_comparison_and_pika_roadmap.md), [`2026-11-06_detangle_comparison_and_borrowable_concepts.md`](2026-11-06_detangle_comparison_and_borrowable_concepts.md), [`../../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md).

---

## 1. Timeline & Architecture Snapshot (`2026-11-05` Through `2026-11-06`)

### 1.1 Yesterday (`2026-11-05`, `42e3d3bb`..`eeae6225`)
- **Pika `token_refiner` LoRA extraction (`42e3d3bb`)**: Wired `h3_lora_branches` metadata into [`lora_branch.py`](../../../lora_branch.py#L193-L245) and [`control_vector_nodes.py::build_refiner_runner`](../../../control_vector_nodes.py#L439-L520) so Pika H3 Flex's 24 `token_refiner.blocks.{0,1}` LoRA tensors are applied during vector extraction (`cos(base, pika_svd995) = 0.9957–0.9967`).
- **ExLlamaV3 `Qwen3-VL-32B` `layers.49 -> lm_head` readout (`bc009d59`..`782e5832`)**: Built [`exl3_qwen3vl/label_h3_control_vectors.py`](exl3_qwen3vl/label_h3_control_vectors.py#L640-L721) for 7-prefix cloze log-odds verbalization and coupled-Gumbel counterfactual continuations.
- **Root-cause diagnosis of `scene_text` norm dilution (`877dc9ba`, `eeae6225`)**: Diagnosed why early physical-action renders (`Arms 25..30` in [`bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md#L76-L121)) under-steered: averaging over all ~105 scene tokens when only an 8–12 word clause differed diluted `raw_norm` by `2.12x–8.21x` (`10.35` under `scene_text` vs. `23.57` under `from_first_diff` and `85.01` under `diff_span` on `head_swell`). Added `differing_token_bounds` and `pooling="diff_span"` (`[first_diff:end_diff]`) in [`h3_spans.py`](../../../h3_spans.py#L695-L760).

### 1.2 Last ~9 Hours Today (`2026-11-06 08:55`..`17:40`, `7a791660`..`0d2ad8a9` + Active Worktrees)
1. **Bank standardization, `diff_span` default, and `Control Vector Studio` UI (`7a791660`..`65d76c5d`)**:
   - Standardized `non_diegetic_music: N/A` across all `.toml` banks (`07915ae7`), made `pooling="diff_span"` the default across all CLI and ComfyUI entry points (`65d76c5d`), and rebuilt all **64 `diff_span` `.safetensors` vectors** (`32` banks $\times$ `fl2va`/`ref2va` after retiring 2 bare-frame E2.3 banks in `ceb1913c`) into a uniform `raw_norm` band (`49.45..87.03`, mean `split_half_cosine = 0.923`).
   - Added `mode="add_norm_matched"` (`NORM_MATCHED_TARGET = 67.5`, [`control_vectors.py`](../../../control_vectors.py#L2107)), dialogue/section inference masks (`scene_only`, `after_dialogue`, `description_only`, `soundscape_only` in [`h3_spans.py`](../../../h3_spans.py#L422-L492)), semantic pair linting ([`lint_pair_set`](../../../control_vectors.py#L816-L919)), and same-family Pika LoRA variant compatibility ([`_lora_family_signature`](../../../control_vectors.py#L2005-L2047)).
2. **The Jacobian Lens (`j_lens.py`, `efbef735`, `c90997fc`, `ceb1913c`..`0d2ad8a9`)**:
   - Built [`j_lens.py`](../../../j_lens.py#L1-L2298) backed by `<comfy>/models/j_lens/j_lens_h3.safetensors`:
     - **`14.5 ms` CPU `5376`-d verbalization** via empirical-covariance Wiener inversion $J_{\text{Wiener}}^{(f)}$ and $\bar{J}_{49\to 63}$ across 7 syntactic cloze baselines (`union_idx` = `27,407` tokens, reproducing the full `151,936`-token `log_softmax` partition function to `0.0098` nats).
     - **`< 2 ms` CPU word-to-vector synthesis** ([`synthesize_from_words`](../../../j_lens.py#L1709-L1855)) via spectral-filtered $\bar{J}_{\text{ref}}^{(f)}\Sigma_{49,\text{filt}}^{0.30}$ with `1/N` per-word pooling ([`_mean_word_direction`](../../../j_lens.py#L1686-L1706)) so multi-piece BPE words (`"8-bit"`, `"snowdrift"`) do not outweigh single-token words.
     - **Pre-render readiness, basis-aware readout, token receptor heatmaps, and recipe composition (`0d2ad8a9`)**: [`assess_steering_readiness`](../../../j_lens.py#L448-L625) (`basis="unigram_dictionary"` vs. `"post_refiner"` + `basis_note`), [`extract_prompt_subject_words`](../../../j_lens.py#L1265-L1375), [`analyze_prompt_coupling`](../../../j_lens.py#L1425-L1578) (`token_heatmap`), [`decode_recipe`](../../../j_lens.py#L728-L830), and [`save_composed_recipe_vector`](../../../j_lens.py#L1895-L2020).
3. **Canonical 10-pair compiler, multi-span `differing_token_mask`, runtime holds, AV stream scaling, `ref_vae_end_percent`, and media vectors (`fd4f246a`..`93dbe8d3`)**:
   - Landed [`h3_pair_compiler.py`](../../../h3_pair_compiler.py#L1-L1247) with shared lead-in wrappers (`690c6969`) and `{props}`/`{surface}` binding, coupled `description + overall_soundscape` pairs, `hold:` compilation, and `target_prompt` `Pair 0` (`weight = 2.5`) injection (`93dbe8d3`).
   - Landed multi-span `differing_token_mask` (`difflib.SequenceMatcher` with `min_shared_run=8` in [`h3_spans.py`](../../../h3_spans.py#L635-L714)), inference-time cross-vector `mode="hold"` (`fd4f246a`), `video_scale`/`audio_scale`/`split_end_percent` and `ref_vae_end_percent`/`ref_vae_aug` (`87c0d1de`), [`MiniMaxH3MakeMediaControlVector`](../../../control_vector_nodes.py#L2244-L2337) (`ef91d35c`), chained `MiniMaxH3LoRABranch` (`485e8045`), and Wave 1+2 Pika verification renders (`Arms 53..66`, `9dc63ef6`).
4. **3-stage contextual workbench, `sculpt` CLI, 5 invariants, and `C1`–`C7` checks (`4ea73adb`)**:
   - Landed the 3-stage `:8787` Workbench UI and `sculpt` CLI in [`bench/control_vectors.py`](../../../bench/control_vectors.py) and [`bench/control_vector/web/index.html`](../../../bench/control_vector/web/index.html), enforced all 5 system-wide invariants across [`control_vector_nodes.py`](../../../control_vector_nodes.py), [`control_vectors.py`](../../../control_vectors.py), [`h3_spans.py`](../../../h3_spans.py), and [`j_lens.py`](../../../j_lens.py), and verified all `C1`–`C7` assertions in [`bench/check_control_vectors.py`](../../../bench/check_control_vectors.py).

---

## 2. What's Good & Why

1. **Zero-Dilution `diff_span` Pooling Paired with Shared Lead-In Wrappers ([`h3_spans.py:635-714`](../../../h3_spans.py#L635-L714), [`h3_pair_compiler.py:288-361, 776-789`](../../../h3_pair_compiler.py#L288-L361))**:
   - **Why it works**: Because `Qwen3-VL` (`layers.0..49`) is strictly causal (`is_causal=True`), every token before `first_diff` is identical between Prompt $A$ and Prompt $B$ at the encoder (`||Delta e|| = 0`). Earlier today (`Arms 55..56` in [`bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md#L161-L162)), wrapping bare Pole-$B$ words inside distinct 5–8 word template clauses against a different Pole-$A$ default clause caused `diff_span` to average the template wrapper tokens (`top1 = +1.47..+1.60`, `ref_pass = 0.340`). Moving the grammatical lead-in (`"The scene is overtaken by "`, `"framed in a shot with "`, etc.) into the **shared prefix** of both Pole $A$ and Pole $B$ (`690c6969`) forces `first_diff` to land directly on the user's contrast words (`top1 = +3.32..+3.87`, `ref_pass = 0.502..0.525`).
2. **Broadcast Attention Value Path ($\beta = 0.5$) + Shrinkage-Covariance Wiener Inversion in [`j_lens.py`](../../../j_lens.py#L21-L51)**:
   - **Why it works**: Differentiating `TokenRefiner` on a single isolated token ($\beta = 0$, MLP + residual only) misses the cross-token $W_O W_V$ value accumulation when a steering vector is broadcast across a span ($\cos = 0.5091$ against true nonlinear `TokenRefiner` diffs). Including $\beta W_O W_V J_{\text{RMSNorm}_1}$, keeping the `52.4x` token-0 attention sink (`||H_49[:,0]|| = 15,309` vs. `292.3`) in the forward pass while excluding `[:, 0, :]` from covariance/RMSNorm expectations, and inverting with shrinkage covariance $\Sigma_{49}$ (`J_wiener`) raises downstream $\bar{J}_{49\to 63}$ cosine recovery to **`0.8778`** across all 35 dual vectors on disk.
3. **RoPE-Cursor-Invariant `ref_vae_end_percent` Gating ([`control_vector_nodes.py:1132-1185`](../../../control_vector_nodes.py#L1132-L1185))**:
   - **Why it works**: Dropping `payload["refs"]` on late sampling steps would shrink `PackedLayout.seq_len` by `7,360` tokens mid-sampling and shift the 3D RoPE temporal `cursor` (`ref_img` advances `cursor` before `video` in `MiniMaxH3Model._build_rope`). Zeroing non-keyframe `cond_video_latents` and setting `visual_cond_noise_aug = 0.0` while preserving `refs` and `layout` replaces the reference patches with seeded unit Gaussian noise at `seg_t["ref_img"] = max(t_v, 0.0) = t_v`, keeping `PackedLayout` and every 3D RoPE coordinate 100% invariant (`Arm 66`).
4. **32-px M-RoPE Grid Matching for Media Control Vectors ([`match_media_pair_geometry`](../../../control_vector_nodes.py#L1898-L1942))**:
   - **Why it works**: Snapping both `image_a` and `image_b` to a shared 32-px multiple before encoding guarantees identical `grid_thw`, identical vision token count $N_{\text{vis}}$, and bit-for-bit identical trailing text M-RoPE `position_ids`, while pooling `spans["scene"]` excludes the pre-vision `"<Picture 1>: "` label tokens that never attended to the image under causal `Qwen3-VL`.
5. **Inference-Time Cross-Vector `mode="hold"` + Section Masks (`description_only`, `soundscape_only`)**:
   - **Why it works**: Verified in `Arms 57..60`: holding out `chaos_panic` (`mode="hold", strength=1.0`) from `cursed_cctv` stripped the unscripted door crash (`audio RMS -64%`), yellow hazard sign, and messy linen pile while preserving 100% of the green-phosphor CCTV look; `mask="soundscape_only"` isolated `operatic_bellow` to the 35 soundscape tokens (`pixel MAD = 15.43` vs. `56.53` unmasked).

---

## 3. What Looks Off & Why (Disk-Verified Across All Sessions)

### 3.1 Runtime Caching, Validation & Silent Fallback Bugs in `control_vector_nodes.py`

1. **`MiniMaxH3ApplyControlVectors.fingerprint_inputs` Misses Cross-Family Sibling Resolution ([`control_vector_nodes.py:2547-2576`](../../../control_vector_nodes.py#L2547-L2576))**:
   - In ComfyUI's execution cache (`execution.py`), `fingerprint_inputs` (`IS_CHANGED`) runs before node execution without wired upstream outputs (`model` is not passed).
   - At `L2562`, `fingerprint_inputs` calls `resolve_vector_file_path(raw)` **without `preferred_family`**, whereas `attach_control_vector` (`L1601`) calls `resolve_vector_file_path(raw_name, preferred_family=model_family)`.
   - If a user selects an `__fl2va__` filename on a `ref2va` graph (which `resolve_vector_file_path` auto-redirects to the `__ref2va__` sibling at `execute()` time, `L1479-L1501`) and rebuilds the `__ref2va__` file on disk, `fingerprint_inputs` stats only the `__fl2va__` file and serves a stale cached model. (Note: `validate_inputs` passing `family="fl2va"` at `L2617` is benign because `fl2va` and `ref2va` share the identical `27,407`-token `vocab_union_indices` in `j_lens_h3.safetensors`.)

2. **Silent Fallback of `@prompt_subject` (`deferred_against_prompt`) When `_h3_span_masks` Is Absent ([`control_vector_nodes.py:1196-1242`](../../../control_vector_nodes.py#L1196-L1242))**:
   - Every shipped `t2va` and `ref2va` workflow in `workflows/` is built by [`workflows/build_workflows.py:1501-1635`](../../../workflows/build_workflows.py#L1501-L1635) with [`MiniMaxH3Conditioning`](../../../conditioning.py#L159) (`t2va`) or [`MiniMaxH3ReferenceConditioning`](../../../reference_conditioning.py#L1248) (`ref2va`) at node `"5"`, both of which call `attach_span_masks` and populate `_h3_span_masks` and `_h3_tail_text`.
   - However, if a user manually wires a core conditioning node (`CLIPTextEncode`, `MiniMaxH3ImageToVideo`, or `MiniMaxH3ReferenceToVideo`), `_h3_span_masks` is `None` (and cannot be reconstructed downstream by `MiniMaxH3Preflight` because `cond` carries only `[1, L, 5120]` floats and binary `0/1` `minimax_token_tags`, not token IDs).
   - When `_h3_span_masks` is `None`, `live_tail_text` at `control_vector_nodes.py:1202-1210` is `""`. Unlike `_resolve_custom_inference_mask` (`L1093`, which raises a clear `ValueError`), `_spec_effective_vector` (`L1240-L1241`) **silently falls back to `sp.vector`** (`AGAINST_PRESETS["replace_human"]`) without logging any warning.

3. **Non-Exclusive `vector_name` + Inline `words` Double-Injection on a Single Node ([`control_vector_nodes.py:1600-1730`](../../../control_vector_nodes.py#L1600-L1730))**:
   - On shipped workflows (`workflows/h3_text_to_video_pika_control_vector_api.json:223`), `vector_name` is pre-populated (`film_16mm_1974...`). If a user types inline `words` into that node without changing `vector_name` to `"none"`, `attach_control_vector` (`L1600-L1730`) attaches **both** the file vector and the synthesized word vector with the same `strength`, `mode`, and `hold_words`. Under `mode="add_norm_matched"` at `strength=1.15`, the node injects `77.6 + 77.6 = 155.2` total post-refiner shift.

---

### 3.2 Mathematical & Norm Inconsistencies Across Hold Paths and `vector.encoder`

4. **Four Conflicting L2-Norm & $\alpha$-Bound Semantics for Hold Projection**:
   Projecting out a hold direction currently has four different norm/bound behaviors across the codebase:
   - **Path 1 — Build-time `.toml` `[[hold]]` ([`control_vector_nodes.py:867`](../../../control_vector_nodes.py#L867), [`bench/control_vector/build_cmd.py:130-165`](../../../bench/control_vector/build_cmd.py#L130-L165))**: `final_vec = cv.project_out(raw_vec, basis)` does **not** restore `raw_vec`'s L2 norm (`head_swell` drops `85.01 -> 77.65`), and [`control_vectors.py::project_out`](../../../control_vectors.py#L1169) restricts `alpha` to `[0.0, 1.0]`.
   - **Path 2 — Inference-time chained `mode="hold"` ([`control_vectors.py:2110-2130, 2335`](../../../control_vectors.py#L2110-L2130))**: `_apply_holds_to_shift64` projects the hold out of `total_shift64` and does **not** restore L2 norm, and performs no bounds check on `h_alpha` (nor does `attach_control_vector` clamp `strength * _spec_stream_scale(...)` on `mode="hold"` at [`control_vector_nodes.py:1309, 1322`](../../../control_vector_nodes.py#L1309-L1322)).
   - **Path 3 — `j_lens.py::decode_safetensors` / `decode_recipe` / `save_composed_recipe_vector` ([`j_lens.py:686-689, 761, 1961`](../../../j_lens.py#L686-L689))**: applies `apply_soft_holds(v_pr, hold_tensors_pr, alpha=alpha)` (`alpha in [0.0, 2.0]`) **without** re-normalizing, reporting `norm_after_holds < norm_before_holds`.
   - **Path 4 — `MiniMaxH3ApplyControlVectors` `hold_words` on `vector_name` ([`control_vector_nodes.py:1633-1641`](../../../control_vector_nodes.py#L1633-L1641)) and `cvman-holistic-workbench`'s `_apply_post_refiner_holds` (`bench/control_vector/helpers.py:114-142` on `86592f0c`)**: **re-normalizes** the vector back to `orig_nrm` (`vec_held * (orig_nrm / res_nrm)`).
   - **Why this hurts**: A user who inspects a vector with `--hold-words` in `inspect` or `j_lens.py` sees the un-renormalized readout (`norm_after_holds < norm_before_holds`), while `MiniMaxH3ApplyControlVectors` silently rescales `vector_name` back to `orig_nrm` under `mode="add"`—and if over-projection (`alpha > 1.0`) is used, rescaling to `orig_nrm` amplifies the inverted nuisance direction.

5. **`vector.encoder` Ignores Both `[[hold]]` Projections and `--unit-mean` During `make` / `build` ([`control_vector_nodes.py:911-929`](../../../control_vector_nodes.py#L911-L929), [`bench/control_vector/build_cmd.py:160-185`](../../../bench/control_vector/build_cmd.py#L160-L185))**:
   - When `[[hold]]` pairs exist in a `.toml` file, `holds` are projected out of `vector.post_refiner` (`final_vec`), **never** out of `vector.encoder` (`enc_raw_vec`).
   - Likewise, `raw_vec` uses `unit_mean=use_unit_mean`, whereas `enc_raw_vec = cv.mean_difference(enc_a_mat, enc_b_mat, w_vec)` omits `unit_mean`.
   - Downstream, [`j_lens.py::decode_safetensors`](../../../j_lens.py#L703-L725) uses `vector.encoder` to compute `EN(+)`/`EN(-)`, `refiner_transmission = ||J_ref @ v_enc||_2`, `refiner_forward_cosine`, and `wiener_j63_cosine`—comparing an *un-held* `vector.encoder` against a *held* `vector.post_refiner`.

---

### 3.3 Prose-vs-Code Mismatch & Hold Clamping on AV Stream Split (`video_scale`, `audio_scale`)

6. **Per-Stream Conditioning + Velocity Splicing vs. Prose Velocity-Delta Formula ([`control_vector_nodes.py:1282-1400`](../../../control_vector_nodes.py#L1282-L1400) vs. [`docs/wiki/control_vectors.md:113`](../../wiki/control_vectors.md#L113))**:
   - [`docs/wiki/control_vectors.md:113`](../../wiki/control_vectors.md#L113) and [`docs/research/control_vectors/2026-11-06_mainodes_comparison_and_pika_roadmap.md:118`](2026-11-06_mainodes_comparison_and_pika_roadmap.md#L118) claim that `video_scale` and `audio_scale` blend a single steered pass with an unsteered base pass in velocity space ($\mathbf{u}_t^{\text{base}} + w\Delta\mathbf{u}_t$).
   - In reality, [`_build_steered_context`](../../../control_vector_nodes.py#L1289-L1372) constructs per-stream steered conditioning tensors `ctx_video` (scaling each spec by `video_scale`) and `ctx_audio` (scaling each spec by `audio_scale`), executes two DiT passes (`out_vid`, `out_aud`), and splices `[out_vid[0], out_aud[1]]` (`L1382-L1400`).
   - **Why the code's per-stream conditioning + splicing is architecturally superior**: Suppose a user chains Node 1 (`cursed_cctv`, `video_scale=1.0, audio_scale=0.0`), Node 2 (`operatic_bellow`, `video_scale=0.0, audio_scale=1.0`), and Node 3 (`vocal_strain`, `mode="hold", video_scale=0.0, audio_scale=1.0`). Per-stream conditioning + splicing evaluates `ctx_video` (visual style only) and `ctx_audio` (audio style minus vocal strain) on the DiT's nonlinear conditioning manifold in **at most 2 DiT passes total**, whereas a 2-pass `(out_base, out_steered)` velocity blend cannot steer video with Vector A and audio with Vector B without cross-contaminating both vectors in `ctx_steered`.
   - **What needs fixing in code and prose**:
     1. At [`control_vector_nodes.py:1309, 1322`](../../../control_vector_nodes.py#L1309-L1322), `s0 = sp.strength * _spec_stream_scale(sp, stream_attr)` is currently unclamped for `mode="hold"` (and `_apply_holds_to_shift64` at [`control_vectors.py:2110`](../../../control_vectors.py#L2110) has no bounds check). Clamping `s_eff = max(0.0, min(2.0, sp.strength * _spec_stream_scale(sp, stream_attr)))` when `_effective_entry_mode(sp) == "hold"` lets `video_scale=0.0, audio_scale=1.0` cleanly apply an audio-only hold (`alpha_vid=0.0, alpha_aud=1.0`) while guaranteeing a negative stream scale never inverts a hold into an addition.
     2. Update [`docs/wiki/control_vectors.md:113`](../../wiki/control_vectors.md#L113) so the prose states the per-stream conditioning + output-splice formula accurately.

---

### 3.4 Recipe Metadata, Morphological Stemmer, & Stale Documentation Links

7. **`j_lens.py::save_synthesized_vector` / `save_composed_recipe_vector` Hardcodes Pika LoRA Recipe Hashes ([`j_lens.py:127-164, 1882, 2005`](../../../j_lens.py#L127-L164))**:
   - `save_synthesized_vector` and `save_composed_recipe_vector` stamp `_DEFAULT_RECIPES[family]` (including `pika_h3_flex_*_svd995_bf16.safetensors` in `recipe.loras`) into every saved `.safetensors` file.
   - Loading that file on a stock 30-step base H3 UNet (`loras = []`) with default `allow_recipe_mismatch = False` fails `check_vector_compatibility` ([`control_vectors.py:2036-2047`](../../../control_vectors.py#L2036-L2047)), even though `j_lens` synthesizes on CPU from a single static `j_lens_h3.safetensors` bundle (`cos(base, pika_svd995) = 0.9966`).
   - **Important constraint (`C5` in [`bench/check_control_vectors.py:1570-1580, 1641-1658`](../../../bench/check_control_vectors.py#L1570-L1658))**: `C5` explicitly asserts that extracted `.safetensors` vectors with a LoRA refuse when loaded on `loras=[]` (`L1575`) or `blocks="0-24"` (`L1651`). Thus the automatic warning relaxation in `check_vector_compatibility` must be scoped specifically to `str(metadata.get("provenance", {}).get("made_by", "")).startswith("j_lens.")`.

8. **Morphological Stemmer Prefix Expansion Bug on `"operatic"` in [`j_lens.py::_resolve_word_indices`](../../../j_lens.py#L1650-L1666)**:
   - Discovered by `mrprompt`'s disk test: when resolving `"operatic"`, `_resolve_word_indices` (`L1650-L1665`) strips suffix `"ic"` $\to$ `stem = "operat"` (length 6 $\ge$ 4). Because `"operat"` is not in `word_to_idxs`, it falls through to `vw.startswith("operat") and len(vw) <= len(stem) + 4` (`L1660`), matching **`["operation", "operator", "operating", "operative"]`** instead of **`"opera"`** (which *is* in `word_to_idxs`)!
   - Checking vowel-restored / truncated roots (`stem[:-1]`, `stem + "e"`, `stem[:-2] + "a"` for `-atic`) in `word_to_idxs` before falling back to prefix expansion fixes `"operatic" -> "opera"`, `"dramatic" -> "drama"`, `"chromatic" -> "chroma"`, etc.

9. **Stale Line-Number Citations in [`docs/wiki/control_vectors.md`](../../wiki/control_vectors.md)**:
   - [`docs/wiki/control_vectors.md:94`](../../wiki/control_vectors.md#L94) links `resolve_vector_file_path` to `control_vector_nodes.py#L97-L162` (actually at [`control_vector_nodes.py#L1438-L1507`](../../../control_vector_nodes.py#L1438-L1507)).
   - [`docs/wiki/control_vectors.md:107`](../../wiki/control_vectors.md#L107) cites `control_vectors.py:926` for `NORM_MATCHED_TARGET = 67.5` (actually at [`control_vectors.py#L2107`](../../../control_vectors.py#L2107)).

---

### 3.5 Multi-Span Cache Eviction & Active Worktree Cleanup (`h3_spans.py` and `cvman-holistic-workbench`)

10. **Single-Span Eviction Gap in `_DIFF_MASK_CACHE` ([`h3_spans.py:708-713`](../../../h3_spans.py#L708-L713))**:
    - At [`h3_spans.py:708`](../../../h3_spans.py#L708), `differing_token_bounds(a, b)` only writes to `_DIFF_MASK_CACHE` when `not bool(mask_a[first_diff:end_a].all())` (i.e. when the pair is multi-span with an interior shared run $\ge 8$ tokens).
    - If a baseline `Prompt A` is first paired with a multi-span `Prompt B1` (populating `_DIFF_MASK_CACHE[(tuple(a), first_diff, end_a)]`) and later paired with a single-span `Prompt B2` that has the same outer `(first_diff, end_a)` bounds, `differing_token_bounds` does not evict the stale multi-span entry.
    - Evicting `_DIFF_MASK_CACHE.pop((tuple(a), first_diff, end_a), None)` (and `b`) in an `else:` branch at `h3_spans.py:713` and passing `other_ids` at the `pooling_mask` call sites (`control_vector_nodes.py`, `bench/control_vectors.py`) eliminates the collision completely.

11. **Remaining Invariant 2 & 3 Seams in `cvman-holistic-workbench` (`86592f0c`)**:
    - On `cvman-holistic-workbench` (`86592f0c`), `bench/control_vector/helpers.py:66` still has `_inject_target_prompt_into_pairset` (duplicating `h3_pair_compiler.py::compile_words_to_pairset` on `main`) and `bench/control_vector/helpers.py:114` still has `scrubbed = scrubbed * (norm_before / norm_after_raw)` inside `_apply_post_refiner_holds`.

---

## 4. Missed Opportunities (Verified on Disk)

1. **Intra-Word Cloze Admissibility Weighting in `j_lens.py::_mean_word_direction` (`< 0.1 ms` CPU)**:
   - `models/j_lens/j_lens_h3.safetensors` stores `cloze_base_union` (`[7, 27407]`), which maps to the `24,912` rows of `M_centered` via `valid_pos_in_union` (`[24912]`).
   - As `mrprompt` verified against disk:
     - **Within a single word's matched BPE variants `w_ids`** (e.g. `"pixel"` matches space-prefixed `" pixel"` at $\max_p \ell_p = -7.76$, `" Pixel"` at $-11.21$, and mid-word suffix `"pixel"` at $-15.75$), weighting the BPE variants within `w_ids` by $w_i = \sigma((\max_p \ell_p(u_i) + 11.5) / 2.0)$ cleanly down-weights mid-word continuation tokens (`0.866` vs. `0.536` vs. `0.107`) before unit-normalizing that word's vector.
     - **Across distinct words (`word_vecs`)**, however, every word must retain unit L2 norm (`1/N` per-word pooling) so high-prior modifiers (`"grain"`, $\max_p \ell_p = -5.03$, $\sigma = 0.962$) never outweigh specific nouns (`"blizzard"`, $\max_p \ell_p = -16.45$, $\sigma = 0.078$).
2. **Actionable Step-0 Receptor Remediation Hints in `MiniMaxH3ApplyControlVectors`**:
   - Step 0 (`control_vector_nodes.py:1242-1266`) already runs `_j_lens.analyze_prompt_coupling` and logs `[h3] control vector %s prompt coupling: %s`. Adding a one-line remediation hint when non-subject props or soundscape tokens dominate the top receptors (e.g., suggesting `mask="description_only"` or `hold_words`) turns that diagnostic into immediate workflow guidance.
3. **Within-Clip Temporal Envelopes Respecting H3's `FRAME_PER_TOKEN = (1, 4, 4, 4, 4)` Clock**:
   - Extending the 2-pass `DIFFUSION_MODEL` wrapper (`control_vector_nodes.py:1374-1400`) to support a temporal ramp $M(t) \in [0, 1]$ along video latent frames $T_{\text{lat}}$ (mapped via `1, 4, 4, 4, 4` pixel frames) and audio latent frames $T_{\text{aud}}$ (`40 Hz`) allows physical transformations (`zero_gravity`, `head_swell`) to ramp in after `<d>...</d>` dialogue finishes without shortening dialogue lines.

---

## 5. Holistic Resolution Blueprint (Converged Across All Sessions)

1. **Invariant 1 — Twin-Space Symmetry (`vector.encoder` & `vector.post_refiner`)**:
   - Pass `unit_mean=use_unit_mean` and project out the encoder-space `[[hold]]` basis (`cv.hold_basis` + `cv.project_out`) on `vector.encoder` (`5120`-d) identically to `vector.post_refiner` (`5376`-d) in `control_vector_nodes.py:911-929` and `bench/control_vector/build_cmd.py:160-185`.
2. **Invariant 2 — Hold Projection Never Secretly Re-Normalizes; $\alpha \in [0.0, 2.0]$ Everywhere**:
   - Remove `vec_held * (orig_nrm / res_nrm)` at `control_vector_nodes.py:1641` and in `cvman`'s `_apply_post_refiner_holds` (`bench/control_vector/helpers.py:114`), leaving magnitude standardization exclusively to `mode="add_norm_matched"` (`67.5`).
   - Unify `alpha` validation to `[0.0, 2.0]` in `control_vectors.py::project_out` (`L1169`) and `control_vectors.py::_apply_holds_to_shift64` (`L2110-2130`), and update `bench/check_control_vectors.py:1840-1844`.
3. **Invariant 3 — One Compiler Owner (`h3_pair_compiler.py`) & Exact Multi-Span Diff Masks (`h3_spans.py`)**:
   - Delegate all short-spec, `hold:`, `soundscape:`, and `target_prompt` compilation in `bench/control_vectors.py` directly to `h3_pair_compiler.py`.
   - Evict single-span keys in `h3_spans.py::differing_token_bounds` (`L708-713`) and pass `other_ids` to `pooling_mask` in `control_vector_nodes.py` and `bench/control_vectors.py`.
4. **Invariant 4 — Basis-Aware J-Lens Readout, Intra-Word Cloze Weighting, & Stemmer Accuracy (`j_lens.py`)**:
   - Fix `-atic -> -a` / `-e` root stem resolution (`stem[:-1]` and `stem + "e"`) before prefix expansion in `j_lens.py::_resolve_word_indices` (`L1658-1666`), and weight intra-word BPE variants `w_ids` by `piece_weights` (`torch.sigmoid((torch.log_softmax(cloze_base, dim=-1)[:, valid_pos].max(dim=0).values + 11.5) / 2.0)` in `load_j_lens_bundle` at `L211-221`) while keeping `1/N` unit-norm pooling (`F.normalize(w_vec, dim=0)`) across distinct words in `_mean_word_direction` (`L1697-1725`).
5. **Invariant 5 — Per-Stream Conditioning + Splicing, Clamped Stream Holds, & Node Hygiene (`control_vector_nodes.py`, `control_vectors.py`)**:
   - Keep 2-pass per-stream conditioning (`ctx_video`, `ctx_audio`) + `[out_vid[0], out_aud[1]]` splicing across all 5 `APPLY_MODES`, clamp `mode="hold"` stream-scaled strength to `[0.0, 2.0]` at `control_vector_nodes.py:1309, 1322`, and update `docs/wiki/control_vectors.md:113`.
   - Stat `preferred_family in (None, "fl2va", "ref2va")` in `fingerprint_inputs` (`L2547-2576`), log warnings when `@prompt_subject` lacks `_h3_span_masks` (`L1241`) or when both `vector_name != "none"` and `words` are set on one node (`L1600-1730`), and scope Pika-vs-base LoRA warning relaxation in `check_vector_compatibility` (`control_vectors.py:2036-2047`) to `provenance.made_by.startswith("j_lens.")`.
