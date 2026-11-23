# Preset Catalog, File Contract, and CLI Reference

Last updated: 2026-11-22 (`48` active tracked packs in `prompt_bank/control_vectors/*.toml`, `5` retired packs in `_retired/`, `--families h3_image` across all `41` visual stems, `--force-low-pair-erasure` / `MIN_ERASURE_PAIRS = 6`, `erasure.mu_a` vision centering via `--pooling vision_tokens`, Gate B `med_fd < 10` & Gate D lint, `1D_E%` / `|mu_d|/E|d|` telemetry, `sculpt` default `add_norm_matched_erasure`, and `generate-pairs --from-concept/--repair` via `heylook`).

All command-line operations live in [`bench/control_vectors.py`](../../../bench/control_vectors.py) (with implementation modules under `bench/control_vector/`). Always invoke it with the ComfyUI virtualenv Python directly—never `uv run`.

---

## 1. CLI subcommands (`bench/control_vectors.py`)

| Subcommand (aliases) | GPU? | Purpose & key flags |
|---|---|---|
| `build` (`from-toml`) | Yes (for `.toml`/`.md`/`--words`); CPU for `--from-words` | Build reusable `fl2va` (default; shared across `{fl2va, ref2va}` at `cos ~0.99`), `ref2va`, `fasth3`, and/or native `h3_image` (`J_ref_h3_image`, zero LoRAs) `.safetensors` + `.md` cards from a `.toml` or LLM `.md` file, `--all` across `prompt_bank/control_vectors/`, or `--words "pole_a -> pole_b"`. Key flags: `--families {both,fl2va,ref2va,h3_image}` (default `fl2va`), `--pooling diff_span`, `--ref-images`, `--hold`, `--hold-words`, `--hold-alpha`, `--no-holds`, `--no-lora`, `--overwrite`, `--force-low-pair-erasure` (compiled `< 6`-pair `--words` builds skip `erasure.vr` unless passed). Logs `1D_E%` / `|mu_d|/E|d|` erasure alignment. *(For browser UI authoring/building, use `/lab` inside H3 Pocket Console — see [`studio_ui_guide.md`](studio_ui_guide.md); the former standalone `ui` subcommand was retired 2026-11-22.)* |
| `from-prompt` | Yes | Build an in-place scene vector from `--target-prompt` / `--target-prompt-file` + `--words "pole_a -> pole_b"`, **or** build a multi-scene vector directly from `prompt_bank/` (`--bank-scenes N`, supporting rotated Pole-A baselines `a1 \| a2 -> b` or target-only `--words "<target clause>"`; skips `erasure.vr` when compiled `< MIN_ERASURE_PAIRS = 6` pairs unless `--force-low-pair-erasure` is passed). |
| `make` | Yes | Low-level single/dual-family pair builder sharing `encode_pair_set` + `pool_pair_set` with `build` and `MiniMaxH3MakeControlVector` (supports `--force-low-pair-erasure`). |
| `make-media` (`from-images`) | Yes | Build a vector from contrasting `<Picture 1>` images (`--image-a`, `--image-b`, `--scene-prompt`, `--short-edge 512`, `--pooling scene_text` or `vision_tokens` emitting dataset-centered `erasure.mu_a` via `build_spatial_vision_erasure`). *(Ablation 7 / Dual-Lock `fl2va` note: image->text patch erasure does not override raw VAE `cond_video_latents` chroma lock across steps `[0, 0.33]` or replace 10-pair `.toml` for global palette overrides.)* |
| `words` (`from-words`) | **No (`< 2 ms` CPU)** | Synthesize `.safetensors` vectors from single-token words + `Model2Vec` multi-token/2-word lexicon (`pos_words`, `--against-words`, `--hold-words`, `--hold-alpha`, `--prompt`). *(Ablation 6 sketchpad — does not replace 10-pair contextual erasure.)* |
| `sculpt` | **No (`< 5 ms` CPU)** | Compose existing `.safetensors` vectors (`--modes` defaults to `add_norm_matched_erasure` matching `MiniMaxH3Steer`; also `add_norm_matched`, `add`, `hold`), preserving up to rank-`16` union erasure (`--strengths`, `--invert`, `--hold`, `--hold-words`, `--hold-alpha`). |
| `inspect` | **No (`< 15 ms` CPU)**; GPU only with `--meter` | Print provenance, norms, J-Lens top `+`/`-` words, `erasure.mu_d`/`vr[0..3]` X-Ray readout, readiness recommendation, and optional `--prompt` receptor/erasure scan. `--meter <prompt>` measures live post-refiner shift against the prompt's row median (defaults to `--mask scene_only`). |
| `grade` | Yes (`--no-dit` skips DiT) | Numerical pre-render evaluation across `--strengths` in `--mode` (`add_norm_matched_erasure` default when erasure is present, `add_norm_matched`, `add`, `project_out`, `amplify`) and `--mask` (default `scene_only`, supporting single fields and `field_a+field_b`), reporting context cosine/norm and DiT velocity shifts vs. `perm_null` and `flip_null`. |
| `from-md` (`import-md`) | **No (CPU)** | Parse and validate an LLM Markdown response (`### Test Scene Prompt` + `#### Pair 1..N` + optional `#### Hold`) into `data/control_vectors/<stem>.toml` (`--out`, `--test-prompt-out`). |
| `generate-pairs` | `heylook` API or local LLM | Author a Gates A–D validated 10-pair `.md` / `.toml` pack (`heylook` default provider with `HEYLOOK_API_KEY` / explicit `0640` or `0600` `HEYLOOK_API_KEY_FILE` or `H3_HEYLOOK_API_KEY_FILE` / mode-`0600` `~/.heylook/api_key` auth support, `--from-concept <toml>` for `data/edit_concepts/` synthesis, `--repair <toml>` for in-place pair repair, or local `transformers`/`exllamav3` with [`h3_control_vector_system_prompt.md`](../system_prompts/h3_control_vector_system_prompt.md)). |
| `rebuild` | Yes | Rebuild one stem (`--vector <stem>`) or every `.safetensors` vector in `models/control_vectors/` from `prompt_bank/control_vectors/` (omitting `--families` preserves existing on-disk families per stem and defaults new stems to `fl2va`; auto-upgrades pair vectors missing `erasure.{vr,diag}` even without `--force`, honoring `--force-low-pair-erasure` on compiled low-pair builds). |
| `status` | CPU (`--rebuild-stale` uses GPU) | Check J-Lens bundle & control vector staleness/permissions (`--families fl2va` default, `--rebuild-stale` to fix). |
| `build-j-lens` | `--verify-only` is CPU; full build is GPU | Verify (`--verify-only`) or recalibrate `models/j_lens/j_lens_h3.safetensors` (`--base-bundle`, `--num-prompts 24`, `--max-tokens 64`). |

---

## 2. `.safetensors` filename and metadata contract

Built vectors live in `<ComfyUI>/models/control_vectors/` alongside a human-readable `<same_stem>.md` J-Lens report card:

```text
<name>__<family>__<pooling>[__lora_<8hex>].safetensors
```
Example: `film_16mm_1974_neutral__ref2va__diff_span__lora_865ee189.safetensors` (presented in `MiniMaxH3Steer.vector` and `MiniMaxH3LoadControlVector.vector` as the deduplicated stem `film_16mm_1974_neutral`, and resolved to the loaded checkpoint's `fl2va`/`ref2va` + LoRA sibling at run time).

### Tensors stored inside each `.safetensors` file
- `vector.post_refiner` (`float32 [5376]`): raw mean `B - A` post-refiner direction (after any `[[hold]]` / `hold_words` projection). Applied at runtime by `MiniMaxH3Steer` (and legacy `MiniMaxH3ApplyControlVectors`).
- `vector.encoder` (`float32 [5120]`, optional, included by default): unit-norm `layers.49` direction (token `0` excluded).
- `holds.post_refiner` (`float32 [K, 5376]`, when `[[hold]]` groups exist): orthonormal Gram-Schmidt hold basis.
- `erasure.mu_d` (`float32 [5376]`), `erasure.vr` (`float32 [r, 5376]`, orthonormal, `r <= 8`), `erasure.diag` (`float32 [r]`, in `[0, 1]`), and optional `erasure.mu_a` (`float32 [5376]` dataset Pole-A centering for `pooling="vision_tokens"` via `steering/extract.py::build_spatial_vision_erasure`), omitted cleanly on compiled `<compiled:...>` builds with `< MIN_ERASURE_PAIRS = 6` unless `--force-low-pair-erasure` is passed (`N_A >= 2` for file `.toml` builds): row-softmax (`tau = 0.15`) Pole-A erasure basis & ridge weights used by default `MiniMaxH3Steer` `mode = "add_norm_matched_erasure"`.

### Top-level JSON `.safetensors` metadata (`format = "h3-control-vector/1"`)
Validated on every load by [`steering/extract.py::load_vector`](../../../control_vectors.py):
- `format` (`"h3-control-vector/1"`), `name`, `quality`, `family` (`"fl2va"`, `"ref2va"`, `"fasth3"`, `"h3_image"`, or `"custom:<12 hex chars>"`).
- `recipe` (`{checkpoint, loras, encoder}`), `measured_under` (`{mode, frame, prefix, pooling}`), `points` (`post_refiner: {unit: "mean_pair_difference", raw_norm, median_row_norm}`, plus `encoder` when saved).
- `pairs` (`{path, sha256, n_pairs, n_holds}`, plus `differing_sections`, `a_side`, `b_side`, and per-pair `pooled_tokens` produced by `control_vectors.pairs_record`).
- `quality_report` (`{mean_pairwise_cosine, split_half_cosine, per_pair_cosine}`), `provenance` (`{core_revision, pack_commit, created_utc, made_by}`), and optional `j_lens` (top `+`/`-` word gains, `b_side_alignment`, `refiner_transmission`, `off_target_words`).

---

## 3. Active Control Vector Bank (`prompt_bank/control_vectors/*.toml`, 48 packs)

Every active `.toml` file below passes `bench/check_control_vectors.py` and builds across `fl2va` (`48`) and `ref2va` (`48`); all `41` visual/look/weather/morph stems also build natively for `h3_image` (`137` `.safetensors` files in `bank/`). Each category shares one default inference recipe (`mode = "add_norm_matched_erasure"`, `NORM_MATCHED_TARGET = 67.5`, falling back cleanly to `add_norm_matched` when erasure is absent):

| Category | Active stems (`prompt_bank/control_vectors/<stem>.toml`) | Recommended `target` | `strength` | `[start, end)` | `ReleaseRefs` (`ref2va`) & `[[hold]]` notes |
|---|---|---|---|---|---|
| **Optical, film stocks & tone utilities** (11) | `blacklight_uv_rave`, `cursed_cctv` *(holds)*, `de_llm_slop` *(subtle de-gloss utility)*, `film_16mm_1974` *(holds)*, `film_16mm_1974_neutral` *(holds `warm_colour_cast`)*, `look_amber_candlelight`, `miniature_tiltshift`, `noir`, `thermal_infrared` *(holds)*, `vhs_1991_camcorder`, `x_ray_fluoroscope` | `scene_only` (`t2va`); `description_only` (`ref2va`) | `1.0 .. 1.5` (`0.6 .. 0.8` when stacking 3+ visual sliders; `1.2 .. 1.7` on `thermal_infrared`) | `0.0 .. 1.0` | Keep `<Picture 1>` `end_percent = 1.0` |
| **Lighting, weather & environments** (9) | `bioluminescent_glow` *(holds)*, `cyberpunk_neon_rain` *(holds)*, `env_torrential_rain_drench`, `golden_hour_glow`, `heavy_blizzard`, `power_outage_flashlight` *(darkens ambient lighting; no narrow flashlight cone in `t=1` check)*, `strobe_club_pulse` *(holds)*, `sunken_coral_reef` *(holds)*, `underwater_submerged` *(holds)* | `scene_only` (`t2va`); `description_only` (`ref2va`) | `1.0 .. 1.6` (`1.3 .. 1.8` on `sunken_coral_reef`) | `0.0 .. 0.83` (`0.0 .. 1.0` for lighting) | Keep `1.0` for lighting/weather; `0.33 .. 0.67` on `underwater_submerged` / `sunken_coral_reef` |
| **Stop-motion, animation & material morphs** (10) | `charcoal_sketch_animated`, `claymation_stopmotion` *(holds)*, `crystallize_ice` *(holds)*, `felt_muppet` *(holds)*, `knitted_wool_world`, `mercury_liquid_chrome`, `molten_lava_golem` *(holds)*, `rubber_hose_1930s` *(holds)*, `toy_brick_world`, `watercolor_wash` | `scene_only` (`t2va`); `look_only` or `scene_only` (`ref2va`) | `1.3 .. 1.9` | `0.0 .. 0.50` | **`end_percent = 0.16 .. 0.33`** loosens pixel lock on `ref2va` (Dual-Lock note: `tags == 0` vision tokens still preserve subject identity/palette cleanly at `0.33`) |
| **Body morphs & surreal physics** (8) | `balloon_head_inflate` *(holds)*, `elastic_neck_stretch` *(holds)*, `fever_melt` *(holds)*, `head_swell` *(holds)*, `magnetic_attraction` *(holds)*, `smoke_dissolve` *(holds)*, `wind_tunnel_gale` *(holds)*, `zero_gravity` *(holds)* | `scene_only` (`t2va`); `look_only` for body morphs / `description_only` for physics (`ref2va`) | `1.2 .. 1.9` | `0.0 .. 0.50` (body/gravity) or `0.17 .. 0.83` (gale/magnet) | `0.16 .. 0.33` for head/neck/smoke morphs (subject to Dual-Lock); `0.50 .. 1.0` for zero-gravity/magnet/gale |
| **Action, pyrotechnics & physical comedy** (3) | `chaos_panic` *(useful as `hold_vector`)*, `infomercial_fail` *(holds)*, `pyro_chain_explosions` | `scene_only` (`t2va`); `description_only` (`ref2va`) | `1.2 .. 1.7` | `0.0 .. 0.50` (or `0.17 .. 0.67` on `t2va`) | `0.33 .. 0.50` |
| **Camera rig movement** (1) | `handheld_documentary_zoom` *(holds)* | `scene_only` or `description_only` | `0.9 .. 1.5` | `0.0 .. 0.67` | Keep `1.0` |
| **Acoustic space & vocal delivery** (6) | `auctioneer_rapid_patter`, `operatic_bellow`, `pyro_klaxon_audio`, `reverb_cathedral`, `vocal_strain`, `whisper_asmr` | **`soundscape_only`** (pure audio) or `scene_only` (vocal delivery) | `1.0 .. 1.4` | `0.0 .. 1.0` | Keep `1.0` (`fl2va` + `ref2va` only; audio stems do not build `h3_image`) |

Offline invariant suite (runs on CPU with `CUDA_VISIBLE_DEVICES=`):
```bash
CUDA_VISIBLE_DEVICES= /path/to/comfy-venv/bin/python bench/check_control_vectors.py
```

*(Bank layout note: all `48` active packs live tracked in `prompt_bank/control_vectors/*.toml`; retired packs `earthquake_tremor`, `lettering`, `paper_origami`, `stained_glass_cathedral`, and `wes_anderson_symmetry` live in `prompt_bank/control_vectors/_retired/` with their artifacts in `<comfy>/models/control_vectors/_retired/`; unbuilt Nov-11/12 legacy pairs sit in `data/archive/control_vectors_2026-11-12_legacy/` while `data/control_vectors/` remains free for new interactive authoring.)*
