# Preset Catalog, File Contract, and CLI Reference

Last updated: 2026-11-11.

All command-line operations live in [`bench/control_vectors.py`](../../../bench/control_vectors.py) (with implementation modules under `bench/control_vector/`). Always invoke it with the ComfyUI virtualenv Python directly—never `uv run`.

---

## 1. CLI subcommands (`bench/control_vectors.py`)

| Subcommand (aliases) | GPU? | Purpose & key flags |
|---|---|---|
| `ui` | On demand | Launch the local Control Vector Studio web app (`--host 127.0.0.1`, `--port 0`, `--token`, `--no-browser`, `--vec-dir`, `--toml-dir`). See [`studio_ui_guide.md`](studio_ui_guide.md). |
| `build` (`from-toml`) | Yes (for `.toml`/`.md`/`--words`); CPU for `--from-words` | Build reusable `fl2va` and/or `ref2va` `.safetensors` + `.md` cards from a `.toml` or LLM `.md` file, `--all` across `prompt_bank/control_vectors/`, or `--words "pole_a -> pole_b"`. Key flags: `--families {both,fl2va,ref2va}`, `--pooling diff_span`, `--ref-images`, `--hold`, `--hold-words`, `--hold-alpha`, `--no-holds`, `--no-lora`, `--overwrite`. |
| `from-prompt` | Yes | Build an in-place scene vector from `--target-prompt` / `--target-prompt-file` + `--words "pole_a -> pole_b"`, **or** build a multi-scene vector directly from `prompt_bank/` (`--bank-scenes N`, supporting rotated Pole-A baselines `a1 \| a2 -> b` or target-only `--words "<target clause>"`). |
| `make` | Yes | Low-level single/dual-family pair builder sharing `encode_pair_set` + `pool_pair_set` with `build` and `MiniMaxH3MakeControlVector`. |
| `make-media` (`from-images`) | Yes | Build a vector from contrasting `<Picture 1>` images (`--image-a`, `--image-b`, `--scene-prompt`, `--short-edge 512`, `--pooling scene_text`). |
| `words` (`from-words`) | **No (`< 2 ms` CPU)** | Synthesize `fl2va` + `ref2va` `.safetensors` vectors from single-token English words (`pos_words`, `--against-words`, `--hold-words`, `--hold-alpha`, `--prompt`). |
| `sculpt` | **No (`< 5 ms` CPU)** | Compose existing `.safetensors` vectors (`--modes add,add_norm_matched,hold`, `--strengths`, `--invert`, `--hold`, `--hold-words`, `--hold-alpha`) into a new `.safetensors` file. |
| `inspect` | **No (`< 15 ms` CPU)**; GPU only with `--meter` | Print provenance, norms, J-Lens top `+`/`-` words, readiness recommendation, and optional `--prompt` receptor scan. `--meter <prompt>` measures live post-refiner shift against the prompt's row median (defaults to `--mask scene_only`, matching `MiniMaxH3Steer.target` as of `f8cdd538`). |
| `grade` | Yes (`--no-dit` skips DiT) | Numerical pre-render evaluation across `--strengths` in `--mode` (default `add_norm_matched`, matching the node as of `1a216404`) and `--mask` (default `scene_only`, supporting single fields and `field_a+field_b`), reporting context cosine/norm and DiT velocity shifts vs. `perm_null` and `flip_null`. |
| `from-md` (`import-md`) | **No (CPU)** | Parse and validate an LLM Markdown response (`### Test Scene Prompt` + `#### Pair 1..N` + optional `#### Hold`) into `data/control_vectors/<stem>.toml` (`--out`, `--test-prompt-out`). |
| `generate-pairs` | Yes (LLM) | Run a local `transformers` (4-bit default) or `exllamav3` LLM with [`h3_control_vector_system_prompt.md`](../system_prompts/h3_control_vector_system_prompt.md) to author a 10-pair `.md` / `.toml` pack from a concept string. |
| `rebuild` | Yes | Rebuild one stem (`--vector <stem>`) or every `.safetensors` vector in `models/control_vectors/` from `prompt_bank/control_vectors/` (`--families both`, `--force`). |
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

### Top-level JSON `.safetensors` metadata (`format = "h3-control-vector/1"`)
Validated on every load by [`control_vectors.py::load_vector`](../../../control_vectors.py):
- `format` (`"h3-control-vector/1"`), `name`, `quality`, `family` (`"fl2va"`, `"ref2va"`, `"fasth3"`, or `"custom:<12 hex chars>"`).
- `recipe` (`{checkpoint, loras, encoder}`), `measured_under` (`{mode, frame, prefix, pooling}`), `points` (`post_refiner: {unit: "mean_pair_difference", raw_norm, median_row_norm}`, plus `encoder` when saved).
- `pairs` (`{path, sha256, n_pairs, n_holds}`, plus `differing_sections`, `a_side`, `b_side`, and per-pair `pooled_tokens` produced by `control_vectors.pairs_record`).
- `quality_report` (`{mean_pairwise_cosine, split_half_cosine, per_pair_cosine}`), `provenance` (`{core_revision, pack_commit, created_utc, made_by}`), and optional `j_lens` (top `+`/`-` word gains, `b_side_alignment`, `refiner_transmission`, `off_target_words`).

---

## 3. Starter bank (`prompt_bank/control_vectors/*.toml`, 38 packs)

Every `.toml` file below is checked by `bench/check_control_vectors.py` and builds cleanly for both `fl2va` and `ref2va`:

| Category | Pack stems (`prompt_bank/control_vectors/<stem>.toml`) | Recommended runtime `target` (`mask`) & notes |
|---|---|---|
| **Film stocks, cameras & optical looks** (8) | `film_16mm_1974`, `film_16mm_1974_neutral` *(holds `warm_colour_cast`)*, `noir`, `vhs_1991_camcorder`, `cursed_cctv`, `thermal_infrared`, `miniature_tiltshift`, `wes_anderson_symmetry` | `scene_only` for whole shot; `description_only` on `ref2va` to grade the room while keeping `<Subject 1>` clean. |
| **Lighting, atmosphere & world environments** (6) | `cyberpunk_neon_rain`, `bioluminescent_glow`, `strobe_club_pulse`, `underwater_submerged`, `sunken_coral_reef`, `lettering` | `scene_only`, `strength = 1.0..1.75`. |
| **Stop-motion, animation & material replacement** (6) | `claymation_stopmotion`, `felt_muppet`, `paper_origami`, `rubber_hose_1930s`, `crystallize_ice`, `molten_lava_golem` | `scene_only`, `strength = 1.5..2.25`, `end_percent = 0.50`; on `ref2va` set `MiniMaxH3ReleaseReferences` `end_percent = 0.16..0.33` so live-action `<Picture 1>` VAE latents release silhouette/skin. |
| **Body morphs & surreal physics** (8) | `balloon_head_inflate`, `head_swell`, `elastic_neck_stretch`, `fever_melt`, `smoke_dissolve`, `zero_gravity`, `magnetic_attraction`, `wind_tunnel_gale` | `scene_only`, `end_percent = 0.50` at high strength (`>= 2.0`); lower `MiniMaxH3ReleaseReferences` `end_percent` on `ref2va` for geometry morphs. |
| **Action, pyrotechnics & physical comedy** (3) | `pyro_chain_explosions`, `chaos_panic`, `infomercial_fail` | `scene_only` (carries coupled foley when `overall_soundscape:` differs in pairs; isolate streams via `MiniMaxH3Steer.streams` if needed). `chaos_panic` also works great wired into `MiniMaxH3Steer.hold_vector`! |
| **Camera rig movement** (1) | `handheld_documentary_zoom` | `scene_only` or `description_only`, `strength = 0.75..1.5`. |
| **Acoustic space & vocal delivery** (6) | `reverb_cathedral`, `pyro_klaxon_audio`, `whisper_asmr`, `operatic_bellow`, `vocal_strain`, `auctioneer_rapid_patter` | **`target = "soundscape_only"`** (mandatory so `overall_soundscape:` rows are targeted instead of visual prose). |

Offline invariant suite (runs on CPU with `CUDA_VISIBLE_DEVICES=`):
```bash
CUDA_VISIBLE_DEVICES= /path/to/comfy-venv/bin/python bench/check_control_vectors.py
```
