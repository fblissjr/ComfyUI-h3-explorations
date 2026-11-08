# 5. Preset Catalog, File Contracts & CLI Reference

**Parent guide**: [`docs/wiki/control_vectors.md`](../control_vectors.md)
**Source files**: [`bench/control_vectors.py`](../../../bench/control_vectors.py), [`control_vectors.py`](../../../control_vectors.py), [`control_vector_nodes.py`](../../../control_vector_nodes.py)

---

## 1. Complete CLI Reference (`bench/control_vectors.py`)

Always invoke with the ComfyUI venv Python (`PY="$(cd ../.. && pwd)/.venv/bin/python"`):

| Subcommand | Example Command | What It Does |
|---|---|---|
| **`ui`** | `"$PY" bench/control_vectors.py ui` | Launches the local 6-screen Control Vector Studio web app (`127.0.0.1`). |
| **`rebuild`** (all in folder or single vector) | `"$PY" bench/control_vectors.py rebuild` *(or `rebuild /path/to/folder` or `rebuild film_16mm_1974`)* | Dynamically discovers and rebuilds all control vectors in a folder (defaulting to `models/control_vectors/` + `prompt_bank/control_vectors/` with zero hardcoded vector names) or a single specified control vector (`.toml`, `.safetensors`, or stem), sharing a single loaded Qwen3-VL encoder pass. |
| **`build`** (from clause contrast) | `"$PY" bench/control_vectors.py build --words "clear mild air -> scorching heatwave" --name scorching --save-toml --overwrite` | Compiles 10 H3 scene pairs from `Pole A -> Pole B`, optionally writes `prompt_bank/control_vectors/<name>.toml` (`--save-toml`), and builds both `fl2va` and `ref2va` `.safetensors` files. |
| **`build`** (from `.md` or `.toml`) | `"$PY" bench/control_vectors.py build prompt_bank/control_vectors/film_16mm_1974.toml --families both --overwrite` | Builds `fl2va` and `ref2va` `.safetensors` vectors from an LLM Markdown file or `.toml` pair bank. |
| **`generate-pairs`** | `"$PY" bench/control_vectors.py generate-pairs --concept "calm air -> scorching heatwave" --name scorching --save-toml --overwrite` *(or pass `<concept>` positionally)* | Generates 10 surgical Prompt A / Prompt B pairs using a local LLM (`transformers` or `exllamav3`) and `h3_control_vector_system_prompt.md`, automatically repairing any multi-clause drift. |
| **`make-media`** | `"$PY" bench/control_vectors.py make-media --image-a flat.png --image-b graded.png --output-name film_grade --families ref2va --overwrite` | Builds a media control vector (`B - A`) from contrasting reference images (`image_a` baseline vs. `image_b` positive/steered) snapped to an identical 32-px M-RoPE grid (`--short-edge 512`). |
| **`sculpt`** | `"$PY" bench/control_vectors.py sculpt blizzard --invert --name anti_blizzard` | Inverts (`--invert`), blends (`--modes add,add_norm_matched`), or orthogonalizes (`--hold`, `--hold-words`) existing vectors in `< 5 ms` on CPU. |
| **`words`** | `"$PY" bench/control_vectors.py words "scorching, desert, heatwave, glare" --name scorching_fast` | Synthesizes `fl2va` and `ref2va` vectors in `< 2 ms` on CPU from single English words via J-Lens. |
| **`from-md`** | `"$PY" bench/control_vectors.py from-md my_pairs.md --name custom_look --overwrite` | Converts an LLM Markdown document (authored with `h3_control_vector_system_prompt.md`) into a validated `.toml` pair file. |
| **`inspect`** | `"$PY" bench/control_vectors.py inspect film_16mm_1974__fl2va__diff_span__lora_2a8c2b52.safetensors --prompt "A wide shot..."` | Prints vector metadata, recipe compatibility (`--check-dit`), J-Lens readout, and contextual per-token receptor metering. |
| **`grade`** | `"$PY" bench/control_vectors.py grade --vector blizzard__fl2va__diff_span__lora_2a8c2b52.safetensors --prompt t2va_look_anchor --no-dit` | Grades post-refiner and optional DiT velocity movement against `flip_null`, `random_null`, and prompt-edit baselines. |

### Key CLI Flags (`--dit` Replaces `--unet`)
- `--dit <checkpoint.safetensors>` — Explicit DiT diffusion checkpoint filename override (replaces legacy `--unet`, which remains accepted as a hidden alias for backwards compatibility).
- `--check-dit <checkpoint.safetensors>` — On `inspect`, verifies that the vector's `recipe.checkpoint` matches the given DiT checkpoint (replaces legacy `--check-unet`).
- `--families {both,fl2va,ref2va,fl2va,ref2va}` — Which H3 model families to build in one pass (default `both`).
- `--pooling {diff_span,from_first_diff,scene_text,non_vision,all_rows,all}` — Token span pooling strategy (default `diff_span`).
- `--save-toml` — When building with `--words`, also saves the compiled 10-pair `.toml` file under `prompt_bank/control_vectors/<name>.toml`.
- `--invert` — On `sculpt`, flips the primary vector's sign (`strength = -1.0`).

---

## 2. File Naming & `.safetensors` Metadata Contract

Every built vector is saved under `models/control_vectors/` (gitignored) with an auto-generated Markdown sidecar report (`<filename>.md`):

```text
<stem>__<family>[__<pooling>][__ref_<sha8>|__prefix_<sha8>][__noholds][__lora_<sha8>].safetensors
```
*(where `pooling` is omitted only when `pooling == "scene_text"`)*

Examples (`control_vectors.py::default_vector_filename`):
- `film_16mm_1974__fl2va__diff_span__lora_2a8c2b52.safetensors` *(Pika `fl2va` LoRA recipe)*
- `film_16mm_1974__ref2va__diff_span__lora_865ee189.safetensors` *(Pika `ref2va` LoRA recipe)*
- `film_16mm_1974__fl2va__diff_span.safetensors` *(base checkpoint with `--no-lora`)*

Each `.safetensors` file stores:
- `vector.post_refiner`: `float32[5376]` tensor in natural mean-pair-difference units (`NORM_MATCHED_TARGET = 67.5` when applied with `mode = "add_norm_matched"`).
- `vector.encoder` (optional, default included): unit-normalized `float32[5120]` tensor (`row 0` excluded).
- `holds.post_refiner` (when `[[hold]]` pairs are present): `float32[K, 5376]` orthonormal hold basis.
- Top-level JSON metadata keys in `__metadata__` matching `FORMAT_VERSION = "h3-control-vector/1"` (`format`, `name`, `quality`, `family`, `pairs`, `recipe`, `measured_under`, `points`, `quality_report`, `provenance`).

---

## 3. Starter `.toml` Pair Banks (`prompt_bank/control_vectors/`)

The 32 `.toml` files in `prompt_bank/control_vectors/` are **starter examples and editable templates** — you can load any of them into the Studio UI (**Screen 3 → Load Existing `.toml`**), inspect how its Prompt A / Prompt B pairs are structured, or invert/combine them in **Screen 5 (`Sculpt`)**:

| Category | Starter `.toml` Stems in `prompt_bank/control_vectors/` (`32` total) |
|---|---|
| **Film Stock, Era Look & Lighting** | `film_16mm_1974`, `noir`, `cyberpunk_neon_rain`, `vhs_1991_camcorder`, `cursed_cctv`, `strobe_club_pulse`, `bioluminescent_glow`, `thermal_infrared`, `wes_anderson_symmetry`, `miniature_tiltshift`, `lettering` |
| **Material, Animation & Surreal Transforms** | `claymation_stopmotion`, `felt_muppet`, `paper_origami`, `rubber_hose_1930s`, `crystallize_ice`, `fever_melt`, `smoke_dissolve`, `elastic_neck_stretch`, `head_swell` |
| **Physics, Weather & Environment** | `underwater_submerged`, `wind_tunnel_gale`, `zero_gravity`, `magnetic_attraction` |
| **Camera, Action & Behavioral Dynamics** | `handheld_documentary_zoom`, `chaos_panic`, `infomercial_fail` |
| **Speech Delivery & Acoustic Space** | `auctioneer_rapid_patter`, `operatic_bellow`, `vocal_strain`, `whisper_asmr`, `reverb_cathedral` |
