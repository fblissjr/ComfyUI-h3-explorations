# 5. Preset Catalog & CLI Reference

[← Back to Control Vectors Overview](../control_vectors.md)

This page documents every CLI subcommand in [`bench/control_vectors.py`](../../../bench/control_vectors.py) ([`bench/control_vector/cli.py`](../../../bench/control_vector/cli.py)), the `.safetensors` filename & metadata contract, and the 32 generalizable multi-scene `.toml` banks in `prompt_bank/control_vectors/`.

---

## 1. CLI Subcommands by Creation / Inspection Role

Run all commands with the ComfyUI virtualenv Python (`COMFY="$(cd ../.. && pwd)" && PY="$COMFY/.venv/bin/python"`):

### A. The Four Creation Subcommands

| Subcommand (and Aliases) | Purpose & Key Flags | Example |
|---|---|---|
| **`from-prompt`** | **Method 1 — In-Place Scene-Paired Contrast**: Substitutes `pole_a -> pole_b` directly inside `--target-prompt` (with optional `--ref-images` for `<Picture N>` vision conditioning).<br>• `--name <stem>` (required)<br>• `--target-prompt <path_or_text>` (required)<br>• `--words "pole_a -> pole_b; soundscape: a -> b; hold(g): a -> b"` (required)<br>• `--ref-images <img1.png> [...]` (optional)<br>• `--families {both,fl2va,ref2va}` (default `both`)<br>• `--hold-words "w1, w2"` (optional)<br>• `--save-toml [<path>]` (optional) | `"$PY" bench/control_vectors.py from-prompt --name diner_sitcom --target-prompt my_diner.txt --ref-images subject1.png --words "Cinematic. -> A 1990s multi-camera studio sitcom.; warm, amused voice -> punchy sitcom delivery" --families ref2va --overwrite` |
| **`from-toml`**<br>*(aliases: `build`, `make`)* | **Method 2 — Multi-Scene Reusable Library Vector**: Builds 10-pair `.toml` or `.md` pair files (or all 32 library `.toml` files with `--all`).<br>• `<pairs.toml\|.md>` or `--all`<br>• `--families {both,fl2va,ref2va}` (default `both`)<br>• `--pooling {diff_span,from_first_diff,scene_text,all_tag1}` (default `diff_span`)<br>• `--hold-words "w1, w2"` (optional)<br>• `--no-lora` (build base refiner instead of default Pika LoRA) | `"$PY" bench/control_vectors.py from-toml --all --families both --overwrite` |
| **`from-words`**<br>*(alias: `words`)* | **Method 3 — Instant `< 2 ms` CPU Unigram Synthesis via J-Lens**: Synthesizes a vector from vocabulary words on CPU with `0 GB` GPU VRAM.<br>• `<words>` (`"blizzard, snowstorm, frost"`)<br>• `--against-words` (`@neutral`, `@prompt_subject`, `@replace_human`, or comma list)<br>• `--hold-words "w1, w2"` (`--hold-alpha 1.0`)<br>• `--target-prompt <path_or_text>` (optional, for `@prompt_subject`)<br>• `--name <stem>` | `"$PY" bench/control_vectors.py from-words "blizzard, snowstorm, frost" --against-words "sunlight, clear, mild" --hold-words "coat, parka" --name fast_blizzard` |
| **`from-images`**<br>*(alias: `make-media`)* | **Method 4 — Visual Look Transfer (`Image A -> Image B`)**: Extracts the visual difference between two reference stills across `pooling = "scene_text"`.<br>• `--name <stem>` (required)<br>• `--image-a <baseline.png>` & `--image-b <target.png>` (required)<br>• `--short-edge 512` (32-px M-RoPE grid snap)<br>• `--pooling {scene_text,vision_tokens,all_tag1,all_rows}` | `"$PY" bench/control_vectors.py from-images --name look_grade --image-a flat.png --image-b graded.png --overwrite` |

### B. Sculpting, Rebuilding, J-Lens Maintenance & Diagnostics

| Subcommand | Purpose & Key Flags | Example |
|---|---|---|
| **`sculpt`** | **`< 5 ms` CPU Subspace Sculptor**: Invert (`--invert`), blend (`--modes`, `--strengths`), or orthogonalize (`--hold`, `--hold-words`) existing vectors on CPU. | `"$PY" bench/control_vectors.py sculpt blizzard --invert --name anti_blizzard` |
| **`rebuild`** | **Batch or Single-Vector Re-Extraction**: Rebuilds all vectors in `models/control_vectors/` (or a specific folder/stem) from their embedded `metadata["pairs"]` or `.toml` source in a single shared model pass. | `"$PY" bench/control_vectors.py rebuild` |
| **`build-j-lens`** | **Verify or Rebuild `models/j_lens/j_lens_h3.safetensors`**:<br>• `--verify-only` (`< 50 ms` CPU schema/round-trip check)<br>• `--num-prompts 24` (recompute `Sigma_49_p030`, `J_ref_*`, and `J_wiener_*` from local checkpoints) | `"$PY" bench/control_vectors.py build-j-lens --verify-only` |
| **`inspect`** | **Inspect Metadata, J-Lens Readouts & Prompt Receptor Coupling**: Prints tensor norms, split-half cosine, `j_lens (+)/(-)`, and optional `--prompt` / `--meter` token coupling. | `"$PY" bench/control_vectors.py inspect film_16mm_1974__fl2va__diff_span__lora_2a8c2b52.safetensors --prompt my_scene.txt` |
| **`generate-pairs`** / **`from-md`** | **LLM Pair Authoring**: Runs a local LLM (`--llm-provider transformers` or `exllamav3`) with `h3_control_vector_system_prompt.md`, or converts an external LLM `.md` response into `.toml` (`from-md`) or `.safetensors` (`from-toml response.md`). | `"$PY" bench/control_vectors.py from-toml response.md --name custom_concept` |
| **`grade`** / **`ui`** | Grade activation shifts on captured tensors (`grade`), or start the browser Studio UI (`ui`). | `"$PY" bench/control_vectors.py grade --help` |

---

## 2. File Naming & `.safetensors` Metadata Contract

Every built control vector is written to `models/control_vectors/` as:
```text
<stem>__<family>__<pooling>[__lora_<sha8>].safetensors
<stem>__<family>__<pooling>[__lora_<sha8>].md
```
- **`<family>`**: `fl2va` (used by `t2va`, `i2va`, `fl2va`, `l2va`, `P-H3-Image` T2I, and 1–8 ref `P-H3-Image` Edit) or `ref2va` (used by `ref2va` and `P-H3-Image` Guide Edit).
- **`<pooling>`**: `diff_span`, `from_first_diff`, `scene_text`, or `all_tag1`.
- **`__lora_<sha8>`**: 8-hex digest of the attached `condition_proj` / `token_refiner` LoRA recipe (`2a8c2b52` for default `pika_h3_flex_fl2va_250steps_svd995_bf16.safetensors`, `865ee189` for default `pika_h3_flex_ref2va_300steps_svd995_bf16.safetensors`).
- **Tensors**: `vector.post_refiner` (`float32[5376]`) and `vector.encoder` (`float32[5120]`).
- **Header Metadata (`metadata["control_vector"]` JSON)**: Stores `schema_version`, `name`, `description`, `model_family`, `basis` (`"post_refiner"` or `"unigram_synthesis"`), `pooling`, `norm_post_refiner`, `split_half_cosine`, `recommended_strength_add_norm_matched`, `refiner_fingerprint`, `j_lens_readout`, and full `pairs` provenance (`pairs_ab`, `holds_ab`, `words_spec`, `target_prompt`) so `rebuild` is 100% lossless.

---

## 3. The 32 Generalizable Library `.toml` Banks (`prompt_bank/control_vectors/`)

`prompt_bank/control_vectors/` tracks 32 generalizable 10-pair `.toml` banks (each containing 10 diverse `Prompt A` / `Prompt B` scenes that work across any subject or setting):

| Category | Tracked `.toml` Stems in `prompt_bank/control_vectors/` |
|---|---|
| **Film Stock, Optics & Era Formats (8)** | `film_16mm_1974`, `vhs_1991_camcorder`, `noir_chiaroscuro`, `bleach_bypass_grit`, `infrared_aerochrome`, `cyberpunk_neon`, `golden_hour`, `miniature_tilt_shift` |
| **Atmosphere, Weather & Environments (6)** | `blizzard`, `scorching_heatwave`, `bioluminescent_forest`, `underwater_submerged`, `post_apocalyptic_ruins`, `renaissance_oil` |
| **Physics, Material & World Rules (6)** | `zero_gravity`, `cryogenic_freeze`, `time_freeze`, `slow_motion_float`, `claymation_stopmotion`, `anime_cel_shaded` |
| **Camera Rig, Motion & Energy (4)** | `handheld_urgency`, `kinetic_energy`, `chaos_panic`, `dolly_zoom_vertigo` |
| **Vocal Delivery & Acoustic Space (4)** | `whispered_intimate`, `cathedral_reverb`, `vinyl_crackle_warmth`, `muffled_underwater_audio` |
| **Anatomical, Prop & Material Transforms (4)** | `era_1970s_props`, `marble_statue_skin`, `puppet_strings_marionette`, `giant_colossus_scale` |

> [!NOTE]
> Scene-specific vectors tailored to a single shot (such as a `from-prompt` genre shift on one specific `ref2va` diner prompt) belong in your local `models/control_vectors/` folder or `--target-prompt` CLI commands rather than git-tracked `prompt_bank/control_vectors/*.toml` files.
