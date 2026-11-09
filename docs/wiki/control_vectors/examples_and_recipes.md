# 6. Practical Examples & Recipes

[← Back to Control Vectors Overview](../control_vectors.md)

This page provides concrete, copy-pasteable recipes across all four creation methods (`from-prompt`, `from-toml`, `from-words`, `from-images`) and `< 5 ms` CPU sculpting (`sculpt`).

---

## Recipe 1: Aligning a `ref2va` Scene Prompt & Building Scene-Specific `from-prompt` Vectors

Suppose you have a single-subject `ref2va` diner scene and want to steer its genre, lighting, camera energy, or soundscape without altering the prompt text at render time.

### Step 1 — Ensure Your Baseline `ref2va` Prompt Is Strictly Aligned (`vendor_guides/ref_en.md`)
Before building a `from-prompt` vector, verify these four `vendor_guides/ref_en.md` alignment rules on your baseline `ref2va` prompt:
1. **`retention_analysis:` degree**: Use `completely_preserved` (or `fully_preserved`) whenever `<Subject 1>`'s face, hair, and identity from `<Picture 1>` are preserved as-is. Reserve `partially_preserved` only when intentionally keeping one attribute while altering another (e.g., keeping facial identity while replacing wardrobe).
2. **Style sentence before `[Shot 1]` in `detailed_description:`**: Open `detailed_description:` with a complete style/lighting sentence before `[Shot 1]` (e.g., `The target video uses a live-action cinematic style with warm natural sunlight. [Shot 1] ...`). Giving the opening style a clean clause also gives `from-prompt` an ideal `pole_a` anchor to replace!
3. **`[Language]` tag inside `<d>`**: Always write `<d>[English] What are you looking at? You like what you see?</d>`, never bare `<d>...</d>`.
4. **Clean spelling in `overall_soundscape:`**: Fix typos (`Atmospheric`, not `Atmospheic`) so `soundscape:` substring replacements match cleanly.

Save this aligned baseline prompt as `diner_ref2va.txt`:
```text
subject_definitions:
<Subject 1> is the young woman in <Picture 1>, with very long, loose, wavy blonde hair parted in the center and falling past her chest in sun-streaked strands, blue eyes, light freckles across her nose and cheeks.

summary:
[reference generation] <Subject 1> stands inside a diner looking around and speaking to the camera.

retention_analysis:
<Subject 1> (appears in [Shot 1]): completely_preserved - her face, the long wavy center-parted blonde hair, the light freckles, the blue eyes, and her identity from <Picture 1> are completely preserved.

detailed_description:
The target video uses a live-action cinematic style with warm natural sunlight. [Shot 1] Inside a sunlit diner, <Subject 1> stands in front of a table. The camera holds a medium shot as <Subject 1> (S1) says with an amused expression in a warm, amused voice, <d>[English] What are you looking at? You like what you see?</d> before settling her lips.

overall_soundscape:
Atmospheric diner sounds and faint murmurs of distant patrons.

non_diegetic_music:
N/A
```

### Step 2 — Build Scene-Paired `from-prompt` Vectors (With Optional `--ref-images`)

#### Example 1A: 1990s Multi-Camera Studio Sitcom (Coupled Style + Vocal Delivery + Live Audience Laugh Track)
```bash
COMFY="$(cd ../.. && pwd)"
PY="$COMFY/.venv/bin/python"

"$PY" bench/control_vectors.py from-prompt \
  --name diner_90s_sitcom \
  --target-prompt diner_ref2va.txt \
  --ref-images subject1.png \
  --words "The target video uses a live-action cinematic style with warm natural sunlight. -> The target video uses a brightly lit 1990s multi-camera studio sitcom style with flat three-point stage lighting and warm television broadcast color.; says with an amused expression in a warm, amused voice -> delivers her line with theatrical sitcom timing and a playful grin in a bright, punchy comedic voice; soundscape: Atmospheric diner sounds and faint murmurs of distant patrons. -> Quiet sitcom studio room tone followed by a live studio audience bursting into uproarious laughter and applause." \
  --families ref2va --overwrite
```

#### Example 1B: Psychological Horror Dread (Chiaroscuro Flicker + Hushed Delivery + Eerie Sub-Bass Drone)
```bash
"$PY" bench/control_vectors.py from-prompt \
  --name diner_horror_dread \
  --target-prompt diner_ref2va.txt \
  --ref-images subject1.png \
  --words "The target video uses a live-action cinematic style with warm natural sunlight. -> The target video uses a psychological horror style with sickly green-cyan fluorescent shadow contrast and oppressive darkness.; Inside a sunlit diner -> Inside a shadowy, deserted late-night diner under a flickering fluorescent tube; says with an amused expression in a warm, amused voice -> whispers with an unblinking, unsettling stare in a cold, hushed voice; soundscape: Atmospheric diner sounds and faint murmurs of distant patrons. -> Oppressive dead-air room tone, the erratic electrical buzz of a failing fluorescent ballast, and a low, metallic sub-bass tension drone." \
  --families ref2va --overwrite
```

#### Example 1C: 35mm Indie Arthouse Drama (Anamorphic Shallow DOF + Intimate Delivery)
```bash
"$PY" bench/control_vectors.py from-prompt \
  --name diner_indie_arthouse \
  --target-prompt diner_ref2va.txt \
  --ref-images subject1.png \
  --words "The target video uses a live-action cinematic style with warm natural sunlight. -> The target video uses a melancholic 35mm anamorphic indie arthouse film style with heavy photochemical grain, soft amber halation, and razor-thin depth of field.; The camera holds a medium shot -> The handheld 35mm camera drifts in a close medium shot with gentle breathing motion" \
  --families ref2va --overwrite
```
- **ComfyUI Runtime Settings**: Load `diner_90s_sitcom__ref2va__diff_span__lora_865ee189.safetensors` on `MiniMaxH3ApplyControlVectors` with your **unmodified** `diner_ref2va.txt` prompt, `strength = 1.15`, `mode = "add_norm_matched"`, `mask = "scene_only"`, `start_percent = 0.00, end_percent = 1.00`, `ref_vae_end_percent = 1.00`.

---

## Recipe 2: Reusable Multi-Scene Library Vectors (`from-toml`)

When you want a generalizable vector that works on *any* prompt (`t2va`, `fl2va`, `ref2va`, or `P-H3-Image`), build from the 10-pair `.toml` banks in `prompt_bank/control_vectors/`:

```bash
# Build 1974 16mm Kodak film stock (both fl2va and ref2va)
"$PY" bench/control_vectors.py from-toml prompt_bank/control_vectors/film_16mm_1974.toml --families both --overwrite

# Build scorching desert heatwave (coupled visual + cicada/wind soundscape)
"$PY" bench/control_vectors.py from-toml prompt_bank/control_vectors/scorching_heatwave.toml --families both --overwrite
```
- **Pose-Locked Optical Grade (`start_percent = 0.33, end_percent = 1.00`)**: On an 8-step Pika workflow, setting `start_percent = 0.33` lets steps `0` and `1` lock the exact actor blocking and camera framing of your unsteered seed, and applies `film_16mm_1974` only to surface grain, halation, and color grade across steps `2..7`.

---

## Recipe 3: Instant `< 2 ms` CPU Word Synthesis (`from-words`)

When iterating rapidly without loading the 32B Qwen3-VL encoder on GPU:

```bash
"$PY" bench/control_vectors.py from-words "neon, cyberpunk, rain, magenta, cyan" \
  --against-words "daylight, sunny, natural, clear" \
  --hold-words "umbrella, coat" \
  --name fast_cyberpunk
```
Or type `neon, cyberpunk, rain, magenta, cyan` directly into the `words` field on `MiniMaxH3ApplyControlVectors` (with `vector_file = "none"`).

---

## Recipe 4: Visual Look Transfer from Two Reference Images (`from-images`)

When you have an ungraded still (`flat.png`) and a color-graded or stylized reference still (`graded.png`):

```bash
"$PY" bench/control_vectors.py from-images \
  --name bleach_silver_grade \
  --image-a flat.png \
  --image-b graded.png \
  --pooling scene_text --overwrite
```

---

## Recipe 5: `< 5 ms` CPU Sculpting — Flip (`--invert`), Blend & Orthogonal `hold`

```bash
# 1. Flip blizzard (-1.0x) on CPU in < 5 ms
"$PY" bench/control_vectors.py sculpt blizzard --invert --name anti_blizzard

# 2. Blend scorching_heatwave + film_16mm_1974 and scrub out wardrobe drift
"$PY" bench/control_vectors.py sculpt scorching_heatwave film_16mm_1974 \
  --modes add_norm_matched,add_norm_matched \
  --strengths 1.0,0.6 \
  --hold-words "hat, sunglasses, coat" \
  --name desert_16mm
```
