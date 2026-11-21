# Multi-Vector DiT Steering: A 4-Lab Progressive Guide

Last updated: 2026-11-21  
Companion plain-English overview: [`how_pole_a_erasure_works.md`](how_pole_a_erasure_works.md)  
Full paper & benchmark: [`2026-11-21_from_late_interaction_retrieval_to_av_dit_steering.md`](../research/control_vectors/2026-11-21_from_late_interaction_retrieval_to_av_dit_steering.md) · [`2026-11-21_multivector_erasure_benchmark.md`](../../../bench/results/2026-11-21_multivector_erasure_benchmark.md)

This guide is written for engineers and creators who have worked with LLMs, DiT video/image models, and embedding or multi-vector retrieval (`ColBERT`, `PLAID`/`WARP`, `PyLate`, `Model2Vec`, `ColQwen`) and want to **see, hear, and build** with single-pass (`1x`) multi-vector subspace steering in MiniMax H3.

Every lab below builds directly on the previous one — starting from a zero-GPU `< 15 ms` terminal X-ray and a 1-click split-screen A/B workflow, then adding bipolar material sweeps + `< 2 ms` static word-subspace sketching, multi-slider sight+sound world composition, and finally `<Picture 1>` visual contrast + moving `[F, H, W]` reality portals.

---

## Retrieval Intuition → DiT Conditioning in 60 Seconds

If you have used late-interaction retrieval, you already know why single-vector dense pooling (`mean(tokens)`) hits a wall: averaging a passage into one vector blurs distinct clauses together and lets high-frequency framing tokens drown rare semantic words.

Inside MiniMax H3 (`Qwen3-VL-32B` `5120`-D causal encoder $\to$ 2-block bidirectional `TokenRefiner` `5376`-D $\to$ 60 joint audio-video DiT blocks), **the exact same failure happens when steering a render at `1x` cost — plus a second trap unique to generative attention**:

| Retrieval Concept (2020–Nov 2026) | What It Does in Search | What Happens Inside H3's `5376`-D `TokenRefiner` Space |
| :--- | :--- | :--- |
| **Dense 1D Pooling** (`mean(B) - mean(A)`) | Collapses multi-aspect documents into one vector | Adds one constant arrow $H + s\hat{v}_{\text{1D}}$ (`add_norm_matched`). **Fails on conflicting prompts:** warm daylight/tungsten tokens in $H$ stay intact and veto `noir` (`0/5` stills, `0/3` `192f` AV clips turn B&W). |
| **Appended Token Bags** (`ColBERT` / `MUVERA` virtual keys) | Stores `[K, D]` token vectors for read-only MaxSim | **Breaks `s = 0` slider neutrality:** merely appending $K = 12$ keys shifts H3's cross/self-attention softmax denominator (`MAE = 22.7–54.5` at zero amplitude). Steering must edit `[L, 5376]` *in place*. |
| **Balanced Sinkhorn OT** (doubly-stochastic transport) | Enforces uniform row *and* column marginals | **Collapses identically back to `v_1d` (`max_err = 8.88e-16`)** because uniform column sums force $\frac{1}{n_A}\sum_i((TB)_i - A_i) \equiv \bar{B} - \bar{A}$. |
| **Asymmetric `ColBERT` Soft-MaxSim (`tau = 0.15`) + Ridge Subspace Erasure** (`add_norm_matched_erasure`) | Right-stochastic row matching (`PyLate` / `ColBERT`) without column constraints | **The `1x` unlock:** pairs each Pole-A token to its closest Pole-B partners (`erasure.mu_d`, rescuing `"black"` on `noir`) *and* subtracts your prompt's centered projection onto the rank-8 Pole-A subspace (`-clip(s,0,1) P_{V_r,\lambda}(H - \bar{H})`). |
| **Minish Lab `Model2Vec` + PCA De-Coning** | Distills contextual subwords into fast static whole-word/phrase tables | Powers `< 2 ms` CPU multi-token & 2-word phrase lookup (`origami`, `muppet`, `silver gelatin`, `direct flash`) + `against`-SVD erasure in `MiniMaxH3WordVector`. |
| **`ColQwen` Visual Patch Late Interaction** | Keeps M-RoPE visual tokens unpooled | Powers `MiniMaxH3MakeMediaControlVector` extracting rank-8 `erasure.{mu_d,vr,diag}` directly from `<Picture 1>` `image_a` vs. `image_b` visual pairs. |

### Shell Setup (Run Once per Terminal)

Per `AGENTS.md` Rule 1, always invoke ComfyUI's `.venv` Python directly (never `uv`) and source `h3.env`:

```bash
COMFY="$(cd ../.. && pwd)" && PY="$COMFY/.venv/bin/python"
set -a; . ./h3.env; set +a
```

---

## Lab 1 — Breaking the 1D Prompt Veto (CPU Subspace X-Ray + 1-Click Split A/B)

**Goal:** See *before spending 1 second of GPU time* why legacy 1D `noir` gets vetoed by a warm sitcom kitchen prompt, and prove the fix in one click with a matched-seed side-by-side `141`-frame (`5.88s`, `on_both_clocks`) audio-video render.

### Step 1.1 — X-Ray `noir` Against `t2va_sitcom_kitchen.txt` on CPU (`< 15 ms`)

`bench/control_vectors.py inspect` uses a Jacobian-lens word readout for H3 control vectors, adapted from Anthropic's Jacobian lens (J-lens) ([`j_lens.py`](../../../j_lens.py)), to decode both `erasure.mu_d` (*what the vector adds*) and `erasure.vr[0..3]` (*which Pole-A axes it erases*), alongside per-word erasure norms on your scene prompt:

```bash
CUDA_VISIBLE_DEVICES="" "$PY" bench/control_vectors.py inspect noir \
  --prompt prompt_bank/t2va_sitcom_kitchen.txt
```

Inspect the four diagnostic lines in the output:

```text
  j_lens (PR +)   : noir (+1.36), film (+1.23), vintage (+1.21), original (+1.19), stark (+1.18), decades (+1.13), outdated (+1.12), grain (+1.11)
  erasure payload : rank=8  cos(mu_d,v_1d)=0.9702  diag=[0.87, 0.811, 0.802, 0.744, 0.709, 0.67, 0.635, 0.629]  [rescued vs 1D: black, footage, archival]
  j_lens (mu_d +) : vintage (+1.77), stark (+1.60), noir (+1.56), black (+1.52), film (+1.49), footage (+1.49), grain (+1.42), archival (+1.23)
    vr[1] (d=0.81): warm (+2.02), natural (+1.91), lighting (+1.67), warmth (+1.63), tone (+1.54), tones (+1.50)
  prompt coupling : receptors=[black (+0.35), white (+0.23), golden (+0.20), tungsten (+0.17)]
                    subject=[suburban (+0.06), father (+0.10), golden (+0.20), retriever (+0.04)]
                    erases=[wall (0.41), warm (0.36), brown (0.33), tungsten (0.33)]
```

1. **Why `v_1d` washed out `"black"`:** In 1D mean-pooling (`j_lens (PR +)`), `"black"` dropped to rank 9 (`+1.10`) behind generic era words (`original, decades, outdated`). Asymmetric `tau = 0.15` token matching (`erasure.mu_d`) rescues **`black (+1.52)`**, `footage`, and `archival` into the top 8 (`28 / 38` bank vectors show similar target-word rescues).
2. **Why `erasure.vr` spares the characters:** On `t2va_sitcom_kitchen.txt`, `vr[1]` (`diag = 0.81`) zeroes `wall (0.41)`, `warm (0.36)`, `brown (0.33)`, and `tungsten (0.33)` while leaving `father`, `daughter`, and `retriever` intact. *(You can also view this interactively in Control Vector Studio via `./bench/control_vector/studio.sh start` $\to$ **Library** $\to$ **X-Ray**.)*

### Step 1.2 — Queue the 1-Click Same-Seed Split-Screen Workflow

Load [`workflows/h3_text_to_video_pika_1d_vs_erasure_ab_api.json`](../../../workflows/h3_text_to_video_pika_1d_vs_erasure_ab_api.json) in ComfyUI (`:8188`).

- **How it is wired:** One `MiniMaxH3Conditioning` (`t2va_sitcom_kitchen`, `1152x768 @ 141f`) and one `RandomNoise` (`seed = 730451892`) fan into two parallel 6-step Pika sampler branches:
  - **Branch A (Left):** `MiniMaxH3Steer(vector_name="noir", strength=1.55, mode="add_norm_matched", target="scene_only")`
  - **Branch B (Right):** `MiniMaxH3Steer(vector_name="noir", strength=1.55, mode="add_norm_matched_erasure", target="scene_only")`
  - **`MiniMaxH3DirectorHUD(hud_layout="split_ab_card")`:** Renders a synchronized side-by-side comparison card in one `.mp4`.
- **What to watch and listen for:** Both halves share identical blocking, percolator foley, and dialogue timing, but **Branch A stays warm color** (sunlit yellow kitchen walls and wood cabinets vetoing the 1D arrow), while **Branch B snaps the entire room, costumes, and shadows into crisp 35mm silver-gelatin monochrome**.

---

## Lab 2 — Style Without Lobotomy, Bipolar Sweeps & `< 2 ms` `WordVector` Sketching

**Goal:** Understand *why* subtracting the right rank-8 subspace preserves scene characters where blunt shrinkage deletes them, run a bipolar `-1.0 -> +1.5` sweep, and prototype zero-GPU multi-token concepts in `< 2 ms` with `MiniMaxH3WordVector` + `hold_words`.

### Step 2.1 — Why Specific `V_r` Beats Blunt Prompt Dampening (`paper_origami`)

Run the CPU X-ray on `paper_origami` against the same `t2va_sitcom_kitchen.txt` scene:

```bash
CUDA_VISIBLE_DEVICES="" "$PY" bench/control_vectors.py inspect paper_origami \
  --prompt prompt_bank/t2va_sitcom_kitchen.txt
```

Look at `erases=[shape (0.34), live-action (0.28), kitchen (0.22), wall (0.21)]` and `vr[2..3]` (`realistic, realism, texture, real, natural`). In our controlled specificity benchmark (`ablation2_vr_variants_fl2va.png` and `ablation3_rank_and_ridge.png` at matched `RMS = 101.25`):
- **At `r = 0` (`1D` `add_norm_matched`):** Pushing `paper_origami` at `+1.5` against un-erased `"live-action suburban father and teenage daughter"` tokens creates a representational tug-of-war that **completely deletes the teenage daughter and her chair from the kitchen table**.
- **With `S4_iso_shrink` (blunt uniform prompt shrinkage):** Dampening all prompt tokens equally turns surfaces into paper but **deletes human figures into empty geometric corridors (`0/5` subject preservation)**.
- **At `r = 8` (`add_norm_matched_erasure`):** Subtracting only the `live / realistic / skin / cloth` subspace lets **both** the folded-paper father *and* the folded-paper daughter (complete with folded horn-rimmed glasses, toast, golden retriever, steaming percolator, and `"SUNBEAM"` wall clock) materialize cleanly!
- **Polarity clamp (`strength < 0`):** Try setting `strength = -0.80` on `MiniMaxH3Steer`. Because `control_vectors.py` clamps erasure via `max(0.0, min(s, 1.0))`, negative strengths automatically turn Pole-A erasure **off** (`0.0`) so reverse sweeps (`-1.0 -> 0.0`) push smoothly toward crisp live-action realism without ever amplifying Pole-A noise.

### Step 2.2 — Zero-File `< 2 ms` Sketching with `MiniMaxH3WordVector` (`Model2Vec` + `against`-SVD)

What if you want to test a brand-new idea right now without building a 10-pair `.toml` on the GPU?

Load [`workflows/h3_text_to_video_pika_word_vector_erasure_api.json`](../../../workflows/h3_text_to_video_pika_word_vector_erasure_api.json) (`prompt_bank/t2va_hardware_aisle_short.txt`, `1152x768 @ 141f`).

- **What changed under the hood (`j_lens.py::synthesize_from_words`):**
  1. **Multi-token & 2-word `Model2Vec` lexicon:** Single-token words (`noir`, `frost`, `grain`) resolve from the `24,912`-word Wiener table, while multi-token BPE words and 2-word terms (`origami`, `muppet`, `chiaroscuro`, `plasticine`, `halation`, `silver gelatin`, `paper origami`, `direct flash`) resolve automatically from `$H3_MODEL_STORE/j_lens/model2vec_h3_lexicon.safetensors` (with BPE subword SIF fallback if absent).
  2. **Instant `against`-SVD erasure (`< 2 ms` CPU):** Passing `@replace_digital, sunny, warm, indoor, afternoon` (or any `>= 2` opposing words/presets in `against_words`) projects each individual `against` word into `5376`-D post-refiner space and computes a centered SVD `erasure.{mu_d,vr,diag}` on the fly — feeding straight into `MiniMaxH3Steer(mode="add_norm_matched_erasure")`.
  3. **Per-row `hold_words` protection:** Setting `hold_words = "coat, parka"` (`hold_alpha = 0.5`) projects winter-coat directions out of **every** steered row (`_apply_holds_to_rows64`) so pushing `blizzard, snowstorm, frost, freezing, gale` frosts the hardware aisle without forcing heavy parkas onto the shoppers.
- **Know the honest lexical ceiling (Ablation 6 & the Pole-A Coverage Law):** Uncontextualized word vectors (`cos ≈ 0.35–0.52` with 10-pair GPU vectors) are awesome `< 2 ms` mood/weather/texture sketchpads, **not** replacements for 10-pair contextual `.toml` builds on hard global overrides (`WordVec` `noir` stays tinted at `sat = 0.323` where the 12-pair contextual `noir` bank hits pure B&W `0.0264`). Once a `WordVector` sketch excites you, graduate it to an **8–12 pair `.toml`** (`N >= 8` covers `80.5%–89.2%` of the true contextual Pole-A subspace vs. `40.6%` at `N = 1`; see [`building_pairs_and_vectors.md`](building_pairs_and_vectors.md)).

---

## Lab 3 — Composing Sight + Sound Worlds (`Chained Steer` vs. Rank-16 `Sculpt`)

**Goal:** Combine visual film stock, kinetic energy, and `40 Hz` stereo acoustic space in a single `1x` joint audio-video pass — and know when to **chain nodes** vs. **sculpt a single composite file**.

### Step 3.1 — Run the 3-Vector Sight + Sound Stack (`t2va_cooking_show_meltdown`)

Load [`workflows/h3_text_to_video_pika_stacked_erasure_api.json`](../../../workflows/h3_text_to_video_pika_stacked_erasure_api.json) (`prompt_bank/t2va_cooking_show_meltdown.txt`, `1152x768 @ 192f / 8.00s`).

Three `MiniMaxH3Steer` nodes sit in series before the sampler:
1. **`noir` (`+1.55`, `target="scene_only"`):** Erases bright morning studio floodlights and pastel set color across `[Shot 1]..[Shot 3]` (`sat_mean` drops **`3.0x` from `0.2141` in 1D to `0.0715`**).
2. **`chaos_panic` (`+0.90`, `target="scene_only"`):** Amplifies kinetic slapstick urgency as the beetroot blender lid detonates.
3. **`reverb_cathedral` (`+1.25`, `target="soundscape_only"`):** Aimed strictly at `overall_soundscape:` and `non_diegetic_music:`, erasing dry studio room damping on H3's `40 Hz` audio stream (`ablation8b`: `reverb_cathedral` increases late acoustic tail decay while keeping `"Mon dieu, shut the power off before it covers the monitors!"` intelligible and lip-synced).

### Step 3.2 — The Subspace & Norm Budget Law: Chaining vs. `sculpt`

You can also fuse multiple vectors offline on CPU in `< 50 ms` using `bench/control_vectors.py sculpt`, which preserves up to **rank-16 union erasure** (`COMPOSED_ERASURE_MAX_RANK = 16` in [`control_vectors.py`](../../../control_vectors.py)):

```bash
CUDA_VISIBLE_DEVICES="" "$PY" bench/control_vectors.py sculpt \
  noir:1.0 chaos_panic:0.6 \
  --hold-words "blur,darkness" \
  -o "$H3_MODEL_STORE/control_vectors/noir_slapstick_sculpt.safetensors"
```

**Which one should you choose?**
- **Chain separate `MiniMaxH3Steer` nodes** when stacking **two strong visual overrides** (like `noir +1.55` + `chaos_panic +0.90`) or targeting **different prompt sections** (`scene_only` + `soundscape_only`): each chained node evaluates its full `|s_i| * 67.5` RMS shift against unsteered `base_rows64` before summing in `float64` (`control_vectors.py:3454-3492`), giving `noir` full erasure authority.
- **Use `sculpt` (`COMPOSED_ERASURE_MAX_RANK = 16`)** when you want a **single reusable preset knob** with baked `hold_words` (`calm_cursed_cctv`) where one shared `|s| * 67.5` RMS budget is split across subtle blended traits.

---

## Lab 4 — The Full Creative Stack (`1x` Expander + `ColQwen` Media Contrast + Moving Reality Wipe)

**Goal:** Combine `MiniMaxH3Steer` with the rest of the pack — `MiniMaxH3PromptExpander`, `<Picture 1>` `ColQwen` visual contrast (`MiniMaxH3MakeMediaControlVector`), and 3D `[F, H, W]` spatial wipes (`MiniMaxH3SpatialDirector`) — while respecting multi-modal vision-token boundaries.

### Step 4.1 — `ColQwen`-Style Visual Contrast Slider on `<Picture 1>` (`ref2va_scene_kitchen`)

Load [`workflows/h3_image_ref_make_media_erasure_pika_api.json`](../../../workflows/h3_image_ref_make_media_erasure_pika_api.json) (`1152x768 @ 141f`, `ref2va_scene_kitchen`).

- **How it works:** Instead of writing text pairs, `MiniMaxH3MakeMediaControlVector` encodes `image_a` (Pole A baseline look) and `image_b` (Pole B target grade/lighting) as M-RoPE-aligned `<Picture 1>` conditioning sequences and extracts both `vector.post_refiner` and rank-8 patch/token `erasure.{mu_d,vr,diag}`.
- **Two-gear `<Picture 1>` rule:**
  - Setting `MiniMaxH3Steer(target="description_only")` steers the kitchen environment while leaving `<Subject 1>`'s `subject_definitions:` and `retention_analysis:` untouched (**Selective-Color / Room-Grade Subject Lock**).
  - Remember from the **Pole-A Coverage Law** (`ablation7`): a single image pair (`N = 1`) captures `~40%` of an 8-D Pole-A subspace — great for cooling/warming ambient room lighting from reference stills, while 3–4 diverse `image_a / image_b` slots on `MiniMaxH3MakeMediaControlVector` (or a 10-pair `.toml`) are needed for hard monochrome flips.
  - And remember the **`fl2va` vision-token lock** ([`workflows/h3_first_last_frame_pika_control_vector_api.json`](../../../workflows/h3_first_last_frame_pika_control_vector_api.json)): when both `first_frame` and `last_frame` are bright color stills, their `843 + 843` unsteered `tags == 0` vision tokens lock the clip's color palette even if `MiniMaxH3ReleaseReferences(0.33)` releases motion. Pre-grade keyframe stills in `MiniMax-H3-Image` first if you want a full-frame `fl2va` palette swap.

### Step 4.2 — Zero-Rewrite Moving `vertical_split` Reality Wipe (`t2va_disco_roller_rink`)

Load [`workflows/h3_spatial_portal_pika_erasure_api.json`](../../../workflows/h3_spatial_portal_pika_erasure_api.json) (`prompt_bank/t2va_disco_roller_rink.txt`, `1152x768 @ 141f / 5.88s`).

- **Why this is the signature showcase:** Feed the **exact same** `"DISCO ROLLERLAND"` gold-lamé roller-disco prompt into both inputs of `MiniMaxH3EditGuider` (`same_prompt = True`, zero prompt edits!), wire `MiniMaxH3SpatialDirector`'s sweeping `vertical_split` (`x = 0.68 -> 0.50 -> 0.30`), and set `MiniMaxH3Steer(vector_name="noir", strength=1.60, mode="add_norm_matched_erasure", edit_pass="edited_only")`.
- **Result (`showcase2_disco_wipe_stacked_141f.mp4`):** Legacy `1D` was vetoed (`0.6003` left vs. `0.4286` right — still warm gold disco on both sides). With `add_norm_matched_erasure`, the sweeping cyan curtain slices cleanly across the spinning skater and mirror ball into **glowing gold disco on the left (`sat = 0.6012`) and razor-sharp 35mm silver-gelatin noir on the right (`sat = 0.1018`, `5.91x` contrast)** while keeping belted disco vocals 100% continuous! *(Keep vector-only spatial portals at `>= 30%` frame area via `vertical_split`; tiny `~10%` circles get pulled back toward the surrounding `90%` warm context by H3's global 3D self-attention unless paired with explicit `edited` prose.)*

---

## Quick Decision Checklist: `1x` Expander vs. `1x` Erasure Slider vs. `2x` EditGuider

| What You Want to Do | Reach For | Cost | Why |
| :--- | :--- | :---: | :--- |
| Author a fresh scene, mutually exclusive lighting setup (`look_crimson_darkroom`), wardrobe, or multi-shot dialogue | **`MiniMaxH3PromptExpander`** (or `MiniMaxH3ImagePromptExpander`) | **`1x`** | Writes native `[Shot N]` blocking, lighting physics, and synced `<d>` speech cleanly from scratch. |
| Dial film stock, monochrome/noir, tactile medium (`claymation_stopmotion`, `paper_origami`), weather, or acoustic space (`reverb_cathedral`, `whisper_asmr`) on *any* existing prompt | **`MiniMaxH3Steer`** (`mode="add_norm_matched_erasure"` default, 8–12 pair `bank/` vector) | **`1x`** | `erasure.vr` deletes conflicting daylight/live-action/dry-room tokens in place while `erasure.mu_d` steers cleanly at zero extra DiT passes. |
| Rapidly test a new mood/material/weather idea + `hold_words` in `< 2 ms` without building a `.toml` | **`MiniMaxH3WordVector` $\to$ `MiniMaxH3Steer`** (`words` + `against_words` + `hold_words`) | **`1x`** | Uses `Model2Vec` phrases + `< 2 ms` CPU `against`-SVD erasure; promote winners to an 8–12 pair `.toml` when you need full contextual override power. |
| Swap a specific garment/prop (`black sweater -> rust-orange 1970s blouse`), lock an exact existing take (`lock_take`), or carve a moving `[F, H, W]` spatial portal | **`MiniMaxH3EditGuider`** (+ `MiniMaxH3PromptEdit` / `MiniMaxH3AutoEditPrompt` / `MiniMaxH3SpatialDirector`) | **`2x`** *(~`1.5x` w/ `last_step`)* | Subtracts predicted latent velocities *after* all 60 DiT blocks, preserving exact clause-to-garment and `[F, H, W]` spatial binding. |
