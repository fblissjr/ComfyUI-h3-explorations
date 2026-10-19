# What made what: models, adapters and the 2026-09-26 renders

Written 2026-09-27 by the VAE session (vaedude) with fastdude, for the
owner's question: which model file is what, which is a hybrid of what, which
code made it, what it is good and bad at, and which files in `Video/` and
`latents/` came from it. It covers every render we queued on 2026-09-26/27
(196 rows across 19 run records) and everything in `latents/`.

fastdude's half, `2026-09-27_inventory_fastdude.md` (5cc342d8), has each
record's time window, the contamination notes, and good/bad-at per distill;
its rankings table is quoted below. The per-render detail is
`2026-09-27_render_inventory.json`, from
`bench/render_inventory.py` (0e02db17). It resolves each row's graph as
rendered, applies its patches, and lists every output file. File names below
are basenames, relative to the output share. **Everything here is one seed and
at most a few scenes per arm. Rankings are our reading, not a measurement.**
The mp4 encoder's dark blocking (`2026-09-27_o1_lossless.md`) is left out.

## How the output files are named

- **Clips**, in `Video/`: `<graph prefix>_<label>_000NN.mp4` (silent),
  `..._000NN-audio.mp4` (with sound) and `..._000NN.png` (first frame).
- **Final latents**, in `latents/`: `<graph prefix>_video_<label>_000NN_.latent`
  and `..._audio_...`, written by the `_savelat` graphs. Any of them can be
  decoded again with `h3_decode_saved_latent_api.json`, with no sampling.
- **Per-step predictions**, in `latents/`:
  `<graph prefix>_x0_<label>_<stamp>_<step>_video.latent`, one per sampling
  step, from the `_x0` graphs (`step_x0_observer.py`).
- **Labels** are `<scene>__<arm>`, for example `look_anchor__pdd8_s085`: the
  look_anchor scene, PDD8 at strength 0.85. Unlabelled names (`ship`,
  `r64_branch`, `scout`) are the earlier single-purpose runs listed under each
  model below.

## The model files, and what each one is

| file | what it is | made by | used for |
|---|---|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot` | **The base.** MiniMax H3 FL2VA (text/first-last-frame to video+audio), pruned curve form, int8 | Comfy-Org release | every base, PDD and FlashGen render |
| `minimax_h3_ref2va_pruned_int8_convrot` | The base's reference-to-video partition | Comfy-Org release | ref2va renders |
| `fastvideo_fasth3_8step_v2_pruned_int8_convrot` | **FastH3 V2**: FastVideo's full fine-tune of fl2va (data-free DMD2, trained with VSA sparse attention), plus 50 VSA gates | FastVideo release (FastVideo-FastH3-Comfy) | FastH3 renders |
| `minimax_h3_fl2va_pruned_int8_convrot_fasth3adaln` | **Hybrid (ours):** the base's weights with FastH3's timestep conditioning (time table + all adaln projections), no gates | `bench/build_adaln_swap.py` @ 91db4fc2 | the swap test |
| `fastvideo_fasth3_8step_v2_pruned_int8_convrot_baseadaln` | **Hybrid (ours):** FastH3's weights and gates with the base's timestep conditioning | `bench/build_adaln_swap.py` @ 91db4fc2 | the swap test |
| `fastvideo_fasth3_8step_v2_pruned_int8_convrot_temb_a05`, `_a075` | **Hybrid (ours), never rendered:** FastH3 with its conditioning blended 50% / 75% toward FastH3 from the base | `bench/build_adaln_blend.py` @ 135b449e | nothing (see "Safe to delete") |
| `loras/h3/minimax_h3_fl2va_pdd_8step_comfy` | **PDD** (alibaba-pai's parallel decoding distill, 8 steps on a 32-point grid) converted for ComfyUI: backbone LoRA, adaln baked into fl2va's curve basis, 32-head output bank | `bench/convert_pdd_lora.py` @ eb791f8, from `MiniMax-H3-FL2VA-Acc-8Step` | every PDD render |
| `loras/h3/minimax_h3_fl2va_fasth3temb_pdd_8step_comfy` | **PDD re-baked onto FastH3's conditioning (ours), never rendered.** Loads only on `..._fasth3adaln` | `bench/convert_pdd_lora.py` via `bench/build_pdd_base_shim.py` @ cb21249c | nothing (see "Safe to delete") |
| `loras/h3/minimax_h3_flashgen_4step_v1.0_768p_fl2va_pruned_rank64_comfy` | **FlashGen** (Beidouqixing's data-free 4-step VSD distill) at its full rank 64, converted exactly | `bench/convert_flashgen_lora.py` @ c7ad7f06 (0.142.1) | every FlashGen render |
| `loras/h3/minimax_h3_flashgen_4step_v1.0_768p_ref2va_pruned_rank64_comfy` | The same, re-fitted to ref2va's time basis. An untested transfer: FlashGen was trained on FL2VA | `convert_flashgen_lora.py --partition Ref2VA` @ e7d08d19 | ref2va FlashGen |
| `loras/h3/minimax_h3_4step_lora_flashgen_v1.0_768p_fl2va_pruned_avg_rank_13_bf16` | kijai's FlashGen conversion: resized to about rank 13 (keeps about 95% of each delta) | kijai (HF `Kijai/MiniMax-H3-experimental`) | one comparison render (`r13`) |
| `minimax_h3_video_vae_int8_convrot` | The video VAE, int8. **The shipped default since 2026-09-26** | Comfy-Org release | most renders |
| `minimax_h3_video_vae_fp16` | The video VAE, fp16 (the official fp32 cast down) | official release | the int8-vs-fp16 comparisons |
| `taeh3` | A tiny approximate decoder, for draft previews only | madebyollin | the scout drafts |

The lightx2v turbo LoRA appears in one smoke row (`smoke_turbo_branch`); that
lane is retired (0.156.0).

## What each configuration made, and what we think of it

### Base (fl2va, 16 steps): the reference
- **Rows**: `distill_run` (`<scene>__base`, 9 scenes); `distill_compare_s1`
  (`diner_base`, `subway_base`); `subway_v2_s1` (`base`); `ref_pathway_1344`
  (ref2va, 11 rows). The base's prompts predate the 0.154.6 bank edits, so they
  do not pair with the distills' rows.
- **Read.** The quality reference: the most natural motion, and slow (16
  steps).

### Base at 8 steps on FastH3's settings (`fl2va_contract`)
- **Rows**: `fasth3_swap`, three scenes. `clone_base_x0` (`subway_chase__base_euler32`,
  32 Euler steps on PDD's grid, with 32 per-step x0 latents) is the related
  clone control.
- **Read.** Under-converged, soft, almost static: the floor, not something to
  use. The 32-step control drew the same two people at 1 s as PDD8, which is
  how the clone was traced to the seed and prompt.

### PDD8 (fl2va + the PDD sidecar)
| arm | label suffix | read |
|---|---|---|
| **PDD8, exact branch** (the default) | `__pdd8` (13 scenes, looks, spec) | Natural, flatter and dimmer than the others (the owner: "the latter usually - but usually naturally so"), and the least motion. The flat look is a defect in some scenes, worst in lamp-lit interiors (`2026-09-26_distill_signatures.md`). Quietest audio. |
| PDD8, merged into int8 | `__pdd8_merge` | A different take with the same character and the same audio (`2026-09-27_ladder.md`). No reason to prefer it. |
| **PDD6** (tail-weighted six steps) | `__pdd6` | Within 5-10% of PDD8's detail at three quarters of the steps. The faster-PDD candidate. |
| PDD4 | `__pdd4` | 22-32% less fine detail, the least motion, more boil. Not recommended. |
| PDD8 at strength 0.85 / 0.7 | `__pdd8_s085`, `__pdd8_s07`, `__pdd8_s07all` | Worse: less detail and motion, and the flat grade is unchanged (`2026-09-27_pdd_strength.md`). Keep 1.0. |
| **PDD8, then FlashGen finishing** (reverse switch) | `subway_chase__rev_h080`, `__rev_h063` | Keeps PDD8's take and lifts its dim highlights to the others' level. The best fix for PDD8's look so far, on one scene (`2026-09-27_reverse_switch.md`). |

### FlashGen (fl2va + the rank-64 LoRA through the exact branch)
From fastdude's reads and measures, reviewed by fastdude:

| arm | label suffix | read |
|---|---|---|
| FlashGen, full | `__flashgen`, `r64_branch`, `ship`, `scout` | Fast (4 steps). Hazy, lifted blacks, coolest colour, brightest audio. Followed role and direction beats less well than the base on the subway, by the owner's eye at one seed (the checklists are unscored). |
| **FlashGen, blocks 34-49 only** | `__flashgen_blk34_49` | A finished 4-step render with deeper blacks and about half the haze. **The candidate best FlashGen setting**, pending the owner's eye (fastdude's FT1). |
| FlashGen, blocks 0-33 only | `__flashgen_blk0_33` | Broken: near-black. |
| FlashGen, blocks 0-49 | `__flashgen_blk0_49` | The same as full; the refiner and final layer do not matter. |
| FlashGen at 0.8 / 1.2 | `__flashgen_s08`, `__flashgen_s12` | More FlashGen means *less* haze: on courtroom, 1.2 gives more contrast (.280 against .258), more saturation and less haze (.098 against .116) than full, and 0.8 goes the other way; kpop barely moves (`2026-09-27_flashgen_variants_tone.json`). |
| FlashGen without adaln | `__flashgen_noadaln` | Matches full on tone within noise on both scenes (`2026-09-27_flashgen_variants_tone.json`): FlashGen barely touches the timestep conditioning. |
| FlashGen, dense attention | `__flashgen_dense` | Matches the Sol arm on tone (haze, contrast, saturation, detail) on kpop and samurai (`2026-09-27_flashgen_variants_tone.json`). Its effect on adherence is unscored, for the owner's checklist read. |
| FlashGen merged into int8 (the old path) | `r64`, `subway_r64`, `fast_r64` | Loses most of FlashGen's delta, which is why the exact branch became the default. |
| kijai's rank-13 resize | `r13` | Lossy by construction. The rank-64 conversion is exact. |
| FlashGen on i2v / on ref2va | `i2v_flashgen`, `r2v_flashgen`, `*_fp16`, `*_int8` | It holds up by eye off T2VA (`2026-09-26_flashgen_tasks_s1.md`). Ref2va is an untested transfer. |

### FastH3 V2
| arm | label suffix | read |
|---|---|---|
| **FastH3, FastVideo's contract** (10/3, its 8 trained steps, VSA keep 20% from step 0) | `__fasth3`, `contract` | The most fine detail and chroma of every distill on 13 of 13 scenes, warm and saturated. The owner: "super high detail like almost way too much ... ai generated in polish". |
| FastH3, ComfyUI template settings | `template`, `contract_attn`, `contract_sampling` | Off FastH3's training contract; use the contract. |
| FastH3, VSA off | `__fasth3_novsa` | Half the fine detail, with the grade shifting. It removes the sparsity and the learned gates together (fastdude, 1eacb74f). |
| FastH3 weights + base conditioning | `__swap_baseadaln` | Near frame-identical to FastH3 (`2026-09-27_fasth3_swap.md`). |
| Base weights + FastH3 conditioning | `__swap_fasth3adaln` | Near frame-identical to the base at 8 steps: under-converged. |

The swap showed that FastH3's speed and its look live in its weights and VSA
gates, not in its timestep conditioning.

### Reference pathway (#26), ref2va at 1344x768
- **Rows**: `ref_pathway_1344_arms`. Graphs are `h3_probe_ref_pathway_<arm>`,
  labels `typed_both`, `typed_encoder`, `native_both`, `native_encoder` and
  `fl2va_encoder`, at two seeds (20260930-31) plus a warmup.
- **What it asks**: how reference images reach the model. Typed or native
  reference roles, through both paths or the encoder only, and fl2va with
  encoder-only references.
- **Status**: blinded as `ref_pathway_1344_2026-09-26` with four controlled
  pairs, key sealed, **not scored yet**. No ranking until the owner scores it.

### VAE and draft runs (not model comparisons)
- **`int8_vae_e2e_s1`, `int8_vae_tasks_s1`** (`ship_int8`, `r2v_fp16`,
  `r2v_int8`, `i2v_fp16`, `i2v_int8`): the int8 VAE against fp16, on FlashGen
  renders. The owner could not tell them apart, int8 is about 12 s faster per
  render, and it became the default.
- **`draft_decode_s1`** (`ship`, `draft`, `keep`, `int8`) and the
  **`scout_*`** runs (`scout`, `full`, `warmup_full`): taeh3 draft previews
  and their full decodes. The owner declined drafts (not worth it; taeh3
  ghosts).
- **`telemetry_overhead_s1`** (`ship`): the first pipeline telemetry records.

## Rankings, 2026-09-27 (provisional: one seed, mostly measured, not judged)

fastdude's table, by use (from `2026-09-27_inventory_fastdude.md`):

| use | ranking | basis |
|---|---|---|
| close-ups, dialogue, low motion | PDD8 > FastH3 > FlashGen | owner reads, detail and grade measures |
| black-and-white, rich blacks | PDD8 > FlashGen > FastH3 | chroma leak (FastH3 worst), shadow depth (FlashGen shallowest) |
| detail | FastH3 > PDD8 ≈ FlashGen | detail measure on every look; FastH3's is "too much" by eye |
| natural grade | PDD8 > FastH3 > FlashGen | owner's "naturally so"; FlashGen's haze, FastH3's orange pull |
| speed | FlashGen > FastH3 ≈ PDD6 > PDD8 | sampler times in the rows |
| FlashGen configurations | r64 exact branch = late-only candidate > r13 > r64 merged | int8 requant loss; FT1 |
| PDD configurations | exact 8 ≈ reverse switch (brighter) > PDD6 > PDD4 > merged | the ladder and reverse reads |

Our joint picks, to try first:
- **Default look, any scene:** PDD8, exact branch.
- **PDD8 where its flat, dim grade bothers you:** PDD8 then a FlashGen finish
  at 0.8 (the reverse switch). Tested on one scene.
- **Faster PDD:** PDD6.
- **Fastest usable:** FlashGen with `blocks="34-49"`, pending your eye; full
  FlashGen if the haze does not matter for the scene.
- **Most detail:** FastH3 on its contract, knowing it over-polishes.
- **The quality reference:** base fl2va at 16 steps.

Not recommended:
- PDD4;
- PDD at strength below 1;
- FlashGen blocks 0-33 (broken);
- the merged FlashGen path;
- FastH3 on template settings;
- both swap hybrids (research only);
- the base at 8 steps.

## Safe to delete (the owner's call)

**Deleted 2026-09-27** by the owner, after fastdude and the VAE session both
cleared them: the five research files below, the unpruned V2, and
`internal/pdd_shims/control_fl2va_pdd_8step_comfy`. Hashes and rebuild
commands are in `2026-09-27_research_files_deleted.md` (9d222c19). **Kept**:
`internal/pdd_shims/shim_fasth3_temb` and `shim_fl2va_temb`, and the bf16
pruned FastH3 and fl2va until `docs/state/open_experiments.md` #35 lands; that
record says why. The list below is as written before the deletion.

Built on a reading the swap renders overturned, and not worth keeping for
use. About 84 GB together (`du -h` on 2026-09-27).
- `fastvideo_fasth3_8step_v2_pruned_int8_convrot_temb_a05` and `_a075`, 21 GB
  each: never rendered.
- `minimax_h3_fl2va_pruned_int8_convrot_fasth3adaln` (20 GB) and
  `fastvideo_fasth3_8step_v2_pruned_int8_convrot_baseadaln` (21 GB): their
  question is answered.
- `loras/h3/minimax_h3_fl2va_fasth3temb_pdd_8step_comfy` (1.1 GB): never
  rendered, and it loads only on a hybrid above.
- The unpruned FastH3 V2 download (66 GB, under the owner's Storage, not in
  ComfyUI's models folder) served the #4 read and nothing else.
- `internal/pdd_shims/` is local and gitignored.
Each can be rebuilt from its script and commit.

## Renamed on disk, 2026-09-27 (the owner asked for clearer names)

*2026-09-27, later: the five files in the "now" column were deleted, and
the two research folders with them ("Safe to delete" above).*

The five research files built last night moved into research subfolders,
each with a README. Run records, manifests and each file's own metadata keep
the old names; this table maps them. Nothing shipped referenced any of them.

| old name | now |
|---|---|
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot_fasth3adaln` | `diffusion_models/h3_research/hybrid__fl2va-weights__fasth3-timecond__int8` |
| `diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot_baseadaln` | `diffusion_models/h3_research/hybrid__fasth3v2-weights-and-gates__fl2va-timecond__int8` |
| `diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot_temb_a05` | `diffusion_models/h3_research/hybrid__fasth3v2__timecond-blend-50pct-fasth3__int8` |
| `diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot_temb_a075` | `diffusion_models/h3_research/hybrid__fasth3v2__timecond-blend-75pct-fasth3__int8` |
| `loras/h3/minimax_h3_fl2va_fasth3temb_pdd_8step_comfy` | `loras/h3/research/pdd8-sidecar__for-hybrid__fl2va-weights__fasth3-timecond` |

Left in place:
- **The FlashGen rank-64 LoRAs and FastH3 V2.** Shipped graphs and
  `h3_config` reference them, and their names already say what they are.
- **Not in the models folders, on purpose:** the bf16 FlashGen and FastH3
  inputs and the unpruned V2 (the owner's download folder, outside ComfyUI), and the PDD shims (`internal/`).

## What's in `latents/`

- **Final latents, video and audio**, one pair per `_savelat` render: every
  `__pdd8`, `__pdd8_*`, `__pdd6`, `__pdd4`, `__flashgen*`, `__fasth3*`,
  `__swap_*`, `__fl2va_contract`, `scout` and `warmup_*` label above. They
  are for re-decoding without sampling, and for the latent reads.
- **Per-step x0 latents**:
  - 8 each for `subway_chase__pdd8`, `__pdd8_s085` and `__pdd8_s07`;
  - 32 for `subway_chase__base_euler32`.
  They are the clone read's inputs (`2026-09-27_clone_base_control.md`), and
  the biggest group by count.
- **`_contaminated_2026-09-26/`**: fastdude's quarantine of the first launch's
  PDD and FlashGen latents, rendered under the branch-stacking bug fixed in
  0.154.8 (`2026-09-26_followup_contamination.md`). Not usable.
- **`h3_block_propagation/`, `h3_dense_block_cost/`, `h3_exact_verify/`**:
  older probe outputs, not from yesterday.

`bench/render_inventory.py --unclaimed` lists anything no run row names.
