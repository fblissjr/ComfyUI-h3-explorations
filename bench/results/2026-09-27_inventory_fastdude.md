# Inventory, this session's half: the 2026-09-26 renders

Written 2026-09-27 for the owner, alongside the VAE session's
`2026-09-27_inventory.md`, which carries the per-file table
(`bench/render_inventory.py`) and includes this file by pointer. Basenames
only. The mp4 encoder issue is set aside, as the owner asked.

**Resolve a row's configuration by its `graph_sha256`, never by today's
graph file.** Several graphs have changed or been deleted since they
rendered:
- the INT8 video VAE became the default at 0.151.0, 16:29. So
  `distill_compare_s1` is mixed: its PDD8, FlashGen and FastH3 rows (16:00 to
  16:12) decoded through fp16, and its two base rows (16:48 and 16:57, a
  later invocation) through INT8. The VAE session's `render_inventory.py`
  resolves this by hash;
- `h3_probe_t2v_flashgen_r64_4step_branch` was removed in 0.147.1;
- the Turbo graphs were removed in 0.156.0.

## Model files these records use

| file | what it is | made by |
|---|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot` | the base: the shipped pruned int8 ConvRot FL2VA DiT | `h3_config.MODELS["unet_fl2va"]`; provenance in `docs/attention/quant_policy.md` |
| `minimax_h3_ref2va_pruned_int8_convrot` | the Ref2VA partition, the same form | `MODELS["unet_ref2va"]` |
| `fastvideo_fasth3_8step_v2_pruned_int8_convrot` | FastVideo's FastH3 V2: a full DMD2-distilled T2VA DiT with its own curve basis and 50 VSA gates | HF `FastVideo/FastVideo-FastH3-Comfy`, as downloaded; `MODELS["unet_fasth3_v2"]` |
| `h3/minimax_h3_fl2va_pdd_8step_comfy` | alibaba-pai's PDD 8-step (FL2VA): backbone and refiner LoRA, adaln baked into fl2va's curve basis, and the 32-head bank | `bench/convert_pdd_lora.py`, converter v3 at `eb791f8` (stamped in the file's metadata), from `MiniMax-H3-FL2VA-Acc-8Step` |
| `h3/minimax_h3_flashgen_4step_v1.0_768p_fl2va_pruned_rank64_comfy` | Beidouqixing's FlashGen v1.0, at its full rank 64: qkv rows permuted to ComfyUI's bands, adaln re-expressed on fl2va's curve basis | `bench/convert_flashgen_lora.py` at `c7ad7f06` (0.142.1); record `2026-09-25_flashgen_lora_conversion.json`; exact on every non-adaln module (`2026-09-26_flashgen_weights.json`) |
| `h3/minimax_h3_flashgen_4step_v1.0_768p_ref2va_pruned_rank64_comfy` | the same LoRA converted for the Ref2VA basis | the same script, `--partition Ref2VA`, at `e7d08d19` (0.147.0); record `2026-09-26_flashgen_lora_conversion_ref2va.json` |
| `h3/minimax_h3_4step_lora_flashgen_v1.0_768p_fl2va_pruned_avg_rank_13_bf16` | kijai's lossy resize of FlashGen (sv_fro 0.95, average rank 13) | HF `Kijai/MiniMax-H3-experimental`, not ours |
| `h3/lightx2v_Minimax-h3-Turbo/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16` | LightX2V Turbo 8-step, used once as a smoke test | LightX2V's release. The lane was retired 2026-09-26 (0.156.0) |
| `minimax_h3_video_vae_fp16`, `_int8_convrot`, `minimax_h3_audio_vae_fp32` | the decoders | the INT8 switch is the VAE session's (0.151.0, `aadfb77a`) |

*Deleted 2026-09-27 by the owner:*
- the VAE session's five FastH3 conditioning research files, which none of
  these records rendered except the swap;
- the per-step x0 snapshots of the PDD strength runs on subway, both the
  VAE session's and this session's after-run. Their final latents stay.

Hashes, rebuild commands and what stayed are in
`2026-09-27_research_files_deleted.md`.

## The records, and what each render is

Clips are `Video/<graph stem>_<label>_00001-audio.mp4` and latents
`latents/<graph stem>_{video,audio}_<label>_00001_.latent`, where a graph
has a `_savelat` or `_x0` twin. Seed 730451892 throughout, unless a label
carries `__s<seed>`.

- **`flashgen_lora_path_s1`** (11:07-13:03): how FlashGen's LoRA reaches the
  int8 weight.
  - kijai's r13 merged against our r64 merged against r64 through the exact
    branch, then the faster branch, and one dense arm.
  - Finding: the stock merge loses FlashGen's sub-int8-step delta
    (`2026-09-26_int8_lora_requant.json`). The branch ships.
- **`flashgen_tasks_s1`** (13:05-13:09): FlashGen on i2v and on ref2va
  through the ref2va conversion. First renders: FlashGen was trained on
  neither task.
- **`fasth3_contract_s1`** (14:17-14:29): FastH3 on ComfyUI's template
  against FastVideo's own contract, and each half of it. The contract ships
  as the FastH3 probe.
- **`distill_compare_s1`** (16:00-16:57): the base, PDD8, FlashGen and FastH3
  on diner and the old subway prompt. PDD was merged, pre-0.154.0, when that
  was the only path. The VAE is mixed (above).
- **`subway_v2_s1`** (17:18-17:29): the same four on the rewritten subway
  prompt. The owner's reads are in the record.
- **`distill_run`** (20:08-20:50, stopped): the frozen-row probe and nine
  base arms, on bank prompts that were later made more specific (0.154.6),
  so those base arms carry the earlier text.
- **`followup`** (21:01-01:24). Some rows are contaminated.
  - **Contaminated, first launch:** the PDD and FlashGen rows were stacked
    by the 0.154.8 bug. Their files are in `Video/_contaminated_2026-09-26/`
    and `latents/_contaminated_2026-09-26/`.
    `2026-09-26_followup_contamination.md` lists them. Use the later row per
    label.
    - *Deleted 2026-09-27 by the owner:* both `_contaminated_2026-09-26/`
      folders. The contamination record keeps what each row was
      (`2026-09-27_research_files_deleted.md`, last paragraph).
  - **The looks:** anchor, noir, neon and anime on PDD8, FlashGen and FastH3.
  - **Core:** 11 bank scenes on the three distills, plus FlashGen on
    `subway_chase_short`.
  - **Ladder:** PDD8 merged, PDD4 and PDD6, and FastH3 with VSA off.
    `subway_chase__pdd8` is the x0 twin, with 8 per-step latents.
  - **FlashGen extras:** dense, strength 0.8 and 1.2, and no adaln.
  - **radio_drama_v2:** the silent-listeners, deadpan-comedy text (0.154.7).
  - **The reverse switch:** `rev_h063` and `rev_h080`, PDD first and
    FlashGen finishing, graphs by the VAE session.
  - **The specificity ladder:** `spec_typical`, `_specific` and `_unusual`,
    on the three distills.
- **`flashgen_transplant`** (01:28-01:59): FlashGen with `blocks` set to
  "0-49", "0-33" and "34-49", on look_anchor and slapstick. The timings are
  skewed (interleaved with the swap); the renders are valid.

## Good at, bad at

This rests on one seed. The owner's reads are quoted where they exist, and
the rest comes from measures: `2026-09-26_distill_signatures.md`,
`2026-09-26_followup_looks.md`, and the predictions file's verdicts.

- **PDD8 (exact branch, strength 1.0):**
  - **Good:** close-ups and medium shots, detail, and colour when asked for:
    the most chroma on neon. It keeps black-and-white clean and blacks deep.
  - **Bad:** low contrast and dim highlights ("naturally so", the owner),
    and the slowest distill.
  - **Fixable:** the reverse switch lifts the highlights without changing
    the scene (`2026-09-27_reverse_switch.md`).
  - **Faster option:** PDD6 is within 5-10% of PDD8's detail at three
    quarters of the steps. PDD4 loses more.
- **FlashGen (rank 64, exact branch):**
  - **Good:** the fastest, and motion stays coherent. It leaks the least
    colour into black-and-white.
  - **Bad:** hazy, lifted blacks on every look, and the least saturated. It
    lost who-does-what on the subway.
  - **Its structure** (FT1): the late blocks alone (`blocks="34-49"`) make a
    finished render with deeper blacks and half the haze. The early blocks
    carry the haze.
- **FastH3 (contract):**
  - **Good:** the most detail and colour, and the owner rated its audio best.
  - **Bad:** "super high detail like almost way too much causing it to look
    a bit ai generated in polish" (the owner). It pulls warm and orange, and
    leaks the most colour into black-and-white.
  - **Where the look lives:** in its trained attention path and backbone, not
    in its time conditioning (the VAE session's swap). VSA off halves the
    detail and adds haze.

## Rankings (provisional: one seed, and mostly measured, not judged)

| use | ranking | basis |
|---|---|---|
| close-ups, dialogue, low motion | PDD8 > FastH3 > FlashGen | owner reads, detail and grade measures |
| black-and-white, rich blacks | PDD8 > FlashGen > FastH3 | chroma leak (FastH3 worst), shadow depth (FlashGen shallowest) |
| detail | FastH3 > PDD8 ≈ FlashGen | detail measure on every look; FastH3's is "too much" by eye |
| natural grade | PDD8 > FastH3 > FlashGen | owner's "naturally so"; FlashGen's haze, FastH3's orange pull |
| speed | FlashGen > FastH3 ≈ PDD6 > PDD8 | sampler times in the rows |
| FlashGen configurations | r64 exact branch = late-only candidate > r13 > r64 merged | int8 requant loss; FT1. Late-only needs the owner's eye and an adherence test |
| PDD configurations | exact 8 ≈ reverse switch (brighter) > PDD6 > PDD4 > merged | the ladder and reverse reads; merged is a different take through rounding noise |

What would move these: the owner's eye on the follow-up clips (the list in
`2026-09-27_followup_summary.md`), and base renders on the scenes worth it.
