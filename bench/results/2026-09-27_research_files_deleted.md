# Research model files deleted, with their hashes and rebuild commands (2026-09-27)

The owner approved deleting the FastH3 conditioning research files after their
questions were answered (`2026-09-27_fasth3_swap.md`, S2 to S4). Each was built
on the CPU from files that stay on disk, so each can be rebuilt. The hashes
below were taken before deletion (`sha256sum`, 2026-09-27); a rebuild that
matches is the same file.

| file (as renamed 2026-09-27, `0361f5f7`) | sha256 | built by |
|---|---|---|
| `diffusion_models/h3_research/hybrid__fl2va-weights__fasth3-timecond__int8` | `230efd709557d18eb99ac2ef421c5977d0761b7717039f516d7c50eef4605c01` | `bench/build_adaln_swap.py` @ `91db4fc2` |
| `diffusion_models/h3_research/hybrid__fasth3v2-weights-and-gates__fl2va-timecond__int8` | `75257313cc8e233fd4963d8693e622e70adb3ab670383e6f4f5b0c314d10b3d0` | `bench/build_adaln_swap.py` @ `91db4fc2` |
| `diffusion_models/h3_research/hybrid__fasth3v2__timecond-blend-50pct-fasth3__int8` | `23e9612474b35f7cfa9e5b89c4c5006ddab3307ea937bf5a9a2a57c4a86d1ce8` | `bench/build_adaln_blend.py` @ `135b449e` |
| `diffusion_models/h3_research/hybrid__fasth3v2__timecond-blend-75pct-fasth3__int8` | `f015cd6e5391299cadbcfac7aa3d9be95e4cc69b43418631455bb8bea01aadf8` | `bench/build_adaln_blend.py` @ `135b449e` |
| `loras/h3/research/pdd8-sidecar__for-hybrid__fl2va-weights__fasth3-timecond` | `f25461a4272d5658cc86a9cf6e67c68ccd4a5dda91572d69804312e66c5d5d1a` | `bench/convert_pdd_lora.py` via `bench/build_pdd_base_shim.py` @ `cb21249c` |

The files record their old names in run rows and manifests; the mapping is in
`2026-09-27_inventory.md`, "Renamed on disk".

## Rebuilding

Sources, all kept: `h3_config.MODELS["unet_fl2va"]` (`F`) and
`h3_config.MODELS["unet_fasth3_v2"]` (`H`), both int8.

- **The swap pair.** `build_adaln_swap.py` streams `--backbone` byte for byte
  and takes the conditioning set (`adaln_t_table` and every
  `adaln_proj.linear`) whole from `--donor`, with no requantisation:
  - fl2va weights, FastH3 conditioning: `--backbone F --donor H`;
  - FastH3 weights and gates, fl2va conditioning: `--backbone H --donor F`.
- **The blends.** `build_adaln_blend.py --backbone H --cond-a F --cond-b H
  --alpha 0.5` or `0.75`. The argument order is from the script's usage line;
  confirm it against the hash.
- **The PDD sidecar.** `build_pdd_base_shim.py` writes a `--base` shim from a
  diffusers-form time embedder. The FastH3 one is kept,
  `internal/pdd_shims/shim_fasth3_temb.safetensors`, as is the fl2va control
  shim. Then `convert_pdd_lora.py --pruned` runs against the fl2va-weights,
  FastH3-conditioning swap file (`cb21249c`'s message).

## What stayed, and why

- **`internal/pdd_shims/shim_fasth3_temb` and `shim_fl2va_temb`.** The first
  is the only small copy of FastH3's time embedder; otherwise it exists only in
  the unpruned V2, which was deleted. Baking a base-trained adapter (PDD,
  FlashGen) into FastH3's own basis needs it. The shim's int8 probe was taken
  from the swap file, so a bake for FastH3 itself needs a new shim with FastH3's
  probe; the embedder tensors carry over.
- **The bf16 pruned FastH3 and fl2va** (the owner's download folder), until
  `docs/state/open_experiments.md` #35 lands. The rank record
  (`2026-09-26_fasth3_lora_rank.json`) compares the two int8 files, whose
  rounding noise is on the order of FastH3's backbone drift (`edf70ef6`), so
  its low energy is most likely a noise floor, not the drift's rank. The bf16
  pair is what measures the drift's true rank, and it is the source for a
  merge dial on it.

Deleted at the same time, not rebuildable and not needed: the unpruned V2
(re-downloadable, `FastVideo/FastVideo-FastH3-8-Step-V2`),
`internal/pdd_shims/control_fl2va_pdd_8step_comfy` (a byte copy of the shipped
PDD sidecar), the stacking-bug renders (`2026-09-26_followup_contamination.md`)
and the per-step x0 snapshots of the PDD strength runs on subway (their final
latents stay).
