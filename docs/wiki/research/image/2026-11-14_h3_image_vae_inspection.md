# The MiniMax-H3-Image VAE: is anything wrong with it, or with how ComfyUI runs it? (2026-11-14)

- **Date:** 2026-11-14
- **Question (the owner's):** the release's image VAE decodes badly enough that the graphs moved to P-H3's VAE
  (`docs/wiki/image/h3_image.md` item 4). Is anything wrong in its weights, its config, Comfy-Org's converted file,
  or ComfyUI's implementation?
- **Answer:** nothing found wrong in any of the four. ComfyUI runs the release decoder the way the release's own
  servers do. What looks bad is the decoder itself, a small ViT that decodes one 256 px tile at a time; P-H3's larger
  decoder, on the same latent, is closer to the picture. Follow-ups the same day (section 8) tried the H3 video
  decoder, its single-frame fine-tune `t1`, averaging, and real renders: nothing beats what the graphs ship, and no
  decode node is worth making.
- **Numbers:** [`bench/results/2026-11-14_h3_image_vae_tiles.md`](../../../../bench/results/2026-11-14_h3_image_vae_tiles.md).
  This page carries directions and pointers only.

---

## 1. The weights and the files

Read by `data/h3_image_vae_2026-11-14/vae_stats.py` (output beside it, untracked), on the release's `vae/` folder,
Comfy-Org's `minimax_h3_image_vae_fp16.safetensors` and P-H3's VAE:

- No NaN or inf anywhere. The only all-zero tensor is `decoder.mask_token`, which inference never reads (core keeps
  it so the file loads cleanly: `comfy/ldm/minimax/vae.py::ViT3DDecoder`; diffusers drops it:
  `coderef/diffusers/scripts/convert_minimax_h3_to_diffusers.py::MINIMAX_H3_VIDEO_VAE_DROPPED_KEYS`).
- **Comfy-Org's fp16 file is the plain fp16 cast of every release tensor**, nothing more. No value exceeds the fp16
  range. A few layer-scale vectors (`scale1`/`scale2` of some late blocks) have values in fp16's subnormal range and
  lose precision in the cast; `h3_image.md` item 4 already records fp16 and fp32 decoding alike, so it does not show.
- The encoder (with `quant_conv`) is stored bf16-born in an fp32 file, consistent with it being the H3 video VAE's
  encoder; the decoder holds genuine fp32 values. Nothing was saved through a lossy cast.
- Layer scales, register tokens, `norm_out` and `proj_out` sit in the same range as P-H3's decoder. Nothing is dead,
  saturated or out of family.

## 2. The config against core

The release's `vae/config.json` (`vit_decoder_kwargs`, latent statistics, tiling keys) matches what core hard-codes:
24 layers, 16 heads, rope over half the head at theta 100 (`comfy/ldm/minimax/vae.py::MiniMaxH3ImageVAE`),
the same `LATENTS_MEAN` / `LATENTS_STD` (`comfy/ldm/minimax/vae.py`), and the release's tile size and minimum overlap
(`vae_tile_size`, `vae_tile_overlap_min`; core's defaults in `MiniMaxH3VideoVAE.__init__`). The diffusers port's
`vae_diffusers/config.json` agrees with both.

## 3. ComfyUI's decoder against three references

Core's decode path (`comfy/ldm/minimax/vae.py`: `H3ImageVAE`, `MiniMaxH3ImageVAE`, `ViT3DDecoder`, `Attention`,
`TransformerBlock`, `create_token_ids`, `RotaryEmbeddingND`, `split_tiles`, `tiled_decode`, `_finalize_pixels`) was
compared with:

- diffusers `src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_image.py` (`MiniMaxH3ImageViTDecoder2d`,
  `AutoencoderKLMiniMaxH3Image._decode`, `_split_tiles`, `_stitch_tiles`), line by line;
- vllm-omni `vllm_omni/diffusion/models/minimax_h3/image_vae.py`: pixel normalization, position grid, suffix tokens,
  `norm_out`, tile layout and assembly;
- sglang `python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3.py`, its port of the original decoder
  (`minimax_h3_video_vae/vae_vit.py`, `vit_utils.py`) and `configs/pipeline_configs/minimax_h3_image.py`: pixel
  normalization, position grid, register tokens, tile settings, decode precision.

On what each was read for, all agree: ImageNet pixel normalization in a `[0, 1]` base; rope positions normalized to `[-1, 1]` per tile
axis with the four register tokens and one zero token appended at position zero; per-head RMS q/k norm without
affine; split-half rope over the first half of each head; RMS-norm blocks with layer scales; LayerNorm before
`proj_out`; tiles laid out by the same rule. Core reads the fused qkv per head and gates the swiglu's first half,
which is the release layout (held by `bench/check_h3_image_vae_convert.py`).

Two differences, neither a cause of the stipple:

- **Decode precision.** sglang's image pipeline requires a bf16 VAE decode (`vae_decode_precision = "bf16"`), and
  vllm-omni's module docstring calls bf16 "the released image recipe". Core's fp16 decode is at least as precise.
- **Tile stitching** (section 4).

**Upstream at the GitHub tip on 2026-11-14** (sglang `999a6fa3dc`, vllm-omni `a0ca65891`, diffusers `43e30028a`): the
image-VAE code is as first landed (sglang `170804b40a`, vllm-omni `eb8b991ff`), with one follow-up each that only
lets the same decode run on other hardware (sglang `0f512dbfc5` for AMD, vllm-omni `fe0557dc2` for MUSA). The one
new image-output change is sglang `0ad34887ef` (#50101), last-bit rounding in sglang's own fused QK-norm + RoPE
kernel on SM100 and newer, with refreshed B200 reference images in `ce14d6d3ca`; it does not apply to an RTX 4090 or
to comfy-kitchen's kernel. **No sglang issue or PR reports image-VAE quality**: searched by title, body and comments
for H3-Image with vae, decode, tile, artifacts, stipple, colour and grain. vllm-omni's and diffusers' trackers were
not searched.

## 4. Tile stitching

The decoder never sees more than one tile. The picture is cut into overlapping tiles, each decoded on its own with
rope positions spanning `[-1, 1]` across that tile, and the overlaps are blended linearly. The overlap bands cover a
large share of the picture, not a thin seam (Table 5 of the record).

ComfyUI and the release stitch differently where a row overlap crosses a column overlap. diffusers (and vllm-omni,
which assembles the same way) blends each tile against its neighbours' *raw* decodes; core blends against the
already-blended canvas (`tiled_decode`'s own comment says this is on purpose).

- **Stand-in decoder** (`data/h3_image_vae_2026-11-14/stitch_probe.py`, no model): both stitchings return the
  picture exactly when tiles agree, and use the same tile layout at every size tried. When tiles disagree, diffusers'
  output steps by the full disagreement at every crossing and core's does not.
- **Real pictures** (record, Table 5): overall PSNR is equal to within a few hundredths of a dB, with diffusers' a
  hair higher every time and somewhat higher inside the crossings. Locally the two can differ a lot at crossings on
  text and hard edges: a step against a four-way blend. Which one reads better is a question for the eye, not
  answered here.

Neighbouring tiles disagree in fine detail by somewhat less than each tile's own error, and the blend averages that
disagreement, which softens fine detail in the overlaps a little (record, Table 3). It is not a consistent loss:
overlaps score below the rest of the picture on two pictures, above it on one, and mixed on the fourth.

## 5. Tile geometry is not a lever

Core hard-codes the release's geometry and ignores `VAEDecodeTiled`'s settings
(`comfy/ldm/minimax/vae.py::MiniMaxH3VideoVAE.decode_tiled`). The record's Table 2 tries larger tiles, a wider
overlap and no tiling. Every larger tile, and untiled, is worse on every picture for every decoder, sharply so for
the two 24-layer decoders, whose untiled output shows a patch grid (`crops_<picture>.png`, last column). A wider
overlap at the release's tile size changes nothing for those two. The positional encoding stretches across whatever tile it is
given (`create_token_ids`), so a bigger tile packs the positions closer than at the release's size; the results are
consistent with the decoders having been trained at that size, which the release does not state. No new
decode node is needed for this.

## 6. The three decoders on one latent

Same encoder, same latent, three decoders (record, Tables 1 and 4; crops and 3x zooms in
`data/h3_image_vae_2026-11-14/`):

- **P-H3's decoder** is closer to the picture on all four test pictures and carries less fine luma energy than the
  picture: smoother. The **release decoder** keeps about the picture's fine-detail energy but places it worse. On the
  zooms it adds pinpoint speckle on a dark wall and a dotted orange pattern on a lit metal rail that P-H3's does not.
  That matches the shadow stipple recorded in `h3_image.md` item 4. The shadow colour-grain ratio favours P-H3 on
  one picture only, so the stipple is not simply "more colour energy in shadows" and the ratio is a weak instrument
  for it.
- **Small lettering breaks the same way under these three decoders** (`crops_text_panel.png`). *Corrected the same
  day (section 8): a different decoder, `t1`, keeps much of it on encoded pictures, so the decoder carries much of
  the lettering, not only the latent.*
- **The HDR fine-tune** (`minimax_h3_image_hdr_vae_{fp32,fp16}.safetensors`, converted the same day by
  `bench/convert_h3_image_vae_from_diffusers.py`, commit `e78f6925`) shares the release encoder bit for bit and
  fine-tunes only `post_quant_conv` and the decoder. It carries more fine detail than the release decoder on three of the four pictures,
  with lower PSNR on every picture, and more colour grain on some pictures and less on others. It is a style choice, not a fix
  for the stipple.

## 7. What this does not settle

- **DiT latents.** Everything here decodes an encoded picture. A sampled latent may sit off the encoder's manifold,
  where the decoders could rank differently. The next step is the owner's eye on stacked decodes of the same sampled
  latents (release, P-H3, HDR), which needs a render.
- **Which stitching reads better.** Core's choice trades a hair of PSNR for no seam step at crossings; the eye has
  not judged it. Leave it unless a seam is seen.
- vllm-omni's and diffusers' issue trackers.

## 8. Follow-ups: averaging, the video decoder, and real renders

Record section 6. In short:

- **Averaging decodes does not help.** Decodes made with the tile grid shifted, averaged, score the same as one: the
  decoder's error does not depend on where the tiles sit. (A mirrored-latent arm was a wrong test, since a mirrored
  latent is not a mirrored picture's latent.)
- **All H3 VAEs share one encoder and one latent space.** P-H3's decoder is a fine-tune of the H3 video decoder with a
  one-frame output; the release image decoder is a separate, smaller model, not cut from the video decoder.
- **The video decoder.** Used as core uses it for a still (the latent decoded as a clip of copies), it decodes
  MiniMax-H3-Image latents about as well as the release decoder. Fed one latent directly with no retraining (the
  "surgery"), it is far worse, which is why `t1` exists.
- **`t1`** (`workflows/h3_config.py::IMAGE_VAE`, the community single-frame fine-tune of the video decoder that the
  H3 video model's one-frame path ships) ranks first on encoded pictures, with clearly more legible small text, and
  **smears real DiT output** (bread and wood grain turn painterly). Encoded-picture rankings do not predict DiT
  latents; only renders do.
- **On real renders** (the shipped text-to-image and edit graphs, sampled once, decoded four ways, full resolution
  in `$H3_OUTPUT_DIR/Image/vae_compare_2026-11-14/`): `release` and the video decoder look alike, crisp with a fine
  speckle; `p-h3` is a little softer and cleaner; `t1` smears. Lettering at 16 MP was legible under all four.

What would actually move the decode is a decoder trained on what the DiT samples, not on encoded pictures. That is
training, not a node.

## Pointers

| what | where |
|---|---|
| the follow-up probes and render crops | `data/h3_image_vae_2026-11-14/` (`ensemble_probe.py`, `video_decoder_probe.py`, `render_compare.py`, `render_*.png`) |
| the numbers, conditions and readings | [`bench/results/2026-11-14_h3_image_vae_tiles.md`](../../../../bench/results/2026-11-14_h3_image_vae_tiles.md) |
| the probe that made them | [`bench/probe_h3_image_vae_tiles.py`](../../../../bench/probe_h3_image_vae_tiles.py) |
| the diffusers-to-ComfyUI converter and its check | [`bench/convert_h3_image_vae_from_diffusers.py`](../../../../bench/convert_h3_image_vae_from_diffusers.py), [`bench/check_h3_image_vae_convert.py`](../../../../bench/check_h3_image_vae_convert.py) |
| results JSON, crops, zooms, the two throwaway probes and their outputs | `data/h3_image_vae_2026-11-14/` (untracked) |
| the page that owns the MiniMax-H3-Image VAE choice | [`docs/wiki/image/h3_image.md`](../../image/h3_image.md) item 4 |
