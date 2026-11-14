# The MiniMax-H3-Image VAE's decode tiles, and its decoder against P-H3's (2026-11-14)

lane: image

Date: 2026-11-14

The measured half of [`docs/wiki/research/image/2026-11-14_h3_image_vae_inspection.md`](../../docs/wiki/research/image/2026-11-14_h3_image_vae_inspection.md),
which carries the question, the static inspection and what this means. No DiT ran: every number below is an
encoded picture decoded again.

## 1. What was run

```
CUDA_VISIBLE_DEVICES=0 <comfy venv python> bench/probe_h3_image_vae_tiles.py --cuda \
    --release <MiniMax-H3-Image release folder> --out data/h3_image_vae_2026-11-14 \
    --vae release=<comfy>/models/vae/minimax_h3_image_vae_fp32.safetensors \
    --vae p-h3=<comfy>/models/vae/p-h3-image_vae_bf16.safetensors \
    --vae hdr=<comfy>/models/vae/minimax_h3_image_hdr_vae_fp32.safetensors
```

- **Code.** Repo at `e78f6925` with `bench/probe_h3_image_vae_tiles.py` as committed beside this record; core's
  `comfy/ldm/minimax/vae.py` classes (`MiniMaxH3ImageVAE`, `PH3ImageVAE`) called directly, not through a graph.
  diffusers' `_stitch_tiles` and `_blend` lifted from `coderef/diffusers` at `43e30028a`.
- **Card.** RTX 4090, torch `2.15.0+cu134`, every decoder in float32 (P-H3's bf16 file cast up), one decoder on the
  card at a time. ComfyUI's server held no models during the run (unloaded through its `/free` endpoint, queue empty).
- **Files.** `release` is the release's `vae/` weights as ComfyUI loads them; `p-h3` is PrunaAI's P-H3-Image VAE;
  `hdr` is the decoder fine-tune converted the same day by `bench/convert_h3_image_vae_from_diffusers.py`. All three
  share the encoder bit for bit (`h3_image.md` item 4 for `p-h3`; the converter's `--base` report for `hdr`).
- **Encoding.** Each picture encoded once by `release`'s encoder through `H3ImageVAE.encode` (core's fixed-seed
  posterior sample), so every decoder decodes the same latent.
- **Outputs**, untracked: `data/h3_image_vae_2026-11-14/results.json` (everything below),
  `crops_<picture>.png` (darkest and most detailed 256 px window; columns: picture, release, p-h3, hdr at 256/64,
  release untiled), `zoom3x_dark_{reference1,station}.png` (3x of the darkest window's centre, first four columns).

What the columns mean (`bench/probe_h3_image_vae_tiles.py` docstring): **luma detail kept** is the high-pass
(picture minus a 5x5 box blur) energy of the decode's luma over the picture's; above 1 means the decode carries more
fine luma energy than the picture, below 1 less. **Shadow chroma grain** is the same ratio for Cb and Cr over pixels
whose picture luma is under `0.15`; above 1 means added colour speckle in shadows. Both are energies, not placement:
a decode can keep the energy and still put it in the wrong places, which PSNR sees.

## 2. Tables

### Table 1. Each decoder at the release's tiling (256 px tiles, 64 px minimum overlap)

| picture | decoder | PSNR dB | luma detail kept | shadow chroma grain |
|---|---|---|---|---|
| `reference1` | release | 29.91 | 1.06 | 1.98 |
| `reference1` | p-h3 | 31.47 | 0.79 | 1.05 |
| `reference1` | hdr | 29.47 | 1.11 | 1.78 |
| `text_panel` | release | 25.45 | 0.98 | 2.31 |
| `text_panel` | p-h3 | 26.16 | 0.88 | 3.10 |
| `text_panel` | hdr | 24.73 | 1.18 | 4.27 |
| `dark_portrait` | release | 32.83 | 1.03 | 1.27 |
| `dark_portrait` | p-h3 | 34.22 | 0.70 | 1.38 |
| `dark_portrait` | hdr | 32.20 | 1.00 | 1.16 |
| `station` | release | 24.23 | 1.09 | 1.19 |
| `station` | p-h3 | 25.50 | 0.76 | 1.27 |
| `station` | hdr | 23.50 | 1.13 | 1.82 |

### Table 2. Tile geometry (tile/overlap in px), PSNR dB / luma detail kept / shadow chroma grain

| picture | decoder | 256/64 | 256/128 | 384/96 | 512/64 | 512/128 | untiled |
|---|---|---|---|---|---|---|---|
| `reference1` | release | 29.91 / 1.06 / 1.98 | 29.88 / 1.06 / 1.89 | 25.09 / 1.63 / 2.93 | 22.34 / 2.49 / 3.78 | 22.34 / 2.49 / 3.78 | 20.08 / 3.35 / 4.84 |
| `reference1` | p-h3 | 31.47 / 0.79 / 1.05 | 31.27 / 0.78 / 1.06 | 26.97 / 0.74 / 1.14 | 23.15 / 0.87 / 1.38 | 23.15 / 0.87 / 1.38 | 18.52 / 0.63 / 1.16 |
| `reference1` | hdr | 29.47 / 1.11 / 1.78 | 29.39 / 1.12 / 1.75 | 24.84 / 1.65 / 3.04 | 22.21 / 2.45 / 4.13 | 22.21 / 2.45 / 4.13 | 20.04 / 3.23 / 5.39 |
| `text_panel` | release | 25.45 / 0.98 / 2.31 | 25.47 / 0.99 / 2.27 | 19.77 / 1.30 / 4.72 | 17.71 / 1.54 / 7.83 | 17.71 / 1.54 / 7.83 | 16.37 / 1.81 / 10.21 |
| `text_panel` | p-h3 | 26.16 / 0.88 / 3.10 | 25.94 / 0.90 / 3.11 | 21.43 / 0.86 / 3.47 | 18.78 / 0.80 / 3.58 | 18.78 / 0.80 / 3.58 | 16.35 / 0.67 / 3.55 |
| `text_panel` | hdr | 24.73 / 1.18 / 4.27 | 24.73 / 1.18 / 4.17 | 19.41 / 1.46 / 6.56 | 17.54 / 1.67 / 9.82 | 17.54 / 1.67 / 9.82 | 16.34 / 1.84 / 12.77 |
| `dark_portrait` | release | 32.83 / 1.03 / 1.27 | 32.86 / 1.01 / 1.27 | 29.24 / 1.35 / 2.33 | 26.83 / 1.72 / 3.77 | 26.83 / 1.72 / 3.77 | 24.81 / 2.08 / 4.46 |
| `dark_portrait` | p-h3 | 34.22 / 0.70 / 1.38 | 34.20 / 0.68 / 1.36 | 29.51 / 0.63 / 1.61 | 25.90 / 0.62 / 1.71 | 25.90 / 0.62 / 1.71 | 22.66 / 0.45 / 1.26 |
| `dark_portrait` | hdr | 32.20 / 1.00 / 1.16 | 32.22 / 0.99 / 1.14 | 28.71 / 1.39 / 2.39 | 26.26 / 1.94 / 4.45 | 26.26 / 1.94 / 4.45 | 24.25 / 2.51 / 5.84 |
| `station` | release | 24.23 / 1.09 / 1.19 | 24.19 / 1.08 / 1.17 | 19.89 / 1.61 / 2.59 | 17.56 / 2.17 / 4.45 | 17.56 / 2.17 / 4.45 | 15.88 / 2.79 / 6.32 |
| `station` | p-h3 | 25.50 / 0.76 / 1.27 | 25.36 / 0.75 / 1.27 | 22.17 / 0.71 / 1.37 | 19.75 / 0.74 / 1.53 | 19.75 / 0.74 / 1.53 | 17.17 / 0.62 / 1.91 |
| `station` | hdr | 23.50 / 1.13 / 1.82 | 23.47 / 1.13 / 1.79 | 19.67 / 1.62 / 3.41 | 17.54 / 2.16 / 5.30 | 17.54 / 2.16 / 5.30 | 15.99 / 2.66 / 6.83 |

### Table 3. Neighbouring tiles in their overlaps (RMS, pixels in [0, 1]; mean over neighbour pairs)

| picture | decoder | pairs | tiles disagree | ... in fine detail | each tile's own error | picture's fine detail | one tile's fine detail | blended fine detail |
|---|---|---|---|---|---|---|---|---|
| `reference1` | release | 40 | 0.0304 | 0.0231 | 0.0376 | 0.0327 | 0.0344 | 0.0322 |
| `reference1` | p-h3 | 40 | 0.0314 | 0.0226 | 0.0352 | 0.0327 | 0.0309 | 0.0281 |
| `reference1` | hdr | 40 | 0.0305 | 0.0223 | 0.0412 | 0.0327 | 0.0340 | 0.0321 |
| `text_panel` | release | 31 | 0.0506 | 0.0445 | 0.0641 | 0.0838 | 0.0829 | 0.0805 |
| `text_panel` | p-h3 | 31 | 0.0576 | 0.0506 | 0.0660 | 0.0838 | 0.0791 | 0.0757 |
| `text_panel` | hdr | 31 | 0.0547 | 0.0486 | 0.0702 | 0.0838 | 0.0907 | 0.0879 |
| `dark_portrait` | release | 22 | 0.0137 | 0.0101 | 0.0209 | 0.0153 | 0.0156 | 0.0149 |
| `dark_portrait` | p-h3 | 22 | 0.0143 | 0.0098 | 0.0182 | 0.0153 | 0.0135 | 0.0127 |
| `dark_portrait` | hdr | 22 | 0.0142 | 0.0102 | 0.0228 | 0.0153 | 0.0152 | 0.0146 |
| `station` | release | 31 | 0.0494 | 0.0408 | 0.0614 | 0.0573 | 0.0587 | 0.0557 |
| `station` | p-h3 | 31 | 0.0517 | 0.0414 | 0.0590 | 0.0573 | 0.0507 | 0.0468 |
| `station` | hdr | 31 | 0.0498 | 0.0408 | 0.0683 | 0.0573 | 0.0589 | 0.0561 |

### Table 4. RMS error of unstitched tiles by distance from the tile's nearest edge

| picture | decoder | 0-16px | 16-32px | 32-48px | 48-64px | 64-80px | 80-96px | 96-112px | 112-128px |
|---|---|---|---|---|---|---|---|---|---|
| `reference1` | release | 0.0415 | 0.0333 | 0.0335 | 0.0339 | 0.0324 | 0.0347 | 0.0367 | 0.0373 |
| `reference1` | p-h3 | 0.0411 | 0.0304 | 0.0289 | 0.0282 | 0.0266 | 0.0277 | 0.0302 | 0.0300 |
| `reference1` | hdr | 0.0454 | 0.0365 | 0.0354 | 0.0353 | 0.0335 | 0.0356 | 0.0381 | 0.0393 |
| `text_panel` | release | 0.0677 | 0.0610 | 0.0606 | 0.0562 | 0.0569 | 0.0528 | 0.0503 | 0.0488 |
| `text_panel` | p-h3 | 0.0731 | 0.0616 | 0.0591 | 0.0523 | 0.0514 | 0.0462 | 0.0435 | 0.0414 |
| `text_panel` | hdr | 0.0734 | 0.0658 | 0.0655 | 0.0614 | 0.0612 | 0.0574 | 0.0549 | 0.0525 |
| `dark_portrait` | release | 0.0272 | 0.0224 | 0.0199 | 0.0227 | 0.0225 | 0.0282 | 0.0289 | 0.0348 |
| `dark_portrait` | p-h3 | 0.0270 | 0.0203 | 0.0175 | 0.0193 | 0.0194 | 0.0236 | 0.0245 | 0.0308 |
| `dark_portrait` | hdr | 0.0302 | 0.0242 | 0.0220 | 0.0240 | 0.0243 | 0.0281 | 0.0293 | 0.0348 |
| `station` | release | 0.0704 | 0.0593 | 0.0613 | 0.0641 | 0.0633 | 0.0621 | 0.0661 | 0.0571 |
| `station` | p-h3 | 0.0710 | 0.0558 | 0.0542 | 0.0553 | 0.0535 | 0.0518 | 0.0547 | 0.0477 |
| `station` | hdr | 0.0777 | 0.0657 | 0.0664 | 0.0690 | 0.0684 | 0.0665 | 0.0707 | 0.0626 |

### Table 5. Core's stitching against diffusers' on the same raw tiles

| picture | decoder | overlap area | crossing area | PSNR core / diffusers | in crossings core / diffusers | core in overlaps / outside | max abs core - diffusers |
|---|---|---|---|---|---|---|---|
| `reference1` | release | 0.44 | 0.06 | 29.91 / 29.93 | 28.59 / 28.76 | 29.63 / 30.14 | 0.222 |
| `reference1` | p-h3 | 0.44 | 0.06 | 31.47 / 31.52 | 29.82 / 30.29 | 31.10 / 31.79 | 0.199 |
| `reference1` | hdr | 0.44 | 0.06 | 29.47 / 29.48 | 27.63 / 27.71 | 29.02 / 29.86 | 0.225 |
| `text_panel` | release | 0.42 | 0.06 | 25.45 / 25.48 | 24.15 / 24.55 | 24.69 / 26.11 | 0.635 |
| `text_panel` | p-h3 | 0.42 | 0.06 | 26.16 / 26.21 | 23.60 / 24.13 | 25.08 / 27.16 | 0.832 |
| `text_panel` | hdr | 0.42 | 0.06 | 24.73 / 24.76 | 23.31 / 23.72 | 23.93 / 25.43 | 0.739 |
| `dark_portrait` | release | 0.59 | 0.13 | 32.83 / 32.84 | 34.35 / 34.45 | 32.80 / 32.87 | 0.090 |
| `dark_portrait` | p-h3 | 0.59 | 0.13 | 34.22 / 34.23 | 35.64 / 35.76 | 34.17 / 34.30 | 0.154 |
| `dark_portrait` | hdr | 0.59 | 0.13 | 32.20 / 32.21 | 33.42 / 33.54 | 32.38 / 31.96 | 0.113 |
| `station` | release | 0.42 | 0.06 | 24.23 / 24.24 | 25.41 / 25.65 | 24.56 / 24.01 | 0.416 |
| `station` | p-h3 | 0.42 | 0.06 | 25.50 / 25.52 | 26.07 / 26.49 | 25.63 / 25.40 | 0.400 |
| `station` | hdr | 0.42 | 0.06 | 23.50 / 23.51 | 24.34 / 24.55 | 23.73 / 23.34 | 0.499 |

Pictures: `reference1` = `examples/reference1.png` whole, 1024x1024; `text_panel` = `assets/showcase/text.jpg` cropped at [1952, 0, 1024, 832], 1024x832; `dark_portrait` = `assets/showcase/photo.jpg` cropped at [1008, 0, 976, 544], 976x544; `station` = `assets/showcase/scene.jpg` cropped at [0, 848, 1024, 832], 1024x832

`512/64` and `512/128` give identical rows because on these sizes `split_tiles` lands both on the same layout (the
slack is spread over the overlaps until they meet the tile count, so both end with the same overlaps).

## 3. The two throwaway probes behind the static claims

Both live beside their outputs in the untracked `data/h3_image_vae_2026-11-14/`, with no check:

- `stitch_probe.py` → `stitch_probe.txt`. Core's real `tiled_decode` and diffusers' `_stitch_tiles` around a stand-in
  decoder that returns a known smooth picture plus a per-tile constant offset. With the offset off, both return the
  picture to float32 rounding, and the tile layout matches diffusers' `_split_tiles` at every size tried. With
  the offset on, they differ only at row-and-column overlap crossings, where diffusers' output steps by the full
  tile disagreement and core's does not.
- `vae_stats.py` → `vae_stats.txt`. The release `vae/` weights against Comfy-Org's fp16 file and P-H3's: finiteness,
  all-zero tensors, fp16 range and subnormals, which tensors are bf16-born, layer-scale, register-token and
  output-layer magnitudes per block.

## 4. Readings

- **The tile geometry is the decoders' own.** Every larger tile, and untiled, lowers PSNR on every picture and
  decoder (Table 2), and for the 24-layer decoders (`release`, `hdr`) it also multiplies both energy ratios: the
  crops' untiled column shows a patch grid. A wider overlap at 256 (`256/128`) is within a few hundredths of a dB of
  `256/64` for `release` and `hdr`, and a few tenths lower for `p-h3` on two pictures. Nothing here argues for changing core's hard-coded geometry.
- **Neighbouring tiles disagree by somewhat less than each tile's own error** (Table 3), and most of that
  disagreement is fine detail. The blend averages it: the blended fine-detail energy is below one tile's, a little for `release` and
  `hdr`, more for `p-h3`. PSNR inside overlaps is lower than outside on `reference1` and `text_panel`, higher on `station`,
  and mixed on `dark_portrait` (Table 5), so the blend is not a consistent loss.
- **Each tile's outer 16 px is worse than the band just inside it** (Table 4) on every picture and decoder (on
  `dark_portrait` the tiles' centres are worse still, which follows that picture's content). The outer band falls
  inside the overlap on interior edges and gets the low blend weight, which is what the overlap is for. The picture's own border has no
  neighbour, so its outer 16 px is decoded at that edge error.
- **Core's stitching against the release's on real pictures** (Table 5): overall PSNR equal to within a few hundredths
  of a dB, diffusers' a hair higher every time, and higher by up to about half a dB inside the crossings. Locally the two
  differ by large amounts at crossings on text and high-contrast detail (last column): diffusers' step against core's
  four-way blend. This record does not say which reads better; that is the eye's.
- **`p-h3` against `release`, same latent** (Table 1): higher PSNR on all four pictures, less fine luma energy than
  the picture on all four (`release` stays near the picture's, above it on three). Shadow chroma grain is lower for `p-h3` on `reference1` and
  higher on the other three. So `p-h3` is closer to the picture and smoother; the `release` decoder keeps texture
  energy but places it worse. On the zooms, `release` shows pinpoint speckle on `reference1`'s dark wall and a dotted
  orange pattern on `station`'s bench rail that `p-h3` does not.
- **Small lettering breaks the same way under all three decoders** (`crops_text_panel.png`, the "TRUE"/"FALSE"
  table). A failure shared by three decoders of one latent is in the latent, not the decoder. *Does not hold:
  section 7.*
- **`hdr`** carries fine luma energy at or above the picture's on all four, more than `release` on three, at lower PSNR
  than `release` on all four; its shadow chroma grain is above `release`'s on `text_panel` and `station` and below it
  on the other two. Its model card's reconstruction table points the same way on PSNR and detail.

## 5. What this cannot see

- Latents from the DiT. An encoded picture's latent is on the encoder's manifold; a sampled one may not be, and a
  decoder can behave differently there. The repo's DiT-latent observations are `docs/wiki/image/h3_image.md` item 4.
- The eye. PSNR and the two energy ratios are proxies; the stacked crops are for looking, not verdicts.
- Four pictures, three of them JPEG montage crops whose own compression artifacts sit in the reference.
- Decode precision. Everything ran float32; the shipped graphs and both servers decode at lower precision
  (`h3_image.md` item 4 records fp16 and fp32 decoding alike for `release`).

## 6. Follow-ups the same day: averaging, the video decoder, and real renders

### 6.1 Averaging decodes (throwaway `data/h3_image_vae_2026-11-14/ensemble_probe.py`, log beside it)

CPU float32, `reference1`, `release` only; stopped there because the answer was plain. The mean of decodes made
with the tile grid shifted by 128 px (two or four grids) scored the same PSNR as one decode, so the decoder's error
does not depend on where the grid sits and averaging cannot cancel it. The mirrored-latent arm scored far lower,
but the test was wrong: a mirrored latent is not the latent of a mirrored picture (the encoder's filters are not
mirror-symmetric), so that arm says nothing about the decoder.

### 6.2 The video decoder and its fine-tune on encoded pictures (throwaway `video_decoder_probe.py`, `video_decoder_results.json`, `video_decoder_probe.log`, `crops_video_<picture>.png`)

Same pictures, same latents and same metrics as Tables 1 and 2, on the card in float32. `video_clip` is core's still
path for the stock video VAE (`minimax_h3_video_vae_fp16.safetensors`: the latent decoded as a clip of copies, one
frame kept); `video_direct_k` is that decoder fed the one latent alone, keeping output frame `k` (untrained surgery);
`t1` is `minimax_h3_t1_image_vae_step1597.safetensors` (`workflows/h3_config.py::IMAGE_VAE`). Cells: PSNR dB / luma
detail kept / shadow chroma grain.

### Table 6. Video-decoder arms against the image decoders, encoded pictures

| picture | release | p-h3 | t1 | video_clip | video_direct_0 | video_direct_3 | video_direct_mean |
|---|---|---|---|---|---|---|---|
| `reference1` | 29.91 / 1.06 / 1.98 | 31.47 / 0.79 / 1.05 | 32.57 / 0.69 / 0.85 | 29.84 / 0.82 / 0.59 | 20.54 / 0.29 / 0.72 | 21.05 / 0.33 / 0.66 | 20.64 / 0.28 / 0.66 |
| `text_panel` | 25.45 / 0.98 / 2.31 | 26.16 / 0.88 / 3.10 | 26.64 / 0.77 / 1.55 | 24.65 / 0.69 / 1.15 | 16.48 / 0.17 / 0.81 | 16.90 / 0.24 / 0.80 | 16.61 / 0.18 / 0.71 |
| `dark_portrait` | 32.83 / 1.03 / 1.27 | 34.22 / 0.70 / 1.38 | 35.16 / 0.58 / 0.53 | 32.93 / 0.68 / 0.71 | 25.03 / 0.25 / 0.88 | 25.24 / 0.29 / 0.82 | 24.98 / 0.24 / 0.78 |
| `station` | 24.23 / 1.09 / 1.19 | 25.50 / 0.76 / 1.27 | 26.33 / 0.63 / 0.81 | 24.57 / 0.79 / 0.79 | 16.24 / 0.29 / 0.63 | 16.77 / 0.34 / 0.65 | 16.34 / 0.29 / 0.58 |

Weights (`vae_stats.py`'s method, run inline): every image VAE's encoder equals the video VAE's (bitwise or equal
after fp16 rounding on all 118 tensors; `t1` bitwise). P-H3's decoder has the video decoder's shapes except a
one-frame `proj_out`, and its weights sit close to the video decoder's: a fine-tune of it. The `release` decoder's
weights match no slice of the video decoder's better than random vectors do: trained on its own.

### 6.3 Real renders: one DiT sample, four decoders

The shipped `workflows/image/h3_image_t2i_api.json` (seed 42, 4736x3552) and `h3_image_edit_api.json` (seed 1234,
`h3_image_reference1.png`, 4096x4096), each queued four times on ComfyUI with only a decode-only `VAELoader` feeding
`VAEDecode` changed (references still encoded through the graph's P-H3 VAE, so sampling ran once per graph and was
cached), saved from `VAEDecode` at full resolution before the downscale: `$H3_OUTPUT_DIR/Image/vae_compare_2026-11-14/`.
Decoders: `minimax_h3_image_vae_fp16` (release, as shipped), `p-h3-image_vae_bf16`, `minimax_h3_t1_image_vae_step1597`,
`minimax_h3_video_vae_fp16`. Crops (`render_compare.py`, columns release, p-h3, t1, video):
`render_t2i_{dark,detail}{,_x2}.png`, `render_edit_{dark,detail,sign}{,_x2}.png`,
`render_edit_{mug_left,shelf_sign,poster,dark_wall,hands_watch}.png`.

What the crops show (a session's look, listed for the owner's eye, not a verdict):

- **`t1` smears DiT output.** On the bread and the dark wood grain it turns texture painterly and wavy; on encoded
  pictures it scored best. A sampled latent is not an encoded one, and `t1` was tuned on encoded ones.
- **`release` and `video`** look alike: crisp, with a fine speckle over texture. **`p-h3`** is a little softer and
  cleaner.
- **Lettering at this size is legible under all four** (mugs, shelf sign); the edit's text is large at 16 MP.

## 7. Readings corrected by section 6

- Section 4's "small lettering breaks the same way under all three decoders ... in the latent, not the decoder" does
  not hold: on encoded pictures `t1` keeps the `text_panel` lettering clearly more legible than `release` and `p-h3`
  (`crops_video_text_panel.png`; `hdr` breaks it like `release` in `crops_text_panel.png`), so the decoder carries
  much of it. (On the real renders at 16 MP the text was
  large enough that no decoder broke it.)
- The encoded-picture ranking does not carry over to DiT latents (`t1` first on Table 6, worst on the renders), so
  Tables 1 to 6 rank decoders on encoded pictures only.
