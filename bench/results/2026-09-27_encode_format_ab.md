# #38: the save format, measured (2026-09-27)

`docs/state/open_experiments.md` #38. O1 (`2026-09-27_o1_lossless.md`) found that
the dark blocking comes from the save (8-bit 4:2:0 h264 at crf 19), not the
model. This measures the candidates against the VAE's own frames, with no
render. `bench/encode_format_ab.py` (eb339fea) decodes the first 22 latent
frames (73 video frames) of four saved latents on the card. It encodes them
with each format's VHS arguments, reads them back, and scores them. Raw
numbers: `2026-09-27_encode_format_ab.json`. Four clips, one seed, 73 frames
each; installed kitchen `0.2.35+sol.fc32da2.up.c8c7825`, RTX 4090, run while
no server was up.

Clips: look_noir and look_anchor, on PDD8 (`text_to_video_pdd_savelat`) and
FlashGen (`text_to_video_flashgen_savelat`).

## Dark block-edge excess over the lossless frames (dark 8 px)

The `dark_block8` of O1, candidate minus the VAE's frames. Lower is better,
and 0 means the encode adds no dark blocking. Bitrate is the mean ratio to
today's format.

| candidate | noir PDD8 | noir FlashGen | anchor PDD8 | anchor FlashGen | mean | size vs today |
|---|---|---|---|---|---|---|
| `h264_crf19_8bit` (today) | +0.047 | -0.003 | +0.069 | +0.015 | +0.032 | 1.00x |
| `h264_crf14_8bit` | +0.076 | +0.021 | +0.101 | +0.046 | +0.061 | 2.47x |
| `h264_crf10_8bit` | +0.071 | +0.040 | +0.087 | +0.062 | +0.065 | 5.30x |
| `h264_crf19_10bit` | +0.026 | -0.015 | +0.029 | -0.002 | +0.010 | 0.94x |
| `h265_crf22_10bit` | +0.035 | -0.004 | +0.031 | +0.007 | +0.017 | 0.52x |
| `h265_crf18_10bit` | +0.049 | +0.004 | +0.055 | +0.020 | +0.032 | 1.04x |
| `av1_crf23_10bit` | +0.064 | +0.019 | +0.058 | +0.043 | +0.046 | 0.68x |
| `ffv1_floor_8bit` (lossless 8-bit 4:2:0) | +0.006 | +0.003 | +0.011 | +0.002 | +0.005 | 26.6x |
| `ffv1_floor_10bit` (lossless 10-bit 4:2:0) | -0.000 | -0.002 | -0.006 | -0.005 | -0.004 | 48.4x |

## Mean absolute luma error in the darks (8-bit levels)

| candidate | noir PDD8 | noir FlashGen | anchor PDD8 | anchor FlashGen |
|---|---|---|---|---|
| `h264_crf19_8bit` (today) | 1.00 | 1.01 | 0.86 | 1.08 |
| `h264_crf14_8bit` | 0.80 | 0.79 | 0.70 | 0.88 |
| `h264_crf10_8bit` | 0.63 | 0.62 | 0.57 | 0.70 |
| `h264_crf19_10bit` | 0.91 | 0.94 | 0.75 | 0.99 |
| `h265_crf22_10bit` | 1.00 | 1.02 | 0.83 | 1.08 |
| `h265_crf18_10bit` | 0.82 | 0.84 | 0.69 | 0.91 |
| `av1_crf23_10bit` | 0.91 | 0.95 | 0.79 | 1.01 |

## What it says

- **The blocking comes from the lossy codec, not from 8-bit 4:2:0 itself.**
  Lossless 8-bit adds almost none (+0.005 mean).
- **A lower crf in 8-bit makes the blocking worse, not better.** crf 14 and
  crf 10 double the excess, while cutting the error and multiplying the size
  by 2.5 and 5.3. So "just raise the quality" is out. A likely reason, which
  is inference and not tested here: the codec's deblocking weakens as its
  quantiser drops.
- **10-bit is what reduces it.** H.264 10-bit cuts the mean excess from
  +0.032 to +0.010 at the same size. H.265 10-bit at VHS's default crf 22
  cuts it to +0.017, at about half the size, with the same dark error as
  today. The same trend inside H.265: crf 18 blocks more than crf 22.
- **The excess is on the PDD8 clips.** The FlashGen clips are near zero even
  today, consistent with FlashGen lifting blacks, so fewer pixels sit near
  the floor.
- **#38's bar is not met by any candidate.** The bar was dark_block8 within
  0.02 of lossless on every clip. The nearest are H.264 10-bit (+0.026 and
  +0.029 on PDD8) and H.265 10-bit crf 22 (+0.035 and +0.031); today's
  format sits at +0.047 and +0.069.

## Recommendation, for the owner

- Switch to **H.265 10-bit at crf 22** (VHS `video/h265-mp4`, `yuv420p10le`,
  tagged `hvc1`). It measures second best on blocking and matches today's
  dark error at half the size. It plays wherever HEVC does.
- H.264 10-bit measures best, but few players take it.
- The switch is gated on:
  - the owner's players;
  - one pair seen by eye (the metric is O1's, and has not been checked
    against the owner's read);
  - the clip-measure tools reading 10-bit with accurate rounding. swscale's
    default path reads a full-white 10-bit frame back as 253, and
    `encode_format_ab.py`'s `READ_BACK` documents that.
- Every clip measure after the switch is on a different encode, so the
  CHANGELOG entry must say so.

## The whole clip, and the pair for the owner (2026-09-27, later)

The same comparison on all 345 frames of look_noir PDD8 (the clean rerun,
render `2026-09-26_followup:look_noir__pdd8:2`), for today's format and the
recommendation only. Raw numbers:
`2026-09-27_encode_format_ab_fullclip.json`.

| candidate | dark 8 px excess | dark error | bitrate |
|---|---|---|---|
| `h264_crf19_8bit` (today) | +0.048 | 0.71 | 2481 kbps |
| `h265_crf22_10bit` | +0.003 | 0.71 | 1337 kbps |

On the whole clip, H.265 10-bit removes nearly all the added dark blocking
at the same dark error and 54% of the bitrate. Both encodes are silent
copies in `Video/review_38/` on the output share, and are on the board's "For
your eye" tab as `fmt38-look_noir_pdd8`.

## The owner's read, and what it says about the measure (2026-09-27)

The owner watched the pair: "Dont think I can see a difference. I wonder if
its just because of the resolution limits?"

It is not resolution. The two encodes, read back with accurate rounding,
differ by 0.73 levels on average in the dark pixels (99th percentile 4.3),
and 1.06 over the whole frame. The dark 192 px window where they differ
most (frame 276) was saved enlarged 4x, once as-is and once with the shadows
lifted, in `Video/review_38/zoom4x_*.png`. Even there the two are hard to tell
apart: H.264 has slightly more fine speckle, and neither shows visible 8 px
blocking.

- **Calibration of O1's `dark_block8`:** an excess of about +0.05 over
  lossless is below what the owner sees at normal viewing. It detects the
  codec, but at this size the codec's contribution does not matter to the
  eye.
- **So the save format stays H.264 8-bit crf 19.** H.265 10-bit would halve
  file sizes at equal quality, a convenience rather than a fix, and only
  where the owner's players take HEVC.
- **O1 is open again.** Whatever the owner saw as "blocky and splotchy"
  blacks is not the codec's 8 px blocking. If a specific clip shows it, the
  test is O1's own: decode that clip's latent losslessly and look at the
  same frames enlarged.
