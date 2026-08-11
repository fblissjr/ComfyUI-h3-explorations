# MiniMax H3: every legal resolution, and why the set is what it is

Last updated: 2026-08-11.

Every number here comes from running `comfy_extras/nodes_minimax_h3.py` and
`comfy/ldm/minimax/model.py`, not from reading them.

## What to type

Ask for an aspect ratio. These are the fourteen you are most likely to want,
with what `adapt_canvas()` returns for each.

| Ask for | Resolution | Latent | Video tokens/frame | Attention |
|---|---|---|---|---|
| 21:9 | 1536x672 | 96x42 | 1008 | 1.00x |
| 2:1 | 1440x704 | 90x44 | 990 | 0.96x |
| 16:9 | 1344x768 | 84x48 | 1008 | 1.00x |
| 5:3 | 1280x768 | 80x48 | 960 | 0.91x |
| 3:2 | 1152x768 | 72x48 | 864 | 0.73x |
| 4:3 | 1024x768 | 64x48 | 768 | 0.58x |
| 5:4 | 960x768 | 60x48 | 720 | 0.51x |
| 1:1 | 768x768 | 48x48 | 576 | 0.33x |
| 4:5 | 768x960 | 48x60 | 720 | 0.51x |
| 3:4 | 768x1024 | 48x64 | 768 | 0.58x |
| 2:3 | 768x1152 | 48x72 | 864 | 0.73x |
| 9:16 | 768x1344 | 48x84 | 1008 | 1.00x |
| 1:2 | 704x1440 | 44x90 | 990 | 0.96x |
| 9:21 | 672x1536 | 42x96 | 1008 | 1.00x |

Attention is quoted against 1:1 at the same frame count, and it goes as the
square of the token count, because attention is O(S^2) and video tokens
dominate S. 1:1 costs a third of 16:9. That is the largest single lever in
this repo, larger than any kernel or sparsity setting.

## The complete set

`adapt_canvas()` produces exactly 95 distinct resolutions across the whole
legal aspect range of 1/4 to 4. They are symmetric, so the 48 below cover
landscape and square, and every one has a portrait mirror at identical cost:
768x1344 costs what 1344x768 costs, because video tokens per frame are
`(W/32) x (H/32)` and that is symmetric.

The last column flags the 26 resolutions that do not reproduce themselves.
Type one of those back into `adapt_canvas()` and you get a neighbour,
because the area cap scales the pair and the rounding does not land where it
started. They are reachable, they are legal, they are simply not fixed
points. The other 69 are stable.

| Resolution | Latent | Video tokens/frame | Attention | Re-derives |
|---|---|---|---|---|
| 2016x512 | 126x32 | 1008 | 1.00x |  |
| 1984x512 | 124x32 | 992 | 0.97x |  |
| 1952x512 | 122x32 | 976 | 0.94x | no |
| 1952x544 | 122x34 | 1037 | 1.06x | no |
| 1920x544 | 120x34 | 1020 | 1.02x |  |
| 1888x544 | 118x34 | 1003 | 0.99x |  |
| 1856x544 | 116x34 | 986 | 0.96x | no |
| 1856x576 | 116x36 | 1044 | 1.07x | no |
| 1824x576 | 114x36 | 1026 | 1.04x | no |
| 1792x576 | 112x36 | 1008 | 1.00x |  |
| 1760x576 | 110x36 | 990 | 0.96x |  |
| 1728x576 | 108x36 | 972 | 0.93x | no |
| 1728x608 | 108x38 | 1026 | 1.04x |  |
| 1696x608 | 106x38 | 1007 | 1.00x |  |
| 1664x608 | 104x38 | 988 | 0.96x | no |
| 1664x640 | 104x40 | 1040 | 1.06x | no |
| 1632x640 | 102x40 | 1020 | 1.02x |  |
| 1600x640 | 100x40 | 1000 | 0.98x |  |
| 1568x640 | 98x40 | 980 | 0.95x | no |
| 1568x672 | 98x42 | 1029 | 1.04x | no |
| 1536x672 | 96x42 | 1008 | 1.00x |  |
| 1504x672 | 94x42 | 987 | 0.96x |  |
| 1504x704 | 94x44 | 1034 | 1.05x | no |
| 1472x704 | 92x44 | 1012 | 1.01x |  |
| 1440x704 | 90x44 | 990 | 0.96x |  |
| 1440x736 | 90x46 | 1035 | 1.05x | no |
| 1408x736 | 88x46 | 1012 | 1.01x |  |
| 1376x736 | 86x46 | 989 | 0.96x |  |
| 1376x768 | 86x48 | 1032 | 1.05x | no |
| 1344x768 | 84x48 | 1008 | 1.00x |  |
| 1312x768 | 82x48 | 984 | 0.95x |  |
| 1280x768 | 80x48 | 960 | 0.91x |  |
| 1248x768 | 78x48 | 936 | 0.86x |  |
| 1216x768 | 76x48 | 912 | 0.82x |  |
| 1184x768 | 74x48 | 888 | 0.78x |  |
| 1152x768 | 72x48 | 864 | 0.73x |  |
| 1120x768 | 70x48 | 840 | 0.69x |  |
| 1088x768 | 68x48 | 816 | 0.66x |  |
| 1056x768 | 66x48 | 792 | 0.62x |  |
| 1024x768 | 64x48 | 768 | 0.58x |  |
| 992x768 | 62x48 | 744 | 0.54x |  |
| 960x768 | 60x48 | 720 | 0.51x |  |
| 928x768 | 58x48 | 696 | 0.48x |  |
| 896x768 | 56x48 | 672 | 0.44x |  |
| 864x768 | 54x48 | 648 | 0.41x |  |
| 832x768 | 52x48 | 624 | 0.38x |  |
| 800x768 | 50x48 | 600 | 0.35x |  |
| 768x768 | 48x48 | 576 | 0.33x |  |

## Why there is no resolution setting

You choose a ratio. `adapt_canvas()` derives the pixels, in three steps:

1. The short edge starts at 768.
2. If the area exceeds 768 x 1344 = 1,032,192, the whole resolution
   scales down until it fits.
3. Each axis rounds to a multiple of 32.

Asking for 4K returns the same resolution as asking for 720p at the same
ratio. There is no quality dial here, only a shape choice.

One caveat that matters in practice: core's conditioning nodes do not call
`adapt_canvas()` on the video resolution at all. `MiniMaxH3ImageToVideo`,
`MiniMaxH3ReferenceToVideo` and `EmptyMiniMaxH3LatentAV` take `width` and
`height` as plain integers at `min=32, step=32`, so whatever multiple of 32
you type is what you get. The 768 short edge and the area cap describe the
family the model was trained on, and the reference pipeline enforces them.
Inside ComfyUI the only places they are applied are core's reference-video
sizing and this repo's `MiniMaxH3KeyframeCanvas`, which derives the
resolution from your first keyframe.

So: 32-divisibility is a hard requirement of the architecture. The 768 and
the area cap are the trained distribution. You can leave the second without
leaving the first, and nothing will stop you.

## Where the 32 comes from

Two stages multiply, and neither is Qwen's.

The video VAE compresses space by 16, so 1344x768 encodes to an 84x48
latent.

The DiT then patchifies that latent with `patch_size=(1, 2, 2)`, 2x2 in
space and 1 in time, before anything is attended. 84x48 becomes 42x24, which
is 1008 video tokens per latent frame.

16 x 2 = 32. Divisibility by 16 alone is not sufficient: it leaves an odd
latent axis, and the 2x2 patchify cannot tile it.

Qwen3-VL's vision tower separately lands on 32, using `patch_size=16` with
`merge_size=2`. That governs images entering the text encoder, on Qwen's own
grid, after its own resize between `min_pixels` and `max_pixels`. It places
no constraint on the video resolution. The two 32s meet in the same graph
because a keyframe or reference image is consumed on both paths: VAE-encoded
into conditioning tokens that sit in the DiT sequence, and turned into a
vision block for the `<Picture 1>:` part of the prompt. Only the first is
tied to the video's resolution.

## Two things that surprise people

The short edge is not always 768. It is 768 only while the area cap does not
bind, which holds from roughly 3:4 through 7:4. Outside that the cap takes
over and the short edge shrinks: 21:9 returns 1536x672, and 9:21 returns
672x1536. If you expected 768 on the short edge at 21:9, the area cap is why
you did not get it.

1.00x is not the cost ceiling. Rounding each axis to 32 can land a resolution
above the 16:9 token count. Ask for 23:7 and you get 1856x576, which is 1044
video tokens per frame against 16:9's 1008, so 1.073x the attention for no
extra pixels. The worst case in the whole set is 7.3% over 16:9. Nearby
ratios do not do this: 29:9 returns 1824x576 at 1.036x, and 4:1 returns
2016x512 at exactly 1.00x. Stay on the fourteen above unless you have a
reason.

## The length axis

Frame counts round up to `17n + 5`, which comes from the video VAE's
temporal chunking, not from the DiT.

| Ask for | You get | Duration | Latent frames |
|---|---|---|---|
| 200 | 209 | 8.708s | 62 |
| 300 | 311 | 12.958s | 92 |
| 345 | 345 | 14.375s | 102 |
| 346 | 362 | 15.083s | 107 |

345 is the ceiling, not 362. The reference generates 5 to 15 seconds at 24
fps and applies that limit after the rounding, so 362 frames at 15.083s is
refused while 345 at 14.375s passes. There is no on-grid count at exactly
15.0 seconds. Ask for 346 and you get 362, which is why the check has to run
on the rounded number rather than the requested one.

345 is the frame-count ceiling and not the sequence-length ceiling. The
chaining packs pin context from a previous clip, which adds conditioning
tokens, so a chained shot runs longer in sequence length than the same clip
alone at the same frame count.

## The int32 threshold on the length axis

H3 builds q, k and v as three views of one fused buffer, so the stride
between them is `3 x heads x head_dim` = 3 x 56 x 128 = 21504. Signed int32
offsets cross at `2^31 / 21504` = 99,864 tokens.

| Frames | Sequence length | |
|---|---|---|
| 124 | 41,822 | |
| 260 | 82,594 | |
| 311 | 97,884 | under |
| 328 | 102,982 | over |
| 345 | 108,078 | over, and this is the shipped default |

The crossing sits between 311 and 328 frames at 1344x768, so the workflows
in this repo ship past it. That is safe here only because `build_kernel()`
refuses any sageattention without `sageattn_consume`, which means every user
of this node is on the fork carrying the int64 specialization.

KJNodes' `MiniMaxH3TokenCounter` computes its warning from the contiguous
stride, `heads x head_dim` = 7168, and so places the threshold at 299,593
tokens. At our lengths it stays silent while the fused layout is already
past its own crossing. Silence from that warning is not clearance.

## Where these numbers live

`adapt_canvas`, the 768, the cap and the 32 are in
`comfy_extras/nodes_minimax_h3.py`. The 2x2 patchify is `patchify_video` in
`comfy/ldm/minimax/model.py`. The sequence lengths above come from that
file's `PackedLayout`. The aspect and duration limits this repo enforces are
in `h3_rules.py`, which cites the diffusers reference for each.
