# Core's H3 VAE tile blending before and after #16436, 2026-09-25

**Result: the change touches only the overlap bands, and at these canvases
it neither helps nor hurts measurably.** Comfy-Org/ComfyUI#16436 (`fc584aaa`,
reached this checkout in the 2026-09-22 pull) changed
`comfy/ldm/minimax/vae.py::MiniMaxH3VideoVAE.tiled_decode`. Tiles now blend
against their neighbours as already blended, not against their raw decode. On
three real frames the old and new decodes are identical outside the overlap
bands, differ inside them, and sit equally close to the source.

**What it informs.** This is rebuild-record policy, not an owner decision.
A render made before 2026-09-22 is not bit-comparable with one made after it.
The difference is confined to the overlap bands and does not change the
source fidelity by a measurable amount, so no quality decision rides on it.

## Method

- Frames: three first-frame PNGs at 1344x768 from shipped-graph renders
  (`h3_candidate_t2v_pdd8_baked_0000{2,4,7}.png` on the output share).
- Each frame is encoded by core (`comfy.sd.VAE.encode`, with
  `h3_config.MODELS["video_vae"]`) to a one-frame latent and un-normalised as
  `decode` does. It is then decoded twice on the GPU by the same
  `MiniMaxH3VideoVAE` instance:
  - by the current `tiled_decode`;
  - by the pre-change `tiled_decode`, taken from `git show
    fc584aaa~1:comfy/ldm/minimax/vae.py` and bound to the instance.

  Core's own `MiniMaxH3VideoVAE.decode` on the same latent returned the same
  PSNR against the source as the new path, which confirms the harness
  reproduces core's decode.
- Overlap bands come from the instance's own `split_tiles` at that canvas.
- "Seam step" is the mean absolute pixel step across the centre line of each
  band. The control is the same measure on interior lines outside every
  band, and on the source image at the same lines.

## Result

| frame | max diff outside bands | max diff in bands | mean diff in bands | PSNR vs source, old / new | seam row step, old / new | seam col step, old / new | interior row step (control) |
|---|---|---|---|---|---|---|---|
| 00002 | 0 | 0.408 | 1.34e-3 | 18.140 / 18.134 | 0.01127 / 0.01123 | 0.01052 / 0.01082 | 0.00946 |
| 00004 | 0 | 0.198 | 9.78e-4 | 18.697 / 18.690 | 0.01370 / 0.01375 | 0.01218 / 0.01221 | 0.01007 |
| 00007 | 0 | 0.189 | 1.31e-3 | 19.201 / 19.192 | 0.00786 / 0.00775 | 0.00845 / 0.00856 | 0.00841 |

Pixels are in [0, 1]. At 1344x768 the bands cover just over half the canvas
(`split_tiles` gives 4 row tiles and 7 column tiles). The largest per-pixel
changes are where a horizontal and a vertical band cross, which is what the
change targets. The seam steps move both ways by amounts well inside the
spread between frames.

## What this does not establish

- One frame per clip. Temporal chunking also runs this function per chunk,
  and nothing here tests motion across a seam.
- **The round-trip PSNR is low for a VAE, and that is not explained here.**
  Core's own decode of core's own one-frame encode gives the same figure, so
  it is a property of the one-frame still path in a standalone process, not of
  this harness. Longer clips could not be tried standalone: they ran out of
  memory in `comfy.sd.VAE.decode` and in the kitchen group-norm kernel outside
  the server's memory management. If keyframe or reference-still fidelity is
  ever in question, the instrument is a round trip through the server's own
  nodes.

## Reproduce

    python bench/compare_vae_tiled_decode.py FRAME.png [...] --rev fc584aaa~1

It prints the rows above per frame. No server may hold the card while it runs.
