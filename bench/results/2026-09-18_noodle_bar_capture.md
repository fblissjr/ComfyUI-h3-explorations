# The noodle bar, captured: a capture metric that sees the morph, 2026-09-18

> **2026-09-18, read first:** the noodle bar prompt is written for 107 frames and post office for 141; both were rendered here at 345, past the end of their scripts, where the model improvises. What this record measured stands; what it is evidence of changed. See `2026-09-18_off_length_prompts.md`.

Model: MiniMax H3, int8 convrot checkpoint, base 16-step text to video with
audio, trained canvas, full length. Card: RTX 4090, alone. Kitchen
`0.2.35+sol.8176242`.

## The capture

*Recycled 2026-09-18 at the owner's word, once the render was known to be over-length: the tensors are gone, the manifest and a deletion record remain under the capture root, and the repo keeps `2026-09-18_capture_manifest_noodle_bar_over_length.json` and `2026-09-18_capture_inventory_noodle_bar_over_length.json`. The numbers below cannot be re-derived; they stand as recorded.*

Set `2026-09-18_noodle_bar_sage_chain_plain` under the capture root (manifest
and retention file beside the tensors; kept to 2026-10-31). Blocks 0, 24, 45,
48, 49 at steps 4, 8, 12, 15, post-RoPE q/k/v, plain token order. Blocks 45 and
48 are two of the three loud-K blocks and had never been captured.

It is the first capture of a sample with a KNOWN visible artifact. The graph
is the one embedded in `Video/sol_options_noodlebar/sage_chain_s730451892_*`
(2026-09-17, sage chain: sage `fp8++ balanced` dense plus Sol with
`qk_balance`), which the owner judged to morph and duplicate a figure around
four seconds. The armed render is bit-identical to that clip in video and
audio (`ffmpeg -lavfi psnr` infinite, audio md5 equal), so these tensors are
that sample. The capture server ran with the memory compiler off, which is
numerically inert (`2026-09-17_sol_options_noodlebar_batch.md`).

## 1. Token ordering with the CUDA kernel, two new scenes

`bench/sweep_sol_orderings_on_capture.py`, same method as
`2026-09-17_sol_orderings.md`. Data: `2026-09-18_sol_orderings_noodle_bar.json`,
`2026-09-18_sol_orderings_courtroom.json`.

- **Noodle bar.** `3d` is below plain order on blocks 0, 24, 45 and 48 at
  every tau. On 45 and 48 the error roughly halves at equal density, the
  largest reorder gain measured on any block so far. On block 49 it is lower
  at the step-12 cell and mixed at the step-4 cell.
- **Courtroom.** `3d` is far below plain order on blocks 24 and 0. On block 49
  it is slightly better at tau 0.8 and below, and WORSE at tau 1.0 and above,
  at both captured steps. The float reference says the same
  (`2026-09-18_sol_block_size.md`, finding 4). So "the reorder wins every
  cell" (2026-09-17, one scene) does not survive a second scene on the last
  block; it survives everywhere else measured.

## 2. Where the error sits

`bench/map_sol_error_on_capture.py`: per-token error of one Sol call against
exact attention, CUDA kernel, tau 1.0, head prefix of 8, by latent frame and
inside a region. The region was read off the clip BEFORE the numbers were
looked at, but by the person running the analysis, not blind: frames at 3.5 to
5.0 seconds (`internal/` holds the contact sheets) put the morphing figures in
the bottom third of the picture, right of centre, which is token-grid latent
frames 21 to 31, rows 16 to 24, columns 16 to 42. Data:
`2026-09-18_sol_error_map_noodle_bar.json`.

Mean per-token error inside the region over the mean of the whole video:

| cell | plain order | `3d` |
|---|---|---|
| b45_s8 | 1.44x | 1.06x |
| b48_s8 | 1.43x | 0.99x |
| b49_s8 | 1.52x | 0.89x |
| b48_s12 | 1.44x | 1.07x |
| b24_s8 | 0.89x | 0.96x |
| b24_s12 | 0.92x | 0.94x |
| b0_s4 | 1.08x | 1.04x |

Controls, on b48_s8 and b49_s8 under plain order
(`2026-09-18_sol_error_map_noodle_bar_control_*.json`):

| region | b48_s8 | b49_s8 |
|---|---|---|
| the morph box at the morph's time | 1.43x | 1.52x |
| the static neon sign, same latent frames | 0.79x | 0.82x |
| the same picture area, latent frames 60 to 70 | 0.85x | 1.11x |
| the same picture area, latent frames 2 to 12 | 1.01x | 1.16x |

## What it says

1. **On the deep blocks, plain order concentrates Sol's error where and when
   the morph happens, and the reorder removes the concentration.** The region
   carries about one and a half times the video's mean error on blocks 45, 48
   and 49 under plain order and about the mean under `3d`. In absolute terms
   the error inside the region on block 48 falls to about a third.
2. **It is specific.** The same picture area earlier and later in the clip is
   near the mean, and the static sign during the morph is below it. It is not
   "the bottom of the frame is always harder".
3. **It is a deep-block effect.** Blocks 0 and 24 show no concentration under
   either ordering, although the reorder lowers their whole-video error too.
4. The latent frames with the highest plain-order error on blocks 48 and 49
   include 32 to 37, which is about 5.2 to 6.0 seconds; the owner noted a
   figure disappearing "between 5-6s" on another plain-order arm of this
   scene. Suggestive, not tested.
5. This is the first time a capture metric has agreed with the eye about THIS
   artifact: whole-call error never separated the arms the owner separated.
   The instrument that does is local error on deep blocks.

## What it does not say

One scene, one seed, one region chosen by eye by the analyst. Error of one
attention call against exact attention on the same inputs, not the error a
trajectory accumulates. It does not show that this error CAUSES the morph;
it shows the two coincide in space and time and that the lever which removes
one removes the other. A head prefix of 8 of 56.

## Next

The same map on the panel scenes once the owner's verdicts are in: if scenes
the owner calls clean show no concentrated region and scenes they call morphed
do, the local map becomes a screening instrument for a default, which a
whole-call number never was.
