# Why keyframes and references are sized differently: the packing rule

**Status:** Source-verified across two implementations
**Observation date:** 2026-08-24
**Scope:** installed ComfyUI `comfy/ldm/minimax/model.py` and
`comfy_extras/nodes_minimax_h3.py`; DiffSynth-Studio
`diffsynth/pipelines/minimax_h3_audio_video.py`. Corroborating comment read in
sglang. Code read, nothing executed.

Until now this record treated the 2048-short-edge reference convention as
something four implementations do without a stated rationale. There is a
structural reason, and it is legible in the packed-sequence layout rather than in
any documented policy.

## The two conditioning kinds are packed differently

**SOURCE.** ComfyUI carries them under two different conditioning keys —
`minimax_keyframes` from `MiniMaxH3ImageToVideo`, `minimax_refs` from
`MiniMaxH3ReferenceToVideo` — and the DiT lays them out differently:

| | keyframe (FL2VA / I2VA) | reference (Ref2VA) |
|---|---|---|
| time position | on the **target timeline**: `cursor + FRAME_RESCALE * resolved_frame_index` | its **own sequential slot**, starting at `text_len`, advancing by `_ref_t_span` per block |
| spatial grid | the **target canvas** grid, shared with the generated rows | its **own** grid, from that reference's own latent height and width |
| span consumed | none of its own; it sits at a frame position that already exists | image 1.0, audio `ref_audio_t`, video `max(ref_audio_t, sum(_video_t_spans))` |
| packing order | after the references, which reserve the span ahead of the targets | between the text rows and the target rows |

**SOURCE.** DiffSynth is the same design, independently written. References:
`cursor, t_cursor = text_len, float(text_len)`, then per image block
`g[sl, 0] = t_cursor` with the grid built from the reference's own latent dims
under an explicit `# Own spatial grid`, then `t_cursor += 1.0`. Video blocks
advance by `max(float(ref_at), self._video_t_span(lt_r))`, matching ComfyUI's
`_ref_t_span` term for term. Keyframes: `cond_t = float(text_len)` for the first
frame and `float(text_len) + temporal_span - self._FRAME_RESCALE` for the last.

**SOURCE.** ComfyUI generalizes the keyframe case: `MiniMaxH3AddGuide` anchors at
any frame index, so `resolved_frame_index` is arbitrary where DiffSynth exposes
only first and last. The rule is the same; ComfyUI's is the superset. sglang's
`canvas.py` records the complementary role from the geometry side: "A request's
first semantic keyframe is its geometry anchor."

## What that explains

**INFERENCE, from the layout above.** A keyframe *is* a frame of the output. It
occupies a position on the target timeline and shares the target spatial grid, so
it must arrive at canvas geometry — it is standing in for generated rows. This is
why `MiniMaxH3ImageToVideo` stretches the first frame to the canvas and
cover-crops the last one.

A reference is *not* on the output timeline. It receives its own rotary slot and
its own spatial grid, so nothing structurally forces it to canvas geometry, and
it is free to carry more spatial detail than the video will ever be generated at.

That is the mechanism behind the role split
[`2026-08-24_still_policy_token_cost.md`](2026-08-24_still_policy_token_cost.md)
records as a serving convention. It upgrades "four implementations do this" to
"the packing makes it possible", which is a stronger footing for the role-aware
policy.

**It does not explain the specific value.** Why references are free to be larger
is now structural. Why 2048 rather than 1536 or 3072 remains a serving
convention, unattested by the release. Do not let the mechanism promote the
constant.

## Consequence for AWQ calibration: smaller than it looks

**SOURCE.** The Qwen presentation does **not structurally distinguish a
keyframe from a reference still**. In `comfy/text_encoders/minimax.py` the FL2VA
path emits `"<Picture i>: "` plus a vision block per image, and the Ref2VA image
path emits the same form under its own counter. `MiniMaxH3ImageToVideo` calls
`clip.tokenize(prompt, images=images)` without passing the resolved frame index.
The DiT packing difference is downstream of layer 50 and the encoder never sees
that index.

The other reference modalities are distinct at Qwen's input:

| conditioning role | Qwen presentation | structural temporal marker |
|---|---|---|
| keyframe / first-last-frame guide | `<Picture i>:` plus vision block | no |
| reference still | `<Picture i>:` plus vision block | no |
| reference video | `<Video k>:` followed by `<T.T seconds>` plus a vision block per temporal pair | yes |
| reference audio | `<Audio j>:` text label | no audio tensor reaches Qwen |

**INFERENCE.** Qwen can learn or use a keyframe's temporal role only through
the accompanying prose, such as “at 0.00 seconds into the target video.” The
DiT still anchors that guide structurally even if the prose omits the timing.
Reference-video timing is different: timestamp markers are part of Qwen's
presentation as well as the downstream reference structure.

So for the calibration lane:

- **No structural token-format distinction between keyframe and reference
  still.** They have the same shape of input to the encoder, which is consistent
  with the measured
  finding that every vision-bearing family shares one traced graph and one key
  set ([`2026-08-24_calibration_input_seam.md`](2026-08-24_calibration_input_seam.md)).
  Their role and temporal prose must nevertheless be explicit manifest fields;
  `<Picture i>` alone cannot identify the stratum.
- **A geometry consequence, and it is the reason Decision 2 is role-aware.**
  A keyframe arrives at canvas geometry, roughly 1,008 merged tokens at
  1344x768. A reference arrives at its own geometry, up to roughly 7,296 merged
  tokens for a 16:9 still at a 2048 short edge. Same presentation, up to a 7x
  difference in visual token count.
- **So the manifest strata are geometry strata, not presentation strata.** A
  calibration population that samples "FL2VA rows" and "Ref2VA still rows" is
  varying token count and spatial detail, not sequence structure. Both must be
  present for the same reason, and the rejected preflight's practice of treating
  every image row alike is wrong on the geometry axis even where its presentation
  would have been right.
- **Temporal prose is part of the keyframe stratum.** A keyframe trace must
  retain its source prompt wording and resolved frame role. Normalizing away
  phrases that name `0.00 seconds`, a last frame, or another target time removes
  the only encoder-visible distinction while leaving the DiT-visible anchor
  intact.
- **Both branches may coexist.** The implementation handles keyframes and
  references independently, computes the target origin after the references'
  reserved span, and then overlays keyframes on that target timeline. A v2
  population should therefore include mixed keyframe-plus-reference requests
  when honest source data supports them rather than assuming the task families
  are mutually exclusive.

## Bounds

- Read from code in two implementations, not executed and not compared against a
  running model. No claim here has been checked by a capture.
- The sglang corroboration is one comment on the geometry-anchor role, not a
  line-by-line reading of its packing.
- Nothing here bears on numerical fidelity, on the benchmark, or on whether the
  positioning is implemented identically across the three stacks. Two
  independent implementations agreeing on a design is not proof that their
  outputs match; that would need a capture on both.
