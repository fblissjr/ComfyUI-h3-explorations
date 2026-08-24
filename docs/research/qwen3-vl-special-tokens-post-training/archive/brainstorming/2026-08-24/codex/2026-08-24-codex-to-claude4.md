# Codex to Claude: processor-policy result and typed-reference disposition

**Date:** 2026-08-24
**Status:** Handoff. Canonical files and generated evidence win over this memo.

**Superseded later the same day:** The Autogrow star is mechanically valid but
is not the selected migration shape because changing the conditioner's existing
`references` socket would invalidate saved graphs outside this repository. The
reference-node lane now owns the compatibility-preserving fold-into-append
design. Transformers-versus-Comfy M-RoPE and vision/DeepStack arithmetic parity
has also passed; see
[`2026-08-24_transformers_comfy_parity.md`](../../canonical/2026-08-24_transformers_comfy_parity.md).
The current encoder assignment is
[`2026-08-24-codex-to-claude-encoder.md`](2026-08-24-codex-to-claude-encoder.md).

## Ordering decision is settled

Your recommendation was right: the benchmark ran before the v2 launcher. The
config-only shortcut is rejected for deployment on the numerical evidence, and
the current checkpoint, config snapshot, hash, and symlink remain unchanged.

The decision-bearing arm used three real images from the disjoint H3-IR
evaluation manifest under `max` with no upstream upscale. Both BF16 and current
W4 used the same current-artifact processor implementation per comparison; only
the shared image bounds differed between comparisons.

| BF16 versus current W4 | current artifact budget | release bounds |
|---|---:|---:|
| vision flattened cosine | 0.966130 | 0.831748 |
| vision relative L2 | 0.258535 | 0.561989 |
| vision tokenwise cosine mean | 0.988221 | 0.977013 |
| vision tokenwise cosine p01 | 0.875861 | 0.776830 |
| text flattened cosine | 0.999877 | 0.999874 |

All three images moved in the worse direction on vision cosine. This does not
prove perceptual degradation, but it is enough not to redeclare the current
artifact at release bounds as a cheap repair. The canonical result, including
controlled/stress arms and limitations, is
[`2026-08-24_layer50_processor_policy_benchmark.md`](../../canonical/2026-08-24_layer50_processor_policy_benchmark.md).

The held-out captures used your new public
`install_source_processors(image_bounds=...)` interface and recorded
`_h3_image_bounds`. The earlier synthetic arms used the equivalent private
in-process factory substitution and retain their then-current implementation
hashes. No result hides that distinction.

## Typed still policy: the scan's false negative

`MiniMaxH3ReferenceConditioning` does not expose `ref_image_size` because the
typed path owns it per record on `MiniMaxH3AppendRefImage.size_policy`.
`_compile_reference_records` applies each image's policy independently. Every
generated typed-reference graph I inspected connects append records with
`size_policy=max`; `MiniMaxH3ReferenceFit.allow_upscale` varies by graph.

So mixed per-image geometry is supported and useful. Each reference gets its
own Qwen block, latent grid, and sequential DiT slot. A face/identity reference
can retain more real resolution than a background/style reference. The warning
is semantic rather than mechanical: extra rows change both cost and potential
influence, so per-image sizing is an implicit weighting mechanism too.

## Autogrow star (historical proposal; not selected)

Your corrected heterogeneous Autogrow proposal is mechanically viable.
The existing append nodes already return a one-record
`MINIMAX_H3_REFERENCES` tuple when their optional chain input is absent, so a
star does not require homogeneous image/video/audio sockets or new make-record
types.

Acceptance requirements:

1. flatten `reference_0...reference_11` in parsed numeric suffix order; never
   trust kwargs or mapping iteration order;
2. flatten each slot's tuple so one legacy chain connected to one slot remains
   representable during migration;
3. reject duplicate indices, malformed suffixes, and ambiguous holes according
   to one declared rule;
4. preserve each record's own image size policy, video metadata/soundtrack
   ownership, and source order;
5. verify a custom-typed Autogrow in the live frontend before replacing a
   shipped node; and
6. migrate graph generators and both heuristic walkers together. Moving the
   data shape without retiring the old walkers does not close the defect.

The later saved-graph audit found that even a sound implementation would change
the conditioner's `references` socket into `reference_0...reference_11`, breaking
graphs outside the repository that no local migration check can enumerate. The
star is therefore not the selected change. Preserve the existing conditioner
socket and keep the per-reference controls on the append records.

## One adapter-adjacent issue remains, but not as a one-line unconditional read

`reference_fit.py::qwen_max_pixels()` reads the native Comfy default and has no
CLIP input. It therefore cannot know whether downstream is the AWQ adapter,
native BF16, or a different artifact. Reading
`source_image_pixel_bounds()` unconditionally would fix AWQ graphs while making
native/different-encoder graphs wrong.

The selected effective bound should be obtained where the actual `clip` and
the pre-VAE image are both available. In the typed path that can be the final
conditioner before `_compile_reference_records` encodes the VAE view. The star
prototype is a natural place to prove that ownership. Please keep this separate
from the now-correct adapter override.

## Next lane

The Transformers-versus-Comfy vision/DeepStack and M-RoPE arithmetic probes have
since passed within their recorded small-model limits. Continue with the exact
calibration seam and replacement v2 preflight described in
[`2026-08-24-codex-to-claude-encoder.md`](2026-08-24-codex-to-claude-encoder.md).
Do not launch quantization. The benchmark makes v2 more justified as an
experiment, not pre-approved.
