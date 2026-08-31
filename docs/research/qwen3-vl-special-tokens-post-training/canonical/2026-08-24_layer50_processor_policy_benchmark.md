# Layer-50 BF16 versus current-W4 processor-policy benchmark

**Status:** Measured numerical result; not a render-quality finding
**Observation date:** 2026-08-24
**Hardware:** RTX 4090
**Decision:** Do not widen the deployed current-W4 artifact's default image
budget from its calibration-time snapshot on this evidence.

## Question and isolation boundary

The benchmark asked whether the current W4 weights could safely be served with
the release-declared Qwen image bounds before paying for a reference-faithful
requantization.

Each result compares BF16 against the current W4 at the raw state after language
layer index 49. Both models receive identical native-H3 presentation, decoded
media, normalized visual patches, grids, token IDs, token tags, attention masks,
M-RoPE position IDs, and pre-language-layer embeddings. The comparator refuses
metrics if any of those fields differ.

The two policies are separate weight-only comparisons:

- **current:** the artifact's Qwen2VL image processor at its declared
  200,704--301,056-pixel budget; and
- **release-bounds:** the same artifact processor implementation, including its
  uint8 boundary and bicubic path, with only `size` changed in memory to the
  release declaration of 65,536--16,777,216 pixels.

The second arm uses
`h3_awq_encoder.py::install_source_processors(image_bounds=...)`. It does not
edit the config snapshot, update its hash, repoint the model symlink, or mix in
Comfy's separate float/bilinear image path.

Producers:

- [`capture_h3_encoder_states.py`](../../../../bench/capture_h3_encoder_states.py)
- [`compare_h3_encoder_captures.py`](../../../../bench/compare_h3_encoder_captures.py)

The comparison JSON files contain model/config/code hashes, the exact effective
processor record, source media hashes, grids, sequence lengths, capture
provenance, alignment hashes, and float64-accumulated metrics.

## Decision-bearing held-out result

**MEASURED.** Three real images were selected from the disjoint H3-IR evaluation
manifest. Their decoded sizes were 2048x1152, 1408x2112, and 2752x1536. All
dimensions were already divisible by 32. No image was upscaled; this is the
proposed `max`/no-upscale serving population, not the 2048-short-edge vendor
upscale convention. A controlled single-reference prompt was used so image
content and geometry, rather than differing prose, determined the visual arm.

| BF16 versus current W4 | current artifact budget | release bounds |
|---|---:|---:|
| total sequence rows | 933 | 9,447 |
| all flattened cosine | 0.973692 | 0.852829 |
| all relative L2 | 0.228807 | 0.527077 |
| text flattened cosine | 0.999877 | 0.999874 |
| vision flattened cosine | 0.966130 | 0.831748 |
| vision relative L2 | 0.258535 | 0.561989 |
| vision tokenwise cosine mean | 0.988221 | 0.977013 |
| vision tokenwise cosine p01 | 0.875861 | 0.776830 |

The direction was consistent per image:

| source SHA-256 prefix | decoded size | current vision cosine | release-bounds vision cosine |
|---|---:|---:|---:|
| `0e01357e2731` | 2048x1152 | 0.937415 | 0.787134 |
| `57291cf2b7c1` | 1408x2112 | 0.999529 | 0.898182 |
| `89184cd6cd93` | 2752x1536 | 0.977312 | 0.812418 |

Evidence:

- [`current real-image comparison`](../../../../bench/results/archive/v2_encoder/2026-08-24_layer50_bf16_vs_w4_current_real3.json)
- [`release-bounds real-image comparison`](../../../../bench/results/archive/v2_encoder/2026-08-24_layer50_bf16_vs_w4_release_bounds_real3.json)

**INTERPRETATION.** More source detail reaches Qwen under the wider budget, but
the current W4 language weights track BF16 substantially less closely on that
expanded visual distribution. This is consistent with a calibration-distribution
gap. It does not establish which individual AWQ scale or activation outlier
caused the gap.

## Controlled and stress results

**MEASURED.** A 1920x1088 deterministic reference, representing a 1080p source
under `max` with upscaling off, moved from 264 to 2,040 merged visual tokens.
Vision cosine changed from 0.999298 to 0.926524 and vision relative L2 from
0.109391 to 0.397593. The direction agrees with all three real images.

- [`current 1080p controlled comparison`](../../../../bench/results/archive/v2_encoder/2026-08-24_layer50_bf16_vs_w4_current_1080p_max.json)
- [`release-bounds 1080p controlled comparison`](../../../../bench/results/archive/v2_encoder/2026-08-24_layer50_bf16_vs_w4_release_bounds_1080p_max.json)

**MEASURED.** A separate vendor-upscale stress fixture used a 3648x2048
reference. It moved from 264 to 7,296 merged visual tokens. Its flattened
metrics were mixed: all cosine was 0.943148 versus 0.942123 and vision cosine
was 0.921369 versus 0.933595, while the vision tokenwise cosine mean fell from
0.979722 to 0.962720 and its low tail worsened. This is why one synthetic
fixture was not used to decide the serving policy.

- [`current 2048-upscale stress comparison`](../../../../bench/results/archive/v2_encoder/2026-08-24_layer50_bf16_vs_w4_current_single_ref.json)
- [`release-bounds 2048-upscale stress comparison`](../../../../bench/results/archive/v2_encoder/2026-08-24_layer50_bf16_vs_w4_release_bounds_single_ref.json)

**MEASURED.** Under the current artifact policy, the original controlled
six-family substrate produced aggregate cosine 0.989304. Text rows were
0.999774; vision rows were 0.985896. The ordered multi-image Ref2VA fixture was
the weakest family at 0.943023 overall and 0.921021 on vision rows. This is the
first numerical BF16-versus-current-W4 layer-50 baseline; it is not a corpus
estimate.

- [`current-policy controlled-family comparison`](../../../../bench/results/archive/v2_encoder/2026-08-24_layer50_bf16_vs_w4_current_controlled.json)

## Feasibility observation

**MEASURED.** W4 completed a two-reference release-bounds fixture carrying
7,296 plus 4,096 merged visual tokens. BF16 under Comfy's default dynamic-offload
reserve ran out of VRAM while entering that fixture. A single 7,296-token
reference completed after both compared arms were run with an additional 4 GiB
dynamic-offload reserve, recorded in their manifests. The two-reference BF16
case was not retried with that reserve, so this is a bounded feasibility
observation rather than a maximum supported sequence claim.

## Decision and limitations

**DECISION.** Keep the deployed artifact's declared processor snapshot and hash
unchanged. The in-memory override remains a measurement interface, not a new
default. The config-only shortcut is not accepted as the reference-fidelity
repair.

**INFERENCE.** The consistent held-out direction strengthens the case for a v2
quant calibrated on the intended role-aware reference geometry. It does not
authorize that quantization run; the replacement calibration preflight still
has to pass. The separate Transformers-versus-Comfy arithmetic-parity gate has
passed within the limits recorded in
[`2026-08-24_transformers_comfy_parity.md`](2026-08-24_transformers_comfy_parity.md).

Limits:

- only three real images were measured, with one controlled prompt each;
- each processor policy has its own matching BF16 reference because its visual
  sequence is different. These metrics rank W4 fidelity within a geometry; they
  do not rank the information content or H3 quality of the two geometries
  against each other. The result is evidence that the current W4 does not carry
  safely to release bounds, not evidence that release-sized references are bad
  for H3;
- the real-image arm is single-reference, while the current-policy controlled
  substrate separately covers ordered multi-image presentation;
- no DiT render was produced, so no perceptual, identity, motion, or lip-sync
  claim follows;
- flattened metrics can be dominated by high-magnitude coordinates, which is
  why relative L2 and tokenwise distributions are reported beside them; and
- no numerical threshold was chosen after seeing the result and relabeled as a
  release criterion.

The next relevant encoder comparison is BF16 versus a correctly calibrated v2
under the same intended policy. If a config-only serving experiment is ever
reconsidered, it first needs a preregistered distributional render evaluation;
this record is evidence against making that experiment the deployed default.
