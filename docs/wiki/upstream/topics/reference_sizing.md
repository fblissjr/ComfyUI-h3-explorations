# Reference stills: how each implementation sizes them (H3 video ref2va)

last updated: 2026-11-15 (page created from the 2026-11-15 sweep and the owner's question that day)

**Scope: H3 video ref2va reference stills**, the `<Picture N>` images that ride beside the target rows on every
step. It does not cover MiniMax-H3-Image edit references (their own bounds:
[`../../image/h3_image.md`](../../image/h3_image.md)), reference videos, or keyframes.

**Status: open, the owner's call. Nothing was changed (owner, 2026-11-15: "Dont make any changes but please
document all this").** The case for changing our default is below. So is what the change would involve, so it can
be picked up without re-deriving any of it.

---

## What each implementation does today

The three modes sglang names, used below:
- `diffusers`: scale the short edge to 2048, enlarging small stills.
- `max`: shrink only a still whose short edge is past 2048.
- `match`: shrink to the canvas area.

Every mode keeps the aspect ratio and rounds each axis to 32.

| implementation | rank | default for a small still | where |
|---|---|---|---|
| sglang | 1 | **not enlarged** (`max`) since `92a6910ef2` (#47907, 2026-11-05); `diffusers` before that. FastH3 Omni keeps `diffusers` | `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/constants.py::MINIMAX_H3_DEFAULT_REFERENCE_RESIZE_MODE`, `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py::minimax_h3_resolve_reference_image_shape` |
| vllm-omni | 2 | **not enlarged**: keeps the still's own size rounded to 32, shrinking only to a canvas-area cap on timeline-guided requests | `coderef/vllm-omni/vllm_omni/model_executor/models/minimax_h3/preprocessing.py::resolve_minimax_h3_reference_image_shape` |
| diffusers | 3 | **enlarged** to a 2048 short edge ("2048 for the released checkpoint") | `coderef/diffusers/src/diffusers/modular_pipelines/minimax_h3/before_encoder.py` (`reference_image_short_edge`), `.../minimax_h3/references.py` |
| ComfyUI core | trusted least | **not enlarged** in either `ref_image_size` mode (`match` default, `max`) | `comfy_extras/nodes_minimax_h3.py`, the reference-to-video node's `ref_image_size` and `ref_image` tooltips |
| DiffSynth | periphery | **enlarged**: `scale = short_edge / min(w, h)`, unclamped, short edge 2048 | `coderef/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py::_resolve_reference_image_shape` |
| LightX2V | periphery | **enlarged** (`diffusers` mode) | `coderef/LightX2V/lightx2v/models/networks/minimax_h3/packing_ref2av.py::DEFAULT_REFERENCE_IMAGE_RESIZE_MODE` |
| **this pack** | | **enlarged**: `MiniMaxH3AppendRefImage` `size_policy=max`, `dit_short_edge=2048`, `allow_upscale=True` | `reference_conditioning.py` (the `allow_upscale` input); the geometry is `reference_geometry.py` |

By the owner's ranking ([`../index.md`](../index.md), "When upstreams disagree"), the top two engines and core now
agree on not enlarging, and only the third engine and the periphery still enlarge.

## How we got here

- **2026-08-28.** Our video default turned upscaling **off**.
- **2026-09-13.** The owner set `MiniMaxH3AppendRefImage` to **vendor parity**, `allow_upscale=True`, "which is what
  sglang, diffusers and DiffSynth do" ([`../../decisions/2026-09-11_to_2026-09-15.md`](../../decisions/2026-09-11_to_2026-09-15.md)).
  **That parity is the only reason the default is `True`.**
- **2026-08-16, image path only.** Single-frame graphs moved to `ref_upscale=False` after a same-seed A/B on two
  subjects found the same identity at a fraction of the wall clock (`workflows/h3_config.py`, the comment block
  above `IMAGE_EDIT_BUDGET`). Video kept `True`: "identity has to survive motion as well as a still".
- **2026-11-05.** sglang moved its native default to `max` (#47907).
  - The PR is a speed change from an sglang contributor, not a MiniMax preprocessing decision. Its motive was an
    sglang ref2va workload running slower than vllm-omni.
  - It deleted the comment calling `diffusers` "the original Diffusers and SGLang reference-image behavior", which
    said it "should remain the compatibility default unless a checkpoint specifies another preprocessing policy".
  - Its quality table (VBench-style proxies, small-reference cases; in the PR body) has `max` slightly below
    `diffusers` on nearly every row, reference-subject similarity included. Its author calls the review informal and
    says it is not a quality-equivalence study.
- **2026-11-15.** The sweep found the change ([`../sweeps/2026-11-15.md`](../sweeps/2026-11-15.md), item 1).

## The case for turning `allow_upscale` off on video

- **The parity it was set for no longer exists.** The top two engines and core all leave small stills at their own
  size.
- **The cost is paid on every step.** Enlarging one 1024² still added tokens to both the reference rows and the text
  segment, since Qwen3-VL reads the resized still too. The cost scales with the number of references (measured
  2026-08-13 and 2026-08-15, [`../../core/h3_references.md`](../../core/h3_references.md)). It also brings long or
  multi-reference renders closer to the `int32` QKV threshold.
- **The evidence on quality is small and points one way.** Our image-path A/B found no identity loss. sglang's table
  shows small drops it does not call significant.

## The case for keeping it

- **No video A/B with motion has been run here.** That was the stated reason video kept `True` when the image path
  went off.
- **The release's library ports enlarge.** diffusers says 2048 "for the released checkpoint". If training saw
  references at a 2048 short edge, enlarging matches training. That is reasoned; we have no training record.
- **Masked-lane head crops are the case that matters most.** They are small, so they get the largest enlargement,
  and identity is the whole point of that lane.

## What would settle it

`workflows/h3_probe_reference_upscale.json` exists for exactly this question: the same references with upscaling
off (`workflows/build_workflows.py`, the `r2v-upscale` arm). The cross-scene ablation is `bench/refview2_arms.json`.
A stacked pair at one seed on a masked-lane head-crop load would decide it, judged by the owner on playback.

## What the change would involve (not done)

- `reference_conditioning.py`: `allow_upscale` default `True` to `False`, and the comment that says `True` is
  "sglang's default `diffusers` resize mode", which is stale since 2026-11-05.
- `workflows/h3_config.py` / `workflows/build_workflows.py`: the video reference graphs' `ref_upscale`, then a
  rebuild of the generated graphs.
- Check fixtures that expect `True` (`bench/check_typed_reference_consumers.py`).
- `../../core/h3_references.md`: the defaults paragraph.
- A node-default change, so it is built in a worktree outside `custom_nodes`, landed when no renders are in flight,
  and picked up after the owner restarts ComfyUI.
- The FastH3 Omni and Pika notes: sglang's Pika inherits `max`, while Pika's own `buckets.yaml` enlarges
  (`../../references.md`, "Pika H3 Flex"). Pika graphs here use the same append node.
