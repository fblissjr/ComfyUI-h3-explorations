# Reference stills: how each implementation sizes them (H3 video ref2va)

last updated: 2026-11-22 (Pika ref2va graphs moved to `tiered` on the owner's go, "2026-11-22: Pika" below; sglang's Pika enlarges small stills again, citing Pika's release); 2026-11-15 (owner decided to flip `MiniMaxH3AppendRefImage.allow_upscale` default to `False` following sglang #47907)

**Scope: H3 video ref2va reference stills**, the `<Picture N>` images that ride beside the target rows on every
step. It does not cover MiniMax-H3-Image edit references (their own bounds:
[`../../image/h3_image.md`](../../image/h3_image.md)), reference videos, or keyframes.

**Status: decided 2026-11-15 (flipped to `allow_upscale=False`).** After reviewing sglang PR #47907 (`92a6910ef2`)
alongside vllm-omni and ComfyUI core, the owner aligned `MiniMaxH3AppendRefImage` and default video `ref2va` graphs
on `allow_upscale=False`, while keeping `ref_upscale=True` on byte-pinned instruments (`_savelat`, `_x0`, `_stamped`),
`h3_probe_refview2_*`, `h3_probe_r2v_pika_motion_transfer_6step*`, and masked-song head-crop graphs, and inverting
`workflows/h3_probe_reference_upscale_api.json` back to `True` as the non-default upscale probe.

**Pika ref2va: decided 2026-11-22 (moved to `tiered`).** The shipped Pika H3 Flex ref2va video graphs size stills
by Pika's release rule (`h3_config.PIKA_REF_SIZE_POLICY`), so a small still is enlarged. Native H3 graphs and the
node default are unchanged. See "2026-11-22: Pika" below.

---

## What each implementation does today

The three modes sglang names, used below:
- `diffusers`: scale the short edge to 2048, enlarging small stills.
- `max`: shrink only a still whose short edge is past 2048.
- `match`: shrink to the canvas area.

Every mode keeps the aspect ratio and rounds each axis to 32.

| implementation | rank | default for a small still | where |
|---|---|---|---|
| sglang | 1 | **not enlarged** (`max`) since `92a6910ef2` (#47907, 2026-11-05); `diffusers` before that. FastH3 Omni keeps `diffusers`. **Pika enlarges** (`diffusers`) since `44a391941a` (#51935, 2026-11-20) | `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/constants.py::MINIMAX_H3_DEFAULT_REFERENCE_RESIZE_MODE`, `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py::minimax_h3_resolve_reference_image_shape` |
| vllm-omni | 2 | **not enlarged**: keeps the still's own size rounded to 32, shrinking only to a canvas-area cap on timeline-guided requests | `coderef/vllm-omni/vllm_omni/model_executor/models/minimax_h3/preprocessing.py::resolve_minimax_h3_reference_image_shape` |
| diffusers | 3 | **enlarged** to a 2048 short edge ("2048 for the released checkpoint") | `coderef/diffusers/src/diffusers/modular_pipelines/minimax_h3/before_encoder.py` (`reference_image_short_edge`), `.../minimax_h3/references.py` |
| ComfyUI core | trusted least | **not enlarged** in either `ref_image_size` mode (`match` default, `max`) | `comfy_extras/nodes_minimax_h3.py`, the reference-to-video node's `ref_image_size` and `ref_image` tooltips |
| DiffSynth | periphery | **enlarged**: `scale = short_edge / min(w, h)`, unclamped, short edge 2048 | `coderef/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py::_resolve_reference_image_shape` |
| LightX2V | periphery | **enlarged** (`diffusers` mode) | `coderef/LightX2V/lightx2v/models/networks/minimax_h3/packing_ref2av.py::DEFAULT_REFERENCE_IMAGE_RESIZE_MODE` |
| **this pack** | | **not enlarged** (`size_policy=max`, `dit_short_edge=2048`, `allow_upscale=False` default since 2026-11-15; `True` kept opt-in and pinned on historical instruments & masked head crops). **Pika ref2va graphs enlarge** (`size_policy=tiered`, `h3_config.PIKA_REF_SIZE_POLICY`) since 2026-11-22 | `reference_conditioning.py` (`allow_upscale` input); `reference_geometry.py` |

By the owner's ranking ([`../index.md`](../index.md), "When upstreams disagree"), the top two engines and core now
agree on not enlarging for native H3, and only the third engine and the periphery still enlarge. *2026-11-22: for
Pika this no longer holds; see "2026-11-22: Pika" below.*

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
- **The cost is paid on every step.** Under our `qwen_view="separate"` (`512`) default (since 2026-10-03), enlarging
  one `1024x1024` still bloated DiT `ref_img` tokens `1,024 -> 4,096` (`+3,072` rows attended every step); under
  `qwen_view="shared"`, it doubled again in Qwen3-VL's text segment (`+3,072` more tokens, measured 2026-08-13 and
  2026-08-15, [`../../core/h3_references.md`](../../core/h3_references.md)). It also brings long or multi-reference
  renders closer to the `int32` QKV threshold.
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

## What the change touched (implemented 2026-11-15)

- [`reference_conditioning.py`](../../../../reference_conditioning.py): `allow_upscale` schema default and bare `"max"` execution fallback flipped `True -> False`, with updated comment/tooltip citing sglang #47907 (`92a6910ef2`).
- [`workflows/build_workflows.py`](../../../../workflows/build_workflows.py) and `workflows/`: `build_api(..., ref_upscale=False)` default; `h3_probe_reference_upscale` inverted to `True`; byte-pinned instruments (`_savelat`, `_x0`, `_stamped`), `h3_probe_refview2_*`, Pika motion-transfer parity arms, and masked-song head-crop templates pinned to `ref_upscale=True`; all API graphs regenerated.
- [`bench/preflight_graph.py`](../../../../bench/preflight_graph.py), [`bench/generate_capture_manifest.py`](../../../../bench/generate_capture_manifest.py), [`bench/check_reference_runtime.py`](../../../../bench/check_reference_runtime.py), [`bench/check_typed_reference_consumers.py`](../../../../bench/check_typed_reference_consumers.py), [`bench/check_widget_deviations.py`](../../../../bench/check_widget_deviations.py): updated omitted/default `allow_upscale` expectations to `False`.
- [`../../core/h3_references.md`](../../core/h3_references.md) & [`../../research/sglang_comparison.md`](../../research/sglang_comparison.md): updated node defaults & Gap 6.
- ComfyUI `:8188` requires a service reload when convenient to pick up the `MiniMaxH3AppendRefImage` default change for UI drags; regenerated `*_api.json` graphs send `"size_policy.allow_upscale"` explicitly on every step.
- The FastH3 Omni and Pika notes: sglang's Pika inherits `max`, while Pika's own `buckets.yaml` enlarges (`../../references.md`, "Pika H3 Flex"). Pika showcase parity graphs here keep `ref_upscale=True`. *Corrected 2026-11-22: sglang's Pika no longer inherits `max` (below).*

## 2026-11-22: Pika

**Decided 2026-11-22: the Pika ref2va graphs moved to `tiered`.** The owner relayed the Gemini lead's review with
its go, and the move followed. The 2026-11-15 decision stands for native H3 and for the node default.

- **What moved.** sglang `44a391941a` (#51935, a `[Fix]` PR, 2026-11-20) sets
  `PikaH3FlexSamplingParams.reference_resize_mode = "diffusers"`, so sglang's Pika enlarges small stills to a 2048
  short edge again. Its comment says the released pipeline "encodes image references at a 2048 px short edge from
  the 768 tier upwards, upscaling smaller ones". The PR says #47907 had carried Pika along without anyone choosing
  it. Native H3 stays `max`.
- **Its motive is fidelity to the distill's trainer,** not speed: the opposite kind of evidence from #47907
  (`../index.md`, "Running a sweep", on reading a moved default with its PR's motive).
- **The rules that bear on it.**
  - A distill is run the way its trainer says (`../../references.md`, "A distill's reference is its trainer's
    contract", 2026-09-26). That page's "Pika H3 Flex" entry lists the tiered sizing as part of Pika's contract.
  - The "case for turning `allow_upscale` off" above rests on the top two engines and core leaving small stills
    alone, which for Pika is now false at rank 1. vllm-omni has no merged Pika path, and core has no Pika-specific
    path.
- **What exists here.** The shipped Pika ref2va graphs send `size_policy=max`, `dit_short_edge=2048`,
  `allow_upscale=False` (node 50 of each `*pika*` ref2va graph). `size_policy="tiered"`
  (`reference_geometry.py::tier_short_edge`) is a port of Pika's `reference_short_edge_for_canvas`, and
  `workflows/build_workflows.py` takes `ref_size_policy`.
- **What moved (2026-11-22).** `h3_config.PIKA_REF_SIZE_POLICY = "tiered"`, passed as `ref_size_policy` by the ten
  Pika ref2va entries in `workflows/build_workflows.py`, eleven graphs with `workflows/daily/h3_preview_pika_ref2va_api.json`.
  One of them is `workflows/h3_guide_video_edit_api.json`, whose name does not say Pika (found by the Gemini lead).
  The single-frame Pika edits keep `match`; the motion-transfer parity arms keep `max` with upscale on, the same
  size from the 768 tier up. `bench/check_distill_settings.py` now fails a Pika ref2va video graph whose stills
  are not enlarged that way.
- **What it costs.** Under the shipped `qwen_view="separate"` only the DiT's reference rows grow: a `1024x1024`
  still goes from 1,024 to 4,096 rows ("What references cost" in [`../../core/h3_references.md`](../../core/h3_references.md)).
  `bench/preflight_graph.py` prices a graph's whole sequence; it reads the input folders from `<comfy>/start.sh`
  and leaves out a still it cannot find. Wall time and peak memory with Pika were not measured.
- **What was not done.** No render compared the two on Pika. A same-seed stacked pair on one Pika ref2va graph
  would show whether the difference is visible.
