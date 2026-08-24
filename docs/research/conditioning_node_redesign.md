# Reference node redesign: fold the fit into the append

**Status:** Design, not yet implemented
**Date:** 2026-08-24
**Scope:** `reference_conditioning.py`, `reference_fit.py`,
`workflows/build_workflows.py`, and every graph that wires an image reference —
`workflows/h3_image_ref_plus_text_to_video.json` and its `_api` twin among them.

Today one image reference costs four nodes: `LoadImage` ->
`MiniMaxH3ReferenceFit` -> `MiniMaxH3AppendRefImage` ->
`MiniMaxH3ReferenceConditioning`. This proposes three, by folding the fit into
the append. It is not a tidiness change; the reasons are a measured redundant
resample, a class of static-analysis defect, and four dead code paths.

## What was considered and rejected

**An Autogrow "star"** — `reference_0 .. reference_11` typed
`MINIMAX_H3_REFERENCES` on the conditioner, replacing the chain. Verified
buildable against installed ComfyUI: Autogrow accepts a custom type, slot order
is canonical and independent of JSON key order, gaps are free, and core already
ships the precedent in `nodes_gaussian_splat.py`'s `MergeSplat`. On a greenfield
repo it is the better design.

Rejected here on saved-graph compatibility. The conditioner's single
`references` link would become `reference_0..11`, so **every saved graph wiring
`"references": [id, 0]` on the conditioner stops validating** — including the
owner's graphs outside this repo, which no check can see. That is precisely what
`bench/check_node_ids.py` exists to prevent, and its manifest is committed for
that reason. Folding the fit is the *permitted* change shape (an input appended
at the end); the star is the forbidden one.

It also buys no static-analysis advantage: the win below comes from folding the
fit, which both options do.

## Why fold

**1. The split expresses nothing any graph uses.** Across the 40 `_api` graphs
carrying the typed conditioner: zero vary fit settings between slots, zero vary
`size_policy` between slots, zero append nodes lack a fit upstream, and zero
consumers read `MiniMaxH3ReferenceFit`'s second output. It is 1:1 with every
image reference and never shared.

**2. It costs a redundant resample and a redundant quantization, measured.**
`comfy/utils.py::lanczos` has no identity short-circuit and round-trips through
PIL uint8 unconditionally; `_resize` does not guard either. So the fit resizes,
then `_compile_reference_records` resizes *again* under `size_policy='max'` —
usually a no-op. Measured on this box, CPU, 3-run mean:

| resize | cost |
|---|---:|
| 1024x1024 -> 2048x2048 (fit, upscale arms) | 81.0 ms |
| 2048x2048 -> 2048x2048 (compiler, pure no-op) | 68.8 ms |
| 1024x1024 -> 1024x1024 (both, no-upscale arms) | 11.0 ms |

Every shipped image reference pays a full redundant lanczos pass **and a second
float32 -> uint8 -> float32 quantization**, purely because the sizing decision
and its consumer are two nodes apart. The quantization is the part that is not
merely wall clock. Note it is the second of three: `h3_awq_encoder.py::
_source_image_patches` performs a third on the way into the conditioner.

**3. It puts the sizing decision inside the validated chain.** `allow_upscale`
currently lives on a node the chain model cannot see, with three consequences:
`bench/preflight_graph.py::_reference_media` reads `size_policy` and never reads
`allow_upscale`; `bench/audit_shipped_reference_bounds.py::_trace_to_loader`
reaches fit arguments only by walking "the first linked input" for eight hops,
which its own docstring concedes is the only shape these graphs use; and no
check anywhere asserts `allow_upscale` on any graph. After folding,
`resolve_chain_entries` already returns the append node's dict, so all three
read a field. That is a structural fix rather than a fourth heuristic walker —
what `CLAUDE.md`'s "no new check until a drift instance appears" asks for.

## The change

### `MiniMaxH3AppendRefImage` — inputs appended, never reordered

```
current:  image, size_policy, references
proposed: image, size_policy, references, allow_upscale, short_edge, keep_towers_matched
```

The three new inputs go **after `references`**. `bench/check_node_ids.py`
permits exactly this shape ("an input appended at the end") and rejects
reordering or renaming, because widget values map positionally in every saved
graph. `node_id` is unchanged. Old saved graphs keep working on the defaults.

Defaults must reproduce today's shipped behaviour, which is **not** uniform:
`allow_upscale=False` on 28 graphs and `True` on six, including
`h3_image_ref_plus_text_to_video`. So the node default is `False` and the six
graphs set it explicitly, exactly as they do now.

New outputs, appended:

```
current:  references
proposed: references, image, latent_rows
```

`image` passes through the fitted tensor so a graph can still see and save what
the model will receive — the one thing the split genuinely bought. `latent_rows`
stays a readout even though nothing wires it. Appending outputs is permitted;
reordering is not, since links are integer slots.

### One resample, not two

The folded node performs the aspect gate, the upscale decision, the Qwen-ceiling
clamp and **one** resize, then hands the compiler a tensor already at its target
geometry. `_compile_reference_records` keeps its `size_policy` branch for
correctness but will find nothing to do; add an identity guard there so a no-op
resize costs nothing rather than 68.8 ms and a quantization.

### Four dead paths to delete, not migrate

All four are dead on every shipped graph today, and stay dead under any topology:

| path | why it is dead |
|---|---|
| `lift_downstream_clamp` (input + logic) | `_downstream_ref_image_size` matches only `class_type == "MiniMaxH3ReferenceToVideo"`, which **zero graphs wire**, so it always returns `None` and the warning can never fire |
| `_downstream_ref_image_size` | same |
| `arm_short_edge_override` / `disarm_short_edge_override` | installs on `core.MiniMaxH3ReferenceToVideo.execute`, never executed; and it mutates `core.REF_IMAGE_SHORT_EDGE` while both consumers bound the **value** at import time, so the arm is invisible to them regardless |
| `MiniMaxH3ReferenceVideoFit` | in the node manifest, wired by zero graphs |

Deleting `lift_downstream_clamp` is an input **removal**, which
`check_node_ids.py` treats as a breaking change — so it needs a deliberate
manifest update and should be called out rather than slipped in.

`bench/check_short_edge_override.py` retires with the code it guards. Worth
recording why: it is green today because its fixture supplies synthetic prompts
containing `MiniMaxH3ReferenceToVideo`, so its input pre-satisfies the outcome
and it can never notice that no shipped graph contains that node.

### What this does *not* fix

`reference_fit.py::qwen_max_pixels()` introspects Comfy's native
`process_qwen2vl_images` default of 12,845,056. Under the AWQ adapter that
function is never called — `preprocess_embed` is replaced — and the real ceiling
is 301,056, **42x lower**. So `keep_towers_matched` finds nothing to clamp and
the tower-split warning cannot fire on any shipped graph, precisely where the
split is largest.

**Reading `h3_awq_encoder.source_image_pixel_bounds()` unconditionally is the
wrong fix** (Codex's correction): the fit has no `clip` input and cannot know
whether downstream is the AWQ adapter, native BF16, or another artifact, so an
unconditional read would fix AWQ graphs and break native ones. The effective
bound must be resolved where both the `clip` and the pre-VAE image are in scope
— the conditioner, before `_compile_reference_records` encodes the VAE view.
That is a separate change and should not ride along with this one.

### Two smaller defects worth folding in

- **`MiniMaxH3AppendRefImage` silently drops frames.** `_resize(image[:1], ...)`
  keeps only the first image and `_image_shape` validates `count >= 1` without
  warning on more. Wire a video loader's IMAGE output and frames 2..N vanish
  with no message. `reference_fit.py` already logs
  `"reference carries %d images; using the first"` — the folded node should
  inherit that, since it is the mandatory node on the typed path.
- **Contradictory settings go inert rather than erroring.** A fit above 2048 is
  re-clamped by a downstream `max`, and a `match` can undo a large fit entirely.
  Inside one node these become checkable, and a warning is cheap.

## Migration

1. Fold, with the input and output appends above. Do not touch `node_id`.
2. Update `bench/node_id_manifest.json` deliberately for the appends and for the
   `lift_downstream_clamp` removal.
3. Regenerate. **Both emission paths** —
   `workflows/build_workflows.py:982-989` (API) and `:4105-4118` (UI) — must be updated
   together. They currently agree across all 49 UI/API pairs, and a change
   touching one would not be caught by any check.
4. Diff the regenerated graphs. ~30 fit nodes disappear across 40 `_api` graphs
   plus UI twins; every remaining append node gains three widget values.
5. Confirm the six `allow_upscale=True` graphs still carry it, including
   `h3_image_ref_plus_text_to_video`.
6. Re-run `check_node_ids.py`, `check_reference_runtime.py`,
   `show_red_reference_runtime.py`, `check_reference_fit.py`,
   `check_graph_discovery.py`, `audit_shipped_reference_bounds.py`.

`bench/check_reference_runtime.py::append_is_copy_on_add_and_ordered` survives — the
chain is unchanged. `bench/check_reference_contracts.py` is unaffected; its
subject is core's `MiniMaxH3ReferenceToVideo`, which this does not touch.

## What stays as it is

The chain itself, the per-reference record model, `reference_order.py`'s label
assignment and ownership validation, and `MiniMaxH3ReferenceConditioning`'s
schema. Per-reference sizing stays per-reference and is **not** hoisted onto the
conditioner as one global knob — each record already carries its own policy,
geometry, latent grid and rotary slot, and mixed geometry is supported by
construction.
