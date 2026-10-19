# Where Sol sits in a graph, and how it silently stops running

Detail page of [`sol_attn.md`](sol_attn.md). Every trap here renders a plausible clip; none raises.

## The order

```
Load Diffusion Model -> Model Attention Backend (kitchen) -> MiniMax H3 Sparse Attention -> BasicGuider
```

Sol comes after the dense node: it composes with the attention override already installed and hands every call it
declines to it. The default dense node is core's Model Attention Backend on "comfy kitchen attention"
(`h3_config.DENSE_BACKEND_NODE`, since 2026-09-15); the arms in `bench/check_attention_defaults.py::FLOOR_STEMS` wire
our sage node or nothing in its place. They **alternate rather than stack**: inside the sigma window Sol runs and the
dense kernel is bypassed; outside it, and on the blocks in `dense_blocks`, the dense kernel runs.

Since 2026-09-17 no generated graph wires `SageChainAssert` (owner). The node stays registered so saved graphs still
load.

## What declines to the dense kernel

- **Steps outside `start_percent`/`end_percent`.** Both gates compare sigmas, not steps, so the count of dense steps
  depends on the schedule ([`sol_measured_arms.md`](sol_measured_arms.md), "The sigma window is not a step
  fraction").
- **Blocks named in `dense_blocks`.** The dense kernel, not exact attention. `MiniMaxH3ExactBlocks` is the node that
  reaches ComfyUI's own unaccelerated attention at a block ([`sol_dense_blocks.md`](sol_dense_blocks.md)).
- **Calls shorter than `min_tokens`.** The 2 token-refiner calls (about 311 rows on the shipped t2v graph) fall below
  it; every DiT call (31k-128k tokens) is above it. The shipped t2v graphs hand the sampler already-projected text, so
  there are no sampler-time refiner rows at all (the first live route record, 2026-09-01).
- **A masked attention call** declines to the fallback. Until 2026-08-30 it did so with no log line even under
  `verbose`; it now warns once per process. Unreachable on every shipped graph (no node here writes `noise_mask`, no
  graph wires a mask-typed node into the DiT); it would arrive quietly the day a graph adopts masked H3 attention.
- **Anything else `sol_attn_h3.py::_ineligible` refuses**: no kitchen `sol_attn`, a dtype the caller does not accept,
  a head dim other than 128, cross-attention, or a q/k shape mismatch. The reason is the verbose log line and the
  route record's.

## Traps

**An unreachable Sol node is expensive and silent.** A hand-built graph with an active Sol node whose MODEL output
feeds nothing renders dense and looks normal. `bench/preflight_graph.py` reports it, on UI-format graphs too
(`bench/results/2026-08-18_attention_defaults.json` is the run that found it).

**`ModelAttentionBackend` must not be downstream of Sol.** Core's `set_model_optimized_attention`
(`comfy/model_patcher.py`) assigns `optimized_attention_override` unconditionally and does not chain, so downstream
of Sol it deletes Sol and the graph still renders. When it selects "comfy kitchen attention" it also copies the
backend's `container_function` onto the override, and `wrap_attn` dispatches straight to it; H3 is on that
container path (`comfy/ldm/minimax/model.py` wraps q/k/v in `AttentionTensorContainer`). Re-verified against ComfyUI
`55b6a9b1` after the kitchen attention merge (`bf4c9a08`, #15479).

**A forward-patching sage node makes the backend node nearly inert.** KJNodes' sage node and ours both replace
`diffusion_model.blocks.{i}.attn.forward`, deleting the call site, so a kitchen backend override reaches only the 2
token-refiner blocks, in either order. From the outside that is indistinguishable from the backend being slow.

**KJNodes' `MiniMaxH3MemoryEfficientSageAttentionPatch` silently replaces our sage node.** Both write
`diffusion_model.blocks.{i}.attn.forward` unconditionally (`nodes/ltxv_nodes.py` in KJNodes, `nodes.py` here); last
node wins. On sm89 its forward hardcodes fp8 PV with no mode input. A chain assert that only tests the patch key list
is non-empty reports green through it. Read from source on 2026-08-14, not rendered. KJNodes' other attention node,
`MiniMaxLowVRAMAttention`, yields to an existing patch (`if attn_key in m.object_patches: continue`). No shipped graph
wires either.

**Turning Sol on does not protect a graph from a replaced dense kernel.** It was once argued here that with Sol on
the dense kernel "gets nothing"; that forgot the sigma window. Every step outside the window runs on whatever dense
kernel is installed, so a swap there reaches the warm-up steps that set structure and, before 2026-09-11, the final
step. Whether those steps are more precision-sensitive than the middle is unmeasured.

**A capture-node or VSA graph replaces the block attention Sol would override.** That is why the FastH3 contract arms
are in `SOL_EXEMPT_STEMS` rather than carrying Sol.
