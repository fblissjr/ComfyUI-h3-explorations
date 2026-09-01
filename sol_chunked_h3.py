"""Sol-Attn's chunked producer as H3's attention forward: Q, K, V never built.

## What this is

comfy-kitchen's second Sol entry, `sol_attn_chunked`, takes chunks of the
fused `qkv_proj` output (`[M, 3*H*128]` bf16, 64-aligned starts), applies the
per-head RMS norm and the split-half rope itself, chunk by chunk, and hands
the routed attention back without ever holding the full Q, K and V tensors.
Upstream reports about 5 GB less peak at 113k tokens. At the canonical
104,361-token sequence on a 24 GB card that is the size of lever that decides
whether a reference-heavy arm renders at all (`docs/open_experiments.md` #25).

`sol_attn_h3.py` cannot reach it: `optimized_attention` hands that node
finished Q, K and V, so the saving is already spent by the time its override
runs. This node reaches the projection instead. It is a forward for H3's
`Attention` module -- the same seam `MiniMaxH3SageAttention` patches -- and it
owns exactly what core's forward does between the projection and the kernel:

    q, k, v = qkv_proj(x).split(H * 128)          # q | k | v blocks, the producer's layout
    rms_rope_split_half_(q, k, rope_freqs, qw, kw, eps, rot)   # comfy-kitchen's fused op
    out = attention(q, k, v); out_proj(out)

The producer applies the SAME fused norm-and-rope per chunk, so the per-chunk
arithmetic is core's own, not a re-implementation. What differs is the
routing threshold: the direct path centres keys on the current call's K-mean,
the producer on the PREVIOUS step's, returned by each call and fed to the
next (it runs twice on a module's first call to bootstrap). So counts and
output are close to the direct path, not bitwise equal to it; the check
reports how close.

## How it composes: through Sol's gate, not around it

The Sol node's composition gate (`sol_attn_h3._compose_module_patch`) already
prefers a published `sol_take_forward` delegate over the stock forward for
the calls it takes. This node publishes that delegate. So the arrangement is:

    Sage forward patch  (declined calls: outside the window, below min_tokens)
    Sol gate            decides per call from `sol_compose`
      taken  -> THIS forward: qkv_proj per chunk -> sol_attn_chunked -> out_proj
      declined -> Sage, as before

Two consequences. It needs a foreign forward patch below Sol to exist at
all, because the gate only wraps one; with no Sage node the stock forward
runs and this delegate is never consulted, and the node says so at patch
time. And `dense_blocks` are still decided by Sol's override: for a block in
that list this forward runs core's stock forward, which reaches
`optimized_attention` and lets the override route it `dense_block` exactly
as today.

## What it records

Armed (`H3_SOL_OBSERVE`), every call it takes is a `sol_chunked` row with
counts from the producer's own launch (the branch adds `blk_cnt` to
`sol_attn_chunked` too), `path: chunked_delegate`, and the allocator's
high-water mark like every other row. The memory claim is graded on that
field: same graph and seed with and without this node, last row's
`peak_alloc_bytes` per render.

## What it does not change

Nothing in `sol_attn_h3.py`, nothing in the Sage node, no default, no graph.
Remove the node and the arrangement is exactly what it was. It refuses
anything that is not an H3-shaped call (2-D bf16 CUDA tokens with a rope
table) by running the stock forward, so a non-H3 block cannot be fed to a
producer that assumes H3's layout.
"""

import logging

import torch

import comfy.model_management
from comfy_api.latest import io

from . import sol_observe
from .sol_attn_h3 import _sink_blocks, _stats, BLOCK_SIZE

try:
    from comfy_kitchen.backends import cuda as _ck_cuda
except Exception:  # pragma: no cover
    _ck_cuda = None


def _prompt_id():
    try:
        from comfy_execution.utils import get_executing_context
        ctx = get_executing_context()
        return None if ctx is None else ctx.prompt_id
    except Exception:                                  # noqa: BLE001
        return None


def make_chunked_forward(chunk_rows=4096, verbose=False):
    """The `sol_take_forward` delegate: `(module, x, rope_freqs=, transformer_options=)`.

    `chunk_rows` must be a multiple of 64: the producer requires 64-aligned
    chunk starts because its blocks are 64 rows.
    """
    if chunk_rows % BLOCK_SIZE or chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be a positive multiple of {BLOCK_SIZE}, got {chunk_rows}")
    # per module: (prompt id, kmean, vscale) -- last step's statistics, reset per prompt
    state = {}
    _logged = set()

    def log_once(key, msg, level=logging.INFO):
        if key not in _logged:
            _logged.add(key)
            logging.log(level, f"[h3-sol-chunked] {msg}")

    def forward(module, x, rope_freqs=None, transformer_options=None, **kwargs):
        options = transformer_options if isinstance(transformer_options, dict) else {}
        stock = type(module).forward

        def fallback():
            return stock(module, x, rope_freqs=rope_freqs, transformer_options=transformer_options, **kwargs)

        if (_ck_cuda is None or rope_freqs is None or not torch.is_tensor(x) or x.ndim != 2
                or x.device.type != "cuda" or x.dtype != torch.bfloat16):
            return fallback()
        gate = options.get("sol_compose") or {}
        settings = gate.get("settings") or {}
        block = options.get("sol_block")
        dense = set(settings.get("dense_blocks") or [])
        if block is not None and block in dense:
            return fallback()          # Sol's override routes it dense_block, as today

        tokens = int(x.shape[0])
        heads, head_dim = int(module.heads), int(module.head_dim)
        tau = float(settings.get("tau", 1.0))
        profile = settings.get("tau_profile") or {}
        block_tau = float(profile.get(str(block), tau)) if block is not None else tau
        topk = float(settings.get("topk_ratio", 0.0))
        tail = bool(settings.get("tail", True))
        sink, sink_q = _sink_blocks(options, tokens, settings.get("sink_conditioning", "off"))
        # detached: comfy-kitchen's rope refuses any input that requires grad,
        # and a loaded weight never does, but a bench stub's might
        qw = comfy.model_management.cast_to(module.q_norm.weight, device=x.device).detach()
        kw = comfy.model_management.cast_to(module.k_norm.weight, device=x.device).detach()

        def chunks():
            for start in range(0, tokens, chunk_rows):
                yield module.qkv_proj(x[start:start + chunk_rows])

        pid = _prompt_id()
        prev = state.get(id(module))
        kmean = vscale = None
        if prev is not None and prev[0] == pid:
            kmean, vscale = prev[1], prev[2]
        observing = sol_observe.enabled()
        counts = None
        if observing:
            counts = torch.empty((1, heads, (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE),
                                 dtype=torch.int32, device=x.device)
        try:
            out, kmean_next, vscale_next = _ck_cuda.sol_attn_chunked(
                chunks, tokens, heads, rope_freqs, (qw, kw), kmean=kmean, vscale=vscale,
                tau=block_tau, topk_ratio=topk, sink_blocks=list(sink), sink_q=list(sink_q),
                rope_eps=float(module.q_norm.eps), tail=tail, blk_cnt=counts)
        except Exception as exc:                       # noqa: BLE001
            log_once(("error", type(exc).__name__, str(exc)[:80]),
                     f"chunked producer failed ({exc}); this call ran on the stock forward",
                     level=logging.ERROR)
            _stats["errors"] += 1
            if observing:
                sol_observe.record(route="chunked_error", reason=f"{type(exc).__name__}: {exc}"[:200],
                                   counts=None, options=options, settings=settings, block=block,
                                   block_tau=block_tau, tokens=tokens, batch=1, heads=heads,
                                   sink=sink, sink_q=sink_q, tail=tail, topk_ratio=topk,
                                   min_tokens=int(settings.get("min_tokens", 0)), path="chunked_delegate")
            return fallback()
        state[id(module)] = (pid, kmean_next, vscale_next)
        _stats["sparse"] += 1
        options["h3_attn_route"] = "sol_chunked"
        if verbose:
            log_once((tokens, heads, "chunked"),
                     f"chunked producer took ({tokens}, {heads}) in {chunk_rows}-row chunks, "
                     f"tau {block_tau}, sinks {sink} {sink_q}")
        if observing:
            sol_observe.record(route="sol_chunked", reason=None, counts=counts, options=options,
                               settings=settings, block=block, block_tau=block_tau, tokens=tokens,
                               batch=1, heads=heads, sink=sink, sink_q=sink_q, tail=tail,
                               topk_ratio=topk, min_tokens=int(settings.get("min_tokens", 0)),
                               path="chunked_delegate")
        return module.out_proj(out.view(tokens, heads * head_dim))

    setattr(forward, "_h3_chunked", True)
    return forward


class MiniMaxH3SolChunked(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SolChunked",
            display_name="MiniMax H3 Sol-Attn chunked producer",
            is_experimental=True,
            category="model/attention/minimax",
            description=(
                "Feeds Sol-Attn from chunks of the fused QKV projection so the full "
                "Q, K and V are never built: comfy-kitchen's sol_attn_chunked as "
                "H3's attention forward, for the calls Sol takes. Wire it AFTER "
                "MiniMaxH3SolAttn, which must sit on a graph that also carries "
                "MiniMaxH3SageAttention (or another forward patch) below it; the "
                "Sol gate hands the calls it takes to this node and the rest to "
                "Sage, exactly as before. Memory lever, not a quality knob: the "
                "producer routes from the previous step's key statistics, so "
                "output and counts are close to the direct path, not identical."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("chunk_rows", default=4096, min=64, max=1 << 17, step=64,
                             tooltip="Rows of the packed sequence per projection chunk; a "
                                     "multiple of 64 because the producer's blocks are 64 "
                                     "rows. Smaller chunks hold less at once and launch more."),
                io.Boolean.Input("verbose", default=False),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, chunk_rows, verbose) -> io.NodeOutput:
        if _ck_cuda is None or not hasattr(_ck_cuda, "sol_attn_chunked"):
            raise RuntimeError("the installed comfy_kitchen has no CUDA sol_attn_chunked; "
                               "bench/check_sol_kernel.py reports which build is installed")
        import inspect
        if "blk_cnt" not in inspect.signature(_ck_cuda.sol_attn_chunked).parameters and sol_observe.enabled():
            raise RuntimeError("H3_SOL_OBSERVE is set but this build's sol_attn_chunked takes no "
                               "blk_cnt; rebuild from the branch or start the server unarmed")
        m = model.clone()
        to = m.model_options["transformer_options"]
        if "sol_compose" not in to:
            raise RuntimeError("wire MiniMaxH3SolAttn before this node: it publishes the gate "
                               "(sol_compose) this delegate is consulted through")
        has_patch = any(k.endswith(".forward") and "attn" in k.rsplit(".", 2)[-2].lower()
                        for k in m.object_patches)
        if not has_patch:
            logging.warning("[h3-sol-chunked] no attention forward patch is present below Sol "
                            "(no MiniMaxH3SageAttention?); Sol's gate only wraps a patched "
                            "forward, so this delegate will never be consulted")
        to["sol_take_forward"] = make_chunked_forward(chunk_rows=chunk_rows, verbose=verbose)
        logging.info(f"[h3-sol-chunked] delegate published: {chunk_rows}-row chunks")
        return io.NodeOutput(m)
