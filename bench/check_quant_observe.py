#!/usr/bin/env python3
"""Grade the Tier 1 activation observer before it is given a render.

The observer's whole value is that its output can be trusted to be complete,
so the cases here are about completeness and non-perturbation rather than
about the statistics themselves.

**The case that earns the file is `fused_fc2_is_reached`.** `mlp.fc2.forward`
is never called on the shipped INT8 path -- `comfy.ops.linear_input_act` owns
it for the SwiGLU fusion -- so the obvious observer records three kinds and
looks complete. That exact defect already shipped once here, in
`unmerged_blocks` on 2026-08-30. A check that only counted rows would pass on
the broken design, so this one asserts the KIND SET and drives the real fused
helper rather than a stand-in.

**`wrapper_does_not_perturb` is the other load-bearing one.** An observer that
changes the tensor makes every number it records a statement about a different
model. The MLP wrapper replicates core's two-line body instead of calling it
(to avoid doubling a production-sized matmul), which is exactly the kind of
replication that can drift, so it is compared against core's own expression
bit-for-bit.

Needs CUDA and comfy-kitchen: the int8 path is where the fused helper lives,
and a CPU stand-in would exercise a different branch than the one that ships.
Exits 2 SKIP without them rather than passing on a weaker path.

    uv run --active --no-sync python bench/check_quant_observe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# `_HERE` is the DIRECTORY, so its parents are [0] the pack, [1] custom_nodes,
# [2] the ComfyUI root -- the off-by-one `CLAUDE.md` names, and which cost this
# file two runs. `Path(__file__).resolve().parents[2]` would be custom_nodes.
_PACK = _HERE.parents[0]
sys.path.insert(0, str(_HERE.parents[2]))

FAILED: list[str] = []


def check(name, fn):
    try:
        fn()
        print(f"  ok    {name}")
    except AssertionError as exc:
        FAILED.append(name)
        print(f"  FAIL  {name}: {exc}")
    except Exception as exc:                          # noqa: BLE001
        FAILED.append(name)
        print(f"  ERROR {name}: {type(exc).__name__}: {exc}")


def main() -> int:
    try:
        import torch
        from comfy import ops
        from comfy_kitchen.backends.eager.quantization import (
            quantize_int8_convrot_weight)
    except Exception as exc:                          # noqa: BLE001
        print(f"SKIP: needs comfy and comfy-kitchen ({exc})")
        return 2
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA; the fused fc2 helper is the subject")
        return 2

    # The pack's directory name carries a hyphen, so it is not an importable
    # identifier, and both modules under test use relative imports and so need
    # a package to resolve against. Synthesise the namespace rather than
    # executing the pack's `__init__.py`, which would import every node and
    # pull in the whole dependency surface for two modules.
    import importlib, importlib.machinery, importlib.util
    spec = importlib.machinery.ModuleSpec("h3pack", None, is_package=True)
    spec.submodule_search_locations = [str(_PACK)]
    sys.modules["h3pack"] = importlib.util.module_from_spec(spec)
    qo = importlib.import_module("h3pack.quant_observe")
    obs = importlib.import_module("h3pack.dit_observe")

    HID, FFN = 256, 512
    O = ops.mixed_precision_ops({"probe": True}, compute_dtype=torch.bfloat16)

    def build_mlp():
        """An MLP whose fc1/fc2 are real int8_convrot linears, like the DiT's."""
        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = O.Linear(HID, FFN * 2, bias=False,
                                    device="cuda", dtype=torch.bfloat16)
                self.fc2 = O.Linear(FFN, HID, bias=False,
                                    device="cuda", dtype=torch.bfloat16)

            def forward(self, x):
                return ops.linear_input_act(self.fc2, self.fc1(x), "swiglu")

        m = MLP()
        conf = {"format": "int8_tensorwise", "convrot": True,
                "convrot_groupsize": 256}
        blob = torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)
        for lin, shape in ((m.fc1, (FFN * 2, HID)), (m.fc2, (HID, FFN))):
            w = torch.randn(*shape, device="cuda", dtype=torch.float32)
            q, s = quantize_int8_convrot_weight(w, 256, stochastic_rounding=0)
            lin.load_state_dict({"weight": q, "weight_scale": s,
                                 "comfy_quant": blob}, strict=True)
        return m

    def arm(tmp):
        obs._rows.clear(); obs._failures.clear(); obs._context.clear()
        obs._dir = Path(tmp)
        obs._armed = True
        obs._path = None

    import tempfile
    tmp = tempfile.mkdtemp()
    x = torch.randn(64, HID, device="cuda", dtype=torch.bfloat16)

    def wrapper_does_not_perturb():
        """The replicated MLP body must equal core's, bit for bit."""
        m = build_mlp()
        arm(tmp)
        want = m.forward(x)
        got = qo.make_mlp_forward(m, 0)(x)
        assert torch.equal(want, got), (
            f"the MLP wrapper changed the output; max|d| "
            f"{float((want - got).abs().max())}. Every statistic it records "
            f"would then be about a different model.")

    def fused_fc2_is_reached():
        """Both MLP kinds recorded -- the defect this file exists for."""
        m = build_mlp()
        arm(tmp)
        qo.make_mlp_forward(m, 0)(x)
        kinds = {r["kind"] for r in obs._rows}
        assert kinds == {"mlp.fc1", "mlp.fc2"}, (
            f"recorded {sorted(kinds)}; a patch on `fc2.forward` records only "
            f"mlp.fc1 and looks complete. This is the 2026-08-30 defect.")
        fc2 = next(r for r in obs._rows if r["kind"] == "mlp.fc2")
        assert fc2["in_features"] == FFN, (
            f"fc2 input width {fc2['in_features']}, expected {FFN}. Recording "
            f"the PRE-swiglu tensor ({FFN * 2}) would be the wrong tensor: the "
            f"activation quantiser never sees it.")
        assert not obs._failures, f"failures recorded: {obs._failures}"

    def linear_wrapper_records_and_returns():
        m = build_mlp()
        arm(tmp)
        want = m.fc1(x)
        got = qo.make_linear_forward(m.fc1.forward, 3, "attn.qkv_proj")(x)
        assert torch.equal(want, got), "the linear wrapper changed the output"
        r = obs._rows[0]
        assert r["block"] == 3 and r["kind"] == "attn.qkv_proj"
        assert len(r["chan_absmax"]) == HID, (
            f"chan_absmax has {len(r['chan_absmax'])} entries for "
            f"in_features {HID}; the SmoothQuant scale vector is this column "
            f"and a wrong length makes it unusable")

    def shape_check_is_red_when_a_kind_is_missing():
        """The red proof: three kinds must NOT report ok.

        Driven rather than asserted, because the whole failure mode is a
        record that looks complete. `expect_kinds` is 4 and is not inferred
        from what reported -- inferring it is what made the original defect
        invisible.
        """
        m = build_mlp()
        arm(tmp)
        for i in range(2):
            obs.set_context(step=0, sigma=1.0)
            qo.make_linear_forward(m.fc1.forward, i, "attn.qkv_proj")(x)
            qo.make_linear_forward(m.fc1.forward, i, "attn.out_proj")(x)
            qo.make_linear_forward(m.fc1.forward, i, "mlp.fc1")(x)
        sc = obs._shape_check(expect_blocks=2)
        assert sc["ok"] is False, (
            "three kinds over two blocks reported ok; a capture missing fc2 "
            "everywhere is indistinguishable from one that never wanted it")
        assert sc["kinds_seen"] == ["attn.out_proj", "attn.qkv_proj", "mlp.fc1"]
        # and green when the fourth arrives
        for i in range(2):
            qo.make_linear_forward(m.fc1.forward, i, "mlp.fc2")(x)
        assert obs._shape_check(expect_blocks=2)["ok"] is True, (
            "still red with all four kinds at both blocks -- the assertion is "
            "not measuring what it claims")



    def fc2_does_not_materialise_the_activation():
        """The bug the small-shape cases could not see: 2.79 GiB at production.

        `INPUT_ACT_EAGER["swiglu"]` returns a NEW tensor, so applying it to the
        whole fc1 output allocates `rows x ffn` -- 104361 x 14336, about
        2.79 GiB -- which is exactly the allocation `linear_input_act`'s fusion
        exists to avoid, on a 24 GiB card holding a 19.5 GiB checkpoint. Every
        other case here runs 64 rows, where the difference is invisible.

        Measured rather than asserted structurally: run enough rows that the
        chunked path and the whole-tensor path differ by more than allocator
        noise, and require the peak to sit nearer one chunk than the whole.
        """
        ffn = 1024
        rows = 16 * obs.chunk_rows(ffn, 2)
        big = torch.randn(rows, ffn * 2, device="cuda", dtype=torch.bfloat16)
        whole = rows * ffn * 2                      # bytes, bf16, if not chunked
        arm(tmp)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        obs.record(0, "mlp.fc2", big,
                   transform=ops.INPUT_ACT_EAGER["swiglu"])
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - before
        # The property is that peak does NOT grow with rows -- a threshold
        # against the whole-activation size only holds at one row count, which
        # is how the first version of this case failed against working code.
        small = torch.randn(obs.chunk_rows(ffn, 2), ffn * 2,
                            device="cuda", dtype=torch.bfloat16)
        arm(tmp)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        b2 = torch.cuda.memory_allocated()
        obs.record(0, "mlp.fc2", small,
                   transform=ops.INPUT_ACT_EAGER["swiglu"])
        torch.cuda.synchronize()
        one = torch.cuda.max_memory_allocated() - b2
        assert peak < one * 1.5, (
            f"peak grew with rows: {peak / 2**20:.0f} MiB over 16 chunks "
            f"against {one / 2**20:.0f} MiB over one. The transform must be "
            f"applied PER CHUNK; applied whole it reproduces the "
            f"{whole / 2**20:.0f} MiB allocation the fused path avoids.")
        assert peak < whole, (
            f"peak {peak / 2**20:.0f} MiB is not below the whole activation's "
            f"{whole / 2**20:.0f} MiB, so nothing was saved")
        r = obs._rows[0]
        assert r["in_features"] == ffn, (
            f"in_features {r['in_features']} should be the POST-swiglu width "
            f"{ffn}; recording {ffn * 2} means the transform never ran")

    def step_index_uses_the_sigma_scale():
        """`timestep` is 0..1000 and `sample_sigmas` are 0..1.

        Core converts with `sigma_v = timestep.flatten()[0] / 1000.0`
        (`comfy/ldm/minimax/model.py:561`). Comparing the raw value against the
        schedule makes every step the argmin of ~1000 against ~1.0 -- the same
        entry every time -- so every row carries the wrong step label while the
        file looks complete. Red-proves the scale, not merely the plumbing.
        """
        sched = [1.0, 0.973, 0.923, 0.8, 0.0]
        arm(tmp)

        def base(x, t, c, to, *a, **kw):
            return None

        fwd = qo.make_outer_forward(base, expect_blocks=1, capture_id="cid")
        for want, sigma in ((1, 0.973), (3, 0.8)):
            fwd(None, torch.tensor([sigma * 1000.0]), None,
                {"sample_sigmas": sched})
            got = obs._context.get("step")
            assert got == want, (
                f"timestep {sigma * 1000.0} against schedule {sched} mapped to "
                f"step {got}, expected {want}. Comparing the raw 0..1000 value "
                f"to 0..1 sigmas pins every step to one index.")
        assert abs(obs._context["sigma"] - 0.8) < 1e-6, (
            f"sigma recorded as {obs._context['sigma']}, expected 0.8 -- the "
            f"raw timestep is 1000x the sigma")

    def outer_patch_chains_rather_than_replaces():
        """The composition bug, red-proved against a stub ModelPatcher.

        **The existing cases could not have caught this**: they drive the
        wrappers directly and never run `execute`, so nothing exercised how the
        patches are INSTALLED. The first version took `dm.forward` -- the raw
        bound method -- for the outer patch, which on any graph wiring
        `MiniMaxH3PDDLoRA` would silently replace that node's step tracker,
        because `add_object_patch` is last-writer-wins. Found by an external
        review before it ran.

        The property asserted is that a forward installed BEFORE this observer
        still executes after it, which is exactly what chaining from
        `get_model_object` buys and what chaining from `dm.forward` loses.
        """
        calls = []

        class StubLinear:
            def forward(self, x):
                return x

        class StubMLP:
            def __init__(self):
                self.fc1 = StubLinear()
                self.fc2 = StubLinear()

        class StubAttn:
            def __init__(self):
                self.qkv_proj = StubLinear()
                self.out_proj = StubLinear()

        class StubBlock:
            def __init__(self):
                self.attn = StubAttn()
                self.mlp = StubMLP()

        class StubDM:
            def __init__(self):
                self.blocks = [StubBlock(), StubBlock()]

            def forward(self, *a, **kw):
                calls.append("raw")

        class StubModel:
            def __init__(self, dm):
                self.dm = dm
                self.object_patches = {}

            def clone(self):
                n = StubModel(self.dm)
                n.object_patches = dict(self.object_patches)
                return n

            def add_object_patch(self, k, v):
                self.object_patches[k] = v

            def get_model_object(self, name):
                if name in self.object_patches:
                    return self.object_patches[name]
                if name == "diffusion_model":
                    return self.dm
                obj = self.dm
                for part in name.split(".")[1:]:
                    # `blocks.0` indexes a list; ComfyUI's own resolver does
                    # the same, and a stub that only does getattr would fail
                    # on the real key shape rather than on the property.
                    obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
                return obj

        dm = StubDM()
        model = StubModel(dm)

        def prior(*a, **kw):
            calls.append("prior")
            return dm.forward(*a, **kw)

        model.add_object_patch("diffusion_model.forward", prior)

        real_is_h3 = qo._is_minimax_h3
        qo._is_minimax_h3 = lambda _dm: True
        try:
            arm(tmp)
            out = qo.MiniMaxH3QuantObserve.execute(model, blocks="-1")
        finally:
            qo._is_minimax_h3 = real_is_h3

        patched = out.result[0] if hasattr(out, "result") else out[0]
        patched.object_patches["diffusion_model.forward"](
            None, None, None, {})
        assert "prior" in calls, (
            f"the forward installed before this observer did not run: {calls}. "
            f"Chaining from `dm.forward` instead of `get_model_object` "
            f"replaces it, which on a PDD graph kills head selection silently.")

    def inert_without_the_env_var():
        m = build_mlp()
        obs._rows.clear(); obs._failures.clear()
        obs._dir = None
        obs._armed = False
        qo.make_mlp_forward(m, 0)(x)
        assert not obs._rows, (
            "recorded with H3_QUANT_OBSERVE unset; the env gate is the reason "
            "a node left in a saved graph cannot arm a later render")

    print("quant observer")
    check("wrapper does not perturb", wrapper_does_not_perturb)
    check("fused fc2 is reached", fused_fc2_is_reached)
    check("linear wrapper records and returns", linear_wrapper_records_and_returns)
    check("shape check is red when a kind is missing",
          shape_check_is_red_when_a_kind_is_missing)
    check("fc2 does not materialise the activation",
          fc2_does_not_materialise_the_activation)
    check("step index uses the sigma scale", step_index_uses_the_sigma_scale)
    check("outer patch chains rather than replaces",
          outer_patch_chains_rather_than_replaces)
    check("inert without the env var", inert_without_the_env_var)

    if FAILED:
        print(f"\n{len(FAILED)} case(s) FAILED: {', '.join(FAILED)}")
        return 1
    print("\nall ok -- the observer records four kinds and changes nothing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
