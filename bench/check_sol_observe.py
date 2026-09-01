#!/usr/bin/env python3
"""Grade the Sol route observer before it is given a render.

The observer's value is that a row can be trusted to describe the call it
came from: the right prompt, the right block, the counts the kernel actually
produced, and a failure that stops the capture rather than thinning it. So
the cases are about identity, completeness and non-perturbation; the count
semantics themselves are pinned upstream (`tests/test_sol_attn.py` on the
`sol-blk-cnt` branch) with closed-form equalities at both tau extremes.

Claims, i.e. what breaks if a case is deleted:

  inert_without_the_env_var
      unarmed, the node passes NO `blk_cnt` keyword, writes nothing, and its
      output is the same bytes as a direct kernel call. This is the `enabled()`
      defect class `pdd_observe.py` shipped (`bool(str(Path("")))`), and the
      compatibility claim for an older installed wheel.
  every_route_is_recorded
      one row per override call for every exit -- masked, dense_block,
      outside_range, ineligible, sol, kernel_error -- with the route name and a
      reason, plus header and config rows. A recorder that only saw Sol calls
      would report a render as all-Sol.
  identity_does_not_mix_prompts
      two calls under two executing contexts carry two prompt ids, read at
      call time; the conditioning uuids and cond_or_uncond lists are recorded
      whole. This is the cached-patched-model case.
  wrong_slice_is_red_and_escapes_the_fallback
      a count tensor above NTB, below its forced floor, or with a sink_q row
      not at NTB is written as an `error` row AND raises out of the override;
      the dense fallback is NOT called. A recorder that swallowed its own
      failure would leave a plausible partial file and a finished render.
  summaries_agree_with_an_independent_reduction
      the row's densities and per-head means are recomputed from the raw
      sidecar bytes with a reduction written here, not imported, including
      the forced floor from a set-based definition and the CRC.
  armed_with_old_wheel_fails_at_patch_time
      `_require_kernel` raises when armed against a `sol_attn` without
      `blk_cnt`, and passes unarmed. Otherwise an armed server on a stale wheel
      renders and records nothing.
  stale_block_label_is_cleared
      after block 49's forward completes, `sol_block` is absent; a following
      refiner-shaped call is recorded with no block and `scope: unknown`; the
      block's output is bit-identical. The outer forward's `finally` also
      drops `h3_segments` with the two spans.
  observer_only_block_indexing
      armed with empty dense_blocks and no tau profile, `_apply_patch` still
      installs the block hooks, and a synthetic block call records its true
      index.
  raw_off_writes_no_sidecar
      `raw=0` leaves no `.u16` file and no raw pointer, and the row is
      otherwise complete.

Needs CUDA and an installed comfy_kitchen whose `sol_attn` takes `blk_cnt`;
exits 2 SKIP without either rather than passing on a weaker path.

    uv run --active --no-sync python bench/check_sol_observe.py
"""

from __future__ import annotations

import inspect
import shutil
import sys
import tempfile
import types
import uuid
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

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


# A PackedLayout stand-in so `install_h3_morton` can patch this module's class
# (it patches `sys.modules[type(model).__module__].PackedLayout`).
class PackedLayout:
    def __init__(self, *_args, **_kwargs):
        self.segments = []
        self.position_ids = None


def main() -> int:
    try:
        import numpy as np
        import torch
        import comfy_kitchen as ck
        from _live_sol import live_sol, sol_observe
    except Exception as exc:                          # noqa: BLE001
        print(f"SKIP: needs torch, comfy_kitchen and the pack ({exc})")
        return 2
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA; the kernel is the subject")
        return 2
    if "blk_cnt" not in inspect.signature(ck.sol_attn).parameters:
        print("SKIP: the installed comfy_kitchen.sol_attn has no blk_cnt; "
              "rebuild from the sol-blk-cnt branch (vendor/rebuild_kernel.sh)")
        return 2

    node = live_sol()
    obs = sol_observe()
    from comfy_execution.utils import CurrentNodeContext

    torch.manual_seed(0)
    b, h, t, d = 1, 2, 1024 + 40, 128                # 17 key blocks, ragged tail
    n = (t + 63) // 64
    q, k, v = (torch.randn(b, h, t, d, device="cuda", dtype=torch.bfloat16)
               for _ in range(3))
    k[:, :, :64] += 2.0 * q[:, :, :64]               # something for the router to find
    tmp = Path(tempfile.mkdtemp(prefix="sol_observe_"))

    calls = {"dense": 0}

    def dense_func(qq, kk, vv, heads, **kw):
        calls["dense"] += 1
        return torch.zeros_like(qq)

    def settings():
        return {"node": "test", "tau": 1.0, "topk_ratio": 0.0, "tail": True,
                "min_tokens": 64, "dense_blocks": [3], "n_blocks": 50}

    def make(**kw):
        base = dict(tau=1.0, min_tokens=64, sigma_start=10.0, sigma_end=0.1,
                    dense_blocks=frozenset({3}), settings=settings())
        base.update(kw)
        return node.make_override(**base)

    def call(override, opts, mask=None, qq=None, kk=None, vv=None):
        qq = q if qq is None else qq
        return override(dense_func, qq, k if kk is None else kk, v if vv is None else vv,
                        h, mask=mask, skip_reshape=True, skip_output_reshape=True,
                        transformer_options=opts)

    def opts(**extra):
        o = {"sigmas": torch.tensor([1.0]), "sample_sigmas": torch.tensor([2.0, 1.0, 0.5, 0.0]),
             "sol_block": 5}
        o.update(extra)
        return o

    def kernel(**kw):
        qs, ks, vs = (x.transpose(1, 2).contiguous() for x in (q, k, v))
        return ck.sol_attn(qs, ks, vs, **kw).transpose(1, 2)

    class Spy:
        """Stands in for the comfy_kitchen module the node holds: records the
        kwargs of every sol_attn call, optionally mutates the count buffer
        after the real kernel, optionally raises."""
        def __init__(self, after=None, raise_exc=None, signature=None):
            self.kwargs = []
            self.after = after
            self.raise_exc = raise_exc
            real = ck.sol_attn

            def sol_attn(*a, **kw):
                self.kwargs.append(dict(kw))
                if self.raise_exc is not None:
                    raise self.raise_exc
                out = real(*a, **kw)
                if self.after is not None and kw.get("blk_cnt") is not None:
                    self.after(kw["blk_cnt"])
                return out
            self.sol_attn = signature if signature is not None else sol_attn

    real_ck = node._ck

    def use(spy):
        setattr(node, "_ck", spy)

    def restore():
        setattr(node, "_ck", real_ck)

    def newdir(name):
        d = tmp / name
        d.mkdir()
        return d

    def rows_in(d):
        files = sorted(d.glob("sol_observe_*.jsonl"))
        assert len(files) == 1, f"expected one jsonl in {d}, found {len(files)}"
        return files[0], obs.read_rows(files[0])

    # ---- cases -----------------------------------------------------------

    def inert_without_the_env_var():
        obs.arm(None)
        assert obs.enabled() is False
        d = newdir("inert")
        spy = Spy()
        use(spy)
        try:
            got = call(make(), opts())
        finally:
            restore()
        assert spy.kwargs and all("blk_cnt" not in kw for kw in spy.kwargs), \
            f"unarmed call passed blk_cnt: {[sorted(kw) for kw in spy.kwargs]}"
        assert torch.equal(got, kernel(tau=1.0, tail=True)), "unarmed output is not the kernel's bytes"
        assert not any(d.iterdir()), "unarmed run wrote files"
        assert not list(tmp.glob("*.jsonl")), "unarmed run wrote a jsonl somewhere"

    def every_route_is_recorded():
        d = newdir("routes")
        obs.arm(f"dir={d}")
        assert obs.enabled()
        calls["dense"] = 0
        ov = make()
        call(ov, opts(), mask=torch.ones(1, 1, t, t, device="cuda"))          # masked
        call(ov, opts(sol_block=3))                                          # dense_block
        call(ov, opts(sigmas=torch.tensor([20.0])))                          # outside_range
        short = tuple(x[:, :, :32].contiguous() for x in (q, k, v))
        call(ov, opts(), qq=short[0], kk=short[1], vv=short[2])              # ineligible
        got = call(ov, opts())                                               # sol
        spy = Spy(raise_exc=RuntimeError("synthetic kernel failure"))
        use(spy)
        import logging
        logging.disable(logging.ERROR)      # the node logs the traceback; it is the fixture, not a failure
        try:
            call(ov, opts())                                                 # kernel_error
        finally:
            logging.disable(logging.NOTSET)
            restore()
        assert torch.equal(got, kernel(tau=1.0, tail=True)), "armed output differs from the kernel's bytes"
        assert calls["dense"] == 5, f"dense fallback called {calls['dense']} times, want 5"
        path, rows = rows_in(d)
        kinds = [r["kind"] for r in rows]
        assert kinds[0] == "header" and kinds[1] == "config", kinds[:2]
        assert rows[0]["timing_quotable"] is False and rows[0]["comfy_kitchen_version"]
        assert rows[1]["settings"]["dense_blocks"] == [3]
        callrows = [r for r in rows if r["kind"] == "call"]
        assert [r["route"] for r in callrows] == \
            ["masked", "dense_block", "outside_range", "ineligible", "sol", "kernel_error"], \
            [r["route"] for r in callrows]
        assert callrows[3]["reason"] == "seq 32 < 64", callrows[3]["reason"]
        assert "RuntimeError" in callrows[5]["reason"]
        assert all(r["config"] == rows[1]["digest"] for r in callrows)
        assert all(r["identity_source"] == "no_executing_context" for r in callrows)
        sol = callrows[4]
        assert sol["block"] == 5 and sol["scope"] == "dit"
        assert sol["schedule"]["state"] == "matched" and sol["schedule"]["schedule_index"] == 1
        assert sol["schedule"]["n_intervals"] == 3 and sol["schedule"]["schedule_len"] == 4
        assert sol["NTB"] == n and sol["shape_ok"] is True
        assert len(sol["per_head"]["kernel_mean"]) == h
        assert sol["raw"]["nbytes"] == b * h * n * 2 and "crc32" in sol["raw"]
        assert 0 < sol["kernel_density"]["mean"] <= 1.0
        assert sol["routed_density"] is not None and 0 <= sol["routed_density"]["mean"] <= 1.0
        assert rows[0]["denominators"]["kernel_density"].startswith("cnt / NTB")
        # the dense_block row names the block; the ineligible row still has one
        assert callrows[1]["reason"] == "block 3 in dense_blocks"
        obs.arm(None)

    def identity_does_not_mix_prompts():
        d = newdir("identity")
        obs.arm(f"dir={d}")
        ov = make()
        u1, u2 = uuid.uuid4(), uuid.uuid4()
        with CurrentNodeContext("prompt-A", "13", None):
            call(ov, opts(uuids=[u1], cond_or_uncond=[0]))
        with CurrentNodeContext("prompt-B", "13", 2):
            call(ov, opts(uuids=[u2, u1], cond_or_uncond=[0, 1]))
        _, rows = rows_in(d)
        callrows = [r for r in rows if r["kind"] == "call"]
        assert [r["prompt_id"] for r in callrows] == ["prompt-A", "prompt-B"]
        assert all(r["executing_node_id"] == "13" for r in callrows)
        assert [r["list_index"] for r in callrows] == [None, 2]
        assert callrows[0]["conditioning_uuids"] == [str(u1)]
        assert callrows[1]["conditioning_uuids"] == [str(u2), str(u1)]
        assert callrows[1]["cond_or_uncond"] == [0, 1]
        assert all(r["identity_source"] == "comfy_execution.utils" for r in callrows)
        obs.arm(None)

    def wrong_slice_is_red_and_escapes_the_fallback():
        d = newdir("wrong")
        obs.arm(f"dir={d}")
        ov = make(sink_conditioning="exact_kv_and_rows")
        # a layout: video from row 256 -> sink blocks [0, 4); audio rows [128, 256) -> sink_q [2, 4)
        layout = dict(sol_h3_video_span=(256, t), sol_h3_audio_span=(128, 256))
        # Each mutation trips its own clause: the floor case lowers a NON-sink_q
        # row (block 10 has floor sink 4 + diagonal 3), the sink_q case lowers a
        # sink_q row only.
        mutations = {
            "above NTB": lambda c: c.fill_(n + 1),
            "below forced floor": lambda c: c[:, :, 10].fill_(1),
            "sink_q row not NTB": lambda c: c[:, :, 2].fill_(n - 1),
        }
        for label, mutate in mutations.items():
            calls["dense"] = 0
            use(Spy(after=mutate))
            try:
                raised = None
                try:
                    call(ov, opts(**layout))
                except obs.SolObserveError as exc:
                    raised = exc
            finally:
                restore()
            assert raised is not None, f"{label}: no SolObserveError"
            assert calls["dense"] == 0, f"{label}: the dense fallback ran; the error was swallowed"
        _, rows = rows_in(d)
        errs = [r for r in rows if r["kind"] == "error"]
        assert len(errs) == 3 and all(r["stage"] == "shape_check" for r in errs), \
            [(r["kind"], r.get("stage")) for r in rows]
        assert "exceeds NTB" in errs[0]["message"], errs[0]["message"]
        assert "forced floor" in errs[1]["message"], errs[1]["message"]
        assert "sink_q" in errs[2]["message"], errs[2]["message"]
        assert not [r for r in rows if r["kind"] == "call" and r["route"] == "sol"], \
            "a sol row was written despite the failed shape check"
        obs.arm(None)

    def summaries_agree_with_an_independent_reduction():
        d = newdir("summaries")
        obs.arm(f"dir={d}")
        ov = make(sink_conditioning="exact_kv_and_rows")
        segs = [(0, 128, "text"), (128, 256, "audio"), (256, t, "video")]
        call(ov, opts(sol_h3_video_span=(256, t), sol_h3_audio_span=(128, 256), h3_segments=segs))
        path, rows = rows_in(d)
        sol = [r for r in rows if r["kind"] == "call"][0]
        assert sol["route"] == "sol" and sol["sink_blocks"] == [0, 4] and sol["sink_q"] == [2, 4]
        counts = obs.read_raw(path, sol).numpy()                   # (B, H, NQ), CRC checked
        assert counts.shape == (b, h, n)
        # independent forced floor: sink set union diagonal set, per query block
        sink = set(range(0, 4))
        forced = np.array([len(sink | {x for x in (qb - 1, qb, qb + 1) if 0 <= x < n})
                           for qb in range(n)], dtype=np.int64)
        sinkq = np.zeros(n, dtype=bool)
        sinkq[2:4] = True
        assert (counts[:, :, sinkq] == n).all()
        assert (counts >= forced[None, None, :]).all() and (counts <= n).all()
        kernel = counts / n
        assert abs(kernel.mean() - sol["kernel_density"]["mean"]) < 1e-9, (kernel.mean(), sol["kernel_density"])
        assert abs(kernel.min() - sol["kernel_density"]["min"]) < 1e-9
        assert abs(kernel.max() - sol["kernel_density"]["max"]) < 1e-9
        per_head = kernel.mean(axis=(0, 2))
        assert np.allclose(per_head, sol["per_head"]["kernel_mean"], atol=1e-9), (per_head, sol["per_head"])
        den = (n - forced)[None, None, :].astype(np.float64)
        routed = (counts - forced[None, None, :]) / den
        routed = routed[:, :, ~sinkq]
        assert abs(routed.mean() - sol["routed_density"]["mean"]) < 1e-9, (routed.mean(), sol["routed_density"])
        assert sol["routed_density"]["n"] == routed.size
        for hh in range(h):
            assert abs(routed[:, hh].mean() - sol["per_head"]["routed_mean"][hh]) < 1e-9
        # segments: overlap-weighted query-segment kernel density, recomputed
        assert [s["kind"] for s in sol["per_segment"]] == ["text", "audio", "video"]
        for seg, (a, bb, _kind) in zip(sol["per_segment"], segs):
            q0, q1 = a // 64, (bb - 1) // 64
            w = np.array([min(bb, (qb + 1) * 64) - max(a, qb * 64) for qb in range(q0, q1 + 1)], float)
            kd = (kernel[:, :, q0:q1 + 1] * w[None, None, :]).sum() / (w.sum() * b * h)
            assert abs(kd - seg["kernel"]) < 1e-9, (seg, kd)
        assert sol["per_segment"][1]["routed"] is None or sol["per_segment"][1]["routed"] >= 0
        assert sol["segments"] == [list(s) for s in segs]
        obs.arm(None)

    def armed_with_old_wheel_fails_at_patch_time():
        d = newdir("oldwheel")

        def old_sol_attn(q, k, v, tau=1.0, scale=None, sink_blocks=None, sink_q=None,
                         key_bias=None, topk_ratio=0.0, tail=True, block_len=None,
                         coarse_gate=None):
            raise AssertionError("must not be called")

        old = types.SimpleNamespace(sol_attn=old_sol_attn)
        obs.arm(f"dir={d}")
        use(old)
        try:
            raised = False
            try:
                node._require_kernel()
            except RuntimeError as exc:
                raised = "H3_SOL_OBSERVE" in str(exc)
            assert raised, "armed _require_kernel accepted a sol_attn without blk_cnt"
            obs.arm(None)
            node._require_kernel()          # unarmed: the old signature is fine
        finally:
            restore()

    def stale_block_label_is_cleared():
        d = newdir("stale")
        obs.arm(f"dir={d}")

        class Blk(torch.nn.Module):
            def forward(self, x, transformer_options=None):
                self.seen = (transformer_options or {}).get("sol_block")
                return x + 1

        model = types.SimpleNamespace(blocks=torch.nn.ModuleList([Blk() for _ in range(50)]))
        assert node._install_block_index(model)
        o = opts()
        del o["sol_block"]
        x = torch.arange(4.0)
        out = model.blocks[49](x, transformer_options=o)
        assert model.blocks[49].seen == 49, model.blocks[49].seen
        assert "sol_block" not in o, "sol_block survived the block's forward"
        assert torch.equal(out, x + 1), "the post-hook changed the block output"
        # a refiner-shaped call next: no block, unknown scope
        call(make(), o)
        _, rows = rows_in(d)
        r = [r for r in rows if r["kind"] == "call"][0]
        assert r["block"] is None and r["scope"] == "unknown", (r["block"], r["scope"])

        # the outer forward drops the segment table with the two spans
        class Stub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([Blk()])

            def rope_freqs(self, position_ids, device):
                return None

            def _forward(self, x, timestep, context, transformer_options={}, **kw):
                return x

        stub = Stub()
        node.install_h3_morton(stub)
        o2 = {"h3_segments": [(0, 8, "text")], "sol_h3_video_span": (8, 16),
              "sol_h3_audio_span": (4, 8)}
        stub._forward(x, None, None, transformer_options=o2)
        assert not any(k in o2 for k in ("h3_segments", "sol_h3_video_span", "sol_h3_audio_span")), o2
        obs.arm(None)

    def observer_only_block_indexing():
        d = newdir("indexing")
        obs.arm(f"dir={d}")
        import comfy.model_patcher

        holder = {}

        class Blk(torch.nn.Module):
            def forward(self, x, transformer_options=None):
                # the block's attention call, through the installed override
                ov = holder["override"]
                ov(dense_func, q, k, v, h, skip_reshape=True, skip_output_reshape=True,
                   transformer_options=transformer_options)
                return x

        class DiT(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([Blk() for _ in range(50)])

        class Sampling:
            def percent_to_sigma(self, p):
                return 1.0 - p

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.diffusion_model = DiT()
                self.model_sampling = Sampling()

        mp = comfy.model_patcher.ModelPatcher(Model(), torch.device("cpu"), torch.device("cpu"))
        result = node._apply_patch(mp, tau=1.0, start_percent=0.0, end_percent=1.0,
                                   min_tokens=64, sink_conditioning="off", morton=False,
                                   morton_curve="3d", dense_blocks="", verbose=False,
                                   tau_profile=None)
        patched = result.args[0] if hasattr(result, "args") else result[0]
        dit = patched.get_model_object("diffusion_model")
        assert id(dit) in node._BLOCK_INDEX_HOOKED, "armed patch did not install the block hooks"
        holder["override"] = patched.model_options["transformer_options"]["optimized_attention_override"]
        o = {"sigmas": torch.tensor([0.5]), "sample_sigmas": torch.tensor([1.0, 0.5, 0.0])}
        dit.blocks[7](torch.zeros(1), transformer_options=o)
        _, rows = rows_in(d)
        cfg = [r for r in rows if r["kind"] == "config"][0]
        assert cfg["settings"]["dense_blocks"] == [] and cfg["settings"]["tau_profile"] == {}
        assert cfg["settings"]["n_blocks"] == 50
        r = [r for r in rows if r["kind"] == "call"][0]
        assert r["block"] == 7 and r["scope"] == "dit" and r["route"] == "sol", (r["block"], r["route"])
        assert "sol_block" not in o
        obs.arm(None)

    def raw_off_writes_no_sidecar():
        d = newdir("rawoff")
        obs.arm(f"dir={d},raw=0")
        call(make(), opts())
        _, rows = rows_in(d)
        sol = [r for r in rows if r["kind"] == "call"][0]
        assert sol["route"] == "sol" and sol.get("raw") is None
        assert sol["kernel_density"] is not None and sol["per_head"]["kernel_mean"]
        assert not list(d.glob("*.u16")), "raw=0 still wrote a sidecar"
        assert rows[0]["raw_sidecar"] is False
        obs.arm(None)

    print("Sol route observer, against the installed kernel:")
    print(f"  B={b} H={h} T={t} ({n} blocks), dir {tmp}\n")
    try:
        for fn in (inert_without_the_env_var, every_route_is_recorded,
                   identity_does_not_mix_prompts, wrong_slice_is_red_and_escapes_the_fallback,
                   summaries_agree_with_an_independent_reduction,
                   armed_with_old_wheel_fails_at_patch_time, stale_block_label_is_cleared,
                   observer_only_block_indexing, raw_off_writes_no_sidecar):
            check(fn.__name__, fn)
    finally:
        obs.arm(None)
        restore()
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILED:
        print(f"FAILED: {', '.join(FAILED)}")
        return 1
    print("all cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
