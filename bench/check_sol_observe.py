#!/usr/bin/env python3
"""Grade the Sol route observer before it is given a render.

The observer's value is that a row can be trusted to describe the call it
came from: the right prompt, the right block, the counts the kernel actually
produced, and a failure that stops the capture rather than thinning it. So
the cases are about identity, completeness and non-perturbation; the count
semantics themselves are pinned upstream (`tests/test_sol_attn.py` on our
comfy-kitchen build branch, `h3-build`) with closed-form equalities at both
tau extremes.

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
      would report a render as all-Sol. The kernel_error call RAISES out of the
      override with the kernel's exception as its cause and never reaches the
      dense fallback: a failed Sol call rendered dense is a render that
      succeeds and is not a Sol render.
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
  stale_block_index_is_not_trusted   (CPU)
      core publishes `block_index` before each DiT block and never clears it,
      so the next step's text-only refiner calls see the last block's index
      beside a layout they are shorter than. A refiner-shaped call under
      `block_index` 49 is recorded with no block, `scope: unknown` and no sink
      pair; the same options on a full-length call record block 49 and the
      layout's sink pair, which is the red control that the gate is not simply
      dropping every index. Replaced `stale_block_label_is_cleared`, whose
      subject (the node's own `sol_block` hooks and the Morton forward's
      `finally`) was deleted with the old node on 2026-09-27.
  apply_sol_records_core_block_and_fallback   (CPU)
      `_apply_sol` on a real ModelPatcher, armed, with empty dense_blocks:
      the installed override records the block index core published (7), and
      the config row names the node and its dense fallback. Then a foreign
      override is put on the hook, as an attention node placed after Sol
      would; the node's ON_PREPARE_STATE callback puts Sol back on top with
      the foreign override as its fallback (a short call reaches it), a second
      run is idempotent, and the next config row names the foreign override
      as `dense_fallback`. Red control: before the callback runs, the foreign
      override IS on top, so the case sees the takeover it exists to undo.
      Replaced `observer_only_block_indexing`, whose subject (the old node's
      block hooks) was deleted.
  raw_off_writes_no_sidecar
      `raw=0` leaves no `.u16` file and no raw pointer, and the row is
      otherwise complete.
  composed_patch_calls_are_recorded
      **the integration hole Codex's review found (2026-09-01).** On the
      canonical graph Sage's per-block forward patch sits under Sol, and
      Sol's composition gate hands a declined call straight to it, so the
      override -- and its recorder -- never runs for the 50 DiT calls of an
      outside-window step. Drives `_compose_module_patch` with a foreign
      forward: outside the window and below min_tokens the foreign forward
      runs, the stock forward does not, and one `route: composed_patch` row
      appears with the gate's verdict leading the reason. The full-length call
      carries core's block index; the short one (below min_tokens, so
      refiner-shaped) is shorter than the layout and records no block. Inside
      the window the stock forward runs and the wrapper writes nothing (the
      override records that call in a real render).
  forced_metadata_is_computed_not_inferred
      `forced.sink` is the clamped sink cardinality and `diag_min/max` the
      diagonal contribution outside sink_q: no sink gives 0 / 2 / 3, a sink
      overlapping the diagonal gives 4 / 0 / 3. The first revision reported
      the MINIMUM of the sum as the sink, which reads 2 with no sink at all.
  query_and_pair_weighting_differ_on_nonuniform_forced
      Codex's five-block fixture (counts all 4, no sink, forced [2,3,3,3,2]):
      the query-weighted `routed_density.mean` is 0.5667 and the
      pair-weighted `ordering_effect_density` is 0.5833, both to 1e-9, so the
      two names cannot be read as one number.
  render_row_names_the_workflow
      the first call under a prompt id writes a `render` row: the running
      graph is read from the server's queue, hashed as provenance.py hashes
      it, and matched to the shipped file -- `h3_text_to_video_api.json`
      submitted as-is names itself, carries `summary.scheduler` with its
      step count, and is `process_render_index` 0; a modified copy
      under a second prompt id gets a null `workflow_file` with a reason and
      index 1 with the first id as prior; without a server the row says the
      prompt was unavailable. This is what lets a reader tell a distilled record
      from a base one, and a cold render from a warm one, without the file.
  undefined_adaptive_figures_are_null
      every query block inside sink_q leaves no free pair: `routed_density`
      is null, `ordering_effect_density.overall` is null with numerator and
      denominator both zero, every per-head and per-segment adaptive value is
      null, and `kernel_density` is still defined. The first follow-up
      emitted {"weighting": "query"} with no mean there -- truthy, so a
      reader indexing `mean` would have raised (Codex's follow-up review).

The cases marked (CPU) run anywhere the node imports, and so do
`armed_with_old_wheel_fails_at_patch_time` and
`query_and_pair_weighting_differ_on_nonuniform_forced`. The rest need CUDA and
an installed comfy_kitchen whose `sol_attn` takes `blk_cnt`. Exit 0 all
passed, 1 a case failed, 2 the kernel cases were not graded (no CUDA, or no
`blk_cnt`) even when the CPU cases passed, rather than passing on a weaker
path.

The block index and segment bounds come from core
(`transformer_options["block_index"]` and `["minimax_h3_layout"]`, read
through `h3_layout`); the fixtures here stand in for both with a layout stub
whose `seq_len` is the call's length.

    <comfy-venv-python> bench/check_sol_observe.py
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


def layout(seq_len, segments):
    """Core's `minimax_h3_layout` as far as this pack reads it: `seq_len` and
    `segments` [(start, stop, kind)] in core's kinds."""
    return types.SimpleNamespace(seq_len=int(seq_len), segments=list(segments))


def main() -> int:
    try:
        import numpy as np
        import torch
        import comfy_kitchen as ck
        import _live_sol
        from _live_sol import live_sol, sol_observe
    except Exception as exc:                          # noqa: BLE001
        print(f"SKIP: needs torch, comfy_kitchen and the pack ({exc})")
        return 2
    cuda = torch.cuda.is_available()
    if not cuda:
        # ComfyUI's model management insists on a device at import unless told
        # to run on the CPU; the CPU cases need no device.
        if str(_live_sol.COMFY) not in sys.path:
            sys.path.append(str(_live_sol.COMFY))
        import comfy.cli_args
        comfy.cli_args.args.cpu = True
    blk_cnt = "blk_cnt" in inspect.signature(ck.sol_attn).parameters
    device = "cuda" if cuda else "cpu"

    node = live_sol()
    obs = sol_observe()
    from comfy_execution.utils import CurrentNodeContext

    torch.manual_seed(0)
    b, h, t, d = 1, 2, 1024 + 40, 128                # 17 key blocks, ragged tail
    n = (t + 63) // 64
    q, k, v = (torch.randn(b, h, t, d, device=device, dtype=torch.bfloat16)
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

    # The target video spans the whole call, so there are no conditioning rows
    # and no sink pair: what every case gets unless it passes `segments`.
    VIDEO_ONLY = [(0, t, "video")]
    # [text][audio][video]: exact_kv_and_rows gives sink blocks [0, 4) and
    # dense query blocks [2, 4).
    T2V_SEGS = [(0, 128, "text"), (128, 256, "audio"), (256, t, "video")]

    def opts(segments=None, **extra):
        """transformer_options as core fills them for a DiT block call:
        `block_index` 5 and a layout as long as the call."""
        o = {"sigmas": torch.tensor([1.0]), "sample_sigmas": torch.tensor([2.0, 1.0, 0.5, 0.0]),
             "block_index": 5,
             "minimax_h3_layout": layout(t, VIDEO_ONLY if segments is None else segments)}
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
        call(ov, opts(block_index=3))                                        # dense_block
        call(ov, opts(sigmas=torch.tensor([20.0])))                          # outside_range
        short = tuple(x[:, :, :32].contiguous() for x in (q, k, v))
        call(ov, opts(), qq=short[0], kk=short[1], vv=short[2])              # ineligible
        got = call(ov, opts())                                               # sol
        assert calls["dense"] == 4, f"dense fallback called {calls['dense']} times, want 4"
        synthetic = RuntimeError("synthetic kernel failure")
        use(Spy(raise_exc=synthetic))
        raised = None
        try:
            call(ov, opts())                                                 # kernel_error
        except RuntimeError as exc:
            raised = exc
        finally:
            restore()
        assert raised is not None, "a kernel failure did not raise out of the override"
        assert raised.__cause__ is synthetic, f"the raise does not carry the kernel's error: {raised!r}"
        assert calls["dense"] == 4, "the kernel failure reached the dense fallback"
        assert torch.equal(got, kernel(tau=1.0, tail=True)), "armed output differs from the kernel's bytes"
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
        assert sol["routed_density"]["weighting"] == "query"
        assert sol["ordering_effect_density"]["weighting"] == "pair"
        assert 0 <= sol["ordering_effect_density"]["overall"] <= 1.0
        assert all(r["path"] == "override" for r in callrows)
        # no sink: the named decomposition must say so, not report the edge diagonal as a sink
        assert (sol["forced"]["sink"], sol["forced"]["diag_min"], sol["forced"]["diag_max"]) == (0, 2, 3), sol["forced"]
        assert rows[0]["denominators"]["kernel_density"].startswith("cnt / NTB")
        # the dense_block row names the block; the short ineligible call is
        # shorter than the layout, so core's index is not trusted for it
        assert callrows[1]["reason"] == "block 3 in dense_blocks" and callrows[1]["block"] == 3
        assert callrows[3]["block"] is None, callrows[3]["block"]
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
        # video from row 256 -> sink blocks [0, 4); audio rows [128, 256) -> sink_q [2, 4)
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
                    call(ov, opts(segments=T2V_SEGS))
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
        segs = T2V_SEGS
        call(ov, opts(segments=segs))
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
        # the PAIR-weighted ratio, summed over free pairs the way analyze_routing.py does
        free = np.broadcast_to((n - forced)[None, None, :] * (~sinkq)[None, None, :], counts.shape)
        num = ((counts - forced[None, None, :]) * (~sinkq)[None, None, :]).sum()
        pair = num / free.sum()                      # both sums over every (b, h, q) outside sink_q
        oe = sol["ordering_effect_density"]
        assert abs(pair - oe["overall"]) < 1e-9, (pair, oe)
        assert oe["numerator"] == int(num) and oe["denominator"] == int(free.sum())
        for hh in range(h):
            ph = ((counts[:, hh] - forced[None, :]) * (~sinkq)[None, :]).sum() / free[:, hh].sum()
            assert abs(ph - sol["per_head"]["ordering_effect"][hh]) < 1e-9
        # forced decomposition with a sink that overlaps the diagonal
        assert (sol["forced"]["sink"], sol["forced"]["diag_min"], sol["forced"]["diag_max"]) == (4, 0, 3), sol["forced"]
        assert sol["forced"]["sink_range_clamped"] == [0, 4]
        # segments: overlap-weighted query-segment kernel density, recomputed
        assert [s["kind"] for s in sol["per_segment"]] == ["text", "audio", "video"]
        for seg, (a, bb, _kind) in zip(sol["per_segment"], segs):
            q0, q1 = a // 64, (bb - 1) // 64
            w = np.array([min(bb, (qb + 1) * 64) - max(a, qb * 64) for qb in range(q0, q1 + 1)], float)
            kd = (kernel[:, :, q0:q1 + 1] * w[None, None, :]).sum() / (w.sum() * b * h)
            assert abs(kd - seg["kernel"]) < 1e-9, (seg, kd)
            # pair-weighted with the same row weights, sink_q rows excluded
            live = (~sinkq)[q0:q1 + 1] * w
            snum = ((counts[:, :, q0:q1 + 1] - forced[None, None, q0:q1 + 1]) * live[None, None, :]).sum()
            sden = (np.broadcast_to((n - forced)[None, None, q0:q1 + 1] * live[None, None, :],
                                    counts[:, :, q0:q1 + 1].shape)).sum()   # over every head, like snum
            want = snum / sden if sden > 0 else None
            got = seg["ordering_effect"]
            assert (want is None and got is None) or abs(want - got) < 1e-9, (seg, want)
        assert sol["per_segment"][1]["routed"] is None and sol["per_segment"][1]["ordering_effect"] is None
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

    def stale_block_index_is_not_trusted():
        d = newdir("stale")
        obs.arm(f"dir={d}")
        ov = make(sink_conditioning="exact_kv_and_rows", dense_blocks=frozenset())
        # Core left the last block's index behind; the next step's refiner call
        # runs on the text span alone, so it is shorter than the layout.
        o = opts(segments=T2V_SEGS, block_index=49)
        short = tuple(x[:, :, :128].contiguous() for x in (q, k, v))
        call(ov, o, qq=short[0], kk=short[1], vv=short[2])
        # red control: the same options on a full-length call do carry the block
        call(ov, o)
        _, rows = rows_in(d)
        refiner, full = [r for r in rows if r["kind"] == "call"]
        assert refiner["T"] == 128 and full["T"] == t, (refiner["T"], full["T"])
        assert refiner["block"] is None and refiner["scope"] == "unknown", \
            (refiner["block"], refiner["scope"])
        assert refiner["sink_blocks"] == [0, 0] and refiner["sink_q"] == [0, 0], \
            "a call shorter than the layout was given the layout's sink pair"
        assert full["block"] == 49 and full["scope"] == "dit", (full["block"], full["scope"])
        assert full["sink_blocks"] == [0, 4] and full["sink_q"] == [2, 4], \
            (full["sink_blocks"], full["sink_q"])
        obs.arm(None)

    def apply_sol_records_core_block_and_fallback():
        d = newdir("apply")
        obs.arm(f"dir={d}")
        import comfy.model_patcher
        import comfy.patcher_extension

        holder = {}

        class Blk(torch.nn.Module):
            def forward(self, x, transformer_options=None):
                # the block's attention call, through whatever override is on the hook
                ov = transformer_options["optimized_attention_override"]
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
        result = node._apply_sol(mp, tau=1.0, quantizer="plain", dense_blocks="",
                                 sink_conditioning="off", token_routing=node.SOL_ROUTING_OFF,
                                 routing_blocks="", start_percent=0.0, end_percent=1.0,
                                 min_tokens=64, verbose=False)
        patched = result.args[0] if hasattr(result, "args") else result[0]
        dit = patched.get_model_object("diffusion_model")
        topts = patched.model_options["transformer_options"]
        sol = topts["optimized_attention_override"]
        o = dict(opts(block_index=7), optimized_attention_override=sol)
        dit.blocks[7](torch.zeros(1), transformer_options=o)
        _, rows = rows_in(d)
        cfg = [r for r in rows if r["kind"] == "config"]
        assert len(cfg) == 1, [r["kind"] for r in rows]
        s = cfg[0]["settings"]
        assert s["node"] == "MiniMaxH3Sol" and s["dense_blocks"] == [] and s["n_blocks"] == 50, s
        assert s["quantizer"] == "plain" and s["tail"] is True and s["topk_ratio"] == 0.0, s
        assert s["dense_fallback"].startswith("stock attention"), s["dense_fallback"]
        r = [r for r in rows if r["kind"] == "call"][0]
        want_route = "sol" if cuda else "ineligible"          # CPU tensors are ineligible
        assert r["block"] == 7 and r["scope"] == "dit" and r["route"] == want_route, \
            (r["block"], r["scope"], r["route"])

        # An attention node placed after Sol puts its override on top.
        seen = {"foreign": 0}

        def foreign_override(func, qq, kk, vv, heads, **kw):
            seen["foreign"] += 1
            return func(qq, kk, vv, heads, **kw)

        topts["optimized_attention_override"] = foreign_override
        assert topts["optimized_attention_override"] is foreign_override   # red control
        callbacks = patched.get_all_callbacks(comfy.patcher_extension.CallbacksMP.ON_PREPARE_STATE)
        assert callbacks, "_apply_sol registered no ON_PREPARE_STATE callback"
        for cb in callbacks:
            cb(patched, None, patched.model_options)
        top = topts["optimized_attention_override"]
        assert top is not foreign_override and top is not sol, \
            "the step callback did not put a fresh Sol override on top"
        for cb in callbacks:
            cb(patched, None, patched.model_options)
        assert topts["optimized_attention_override"] is top, "a second step re-wrapped Sol on top of itself"
        # a call Sol declines (shorter than min_tokens) now reaches the foreign override
        short = tuple(x[:, :, :32].contiguous() for x in (q, k, v))
        dense_before = calls["dense"]
        top(dense_func, *short, h, skip_reshape=True, skip_output_reshape=True,
            transformer_options=opts(block_index=7))
        assert seen["foreign"] == 1 and calls["dense"] == dense_before + 1, (seen, calls)
        _, rows = rows_in(d)
        cfg = [r for r in rows if r["kind"] == "config"]
        assert len(cfg) == 2, [r["kind"] for r in rows]
        assert "foreign_override" in cfg[1]["settings"]["dense_fallback"], cfg[1]["settings"]["dense_fallback"]
        assert topts["sol_compose"]["settings"]["dense_fallback"] == cfg[1]["settings"]["dense_fallback"]
        obs.arm(None)

    def composed_patch_calls_are_recorded():
        d = newdir("composed")
        obs.arm(f"dir={d}")
        seen = {"stock": 0, "patched": 0}

        class Attn(torch.nn.Module):
            heads = 2

            def forward(self, x, transformer_options=None):
                seen["stock"] += 1
                return x

        def patched(x, transformer_options=None):
            seen["patched"] += 1
            return x

        attn = Attn()
        wrapped = node._compose_module_patch(attn, patched)
        gate = {"sigma_start": 10.0, "sigma_end": 0.1, "min_tokens": 64, "settings": settings()}
        x = torch.zeros(t, 256, device="cuda", dtype=torch.bfloat16)
        base = {"sol_compose": gate, "sample_sigmas": torch.tensor([2.0, 1.0, 0.5, 0.0]),
                "block_index": 12, "minimax_h3_layout": layout(t, VIDEO_ONLY)}
        wrapped(x, transformer_options={**base, "sigmas": torch.tensor([20.0])})   # outside the window
        wrapped(x[:32], transformer_options={**base, "sigmas": torch.tensor([1.0])})  # below min_tokens
        assert seen == {"stock": 0, "patched": 2}, seen
        _, rows = rows_in(d)
        callrows = [r for r in rows if r["kind"] == "call"]
        assert len(callrows) == 2, [r["route"] for r in callrows]
        for r in callrows:
            assert r["route"] == "composed_patch" and r["path"] == "composed_patch", (r["route"], r["path"])
            assert r["H"] == 2 and r["B"] == 1
            assert r.get("raw") is None and "kernel_density" not in r
        # the full-length call carries core's index; the short one is shorter
        # than the layout, so core's (stale) index is not trusted for it
        assert (callrows[0]["block"], callrows[0]["scope"]) == (12, "dit"), callrows[0]["block"]
        assert (callrows[1]["block"], callrows[1]["scope"]) == (None, "unknown"), callrows[1]["block"]
        assert callrows[0]["reason"].startswith("outside_range: sigma 20"), callrows[0]["reason"]
        assert callrows[0]["T"] == t and callrows[0]["schedule"]["state"] == "no_match"
        assert callrows[1]["reason"] == "ineligible: seq 32 < 64", callrows[1]["reason"]
        # inside the window the gate takes the call: stock forward, and this wrapper writes nothing
        wrapped(x, transformer_options={**base, "sigmas": torch.tensor([1.0])})
        assert seen == {"stock": 1, "patched": 2}, seen
        _, rows2 = rows_in(d)
        assert len([r for r in rows2 if r["kind"] == "call"]) == 2
        # without settings in the gate there is nothing truthful to write, so nothing is
        wrapped(x, transformer_options={**base, "sol_compose": {k: v for k, v in gate.items() if k != "settings"},
                                        "sigmas": torch.tensor([20.0])})
        _, rows3 = rows_in(d)
        assert len([r for r in rows3 if r["kind"] == "call"]) == 2
        obs.arm(None)

    def forced_metadata_is_computed_not_inferred():
        # the pure function first, so the arithmetic is graded without a kernel
        f = obs.forced_counts(8, (0, 0), (0, 0))
        assert f.tolist() == [2, 3, 3, 3, 3, 3, 3, 2]
        f = obs.forced_counts(8, (0, 2), (0, 0))
        assert f.tolist() == [2, 3, 4, 5, 5, 5, 5, 4], f.tolist()      # diagonal shrinks where it meets the sink
        f = obs.forced_counts(8, (2, 40), (6, 7))
        # range clamped to the blocks that exist; rows 3-5 and 7 have their whole diagonal inside the sink
        assert f.tolist() == [8, 8, 7, 6, 6, 6, 8, 6], f.tolist()
        # and through a record: the row's named decomposition on a real kernel call
        d = newdir("forcedmeta")
        obs.arm(f"dir={d}")
        call(make(), opts())
        call(make(sink_conditioning="exact_kv_and_rows"),
             opts(segments=T2V_SEGS))
        _, rows = rows_in(d)
        a, bb = [r["forced"] for r in rows if r["kind"] == "call"]
        assert (a["sink"], a["diag_min"], a["diag_max"], a["rows_outside_sink_q"]) == (0, 2, 3, n), a
        assert (bb["sink"], bb["diag_min"], bb["diag_max"], bb["rows_outside_sink_q"]) == (4, 0, 3, n - 2), bb
        obs.arm(None)

    def query_and_pair_weighting_differ_on_nonuniform_forced():
        counts = torch.full((1, 1, 5), 4, dtype=torch.int32)
        forced = obs.forced_counts(5, (0, 0), (0, 0))
        assert forced.tolist() == [2, 3, 3, 3, 2]
        _kernel, adaptive = obs.densities(counts, forced, 5, (0, 0))
        query_weighted = float(adaptive.mean())
        pair, per_head, num, den = obs.ordering_effect(counts, forced, 5, (0, 0))
        assert abs(query_weighted - 0.5666666667) < 1e-9, query_weighted
        assert abs(pair - 0.5833333333) < 1e-9 and (num, den) == (7, 12), (pair, num, den)
        assert per_head == [pair]
        assert abs(query_weighted - pair) > 1e-3        # the fixture is nonuniform, so the two differ

    def undefined_adaptive_figures_are_null():
        d = newdir("allsinkq")
        obs.arm(f"dir={d}")
        # a layout whose video starts at the last row: every block is sink and every
        # query block is sink_q, so NTB - forced is zero on every row
        call(make(sink_conditioning="exact_kv_and_rows"),
             opts(segments=[(0, t, "audio"), (t, t, "video")]))
        _, rows = rows_in(d)
        r = [r for r in rows if r["kind"] == "call"][0]
        assert r["route"] == "sol" and r["sink_blocks"] == [0, n] and r["sink_q"] == [0, n], r["sink_q"]
        assert r["routed_density"] is None, r["routed_density"]
        oe = r["ordering_effect_density"]
        assert oe["overall"] is None and (oe["numerator"], oe["denominator"]) == (0, 0), oe
        assert r["per_head"]["routed_mean"] == [None] * h and r["per_head"]["ordering_effect"] == [None] * h
        assert r["per_segment"][0]["routed"] is None and r["per_segment"][0]["ordering_effect"] is None
        assert abs(r["kernel_density"]["mean"] - 1.0) < 1e-12 and r["forced"]["rows_outside_sink_q"] == 0
        assert r["forced"]["diag_min"] is None and r["forced"]["diag_max"] is None
        obs.arm(None)

    def render_row_names_the_workflow():
        import json
        d = newdir("render")
        obs.arm(f"dir={d}")
        wf = Path(node.__file__).resolve().parent / "workflows"
        shipped = json.loads((wf / "h3_text_to_video_api.json").read_text())
        modified = json.loads(json.dumps(shipped))
        modified["5"]["inputs"]["prompt"] = "a different prompt"
        stub = types.ModuleType("server")
        running = {0: (0, "prompt-W", shipped, {}, [], False), 1: (1, "prompt-X", modified, {}, [], False)}
        stub.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(
            prompt_queue=types.SimpleNamespace(currently_running=running)))
        had = sys.modules.get("server")
        sys.modules["server"] = stub
        try:
            with CurrentNodeContext("prompt-W", "10", None):
                call(make(), opts())
                call(make(), opts(block_index=6))        # second call: no second render row
            with CurrentNodeContext("prompt-X", "10", None):
                call(make(), opts())
        finally:
            if had is not None:
                sys.modules["server"] = had
            else:
                del sys.modules["server"]
        _, rows = rows_in(d)
        renders = [r for r in rows if r["kind"] == "render"]
        assert [r["prompt_id"] for r in renders] == ["prompt-W", "prompt-X"], renders
        w, x = renders
        assert w["workflow_file"] == "h3_text_to_video_api.json" and w["match"].startswith("shipped"), w
        assert w["process_render_index"] == 0 and w["prior_prompt_ids"] == []
        assert w["summary"]["pdd"] is None, w["summary"]["pdd"]
        assert w["summary"]["scheduler"] == {"scheduler": "simple", "steps": 16} and w["summary"]["sampler"] == "euler"
        assert w["summary"]["resolution"]["length"] == 362 and w["summary"]["sol_nodes"] == 1
        assert w["graph_sha256"] == obs.graph_sha256(shipped)
        assert x["workflow_file"] is None and x["match"].startswith("no shipped graph"), x
        assert x["process_render_index"] == 1 and x["prior_prompt_ids"] == ["prompt-W"]
        assert x["summary"]["scheduler"]["steps"] == 16  # the summary still describes the graph
        # the render row precedes its first call row
        seqs = {r["prompt_id"]: r["seq"] for r in renders}
        first_call = {}
        for r in rows:
            if r["kind"] == "call":
                first_call.setdefault(r["prompt_id"], r["seq"])
        assert all(seqs[p] < first_call[p] for p in seqs)
        obs.arm(None)
        # no server module at all: the row says so rather than guessing
        d2 = newdir("render_noserver")
        obs.arm(f"dir={d2}")
        had = sys.modules.pop("server", None)
        try:
            import builtins
            real_import = builtins.__import__

            def no_server(name, *a, **k):
                if name == "server":
                    raise ImportError("no server here")
                return real_import(name, *a, **k)
            builtins.__import__ = no_server
            with CurrentNodeContext("prompt-Y", "10", None):
                call(make(), opts())
        finally:
            builtins.__import__ = real_import
            if had is not None:
                sys.modules["server"] = had
        _, rows2 = rows_in(d2)
        y = [r for r in rows2 if r["kind"] == "render"][0]
        assert y["workflow_file"] is None and y["match"].startswith("prompt unavailable"), y["match"]
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

    # Graded anywhere the node imports: no kernel call decides them.
    cpu_cases = (armed_with_old_wheel_fails_at_patch_time,
                 query_and_pair_weighting_differ_on_nonuniform_forced,
                 stale_block_index_is_not_trusted,
                 apply_sol_records_core_block_and_fallback)
    kernel_cases = (inert_without_the_env_var, every_route_is_recorded,
                    identity_does_not_mix_prompts, wrong_slice_is_red_and_escapes_the_fallback,
                    summaries_agree_with_an_independent_reduction,
                    composed_patch_calls_are_recorded, forced_metadata_is_computed_not_inferred,
                    undefined_adaptive_figures_are_null, render_row_names_the_workflow,
                    raw_off_writes_no_sidecar)
    graded = cuda and blk_cnt
    print("Sol route observer" + (", against the installed kernel:" if graded else
                                  ", CPU cases only:"))
    print(f"  B={b} H={h} T={t} ({n} blocks) on {device}, dir {tmp}\n")
    try:
        for fn in cpu_cases + (kernel_cases if graded else ()):
            check(fn.__name__, fn)
    finally:
        obs.arm(None)
        restore()
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILED:
        print(f"FAILED: {', '.join(FAILED)}")
        return 1
    if not graded:
        why = ("no CUDA" if not cuda else
               "the installed comfy_kitchen.sol_attn has no blk_cnt; rebuild with "
               "vendor/rebuild_kernel.sh")
        print(f"SKIP: {len(kernel_cases)} kernel case(s) not graded ({why}); "
              f"the {len(cpu_cases)} CPU case(s) passed")
        return 2
    print("all cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
