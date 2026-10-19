#!/usr/bin/env python3
"""Grade our Sol node's DISPATCH against the algorithm's own eager reference.

## What this covers that `check_solattn_correctness.py` does not

That file grades the KERNEL against the algorithm's eager reference. This
grades everything our node (`MiniMaxH3Sol`) does AROUND the kernel: the
BHND-to-BTHD transpose and back, the scale it forwards, the sink pair it
derives from core's layout, and the selection it passes (tau, with the pooled
tail on and top-k off, always: `_TAIL`, `_TOPK_RATIO`). A defect in any of
those produces a plausible tensor of the right shape and a successful render.

The distinction is the reshape. `optimized_attention` hands H3's attention over
as BHND with `skip_reshape=True`, the kernel wants BTHD, and the output goes
back as BHND. Getting that wrong transposes heads against tokens -- which does
not raise, because both are legal sizes.

## The oracle is the KERNEL, not the algorithm, and the first draft got that wrong

This file was first written to compare the dispatch against the eager
reference, and the numbers looked like a marginal failure: cosine ~0.994 to
0.998 depending on shape and seed, sometimes under the bar. The instinct was
to loosen the bar. That would have been wrong twice over.

Measured instead: **the dispatch is BITWISE identical to a direct
`comfy_kitchen.sol_attn` call** on the same inputs. Every bit of that spread
was the kernel's INT8 arithmetic against fp32 -- which is not the node's doing,
is a property `check_solattn_correctness.py` already owns, and would have been
silently absorbed into a loosened tolerance here.

So the oracle is the kernel. That gives an exact claim rather than a tolerance,
it isolates the layer this file is about, and it needs no O(T^2) score tensor,
so it runs at a realistic sequence length instead of a toy one. **A tolerance
where an equality is available is a check that cannot see small defects.**

## Why it no longer compares against the vendored node

**It used to, and that comparison is finished rather than broken.** Until
2026-08-30 this file asserted that our forked node produced the SAME BYTES as
the vendored upstream one at the shipped settings, which is what made migrating
145 graphs safe. It passed, at both selections, and the result is recorded in
`bench/results/2026-08-30_sol_node_equivalence.json`.

That comparison cannot be re-run and should not be resurrected. `vendor/`
now holds the PRE-MERGE upstream drop, restored to be a pristine reference:
its `_run` passes `centroid_tail` to a kernel that no longer accepts it, so it
raises rather than producing a baseline. Keeping a check that can only skip
would be worse than none -- so the baseline moved to the algorithm, which is
the more durable control anyway and one this repo already trusts.

Claims, i.e. what breaks if a case is deleted:

  dispatch == kernel       the node's override (`make_override`, as
                           `_apply_sol` builds it) produces the SAME BYTES as
                           calling `comfy_kitchen.sol_attn` directly with the
                           transpose done by hand, the tail on, and the sink
                           pair `sink_ranges` gives for the layout core
                           published. Catches a transpose, a dropped scale, a
                           sink pair built wrong or not derived, or a
                           selection other than tau-with-tail. A dense
                           fallback that raises stands in for the chained
                           kernel, so a declined call fails the case rather
                           than matching by accident.
  sink pair reaches        a non-zero sink must change the output. The sink is
    the kernel             derived from H3's layout and passed through two
                           call frames; if it stopped arriving, every
                           conditioning row would be routed sparsely and the
                           render would merely look worse.
  tau reaches the kernel   RED CONTROL. tau is the node's one selection knob.
                           If a different tau does not move the output, it is
                           not connected and the equality above compares a
                           knob that does nothing. (Replaced the pooled_tail
                           control on 2026-09-27: the node no longer exposes
                           the tail, so its one live knob carries the control.
                           The top-k dispatch case went with top-k, which the
                           node no longer reaches.)
  (an OOM exits 2, not 1)  a resident model or a render in flight can leave
                           too little VRAM for these shapes. That is an
                           environment state, not a result. Until 2026-09-04
                           the guard for it wrapped nothing and the check
                           died on a raw traceback with exit 1; now the
                           kernel cases run under `graded_kernel_cases`,
                           which catches the allocator's error, names the
                           cases that were not graded, and exits 2.
                           `--oom-control` proves that with the card masked:
                           a kernel stub that raises on its first call must
                           come back as 2 with no case marked.

  a transposed oracle      RED CONTROL, and the one that earns this file.
    is caught              Compares against a kernel call with heads and tokens
                           swapped. If that still matches, the equality above
                           is not seeing layout at all.

  the sink pair per mode   CPU, pure, run BEFORE the CUDA gate. `_sink_blocks` on a
    (no kernel)            fixture layout for every `sink_conditioning` mode:
                           off is zeros; exact_kv has no dense-query range;
                           exact_kv_and_rows starts the range at the target
                           audio and leaves reference rows sparse;
                           exact_kv_and_all_rows covers every conditioning
                           row; the no-audio-span fallback of exact_kv_and_rows
                           IS the all-rows range; on a t2v-shaped layout the
                           two ranges differ by the text rows alone and on a
                           ref2v-shaped one by the reference rows; a missing
                           video span or a short sequence is zeros in every
                           mode; an unknown mode is refused; and the node's
                           combo lists exactly the modes the function accepts,
                           with the shipped default among them.

  the container entry      CPU, through core's own `wrap_attn`, run BEFORE the
    (no kernel)            CUDA gate. The override carries a
                           `container_function` exactly when its fallback
                           does. A dense block then hands core's containers
                           to the fallback's container entry untaken, and
                           the tensor is gone once that entry has consumed
                           it, which is the point: no frame of ours holds
                           it. RED CONTROL: the same override with the
                           attribute removed reaches the fallback's tensor
                           entry with the tensor still alive, so the case
                           can see a held reference. A call Sol takes gets
                           tensors, an ineligible one goes back to the
                           fallback in containers, and an armed probe's
                           fallback run gets the same q, k and v.
  container entry ==       on the card: a call Sol takes through the
    tensor entry           container entry is the same bytes as through the
                           tensor entry.

Needs CUDA and a comfy_kitchen carrying the merged `sol_attn` for the kernel
cases; the sink and container cases run anywhere the node imports. Exit 0 all passed, 1 a
case failed, 2 the kernel cases were not graded (no CUDA, no kernel, OOM),
even when the sink cases ran and passed.

    python bench/check_sol_node_equivalence.py
    CUDA_VISIBLE_DEVICES= python bench/check_sol_node_equivalence.py   # sink cases only, touches no card
    CUDA_VISIBLE_DEVICES= python bench/check_sol_node_equivalence.py --oom-control
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
COMFY = REPO.parent.parent


def load(name, path, package_dir=None):
    spec = importlib.util.spec_from_file_location(
        name, path,
        submodule_search_locations=[str(package_dir)] if package_dir else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cosine(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(a @ b / (a.norm() * b.norm()))


def sink_cases(node, check):
    """`_sink_blocks` per mode on fixture layouts. Pure, so no tensor and no
    kernel: the sink pair is derived from the layout core publishes
    (`minimax_h3_layout`, read through `h3_layout`), shaped as core builds it."""
    import types
    B = node.BLOCK_SIZE
    modes = node.SINK_CONDITIONING_MODES

    def layout(*segments):
        return {"minimax_h3_layout": types.SimpleNamespace(
            seq_len=segments[-1][1], segments=list(segments))}
    # [text][audio][video]: t2v-shaped, text immediately before the target audio
    t2v = layout((0, 320, "text"), (320, 640, "audio"), (640, 4096, "video"))
    # [text][ref][audio][video]: ref2v-shaped, reference rows between them
    ref = layout((0, 320, "text"), (320, 3840, "ref_img"), (3840, 4160, "audio"),
                 (4160, 8192, "video"))
    T2V, REF = 4096, 8192
    kv = (0, 10)                       # ceil(640 / 64) and ceil(4160 / 64) == 65
    kv_ref = (0, 65)

    def pair(opts, tokens, mode):
        return node._sink_blocks(opts, tokens, mode)

    check("off: zeros", pair(t2v, T2V, "off") == ((0, 0), (0, 0)))
    check("exact_kv: exact keys over conditioning, no dense queries",
          pair(t2v, T2V, "exact_kv") == (kv, (0, 0)) and pair(ref, REF, "exact_kv") == (kv_ref, (0, 0)))
    check("exact_kv_and_rows: dense queries from the target audio to the end of conditioning",
          pair(t2v, T2V, "exact_kv_and_rows") == (kv, (320 // B, 10))
          and pair(ref, REF, "exact_kv_and_rows") == (kv_ref, (3840 // B, 65)))
    check("exact_kv_and_all_rows: dense queries over every conditioning row",
          pair(t2v, T2V, "exact_kv_and_all_rows") == (kv, kv)
          and pair(ref, REF, "exact_kv_and_all_rows") == (kv_ref, kv_ref))
    no_audio_t2v = layout((0, 640, "text"), (640, 4096, "video"))
    check("exact_kv_and_rows without an audio span falls back to the all-rows range",
          pair(no_audio_t2v, T2V, "exact_kv_and_rows") == pair(t2v, T2V, "exact_kv_and_all_rows") == (kv, kv))
    rows_t2v = pair(t2v, T2V, "exact_kv_and_rows")[1]
    all_t2v = pair(t2v, T2V, "exact_kv_and_all_rows")[1]
    rows_ref = pair(ref, REF, "exact_kv_and_rows")[1]
    all_ref = pair(ref, REF, "exact_kv_and_all_rows")[1]
    text_blocks = 320 // B
    ref_blocks = (3840 - 320) // B
    check("the two dense ranges differ by the text rows on t2v and by the reference rows on ref2v",
          rows_t2v[0] - all_t2v[0] == text_blocks
          and rows_ref[0] - all_ref[0] == text_blocks + ref_blocks
          and rows_t2v[1] == all_t2v[1] and rows_ref[1] == all_ref[1],
          f"t2v {rows_t2v[0] - all_t2v[0]} blocks, ref2v {rows_ref[0] - all_ref[0]} blocks")
    check("no layout, or a call shorter than the layout (the text-only refiner), is zeros in every mode",
          all(pair({}, T2V, m) == ((0, 0), (0, 0)) for m in modes)
          and all(pair(t2v, 320, m) == ((0, 0), (0, 0)) for m in modes))
    try:
        pair(t2v, T2V, "exact_kv_rows"); refused = False
    except ValueError:
        refused = True
    check("an unknown mode is refused, not run as exact_kv", refused)
    try:
        schema = node.MiniMaxH3Sol.define_schema()
        combo = next(i for i in schema.inputs if i.id == "sink_conditioning")
        options, default = list(combo.options), combo.default
    except Exception as exc:                                  # noqa: BLE001
        print(f"  SKIP the combo's options   define_schema not readable here: {exc}")
    else:
        # The default is read from the node's constant, not spelled here: the
        # literal "exact_kv_and_rows" this case held went stale on 2026-10-02.
        check("the combo lists exactly the modes the function accepts, default among them",
              options == list(modes) and default in modes and default == node.SOL_SINK_DEFAULT,
              f"{options}, default {default}")


def policy_cases(node, check):
    """The per-segment row plan, the tau map it builds and the table loader, on CPU.

    `row_plan` is pure and `_tau_map` is tensor arithmetic with no kernel, so
    these run without a card. What they hold: the default input changes
    nothing, the three segment classes reproduce `sink_conditioning`'s two
    dense ranges where those can be expressed, a broken run becomes dense map
    columns, and a table that does not fit is refused."""
    import types
    B = node.BLOCK_SIZE
    ref = [(0, 320, "text"), (320, 3840, "ref_img"), (3840, 4160, "audio"), (4160, 8192, "video")]
    REF = 8192
    mode = "exact_kv_and_all_rows"
    kv = (0, 65)

    def plan(row_segments, segments=ref, tokens=REF, m=mode):
        return node.row_plan(segments, tokens, m, row_segments)

    def seg(text, reference, audio):
        return {"text": text, "reference": reference, "audio": audio}

    check("rows 'as sink_conditioning' is sink_ranges, with no dense columns",
          all(plan(None, m=m) == (*node.sink_ranges((4160, REF), (3840, 4160), REF, m), ())
              for m in node.SINK_CONDITIONING_MODES))
    check("text, reference and audio exact == exact_kv_and_all_rows",
          plan(seg(True, True, True)) == (kv, kv, ()))
    check("audio alone exact == exact_kv_and_rows",
          plan(seg(False, False, True)) == (*node.sink_ranges((4160, REF), (3840, 4160), REF,
                                                              "exact_kv_and_rows"), ()))
    check("none exact: no dense query rows, exact keys kept",
          plan(seg(False, False, False)) == (kv, (0, 0), ()))
    check("text alone, and text with references, are one run each, so they go as sink_q",
          plan(seg(True, False, False)) == (kv, (0, 320 // B), ())
          and plan(seg(True, True, False)) == (kv, (0, 3840 // B), ()))
    broken = plan(seg(True, False, True))
    want_cols = tuple(range(0, 320 // B)) + tuple(range(3840 // B, 65))
    check("text and audio exact with references routed: sink_q off, dense columns for both",
          broken == (kv, (0, 0), want_cols), f"{len(broken[2])} dense query blocks")
    check("the exact-keys side stays with sink_conditioning under per-segment rows",
          plan(seg(True, False, True), m="off")[0] == (0, 0)
          and plan(seg(True, False, True), m="exact_kv")[0] == kv)
    check("no layout, or a call shorter than the layout, is nothing exact",
          plan(seg(True, True, True), segments=None) == ((0, 0), (0, 0), ())
          and plan(seg(True, True, True), tokens=320) == ((0, 0), (0, 0), ()))
    kf = [(0, 300, "text"), (300, 1308, "cond"), (1308, 1400, "cond_audio"),
          (1400, 1600, "audio"), (1600, 8192, "video")]
    check("keyframe rows (cond, cond_audio) are reference rows; a block two classes share is exact if either is",
          node.exact_query_blocks(kf, REF, seg(False, True, False))
          == tuple(range(300 // B, (1400 + B - 1) // B))
          and all(node.segment_class(kind) == "reference"
                  for kind in ("cond", "cond_audio", "ref_img", "ref_audio")))

    heads, nq = 4, 16
    flat = node._tau_map(None, 1.25, heads, nq, (), "cpu")
    check("the map with no table row and no dense columns is tau everywhere",
          tuple(flat.shape) == (heads, nq) and flat.dtype == torch.float32
          and bool((flat == 1.25).all()))
    taus = (0.5, 1.0, 2.0, 4.0)
    tm = node._tau_map(taus, 1.25, heads, nq, (0, 1, 9), "cpu")
    by_hand = torch.tensor(taus).view(heads, 1).repeat(1, nq)
    by_hand[:, [0, 1, 9]] = node.DENSE_TAU
    check("a table row fills each head's row, and dense columns override it",
          torch.equal(tm, by_hand))
    try:
        node._tau_map((1.0, 1.0), 1.0, heads, nq, (), "cpu"); refused = False
    except RuntimeError:
        refused = True
    check("a table row of the wrong head count is refused at the call", refused)

    table = node._table
    fixtures = REPO / "bench" / "fixtures"
    t = table.load("sparse_table_synthetic.json", fixtures)
    check("the fixture table loads: heads, integer block keys, tuples of floats",
          t["heads"] == 56 and sorted(t["blocks"]) == [0, 24]
          and all(isinstance(r, tuple) and len(r) == 56 for r in t["blocks"].values()))
    other = table.parse({"schema": 1, "provenance": dict(t["provenance"]), "heads": 56,
                         "blocks": {"0": list(t["blocks"][0]), "24": [1.5] * 56}})
    check("a table's hash follows its values, not its name or provenance",
          len(t["sha256"]) == 64 and other["sha256"] != t["sha256"]
          and table.parse({"schema": 1, "provenance": {"calibrated_on": "a", "tool": "b", "date": "c"},
                           "heads": 56, "blocks": {str(b): list(r) for b, r in t["blocks"].items()}},
                          "renamed")["sha256"] == t["sha256"])
    check("no table ships yet, and the node offers 'none' first",
          table.list_tables() == [] and table.NONE == "none", f"{table.list_tables()}")

    def refuses(fn):
        try:
            fn()
        except table.SparseTableError:
            return True
        return False
    good = {"schema": 1, "provenance": {"calibrated_on": "x", "tool": "y", "date": "z"},
            "heads": 2, "blocks": {"0": [1.0, 2.0]}}
    bad = {
        "an unknown key": dict(good, note="hi"),
        "a missing provenance field": dict(good, provenance={"tool": "y", "date": "z"}),
        "a row of the wrong length": dict(good, blocks={"0": [1.0]}),
        "a negative tau": dict(good, blocks={"0": [1.0, -1.0]}),
        "a non-finite tau": dict(good, blocks={"0": [1.0, float("inf")]}),
        "a tau over the ceiling": dict(good, blocks={"0": [1.0, table.TAU_MAX + 1]}),
        "a block key that is not a plain integer": dict(good, blocks={"00": [1.0, 2.0]}),
        "another schema": dict(good, schema=2),
        "no blocks": dict(good, blocks={}),
    }
    held = [name for name, doc in bad.items() if not refuses(lambda d=doc: table.parse(d))]
    check("the loader refuses every malformed table, and takes the well-formed one",
          not held and table.parse(good)["blocks"] == {0: (1.0, 2.0)}, f"accepted: {held}" if held else "")
    check("a table for another head count, or naming a block the model lacks, is refused",
          refuses(lambda: table.require_fits(t, 48, 50))
          and refuses(lambda: table.require_fits(t, 56, 24))
          and table.require_fits(t, 56, 50) is None)
    check("a table name with a path in it is refused",
          refuses(lambda: table.load("../bench/fixtures/sparse_table_synthetic.json")))
    try:
        schema = node.MiniMaxH3Sol.define_schema()
        ids = [i.id for i in schema.inputs]
        rows = next(i for i in schema.inputs if i.id == "rows")
    except Exception as exc:                                  # noqa: BLE001
        print(f"  SKIP the new inputs' schema   define_schema not readable here: {exc}")
    else:
        check("tau_table and rows are the node's last two inputs, both optional, rows defaulting to 'as sink_conditioning'",
              ids[-2:] == ["tau_table", "rows"] and rows.optional
              and rows.options[0].key == node.ROWS_FOLLOW,
              f"{ids[-2:]}")


def container_cases(node, check):
    """The override's container entry, driven through core's `wrap_attn` with
    core's own container class. CPU tensors, so the kernel never runs: a call
    Sol would take is `_run` stubbed, or ineligible ("not cuda")."""
    import gc
    import types
    import weakref
    from comfy.ldm.modules.attention import AttentionTensorContainer as Box, wrap_attn

    h, t, d = 2, 256, node.HEAD_DIM
    seen = {}

    def consume(entry, tensors):
        """What a dense backend does with q, k and v: use them, drop them."""
        ref = weakref.ref(tensors[0])
        out = tensors[0] * 2
        seen.update(entry=entry, k=tensors[1].clone(), v=tensors[2].clone())
        del tensors[:]
        gc.collect()
        seen["freed"] = ref() is None
        return out

    def fallback(func, q, k, v, heads, **kw):
        tensors = [q, k, v]
        del q, k, v
        return consume("tensors", tensors)

    def fallback_containers(q, k, v, heads, **kw):
        if any(torch.is_tensor(x) for x in (q, k, v)):
            raise AssertionError("the container entry was handed a bare tensor")
        return consume("containers", [q.take(), k.take(), v.take()])

    fallback.container_function = fallback_containers

    def bare(func, q, k, v, heads, **kw):
        return q * 2

    @wrap_attn
    def stock(q, k, v, heads, **kw):
        raise AssertionError("the stock attention ran: the override was skipped")

    def make(previous, dense_blocks=frozenset()):
        return node.make_override(tau=1.0, min_tokens=0, sink_conditioning="off",
                                  dense_blocks=dense_blocks, previous=previous)

    def call(override):
        """One attention call as core's H3 forward makes it. Returns what came
        back, what it should be from `consume`, the inputs and the route."""
        torch.manual_seed(0)
        boxes = [Box(torch.randn(1, h, t, d)) for _ in range(3)]
        q, k, v = (b.peek().clone() for b in boxes)
        options = {"minimax_h3_layout": types.SimpleNamespace(
                       seq_len=t, segments=[(0, 64, "text"), (64, t, "video")]),
                   "block_index": 3, "optimized_attention_override": override}
        seen.clear()
        out = stock(*boxes, h, mask=None, skip_reshape=True, transformer_options=options)
        same = (torch.equal(out, q * 2) and torch.equal(seen.get("k", out), k)
                and torch.equal(seen.get("v", out), v))
        return same, options.get("h3_attn_route")

    check("the override has a container entry exactly when its fallback has one",
          hasattr(make(fallback), "container_function")
          and not hasattr(make(bare), "container_function")
          and not hasattr(make(None), "container_function"))

    same, route = call(make(fallback, dense_blocks=frozenset({3})))
    check("a dense block hands the containers on, and the fallback frees q",
          same and route == "dense_block" and seen["entry"] == "containers" and seen["freed"],
          f"route {route}, fallback entry {seen.get('entry')}, freed {seen.get('freed')}")

    tensor_only = make(fallback, dense_blocks=frozenset({3}))
    del tensor_only.container_function
    same, route = call(tensor_only)
    check("RED CONTROL: without the container entry q is still held",
          same and route == "dense_block" and seen["entry"] == "tensors" and not seen["freed"],
          f"route {route}, fallback entry {seen.get('entry')}, freed {seen.get('freed')}")

    same, route = call(make(fallback))
    check("an ineligible call goes back to the fallback in containers",
          same and route == "ineligible" and seen["entry"] == "containers" and seen["freed"],
          f"route {route}, fallback entry {seen.get('entry')}, freed {seen.get('freed')}")

    # A call Sol takes, with the probe armed: `_run` must get tensors, and the
    # probe's fallback run the same q, k and v after it.
    ran = {}

    def run(q, k, v, *a, **kw):
        ran["tensors"] = all(torch.is_tensor(x) for x in (q, k, v))
        return q * 3

    saved = node._run, node._probe
    node._run = run
    node._probe = types.SimpleNamespace(
        enabled=lambda: True, skip=lambda **kw: None,
        compare=lambda out, dense_fn, **kw: dense_fn())
    try:
        same, route = call(make(fallback))
    finally:
        node._run, node._probe = saved
    check("a call Sol takes gets tensors, and an armed probe's fallback the same q, k, v",
          same and route == "sol" and ran.get("tensors") and seen["entry"] == "containers",
          f"route {route}, kernel got tensors {ran.get('tensors')}, "
          f"fallback entry {seen.get('entry')}")


def load_node():
    """The Sol node module. With a card, through the pack's entrypoint, as the
    server loads it. Without one, the single module under a bare package:
    the entrypoint imports ComfyUI's model management, which insists on a
    device at import, and the sink cases need no device."""
    if torch.cuda.is_available():
        load("h3x", REPO / "__init__.py", package_dir=REPO)
    else:
        import types
        import comfy.cli_args as cli_args
        cli_args.args.cpu = True
        pkg = types.ModuleType("h3x")
        pkg.__path__ = [str(REPO)]
        sys.modules["h3x"] = pkg
    return load("h3x.sol_attn_h3", REPO / "sol_attn_h3.py")

# Every kernel case by name, in the order they run, so an OOM can say which
# were not graded rather than "every case".
KERNEL_CASES = (
    "dispatch == kernel through the node's override",
    "container entry == tensor entry on a call Sol takes",
    "dispatch with blk_cnt == dispatch without",
    "sink pair reaches the kernel",
    "default rows and no table pass no tau_map",
    "a table of equal taus == the scalar call",
    "a per-head table == each head in its own scalar call",
    "text and audio exact, references routed, through the override",
    "tau reaches the kernel",
    "a transposed oracle is caught",
)


def kernel_cases(node, ck, check, device="cuda", shape=(1, 8, 16384, 128)):
    """The dispatch against the kernel call it should be making, bitwise."""
    torch.manual_seed(0)
    # A realistic length, which the kernel oracle allows and an O(T^2) eager
    # oracle would not.
    b, h, t, d = shape
    # BHND, which is how `optimized_attention` hands H3's attention over.
    q, k, v = (torch.randn(b, h, t, d, device=device, dtype=torch.bfloat16)
               for _ in range(3))
    common = dict(skip_reshape=True, skip_output_reshape=True, scale=None,
                  min_tokens=12288, verbose=False,
                  topk_ratio=node._TOPK_RATIO, tail=node._TAIL)

    def dispatch(**kw):
        return node._run(q, k, v, h, **{**common, "tau": 1.0, **kw})

    def kernel(transpose_oracle=False, **kw):
        """The kernel call the dispatch should be making, done by hand."""
        qs, ks, vs = (x.transpose(1, 2).contiguous() for x in (q, k, v))
        if transpose_oracle:                      # heads against tokens
            qs, ks, vs = (x.transpose(1, 2).contiguous() for x in (qs, ks, vs))
        out = ck.sol_attn(qs, ks, vs, **kw)
        return out.transpose(1, 2)

    print("our node's dispatch against the kernel call it should be making:")
    print(f"  B={b} H={h} T={t} D={d} bf16, bitwise\n")

    # Through the override the node installs, on a layout shaped as core
    # publishes it: [text][audio][video], so the sink is derived, not passed.
    import types
    segments = [(0, 320, "text"), (320, 640, "audio"), (640, t, "video")]
    options = {"minimax_h3_layout": types.SimpleNamespace(seq_len=t, segments=segments),
               "block_index": 0}
    mode = "exact_kv_and_rows"
    sink_kv, sink_q = node.sink_ranges((640, t), (320, 640), t, mode)

    def declined(*a, **kw):
        raise AssertionError("the override declined the call and ran the dense fallback")

    override = node.make_override(tau=1.0, min_tokens=12288, sink_conditioning=mode,
                                  qk_balance=False, rotate=False)
    try:
        got = override(declined, q, k, v, h, skip_reshape=True, skip_output_reshape=True,
                       transformer_options=options)
    except torch.OutOfMemoryError:
        raise
    except Exception as exc:                                  # noqa: BLE001
        # The override wraps a kernel failure in a RuntimeError; an OOM under
        # it is still the busy-card state `graded_kernel_cases` reports.
        if isinstance(exc.__cause__, torch.OutOfMemoryError):
            raise exc.__cause__
        check("dispatch == kernel through the node's override", False,
              f"{type(exc).__name__}: {exc}")
    else:
        want = kernel(tau=1.0, tail=True, sink_blocks=list(sink_kv), sink_q=list(sink_q))
        same = torch.equal(got, want) and sink_kv != (0, 0)
        check("dispatch == kernel through the node's override", same,
              f"same bytes, sink {sink_kv} dense rows {sink_q}" if same else
              f"DIFFER: max abs {float((got.float() - want.float()).abs().max()):.3e}, "
              f"sink {sink_kv}")

        # The same call through the container entry, which exists only over
        # a fallback that has one. Clones, because the entry takes them.
        from comfy.ldm.modules.attention import AttentionTensorContainer as Box
        declined.container_function = declined
        boxed = node.make_override(tau=1.0, min_tokens=12288, sink_conditioning=mode,
                                   qk_balance=False, rotate=False, previous=declined)
        via = boxed.container_function(Box(q.clone()), Box(k.clone()), Box(v.clone()), h,
                                       skip_reshape=True, skip_output_reshape=True,
                                       transformer_options=dict(options))
        check("container entry == tensor entry on a call Sol takes", torch.equal(via, got),
              "same bytes")

    # The observer's passthrough: a count buffer handed to `_run` reaches the
    # kernel, comes back bounded, and moves no byte of the output. Graded here
    # rather than in check_sol_observe.py because THIS file owns "dispatch ==
    # kernel", and observation is a second way to call the dispatch.
    n = (t + 63) // 64
    buf = torch.empty(b, h, n, dtype=torch.int32, device=device)
    with_counts = dispatch(tau=1.0, blk_cnt=buf)
    check("dispatch with blk_cnt == dispatch without",
          torch.equal(with_counts, dispatch(tau=1.0))
          and 1 <= int(buf.min()) and int(buf.max()) <= n,
          f"same bytes; counts in [{int(buf.min())}, {int(buf.max())}] of {n} blocks")

    sink = dispatch(tau=1.0, sink_blocks=(0, 4), sink_q=(0, 4))
    check("sink pair reaches the kernel",
          torch.equal(sink, kernel(tau=1.0, tail=True,
                                   sink_blocks=[0, 4], sink_q=[0, 4]))
          and not torch.equal(sink, dispatch(tau=1.0)),
          "a non-zero sink both arrives and changes the output")

    # The per-head table and the per-segment rows, through the override: what
    # reaches the kernel, and that the default reaches it unchanged.
    seen = {}
    real = ck.sol_attn

    def spy(*a, **kw):
        seen["kwargs"] = dict(kw)
        return real(*a, **kw)

    ref_segments = [(0, 320, "text"), (320, 1920, "ref_img"), (1920, 2240, "audio"), (2240, t, "video")]
    ref_options = {"minimax_h3_layout": types.SimpleNamespace(seq_len=t, segments=ref_segments),
                   "block_index": 0}

    def through(**kw):
        ov = node.make_override(tau=1.0, min_tokens=12288, qk_balance=False, rotate=False,
                                sink_conditioning="exact_kv_and_all_rows", **kw)
        seen.clear()
        node._ck.sol_attn = spy
        try:
            return ov(declined, q, k, v, h, skip_reshape=True, skip_output_reshape=True,
                      transformer_options=dict(ref_options))
        finally:
            node._ck.sol_attn = real

    plain = through()
    check("default rows and no table pass no tau_map", "tau_map" not in seen["kwargs"],
          f"kernel keywords {sorted(seen['kwargs'])}")
    all_exact = through(row_segments={"text": True, "reference": True, "audio": True})
    check("default rows and no table pass no tau_map",
          "tau_map" not in seen["kwargs"] and torch.equal(all_exact, plain),
          "per segment with all three exact: same bytes, still no tau_map")
    equal_table = through(tau_table={"name": "equal", "sha256": ""}, table_blocks={0: (1.0,) * h})
    check("a table of equal taus == the scalar call",
          "tau_map" in seen["kwargs"] and torch.equal(equal_table, plain), "same bytes")
    taus = tuple((0.5, 1.0, 2.0)[i % 3] for i in range(h))
    per_head = through(tau_table={"name": "per_head", "sha256": ""}, table_blocks={0: taus})
    ok = True
    for value in (0.5, 2.0):
        alone = kernel(tau=value, tail=True, sink_blocks=[0, 35], sink_q=[0, 35])
        hs = [i for i in range(h) if taus[i] == value]
        ok = ok and torch.equal(per_head[:, hs], alone[:, hs])
    check("a per-head table == each head in its own scalar call",
          ok and not torch.equal(per_head, plain), "same bytes per head; differs from one tau")
    other_block = through(tau_table={"name": "per_head", "sha256": ""}, table_blocks={7: taus})
    check("a per-head table == each head in its own scalar call",
          torch.equal(other_block, plain) and "tau_map" not in seen["kwargs"],
          "a block the table does not list runs the scalar call")
    cnt = torch.zeros(b, h, n, dtype=torch.int32, device=device)
    real_run = node._run

    def counting(*a, **kw):
        kw["blk_cnt"] = cnt
        return real_run(*a, **kw)

    node._run = counting
    try:
        split = through(row_segments={"text": True, "reference": False, "audio": True})
    finally:
        node._run = real_run
    text_b, ref_b, audio_b = range(0, 5), range(5, 30), range(30, 35)
    check("text and audio exact, references routed, through the override",
          seen["kwargs"].get("sink_q") == [0, 0] and "tau_map" in seen["kwargs"]
          and bool((cnt[0][:, list(text_b)] == n).all()) and bool((cnt[0][:, list(audio_b)] == n).all())
          and int(cnt[0][:, list(ref_b)].max()) < n and not torch.equal(split, plain),
          f"text and audio query blocks route all {n} key blocks; reference query blocks at most "
          f"{int(cnt[0][:, list(ref_b)].max())}")

    print("\nred controls:")
    base = dispatch(tau=1.0)
    other = dispatch(tau=2.0)
    moved = not torch.equal(base, other)
    c = cosine(base, other)
    check("tau reaches the kernel", moved,
          f"cos {c:.6f} between tau 1.0 and 2.0 -- connected" if moved else
          "a different tau changed nothing; it is not reaching the kernel and "
          "every case above is vacuous")

    swapped = kernel(transpose_oracle=True, tau=1.0, tail=True)
    caught = not (swapped.shape == base.shape and torch.equal(base, swapped))
    check("a transposed oracle is caught", caught,
          "a heads/tokens swap does not match, so the equality above is "
          "actually seeing layout"
          if caught else
          "a transposed oracle still matches; this file cannot see the defect "
          "class it exists for")


def graded_kernel_cases(node, ck, check, **kw):
    """Run the kernel cases; 0 when they all ran, 2 when the card was busy.

    **An OOM here is not a failure and must not print as one.** This box runs
    a resident ComfyUI, so a model left loaded from a render leaves under a
    GiB free while these shapes want about two, and a render in flight leaves
    less. Before 2026-09-04 the guard for this wrapped nothing, so the check
    died on a torch traceback with exit 1, which reads exactly like a real
    mismatch -- and a check that goes red while the state is correct trains a
    reader to ignore red, which is the one thing docs/method/checks.md says is worse
    than having no check. Exit 2, naming the cases that were not graded and
    the fix. The sink cases before this point stand on their own result."""
    graded = []

    def recording(name, ok, detail=""):
        graded.append(name)
        check(name, ok, detail)

    try:
        kernel_cases(node, ck, recording, **kw)
    except torch.OutOfMemoryError as exc:
        missing = [c for c in KERNEL_CASES if c not in graded]
        try:
            free, total = torch.cuda.mem_get_info()
            vram = f"{free / 2**30:.2f} GiB free of {total / 2**30:.2f}"
        except Exception:                                    # noqa: BLE001
            vram = "VRAM figures unavailable"
        print(f"\n  SKIP  the card was busy: {type(exc).__name__} ({vram}).\n"
              f"        This is an environment state, not a result. A resident model or a\n"
              f"        render in flight is the usual cause; free the card (POST /free with\n"
              f"        unload_models, or wait for the render), then re-run.\n"
              f"        Not graded ({len(missing)} of {len(KERNEL_CASES)}): {', '.join(missing)}")
        return 2
    return 0


def oom_control(node):
    """RED CONTROL for the wrapper, with the card masked: a kernel that raises
    the allocator's error on its first call must come back as 2 with every
    kernel case named as not graded, and no case marked ok or FAIL."""
    class Busy:
        @staticmethod
        def sol_attn(*a, **kw):
            raise torch.OutOfMemoryError("fixture: CUDA out of memory")

    def run(*a, **kw):
        raise torch.OutOfMemoryError("fixture: CUDA out of memory")

    saved = node._run
    node._run = run
    marks = []
    try:
        rc = graded_kernel_cases(node, Busy, lambda n, ok, d="": marks.append(n),
                                 device="cpu", shape=(1, 2, 256, 128))
    except torch.OutOfMemoryError:
        rc = "escaped"                     # the wrapper let the allocator's error through
    finally:
        node._run = saved
    ok = rc == 2 and not marks
    print(f"  {'ok  ' if ok else 'FAIL'} a busy card exits 2 with nothing marked"
          + ("" if ok else f"   rc {rc}, marked {marks}"))
    return 0 if ok else 1


def main():
    sys.path.insert(0, str(COMFY))
    sys.path.insert(0, str(REPO / "bench"))

    node = load_node()

    failures = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    print("the sink pair per mode, on CPU:")
    sink_cases(node, check)
    print()

    print("the per-segment rows, the tau map and the table loader, on CPU:")
    policy_cases(node, check)
    print()

    print("the container entry, on CPU through core's wrap_attn:")
    container_cases(node, check)
    print()

    if "--oom-control" in sys.argv[1:]:
        print("the OOM wrapper, card masked:")
        return oom_control(node) or (1 if failures else 0)

    if not torch.cuda.is_available():
        print("no CUDA; the kernel cannot run. The kernel cases were not graded.")
        if failures:
            print(f"FAILED: {len(failures)} case(s): {', '.join(failures)}")
            return 1
        return 2
    import comfy_kitchen as ck                              # noqa: E402
    if not hasattr(ck, "sol_attn"):
        print("this comfy_kitchen has no sol_attn; the kernel cases were not graded.")
        return 1 if failures else 2

    rc = graded_kernel_cases(node, ck, check, device="cuda")
    if rc == 2:
        return 1 if failures else 2

    print()
    if failures:
        print(f"FAILED: {len(failures)} case(s): {', '.join(failures)}")
        return 1
    print("all cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
