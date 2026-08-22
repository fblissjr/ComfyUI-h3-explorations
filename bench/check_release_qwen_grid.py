#!/usr/bin/env python3
"""The release Qwen video grid, pinned at the boundaries that were got wrong.

Run it with the ComfyUI venv python (`docs/comfy_notes.md`). CPU only, no
model, no server, no GPU.

**The escaped instance, and there are three of it in one day.** On 2026-08-22
the release's duration-aware Qwen resize was computed by hand three times and
was wrong three times: 1152x640 twice (from passing the repeat-padded sampled
count where the release passes the raw one) and a 288-frame activation
threshold (from reading `budget // pixels_per_frame` as the first count that
bites rather than the last that fits, then mapping it backwards). Each was
plausible, each was near the right answer, and nothing in the repo could have
caught any of them. `bench/preflight_graph.py` now reports these numbers on
every graph, so a silent regression would ship into a report people read.

**What it guards, in one sentence:** that the release grid comes from the
release processor, at the RAW sampled count.

The raw-vs-padded guard is the load-bearing one and it is not cosmetic. At
1344x768, 31 raw gives `[16, 42, 74]` and 32 padded gives `[16, 40, 72]` --
the same 16 temporal blocks, an 8% difference in rows, because the clip-wide
pixel budget divides by whatever frame count it was handed. ComfyUI's
pad-to-even happens after sampling and the release never sees it. Case 2 below
fails if `preflight_graph._release_qwen_grid` is ever "fixed" to pass the
padded count.

**Every expectation here is a released artifact's behaviour, not a number this
file computed.** The processor is constructed from `vendor_config/`, which is
the release's own config verbatim.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from preflight_graph import _release_qwen_grid  # noqa: E402

failures: list[str] = []


def check(name, fn):
    try:
        fn()
    except AssertionError as e:
        failures.append(name)
        print(f"  FAIL  {name}: {e}")
    except Exception as e:
        failures.append(name)
        print(f"  ERROR {name}: {type(e).__name__}: {e}")
    else:
        print(f"  ok    {name}")


def grid(n, w, h):
    r = _release_qwen_grid(n, w, h)
    assert r is not None, ("the release processor could not be reached -- this "
                           "check cannot pass by being unable to run")
    g, per = r
    return g, per, g[2] * 16, g[1] * 16


def boundary_1344():
    """25 sampled is the last that fits; 26 is the first that resizes."""
    for n, want in ((24, (1344, 768)), (25, (1344, 768)), (26, (1280, 736))):
        _, _, w, h = grid(n, 1344, 768)
        assert (w, h) == want, f"{n} sampled -> {w}x{h}, expected {want}"


def boundary_960():
    """Our source size: 49 fits, 50 resizes -- and 50 is out of legal reach."""
    for n, want in ((49, (960, 544)), (50, (928, 512))):
        _, _, w, h = grid(n, 960, 544)
        assert (w, h) == want, f"{n} sampled -> {w}x{h}, expected {want}"


def raw_not_padded():
    """The guard. 31 and 32 must NOT agree, or the distinction has been lost."""
    a = grid(31, 1344, 768)
    b = grid(32, 1344, 768)
    assert (a[2], a[3]) == (1184, 672), f"31 raw -> {a[2]}x{a[3]}"
    assert (b[2], b[3]) == (1152, 640), f"32 padded -> {b[2]}x{b[3]}"
    assert a[0][0] == b[0][0] == 16, "both should be 16 temporal blocks"
    assert a[1] != b[1], ("raw and padded produced the same rows, so passing "
                          "the padded count would no longer be detectable")


def legal_target_lengths():
    """294 and 311 are consecutive legal 17n+5 lengths astride the boundary."""
    for frames, resized in ((294, False), (311, True)):
        raw = len(range(0, frames, 12))
        _, _, w, h = grid(raw, 1344, 768)
        got = (w, h) != (1344, 768)
        assert got == resized, (
            f"{frames} frames ({raw} sampled) -> {w}x{h}; expected "
            f"{'a resize' if resized else 'no resize'}")


def shipped_row_counts():
    """The two lengths this repo actually renders."""
    for frames, want in ((124, 6048), (362, 12432)):
        raw = len(range(0, frames, 12))
        g, per, _, _ = grid(raw, 1344, 768)
        assert per * g[0] == want, f"{frames} frames -> {per * g[0]:,}, expected {want:,}"


def unreachable_reports_none():
    """A processor that cannot run must return None, never a local guess."""
    r = _release_qwen_grid(31, 0, 0)
    assert r is None, f"a zero-sized input returned {r} instead of None"


def main() -> int:
    print("release Qwen video grid, from vendor_config/ via the real processor\n")
    check("1344x768: 25 fits, 26 resizes", boundary_1344)
    check("960x544: 49 fits, 50 resizes", boundary_960)
    check("raw sampled count, not repeat-padded", raw_not_padded)
    check("294 unchanged, 311 first resized", legal_target_lengths)
    check("shipped lengths: 124 -> 6,048 rows, 362 -> 12,432", shipped_row_counts)
    check("unreachable input reports None, not a substitute",
          unreachable_reports_none)
    print(f"\n{len(failures)} failure(s)" if failures else
          "\nall ok -- the release grid is read off the release processor")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
