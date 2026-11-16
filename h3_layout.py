"""MiniMax-H3's packed layout, read from what core publishes.

Core's `MiniMaxH3Model._forward` puts the `PackedLayout` it is about to use in
`transformer_options["minimax_h3_layout"]` before the token refiner runs, and
`transformer_options["block_index"]` before each DiT block
(`comfy/ldm/minimax/model.py`). Everything in this pack that needs segment
bounds or a block index reads them here.

**Why this module exists (2026-09-27).** Until then the Sol node got the same
facts by monkeypatching `PackedLayout.__init__` process-wide and replacing
`model._forward` and `model.rope_freqs` (the Morton machinery), and by hooking
every block to publish `sol_block`. Core publishes both now, so the patching
went with Morton (`docs/wiki/archive/2026-09-27_sol_node_redesign.md`).

**The one rule: a published value belongs to a call only when the call is as
long as the layout.** Core sets the layout before the two token-refiner calls
(which run on the text span alone) and never clears `block_index`, so the
next step's refiner calls see the previous step's last block. Gating on
`layout.seq_len == tokens` drops both: a refiner call gets no layout and no
block, which is what the old hooks produced too.
"""

import weakref


def layout_for(options, tokens=None):
    """The published `PackedLayout`, or None. With `tokens`, only when the call
    spans the whole packed sequence."""
    if not isinstance(options, dict):
        return None
    layout = options.get("minimax_h3_layout")
    if layout is None:
        return None
    if tokens is not None and getattr(layout, "seq_len", None) != int(tokens):
        return None
    return layout


def segments(options, tokens=None):
    """[(start, stop, kind), ...] in packed order, or None. Kinds are core's:
    text / cond / cond_audio / ref_img / ref_audio / audio / video, where
    `audio` and `video` are the TARGET streams."""
    layout = layout_for(options, tokens)
    segs = getattr(layout, "segments", None) if layout is not None else None
    if not segs:
        return None
    return [(int(a), int(b), str(kind)) for a, b, kind in segs]


def span(options, kind, tokens=None):
    """(start, stop) of the first segment of `kind`, or None."""
    for a, b, k in segments(options, tokens) or ():
        if k == kind:
            return a, b
    return None


def block_index(options, tokens):
    """The DiT block this call belongs to, or None (refiner calls, no layout,
    no index published)."""
    if layout_for(options, tokens) is None:
        return None
    index = options.get("block_index")
    return None if index is None else int(index)


#: Per layout, the `ref_img` spans that hold a video reference. Weak keys, so
#: a layout core has dropped takes its entry with it.
_VIDEO_REFS = weakref.WeakKeyDictionary()


def video_reference_spans(options, tokens=None):
    """frozenset of (start, stop) for the `ref_img` segments that are video
    references, not pictures. Core labels both `ref_img`
    (`comfy/ldm/minimax/model.py::PackedLayout`); what tells them apart is time:
    a picture's rows all sit at one time position (`_ref_t_span` gives an image
    1.0), a video reference's rows span its latent steps. Empty without a
    layout, or on a layout that publishes no `position_ids`."""
    layout = layout_for(options, tokens)
    if layout is None:
        return frozenset()
    try:
        cached = _VIDEO_REFS.get(layout)
    except TypeError:                       # not weak-referenceable (a test stand-in)
        cached = None
    if cached is not None:
        return cached
    pos = getattr(layout, "position_ids", None)
    spans = set()
    if pos is not None:
        for a, b, kind in segments(options, tokens) or ():
            if kind == "ref_img" and b - a > 1:
                t = pos[a:b, 0]
                if bool((t != t[0]).any()):
                    spans.add((a, b))
    spans = frozenset(spans)
    try:
        _VIDEO_REFS[layout] = spans
    except TypeError:
        pass
    return spans
