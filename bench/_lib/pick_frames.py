"""Which frames of a span a reader looks at: every frame where something changes, the frames either side, and one a second.

    from _lib import pick_frames as pf
    picked = pf.pick(first, last, series={"track_px": [...], "mouth_mode": [...]}, spans=[...], events=[...])

**What it buys.** A sheet of evenly spaced frames shows a short shot at a few fixed places, and what happens between
two of them is never on the sheet. On second_hfp_5889e (5889-5934, 2026-10-21) a four-frame sheet (5889, 5904, 5919,
5934) showed none of 5915-5918: the sapiens face count fell at 5916, and she leaves the frame between 5917 and 5918
(the owner, who saw her still there and swapped on 5916 and 5917). The owner's word that day: deltas are what
matter. So the frames are picked where the figures a tool already holds CHANGE, each with the frame before and the
frame after, and a floor of one a second fills the rest. It is selective, because a list of every frame is no list:
a value under a share of the subject's own usual size reads as nothing, a new level must hold, every change is
scored, and past what was written down and the islands, only the strongest change of each second is picked; the
rest are listed with their scores, so nothing is hidden. A video reader (which watches the clip, then steps to listed
frames) and a stills reader (which reads a sheet) are given the same list.

One function, `pick`, two kinds of caller. A tool that writes a reference's sheet (`bench/crop_reference_to_head.py`,
`bench/draw_face_points.py`, `bench/draw_lid_marks.py`) feeds it what it holds before any capture exists: the face's
points, the mouth it drew, the frames it left bare. `bench/pick_frames.py` feeds it what a capture holds: track,
part, region and render figures, flags, a person's statements, the owner's boxes.

**The inputs**, all on a span's clock (`first` to `last`, source frame numbers; a list's item i is frame first + i):

- `series`: name to a list of values, or to {"values": [...], "rules": {...}, "line": x, "record": bool}. A value of None is not
  measured on that frame and no change is read across it. Numbers are AMOUNTS (rules `zero` and `step` by default;
  `line` and `peak` on request), strings and booleans are LABELS (any change of value), and (x, y, size) triples
  are PLACES (a jump, or a place found or lost). A series given `record` (voice on or off) is picked like a span.
- `spans`: [{"name", "frames": [a, b], "detail"?, "record"?, "score"?}]: a span's first frame and the frame after its
  last are changes, picked on their own frame (what was written down says where). A span given `record: False` is a
  candidate instead, scored `SPAN_SCORE` unless it says.
- `events`: [{"name", "frame", "detail"?}]: a frame somebody or something named.

**What it gives**: `frames`, a list of {source_frame, clip_frame, clip_seconds, reasons} in order, where clip_frame
is the frame's place in the span's clip and clip_seconds that place in seconds at the lane's rate (where a video
reader seeks); `changes`, every change picked, with its score; `also_changed_not_picked`, every measured change the
second's room left out, with its score; `dropped`, the measured changes a cap left out, with their scores; `cap`,
what the cap did; `rules`, the values used.

**Which frames.** Every frame something was written down on (a span's edge, a named frame, a `record` series), every
frame of an island with the frame either side, first and last. Then the measured changes, strongest first, at most
`per_second` a second of clip, each with the frame either side. A weaker change on a frame a record or an island
holds joins it and brings the frames either side too; a weaker change on any other frame already listed joins it as
a reason; any other is listed under `also_changed_not_picked`. Then the floor: a frame in each second that has none.
The seconds are aligned to the span's first frame (frames 0-23, 24-47, ...), so two picked frames can lie up to 47
frames apart. A cap drops only the frames either side of a change, weakest change first, and only when that brings
the list to the cap; a change's own frame, what was written down, an island, first, last and the floor never go.
Where those alone pass the cap, no flank is dropped and `cap` says by how much the list is over. A weaker change that
had joined a flank frame the cap took is named in `dropped` with that frame.

It sees only the figures it is given. A change that no figure shows (a face that turns while every count holds
still) is not picked, except by the floor. An amount under `LEVEL_SHARE` of its own median reads as nothing, so a
subject much smaller than her usual size is not told from no subject, and no change between two such sizes is read;
it is named for what it is ("falls under the level floor"), and "gone" or "appears" only where a count is zero.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from .frames import FPS

#: The floor: each `FLOOR`-frame second of clip (counted from the span's first frame) holds at least one picked frame,
#: its middle frame added only where the second holds none. Reasoned: the lane runs at `frames.FPS` end to end
#: (inherited), and the outside video reader samples about one frame a second, so no second goes unvisited.
FLOOR = FPS
#: An amount steps when it moves by more than this share of its local level (the median over `LOCAL` frames either
#: side) in one frame. Reasoned starting value: half its level in a twenty-fourth of a second is not drift.
STEP_SHARE = 0.5
#: Frames either side that make an amount's local level. Reasoned: a quarter second, short enough that a series
#: which has just fallen to nothing is not measured against what it was a second ago.
LOCAL = 6
#: A number counts as something only at or over this share of its series' own median over the span's frames where
#: it is not zero; under it, it reads as nothing. Reasoned starting value: a quarter of the subject's own usual size.
#: On second_hfp_5889e sapiens_face_px has a nonzero median of 7352, so its level is 1838. After she leaves (from
#: 5918) the count also reads 34, 90, 180, 1327, 17, 747, 336 and 53: the level stops those. It does not stop 2036 on
#: 5925 or 2084 on 5929, which clear it and are picked as islands (a one-frame mask fault is worth a look, so that
#: pick stands).
LEVEL_SHARE = 0.25
#: A step, or a crossing of a line, counts only where the level before held on the frame before it and the level after
#: holds on the frame after it. Reasoned: a level that lasts one frame is an island (below) or a blip, not a new level.
HOLD = 2
#: A peak is a local maximum over `PEAK_REACH` frames either side and, unless a line is given, over this many times
#: the series' median over the span. Starting value, reasoned: a frame-to-frame picture difference three times its
#: usual size is a cut or a sudden change, not the shot's ordinary motion.
PEAK_OVER_MEDIAN = 3.0
#: A run of at most this many frames that is something between nothing on both sides, or nothing between something,
#: is an island, and every frame of it is picked. Reasoned: a one- or two-frame region or matte is a fault of a mask
#: (second_hfp_4647d, 2026-10-21: region_px 0 on 4703-4705, 15104 on 4706, 0 on 4707), not a subject coming and going.
ISLAND = 2
#: Frames either side a peak must top. Reasoned: an eighth of a second, so one opening of the mouth is one peak.
PEAK_REACH = 3
#: A place jumps when it moves by more than this share of its own size in one frame. Reasoned starting value: a head
#: or a part that moves half its own width between two frames is a tracker taking somebody else, or a cut.
JUMP_SHARE = 0.5
#: The measured changes picked a second of clip, the strongest first; the rest are listed, not picked. Reasoned: the
#: floor already visits each second once, so one window of change a second at most doubles what a reader looks at.
PER_SECOND = 1
#: The score of a span's edge given as {"record": False}: a flag the capture rates below likely to fail. Reasoned:
#: lower than any change of state and than a label's change, so such an edge is picked only in a second where no
#: measured change outranks it, and is listed with this score where one does.
SPAN_SCORE = 0.5
#: The score a change of state (something to nothing, nothing to something; a place found or lost) starts from, over
#: any step's: its size over the series' median is added. Reasoned: a subject or a mask coming or going outranks any
#: change in how much of it there is.
STATE_SCORE = 10.0
#: The most rows a sheet writer asks for. Reasoned: twice the six evenly spaced frames it replaced. A cap drops only the
#: frames either side of a change, weakest change first, and only when that brings the sheet to the cap; a change's own
#: frame, what was written down, an island, first, last and one frame in each second never go, so a long or busy span's
#: sheet is longer than this, and its record's `cap` and `over_the_cap` say so.
SHEET_CAP = 12

ORDER = {"at": 0, "first": 0, "last": 0, "before": 1, "after": 1, "floor": 2}


def kind_of(values: list) -> str:
    """`amount`, `label` or `place`, from the values that are there."""
    for v in values:
        if v is None:
            continue
        if isinstance(v, (bool, np.bool_, str)):
            return "label"
        if isinstance(v, (tuple, list, np.ndarray)):
            return "place"
        return "amount"
    return "amount"


def _num(v):
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) else v


def _plain(v):
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return round(float(v), 2)
    if isinstance(v, (tuple, list, np.ndarray)):
        return [_plain(x) for x in v]
    return v


def _runs(state: list) -> list[tuple[int, int, object]]:
    """Runs of equal values as (first index, last index, value); None is a run of its own."""
    out: list[tuple[int, int, object]] = []
    for i, v in enumerate(state):
        if out and out[-1][2] == v and v is not None:
            out[-1] = (out[-1][0], i, v)
        else:
            out.append((i, i, v))
    return out


def _states(name: str, x: list, state: list, med: float, kinds: tuple[str, str], brief: tuple[str, str],
            zero: tuple[tuple[str, str], tuple[str, str]] | None = None) -> list[dict]:
    """A two-state series' changes: a run of at most `ISLAND` frames flanked by the other state on both sides is an
    island, each of its frames a change; any other change of state is a crossing (both sides then hold). With `zero`,
    a change whose low side is exactly 0 takes those names in place of `kinds` and `brief` (zero is gone; a low value
    above zero is only under the level)."""
    out, runs = [], _runs(state)

    def named(pair: tuple[str, str], zpair: int, v, low) -> str:
        return (zero[zpair][0] if v else zero[zpair][1]) if zero and low == 0 else (pair[0] if v else pair[1])

    def island(k: int) -> bool:
        if not 0 < k < len(runs) - 1:
            return False
        s, e, v = runs[k]
        return v is not None and runs[k - 1][2] is not None and runs[k - 1][2] == runs[k + 1][2] != v and e - s + 1 <= ISLAND

    for k, (s, e, v) in enumerate(runs):
        if v is None or k == 0 or runs[k - 1][2] is None:
            continue
        if island(k):
            for f in range(s, e + 1):
                low = x[s - 1] if v else x[f]
                out.append({"by": name, "kind": f"{named(brief, 1, v, low)} of {e - s + 1}", "at": f,
                            "from": x[s - 1], "to": x[f], "then": x[e + 1], "score": STATE_SCORE + abs(x[f] - x[s - 1]) / med, "island": True})
        elif not island(k - 1):
            out.append({"by": name, "kind": named(kinds, 0, v, x[s - 1] if v else x[s]), "at": s, "from": x[s - 1], "to": x[s],
                        "score": STATE_SCORE + abs(x[s] - x[s - 1]) / med})
    return out


def amount_changes(name: str, values: list, rules: set, line: float | None = None) -> list[dict]:
    """The changes of a series of numbers, each with a score. A value under `LEVEL_SHARE` of the series' median where
    it is not nothing counts as nothing. `zero`: something becomes nothing or the reverse and holds, or an island of
    either; `step`: a move of more than `STEP_SHARE` of the local level, both levels holding (`HOLD`); `line`: crossing
    `line` and holding (a brief crossing is listed as one); `peak`: a local maximum over `PEAK_REACH` frames, above `line`
    when given, else above `PEAK_OVER_MEDIAN` times the median."""
    x = [_num(v) for v in values]
    have = [abs(v) for v in x if v is not None]
    some = [v for v in have if v > 0]
    if not some:
        return []
    med = float(np.median(some))
    level = LEVEL_SHARE * med
    on = [None if v is None else abs(v) >= level for v in x]
    out = []
    if "zero" in rules:
        out += _states(name, x, on, med, ("rises over the level floor", "falls under the level floor"),
                       ("island over the level floor", "dip under the level floor"), (("appears", "gone"), ("island", "gap")))
    if "line" in rules and line is not None:
        crossed = _states(name, x, [None if v is None else v > line for v in x], med, (f"rises over {line:g}", f"falls to {line:g} or under"),
                          (f"briefly over {line:g}", f"briefly under {line:g}"))
        for c in crossed:
            c["score"] -= STATE_SCORE                       # a line is crossed by degree, not a thing coming or going
            c.pop("island", None)                           # and a brief crossing is listed, not always picked
        out += crossed
    if "step" in rules:
        for i in range(1, len(x)):
            a, b = x[i - 1], x[i]
            if a is None or b is None or not (on[i - 1] and on[i]):
                continue
            near = [abs(v) for v in x[max(0, i - LOCAL):i + LOCAL + 1] if v is not None and abs(v) >= level]
            if abs(b - a) <= STEP_SHARE * float(np.median(near)):
                continue
            after = [v for v in x[i + 1:i + HOLD] if v is not None]
            before = [v for v in x[max(0, i - HOLD):i - 1] if v is not None]
            if all(abs(v - b) < abs(v - a) for v in after) and all(abs(v - a) < abs(v - b) for v in before):
                out.append({"by": name, "kind": "step", "at": i, "from": a, "to": b, "score": abs(b - a) / med})
    if "peak" in rules:
        over = line if line is not None else PEAK_OVER_MEDIAN * float(np.median(have))
        for i, v in enumerate(x):
            if v is None or v <= over or v < level:
                continue
            near = [u for u in x[max(0, i - PEAK_REACH):i + PEAK_REACH + 1] if u is not None]
            if v >= max(near) and x.index(v, max(0, i - PEAK_REACH)) == i:   # the first of equal tops only
                out.append({"by": name, "kind": "peak", "at": i, "from": round(over, 2), "to": v, "score": v / max(over, 1e-6)})
    return out


def label_changes(name: str, values: list) -> list[dict]:
    """A label's changes: every frame whose value is not the frame before's. Each scores 1, a step the size of a median."""
    return [{"by": name, "kind": "changes", "at": i, "from": values[i - 1], "to": values[i], "score": 1.0}
            for i in range(1, len(values)) if values[i] is not None and values[i - 1] is not None and values[i] != values[i - 1]]


def place_changes(name: str, values: list) -> list[dict]:
    """A place's changes: found or lost (a value against None), or a jump of more than `JUMP_SHARE` of its size,
    scored by the move over the size."""
    out = []
    for i in range(1, len(values)):
        a, b = values[i - 1], values[i]
        if (a is None) != (b is None):
            out.append({"by": name, "kind": "lost" if b is None else "found", "at": i, "from": _plain(a), "to": _plain(b), "score": STATE_SCORE + 1})
        elif a is not None:
            move = math.hypot(b[0] - a[0], b[1] - a[1])
            size = max(float(a[2]), float(b[2]), 1e-6)
            if move > JUMP_SHARE * size:
                out.append({"by": name, "kind": "jumps", "at": i, "from": _plain(a), "to": _plain(b), "score": move / size,
                            "detail": f"moved {move:.0f} px, {move / size:.2f} of its size"})
    return out


def changes_of(first: int, last: int, series: dict | None = None, spans: list | None = None, events: list | None = None) -> list[dict]:
    """Every change the inputs show, as {"by", "kind", "frame", "record", "score"?, "island"?, "from"?, "to"?, "detail"?}, in
    frame order. `record` is a change somebody or something wrote down (a span's edge, a named frame, a series given as
    {"record": True}, such as voice on or off): always picked."""
    n = last - first + 1
    out = []
    for name, s in (series or {}).items():
        spec = s if isinstance(s, dict) else {"values": s}
        values = list(spec["values"])
        if len(values) != n:
            raise ValueError(f"series {name!r} has {len(values)} values for a span of {n} frames ({first} to {last})")
        kind = kind_of(values)
        if kind == "label":
            got = label_changes(name, values)
        elif kind == "place":
            got = place_changes(name, values)
        else:
            got = amount_changes(name, values, set(spec.get("rules", ("zero", "step"))), spec.get("line"))
        for c in got:
            c["frame"] = first + c.pop("at")
            c["record"] = bool(spec.get("record"))
            out.append(c)
    for s in spans or []:
        a, b = int(s["frames"][0]), int(s["frames"][1])
        for frame, kind in ((a, "starts"), (b + 1, "has ended")):
            if first <= frame <= last and not (kind == "starts" and frame == first):
                out.append({"by": s["name"], "kind": kind, "frame": frame, "detail": s.get("detail", f"{a}-{b}"),
                            "record": s.get("record", True), **({} if s.get("record", True) else {"score": s.get("score", SPAN_SCORE)})})
    for e in events or []:
        if first <= int(e["frame"]) <= last:
            out.append({"by": e["name"], "kind": "event", "frame": int(e["frame"]), "record": True, **({"detail": e["detail"]} if e.get("detail") else {})})
    return sorted(out, key=lambda c: (c["frame"], c["by"]))


def _fmt(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def short(reason: dict) -> str:
    """One reason in a few words, for a tile's label."""
    role = reason["role"]
    if role in ("first", "last"):
        return f"{role} frame"
    if role == "floor":
        return "one a second"
    ch = reason["change"]
    what = f"{ch['by']}: {ch['detail']}" if ch["kind"] == "event" and ch.get("detail") else f"{ch['by']} {ch['kind']}"
    if ch["kind"] == "peak":
        what += f" {ch['to']:.4g} over line {ch['from']:.4g}"
    elif "from" in ch and "to" in ch and not isinstance(ch["to"], list):
        what += f" {_fmt(ch['from'])}>{_fmt(ch['to'])}" + (f">{_fmt(ch['then'])}" if ch.get("then") is not None else "")
    return what if role == "at" else f"{role} {ch['frame']}: {what}"


def pick(first: int, last: int, series: dict | None = None, spans: list | None = None, events: list | None = None, *,
         floor: int = FLOOR, cap: int | None = None, per_second: int = PER_SECOND) -> dict:
    """The picked frames, each with every reason that put it there; the module's docstring says which frames."""
    if last < first:
        raise ValueError(f"a span from {first} to {last} holds no frame")
    found = changes_of(first, last, series, spans, events)
    always = [c for c in found if c["record"] or c.get("island")]
    for c in found:
        c["picked"] = False
    written = {c["frame"] for c in always}
    room: dict[int, int] = {}
    chosen, rest = [], []
    for c in sorted((c for c in found if not (c["record"] or c.get("island"))), key=lambda c: (-c.get("score", 0), c["frame"])):
        second = (c["frame"] - first) // FPS
        if room.get(second, 0) < per_second:
            room[second] = room.get(second, 0) + 1
            chosen.append(c)
        else:
            rest.append(c)
    on_written = [c for c in rest if c["frame"] in written]      # joins a record's or an island's frame, with its flanks
    beside = [c for c in rest if c["frame"] not in written]       # joins any other listed frame as a reason, or is left out
    owners = [c for c in always if c.get("island")] + chosen + on_written    # the changes whose flanks are listed

    def layout(flanked: list[dict]) -> dict[int, list[dict]]:
        why: dict[int, list[dict]] = {}

        def add(f: int, reason: dict) -> None:
            if first <= f <= last and reason not in why.setdefault(f, []):
                why[f].append(reason)

        add(first, {"role": "first"})
        add(last, {"role": "last"})
        for c in always + chosen + on_written:
            add(c["frame"], {"role": "at", "change": c})
        for c in flanked:
            add(c["frame"] - 1, {"role": "before", "change": c})
            add(c["frame"] + 1, {"role": "after", "change": c})
        for k in range((last - first) // floor + 1):          # the floor: a frame in each second that has none
            lo, hi = first + k * floor, min(first + (k + 1) * floor - 1, last)
            if not any(lo <= f <= hi for f in why):
                add((lo + hi) // 2, {"role": "floor"})
        for c in beside:                               # a weaker change joins any frame on the list, a floor frame too
            if c["frame"] in why:
                add(c["frame"], {"role": "at", "change": c})
        return why

    kept, dropped = list(owners), []
    why = full = layout(kept)
    floor_of_cap = len(layout([]))                    # what no cap can take: changes' own frames, records, islands, first, last, floor
    if cap is not None and len(why) > cap and floor_of_cap <= cap:
        while len(why) > cap:                          # flanks go, weakest change first, only when that reaches the cap
            weakest = min(kept, key=lambda c: (c.get("score", 0), -c["frame"]))
            kept.remove(weakest)
            dropped.append(weakest)
            why = layout(kept)
    over = cap is not None and len(why) > cap

    def same(c: dict) -> dict:
        return {k: v for k, v in c.items() if k != "picked"}

    for c in always + chosen + on_written:
        c["picked"] = True
    joined = {id(c): any(r.get("change") is not None and same(r["change"]) == same(c) for r in why.get(c["frame"], [])) for c in beside}
    was = {id(c): any(r.get("change") is not None and same(r["change"]) == same(c) for r in full.get(c["frame"], [])) for c in beside}
    for c in beside:                                   # by what it says, not which copy: twin flags are picked or not together
        c["picked"] = joined[id(c)]
    lost = [c for c in beside if was[id(c)] and not joined[id(c)]]       # joined a flank frame the cap took
    left_out = [c for c in beside if not c["picked"] and not was[id(c)]]
    for rs in why.values():
        rs.sort(key=lambda r: (ORDER[r["role"]], _rank(r.get("change")), -r.get("change", {}).get("score", 0), r.get("change", {}).get("frame", 0)))
    frames = [{"source_frame": f, "clip_frame": f - first, "clip_seconds": round((f - first) / FPS, 3),
               "reasons": _unique([_reason_out(r) for r in why[f]])} for f in sorted(why)]
    shown = {name: [_plain(v) for v in (spec["values"] if isinstance(spec, dict) else spec)] for name, spec in (series or {}).items()}
    flanks_gone = [d for d in _unique([{**_change_out(c), "frames_dropped": [f for f in (c["frame"] - 1, c["frame"] + 1) if f not in why]}
                                       for c in sorted(dropped, key=lambda c: c["frame"])]) if d["frames_dropped"]]
    flanks_gone += _unique([{**_change_out(c), "its_frame_dropped": c["frame"], "why": "it joined a flank frame the cap took"}
                            for c in sorted(lost, key=lambda c: c["frame"])])
    n_lost = sum(1 for d in flanks_gone if "its_frame_dropped" in d)
    return {"span": [first, last], "frames": frames, "changes": _unique([_change_out(c) for c in found if c["picked"]]),
            "also_changed_not_picked": _unique([_change_out(c) for c in sorted(left_out, key=lambda c: (c["frame"], c["by"]))]),
            "dropped": flanks_gone,
            "series": shown, "over_the_cap": over,
            "cap": (None if cap is None else
                    f"{len(why)} frames for a cap of {cap}: "
                    + (f"the frames either side of {len(flanks_gone) - n_lost} weakest change(s) dropped; " if len(flanks_gone) > n_lost else "")
                    + (f"{n_lost} weaker change(s) that had joined those frames went with them; " if n_lost else "")
                    + (f"over the cap by {len(why) - cap}: the changes' own frames, what was written down, the islands, first, last "
                       f"and a frame in each second without one come to {floor_of_cap} alone, so no flank was dropped" if over else "within it")),
            "rules": {"floor_frames": floor, "per_second": per_second, "step_share": STEP_SHARE, "local_frames": LOCAL,
                      "level_share": LEVEL_SHARE, "hold_frames": HOLD, "peak_over_median": PEAK_OVER_MEDIAN, "peak_reach": PEAK_REACH,
                      "island_frames": ISLAND, "jump_share": JUMP_SHARE, "state_score": STATE_SCORE, "span_score": SPAN_SCORE, "cap": cap,
                      "which_are": "starting values, reasoned (bench/_lib/pick_frames.py says why each)"}}


def _unique(items: list) -> list:
    """The items with exact repeats left out (two flags of one rule with one edge say one thing), in order."""
    out = []
    for x in items:
        if x not in out:
            out.append(x)
    return out


def _rank(c: dict | None) -> int:
    """Which reason a frame is labelled with first: an island, then a measured change, then a named frame, then a span's edge."""
    if c is None:
        return 0
    if c.get("island"):
        return 0
    if c["kind"] == "event":
        return 2
    if c["kind"] in ("starts", "has ended"):
        return 3
    return 1


def _change_out(c: dict) -> dict:
    return {k: _plain(v) for k, v in c.items()}


def _reason_out(r: dict) -> dict:
    out = {"role": r["role"], "short": short(r)}
    if "change" in r:
        out["change"] = _change_out(r["change"])
    return out


def frames_of(picked: dict) -> list[int]:
    return [p["source_frame"] for p in picked["frames"]]


def first_reasons(picked: dict) -> dict[int, str]:
    """Each picked frame's first reason, short: what a tile is labelled with."""
    return {p["source_frame"]: p["reasons"][0]["short"] for p in picked["frames"]}


# ------------------------------------------------------------------ figures the callers share

def points_series(points: dict) -> dict:
    """What a pose model's points say a frame (`_lib/face_points.py::load`'s dict, cut to the span): whether the face is
    there, its height, and where the nose is, as a place the size of the face. A frame whose face points score under
    `h3_config.POSE_READING["eye_ring_score_at_least"]` (inherited: under it the eye rings' points lay far off the face)
    counts as no face, so a face that is lost is one change and the points the model goes on placing after it are not."""
    from . import face_points as fp
    from . import REPO
    if str(REPO / "workflows") not in sys.path:
        sys.path.insert(0, str(REPO / "workflows"))
    import h3_config
    idx = points["index"]
    line = h3_config.POSE_READING["eye_ring_score_at_least"]
    ids = fp.face_ids(idx)
    seen = [bool(np.median(s[ids]) >= line) for s in points["scores"]]
    heights = [fp.face_height(p, idx) if ok else None for p, ok in zip(points["keypoints"], seen)]
    nose = [(float(p[idx["tip_of_nose"]][0]), float(p[idx["tip_of_nose"]][1]), h) if h is not None else None
            for p, h in zip(points["keypoints"], heights)]
    return {"face_points": ["scored" if ok else "under the line" for ok in seen], "face_height_px": heights, "nose": nose}


def box_of(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """[x0, y0, x1, y1) of a mask [h, w] of bool, or None for an empty one."""
    ys, xs = np.nonzero(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if len(xs) else None


def held_boxes(boxes: list) -> tuple[list, list[int]]:
    """Every frame's box, a frame with none holding the last box before it (or, before the first box, the first after
    it); and which frames were held. All None when no frame has a box."""
    have = [i for i, b in enumerate(boxes) if b is not None]
    if not have:
        return list(boxes), []
    out, held, last = [], [], boxes[have[0]]
    for i, b in enumerate(boxes):
        if b is None:
            held.append(i)
            out.append(last)
        else:
            out.append(b)
            last = b
    return out, held


def difference_in_boxes(frames: np.ndarray, boxes: list) -> list[float | None]:
    """Mean absolute difference, in grey levels, of each frame from the frame before inside that frame's box (None on
    the first frame and where there is no box). `frames` [n, h, w] or [n, h, w, 3]."""
    out: list[float | None] = [None]
    for i in range(1, len(frames)):
        b = boxes[i]
        if b is None or b[2] <= b[0] or b[3] <= b[1]:
            out.append(None)
            continue
        x0, y0, x1, y1 = b
        a, c = frames[i - 1][y0:y1, x0:x1].astype(np.float32), frames[i][y0:y1, x0:x1].astype(np.float32)
        out.append(round(float(np.abs(c - a).mean()), 3))
    return out


def centroids(masks) -> list:
    """Each frame's mask as a place: (x, y, the side of a square of its area), None where it is empty."""
    out = []
    for m in masks:
        ys, xs = np.nonzero(m)
        out.append((float(xs.mean()), float(ys.mean()), float(math.sqrt(len(xs)))) if len(xs) else None)
    return out


def sheet_picks(first: int, count: int, series: dict | None = None, spans: list | None = None, events: list | None = None) -> dict:
    """`pick` as a sheet writer calls it: the span's own frames, capped at `SHEET_CAP` rows."""
    return pick(first, first + count - 1, series, spans, events, cap=SHEET_CAP)


def nose_boxes(points: dict, size: tuple[int, int]) -> list:
    """Each frame's square round the nose, as the reference sheets cut it: 2.2 face heights a side at the span's median,
    120 px at least, kept inside the picture. The box a sheet writer measures the picture's change in."""
    from . import face_points as fp
    w, h = size
    idx = points["index"]
    side = min(max(120, int(2.2 * np.median([fp.face_height(p, idx) for p in points["keypoints"]]))), w, h)
    out = []
    for p in points["keypoints"]:
        x, y = p[idx["tip_of_nose"]]
        if not (np.isfinite(x) and np.isfinite(y)):
            out.append(None)
            continue
        x0, y0 = int(min(max(x - side / 2, 0), w - side)), int(min(max(y - side * 0.6, 0), h - side))
        out.append((x0, y0, x0 + side, y0 + side))
    return out


def labels_of(count: int, first: int, frames, yes: str, no: str) -> list[str]:
    """A label a frame: `yes` on the source frames named, `no` on the rest."""
    named = set(frames or [])
    return [yes if first + i in named else no for i in range(count)]
