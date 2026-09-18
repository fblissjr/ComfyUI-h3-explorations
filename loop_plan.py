"""Where a loop's windows fall on its track, and which text each window renders.

`docs/h3_audio_freeze.md` owns the lane; `audio_freeze_song.py` is the node
that uses this first. Imports nothing from the pack, so
`bench/check_audio_freeze.py` exercises it without the node, and the next loop
node plans the same way.

**Lengths.** Windows are at most `window_frames` long with `context_frames` of
the previous window frozen at their head, so each adds its length minus the
context (the first adds all of it). Every length is on both clocks
(`CHAIN_LENGTHS`), so what a window adds moves in steps of `GRID` frames.
`segment_lengths` gives each part of the track the fewest windows that end it
nearest its target, each as long as the windows after it allow, so most
windows are `window_frames` and the short ones come last. The last part ends
at or past the end of the track; the window node pads silence past it.

**The timeline** (optional) is `mm:ss label` lines, the first at 00:00: song
sections, lines of dialogue, story beats. Each entry starts a part, and each
part aims at its own entry's time rather than at where the part before ended,
so a boundary lands within half a `GRID` step and no error carries forward.
An entry too short to hold one window is refused by name; entries past the
covered length are left out. With no timeline the whole track is one part.

**The prompt** is one text for every window, or with a timeline one block per
label, each opened by a line `--- label`. Every label in the timeline needs a
block and every block a label in the timeline. A window whose text has an `At
mm:ss` time at or past its own length is refused: windows differ in length, and
a beat the window never reaches renders a different scene from the one written.
(Since 2026-09-18 shot headers carry no time, so this is a time that splits
action inside a shot, the one place the house still writes one.)

**Uses of the lists.** `Plan.uses` is one text per use: per timeline entry, so
every window of one chorus shares a filled-in text and the next chorus takes
the next values, or per window with no timeline. The node fills them through
`prompt_lists.fill_windows` and hands the result to `place_windows`.

(`prompt_mode`, random window lengths and `frames:` lines went on 2026-09-14;
the owner cut them for the timeline.)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from comfy_extras.nodes_minimax_h3 import FPS

# Lengths on both clocks: video runs (17k + 5) whose frame count is a multiple
# of 3, so the audio slice is whole latent steps. 39 + 51k.
CHAIN_LENGTHS = (141, 192, 243, 294, 345)
#: Frames between neighbouring lengths on both clocks, so what a window adds
#: moves in these steps and a timeline entry lands within half of one.
#: Inherited from the grid above.
GRID = 51

TIMELINE_LINE = re.compile(r"^(\d+):([0-5]\d(?:\.\d+)?)\s+(\S.*)$")
BLOCK_LINE = re.compile(r"^---(.*)$")
#: A clock time in prompt text: "At 00:04.500, ...". Until 2026-09-18 that was how
#: a later shot opened; the house now writes one only to split action INSIDE a
#: shot (`docs/prompting.md` section 3.1), and that is what this still guards.
CUT_TIME = re.compile(r"\bAt (\d+):(\d{2}(?:\.\d+)?)")


def clock(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def check_window_settings(window_frames: int, context_frames: int) -> None:
    if window_frames not in CHAIN_LENGTHS:
        raise ValueError(f"window_frames {window_frames} is not on both clocks; use one of {CHAIN_LENGTHS}")
    if context_frames % 17 != 5 or (context_frames * 5) % 3 != 0 or context_frames >= window_frames:
        raise ValueError(f"context_frames {context_frames} must be 39, 90 or 141 and shorter than the window")


def parse_timeline(text: str) -> list[tuple[float, str]]:
    """`mm:ss label` lines as (seconds, label); blank lines and # lines skipped. Empty is no timeline."""
    entries: list[tuple[float, str]] = []
    for raw in (text or "").replace("\r\n", "\n").split("\n"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = TIMELINE_LINE.match(line)
        if not m:
            raise ValueError(f"timeline line {line!r} is not `mm:ss label`, e.g. `00:32 chorus`")
        seconds = int(m.group(1)) * 60 + float(m.group(2))
        if entries and seconds <= entries[-1][0]:
            raise ValueError(f"timeline line {line!r} is not later than the line before it")
        entries.append((seconds, m.group(3).strip()))
    if entries and entries[0][0] != 0:
        raise ValueError(f"the timeline starts at {clock(entries[0][0])}; its first line must be at 00:00")
    return entries


def parse_prompt_blocks(text: str) -> dict[str | None, str]:
    """{None: prompt} for a prompt with no `---` line, else {label: block} per `--- label` line."""
    lines = (text or "").replace("\r\n", "\n").split("\n")
    if not any(BLOCK_LINE.match(line.strip()) for line in lines):
        body = (text or "").strip()
        if not body:
            raise ValueError("the prompt is empty")
        return {None: body}
    blocks: dict[str | None, str] = {}
    label: str | None = None
    current: list[str] = []

    def close():
        body = "\n".join(current).strip()
        if label is None:
            if body:
                raise ValueError("the prompt has text before its first `--- label` line; every block needs a label")
        elif not body:
            raise ValueError(f"the prompt block `--- {label}` is empty")
        else:
            blocks[label] = body

    for raw in lines:
        m = BLOCK_LINE.match(raw.strip())
        if not m:
            current.append(raw)
            continue
        close()
        label = m.group(1).strip()
        if not label:
            raise ValueError("a `---` line needs the timeline label its block is for, e.g. `--- chorus`")
        if label in blocks:
            raise ValueError(f"two prompt blocks are labelled {label!r}")
        current = []
    close()
    return blocks


def texts_for_entries(blocks: dict[str | None, str], entries: list[tuple[float, str]]) -> list[str]:
    """One prompt per timeline entry (one in all with no timeline), checked against the labels."""
    if None in blocks:
        return [blocks[None]] * max(len(entries), 1)
    if not entries:
        raise ValueError("labelled prompt blocks need a timeline: add `mm:ss label` lines to `timeline`")
    labels = [label for _t, label in entries]
    missing = [label for label in dict.fromkeys(labels) if label not in blocks]
    if missing:
        raise ValueError(f"no prompt block for the timeline label(s) {missing}; add a `--- {missing[0]}` block")
    extra = [label for label in blocks if label not in labels]
    if extra:
        raise ValueError(f"the prompt block(s) {extra} match no timeline label; the timeline has "
                         f"{list(dict.fromkeys(labels))}")
    return [blocks[label] for label in labels]


def _adds(first: bool, window_frames: int, context_frames: int) -> tuple[int, int]:
    """Least and most frames one window adds: all of it for the track's first window, else minus the context."""
    lengths = [n for n in CHAIN_LENGTHS if context_frames < n <= window_frames]
    cut = 0 if first else context_frames
    return lengths[0] - cut, lengths[-1] - cut


def _bounds(count: int, first: bool, window_frames: int, context_frames: int) -> tuple[int, int]:
    if count == 0:
        return 0, 0
    a0, b0 = _adds(first, window_frames, context_frames)
    a, b = _adds(False, window_frames, context_frames)
    return a0 + a * (count - 1), b0 + b * (count - 1)


def segment_lengths(target: int, first: bool, final: bool, window_frames: int, context_frames: int) -> list[int]:
    """Window lengths for one part of the track, adding `target` frames as nearly as the grid allows.

    The fewest windows whose total lands nearest `target` (at or past it when
    `final`), each as long as the windows after it allow. The totals a count
    of windows can add run in `GRID` steps from its least to its most, so
    nearest is within half a step whenever the target is at least one window.
    """
    best = None
    count = 1
    while True:
        lo, hi = _bounds(count, first, window_frames, context_frames)
        if best is not None and lo > best[2] + GRID:
            break
        if final:
            if target <= hi:
                total = lo if target <= lo else lo + math.ceil((target - lo) / GRID) * GRID
                key = (total - target, count)
                if best is None or key < best[0]:
                    best = (key, count, total)
        else:
            clamped = min(max(target, lo), hi)
            total = min(lo + ((clamped - lo + GRID // 2) // GRID) * GRID, hi)
            key = (abs(total - target), count)
            if best is None or key < best[0]:
                best = (key, count, total)
        count += 1
    _key, count, remaining = best
    lengths = []
    for j in range(count):
        head = first and j == 0
        most = _adds(head, window_frames, context_frames)[1]
        rest_least = _bounds(count - j - 1, False, window_frames, context_frames)[0]
        add = min(most, remaining - rest_least)
        lengths.append(add + (0 if head else context_frames))
        remaining -= add
    return lengths


def plan_windows(total_frames: int, window_frames: int, context_frames: int,
                 entries: list[tuple[float, str]] | tuple = ()) -> list[tuple[int, list[int]]]:
    """(entry index, window lengths) for each timeline entry the track reaches; one part with no timeline."""
    check_window_settings(int(window_frames), int(context_frames))
    if total_frames <= 0:
        raise ValueError("the track is empty")
    starts = [int(round(t * FPS)) for t, _label in entries] or [0]
    kept = [i for i, f in enumerate(starts) if f < total_frames]
    plan, covered = [], 0
    for k, i in enumerate(kept):
        final = k + 1 == len(kept)
        end = total_frames if final else starts[kept[k + 1]]
        target = end - covered
        first = covered == 0
        least = _adds(first, window_frames, context_frames)[0]
        if target <= 0 or (not final and target + GRID // 2 < least):
            name = f"the timeline entry {entries[i][1]!r} at {clock(entries[i][0])}" if entries else "the track"
            raise ValueError(
                f"{name} lasts {max(target, 0) / FPS:.2f}s from where the windows before it end, less than "
                f"one window adds here ({least / FPS:.2f}s at context_frames {context_frames}); merge it "
                "into a neighbouring entry")
        lengths = segment_lengths(target, first, final, int(window_frames), int(context_frames))
        plan.append((i, lengths))
        covered += sum(n - (0 if first and j == 0 else context_frames) for j, n in enumerate(lengths))
    return plan


@dataclass
class Plan:
    entries: list[tuple[float, str]]
    segments: list[tuple[int | None, list[int]]]  # (entry index or None, window lengths)
    uses: list[str]                               # one text per use of the lists


@dataclass
class Window:
    number: int
    frames: int
    start: float        # seconds, where the window node slices the track from
    first_frame: int    # the first new frame this window adds, on the track's clock
    entry: int | None   # index into the timeline, None with no timeline
    text: str


def plan_song(total_frames: int, window_frames: int, context_frames: int, prompt: str, timeline: str) -> Plan:
    """The windows' lengths and the texts to fill; raises on a bad timeline, block or window setting."""
    entries = parse_timeline(timeline)
    entry_texts = texts_for_entries(parse_prompt_blocks(prompt), entries)
    segments = plan_windows(total_frames, window_frames, context_frames, entries)
    if entries:
        # every entry is a use, including those past the covered length, so a
        # list used only there is not refused as unused; entries run in order,
        # so the entries a run reaches take the same values either way
        return Plan(entries, list(segments), entry_texts)
    lengths = segments[0][1]
    return Plan(entries, [(None, lengths)], [entry_texts[0]] * len(lengths))


def place_windows(plan: Plan, filled: list[str], context_frames: int) -> list[Window]:
    """Each window's start, first new frame and filled text; refuses a cut past a window's end."""
    if len(filled) != len(plan.uses):
        raise ValueError(f"{len(filled)} filled texts for {len(plan.uses)} uses")
    context_frames = int(context_frames)
    if plan.entries:
        per_window = [(i, n, filled[i]) for i, lengths in plan.segments for n in lengths]
    else:
        per_window = [(None, n, filled[j]) for j, n in enumerate(plan.segments[0][1])]
    windows, start, covered = [], 0.0, 0
    for j, (entry, frames, text) in enumerate(per_window):
        windows.append(Window(j + 1, frames, start, covered, entry, text))
        covered += frames - (context_frames if j else 0)
        # the window node's own arithmetic for next_start_seconds
        start = start + (frames - context_frames) / FPS
    for w in windows:
        seconds = w.frames / FPS
        entry = w.entry
        for m in CUT_TIME.finditer(w.text):
            at = int(m.group(1)) * 60 + float(m.group(2))
            if at >= seconds:
                where = f" ({plan.entries[entry][1]})" if entry is not None else ""
                raise ValueError(
                    f"window {w.number}{where} is {seconds:.2f}s long but its prompt cuts at "
                    f"{m.group(1)}:{m.group(2)}; some windows are shorter than window_frames, so move "
                    "the cut earlier or give that entry a block of its own")
    return windows
