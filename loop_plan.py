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

**The step grid.** A latent step is a run of frames (core's `FRAME_PER_TOKEN`,
a cycle of `STEP_CYCLE` frames), and every window of a load starts a whole
number of cycles after the load's first frame, so a load cuts its steps at the
same frames in every window. A cut of the source that falls inside a step
gives that step frames of two shots (`split_steps`); which cuts those are is
set by the load's first frame alone (`first_frame_choices`).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from comfy.ldm.minimax.model import FRAME_PER_TOKEN
from comfy_extras.nodes_minimax_h3 import FPS

# Lengths on both clocks: video runs (17k + 5) whose frame count is a multiple
# of 3, so the audio slice is whole latent steps. 39 + 51k.
CHAIN_LENGTHS = (141, 192, 243, 294, 345)
#: Frames between neighbouring lengths on both clocks, so what a window adds
#: moves in these steps and a timeline entry lands within half of one.
#: Inherited from the grid above.
GRID = 51
#: Frames in one cycle of latent steps. Every length in `CHAIN_LENGTHS` and every context
#: `check_window_settings` allows is the same count past a multiple of it, so what a window adds is a
#: whole number of cycles. Inherited from core's `FRAME_PER_TOKEN`.
STEP_CYCLE = sum(FRAME_PER_TOKEN)

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


def frames_covered(lengths: list[int], context_frames: int) -> int:
    """The frames a run of windows covers from its first frame: every window after the first
    adds its length less the context it shares with the one before."""
    return sum(lengths) - int(context_frames) * (len(lengths) - 1) if lengths else 0


def frames_read(total_frames: int, window_frames: int, context_frames: int, timeline: str = "") -> int:
    """How many frames of the track a run's windows read, counted from frame zero.

    Never fewer than `total_frames` and usually more: the last window ends at or past the end
    of the track ("Lengths" above). A loader that holds the track's own picture for the song
    node has to load this many and no more: what it loads past this, nothing reads, and what
    the mask's tracker does with it is wasted (`workflows/build_workflows.py`, `freeze_song_source`).
    """
    segments = plan_windows(total_frames, window_frames, context_frames, parse_timeline(timeline))
    return frames_covered([n for _entry, lengths in segments for n in lengths], context_frames)


def frames_kept(covered: int, track_seconds: float) -> int:
    """How many of the frames a run's windows cover are written: all of them, or as many as the
    track runs when it ends first.

    The last window ends at or past the end of the track it was planned from ("Lengths" above).
    When the track on hand is longer than that, as with an `extent` of its first seconds, nothing
    is past the track and every frame is kept. When it is not, the frames past its end are
    dropped from the last window before it is encoded, so the finished video ends with its
    track, on a whole frame (`loop_output.join_and_mux` says why not in the join). A last
    window that a run before 2026-10-06 stored holds its tail still, under a key that still
    matches: it is not reused, and renders once more (`loop_resume.stored_frames`).
    """
    return min(int(covered), int(math.ceil(float(track_seconds) * FPS)))


def frames_written(lengths: list[int], context_frames: int, kept: int, head: int = 0) -> list[int]:
    """What each window of a run writes to its file, in order: its length less the context it
    shares with the window before, and for the last, less the frames past `kept`
    (`frames_kept`). The sum is `kept`, less `head`.

    `head` is for a run that continues from another run's last window (the song node's
    `continue_from`): its first `head` frames are that window's context, so its first window
    writes its length less them, as every later window does. `kept` and the frames covered
    still count from the first frame of the run's own track, context included.

    The song node cuts a window to this, stores the count with it and reuses a stored window
    only when its file holds this many, so the join copies whole files. A cut that reached
    what the last window writes is refused here, before anything is encoded: the planner
    never plans one (`bench/check_audio_freeze.py` holds that over every window length), and
    a slice with a negative end would write the wrong frames without a word.
    """
    writes = [int(n) - (int(context_frames) if i else 0) for i, n in enumerate(lengths)]
    if not writes:
        return writes
    if not 0 <= int(head) < writes[0]:
        raise ValueError(f"a head of {head} frames does not fit the {writes[0]} the first window of {list(lengths)} holds")
    writes[0] -= int(head)
    cut = frames_covered(lengths, context_frames) - int(kept)
    if not 0 <= cut < writes[-1]:
        raise ValueError(f"windows {list(lengths)} with {context_frames} of context cover "
                         f"{frames_covered(lengths, context_frames)} frames and {kept} are kept: a cut of {cut} "
                         f"does not fit the {writes[-1]} frames the last window writes")
    writes[-1] -= cut
    return writes


def extent_shortfall(track_seconds: float, asked_seconds: float | None, total_frames: int,
                     source_frames: int | None = None, asked_reads: int | None = None) -> str | None:
    """One line for the song node's report when a run cannot cover what it was asked for, or None.

    Two ways it happens, and neither is refused: a clip that is simply shorter than the
    extent has to render without anyone touching a widget.

    - The track is shorter than the extent asked for, by a frame or more. On a graph whose
      track is its source video's own audio, that is what a loader cap under the extent looks
      like, since the loader cuts the audio where it cuts the frames: the line gives the
      frames the source holds and the frames the asked extent reads (`asked_reads`).
    - The source's picture is shorter than the track (`source_frames` under `total_frames`):
      the frames past its end hold the last one, unmasked, to the end of the run. A source
      a few frames short of the PLAN but not of the track is not this: those frames are past
      the track and are not written (`frames_kept`).
    """
    parts = []
    if asked_seconds is not None and float(asked_seconds) - float(track_seconds) >= 1.0 / FPS:
        line = (f"shorter than asked: the track is {float(track_seconds):.2f}s and this run was asked for "
                f"{float(asked_seconds):g}s, so it covers the track and no more")
        if source_frames is not None:
            line += (f". The source video holds {int(source_frames)} frames"
                     + (f" and {float(asked_seconds):g}s reads {int(asked_reads)}" if asked_reads else "")
                     + "; if the file is longer than that, raise the loader's frame_load_cap"
                     + (f" to {int(asked_reads)}" if asked_reads else ""))
        parts.append(line)
    if source_frames is not None and int(source_frames) < int(total_frames):
        short = int(total_frames) - int(source_frames)
        parts.append(f"the source's picture ends early: it holds {int(source_frames)} frames and the track runs "
                     f"{int(total_frames)}, so the last {short} hold its final frame, unmasked. If the file is "
                     "longer than that, raise the loader's frame_load_cap")
    return "; ".join(parts) or None


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


def step_span(frame: int) -> tuple[int, int]:
    """The first frame and the length of the latent step that holds `frame`, both counted from a load's
    first frame. The same in every window of the load ("The step grid" above)."""
    frame = int(frame)
    if frame < 0:
        raise ValueError(f"frame {frame} is before the load's first frame")
    at = frame // STEP_CYCLE * STEP_CYCLE
    for n in FRAME_PER_TOKEN:
        if frame < at + n:
            return at, n
        at += n
    raise AssertionError("a cycle of FRAME_PER_TOKEN holds every frame of it")


def split_steps(cuts, first_frame: int = 0, present=None, frames: int | None = None) -> list[dict]:
    """The latent steps of a load that hold frames of more than one shot, and what that does to each.

    `cuts` are frames of the source that start a shot and `first_frame` the source frame the load starts on.
    `present` is the frames the subject is on, as `(first, last)` ranges of source frames, both ends counted
    (a shot table's shots with the subject, or finer); None is "on every frame", the worst case for the
    region. `frames` is the load's length when the load stops before the cuts do.

    One entry per split step, in source frames: `step` (its first and last frame), `cuts` (those inside it),
    `across` (frames on a side of a cut the subject is on no frame of, while another side has it: the
    region is carried onto another shot's picture; `video_mask.cut_gate` leaves them as the source's) and
    `shared` (frames of the sides that have the subject, when more than one does: each side is given the
    other side's region too). A step no side of which has the subject is not listed: nothing regenerates.
    The rule is `cut_gate`'s, on ranges where that one reads a mask.
    """
    first = int(first_frame)
    end = None if frames is None else first + int(frames)
    on = None if present is None else [(int(a), int(b)) for a, b in present]
    inside: dict[int, list[int]] = {}
    for cut in sorted({int(c) for c in cuts}):
        if cut <= first or (end is not None and cut >= end):
            continue
        at, _n = step_span(cut - first)
        if first + at != cut:
            inside.setdefault(at, []).append(cut)
    out = []
    for at, held in sorted(inside.items()):
        a, n = step_span(at)
        last = first + a + n if end is None else min(first + a + n, end)
        edges = [first + a] + held + [last]
        sides = list(zip(edges, edges[1:]))
        has = [on is None or any(x < b and y >= lo for x, y in on) for lo, b in sides]
        if not any(has):
            continue
        across = [f for (lo, b), h in zip(sides, has) if not h for f in range(lo, b)]
        shared = [f for (lo, b), h in zip(sides, has) if h for f in range(lo, b)] if sum(has) > 1 else []
        out.append({"step": [first + a, last - 1], "cuts": held, "across": across, "shared": shared})
    return out


def first_frame_choices(cuts, first_frame: int, present=None, frames: int | None = None) -> list[dict]:
    """`split_steps` for each start of a load from `first_frame` back to one cycle of latent steps before it,
    the best first.

    Starting a load a few frames early moves every step's edge by as many, so a cut that splits a step
    under one start falls on an edge under another. A load only ever starts earlier here: the frames from
    `first_frame` on are all still rendered, and the ones before it are the caller's to drop. A start
    before frame zero is left out. With `frames` (the length from `first_frame`) the load's end stays
    where it was.

    Each entry: `first_frame`, `earlier` (how many frames before the one asked for), `split` (the steps),
    `across` and `shared` (how many frames of each). Ordered by fewest frames of the two together, then
    fewest split steps, then fewest frames added. Reasoned, not measured: whether a frame of a split step
    on the subject's own side is any worse than its neighbours is not known (2026-10-10).
    """
    first = int(first_frame)
    out = []
    for k in range(min(STEP_CYCLE, first + 1)):
        split = split_steps(cuts, first - k, present, None if frames is None else int(frames) + k)
        out.append({"first_frame": first - k, "earlier": k, "split": split,
                    "across": sum(len(s["across"]) for s in split), "shared": sum(len(s["shared"]) for s in split)})
    return sorted(out, key=lambda c: (c["across"] + c["shared"], len(c["split"]), c["earlier"]))
