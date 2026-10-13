#!/usr/bin/env python3
"""Signals of a clip's own sound, a row a source frame, for whoever sets a window's first and last frame: phrases, a beat grid, the sound's offset to the picture.

    <python> bench/clip_signals.py offset --out AUDIO_DIR COPY [COPY ...]          # sound_offset.record.json
    <python> bench/clip_signals.py phrases AUDIO_DIR                               # phrases_per_frame.csv + phrases.record.json
    <python> bench/clip_signals.py dips AUDIO_DIR                                  # dips_per_frame.csv + dips.record.json
    <python> bench/clip_signals.py beats AUDIO_DIR [--shots data/<stem>/shots.json] # beats_per_frame.csv + beats.record.json
    <python> bench/clip_signals.py at AUDIO_DIR --frames FIRST-LAST [--name LOAD] [--shots data/<stem>/shots.json]

**What it buys.** A load began where its subject's face first showed and ended where it stopped showing, and the
sound was read for one thing: whether a text may say the subject sings (`bench/voice_spans.py`). The model hears
its window's sound, so a window that begins in the middle of a sung phrase or off the beat begins in the middle of
something. The owner's word of 2026-10-13: no rule; capture it, and have it in front of whoever sets a window's
edges. The wants are `docs/wiki/window_edges_data.md`; this tool is the three of them that need only the two
stems and the copies themselves, with no card and no library the environment did not already hold.

**Nothing here is a rule.** No step refuses a load, moves an edge or colors anything. Every setting below carries
where it is from, is written into the signal's record, and decides only what is LISTED as a phrase or a beat; the
levels and the onset curve the listing was made from are in the table beside it, so a reader can cut otherwise.

**Where it lives.** `AUDIO_DIR` is the video's audio folder, the one `voice_spans.py` made (`vocals.wav`,
`rest.wav`, `voice_spans.json`): these are signals of the same sound on the same clock, the SOURCE's frames, and
are read from the same two stems. One file a signal, a row a source frame, and beside each a record: every file
it was read from with its size and hash, the libraries and their versions, the settings, the time, the commit,
what it cannot see in words, and the table's own hash as a fingerprint a capture can name. A cell with nothing to
say is EMPTY, never zero.

**offset.** One figure a copy: the time of the first audio packet against the first picture, as the container
has it, in seconds and in the copy's frames. Read with PyAV. It cannot see what is HEARD: an encoder's priming
samples sit in the first packet's time and are not sound, and an edit list can hide them from a player.

**phrases.** The vocals stem's level at `STEPS` steps a frame, where the voice table has one. THE LEVEL IS THE
POWER OF BOTH CHANNELS, not of their mean: where the two channels differ their mean reads low, and on the first
real run ten of thirty-five breaks hung on that (a reader's finding, 2026-10-13). The voice table's own level IS
of the mean, and its line was set on that scale; the record says so. THE VOICED RUNS ARE THE TOOL'S OWN, made from
this same level by the voice table's own rule (its line, its join, its least length: `voice_spans.voiced_spans`,
imported), so that a run's edges and what is quiet inside it are on ONE scale; `voice_spans.json` is read for the
frame rate, the frame count and the line, and the record lists the frames on which its runs and these differ. The
voice table itself is not changed: a load's text leans on it as it is. Inside each voiced run, every stretch
under that table's line is a GAP, listed with its length and its deepest
level; a gap of `BREAK` steps or more is a phrase break, and the stretches between breaks are the phrases. It
cannot see: a break sung through with no drop in level; a held note from a new phrase; whose voice it is, or a
backing voice (the stem holds every voice); a breath the separation put in the other stem.

**dips.** What the line cannot see, as data: a breath inside unbroken singing does not fall under the voice
table's line, so the phrases pass over it (on the first real run a "phrase" a whole voiced run long held a dip of
several frames a reader found by hand). Every step's level is set against THE SUNG LEVEL ROUND IT (a high
percentile of the level within a window either side), and every stretch that lies `DIP_FLOOR` decibels or more
under it for `DIP_LEAST` steps or more, and touches a voiced run, is listed: where it starts and ends, how long,
how deep at its deepest, the level there, how many of its frames are under the voice table's line, and the
vocals stem's own onset strength where the dip ends (a line that begins after a breath shows as a spike there
even where the level barely moved). NOTHING IS CUT: the floor only bounds the list, and whether a dip is a
breath between two sung lines, a consonant inside one or the separation's residue is a reading of the table, the
reader's. It cannot see which of those a dip is, nor a breath with no fall in level at all.

**beats.** From the rest stem: an onset curve (the rise of EACH channel's spectrum from one short step to the next,
summed: of the channels' mean a wide hit can cancel and lose its start), a
tempo from that curve's likeness to itself a beat later, and a grid laid by the tracker of Ellis (2007), which
walks the curve keeping beats near one period apart. Beside the grid: the tempo, the three strongest tempo
candidates (a tempo and its double or half are the same music to this method), how much stronger the curve is on
the beats than off them, and the tempo found in each stretch of `LOCAL_SECONDS`, so a track whose tempo moves
shows as a spread and the one grid is read as the poor fit it is. A beat's time and the per-frame onset column
are on ONE clock (the middle of the tracker's window). Each beat keeps its own strength, as it is and against the
middle beat's, and the frames since the beat before; the record says where the STEADY RUN of beats begins and
ends, since the tracker lays beats before a pulse starts and after it stops, on single sounds or on nothing.
Where the curve is as like itself two beats on, the record says every other beat may be the pulse and gives the
beats' strength on each of the two ways of taking them. Where the curve is also like itself at about HALF the
period, the place half-way between each pair of beats is written beside the grid with the curve's strength there: a hit of another band can sit there (on the first real
run a low hit sat half-way between every pair of beats). It is a place half-way, never called a beat: which of
the two a listener counts is not something these figures can say. NO DOWNBEAT and no count in fours is given.
With `--shots` the record's cuts are printed against the grid once, with what cuts placed by chance would give.
It cannot see: a tempo that drifts or changes (one fit); a bar's first beat; music with no steady pulse.

**at.** The figures for a load's two edges, printed and not stored: the voiced run and the phrase each edge falls
in and the frames from that phrase's start and to its end; the nearest beat before and after; for the first
frame, how many frames earlier the load would begin on the phrase's start and on the beat before, and whether
that frame is inside the same shot (`--shots`); for the last frame, the same looking forward, since the frames
after a load's end are held open. Then what lies INSIDE the load, between its two edges: every phrase that starts
or ends there, every beat and every half-way place, each with its place counted from the load's first frame. A
reader's print: summaries are never stored.

The settings' checks are `bench/check_clip_signals.py`, on made sound.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

#: Steps a video frame at which the vocals stem's level is read. Reasoned: a frame is the grain the voice table
#: has, and a phrase's edge inside a frame is what it cannot say; four steps is about a hundredth of a second.
STEPS = 4
#: A gap under the voice table's line of this many steps or more is a phrase break (three frames). Reasoned, and
#: UNTESTED ON ANY TRACK: a stop consonant closes the voice for under a tenth of a second and a breath for longer.
#: Every gap is listed with its length whichever side of this it falls.
BREAK = 3 * STEPS
#: The sung level round a step: this percentile of the level within this many frames either side. INHERITED from
#: the blind reader's own check of the first run (2026-10-13), who chose them for that check and tested them on
#: no other track: a starting value.
DIP_PERCENTILE, DIP_WINDOW_FRAMES = 90.0, 36
#: The list's floor: a stretch this many decibels or more under the sung level round it, for this many steps or
#: more (half a frame). The floor bounds the LIST and cuts nothing. Reasoned: half the depth the reader listed at,
#: so that a reader who cuts at any depth from there up finds every dip in the table; untested.
DIP_FLOOR, DIP_LEAST = 6.0, STEPS // 2
#: The onset curve's sound rate, window and step. Inherited: the usual values for a beat tracker's curve
#: (Ellis 2007 used 8 kHz and 4 ms; these are the common later ones at half the stems' rate).
BEAT_RATE, WINDOW, HOP = 22050, 1024, 256
#: The tempos looked for, a minute's beats, and the tempo the search leans to with its width in octaves.
#: Inherited: Ellis 2007 (a log-normal weight about 120, one octave wide).
TEMPO_RANGE, TEMPO_CENTER, TEMPO_OCTAVES = (50.0, 220.0), 120.0, 1.0
#: How hard the tracker holds beats to one period apart. Inherited: Ellis 2007's own value.
TIGHTNESS = 100.0
#: The seconds from a step of the onset curve to the middle of the window it is the rise into: the one clock of
#: a beat's time and of the per-frame onset column. Derived from the window.
HALF_WINDOW = (WINDOW / 2) / BEAT_RATE
#: The stretch a local tempo is read in, and its step. Reasoned: long enough to hold a dozen beats at the slowest
#: tempo looked for, short enough that a tempo moving over a minute shows.
LOCAL_SECONDS, LOCAL_STEP = 16.0, 8.0
#: A tempo candidate this near twice the tempo, as a share of it, is taken as "the curve is also like itself at
#: half the period", and the half-way places are then written. Reasoned: one step of the curve's own lag near
#: 120 a minute is about a fortieth of the period, and the candidates of a split peak sit a step either side.
NEAR = 0.05
#: The steady run: beats whose gap to the beat before is within this share of the middle gap, and whose own
#: strength is at least this share of the middle beat's. STARTING VALUES from one reading of one track
#: (2026-10-13): inside the pulse the gaps moved by about a thirtieth and before it by a tenth and more; two beats
#: laid on silence had a twentieth and a tenth of a strong onset. The figures themselves are in every row.
STEADY_GAP, STEADY_STRENGTH = 0.08, 0.25

CANNOT = {
    "offset": "what is heard: an encoder's priming samples sit in the first packet's time and are not sound, and an edit list can hide them from a player; "
              "a drift later in the file (this is the start only). A note of 2026-10-10 that a source's sound starts a part of a frame before its picture is "
              "therefore neither confirmed nor refuted by a figure of zero here",
    "phrases": "a break sung through with no drop in level; a held note from a new phrase; whose voice it is, or a backing voice (the stem holds every voice); "
               "a breath the separation put in the other stem",
    "dips": "which of a breath between two sung lines, a consonant inside one and the separation's residue a dip is; a breath with no fall in level; "
            "whose voice falls (the stem holds every voice). The sung level round a step is a percentile over a window, so the first dip after a long "
            "silence is measured against less singing than the others",
    "beats": "a tempo that drifts or changes (the grid is one fit; the local tempos show it and do not mend it); a bar's first beat (no downbeat and no count "
             "in fours is given); music with no steady pulse; a tempo from its double or its half, or which of a beat and the place half-way a listener counts",
}


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def about(path: Path) -> dict:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def record(what: str, read_from: list[Path], settings: dict, table: Path | None, more: dict) -> dict:
    import scipy
    commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "not in a repository"
    out = {"signal": what, "made_at": now(), "made_by": "bench/clip_signals.py", "tool_commit": commit,
           "read_from": [about(p) for p in read_from], "libraries": {"numpy": np.__version__, "scipy": scipy.__version__},
           "settings": settings, "cannot_see": CANNOT[what]}
    if table is not None:
        out["rows_are"] = "source frames, counted from 0; an empty cell has nothing to say and is not a zero"
        out["table"] = {"file": table.name, "sha256": about(table)["sha256"]}
        out["fingerprint"] = out["table"]["sha256"]
    out.update(more)
    return out


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """A wav as float in -1..1, [samples, channels], and its rate."""
    from scipy.io import wavfile
    rate, data = wavfile.read(str(path))
    data = data.astype(np.float32) / (32768.0 if data.dtype == np.int16 else 1.0)
    return (data if data.ndim == 2 else data[:, None]), int(rate)


def step_levels(sound: np.ndarray, rate: int, fps: float, frames: int, steps: int = STEPS) -> np.ndarray:
    """The level in dBFS of each of `steps` equal stretches of each video frame: [frames, steps]. `sound` is
    [samples, channels] and the level is the power of EVERY channel's samples, not of the channels' mean: two
    channels that differ cancel in their mean and read low. A stretch past the sound's end is NaN."""
    out = np.full((frames, steps), np.nan, np.float32)
    for k in range(frames * steps):
        seg = sound[int(round(k * rate / fps / steps)):int(round((k + 1) * rate / fps / steps))]
        if len(seg):
            out[k // steps, k % steps] = 20.0 * np.log10(max(float(np.sqrt(np.mean(seg ** 2))), 1e-7))
    return out


def own_runs(level: np.ndarray, line: float, steps: int = STEPS) -> list[list[int]]:
    """The voiced runs of this table's own level, by the voice table's own rule: a frame is voiced when its level
    (the power over its steps) is over the line; gaps of `voice_spans.JOIN` frames or fewer are closed and runs
    under `voice_spans.LEAST` frames dropped. One scale for the runs and for what is quiet inside them."""
    import voice_spans
    power = np.nanmean(10.0 ** (np.nan_to_num(level.astype(np.float64), nan=-140.0) / 10.0), axis=1)
    frame = 10.0 * np.log10(np.maximum(power, 1e-14))
    return [[int(a), int(b)] for a, b in voice_spans.voiced_spans(frame.astype(np.float32), line)]


def runs_differ(mine: list[list[int]], theirs: list[list[int]], frames: int) -> dict:
    """The frames voiced in one set of runs and not in the other, as runs."""
    a, b = np.zeros(frames, bool), np.zeros(frames, bool)
    for lo, hi in mine:
        a[lo:hi + 1] = True
    for lo, hi in theirs:
        b[lo:hi + 1] = True

    def as_runs(mask):
        out, start = [], None
        for n, on in enumerate(list(mask) + [False]):
            if on and start is None:
                start = n
            elif not on and start is not None:
                out.append([start, n - 1])
                start = None
        return out
    return {"voiced_here_and_not_in_the_voice_table": as_runs(a & ~b), "voiced_in_the_voice_table_and_not_here": as_runs(b & ~a),
            "frames_that_differ": int((a != b).sum())}


def gaps_and_phrases(level: np.ndarray, runs: list[list[int]], line: float, steps: int = STEPS, least: int = BREAK) -> tuple[list[dict], list[dict]]:
    """Every gap under `line` inside each voiced run, and the phrases the breaks leave. `level` is [frames, steps];
    places are in steps, given back as frames (a step is 1/steps of a frame)."""
    flat = level.reshape(-1)
    gaps, phrases = [], []
    for n, (first, last) in enumerate(runs, 1):
        lo, hi = first * steps, (last + 1) * steps                     # the run in steps, one past its last
        under = ~(flat[lo:hi] > line)                                  # NaN counts as under: no sound is no voice
        start, here = None, []
        for k, u in enumerate(list(under) + [False]):
            if u and start is None:
                start = k
            elif not u and start is not None:
                here.append((lo + start, lo + k))
                start = None
        cuts = []
        for a, b in here:
            deep = flat[a:b]
            is_break = b - a >= least
            gaps.append({"run": n, "first_frame": round(a / steps, 2), "frames": round((b - a) / steps, 2),
                         "deepest_dbfs": None if np.all(np.isnan(deep)) else round(float(np.nanmin(deep)), 1), "a_break": bool(is_break)})
            if is_break:
                cuts.append((a, b))
        edge = lo
        for a, b in cuts + [(hi, hi)]:
            if a > edge:
                phrases.append({"phrase": len(phrases) + 1, "run": n, "run_frames": [first, last], "starts": round(edge / steps, 2), "ends": round(a / steps, 2),
                                "first_frame": edge // steps, "last_frame": (a - 1) // steps})
            edge = b
    return gaps, phrases


def dips_of(level: np.ndarray, runs: list[list[int]], line: float, steps: int = STEPS) -> tuple[np.ndarray, list[dict]]:
    """The sung level round every step, and every dip under it down to the floor that touches a voiced run.
    `level` is [frames, steps]; places are given back in frames."""
    flat = level.reshape(-1).astype(np.float64)
    n, half = len(flat), DIP_WINDOW_FRAMES * steps
    ref = np.full(n, np.nan)
    for k in range(n):
        seg = flat[max(k - half, 0):k + half + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            ref[k] = np.percentile(seg, DIP_PERCENTILE)
    depth = ref - flat
    voiced = np.zeros(n, bool)
    for lo, hi in runs:
        voiced[lo * steps:(hi + 1) * steps] = True
    under = np.nan_to_num(depth, nan=0.0) >= DIP_FLOOR
    out, start = [], None
    for k, u in enumerate(list(under) + [False]):
        if u and start is None:
            start = k
        elif not u and start is not None:
            a, b, start = start, k, None
            if b - a < DIP_LEAST or not voiced[a:b].any():
                continue
            deep = a + int(np.nanargmax(depth[a:b]))
            run = next((i for i, (lo, hi) in enumerate(runs, 1) if lo * steps < b and a < (hi + 1) * steps), None)
            out.append({"dip": len(out) + 1, "starts": round(a / steps, 2), "ends": round(b / steps, 2), "frames": round((b - a) / steps, 2),
                        "deepest_db_under_the_sung_level": round(float(depth[deep]), 1), "deepest_dbfs": round(float(flat[deep]), 1),
                        "deepest_at": round(deep / steps, 2), "the_sung_level_there_dbfs": round(float(ref[deep]), 1),
                        "frames_under_the_voice_tables_line": round(float((flat[a:b] <= line).sum()) / steps, 2),
                        "voiced_run": run, "wholly_inside_the_run": bool(voiced[a:b].all())})
    return ref.reshape(level.shape), out


def onset_curve(sound: np.ndarray, rate: int) -> tuple[np.ndarray, float]:
    """The onset curve of a sound, [samples, channels], and its steps a second: the rise of EACH channel's spectrum,
    summed. Not of the channels' mean, in which a hit that differs between them cancels and loses its start. Step k
    of the curve is the rise into the window whose MIDDLE is at k / steps-a-second + `HALF_WINDOW` seconds: every
    time read off the curve has that added."""
    from scipy import signal
    sound = sound if sound.ndim == 2 else sound[:, None]
    flux = None
    for c in range(sound.shape[1]):
        one = sound[:, c]
        if rate != BEAT_RATE:
            g = np.gcd(rate, BEAT_RATE)
            one = signal.resample_poly(one, BEAT_RATE // g, rate // g)
        _, _, z = signal.stft(one, fs=BEAT_RATE, nperseg=WINDOW, noverlap=WINDOW - HOP, boundary=None, padded=False)
        mag = np.log1p(1000.0 * np.abs(z))
        rise = np.maximum(mag[:, 1:] - mag[:, :-1], 0.0).sum(0)
        flux = rise if flux is None else flux + rise
    flux = np.concatenate([[0.0], flux])
    per_second = BEAT_RATE / HOP
    wide = max(int(per_second), 1)
    curve = np.maximum(flux - np.convolve(flux, np.ones(wide) / wide, mode="same"), 0.0)
    return curve.astype(np.float64), per_second


def tempo_of(curve: np.ndarray, per_second: float) -> tuple[float, list[dict]]:
    """The tempo the curve is most like itself at, leaning to `TEMPO_CENTER`, and the three strongest candidates unweighted."""
    x = curve - curve.mean()
    if not np.any(x):
        return float("nan"), []
    lags = np.arange(int(per_second * 60.0 / TEMPO_RANGE[1]), int(per_second * 60.0 / TEMPO_RANGE[0]) + 1)
    lags = lags[lags < len(x) - 1]
    own = float(np.dot(x, x))
    like = np.array([float(np.dot(x[:-lag], x[lag:])) / own for lag in lags])
    bpm = 60.0 * per_second / lags
    weight = np.exp(-0.5 * (np.log2(bpm / TEMPO_CENTER) / TEMPO_OCTAVES) ** 2)
    best = int(np.argmax(like * weight))
    peaks = [k for k in range(1, len(like) - 1) if like[k] >= like[k - 1] and like[k] >= like[k + 1]]
    top = sorted(peaks, key=lambda k: -like[k])[:3]
    return float(bpm[best]), [{"bpm": round(float(bpm[k]), 1), "likeness": round(float(like[k]), 3)} for k in top]


def track(curve: np.ndarray, per_second: float, bpm: float) -> np.ndarray:
    """Ellis's tracker: the beats' places on the curve, in its own steps."""
    period = 60.0 * per_second / bpm
    sd = max(period / 32.0, 1.0)
    win = np.exp(-0.5 * (np.arange(-int(4 * sd), int(4 * sd) + 1) / sd) ** 2)
    local = np.convolve(curve / (curve.std() or 1.0), win, mode="same")
    back = np.arange(-int(round(2 * period)), -int(round(period / 2)) + 1)
    cost = -TIGHTNESS * np.log(-back / period) ** 2
    score, prev = local.copy(), np.full(len(local), -1, int)
    for t in range(len(local)):
        at = t + back
        ok = at >= 0
        if not ok.any():
            continue
        cand = score[at[ok]] + cost[ok]
        k = int(np.argmax(cand))
        if cand[k] > 0:                      # a beat is chained to an earlier one only when that adds to it
            score[t] += cand[k]
            prev[t] = at[ok][k]
    t = int(np.argmax(score[-int(round(period)):]) + len(score) - int(round(period))) if len(score) > period else int(np.argmax(score))
    beats = [t]
    while prev[beats[-1]] >= 0:
        beats.append(int(prev[beats[-1]]))
    return np.array(beats[::-1], float)


def local_tempos(curve: np.ndarray, per_second: float) -> list[dict]:
    out, n, step = [], int(LOCAL_SECONDS * per_second), int(LOCAL_STEP * per_second)
    for a in range(0, max(len(curve) - n, 0) + 1, step):
        bpm, _ = tempo_of(curve[a:a + n], per_second)
        if not np.isnan(bpm):
            out.append({"from_second": round(a / per_second, 1), "to_second": round((a + n) / per_second, 1), "bpm": round(bpm, 1)})
    return out


def read_spans(folder: Path) -> dict:
    spans = folder / "voice_spans.json"
    if not spans.is_file():
        raise SystemExit(f"{spans} is not there: the voiced runs, the frame rate and the frame count are read from the voice table's own record "
                         "(bench/voice_spans.py spans)")
    return json.loads(spans.read_text())


def step_offset(a) -> int:
    import av
    out, rows = Path(a.out), []
    for name in a.copies:
        path = Path(name)
        with av.open(str(path)) as c:
            video, audio = (c.streams.video or [None])[0], (c.streams.audio or [None])[0]
            if video is None:
                raise SystemExit(f"{path} has no picture stream")
            row = {"copy": path.name, **about(path), "frames_a_second": float(video.average_rate), "has_sound": audio is not None}
            firsts = {}
            if audio is not None:
                for packet in c.demux(video, audio):
                    if packet.pts is None:
                        continue
                    kind = "video" if packet.stream is video else "audio"
                    seen = firsts.setdefault(kind, [])
                    if len(seen) < 64:                         # pictures can be stored out of order: the earliest of the first few
                        seen.append(float(packet.pts * packet.stream.time_base))
                    if all(len(firsts.get(k, [])) >= 64 for k in ("video", "audio")):
                        break
                v, s = min(firsts["video"]), min(firsts["audio"])
                row.update(first_picture_seconds=round(v, 6), first_sound_packet_seconds=round(s, 6), sound_starts_after_picture_seconds=round(s - v, 6),
                           sound_starts_after_picture_frames=round((s - v) * float(video.average_rate), 3),
                           the_streams_own_start_seconds={"video": None if video.start_time is None else round(float(video.start_time * video.time_base), 6),
                                                          "audio": None if audio.start_time is None else round(float(audio.start_time * audio.time_base), 6)})
        rows.append(row)
        print(f"{row['copy']}: " + (f"the first sound packet is {row['sound_starts_after_picture_seconds']:+.4f} s against the first picture "
                                    f"({row['sound_starts_after_picture_frames']:+.2f} of its frames; positive is the sound later)" if row["has_sound"] else "no sound stream"))
    import av as _av
    rec = record("offset", [Path(n) for n in a.copies], {"packets_read_for_the_earliest": 64}, None, {"copies": rows})
    rec["libraries"]["av"] = _av.__version__
    out.mkdir(parents=True, exist_ok=True)
    (out / "sound_offset.record.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"written: {out / 'sound_offset.record.json'}. It cannot see: {CANNOT['offset']}")
    return 0


def step_phrases(a) -> int:
    folder = Path(a.folder)
    spans = read_spans(folder)
    fps, frames, line, theirs = float(spans["fps"]), int(spans["frames"]), float(spans["threshold_dbfs"]), spans["voiced_spans_inclusive"]
    sound, rate = read_wav(folder / "vocals.wav")
    level = step_levels(sound, rate, fps, frames)
    runs = own_runs(level, line)
    gaps, phrases = gaps_and_phrases(level, runs, line)
    of = {}
    for p in phrases:
        for n in range(p["first_frame"], p["last_frame"] + 1):
            of.setdefault(n, p["phrase"])
    run_of = {n: k for k, (lo, hi) in enumerate(runs, 1) for n in range(lo, hi + 1)}
    table = folder / "phrases_per_frame.csv"
    with open(table, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame"] + [f"vocals_dbfs_step_{k + 1}_of_{STEPS}" for k in range(STEPS)] + ["voiced_run", "phrase"])
        for n in range(frames):
            w.writerow([n] + ["" if np.isnan(v) else round(float(v), 1) for v in level[n]] + [run_of.get(n, ""), of.get(n, "")])
    rec = record("phrases", [folder / "vocals.wav", folder / "voice_spans.json"],
                 {"steps_a_frame": STEPS, "a_break_is_a_gap_of_steps_or_more": BREAK, "a_break_is_frames": BREAK / STEPS, "the_line_dbfs": line,
                  "the_line_is": "the voice table's own (voice_spans.json threshold_dbfs)",
                  "the_level_is": "the power of both channels of the vocals stem. The voice table's own level (voice_per_frame.csv) and its voiced runs are of "
                                  "the two channels' MEAN, which reads lower wherever the channels differ; its line was set on that scale and is used here on "
                                  "this one, so a stretch can be over the line here and under it there",
                  "channels": int(sound.shape[1]), "fps": fps, "frames": frames,
                  "where_the_settings_are_from": "bench/clip_signals.py, beside each constant"},
                 table, {"voiced_runs": len(runs), "voiced_runs_inclusive": runs,
                         "the_voiced_runs_are": "this table's own, made from its own level by the voice table's rule (voice_spans.voiced_spans: its line, its "
                                                "join, its least length); NOT read from voice_spans.json, whose runs are of the channels' mean. A load's text "
                                                "leans on the voice table, which is not changed by this",
                         "against_the_voice_tables_runs": {"its_runs": len(theirs), **runs_differ(runs, theirs, frames)},
                         "phrases": phrases, "gaps": gaps})
    (folder / "phrases.record.json").write_text(json.dumps(rec, indent=1) + "\n")
    lengths = sorted(p["last_frame"] - p["first_frame"] + 1 for p in phrases)
    differ = runs_differ(runs, theirs, frames)
    print(f"the voiced runs are this table's own ({len(runs)}; the voice table has {len(theirs)}, and {differ['frames_that_differ']} frame(s) are voiced in one and not "
          "the other: its level is the channels' mean)")
    print(f"{len(phrases)} phrase(s) in {len(runs)} voiced run(s) over {frames} frames; {len(gaps)} gap(s) under {line} dBFS inside the runs, "
          f"{sum(1 for g in gaps if g['a_break'])} of them {BREAK / STEPS:g} frames or longer and taken as breaks"
          + (f"; a phrase is {lengths[0]} to {lengths[-1]} frames, {lengths[len(lengths) // 2]} at the middle" if lengths else ""))
    print(f"written: {table} and {folder / 'phrases.record.json'}. It cannot see: {CANNOT['phrases']}")
    return 0


def shown(place: float) -> float:
    """A place in frames as it is written: cut to a hundredth, never rounded up, so a place written is in the frame
    its whole part names (a place a thousandth short of a frame was written as the next frame's)."""
    return float(np.floor(float(place) * 100.0) / 100.0)


def step_dips(a) -> int:
    folder = Path(a.folder)
    spans = read_spans(folder)
    fps, frames, line = float(spans["fps"]), int(spans["frames"]), float(spans["threshold_dbfs"])
    sound, rate = read_wav(folder / "vocals.wav")
    level = step_levels(sound, rate, fps, frames)
    runs = own_runs(level, line)                                  # the same runs as the phrases table's: one scale
    ref, dips = dips_of(level, runs, line)
    # the vocals stem's own onset curve, on the clock of the beats' table, and its strength where each dip ends
    curve, per_second = onset_curve(sound, rate)
    top = float(np.percentile(curve, 99)) if np.any(curve) else 0.0
    at = lambda frame: int(np.clip(round((frame / fps - HALF_WINDOW) * per_second), 0, max(len(curve) - 1, 0)))        # noqa: E731
    onset = np.full(frames, np.nan)
    for n in range(frames):
        seg = curve[at(n):max(at(n + 1), at(n) + 1)]
        if len(seg):
            onset[n] = seg.max()
    for d in dips:
        seg = curve[at(d["ends"] - 0.5):at(d["ends"] + 1.5) + 1]
        d["vocal_onset_where_it_ends_over_the_songs_strong_onset"] = round(float(seg.max()) / top, 3) if len(seg) and top else None
    of = {}
    for d in dips:
        for n in range(int(d["starts"]), int(np.ceil(d["ends"]))):
            of.setdefault(n, d["dip"])
    table = folder / "dips_per_frame.csv"
    with open(table, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "the_sung_level_round_it_dbfs"] + [f"db_under_it_step_{k + 1}_of_{STEPS}" for k in range(STEPS)] + ["vocal_onset_strength", "dip"])
        for n in range(frames):
            cells = ["" if np.isnan(ref[n, k]) or np.isnan(level[n, k]) else round(float(ref[n, k] - level[n, k]), 1) for k in range(STEPS)]
            w.writerow([n, "" if np.all(np.isnan(ref[n])) else round(float(np.nanmax(ref[n])), 1)] + cells
                       + ["" if np.isnan(onset[n]) else round(float(onset[n]), 3), of.get(n, "")])
    rec = record("dips", [folder / "vocals.wav", folder / "voice_spans.json"],
                 {"steps_a_frame": STEPS, "the_sung_level_is_percentile": DIP_PERCENTILE, "within_frames_either_side": DIP_WINDOW_FRAMES,
                  "listed_from_db_under_it": DIP_FLOOR, "listed_from_steps": DIP_LEAST, "the_voice_tables_line_dbfs": line,
                  "the_level_is": "the power of both channels of the vocals stem, as in the phrases table",
                  "the_voiced_runs_are": "this tool's own, from that level by the voice table's rule, as in the phrases table",
                  "the_songs_strong_onset_is": "the 99th percentile of the vocals stem's own onset curve", "fps": fps, "frames": frames,
                  "where_the_settings_are_from": "bench/clip_signals.py, beside each constant"},
                 table, {"voiced_runs_inclusive": runs, "said": "a list, not a cut: the floor bounds what is listed and nothing here says which dip is a break between two sung lines",
                         "dips": dips})
    (folder / "dips.record.json").write_text(json.dumps(rec, indent=1) + "\n")
    inside = [d for d in dips if d["wholly_inside_the_run"]]
    over = [d for d in inside if d["frames_under_the_voice_tables_line"] < BREAK / STEPS]
    print(f"{len(dips)} dip(s) of {DIP_FLOOR:g} dB or more under the sung level round them, for {DIP_LEAST / STEPS:g} frame or more, touching a voiced run; "
          f"{len(inside)} wholly inside a run, and {len(over)} of those are under the voice table's line for less than the {BREAK / STEPS:g} frames a phrase break takes: "
          "the dips the phrases pass over")
    print(f"written: {table} and {folder / 'dips.record.json'}. It cannot see: {CANNOT['dips']}")
    return 0


def grid(sound: np.ndarray, rate: int) -> dict:
    """The onset curve, the tempo, the beats in seconds and the figures beside them. `sound` is [samples, channels]."""
    curve, per_second = onset_curve(sound, rate)
    bpm, candidates = tempo_of(curve, per_second)
    if np.isnan(bpm):
        return {"curve": curve, "per_second": per_second, "bpm": None, "candidates": [], "beats": np.array([]), "strength": np.array([]), "on_over_off": None,
                "local": [], "half": None, "steady": None, "every_other": None}
    places = track(curve, per_second, bpm)
    at = places.astype(int)
    off = np.ones(len(curve), bool)
    for k in (-1, 0, 1):
        off[np.clip(at + k, 0, len(curve) - 1)] = False
    around = lambda i: float(curve[max(int(i) - 1, 0):int(i) + 2].max())            # noqa: E731
    strength = np.array([around(i) for i in at])
    # the place half-way between each pair of beats, where the curve is also like itself at about half the period
    near = [c for c in candidates if abs(c["bpm"] / (2.0 * bpm) - 1.0) <= NEAR]
    half = None
    if near and len(places) > 1:
        mid = (places[:-1] + places[1:]) / 2.0
        half = {"places": mid / per_second + HALF_WINDOW, "strength": np.array([around(round(i)) for i in mid]), "the_candidates_near_twice_the_tempo": near}
    # the steady run: the longest run of beats each within STEADY_GAP of the middle gap from the beat before, and no weaker than STEADY_STRENGTH of the middle beat
    steady = None
    if len(places) > 2:
        gaps = np.diff(places)
        ok = np.concatenate([[False], np.abs(gaps / np.median(gaps) - 1.0) <= STEADY_GAP]) & (strength >= STEADY_STRENGTH * np.median(strength))
        ok[:-1] |= ok[1:] & (strength[:-1] >= STEADY_STRENGTH * np.median(strength))      # the beat a steady gap starts from is in the run too
        best, start = (0, 0), None
        for k, good in enumerate(list(ok) + [False]):
            if good and start is None:
                start = k
            elif not good and start is not None:
                best, start = max(best, (k - start, start)), None
        if best[0]:
            steady = (best[1], best[1] + best[0] - 1)
    # the other way round: a candidate near HALF the tempo, where every other beat may be the pulse a listener counts
    slower = [c for c in candidates if abs(c["bpm"] / (0.5 * bpm) - 1.0) <= NEAR]
    other = None if not slower or len(strength) < 4 else {
        "the_candidates_near_half_the_tempo": slower,
        "onset_strength_on_the_beats_counted_from_the_first": round(float(strength[0::2].mean()), 3),
        "onset_strength_on_the_beats_counted_from_the_second": round(float(strength[1::2].mean()), 3),
        "said": "the curve is as like itself two beats on as one: every other beat may be the pulse a listener counts. The strength of the beats on each of "
                "the two ways of taking every other one is given; which is the pulse these figures cannot say"}
    return {"curve": curve, "per_second": per_second, "bpm": bpm, "candidates": candidates, "beats": places / per_second + HALF_WINDOW, "strength": strength,
            "on_over_off": round(float(strength.mean() / (curve[off].mean() or 1e-9)), 2) if off.any() else None,
            "local": local_tempos(curve, per_second), "half": half, "steady": steady, "every_other": other}


def step_beats(a) -> int:
    folder = Path(a.folder)
    spans = read_spans(folder)
    fps, frames = float(spans["fps"]), int(spans["frames"])
    sound, rate = read_wav(folder / "rest.wav")
    g = grid(sound, rate)
    beat_frames = g["beats"] * fps                                   # a beat's place in frames, with its part of a frame
    # the per-frame onset column on the beats' own clock: a step of the curve is the rise into a window whose middle is HALF_WINDOW later
    onset = np.full(frames, np.nan)
    edges = np.clip(np.round((np.arange(frames + 1) / fps - HALF_WINDOW) * g["per_second"]).astype(int), 0, None)
    for n in range(frames):
        seg = g["curve"][edges[n]:max(edges[n + 1], edges[n] + 1)]
        if len(seg):
            onset[n] = seg.max()
    middle = float(np.median(g["strength"])) if len(g["strength"]) else None
    in_frame, half_in = {}, {}
    for k, (place, st) in enumerate(zip(beat_frames, g["strength"]), 1):
        if 0 <= int(place) < frames:
            in_frame.setdefault(int(place), (k, place, st, None if k == 1 else place - beat_frames[k - 2]))
    if g["half"] is not None:
        for place, st in zip(g["half"]["places"] * fps, g["half"]["strength"]):
            if 0 <= int(place) < frames:
                half_in.setdefault(int(place), (place, st))
    steady = g["steady"]
    table = folder / "beats_per_frame.csv"
    with open(table, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "onset_strength", "beat", "beat_at_frame", "beat_onset_strength", "beat_strength_over_the_middle_beat", "frames_since_the_beat_before",
                    "in_the_steady_run", "half_way_at_frame", "half_way_onset_strength"])
        for n in range(frames):
            b, h = in_frame.get(n), half_in.get(n)
            w.writerow([n, "" if np.isnan(onset[n]) else round(float(onset[n]), 3)]
                       + (["", "", "", "", "", ""] if b is None else [b[0], shown(b[1]), round(float(b[2]), 3), round(float(b[2]) / middle, 3) if middle else "",
                                                                      "" if b[3] is None else round(float(b[3]), 2),
                                                                      "yes" if steady and steady[0] <= b[0] - 1 <= steady[1] else ""])
                       + (["", ""] if h is None else [shown(h[0]), round(float(h[1]), 3)]))
    local = [x["bpm"] for x in g["local"]]
    more = {"tempo_bpm": None if g["bpm"] is None else round(g["bpm"], 2), "a_beat_is_frames": None if g["bpm"] is None else round(60.0 / g["bpm"] * fps, 3),
            "tempo_candidates_unweighted": g["candidates"], "beats": int(len(beat_frames)),
            "the_fit": {"onset_strength_on_the_beats_over_off_them": g["on_over_off"],
                        "local_tempo_bpm": {"stretches": g["local"], "least": min(local) if local else None, "most": max(local) if local else None,
                                            "said": "a tempo that holds gives one figure in every stretch. A spread here is one of three things, and the "
                                                    "figures say which only in part: the same pulse and its half or double (one stretch's figure about twice "
                                                    "another's, and nothing between), a tempo that moves, or a pulse the method loses; for the last two one "
                                                    "grid is a poor fit"}},
            "the_steady_run": None if steady is None else {
                "first_beat": steady[0] + 1, "last_beat": steady[1] + 1, "beats": steady[1] - steady[0] + 1,
                "from_frame": shown(beat_frames[steady[0]]), "to_frame": shown(beat_frames[steady[1]]),
                "beats_before_it": steady[0], "beats_after_it": int(len(beat_frames)) - 1 - steady[1],
                "said": "the longest run of beats each within the gap setting of the middle gap from the beat before and no weaker than the strength setting "
                        "of the middle beat. The tracker lays beats outside it too, on single sounds or on nothing: they are in the table with their own "
                        "strength and are not on a steady pulse by these figures"},
            "half_way": None if g["half"] is None else {
                "places": int(len(g["half"]["places"])), "the_candidates_near_twice_the_tempo": g["half"]["the_candidates_near_twice_the_tempo"],
                "onset_strength_half_way_over_on_the_beats": round(float(g["half"]["strength"].mean() / (g["strength"].mean() or 1e-9)), 3),
                "said": "the curve is also like itself at about half the period, so the place half-way between each pair of beats is written. It is a place "
                        "half-way, not a beat: a hit of another band can sit there, and which of the two a listener counts these figures cannot say"},
            "every_other_beat": g["every_other"],
            "no_count_in_fours": "none is given: no downbeat can be read without a trained model, and a count has no place to start from"}
    if a.shots:
        shots = json.loads(Path(a.shots).read_text())
        cuts = [s["first"] for s in shots["shots"] if 0 < s["first"] < frames]
        if len(beat_frames):
            offs = [{"cut": c, "nearest_beat_at_frame": round(float(beat_frames[np.argmin(np.abs(beat_frames - c))]), 2),
                     "frames_from_it": round(float(c - beat_frames[np.argmin(np.abs(beat_frames - c))]), 2)} for c in cuts]
            dist = sorted(abs(o["frames_from_it"]) for o in offs)
            more["cuts_against_the_grid"] = {"cuts": offs, "a_cut_is_frames_from_a_beat": {"least": dist[0], "middle": dist[len(dist) // 2], "most": dist[-1]},
                                             "cuts_placed_by_chance_would_be": {"middle": round(more["a_beat_is_frames"] / 4, 2), "most": round(more["a_beat_is_frames"] / 2, 2)},
                                             "read_from": about(Path(a.shots))}
    rec = record("beats", [folder / "rest.wav", folder / "voice_spans.json"] + ([Path(a.shots)] if a.shots else []),
                 {"the_onset_curve_is": "the rise of each channel's spectrum, summed; not of the channels' mean", "channels": int(sound.shape[1]),
                  "sound_rate": BEAT_RATE, "window": WINDOW, "step": HOP, "tempos_looked_for_bpm": list(TEMPO_RANGE), "leans_to_bpm": TEMPO_CENTER,
                  "lean_octaves": TEMPO_OCTAVES, "tightness": TIGHTNESS, "local_tempo_seconds": LOCAL_SECONDS, "local_tempo_step_seconds": LOCAL_STEP,
                  "near_twice_the_tempo": NEAR, "steady_gap": STEADY_GAP, "steady_strength": STEADY_STRENGTH, "half_window_seconds": HALF_WINDOW, "fps": fps, "frames": frames, "where_the_settings_are_from": "bench/clip_signals.py, beside each constant"}, table, more)
    (folder / "beats.record.json").write_text(json.dumps(rec, indent=1) + "\n")
    if g["bpm"] is None:
        print("no pulse: the onset curve is flat, and no beat is listed")
    else:
        print(f"tempo {more['tempo_bpm']} a minute, a beat every {more['a_beat_is_frames']} frames; {more['beats']} beat(s); the onset curve is "
              f"{g['on_over_off']} times as strong on the beats as off them; local tempo {more['the_fit']['local_tempo_bpm']['least']} to "
              f"{more['the_fit']['local_tempo_bpm']['most']} over {len(local)} stretch(es) of {LOCAL_SECONDS:g} s; candidates unweighted: {g['candidates']}")
        if more["the_steady_run"]:
            r = more["the_steady_run"]
            print(f"the steady run: beats {r['first_beat']} to {r['last_beat']}, frames {r['from_frame']} to {r['to_frame']}; {r['beats_before_it']} beat(s) before it and "
                  f"{r['beats_after_it']} after it are laid on no steady pulse")
        if more["half_way"]:
            h = more["half_way"]
            print(f"half-way places written: {h['places']} (a candidate near twice the tempo: {h['the_candidates_near_twice_the_tempo']}); the onset curve half-way is "
                  f"{h['onset_strength_half_way_over_on_the_beats']} of its strength on the beats. A place half-way is not called a beat")
        else:
            print("no half-way places written: no tempo candidate lies near twice the tempo")
        if more["every_other_beat"]:
            e = more["every_other_beat"]
            print(f"a candidate lies near half the tempo ({e['the_candidates_near_half_the_tempo']}): every other beat may be the pulse; the beats' strength counted from "
                  f"the first is {e['onset_strength_on_the_beats_counted_from_the_first']} and from the second {e['onset_strength_on_the_beats_counted_from_the_second']}")
        if "cuts_against_the_grid" in more:
            c = more["cuts_against_the_grid"]
            print(f"the record's {len(c['cuts'])} cut(s) against the grid: a cut is {c['a_cut_is_frames_from_a_beat']['least']} to {c['a_cut_is_frames_from_a_beat']['most']} "
                  f"frames from a beat, {c['a_cut_is_frames_from_a_beat']['middle']} at the middle; cuts placed by chance would be "
                  f"{c['cuts_placed_by_chance_would_be']['middle']} at the middle")
    print(f"written: {table} and {folder / 'beats.record.json'}. It cannot see: {CANNOT['beats']}")
    return 0


def edge_figures(frame: int, phrases: list[dict], runs: list[list[int]], beats: np.ndarray) -> dict:
    """What the signals say of one frame: its run, its phrase, and the beats either side."""
    out: dict = {"frame": frame}
    run = next(((k, r) for k, r in enumerate(runs, 1) if r[0] <= frame <= r[1]), None)
    out["voiced_run"] = None if run is None else {"run": run[0], "frames": run[1], "frames_from_its_start": frame - run[1][0], "frames_to_its_end": run[1][1] - frame}
    ph = next((p for p in phrases if p["first_frame"] <= frame <= p["last_frame"]), None)
    out["phrase"] = None if ph is None else {"phrase": ph["phrase"], "frames": [ph["first_frame"], ph["last_frame"]], "frames_from_its_start": frame - ph["first_frame"],
                                             "frames_to_its_end": ph["last_frame"] - frame}
    before = [p for p in phrases if p["last_frame"] < frame]
    after = [p for p in phrases if p["first_frame"] > frame]
    out["phrase_before"] = None if not before else {"phrase": before[-1]["phrase"], "ended_frames_ago": frame - before[-1]["last_frame"]}
    out["phrase_after"] = None if not after else {"phrase": after[0]["phrase"], "starts_in_frames": after[0]["first_frame"] - frame}
    if len(beats):
        b, f = beats[beats <= frame], beats[beats > frame]
        out["beat_before"] = None if not len(b) else {"at_frame": round(float(b[-1]), 2), "frames_ago": round(float(frame - b[-1]), 2)}
        out["beat_after"] = None if not len(f) else {"at_frame": round(float(f[0]), 2), "in_frames": round(float(f[0] - frame), 2)}
    return out


def read_signals(folder: Path) -> tuple[list[dict], list[list[int]], np.ndarray, list[str], np.ndarray, list[dict] | None]:
    absent, phrases, beats, halves, dips = [], [], np.array([]), np.array([]), None
    if (folder / "dips.record.json").is_file():
        dips = json.loads((folder / "dips.record.json").read_text())["dips"]
    else:
        absent.append("dips (no dips.record.json: `clip_signals.py dips`)")
    runs = read_spans(folder)["voiced_spans_inclusive"]
    if (folder / "phrases.record.json").is_file():
        made = json.loads((folder / "phrases.record.json").read_text())
        phrases, runs = made["phrases"], made.get("voiced_runs_inclusive", runs)      # the tool's own runs, on the phrases' scale
    else:
        absent.append("phrases (no phrases.record.json: `clip_signals.py phrases`)")
    if (folder / "beats_per_frame.csv").is_file():
        with open(folder / "beats_per_frame.csv") as fh:
            table = list(csv.DictReader(fh))
        beats = np.array([float(r["beat_at_frame"]) for r in table if r["beat_at_frame"] != ""])
        halves = np.array([float(r["half_way_at_frame"]) for r in table if r.get("half_way_at_frame", "") != ""])
    else:
        absent.append("beats (no beats_per_frame.csv: `clip_signals.py beats`)")
    return phrases, runs, beats, absent, halves, dips


def step_at(a) -> int:
    folder = Path(a.folder)
    first, _, last = a.frames.partition("-")
    if not (first.isdigit() and last.isdigit()) or int(first) > int(last):
        raise SystemExit(f"--frames {a.frames!r}: write FIRST-LAST, two source frames")
    first, last = int(first), int(last)
    phrases, runs, beats, absent, halves, dips = read_signals(folder)
    shot = None
    if a.shots:
        shot = next((s for s in json.loads(Path(a.shots).read_text())["shots"] if s["first"] <= first <= s["last"]), None)
    print(f"{a.name or 'a load'}: frames {first}-{last} ({last - first + 1}), read from {folder}" + (f"; its shot in the record: {shot['first']}-{shot['last']}" if shot else ""))
    for word in absent:
        print(f"  ABSENT, and said so: {word}")
    for which, frame in (("FIRST", first), ("LAST", last)):
        e = edge_figures(frame, phrases, runs, beats)
        r, p = e["voiced_run"], e["phrase"]
        print(f"  THE {which} FRAME, {frame}:")
        print("    voice: " + ("not in a voiced run" if r is None else f"in the voiced run {r['frames'][0]}-{r['frames'][1]}, {r['frames_from_its_start']} frames after its start and "
                                                                     f"{r['frames_to_its_end']} before its end"))
        if "phrases" not in " ".join(absent):
            print("    phrase: " + (f"in phrase {p['phrase']} ({p['frames'][0]}-{p['frames'][1]}), {p['frames_from_its_start']} frames after its start and {p['frames_to_its_end']} before its end"
                                    if p else "in no phrase" + (f"; the one before ended {e['phrase_before']['ended_frames_ago']} frames ago" if e["phrase_before"] else "")
                                    + (f"; the next starts in {e['phrase_after']['starts_in_frames']} frames" if e["phrase_after"] else "")))
        if len(beats):
            print("    beat: " + "; ".join(x for x in (f"the one before at frame {e['beat_before']['at_frame']}, {e['beat_before']['frames_ago']} frames ago" if e.get("beat_before") else "",
                                                       f"the next at frame {e['beat_after']['at_frame']}, in {e['beat_after']['in_frames']} frames" if e.get("beat_after") else "") if x))
        if which == "FIRST":
            for what, to in (("the phrase's start", p["frames"][0] if p else None), ("the beat before", int(np.floor(e["beat_before"]["at_frame"])) if e.get("beat_before") else None)):
                if to is None or to == frame:
                    continue
                where = "" if shot is None else (", inside the same shot" if to >= shot["first"] else f", which is BEFORE THE SHOT'S FIRST FRAME {shot['first']}: reaching it is a "
                                                 "load across a cut, another tool's")
                print(f"    to begin on {what}: {frame - to} frame(s) earlier, on frame {to}{where}")
        else:
            if p and p["frames_to_its_end"] > 0:
                where = "" if shot is None else (", inside the same shot" if p["frames"][1] <= shot["last"] else f", past the shot's last frame {shot['last']}")
                print(f"    the phrase runs {p['frames_to_its_end']} frame(s) past the load's last frame, to frame {p['frames'][1]}{where}")
    # what lies between the two edges, each with its place counted from the load's first frame
    inside = []
    for p in phrases:
        if first < p["first_frame"] <= last:
            inside.append((p["first_frame"], f"phrase {p['phrase']} STARTS on frame {p['first_frame']}"))
        if first <= p["last_frame"] < last:
            inside.append((p["last_frame"], f"phrase {p['phrase']} ENDS on frame {p['last_frame']}"))
    print("  INSIDE THE LOAD, between its first and last frame (the load's own frame in brackets, counted from 0):")
    if "phrases" not in " ".join(absent):
        print("    phrases: " + ("; ".join(f"{text} [{at - first}]" for at, text in sorted(inside)) or "none starts or ends inside it"))
    if dips is not None:
        # every listed dip that touches the load or lies within a second after its last frame, where the held tail is
        near = [d for d in dips if d["ends"] > first and d["starts"] < last + 1 + 24]
        print(f"    dips under the sung level (a list down to its floor, nothing cut; the last ones lie after the load's end): {len(near)}"
              + ("".join(f"\n      from {d['starts']:g} for {d['frames']:g} frames, {d['deepest_db_under_the_sung_level']:g} dB under at its deepest ({d['deepest_dbfs']:g} dBFS), "
                         f"{d['frames_under_the_voice_tables_line']:g} of them under the voice table's line; the voice's onset where it ends "
                         f"{d['vocal_onset_where_it_ends_over_the_songs_strong_onset']} of a strong one [{d['starts'] - first:g}]" for d in near)))
    if len(beats):
        here = [b for b in beats if first <= b < last + 1]
        print(f"    beats: {len(here)}" + (": " + ", ".join(f"{b:g} [{b - first:g}]" for b in here) if here else ""))
        if len(halves):
            there = [b for b in halves if first <= b < last + 1]
            print(f"    places half-way between beats (not beats; a hit of another band can sit there): {len(there)}" + (": " + ", ".join(f"{b:g} [{b - first:g}]" for b in there) if there else ""))
        else:
            print("    places half-way between beats: none is written for this clip")
    print("  These are figures. Nothing here says where an edge should be.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[0])
    sub = ap.add_subparsers(dest="what", required=True)
    o = sub.add_parser("offset")
    o.add_argument("copies", nargs="+")
    o.add_argument("--out", required=True)
    for name in ("phrases", "dips", "beats", "at"):
        p = sub.add_parser(name)
        p.add_argument("folder")
        if name in ("beats", "at"):
            p.add_argument("--shots")
        if name == "at":
            p.add_argument("--frames", required=True, metavar="FIRST-LAST")
            p.add_argument("--name")
    a = ap.parse_args()
    return {"offset": step_offset, "phrases": step_phrases, "dips": step_dips, "beats": step_beats, "at": step_at}[a.what](a)


if __name__ == "__main__":
    sys.exit(main())
