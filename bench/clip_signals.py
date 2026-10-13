#!/usr/bin/env python3
"""Signals of a clip's own sound, a row a source frame, for whoever sets a window's first and last frame: phrases, a beat grid, the sound's offset to the picture.

    <python> bench/clip_signals.py offset --out AUDIO_DIR COPY [COPY ...]          # sound_offset.record.json
    <python> bench/clip_signals.py phrases AUDIO_DIR                               # phrases_per_frame.csv + phrases.record.json
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

**phrases.** The vocals stem's level at `STEPS` steps a frame, where the voice table has one. Inside each voiced
run of `voice_spans.json`, every stretch under that table's own line is a GAP, listed with its length and its
deepest level; a gap of `BREAK` steps or more is a phrase break, and the stretches between breaks are the
phrases. It cannot see: a break sung through with no drop in level; a held note from a new phrase; whose voice
it is, or a backing voice (the stem holds every voice); a breath the separation put in the other stem.

**beats.** From the rest stem: an onset curve (the rise of the spectrum from one short step to the next), a
tempo from that curve's likeness to itself a beat later, and a grid laid by the tracker of Ellis (2007), which
walks the curve keeping beats near one period apart. Beside the grid: the tempo, the three strongest tempo
candidates (a tempo and its double or half are the same music to this method), how much stronger the curve is on
the beats than off them, and the tempo found in each stretch of `LOCAL_SECONDS`, so a track whose tempo moves
shows as a spread and the one grid is read as the poor fit it is. NO DOWNBEAT is claimed: for each of the four
ways of counting the beats in fours the record gives the low band's level on that count, a figure and no more.
With `--shots` the record's cuts are printed against the grid once, with what cuts placed by chance would give.
It cannot see: a tempo that drifts or changes (one fit); a bar's first beat; music with no steady pulse.

**at.** The figures for a load's two edges, printed and not stored: the voiced run and the phrase each edge falls
in and the frames from that phrase's start and to its end; the nearest beat before and after; for the first
frame, how many frames earlier the load would begin on the phrase's start and on the beat before, and whether
that frame is inside the same shot (`--shots`); for the last frame, the same looking forward, since the frames
after a load's end are held open. A reader's print: summaries are never stored.

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
#: The onset curve's sound rate, window and step. Inherited: the usual values for a beat tracker's curve
#: (Ellis 2007 used 8 kHz and 4 ms; these are the common later ones at half the stems' rate).
BEAT_RATE, WINDOW, HOP = 22050, 1024, 256
#: The tempos looked for, a minute's beats, and the tempo the search leans to with its width in octaves.
#: Inherited: Ellis 2007 (a log-normal weight about 120, one octave wide).
TEMPO_RANGE, TEMPO_CENTER, TEMPO_OCTAVES = (50.0, 220.0), 120.0, 1.0
#: How hard the tracker holds beats to one period apart. Inherited: Ellis 2007's own value.
TIGHTNESS = 100.0
#: The stretch a local tempo is read in, and its step. Reasoned: long enough to hold a dozen beats at the slowest
#: tempo looked for, short enough that a tempo moving over a minute shows.
LOCAL_SECONDS, LOCAL_STEP = 16.0, 8.0
#: The low band whose level is given for each way of counting in fours, in Hz. Reasoned: a kick drum and a bass
#: note sit under it; nothing was measured on a track to set it.
LOW_HZ = 150.0

CANNOT = {
    "offset": "what is heard: an encoder's priming samples sit in the first packet's time and are not sound, and an edit list can hide them from a player; "
              "a drift later in the file (this is the start only)",
    "phrases": "a break sung through with no drop in level; a held note from a new phrase; whose voice it is, or a backing voice (the stem holds every voice); "
               "a breath the separation put in the other stem",
    "beats": "a tempo that drifts or changes (the grid is one fit; the local tempos show it and do not mend it); a bar's first beat (no downbeat is claimed); "
             "music with no steady pulse; a tempo from its double or its half",
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
           "settings": settings, "cannot_see": CANNOT[what],
           "rows_are": "source frames, counted from 0; an empty cell has nothing to say and is not a zero"}
    if table is not None:
        out["table"] = {"file": table.name, "sha256": about(table)["sha256"]}
        out["fingerprint"] = out["table"]["sha256"]
    out.update(more)
    return out


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """A wav as float mono in -1..1, and its rate."""
    from scipy.io import wavfile
    rate, data = wavfile.read(str(path))
    data = data.astype(np.float32) / (32768.0 if data.dtype == np.int16 else 1.0)
    return (data.mean(1) if data.ndim == 2 else data), int(rate)


def step_levels(mono: np.ndarray, rate: int, fps: float, frames: int, steps: int = STEPS) -> np.ndarray:
    """The level in dBFS of each of `steps` equal stretches of each video frame: [frames, steps]. A stretch past the
    sound's end is NaN: there is nothing to read there."""
    out = np.full((frames, steps), np.nan, np.float32)
    for k in range(frames * steps):
        seg = mono[int(round(k * rate / fps / steps)):int(round((k + 1) * rate / fps / steps))]
        if len(seg):
            out[k // steps, k % steps] = 20.0 * np.log10(max(float(np.sqrt(np.mean(seg ** 2))), 1e-7))
    return out


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


def onset_curve(mono: np.ndarray, rate: int) -> tuple[np.ndarray, float, np.ndarray]:
    """The onset curve of a signal, its steps a second, and the low band's level at each step."""
    from scipy import signal
    if rate != BEAT_RATE:
        g = np.gcd(rate, BEAT_RATE)
        mono = signal.resample_poly(mono, BEAT_RATE // g, rate // g)
    f, _, z = signal.stft(mono, fs=BEAT_RATE, nperseg=WINDOW, noverlap=WINDOW - HOP, boundary=None, padded=False)
    mag = np.log1p(1000.0 * np.abs(z))
    flux = np.maximum(mag[:, 1:] - mag[:, :-1], 0.0).sum(0)
    flux = np.concatenate([[0.0], flux])
    per_second = BEAT_RATE / HOP
    wide = max(int(per_second), 1)
    curve = np.maximum(flux - np.convolve(flux, np.ones(wide) / wide, mode="same"), 0.0)
    low = 20.0 * np.log10(np.maximum(np.abs(z[f <= LOW_HZ]).mean(0), 1e-7))
    return curve.astype(np.float64), per_second, low


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
    fps, frames, line, runs = float(spans["fps"]), int(spans["frames"]), float(spans["threshold_dbfs"]), spans["voiced_spans_inclusive"]
    mono, rate = read_wav(folder / "vocals.wav")
    level = step_levels(mono, rate, fps, frames)
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
                  "the_line_is": "the voice table's own (voice_spans.json threshold_dbfs)", "fps": fps, "frames": frames,
                  "where_the_settings_are_from": "bench/clip_signals.py, beside each constant"},
                 table, {"voiced_runs": len(runs), "phrases": phrases, "gaps": gaps})
    (folder / "phrases.record.json").write_text(json.dumps(rec, indent=1) + "\n")
    lengths = sorted(p["last_frame"] - p["first_frame"] + 1 for p in phrases)
    print(f"{len(phrases)} phrase(s) in {len(runs)} voiced run(s) over {frames} frames; {len(gaps)} gap(s) under {line} dBFS inside the runs, "
          f"{sum(1 for g in gaps if g['a_break'])} of them {BREAK / STEPS:g} frames or longer and taken as breaks"
          + (f"; a phrase is {lengths[0]} to {lengths[-1]} frames, {lengths[len(lengths) // 2]} at the middle" if lengths else ""))
    print(f"written: {table} and {folder / 'phrases.record.json'}. It cannot see: {CANNOT['phrases']}")
    return 0


def grid(mono: np.ndarray, rate: int) -> dict:
    """The onset curve, the tempo, the beats in seconds and the figures beside them."""
    curve, per_second, low = onset_curve(mono, rate)
    bpm, candidates = tempo_of(curve, per_second)
    if np.isnan(bpm):
        return {"curve": curve, "per_second": per_second, "bpm": None, "candidates": [], "beats": np.array([]), "strength": np.array([]), "on_over_off": None,
                "local": [], "phases": []}
    places = track(curve, per_second, bpm)
    at = places.astype(int)
    off = np.ones(len(curve), bool)
    for k in (-1, 0, 1):
        off[np.clip(at + k, 0, len(curve) - 1)] = False
    strength = np.array([curve[max(i - 1, 0):i + 2].max() for i in at])
    phases = [{"count_from_beat": k + 1, "beats": int(len(at[k::4])), "low_band_dbfs_on_them": round(float(np.mean(low[np.clip(at[k::4], 0, len(low) - 1)])), 1)}
              for k in range(4) if len(at[k::4])]
    return {"curve": curve, "per_second": per_second, "bpm": bpm, "candidates": candidates, "beats": places / per_second + (WINDOW / 2) / BEAT_RATE, "strength": strength,
            "on_over_off": round(float(strength.mean() / (curve[off].mean() or 1e-9)), 2) if off.any() else None,
            "local": local_tempos(curve, per_second), "phases": phases}


def step_beats(a) -> int:
    folder = Path(a.folder)
    spans = read_spans(folder)
    fps, frames = float(spans["fps"]), int(spans["frames"])
    mono, rate = read_wav(folder / "rest.wav")
    g = grid(mono, rate)
    beat_frames = g["beats"] * fps                                   # a beat's place in frames, with its part of a frame
    onset = np.full(frames, np.nan)
    edges = np.round(np.arange(frames + 1) / fps * g["per_second"]).astype(int)
    for n in range(frames):
        seg = g["curve"][edges[n]:max(edges[n + 1], edges[n] + 1)]
        if len(seg):
            onset[n] = seg.max()
    in_frame = {}
    for k, (place, s) in enumerate(zip(beat_frames, g["strength"]), 1):
        if 0 <= int(place) < frames:
            in_frame.setdefault(int(place), (k, place, s))
    table = folder / "beats_per_frame.csv"
    with open(table, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "onset_strength", "beat", "beat_at_frame", "beat_onset_strength", "count_in_fours_from_the_first_beat"])
        for n in range(frames):
            b = in_frame.get(n)
            w.writerow([n, "" if np.isnan(onset[n]) else round(float(onset[n]), 3)]
                       + (["", "", "", ""] if b is None else [b[0], round(float(b[1]), 2), round(float(b[2]), 3), (b[0] - 1) % 4 + 1]))
    local = [x["bpm"] for x in g["local"]]
    more = {"tempo_bpm": None if g["bpm"] is None else round(g["bpm"], 2), "a_beat_is_frames": None if g["bpm"] is None else round(60.0 / g["bpm"] * fps, 3),
            "tempo_candidates_unweighted": g["candidates"], "beats": int(len(beat_frames)),
            "the_fit": {"onset_strength_on_the_beats_over_off_them": g["on_over_off"],
                        "local_tempo_bpm": {"stretches": g["local"], "least": min(local) if local else None, "most": max(local) if local else None,
                                            "said": "a tempo that holds gives one figure in every stretch; a spread here is a tempo that moves or a pulse the method "
                                                    "loses, and one grid is then a poor fit"}},
            "counting_in_fours": {"said": "no downbeat is claimed: the low band's level on each of the four ways of counting, a figure", "the_four": g["phases"]}}
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
                 {"sound_rate": BEAT_RATE, "window": WINDOW, "step": HOP, "tempos_looked_for_bpm": list(TEMPO_RANGE), "leans_to_bpm": TEMPO_CENTER,
                  "lean_octaves": TEMPO_OCTAVES, "tightness": TIGHTNESS, "local_tempo_seconds": LOCAL_SECONDS, "local_tempo_step_seconds": LOCAL_STEP,
                  "low_band_hz": LOW_HZ, "fps": fps, "frames": frames, "where_the_settings_are_from": "bench/clip_signals.py, beside each constant"}, table, more)
    (folder / "beats.record.json").write_text(json.dumps(rec, indent=1) + "\n")
    if g["bpm"] is None:
        print("no pulse: the onset curve is flat, and no beat is listed")
    else:
        print(f"tempo {more['tempo_bpm']} a minute, a beat every {more['a_beat_is_frames']} frames; {more['beats']} beat(s); the onset curve is "
              f"{g['on_over_off']} times as strong on the beats as off them; local tempo {more['the_fit']['local_tempo_bpm']['least']} to "
              f"{more['the_fit']['local_tempo_bpm']['most']} over {len(local)} stretch(es) of {LOCAL_SECONDS:g} s; candidates unweighted: {g['candidates']}")
        print("counting in fours, the low band on each count (no downbeat is claimed): " + "; ".join(f"from beat {p['count_from_beat']}: {p['low_band_dbfs_on_them']} dBFS" for p in g["phases"]))
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


def read_signals(folder: Path) -> tuple[list[dict], list[list[int]], np.ndarray, list[str]]:
    absent, phrases, beats = [], [], np.array([])
    runs = read_spans(folder)["voiced_spans_inclusive"]
    if (folder / "phrases.record.json").is_file():
        phrases = json.loads((folder / "phrases.record.json").read_text())["phrases"]
    else:
        absent.append("phrases (no phrases.record.json: `clip_signals.py phrases`)")
    if (folder / "beats_per_frame.csv").is_file():
        with open(folder / "beats_per_frame.csv") as fh:
            beats = np.array([float(r["beat_at_frame"]) for r in csv.DictReader(fh) if r["beat_at_frame"] != ""])
    else:
        absent.append("beats (no beats_per_frame.csv: `clip_signals.py beats`)")
    return phrases, runs, beats, absent


def step_at(a) -> int:
    folder = Path(a.folder)
    first, _, last = a.frames.partition("-")
    if not (first.isdigit() and last.isdigit()) or int(first) > int(last):
        raise SystemExit(f"--frames {a.frames!r}: write FIRST-LAST, two source frames")
    first, last = int(first), int(last)
    phrases, runs, beats, absent = read_signals(folder)
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
    print("  These are figures. Nothing here says where an edge should be.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[0])
    sub = ap.add_subparsers(dest="what", required=True)
    o = sub.add_parser("offset")
    o.add_argument("copies", nargs="+")
    o.add_argument("--out", required=True)
    for name in ("phrases", "beats", "at"):
        p = sub.add_parser(name)
        p.add_argument("folder")
        if name != "phrases":
            p.add_argument("--shots")
        if name == "at":
            p.add_argument("--frames", required=True, metavar="FIRST-LAST")
            p.add_argument("--name")
    a = ap.parse_args()
    return {"offset": step_offset, "phrases": step_phrases, "beats": step_beats, "at": step_at}[a.what](a)


if __name__ == "__main__":
    sys.exit(main())
