#!/usr/bin/env python3
"""`bench/clip_signals.py` on made sound: each case is a way a phrase edge, a beat or an offset could be a frame out, or a poor fit could read as a good one.

  phrases_come_back_to_the_frame        tone bursts with silences on known frames: each burst is a phrase on its own frames, a gap of
                                        `BREAK` steps or more is a break, a shorter one is LISTED as a gap and breaks nothing, and a
                                        phrase's edge inside a frame is given to the step
  a_clip_with_no_voice_has_no_phrase    a silent vocals stem: no phrase, no gap, and every frame still has its row, the phrase cell empty
  a_row_a_frame_and_no_more             both tables have exactly the video's frames as rows; a beat's cells are empty where no beat falls
  clicks_come_back_as_their_tempo_and_grid   clicks at a known tempo: that tempo, a beat on every click to within half a frame, and
                                        the same local tempo in every stretch
  a_drifting_click_is_reported          clicks whose tempo climbs: the local tempos spread across the climb and the beats sit farther
                                        from the clicks than the steady track's; nothing calls it a grid that fits
  silence_has_no_pulse                  a rest stem with no sound: no tempo, no beat, the beat cells empty
  the_record_says_what_it_read          each record names its inputs with their hashes, the settings, what the signal cannot see, and a
                                        fingerprint that is the table's own hash
  an_edge_is_printed_not_ruled          `at` prints, for a load's first and last frame, the run and the phrase with the frames to their
                                        ends and the beats either side; a start before the shot's first frame is said to cross a cut;
                                        an absent signal is said absent
  the_offset_is_the_containers          a made copy whose sound starts a known time after its picture gives that time, with its sign; a
                                        hundredth of a second is read as that and not as nothing; a copy with no sound says so and gives
                                        no figure; the record has no line about rows, having none
  the_level_is_both_channels_power      two channels that cancel in their mean still read at their level, and their phrase is found
  two_frames_do_not_break_three_do      a gap of exactly `BREAK` steps is a break and one a step short is not; two tones back to back with
                                        no gap are ONE phrase, the method's own blind spot seen as such
  one_clock_in_one_table                a beat's row is the row whose onset is the strongest round it, and a place is written in the frame
                                        its whole part names, never rounded up onto the next
  an_accent_every_second_click          the tempo comes back as the accents', the place half-way between each pair of beats is written with
                                        the weak click on it and is not called a beat; a track with nothing half-way gets no such column
  beats_before_the_pulse_are_said       clicks that start late after two stray sounds: the steady run begins where the pulse does, and the
                                        beats laid before it are in the table with their own strength, outside the run

    <comfy venv python> bench/check_clip_signals.py

numpy, scipy and PyAV; ffmpeg for the one made copy. No model, no weights, no card.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402

import clip_signals as cs  # noqa: E402
import voice_spans as vs  # noqa: E402

RATE, FPS = vs.RATE, 24.0
TOOL = Path(__file__).resolve().parent / "clip_signals.py"


def tones(frames: int, on: list[tuple[float, float]], amp: float = 0.1) -> np.ndarray:
    """A mono signal of `frames` video frames with a tone on [first, one past the last) in frames, which may fall inside a frame."""
    n = int(round(frames * RATE / FPS))
    out = np.zeros(n, np.float32)
    t = np.arange(n) / RATE
    for a, b in on:
        lo, hi = int(round(a * RATE / FPS)), int(round(b * RATE / FPS))
        out[lo:hi] = amp * np.sin(2 * np.pi * 440.0 * t[lo:hi])
    return out


def clicks(times: np.ndarray, seconds: float, amps=None) -> np.ndarray:
    """A short burst of noise at each time, on silence; `amps` is each burst's size where they are not all one."""
    rng = np.random.default_rng(7)
    out = np.zeros(int(seconds * RATE), np.float32)
    burst = (rng.standard_normal(int(0.02 * RATE)) * np.exp(-np.arange(int(0.02 * RATE)) / (0.004 * RATE))).astype(np.float32) * 0.5
    for n, t in enumerate(times):
        k = int(round(t * RATE))
        if k + len(burst) < len(out):
            out[k:k + len(burst)] += burst * (1.0 if amps is None else float(amps[n]))
    return out


def folder_of(root: Path, name: str, vocals: np.ndarray, rest: np.ndarray, frames: int, right: np.ndarray | None = None, runs: list | None = None) -> Path:
    """An audio folder as voice_spans leaves one: the two stems and the voice table's record, made by its own functions.
    `right` is the vocals' other channel where it is not the same; `runs` the voiced runs where they are given."""
    d = root / name
    d.mkdir()
    vs.write_wav(d / "vocals.wav", np.stack([vocals, vocals if right is None else right]))
    vs.write_wav(d / "rest.wav", np.stack([rest, rest]))
    level = vs.frame_levels(vocals, RATE, FPS)[:frames]
    (d / "voice_spans.json").write_text(json.dumps({"fps": FPS, "frames": frames, "threshold_dbfs": vs.THRESHOLD_DB,
                                                    "voiced_spans_inclusive": runs if runs is not None else vs.voiced_spans(level)}))
    return d


def run(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOL), *argv], capture_output=True, text=True)


def rows(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    FRAMES = 240
    # one voiced run of 20-139: a phrase 20-59.5, a breath of 4.5 frames, a phrase 64-99, a one-frame stop inside the next
    # sound (a gap, not a break), a phrase to 139; then silence, and a second run 180-209
    sung = tones(FRAMES, [(20, 59.5), (64, 100), (101, 140), (180, 210)])
    steady_t = np.arange(0.5, 39.5, 0.5) + 0.5 / FPS                             # 120 a minute, each click in the middle of a frame
    drift_t = np.cumsum(60.0 / np.linspace(96.0, 150.0, 80)) + 0.5               # 96 climbing to 150
    seconds = 40.0
    voiced = folder_of(root, "voiced", sung, clicks(steady_t, FRAMES / FPS), FRAMES)
    silent = folder_of(root, "silent", np.zeros_like(sung), np.zeros_like(sung), FRAMES)
    steady = folder_of(root, "steady", np.zeros(int(seconds * RATE), np.float32), clicks(steady_t, seconds), int(seconds * FPS))
    drift = folder_of(root, "drift", np.zeros(int(seconds * RATE), np.float32), clicks(drift_t[drift_t < seconds - 0.5], seconds), int(seconds * FPS))
    shots = root / "shots.json"
    shots.write_text(json.dumps({"shots": [{"first": 0, "last": 59}, {"first": 60, "last": 239}]}))
    # an accent on every second click; and the same with the click between so faint that it is a hit half-way and not a beat
    accent = folder_of(root, "accent", np.zeros(int(seconds * RATE), np.float32), clicks(steady_t, seconds, [1.0 if k % 2 == 0 else 0.3 for k in range(len(steady_t))]),
                       int(seconds * FPS))
    faint = folder_of(root, "faint", np.zeros(int(seconds * RATE), np.float32), clicks(steady_t, seconds, [1.0 if k % 2 == 0 else 0.03 for k in range(len(steady_t))]),
                      int(seconds * FPS))
    slow_t = np.arange(1.0, 39.0, 1.0) + 0.5 / FPS                               # 60 a minute, nothing half-way
    slow = folder_of(root, "slow", np.zeros(int(seconds * RATE), np.float32), clicks(slow_t, seconds), int(seconds * FPS))
    # two stray sounds, then nothing, then the pulse from 12 s
    late_t = np.arange(12.0, 39.0, 0.5) + 0.5 / FPS
    late = folder_of(root, "late", np.zeros(int(seconds * RATE), np.float32), clicks(np.concatenate([[1.3, 4.9], late_t]), seconds), int(seconds * FPS))
    # two channels that cancel in their mean: a tone on the left and the same tone turned over on the right, with a silence of 6 frames
    left = tones(FRAMES, [(20, 40), (46, 60)])
    cancel = folder_of(root, "cancel", left, np.zeros_like(left), FRAMES, right=-left, runs=[[20, 59]])
    # gaps of exactly two frames and exactly three; then two tones back to back with no gap
    edge = tones(FRAMES, [(20, 40), (42, 60), (63, 80)])
    back = edge.copy()
    n0, n1 = int(round(100 * RATE / FPS)), int(round(120 * RATE / FPS))
    back[n0:n1] = 0.1 * np.sin(2 * np.pi * 440.0 * np.arange(n1 - n0) / RATE)
    n2 = int(round(140 * RATE / FPS))
    back[n1:n2] = 0.1 * np.sin(2 * np.pi * 660.0 * np.arange(n2 - n1) / RATE)
    gaps = folder_of(root, "gaps", back, np.zeros_like(back), FRAMES)
    ran = {d.name: (run("phrases", str(d)), run("beats", str(d), "--shots", str(shots))) for d in (voiced, silent, steady, drift, accent, faint, slow, late, cancel, gaps)}

    def phrases_come_back():
        assert ran["voiced"][0].returncode == 0, ran["voiced"][0].stderr
        rec = json.loads((voiced / "phrases.record.json").read_text())
        got = [(p["first_frame"], p["last_frame"], p["starts"], p["ends"], p["run"]) for p in rec["phrases"]]
        assert got == [(20, 59, 20.0, 59.5, 1), (64, 139, 64.0, 140.0, 1), (180, 209, 180.0, 210.0, 2)], got
        gaps = [(g["first_frame"], g["frames"], g["a_break"]) for g in rec["gaps"]]
        assert gaps == [(59.5, 4.5, True), (100.0, 1.0, False)], gaps
        table = rows(voiced / "phrases_per_frame.csv")
        assert [r["phrase"] for r in table[58:65]] == ["1", "1", "", "", "", "", "2"] and table[100]["phrase"] == "2" and table[150]["phrase"] == "", [r["phrase"] for r in table[58:65]]
        assert float(table[59]["vocals_dbfs_step_2_of_4"]) > -30 and float(table[59]["vocals_dbfs_step_3_of_4"]) < -80, table[59]
        return "three phrases on their frames, a 4.5-frame breath a break, a one-frame stop a gap that breaks nothing"

    def no_voice_no_phrase():
        rec = json.loads((silent / "phrases.record.json").read_text())
        table = rows(silent / "phrases_per_frame.csv")
        assert rec["phrases"] == [] and rec["gaps"] == [] and len(table) == FRAMES and all(r["phrase"] == "" and r["voiced_run"] == "" for r in table), rec["phrases"]
        assert "0 phrase(s) in 0 voiced run(s)" in ran["silent"][0].stdout, ran["silent"][0].stdout

    def a_row_a_frame():
        for d, n in ((voiced, FRAMES), (steady, int(seconds * FPS))):
            for name in ("phrases_per_frame.csv", "beats_per_frame.csv"):
                table = rows(d / name)
                assert len(table) == n and [int(r["frame"]) for r in table] == list(range(n)), (d.name, name, len(table))
        beats = rows(steady / "beats_per_frame.csv")
        with_beat = [r for r in beats if r["beat"] != ""]
        assert 70 <= len(with_beat) <= 80 and all(r["beat_at_frame"] == "" and r["frames_since_the_beat_before"] == "" and r["in_the_steady_run"] == ""
                                                  for r in beats if r["beat"] == ""), len(with_beat)
        assert "count_in_fours_from_the_first_beat" not in beats[0] and all(r["half_way_at_frame"] == "" for r in beats), "a steady click has a count in fours or a half-way place"
        assert "0" not in {r["beat"] for r in beats}, "a frame with no beat reads beat 0"

    def nearest(d: Path, truth: np.ndarray) -> np.ndarray:
        got = np.array([float(r["beat_at_frame"]) for r in rows(d / "beats_per_frame.csv") if r["beat_at_frame"] != ""])
        return np.array([np.min(np.abs(got - t * FPS)) for t in truth])

    def clicks_come_back():
        assert ran["steady"][1].returncode == 0, ran["steady"][1].stderr
        rec = json.loads((steady / "beats.record.json").read_text())
        assert abs(rec["tempo_bpm"] - 120.0) < 1.5 and abs(rec["a_beat_is_frames"] - 12.0) < 0.2, rec["tempo_bpm"]
        off = nearest(steady, steady_t[2:-2])
        assert np.median(off) < 0.5 and off.max() < 1.0, (float(np.median(off)), float(off.max()))
        local = rec["the_fit"]["local_tempo_bpm"]
        assert local["most"] - local["least"] < 3.0 and len(local["stretches"]) >= 3, local
        assert rec["the_fit"]["onset_strength_on_the_beats_over_off_them"] > 5, rec["the_fit"]
        assert "counting_in_fours" not in rec and "no downbeat" in rec["no_count_in_fours"] and rec["half_way"] is None, rec.get("half_way")
        assert "no half-way places written" in ran["steady"][1].stdout, ran["steady"][1].stdout
        cuts = rec["cuts_against_the_grid"]
        assert [c["cut"] for c in cuts["cuts"]] == [60] and abs(cuts["cuts"][0]["frames_from_it"]) < 0.5 and cuts["cuts_placed_by_chance_would_be"]["middle"] == 3.0, cuts
        return f"tempo {rec['tempo_bpm']}, a beat within {off.max():.2f} frames of every click"

    def a_drifting_click():
        rec = json.loads((drift / "beats.record.json").read_text())
        local = rec["the_fit"]["local_tempo_bpm"]
        # each stretch holds a part of the climb and gives its middle: the first sits low in 96-150, the last high, well apart
        assert local["most"] - local["least"] > 15 and local["least"] < 112 and local["most"] > 125 and "the same pulse and its half or double" in local["said"], local
        off, held = nearest(drift, drift_t[(drift_t > 2) & (drift_t < seconds - 2)]), nearest(steady, steady_t[2:-2])
        assert np.mean(off) > 3 * np.mean(held) and off.max() > 1.0, (float(np.mean(off)), float(np.mean(held)))
        said = ran["drift"][1].stdout
        assert f"local tempo {local['least']} to {local['most']}" in said and "fits" not in said and "good" not in said.lower(), said
        return f"local tempo {local['least']} to {local['most']}; the beats lie {np.mean(off):.2f} frames from the clicks against {np.mean(held):.2f} on the steady track"

    def silence_has_no_pulse():
        rec = json.loads((silent / "beats.record.json").read_text())
        table = rows(silent / "beats_per_frame.csv")
        assert rec["tempo_bpm"] is None and rec["beats"] == 0 and all(r["beat"] == "" for r in table) and "no pulse" in ran["silent"][1].stdout, rec["tempo_bpm"]

    def the_record_says():
        import hashlib
        for d, name, inputs in ((voiced, "phrases", {"vocals.wav", "voice_spans.json"}), (steady, "beats", {"rest.wav", "voice_spans.json", "shots.json"})):
            rec = json.loads((d / f"{name}.record.json").read_text())
            assert {Path(x["path"]).name for x in rec["read_from"]} == inputs and all(len(x["sha256"]) == 64 and x["bytes"] > 0 for x in rec["read_from"]), rec["read_from"]
            assert rec["fingerprint"] == hashlib.sha256((d / rec["table"]["file"]).read_bytes()).hexdigest(), "the fingerprint is not the table's hash"
            assert rec["cannot_see"] == cs.CANNOT[name] and rec["settings"] and rec["libraries"]["scipy"] and rec["made_at"].endswith("Z"), rec["settings"]
        assert json.loads((voiced / "phrases.record.json").read_text())["settings"]["a_break_is_a_gap_of_steps_or_more"] == cs.BREAK

    def an_edge_is_printed():
        r = run("at", str(voiced), "--frames", "70-120", "--name", "a_load", "--shots", str(shots))
        said = r.stdout
        assert r.returncode == 0 and "a_load: frames 70-120 (51)" in said and "its shot in the record: 60-239" in said, said + r.stderr
        assert "in the voiced run 20-139, 50 frames after its start and 69 before its end" in said, said
        assert "in phrase 2 (64-139), 6 frames after its start and 69 before its end" in said and "to begin on the phrase's start: 6 frame(s) earlier, on frame 64, inside the same shot" in said, said
        assert "in phrase 2 (64-139), 56 frames after its start and 19 before its end" in said and "the phrase runs 19 frame(s) past the load's last frame, to frame 139, inside the same shot" in said, said
        assert "the one before at frame" in said and "the next at frame" in said and "Nothing here says where an edge should be" in said, said
        # what lies inside the load: nothing of a phrase between 70 and 120, and its beats with their place in the load
        assert "INSIDE THE LOAD" in said and "phrases: none starts or ends inside it" in said and "    beats: 5: " in said and "places half-way between beats: none is written" in said, said
        across = run("at", str(voiced), "--frames", "50-70", "--shots", str(shots)).stdout
        assert "phrases: phrase 1 ENDS on frame 59 [9]; phrase 2 STARTS on frame 64 [14]" in across, across
        cut = run("at", str(voiced), "--frames", "62-70", "--shots", str(shots)).stdout
        assert "in no phrase; the one before ended 3 frames ago; the next starts in 2 frames" in cut, cut
        early = run("at", str(voiced), "--frames", "60-70", "--shots", str(shots)).stdout
        assert "BEFORE THE SHOT'S FIRST FRAME 60" in early and "a load across a cut" in early, early
        bare = root / "bare"
        bare.mkdir()
        (bare / "voice_spans.json").write_text((voiced / "voice_spans.json").read_text())
        absent = run("at", str(bare), "--frames", "70-120").stdout
        assert "ABSENT, and said so: phrases" in absent and "ABSENT, and said so: beats" in absent and "phrase:" not in absent and "beat:" not in absent, absent

    def the_offset():
        clip = root / "late.mov"
        made = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=24:duration=2", "-itsoffset", "0.5", "-f", "lavfi", "-i",
                               "sine=frequency=440:duration=1.5", "-c:v", "mpeg4", "-c:a", "pcm_s16le", str(clip)], capture_output=True, text=True)
        assert made.returncode == 0, made.stderr
        r = run("offset", "--out", str(root / "off"), str(clip))
        assert r.returncode == 0, r.stderr
        row = json.loads((root / "off" / "sound_offset.record.json").read_text())["copies"][0]
        assert abs(row["sound_starts_after_picture_seconds"] - 0.5) < 0.02 and abs(row["sound_starts_after_picture_frames"] - 12.0) < 0.5 and row["has_sound"], row
        assert "+0.5" in r.stdout and "positive is the sound later" in r.stdout, r.stdout
        rec = json.loads((root / "off" / "sound_offset.record.json").read_text())
        assert "rows_are" not in rec and "table" not in rec and "neither confirmed nor refuted" in rec["cannot_see"], sorted(rec)
        # a hundredth of a second is a quarter of a frame at 24 a second, and is read as that; a copy with no sound gives no figure
        near, mute = root / "near.mov", root / "mute.mov"
        for clip, more in ((near, ["-itsoffset", "0.010", "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5", "-c:a", "pcm_s16le"]), (mute, ["-an"])):
            made = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=24:duration=2", *more, "-c:v", "mpeg4", str(clip)], capture_output=True, text=True)
            assert made.returncode == 0, made.stderr
        r2 = run("offset", "--out", str(root / "off2"), str(near), str(mute))
        assert r2.returncode == 0, r2.stderr
        two = json.loads((root / "off2" / "sound_offset.record.json").read_text())["copies"]
        assert abs(two[0]["sound_starts_after_picture_seconds"] - 0.010) < 0.002 and abs(two[0]["sound_starts_after_picture_frames"] - 0.24) < 0.05, two[0]
        assert two[1]["has_sound"] is False and "sound_starts_after_picture_seconds" not in two[1] and "mute.mov: no sound stream" in r2.stdout, two[1]
        return f"{row['sound_starts_after_picture_seconds']:+.3f} s, {row['sound_starts_after_picture_frames']:+.2f} frames; {two[0]['sound_starts_after_picture_frames']:+.2f} frames for a hundredth of a second"

    def the_level_is_both_channels():
        rec = json.loads((cancel / "phrases.record.json").read_text())
        got = [(p["first_frame"], p["last_frame"]) for p in rec["phrases"]]
        assert got == [(20, 39), (46, 59)], got
        table = rows(cancel / "phrases_per_frame.csv")
        assert float(table[30]["vocals_dbfs_step_1_of_4"]) > -30 and float(table[42]["vocals_dbfs_step_1_of_4"]) < -80, (table[30], table[42])
        mean = vs.frame_levels((left - left) / 2, RATE, FPS)                    # what the two channels' mean reads: nothing
        assert mean[30] < -100 and "the power of both channels" in rec["settings"]["the_level_is"] and "MEAN" in rec["settings"]["the_level_is"] \
            and rec["settings"]["channels"] == 2, rec["settings"]
        return f"the phrases {got} are found where the channels' mean reads {mean[30]:.0f} dBFS"

    def two_frames_do_not_break():
        rec = json.loads((gaps / "phrases.record.json").read_text())
        got = [(p["first_frame"], p["last_frame"]) for p in rec["phrases"]]
        found = [(g["first_frame"], g["frames"], g["a_break"]) for g in rec["gaps"]]
        assert got == [(20, 59), (63, 79), (100, 139)], got
        assert found == [(40.0, 2.0, False), (60.0, 3.0, True)], found
        assert cs.BREAK == 3 * cs.STEPS
        return "a two-frame gap is listed and breaks nothing, a three-frame gap is a break, two tones back to back are one phrase 100-139"

    def one_clock():
        table = rows(steady / "beats_per_frame.csv")
        onset = [float(r["onset_strength"]) if r["onset_strength"] != "" else 0.0 for r in table]
        beats = [(n, float(r["beat_at_frame"])) for n, r in enumerate(table) if r["beat"] != ""]
        off = [n for n, _ in beats[2:-2] if onset[n] < max(onset[max(n - 2, 0):n + 3])]
        assert not off, f"beat rows that are not the strongest onset round them: {off[:8]}"
        assert all(int(place) == n for n, place in beats), [b for b in beats if int(b[1]) != b[0]][:4]
        truth = {int(t * FPS) for t in steady_t}
        assert sum(n in truth for n, _ in beats) >= len(beats) - 4, "the beats' rows are not the clicks' frames"
        assert cs.shown(4769.997) == 4769.99 and cs.shown(12.0) == 12.0 and cs.shown(0.999999) == 0.99
        return f"{len(beats)} beat rows, each the strongest onset round it and on its click's own frame"

    def an_accent():
        # the accent at a third of the strength: every click is a beat, and the record says every other one may be the pulse
        both = json.loads((accent / "beats.record.json").read_text())
        e = both["every_other_beat"]
        assert abs(both["tempo_bpm"] - 120.0) < 1.5 and both["half_way"] is None and e and any(abs(c["bpm"] - 60.0) < 3.5 for c in e["the_candidates_near_half_the_tempo"]), (both["tempo_bpm"], e)
        a, b = e["onset_strength_on_the_beats_counted_from_the_first"], e["onset_strength_on_the_beats_counted_from_the_second"]
        assert max(a, b) > 1.15 * min(a, b) and "every other beat may be the pulse" in ran["accent"][1].stdout and "cannot say" in e["said"], (a, b)
        # the click between so faint that the tempo is the accents': the place half-way is written, with the faint click on it
        rec = json.loads((faint / "beats.record.json").read_text())
        table = rows(faint / "beats_per_frame.csv")
        assert abs(rec["tempo_bpm"] - 60.0) < 1.0, rec["tempo_bpm"]
        h = rec["half_way"]
        assert h and h["places"] >= 30 and any(abs(c["bpm"] - 120.0) < 6.5 for c in h["the_candidates_near_twice_the_tempo"]) and "not a beat" in h["said"], h
        strong, weak = {int(t * FPS) for t in steady_t[0::2]}, {int(t * FPS) for t in steady_t[1::2]}
        beat_rows = [n for n, r in enumerate(table) if r["beat"] != ""][2:-2]
        half_rows = [n for n, r in enumerate(table) if r["half_way_at_frame"] != ""][2:-2]
        assert sum(n in strong for n in beat_rows) >= len(beat_rows) - 2 and sum(n in weak for n in half_rows) >= len(half_rows) - 2, (beat_rows[:6], half_rows[:6])
        assert 0.02 < h["onset_strength_half_way_over_on_the_beats"] < 0.9 and "A place half-way is not called a beat" in ran["faint"][1].stdout, h
        said = run("at", str(faint), "--frames", "100-160").stdout
        assert "places half-way between beats (not beats; a hit of another band can sit there): " in said and "beats: " in said, said
        slow_rec = json.loads((slow / "beats.record.json").read_text())
        assert abs(slow_rec["tempo_bpm"] - 60.0) < 1.0 and slow_rec["half_way"] is None and all(r["half_way_at_frame"] == "" for r in rows(slow / "beats_per_frame.csv")), slow_rec["half_way"]
        return (f"an accent at a third: tempo {both['tempo_bpm']}, every other beat named ({a} against {b}); a faint click between: tempo {rec['tempo_bpm']}, "
                f"{h['places']} half-way places on the faint clicks; none written for clicks with nothing between them")

    def beats_before_the_pulse():
        rec = json.loads((late / "beats.record.json").read_text())
        table = rows(late / "beats_per_frame.csv")
        r = rec["the_steady_run"]
        assert r and abs(r["from_frame"] - (12.0 * FPS + 0.5)) < 13 and r["beats_before_it"] >= 1 and r["beats"] >= 45, r
        before = [x for x in table if x["beat"] != "" and float(x["beat_at_frame"]) < 11.0 * FPS]
        assert before and all(x["in_the_steady_run"] == "" and x["beat_strength_over_the_middle_beat"] != "" for x in before), before[:3]
        inside = [x for x in table if x["beat"] != "" and 14.0 * FPS < float(x["beat_at_frame"]) < 36.0 * FPS]
        assert all(x["in_the_steady_run"] == "yes" and abs(float(x["frames_since_the_beat_before"]) - 12.0) < 1.0 for x in inside), [x for x in inside if x["in_the_steady_run"] != "yes"][:3]
        assert "laid on no steady pulse" in ran["late"][1].stdout and json.loads((steady / "beats.record.json").read_text())["the_steady_run"]["beats_before_it"] <= 1
        return f"the steady run starts at frame {r['from_frame']}; {r['beats_before_it']} beat(s) before it are in the table, outside the run"

    case("phrases come back to the frame", phrases_come_back)
    case("a clip with no voice has no phrase", no_voice_no_phrase)
    case("a row a frame and no more", a_row_a_frame)
    case("clicks come back as their tempo and grid", clicks_come_back)
    case("a drifting click is reported", a_drifting_click)
    case("silence has no pulse", silence_has_no_pulse)
    case("the record says what it read", the_record_says)
    case("an edge is printed, not ruled", an_edge_is_printed)
    case("the offset is the container's", the_offset)
    case("the level is both channels' power", the_level_is_both_channels)
    case("two frames do not break a phrase, three do", two_frames_do_not_break)
    case("one clock in one table", one_clock)
    case("an accent every second click", an_accent)
    case("beats before the pulse are said", beats_before_the_pulse)

if __name__ == "__main__":
    sys.exit(finish())
