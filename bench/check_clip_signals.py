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
  the_offset_is_the_containers          a made copy whose sound starts a known time after its picture gives that time, with its sign

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


def clicks(times: np.ndarray, seconds: float) -> np.ndarray:
    """A short burst of noise at each time, on silence."""
    rng = np.random.default_rng(7)
    out = np.zeros(int(seconds * RATE), np.float32)
    burst = (rng.standard_normal(int(0.02 * RATE)) * np.exp(-np.arange(int(0.02 * RATE)) / (0.004 * RATE))).astype(np.float32) * 0.5
    for t in times:
        k = int(round(t * RATE))
        if k + len(burst) < len(out):
            out[k:k + len(burst)] += burst
    return out


def folder_of(root: Path, name: str, vocals: np.ndarray, rest: np.ndarray, frames: int) -> Path:
    """An audio folder as voice_spans leaves one: the two stems and the voice table's record, made by its own functions."""
    d = root / name
    d.mkdir()
    for stem, x in (("vocals", vocals), ("rest", rest)):
        vs.write_wav(d / f"{stem}.wav", np.stack([x, x]))
    level = vs.frame_levels(vocals, RATE, FPS)[:frames]
    (d / "voice_spans.json").write_text(json.dumps({"fps": FPS, "frames": frames, "threshold_dbfs": vs.THRESHOLD_DB,
                                                    "voiced_spans_inclusive": vs.voiced_spans(level)}))
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
    steady_t = np.arange(0.5, 39.5, 0.5)                                         # 120 a minute
    drift_t = np.cumsum(60.0 / np.linspace(96.0, 150.0, 80)) + 0.5               # 96 climbing to 150
    seconds = 40.0
    voiced = folder_of(root, "voiced", sung, clicks(steady_t, FRAMES / FPS), FRAMES)
    silent = folder_of(root, "silent", np.zeros_like(sung), np.zeros_like(sung), FRAMES)
    steady = folder_of(root, "steady", np.zeros(int(seconds * RATE), np.float32), clicks(steady_t, seconds), int(seconds * FPS))
    drift = folder_of(root, "drift", np.zeros(int(seconds * RATE), np.float32), clicks(drift_t[drift_t < seconds - 0.5], seconds), int(seconds * FPS))
    shots = root / "shots.json"
    shots.write_text(json.dumps({"shots": [{"first": 0, "last": 59}, {"first": 60, "last": 239}]}))
    ran = {d.name: (run("phrases", str(d)), run("beats", str(d), "--shots", str(shots))) for d in (voiced, silent, steady, drift)}

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
        assert 70 <= len(with_beat) <= 80 and all(r["beat_at_frame"] == "" and r["count_in_fours_from_the_first_beat"] == "" for r in beats if r["beat"] == ""), len(with_beat)
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
        assert len(rec["counting_in_fours"]["the_four"]) == 4 and "no downbeat is claimed" in rec["counting_in_fours"]["said"]
        cuts = rec["cuts_against_the_grid"]
        assert [c["cut"] for c in cuts["cuts"]] == [60] and abs(cuts["cuts"][0]["frames_from_it"]) < 0.5 and cuts["cuts_placed_by_chance_would_be"]["middle"] == 3.0, cuts
        return f"tempo {rec['tempo_bpm']}, a beat within {off.max():.2f} frames of every click"

    def a_drifting_click():
        rec = json.loads((drift / "beats.record.json").read_text())
        local = rec["the_fit"]["local_tempo_bpm"]
        # each stretch holds a part of the climb and gives its middle: the first sits low in 96-150, the last high, well apart
        assert local["most"] - local["least"] > 15 and local["least"] < 112 and local["most"] > 125, local
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
        return f"{row['sound_starts_after_picture_seconds']:+.3f} s, {row['sound_starts_after_picture_frames']:+.2f} frames"

    case("phrases come back to the frame", phrases_come_back)
    case("a clip with no voice has no phrase", no_voice_no_phrase)
    case("a row a frame and no more", a_row_a_frame)
    case("clicks come back as their tempo and grid", clicks_come_back)
    case("a drifting click is reported", a_drifting_click)
    case("silence has no pulse", silence_has_no_pulse)
    case("the record says what it read", the_record_says)
    case("an edge is printed, not ruled", an_edge_is_printed)
    case("the offset is the container's", the_offset)

if __name__ == "__main__":
    sys.exit(finish())
