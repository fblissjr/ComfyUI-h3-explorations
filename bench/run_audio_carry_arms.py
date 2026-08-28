#!/usr/bin/env python3
"""Render the audio-carry ablation and grade it on ENERGY, not on rel L2.

## The question

`docs/research/pdd/audio_under_pdd.md` blames PDD's audio penalty on ComfyUI
freezing the audio change-of-variable's coefficients at each block's STARTING
sigma. `bench/analyze_pdd_stream_energy.py` established two things that shape
this run:

  * the penalty is an ENERGY collapse -- 7.1 dB down at `u4`, 11.2 dB at `opt4`
    -- not a decorrelation, so audio energy is the observable here and audio
    rel L2 is not. That metric saturates at sqrt(2) for a ONE-frame shift, so
    it has no gradation left to report.
  * no experiment that varies the PARTITION can attribute any of it to the
    transform, because every partition-derived statistic ranks the arms
    identically to plain coarseness.

So this varies the transform instead, at a fixed partition, via
`MiniMaxH3AudioCarryProbe`. Heads, schedule, adaln grid, seed, prompt, sampler
and every weight are held; only where the transform samples sigma moves.

## Why `block_start` is rendered rather than assumed

It is arithmetically the identity to ~1e-6 (`bench/check_audio_carry_inversion.py`
asserts that). It is NOT a reproduction of `off` as a render: 1e-6 diverges a
sampling trajectory completely, which `CLAUDE.md` measured at frame 0 under a
deterministic sampler. That is exactly what makes it worth the GPU -- it is a
second sample at the SAME knob, so `|block_start - off|` is the noise floor
that `|block_mean - off|` has to clear before it means anything.

**Without it this script would be one clip per arm on a numerical knob, which
`CLAUDE.md` says answers nothing.** It still is not a distribution; it is one
pair. A result inside the floor is "not measurable this way", never "no effect".

## The built-in inertness detector

The probe patches `diffusion_model.forward`, the same key `MiniMaxH3PDDLoRA`
uses. It chains, because `ModelPatcher.get_model_object` returns the already
patched forward -- but only if the probe is inserted AFTER the PDD node, which
`build` does by construction. If it were ever inserted before, PDD would clobber
it and every arm would be the stock path. The tell is free: `block_mean` and
`block_start` would come out identical, and identical arms are reported as a
FAILED RUN rather than as a null result.
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "workflows"))
import h3_config  # noqa: E402

SRV = "http://127.0.0.1:8188"
BASE = REPO / "workflows/h3_text_to_video_pdd_4step_api.json"
OUT = h3_config.output_dir()
SUB = "carryprobe"

#: Matched to `bench/grade_pdd_partitions.py`'s length, canvas and seed. That
#: makes the ARMS reproducible, not cross-comparable: see CROSS_RUN_NOTE below,
#: where the graph's prompt turned out to have moved underneath both.
LENGTH, SEED = 362, 730451892
CANVAS = "1152x768  3/2  864 tok/frame  0.73x"

#: `opt4`'s sigma vector, from `grade_pdd_partitions.py::MANUAL`. Its widest
#: block is 28 grid points, where the frozen coefficient is furthest from the
#: block's average -- the largest effect the ablation can show. It is also the
#: off-distribution partition, which does NOT confound this comparison: both
#: modes run the same partition, so it cancels.
OPT4 = "1.0, 0.631579, 0.444444, 0.27907, 0.0"

#: arm -> (partition, probe mode). Every partition renders all three modes,
#: because each one needs its OWN baseline and its own noise floor -- the two
#: partitions differ in block width, which is exactly the thing that might make
#: them differ in numerical sensitivity too.
ARMS = {
    "u8_off":     ("u8", "off"),
    "u8_start":   ("u8", "block_start"),
    "u8_mean":    ("u8", "block_mean"),
    "u4_off":     ("u4", "off"),
    "u4_start":   ("u4", "block_start"),
    "u4_mean":    ("u4", "block_mean"),
    "opt4_off":   ("opt4", "off"),
    "opt4_start": ("opt4", "block_start"),
    "opt4_mean":  ("opt4", "block_mean"),
}

#: Widest block per partition, which is the dose the ablation is supposed to
#: respond to. `u8` is here because the first run had only TWO points and one of
#: them (`opt4`) is off-distribution -- so the dose-response rested on an arm
#: nobody should ship. `u8` and `u4` are both legal under the trained envelope,
#: which lets the relationship stand on legal partitions alone and makes `opt4`
#: a third point rather than half the evidence.
WIDEST_BLOCK = {"u8": 4, "u4": 8, "opt4": 28}

#: **Everything below is graded WITHIN this run, against each partition's own
#: `off` arm, and never against `bench/results/2026-08-28_pdd_stream_energy.json`.**
#:
#: The reason is measured, not precautionary. The first `u4_off` render came
#: out 1.7 dB above the `u4` recorded in that file, on what should have been an
#: identical graph -- because `workflows/h3_text_to_video_pdd_4step_api.json`
#: had its PROMPT rewritten at 15:25 on 2026-08-28 (`b62e95d`), after those
#: arms rendered at 13:53-14:29. A different prompt is different audio, so the
#: two sets are not comparable and the `off` arm is what caught it.
#:
#: This costs the ablation nothing: a ratio between two arms cancels whatever
#: reference it is taken against. It costs the ABSOLUTE deficit, which stays
#: owned by the earlier record at the prompt that record was rendered with.
CROSS_RUN_NOTE = (
    "graded within this run only; the base graph's prompt changed in b62e95d "
    "after bench/results/2026-08-28_pdd_stream_energy.json was measured")


def post(path, payload):
    req = urllib.request.Request(f"{SRV}/{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def get(path):
    return json.loads(urllib.request.urlopen(f"{SRV}/{path}").read())


def build(arm, seed=SEED, tag=""):
    partition, mode = ARMS[arm]
    g = json.loads(BASE.read_text())
    g["27"]["inputs"]["length"] = LENGTH
    g["27"]["inputs"]["shape.wide_resolution"] = CANVAS
    g["6"]["inputs"]["noise_seed"] = seed
    g.pop("13", None)                        # muxer: re-encodes, we want tensors
    if partition == "opt4":
        g["18"]["inputs"]["steps"] = 0       # see grade_pdd_partitions: 0 feeds
        g["60"] = {"class_type": "ManualSigmas",   # only the SIGMAS output, which
                   "inputs": {"sigmas": OPT4}}     # ManualSigmas replaces here
        g["10"]["inputs"]["sigmas"] = ["60", 0]
    else:                                    # uniform: the node emits the schedule
        g["18"]["inputs"]["steps"] = int(partition[1:])
        g["10"]["inputs"]["sigmas"] = ["18", 1]
    # AFTER the PDD node (18) and before the shift node (19), so the probe's
    # wrapper chains onto PDD's rather than being overwritten by it.
    g["62"] = {"class_type": "MiniMaxH3AudioCarryProbe",
               "inputs": {"model": ["18", 0], "mode": mode}}
    g["19"]["inputs"]["model"] = ["62", 0]
    g["98"] = {"class_type": "SaveImage",
               "inputs": {"images": ["11", 0], "filename_prefix": f"{SUB}/{arm}{tag}_v"}}
    g["99"] = {"class_type": "SaveAudio",
               "inputs": {"audio": ["12", 0], "filename_prefix": f"{SUB}/{arm}{tag}_a"}}
    return g


def run(arm, seed=SEED, tag=""):
    try:
        pid = post("prompt", {"prompt": build(arm, seed, tag),
                              "client_id": f"carry-{arm}{tag}"})["prompt_id"]
    except urllib.error.HTTPError as e:
        return {"arm": arm, "error": e.read().decode()[:800]}
    print(f"  {arm}{tag}: {pid}", flush=True)
    while pid not in (h := get(f"history/{pid}")):
        time.sleep(3)
    rec = h[pid]
    if rec.get("status", {}).get("status_str") != "success":
        msgs = [m for m in rec["status"].get("messages", []) if m[0] == "execution_error"]
        return {"arm": arm, "error": json.dumps(msgs)[:800]}
    return {"arm": arm + tag,
            "audio": [OUT / a.get("subfolder", "") / a["filename"]
                      for a in rec["outputs"]["99"].get("audio", [])]}


def audio(path):
    raw = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", str(path), "-f", "f32le", "-acodec",
         "pcm_f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def db(x, ref):
    return 20.0 * np.log10(x.std() / ref.std())


def main() -> int:
    want = sys.argv[1:] or list(ARMS)
    bad = [a for a in want if a not in ARMS]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; known: {list(ARMS)}")
    for part in WIDEST_BLOCK:
        if any(ARMS[a][0] == part for a in want) and f"{part}_off" not in want:
            raise SystemExit(
                f"{part} arms were requested without {part}_off. Every number "
                f"here is relative to that partition's own `off` render, "
                f"because the base graph's prompt changed after the earlier "
                f"records were measured and nothing cross-run is comparable.")
    print(f"length={LENGTH} seed={SEED} canvas={CANVAS}")
    print(f"arms: {want}")
    print(f"NOTE: {CROSS_RUN_NOTE}\n")

    res = {}
    for a in want:
        print(f"[{a}]", flush=True)
        res[a] = run(a)
        if res[a].get("error"):
            print(f"  ERROR {res[a]['error'][:400]}", flush=True)

    wav = {}
    for a in want:
        if not res[a].get("error") and res[a].get("audio"):
            wav[a] = audio(res[a]["audio"][0])

    rows = {}
    print(f"\n{'arm':<12}{'audio rms':>12}{'dB vs own off':>16}"
          f"{'gain vs off':>13}{'corr vs off':>13}")
    print("-" * 67)
    for a in want:
        if a not in wav:
            continue
        base_arm = f"{ARMS[a][0]}_off"
        x = wav[a]
        b = wav.get(base_arm)
        row = {"rms": float(x.std())}
        if b is not None:
            n = min(x.size, b.size)
            xa, ba = x[:n], b[:n]
            row["rms_db_vs_own_off"] = float(db(xa, ba))
            row["gain_vs_own_off"] = float(xa @ ba / (ba @ ba))
            row["corr_vs_own_off"] = float(
                xa @ ba / (np.linalg.norm(xa) * np.linalg.norm(ba)))
        rows[a] = row
        print(f"{a:<12}{row['rms']:>12.6f}"
              f"{row.get('rms_db_vs_own_off', float('nan')):>16.2f}"
              f"{row.get('gain_vs_own_off', float('nan')):>13.4f}"
              f"{row.get('corr_vs_own_off', float('nan')):>13.4f}")

    verdict = []
    for part in WIDEST_BLOCK:
        st, mn = f"{part}_start", f"{part}_mean"
        if st not in rows or mn not in rows:
            continue
        floor = abs(rows[st]["rms_db_vs_own_off"])
        eff = rows[mn]["rms_db_vs_own_off"] - rows[st]["rms_db_vs_own_off"]
        verdict.append(
            f"{part} (widest block {WIDEST_BLOCK[part]}): NOISE FLOOR "
            f"(off vs block_start, same knob, different sample) {floor:.2f} dB")
        if abs(eff) < 1e-9:
            verdict.append(
                f"{part}: block_mean and block_start are IDENTICAL. That is "
                f"NOT a null result -- the probe was inert (clobbered patch, "
                f"or no sample_sigmas reached it). FAILED RUN.")
        else:
            call = ("clears the floor" if abs(eff) > 2 * floor else
                    "INSIDE the floor, so not measurable this way")
            verdict.append(
                f"{part} (widest block {WIDEST_BLOCK[part]}): ablation moves "
                f"audio energy {eff:+.2f} dB against block_start ({call})")
    print()
    for v in verdict:
        print(f"  {v}")

    out = REPO / "bench/results/2026-08-28_audio_carry_ablation.json"
    out.write_text(json.dumps({
        "date": "2026-08-28",
        "script": "bench/run_audio_carry_arms.py",
        "length": LENGTH, "canvas": CANVAS, "seed": SEED,
        "grading": CROSS_RUN_NOTE,
        "arms": rows, "widest_block": WIDEST_BLOCK, "verdict": verdict,
        "errors": {a: res[a]["error"] for a in want if res[a].get("error")},
        "do_not_rely_on": [
            "ONE pair per partition. block_start is a noise-floor estimate "
            "from a single second sample, not a distribution. An effect "
            "inside it is 'not measurable this way', never 'no effect'.",
            "NOT comparable to bench/results/2026-08-28_pdd_stream_energy.json "
            "or to the pddref renders: the base graph's prompt was rewritten "
            "in b62e95d after those were measured. The first u4_off render "
            "landed 1.7 dB off the recorded u4, which is how this was found. "
            "Every figure here is within-run.",
            "The carry re-evaluation is PARTIAL by construction: the network "
            "already ran on x_a * carry(sigma_start) and no post-hoc "
            "correction reaches what it saw. A null result therefore does not "
            "clear the transform, it clears the velocity coefficient B.",
            "Audio energy is the observable because the collapse is an energy "
            "collapse. It is not a quality measure and no clip was judged.",
            "opt4 is the off-distribution partition. That cancels between its "
            "two modes but its magnitudes are not a statement about a "
            "partition anyone should ship.",
            "One seed, one length, one canvas, one prompt -- the market scene "
            "that docs/prompt_audit.md verdicts `rewrite`.",
        ],
    }, indent=1) + "\n")
    print(f"\nwrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
