#!/usr/bin/env python3
"""Grade PDD block partitions against the trajectory they were distilled from.

## Why this is not a perceptual comparison

Every quality question in this lane has dead-ended at "a person must look",
because `CLAUDE.md` is right that a rendered clip cannot A/B a numerical change:
two arms differing in a knob are different samples, not better and worse
versions of one.

PDD is the exception, and the reason is specific to it. The 32 published heads
ARE a distillation of a 32-point trajectory, and `steps=32` runs block width 1
-- every head used on its own interval, no fusion at all. That is not "more
steps"; it is **the trajectory the coarser partitions are approximations of**.
So there is a ground truth here, the comparison is against it rather than
between arms, and it needs no human and no distribution.

Two further things make it exact rather than approximate:

  * the node's SIGMAS output emits the closed form, and
    `BasicScheduler(simple, 32)` sits 4.95e-3 away from it because
    `1000 % 32 != 0`. The reference is only reachable through the output added
    this session.
  * every arm is a subset of the same 32 knots, so each one's fused heads are
    exact for the span it steps. Nothing here is off-grid.

## The arms

  ref32   32 evaluations, width 1        the reference trajectory
  u4      uniform [8,8,8,8]              what the 4-step graphs ship
  u8      uniform [4]*8                  what the authors ship
  opt4    [28,2,1,1]                     the fusion-loss optimum at 4 evals
  u4b/u4c byte-identical repeats of u4    the NOISE FLOOR, and the reason any
                                          ordering below can be believed. Without
                                          them a difference cannot be told from
                                          run-to-run variation, which this
                                          pipeline was wrongly assumed to have.

## What it measures, and why per stream

Video and audio are decoded and compared SEPARATELY. The lane opened with
"4-step audio sounds wrong while the video looks acceptable", and a single
blended number cannot answer that. Audio is ~1% of the packed sequence, so a
joint metric would be dominated by video and say nothing about the complaint.

Everything is compared on decoded output, never on file bytes: SaveImage embeds
the prompt JSON, which already produced one false negative in this lane.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

SRV = "http://127.0.0.1:8188"
REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "workflows/h3_text_to_video_pdd_4step_api.json"
OUT = Path(os.environ.get("H3_OUTPUT_DIR", "")) if os.environ.get(
    "H3_OUTPUT_DIR") else Path(__file__).resolve().parents[2] / "output"
LENGTH, SEED = 39, 730451892

#: [28,2,1,1] on the shift-12 grid. Written out rather than recomputed so the
#: arm is legible in the payload and in the record.
OPT4 = "1.0, 0.631579, 0.444444, 0.27907, 0.0"

ARMS = {"ref32": 32, "u8": 8, "u4": 4, "u4b": 4, "u4c": 4, "opt4": None}


def post(path, payload):
    req = urllib.request.Request(f"{SRV}/{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def get(path):
    return json.loads(urllib.request.urlopen(f"{SRV}/{path}").read())


def build(arm):
    g = json.loads(BASE.read_text())
    g["27"]["inputs"]["length"] = LENGTH
    g["6"]["inputs"]["noise_seed"] = SEED
    g.pop("13", None)                       # muxer: re-encodes, and we compare tensors
    steps = ARMS[arm]
    if steps is None:                       # the non-uniform arm
        g["18"]["inputs"]["steps"] = 4      # heads still fuse per span from the schedule
        g["60"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": OPT4}}
        g["10"]["inputs"]["sigmas"] = ["60", 0]
    else:
        g["18"]["inputs"]["steps"] = steps
        g["10"]["inputs"]["sigmas"] = ["18", 1]
    g["98"] = {"class_type": "SaveImage",
               "inputs": {"images": ["11", 0], "filename_prefix": f"pddref/{arm}_v"}}
    g["99"] = {"class_type": "SaveAudio",
               "inputs": {"audio": ["12", 0], "filename_prefix": f"pddref/{arm}_a"}}
    return g


def run(arm):
    try:
        pid = post("prompt", {"prompt": build(arm), "client_id": f"pddref-{arm}"})["prompt_id"]
    except urllib.error.HTTPError as e:
        return {"arm": arm, "error": e.read().decode()[:600]}
    print(f"  {arm}: {pid}", flush=True)
    while pid not in (h := get(f"history/{pid}")):
        time.sleep(3)
    rec = h[pid]
    if rec.get("status", {}).get("status_str") != "success":
        msgs = [m for m in rec["status"].get("messages", []) if m[0] == "execution_error"]
        return {"arm": arm, "error": json.dumps(msgs)[:600]}
    return {"arm": arm,
            "frames": [OUT / i.get("subfolder", "") / i["filename"]
                       for i in rec["outputs"]["98"]["images"]],
            "audio": [OUT / a.get("subfolder", "") / a["filename"]
                      for a in rec["outputs"]["99"].get("audio", [])]}


def video_array(paths):
    return np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
                     for p in sorted(paths)])


def audio_array(path):
    """Decode FLAC to float PCM via ffmpeg -- no extra dependency, lossless in."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", str(path), "-f", "f32le", "-acodec",
         "pcm_f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


arms = sys.argv[1:] or list(ARMS)
print(f"length={LENGTH} seed={SEED} arms={arms}")
res = {}
for a in arms:
    print(f"[{a}]", flush=True)
    res[a] = run(a)
    if res[a].get("error"):
        print(f"  ERROR {res[a]['error'][:300]}", flush=True)

if "ref32" not in res or res["ref32"].get("error"):
    print("\nno reference; nothing to grade against")
    raise SystemExit(1)

rv = video_array(res["ref32"]["frames"])
ra = audio_array(res["ref32"]["audio"][0]) if res["ref32"]["audio"] else None
print(f"\nreference: {rv.shape[0]} frames {rv.shape[1:3]}, "
      f"{0 if ra is None else ra.size} audio samples")
print(f"\n{'arm':<7} {'VIDEO rel L2':>14} {'VIDEO max|d|':>13} {'AUDIO rel L2':>14}")
print("-" * 52)
rows = {}
for a in arms:
    if a == "ref32" or res[a].get("error"):
        continue
    v = video_array(res[a]["frames"])
    vrel = float(np.linalg.norm(v - rv) / np.linalg.norm(rv))
    vmax = float(np.abs(v - rv).max())
    arel = None
    if ra is not None and res[a]["audio"]:
        x = audio_array(res[a]["audio"][0])
        n = min(x.size, ra.size)
        arel = float(np.linalg.norm(x[:n] - ra[:n]) / (np.linalg.norm(ra[:n]) + 1e-12))
    rows[a] = {"video_rel_l2": vrel, "video_max_abs": vmax, "audio_rel_l2": arel}
    print(f"{a:<7} {vrel:>14.5f} {vmax:>13.5f} "
          f"{'n/a' if arel is None else f'{arel:>14.5f}'}")

# Pairwise distances, including the same-arm repeats that ARE the noise floor.
# The committed record reported these while this script computed only
# arm-vs-reference, so three of its numbers could not be reproduced by the file
# it named. They are computed here now, and written to the path the record uses.
pairs = {}
have = [a for a in arms if not res[a].get("error")]
for i, a in enumerate(have):
    for b in have[i + 1:]:
        va, vb = video_array(res[a]["frames"]), video_array(res[b]["frames"])
        entry = {"video_rel_l2": float(np.linalg.norm(va - vb) / np.linalg.norm(vb))}
        if res[a]["audio"] and res[b]["audio"]:
            xa, xb = audio_array(res[a]["audio"][0]), audio_array(res[b]["audio"][0])
            n = min(xa.size, xb.size)
            entry["audio_rel_l2"] = float(
                np.linalg.norm(xa[:n] - xb[:n]) / (np.linalg.norm(xb[:n]) + 1e-12))
        pairs[f"{a}_vs_{b}"] = entry

out = REPO / "bench/results/2026-08-28_pdd_partition_fidelity.json"
out.write_text(json.dumps(
    {"date": "2026-08-28", "script": "bench/grade_pdd_partitions.py",
     "length": LENGTH, "seed": SEED, "opt4_sigmas": OPT4,
     "reference": "steps=32, block width 1, the distilled trajectory",
     "against_reference": rows, "pairwise": pairs,
     # Emitted by the script rather than typed into the record, because a
     # hand-added caveat is the same defect as a hand-added number: it drifts
     # from what the run actually supports and cannot be regenerated.
     "do_not_rely_on": [
       "These magnitudes are NOT calibrated to perception. Nothing here says "
       "what rel L2 0.5 looks like or 1.0 sounds like. Only the ORDERING and "
       "the exactly-zero same-arm floor are load-bearing.",
       "Raw-waveform L2 is phase-sensitive and a poor audio metric in general. "
       "It is used here only because every arm shares a seed and the pipeline "
       "is bit-deterministic, so a phase offset is not an innocent "
       "explanation -- but an uncorrelated waveform still does not mean it "
       "sounds bad.",
       "The reference is the distilled 32-point trajectory, NOT a claim that "
       "it is the best-sounding render. Faithfulness to it is a different "
       "question from quality.",
       "1344x768 is a TRAINED canvas, so the canvas is not the cheap axis "
       "here; only length=39 is. An earlier hand-written version of this "
       "record called the canvas cheap, contradicting CLAUDE.md.",
       "One seed, one length, one canvas, one partition family."]},
    indent=1) + "\n")
print(f"\nwrote {out.relative_to(REPO)}")
