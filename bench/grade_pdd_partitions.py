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

## What it measures, and why per stream

Video and audio are decoded and compared SEPARATELY. The lane opened with
"4-step audio sounds wrong while the video looks acceptable", and a single
blended number cannot answer that. Audio is ~1% of the packed sequence, so a
joint metric would be dominated by video and say nothing about the complaint.

Everything is compared on decoded output, never on file bytes: SaveImage embeds
the prompt JSON, which already produced one false negative in this lane.
"""
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

SRV = "http://127.0.0.1:8188"
REPO = Path("/home/fbliss/ComfyUI/custom_nodes/ComfyUI-h3-explorations")
BASE = REPO / "workflows/h3_text_to_video_pdd_4step_api.json"
OUT = Path("/mnt/hub/ai/img/output")
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

Path(__file__).with_name("pdd_reference_trajectory_result.json").write_text(
    json.dumps({"length": LENGTH, "seed": SEED, "opt4_sigmas": OPT4,
                "reference": "steps=32, block width 1, the distilled trajectory",
                "arms": rows}, indent=1) + "\n")
print("\nwrote pdd_reference_trajectory_result.json")
