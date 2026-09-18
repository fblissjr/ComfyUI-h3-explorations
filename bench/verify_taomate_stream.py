#!/usr/bin/env python3
"""Run the TaoMate stream sampler on the server, for its checks and its throwaway render.

`docs/h3_taomate.md` section 7.3. Two modes, each a rewired TaoMate probe graph.

**`verify_whole_clip`** (step 2). From `workflows/h3_probe_taomate_3step_api.json`:
- sage, Sol and the chain assert are removed, because the sampler refuses them
  and the stock reference must run the same plain attention;
- `KSamplerSelect` becomes `MiniMaxH3TaoMateStreamSampler` in that mode;
- the canvas and length are set explicitly, small by default;
- decoding and the video writer become `PreviewAny` on the sampler's latent,
  so nothing is decoded and nothing is written to the output share.

The node runs core's own euler sampler first, then its hooked loop over the
same inputs, and logs the deviation between the two latents. This script
reads that line and grades it.

**`stream`** (steps 4 and 5). From
`workflows/h3_probe_taomate_3step_audio_freeze_api.json`, with the same
attention chain removed and the sampler in stream mode. The track, prompt,
seed and output prefix come from the flags. It decodes and writes a clip, and
records the per-chunk log lines: wall time, peak allocated memory, cache
tokens.

A cheaper canvas proves the harness and the hook, not a render (CLAUDE.md,
"1344x768 is a trained canvas").

    <comfy venv python> bench/verify_taomate_stream.py --mode verify_whole_clip \\
        [--width 864 --height 480 --length 124] [--record bench/results/<date>_<name>.json]
    <comfy venv python> bench/verify_taomate_stream.py --mode stream --prompt-id t2va_studio_dancer \\
        --audio "Drum Machine Pulse.mp3" --seed 1101 --prefix Video/h3_taomate_stream_throwaway ...

Exit codes: 0 passed (verify: within `MATCH_REL_RMS`; stream: the run
succeeded and logged every chunk), 1 verify out of bound, 2 the run failed or
its log lines were not found.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workflows"))

import taomate_streaming as tm  # noqa: E402

PROBES = {
    "verify_whole_clip": REPO / "workflows" / "h3_probe_taomate_3step_api.json",
    "control_text_only": REPO / "workflows" / "h3_probe_taomate_3step_api.json",
    "stream": REPO / "workflows" / "h3_probe_taomate_3step_audio_freeze_api.json",
    # The whole-clip control for the stream render: the stock sampler on the
    # same freeze probe with every attention patch stripped, so TaoMate runs
    # over the whole clip on dense attention. Separates the attention stack
    # from the whole-clip regime.
    "plain_render": REPO / "workflows" / "h3_probe_taomate_3step_audio_freeze_api.json",
}
#: Model-path nodes the check keeps: loading, the LoRA and the shift. Every
#: other node between the guider and these is an attention patch or an assert
#: on one, and is stripped.
MODEL_PATH_KEEP = ("UNETLoader", "LoraLoaderModelOnly", "MiniMaxH3SigmaShift")
#: Reasoned: the hooked loop computes the same attention with torch's SDPA on
#: the same inputs, so any difference is kernel dispatch and float rounding.
#: A wiring defect (a wrong position, timestep or row order) moves the latent
#: by orders of magnitude more than this.
MATCH_REL_RMS = 1e-3
VERIFY_TAG = "[taomate] verify_whole_clip "
CHUNK_TAG = "[taomate] request "
_STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)")


def _post(host, path, payload):
    req = urllib.request.Request(f"http://{host}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _get(host, path):
    with urllib.request.urlopen(f"http://{host}{path}", timeout=60) as resp:
        return json.loads(resp.read())


def build_graph(args) -> dict:
    doc = json.loads(PROBES[args.mode].read_text(encoding="utf-8"))
    by_class = {}
    for nid, node in doc.items():
        by_class.setdefault(node["class_type"], []).append(nid)
    # Strip whatever attention chain sits on the model path, by walking back
    # from the guider until a node that only loads or shifts the model. By
    # structure rather than by class name: the shipped chain changed on
    # 2026-09-15 from sage + Sol to core's attention backend + Sol, and the
    # sampler refuses both.
    guider = doc[by_class["BasicGuider"][0]]
    source = guider["inputs"]["model"]
    stripped = []
    while doc[source[0]]["class_type"] not in MODEL_PATH_KEEP:
        stripped.append(source[0])
        source = doc[source[0]]["inputs"]["model"]
    guider["inputs"]["model"] = source
    for nid in stripped:
        del doc[nid]
    cond = doc[by_class["MiniMaxH3Conditioning"][0]]
    cond["inputs"].update(width=args.width, height=args.height, length=args.length)
    remove = ("MiniMaxH3Resolution",)
    whole_clip = args.mode in ("verify_whole_clip", "control_text_only")
    if whole_clip:
        remove += ("VAEDecode", "VAEDecodeAudio", "VHS_VideoCombine")
    for cls in remove:
        for nid in by_class.get(cls, []):
            del doc[nid]
    if args.mode != "plain_render":
        doc[by_class["KSamplerSelect"][0]] = {
            "class_type": "MiniMaxH3TaoMateStreamSampler",
            "inputs": {"mode": args.mode, "cache_device": args.cache_device}}
    if whole_clip:
        doc["900"] = {"class_type": "PreviewAny",
                      "inputs": {"source": [by_class["SamplerCustomAdvanced"][0], 0]}}
        return doc
    if args.prompt_id:
        import prompts
        # stripped (the house convention: one character is a different sample),
        # and at the length the bank entry was written for unless told otherwise
        cond["inputs"]["prompt"] = prompts.text(args.prompt_id).strip()
        want = (prompts.entry(args.prompt_id) or {}).get("frames")
        if want and int(want) != int(args.length) and not args.allow_off_length:
            raise SystemExit(f"bank prompt {args.prompt_id!r} is written for {want} frames and --length is "
                             f"{args.length}; pass --length {want}, or --allow-off-length if the mismatch is the test")
    if args.seed is not None:
        doc[by_class["RandomNoise"][0]]["inputs"]["noise_seed"] = args.seed
    if args.audio:
        doc[by_class["LoadAudio"][0]]["inputs"]["audio"] = args.audio
    doc[by_class["VHS_VideoCombine"][0]]["inputs"]["filename_prefix"] = args.prefix
    return doc


def git_head(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def log_lines_since(host, since_iso: str, tag: str) -> list[str]:
    logs = _get(host, "/internal/logs")
    text = logs if isinstance(logs, str) else json.dumps(logs)
    out = []
    for line in text.replace("\\n", "\n").split("\n"):
        m = _STAMP.match(line.strip())
        if tag in line and m and m.group(1) >= since_iso:
            out.append(line.strip())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the TaoMate stream sampler's check or throwaway render.")
    ap.add_argument("--mode", choices=tuple(PROBES), default="verify_whole_clip")
    ap.add_argument("--host", default="127.0.0.1:8188")
    ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--length", type=int, default=tm.frame_count(1))
    ap.add_argument("--allow-off-length", action="store_true",
                    help="render a bank prompt at a length its bank entry does not declare")
    ap.add_argument("--cache-device", default=None, choices=("cpu_pinned", "cpu", "gpu"),
                    help="default: gpu for verify (no cache is used), cpu_pinned for stream")
    ap.add_argument("--prompt-id", help="stream: a prompt bank id")
    ap.add_argument("--audio", help="stream: a file in ComfyUI's input directory")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--prefix", default="Video/h3_taomate_stream_throwaway")
    ap.add_argument("--timeout", type=float, default=7200.0)
    ap.add_argument("--record", type=Path)
    args = ap.parse_args(argv)
    if args.cache_device is None:
        args.cache_device = "cpu" if args.mode == "stream" else "gpu"
    whole_clip = args.mode in ("verify_whole_clip", "control_text_only")

    graph = build_graph(args)
    graph_sha = hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest()
    since = dt.datetime.now().isoformat()
    started = time.time()
    prompt_id = _post(args.host, "/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})["prompt_id"]
    print(f"queued {prompt_id}: {args.mode}, {args.width}x{args.height}, {args.length} frames", flush=True)
    history = None
    while time.time() - started < args.timeout:
        hist = _get(args.host, f"/history/{prompt_id}")
        if prompt_id in hist:
            history = hist[prompt_id]
            break
        time.sleep(5)
    if history is None:
        print("FAIL  timed out waiting for the run")
        return 2
    status = history.get("status", {})
    if status.get("status_str") != "success":
        errors = [m for m in status.get("messages", []) if m[0] == "execution_error"]
        print(f"FAIL  run did not succeed: {json.dumps(errors)[:3000]}")
        return 2
    wall = time.time() - started

    record = {
        "date": dt.date.today().isoformat(),
        "mode": args.mode,
        "produced_by": "bench/verify_taomate_stream.py",
        "graph_from": str(PROBES[args.mode].relative_to(REPO)),
        "graph_sha256": graph_sha,
        "prompt_id": prompt_id,
        "canvas": f"{args.width}x{args.height}",
        "length": args.length,
        "cache_device": args.cache_device,
        "repo_commit": git_head(REPO),
        "comfy_commit": git_head(REPO.parent.parent),
        "submit_to_finish_s": round(wall, 1),
        "is_not": "a quality statement; a small canvas proves the harness and the hook",
    }
    if whole_clip:
        tag = f"[taomate] {args.mode} "
        lines = log_lines_since(args.host, since, tag)
        if not lines:
            print(f"FAIL  the run succeeded but no {args.mode} report is in the server log buffer")
            return 2
        report = json.loads(lines[-1].split(tag, 1)[1].strip().replace('\\"', '"'))
        matched = all(report[s]["rel_rms"] <= MATCH_REL_RMS for s in ("video", "audio"))
        hooked = int(report.get("hook_calls", 0)) > 0
        for stream in ("video", "audio"):
            r = report[stream]
            print(f"  {stream}: exact {r['exact']}, max_abs {r['max_abs']:.3e}, rel_rms {r['rel_rms']:.3e}")
        print(f"  block hook calls: {report.get('hook_calls')}")
        if args.mode == "verify_whole_clip":
            # the hook must have run AND reproduced core
            passed = matched and hooked
            print(("ok    " if passed else "FAIL  ") + "hooked whole-clip loop vs core euler "
                  f"(bound rel_rms {MATCH_REL_RMS}, reasoned; hook ran: {hooked})")
            what = ("core's euler sampler and the sampler's own loop through its block attention hook, "
                    "same graph inputs, latents compared")
        else:
            # the control: upstream's text-only routing must NOT reproduce core
            passed = hooked and not matched
            print(("ok    " if passed else "FAIL  ") + "control: text-only routing departs from core "
                  f"(must exceed rel_rms {MATCH_REL_RMS}; hook ran: {hooked})")
            what = ("control for verify_whole_clip: the hooked loop with upstream's text-only routing, "
                    "which must differ from core's euler latent")
        record.update(what=what, match_bound_rel_rms=MATCH_REL_RMS, report=report, passed=passed)
        code = 0 if passed else 1
    elif args.mode == "plain_render":
        outputs = [f"{i.get('subfolder', '')}/{i['filename']}" for o in history.get("outputs", {}).values()
                   for v in o.values() if isinstance(v, list) for i in v
                   if isinstance(i, dict) and "filename" in i]
        passed = bool(outputs)
        print(("ok    " if passed else "FAIL  ") + f"plain whole-clip render; outputs {outputs}")
        record.update(what=("the TaoMate adapter over the whole clip with the stock sampler and every "
                            "attention patch stripped: the dense-attention whole-clip control"),
                      prompt_bank_id=args.prompt_id, audio=args.audio, seed=args.seed,
                      outputs=outputs, passed=passed)
        code = 0 if passed else 2
    else:
        lines = log_lines_since(args.host, since, CHUNK_TAG)
        expected = len(tm.run_plan(tm.requests_for(
            ((args.length - 5) // 17) * 5 + 2) or 1))
        outputs = [f"{i.get('subfolder', '')}/{i['filename']}" for o in history.get("outputs", {}).values()
                   for v in o.values() if isinstance(v, list) for i in v
                   if isinstance(i, dict) and "filename" in i]
        for line in lines:
            print("  " + line.split(" - ", 1)[-1])
        passed = len(lines) >= expected
        print(("ok    " if passed else "FAIL  ") + f"{len(lines)} chunk log line(s), expected {expected}; "
              f"outputs {outputs}")
        record.update(what="the TaoMate stream sampler's chunked, cached run with a frozen track",
                      prompt_bank_id=args.prompt_id, audio=args.audio, seed=args.seed,
                      chunk_log=lines, outputs=outputs, passed=passed)
        code = 0 if passed else 2

    if args.record is not None:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"record {args.record}")
    return code


if __name__ == "__main__":
    sys.exit(main())
