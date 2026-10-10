#!/usr/bin/env python3
"""Is ComfyUI's H3 keyframe VAE encode byte-deterministic on this card?

Subject: vllm-omni `30d6a0b4e` (#7191) wraps its keyframe VAE encode in
`cudnn.flags(benchmark=False, deterministic=True, allow_tf32=True)` after
keyframe latents varied between requests. Our keyframes go through core's
`vae.encode` (`comfy_extras/nodes_minimax_h3.py`: the conditioning node's
first/last-frame keyframes and `MiniMaxH3AddGuide`) under global flags.
`torch.backends.cudnn.benchmark` is already False here
(`comfy/model_management.py`), and the video VAE runs fp16, so `allow_tf32`
does not reach it. What remains is `deterministic=False`: cuDNN may pick a
different algorithm when the free workspace differs, and node caching means
an input is encoded once per server process, so a drift would show up across
restarts and VRAM states rather than inside one session.
`docs/open_experiments.md` #30 names this test.

Arms, each a FRESH process so no cuDNN or allocator state carries over:

  asfound      free VRAM as found, default flags
  fill         a ballast allocation leaving only the encode's peak plus a margin
  asfound2     the first arm again: process-to-process at the same VRAM state
  asfound_det  as found, `cudnn.deterministic=True` (vllm-omni's flags)
  fill_det     ballast, `cudnn.deterministic=True`

The ballast is sized from the `asfound` arm's own measured encode peak plus
`--headroom-gib`, unless `--margin-gib` fixes the free amount outright. The
first version left a guessed 8 GiB free and the encode ran out of memory,
which is why the peak is now read rather than assumed.

Inside each process the encode repeats `--repeats` times, which separates
in-process repeatability from cross-process agreement. A child records whether
ComfyUI fell back to tiled encoding (it logs a retry on OOM), because a tiled
latent differs for a reason that is not cuDNN's and must not be read as drift.

The image is synthetic and seeded -- no asset from the media directories --
smooth gradients plus seeded noise at the shipped canvas, resized through
core's own `_resize(..., "center")` as the keyframe path does.

Writes `data/results/<date>_keyframe_encode_determinism.json`: per arm the
sha256 of each latent's bytes, the max absolute difference against the first
arm, free VRAM at encode, the encode's peak, the flags, and the substrate.
Needs the card and no render in flight (it asks the server's /queue).

    <comfy-venv-python> bench/measure_keyframe_encode_determinism.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import tempfile
import urllib.request
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COMFY = REPO.parent.parent
sys.path.insert(0, str(REPO / "workflows"))

from h3_config import CANVAS, MODELS  # noqa: E402

ARMS = [
    ("asfound", False, False),
    ("fill", True, False),
    ("asfound2", False, False),
    ("asfound_det", False, True),
    ("fill_det", True, True),
]


def _child(args) -> int:
    sys.path.insert(0, str(COMFY))
    import comfy.options
    comfy.options.enable_args_parsing()
    sys.argv = [sys.argv[0]]
    import torch
    import comfy.sd
    import comfy.utils
    import folder_paths
    from comfy_extras.nodes_minimax_h3 import _resize

    tiled = []

    class _Tiled(logging.Handler):
        def emit(self, record):
            if "tiled" in record.getMessage().lower():
                tiled.append(record.getMessage())

    logging.getLogger().addHandler(_Tiled())

    path = folder_paths.get_full_path("vae", MODELS["video_vae"])
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(path))

    w, h = CANVAS["width"], CANVAS["height"]
    g = torch.Generator().manual_seed(20260910)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    base = torch.stack([xx, yy, (xx + yy) / 2], dim=-1)
    img = (0.8 * base + 0.2 * torch.rand(h, w, 3, generator=g)).clamp(0, 1)[None]
    img = _resize(img, w, h, "center")

    ballast = None
    if args.fill_free_gib is not None:
        free, _ = torch.cuda.mem_get_info()
        want = free - int(args.fill_free_gib * 2**30)
        if want > 0:
            ballast = torch.empty(want, dtype=torch.uint8, device="cuda")
    ballast_bytes = ballast.numel() if ballast is not None else 0

    hashes, latents = [], []
    with torch.backends.cudnn.flags(enabled=True, benchmark=False,
                                    deterministic=bool(args.deterministic),
                                    allow_tf32=torch.backends.cudnn.allow_tf32):
        free_at_encode = torch.cuda.mem_get_info()[0]
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.repeats):
            lat = vae.encode(img).detach().cpu().contiguous()
            hashes.append(hashlib.sha256(lat.view(torch.uint8).numpy().tobytes()).hexdigest())
            latents.append(lat)
    peak = torch.cuda.max_memory_allocated() - ballast_bytes
    torch.save(latents[0], args.save)
    del ballast
    print(json.dumps(dict(
        free_gib_at_encode=round(free_at_encode / 2**30, 2),
        encode_peak_gib=round(peak / 2**30, 2),
        ballast_gib=round(ballast_bytes / 2**30, 2),
        deterministic=bool(args.deterministic),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        vae_dtype=str(getattr(vae, "vae_dtype", None)),
        latent_shape=list(latents[0].shape), latent_dtype=str(latents[0].dtype),
        hashes=hashes, repeats_identical=len(set(hashes)) == 1,
        tiled_fallback=tiled)))
    return 0


def _queue_busy(host: str):
    try:
        with urllib.request.urlopen(f"http://{host}/queue", timeout=3) as r:
            q = json.load(r)
        return len(q.get("queue_running", [])) + len(q.get("queue_pending", []))
    except Exception:
        return None


def _substrate():
    import torch
    out = dict(torch=torch.__version__, cudnn=torch.backends.cudnn.version(),
               python=sys.version.split()[0],
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    try:
        out["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        out["driver"] = None
    for name, root in (("repo_commit", REPO), ("comfy_commit", COMFY)):
        try:
            out[name] = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                                       capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            out[name] = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--fill-free-gib", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--deterministic", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--save", help=argparse.SUPPRESS)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--headroom-gib", type=float, default=2.0,
                    help="free VRAM the ballast leaves beyond the as-found arm's "
                         "measured encode peak (reasoned: enough that a regular, "
                         "untiled encode still fits, little enough that cuDNN's "
                         "workspace is constrained)")
    ap.add_argument("--margin-gib", type=float, default=None,
                    help="fix the free VRAM the ballast leaves, overriding the "
                         "peak-derived amount")
    ap.add_argument("--host", default="127.0.0.1:8188")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.child:
        return _child(args)

    busy = _queue_busy(args.host)
    if busy:
        print(f"refusing: the server at {args.host} has {busy} prompt(s) queued or running")
        return 1

    import torch
    rows, fill_free = [], args.margin_gib
    with tempfile.TemporaryDirectory() as tmp:
        for label, fill, det in ARMS:
            save = str(Path(tmp) / f"{label}.pt")
            cmd = [sys.executable, str(Path(__file__).resolve()), "--child",
                   "--repeats", str(args.repeats), "--deterministic", str(int(det)),
                   "--save", save]
            if fill:
                if fill_free is None:
                    fill_free = rows[0]["encode_peak_gib"] + args.headroom_gib
                cmd += ["--fill-free-gib", str(fill_free)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(proc.stdout[-2000:], proc.stderr[-4000:])
                print(f"FAIL: arm {label} exited {proc.returncode}")
                return 1
            row = json.loads(proc.stdout.strip().splitlines()[-1])
            row["label"] = label
            row["_save"] = save
            rows.append(row)
            print(f"  {label:12} free {row['free_gib_at_encode']:>6} GiB  peak "
                  f"{row['encode_peak_gib']:>5} GiB  det={det!s:5} "
                  f"repeats identical={row['repeats_identical']}  "
                  f"tiled={bool(row['tiled_fallback'])}  {row['hashes'][0][:16]}")
        ref = torch.load(rows[0]["_save"], weights_only=True)
        for row in rows:
            lat = torch.load(row.pop("_save"), weights_only=True)
            row["same_bytes_as_first_arm"] = row["hashes"][0] == rows[0]["hashes"][0]
            row["max_abs_diff_vs_first_arm"] = float((lat.float() - ref.float()).abs().max())

    verdict = ("byte-identical across every arm and repeat"
               if len({r["hashes"][0] for r in rows}) == 1 and all(r["repeats_identical"] for r in rows)
               else "NOT byte-identical: read the per-arm rows")
    print(f"\n  {verdict}")
    out = Path(args.out) if args.out else HERE.parent / "data" / "results" / f"{date.today().isoformat()}_keyframe_encode_determinism.json"
    out.write_text(json.dumps(dict(
        produced_by="bench/measure_keyframe_encode_determinism.py",
        question="does the H3 keyframe VAE encode give the same latent bytes across "
                 "fresh processes, free-VRAM states and cudnn.deterministic "
                 "(vllm-omni 30d6a0b4e; docs/open_experiments.md #30)",
        vae=MODELS["video_vae"], canvas=CANVAS, repeats=args.repeats,
        fill_free_gib=fill_free, headroom_gib=args.headroom_gib,
        substrate=_substrate(), arms=rows, verdict=verdict), indent=2) + "\n")
    print(f"  wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
