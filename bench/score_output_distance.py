#!/usr/bin/env python3
"""Score renders by their saved latents' distance from a reference render at the same seed.

    python bench/score_output_distance.py --output-root <the server's output directory> \\
        --stem h3_probe_r2v_step_switch_pdd8_flashgen_h080_savelat \\
        --reference dense=sweep_a --arm shipped=ship:2 --arm stock=od_market__stock \\
        --other-seed ship:1 --other-seed ship:3 --other-seed-of shipped \\
        --out data/results/<date>_<what>.json

The generalisation of `bench/score_sol_dense_blocks_panel.py`, which needs
the server that rendered its rows: this one reads the files. A render is named
by the label `bench/run_graph_arms.py` put in its file names and the file
counter (`label:counter`, counter 1 when omitted), on a `*_savelat` graph
whose four `SaveLatent` prefixes end in `_video`, `_audio`, `_pass1_video` and
`_pass1_audio`.

Per arm against the reference: relative L2 and cosine of the final video
latent, of the same latent average-pooled over 8x8 latent pixels (what
happens, as against how it looks), of the video latent after the first
sampler, and of the final audio latent; and, when the clips are there, EBU
R128 integrated loudness of each clip's audio against the reference's
(`bench/measure_clip_loudness.py::ebur128`) and the number of cuts in the clip
(`bench/measure_clip_delta.py::cut_times`), which `--shots` sets against the
number of shots the prompt scripts.

`--other-seed` names renders of ONE configuration at other seeds, and
`--other-seed-of` the arm that is the same configuration at the reference's
seed: their mutual distance is what an unrelated sample of the same prompt
scores, the top of the scale. Every arm's distance is also given as a share
of it.

Descriptive. A rendered clip cannot A/B a numerical change (AGENTS.md): this
measures how far a change moved the sample, never whether the new sample is
worse.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import safetensors.torch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_clip_delta import cut_times  # noqa: E402
from measure_clip_loudness import ebur128, has_audio  # noqa: E402

KINDS = ("video", "audio", "pass1_video")
COARSE = 8      # latent pixels per side of the pooling window. **Reasoned**: a 48x84 latent frame becomes 6x10.


def files(root: Path, stem: str, spec: str) -> dict:
    label, _, counter = spec.partition(":")
    n = int(counter or 1)
    out = {kind: root / "latents" / f"{stem}_{kind}_{label}_{n:05d}_.latent" for kind in KINDS}
    out["clip"] = root / "Video" / f"{stem}_{label}_{n:05d}-audio.mp4"
    missing = [p.name for k, p in out.items() if k != "clip" and not p.is_file()]
    if missing:
        raise SystemExit(f"{spec}: no such latent file(s): {missing}")
    return out


def load(path: Path) -> torch.Tensor:
    return safetensors.torch.load_file(str(path))["latent_tensor"].double()


def pair(a: torch.Tensor, b: torch.Tensor) -> dict:
    return {"rel": float((a - b).norm() / b.norm()),
            "cos": float((a * b).sum() / (a.norm() * b.norm()))}


def coarse(z: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.avg_pool3d(z, (1, COARSE, COARSE))


def compare(a: dict, ref: dict) -> dict:
    za, zr = load(a["video"]), load(ref["video"])
    if za.shape != zr.shape:
        raise SystemExit(f"{a['video'].name}: shape {tuple(za.shape)} against the reference's {tuple(zr.shape)}")
    return {"video": pair(za, zr), "video_coarse": pair(coarse(za), coarse(zr)),
            "pass1_video": pair(load(a["pass1_video"]), load(ref["pass1_video"])),
            "audio": pair(load(a["audio"]), load(ref["audio"]))}


def loudness(f: dict) -> dict | None:
    clip = f["clip"]
    if not clip.is_file() or not has_audio(clip):
        return None
    return ebur128(clip)


def cuts(f: dict) -> list | None:
    return cut_times(f["clip"]) if f["clip"].is_file() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--stem", required=True, help="the graph's file-name stem, up to the latent kind")
    ap.add_argument("--reference", required=True, metavar="NAME=LABEL[:N]")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=LABEL[:N]")
    ap.add_argument("--other-seed", action="append", default=[], metavar="LABEL[:N]")
    ap.add_argument("--other-seed-of", default=None, metavar="NAME", help="the arm the other seeds are a configuration of")
    ap.add_argument("--shots", type=int, default=None, help="shots the prompt scripts: a clip should hold one cut fewer")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ref_name, _, ref_spec = args.reference.partition("=")
    ref = files(args.output_root, args.stem, ref_spec)
    arms = {}
    for spec in args.arm:
        name, _, where = spec.partition("=")
        arms[name] = files(args.output_root, args.stem, where)

    ceiling = None
    if args.other_seed:
        if args.other_seed_of not in arms:
            raise SystemExit("--other-seed needs --other-seed-of naming one of the arms")
        group = [arms[args.other_seed_of]] + [files(args.output_root, args.stem, s) for s in args.other_seed]
        pairs = [compare(group[i], group[j]) for i in range(len(group)) for j in range(i + 1, len(group))]
        ceiling = {k: sum(p[k]["rel"] for p in pairs) / len(pairs) for k in pairs[0]}
        ceiling["pairs"] = len(pairs)

    ref_loud = loudness(ref)
    result = {"measured_by": "bench/score_output_distance.py", "stem": args.stem,
              "scripted_shots": args.shots,
              "reference": {"name": ref_name, "files": {k: p.name for k, p in ref.items()}, "loudness": ref_loud,
                            "cut_times_s": cuts(ref)},
              "other_seed_distance": ceiling, "arms": {}}
    ref_cuts = result["reference"]["cut_times_s"]
    print(f"{'arm':<16} {'video':>7} {'coarse':>7} {'pass1':>7} {'audio':>7} {'of ceiling':>10} {'LU vs ref':>9} {'cuts':>5}")
    print(f"{ref_name:<16} {'(the reference)':>47} {'':>9} {len(ref_cuts) if ref_cuts is not None else '-':>5}")
    for name, f in arms.items():
        c = compare(f, ref)
        loud = loudness(f)
        entry = {"files": {k: p.name for k, p in f.items()}, **c, "loudness": loud, "cut_times_s": cuts(f),
                 "video_share_of_other_seed": c["video"]["rel"] / ceiling["video"] if ceiling else None,
                 "loudness_delta_lu": (loud["integrated_lufs"] - ref_loud["integrated_lufs"]) if loud and ref_loud else None}
        result["arms"][name] = entry
        share = f"{entry['video_share_of_other_seed']:.2f}" if ceiling else "-"
        lu = f"{entry['loudness_delta_lu']:+.1f}" if entry["loudness_delta_lu"] is not None else "-"
        print(f"{name:<16} {c['video']['rel']:>7.3f} {c['video_coarse']['rel']:>7.3f} {c['pass1_video']['rel']:>7.3f} "
              f"{c['audio']['rel']:>7.3f} {share:>10} {lu:>9} "
              f"{len(entry['cut_times_s']) if entry['cut_times_s'] is not None else '-':>5}")
    if ceiling:
        print(f"{'other seed':<16} {ceiling['video']:>7.3f} {ceiling['video_coarse']:>7.3f} {ceiling['pass1_video']:>7.3f} "
              f"{ceiling['audio']:>7.3f}   ({ceiling['pairs']} pairs)")
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
