#!/usr/bin/env python3
"""Decode one latent with core's current and an earlier `tiled_decode`, and compare seams.

Comfy-Org/ComfyUI#16436 (`fc584aaa`) changed how core's H3 video VAE blends its
spatial tiles. A render from before that change is not bit-comparable with one
after it. This probe says where the two differ and whether either sits closer
to the source. The record is `bench/results/2026-09-25_vae_tile_seam_blend.md`.

Each frame (a PNG at a trained canvas) is encoded by core to a one-frame latent,
un-normalised as `decode` does, and decoded twice by the same
`MiniMaxH3VideoVAE` instance: once by the current `tiled_decode`, once by the
revision's, taken from `git show <rev>:comfy/ldm/minimax/vae.py` and bound to
the instance. Nothing in core is edited. Needs the GPU and the ComfyUI venv;
run it with no server holding the card.

    python bench/compare_vae_tiled_decode.py FRAME.png [FRAME.png ...] [--rev fc584aaa~1]
"""

from __future__ import annotations

import argparse
import ast
import math
import subprocess
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMFY = REPO.parents[1]


def old_tiled_decode(rev: str, namespace: dict):
    src = subprocess.run(["git", "-C", str(COMFY), "show", f"{rev}:comfy/ldm/minimax/vae.py"],
                         check=True, capture_output=True, text=True).stdout
    cls = next(n for n in ast.parse(src).body
               if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3VideoVAE")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "tiled_decode")
    ns = dict(namespace)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), f"{rev}:tiled_decode", "exec"), ns)
    return ns["tiled_decode"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", nargs="+", type=Path)
    ap.add_argument("--rev", default="fc584aaa~1", help="core revision holding the earlier tiled_decode")
    args = ap.parse_args()

    sys.path.insert(0, str(COMFY))
    sys.path.insert(0, str(REPO / "workflows"))
    sys.argv = sys.argv[:1]
    import numpy as np
    import torch
    from PIL import Image
    import comfy.sd
    import comfy.utils
    import comfy.model_management as mm
    import comfy.ldm.minimax.vae as V
    import folder_paths
    import h3_config

    old = old_tiled_decode(args.rev, vars(V))
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(
        folder_paths.get_full_path_or_raise("vae", h3_config.MODELS["video_vae"])))
    m = vae.first_stage_model
    dev = mm.get_torch_device()

    for path in args.frames:
        img = torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0)[None]
        z = vae.encode(img)
        m.to(dev)
        dtype = next(m.parameters()).dtype
        with torch.inference_mode():
            zz = z.to(dev)
            zr = (zz * m.latents_std.view(1, -1, 1, 1, 1).to(zz)
                  + m.latents_mean.view(1, -1, 1, 1, 1).to(zz)).to(dtype)
            new = m._finalize_pixels(m.tiled_decode(zr).float())[:, :, -1]
            was = m._finalize_pixels(types.MethodType(old, m)(zr).float())[:, :, -1]
        H, W = new.shape[-2:]
        yi, _, yo = m.split_tiles(H)
        xi, _, xo = m.split_tiles(W)
        band = torch.zeros(H, W, dtype=torch.bool)
        for i in range(len(yi) - 1):
            band[yi[i + 1]: yi[i + 1] + yo[i], :] = True
        for j in range(len(xi) - 1):
            band[:, xi[j + 1]: xi[j + 1] + xo[j]] = True
        d = (new - was).abs().amax(dim=1)[0].cpu()
        src = img[0].permute(2, 0, 1)[None].to(new.device)

        def psnr(a, mask=None):
            e = ((a - src) ** 2).mean(dim=1)[0].cpu()
            return 10 * math.log10(1.0 / (e[mask] if mask is not None else e).mean().item())

        def rows(x, rs):
            return float(np.mean([(x[0, :, r] - x[0, :, r - 1]).abs().mean().item() for r in rs]))

        def cols(x, cs):
            return float(np.mean([(x[0, :, :, c] - x[0, :, :, c - 1]).abs().mean().item() for c in cs]))

        seam_r = [yi[i + 1] + yo[i] // 2 for i in range(len(yi) - 1)]
        seam_c = [xi[j + 1] + xo[j] // 2 for j in range(len(xi) - 1)]
        ctrl_r = [r for r in range(8, H - 8, 37) if not band[r, 0]][:12]
        print(f"{path.name}  {W}x{H}  tiles {len(yi)}x{len(xi)}  bands {band.float().mean().item():.3f} of the canvas")
        print(f"  max |new-old| outside bands {d[~band].max().item():.3g}   in bands max {d[band].max().item():.3g} mean {d[band].mean().item():.3g}")
        print(f"  PSNR vs source   old {psnr(was):.3f}  new {psnr(new):.3f}   in bands old {psnr(was, band):.3f}  new {psnr(new, band):.3f}")
        print(f"  seam row step    old {rows(was, seam_r):.5f}  new {rows(new, seam_r):.5f}   interior control {rows(new, ctrl_r):.5f}")
        print(f"  seam col step    old {cols(was, seam_c):.5f}  new {cols(new, seam_c):.5f}")
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
