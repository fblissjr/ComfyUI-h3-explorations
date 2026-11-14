#!/usr/bin/env python3
"""How the MiniMax-H3-Image VAE's 256 px decode tiles shape its pictures, measured on real pictures with no DiT.

Each test picture is encoded once with the image VAE's encoder (the same encoder in every image VAE
here) and its latent decoded by each decoder named, through core's own classes, on CPU in float32:

1. **Tile disagreement.** At the release's geometry (256 px tiles, 64 px minimum overlap) every
   tile is decoded on its own; in each overlap the two neighbours' raw decodes are compared with
   each other, overall and in their fine detail (each minus a 5x5 box blur), beside each decode's
   own error against the picture. Two decodes that disagree in fine detail are averaged by the
   linear blend, which softens what neither got wrong.
2. **Error against distance from the tile edge.** The same raw tiles, unstitched: RMS error against
   the picture in 16 px bands of distance from the tile's nearest edge.
3. **Stitching.** Core's (blends each tile against the already-blended canvas) against diffusers'
   `_stitch_tiles` (blends against the neighbours' raw decodes), lifted from diffusers' source: PSNR
   overall and inside the row-and-column overlap crossings, and the largest step between
   neighbouring pixels beyond the picture's own.
4. **Tile geometry.** Core's decode at several (tile, overlap) settings and untiled: PSNR, fine
   luma detail kept (high-pass energy of the decode over the picture's), and chroma grain in
   shadows (high-pass energy of Cb and Cr where the picture's luma is under 0.15, decode over
   picture; above 1 means the decode added colour speckle there).

`--cuda` runs on the card (float32, one decoder resident at a time); without it, CPU.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/probe_h3_image_vae_tiles.py \
        --release <MiniMax-H3-Image folder> --out data/h3_image_vae_<date> \
        --vae release=<file> --vae p-h3=<file> [--vae hdr=<file>] [--cuda]

Writes `results.json` and, per picture, stacked crops (`crops_<picture>.png`) to `--out`. Reads
nothing over the network; needs a card only with `--cuda`. Cannot see: how a DiT's latents (rather than an encoded
picture's) decode, or anything about the eye's judgement of the result.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import bootstrap  # noqa: E402

PACK = Path(__file__).resolve().parent.parent
CUDA = "--cuda" in sys.argv
bootstrap(cpu=not CUDA)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402

R = 16
DARK = 0.15
GEOMETRIES = [(256, 64), (256, 128), (384, 96), (512, 64), (512, 128), None]   # None: untiled

# (name, file under the release folder, crop box (left, top, width, height) or None for the whole)
PICTURES = [
    ("reference1", "examples/reference1.png", None),
    ("text_panel", "assets/showcase/text.jpg", (1952, 0, 1024, 832)),
    ("dark_portrait", "assets/showcase/photo.jpg", (1008, 0, 976, 544)),
    ("station", "assets/showcase/scene.jpg", (0, 848, 1024, 832)),
]


def diffusers_stitch():
    src = PACK / "coderef/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_image.py"
    tree = ast.parse(src.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AutoencoderKLMiniMaxH3Image")
    ns = {"torch": torch, "math": math}
    for fn in cls.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in ("_blend", "_stitch_tiles"):
            fn.decorator_list = []
            exec(compile(ast.Module([fn], []), str(src), "exec"), ns)

    class Ref:
        _blend = ns["_blend"]
        _stitch_tiles = ns["_stitch_tiles"]
    return Ref()


def load_picture(release: Path, rel: str, box) -> torch.Tensor:
    im = Image.open(release / rel).convert("RGB")
    if box:
        left, top, w, h = box
        im = im.crop((left, top, left + w, top + h))
    w, h = im.size
    im = im.crop((0, 0, w - w % R, h - h % R))
    return torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0)[None]      # [1, H, W, 3]


def ycbcr(x):  # x [3, H, W] in [0, 1]
    r, g, b = x
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y, 0.564 * (b - y), 0.713 * (r - y)


def highpass(t):  # [..., H, W]
    t4 = t.reshape(-1, 1, *t.shape[-2:])
    low = F.avg_pool2d(F.pad(t4, (2, 2, 2, 2), mode="reflect"), 5, stride=1)
    return (t4 - low).reshape(t.shape)


def psnr(a, b, mask=None):
    d = (a - b) ** 2
    mse = d.mean() if mask is None else d[..., mask].mean()
    return float(-10 * torch.log10(mse.clamp_min(1e-12)))


def picture_metrics(dec, pic):  # both [3, H, W] in [0, 1]
    yp, cbp, crp = ycbcr(pic)
    yd, cbd, crd = ycbcr(dec)
    dark = yp < DARK
    out = {"psnr": psnr(dec, pic),
           "luma_detail_kept": float((highpass(yd) ** 2).mean() / (highpass(yp) ** 2).mean())}
    if dark.float().mean() > 0.01:
        grain = lambda cb, cr: float(((highpass(cb) ** 2 + highpass(cr) ** 2)[dark]).mean())  # noqa: E731
        out["shadow_chroma_grain"] = grain(cbd, crd) / grain(cbp, crp)
        out["shadow_fraction"] = float(dark.float().mean())
    return out


def split(fsm, n):
    return fsm.split_tiles(n)


def raw_tiles(fsm, z):
    """Every tile's raw decode at the model's current geometry, finalized to [0, 1] pixels."""
    zd = z * fsm.latents_std.view(1, -1, 1, 1, 1) + fsm.latents_mean.view(1, -1, 1, 1, 1)
    H, W = z.shape[-2] * R, z.shape[-1] * R
    yi, yl, yo = split(fsm, H)
    xi, xl, xo = split(fsm, W)
    rows = []
    for i, li in zip(yi, yl):
        row = []
        for j, lj in zip(xi, xl):
            t = fsm._decode_pixels(zd[..., i // R:(i + li) // R, j // R:(j + lj) // R])
            row.append(t[:, :, 0])                                         # raw decoder space [1, 3, h, w]
        rows.append(row)
    return rows, (yi, yl, yo), (xi, xl, xo)


def finalize(fsm, t):
    return fsm._finalize_pixels(t.unsqueeze(2))[0, :, 0]


def tile_study(fsm, z, pic, ref):
    rows, (yi, yl, yo), (xi, xl, xo) = raw_tiles(fsm, z)
    fin = [[finalize(fsm, t) for t in row] for row in rows]
    pairs = []
    for i in range(len(rows)):
        for j in range(len(rows[0])):
            a = fin[i][j]
            for (ni, nj, axis) in ((i, j + 1, "x"), (i + 1, j, "y")):
                if ni >= len(rows) or nj >= len(rows[0]):
                    continue
                b = fin[ni][nj]
                if axis == "x":
                    o = xo[j]
                    pa, pb = a[:, :, -o:], b[:, :, :o]
                    pp = pic[:, yi[i]:yi[i] + yl[i], xi[nj]:xi[nj] + o]
                else:
                    o = yo[i]
                    pa, pb = a[:, -o:, :], b[:, :o, :]
                    pp = pic[:, yi[ni]:yi[ni] + o, xi[j]:xi[j] + xl[j]]
                rms = lambda t: float(t.pow(2).mean().sqrt())  # noqa: E731
                pairs.append({
                    "disagree": rms(pa - pb), "disagree_detail": rms(highpass(pa) - highpass(pb)),
                    "err_a": rms(pa - pp), "err_b": rms(pb - pp), "picture_detail": rms(highpass(pp)),
                    "blend_detail": rms(highpass((pa + pb) / 2)), "a_detail": rms(highpass(pa)),
                })
    mean = lambda k: float(np.mean([p[k] for p in pairs]))  # noqa: E731
    # error against distance from the tile's nearest edge, raw tiles
    bands = {}
    for i in range(len(rows)):
        for j in range(len(rows[0])):
            t = fin[i][j]
            h, w = t.shape[-2:]
            pp = pic[:, yi[i]:yi[i] + h, xi[j]:xi[j] + w]
            yy = torch.arange(h, device=t.device).view(-1, 1)
            xx = torch.arange(w, device=t.device).view(1, -1)
            dist = torch.minimum(torch.minimum(yy, h - 1 - yy), torch.minimum(xx, w - 1 - xx))
            e = ((t - pp) ** 2).mean(0)
            for lo in range(0, 128, 16):
                m = (dist >= lo) & (dist < lo + 16)
                if m.any():
                    s = bands.setdefault(lo, [0.0, 0])
                    s[0] += float(e[m].sum())
                    s[1] += int(m.sum())
    edge = {f"{lo}-{lo + 16}px": math.sqrt(s / n) for lo, (s, n) in sorted(bands.items())}
    # stitching: core's (through its own decode) against diffusers' on the same raw tiles
    core_img = fsm.decode(z)[0, :, 0].float()
    ref_img = fsm._finalize_pixels(ref._stitch_tiles([[t.unsqueeze(2) for t in r] for r in rows], yo, xo))[0, :, 0]
    H, W = pic.shape[-2:]
    rb = torch.zeros(H, dtype=torch.bool, device=pic.device)
    cb = torch.zeros(W, dtype=torch.bool, device=pic.device)
    for k, o in enumerate(yo):
        rb[yi[k + 1]:yi[k + 1] + o] = True
    for k, o in enumerate(xo):
        cb[xi[k + 1]:xi[k + 1] + o] = True
    cross = rb.view(-1, 1) & cb.view(1, -1)
    band = rb.view(-1, 1) | cb.view(1, -1)

    def step(img):
        r = img - pic
        return max(float(r.diff(dim=1).abs().max()), float(r.diff(dim=2).abs().max()))
    stitch = {
        "core_psnr": psnr(core_img, pic), "diffusers_psnr": psnr(ref_img, pic),
        "core_psnr_crossings": psnr(core_img, pic, cross), "diffusers_psnr_crossings": psnr(ref_img, pic, cross),
        "core_max_step": step(core_img), "diffusers_max_step": step(ref_img),
        "max_abs_core_minus_diffusers": float((core_img - ref_img).abs().max()),
        "overlap_area": float(band.float().mean()), "crossing_area": float(cross.float().mean()),
        "core_psnr_in_overlaps": psnr(core_img, pic, band), "core_psnr_outside_overlaps": psnr(core_img, pic, ~band),
    }
    return {
        "pairs": len(pairs),
        "disagree_rms": mean("disagree"), "disagree_detail_rms": mean("disagree_detail"),
        "own_error_rms": (mean("err_a") + mean("err_b")) / 2, "picture_detail_rms": mean("picture_detail"),
        "one_tile_detail_rms": mean("a_detail"), "blended_detail_rms": mean("blend_detail"),
        "error_by_distance_from_tile_edge": edge, "stitching": stitch,
    }, core_img


def crops(pic, outs, name, out_dir):
    """Two 256 px crops (darkest and most detailed window on a 64 px grid), stacked: picture, then each decode."""
    y, _, _ = ycbcr(pic)
    H, W = y.shape
    best_dark = best_detail = (math.inf, 0, 0)
    hp = highpass(y).abs()
    for top in range(0, H - 255, 64):
        for left in range(0, W - 255, 64):
            d = float(y[top:top + 256, left:left + 256].mean())
            s = float(hp[top:top + 256, left:left + 256].mean())
            if d < best_dark[0]:
                best_dark = (d, top, left)
            if best_detail[0] == math.inf or s > best_detail[0]:
                best_detail = (s, top, left)
    to_img = lambda t: Image.fromarray((t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))  # noqa: E731
    labels = ["picture"] + list(outs)
    tiles = [pic] + list(outs.values())
    sheet = Image.new("RGB", (256 * len(tiles), 512), "white")
    for k, t in enumerate(tiles):
        for r, (_, top, left) in enumerate((best_dark, best_detail)):
            sheet.paste(to_img(t[:, top:top + 256, left:left + 256]), (256 * k, 256 * r))
    path = out_dir / f"crops_{name}.png"
    sheet.save(path)
    return {"file": path.name, "columns": labels, "rows": [f"darkest 256 px at (x={best_dark[2]}, y={best_dark[1]})",
                                                           f"most detailed 256 px at (x={best_detail[2]}, y={best_detail[1]})"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vae", action="append", required=True, help="name=file; the first is the release decoder")
    ap.add_argument("--pictures", nargs="*", help="limit to these picture names")
    ap.add_argument("--cuda", action="store_true", help="run on the card")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    ref = diffusers_stitch()
    dev = torch.device("cuda" if args.cuda else "cpu")

    vaes = {}
    for spec in args.vae:
        name, path = spec.split("=", 1)
        sd, md = comfy.utils.load_torch_file(path, return_metadata=True)
        v = comfy.sd.VAE(sd=sd, metadata=md)
        fsm = v.first_stage_model.float()
        assert not any(p.is_meta for p in fsm.parameters()), name
        vaes[name] = (v, fsm)
        print(f"loaded {name}: {type(fsm).__name__}", flush=True)
    first = next(iter(vaes))

    results = {"geometries": [list(g) if g else "untiled" for g in GEOMETRIES], "dark_threshold": DARK, "pictures": {}}
    for pname, rel, box in PICTURES:
        if args.pictures and pname not in args.pictures:
            continue
        t0 = time.time()
        x = load_picture(args.release, rel, box)
        pic = x[0].permute(2, 0, 1).to(dev)
        enc = vaes[first][1].to(dev)
        z = enc.encode(pic[None, :, None] * 2 - 1).float()            # [-1, 1] in, normalized latent out
        enc.to("cpu")
        rec = {"file": rel, "crop": box, "size": [pic.shape[2], pic.shape[1]], "decoders": {}}
        shown = {}
        for name, (_, fsm) in vaes.items():
            fsm.to(dev)
            fsm.tiling, fsm.tile_size, fsm.tile_overlap_min = True, 256, 64
            study, img = tile_study(fsm, z, pic, ref)
            geo = {}
            for g in GEOMETRIES:
                if g is None:
                    fsm.tiling = False
                    key = "untiled"
                else:
                    fsm.tiling, (fsm.tile_size, fsm.tile_overlap_min) = True, g
                    key = f"{g[0]}/{g[1]}"
                dec = img if g == (256, 64) else fsm.decode(z)[0, :, 0].float()
                geo[key] = picture_metrics(dec, pic)
                if name == first and key == "untiled":
                    shown[f"{name} untiled"] = dec
            fsm.tiling, fsm.tile_size, fsm.tile_overlap_min = True, 256, 64
            shown[f"{name} 256/64"] = img
            rec["decoders"][name] = {"tiles_256_64": study, "geometry": geo}
            fsm.to("cpu")
            if CUDA:
                torch.cuda.empty_cache()
            print(f"  {pname} {name}: {json.dumps(geo['256/64'])}", flush=True)
        order = [k for k in shown if k.endswith("256/64")] + [k for k in shown if k.endswith("untiled")]
        rec["crops"] = crops(pic, {k: shown[k] for k in order}, pname, args.out)
        rec["seconds"] = round(time.time() - t0, 1)
        results["pictures"][pname] = rec
        (args.out / "results.json").write_text(json.dumps(results, indent=1))
        print(f"{pname} done in {rec['seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
