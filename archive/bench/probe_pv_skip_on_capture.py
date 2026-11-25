#!/usr/bin/env python3
"""How many (warp, key tile) pairs of dense H3 attention carry a
negligible P*V term, and what skipping them costs, on captured q/k/v.

The kitchen dense INT8 kernel (qk_int_sv_i8_cuda.cuh, sm_89 path) walks key
tiles of 64 in index order; each warp holds 16 query rows. Per row and tile it
has tile_scale = 2^-(running_max - tile_max) before it issues the P*V MMAs,
which are half the kernel's tensor-core work. This asks: if a warp skipped
P*V when every one of its 16 rows has tile_scale < 2^-T, how many tiles go,
and how far does the output move? Denominators stay exact (the kernel has
them before P*V either way).

Three rules:
  causal  running max in the kernel's own tile order
  seeded  the same, with the running max seeded from the row's own tile
          (a kernel that visits the diagonal tile first)
  oracle  against the row's final max (upper bound on what any order gives)

Reference for scale: the same output with P rounded to u8 against each
tile's own max, which is what the kernel does to every tile it keeps.

fp32 emulation on the capture's bf16 q/k/v; no INT8 q/k/v error is modelled.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

TILE = 64
WARP = 16
THRESH = (8, 12, 16, 20, 24, 29)


def sample_warps(segments, per_seg):
    """Warp start rows (multiples of 16), stratified by segment."""
    out = []
    for name, (a, b) in segments.items():
        lo, hi = (a + WARP - 1) // WARP, b // WARP
        n = min(per_seg[name], max(hi - lo, 0))
        if n <= 0:
            continue
        idx = torch.linspace(lo, hi - 1, n).round().long().unique()
        out += [(name, int(i) * WARP) for i in idx]
    return out


def run_cell(path, segments, per_seg, heads, device):
    data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    q, k, v = data["q"], data["k"], data["v"]          # [1, H, S, D] bf16
    _, H, S, D = q.shape
    n_tiles = (S + TILE - 1) // TILE
    pad = n_tiles * TILE - S
    scale_log2 = (D ** -0.5) / math.log(2.0)
    warps = sample_warps(segments, per_seg)
    rows = torch.tensor([r + i for _, r in warps for i in range(WARP)])
    seg_of_warp = [name for name, _ in warps]
    own_tile = (torch.tensor([r for _, r in warps]) // TILE).to(device)
    W = len(warps)
    video_start = segments["video"][0]
    prefix_tiles = (video_start + TILE - 1) // TILE

    head_ids = list(range(H)) if heads <= 0 else \
        torch.linspace(0, H - 1, heads).round().long().unique().tolist()
    acc = {}

    def add(key, val, n=1):
        s = acc.setdefault(key, [0.0, 0])
        s[0] += float(val)
        s[1] += n

    worst = {}
    for h in head_ids:
        qh = q[0, h][rows].to(device, torch.float32)       # [W*16, D]
        kh = k[0, h].to(device, torch.float32)             # [S, D]
        vh = v[0, h].to(device, torch.float32)
        l2 = (qh @ kh.T) * scale_log2                      # log2-domain scores
        if pad:
            l2p = torch.nn.functional.pad(l2, (0, pad), value=float("-inf"))
        else:
            l2p = l2
        lt = l2p.view(W * WARP, n_tiles, TILE)
        tile_m = lt.amax(dim=-1)                           # [R, T]
        m_final = tile_m.amax(dim=-1, keepdim=True)
        run = torch.cummax(tile_m, dim=1).values
        seed = tile_m.gather(1, own_tile.repeat_interleave(WARP)[:, None])
        run_seeded = torch.maximum(run, seed)
        gaps = {"causal": tile_m - run, "seeded": tile_m - run_seeded,
                "oracle": tile_m - m_final}                # <= 0, log2

        p = torch.softmax(l2 * math.log(2.0), dim=-1)      # exact fp32 rows
        out_full = p @ vh
        norm_full = out_full.norm(dim=-1).clamp_min(1e-20)

        # the kernel's own u8 rounding of P against each tile's max
        pu8 = torch.round(255.0 * torch.exp2(lt - tile_m[..., None])).clamp_(0, 255)
        w_u8 = (pu8 * torch.exp2(tile_m - m_final)[..., None]).view(W * WARP, -1)[:, :S]
        out_u8 = (w_u8 @ vh) / w_u8.sum(-1, keepdim=True)
        err_u8 = (out_u8 - out_full).norm(dim=-1) / norm_full
        del pu8, w_u8, out_u8

        p_tile = torch.nn.functional.pad(p, (0, pad)).view(W * WARP, n_tiles, TILE)
        for rule, gap in gaps.items():
            wgap = gap.view(W, WARP, n_tiles).amax(dim=1)  # warp skips only if all rows agree
            for T in THRESH:
                skip_w = wgap < -T                          # [W, T]
                skip_r = skip_w.repeat_interleave(WARP, dim=0)
                keep = (~skip_r)[..., None]
                out_skip = (p_tile * keep).view(W * WARP, -1)[:, :S] @ vh
                err = (out_skip - out_full).norm(dim=-1) / norm_full
                mass = (p_tile.sum(-1) * skip_r).sum(-1)
                for name in set(seg_of_warp):
                    wm = torch.tensor([s == name for s in seg_of_warp], device=device)
                    rm = wm.repeat_interleave(WARP)
                    key = (rule, T, name)
                    add(key + ("skip_frac",), skip_w[wm].float().mean())
                    add(key + ("skip_frac_prefix",), skip_w[wm][:, :prefix_tiles].float().mean())
                    add(key + ("skip_frac_video",), skip_w[wm][:, prefix_tiles:].float().mean())
                    add(key + ("err_mean",), err[rm].mean())
                    add(key + ("mass_mean",), mass[rm].mean())
                    wk = key + ("err_max",)
                    worst[wk] = max(worst.get(wk, 0.0), float(err[rm].max()))
        for name in set(seg_of_warp):
            rm = torch.tensor([s == name for s in seg_of_warp], device=device).repeat_interleave(WARP)
            add(("u8", 0, name, "err_mean"), err_u8[rm].mean())
            wk = ("u8", 0, name, "err_max")
            worst[wk] = max(worst.get(wk, 0.0), float(err_u8[rm].max()))
        del l2, l2p, lt, p, p_tile, out_full

    res = {}
    for key, (s, n) in acc.items():
        res["/".join(map(str, key))] = s / n
    for key, val in worst.items():
        res["/".join(map(str, key))] = val
    return {"cell": Path(path).name, "tokens": S, "heads": len(head_ids), "warps": W,
            "tiles": n_tiles, "prefix_tiles": prefix_tiles, "stats": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir")
    ap.add_argument("--cells", nargs="*", default=None, help="substrings like b40_s2")
    ap.add_argument("--heads", type=int, default=0, help="0 = all")
    ap.add_argument("--video-warps", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.capture_dir)
    man = json.loads((root / "manifest.json").read_text())
    ta = man["token_accounting"]
    text, ref, audio, video = (ta["text_tokens"], ta["reference_tokens"],
                               ta["audio_tokens"], ta["video_tokens"])
    assert text + ref + audio + video == ta["total_sequence_length"]
    segments = {"text": (0, text), "ref": (text, text + ref),
                "audio": (text + ref, text + ref + audio),
                "video": (text + ref + audio, text + ref + audio + video)}
    per_seg = {"text": 8, "ref": 8, "audio": 4, "video": args.video_warps}
    files = sorted(root.glob("qkv_*.pt"))
    if args.cells:
        files = [f for f in files if any(c + "_" in f.name for c in args.cells)]
    out = []
    for f in files:
        with torch.no_grad():
            r = run_cell(f, segments, per_seg, args.heads, args.device)
        out.append(r)
        s = r["stats"]
        print(r["cell"], flush=True)
        for rule in ("causal", "seeded", "oracle"):
            for T in THRESH:
                print(f"  {rule:7s} T={T:2d} video rows: skip {s[f'{rule}/{T}/video/skip_frac']:.3f} "
                      f"(prefix {s[f'{rule}/{T}/video/skip_frac_prefix']:.3f}, video keys "
                      f"{s[f'{rule}/{T}/video/skip_frac_video']:.3f})  err mean "
                      f"{s[f'{rule}/{T}/video/err_mean']:.2e} max {s[f'{rule}/{T}/video/err_max']:.2e}  "
                      f"mass {s[f'{rule}/{T}/video/mass_mean']:.2e}", flush=True)
        print(f"  u8-P floor, video rows: err mean {s['u8/0/video/err_mean']:.2e} "
              f"max {s['u8/0/video/err_max']:.2e}", flush=True)
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
