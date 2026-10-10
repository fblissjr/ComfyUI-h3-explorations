#!/usr/bin/env python3
"""Does Comfy-Org/ComfyUI issue 16604 reproduce on this checkout's PackedLayout?

The report: `MiniMaxH3ReferenceToVideo` fails in `MiniMaxH3Model._forward` at
`all_video_rows[~img_update] = cond_video_rows` with "shape mismatch: value
tensor of shape [N_ref, 96] cannot be broadcast to indexing result of shape
[N_video, 96]" once the packed sequence passes about 16384 rows, and suspects an
index or buffer limit, or a layout reused after the references changed.

This builds the layout the way `MiniMaxH3Model.extra_conds` does
(`comfy/model_base.py`, the `PackedLayout(...)` call), builds the condition rows
with core's own `_cond_video_rows`, and runs the same mask assembly as
`_forward` on the CPU. Per case it checks that the count of rows `~img_update`
selects equals the count of condition rows, and that the assembly puts each row
where the layout says. Cases are the report's table, one that is much larger
than any of them, several references, and keyframes plus a reference.

Two controls make the check able to fail: a layout built for one reference and
run against another reference's latent (the "stale layout" hypothesis, which
must raise the report's message), and the report's own 405-row / 14985-row
shape when `img_update` is inverted: `~img_update` then selects exactly the video rows.

CPU only, no card, nothing in core is patched or imported beyond the model file.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/probe_ref_rows_16604.py \\
        --record data/results/<date>_ref_rows_16604.json
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import torch

COMFY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMFY_ROOT))

import comfy.cli_args  # noqa: E402

comfy.cli_args.args.cpu = True  # no CUDA context; the sibling checks do the same
from comfy.ldm.minimax import model as mm  # noqa: E402

PATCH = (1, 2, 2)
LATENT_C = 24
VAE_STRIDE = 16  # pixels per latent cell, as MiniMaxH3ReferenceToVideo's th // 16
FAKE = types.SimpleNamespace(patch_size=PATCH)


def video_latent_t(frames: int) -> int:
    # the report's formula: ((fc - 5) // 17) * 5 + 2
    return ((frames - 5) // 17) * 5 + 2


def ref_image(h_px: int, w_px: int) -> dict:
    lh, lw = h_px // VAE_STRIDE, w_px // VAE_STRIDE
    z = torch.randn(1, LATENT_C, 1, lh, lw)
    return {"kind": "image", "latent_h": lh, "latent_w": lw, "latent": z}


def ref_video(h_px: int, w_px: int, frames: int) -> dict:
    lh, lw = h_px // VAE_STRIDE, w_px // VAE_STRIDE
    lt = video_latent_t(frames)
    z = torch.randn(1, LATENT_C, lt, lh, lw)
    return {"kind": "video", "latent_t": lt, "latent_h": lh, "latent_w": lw, "ref_audio_t": 0, "latent": z}


def keyframe(h_px: int, w_px: int, index: int) -> dict:
    lh, lw = h_px // VAE_STRIDE, w_px // VAE_STRIDE
    return {"resolved_frame_index": index, "latent": torch.randn(1, LATENT_C, 1, lh, lw)}


def run_case(name: str, w_px: int, h_px: int, frames: int, refs=None, keyframes=None, layout_refs=None,
             text_len: int = 512, invert_video: bool = False) -> dict:
    """One case; `layout_refs` builds the layout from different refs than the latents come from."""
    lat_t = video_latent_t(frames)
    lat_h, lat_w = h_px // VAE_STRIDE, w_px // VAE_STRIDE
    audio_t = round(frames / 24 * 40)
    # extra_conds: the layout and the cond latents come from the same refs and keyframes
    layout = mm.PackedLayout(text_len, lat_t, lat_h, lat_w, audio_t,
                             keyframes=keyframes, refs=layout_refs if layout_refs is not None else refs)
    payload = {"cond_video_latents": [k["latent"] for k in (keyframes or []) if k.get("latent") is not None]
               + [r["latent"] for r in (refs or []) if "latent" in r], "seed": 0, "visual_cond_noise_aug": 1.0}
    video_x = torch.randn(1, LATENT_C, lat_t, lat_h, lat_w)
    video_rows = mm.patchify_video(video_x, PATCH)
    cond_rows = mm.MiniMaxH3Model._cond_video_rows(FAKE, payload, "cpu")

    img_update = layout.img_update
    if invert_video:
        img_update = ~img_update  # ref rows True, video rows False: what the report's counts imply
    n_false = int((~img_update).sum())
    out = {"case": name, "canvas": f"{w_px}x{h_px}", "frames": frames, "seq_len": layout.seq_len,
           "video_rows": int(video_rows.shape[0]), "cond_rows": None if cond_rows is None else int(cond_rows.shape[0]),
           "rows_selected_by_not_img_update": n_false, "img_update_len": int(img_update.shape[0])}
    try:
        all_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=torch.float32)
        all_rows[~img_update] = cond_rows
        all_rows[img_update] = video_rows
    except RuntimeError as e:
        out["error"] = str(e)
        out["ok"] = False
        return out
    # each cond row landed where the layout says: the False rows, in order
    out["ok"] = bool(torch.equal(all_rows[~img_update], cond_rows) and torch.equal(all_rows[img_update], video_rows))
    return out


def cases() -> list[dict]:
    r = []
    # the report's table: one reference image at the generation canvas, ref_image_size=match
    for label, w, h, f in [("107f 864x480", 864, 480, 107), ("124f 864x480", 864, 480, 124),
                           ("243f 864x480", 864, 480, 243), ("107f 1280x736", 1280, 736, 107)]:
        r.append(run_case("report: " + label, w, h, f, refs=[ref_image(h, w)]))
    # much larger than the report, and the owner's canvases at 345 frames (dense packed sequence)
    r.append(run_case("1344x768 345f, one image ref", 1344, 768, 345, refs=[ref_image(768, 1344)]))
    r.append(run_case("1344x768 345f, three image refs", 1344, 768, 345,
                      refs=[ref_image(768, 1344), ref_image(512, 512), ref_image(1024, 640)]))
    r.append(run_case("1152x768 345f, image + video ref", 1152, 768, 345,
                      refs=[ref_image(768, 1152), ref_video(480, 864, 53)]))
    r.append(run_case("864x480 124f, keyframe + image ref", 864, 480, 124,
                      keyframes=[keyframe(480, 864, 0)], refs=[ref_image(480, 864)]))
    # controls: the check must be able to fail
    r.append(run_case("control: stale layout (built for a 512x512 ref, run with a 864x480 ref)", 864, 480, 124,
                      refs=[ref_image(480, 864)], layout_refs=[ref_image(512, 512)]))
    r.append(run_case("control: img_update inverted (selects the whole video segment, as the report)", 864, 480, 124,
                      refs=[ref_image(480, 864)], invert_video=True))
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--record", type=Path, help="write the case table as JSON here")
    args = ap.parse_args()
    rows = cases()
    for c in rows:
        status = "OK  " if c["ok"] else "FAIL"
        print(f"{status} {c['case']:<78} seq={c['seq_len']:>7} video={c['video_rows']:>7} cond={c['cond_rows']} "
              f"selected={c['rows_selected_by_not_img_update']}" + (f"  {c['error']}" if "error" in c else ""))
    if args.record:
        args.record.write_text(json.dumps({"torch": torch.__version__, "cases": rows}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
