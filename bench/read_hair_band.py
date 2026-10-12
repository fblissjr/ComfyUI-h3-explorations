#!/usr/bin/env python3
"""A hair band read as figures, frame by frame, on the band's own pixels: for the load that has it and a load of the same frames that does not.

    <python> bench/read_hair_band.py --band BAND_LOAD.built.json --pair PAIR_LOAD.built.json            # the plans alone
    <python> bench/read_hair_band.py --band ... --pair ... --source CLIP \\
        --render-band BAND.mp4[@FIRST] --render-pair PAIR.mp4[@FIRST] [--out FOLDER]                     # and the renders

**What it is for.** A hair band (`bench/masked_job.py::hair_band`) takes a rectangle under a head-and-hair part
out of the clothing keep-out, so that the region's lower edge there holds still from frame to frame and hair
that falls past the original's jaw is not drawn on one frame and given back on the next. Whether it does that
was a thing seen on playback. This prints what can be counted of it, for the band load and for a PAIR, a load
of the same frames without the band, on the same pixels, so the two are like for like.

**Where it is read.** The band on a frame is the keep-out's mask less the cut copy the build wrote: the pixels
the band took out. Both are files the band load's build record names (`hair_band.made_from.keep_out`,
`hair_band.file`); nothing is found by a key. Each load's region is its own capture's (`runs/<load>/region.npz`),
which after a plan is the node's.

**Four figures a frame**, each for the band load and for the pair:

1. THE CAUSE, from the plans, with no render: of the cells that hold band pixels, how many are open
   (regenerated) and how many are not, and how many changed between open and not since the frame before. The
   whole region's cells are printed beside, since that is the count a person makes by hand.
2. DRAWN: the share of the band's pixels whose grey level differs from the source's by more than that frame's
   floor. The floor is taken from the pixels the load left alone on that frame (outside its region), at
   `FLOOR_PERCENTILE`.
3. DRAWN AND UNDRAWN: of the pixels in the band on two consecutive frames, the share drawn on one and not on
   the other. Beside it, the share of the same pixels the SOURCE itself changes by more than the floor between
   the two frames: a head that moves makes pixels switch honestly. All three shares (drawn, switching, the
   source's change) are of the band's pixels, so they can be set beside each other and beside the floor's own
   noise: with the floor at the 99th percentile about a hundredth of untouched pixels read as drawn, and about
   twice that as switching, with nothing drawn at all.
4. THE LOWER EDGE: how far down the band the drawn pixels reach, in pixels from the band's top, the median over
   the band's columns; and its step from the frame before. A weak figure where little is drawn: the lowest of a
   few scattered pixels is the floor's noise, not an edge. The printout says each load's drawn share beside
   `EDGE_WEAK_UNDER` so the edge is not read alone.

Then the medians of each, and the frames with the largest switching and the largest edge step, by source frame
number, so a person can open them on the stacked pair. There is no threshold and no word of verdict.

**What the pair differs in besides the band** is printed first, from the two records' loads and texts: a figure
that differs between the two renders carries every one of those differences, not the band alone.

**What it cannot see.** Whether what is drawn in the band is hair, a strap or an apron: a difference from the
source is not "hair". A flicker smaller than the floor. Hair that falls past the band, or past the margin.
Anything the eye takes from motion that a frame-to-frame count of pixels does not hold.

No model, no card. Needs ffmpeg for the mask videos, the renders and the source.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib.frames import FIT, read_mask, stream  # noqa: E402

#: A pixel the load left alone differs from the source by encode noise. A band pixel counts as drawn when it
#: differs by more than this percentile of those pixels on the same frame. Reasoned, a starting value, measured
#: on nothing: at 99 about one untouched pixel in a hundred would read as drawn, which is the figure's own noise
#: and is printed (`floor`) so a reader can set a figure against it.
FLOOR_PERCENTILE = 99.0
#: Under about this share of the band drawn, the lower edge (figure 4) is the floor's noise and not an edge. Seen
#: on ONE run, the tool's first (2026-10-12, a 16-frame load and its pair): the pair, whose drawn share was the
#: floor's own noise (a median of 0.018), stepped its edge by the same median, 3 px a frame, as the band load at
#: 0.057. The lead's ruling of that evening put the line at about a tenth. A starting value; it flags nothing and
#: changes no figure, it is only printed beside the edge.
EDGE_WEAK_UNDER = 0.1
#: Keys of a load that are the band itself, or only name the load: never a "difference besides the band".
NOT_A_DIFFERENCE = ("name", "hair_band", "hair_band_cuts")


def cells_any(mask: np.ndarray, cell: int) -> np.ndarray:
    """[n, h, w] of bool to [n, h / cell, w / cell]: true where any pixel of the cell is."""
    n, h, w = mask.shape
    return mask.reshape(n, h // cell, cell, w // cell, cell).any(axis=(2, 4))


def cells_up(cells: np.ndarray, cell: int) -> np.ndarray:
    return np.repeat(np.repeat(cells, cell, axis=-2), cell, axis=-1)


def differences(band: dict, pair: dict) -> list[str]:
    """What the two build records' loads and texts differ in besides the band, a line each; empty when nothing."""
    a, b = band.get("load") or {}, pair.get("load") or {}
    out = [f"{key}: {a.get(key)!r} in the band load, {b.get(key)!r} in the pair" for key in sorted(set(a) | set(b))
           if key not in NOT_A_DIFFERENCE and a.get(key) != b.get(key)]
    ta, tb = (band.get("text") or {}), (pair.get("text") or {})
    if ta.get("sha256") != tb.get("sha256"):
        out.append(f"the text itself: {ta.get('file')!r} in the band load, {tb.get('file')!r} in the pair (the hashes differ)")
    return out


def cause(band_cells: np.ndarray, region: np.ndarray) -> list[dict]:
    """Figure 1 for one load: per frame, the band's cells that are open, that are not, and that changed since the frame before."""
    rows = []
    for n in range(len(band_cells)):
        open_, both = band_cells[n] & region[n], None
        changed = None
        if n:
            both = band_cells[n] & band_cells[n - 1]
            changed = int((both & (region[n] != region[n - 1])).sum())
        rows.append({"region_cells": int(region[n].sum()), "band_cells": int(band_cells[n].sum()), "open": int(open_.sum()),
                     "not_open": int((band_cells[n] & ~region[n]).sum()), "changed": changed})
    return rows


def floor_of(diff: np.ndarray, untouched: np.ndarray) -> float | None:
    """The grey-level difference a pixel the load left alone reaches at `FLOOR_PERCENTILE`; none when it left none alone."""
    return float(np.percentile(diff[untouched], FLOOR_PERCENTILE)) if untouched.any() else None


def edge_depth(drawn: np.ndarray, band: np.ndarray) -> float | None:
    """How far down the band the drawn pixels reach: per column, from the band's top pixel to the lowest drawn one
    (0 when none is drawn), the median over the columns that hold band pixels. None for a frame with no band."""
    cols = np.nonzero(band.any(axis=0))[0]
    if not len(cols):
        return None
    depths = []
    for x in cols:
        top = int(np.argmax(band[:, x]))
        rows = np.nonzero(drawn[:, x])[0]
        depths.append(0 if not len(rows) else int(rows[-1]) - top + 1)
    return float(np.median(depths))


def pictures(band: np.ndarray, region_px: np.ndarray, source: np.ndarray, render: np.ndarray) -> list[dict]:
    """Figures 2 to 4 for one render: `band` and `region_px` [n, h, w] of bool, `source` and `render` [n, h, w] grey."""
    rows, before = [], None
    for n in range(len(band)):
        diff = np.abs(render[n].astype(np.int16) - source[n].astype(np.int16))
        floor = floor_of(diff, ~region_px[n])
        row = {"floor": None if floor is None else round(floor, 2), "drawn_share": None, "switch_share": None, "source_moved_share": None,
               "edge_px": None, "edge_step_px": None}
        if floor is not None and band[n].any():
            drawn = band[n] & (diff > floor)
            row["drawn_share"] = round(float(drawn.sum()) / float(band[n].sum()), 4)
            row["edge_px"] = edge_depth(drawn, band[n])
            if before is not None:
                both = band[n] & before["band"]
                if both.any():
                    row["switch_share"] = round(float(((drawn ^ before["drawn"]) & both).sum()) / float(both.sum()), 4)
                    moved = np.abs(source[n].astype(np.int16) - source[n - 1].astype(np.int16)) > floor
                    row["source_moved_share"] = round(float((moved & both).sum()) / float(both.sum()), 4)
                if row["edge_px"] is not None and before["edge"] is not None:
                    row["edge_step_px"] = abs(row["edge_px"] - before["edge"])
            before = {"band": band[n], "drawn": drawn, "edge": row["edge_px"]}
        else:
            before = None
        rows.append(row)
    return rows


def load_of(record_path: str) -> dict:
    """What this tool needs of one load's build record: its frames, its capture, and its region there."""
    rec = json.loads(Path(record_path).read_text())
    load, capture = rec.get("load") or {}, Path(str(rec.get("capture")))
    name, first, frames = str(load.get("name") or ""), int(load["first"]), int(load["frames"])
    if not name:
        raise SystemExit(f"{record_path}: the build record's load has no name")
    m = json.loads((capture / "manifest.json").read_text())
    if not any(r["name"] == name for r in m["runs"]):
        raise SystemExit(f"{record_path}: its capture {capture} holds no run named {name!r}")
    z = np.load(capture / "runs" / name / "region.npz")
    lead = first - int(m["first_frame"])
    if lead < 0 or lead + frames > len(z["region"]):
        raise SystemExit(f"{record_path}: the load is source frames {first}-{first + frames - 1} and its capture holds "
                         f"{m['first_frame']}-{m['first_frame'] + len(z['region']) - 1}")
    run = next(r for r in m["runs"] if r["name"] == name)
    return {"record": rec, "path": record_path, "name": name, "first": first, "frames": frames, "size": tuple(m["size"]), "capture": str(capture),
            "region": z["region"][lead:lead + frames], "region_is": "the node's plan" if run.get("carried_from_the_node") else
            "worked out by the capture from the masks, not the node's"}


def band_of(load: dict) -> np.ndarray:
    """The band's pixels on each frame of the load: the keep-out less the cut copy, from the two files the record names."""
    hb = load["record"].get("hair_band")
    if not isinstance(hb, dict) or not hb.get("file"):
        raise SystemExit(f"{load['path']}: this load's build record has no hair band (`hair_band` is {hb!r}): give the band load as --band")
    keep, _ = read_mask(hb["made_from"]["keep_out"], load["size"], int(hb["made_from"]["keep_outs_first_frame_is_source_frame"]), load["first"], load["frames"])
    cut, covered = read_mask(hb["file"], load["size"], int(hb["its_first_frame_is_source_frame"]), load["first"], load["frames"])
    if not covered.all():
        raise SystemExit(f"{hb['file']} does not cover every frame of the load")
    return keep & ~cut


def render_arg(text: str, first: int) -> tuple[str, int]:
    path, at, begins = text.rpartition("@") if text.rpartition("@")[2].isdigit() else (text, "", "")
    return path, int(begins) if at else first


def grey(path: str, size: tuple[int, int], at: int, first: int, frames: int, vf: str) -> np.ndarray:
    if first < at:
        raise SystemExit(f"{path} begins on source frame {at}, after the load's first frame {first}")
    got = list(stream(path, size, first - at, frames, vf=vf))
    if len(got) != frames:
        raise SystemExit(f"{path}: {len(got)} of the load's {frames} frames could be read from source frame {first}")
    return np.stack(got)


def read(a: argparse.Namespace) -> dict:
    band, pair = load_of(a.band), load_of(a.pair)
    if (band["first"], band["frames"], band["size"]) != (pair["first"], pair["frames"], pair["size"]):
        raise SystemExit(f"the band load is source frames {band['first']}-{band['first'] + band['frames'] - 1} at {band['size']} and the pair "
                         f"{pair['first']}-{pair['first'] + pair['frames'] - 1} at {pair['size']}: a pair is a load of the same frames")
    px = band_of(band)
    cell = band["size"][0] // band["region"].shape[2]
    cells = cells_any(px, cell)
    out = {"band": band["name"], "pair": pair["name"], "first": band["first"], "frames": band["frames"], "cell_px": cell,
           "regions_are": {band["name"]: band["region_is"], pair["name"]: pair["region_is"]},
           "the_pair_differs_besides_the_band_in": differences(band["record"], pair["record"]),
           "cause": {x["name"]: cause(cells, x["region"]) for x in (band, pair)}, "band_px": [int(v) for v in px.sum(axis=(1, 2))]}
    renders = [a.render_band, a.render_pair, a.source]
    if any(renders) and not all(renders):
        raise SystemExit("the renders are read together: give --render-band, --render-pair and --source, or none of them")
    if all(renders):
        w, h = band["size"]
        source = grey(a.source, (w, h), 0, band["first"], band["frames"], FIT.format(w=w, h=h))
        out["pictures"] = {}
        for x, given in ((band, a.render_band), (pair, a.render_pair)):
            path, at = render_arg(given, x["first"])
            out["pictures"][x["name"]] = pictures(px, cells_up(x["region"], cell), source, grey(path, (w, h), at, x["first"], x["frames"], f"scale={w}:{h}"))
        out["floor_percentile"] = FLOOR_PERCENTILE
    return out


def table(out: dict) -> list[dict]:
    """One row a frame, the band load's and the pair's figures side by side, keyed for a csv."""
    b, p = out["band"], out["pair"]
    rows = []
    for n in range(out["frames"]):
        row = {"source_frame": out["first"] + n, "band_px": out["band_px"][n]}
        for tag, name in (("band", b), ("pair", p)):
            row.update({f"{k}__{tag}": v for k, v in out["cause"][name][n].items()})
            if "pictures" in out:
                row.update({f"{k}__{tag}": v for k, v in out["pictures"][name][n].items()})
        rows.append(row)
    return rows


def say(out: dict) -> None:
    b, p = out["band"], out["pair"]
    print(f"the hair band of {b}, against {p}: source frames {out['first']}-{out['first'] + out['frames'] - 1}; a cell is {out['cell_px']} px")
    for name, how in out["regions_are"].items():
        print(f"  the region of {name}: {how}")
    besides = out["the_pair_differs_besides_the_band_in"]
    print("  besides the band, the two records' loads and texts differ in " + ("nothing" if not besides else f"{len(besides)} thing(s), and every "
          "figure that differs between the two renders carries them too:"))
    for line in besides:
        print(f"    {line}")
    rows = table(out)
    num = lambda v: "   -" if v is None else f"{v:4d}" if isinstance(v, int) else f"{v:6.3f}"  # noqa: E731
    print("\nTHE CAUSE, from the plans (cells): source frame | the whole region, band/pair | cells holding band pixels | open, band/pair | "
          "not open, band/pair | changed since the frame before, band/pair")
    for r in rows:
        print(f"  {r['source_frame']} | {num(r['region_cells__band'])} /{num(r['region_cells__pair'])} | {num(r['band_cells__band'])} | "
              f"{num(r['open__band'])} /{num(r['open__pair'])} | {num(r['not_open__band'])} /{num(r['not_open__pair'])} | "
              f"{num(r['changed__band'])} /{num(r['changed__pair'])}")
    if "pictures" in out:
        print(f"\nTHE RENDERS, on the band's pixels: source frame | floor (grey levels, the {out['floor_percentile']:g}th percentile where the load "
              "left the source), band/pair | drawn share, band/pair | drawn on one of two frames and not the other, band/pair | the source's own "
              "change there, by the band's floor | lower edge px, band/pair | its step, band/pair")
        for r in rows:
            print(f"  {r['source_frame']} | {num(r['floor__band'])} /{num(r['floor__pair'])} | {num(r['drawn_share__band'])} /{num(r['drawn_share__pair'])} | "
                  f"{num(r['switch_share__band'])} /{num(r['switch_share__pair'])} | {num(r['source_moved_share__band'])} | "
                  f"{num(r['edge_px__band'])} /{num(r['edge_px__pair'])} | {num(r['edge_step_px__band'])} /{num(r['edge_step_px__pair'])}")
    print("\nover the load, the median (and the most), band load / pair:")
    keys = ["open", "not_open", "changed"] + (["drawn_share", "switch_share", "edge_px", "edge_step_px"] if "pictures" in out else [])
    for key in keys:
        got = {tag: [r[f"{key}__{tag}"] for r in rows if r[f"{key}__{tag}"] is not None] for tag in ("band", "pair")}
        told = {tag: "nothing to read" if not v else f"{float(np.median(v)):.3f} ({float(np.max(v)):.3f})" for tag, v in got.items()}
        print(f"  {key}: {told['band']} / {told['pair']}")
    if "pictures" in out:
        got = [r["source_moved_share__band"] for r in rows if r["source_moved_share__band"] is not None]
        print("  the source's own change on the band's pixels: " + ("nothing to read" if not got else f"{float(np.median(got)):.3f} ({float(np.max(got)):.3f})"))
        for key, what in (("switch_share", "the most drawn and undrawn"), ("edge_step_px", "the largest step of the lower edge")):
            for tag, name in (("band", b), ("pair", p)):
                got = [(r[f"{key}__{tag}"], r["source_frame"]) for r in rows if r[f"{key}__{tag}"] is not None]
                if got:
                    top = sorted(got, reverse=True)[:3]
                    print(f"  {what}, {name}: " + ", ".join(f"frame {f} ({v:.3f})" for v, f in top))
    if "pictures" in out:
        shares = {name: [r["drawn_share"] for r in out["pictures"][name] if r["drawn_share"] is not None] for name in (b, p)}
        print(f"\nthe lower edge is the lowest of the drawn pixels in each column: under a drawn share of about {EDGE_WEAK_UNDER:g} it is the "
              "floor's noise and not an edge (seen on one run). The drawn share here, at the median: "
              + "; ".join(f"{name} {float(np.median(v)):.3f}" if v else f"{name} nothing to read" for name, v in shares.items()))
        noise = (100.0 - out["floor_percentile"]) / 100.0
        print(f"\nthe floor's own noise: about {noise:g} of pixels a load left alone read as drawn, and about {2 * noise:g} as switching, with nothing "
              "drawn. A drawn share near that is nothing drawn, and its switching and edge are the noise's")
    print("\nit cannot see: whether what is drawn in the band is hair; a flicker under the floor; hair past the band or the margin")


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--band", required=True, metavar="BUILT.json", help="the build record of the load that has the hair band")
    ap.add_argument("--pair", required=True, metavar="BUILT.json", help="the build record of a load of the same frames without it")
    ap.add_argument("--render-band", metavar="R.mp4[@FIRST]", help="the band load's render; @FIRST when its frame 0 is not the load's first frame")
    ap.add_argument("--render-pair", metavar="R.mp4[@FIRST]")
    ap.add_argument("--source", help="the clip both loads were made from")
    ap.add_argument("--out", metavar="FOLDER", help="also write the table as <band>_against_<pair>.csv and .json there (under data/, never the share)")
    a = ap.parse_args()
    out = read(a)
    say(out)
    if a.out:
        folder = Path(a.out)
        folder.mkdir(parents=True, exist_ok=True)
        stem = folder / f"hair_band_{out['band']}_against_{out['pair']}"
        rows = table(out)
        with open(stem.with_suffix(".csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        stem.with_suffix(".json").write_text(json.dumps({**out, "inputs": {k: getattr(a, k) for k in ("band", "pair", "render_band", "render_pair", "source")}}, indent=1) + "\n")
        print(f"wrote {stem.with_suffix('.csv')} and {stem.with_suffix('.json').name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
