#!/usr/bin/env python3
"""MiniMax-H3-Image: the release-to-ComfyUI layout, the size rule and the shipped graphs' chain.

No model file and no card. What each case defends, i.e. what breaks if it is deleted:

- `qkv order is core's split` -- `bench/h3_image/release.py::comfy_qkv_order` turns the release's
  per-head `[q|k|v]` rows into the `[q_all|k_all|v_all]` rows core splits. Checked against core's
  own split on a random projection: q, k and v taken from the reordered weight the way
  `Attention.forward` takes them equal the per-head rows of the original. Control: the weight left
  in release order must NOT pass.
- `stacked heads read back whole` -- `stack_heads` writes head 0 whole and later heads as offsets;
  read through core's `_pdd_head` over one grid interval, each head comes back as it was. Control:
  whole heads stacked without the offset must NOT read back.
- `the writer's file opens and holds its tensors` -- `write_safetensors` against the safetensors
  library, bf16 included, with a planned dtype that disagrees refused.
- `sizes are the model card's` -- `h3_config.h3_image_size` at 16 megapixels gives the card's size
  table (transcribed below by hand from the release's README), for each of its ratios.
- `the node's shapes are the vendor's ratios` -- `H3_IMAGE_ASPECTS` is the `aspect_ratio` enum of
  the vendor's `request_params.json`, read from the vendored file.
- `a picture gives its own shape` and `no picture is refused` -- `resolve_size` on
  'same as the picture'.
- `the shipped graphs` -- each `workflows/image/h3_image*_api.json` and its probes hands the
  sampler the grid in `H3_IMAGE_MANUAL_SIGMAS` through `ManualSigmas`, has no negative prompt or
  scheduler node, and an edit wires its picture into both the conditioning node and the size node
  with nothing between the loader and either.

`--release <MiniMax-H3-Image folder>` adds: the grid constant equals the release's
`pdd_config.json`. Without it that comparison is reported as skipped, not passed.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_h3_image.py [--release <folder>]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "workflows"))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

import h3_config  # noqa: E402
from h3_image import release as rel  # noqa: E402

# The model card's "Sizes" table, typed from the release's README.md (width, height).
CARD_SIZES = {
    "1:1": (4096, 4096), "4:3": (4736, 3552), "3:4": (3552, 4736), "3:2": (5024, 3360),
    "2:3": (3360, 5024), "16:9": (5472, 3072), "9:16": (3072, 5472), "21:9": (6272, 2688),
}

fails = 0


def report(ok: bool, name: str, detail: str = "") -> None:
    global fails
    fails += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail and not ok else ""))


def core_split(weight: torch.Tensor, x: torch.Tensor, heads: int, head_dim: int):
    """q, k, v as core takes them from a fused projection: three equal runs, each [heads, head_dim]."""
    q, k, v = (x @ weight.T).chunk(3, dim=-1)
    return tuple(t.view(heads, head_dim) for t in (q, k, v))


def qkv_order() -> None:
    heads, head_dim, hidden = 4, 6, 10
    g = torch.Generator().manual_seed(0)
    release_w = torch.randn(heads * 3 * head_dim, hidden, generator=g)
    x = torch.randn(hidden, generator=g)
    per_head = (x @ release_w.T).view(heads, 3, head_dim)          # the release's own layout
    want = tuple(per_head[:, i] for i in range(3))
    got = core_split(rel.comfy_qkv_order(release_w, heads, head_dim), x, heads, head_dim)
    report(all(torch.equal(a, b) for a, b in zip(got, want)), "qkv order is core's split")
    left = core_split(release_w, x, heads, head_dim)
    report(not all(torch.equal(a, b) for a, b in zip(left, want)),
           "control: a weight left in release order does not pass")


def read_head(bank: torch.Tensor, steps: int, k: int) -> torch.Tensor:
    """Head `k` as core's `_pdd_head` composes it over the single interval k -> k+1."""
    rows = bank.reshape(steps, -1, bank.shape[-1])
    return rows[0] if k == 0 else rows[0] + rows[k]


def stacked_heads() -> None:
    steps, out, hidden = 6, 5, 7
    heads = torch.randn(steps, out, hidden, generator=torch.Generator().manual_seed(1))
    bank = rel.stack_heads(heads)
    report(bank.shape == (steps * out, hidden) and
           all(torch.allclose(read_head(bank, steps, k), heads[k], atol=1e-6) for k in range(steps)),
           "stacked heads read back whole")
    plain = heads.reshape(steps * out, hidden)
    report(not all(torch.allclose(read_head(plain, steps, k), heads[k], atol=1e-6) for k in range(steps)),
           "control: whole heads stacked without the offset do not read back")


def writer() -> None:
    tensors = {"a.weight": torch.randn(3, 4).to(torch.bfloat16), "b.bias": torch.arange(5, dtype=torch.float32),
               "c": torch.tensor([[1, -2]], dtype=torch.int8)}
    entries = [(k, rel._ST_DTYPE[v.dtype], tuple(v.shape)) for k, v in tensors.items()]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.safetensors"
        size = rel.write_safetensors(path, entries, tensors.__getitem__, {"config": "{}"})
        with safe_open(str(path), "pt") as f:
            same = all(torch.equal(f.get_tensor(k), v) and f.get_tensor(k).dtype == v.dtype for k, v in tensors.items())
            meta = f.metadata() == {"config": "{}"}
        report(same and meta and size == path.stat().st_size, "the writer's file opens and holds its tensors")
        wrong = [("a.weight", "F32", (3, 4))]
        try:
            rel.write_safetensors(Path(tmp) / "y.safetensors", wrong, tensors.__getitem__, {})
            report(False, "control: a planned dtype that disagrees is refused")
        except ValueError:
            report(True, "control: a planned dtype that disagrees is refused")


def sizes() -> None:
    got = {r: h3_config.h3_image_size(16.0, *(int(v) for v in r.split(":"))) for r in CARD_SIZES}
    report(got == CARD_SIZES, "sizes are the model card's", f"{got}")
    params = json.loads((REPO / "vendor_guides" / "h3_image" / "prompt" / "t2i" / "request_params.json").read_text())
    enum = params["response_format"]["json_schema"]["schema"]["properties"]["aspect_ratio"]["enum"]
    report(list(h3_config.H3_IMAGE_ASPECTS) == enum, "the node's shapes are the vendor's ratios",
           f"{h3_config.H3_IMAGE_ASPECTS} against {enum}")
    sys.path.insert(0, str(REPO))
    try:
        from h3_image_nodes import resolve_size
    except Exception as exc:  # comfy_api not importable outside ComfyUI's tree
        report(False, "the size node imports", repr(exc))
        return
    report(resolve_size(16.0, h3_config.H3_IMAGE_ASPECT_OF_PICTURE, (1500, 1000)) == CARD_SIZES["3:2"],
           "a picture gives its own shape")
    try:
        resolve_size(16.0, h3_config.H3_IMAGE_ASPECT_OF_PICTURE, None)
        report(False, "no picture is refused")
    except ValueError:
        report(True, "no picture is refused")


def graphs() -> None:
    folder = REPO / "workflows" / "image"
    paths = sorted(folder.glob("h3_image_*_api.json")) + sorted(folder.glob("h3_probe_h3_image_*_api.json"))
    if not paths:
        report(False, "the shipped graphs", "none found under workflows/image/")
        return
    for path in paths:
        g = json.loads(path.read_text())
        kinds = {n["class_type"] for n in g.values()}
        sig = [n for n in g.values() if n["class_type"] == "ManualSigmas"]
        ok = len(sig) == 1 and sig[0]["inputs"]["sigmas"] == h3_config.H3_IMAGE_MANUAL_SIGMAS
        ok &= not kinds & {"KSampler", "BasicScheduler", "CFGGuider", "ConditioningZeroOut", "CLIPTextEncode"}
        cond = next(n for n in g.values() if n["class_type"] == "MiniMaxH3ImageConditioning")["inputs"]
        size = next(n for n in g.values() if n["class_type"] == "MiniMaxH3ImageSize")["inputs"]
        ok &= isinstance(cond["width"], list) and isinstance(cond["height"], list)
        pictures = [v for k, v in cond.items() if k.startswith("images.")]
        if pictures:
            ok &= all(g[src[0]]["class_type"] == "LoadImage" for src in pictures)
            ok &= size.get("picture") == pictures[0] and size["shape"] == h3_config.H3_IMAGE_ASPECT_OF_PICTURE
        else:
            ok &= size["shape"] in h3_config.H3_IMAGE_ASPECTS
        report(bool(ok), f"the shipped graph {path.name}")


def release_grid(folder: Path | None) -> None:
    if folder is None:
        print("  skip  the grid constant against the release's pdd_config.json (no --release)")
        return
    meta = json.loads(rel.metadata(json.loads((folder / "pdd_config.json").read_text()))["config"])
    ours = [float(s) for s in h3_config.H3_IMAGE_MANUAL_SIGMAS.split(",")]
    report(ours == meta["transformer"]["pdd_sigmas"], "the grid constant is the release's pdd_config.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--release", type=Path)
    args = ap.parse_args()
    qkv_order()
    stacked_heads()
    writer()
    sizes()
    graphs()
    release_grid(args.release)
    print("")
    print(f"  {fails} failure(s)" if fails else "  ok    MiniMax-H3-Image layout, sizes and graphs hold")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
