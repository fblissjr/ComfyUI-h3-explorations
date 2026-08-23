#!/usr/bin/env python3
"""Does any shipped graph hand Qwen a reference image it will shrink?

Run it with the ComfyUI venv python (`docs/comfy_notes.md`). CPU only.

`bench/measure_qwen_bounds_bite.py` established *where* ComfyUI's Qwen pixel
ceiling starts resizing a reference prepared to a 2048 short edge. It answered
that with synthetic sizes swept along a ratio axis. This script answers the
other half -- whether the graphs this repo actually ships ever reach it -- and
that question is not answerable from a threshold plus arithmetic, because the
size Qwen sees is the size `MiniMaxH3ReferenceFit` produced, not the size on
disk.

So every row is driven, not derived. The graph is walked for real, the file is
opened for its real dimensions, the real `MiniMaxH3ReferenceFit.execute` is
called with the arguments that graph wires, and the real
`process_qwen2vl_images` is called on the result. A row says SHRUNK because the
helper returned a smaller grid, not because a ratio was compared to 3.0625.

**The control, and what its absence would hide.** The expected result is an
empty list, and an empty list is exactly what a detector that cannot fire
returns. Two synthetic references are pushed through the same two stages as
every shipped row:

  detector arm   a 7168x2048 reference goes straight to the Qwen helper and
                 must come back SHRUNK. If it does not, `_grid` is not
                 measuring what this script claims and every green row above
                 it is unreadable.
  fit-path arm   an 896x256 source with `allow_upscale=True` and the repo's
                 Qwen/VAE tower guard deliberately disabled must be fitted UP
                 to a size that then trips. The unsafe setting is intentional:
                 the control has to prove this audit can see a fit-fed failure,
                 while shipped graphs keep `keep_towers_matched=True`.

Both arms must trip or the script exits non-zero, whatever the shipped rows
said.

Discovery is `h3_config.graph_paths(include_bench=True)`, per
`bench/check_graph_discovery.py`: the property being audited is true of any
graph that renders a reference, bench graphs included.

    <comfy-venv>/bin/python bench/audit_shipped_reference_bounds.py

`--input-root PATH` reads reference files from somewhere else. That exists so
this script can be shown red: point it at a root holding one deliberately
over-wide reference and the tripping list must name the graph that loads it.
Never pass it for a real audit -- the answer would then be about a directory
the server does not read.

**What that red run corrected.** The first decoy was 3584x1024 -- 3.5:1, well
past the "3.0625:1" figure this audit was written against -- and it came back
untouched. The ceiling is a pixel count, not a ratio; the ratio only names it
because `measure_qwen_bounds_bite.py` swept ratios with the short edge pinned
at 2048, where the two coincide. A wide reference at a small short edge does
not trip, so any future claim of the form "ratio R is safe" is unsupported
unless it also says at what short edge.

Exit codes: 0 when the audit answered, 1 when it could not -- no reference
priced, or a control that did not fire. A graph that trips is REPORTED, not
refused; this is an audit, and the shipped set is the owner's call.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO.parents[1]))         # ComfyUI root, first
sys.path.insert(1, str(_REPO))

# Native ComfyUI's socket conditioner remains supported for historical and
# bench graphs; shipped graphs use this repo's typed conditioner. This audit
# follows both so the local handling does not make the native gap invisible.
REF_CONSUMERS = ("MiniMaxH3ReferenceToVideo",
                 "MiniMaxH3ReferenceConditioning")
REF_INPUT_PREFIX = "ref_images."

from reference_order import ChainError, resolve_chain_entries  # noqa: E402

# Synthetic arms. The first size is one `measure_qwen_bounds_bite.py` recorded
# as resized; the second is a source small enough that only an upscaling fit
# can carry it over the ceiling.
DETECTOR_ARM = (7168, 2048)
FIT_PATH_ARM = (896, 256)


def _grid(w: int, h: int):
    """(grid_h, grid_w) the Qwen helper returns for a w x h input."""
    import torch
    from comfy.text_encoders.qwen_vl import process_qwen2vl_images
    img = torch.zeros(1, h, w, 3)
    try:
        _, grid = process_qwen2vl_images(img, patch_size=16,
                                         image_mean=[0.5] * 3, image_std=[0.5] * 3)
    finally:
        del img
    return int(grid[0][1]), int(grid[0][2])


def _qwen_sees(w: int, h: int):
    """What the helper hands the encoder, and whether it shrank the input."""
    gh, gw = _grid(w, h)
    out_w, out_h = gw * 16, gh * 16
    return (out_w, out_h), (out_w, out_h) != (w, h)


def _fitted(src_w: int, src_h: int, fit_args: dict | None):
    """Drive the real node, so the mode logic this depends on is executed.

    `fit_args` is None when a graph wires a LoadImage straight into the
    reference input with no fit node between them. Core then applies its own
    downscale-only clamp, which is what `allow_upscale=False` reproduces --
    recorded in the row as `fit: "core clamp"` so the two cases stay legible.
    """
    import torch
    from reference_fit import MiniMaxH3ReferenceFit
    args = {"allow_upscale": False} if fit_args is None else dict(fit_args)
    args.pop("lift_downstream_clamp", None)
    img = torch.zeros(1, src_h, src_w, 3)
    try:
        out = MiniMaxH3ReferenceFit.execute(img, **args)
        tensor = out[0] if isinstance(out, (tuple, list)) else out.result[0]
        return int(tensor.shape[2]), int(tensor.shape[1])
    finally:
        del img


def _trace_to_loader(wf: dict, link: list):
    """Walk back from a reference input to the LoadImage that feeds it.

    Returns (filename, fit_args or None). Follows the first linked input at
    each hop, which is the only shape these graphs use; a node reached with no
    linked inputs and no filename ends the walk and the caller reports it.
    """
    fit_args = None
    node_id = link[0]
    for _ in range(8):
        node = wf.get(node_id)
        if node is None:
            return None, fit_args
        ct = node.get("class_type")
        if ct == "LoadImage":
            return node["inputs"].get("image"), fit_args
        if ct == "MiniMaxH3ReferenceFit" and fit_args is None:
            fit_args = {k: v for k, v in node["inputs"].items()
                        if not isinstance(v, list)}
        linked = [v for v in node.get("inputs", {}).values()
                  if isinstance(v, list)]
        if not linked:
            return None, fit_args
        node_id = linked[0][0]
    return None, fit_args


def _input_dir(override: str | None = None) -> tuple[Path, str]:
    """Where the server this repo runs against actually reads inputs from.

    `folder_paths.get_input_directory()` answers with ComfyUI's *default*,
    which is not where these graphs' files live: the input root is a launch
    argument in the ComfyUI launcher script. Reading the launcher is what makes
    this script agree with the running server instead of with a fresh
    checkout -- the first run of this file resolved zero files and printed a
    clean empty list, which is the exact shape of the answer it was written to
    produce.
    """
    import folder_paths
    if override:
        return Path(override), "--input-root override"
    launcher = _REPO.parents[1] / "start.sh"
    if launcher.is_file():
        for line in launcher.read_text().splitlines():
            line = line.strip()
            if line.startswith("--input-directory") and not line.startswith("#"):
                cand = Path(line.split(None, 1)[1].strip().strip("\\").strip())
                if cand.is_dir():
                    return cand, "launcher --input-directory"
    return Path(folder_paths.get_input_directory()), "folder_paths default"


def _resolve(name: str, root: Path) -> Path | None:
    p = root / name
    return p if p.is_file() else None


def _controls() -> tuple[list[dict], bool]:
    rows = []
    seen, shrunk = _qwen_sees(*DETECTOR_ARM)
    rows.append({"arm": "detector", "source": list(DETECTOR_ARM),
                 "into_qwen": list(DETECTOR_ARM), "qwen_sees": list(seen),
                 "shrunk": shrunk, "must_shrink": True})
    fw, fh = _fitted(*FIT_PATH_ARM, {"allow_upscale": True, "short_edge": 2048,
                                    "keep_towers_matched": False})
    seen2, shrunk2 = _qwen_sees(fw, fh)
    rows.append({"arm": "fit-path", "source": list(FIT_PATH_ARM),
                 "into_qwen": [fw, fh], "qwen_sees": list(seen2),
                 "shrunk": shrunk2, "must_shrink": True})
    return rows, all(r["shrunk"] for r in rows)


def main() -> int:
    try:
        import torch  # noqa: F401
        import comfy.text_encoders.qwen_vl  # noqa: F401
        import folder_paths  # noqa: F401
    except ImportError as exc:
        print(f"ComfyUI is not importable from here ({exc}); run this with the "
              f"ComfyUI venv python (see docs/comfy_notes.md)")
        return 2

    from workflows import h3_config

    override = None
    if "--input-root" in sys.argv:
        override = sys.argv[sys.argv.index("--input-root") + 1]
    in_root, in_how = _input_dir(override)
    print(f"reference files read from {in_how}\n")

    rows, unresolved, chain_errors = [], [], []
    for path in h3_config.graph_paths(_REPO / "workflows", include_bench=True):
        try:
            wf = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(wf, dict):
            continue
        rel = str(path.relative_to(_REPO))
        for node in wf.values():
            if not isinstance(node, dict):
                continue
            if node.get("class_type") not in REF_CONSUMERS:
                continue
            inputs = node.get("inputs", {})
            image_links = []
            if node.get("class_type") == "MiniMaxH3ReferenceConditioning":
                terminal = inputs.get("references")
                if not (isinstance(terminal, list) and len(terminal) == 2):
                    chain_errors.append({"graph": rel,
                                         "why": "typed conditioner has no valid terminal chain link"})
                    continue
                try:
                    entries = resolve_chain_entries(wf, str(terminal[0]))
                except ChainError as exc:
                    chain_errors.append({"graph": rel, "why": str(exc)})
                    continue
                image_links = [
                    (f"references.{append_id}", append["inputs"].get("image"))
                    for append_id, append, kind in entries if kind == "image"
                ]
            else:
                image_links = [
                    (key, val) for key, val in sorted(inputs.items())
                    if key.startswith(REF_INPUT_PREFIX) and isinstance(val, list)
                ]
            for key, val in image_links:
                if not isinstance(val, list):
                    unresolved.append({"graph": rel, "input": key,
                                       "why": "image append has no linked image"})
                    continue
                name, fit_args = _trace_to_loader(wf, val)
                if name is None:
                    unresolved.append({"graph": rel, "input": key,
                                       "why": "no LoadImage on the traced path"})
                    continue
                src = _resolve(name, in_root)
                if src is None:
                    unresolved.append({"graph": rel, "input": key,
                                       "file": name, "why": "file not on disk"})
                    continue
                from PIL import Image
                with Image.open(src) as im:
                    sw, sh = im.size
                fw, fh = _fitted(sw, sh, fit_args)
                seen, shrunk = _qwen_sees(fw, fh)
                rows.append({
                    "graph": rel, "input": key, "file": name,
                    "source": [sw, sh],
                    "source_ratio": round(max(sw, sh) / min(sw, sh), 4),
                    "fit": "core clamp" if fit_args is None else fit_args,
                    "into_qwen": [fw, fh],
                    "into_qwen_ratio": round(max(fw, fh) / min(fw, fh), 4),
                    "qwen_sees": list(seen), "shrunk": shrunk,
                })

    tripping = [r for r in rows if r["shrunk"]]

    print(f"{'graph':<52}{'file':<46}{'source':<12}{'into qwen':<12}"
          f"{'ratio':<8}{'shrunk'}")
    for r in sorted(rows, key=lambda r: -r["into_qwen_ratio"]):
        g = r["graph"].replace("workflows/", "")
        f = r["file"].split("/")[-1]
        src = "{}x{}".format(*r["source"])
        into = "{}x{}".format(*r["into_qwen"])
        print(f"{g[:51]:<52}{f[:45]:<46}{src:<12}{into:<12}"
              f"{r['into_qwen_ratio']:<8}{'SHRUNK' if r['shrunk'] else 'no'}")

    ctrl_rows, ctrl_ok = _controls()
    print("\n--- controls (both must SHRUNK, or the empty list above means "
          "nothing) ---")
    for c in ctrl_rows:
        s_src = "{}x{}".format(*c["source"])
        s_in = "{}x{}".format(*c["into_qwen"])
        s_see = "{}x{}".format(*c["qwen_sees"])
        print(f"  {c['arm']:<12}{s_src:<12}-> into qwen {s_in:<12}"
              f"-> sees {s_see:<12}"
              f"{'SHRUNK' if c['shrunk'] else 'NOT SHRUNK -- CONTROL FAILED'}")

    if unresolved:
        print("\n--- reference inputs this script could not price ---")
        for u in unresolved:
            print(f"  {u['graph']}  {u['input']}  {u.get('file', '')}  {u['why']}")
    if chain_errors:
        print("\n--- typed reference chains this script could not resolve ---")
        for failure in chain_errors:
            print(f"  {failure['graph']}  {failure['why']}")

    widest = max((r["into_qwen_ratio"] for r in rows), default=None)
    print()
    if tripping:
        print(f"  {len(tripping)} reference input(s) are shrunk by the Qwen "
              f"ceiling before the encoder sees them:")
        for r in tripping:
            print(f"    {r['graph']}  {r['input']}  {r['file']}")
    else:
        print("  no shipped reference input is shrunk by the Qwen ceiling.")
        if widest is not None:
            print(f"  the widest reference any graph sends is {widest}:1, "
                  f"and the ceiling first bit at 3.25:1 in "
                  f"bench/results/2026-08-21_qwen_bounds_bite.json")

    record = {
        "question": "does any shipped graph hand Qwen a reference it shrinks?",
        "discovery": "h3_config.graph_paths(include_bench=True)",
        "ref_consumers": list(REF_CONSUMERS),
        "reference_inputs": rows,
        "tripping": [{"graph": r["graph"], "input": r["input"],
                      "file": r["file"]} for r in tripping],
        "widest_ratio_into_qwen": widest,
        "ceiling_is": "a pixel count, not a ratio; see the module docstring",
        "unresolved": unresolved,
        "chain_errors": chain_errors,
        "controls": ctrl_rows,
        "controls_hold": ctrl_ok,
    }
    out = _REPO / "bench" / "results" / "2026-08-21_shipped_reference_bounds.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(_REPO)}")

    if not rows:
        print("\nNOTHING WAS PRICED: no reference input resolved to a file on "
              "disk, so this run answers nothing. An empty tripping list here "
              "is indistinguishable from a clean result and must not be read "
              "as one.")
        return 1
    if chain_errors:
        print("\nCHAIN ERROR: at least one typed reference plan could not be "
              "resolved completely, so this audit refuses a partial answer.")
        return 1
    if not ctrl_ok:
        print("\nCONTROL FAILED: the detector did not fire on a reference "
              "known to trip, so the shipped rows above are not evidence.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
