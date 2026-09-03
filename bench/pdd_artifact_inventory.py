#!/usr/bin/env python3
"""The PDD artifact inventory, written into `docs/pdd_artifacts.md`'s
generated region from the files themselves.

That page is the master record of every PDD weight file on this box: what
each one is, when it was made, by which converter version and commit, what
changed between versions, and which file to load on which checkpoint. The
parts a file knows about itself are GENERATED here -- converter version,
conversion date and commit (`h3_pdd_converted_on`, `h3_pdd_converter_commit`,
stamped since 2026-09-03; earlier files fall back to the file's mtime, marked
as such), backbone kind, adaln form, which checkpoint the probe was cut from,
what it loads on -- plus two facts the files cannot know: which constant in
`workflows/h3_config.py` names them, and whether they sit in `pdd_archive/`.
The parts that are decisions and history -- the glossary, the decision table,
the converter and artifact changelogs -- are written by hand in the same page,
outside the markers, and this script never touches them.

Status, derived:

  current    converted by this repo, named by an `h3_config` `*_LORA` constant
  current, unwired
             converted by this repo, in the loras folder, named by no constant
  archived   under `pdd_archive/`: superseded, kept for the dated records and
             saved graphs that name it; never wire a new graph to it
  reference  not converted by this repo: the alibaba-pai source, or Kijai's
             conversion, read by `bench/compare_pdd_conversions.py` as a control

`--check` regenerates the region in memory and exits 1 if the committed page
differs there, the `bench/build_prompt_catalogue.py --check` shape. It needs
the files on disk and says so when they are not: a missing loras folder is
exit 2, not a pass.

    uv run --active --no-sync python bench/pdd_artifact_inventory.py
    uv run --active --no-sync python bench/pdd_artifact_inventory.py --check
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "workflows"))
import h3_config  # noqa: E402

LORAS = HERE.parents[2] / "models" / "loras" / "h3"
ARCHIVE = LORAS / "pdd_archive"
OUT = REPO / "docs" / "pdd_artifacts.md"
PATTERN = re.compile(r"pdd|Acc-8Step", re.IGNORECASE)
BEGIN = "<!-- BEGIN GENERATED: bench/pdd_artifact_inventory.py -->"
END = "<!-- END GENERATED -->"


def _named_in_config() -> dict[str, str]:
    """{basename: constant} for every `*_LORA` constant naming a PDD file."""
    out = {}
    for name in dir(h3_config):
        if name.endswith("_LORA"):
            value = getattr(h3_config, name)
            if isinstance(value, str) and PATTERN.search(value):
                out[Path(value).name] = name
    return out


def _made_on(path: Path, meta: dict) -> str:
    stamped = meta.get("h3_pdd_converted_on")
    if stamped:
        return stamped
    # The link target's mtime, for files that predate the stamp or were not
    # made here. A copy resets it, which is why the stamp exists.
    ts = path.resolve().stat().st_mtime
    return _dt.date.fromtimestamp(ts).isoformat() + " (mtime)"


def inventory() -> list[dict]:
    from safetensors import safe_open
    named = _named_in_config()
    rows = []
    for folder in (LORAS, ARCHIVE):
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.safetensors")):
            if not PATTERN.search(path.name):
                continue
            with safe_open(path, framework="pt") as f:
                meta = f.metadata() or {}
                keys = list(f.keys())
            ours = "h3_pdd_converter_version" in meta
            adaln = loads_on = "-"
            if folder == ARCHIVE:
                status = "archived"
            elif not ours:
                status = ("reference (alibaba-pai source)"
                          if any(k.startswith("transformer_blocks.") for k in keys)
                          else "reference (Kijai's conversion)")
            elif path.name in named:
                status = f"current (`h3_config.{named[path.name]}`)"
            else:
                status = "current, unwired"
            if ours:
                adaln = (meta.get("h3_pdd_adaln_form")
                         or ("baked" if any(k.startswith("h3_pdd.adaln_baked.") for k in keys)
                             else "2688" if any(k.startswith("h3_pdd.adaln.") for k in keys)
                             else "-"))
                loads_on = (meta.get("h3_pdd_loads_on")
                            or (f"the pruned build only ({meta['h3_pdd_pruned_base']})"
                                if meta.get("h3_pdd_pruned_base") else "either build"))
            rows.append({
                "file": (f"pdd_archive/{path.name}" if folder == ARCHIVE else path.name),
                "status": status,
                "converter": meta.get("h3_pdd_converter_version", "-") if ours else "-",
                "made_on": _made_on(path, meta),
                "commit": meta.get("h3_pdd_converter_commit", "-") if ours else "-",
                "backbone": (meta.get("h3_pdd_backbone", "full") if ours else "-"),
                "adaln": adaln,
                "probe_of": (meta.get("h3_pdd_backbone_probe_of", "none") if ours else "-"),
                "loads_on": loads_on,
                "tensors": len(keys),
            })
    return rows


def render_region(rows: list[dict]) -> str:
    lines = [
        BEGIN,
        "",
        "| file | status | converter | made on | commit | backbone | adaln form | probe cut from | loads on | tensors |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| `{r['file']}` | {r['status']} | {r['converter']} | "
                     f"{r['made_on']} | {r['commit']} | {r['backbone']} | "
                     f"{r['adaln']} | {r['probe_of']} | {r['loads_on']} | "
                     f"{r['tensors']} |")
    lines += ["", END]
    return "\n".join(lines)


def splice(page: str, region: str) -> str:
    i = page.find(BEGIN)
    j = page.find(END)
    if i < 0 or j < 0 or j < i:
        raise SystemExit(
            f"{OUT.relative_to(REPO)} has no generated region: it needs the "
            f"lines `{BEGIN}` and `{END}`, in that order, around the table. "
            f"The hand-written sections live outside them and this script "
            f"never touches those.")
    return page[:i] + region + page[j + len(END):]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the page's generated region differs from "
                         "a fresh render")
    args = ap.parse_args(argv)
    if not LORAS.exists():
        print(f"DID NOT RUN: {LORAS.relative_to(HERE.parents[2])} is not on "
              f"this box. Nothing was inventoried, which is not a pass.")
        return 2
    rows = inventory()
    if not rows:
        print("DID NOT RUN: no PDD files found; nothing was inventoried.")
        return 2
    if not OUT.exists():
        raise SystemExit(f"{OUT.relative_to(REPO)} does not exist; write the "
                         f"hand sections and the two marker lines first.")
    page = OUT.read_text()
    updated = splice(page, render_region(rows))
    if args.check:
        if page != updated:
            print(f"STALE: the generated region of {OUT.relative_to(REPO)} "
                  f"differs from the files on disk; regenerate it.")
            return 1
        print(f"  ok    {OUT.relative_to(REPO)} inventory matches {len(rows)} "
              f"file(s) on disk")
        return 0
    OUT.write_text(updated)
    print(f"wrote the inventory region of {OUT.relative_to(REPO)} "
          f"({len(rows)} file(s))")
    for r in rows:
        print(f"  {r['file']:70s} {r['converter']:>2s}  {r['made_on']:19s} {r['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
