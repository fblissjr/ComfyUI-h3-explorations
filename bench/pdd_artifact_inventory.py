#!/usr/bin/env python3
"""The PDD artifact record and the generated regions of `docs/pdd_artifacts.md`,
derived from the files, from git, and from the node's source -- never typed.

`docs/pdd_artifacts.md` is the master page for every PDD weight file on this
box. VISION.md's rule for it is the rule for everything here: prose carries
pointers and records carry numbers, and anything a file or a commit already
knows is generated, never retyped. So this script owns three things:

1. **The dated fingerprint record**, `bench/results/<date>_pdd_artifact_fingerprints.json`
   (`--record`). For every PDD file: its metadata, a content hash by the recipe
   `bench/results/2026-08-28_pdd_conversion_reproducibility.json` established
   (sha256 over sorted metadata pairs, then every tensor's bytes in sorted key
   order; independent of header layout, so two files with the same content
   hash ARE the same file whatever safetensors did to the header), and per
   tensor GROUP (block index normalised to `N`) the dtype, shape, count and a
   sha256 over the group's bytes. Sidecar tensors (`h3_pdd.*`) are recorded
   per tensor as well. This is the point-in-time reference: what each file WAS
   on that date, checkable later without trusting anyone's sentence.

2. **Three generated regions of the page**, each between a `BEGIN GENERATED`
   / `END GENERATED` marker pair:
   - `inventory`: one row per file, from its metadata and the record;
   - `sidecar-tensors`: one row per sidecar tensor group across every file --
     dtype, shape, which files carry it, the converter commit and date that
     INTRODUCED it (`git log -S` over `bench/convert_pdd_lora.py`, so the
     "since" column is read from history, not remembered), and the functions
     in `pdd_lora.py` that read it (the enclosing `def` of every source line
     naming the key, from the AST);
   - `version-diff`: for each partition, the archived version-1 file against
     the current `_comfy` file, and `_comfy` against `_adaln2688`: groups
     added, removed, changed (group hash differs) and unchanged, so "what
     changed between versions" is computed rather than narrated.

3. **Validation of the hand-written parts against the derived ones.** The WHY
   of each tensor is a decision, so it stays prose -- but every sidecar family
   present in the record must have a `### <family>` heading in the page's
   provenance section, every commit hash the page names (a backticked 7-hex
   token) must resolve in this repo, and the files on disk must match the
   latest record. `--check` runs all three and the region comparison; a
   sidecar tensor nobody documented, a commit that does not exist, or a file
   regenerated without a new record all go red.

Statuses in the inventory are derived: `current` (named by an `h3_config`
`*_LORA` constant), `current, unwired`, `archived` (under `pdd_archive/`),
`reference` (not this repo's conversion: the alibaba-pai source or Kijai's).

Needs the loras folder on this box; exits 2, not 0, without it.

    uv run --active --no-sync python bench/pdd_artifact_inventory.py --record   # new dated record, then regenerate
    uv run --active --no-sync python bench/pdd_artifact_inventory.py            # regenerate from the latest record
    uv run --active --no-sync python bench/pdd_artifact_inventory.py --check
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "workflows"))
import h3_config  # noqa: E402

LORAS = HERE.parents[2] / "models" / "loras" / "h3"
ARCHIVE = LORAS / "pdd_archive"
OUT = REPO / "docs" / "pdd_artifacts.md"
RESULTS = HERE / "results"
CONVERTER = REPO / "bench" / "convert_pdd_lora.py"
NODE = REPO / "pdd_lora.py"
PATTERN = re.compile(r"pdd|Acc-8Step", re.IGNORECASE)
RECORD_GLOB = "*_pdd_artifact_fingerprints.json"
RECIPE = ("sha256 over sorted metadata pairs then every tensor's bytes in "
          "sorted key order; independent of header layout")

#: Sidecar FAMILIES: the prefix a tensor group belongs to, the token `git log
#: -S` searches the converter's history for (the literal as the converter
#: source spells it), and the heading the page's provenance section must carry.
#: A group matching none of these is reported as undocumented and goes red:
#: adding a sidecar tensor means adding its family here AND its WHY to the page.
FAMILIES: list[tuple[str, str, str]] = [
    ("h3_pdd.bank.", "h3_pdd.bank.", "h3_pdd.bank.*"),
    ("h3_pdd.base_video_out", "h3_pdd.base_video_out", "h3_pdd.base_video_out"),
    ("h3_pdd.adaln_baked.", "h3_pdd.adaln_baked", "h3_pdd.adaln_baked.*"),
    ("h3_pdd.adaln_table", "h3_pdd.adaln_table", "h3_pdd.adaln_table"),
    ("h3_pdd.adaln.blocks.N.alpha", "h3_pdd.adaln.blocks.{i}.alpha", "h3_pdd.adaln.blocks.N.alpha"),
    ("h3_pdd.adaln.", "h3_pdd.adaln.blocks", "h3_pdd.adaln.blocks.N.lora_A / lora_B"),
    ("h3_pdd.silu_temb_grid", "h3_pdd.silu_temb_grid", "h3_pdd.silu_temb_grid"),
    ("h3_pdd.backbone_probe_base", "h3_pdd.backbone_probe_base", "h3_pdd.backbone_probe_base / _base_scale"),
    ("h3_pdd.backbone_probe_scale", "h3_pdd.backbone_probe_scale", "h3_pdd.backbone_probe_scale"),
    ("h3_pdd.backbone_probe", 'h3_pdd.backbone_probe"', "h3_pdd.backbone_probe"),
]


def family_of(group: str) -> tuple[str, str, str] | None:
    for fam in FAMILIES:
        if group.startswith(fam[0]):
            return fam
    return None


def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def group_key(key: str) -> str:
    return re.sub(r"\.blocks\.\d+\.", ".blocks.N.", key)


# --- fingerprints --------------------------------------------------------------

def fingerprint(path: Path) -> dict:
    """Metadata, content hash, per-group and per-sidecar-tensor hashes."""
    from safetensors import safe_open
    content = hashlib.sha256()
    groups: dict[str, dict] = {}
    sidecar: dict[str, dict] = {}
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
        for k in sorted(meta):
            content.update(f"{k}={meta[k]}".encode())
        keys = sorted(f.keys())
        for k in keys:
            t = f.get_tensor(k)
            data = t.contiguous().numpy().tobytes() if t.dtype != __import__("torch").bfloat16 \
                else t.contiguous().view(__import__("torch").int16).numpy().tobytes()
            content.update(data)
            g = group_key(k)
            entry = groups.setdefault(g, {"dtype": str(t.dtype).replace("torch.", ""),
                                          "shape": list(t.shape), "count": 0,
                                          "_h": hashlib.sha256()})
            entry["count"] += 1
            entry["_h"].update(data)
            if k.startswith("h3_pdd."):
                sidecar[k] = {"dtype": entry["dtype"], "shape": list(t.shape),
                              "sha256": hashlib.sha256(data).hexdigest()}
    for g in groups.values():
        g["sha256"] = g.pop("_h").hexdigest()
    return {"metadata": meta, "content_sha256": content.hexdigest(),
            "tensors": len(keys), "groups": groups, "sidecar_tensors": sidecar}


def _named_in_config() -> dict[str, str]:
    out = {}
    for name in dir(h3_config):
        if name.endswith("_LORA"):
            value = getattr(h3_config, name)
            if isinstance(value, str) and PATTERN.search(value):
                out[Path(value).name] = name
    return out


def _status(path: Path, folder: Path, meta: dict, keys_groups: dict, named: dict) -> str:
    ours = "h3_pdd_converter_version" in meta
    if folder == ARCHIVE:
        return "archived"
    if not ours:
        return ("reference (alibaba-pai source)"
                if any(g.startswith("transformer_blocks.") for g in keys_groups)
                else "reference (Kijai's conversion)")
    if path.name in named:
        return f"current (`h3_config.{named[path.name]}`)"
    return "current, unwired"


def _made_on(path: Path, meta: dict) -> str:
    stamped = meta.get("h3_pdd_converted_on")
    if stamped:
        return stamped
    ts = path.resolve().stat().st_mtime
    return _dt.date.fromtimestamp(ts).isoformat() + " (mtime)"


def build_record() -> dict:
    named = _named_in_config()
    files = []
    for folder in (LORAS, ARCHIVE):
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.safetensors")):
            if not PATTERN.search(path.name):
                continue
            fp = fingerprint(path)
            meta = fp["metadata"]
            ours = "h3_pdd_converter_version" in meta
            groups = fp["groups"]
            files.append({
                "file": f"pdd_archive/{path.name}" if folder == ARCHIVE else path.name,
                "status": _status(path, folder, meta, groups, named),
                "ours": ours,
                "converter": meta.get("h3_pdd_converter_version", "-") if ours else "-",
                "made_on": _made_on(path, meta),
                "commit": meta.get("h3_pdd_converter_commit", "-") if ours else "-",
                "backbone": meta.get("h3_pdd_backbone", "full") if ours else "-",
                "adaln": ((meta.get("h3_pdd_adaln_form")
                           or ("baked" if any(g.startswith("h3_pdd.adaln_baked.") for g in groups)
                               else "2688" if any(g.startswith("h3_pdd.adaln.") for g in groups)
                               else "-")) if ours else "-"),
                "probe_of": meta.get("h3_pdd_backbone_probe_of", "none") if ours else "-",
                "loads_on": ((meta.get("h3_pdd_loads_on")
                              or (f"the pruned build only ({meta['h3_pdd_pruned_base']})"
                                  if meta.get("h3_pdd_pruned_base") else "either build"))
                             if ours else "-"),
                **fp,
            })
    return {
        "recorded": _dt.date.today().isoformat(),
        "produced_by": "bench/pdd_artifact_inventory.py --record",
        "what": ("point-in-time fingerprint of every PDD weight file on this box: "
                 "metadata, content hash, per-group and per-sidecar-tensor hashes"),
        "content_hash_recipe": RECIPE,
        "repo_commit": _git("rev-parse", "--short", "HEAD"),
        "files": files,
    }


def latest_record() -> tuple[Path, dict] | tuple[None, None]:
    paths = sorted(RESULTS.glob(RECORD_GLOB))
    if not paths:
        return None, None
    return paths[-1], json.loads(paths[-1].read_text())


# --- derived facts from git and the node --------------------------------------

def introduced(token: str) -> str:
    """`commit date` of the first converter commit adding the token, from git."""
    out = _git("log", "-S", token, "--reverse", "--format=%h %ad", "--date=short",
               "--", str(CONVERTER.relative_to(REPO)))
    return out.splitlines()[0] if out else "not found in the converter's history"


def node_readers(prefix: str) -> list[str]:
    """Functions in pdd_lora.py whose body names the key prefix."""
    src = NODE.read_text()
    tree = ast.parse(src)
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.append((node.lineno, node.end_lineno, node.name))
    needle = prefix.replace(".blocks.N.", ".blocks.")
    found = []
    for i, line in enumerate(src.splitlines(), 1):
        if needle in line or prefix in line:
            inner = [s for s in spans if s[0] <= i <= s[1]]
            if inner:
                found.append(min(inner, key=lambda s: s[1] - s[0])[2])
    return sorted(set(found))


# --- regions ----------------------------------------------------------------------

def marker(name: str) -> tuple[str, str]:
    return (f"<!-- BEGIN GENERATED: {name} (bench/pdd_artifact_inventory.py) -->",
            f"<!-- END GENERATED: {name} -->")


def render_inventory(rec: dict) -> str:
    lines = [
        "| file | status | converter | made on | commit | backbone | adaln form | probe cut from | loads on | tensors | content sha256 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rec["files"]:
        lines.append(f"| `{r['file']}` | {r['status']} | {r['converter']} | {r['made_on']} | "
                     f"{r['commit']} | {r['backbone']} | {r['adaln']} | {r['probe_of']} | "
                     f"{r['loads_on']} | {r['tensors']} | `{r['content_sha256'][:12]}` |")
    lines.append("")
    lines.append(f"Record: `bench/results/{rec['_path']}` (recorded {rec['recorded']} at "
                 f"repo commit `{rec['repo_commit']}`; content hash recipe: {RECIPE}).")
    return "\n".join(lines)


def _short(name: str) -> str:
    n = name.replace("pdd_archive/", "archive:").replace("minimax_h3_", "").replace(".safetensors", "")
    return n.replace("_pdd_8step_", " ").replace("_comfy", "")


def render_sidecar(rec: dict) -> str:
    ours = [r for r in rec["files"] if r["ours"]]
    groups: dict[str, dict] = {}
    for r in ours:
        for g, info in r["groups"].items():
            if not g.startswith("h3_pdd."):
                continue
            e = groups.setdefault(g, {"dtype": info["dtype"], "shape": info["shape"],
                                      "count": info["count"], "files": []})
            e["files"].append(_short(r["file"]))
    lines = [
        "| sidecar tensor group | dtype | shape | per file | carried by | introduced in the converter (git) | read in `pdd_lora.py` by |",
        "|---|---|---|---|---|---|---|",
    ]
    undocumented = []
    for g in sorted(groups):
        e = groups[g]
        fam = family_of(g)
        if fam is None:
            undocumented.append(g)
            intro, readers = "UNMAPPED family", []
        else:
            intro = introduced(fam[1])
            readers = node_readers(fam[0])
        lines.append(f"| `{g}` | {e['dtype']} | `{e['shape']}` | x{e['count']} | "
                     f"{', '.join(e['files'])} | `{intro.split()[0]}` {intro.split()[-1] if ' ' in intro else ''} | "
                     f"{', '.join(f'`{f}`' for f in readers) or 'nothing (see its family note)'} |")
    rec["_undocumented"] = undocumented
    return "\n".join(lines)


def render_diff(rec: dict) -> str:
    by = {r["file"]: r for r in rec["files"]}
    pairs = []
    for part in ("fl2va", "ref2va"):
        cur = f"minimax_h3_{part}_pdd_8step_comfy.safetensors"
        v1 = f"pdd_archive/minimax_h3_{part}_pdd_8step_comfy_v1_2026-08-28.safetensors"
        alt = f"minimax_h3_{part}_pdd_8step_adaln2688_comfy.safetensors"
        if v1 in by and cur in by:
            pairs.append((f"{part}: archived v1 -> current `_comfy`", by[v1], by[cur]))
        if cur in by and alt in by:
            pairs.append((f"{part}: `_comfy` -> `_adaln2688` (the two forms)", by[cur], by[alt]))
    lines = []
    for title, a, b in pairs:
        ga, gb = a["groups"], b["groups"]
        added = sorted(set(gb) - set(ga))
        removed = sorted(set(ga) - set(gb))
        changed = sorted(g for g in set(ga) & set(gb) if ga[g]["sha256"] != gb[g]["sha256"])
        same = len(set(ga) & set(gb)) - len(changed)
        lines.append(f"**{title}** (`{a['file']}` -> `{b['file']}`): "
                     f"{same} tensor group(s) byte-identical, {len(changed)} changed, "
                     f"{len(added)} added, {len(removed)} removed.")
        for label, items in (("added", added), ("removed", removed), ("changed", changed)):
            if items:
                lines.append(f"- {label}: " + ", ".join(f"`{g}`" for g in items))
        ma, mb = a["metadata"], b["metadata"]
        mk = sorted(k for k in set(ma) | set(mb) if ma.get(k) != mb.get(k))
        if mk:
            lines.append("- metadata keys that differ: " + ", ".join(f"`{k}`" for k in mk))
        lines.append("")
    return "\n".join(lines).rstrip()


def splice(page: str, name: str, body: str) -> str:
    b, e = marker(name)
    i, j = page.find(b), page.find(e)
    if i < 0 or j < 0 or j < i:
        raise SystemExit(f"{OUT.relative_to(REPO)} lacks the marker pair for "
                         f"region {name!r}: `{b}` ... `{e}`.")
    return page[:i] + b + "\n\n" + body + "\n\n" + e + page[j + len(e):]


def regenerate(page: str, rec: dict) -> str:
    page = splice(page, "inventory", render_inventory(rec))
    page = splice(page, "sidecar-tensors", render_sidecar(rec))
    page = splice(page, "version-diff", render_diff(rec))
    return page


# --- validation ------------------------------------------------------------------

def validate(page: str, rec: dict) -> list[str]:
    problems = []
    for g in rec.get("_undocumented", []):
        problems.append(f"sidecar tensor group `{g}` matches no FAMILY in "
                        f"bench/pdd_artifact_inventory.py; add it and its WHY")
    present = {fam[2] for r in rec["files"] if r["ours"] for g in r["groups"]
               if g.startswith("h3_pdd.") for fam in [family_of(g)] if fam}
    for heading in sorted(present):
        if f"### `{heading}`" not in page:
            problems.append(f"the page has no provenance heading `### `{heading}``"
                            f" for a sidecar family the files carry")
    for h in sorted(set(re.findall(r"`([0-9a-f]{7})`", page))):
        try:
            _git("cat-file", "-e", f"{h}^{{commit}}")
        except subprocess.CalledProcessError:
            problems.append(f"the page names commit `{h}`, which does not exist in this repo")
    return problems


def check_disk_matches(rec: dict) -> list[str]:
    fresh = build_record()
    old = {r["file"]: r["content_sha256"] for r in rec["files"]}
    new = {r["file"]: r["content_sha256"] for r in fresh["files"]}
    problems = []
    for f in sorted(set(old) | set(new)):
        if f not in new:
            problems.append(f"`{f}` is in the record but not on disk")
        elif f not in old:
            problems.append(f"`{f}` is on disk but not in the record")
        elif old[f] != new[f]:
            problems.append(f"`{f}` changed since the record")
    if problems:
        problems.append("run `bench/pdd_artifact_inventory.py --record` to write a "
                        "new dated record, then regenerate")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true",
                    help="fingerprint every file into a new dated record, then regenerate")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the regions, the provenance headings, the named "
                         "commits or the files on disk disagree with the record")
    args = ap.parse_args(argv)
    if not LORAS.exists():
        print(f"DID NOT RUN: {LORAS.relative_to(HERE.parents[2])} is not on this "
              f"box. Nothing was inventoried, which is not a pass.")
        return 2
    if not OUT.exists():
        raise SystemExit(f"{OUT.relative_to(REPO)} does not exist; write the hand "
                         f"sections and the marker pairs first.")

    if args.record:
        rec = build_record()
        path = RESULTS / f"{rec['recorded']}_pdd_artifact_fingerprints.json"
        path.write_text(json.dumps(rec, indent=2) + "\n")
        print(f"wrote {path.relative_to(REPO)} ({len(rec['files'])} file(s))")
    path, rec = latest_record()
    if rec is None:
        raise SystemExit("no fingerprint record under bench/results/; run --record first")
    assert path is not None
    rec["_path"] = path.name

    page = OUT.read_text()
    updated = regenerate(page, rec)
    problems = validate(updated, rec)
    if args.check:
        if page != updated:
            problems.insert(0, f"a generated region of {OUT.relative_to(REPO)} differs "
                               f"from the record {path.name}; regenerate")
        problems += check_disk_matches(rec)
        if problems:
            print("RED:")
            for p in problems:
                print(f"  - {p}")
            return 1
        print(f"  ok    {OUT.relative_to(REPO)} regions, provenance headings and named "
              f"commits agree with {path.name}, and the {len(rec['files'])} file(s) "
              f"on disk match it")
        return 0
    OUT.write_text(updated)
    print(f"regenerated the three regions of {OUT.relative_to(REPO)} from {path.name}")
    for p in problems:
        print(f"  WARN  {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
